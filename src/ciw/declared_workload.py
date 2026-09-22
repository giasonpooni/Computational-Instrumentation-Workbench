"""Bounded native schematic assessment and integer numerical execution.

These are workbench objects, not observations or state estimates. Repository
and executable paths are trusted host bindings and never read from artifacts.
Replay checks reproducibility; it does not authenticate submitted certificates.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
import struct
import tempfile
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .exchange import _identity, _read
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _now, _keys

MAX_BYTES = 4 * 1024 * 1024
SOURCE_LIMIT = 262144
KINDS = frozenset({"schematic-assessment", "numerical-heat"})
PINS = {
    "schematic-assessment": {"role": "sra", "revision": "a6e79585950bb6860e5edce5ebd2cce39ea481f2",
                             "source_root": "src", "module": "schematics.eligibility"},
    "numerical-heat": {"role": "scr", "revision": "a59aba283b0304faeeb3e5d305087e7709e171ca",
                       "source_root": ".", "module": "execution.engine"},
}
RESULT_SCHEMA = "ciw.declared-workload-result.v1"
VERIFY_SCHEMA = "ciw.declared-workload-verification.v1"
GRAPH_SCHEMA = "NsObservabilitySchematic@0.1"
HEAT_DESCRIPTOR = (
    b"scout.native.heat-diffusion-kernel.v1\n"
    b"input: [steps u32 LE][n u32 LE][n x i64 LE]; 3<=n<=4096; steps<=100000; |u|<=2^40\n"
    b"per step, Jacobi, Dirichlet ends fixed: u'_i = u_i + (u_{i-1} - 2u_i + u_{i+1})/4\n"
    b"(integer division, truncation toward zero; alpha=1/4 within stability bound 1/2)\n"
    b"output: n x i64 LE final values; exit 0\n"
    b"faults: 2=malformed, 3=n<3, 4=n/steps bound, 5=value bound"
)
HEAT_POLICY = {"unit": "1", "arithmetic": "signed_integer_truncation_toward_zero",
               "boundary": "fixed_endpoints", "alpha": "1/4", "covariance_status": "not_applicable"}
ASSESSMENT_POLICY = {"companion_execution": "not_performed", "certificate_authentication": "not_performed",
                     "observability": "not_evaluated", "retrieval": "declared_two_hop_neighborhood"}
AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed", "physical_truth": "not_established"}
_EDGES = {"input": ("variable", "function"), "output": ("function", "variable"),
          "measures": ("measurement", "variable"), "actuates": ("variable", "function"),
          "binds": ("evidence", "measurement"), "linearizes": ("certificate", "function"),
          "certifies": ("certificate", "function"), "observes": ("observer", "function"),
          "senses": ("observer", "measurement"), "composed_in": ("function", "function")}


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("Require a bounded nonempty identifier")


def _graph(graph):
    _keys(graph, {"schema", "meta", "nodes", "edges"})
    if graph["schema"] != GRAPH_SCHEMA or not isinstance(graph["meta"], dict) or graph["meta"].get("schema") != GRAPH_SCHEMA:
        raise ValueError("Require explicit matching graph and metadata schemas")
    if not isinstance(graph["nodes"], list) or not 1 <= len(graph["nodes"]) <= 256:
        raise ValueError("Schematic node budget is 1..256")
    if not isinstance(graph["edges"], list) or len(graph["edges"]) > 1024:
        raise ValueError("Schematic edge budget is 1024")
    nodes = {}
    for node in graph["nodes"]:
        _keys(node, {"id", "kind", "attrs"})
        _text(node["id"])
        if (node["id"] in nodes or node["kind"] not in
                ("variable", "function", "measurement", "prior", "constraint", "certificate", "observer", "evidence") or
                not isinstance(node["attrs"], dict)):
            raise ValueError("Invalid or duplicate schematic node")
        if node["kind"] == "function" and node["attrs"].get("class", "unknown") not in ("linear", "lpv", "nonlinear", "unknown"):
            raise ValueError("Undeclared function class")
        if node["kind"] == "certificate" and node["attrs"].get("owner") not in ("jspt", "plsr", "rci", "cse"):
            raise ValueError("Unknown certificate owner")
        nodes[node["id"]] = node
    for edge in graph["edges"]:
        _keys(edge, {"kind", "src", "dst", "attrs"})
        if (not isinstance(edge["kind"], str) or edge["kind"] not in _EDGES or not isinstance(edge["attrs"], dict) or
                not isinstance(edge["src"], str) or not isinstance(edge["dst"], str) or
                edge["src"] not in nodes or edge["dst"] not in nodes or
                (nodes[edge["src"]]["kind"], nodes[edge["dst"]]["kind"]) != _EDGES[edge["kind"]]):
            raise ValueError("Invalid declared typed edge")
    return nodes


def _source(kind, raw):
    if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
        raise ValueError("Declared workload source exceeds byte budget")
    source = _json(raw)
    canonical(source)
    _keys(source, {"schema", "experiment_id", "configuration"} |
          ({"schematic", "queries"} if kind == "schematic-assessment" else {"initial_values", "steps"}))
    if source["schema"] != "ciw." + kind + "-source.v1":
        raise ValueError("Unsupported declared workload source")
    _text(source["experiment_id"])
    policy = ASSESSMENT_POLICY if kind == "schematic-assessment" else HEAT_POLICY
    if canonical(source["configuration"]) != canonical(policy):
        raise ValueError("Require the explicit bounded workload policy")
    if kind == "schematic-assessment":
        nodes = _graph(source["schematic"])
        queries = source["queries"]
        if not isinstance(queries, list) or len(queries) > 16 or any(not isinstance(q, str) or q not in nodes for q in queries) or len(set(queries)) != len(queries):
            raise ValueError("Queries must name at most 16 distinct declared nodes")
    else:
        values = source["initial_values"]
        if (not isinstance(values, list) or not 3 <= len(values) <= 256 or
                any(type(v) is not int or abs(v) > 2**40 for v in values) or
                type(source["steps"]) is not int or not 0 <= source["steps"] <= 1024):
            raise ValueError("Heat workload requires 3..256 integer cells, |u|<=2^40, and 0..1024 steps")
    return source


_BOOTSTRAP = r'''
import dataclasses, json, struct, sys
from pathlib import Path
role, root, engine = sys.argv[1:4]
sys.path.insert(0, root)
source = json.loads(sys.stdin.buffer.read())
if role == 'sra':
    from schematics.io import from_dict, to_dict
    from schematics.annotate import invalidate_stale_results, observer_next_step
    from schematics.eligibility import decide
    from schematics.retrieve import blanket
    graph = from_dict(source['schematic'])
    invalidate_stale_results(graph)
    data = {'schematic': to_dict(graph), 'decisions': [dataclasses.asdict(d) for d in decide(graph)],
            'neighborhoods': {q: blanket(graph, q) for q in source['queries']},
            'next_step': observer_next_step(graph)}
else:
    from execution.specification import ExecutionSpecification, HEAT_DIFFUSION_DESCRIPTOR, encode_heat_input
    from execution.engine import run_specification
    spec = ExecutionSpecification(HEAT_DIFFUSION_DESCRIPTOR, b'', encode_heat_input(source['steps'], source['initial_values']))
    result = run_specification(spec, cli_path=Path(engine))
    if result.status != 'completed' or result.exit_code != 0 or result.output is None:
        raise ValueError('Declared heat execution did not complete')
    data = dataclasses.asdict(result)
    data['specification'] = {k: v.hex() for k, v in data['specification'].items()}
    data['output'] = result.output.hex()
    data['values'] = list(struct.unpack('<' + 'q' * len(source['initial_values']), result.output))
print(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
'''


def _commit(tag, fields):
    tag = ("scout.execution." + tag + ".v1").encode()
    raw = struct.pack("<Q", len(tag)) + tag + struct.pack("<Q", len(fields))
    return sha256(raw + b"".join(struct.pack("<Q", len(f)) + f for f in fields)).hexdigest()


def _check_data(kind, source, data):
    if kind == "schematic-assessment":
        _keys(data, {"schematic", "decisions", "neighborhoods", "next_step"})
        nodes = _graph(data["schematic"])
        original = source["schematic"]
        if canonical(data["schematic"]["edges"]) != canonical(original["edges"]) or canonical(data["schematic"]["meta"]) != canonical(original["meta"]) or list(nodes) != [n["id"] for n in original["nodes"]]:
            raise ValueError("Assessment must retain the declared graph topology")
        # Native stale marking retains every original numeric and binding field.
        for old in original["nodes"]:
            new = nodes[old["id"]]
            if canonical(new) != canonical(old):
                attrs = dict(old["attrs"])
                attrs.update(historical_result="SAMPLED", result="NOT_ELIGIBLE", currentness="stale",
                             stale_reason="missing, stale, ambiguous or superseded Jacobian/covariance input binding")
                if old["kind"] != "certificate" or old["attrs"].get("result") != "SAMPLED" or canonical(new) != canonical({**old, "attrs": attrs}):
                    raise ValueError("Assessment altered retained schematic evidence")
        if not isinstance(data["decisions"], list) or len(data["decisions"]) > 4096:
            raise ValueError("Invalid native eligibility decisions")
        for decision in data["decisions"]:
            _keys(decision, {"tool", "owner", "status", "node_id", "reason"})
            for value in decision.values():
                _text(value)
            if decision["node_id"] not in nodes or decision["status"] not in ("ELIGIBLE", "NOT_ELIGIBLE"):
                raise ValueError("Eligibility is not solver execution or observability")
        neighborhoods = {}
        for query in source["queries"]:
            adjacent = {query}
            for _ in range(2):
                adjacent |= {e["dst"] for e in original["edges"] if e["src"] in adjacent} | {e["src"] for e in original["edges"] if e["dst"] in adjacent}
            neighborhoods[query] = sorted(adjacent)
        if data["neighborhoods"] != neighborhoods:
            raise ValueError("Retrieval differs from declared two-hop topology")
        _text(data["next_step"])
    else:
        _keys(data, {"specification", "specification_identity", "program_identity", "input_identity", "engine_occurrence",
                     "status", "exit_code", "output", "output_identity", "computation_identity", "detail", "values"})
        inputs = struct.pack("<II", source["steps"], len(source["initial_values"])) + b"".join(struct.pack("<q", v) for v in source["initial_values"])
        spec = {"program": HEAT_DESCRIPTOR.hex(), "configuration": "", "input_payload": inputs.hex()}
        if data["specification"] != spec or data["status"] != "completed" or type(data["exit_code"]) is not int or data["exit_code"] != 0 or type(data["engine_occurrence"]) is not int or data["engine_occurrence"] != 0 or data["detail"] is not None:
            raise ValueError("Numerical result differs from the declared native specification")
        if not isinstance(data["values"], list) or len(data["values"]) != len(source["initial_values"]) or any(type(v) is not int or abs(v) > 2**40 for v in data["values"]):
            raise ValueError("Invalid native integer field")
        output = b"".join(struct.pack("<q", v) for v in data["values"])
        if data["output"] != output.hex():
            raise ValueError("Native output bytes differ from displayed field")
        expected = {"program_identity": _commit("program", [HEAT_DESCRIPTOR]), "input_identity": _commit("input", [inputs]),
                    "output_identity": _commit("output", [output]), "specification_identity": _commit("specification", [HEAT_DESCRIPTOR, b"", inputs])}
        expected["computation_identity"] = _commit("computation", [bytes.fromhex(expected[k]) for k in ("program_identity", "input_identity", "output_identity")] + [struct.pack("<I", 0)])
        if any(data[k] != v for k, v in expected.items()):
            raise ValueError("Native execution commitment mismatch")


def _verification(bundle, reproduced):
    if canonical(bundle["steps"][0]["numerical_result"]) != canonical(reproduced["numerical_result"]):
        raise ValueError("Native workload replay mismatch")
    value = {"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
             "independent": False, "method": "same_runtime_fresh_occurrence_reproduction",
             "runtime_digest": digest(bundle["runtimes"]), "reproduction": reproduced,
             "authority": AUTHORITY}
    value["verification_id"] = byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(value))
    return value


class DeclaredWorkflow:
    MAX_BYTES = MAX_BYTES

    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError("Unsupported declared workload")
        self.kind, self.pin = kind, PINS[kind]
        self.role = self.pin["role"]
        self.ROLES = {self.role} | ({"engine"} if self.role == "scr" else set())
        self.SOURCE_SCHEMA = "ciw." + kind + "-source.v1"
        self.schema = "ciw." + kind + "-session.v1"
        self.operation = "ciw." + kind + ".v1"

    def _source(self, raw):
        return _source(self.kind, raw)

    def _adapters(self, repositories, expected=None):
        if set(repositories) != self.ROLES:
            raise ValueError("Bind the declared native provider and, for SCR, its engine")
        retained = expected[self.role] if expected else {}
        adapter = PinnedSubprocessAdapter(repositories[self.role], self.pin["revision"], self.pin["module"], source_root=self.pin["source_root"],
            expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"), expected_dependencies=retained.get("dependencies"))
        runtime = adapter.runtime_identity()
        engine = None
        if self.role == "scr":
            engine = _read(Path(repositories["engine"]), 32 * 1024 * 1024)
            runtime["engine"] = {"sha256": byte_digest(engine), "byte_count": len(engine),
                                 "source_binding": "operator_asserted_not_attested"}
        if retained and self._runtime_projection(runtime) != self._runtime_projection(retained):
            raise ValueError("Native workload runtime differs from the retained execution")
        return adapter, runtime, engine

    @staticmethod
    def _runtime_projection(runtime):
        return {k: v for k, v in runtime.items() if k not in {"repository_root", "python_executable"}}

    def _step(self, source, evidence_id, bound):
        adapter, runtime, engine = bound
        if self._runtime_projection(adapter.runtime_identity()) != self._runtime_projection({k: v for k, v in runtime.items() if k != "engine"}):
            raise ValueError("Native provider identity changed")
        # Execute a private snapshot of the host-bound engine, eliminating a
        # pathname swap between digesting and invoking the native executable.
        with tempfile.TemporaryDirectory(prefix="ciw-native-workload-") as root:
            executable = Path(root) / "execution-cli"
            if engine is not None:
                executable.write_bytes(engine)
                executable.chmod(0o700)
            code, raw = adapter._run(_BOOTSTRAP, [self.role, str(adapter.source_root), str(executable)], canonical(source))
            if engine is not None and _read(executable, 32 * 1024 * 1024) != engine:
                raise ValueError("Native executable changed during execution")
        adapter.runtime_identity()
        if code:
            raise AdapterRefusal("DECLARED_WORKLOAD_REFUSED", "Pinned " + self.role + " refused the declared workload")
        data = _json(raw)
        _check_data(self.kind, source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": data}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
                "input_refs": [evidence_id], "request": source, "request_sha256": digest(source),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id] or canonical(step["request"]) != canonical(source):
            raise ValueError("Native request/operation/evidence binding mismatch")
        if not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]):
            raise ValueError("Invalid native execution occurrence")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(self.kind, source, result["data"])
        if (result["schema"] != RESULT_SCHEMA or result["authority"] != AUTHORITY or result["operation_id"] != self.operation or
                result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence_id] or
                result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"}) or result["result_id"] != step["result_id"] or
                canonical(step["numerical_result"]) != canonical({"operation_id": self.operation, "data": result["data"]})):
            raise ValueError("Native result binding or authority mismatch")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("Workload step content mismatch")

    def _check_verification(self, bundle, verification, source, evidence):
        _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "reproduction", "authority", "verification_id"})
        self._validate_step(verification["reproduction"], source, evidence)
        old, new = bundle["steps"][0], verification["reproduction"]
        if old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or verification != _verification(bundle, new):
            raise ValueError("Verification must bind fresh native reproduction and limited authority")
        _identity(verification, "verification_id")

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != _bundle_digest(bundle):
                raise ValueError("Native workload bundle binding mismatch")
            if not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Invalid workload session occurrence")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            if evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()} or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]} or canonical(bundle["configuration"]) != canonical(source["configuration"]):
                raise ValueError("Workload source/configuration binding mismatch")
            if set(bundle["runtimes"]) != {self.role}:
                raise ValueError("Unexpected workload runtime")
            runtime = bundle["runtimes"][self.role]
            _keys(runtime, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root", "python_executable", "python_sha256", "python_version", "dependencies"} | ({"engine"} if self.role == "scr" else set()))
            if runtime["schema"] != "ciw.subprocess-runtime.v1" or any(runtime[k] != self.pin[k] for k in ("revision", "module", "source_root")) or not re.fullmatch("[a-f0-9]{40}", runtime["source_tree"]) or not re.fullmatch("[a-f0-9]{64}", runtime["python_sha256"]):
                raise ValueError("Unapproved native runtime pin")
            for k in ("adapter_version", "repository_root", "python_executable", "python_version"):
                _text(runtime[k])
            if not isinstance(runtime["dependencies"], dict):
                raise ValueError("Invalid dependency identities")
            if self.role == "scr":
                engine = runtime["engine"]
                _keys(engine, {"sha256", "byte_count", "source_binding"})
                if not re.fullmatch("sha256:[a-f0-9]{64}", engine["sha256"]) or type(engine["byte_count"]) is not int or not 1 <= engine["byte_count"] <= 32 * 1024 * 1024 or engine["source_binding"] != "operator_asserted_not_attested":
                    raise ValueError("Invalid host-bound executable identity")
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed declared workload session") from exc

    def _execute(self, raw, bound):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = self._step(source, evidence, bound)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(source["configuration"]), "runtimes": {self.role: bound[1]}, "steps": [step]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, self._step(source, evidence, bound))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories))

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute(raw, self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw." + self.kind + "-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        return {"session": fresh, "replay_receipt": receipt}
