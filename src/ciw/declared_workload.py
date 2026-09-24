"""Bounded native schematic assessment and integer numerical execution.

These are workbench objects, not observations or state estimates. Repository
and executable paths are trusted host bindings and never read from artifacts.
Replay checks reproducibility; it does not authenticate submitted certificates.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import struct
import tempfile

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .core.canonical import byte_digest, canonical, exact_keys
from .exchange import _read
from .pipelines import provider_pin
from .pipelines.runner import (AUTHORITY, MAX_BYTES, RESULT_SCHEMA, VERIFY_SCHEMA, PipelineRunner,  # noqa: F401
                               text as _text, verification as _verification)

SOURCE_LIMIT = 262144
KINDS = frozenset({"schematic-assessment", "numerical-heat"})
PINS = {kind: provider_pin(kind) for kind in sorted(KINDS)}
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
_EDGES = {"input": ("variable", "function"), "output": ("function", "variable"),
          "measures": ("measurement", "variable"), "actuates": ("variable", "function"),
          "binds": ("evidence", "measurement"), "linearizes": ("certificate", "function"),
          "certifies": ("certificate", "function"), "observes": ("observer", "function"),
          "senses": ("observer", "measurement"), "composed_in": ("function", "function")}


def _graph(graph):
    exact_keys(graph, {"schema", "meta", "nodes", "edges"})
    if graph["schema"] != GRAPH_SCHEMA or not isinstance(graph["meta"], dict) or graph["meta"].get("schema") != GRAPH_SCHEMA:
        raise ValueError("Require explicit matching graph and metadata schemas")
    if not isinstance(graph["nodes"], list) or not 1 <= len(graph["nodes"]) <= 256:
        raise ValueError("Schematic node budget is 1..256")
    if not isinstance(graph["edges"], list) or len(graph["edges"]) > 1024:
        raise ValueError("Schematic edge budget is 1024")
    nodes = {}
    for node in graph["nodes"]:
        exact_keys(node, {"id", "kind", "attrs"})
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
        exact_keys(edge, {"kind", "src", "dst", "attrs"})
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
    exact_keys(source, {"schema", "experiment_id", "configuration"} |
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
        exact_keys(data, {"schematic", "decisions", "neighborhoods", "next_step"})
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
            exact_keys(decision, {"tool", "owner", "status", "node_id", "reason"})
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
        exact_keys(data, {"specification", "specification_identity", "program_identity", "input_identity", "engine_occurrence",
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


class DeclaredWorkflow(PipelineRunner):
    """Bounded SRA assessment or SCR integer execution through the shared runner."""

    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError("Unsupported declared workload")
        super().__init__(kind, PINS[kind])
        if self.role == "scr":
            self.ROLES = self.ROLES | {"engine"}
            self.RUNTIME_EXTRA = frozenset({"engine"})

    def parse_source(self, raw):
        return _source(self.kind, raw)

    def bind_extra(self, repositories, adapter, runtime):
        if self.role != "scr":
            return None
        engine = _read(Path(repositories["engine"]), 32 * 1024 * 1024)
        runtime["engine"] = {"sha256": byte_digest(engine), "byte_count": len(engine),
                             "source_binding": "operator_asserted_not_attested"}
        return engine

    def check_runtime(self, runtime):
        if self.role != "scr":
            return
        engine = runtime["engine"]
        exact_keys(engine, {"sha256", "byte_count", "source_binding"})
        if (not isinstance(engine["sha256"], str) or not re.fullmatch("sha256:[a-f0-9]{64}", engine["sha256"]) or
                type(engine["byte_count"]) is not int or not 1 <= engine["byte_count"] <= 32 * 1024 * 1024 or
                engine["source_binding"] != "operator_asserted_not_attested"):
            raise ValueError("Invalid host-bound executable identity")

    def invoke(self, source, bound):
        adapter, _, engine = bound
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
        # Re-verify the provider before its outcome is classified.
        self._unchanged(adapter, bound[1], "during")
        if code:
            raise AdapterRefusal("DECLARED_WORKLOAD_REFUSED", "Pinned " + self.role + " refused the declared workload")
        return _json(raw)

    def check_data(self, source, data):
        _check_data(self.kind, source, data)
