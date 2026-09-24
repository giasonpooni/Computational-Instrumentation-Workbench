"""Selected retained schematic → pinned SRA/JSPT/PLSR companion execution.

The initial operation covers the native scalar quadratic-drag state derivative.
Its Lyapunov sample belongs to the continuous linear surrogate e_dot = J e;
no equilibrium, nonlinear region, observer, or physical state is certified.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .declared_workload import (AUTHORITY, DeclaredWorkflow, MAX_BYTES, RESULT_SCHEMA,
                               SOURCE_LIMIT, _graph, _text, _verification)
from .core.canonical import canonical, digest, byte_digest, bundle_digest, exact_keys, utc_now

KIND = "schematic-companions"
SOURCE_SCHEMA = "ciw.schematic-companions-source.v1"
OPERATION = "ciw.schematic-companions.v1"
ROLES = frozenset({"sra", "jspt", "plsr"})
PINS = {
    "sra": {"revision": "a6e79585950bb6860e5edce5ebd2cce39ea481f2", "source_root": "src", "module": "schematics.eligibility"},
    "jspt": {"revision": "7399ab03087b27683620b4c57f97b2ac14546c7f", "source_root": "src", "module": "sensitivity.jacobian"},
    "plsr": {"revision": "9d0e7b4a1162e038150a71c63d986945d78135d4", "source_root": "src", "module": "lyapunov.runtime"},
}
POLICY = {
    "companion_execution": "selected_function_only",
    "model_semantics": "continuous_scalar_state_derivative",
    "model_parameter_uncertainty": "declared_exact",
    "cross_covariance": "not_applicable_single_declared_input",
    "covariance": "declared_input_only_first_order",
    "coordinate_metric": "declared_SI_scalar",
    "rank_absolute_tolerance": 1e-10,
    "stability_scope": "continuous_linear_surrogate_e_dot_equals_J_e",
    "lyapunov_probe": "native_x_star_argument_in_linear_surrogate",
    "equilibrium_status": "not_established",
    "observability": "not_evaluated",
    "certificate_authentication": "not_performed",
}
TOOLS = ("jspt.jacobian_at", "jspt.local_structure", "jspt.first_order_covariance", "lyapunov.evaluate")
NATIVE_PROVENANCE = {"kind": "adapter_call_record", "authentication": "not_authenticated",
                     "kernel_revision_status": "declared_pin_not_verified", "independent_verification": "not_claimed"}
NATIVE_JSPT = {"repo": "giasonpooni/Jacobian-Sensitivity-Propagation-Testbed", "sha": PINS["jspt"]["revision"], "import": "sensitivity"}
NATIVE_PLSR = {"repo": "giasonpooni/Parameterized-Lyapunov-Stability-Runtime", "sha": PINS["plsr"]["revision"], "import": "lyapunov"}


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
        raise ValueError("Companion source exceeds byte budget")
    source = _json(raw)
    canonical(source)
    exact_keys(source, {"schema", "experiment_id", "configuration", "schematic", "function_id"})
    if source["schema"] != SOURCE_SCHEMA or source["configuration"] != POLICY:
        raise ValueError("Require explicit bounded companion semantics")
    _text(source["experiment_id"])
    _text(source["function_id"])
    nodes = _graph(source["schematic"])
    if len(nodes) > 252 or len(source["schematic"]["edges"]) > 1020:
        raise ValueError("Leave room for four declared companion certificates")
    target = nodes.get(source["function_id"])
    if target is None or target["kind"] != "function":
        raise ValueError("Select one explicitly declared schematic function")
    attrs = target["attrs"]
    if set(attrs) - {"class", "model_ref", "x_star", "c", "law", "chart", "units", "sigma_x", "rank", "invisible_dim"}:
        raise ValueError("Selected native scalar function has undeclared model or uncertainty fields")
    if attrs.get("model_ref") != "jspt.reference.quadratic_drag" or attrs.get("class") not in {"nonlinear", "unknown"}:
        raise ValueError("Initial companion operation requires the declared native scalar quadratic-drag model")
    x = attrs.get("x_star")
    if not isinstance(x, list) or len(x) != 1 or not _finite(x[0]) or abs(x[0]) > 1e6:
        raise ValueError("Declare one finite bounded x_star coordinate")
    if not _finite(attrs.get("c")) or not 0 <= attrs["c"] <= 1e6 or attrs.get("chart") != "identity" or attrs.get("units") != "SI":
        raise ValueError("Declare exact nonnegative c, identity chart, and SI scalar coordinates")
    if "sigma_x" in attrs:
        sigma = attrs["sigma_x"]
        if not isinstance(sigma, list) or len(sigma) != 1 or not isinstance(sigma[0], list) or len(sigma[0]) != 1 or not _finite(sigma[0][0]) or not 0 <= sigma[0][0] <= 1e12:
            raise ValueError("Declared input covariance must be a finite nonnegative scalar matrix")
    for identifier in _certificate_ids(source["function_id"]):
        if identifier in nodes and nodes[identifier]["kind"] != "certificate":
            raise ValueError("Native certificate identity collides with a declared non-certificate node")
    return source


def _finite(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _scalar_matrix(value):
    return (isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
            and len(value[0]) == 1 and _finite(value[0][0]))


def _scalar_basis(value, count):
    return (isinstance(value, list) and len(value) == count
            and all(isinstance(column, list) and len(column) == 1 and _finite(column[0])
                    and math.isclose(abs(column[0]), 1.0, rel_tol=1e-12, abs_tol=1e-12)
                    for column in value))


def _certificate_ids(target):
    return ("cert:jspt:" + target, "cert:jspt.structure:" + target,
            "cert:jspt.cov:" + target, "cert:lyapunov:" + target)


def _native_digest(namespace, value):
    return "sha256:" + sha256((namespace + "\0" + json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)).encode()).hexdigest()


def _declaration(graph, target):
    nodes = {node["id"]: node for node in graph["nodes"]}
    attrs = {k: v for k, v in nodes[target]["attrs"].items() if k not in {"rank", "invisible_dim", "sigma_x"}}
    ports = []
    for edge in graph["edges"]:
        if edge["kind"] in {"input", "output", "actuates"} and target in {edge["src"], edge["dst"]}:
            other = nodes[edge["dst"] if edge["src"] == target else edge["src"]]
            ports.append({"edge": edge["kind"], "src": edge["src"], "dst": edge["dst"], "attrs": edge["attrs"],
                          "node_kind": other["kind"], "node_attrs": other["attrs"]})
    return {"target": target, "function": attrs, "ports": ports}


_BOOTSTRAP = r'''
import dataclasses, importlib, json, pathlib, sys
roots = dict(zip(('sra', 'jspt', 'plsr'), sys.argv[1:4]))
for root in reversed(list(roots.values())):
    sys.path.insert(0, root)
for role, module in (('sra', 'schematics'), ('jspt', 'sensitivity'), ('plsr', 'lyapunov')):
    loaded = importlib.import_module(module)
    if not pathlib.Path(loaded.__file__).resolve().is_relative_to(pathlib.Path(roots[role]).resolve()):
        raise ValueError('Companion module resolved outside pinned source root')
from schematics.io import from_dict, to_dict
from schematics.eligibility import decide
from schematics.annotate import invalidate_stale_results
from schematics.adapters.jspt import call_jacobian_at
from schematics.adapters.structure import call_local_structure
from schematics.adapters.covariance import call_first_order_covariance
from schematics.adapters.plsr import call_evaluate
source = json.loads(sys.stdin.buffer.read())
graph, target = from_dict(source['schematic']), source['function_id']
invalidate_stale_results(graph)
before = [dataclasses.asdict(d) for d in decide(graph) if d.node_id == target]
events = []
for tool, call in (('jspt.jacobian_at', call_jacobian_at), ('jspt.local_structure', call_local_structure),
                   ('jspt.first_order_covariance', call_first_order_covariance), ('lyapunov.evaluate', call_evaluate)):
    decision = next(d for d in decide(graph) if d.node_id == target and d.tool == tool)
    if decision.status.value == 'ELIGIBLE':
        event = dataclasses.asdict(call(graph, target))
    else:
        event = {'tool': tool, 'owner': decision.owner, 'node_id': target, 'result': 'NOT_ELIGIBLE',
                 'detail': {'reason': decision.reason}}
    events.append(event)
invalidate_stale_results(graph)
data = {'schematic': to_dict(graph), 'function_id': target, 'decisions_before': before, 'events': events,
        'decisions_after': [dataclasses.asdict(d) for d in decide(graph) if d.node_id == target], 'scope': source['configuration']}
print(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
'''


def numerical_projection(data):
    """Remove only identities freshly generated by this operation, never history."""
    projected = deepcopy(data)
    event = data["events"][0]
    if event["result"] == "SAMPLED":
        result_id, execution_id = event["detail"]["result_ref"], event["detail"]["execution_ref"]
        def replace(value):
            if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list): return [replace(v) for v in value]
            if value == result_id: return "$current_jacobian_result"
            if value == execution_id: return "$current_jacobian_execution"
            return value
        projected = replace(projected)
    return {"operation_id": OPERATION, "data": projected}


def native_occurrences(bundle):
    """New native calls in this bundle, excluding upstream and submitted history."""
    found = []
    for step in (*bundle["steps"], bundle["verification"]["reproduction"]):
        event = step["result"]["data"]["events"][0]
        if event["result"] == "SAMPLED":
            found.append(event["detail"]["execution_ref"])
    if len(found) != len(set(found)):
        raise ValueError("Companion primary and reproduction must have distinct native occurrences")
    return set(found)


def _check_data(source, data):
    exact_keys(data, {"schematic", "function_id", "decisions_before", "events", "decisions_after", "scope"})
    if data["function_id"] != source["function_id"] or data["scope"] != POLICY:
        raise ValueError("Companion result scope or selection differs")
    nodes, old = _graph(data["schematic"]), _graph(source["schematic"])
    target, ids = source["function_id"], _certificate_ids(source["function_id"])
    if data["schematic"]["meta"] != source["schematic"]["meta"] or set(nodes) - set(old) - set(ids) or not set(old) <= set(nodes):
        raise ValueError("Companions must retain all declared nodes and metadata")
    for identifier, node in old.items():
        if identifier == target:
            actual = deepcopy(nodes[identifier])
            for key in {"rank", "invisible_dim"}:
                actual["attrs"].pop(key, None)
            expected = deepcopy(node)
            for key in {"rank", "invisible_dim"}:
                expected["attrs"].pop(key, None)
            if canonical(actual) != canonical(expected):
                raise ValueError("Companion execution changed its declared function")
        elif identifier not in ids and canonical(node) != canonical(nodes[identifier]):
            # Stale marking may modify only a certificate's status while retaining
            # its numerical and binding evidence. The submitted graph remains in
            # the immutable source and in upstream_assessment.
            expected = deepcopy(node)
            expected["attrs"].update(historical_result="SAMPLED", result="NOT_ELIGIBLE", currentness="stale",
                stale_reason="missing, stale, ambiguous or superseded Jacobian/covariance input binding")
            if node["kind"] != "certificate" or node["attrs"].get("result") != "SAMPLED" or canonical(expected) != canonical(nodes[identifier]):
                raise ValueError("Companion execution altered unrelated evidence")
    edges = source["schematic"]["edges"]
    if data["schematic"]["edges"][:len(edges)] != edges:
        raise ValueError("Companion execution changed declared topology")
    expected_additions = [{"kind": "certifies" if i == ids[3] else "linearizes", "src": i, "dst": target, "attrs": {}}
                          for i in ids if i not in old and i in nodes]
    if data["schematic"]["edges"][len(edges):] != expected_additions:
        raise ValueError("Unexpected companion certificate edges")
    if not isinstance(data["events"], list) or len(data["events"]) != 4:
        raise ValueError("Require the four declared companion outcomes")
    for tool, event in zip(TOOLS, data["events"]):
        exact_keys(event, {"tool", "owner", "node_id", "result", "detail"})
        if event["tool"] != tool or event["node_id"] != target or event["owner"] != ("plsr" if tool == TOOLS[3] else "jspt") or event["result"] not in {"SAMPLED", "REFUSED", "NOT_CHECKED", "NOT_ELIGIBLE"} or not isinstance(event["detail"], dict):
            raise ValueError("Invalid native companion outcome")
        if event["result"] != "NOT_ELIGIBLE" and event["detail"].get("pin") != (NATIVE_PLSR if tool == TOOLS[3] else NATIVE_JSPT):
            raise ValueError("Native event must retain its declared companion pin")
    attrs = old[target]["attrs"]
    decision_tools = ([TOOLS[0]] + (["jspt.sweep_perturbation_scale"] if attrs["class"] != "unknown" else [])
                      + ["jspt.check_coordinate_consistency", *TOOLS[1:]])
    for label in ("decisions_before", "decisions_after"):
        if not isinstance(data[label], list) or len(data[label]) != len(decision_tools):
            raise ValueError("Invalid native eligibility table")
        for tool, decision in zip(decision_tools, data[label]):
            exact_keys(decision, {"tool", "owner", "status", "node_id", "reason"})
            if (decision["tool"] != tool or decision["owner"] != ("plsr" if tool == TOOLS[3] else "jspt")
                    or decision["node_id"] != target or decision["status"] not in {"ELIGIBLE", "NOT_ELIGIBLE"}):
                raise ValueError("Eligibility is not execution or observability")
            _text(decision["reason"])
            if (attrs["class"] == "unknown" and decision["status"] != "NOT_ELIGIBLE"
                    or tool == "jspt.check_coordinate_consistency" and decision["status"] != "NOT_ELIGIBLE"
                    or attrs["class"] == "nonlinear" and tool in {TOOLS[0], "jspt.sweep_perturbation_scale"} and decision["status"] != "ELIGIBLE"):
                raise ValueError("Eligibility differs from the declared scalar model and identity chart")
    jacobian = data["events"][0]
    if attrs["class"] == "unknown" and any(event["result"] != "NOT_ELIGIBLE" for event in data["events"]):
        raise ValueError("Unknown function class cannot execute companions")
    if jacobian["result"] == "SAMPLED":
        cert = nodes[ids[0]]["attrs"]
        binding = cert["binding"]
        exact_keys(binding, {"schema", "operation", "execution_id", "declaration_id", "call_inputs", "kernel", "provenance", "result_id"})
        if not isinstance(binding["execution_id"], str) or not re.fullmatch(r"sra-execution:[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", binding["execution_id"]):
            raise ValueError("Invalid native Jacobian occurrence")
        if binding["execution_id"] in canonical(source["schematic"]).decode():
            raise ValueError("New native Jacobian occurrence cannot reuse submitted history")
        expected_A = [[-2.0 * float(attrs["c"]) * abs(float(attrs["x_star"][0]))]]
        if (not _scalar_matrix(cert["A"]) or cert["A"] != expected_A or cert.get("fixture") is not False or cert.get("tool") != TOOLS[0]
                or cert.get("owner") != "jspt" or cert.get("pin") != NATIVE_JSPT["repo"] + "@" + NATIVE_JSPT["sha"]
                or binding["schema"] != "NsJacobianBinding@0.1" or binding["operation"] != TOOLS[0]
                or binding["kernel"] != NATIVE_JSPT or binding["provenance"] != NATIVE_PROVENANCE
                or binding["declaration_id"] != _native_digest("sra.declaration.v1", _declaration(source["schematic"], target))
                or canonical(binding["call_inputs"]) != canonical({"model_ref": attrs["model_ref"], "x_star": [float(attrs["x_star"][0])], "parameters": {"c": float(attrs["c"])}})
                or binding["result_id"] != _native_digest("sra.jacobian-result.v1", {**{k: v for k, v in binding.items() if k != "result_id"}, "A": cert["A"]})
                or jacobian["detail"].get("result_ref") != binding["result_id"] or jacobian["detail"].get("execution_ref") != binding["execution_id"]
                or not _scalar_matrix(jacobian["detail"].get("A")) or jacobian["detail"].get("A") != cert["A"] or jacobian["detail"].get("fixture") is not False
                or jacobian["detail"].get("provenance") != NATIVE_PROVENANCE):
            raise ValueError("Native Jacobian declaration, numerical, or occurrence binding mismatch")
        for index, event in enumerate(data["events"][1:], 1):
            if event["result"] == "SAMPLED":
                dependent = nodes[ids[index]]["attrs"]
                native = NATIVE_PLSR if index == 3 else NATIVE_JSPT
                if dependent.get("source_result_ref") != binding["result_id"] or dependent.get("source_execution_ref") != binding["execution_id"] or dependent.get("tool") != TOOLS[index] or dependent.get("result") != "SAMPLED" or dependent.get("owner") != event["owner"] or dependent.get("pin") != native["repo"] + "@" + native["sha"]:
                    raise ValueError("Dependent native certificate uses another Jacobian occurrence")
                if index == 1:
                    rank = int(abs(expected_A[0][0]) > POLICY["rank_absolute_tolerance"])
                    singular = dependent.get("singular_values")
                    counts = (dependent.get("rank"), event["detail"].get("rank"), event["detail"].get("invisible_dim"),
                              nodes[target]["attrs"].get("rank"), nodes[target]["attrs"].get("invisible_dim"))
                    if (any(type(value) is not int for value in counts) or counts != (rank, rank, 1 - rank, rank, 1 - rank)
                            or not isinstance(singular, list) or len(singular) != 1 or not _finite(singular[0]) or singular[0] != abs(expected_A[0][0])
                            or not _scalar_basis(dependent.get("invisible"), 1 - rank) or not _scalar_basis(dependent.get("visible"), rank)):
                        raise ValueError("Local structure must describe the scalar Jacobian, not observability")
                if index == 2:
                    expected = expected_A[0][0] * attrs["sigma_x"][0][0] * expected_A[0][0]
                    actual = dependent.get("sigma_y")
                    if (not _scalar_matrix(dependent.get("sigma_x")) or dependent.get("sigma_x") != attrs["sigma_x"] or dependent.get("exact_for_affine") is not False
                            or not isinstance(actual, list) or len(actual) != 1 or not isinstance(actual[0], list) or len(actual[0]) != 1
                            or not _finite(actual[0][0]) or not math.isclose(actual[0][0], expected, rel_tol=1e-12, abs_tol=0.0)
                            or not _scalar_matrix(event["detail"].get("sigma_y")) or event["detail"].get("sigma_y") != actual):
                        raise ValueError("Covariance must propagate the declared single-input covariance")
                if index == 3:
                    matrix = dependent.get("P")
                    if not isinstance(matrix, list) or len(matrix) != 1 or not isinstance(matrix[0], list) or len(matrix[0]) != 1 or not _finite(matrix[0][0]):
                        raise ValueError("Require a finite scalar Lyapunov matrix")
                    p, a, x = matrix[0][0], expected_A[0][0], attrs["x_star"][0]
                    if (a >= 0 or p <= 0 or not math.isclose(2 * a * p, -1.0, rel_tol=1e-12, abs_tol=1e-12)
                            or not _scalar_matrix(dependent.get("A")) or dependent.get("A") != expected_A or not _finite(dependent.get("V")) or not _finite(dependent.get("decrease"))
                            or not math.isclose(dependent["V"], p * x * x, rel_tol=1e-12, abs_tol=1e-12)
                            or not math.isclose(dependent["decrease"], 2 * a * p * x * x, rel_tol=1e-12, abs_tol=1e-12)
                            or dependent.get("verdict") != "certified"
                            or not _finite(event["detail"].get("V")) or not _finite(event["detail"].get("decrease"))
                            or any(event["detail"].get(k) != dependent.get(k) for k in ("V", "decrease", "verdict"))):
                        raise ValueError("Lyapunov sample must bind the declared continuous linear surrogate")
    elif any(event["result"] == "SAMPLED" for event in data["events"][1:]):
        raise ValueError("Companions require the freshly selected Jacobian")
    if "sigma_x" not in attrs and data["events"][2]["result"] != "NOT_ELIGIBLE":
        raise ValueError("Input covariance must never be invented")


def validate_upstream(bundle, upstream):
    """Bind to the exact retained assessment result, not a reconstructed graph."""
    DeclaredWorkflow("schematic-assessment")._validate(upstream)
    if canonical(bundle["upstream_assessment"]) != canonical(upstream):
        raise ValueError("Companion upstream assessment differs from the selected retained bundle")
    result = upstream["steps"][0]
    expected = {"bundle_id": upstream["bundle_digest"], "result_id": result["result_id"],
                "numerical_result_id": result["numerical_result_id"], "graph_digest": digest(result["result"]["data"]["schematic"])}
    if bundle["upstream_binding"] != expected:
        raise ValueError("Companion upstream result identity binding mismatch")
    return result["result"]["data"]["schematic"]


class SchematicCompanionWorkflow(DeclaredWorkflow):
    """The existing declared-workload lifecycle with three explicit provider pins."""
    ROLES = ROLES
    SOURCE_SCHEMA = SOURCE_SCHEMA

    def __init__(self):
        self.kind, self.role, self.pin = KIND, "sra", PINS["sra"]
        self.schema, self.operation = "ciw.schematic-companions-session.v1", OPERATION

    def _source(self, raw): return _source(raw)

    def _adapters(self, repositories, expected=None):
        if set(repositories) != ROLES:
            raise ValueError("Bind exactly the native SRA, JSPT and PLSR companion checkouts")
        old = expected["sra"] if expected else None
        adapters, runtimes = {}, {}
        for role, pin in PINS.items():
            retained = (old if role == "sra" else old["companions"][role]) if old else {}
            adapters[role] = PinnedSubprocessAdapter(repositories[role], pin["revision"], pin["module"], source_root=pin["source_root"],
                expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"), expected_dependencies=retained.get("dependencies"))
            runtimes[role] = adapters[role].runtime_identity()
        runtime = {**runtimes["sra"], "companions": {role: runtimes[role] for role in ("jspt", "plsr")}}
        if old and self._runtime_projection(runtime) != self._runtime_projection(old):
            raise ValueError("Companion runtime differs from the retained execution")
        return adapters, runtime, None

    @staticmethod
    def _runtime_projection(runtime):
        value = DeclaredWorkflow._runtime_projection(runtime)
        value["companions"] = {role: DeclaredWorkflow._runtime_projection(item) for role, item in runtime["companions"].items()}
        return value

    def _step(self, source, evidence_id, bound, upstream_result_id):
        adapters, runtime, _ = bound
        before = {role: adapter.runtime_identity() for role, adapter in adapters.items()}
        current = {**before["sra"], "companions": {r: before[r] for r in ("jspt", "plsr")}}
        if self._runtime_projection(current) != self._runtime_projection(runtime):
            raise ValueError("Companion runtime changed before execution")
        code, raw = adapters["sra"]._run(_BOOTSTRAP, [str(adapters[r].source_root) for r in ("sra", "jspt", "plsr")], canonical(source))
        for role, adapter in adapters.items():
            if adapter.runtime_identity() != before[role]:
                raise ValueError("Companion runtime changed during execution")
        if code:
            raise AdapterRefusal("SCHEMATIC_COMPANION_REFUSED", "Pinned schematic companion process refused the declared operation")
        data = _json(raw)
        _check_data(source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        refs = [evidence_id, upstream_result_id]
        result = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": occurrence,
                  "input_refs": refs, "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = numerical_projection(data)
        return {"runtime_ref": "sra", "operation_id": OPERATION, "execution_id": occurrence, "input_refs": refs,
                "request": source, "request_sha256": digest(source), "result": result, "result_sha256": digest(result),
                "result_id": result["result_id"], "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        exact_keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        refs = step["input_refs"]
        if (step["runtime_ref"] != "sra" or step["operation_id"] != OPERATION or not isinstance(refs, list) or len(refs) != 2
                or refs[0] != evidence_id or not isinstance(refs[1], str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", refs[1])
                or canonical(step["request"]) != canonical(source) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Companion request or execution binding mismatch")
        result = step["result"]
        exact_keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(source, result["data"])
        if result["schema"] != RESULT_SCHEMA or result["operation_id"] != OPERATION or result["authority"] != AUTHORITY or result["execution_ref"] != step["execution_id"] or result["input_refs"] != refs or result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"}) or result["result_id"] != step["result_id"] or canonical(step["numerical_result"]) != canonical(numerical_projection(result["data"])):
            raise ValueError("Companion result or numerical identity mismatch")
        for key, value in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(value): raise ValueError("Companion content digest mismatch")

    def _check_verification(self, bundle, verification, source, evidence):
        super()._check_verification(bundle, verification, source, evidence)
        if verification["reproduction"]["input_refs"] != [evidence, bundle["upstream_binding"]["result_id"]]:
            raise ValueError("Companion reproduction must consume the same retained upstream result")
        first, second = bundle["steps"][0]["result"]["data"], verification["reproduction"]["result"]["data"]
        if first["events"][0]["result"] == "SAMPLED" and first["events"][0]["detail"]["execution_ref"] == second["events"][0]["detail"]["execution_ref"]:
            raise ValueError("Companion reproduction must use a fresh native Jacobian occurrence")

    def _validate(self, bundle):
        try:
            exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification", "upstream_assessment", "upstream_binding"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Companion bundle binding mismatch")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = _source(raw)
            if evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()} or bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]} or bundle["configuration"] != POLICY:
                raise ValueError("Companion source evidence mismatch")
            if canonical(validate_upstream(bundle, bundle["upstream_assessment"])) != canonical(source["schematic"]):
                raise ValueError("Source schematic must exactly match the selected assessment result")
            if set(bundle["runtimes"]) != {"sra"}:
                raise ValueError("Unexpected companion runtime roles")
            primary = bundle["runtimes"]["sra"]
            if set(primary["companions"]) != {"jspt", "plsr"}: raise ValueError("Missing native companion pins")
            for role in ROLES:
                runtime = primary if role == "sra" else primary["companions"][role]
                exact_keys(runtime, {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root", "python_executable", "python_sha256", "python_version", "dependencies"} | ({"companions"} if role == "sra" else set()))
                if runtime["schema"] != "ciw.subprocess-runtime.v1" or any(runtime[k] != PINS[role][k] for k in ("revision", "module", "source_root")) or not re.fullmatch(r"[0-9a-f]{40}", runtime["source_tree"]) or not re.fullmatch(r"[0-9a-f]{64}", runtime["python_sha256"]):
                    raise ValueError("Unapproved companion provider pin")
                for key in ("adapter_version", "repository_root", "python_executable", "python_version"): _text(runtime[key])
                if not isinstance(runtime["dependencies"], dict): raise ValueError("Invalid companion dependency identities")
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            if step["input_refs"][1] != bundle["upstream_binding"]["result_id"]:
                raise ValueError("Companion execution must consume its selected upstream result")
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            native_occurrences(bundle)
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed companion session") from exc

    def _execute_selected(self, raw, upstream, bound):
        source = _source(raw)
        DeclaredWorkflow("schematic-assessment")._validate(upstream)
        selected = upstream["steps"][0]
        if canonical(source["schematic"]) != canonical(selected["result"]["data"]["schematic"]):
            raise ValueError("Source schematic must exactly match the selected assessment result")
        evidence = byte_digest(raw)
        step = self._step(source, evidence, bound, selected["result_id"])
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": utc_now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(POLICY), "runtimes": {"sra": bound[1]}, "steps": [step],
                  "upstream_assessment": deepcopy(upstream), "upstream_binding": {"bundle_id": upstream["bundle_digest"], "result_id": selected["result_id"],
                    "numerical_result_id": selected["numerical_result_id"], "graph_digest": digest(selected["result"]["data"]["schematic"])}}
        bundle["bundle_digest"] = bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, self._step(source, evidence, bound, selected["result_id"]))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, upstream_assessment, repositories):
        _source(raw)
        return self._execute_selected(raw, upstream_assessment, self._adapters(repositories))

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute_selected(raw, bundle["upstream_assessment"], self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw.schematic-companions-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._check_verification(bundle, receipt["verification"], _source(raw), byte_digest(raw))
        return {"session": fresh, "replay_receipt": receipt}
