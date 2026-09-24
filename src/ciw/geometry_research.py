"""Pinned mathematical providers; retained records are not calibrated states.

The native repositories own the algorithms. Offline validation checks the
declared profile, shape, identities and historical reproduction, never imports
or executes a provider and never authenticates a fabricated numerical result.
"""
from __future__ import annotations

from copy import deepcopy
import re
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .declared_workload import AUTHORITY, RESULT_SCHEMA, _text
from .geodesic_reference import GeodesicReferenceWorkflow
from .pipelines import provider_pin
from .telemetry import canonical, digest, _keys

SOURCE_LIMIT = 128 * 1024
KINDS = frozenset({"covariance-geometry", "mesh-path", "translation-flow"})
# The pipeline descriptors are the pin definition; this module executes them.
PINS = {kind: provider_pin(kind) for kind in sorted(KINDS)}
REQUEST_SCHEMAS = {"covariance-geometry": "covariance-geometry-request-v1",
    "mesh-path": "isgt.edge-geodesic-request.v1", "translation-flow": "tsde.square-tiled-flow-request.v1"}
RESULT_SCHEMAS = {"covariance-geometry": "covariance-geometry-result-v1",
    "mesh-path": "isgt.edge-geodesic-result.v1", "translation-flow": "tsde.square-tiled-flow-result.v1"}
CLAIMS = {"covariance-geometry": "declared-spd-affine-invariant-geometry",
    "mesh-path": "edge_constrained_upper_bound_on_declared_piecewise_flat_mesh",
    "translation-flow": "exact_rational_translation_flow_prefix_on_declared_square_tiled_surface"}
POLICIES = {
    "covariance-geometry": {"metric": "affine_invariant", "covariance_basis": "declared_matrices",
        "physical_calibration": "not_established", "state_admission": "not_performed"},
    "mesh-path": {"algorithm": "edge_dijkstra", "path_scope": "edge_constrained_upper_bound",
        "physical_calibration": "not_established", "state_admission": "not_performed"},
    "translation-flow": {"surface": "declared_square_tiled", "arithmetic": "exact_rational",
        "partial_trajectory": "retain_explicit_status", "state_admission": "not_performed"},
}


def _same(actual, expected, message):
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


def _contract(kind):
    if kind == "covariance-geometry":
        from . import geometry_covariance_contract as contract
    elif kind == "mesh-path":
        from . import geometry_mesh_contract as contract
    elif kind == "translation-flow":
        from . import geometry_translation_contract as contract
    else:
        raise ValueError("Unsupported geometry provider kind")
    return contract


def _source(kind, raw):
    try:
        if not isinstance(raw, bytes) or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Geometry source requires 1..131072 exact bytes")
        source = _json(raw)
        _keys(source, {"schema", "experiment_id", "configuration", "request"})
        if source["schema"] != "ciw." + kind + "-source.v1":
            raise ValueError("Unsupported geometry source schema")
        _text(source["experiment_id"])
        _same(source["configuration"], POLICIES[kind], "Require the declared mathematical scope")
        request = source["request"]
        if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMAS[kind]:
            raise ValueError("Unexpected native request profile")
        _contract(kind).validate_request(request)
        if kind == "covariance-geometry" and request["experiment_id"] != source["experiment_id"]:
            raise ValueError("Native covariance request identity differs")
        return source
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed geometry source") from exc


_BOOTSTRAP = r'''
import json, sys
role, root = sys.argv[1:3]
sys.path.insert(0, root)
if role == 'cggt':
    from covariance_geometry import run
elif role == 'isgt':
    from intrinsic_surface_geodesics import run
elif role == 'tsde':
    from translation_surface_dynamics import run
else:
    raise ValueError('Unsupported fixed geometry provider')
source = json.loads(sys.stdin.buffer.read())
print(json.dumps(run(source['request']), sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False))
'''


def _check_data(kind, source, data):
    """Validate retained envelopes and numerical shapes, not scientific truth."""
    if not isinstance(data, dict) or data.get("schema") != RESULT_SCHEMAS[kind]:
        raise ValueError("Unsupported native geometry result")
    _same(data["request"], source["request"], "Native result names another request")
    if data["request_digest"] != digest(source["request"]):
        raise ValueError("Native request digest mismatch")
    if data["artifact_digest"] != digest({k:v for k,v in data.items() if k != "artifact_digest"}):
        raise ValueError("Native geometry artifact digest mismatch")
    if data["claim_scope"] != CLAIMS[kind]:
        raise ValueError("Native geometry claim exceeds its bounded profile")
    _contract(kind).validate_result(source["request"], data)


class GeometryResearchWorkflow(GeodesicReferenceWorkflow):
    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError("Unsupported geometry provider kind")
        self.kind, self.pin = kind, PINS[kind]
        self.role, self.ROLES = self.pin["role"], {self.pin["role"]}
        self.SOURCE_SCHEMA = "ciw." + kind + "-source.v1"
        self.schema, self.operation = "ciw." + kind + "-session.v1", "ciw." + kind + ".v1"

    def _source(self, raw):
        return _source(self.kind, raw)

    def _check_runtime(self, runtime):
        if runtime["source_tree"] != self.pin["source_tree"] or runtime["adapter_version"] != "ciw-pinned-subprocess-v1":
            raise ValueError("Geometry provider source or adapter differs from its pin")
        if not re.fullmatch(r"\d+\.\d+\.\d+", runtime["python_version"]):
            raise ValueError("Invalid geometry Python version")
        if tuple(map(int, runtime["python_version"].split(".")[:2])) < (3, 11):
            raise ValueError("Geometry providers require Python 3.11 or newer")
        _keys(runtime["dependencies"], {"numpy", "scipy"})
        if runtime["dependencies"]["numpy"] != "2.4.3":
            raise ValueError("Geometry profile requires the workbench NumPy 2.4.3 pin")
        if runtime["dependencies"]["scipy"] is not None:
            _text(runtime["dependencies"]["scipy"])

    def _step(self, source, evidence_id, bound):
        adapter, runtime, _ = bound
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(runtime), "Geometry provider changed before execution")
        code, raw = adapter._run(_BOOTSTRAP, [self.role, str(adapter.source_root)], canonical(source))
        _same(self._runtime_projection(adapter.runtime_identity()), self._runtime_projection(runtime), "Geometry provider changed during execution")
        if code:
            raise AdapterRefusal("GEOMETRY_PROVIDER_REFUSED", "Pinned " + self.role + " refused the declared mathematical request")
        data = _json(raw)
        _check_data(self.kind, source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": deepcopy(AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": deepcopy(data)}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
            "input_refs": [evidence_id], "request": deepcopy(source), "request_sha256": digest(source),
            "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
            "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
            "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if (step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id] or
                not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Invalid geometry operation, evidence or execution occurrence")
        _same(step["request"], source, "Geometry request differs from retained source")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(self.kind, source, result["data"])
        _same(result["authority"], AUTHORITY, "Geometry calculation cannot confer state or physical authority")
        if (result["schema"] != RESULT_SCHEMA or result["operation_id"] != self.operation or result["execution_ref"] != step["execution_id"] or
                result["input_refs"] != [evidence_id] or result["result_id"] != step["result_id"] or
                result["result_id"] != digest({k:v for k,v in result.items() if k != "result_id"})):
            raise ValueError("Geometry result binding mismatch")
        _same(step["numerical_result"], {"operation_id": self.operation, "data": result["data"]}, "Geometry numerical projection mismatch")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("Geometry step content binding mismatch")
