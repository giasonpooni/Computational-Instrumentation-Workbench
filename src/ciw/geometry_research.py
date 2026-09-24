"""Pinned mathematical providers; retained records are not calibrated states.

The native repositories own the algorithms. Offline validation checks the
declared profile, shape, identities and historical reproduction, never imports
or executes a provider and never authenticates a fabricated numerical result.
"""
from __future__ import annotations

import re

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .pipelines import provider_pin
from .pipelines.runner import PipelineRunner, same as _same, text as _text
from .core.canonical import canonical, digest, exact_keys

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
        exact_keys(source, {"schema", "experiment_id", "configuration", "request"})
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


class GeometryResearchWorkflow(PipelineRunner):
    LABEL = "Geometry"

    def __init__(self, kind):
        if kind not in KINDS:
            raise ValueError("Unsupported geometry provider kind")
        super().__init__(kind, PINS[kind])

    def parse_source(self, raw):
        return _source(self.kind, raw)

    def check_runtime(self, runtime):
        if runtime["adapter_version"] != "ciw-pinned-subprocess-v1":
            raise ValueError("Geometry provider adapter differs from its pin")
        if not isinstance(runtime["python_version"], str) or not re.fullmatch(r"\d+\.\d+\.\d+", runtime["python_version"]):
            raise ValueError("Invalid geometry Python version")
        if tuple(map(int, runtime["python_version"].split(".")[:2])) < (3, 11):
            raise ValueError("Geometry providers require Python 3.11 or newer")
        exact_keys(runtime["dependencies"], {"numpy", "scipy"})
        if runtime["dependencies"]["numpy"] != "2.4.3":
            raise ValueError("Geometry profile requires the workbench NumPy 2.4.3 pin")
        if runtime["dependencies"]["scipy"] is not None:
            _text(runtime["dependencies"]["scipy"])

    def invoke(self, source, bound):
        adapter = bound[0]
        code, raw = adapter._run(_BOOTSTRAP, [self.role, str(adapter.source_root)], canonical(source))
        if code:
            raise AdapterRefusal("GEOMETRY_PROVIDER_REFUSED", "Pinned " + self.role + " refused the declared mathematical request")
        return _json(raw)

    def check_data(self, source, data):
        _check_data(self.kind, source, data)
