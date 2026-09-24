"""Pinned GTE circle reconciliation in the shared workbench catalog.

CIW retains exact request bytes and native results. GTE owns projection and
uncertainty propagation. A declared plane is not a surveyed frame and passing
a geometric policy does not authorize physical or construction acceptance.
"""
from __future__ import annotations

import base64
from datetime import datetime

from .adapters.gte_records import validate_payload, _array, _covariance
from .adapters.subprocess import _json
from .pipelines import provider_pin
from .pipelines.runner import PipelineRunner, text as _text
from .core.records import number
from .geodesic import GTE_OPERATION, _make_run, _observations, _parse
from .core.canonical import canonical, exact_keys

KIND = "geometric-circle"
SOURCE_SCHEMA = "ciw.geometric-circle-source.v1"
OPERATION = "ciw.geometric-circle.v1"
ROLES = frozenset({"gte"})
PIN = provider_pin(KIND)
SOURCE_TREE = PIN["source_tree"]
POLICY = {"geometry_uncertainty": "fixed_exact", "frame_authority": "declared_not_surveyed",
          "selection": "full_retained_batch", "covariance": "native_first_order_full_joint",
          "input_covariance_validation": "exact_symmetric_positive_semidefinite_before_retention",
          "cross_covariance": "full_matrix_declared", "physical_validation": "not_performed",
          "bim_mapping": "not_performed", "state_admission": "not_performed"}
MAX_SAMPLES = 16
SOURCE_LIMIT = 128 * 1024
REQUEST_LIMIT = 96 * 1024


def request_bytes(source):
    value = source["request_bytes_b64"]
    if not isinstance(value, str):
        raise ValueError("Require exact GTE request bytes as canonical base64")
    raw = base64.b64decode(value, validate=True)
    if not 1 <= len(raw) <= REQUEST_LIMIT or base64.b64encode(raw).decode() != value:
        raise ValueError("GTE request byte budget or canonical base64 violated")
    return raw


def request(source):
    return _parse(request_bytes(source))


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
        raise ValueError("Geometric source exceeds byte budget")
    source = _json(raw)
    canonical(source)
    exact_keys(source, {"schema", "experiment_id", "configuration", "request_bytes_b64"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(POLICY):
        raise ValueError("Require bounded geometric-circle source and explicit policy")
    _text(source["experiment_id"])
    observations = _observations(request(source))
    if len(observations["time_s"]) > MAX_SAMPLES:
        raise ValueError("Shared geometric-circle sample budget is 1..16")
    # Projection domain, circle validity and covariance applicability remain
    # native GTE decisions. Invalid native requests yield no scientific result.
    return source


def _check_data(source, data):
    run = _make_run(request_bytes(source))
    declared = request(source)
    obs, con, policy = declared["observations"], declared["constraint"], declared["policy"]
    exact_keys(obs, {"source_id", "observation_ids", "time_origin", "time_s", "points_m",
                "coordinate_frame", "quantity_order", "unit", "covariance"})
    count = len(obs["time_s"])
    _text(obs["source_id"])
    _text(obs["coordinate_frame"])
    epoch = datetime.fromisoformat(obs["time_origin"].replace("Z", "+00:00"))
    if epoch.tzinfo is None or epoch.utcoffset() is None or obs["unit"] != "m" or obs["quantity_order"] != ["x", "y"]:
        raise ValueError("GTE result requires declared metre coordinates and explicit time origin")
    ids = obs["observation_ids"]
    if not isinstance(ids, list) or len(ids) != count:
        raise ValueError("Require one GTE observation identity per sample")
    for identifier in ids:
        _text(identifier)
    if len(set(ids)) != count:
        raise ValueError("Duplicate GTE observation identity")
    covariance = obs["covariance"]
    exact_keys(covariance, {"matrix", "ordering", "unit", "meaning", "source"})
    if (covariance["ordering"] != "sample-major:x,y" or covariance["unit"] != "m^2" or
            covariance["meaning"] != "joint_observation_covariance"):
        raise ValueError("GTE input covariance semantics differ")
    _text(covariance["source"])
    # This shared profile accepts an exact symmetric PSD input domain. GTE's
    # wider roundoff-tolerant input domain remains available through its native
    # operation, but cannot silently average/repair this retained declaration.
    # Reuse the existing bounded rational-domain validator; no propagation or
    # projection mathematics is implemented in the workbench.
    from .calibrated_window import _covariance as exact_covariance
    exact_covariance(covariance["matrix"], 2 * count)
    _covariance(covariance["matrix"], 2 * count, "declared input covariance")
    exact_keys(con, {"constraint_id", "version", "kind", "metric", "coordinate_frame", "center_m",
                "radius_m", "valid_time_s", "geometry_uncertainty"})
    _text(con["constraint_id"])
    _text(con["version"])
    _array(con["center_m"], (2,), "circle center")
    _array(con["valid_time_s"], (2,), "constraint validity")
    start, end = con["valid_time_s"]
    if (con["kind"] != "circle" or con["metric"] != "euclidean" or con["coordinate_frame"] != obs["coordinate_frame"] or
            con["geometry_uncertainty"] != "fixed_exact" or number(con["radius_m"], "circle radius") <= 0 or
            not start < end or not all(start <= timestamp < end for timestamp in obs["time_s"]) or
            any(point == con["center_m"] for point in obs["points_m"])):
        raise ValueError("Native GTE result requires applicable fixed-exact geometry")
    exact_keys(policy, {"max_correction_m", "max_linearization_ratio"})
    if number(policy["max_correction_m"], "correction limit") < 0 or number(policy["max_linearization_ratio"], "linearization limit") <= 0:
        raise ValueError("Invalid GTE candidate policy")
    validate_payload(GTE_OPERATION, data, run, {},
                     {"interval_s": [0.0, run["metadata"]["duration_s"]]})


class GeometricCircleWorkflow(PipelineRunner):
    LABEL = "GTE"

    def __init__(self):
        super().__init__(KIND, PIN)

    def parse_source(self, raw):
        return _source(raw)

    def invoke(self, source, bound):
        return bound[0].invoke(GTE_OPERATION, request(source))

    def check_data(self, source, data):
        _check_data(source, data)


workflow = GeometricCircleWorkflow()
