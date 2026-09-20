"""Offline GTE result schema and source bindings; no projection or propagation."""
from __future__ import annotations

import numpy as np

from ..core.records import mapping, number
from ..core.identities import digest
from .protocol import AdapterRefusal


def _fields(value, fields, name):
    mapping(value, name)
    if set(value) != set(fields.split()):
        raise ValueError(f"Invalid GTE {name} fields")


def _array(value, shape, name):
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"GTE {name} must have shape {shape}")
    for item in value:
        if len(shape) > 1:
            _array(item, shape[1:], name)
        else:
            number(item, name)


def _covariance(value, size, name):
    _array(value, (size, size), name)
    # Covariance domain validation is structural; neither Jacobians nor
    # transformed values are recalculated by this offline reader.
    matrix = np.array(value, dtype=float)
    variances = np.diag(matrix)
    invalid = f"GTE {name} must be symmetric positive semidefinite"
    if np.any(variances < 0):
        raise ValueError(invalid)
    zero = variances == 0
    if np.any(matrix[zero, :] != 0) or np.any(matrix[:, zero] != 0):
        raise ValueError(invalid)
    active = ~zero
    if np.any(active):
        standard_deviations = np.sqrt(variances[active])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            normalized = (matrix[np.ix_(active, active)] / standard_deviations[:, None]
                          / standard_deviations[None, :])
        if (not np.all(np.isfinite(normalized))
                or np.max(np.abs(normalized-normalized.T)) > 1e-10
                or np.linalg.eigvalsh(0.5*normalized + 0.5*normalized.T)[0] < -1e-10):
            raise ValueError(invalid)


def validate_payload(operation_id, data, run, parameters, selection):
    from ..geodesic import _inputs

    request = _inputs(run, parameters)
    obs = request["observations"]
    count = len(obs["time_s"])
    start, end = selection["interval_s"]
    if not all(start <= timestamp < end for timestamp in obs["time_s"]):
        raise AdapterRefusal("selection_scope", "GTE requires a selection containing the full retained batch")
    _fields(data, "schema method source_id source_observation_ids time_s time_origin coordinate_frame "
            "observed_points_m projected_points_m reconciled_points_m constraint policy diagnostics "
            "uncertainty geodesic_steps reconciliation self_checks", "result")
    if (data["schema"] != "gte.circle-result.v1"
            or data["method"] != "euclidean-nearest-circle-projection-v1"):
        raise ValueError("Unknown GTE result schema or method")
    bindings = {"source_id": obs["source_id"], "source_observation_ids": obs["observation_ids"], "time_s": obs["time_s"],
                "time_origin": obs["time_origin"], "coordinate_frame": obs["coordinate_frame"],
                "observed_points_m": obs["points_m"], "constraint": request["constraint"],
                "policy": request["policy"]}
    if any(digest(data[key]) != digest(value) for key, value in bindings.items()):
        raise ValueError("GTE result source bindings differ from retained request")
    _array(data["projected_points_m"], (count, 2), "projected points")
    diagnostics = data["diagnostics"]
    _fields(diagnostics, "radial_residual_before_m radial_residual_after_m correction_vectors_m "
            "correction_norm_m jacobians linearization_ratio", "diagnostics")
    for key in ("radial_residual_before_m", "radial_residual_after_m", "correction_norm_m", "linearization_ratio"):
        _array(diagnostics[key], (count,), key)
    for key in ("correction_norm_m", "linearization_ratio"):
        if any(value < 0 for value in diagnostics[key]):
            raise ValueError("GTE correction norms and linearization ratios must be nonnegative")
    _array(diagnostics["correction_vectors_m"], (count, 2), "correction vectors")
    _array(diagnostics["jacobians"], (count, 2, 2), "Jacobians")
    uncertainty = data["uncertainty"]
    _fields(uncertainty, "method input_joint_covariance tangent_joint_covariance ambient_joint_covariance "
            "tangent_bases reference_points_m ordering ambient_ordering unit coordinate_frame source "
            "geometry_uncertainty limitations", "uncertainty")
    expected_uncertainty = {
        "method": "first-order-deterministic-projection",
        "input_joint_covariance": obs["covariance"]["matrix"],
        "reference_points_m": data["projected_points_m"],
        "ordering": "sample-major:arc_length", "ambient_ordering": "sample-major:x,y",
        "unit": "m^2", "coordinate_frame": obs["coordinate_frame"],
        "source": obs["covariance"]["source"], "geometry_uncertainty": "fixed_exact",
        "limitations": ["first_order_local_only", "not_bayesian_posterior", "geometry_treated_as_exact",
                        "ambient_covariance_rank_deficient", "no_physical_accuracy_claim"],
    }
    if any(digest(uncertainty[key]) != digest(value) for key, value in expected_uncertainty.items()):
        raise ValueError("GTE uncertainty binding, ordering or claim scope changed")
    for key, size in (("input_joint_covariance", count*2), ("tangent_joint_covariance", count),
                      ("ambient_joint_covariance", count*2)):
        _covariance(uncertainty[key], size, key)
    _array(uncertainty["tangent_bases"], (count, 2), "tangent bases")
    steps = data["geodesic_steps"]
    _fields(steps, "signed_arc_m ambiguous meaning", "geodesic steps")
    if (steps["meaning"] != "shortest_arcs_not_physical_trajectory"
            or not isinstance(steps["ambiguous"], list) or len(steps["ambiguous"]) != count-1
            or not isinstance(steps["signed_arc_m"], list) or len(steps["signed_arc_m"]) != count-1):
        raise ValueError("Invalid GTE arc declarations")
    for arc, ambiguous in zip(steps["signed_arc_m"], steps["ambiguous"]):
        if type(ambiguous) is not bool or (arc is None) != ambiguous:
            raise ValueError("GTE ambiguous arc must be explicitly null")
        if arc is not None:
            number(arc, "signed arc")
    reconciliation = data["reconciliation"]
    _fields(reconciliation, "status reasons scope", "reconciliation")
    reasons = []
    if any(value > request["policy"]["max_correction_m"] for value in diagnostics["correction_norm_m"]):
        reasons.append("correction_limit_exceeded")
    if any(value > request["policy"]["max_linearization_ratio"] for value in diagnostics["linearization_ratio"]):
        reasons.append("linearization_limit_exceeded")
    if reconciliation != {"status": "held" if reasons else "eligible", "reasons": reasons,
                          "scope": "declared_geometric_policy_only"}:
        raise ValueError("GTE reconciliation status contradicts retained policy diagnostics")
    if digest(data["reconciled_points_m"]) != digest(None if reasons else data["projected_points_m"]):
        raise ValueError("GTE reconciled coordinates contradict the reconciliation status")
    checks = data["self_checks"]
    _fields(checks, "max_circle_residual_m max_tangent_covariance_asymmetry_m2 interpretation", "self checks")
    if checks["interpretation"] != "numerical_diagnostics_not_independent_verification":
        raise ValueError("GTE self checks cannot confer independent verification")
    for key in ("max_circle_residual_m", "max_tangent_covariance_asymmetry_m2"):
        if number(checks[key], key) < 0:
            raise ValueError("GTE maximum residual diagnostics must be nonnegative")
