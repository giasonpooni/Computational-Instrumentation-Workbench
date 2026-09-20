"""Saved FSRT payload contract: shape and source relations, never estimation."""
from __future__ import annotations

import math

from ..core.records import mapping, number


def _vector(value, size, name, *, missing=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")
    for item in value:
        if item is None and missing:
            continue
        number(item, name)


def _covariance(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a 2 by 2 covariance")
    for row in value:
        _vector(row, 2, name)
    a, b = value[0]
    c, d = value[1]
    if a < 0 or d < 0 or not math.isclose(b, c, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{name} must be symmetric with nonnegative variances")
    # Validate the covariance domain without running the estimator. The scaled
    # correlation check avoids overflow for finite but very large variances.
    if a == 0 or d == 0:
        if b != 0 or c != 0:
            raise ValueError(f"{name} has cross-covariance with zero variance")
    elif abs((b / math.sqrt(a)) / math.sqrt(d)) > 1.0 + 1e-10:
        raise ValueError(f"{name} is not positive semidefinite")


def validate_payload(operation_id, data, run, parameters, selection):
    from ..investigation import _fsrt_inputs

    expected = _fsrt_inputs(run, parameters)
    required = {"model", "calibrated_observation", "estimate", "unprojected_estimate",
                "residuals", "diagnostics", "assumptions", "observation_evidence_ids"}
    if set(data) != required:
        raise ValueError("Invalid saved FSRT data fields")
    if data["model"] != expected["model"] or data["calibrated_observation"] != expected["observations"][0]:
        raise ValueError("FSRT inputs do not match retained evidence and declared parameters")
    model = mapping(data["model"], "model")
    if set(model) != {"kind", "prior_mean", "prior_std", "total_mass_kg", "total_mass_variance_kg2"} or model["kind"] != "reservoir2-linear-v1":
        raise ValueError("Invalid FSRT model declaration")
    _vector(model["prior_mean"], 2, "prior_mean")
    if (any(value < 0 for value in model["prior_mean"])
            or number(model["prior_std"], "prior_std") <= 0
            or number(model["total_mass_kg"], "total_mass_kg") < 0
            or number(model["total_mass_variance_kg2"], "total_mass_variance_kg2") < 0):
        raise ValueError("Invalid FSRT model bounds")
    observation = expected["observations"][0]
    if data["observation_evidence_ids"] != observation["evidence_ids"]:
        raise ValueError("FSRT observation evidence identities do not match")
    for name in ("estimate", "unprojected_estimate"):
        state = mapping(data[name], name)
        fields = {"values", "covariance", "unit"} | ({"kind", "t"} if name == "estimate" else set())
        if set(state) != fields or state["unit"] != "kg":
            raise ValueError("Invalid FSRT state descriptor")
        _vector(state["values"], 2, name)
        if any(value < 0 for value in state["values"]):
            raise ValueError("FSRT successful states must remain in the nonnegative mass domain")
        _covariance(state["covariance"], name)
    if data["estimate"]["kind"] != "estimated_state" or number(data["estimate"]["t"], "estimate time") != observation["t"]:
        raise ValueError("FSRT estimate kind or time does not match its observation")
    residual = mapping(data["residuals"], "residuals")
    if set(residual) != {"innovation", "innovation_variance", "balance_before", "balance_after", "correction", "unit"} or residual["unit"] != "kg":
        raise ValueError("Invalid FSRT residual descriptor")
    for name in ("innovation", "innovation_variance"):
        _vector(residual[name], 2, name, missing=True)
        for value, present in zip(residual[name], observation["mask"]):
            if (value is None) == present:
                raise ValueError("FSRT innovation missingness differs from the observation")
    if any(value is not None and value <= 0 for value in residual["innovation_variance"]):
        raise ValueError("FSRT innovation variances must be positive when observed")
    _vector(residual["balance_before"], 1, "balance_before")
    diagnostics = mapping(data["diagnostics"], "diagnostics")
    if set(diagnostics) != {"physical_model_status", "reconciliation_status", "fault_attribution",
                           "consistency_statistic", "consistency_threshold", "confidence",
                           "missing_sources", "physical_truth_verified", "state_domain_status", "scope"}:
        raise ValueError("Invalid FSRT diagnostic fields")
    score = number(diagnostics["consistency_statistic"], "consistency_statistic")
    threshold = number(diagnostics["consistency_threshold"], "consistency_threshold")
    if score < 0 or threshold <= 0 or diagnostics["confidence"] != 0.999:
        raise ValueError("Invalid FSRT diagnostic bounds")
    disagreement = score > threshold
    expected_status = "physical_model_disagreement" if disagreement else "consistent"
    if diagnostics["physical_model_status"] != expected_status:
        raise ValueError("FSRT disagreement status contradicts its diagnostic score")
    if diagnostics["reconciliation_status"] != ("model_inconsistent" if disagreement else "ok"):
        raise ValueError("FSRT reconciliation status contradicts its diagnostic score")
    if diagnostics["fault_attribution"] != ("confounded_or_unidentifiable" if disagreement or not all(observation["mask"]) else "not_tested"):
        raise ValueError("FSRT fault attribution exceeds its declared model")
    if diagnostics["physical_truth_verified"] is not False or diagnostics["state_domain_status"] != "nonnegative_masses":
        raise ValueError("FSRT cannot confer physical verification")
    if diagnostics["missing_sources"] != [name for name, present in zip(observation["source_ids"], observation["mask"]) if not present]:
        raise ValueError("FSRT missing source declaration is inconsistent")
    if not isinstance(diagnostics["scope"], str) or not diagnostics["scope"]:
        raise ValueError("FSRT diagnostic scope must be explicit")
    if disagreement:
        if residual["balance_after"] is not None or residual["correction"] is not None:
            raise ValueError("Held reconciliation cannot carry a correction")
        for key in ("values", "covariance"):
            if data["estimate"][key] != data["unprojected_estimate"][key]:
                raise ValueError("Held reconciliation must retain the unprojected state")
    else:
        _vector(residual["balance_after"], 1, "balance_after")
        _vector(residual["correction"], 2, "correction")
    assumptions = {"prior_independent_of_observations": True,
                   "declared_total_independent_of_observations": True,
                   "topology": {"reservoirs": observation["source_ids"], "balance_coefficients": [1, 1]},
                   "temporal_covariance": "not_applicable_single_snapshot"}
    if data["assumptions"] != assumptions:
        raise ValueError("FSRT assumptions do not match the single-snapshot contract")
