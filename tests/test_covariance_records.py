"""Offline source binding and tamper gates for covariance-bearing result records."""

from copy import deepcopy

import pytest

from ciw.adapters.covariance_records import (
    validate_fsrt_payload, validate_jspt_payload, validate_result_dependencies,
)
from ciw.core.covariance import covariance_identity, create_covariance_artifact
from ciw.core.identities import content_identity
from ciw.operations.runner import seal


FSRT = "fsrt.tank-reconstruct.v2"
JSPT = "jspt.covariance-propagate.v1"
STATE = ["tank-1.mass", "tank-2.mass"]
SOURCE_RESULT = "result-" + "1" * 32


def _reseal(artifact):
    artifact["covariance_id"] = covariance_identity(artifact)


@pytest.fixture
def fsrt_record(monkeypatch):
    """A wire fixture with explicit scientific outputs; no estimator is executed."""
    import ciw.investigation as investigation

    evidence = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    sources = ["tank-1", "tank-2"]
    model = {"kind": "reservoir2-linear-v1", "prior_mean": [70.0, 30.0], "prior_std": 5.0,
             "total_mass_kg": 100.0, "total_mass_variance_kg2": 0.25}
    observation = {"t": 0, "arrival_t": 0, "values": [70.0, 30.0],
                   "covariance": [[0.4, 0.1], [0.1, 0.9]], "mask": [True, True],
                   "source_ids": sources, "evidence_ids": evidence, "unit": "kg"}
    independence = {"prior_independent_of_observations": True,
                    "declared_total_independent_of_observations": True,
                    "prior_independent_of_declared_total": True}
    upstream = {**independence, "shared_dependencies": ["shared-calibration-source"],
                "coverage": {"temperature": "excluded", "cross_sensor_correlation": "retained"}}
    observed_artifact = create_covariance_artifact(
        matrix=observation["covariance"], quantity_ids=sources, units=["kg", "kg"],
        frame="reservoir2.mass", reference_values=observation["values"], method="declared_measurement_covariance",
        basis={"kind": "calibrated_observation", "id": "observation-basis"},
        provenance={"provider": "fixture", "source_evidence_ids": evidence,
                    "source_covariance_ids": [], "metadata": upstream},
        assumptions=["Declared correlated observation errors"],
    )
    inputs = {"model": model, "observations": [observation], "state_order": STATE,
              "observation_covariance": observed_artifact}
    monkeypatch.setattr(investigation, "_fsrt_inputs", lambda run, params: deepcopy({
        "model": model, "observations": [observation]}))
    monkeypatch.setattr(investigation, "_fsrt_inputs_v2", lambda run, params: deepcopy(inputs))
    data = {
        "model": deepcopy(model), "calibrated_observation": deepcopy(observation),
        "estimate": {"kind": "estimated_state", "t": 0, "values": [70.0, 30.0],
                     "covariance": [[0.1, -0.05], [-0.05, 0.1]], "unit": "kg"},
        "unprojected_estimate": {"values": [70.0, 30.0], "covariance": [[0.3, 0.1], [0.1, 0.6]], "unit": "kg"},
        "residuals": {"innovation": [0.0, 0.0], "innovation_variance": [25.4, 25.9],
                      "balance_before": [0.0], "balance_after": [0.0], "correction": [0.0, 0.0], "unit": "kg"},
        "diagnostics": {"physical_model_status": "consistent", "reconciliation_status": "ok",
                        "fault_attribution": "not_tested", "consistency_statistic": 0.0,
                        "consistency_threshold": 10.0, "confidence": 0.999, "missing_sources": [],
                        "physical_truth_verified": False, "state_domain_status": "nonnegative_masses",
                        "scope": "Declared synthetic wire fixture; no numerical or physical verification"},
        "assumptions": {"prior_independent_of_observations": True,
                        "declared_total_independent_of_observations": True,
                        "topology": {"reservoirs": sources, "balance_coefficients": [1, 1]},
                        "temporal_covariance": "not_applicable_single_snapshot"},
        "observation_evidence_ids": evidence, "state_order": STATE,
    }
    metadata = {**independence, "shared_dependencies": upstream["shared_dependencies"],
                "state_order": STATE, "source_order": sources, "observed_mask": [True, True],
                "reference_role": "artifact_quantity_values; absent input coordinates are declared references, not observations",
                "upstream_covariance_metadata": deepcopy(upstream)}
    assumptions = observed_artifact["assumptions"] + [
        "Prior, observation errors and declared total are mutually independent except correlations inside the observation matrix.",
        "The prior is isotropic Gaussian with the declared prior_std; the total has its declared independent variance.",
        "One simultaneous snapshot, no temporal covariance model or physical verification.",
    ]

    def stage(name, matrix, refs, quantities, kind, covariance_ids, evidence_ids, method):
        stage_metadata = {**metadata, "stage": name}
        if name == "reconciled":
            stage_metadata["reconciliation_status"] = "ok"
        return create_covariance_artifact(
            matrix=matrix, quantity_ids=quantities, units=["kg"] * len(quantities), frame="reservoir2.mass",
            reference_values=refs, method=method, basis={"kind": kind, "id": FSRT + ":" + name},
            provenance={"provider": FSRT, "source_evidence_ids": evidence_ids,
                        "source_covariance_ids": covariance_ids, "metadata": stage_metadata},
            assumptions=assumptions,
        )

    prior = stage("prior", [[25.0, 0.0], [0.0, 25.0]], model["prior_mean"], STATE, "parameter", [], [],
                  "declared_isotropic_gaussian_prior")
    total = stage("declared_total", [[0.25]], [100.0], ["total_mass"], "parameter", [], [],
                  "declared_independent_total_mass")
    refs = [observed_artifact["covariance_id"], prior["covariance_id"]]
    innovation = stage("innovation", [[25.4, 0.1], [0.1, 25.9]], [0.0, 0.0], sources, "residual", refs,
                       evidence, "kalman_innovation_prior_plus_observation_covariance")
    posterior = stage("posterior", data["unprojected_estimate"]["covariance"], [70.0, 30.0], STATE,
                      "estimated_state", refs, evidence, "existing_kalman_joseph_update_before_balance")
    reconciled = stage("reconciled", data["estimate"]["covariance"], [70.0, 30.0], STATE, "estimated_state",
                       [posterior["covariance_id"], total["covariance_id"]], evidence,
                       "existing_balance_gate_and_reconciliation")
    data["covariance_artifacts"] = {"observation": observed_artifact, "prior": prior,
        "declared_total": total, "innovation": innovation, "posterior": posterior, "reconciled": reconciled}
    return data


@pytest.fixture
def jspt_record(fsrt_record):
    source = fsrt_record["covariance_artifacts"]["reconciled"]
    mapping = {"jacobian": [[0.5, 0.5], [1.0, -1.0]], "output_quantity_ids": ["mean_mass", "mass_difference"],
               "output_units": ["kg", "kg"], "output_frame": "mass-summary",
               "output_reference_values": [50.0, 40.0], "map_kind": "linear"}
    parameters = {**mapping, "source_result_id": SOURCE_RESULT, "source_artifact": "reconciled",
                  "source_covariance": deepcopy(source)}
    assumptions = source["assumptions"] + [
        "Jacobian columns follow input quantity_ids; rows follow output_quantity_ids.",
        "Jacobian coefficients and input/output reference values are caller declarations.",
        "Coefficient units are output_units[row]/input units[column]; units are declared, not inferred.",
        "The covariance push is exact for the declared fixed linear map in exact arithmetic.",
    ]
    output = create_covariance_artifact(
        matrix=[[0.025, 0.0], [0.0, 0.3]], quantity_ids=mapping["output_quantity_ids"], units=["kg", "kg"],
        frame="mass-summary", reference_values=[50.0, 40.0], method="linear_covariance_pushforward",
        basis={"kind": "coordinate", "id": content_identity({"operation_id": JSPT,
              "source_covariance_id": source["covariance_id"], **mapping})},
        provenance={"provider": "JSPT:" + JSPT, "source_evidence_ids": source["provenance"]["source_evidence_ids"],
                    "source_covariance_ids": [source["covariance_id"]], "metadata": {
                        "kernel": "sensitivity.covariance.first_order_covariance", "map_kind": "linear",
                        "jacobian_source": "caller_declared", "jacobian": mapping["jacobian"],
                        "input_quantity_ids": source["quantity_ids"], "input_units": source["units"],
                        "input_frame": source["frame"], "input_basis": source["basis"],
                        "linearization_point": source["reference_values"], "reference_values_source": "caller_declared",
                        "source_uncertainty_context": deepcopy(source["provenance"]["metadata"]),
                    }}, assumptions=assumptions,
    )
    data = {"schema": "jspt.covariance-result.v1", "operation_id": JSPT, "map_kind": "linear",
            "jacobian_source": "caller_declared", "jacobian": mapping["jacobian"], "input_covariance": deepcopy(source),
            "output_covariance": output, "linearization_point": source["reference_values"],
            "output_reference_values": [50.0, 40.0], "checks": {
                "input_positive_semidefinite": True, "output_positive_semidefinite": True,
                "input_covariance_symmetrized": False, "coordinate_chart_guarded": False,
                "jacobian_verified": False, "reference_values_verified": False,
                "physical_units_verified": False, "monte_carlo_performed": False,
            }, "limits": [
                "This operation propagates an explicit caller-declared Jacobian; it does not compute or verify derivatives.",
                "Reference values and physical unit/frame compatibility are caller declarations, not physical validation.",
                "No nonlinear Monte Carlo comparison, model adequacy check, certificate or execution authorization is produced.",
            ]}
    return data, parameters


def test_fsrt_covariance_schema_preserves_all_stages_without_executing_engine(fsrt_record, monkeypatch):
    import ciw.investigation as investigation
    monkeypatch.setattr(investigation, "_runtime", lambda *a, **k: pytest.fail("Offline validation invoked an engine"))
    before = deepcopy(fsrt_record)
    validate_fsrt_payload(FSRT, fsrt_record, {}, {}, {})
    assert fsrt_record == before


@pytest.mark.parametrize(("stage", "mutate"), [
    ("observation", lambda a: a["quantity_ids"].reverse()),
    ("prior", lambda a: a["matrix"][0].__setitem__(0, 26.0)),
    ("declared_total", lambda a: a["reference_values"].__setitem__(0, 90.0)),
    ("innovation", lambda a: a["matrix"][0].__setitem__(0, 26.4)),
    ("posterior", lambda a: a["matrix"][0].__setitem__(0, 0.4)),
    ("reconciled", lambda a: a["reference_values"].__setitem__(0, 71.0)),
    ("reconciled", lambda a: a["provenance"]["source_covariance_ids"].reverse()),
    ("reconciled", lambda a: a["provenance"]["metadata"].pop("upstream_covariance_metadata")),
    ("reconciled", lambda a: a["provenance"]["metadata"].__setitem__("prior_independent_of_observations", 1)),
])
def test_rehashed_fsrt_artifact_cannot_break_scientific_or_provenance_binding(fsrt_record, stage, mutate):
    retained = deepcopy(fsrt_record)
    artifact = retained["covariance_artifacts"][stage]
    mutate(artifact)
    _reseal(artifact)
    # An attacker may reseal both content levels; source relations still fail.
    outer = seal({"data": retained})
    with pytest.raises(ValueError):
        validate_fsrt_payload(FSRT, outer["data"], {}, {}, {})


def test_fsrt_innovation_cannot_drop_a_real_cross_covariance(fsrt_record):
    retained = deepcopy(fsrt_record)
    artifact = retained["covariance_artifacts"]["innovation"]
    artifact["matrix"][0][1] = artifact["matrix"][1][0] = 0.0
    _reseal(artifact)
    with pytest.raises(ValueError, match="retained scientific data"):
        validate_fsrt_payload(FSRT, retained, {}, {}, {})


def test_jspt_schema_binds_declared_map_and_retains_nested_uncertainty_context(jspt_record):
    data, parameters = jspt_record
    before = deepcopy(data)
    validate_jspt_payload(JSPT, data, {}, parameters, {})
    assert data == before
    assert data["output_covariance"]["provenance"]["metadata"]["source_uncertainty_context"][
        "upstream_covariance_metadata"]["coverage"]["temperature"] == "excluded"


@pytest.mark.parametrize("mutate", [
    lambda a: a["quantity_ids"].reverse(),
    lambda a: a["units"].__setitem__(0, "g"),
    lambda a: a.__setitem__("frame", "different-frame"),
    lambda a: a["reference_values"].__setitem__(0, 51.0),
    lambda a: a["basis"].__setitem__("id", "different-basis"),
    lambda a: a.__setitem__("method", "verified_derivative"),
    lambda a: a["provenance"]["source_covariance_ids"].__setitem__(0, "sha256:" + "f" * 64),
    lambda a: a["provenance"]["metadata"].pop("source_uncertainty_context"),
    lambda a: a["provenance"]["metadata"]["source_uncertainty_context"]["upstream_covariance_metadata"][
        "coverage"].__setitem__("temperature", "included"),
])
def test_rehashed_jspt_output_cannot_rewrite_map_or_source_semantics(jspt_record, mutate):
    data, parameters = deepcopy(jspt_record)
    mutate(data["output_covariance"])
    _reseal(data["output_covariance"])
    with pytest.raises(ValueError):
        validate_jspt_payload(JSPT, seal({"data": data})["data"], {}, parameters, {})


@pytest.mark.parametrize(("field", "value"), [
    ("jacobian_verified", True), ("physical_units_verified", True), ("monte_carlo_performed", True),
    ("input_covariance_symmetrized", True), ("input_positive_semidefinite", 1),
])
def test_jspt_checks_cannot_promote_declarations_or_hide_numerical_changes(jspt_record, field, value):
    data, parameters = deepcopy(jspt_record)
    data["checks"][field] = value
    with pytest.raises(ValueError, match="verification"):
        validate_jspt_payload(JSPT, data, {}, parameters, {})


def _results(fsrt_data, jspt_data, parameters):
    result_id = "result-" + "2" * 32
    return {
        SOURCE_RESULT: {"result_id": SOURCE_RESULT, "operation_id": FSRT, "data": deepcopy(fsrt_data)},
        result_id: {"result_id": result_id, "operation_id": JSPT, "data": deepcopy(jspt_data),
                    "parameters": deepcopy(parameters)},
    }, result_id


def test_dependencies_require_retained_result_and_exact_named_covariance(fsrt_record, jspt_record):
    data, parameters = jspt_record
    results, identity = _results(fsrt_record, data, parameters)
    validate_result_dependencies(results)
    for mutate in (
        lambda rs: rs.pop(SOURCE_RESULT),
        lambda rs: rs[identity]["parameters"].__setitem__("source_result_id", identity),
        lambda rs: rs[identity]["parameters"].__setitem__("source_artifact", "posterior"),
    ):
        changed = deepcopy(results)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_result_dependencies(changed)


def test_dependency_cycles_refuse_even_when_each_selected_artifact_matches(jspt_record):
    data, parameters = jspt_record
    left, right = "result-" + "2" * 32, "result-" + "3" * 32
    common_artifact = data["output_covariance"]
    results = {
        identity: {"result_id": identity, "operation_id": JSPT,
                   "data": {"output_covariance": common_artifact},
                   "parameters": {**parameters, "source_result_id": other,
                                  "source_artifact": "output_covariance", "source_covariance": common_artifact}}
        for identity, other in ((left, right), (right, left))
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_result_dependencies(results)
