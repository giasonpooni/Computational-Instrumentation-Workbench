"""Scientific and identity-boundary checks for the pinned identified-design path."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw import calibrated_observable as calibrated
from ciw import identified_design as design
from ciw.core.canonical import bundle_digest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/identified-design/source.json").read_bytes()
UPSTREAM_SOURCE = (ROOT / "examples/calibrated-observable/source.json").read_bytes()


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_IDENTIFIED_DESIGN_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_IDENTIFIED_DESIGN_STACK_ROOT to the eleven pinned role checkouts")
    return {role: Path(root) / role for role in design.ROLES}


@pytest.fixture(scope="module")
def upstream(repositories):
    return calibrated.create_session(UPSTREAM_SOURCE,
                                     {role: repositories[role] for role in calibrated.ROLES})


@pytest.fixture(scope="module")
def bundle(upstream, repositories):
    return design.create_session(SOURCE, upstream, repositories)


def altered_source(mutate):
    source = json.loads(SOURCE)
    mutate(source)
    return calibrated.canonical(source)


def step_data(bundle, role):
    step, = [step for step in bundle["steps"] if step["runtime_ref"] == role]
    return step["result"].get("data", step["result"])


def test_all_eleven_owner_pins_extend_the_calibrated_graph():
    assert design.ROLES == calibrated.ROLES | {"sidt", "edspt", "ywir"}
    assert [role for role, _ in design.OPERATIONS] == ["sidt", "oit", "gsie", "edspt", "ywir"]


def test_source_identity_requires_exact_bytes_without_duplicate_fields():
    with pytest.raises(ValueError, match="exact bytes"):
        design._source(SOURCE.decode(), None)
    with pytest.raises(AdapterRefusal) as caught:
        design._source(b'{"schema":"first","schema":"second"}', None)
    assert caught.value.code == "MALFORMED_RESPONSE"


def test_completed_session_retains_upstream_and_read_only_inspection(bundle, upstream):
    assert bundle["schema"] == "ciw.identified-design-session.v1"
    assert bundle["upstream"] == upstream
    assert bundle["verification"]["outcome"] == "passed"
    assert bundle["verification"]["independent"] is False
    assert set(bundle["runtimes"]) == design.ROLES
    assert [(step["runtime_ref"], step["operation_id"]) for step in bundle["steps"]] == list(design.OPERATIONS)
    inspected = design.inspect_session(bundle)
    assert inspected["status"] == "content_consistent"
    assert inspected["numerical_replay"] == "not_performed"
    assert inspected["decision"]["state_admission"] == "not_performed"
    assert inspected["decision"]["acquisition"] == "not_performed"
    assert step_data(bundle, "ywir")["admitted"] is True
    assert step_data(bundle, "ywir")["advisory_token_cap"] == 16


def test_identified_conservation_dynamics_retain_unknown_parameter_uncertainty(bundle):
    result = step_data(bundle, "sidt")["numerical_result"]
    assert result["status"] == "identified"
    np.testing.assert_allclose(result["candidate"]["A"], [[.9, .1], [.1, .9]], atol=1e-13)
    np.testing.assert_allclose(result["candidate"]["B"], [[.05], [-.05]], atol=1e-13)
    assert result["diagnostics"]["rank"] == 3
    assert result["parameter_covariance"]["status"] == "unknown"
    assert result["parameter_covariance"]["matrix"] is None
    assert bundle["steps"][0]["result"]["covariance"]["status"] == "unknown"
    assert bundle["decision"]["uncertainty_scope"] == "conditional_on_identified_point_model"


def test_prediction_uses_exact_retained_gsie_prior_and_declared_future_step(bundle, upstream):
    prior = step_data(upstream, "gsie")
    predicted = step_data(bundle, "gsie")
    model = step_data(bundle, "sidt")["numerical_result"]["candidate"]
    transition = np.asarray(model["A"])
    process = np.asarray(json.loads(SOURCE)["prediction"]["process_covariance"])
    np.testing.assert_allclose(predicted["mean"], transition @ prior["mean"], atol=1e-13)
    np.testing.assert_allclose(predicted["covariance"],
                               transition @ prior["covariance"] @ transition.T + process,
                               atol=1e-14)
    assert predicted["time"] == prior["time"] + 1
    assert predicted["frame_id"] == prior["frame_id"]
    assert predicted["units"] == prior["units"]
    assert predicted["predecessor_state_id"] == prior["state_id"]
    assert predicted["model_result_id"] == bundle["steps"][0]["numerical_result_id"]
    assert predicted["parameter_covariance_status"] == "unknown"
    # The accepted conservation posterior is singular; it was not substituted.
    assert bundle["steps"][2]["request"]["inputs"]["prior"]["covariance"] == prior["covariance"]


def assert_candidate_reductions(bundle):
    declaration = bundle["configuration"]["design"]
    predicted = np.asarray(step_data(bundle, "gsie")["covariance"])
    scale = np.asarray(declaration["state_scales"])
    prior = predicted / scale[:, None] / scale[None, :]
    ranked = step_data(bundle, "edspt")
    candidates = {candidate["candidate_id"]: candidate for candidate in declaration["candidates"]}
    np.testing.assert_allclose(ranked["prior_covariance"], prior, atol=1e-14)
    assert ranked["prior_a_opt_trace_covariance"] == pytest.approx(np.trace(prior))
    for score in ranked["scores"]:
        candidate = candidates[score["candidate_id"]]
        observation = np.asarray(candidate["observation_matrix"]) * scale[None, :]
        noise = np.asarray(candidate["noise_covariance"])
        innovation = observation @ prior @ observation.T + noise
        gain = np.linalg.solve(innovation, observation @ prior).T
        remainder = np.eye(2) - gain @ observation
        posterior = remainder @ prior @ remainder.T + gain @ noise @ gain.T
        np.testing.assert_allclose(np.linalg.inv(score["numerical_score"]["posterior_precision"]),
                                   posterior, atol=1e-13)
        reduction = np.trace(prior) - np.trace(posterior)
        assert score["a_opt_trace_reduction"] == pytest.approx(reduction)
        logdet_gain = np.linalg.slogdet(prior)[1] - np.linalg.slogdet(posterior)[1]
        assert score["d_opt_logdet_gain"] == pytest.approx(logdet_gain)
        assert score["expected_uncertainty_reduction"] == pytest.approx(
            reduction if declaration["criterion"] == "a_opt" else logdet_gain)
        assert score["model_result_id"] == bundle["steps"][0]["numerical_result_id"]


def test_candidate_reductions_equal_independent_joseph_covariance_calculation(bundle):
    assert_candidate_reductions(bundle)
    ranked = step_data(bundle, "edspt")
    assert ranked["selected_candidate_id"] == "sensor:tank-2:precise"


def test_observability_and_budget_exclusions_remain_visible(bundle):
    observability = step_data(bundle, "oit")
    rows = {row["candidate_id"]: row for row in observability["candidates"]}
    assert rows["sensor:sum-only"]["status"] == "unobservable"
    assert rows["sensor:sum-only"]["rank"] == 1
    for name in ("sensor:tank-1:standard", "sensor:tank-2:precise", "sensor:both:premium"):
        assert rows[name]["status"] == "observable"
    np.testing.assert_allclose(observability["transition"], [[.9, .1], [.1, .9]], atol=1e-13)
    ranked = step_data(bundle, "edspt")
    scores = {row["candidate_id"]: row for row in ranked["scores"]}
    assert "sensor:sum-only" not in scores
    assert scores["sensor:both:premium"]["eligibility"] == "over_budget"
    assert scores["sensor:both:premium"]["expected_uncertainty_reduction"] > scores["sensor:tank-2:precise"]["expected_uncertainty_reduction"]
    assert ranked["ranked_candidate_ids"] == ["sensor:tank-2:precise", "sensor:tank-1:standard"]
    assert ranked["advisory_only"] is True


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["identification"].update(state_names=["tank-2.mass", "tank-1.mass"]), "coordinates"),
    (lambda s: s["identification"].update(state_units=["g", "g"]), "coordinates"),
    (lambda s: s["identification"].update(state_frame="different-frame"), "coordinates"),
    (lambda s: s["identification"].update(clock_frame="different-clock"), "reference clock"),
    (lambda s: s["identification"]["training"]["sample_times"].__setitem__(-1, 6), "after"),
    (lambda s: s["prediction"].update(prior_source="cbsr_reconciled"), "retained GSIE"),
    (lambda s: s["prediction"].update(uncertainty_scope="total_predictive_uncertainty"), "conditional"),
    (lambda s: s["prediction"].update(prior_process_crosscov_policy="unknown"), "prior-process"),
    (lambda s: s["prediction"].update(process_covariance_evidence_refs=[]), "evidence"),
    (lambda s: s["prediction"].update(process_covariance_evidence_refs=[s["identification"]["model_id"]]), "identities"),
    (lambda s: s["prediction"].update(next_input=[1]), "zero input"),
    (lambda s: s["design"].update(budget=True), "budget"),
    (lambda s: s["design"].update(horizon=10**9), "horizon"),
    (lambda s: s["design"].update(cost_unit="inference_token"), "separate"),
    (lambda s: s["design"].update(state_scales=[1, 0]), "positive"),
    (lambda s: s["design"]["candidates"][0].update(prior_cross_covariance_policy="unknown"), "cross-covariance"),
    (lambda s: s["design"]["candidates"][0].update(cost_unit="USD"), "cost unit"),
    (lambda s: s["design"]["candidates"][0].update(cost=-1), "cost"),
    (lambda s: s["design"]["candidates"][0].update(observation_matrix=[[True, 0]]), "numeric"),
    (lambda s: s["design"]["candidates"][0].update(evidence_refs=[]), "evidence"),
    (lambda s: s["design"]["candidates"].append(deepcopy(s["design"]["candidates"][0])), "unique"),
    (lambda s: s["token_admission"].update(budget_unit="measurement_credit"), "inference-token"),
])
def test_unsupported_scientific_or_budget_claims_refuse_before_execution(upstream, mutate, match):
    with pytest.raises(ValueError, match=match):
        design._source(altered_source(mutate), upstream)


def test_lower_observation_budget_changes_feasible_choice(upstream, repositories):
    changed = design.create_session(altered_source(lambda s: s["design"].update(budget=2)),
                                    upstream, repositories)
    assert changed["decision"]["selected_candidate_id"] == "sensor:tank-1:standard"
    assert step_data(changed, "edspt")["ranked_candidate_ids"] == ["sensor:tank-1:standard"]


def test_scaled_coordinate_d_optimal_design_preserves_physical_information(upstream, repositories):
    changed = design.create_session(altered_source(
        lambda s: s["design"].update(state_scales=[2, 3], criterion="d_opt")), upstream, repositories)
    assert_candidate_reductions(changed)
    assert step_data(changed, "edspt")["coordinates"] == [
        {"name": "tank-1.mass", "scale": 2.0, "unit": "kg"},
        {"name": "tank-2.mass", "scale": 3.0, "unit": "kg"},
    ]


def test_ill_conditioned_candidates_are_excluded_while_informative_alternative_survives(upstream, repositories):
    changed = design.create_session(altered_source(
        lambda s: s["design"].update(condition_limit=2, budget=10)), upstream, repositories)
    rows = {row["candidate_id"]: row for row in step_data(changed, "oit")["candidates"]}
    assert rows["sensor:tank-1:standard"]["status"] == "ill_conditioned"
    assert rows["sensor:tank-2:precise"]["status"] == "ill_conditioned"
    assert rows["sensor:both:premium"]["status"] == "observable"
    assert changed["decision"]["selected_candidate_id"] == "sensor:both:premium"


def test_token_denial_keeps_the_scientific_ranking(bundle, upstream, repositories):
    changed = design.create_session(altered_source(lambda s: s["token_admission"].update(token_budget=0)),
                                    upstream, repositories)
    assert changed["steps"][3]["numerical_result_id"] == bundle["steps"][3]["numerical_result_id"]
    assert changed["decision"]["selected_candidate_id"] == "sensor:tank-2:precise"
    assert changed["decision"]["token_admitted"] is False
    gate = step_data(changed, "ywir")
    assert gate["support_code"] == "BUDGET_CLOSED"
    assert gate["advisory_token_cap"] == 0
    assert gate["authority_scope"] == "advisory_only"
    assert "reservation_id" not in gate


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s["identification"]["training"].update(states=[[50, 50]] * 9, inputs=[[0]] * 8), "DESIGN_MODEL_NONIDENTIFIABLE"),
    (lambda s: s["design"].update(budget=0), "DESIGN_NO_AFFORDABLE_CANDIDATE"),
    (lambda s: s["design"].update(condition_limit=None), "DESIGN_NO_OBSERVABLE_CANDIDATE"),
])
def test_uninformative_or_unaffordable_workloads_refuse(upstream, repositories, mutate, code):
    with pytest.raises(AdapterRefusal) as caught:
        design.create_session(altered_source(mutate), upstream, repositories)
    assert caught.value.code == code


@pytest.mark.parametrize("mutate", [
    lambda b: b.update(session_id=b["upstream"]["session_id"]),
    lambda b: b.update(session_id=b["upstream_replay"]["session"]["session_id"]),
    lambda b: b["configuration"]["design"].update(budget=1000),
    lambda b: b["configuration"]["design"]["state_scales"].__setitem__(0, True),
    lambda b: b["steps"][2]["request"]["inputs"]["dynamics"].update(matrix=[[1, 0], [0, 1]]),
    lambda b: b["steps"][3]["request"]["inputs"].update(model_result_id="substituted-model"),
    lambda b: b["steps"][3]["result"]["data"].update(selected_candidate_id="sensor:both:premium"),
    lambda b: b["steps"][4].update(execution_id=b["steps"][3]["execution_id"]),
    lambda b: b["decision"].update(acquisition="authorized"),
    lambda b: b["upstream_replay"]["replay_receipt"].update(source_bundle_digest="unbound-source"),
])
def test_outer_rehash_cannot_hide_broken_model_operation_or_authority_binding(bundle, mutate):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    mutate(changed)
    changed["bundle_digest"] = bundle_digest(changed)
    with pytest.raises(ValueError):
        design.inspect_session(changed)


@pytest.mark.parametrize("field,value", [
    ("selected_candidate_id", "sensor:both:premium"),
    ("selection_content_id", "substituted-selection"),
    ("authority_scope", "physical_acquisition"),
    ("support_code", "UNBOUNDED_AUTHORITY"),
])
def test_fully_rehashed_token_claims_still_bind_selection_and_authority(bundle, field, value):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    step = changed["steps"][-1]
    data = step["result"]["data"]
    data[field] = value
    data.pop("result_content_id")
    namespace = "ywir:token-admission"
    data["result_content_id"] = namespace + ":" + sha256(namespace.encode() + b"\0" + calibrated.canonical(data)).hexdigest()
    step["result"].pop("result_id")
    design._seal_artifact(step["result"], "result_id")
    step["result_id"] = step["result"]["result_id"]
    step["result_sha256"] = calibrated.digest(step["result"])
    step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(data)}
    step["numerical_result_id"] = calibrated.digest(step["numerical_result"])
    changed["bundle_digest"] = bundle_digest(changed)
    with pytest.raises(ValueError, match="semantic binding"):
        design.inspect_session(changed)


def test_fully_rehashed_advisory_forgery_fails_pinned_reexecution(bundle, repositories):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    step = changed["steps"][-1]
    data = step["result"]["data"]
    data["details"] = "Forged advisory explanation"
    data.pop("result_content_id")
    namespace = "ywir:token-admission"
    data["result_content_id"] = namespace + ":" + sha256(namespace.encode() + b"\0" + calibrated.canonical(data)).hexdigest()
    result = step["result"]
    result.pop("result_id")
    design._seal_artifact(result, "result_id")
    step["result_id"] = result["result_id"]
    step["result_sha256"] = calibrated.digest(result)
    step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(data)}
    step["numerical_result_id"] = calibrated.digest(step["numerical_result"])
    changed["bundle_digest"] = bundle_digest(changed)
    assert design.inspect_session(changed)["status"] == "content_consistent"
    with pytest.raises(ValueError, match="pinned recomputation"):
        design.replay_session(changed, repositories)


def test_original_replay_occurrences_are_distinct_but_numerics_match(bundle, repositories, tmp_path):
    replay = design.replay_session(bundle, repositories)
    fresh = replay["session"]
    assert replay["replay_receipt"]["numerical_match"] is True
    assert fresh["session_id"] != bundle["session_id"]
    for before, after in zip(bundle["steps"], fresh["steps"]):
        assert before["numerical_result_id"] == after["numerical_result_id"]
        assert before["execution_id"] != after["execution_id"]
        assert before["result_id"] != after["result_id"]
    path = design.save_session(fresh, tmp_path)
    assert design.read_session(path) == fresh
    with pytest.raises(ValueError, match="overwrite"):
        design.save_session(fresh, tmp_path)


def test_inspection_cli_does_not_reexecute_or_admit(bundle, tmp_path):
    path = design.save_session(bundle, tmp_path)
    run = subprocess.run([sys.executable, "-m", "ciw", "identified-design", "inspect", str(path)],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    assert result["status"] == "content_consistent"
    assert result["numerical_replay"] == "not_performed"
    assert result["decision"]["state_admission"] == "not_performed"
