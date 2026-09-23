"""Analytic and adversarial checks for the pinned two-channel process path."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.calibrated_observable import (
    OPERATIONS, ROLES, _numerical, _source, canonical, create_session, digest,
    inspect_session, read_session, replay_session, save_session,
)
from ciw.telemetry import _bundle_digest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/calibrated-observable/source.json").read_bytes()


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_CALIBRATED_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_CALIBRATED_STACK_ROOT to the eight pinned role checkouts")
    return {role: Path(root) / role for role in ROLES}


@pytest.fixture(scope="module")
def bundle(repositories):
    return create_session(SOURCE, repositories)


def altered_source(mutate):
    value = json.loads(SOURCE)
    mutate(value)
    return canonical(value)


def test_source_preserves_raw_values_timestamps_and_covariance():
    source = _source(SOURCE)
    assert [c["observation"]["device_time"] for c in source["channels"]] == [1005, 2004]
    assert [c["observation"]["raw_value"] for c in source["channels"]] == [51000, 45000]
    assert source["calibrated_covariance"]["matrix"] == [[1, .25], [.25, 1]]


@pytest.mark.parametrize("section,field,value", [
    ("observability", "model_id", "substituted-model"),
    ("observability", "transition", [[1, 1], [0, 1]]),
    ("observability", "observation", [[1, 0], [0, .5]]),
])
def test_observability_must_assess_exact_estimator_model(section, field, value):
    raw = altered_source(lambda s: s["configuration"][section].update({field: value}))
    with pytest.raises(ValueError, match="exact estimator model"):
        _source(raw)


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["configuration"]["gsie"].update(prior_measurement_crosscov_policy="unknown"), "independence"),
    (lambda s: s["configuration"]["gsie"]["dynamics"].update(process_covariance=[[1, 0], [0, 1]]), "stationary"),
    (lambda s: s["configuration"]["alignment"].update(measurement_time_policy="ignore"), "explicit"),
    (lambda s: s["channels"].pop(), "two channels"),
])
def test_unsupported_claims_cannot_enter_path(mutate, match):
    with pytest.raises(ValueError, match=match):
        _source(altered_source(mutate))


def test_bounded_exact_source_bytes_required():
    with pytest.raises(ValueError, match="exact bytes"):
        _source(SOURCE.decode())


@pytest.mark.parametrize("field", ["epoch_utc", "valid_from", "valid_until"])
def test_submicrosecond_timestamps_refuse_before_runtime_resolution(field, monkeypatch):
    experiment = json.loads(SOURCE)
    target = experiment if field == "epoch_utc" else experiment["channels"][0]["calibration_profile"]
    # Truncating this lower bound would incorrectly admit the fixture at 5 s.
    target[field] = "2026-01-01T00:00:05.0000009Z"

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Timestamp precision must be validated before binding providers")

    monkeypatch.setattr("ciw.calibrated_observable._adapters", unexpected_runtime)
    with pytest.raises(ValueError, match=field + ".*microsecond precision"):
        create_session(canonical(experiment), {})


@pytest.mark.parametrize("field,instant", [
    ("epoch_utc", "2026-01-01T00:00:00.123456000Z"),
    ("epoch_utc", "2026-01-01 00:00:00.123456000+00:00"),
    ("valid_from", "2025-12-01T01:00:00.123456000+01:00"),
    ("valid_from", "2025-12-01 01:00:00,123456000+01:00"),
    ("valid_until", "2026-12-01T00:00:00.000000000Z"),
])
def test_exact_timestamp_trailing_zeros_preserve_source_declarations(field, instant):
    experiment = json.loads(SOURCE)
    target = experiment if field == "epoch_utc" else experiment["channels"][0]["calibration_profile"]
    target[field] = instant
    assert _source(canonical(experiment)) == experiment


def test_fractional_timezone_offset_cannot_silently_become_utc():
    raw = altered_source(lambda s: s.update(epoch_utc="2026-01-01T00:00:00+00:00:00.5"))
    with pytest.raises(ValueError, match="offset.*precision loss"):
        _source(raw)


def test_complete_path_matches_analytic_process_solution(bundle):
    assert bundle["verification"]["outcome"] == "passed"
    assert bundle["verification"]["independent"] is False
    assert [(s["runtime_ref"], s["operation_id"]) for s in bundle["steps"]] == list(OPERATIONS)
    data = {step["runtime_ref"]: step["result"].get("data", step["result"]) for step in bundle["steps"]}
    assert data["tbrt"]["target_time"] == 5
    assert [c["event_time"] for c in data["tbrt"]["channels"]] == [5, 5]
    assert data["mcur"]["values"] == [52, 46]
    assert data["mcur"]["covariance"]["matrix"] == [[1, .25], [.25, 1]]
    assert data["oit"]["status"] == "observable"
    assert data["oit"]["rank"] == 2
    assert data["oit"]["condition_number"] == pytest.approx(1)
    assert data["gsie"]["mean"] == pytest.approx([607 / 12, 589 / 12])
    assert data["gsie"]["innovation"] == [2, -4]
    assert data["gsie"]["innovation_covariance"] == [[1.25, .25], [.25, 1.25]]
    assert data["cbsr"]["status"] == "accepted"
    assert data["cbsr"]["reconciled"]["estimate"] == pytest.approx([50.75, 49.25])
    assert data["fdir"]["detection"]["nis"] == pytest.approx(58 / 3)
    assert data["fdir"]["detection"]["innovation_covariance"] == data["gsie"]["innovation_covariance"]
    assert data["fdir"]["isolability"]["status"] == "isolated"
    assert data["fdir"]["isolability"]["isolated_fault"] == "sensor:tank-2:bias"
    assert data["fdir"]["detection"]["source_ids"] == [bundle["steps"][i]["result_id"] for i in (4, 5)]
    assert inspect_session(bundle)["numerical_replay"] == "not_performed"


def test_replay_preserves_numerics_with_fresh_occurrences(bundle, repositories, tmp_path):
    replay = replay_session(bundle, repositories)
    fresh = replay["session"]
    assert replay["replay_receipt"]["numerical_match"] is True
    assert replay["replay_receipt"]["admission"] == "not_performed"
    assert fresh["session_id"] != bundle["session_id"]
    assert set(replay["replay_results"]) == {step["execution_id"] for step in bundle["steps"]}
    for before, after in zip(bundle["steps"], fresh["steps"]):
        assert before["numerical_result_id"] == after["numerical_result_id"]
        assert before["execution_id"] != after["execution_id"]
        assert before["result_id"] != after["result_id"]
    path = save_session(fresh, tmp_path)
    assert read_session(path) == fresh
    with pytest.raises(ValueError, match="overwrite"):
        save_session(fresh, tmp_path)


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s["channels"][0]["calibration_joint_covariance"].update(cross_covariance_policy="unknown"), "CALIBRATED_MCUR_REFUSED"),
    (lambda s: s["calibrated_covariance"].update(cross_covariance_policy="unknown"), "CALIBRATED_MCUR_REFUSED"),
    (lambda s: s["channels"][0].update(clock_model=None), "CALIBRATED_FSRT_REFUSED"),
    (lambda s: s["channels"][0]["clock_model"].update(synchronization_evidence_ids=[]), "CALIBRATED_FSRT_REFUSED"),
    (lambda s: s["channels"][0]["clock_model"]["source_frame"].update(clock_id="substituted-clock"), "CALIBRATED_FSRT_REFUSED"),
    (lambda s: s["channels"][0]["clock_model"].update(reference_origin=1e16), "CALIBRATED_TBRT_REFUSED"),
    (lambda s: s["channels"][0]["calibration_profile"].update(valid_until="2025-12-31T00:00:00Z"), "CALIBRATED_MCUR_REFUSED"),
])
def test_missing_or_unknown_measurement_authority_refuses(repositories, mutate, code):
    with pytest.raises(AdapterRefusal) as caught:
        create_session(altered_source(mutate), repositories)
    assert caught.value.code == code


@pytest.mark.parametrize("diagonal,limit", [(1e-9, 1e8), (0, 1e8), (1, None)],
                         ids=["ill-conditioned", "unobservable", "unresolved"])
def test_noninformative_observability_blocks_estimation(repositories, diagonal, limit):
    source = json.loads(SOURCE)
    matrix = [[1, 0], [0, diagonal]]
    source["configuration"]["observability"].update(observation=matrix, condition_limit=limit)
    source["configuration"]["gsie"]["observation_model"]["matrix"] = matrix
    with pytest.raises(AdapterRefusal) as caught:
        create_session(canonical(source), repositories)
    assert caught.value.code == "CALIBRATED_GSIE_REFUSED"


def test_unknown_fault_crosscovariance_keeps_detection_but_refuses_unique_nomination(repositories):
    raw = altered_source(lambda s: s["configuration"]["fdir"].update(cross_covariance_policy="unknown"))
    fault = create_session(raw, repositories)["steps"][-1]["result"]["data"]
    assert fault["detection"]["status"] == "statistical_anomaly"
    assert fault["isolability"]["status"] == "ambiguous"
    assert fault["isolability"]["isolated_fault"] is None


def test_reconciliation_hold_retains_candidate_and_residual_diagnostics(repositories):
    raw = altered_source(lambda s: s["configuration"]["cbsr"].update(max_normalized_residual=.1))
    result = create_session(raw, repositories)
    reconciliation = result["steps"][5]["result"]["data"]
    assert reconciliation["status"] == "held"
    assert reconciliation["reconciled"] is None
    assert reconciliation["candidate"]["estimate"] == result["steps"][4]["result"]["data"]["mean"]
    assert result["steps"][6]["result"]["data"]["cbsr_status"] == "held"


@pytest.mark.parametrize("mutate", [
    lambda b: b["configuration"]["gsie"]["prior"].update(mean=[999, 999]),
    lambda b: b["steps"][3]["request"]["inputs"]["declaration"].update(model_id="substituted"),
    lambda b: b["steps"][4]["result"]["data"].update(mean=[999, 999]),
    lambda b: b["steps"][4].update(input_refs=[b["steps"][0]["result_id"]]),
    lambda b: b["steps"][1].update(operation_id="invented.clock.v1"),
    lambda b: b["steps"][4].update(execution_id=b["steps"][3]["execution_id"]),
])
def test_content_tampering_rejected_after_outer_rehash(bundle, mutate):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    mutate(changed)
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError):
        inspect_session(changed)


@pytest.mark.parametrize("replacement", [True, 1], ids=["boolean", "integer"])
@pytest.mark.parametrize("target", ["configuration", "request"])
def test_equal_python_numbers_cannot_substitute_retained_json(bundle, target, replacement):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    declaration = (changed["configuration"]["observability"] if target == "configuration"
                   else changed["steps"][3]["request"]["inputs"]["declaration"])
    assert type(declaration["transition"][0][0]) is float
    declaration["transition"][0][0] = replacement
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError, match="Configuration differs|Request differs"):
        inspect_session(changed)


def test_rehashed_numerical_projection_must_preserve_json_number_type(bundle):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    step = changed["steps"][3]
    assert type(step["numerical_result"]["data"]["rank"]) is int
    step["numerical_result"]["data"]["rank"] = float(step["numerical_result"]["data"]["rank"])
    step["numerical_result_id"] = digest(step["numerical_result"])
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError, match="numerical projection"):
        inspect_session(changed)


def test_replay_rejects_a_self_consistently_rehashed_numerical_forgery(bundle, repositories):
    changed = deepcopy(bundle)
    changed.pop("verification", None)
    step = changed["steps"][-1]
    step["result"]["data"]["detection"]["nis"] = 999
    result = step["result"]
    result["result_id"] = digest({key: value for key, value in result.items() if key != "result_id"})
    step["result_id"] = result["result_id"]
    step["result_sha256"] = digest(result)
    step["numerical_result"] = _numerical("fdir", result)
    step["numerical_result_id"] = digest(step["numerical_result"])
    changed["bundle_digest"] = _bundle_digest(changed)
    assert inspect_session(changed)["status"] == "content_consistent"
    with pytest.raises(ValueError, match="pinned recomputation"):
        replay_session(changed, repositories)


def test_inspection_cli_retains_read_only_boundary(bundle, tmp_path):
    path = save_session(bundle, tmp_path)
    run = subprocess.run([sys.executable, "-m", "ciw", "calibrated-observable", "inspect", str(path)],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result["status"] == "content_consistent"
    assert result["numerical_replay"] == "not_performed"
    assert result["admission"] == "not_performed"
