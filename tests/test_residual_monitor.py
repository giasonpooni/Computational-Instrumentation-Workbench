"""Native monitoring reuses retained innovations and never estimates state."""
from copy import deepcopy
import os
from pathlib import Path
import runpy

import pytest

from ciw import calibrated_window as window
from ciw import residual_monitor as monitor
from ciw.adapters.subprocess import PinnedSubprocessAdapter
from ciw.declared_workload import _verification

ROOT = Path(__file__).resolve().parents[1]


def source(ids):
    return {"schema": monitor.SOURCE_SCHEMA, "experiment_id": "fixed-reference-residual-sequence",
            "window_bundle_ids": ids, "configuration": deepcopy(monitor.DEFAULT_CONFIGURATION)}


@pytest.mark.parametrize("fault", ["duplicate", "empty", "too-many", "malformed-id", "implicit-independence", "false-alarm", "feedback", "unique-signature", "boolean-threshold", "negative-drift", "reset", "scope", "extra", "large"])
def test_invalid_or_undeclared_monitor_source_refuses(fault):
    value = source(["sha256:" + "1" * 64])
    config = value["configuration"]
    if fault == "duplicate": value["window_bundle_ids"] *= 2
    elif fault == "empty": value["window_bundle_ids"] = []
    elif fault == "too-many": value["window_bundle_ids"] = ["sha256:" + f"{i:064x}" for i in range(17)]
    elif fault == "malformed-id": value["window_bundle_ids"][0] = "execution-not-bundle"
    elif fault == "implicit-independence": config["temporal_covariance_policy"] = "declared_zero"
    elif fault == "false-alarm": config["false_alarm_probability"] = 0.05
    elif fault == "feedback": config["prior_feedback"] = "previous_posterior"
    elif fault == "unique-signature": del config["fault_signatures"]["process.bias"]
    elif fault == "boolean-threshold": config["detection_threshold"] = True
    elif fault == "negative-drift": config["cusum"]["drift"] = -1
    elif fault == "reset": config["cusum"]["reset_on_alarm"] = "yes"
    elif fault == "scope": config["observability"]["horizon"] = 2
    elif fault == "extra": value["provider_path"] = "/untrusted"
    elif fault == "large": value["experiment_id"] = "x" * 300000
    with pytest.raises(ValueError): monitor._source(monitor.canonical(value))


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_ACQUIRED_STREAM_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_ACQUIRED_STREAM_STACK_ROOT to exact initialized native providers")
    return {role: Path(root) / role for role in window.ROLES | monitor.ROLES}


@pytest.fixture(scope="module")
def retained(repositories):
    declarations = runpy.run_path(str(ROOT / "examples/acquired-stream/make_sequence.py"))["templates"]()
    windows = [window.create_session(window.canonical(item), {role: repositories[role] for role in window.ROLES}) for item in declarations]
    upstreams = {item["bundle_digest"]: item for item in windows}
    raw = b"\n" + monitor.canonical(source(list(upstreams))) + b"\n "
    providers = {role: repositories[role] for role in monitor.ROLES}
    original = monitor.create_session(raw, upstreams, providers)
    replay = monitor.replay_session(original, providers)["session"]
    held_source = source(list(upstreams))
    held_source["configuration"]["observability"]["condition_limit"] = None
    held = monitor.create_session(monitor.canonical(held_source), upstreams, providers)
    return raw, original, replay, held, declarations


def test_native_detects_candidate_and_preserves_unknown_dependence(retained):
    raw, original, replay, held, _ = retained
    assert monitor._validate(original) == raw == monitor._validate(replay)
    data = original["steps"][0]["result"]["data"]
    assert [row["binding"]["target_time"] for row in data["rows"]] == [2, 4, 6]
    assert [row["detection"]["raw_residual"] for row in data["rows"]] == [[0.0], [0.5], [10.0]]
    assert all(row["observability"]["status"] == "observable" for row in data["rows"])
    assert data["rows"][-1]["cusum"]["status"] == "statistical_anomaly"
    assert data["rows"][-1]["interpretation"]["diagnostic_drift_candidate"] is True
    assert data["rows"][-1]["isolability"]["status"] == "ambiguous"
    assert all(row["isolability"]["isolated_fault"] is None for row in data["rows"])
    assert all(row["interpretation"]["alarm_authority"] == "held_unknown_temporal_dependence" for row in data["rows"])
    assert all(pair["residual_cross_covariance"] == "unknown" for pair in data["overlaps"])
    assert all(pair["shared_reference_prior"] == "synthetic:fixed-calibrated-reference-prior" for pair in data["overlaps"])
    for record in data["rows"]:
        native = original["upstream_windows"][record["binding"]["bundle_id"]]["steps"][3]
        diagnostics = native["result"]["result_artifact"]["diagnostics"]
        assert record["detection"]["raw_residual"] == diagnostics["innovation"]
        assert record["detection"]["innovation_covariance"] == diagnostics["innovation_covariance"]
        assert record["detection"]["source_ids"] == [native["result_id"]]
        assert diagnostics["observability"] == "not_evaluated"
    assert original["upstream_windows"] == replay["upstream_windows"]
    assert original["steps"][0]["numerical_result_id"] == replay["steps"][0]["numerical_result_id"]
    assert original["steps"][0]["execution_id"] != replay["steps"][0]["execution_id"]
    assert original["steps"][0]["result_id"] != replay["steps"][0]["result_id"]
    assert data["final_state"]["sample_count"] == 3
    assert original["verification"]["independent"] is False
    assert original["runtimes"]["fdir"]["companions"]["oit"]["revision"] == monitor.PINS["oit"]["revision"]


def test_native_observability_hold_preserves_unadvanced_state(retained):
    held = retained[3]
    data = held["steps"][0]["result"]["data"]
    assert data["final_state"] == {"positive": 0.0, "negative": 0.0, "sample_count": 0}
    for row in data["rows"]:
        assert row["observability"]["status"] == "unresolved"
        assert row["cusum"] is None
        assert row["monitor_state_before"] == row["monitor_state_after"] == data["final_state"]
        assert row["interpretation"]["observability_gate"] == "held"
        assert row["interpretation"]["diagnostic_drift_candidate"] is False


def test_replay_or_reordered_window_cannot_count_twice(retained):
    bundle = retained[1]
    upstreams = bundle["upstream_windows"]
    ids = list(upstreams)
    with pytest.raises(ValueError, match="increase strictly"):
        monitor._request(source(list(reversed(ids))), upstreams)
    with pytest.raises(ValueError, match="increase strictly"):
        monitor._request(source([ids[0], ids[0]]), upstreams)


def test_restore_and_inspection_do_not_execute_estimator_or_provider(retained, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Retained validation cannot execute providers")
    monkeypatch.setattr(PinnedSubprocessAdapter, "__init__", forbidden)
    monkeypatch.setattr(window, "create_session", forbidden)
    monkeypatch.setattr(window, "replay_session", forbidden)
    assert monitor._validate(retained[1]) == retained[0]
    monitor.validate_upstreams(retained[1], retained[1]["upstream_windows"])


def test_generic_fabricated_pass_receipt_is_not_native_set_verification(retained):
    child = deepcopy(next(iter(retained[1]["upstream_windows"].values())))
    proof = {"schema": "fabricated-proof.v1", "subject_ref": child["bundle_digest"], "outcome": "passed", "independent": False}
    proof["verification_id"] = monitor.byte_digest(proof["schema"].encode() + b"\0" + monitor.canonical(proof))
    child["verification"] = proof
    with pytest.raises(ValueError, match="SET receipt"):
        monitor._unwrap(child)


def test_native_rank_and_count_fields_are_exact_integers():
    with pytest.raises(ValueError, match="exact integer"):
        monitor._assert_numerical({"rank": 1.000000000001}, {"rank": 1})
    with pytest.raises(ValueError, match="exact integer"):
        monitor._assert_numerical({"sample_count": 2.0}, {"sample_count": 2})


def _reseal(bundle):
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = monitor.digest({k: v for k, v in result.items() if k != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = monitor.digest(result)
        step["request_sha256"] = monitor.digest(step["request"])
        step["numerical_result"] = {"operation_id": monitor.OPERATION, "data": deepcopy(result["data"])}
        step["numerical_result_id"] = monitor.digest(step["numerical_result"])
    bundle["bundle_digest"] = monitor._bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])


@pytest.mark.parametrize("fault", ["residual", "covariance", "covariance-roundoff", "cusum", "isolation", "observability", "scope", "overlap", "upstream", "tree", "proof"])
def test_resealed_false_diagnostics_or_bindings_refuse(retained, fault):
    bundle = deepcopy(retained[1])
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        data = step["result"]["data"]
        row = data["rows"][-1]
        if fault == "residual": row["detection"]["raw_residual"][0] = 0
        elif fault == "covariance": row["detection"]["innovation_covariance"][0][0] *= 2
        elif fault == "covariance-roundoff": row["detection"]["innovation_covariance"][0][0] += 1e-15
        elif fault == "cusum": row["cusum"]["prior_state"]["sample_count"] += 1
        elif fault == "isolation": row["isolability"]["isolated_fault"] = "sensor.bias"
        elif fault == "observability": row["observability"]["diagnostics"]["rank"] = 0
        elif fault == "scope": data["policy"]["temporal_covariance_policy"] = "declared_zero"
        elif fault == "overlap": data["overlaps"][0]["residual_cross_covariance"] = "declared_zero"
    if fault == "upstream": del bundle["upstream_windows"][next(iter(bundle["upstream_windows"]))]
    elif fault == "tree": bundle["runtimes"]["fdir"]["companions"]["oit"]["source_tree"] = "0" * 40
    elif fault == "proof": bundle["verification"]["reproduction"] = deepcopy(bundle["steps"][0])
    _reseal(bundle)
    with pytest.raises(ValueError): monitor._validate(bundle)


def test_different_reference_prior_is_not_implicit_feedback(retained, repositories):
    declaration = deepcopy(retained[4][-1])
    declaration["configuration"]["gsie"]["prior"]["mean"] = [8.0]
    declaration["configuration"]["gsie"]["prior"]["state_id"] = "forbidden:previous-posterior"
    altered = window.create_session(window.canonical(declaration), {role: repositories[role] for role in window.ROLES})
    original = retained[1]
    first = next(iter(original["upstream_windows"]))
    selected = {first: original["upstream_windows"][first], altered["bundle_digest"]: altered}
    with pytest.raises(ValueError, match="reference prior"):
        monitor._request(source(list(selected)), selected)


def test_explicit_positive_direction_and_post_alarm_reset(retained, repositories):
    value = source(list(retained[1]["upstream_windows"]))
    value["configuration"]["cusum"].update(direction="positive", reset_on_alarm=True)
    output = monitor.create_session(monitor.canonical(value), retained[1]["upstream_windows"], {role: repositories[role] for role in monitor.ROLES})
    row = output["steps"][0]["result"]["data"]["rows"][-1]
    assert row["cusum"]["observed_state"]["positive"] >= value["configuration"]["cusum"]["threshold"]
    assert row["cusum"]["next_state"] == {"positive": 0.0, "negative": 0.0, "sample_count": 3}
