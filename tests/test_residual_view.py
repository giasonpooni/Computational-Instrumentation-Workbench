"""The residual view copies actual native diagnostics without new inference."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from ciw import calibrated_window, residual_monitor, residual_view
from ciw.core.canonical import canonical, byte_digest

ROOT = Path(__file__).resolve().parents[1]


def record(bundle):
    return {"kind": "residual-monitor", "bundle_id": bundle["bundle_digest"],
            "upstream_bundle_id": None, "native": bundle}


def source(raw):
    return {"label": "Native residual view test", "evidence_id": byte_digest(raw),
            "source_id": "retained-view-source", "bytes_b64": base64.b64encode(raw).decode()}


@pytest.fixture(scope="module")
def retained():
    stack = os.environ.get("CIW_ACQUIRED_STREAM_STACK_ROOT")
    if not stack:
        pytest.skip("set CIW_ACQUIRED_STREAM_STACK_ROOT for native residual view verification")
    repositories = Path(stack)
    template = json.loads((ROOT / "examples/calibrated-window/source.json").read_bytes())
    window_paths = {role: repositories / role for role in calibrated_window.ROLES}
    monitor_paths = {role: repositories / role for role in residual_monitor.ROLES}
    first = calibrated_window.create_session(canonical(template), window_paths)
    later = deepcopy(template)
    later["experiment_id"] = "experiment:overlapping-window"
    later["samples"] = [deepcopy(template["samples"][1]), {
        **template["samples"][1], "observation_id": "sample:2", "artifact_id": "evidence:raw-2",
        "indicated_value": 6, "raw_value": 6, "device_time": 14, "received_at": 2}]
    later["joint_covariance"]["order"] = calibrated_window.covariance_order(later)
    later["configuration"]["window"].update(start=1, end=3, received_by=3, decision_time=3)
    later["configuration"]["gsie"]["target_time"] = 3
    second = calibrated_window.create_session(canonical(later), window_paths)
    upstreams = {bundle["bundle_digest"]: bundle for bundle in (first, second)}
    declaration = {"schema": residual_monitor.SOURCE_SCHEMA, "experiment_id": "experiment:residual-view",
                   "window_bundle_ids": list(upstreams), "configuration": deepcopy(residual_monitor.DEFAULT_CONFIGURATION)}
    raw = canonical(declaration)
    original = residual_monitor.create_session(raw, upstreams, monitor_paths)
    replay = residual_monitor.replay_session(original, monitor_paths)["session"]
    unresolved = deepcopy(declaration)
    unresolved["configuration"]["observability"]["condition_limit"] = None
    held = residual_monitor.create_session(canonical(unresolved), upstreams, monitor_paths)
    return original, replay, declaration, held, unresolved


def project(bundle, declaration):
    return residual_view.project(record(bundle), source(canonical(declaration)), declaration, 7)


def test_projection_preserves_actual_native_diagnostics_and_unknown_joint_covariance(retained):
    original, _, declaration, _, _ = retained
    data = original["steps"][0]["result"]["data"]
    view = project(original, declaration)
    panels = {panel["panel_id"]: panel for panel in view["panels"]}
    assert view["fusion_context"] is None
    assert view["object_context"]["object_kind"] == "residual_sequence"
    assert view["object_context"]["rows"] == data["rows"]
    assert view["object_context"]["policy"]["false_alarm_probability"] == "not_established"
    assert view["object_context"]["overlaps"][0]["shared_observation_ids"] == ["sample:1"]
    assert view["object_context"]["overlaps"][0]["shared_calibration_profile"] == "calibration:level-affine"
    assert view["object_context"]["scope"]["model"]["prior"]["state_id"] == "synthetic:independent-prior"
    assert panels["innovation"]["values"] == [row["detection"]["raw_residual"][0] for row in data["rows"]]
    assert panels["innovation"]["context"]["per_window_covariance"] == [row["detection"]["innovation_covariance"] for row in data["rows"]]
    assert panels["normalized-residual"]["values"] == [row["detection"]["marginal_normalized_residual"][0] for row in data["rows"]]
    assert panels["nis"]["values"] == [row["detection"]["nis"] for row in data["rows"]]
    for side in ("positive", "negative"):
        assert panels["cusum-" + side]["values"] == [row["cusum"]["observed_state"][side] for row in data["rows"]]
        assert panels["cusum-" + side]["context"]["threshold"] == declaration["configuration"]["cusum"]["threshold"]
    assert data["rows"][-1]["isolability"]["status"] == "ambiguous"
    assert "ambiguous" in view["object_context"]["summary"]
    for panel in panels.values():
        assert panel["covariance"] is None and panel["marginal_standard_deviation"] is None
        assert panel["context"]["event_times"] == [2, 3]
        assert panel["context"]["sample_event_times"] == [[0, 1], [1, 2]]
        assert panel["context"]["device_times"] == [[10, 12], [12, 14]]
        assert panel["context"]["temporal_covariance_policy"] == "unknown"
        assert panel["context"]["prior_feedback"] == "not_performed"
        assert panel["provenance"]["window_bindings"] == [row["binding"] for row in data["rows"]]
    assert view["upstream_bundle_ids"] == declaration["window_bundle_ids"]
    assert view["graph"]["nodes"][0]["input_refs"] == original["steps"][0]["input_refs"]


def test_projection_does_not_execute_and_cannot_mutate_retained_values(retained, monkeypatch):
    original, _, declaration, _, _ = retained
    before = deepcopy((original, declaration))
    def deny(*args, **kwargs):
        raise AssertionError("Inspection executed a provider")
    monkeypatch.setattr(residual_monitor.workflow, "_adapters", deny)
    monkeypatch.setattr(residual_monitor.workflow, "_step", deny)
    monkeypatch.setattr(calibrated_window, "_adapters", deny)
    descriptor = source(canonical(declaration))
    view = residual_view.project(record(original), descriptor, declaration, 7)
    view["panels"][0]["values"][0] = -999
    view["panels"][0]["context"]["per_window_covariance"][0][0][0] = -999
    view["object_context"]["rows"][0]["binding"]["source_observation_ids"].clear()
    view["object_context"]["scope"]["model"]["prior"]["mean"][0] = -999
    view["upstream_bundle_ids"].clear()
    view["runtimes"].clear()
    view["verification"].clear()
    assert (original, declaration) == before
    assert descriptor == source(canonical(declaration))


def test_replay_retains_science_and_refreshes_monitor_occurrence_identity(retained):
    original, replay, declaration, _, _ = retained
    initial, fresh = project(original, declaration), project(replay, declaration)
    assert initial["bundle_id"] != fresh["bundle_id"]
    assert fresh["replay_source_bundle_ids"] == [initial["bundle_id"]]
    assert initial["object_context"] == fresh["object_context"]
    for old, new in zip(initial["panels"], fresh["panels"], strict=True):
        assert old["values"] == new["values"] and old["context"] == new["context"]
        assert old["provenance"]["result_id"] != new["provenance"]["result_id"]
        assert old["provenance"]["execution_id"] != new["provenance"]["execution_id"]
        assert old["provenance"]["window_bindings"] == new["provenance"]["window_bindings"]


def test_held_observability_has_no_fabricated_cusum_zero(retained):
    _, _, _, held, declaration = retained
    view = project(held, declaration)
    assert "unresolved" in view["object_context"]["summary"]
    assert not any(panel["panel_id"].startswith("cusum-") for panel in view["panels"])
    for row in view["object_context"]["rows"]:
        assert row["cusum"] is None
        assert row["interpretation"]["observability_gate"] == "held"
        assert row["monitor_state_before"] == row["monitor_state_after"]
    assert {panel["panel_id"] for panel in view["panels"]} == {"innovation", "normalized-residual", "nis"}


def test_cusum_plot_keeps_observed_alarm_before_declared_reset(retained):
    original, _, declaration, _, _ = retained
    declaration = deepcopy(declaration)
    declaration["configuration"]["cusum"]["reset_on_alarm"] = True
    repositories = Path(os.environ["CIW_ACQUIRED_STREAM_STACK_ROOT"])
    monitor = residual_monitor.create_session(canonical(declaration), original["upstream_windows"],
        {role: repositories / role for role in residual_monitor.ROLES})
    data = monitor["steps"][0]["result"]["data"]
    recurrence = data["rows"][-1]["cusum"]
    assert recurrence["observed_state"]["positive"] >= declaration["configuration"]["cusum"]["threshold"]
    assert recurrence["next_state"]["positive"] == 0
    view = project(monitor, declaration)
    positive = next(panel for panel in view["panels"] if panel["panel_id"] == "cusum-positive")
    assert positive["values"][-1] == recurrence["observed_state"]["positive"]
    assert positive["context"]["accumulator_phase"] == "observed_before_optional_reset"
