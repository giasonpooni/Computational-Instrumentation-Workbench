"""Exact acquired record mapping and actual native numerical child replay."""
from copy import deepcopy
import os
from pathlib import Path
import runpy

import pytest

from ciw import acquired_dataset as acquisition
from ciw import acquired_window as bridge
from ciw import calibrated_window as window

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / "examples/acquired-window/make_source.py"))


def declaration():
    template = HELPERS["default_template"]()
    return {"schema": bridge.SOURCE_SCHEMA, "experiment_id": template["experiment_id"],
            "selections": [{"observation_id": "a" * 64, "record_id": "b" * 64, "document_id": "c" * 64,
                            "snapshot_index": 0, "row_index": 0}],
            "declaration": {k: v for k, v in template.items() if k not in {"schema", "samples"}},
            "configuration": deepcopy(bridge.POLICY)}


@pytest.mark.parametrize("fault", ["empty", "duplicates", "boolean-index", "negative-index", "bad-id", "override", "schema", "policy", "experiment"])
def test_mapping_source_rejects_ambiguous_selection(fault):
    source = declaration()
    if fault == "empty": source["selections"] = []
    if fault == "duplicates": source["selections"] *= 2
    if fault == "boolean-index": source["selections"][0]["row_index"] = True
    if fault == "negative-index": source["selections"][0]["snapshot_index"] = -1
    if fault == "bad-id": source["selections"][0]["record_id"] = "arbitrary"
    if fault == "override": source["declaration"]["samples"] = []
    if fault == "schema": source["schema"] = "guess"
    if fault == "policy": source["configuration"]["event_time_order"] = "acquisition_cursor"
    if fault == "experiment": source["experiment_id"] = "another"
    with pytest.raises(ValueError): bridge._source(bridge.canonical(source))


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_ACQUIRED_STREAM_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_ACQUIRED_STREAM_STACK_ROOT to the exact role-named provider checkouts")
    return {role: Path(root) / role for role in window.ROLES | {"ppda"}}


@pytest.fixture(scope="module")
def retained(repositories):
    template = HELPERS["default_template"]()
    acquired_source = HELPERS["build_acquisition"]([template])
    upstream = acquisition.create_session(acquisition.canonical(acquired_source), {"ppda": repositories["ppda"]})
    source = HELPERS["build_mapping"](upstream, template, 0)
    raw = b"\n" + bridge.canonical(source) + b"\n "
    original = bridge.create_session(raw, upstream, {k: v for k, v in repositories.items() if k in bridge.ROLES})
    replay = bridge.replay_session(original, {k: v for k, v in repositories.items() if k in bridge.ROLES})
    return template, upstream, source, raw, original, replay


def test_exact_retained_mapping_and_native_numbers(retained):
    template, upstream, source, raw, original, replay = retained
    assert bridge._validate(original) == raw
    assert original["upstream_acquisition"] == upstream
    mapped = bridge.mapped_source(original)
    assert [s["raw_value"] for s in mapped["samples"]] == [2, 4]
    assert [s["device_time"] for s in mapped["samples"]] == [10, 12]
    assert [s["received_at"] for s in mapped["samples"]] == [0, 1]
    assert original["steps"] == original["child_window"]["steps"]
    assert original["steps"][2]["numerical_result"]["mean"] == 7
    assert original["steps"][2]["numerical_result"]["variance"] == 5.25
    assert original["steps"][3]["result"]["result_artifact"]["components"][0]["value"] == pytest.approx(1.12)
    assert original["verification"]["schema"] == bridge.VERIFICATION_SCHEMA
    assert original["verification"]["independent"] is False
    assert original["verification"]["child_verification_id"] == original["child_window"]["verification"]["verification_id"]
    for selected, sample in zip(original["acquisition_binding"]["selected_records"], mapped["samples"]):
        assert sample["observation_id"] == "ppda-observation:" + selected["observation_id"]
        assert sample["artifact_id"] == "ppda-record:" + selected["record_id"]
        assert selected["snapshot_sha256"] == upstream["steps"][0]["result"]["data"]["runs"][0]["snapshot_sha256"]
    assert original["source"]["evidence"][0]["artifact_ref"] != original["child_window"]["source"]["evidence"][0]["artifact_ref"]


def test_replay_freezes_acquisition_and_freshens_only_window(retained):
    original, replay = retained[4:]
    fresh, receipt = replay["session"], replay["replay_receipt"]
    bridge.validate_replay(original, fresh, receipt)
    assert fresh["acquisition_binding"] == original["acquisition_binding"]
    assert fresh["upstream_acquisition"] == original["upstream_acquisition"]
    assert fresh["child_window"]["replay_receipts"][0]["source_bundle_digest"] == original["child_window"]["bundle_digest"]
    assert fresh["session_id"] != original["session_id"]
    for old, new in zip(original["steps"], fresh["steps"]):
        assert old["numerical_result_id"] == new["numerical_result_id"]
        assert old["execution_id"] != new["execution_id"]
        assert old["result_id"] != new["result_id"]


@pytest.mark.parametrize("fault", ["observation", "record", "document", "row", "snapshot", "channel", "frame", "clock", "calibration", "unknown", "covariance-order", "raw-override"])
def test_mapping_faults_refuse_before_provider_binding(retained, monkeypatch, fault):
    source = deepcopy(retained[2])
    if fault in {"observation", "record", "document"}: source["selections"][0][fault + "_id"] = "0" * 64
    if fault == "row": source["selections"][0]["row_index"] = 1
    if fault == "snapshot": source["selections"][0]["snapshot_index"] = 1
    if fault == "channel": source["declaration"]["channel_id"] = "different"
    if fault == "frame": source["declaration"]["frame"]["id"] = "different"
    if fault == "clock": source["declaration"]["clock_model"]["model_id"] = "different"
    if fault == "calibration": source["declaration"]["calibration_profile"]["artifact_id"] = "different"
    if fault == "unknown": source["declaration"]["joint_covariance"]["cross_covariance_policy"] = "unknown"
    if fault == "covariance-order": source["declaration"]["joint_covariance"]["order"].reverse()
    if fault == "raw-override": source["selections"][0]["raw_value"] = 99
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid mapping reached native provider binding")
    monkeypatch.setattr(window, "_adapters", forbidden)
    with pytest.raises(ValueError): bridge.create_session(bridge.canonical(source), retained[1], {})


@pytest.mark.parametrize("fault", ["outer-steps", "child-bytes", "selected-record", "upstream-bytes", "receipt-subject", "receipt-authority", "receipt-child"])
def test_resealed_outer_tampering_cannot_break_lineage(retained, fault):
    bundle = deepcopy(retained[4])
    if fault == "outer-steps": bundle["steps"][0]["request"]["source"]["samples"][0]["device_time"] = 11
    if fault == "child-bytes": bundle["child_window"]["source"]["evidence"][0]["bytes_b64"] += "AA=="
    if fault == "selected-record": bundle["acquisition_binding"]["selected_records"][0]["snapshot_sha256"] = "sha256:" + "0" * 64
    if fault == "upstream-bytes": bundle["upstream_acquisition"]["source"]["evidence"][0]["bytes_b64"] += "AA=="
    bundle["bundle_digest"] = bridge._bundle_digest(bundle)
    bundle["verification"] = bridge._verification(bundle)
    if fault == "receipt-subject": bundle["verification"]["subject_ref"] = bundle["child_window"]["bundle_digest"]
    if fault == "receipt-authority": bundle["verification"]["independent"] = True
    if fault == "receipt-child": bundle["verification"]["child_verification_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError): bridge._validate(bundle)


def test_offline_inspection_never_invokes_providers(retained, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Inspection executed a provider")
    monkeypatch.setattr(window, "_adapters", forbidden)
    monkeypatch.setattr(acquisition, "_adapters", forbidden)
    assert bridge.mapped_source(retained[4])["channel_id"] == retained[0]["channel_id"]


def test_replay_receipt_must_retain_original_child_proof(retained):
    original, replay = retained[4:]
    wrong = deepcopy(replay["replay_receipt"])
    wrong["verification"] = deepcopy(replay["session"]["verification"])
    wrong["replay_id"] = bridge.digest({k: v for k, v in wrong.items() if k != "replay_id"})
    fresh = deepcopy(replay["session"])
    fresh["replay_receipts"] = [wrong]
    with pytest.raises(ValueError): bridge.validate_replay(original, fresh, wrong)


def test_selected_upstream_is_exact_occurrence(retained):
    modified = deepcopy(retained[1])
    modified["session_id"] = "session-" + "0" * 32
    with pytest.raises(ValueError): bridge.validate_upstream(retained[4], modified)
