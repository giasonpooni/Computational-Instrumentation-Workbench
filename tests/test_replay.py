"""Saved workspace integrity and discovery without scientific recomputation."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ciw.instruments import make_demo_run
from ciw.session import Session, read_json


def request(kind, payload=None):
    return {"protocol_version": 1, "request_id": "replay-test", "type": kind,
            "payload": {} if payload is None else payload}


@pytest.fixture(scope="module")
def workspace_template(tmp_path_factory):
    directory = tmp_path_factory.mktemp("replay-source")
    session = Session(make_demo_run(), directory)
    for kind in ("analysis.stats", "analysis.spectrum"):
        response = session.handle(request(kind))
        assert response["type"] == "response", response
    path = session.save_workspace(directory / "workspace.json")
    return read_json(path)


def write_workspace(tmp_path, workspace):
    path = tmp_path / "saved-workspace.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    return path


def test_saved_results_are_discoverable_and_identical_without_recomputation(tmp_path, workspace_template):
    path = write_workspace(tmp_path, workspace_template)
    target = tmp_path / "restored"
    with patch("ciw.session.compute_statistics", side_effect=AssertionError("must not recompute")), \
            patch("ciw.session.compute_spectrum", side_effect=AssertionError("must not recompute")):
        restored = Session.from_workspace(path, target)
        snapshot = restored.snapshot()
        listed = restored.handle(request("result.list"))
        assert listed["type"] == "response"
        assert listed["payload"]["results"] == snapshot["results"]
        assert len(snapshot["results"]) == 2
        assert snapshot["selection"] == workspace_template["selection"]
        for summary, original in zip(snapshot["results"], workspace_template["results"]):
            assert summary == {key: original[key] for key in summary}
            assert "data" not in summary
            fetched = restored.handle(request("result.get", {"result_id": summary["result_id"]}))
            assert fetched["payload"] == original
            assert read_json(target / (original["result_id"] + ".json")) == original
            assert original["recording_file"] == restored.recording_file
            assert read_json(target / original["recording_file"]) == workspace_template["run"]
        second_save = restored.save_workspace(target / "resaved.json")
        resaved = read_json(second_save)
        assert resaved["results"] == workspace_template["results"]
        assert resaved["run"] == workspace_template["run"]
    # Mutating returned discovery data must not change the persisted results.
    snapshot["results"][0]["interval_s"][0] = -100
    assert restored.snapshot()["results"][0]["interval_s"][0] == 0
    assert restored.handle(request("result.list", {"unexpected": True}))["type"] == "error"


def test_session_rejects_mismatched_scientific_evidence_before_writing(tmp_path):
    run = make_demo_run()
    run["channels"]["q"]["values"][0] += 100
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError, match="Evidence integrity"):
            Session(run, tmp_path / "new-output")
        writer.assert_not_called()


@pytest.mark.parametrize("path,value", [
    (("selection", "channel"), "absent"),
    (("selection", "interval_s"), [1, 0]),
    (("selection", "cursor_s"), -1),
    (("selection", "revision"), True),
    (("run", "channels", "q", "values", 0), 500),
    (("run", "metadata", "provenance", "dtype"), "float32"),
    (("saved_at",), "not-a-date"),
    (("view_settings",), []),
    (("results", 0, "operation_id"), "unimplemented.v1"),
    (("results", 0, "execution_id"), "execution-fake"),
    (("results", 0, "result_id"), "../../outside"),
    (("results", 0, "run_id"), "other-run"),
    (("results", 0, "evidence_id"), "other-evidence"),
    (("results", 0, "recording_file"), "../other-recording.json"),
    (("results", 0, "verification_status"), "verified"),
    (("results", 0, "verification_id"), "invented-verification"),
    (("results", 0, "selection_revision"), -1),
    (("results", 0, "selection_revision"), 1),
    (("results", 0, "selection_revision"), True),
    (("results", 0, "channel"), "nonexistent"),
    (("results", 0, "interval_s"), [-1, 100]),
    (("results", 0, "interval_s"), [0.001, 0.002]),
    (("results", 0, "created_at"), "2026-09-19T12:00:00"),
    (("results", 0, "data"), "not-numerical-data"),
    (("results", 0, "data", "mean"), "0.2"),
    (("results", 0, "data", "mean"), float("nan")),
    (("results", 0, "data", "mean"), 10**400),
    (("results", 0, "data", "sample_count"), 767),
    (("results", 0, "data", "sample_count"), True),
    (("results", 0, "data", "unit"), "cm"),
    (("results", 0, "data", "rms"), -1),
    (("results", 0, "data", "minimum"), 1000),
    (("results", 1, "data", "method"), "welch"),
    (("results", 1, "data", "window"), "rectangular"),
    (("results", 1, "data", "detrend"), "linear"),
    (("results", 1, "data", "scaling"), "spectrum"),
    (("results", 1, "data", "sample_rate_hz"), 128),
    (("results", 1, "data", "unit"), "m"),
    (("results", 1, "data", "psd"), [1]),
    (("results", 1, "data", "psd", 0), -1),
    (("results", 1, "data", "psd", 0), float("inf")),
    (("results", 1, "data", "psd", 0), True),
    (("results", 1, "data", "frequency_hz"), []),
    (("results", 1, "data", "frequency_hz", 1), 0.5),
    (("results", 1, "data", "frequency_hz", 0), "0"),
    (("results", 1, "data", "peak_frequency_hz"), None),
    (("results", 1, "data", "peak_frequency_hz"), 31),
])
def test_invalid_workspace_is_rejected_before_any_write(tmp_path, workspace_template, path, value):
    workspace = deepcopy(workspace_template)
    parent = workspace
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    source = write_workspace(tmp_path, workspace)
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError):
            Session.from_workspace(source, tmp_path / "not-created")
        writer.assert_not_called()
    assert not (tmp_path / "not-created").exists()


def test_changed_render_record_breaks_saved_result_file_binding(tmp_path, workspace_template):
    workspace = deepcopy(workspace_template)
    # A render-only change leaves scientific evidence intact, but changes the
    # exact full-record file to which an existing result was bound.
    workspace["run"]["render"]["transform"]["scale"][0] = 3.0
    source = write_workspace(tmp_path, workspace)
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError, match="recording_file"):
            Session.from_workspace(source, tmp_path / "not-created")
        writer.assert_not_called()


@pytest.mark.parametrize("identity", ["result_id", "execution_id"])
def test_duplicate_result_or_execution_id_is_rejected(tmp_path, workspace_template, identity):
    workspace = deepcopy(workspace_template)
    workspace["results"][1][identity] = workspace["results"][0][identity]
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError, match="duplication"):
            Session.from_workspace(write_workspace(tmp_path, workspace), tmp_path / "not-created")
        writer.assert_not_called()


def test_source_binding_is_required_even_when_other_identities_match(tmp_path, workspace_template):
    workspace = deepcopy(workspace_template)
    del workspace["results"][0]["recording_file"]
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError, match="source binding"):
            Session.from_workspace(write_workspace(tmp_path, workspace), tmp_path / "not-created")
        writer.assert_not_called()


def test_zero_psd_can_replay_without_inventing_a_peak(tmp_path, workspace_template):
    workspace = deepcopy(workspace_template)
    spectrum = workspace["results"][1]["data"]
    # Replay validates stored schema and consistency, not numerical correctness.
    # It must not recalculate the source signal to replace an existing result.
    spectrum["psd"] = [0.0] * len(spectrum["psd"])
    spectrum["peak_frequency_hz"] = None
    with patch("ciw.session.compute_spectrum", side_effect=AssertionError("must not recompute")):
        restored = Session.from_workspace(write_workspace(tmp_path, workspace), tmp_path / "restored")
    saved = restored.results[workspace["results"][1]["result_id"]]
    assert saved["data"] == spectrum
    assert saved["verification_status"] == "not_verified"


def test_zero_psd_with_a_peak_is_invalid(tmp_path, workspace_template):
    workspace = deepcopy(workspace_template)
    spectrum = workspace["results"][1]["data"]
    spectrum["psd"] = [0.0] * len(spectrum["psd"])
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError, match="null peak"):
            Session.from_workspace(write_workspace(tmp_path, workspace), tmp_path / "not-created")
        writer.assert_not_called()
