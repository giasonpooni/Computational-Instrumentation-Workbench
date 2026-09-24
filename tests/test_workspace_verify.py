"""``ciw workspace verify`` reopens a saved workspace offline and reports replayability on this host."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import reference_workflow as base
from ciw import workspace_verify
from ciw.cli import main
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.workbench import _workflow

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "retained" / "workbench.json"
KINDS = {"energy-accuracy", "thermal-observer", "machine-manifest", "project-graph", "uncertainty-validation"}


def _first_number(node, path=()):
    if isinstance(node, dict):
        for key in sorted(node):
            found = _first_number(node[key], path + (key,))
            if found is not None:
                return found
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found = _first_number(item, path + (index,))
            if found is not None:
                return found
    elif type(node) in (int, float):
        return path
    return None


def _workspace(tmp_path, retained, name="workspace.json"):
    seed = Session(make_demo_run(), tmp_path / ("session-" + name))
    saved = json.loads(seed.save_workspace(tmp_path / ("seed-" + name)).read_text(encoding="utf-8"))
    if retained is not None:
        saved.update({"workspace_version": 3, "workbench": retained})
    path = tmp_path / name
    path.write_text(json.dumps(saved, allow_nan=False), encoding="utf-8")
    return path


def test_retained_workbench_file_verifies_and_reports_replayability_per_bundle(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("Verification started a process"))
    report = workspace_verify.verify(FIXTURE)
    assert report["schema"] == workspace_verify.SCHEMA
    assert (report["form"], report["status"], report["refusal"], report["providers_executed"]) == ("retained-workbench", "valid", None, False)
    assert report["retained"] == {"schema": "ciw.retained-workbench.v1", "revision": 10, "sources": 5, "bundles": 5, "candidates": 0}
    assert base.CODE_DIGEST.fullmatch(report["host"]["kernel_probe"]) and report["host"]["numpy_version"]
    entries = {entry["kind"]: entry for entry in report["bundles"]}
    assert set(entries) == KINDS
    retained = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for record in retained["bundles"]:
        entry = entries[record["kind"]]
        workflow = _workflow(record["kind"])
        native = record["native"]
        expected = base.identity_differences(workflow._runtime_projection(workflow._runtime_identity()),
                                             workflow._runtime_projection(native["runtimes"][workflow.role]))
        assert entry["differences"] == expected
        assert entry["replay_here"] == (workspace_verify.MATCHES if not expected else workspace_verify.DIFFERS)
        assert entry["validation"] == "valid" and entry["bundle_id"] == record["bundle_id"]
        assert entry["numerical_result_ids"] == [native["steps"][0]["numerical_result_id"]]
        assert entry["execution_ids"] == [native["steps"][0]["execution_id"]]
        assert entry["retained_verification_outcome"] == "passed" and entry["replay_receipts"] == 0
    assert "kernel_probe" not in " ".join(entries["project-graph"]["differences"])
    assert report["replayable_here"] == sum(1 for entry in entries.values() if entry["replay_here"] == workspace_verify.MATCHES)


def test_a_workspace_with_and_without_retained_bundles_verifies(tmp_path):
    retained = json.loads(FIXTURE.read_text(encoding="utf-8"))
    path = _workspace(tmp_path, retained)
    before = set(tmp_path.rglob("*"))
    report = workspace_verify.verify(path)
    assert set(tmp_path.rglob("*")) == before, "verification wrote next to the workspace"
    assert (report["form"], report["workspace_version"], report["status"]) == ("workspace", 3, "valid")
    assert {entry["kind"] for entry in report["bundles"]} == KINDS
    plain = workspace_verify.verify(_workspace(tmp_path, None, "plain.json"))
    assert (plain["form"], plain["workspace_version"], plain["status"], plain["bundles"]) == ("workspace", 1, "valid", [])
    assert plain["retained"]["bundles"] == 0 and plain["replayable_here"] == 0


def test_a_changed_retained_number_is_reported_against_its_bundle_only(tmp_path):
    retained = json.loads(FIXTURE.read_text(encoding="utf-8"))
    changed = deepcopy(retained)
    target = next(record for record in changed["bundles"] if record["kind"] == "machine-manifest")
    data = target["native"]["steps"][0]["result"]["data"]
    leaf = _first_number(data)
    parent = data
    for key in leaf[:-1]:
        parent = parent[key]
    parent[leaf[-1]] = parent[leaf[-1]] + 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    report = workspace_verify.verify(path)
    assert report["status"] == "invalid" and report["refusal"]
    statuses = {entry["kind"]: entry["validation"] for entry in report["bundles"]}
    assert statuses == {kind: ("invalid" if kind == "machine-manifest" else "valid") for kind in KINDS}
    assert "refusal" in next(entry for entry in report["bundles"] if entry["kind"] == "machine-manifest")
    workspace = _workspace(tmp_path, changed)
    assert workspace_verify.verify(workspace)["status"] == "invalid"


def test_terminal_verb_prints_the_report_and_exits_by_status(tmp_path, capsys):
    assert main(["workspace", "verify", str(FIXTURE)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "valid" and len(report["bundles"]) == 5
    changed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    changed["bundles"][0]["native"]["steps"][0]["result"]["data"]["schema"] = "tampered"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert main(["workspace", "verify", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert main(["workspace", "verify", str(other)]) == 2
