"""Workspace format versions bound what a saved workspace may carry; reopening refuses before any write."""
import json

import pytest

from ciw.instruments import make_demo_run
from ciw.session import Session


def _saved(tmp_path):
    session = Session(make_demo_run(), tmp_path / "original")
    path = session.save_workspace(tmp_path / "workspace.json")
    workspace = json.loads(path.read_text())
    assert workspace["workspace_version"] == 1 and "executions" not in workspace
    return path, workspace


def _reopen(tmp_path, workspace):
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(workspace))
    restored = tmp_path / "restored"
    with pytest.raises(ValueError) as caught:
        Session.from_workspace(path, restored)
    assert not restored.exists() or not any(restored.iterdir())
    return str(caught.value)


def test_a_version_one_workspace_cannot_carry_executions(tmp_path):
    path, workspace = _saved(tmp_path)
    Session.from_workspace(path, tmp_path / "reopened")
    workspace["executions"] = [{"execution_id": "execution-" + "0" * 32}]
    assert "Executions require workspace version 2" in _reopen(tmp_path, workspace)


@pytest.mark.parametrize("version", [1, 2])
def test_only_version_three_retains_workbench_records(tmp_path, version):
    _, workspace = _saved(tmp_path)
    workspace["workspace_version"] = version
    workspace["workbench"] = {"schema": "ciw.retained-workbench.v1"}
    assert "workspace version 3" in _reopen(tmp_path, workspace)


@pytest.mark.parametrize("version", [0, 4, "1", True, None])
def test_unknown_workspace_versions_are_refused(tmp_path, version):
    _, workspace = _saved(tmp_path)
    workspace["workspace_version"] = version
    assert "Unsupported workspace format" in _reopen(tmp_path, workspace)
