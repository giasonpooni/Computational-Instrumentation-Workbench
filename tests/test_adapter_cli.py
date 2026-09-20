"""Terminal controls preserve generic sampling and distinct scientific states."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest

from ciw.cli import health_remote, main, print_investigation
from ciw.instruments import make_demo_run
from ciw.session import Session


@pytest.mark.parametrize("sample_rate", [None, 9.0])
def test_health_accepts_snapshot_without_assuming_uniform_samples(tmp_path, sample_rate):
    # Health inspects a metadata snapshot, not the full scientific arrays. A
    # declared rate need not imply sample_count/rate == duration for event data.
    payload = deepcopy(Session(make_demo_run(), tmp_path).snapshot())
    payload["run"]["instrument"] = "test-event-source.v1"
    payload["run"]["metadata"].update(sample_count=1, sample_rate_hz=sample_rate)
    response = {"protocol_version": 1, "type": "response", "payload": payload}
    with patch("ciw.cli.request_remote", new=AsyncMock(return_value=response)):
        assert asyncio.run(health_remote("ws://127.0.0.1:8765"))["status"] == "healthy"


@pytest.mark.parametrize("sample_rate", [True, -1, 0, "1", float("inf")])
def test_health_rejects_invalid_declared_rate_for_external_source(tmp_path, sample_rate):
    payload = deepcopy(Session(make_demo_run(), tmp_path).snapshot())
    payload["run"]["instrument"] = "test-event-source.v1"
    payload["run"]["metadata"]["sample_rate_hz"] = sample_rate
    response = {"protocol_version": 1, "type": "response", "payload": payload}
    with patch("ciw.cli.request_remote", new=AsyncMock(return_value=response)):
        with pytest.raises(ValueError, match="recording bounds"):
            asyncio.run(health_remote("ws://127.0.0.1:8765"))


def test_terminal_keeps_observation_estimate_and_refusal_distinct(capsys):
    summary = {
        "status": "refused", "workspace_file": "retained/workspace.json", "run_id": "run:test",
        "terminal_rows": [
            {"state": "raw", "quantity": "mass_a", "value": 500, "unit": "count"},
            {"state": "calibrated", "quantity": "mass_a", "value": 5, "unit": "kg"},
            {"state": "estimated", "quantity": "level_a", "value": None, "unit": "m"},
            {"state": "residual", "quantity": "mass_a", "value": None, "unit": "kg"},
        ],
        "refusals": [{"refusal": {"code": "unidentifiable", "message": "State is not identifiable"}}],
        "note": "Physical validation is not established.",
    }
    original = deepcopy(summary)
    print_investigation(summary)
    output = capsys.readouterr().out
    for label in ("raw", "calibrated", "estimated", "residual", "unidentifiable", "--json"):
        assert label in output
    assert summary == original
    assert "null" in output


def test_investigation_rejects_non_object_input_without_creating_workspace(tmp_path, capsys):
    inputs = tmp_path / "inputs.json"
    inputs.write_text("[]", encoding="utf-8")
    output = tmp_path / "output"
    assert main(["investigation", "create", "--inputs", str(inputs),
                 "--rci-repo", "unbound-rci", "--fsrt-repo", "unbound-fsrt",
                 "--output-dir", str(output)]) == 2
    assert "must be a JSON object" in capsys.readouterr().err
    assert not output.exists()


def test_real_cli_create_inspect_and_offline_replay(tmp_path):
    """Opt-in deployment gate against actual, independently pinned repositories."""
    repositories = {name: os.environ.get(f"CIW_{name.upper()}_REPO") for name in ("rci", "fsrt")}
    if not all(repositories.values()):
        pytest.skip("Set CIW_RCI_REPO and CIW_FSRT_REPO to the pinned source checkouts")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    def run(*arguments):
        completed = subprocess.run([sys.executable, "-m", "ciw", *map(str, arguments)],
                                   cwd=root, env=environment, capture_output=True, text=True,
                                   timeout=90, check=False)
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    binding = ["--rci-repo", repositories["rci"], "--fsrt-repo", repositories["fsrt"]]
    created = json.loads(run("investigation", "create", "--inputs",
                             root / "examples/adapters/two-reservoir.json", *binding,
                             "--output-dir", tmp_path / "created", "--json"))
    workspace = tmp_path / "created/workspace.json"
    before = workspace.read_bytes()
    inspected = json.loads(run("investigation", "inspect", workspace, "--evaluated-at",
                              created["calibration"][0]["serving"]["evaluated_at"], "--json"))
    # Serving-time expiry is a display fact, evaluated afresh without changing
    # any immutable source or result record.
    for summary in (created, inspected):
        assert all(item["applicable_at_acquisition"] for item in summary["calibration"])
        for item in summary["calibration"]:
            item.pop("evaluated_at")
    assert inspected == created
    assert workspace.read_bytes() == before
    display = run("investigation", "inspect", workspace)
    for state in ("observation", "calibrated_observation", "estimated_state", "residual"):
        assert state in display
    replayed = json.loads(run("investigation", "replay", workspace, *binding,
                              "--output-dir", tmp_path / "replayed", "--json"))
    assert replayed["evidence_id"] == created["evidence_id"]
    assert replayed["results"][-1]["data"] == created["results"][-1]["data"]
    assert replayed["results"][-1]["result_id"] != created["results"][-1]["result_id"]
    assert replayed["execution_ids"][-1] != created["execution_ids"][-1]
    assert workspace.read_bytes() == before
