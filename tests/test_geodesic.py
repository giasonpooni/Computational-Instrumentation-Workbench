"""Real pinned GTE → CIW session → offline inspection/replay integration gates."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.cli import main
from ciw.core.identities import evidence_id
from ciw.geodesic import (
    GTE_OPERATION, bind_gte, create_investigation, inspect_investigation, replay_investigation,
)
from ciw.operations.runner import seal
from ciw.session import Session, read_json


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples/adapters/circle.json"


@pytest.fixture(scope="module")
def gte_repo():
    repository = os.environ.get("CIW_GTE_REPO")
    if not repository:
        pytest.skip("Set CIW_GTE_REPO to exercise the real pinned GTE subprocess")
    return Path(repository).resolve()


@pytest.fixture(scope="module")
def investigation(gte_repo, tmp_path_factory):
    directory = tmp_path_factory.mktemp("ciw-gte")
    # Unicode and noncanonical whitespace exercise exact evidence-byte retention.
    inputs = json.loads(EXAMPLE.read_text())
    inputs["observations"]["source_id"] = "Échantillon — retained bytes"
    source = directory / "source.json"
    source.write_bytes((json.dumps(inputs, indent=3, ensure_ascii=False) + "\n\n").encode())
    before = set(sys.modules)
    summary = create_investigation(source, gte_repo, directory / "created")
    return {"summary": summary, "workspace": read_json(Path(summary["workspace_file"])),
            "source": source, "new_modules": set(sys.modules)-before}


def _request(parameters=None):
    return {"protocol_version": 1, "request_id": "gte-test", "type": "operation.execute",
            "payload": {"operation_id": GTE_OPERATION, "parameters": parameters or {}}}


def test_non_object_or_duplicate_input_refuses_before_output(tmp_path, capsys):
    source, output = tmp_path / "input.json", tmp_path / "must-not-write"
    for raw in (b"[]", b'{"schema":"a","schema":"b"}'):
        source.write_bytes(raw)
        assert main(["geodesic", "create", "--inputs", str(source), "--gte-repo", "absent",
                     "--output-dir", str(output)]) == 2
        assert not output.exists()
    assert "Duplicate JSON key" in capsys.readouterr().err


def test_exact_raw_request_covariance_and_identity_are_retained(investigation):
    summary, workspace = investigation["summary"], investigation["workspace"]
    assert summary["status"] == "completed", summary["refusals"]
    source = workspace["run"]["metadata"]["gte_source"]
    assert base64.b64decode(source["request_bytes_b64"]) == investigation["source"].read_bytes()
    result = workspace["results"][0]
    assert result["data"]["observed_points_m"] == source["request"]["observations"]["points_m"]
    assert result["data"]["uncertainty"]["input_joint_covariance"] == source["request"]["observations"]["covariance"]["matrix"]
    assert result["verification_id"] is None and result["verification_status"] == "not_verified"
    assert len({result["result_id"], result["execution_id"], result["evidence_id"], result["operation_id"]}) == 4
    uncertainty = result["data"]["uncertainty"]
    assert np.asarray(uncertainty["tangent_joint_covariance"]).shape == (2, 2)
    assert np.asarray(uncertainty["ambient_joint_covariance"]).shape == (4, 4)
    assert uncertainty["tangent_joint_covariance"][0][1] != 0
    np.testing.assert_allclose(result["data"]["projected_points_m"], [[1.0, 0.0], [0.0, 1.0]])
    assert result["data"]["diagnostics"]["radial_residual_before_m"][0] != 0
    assert not any(name == "geodesic_telemetry" or name.startswith("geodesic_telemetry.")
                   for name in investigation["new_modules"])
    pin = {role: __import__("ciw.pipelines", fromlist=["provider_descriptor"]).provider_descriptor(role)["pin"] for role in ("rci", "fsrt", "jspt", "gte")}["gte"]
    assert result["runtime"]["revision"] == pin["revision"]
    assert result["runtime"]["module"] == pin["module"]


def test_offline_inspection_does_not_execute_or_mutate(investigation, monkeypatch):
    import ciw.geodesic as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection attempted to load a scientific runtime")

    monkeypatch.setattr(module, "_runtime", forbidden)
    source = Path(investigation["summary"]["workspace_file"])
    before = {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()}
    assert inspect_investigation(source) == investigation["summary"]
    assert {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()} == before


def test_replay_appends_events_without_replacing_evidence(investigation, gte_repo, tmp_path):
    original = investigation["summary"]
    replay = replay_investigation(original["workspace_file"], gte_repo, tmp_path / "replay")
    assert replay["replay_data_digest_matches"] is True
    assert replay["evidence_id"] == original["evidence_id"]
    assert replay["run_id"] == original["run_id"]
    assert replay["results"][0] == original["results"][0]
    assert replay["results"][-1]["result_id"] != original["results"][0]["result_id"]
    assert replay["execution_ids"][-1] != original["execution_ids"][0]


@pytest.mark.parametrize("kind", ["held", "stale", "singular", "covariance"])
def test_domain_hold_and_refusals_remain_distinct(kind, gte_repo, tmp_path):
    inputs = json.loads(EXAMPLE.read_text())
    if kind == "held":
        inputs["policy"]["max_correction_m"] = 0.001
    elif kind == "stale":
        inputs["constraint"]["valid_time_s"] = [0, 1]
    elif kind == "singular":
        inputs["observations"]["points_m"][0] = [0, 0]
    else:
        inputs["observations"]["covariance"]["matrix"][0][0] = -1
    source = tmp_path / "input.json"
    source.write_text(json.dumps(inputs))
    summary = create_investigation(source, gte_repo, tmp_path / "created")
    assert inspect_investigation(summary["workspace_file"]) == summary
    if kind == "held":
        assert summary["status"] == "completed"
        data = summary["results"][0]["data"]
        assert data["reconciliation"]["status"] == "held"
        assert data["reconciled_points_m"] is None
        assert data["projected_points_m"] is not None
        assert data["observed_points_m"] == inputs["observations"]["points_m"]
    else:
        assert summary["status"] == "refused" and summary["results"] == []
        assert summary["refusals"][0]["refusal"]["code"] == {
            "stale": "constraint_not_valid", "singular": "numeric_geometry", "covariance": "input_covariance",
        }[kind]
        replay = replay_investigation(summary["workspace_file"], gte_repo, tmp_path / "replay")
        assert replay["replay_data_digest_matches"] is True
        assert replay["status"] == "refused" and replay["results"] == []


def test_shared_session_executes_full_batch_and_rejects_observation_overrides(investigation, gte_repo, tmp_path):
    session = Session.from_workspace(Path(investigation["summary"]["workspace_file"]), tmp_path)
    bind_gte(session, gte_repo)
    selection = session.handle({"protocol_version": 1, "request_id": "select", "type": "selection.update",
                                "payload": {"expected_revision": 0, "interval_s": [0.0, 0.5], "channel": "y"}})
    assert selection["type"] == "response"
    narrowed = session.handle(_request())
    assert narrowed["payload"]["status"] == "refused"
    assert narrowed["payload"]["execution"]["refusal"]["code"] == "selection_scope"
    assert narrowed["payload"]["result"] is None
    session.handle({"protocol_version": 1, "request_id": "select-all", "type": "selection.update",
                    "payload": {"expected_revision": 1, "interval_s": [0.0, 2.0]}})
    success = session.handle(_request())
    assert success["payload"]["status"] == "completed"
    assert len(success["payload"]["result"]["data"]["time_s"]) == 2
    refusal = session.handle(_request({"observations": {"points_m": [[1, 0]]}}))
    assert refusal["payload"]["status"] == "refused"
    assert refusal["payload"]["execution"]["refusal"]["code"] == "invalid_parameters"


def test_policy_override_preserves_evidence_and_replays_held_result(investigation, gte_repo, tmp_path):
    session = Session.from_workspace(Path(investigation["summary"]["workspace_file"]), tmp_path / "override")
    bind_gte(session, gte_repo)
    evidence = deepcopy(session.run)
    policy = {"max_correction_m": 0.001, "max_linearization_ratio": 0.1}
    response = session.handle(_request({"policy": policy}))
    assert response["payload"]["status"] == "completed"
    result = response["payload"]["result"]
    assert result["data"]["reconciliation"]["status"] == "held"
    assert result["parameters"] == {"policy": policy}
    assert session.run == evidence
    # Replay must use the invocation's complete interval even when the saved
    # viewer selection later narrows to a single sample.
    session.handle({"protocol_version": 1, "request_id": "move-display", "type": "selection.update",
                    "payload": {"expected_revision": 0, "interval_s": [0.0, 0.5]}})
    path = session.save_workspace(session.output_dir / "workspace.json")
    replay = replay_investigation(path, gte_repo, tmp_path / "replay")
    assert replay["replay_data_digest_matches"] is True
    assert replay["results"][-1]["parameters"] == {"policy": policy}
    assert replay["results"][-1]["data"] == result["data"]
    assert replay["evidence_id"] == evidence["evidence_id"]


def test_resealed_source_channel_mismatch_refuses_before_destination_writes(investigation, tmp_path):
    workspace = deepcopy(investigation["workspace"])
    workspace["run"]["channels"]["x"]["values"][0] = 10
    workspace["run"]["evidence_id"] = evidence_id(workspace["run"])
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(workspace))
    output = tmp_path / "must-not-write"
    with pytest.raises(ValueError, match="differ from retained observations"):
        Session.from_workspace(path, output)
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["source", "covariance", "asymmetric_covariance", "mixed_covariance", "claims",
                                        "held", "boolean_time"])
def test_resealed_result_still_must_satisfy_source_and_scope(investigation, tmp_path, corruption):
    workspace = deepcopy(investigation["workspace"])
    result = workspace["results"][0]
    if corruption == "source":
        result["data"]["observed_points_m"][0][0] = 3
    elif corruption == "covariance":
        result["data"]["uncertainty"]["tangent_joint_covariance"] = [[1, 2], [2, 1]]
    elif corruption == "asymmetric_covariance":
        covariance = result["data"]["uncertainty"]["tangent_joint_covariance"]
        covariance[0][1] = covariance[0][1] + 0.25 * abs(covariance[0][0])
    elif corruption == "mixed_covariance":
        result["data"]["uncertainty"]["ambient_joint_covariance"] = np.diag([1e12, 1e12, -1, -1]).tolist()
    elif corruption == "claims":
        result["data"]["uncertainty"]["limitations"] = []
    elif corruption == "boolean_time":
        result["data"]["time_s"][1] = True
    else:
        result["data"]["reconciliation"]["status"] = "held"
    seal(result)
    path, output = tmp_path / "tampered.json", tmp_path / "must-not-write"
    path.write_text(json.dumps(workspace))
    with pytest.raises(ValueError) as caught:
        Session.from_workspace(path, output)
    assert not output.exists()
    if corruption == "asymmetric_covariance":
        assert "symmetric" in str(caught.value).lower(), caught.value


def test_resealed_parsed_source_cannot_replace_numeric_literal_with_bool(investigation, tmp_path):
    workspace = deepcopy(investigation["workspace"])
    workspace["run"]["metadata"]["gte_source"]["request"]["constraint"]["radius_m"] = True
    workspace["run"]["evidence_id"] = evidence_id(workspace["run"])
    source, output = tmp_path / "boolean-source.json", tmp_path / "must-not-write"
    source.write_text(json.dumps(workspace))
    with pytest.raises(ValueError, match="differs from its retained original bytes"):
        Session.from_workspace(source, output)
    assert not output.exists()


@pytest.mark.parametrize("refused", [False, True])
def test_real_terminal_create_inspect_replay(gte_repo, tmp_path, refused):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def run(*arguments):
        process = subprocess.run([sys.executable, "-m", "ciw", "geodesic", *map(str, arguments)],
                                 env=environment, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stderr + process.stdout
        return json.loads(process.stdout)

    inputs = json.loads(EXAMPLE.read_text())
    if refused:
        inputs["constraint"]["valid_time_s"] = [0.0, 1.0]
    source = tmp_path / "request.json"
    source.write_text(json.dumps(inputs))
    created = run("create", "--inputs", source, "--gte-repo", gte_repo,
                  "--output-dir", tmp_path / "created", "--json")
    inspected = run("inspect", created["workspace_file"], "--json")
    assert inspected == created
    replay = run("replay", created["workspace_file"], "--gte-repo", gte_repo,
                 "--output-dir", tmp_path / "replay", "--json")
    assert replay["replay_data_digest_matches"] is True
    assert created["status"] == ("refused" if refused else "completed")
    assert replay["status"] == created["status"]
