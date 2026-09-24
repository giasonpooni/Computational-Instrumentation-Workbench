"""Real RCI → saved CIW investigation → FSRT gates, using operator-bound checkouts.

Set CIW_RCI_REPO and CIW_FSRT_REPO to the clean, pinned domain checkouts and
run with the shared adapter environment's Python. These are integration gates;
they are explicitly skipped when the external scientific sources are absent.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.core.identities import digest, evidence_id
from ciw.investigation import (
    FSRT_OPERATION, bind_fsrt, create_investigation,
    inspect_investigation, replay_investigation,
)
from ciw.operations.runner import seal
from ciw.session import Session, read_json


pytestmark = pytest.mark.skipif(
    not (os.environ.get("CIW_RCI_REPO") and os.environ.get("CIW_FSRT_REPO")),
    reason="Set CIW_RCI_REPO and CIW_FSRT_REPO to exercise pinned domain subprocesses",
)


@pytest.fixture(scope="module")
def sources():
    return Path(os.environ["CIW_RCI_REPO"]), Path(os.environ["CIW_FSRT_REPO"])


@pytest.fixture(scope="module")
def inputs():
    path = Path(__file__).parents[1] / "examples" / "adapters" / "two-reservoir.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def investigation(tmp_path_factory, sources, inputs):
    output = tmp_path_factory.mktemp("calibrated-investigation") / "original"
    before_modules = set(sys.modules)
    summary = create_investigation(deepcopy(inputs), *sources, output,
                                   python_executable=sys.executable)
    return {
        "summary": summary,
        "path": Path(summary["workspace_file"]),
        "workspace": read_json(Path(summary["workspace_file"])),
        "new_modules": set(sys.modules) - before_modules,
    }


def _request(operation_id, parameters=None):
    return {"protocol_version": 1, "request_id": "integration-operation",
            "type": "operation.execute", "payload": {
                "operation_id": operation_id, "parameters": parameters or {}}}


def test_raw_bytes_survive_with_distinct_corrected_and_estimation_identities(investigation, inputs):
    summary = investigation["summary"]
    workspace = investigation["workspace"]
    assert summary["status"] == "completed"
    assert len(workspace["results"]) == len(workspace["executions"]) == 1
    run = workspace["run"]
    assert run["evidence_id"] == evidence_id(run)
    assert run["time_s"] == [0.0] and run["metadata"]["sample_rate_hz"] is None
    identities = {run["evidence_id"]}
    for provided, sensor in zip(inputs["sensors"], run["metadata"]["rci_source"]["sensors"]):
        original = base64.b64decode(provided["request"]["inputs"]["records"][0]["raw_record_b64"])
        record = sensor["measurement"]["records"][0]
        assert base64.b64decode(record["raw_record_b64"]) == original
        assert record["source_evidence_digest"] == hashlib.sha256(original).hexdigest()
        assert record["source_evidence_digest"] != record["derived_evidence_digest"]
        raw_channel = run["channels"][f"raw.{sensor['name']}"]
        calibrated = run["channels"][f"calibrated.{sensor['name']}"]
        assert raw_channel["kind"] == "observation"
        assert calibrated["kind"] == "calibrated_observation"
        identities.update((raw_channel["evidence_id"], calibrated["evidence_id"]))
    result = workspace["results"][0]
    identities.update((result["execution_id"], result["result_id"]))
    assert len(identities) == 7
    assert result["evidence_id"] == run["evidence_id"]
    assert result["verification_id"] is None
    assert result["verification_status"] == "not_verified"
    assert {r["state"] for r in summary["terminal_rows"]} >= {
        "observation", "calibrated_observation", "estimated_state", "residual", "diagnostic"}


def test_parameter_covariance_and_all_uncertainty_contributions_are_retained(investigation, inputs):
    sensors = investigation["workspace"]["run"]["metadata"]["rci_source"]["sensors"]
    for provided, sensor in zip(inputs["sensors"], sensors):
        uncertainty = sensor["measurement"]["uncertainty"]
        parameter_covariance = provided["request"]["inputs"]["calibration"]["parameter_covariance"]
        assert uncertainty["parameter_covariance"] == parameter_covariance
        assert parameter_covariance[0][1] != 0
        jacobian = np.asarray(uncertainty["parameter_jacobian"])
        expected = jacobian @ np.asarray(parameter_covariance) @ jacobian.T
        diagonalized = jacobian @ np.diag(np.diag(parameter_covariance)) @ jacobian.T
        assert not np.allclose(expected, diagonalized, rtol=1e-12, atol=0)
        np.testing.assert_allclose(uncertainty["parameter_contribution_covariance"], expected)
        total = sum(np.asarray(uncertainty[name]) for name in (
            "parameter_contribution_covariance", "raw_contribution_covariance", "residual_covariance"))
        np.testing.assert_allclose(uncertainty["output_covariance"], total)
        assert uncertainty["traceability"] == "none_claimed"
    assert investigation["workspace"]["run"]["metadata"]["rci_source"]["cross_assembly_independent"] is True


def test_domain_engines_execute_in_separate_pinned_processes(investigation):
    assert not any(name == package or name.startswith(package + ".")
                   for name in investigation["new_modules"] for package in ("instrument_chain", "set_lcm"))
    workspace = investigation["workspace"]
    rci = workspace["run"]["metadata"]["rci_source"]["runtime"]
    fsrt = workspace["executions"][0]["runtime"]
    pins = {role: __import__("ciw.pipelines", fromlist=["provider_descriptor"]).provider_descriptor(role)["pin"] for role in ("rci", "fsrt", "jspt", "gte")}
    assert rci["revision"] == pins["rci"]["revision"]
    assert fsrt["revision"] == pins["fsrt"]["revision"]
    assert rci["module"] == "instrument_chain.ciw_adapter"
    assert fsrt["module"] == "set_lcm.bridge.ciw"
    assert rci["python_sha256"] == fsrt["python_sha256"]


def test_offline_inspection_needs_no_domain_runtime_and_leaves_source_unchanged(investigation, monkeypatch):
    import ciw.investigation as module

    def no_runtime(*args, **kwargs):
        raise AssertionError("Read-only reopening attempted to execute a scientific runtime")

    monkeypatch.setattr(module, "_runtime", no_runtime)
    monkeypatch.delenv("CIW_RCI_REPO", raising=False)
    monkeypatch.delenv("CIW_FSRT_REPO", raising=False)
    source = investigation["path"]
    before = {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()}
    summary = inspect_investigation(source)
    assert summary["evidence_id"] == investigation["summary"]["evidence_id"]
    assert summary["results"] == investigation["summary"]["results"]
    assert {p.name: p.read_bytes() for p in source.parent.iterdir() if p.is_file()} == before


def test_offline_replay_retains_evidence_and_allocates_new_execution_and_result(investigation, sources, tmp_path):
    original = investigation["summary"]
    replayed = replay_investigation(investigation["path"], *sources, tmp_path / "replay",
                                     python_executable=sys.executable)
    assert replayed["status"] == "completed" and replayed["replay_data_digest_matches"] is True
    assert replayed["evidence_id"] == original["evidence_id"]
    assert replayed["run_id"] == original["run_id"]
    assert len(replayed["results"]) == 2
    assert replayed["results"][0] == original["results"][0]
    old, new = replayed["results"]
    assert new["execution_id"] != old["execution_id"]
    assert new["result_id"] != old["result_id"]
    assert digest(new["data"]) == digest(old["data"])
    assert new["verification_id"] is None and new["verification_status"] == "not_verified"


@pytest.mark.parametrize("state", ["missing", "expired", "mismatched"])
def test_invalid_calibration_creates_neither_result_nor_output_directory(state, inputs, sources, tmp_path):
    altered = deepcopy(inputs)
    # The mismatched second assembly proves the first successful calibration
    # does not publish a partial investigation.
    request = altered["sensors"][1 if state == "mismatched" else 0]["request"]["inputs"]
    if state == "missing":
        del request["calibration"]
    elif state == "expired":
        request["calibration"]["valid_until"] = "2026-01-02T00:00:00Z"
    else:
        request["calibration"]["installation_id"] = "unrelated-installation"
    output = tmp_path / state
    with pytest.raises(AdapterRefusal) as refusal:
        create_investigation(altered, *sources, output, python_executable=sys.executable)
    assert refusal.value.code == "calibration_unavailable"
    assert refusal.value.reason_code == {
        "missing": "calibration_missing", "expired": "calibration_expired",
        "mismatched": "calibration_mismatch",
    }[state]
    assert not output.exists()


def test_shared_session_executes_bound_provider_and_retains_unknown_provider_refusal(investigation, sources, tmp_path):
    session = Session.from_workspace(investigation["path"], tmp_path / "shared-session")
    bind_fsrt(session, sources[1], python_executable=sys.executable)
    success = session.handle(_request(FSRT_OPERATION))
    assert success["type"] == "response" and success["payload"]["status"] == "completed"
    assert success["payload"]["result"]["evidence_id"] == session.run["evidence_id"]
    count = len(session.results)
    result_files = {p.name for p in session.output_dir.glob("result-*.json")}
    refused = session.handle(_request("unbound-provider.v1"))
    assert refused["type"] == "response" and refused["payload"]["status"] == "refused"
    assert refused["payload"]["result"] is None
    execution = refused["payload"]["execution"]
    assert execution["refusal"]["code"] == "operation_unavailable"
    assert execution["execution_id"] in session.executions
    assert len(session.results) == count
    assert {p.name for p in session.output_dir.glob("result-*.json")} == result_files


def test_tampered_scientific_evidence_refuses_reopen_before_destination_writes(investigation, tmp_path):
    workspace = deepcopy(investigation["workspace"])
    workspace["run"]["channels"]["calibrated.tank-1"]["values"][0] += 1.0
    source = tmp_path / "tampered-evidence.json"
    source.write_text(json.dumps(workspace))
    output = tmp_path / "must-not-write"
    with pytest.raises(ValueError, match="integrity"):
        Session.from_workspace(source, output)
    assert not output.exists()

    # Rehashing the outer scientific record cannot make inconsistent retained
    # raw/derived source bindings acceptable to the offline reader.
    workspace = deepcopy(investigation["workspace"])
    sensor = workspace["run"]["metadata"]["rci_source"]["sensors"][0]
    raw = sensor["request"]["inputs"]["records"][0]
    raw["raw_record_b64"] = base64.b64encode(base64.b64decode(raw["raw_record_b64"]) + b" ").decode()
    workspace["run"]["evidence_id"] = evidence_id(workspace["run"])
    source = tmp_path / "tampered-source-binding.json"
    source.write_text(json.dumps(workspace))
    with pytest.raises(ValueError, match="raw observation bytes"):
        Session.from_workspace(source, output)
    assert not output.exists()


def test_tampered_result_or_missing_execution_refuses_reopen_before_writes(investigation, tmp_path):
    for corruption in ("result", "execution"):
        workspace = deepcopy(investigation["workspace"])
        if corruption == "result":
            workspace["results"][0]["data"]["estimate"]["values"][0] += 1.0
        else:
            workspace["executions"] = []
        source = tmp_path / (corruption + ".json")
        source.write_text(json.dumps(workspace))
        output = tmp_path / (corruption + "-output")
        with pytest.raises(ValueError):
            Session.from_workspace(source, output)
        assert not output.exists()


def test_resealed_result_still_must_satisfy_physical_schema_and_source_binding(investigation, tmp_path):
    corruptions = {
        "covariance_shape": lambda data: data["estimate"].__setitem__("covariance", [[1.0]]),
        "indefinite_covariance": lambda data: data["estimate"].__setitem__("covariance", [[1.0, 2.0], [2.0, 1.0]]),
        "physical_truth": lambda data: data["diagnostics"].__setitem__("physical_truth_verified", True),
        "negative_mass": lambda data: data["estimate"]["values"].__setitem__(0, -1.0),
        "different_model": lambda data: data["model"].__setitem__("total_mass_kg", 200.0),
        "different_source": lambda data: data["observation_evidence_ids"].__setitem__(0, "sha256:" + "0" * 64),
    }
    for label, corrupt in corruptions.items():
        workspace = deepcopy(investigation["workspace"])
        result = workspace["results"][0]
        corrupt(result["data"])
        result.pop("record_digest")
        seal(result)
        source = tmp_path / (label + ".json")
        source.write_text(json.dumps(workspace))
        output = tmp_path / (label + "-output")
        with pytest.raises(ValueError):
            Session.from_workspace(source, output)
        assert not output.exists()


def test_unicode_domain_metadata_preserves_native_rci_digests(inputs, sources, tmp_path):
    altered = deepcopy(inputs)
    request = altered["sensors"][0]["request"]["inputs"]
    request["assembly_toml"] = request["assembly_toml"].replace(
        'title = "Simulated reservoir', 'title = "Réservoir simulé — reservoir').replace(
        'sigma_reason = "Declared', 'sigma_reason = "Étalonnage simulé — declared')
    request["calibration"]["assembly_digest"] = hashlib.sha256(request["assembly_toml"].encode()).hexdigest()
    raw = json.loads(base64.b64decode(request["records"][0]["raw_record_b64"]))
    raw["sigma_reason"] = raw["sigma_reason"].replace("Declared", "Étalonnage simulé — declared")
    original_bytes = (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode()
    request["records"][0]["raw_record_b64"] = base64.b64encode(original_bytes).decode()
    summary = create_investigation(altered, *sources, tmp_path / "unicode",
                                   python_executable=sys.executable)
    assert summary["status"] == "completed"
    workspace = read_json(Path(summary["workspace_file"]))
    measurement = workspace["run"]["metadata"]["rci_source"]["sensors"][0]["measurement"]
    assert base64.b64decode(measurement["records"][0]["raw_record_b64"]) == original_bytes
    assert measurement["uncertainty"]["residual_sigma_reason"].startswith("Étalonnage simulé")
    # This exercises the native UTF-8 commitment; CIW's historical ASCII JSON
    # canonicalization is intentionally different and cannot verify this hash.
    assert digest(measurement["uncertainty"]) != measurement["uncertainty_digest"]
    assert inspect_investigation(Path(summary["workspace_file"]))["results"] == summary["results"]


def test_replay_preserves_last_model_override_and_physical_disagreement(investigation, sources, tmp_path):
    session = Session.from_workspace(investigation["path"], tmp_path / "override")
    bind_fsrt(session, sources[1], python_executable=sys.executable)
    original_model = deepcopy(session.run["metadata"]["model"])
    override = {**original_model, "total_mass_kg": 50.0}
    response = session.handle(_request(FSRT_OPERATION, {"model": override}))
    assert response["type"] == "response" and response["payload"]["status"] == "completed"
    result = response["payload"]["result"]
    diagnostics = result["data"]["diagnostics"]
    assert diagnostics["physical_model_status"] == "physical_model_disagreement"
    assert diagnostics["fault_attribution"] == "confounded_or_unidentifiable"
    assert diagnostics["reconciliation_status"] == "model_inconsistent"
    assert result["data"]["estimate"]["values"] == result["data"]["unprojected_estimate"]["values"]
    assert result["data"]["residuals"]["correction"] is None
    assert session.run["metadata"]["model"] == original_model
    workspace_path = session.save_workspace(session.output_dir / "workspace.json")
    replayed = replay_investigation(workspace_path, *sources, tmp_path / "override-replay",
                                     python_executable=sys.executable)
    assert replayed["status"] == "completed" and replayed["replay_data_digest_matches"] is True
    latest = replayed["results"][-1]
    assert latest["parameters"] == {"model": override}
    assert latest["data"]["model"] == override
    assert latest["data"]["diagnostics"] == diagnostics
    assert digest(latest["data"]) == digest(result["data"])
    assert latest["execution_id"] != result["execution_id"]
    assert latest["result_id"] != result["result_id"]
    assert latest["evidence_id"] == session.run["evidence_id"]
