import base64
import json
from pathlib import Path

import pytest

from ciw import cli, energy_records
from ciw.instruments import make_demo_run
from ciw.lab.bridge import classify_workspace
from ciw.lab.evidence import validate_finding
from ciw.session import Session

ROOT = Path(__file__).resolve().parents[1]


def _request(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "lab-bridge", "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lab-bridge")
    session = Session(make_demo_run(), directory / "out")
    _request(session, "operation.execute", {"operation_id": "statistics.v1",
                                            "parameters": {"channel": "v", "interval_s": [1.0, 2.0]}})
    synthetic = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text())
    physical = dict(synthetic, origin="physical_measurement", run_id="energy-run-" + "2" * 32)
    physical.pop("log_digest")
    for log in (synthetic, energy_records.seal(physical)):
        source = _request(session, "source.add", {"kind": "energy-accuracy", "label": log["origin"],
                                                  "bytes_b64": base64.b64encode(json.dumps(log).encode()).decode()})
        original = _request(session, "operation.execute", {"operation_id": "ciw.energy-accuracy.v1",
                                                           "parameters": {"source_id": source["source_id"]}})
        _request(session, "bundle.replay", {"bundle_id": original["bundle_id"]})
    return session.save_workspace(directory / "workspace.json")


def _labels(items, kind):
    return [[f["evidence_status"] for f in item["findings"]] for item in items if item["kind"] == kind]


def test_every_retained_result_receives_validated_labels(workspace):
    result = classify_workspace(workspace)
    assert result["validated_without_provider_execution"] is True
    for item in result["items"]:
        for record in item["findings"]:
            validate_finding(record)
    assert _labels(result["items"], "run") == [["synthetic", "not_established"]]
    assert _labels(result["items"], "operation_result") == [["synthetic", "not_established"]]
    bundles = _labels(result["items"], "workbench_bundle")
    assert ["synthetic", "not_established"] in bundles
    assert ["synthetic", "not_established", "numerically_verified"] in bundles
    assert ["not_established", "hardware_measured", "numerically_verified"] in bundles
    assert result["label_counts"]["independently_verified"] == 0
    assert result["label_counts"]["hardware_measured"] == 2


def test_physical_claims_need_a_declared_physical_log(workspace):
    result = classify_workspace(workspace)
    physical = [f for item in result["items"] for f in item["findings"] if f["domain"] == "physical"]
    measured = [f for f in physical if f["evidence_status"] == "hardware_measured"]
    assert len(physical) == 6 and len(measured) == 2
    for record in measured:
        acquisition = record["basis"]["acquisition"]
        assert acquisition["calibration"] == "vendor_counter_accuracy_not_declared"
        assert "not authenticated" in record["basis"]["notes"]


def test_classification_binds_no_provider_and_changes_nothing(workspace, monkeypatch):
    from ciw.adapters import subprocess as adapters

    before = workspace.read_bytes()
    monkeypatch.setattr(adapters.PinnedSubprocessAdapter, "__init__",
                        lambda *_, **__: pytest.fail("classification must not bind a provider"))
    classify_workspace(workspace)
    assert workspace.read_bytes() == before


def test_reopen_recomputes_the_builtin_energy_analysis(workspace, monkeypatch):
    # Recorded behavior: the energy-accuracy validator re-runs its offline log
    # analysis on reopen, so "validated without execution" would overstate it.
    calls = []
    original = energy_records.analyze
    monkeypatch.setattr(energy_records, "analyze", lambda log: calls.append(1) or original(log))
    classify_workspace(workspace)
    assert calls


def test_tampered_workspace_is_refused_before_labelling(workspace, tmp_path):
    saved = json.loads(workspace.read_text())
    saved["run"]["channels"]["v"]["values"][0] += 1.0
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        classify_workspace(forged)


def test_cli_classify(workspace, capsys):
    assert cli.main(["lab", "classify", str(workspace)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "ciw.lab-workspace-classification.v1"
