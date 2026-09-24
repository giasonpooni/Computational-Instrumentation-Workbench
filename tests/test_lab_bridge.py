import base64
import json
from pathlib import Path

import pytest

from ciw import cli, energy_records
from ciw.core.identities import evidence_id
from ciw.instruments import make_demo_run
from ciw.lab import bridge
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
    synthetic = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text(encoding="utf-8"))
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
    assert ["not_established", "not_established", "numerically_verified"] in bundles
    assert result["label_counts"]["independently_verified"] == 0
    assert result["label_counts"]["hardware_measured"] == 0


def test_a_resealed_log_that_calls_itself_physical_is_not_hardware_measured(workspace):
    # The fixture above is the synthetic baseline with origin changed to
    # physical_measurement and resealed: the seal is an unkeyed self-digest.
    result = classify_workspace(workspace)
    physical = [f for item in result["items"] for f in item["findings"] if f["domain"] == "physical"]
    assert len(physical) == 6 and {f["evidence_status"] for f in physical} == {"not_established"}
    declared = [f for f in physical if "physical_measurement" in f["basis"].get("notes", "")]
    assert len(declared) == 2 and all("acquisition" not in f["basis"] and f["value"] is None for f in declared)
    assert "unkeyed" in result["origin_authentication"] and "not authenticated" in result["origin_authentication"]


def _physical_log(**changes):
    log = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text())
    log.update(origin="physical_measurement", raw_sha256="ab" * 32)
    log["sensor"] = dict(log["sensor"], name="bench device", driver_version="550.54", nvml_version="12.550")
    log["clock"] = dict(log["clock"], epoch_id="boot-7", implementation="time.monotonic_ns")
    log["runtime"] = dict(log["runtime"], implementation={"profile": "ciw.fixed-gaussian-gpu-energy.v1", "code_sha256": "d" * 64})
    log["runtime"]["workload"] = dict(log["runtime"]["workload"], device_name="bench device")
    return dict(log, **changes)


def test_hardware_label_needs_retained_raw_bytes_and_no_generated_declaration():
    log = _physical_log()
    acquisition = bridge._energy_acquisition(log, {"ab" * 32})
    assert acquisition["raw_sha256"] == "ab" * 32 and acquisition["device"].startswith("nvml:GPU-")
    assert acquisition["calibration"] == "vendor_counter_accuracy_not_declared"
    assert bridge._energy_acquisition(log, set()) is None                       # raw bytes not in the workspace
    assert bridge._energy_acquisition(dict(log, raw_sha256=None), {"ab" * 32}) is None
    assert bridge._energy_acquisition(dict(log, origin="synthetic_fixture"), {"ab" * 32}) is None
    assert bridge._energy_acquisition(dict(log, provenance={"generator": "ciw.fake"}), {"ab" * 32}) is None
    fixture = dict(log, sensor=dict(log["sensor"], name="synthetic fixture device"))
    assert bridge._energy_acquisition(fixture, {"ab" * 32}) is None
    no_calibration = dict(log, sensor={k: v for k, v in log["sensor"].items() if k != "accuracy_j"})
    assert bridge._energy_acquisition(no_calibration, {"ab" * 32}) is None
    no_clock = dict(log, clock={k: v for k, v in log["clock"].items() if k != "epoch_id"})
    assert bridge._energy_acquisition(no_clock, {"ab" * 32}) is None


def _run_workspace(directory, provenance):
    run = make_demo_run()
    run["metadata"]["provenance"] = provenance
    run["evidence_id"] = evidence_id(run)
    session = Session(run, directory / "out")
    _request(session, "operation.execute", {"operation_id": "statistics.v1",
                                            "parameters": {"channel": "v", "interval_s": [1.0, 2.0]}})
    return session.save_workspace(directory / "workspace.json")


@pytest.mark.parametrize("provenance, label", [
    ({"generator": "Keysight 33500B function generator driving the shaker", "source": "accelerometer recording"},
     "not_established"),
    ({"source": "synthetic aperture radar acquisition, field campaign 2026-03"}, "not_established"),
    ({"source": "bench accelerometer recording"}, "not_established"),
    # Acquisition code named as the generator: a dotted identity is not a CIW generator identity.
    ({"generator": "shaker_capture.py", "source": "bench accelerometer recording"}, "not_established"),
    ({"generator": "pymeasure.instruments.agilent.Agilent33500", "source": "accelerometer recording"},
     "not_established"),
    ({"generator": "NI.DAQmx", "source": "bench accelerometer recording"}, "not_established"),
    ({"source": "numpy draws", "generator": "numpy.random.default_rng"}, "not_established"),
    ({"source": "demo", "generator": "ciw.instruments.make_demo_run", "generator_version": 1}, "synthetic"),
])
def test_synthetic_needs_a_structured_generator_declaration(tmp_path, provenance, label):
    result = classify_workspace(_run_workspace(tmp_path, provenance))
    assert _labels(result["items"], "run") == [[label, "not_established"]]
    assert _labels(result["items"], "operation_result") == [[label, "not_established"]]


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
    saved = json.loads(workspace.read_text(encoding="utf-8"))
    saved["run"]["channels"]["v"]["values"][0] += 1.0
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity"):
        classify_workspace(forged)


def test_cli_classify(workspace, capsys):
    assert cli.main(["lab", "classify", str(workspace)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "ciw.lab-workspace-classification.v1"
