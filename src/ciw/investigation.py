"""RCI acquisition/calibration and FSRT estimation through the shared session.

This module maps records and identities only. Domain transforms, covariance
propagation, estimation and physical checks remain in the pinned repositories.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from .adapters.protocol import AdapterRefusal, InstrumentManifest
from .adapters.subprocess import PinnedSubprocessAdapter
from .core.identities import digest, evidence_id
from .operations.registry import Operation
from .session import Session, read_json


RCI_OPERATION = "rci.calibrate.v1"
FSRT_OPERATION = "fsrt.tank-reconstruct.v1"


def _runtime(name, repo, python_executable=None, expected=None):
    pins = json.loads(files("ciw").joinpath("adapter-runtimes.json").read_text())
    spec = pins[name]
    kwargs = {}
    if expected:
        if expected["revision"] != spec["revision"] or expected["module"] != spec["module"]:
            raise AdapterRefusal("runtime_mismatch", "Saved runtime does not match the supported adapter pin")
        kwargs = {"expected_python_sha256": expected["python_sha256"],
                  "expected_python_version": expected["python_version"],
                  "expected_dependencies": expected["dependencies"]}
    return PinnedSubprocessAdapter(Path(repo), spec["revision"], spec["module"],
                                   python_executable=python_executable or sys.executable, **kwargs)


def _rci_digest(value):
    # RCI canonicalizes UTF-8 literally; preserve its native commitments rather
    # than applying CIW's historical ASCII-escaped JSON encoding to them.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _validate_batch(batch, request):
    if batch.get("schema") != "measurement-record-batch.v1" or len(batch.get("records", [])) != 1:
        raise AdapterRefusal("unsupported_measurement_batch", "The first tank slice requires one record per assembly")
    record = batch["records"][0]
    if record.get("raw_record_b64") != request["inputs"]["records"][0]["raw_record_b64"]:
        raise ValueError("Adapter changed raw observation bytes")
    raw = base64.b64decode(record["raw_record_b64"], validate=True)
    native = json.loads(raw)
    if (record["raw"] != {"value": native["raw"], "unit": native["raw_unit"]}
            or record["observation_id"] != native["observation_id"]
            or record["observed_at"] != request["inputs"]["records"][0]["observed_at"]):
        raise ValueError("Measurement record does not match original raw observation or acquisition time")
    if hashlib.sha256(raw).hexdigest() != record["source_evidence_digest"]:
        raise ValueError("Raw observation evidence mismatch")
    if _rci_digest({k: v for k, v in record.items() if k != "derived_evidence_digest"}) != record["derived_evidence_digest"]:
        raise ValueError("Derived measurement evidence mismatch")
    if record["source_evidence_digest"] == record["derived_evidence_digest"]:
        raise ValueError("Raw and corrected evidence identities must differ")
    if _rci_digest(batch["uncertainty"]) != batch["uncertainty_digest"]:
        raise ValueError("Uncertainty evidence mismatch")
    if record["uncertainty_digest"] != batch["uncertainty_digest"] or record["covariance_row"] != 0:
        raise ValueError("Measurement uncertainty binding mismatch")
    if (record.get("calibration_state") != "applicable"
            or batch["calibration"] != request["inputs"]["calibration"]
            or _rci_digest(batch["calibration"]) != batch["calibration_digest"]
            or record["calibration_digest"] != batch["calibration_digest"]):
        raise ValueError("Calibration source binding mismatch")
    if (batch["assembly_toml"] != request["inputs"]["assembly_toml"]
            or hashlib.sha256(batch["assembly_toml"].encode()).hexdigest() != batch["assembly_digest"]
            or record["assembly_digest"] != batch["assembly_digest"]):
        raise ValueError("Assembly source binding mismatch")
    covariance = batch["uncertainty"]["output_covariance"]
    if len(covariance) != 1 or len(covariance[0]) != 1:
        raise ValueError("Single-record output covariance must be 1 by 1")


def _validate_source(run):
    try:
        return _validate_source_fields(run)
    except (KeyError, TypeError, IndexError, UnicodeError) as exc:
        raise ValueError("Invalid retained RCI source structure") from exc


def _validate_source_fields(run):
    source = run["metadata"]["rci_source"]
    if source.get("schema") != "ciw.rci-source.v1" or source.get("cross_assembly_independent") is not True:
        raise AdapterRefusal("unsupported_covariance", "Cross-assembly independence must be explicitly declared")
    sensors = source.get("sensors")
    if not isinstance(sensors, list) or len(sensors) != 2:
        raise ValueError("Exactly two independent mass assemblies are required")
    names = [sensor.get("name") for sensor in sensors]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 2:
        raise ValueError("Retained sensor names must be distinct nonempty strings")
    if set(run["channels"]) != {f"{kind}.{name}" for name in names for kind in ("raw", "calibrated")}:
        raise ValueError("Retained channels do not match the two-assembly source")
    if run["time_s"] != [0.0] or run["metadata"]["sample_count"] != 1:
        raise ValueError("Retained RCI tank slice must describe one simultaneous observation")
    for sensor in sensors:
        _validate_batch(sensor["measurement"], sensor["request"])
        for state, key in (("observation", "raw"), ("calibrated_observation", "calibrated")):
            channel = run["channels"][f"{key}.{sensor['name']}"]
            record = sensor["measurement"]["records"][0]
            expected_id = "sha256:" + record["source_evidence_digest" if key == "raw" else "derived_evidence_digest"]
            if (channel["values"] != [record[key]["value"]] or channel["unit"] != record[key]["unit"]
                    or channel.get("kind") != state or channel.get("evidence_id") != expected_id):
                raise ValueError("Channel does not match retained source evidence")
    records = [s["measurement"]["records"][0] for s in sensors]
    bindings = [s["measurement"]["calibration"] for s in sensors]
    for field in ("assembly_id", "installation_id", "calibration_id"):
        if bindings[0][field] == bindings[1][field]:
            raise AdapterRefusal("unsupported_covariance", f"Independent assemblies need distinct {field}")
    if records[0]["observed_at"] != records[1]["observed_at"]:
        raise AdapterRefusal("unsupported_temporal_covariance", "First FSRT operation requires simultaneous observations")
    return source


def _make_run(inputs, sensors, runtime):
    channels = {}
    for sensor in sensors:
        record = sensor["measurement"]["records"][0]
        for key, kind, evidence_field in (("raw", "observation", "source_evidence_digest"),
                                          ("calibrated", "calibrated_observation", "derived_evidence_digest")):
            channels[f"{key}.{sensor['name']}"] = {
                "unit": record[key]["unit"], "values": [record[key]["value"]], "kind": kind,
                "evidence_id": "sha256:" + record[evidence_field],
                "calibration_state": "not_applied" if key == "raw" else record["calibration_state"],
            }
    manifest = InstrumentManifest(
        instrument_id="org.notationsystems.rci", version="1", role="measurement_adapter",
        inputs=("measurement-record.v1",), outputs=("run.v1",),
        units={name: channel["unit"] for name, channel in channels.items()}, frames=("two-reservoir-mass",),
        sampling={"mode": "single_simultaneous_event", "duration_s_semantics": "selection support only; not acquisition cadence"},
        normalization={"raw": "unchanged", "calibrated": "declared RCI transform"},
        supported_operations=(), determinism={"pinned_runtime": True, "offline_replay": True},
        tolerance_policy={"rci_records": "canonical digest equality", "fsrt": "same pinned runtime; digest equality"},
        calibration_requirements={"binding": "assembly/version/installation/digest", "validity": "at acquisition", "parameter_covariance": "retained"},
    )
    run = {"run_schema": "run.v1", "run_id": "run-" + uuid.uuid4().hex,
           "instrument": manifest.instrument_id, "time_s": [0.0], "channels": channels, "render": {},
           "metadata": {"sample_count": 1, "sample_rate_hz": None, "duration_s": 1.0,
                        "coordinate_frame": "two-reservoir-mass", "manifest": manifest.to_dict(),
                        "model": inputs["model"],
                        "provenance": {"source": inputs["source_description"], "time_reference": "simultaneous acquisition instant; 1 s selection support", "claim_scope": "declared calibration and model only"},
                        "rci_source": {"schema": "ciw.rci-source.v1", "cross_assembly_independent": inputs["cross_assembly_independent"],
                                       "runtime": runtime, "sensors": sensors}}}
    _validate_source(run)
    run["evidence_id"] = evidence_id(run)
    return run


def _fsrt_inputs(run, parameters):
    if set(parameters) - {"model"}:
        raise AdapterRefusal("invalid_parameters", "Only an explicit model is accepted; observations come from retained evidence")
    source = _validate_source(run)
    records = [s["measurement"]["records"][0] for s in source["sensors"]]
    if any(record["calibrated"]["unit"] != "kg" for record in records):
        raise AdapterRefusal("unit_mismatch", "The two-reservoir model requires calibrated masses in kg")
    covariance = [[0.0, 0.0], [0.0, 0.0]]
    for i, sensor in enumerate(source["sensors"]):
        covariance[i][i] = sensor["measurement"]["uncertainty"]["output_covariance"][0][0]
    return {"model": copy.deepcopy(parameters.get("model", run["metadata"]["model"])),
            "observations": [{"t": 0, "arrival_t": 0,
                "values": [r["calibrated"]["value"] for r in records], "covariance": covariance,
                "mask": [True, True], "source_ids": [s["name"] for s in source["sensors"]],
                "evidence_ids": ["sha256:" + r["derived_evidence_digest"] for r in records], "unit": "kg"}]}


def bind_fsrt(session, repo, python_executable=None, expected=None):
    adapter = _runtime("fsrt", repo, python_executable, expected)
    session.operations.register(Operation(FSRT_OPERATION, "state_estimator",
        lambda run, parameters: adapter.invoke(FSRT_OPERATION, _fsrt_inputs(run, parameters)),
        adapter.runtime_identity))
    return adapter


def _summary(session, path):
    rows = []
    for name, channel in session.run["channels"].items():
        rows.append({"state": channel.get("kind", "observation"), "quantity": name,
                     "value": channel["values"], "unit": channel["unit"], "evidence_id": channel.get("evidence_id", session.run["evidence_id"])})
    results = list(session.results.values())
    for result in results:
        data = result["data"]
        if result["operation_id"] == FSRT_OPERATION:
            rows.append({"state": "estimated_state", "quantity": "reservoir masses", "value": data["estimate"]["values"], "unit": "kg", "evidence_id": result["result_id"]})
            rows.append({"state": "residual", "quantity": "physical balance before reconciliation", "value": data["residuals"].get("balance_before"), "unit": "kg", "evidence_id": result["result_id"]})
            rows.append({"state": "diagnostic", "quantity": "physical model / fault attribution", "value": data["diagnostics"]["physical_model_status"] + " / " + data["diagnostics"]["fault_attribution"], "unit": "—", "evidence_id": result["result_id"]})
    refusals = [e for e in session.executions.values() if e["status"] == "refused"]
    latest = next(reversed(session.executions.values()), None) if session.executions else None
    evaluated_at = datetime.now(timezone.utc)
    calibration = []
    for sensor in session.run["metadata"].get("rci_source", {}).get("sensors", []):
        profile = sensor["measurement"]["calibration"]
        calibration.append({"source": sensor["name"], "applicable_at_acquisition": True,
                            "expired": evaluated_at >= datetime.fromisoformat(profile["valid_until"]),
                            "evaluated_at": evaluated_at.isoformat(), "valid_until": profile["valid_until"]})
    return {"calibration": calibration, "status": "refused" if latest and latest["status"] == "refused" else "completed",
            "workspace_file": str(path), "run_id": session.run["run_id"], "evidence_id": session.run["evidence_id"],
            "results": copy.deepcopy(results), "refusals": copy.deepcopy(refusals), "terminal_rows": rows,
            "execution_ids": list(session.executions),
            "note": "Declared calibration and model evaluation; no physical verification or traceability conferred."}


def _evaluate_and_save(session, path, parameters=None):
    response = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                               "type": "operation.execute", "payload": {"operation_id": FSRT_OPERATION, "parameters": parameters if parameters is not None else {"model": session.run["metadata"]["model"]}}})
    if response["type"] == "error":
        raise ValueError(response["payload"]["message"])
    session.save_workspace(path)
    return _summary(session, path)


def create_investigation(inputs, rci_repo, fsrt_repo, output_dir, python_executable=None):
    required = {"schema", "sensors", "cross_assembly_independent", "model", "source_description"}
    if not isinstance(inputs, dict) or set(inputs) != required or inputs["schema"] != "ciw.tank-investigation-input.v1":
        raise ValueError("Expected ciw.tank-investigation-input.v1 with sensors, model, independence and source description")
    if not isinstance(inputs["sensors"], list) or len(inputs["sensors"]) != 2:
        raise ValueError("Exactly two sensor requests are required")
    if (not isinstance(inputs["model"], dict) or not isinstance(inputs["source_description"], str)
            or not inputs["source_description"].strip()):
        raise ValueError("An explicit model and evidence source description are required")
    if inputs["cross_assembly_independent"] is not True:
        raise AdapterRefusal("unsupported_covariance", "Independent calibration sources must be explicitly declared")
    for sensor in inputs["sensors"]:
        if not isinstance(sensor, dict) or set(sensor) != {"name", "request"} or not isinstance(sensor["request"], dict):
            raise ValueError("Each sensor needs a name and explicit RCI request")
    names = [s.get("name") for s in inputs["sensors"]]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 2:
        raise ValueError("Two distinct sensor names are required")
    adapter = _runtime("rci", rci_repo, python_executable)
    sensors = []
    # No workspace or result is written unless all calibrations are applicable.
    for sensor in inputs["sensors"]:
        request = sensor["request"]
        if request.get("schema") != "ciw.adapter-request.v1" or request.get("operation_id") != RCI_OPERATION:
            raise ValueError("Expected an explicit rci.calibrate.v1 request")
        measurement = adapter.invoke(RCI_OPERATION, request["inputs"])
        _validate_batch(measurement, request)
        sensors.append({"name": sensor["name"], "request": copy.deepcopy(request), "measurement": measurement})
    run = _make_run(inputs, sensors, adapter.runtime_identity())
    session = Session(run, Path(output_dir))
    bind_fsrt(session, fsrt_repo, python_executable)
    return _evaluate_and_save(session, Path(output_dir) / "workspace.json")


def inspect_investigation(path):
    # Restore validation never imports/executes a domain provider, and never
    # writes into the source investigation during read-only inspection.
    with tempfile.TemporaryDirectory(prefix="ciw-inspect-") as directory:
        session = Session.from_workspace(Path(path), Path(directory))
        _validate_source(session.run)
        return _summary(session, path)


def replay_investigation(path, rci_repo, fsrt_repo, output_dir, python_executable=None):
    # Validate the whole saved object before invoking any scientific code.
    with tempfile.TemporaryDirectory(prefix="ciw-replay-validate-") as directory:
        original = Session.from_workspace(Path(path), Path(directory))
        source = _validate_source(original.run)
        rci = _runtime("rci", rci_repo, python_executable, source["runtime"])
        for sensor in source["sensors"]:
            actual = rci.invoke(RCI_OPERATION, sensor["request"]["inputs"])
            if digest(actual) != digest(sensor["measurement"]):
                raise AdapterRefusal("replay_mismatch", "Replayed calibration differs from retained derived evidence")
        successful = [r for r in original.results.values() if r["operation_id"] == FSRT_OPERATION]
        previous = [e for e in original.executions.values() if e["operation_id"] == FSRT_OPERATION]
        if not previous:
            raise AdapterRefusal("replay_unavailable", "No FSRT invocation retained for replay")
        expected = previous[-1]["runtime"]
        parameters = previous[-1]["parameters"]
        fsrt = _runtime("fsrt", fsrt_repo, python_executable, expected)
        # Pin validation precedes any writes into the requested destination.
        fsrt.runtime_identity()
    session = Session.from_workspace(Path(path), Path(output_dir))
    bind_fsrt(session, fsrt_repo, python_executable, expected)
    summary = _evaluate_and_save(session, Path(output_dir) / "workspace.json", parameters)
    if successful and previous[-1]["status"] == "completed" and summary["status"] == "completed":
        comparison = digest(successful[-1]["data"]) == digest(summary["results"][-1]["data"])
        summary["replay_data_digest_matches"] = comparison
        if not comparison:
            raise AdapterRefusal("replay_mismatch", "New FSRT result differs from retained data; both results were retained")
    return summary
