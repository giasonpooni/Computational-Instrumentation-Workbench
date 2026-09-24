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
from importlib.resources import files
from pathlib import Path

from .adapters.protocol import AdapterRefusal, InstrumentManifest
from .adapters.subprocess import PinnedSubprocessAdapter
from .core.identities import digest, evidence_id
from .operations.registry import Operation
from .session import Session, _reject_constant


RCI_OPERATION = "rci.calibrate.v1"
FSRT_OPERATION = "fsrt.tank-reconstruct.v1"
RCI_OPERATION_V2 = "rci.calibrate.v2"
FSRT_OPERATION_V2 = "fsrt.tank-reconstruct.v2"


def _runtime(name, repo, python_executable=None, expected=None):
    pins = json.loads(files("ciw").joinpath("adapter-runtimes.json").read_text())
    spec = pins[name]
    kwargs = {}
    if expected:
        allowed = [spec, *spec.get("historical", [])]
        matches = [entry for entry in allowed if expected["revision"] == entry["revision"]
                   and expected["module"] == entry["module"]]
        if not matches:
            raise AdapterRefusal("runtime_mismatch", "Saved runtime does not match the supported adapter pin")
        spec = matches[0]
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
    version = "v2" if request.get("operation_id") == RCI_OPERATION_V2 else "v1"
    if batch.get("schema") != f"measurement-record-batch.{version}" or len(batch.get("records", [])) != 1:
        raise AdapterRefusal("unsupported_measurement_batch", "The first tank slice requires one record per assembly")
    record = batch["records"][0]
    if record.get("raw_record_b64") != request["inputs"]["records"][0]["raw_record_b64"]:
        raise ValueError("Adapter changed raw observation bytes")
    raw = base64.b64decode(record["raw_record_b64"], validate=True)
    native = json.loads(raw, parse_constant=_reject_constant)
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
    if version == "v2":
        from .adapters.rci_records import validate_v2_provenance
        validate_v2_provenance(batch, request)
        if record.get("schema") != "measurement-record.v2":
            raise ValueError("Expected v2 calibrated record")
        basis = batch["calibration"].get("covariance_basis")
        if not isinstance(basis, dict) or batch["uncertainty"].get("covariance_basis") != basis:
            raise ValueError("Covariance basis must remain bound to its profile")
        # Full scientific consistency remains the domain provider's responsibility;
        # inspection verifies the retained shape, provenance and source relations.
        _measurement_covariance(batch, "sample")


def _validate_source(run):
    try:
        return _validate_source_fields(run)
    except (KeyError, TypeError, IndexError, UnicodeError) as exc:
        raise ValueError("Invalid retained RCI source structure") from exc


def _validate_source_fields(run):
    source = run["metadata"]["rci_source"]
    if source.get("schema") not in {"ciw.rci-source.v1", "ciw.rci-source.v2"} or source.get("cross_assembly_independent") is not True:
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
        required_operation = RCI_OPERATION_V2 if source["schema"] == "ciw.rci-source.v2" else RCI_OPERATION
        if sensor["request"].get("operation_id") != required_operation:
            raise ValueError("Calibration operation and retained source versions disagree")
        _validate_batch(sensor["measurement"], sensor["request"])
        if source["schema"] == "ciw.rci-source.v2":
            if sensor["request"].get("operation_id") != RCI_OPERATION_V2:
                raise ValueError("The v2 source requires v2 calibration")
            if sensor.get("covariance") != _measurement_covariance(sensor["measurement"], sensor["name"]):
                raise ValueError("Retained covariance artifact differs from calibrated evidence")
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
    if source["schema"] == "ciw.rci-source.v2":
        _validate_model_independence(source.get("model_independence"))
        left, right = (_dependencies(sensor["measurement"]["calibration"]["covariance_basis"]) for sensor in sensors)
        if left & right:
            raise AdapterRefusal("unsupported_cross_assembly_dependence",
                                 "Declared independent assemblies share uncertainty sources; an explicit joint model is required")
        from .calibration_status import calibration_status
        calibration_status(run)
    return source


def _validate_model_independence(value):
    flags = {"prior_independent_of_observations", "declared_total_independent_of_observations", "prior_independent_of_declared_total"}
    if (not isinstance(value, dict) or set(value) != flags | {"reason"}
            or any(value[key] is not True for key in flags)
            or not isinstance(value["reason"], str) or not value["reason"].strip()):
        raise AdapterRefusal("unsupported_model_dependence", "V2 requires an explicit reason and mutually independent prior, observations, and declared total")


def _dependencies(basis):
    """Identifiers declare dependence; distinct identifiers never prove independence."""
    result = set()
    blocks = [*basis.get("parameter_components", []), basis.get("raw", {}), basis.get("residual", {})]
    for block in blocks:
        if block.get("status") == "excluded" and block.get("represented_by") is None:
            continue
        for name in ("dependency_ids", "shared_source_ids"):
            items = block.get(name)
            if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
                raise ValueError("Covariance provenance requires explicit dependency and shared-source lists")
            result.update(items)
    return result


def _measurement_covariance(batch, name):
    from .core.covariance import create_covariance_artifact
    record = batch["records"][0]
    uncertainty = batch["uncertainty"]
    basis = batch["calibration"]["covariance_basis"]
    return create_covariance_artifact(
        matrix=uncertainty["output_covariance"], quantity_ids=[name], units=[record["calibrated"]["unit"]],
        frame="reservoir2.mass", reference_values=[record["calibrated"]["value"]],
        method=uncertainty["method"], basis={"kind": "calibrated_observation", "id": "sha256:" + batch["uncertainty_digest"]},
        provenance={"provider": "rci.calibrate.v2", "source_evidence_ids": ["sha256:" + record["derived_evidence_digest"]],
                    "source_covariance_ids": [], "metadata": {"covariance_basis": basis,
                    "shared_dependencies": sorted(_dependencies(basis)), "calibration_digest": batch["calibration_digest"],
                    "covariance_coverage": uncertainty.get("covariance_coverage"), "parameterization": uncertainty.get("parameterization")}},
        assumptions=["First-order native scale/zero calibration", "Declared raw/parameter and residual independence", "No traceability conferred by this artifact"],
    )


def _make_run(inputs, sensors, runtime):
    is_v2 = inputs["schema"] == "ciw.tank-investigation-input.v2"
    if is_v2:
        for sensor in sensors:
            sensor["covariance"] = _measurement_covariance(sensor["measurement"], sensor["name"])
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
        instrument_id="org.notationsystems.rci", version="2" if is_v2 else "1", role="measurement_adapter",
        inputs=("measurement-record.v2" if is_v2 else "measurement-record.v1",), outputs=("run.v1",),
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
                        "rci_source": {"schema": "ciw.rci-source.v2" if is_v2 else "ciw.rci-source.v1", "cross_assembly_independent": inputs["cross_assembly_independent"],
                                       "runtime": runtime, "sensors": sensors}}}
    if is_v2:
        run["metadata"]["rci_source"]["model_independence"] = copy.deepcopy(inputs["model_independence"])
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
    operation = fsrt_operation(session.run)
    session.operations.register(Operation(operation, "state_estimator",
        lambda run, parameters: adapter.invoke(operation, _fsrt_inputs_v2(run, parameters) if operation == FSRT_OPERATION_V2 else _fsrt_inputs(run, parameters)),
        adapter.runtime_identity))
    return adapter


def fsrt_operation(run):
    return FSRT_OPERATION_V2 if run["metadata"].get("rci_source", {}).get("schema") == "ciw.rci-source.v2" else FSRT_OPERATION


def _fsrt_inputs_v2(run, parameters):
    from .core.covariance import create_covariance_artifact
    inputs = _fsrt_inputs(run, parameters)
    sensors = run["metadata"]["rci_source"]["sensors"]
    observation = inputs["observations"][0]
    independence = run["metadata"]["rci_source"]["model_independence"]
    inputs["state_order"] = ["tank-1.mass", "tank-2.mass"]
    inputs["observation_covariance"] = create_covariance_artifact(
        matrix=observation["covariance"], quantity_ids=observation["source_ids"], units=["kg", "kg"],
        frame="reservoir2.mass", reference_values=observation["values"], method="declared_independent_assembly_blocks",
        basis={"kind": "calibrated_observation", "id": run["evidence_id"]},
        provenance={"provider": "ciw.rci-source.v2", "source_evidence_ids": observation["evidence_ids"],
                    "source_covariance_ids": [sensor["covariance"]["covariance_id"] for sensor in sensors],
                    "metadata": {"shared_dependencies": sorted(set().union(*[_dependencies(s["measurement"]["calibration"]["covariance_basis"]) for s in sensors])),
                                 **copy.deepcopy(independence),
                                 "uncertainty_context": [{"source": s["name"], "covariance_basis": s["measurement"]["calibration"]["covariance_basis"],
                                      "covariance_coverage": s["measurement"]["uncertainty"]["covariance_coverage"],
                                      "parameterization": s["measurement"]["uncertainty"]["parameterization"]} for s in sensors]}},
        assumptions=["Caller explicitly declares independent assemblies; distinct IDs alone do not prove independence",
                     "Prior, observations, and declared total are mutually independent under the declared model"],
    )
    return inputs


def _summary(session, path, evaluated_at=None):
    rows = []
    for name, channel in session.run["channels"].items():
        rows.append({"state": channel.get("kind", "observation"), "quantity": name,
                     "value": channel["values"], "unit": channel["unit"], "evidence_id": channel.get("evidence_id", session.run["evidence_id"])})
    results = list(session.results.values())
    for result in results:
        data = result["data"]
        if result["operation_id"] in {FSRT_OPERATION, FSRT_OPERATION_V2}:
            rows.append({"state": "estimated_state", "quantity": "reservoir masses", "value": data["estimate"]["values"], "unit": "kg", "evidence_id": result["result_id"]})
            rows.append({"state": "residual", "quantity": "physical balance before reconciliation", "value": data["residuals"].get("balance_before"), "unit": "kg", "evidence_id": result["result_id"]})
            rows.append({"state": "diagnostic", "quantity": "physical model / fault attribution", "value": data["diagnostics"]["physical_model_status"] + " / " + data["diagnostics"]["fault_attribution"], "unit": "—", "evidence_id": result["result_id"]})
    refusals = [e for e in session.executions.values() if e["status"] == "refused"]
    latest = next(reversed(session.executions.values()), None) if session.executions else None
    from .calibration_status import calibration_status
    calibration = calibration_status(session.run, evaluated_at)
    # Retain the first slice's summary aliases; authoritative status also carries
    # explicit acquisition and serving provenance shared with live read APIs.
    for item in calibration:
        item.update(applicable_at_acquisition=item["acquisition"]["applicable_at_acquisition"],
                    expired=item["serving"]["expired"], evaluated_at=item["serving"]["evaluated_at"])
    covariances = []
    for sensor in session.run["metadata"].get("rci_source", {}).get("sensors", []):
        if "covariance" in sensor:
            covariances.append({"name": "calibrated." + sensor["name"], **copy.deepcopy(sensor["covariance"])})
    for result in results:
        for name, artifact in result["data"].get("covariance_artifacts", {}).items():
            if artifact is not None:
                covariances.append({"name": name, "result_id": result["result_id"], **copy.deepcopy(artifact)})
        if "output_covariance" in result["data"] and result["operation_id"] == "jspt.covariance-propagate.v1":
            covariances.append({"name": "propagated", "result_id": result["result_id"], **copy.deepcopy(result["data"]["output_covariance"])})
    return {"calibration": calibration, "status": "refused" if latest and latest["status"] == "refused" else "completed",
            "workspace_file": str(path), "run_id": session.run["run_id"], "evidence_id": session.run["evidence_id"],
            "results": copy.deepcopy(results), "refusals": copy.deepcopy(refusals), "terminal_rows": rows, "covariances": covariances,
            "execution_ids": list(session.executions),
            "note": "Declared calibration and model evaluation; no physical verification or traceability conferred."}


def _evaluate_and_save(session, path, parameters=None):
    response = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                               "type": "operation.execute", "payload": {"operation_id": fsrt_operation(session.run), "parameters": parameters if parameters is not None else {"model": session.run["metadata"]["model"]}}})
    if response["type"] == "error":
        raise ValueError(response["payload"]["message"])
    session.save_workspace(path)
    return _summary(session, path)


def create_investigation(inputs, rci_repo, fsrt_repo, output_dir, python_executable=None):
    required = {"schema", "sensors", "cross_assembly_independent", "model", "source_description"}
    if isinstance(inputs, dict) and inputs.get("schema") == "ciw.tank-investigation-input.v2":
        required.add("model_independence")
        _validate_model_independence(inputs.get("model_independence"))
    if not isinstance(inputs, dict) or set(inputs) != required or inputs["schema"] not in {"ciw.tank-investigation-input.v1", "ciw.tank-investigation-input.v2"}:
        raise ValueError("Expected a supported versioned tank input with sensors, model, independence and source description")
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
    operation = RCI_OPERATION_V2 if inputs["schema"] == "ciw.tank-investigation-input.v2" else RCI_OPERATION
    sensors = []
    # No workspace or result is written unless all calibrations are applicable.
    for sensor in inputs["sensors"]:
        request = sensor["request"]
        if request.get("schema") != "ciw.adapter-request.v1" or request.get("operation_id") != operation:
            raise ValueError("Expected an explicit calibration request matching the investigation version")
        measurement = adapter.invoke(operation, request["inputs"])
        _validate_batch(measurement, request)
        sensors.append({"name": sensor["name"], "request": copy.deepcopy(request), "measurement": measurement})
    run = _make_run(inputs, sensors, adapter.runtime_identity())
    session = Session(run, Path(output_dir))
    bind_fsrt(session, fsrt_repo, python_executable)
    return _evaluate_and_save(session, Path(output_dir) / "workspace.json")


def inspect_investigation(path, evaluated_at=None):
    # Restore validation never imports/executes a domain provider, and never
    # writes into the source investigation during read-only inspection.
    with tempfile.TemporaryDirectory(prefix="ciw-inspect-") as directory:
        session = Session.from_workspace(Path(path), Path(directory))
        _validate_source(session.run)
        return _summary(session, path, evaluated_at)


def replay_investigation(path, rci_repo, fsrt_repo, output_dir, python_executable=None):
    # Validate the whole saved object before invoking any scientific code.
    with tempfile.TemporaryDirectory(prefix="ciw-replay-validate-") as directory:
        original = Session.from_workspace(Path(path), Path(directory))
        source = _validate_source(original.run)
        rci = _runtime("rci", rci_repo, python_executable, source["runtime"])
        for sensor in source["sensors"]:
            actual = rci.invoke(sensor["request"]["operation_id"], sensor["request"]["inputs"])
            if digest(actual) != digest(sensor["measurement"]):
                raise AdapterRefusal("replay_mismatch", "Replayed calibration differs from retained derived evidence")
        operation = fsrt_operation(original.run)
        successful = [r for r in original.results.values() if r["operation_id"] == operation]
        previous = [e for e in original.executions.values() if e["operation_id"] == operation]
        if not previous:
            raise AdapterRefusal("replay_unavailable", "No FSRT invocation retained for replay")
        expected = previous[-1]["runtime"]
        if expected is None:
            raise AdapterRefusal("replay_unavailable", "The retained attempt had no bound numerical runtime; execute a new operation explicitly")
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
