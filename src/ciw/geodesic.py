"""GTE record mapping and replay; circle mathematics stays in its pinned engine."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import uuid

from .adapters.protocol import AdapterRefusal, InstrumentManifest
from .core.identities import digest, evidence_id
from .core.records import finite_tree, mapping, number, string
from .investigation import _runtime
from .operations.registry import Operation
from .session import Session, _reject_constant


GTE_OPERATION = "gte.project-circle.v1"
GTE_INSTRUMENT = "org.notationsystems.gte"
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def _parse(raw: bytes) -> dict:
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("GTE request exceeds the 4 MiB evidence limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=_reject_constant)
    mapping(value, "GTE request")
    finite_tree(value, "GTE request")
    return value


def _observations(request):
    """Read transport declarations only; validity and projection belong to GTE."""
    if set(request) != {"schema", "observations", "constraint", "policy"} or request["schema"] != "gte.circle-request.v1":
        raise ValueError("Expected gte.circle-request.v1 with observations, constraint and policy")
    observations = mapping(request["observations"], "observations")
    mapping(request["constraint"], "constraint")
    mapping(request["policy"], "policy")
    times, points = observations.get("time_s"), observations.get("points_m")
    if not isinstance(times, list) or not 1 <= len(times) <= 128:
        raise ValueError("GTE requires between 1 and 128 sample times")
    if not isinstance(points, list) or len(points) != len(times):
        raise ValueError("GTE points must match sample times")
    previous = -1.0
    for timestamp, point in zip(times, points):
        timestamp = number(timestamp, "sample time")
        if timestamp < 0 or timestamp <= previous:
            raise ValueError("GTE sample times must be nonnegative and strictly increasing")
        previous = timestamp
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("GTE points must have two coordinates")
        for value in point:
            number(value, "point coordinate")
    string(observations.get("coordinate_frame"), "coordinate frame")
    string(observations.get("unit"), "observation unit")
    string(observations.get("time_origin"), "time origin")
    return observations


def _duration(times):
    # The extra second supplies a half-open selection interval for the final
    # event; it is not a measured sample period or an extrapolation of evidence.
    duration = times[-1] + 1.0
    if not math.isfinite(duration) or duration <= times[-1]:
        raise ValueError("Sample times exceed CIW's representable selection support")
    return duration


def validate_source(run):
    """Validate retained byte/channel bindings without importing the GTE engine."""
    try:
        source = mapping(run["metadata"]["gte_source"], "GTE source")
        if set(source) != {"schema", "request_bytes_b64", "request_sha256", "request"} or source["schema"] != "ciw.gte-source.v1":
            raise ValueError("Invalid GTE source declaration")
        raw = base64.b64decode(source["request_bytes_b64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != source["request_sha256"]:
            raise ValueError("GTE original request byte digest mismatch")
        request = _parse(raw)
        if digest(request) != digest(source["request"]):
            raise ValueError("GTE parsed request differs from its retained original bytes")
        observation = _observations(request)
        expected_channels = {
            axis: {"unit": observation["unit"], "kind": "observation",
                   "values": [point[i] for point in observation["points_m"]]}
            for i, axis in enumerate(("x", "y"))
        }
        metadata = run["metadata"]
        if (run["instrument"] != GTE_INSTRUMENT or digest(run["channels"]) != digest(expected_channels)
                or digest(run["time_s"]) != digest(observation["time_s"])
                or metadata["sample_count"] != len(observation["time_s"])
                or metadata["duration_s"] != _duration(observation["time_s"])
                or metadata["coordinate_frame"] != observation["coordinate_frame"]
                or metadata["sample_rate_hz"] is not None
                or metadata["provenance"]["time_origin"] != observation["time_origin"]):
            raise ValueError("GTE channels or metadata differ from retained observations")
        return request
    except (KeyError, TypeError, IndexError, UnicodeError) as exc:
        raise ValueError("Invalid retained GTE source structure") from exc


def _make_run(raw):
    request = _parse(raw)
    observations = _observations(request)
    channels = {axis: {"unit": observations["unit"], "kind": "observation",
                       "values": [point[i] for point in observations["points_m"]]}
                for i, axis in enumerate(("x", "y"))}
    manifest = InstrumentManifest(
        instrument_id=GTE_INSTRUMENT, version="1", role="operation_provider",
        inputs=("gte.circle-request.v1",), outputs=("run.v1",),
        units={axis: observations["unit"] for axis in channels},
        frames=(observations["coordinate_frame"],),
        sampling={"mode": "retained_events", "duration_s_semantics": "last timestamp plus 1 s selection support only"},
        normalization={"observations": "unchanged", "request": "exact bytes retained"},
        supported_operations=(), determinism={"pinned_runtime": True, "offline_replay": True},
        tolerance_policy={"replay": "same runtime and exact scientific data digest"},
        calibration_requirements={"geometry": "declared fixed exact; physical calibration not supplied"},
    )
    run = {"run_schema": "run.v1", "run_id": "run-" + uuid.uuid4().hex,
           "instrument": GTE_INSTRUMENT, "time_s": deepcopy(observations["time_s"]),
           "channels": channels, "render": {}, "metadata": {
               "sample_count": len(observations["time_s"]), "sample_rate_hz": None,
               "duration_s": _duration(observations["time_s"]),
               "coordinate_frame": observations["coordinate_frame"], "manifest": manifest.to_dict(),
               "provenance": {"time_origin": observations["time_origin"],
                              "claim_scope": "declared geometric reconciliation candidate only"},
               "gte_source": {"schema": "ciw.gte-source.v1", "request": request,
                              "request_bytes_b64": base64.b64encode(raw).decode("ascii"),
                              "request_sha256": hashlib.sha256(raw).hexdigest()}}}
    validate_source(run)
    run["evidence_id"] = evidence_id(run)
    return run


def _inputs(run, parameters):
    if set(parameters) - {"constraint", "policy"}:
        raise AdapterRefusal("invalid_parameters", "Only whole constraint/policy overrides are accepted; observations remain retained evidence")
    request = deepcopy(validate_source(run))
    for key, value in parameters.items():
        request[key] = deepcopy(mapping(value, key))
    return request


def bind_gte(session, gte_repo, python_executable=None, expected=None):
    adapter = _runtime("gte", gte_repo, python_executable, expected)
    # Geometric computation uses the existing backend role without asserting
    # Bayesian inference or a dynamics model.
    session.operations.register(Operation(
        GTE_OPERATION, "backend",
        lambda run, parameters: adapter.invoke(GTE_OPERATION, _inputs(run, parameters)),
        adapter.runtime_identity))
    return adapter


def _summary(session, path):
    results = [r for r in session.results.values() if r["operation_id"] == GTE_OPERATION]
    executions = [e for e in session.executions.values() if e["operation_id"] == GTE_OPERATION]
    request = validate_source(session.run)
    rows = [{"state": "observation", "quantity": "retained points", "value": request["observations"]["points_m"], "unit": request["observations"]["unit"]}]
    for result in results:
        data = result["data"]
        rows.extend([
            {"state": "candidate", "quantity": "projected points", "value": data["projected_points_m"], "unit": "m"},
            {"state": "reconciled", "quantity": "eligible points", "value": data["reconciled_points_m"], "unit": "m"},
            {"state": "residual", "quantity": "radial before", "value": data["diagnostics"]["radial_residual_before_m"], "unit": "m"},
            {"state": "diagnostic", "quantity": "reconciliation", "value": data["reconciliation"]["status"], "unit": "—"},
        ])
    return {"status": executions[-1]["status"] if executions else "not_executed",
            "workspace_file": str(path), "run_id": session.run["run_id"], "evidence_id": session.run["evidence_id"],
            "results": deepcopy(results), "execution_ids": [e["execution_id"] for e in executions],
            "refusals": deepcopy([e for e in executions if e["status"] == "refused"]), "terminal_rows": rows,
            "note": "Full retained batch; selection must contain every sample. Geometric eligibility does not establish physical validation or authority."}


def _evaluate_and_save(session, path, parameters=None):
    response = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                               "type": "operation.execute", "payload": {"operation_id": GTE_OPERATION, "parameters": parameters or {}}})
    if response["type"] == "error":
        raise ValueError(response["payload"]["message"])
    session.save_workspace(path)
    return _summary(session, path)


def create_investigation(inputs_path, gte_repo, output_dir, python_executable=None):
    with Path(inputs_path).open("rb") as stream:
        run = _make_run(stream.read(MAX_REQUEST_BYTES + 1))
    # Validate the local executable pin before writing source or execution data.
    adapter = _runtime("gte", gte_repo, python_executable)
    adapter.runtime_identity()
    session = Session(run, Path(output_dir))
    bind_gte(session, gte_repo, python_executable)
    return _evaluate_and_save(session, Path(output_dir) / "workspace.json")


def inspect_investigation(path):
    with tempfile.TemporaryDirectory(prefix="ciw-gte-inspect-") as directory:
        session = Session.from_workspace(Path(path), Path(directory))
        validate_source(session.run)
        return _summary(session, path)


def replay_investigation(path, gte_repo, output_dir, python_executable=None):
    with tempfile.TemporaryDirectory(prefix="ciw-gte-replay-") as directory:
        original = Session.from_workspace(Path(path), Path(directory))
        validate_source(original.run)
        previous = [e for e in original.executions.values() if e["operation_id"] == GTE_OPERATION]
        if not previous or previous[-1]["runtime"] is None:
            raise AdapterRefusal("replay_unavailable", "No GTE execution with retained runtime identity")
        expected = previous[-1]["runtime"]
        parameters = previous[-1]["parameters"]
        adapter = _runtime("gte", gte_repo, python_executable, expected)
        adapter.runtime_identity()
        old_result = original.results.get(previous[-1]["result_id"])
    session = Session.from_workspace(Path(path), Path(output_dir))
    bind_gte(session, gte_repo, python_executable, expected)
    # Reproduce the captured scientific support even if the user moved the
    # current selection after the last invocation. Selection history advances;
    # retained execution and result records remain unchanged.
    captured = {key: previous[-1][key] for key in ("channel", "interval_s")}
    if any(session.selection[key] != value for key, value in captured.items()):
        selection = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                                    "type": "selection.update", "payload": {
                                        "expected_revision": session.selection["revision"], **captured}})
        if selection["type"] == "error":
            raise ValueError(selection["payload"]["message"])
    summary = _evaluate_and_save(session, Path(output_dir) / "workspace.json", parameters)
    if old_result is not None:
        matches = (summary["status"] == "completed"
                   and digest(old_result["data"]) == digest(summary["results"][-1]["data"]))
    else:
        latest = list(session.executions.values())[-1]
        matches = (latest["status"] == "refused"
                   and latest["refusal"] == previous[-1]["refusal"])
    summary["replay_data_digest_matches"] = matches
    if not matches:
        raise AdapterRefusal("replay_mismatch", "GTE replay differs from retained outcome; both invocations retained")
    return summary
