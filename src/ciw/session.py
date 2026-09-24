"""Authoritative shared selection, immutable analysis records, and saved workspaces."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .operations.registry import default_registry as default_operations, valid_operation_id
from .operations.runner import execute as execute_operation, check_seal, validate_execution
from .adapters.protocol import AdapterRefusal
from .core.identities import validate_evidence_identity
from .calibration_status import calibration_status

from .instruments import (
    compute_spectrum, compute_statistics, inspect_sample, run_metadata, validate_run,
)

LOG = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
_RESULT_SUMMARY_FIELDS = (
    "result_id", "operation_id", "execution_id", "channel", "interval_s",
    "created_at", "verification_status", "selection_revision",
)


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def envelope(kind: str, payload: Any, request_id: str | None = None) -> dict:
    return {"protocol_version": PROTOCOL_VERSION, "request_id": request_id,
            "type": kind, "payload": payload}


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path: Path) -> Any:
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant,
                      object_pairs_hook=unique_pairs)


def write_json(path: Path, data: Any) -> Path:
    """Replace only a completely serialized file, on the same filesystem."""
    path = Path(path)
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ciw-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _number(value: Any, name: str) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ProtocolError("invalid_payload", f"{name} must be a finite number")
    return float(value)


def _keys(payload: dict, allowed: set[str], required: set[str] | None = None) -> None:
    if payload.keys() - allowed:
        raise ProtocolError("invalid_payload", "Unknown payload fields: " + ", ".join(sorted(payload.keys() - allowed)))
    if required and required - payload.keys():
        raise ProtocolError("invalid_payload", "Missing payload fields: " + ", ".join(sorted(required - payload.keys())))


def _evaluated_at(payload: dict) -> str | None:
    if "evaluated_at" in payload and not isinstance(payload["evaluated_at"], str):
        raise ProtocolError("invalid_payload", "evaluated_at must be a timezone-aware ISO timestamp")
    return payload.get("evaluated_at")


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_evidence(run: dict) -> None:
    validate_run(run)
    validate_evidence_identity(run)
    if run["instrument"] == "analytic-damped-oscillator.v1":
        # Display geometry is outside the evidence identity; bind it to the
        # retained channels so a reopened workspace cannot draw other values.
        from .adapters.oscillator import validate_render_binding
        validate_render_binding(run)
    if run["instrument"] == "org.notationsystems.rci":
        from .investigation import _validate_source
        _validate_source(run)
    elif run["instrument"] == "org.notationsystems.gte":
        from .geodesic import validate_source
        validate_source(run)


def _recording_file(run: dict) -> str:
    return "recording-" + _digest(run) + ".json"


def _interval_for_run(run: dict, value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError("invalid_payload", "interval_s must be [start, end]")
    start, end = (_number(v, "interval_s") for v in value)
    if not 0 <= start < end <= run["metadata"]["duration_s"]:
        raise ProtocolError("invalid_payload", "Require 0 <= start < end <= recording duration")
    if not any(start <= t < end for t in run["time_s"]):
        raise ProtocolError("invalid_payload", "The selected interval contains no retained samples")
    return [start, end]


def _channel_for_run(run: dict, value: Any) -> str:
    if not isinstance(value, str) or value not in run["channels"]:
        raise ProtocolError("invalid_payload", "Unknown channel")
    return value


def _cursor_for_run(run: dict, value: Any) -> float:
    value = _number(value, "cursor_s")
    if not run["time_s"][0] <= value <= run["time_s"][-1]:
        raise ProtocolError("invalid_payload", "Cursor must lie within retained sample timestamps")
    return value


def _identity(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) != len(prefix) + 32:
        raise ValueError(f"Invalid saved {prefix.rstrip('-')} identity")
    suffix = value[len(prefix):]
    try:
        canonical = uuid.UUID(hex=suffix).hex
    except ValueError as exc:
        raise ValueError(f"Invalid saved {prefix.rstrip('-')} identity") from exc
    if canonical != suffix:
        raise ValueError(f"Invalid saved {prefix.rstrip('-')} identity")
    return value


def _timestamp(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


def _validate_saved_result(result: Any, run: dict, revision: int, recording_file: str) -> None:
    required = {"result_id", "execution_id", "operation_id", "evidence_id", "run_id",
                "selection_revision", "channel", "interval_s", "created_at", "data",
                "verification_id", "verification_status", "recording_file"}
    if not isinstance(result, dict) or not required <= result.keys():
        raise ValueError("Saved result is missing identity, source binding, or data")
    _identity(result["result_id"], "result-")
    _identity(result["execution_id"], "execution-")
    if result["run_id"] != run["run_id"] or result["evidence_id"] != run["evidence_id"]:
        raise ValueError("Saved result does not refer to this evidence")
    if result["recording_file"] != recording_file:
        raise ValueError("Saved result recording_file does not match the recorded content")
    if result["verification_status"] != "not_verified" or result["verification_id"] is not None:
        raise ValueError("Protocol v1 saved results must remain not_verified with verification_id null")
    saved_revision = result["selection_revision"]
    if type(saved_revision) is not int or not 0 <= saved_revision <= revision:
        raise ValueError("Saved result selection_revision is outside the workspace history")
    _timestamp(result["created_at"], "Saved result created_at")
    channel = _channel_for_run(run, result["channel"])
    start, end = _interval_for_run(run, result["interval_s"])
    if result.get("schema") == "ciw.operation-result.v1":
        check_seal(result)
        if (not isinstance(result.get("runtime"), dict)
                or not isinstance(result.get("parameters"), dict)
                or not isinstance(result.get("data"), dict)
                or result.get("role") not in {"analysis", "state_estimator", "calibration", "verification", "decision", "backend"}
                or not valid_operation_id(result.get("operation_id"))):
            raise ValueError("Invalid saved operation result")
        for key, value in (("channel", channel), ("interval_s", [start, end])):
            if key in result["parameters"] and result["parameters"][key] != value:
                raise ValueError("Saved operation parameters contradict its captured selection")
        from .operations.schemas import validate_payload, validate_role
        validate_role(result["operation_id"], result["role"])
        validate_payload(result["operation_id"], result["data"], run, result["parameters"],
                         {"channel": channel, "interval_s": [start, end]})
        return
    from .operations.schemas import validate_payload
    validate_payload(result["operation_id"], result["data"], run, {},
                     {"channel": channel, "interval_s": [start, end]})
    json.dumps(result, allow_nan=False)


class Session:
    def __init__(self, run: dict, output_dir: Path, *, operations=None):
        _validate_evidence(run)
        self.run = copy.deepcopy(run)
        self.output_dir = Path(output_dir)
        self.session_id = "session-" + uuid.uuid4().hex
        self.selection = {
            "run_id": run["run_id"], "channel": "q" if "q" in run["channels"] else next(iter(run["channels"])),
            "interval_s": [0.0, run["metadata"]["duration_s"]], "cursor_s": run["time_s"][0],
            "coordinate_frame": run["metadata"]["coordinate_frame"], "revision": 0,
        }
        self.results: dict[str, dict] = {}
        self.executions: dict[str, dict] = {}
        self.operations = operations if operations is not None else default_operations()
        # Native workflow artifacts share this operator session without being
        # coerced into the legacy single-recording analysis schema.
        from .workbench import Workbench
        self.workbench = Workbench()
        self._lock = threading.RLock()
        self._pending_operations = 0
        self.recording_file = _recording_file(self.run)
        write_json(self.output_dir / self.recording_file, self.run)

    def snapshot(self, evaluated_at: str | None = None) -> dict:
        with self._lock:
            snapshot = {"session_id": self.session_id, "run": run_metadata(self.run),
                        "selection": copy.deepcopy(self.selection), "results": self._result_summaries(),
                        "workbench": self.workbench.snapshot()}
            calibration = calibration_status(self.run, evaluated_at)
            if calibration:
                snapshot["calibration"] = calibration
            return snapshot

    def _result_summaries(self) -> list[dict]:
        with self._lock:
            return [{key: copy.deepcopy(result[key]) for key in _RESULT_SUMMARY_FIELDS}
                    for result in self.results.values()]

    def _interval(self, value: Any) -> list[float]:
        return _interval_for_run(self.run, value)

    def _channel(self, value: Any) -> str:
        return _channel_for_run(self.run, value)

    def _cursor(self, value: Any) -> float:
        return _cursor_for_run(self.run, value)

    def handle(self, request: Any) -> dict:
        request_id = None
        try:
            if not isinstance(request, dict):
                raise ProtocolError("invalid_request", "Request must be an object")
            candidate = request.get("request_id")
            if isinstance(candidate, str) and 0 < len(candidate) <= 128:
                request_id = candidate
            if request_id is None:
                raise ProtocolError("invalid_request", "request_id must be a nonempty string of at most 128 characters")
            if type(request.get("protocol_version")) is not int or request["protocol_version"] != PROTOCOL_VERSION:
                raise ProtocolError("unsupported_version", "Supported protocol_version is 1")
            if request.keys() != {"protocol_version", "request_id", "type", "payload"}:
                raise ProtocolError("invalid_request", "Use protocol_version, request_id, type and payload only")
            kind, payload = request["type"], request["payload"]
            if not isinstance(kind, str) or not isinstance(payload, dict):
                raise ProtocolError("invalid_request", "type must be a string and payload an object")
            result = self._dispatch(kind, payload)
            response = envelope("response", result, request_id)
            if kind == "result.get" and payload["result_id"] in self.results:
                # The payload remains the exact retained result, including its
                # seal. Serving-time metadata belongs outside that artifact.
                calibration = calibration_status(self.run, _evaluated_at(payload))
                if calibration:
                    response["calibration"] = calibration
            return response
        except (ProtocolError, AdapterRefusal) as exc:
            return envelope("error", {"code": exc.code, "message": str(exc)}, request_id)
        except (ValueError, TypeError) as exc:
            return envelope("error", {"code": "invalid_payload", "message": str(exc)}, request_id)
        except OSError:
            return envelope("error", {"code": "storage_error", "message": "Unable to persist the requested record"}, request_id)
        except Exception:  # noqa: BLE001 - a defect inside one request must not end the client's connection
            # Retained state is already consistent here: every mutation happens
            # under the lock and every reservation is released in a finally.
            # The traceback goes to the service log; the client gets an explicit
            # envelope rather than a closed socket.
            LOG.exception("Request %s failed inside the service", request_id)
            return envelope("error", {"code": "internal_error",
                                      "message": "The request failed inside the service; see the service log"}, request_id)

    def _dispatch(self, kind: str, payload: dict) -> dict:
        if kind == "session.get":
            _keys(payload, {"evaluated_at"})
            return self.snapshot(_evaluated_at(payload))
        if kind == "run.get":
            _keys(payload, set())
            return copy.deepcopy(self.run)
        if kind == "source.add":
            return self.workbench.add_source(payload)
        if kind == "source.list":
            _keys(payload, set())
            return {"sources": self.workbench.list_sources()}
        if kind == "spatial.list":
            _keys(payload, set())
            return {"sources": self.workbench.spatial_sources()}
        if kind == "spatial.inspect":
            return self.workbench.inspect_spatial(payload)
        if kind == "source.get":
            _keys(payload, {"source_id"}, {"source_id"})
            return self.workbench.get_source(payload["source_id"])
        if kind == "bundle.list":
            _keys(payload, set())
            return {"bundles": self.workbench.list_bundles()}
        if kind == "bundle.replayability":
            # Read-only: compares retained runtime identities with this host's
            # and names the differing fields; nothing executes or rebinds.
            _keys(payload, set())
            from .workspace_verify import bundle_reports, host_identity
            return {"host": host_identity(), "bundles": bundle_reports(self.workbench)}
        if kind == "bundle.get":
            _keys(payload, {"bundle_id"}, {"bundle_id"})
            return self.workbench.get_bundle(payload["bundle_id"])
        if kind == "bundle.replay":
            return self.workbench.replay(payload)
        if kind == "fusion.list":
            _keys(payload, set())
            return {"contexts": self.workbench.fusion_contexts()}
        if kind == "experiment.inspect":
            return self.workbench.inspect_experiment(payload)
        if kind == "instrument.list":
            _keys(payload, set())
            return {"instruments": self.workbench.instrument_views()}
        if kind == "instrument.inspect":
            return self.workbench.inspect_instrument(payload)
        if kind == "candidate.list":
            _keys(payload, set())
            return {"candidates": self.workbench.list_candidates()}
        if kind == "candidate.get":
            _keys(payload, {"candidate_id"}, {"candidate_id"})
            return self.workbench.get_candidate(payload["candidate_id"])
        if kind == "selection.update":
            _keys(payload, {"expected_revision", "channel", "interval_s", "cursor_s"}, {"expected_revision"})
            with self._lock:
                revision = payload["expected_revision"]
                if type(revision) is not int or revision < 0:
                    raise ProtocolError("invalid_payload", "expected_revision must be a nonnegative integer")
                if revision != self.selection["revision"]:
                    raise ProtocolError("revision_conflict", "Selection changed; refresh session.get before retrying")
                if len(payload) == 1:
                    raise ProtocolError("invalid_payload", "Provide a channel, interval_s or cursor_s to update")
                updated = copy.deepcopy(self.selection)
                if "channel" in payload:
                    updated["channel"] = self._channel(payload["channel"])
                if "interval_s" in payload:
                    updated["interval_s"] = self._interval(payload["interval_s"])
                if "cursor_s" in payload:
                    updated["cursor_s"] = self._cursor(payload["cursor_s"])
                updated["revision"] += 1
                self.selection = updated
                return copy.deepcopy(updated)
        if kind == "sample.get":
            _keys(payload, {"time_s"}, {"time_s"})
            return inspect_sample(self.run, self._cursor(payload["time_s"]))
        if kind in {"analysis.stats", "analysis.spectrum"}:
            _keys(payload, {"channel", "interval_s"})
            with self._lock:
                selected = copy.deepcopy(self.selection)
            channel = self._channel(payload.get("channel", selected["channel"]))
            interval = self._interval(payload.get("interval_s", selected["interval_s"]))
            operation_id = "statistics.v1" if kind == "analysis.stats" else "spectrum.periodogram.v1"
            data = self.operations.get(operation_id).execute(self.run, {"channel": channel, "interval_s": interval})
            result_id = "result-" + uuid.uuid4().hex
            result = {
                "result_id": result_id, "evidence_id": self.run["evidence_id"],
                "operation_id": "statistics.v1" if kind == "analysis.stats" else "spectrum.periodogram.v1",
                "execution_id": "execution-" + uuid.uuid4().hex,
                "verification_id": None, "verification_status": "not_verified",
                "run_id": self.run["run_id"], "selection_revision": selected["revision"],
                "channel": channel, "interval_s": interval, "created_at": utc_now(),
                "recording_file": self.recording_file, "data": data,
            }
            with self._lock:
                if len(self.results) >= 1024:
                    raise ProtocolError("capacity_exceeded", "Save the workspace and start a new session after 1024 results")
                write_json(self.output_dir / f"{result_id}.json", result)
                self.results[result_id] = result
            return copy.deepcopy(result)
        if kind == "operation.list":
            _keys(payload, set())
            return {"operations": self.operations.describe() + self.workbench.describe_operations()}
        if kind == "execution.list":
            _keys(payload, set())
            with self._lock:
                return {"executions": copy.deepcopy(list(self.executions.values()))
                        + self.workbench.native_executions() + self.workbench.candidate_executions()}
        if kind == "operation.execute":
            _keys(payload, {"operation_id", "parameters"}, {"operation_id"})
            if (not valid_operation_id(payload["operation_id"])
                    or not isinstance(payload.get("parameters", {}), dict)):
                raise ProtocolError("invalid_payload", "operation_id must be versioned and parameters must be an object")
            from .workbench import WORKFLOW_OPERATION_IDS, CANDIDATE_OPERATIONS
            if payload["operation_id"] in CANDIDATE_OPERATIONS:
                return self.workbench.execute_candidate(payload["operation_id"], payload.get("parameters", {}))
            if payload["operation_id"] in WORKFLOW_OPERATION_IDS:
                parameters = copy.deepcopy(payload.get("parameters", {}))
                allowed = {"source_id", "configuration"} if payload["operation_id"] == "ciw.telemetry.v1" else {"source_id", "upstream_bundle_id"}
                _keys(parameters, allowed, {"source_id", "configuration"} if payload["operation_id"] == "ciw.telemetry.v1" else {"source_id"})
                return self.workbench.execute({"operation_id": payload["operation_id"], **parameters})
            # Reserve bounded capacity and capture the scientific request. A
            # subprocess may wait up to its deadline, so it cannot own the
            # shared selection/publication lock while it calculates.
            with self._lock:
                if (len(self.results) + self._pending_operations >= 1024
                        or len(self.executions) + self._pending_operations >= 1024):
                    raise ProtocolError("capacity_exceeded", "Save and start a new session after 1024 operations")
                selected = copy.deepcopy(self.selection)
                parameters = copy.deepcopy(payload.get("parameters", {}))
                if "channel" in parameters:
                    selected["channel"] = self._channel(parameters["channel"])
                if "interval_s" in parameters:
                    selected["interval_s"] = self._interval(parameters["interval_s"])
                captured_run = copy.deepcopy(self.run)
                recording_file = self.recording_file
                operation_id = payload["operation_id"]
                operations = self.operations
                self._pending_operations += 1
            try:
                execution, result = execute_operation(
                    operations, captured_run, selected, recording_file, operation_id, parameters)
                with self._lock:
                    # Legacy analysis can publish while this provider runs;
                    # recheck before writing either half of the operation pair.
                    if (len(self.executions) >= 1024
                            or (result is not None and len(self.results) >= 1024)):
                        raise ProtocolError("capacity_exceeded", "Save and start a new session after 1024 operations")
                    write_json(self.output_dir / (execution["execution_id"] + ".json"), execution)
                    if result is not None:
                        write_json(self.output_dir / (result["result_id"] + ".json"), result)
                        self.results[result["result_id"]] = result
                    self.executions[execution["execution_id"]] = execution
            finally:
                with self._lock:
                    self._pending_operations -= 1
            if result is None:
                return {"status": "refused", "execution": copy.deepcopy(execution), "result": None}
            return {"status": "completed", "execution": copy.deepcopy(execution), "result": copy.deepcopy(result)}
        if kind == "result.list":
            _keys(payload, {"evaluated_at"})
            result = {"results": self._result_summaries() + self.workbench.native_result_summaries()}
            calibration = calibration_status(self.run, _evaluated_at(payload))
            if calibration:
                result["calibration"] = calibration
                result["calibration_scope"] = {
                    "run_id": self.run["run_id"], "result_ids": list(self.results),
                }
            return result
        if kind == "result.get":
            _keys(payload, {"result_id", "evaluated_at"}, {"result_id"})
            result_id = payload["result_id"]
            with self._lock:
                if not isinstance(result_id, str):
                    raise ProtocolError("not_found", "Result not found in this session")
                if result_id in self.results:
                    return copy.deepcopy(self.results[result_id])
                native = self.workbench.get_native_result(result_id)
                if native is not None:
                    return native
                raise ProtocolError("not_found", "Result not found in this session")
        if kind == "workspace.save":
            _keys(payload, set())
            path = self.save_workspace(self.output_dir / "workspace.json")
            return {"workspace_file": str(path)}
        raise ProtocolError("unknown_command", f"Unknown request type: {kind}")

    def save_workspace(self, path: Path) -> Path:
        with self._lock:
            workspace = {"workspace_version": 1, "saved_at": utc_now(), "run": self.run,
                         "selection": self.selection, "results": list(self.results.values()),
                         "view_settings": {}}
            if self.executions:
                workspace["workspace_version"] = 2
                workspace["executions"] = list(self.executions.values())
            retained = self.workbench.serialize()
            if retained["sources"] or retained["bundles"]:
                workspace["workspace_version"] = 3
                workspace["workbench"] = retained
            return write_json(path, workspace)

    @classmethod
    def from_workspace(cls, path: Path, output_dir: Path | None = None) -> Session:
        """Reopen stored evidence/results without executing an analysis."""
        workspace = read_json(path)
        if not isinstance(workspace, dict) or type(workspace.get("workspace_version")) is not int or workspace["workspace_version"] not in (1, 2, 3):
            raise ValueError("Unsupported workspace format")
        # The document's key set is defined entirely by this module; a key it
        # does not know would survive a reopen unvalidated and vanish on the
        # next save, so it is refused like any other unknown payload key.
        # Version-specific presence of executions and workbench is checked below
        # with its own message; this refuses keys no version defines.
        allowed = {"workspace_version", "saved_at", "run", "selection", "results", "view_settings", "executions", "workbench"}
        unknown = sorted(set(workspace) - allowed)
        if unknown:
            raise ValueError(f"Unknown workspace keys: {', '.join(map(str, unknown))}")
        from .workbench import Workbench
        if workspace["workspace_version"] == 3:
            retained_workbench = Workbench.restore(workspace.get("workbench"))
        else:
            if "workbench" in workspace:
                raise ValueError("Retained workbench sources require workspace version 3")
            retained_workbench = Workbench()
        run = workspace.get("run")
        _validate_evidence(run)
        selection = workspace.get("selection")
        results = workspace.get("results")
        expected = {"run_id", "channel", "interval_s", "cursor_s", "coordinate_frame", "revision"}
        if not isinstance(selection, dict) or selection.keys() != expected:
            raise ValueError("Invalid saved selection")
        if selection["run_id"] != run["run_id"] or selection["coordinate_frame"] != run["metadata"]["coordinate_frame"]:
            raise ValueError("Saved selection does not refer to the recording")
        if type(selection["revision"]) is not int or selection["revision"] < 0:
            raise ValueError("Invalid saved selection revision")
        _channel_for_run(run, selection["channel"])
        _interval_for_run(run, selection["interval_s"])
        _cursor_for_run(run, selection["cursor_s"])
        if not isinstance(results, list) or len(results) > 1024:
            raise ValueError("Invalid saved results")
        if "saved_at" in workspace:
            _timestamp(workspace["saved_at"], "Workspace saved_at")
        if not isinstance(workspace.get("view_settings", {}), dict):
            raise ValueError("Invalid workspace view_settings")
        json.dumps(workspace, allow_nan=False)
        recording_file = _recording_file(run)
        # Validate the entire workspace before constructing a session or writing
        # any evidence/results. No scientific compute functions are invoked.
        result_map = {}
        execution_ids = set()
        for result in results:
            _validate_saved_result(result, run, selection["revision"], recording_file)
            result_id = result["result_id"]
            if result_id in result_map or result["execution_id"] in execution_ids:
                raise ValueError("Saved result identity mismatch or duplication")
            result_map[result_id] = result
            execution_ids.add(result["execution_id"])
        from .adapters.covariance_records import validate_result_dependencies
        validate_result_dependencies(result_map)
        executions = workspace.get("executions", [])
        if not isinstance(executions, list) or len(executions) > 1024:
            raise ValueError("Invalid saved executions")
        if workspace["workspace_version"] == 1 and executions:
            raise ValueError("Executions require workspace version 2")
        execution_map = {}
        for execution in executions:
            validate_execution(execution, run, selection["revision"], result_map)
            eid = execution["execution_id"]
            if eid in execution_map:
                raise ValueError("Duplicate execution identity")
            if eid in execution_ids and execution.get("result_id") not in result_map:
                raise ValueError("Execution identity collision")
            execution_map[eid] = execution
        for result in result_map.values():
            if result.get("schema") == "ciw.operation-result.v1":
                execution = execution_map.get(result["execution_id"])
                if (execution is None or execution["status"] != "completed"
                        or execution["result_id"] != result["result_id"]):
                    raise ValueError("Operation result is missing its completed execution")
        native_occurrences = {entry["execution_id"] for entry in retained_workbench.native_executions()}
        native_occurrences.update(entry["execution_id"] for entry in retained_workbench.candidate_executions())
        native_results = {entry["result_id"] for entry in retained_workbench.native_result_summaries()}
        if (execution_ids | set(execution_map) | set(result_map)) & (native_occurrences | native_results):
            raise ValueError("Identity collision between recording operations and retained workflows")
        restored = cls(run, output_dir or Path(path).parent)
        restored.selection = copy.deepcopy(selection)
        restored.results = copy.deepcopy(result_map)
        restored.executions = copy.deepcopy(execution_map)
        restored.workbench = retained_workbench
        for result in restored.results.values():
            write_json(restored.output_dir / (result["result_id"] + ".json"), result)
        for execution in restored.executions.values():
            write_json(restored.output_dir / (execution["execution_id"] + ".json"), execution)
        return restored
