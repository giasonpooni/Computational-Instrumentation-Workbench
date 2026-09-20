"""Authoritative shared selection, immutable analysis records, and saved workspaces."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instruments import (
    compute_spectrum, compute_statistics, inspect_sample, run_metadata, validate_run,
)

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
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


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


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_evidence(run: dict) -> None:
    validate_run(run)
    # This is content integrity, not an assertion of authenticity or verification.
    # Keep this scope aligned with make_demo_run's scientific-record digest.
    scientific_record = {key: run[key] for key in ("instrument", "metadata", "time_s", "channels")}
    if run["evidence_id"] != "sha256:" + _digest(scientific_record):
        raise ValueError("Evidence integrity mismatch: scientific content does not match evidence_id")


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
    # Counting retained timestamps checks the stored shape, without evaluating
    # statistics, applying a window, or executing a Fourier transform.
    expected_count = sum(start <= time < end for time in run["time_s"])
    data = result["data"]
    if not isinstance(data, dict):
        raise ValueError("Saved result data must be an object")
    count = data.get("sample_count")
    if type(count) is not int or count != expected_count:
        raise ValueError("Saved result sample_count does not match its source interval")
    unit = run["channels"][channel]["unit"]
    operation = result["operation_id"]
    if operation == "statistics.v1":
        if data.keys() != {"sample_count", "mean", "minimum", "maximum", "rms", "unit"}:
            raise ValueError("Invalid saved statistics data fields")
        if data["unit"] != unit:
            raise ValueError("Saved statistics unit does not match its source channel")
        mean, minimum, maximum, rms = (_number(data[key], f"Saved statistics {key}")
                                      for key in ("mean", "minimum", "maximum", "rms"))
        if minimum > maximum or rms < 0:
            raise ValueError("Saved statistics have invalid bounds or negative RMS")
        if ((mean < minimum and not math.isclose(mean, minimum, rel_tol=1e-12))
                or (mean > maximum and not math.isclose(mean, maximum, rel_tol=1e-12))):
            raise ValueError("Saved statistics mean is outside its bounds")
    elif operation == "spectrum.periodogram.v1":
        fields = {"sample_count", "method", "window", "detrend", "scaling", "frequency_hz",
                  "psd", "unit", "peak_frequency_hz", "sample_rate_hz"}
        if data.keys() != fields or count < 4:
            raise ValueError("Invalid saved spectrum data fields or sample count")
        for name, expected in (("method", "periodogram"), ("window", "hann"),
                               ("detrend", "constant"), ("scaling", "density")):
            if data[name] != expected:
                raise ValueError(f"Unsupported saved spectrum {name}")
        sample_rate = _number(data["sample_rate_hz"], "Saved spectrum sample_rate_hz")
        if sample_rate != run["metadata"]["sample_rate_hz"] or data["unit"] != f"({unit})^2/Hz":
            raise ValueError("Saved spectrum sample rate or density unit does not match its source")
        frequencies, psd = data["frequency_hz"], data["psd"]
        size = count // 2 + 1
        if not isinstance(frequencies, list) or not isinstance(psd, list) or len(frequencies) != size or len(psd) != size:
            raise ValueError("Saved spectrum arrays do not match the one-sided transform shape")
        for index, (frequency, density) in enumerate(zip(frequencies, psd)):
            frequency = _number(frequency, "Saved spectrum frequency_hz")
            density = _number(density, "Saved spectrum psd")
            if density < 0:
                raise ValueError("Saved spectrum density must be nonnegative")
            expected_frequency = index * (sample_rate / count)
            if not math.isclose(frequency, expected_frequency, rel_tol=1e-12, abs_tol=sample_rate * 1e-14):
                raise ValueError("Saved spectrum frequency grid does not match its sample rate and count")
        peak = data["peak_frequency_hz"]
        maximum = max(psd)
        if maximum == 0:
            if peak is not None:
                raise ValueError("A zero saved spectrum must have a null peak")
        elif _number(peak, "Saved spectrum peak_frequency_hz") != frequencies[psd.index(maximum)]:
            raise ValueError("Saved spectrum peak does not match its stored density array")
    else:
        raise ValueError("Unsupported saved result operation_id")
    json.dumps(result, allow_nan=False)


class Session:
    def __init__(self, run: dict, output_dir: Path):
        _validate_evidence(run)
        self.run = copy.deepcopy(run)
        self.output_dir = Path(output_dir)
        self.session_id = "session-" + uuid.uuid4().hex
        self.selection = {
            "run_id": run["run_id"], "channel": "q",
            "interval_s": [0.0, run["metadata"]["duration_s"]], "cursor_s": 0.0,
            "coordinate_frame": run["metadata"]["coordinate_frame"], "revision": 0,
        }
        self.results: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.recording_file = _recording_file(self.run)
        write_json(self.output_dir / self.recording_file, self.run)

    def snapshot(self) -> dict:
        with self._lock:
            return {"session_id": self.session_id, "run": run_metadata(self.run),
                    "selection": copy.deepcopy(self.selection), "results": self._result_summaries()}

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
            return envelope("response", result, request_id)
        except ProtocolError as exc:
            return envelope("error", {"code": exc.code, "message": str(exc)}, request_id)
        except (ValueError, TypeError) as exc:
            return envelope("error", {"code": "invalid_payload", "message": str(exc)}, request_id)
        except OSError:
            return envelope("error", {"code": "storage_error", "message": "Unable to persist the requested record"}, request_id)

    def _dispatch(self, kind: str, payload: dict) -> dict:
        if kind == "session.get":
            _keys(payload, set())
            return self.snapshot()
        if kind == "run.get":
            _keys(payload, set())
            return copy.deepcopy(self.run)
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
            compute = compute_statistics if kind == "analysis.stats" else compute_spectrum
            data = compute(self.run, channel, interval)
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
        if kind == "result.list":
            _keys(payload, set())
            return {"results": self._result_summaries()}
        if kind == "result.get":
            _keys(payload, {"result_id"}, {"result_id"})
            result_id = payload["result_id"]
            with self._lock:
                if not isinstance(result_id, str) or result_id not in self.results:
                    raise ProtocolError("not_found", "Result not found in this session")
                return copy.deepcopy(self.results[result_id])
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
            return write_json(path, workspace)

    @classmethod
    def from_workspace(cls, path: Path, output_dir: Path | None = None) -> Session:
        """Reopen stored evidence/results without executing an analysis."""
        workspace = read_json(path)
        if not isinstance(workspace, dict) or type(workspace.get("workspace_version")) is not int or workspace["workspace_version"] != 1:
            raise ValueError("Unsupported workspace format")
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
        restored = cls(run, output_dir or Path(path).parent)
        restored.selection = copy.deepcopy(selection)
        restored.results = copy.deepcopy(result_map)
        for result in restored.results.values():
            write_json(restored.output_dir / (result["result_id"] + ".json"), result)
        return restored
