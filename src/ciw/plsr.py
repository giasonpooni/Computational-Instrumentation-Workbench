"""Portable PLSR terminal runs, with input binding and explicit replay.

This is separate from the oscillator-specific WebSocket workspace. It never
converts a Lyapunov sample into fabricated telemetry or viewport geometry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from . import plsr_engine as engine

BUNDLE_SCHEMA = "ciw-plsr-run-v1"
OPERATION_ID = "plsr.verdict.v1"
_FIELDS = {
    "bundle_schema", "instrument", "evidence_id", "operation_id",
    "execution_id", "result_id", "created_at", "verification_id",
    "verification_status", "model", "sample", "record", "runtime",
    "replay_of", "bundle_digest",
}
_RUNTIME_FIELDS = {
    "repository", "commit", "package_version", "source_digest",
    "python_version", "numpy_version", "jsonschema_version", "adapter_version",
}
_PIN_FIELDS = {"repository", "commit", "package_version", "source_digest", "adapter_version"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("ascii")).hexdigest()


def _bundle_digest(bundle: dict) -> str:
    return _digest({key: value for key, value in bundle.items() if key != "bundle_digest"})


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _float(value: str) -> float:
    number = float(value)
    if number == 0.0 and Decimal(value) != 0:
        raise ValueError("Nonzero JSON number underflows float64")
    return number


def _read(path: Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                       parse_constant=_reject_constant, parse_float=_float)
    # Also refuse exponent overflow (1e999), which is not a JSON constant token.
    _canonical(value)
    return value


def _write_immutable(path: Path, value: Any) -> Path:
    """Publish complete bytes atomically; never replace different saved content."""
    path = Path(path)
    content = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".plsr-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link makes the already-complete temporary inode visible
            # exclusively. os.replace would silently replace another result.
            os.link(temporary, path)
        except FileExistsError:
            if _canonical(_read(path)) != _canonical(value):
                raise ValueError(f"Refusing to overwrite existing file with different content: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _identity(value: Any, prefix: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", value):
        raise ValueError(f"Invalid {prefix.rstrip('-')} identity")


def _hash(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA-256 digest")


def _evidence_id(model_digest: str, sample: dict) -> str:
    return "sha256:" + _digest({"model_artifact_digest": model_digest, "sample": sample})


def _validate_bundle(bundle: Any) -> dict:
    if not isinstance(bundle, dict) or bundle.keys() != _FIELDS:
        raise ValueError("Invalid ciw-plsr-run-v1 fields")
    if (bundle["bundle_schema"] != BUNDLE_SCHEMA or bundle["instrument"] != "plsr"
            or bundle["operation_id"] != OPERATION_ID):
        raise ValueError("Unsupported PLSR bundle or operation")
    _hash(bundle["bundle_digest"])
    if bundle["bundle_digest"] != _bundle_digest(bundle):
        raise ValueError("PLSR bundle digest mismatch")
    if bundle["verification_id"] is not None or bundle["verification_status"] != "not_verified":
        raise ValueError("PLSR run verification must remain not_verified with no verification identity")
    _identity(bundle["result_id"], "result-")
    _identity(bundle["execution_id"], "execution-")
    try:
        stamp = datetime.fromisoformat(bundle["created_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("created_at must be a timezone-aware ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    runtime = bundle["runtime"]
    if (not isinstance(runtime, dict) or runtime.keys() != _RUNTIME_FIELDS
            or any(not isinstance(v, str) or not v.strip() for v in runtime.values())):
        raise ValueError("Invalid saved runtime identity")
    supported = engine.runtime_identity()
    if any(runtime[key] != supported[key] for key in _PIN_FIELDS):
        raise ValueError("Saved runtime or adapter pin is not supported by this installation")
    model = engine.model_from_dict(bundle["model"])
    sample = engine.validate_sample(model, bundle["sample"])
    if bundle["evidence_id"] != _evidence_id(model.artifact_digest, sample):
        raise ValueError("Saved evidence identity does not bind this model and sample")
    engine.validate_record(model, sample, bundle["record"])
    replay = bundle["replay_of"]
    if replay is not None:
        fields = {"source_result_id", "source_bundle_digest", "source_record_digest", "record_digest_matches"}
        if not isinstance(replay, dict) or replay.keys() != fields:
            raise ValueError("Invalid replay source fields")
        _identity(replay["source_result_id"], "result-")
        _hash(replay["source_bundle_digest"])
        _hash(replay["source_record_digest"])
        if replay["source_result_id"] == bundle["result_id"]:
            raise ValueError("Replay must create a new result identity")
        matches = replay["source_record_digest"] == bundle["record"]["record_digest"]
        if type(replay["record_digest_matches"]) is not bool or replay["record_digest_matches"] != matches:
            raise ValueError("Replay comparison does not match the retained record digests")
    return bundle


def import_model(source: Path, output: Path) -> dict:
    """Validate a received sealed artifact and retain it without resealing."""
    model = engine.load_model(source)
    identity = engine.runtime_identity()
    _write_immutable(output, model.to_dict())
    return {"model_file": str(output), "model_artifact_digest": model.artifact_digest,
            "artifact_schema": model.to_dict()["artifact_schema"], "runtime": identity}


def _evaluate(model: Any, sample: dict, output_dir: Path, source: dict | None = None) -> dict:
    sample = engine.validate_sample(model, sample)
    identity = engine.runtime_identity()
    record = engine.evaluate(model, sample)
    result_suffix = uuid.uuid4().hex
    replay = None
    if source is not None:
        replay = {"source_result_id": source["result_id"],
                  "source_bundle_digest": source["bundle_digest"],
                  "source_record_digest": source["record"]["record_digest"],
                  "record_digest_matches": source["record"]["record_digest"] == record["record_digest"]}
    bundle = {
        "bundle_schema": BUNDLE_SCHEMA, "instrument": "plsr",
        "evidence_id": _evidence_id(model.artifact_digest, sample), "operation_id": OPERATION_ID,
        "execution_id": "execution-" + uuid.uuid4().hex, "result_id": "result-" + result_suffix,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verification_id": None, "verification_status": "not_verified",
        "model": model.to_dict(), "sample": sample, "record": record,
        "runtime": identity, "replay_of": replay,
    }
    bundle["bundle_digest"] = _bundle_digest(bundle)
    _validate_bundle(bundle)
    path = Path(output_dir) / f"run-{result_suffix}.json"
    _write_immutable(path, bundle)
    return {"saved_file": str(path), "bundle": bundle}


def evaluate_run(model_path: Path, sample_path: Path, output_dir: Path) -> dict:
    """Evaluate explicit sample data once and save all inputs and evidence."""
    return _evaluate(engine.load_model(model_path), _read(sample_path), output_dir)


def evaluate_sample(model: Any, sample: dict, output_dir: Path) -> dict:
    """Evaluate an already loaded model and in-memory sample, retaining a bundle."""
    return _evaluate(model, sample, output_dir)


def inspect_run(path: Path) -> dict:
    """Check saved bindings without executing the scientific verdict again."""
    return {"saved_file": str(path), "bundle": _validate_bundle(_read(path))}


def replay_run(path: Path, output_dir: Path) -> dict:
    """Reevaluate retained inputs with the supported pin and new run identities."""
    source = inspect_run(path)["bundle"]
    return _evaluate(engine.model_from_dict(source["model"]), source["sample"], output_dir, source)
