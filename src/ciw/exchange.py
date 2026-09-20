"""Bounded, read-only inspection of the external instrument-exchange v1 seam.

This is not a session importer or an admission gate. Native CIW records and
covariance artifacts remain unchanged. A hash establishes content consistency,
not authorship, measurement truth, execution behavior, or verification authority.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib import resources
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import ModuleType


ADAPTER_VERSION = "ciw-exchange-inspector.v1"
MAX_ARTIFACT_BYTES = 1_048_576
MAX_ARTIFACTS = 32
MAX_COMPONENTS = 64
MAX_TOTAL_BYTES = 8_388_608
MAX_JSON_DEPTH = 64
OBSERVATION_SCHEMA = "notation.instrument.observation-batch.v1"
RESULT_SCHEMA = "notation.instrument.result-artifact.v1"
VERIFICATION_SCHEMA = "notation.instrument.verification-artifact.v1"
_SCHEMAS = {
    OBSERVATION_SCHEMA: ("batch_id", "validate_observation_batch"),
    RESULT_SCHEMA: ("result_id", "validate_result_artifact"),
    VERIFICATION_SCHEMA: ("verification_id", "validate_verification_artifact"),
}


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON member: {name}")
        result[name] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number overflows float64")
    return number


def _read(path: Path, limit: int) -> bytes:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"expected a regular file: {path}")
    # Recheck the actual descriptor: a pathname check alone can race a swap to
    # a FIFO/device. Nonblocking open prevents a substituted FIFO from hanging.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"expected a regular file: {path}")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"file exceeds {limit} bytes: {path}")
    return raw


def _validator(repo: Path) -> tuple[ModuleType, dict]:
    """Execute only allowlisted source bytes, never an artifact-supplied module.

    The module is standalone stdlib code. Loading the already-checked bytes
    avoids both importing a checkout's package initializer and a second source
    read between checking and execution. The local Python runtime is trusted.
    """
    manifest = json.loads(resources.files("ciw").joinpath("exchange-runtime.json").read_text())
    source = _read(repo / manifest["path"], 131_072).replace(b"\r\n", b"\n")
    if sha256(source).hexdigest() != manifest["sha256"]:
        raise ValueError("exchange validator source differs from the approved source pin")
    name = "_ciw_exchange_validator_" + manifest["sha256"]
    module = ModuleType(name)
    # dataclasses resolves annotations through sys.modules while creating types.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, str(repo / manifest["path"]), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module, manifest


def _identity(artifact: dict, field: str) -> str:
    if field == "batch_id":
        return "caller_declared_reference"
    payload = {key: value for key, value in artifact.items() if key != field}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    expected = "sha256:" + sha256(artifact["schema"].encode("utf-8") + b"\x00" + canonical).hexdigest()
    if artifact[field] != expected:
        raise ValueError(f"{field} does not match the artifact content")
    return "content_recomputed_not_authenticated"


def inspect_exchange(paths: list[Path], *, validator_repo: Path) -> dict:
    """Validate supported artifacts without mutating files or scientific state.

    Links name only supplied records. A matched reference is not proof of input
    consumption, source admission, or verifier independence. Failed verification
    claims are valid records and are never turned into a successful verdict.
    """
    if not 1 <= len(paths) <= MAX_ARTIFACTS:
        raise ValueError(f"supply between 1 and {MAX_ARTIFACTS} artifacts")
    parsed = []
    total = 0
    for value in paths:
        path = Path(value)
        raw = _read(path, MAX_ARTIFACT_BYTES)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("combined exchange inputs exceed the byte budget")
        try:
            artifact = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                                  parse_constant=_constant, parse_float=_float)
        except (UnicodeDecodeError, RecursionError) as exc:
            raise ValueError("exchange input must be bounded UTF-8 JSON") from exc
        pending = [(artifact, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > MAX_JSON_DEPTH:
                raise ValueError("exchange input must have bounded JSON nesting")
            if isinstance(node, dict):
                pending.extend((item, depth + 1) for item in node.values())
            elif isinstance(node, list):
                pending.extend((item, depth + 1) for item in node)
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("schema"), str)
                or artifact["schema"] not in _SCHEMAS):
            raise ValueError("unsupported instrument-exchange schema")
        if artifact["schema"] != VERIFICATION_SCHEMA:
            components = artifact.get("components")
            if not isinstance(components, list) or not 1 <= len(components) <= MAX_COMPONENTS:
                raise ValueError(f"components must contain 1 to {MAX_COMPONENTS} quantities")
        parsed.append((path, raw, artifact))

    validator, manifest = _validator(Path(validator_repo))
    records = []
    by_id = {}
    for path, raw, artifact in parsed:
        identity_field, check = _SCHEMAS[artifact["schema"]]
        covariance = getattr(validator, check)(artifact)
        identity_status = _identity(artifact, identity_field)
        identity = artifact[identity_field]
        if identity in by_id:
            raise ValueError("duplicate or ambiguous artifact identity in inspection inputs")
        by_id[identity] = artifact["schema"]
        report = {
            "path": str(path),
            "source_bytes_sha256": sha256(raw).hexdigest(),
            "source_bytes_count": len(raw),
            "identity_field": identity_field,
            "identity_status": identity_status,
            "artifact": artifact,
            "conformance": "passed",
            "covariance_validation": asdict(covariance) if covariance is not None else None,
        }
        records.append(report)

    links = []
    for report in records:
        artifact = report["artifact"]
        if artifact["schema"] == RESULT_SCHEMA:
            references = [("input_refs", reference) for reference in artifact["input_refs"]]
        elif artifact["schema"] == VERIFICATION_SCHEMA:
            references = [("subject_ref", artifact["subject_ref"])]
        else:
            references = [("source_artifact_refs", reference)
                          for reference in artifact["source_artifact_refs"]]
        for field, reference in references:
            links.append({"from": artifact[report["identity_field"]], "field": field,
                          "to": reference, "target_schema": by_id.get(reference),
                          "status": "matched_supplied_reference" if reference in by_id
                          else "unresolved_external_reference"})

    return {
        "schema": "ciw.exchange-inspection.v1",
        "adapter_version": ADAPTER_VERSION,
        "status": "conformant",
        "validator": manifest,
        "artifacts": records,
        "links": links,
        "authority": {
            "may_authorize": False,
            "native_workspace_import": "not_performed",
            "source_admission": "not_assessed",
            "execution_behavior": "not_assessed",
            "verification_independence": "not_established",
            "physical_validation": "not_established",
        },
        "limits": {"max_artifact_bytes": MAX_ARTIFACT_BYTES,
                   "max_total_bytes": MAX_TOTAL_BYTES,
                   "max_artifacts": MAX_ARTIFACTS, "max_components": MAX_COMPONENTS,
                   "max_json_depth": MAX_JSON_DEPTH},
    }
