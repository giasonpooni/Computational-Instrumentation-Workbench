"""Content identities and distinct workbench event identities.

Content integrity is deliberately separate from physical validity, authenticity,
execution, and verification. Render representations never change run evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
import uuid


SCIENTIFIC_FIELDS = ("instrument", "metadata", "time_s", "channels")
EVENT_KINDS = frozenset({"session", "execution", "result", "verification", "decision"})


def canonical_json(value: Any) -> str:
    """Return the existing CIW v1 content canonicalization without changing hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def content_identity(value: Any) -> str:
    return "sha256:" + digest(value)


def evidence_id(run: dict) -> str:
    """Bind all retained scientific content, including adapter/calibration metadata."""
    return content_identity({key: run[key] for key in SCIENTIFIC_FIELDS})


def validate_evidence_identity(run: dict) -> None:
    if run.get("evidence_id") != evidence_id(run):
        raise ValueError("Evidence integrity mismatch: scientific content does not match evidence_id")


def new_identity(kind: str) -> str:
    """Allocate a new event identity; evidence and operation identities are not events."""
    if kind not in EVENT_KINDS:
        raise ValueError(f"Unsupported event identity kind: {kind}")
    return kind + "-" + uuid.uuid4().hex


def validate_identity(value: Any, kind: str) -> str:
    if kind not in EVENT_KINDS:
        raise ValueError(f"Unsupported event identity kind: {kind}")
    prefix = kind + "-"
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) != len(prefix) + 32:
        raise ValueError(f"Invalid {kind} identity")
    try:
        valid = uuid.UUID(hex=value[len(prefix):]).hex == value[len(prefix):]
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid {kind} identity")
    return value
