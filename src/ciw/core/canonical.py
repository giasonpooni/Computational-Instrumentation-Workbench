"""Canonical record content: the kernel's identity utilities.

Every retained workbench record (sources, steps, results, bundles,
verifications, replay receipts, the project graph) is identified by the
SHA-256 of its canonical JSON bytes: sorted keys, no insignificant whitespace,
UTF-8 without ASCII escaping, no NaN or infinities, string keys only and at
most 64 levels of nesting. Changing any rule here changes every retained
identity, so this module is part of the frozen kernel.

``core.identities.canonical_json`` is the older ``run.v1`` evidence
canonicalization (ASCII-escaped). Both remain because each is bound into
identities that already exist; neither may be substituted for the other.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math

MAX_DEPTH = 64


def canonical(value) -> bytes:
    """Canonical JSON bytes of a record."""
    pending = [(value, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise ValueError("Canonical JSON exceeds the nesting budget")
        if isinstance(node, dict):
            if any(not isinstance(key, str) for key in node):
                raise ValueError("Canonical JSON keys must be strings")
            pending.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, (list, tuple)):
            pending.extend((child, depth + 1) for child in node)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    """Content identity of a record."""
    return "sha256:" + sha256(canonical(value)).hexdigest()


def byte_digest(raw: bytes) -> str:
    """Content identity of retained bytes."""
    return "sha256:" + sha256(raw).hexdigest()


def bundle_digest(bundle: dict) -> str:
    """Identity of a session bundle; its verification and replay receipts refer to it."""
    return digest({key: value for key, value in bundle.items()
                   if key not in {"bundle_digest", "verification", "replay_receipts"}})


def exact_keys(value, required, optional=()) -> None:
    """Require a mapping with every required key and nothing undeclared."""
    if not isinstance(value, dict) or not set(required) <= value.keys() <= set(required) | set(optional):
        raise ValueError("Invalid record fields; require " + ", ".join(sorted(required)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_instant(epoch: str, seconds) -> str:
    """The instant ``seconds`` after ``epoch``, refused unless exact at microseconds."""
    if type(seconds) not in (int, float) or not math.isfinite(seconds):
        raise ValueError("Mapped time must be a finite real number")
    instant = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    mapped = instant + timedelta(seconds=seconds)
    if (mapped - instant).total_seconds() != seconds:
        raise ValueError("Mapped time cannot be represented at microsecond precision")
    return mapped.isoformat().replace("+00:00", "Z")
