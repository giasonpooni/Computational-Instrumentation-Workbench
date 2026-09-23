"""Append-only, hash-chained evidence ledger with exact content-addressed bytes.

Layout of a ledger directory::

    ledger.jsonl              one canonical-JSON entry per line
    blobs/<aa>/<64 hex>       exact retained bytes, named by their SHA-256

Each entry binds its predecessor's identity, so editing, removing or
reordering any line breaks every later identity. Blob names are validated as
hexadecimal digests before any path is formed, which excludes traversal.
References may only point to earlier entries, so evidence graphs are acyclic.

Stored bytes are never rewritten. A body schema may later gain a successor; a
registered migration produces a *view* in the new schema while the retained
line keeps the bytes that were originally hashed.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ._common import Refusal, bounded_json, canonical_json, content_identity, mapping, plain, text, utc_now

ENTRY_SCHEMA = "ciw.ledger-entry.v1"
LEDGER_FILE = "ledger.jsonl"
BLOB_DIR = "blobs"
MAX_ENTRY_BYTES = 1 << 20
MAX_BLOB_BYTES = 64 << 20
MAX_ENTRIES = 1_000_000
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")

KINDS = frozenset({
    "source_evidence", "runtime_identity", "parameter_identity", "experiment_spec", "execution_plan",
    "execution", "numerical_result", "verification", "replay_receipt", "observation", "calibration",
    "frame_registry", "fusion_state", "claim", "decision_proposal", "authority_decision", "agent_proposal",
    "proposal_disposition", "report", "acquisition_request", "hardware_capture", "governance", "migration",
    "design_recommendation", "use_case_requirements", "telemetry", "physical_protocol",
})


def _require(body: dict, name: str, keys: set[str]) -> None:
    missing = keys - body.keys()
    if missing:
        raise Refusal("malformed_entry", f"{name} body is missing {sorted(missing)}")


def _status(allowed: set[str]) -> Callable[[dict], None]:
    def check(body: dict) -> None:
        if body.get("status") not in allowed:
            raise Refusal("malformed_entry", f"status must be one of {sorted(allowed)}")
    return check


# Minimal structural contracts per body schema. Domain modules validate meaning;
# the ledger only guarantees each kind carries the fields its readers rely on.
BODY_SCHEMAS: dict[str, tuple[str, set[str], Callable[[dict], None] | None]] = {
    "ciw.science.source-evidence.v1": ("source_evidence", {"label", "blob", "media_type"}, None),
    "ciw.science.runtime-identity.v1": ("runtime_identity", {"runtime"}, None),
    "ciw.science.parameter-identity.v1": ("parameter_identity", {"parameters"}, None),
    "ciw.science.experiment-spec.v1": ("experiment_spec", {"spec", "spec_identity"}, None),
    "ciw.science.execution-plan.v1": ("execution_plan", {"plan_identity", "jobs"}, None),
    "ciw.science.execution.v1": ("execution", {"job_id", "solver_id", "status"},
                                 _status({"completed", "refused"})),
    "ciw.science.numerical-result.v1": ("numerical_result", {"job_id", "solver_id", "result", "result_identity"}, None),
    "ciw.science.verification.v1": ("verification", {"subject", "verdicts", "status"},
                                    _status({"passed", "failed", "incomplete"})),
    "ciw.science.replay-receipt.v1": ("replay_receipt", {"original", "status", "comparison"},
                                      _status({"reproduced", "within_tolerance", "diverged", "refused"})),
    "ciw.science.observation.v1": ("observation", {"observable", "acquisition", "value", "unit"}, None),
    "ciw.science.calibration.v1": ("calibration", {"calibration_id", "version"}, None),
    "ciw.science.frame-registry.v1": ("frame_registry", {"registry", "registry_identity"}, None),
    "ciw.science.fusion-state.v1": ("fusion_state", {"stage", "state"}, None),
    "ciw.science.claim.v1": ("claim", {"statement", "claim_class", "status"}, None),
    "ciw.science.decision-proposal.v1": ("decision_proposal", {"action", "proposer"}, None),
    "ciw.science.authority-decision.v1": ("authority_decision", {"proposal", "authorized", "reasons"}, None),
    "ciw.science.agent-proposal.v1": ("agent_proposal", {"role", "proposal_kind", "payload"}, None),
    "ciw.science.proposal-disposition.v1": ("proposal_disposition", {"proposal", "disposition", "reasons"}, None),
    "ciw.science.report.v1": ("report", {"blob", "format", "ledger_head"}, None),
    "ciw.science.acquisition-request.v1": ("acquisition_request", {"instrument", "observable", "status"}, None),
    "ciw.science.hardware-capture.v1": ("hardware_capture", {"blob", "summary"}, None),
    "ciw.science.governance.v1": ("governance", {"action", "subject"}, None),
    "ciw.science.migration.v1": ("migration", {"from_schema", "to_schema"}, None),
    "ciw.science.design-recommendation.v1": ("design_recommendation", {"ranking"}, None),
    "ciw.science.use-case-requirements.v1": ("use_case_requirements", {"use_case", "requirements"}, None),
    "ciw.science.telemetry.v1": ("telemetry", {"channels"}, None),
    "ciw.science.physical-protocol.v1": ("physical_protocol", {"protocol", "protocol_identity"}, None),
}
# (from body schema) -> (to body schema, pure function producing the new body)
MIGRATIONS: dict[str, tuple[str, Callable[[dict], dict]]] = {}


def register_body_schema(schema: str, kind: str, required: set[str],
                         check: Callable[[dict], None] | None = None) -> None:
    if schema in BODY_SCHEMAS:
        raise Refusal("duplicate_schema", f"Body schema {schema!r} is already registered")
    if kind not in KINDS:
        raise Refusal("unknown_kind", f"Entry kind {kind!r} is not declared")
    BODY_SCHEMAS[schema] = (kind, set(required), check)


def register_migration(source: str, target: str, function: Callable[[dict], dict]) -> None:
    if source not in BODY_SCHEMAS or target not in BODY_SCHEMAS:
        raise Refusal("unknown_schema", "Both migration schemas must be registered")
    if BODY_SCHEMAS[source][0] != BODY_SCHEMAS[target][0]:
        raise Refusal("migration_kind_change", "A migration cannot change the entry kind")
    if source in MIGRATIONS:
        raise Refusal("duplicate_migration", f"{source!r} already migrates to {MIGRATIONS[source][0]!r}")
    MIGRATIONS[source] = (target, function)


def entry_identity(entry: dict) -> str:
    return content_identity({key: value for key, value in entry.items() if key != "entry_id"})


def blob_identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    entries: int
    blobs: int
    head: str | None
    problems: tuple[dict, ...]

    def to_json(self) -> dict:
        return {"ok": self.ok, "entries": self.entries, "blobs": self.blobs, "head": self.head,
                "problems": list(self.problems)}


class Ledger:
    """A single-writer append-only ledger rooted at a directory."""

    def __init__(self, root: Path, *, now: Callable[[], Any] = utc_now, entries: list[dict] | None = None):
        self.root = Path(root)
        self._now = now
        self._entries: list[dict] = entries or []
        self._index = {entry["entry_id"]: entry for entry in self._entries}

    # -------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, root: Path, *, now: Callable[[], Any] = utc_now) -> "Ledger":
        root = Path(root)
        if (root / LEDGER_FILE).exists():
            raise Refusal("ledger_exists", f"{root} already holds a ledger; open it instead")
        (root / BLOB_DIR).mkdir(parents=True, exist_ok=True)
        (root / LEDGER_FILE).touch()
        return cls(root, now=now)

    @classmethod
    def open(cls, root: Path, *, now: Callable[[], Any] = utc_now, strict: bool = True) -> "Ledger":
        root = Path(root)
        path = root / LEDGER_FILE
        if not path.is_file():
            raise Refusal("ledger_missing", f"No ledger at {root}")
        entries, problems = _read_entries(path)
        if strict and problems:
            raise Refusal("ledger_tampered", "Ledger bytes are not canonical, complete entries", problems=problems[:20])
        ledger = cls(root, now=now, entries=[entry for entry in entries if isinstance(entry, dict) and "entry_id" in entry])
        if strict:
            report = ledger.verify(problems)
            if not report.ok:
                raise Refusal("ledger_tampered", "Ledger integrity verification failed", problems=report.problems[:20])
        return ledger

    @classmethod
    def open_or_create(cls, root: Path, **options: Any) -> "Ledger":
        return cls.open(root, **options) if (Path(root) / LEDGER_FILE).exists() else cls.create(root, **options)

    # -------------------------------------------------------------- blobs
    def _blob_path(self, identity: str) -> Path:
        if not isinstance(identity, str) or IDENTITY.fullmatch(identity) is None:
            raise Refusal("malformed_identity", "Blob identity must be sha256:<64 lowercase hex>")
        hexdigest = identity[7:]
        return self.root / BLOB_DIR / hexdigest[:2] / hexdigest

    def put_blob(self, data: bytes) -> str:
        if not isinstance(data, (bytes, bytearray)):
            raise Refusal("malformed_record", "Blob content must be bytes")
        if len(data) > MAX_BLOB_BYTES:
            raise Refusal("oversized_artifact", f"Blob exceeds {MAX_BLOB_BYTES} bytes")
        identity = blob_identity(bytes(data))
        path = self._blob_path(identity)
        if path.exists():
            if blob_identity(path.read_bytes()) != identity:
                raise Refusal("blob_tampered", f"Retained blob {identity} no longer matches its identity")
            return identity
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return identity

    def get_blob(self, identity: str) -> bytes:
        path = self._blob_path(identity)
        if not path.is_file():
            raise Refusal("blob_missing", f"Retained blob {identity} is absent")
        if path.stat().st_size > MAX_BLOB_BYTES:
            raise Refusal("oversized_artifact", f"Blob {identity} exceeds the size bound")
        data = path.read_bytes()
        if blob_identity(data) != identity:
            raise Refusal("blob_tampered", f"Retained blob {identity} no longer matches its identity")
        return data

    def put_json_blob(self, value: Any) -> str:
        return self.put_blob(canonical_json(plain(value)).encode("utf-8"))

    def get_json_blob(self, identity: str) -> Any:
        return json.loads(self.get_blob(identity))

    # -------------------------------------------------------------- entries
    def append(self, schema: str, body: dict, *, refs: list[str] | tuple[str, ...] = (),
               blobs: list[str] | tuple[str, ...] = ()) -> dict:
        if schema not in BODY_SCHEMAS:
            raise Refusal("unknown_schema", f"Body schema {schema!r} is not registered")
        if len(self._entries) >= MAX_ENTRIES:
            raise Refusal("oversized_input", "Ledger reached its entry bound")
        kind, required, check = BODY_SCHEMAS[schema]
        body = plain(mapping(body, "entry body"))
        _require(body, schema, required)
        if check is not None:
            check(body)
        refs, blobs = list(refs), list(blobs)
        for ref in refs:
            if ref not in self._index:
                raise Refusal("dangling_reference", f"Reference {ref} is not an earlier ledger entry")
        for identity in blobs:
            if not self._blob_path(identity).is_file():
                raise Refusal("blob_missing", f"Blob {identity} must be retained before it is referenced")
        entry = {
            "schema": ENTRY_SCHEMA, "sequence": len(self._entries), "kind": kind, "body_schema": schema,
            "previous": self._entries[-1]["entry_id"] if self._entries else None,
            "recorded_at": self._now().isoformat(), "refs": sorted(set(refs)), "blobs": sorted(set(blobs)),
            "body": body,
        }
        entry["entry_id"] = entry_identity(entry)
        line = bounded_json(entry, "ledger entry", MAX_ENTRY_BYTES)
        with (self.root / LEDGER_FILE).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._entries.append(entry)
        self._index[entry["entry_id"]] = entry
        return entry

    def get(self, entry_id: str) -> dict:
        if entry_id not in self._index:
            raise Refusal("unknown_entry", f"No ledger entry {entry_id}")
        return self._index[entry_id]

    def entries(self, kind: str | None = None) -> Iterator[dict]:
        for entry in self._entries:
            if kind is None or entry["kind"] == kind:
                yield entry

    def head(self) -> str | None:
        return self._entries[-1]["entry_id"] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    def referrers(self, entry_id: str, kind: str | None = None) -> list[dict]:
        return [entry for entry in self._entries if entry_id in entry["refs"] and (kind is None or entry["kind"] == kind)]

    def closure(self, entry_ids: list[str]) -> list[dict]:
        """All entries reachable through references, in ledger order."""
        pending, seen = list(entry_ids), set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.get(current)["refs"])
        return [entry for entry in self._entries if entry["entry_id"] in seen]

    def view(self, entry: dict) -> dict:
        """The entry body in its newest registered schema; stored bytes stay unchanged."""
        schema, body, applied = entry["body_schema"], entry["body"], []
        while schema in MIGRATIONS:
            target, function = MIGRATIONS[schema]
            body = function(json.loads(canonical_json(body)))
            _require(body, target, BODY_SCHEMAS[target][1])
            applied.append({"from": schema, "to": target})
            schema = target
        return {"entry_id": entry["entry_id"], "kind": entry["kind"], "body_schema": schema, "body": body,
                "migrations": applied, "stored_body_schema": entry["body_schema"]}

    # -------------------------------------------------------------- verification
    def verify(self, read_problems: list[dict] | None = None) -> VerifyReport:
        problems = list(read_problems or [])
        previous, seen, blobs = None, set(), set()
        for position, entry in enumerate(self._entries):
            found = _entry_problems(entry, position, previous, seen)
            problems.extend(found)
            previous = entry.get("entry_id") if isinstance(entry, dict) else None
            seen.add(previous)
            if not any(item["problem"] in {"malformed_entry_fields", "malformed_blob_identity", "not_an_object"}
                       for item in found):
                blobs.update(entry["blobs"])
        for identity in sorted(blobs):
            try:
                self.get_blob(identity)
            except Refusal as exc:
                problems.append({"blob": identity, "problem": exc.code})
        blob_root = self.root / BLOB_DIR
        if blob_root.is_dir():
            for path in blob_root.rglob("*"):
                if path.is_file() and (path.suffix == ".partial" or re.fullmatch(r"[0-9a-f]{64}", path.name) is None
                                       or path.parent.name != path.name[:2]):
                    problems.append({"path": str(path.relative_to(self.root)), "problem": "unexpected_file"})
        return VerifyReport(not problems, len(self._entries), len(blobs), self.head(), tuple(problems))


def _entry_problems(entry: Any, position: int, previous: str | None, seen: set[str]) -> list[dict]:
    if not isinstance(entry, dict):
        return [{"sequence": position, "problem": "not_an_object"}]
    problems = []
    if entry.get("schema") != ENTRY_SCHEMA:
        problems.append({"sequence": position, "problem": "unsupported_entry_schema"})
    if entry.get("sequence") != position:
        problems.append({"sequence": position, "problem": "sequence_gap"})
    if entry.get("previous") != previous:
        problems.append({"sequence": position, "problem": "broken_chain"})
    if entry.get("entry_id") != entry_identity(entry):
        problems.append({"sequence": position, "problem": "identity_mismatch"})
    schema = entry.get("body_schema")
    if schema not in BODY_SCHEMAS or BODY_SCHEMAS[schema][0] != entry.get("kind"):
        problems.append({"sequence": position, "problem": "unknown_body_schema"})
    refs, blobs = entry.get("refs"), entry.get("blobs")
    if not isinstance(refs, list) or not isinstance(blobs, list) or not isinstance(entry.get("body"), dict):
        return problems + [{"sequence": position, "problem": "malformed_entry_fields"}]
    for ref in refs:
        if not isinstance(ref, str) or ref not in seen:
            problems.append({"sequence": position, "problem": "forward_or_dangling_reference", "ref": str(ref)})
    for identity in blobs:
        if not isinstance(identity, str) or IDENTITY.fullmatch(identity) is None:
            problems.append({"sequence": position, "problem": "malformed_blob_identity"})
    return problems


def _read_entries(path: Path) -> tuple[list[dict], list[dict]]:
    entries, problems = [], []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream):
            if len(raw) > MAX_ENTRY_BYTES + 1:
                problems.append({"line": number, "problem": "oversized_entry"})
                continue
            if not raw.endswith(b"\n"):
                problems.append({"line": number, "problem": "truncated_entry"})
            try:
                entry = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                problems.append({"line": number, "problem": "unparseable_entry"})
                continue
            if isinstance(entry, dict) and canonical_json(entry).encode("utf-8") != raw.rstrip(b"\n"):
                problems.append({"line": number, "problem": "noncanonical_bytes"})
            entries.append(entry)
    return entries, problems


def source_evidence(ledger: Ledger, label: str, data: bytes, media_type: str, **context: Any) -> dict:
    """Retain exact source bytes and record them as evidence."""
    text(label, "label", 256)
    text(media_type, "media_type", 128)
    identity = ledger.put_blob(data)
    return ledger.append("ciw.science.source-evidence.v1",
                         {"label": label, "blob": identity, "media_type": media_type, "bytes": len(data), **context},
                         blobs=[identity])
