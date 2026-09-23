"""Offline evidence bundles and signed updates for low-resource, delayed synchronization.

Scope. ``export_bundle`` writes a deterministic ZIP (``.ciwb``) holding ledger
lines byte-identical to the source, exactly the blobs those lines reference, a
canonical-JSON manifest (``ciw.evidence-bundle.v1``) listing every member's
SHA-256 and size, and optionally an Ed25519 signature over the exact manifest
bytes. ``inspect_bundle`` verifies a bundle offline and read-only, never
extracting to paths taken from the archive; ``import_bundle`` materializes a
clean, complete and (by default) trusted bundle as a new ledger.
``sign_update``/``verify_update`` carry schema libraries or provider pins between
offline machines under the same keys.

Limits. A valid signature shows that the holder of a key the caller trusts
signed the manifest, not that the evidence is true. Key distribution, rotation
and revocation are out of scope (``trusted_keys`` is a caller-supplied map), and
the Ed25519 code is pure Python and not constant time. Byte-identical re-export
assumes the same zlib; the signature covers uncompressed member digests, so it
does not depend on the compressor. A subset bundle proves entry identities and
in-bundle references but not chain continuity, so it inspects but never imports.
Reads are bounded per member, in total and by compression ratio, so a large but
extremely compressible legitimate blob is refused rather than trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
from typing import Any
import zipfile
import zlib

from . import ed25519
from ._common import Refusal, bounded_json, canonical_json, integer, mapping, require_keys, text
from .ledger import (BLOB_DIR, BODY_SCHEMAS, ENTRY_SCHEMA, IDENTITY, LEDGER_FILE, MAX_BLOB_BYTES as LEDGER_BLOB_BYTES,
                     Ledger, blob_identity, entry_identity)

MANIFEST_SCHEMA = "ciw.evidence-bundle.v1"
SIGNATURE_SCHEMA = "ciw.bundle-signature.v1"
INSPECTION_SCHEMA = "ciw.bundle-inspection.v1"
EXPORT_SCHEMA = "ciw.bundle-export.v1"
UPDATE_SCHEMA = "ciw.signed-update.v1"
MANIFEST, LEDGER_MEMBER, SIGNATURE = "manifest.json", "ledger.jsonl", "signature.json"
MEMBER = re.compile(r"manifest\.json|ledger\.jsonl|signature\.json|blobs/[0-9a-f]{64}")
DATE_TIME = (1980, 1, 1, 0, 0, 0)
EXTERNAL_ATTR = 0o100644 << 16
COMPRESS_LEVEL = 9
MAX_BUNDLE_BYTES = 2 << 30
MAX_MEMBERS = 65000
MAX_CENTRAL_DIRECTORY = 16 << 20
MAX_MANIFEST_BYTES = 16 << 20
MAX_SIGNATURE_BYTES = 4096
MAX_LEDGER_BYTES = 256 << 20
MAX_BLOB_BYTES = LEDGER_BLOB_BYTES
MAX_TOTAL_BYTES = 1 << 30
MAX_RATIO = 200
RATIO_FLOOR = 1 << 20
MAX_PROBLEMS = 1000
MAX_UPDATE_METADATA = 64 << 10
MAX_UPDATE_PAYLOAD = 256 << 20
SIGNATURE_STATUSES = ("valid", "invalid", "unsigned", "untrusted_key")
_DRIVE = re.compile(r"[A-Za-z]:")


# ---------------------------------------------------------------------- keys
def _secret(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
        raise Refusal("malformed_key", "An Ed25519 secret key is 32 bytes")
    return bytes(value)


def _hex(value: Any, size: int, name: str) -> bytes:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{2 * size}}}", value) is None:
        raise Refusal("malformed_record", f"{name} must be {2 * size} lowercase hex digits")
    return bytes.fromhex(value)


def _trusted(value: Any) -> dict[str, bytes]:
    if value is None:
        return {}
    result = {}
    for key, public in mapping(value, "trusted_keys").items():
        text(key, "trusted key id", 128)
        if not isinstance(public, (bytes, bytearray)) or len(public) != 32:
            raise Refusal("malformed_key", f"Trusted key {key!r} must be a 32-byte Ed25519 public key")
        result[key] = bytes(public)
    return result


def key_fingerprint(public: bytes) -> str:
    """Default key id: ``ed25519:`` plus the first 32 hex digits of SHA-256(public key)."""
    return "ed25519:" + hashlib.sha256(public).hexdigest()[:32]


# ---------------------------------------------------------------------- export
def _verified_lines(ledger: Ledger) -> dict[str, bytes]:
    report = ledger.verify()
    if not report.ok:
        raise Refusal("ledger_unverified", "Only a verified ledger can be exported", problems=list(report.problems[:20]))
    path = ledger.root / LEDGER_FILE
    if path.stat().st_size > MAX_LEDGER_BYTES:
        raise Refusal("oversized_input", f"Ledger exceeds {MAX_LEDGER_BYTES} bytes")
    lines = {entry["entry_id"]: (canonical_json(entry) + "\n").encode("utf-8") for entry in ledger.entries()}
    if path.read_bytes() != b"".join(lines.values()):
        raise Refusal("ledger_noncanonical", "On-disk ledger bytes differ from the verified in-memory entries")
    return lines


def _write(archive: zipfile.ZipFile, name: str, data: bytes) -> dict:
    info = zipfile.ZipInfo(name, date_time=DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    if len(data) > RATIO_FLOOR:
        packer = zlib.compressobj(COMPRESS_LEVEL, zlib.DEFLATED, -15)
        if len(data) > MAX_RATIO * max(1, len(packer.compress(data)) + len(packer.flush())):
            info.compress_type = zipfile.ZIP_STORED  # never emit what the inspector refuses as a bomb
    info.create_system = 3
    info.external_attr = EXTERNAL_ATTR
    archive.writestr(info, data, compresslevel=COMPRESS_LEVEL)
    return {"name": name, "identity": blob_identity(data), "size": len(data)}


def export_bundle(ledger: Ledger, destination: Path, *, entry_ids: list[str] | None = None,
                  signing_key: bytes | None = None, key_id: str | None = None) -> dict:
    """Write a deterministic ``.ciwb`` bundle of the whole ledger or of the closure of ``entry_ids``."""
    if not isinstance(ledger, Ledger):
        raise Refusal("malformed_record", "ledger must be a Ledger")
    destination = Path(destination)
    if destination.exists():
        raise Refusal("destination_exists", f"{destination} already exists; bundles are never overwritten")
    if signing_key is None and key_id is not None:
        raise Refusal("malformed_record", "key_id names a signing key, but none was given")
    secret = None if signing_key is None else _secret(signing_key)
    public = None if secret is None else ed25519.public_key(secret)
    if secret is not None:
        key_id = key_fingerprint(public) if key_id is None else text(key_id, "key_id", 128)
    lines = _verified_lines(ledger)
    if entry_ids is None:
        selected, requested = list(ledger.entries()), None
    else:
        if not isinstance(entry_ids, (list, tuple)) or not entry_ids:
            raise Refusal("malformed_record", "entry_ids must be a nonempty list of entry identities")
        for entry_id in entry_ids:
            if not isinstance(entry_id, str) or IDENTITY.fullmatch(entry_id) is None:
                raise Refusal("malformed_identity", "entry_ids must be sha256:<64 lowercase hex>")
        selected, requested = ledger.closure(list(entry_ids)), sorted(set(entry_ids))
    blob_ids = sorted({identity for entry in selected for identity in entry["blobs"]})
    if len(blob_ids) + 3 > MAX_MEMBERS:
        raise Refusal("oversized_input", f"A bundle holds at most {MAX_MEMBERS} members", blobs=len(blob_ids))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = destination.open("xb")
    except FileExistsError as exc:
        raise Refusal("destination_exists", f"{destination} already exists; bundles are never overwritten") from exc
    try:
        with handle, zipfile.ZipFile(handle, "w") as archive:
            members = [_write(archive, "blobs/" + identity[7:], ledger.get_blob(identity)) for identity in blob_ids]
            members.append(_write(archive, LEDGER_MEMBER, b"".join(lines[entry["entry_id"]] for entry in selected)))
            if sum(item["size"] for item in members) > MAX_TOTAL_BYTES:
                raise Refusal("oversized_input", f"Bundle content exceeds {MAX_TOTAL_BYTES} bytes")
            manifest = {"schema": MANIFEST_SCHEMA, "entry_schema": ENTRY_SCHEMA, "complete": len(selected) == len(ledger),
                        "entries": len(selected), "head": selected[-1]["entry_id"] if selected else None,
                        "source_head": ledger.head(), "source_entries": len(ledger), "selected": requested,
                        "members": sorted(members, key=lambda item: item["name"])}
            manifest_bytes = canonical_json(manifest).encode("utf-8")
            _write(archive, MANIFEST, manifest_bytes)
            if secret is not None:
                signature = {"schema": SIGNATURE_SCHEMA, "key_id": key_id, "public_key": public.hex(),
                             "signature": ed25519.sign(secret, manifest_bytes).hex()}
                _write(archive, SIGNATURE, canonical_json(signature).encode("utf-8"))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"schema": EXPORT_SCHEMA, "path": str(destination), "bundle_identity": _file_identity(destination),
            "bytes": destination.stat().st_size, "manifest_identity": blob_identity(manifest_bytes),
            "complete": manifest["complete"], "entries": len(selected), "blobs": len(blob_ids),
            "signed": secret is not None, "key_id": key_id}


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ---------------------------------------------------------------------- inspection
def _unsafe(name: str) -> bool:
    parts = name.split("/")
    return (not name or "\x00" in name or "\\" in name or name.startswith("/") or _DRIVE.match(name) is not None
            or any(part in ("", ".", "..") for part in parts[:-1]) or parts[-1] in (".", ".."))


def _limit(name: str) -> int:
    return {MANIFEST: MAX_MANIFEST_BYTES, SIGNATURE: MAX_SIGNATURE_BYTES,
            LEDGER_MEMBER: MAX_LEDGER_BYTES}.get(name, MAX_BLOB_BYTES)


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    """Read one member with a hard limit; the declared size was already checked against it."""
    try:
        with archive.open(info) as stream:
            data = stream.read(limit + 1)
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError, NotImplementedError, ValueError) as exc:
        raise Refusal("unreadable_member", f"{info.orig_filename}: {exc}") from exc
    if len(data) > limit:
        raise Refusal("oversized_member", f"{info.orig_filename} exceeds {limit} bytes")
    if len(data) != info.file_size:
        raise Refusal("unreadable_member", f"{info.orig_filename} size differs from its directory entry")
    return data


def _central_directory(path: Path, size: int) -> int:
    """Bound the central directory from its end record before zipfile parses every entry of it."""
    with path.open("rb") as stream:
        stream.seek(max(0, size - 22 - 65535))
        tail = stream.read()
    index = len(tail) - 22 if tail[-22:-18] == b"PK\x05\x06" else tail.rfind(b"PK\x05\x06")
    if index < 0 or len(tail) - index < 22:
        raise Refusal("bundle_unreadable", f"{path} has no ZIP end-of-central-directory record")
    if index >= 20 and tail[index - 20:index - 16] == b"PK\x06\x07":
        raise Refusal("bundle_unreadable", "ZIP64 archives are not evidence bundles")
    entries, directory = struct.unpack_from("<HI", tail, index + 10)
    if entries > MAX_MEMBERS or directory > MAX_CENTRAL_DIRECTORY:
        raise Refusal("oversized_input", "Bundle central directory exceeds its bound", entries=entries,
                      directory_bytes=directory)
    return entries


def _open_archive(path: Any) -> zipfile.ZipFile:
    path = Path(path)
    if not path.is_file():
        raise Refusal("bundle_missing", f"No bundle at {path}")
    size = path.stat().st_size
    if size > MAX_BUNDLE_BYTES:
        raise Refusal("oversized_input", f"Bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    declared = _central_directory(path, size)
    try:
        archive = zipfile.ZipFile(path, "r")
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError, EOFError) as exc:
        raise Refusal("bundle_unreadable", f"{path} is not a readable ZIP bundle: {exc}") from exc
    if len(archive.infolist()) != declared:
        archive.close()
        raise Refusal("bundle_unreadable", "The central directory hides or invents members relative to its end record",
                      declared=declared)
    return archive


def _manifest(data: bytes) -> dict:
    try:
        manifest = json.loads(data)
        if canonical_json(manifest).encode("utf-8") != data:
            raise Refusal("malformed_record", "manifest.json is not canonical JSON")
        require_keys(manifest, "manifest", {"schema", "entry_schema", "complete", "entries", "head", "source_head",
                                            "source_entries", "selected", "members"})
        if manifest["schema"] != MANIFEST_SCHEMA or manifest["entry_schema"] != ENTRY_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {MANIFEST_SCHEMA} over {ENTRY_SCHEMA}")
        if not isinstance(manifest["complete"], bool):
            raise Refusal("malformed_record", "complete must be a boolean")
        integer(manifest["entries"], "entries", minimum=0)
        integer(manifest["source_entries"], "source_entries", minimum=0)
        for key in ("head", "source_head"):
            if manifest[key] is not None and (not isinstance(manifest[key], str) or not IDENTITY.fullmatch(manifest[key])):
                raise Refusal("malformed_identity", f"{key} must be an entry identity or null")
        selected = manifest["selected"]
        if selected is not None and (not isinstance(selected, list) or selected != sorted(set(selected)) or any(
                not isinstance(item, str) or not IDENTITY.fullmatch(item) for item in selected)):
            raise Refusal("malformed_record", "selected must be a sorted list of unique entry identities or null")
        members = manifest["members"]
        if not isinstance(members, list) or len(members) > MAX_MEMBERS:
            raise Refusal("malformed_record", "members must be a bounded array")
        names = []
        for item in members:
            require_keys(item, "manifest member", {"name", "identity", "size"})
            if not isinstance(item["name"], str) or not MEMBER.fullmatch(item["name"]) or item["name"] in (MANIFEST, SIGNATURE):
                raise Refusal("malformed_record", f"manifest lists an invalid member name {str(item['name'])[:80]!r}")
            if not isinstance(item["identity"], str) or not IDENTITY.fullmatch(item["identity"]):
                raise Refusal("malformed_identity", "member identity must be sha256:<64 lowercase hex>")
            integer(item["size"], "member size", minimum=0)
            names.append(item["name"])
        if names != sorted(set(names)):
            raise Refusal("malformed_record", "manifest members must be sorted and unique")
        return manifest
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, Refusal):
            raise Refusal("malformed_manifest", str(exc), reason=exc.code) from exc
        raise Refusal("malformed_manifest", f"manifest.json is not JSON: {exc}") from exc


def _check_ledger(data: bytes, manifest: dict, present: set[str], listed: set[str], flag) -> list[dict]:
    if data and not data.endswith(b"\n"):
        flag("truncated_ledger")
    raw_lines = (data[:-1] if data.endswith(b"\n") else data).split(b"\n") if data else []
    entries = []
    for number, raw in enumerate(raw_lines):
        try:
            entry = json.loads(raw)
            canonical = canonical_json(entry).encode("utf-8")
        except (UnicodeDecodeError, ValueError, RecursionError):
            flag("unparseable_entry", line=number)
            continue
        if not isinstance(entry, dict):
            flag("unparseable_entry", line=number)
            continue
        if canonical != raw:
            flag("noncanonical_entry", line=number)
        if entry.get("entry_id") != entry_identity(entry):
            flag("entry_identity_mismatch", line=number)
        if not (isinstance(entry.get("refs"), list) and isinstance(entry.get("blobs"), list)
                and isinstance(entry.get("body"), dict) and isinstance(entry.get("entry_id"), str)):
            flag("malformed_entry", line=number)
            continue
        schema = entry.get("body_schema")
        if entry.get("schema") != ENTRY_SCHEMA or schema not in BODY_SCHEMAS or BODY_SCHEMAS[schema][0] != entry.get("kind"):
            flag("unknown_body_schema", line=number)
        entries.append(entry)
    every = {entry["entry_id"] for entry in entries}
    earlier, referenced, previous, last_sequence = set(), set(), None, -1
    for position, entry in enumerate(entries):
        entry_id = entry["entry_id"]
        for ref in entry["refs"]:
            if ref not in earlier:
                flag("forward_reference" if ref in every else "ref_outside_bundle", entry=entry_id, ref=str(ref)[:80])
        for identity in entry["blobs"]:
            if not isinstance(identity, str) or IDENTITY.fullmatch(identity) is None:
                flag("malformed_blob_identity", entry=entry_id)
                continue
            referenced.add(identity)
            if "blobs/" + identity[7:] not in present:
                flag("blob_missing", entry=entry_id, blob=identity)
        sequence = entry.get("sequence")
        if manifest["complete"]:
            if sequence != position or entry.get("previous") != previous:
                flag("broken_chain", position=position)
        elif isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= last_sequence:
            flag("entry_order", position=position)
        else:
            last_sequence = sequence
        if entry_id in earlier:
            flag("duplicate_entry", entry=entry_id)
        earlier.add(entry_id)
        previous = entry_id
    for name in sorted(present | listed):
        if "sha256:" + name[6:] not in referenced:
            flag("unreferenced_blob", name=name)
    head = entries[-1]["entry_id"] if entries else None
    if manifest["entries"] != len(entries):
        flag("entry_count_mismatch", declared=manifest["entries"], found=len(entries))
    if manifest["head"] != head:
        flag("head_mismatch")
    if manifest["complete"] and (manifest["source_head"] != head or manifest["source_entries"] != len(entries)):
        flag("incomplete_bundle_claimed_complete")
    for entry_id in manifest["selected"] or []:
        if entry_id not in earlier:
            flag("selected_entry_missing", entry=entry_id)
    return entries


def _signature(data: bytes | None, manifest_bytes: bytes | None, trusted: dict[str, bytes], flag) -> dict:
    if data is None:
        return {"status": "unsigned", "key_id": None, "trusted": False}
    try:
        record = json.loads(data)
        if canonical_json(record).encode("utf-8") != data:
            raise Refusal("malformed_record", "signature.json is not canonical JSON")
        require_keys(record, "signature", {"schema", "key_id", "public_key", "signature"})
        if record["schema"] != SIGNATURE_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {SIGNATURE_SCHEMA}")
        key_id = text(record["key_id"], "key_id", 128)
        public, signature = _hex(record["public_key"], 32, "public_key"), _hex(record["signature"], 64, "signature")
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        flag("malformed_signature", reason=getattr(exc, "code", "unparseable"))
        return {"status": "invalid", "key_id": None, "trusted": False}
    if manifest_bytes is None:
        status = "invalid"
    elif key_id in trusted:
        if trusted[key_id] != public:
            flag("signature_key_mismatch", key_id=key_id)
        status = "valid" if ed25519.verify(trusted[key_id], manifest_bytes, signature) else "invalid"
    else:
        status = "untrusted_key" if ed25519.verify(public, manifest_bytes, signature) else "invalid"
    if status == "invalid":
        flag("signature_invalid", key_id=key_id)
    return {"status": status, "key_id": key_id, "trusted": status == "valid"}


def _inspect(archive: zipfile.ZipFile, path: Path, trusted: dict[str, bytes]) -> tuple[dict, bytes | None, list]:
    problems: list[dict] = []

    def flag(problem: str, **detail: Any) -> None:
        problems.append({"problem": problem, **detail})

    contents: dict[str, bytes] = {}
    digests: dict[str, tuple[str, int]] = {}
    blob_infos, seen, rejected, declared_total = [], set(), set(), 0
    for info in archive.infolist():
        name = info.orig_filename
        if _unsafe(name):
            flag("unsafe_member_name", name=name[:200])
            continue
        if MEMBER.fullmatch(name) is None:
            flag("unexpected_member", name=name[:200])
            continue
        if name in seen:
            flag("duplicate_member", name=name)
            rejected.add(name)
            continue
        seen.add(name)
        if info.flag_bits & 0x1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            flag("unsupported_member_encoding", name=name)
            rejected.add(name)
            continue
        limit = _limit(name)
        if info.file_size > limit:
            flag("oversized_member", name=name, size=info.file_size, limit=limit)
            rejected.add(name)
            continue
        if info.file_size > RATIO_FLOOR and info.file_size > MAX_RATIO * max(info.compress_size, 1):
            flag("compression_ratio_exceeded", name=name, size=info.file_size, compressed=info.compress_size)
            rejected.add(name)
            continue
        declared_total += info.file_size
        if declared_total > MAX_TOTAL_BYTES:
            flag("oversized_bundle", limit=MAX_TOTAL_BYTES)
            rejected.add(name)
            break
        try:
            data = _read_member(archive, info, limit)
        except Refusal as exc:
            flag(exc.code, name=name)
            rejected.add(name)
            continue
        digests[name] = (blob_identity(data), len(data))
        if name.startswith("blobs/"):
            blob_infos.append((name, info))
            if digests[name][0] != "sha256:" + name[6:]:
                flag("blob_digest_mismatch", name=name)
        else:
            contents[name] = data
    manifest_bytes, manifest = contents.get(MANIFEST), None
    if manifest_bytes is None:
        if MANIFEST not in rejected:
            flag("manifest_missing")
    else:
        try:
            manifest = _manifest(manifest_bytes)
        except Refusal as exc:
            flag("malformed_manifest", reason=exc.detail.get("reason", exc.code), message=str(exc)[:200])
    entries: list[dict] = []
    if manifest is not None:
        listed = {item["name"]: item for item in manifest["members"]}
        for name, item in listed.items():
            if name not in digests:
                if name not in rejected:
                    flag("member_missing", name=name)
            elif digests[name] != (item["identity"], item["size"]):
                flag("digest_mismatch", name=name)
        for name in digests:
            if name not in listed and name not in (MANIFEST, SIGNATURE):
                flag("unlisted_member", name=name)
        if LEDGER_MEMBER in contents:
            entries = _check_ledger(contents[LEDGER_MEMBER], manifest,
                                    {name for name in digests if name.startswith("blobs/")},
                                    {name for name in listed if name.startswith("blobs/")}, flag)
    signature = ({"status": "invalid", "key_id": None, "trusted": False} if SIGNATURE in rejected
                 else _signature(contents.get(SIGNATURE), manifest_bytes, trusted, flag))
    report = {
        "schema": INSPECTION_SCHEMA, "ok": not problems, "path": str(path), "bundle_identity": _file_identity(path),
        "bytes": path.stat().st_size, "manifest_identity": None if manifest_bytes is None else blob_identity(manifest_bytes),
        "complete": None if manifest is None else manifest["complete"], "entries": len(entries),
        "head": None if manifest is None else manifest["head"],
        "source_head": None if manifest is None else manifest["source_head"],
        "blobs": sum(name.startswith("blobs/") for name in digests), "members": len(archive.infolist()),
        "signature": signature, "problems": problems[:MAX_PROBLEMS], "problems_total": len(problems),
    }
    return report, contents.get(LEDGER_MEMBER), blob_infos


def inspect_bundle(path: Path, *, trusted_keys: dict[str, bytes] | None = None) -> dict:
    """Verify a bundle offline and read-only; problems are reported, never repaired."""
    trusted = _trusted(trusted_keys)
    with _open_archive(path) as archive:
        report, _, _ = _inspect(archive, Path(path), trusted)
    return report


# ---------------------------------------------------------------------- import
_SIGNATURE_REFUSALS = {"unsigned": "bundle_unsigned", "untrusted_key": "bundle_key_untrusted",
                       "invalid": "bundle_signature_invalid"}


def import_bundle(path: Path, destination: Path, *, trusted_keys: dict[str, bytes] | None = None,
                  require_signature: bool = True) -> Ledger:
    """Materialize a clean, complete (and by default validly signed) bundle as a new ledger."""
    trusted = _trusted(trusted_keys)
    destination = Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise Refusal("destination_not_empty", f"{destination} exists and is not an empty directory")
    with _open_archive(path) as archive:
        report, ledger_bytes, blob_infos = _inspect(archive, Path(path), trusted)
        if not report["ok"]:
            raise Refusal("bundle_rejected", "Bundle inspection found problems", problems=report["problems"][:20])
        if not report["complete"]:
            raise Refusal("incomplete_bundle", "A subset bundle cannot prove chain continuity and is not importable",
                          entries=report["entries"])
        status = report["signature"]["status"]
        if require_signature and status != "valid":
            raise Refusal(_SIGNATURE_REFUSALS[status], f"Bundle signature status is {status!r}",
                          key_id=report["signature"]["key_id"])
        created = not destination.exists()
        try:
            ledger = Ledger.create(destination)
            for name, info in blob_infos:
                data = _read_member(archive, info, MAX_BLOB_BYTES)
                if blob_identity(data) != "sha256:" + name[6:]:
                    raise Refusal("blob_digest_mismatch", f"{name} changed after inspection")
                ledger.put_blob(data)
            with (destination / LEDGER_FILE).open("wb") as stream:
                stream.write(ledger_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            return Ledger.open(destination)
        except BaseException:
            if created:
                shutil.rmtree(destination, ignore_errors=True)
            else:
                (destination / LEDGER_FILE).unlink(missing_ok=True)
                shutil.rmtree(destination / BLOB_DIR, ignore_errors=True)
            raise


# ---------------------------------------------------------------------- signed updates
def _payload(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise Refusal("malformed_record", "An update payload must be bytes")
    if len(value) > MAX_UPDATE_PAYLOAD:
        raise Refusal("oversized_input", f"Update payload exceeds {MAX_UPDATE_PAYLOAD} bytes")
    return bytes(value)


def _update_message(metadata: Any, payload_identity: str, key_id: str) -> bytes:
    signed = {"metadata": metadata, "payload_identity": payload_identity, "key_id": key_id}
    return bounded_json(signed, "signed update", MAX_UPDATE_METADATA + 4096).encode("utf-8")


def sign_update(payload: bytes, secret: bytes, key_id: str, metadata: dict) -> dict:
    """Sign a schema library, provider pin set or similar payload for offline distribution."""
    payload, secret = _payload(payload), _secret(secret)
    key_id = text(key_id, "key_id", 128)
    metadata = json.loads(bounded_json(mapping(metadata, "metadata"), "metadata", MAX_UPDATE_METADATA))
    identity = blob_identity(payload)
    return {"schema": UPDATE_SCHEMA, "metadata": metadata, "payload_identity": identity, "key_id": key_id,
            "public_key": ed25519.public_key(secret).hex(),
            "signature": ed25519.sign(secret, _update_message(metadata, identity, key_id)).hex()}


def verify_update(envelope: dict, payload: bytes, trusted_keys: dict[str, bytes]) -> dict:
    """Return the update metadata if a trusted key signed exactly this payload; refuse otherwise."""
    trusted = _trusted(trusted_keys)
    payload = _payload(payload)
    try:
        require_keys(envelope, "signed update", {"schema", "metadata", "payload_identity", "key_id", "public_key",
                                                 "signature"})
        if envelope["schema"] != UPDATE_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {UPDATE_SCHEMA}")
        mapping(envelope["metadata"], "metadata")
        key_id = text(envelope["key_id"], "key_id", 128)
        if not isinstance(envelope["payload_identity"], str) or not IDENTITY.fullmatch(envelope["payload_identity"]):
            raise Refusal("malformed_identity", "payload_identity must be sha256:<64 lowercase hex>")
        public = _hex(envelope["public_key"], 32, "public_key")
        signature = _hex(envelope["signature"], 64, "signature")
        message = _update_message(envelope["metadata"], envelope["payload_identity"], key_id)
    except Refusal as exc:
        raise Refusal("update_malformed", f"Malformed signed update: {exc}", reason=exc.code) from exc
    if key_id not in trusted:
        raise Refusal("update_key_untrusted", f"Key {key_id!r} is not trusted", key_id=key_id)
    if trusted[key_id] != public:
        raise Refusal("update_key_untrusted", f"The envelope's public key is not the trusted key {key_id!r}",
                      key_id=key_id)
    if not ed25519.verify(trusted[key_id], message, signature):
        raise Refusal("update_signature_invalid", "The update signature does not verify", key_id=key_id)
    if blob_identity(payload) != envelope["payload_identity"]:
        raise Refusal("update_payload_mismatch", "The payload is not the payload that was signed",
                      signed=envelope["payload_identity"], found=blob_identity(payload))
    return json.loads(canonical_json(envelope["metadata"]))


__all__ = [
    "MANIFEST_SCHEMA", "SIGNATURE_SCHEMA", "UPDATE_SCHEMA", "SIGNATURE_STATUSES", "key_fingerprint", "export_bundle",
    "inspect_bundle", "import_bundle", "sign_update", "verify_update",
]
