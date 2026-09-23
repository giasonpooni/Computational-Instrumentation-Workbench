"""Offline evidence bundles are deterministic, bounded, signed and never trusted blindly."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings
import zipfile

import pytest

from ciw.science import bundle as bd
from ciw.science import ed25519
from ciw.science import hardware as hw
from ciw.science._common import Refusal, canonical_json
from ciw.science.ledger import Ledger, source_evidence

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = hw.validate_layouts(json.loads((ROOT / "examples/science/hardware/layouts.json").read_text(encoding="utf-8")))
ALICE, MALLORY = bytes(range(32)), bytes(range(32, 64))
TRUSTED = {"alice": ed25519.public_key(ALICE)}


def fixed_now():
    return datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def source(tmp_path):
    ledger = Ledger.create(tmp_path / "source", now=fixed_now)
    a = source_evidence(ledger, "alpha", b"alpha bytes", "application/octet-stream")
    b = ledger.append("ciw.science.claim.v1", {"statement": "alpha is retained", "claim_class": "measured",
                                               "status": "open"}, refs=[a["entry_id"]])
    source_evidence(ledger, "gamma", b"gamma bytes", "text/plain")
    report = ledger.put_blob(b"# report\n")
    d = ledger.append("ciw.science.report.v1", {"blob": report, "format": "markdown", "ledger_head": b["entry_id"]},
                      refs=[b["entry_id"]], blobs=[report])
    ledger.ids = [a["entry_id"], b["entry_id"], None, d["entry_id"]]
    return ledger


def members(path):
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def rewrite(source_path, target, replace=None, add=()):
    """Copy a bundle member by member, replacing (bytes) or removing (None) members, then appending extras."""
    replace = replace or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(source_path) as old, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as new:
            for info in old.infolist():
                data = replace.get(info.filename, old.read(info))
                if data is not None:
                    new.writestr(info.filename, data)
            for name, data in add:
                new.writestr(name, data)
    return target


def add_raw_name(source_path, target, name, data=b"payload"):
    """Append a member whose stored name bytes are exactly ``name`` (zipfile rewrites backslashes on Windows)."""
    placeholder = name.replace("\\", "|")
    rewrite(source_path, target, add=[(placeholder, data)])
    raw = target.read_bytes()
    assert raw.count(placeholder.encode()) == 2
    target.write_bytes(raw.replace(placeholder.encode(), name.encode()))
    return target


def problems(report):
    return {item["problem"] for item in report["problems"]}


# ---------------------------------------------------------------------- round trip and determinism
def test_signed_round_trip_is_byte_identical(source, tmp_path):
    result = bd.export_bundle(source, tmp_path / "out.ciwb", signing_key=ALICE, key_id="alice")
    assert result["complete"] and result["entries"] == 4 and result["blobs"] == 3 and result["signed"]
    report = bd.inspect_bundle(tmp_path / "out.ciwb", trusted_keys=TRUSTED)
    assert report["ok"], report["problems"]
    assert report["signature"] == {"status": "valid", "key_id": "alice", "trusted": True}
    assert report["complete"] is True and report["head"] == source.head() and report["blobs"] == 3
    names = list(members(tmp_path / "out.ciwb"))
    assert names == sorted(names) and names[-3:] == ["ledger.jsonl", "manifest.json", "signature.json"]
    imported = bd.import_bundle(tmp_path / "out.ciwb", tmp_path / "copy", trusted_keys=TRUSTED)
    assert [entry["entry_id"] for entry in imported.entries()] == [entry["entry_id"] for entry in source.entries()]
    assert (tmp_path / "copy/ledger.jsonl").read_bytes() == (source.root / "ledger.jsonl").read_bytes()
    assert imported.get_blob(source.get(source.ids[0])["body"]["blob"]) == b"alpha bytes"
    assert imported.verify().ok


def test_export_is_deterministic_and_never_overwrites(source, tmp_path):
    first = bd.export_bundle(source, tmp_path / "a.ciwb", signing_key=ALICE, key_id="alice")
    second = bd.export_bundle(source, tmp_path / "b.ciwb", signing_key=ALICE, key_id="alice")
    assert (tmp_path / "a.ciwb").read_bytes() == (tmp_path / "b.ciwb").read_bytes()
    assert first["bundle_identity"] == second["bundle_identity"]
    with zipfile.ZipFile(tmp_path / "a.ciwb") as archive:
        assert {(info.date_time, info.external_attr, info.compress_type, info.create_system)
                for info in archive.infolist()} == {(bd.DATE_TIME, bd.EXTERNAL_ATTR, zipfile.ZIP_DEFLATED, 3)}
    before = (tmp_path / "a.ciwb").read_bytes()
    with pytest.raises(Refusal) as caught:
        bd.export_bundle(source, tmp_path / "a.ciwb")
    assert caught.value.code == "destination_exists" and (tmp_path / "a.ciwb").read_bytes() == before
    manifest = json.loads(members(tmp_path / "a.ciwb")["manifest.json"])
    assert manifest["schema"] == bd.MANIFEST_SCHEMA and manifest["source_head"] == source.head()
    assert [item["name"] for item in manifest["members"]] == sorted(item["name"] for item in manifest["members"])


def test_default_key_id_is_a_fingerprint(source, tmp_path):
    result = bd.export_bundle(source, tmp_path / "f.ciwb", signing_key=ALICE)
    assert result["key_id"] == bd.key_fingerprint(TRUSTED["alice"])
    report = bd.inspect_bundle(tmp_path / "f.ciwb", trusted_keys={result["key_id"]: TRUSTED["alice"]})
    assert report["signature"]["status"] == "valid"


# ---------------------------------------------------------------------- tampering and signatures
def test_tampered_members_are_detected(source, tmp_path):
    bd.export_bundle(source, tmp_path / "out.ciwb", signing_key=ALICE, key_id="alice")
    content = members(tmp_path / "out.ciwb")
    blob = next(name for name in content if name.startswith("blobs/"))
    report = bd.inspect_bundle(rewrite(tmp_path / "out.ciwb", tmp_path / "blob.ciwb", {blob: b"forged"}),
                               trusted_keys=TRUSTED)
    assert not report["ok"] and {"blob_digest_mismatch", "digest_mismatch"} <= problems(report)
    assert report["signature"]["status"] == "valid"
    lines = content["ledger.jsonl"].split(b"\n")
    entry = json.loads(lines[1])
    entry["body"]["statement"] = "alpha was never retained"
    lines[1] = canonical_json(entry).encode()
    report = bd.inspect_bundle(rewrite(tmp_path / "out.ciwb", tmp_path / "line.ciwb", {"ledger.jsonl": b"\n".join(lines)}))
    assert {"digest_mismatch", "entry_identity_mismatch"} <= problems(report)
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "line.ciwb", tmp_path / "copy", trusted_keys=TRUSTED)
    assert caught.value.code == "bundle_rejected" and not (tmp_path / "copy").exists()


def test_forged_signatures_are_invalid(source, tmp_path):
    bd.export_bundle(source, tmp_path / "forged.ciwb", signing_key=MALLORY, key_id="alice")
    report = bd.inspect_bundle(tmp_path / "forged.ciwb", trusted_keys=TRUSTED)
    assert report["signature"]["status"] == "invalid" and "signature_invalid" in problems(report)
    assert "signature_key_mismatch" in problems(report)
    bd.export_bundle(source, tmp_path / "good.ciwb", signing_key=ALICE, key_id="alice")
    manifest = json.loads(members(tmp_path / "good.ciwb")["manifest.json"])
    manifest["source_entries"] = 99
    edited = rewrite(tmp_path / "good.ciwb", tmp_path / "edited.ciwb",
                     {"manifest.json": canonical_json(manifest).encode()})
    report = bd.inspect_bundle(edited, trusted_keys=TRUSTED)
    assert report["signature"]["status"] == "invalid" and not report["ok"]
    for path in (tmp_path / "forged.ciwb", edited):
        with pytest.raises(Refusal) as caught:
            bd.import_bundle(path, tmp_path / "copy", trusted_keys=TRUSTED, require_signature=False)
        assert caught.value.code == "bundle_rejected"


def test_untrusted_and_unsigned_bundles(source, tmp_path):
    bd.export_bundle(source, tmp_path / "mallory.ciwb", signing_key=MALLORY, key_id="mallory")
    report = bd.inspect_bundle(tmp_path / "mallory.ciwb", trusted_keys=TRUSTED)
    assert report["ok"] and report["signature"]["status"] == "untrusted_key"
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "mallory.ciwb", tmp_path / "one", trusted_keys=TRUSTED)
    assert caught.value.code == "bundle_key_untrusted"
    bd.export_bundle(source, tmp_path / "plain.ciwb")
    assert bd.inspect_bundle(tmp_path / "plain.ciwb")["signature"]["status"] == "unsigned"
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "plain.ciwb", tmp_path / "two", trusted_keys=TRUSTED)
    assert caught.value.code == "bundle_unsigned" and not (tmp_path / "two").exists()
    assert len(bd.import_bundle(tmp_path / "plain.ciwb", tmp_path / "two", require_signature=False)) == 4
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "plain.ciwb", tmp_path / "two", require_signature=False)
    assert caught.value.code == "destination_not_empty"


# ---------------------------------------------------------------------- hostile archives
@pytest.mark.parametrize("name", ["../x", "/abs", "blobs/../../x", "C:\\x", "C:/x", "blobs\\" + "a" * 64,
                                  "blobs/./" + "a" * 64])
def test_path_traversal_member_names_are_refused(source, tmp_path, name):
    bd.export_bundle(source, tmp_path / "out.ciwb", signing_key=ALICE, key_id="alice")
    hostile = add_raw_name(tmp_path / "out.ciwb", tmp_path / "hostile.ciwb", name)
    report = bd.inspect_bundle(hostile, trusted_keys=TRUSTED)
    assert not report["ok"] and "unsafe_member_name" in problems(report)
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(hostile, tmp_path / "copy", trusted_keys=TRUSTED)
    assert caught.value.code == "bundle_rejected"
    assert not (tmp_path / "x").exists() and not (tmp_path / "copy").exists()


def test_unexpected_duplicate_missing_and_unlisted_members(source, tmp_path, monkeypatch):
    bd.export_bundle(source, tmp_path / "out.ciwb")
    content = members(tmp_path / "out.ciwb")
    blob = next(name for name in content if name.startswith("blobs/"))
    extra = b"unlisted evidence"
    cases = {
        "unexpected_member": {"add": [("notes.txt", b"x")]},
        "duplicate_member": {"add": [("ledger.jsonl", content["ledger.jsonl"])]},
        "member_missing": {"replace": {blob: None}},
        "unlisted_member": {"add": [("blobs/" + hashlib.sha256(extra).hexdigest(), extra)]},
        "manifest_missing": {"replace": {"manifest.json": None}},
    }
    for index, (expected, options) in enumerate(cases.items()):
        report = bd.inspect_bundle(rewrite(tmp_path / "out.ciwb", tmp_path / f"case{index}.ciwb", **options))
        assert not report["ok"] and expected in problems(report), (expected, report["problems"])
    assert "blob_missing" in problems(bd.inspect_bundle(tmp_path / "case2.ciwb"))
    assert "unreferenced_blob" in problems(bd.inspect_bundle(tmp_path / "case3.ciwb"))
    raw = (tmp_path / "out.ciwb").read_bytes()
    hidden = bytearray(raw)
    hidden[raw.rindex(b"PK\x01\x02", 0, raw.rindex(b"PK\x01\x02")) + 32] = 0xFF  # a comment swallows the last entry
    (tmp_path / "hidden.ciwb").write_bytes(bytes(hidden))
    with pytest.raises(Refusal) as caught:
        bd.inspect_bundle(tmp_path / "hidden.ciwb")
    assert caught.value.code == "bundle_unreadable"
    monkeypatch.setattr(bd, "MAX_MEMBERS", 3)
    with pytest.raises(Refusal) as caught:
        bd.inspect_bundle(tmp_path / "out.ciwb")
    assert caught.value.code == "oversized_input"
    (tmp_path / "junk.ciwb").write_bytes(b"not a zip")
    with pytest.raises(Refusal) as caught:
        bd.inspect_bundle(tmp_path / "junk.ciwb")
    assert caught.value.code == "bundle_unreadable"


def test_zip_bombs_are_refused_before_reading(source, tmp_path, monkeypatch):
    bd.export_bundle(source, tmp_path / "out.ciwb")
    read = []
    original = bd._read_member
    monkeypatch.setattr(bd, "_read_member", lambda archive, info, limit: read.append(info.filename) or original(
        archive, info, limit))
    big, dense = b"\x00" * (1 << 20), b"\x00" * 60_000
    big_name, dense_name = ("blobs/" + hashlib.sha256(data).hexdigest() for data in (big, dense))
    monkeypatch.setattr(bd, "MAX_BLOB_BYTES", 1 << 16)
    report = bd.inspect_bundle(rewrite(tmp_path / "out.ciwb", tmp_path / "big.ciwb", add=[(big_name, big)]))
    assert "oversized_member" in problems(report) and big_name not in read
    monkeypatch.setattr(bd, "RATIO_FLOOR", 1024)
    monkeypatch.setattr(bd, "MAX_RATIO", 50)
    report = bd.inspect_bundle(rewrite(tmp_path / "out.ciwb", tmp_path / "dense.ciwb", add=[(dense_name, dense)]))
    assert "compression_ratio_exceeded" in problems(report) and dense_name not in read
    monkeypatch.setattr(bd, "MAX_RATIO", 10 ** 6)
    monkeypatch.setattr(bd, "MAX_TOTAL_BYTES", 30_000)
    report = bd.inspect_bundle(tmp_path / "dense.ciwb")
    assert "oversized_bundle" in problems(report) and dense_name not in read
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "dense.ciwb", tmp_path / "copy", require_signature=False)
    assert caught.value.code == "bundle_rejected"


def test_export_never_emits_what_its_inspector_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "RATIO_FLOOR", 1024)
    monkeypatch.setattr(bd, "MAX_RATIO", 50)
    ledger = Ledger.create(tmp_path / "sparse", now=fixed_now)
    source_evidence(ledger, "zeros", b"\x00" * 60_000, "application/octet-stream")
    bd.export_bundle(ledger, tmp_path / "sparse.ciwb")
    with zipfile.ZipFile(tmp_path / "sparse.ciwb") as archive:
        stored = {info.filename for info in archive.infolist() if info.compress_type == zipfile.ZIP_STORED}
    assert stored == {"blobs/" + hashlib.sha256(b"\x00" * 60_000).hexdigest()}
    assert bd.inspect_bundle(tmp_path / "sparse.ciwb")["ok"]
    monkeypatch.setattr(bd, "MAX_TOTAL_BYTES", 1000)
    with pytest.raises(Refusal) as caught:
        bd.export_bundle(ledger, tmp_path / "big.ciwb")
    assert caught.value.code == "oversized_input" and not (tmp_path / "big.ciwb").exists()


# ---------------------------------------------------------------------- subsets and hardware captures
def test_subset_export_is_a_closure_that_inspects_but_does_not_import(source, tmp_path):
    a, b, _, d = source.ids
    result = bd.export_bundle(source, tmp_path / "subset.ciwb", entry_ids=[d], signing_key=ALICE, key_id="alice")
    assert result["complete"] is False and result["entries"] == 3 and result["blobs"] == 2
    manifest = json.loads(members(tmp_path / "subset.ciwb")["manifest.json"])
    assert manifest["selected"] == [d] and manifest["complete"] is False
    assert manifest["source_entries"] == 4 and manifest["head"] == d
    lines = members(tmp_path / "subset.ciwb")["ledger.jsonl"].splitlines(keepends=True)
    source_lines = (source.root / "ledger.jsonl").read_bytes().splitlines(keepends=True)
    assert lines == [source_lines[0], source_lines[1], source_lines[3]]
    report = bd.inspect_bundle(tmp_path / "subset.ciwb", trusted_keys=TRUSTED)
    assert report["ok"], report["problems"]
    assert report["complete"] is False and report["signature"]["status"] == "valid"
    with pytest.raises(Refusal) as caught:
        bd.import_bundle(tmp_path / "subset.ciwb", tmp_path / "copy", trusted_keys=TRUSTED)
    assert caught.value.code == "incomplete_bundle" and not (tmp_path / "copy").exists()
    only_b = bd.export_bundle(source, tmp_path / "b.ciwb", entry_ids=[b])
    assert only_b["entries"] == 2 and only_b["blobs"] == 1
    with pytest.raises(Refusal) as caught:
        bd.export_bundle(source, tmp_path / "bad.ciwb", entry_ids=["sha256:" + "0" * 64])
    assert caught.value.code == "unknown_entry"


def test_retained_capture_survives_offline_transfer_and_replays(tmp_path):
    ledger = Ledger.create(tmp_path / "field", now=fixed_now)
    data = hw.synthetic_stream(LAYOUTS, seed=11, count=12)
    capture = hw.parse_stream(data, LAYOUTS, tick_rate_hz=1e8, device_clock="fpga_counter")
    entry = hw.retain_capture(ledger, data, capture)
    bd.export_bundle(ledger, tmp_path / "field.ciwb", signing_key=ALICE, key_id="alice")
    imported = bd.import_bundle(tmp_path / "field.ciwb", tmp_path / "lab", trusted_keys=TRUSTED)
    result = hw.replay_retained(imported, entry["entry_id"], LAYOUTS)
    assert result["status"] == "reproduced"
    assert imported.get(entry["entry_id"])["body"]["summary"]["acquisition"] == "synthetic"


# ---------------------------------------------------------------------- signed updates
def test_signed_updates():
    payload = json.dumps({"schemas": ["ciw.telemetry-layouts.v1"]}).encode()
    metadata = {"kind": "schema_library", "version": "2026.09"}
    envelope = bd.sign_update(payload, ALICE, "alice", metadata)
    assert envelope["schema"] == bd.UPDATE_SCHEMA and bd.verify_update(envelope, payload, TRUSTED) == metadata
    failures = [
        (envelope, payload + b" ", TRUSTED, "update_payload_mismatch"),
        (envelope, payload, {"bob": ed25519.public_key(MALLORY)}, "update_key_untrusted"),
        (bd.sign_update(payload, MALLORY, "alice", metadata), payload, TRUSTED, "update_key_untrusted"),
        ({**envelope, "signature": ("0" if envelope["signature"][0] != "0" else "1") + envelope["signature"][1:]},
         payload, TRUSTED, "update_signature_invalid"),
        ({**envelope, "metadata": {**metadata, "version": "2099.01"}}, payload, TRUSTED, "update_signature_invalid"),
        ({key: value for key, value in envelope.items() if key != "signature"}, payload, TRUSTED, "update_malformed"),
        ({**envelope, "public_key": "zz"}, payload, TRUSTED, "update_malformed"),
    ]
    for candidate, data, trusted, code in failures:
        with pytest.raises(Refusal) as caught:
            bd.verify_update(candidate, data, trusted)
        assert caught.value.code == code
    forged = {**envelope, "public_key": ed25519.public_key(MALLORY).hex()}
    with pytest.raises(Refusal) as caught:
        bd.verify_update(forged, payload, TRUSTED)
    assert caught.value.code == "update_key_untrusted"
