"""Native evidence acquisition, replay, malformed input and lineage boundaries."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from ciw import acquired_dataset as a
from ciw.adapters.ppda_acquisition import AcquisitionAdapter
from ciw.adapters.protocol import AdapterRefusal
from ciw.core.canonical import bundle_digest

ROOT = Path(__file__).resolve().parents[1]


def source():
    return json.loads((ROOT / "examples/acquired-dataset/source.json").read_bytes())


@pytest.mark.parametrize("fault", ["path", "boolean-sequence", "reorder", "revision", "nan", "duplicate-json-key", "request-time", "policy", "snapshot-budget"])
def test_refuse_ambiguous_or_unbounded_sources(fault):
    value = source()
    if fault == "path": value["plan_id"] = "../outside"
    if fault == "policy": value["configuration"]["event_time_order"] = "inferred"
    if fault == "request-time": value["snapshots"][1]["requested_at"] = value["snapshots"][0]["requested_at"]
    if fault == "snapshot-budget": value["snapshots"] *= 3
    if fault in {"boolean-sequence", "reorder", "revision"}:
        index = 1 if fault == "revision" else 0
        rows = json.loads(base64.b64decode(value["snapshots"][index]["bytes_b64"]))
        if fault == "boolean-sequence": rows[0]["sequence"] = True
        if fault == "reorder": rows.reverse()
        if fault == "revision": rows[0]["raw_value"] = 999
        value["snapshots"][index]["bytes_b64"] = base64.b64encode(a.canonical(rows)).decode()
    if fault in {"nan", "duplicate-json-key"}:
        raw = b'[{"sequence":1,"value":NaN}]' if fault == "nan" else b'[{"sequence":1,"sequence":2}]'
        value["snapshots"][0]["bytes_b64"] = base64.b64encode(raw).decode()
    with pytest.raises((ValueError, AdapterRefusal)):
        a._source(a.canonical(value))


@pytest.fixture(scope="module")
def bindings():
    root = os.environ.get("CIW_PPDA_REPO")
    if not root:
        pytest.skip("set CIW_PPDA_REPO with the exact initialized PPDA checkout")
    return {"ppda": Path(root)}


@pytest.fixture(scope="module")
def retained(bindings):
    raw = b"\n" + a.canonical(source()) + b"\n "
    original = a.create_session(raw, bindings)
    replay = a.replay_session(original, bindings)["session"]
    output = os.environ.get("CIW_ACQUISITION_FIXTURE_DIR")
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        for name, value in (("original", original), ("replay", replay)):
            (path / (name + ".json")).write_bytes(a.canonical(value))
    return raw, original, replay


def test_native_incremental_acquisition_restart_and_exact_snapshot_retention(retained):
    raw, original, replay = retained
    assert a._validate(original) == raw == a._validate(replay)
    data = original["steps"][0]["result"]["data"]
    assert [len(run["result"]["artifacts"]) for run in data["runs"]] == [2, 1, 0]
    assert [run["checkpoint_after"]["position"] for run in data["runs"]] == ["000000000002", "000000000003", "000000000003"]
    assert data["runs"][1]["checkpoint_before"] == data["runs"][0]["checkpoint_after"]
    assert data["restored_pool_fingerprint"] == data["runs"][-1]["pool_fingerprint"]
    assert data["runs"][-2]["pool_fingerprint"] == data["runs"][-1]["pool_fingerprint"]
    assert len(data["evidence"]["observations"]) == 3
    rows = sorted((item["content"] for item in data["evidence"]["observations"]), key=lambda v: v["sequence"])
    assert [item["device_time"] for item in rows] == [10.0, 9.0, 12.0]
    assert rows[1]["raw_value"] is None and rows[1]["value_absence"] == "not_reported"
    assert all(row["crosscov_policy"] == "unknown" and row["clock_mapping_ref"] is None for row in rows)
    assert original["steps"][0]["numerical_result_id"] == replay["steps"][0]["numerical_result_id"]
    assert original["steps"][0]["result_id"] != replay["steps"][0]["result_id"]
    assert original["verification"]["independent"] is False
    assert original["runtimes"]["ppda"]["vendor"]["revision"] == a.VENDOR_REVISION


@pytest.mark.parametrize("fault", ["unknown-to-zero", "missing-to-zero", "timestamp", "checkpoint", "lineage", "source", "vendor", "freshness"])
def test_resealed_adversarial_changes_refused(retained, fault):
    bundle = deepcopy(retained[1])
    data = bundle["steps"][0]["result"]["data"]
    if fault == "unknown-to-zero": data["evidence"]["observations"][0]["content"]["crosscov_policy"] = "declared_zero"
    if fault == "missing-to-zero":
        next(o for o in data["evidence"]["observations"] if o["content"]["raw_value"] is None)["content"]["raw_value"] = 0.0
    if fault == "timestamp": data["evidence"]["records"][0]["raw_content"] += " "
    if fault == "checkpoint": data["runs"][0]["checkpoint_after"]["position"] = "000000000003"
    if fault == "lineage": data["evidence"]["observations"][0]["record_ids"] = ["0" * 64]
    if fault == "source": data["source_definition"]["source_id"] = "another-source"
    if fault == "vendor": bundle["runtimes"]["ppda"]["vendor"]["revision"] = "0" * 40
    if fault == "freshness": bundle["verification"]["reproduction"] = deepcopy(bundle["steps"][0])
    bundle["bundle_digest"] = bundle_digest(bundle)
    with pytest.raises(ValueError): a._validate(bundle)


def test_retained_validation_never_executes_provider(retained, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline inspection must not execute provider code")
    monkeypatch.setattr(AcquisitionAdapter, "__init__", forbidden)
    assert a._validate(retained[1]) == retained[0]


def test_native_unicode_and_nested_missingness_survive(bindings):
    value = source()
    rows = [{"sequence":0,"channel":"température","value":None,"uncertainty":{"status":"unknown","matrix":None},"value_absence":"not_reported"}]
    value["snapshots"] = [{"requested_at":"2026-09-22T00:00:00Z","bytes_b64":base64.b64encode(a.canonical(rows)).decode()}]
    bundle = a.create_session(a.canonical(value), bindings)
    assert bundle["steps"][0]["result"]["data"]["evidence"]["observations"][0]["content"] == rows[0]


def test_changed_vendor_source_is_refused_before_import(bindings, tmp_path):
    root = tmp_path / "provider"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(bindings["ppda"]), str(root)], check=True)
    vendor = root / "vendor/scout-retrieval-agent"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(bindings["ppda"] / "vendor/scout-retrieval-agent"), str(vendor)], check=True)
    path = vendor / "evidence/types.py"
    path.write_bytes(path.read_bytes() + b"\n# unreviewed drift\n")
    with pytest.raises(AdapterRefusal, match="Tracked file bytes differ"):
        a._adapters({"ppda": root})
