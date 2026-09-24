"""Each record check of the retained workbench fails a test when it is removed.

The shared reference lifecycle repeats several of these checks inside its
workflows, so a mutation probe that dropped them from the workbench found no
failing test. The cases here exercise the workbench checks themselves: with
real retained bundles where a provider-free workflow exists, and with a stub
workflow where the workbench check is the only guard for provider-backed kinds.
"""
import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import workbench as wb
from ciw.energy_workflow import EnergyAccuracyWorkflow
from ciw.workbench import Workbench, _validate_receipts, _validate_record

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "retained" / "workbench.json"
ENERGY = ROOT / "examples" / "energy-accuracy" / "baseline.json"
OTHER_ENERGY = ROOT / "examples" / "energy-accuracy" / "missing.json"
FOREIGN = "sha256:" + "f" * 64


def catalog():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def record_of(value, kind):
    return next(item for item in value["bundles"] if item["kind"] == kind)


def source_of(value, kind):
    return next(item for item in value["sources"] if item["kind"] == kind)


def test_the_fixture_catalog_restores_before_any_tampering():
    Workbench.restore(catalog())


def test_a_bundle_must_name_a_retained_source_of_its_own_kind():
    tampered = catalog()
    record_of(tampered, "energy-accuracy")["source_id"] = source_of(tampered, "thermal-observer")["source_id"]
    with pytest.raises(ValueError, match="retained source of the same kind"):
        Workbench.restore(tampered)


def test_a_bundle_must_carry_its_native_identity():
    tampered = catalog()
    record_of(tampered, "energy-accuracy")["bundle_id"] = FOREIGN
    with pytest.raises(ValueError, match="exact retained source bytes"):
        Workbench.restore(tampered)


def test_a_bundle_must_match_the_bytes_of_the_source_it_names():
    workbench = Workbench.restore(catalog())
    other = workbench.add_source({"kind": "energy-accuracy", "label": "other log",
                                  "bytes_b64": base64.b64encode(OTHER_ENERGY.read_bytes()).decode()})
    tampered = workbench.serialize()
    record_of(tampered, "energy-accuracy")["source_id"] = other["source_id"]
    with pytest.raises(ValueError, match="exact retained source bytes"):
        Workbench.restore(tampered)


def test_a_catalog_may_not_list_the_same_bundle_twice():
    tampered = catalog()
    tampered["bundles"].append(deepcopy(tampered["bundles"][0]))
    tampered["revision"] += 1
    with pytest.raises(ValueError, match="Duplicate retained bundle identity"):
        Workbench.restore(tampered)


class _StubWorkflow:
    """Stands in for a provider-backed workflow whose own validation already passed."""

    def __init__(self, raw):
        self.raw = raw

    def _validate(self, native):
        return self.raw


def _stub_record(monkeypatch, kind, upstream, native):
    raw = b'{"stub": true}'
    monkeypatch.setattr(wb, "_workflow", lambda requested: _StubWorkflow(raw))
    sources = {"source:stub": {"kind": kind, "bytes_b64": base64.b64encode(raw).decode()}}
    record = {"kind": kind, "source_id": "source:stub", "upstream_bundle_id": upstream,
              "bundle_id": native["bundle_digest"], "native": native}
    return record, sources


def test_a_retained_record_must_preserve_its_native_verification_artifact(monkeypatch):
    record, sources = _stub_record(monkeypatch, "machine-manifest", None, {"bundle_digest": FOREIGN})
    with pytest.raises(ValueError, match="native verification artifact"):
        _validate_record(record, sources)


def test_an_upstream_kind_needs_an_explicitly_selected_upstream_bundle(monkeypatch):
    assert "identified-design" in wb.UPSTREAM_KINDS
    record, sources = _stub_record(monkeypatch, "identified-design", None,
                                   {"bundle_digest": FOREIGN, "verification": {}})
    with pytest.raises(ValueError, match="explicitly selected upstream bundle"):
        _validate_record(record, sources)


def test_a_kind_without_upstream_may_not_declare_one(monkeypatch):
    assert "machine-manifest" not in wb.UPSTREAM_KINDS
    record, sources = _stub_record(monkeypatch, "machine-manifest", "sha256:" + "1" * 64,
                                   {"bundle_digest": FOREIGN, "verification": {}})
    with pytest.raises(ValueError, match="no implicit upstream bundle"):
        _validate_record(record, sources)


@pytest.fixture(scope="module")
def replayed():
    workflow = EnergyAccuracyWorkflow()
    bundle = workflow.create_session(ENERGY.read_bytes(), {})
    return workflow.replay_session(bundle, {})["session"]


def _reseal_receipt(receipt):
    receipt["replay_id"] = wb._digest({key: value for key, value in receipt.items() if key != "replay_id"})
    return receipt


def test_a_real_replay_receipt_passes_the_workbench_receipt_check(replayed):
    assert _validate_receipts(deepcopy(replayed), "energy-accuracy") is None


def test_a_native_replay_retains_exactly_one_receipt(replayed):
    doubled = deepcopy(replayed)
    doubled["replay_receipts"].append(deepcopy(doubled["replay_receipts"][0]))
    with pytest.raises(ValueError, match="retains one receipt"):
        _validate_receipts(doubled, "energy-accuracy")


def test_a_receipt_must_name_the_fresh_bundle_it_sits_in(replayed):
    tampered = deepcopy(replayed)
    receipt = tampered["replay_receipts"][0]
    receipt["replayed_bundle_digest"] = FOREIGN
    _reseal_receipt(receipt)
    with pytest.raises(ValueError, match="does not bind the retained fresh bundle"):
        _validate_receipts(tampered, "energy-accuracy")


def test_a_receipt_identity_must_match_its_content(replayed):
    tampered = deepcopy(replayed)
    tampered["replay_receipts"][0]["replay_id"] = FOREIGN
    with pytest.raises(ValueError, match="does not bind the retained fresh bundle"):
        _validate_receipts(tampered, "energy-accuracy")
