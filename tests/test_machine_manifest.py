"""Deterministic machine-manifest compilation remains read-only and replayable."""

from copy import deepcopy
from hashlib import sha256

import pytest

from ciw import machine_manifest as manifest


def _source(source_id, kind, content, machine_id):
    source_digest = "sha256:" + sha256(content.encode()).hexdigest()
    return {
        "source_id": source_id,
        "machine_id": machine_id,
        "kind": kind,
        "locator": "fixture://" + source_id,
        "content_utf8": content,
        "source_digest": source_digest,
    }


def _fixture():
    machine_id = "machine:fixture"
    specs = [
        ("source:counts", "documentation", "decoded-counts"),
        ("source:gear", "documentation", "gear-ratio"),
        ("source:lead", "documentation", "lead"),
        ("source:sign", "observation", "orientation"),
        ("source:home", "observation", "homing"),
        ("source:frame", "documentation", "frame"),
        ("source:time", "observation", "clock"),
        ("source:firmware", "firmware_configuration", "firmware-config"),
        ("source:uncertainty", "validation", "joint-prior"),
        ("source:offset", "observation", "reference-offset"),
    ]
    sources = [_source(*item, machine_id) for item in specs]
    source_map = {item[0]: source for item, source in zip(specs, sources)}

    claims = {
        "C": {"value": 1000, "unit": "count/rev_motor", "status": "validated", "evidence_refs": ["evidence:C"]},
        "g": {"value": 2.0, "unit": "rev_motor/rev_output", "status": "validated", "evidence_refs": ["evidence:g"]},
        "L": {"value": 0.01, "unit": "m/rev_output", "status": "validated", "evidence_refs": ["evidence:L"]},
        "s": {"value": 1, "unit": "1", "status": "validated", "evidence_refs": ["evidence:s"]},
        "N0": {"value": 100, "unit": "count", "status": "validated", "evidence_refs": ["evidence:N0"]},
        "x0": {"value": 1.0, "unit": "m", "status": "validated", "evidence_refs": ["evidence:x0"]},
        "count_basis": {"value": "firmware_decoded_counts", "unit": None, "status": "validated", "evidence_refs": ["evidence:count_basis"]},
        "firmware": {"value": {"id": "drive-fw-1", "configuration_digest": source_map["source:firmware"]["source_digest"]}, "unit": None, "status": "validated", "evidence_refs": ["evidence:firmware"]},
        "homing": {"value": {"method": "hard-stop", "reference_established": True}, "unit": None, "status": "validated", "evidence_refs": ["evidence:homing"]},
        "frame": {"value": "frame:carriage", "unit": None, "status": "validated", "evidence_refs": ["evidence:frame"]},
        "time_basis": {"value": "clock:machine-utc", "unit": None, "status": "validated", "evidence_refs": ["evidence:time_basis"]},
    }
    covariance = [[0.0] * 6 for _ in range(6)]
    for index, value in enumerate((4.0, 0.01, 1e-6, 1e-8, 1.0, 1e-8)):
        covariance[index][index] = value
    uncertainty = {
        "method": "first_order_joint_covariance",
        "coordinates": list(manifest.COORDINATES),
        "units": ["count", "count", "m", "m/rev_output", "count/rev_motor", "rev_motor/rev_output"],
        "covariance": covariance,
        "basis": "declared_prior_covariance",
        "distribution": "unspecified",
        "confidence_level": None,
        "status": "validated",
        "evidence_refs": ["evidence:uncertainty"],
    }
    evidence_items = []
    for key, claim in claims.items():
        source_id = {
            "C": "source:counts", "g": "source:gear", "L": "source:lead", "s": "source:sign",
            "N0": "source:offset", "x0": "source:offset", "count_basis": "source:counts",
            "firmware": "source:firmware", "homing": "source:home", "frame": "source:frame",
            "time_basis": "source:time",
        }[key]
        source = source_map[source_id]
        evidence_items.append({
            "evidence_id": claim["evidence_refs"][0], "machine_id": machine_id,
            "source_id": source_id, "source_digest": source["source_digest"], "status": "validated",
            "claims": {key: manifest.digest({"value": claim["value"], "unit": claim["unit"]})},
        })
    evidence_items.append({
        "evidence_id": "evidence:uncertainty", "machine_id": machine_id,
        "source_id": "source:uncertainty", "source_digest": source_map["source:uncertainty"]["source_digest"],
        "status": "validated", "claims": {"uncertainty": manifest.digest({key: value for key, value in uncertainty.items() if key not in {"status", "evidence_refs"}})},
    })
    evidence = manifest.seal({
        "schema": manifest.EVIDENCE_SCHEMA, "role": "evidence_bundle", "machine_id": machine_id,
        "origin": "synthetic_fixture", "sources": sources, "evidence": evidence_items,
    })
    candidate = manifest.seal({
        "schema": manifest.CANDIDATE_SCHEMA, "role": "candidate_manifest", "machine_id": machine_id,
        "status": "candidate", "claims": claims, "uncertainty": uncertainty,
        "evidence_bundle_digest": evidence["artifact_digest"],
    })
    report = manifest.challenge(candidate, evidence)
    return evidence, candidate, report


def test_manifest_challenge_compile_evaluate_and_inspect_are_read_only():
    evidence, candidate, report = _fixture()
    assert report["status"] == "validated"
    compiled = manifest.compile(candidate, evidence, report)
    result = manifest.evaluate(compiled, 300)
    assert result["position"] == pytest.approx(1.001)
    assert result["physical_validation"] == "not_performed"
    inspection = manifest.inspect(compiled)
    assert inspection["activation"] == "read_only"
    assert inspection["state_admission"] == "not_performed"


def test_manifest_rejects_tampering_and_stale_challenge():
    evidence, candidate, report = _fixture()
    tampered = deepcopy(candidate)
    tampered["claims"]["L"]["value"] = 0.02
    with pytest.raises(ValueError, match="digest mismatch"):
        manifest.validate(tampered)
    changed = manifest.seal({key: value for key, value in tampered.items() if key != "artifact_digest"})
    with pytest.raises(ValueError, match="stale or differs"):
        manifest.compile(changed, evidence, report)
