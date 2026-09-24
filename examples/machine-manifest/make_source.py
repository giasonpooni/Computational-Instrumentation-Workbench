"""Author the evidence-bound encoder example and print its exact source.

Ten declared documentation, observation, firmware and validation sources back a
candidate manifest for one leadscrew carriage axis: counts per motor turn, a
gear ratio, the lead, orientation, a homing offset, the frame and clock, the
firmware configuration digest and a declared joint prior covariance. The
challenge report is the deterministic comparison of the candidate against that
evidence, and the request asks for the position of one decoded count value.
Every value is a synthetic fixture; nothing is read from a device.
"""
from hashlib import sha256
import json

from ciw import machine_manifest as manifest
from ciw import machine_workflow

MACHINE_ID = "machine:example-carriage"
SOURCES = [
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
CLAIM_SOURCES = {"C": "source:counts", "g": "source:gear", "L": "source:lead", "s": "source:sign",
                 "N0": "source:offset", "x0": "source:offset", "count_basis": "source:counts",
                 "firmware": "source:firmware", "homing": "source:home", "frame": "source:frame",
                 "time_basis": "source:time"}


def _declared(source_id, kind, content):
    return {"source_id": source_id, "machine_id": MACHINE_ID, "kind": kind, "locator": "fixture://" + source_id,
            "content_utf8": content, "source_digest": "sha256:" + sha256(content.encode()).hexdigest()}


def evidence_and_candidate():
    sources = [_declared(*item) for item in SOURCES]
    source_map = {source["source_id"]: source for source in sources}
    claims = {
        "C": {"value": 1000, "unit": "count/rev_motor"},
        "g": {"value": 2.0, "unit": "rev_motor/rev_output"},
        "L": {"value": 0.01, "unit": "m/rev_output"},
        "s": {"value": 1, "unit": "1"},
        "N0": {"value": 100, "unit": "count"},
        "x0": {"value": 1.0, "unit": "m"},
        "count_basis": {"value": "firmware_decoded_counts", "unit": None},
        "firmware": {"value": {"id": "drive-fw-1", "configuration_digest": source_map["source:firmware"]["source_digest"]}, "unit": None},
        "homing": {"value": {"method": "hard-stop", "reference_established": True}, "unit": None},
        "frame": {"value": "frame:carriage", "unit": None},
        "time_basis": {"value": "clock:machine-utc", "unit": None},
    }
    for key, claim in claims.items():
        claim.update({"status": "validated", "evidence_refs": ["evidence:" + key]})
    covariance = [[0.0] * 6 for _ in range(6)]
    for index, value in enumerate((4.0, 0.01, 1e-6, 1e-8, 1.0, 1e-8)):
        covariance[index][index] = value
    uncertainty = {"method": "first_order_joint_covariance", "coordinates": list(manifest.COORDINATES),
                   "units": ["count", "count", "m", "m/rev_output", "count/rev_motor", "rev_motor/rev_output"],
                   "covariance": covariance, "basis": "declared_prior_covariance", "distribution": "unspecified",
                   "confidence_level": None, "status": "validated", "evidence_refs": ["evidence:uncertainty"]}
    items = []
    for key, claim in claims.items():
        source = source_map[CLAIM_SOURCES[key]]
        items.append({"evidence_id": claim["evidence_refs"][0], "machine_id": MACHINE_ID,
                      "source_id": source["source_id"], "source_digest": source["source_digest"], "status": "validated",
                      "claims": {key: manifest.digest({"value": claim["value"], "unit": claim["unit"]})}})
    items.append({"evidence_id": "evidence:uncertainty", "machine_id": MACHINE_ID, "source_id": "source:uncertainty",
                  "source_digest": source_map["source:uncertainty"]["source_digest"], "status": "validated",
                  "claims": {"uncertainty": manifest.digest({key: value for key, value in uncertainty.items()
                                                            if key not in {"status", "evidence_refs"}})}})
    evidence = manifest.seal({"schema": manifest.EVIDENCE_SCHEMA, "role": "evidence_bundle", "machine_id": MACHINE_ID,
                              "origin": "synthetic_fixture", "sources": sources, "evidence": items})
    candidate = manifest.seal({"schema": manifest.CANDIDATE_SCHEMA, "role": "candidate_manifest", "machine_id": MACHINE_ID,
                               "status": "candidate", "claims": claims, "uncertainty": uncertainty,
                               "evidence_bundle_digest": evidence["artifact_digest"]})
    return evidence, candidate


def source():
    evidence, candidate = evidence_and_candidate()
    return {"schema": machine_workflow.SOURCE_SCHEMA, "experiment_id": "machine:example-carriage",
            "configuration": dict(machine_workflow.CONFIGURATION), "evidence_bundle": evidence,
            "candidate_manifest": candidate, "challenge_report": manifest.challenge(candidate, evidence),
            "request": {"counts": 300, "covariance": None}}


if __name__ == "__main__":
    print(json.dumps(source(), indent=1, sort_keys=True))
