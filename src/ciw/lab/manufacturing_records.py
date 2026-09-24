"""Measurement protocols, retention records and the acceptance boundary for manufacturing tasks.

Scope: the machine-readable measurement-protocol record
(``ciw.lab-measurement-protocol.v1``) with empty hardware-evidence slots, the
retention record for raw measurements with calibration, frame-chain and clock
metadata (``ciw.lab-measurement-retention.v1``), their validators, a
comparator that refuses to compare a prediction with anything but acquired
hardware evidence, and the policy that keeps production acceptance outside
the workbench.

Non-claims: a protocol that validates is a well-formed plan, not an executed
experiment. A retention record that validates is well-formed metadata. A
record of kind ``schema_fixture``, or one carrying the fixture's markers, is
refused as hardware evidence; a record of kind ``measurement`` with matching
raw bytes yields acquisition fields, but these checks cannot tell whether the
bytes came from an instrument (the runner additionally requires a hardware
probe in the same task). No measurement exists here. The acceptance-language
screen is a vocabulary check on this section's findings, not a general proof
that free-text claims are filed in the right domain. The workbench never
accepts or rejects production parts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re

import numpy as np

from ..core.identities import content_identity
from .evidence import AUTHORITY_DOMAINS, LABELS

PROTOCOL_SCHEMA = "ciw.lab-measurement-protocol.v1"
RETENTION_SCHEMA = "ciw.lab-measurement-retention.v1"

PROTOCOL_FIELDS = ("schema", "protocol_id", "task_id", "title", "purpose", "specimen", "fixtures", "datum_frames",
                   "frame_chain", "markers", "instruments", "required_raw_data", "calibration_artifacts",
                   "environment", "procedure", "predicted_quantities", "acceptance_criteria", "hardware_measured",
                   "production_acceptance")
RETENTION_FIELDS = ("schema", "record_kind", "protocol_id", "raw", "instrument", "calibration", "frame_chain", "clock")
ACQUISITION_FIELDS = ("device", "raw_sha256", "acquired_at", "calibration")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# Markers of the schema fixture: none of them may reach hardware evidence, whatever record_kind says.
FIXTURE_RAW_PREFIX = b"SCHEMA FIXTURE"
FIXTURE_SERIAL_PREFIX = "FIXTURE-"


class RecordRefusal(ValueError):
    """A protocol, retention record or decision request violates the manufacturing contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def refusal_code(function, *args, **kwargs) -> str | None:
    """Run ``function`` and return the refusal code it raised, or None if it returned."""
    try:
        function(*args, **kwargs)
    except RecordRefusal as exc:
        return exc.code
    return None


def _require(record, fields, what):
    if not isinstance(record, dict):
        raise RecordRefusal("not_an_object", f"{what} must be an object")
    missing = [name for name in fields if name not in record or record[name] in (None, "")]
    if missing:
        raise RecordRefusal("missing_field", f"{what} is missing required fields: {missing}")


# Measurement protocols -------------------------------------------------------
def validate_protocol(record: dict) -> dict:
    """Refuse malformed protocols, filled hardware slots without acquisition, and decisions."""
    _require(record, PROTOCOL_FIELDS, "Measurement protocol")
    if record["schema"] != PROTOCOL_SCHEMA:
        raise RecordRefusal("wrong_schema", f"Expected {PROTOCOL_SCHEMA}")
    for name in ("markers", "instruments", "required_raw_data", "calibration_artifacts", "procedure",
                 "predicted_quantities", "acceptance_criteria", "datum_frames", "fixtures"):
        if not isinstance(record[name], list) or not record[name]:
            raise RecordRefusal("missing_field", f"Measurement protocol field {name} must be a nonempty list")
    for quantity in record["predicted_quantities"]:
        _require(quantity, ("id", "quantity", "value", "unit", "evidence_status", "finding_claim"), "Predicted quantity")
        if quantity["evidence_status"] not in LABELS:
            raise RecordRefusal("unknown_label", f"Unknown evidence status: {quantity['evidence_status']!r}")
        if quantity["evidence_status"] == "hardware_measured":
            raise RecordRefusal("prediction_labelled_measured", "A prediction cannot carry a hardware_measured label")
    for criterion in record["acceptance_criteria"]:
        _require(criterion, ("id", "statement", "status", "test"), "Acceptance criterion")
        if criterion["status"] != "hypothesis":
            raise RecordRefusal("criterion_is_decision",
                                "Acceptance criteria are hypotheses to test, not decisions taken by the workbench")
    for instrument in record["instruments"]:
        _require(instrument, ("id", "kind", "declared_standard_uncertainty_mm", "status"), "Instrument")
        if instrument["status"] != "declared_not_verified":
            raise RecordRefusal("instrument_claimed_verified",
                                "Instrument uncertainty is declared; verification needs a retained calibration record")
    slots = record["hardware_measured"]
    _require(slots, ("status", "records"), "Hardware evidence slot")
    if slots["status"] == "not_acquired":
        if slots["records"]:
            raise RecordRefusal("measurement_without_acquisition",
                                "Hardware records present while the slot is marked not_acquired")
    else:
        if not slots["records"]:
            raise RecordRefusal("measurement_without_acquisition", "Hardware slot claims data but holds no records")
        for entry in slots["records"]:
            acquisition = entry.get("acquisition") if isinstance(entry, dict) else None
            if not isinstance(acquisition, dict) or any(not acquisition.get(k) for k in ACQUISITION_FIELDS):
                raise RecordRefusal("measurement_without_acquisition",
                                    "Every hardware record needs device, raw digest, acquisition time and calibration")
            if not DIGEST_PATTERN.fullmatch(str(acquisition["raw_sha256"])) \
                    or not UTC_PATTERN.match(str(acquisition["acquired_at"])):
                raise RecordRefusal("acquisition_malformed",
                                    "Hardware records need a SHA-256 raw digest and an ISO 8601 time with an offset")
            if not IDENTITY_PATTERN.fullmatch(str(entry.get("retention_identity", ""))):
                raise RecordRefusal("retention_record_missing",
                                    "Every hardware record must cite the identity of its T139 retention record")
    if record["production_acceptance"] != "outside_system":
        raise RecordRefusal("acceptance_inside_system", "Production acceptance is decided outside the workbench")
    return record


def protocol_refusal_matrix(record: dict) -> dict:
    """Mutate a valid protocol in every refused way; map mutation -> observed refusal code."""
    def mutated(change):
        copy = deepcopy(record)
        change(copy)
        return copy

    cases = {
        "missing_markers": (lambda r: r.pop("markers"), "missing_field"),
        "filled_slot_without_acquisition": (lambda r: r["hardware_measured"].update(
            status="acquired", records=[{"value": 1.0}]), "measurement_without_acquisition"),
        "records_in_empty_slot": (lambda r: r["hardware_measured"]["records"].append({"value": 1.0}),
                                  "measurement_without_acquisition"),
        "slot_with_malformed_acquisition": (lambda r: r["hardware_measured"].update(status="acquired", records=[{
            "acquisition": {"device": "camera:1", "raw_sha256": "not-a-digest", "acquired_at": "yesterday",
                            "calibration": "CERT"}, "retention_identity": "sha256:" + "a" * 64}]),
            "acquisition_malformed"),
        "slot_without_retention_record": (lambda r: r["hardware_measured"].update(status="acquired", records=[{
            "acquisition": {"device": "camera:1", "raw_sha256": "a" * 64, "acquired_at": "2026-09-23T00:00:00Z",
                            "calibration": "CERT"}}]), "retention_record_missing"),
        "criterion_marked_accepted": (lambda r: r["acceptance_criteria"][0].update(status="accepted"),
                                      "criterion_is_decision"),
        "prediction_labelled_measured": (lambda r: r["predicted_quantities"][0].update(
            evidence_status="hardware_measured"), "prediction_labelled_measured"),
        "instrument_claimed_verified": (lambda r: r["instruments"][0].update(status="verified"),
                                        "instrument_claimed_verified"),
        "acceptance_inside_system": (lambda r: r.update(production_acceptance="accepted"), "acceptance_inside_system"),
    }
    return {name: {"expected": expected, "observed": refusal_code(validate_protocol, mutated(change))}
            for name, (change, expected) in cases.items()}


# Retention records -----------------------------------------------------------
def _frame_link(link, index):
    _require(link, ("parent", "child", "rotation", "translation_mm", "covariance", "source"), f"Frame link {index}")
    rotation = np.asarray(link["rotation"], dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9) \
            or np.linalg.det(rotation) <= 0:
        raise RecordRefusal("frame_rotation_invalid", f"Frame link {index} rotation is not a proper rotation")
    if np.asarray(link["translation_mm"], dtype=float).shape != (3,):
        raise RecordRefusal("frame_translation_invalid", f"Frame link {index} translation must have three components")
    covariance = np.asarray(link["covariance"], dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise RecordRefusal("frame_covariance_invalid", f"Frame link {index} covariance must be a finite 6 x 6 matrix")
    # Scale-aware tolerances: a rank-deficient covariance in mm^2 at robot lever arms has
    # rounding-level negative eigenvalues far above any absolute threshold.
    scale = max(1.0, float(np.max(np.abs(covariance))))
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if np.max(np.abs(covariance - covariance.T)) > 1e-12 * scale \
            or eigenvalues[0] < -1e-12 * max(1.0, float(eigenvalues[-1])):
        raise RecordRefusal("frame_covariance_invalid", f"Frame link {index} covariance is not symmetric positive semidefinite")


def validate_retention(record: dict, raw_bytes: dict | None = None) -> dict:
    """Refuse retention records with missing metadata, broken frame chains or mismatched digests.

    ``raw_bytes`` maps raw file names to their bytes; when given, every listed
    digest is recomputed.
    """
    _require(record, RETENTION_FIELDS, "Retention record")
    if record["schema"] != RETENTION_SCHEMA:
        raise RecordRefusal("wrong_schema", f"Expected {RETENTION_SCHEMA}")
    if record["record_kind"] not in ("measurement", "schema_fixture"):
        raise RecordRefusal("unknown_record_kind", "record_kind must be measurement or schema_fixture")
    if not isinstance(record["raw"], list) or not record["raw"]:
        raise RecordRefusal("raw_missing", "A retention record must list its raw files")
    for entry in record["raw"]:
        _require(entry, ("name", "sha256", "bytes", "media_type"), "Raw file entry")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
            raise RecordRefusal("raw_digest_invalid", "Raw digests must be lowercase SHA-256 hex")
        if raw_bytes is not None:
            data = raw_bytes.get(entry["name"])
            if data is None or hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
                raise RecordRefusal("raw_digest_mismatch", f"Raw file {entry['name']} does not match its digest")
    _require(record["instrument"], ("id", "kind", "serial"), "Instrument identity")
    calibration = record["calibration"]
    _require(calibration, ("status",), "Calibration reference")
    if calibration["status"] == "applied":
        _require(calibration, ("reference", "sha256", "valid_from", "valid_until"), "Applied calibration")
        if not DIGEST_PATTERN.fullmatch(str(calibration["sha256"])):
            raise RecordRefusal("calibration_digest_invalid", "Calibration digests must be lowercase SHA-256 hex")
        if not all(UTC_PATTERN.match(str(calibration[k])) for k in ("valid_from", "valid_until")):
            raise RecordRefusal("calibration_window_invalid", "Calibration validity needs ISO 8601 times with offsets")
    elif calibration["status"] != "not_applied":
        raise RecordRefusal("calibration_status_invalid", "Calibration status must be applied or not_applied")
    chain = record["frame_chain"]
    if not isinstance(chain, list) or not chain:
        raise RecordRefusal("frame_chain_missing", "A retention record needs its frame chain")
    for index, link in enumerate(chain):
        _frame_link(link, index)
        if index and chain[index - 1]["child"] != link["parent"]:
            raise RecordRefusal("frame_chain_broken", f"Frame link {index} does not start where link {index - 1} ends")
    clock = record["clock"]
    _require(clock, ("source", "acquired_at", "synchronization", "uncertainty_s"), "Clock")
    if not UTC_PATTERN.match(str(clock["acquired_at"])):
        raise RecordRefusal("clock_without_timezone", "Acquisition time must be ISO 8601 with an explicit offset")
    if calibration["status"] == "applied":
        acquired = _instant(clock["acquired_at"])
        if not _instant(calibration["valid_from"]) <= acquired <= _instant(calibration["valid_until"]):
            raise RecordRefusal("calibration_expired", "Acquisition time lies outside the calibration validity window")
    return record


def _instant(text: str) -> datetime:
    # Python 3.11 parses both 'Z' and '+hh:mm'; UTC_PATTERN has already required an explicit offset.
    return datetime.fromisoformat(text)


def raw_manifest(record: dict) -> bytes:
    """Canonical bytes listing every raw file (name, digest, size, media type) of a retention record.

    The acquisition digest is the SHA-256 of these bytes, so it binds every raw
    file, not only the first; a task that cites it must retain these bytes.
    """
    entries = sorted(({k: entry[k] for k in ("name", "sha256", "bytes", "media_type")} for entry in record["raw"]),
                     key=lambda entry: entry["name"])
    return json.dumps({"schema": RETENTION_SCHEMA, "raw": entries}, sort_keys=True, separators=(",", ":")).encode("utf-8")


def to_acquisition(record: dict, raw_bytes: dict | None = None) -> dict:
    """The acquisition fields of a hardware_measured finding; refuses anything but a real measurement."""
    validate_retention(record, raw_bytes)
    if record["record_kind"] != "measurement":
        raise RecordRefusal("fixture_is_not_measurement", "A schema fixture cannot supply hardware evidence")
    if raw_bytes is None:
        raise RecordRefusal("raw_bytes_not_presented", "Hardware evidence requires the raw bytes to recompute digests")
    # record_kind is self-declared: refuse the fixture's own markers whatever the record calls itself.
    calibration = record["calibration"]
    if (any(bytes(data).startswith(FIXTURE_RAW_PREFIX) for data in raw_bytes.values())
            or str(record["instrument"]["serial"]).startswith(FIXTURE_SERIAL_PREFIX)
            or (calibration["status"] == "applied" and set(str(calibration["sha256"])) == {"0"})):
        raise RecordRefusal("fixture_is_not_measurement", "Schema-fixture bytes, serials or calibration digests "
                                                          "cannot supply hardware evidence")
    return {"device": f"{record['instrument']['kind']}:{record['instrument']['id']}:{record['instrument']['serial']}",
            "raw_sha256": hashlib.sha256(raw_manifest(record)).hexdigest(), "acquired_at": record["clock"]["acquired_at"],
            "calibration": calibration.get("reference", "not_applied") if calibration["status"] == "applied"
            else "not_applied"}


def retention_identity(record: dict) -> str:
    return content_identity(record)


def normalized_error(measured, predicted, u_measured, u_predicted) -> np.ndarray:
    """E_n = |m - p| / sqrt(U_m^2 + U_p^2) with expanded uncertainties; E_n <= 1 is agreement."""
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    return np.abs(measured - predicted) / np.sqrt(np.asarray(u_measured) ** 2 + np.asarray(u_predicted) ** 2)


def _compare(predicted_values, predicted_uncertainty, measured, labels=None, mismatch="stations_mismatch") -> dict:
    """E_n item by item between a prediction and a measurement record; refuses without hardware evidence."""
    if measured is None:
        raise RecordRefusal("measurement_absent", "No measured value exists for this prediction")
    acquisition = to_acquisition(measured.get("record"), measured.get("raw_bytes"))
    predicted_values = np.asarray(predicted_values, dtype=float)
    values = np.asarray(measured.get("values_mm", []), dtype=float)
    u_measured = np.asarray(measured.get("expanded_uncertainty_mm", []), dtype=float)
    if values.shape != predicted_values.shape or u_measured.shape not in ((), predicted_values.shape) \
            or (labels is not None and list(measured.get("labels", [])) != list(labels)):
        raise RecordRefusal(mismatch, "Measured values must be given at the predicted stations or marker pairs")
    en = normalized_error(values, predicted_values, u_measured, predicted_uncertainty)
    return {"acquisition": acquisition, "normalized_error": en.tolist(), "agrees": bool(np.all(en <= 1.0))}


def compare_separation(predicted: dict, measured: dict | None) -> dict:
    """Compare predicted and measured separations station by station; refuses without hardware evidence.

    ``measured`` = {"record": retention record, "raw_bytes": {name: bytes},
    "values_mm": [...], "expanded_uncertainty_mm": [...]} at the predicted
    stations. The record must be a real measurement whose raw bytes match.
    """
    return _compare(predicted["separation_mm"], predicted["expanded_uncertainty_mm"], measured)


def compare_pair_distances(predicted: dict, measured: dict | None) -> dict:
    """Compare predicted and measured marker-pair quantities pair by pair (E_n); refuses without hardware evidence.

    ``predicted`` = {"pairs": [...], "values_mm": [...], "expanded_uncertainty_mm": [...]}
    (plate chords, which equal the geodesic distances, or cylinder chord-geodesic
    gaps); ``measured`` adds "record" and "raw_bytes" as for
    :func:`compare_separation` and must list the same pairs under "labels".
    """
    return _compare(predicted["values_mm"], predicted["expanded_uncertainty_mm"], measured, labels=predicted["pairs"],
                    mismatch="pairs_mismatch")


# Production acceptance -------------------------------------------------------
# Words that state an acceptance or rejection decision (not the topic of acceptance): a claim or
# string value using them outside an authority domain is refused by the screen.
DECISION_WORDS = re.compile(r"\b(accepted|approved|rejected|scrapped|quarantined|signed[ -]off|dispositioned|"
                            r"released (?:for|to) (?:production|use|shipment)|pass(?:ed|es) (?:inspection|acceptance)|"
                            r"certified (?:for|as) (?:production|use|conforming))\b",
                            re.IGNORECASE)


def screen_acceptance_language(findings) -> list:
    """Refuse findings outside the authority domains whose claim or string value states an acceptance decision.

    Returns the screened claims. This is a vocabulary check: it catches the
    decision words above in any domain that could be established, not every
    paraphrase; which domain a free-text claim belongs to remains a review
    question.
    """
    for record in findings:
        if record["domain"] in AUTHORITY_DOMAINS:
            continue
        texts = [record["claim"]] + _strings(record.get("value"))
        hit = next((m.group(0) for m in map(DECISION_WORDS.search, texts) if m), None)
        if hit:
            raise RecordRefusal("acceptance_outside_authority_domain",
                                f"Claim states an acceptance decision ({hit!r}) outside an authority domain: "
                                f"{record['claim']!r}")
    return [record["claim"] for record in findings]


def _strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


@dataclass(frozen=True)
class AcceptancePolicy:
    """Production acceptance is an external authority decision; the workbench records it as not performed."""

    authority: str = "external"
    decisions_performed: bool = False

    def decide(self, request: dict):
        raise RecordRefusal("production_acceptance_outside_system",
                            "Production acceptance is decided outside the workbench; no decision was made")

    def record(self, request: dict) -> dict:
        return {"decision": "not_performed", "authority": self.authority,
                "request": deepcopy(request), "statement": "The workbench supplies evidence records only."}
