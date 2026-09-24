"""Measurement protocols, retention records and the acceptance boundary for manufacturing tasks.

Scope: the machine-readable measurement-protocol record
(``ciw.lab-measurement-protocol.v1``) with empty hardware-evidence slots and
the checks that make it executable as written (declared fixtures with what
they locate, instrument settings, numbered procedure steps that name only
declared devices and datums, raw formats of the operator captures), the
retention record for raw measurements with calibration, frame-chain and clock
metadata (``ciw.lab-measurement-retention.v1``), their validators (including
the tape layout: which tapes are on the specimen together, their jig slots and
their clearance from markers and datum probe points), the reader
of operator captures in the formats the protocols define, comparators that
refuse to compare a prediction with anything but acquired hardware evidence,
and the policy that keeps production acceptance outside the workbench.

Non-claims: a protocol that validates is a well-formed plan, not an executed
experiment. A retention record that validates is well-formed metadata. A
record of kind ``schema_fixture``, or one carrying the fixture's markers, is
refused as hardware evidence; a record of kind ``measurement`` with matching
raw bytes yields acquisition fields, but these checks cannot tell whether the
bytes came from an instrument (the runner additionally requires a hardware
probe in the same task). A capture that parses is well-formed text; its
origin is the operator's declaration and nothing authenticates it, and a
synthetic capture never supplies acquisition fields. No measurement exists
here. The acceptance-language screen is a vocabulary check on this section's
findings, in addition to the lab-wide authority-phrase screen of
``ciw.lab.evidence``; neither proves that free-text claims are filed in the
right domain. The workbench never accepts or rejects production parts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import math
import re

import numpy as np

from ..core.identities import content_identity
from .evidence import AUTHORITY_DOMAINS, LABELS

PROTOCOL_SCHEMA = "ciw.lab-measurement-protocol.v1"
RETENTION_SCHEMA = "ciw.lab-measurement-retention.v1"

PROTOCOL_FIELDS = ("schema", "protocol_id", "task_id", "title", "purpose", "specimen", "fixtures", "datum_frames",
                   "frame_chain", "markers", "instruments", "required_raw_data", "raw_formats", "calibration_artifacts",
                   "environment", "procedure", "predicted_quantities", "acceptance_criteria", "hardware_measured",
                   "production_acceptance")
FIXTURE_FIELDS = ("id", "kind", "locates", "contacts")
INSTRUMENT_FIELDS = ("id", "kind", "declared_standard_uncertainty_mm", "status", "settings")
STEP_FIELDS = ("step", "action", "uses", "outputs")
RAW_FORMAT_FIELDS = ("role", "schema", "media_type", "instrument", "columns", "reader")
# Device words in free text (a procedure step's action, a path's realization) and the declared device
# each needs among that step's or path's ``uses``: a fixture whose kind contains the keyword, or the
# instrument with that id. A step that names a device it does not declare is not executable as written.
DEVICE_WORDS = (
    (re.compile(r"\bjigs?\b", re.IGNORECASE), "fixture", "jig"),
    (re.compile(r"\bnests?\b", re.IGNORECASE), "fixture", "nest"),
    (re.compile(r"\bV-blocks?\b", re.IGNORECASE), "fixture", "v-block"),
    (re.compile(r"\bCMM\b", re.IGNORECASE), "instrument", "cmm"),
    (re.compile(r"\b(?:cameras?|photogrammetr\w*)\b", re.IGNORECASE), "instrument", "camera"),
    (re.compile(r"\btrackers?\b", re.IGNORECASE), "instrument", "tracker"),
    (re.compile(r"\bscanners?\b", re.IGNORECASE), "instrument", "scanner"),
    (re.compile(r"\bfilm\b", re.IGNORECASE), "instrument", "film"),
    (re.compile(r"\btape[- ]measures?\b", re.IGNORECASE), "instrument", "tape-measure"),
    (re.compile(r"\binterferometers?\b", re.IGNORECASE), "instrument", "interferometer"),
)
# Identifiers of fixtures and calibration artifacts (FX-321, JIG-START-01, VB-01, SB-1000, GS-25.4, SG-200).
DEVICE_ID = re.compile(r"\b(?:FX|JIG|VB|SB|GS|SG)-[0-9A-Z]+(?:[.-][0-9A-Z]+)*")
# A step that measures datum targets must use a fixture that declares them.
DATUM_TARGETS = re.compile(r"\bdatum targets?\b", re.IGNORECASE)
# Structured step fields: the tapes a step lays, the tapes it places targets on, the tapes it removes, and the
# earlier steps a repeat step repeats (by number).
STEP_LAYOUT_KEYS = ("lays", "places", "removes")
LAYOUT_FIELDS = ("tape_width_mm", "target_diameter_mm", "marker_diameter_mm")
RETENTION_FIELDS = ("schema", "record_kind", "protocol_id", "raw", "instrument", "calibration", "frame_chain", "clock")
ACQUISITION_FIELDS = ("device", "raw_sha256", "acquired_at", "calibration")
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
# Markers of the schema fixture: none of them may reach hardware evidence, whatever record_kind says.
FIXTURE_RAW_PREFIX = b"SCHEMA FIXTURE"
FIXTURE_SERIAL_PREFIX = "FIXTURE-"
# First line of every synthetic operator capture (read_capture); such bytes never reach hardware evidence.
SYNTHETIC_CAPTURE_PREFIX = b"# SYNTHETIC CAPTURE - NOT A MEASUREMENT"


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
    """Refuse malformed or unexecutable protocols, filled hardware slots without acquisition, and decisions."""
    _require(record, PROTOCOL_FIELDS, "Measurement protocol")
    if record["schema"] != PROTOCOL_SCHEMA:
        raise RecordRefusal("wrong_schema", f"Expected {PROTOCOL_SCHEMA}")
    for name in ("markers", "instruments", "required_raw_data", "raw_formats", "calibration_artifacts", "procedure",
                 "predicted_quantities", "acceptance_criteria", "datum_frames", "fixtures"):
        if not isinstance(record[name], list) or not record[name]:
            raise RecordRefusal("missing_field", f"Measurement protocol field {name} must be a nonempty list")
    _executable(record)
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


def _uses(entry, where, fixtures, instruments, artifacts) -> list:
    """The declared devices a procedure step or path uses; refuses an identifier nothing declares."""
    uses = entry.get("uses")
    if not isinstance(uses, list) or not all(isinstance(name, str) and name for name in uses):
        raise RecordRefusal("missing_field", f"{where} must list the devices it uses (uses: [...])")
    unknown = [name for name in uses if name not in fixtures and name not in instruments and name not in artifacts]
    if unknown:
        raise RecordRefusal("undeclared_device", f"{where} uses devices the protocol does not declare: {unknown}")
    return uses


def _named_devices(text, uses, where, fixtures, instruments) -> None:
    """Refuse text that names a device (a jig, a CMM, FX-321, ...) absent from the devices its entry uses."""
    for token in DEVICE_ID.findall(text):
        if token not in uses:
            raise RecordRefusal("undeclared_device", f"{where} names {token}, which is not among the devices it uses")
    for pattern, kind, keyword in DEVICE_WORDS:
        if not pattern.search(text):
            continue
        if kind == "instrument":
            present = keyword in uses and keyword in instruments
        else:
            present = any(name in fixtures and keyword in str(fixtures[name]["kind"]).lower() for name in uses)
        if not present:
            raise RecordRefusal("undeclared_device",
                                f"{where} names a {keyword} but uses no declared {kind} of that kind: {text!r}")


def _executable(record) -> None:
    """Every fixture, instrument setting, datum, step, path device and raw format is declared and consistent.

    Fixtures carry what they locate and how; instruments carry their settings;
    procedure steps are numbered objects naming the devices they use, the
    datums they touch and their outputs; a step's action or a path's
    realization that names a device (by word or identifier) must use a
    declared one of that kind; each raw format names a declared instrument.
    """
    fixtures = {}
    for fixture in record["fixtures"]:
        _require(fixture, FIXTURE_FIELDS, "Fixture")
        fixtures[fixture["id"]] = fixture
    instruments = {}
    for instrument in record["instruments"]:
        _require(instrument, INSTRUMENT_FIELDS, "Instrument")
        if not isinstance(instrument["settings"], dict) or not instrument["settings"]:
            raise RecordRefusal("missing_field", f"Instrument {instrument['id']} settings must be a nonempty object")
        instruments[instrument["id"]] = instrument
    artifacts = {artifact.get("id") for artifact in record["calibration_artifacts"] if isinstance(artifact, dict)}
    datums = {datum.get("id") for datum in record["datum_frames"] if isinstance(datum, dict)}
    for name, fixture in fixtures.items():
        unknown = [d for d in fixture["locates"] if d not in datums] if isinstance(fixture["locates"], list) else [None]
        if unknown:
            raise RecordRefusal("undeclared_datum", f"Fixture {name} locates undeclared datums: {unknown}")
    for index, step in enumerate(record["procedure"], start=1):
        if not isinstance(step, dict):
            raise RecordRefusal("procedure_step_malformed", f"Procedure step {index} must be an object with {STEP_FIELDS}")
        _require(step, STEP_FIELDS, f"Procedure step {index}")
        if step["step"] != index or not isinstance(step["outputs"], list) or not step["outputs"]:
            raise RecordRefusal("procedure_step_malformed",
                                f"Procedure step {index} must be numbered {index} and list its outputs")
        where = f"Procedure step {index}"
        uses = _uses(step, where, fixtures, instruments, artifacts)
        unknown = [d for d in step.get("datums", []) if d not in datums]
        if unknown:
            raise RecordRefusal("undeclared_datum", f"{where} touches undeclared datums: {unknown}")
        _named_devices(str(step["action"]), uses, where, fixtures, instruments)
        if DATUM_TARGETS.search(str(step["action"])) and not any(
                isinstance(fixtures[name].get("datum_targets"), list) and fixtures[name]["datum_targets"]
                for name in uses if name in fixtures):
            raise RecordRefusal("fixture_targets_undeclared",
                                f"{where} measures datum targets, but no fixture it uses declares any")
        _repeat(step, index, where, record["procedure"])
    for path in record.get("paths", []):
        where = f"Path {path.get('id')}"
        uses = _uses(path, where, fixtures, instruments, artifacts)
        _named_devices(str(path.get("realization", "")), uses, where, fixtures, instruments)
    for entry in record["raw_formats"]:
        _require(entry, RAW_FORMAT_FIELDS, "Raw format")
        if entry["instrument"] not in instruments:
            raise RecordRefusal("undeclared_device",
                                f"Raw format {entry['role']} names instrument {entry['instrument']!r}, which is not declared")
    _tape_layout(record, fixtures)
    # A hypothesis may be tested only with what the protocol acquires: a criterion that names a device must name
    # a declared one (a surface distance "by tape-measure" with no tape measure declared is not testable as written).
    declared = [*fixtures, *instruments, *artifacts]
    for criterion in record["acceptance_criteria"]:
        if isinstance(criterion, dict):
            for key in ("statement", "test"):
                _named_devices(str(criterion.get(key) or ""), declared, f"Acceptance criterion {criterion.get('id')}",
                               fixtures, instruments)


def _repeat(step, index, where, procedure) -> None:
    """A repeat step lists the earlier steps it repeats; none of them may lay, place or remove tapes or targets."""
    action = str(step["action"])
    if "repeats" not in step:
        if action.startswith("Repeat"):
            raise RecordRefusal("procedure_step_malformed", f"{where} repeats steps without listing them (repeats: [...])")
        return
    repeats = step["repeats"]
    if not isinstance(repeats, list) or not repeats or any(
            not isinstance(number, int) or isinstance(number, bool) or not 1 <= number < index for number in repeats):
        raise RecordRefusal("procedure_step_malformed", f"{where} must repeat earlier steps, listed by number")
    laying = [number for number in repeats if any(key in procedure[number - 1] for key in STEP_LAYOUT_KEYS)]
    if laying:
        raise RecordRefusal("repeat_includes_laying",
                            f"{where} repeats steps {laying}, which lay, place or remove tapes or targets")
    if not action.startswith("Repeat step") or any(not re.search(rf"\b{number}\b", action) for number in repeats):
        raise RecordRefusal("procedure_step_malformed", f"{where} must name the steps {repeats} it repeats")


def _number(value, where) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise RecordRefusal("missing_field", f"{where} must be a finite number")
    return float(value)


def _polyline(points, where) -> np.ndarray:
    try:
        line = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        line = np.empty((0, 3))
    if line.ndim != 2 or line.shape[1] != 3 or len(line) < 2 or not np.all(np.isfinite(line)) \
            or np.min(np.linalg.norm(np.diff(line, axis=0), axis=1)) <= 0.0:
        raise RecordRefusal("path_geometry_missing",
                            f"{where} needs its predicted centreline as distinct 3D points in mm (centreline_mm)")
    return line


def point_polyline_distance(point, line) -> float:
    """Smallest distance from a 3D point to a polyline."""
    point, line = np.asarray(point, dtype=float), np.asarray(line, dtype=float)
    start, direction = line[:-1], np.diff(line, axis=0)
    t = np.clip(np.einsum("ij,ij->i", point - start, direction) / np.einsum("ij,ij->i", direction, direction), 0.0, 1.0)
    return float(np.min(np.linalg.norm(start + direction * t[:, None] - point, axis=1)))


def polyline_distance(first, second) -> float:
    """Smallest distance between two polylines: exact closest points of every segment pair (clamped solution)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    p, d1 = first[:-1, None, :], np.diff(first, axis=0)[:, None, :]
    q, d2 = second[None, :-1, :], np.diff(second, axis=0)[None, :, :]
    r = p - q
    a, e = np.sum(d1 * d1, axis=-1), np.sum(d2 * d2, axis=-1)
    b, c, f = np.sum(d1 * d2, axis=-1), np.sum(d1 * r, axis=-1), np.sum(d2 * r, axis=-1)
    denominator = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denominator > 1e-12 * a * e, np.clip((b * f - c * e) / denominator, 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
    return float(np.min(np.linalg.norm(p + d1 * s[..., None] - (q + d2 * t[..., None]), axis=-1)))


def slot_gap(one: dict, two: dict, length: float) -> float:
    """Smallest centreline distance of two jig slots over their length (0 if they cross).

    A slot's centreline is offset laterally and rotated in heading from the
    nominal slot at the slot exit: y(x) = delta + x tan(theta) for x in
    [-length, 0].
    """
    exit_ = _number(one["lateral_offset_mm"], "Slot offset") - _number(two["lateral_offset_mm"], "Slot offset")
    back = exit_ - length * (math.tan(_number(one["heading_offset_rad"], "Slot heading"))
                             - math.tan(_number(two["heading_offset_rad"], "Slot heading")))
    return 0.0 if exit_ * back <= 0.0 else min(abs(exit_), abs(back))


def _tape_layout(record, fixtures) -> None:
    """Tapes on the specimen together, the jig slots of one laying, and tape clearance from markers.

    The procedure says which tapes are on the specimen when (``lays`` and
    ``removes``) and where targets go (``places``); each tape path carries
    its predicted centreline. Refused: two slots used in one laying less than
    one slot width apart (edge to edge), two tapes on the specimen together
    whose centrelines come closer than a tape width plus a target diameter
    (a target on one must clear the other tape and its targets), a tape whose
    centreline passes a specimen marker or a declared datum probe point closer
    than half the wider of tape and target plus half a marker, a target placed
    on a tape that is not on the specimen, and a tape that is never laid.
    Distances between sampled 3D points are chords, never longer than surface
    distances, so the check errs on the side of refusal.
    """
    tapes = {path.get("id"): path for path in record.get("paths", []) if isinstance(path, dict) and path.get("kind") == "tape"}
    procedure = record["procedure"]
    if not tapes and not any(key in step for step in procedure for key in STEP_LAYOUT_KEYS):
        return
    layout = record.get("tape_layout")
    _require(layout, LAYOUT_FIELDS, "Tape layout")
    tape, target, marker = (_number(layout[key], f"Tape layout {key}") for key in LAYOUT_FIELDS)
    lines = {name: _polyline(path.get("centreline_mm"), f"Path {name}") for name, path in tapes.items()}
    obstacles = [(entry.get("id"), entry.get("position_mm")) for entry in record["markers"]
                 if isinstance(entry, dict) and entry.get("kind") == "specimen marker"]
    obstacles += [(f"datum {datum.get('id')} probe point {k + 1}", point) for datum in record["datum_frames"]
                  if isinstance(datum, dict) for k, point in enumerate(datum.get("probe_points_mm") or [])]
    clearance = 0.5 * (max(tape, target) + marker)
    for name, line in sorted(lines.items()):
        for label, position in obstacles:
            point = np.asarray(position, dtype=float) if position is not None else np.empty(0)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise RecordRefusal("path_geometry_missing", f"Marker or probe point {label} needs a 3D position_mm")
            distance = point_polyline_distance(point, line)
            if distance < clearance:
                raise RecordRefusal("path_over_marker", f"Tape {name} passes {distance:.2f} mm from {label}; it must "
                                                        f"clear it by {clearance:.2f} mm")
    slots = {}
    for fixture_id, fixture in fixtures.items():
        geometry = fixture.get("geometry")
        if isinstance(geometry, dict) and isinstance(geometry.get("slots"), list):
            for slot in geometry["slots"]:
                slots[slot.get("slot")] = (fixture_id, slot, _number(geometry.get("slot_width_mm"), "Slot width"),
                                           _number(geometry.get("slot_length_mm"), "Slot length"))
    present, laid = [], set()
    for index, step in enumerate(procedure, start=1):
        where = f"Procedure step {index}"
        entries = {key: step.get(key, []) for key in STEP_LAYOUT_KEYS}
        if any(not isinstance(names, list) or any(name not in tapes for name in names) for names in entries.values()):
            raise RecordRefusal("procedure_step_malformed", f"{where} lays, places or removes tapes that are not tape paths")
        if entries["lays"]:
            used = []
            for name in entries["lays"]:
                if name not in slots or slots[name][0] not in step["uses"]:
                    raise RecordRefusal("procedure_step_malformed", f"{where} lays tape {name} from no jig slot it uses")
                used.append(slots[name])
            for (_, one, width, length), (_, two, _, _) in itertools.combinations(used, 2):
                if slot_gap(one, two, length) < 2.0 * width:
                    raise RecordRefusal("jig_slots_overlap", f"{where} lays tapes from slots {one.get('slot')} and "
                                                             f"{two.get('slot')}, which are less than one slot width apart")
            for name in entries["lays"]:
                for other in present:
                    distance = polyline_distance(lines[name], lines[other])
                    if distance < tape + target:
                        raise RecordRefusal("tapes_overlap", f"{where} lays tape {name} while tape {other} is on the "
                                                             f"specimen; they come {distance:.2f} mm apart, less than "
                                                             f"{tape + target:.2f} mm")
                present.append(name)
                laid.add(name)
        for name in entries["places"]:
            if name not in present:
                raise RecordRefusal("procedure_step_malformed", f"{where} places targets on tape {name}, which is not laid")
        for name in entries["removes"]:
            if name not in present:
                raise RecordRefusal("procedure_step_malformed", f"{where} removes tape {name}, which is not laid")
            present.remove(name)
    never = sorted(set(tapes) - laid)
    if never:
        raise RecordRefusal("tape_not_laid", f"No procedure step lays the tape paths {never}")


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
        "step_names_undeclared_jig": (lambda r: r["procedure"][0].update(
            action="Lay the tape from the start jig.", uses=[]), "undeclared_device"),
        "step_uses_undeclared_device": (lambda r: r["procedure"][0]["uses"].append("JIG-MISSING-01"),
                                        "undeclared_device"),
        "unstructured_procedure_step": (lambda r: r["procedure"].__setitem__(0, "Soak the specimen."),
                                        "procedure_step_malformed"),
        "instrument_without_settings": (lambda r: r["instruments"][0].pop("settings"), "missing_field"),
        "raw_format_of_undeclared_instrument": (lambda r: r["raw_formats"][0].update(instrument="interferometer"),
                                                "undeclared_device"),
        "criterion_names_undeclared_instrument": (lambda r: r["acceptance_criteria"][0].update(
            test=str(r["acceptance_criteria"][0]["test"]) + "; surface distance by tape-measure and laser interferometer"),
            "undeclared_device"),
        "step_measures_targets_of_a_fixture_without_any": (_targets_undeclared, "fixture_targets_undeclared"),
    }
    procedure = record["procedure"]
    tapes = [path for path in record.get("paths", []) if isinstance(path, dict) and path.get("kind") == "tape"]
    lay_steps = [k for k, step in enumerate(procedure) if isinstance(step, dict) and step.get("lays")]
    remove_steps = [k for k, step in enumerate(procedure) if isinstance(step, dict) and step.get("removes")]
    repeat_steps = [k for k, step in enumerate(procedure) if isinstance(step, dict) and "repeats" in step]
    if repeat_steps:
        first = repeat_steps[0]
        cases["repeat_of_itself"] = (lambda r: r["procedure"][first].update(repeats=[first + 1]), "procedure_step_malformed")
    if lay_steps and repeat_steps and repeat_steps[-1] > lay_steps[0]:
        last = repeat_steps[-1]
        cases["repeat_includes_laying"] = (lambda r: r["procedure"][last].update(
            repeats=sorted({*r["procedure"][last]["repeats"], lay_steps[0] + 1})), "repeat_includes_laying")
    if len(tapes) >= 2 and lay_steps:
        both = [tapes[0]["id"], tapes[1]["id"]]
        cases["two_offset_tapes_in_one_laying"] = (lambda r: r["procedure"][lay_steps[0]].update(lays=both),
                                                   "jig_slots_overlap")
    if len(lay_steps) >= 2 and remove_steps and remove_steps[0] < lay_steps[1]:
        cases["tape_left_on_the_specimen"] = (lambda r: r["procedure"][remove_steps[0]].update(removes=[]),
                                              "tapes_overlap")
    if tapes:
        line = tapes[0].get("centreline_mm") or [[0.0, 0.0, 0.0]]
        cases["specimen_marker_under_a_tape"] = (lambda r: r["markers"].append(
            {"id": "MX", "kind": "specimen marker", "position_mm": list(line[len(line) // 2])}), "path_over_marker")
    return {name: {"expected": expected, "observed": refusal_code(validate_protocol, mutated(change))}
            for name, (change, expected) in cases.items()}


def _targets_undeclared(record) -> None:
    """Mutation: the first fixture declares no datum targets, and the first step measures them."""
    fixture = record["fixtures"][0]
    fixture.pop("datum_targets", None)
    record["procedure"][0].update(action=f"Measure the four datum targets of {fixture['id']}.", uses=[fixture["id"]])


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
    if (any(bytes(data).startswith((FIXTURE_RAW_PREFIX, SYNTHETIC_CAPTURE_PREFIX)) for data in raw_bytes.values())
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


# Operator captures -------------------------------------------------------------
# Raw formats the protocols define for operator captures (``ciw lab run --capture ROLE=PATH``): a header
# of "# key: value" lines, then a CSV table with exactly these columns.
CAPTURE_SCHEMAS = {"ciw.lab-mfg-target-capture.v1": ("target", "x_mm", "y_mm", "z_mm", "u_mm"),
                   "ciw.lab-mfg-distance-capture.v1": ("pair", "distance_mm", "u_mm")}
CAPTURE_HEADER = ("schema", "protocol", "instrument", "frame", "unit", "origin")
CAPTURE_ORIGINS = ("measurement", "synthetic")
_HEADER_LINE = re.compile(r"#\s*([a-z_]+)\s*:\s*(.*\S)\s*$")


def read_capture(data: bytes, expected: dict | None = None) -> dict:
    """Parse the bytes of a section-9 operator capture into ``{header..., "rows": {id: [floats]}}``.

    The header declares schema (a key of :data:`CAPTURE_SCHEMAS`), protocol,
    instrument, frame, unit (``mm``) and origin (``measurement`` or
    ``synthetic``). A synthetic capture begins with
    :data:`SYNTHETIC_CAPTURE_PREFIX` and declares origin synthetic, and only
    then. Rows have the schema's columns, a unique nonempty identifier, finite
    coordinates or distances and a finite positive standard uncertainty (a zero
    uncertainty would make every normalized error infinite or undefined).
    ``expected`` may name the schema, protocol, instrument and frame the
    header must declare (refused with ``capture_expected``) and the row ``ids``
    the table must hold exactly (``capture_ids``). The origin is the operator's
    declaration: nothing here authenticates the bytes.
    """
    expected = dict(expected or {})
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        raise RecordRefusal("capture_encoding", "An operator capture must be UTF-8 text") from None
    header, table = {}, []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            match = _HEADER_LINE.fullmatch(stripped)
            if match and not table:
                if match.group(1) in header:
                    raise RecordRefusal("capture_header", f"Capture header key {match.group(1)} is repeated")
                header[match.group(1)] = match.group(2)
            continue
        table.append(stripped)
    missing = [key for key in CAPTURE_HEADER if key not in header]
    if missing:
        raise RecordRefusal("capture_header", f"Capture header lacks {missing}")
    synthetic = bytes(data).startswith(SYNTHETIC_CAPTURE_PREFIX)
    if header["origin"] not in CAPTURE_ORIGINS or header["unit"] != "mm" \
            or synthetic != (header["origin"] == "synthetic"):
        raise RecordRefusal("capture_header", "Capture unit must be mm and origin measurement or synthetic; a synthetic "
                                              "capture, and only one, starts with the synthetic banner")
    columns = CAPTURE_SCHEMAS.get(header["schema"])
    if columns is None:
        raise RecordRefusal("capture_schema", f"Unknown capture schema {header['schema']!r}")
    for key in ("schema", "protocol", "instrument", "frame"):
        if key in expected and header[key] != expected[key]:
            raise RecordRefusal("capture_expected", f"Capture declares {key} {header[key]!r}, not {expected[key]!r}")
    if not table or tuple(cell.strip() for cell in table[0].split(",")) != columns:
        raise RecordRefusal("capture_columns", f"Capture table must start with the columns {columns}")
    rows = {}
    for line in table[1:]:
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(columns) or not cells[0]:
            raise RecordRefusal("capture_row", f"Capture row does not have the {len(columns)} columns {columns}")
        try:
            values = [float(cell) for cell in cells[1:]]
        except ValueError:
            raise RecordRefusal("capture_value", f"Capture row {cells[0]} holds a value that is not a number") from None
        if not all(np.isfinite(values)) or values[-1] <= 0:
            raise RecordRefusal("capture_value", f"Capture row {cells[0]} must be finite with a positive uncertainty")
        if cells[0] in rows:
            raise RecordRefusal("capture_duplicate", f"Capture row {cells[0]} is repeated")
        rows[cells[0]] = values
    ids = expected.get("ids")
    if ids is not None and set(rows) != set(ids):
        raise RecordRefusal("capture_ids", f"Capture rows differ from the protocol's: missing "
                                           f"{sorted(set(ids) - set(rows))}, unexpected {sorted(set(rows) - set(ids))}")
    return dict(header, rows=rows)


def write_capture(schema: str, protocol: str, instrument: str, frame: str, rows: dict, origin: str = "synthetic") -> bytes:
    """Bytes of a capture in the :func:`read_capture` format; a synthetic one starts with the synthetic banner."""
    columns = CAPTURE_SCHEMAS[schema]
    lines = [SYNTHETIC_CAPTURE_PREFIX.decode("ascii")] if origin == "synthetic" else []
    lines += [f"# {key}: {value}" for key, value in (("schema", schema), ("protocol", protocol),
                                                    ("instrument", instrument), ("frame", frame), ("unit", "mm"),
                                                    ("origin", origin))]
    lines.append(",".join(columns))
    lines += [",".join([name] + [repr(float(v)) for v in values]) for name, values in rows.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


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
    decision words above (in claims and in string values) in any domain that
    could be established, not every paraphrase. It complements
    ``evidence.screen_authority_claim``, which refuses authority-outcome
    phrases in every section's claims; which domain a free-text claim belongs
    to remains a review question.
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
