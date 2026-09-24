"""Manufacturing and robotic use cases T126-T141: protocols, path sensitivities, rankings and boundaries.

Scope: machine-readable measurement protocols a technician can execute as
written (declared fixtures including the tape start jig, datum schemes,
instrument settings, numbered steps, raw formats of the instrument exports)
for a flat-plate, a rolled-cylinder and a domed-coupon control experiment and
for the as-built scan of the coupon; metrology procedures (curvature sampling,
as-built dome fit, registration, calibration artifacts, datum frames, Gage R&R
and type-1 procedures); model sensitivities of tape placement, filament
winding, coating or welding and robotic inspection paths from the Jacobi
transfer; path rankings by calibration tolerance, focus margin and focal
clearance ratio; uncertainty budgets; the reader and comparison of operator
captures in the protocols' raw formats; the retention schema for future
measurements; and the production-acceptance boundary.

Non-claims: nothing in this section is measured. Every specimen, instrument
uncertainty, friction coefficient, steering limit and tolerance is a declared
input. Predictions are labelled from their computational basis; every claim
about a physical part, a real instrument, a real calibration, machine safety,
customer demand or production acceptance is recorded as a ``not_established``
finding in its physical or authority domain. An operator capture is read,
retained and compared computationally but is not authenticated, and no
metrology instrument probe exists, so it never supports a physical label.
Hardware-evidence slots in the protocols are empty, and the workbench never
accepts or rejects production parts.
"""
from __future__ import annotations

from copy import deepcopy
import functools
import hashlib
import itertools
import json
import math

import numpy as np

from .. import __version__
from . import integrators, jacobi, svg
from . import manufacturing_geometry as geo
from . import manufacturing_metrology as met
from . import manufacturing_records as rec
from .evidence import AUTHORITY_DOMAINS, DOMAINS, EvidenceRefusal, finding, holds, supported_label, validate_finding
from .registry import task

MODULE = "src/ciw/lab/manufacturing.py"
GEOMETRY = "src/ciw/lab/manufacturing_geometry.py"
METROLOGY = "src/ciw/lab/manufacturing_metrology.py"
RECORDS = "src/ciw/lab/manufacturing_records.py"
DOC = "docs/lab/MANUFACTURING.md"
TESTS = "tests/test_lab_manufacturing.py"
SEED = 20260926

# Declared instrument standard uncertainties (1 sigma per coordinate). These
# are planning inputs in the style of specification sheets, not properties of
# any device; the protocols mark them declared_not_verified.
INSTRUMENTS = {
    "camera": {"id": "camera", "kind": "photogrammetry camera system with coded targets",
               "declared_standard_uncertainty_mm": 0.02, "status": "declared_not_verified",
               "settings": {"stations": "12 convergent stations: 4 heights x 3 roll angles",
                            "targets": "6 mm coded retro-reflective targets", "scale": "SB-1000 in every image set",
                            "reference": "the holder's datum targets (coded adapters in their SMR nests) in every "
                                         "image set, so image sets taken while different tapes are on the specimen "
                                         "share the FIXTURE frame; the declared uncertainty covers a target "
                                         "coordinate in that frame, registration of its image set included",
                            "exposure": "fixed exposure with ring flash; lossless images"}},
    "tracker": {"id": "tracker", "kind": "laser tracker with 1.5 in SMR",
                "declared_standard_uncertainty_mm": 0.015, "length_dependent_um_per_m": 6.0,
                "status": "declared_not_verified",
                "settings": {"target": "1.5 in SMR", "mode": "stable point, 2 s averaging", "warm_up_h": 1}},
    "cmm": {"id": "cmm", "kind": "bridge CMM with touch-trigger probe",
            "declared_standard_uncertainty_mm": 0.002, "status": "declared_not_verified",
            "settings": {"probe": "touch-trigger probe, 2 mm ruby stylus", "approach_speed_mm_s": 3.0,
                         "qualification": "stylus qualified on GS-25.4 at the start of each session",
                         "points": "as listed in each procedure step"}},
    "scanner": {"id": "scanner", "kind": "laser line scanner", "declared_standard_uncertainty_mm": 0.01,
                "native_point_spacing_mm": 0.05, "status": "declared_not_verified",
                "settings": {"standoff_mm": 100.0, "max_incidence_deg": 30.0, "point_spacing_mm": 0.05,
                             "exposure": "automatic per pass, logged with the pass"}},
    "film": {"id": "film", "kind": "unrolled-film gauge: flexible polyester film with a printed 0.5 mm scale",
             "declared_standard_uncertainty_mm": 0.05, "status": "declared_not_verified",
             "settings": {"graduation_mm": 0.5, "reading": "10x loupe at both marker centres",
                          "laying": "laid from marker to marker without tension or in-plane steering"}},
}
COVERAGE_K = 2.0
PAIR_U = math.sqrt(2.0) * INSTRUMENTS["camera"]["declared_standard_uncertainty_mm"]
# A measured chord-geodesic gap: film surface distance minus camera chord.
GAP_U = math.hypot(INSTRUMENTS["film"]["declared_standard_uncertainty_mm"], PAIR_U)


# Common builders ---------------------------------------------------------------
def _check(kind, reference, observed, tolerance, comparison="abs_le"):
    observed, tolerance = float(observed), float(tolerance)
    if comparison == "le" and observed < 0:
        # le bounds a nonnegative magnitude; a negative value is a sign mistake, not a pass.
        raise ValueError(f"le check '{reference}' received a negative observed value {observed!r}")
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": bool(holds(observed, tolerance, comparison))}


def _refusal(reference, expected, observed):
    observed = observed or "none"
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _u(kind, value, basis):
    """Per-finding uncertainty: kind is truncation_bound, monte_carlo_95ci, roundoff or reference_error."""
    return {"kind": kind, "value": value, "basis": basis}


EXACT = {"kind": "roundoff", "value": 0.0, "basis": "exact logic or integer count"}


def _generator(name, **extra):
    return dict({"name": name, "seed": SEED}, **extra)


def _r(value, digits=12):
    """Round nested floats for retained tables (findings keep full precision)."""
    if isinstance(value, dict):
        return {k: _r(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_r(v, digits) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if value == 0 or not math.isfinite(value) else float(f"{value:.{digits}g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


def _fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failures,
            assumptions, next_task):
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


def _task(task_id, tests, extra_files=()):
    """Register a section task whose findings pass the acceptance-language screen before the runner sees them."""
    tests = (*tests, "test_every_task_is_registered_and_reports_honestly")
    register = task(task_id, changed_files=(MODULE, GEOMETRY, METROLOGY, RECORDS, DOC, *extra_files),
                    regression_tests=tuple(f"{TESTS}::{name}" for name in tests))

    def decorate(function):
        @functools.wraps(function)
        def screened(ctx):
            outcome = function(ctx)
            # A refusal here blocks the task: an acceptance decision cannot be filed in a computational domain.
            rec.screen_acceptance_language(outcome.get("findings", []))
            return outcome
        return register(screened)
    return decorate


def _not_measured(claim, domain="physical"):
    return finding(claim, domain, None, {})


# Measurement protocols -----------------------------------------------------------
FRAME_CHAIN = [
    {"parent": "INSTRUMENT", "child": "WORLD", "source": "instrument registration to the reference network"},
    {"parent": "WORLD", "child": "FIXTURE", "source": "fixture datum targets measured by the tracker"},
    {"parent": "FIXTURE", "child": "PART", "source": "3-2-1 datum frame from probed A/B/C features"},
    {"parent": "PART", "child": "CAD", "source": "nominal model placement in the datum frame (declared)"},
]
AXIS_FRAME_SOURCE = "axis-primary datum frame: cylinder-fit axis A, axial scribe B, end face C"
# Layout of what sits on a specimen (mm). Tapes are 3 mm wide; coded targets and specimen markers are 6 mm. An
# offset tape starts 2 mm from the nominal one, and on the coupon every lateral offset crosses the nominal tape
# at the focus, so no two tapes of a protocol can lie on the specimen together: each is laid from its own jig
# insert, measured and removed before the next, all in the PART frame of the specimen's own datums.
TAPE_WIDTH_MM = 3.0
TARGET_DIAMETER_MM = 6.0
MARKER_DIAMETER_MM = 6.0
TAPE_LAYOUT = {"tape_width_mm": TAPE_WIDTH_MM, "target_diameter_mm": TARGET_DIAMETER_MM,
               "marker_diameter_mm": MARKER_DIAMETER_MM,
               "sequence": "one tape on the specimen at a time: seat the start jig with that tape's insert, lay the "
                           "tape, remove the jig, measure the tape and its targets (with re-seated replicates) and "
                           "remove them before the next tape is laid; every tape is measured in the PART frame of the "
                           "specimen's own datums",
               "rules": "tapes on the specimen together: centrelines at least a tape width plus a target diameter "
                        "apart; slots used in one laying: at least one slot width apart; every tape centreline clears "
                        "each specimen marker and datum probe point by half the wider of tape and target plus half a "
                        "marker (manufacturing_records._tape_layout, on the predicted centrelines)"}
# Datum schemes. A plate or a formed coupon rests on its back face (3-2-1); a tube has no face to rest on,
# so its primary datum is the axis of a fitted cylinder and the frame is built axis first.
DATUMS = [
    {"id": "A", "role": "primary", "feature": "specimen back face", "points": 3,
     "constrains": "z translation, rotations about x and y"},
    {"id": "B", "role": "secondary", "feature": "long reference edge", "points": 2,
     "constrains": "y translation, rotation about z"},
    {"id": "C", "role": "tertiary", "feature": "end stop face", "points": 1, "constrains": "x translation"},
    {"id": "PART", "role": "datum reference frame", "feature": "3-2-1 construction",
     "definition": "z normal of plane A; x = B direction projected into A; origin on A, B plane and C plane "
                   "(ciw.lab.manufacturing_metrology.datum_frame_321)"},
]
# The tube lies in VB-01 with scribe B straight up; the vees sit between the marker rings and touch the tube 45 deg
# either side of the bottom, where the seam weld faces down. Only the upper half (within 90 deg of B) is
# accessible to the camera, tracker and CMM. Datum A is probed between markers on that half, at angles clear of
# both helix tapes (which start 72 deg from B on end face C and cross the rings between markers).
CYLINDER_RINGS_MM = (50.0, 150.0, 250.0)
ACCESSIBLE_ARC_DEG = 90.0
DATUM_A_ANGLES_DEG = (-75.0, -15.0, 45.0, 75.0)
SCRIBE_POINTS_MM = (40.0, 260.0)
HELIX_START_DEG = -72.0
ROUNDNESS_TOLERANCE_MM = 0.1


def axis_probe_points() -> dict:
    """Chart points (phi rad, z mm) the CMM probes for datum A (four per ring, between markers) and B (scribe)."""
    return {"A": [(math.radians(angle), z) for z in CYLINDER_RINGS_MM for angle in DATUM_A_ANGLES_DEG],
            "B": [(0.0, z) for z in SCRIBE_POINTS_MM]}


def _cylinder_points(chart) -> list:
    return [[round(float(v), 6) for v in geo.CYLINDER.embedding(np.asarray(u, dtype=float))] for u in chart]


AXIS_DATUMS = [
    {"id": "A", "role": "primary", "feature": "tube axis: least-squares cylinder through CMM points on the three marker "
     "rings (z = 50, 150, 250 mm) at phi = -75, -15, 45 and 75 deg from scribe B: between markers, on the upper half "
     "that faces away from the vees, clear of the tapes", "points": len(axis_probe_points()["A"]),
     "constrains": "x and y translation, rotations about x and y",
     "construction": "ciw.lab.manufacturing_metrology.fit_cylinder",
     "probe_points_mm": _cylinder_points(axis_probe_points()["A"])},
    {"id": "B", "role": "secondary", "feature": "axial scribe line at phi = 0, probed at z = 40 and 260 mm", "points": 2,
     "constrains": "rotation about z", "probe_points_mm": _cylinder_points(axis_probe_points()["B"])},
    {"id": "C", "role": "tertiary", "feature": "end face at z = 0", "points": 3, "constrains": "z translation"},
    {"id": "PART", "role": "datum reference frame", "feature": "axis-primary construction",
     "definition": "z along the fitted axis A, pointing from end face C into the tube; origin where A meets plane C; "
                   "x from the axis towards scribe B, perpendicular to z "
                   "(ciw.lab.manufacturing_metrology.datum_frame_axis)"},
]


def _datum_targets(prefix, positions, frame) -> list:
    return [{"id": f"{prefix}-T{k}", "position_mm": list(position), "frame": frame,
             "nest": "1.5 in SMR nest", "adapters": ["1.5 in SMR (tracker)",
                                                     "coded photogrammetry target adapter with the same centre (camera)"]}
            for k, position in enumerate(positions, start=1)]


NEST = {"id": "FX-321", "kind": "3-2-1 kinematic nest", "locates": ["A", "B", "C"],
        "contacts": "three 12 mm spherical rests on A, two cylindrical side stops on B, one end stop on C",
        "geometry": "rests at 10% and 90% of the specimen length under A; side stops 200 mm apart along B; a 400 x 400 mm "
                    "base plate",
        "clamping": "normal to A, over the rests, torque declared per specimen",
        "datum_targets": _datum_targets("FX-321", [(-200.0, -200.0, 0.0), (200.0, -200.0, 0.0), (200.0, 200.0, 0.0),
                                                   (-200.0, 200.0, 0.0)],
                                        "FIXTURE: base plate centre, z up (declared)")}
V_BLOCKS = {"id": "VB-01", "kind": "V-block pair (90 deg vees) with an end stop", "locates": ["A", "C"],
            "contacts": "two 90 deg vees give four line contacts on the tube (A), 45 deg either side of the bottom; an end "
                        "stop touches the end face (C); the tube is turned until the axial scribe (B) faces straight up, "
                        "away from the vees, which puts the seam weld at the bottom between the vee contact lines",
            "geometry": "vee centres 100 mm and 200 mm from the end stop, between the marker rings at 50, 150 and 250 mm, "
                        "so no ring rests in a vee; the upper half of the tube (within 90 deg of scribe B) stays open to "
                        "the camera, tracker and CMM; a 360 x 200 mm base plate",
            "clamping": "one strap over each vee, torque declared per specimen",
            "datum_targets": _datum_targets("VB-01", [(-30.0, -100.0, 0.0), (330.0, -100.0, 0.0), (330.0, 100.0, 0.0),
                                                      (-30.0, 100.0, 0.0)],
                                            "FIXTURE: x along the tube axis from the end stop, z up (declared)")}
REQUIRED_RAW = [
    "raw camera images (lossless) with exposure metadata and SHA-256 digests",
    "tracker/CMM native point files with instrument serial and firmware",
    "calibration certificates of scale bars, gauge sphere and step gauge (digest and validity window)",
    "frame-chain transforms with covariances as computed at the time of measurement",
    "environment log: air and part temperature, humidity, time-stamped with an explicit UTC offset",
    "operator, fixture and specimen identities; clock source and synchronization method",
]
CALIBRATION_ARTIFACTS = [
    {"id": "SB-1000", "kind": "carbon scale bar", "nominal_mm": 1000.0, "use": "photogrammetry scale"},
    {"id": "GS-25.4", "kind": "ceramic gauge sphere", "nominal_diameter_mm": 25.4, "use": "probe and tracker SMR check"},
    {"id": "SG-200", "kind": "step gauge", "steps_mm": [10.0 * k for k in range(1, 21)], "use": "scale error"},
]
ENVIRONMENT = {"temperature_C": "20 +/- 1 (declared)", "soak_time_h": 4,
               "thermal_expansion_note": "aluminium 23e-6/K: 1 K over 300 mm is 7 um; log part temperature"}
# Paths are realized physically as geodesics, not drawn from the model: a robot tracing the
# computed offset path would reproduce its own program, and the comparison would test only
# the robot. The realized start pose is measured and the prediction conditioned on it.
START_JIG = "JIG-START-01"
PATH_REALIZATION = ("centreline of a 3 mm unsteered adhesive tape laid from the slot of its own insert of the start jig "
                    "JIG-START-01, which sets the start point and heading; with no in-plane steering the tape follows a "
                    "geodesic")
START_POSE_PROCEDURE = ("Probe each tape centreline with the CMM at s = 0 and s = 20 mm; the relative start offset and "
                        "heading of the offset tape condition the prediction (T138).")
# Declared relative start-pose error of an offset tape: its insert and the laying, and the re-seating of the jig
# (each tape is laid after its own seating, so the relative pose carries two seatings). The CMM estimate of the
# pose has offset sqrt(2) u_cmm and heading 2 u_cmm / 20 mm (difference of two headings).
EXECUTION_INSERT = {"lateral_mm": 0.05, "heading_rad": 5e-4, "status": "declared_not_verified",
                    "source": "slot insert and unsteered tape laying, relative pose of the offset tape"}
JIG_RESEAT = {"lateral_mm": 0.01, "heading_rad": 1e-4, "status": "declared_not_verified",
              "source": "re-seating the start jig against datums B and C, 1 sigma per seating"}
RESEAT_RELATIVE = {"lateral_mm": math.sqrt(2.0) * JIG_RESEAT["lateral_mm"],
                   "heading_rad": math.sqrt(2.0) * JIG_RESEAT["heading_rad"]}
EXECUTION = {"lateral_mm": math.hypot(EXECUTION_INSERT["lateral_mm"], RESEAT_RELATIVE["lateral_mm"]),
             "heading_rad": math.hypot(EXECUTION_INSERT["heading_rad"], RESEAT_RELATIVE["heading_rad"]),
             "status": "declared_not_verified",
             "source": "relative start pose of the offset tape: insert and laying (0.05 mm, 0.5 mrad) and two seatings "
                       "of the start jig (0.01 mm, 0.1 mrad each)",
             "terms": {"insert_and_laying": {k: EXECUTION_INSERT[k] for k in ("lateral_mm", "heading_rad")},
                       "jig_reseating": dict(RESEAT_RELATIVE)}}
START_POSE_U = {"lateral_mm": math.sqrt(2.0) * INSTRUMENTS["cmm"]["declared_standard_uncertainty_mm"],
                "heading_rad": 2.0 * INSTRUMENTS["cmm"]["declared_standard_uncertainty_mm"] / 20.0,
                "source": "CMM centreline points at s = 0 and 20 mm on both tapes"}
EXECUTION_CLAIM = ("The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and "
                   "0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig")
TARGET_CAPTURE = "ciw.lab-mfg-target-capture.v1"
DISTANCE_CAPTURE = "ciw.lab-mfg-distance-capture.v1"


def start_jig(paths, contacts) -> dict:
    """The start jig of a tape protocol: one single-slot insert per path, offset from the nominal slot as declared.

    One tape is laid per seating: the jig is seated against datums B and C
    with the insert of that tape, the tape laid and the jig removed. It is not
    verified before laying: the start-pose step probes every laid tape, and
    the prediction is budgeted open loop (insert, laying and re-seating terms)
    or conditioned on that pose.
    """
    base = next(path for path in paths if "realization" in path)
    slots = [{"slot": base["id"], "insert": f"insert {base['id']}", "lateral_offset_mm": 0.0, "heading_offset_rad": 0.0}]
    slots += [{"slot": path["id"], "insert": f"insert {path['id']}", "lateral_offset_mm": path.get("lateral_offset_mm", 0.0),
               "heading_offset_rad": path.get("heading_offset_rad", 0.0)} for path in paths if path.get("of") == base["id"]]
    return {"id": START_JIG, "kind": "tape start jig with one single-slot insert per tape", "locates": ["B", "C"],
            "contacts": contacts,
            "geometry": {"slots": slots, "slot_width_mm": 3.05, "slot_length_mm": 20.0, "slots_per_insert": 1,
                         "offsets": "each insert's slot centreline is offset laterally and rotated in heading from the "
                                    "nominal slot at the slot exit, where the tape leaves the jig"},
            "use": "one tape per seating: seat the jig against B and C with the insert of the tape to be laid, lay that "
                   "tape, remove the jig; re-seat it for the next tape after the previous one is removed",
            "declared_realization_tolerance": {"lateral_mm": EXECUTION_INSERT["lateral_mm"],
                                               "heading_rad": EXECUTION_INSERT["heading_rad"],
                                               "level": "1 sigma, relative pose of an offset insert's tape (insert and "
                                                        "laying)",
                                               "status": "declared_not_verified"},
            "declared_reseat_repeatability": {"lateral_mm": JIG_RESEAT["lateral_mm"],
                                              "heading_rad": JIG_RESEAT["heading_rad"],
                                              "level": "1 sigma per seating against B and C; two seatings enter the "
                                                       "relative pose of an offset tape",
                                              "status": "declared_not_verified"},
            "verification": "no pre-laying check: the start-pose step probes each laid tape with the CMM at s = 0 and "
                            "20 mm, and the prediction is budgeted open loop or conditioned on that pose (T138, T140)"}


def tape_centreline(surface, start, heading, length, lateral=0.0, dheading=0.0, spacing=5.0) -> list:
    """Predicted 3D centreline of a tape (exact start perturbation, RK4 at about 1 mm), sampled every ``spacing`` mm.

    Used by the protocol's layout checks: chords between these points never
    exceed surface distances, so a clearance computed from them is conservative.
    """
    samples = max(1, round(length / spacing))
    y0 = jacobi.perturbed_start(surface, start, heading, lateral=lateral, heading_change=dheading)
    states = integrators.integrate_fixed(surface.geodesic_rhs, y0, length, 5 * samples, "rk4")[1][::5]
    return [[round(float(v), 4) for v in surface.embedding(y[:2])] for y in states]


def capture_format(role, schema, instrument, ids, reader) -> dict:
    """A raw format an operator capture of ``role`` must follow (manufacturing_records.read_capture)."""
    return {"role": role, "schema": schema, "media_type": "text/csv", "instrument": instrument,
            "frame": "CAD" if schema == TARGET_CAPTURE else "surface",
            "header": "lines '# key: value' for " + ", ".join(rec.CAPTURE_HEADER) + "; unit mm; origin measurement "
                      "or synthetic (a synthetic capture starts with '" + rec.SYNTHETIC_CAPTURE_PREFIX.decode() + "')",
            "columns": list(rec.CAPTURE_SCHEMAS[schema]), "ids": list(ids), "reader": reader,
            "coordinates": "CAD (model) frame through the declared PART -> CAD placement" if schema == TARGET_CAPTURE
                           else "surface distance between the two marker centres of the pair",
            "uncertainty": "u_mm is the declared standard uncertainty of each row (positive); for targets, of each "
                           "coordinate in the common frame, including the registration of its image set"}


def _step(action, uses, outputs, datums=(), check=None, key=None, lays=(), places=(), removes=()) -> dict:
    """A procedure step; ``key`` lets a repeat step refer to it before steps are numbered."""
    record = {"action": action, "uses": list(uses), "outputs": list(outputs)}
    if datums:
        record["datums"] = list(datums)
    if check:
        record["check"] = check
    for name, value in (("lays", lays), ("places", places), ("removes", removes)):
        if value:
            record[name] = list(value)
    if key:
        record["_key"] = key
    return record


def _repeat(keys, holder, what) -> dict:
    """A repeat of earlier measurement steps (by key), numbered and worded when the protocol is assembled."""
    return {"_repeat": list(keys), "_what": what, "uses": [holder], "outputs": [f"two further replicates ({what})"]}


def _listing(numbers) -> str:
    text = [str(n) for n in numbers]
    return text[0] if len(text) == 1 else ", ".join(text[:-1]) + " and " + text[-1]


CHECK_TEXT = {"camera": "the camera images SB-1000", "tracker": "the tracker measures the SB-1000 end points with its SMR",
              "cmm": "the CMM probes GS-25.4 at 25 points", "scanner": "the scanner scans GS-25.4",
              "film": "the film is read against the SG-200 steps at 50, 100 and 150 mm"}


def _verification_step(instruments, when) -> dict:
    uses = sorted({"GS-25.4", "SB-1000"} | ({"SG-200"} if "film" in instruments else set()))
    return _step(f"Check the instruments {when} the acquisition: " + "; ".join(CHECK_TEXT[key] for key in instruments)
                 + ". Retain every check.", [*instruments, *uses], [f"instrument checks {when} the acquisition"],
                 check="each artifact reading lies within 2 U of its certificate value, U from the declared instrument "
                       "uncertainty; otherwise stop and recalibrate")


def roundness_threshold() -> float:
    """Largest recorded ring roundness the cylinder model accepts: form tolerance plus 2 U (k = 2) of the CMM."""
    return ROUNDNESS_TOLERANCE_MM + 2.0 * COVERAGE_K * INSTRUMENTS["cmm"]["declared_standard_uncertainty_mm"]


def locate_step(holder) -> dict:
    if holder["id"] == V_BLOCKS["id"]:
        angles = ", ".join(f"{a:g}" for a in DATUM_A_ANGLES_DEG)
        return _step(f"Lay the tube in VB-01 (V-block pair) with the end face against the end stop and the axial scribe "
                     f"B facing straight up, away from the vees; clamp; probe datum A with the CMM at phi = {angles} deg "
                     f"from scribe B on each marker ring (between markers), scribe B at z = 40 and 260 mm and end face "
                     "C at 3 points; fit the axis A (fit_cylinder), record the fitted radius and the roundness of each "
                     "ring over the probed arc (peak to valley of the radial residuals), and build the PART frame "
                     "(datum_frame_axis).",
                     ["VB-01", "cmm"], ["datum point file", "fitted axis and radius", "roundness of each ring",
                                        "PART frame"], ("A", "B", "C", "PART"),
                     check=f"every ring's roundness is at most the declared form tolerance {ROUNDNESS_TOLERANCE_MM:g} mm "
                           f"plus 2 U of the CMM ({roundness_threshold():.3f} mm); otherwise stop and record the tube as "
                           "outside the cylinder model (seam weld and out-of-roundness are not modelled); re-seating "
                           "does not change the form", key="locate")
    return _step("Locate the specimen in FX-321 (3-2-1 nest); clamp; probe datum A (3 points), B (2 points) and C "
                 "(1 point) with the CMM and build the PART frame (datum_frame_321).", ["FX-321", "cmm"],
                 ["datum point file", "PART frame"], ("A", "B", "C", "PART"), key="locate")


def build_protocol(task_id, protocol_id, title, purpose, specimen, markers, paths, predicted, criteria,
                   instruments, procedure, raw_formats, holder=NEST, extra=None) -> dict:
    """A validated ciw.lab-measurement-protocol.v1 record; ``procedure`` holds the protocol's own steps.

    Common steps surround them: soak, instrument checks before, locating the
    specimen in its holder and building the PART frame (key ``locate``) and,
    after them, instrument checks and retention. Repeat steps in ``procedure``
    name earlier steps by key and are numbered and worded here. The start jig
    is declared as a fixture, and the tape layout recorded, whenever a path
    is a tape.
    """
    axis = holder["id"] == V_BLOCKS["id"]
    fixtures = [deepcopy(holder)]
    tapes = any(path.get("kind") == "tape" for path in paths)
    if tapes:
        fixtures.append(start_jig(paths, "a vee foot on the tube, a pointer on the axial scribe B and a stop against "
                                         "end face C" if axis else
                                  "two dowel pins against datum B and a stop face against datum C"))
    steps = [_step("Soak the specimen and its holder for 4 h at 20 +/- 1 C; log air and part temperature every "
                   "10 min.", [holder["id"]], ["environment log with an explicit UTC offset"]),
             _verification_step(instruments, "before"), locate_step(holder), *deepcopy(procedure),
             _verification_step(instruments, "after"),
             _step("Retain every raw file, replicate and instrument check with its T139 retention record "
                   "(ciw.lab-measurement-retention.v1).", [], ["retained raw data and retention records"])]
    numbers = {step["_key"]: index for index, step in enumerate(steps, start=1) if "_key" in step}
    numbered = []
    for index, step in enumerate(steps, start=1):
        step.pop("_key", None)
        if "_repeat" in step:
            repeats = [numbers[key] for key in step.pop("_repeat")]
            what = step.pop("_what")
            # A repeat uses the holder it re-seats the specimen in and every device of the steps it repeats.
            uses = list(dict.fromkeys([*step["uses"], *(name for n in repeats for name in steps[n - 1]["uses"])]))
            step = {"action": f"Repeat steps {_listing(repeats)} twice more, removing the specimen from {holder['id']} "
                              f"and re-seating it between repeats ({what}).", "repeats": repeats,
                    **dict(step, uses=uses)}
        numbered.append(dict(step=index, **step))
    chain = deepcopy(FRAME_CHAIN)
    if axis:
        chain[2]["source"] = AXIS_FRAME_SOURCE
    record = {"schema": rec.PROTOCOL_SCHEMA, "protocol_id": protocol_id, "task_id": task_id, "title": title,
              "purpose": purpose, "specimen": specimen, "fixtures": fixtures,
              "datum_frames": deepcopy(AXIS_DATUMS if axis else DATUMS), "frame_chain": chain, "markers": markers,
              "paths": paths, "instruments": [deepcopy(INSTRUMENTS[key]) for key in instruments],
              "required_raw_data": list(REQUIRED_RAW), "raw_formats": raw_formats,
              "calibration_artifacts": deepcopy(CALIBRATION_ARTIFACTS), "environment": dict(ENVIRONMENT),
              "procedure": numbered, "predicted_quantities": predicted, "acceptance_criteria": criteria,
              "hardware_measured": {"status": "not_acquired", "records": []},
              "production_acceptance": "outside_system"}
    if tapes:
        record["tape_layout"] = deepcopy(TAPE_LAYOUT)
    record.update(deepcopy(extra or {}))
    return rec.validate_protocol(record)


def tape_paths(base, offsets, surface, start) -> list:
    """The nominal tape path and its offset paths, each laid from its own jig insert, with predicted centrelines."""
    length, heading = base["length_mm"], base["heading_rad"]
    paths = [dict(base, kind="tape", realization=PATH_REALIZATION, uses=[START_JIG],
                  centreline_mm=tape_centreline(surface, start, heading, length))]
    return paths + [dict(offset, kind="tape", of=base["id"], uses=[START_JIG],
                         centreline_mm=tape_centreline(surface, start, heading, length,
                                                       offset.get("lateral_offset_mm", 0.0),
                                                       offset.get("heading_offset_rad", 0.0)))
                    for offset in offsets]


def start_pose_ids(paths) -> list:
    """Row identifiers of the start-pose capture: each tape centreline at s = 0 and s = 20 mm."""
    return [f"{path['id']}-S{s}" for path in paths for s in (0, 20)]


def tape_block(path, holder, places, rows=None) -> list:
    """Steps for one tape: lay it alone, probe its start pose, place and image its targets, replicate, remove it."""
    name = path["id"]
    export = (f"export their coordinates as rows {rows} of the photogrammetry capture" if rows
              else "retain their coordinates (no task reads them yet)")
    return [
        _step(f"Seat the start jig {START_JIG} against datums B and C with the insert of tape {name}; lay tape {name} "
              "from its slot without in-plane steering (a robot may carry the tape head but must not steer it; never "
              "draw a path traced from the model); remove the jig. No other tape is on the specimen.", [START_JIG],
              [f"tape {name}: identity and laying time"], ("B", "C"), key=f"lay {name}", lays=[name]),
        _step(f"Probe the centreline of tape {name} with the CMM at s = 0 and s = 20 mm; export the points as rows "
              f"{name}-S0 and {name}-S20 of the cmm capture.", ["cmm"],
              [f"cmm capture rows {name}-S0 and {name}-S20 (ciw.lab-mfg-target-capture.v1)"], key=f"pose {name}"),
        _step(f"Place 6 mm coded targets on the centreline of tape {name} {places}.", [], [f"targets on tape {name}"],
              places=[name]),
        _step(f"Measure the targets on tape {name} with the camera, with SB-1000 and the {holder} datum targets "
              f"(coded adapters) in every image set; {export}.", ["camera", "SB-1000", holder],
              ["raw images", f"target coordinates of tape {name}"], key=f"image {name}"),
        _repeat(["locate", f"pose {name}", f"image {name}"], holder,
                f"replicates of tape {name}; the tape and its targets stay in place"),
        _step(f"Remove the targets and tape {name}.", [], [f"tape {name} removed"], removes=[name])]


def register_step(holder) -> dict:
    return _step(f"Measure the four datum targets of {holder['id']} (a 1.5 in SMR seated on each) with the tracker to "
                 "register FIXTURE in WORLD (frame chain).", [holder["id"], "tracker"],
                 ["WORLD -> FIXTURE transform and its covariance"])


def _predicted(identifier, quantity, value, unit, record):
    return {"id": identifier, "quantity": quantity, "value": value, "unit": unit,
            "evidence_status": record["evidence_status"], "finding_claim": record["claim"]}


def _protocol_findings(protocol, claim):
    matrix = rec.protocol_refusal_matrix(protocol)
    checks = [_refusal(f"protocol mutation {name}", case["expected"], case["observed"])
              for name, case in sorted(matrix.items())]
    matched = sum(case["observed"] == case["expected"] for case in matrix.values())
    return finding(claim, "computational_pipeline", matched, {"checks": checks}, unit="refused mutations",
                   uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})


# T126 flat plate ---------------------------------------------------------------------
PLATE_PITCH = 60.0
PLATE_TAPE_START = (-120.0, 30.0)   # between the marker rows y = 0 and y = 60
# Shooting starts this far off the chord direction, so the heading is solved for, not assumed.
SHOOTING_OFFSET_RAD = 0.1


def shoot_geodesic(surface, a, b, heading, horizon, steps=8, iterations=20) -> dict:
    """Heading and arclength of the geodesic from ``a`` through ``b`` by Newton on the miss distance (plane).

    For a launch heading the geodesic and its Jacobi fields are integrated to
    ``horizon``; s* is where the along-track coordinate (along the embedded
    launch direction, which is constant on a plane) passes ``b`` and the miss m
    is the signed offset of the path point at s* from ``b`` along the path
    normal. The heading Jacobi field gives dm/dheading = j_head(s*), so the
    Newton step is -m / j_head(s*). Valid for zero-curvature charts, where the
    along-track direction does not turn.
    """
    a = np.asarray(a, dtype=float)
    target = surface.embedding(np.asarray(b, dtype=float))
    for count in range(1, iterations + 1):
        path = jacobi.transfer(surface, a, heading, horizon, steps=steps)
        direction = surface.embedding_jacobian(a) @ path.velocities[0]
        along = np.array([(surface.embedding(u) - target) @ direction for u in path.points])
        along_rate = np.array([(surface.embedding_jacobian(u) @ v) @ direction for u, v in zip(path.points, path.velocities)])
        arclength = integrators.hermite_zeros(path.s, along, along_rate)[0]
        hit = jacobi.transfer(surface, a, heading, arclength, steps=steps)
        u, v = hit.points[-1], hit.velocities[-1]
        miss = float((surface.embedding(u) - target) @ (surface.embedding_jacobian(u) @ surface.normal(u, v)))
        step = -miss / hit.states[-1, 6]
        heading += step
        if abs(step) < 1e-12:  # Newton is cubic here (the step is -tan of the error): the next error is ~1e-16
            break
    closure = float(np.linalg.norm(surface.embedding(hit.points[-1]) - target))
    return {"heading_rad": heading, "arclength_mm": arclength, "closure_mm": closure, "iterations": count}


@functools.lru_cache(maxsize=1)
def plate_study() -> dict:
    coords = [-120.0, -60.0, 0.0, 60.0, 120.0]
    markers = [np.array([x, y]) for y in coords for x in coords]
    pairs = [(0, j) for j in range(1, 25)] + [(12, j) for j in range(25) if j not in (0, 12)]
    rows, max_gap, max_closure, max_heading = [], 0.0, 0.0, 0.0
    # The geodesic distance comes from shooting: Newton on the miss distance from a launch heading
    # 0.1 rad off the chord direction, over a fixed horizon half as long again as the plate diagonal.
    # On the plane this is a zero-curvature sanity check of the shooting machinery (RK4 and the
    # Jacobi heading field are exact there); the chord formula enters only the comparison.
    horizon = 1.5 * 2.0 * math.sqrt(2.0) * 120.0
    for i, j in pairs:
        a, b = markers[i], markers[j]
        delta = b - a
        chord_heading = math.atan2(delta[1], delta[0])
        shot = shoot_geodesic(geo.PLATE, a, b, chord_heading + SHOOTING_OFFSET_RAD, horizon)
        geodesic, closure = shot["arclength_mm"], shot["closure_mm"]
        chord = float(np.linalg.norm(geo.PLATE.embedding(b) - geo.PLATE.embedding(a)))
        heading_error = abs(math.remainder(shot["heading_rad"] - chord_heading, 2.0 * math.pi))
        max_gap, max_closure = max(max_gap, abs(geodesic - chord)), max(max_closure, closure)
        max_heading = max(max_heading, heading_error)
        rows.append({"pair": [f"M{i:02d}", f"M{j:02d}"], "geodesic_mm": geodesic, "chord_mm": chord,
                     "rk4_closure_mm": closure, "shooting_iterations": shot["iterations"],
                     "heading_minus_chord_direction_rad": heading_error})
    # The tape paths run along y = 30, midway between two marker rows (the plane is homogeneous, so the
    # prediction does not depend on where they run).
    control = jacobi.transfer(geo.PLATE, list(PLATE_TAPE_START), 0.0, 240.0, steps=24)
    phi = control.matrix()
    phi_error = float(np.max(np.abs(phi - np.array([[1.0, 240.0], [0.0, 1.0]]))))
    stations = control.s[::6]
    lateral, dheading = 2.0, 0.005
    nonlinear = geo.separation_nonlinear(geo.PLATE, list(PLATE_TAPE_START), 0.0, 240.0, 24, lateral, dheading,
                                         base=control)
    linear = geo.separation_linear(control, lateral, dheading)
    # On the plane the exactly rotated path separates as delta + s sin(dtheta).
    exact = lateral + control.s * math.sin(dheading)
    return {"markers": [{"id": f"M{k:02d}", "u_mm": m.tolist()} for k, m in enumerate(markers)], "pairs": rows,
            "max_gap_mm": max_gap, "max_closure_mm": max_closure, "max_heading_error_rad": max_heading,
            "phi_end": phi.tolist(), "phi_error": phi_error,
            "det_drift": float(np.max(np.abs(control.determinant() - 1.0))), "stations_mm": stations.tolist(),
            "lateral_separation_mm": (lateral * control.states[::6, 4]).tolist(),
            "heading_separation_mm": (dheading * control.states[::6, 6]).tolist(),
            "nonlinear_minus_exact_mm": float(np.max(np.abs(nonlinear - exact))),
            "nonlinear_minus_linear_mm": float(np.max(np.abs(nonlinear - linear)))}


@_task("T126", ("test_flat_plate_protocol_is_a_zero_curvature_control", "test_protocols_refuse_filled_slots_and_decisions"))
def flat_plate_control(ctx):
    study = ctx.memo("mfg.plate", plate_study)
    f_gap = finding("Flat-plate control: chord and geodesic marker distances coincide", "numerical",
                    study["max_gap_mm"],
                    {"derivation": "Plane geodesics are straight segments, so chord = geodesic (docs/lab/MANUFACTURING.md#t126)",
                     "checks": [_check("analytic", "max |shooting geodesic arclength - chord| over 47 marker pairs, heading "
                                       "solved by Newton from 0.1 rad off the chord direction (mm)",
                                       study["max_gap_mm"], 1e-9),
                                _check("analytic", "RK4 geodesic of that arclength closes onto the target marker (mm)",
                                       study["max_closure_mm"], 1e-9),
                                _check("analytic", "max |solved launch heading - chord direction| (rad)",
                                       study["max_heading_error_rad"], 1e-12)]},
                    unit="mm",
                    uncertainty=_u("roundoff", max(study["max_gap_mm"], study["max_closure_mm"]),
                                   "shooting arclength vs chord and RK4 closure onto the target marker (mm)"),
                    tolerance={"abs": 1e-9, "rel": 0})
    f_phi = finding("Flat-plate Jacobi transfer is [[1, s], [0, 1]] along the 240 mm control path", "numerical",
                    study["phi_end"],
                    {"checks": [_check("analytic", "max |Phi(240) - [[1, 240], [0, 1]]|", study["phi_error"], 1e-9),
                                _check("invariant", "Wronskian det Phi - 1", study["det_drift"], 1e-12),
                                _check("analytic", "exactly offset path minus delta + s sin(dtheta) (mm)",
                                       study["nonlinear_minus_exact_mm"], 1e-9)]},
                    uncertainty=_u("roundoff", max(study["phi_error"], study["det_drift"]),
                                   "deviation of Phi from the closed form"),
                    tolerance={"abs": 1e-9, "rel": 1e-12})
    predicted = [
        _predicted("P1", "max |chord - geodesic| over the marker pairs", 0.0, "mm", f_gap),
        _predicted("P2", "separation of a 2 mm laterally offset path at stations 0..240 mm",
                   _r(study["lateral_separation_mm"]), "mm", f_phi),
        _predicted("P3", "separation of a 5 mrad heading-offset path at stations 0..240 mm",
                   _r(study["heading_separation_mm"]), "mm", f_phi)]
    criteria = [
        {"id": "H1", "status": "hypothesis", "statement": "Measured chord equals predicted geodesic distance for every marker pair",
         "test": f"E_n = |m - p| / U <= 1 with U = k u_pair, k = 2, u_pair = {PAIR_U:.4f} mm (camera)"},
        {"id": "H2", "status": "hypothesis", "statement": "Offset-path separation is constant (lateral) and linear in s (heading)",
         "test": "slope of heading-offset separation equals the heading difference measured at the start (CMM) within "
                 "2 sigma of the regression slope"}]
    paths = tape_paths({"id": "N0", "start_u_mm": list(PLATE_TAPE_START), "heading_rad": 0.0, "length_mm": 240.0},
                       [{"id": "L2", "lateral_offset_mm": 2.0}, {"id": "H5", "heading_offset_rad": 0.005}],
                       geo.PLATE, PLATE_TAPE_START)
    markers = [dict(m, kind="specimen marker", position_mm=[*m["u_mm"], 0.0]) for m in study["markers"]]
    protocol = build_protocol(
        "T126", "MFG-FLAT-PLATE-01", "Flat-plate zero-curvature control",
        "Establish the measurement chain on a specimen where chord, geodesic and Jacobi predictions are trivial, "
        "so any residual belongs to the instruments, frames or procedure.",
        {"kind": "flat plate", "material": "6082-T6 aluminium (declared)", "nominal_mm": [300.0, 300.0, 6.0],
         "surface_model": geo.PLATE.describe(), "declared_flatness_mm": 0.05},
        markers, paths, predicted, criteria, ("camera", "tracker", "cmm"),
        [register_step(NEST),
         _step("Measure all 25 coded markers with the camera (12 stations, SB-1000 and the FX-321 datum targets in every "
               "image set) and probe their centres with the CMM; export the camera coordinates as the photogrammetry "
               "capture.", ["camera", "cmm", "SB-1000", "FX-321"],
               ["raw images", "photogrammetry capture (ciw.lab-mfg-target-capture.v1)", "CMM marker point file"],
               key="markers"),
         _repeat(["locate", "markers"], "FX-321", "replicates of the marker measurement"),
         *[step for path in paths for step in tape_block(path, "FX-321", "every 30 mm from s = 0 to 240 mm")]],
        [capture_format("photogrammetry", TARGET_CAPTURE, "camera", [m["id"] for m in study["markers"]],
                        "ciw.lab.manufacturing_records.read_capture; T138 compares the 47 marker-pair chords"),
         capture_format("cmm", TARGET_CAPTURE, "cmm", start_pose_ids(paths),
                        "ciw.lab.manufacturing_records.read_capture; retained by T138, not yet used to condition the "
                        "plate prediction")])
    ctx.artifact_json("protocol-flat-plate.json", protocol)
    ctx.artifact_json("plate-pairs.json", _r(study["pairs"]))
    ctx.artifact_text("plate-separation.svg", svg.line_plot(
        [("2 mm lateral offset", study["stations_mm"], study["lateral_separation_mm"]),
         ("5 mrad heading offset", study["stations_mm"], study["heading_separation_mm"])],
        title="Flat plate: predicted offset-path separation", xlabel="arclength s (mm)", ylabel="separation (mm)"))
    findings = [f_gap, f_phi,
                _protocol_findings(protocol, "Flat-plate protocol record validates and refuses malformed variants"),
                _not_measured("Measured marker chords on the physical plate equal the predicted geodesic distances "
                              "within instrument uncertainty"),
                _not_measured("The declared instrument uncertainties (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm) "
                              "hold for the instruments that will be used", "calibration")]
    fields = _fields(
        "On a flat plate the chord between markers equals their geodesic distance and the Jacobi transfer is "
        "[[1, s], [0, 1]], so the plate isolates instrument, frame and procedure error from curvature.",
        "Plane z = 0; geodesics are straight lines; j'' = 0 gives j_lat = 1, j_head = s; offset-path separation "
        "e(s) = delta + s dtheta.",
        ["Declared 300 x 300 x 6 mm plate, 5 x 5 marker grid at 60 mm pitch",
         "Offset perturbations: 2 mm lateral, 5 mrad heading", "Declared instruments (camera, tracker, CMM)"],
        "No observation. The protocol specifies camera/CMM observation of the markers, the tape start poses and the "
        "offset tapes; its hardware-evidence slot is empty.",
        "chord - geodesic = 0; det Phi = 1; separation of offset paths linear in s.",
        "For 47 marker pairs, find the geodesic distance by shooting (Newton on the miss distance with the Jacobi "
        "heading field, launched 0.1 rad off the chord direction; the arclength where the converged geodesic passes "
        "the other marker) and compare it with the chord; integrate the control path and its exact offset paths; "
        "assemble and validate the ciw.lab-measurement-protocol.v1 record (declared fixtures including the start jig, "
        "instrument settings, numbered steps naming only declared devices and datums, raw formats of the captures); "
        "mutate it and confirm each mutation is refused with its code.",
        f"max |chord - geodesic| = {study['max_gap_mm']:.3g} mm; max |Phi(240) - exact| = {study['phi_error']:.3g}; "
        f"heading-offset separation at 240 mm = {study['heading_separation_mm'][-1]:.4g} mm.",
        "Rounding only (flat metric, zero Christoffel symbols). Measurement uncertainty is declared, not known.",
        ["shooting from a wrong launch heading (solved by Newton)", "RK4 closure onto the target marker",
         "Wronskian drift", "nonlinear vs linear offset separation",
         "protocol with filled hardware slot but no acquisition", "acceptance criterion marked as a decision",
         "prediction labelled hardware_measured", "instrument uncertainty claimed verified",
         "procedure step or path naming a jig, instrument or artifact the protocol does not declare",
         "unnumbered procedure step, instrument without settings, raw format of an undeclared instrument",
         "offset tapes laid together or from overlapping jig slots, a tape over a marker, a repeat that lays a tape, "
         "datum targets of a fixture that declares none, a criterion naming an undeclared instrument (all refused)"],
        ["Real plate flatness (declared 0.05 mm) and thermal state are not measured; their effect is bounded "
         "only by the declared tolerance.",
         "Instrument uncertainties are declared planning values, not calibrated values.",
         "On the plane the chord-geodesic comparison is a zero-curvature sanity check of the shooting machinery: "
         "RK4 and the Jacobi heading field are exact there, so it cannot detect curvature errors."],
        "Acquire MFG-FLAT-PLATE-01 and bind its camera export to T138 (ciw lab run T138 --capture "
        "photogrammetry=<targets.csv>, format ciw.lab-mfg-target-capture.v1), then retain the run with ciw lab hardware "
        "retain; T138's pair comparison of the captured chords is computational, and a physical label needs a "
        "metrology instrument probe or a signed-capture trust anchor (deferred research question). Open: bound the "
        "plate flatness effect by measuring it rather than by the declared tolerance.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T127 rolled cylinder -------------------------------------------------------------------
def gap_exact(distance, radius):
    return distance - 2.0 * radius * math.sin(distance / (2.0 * radius))


def gap_radius_derivative(dphi, dz, radius):
    """d(gap)/dR at fixed marker angles and heights: gap = hypot(R dphi, dz) - hypot(2 R sin(dphi / 2), dz)."""
    geodesic, chord = math.hypot(radius * dphi, dz), math.hypot(2.0 * radius * math.sin(dphi / 2.0), dz)
    return radius * dphi ** 2 / geodesic - 4.0 * radius * math.sin(dphi / 2.0) ** 2 / chord


def resolvable_separation(radius, target_gap):
    """Smallest circumferential arc length whose chord-geodesic gap reaches ``target_gap`` (bisection)."""
    lo, hi = 0.0, math.pi * radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap_exact(mid, radius) < target_gap else (lo, mid)
    return 0.5 * (lo + hi)


def _ring_marker(u, rings) -> str:
    """Identifier of the cylinder marker at chart point (phi, z): ring index and 30 degree position."""
    return f"C{rings.index(float(u[1]))}-{round(math.degrees(float(u[0])) / 30.0) % 12:02d}"


def chord_radius_derivative(dphi, dz, radius):
    """d(chord)/dR at fixed marker angles and heights: chord = hypot(2 R sin(dphi / 2), dz)."""
    chord = math.hypot(2.0 * radius * math.sin(dphi / 2.0), dz)
    return 4.0 * radius * math.sin(dphi / 2.0) ** 2 / chord


@functools.lru_cache(maxsize=1)
def cylinder_study() -> dict:
    radius = geo.CYLINDER_RADIUS
    rings = CYLINDER_RINGS_MM
    markers = [{"id": f"C{r:d}-{k:02d}", "u": [math.radians(30.0 * k), z],
                "faces_up": abs(math.remainder(30.0 * k, 360.0)) <= ACCESSIBLE_ARC_DEG}
               for r, z in enumerate(rings) for k in range(12)]
    # Every pair uses markers on the upper half (within 90 deg of scribe B), which faces away from the vees:
    # circumferential pairs are placed about B, from -30 floor(d / 60) deg to d deg later.
    pair_specs = [((math.radians(-30 * (d // 60)), 150.0), (math.radians(d - 30 * (d // 60)), 150.0),
                   f"circumferential {d} deg") for d in (30, 60, 90, 120, 150, 180)]
    pair_specs += [((0.0, 50.0), (0.0, 150.0), "axial 100 mm"), ((0.0, 50.0), (0.0, 250.0), "axial 200 mm"),
                   ((0.0, 50.0), (math.radians(60), 150.0), "helical 60 deg / 100 mm"),
                   ((0.0, 50.0), (math.radians(90), 250.0), "helical 90 deg / 200 mm")]
    # max_series is the signed margin |series - gap| - bound (negative when the bound holds with room).
    rows, max_closure, max_series = [], 0.0, -math.inf
    for a, b, name in pair_specs:
        a, b = np.array(a), np.array(b)
        arc, dz = radius * (b[0] - a[0]), b[1] - a[1]
        geodesic = math.hypot(arc, dz)
        chord = float(np.linalg.norm(geo.CYLINDER.embedding(b) - geo.CYLINDER.embedding(a)))
        path = jacobi.transfer(geo.CYLINDER, a, math.atan2(dz, arc), geodesic, steps=16)
        closure = float(np.linalg.norm(geo.CYLINDER.embedding(path.points[-1]) - geo.CYLINDER.embedding(b)))
        row = {"pair": name, "markers": [_ring_marker(a, rings), _ring_marker(b, rings)], "geodesic_mm": geodesic,
               "chord_mm": chord, "gap_mm": geodesic - chord, "rk4_closure_mm": closure,
               "dphi_rad": float(b[0] - a[0]), "dz_mm": float(dz)}
        if dz == 0.0:
            d = arc
            series = d ** 3 / (24 * radius ** 2) - d ** 5 / (1920 * radius ** 4)
            bound = d ** 7 / (322560 * radius ** 6)
            row.update(series_mm=series, series_remainder_bound_mm=bound)
            max_series = max(max_series, abs(series - row["gap_mm"]) - bound)
        rows.append(row)
        max_closure = max(max_closure, closure)
    helix = jacobi.transfer(geo.CYLINDER, [math.radians(HELIX_START_DEG), 0.0], math.radians(45.0), 300.0, steps=30)
    flat_error = float(np.max(np.abs(helix.matrix() - np.array([[1.0, 300.0], [0.0, 1.0]]))))
    resolvable = {}
    for key in ("camera", "tracker", "cmm"):
        u = INSTRUMENTS[key]["declared_standard_uncertainty_mm"]
        target = COVERAGE_K * math.sqrt(2.0) * u
        exact = resolvable_separation(radius, target)
        resolvable[key] = {"target_gap_mm": target, "min_arc_mm": exact,
                           "series_estimate_mm": (24 * radius ** 2 * target) ** (1 / 3),
                           "min_angle_deg": math.degrees(exact / radius)}
    series_rel = max(abs(v["series_estimate_mm"] - v["min_arc_mm"]) / v["min_arc_mm"] for v in resolvable.values())
    return {"markers": markers, "pairs": rows, "max_closure_mm": max_closure, "series_excess_mm": max_series,
            "flat_transfer_error": flat_error, "resolvable": resolvable, "resolvable_series_rel": series_rel,
            "axial_gap_mm": max(abs(r["gap_mm"]) for r in rows if r["pair"].startswith("axial"))}


AXIS_POSE = ((0.1, -0.2, 0.3), (10.0, -20.0, 5.0))   # rotation vector (rad) and translation (mm) of the test tube


def axis_frame_study() -> dict:
    """The MFG-CYLINDER-01 datum scheme on noise-free synthetic CMM points of a tube in a known pose.

    The protocol's CMM points (four per marker ring between markers on the
    upper half, A; two on the axial scribe, B) and three on the end face (C)
    are moved by a known rigid pose; the cylinder is fitted from a start axis
    tilted by about 0.02 rad and offset by 1.4 mm with a radius 1 mm short, and
    the axis-primary frame is built. Without noise the frame must equal the
    pose and the radius the nominal one. A scribe on the axis and an end face
    parallel to it must be refused.
    """
    radius = geo.CYLINDER_RADIUS
    pose = met.transform(met.exp_so3(AXIS_POSE[0]), AXIS_POSE[1])

    def moved(points):
        points = np.asarray(points, dtype=float)
        return points @ pose[:3, :3].T + pose[:3, 3]

    probes = axis_probe_points()
    rings = [geo.CYLINDER.embedding(np.array(u)) for u in probes["A"]]
    scribe = moved([geo.CYLINDER.embedding(np.array(u)) for u in probes["B"]])
    face = moved([[0.0, 50.0, 0.0], [50.0, -30.0, 0.0], [-40.0, -20.0, 0.0]])
    start_point = moved([[1.0, -1.0, 150.0]])[0]
    start_direction = pose[:3, :3] @ np.array([0.02, 0.01, 1.0])
    point, direction, fitted, residual = met.fit_cylinder(moved(rings), start_point, start_direction, radius - 1.0)
    frame = met.datum_frame_axis(point, direction, scribe, face)
    parallel_face = moved([[radius, 0.0, 0.0], [radius, 0.0, 100.0], [radius * math.cos(0.3), radius * math.sin(0.3), 50.0]])
    return {"pose_error": float(np.max(np.abs(met.pose_difference(frame, pose)))),
            "radius_error_mm": abs(fitted - radius), "residual_mm": float(np.max(np.abs(residual))),
            "scribe_on_axis": _metrology_refusal(met.datum_frame_axis, point, direction, [point, point + 10.0 * direction],
                                                 face),
            "face_parallel_to_axis": _metrology_refusal(met.datum_frame_axis, point, direction, scribe, parallel_face),
            "too_few_points": _metrology_refusal(met.fit_cylinder, moved(rings)[:5], start_point, start_direction, radius)}


@_task("T127", ("test_cylinder_protocol_predicts_chord_geodesic_gaps", "test_protocols_refuse_filled_slots_and_decisions"))
def rolled_cylinder_control(ctx):
    study = ctx.memo("mfg.cylinder", cylinder_study)
    gaps = {r["pair"]: r["gap_mm"] for r in study["pairs"]}
    f_gap = finding("Rolled-cylinder chord-geodesic gaps for the protocol marker pairs", "numerical", gaps,
                    {"derivation": "geodesic = sqrt((R dphi)^2 + dz^2) on the development; chord = |X(b) - X(a)|",
                     "checks": [_check("analytic", "RK4 geodesic closure onto the target marker (mm)", study["max_closure_mm"], 1e-9),
                                _check("analytic", "max over circumferential pairs of |gap - 2-term series| minus the "
                                       "alternating-series bound (signed margin, mm)",
                                       study["series_excess_mm"], 0.0, "signed_le"),
                                _check("invariant", "axial (ruling) pairs have zero gap (mm)", study["axial_gap_mm"], 1e-9)]},
                    unit="mm",
                    uncertainty=_u("roundoff", study["max_closure_mm"], "closed forms; RK4 closure onto the target marker"),
                    tolerance={"abs": 1e-9, "rel": 1e-9})
    f_flat = finding("Rolled-cylinder Jacobi transfer equals the flat-plate transfer (K = 0)", "numerical",
                     study["flat_transfer_error"],
                     {"checks": [_check("analytic", "max |Phi_cylinder(300) - [[1, 300], [0, 1]]|", study["flat_transfer_error"], 1e-9)]},
                     uncertainty=_u("roundoff", study["flat_transfer_error"], "deviation of Phi from the flat closed form"),
                     tolerance={"abs": 1e-9, "rel": 0})
    ninety = next(r for r in study["pairs"] if r["pair"] == "circumferential 90 deg")
    f_counter = finding("A marker chord differs from the surface distance on a developable (intrinsically flat) part",
                        "numerical", ninety["gap_mm"],
                        {"checks": [_check("analytic", "gap at 90 deg exceeds k u_pair (camera) by the ratio",
                                           ninety["gap_mm"] / (COVERAGE_K * PAIR_U), 1.0, "ge")]},
                        unit="mm",
                        uncertainty=_u("roundoff", study["max_closure_mm"], "closed form; RK4 closure"),
                        tolerance={"abs": 1e-9, "rel": 1e-9},
                        counterexample={"statement": "On an intrinsically flat (developable) part the straight chord between "
                                                     "two markers equals their surface distance, as on the flat plate",
                                        "witness": {"radius_mm": geo.CYLINDER_RADIUS, "dphi_deg": 90,
                                                    "geodesic_mm": ninety["geodesic_mm"], "chord_mm": ninety["chord_mm"]}})
    f_res = finding("Minimum circumferential marker separation that resolves the chord-geodesic gap at k = 2",
                    "numerical", {k: v["min_arc_mm"] for k, v in study["resolvable"].items()},
                    {"checks": [_check("analytic", "bisection root vs (24 R^2 g)^(1/3) series (relative)",
                                       study["resolvable_series_rel"], 0.01)]},
                    unit="mm",
                    uncertainty=_u("roundoff", 1e-12, "200-step bisection of the closed-form gap"),
                    tolerance={"abs": 1e-8, "rel": 1e-9})
    axis = ctx.memo("mfg.axis-frame", axis_frame_study)
    f_axis = finding("The axis-primary datum frame (cylinder-fit axis A, scribe B, end face C) recovers a known tube pose "
                     "from noise-free synthetic CMM points and refuses degenerate datums", "numerical",
                     {"pose_error": axis["pose_error"], "radius_error_mm": axis["radius_error_mm"]},
                     {"generator": _generator("noise-free CMM points on the MFG-CYLINDER-01 datum features in a known pose",
                                              seed=None, rotation_vector_rad=list(AXIS_POSE[0]),
                                              translation_mm=list(AXIS_POSE[1])),
                      "checks": [_check("analytic", "max |pose difference| of the built PART frame from the known pose "
                                        "(mm and rad)", axis["pose_error"], 1e-9),
                                 _check("analytic", "|fitted radius - 100 mm| (mm)", axis["radius_error_mm"], 1e-9),
                                 _refusal("scribe datum (B) on the axis (A)", "datum_degenerate", axis["scribe_on_axis"]),
                                 _refusal("end face (C) parallel to the axis (A)", "datum_degenerate",
                                          axis["face_parallel_to_axis"]),
                                 _refusal("cylinder fit from five points", "cylinder_underdetermined",
                                          axis["too_few_points"])]},
                     uncertainty=_u("roundoff", max(axis["pose_error"], axis["radius_error_mm"]),
                                    "noise-free recovery error of the fit and frame"),
                     tolerance={"abs": 1e-9, "rel": 0})
    u_radius = 0.1 / math.sqrt(3.0)
    predicted = [_predicted(f"G{k}", f"chord-geodesic gap, {name}", _r(value), "mm", f_gap)
                 for k, (name, value) in enumerate(gaps.items())]
    predicted += [_predicted(f"K{k}", f"marker chord, {r['pair']}", _r(r["chord_mm"]), "mm", f_gap)
                  for k, r in enumerate(study["pairs"])]
    predicted.append(_predicted("J1", "Jacobi transfer along a 45 deg helix of 300 mm", [[1.0, 300.0], [0.0, 1.0]], "1, mm", f_flat))
    criteria = [
        {"id": "H1", "status": "hypothesis", "statement": "The film surface distance minus the camera chord equals the "
         "predicted chord-geodesic gap for every marker pair",
         "test": f"E_n <= 1 with U_m = 2 sqrt(u_film^2 + u_pair^2) = {COVERAGE_K * GAP_U:.4f} mm (film 0.05 mm, camera "
                 f"pair {PAIR_U:.4f} mm) and U_p = 2 |d gap / dR| u_R, u_R = 0.1 / sqrt(3) mm (the T140 radius term)"},
        {"id": "H2", "status": "hypothesis", "statement": "Separation of offset helices is the flat-plate separation",
         "test": "fit slope and intercept of the 3D distances between the two tape centrelines (their difference from "
                 "the surface separation is below 1e-4 mm at 2 mm on R = 100 mm); compare with the flat-plate control T126"},
        {"id": "H3", "status": "hypothesis", "statement": "The camera chord alone differs from the predicted geodesic "
         "distance for circumferential pairs of 30 deg and more (the extrinsic signature)",
         "test": f"|chord - geodesic| / sqrt((2 u_pair)^2 + (2 |d chord / dR| u_R)^2) > 1; separations below the T127 "
                 f"resolvable arc ({study['resolvable']['camera']['min_arc_mm']:.1f} mm with the camera) are not tested"}]
    helix_start = (math.radians(HELIX_START_DEG), 0.0)
    paths = tape_paths({"id": "HX45", "start_u": list(helix_start), "heading_rad": math.radians(45.0), "length_mm": 300.0},
                       [{"id": "HX45-L2", "lateral_offset_mm": 2.0}], geo.CYLINDER, helix_start)
    pair_ids = [r["pair"] for r in study["pairs"]]
    markers = [dict(m, kind="specimen marker", position_mm=_cylinder_points([m["u"]])[0]) for m in study["markers"]]
    visible = [m["id"] for m in study["markers"] if m["faces_up"]]
    protocol = build_protocol(
        "T127", "MFG-CYLINDER-01", "Rolled-cylinder control: extrinsic curvature without intrinsic curvature",
        "Separate extrinsic effects (chord vs geodesic) from intrinsic ones (Jacobi separation), which must "
        "match the flat plate.",
        {"kind": "rolled tube", "material": "rolled and seam-welded 3 mm aluminium (declared)",
         "nominal_radius_mm": geo.CYLINDER_RADIUS, "length_mm": 300.0, "declared_radius_tolerance_mm": 0.1,
         "declared_roundness_tolerance_mm": ROUNDNESS_TOLERANCE_MM,
         "seam_weld": "along phi = 180 deg, opposite scribe B: it faces down between the vee contact lines (not modelled)",
         "surface_model": geo.CYLINDER.describe(), "marker_pairs": {r["pair"]: r["markers"] for r in study["pairs"]}},
        markers, paths, predicted, criteria, ("camera", "tracker", "cmm", "film"),
        [register_step(V_BLOCKS),
         _step(f"Measure the {len(visible)} markers on the upper half of the three rings (within 90 deg of scribe B) with "
               "the camera (SB-1000 and the VB-01 datum targets in every image set) and the tracker; export the camera "
               "coordinates as the photogrammetry capture. The lower markers face into the vees and the table and are "
               "not measured.", ["camera", "tracker", "SB-1000", "VB-01"],
               ["raw images", "photogrammetry capture (ciw.lab-mfg-target-capture.v1)", "tracker point file"],
               key="markers"),
         _step("Lay the film gauge from marker centre to marker centre of each of the ten marker pairs (all on the upper "
               "half) without tension or in-plane steering, read the surface distance under a 10x loupe, and export the "
               "readings as the film capture.", ["film"], ["film capture (ciw.lab-mfg-distance-capture.v1)"],
               check="two readings per pair agree within 2 x 0.05 mm; otherwise re-lay the film", key="film"),
         _repeat(["locate", "markers", "film"], "VB-01", "replicates of the datum, marker and film measurements"),
         *[step for path in paths for step in tape_block(path, "VB-01", "every 30 mm of arclength from s = 0 to 300 mm")]],
        [capture_format("photogrammetry", TARGET_CAPTURE, "camera", visible,
                        "ciw.lab.manufacturing_records.read_capture; T138 compares the ten pair chords (H3) and, with "
                        "the film capture, the chord-geodesic gaps (H1)"),
         capture_format("film", DISTANCE_CAPTURE, "film", pair_ids,
                        "ciw.lab.manufacturing_records.read_capture; T138 compares film minus chord with the predicted "
                        "gaps (H1)"),
         capture_format("cmm", TARGET_CAPTURE, "cmm", start_pose_ids(paths),
                        "ciw.lab.manufacturing_records.read_capture; retained by T138, not yet used to condition the "
                        "helix prediction")],
        holder=V_BLOCKS)
    ctx.artifact_json("protocol-rolled-cylinder.json", protocol)
    ctx.artifact_json("cylinder-pairs.json", _r({"pairs": study["pairs"], "resolvable": study["resolvable"],
                                                 "axis_frame": axis}))
    circ = [r for r in study["pairs"] if "series_mm" in r]
    ctx.artifact_text("cylinder-gap.svg", svg.line_plot(
        [("exact gap", [r["geodesic_mm"] for r in circ], [r["gap_mm"] for r in circ]),
         ("d^3/24R^2 - d^5/1920R^4", [r["geodesic_mm"] for r in circ], [r["series_mm"] for r in circ])],
        title="Rolled cylinder R = 100 mm: chord-geodesic gap", xlabel="arc length (mm)", ylabel="gap (mm)"))
    findings = [f_gap, f_flat, f_counter, f_res, f_axis,
                _protocol_findings(protocol, "Rolled-cylinder protocol record validates and refuses malformed variants"),
                _not_measured("Measured chords and surface distances on the physical tube match the predicted gaps"),
                _not_measured("The physical tube radius and roundness lie within the declared +/- 0.1 mm", "calibration"),
                _not_measured("The unrolled-film gauge achieves the declared 0.05 mm on marker-to-marker surface distances",
                              "calibration")]
    res = study["resolvable"]
    fields = _fields(
        "A rolled cylinder has zero Gaussian curvature, so its Jacobi transfer equals the plate's, while its "
        "extrinsic curvature makes marker chords shorter than geodesic distances by d^3/(24 R^2) + O(d^5); a surface "
        "distance measured independently of the chord (an unrolled-film gauge) exposes that gap.",
        "Cylinder X(phi, z) = (R cos phi, R sin phi, z), R = 100 mm; development (R phi, z) is an isometry; "
        "gap(d) = d - 2 R sin(d / 2R) for circumferential pairs; resolvable arc solves gap(d) = k sqrt(2) u. Datum frame: "
        "least-squares cylinder axis A, scribe B, end face C, z along A, origin at A meets C, x towards B.",
        ["Declared tube R = 100 mm, length 300 mm, three rings of 12 markers", "Ten marker pairs (circumferential, axial, helical)",
         "Declared instruments (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm, film gauge 0.05 mm)",
         f"Radius tolerance +/- 0.1 mm (u_R = {u_radius:.4f} mm)"],
        "No observation. The protocol specifies CMM datum probing in a V-block pair, marker, film, tape start-pose and "
        "tape centreline measurements and the raw formats of their captures; the hardware slot is empty.",
        "gap = 0 on rulings (axial pairs); Phi_cylinder = Phi_plate; circumferential gaps follow the alternating series; "
        "the axis-primary frame recovers a known pose exactly without noise.",
        "Closed-form gaps checked by RK4 geodesic closure and by the series with its remainder bound; helix "
        "transfer compared with [[1, s], [0, 1]]; resolvable separation by bisection; the datum scheme (cylinder fit and "
        "axis-primary frame) run on noise-free synthetic points in a known pose, and its degenerate datums refused; "
        "protocol validated and mutated.",
        f"gap at 90 deg = {ninety['gap_mm']:.4f} mm (chord {ninety['chord_mm']:.3f} vs geodesic {ninety['geodesic_mm']:.3f}); "
        f"minimum resolvable arc: camera {res['camera']['min_arc_mm']:.1f} mm, tracker {res['tracker']['min_arc_mm']:.1f} mm, "
        f"CMM {res['cmm']['min_arc_mm']:.1f} mm; datum frame recovered to {axis['pose_error']:.1e}.",
        "Closed forms; RK4 closure at rounding level. The radius tolerance term is budgeted in T140; the film and camera "
        f"terms give U_m = {COVERAGE_K * GAP_U:.4f} mm for a measured gap.",
        ["RK4 closure", "series remainder bound", "axial rulings give zero gap", "flat Jacobi transfer",
         "axis-primary datum frame: known pose, scribe on the axis, end face parallel to the axis, too few points",
         "protocol refusal matrix, including the tape layout (tapes alone, clear of markers and datum points)"],
        ["Seam weld and out-of-roundness are not modelled; a real tube is not a perfect cylinder. The protocol puts the "
         f"seam at the bottom between the vees and stops the run when a ring's roundness exceeds "
         f"{roundness_threshold():.3f} mm (declared form tolerance plus 2 U of the CMM).",
         "Marker centre offsets (target thickness) are not modelled.",
         "The film is assumed to lie on the pair's geodesic when laid without steering, like the tapes."],
        "Acquire MFG-CYLINDER-01 and bind its exports to T138 (ciw lab run T138 --capture photogrammetry=<targets.csv> "
        "--capture film=<film.csv>), then retain the run with ciw lab hardware retain; T138's comparison of the "
        "captured gaps and chords is computational, and a physical label needs a metrology instrument probe or a "
        "signed-capture trust anchor (deferred research question). Open: model the seam weld and out-of-roundness, "
        "which the cylinder prediction omits.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T128 domed coupon ------------------------------------------------------------------------
NOMINAL_STATIONS = 8
COUPON_TARGETS = [f"{tape}{k}" for tape in "NLH" for k in range(NOMINAL_STATIONS + 1)]


@functools.lru_cache(maxsize=1)
def nominal_study() -> dict:
    """Nominal-route predictions (deterministic; cached because T128, T134, T138 and T140 share them)."""
    route = geo.route_to_edge(geo.COUPON, geo.STATION, 0.0, geo.COUPON_X[1], name="nominal")
    coarse = geo.route_to_edge(geo.COUPON, geo.STATION, 0.0, geo.COUPON_X[1], name="nominal-h2", step=2.0)
    length = route.length
    steps = NOMINAL_STATIONS * 26
    fine = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps)
    half = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps // 2)
    stride = steps // NOMINAL_STATIONS
    remainder = {}
    for label, heading in (("on_axis", 0.0), ("off_axis", math.radians(10.0))):
        base = fine if heading == 0.0 else None
        span = length if heading == 0.0 else 200.0
        errors = []
        for eps in (0.4, 0.2, 0.1):
            transfer = base or jacobi.transfer(geo.COUPON, geo.STATION, heading, span, steps=steps)
            base = transfer
            linear = geo.separation_linear(transfer, eps, eps * 1e-2)
            nonlinear = geo.separation_nonlinear(geo.COUPON, geo.STATION, heading, span, steps, eps, eps * 1e-2, base=transfer)
            errors.append(float(np.max(np.abs(nonlinear - linear))))
        pairwise = [math.log2(errors[0] / errors[1]), math.log2(errors[1] / errors[2])]
        remainder[label] = {"eps_mm": [0.4, 0.2, 0.1], "max_remainder_mm": errors,
                            "order": integrators.observed_order([0.4, 0.2, 0.1], errors),
                            "order_spread": abs(pairwise[0] - pairwise[1]) / 2.0}
    lateral, dheading = 2.0, 0.005
    crossings = {}
    for size in (lateral, 0.5 * lateral):
        separation = geo.separation_nonlinear(geo.COUPON, geo.STATION, 0.0, length, steps, size, 0.0, base=fine)
        zeros = integrators.hermite_zeros(fine.s, separation, np.gradient(separation, fine.s))
        crossings[size] = zeros[0] if zeros else float("nan")
        if size == lateral:
            nonlinear_lateral = separation
    # On the symmetry axis the separation is odd in the offset, so the crossing moves as offset^2;
    # Richardson in offset^2 recovers the delta -> 0 limit, which must be the focal point.
    extrapolated = (4.0 * crossings[0.5 * lateral] - crossings[lateral]) / 3.0
    return {"length_mm": length, "focal_mm": route.focal[0], "focal_h2_mm": coarse.focal[0],
            "phi_end": fine.matrix().tolist(), "phi_richardson": float(np.max(np.abs(fine.matrix() - half.matrix())) / 15.0),
            "det_drift": float(np.max(np.abs(fine.determinant() - 1.0))),
            "stations_mm": fine.s[::stride].tolist(),
            "j_lat": fine.states[::stride, 4].tolist(), "j_head_mm": fine.states[::stride, 6].tolist(),
            "j_lat_half_step": half.states[::stride // 2, 4].tolist(),
            "lateral_linear_mm": (lateral * fine.states[::stride, 4]).tolist(),
            "lateral_nonlinear_mm": nonlinear_lateral[::stride].tolist(),
            "heading_linear_mm": (dheading * fine.states[::stride, 6]).tolist(),
            "nonlinear_crossing_mm": crossings[lateral], "nonlinear_crossing_half_mm": crossings[0.5 * lateral],
            "crossing_extrapolated_mm": extrapolated,
            "remainder": remainder, "profile_s": fine.s.tolist(), "profile_j_lat": fine.states[:, 4].tolist(),
            "profile_j_head": fine.states[:, 6].tolist(), "lateral_mm": lateral, "dheading_rad": dheading}


COUPON_CHECK_POINTS = ((-60.0, 0.0), (-25.0, 3.0), (0.0, 0.0), (12.0, -7.0), (34.6, 0.0), (80.0, 20.0))


def _fd_christoffel(surface, u, step=1e-4):
    """Christoffel symbols from central differences of the ciw metric (no metric_derivatives)."""
    dg = np.array([(surface.metric(u + step * e) - surface.metric(u - step * e)) / (2 * step) for e in np.eye(2)])
    ginv = np.linalg.inv(surface.metric(u))
    lowered = np.einsum("ijl->lij", dg) + np.einsum("jil->lij", dg) - dg
    return 0.5 * np.einsum("kl,lij->kij", ginv, lowered)


def _fd_monge_curvature(surface, u, step=1e-2):
    """K = (f_xx f_yy - f_xy^2) / (1 + |grad f|^2)^2 from central differences of the height values only."""
    x, y = u

    def f(dx, dy):
        return surface.height(np.array([x + dx, y + dy]))

    f0 = f(0.0, 0.0)
    fx, fy = (f(step, 0.0) - f(-step, 0.0)) / (2 * step), (f(0.0, step) - f(0.0, -step)) / (2 * step)
    fxx = (f(step, 0.0) - 2 * f0 + f(-step, 0.0)) / step ** 2
    fyy = (f(0.0, step) - 2 * f0 + f(0.0, -step)) / step ** 2
    fxy = (f(step, step) - f(step, -step) - f(-step, step) + f(-step, -step)) / (4 * step ** 2)
    return (fxx * fyy - fxy ** 2) / (1 + fx ** 2 + fy ** 2) ** 2


def independent_coupon_checks(length) -> dict:
    """Second derivations of the coupon transfer and geometry, same-origin always and different-origin when available.

    Always: ciw's adaptive Dormand-Prince 5(4) integration of the same augmented
    equations, Christoffel symbols from central differences of the ciw metric
    and Gaussian curvature from central differences of the height values in the
    Monge formula (not from the analytic derivatives the surface uses). When
    installed: scipy's DOP853 integrating the ciw right-hand side (independent
    time stepping only; the geometry code is shared) and a sympy derivation of
    the Christoffel symbols and curvature from the height formula (independent
    geometry at six points).
    """
    import importlib.util

    surface = geo.COUPON
    y0 = jacobi.initial_state(surface, geo.STATION, 0.0)
    rhs = jacobi.rhs(surface)
    rk4 = jacobi.transfer(surface, geo.STATION, 0.0, length, steps=NOMINAL_STATIONS * 26).states[-1]
    _, states, _ = integrators.integrate_adaptive(rhs, y0, length, rtol=1e-11, atol=1e-12)
    points = [np.array(p) for p in COUPON_CHECK_POINTS]
    ours = [(surface.christoffel(u), surface.gaussian_curvature(u)) for u in points]
    out = {"ciw_integrator": {"max_difference": float(np.max(np.abs(rk4 - states[-1])))},
           "ciw_geometry": {"max_christoffel_difference": float(max(np.max(np.abs(g - _fd_christoffel(surface, u)))
                                                                    for u, (g, _) in zip(points, ours))),
                            "max_curvature_difference": float(max(abs(k - _fd_monge_curvature(surface, u))
                                                                  for u, (_, k) in zip(points, ours)))},
           "scipy": None, "sympy": None}
    if importlib.util.find_spec("scipy") is not None:
        import scipy
        from scipy.integrate import solve_ivp
        solution = solve_ivp(lambda _, y: rhs(y), (0.0, length), y0, method="DOP853", rtol=1e-12, atol=1e-12)
        out["scipy"] = {"checker": "scipy.integrate.solve_ivp DOP853", "revision": scipy.__version__,
                        "max_difference": float(np.max(np.abs(rk4 - solution.y[:, -1])))}
    if importlib.util.find_spec("sympy") is not None:
        import sympy as sp
        x, y = sp.symbols("x y", real=True)
        height = sp.nsimplify(geo.DOME_HEIGHT) * sp.exp(-(x ** 2 + y ** 2) / (2 * sp.nsimplify(geo.DOME_SIGMA) ** 2))
        coords = (x, y)
        grad = [sp.diff(height, c) for c in coords]
        metric = sp.Matrix(2, 2, lambda i, j: (1 if i == j else 0) + grad[i] * grad[j])
        inverse = metric.inv()
        gamma = [[[sum(inverse[k, l] * (sp.diff(metric[j, l], coords[i]) + sp.diff(metric[i, l], coords[j])
                                          - sp.diff(metric[i, j], coords[l])) for l in range(2)) / 2
                   for j in range(2)] for i in range(2)] for k in range(2)]
        hess = [[sp.diff(height, a, b) for b in coords] for a in coords]
        curvature = (hess[0][0] * hess[1][1] - hess[0][1] ** 2) / (1 + grad[0] ** 2 + grad[1] ** 2) ** 2
        evaluate = sp.lambdify((x, y), [gamma, curvature], "math")
        reference = [evaluate(*u) for u in points]
        out["sympy"] = {"checker": "sympy symbolic differentiation", "revision": sp.__version__,
                        "max_christoffel_difference": float(max(np.max(np.abs(np.asarray(a[0], dtype=float)
                                                                               - np.asarray(b[0], dtype=float)))
                                                                for a, b in zip(ours, reference))),
                        "max_curvature_difference": float(max(abs(a[1] - float(b[1])) for a, b in zip(ours, reference)))}
    return out


def _optional_finding(claim, result, reference, observed, threshold, module, **extra):
    """A different-origin check when its module ran; otherwise an honestly unestablished claim with no value.

    The claim names the checker, so its label depends only on whether that
    module is installed (numpy-only regeneration differs by exactly these labels).
    """
    if result is None:
        return finding(claim, "numerical", None, {"notes": f"{module} is not installed here; the check did not run"},
                       expected_not_established=True)
    check = _check("high_precision", reference, observed, threshold)
    basis = {"independent_check": dict(check, producer={"implementation": "ciw.lab.jacobi", "revision": f"ciw {__version__}"},
                                       checker={"implementation": result["checker"], "revision": result["revision"]})}
    return finding(claim, "numerical", observed, basis, **extra)


@_task("T128", ("test_coupon_protocol_predicts_focal_crossing", "test_coupon_report_wording_does_not_depend_on_optional_modules",
                "test_protocols_refuse_filled_slots_and_decisions"))
def curved_coupon(ctx):
    study = ctx.memo("mfg.nominal", nominal_study)
    length, focal = study["length_mm"], study["focal_mm"]
    f_phi = finding("Domed-coupon Jacobi transfer at the end of the nominal route", "numerical",
                    {"length_mm": length, "phi_end": study["phi_end"]},
                    {"checks": [_check("invariant", "Wronskian det Phi - 1 along the route", study["det_drift"], 1e-8),
                                _check("self_convergence", "Richardson estimate |Phi(h) - Phi(2h)| / 15", study["phi_richardson"], 1e-5)]},
                    uncertainty=_u("truncation_bound", study["phi_richardson"],
                                   "RK4 Richardson estimate |Phi(h) - Phi(2h)| / 15"),
                    tolerance={"abs": 1e-6, "rel": 1e-7})
    f_focal = finding("Laterally offset routes cross the nominal route at the first focal point (small-offset limit)",
                      "numerical", focal,
                      {"checks": [_check("self_convergence", "focal point at h = 1 mm vs h = 2 mm (mm)",
                                         focal - study["focal_h2_mm"], 1e-3),
                                  _check("self_convergence", "offset crossings (2 and 1 mm) extrapolated in offset^2 "
                                         "minus the focal point (mm)", study["crossing_extrapolated_mm"] - focal, 0.02),
                                  _check("analytic", "focal point lies inside the route (s_f / L)", focal / length, 1.0, "le")]},
                      unit="mm",
                      uncertainty=_u("truncation_bound", abs(focal - study["focal_h2_mm"]) / 15.0,
                                     "RK4 Richardson estimate of the focal location (mm)"),
                      tolerance={"abs": 1e-4, "rel": 1e-7})
    rem = study["remainder"]
    f_order = finding("Linearized separation remainder is at least second order (third order on the symmetry axis)",
                      "numerical", {k: v["order"] for k, v in rem.items()},
                      {"checks": [_check("self_convergence", "off-axis remainder order", rem["off_axis"]["order"], 1.8, "ge"),
                                  _check("self_convergence", "on-axis remainder order - 3 (odd symmetry)", rem["on_axis"]["order"] - 3.0, 0.3)]},
                      uncertainty=_u("reference_error", max(v["order_spread"] for v in rem.values()),
                                     "half-spread of pairwise orders from three perturbation sizes"),
                      tolerance={"abs": 1e-4, "rel": 1e-5})
    independent = ctx.memo("mfg.independent", lambda: independent_coupon_checks(length))
    integ, geom = independent["ciw_integrator"], independent["ciw_geometry"]
    f_integrator = finding("The RK4 coupon transfer agrees with ciw's adaptive Dormand-Prince 5(4) integration of the "
                           "same equations", "numerical", integ["max_difference"],
                           {"checks": [_check("cross_implementation", "max |state(L)| difference, RK4 (h ~ 1 mm) vs "
                                              "ciw Dormand-Prince 5(4) at rtol 1e-11", integ["max_difference"], 1e-5)]},
                           uncertainty=_u("truncation_bound", integ["max_difference"], "difference between the two integrations"),
                           tolerance={"abs": 1e-5, "rel": 0})
    f_geometry = finding("Coupon Christoffel symbols and Gaussian curvature agree with finite-difference derivations "
                         "from the metric and from the height values", "numerical",
                         {"christoffel": geom["max_christoffel_difference"], "curvature": geom["max_curvature_difference"]},
                         {"checks": [_check("cross_implementation", "max |Gamma - central differences of the metric| "
                                            "at six points (1/mm)", geom["max_christoffel_difference"], 1e-10),
                                     _check("cross_implementation", "max |K - Monge formula on central differences of the "
                                            "height (step 0.01 mm)| at six points (1/mm^2)", geom["max_curvature_difference"], 1e-9)]},
                         uncertainty=_u("truncation_bound", geom["max_curvature_difference"],
                                        "O(h^2) error of the finite-difference curvature (1/mm^2)"),
                         tolerance={"abs": 1e-9, "rel": 0})
    scipy_result, sympy_result = independent["scipy"], independent["sympy"]
    f_scipy = _optional_finding(
        "scipy DOP853 integrating the ciw right-hand side agrees with the RK4 coupon transfer (independent time stepping; "
        "the geometry code is shared)", scipy_result, "max |state(L)| difference, RK4 (h ~ 1 mm) vs scipy DOP853 at rtol 1e-12",
        None if scipy_result is None else scipy_result["max_difference"], 1e-5, "scipy",
        uncertainty=_u("truncation_bound", None if scipy_result is None else scipy_result["max_difference"],
                       "difference between the two integrations"),
        tolerance={"abs": 1e-5, "rel": 0})
    f_sympy = _optional_finding(
        "A sympy derivation of the coupon Christoffel symbols and Gaussian curvature from the height formula agrees with "
        "ciw at six points", sympy_result, "max |Gamma| (1/mm) and |K| (1/mm^2) differences at six points",
        None if sympy_result is None else max(sympy_result["max_christoffel_difference"], sympy_result["max_curvature_difference"]),
        1e-12, "sympy", uncertainty=_u("roundoff", 1e-15, "double-precision evaluation of both derivations"),
        tolerance={"abs": 1e-12, "rel": 0})
    prediction = ctx.memo("mfg.prediction", separation_prediction)
    end_u = prediction["open_loop_expanded_mm"][-1]
    combined = math.hypot(end_u, COVERAGE_K * PAIR_U)
    plate_minus_coupon = 2.0 - study["lateral_nonlinear_mm"][-1]
    f_flat = finding("Curvature signature relative to the flat-plate control at the route end", "numerical",
                     {"coupon_lateral_2mm": study["lateral_nonlinear_mm"][-1], "plate_lateral_2mm": 2.0,
                      "coupon_heading_5mrad": study["heading_linear_mm"][-1], "plate_heading_5mrad": 0.005 * length,
                      "expanded_uncertainty_mm": combined},
                     {"checks": [_check("analytic", "(plate - coupon separation of the 2 mm offset route at L) / combined "
                                        "k = 2 uncertainty (open-loop prediction and camera pair)", plate_minus_coupon / combined,
                                        1.0, "ge"),
                                 _check("self_convergence", "Richardson solver estimate of the end separation, "
                                        "2 |j_lat(h) - j_lat(2h)| / 15 (mm)", prediction["solver_mm"][-1], 1e-5)]},
                     unit="mm",
                     uncertainty=_u("reference_error", end_u, "k = 2 open-loop prediction uncertainty at the end (mm)"),
                     tolerance={"abs": 1e-6, "rel": 1e-7})
    stations = [round(s, 6) for s in study["stations_mm"]]
    predicted = [
        _predicted("S1", "separation of the 2 mm laterally offset route at stations k L / 8 (nonlinear)",
                   _r(study["lateral_nonlinear_mm"]), "mm", f_flat),
        _predicted("S2", "separation of the 5 mrad heading-offset route at stations k L / 8",
                   _r(study["heading_linear_mm"]), "mm", f_flat),
        _predicted("F1", "arclength where the 2 mm laterally offset route crosses the nominal route (nonlinear)",
                   _r(study["nonlinear_crossing_mm"]), "mm", f_focal),
        _predicted("F0", "first focal point (offset -> 0 limit of the crossing)", _r(focal), "mm", f_focal),
        _predicted("J1", "Jacobi transfer at the route end", _r(study["phi_end"]), "1, mm", f_phi)]
    criteria = [
        {"id": "H1", "status": "hypothesis", "statement": "The 2 mm offset tape crosses the nominal tape where the geodesic "
         "re-integrated from its measured start pose crosses (near the focal point F0)",
         "test": "crossing arclength (linear interpolation between the two stations whose separations change sign, "
                 "applied alike to the measured and the re-integrated predicted separations) within the T140 expanded "
                 "uncertainty of the focal distance (conditioned row)"},
        {"id": "H2", "status": "hypothesis", "statement": "Conditioned on the measured start pose, the measured separations follow "
         "the Jacobi prediction, not the flat-plate one",
         "test": "E_n <= 1 at stations 1..8 against the prediction re-integrated from the CMM start pose, with U_p from dome "
                 "tolerances, the start-pose estimate and solver error (T138) and U_m from the camera pair; E_n > 1 against "
                 "the flat-plate prediction at the last two stations"}]
    paths = tape_paths({"id": "N", "start_u_mm": list(geo.STATION), "heading_rad": 0.0, "end": "x = 140 mm",
                        "length_mm": length},
                       [{"id": "L", "lateral_offset_mm": 2.0, "construction": "exp map along the start normal"},
                        {"id": "H", "heading_offset_rad": 0.005}], geo.COUPON, geo.STATION)
    protocol = build_protocol(
        "T128", "MFG-COUPON-01", "Domed coupon: Jacobi focusing of offset paths",
        "Measure the separation of exactly offset robot paths across a dome, where positive curvature focuses "
        "laterally offset paths so that they cross the nominal path inside the coupon.",
        {"kind": "domed coupon", "material": "formed 2 mm aluminium sheet (declared)",
         "surface_model": geo.COUPON.describe(), "chart_extent_mm": {"x": list(geo.COUPON_X), "y": list(geo.COUPON_Y)},
         "declared_height_tolerance_mm": 0.2, "declared_sigma_tolerance_mm": 0.5,
         "crest_principal_radius_mm": geo.DOME_SIGMA ** 2 / geo.DOME_HEIGHT},
        [{"id": f"{tape}{k}", "route": route, "s_mm": s, "kind": "tape target", "on": tape}
         for tape, route in (("N", "nominal"), ("L", "lateral 2 mm"), ("H", "heading 5 mrad"))
         for k, s in enumerate(stations)],
        paths, predicted, criteria, ("camera", "tracker", "cmm", "scanner"),
        [register_step(NEST),
         _step("Scan the coupon surface with the scanner following protocol MFG-SCAN-01 (T129) to identify the as-built "
               "dome and its covariance before laying the tapes.", ["scanner"], ["MFG-SCAN-01 records"]),
         *[step for path in paths for step in tape_block(
             path, "FX-321", "at the nine stations s = k L / 8", f"{path['id']}0-{path['id']}{NOMINAL_STATIONS}")]],
        [capture_format("photogrammetry", TARGET_CAPTURE, "camera", COUPON_TARGETS,
                        "ciw.lab.manufacturing_records.read_capture; T138 reduces the lateral tape targets to separations "
                        "(projection on the model's in-surface normal of the nominal route) and compares them (H2)"),
         capture_format("cmm", TARGET_CAPTURE, "cmm", start_pose_ids(paths),
                        "ciw.lab.manufacturing_records.read_capture; T138 estimates the realized start pose of the "
                        "lateral tape and re-integrates the prediction from it (conditioned H2)")])
    ctx.artifact_json("protocol-domed-coupon.json", protocol)
    ctx.artifact_json("coupon-predictions.json", _r({k: study[k] for k in (
        "length_mm", "focal_mm", "stations_mm", "j_lat", "j_head_mm", "lateral_linear_mm", "lateral_nonlinear_mm",
        "heading_linear_mm", "remainder")}))
    s = study["profile_s"]
    ctx.artifact_text("coupon-separation.svg", svg.line_plot(
        [("coupon: 2 mm lateral", s, [2.0 * v for v in study["profile_j_lat"]]),
         ("plate: 2 mm lateral", [s[0], s[-1]], [2.0, 2.0]),
         ("coupon: 5 mrad heading", s, [0.005 * v for v in study["profile_j_head"]]),
         ("plate: 5 mrad heading", [s[0], s[-1]], [0.0, 0.005 * s[-1]])],
        title="Domed coupon vs flat plate: predicted offset separation", xlabel="arclength s (mm)",
        ylabel="separation (mm)", markers=False))
    findings = [f_focal, f_phi, f_order, f_flat, f_integrator, f_geometry, f_scipy, f_sympy,
                _protocol_findings(protocol, "Domed-coupon protocol record validates and refuses malformed variants"),
                _not_measured("Measured separations of offset routes on the physical coupon follow the Jacobi prediction "
                              "and cross at the predicted focal point"),
                _not_measured("The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance",
                              "calibration"),
                _not_measured("An unsteered 3 mm tape laid on the coupon follows a geodesic of the as-built surface "
                              "(no in-plane bending, lift-off or slip)"),
                _not_measured(EXECUTION_CLAIM, "calibration")]
    fields = _fields(
        "Across the dome, positive Gaussian curvature focuses laterally offset geodesic routes: the 2 mm offset "
        "route crosses the nominal route inside the coupon, a signature absent on the flat plate and on the cylinder.",
        "Coupon z = h exp(-(x^2 + y^2) / (2 sigma^2)), h = 10 mm, sigma = 20 mm; nominal route from (-60, 0) along +x "
        "to x = 140; separation = delta j_lat + dtheta j_head with j'' + K j = 0.",
        ["Declared domed coupon and chart extent x in [-60, 140], y in [-100, 100] mm",
         "Perturbations: 2 mm lateral, 5 mrad heading; remainder study eps = 0.4, 0.2, 0.1 mm",
         "Declared relative start-pose error of the offset tape for the open-loop uncertainty (1 sigma): insert and "
         "laying 0.05 mm / 0.5 mrad, and two seatings of the start jig at 0.01 mm / 0.1 mrad each"],
        "No observation. The protocol specifies CMM probing of the tape start poses and photogrammetry of coded targets "
        "on the tape centrelines at the stations; its hardware slot is empty.",
        "det Phi = 1; focal point independent of step size; linear remainder O(eps^2) (O(eps^3) on the symmetry axis).",
        "RK4 transfer at h = 1 and 2 mm, Richardson estimate, exactly perturbed routes (exp-map lateral start) at "
        "three sizes on and off the symmetry axis, crossing of the nonlinear 2 mm offset route, comparison with ciw's "
        "adaptive Dormand-Prince integrator and finite-difference geometry (always), and with scipy DOP853 and a sympy "
        "derivation (separate findings, established only when those modules are installed), protocol validation.",
        f"L = {length:.4f} mm; first focal point s_f = {focal:.3f} mm (s_f / L = {focal / length:.3f}); the 2 mm offset "
        f"route crosses at {study['nonlinear_crossing_mm']:.3f} mm and ends {study['lateral_nonlinear_mm'][-1]:.4f} mm "
        f"from the nominal route (plate: +2 mm; difference {plate_minus_coupon / combined:.1f} x the combined k = 2 "
        f"uncertainty); remainder orders {rem['on_axis']['order']:.2f} on axis, {rem['off_axis']['order']:.2f} off axis; "
        f"RK4 vs the ciw adaptive integrator: {integ['max_difference']:.1e}; Christoffel symbols vs finite differences: "
        f"{geom['max_christoffel_difference']:.1e}; curvature vs finite differences: {geom['max_curvature_difference']:.1e}.",
        "Solver error ~1e-6 (Richardson); geometry, start-pose and instrument terms are budgeted in T138 and T140.",
        ["step refinement of Phi and of the focal point", "Wronskian", "second- vs third-order remainder (symmetry)",
         "nonlinear crossing vs linear focal point", "plate signature vs combined uncertainty",
         "protocol refusal matrix, including the tape layout (the three tapes cross or come within 0.54 mm, so each is "
         "laid, measured and removed alone)",
         "circular test (a path drawn from its own program): paths are realized as unsteered tapes instead"],
        ["The formed coupon will not be an exact Gaussian; the T129 scan protocol (MFG-SCAN-01) fits the as-built dome "
         "and tests the Gaussian model by its residual, and a refuted model needs a prediction on the scanned surface.",
         "The tapes are assumed to follow geodesics; their realized start pose is measured and conditioned on (T138), "
         "while in-plane tape bending along the route is not modelled.",
         "Geometry independence (sympy) covers six points; elsewhere the geometry is checked against ciw finite differences only."],
        "Acquire MFG-SCAN-01 and MFG-COUPON-01 and bind the station-target and start-pose exports to T138 (ciw lab run "
        "T138 --capture photogrammetry=<targets.csv> --capture cmm=<start-pose.csv>), then retain the run with ciw lab "
        "hardware retain; the comparison is computational until a metrology instrument probe or a signed-capture trust "
        "anchor exists (deferred research question). Open: predict the separation on a surface fitted to the scan when "
        "the Gaussian model test rejects the as-built dome, which is not implemented.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T129 surface metrology ------------------------------------------------------------------
SCAN_NOISE = INSTRUMENTS["scanner"]["declared_standard_uncertainty_mm"]
NATIVE_SPACING = INSTRUMENTS["scanner"]["native_point_spacing_mm"]
MIN_SAMPLES = 21
# Registration targets of MFG-SCAN-01 (chart mm): off the dome, so none occludes the crest windows.
SCAN_TARGETS_U = ((-50.0, -80.0), (-50.0, 80.0), (40.0, -80.0), (40.0, 80.0), (130.0, -80.0), (130.0, 80.0))
# As-built dome identification: the global fit uses the scan decimated to a 4 mm grid; declared
# systematic terms are a relative scanner scale uncertainty (after the SG-200 check) and the
# target registration of the scan to the PART frame.
SCAN_GRID_MM = 4.0
SCAN_SCALE_U = 20e-6
SCAN_TRIALS = 200
SCAN_STANDOFF = {"standoff_mm": 100.0, "max_incidence_deg": 30.0, "status": "declared_not_verified"}
# Synthetic as-built dome for the recovery test (height, sigma, x0, y0, z0, tilt_x, tilt_y) and an
# elliptical dome within the declared width tolerance for the model test.
AS_BUILT_TRUTH = (10.15, 19.7, 0.4, -0.3, 0.05, 2e-4, -1e-4)
ELLIPTICAL_SIGMA_MM = (20.5, 19.5)
CHI2_Z = 3.09  # one-sided 99.9% normal quantile for the chi2 / dof model test


def crest_profile(x, height=geo.DOME_HEIGHT, sigma=geo.DOME_SIGMA):
    return height * np.exp(-np.asarray(x) ** 2 / (2.0 * sigma ** 2))


def cylinder_profile(x, radius=geo.CYLINDER_RADIUS):
    return np.sqrt(radius ** 2 - np.asarray(x) ** 2) - radius


PROFILE_SPECIMENS = {
    # name: (profile, signed vertex curvature z''(0), quartic coefficient z''''(0) / 24, tolerance)
    "coupon crest": (crest_profile, -geo.DOME_HEIGHT / geo.DOME_SIGMA ** 2,
                     geo.DOME_HEIGHT / (8 * geo.DOME_SIGMA ** 4), 0.05 * geo.DOME_HEIGHT / geo.DOME_SIGMA ** 2),
    "cylinder circumference": (cylinder_profile, -1.0 / geo.CYLINDER_RADIUS, -1.0 / (8 * geo.CYLINDER_RADIUS ** 3),
                               0.05 / geo.CYLINDER_RADIUS),
    "flat plate": (lambda x: np.zeros_like(np.asarray(x, dtype=float)), 0.0, 0.0, 1e-4),
}


def evaluate_design(profile, curvature, window, spacing, noise=SCAN_NOISE):
    """Exact bias (noise-free fit) and exact noise standard deviation of 2c for one design."""
    x = met.quadratic_design(window, spacing)
    return {"samples": len(x), "bias_per_mm": met.quadratic_curvature(x, profile(x)) - curvature,
            "noise_std_per_mm": met.curvature_noise_std(x, noise)}


def _discrete_quartic_bias(window, spacing, quartic):
    """Leading bias of 2c on the actual sample set: 2 q times the x^2 coefficient fitted to x^4."""
    x = met.quadratic_design(window, spacing)
    return 2.0 * quartic * float((np.linalg.pinv(np.column_stack([np.ones_like(x), x, x * x])) @ x ** 4)[2])


def metrology_study() -> dict:
    designs = {}
    for name, (profile, curvature, quartic, tolerance) in PROFILE_SPECIMENS.items():
        design = met.sampling_design(abs(curvature), abs(quartic), tolerance, SCAN_NOISE)
        spacing = min(design["max_spacing_mm"], design["window_mm"] / (MIN_SAMPLES - 1))
        exact = evaluate_design(profile, curvature, design["window_mm"], spacing)
        repeats = max(1, math.ceil(NATIVE_SPACING / design["max_spacing_mm"]))
        designs[name] = dict(design, used_spacing_mm=spacing, repeats_at_native_spacing=repeats, **exact,
                             worst_error_per_mm=abs(exact["bias_per_mm"]) + COVERAGE_K * exact["noise_std_per_mm"])
    profile, curvature, quartic, _ = PROFILE_SPECIMENS["coupon crest"]
    ladder = {}
    for relative in (0.05, 0.02, 0.01):
        design = met.sampling_design(abs(curvature), quartic, relative * abs(curvature), SCAN_NOISE)
        repeats = max(1, math.ceil(NATIVE_SPACING / design["max_spacing_mm"]))
        # Exact check of the repeat rule: m averaged scans at the native spacing, noise sigma / sqrt(m).
        exact = evaluate_design(profile, curvature, design["window_mm"], NATIVE_SPACING, SCAN_NOISE / math.sqrt(repeats))
        ladder[f"{relative:.0%}"] = {"window_mm": design["window_mm"], "max_spacing_mm": design["max_spacing_mm"],
                                     "repeats_at_native_spacing": repeats, "tolerance_per_mm": relative * abs(curvature),
                                     "exact_worst_error_per_mm": abs(exact["bias_per_mm"]) + COVERAGE_K * exact["noise_std_per_mm"]}
    # The osculating circle has quartic kappa^3 / 8; the Gaussian crest has h / (8 sigma^4), four times larger.
    tolerance = PROFILE_SPECIMENS["coupon crest"][3]
    circle = met.sampling_design(abs(curvature), abs(curvature) ** 3 / 8.0, tolerance, SCAN_NOISE)
    x_circle = met.quadratic_design(circle["window_mm"], circle["window_mm"] / 40)
    circle_bias = met.quadratic_curvature(x_circle, profile(x_circle)) - curvature
    # Monte Carlo on the coupon design, and a window sweep at its spacing.
    design = designs["coupon crest"]
    generator = met.rng(SEED)
    trials = 4000
    sweep = {}
    for factor in (1 / 3, 1 / 2, 1.0, 1.5, 2.0, 3.0):
        x = met.quadratic_design(factor * design["window_mm"], design["used_spacing_mm"])
        row = np.linalg.pinv(np.column_stack([np.ones_like(x), x, x * x]))[2]
        noisy = profile(x)[None, :] + generator.normal(0.0, SCAN_NOISE, (trials, len(x)))
        errors = 2.0 * noisy @ row - curvature
        exact = evaluate_design(profile, curvature, factor * design["window_mm"], design["used_spacing_mm"])
        sweep[f"{factor:.4g}"] = {"window_mm": factor * design["window_mm"], "samples": len(x),
                                   "mc_mean_error": float(errors.mean()), "mc_std": float(errors.std(ddof=1)),
                                   "mc_rms": float(math.sqrt(np.mean(errors ** 2))),
                                   "exact_bias": exact["bias_per_mm"], "exact_std": exact["noise_std_per_mm"],
                                   "fraction_within_tolerance": float(np.mean(np.abs(errors) <= tolerance))}
    # Averaging m aligned repeat scans divides the point noise by sqrt(m).
    x = met.quadratic_design(design["window_mm"], NATIVE_SPACING)
    row = np.linalg.pinv(np.column_stack([np.ones_like(x), x, x * x]))[2]
    repeats = 4
    single = 2.0 * (generator.normal(0.0, SCAN_NOISE, (trials, len(x))) @ row)
    averaged = 2.0 * (generator.normal(0.0, SCAN_NOISE, (trials, repeats, len(x))).mean(axis=1) @ row)
    repeat_ratio = float(averaged.std(ddof=1) / single.std(ddof=1))
    # Rigid registration of repeat scans on the six MFG-SCAN-01 targets.
    targets_u = list(SCAN_TARGETS_U)
    reference = geo.embed(geo.COUPON, targets_u)
    rotation = met.exp_so3(np.array([0.1, 0.2, 0.3]))
    translation = np.array([500.0, -200.0, 100.0])
    moved = reference @ rotation.T + translation
    registration_trials = 2000
    ssr, vectors = np.empty(registration_trials), np.empty((registration_trials, 3))
    for n in range(registration_trials):
        measured = moved + generator.normal(0.0, SCAN_NOISE, moved.shape)
        r_est, t_est = met.kabsch(reference, measured)
        ssr[n] = float(((reference @ r_est.T + t_est - measured) ** 2).sum())
        vectors[n] = met.rotation_vector(r_est @ rotation.T)
    dof = 3 * len(targets_u) - 6
    predicted_cov = met.registration_rotation_covariance(moved, SCAN_NOISE)
    empirical_cov = np.cov(vectors.T)
    lever = float(np.max(np.linalg.norm(moved - moved.mean(axis=0), axis=1)))
    return {"designs": designs, "ladder": ladder,
            "circle_rule": {"window_mm": circle["window_mm"], "actual_bias_per_mm": circle_bias,
                            "tolerance_per_mm": tolerance, "bias_over_tolerance": abs(circle_bias) / tolerance,
                            "quartic_ratio_gauss_over_circle": quartic / (abs(curvature) ** 3 / 8.0)},
            "series_bias": met.curvature_bias_series(design["window_mm"], quartic),
            "discrete_series_bias": _discrete_quartic_bias(design["window_mm"], design["used_spacing_mm"], quartic),
            "sweep": sweep, "trials": trials,
            "repeat": {"repeats": repeats, "std_ratio": repeat_ratio, "expected": 1 / math.sqrt(repeats)},
            "registration": {"targets": len(targets_u), "trials": registration_trials, "noise_mm": SCAN_NOISE,
                             "ssr_ratio": float(ssr.mean() / (SCAN_NOISE ** 2 * dof)),
                             "ssr_ratio_se": math.sqrt(2.0 / (registration_trials * dof)),
                             "rotation_cov_rel_frobenius": float(np.linalg.norm(empirical_cov - predicted_cov)
                                                                 / np.linalg.norm(predicted_cov)),
                             "rotation_std_urad": (1e6 * np.sqrt(np.diag(predicted_cov))).tolist(),
                             "lever_arm_mm": lever,
                             "lever_arm_error_mm": lever * math.sqrt(float(np.max(np.linalg.eigvalsh(predicted_cov))))}}


def _scan_grid():
    """Chart points of the decimated global-fit grid over the coupon extent."""
    xs = np.arange(geo.COUPON_X[0], geo.COUPON_X[1] + 0.5 * SCAN_GRID_MM, SCAN_GRID_MM)
    ys = np.arange(geo.COUPON_Y[0], geo.COUPON_Y[1] + 0.5 * SCAN_GRID_MM, SCAN_GRID_MM)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.ravel(), grid_y.ravel()


@functools.lru_cache(maxsize=1)
def as_built_study() -> dict:
    """Synthetic verification of the MFG-SCAN-01 dome fit and the scan-derived dome covariance (shared with T138, T140).

    The covariance over (height, sigma, x0, y0) that T138 and T140 use is the
    first-order white-noise covariance at the nominal dome on the 4 mm grid,
    plus the declared scanner scale term on height and width and the target
    registration term on the dome centre. It is valid only while the Gaussian
    model holds, which the chi2 / dof test of the protocol decides.
    """
    x, y = _scan_grid()
    names = met.DOME_PARAMETERS[:4]
    nominal = np.array([geo.DOME_HEIGHT, geo.DOME_SIGMA, 0.0, 0.0, 0.0, 0.0, 0.0])
    truth = np.array(AS_BUILT_TRUTH)
    dof = len(x) - len(met.DOME_PARAMETERS)
    threshold = 1.0 + CHI2_Z * math.sqrt(2.0 / dof)
    clean = met.dome_surface(truth, x, y)
    recovered, _, _ = met.fit_dome(x, y, clean, nominal)
    jac = met.dome_jacobian(truth, x, y)
    linear_std = SCAN_NOISE * np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)))
    generator = met.rng(SEED + 9)
    fits, chi2 = np.empty((SCAN_TRIALS, len(truth))), np.empty(SCAN_TRIALS)
    for n in range(SCAN_TRIALS):
        params, _, residual = met.fit_dome(x, y, clean + generator.normal(0.0, SCAN_NOISE, len(x)), nominal)
        fits[n], chi2[n] = params, float(residual @ residual) / (SCAN_NOISE ** 2 * dof)
    mc_std = fits.std(axis=0, ddof=1)
    # Model test: an elliptical dome inside the declared width tolerance is not a Gaussian of revolution.
    sx, sy = ELLIPTICAL_SIGMA_MM
    elliptical = geo.DOME_HEIGHT * np.exp(-x ** 2 / (2.0 * sx ** 2) - y ** 2 / (2.0 * sy ** 2))
    elliptical_fit, _, residual = met.fit_dome(x, y, elliptical + generator.normal(0.0, SCAN_NOISE, len(x)), nominal)
    elliptical_chi2 = float(residual @ residual) / (SCAN_NOISE ** 2 * dof)
    # Scan-derived covariance at the nominal dome: white noise + declared scale + target registration.
    jac_nominal = met.dome_jacobian(nominal, x, y)
    white = SCAN_NOISE ** 2 * np.linalg.inv(jac_nominal.T @ jac_nominal)[:4, :4]
    shape = np.array([geo.DOME_HEIGHT, geo.DOME_SIGMA, 0.0, 0.0])
    scale = SCAN_SCALE_U ** 2 * np.outer(shape, shape)
    targets = geo.embed(geo.COUPON, SCAN_TARGETS_U)
    lever = geo.COUPON.embedding(np.zeros(2)) - targets.mean(axis=0)
    centre = (SCAN_NOISE ** 2 / len(SCAN_TARGETS_U) * np.eye(3)
              + met.hat(lever) @ met.registration_rotation_covariance(targets, SCAN_NOISE) @ met.hat(lever).T)
    registration = np.zeros((4, 4))
    registration[2:, 2:] = centre[:2, :2]
    covariance = white + scale + registration
    std = np.sqrt(np.diag(covariance))
    return {"grid_mm": SCAN_GRID_MM, "points": int(len(x)), "dof": int(dof), "trials": SCAN_TRIALS,
            "truth": truth.tolist(), "recovered": recovered.tolist(),
            "recovery_error": float(np.max(np.abs(recovered - truth))),
            "linear_std": linear_std.tolist(), "mc_std": mc_std.tolist(), "mc_mean": fits.mean(axis=0).tolist(),
            "std_rel_difference": float(np.max(np.abs(mc_std[:4] / linear_std[:4] - 1.0))),
            "std_abs_difference_mm": float(np.max(np.abs(mc_std[:4] - linear_std[:4]))),
            "bias_in_se": float(np.max(np.abs(fits.mean(axis=0)[:4] - truth[:4]) / (linear_std[:4] / math.sqrt(SCAN_TRIALS)))),
            "chi2_mean": float(chi2.mean()), "chi2_threshold": threshold,
            "false_alarm_fraction": float(np.mean(chi2 > threshold)),
            "elliptical_sigma_mm": list(ELLIPTICAL_SIGMA_MM), "elliptical_chi2": elliptical_chi2,
            "elliptical_fit": elliptical_fit.tolist(),
            "parameters": list(names), "covariance_mm2": covariance.tolist(),
            "standard_uncertainty_mm": {name: float(value) for name, value in zip(names, std)},
            "components_mm": {"white_noise": dict(zip(names, np.sqrt(np.diag(white)).tolist())),
                              "scale": dict(zip(names, np.sqrt(np.diag(scale)).tolist())),
                              "registration": dict(zip(names, np.sqrt(np.diag(registration)).tolist()))},
            "correlation_height_sigma": float(covariance[0, 1] / (std[0] * std[1])),
            "declared": {"noise_mm": SCAN_NOISE, "scale_relative": SCAN_SCALE_U, "targets_u_mm": [list(t) for t in SCAN_TARGETS_U],
                         "status": "declared_not_verified"}}


def scan_protocol(f_design, f_fit, f_model, scan, crest) -> dict:
    """MFG-SCAN-01: the surface metrology protocol that identifies the as-built coupon for T128, T138 and T140."""
    max_slope_deg = math.degrees(math.atan(geo.DOME_HEIGHT / geo.DOME_SIGMA * math.exp(-0.5)))
    predicted = [
        _predicted("D1", "scan-derived standard uncertainty of the fitted dome height, width and centre (4 mm grid)",
                   _r(scan["standard_uncertainty_mm"]), "mm", f_fit),
        _predicted("D2", "chi2 / dof threshold of the Gaussian model test (99.9%, independent point noise)",
                   _r(scan["chi2_threshold"]), "1", f_model),
        _predicted("C1", "crest curvature window and maximum spacing for 5% at k = 2",
                   _r({"window_mm": crest["window_mm"], "spacing_mm": crest["used_spacing_mm"]}), "mm", f_design)]
    criteria = [
        {"id": "H1", "status": "hypothesis", "statement": "The formed coupon is a Gaussian dome of revolution within the "
         "scanner noise", "test": f"chi2 / dof of the global fit residual <= {scan['chi2_threshold']:.3f}"},
        {"id": "H2", "status": "hypothesis", "statement": "The fitted height and width lie within the declared forming "
         "tolerances (10 +/- 0.2 mm, 20 +/- 0.5 mm)", "test": "|fitted - nominal| + 2 u <= tolerance, u from D1"},
        {"id": "H3", "status": "hypothesis", "statement": "The crest curvature of the local-window fit equals h / sigma^2 of "
         "the fitted dome", "test": "E_n <= 1 with U from the T129 sampling design and the fit covariance"}]
    plan = {"scan_plan": {
        "instrument_setup": dict(SCAN_STANDOFF, scanner="laser line scanner (declared 0.01 mm point noise, 0.05 mm native "
                                                         "spacing)", coupon_max_slope_deg=max_slope_deg,
                                 passes="two orthogonal raster passes over x in [-60, 140], y in [-100, 100] mm"),
        "targets": "six sphere-mounted targets off the dome, measured by the CMM in the PART frame and by the scanner",
        "sampling": {"global_fit_grid_mm": SCAN_GRID_MM,
                     "crest_windows": {"window_mm": crest["window_mm"], "max_spacing_mm": crest["used_spacing_mm"]}},
        "filtering": "reject points above the incidence limit or flagged by the scanner; no smoothing before either fit; "
                     "reject fit residuals beyond 5 sigma and report their count",
        "surface_fit": {"model": "z = z0 + a_x x + a_y y + h exp(-((x - x0)^2 + (y - y0)^2) / (2 sigma^2))",
                        "parameters": list(met.DOME_PARAMETERS), "method": "Gauss-Newton least squares from the nominal dome",
                        "covariance": "sigma^2 (J^T J)^-1 plus the declared scanner scale term on height and width and "
                                      "the target registration term on the centre",
                        "model_test": "chi2 / dof against the 99.9% threshold (H1)",
                        "consumers": ["MFG-COUPON-01 (T128): the as-built coupon under the tapes",
                                      "T138 scan-conditioned U_p", "T140 scan-conditioned geometry terms"]},
        "retained_raw_data": ["native point clouds of both passes with scanner settings and SHA-256 digests",
                              "target coordinates from the CMM and the scanner", "registration transforms and residuals",
                              "fit parameters, covariance, residual map and outlier count"]}}
    return build_protocol(
        "T129", "MFG-SCAN-01", "Surface metrology: as-built scan of the domed coupon",
        "Identify the as-built dome (height, width, centre) with its covariance and test the Gaussian model before the "
        "coupon separations of MFG-COUPON-01 are predicted and compared.",
        {"kind": "domed coupon", "surface_model": geo.COUPON.describe(),
         "chart_extent_mm": {"x": list(geo.COUPON_X), "y": list(geo.COUPON_Y)},
         "declared_height_tolerance_mm": 0.2, "declared_sigma_tolerance_mm": 0.5},
        [{"id": f"RT{k + 1}", "u_mm": list(u), "kind": "sphere-mounted registration target"} for k, u in enumerate(SCAN_TARGETS_U)],
        [{"id": "RASTER-X", "kind": "scan pass", "direction": "+x", "uses": ["scanner"]},
         {"id": "RASTER-Y", "kind": "scan pass", "direction": "+y", "uses": ["scanner"]}],
        predicted, criteria, ("scanner", "cmm"),
        [_step("Measure the six registration targets RT1-RT6 with the CMM in the PART frame and export them as the cmm "
               "capture.", ["cmm"], ["cmm capture (ciw.lab-mfg-target-capture.v1)"], key="targets"),
         _step(f"Set the scanner to the declared {SCAN_STANDOFF['standoff_mm']:g} mm standoff; scan both raster passes "
               f"at the native spacing, rejecting points with incidence above {SCAN_STANDOFF['max_incidence_deg']:g} deg.",
               ["scanner"], ["native point cloud of each pass with the scanner settings"], key="scan"),
         _repeat(["locate", "targets", "scan"], "FX-321", "replicates of the target measurement and both raster passes"),
         _step("Register each pass of every replicate to the targets (Kabsch) and retain the transform and its "
               "residual.", [],
               ["registration transform and residual per pass"],
               check="registration residual consistent with chi-square(3N - 6) at the declared noise"),
         _step("Fit the as-built dome on the 4 mm grid; fit the crest curvature in the T129 windows at full density.", [],
               ["fit parameters, covariance, residual map and outlier count"]),
         _step("Report the fitted parameters with their covariance and the chi2 / dof model test (H1) before any tape is "
               "laid.", [], ["as-built dome record for MFG-COUPON-01, T138 and T140"])],
        [capture_format("cmm", TARGET_CAPTURE, "cmm", [f"RT{k + 1}" for k in range(len(SCAN_TARGETS_U))],
                        "ciw.lab.manufacturing_records.read_capture; no task reads it yet"),
         {"role": "scanner", "schema": "native point cloud per pass: x_mm, y_mm, z_mm, incidence_deg, flag",
          "media_type": "text/csv", "instrument": "scanner", "frame": "scanner, registered to PART by the targets",
          "columns": ["x_mm", "y_mm", "z_mm", "incidence_deg", "flag"], "ids": ["RASTER-X", "RASTER-Y"],
          "reader": "none yet: T129 fits synthetic scans only; a reader that runs fit_dome on a bound scan is open work"}],
        extra=plan)


@_task("T129", ("test_metrology_sampling_design_and_counterexamples", "test_scan_protocol_and_as_built_fit",
                "test_protocols_refuse_filled_slots_and_decisions"))
def surface_metrology(ctx):
    study = ctx.memo("mfg.metrology", metrology_study)
    designs, sweep, crest = study["designs"], study["sweep"], study["designs"]["coupon crest"]
    unit = sweep["1"]
    se_std = unit["exact_std"] / math.sqrt(2 * (study["trials"] - 1))
    f_design = finding("Sampling design resolving profile curvature to tolerance at k = 2", "numerical",
                       {name: {"window_mm": d["window_mm"], "spacing_mm": d["used_spacing_mm"], "samples": d["samples"],
                               "repeats": d["repeats_at_native_spacing"], "tolerance_per_mm": d["tolerance_per_mm"]}
                        for name, d in designs.items()},
                       {"derivation": "bias (3/7) q W^2 and noise sigma sqrt(720 d) W^(-5/2) of a quadratic fit (docs/lab/MANUFACTURING.md#t129)",
                        "checks": [_check("analytic", f"exact |bias| + 2 std minus tolerance, {name} (1/mm)",
                                          d["worst_error_per_mm"] - d["tolerance_per_mm"], 0.0, "signed_le")
                                   for name, d in designs.items()]},
                       uncertainty=_u("roundoff", 1e-15, "exact discrete normal equations for each design"),
                       tolerance={"abs": 1e-9, "rel": 1e-9})
    f_mc = finding("Monte Carlo curvature error on the coupon crest matches the derived bias and noise", "numerical",
                   {"exact_bias": unit["exact_bias"], "mc_mean_error": unit["mc_mean_error"], "exact_std": unit["exact_std"],
                    "mc_std": unit["mc_std"], "fraction_within_tolerance": unit["fraction_within_tolerance"]},
                   {"generator": _generator("gaussian scanner noise", noise_mm=SCAN_NOISE, trials=study["trials"]),
                    "checks": [_check("analytic", "(MC mean - exact bias) / (4 std / sqrt(n))",
                                      (unit["mc_mean_error"] - unit["exact_bias"]) / (4 * unit["exact_std"] / math.sqrt(study["trials"])), 1.0),
                               _check("analytic", "(MC std - exact std) / (4 standard errors)", (unit["mc_std"] - unit["exact_std"]) / (4 * se_std), 1.0),
                               _check("analytic", "exact noise-free bias vs leading quartic bias on the sample set (relative)",
                                      (unit["exact_bias"] - study["discrete_series_bias"]) / study["discrete_series_bias"], 0.05),
                               _check("analytic", "fraction of trials within tolerance", unit["fraction_within_tolerance"], 0.95, "ge"),
                               _check("analytic", "(averaged-repeat std ratio - 1/sqrt(4)) / 0.05",
                                      (study["repeat"]["std_ratio"] - study["repeat"]["expected"]) / 0.05, 1.0)]},
                   unit="1/mm",
                   uncertainty=_u("monte_carlo_95ci", 1.96 * unit["exact_std"] / math.sqrt(study["trials"]),
                                  "95% half-width of the MC mean error (1/mm)"),
                   tolerance={"abs": 1e-9, "rel": 1e-6})
    circle = study["circle_rule"]
    f_circle = finding("The osculating-circle window rule under-predicts the coupon-crest curvature bias", "numerical",
                       circle["bias_over_tolerance"],
                       {"checks": [_check("analytic", "actual noise-free bias at the circle-rule window / tolerance",
                                          circle["bias_over_tolerance"], 1.0, "ge")]},
                       uncertainty=_u("roundoff", 1e-12, "noise-free exact least-squares fit"),
                       tolerance={"abs": 1e-9, "rel": 1e-7},
                       counterexample={"statement": "A window chosen from the osculating circle keeps the quadratic-fit "
                                                    "curvature bias within tolerance on any convex profile",
                                       "witness": {"profile": "Gaussian crest h = 10 mm, sigma = 20 mm",
                                                   "window_mm": circle["window_mm"],
                                                   "quartic_ratio_gauss_over_circle": circle["quartic_ratio_gauss_over_circle"],
                                                   "bias_per_mm": circle["actual_bias_per_mm"]}})
    small = sweep[f"{1 / 3:.4g}"]
    f_window = finding("A smaller fitting window at fixed spacing can make the curvature estimate worse", "numerical",
                       {"rms_at_design_window": unit["mc_rms"], "rms_at_one_third_window": small["mc_rms"]},
                       {"generator": _generator("gaussian scanner noise", noise_mm=SCAN_NOISE),
                        "checks": [_check("analytic", "MC RMS error ratio (W/3 over W)", small["mc_rms"] / unit["mc_rms"], 2.0, "ge")]},
                       unit="1/mm",
                       uncertainty=_u("monte_carlo_95ci", 1.96 * small["mc_rms"] / math.sqrt(2 * study["trials"]),
                                      "95% half-width of the MC RMS at W/3 (1/mm)"),
                       tolerance={"abs": 1e-9, "rel": 1e-6},
                       counterexample={"statement": "Sampling a smaller neighbourhood (more local fit) always improves the "
                                                    "curvature estimate",
                                       "witness": {"window_mm": small["window_mm"], "spacing_mm": crest["used_spacing_mm"],
                                                   "rms_error_per_mm": small["mc_rms"]}})
    reg = study["registration"]
    f_reg = finding("Registration residual and rotation covariance of repeat scans follow chi-square(3N - 6) and "
                    "the inertia formula", "numerical",
                    {"ssr_ratio": reg["ssr_ratio"], "rotation_std_urad": reg["rotation_std_urad"],
                     "lever_arm_error_mm": reg["lever_arm_error_mm"]},
                    {"generator": _generator("kabsch registration trials", targets=reg["targets"], trials=reg["trials"]),
                     "checks": [_check("analytic", "(mean SSR / sigma^2 (3N - 6) - 1) / (4 SE)", (reg["ssr_ratio"] - 1) / (4 * reg["ssr_ratio_se"]), 1.0),
                                _check("analytic", "relative Frobenius difference of rotation covariance (MC vs inertia formula)",
                                       reg["rotation_cov_rel_frobenius"], 0.1)]},
                    uncertainty=_u("monte_carlo_95ci", 1.96 * reg["ssr_ratio_se"], "95% half-width of the mean SSR ratio"),
                    tolerance={"abs": 1e-9, "rel": 1e-6})
    f_ladder = finding("Repeat scans needed at the native 0.05 mm spacing versus curvature tolerance (coupon crest)",
                       "numerical", {k: v["repeats_at_native_spacing"] for k, v in study["ladder"].items()},
                       {"derivation": "m = ceil(native spacing / required spacing): averaging m scans divides noise by sqrt(m)",
                        "checks": [_check("analytic", f"exact |bias| + 2 std with m averaged scans minus tolerance, {k} (1/mm)",
                                          v["exact_worst_error_per_mm"] - v["tolerance_per_mm"], 0.0, "signed_le")
                                   for k, v in study["ladder"].items()]},
                       unit="scans", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    scan = ctx.memo("mfg.as-built", as_built_study)
    scan_generator = _generator("synthetic coupon scans", seed=SEED + 9, grid_mm=SCAN_GRID_MM, noise_mm=SCAN_NOISE,
                                trials=SCAN_TRIALS)
    u_scan = scan["standard_uncertainty_mm"]
    f_fit = finding("The as-built dome fit (height, width, centre, base plane) recovers a perturbed dome and has its "
                    "linearized covariance on synthetic scans", "numerical",
                    {"scan_standard_uncertainty_mm": u_scan, "correlation_height_sigma": scan["correlation_height_sigma"],
                     "white_noise_std_mm": scan["components_mm"]["white_noise"], "points": scan["points"]},
                    {"generator": scan_generator,
                     "checks": [_check("analytic", "noise-free recovery of a perturbed as-built dome from the nominal start "
                                       "(max |parameter error|)", scan["recovery_error"], 1e-8),
                                _check("analytic", "max over height, width and centre of |MC std / linearized std - 1| "
                                       "(tolerance 4 standard errors, 4 / sqrt(2 (n - 1)))", scan["std_rel_difference"],
                                       4.0 / math.sqrt(2.0 * (SCAN_TRIALS - 1))),
                                _check("analytic", "max over height, width and centre of |MC mean - truth| in standard "
                                       "errors of the mean", scan["bias_in_se"], 4.0, "le"),
                                _check("analytic", "(mean chi2 / dof of the Gaussian scans - 1) / (4 SE)",
                                       (scan["chi2_mean"] - 1.0) / (4.0 * math.sqrt(2.0 / (scan["dof"] * SCAN_TRIALS))), 1.0)]},
                    unit="mm",
                    uncertainty=_u("reference_error", scan["std_abs_difference_mm"],
                                   "max |MC std - linearized std| over height, width and centre (mm); MC sampling bounded"),
                    tolerance={"abs": 1e-9, "rel": 1e-6})
    f_model = finding("The fit residual flags an elliptical as-built dome inside the declared width tolerance "
                      "(chi2 model test)", "numerical",
                      {"elliptical_chi2_per_dof": scan["elliptical_chi2"], "threshold": scan["chi2_threshold"],
                       "false_alarm_fraction": scan["false_alarm_fraction"]},
                      {"generator": scan_generator,
                       "checks": [_check("analytic", "chi2 / dof of the elliptical dome (sigma_x 20.5, sigma_y 19.5 mm) "
                                         "over the 99.9% threshold", scan["elliptical_chi2"] / scan["chi2_threshold"], 1.0, "ge"),
                                  _check("analytic", "fraction of Gaussian scans the threshold flags (false alarms)",
                                         scan["false_alarm_fraction"], 0.02, "le")]},
                      uncertainty=_u("monte_carlo_95ci",
                                     1.96 * math.sqrt((2.0 + 4.0 * max(scan["elliptical_chi2"] - 1.0, 0.0)) / scan["dof"]),
                                     "95% half-width of the noise spread of the elliptical chi2 / dof (non-central chi-square)"),
                      tolerance={"abs": 1e-6, "rel": 1e-6},
                      counterexample={"statement": "A dome inside the declared forming tolerances is described by the "
                                                   "nominal Gaussian model, so its parameters alone fix the prediction",
                                      "witness": {"sigma_x_mm": ELLIPTICAL_SIGMA_MM[0], "sigma_y_mm": ELLIPTICAL_SIGMA_MM[1],
                                                  "chi2_per_dof": scan["elliptical_chi2"],
                                                  "threshold": scan["chi2_threshold"]}})
    protocol = scan_protocol(f_design, f_fit, f_model, scan, crest)
    ctx.artifact_json("protocol-surface-scan.json", protocol)
    ctx.artifact_json("as-built-fit.json", _r(scan))
    ctx.artifact_json("metrology-design.json", _r({k: study[k] for k in ("designs", "ladder", "circle_rule", "sweep", "repeat", "registration")}))
    windows = [v["window_mm"] for v in sweep.values()]
    ctx.artifact_text("curvature-window-sweep.svg", svg.line_plot(
        [("MC RMS error", windows, [v["mc_rms"] for v in sweep.values()]),
         ("exact sqrt(bias^2 + std^2)", windows, [math.hypot(v["exact_bias"], v["exact_std"]) for v in sweep.values()]),
         ("tolerance", [windows[0], windows[-1]], [crest["tolerance_per_mm"]] * 2)],
        title="Coupon crest: curvature error vs fitting window", xlabel="window W (mm)", ylabel="error (1/mm)", logy=True))
    findings = [f_design, f_mc, f_circle, f_window, f_reg, f_ladder, f_fit, f_model,
                _protocol_findings(protocol, "Surface-scan protocol record validates and refuses malformed variants"),
                _not_measured("A real laser line scanner achieves the declared 0.01 mm point noise on the coupon surface "
                              "(finish, incidence angle, speckle)", "sensor_performance"),
                _not_measured("The formed coupon passes the Gaussian model test and its fitted height and width lie within "
                              "the declared forming tolerances"),
                _not_measured("The declared scanner scale (20 ppm) and target registration uncertainties hold for the scan",
                              "calibration")]
    ladder = study["ladder"]
    fields = _fields(
        "A surface metrology protocol for the coupon needs two things: the sampling that resolves curvature (a local "
        "quadratic fit whose bias is set by the quartic profile term and whose noise grows as the window shrinks, so "
        "the window must come from the actual profile, not from its osculating circle), and an identification of the "
        "as-built dome (height, width, centre) with a covariance that can replace the declared forming tolerances in "
        "T138 and T140, valid only while a residual test accepts the Gaussian model.",
        "Local fit: least-squares z = a + b x + c x^2 over a centred window W with spacing d: bias(2c) = (3/7) q W^2, "
        "std(2c) = sigma sqrt(720 d) W^(-5/2) (large-n); tolerance split half bias, half k = 2 noise. "
        "Global fit: z = z0 + a_x x + a_y y + h exp(-((x - x0)^2 + (y - y0)^2) / (2 sigma^2)) by Gauss-Newton; "
        "covariance sigma^2 (J^T J)^-1 plus declared scale and registration terms; model test chi2 / dof <= "
        "1 + 3.09 sqrt(2 / dof). Registration: Kabsch; E[SSR] = sigma^2 (3N - 6); rotation covariance "
        "sigma^2 (sum |p|^2 I - p p^T)^-1.",
        ["Declared scanner noise 0.01 mm per point, native spacing 0.05 mm, relative scale uncertainty 20 ppm, standoff "
         "100 mm, incidence limit 30 deg",
         "Profiles: coupon crest (Gaussian), cylinder circumference (R = 100 mm), flat plate",
         "Six registration targets off the dome; global fit grid 4 mm over the coupon extent",
         f"Seeded synthetic noise (PCG64 seeds {SEED} and {SEED + 9}), 4000 fits per window, 2000 registrations, "
         f"{SCAN_TRIALS} synthetic coupon scans"],
        "Synthetic scans only; no scanner was used. MFG-SCAN-01 specifies the scan, its registration, the fits and the "
        "retained raw data; its hardware-evidence slot is empty.",
        "Exact design error <= tolerance; MC mean and spread equal the exact bias and noise; registration residual "
        "has 3N - 6 degrees of freedom; the dome fit is exact without noise, its MC spread equals the linearized "
        "covariance, and the model test flags a non-Gaussian dome.",
        "Derive the local-fit design, evaluate it exactly (discrete normal equations), verify by Monte Carlo, sweep the "
        "window at fixed spacing, compare the circle rule with the actual profile, simulate repeat-scan registration; fit "
        "a perturbed as-built dome without noise and on seeded synthetic scans, test an elliptical dome against the chi2 "
        "threshold, assemble the scan-derived dome covariance, and build, validate and mutate the MFG-SCAN-01 protocol.",
        f"Coupon crest at 5%: window {crest['window_mm']:.2f} mm, spacing {crest['used_spacing_mm']:.3f} mm; repeats at "
        f"0.05 mm native spacing: 5% -> {ladder['5%']['repeats_at_native_spacing']}, 2% -> {ladder['2%']['repeats_at_native_spacing']}, "
        f"1% -> {ladder['1%']['repeats_at_native_spacing']}; circle rule gives bias {circle['bias_over_tolerance']:.2f} x tolerance. "
        f"As-built fit on {scan['points']} points: scan-derived standard uncertainty height {u_scan['height_mm']:.2g} mm, "
        f"width {u_scan['sigma_mm']:.2g} mm, centre ({u_scan['x0_mm']:.2g}, {u_scan['y0_mm']:.2g}) mm (declared tolerances: "
        f"{0.2 / math.sqrt(3.0):.3f} and {0.5 / math.sqrt(3.0):.3f} mm); the elliptical dome gives chi2 / dof "
        f"{scan['elliptical_chi2']:.1f} against a threshold of {scan['chi2_threshold']:.3f}.",
        "Monte Carlo standard errors are stated in the checks; the design rules are exact for the declared profiles; the "
        "scan-derived covariance is first order and holds only under the Gaussian model and the declared scanner terms.",
        ["circle-rule window on a non-circular crest", "window too small (noise) and too large (bias)",
         "continuum vs discrete noise formula", "repeat averaging", "registration residual degrees of freedom",
         "non-Gaussian (elliptical) as-built dome flagged by the residual test", "false alarms of the model test",
         "scan protocol refusal matrix"],
        ["Surface finish, incidence angle and scanner nonlinearity are not modelled; noise is white and Gaussian.",
         "Profiles are taken through the principal direction at the vertex; slope correction is neglected.",
         "At 1% tolerance a quadratic fit needs about 100 repeat scans: a higher-order local fit is a deferred research question.",
         "If the model test rejects the Gaussian dome, the scan-derived covariance does not apply and the prediction must "
         "be re-integrated on a surface fitted to the scan (not implemented)."],
        "Open: a reader of the MFG-SCAN-01 point-cloud format that runs fit_dome and the chi2 model test on a bound "
        "scanner capture (none exists; T129 fits synthetic scans only), a prediction on a surface fitted to the scan "
        "for an as-built dome the model test rejects, and a higher-order local fit for 1% curvature tolerance "
        "(deferred research questions).")
    return {"state": "completed", "fields": fields, "findings": findings}


# T130 calibration artifacts and datum frames ---------------------------------------------
DATUM_POINTS = {"A": [[10.0, 10.0, 0.0], [190.0, 10.0, 0.0], [100.0, 190.0, 0.0]],
                "B": [[20.0, 0.0, 5.0], [180.0, 0.0, 5.0]], "C": [0.0, 100.0, 5.0]}
TRACKER_NOISE = INSTRUMENTS["tracker"]["declared_standard_uncertainty_mm"]
CMM_NOISE = INSTRUMENTS["cmm"]["declared_standard_uncertainty_mm"]


def _metrology_refusal(function, *args):
    """Run ``function`` and return the MetrologyRefusal code it raised, or None if it returned."""
    try:
        function(*args)
    except met.MetrologyRefusal as exc:
        return exc.code
    return None


def _datum_measure(pose, noise=None, generator=None):
    points = {k: np.atleast_2d(np.asarray(v, dtype=float)) for k, v in DATUM_POINTS.items()}
    out = {}
    for key, value in points.items():
        moved = value @ pose[:3, :3].T + pose[:3, 3]
        if generator is not None:
            moved = moved + generator.normal(0.0, noise, moved.shape)
        out[key] = moved
    return met.datum_frame_321(out["A"], out["B"], out["C"][0])


def artifacts_study() -> dict:
    generator = met.rng(SEED + 1)
    # Gauge sphere probed on a 75 degree cap.
    centre, radius = np.array([12.0, -4.0, 30.0]), 12.7
    nominal_points = met.cap_points(centre, radius, 25, math.radians(75.0))
    _, _, jac = met.sphere_fit(nominal_points)
    linear_std = CMM_NOISE * np.sqrt(np.diag(np.linalg.inv(jac.T @ jac)))
    fits = np.array([np.append(*met.sphere_fit(met.cap_points(centre, radius, 25, math.radians(75.0), generator, CMM_NOISE))[:2])
                     for _ in range(1000)])
    sphere = {"linear_std_mm": linear_std.tolist(), "mc_std_mm": fits.std(axis=0, ddof=1).tolist(),
              "mc_bias_mm": (fits.mean(axis=0) - np.append(centre, radius)).tolist(), "trials": 1000}
    # Step gauge with a declared scale error and offset.
    nominal = 10.0 * np.arange(1, 21)
    scale, offset, noise = 50e-6, 0.001, 0.0005
    solution, unscaled = met.step_gauge_fit(nominal, (1 + scale) * nominal + offset)
    design = np.column_stack([nominal, np.ones_like(nominal)])
    pinv = np.linalg.pinv(design)
    samples = ((1 + scale) * nominal + offset)[None, :] + generator.normal(0.0, noise, (2000, len(nominal)))
    estimates = (samples - nominal[None, :]) @ pinv.T
    step = {"noise_free_recovery": [float(solution[0] - scale), float(solution[1] - offset)],
            "linear_std": (noise * np.sqrt(np.diag(unscaled))).tolist(), "mc_std": estimates.std(axis=0, ddof=1).tolist(),
            "mc_mean_minus_truth": (estimates.mean(axis=0) - np.array([scale, offset])).tolist()}
    # 3-2-1 datum frame: orthonormality, equivariance and covariance.
    part_pose = met.transform(met.exp_so3([0.0, 0.0, math.radians(10.0)]), [50.0, 30.0, 20.0])
    frame = _datum_measure(part_pose)
    orthonormal = float(np.max(np.abs(frame[:3, :3].T @ frame[:3, :3] - np.eye(3))))
    motion = met.transform(met.exp_so3([0.3, -0.2, 0.5]), [100.0, -50.0, 25.0])
    equivariance = float(np.max(np.abs(_datum_measure(motion @ part_pose) - motion @ frame)))
    recovery = float(np.max(np.abs(frame - part_pose)))
    flat = np.concatenate([np.atleast_2d(DATUM_POINTS[k]) for k in ("A", "B", "C")])
    moved = flat @ part_pose[:3, :3].T + part_pose[:3, 3]

    def frame_of(points):
        return met.datum_frame_321(points[:3], points[3:5], points[5])

    jac = np.zeros((6, flat.size))
    step_size = 1e-5
    for index in range(flat.size):
        plus, minus = moved.copy(), moved.copy()
        plus.flat[index] += step_size
        minus.flat[index] -= step_size
        jac[:, index] = (met.pose_difference(frame_of(plus), frame) - met.pose_difference(frame_of(minus), frame)) / (2 * step_size)
    datum_cov = TRACKER_NOISE ** 2 * jac @ jac.T
    samples = np.array([met.pose_difference(frame_of(moved + generator.normal(0.0, TRACKER_NOISE, moved.shape)), frame)
                        for _ in range(3000)])
    datum_rel = float(np.linalg.norm(np.cov(samples.T) - datum_cov) / np.linalg.norm(datum_cov))
    # Degenerate datum features are refused: collinear primary points, and a secondary direction
    # normal to the primary plane (its projection into A vanishes).
    refusals = {"collinear_primary": _metrology_refusal(
                    met.datum_frame_321, [[10.0, 10.0, 0.0], [100.0, 10.0, 0.0], [190.0, 10.0, 0.0]],
                    DATUM_POINTS["B"], DATUM_POINTS["C"]),
                "secondary_normal_to_primary": _metrology_refusal(
                    met.datum_frame_321, DATUM_POINTS["A"], [[20.0, 0.0, 5.0], [20.0, 0.0, 25.0]], DATUM_POINTS["C"])}
    # Frame chain INSTRUMENT -> WORLD -> FIXTURE -> PART -> CAD.
    def diag(translation, rotation):
        return np.diag([translation ** 2] * 3 + [rotation ** 2] * 3)

    links = [(met.transform(met.exp_so3([0.0, 0.0, math.radians(30.0)]), [-2500.0, 800.0, -1200.0]), diag(0.010, 5e-6)),
             (met.transform(met.exp_so3([math.radians(5.0), 0.0, math.radians(-20.0)]), [1200.0, 300.0, 900.0]), diag(0.020, 20e-6)),
             (part_pose, datum_cov),
             (met.transform(np.eye(3), [0.0, 0.0, -2.0]), diag(0.005, 10e-6))]
    names = ["INSTRUMENT->WORLD", "WORLD->FIXTURE", "FIXTURE->PART", "PART->CAD"]
    pose, chain_cov = met.compose_chain(links)
    chain = {}
    for label, point in (("cad_origin", np.zeros(3)), ("coupon_far_corner", np.array([200.0, 200.0, 10.0]))):
        linear = met.point_covariance(pose, chain_cov, point)
        mc = met.sample_chain_point(links, point, generator, 3000)
        # Linearization error of the first-order std: sigma points through the exact SE(3) products.
        sigma_std = np.sqrt(np.diag(met.sigma_point_chain_covariance(links, point)))
        chain[label] = {"linear_std_mm": np.sqrt(np.diag(linear)).tolist(), "mc_std_mm": mc.std(axis=0, ddof=1).tolist(),
                        "sigma_point_std_mm": sigma_std.tolist(),
                        "linearization_mm": float(np.max(np.abs(sigma_std - np.sqrt(np.diag(linear))))),
                        "rel_frobenius": float(np.linalg.norm(np.cov(mc.T) - linear) / np.linalg.norm(linear))}
    contributions = {}
    for k, name in enumerate(names):
        only = [(p, c if j == k else np.zeros((6, 6))) for j, (p, c) in enumerate(links)]
        pose_k, cov_k = met.compose_chain(only)
        contributions[name] = float(np.sqrt(np.trace(met.point_covariance(pose_k, cov_k, np.array([200.0, 200.0, 10.0])))))
    return {"sphere": sphere, "step_gauge": step, "datum": {"orthonormality": orthonormal, "equivariance": equivariance,
                                                            "recovery": recovery, "covariance_rel_frobenius": datum_rel,
                                                            "std_translation_mm": np.sqrt(np.diag(datum_cov)[:3]).tolist(),
                                                            "std_rotation_urad": (1e6 * np.sqrt(np.diag(datum_cov)[3:])).tolist(),
                                                            "refusals": refusals},
            "chain": chain, "contributions_rss_mm": contributions,
            "chain_linearization_mm": max(v["linearization_mm"] for v in chain.values()),
            "links": [{"parent_child": n, "rotation": p[:3, :3].tolist(), "translation_mm": p[:3, 3].tolist(),
                       "covariance": c.tolist()} for n, (p, c) in zip(names, links)]}


@_task("T130", ("test_artifacts_datum_frames_and_chain_covariance",))
def calibration_artifacts(ctx):
    study = ctx.memo("mfg.artifacts", artifacts_study)
    chain, datum, sphere, step = study["chain"], study["datum"], study["sphere"], study["step_gauge"]
    corner = chain["coupon_far_corner"]
    f_chain = finding("First-order covariance of the INSTRUMENT->CAD frame chain predicts the Monte Carlo spread of "
                      "coupon points", "numerical",
                      {k: v["linear_std_mm"] for k, v in chain.items()},
                      {"generator": _generator("left-perturbed SE(3) frame chain", samples=3000),
                       "checks": [_check("analytic", f"relative Frobenius difference MC vs first order, {k}", v["rel_frobenius"], 0.1)
                                  for k, v in chain.items()]},
                      unit="mm",
                      uncertainty=_u("truncation_bound", study["chain_linearization_mm"],
                                     "linearization error of the first-order std: symmetric sigma points through the exact "
                                     "SE(3) chain minus first order, max over points and axes (mm)"),
                      tolerance={"abs": 1e-12, "rel": 1e-9})
    refused = datum["refusals"]
    f_datum = finding("The 3-2-1 datum frame is orthonormal, equivariant under rigid motion and has the propagated covariance",
                      "numerical", {"std_translation_mm": datum["std_translation_mm"], "std_rotation_urad": datum["std_rotation_urad"]},
                      {"checks": [_check("exact_arithmetic", "max |R^T R - I|", datum["orthonormality"], 1e-12),
                                  _check("exact_arithmetic", "max |frame(M points) - M frame(points)|", datum["equivariance"], 1e-9),
                                  _check("exact_arithmetic", "noise-free recovery of the part pose", datum["recovery"], 1e-9),
                                  _check("analytic", "relative Frobenius difference MC vs Jacobian covariance", datum["covariance_rel_frobenius"], 0.1),
                                  _refusal("collinear primary datum points (A)", "datum_degenerate", refused["collinear_primary"]),
                                  _refusal("secondary datum direction (B) normal to the primary plane (A)", "datum_degenerate",
                                           refused["secondary_normal_to_primary"])]},
                      uncertainty=_u("monte_carlo_95ci", 1.96 * math.sqrt(2.0 / 3000),
                                     "relative 95% half-width of MC covariance entries"),
                      tolerance={"abs": 1e-9, "rel": 1e-6})
    sphere_rel = max(abs(m / l - 1) for m, l in zip(sphere["mc_std_mm"], sphere["linear_std_mm"]))
    sphere_bias = max(abs(b) / l for b, l in zip(sphere["mc_bias_mm"], sphere["linear_std_mm"])) * math.sqrt(sphere["trials"])
    f_sphere = finding("Gauge-sphere fit on a 75 degree cap recovers centre and radius with the linearized covariance",
                       "numerical", {"linear_std_mm": sphere["linear_std_mm"], "mc_std_mm": sphere["mc_std_mm"]},
                       {"generator": _generator("CMM probing noise", noise_mm=CMM_NOISE, points=25, trials=sphere["trials"]),
                        "checks": [_check("analytic", "max relative std difference (MC vs linearized)", sphere_rel, 0.12),
                                   _check("analytic", "max |MC mean - truth| in standard errors", sphere_bias, 4.0, "le")]},
                       unit="mm",
                       uncertainty=_u("monte_carlo_95ci", 1.96 / math.sqrt(2 * (sphere["trials"] - 1)),
                                      "relative 95% half-width of MC standard deviations"),
                       tolerance={"abs": 1e-12, "rel": 1e-6})
    step_rel = max(abs(m / l - 1) for m, l in zip(step["mc_std"], step["linear_std"]))
    f_step = finding("Step-gauge fit recovers a declared 50 ppm scale error and 1 um offset", "numerical",
                     {"linear_std": step["linear_std"], "mc_std": step["mc_std"]},
                     {"generator": _generator("step gauge noise", noise_mm=0.0005, trials=2000),
                      "checks": [_check("exact_arithmetic", "noise-free recovery of scale error", step["noise_free_recovery"][0], 1e-12),
                                 _check("exact_arithmetic", "noise-free recovery of offset (mm)", step["noise_free_recovery"][1], 1e-12),
                                 _check("analytic", "max relative std difference (MC vs linearized)", step_rel, 0.1)]},
                     uncertainty=_u("monte_carlo_95ci", 1.96 / math.sqrt(2 * 1999),
                                    "relative 95% half-width of MC standard deviations"),
                     tolerance={"abs": 1e-12, "rel": 1e-6})
    ctx.artifact_json("frame-chain.json", _r({"links": study["links"], "chain": chain,
                                              "contributions_rss_mm": study["contributions_rss_mm"], "datum": datum}))
    ctx.artifact_json("artifacts.json", _r({"gauge_sphere": dict(sphere, nominal_diameter_mm=25.4, probe_points=25,
                                                                 cap_polar_deg=75),
                                            "step_gauge": dict(step, steps_mm=[10.0 * k for k in range(1, 21)]),
                                            "datum_points_part_mm": DATUM_POINTS}))
    findings = [f_chain, f_datum, f_sphere, f_step,
                _not_measured("The physical gauge sphere, step gauge and scale bar have their certified dimensions, and the "
                              "lab frame chain has the declared covariances", "calibration")]
    dominant = max(study["contributions_rss_mm"], key=study["contributions_rss_mm"].get)
    fields = _fields(
        "A frame chain INSTRUMENT -> WORLD -> FIXTURE -> PART -> CAD with left-perturbation covariances propagates to "
        "point uncertainty by first-order adjoints, and the 3-2-1 datum and artifact fits have the linearized "
        "covariances their Jacobians predict.",
        "T_true = exp(xi) T; composition C = sum Ad(T_1..T_k-1) C_k Ad^T; point covariance [I, -(Tp)^] C [I, -(Tp)^]^T; "
        "3-2-1 datum: A plane normal z, B direction x, origin on the A, B and C planes; geometric sphere fit; "
        "step gauge m = (1 + e) L + b.",
        ["Declared link covariances (tracker registration 0.010 mm / 5 urad, fixture 0.020 mm / 20 urad, CAD 0.005 mm / 10 urad)",
         "Datum points probed with the tracker (0.015 mm); gauge sphere probed with the CMM (0.002 mm)",
         f"Seeded synthetic noise (seed {SEED + 1})"],
        "Synthetic probing only; no artifact was measured.",
        "Datum frame orthonormal and equivariant; MC covariances equal first-order covariances within sampling error.",
        "Build each artifact model, fit noise-free and noisy synthetic data, propagate covariances analytically and by "
        "Monte Carlo, and attribute the corner-point uncertainty to chain links.",
        f"Coupon far-corner std (linear) {[round(v, 4) for v in corner['linear_std_mm']]} mm; largest link contribution "
        f"{dominant} ({study['contributions_rss_mm'][dominant]:.4f} mm RSS); datum rotation std "
        f"{[round(v, 1) for v in datum['std_rotation_urad']]} urad.",
        f"Monte Carlo sampling error of covariances is about sqrt(2 / N); the first-order stds neglect higher-order terms, "
        f"which symmetric sigma points through the exact SE(3) chain estimate at {study['chain_linearization_mm']:.1e} mm.",
        ["collinear primary datum and B normal to A (refused)", "rigid-motion equivariance", "cap-only sphere probing",
         "linearization of the chain vs exact SE(3) sampling"],
        ["All link covariances are declared; real ones come from the instrument and fixture calibration records (T139).",
         "Thermal drift and probe lobing are not modelled."],
        "Open: estimate the link covariances from the repeated datum probing of a real acquisition (the three re-seated "
        "repeats of each protocol) instead of declaring them, which needs a reader of datum-probing exports that does "
        "not exist, and add probe lobing and thermal drift to the chain model.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T131 Gage R&R ---------------------------------------------------------------------------
GAGE_SIGMA = {"part": 0.050, "operator": 0.006, "interaction": 0.004, "repeatability": 0.010}
GAGE_DESIGN = (10, 3, 3)


def gage_study() -> dict:
    generator = met.rng(SEED + 2)
    p, o, r = GAGE_DESIGN
    truth = {k: v ** 2 for k, v in GAGE_SIGMA.items()}
    grr_true = truth["repeatability"] + truth["operator"] + truth["interaction"]
    percent_true = 100.0 * math.sqrt(grr_true / (grr_true + truth["part"]))
    example = met.gage_rr_anova(met.simulate_gage_study(generator, p, o, r, GAGE_SIGMA))
    studies = 2000
    raw = {k: np.empty(studies) for k in truth}
    percent, truncated_operator, identity = np.empty(studies), 0, 0.0
    for n in range(studies):
        result = met.gage_rr_anova(met.simulate_gage_study(generator, p, o, r, GAGE_SIGMA))
        for key in truth:
            raw[key][n] = result["raw_components"][key]
        percent[n] = result["percent_grr"]
        truncated_operator += "operator" in result["truncated"]
        ss = result["ss"]
        identity = max(identity, abs(ss["total"] - ss["part"] - ss["operator"] - ss["interaction"] - ss["error"]) / ss["total"])
    recovery = {k: {"true": truth[k], "mean": float(raw[k].mean()), "se": float(raw[k].std(ddof=1) / math.sqrt(studies))}
                for k in truth}
    # P(MS_O < MS_PO) = P(F(o - 1, (p - 1)(o - 1)) < c) with c = E[MS_PO] / E[MS_O]; for o - 1 = 2 the F
    # distribution function is 1 - (1 + 2x / d2)^(-d2 / 2).
    d2 = (p - 1) * (o - 1)
    ems_po = truth["repeatability"] + r * truth["interaction"]
    ratio = ems_po / (ems_po + p * r * truth["operator"])
    truncation_theory = 1.0 - (1.0 + 2.0 * ratio / d2) ** (-d2 / 2.0) if o == 3 else float("nan")
    # The raw GRR estimate is c_E MS_E + c_PO MS_PO + c_O MS_O with independent scaled chi-square mean
    # squares, so its exact variance is sum c^2 2 E[MS]^2 / df; %GRR spreads because this does.
    ems = {"error": truth["repeatability"], "interaction": ems_po, "operator": ems_po + p * r * truth["operator"]}
    dfs = {"error": p * o * (r - 1), "interaction": d2, "operator": o - 1}
    coefficients = {"error": 1.0 - 1.0 / r, "interaction": 1.0 / r - 1.0 / (p * r), "operator": 1.0 / (p * r)}
    grr_raw = raw["repeatability"] + raw["operator"] + raw["interaction"]
    centred = grr_raw - grr_raw.mean()
    grr_spread = {"exact_mean": grr_true, "exact_variance": sum(coefficients[k] ** 2 * 2 * ems[k] ** 2 / dfs[k] for k in ems),
                  "mc_mean": float(grr_raw.mean()), "mc_mean_se": float(grr_raw.std(ddof=1) / math.sqrt(studies)),
                  "mc_variance": float(grr_raw.var(ddof=1)),
                  "mc_variance_se": float(math.sqrt(max(np.mean(centred ** 4) - np.mean(centred ** 2) ** 2, 0.0) / studies))}
    return {"design": list(GAGE_DESIGN), "sigma": GAGE_SIGMA, "percent_grr_true": percent_true,
            "ndc_true": 1.41 * math.sqrt(truth["part"] / grr_true), "example": example, "studies": studies,
            "recovery": recovery, "identity": identity, "grr_spread": grr_spread,
            "percent_grr_quantiles": {q: float(np.quantile(percent, float(q))) for q in ("0.05", "0.5", "0.95")},
            "percent_grr_quantile_halfwidth": {q: _quantile_halfwidth(percent, float(q)) for q in ("0.05", "0.5", "0.95")},
            "operator_truncation_fraction": truncated_operator / studies, "operator_truncation_theory": truncation_theory}


def _quantile_halfwidth(sample, level, z=1.96) -> float:
    """Half-width (in the sample's units) of the distribution-free 95% interval of a sample quantile.

    The interval runs between the order statistics of rank n p -+ z sqrt(n p (1 - p))
    (normal approximation to the binomial count below the quantile).
    """
    ordered = np.sort(np.asarray(sample, dtype=float))
    n = len(ordered)
    spread = z * math.sqrt(n * level * (1.0 - level))
    lo = min(max(int(math.floor(n * level - spread)) - 1, 0), n - 1)
    hi = min(max(int(math.ceil(n * level + spread)) - 1, 0), n - 1)
    return 0.5 * float(ordered[hi] - ordered[lo])


# The procedure of a real study. Each protocol has one specimen, so the ten "parts" are ten features of
# it whose true values span the measured range; %GRR is therefore reported against the declared
# tolerance (P/T), not only against the between-feature spread, which is not a process variation.
GAGE_PARTS = {
    "MFG-FLAT-PLATE-01": ["M00-M01", "M00-M02", "M00-M03", "M00-M04", "M00-M06", "M00-M12", "M00-M18", "M00-M24",
                          "M12-M13", "M12-M18"],
    "MFG-CYLINDER-01": ["circumferential 30 deg", "circumferential 60 deg", "circumferential 90 deg",
                        "circumferential 120 deg", "circumferential 150 deg", "circumferential 180 deg", "axial 100 mm",
                        "axial 200 mm", "helical 60 deg / 100 mm", "helical 90 deg / 200 mm"],
    # One tape is on the coupon at a time (MFG-COUPON-01), so a separation between tapes is never a stable part.
    # The study lays the lateral tape L once and leaves it in place for all rounds; its ten gage targets at
    # s = k L / 9 have offsets from the model's nominal route that span the measured range (2 mm to -1 mm).
    "MFG-COUPON-01": [f"lateral tape L, gage target G{k} at s = {k} L / 9: offset normal to the nominal route"
                      for k in range(10)],
}
GAGE_SETUP = {
    "MFG-FLAT-PLATE-01": "the 25 markers stay on the plate; no tape is laid",
    "MFG-CYLINDER-01": "the markers stay on the tube; no tape is laid; each pair is measured as its camera chord",
    "MFG-COUPON-01": "lay the lateral tape L once (JIG-START-01 with insert L, as in MFG-COUPON-01) with ten 6 mm coded "
                     "targets at s = k L / 9 (k = 0...9); it stays in place, alone on the coupon, for all rounds",
}


def gage_procedure() -> dict:
    """Machine-readable Gage R&R and type-1 procedure (a plan; the thresholds are hypotheses)."""
    p, o, r = GAGE_DESIGN
    operators = [f"O{k + 1}" for k in range(o)]
    studies = {}
    for protocol, parts in GAGE_PARTS.items():
        rounds = []
        for replicate in range(1, r + 1):
            # Seeded, platform-independent randomization: order each operator's block by a SHA-256 key.
            blocks = [{"operator": operator,
                       "order": sorted(parts, key=lambda part: hashlib.sha256(
                           f"{protocol}|{replicate}|{operator}|{part}".encode("utf-8")).hexdigest())}
                      for operator in sorted(operators, key=lambda name: hashlib.sha256(
                          f"{protocol}|{replicate}|{name}".encode("utf-8")).hexdigest())]
            holder = V_BLOCKS["id"] if protocol == "MFG-CYLINDER-01" else NEST["id"]
            rounds.append({"replicate": replicate, "before": f"remove the specimen from {holder}, re-seat it, re-probe "
                                                             "datums A, B, C and rebuild the PART frame",
                           "blocks": blocks})
        studies[protocol] = {"setup": GAGE_SETUP[protocol],
                             "parts": [{"id": f"P{k + 1:02d}", "feature": part} for k, part in enumerate(parts)],
                             "operators": operators, "replicates": r, "rounds": rounds}
    return {"schema": "ciw.lab-gage-rr-procedure.v1", "design": {"parts": p, "operators": o, "replicates": r},
            "studies": studies,
            "blinding": "features are presented by coded identity; operators do not see earlier readings or other "
                        "operators' results; a recorder enters readings",
            "reference_variation": "declared tolerance of each measurand (P/T = 6 s_GRR / T); %GRR against the "
                                   "between-feature spread is reported but is not a process variation",
            "type1_study": {"reference": "SG-200 step at 100 mm (certificate value)", "readings": 25,
                            "operator": "O1", "setup": "one setup, no re-fixturing between readings",
                            "report": "bias against the certificate and repeatability s",
                            "hypothesis": "Cg = 0.2 T / (6 s) >= 1.33 and Cgk = (0.1 T - |bias|) / (3 s) >= 1.33"},
            "analysis": "ANOVA method (T131); raw components are reported, negative ones truncated at zero and flagged; "
                        "no interaction pooling",
            "hypotheses": ["%GRR (P/T) <= 10%", "ndc >= 5"],
            "status": "plan: no study has been performed"}


@_task("T131", ("test_gage_rr_recovers_components_and_refuses_unbalanced",))
def gage_rr(ctx):
    study = ctx.memo("mfg.gage", gage_study)
    recovery, quant = study["recovery"], study["percent_grr_quantiles"]
    f_recovery = finding("ANOVA Gage R&R recovers the declared variance components on synthetic studies", "numerical",
                         {k: v["mean"] for k, v in recovery.items()},
                         {"generator": _generator("crossed random-effects study", design=study["design"],
                                                  sigma_mm=study["sigma"], studies=study["studies"]),
                          "checks": [_check("analytic", f"(mean raw estimate - true) / (4 SE), {k}",
                                            (v["mean"] - v["true"]) / (4 * v["se"]), 1.0) for k, v in recovery.items()]
                          + [_check("exact_arithmetic", "max relative SS decomposition residual", study["identity"], 1e-12)]},
                         unit="mm^2",
                         uncertainty=_u("monte_carlo_95ci", {k: 1.96 * v["se"] for k, v in recovery.items()},
                                        "95% half-widths of the mean component estimates (mm^2)"),
                         tolerance={"abs": 1e-15, "rel": 1e-6})
    spread = study["grr_spread"]
    f_spread = finding("Sampling spread of %GRR from a single 10 x 3 x 3 study", "numerical",
                       {"true": study["percent_grr_true"], **quant},
                       {"generator": _generator("crossed random-effects study", studies=study["studies"]),
                        "checks": [_check("analytic", "(MC mean of the raw GRR variance - truth) / (4 SE)",
                                          (spread["mc_mean"] - spread["exact_mean"]) / (4 * spread["mc_mean_se"]), 1.0),
                                   _check("analytic", "(MC variance of the raw GRR estimate - exact sum c^2 2 E[MS]^2 / df) "
                                          "/ (4 SE)", (spread["mc_variance"] - spread["exact_variance"]) / (4 * spread["mc_variance_se"]), 1.0),
                                   _check("analytic", "true %GRR inside the MC 5-95% interval (distance to the nearer end)",
                                          min(study["percent_grr_true"] - quant["0.05"], quant["0.95"] - study["percent_grr_true"]),
                                          0.0, "signed_ge")]},
                       unit="%",
                       uncertainty=_u("monte_carlo_95ci", study["percent_grr_quantile_halfwidth"],
                                      "half-widths of the distribution-free 95% order-statistic intervals of the sample "
                                      "%GRR quantiles (%); the true %GRR is exact"),
                       tolerance={"abs": 1e-9, "rel": 1e-6})
    fraction, theory = study["operator_truncation_fraction"], study["operator_truncation_theory"]
    f_trunc = finding("Fraction of 10 x 3 x 3 studies whose raw operator component is negative (truncated to zero)",
                      "numerical", {"monte_carlo": fraction, "f_distribution": theory},
                      {"generator": _generator("crossed random-effects study", studies=study["studies"]),
                       "checks": [_check("analytic", "(MC fraction - F(2, 18) probability) / (4 binomial SE)",
                                         (fraction - theory) / (4 * math.sqrt(theory * (1 - theory) / study["studies"])), 1.0)]},
                      uncertainty=_u("monte_carlo_95ci", 1.96 * math.sqrt(theory * (1 - theory) / study["studies"]),
                                     "binomial 95% half-width"),
                      tolerance={"abs": 1e-12, "rel": 1e-9})
    code = None
    try:
        met.gage_rr_anova(np.ones((10, 3)))
    except met.MetrologyRefusal as exc:
        code = exc.code
    missing = None
    try:
        data = np.ones((10, 3, 3))
        data[2, 1, 0] = np.nan
        met.gage_rr_anova(data)
    except met.MetrologyRefusal as exc:
        missing = exc.code
    refusals = [_refusal("two-way array without replicates", "unbalanced_design", code),
                _refusal("missing reading", "unbalanced_design", missing)]
    f_refuse = finding("Gage R&R refuses unbalanced or incomplete designs", "computational_pipeline",
                       sum(check["passed"] for check in refusals), {"checks": refusals},
                       unit="refusals", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    ctx.artifact_json("gage-rr.json", _r({k: study[k] for k in ("design", "sigma", "percent_grr_true", "ndc_true", "recovery",
                                                                 "percent_grr_quantiles", "percent_grr_quantile_halfwidth",
                                                                 "grr_spread", "operator_truncation_fraction", "example")}))
    ctx.artifact_json("gage-rr-procedure.json", gage_procedure())
    findings = [f_recovery, f_spread, f_trunc, f_refuse,
                _not_measured("The real gage (instrument, fixture and operators) has %GRR below 10% on the coupon features",
                              "sensor_performance"),
                finding("The measurement system is approved for production use", "production_acceptance", "not_performed", {})]
    fields = _fields(
        "The ANOVA method recovers the variance components of a balanced crossed study without bias (before "
        "truncation), and a single 10 x 3 x 3 study estimates %GRR only within a wide sampling interval; the procedure "
        "of a real study must therefore fix its parts, run order, blinding, re-fixturing and reference variation in "
        "advance.",
        "y_ijk = mu + P_i + O_j + (PO)_ij + e_ijk; expected mean squares give s_e^2 = MS_E, s_po^2 = (MS_PO - MS_E)/r, "
        "s_o^2 = (MS_O - MS_PO)/(p r), s_p^2 = (MS_P - MS_PO)/(o r); %GRR = 100 sqrt(GRR / total); ndc = 1.41 s_p / s_GRR; "
        "type-1 study Cg = 0.2 T / (6 s), Cgk = (0.1 T - |bias|) / (3 s).",
        ["Declared synthetic truth (mm): part 0.050, operator 0.006, interaction 0.004, repeatability 0.010",
         f"Design 10 parts x 3 operators x 3 replicates; {study['studies']} simulated studies (seed {SEED + 2})",
         "Procedure (gage-rr-procedure.json): the ten parts of each protocol are ten features of its specimen (plate "
         "marker pairs, cylinder marker pairs, the offsets of ten gage targets on the coupon's lateral tape, which is "
         "laid once and left alone on the coupon for the study); SHA-256-keyed run order per replicate and "
         "operator; re-fixturing before every replicate round; blinded coded features; a type-1 study on the SG-200 "
         "100 mm step"],
        "Synthetic readings only; no gage, operator or part was involved.",
        "SS_total = SS_P + SS_O + SS_PO + SS_E exactly; raw component estimators are unbiased.",
        "Simulate studies, estimate components, compare means with truth in standard errors, record the %GRR "
        "sampling distribution and the truncation frequency, check refusals of unbalanced data, and write the "
        "procedure of the real crossed and type-1 studies with their thresholds as hypotheses.",
        f"True %GRR {study['percent_grr_true']:.2f}% (ndc {study['ndc_true']:.2f}); single-study 90% interval "
        f"[{quant['0.05']:.1f}%, {quant['0.95']:.1f}%]; operator component truncated in "
        f"{100 * study['operator_truncation_fraction']:.1f}% of studies (F-distribution: "
        f"{100 * study['operator_truncation_theory']:.1f}%).",
        "Monte Carlo standard errors are in the checks; quantiles are sample quantiles of 2000 studies with "
        "distribution-free 95% order-statistic intervals (in %).",
        ["SS decomposition", "negative variance estimates (truncation)", "unbalanced or missing data (refused)",
         "wide single-study interval"],
        ["AIAG interaction pooling (p > 0.25) is not applied; components are reported unpooled.",
         "Normal random effects; real operator effects may be systematic or drift in time.",
         "Features of one specimen stand in for parts, so the between-part variance is a spread of measurands, not of "
         "a process; the procedure reports %GRR against the declared tolerance (P/T) for that reason."],
        "Open: read the 10 x 3 x 3 readings of a real study run by gage-rr-procedure.json into gage_rr_anova (no reader "
        "of study exports exists) and replace T140's declared instrument terms by the measured repeatability; decide "
        "whether to apply the AIAG interaction-pooling rule, which is not applied.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T132 fibre/tape placement ----------------------------------------------------------------
PLACEMENT = {"fibre_angle_deg": 45.0, "course_mm": 500.0, "sigma_lateral_mm": 0.10, "sigma_heading_rad": 5e-4,
             "sigma_radius_mm": 0.05, "sigma_encoder_rad": 5e-5, "spec_mm": 0.5, "min_steering_radius_mm": 635.0}
STEERED = {"theta0_deg": 30.0, "theta1_deg": 60.0, "axial_mm": 300.0}


def placement_deviation(s, theta, delta, dheading, dradius, dphi, radius=geo.CYLINDER_RADIUS):
    """Exact lateral deviation (development) of the placed course from the intended geodesic.

    The head places the tow at commanded machine angles on a mandrel of radius
    R + dR: the course angle becomes atan((1 + dR/R) tan theta) + dheading, the
    start shifts by (R + dR) dphi circumferentially and by delta laterally.
    """
    actual = np.arctan((1.0 + dradius / radius) * np.tan(theta)) + dheading
    lateral = np.array([math.cos(theta), -math.sin(theta)])
    start_x = (radius + dradius) * dphi + delta * lateral[0]
    start_z = delta * lateral[1]
    x = start_x + s * np.sin(actual)
    z = start_z + s * np.cos(actual)
    return x * lateral[0] + z * lateral[1]


def placement_rss(s, spec=PLACEMENT, radius=geo.CYLINDER_RADIUS):
    theta = math.radians(spec["fibre_angle_deg"])
    fixed = spec["sigma_lateral_mm"] ** 2 + (radius * math.cos(theta) * spec["sigma_encoder_rad"]) ** 2
    growth = spec["sigma_heading_rad"] ** 2 + (math.sin(theta) * math.cos(theta) * spec["sigma_radius_mm"] / radius) ** 2
    return math.sqrt(fixed + s ** 2 * growth), fixed, growth


def placement_study() -> dict:
    radius = geo.CYLINDER_RADIUS
    cyl = geo.CYLINDER
    theta0, theta1 = math.radians(STEERED["theta0_deg"]), math.radians(STEERED["theta1_deg"])
    axial = STEERED["axial_mm"]
    rate = (theta1 - theta0) / axial
    kg_err = kg3_err = kn_err = 0.0
    kg_max = 0.0
    for z in np.linspace(0.0, axial, 61):
        theta = theta0 + rate * z
        phi = (math.log(math.cos(theta0)) - math.log(math.cos(theta))) / (rate * radius)
        u = np.array([phi, z])
        du = np.array([math.tan(theta) / radius, 1.0])
        ddu = np.array([rate / (math.cos(theta) ** 2 * radius), 0.0])
        kg, kn = geo.chart_curvatures(cyl, u, du, ddu)
        kg3, kn3 = geo.embedded_curvatures(cyl, u, du, ddu)
        # Heading from the circumferential direction is 90 deg - theta, so kappa_g = -dtheta/ds = -rate cos(theta).
        kg_err = max(kg_err, abs(kg + rate * math.cos(theta)))
        kg3_err = max(kg3_err, abs(kg - kg3), abs(kn - kn3))
        kn_err = max(kn_err, abs(kn + math.sin(theta) ** 2 / radius))
        kg_max = max(kg_max, abs(kg))
    helix_kg = max(abs(geo.chart_curvatures(cyl, [0.3, z], [math.tan(theta0) / radius, 1.0], [0.0, 0.0])[0])
                   for z in (0.0, 100.0, 200.0))
    # Steered course end versus the geodesic along its initial tangent (development coordinates).
    end_x = (math.log(math.cos(theta0)) - math.log(math.cos(theta1))) / rate
    steered_deviation = end_x * math.cos(theta0) - axial * math.sin(theta0)
    # Independent route: integrate x'(z) = tan(theta(z)) by composite Simpson on 2000 panels.
    zs = np.linspace(0.0, axial, 2001)
    slope = np.tan(theta0 + rate * zs)
    simpson_x = (axial / 6000.0) * (slope[0] + slope[-1] + 4 * slope[1:-1:2].sum() + 2 * slope[2:-1:2].sum())
    steered_quadrature_error = abs(simpson_x - end_x)
    # Flat Jacobi transfer on the mandrel and the exactly perturbed course.
    theta = math.radians(PLACEMENT["fibre_angle_deg"])
    course = PLACEMENT["course_mm"]
    heading = math.pi / 2 - theta
    transfer = jacobi.transfer(cyl, [0.0, 0.0], heading, course, steps=50)
    phi_error = float(np.max(np.abs(transfer.matrix() - np.array([[1.0, course], [0.0, 1.0]]))))
    delta, dpsi = 0.3, 1e-3
    separation = geo.separation_nonlinear(cyl, [0.0, 0.0], heading, course, 50, delta, dpsi, base=transfer)
    exact_error = float(np.max(np.abs(separation - (delta + transfer.s * math.sin(dpsi)))))
    # Tolerance stack: RSS model versus exact deviations under sampled sources.
    sigma_course, fixed, growth = placement_rss(course)
    length_max = math.sqrt(max(0.0, (PLACEMENT["spec_mm"] / COVERAGE_K) ** 2 - fixed) / growth)
    generator = met.rng(SEED + 3)
    samples = 20000
    draws = generator.standard_normal((4, samples))
    deviation = placement_deviation(length_max, theta, PLACEMENT["sigma_lateral_mm"] * draws[0],
                                    PLACEMENT["sigma_heading_rad"] * draws[1], PLACEMENT["sigma_radius_mm"] * draws[2],
                                    PLACEMENT["sigma_encoder_rad"] * draws[3])
    mc_std = float(deviation.std(ddof=1))
    counter = float(placement_deviation(1000.0, theta, 0.0, 0.0, 0.2, 0.0))
    stations = np.linspace(0.0, 1000.0, 21)
    return {"max_kappa_g_per_mm": kg_max, "min_steering_radius_mm": 1.0 / kg_max, "kappa_g_closed_form_error": kg_err,
            "chart_vs_embedded_error": kg3_err, "kappa_n_error": kn_err, "helix_kappa_g": helix_kg,
            "steered_end_deviation_mm": steered_deviation, "steered_quadrature_error_mm": steered_quadrature_error,
            "phi_error": phi_error, "perturbed_exact_error": exact_error,
            "sigma_at_course_mm": sigma_course, "length_max_mm": length_max, "rss_at_length_max_mm": placement_rss(length_max)[0],
            "mc_std_mm": mc_std, "mc_samples": samples, "radius_counterexample_mm": counter,
            "rss_curve": {"s_mm": stations.tolist(), "k_sigma_mm": [COVERAGE_K * placement_rss(s)[0] for s in stations]}}


@_task("T132", ("test_placement_curvature_stack_and_radius_counterexample",))
def placement_tolerance(ctx):
    study = ctx.memo("mfg.placement", placement_study)
    f_kg = finding("Geodesic curvature of a 30-to-60 degree variable-angle steered course on the R = 100 mm mandrel",
                   "numerical", {"max_kappa_g_per_mm": study["max_kappa_g_per_mm"],
                                 "min_steering_radius_mm": study["min_steering_radius_mm"]},
                   {"derivation": "development (R phi, z) with N the tangent rotated +90 deg: signed kappa_g = -theta'(z) cos theta "
                                  "(theta from the axis, heading from the circumferential direction 90 deg - theta), "
                                  "|kappa_g| = theta' cos theta (docs/lab/MANUFACTURING.md#t132)",
                    "checks": [_check("analytic", "chart kappa_g vs -theta' cos theta (1/mm)", study["kappa_g_closed_form_error"], 1e-12),
                               _check("cross_implementation", "chart (Christoffel) vs embedded 3D curvatures (1/mm)", study["chart_vs_embedded_error"], 1e-12),
                               _check("analytic", "kappa_n vs -sin^2 theta / R (1/mm)", study["kappa_n_error"], 1e-12),
                               _check("analytic", "constant-angle helix kappa_g (1/mm)", study["helix_kappa_g"], 1e-15)]},
                   uncertainty=_u("roundoff", max(study["kappa_g_closed_form_error"], study["chart_vs_embedded_error"]),
                                  "closed-form vs chart vs embedded evaluation (1/mm)"),
                   tolerance={"abs": 1e-12, "rel": 1e-9})
    f_flat = finding("Lateral deviation on the mandrel follows e(s) = delta + s sin(dpsi) (flat Jacobi transfer)",
                     "numerical", {"phi_error": study["phi_error"], "perturbed_exact_error_mm": study["perturbed_exact_error"]},
                     {"checks": [_check("analytic", "max |Phi(500) - [[1, 500], [0, 1]]|", study["phi_error"], 1e-9),
                                 _check("analytic", "exactly perturbed course minus delta + s sin(dpsi) (mm)", study["perturbed_exact_error"], 1e-9)]},
                     uncertainty=_u("roundoff", study["perturbed_exact_error"], "exact perturbation vs closed form (mm)"),
                     tolerance={"abs": 1e-9, "rel": 0})
    rel = study["mc_std_mm"] / study["rss_at_length_max_mm"] - 1.0
    f_stack = finding("Tolerance stack of the placed course and the longest course meeting a 0.5 mm lateral spec at k = 2",
                      "numerical", {"sigma_at_500mm": study["sigma_at_course_mm"], "length_max_mm": study["length_max_mm"]},
                      {"generator": _generator("placement error sources", sigma=dict(PLACEMENT), samples=study["mc_samples"]),
                       "checks": [_check("analytic", "exact-model MC std / RSS - 1 at L_max (4 standard errors)", rel,
                                         4.0 / math.sqrt(2 * study["mc_samples"]))]},
                      unit="mm",
                      uncertainty=_u("reference_error", abs(rel),
                                     "relative difference between the exact-model MC std and the RSS at L_max"),
                      tolerance={"abs": 1e-9, "rel": 1e-9})
    f_counter = finding("Programming a helix in machine angles transfers mandrel radius error into lateral drift", "numerical",
                        study["radius_counterexample_mm"],
                        {"checks": [_check("analytic", "drift at 1 m / first-order L sin(theta) cos(theta) dR / R - 1",
                                           study["radius_counterexample_mm"] / (1000.0 * 0.5 * 0.2 / geo.CYLINDER_RADIUS) - 1.0, 0.01),
                                    _check("analytic", "drift exceeds the 0.5 mm spec (mm)", study["radius_counterexample_mm"], 0.5, "ge")]},
                        unit="mm",
                        uncertainty=_u("roundoff", 1e-12, "closed-form development model"),
                        tolerance={"abs": 1e-9, "rel": 1e-9},
                        counterexample={"statement": "A helix programmed in machine coordinates (phi, z) is insensitive to "
                                                     "mandrel radius error because it is a geodesic on every cylinder",
                                        "witness": {"radius_mm": geo.CYLINDER_RADIUS, "radius_error_mm": 0.2,
                                                    "fibre_angle_deg": 45.0, "course_mm": 1000.0}})
    f_steer = finding("End deviation of the steered course from the geodesic along its initial tangent", "numerical",
                      study["steered_end_deviation_mm"],
                      {"derivation": "development closed form x(z) = (ln cos theta0 - ln cos theta(z)) / theta'",
                       "checks": [_check("analytic", "closed-form end abscissa vs Simpson quadrature of tan theta (mm)",
                                         study["steered_quadrature_error_mm"], 1e-8)]},
                      unit="mm",
                      uncertainty=_u("roundoff", 1e-12, "closed-form development model"),
                      tolerance={"abs": 1e-9, "rel": 1e-9})
    curve = study["rss_curve"]
    ctx.artifact_text("placement-stack.svg", svg.line_plot(
        [("2 sigma lateral (RSS)", curve["s_mm"], curve["k_sigma_mm"]),
         ("spec 0.5 mm", [curve["s_mm"][0], curve["s_mm"][-1]], [0.5, 0.5])],
        title="Tape placement on R = 100 mm: lateral tolerance stack", xlabel="course length (mm)", ylabel="2 sigma (mm)"))
    ctx.artifact_json("placement.json", _r(dict(study, sources=PLACEMENT, steered=STEERED)))
    findings = [f_kg, f_flat, f_stack, f_counter, f_steer,
                _not_measured("Tows placed by a real AFP head follow the programmed course within the stack, with gaps and "
                              "overlaps inside 0.5 mm"),
                _not_measured("The declared 635 mm minimum steering radius avoids tow wrinkling for the placed material")]
    fields = _fields(
        "On a cylindrical mandrel geodesic placement paths are helices (straight in the development); a steered "
        "variable-angle course has geodesic curvature of magnitude theta' cos theta; lateral placement errors grow linearly "
        "(flat Jacobi transfer) and radius error enters through the machine-angle programming.",
        "Development (R phi, z); kappa_g = <u'' + Gamma(u', u'), N> / |u'|^2 with N the tangent rotated +90 deg "
        "(signed -theta'(z) cos theta for theta measured from the axis); kappa_n = II(u', u') / I(u', u'); "
        "e(s) = delta + R cos(theta) dphi + s (dpsi + sin(theta) cos(theta) dR / R); RSS with k = 2.",
        ["Declared mandrel R = 100 mm; course 500 mm at 45 deg; steered course 30 -> 60 deg over 300 mm axial",
         "Declared 1 sigma sources: head lateral 0.10 mm, heading 0.5 mrad, radius 0.05 mm, encoder 50 urad",
         "Declared spec 0.5 mm (k = 2) and minimum steering radius 635 mm"],
        "No observation: placement is modelled, not executed.",
        "Helix kappa_g = 0; chart and embedded curvatures agree; Phi = [[1, s], [0, 1]]; MC std of the exact "
        "deviation model equals the RSS.",
        "Evaluate curvatures by two routes along the steered course, integrate the transfer and an exactly perturbed "
        "course, sample the four error sources through the exact development model, and evaluate a radius-error witness.",
        f"max kappa_g {study['max_kappa_g_per_mm']:.5f} /mm (steering radius {study['min_steering_radius_mm']:.1f} mm); "
        f"2 sigma at 500 mm = {COVERAGE_K * study['sigma_at_course_mm']:.3f} mm; longest course within 0.5 mm: "
        f"{study['length_max_mm']:.1f} mm; radius witness drift {study['radius_counterexample_mm']:.3f} mm at 1 m.",
        "Closed forms; Monte Carlo standard error of the std is about 0.5% with 20000 samples.",
        ["chart vs embedded curvature", "helix is geodesic", "nonlinear vs linear deviation", "RSS vs exact sampling",
         "radius error through machine-angle programming"],
        ["Tow width, compaction and tack are not modelled; tows are curves, not strips.",
         "Error sources are independent and Gaussian with declared sigmas."],
        "Open: model tows as strips with width, compaction and tack instead of curves, and compare the placement stack "
        "with process data (tow positions, gaps and overlaps at the steering radius), none of which exists here.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T133 winding path sensitivity -------------------------------------------------------------
WINDING = {"length_mm": 1500.0, "steps": 750, "friction_mu": 0.2, "regimes_deg": (50.0, 70.0)}


def _peak(s, values):
    """First interior local maximum refined by a parabola through three samples."""
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            a, b, c = values[i - 1], values[i], values[i + 1]
            denominator = a - 2 * b + c
            offset = 0.5 * (a - c) / denominator if denominator else 0.0
            return float(b - 0.25 * (a - c) * offset), float(s[i] + offset * (s[1] - s[0]))
    return float("nan"), float("nan")


def clairaut_period(c, nodes=200) -> tuple[float, float]:
    """(period P, azimuth advance per period) of a torus geodesic with Clairaut constant c, by Gauss-Legendre quadrature.

    Along the geodesic r dtheta/ds = +-sqrt(1 - c^2 / rho^2) and dphi/ds = c / rho^2. A circulating
    winding (|c| < R - r) advances theta by 2 pi per period; a librating one turns at
    rho = |c| and the substitution theta = theta_turn sin(tau) removes the endpoint singularity.
    """
    big, small = geo.TORUS_MAJOR, geo.TORUS_MINOR
    x, w = np.polynomial.legendre.leggauss(nodes)
    if abs(c) > big - small:
        turn = math.acos((abs(c) - big) / small)
        tau = 0.25 * math.pi * (x + 1.0)
        theta, jac = turn * np.sin(tau), turn * np.cos(tau)
        scale = 4.0 * 0.25 * math.pi
    else:
        theta, jac, scale = math.pi * (x + 1.0), np.ones_like(x), math.pi
    rho = big + small * np.cos(theta)
    root = np.sqrt(rho ** 2 - c ** 2)
    return (float(scale * np.sum(w * small * jac * rho / root)),
            float(scale * np.sum(w * c * small * jac / (rho * root))))


@functools.lru_cache(maxsize=1)
def winding_study() -> dict:
    torus = geo.TORUS
    big, small = geo.TORUS_MAJOR, geo.TORUS_MINOR
    length, steps = WINDING["length_mm"], WINDING["steps"]
    regimes = {}
    for psi_deg in WINDING["regimes_deg"]:
        psi = math.radians(psi_deg)
        transfer = jacobi.transfer(torus, [0.0, 0.0], psi, length, steps=steps)
        clairaut = np.array([torus.clairaut(y[:2], y[2:4]) for y in transfer.states])
        remainders, perturbed = [], None
        for size in (1e-3, 5e-4):
            start = jacobi.perturbed_start(torus, [0.0, 0.0], psi, 0.0, size)
            _, states = integrators.integrate_fixed(torus.geodesic_rhs, start, length, steps, "rk4")
            separation = jacobi.normal_separation(torus, transfer.states[:, :4], states)
            remainders.append(float(np.max(np.abs(separation - size * transfer.states[:, 6]))))
            perturbed = states if perturbed is None else perturbed
        c0 = clairaut[0]
        librating = abs(c0) > big - small
        # One-period monodromy. The rotational Killing field gives a periodic normal Jacobi field, so
        # trace M = 2 and M is parabolic: the heading column grows linearly, j_head(nP) = n M01.
        period, advance = clairaut_period(c0)
        period_steps = int(math.ceil(period / 2.0))
        monodromy = jacobi.transfer(torus, [0.0, 0.0], psi, period, steps=period_steps).matrix()
        monodromy_h2 = jacobi.transfer(torus, [0.0, 0.0], psi, period, steps=period_steps // 2).matrix()
        three = jacobi.transfer(torus, [0.0, 0.0], psi, 3.0 * period, steps=3 * period_steps).matrix()
        # Mechanism: a heading error dpsi changes c by -rho0 sin(psi) dpsi and hence the azimuth advance
        # per period; the normal separation after one period is M01 dpsi = rho0^2 sin^2(psi) (d advance / dc) dpsi.
        rho0 = big + small
        dc = 1e-5 * abs(c0)
        clairaut_rate = rho0 ** 2 * math.sin(psi) ** 2 * (clairaut_period(c0 + dc)[1] - clairaut_period(c0 - dc)[1]) / (2 * dc)
        crossings = integrators.hermite_zeros(transfer.s, transfer.states[:, 1] - (0.0 if librating else 2.0 * math.pi),
                                              transfer.states[:, 3])
        crossings = [z for z in crossings if z > 1e-6]
        period_integrated = crossings[1] if librating else crossings[0]
        row = {"heading_from_parallel_deg": psi_deg, "clairaut_mm": c0, "regime": "librating" if librating else "circulating",
               "clairaut_drift_rel": float(np.max(np.abs(clairaut - c0)) / abs(c0)),
               "det_drift": float(np.max(np.abs(transfer.determinant() - 1.0))),
               "period_mm": period, "period_integrated_mm": period_integrated, "azimuth_advance_rad": advance,
               "monodromy": monodromy.tolist(), "monodromy_trace_minus_2": float(np.trace(monodromy) - 2.0),
               "monodromy_richardson": float(np.max(np.abs(monodromy - monodromy_h2)) / 15.0),
               "secular_rate": abs(monodromy[0, 1]) / period, "m01_mm": float(monodromy[0, 1]),
               "m01_clairaut_mm": clairaut_rate, "three_period_ratio": float(three[0, 1] / (3.0 * monodromy[0, 1])),
               "max_abs_j_head_mm": float(np.max(np.abs(transfer.states[:, 6]))),
               "conjugate_points_mm": transfer.conjugate_points(),
               "remainder_order": integrators.observed_order([1e-3, 5e-4], remainders),
               "theta_range_rad": [float(transfer.states[:, 1].min()), float(transfer.states[:, 1].max())]}
        if librating:
            # The path turns where rho = |c|.
            cos_turn = (abs(c0) - big) / small
            turn = math.acos(cos_turn)
            numeric, where = _peak(transfer.s, transfer.states[:, 1])
            perturbed_c = torus.clairaut(perturbed[0, :2], perturbed[0, 2:4])
            perturbed_turn = math.acos((abs(perturbed_c) - big) / small)
            numeric_perturbed, _ = _peak(transfer.s, perturbed[:, 1])
            predicted_shift = (big + small) * math.sin(psi) * 1e-3 / (small * math.sin(turn))
            row.update(turn_clairaut_rad=turn, turn_numeric_rad=numeric, turn_arclength_mm=where,
                       turn_shift_numeric_rad=numeric_perturbed - numeric, turn_shift_linear_rad=predicted_shift,
                       turn_shift_clairaut_rad=perturbed_turn - turn)
        regimes[f"psi{psi_deg:.0f}"] = row
        regimes[f"psi{psi_deg:.0f}"]["profile"] = {"s": transfer.s[::5].tolist(), "j_head": transfer.states[::5, 6].tolist()}
    cylinder = jacobi.transfer(geo.CYLINDER, [0.0, 0.0], math.radians(50.0), length, steps=60)
    cylinder_error = float(np.max(np.abs(cylinder.states[:, 6] - cylinder.s)))
    # Constant-angle (loxodrome) winding: slippage tendency |kappa_g / kappa_n|.
    slippage = {}
    agreement = 0.0
    for psi_deg in WINDING["regimes_deg"]:
        psi = math.radians(psi_deg)
        ratios, kgs, cs = [], [], []
        for theta in np.linspace(-math.pi, math.pi, 361)[:-1]:
            rho = big + small * math.cos(theta)
            du = np.array([math.cos(psi) / rho, math.sin(psi) / small])
            ddu = np.array([math.cos(psi) * small * math.sin(theta) / rho ** 2 * du[1], 0.0])
            u = np.array([0.0, theta])
            kg, kn = geo.chart_curvatures(torus, u, du, ddu)
            kg3, kn3 = geo.embedded_curvatures(torus, u, du, ddu)
            agreement = max(agreement, abs(kg - kg3), abs(kn - kn3))
            ratios.append(abs(kg) / abs(kn))
            kgs.append(abs(kg))
            cs.append(rho * math.cos(psi))
        ratios = np.array(ratios)
        slippage[f"psi{psi_deg:.0f}"] = {"max_ratio": float(ratios.max()), "max_abs_kappa_g_per_mm": float(max(kgs)),
                                         "fraction_above_mu": float(np.mean(ratios > WINDING["friction_mu"])),
                                         "clairaut_variation_mm": float(max(cs) - min(cs))}
    return {"regimes": regimes, "cylinder_j_head_error": cylinder_error, "slippage": slippage,
            "curvature_route_agreement": agreement}


@_task("T133", ("test_winding_clairaut_sensitivity_and_slippage",))
def winding_sensitivity(ctx):
    study = ctx.memo("mfg.winding", winding_study)
    librating, circulating = study["regimes"]["psi50"], study["regimes"]["psi70"]
    regimes = study["regimes"]
    f_clairaut = finding("The Clairaut constant is conserved along geodesic windings of the torus mandrel", "numerical",
                         {k: v["clairaut_mm"] for k, v in regimes.items()},
                         {"checks": [_check("invariant", f"relative Clairaut drift, {k}", v["clairaut_drift_rel"], 1e-8)
                                     for k, v in regimes.items()]
                          + [_check("invariant", f"Wronskian drift, {k}", v["det_drift"], 1e-8) for k, v in regimes.items()]},
                         unit="mm",
                         uncertainty=_u("roundoff", max(v["clairaut_drift_rel"] * abs(v["clairaut_mm"]) for v in regimes.values()),
                                        "max Clairaut drift along the integration (mm)"),
                         tolerance={"abs": 1e-9, "rel": 1e-12})
    f_rate = finding("Secular growth rate of the heading-error Jacobi field per unit arclength over one winding period, "
                     "relative to the cylinder (rate 1)", "numerical",
                     {"psi50_librating": librating["secular_rate"], "psi70_circulating": circulating["secular_rate"],
                      "cylinder": 1.0, "period_mm": {k: v["period_mm"] for k, v in regimes.items()}},
                     {"derivation": "surface of revolution: the rotational Killing field is a periodic normal Jacobi field, "
                                    "so the one-period monodromy M has trace 2 and j_head(nP) = n M01 (docs/lab/MANUFACTURING.md#t133)",
                      "checks": [_check("analytic", "cylinder j_head = s (mm)", study["cylinder_j_head_error"], 1e-9)]
                      + [_check("invariant", f"trace of the one-period monodromy minus 2, {k}", v["monodromy_trace_minus_2"], 1e-6)
                         for k, v in regimes.items()]
                      + [_check("analytic", f"M01 vs Clairaut quadrature rho0^2 sin^2(psi) d(advance)/dc (relative), {k}",
                                v["m01_mm"] / v["m01_clairaut_mm"] - 1.0, 1e-5) for k, v in regimes.items()]
                      + [_check("invariant", f"j_head(3P) / (3 M01) - 1 (linear growth), {k}", v["three_period_ratio"] - 1.0, 1e-6)
                         for k, v in regimes.items()]
                      + [_check("analytic", f"integrated period vs quadrature (mm), {k}",
                                v["period_integrated_mm"] - v["period_mm"], 1e-5) for k, v in regimes.items()]
                      + [_check("self_convergence", f"nonlinear heading remainder order, {k}", v["remainder_order"], 1.8, "ge")
                         for k, v in regimes.items()]},
                     uncertainty=_u("truncation_bound", max(v["monodromy_richardson"] / v["period_mm"] for v in regimes.values()),
                                    "RK4 Richardson estimate of the monodromy entries over the period (rate units)"),
                     tolerance={"abs": 1e-6, "rel": 1e-6})
    f_regime = finding("The 50 deg winding librates (turns before the inner equator) and the 70 deg winding circulates "
                       "through the negatively curved inner region", "numerical",
                       {k: v["theta_range_rad"] for k, v in regimes.items()},
                       {"checks": [_check("analytic", "psi50: pi - max theta (rad)", math.pi - librating["theta_range_rad"][1], 0.0, "signed_ge"),
                                   _check("analytic", "psi50: pi + min theta (rad)", math.pi + librating["theta_range_rad"][0], 0.0, "signed_ge"),
                                   _check("analytic", "psi70: max theta - pi (rad)", circulating["theta_range_rad"][1] - math.pi, 0.0, "signed_ge")]},
                       unit="rad", uncertainty=_u("truncation_bound", 1e-5, "theta extremes sampled at the 2 mm RK4 "
                                                  "nodes; the checks have margins above 1 rad"),
                       tolerance={"abs": 1e-6, "rel": 1e-6})
    f_turn = finding("Clairaut sensitivity predicts the turnaround-latitude shift of the librating winding", "numerical",
                     {"turn_rad": librating["turn_clairaut_rad"], "shift_per_mrad_rad": librating["turn_shift_linear_rad"]},
                     {"checks": [_check("invariant", "integrated turnaround vs Clairaut turnaround (rad)",
                                        librating["turn_numeric_rad"] - librating["turn_clairaut_rad"], 1e-6),
                                 _check("analytic", "integrated shift vs rho0 sin(psi) dpsi / (r sin theta_turn) (relative)",
                                        librating["turn_shift_numeric_rad"] / librating["turn_shift_linear_rad"] - 1.0, 0.02)]},
                     uncertainty=_u("truncation_bound", abs(librating["turn_numeric_rad"] - librating["turn_clairaut_rad"]),
                                    "parabolic peak refinement vs Clairaut turnaround (rad)"),
                     tolerance={"abs": 1e-7, "rel": 1e-6})
    slip = study["slippage"]
    f_slip = finding("Slippage tendency abs(kappa_g / kappa_n) of constant-angle winding on the torus mandrel", "numerical",
                     {k: {"max_ratio": v["max_ratio"], "fraction_above_mu": v["fraction_above_mu"]} for k, v in slip.items()},
                     {"checks": [_check("cross_implementation", "chart vs embedded curvatures (1/mm)", study["curvature_route_agreement"], 1e-12)]},
                     uncertainty=_u("roundoff", study["curvature_route_agreement"],
                                    "chart vs embedded curvature evaluation (1/mm)"),
                     tolerance={"abs": 1e-9, "rel": 1e-9})
    f_counter = finding("A constant winding angle is not geodesic on the torus mandrel", "numerical",
                        slip["psi50"]["max_abs_kappa_g_per_mm"],
                        {"checks": [_check("analytic", "max |kappa_g| of the 50 deg loxodrome (1/mm)",
                                           slip["psi50"]["max_abs_kappa_g_per_mm"], 1e-4, "ge"),
                                    _check("analytic", "Clairaut quantity variation along the loxodrome (mm)",
                                           slip["psi50"]["clairaut_variation_mm"], 1.0, "ge")]},
                        unit="1/mm",
                        uncertainty=_u("roundoff", study["curvature_route_agreement"],
                                       "chart vs embedded curvature evaluation (1/mm)"),
                        tolerance={"abs": 1e-12, "rel": 1e-9},
                        counterexample={"statement": "A constant winding angle (as on a cylinder) is a geodesic, "
                                                     "slip-free path on every mandrel of revolution",
                                        "witness": {"mandrel": geo.TORUS.describe(), "heading_from_parallel_deg": 50,
                                                    "max_slippage_ratio": slip["psi50"]["max_ratio"]}})
    ctx.artifact_text("winding-heading-sensitivity.svg", svg.line_plot(
        [("torus psi = 50 deg (librating)", librating["profile"]["s"], librating["profile"]["j_head"]),
         ("torus psi = 70 deg (circulating)", circulating["profile"]["s"], circulating["profile"]["j_head"]),
         ("cylinder", [0.0, WINDING["length_mm"]], [0.0, WINDING["length_mm"]])],
        title="Winding: heading-error Jacobi field j_head", xlabel="arclength (mm)", ylabel="j_head (mm per rad)", markers=False))
    ctx.artifact_json("winding.json", _r({"regimes": {k: {kk: vv for kk, vv in v.items() if kk != "profile"}
                                                      for k, v in regimes.items()},
                                          "slippage": slip, "friction_mu_declared": WINDING["friction_mu"]}))
    findings = [f_clairaut, f_rate, f_regime, f_turn, f_slip, f_counter,
                _not_measured(f"Fibre does not slip on a real mandrel wherever abs(kappa_g / kappa_n) <= {WINDING['friction_mu']} "
                              "(the friction coefficient is declared, not measured)")]
    fields = _fields(
        "Geodesic winding on a torus conserves the Clairaut constant, so heading errors move the turnaround latitude "
        "predictably; because a heading error changes the Clairaut constant and with it the azimuth advance per "
        "period, the heading-error Jacobi field grows secularly (linearly) in both the librating and the circulating "
        "regime, at a rate set by d(advance)/dc; constant-angle winding is not geodesic on the torus and needs friction "
        "abs(kappa_g / kappa_n).",
        "Torus R = 150 mm, r = 50 mm, K = cos(theta) / (r (R + r cos theta)); Clairaut c = rho^2 dphi/ds; "
        "period P(c) and advance A(c) by quadrature of r dtheta / sqrt(1 - c^2 / rho^2); one-period monodromy M with "
        "trace 2 (Killing field), j_head(nP) = n M01, M01 = rho0^2 sin^2(psi) dA/dc; rate = |M01| / P; "
        "theta_turn = acos((|c| - R) / r); d theta_turn = rho0 sin(psi) dpsi / (r sin theta_turn); "
        "loxodrome dphi/ds = cos(psi)/rho, dtheta/ds = sin(psi)/r.",
        ["Declared torus mandrel (150, 50) mm and cylinder R = 100 mm", "Windings launched on the outer equator at "
         "50 and 70 deg from the parallel, integrated over 1500 mm and over one and three periods",
         "Declared friction coefficient mu = 0.2"],
        "No observation: winding is modelled, not executed.",
        "Clairaut and Wronskian drift at rounding level; trace M = 2; j_head(3P) = 3 M01; M01 equals the Clairaut "
        "quadrature; remainder of the heading linearization O(dpsi^2).",
        "Integrate geodesic and Jacobi fields, compute the one-period monodromy at the quadrature period (and at half "
        "the steps), compare M01 with the Clairaut quadrature and with three periods, perturb the heading exactly at "
        "two sizes, locate turnarounds by parabolic refinement, evaluate loxodrome curvature by chart and embedded routes.",
        f"psi = 50 deg: librating (theta within +/- {math.degrees(librating['theta_range_rad'][1]):.2f} deg, turnaround "
        f"{math.degrees(librating['turn_clairaut_rad']):.2f} deg), period {librating['period_mm']:.3f} mm, secular rate "
        f"{librating['secular_rate']:.4f}; psi = 70 deg: circulating (passes the inner equator), period "
        f"{circulating['period_mm']:.3f} mm, secular rate {circulating['secular_rate']:.4f} (cylinder: 1); loxodrome "
        f"slippage max {slip['psi50']['max_ratio']:.3f} (50 deg), {slip['psi70']['max_ratio']:.3f} (70 deg) against "
        "declared mu = 0.2.",
        "RK4 at about 2 mm steps; monodromy Richardson estimate in the finding uncertainty; Clairaut drift below 1e-8 relative.",
        ["Clairaut conservation", "monodromy trace (Killing field)", "linear growth over three periods",
         "Clairaut quadrature of the secular term", "turnaround location by two routes", "regime (theta range)",
         "second-order heading remainder", "chart vs embedded curvature", "cylinder control"],
        ["Fibre bandwidth, tension and resin are not modelled; the fibre is a curve.",
         "The friction coefficient is declared; slip also depends on tension and cure state.",
         "The secular rate is per unit heading error; the growth of a physical winding error also depends on how the "
         "machine corrects the path between layers."],
        "Open: model how a winding machine corrects the path between layers, which decides whether the secular "
        "heading-error growth accumulates in a physical winding, and compare the slippage ratio with slip observed "
        "under real fibre tension and friction (no such data exists here).")
    return {"state": "completed", "fields": fields, "findings": findings}


# T134 coating or welding trajectory ------------------------------------------------------------
TOOLS = {"welding torch": 15.0, "spray gun": 120.0}
TRAJECTORY_BOX = {"lateral_mm": 0.3, "heading_rad": 0.002}


def _tangent_frame(surface, u, velocity):
    t = velocity / math.sqrt(surface.speed_squared(u, velocity))
    return t, surface.normal(u, t)


# The off-axis route exercises the geodesic-torsion term (tau_g = 0 on the symmetry axis); the unit
# normal is differenced one RK4 geodesic step of 1 um either side of each node.
OFF_AXIS_DEG = 10.0
NORMAL_STEP_MM = 1e-3


def _route_frame(surface, y):
    """(X, n, t, kappa_n, tau_g, dn/ds) at one geodesic state, all in R^3 except the two curvatures.

    kappa_n = II(t, t) and tau_g = II(t, N) come from the second fundamental form; dn/ds comes from
    central differences of the embedded unit normal along the geodesic, so it shares neither the
    second fundamental form nor the speed-factor formula.
    """
    u, v = y[:2], y[2:4]
    t, n = _tangent_frame(surface, u, v)
    second = geo.second_fundamental_form(surface, u)
    state = np.asarray(y[:4], dtype=float)
    ahead = integrators.step_rk4(surface.geodesic_rhs, state, NORMAL_STEP_MM)
    behind = integrators.step_rk4(surface.geodesic_rhs, state, -NORMAL_STEP_MM)
    dn = (surface.unit_normal3(ahead[:2]) - surface.unit_normal3(behind[:2])) / (2.0 * NORMAL_STEP_MM)
    return (surface.embedding(u), surface.unit_normal3(u), surface.embedding_jacobian(u) @ t,
            float(t @ second @ t), float(t @ second @ n), dn)


def _tcp_tools(surface, transfer):
    """TCP paths P = X + H n of both tools along a sampled geodesic, with two derivations of |dP/ds|.

    Returns ({tool: summary}, kappa_n at the nodes). Pointwise, |t + H dn/ds| is compared
    with the closed-form factor sqrt((1 - H kappa_n)^2 + (H tau_g)^2) at every node;
    segment-wise, |P(k+1) - P(k)| / ds (the speed of the polyline a robot would follow) is
    compared with the mean factor at the segment ends where 1 - H kappa_n keeps its sign.
    """
    frames = [_route_frame(surface, y) for y in transfer.states]
    base, normals, tangents = (np.array([f[k] for f in frames]) for k in range(3))
    kappa_n, tau_g = np.array([f[3] for f in frames]), np.array([f[4] for f in frames])
    dn = np.array([f[5] for f in frames])
    ds = np.diff(transfer.s)
    tools = {}
    for name, standoff in TOOLS.items():
        signed = 1.0 - standoff * kappa_n
        factors = np.hypot(signed, standoff * tau_g)
        points = base + standoff * normals
        chords = np.diff(points, axis=0)
        smooth = np.sign(signed[1:]) == np.sign(signed[:-1])
        segment_error = np.abs(np.linalg.norm(chords, axis=1) / ds - 0.5 * (factors[1:] + factors[:-1]))
        tools[name] = {"standoff_mm": standoff, "polyline_length_mm": geo.polyline_length(points),
                       "integral_length_mm": float(np.sum(0.5 * (factors[1:] + factors[:-1]) * ds)),
                       "speed_factor_min": float(factors.min()), "speed_factor_max": float(factors.max()),
                       "max_H_tau_g": float(np.max(np.abs(standoff * tau_g))),
                       "tau_term_max": float(np.max(factors - np.abs(signed))),
                       "pointwise_max_difference": float(np.max(np.abs(np.linalg.norm(tangents + standoff * dn, axis=1)
                                                                       - factors))),
                       "segment_max_difference": float(np.max(segment_error[smooth], initial=0.0)),
                       "smooth_segments": int(np.sum(smooth)),
                       "sign_changes_of_1_minus_H_kn": int(np.sum(np.diff(np.sign(signed)) != 0)),
                       "reversed_segments": int(np.sum(np.einsum("ij,ij->i", chords, tangents[:-1]) < 0)),
                       "profile": {"s": transfer.s[::4].tolist(), "factor": factors[::4].tolist()}}
    return tools, kappa_n


@functools.lru_cache(maxsize=1)
def coating_study() -> dict:
    surface = geo.COUPON
    length = nominal_study()["length_mm"]
    steps = NOMINAL_STATIONS * 26
    transfer = jacobi.transfer(surface, geo.STATION, 0.0, length, steps=steps)
    lat, head = TRAJECTORY_BOX["lateral_mm"], TRAJECTORY_BOX["heading_rad"]
    envelope = lat * np.abs(transfer.states[:, 4]) + head * np.abs(transfer.states[:, 6])
    vertex_max = 0.0
    # The route lies on the symmetry axis y = 0, so the (-delta, -dtheta) vertices mirror the evaluated ones.
    for sign in (1.0, -1.0):
        separation = geo.separation_nonlinear(surface, geo.STATION, 0.0, length, steps, lat, sign * head, base=transfer)
        vertex_max = max(vertex_max, float(np.max(np.abs(separation))))
    # Standoff error and tilt from a lateral offset of the tool, by ray casting and by curvature.
    standoff_rows, worst_rel = [], -math.inf
    stride = steps // NOMINAL_STATIONS
    for index in range(0, steps + 1, stride):
        u, v = transfer.states[index, :2], transfer.states[index, 2:4]
        t, n = _tangent_frame(surface, u, v)
        second = geo.second_fundamental_form(surface, u)
        kappa_lateral = float(n @ second @ n)
        direction = geo.lateral_direction3(surface, u, t)
        e = float(envelope[index])
        exact = geo.standoff_error_exact(surface, u, direction, e, TOOLS["welding torch"])
        approx = -0.5 * kappa_lateral * e ** 2
        # Signed excess of the ray-cast error over the series beyond 0.1% (plus 1e-9 mm for rounding); the
        # observed relative difference is below 1e-4, so a coefficient off by 0.1% would fail.
        worst_rel = max(worst_rel, abs(exact - approx) - 1e-3 * abs(approx) - 1e-9)
        standoff_rows.append({"s_mm": float(transfer.s[index]), "lateral_error_mm": e, "kappa_lateral_per_mm": kappa_lateral,
                              "standoff_error_exact_mm": exact, "standoff_error_series_mm": approx,
                              "tilt_rad": abs(kappa_lateral) * e})
    # Tool-centre-point path P = X + H n: speed factor sqrt((1 - H kn)^2 + (H tau_g)^2), on the nominal
    # route (tau_g = 0 by symmetry) and on an off-axis route where the tau_g term is exercised.
    tools, kn_all = _tcp_tools(surface, transfer)
    off_axis = jacobi.transfer(surface, geo.STATION, math.radians(OFF_AXIS_DEG), length, steps=steps)
    off_axis_tools, _ = _tcp_tools(surface, off_axis)
    concave_radius = float(1.0 / kn_all.max())
    # Cylinder control through the same ray caster on a Monge-form cylinder: a ray parallel to a
    # radius at distance e meets the circle sqrt(R^2 - e^2) from the axis, so the error is R - sqrt(R^2 - e^2).
    radius = geo.CYLINDER_RADIUS
    tube = geo.MongeCylinder(radius)
    control = []
    for y0 in (0.0, 30.0, 60.0):
        for e in (1.0, 5.0):
            u = np.array([10.0, y0])
            t = tube.unit_tangent(u, 0.0)
            ray = geo.standoff_error_exact(tube, u, geo.lateral_direction3(tube, u, t), e, TOOLS["welding torch"])
            control.append({"y_mm": y0, "lateral_mm": e, "ray_cast_mm": ray, "closed_form_mm": radius - math.sqrt(radius ** 2 - e ** 2)})
    cylinder = {"cases": control, "max_difference_mm": max(abs(c["ray_cast_mm"] - c["closed_form_mm"]) for c in control),
                "exact_mm_at_1mm": radius - math.sqrt(radius ** 2 - 1.0)}
    return {"length_mm": length, "envelope_max_mm": float(envelope.max()),
            "envelope_argmax_mm": float(transfer.s[int(np.argmax(envelope))]), "vertex_max_mm": vertex_max,
            "standoff": standoff_rows, "standoff_excess": worst_rel, "tools": tools, "off_axis_tools": off_axis_tools,
            "off_axis_heading_deg": OFF_AXIS_DEG, "min_concave_radius_mm": concave_radius, "cylinder_control": cylinder}


@_task("T134", ("test_coating_standoff_and_offset_cusp",))
def trajectory_sensitivity(ctx):
    study = ctx.memo("mfg.coating", coating_study)
    weld, spray = study["tools"]["welding torch"], study["tools"]["spray gun"]
    f_env = finding("Lateral-error envelope of the coupon trajectory under the declared registration box", "numerical",
                    {"max_mm": study["envelope_max_mm"], "at_s_mm": study["envelope_argmax_mm"]},
                    {"derivation": "sup over the box of |delta j_lat + dtheta j_head| = delta |j_lat| + dtheta |j_head|",
                     "checks": [_check("analytic", "max over box vertices of the exactly perturbed separation / linear envelope - 1",
                                       study["vertex_max_mm"] / study["envelope_max_mm"] - 1.0, 0.02)]},
                    unit="mm",
                    uncertainty=_u("reference_error", abs(study["vertex_max_mm"] - study["envelope_max_mm"]),
                                   "second-order remainder at the box vertices (mm)"),
                    tolerance={"abs": 1e-8, "rel": 1e-7})
    worst = max(study["standoff"], key=lambda r: abs(r["standoff_error_exact_mm"]))
    f_standoff = finding("Standoff error from a lateral tool offset follows -kappa_lateral e^2 / 2", "numerical",
                         {"max_abs_standoff_error_mm": abs(worst["standoff_error_exact_mm"]),
                          "max_tilt_rad": max(r["tilt_rad"] for r in study["standoff"]),
                          "cylinder_exact_mm_at_1mm": study["cylinder_control"]["exact_mm_at_1mm"]},
                         {"checks": [_check("analytic", "max over stations of abs(ray-cast - series) - 1e-3 abs(series) - 1e-9 mm",
                                            study["standoff_excess"], 0.0, "signed_le"),
                                     _check("analytic", "Monge-form cylinder: ray-cast standoff error vs R - sqrt(R^2 - e^2) "
                                            "at y = 0, 30, 60 mm and e = 1, 5 mm (mm)",
                                            study["cylinder_control"]["max_difference_mm"], 1e-12)]},
                         unit="mm",
                         uncertainty=_u("roundoff", 1e-14, "Newton ray casting converged to 1e-14 relative"),
                         tolerance={"abs": 1e-10, "rel": 1e-7})
    off = study["off_axis_tools"]
    both = [*study["tools"].values(), *off.values()]
    pointwise = max(t["pointwise_max_difference"] for t in both)
    segment = max(study["tools"]["welding torch"]["segment_max_difference"], off["welding torch"]["segment_max_difference"])
    f_offset = finding("Tool-centre-point path length element is sqrt((1 - H kappa_n)^2 + (H tau_g)^2)", "numerical",
                       {"welding_speed_factor_range": [weld["speed_factor_min"], weld["speed_factor_max"]],
                        "spray_speed_factor_range": [spray["speed_factor_min"], spray["speed_factor_max"]],
                        "off_axis_welding_speed_factor_range": [off["welding torch"]["speed_factor_min"],
                                                                off["welding torch"]["speed_factor_max"]],
                        "off_axis_spray_speed_factor_range": [off["spray gun"]["speed_factor_min"],
                                                              off["spray gun"]["speed_factor_max"]],
                        "off_axis_max_H_tau_g": {name: t["max_H_tau_g"] for name, t in off.items()}},
                       {"checks": [_check("analytic", "max over the nodes of both routes (on axis and 10 deg off axis) and "
                                          "both tools of | |t + H dn/ds| - factor |, dn/ds from central differences of the "
                                          "unit normal", pointwise, 1e-7),
                                   _check("analytic", "welding torch, both routes: max over segments of | |P(k+1) - P(k)| / ds "
                                          "- mean factor at the segment ends |", segment, 2e-3),
                                   _check("analytic", "off-axis welding route: max of factor - |1 - H kappa_n| (the tau_g "
                                          "term the pointwise check resolves)", off["welding torch"]["tau_term_max"], 1e-5, "ge")]},
                       uncertainty=_u("truncation_bound", pointwise,
                                      "central-difference derivative of the unit normal (1 um RK4 steps) vs the closed-form "
                                      "factor, max over nodes, routes and tools"),
                       tolerance={"abs": 1e-9, "rel": 1e-7})
    f_cusp = finding("A spray standoff beyond the concave radius of curvature folds the tool-centre-point path", "numerical",
                     {"spray_standoff_mm": spray["standoff_mm"], "min_concave_radius_mm": study["min_concave_radius_mm"],
                      "reversed_segments": spray["reversed_segments"]},
                     {"checks": [_check("analytic", "sign changes of 1 - H kappa_n along the route", spray["sign_changes_of_1_minus_H_kn"], 2, "ge"),
                                 _check("analytic", "TCP segments running backwards along the route", spray["reversed_segments"], 1, "ge"),
                                 _check("analytic", "welding torch keeps 1 - H kappa_n > 0 (sign changes)", weld["sign_changes_of_1_minus_H_kn"], 0)]},
                     uncertainty=EXACT, tolerance={"abs": 1e-9, "rel": 1e-9},
                     counterexample={"statement": "The standoff (offset) tool path of a smooth surface path is itself a "
                                                  "smooth path the robot can follow at constant speed",
                                     "witness": {"standoff_mm": spray["standoff_mm"],
                                                 "min_concave_radius_mm": study["min_concave_radius_mm"],
                                                 "route": "coupon nominal route across the dome rim"}})
    ctx.artifact_json("trajectory.json", _r({k: v for k, v in study.items() if k not in ("tools", "off_axis_tools")}
                                            | {key: {n: {k: v for k, v in t.items() if k != "profile"} for n, t in study[key].items()}
                                               for key in ("tools", "off_axis_tools")}))
    ctx.artifact_text("tcp-speed-factor.svg", svg.line_plot(
        [(f"{name} (H = {tool['standoff_mm']:g} mm){suffix}", tool["profile"]["s"], tool["profile"]["factor"])
         for key, suffix in (("tools", ""), ("off_axis_tools", f", {OFF_AXIS_DEG:g} deg off axis"))
         for name, tool in study[key].items()],
        title="Coupon routes: TCP speed per surface speed", xlabel="arclength s (mm)", ylabel="|dP/ds|", markers=False))
    findings = [f_env, f_standoff, f_offset, f_cusp,
                _not_measured("The torch or gun on a real cell stays within the predicted lateral and standoff band"),
                finding("The trajectory is safe to execute on a welding or coating robot cell", "machine_safety", None, {})]
    fields = _fields(
        "Along a trajectory across the dome, lateral registration errors propagate by the Jacobi transfer, produce "
        "a second-order standoff error -kappa e^2 / 2 and a first-order tilt kappa e, and the tool-centre-point path "
        "X + H n has speed factor sqrt((1 - H kappa_n)^2 + (H tau_g)^2), which on the symmetry axis (tau_g = 0) is "
        "|1 - H kappa_n| and vanishes where the standoff equals the concave radius.",
        "e(s) = delta j_lat + dtheta j_head; standoff error by ray casting along the programmed axis; TCP path "
        "P = X + H n with dn/ds = -kappa_n t - tau_g N (Weingarten), so |P'| = sqrt((1 - H kappa_n)^2 + (H tau_g)^2).",
        ["Coupon nominal route (T128) and a route launched 10 deg off the symmetry axis (tau_g != 0)",
         "Declared registration box |delta| <= 0.3 mm, |dtheta| <= 2 mrad",
         "Declared standoffs: welding torch 15 mm, spray gun 120 mm"],
        "No observation: trajectories are modelled, not executed.",
        "Envelope equals the sup over box vertices to second order; standoff series within 0.1% (+ 1e-9 mm); |dP/ds| "
        "equals the closed-form factor at every node and the polyline speed segment by segment; no cusp while "
        "H < concave radius.",
        "Integrate the route with Jacobi fields, perturb at the box vertices exactly, ray-cast standoff at nine "
        "stations, build the TCP path for both tools on the nominal and the off-axis route, compare |t + H dn/ds| "
        "(central differences of the unit normal) with the factor at every node and the polyline speed with it "
        "segment by segment, and count reversals.",
        f"max lateral error {study['envelope_max_mm']:.3f} mm at s = {study['envelope_argmax_mm']:.1f} mm; max standoff "
        f"error {abs(worst['standoff_error_exact_mm']):.2e} mm; welding TCP speed factor "
        f"[{weld['speed_factor_min']:.3f}, {weld['speed_factor_max']:.3f}] on axis and "
        f"[{off['welding torch']['speed_factor_min']:.3f}, {off['welding torch']['speed_factor_max']:.3f}] off axis "
        f"(max H tau_g {off['welding torch']['max_H_tau_g']:.3f}); |dP/ds| vs factor {pointwise:.1e} pointwise and "
        f"{segment:.1e} per segment; spray path folds ({spray['reversed_segments']} "
        f"reversed segments; concave radius {study['min_concave_radius_mm']:.1f} mm < 120 mm).",
        "Linear-envelope accuracy is second order in the box size; ray casting converges to 1e-14; the pointwise "
        "speed check is limited by the O(h^2) central difference of the normal, the segment check by the polyline "
        "discretization (O(ds^2)).",
        ["box-vertex nonlinear check", "ray casting vs series",
         "TCP speed factor vs |t + H dn/ds| at every node, on and off the symmetry axis (tau_g term exercised)",
         "TCP polyline speed vs factor segment by segment",
         "cusp detection by two routes (sign of 1 - H kappa_n and reversed segments)"],
        ["Deposition footprint, spray cone and heat input are not modelled; the speed factor is a kinematic proxy.",
         "Robot joint limits and singularities are not checked."],
        "Open: check robot joint limits and singularities along the tool-centre-point paths, and model the deposition "
        "footprint (spray cone, heat input), which the kinematic speed factor does not capture.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T135 robotic inspection scan paths -----------------------------------------------------------
SWATH_MM = 20.0
AREA_SAMPLES = 6400
SPACING_FACTORS = (1.0, 0.9, 0.8, 0.7, 0.6)


def _row_offsets(spacing):
    count = int(math.ceil((geo.COUPON_Y[1] - geo.COUPON_Y[0]) / spacing - 1e-9))
    return [round((k - (count - 1) / 2) * spacing, 9) for k in range(count)]


def scan_study() -> dict:
    surface = geo.COUPON
    span = geo.COUPON_X[1] - geo.COUPON_X[0]
    sample = geo.area_sample(surface, geo.COUPON_X, geo.COUPON_Y, AREA_SAMPLES)
    cache = {}

    def geodesic_row(offset):
        # Rows are mirror images across y = 0, so only |offset| is integrated.
        key = abs(offset)
        if key not in cache:
            transfer = jacobi.transfer(surface, [geo.COUPON_X[0], key], 0.0, 1.3 * span, steps=130)
            points = geo.clip_to_extent(transfer.points)
            cache[key] = (points, float(np.max(transfer.states[:len(points), 4])))
        points, jmax = cache[key]
        return (points if offset >= 0 else points * np.array([1.0, -1.0])), jmax

    def plan(kind, spacing):
        rows, length, jmax = [], 0.0, None
        for offset in _row_offsets(spacing):
            if kind == "geodesic":
                points, j = geodesic_row(offset)
                jmax = max(jmax or 0.0, j)
            else:
                points = np.column_stack([np.linspace(*geo.COUPON_X, 101), np.full(101, offset)])
            rows.append(geo.embed(surface, points))
        transitions = sum(float(np.linalg.norm(rows[k + 1][-1 if k % 2 == 0 else 0] - rows[k][-1 if k % 2 == 0 else 0]))
                          for k in range(len(rows) - 1))
        length = sum(geo.polyline_length(r) for r in rows) + transitions
        return rows, length, jmax

    plans, margins = {}, []
    for kind in ("geodesic", "chart-parallel"):
        for factor in SPACING_FACTORS:
            rows, length, jmax = plan(kind, factor * SWATH_MM)
            coverage, margin = geo.coverage_with_margin(sample, rows, SWATH_MM)
            margins.append(margin)
            plans[f"{kind} x{factor:g}"] = {"kind": kind, "spacing_mm": factor * SWATH_MM, "rows": len(rows),
                                            "length_mm": length, "coverage": coverage, "max_j_lat": jmax, "_rows": rows}
    nominal = plans["geodesic x1"]
    tightened = SWATH_MM / nominal["max_j_lat"]
    rows, length, _ = plan("geodesic", tightened)
    coverage, margin = geo.coverage_with_margin(sample, rows, SWATH_MM)
    margins.append(margin)
    plans["geodesic Jacobi-tightened"] = {"kind": "geodesic", "spacing_mm": tightened, "rows": len(rows), "length_mm": length,
                                          "coverage": coverage, "_rows": rows}
    fine = geo.area_sample(surface, geo.COUPON_X, geo.COUPON_Y, 4 * AREA_SAMPLES, start=AREA_SAMPLES + 1)
    refinement = geo.coverage_fraction(fine, nominal["_rows"], SWATH_MM) - nominal["coverage"]
    # Complete coverage must survive a 4x larger, disjoint sample: re-evaluate every plan that looks complete.
    for name, entry in plans.items():
        if entry["coverage"] >= 1.0 - 1e-12 or name == "geodesic Jacobi-tightened":
            entry["coverage_refined"], margin = geo.coverage_with_margin(fine, entry["_rows"], SWATH_MM)
            margins.append(margin)
    # Mirror symmetry of the rows is assumed in geodesic_row; integrate one negative offset and compare.
    offset = _row_offsets(SWATH_MM)[-1]
    mirrored = geodesic_row(-offset)[0]
    direct = geo.clip_to_extent(jacobi.transfer(surface, [geo.COUPON_X[0], -offset], 0.0, 1.3 * span, steps=130).points)
    mirror_error = float(np.max(np.abs(direct - mirrored))) if direct.shape == mirrored.shape else math.inf
    plate_sample = geo.area_sample(geo.PLATE, geo.COUPON_X, geo.COUPON_Y, AREA_SAMPLES)
    plate_rows = [geo.embed(geo.PLATE, np.column_stack([np.linspace(*geo.COUPON_X, 101), np.full(101, y)]))
                  for y in _row_offsets(SWATH_MM)]
    plate = {"coverage": geo.coverage_fraction(plate_sample, plate_rows, SWATH_MM),
             "row_length_error_mm": max(abs(geo.polyline_length(r) - span) for r in plate_rows)}
    for entry in plans.values():
        entry.pop("_rows")
    complete = sorted((p["length_mm"], name) for name, p in plans.items()
                      if min(p["coverage"], p.get("coverage_refined", 0.0)) >= 1.0 - 1e-12)
    return {"plans": plans, "sampling_refinement_delta": refinement, "plate_control": plate,
            "mirror_error_mm": mirror_error, "min_boundary_margin_mm": min(margins),
            "shortest_complete": {"plan": complete[0][1], "length_mm": complete[0][0]} if complete else None,
            "shorter_incomplete": sorted(name for name, p in plans.items()
                                         if complete and p["length_mm"] < complete[0][0])}


@_task("T135", ("test_scan_plans_coverage_and_counterexample",))
def inspection_scan_paths(ctx):
    study = ctx.memo("mfg.scan", scan_study)
    plans, plate = study["plans"], study["plate_control"]
    f_cov = finding("Coverage and path length of candidate inspection scan plans on the domed coupon", "numerical",
                    {name: {"coverage": p["coverage"], "length_mm": p["length_mm"], "rows": p["rows"]} for name, p in plans.items()},
                    {"checks": [_check("exact_arithmetic", "flat-plate control: 1 - coverage at the swath spacing", 1.0 - plate["coverage"], 1e-12),
                                _check("exact_arithmetic", "flat-plate row length error (mm)", plate["row_length_error_mm"], 1e-9),
                                _check("self_convergence", "coverage change of the nominal geodesic plan on a 4x larger sample",
                                       study["sampling_refinement_delta"], 0.005),
                                _check("invariant", "integrated row at a negative offset vs the mirrored positive row (mm)",
                                       study["mirror_error_mm"], 1e-12),
                                _check("analytic", "min |sample distance - swath / 2| over all evaluated plans (no knife-edge "
                                       "tie within 1e-6 mm)", study["min_boundary_margin_mm"], 1e-6, "ge")]},
                    uncertainty=_u("reference_error", abs(study["sampling_refinement_delta"]),
                                   "coverage change between 6400 and 25600 Halton samples"),
                    tolerance={"abs": 1e-9, "rel": 1e-7})
    nominal = plans["geodesic x1"]
    f_counter = finding("Geodesic rows at the swath spacing leave gaps on the domed coupon", "numerical",
                        1.0 - nominal["coverage"],
                        {"checks": [_check("analytic", "uncovered area fraction of the nominal geodesic plan", 1.0 - nominal["coverage"], 0.01, "ge"),
                                    _check("analytic", "maximum lateral spreading j_lat of its rows", nominal["max_j_lat"], 1.2, "ge")]},
                        uncertainty=_u("reference_error", abs(study["sampling_refinement_delta"]),
                                       "coverage change between 6400 and 25600 Halton samples"),
                        tolerance={"abs": 1e-9, "rel": 1e-7},
                        counterexample={"statement": "Geodesic scan rows launched at the swath spacing cover a curved coupon "
                                                     "as completely as they cover a flat plate",
                                        "witness": {"plan": "geodesic x1", "coverage": nominal["coverage"],
                                                    "plate_coverage": plate["coverage"], "max_j_lat": nominal["max_j_lat"]}})
    shortest = study["shortest_complete"]
    selected = plans[shortest["plan"]]
    f_short = finding("Shortest evaluated scan plan with no uncovered sample among 32000 area samples", "numerical", shortest,
                      {"checks": [_check("analytic", "1 - coverage of the selected plan (6400 samples)", 1.0 - selected["coverage"], 1e-12),
                                  _check("self_convergence", "1 - coverage of the selected plan (25600 further samples)",
                                         1.0 - selected["coverage_refined"], 1e-12)]},
                      uncertainty=_u("reference_error", 3.0 / (5 * AREA_SAMPLES),
                                     "uncovered area fraction that 32000 samples would likely miss (rule of three)"),
                      tolerance={"abs": 1e-9, "rel": 1e-7})
    tight = plans["geodesic Jacobi-tightened"]
    f_tight = finding("Geodesic rows at spacing swath / max j_lat (first-order Jacobi tightening) leave under 0.1% uncovered",
                      "numerical", {"spacing_mm": tight["spacing_mm"], "coverage": tight["coverage"],
                                    "coverage_refined": tight.get("coverage_refined"), "length_mm": tight["length_mm"]},
                      {"derivation": "row separation ~ spacing j_lat(s); keep spacing max j_lat <= swath",
                       "checks": [_check("analytic", "1 - coverage (6400 samples)", 1.0 - tight["coverage"], 1e-3),
                                  _check("self_convergence", "1 - coverage (25600 further samples)",
                                         1.0 - tight.get("coverage_refined", tight["coverage"]), 1e-3)]},
                      uncertainty=_u("reference_error", abs(study["sampling_refinement_delta"]),
                                     "coverage change between 6400 and 25600 Halton samples"),
                      tolerance={"abs": 1e-9, "rel": 1e-7})
    ctx.artifact_json("scan-plans.json", _r(study))
    series = []
    for kind in ("geodesic", "chart-parallel"):
        chosen = sorted((p["length_mm"], p["coverage"]) for name, p in plans.items() if p["kind"] == kind and "Jacobi" not in name)
        series.append((f"{kind} rows", [c[0] for c in chosen], [c[1] for c in chosen]))
    ctx.artifact_text("coverage-vs-length.svg", svg.line_plot(
        series, title="Domed coupon: scan coverage vs total path length", xlabel="path length (mm)", ylabel="area coverage"))
    findings = [f_cov, f_counter, f_short, f_tight,
                _not_measured("The real scanner footprint is a 20 mm swath on this surface at the planned standoff",
                              "sensor_performance")]
    fields = _fields(
        "Positive curvature focuses geodesic scan rows, so rows launched at the swath spacing cross near the dome and "
        "leave gaps; coverage must be bought with path length, and constant-y (chart-parallel) rows reach full "
        "coverage sooner on this coupon.",
        "Rows: geodesics launched along +x from the edge x = -60 mm at offsets k * spacing, or chart lines y = const; "
        "footprint = points within 10 mm (3D) of a row polyline; coverage = covered area / area estimated on "
        "Halton (2, 3) chart points weighted by sqrt(det g); path length = rows + edge transitions.",
        ["Declared domed coupon (T128) and 20 mm scanner swath",
         f"Spacing factors {list(SPACING_FACTORS)} of the swath; {AREA_SAMPLES} and {4 * AREA_SAMPLES} area-weighted "
         "Halton sample points"],
        "No observation: coverage of modelled footprints on the declared surface.",
        "Flat-plate coverage = 1 at the swath spacing; a 4x larger sample changes coverage by < 0.005.",
        "Generate rows for each plan, compute coverage and length, compare with the flat-plate control and with a "
        "first-order Jacobi-tightened spacing.",
        f"nominal geodesic plan coverage {nominal['coverage']:.4f} (plate 1.0); shortest plan with no uncovered sample: "
        f"{shortest['plan']} at {shortest['length_mm']:.1f} mm; Jacobi-tightened spacing {tight['spacing_mm']:.2f} mm "
        f"-> coverage {tight['coverage']:.5f} ({AREA_SAMPLES} samples), {tight.get('coverage_refined', float('nan')):.5f} "
        f"({4 * AREA_SAMPLES} samples).",
        f"Area sampling: coverage changes by {abs(study['sampling_refinement_delta']):.4f} between the two samples; "
        "'complete' means no uncovered sample among 32000 points (uncovered area below about 1e-4 of the coupon).",
        ["knife-edge ties (minimum distance of any sample from a footprint boundary)", "sample refinement",
         "flat-plate control", "row mirror symmetry (one negative offset integrated)"],
        ["The footprint is a 3D distance band; occlusion, incidence limits and scanner depth of field are not modelled.",
         "Edge transitions are straight 3D chords.",
         "First-order Jacobi tightening does not guarantee complete coverage where rows cross beyond a focal point.",
         "Slivers with area below about 1e-4 of the coupon can escape 32000 sample points."],
        "Open: bound uncovered slivers below the sampling resolution (about 1e-4 of the coupon area) with an exact "
        "footprint union instead of area sampling, and add occlusion, incidence and depth-of-field limits to the "
        "footprint.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T136-T137 path rankings ---------------------------------------------------------------------
LATERAL_SPEC_MM = 0.5
# Derated tolerance boxes aim 0.01% inside the spec: far above the RK4 error of the realized
# separation (about 1e-8 relative) so the verified margin is not a rounding artefact.
DERATE_TARGET = 0.9999
FAN_HEADINGS = (0, 5, 10, 15, 20, 25)


@functools.lru_cache(maxsize=1)
def fan_study() -> dict:
    routes = geo.fan(headings_deg=FAN_HEADINGS)
    coarse = geo.fan(headings_deg=FAN_HEADINGS, step=2.0)
    mirror = geo.route_to_edge(geo.COUPON, geo.STATION, math.radians(-5.0), geo.COUPON_X[1], name="fan-5deg")
    small = {a: geo.route_to_edge(geo.COUPON, geo.STATION, math.radians(a), geo.COUPON_X[1], name=f"fan{a:g}").length
             for a in (1.0, 2.0)}
    return {"routes": routes, "summaries": [r.summary() for r in routes], "coarse": [r.summary() for r in coarse],
            "mirror": mirror.summary(), "small_lengths": small}


def _corner_errors(route, delta, theta) -> dict:
    """Max |separation| / spec of the exactly perturbed route at the four corners of the tolerance box."""
    steps = len(route.transfer.s) - 1
    out = {}
    for sign_delta, sign_theta in itertools.product((1.0, -1.0), repeat=2):
        separation = geo.separation_nonlinear(geo.COUPON, geo.STATION, route.heading, route.length, steps,
                                              sign_delta * delta, sign_theta * theta, base=route.transfer)
        out[f"{sign_delta:+.0f}{sign_theta:+.0f}"] = float(np.max(np.abs(separation))) / LATERAL_SPEC_MM
    return out


def _adaptive_at_nodes(surface, y0, nodes, rtol=1e-9, atol=1e-10) -> np.ndarray:
    """Geodesic states at the given arclength nodes by ciw's adaptive Dormand-Prince 5(4), restarted per interval.

    Each restart tries the step count the previous interval needed, so few steps are rejected.
    """
    states, pieces = [np.asarray(y0, dtype=float)], 1
    for span in np.diff(nodes):
        _, piece, stats = integrators.integrate_adaptive(surface.geodesic_rhs, states[-1], float(span), rtol=rtol,
                                                         atol=atol, h0=float(span) / pieces)
        states.append(piece[-1])
        pieces = max(1, int(stats["accepted_steps"]))
    return np.array(states)


def _offset_start(surface, u0, heading, lateral, dheading) -> np.ndarray:
    """Exactly offset start, coded separately from jacobi.perturbed_start.

    The lateral start is the end of the normal geodesic of length |lateral|
    (adaptive integration). On an oriented surface the rotation J by +90
    degrees commutes with parallel transport, so the transported tangent is
    -sign(lateral) J(w) with w the normal geodesic's end velocity; no transport
    equation is integrated. The heading change then rotates that tangent.
    """
    u0 = np.asarray(u0, dtype=float)
    t = surface.unit_tangent(u0, heading)
    if lateral:
        sign = math.copysign(1.0, lateral)
        start = np.concatenate([u0, sign * surface.normal(u0, t)])
        _, states, _ = integrators.integrate_adaptive(surface.geodesic_rhs, start, abs(lateral), rtol=1e-12, atol=1e-14)
        u0, t = states[-1, :2], -sign * surface.normal(states[-1, :2], states[-1, 2:4])
    n = surface.normal(u0, t)
    return np.concatenate([u0, math.cos(dheading) * t + math.sin(dheading) * n])


def independent_corner_ratios(route, delta, theta) -> dict:
    """The corner ratios of :func:`_corner_errors` by a second computation of the exact perturbation.

    Base and perturbed geodesics are integrated with the adaptive Dormand-Prince
    integrator at the route's nodes (not RK4), from a separately coded exp-map
    start, and the normal separation g(u_p - u, N) is evaluated inline. Only the
    surface geometry is shared, so an error in the RK4 perturbation path shows
    up as a disagreement.
    """
    surface, u0 = geo.COUPON, np.asarray(geo.STATION, dtype=float)
    nodes = route.transfer.s
    base = _adaptive_at_nodes(surface, np.concatenate([u0, surface.unit_tangent(u0, route.heading)]), nodes)
    out = {}
    for sign_delta, sign_theta in itertools.product((1.0, -1.0), repeat=2):
        start = _offset_start(surface, u0, route.heading, sign_delta * delta, sign_theta * theta)
        perturbed = _adaptive_at_nodes(surface, start, nodes)
        separation = [surface.inner(y[:2], z[:2] - y[:2], surface.normal(y[:2], y[2:4])) for y, z in zip(base, perturbed)]
        out[f"{sign_delta:+.0f}{sign_theta:+.0f}"] = max(abs(value) for value in separation) / LATERAL_SPEC_MM
    return out


def calibration_ranking(fan) -> dict:
    """First-order tolerance allocation, then derated until the exact perturbation meets the spec at every corner.

    Off-axis routes have no odd symmetry, so all four corners of the (delta,
    dtheta) box are evaluated. Derating scales both tolerances of a route by
    DERATE_TARGET / (worst ratio); with a positive second-order excess one step
    suffices, and the loop repeats until every corner is within the target. The
    loop's stop condition is a convergence diagnostic, not evidence: the derated
    corners are re-evaluated by :func:`independent_corner_ratios`.
    """
    rows = []
    for route in fan["routes"]:
        summary = route.summary()
        delta1 = LATERAL_SPEC_MM / (2.0 * summary["max_abs_j_lat"])
        theta1 = LATERAL_SPEC_MM / (2.0 * summary["max_abs_j_head_mm"])
        first = corners = _corner_errors(route, delta1, theta1)
        factor, rounds = 1.0, 0
        while max(corners.values()) > DERATE_TARGET and rounds < 4:
            factor *= DERATE_TARGET / max(corners.values())
            corners = _corner_errors(route, factor * delta1, factor * theta1)
            rounds += 1
        rows.append({"route": route.name, "length_mm": route.length, "max_abs_j_lat": summary["max_abs_j_lat"],
                     "max_abs_j_head_mm": summary["max_abs_j_head_mm"],
                     "first_order_lateral_tolerance_mm": delta1, "first_order_heading_tolerance_mrad": 1e3 * theta1,
                     "first_order_corner_ratios": first, "derating": factor,
                     "lateral_tolerance_mm": factor * delta1, "heading_tolerance_mrad": 1e3 * factor * theta1,
                     "flat_plate_heading_tolerance_mrad": 1e3 * LATERAL_SPEC_MM / (2 * route.length),
                     "corner_ratios": corners, "derating_converged": max(corners.values()) <= DERATE_TARGET,
                     "independent_corner_ratios": independent_corner_ratios(route, factor * delta1, factor * theta1),
                     "realized_max_error_mm": LATERAL_SPEC_MM * max(corners.values())})
    ranking = [r["route"] for r in sorted(rows, key=lambda r: (-r["heading_tolerance_mrad"], r["route"]))]
    coarse = {c["route"]: c for c in fan["coarse"]}
    richardson = max(abs(r["max_abs_j_head_mm"] - coarse[r["route"]]["max_abs_j_head_mm"]) / (15.0 * r["max_abs_j_head_mm"])
                     for r in rows)
    return {"rows": rows, "ranking": ranking, "richardson_rel": richardson}


@functools.lru_cache(maxsize=1)
def calibration_table() -> dict:
    """The T136 table for the packaged fan (cached: T136, T137 and the tests share it)."""
    return calibration_ranking(fan_study())


@_task("T136", ("test_rankings_by_calibration_tolerance_and_focus_margin",
                "test_ranking_evidence_detects_a_wrong_exact_perturbation"))
def rank_by_calibration(ctx):
    table = ctx.memo("mfg.calibration", calibration_table)
    rows = table["rows"]
    # Evidence comes from the independent re-evaluation of the derated corners; the derating loop's own
    # ratios (its stop condition) are compared with it, not checked against the spec by themselves.
    ratios = [max(r["independent_corner_ratios"].values()) for r in rows]
    agreement = max(abs(r["independent_corner_ratios"][corner] - value) for r in rows
                    for corner, value in r["corner_ratios"].items())
    first_ratios = [max(r["first_order_corner_ratios"].values()) for r in rows]
    f_rank = finding("Candidate coupon routes ranked by the heading calibration tolerance that keeps the exactly perturbed "
                     "route within a 0.5 mm lateral spec", "numerical",
                     {"ranking": table["ranking"],
                      "heading_tolerance_mrad": {r["route"]: r["heading_tolerance_mrad"] for r in rows},
                      "lateral_tolerance_mm": {r["route"]: r["lateral_tolerance_mm"] for r in rows},
                      "derating": {r["route"]: r["derating"] for r in rows}},
                     {"derivation": "first order: tolerance = spec / (2 max |j|) per error source (half the spec each, "
                                    "worst case); then derated by the worst exact corner ratio",
                      "checks": [_check("cross_implementation", "max over the four corners of every derated box of |corner "
                                        "ratio from the RK4 exact perturbation - corner ratio re-evaluated with adaptive "
                                        "Dormand-Prince geodesics from a separately coded exp-map start|", agreement, 1e-4),
                                 _check("invariant", "max realized / spec minus one over the four corners of every route's "
                                        "derated box, re-evaluated independently of the derating loop", max(ratios) - 1.0,
                                        0.0, "signed_le"),
                                 _check("invariant", "min over routes of the worst re-evaluated corner ratio (the allocation "
                                        "is not vacuous)", min(ratios), 0.5, "ge"),
                                 _check("self_convergence", "Richardson estimate of max |j_head| (relative)",
                                        table["richardson_rel"], 1e-6, "le")]},
                     uncertainty=_u("truncation_bound", table["richardson_rel"],
                                    "RK4 Richardson estimate of max |j_head| (relative)"),
                     tolerance={"abs": 1e-9, "rel": 1e-6})
    exceed = [r for r in rows if max(r["first_order_corner_ratios"].values()) > 1.0]
    f_first = finding("The first-order tolerance allocation exceeds the spec at a tolerance corner on some off-axis routes",
                      "numerical", {r["route"]: max(r["first_order_corner_ratios"].values()) for r in rows},
                      {"checks": [_check("analytic", "max first-order corner ratio minus one (exceedance)",
                                         max(first_ratios) - 1.0, 0.0, "signed_ge"),
                                  _check("analytic", "routes whose first-order box exceeds the spec", len(exceed), 1, "ge")]},
                      uncertainty=_u("truncation_bound", table["richardson_rel"],
                                     "RK4 Richardson estimate of max |j_head| (relative)"),
                      tolerance={"abs": 1e-9, "rel": 1e-6},
                      counterexample={"statement": "Allocating spec / (2 max |j|) to each error source keeps the exactly "
                                                   "perturbed route within the spec",
                                      "witness": {"route": max(rows, key=lambda r: max(r["first_order_corner_ratios"].values()))["route"],
                                                  "worst_corner_ratio": max(first_ratios)}})
    straight = rows[0]
    f_lens = finding("The dome loosens the heading tolerance of the straight route relative to a flat plate of equal length",
                     "numerical", straight["heading_tolerance_mrad"] / straight["flat_plate_heading_tolerance_mrad"],
                     {"checks": [_check("analytic", "coupon / flat heading tolerance ratio for the straight route",
                                        straight["heading_tolerance_mrad"] / straight["flat_plate_heading_tolerance_mrad"], 1.0, "ge")]},
                     uncertainty=_u("truncation_bound", table["richardson_rel"],
                                    "RK4 Richardson estimate of max |j_head| (relative)"),
                     tolerance={"abs": 1e-9, "rel": 1e-6})
    ctx.artifact_json("calibration-ranking.json", _r(table))
    ctx.artifact_text("heading-tolerance.svg", svg.line_plot(
        [("coupon (derated)", [float(r["route"][3:-3]) for r in rows], [r["heading_tolerance_mrad"] for r in rows]),
         ("coupon (first order)", [float(r["route"][3:-3]) for r in rows], [r["first_order_heading_tolerance_mrad"] for r in rows]),
         ("flat plate, same length", [float(r["route"][3:-3]) for r in rows], [r["flat_plate_heading_tolerance_mrad"] for r in rows])],
        title="Heading calibration tolerance for 0.5 mm lateral error", xlabel="route heading (deg)", ylabel="tolerance (mrad)"))
    findings = [f_rank, f_first, f_lens,
                _not_measured("The robot, fixture and frame calibration achieves the required heading and lateral tolerances",
                              "calibration")]
    fields = _fields(
        "The calibration tolerance a route requires is the inverse of its Jacobi sensitivity: routes whose heading field "
        "j_head grows least tolerate the largest heading calibration error; the first-order allocation must be derated "
        "where second-order terms push a tolerance corner past the spec.",
        "Lateral error e(s) = delta j_lat(s) + dtheta j_head(s) + O(2); first order delta_req = spec / (2 max|j_lat|), "
        "dtheta_req = spec / (2 max|j_head|); both derated by the worst exact corner ratio until all four corners are "
        "within 0.9999 of the spec; flat reference spec / (2 L).",
        ["Fan of geodesic routes from the station (-60, 0) mm at 0..25 deg to the far edge x = 140 mm",
         "Declared lateral spec 0.5 mm"],
        "No observation: required tolerances are model outputs.",
        "At all four corners of each route's derated tolerance box the exactly perturbed route stays within the spec.",
        "Integrate each route with Jacobi fields, allocate tolerances to first order, evaluate the exact perturbation at "
        "the four box corners (RK4), derate where a corner exceeds the spec, re-evaluate every corner of the derated "
        "boxes with adaptive Dormand-Prince geodesics from a separately coded exp-map start, rank.",
        "Ranking by heading tolerance: " + ", ".join(f"{r['route']} {r['heading_tolerance_mrad']:.3f} mrad" for r in
                                                    sorted(rows, key=lambda r: -r["heading_tolerance_mrad"]))
        + f". First-order corner ratios reach {max(first_ratios):.4f} ({', '.join(r['route'] for r in exceed)} derated by "
        + ", ".join(f"{1 - r['derating']:.2%}" for r in exceed) + ").",
        f"Re-evaluated at the corners of the derated boxes the realized / spec lies in [{min(ratios):.3f}, "
        f"{max(ratios):.5f}] (derating aims at {DERATE_TARGET}) and agrees with the derating loop's RK4 ratios to "
        f"{agreement:.1e}; max |j_head| Richardson error {table['richardson_rel']:.1e} relative.",
        ["exact perturbation at all four box corners", "first-order allocation exceeding the spec (derated)",
         "derated corners re-evaluated by a second computation (an error in the RK4 perturbation path would disagree)",
         "flat-plate reference", "mirror symmetry (T137)"],
        ["Tolerances are allocated half to lateral and half to heading error; other splits rescale every heading "
         "tolerance by the same factor and leave the first-order ranking unchanged.",
         "Only the corners of the tolerance box are evaluated exactly; interior points are covered by the linear model "
         "(where the error is maximal at a corner) plus the small second-order terms.",
         "Tape or robot path-following error along the route (after the start) is not included."],
        "Open: optimize the lateral/heading tolerance split per route instead of half each, and add path-following "
        "error after the start, which the corner evaluation omits.")
    return {"state": "completed", "fields": fields, "findings": findings}


def focus_tiers(summaries, key="focal_clearance_ratio", bound="ratio_is_lower_bound") -> list:
    """Routes in tiers of decreasing margin; routes whose margin is only a lower bound share one unresolved tier.

    Their margins are lower bounds set by the integration horizon, so they are
    tied, not ordered; the tier leads only if every lower bound exceeds every
    resolved margin, which the task checks.
    """
    unresolved = sorted(s["route"] for s in summaries if s[bound])
    resolved = sorted((s for s in summaries if not s[bound]), key=lambda s: (-s[key], s["route"]))
    return ([unresolved] if unresolved else []) + [[s["route"]] for s in resolved]


def _tier_text(tiers, unresolved, what="focus") -> str:
    return " > ".join("{" + ", ".join(tier) + f"}} (no {what} within the horizon: lower bounds, tied)"
                      if set(tier) <= set(unresolved) else ", ".join(tier) for tier in tiers)


def focus_margins(routes) -> list:
    """Focus margin s_c - L of each route as T024 defines it: first conjugate point (zero of j_head) minus length.

    A route whose j_head has no zero within its integration horizon has no
    focus margin (None, as T024 records a censored margin) and only the lower
    bound horizon - L, kept under its own key beside the horizon it depends on.
    ``min_j_head_over_s`` is min j_head(s) / s over (0, L]: positive means no
    conjugate point on the route, so the route is locally length minimizing and
    its margin, found or censored, is positive.
    """
    rows = []
    for route in routes:
        states, s = route.transfer.states, route.transfer.s
        conjugate = route.conjugate[0] if route.conjugate else None
        rows.append({"route": route.name, "length_mm": route.length, "first_conjugate_mm": conjugate,
                     "focus_margin_mm": None if conjugate is None else conjugate - route.length,
                     "focus_margin_lower_bound_mm": route.horizon - route.length if conjugate is None else None,
                     "margin_is_lower_bound": conjugate is None, "horizon_mm": route.horizon,
                     "min_j_head_over_s": float(np.min(states[1:, 6] / s[1:]))})
    return rows


def _margin_or_bound(row) -> float:
    """The focus margin of a route, or its lower bound when censored (for comparisons of like with like only)."""
    return row["focus_margin_lower_bound_mm"] if row["margin_is_lower_bound"] else row["focus_margin_mm"]


@_task("T137", ("test_rankings_by_calibration_tolerance_and_focus_margin",))
def rank_by_focus_margin(ctx):
    fan = ctx.memo("mfg.fan", fan_study)
    summaries, coarse = fan["summaries"], {c["route"]: c for c in fan["coarse"]}
    # The focus margin (T024): s_c - L over conjugate points only.
    margins = focus_margins(fan["routes"])
    margin_ranking = focus_tiers(margins, "focus_margin_mm", "margin_is_lower_bound")
    mirror, plus5 = fan["mirror"], next(s for s in summaries if s["route"] == "fan+5deg")
    # The mirror route and fan+5deg must agree in censoring and in the margin (or its lower bound).
    plus5_row = next(m for m in margins if m["route"] == "fan+5deg")
    mirror_censored = mirror["first_conjugate_mm"] is None
    mirror_difference = (((mirror["horizon_mm"] if mirror_censored else mirror["first_conjugate_mm"]) - mirror["length_mm"])
                         - _margin_or_bound(plus5_row)) if mirror_censored == plus5_row["margin_is_lower_bound"] else 1.0
    conjugate_agree = all((m["first_conjugate_mm"] is None) == (coarse[m["route"]]["first_conjugate_mm"] is None)
                          for m in margins)
    length_drift = max(abs(s["length_mm"] - coarse[s["route"]]["length_mm"]) for s in summaries)
    positive = min(m["min_j_head_over_s"] for m in margins)
    f_margin = finding("Candidate coupon routes ranked by focus margin s_c - L (first conjugate point, a zero of j_head, "
                       "minus the route length, as T024 defines it)", "numerical",
                       {"ranking": margin_ranking, "focus_margin_mm": {m["route"]: m["focus_margin_mm"] for m in margins},
                        "lower_bound": {m["route"]: m["margin_is_lower_bound"] for m in margins},
                        "focus_margin_lower_bound_mm": {m["route"]: m["focus_margin_lower_bound_mm"] for m in margins},
                        "horizon_mm": {m["route"]: m["horizon_mm"] for m in margins},
                        "min_j_head_over_s": {m["route"]: m["min_j_head_over_s"] for m in margins}},
                       {"checks": [_check("analytic", "min over routes and s in (0, L] of j_head(s) / s (positive: no "
                                          "conjugate point on any route, so every focus margin is positive)", positive,
                                          0.1, "ge"),
                                   _check("exact_arithmetic", "conjugate-found flags agree at h = 1 and 2 mm",
                                          0.0 if conjugate_agree else 1.0, 0.0),
                                   _check("invariant", "mirror route -5 deg has the +5 deg focus margin, or the same "
                                          "lower bound when both are censored (mm)", mirror_difference, 1e-9)]},
                       unit="mm",
                       uncertainty=_u("truncation_bound", length_drift / 15.0,
                                      "RK4 Richardson estimate of the route lengths (mm); lower bounds depend on the horizon"),
                       tolerance={"abs": 1e-6, "rel": 1e-6})
    # The focal clearance ratio: nearest zero of j_lat or j_head over the route length.
    ranking = focus_tiers(summaries)
    unresolved = sorted(s["route"] for s in summaries if s["ratio_is_lower_bound"])
    resolved_margins = [s["focal_clearance_ratio"] for s in summaries if not s["ratio_is_lower_bound"]]
    bound_lead = (min(s["focal_clearance_ratio"] for s in summaries if s["ratio_is_lower_bound"]) - max(resolved_margins)
                  if unresolved and resolved_margins else 0.0)
    focus_drift = max(abs(s["nearest_focus_mm"] - coarse[s["route"]]["nearest_focus_mm"])
                      for s in summaries if s["nearest_focus_mm"] is not None)
    bound_agree = all(s["ratio_is_lower_bound"] == coarse[s["route"]]["ratio_is_lower_bound"] for s in summaries)
    shortest = min(summaries, key=lambda s: s["length_mm"])
    shortest_margin = next(m for m in margins if m["route"] == shortest["route"])
    worst = ranking[-1][0] if len(ranking[-1]) == 1 and ranking[-1][0] not in unresolved else None
    f_rank = finding("Candidate coupon routes ranked by focal clearance ratio (nearest focal or conjugate point, a zero "
                     "of j_lat or j_head, over the route length)",
                     "numerical", {"ranking": ranking, "unresolved_tier": unresolved,
                                   "focal_clearance_ratio": {s["route"]: s["focal_clearance_ratio"] for s in summaries},
                                   "lower_bound": {s["route"]: s["ratio_is_lower_bound"] for s in summaries}},
                     {"checks": [_check("self_convergence", "nearest-focus distance at h = 1 vs 2 mm (mm)", focus_drift, 1e-2),
                                 _check("exact_arithmetic", "focus-found flags agree at h = 1 and 2 mm", 0.0 if bound_agree else 1.0, 0.0),
                                 _check("invariant", "mirror route -5 deg has the +5 deg focal clearance ratio",
                                        mirror["focal_clearance_ratio"] - plus5["focal_clearance_ratio"], 1e-9),
                                 _check("analytic", "smallest lower-bound ratio of the unresolved tier minus the largest "
                                        "resolved ratio (the tied tier leads)", bound_lead, 0.0, "signed_ge")]},
                     uncertainty=_u("truncation_bound", focus_drift / 15.0,
                                    "RK4 Richardson estimate of focus locations (mm)"),
                     tolerance={"abs": 1e-6, "rel": 1e-6})
    f_counter = finding("The shortest candidate route has the lowest focal clearance ratio", "numerical",
                        {"shortest": shortest["route"], "length_mm": shortest["length_mm"],
                         "focal_clearance_ratio": shortest["focal_clearance_ratio"],
                         "focus_margin_mm": shortest_margin["focus_margin_mm"],
                         "focus_margin_lower_bound_mm": shortest_margin["focus_margin_lower_bound_mm"]},
                        {"checks": [_check("exact_arithmetic", "shortest route is last in the focal-clearance ranking",
                                           0.0 if worst == shortest["route"] else 1.0, 0.0),
                                    _check("analytic", "its focal point lies inside the route (focal clearance ratio)",
                                           shortest["focal_clearance_ratio"], 1.0, "le")]},
                        uncertainty=_u("truncation_bound", focus_drift / 15.0,
                                       "RK4 Richardson estimate of focus locations (mm)"),
                        tolerance={"abs": 1e-6, "rel": 1e-7},
                        counterexample={"statement": "The shortest route between a station and an edge is also the one "
                                                     "farthest, relative to its length, from a focal or conjugate point "
                                                     "(largest focal clearance ratio)",
                                        "witness": {"route": shortest["route"], "length_mm": shortest["length_mm"],
                                                    "ranked_by": "focal clearance ratio (zeros of j_lat or j_head) / L",
                                                    "nearest_focus_mm": shortest["nearest_focus_mm"],
                                                    "nearest_focus_kind": shortest["nearest_focus_kind"],
                                                    "focus_margin_mm": shortest_margin["focus_margin_mm"],
                                                    "focus_margin_lower_bound_mm":
                                                        shortest_margin["focus_margin_lower_bound_mm"],
                                                    "next_longer": sorted(summaries, key=lambda s: s["length_mm"])[1]["route"]}})
    # Second variation of length for routes to the edge line: L''(0) = j_head(L) j_head'(L) (straight end line).
    straight = next(s for s in summaries if s["route"] == "fan+0deg")
    base = straight["length_mm"]
    d = {a: 2.0 * (length - base) / math.radians(a) ** 2 for a, length in fan["small_lengths"].items()}
    extrapolated = (4.0 * d[1.0] - d[2.0]) / 3.0
    predicted = straight["j_head_end_mm"] * straight["j_head_prime_end"]
    f_variation = finding("The second variation of route length equals the Jacobi index form j_head(L) j_head'(L)", "numerical",
                          {"finite_difference_mm": extrapolated, "index_form_mm": predicted},
                          {"checks": [_check("analytic", "relative difference (Richardson finite difference vs index form)",
                                             extrapolated / predicted - 1.0, 2e-4),
                                      _check("analytic", "straight route is a local length minimum (L'' > 0)", predicted, 0.0, "ge")]},
                          unit="mm",
                          uncertainty=_u("reference_error", abs(extrapolated - predicted),
                                         "residual of the Richardson-extrapolated finite difference against the index "
                                         "form (mm)"),
                          tolerance={"abs": 1e-6, "rel": 1e-5})
    calibration = ctx.memo("mfg.calibration", calibration_table)["ranking"]
    f_conflict = finding("The calibration-tolerance ranking and the focal-clearance ranking put the straight route at "
                         "opposite ends", "numerical", {"calibration_first": calibration[0], "focal_clearance_last": worst},
                         {"checks": [_check("exact_arithmetic", "first by calibration tolerance is last by focal clearance "
                                            "ratio", 0.0 if calibration[0] == worst else 1.0, 0.0)]},
                         uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    ctx.artifact_json("focus-ranking.json", _r({"focus_margin": {"definition": "s_c - L, s_c the first zero of j_head "
                                                                               "(T024); None when no zero lies within "
                                                                               "the horizon, with the lower bound "
                                                                               "horizon - L under its own key",
                                                                 "ranking": margin_ranking, "routes": margins},
                                                "focal_clearance_ratio": {"definition": "nearest zero of j_lat or j_head "
                                                                                        "over L; horizon / L as a lower bound",
                                                                          "ranking": ranking},
                                                "routes": summaries, "coarse": fan["coarse"],
                                                "second_variation": {"finite_difference": d, "extrapolated": extrapolated,
                                                                     "index_form": predicted},
                                                "calibration_ranking": calibration}))
    ctx.artifact_text("margin-vs-length.svg", svg.line_plot(
        [("focal clearance ratio (lower bound when no focus)", [s["length_mm"] for s in summaries],
          [s["focal_clearance_ratio"] for s in summaries]),
         ("ratio = 1 (focus at the route end)", [summaries[0]["length_mm"], summaries[-1]["length_mm"]], [1.0, 1.0])],
        title="Coupon routes: focal clearance ratio vs length", xlabel="route length (mm)", ylabel="s_focus / L"))
    findings = [f_margin, f_rank, f_counter, f_variation, f_conflict,
                _not_measured("Physical paths near a predicted focus show the predicted loss of lateral-error ordering")]
    fields = _fields(
        "Two margins rank the routes differently. The focus margin s_c - L (T024: first conjugate point, a zero of "
        "j_head, minus the route length) is positive for every candidate route, so each is locally length "
        "minimizing to the edge; the focal clearance ratio (nearest zero of j_lat or j_head over L) separates them and "
        "ranks routes differently from length: the straight route over the dome is the shortest candidate route to the "
        "far edge yet has a focal point of lateral offsets inside it.",
        "Conjugate points: zeros of j_head (s > 0); focal points: zeros of j_lat. Focus margin = s_c - L, recorded as "
        "no value when censored (no conjugate point within the horizon), with the lower bound horizon - L kept "
        "separately; focal clearance ratio = s_focus / L with s_focus the nearest focal or conjugate point, or "
        "horizon / L as a lower bound; L''(0) = j_head(L) j_head'(L) for routes from a point to a straight edge line.",
        ["Fan of geodesic routes (T136) at h = 1 and 2 mm; horizon 2.5 x the chart reach",
         "Routes at 1 and 2 deg for the second variation"],
        "No observation: margins and lengths are model outputs.",
        "Focus and conjugate locations and their found flags stable under step halving; mirror symmetry; j_head > 0 on "
        "(0, L]; index form = second variation.",
        "Locate focal and conjugate points by Hermite zeros, rank by the focus margin and by the focal clearance "
        "ratio, compare with lengths and with the T136 ranking, and verify the Jacobi second-variation formula by "
        "Richardson finite differences of route length.",
        f"focus margin: {_tier_text(margin_ranking, [m['route'] for m in margins if m['margin_is_lower_bound']], 'conjugate point')}, "
        f"censored margins (no value) with lower bounds from "
        f"{min((m['focus_margin_lower_bound_mm'] for m in margins if m['margin_is_lower_bound']), default=math.nan):.1f} "
        f"mm (min j_head(s) / s = {positive:.3f}); "
        f"focal clearance ratio: {_tier_text(ranking, unresolved)}; shortest {shortest['route']} "
        f"({shortest['length_mm']:.4f} mm) has ratio {shortest['focal_clearance_ratio']:.3f} with a {shortest['nearest_focus_kind']} "
        f"point at {shortest['nearest_focus_mm']:.2f} mm and focus margin "
        + (f">= {shortest_margin['focus_margin_lower_bound_mm']:.1f} mm (censored)"
           if shortest_margin["margin_is_lower_bound"] else f"{shortest_margin['focus_margin_mm']:.1f} mm") + "; "
        f"L''(0) finite difference {extrapolated:.4f} mm vs index form {predicted:.4f} mm.",
        f"Focus locations agree to {focus_drift:.1e} mm and route lengths to {length_drift:.1e} mm between step sizes; "
        "lower-bound margins and ratios depend on the horizon.",
        ["step refinement", "mirror symmetry", "lower bounds (no focus or conjugate point within the horizon)",
         "second-variation identity", "positivity of j_head along every route"],
        ["No candidate route has a conjugate point within 2.5 x its reach, so the focus margin only bounds them from "
         "below and ties them; the focal clearance ratio, which includes zeros of j_lat, is the quantity that ranks.",
         "The straight route is a local length minimum (L'' > 0, positive focus margin) although it contains a lateral "
         "focal point; a focal point of the start normal does not contradict minimality to the edge line.",
         "The fan is a discrete candidate set; margins between sampled headings are not bounded."],
        "Open: bound the focus margin and the focal clearance ratio between the sampled fan headings with a continuous "
        "heading sweep that tracks the zeros of j_head and j_lat, which the discrete fan leaves unbounded.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T138 predicted versus measured separation ----------------------------------------------------
DOME_TOLERANCE = {"height_mm": 0.2, "sigma_mm": 0.5}


DOME_CENTRE_STEP_MM = 0.1


@functools.lru_cache(maxsize=1)
def sensitivity_study() -> dict:
    """Geometry sensitivity of the nominal-route predictions to the dome height, width and centre.

    Height and width are differenced at their declared-tolerance standard
    uncertainties (so "central" is the tolerance-based standard uncertainty);
    the centre (x0, y0) at 0.1 mm, by moving the start the other way. The
    derivative is central / step (see "steps"), which the scan-conditioned terms use.
    """
    nominal = nominal_study()
    length = nominal["length_mm"]
    steps = NOMINAL_STATIONS * 26
    stride = steps // NOMINAL_STATIONS
    standard = {"height_mm": DOME_TOLERANCE["height_mm"] / math.sqrt(3.0), "sigma_mm": DOME_TOLERANCE["sigma_mm"] / math.sqrt(3.0)}
    differences = dict(standard, x0_mm=DOME_CENTRE_STEP_MM, y0_mm=DOME_CENTRE_STEP_MM)

    def evaluate(height, sigma, x0=0.0, y0=0.0):
        start = (geo.STATION[0] - x0, geo.STATION[1] - y0)  # a dome moved by (x0, y0) is a start moved the other way
        transfer = jacobi.transfer(geo.coupon(height, sigma), start, 0.0, 2 * length, steps=2 * steps)
        focal = transfer.focal_points()[0]
        stations = transfer.states[:steps + 1:stride]
        return {"lateral": 2.0 * stations[:, 4], "heading_end": 0.005 * stations[-1, 6], "focal": focal,
                "j_lat_prime_focal": float(np.interp(focal, transfer.s, transfer.states[:, 5]))}

    def moved(name, amount):
        return evaluate(geo.DOME_HEIGHT + (amount if name == "height_mm" else 0.0),
                        geo.DOME_SIGMA + (amount if name == "sigma_mm" else 0.0),
                        amount if name == "x0_mm" else 0.0, amount if name == "y0_mm" else 0.0)

    center = evaluate(geo.DOME_HEIGHT, geo.DOME_SIGMA)
    out = {"standard_uncertainty": standard, "steps": differences,
           "center": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in center.items()}}
    for name, step in differences.items():
        plus, minus = moved(name, step), moved(name, -step)
        out[name] = {key: {"central": ((plus[key] - minus[key]) / 2.0).tolist() if isinstance(plus[key], np.ndarray)
                                      else (plus[key] - minus[key]) / 2.0,
                           "forward": ((plus[key] - center[key])).tolist() if isinstance(plus[key], np.ndarray)
                                      else plus[key] - center[key]}
                     for key in ("lateral", "heading_end", "focal")}
    return out


def scan_geometry(sensitivity, key, covariance):
    """Standard uncertainty of a prediction from the scan-derived dome covariance: sqrt(g^T C g), g = central / step.

    ``covariance`` is over (height, sigma, x0, y0) as returned by :func:`as_built_study`.
    """
    names = ("height_mm", "sigma_mm", "x0_mm", "y0_mm")
    gradient = [np.asarray(sensitivity[name][key]["central"], dtype=float) / sensitivity["steps"][name] for name in names]
    variance = sum(covariance[i][j] * gradient[i] * gradient[j] for i in range(4) for j in range(4))
    return np.sqrt(np.maximum(variance, 0.0))


def _linearity(forward, central, floor=1e-6) -> float:
    """max |forward / central - 1| over entries whose central difference exceeds ``floor``."""
    forward, central = np.atleast_1d(np.asarray(forward, dtype=float)), np.atleast_1d(np.asarray(central, dtype=float))
    keep = np.abs(central) > floor
    return float(np.max(np.abs(forward[keep] / central[keep] - 1.0))) if keep.any() else 0.0


@functools.lru_cache(maxsize=1)
def separation_prediction() -> dict:
    """Predicted separation of the 2 mm offset tape with its uncertainty components at the stations.

    Components (standard uncertainties, mm): geometry from the dome tolerances
    (central differences, T140), solver from RK4 Richardson, and the start pose of
    the offset tape either open loop (declared insert and laying error and two
    seatings of the jig, kept as separate terms) or conditioned on its CMM
    estimate. Start-pose sensitivities are central
    differences of the exactly re-integrated offset route at the declared step,
    so they include the nonlinearity of a 2 mm offset.
    """
    nominal = nominal_study()
    sensitivity = sensitivity_study()
    length, lateral = nominal["length_mm"], nominal["lateral_mm"]
    steps = NOMINAL_STATIONS * 26
    stride = steps // NOMINAL_STATIONS
    base = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps)

    def offset_route(delta, dheading):
        return geo.separation_nonlinear(geo.COUPON, geo.STATION, 0.0, length, steps, delta, dheading, base=base)[::stride]

    center = np.array(nominal["lateral_nonlinear_mm"])
    start = {}
    for name, (dd, dt) in {"lateral": (EXECUTION["lateral_mm"], 0.0), "heading": (0.0, EXECUTION["heading_rad"])}.items():
        step = dd or dt
        plus, minus = offset_route(lateral + dd, dt), offset_route(lateral - dd, -dt)
        start[name] = {"central": (plus - minus) / (2.0 * step), "forward": (plus - center) / step}
    geometry = np.hypot(np.array(sensitivity["height_mm"]["lateral"]["central"]),
                        np.array(sensitivity["sigma_mm"]["lateral"]["central"]))
    # RK4 Richardson: the error of the h-solution is about |Q(h) - Q(2h)| / 15.
    solver = lateral * np.abs(np.array(nominal["j_lat"]) - np.array(nominal["j_lat_half_step"])) / 15.0

    def start_term(u):
        return np.hypot(u["lateral_mm"] * start["lateral"]["central"], u["heading_rad"] * start["heading"]["central"])

    execution, conditioning = start_term(EXECUTION), start_term(START_POSE_U)
    execution_terms = {name: start_term(u) for name, u in EXECUTION["terms"].items()}
    # After the MFG-SCAN-01 scan (T129) the dome tolerances give way to the scan-derived dome covariance.
    geometry_scan = scan_geometry(sensitivity, "lateral", as_built_study()["covariance_mm2"])
    return {"stations_mm": nominal["stations_mm"], "separation_mm": center.tolist(), "offset_mm": lateral,
            "geometry_mm": geometry.tolist(), "solver_mm": solver.tolist(), "execution_mm": execution.tolist(),
            "execution_terms_mm": {name: term.tolist() for name, term in execution_terms.items()},
            "conditioning_mm": conditioning.tolist(), "geometry_scan_mm": geometry_scan.tolist(),
            "open_loop_expanded_mm": (COVERAGE_K * np.sqrt(geometry ** 2 + solver ** 2 + execution ** 2)).tolist(),
            "conditioned_expanded_mm": (COVERAGE_K * np.sqrt(geometry ** 2 + solver ** 2 + conditioning ** 2)).tolist(),
            "scan_conditioned_expanded_mm": (COVERAGE_K * np.sqrt(geometry_scan ** 2 + solver ** 2
                                                                  + conditioning ** 2)).tolist(),
            "start_sensitivity": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in start.items()},
            "linearity": {"geometry/height_mm": _linearity(sensitivity["height_mm"]["lateral"]["forward"],
                                                           sensitivity["height_mm"]["lateral"]["central"]),
                          "geometry/sigma_mm": _linearity(sensitivity["sigma_mm"]["lateral"]["forward"],
                                                          sensitivity["sigma_mm"]["lateral"]["central"]),
                          "start/lateral": _linearity(start["lateral"]["forward"], start["lateral"]["central"]),
                          "start/heading": _linearity(start["heading"]["forward"], start["heading"]["central"])}}


def start_error_scenario() -> dict:
    """A correct model compared with a tape realized 2 sigma of the declared open-loop start error off its offset.

    The 'measured' separations are the exactly re-integrated realized route: model
    output standing in for a perfect instrument, not a measurement.
    """
    nominal = nominal_study()
    prediction = separation_prediction()
    length, steps = nominal["length_mm"], NOMINAL_STATIONS * 26
    stride = steps // NOMINAL_STATIONS
    base = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps)

    def route(delta, dheading):
        return geo.separation_nonlinear(geo.COUPON, geo.STATION, 0.0, length, steps, delta, dheading, base=base)[::stride]

    realized_delta = nominal["lateral_mm"] + 2.0 * EXECUTION["lateral_mm"]
    realized = route(realized_delta, 0.0)
    u_m = COVERAGE_K * PAIR_U
    predicted = np.array(prediction["separation_mm"])
    no_execution = COVERAGE_K * np.hypot(np.array(prediction["geometry_mm"]), np.array(prediction["solver_mm"]))
    conditioned_u = np.array(prediction["conditioned_expanded_mm"])
    stations = slice(1, None)  # station 0 is where the start offset itself is read
    # Measured branch: the prediction is re-integrated from a CMM estimate of the realized start pose that is
    # off by 2 sigma of the estimate in lateral offset and heading (all four sign corners).
    corners = {}
    for sl, sh in itertools.product((1.0, -1.0), repeat=2):
        estimate = route(realized_delta + sl * COVERAGE_K * START_POSE_U["lateral_mm"],
                         sh * COVERAGE_K * START_POSE_U["heading_rad"])
        corners[f"{sl:+.0f}{sh:+.0f}"] = rec.normalized_error(realized[stations], estimate[stations], u_m,
                                                              conditioned_u[stations]).tolist()
    return {"realized_offset_mm": realized_delta, "realized_mm": realized.tolist(),
            "en_without_execution": rec.normalized_error(realized[stations], predicted[stations], u_m,
                                                         no_execution[stations]).tolist(),
            "en_open_loop": rec.normalized_error(realized[stations], predicted[stations], u_m,
                                                 np.array(prediction["open_loop_expanded_mm"])[stations]).tolist(),
            "en_conditioned_corners": corners,
            "en_conditioned": np.max(np.array(list(corners.values())), axis=0).tolist()}


def _schema_fixture():
    raw = b"SCHEMA FIXTURE - NOT A MEASUREMENT\n"
    identity = np.eye(3).tolist()
    covariance = np.diag([1e-4] * 3 + [1e-10] * 3).tolist()
    record = {"schema": rec.RETENTION_SCHEMA, "record_kind": "schema_fixture", "protocol_id": "MFG-COUPON-01",
              "raw": [{"name": "fixture.txt", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "media_type": "text/plain"}],
              "instrument": {"id": "camera", "kind": "photogrammetry camera system", "serial": "FIXTURE-0000"},
              "calibration": {"status": "applied", "reference": "CERT-FIXTURE", "sha256": "0" * 64,
                              "valid_from": "2026-01-01T00:00:00Z", "valid_until": "2027-01-01T00:00:00Z"},
              "frame_chain": [dict(link, rotation=identity, translation_mm=[0.0, 0.0, 0.0], covariance=covariance)
                              for link in FRAME_CHAIN],
              "clock": {"source": "fixture", "acquired_at": "2026-09-23T00:00:00Z", "synchronization": "none",
                        "uncertainty_s": 1.0}}
    return record, {"fixture.txt": raw}


def pair_predictions(plate, cylinder) -> dict:
    """Predicted marker-pair quantities for the pair comparator (manufacturing_records.compare_pair_distances).

    Plate: the chord of every T126 pair equals its geodesic distance; the declared
    flatness enters at second order (T140), so U_p = 0. Cylinder: the T127
    chord-geodesic gaps (measured as film surface distance minus camera chord,
    H1) with U_p = k |d gap / dR| u_R for the declared radius tolerance
    (u_R = 0.1 / sqrt(3) mm), and the chords themselves with
    U_p = k |d chord / dR| u_R beside the geodesic distances they differ from (H3).
    """
    u_radius = 0.1 / math.sqrt(3.0)
    return {"MFG-FLAT-PLATE-01": {"pairs": ["-".join(r["pair"]) for r in plate["pairs"]],
                                  "values_mm": [r["geodesic_mm"] for r in plate["pairs"]],
                                  "expanded_uncertainty_mm": [0.0] * len(plate["pairs"])},
            "MFG-CYLINDER-01": {"pairs": [r["pair"] for r in cylinder["pairs"]],
                                "values_mm": [r["gap_mm"] for r in cylinder["pairs"]],
                                "expanded_uncertainty_mm": [
                                    COVERAGE_K * abs(gap_radius_derivative(r["dphi_rad"], r["dz_mm"], geo.CYLINDER_RADIUS))
                                    * u_radius for r in cylinder["pairs"]]},
            "MFG-CYLINDER-01 chords": {"pairs": [r["pair"] for r in cylinder["pairs"]],
                                       "values_mm": [r["chord_mm"] for r in cylinder["pairs"]],
                                       "geodesic_mm": [r["geodesic_mm"] for r in cylinder["pairs"]],
                                       "expanded_uncertainty_mm": [
                                           COVERAGE_K * abs(chord_radius_derivative(r["dphi_rad"], r["dz_mm"],
                                                                                    geo.CYLINDER_RADIUS)) * u_radius
                                           for r in cylinder["pairs"]]}}


# Operator captures of the protocols (read by T138) --------------------------------------------
CAPTURE_ROLES = ("photogrammetry", "cmm", "film")
PLATE_PATHS, CYLINDER_PATHS, COUPON_PATHS = ("N0", "L2", "H5"), ("HX45", "HX45-L2"), ("N", "L", "H")


def _pose_ids(paths) -> list:
    return [f"{path}-S{s}" for path in paths for s in (0, 20)]


def capture_expectations(protocol: str) -> dict:
    """What each capture role of a protocol must declare and hold (manufacturing_records.read_capture ``expected``)."""
    if protocol == "MFG-FLAT-PLATE-01":
        roles = {"photogrammetry": (TARGET_CAPTURE, "camera", [f"M{k:02d}" for k in range(25)]),
                 "cmm": (TARGET_CAPTURE, "cmm", _pose_ids(PLATE_PATHS))}
    elif protocol == "MFG-CYLINDER-01":
        cylinder = cylinder_study()
        roles = {"photogrammetry": (TARGET_CAPTURE, "camera", [m["id"] for m in cylinder["markers"] if m["faces_up"]]),
                 "film": (DISTANCE_CAPTURE, "film", [r["pair"] for r in cylinder["pairs"]]),
                 "cmm": (TARGET_CAPTURE, "cmm", _pose_ids(CYLINDER_PATHS))}
    elif protocol == "MFG-COUPON-01":
        roles = {"photogrammetry": (TARGET_CAPTURE, "camera", COUPON_TARGETS),
                 "cmm": (TARGET_CAPTURE, "cmm", _pose_ids(COUPON_PATHS))}
    else:
        raise rec.RecordRefusal("capture_protocol_unsupported", f"T138 compares no capture of protocol {protocol!r}")
    return {role: {"schema": schema, "protocol": protocol, "instrument": instrument,
                   "frame": "CAD" if schema == TARGET_CAPTURE else "surface", "ids": ids}
            for role, (schema, instrument, ids) in roles.items()}


def _coupon_states(lateral, dheading, span, steps) -> np.ndarray:
    """RK4 geodesic states of a tape started at the station with a relative start pose (exp-map lateral offset)."""
    start = jacobi.perturbed_start(geo.COUPON, geo.STATION, 0.0, lateral=lateral, heading_change=dheading)
    return integrators.integrate_fixed(geo.COUPON.geodesic_rhs, start, span, steps, "rk4")[1]


def synthetic_capture(protocol: str, role: str, lateral_mm: float = 2.0, dheading_rad: float = 0.0,
                      noise_mm: float = 0.0, seed: int = SEED + 11) -> bytes:
    """A synthetic operator capture: model output standing in for an instrument export, never a measurement.

    The bytes start with the synthetic banner and declare origin synthetic, so
    read_capture reports them as synthetic and to_acquisition refuses them.
    Coupon targets lie on the exactly integrated nominal tape and on tapes
    started ``lateral_mm`` and ``dheading_rad`` off it (lateral tape L) and 5
    mrad off it (heading tape H); plate and cylinder targets are the declared
    markers and film readings the geodesic distances. ``noise_mm`` adds seeded
    Gaussian noise to every coordinate or distance.
    """
    spec = capture_expectations(protocol)[role]
    generator = met.rng(seed)
    u = INSTRUMENTS[spec["instrument"]]["declared_standard_uncertainty_mm"]
    if protocol == "MFG-COUPON-01":
        tapes = {"N": (0.0, 0.0), "L": (lateral_mm, dheading_rad), "H": (0.0, 0.005)}
        length, steps = nominal_study()["length_mm"], NOMINAL_STATIONS * 26
        stride = steps // NOMINAL_STATIONS
        points = {}
        for tape, (lateral, dheading) in tapes.items():
            if role == "photogrammetry":
                states = _coupon_states(lateral, dheading, length, steps)[::stride]
                points.update({f"{tape}{k}": geo.COUPON.embedding(y[:2]) for k, y in enumerate(states)})
            else:
                states = _coupon_states(lateral, dheading, START_BASELINE_MM, 20)
                points.update({f"{tape}-S0": geo.COUPON.embedding(states[0, :2]),
                               f"{tape}-S20": geo.COUPON.embedding(states[-1, :2])})
    elif protocol == "MFG-FLAT-PLATE-01" and role == "photogrammetry":
        points = {m["id"]: geo.PLATE.embedding(np.array(m["u_mm"], dtype=float)) for m in plate_study()["markers"]}
    elif protocol == "MFG-CYLINDER-01" and role == "photogrammetry":
        points = {m["id"]: geo.CYLINDER.embedding(np.array(m["u"], dtype=float)) for m in cylinder_study()["markers"]}
    elif protocol == "MFG-CYLINDER-01" and role == "film":
        rows = {r["pair"]: [r["geodesic_mm"] + (generator.normal(0.0, noise_mm) if noise_mm else 0.0), u]
                for r in cylinder_study()["pairs"]}
        return rec.write_capture(spec["schema"], protocol, spec["instrument"], spec["frame"], rows)
    else:
        raise ValueError(f"No synthetic {role} capture is defined for {protocol}")
    rows = {name: [*(np.asarray(p) + (generator.normal(0.0, noise_mm, 3) if noise_mm else 0.0)), u]
            for name, p in points.items() if name in spec["ids"]}
    return rec.write_capture(spec["schema"], protocol, spec["instrument"], spec["frame"], rows)


START_BASELINE_MM = 20.0


def estimate_start_pose(rows, nominal="N", offset="L") -> tuple[float, float]:
    """Relative start pose (lateral mm, heading rad) of the offset tape from CMM points at s = 0 and 20 mm.

    The separations e(0) and e(20) of the offset tape's points from the nominal
    tape's, each projected on the model's in-surface normal of the nominal
    route there, give the pose through the Jacobi fields of the nominal route:
    e(0) = delta and e(20) = delta j_lat(20) + dtheta j_head(20). A plain angle
    between the two 20 mm chords would be biased by delta (j_lat(20) - 1) / 20,
    about 0.1 mrad at the station, half the CMM heading uncertainty.
    """
    start = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, START_BASELINE_MM, steps=20)
    point = {name: np.asarray(values[:3], dtype=float) for name, values in rows.items()}
    separation = []
    for index, s in ((0, 0), (-1, 20)):
        y = start.states[index]
        normal = geo.COUPON.embedding_jacobian(y[:2]) @ geo.COUPON.normal(y[:2], y[2:4])
        separation.append(float((point[f"{offset}-S{s}"] - point[f"{nominal}-S{s}"]) @ normal / np.linalg.norm(normal)))
    j_lat, j_head = start.states[-1, 4], start.states[-1, 6]
    return separation[0], float((separation[1] - separation[0] * j_lat) / j_head)


def coupon_separations(rows, offset="L") -> tuple[np.ndarray, np.ndarray]:
    """Separations of the offset tape's station targets from the nominal tape's, and their standard uncertainties.

    Each is the difference of the two targets at a station projected on the
    model's unit in-surface normal of the nominal route there; it equals the
    normal separation of the exactly offset route to second order (checked in
    T138). u = sqrt(u_offset^2 + u_nominal^2) from the capture's u column.
    """
    length, steps = nominal_study()["length_mm"], NOMINAL_STATIONS * 26
    base = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps).states[::steps // NOMINAL_STATIONS]
    values, sigmas = [], []
    for k, y in enumerate(base):
        normal = geo.COUPON.embedding_jacobian(y[:2]) @ geo.COUPON.normal(y[:2], y[2:4])
        a, b = np.asarray(rows[f"N{k}"], dtype=float), np.asarray(rows[f"{offset}{k}"], dtype=float)
        values.append(float((b[:3] - a[:3]) @ normal / np.linalg.norm(normal)))
        sigmas.append(math.hypot(a[3], b[3]))
    return np.array(values), np.array(sigmas)


def _crossing(stations, values, sigmas=None):
    """First arclength where the separation changes sign, by linear interpolation between stations, and its
    standard uncertainty from the ``sigmas`` of the two bracketing separations; (None, None) if there is none.

    s* = s_k + ds v_k / D with D = v_k - v_{k+1}, so ds*/dv_k = -ds v_{k+1} / D^2 and ds*/dv_{k+1} = ds v_k / D^2:
    u(s*) = ds sqrt(v_{k+1}^2 u_k^2 + v_k^2 u_{k+1}^2) / D^2, about u_separation / |slope|.
    """
    for k in range(len(values) - 1):
        if values[k] > 0.0 >= values[k + 1]:
            step, spread = stations[k + 1] - stations[k], values[k] - values[k + 1]
            where = float(stations[k] + values[k] / spread * step)
            if sigmas is None:
                return where, None
            return where, float(step * math.hypot(values[k + 1] * sigmas[k], values[k] * sigmas[k + 1]) / spread ** 2)
    return None, None


def _pair_chords(rows, pairs):
    chords, sigmas = [], []
    for a, b in pairs:
        pa, pb = np.asarray(rows[a], dtype=float), np.asarray(rows[b], dtype=float)
        chords.append(float(np.linalg.norm(pb[:3] - pa[:3])))
        sigmas.append(math.hypot(pa[3], pb[3]))
    return np.array(chords), np.array(sigmas)


def _en_summary(labels, measured, predicted, u_measured, u_predicted) -> dict:
    en = rec.normalized_error(measured, predicted, u_measured, u_predicted)
    return {"labels": list(labels), "measured_mm": measured.tolist(), "predicted_mm": np.asarray(predicted).tolist(),
            "expanded_uncertainty_measured_mm": np.asarray(u_measured).tolist(),
            "expanded_uncertainty_predicted_mm": np.asarray(u_predicted).tolist(), "normalized_error": en.tolist(),
            "max_normalized_error": float(np.max(en)), "within_en_1": bool(np.all(en <= 1.0))}


def compare_captures(raw: dict) -> dict:
    """Parse bound captures and compare them with the prediction of their protocol (E_n); refuses what it cannot read.

    ``raw`` maps capture roles to bytes. Returns the protocol, per-role header
    and digest, and the comparisons: coupon separations at stations 1..8
    against the open-loop prediction and, with a cmm capture, against the
    prediction re-integrated from the estimated start pose; plate marker-pair
    chords; cylinder pair chords against the predicted chords and geodesics
    and, with a film capture, film minus chord against the predicted gaps. A
    computational comparison of unauthenticated bytes, never a measurement.
    """
    headers = {role: rec.read_capture(data) for role, data in sorted(raw.items())}
    protocols = {header["protocol"] for header in headers.values()}
    if len(protocols) != 1:
        raise rec.RecordRefusal("capture_protocol_mismatch", f"Bound captures name different protocols: {sorted(protocols)}")
    protocol = protocols.pop()
    expectations = capture_expectations(protocol)
    unknown = sorted(set(raw) - set(expectations))
    if unknown:
        raise rec.RecordRefusal("capture_role_unsupported", f"Protocol {protocol} defines no capture role {unknown}")
    parsed = {role: rec.read_capture(data, expectations[role]) for role, data in sorted(raw.items())}
    out = {"protocol": protocol,
           "captures": {role: {"sha256": hashlib.sha256(raw[role]).hexdigest(), "bytes": len(raw[role]),
                               "origin": parsed[role]["origin"], "instrument": parsed[role]["instrument"],
                               "rows": len(parsed[role]["rows"])} for role in parsed},
           "origin": "synthetic" if any(p["origin"] == "synthetic" for p in parsed.values())
           else "measurement (declared by the operator; unauthenticated)", "comparisons": {}}
    comparisons = out["comparisons"]
    if protocol == "MFG-COUPON-01":
        prediction = separation_prediction()
        stations = np.array(prediction["stations_mm"])
        tested = slice(1, None)  # station 0 is where the start offset itself is read
        if "photogrammetry" in parsed:
            measured, sigma = coupon_separations(parsed["photogrammetry"]["rows"])
            comparisons["separation, open loop"] = dict(_en_summary(
                [f"L{k}" for k in range(1, NOMINAL_STATIONS + 1)], measured[tested],
                np.array(prediction["separation_mm"])[tested], COVERAGE_K * sigma[tested],
                np.array(prediction["open_loop_expanded_mm"])[tested]))
        if "cmm" in parsed:
            lateral, dheading = estimate_start_pose(parsed["cmm"]["rows"])
            out["start_pose"] = {"lateral_mm": lateral, "heading_rad": dheading}
            if "photogrammetry" in parsed:
                length, steps = nominal_study()["length_mm"], NOMINAL_STATIONS * 26
                base = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=steps)
                conditioned = geo.separation_nonlinear(geo.COUPON, geo.STATION, 0.0, length, steps, lateral, dheading,
                                                       base=base)[::steps // NOMINAL_STATIONS]
                comparisons["separation, conditioned on the captured start pose"] = _en_summary(
                    [f"L{k}" for k in range(1, NOMINAL_STATIONS + 1)], measured[tested], conditioned[tested],
                    COVERAGE_K * sigma[tested], np.array(prediction["conditioned_expanded_mm"])[tested])
                # H1: the measured crossing carries the capture's own uncertainty, propagated through the
                # interpolation; the prediction carries only the prediction terms of the T140 focal row (geometry,
                # start pose, solver), not its declared instrument term, which the capture replaces.
                focal = budget_study()["budget"]["coupon focal distance (conditioned on the measured start pose)"]
                terms = focal["components"]
                crossing_m, u_cross = _crossing(stations, measured, sigma)
                crossing_p, _ = _crossing(stations, conditioned)
                if crossing_m is not None and crossing_p is not None:
                    comparisons["crossing arclength (H1)"] = _en_summary(
                        ["crossing"], np.array([crossing_m]), np.array([crossing_p]), np.array([COVERAGE_K * u_cross]),
                        np.array([COVERAGE_K * math.sqrt(terms["geometry"] ** 2 + terms["execution"] ** 2
                                                         + terms["solver"] ** 2)]))
    elif protocol == "MFG-FLAT-PLATE-01" and "photogrammetry" in parsed:
        plate = plate_study()
        pairs = pair_predictions(plate, cylinder_study())["MFG-FLAT-PLATE-01"]
        chords, sigma = _pair_chords(parsed["photogrammetry"]["rows"], [r["pair"] for r in plate["pairs"]])
        comparisons["marker-pair chord (H1)"] = _en_summary(pairs["pairs"], chords, pairs["values_mm"],
                                                           COVERAGE_K * sigma, pairs["expanded_uncertainty_mm"])
    elif protocol == "MFG-CYLINDER-01" and "photogrammetry" in parsed:
        cylinder = cylinder_study()
        predictions = pair_predictions(plate_study(), cylinder)
        chords, sigma = _pair_chords(parsed["photogrammetry"]["rows"], [r["markers"] for r in cylinder["pairs"]])
        chord = predictions["MFG-CYLINDER-01 chords"]
        comparisons["pair chord"] = _en_summary(chord["pairs"], chords, chord["values_mm"], COVERAGE_K * sigma,
                                                chord["expanded_uncertainty_mm"])
        comparisons["pair chord against the geodesic distance (H3)"] = _en_summary(
            chord["pairs"], chords, chord["geodesic_mm"], COVERAGE_K * sigma, chord["expanded_uncertainty_mm"])
        if "film" in parsed:
            film = parsed["film"]["rows"]
            gaps = np.array([film[pair][0] for pair in chord["pairs"]]) - chords
            u_gap = np.hypot([film[pair][1] for pair in chord["pairs"]], sigma)
            gap = predictions["MFG-CYLINDER-01"]
            comparisons["film minus chord (H1)"] = _en_summary(gap["pairs"], gaps, gap["values_mm"], COVERAGE_K * u_gap,
                                                              gap["expanded_uncertainty_mm"])
    return out


REALIZED_START = (2.1, 4e-4)   # relative start pose of the synthetic lateral tape (mm, rad)


@functools.lru_cache(maxsize=1)
def capture_reduction_study() -> dict:
    """The capture reader and reductions on noise-free synthetic captures, and the reader's refusals.

    Coupon: targets and start-pose points of a lateral tape started 2.1 mm and
    0.4 mrad off the nominal one; the start pose must be recovered and the
    reduced separations must equal the prediction re-integrated from it. Plate
    and cylinder: chords and film-minus-chord gaps must equal the predicted
    ones. Malformed or relabelled variants of a synthetic capture must be
    refused, and so must a retention record that offers its bytes as a
    measurement.
    """
    coupon = compare_captures({role: synthetic_capture("MFG-COUPON-01", role, *REALIZED_START)
                               for role in ("photogrammetry", "cmm")})
    conditioned = coupon["comparisons"]["separation, conditioned on the captured start pose"]
    plate = compare_captures({"photogrammetry": synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")})
    cylinder = compare_captures({role: synthetic_capture("MFG-CYLINDER-01", role) for role in ("photogrammetry", "film")})

    def worst(comparison):
        return float(np.max(np.abs(np.array(comparison["measured_mm"]) - np.array(comparison["predicted_mm"]))))

    data = synthetic_capture("MFG-FLAT-PLATE-01", "photogrammetry")
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    header = [i for i, line in enumerate(lines) if line.startswith("target,")][0]
    expected = capture_expectations("MFG-FLAT-PLATE-01")["photogrammetry"]
    variants = {
        "banner kept, origin relabelled measurement": (text.replace("# origin: synthetic", "# origin: measurement"),
                                                       "capture_header"),
        "origin synthetic without the banner": ("".join(lines[1:]), "capture_header"),
        "schema header missing": (text.replace(f"# schema: {TARGET_CAPTURE}\n", ""), "capture_header"),
        "columns out of order": (text.replace("target,x_mm,y_mm,z_mm,u_mm", "target,y_mm,x_mm,z_mm,u_mm"),
                                 "capture_columns"),
        "row repeated": (text + lines[header + 1], "capture_duplicate"),
        "target row missing": ("".join(lines[:header + 1] + lines[header + 2:]), "capture_ids"),
        "value that is not a number": (text.replace(lines[header + 1].split(",")[1], "n/a", 1), "capture_value"),
        "capture of another protocol": (text.replace("# protocol: MFG-FLAT-PLATE-01", "# protocol: MFG-COUPON-01"),
                                        "capture_expected"),
    }
    refusals = {name: {"expected": code, "observed": rec.refusal_code(rec.read_capture, variant.encode("utf-8"), expected)}
                for name, (variant, code) in variants.items()}
    fixture, _ = _schema_fixture()
    record = deepcopy(fixture)
    record.update(record_kind="measurement", instrument=dict(fixture["instrument"], serial="SN-1"),
                  raw=[{"name": "capture.csv", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "media_type": "text/csv"}])
    record["calibration"] = dict(fixture["calibration"], sha256="c" * 64)
    refusals["synthetic capture offered as a measurement record"] = {
        "expected": "fixture_is_not_measurement",
        "observed": rec.refusal_code(rec.to_acquisition, record, {"capture.csv": data})}
    return {"realized_start": {"lateral_mm": REALIZED_START[0], "heading_rad": REALIZED_START[1]},
            "estimated_start": coupon["start_pose"],
            "start_lateral_error_mm": abs(coupon["start_pose"]["lateral_mm"] - REALIZED_START[0]),
            "start_heading_error_rad": abs(coupon["start_pose"]["heading_rad"] - REALIZED_START[1]),
            "conditioned_separation_error_mm": worst(conditioned),
            "plate_chord_error_mm": worst(plate["comparisons"]["marker-pair chord (H1)"]),
            "cylinder_chord_error_mm": worst(cylinder["comparisons"]["pair chord"]),
            "cylinder_gap_error_mm": worst(cylinder["comparisons"]["film minus chord (H1)"]),
            "cylinder_signature_max_en": cylinder["comparisons"]["pair chord against the geodesic distance (H3)"][
                "max_normalized_error"],
            "refusals": refusals}


@_task("T138", ("test_predicted_separation_has_no_measured_counterpart",
                "test_comparators_compute_normalized_errors_for_a_measurement_record"))
def predicted_vs_measured(ctx):
    prediction = ctx.memo("mfg.prediction", separation_prediction)
    scenario = ctx.memo("mfg.start-scenario", start_error_scenario)
    linearity = prediction["linearity"]
    open_loop, conditioned = np.array(prediction["open_loop_expanded_mm"]), np.array(prediction["conditioned_expanded_mm"])
    predicted = {"stations_mm": prediction["stations_mm"], "separation_mm": prediction["separation_mm"],
                 "expanded_uncertainty_mm": prediction["open_loop_expanded_mm"], "offset_mm": prediction["offset_mm"]}
    scan_conditioned = np.array(prediction["scan_conditioned_expanded_mm"])
    f_pred = finding("Predicted separation of the 2 mm offset route at the MFG-COUPON-01 stations", "numerical",
                     {"separation_mm": prediction["separation_mm"], "open_loop_expanded_mm": prediction["open_loop_expanded_mm"],
                      "conditioned_expanded_mm": prediction["conditioned_expanded_mm"],
                      "scan_conditioned_expanded_mm": prediction["scan_conditioned_expanded_mm"]},
                     {"checks": [_check("self_convergence", "Richardson solver error estimate, max over stations (mm)",
                                        max(prediction["solver_mm"]), 1e-6)]
                      + [_check("self_convergence", f"sensitivity linearity max |forward / central - 1|, {name}", value, 0.1)
                         for name, value in sorted(linearity.items())]},
                     unit="mm",
                     uncertainty=_u("reference_error", float(open_loop.max()),
                                    "k = 2 open-loop expanded uncertainty: dome tolerances, start pose and solver (mm)"),
                     tolerance={"abs": 1e-6, "rel": 1e-6})
    en_bare, en_open = max(scenario["en_without_execution"]), max(scenario["en_open_loop"])
    en_cond = max(scenario["en_conditioned"])
    f_start = finding("A 2-sigma start offset of the declared jig makes a correct model fail E_n <= 1 unless the start "
                      "pose is budgeted or measured", "numerical",
                      {"realized_offset_mm": scenario["realized_offset_mm"], "max_en_without_execution": en_bare,
                       "max_en_open_loop": en_open, "max_en_conditioned": en_cond},
                      {"generator": _generator("exactly re-integrated realized tape (model output as ideal data)",
                                               realized_offset_mm=scenario["realized_offset_mm"]),
                       "checks": [_check("analytic", "max E_n at stations 1..8, U_p without the start-pose term", en_bare, 1.0, "ge"),
                                  _check("analytic", "max E_n at stations 1..8, U_p with the declared start-pose term", en_open, 1.0, "le"),
                                  _check("analytic", "max E_n at stations 1..8, prediction re-integrated from a start pose "
                                                     "estimated with 2-sigma CMM error, conditioned U_p", en_cond, 1.0, "le")]},
                      uncertainty=_u("roundoff", 1e-9, "deterministic re-integration; E_n is a ratio of computed values"),
                      tolerance={"abs": 1e-6, "rel": 1e-6},
                      counterexample={"statement": "A physically correct model passes E_n <= 1 against its open-loop prediction "
                                                   "when U covers only the instrument and the dome tolerances",
                                      "witness": {"realized_offset_mm": scenario["realized_offset_mm"],
                                                  "declared_jig_sigma_mm": EXECUTION["lateral_mm"],
                                                  "max_en_without_execution": en_bare}})
    fixture, fixture_raw = _schema_fixture()
    tampered = deepcopy(fixture)
    tampered["record_kind"] = "measurement"
    codes = {"absent": rec.refusal_code(rec.compare_separation, predicted, None),
             "fixture": rec.refusal_code(rec.compare_separation, predicted, {"record": fixture, "raw_bytes": fixture_raw,
                                                                              "values_mm": predicted["separation_mm"],
                                                                              "expanded_uncertainty_mm": [0.06] * 9}),
             "relabelled": rec.refusal_code(rec.compare_separation, predicted, {"record": tampered, "raw_bytes": fixture_raw,
                                                                                 "values_mm": predicted["separation_mm"],
                                                                                 "expanded_uncertainty_mm": [0.06] * 9}),
             "digest": rec.refusal_code(rec.compare_separation, predicted, {"record": tampered,
                                                                             "raw_bytes": {"fixture.txt": b"altered"},
                                                                             "values_mm": predicted["separation_mm"],
                                                                             "expanded_uncertainty_mm": [0.06] * 9})}
    # The pair comparator serves the plate (chord = geodesic per marker pair) and the cylinder (chord-geodesic
    # gap per pair); it refuses in the same way until a measurement record exists.
    pairs = pair_predictions(ctx.memo("mfg.plate", plate_study), ctx.memo("mfg.cylinder", cylinder_study))
    plate_fixture = {"record": fixture, "raw_bytes": fixture_raw, "labels": pairs["MFG-FLAT-PLATE-01"]["pairs"],
                     "values_mm": pairs["MFG-FLAT-PLATE-01"]["values_mm"],
                     "expanded_uncertainty_mm": COVERAGE_K * PAIR_U}
    codes.update({"plate": rec.refusal_code(rec.compare_pair_distances, pairs["MFG-FLAT-PLATE-01"], None),
                  "cylinder": rec.refusal_code(rec.compare_pair_distances, pairs["MFG-CYLINDER-01"], None),
                  "plate fixture": rec.refusal_code(rec.compare_pair_distances, pairs["MFG-FLAT-PLATE-01"], plate_fixture)})
    refusals = [_refusal("no measured separation exists", "measurement_absent", codes["absent"]),
                _refusal("schema fixture offered as a measurement", "fixture_is_not_measurement", codes["fixture"]),
                _refusal("schema fixture relabelled as a measurement", "fixture_is_not_measurement", codes["relabelled"]),
                _refusal("measurement record whose raw bytes do not match", "raw_digest_mismatch", codes["digest"]),
                _refusal("no measured plate marker-pair distance exists (pair comparator)", "measurement_absent",
                         codes["plate"]),
                _refusal("no measured cylinder chord-geodesic gap exists (pair comparator)", "measurement_absent",
                         codes["cylinder"]),
                _refusal("schema fixture offered as plate marker-pair measurements", "fixture_is_not_measurement",
                         codes["plate fixture"])]
    f_refuse = finding("The comparison refuses to run without acquired hardware evidence", "computational_pipeline",
                       sum(check["passed"] for check in refusals), {"checks": refusals},
                       unit="refusals", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    f_measured = finding("Measured separation on the coupon agrees with the prediction (E_n <= 1 at every station)",
                         "physical", None, {})
    reduction = ctx.memo("mfg.capture-reduction", capture_reduction_study)
    reader_refusals = [_refusal(f"capture variant: {name}", case["expected"], case["observed"])
                       for name, case in sorted(reduction["refusals"].items())]
    worst_chord = max(reduction["plate_chord_error_mm"], reduction["cylinder_chord_error_mm"],
                      reduction["cylinder_gap_error_mm"])
    f_reduce = finding("The capture reader and its reductions recover the separations, start pose, chords and gaps of "
                       "noise-free synthetic captures and refuse malformed or relabelled ones", "computational_pipeline",
                       {key: reduction[key] for key in ("estimated_start", "start_lateral_error_mm",
                                                        "start_heading_error_rad", "conditioned_separation_error_mm",
                                                        "plate_chord_error_mm", "cylinder_chord_error_mm",
                                                        "cylinder_gap_error_mm", "cylinder_signature_max_en")},
                       {"generator": _generator("noise-free synthetic captures (manufacturing.synthetic_capture) of "
                                                "MFG-COUPON-01, MFG-FLAT-PLATE-01 and MFG-CYLINDER-01", seed=None,
                                                realized_offset_mm=REALIZED_START[0],
                                                realized_heading_rad=REALIZED_START[1]),
                        "checks": [_check("analytic", "|estimated - realized| start offset of the lateral tape (mm)",
                                          reduction["start_lateral_error_mm"], 1e-6, "le"),
                                   _check("analytic", "|estimated - realized| start heading of the lateral tape (rad; a "
                                          "twentieth of the CMM heading uncertainty)", reduction["start_heading_error_rad"],
                                          1e-5, "le"),
                                   _check("analytic", "max over stations 1..8 of |separation reduced from the targets - "
                                          "prediction re-integrated from the estimated start pose| (mm)",
                                          reduction["conditioned_separation_error_mm"], 1e-3, "le"),
                                   _check("analytic", "max |captured - predicted| over the plate chords, cylinder chords "
                                          "and cylinder film-minus-chord gaps (mm)", worst_chord, 1e-9, "le"),
                                   _check("analytic", "largest E_n of the cylinder chords against the geodesic distances "
                                          "(the H3 signature is resolved)", reduction["cylinder_signature_max_en"], 1.0,
                                          "ge"),
                                   *reader_refusals]},
                       unit="mm", uncertainty=_u("truncation_bound", reduction["conditioned_separation_error_mm"],
                                                 "second-order remainder of the separation reduction and start-pose "
                                                 "estimate on the synthetic captures (mm)"),
                       tolerance={"abs": 1e-6, "rel": 1e-6})
    # Operator captures bound to this run (ciw lab run T138 --capture ROLE=PATH): retained, parsed and compared
    # computationally. They are unauthenticated and no metrology instrument probe exists, so the physical claims
    # above stay not_established whatever the comparison shows.
    bound = [role for role in CAPTURE_ROLES if ctx.available(f"capture:{role}")]
    raw = {role: ctx.capture(role) for role in bound}
    capture_claim = ("Normalized errors of the bound metrology captures against the prediction of their protocol, "
                     "computed from their unauthenticated bytes")
    compared = None
    if raw:
        try:
            compared, capture_code = compare_captures(raw), None
        except rec.RecordRefusal as exc:
            capture_code = exc.code
        comparisons = compared["comparisons"] if compared else {}
        inputs = {"operator_captures": {role: hashlib.sha256(data).hexdigest() for role, data in raw.items()},
                  "authenticated": False}
        if compared and not comparisons:
            # Roles the protocol defines only for retention (a start-pose file without the targets it would
            # condition) are read and retained; nothing is compared, so the claim is honestly unestablished.
            f_capture = finding(capture_claim, "computational_pipeline", compared,
                                {"notes": f"the bound roles ({', '.join(sorted(raw))}) of {compared['protocol']} define "
                                          "no comparison: they were parsed and retained; a comparison needs the "
                                          "photogrammetry capture", "inputs": inputs},
                                expected_not_established=True)
        else:
            en = [value for c in comparisons.values() for value in c["normalized_error"]]
            basis = {"checks": [_refusal("the bound captures parse in the protocol's raw formats and roles", "none",
                                         capture_code),
                                _check("exact_arithmetic", "comparisons computed from the bound captures (at least "
                                       "one)", len(comparisons), 1.0, "ge"),
                                _check("exact_arithmetic", "non-finite normalized errors",
                                       float(sum(not math.isfinite(v) for v in en)), 0.0)],
                     "inputs": inputs}
            if compared and compared["origin"] == "synthetic":
                basis["generator"] = {"name": "synthetic operator capture (declared in its header)", "seed": None}
            f_capture = finding(capture_claim, "computational_pipeline", compared or {"refusal": capture_code}, basis,
                                uncertainty=_u("roundoff", 1e-12, "double-precision arithmetic on the parsed values; "
                                                                  "the declared measurement uncertainty is inside each "
                                                                  "E_n"),
                                tolerance={"abs": 1e-9, "rel": 1e-9})
    else:
        f_capture = finding(capture_claim, "computational_pipeline", None,
                            {"notes": "no operator capture was bound (roles " + ", ".join(CAPTURE_ROLES) + ")"},
                            expected_not_established=True)
    ctx.artifact_json("predicted-separation.json", _r(dict(prediction, protocol="MFG-COUPON-01", start_error_scenario=scenario,
                                                           execution_declared=EXECUTION, start_pose_uncertainty=START_POSE_U,
                                                           measured="none: no hardware record exists")))
    ctx.artifact_json("pair-predictions.json", _r(dict(pairs, measured="none: no hardware record exists")))
    ctx.artifact_json("capture-reduction.json", _r(reduction))
    if compared:
        ctx.artifact_json("capture-comparison.json", _r(compared))
    if compared:
        summary = "; ".join(f"{name}: max E_n {c['max_normalized_error']:.2f} over {len(c['labels'])}"
                            for name, c in compared["comparisons"].items())
        capture_text = (f" Bound captures ({', '.join(sorted(raw))}, {compared['protocol']}, origin "
                        f"{compared['origin']}): {summary or 'no comparison for these roles'}.")
    elif raw:
        capture_text = f" Bound captures ({', '.join(sorted(raw))}) were refused by the reader ({capture_code})."
    else:
        capture_text = " No operator capture was bound."
    fields = _fields(
        "The predicted separation of the 2 mm offset tape (and its crossing near s = 154 mm) can be compared with "
        "measurement by the normalized error E_n, provided the prediction uncertainty includes the realized start pose "
        "of the tape (budgeted open loop, or removed by conditioning on its measured start pose); the comparison is only "
        "meaningful against acquired hardware evidence, and bytes an operator binds are compared computationally "
        "without becoming that evidence.",
        "E_n = |m - p| / sqrt(U_m^2 + U_p^2), U = 2 u. Open loop: u_p^2 = u_geometry^2 + u_solver^2 + u_start^2 with "
        "u_start from the declared start-pose error (insert and laying 0.05 mm, 0.5 mrad; two jig seatings of 0.01 mm, "
        "0.1 mrad each; in quadrature) times central-difference start sensitivities of the "
        "exactly re-integrated offset route. Conditioned: p is re-integrated from the CMM start pose and u_start uses "
        "its estimate uncertainty (offset sqrt(2) u_cmm, heading 2 u_cmm / 20 mm). Scan-conditioned: u_geometry^2 = "
        "g^T C g with g the dome sensitivities (height, width, centre) and C the MFG-SCAN-01 covariance (T129) instead of "
        "the declared tolerances. Pair comparator: E_n per plate marker pair (chord = geodesic) or cylinder pair "
        "(chord-geodesic gap, U_p = 2 |d gap / dR| u_R). Captures: a separation is the difference of two station "
        "targets projected on the model's in-surface normal of the nominal route; the start pose solves "
        "e(0) = delta, e(20) = delta j_lat(20) + dtheta j_head(20).",
        ["Nominal route and offset stations from T128 (k L / 8)", "Declared dome tolerances height +/- 0.2 mm, sigma +/- 0.5 mm",
         "Scan-derived dome covariance of MFG-SCAN-01 (T129, synthetic verification; declared scanner terms)",
         "Declared relative start-pose error of the offset tape (1 sigma): insert and laying 0.05 mm and 0.5 mrad, two "
         "jig seatings of 0.01 mm and 0.1 mrad each; CMM 0.002 mm",
         "Plate and cylinder marker-pair predictions (T126, T127); declared radius tolerance +/- 0.1 mm",
         "Noise-free synthetic captures of the three control protocols (lateral tape started 2.1 mm and 0.4 mrad off)",
         "Operator captures bound with --capture photogrammetry=, cmm= or film= (none in the retained clean-room run)"],
        "No observation: the measurement slot of MFG-COUPON-01 is empty. The planned observation is photogrammetry of "
        "coded targets on the tape centrelines at the nine stations plus CMM probing of the start pose, exported in the "
        "raw formats the protocols define; a bound export is read, retained and compared, but it is not authenticated "
        "and no metrology instrument probe exists, so it is never hardware evidence here.",
        "The comparators refuse absent measurements, schema fixtures (also when relabelled) and digest mismatches; "
        "start-pose and geometry sensitivities are linear over their declared steps; the capture reader recovers "
        "synthetic captures exactly and refuses malformed or relabelled ones.",
        "Compute the prediction and its uncertainty components (open loop, conditioned on the start pose, and also on "
        f"the as-built scan); evaluate a correct model against a tape realized {scenario['realized_offset_mm'] - 2.0:.3f} "
        "mm (2 sigma of the declared open-loop start error) off its nominal offset with and "
        "without the start-pose term, and against the prediction re-integrated from a CMM start-pose estimate at the "
        "four 2-sigma corners; attempt the separation and pair comparisons with no measurement, with a schema fixture "
        "(as is and relabelled) and with a tampered record, and record the refusals; run the capture reader and "
        "reductions on noise-free synthetic captures and malformed variants; read, retain and compare any operator "
        "capture bound to the run.",
        "Predicted separation (mm) at stations: " + ", ".join(f"{v:.3f}" for v in prediction["separation_mm"])
        + f"; k = 2 uncertainty up to {open_loop.max():.3f} mm open loop, {conditioned.max():.3f} mm conditioned on the "
        f"measured start pose and {scan_conditioned.max():.3f} mm conditioned also on the as-built scan; a "
        f"{scenario['realized_offset_mm'] - 2.0:.3f} mm start error gives max E_n {en_bare:.2f} without the start-pose term, "
        f"{en_open:.2f} with it (open loop) and {en_cond:.2f} when the prediction is re-integrated from a CMM start-pose "
        f"estimate 2 sigma off in offset and heading (conditioned U_p); synthetic captures: start pose recovered to "
        f"{reduction['start_lateral_error_mm']:.1e} mm and {reduction['start_heading_error_rad']:.1e} rad, separations "
        f"to {reduction['conditioned_separation_error_mm']:.1e} mm. No hardware-evidenced measurement exists."
        + capture_text,
        "Prediction uncertainty only (under the declared tolerances geometry dominates beyond mid-route; start pose "
        "dominates near the start in the open-loop case); the scan-conditioned uncertainty holds only if the as-built "
        "coupon passes the MFG-SCAN-01 model test; measurement uncertainty is declared until the protocol is executed, "
        "and a capture's u column is the operator's declaration.",
        ["absent measurement (refused)", "schema fixture as measurement, also relabelled (refused)",
         "raw digest mismatch (refused)", "plate and cylinder pair comparisons without a measurement (refused)",
         "start-pose error outside the prediction uncertainty", "linearity of geometry and start-pose sensitivities",
         "malformed, relabelled or foreign-protocol captures (refused)",
         "synthetic capture offered as a measurement record (refused)",
         "chord-angle start-heading estimate biased by delta (j_lat(20) - 1) / 20 (replaced by the Jacobi-field solve)"],
        ["The physical comparison has not been performed; its outcome is unknown.",
         "The tapes are assumed to follow geodesics after their measured start; in-plane tape bending is not budgeted.",
         "Operator captures are unauthenticated: their origin header and u column are declarations, and no metrology "
         "instrument probe exists on any analysing host. Deferred research question: a signed-capture trust anchor "
         "(an instrument-held key that signs each export, verified by the workbench) or a hardware:metrology probe of "
         "an instrument attached to the analysing host; until one exists, a bound capture yields computational "
         "comparisons only and the physical claims stay not_established.",
         "The plate and cylinder cmm captures (tape start poses) are read and retained but do not yet condition those "
         "predictions."],
        "Deferred research question: define a signed-capture trust anchor (instrument-held signing keys and their "
        "verification) or a hardware:metrology probe, without which no capture can support a physical label. Meanwhile "
        "acquire MFG-COUPON-01 and bind its exports (ciw lab run T138 --capture photogrammetry=<targets.csv> --capture "
        "cmm=<start-pose.csv>), then retain the run with ciw lab hardware retain.")
    return {"state": "partial", "fields": fields, "findings": [f_pred, f_start, f_refuse, f_reduce, f_capture, f_measured,
                                                              _not_measured(EXECUTION_CLAIM, "calibration")]}


# T139 retention of raw measurements -------------------------------------------------------------
# Eigenvalues (mm^2) of the scale test: rank 3 at a 1e6 mm^2 scale, with an explicit -1e-8 mm^2
# eigenvalue. Relative to the largest that is 1e-14, a few times the eigensolver's rounding
# level (6 eps max|lambda| ~ 1.3e-9 mm^2), yet four orders above an absolute 1e-12 threshold.
SCALE_TEST_EIGENVALUES = (1e6, 2e5, 5e4, 0.0, 0.0, -1e-8)
ABSOLUTE_THRESHOLD = 1e-12


def _rank_deficient_covariance(negative=SCALE_TEST_EIGENVALUES[-1]):
    """A 6 x 6 rank-3 covariance at a 1e6 mm^2 scale with one small negative eigenvalue ``negative``."""
    basis, _ = np.linalg.qr(met.rng(SEED + 7).normal(0.0, 1.0, (6, 6)))
    return basis @ np.diag([*SCALE_TEST_EIGENVALUES[:-1], negative]) @ basis.T


# A retention record bound to T139 (ciw lab run T139 --capture retention=<record.json>) describes raw files that
# are bound beside it under the protocols' capture roles; its raw entries are matched to those bytes by digest.
RETENTION_ROLE = "retention"
RETAINED_RAW_ROLES = ("photogrammetry", "cmm", "film", "scanner")
PROTOCOL_IDS = ("MFG-FLAT-PLATE-01", "MFG-CYLINDER-01", "MFG-COUPON-01", "MFG-SCAN-01")


def retain_measurement(record_bytes: bytes, raw: dict) -> dict:
    """Validate a bound retention record against the raw bytes bound with it and build its acquisition fields.

    ``raw`` maps capture roles to bytes. Each raw entry of the record is
    matched to the bound bytes with its SHA-256; an entry without matching
    bytes is refused (raw_digest_mismatch), and so are a schema fixture, the
    fixture's markers and synthetic-capture bytes (to_acquisition). The
    acquisition fields are computational: nothing binds the bytes to an
    instrument here, so they never support a physical label.
    """
    try:
        record = json.loads(bytes(record_bytes).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise rec.RecordRefusal("retention_not_json", "A bound retention record must be a UTF-8 JSON object") from None
    if not isinstance(record, dict):
        raise rec.RecordRefusal("retention_not_json", "A bound retention record must be a UTF-8 JSON object")
    rec.validate_retention(record)
    if record["protocol_id"] not in PROTOCOL_IDS:
        raise rec.RecordRefusal("retention_protocol_unknown", f"No protocol {record['protocol_id']!r} in this section")
    digests = {hashlib.sha256(data).hexdigest(): (role, data) for role, data in sorted(raw.items())}
    matched = {entry["name"]: digests[entry["sha256"]] for entry in record["raw"] if entry["sha256"] in digests}
    acquisition = rec.to_acquisition(record, {name: data for name, (_, data) in matched.items()})
    return {"protocol_id": record["protocol_id"], "record_kind": record["record_kind"],
            "retention_identity": rec.retention_identity(record), "acquisition": acquisition,
            "raw_roles": {name: role for name, (role, _) in sorted(matched.items())},
            "manifest": rec.raw_manifest(record).decode("utf-8")}


@_task("T139", ("test_retention_schema_refusals_and_fixture_boundary",
                "test_bound_retention_record_is_validated_against_its_raw_bytes"))
def retention_schema(ctx):
    fixture, raw = _schema_fixture()
    rec.validate_retention(fixture, raw)

    def mutate(change):
        copy = deepcopy(fixture)
        change(copy)
        return copy

    def break_chain(record):
        record["frame_chain"][1]["parent"] = "SOMEWHERE"

    def bad_rotation(record):
        record["frame_chain"][0]["rotation"] = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def bad_covariance(record):
        record["frame_chain"][0]["covariance"][0][0] = -1.0

    cases = {
        "missing raw list": (lambda r: r.pop("raw"), "missing_field", None),
        "raw digest mismatch": (lambda r: None, "raw_digest_mismatch", {"fixture.txt": b"different bytes"}),
        "calibration without reference": (lambda r: r["calibration"].pop("reference"), "missing_field", None),
        "calibration digest not SHA-256 hex": (lambda r: r["calibration"].update(sha256="CERT-DIGEST"),
                                               "calibration_digest_invalid", None),
        "acquired after the calibration expired": (lambda r: r["clock"].update(acquired_at="2031-01-01T00:00:00Z"),
                                                   "calibration_expired", None),
        "frame chain broken": (break_chain, "frame_chain_broken", None),
        "non-orthonormal rotation": (bad_rotation, "frame_rotation_invalid", None),
        "covariance not PSD": (bad_covariance, "frame_covariance_invalid", None),
        "clock without timezone": (lambda r: r["clock"].update(acquired_at="2026-09-23T00:00:00"), "clock_without_timezone", None),
        "missing clock": (lambda r: r.pop("clock"), "missing_field", None),
        "missing instrument serial": (lambda r: r["instrument"].pop("serial"), "missing_field", None),
    }
    checks = [_refusal(f"retention mutation: {name}", expected, rec.refusal_code(rec.validate_retention, mutate(change), data))
              for name, (change, expected, data) in cases.items()]
    matched = sum(c["passed"] for c in checks)
    f_schema = finding("The retention schema refuses records missing raw digests, calibration, frame-chain or clock metadata",
                       "computational_pipeline", matched, {"checks": checks}, unit="refused mutations",
                       uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    covariance = _rank_deficient_covariance()
    lever = mutate(lambda r: r["frame_chain"][0].update(covariance=covariance.tolist()))
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    # What an absolute rule would see: the asymmetry and the negative eigenvalue against 1e-12 mm^2.
    residue = max(float(np.max(np.abs(covariance - covariance.T))), -float(eigenvalues[0]))
    invalid = mutate(lambda r: r["frame_chain"][0].update(
        covariance=_rank_deficient_covariance(-1e-3 * SCALE_TEST_EIGENVALUES[0]).tolist()))
    f_scale = finding("The retention schema keeps a rank-deficient frame covariance whose rounding-level negative "
                      "eigenvalue an absolute threshold would refuse, and still refuses a negative one at the same scale",
                      "computational_pipeline", {"min_eigenvalue_mm2": float(eigenvalues[0]),
                                                 "max_eigenvalue_mm2": float(eigenvalues[-1])},
                      {"checks": [_refusal("rank-3 covariance at a 1e6 mm^2 scale with a -1e-8 mm^2 eigenvalue (kept by the "
                                           "scale-aware rule)", "none", rec.refusal_code(rec.validate_retention, lever, raw)),
                                  _check("analytic", "rounding residue (max of asymmetry and -min eigenvalue) over an absolute "
                                         "1e-12 mm^2 threshold: an absolute rule would refuse this covariance",
                                         residue / ABSOLUTE_THRESHOLD, 1.0, "ge"),
                                  _check("analytic", "negative eigenvalue relative to the largest (rounding level)",
                                         abs(min(float(eigenvalues[0]), 0.0)) / float(eigenvalues[-1]), 1e-12, "le"),
                                  _refusal("same scale with an eigenvalue of -1e-3 times the largest", "frame_covariance_invalid",
                                           rec.refusal_code(rec.validate_retention, invalid, raw))]},
                      unit="mm^2", uncertainty=_u("roundoff", 6.0 * float(np.finfo(float).eps) * float(eigenvalues[-1]),
                                                  "eigensolver rounding, 6 eps max|lambda| (mm^2)"),
                      tolerance={"abs": 5e-9, "rel": 1e-6})
    reordered = {key: fixture[key] for key in reversed(list(fixture))}
    altered = deepcopy(fixture)
    altered["raw"][0]["sha256"] = "f" * 64
    two_files = deepcopy(fixture)
    two_files["raw"].append({"name": "second.bin", "sha256": "1" * 64, "bytes": 10, "media_type": "application/octet-stream"})
    two_altered = deepcopy(two_files)
    two_altered["raw"][1]["sha256"] = "2" * 64
    identity_checks = [_check("exact_arithmetic", "identity changes under key reordering",
                              0.0 if rec.retention_identity(reordered) == rec.retention_identity(fixture) else 1.0, 0.0),
                       _check("exact_arithmetic", "identity unchanged after altering a raw digest",
                              1.0 if rec.retention_identity(altered) == rec.retention_identity(fixture) else 0.0, 0.0),
                       _check("exact_arithmetic", "raw manifest (acquisition digest) unchanged after altering the second "
                              "raw file's digest", 1.0 if rec.raw_manifest(two_altered) == rec.raw_manifest(two_files) else 0.0, 0.0)]
    f_identity = finding("Retention record identity is independent of key order and bound to every raw digest", "provenance",
                         rec.retention_identity(fixture), {"checks": identity_checks}, uncertainty=EXACT,
                         tolerance={"abs": 0, "rel": 0})
    relabelled = dict(fixture, record_kind="measurement")
    boundary = [_refusal("fixture promoted to acquisition", "fixture_is_not_measurement", rec.refusal_code(rec.to_acquisition, fixture, raw)),
                _refusal("fixture relabelled as a measurement, with its bytes", "fixture_is_not_measurement",
                         rec.refusal_code(rec.to_acquisition, relabelled, raw)),
                _refusal("measurement record without raw bytes", "raw_bytes_not_presented",
                         rec.refusal_code(rec.to_acquisition, relabelled, None))]
    f_boundary = finding("A schema fixture (as is or relabelled as a measurement), or a record without its raw bytes, is "
                         "refused as hardware evidence", "computational_pipeline", sum(c["passed"] for c in boundary),
                         {"checks": boundary}, unit="refusals", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    schema = {"schema": rec.RETENTION_SCHEMA, "required": list(rec.RETENTION_FIELDS),
              "raw": ["name", "sha256", "bytes", "media_type"], "instrument": ["id", "kind", "serial"],
              "calibration": {"status": ["applied", "not_applied"],
                              "applied_requires": ["reference", "sha256 (64 hex)", "valid_from", "valid_until"],
                              "window": "clock.acquired_at within [valid_from, valid_until]"},
              "frame_chain": {"link": ["parent", "child", "rotation", "translation_mm", "covariance", "source"],
                              "continuity": "child of link k is the parent of link k + 1",
                              "rotation": "proper orthonormal 3x3",
                              "covariance": "symmetric PSD 6x6 over (rho, phi), tolerances relative to its scale"},
              "clock": ["source", "acquired_at (ISO 8601 with Z or +hh:mm)", "synchronization", "uncertainty_s"],
              "to_acquisition": "measurement records with matching raw bytes only; schema-fixture bytes, FIXTURE- "
                                "serials and all-zero calibration digests are refused; raw_sha256 = SHA-256 of the "
                                "raw manifest, which the citing task must retain",
              "protocol_hardware_slot": "acquisition with a 64-hex raw digest, an ISO 8601 time and the retention identity"}
    ctx.artifact_json("retention-schema.json", schema)
    # The fixture record and its digests are retained; its raw bytes are not, so no retained
    # artifact carries a digest that a hardware claim could cite.
    ctx.artifact_json("retention-schema-fixture.json", fixture)
    # A retention record bound to this run is validated against the raw bytes bound beside it and retained with
    # them. The bytes are unauthenticated and no metrology probe exists, so the physical claim stays open.
    bound_claim = ("A bound retention record validates against the raw bytes bound with it and yields acquisition "
                   "fields (computational: the bytes are unauthenticated)")
    retained = None
    if ctx.available(f"capture:{RETENTION_ROLE}"):
        record_bytes = ctx.capture(RETENTION_ROLE)
        raw_bound = {role: ctx.capture(role) for role in RETAINED_RAW_ROLES if ctx.available(f"capture:{role}")}
        try:
            retained, retention_code = retain_measurement(record_bytes, raw_bound), None
        except rec.RecordRefusal as exc:
            retention_code = exc.code
        inputs = {"operator_captures": {role: hashlib.sha256(data).hexdigest()
                                        for role, data in {RETENTION_ROLE: record_bytes, **raw_bound}.items()},
                  "authenticated": False}
        f_bound = finding(bound_claim, "computational_pipeline",
                          {key: retained[key] for key in ("protocol_id", "record_kind", "retention_identity",
                                                          "acquisition", "raw_roles")} if retained
                          else {"refusal": retention_code},
                          {"checks": [_refusal("the bound retention record validates, its raw entries match the bound "
                                               "bytes by digest and it is a measurement record without fixture or "
                                               "synthetic markers", "none", retention_code)],
                           "inputs": inputs},
                          uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
        if retained:
            ctx.artifact_text("retention-raw-manifest.json", retained["manifest"])
            ctx.artifact_json("retention-acquisition.json", {k: retained[k] for k in (
                "protocol_id", "retention_identity", "acquisition", "raw_roles")})
    else:
        f_bound = finding(bound_claim, "computational_pipeline", None,
                          {"notes": f"no retention record was bound (role {RETENTION_ROLE}); bind one with its raw files "
                                    "under the roles " + ", ".join(RETAINED_RAW_ROLES)},
                          expected_not_established=True)
    count = 1 if retained else 0
    findings = [f_schema, f_scale, f_identity, f_boundary, f_bound,
                finding("A real measurement with raw bytes, calibration and frame metadata has been retained", "physical",
                        count, {}, unit="records",
                        uncertainty={"kind": "roundoff", "value": 0, "basis": "count of retained measurement records "
                                                                              "(declared by the operator, unauthenticated)"})]
    if retained:
        retention_text = (f"Retained the bound {retained['record_kind']} record of {retained['protocol_id']} "
                          f"({retained['retention_identity'][:19]}...) with its raw files "
                          f"({', '.join(f'{name} as {role}' for name, role in retained['raw_roles'].items())}), matched "
                          "by digest; its acquisition fields were built. The bytes are the operator's and unauthenticated.")
    elif ctx.available(f"capture:{RETENTION_ROLE}"):
        retention_text = f"The bound retention record was refused ({retention_code}); nothing was retained as a measurement."
    else:
        retention_text = "No retention record was bound, so no raw measurement was retained."
    fields = _fields(
        "A retention record that binds raw-byte digests, calibration reference and validity window, frame chain with "
        "covariances and an explicit clock is sufficient to supply the acquisition fields of a hardware_measured "
        "finding, and every omission is refused; a record bound to the run with its raw files is validated against "
        "those bytes and retained with them.",
        "Record = {raw digests, instrument identity, calibration (applied/not_applied), frame chain links (R, t, C), clock}; "
        "identity = SHA-256 of canonical JSON; to_acquisition maps a valid measurement record to device / raw_sha256 "
        "(SHA-256 of the raw manifest) / acquired_at / calibration. A bound record's raw entries are matched to the "
        "bytes bound under the capture roles " + ", ".join(RETAINED_RAW_ROLES) + " by SHA-256.",
        ["Schema fixture with explicit 'SCHEMA FIXTURE - NOT A MEASUREMENT' raw bytes (not retained)",
         f"{len(cases)} refused mutations, one rank-deficient covariance at a 1e6 mm^2 scale with a -1e-8 mm^2 "
         "eigenvalue and one with a negative eigenvalue of -1e-3 of its largest",
         f"Operator captures: a retention record (--capture {RETENTION_ROLE}=) and its raw files (--capture "
         "photogrammetry=, cmm=, film= or scanner=); none in the retained clean-room run"],
        "No instrument is observed here. The validators read the declared schema fixture and its mutations and, when "
        "bound, the operator's retention record and raw files, which are retained and not authenticated.",
        "Valid records validate; each mutation is refused with its code; covariance tolerances scale with the matrix; "
        "a record of kind schema_fixture, or one carrying the fixture's bytes, serial or zero calibration digest, or "
        "synthetic-capture bytes, is refused as hardware evidence; a bound record's raw entries match the bound bytes.",
        ("Completed: " if retained else "Partial: ")
        + f"the retention mechanism was exercised on a synthetic schema fixture (validate it, mutate it {len(cases)} "
        "ways, keep a rank-deficient covariance that an absolute threshold would refuse and refuse a negative one at "
        "the same scale, check identity invariance and the fixture/measurement boundary). " + retention_text
        + ("" if retained else " Not performed: retaining raw measurements, calibration and frame metadata of a real "
           "acquisition, because none was bound; the missing one is the first executed control protocol "
           "(MFG-FLAT-PLATE-01, MFG-CYLINDER-01 or MFG-COUPON-01) with its instrument exports and its retention record "
           "(instrument serials, calibration certificates, frame chain and clock)."),
        f"{matched}/{len(cases)} mutations refused with the expected code; rank-deficient covariance kept (min eigenvalue "
        f"{eigenvalues[0]:.1e} mm^2 against a largest of {eigenvalues[-1]:.1e} mm^2, {residue / ABSOLUTE_THRESHOLD:.0e} "
        f"times an absolute 1e-12 mm^2 threshold); retained measurement records: {count}.",
        "Exact: the outcomes are booleans and refusal codes of deterministic validators; the only floating-point "
        "quantity is the rounding-level eigenvalue of the rank-deficient covariance.",
        ["digest mismatch", "calibration digest and validity window", "broken frame chain",
         "invalid rotation or covariance",
         "scale of covariance tolerances (kept above an absolute threshold, refused when negative at scale)",
         "clock without timezone", "fixture promoted or relabelled as a measurement", "measurement without raw bytes",
         "bound record whose raw entries match no bound bytes, or that lists synthetic-capture bytes (refused)"],
        [("A measurement record was bound and retained with its raw files; its origin is the operator's declaration."
          if retained else
          "No acquisition was bound, so no raw measurement, calibration record or frame metadata has been retained: the "
          "task stays partial until a real acquisition of a control protocol is retained with its record. Synthetic or "
          "absent material never completes a retention task."),
         "Media-type-specific readers (images, point clouds) are not defined; digests cover bytes, not content semantics.",
         "The validators cannot tell whether raw bytes came from an instrument; the runner also requires a hardware "
         "probe in the task that cites them, and no metrology instrument probe or signed-capture trust anchor exists "
         "(deferred research question, T138), so a retained record's acquisition fields support no physical label."],
        "Deferred research question: a signed-capture trust anchor (instrument-held keys that sign each export) or a "
        "hardware:metrology probe, without which a retained record supports no physical label. Meanwhile retain the "
        "first real acquisition of a control protocol: run ciw lab run T139 --capture retention=<record.json> with its "
        "raw files bound under their roles (--capture photogrammetry=<targets.csv> --capture cmm=<start-pose.csv> ...), "
        "the record listing each file with its SHA-256, instrument serial, calibration certificate and validity window, "
        "frame chain with covariances and clock; then retain the run with ciw lab hardware retain.")
    return {"state": "completed" if retained else "partial", "fields": fields, "findings": findings}


# T140 uncertainty budget -----------------------------------------------------------------------
def _classify(components):
    total = sum(v ** 2 for v in components.values())
    dominant = max(components, key=components.get)
    share = components[dominant] ** 2 / total if total else 0.0
    return (dominant if share > 0.5 else f"mixed (largest: {dominant})"), share


PLATE_FLATNESS = {"deviation_mm": 0.05, "sigma_mm": 75.0,
                  "model": "flatness deviation as a Gaussian bump of height u = 0.05 / sqrt(3) mm and width 75 mm, "
                           "centred under the route"}


def _start_term(u, j_lat, j_head):
    return math.hypot(u["lateral_mm"] * j_lat, u["heading_rad"] * j_head)


@functools.lru_cache(maxsize=1)
def budget_study() -> dict:
    nominal = nominal_study()
    sensitivity = sensitivity_study()
    prediction = separation_prediction()
    cylinder = cylinder_study()
    plate = plate_study()
    u_pair = PAIR_U
    length = nominal["length_mm"]
    j_lat_end, j_head_end = nominal["j_lat"][-1], nominal["j_head_mm"][-1]
    budget = {}
    # Q1: cylinder chord-geodesic gap at 90 degrees (fixed markers: no path, no start pose), measured as the
    # film surface distance minus the camera chord.
    ninety = next(r for r in cylinder["pairs"] if r["pair"] == "circumferential 90 deg")
    u_radius = 0.1 / math.sqrt(3.0)
    budget["cylinder gap, 90 deg pair"] = {"value": ninety["gap_mm"], "unit": "mm",
                                           "components": {"instrument": GAP_U,
                                                          "geometry": abs(math.pi / 2 - 2 * math.sin(math.pi / 4)) * u_radius,
                                                          "execution": 0.0, "solver": ninety["rk4_closure_mm"]}}
    # Q2: coupon separation at the route end for a 5 mrad heading offset. Solver: the reported value is
    # the h-solution, whose RK4 error is |Q(h) - Q(2h)| / 15.
    coarse = jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=NOMINAL_STATIONS * 13)
    heading_end = nominal["heading_linear_mm"][-1]
    solver_q2 = abs(heading_end - 0.005 * coarse.states[-1, 6]) / 15.0
    geometry_q2 = math.hypot(sensitivity["height_mm"]["heading_end"]["central"], sensitivity["sigma_mm"]["heading_end"]["central"])
    for label, u in (("open-loop start", EXECUTION), ("conditioned on the measured start pose", START_POSE_U)):
        budget[f"coupon separation at L, 5 mrad heading offset ({label})"] = {
            "value": heading_end, "unit": "mm",
            "components": {"instrument": u_pair, "geometry": geometry_q2,
                           "execution": _start_term(u, j_lat_end, j_head_end), "solver": solver_q2}}
    # Open loop, the start pose has two terms: the insert and laying, and the two seatings of the start jig
    # (each tape is laid after its own seating). Their quadrature sum is the execution component.
    budget["coupon separation at L, 5 mrad heading offset (open-loop start)"]["execution_terms"] = {
        name: _start_term(u, j_lat_end, j_head_end) for name, u in EXECUTION["terms"].items()}
    # Q3: coupon separation at the route end for the 2 mm lateral offset (the T138 quantity): open loop,
    # conditioned on the measured start pose, and conditioned also on the as-built scan (T129).
    scan_covariance = as_built_study()["covariance_mm2"]
    for label, execution, geometry in (
            ("open-loop start", prediction["execution_mm"][-1], prediction["geometry_mm"][-1]),
            ("conditioned on the measured start pose", prediction["conditioning_mm"][-1], prediction["geometry_mm"][-1]),
            ("conditioned on the start pose and the as-built scan", prediction["conditioning_mm"][-1],
             prediction["geometry_scan_mm"][-1])):
        budget[f"coupon separation at L, 2 mm lateral offset ({label})"] = {
            "value": prediction["separation_mm"][-1], "unit": "mm",
            "components": {"instrument": u_pair, "geometry": geometry, "execution": execution,
                           "solver": prediction["solver_mm"][-1]}}
    budget["coupon separation at L, 2 mm lateral offset (open-loop start)"]["execution_terms"] = {
        name: terms[-1] for name, terms in prediction["execution_terms_mm"].items()}
    # Q4: arclength of the first focal point; a start heading error moves the zero of
    # delta j_lat + dtheta j_head by dtheta j_head(s_f) / (delta |j_lat'(s_f)|).
    slope = abs(2.0 * sensitivity["center"]["j_lat_prime_focal"])
    j_head_focal = float(np.interp(nominal["focal_mm"], nominal["profile_s"], nominal["profile_j_head"]))
    geometry_q4 = math.hypot(sensitivity["height_mm"]["focal"]["central"], sensitivity["sigma_mm"]["focal"]["central"])
    for label, geometry in (("conditioned on the measured start pose", geometry_q4),
                            ("conditioned on the start pose and the as-built scan",
                             float(scan_geometry(sensitivity, "focal", scan_covariance)))):
        budget[f"coupon focal distance ({label})"] = {
            "value": nominal["focal_mm"], "unit": "mm",
            "components": {"instrument": u_pair / slope, "geometry": geometry,
                           "execution": START_POSE_U["heading_rad"] * abs(j_head_focal) / slope,
                           "solver": abs(nominal["focal_mm"] - nominal["focal_h2_mm"]) / 15.0}}
    # Q5: flat-plate control, heading-offset separation at 240 mm. Geometry: the declared flatness as a
    # bump centred under the route (the worst placement); its effect is second order in the deviation, so the
    # difference from the plane is used.
    plate_value = plate["heading_separation_mm"][-1]
    bump = jacobi.transfer(geo.coupon(PLATE_FLATNESS["deviation_mm"] / math.sqrt(3.0), PLATE_FLATNESS["sigma_mm"]),
                           [-120.0, 0.0], 0.0, 240.0, steps=48)
    geometry_plate = abs(0.005 * bump.states[-1, 6] - plate_value)
    for label, u in (("open-loop start", EXECUTION), ("conditioned on the measured start pose", START_POSE_U)):
        budget[f"plate separation at 240 mm, 5 mrad heading offset ({label})"] = {
            "value": plate_value, "unit": "mm",
            "components": {"instrument": u_pair, "geometry": geometry_plate, "execution": _start_term(u, 1.0, 240.0),
                           "solver": 0.005 * plate["phi_error"]}}
    budget["plate separation at 240 mm, 5 mrad heading offset (open-loop start)"]["execution_terms"] = {
        name: _start_term(u, 1.0, 240.0) for name, u in EXECUTION["terms"].items()}
    # Q6: control with a deliberately coarse solver (6 steps): the classification must say solver-limited.
    # Its reported value is the coarse solution q(6), whose error is (16/15) |q(6) - q(12)|.
    q = {n: 0.005 * jacobi.transfer(geo.COUPON, geo.STATION, 0.0, length, steps=n).states[-1, 6] for n in (6, 12, 256)}
    coarse_estimate = abs(q[6] - q[12]) * 16 / 15
    budget["coarse-solver control (6 RK4 steps)"] = {"value": q[6], "unit": "mm",
                                                    "components": {"instrument": u_pair, "geometry": geometry_q2,
                                                                   "execution": _start_term(START_POSE_U, j_lat_end, j_head_end),
                                                                   "solver": coarse_estimate},
                                                    "reference_256_steps": q[256]}
    for entry in budget.values():
        entry["dominant"], entry["dominant_share"] = _classify(entry["components"])
        entry["combined_standard_mm"] = math.sqrt(sum(v ** 2 for v in entry["components"].values()))
    linearity = {}
    for key in ("lateral", "heading_end", "focal"):
        for name in ("height_mm", "sigma_mm"):
            linearity[f"{key}/{name}"] = _linearity(sensitivity[name][key]["forward"], sensitivity[name][key]["central"])
    return {"budget": budget, "linearity": linearity,
            "coarse_check": {"estimate": coarse_estimate, "actual": abs(q[6] - q[256])},
            "standard_uncertainty": sensitivity["standard_uncertainty"], "execution_declared": EXECUTION,
            "start_pose_uncertainty": START_POSE_U, "plate_flatness": PLATE_FLATNESS}


@_task("T140", ("test_uncertainty_budget_classifies_limiting_terms",))
def uncertainty_budget(ctx):
    study = ctx.memo("mfg.budget", budget_study)
    budget = study["budget"]
    coarse = study["coarse_check"]
    # Open loop, the execution component is the quadrature sum of the insert-and-laying and jig re-seating terms.
    split = max(abs(math.hypot(*e["execution_terms"].values()) / e["components"]["execution"] - 1.0)
                for e in budget.values() if "execution_terms" in e)
    f_budget = finding("Uncertainty budget per predicted quantity and its limiting term", "numerical",
                       {name: dict({k: v for k, v in e["components"].items()}, value=e["value"], dominant=e["dominant"],
                                   **({"execution_terms": e["execution_terms"]} if "execution_terms" in e else {}))
                        for name, e in budget.items()},
                       {"checks": [_check("self_convergence", f"geometry sensitivity linearity (forward vs central), {k}", v, 0.1)
                                   for k, v in sorted(study["linearity"].items())]
                        + [_check("self_convergence", "coarse control: actual error / Richardson estimate",
                                  coarse["actual"] / coarse["estimate"], 3.0, "le"),
                           _check("self_convergence", "coarse control: Richardson estimate / actual error",
                                  coarse["estimate"] / coarse["actual"], 3.0, "le"),
                           _check("invariant", "open-loop execution component vs the quadrature sum of its insert-and-"
                                  "laying and jig re-seating terms (relative)", split, 1e-9)]},
                       unit="mm",
                       uncertainty=_u("reference_error", max(study["linearity"].values()),
                                      "forward vs central geometry sensitivity (relative)"),
                       tolerance={"abs": 1e-7, "rel": 1e-5})
    focal = budget["coupon focal distance (conditioned on the measured start pose)"]
    f_counter = finding("The coupon focal-distance prediction is geometry-limited, not instrument-limited, under the "
                        "declared dome tolerances, instrument and start-pose uncertainties", "numerical",
                        focal["dominant_share"],
                        {"checks": [_check("analytic", "geometry share of the focal-distance variance minus one half", focal["dominant_share"] - 0.5, 1e-9, "ge"),
                                    _check("exact_arithmetic", "dominant term is geometry", 0.0 if focal["dominant"] == "geometry" else 1.0, 0.0)]},
                        uncertainty=_u("reference_error", max(study["linearity"].values()),
                                       "forward vs central geometry sensitivity (relative)"),
                        tolerance={"abs": 1e-9, "rel": 1e-6},
                        counterexample={"statement": "The uncertainty of a curved-surface prediction compared with a "
                                                     "photogrammetric measurement is limited by the instrument",
                                        "witness": {"quantity": "coupon focal distance",
                                                    "components_mm": focal["components"]}})
    open_q2 = budget["coupon separation at L, 5 mrad heading offset (open-loop start)"]
    cond_q2 = budget["coupon separation at L, 5 mrad heading offset (conditioned on the measured start pose)"]
    f_start = finding("On the coupon, the heading-offset separation at the route end is limited by the start pose open "
                      "loop, and a CMM start-pose measurement removes that limit, under the declared instrument and "
                      "start-pose uncertainties", "numerical",
                      {"open_loop_share": open_q2["dominant_share"], "open_loop_dominant": open_q2["dominant"],
                       "conditioned_execution_share": cond_q2["components"]["execution"] ** 2 / cond_q2["combined_standard_mm"] ** 2},
                      {"checks": [_check("exact_arithmetic", "open-loop dominant term is execution (start pose)",
                                         0.0 if open_q2["dominant"] == "execution" else 1.0, 0.0),
                                  _check("analytic", "conditioned execution variance share (must be below one half)",
                                         cond_q2["components"]["execution"] ** 2 / cond_q2["combined_standard_mm"] ** 2, 0.5, "le")]},
                      uncertainty=_u("reference_error", max(study["linearity"].values()),
                                     "forward vs central geometry sensitivity (relative)"),
                      tolerance={"abs": 1e-9, "rel": 1e-6})
    scan_rows = {"focal distance": ("coupon focal distance (conditioned on the measured start pose)",
                                    "coupon focal distance (conditioned on the start pose and the as-built scan)"),
                 "2 mm lateral separation at L": (
                     "coupon separation at L, 2 mm lateral offset (conditioned on the measured start pose)",
                     "coupon separation at L, 2 mm lateral offset (conditioned on the start pose and the as-built scan)")}
    reduction = {quantity: budget[scanned]["components"]["geometry"] / budget[declared]["components"]["geometry"]
                 for quantity, (declared, scanned) in scan_rows.items()}
    f_scan = finding("Conditioning on the as-built scan (T129) shrinks the geometry term of the coupon focal distance and "
                     "of the 2 mm lateral separation at L at least tenfold, under the declared scanner uncertainties and "
                     "the Gaussian dome model", "numerical",
                     {quantity: {"geometry_declared_tolerances_mm": budget[declared]["components"]["geometry"],
                                 "geometry_as_built_scan_mm": budget[scanned]["components"]["geometry"],
                                 "limiting_after_scan": budget[scanned]["dominant"]}
                      for quantity, (declared, scanned) in scan_rows.items()},
                     {"checks": [_check("analytic", f"geometry term with the scan-derived dome covariance / with the declared "
                                        f"tolerances, {quantity}", ratio, 0.1, "le")
                                 for quantity, ratio in sorted(reduction.items())]},
                     unit="mm",
                     uncertainty=_u("reference_error", max(study["linearity"].values()),
                                    "forward vs central geometry sensitivity (relative); the scan covariance is first order"),
                     tolerance={"abs": 1e-9, "rel": 1e-6})
    control = budget["coarse-solver control (6 RK4 steps)"]
    f_control = finding("The classification detects a solver-limited prediction in the coarse-solver control", "numerical",
                        control["dominant"],
                        {"checks": [_check("exact_arithmetic", "coarse control classified solver-limited",
                                           0.0 if control["dominant"] == "solver" else 1.0, 0.0)]},
                        uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    ctx.artifact_json("uncertainty-budget.json", _r(study))
    lines = ["| Quantity | Value | Instrument | Geometry | Execution (start pose) | Solver | Limiting term |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, e in budget.items():
        c = e["components"]
        lines.append(f"| {name} | {e['value']:.4g} {e['unit']} | {c['instrument']:.2e} | {c['geometry']:.2e} | "
                     f"{c['execution']:.2e} | {c['solver']:.2e} | {e['dominant']} ({100 * e['dominant_share']:.0f}%) |")
    lines += ["", "Open-loop execution terms (insert and laying; two seatings of the start jig):"]
    lines += [f"- {name}: " + ", ".join(f"{term} {value:.2e}" for term, value in e["execution_terms"].items())
              for name, e in budget.items() if "execution_terms" in e]
    ctx.artifact_text("uncertainty-budget.md", "\n".join(lines) + "\n")
    findings = [f_budget, f_counter, f_start, f_scan, f_control,
                _not_measured("The declared instrument uncertainties are the uncertainties of the instruments used",
                              "calibration"),
                _not_measured(EXECUTION_CLAIM, "calibration"),
                _not_measured("The budget contains every significant physical error source (thermal, fixturing, tape "
                              "bending along the route, target centring)")]
    fields = _fields(
        "Under the declared instrument, start-pose and dome uncertainties each predicted quantity has a limiting "
        "uncertainty term: extrinsic chord-geodesic gaps are instrument-limited; heading-offset separations and the "
        "plate separation are limited open loop by the realized start pose (a CMM start-pose measurement removes that "
        "limit on the coupon but not over the 240 mm plate route, where the 20 mm heading baseline still dominates); the "
        "2 mm lateral-offset separation at the coupon route end and the location of a Jacobi focus are geometry-limited "
        "under the declared dome tolerances, because they depend sensitively on the dome shape; conditioning on an "
        "as-built scan (T129) replaces those tolerances by the scan-derived dome covariance and shrinks the geometry term.",
        "u_c^2 = u_instrument^2 + u_geometry^2 + u_execution^2 + u_solver^2; u_geometry from central differences over "
        "rectangular tolerances (u = a / sqrt 3), or g^T C g with the MFG-SCAN-01 dome covariance C over height, width "
        "and centre when conditioned on the scan; u_execution = start-pose uncertainty times the Jacobi fields "
        "(declared insert-and-laying error and two jig seatings open loop, CMM estimate when conditioned); u_solver = |Q(h) - Q(2h)| / 15 for a reported "
        "h-solution and (16/15) |Q(h) - Q(2h)| when the reported value is the coarse one (the 6-step control); "
        "u_instrument(focal) = u_pair / |d sep / ds|; limiting term = variance share > 50%, otherwise mixed.",
        ["Declared camera pair uncertainty sqrt(2) x 0.02 mm; a measured cylinder gap (film surface distance minus "
         "camera chord) adds the declared 0.05 mm film gauge", "Declared tolerances: cylinder radius +/- 0.1 mm, dome height "
         "+/- 0.2 mm, dome sigma +/- 0.5 mm, plate flatness 0.05 mm",
         "Declared start-pose error open loop: insert and laying 0.05 mm / 0.5 mrad and two jig seatings of 0.01 mm / "
         "0.1 mrad (combined 0.052 mm / 0.52 mrad); CMM start-pose estimate 0.0028 mm / 0.2 mrad",
         "Scan-derived dome covariance of MFG-SCAN-01 (T129: synthetic verification, declared scanner noise, scale and "
         "registration)",
         "Predictions from T126, T127, T128 and T138"],
        "No observation: declared instrument noise, declared start-pose errors and model sensitivities only.",
        "Forward and central sensitivities agree (linear regime); the Richardson estimate brackets the actual solver error.",
        "Compute each component, classify, compare open-loop and conditioned start poses and the declared-tolerance and "
        "scan-derived geometry terms, and validate the solver estimate against a 256-step reference in a deliberately "
        "coarse control.",
        "; ".join(f"{name} = {e['value']:.4g} mm, u (instrument, geometry, execution, solver) = ("
                  f"{e['components']['instrument']:.2g}, {e['components']['geometry']:.2g}, "
                  f"{e['components']['execution']:.2g}, {e['components']['solver']:.2g}) mm -> {e['dominant']}"
                  for name, e in budget.items()) + ".",
        "Components are standard uncertainties; classification uses variance shares.",
        ["nonlinear geometry sensitivity (lateral, heading, focal)", "Richardson estimate validity",
         "solver-limited control", "start pose open loop vs conditioned (heading and lateral offsets)",
         "declared dome tolerances vs scan-derived dome covariance", "plate flatness as a second-order geometry term"],
        ["Instrument and start-pose uncertainties are declared; the real budget needs calibration records (T130/T139).",
         "The scan-derived geometry terms hold only if the as-built coupon passes the MFG-SCAN-01 model test.",
         "In-plane tape bending along the route, thermal drift and target centring are not budgeted."],
        "Open: budget in-plane tape bending along the route, thermal drift and target centring, which every row "
        "omits, and replace the declared instrument and start-pose terms by values from retained calibration records "
        "once a real acquisition and its calibration certificates exist.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T141 production acceptance boundary ------------------------------------------------------------
ACCEPTANCE_STATEMENT = "Coupon lot accepted for production"
REJECTION_STATEMENT = "Coupon lot rejected for production"
# A decision word only this section's screen knows, and a paraphrase that neither screen matches.
SECTION_ONLY_STATEMENT = "Coupon lot scrapped"
PARAPHRASE = "Coupon lot fit for shipment to the customer"


def _finding_refusal(claim, domain, basis) -> str | None:
    """The refusal evidence.finding raises for a claim ('authority_outcome_refused' for the phrase screen), or None."""
    try:
        finding(claim, domain, "decision", basis)
    except EvidenceRefusal as exc:
        return "authority_outcome_refused" if "authority outcome" in str(exc) else "other"
    return None


def acceptance_boundary() -> dict:
    """Exhaustively check that no basis establishes production acceptance, and that the policy and screens refuse decisions."""
    passing = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
    acquisition = {"device": "camera:1:SN", "raw_sha256": "0" * 64, "acquired_at": "2026-09-23T00:00:00Z",
                   "calibration": "CERT-1"}
    # A hypothetical independent check: enumerated as a basis shape, never run.
    independent = dict(passing, producer={"implementation": "ciw.lab", "revision": f"ciw {__version__}"},
                       checker={"implementation": "scipy", "revision": "hypothetical (enumerated basis; no check ran)"})
    provider = {"repository": "owner/provider", "revision": "a" * 40, "source_tree": "b" * 40, "executed": True}
    options = {"derivation": (None, "doc"), "generator": (None, {"name": "g"}), "checks": (None, [passing]),
               "provider": (None, provider), "independent_check": (None, independent), "acquisition": (None, acquisition)}
    cases = violations = 0
    for values in itertools.product(*options.values()):
        basis = {key: value for key, value in zip(options, values) if value is not None}
        for domain in sorted(AUTHORITY_DOMAINS):
            cases += 1
            try:
                label = supported_label(basis, domain)
            except EvidenceRefusal:
                label = "refused"
            violations += label not in ("not_established", "refused")
    policy = rec.AcceptancePolicy()
    decisions = {kind: rec.refusal_code(policy.decide, {"part": "coupon-001", "decision": kind})
                 for kind in ("accept", "reject", "conditional")}
    record = finding(ACCEPTANCE_STATEMENT, "production_acceptance", "accepted", {"acquisition": acquisition})
    forged = dict(record, evidence_status="hardware_measured")
    try:
        validate_finding(forged)
        forged_code = None
    except EvidenceRefusal as exc:
        forged_code = "label_refused" if "refused" in str(exc) else "other"
    # The loophole this task recorded, now closed by evidence.screen_authority_claim: the acceptance statement
    # filed in a computational (or physical) domain is refused by evidence.finding and by validate_finding.
    core = {"acceptance filed as computational_pipeline": _finding_refusal(ACCEPTANCE_STATEMENT, "computational_pipeline",
                                                                           {"checks": [passing]}),
            "rejection filed as computational_pipeline": _finding_refusal(REJECTION_STATEMENT, "computational_pipeline",
                                                                          {"checks": [passing]}),
            "acceptance filed as physical with an acquisition record": _finding_refusal(
                ACCEPTANCE_STATEMENT, "physical", {"acquisition": acquisition})}
    honest = finding("Coupon lot conforms to the drawing", "computational_pipeline", "decision", {"checks": [passing]})
    try:
        validate_finding(dict(honest, claim=ACCEPTANCE_STATEMENT))
        core["hand-built computational record carrying the acceptance statement (validate_finding)"] = None
    except EvidenceRefusal as exc:
        core["hand-built computational record carrying the acceptance statement (validate_finding)"] = (
            "authority_outcome_refused" if "authority outcome" in str(exc) else "other")
    # This section's screen: a hand-built record with the statement, and a decision word the core screen passes.
    section = {"acceptance statement in a hand-built computational record": rec.refusal_code(
                   rec.screen_acceptance_language, [dict(honest, claim=ACCEPTANCE_STATEMENT)]),
               "rejection statement in a hand-built computational record": rec.refusal_code(
                   rec.screen_acceptance_language, [dict(honest, claim=REJECTION_STATEMENT)])}
    only_section = finding(SECTION_ONLY_STATEMENT, "computational_pipeline", "decision", {"checks": [passing]})
    section[f"'{SECTION_ONLY_STATEMENT}', which evidence.finding accepts"] = rec.refusal_code(
        rec.screen_acceptance_language, [only_section])
    # What remains open: a paraphrase outside both vocabularies is labelled by its checks.
    paraphrase = finding(PARAPHRASE, "computational_pipeline", "decision", {"checks": [passing]})
    return {"cases": cases, "violations": violations, "decisions": decisions, "honest_label": record["evidence_status"],
            "forged_code": forged_code, "policy_record": policy.record({"part": "coupon-001"}),
            "domains": sorted(AUTHORITY_DOMAINS), "all_domains": len(DOMAINS), "core_refusals": core,
            "section_refusals": section, "section_only_label": only_section["evidence_status"],
            "paraphrase_label": paraphrase["evidence_status"],
            "paraphrase_section_screen": rec.refusal_code(rec.screen_acceptance_language, [paraphrase])}


@_task("T141", ("test_production_acceptance_stays_outside_the_system",))
def acceptance_outside(ctx):
    study = acceptance_boundary()
    plate = ctx.memo("mfg.plate", plate_study)
    matrix_code = rec.refusal_code(rec.validate_protocol, dict(_minimal_protocol(plate), production_acceptance="accepted"))
    criterion = _minimal_protocol(plate)
    criterion["acceptance_criteria"][0]["status"] = "accepted"
    criterion_code = rec.refusal_code(rec.validate_protocol, criterion)
    checks = [_check("exact_arithmetic", "bases establishing a claim filed in an authority domain", study["violations"], 0.0)]
    checks += [_refusal(f"policy.decide({kind})", "production_acceptance_outside_system", code)
               for kind, code in sorted(study["decisions"].items())]
    checks += [_refusal("forged hardware_measured acceptance finding", "label_refused", study["forged_code"]),
               _refusal("protocol declaring acceptance inside the system", "acceptance_inside_system", matrix_code),
               _refusal("protocol criterion marked accepted", "criterion_is_decision", criterion_code)]
    checks += [_refusal(f"section screen: {name}", "acceptance_outside_authority_domain", code)
               for name, code in sorted(study["section_refusals"].items())]
    f_api = finding("No basis establishes a claim filed in an authority domain, and the acceptance policy, the protocol "
                    "validator and this section's acceptance-language screen refuse acceptance decisions",
                    "computational_pipeline", {"basis_domain_cases": study["cases"], "violations": study["violations"],
                                               "refusals": sum(c["passed"] for c in checks[1:])},
                    {"checks": checks}, unit="cases", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0})
    core = study["core_refusals"]
    f_closed = finding("evidence.finding and validate_finding refuse an acceptance or rejection statement filed in a "
                       "computational or physical domain because of its wording", "computational_pipeline",
                       sum(code == "authority_outcome_refused" for code in core.values()),
                       {"checks": [_refusal(f"evidence screen: {name}", "authority_outcome_refused", code)
                                   for name, code in sorted(core.items())]},
                       unit="refusals", uncertainty=EXACT, tolerance={"abs": 0, "rel": 0},
                       counterexample={"statement": "evidence.finding labels a claim from its basis and domain alone, so an "
                                                    "acceptance statement filed in a computational domain with a passing "
                                                    "check is established (the loophole T141 recorded before "
                                                    "evidence.screen_authority_claim)",
                                       "witness": {"claim": ACCEPTANCE_STATEMENT, "domain": "computational_pipeline",
                                                   "refused_by": "ciw.lab.evidence.screen_authority_claim"}})
    f_paraphrase = finding("A paraphrased acceptance statement outside both screened vocabularies, filed in a "
                           "computational domain with a passing check, is still labelled by its checks",
                           "computational_pipeline", study["paraphrase_label"],
                           {"checks": [_check("exact_arithmetic", "label of the paraphrase is established (0 = yes)",
                                              0.0 if study["paraphrase_label"] != "not_established" else 1.0, 0.0),
                                       _refusal("section screen on the paraphrase", "none",
                                                study["paraphrase_section_screen"])]},
                           uncertainty=EXACT, tolerance={"abs": 0, "rel": 0},
                           counterexample={"statement": "The lab API cannot mark production acceptance",
                                           "witness": {"claim": PARAPHRASE, "domain": "computational_pipeline",
                                                       "label": study["paraphrase_label"],
                                                       "screens_passed": ["ciw.lab.evidence.screen_authority_claim",
                                                                          "manufacturing_records.screen_acceptance_language"]}})
    f_domain = finding("Domain assignment of free-text claims is machine-checked across the lab", "computational_pipeline",
                       None, {"notes": "evidence.screen_authority_claim refuses authority-outcome phrases in every "
                                       "section and this section also screens decision words, but both match phrases: "
                                       "a paraphrase outside them is labelled by its checks (recorded here), so the "
                                       "domain of a free-text claim remains a review question."},
                       expected_not_established=True)
    f_accept = finding("Production acceptance of the coupon, cylinder or plate process", "production_acceptance",
                       study["policy_record"]["decision"], {})
    f_ready = finding("The manufacturing protocols and models are ready for industrial use", "industrial_readiness", None, {})
    f_demand = finding("Manufacturers need curvature-aware placement, winding, coating or inspection path checking",
                       "customer_demand", None, {})
    ctx.artifact_json("acceptance-policy.json", {"policy": {"authority": "external", "decisions_performed": False},
                                                 "record": study["policy_record"], "cases": study["cases"],
                                                 "violations": study["violations"], "decisions": study["decisions"],
                                                 "honest_label_with_hardware_basis": study["honest_label"],
                                                 "authority_phrase_screen": core,
                                                 "section_screen": study["section_refusals"],
                                                 "section_only_statement": {"claim": SECTION_ONLY_STATEMENT,
                                                                            "label_by_evidence_finding":
                                                                                study["section_only_label"]},
                                                 "paraphrase": {"claim": PARAPHRASE, "label": study["paraphrase_label"],
                                                                "section_screen": study["paraphrase_section_screen"]}})
    fields = _fields(
        "Production acceptance is an authority decision outside the workbench: no evidence basis makes a claim filed in "
        "an authority domain established, no policy call or protocol field can record a decision, and an acceptance "
        "statement filed in another domain is refused by its wording where the screens know the phrase, while a "
        "paraphrase they do not know is still labelled by its checks.",
        "Label function L(basis, domain) = not_established for every authority domain; AcceptancePolicy.decide always "
        "refuses; protocols require production_acceptance = outside_system and hypothesis-status criteria; "
        "evidence.screen_authority_claim refuses authority-outcome phrases in computational and physical claims in "
        "every section; this section's screen also refuses decision words (accepted, approved, rejected, scrapped, "
        "quarantined, signed off, dispositioned, released for or to production, passed or passes inspection or "
        "acceptance, certified for production) in claims and string values outside the authority domains.",
        [f"{study['cases'] // len(AUTHORITY_DOMAINS)} bases (all combinations of derivation, generator, checks, provider, "
         f"independent check and acquisition) x the five authority domains = {study['cases']} cases",
         "Acceptance requests: accept, reject, conditional",
         f"'{ACCEPTANCE_STATEMENT}' and '{REJECTION_STATEMENT}' filed in computational_pipeline with a passing check, "
         "and the first filed as physical with an acquisition record",
         f"'{SECTION_ONLY_STATEMENT}' (a decision word only the section screen knows) and the paraphrase "
         f"'{PARAPHRASE}'"],
        "No observation: the task enumerates bases and domains through the label function and calls the policy, the "
        "protocol validator and both screens; no instrument, part, customer or acceptance authority is involved.",
        "Zero bases establish a claim filed in an authority domain; every decision request, forged record and screened "
        "statement is refused; the paraphrase shows what the screens cannot see.",
        "Enumerate bases, call the policy, forge a hardware_measured acceptance finding, mutate a protocol, build the "
        "acceptance and rejection statements through evidence.finding and validate_finding in computational and "
        "physical domains, screen hand-built records and a section-only decision word, and build a paraphrase that "
        "neither screen matches.",
        f"{study['cases']} cases, {study['violations']} violations; all decisions refused; honest label of an acceptance "
        f"claim even with hardware acquisition: {study['honest_label']}; evidence.finding refuses the acceptance and "
        f"rejection statements in computational and physical domains "
        f"({sum(code == 'authority_outcome_refused' for code in core.values())}/{len(core)} refusals); the section "
        f"screen refuses '{SECTION_ONLY_STATEMENT}', which evidence.finding labels {study['section_only_label']}; the "
        f"paraphrase is labelled {study['paraphrase_label']}.",
        "Exact: every outcome is a label or refusal code returned by deterministic validators; no quantity is estimated.",
        ["authority domain with hardware acquisition and passing checks", "forged label", "policy decide calls",
         "protocol acceptance field and criterion status",
         "acceptance and rejection statements filed in computational and physical domains (evidence screen)",
         "hand-built records bypassing evidence.finding (validate_finding and the section screen)",
         "decision word outside the evidence screen's phrases (section screen)", "paraphrase outside both screens"],
        ["The external acceptance authority and its criteria are outside the repository.",
         "Both screens match phrases, not meaning: a paraphrased decision filed in a computational domain is labelled "
         "by its checks, so choosing the domain remains a review question in every section.",
         "Customer demand for curvature-aware path checking is recorded, not surveyed: no customer or market data "
         "exists here."],
        "Deferred research question: make a claim's domain derivable from the claim instead of declared by its author "
        "(a claim grammar with typed subjects, or a classifier whose misses are themselves recorded), so that "
        "paraphrased authority outcomes such as the one recorded here are refused as reliably as screened phrases.")
    return {"state": "completed", "fields": fields,
            "findings": [f_api, f_closed, f_paraphrase, f_domain, f_accept, f_ready, f_demand]}


def _minimal_protocol(plate):
    """A small valid protocol used to probe the acceptance boundary."""
    record = finding("Flat-plate control: chord and geodesic marker distances coincide", "numerical", 0.0,
                     {"derivation": "plane geodesics are straight segments"})
    return build_protocol(
        "T141", "MFG-ACCEPTANCE-PROBE", "Acceptance boundary probe", "Probe the protocol validator.",
        {"kind": "flat plate", "surface_model": geo.PLATE.describe()}, plate["markers"][:2],
        [], [_predicted("P1", "chord - geodesic", 0.0, "mm", record)],
        [{"id": "H1", "status": "hypothesis", "statement": "chord = geodesic", "test": "E_n <= 1"}], ("camera", "cmm"), [],
        [capture_format("photogrammetry", TARGET_CAPTURE, "camera", [m["id"] for m in plate["markers"][:2]],
                        "ciw.lab.manufacturing_records.read_capture")])
