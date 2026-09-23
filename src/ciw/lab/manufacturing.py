"""Manufacturing and robotic use cases T126-T141: protocols, path sensitivities, rankings and boundaries.

Scope: machine-readable measurement protocols for a flat-plate, a rolled-
cylinder and a domed-coupon control experiment; metrology procedures
(curvature sampling, registration, calibration artifacts, datum frames, Gage
R&R); model sensitivities of tape placement, filament winding, coating or
welding and robotic inspection paths from the Jacobi transfer; path rankings
by calibration tolerance and focus margin; uncertainty budgets; the retention
schema for future measurements; and the production-acceptance boundary.

Non-claims: nothing in this section is measured. Every specimen, instrument
uncertainty, friction coefficient, steering limit and tolerance is a declared
input. Predictions are labelled from their computational basis; every claim
about a physical part, a real instrument, a real calibration, machine safety
or production acceptance is recorded as a ``not_established`` finding in its
physical or authority domain. Hardware-evidence slots in the protocols are
empty, and the workbench never accepts or rejects production parts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import numpy as np

from . import integrators, jacobi, svg
from . import manufacturing_geometry as geo
from . import manufacturing_metrology as met
from . import manufacturing_records as rec
from .evidence import AUTHORITY_DOMAINS, DOMAINS, EvidenceRefusal, finding, supported_label, validate_finding
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
               "declared_standard_uncertainty_mm": 0.02, "status": "declared_not_verified"},
    "tracker": {"id": "tracker", "kind": "laser tracker with 1.5 in SMR",
                "declared_standard_uncertainty_mm": 0.015, "length_dependent_um_per_m": 6.0,
                "status": "declared_not_verified"},
    "cmm": {"id": "cmm", "kind": "bridge CMM with touch-trigger probe",
            "declared_standard_uncertainty_mm": 0.002, "status": "declared_not_verified"},
    "scanner": {"id": "scanner", "kind": "laser line scanner", "declared_standard_uncertainty_mm": 0.01,
                "native_point_spacing_mm": 0.05, "status": "declared_not_verified"},
}
COVERAGE_K = 2.0
PAIR_U = math.sqrt(2.0) * INSTRUMENTS["camera"]["declared_standard_uncertainty_mm"]


# Common builders ---------------------------------------------------------------
def _check(kind, reference, observed, tolerance, comparison="abs_le"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": bool(holds)}


def _refusal(reference, expected, observed):
    observed = observed or "none"
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


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
    return task(task_id, changed_files=(MODULE, GEOMETRY, METROLOGY, RECORDS, DOC, *extra_files),
                regression_tests=tuple(f"{TESTS}::{name}" for name in tests))


def _not_measured(claim, domain="physical"):
    return finding(claim, domain, None, {})


# Measurement protocols -----------------------------------------------------------
FRAME_CHAIN = [
    {"parent": "INSTRUMENT", "child": "WORLD", "source": "instrument registration to the reference network"},
    {"parent": "WORLD", "child": "FIXTURE", "source": "fixture datum targets measured by the tracker"},
    {"parent": "FIXTURE", "child": "PART", "source": "3-2-1 datum frame from probed A/B/C features"},
    {"parent": "PART", "child": "CAD", "source": "nominal model placement in the datum frame (declared)"},
]
DATUMS = [
    {"id": "A", "role": "primary", "feature": "specimen back face (or mandrel axis for the cylinder)",
     "points": 3, "constrains": "z translation, rotations about x and y"},
    {"id": "B", "role": "secondary", "feature": "long reference edge (or axial scribe line)",
     "points": 2, "constrains": "y translation, rotation about z"},
    {"id": "C", "role": "tertiary", "feature": "end stop (or circumferential scribe)", "points": 1,
     "constrains": "x translation"},
    {"id": "PART", "role": "datum reference frame", "feature": "3-2-1 construction",
     "definition": "z normal of A; x = B direction projected into A; origin on A, B plane and C plane "
                   "(ciw.lab.manufacturing_metrology.datum_frame_321)"},
]
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


def build_protocol(task_id, protocol_id, title, purpose, specimen, markers, paths, predicted, criteria,
                   instruments, procedure) -> dict:
    record = {"schema": rec.PROTOCOL_SCHEMA, "protocol_id": protocol_id, "task_id": task_id, "title": title,
              "purpose": purpose, "specimen": specimen,
              "fixtures": [{"id": "FX-321", "kind": "3-2-1 kinematic nest",
                            "contacts": "three spherical rests on A, two on B, one on C",
                            "clamping": "normal to A, over the rests, torque declared per specimen"}],
              "datum_frames": deepcopy(DATUMS), "frame_chain": deepcopy(FRAME_CHAIN), "markers": markers,
              "paths": paths, "instruments": [deepcopy(INSTRUMENTS[key]) for key in instruments],
              "required_raw_data": list(REQUIRED_RAW), "calibration_artifacts": deepcopy(CALIBRATION_ARTIFACTS),
              "environment": dict(ENVIRONMENT),
              "procedure": ["Soak specimen and fixture; log temperatures.",
                            "Verify instruments on GS-25.4 and SB-1000 before and after; retain both checks.",
                            "Locate the specimen in FX-321; probe datums A, B, C and build the PART frame.",
                            *procedure,
                            "Repeat the full acquisition three times with re-fixturing (T131 design) and retain "
                            "every raw file through the T139 retention record."],
              "predicted_quantities": predicted, "acceptance_criteria": criteria,
              "hardware_measured": {"status": "not_acquired", "records": []},
              "production_acceptance": "outside_system"}
    return rec.validate_protocol(record)


def _predicted(identifier, quantity, value, unit, record):
    return {"id": identifier, "quantity": quantity, "value": value, "unit": unit,
            "evidence_status": record["evidence_status"], "finding_claim": record["claim"]}


def _protocol_findings(protocol, claim):
    matrix = rec.protocol_refusal_matrix(protocol)
    checks = [_refusal(f"protocol mutation {name}", case["expected"], case["observed"])
              for name, case in sorted(matrix.items())]
    matched = sum(case["observed"] == case["expected"] for case in matrix.values())
    return finding(claim, "computational_pipeline", matched, {"checks": checks}, unit="refused mutations",
                   tolerance={"abs": 0, "rel": 0})


# T126 flat plate ---------------------------------------------------------------------
PLATE_PITCH = 60.0


def plate_study() -> dict:
    coords = [-120.0, -60.0, 0.0, 60.0, 120.0]
    markers = [np.array([x, y]) for y in coords for x in coords]
    pairs = [(0, j) for j in range(1, 25)] + [(12, j) for j in range(25) if j not in (0, 12)]
    rows, max_gap, max_closure = [], 0.0, 0.0
    for i, j in pairs:
        a, b = markers[i], markers[j]
        delta = b - a
        distance = float(np.linalg.norm(delta))
        chord = float(np.linalg.norm(geo.PLATE.embedding(b) - geo.PLATE.embedding(a)))
        path = jacobi.transfer(geo.PLATE, a, math.atan2(delta[1], delta[0]), distance, steps=4)
        closure = float(np.linalg.norm(path.points[-1] - b))
        max_gap, max_closure = max(max_gap, abs(distance - chord)), max(max_closure, closure)
        rows.append({"pair": [f"M{i:02d}", f"M{j:02d}"], "geodesic_mm": distance, "chord_mm": chord,
                     "rk4_closure_mm": closure})
    control = jacobi.transfer(geo.PLATE, [-120.0, 0.0], 0.0, 240.0, steps=24)
    phi = control.matrix()
    phi_error = float(np.max(np.abs(phi - np.array([[1.0, 240.0], [0.0, 1.0]]))))
    stations = control.s[::6]
    lateral, dheading = 2.0, 0.005
    nonlinear = geo.separation_nonlinear(geo.PLATE, [-120.0, 0.0], 0.0, 240.0, 24, lateral, dheading, base=control)
    linear = geo.separation_linear(control, lateral, dheading)
    # On the plane the exactly rotated path separates as delta + s sin(dtheta).
    exact = lateral + control.s * math.sin(dheading)
    return {"markers": [{"id": f"M{k:02d}", "u_mm": m.tolist()} for k, m in enumerate(markers)], "pairs": rows,
            "max_gap_mm": max_gap, "max_closure_mm": max_closure, "phi_end": phi.tolist(), "phi_error": phi_error,
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
                     "checks": [_check("analytic", "max |geodesic - chord| over 47 marker pairs", study["max_gap_mm"], 1e-9),
                                _check("analytic", "RK4 geodesic closure onto the target marker", study["max_closure_mm"], 1e-9)]},
                    unit="mm", tolerance={"abs": 1e-9, "rel": 0})
    f_phi = finding("Flat-plate Jacobi transfer is [[1, s], [0, 1]] along the 240 mm control path", "numerical",
                    study["phi_end"],
                    {"checks": [_check("analytic", "max |Phi(240) - [[1, 240], [0, 1]]|", study["phi_error"], 1e-9),
                                _check("invariant", "Wronskian det Phi - 1", study["det_drift"], 1e-12),
                                _check("analytic", "exactly offset path minus delta + s sin(dtheta) (mm)",
                                       study["nonlinear_minus_exact_mm"], 1e-9)]},
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
         "test": "slope of heading-offset separation equals 5 mrad within 2 sigma of the regression slope"}]
    protocol = build_protocol(
        "T126", "MFG-FLAT-PLATE-01", "Flat-plate zero-curvature control",
        "Establish the measurement chain on a specimen where chord, geodesic and Jacobi predictions are trivial, "
        "so any residual belongs to the instruments, frames or procedure.",
        {"kind": "flat plate", "material": "6082-T6 aluminium (declared)", "nominal_mm": [300.0, 300.0, 6.0],
         "surface_model": geo.PLATE.describe(), "declared_flatness_mm": 0.05},
        study["markers"],
        [{"id": "N0", "start_u_mm": [-120.0, 0.0], "heading_rad": 0.0, "length_mm": 240.0},
         {"id": "L2", "of": "N0", "lateral_offset_mm": 2.0}, {"id": "H5", "of": "N0", "heading_offset_rad": 0.005}],
        predicted, criteria, ("camera", "tracker", "cmm"),
        ["Measure all 25 coded markers with the camera (12 stations) and the CMM; retain raw files.",
         "Draw or mark the nominal and offset paths with the robot-held marker; measure path points every 30 mm."])
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
        "No observation. The protocol specifies camera/CMM observation of markers and offset paths; its "
        "hardware-evidence slot is empty.",
        "chord - geodesic = 0; det Phi = 1; separation of offset paths linear in s.",
        "Compute chord and RK4 geodesic for 47 marker pairs; integrate the control path and its exact offset "
        "paths; assemble and validate the ciw.lab-measurement-protocol.v1 record; mutate it seven ways and "
        "confirm each mutation is refused.",
        f"max |chord - geodesic| = {study['max_gap_mm']:.3g} mm; max |Phi(240) - exact| = {study['phi_error']:.3g}; "
        f"heading-offset separation at 240 mm = {study['heading_separation_mm'][-1]:.4g} mm.",
        "Rounding only (flat metric, zero Christoffel symbols). Measurement uncertainty is declared, not known.",
        ["RK4 closure onto the target marker", "Wronskian drift", "nonlinear vs linear offset separation",
         "protocol with filled hardware slot but no acquisition", "acceptance criterion marked as a decision",
         "prediction labelled hardware_measured", "instrument uncertainty claimed verified"],
        ["Real plate flatness (declared 0.05 mm) and thermal state are not measured; their effect is bounded "
         "only by the declared tolerance.",
         "Instrument uncertainties are declared planning values, not calibrated values."],
        "Execute MFG-FLAT-PLATE-01 on hardware, retain it through the T139 record, then compare with T138.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T127 rolled cylinder -------------------------------------------------------------------
def gap_exact(distance, radius):
    return distance - 2.0 * radius * math.sin(distance / (2.0 * radius))


def resolvable_separation(radius, target_gap):
    """Smallest circumferential arc length whose chord-geodesic gap reaches ``target_gap`` (bisection)."""
    lo, hi = 0.0, math.pi * radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap_exact(mid, radius) < target_gap else (lo, mid)
    return 0.5 * (lo + hi)


def cylinder_study() -> dict:
    radius = geo.CYLINDER_RADIUS
    rings = (50.0, 150.0, 250.0)
    markers = [{"id": f"C{r:d}-{k:02d}", "u": [math.radians(30.0 * k), z]} for r, z in enumerate(rings) for k in range(12)]
    pair_specs = [((0.0, 150.0), (math.radians(d), 150.0), f"circumferential {d} deg") for d in (30, 60, 90, 120, 150, 180)]
    pair_specs += [((0.0, 50.0), (0.0, 150.0), "axial 100 mm"), ((0.0, 50.0), (0.0, 250.0), "axial 200 mm"),
                   ((0.0, 50.0), (math.radians(60), 150.0), "helical 60 deg / 100 mm"),
                   ((0.0, 50.0), (math.radians(90), 250.0), "helical 90 deg / 200 mm")]
    rows, max_closure, max_series = [], 0.0, 0.0
    for a, b, name in pair_specs:
        a, b = np.array(a), np.array(b)
        arc, dz = radius * (b[0] - a[0]), b[1] - a[1]
        geodesic = math.hypot(arc, dz)
        chord = float(np.linalg.norm(geo.CYLINDER.embedding(b) - geo.CYLINDER.embedding(a)))
        path = jacobi.transfer(geo.CYLINDER, a, math.atan2(dz, arc), geodesic, steps=16)
        closure = float(np.linalg.norm(geo.CYLINDER.embedding(path.points[-1]) - geo.CYLINDER.embedding(b)))
        row = {"pair": name, "geodesic_mm": geodesic, "chord_mm": chord, "gap_mm": geodesic - chord,
               "rk4_closure_mm": closure}
        if dz == 0.0:
            d = arc
            series = d ** 3 / (24 * radius ** 2) - d ** 5 / (1920 * radius ** 4)
            bound = d ** 7 / (322560 * radius ** 6)
            row.update(series_mm=series, series_remainder_bound_mm=bound)
            max_series = max(max_series, abs(series - row["gap_mm"]) - bound)
        rows.append(row)
        max_closure = max(max_closure, closure)
    helix = jacobi.transfer(geo.CYLINDER, [0.0, 0.0], math.radians(45.0), 300.0, steps=30)
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


@_task("T127", ("test_cylinder_protocol_predicts_chord_geodesic_gaps", "test_protocols_refuse_filled_slots_and_decisions"))
def rolled_cylinder_control(ctx):
    study = ctx.memo("mfg.cylinder", cylinder_study)
    gaps = {r["pair"]: r["gap_mm"] for r in study["pairs"]}
    f_gap = finding("Rolled-cylinder chord-geodesic gaps for the protocol marker pairs", "numerical", gaps,
                    {"derivation": "geodesic = sqrt((R dphi)^2 + dz^2) on the development; chord = |X(b) - X(a)|",
                     "checks": [_check("analytic", "RK4 geodesic closure onto the target marker (mm)", study["max_closure_mm"], 1e-9),
                                _check("analytic", "circumferential gap minus 2-term series beyond the alternating-series bound (mm)",
                                       study["series_excess_mm"], 1e-12, "le"),
                                _check("invariant", "axial (ruling) pairs have zero gap (mm)", study["axial_gap_mm"], 1e-9)]},
                    unit="mm", tolerance={"abs": 1e-9, "rel": 1e-9})
    f_flat = finding("Rolled-cylinder Jacobi transfer equals the flat-plate transfer (K = 0)", "numerical",
                     study["flat_transfer_error"],
                     {"checks": [_check("analytic", "max |Phi_cylinder(300) - [[1, 300], [0, 1]]|", study["flat_transfer_error"], 1e-9)]},
                     tolerance={"abs": 1e-9, "rel": 0})
    ninety = next(r for r in study["pairs"] if r["pair"] == "circumferential 90 deg")
    f_counter = finding("A marker chord differs from the surface distance on a developable (intrinsically flat) part",
                        "numerical", ninety["gap_mm"],
                        {"checks": [_check("analytic", "gap at 90 deg exceeds k u_pair (camera) by the ratio",
                                           ninety["gap_mm"] / (COVERAGE_K * PAIR_U), 1.0, "ge")]},
                        unit="mm", tolerance={"abs": 1e-9, "rel": 1e-9},
                        counterexample={"statement": "On an intrinsically flat (developable) part the straight chord between "
                                                     "two markers equals their surface distance, as on the flat plate",
                                        "witness": {"radius_mm": geo.CYLINDER_RADIUS, "dphi_deg": 90,
                                                    "geodesic_mm": ninety["geodesic_mm"], "chord_mm": ninety["chord_mm"]}})
    f_res = finding("Minimum circumferential marker separation that resolves the chord-geodesic gap at k = 2",
                    "numerical", {k: v["min_arc_mm"] for k, v in study["resolvable"].items()},
                    {"checks": [_check("analytic", "bisection root vs (24 R^2 g)^(1/3) series (relative)",
                                       study["resolvable_series_rel"], 0.01)]},
                    unit="mm", tolerance={"abs": 1e-8, "rel": 1e-9})
    predicted = [_predicted(f"G{k}", f"chord-geodesic gap, {name}", _r(value), "mm", f_gap)
                 for k, (name, value) in enumerate(gaps.items())]
    predicted.append(_predicted("J1", "Jacobi transfer along a 45 deg helix of 300 mm", [[1.0, 300.0], [0.0, 1.0]], "1, mm", f_flat))
    criteria = [
        {"id": "H1", "status": "hypothesis", "statement": "Measured chord equals the predicted chord for every pair and the "
         "surface (tape-measure or unrolled-film) distance equals the predicted geodesic",
         "test": f"E_n <= 1 with U = 2 u_pair = {COVERAGE_K * PAIR_U:.4f} mm plus the radius term of T140"},
        {"id": "H2", "status": "hypothesis", "statement": "Separation of offset helices is the flat-plate separation",
         "test": "fit slope and intercept; compare with the flat-plate control T126"}]
    protocol = build_protocol(
        "T127", "MFG-CYLINDER-01", "Rolled-cylinder control: extrinsic curvature without intrinsic curvature",
        "Separate extrinsic effects (chord vs geodesic) from intrinsic ones (Jacobi separation), which must "
        "match the flat plate.",
        {"kind": "rolled tube", "material": "rolled and seam-welded 3 mm aluminium (declared)",
         "nominal_radius_mm": geo.CYLINDER_RADIUS, "length_mm": 300.0, "declared_radius_tolerance_mm": 0.1,
         "surface_model": geo.CYLINDER.describe()},
        study["markers"],
        [{"id": "HX45", "start_u": [0.0, 0.0], "heading_rad": math.radians(45.0), "length_mm": 300.0},
         {"id": "HX45-L2", "of": "HX45", "lateral_offset_mm": 2.0}],
        predicted, criteria, ("camera", "tracker", "cmm"),
        ["Measure the three marker rings with the camera and tracker; probe the tube radius at the three rings "
         "with the CMM (radius enters T140).",
         "Mark the helix and its offset with the robot-held marker; measure both every 30 mm of arclength."])
    ctx.artifact_json("protocol-rolled-cylinder.json", protocol)
    ctx.artifact_json("cylinder-pairs.json", _r({"pairs": study["pairs"], "resolvable": study["resolvable"]}))
    circ = [r for r in study["pairs"] if "series_mm" in r]
    ctx.artifact_text("cylinder-gap.svg", svg.line_plot(
        [("exact gap", [r["geodesic_mm"] for r in circ], [r["gap_mm"] for r in circ]),
         ("d^3/24R^2 - d^5/1920R^4", [r["geodesic_mm"] for r in circ], [r["series_mm"] for r in circ])],
        title="Rolled cylinder R = 100 mm: chord-geodesic gap", xlabel="arc length (mm)", ylabel="gap (mm)"))
    findings = [f_gap, f_flat, f_counter, f_res,
                _protocol_findings(protocol, "Rolled-cylinder protocol record validates and refuses malformed variants"),
                _not_measured("Measured chords and surface distances on the physical tube match the predicted gaps"),
                _not_measured("The physical tube radius and roundness lie within the declared +/- 0.1 mm", "calibration")]
    res = study["resolvable"]
    fields = _fields(
        "A rolled cylinder has zero Gaussian curvature, so its Jacobi transfer equals the plate's, while its "
        "extrinsic curvature makes marker chords shorter than geodesic distances by d^3/(24 R^2) + O(d^5).",
        "Cylinder X(phi, z) = (R cos phi, R sin phi, z), R = 100 mm; development (R phi, z) is an isometry; "
        "gap(d) = d - 2 R sin(d / 2R) for circumferential pairs; resolvable arc solves gap(d) = k sqrt(2) u.",
        ["Declared tube R = 100 mm, length 300 mm, three rings of 12 markers", "Ten marker pairs (circumferential, axial, helical)",
         "Declared instruments (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm)"],
        "No observation. The protocol specifies marker and path measurements; the hardware slot is empty.",
        "gap = 0 on rulings (axial pairs); Phi_cylinder = Phi_plate; circumferential gaps follow the alternating series.",
        "Closed-form gaps checked by RK4 geodesic closure and by the series with its remainder bound; helix "
        "transfer compared with [[1, s], [0, 1]]; resolvable separation by bisection; protocol validated and mutated.",
        f"gap at 90 deg = {ninety['gap_mm']:.4f} mm (chord {ninety['chord_mm']:.3f} vs geodesic {ninety['geodesic_mm']:.3f}); "
        f"minimum resolvable arc: camera {res['camera']['min_arc_mm']:.1f} mm, tracker {res['tracker']['min_arc_mm']:.1f} mm, "
        f"CMM {res['cmm']['min_arc_mm']:.1f} mm.",
        "Closed forms; RK4 closure at rounding level. The radius tolerance term is budgeted in T140.",
        ["RK4 closure", "series remainder bound", "axial rulings give zero gap", "flat Jacobi transfer",
         "protocol refusal matrix"],
        ["Seam weld and out-of-roundness are not modelled; a real tube is not a perfect cylinder.",
         "Marker centre offsets (target thickness) are not modelled."],
        "Execute MFG-CYLINDER-01 and compare measured gaps in T138; budget the radius term with T140.")
    return {"state": "completed", "fields": fields, "findings": findings}


# T128 domed coupon ------------------------------------------------------------------------
NOMINAL_STATIONS = 8


def nominal_study() -> dict:
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
        remainder[label] = {"eps_mm": [0.4, 0.2, 0.1], "max_remainder_mm": errors,
                            "order": integrators.observed_order([0.4, 0.2, 0.1], errors)}
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
            "lateral_linear_mm": (lateral * fine.states[::stride, 4]).tolist(),
            "lateral_nonlinear_mm": nonlinear_lateral[::stride].tolist(),
            "heading_linear_mm": (dheading * fine.states[::stride, 6]).tolist(),
            "nonlinear_crossing_mm": crossings[lateral], "nonlinear_crossing_half_mm": crossings[0.5 * lateral],
            "crossing_extrapolated_mm": extrapolated,
            "remainder": remainder, "profile_s": fine.s.tolist(), "profile_j_lat": fine.states[:, 4].tolist(),
            "profile_j_head": fine.states[:, 6].tolist(), "lateral_mm": lateral, "dheading_rad": dheading}


@_task("T128", ("test_coupon_protocol_predicts_focal_crossing", "test_protocols_refuse_filled_slots_and_decisions"))
def curved_coupon(ctx):
    study = ctx.memo("mfg.nominal", nominal_study)
    length, focal = study["length_mm"], study["focal_mm"]
    f_phi = finding("Domed-coupon Jacobi transfer at the end of the nominal route", "numerical",
                    {"length_mm": length, "phi_end": study["phi_end"]},
                    {"checks": [_check("invariant", "Wronskian det Phi - 1 along the route", study["det_drift"], 1e-8),
                                _check("self_convergence", "Richardson estimate |Phi(h) - Phi(2h)| / 15", study["phi_richardson"], 1e-5)]},
                    tolerance={"abs": 1e-6, "rel": 1e-7})
    f_focal = finding("Laterally offset routes cross the nominal route at the first focal point", "numerical", focal,
                      {"checks": [_check("self_convergence", "focal point at h = 1 mm vs h = 2 mm (mm)",
                                         focal - study["focal_h2_mm"], 1e-3),
                                  _check("self_convergence", "offset crossings (2 and 1 mm) extrapolated in offset^2 "
                                         "minus the focal point (mm)", study["crossing_extrapolated_mm"] - focal, 0.02),
                                  _check("analytic", "focal point lies inside the route (s_f / L)", focal / length, 1.0, "le")]},
                      unit="mm", tolerance={"abs": 1e-4, "rel": 1e-7})
    rem = study["remainder"]
    f_order = finding("Linearized separation remainder is at least second order (third order on the symmetry axis)",
                      "numerical", {k: v["order"] for k, v in rem.items()},
                      {"checks": [_check("self_convergence", "off-axis remainder order", rem["off_axis"]["order"], 1.8, "ge"),
                                  _check("self_convergence", "on-axis remainder order - 3 (odd symmetry)", rem["on_axis"]["order"] - 3.0, 0.3)]},
                      tolerance={"abs": 1e-4, "rel": 1e-5})
    f_flat = finding("Curvature signature relative to the flat-plate control at the route end", "numerical",
                     {"coupon_lateral_2mm": study["lateral_nonlinear_mm"][-1], "plate_lateral_2mm": 2.0,
                      "coupon_heading_5mrad": study["heading_linear_mm"][-1], "plate_heading_5mrad": 0.005 * length},
                     {"checks": [_check("self_convergence", "linear vs nonlinear 2 mm lateral separation at the end (mm)",
                                        study["lateral_linear_mm"][-1] - study["lateral_nonlinear_mm"][-1], 0.05)]},
                     unit="mm", tolerance={"abs": 1e-6, "rel": 1e-7})
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
        {"id": "H1", "status": "hypothesis", "statement": "The 2 mm offset route crosses the nominal route near the predicted focal point",
         "test": "crossing arclength within the T140 expanded uncertainty of the focal distance"},
        {"id": "H2", "status": "hypothesis", "statement": "Measured separations follow the Jacobi prediction, not the flat-plate one",
         "test": "E_n <= 1 against S1/S2 and E_n > 1 against the flat-plate prediction at the last two stations"}]
    protocol = build_protocol(
        "T128", "MFG-COUPON-01", "Domed coupon: Jacobi focusing of offset paths",
        "Measure the separation of exactly offset robot paths across a dome, where positive curvature focuses "
        "laterally offset paths so that they cross the nominal path inside the coupon.",
        {"kind": "domed coupon", "material": "formed 2 mm aluminium sheet (declared)",
         "surface_model": geo.COUPON.describe(), "chart_extent_mm": {"x": list(geo.COUPON_X), "y": list(geo.COUPON_Y)},
         "declared_height_tolerance_mm": 0.2, "declared_sigma_tolerance_mm": 0.5,
         "crest_principal_radius_mm": geo.DOME_SIGMA ** 2 / geo.DOME_HEIGHT},
        [{"id": f"N{k}", "route": "nominal", "s_mm": s} for k, s in enumerate(stations)]
        + [{"id": f"L{k}", "route": "lateral 2 mm", "s_mm": s} for k, s in enumerate(stations)]
        + [{"id": f"H{k}", "route": "heading 5 mrad", "s_mm": s} for k, s in enumerate(stations)],
        [{"id": "N", "start_u_mm": list(geo.STATION), "heading_rad": 0.0, "end": "x = 140 mm", "length_mm": length},
         {"id": "L", "of": "N", "lateral_offset_mm": 2.0, "construction": "exp map along the start normal"},
         {"id": "H", "of": "N", "heading_offset_rad": 0.005}],
        predicted, criteria, ("camera", "tracker", "scanner"),
        ["Scan the coupon surface (T129 design) to identify the as-built dome before path execution.",
         "Execute the nominal and offset paths with the robot-held marker; place coded targets at the stations.",
         "Measure station targets by photogrammetry; fit the crossing arclength of the lateral offset path."])
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
    findings = [f_focal, f_phi, f_order, f_flat,
                _protocol_findings(protocol, "Domed-coupon protocol record validates and refuses malformed variants"),
                _not_measured("Measured separations of offset routes on the physical coupon follow the Jacobi prediction "
                              "and cross at the predicted focal point"),
                _not_measured("The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance",
                              "calibration")]
    fields = _fields(
        "Across the dome, positive Gaussian curvature focuses laterally offset geodesic routes: the 2 mm offset "
        "route crosses the nominal route inside the coupon, a signature absent on the flat plate and on the cylinder.",
        "Coupon z = h exp(-(x^2 + y^2) / (2 sigma^2)), h = 10 mm, sigma = 20 mm; nominal route from (-60, 0) along +x "
        "to x = 140; separation = delta j_lat + dtheta j_head with j'' + K j = 0.",
        ["Declared domed coupon and chart extent x in [-60, 140], y in [-100, 100] mm",
         "Perturbations: 2 mm lateral, 5 mrad heading; remainder study eps = 0.4, 0.2, 0.1 mm"],
        "No observation. The protocol specifies photogrammetric station targets on executed paths; hardware slot empty.",
        "det Phi = 1; focal point independent of step size; linear remainder O(eps^2) (O(eps^3) on the symmetry axis).",
        "RK4 transfer at h = 1 and 2 mm, Richardson estimate, exactly perturbed routes (exp-map lateral start) at "
        "three sizes on and off the symmetry axis, crossing of the nonlinear 2 mm offset route, protocol validation.",
        f"L = {length:.4f} mm; first focal point s_f = {focal:.3f} mm (s_f / L = {focal / length:.3f}); 2 mm offset "
        f"separation at the end = {study['lateral_nonlinear_mm'][-1]:.4f} mm (plate: 2 mm); remainder orders "
        f"{rem['on_axis']['order']:.2f} on axis, {rem['off_axis']['order']:.2f} off axis.",
        "Solver error ~1e-6 (Richardson); geometry and instrument terms are budgeted in T140.",
        ["step refinement of Phi and of the focal point", "Wronskian", "second- vs third-order remainder (symmetry)",
         "nonlinear crossing vs linear focal point", "protocol refusal matrix"],
        ["The formed coupon will not be an exact Gaussian; T129 scanning identifies the as-built surface.",
         "Robot path execution error is not modelled here; it enters T136 as calibration tolerance."],
        "T138: compare predicted and measured separation once MFG-COUPON-01 has been executed and retained (T139).")
    return {"state": "completed", "fields": fields, "findings": findings}
