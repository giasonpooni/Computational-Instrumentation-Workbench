"""Instrument observation experiments T045-T059.

Scope: typed observation modes and their refusals, the chord-versus-geodesic
correction and its cylinder coefficient, a synthetic pinhole stereo rig
observing markers on a declared cylinder (calibration, distortion,
quantization and pixel noise), encoder and IMU error models, asynchronous and
dropped and stale observations, raw/filtered/smoothed estimates, frame and
clock bookkeeping, and retention without admission. Derivations and designs
are in docs/lab/OBSERVATION.md.

Non-claims: every instrument, surface, marker, signal and noise parameter is
generated from declared values. Agreement with generated truth, closed forms,
sympy or scipy establishes properties of the models and of this code, never
the accuracy, calibration validity or safety of a physical instrument; those
claims are recorded as not_established findings in their physical or
authority domain.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .. import __version__
from . import observation_camera as cam
from . import observation_chord as chord
from . import observation_modes as om
from . import observation_signals as sig
from . import svg
from .evidence import finding
from .registry import task

MODULE = "src/ciw/lab/observation.py"
MODES_FILE = "src/ciw/lab/observation_modes.py"
CAMERA_FILE = "src/ciw/lab/observation_camera.py"
SIGNALS_FILE = "src/ciw/lab/observation_signals.py"
CHORD_FILE = "src/ciw/lab/observation_chord.py"
DOC = "docs/lab/OBSERVATION.md"
TESTS = "tests/test_lab_observation.py"
PRODUCER = {"implementation": "ciw.lab.observation", "revision": __version__}

# Two-sided 99.9 % normal quantile used for every Monte Carlo comparison.
Z999 = sig.normal_quantile(0.9995)
FIELD_KEYS = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
              "experiment", "numerical_result", "uncertainty", "failure_modes_checked", "unresolved_assumptions",
              "recommended_next_task")


def _tests(*names):
    return tuple(f"{TESTS}::{name}" for name in names)


def _fields(**values) -> dict:
    missing = [key for key in FIELD_KEYS if not values.get(key)]
    if missing:
        raise ValueError(f"Report fields missing: {missing}")
    return values


def _check(reference, observed, tolerance, comparison="abs_le", kind="analytic") -> dict:
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance,
             "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _z_check(reference, z) -> dict:
    return _check(f"{reference} (|z| <= {Z999:.4f}, two-sided 99.9 %)", z, Z999)


def refusal_code(action) -> str:
    """The ObservationRefusal code raised by ``action``, or ``accepted``."""
    try:
        action()
    except om.ObservationRefusal as exc:
        return exc.code
    return "accepted"


def _refusal(reference, expected, action) -> dict:
    observed = refusal_code(action)
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _unestablished(claim, domain, reason) -> dict:
    return finding(claim, domain, None, {"notes": reason})


def _generator(name, seed, **parameters) -> dict:
    return {"name": name, "seed": seed, **parameters}


def _floats(values) -> list:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _independent(reference, observed, tolerance, checker, revision) -> dict:
    return dict(_check(reference, observed, tolerance, kind="exact_arithmetic"), producer=dict(PRODUCER),
                checker={"implementation": checker, "revision": revision})


# Observation records -------------------------------------------------------------

EXAMPLE_VALUES = {"intrinsic_geodesic_distance": 0.12, "camera_chord_distance": 0.11292849467900708,
                  "reconstructed_surface_distance": 0.12, "encoder_displacement": 0.0425,
                  "tracker_measurement": (0.125, -0.25, 0.625), "image_residual": (0.25, -0.5),
                  "imu_orientation": (0.001, -0.002, 0.0005)}
FRAME_NAMES = {"surface_chart": "cylinder-r0.1", "camera_rig": "stereo-0", "reconstruction": "cylinder-r0.1",
               "axis": "x", "tracker": "room", "image": "left", "body": "imu-0"}
CYLINDER_MODEL = {"kind": "cylinder_geodesic", "name": "cylinder-r0.1", "radius": 0.1, "path_angle_rad": 0.0}


def example_observation(mode: str, **overrides) -> om.Observation:
    """A well-formed record of ``mode`` with declared (synthetic) references."""
    declared = om.MODES[mode]
    fields = {"unit": declared.unit, "frame_id": f"{declared.frame_kind}:{FRAME_NAMES[declared.frame_kind]}",
              "clock_id": "clock:daq", "clock_basis": declared.clock_basis, "epoch": "epoch:run-0", "time_s": 1.0,
              "calibration_ref": "calibration:declared-synthetic", "sequence": 0, "raw_ref": f"raw:{mode}:0"}
    if declared.requires_surface_model:
        fields["surface_model"] = dict(CYLINDER_MODEL)
    if declared.clock_basis == "arrival":
        fields["latency_s"] = 0.0078125
    value = overrides.pop("value", EXAMPLE_VALUES[mode])
    fields.update(overrides)
    return om.observe(mode, value, **fields)


# --------------------------------------------------------------- T045
@task("T045", changed_files=(MODULE, MODES_FILE, DOC), regression_tests=_tests("test_t045_modes_and_refusals"))
def typed_observation_modes(ctx):
    registry = {name: mode.describe() for name, mode in sorted(om.MODES.items())}
    required = ("quantity", "unit", "frame_kind", "clock_basis", "geometry", "noise_model", "cannot_observe")
    missing = [f"{name}.{key}" for name, entry in registry.items() for key in required if not entry[key]]
    invalid = [name for name, entry in registry.items()
               if entry["clock_basis"] not in om.CLOCK_BASES or entry["geometry"] not in om.GEOMETRY_CLASSES]
    intrinsic = sorted(name for name, entry in registry.items() if entry["geometry"] == "intrinsic")
    extrinsic = sorted(name for name, entry in registry.items() if entry["geometry"] == "extrinsic")
    accepted = sum(refusal_code(lambda m=name: om.validate(example_observation(m))) == "accepted" for name in registry)

    base = example_observation("camera_chord_distance")
    cases = (("frame id removed", "missing_frame", replace(base, frame_id=None)),
             ("clock id removed", "missing_clock", replace(base, clock_id=None)),
             ("clock epoch removed", "missing_epoch", replace(base, epoch=None)),
             ("calibration reference removed", "missing_calibration", replace(base, calibration_ref=None)),
             ("clock basis removed", "missing_clock_basis", replace(base, clock_basis=None)),
             ("arrival stamp on an acquisition-stamped mode", "clock_basis_mismatch",
              replace(base, clock_basis="arrival")),
             ("tracker frame on a camera-rig mode", "frame_kind_mismatch", replace(base, frame_id="tracker:room")),
             ("pixel unit on a metric mode", "unit_mismatch", replace(base, unit="px")),
             ("surface distance without a surface model", "surface_model_required",
              replace(example_observation("reconstructed_surface_distance"), surface_model=None)))
    record_checks = [_refusal(f"validate: {label}", code, lambda o=obs: om.validate(o)) for label, code, obs in cases]

    substitutions = {f"{provided}->{wanted}": refusal_code(
        lambda p=provided, w=wanted: om.require_mode(example_observation(p), w))
        for provided in registry for wanted in registry if provided != wanted}
    refused = sum(code == "mode_substitution" for code in substitutions.values())

    conversions, worst = [], 0.0
    for degrees in (0.0, 30.0, 60.0, 90.0):
        model = dict(CYLINDER_MODEL, path_angle_rad=math.radians(degrees))
        for arc in (0.015, 0.06, 0.12):
            measured = float(chord.helix_chord(arc, 0.1, math.radians(degrees)))
            converted = om.chord_to_surface_distance(example_observation("camera_chord_distance", value=measured),
                                                     model)
            om.validate(converted)
            worst = max(worst, abs(converted.value[0] - arc))
            conversions.append({"alpha_deg": degrees, "arc_m": arc, "chord_m": measured,
                                "recovered_arc_m": converted.value[0], "mode": converted.mode,
                                "derived_from": list(converted.derived_from)})
    chord_record = example_observation("camera_chord_distance")
    derived = om.chord_to_surface_distance(chord_record, CYLINDER_MODEL)
    conversion_checks = [
        _refusal("camera chord converted without a declared surface model", "surface_model_required",
                 lambda: om.chord_to_surface_distance(chord_record)),
        _refusal("camera chord used as an intrinsic geodesic distance", "mode_substitution",
                 lambda: om.require_mode(chord_record, "intrinsic_geodesic_distance")),
        _refusal("model-derived distance used as a direct intrinsic observation", "mode_substitution",
                 lambda: om.require_mode(derived, "intrinsic_geodesic_distance")),
        _check("recovered arc length against the generating arc length (m)", worst, 1e-13, kind="analytic")]

    ctx.artifact_json("observation-modes.json", registry)
    ctx.artifact_json("refusals.json", {"records": [{"case": label, "expected": code, "observed": check["observed_refusal"]}
                                                    for (label, code, _), check in zip(cases, record_checks)],
                                        "substitutions": substitutions, "chord_conversions": conversions})
    findings = [
        finding("Seven typed observation modes declare quantity, unit, frame kind, clock basis, geometry class, "
                "noise model and non-observables", "computational_pipeline",
                {"modes": len(registry), "intrinsic": intrinsic, "extrinsic": extrinsic,
                 "missing_declarations": len(missing), "examples_accepted": accepted},
                {"derivation": f"{DOC}#observation-modes",
                 "checks": [_check("missing declarations", len(missing), 0, kind="exact_arithmetic"),
                            _check("invalid clock basis or geometry class", len(invalid), 0, kind="exact_arithmetic"),
                            _check("well-formed examples accepted minus modes", accepted - len(registry), 0,
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 0, "rel": 0}),
        finding("Validation refuses records lacking frame, clock, epoch, calibration or clock-basis references",
                "computational_pipeline", [check["observed_refusal"] for check in record_checks],
                {"checks": record_checks}, tolerance={"abs": 0, "rel": 0}),
        finding("No observation mode stands in for another: every ordered substitution is refused",
                "computational_pipeline", {"ordered_pairs": len(substitutions), "refused": refused},
                {"checks": [_check("ordered pairs not refused as mode_substitution", len(substitutions) - refused, 0,
                                   kind="exact_arithmetic")] + conversion_checks[1:3]},
                tolerance={"abs": 0, "rel": 0}),
        finding("A camera chord becomes a surface distance only through a declared surface model",
                "numerical", worst, {"checks": [conversion_checks[0], conversion_checks[3]]}, unit="m",
                tolerance={"abs": 1e-12, "rel": 0}),
        _unestablished("The declared noise-model parameters describe real instruments of these modes",
                       "sensor_performance", "Noise parameters are declared placeholders; no instrument was acquired."),
    ]
    fields = _fields(
        hypothesis="A typed registry can make every observation declare its frame, clock basis, calibration and "
                   "geometry class, and can refuse malformed records and silent substitution of one mode for another.",
        mathematical_model="Each mode is a tuple (quantity, unit, components, frame kind, clock basis in "
                           "{acquisition, arrival}, geometry in {intrinsic, extrinsic, none}, noise model, "
                           "non-observables). A camera chord is extrinsic; it maps to a surface distance only through "
                           "an inverted chord-arc relation of a declared surface model (T046/T047).",
        input_data=["Seven declared modes in ciw.lab.observation_modes.MODES",
                    "One synthetic well-formed record per mode (example_observation)",
                    "Helix chords on a declared cylinder R = 0.1 m at alpha = 0, 30, 60, 90 deg"],
        observation_model="No instrument: records are generated with declared (synthetic) calibration references.",
        expected_invariant="Every well-formed record validates; every record with a missing reference or a "
                           "substituted mode is refused with a stable code; conversion needs a declared model.",
        experiment="Validate the registry declarations; strip each reference from a valid record; attempt all 42 "
                   "ordered mode substitutions; convert chords to arc lengths with and without a surface model.",
        numerical_result=f"{len(registry)} modes, {len(missing)} missing declarations; {len(record_checks)} record "
                         f"refusals as expected; {refused}/{len(substitutions)} substitutions refused; chord-to-arc "
                         f"inversion error {worst:.2e} m.",
        uncertainty="Refusals are exact; the chord inversion is a bisection converged to about 1e-16 relative.",
        failure_modes_checked=["missing frame, clock, epoch, calibration or clock basis",
                               "arrival stamp on an acquisition-stamped mode", "frame kind or unit mismatch",
                               "surface distance without surface model", "all 42 mode substitutions",
                               "model-derived distance presented as a direct intrinsic observation"],
        unresolved_assumptions=["Noise-model values are placeholders, not instrument characteristics",
                                "Frame kinds are declared strings; their physical realization is not checked"],
        recommended_next_task="T058: track exact frame and clock bases and apply declared mappings")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T046
@task("T046", changed_files=(MODULE, CHORD_FILE, DOC),
      regression_tests=_tests("test_t046_chord_expansion", "test_chord_derivation_degrades_without_sympy"))
def chord_geodesic_correction(ctx):
    closed = {"c3": "-kappa0^2/24", "c4": "-kappa0*kappa0'/24",
              "c5_constant_curvature_torsion": "kappa^4/1920 + kappa^2*tau^2/720"}
    basis = {"derivation": f"{DOC}#chord-versus-geodesic-correction"}
    symbolic = None
    if ctx.available("module:sympy"):
        symbolic = chord.sympy_versus_closed_form()
        basis["independent_check"] = _independent(
            "sympy Frenet-Serret series of |gamma(s) - gamma(0)| against the closed-form c3, c4, c5 at rational "
            "points (exact)", symbolic["max_abs_difference"], 0.0, "sympy", symbolic["sympy"])
        ctx.artifact_json("sympy-series.json", symbolic)
    sphere = ctx.memo("observation:sphere-chords", chord.sphere_study)
    torus = ctx.memo("observation:torus-chords", chord.torus_counterexample)
    helix = chord.helix_torsion_counterexample()
    ctx.artifact_json("sphere-geodesic-chords.json", sphere)
    ctx.artifact_json("counterexamples.json", {"torus": torus, "helix": helix})
    rows = sphere["rows"]
    ctx.artifact_text("sphere-chord-deficit.svg", svg.line_plot(
        [(f"R = {row['radius']}", row["curve"]["s"], row["curve"]["s_minus_c"]) for row in rows]
        + [(f"s^3/(24R^2), R = {row['radius']}", row["curve"]["s"],
            [x ** 3 / (24 * row["radius"] ** 2) for x in row["curve"]["s"]]) for row in rows[:1]],
        title="Sphere geodesics: chord deficit s - c", xlabel="arc length s", ylabel="s - c", logx=True, logy=True))
    ctx.artifact_text("torus-remainder.svg", svg.line_plot(
        [("start-point kappa", torus["curve"]["s"], torus["curve"]["start"]),
         ("midpoint kappa", torus["curve"]["s"], torus["curve"]["midpoint"])],
        title="Torus geodesic: |s - c - kappa^2 s^3/24|", xlabel="arc length s", ylabel="remainder",
        logx=True, logy=True))
    chord_error = max(row["relative_chord_error"] for row in rows)
    c3_error = max(row["c3_relative_error"] for row in rows)
    findings = [
        finding("Chord expansion c = s - kappa0^2 s^3/24 - kappa0 kappa0' s^4/24 + O(s^5), with "
                "c5 = kappa^4/1920 + kappa^2 tau^2/720 for constant curvature and torsion", "mathematical",
                closed, basis, tolerance={"abs": 0, "rel": 0}),
        finding("RK4 sphere geodesics reproduce the chord 2R sin(s/2R) and the s^3 coefficient kappa^2/24",
                "numerical", {"max_relative_chord_error": chord_error, "max_c3_relative_error": c3_error},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Sphere(R)", None, radii=[0.5, 1.0, 2.0],
                                         steps=400),
                 "checks": [_check("|c_integrated - 2R sin(s/2R)| / R", chord_error, 1e-12),
                            _check("fitted (s - c)/s^3 intercept relative to kappa^2/24", c3_error, 1e-6)]},
                tolerance={"abs": 1e-9, "rel": 0}),
        finding("A torus geodesic with varying curvature keeps the s^4 term kappa0 kappa0'/24 at the start point; "
                "curvature at the arc midpoint removes it", "numerical",
                {"kappa0": torus["kappa0"], "kappa0_prime": torus["kappa0_prime"],
                 "fitted_s4": torus["fitted_start_s4"], "predicted_s4": torus["predicted_c4_residual"],
                 "midpoint_ratio": torus["midpoint_ratio"]},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Torus(2, 1)", None, u0=torus["u0"],
                                         heading_rad=torus["heading_rad"], steps=torus["steps"]),
                 "checks": [_check("fitted s^4 coefficient relative to kappa0 kappa0'/24", torus["start_relative_error"],
                                   1e-4),
                            _check("|kappa0 kappa0'/24| is resolved away from zero", abs(torus["predicted_c4_residual"]),
                                   1e-3, "ge"),
                            _check("midpoint-curvature s^4 coefficient relative to kappa0 kappa0'/24",
                                   torus["midpoint_ratio"], 1e-4)]},
                tolerance={"abs": 1e-10, "rel": 1e-6},
                counterexample={"statement": "c = s - kappa(0)^2 s^3/24 + O(s^5) along every surface geodesic",
                                "witness": {"surface": "torus major 2, minor 1", "u0": torus["u0"],
                                            "heading_rad": torus["heading_rad"], "kappa0": torus["kappa0"],
                                            "kappa0_prime": torus["kappa0_prime"],
                                            "s4_coefficient": torus["fitted_start_s4"]}}),
        finding("Constant curvature alone does not give the circle chord: a cylinder helix with torsion deviates "
                "at order s^5 by kappa^2 tau^2/720", "numerical",
                {"kappa": helix["kappa"], "tau": helix["tau"], "fitted_s5_gap": helix["fitted_s5_gap"],
                 "predicted_s5_gap": helix["predicted_s5_gap"], "max_gap": helix["max_gap"]},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Cylinder(1)", None, alpha_deg=45.0,
                                         length=helix["length"]),
                 "checks": [_check("fitted (c - 2 sin(kappa s/2)/kappa)/s^5 relative to kappa^2 tau^2/720",
                                   helix["relative_error"], 1e-5),
                            _check("largest deviation from the circle chord (resolved)", helix["max_gap"], 1e-6, "ge"),
                            _check("integrated chord against the exact helix chord", helix["max_exact_chord_error"],
                                   1e-13)]},
                tolerance={"abs": 1e-12, "rel": 1e-6},
                counterexample={"statement": "For constant space curvature kappa, c = 2 sin(kappa s/2)/kappa",
                                "witness": {"surface": "cylinder R = 1", "alpha_deg": 45.0, "kappa": helix["kappa"],
                                            "tau": helix["tau"], "s": helix["length"], "gap": helix["max_gap"]}}),
        _unestablished("The chord correction computed from nominal curvature holds for chords measured on a "
                       "physical part", "physical", "No part was measured; curvature, marker placement and surface "
                       "form error of a real object are not modelled."),
    ]
    fields = _fields(
        hypothesis="Along a unit-speed surface geodesic the chord deficit is s - c = kappa^2 s^3/24 + O(s^4), with "
                   "kappa the absolute normal curvature; the O(s^4) term vanishes only when kappa' = 0.",
        mathematical_model="Taylor expansion of gamma(s) - gamma(0) in the Frenet frame (T' = kappa N, "
                           "N' = -kappa T + tau B, B' = -tau N): |delta|^2 = s^2 - kappa^2 s^4/12 - kappa kappa' s^5/12 "
                           "+ O(s^6), so c = s - kappa^2 s^3/24 - kappa kappa' s^4/24 + c5 s^5. For a geodesic k_g = 0, "
                           "hence kappa = |II(T, T)|. For a circle (tau = 0) c = 2 sin(kappa s/2)/kappa exactly.",
        input_data=["Sphere radii 0.5, 1, 2 (RK4, 400 steps over s in [0, R])",
                    "Torus major 2, minor 1, start (0, pi/4), heading 0.6 rad, 800 RK4 steps over s in [0, 0.4]",
                    "Cylinder R = 1 helix at 45 deg, 160 steps over s in [0, 0.8]",
                    "Rational sample points for the exact sympy comparison"],
        observation_model="Chords are Euclidean distances between embedded points of integrated geodesics; no "
                          "instrument noise.",
        expected_invariant="Integrated chords agree with closed forms to integration accuracy; fitted coefficients "
                           "agree with the derived ones.",
        experiment="Derive the series by hand and (when sympy is present) symbolically; integrate geodesics with "
                   "ciw.lab.jacobi; fit polynomial models of (s - c)/s^3 and of the remainders; search for "
                   "counterexamples to the O(s^5) remainder and to the constant-curvature circle formula.",
        numerical_result=f"sphere chord error {chord_error:.1e} R, c3 relative error {c3_error:.1e}; torus s^4 "
                         f"coefficient {torus['fitted_start_s4']:.6e} vs predicted {torus['predicted_c4_residual']:.6e} "
                         f"(midpoint ratio {torus['midpoint_ratio']:.1e}); helix s^5 gap {helix['fitted_s5_gap']:.6e} vs "
                         f"{helix['predicted_s5_gap']:.6e}"
                         + ("" if symbolic is None else f"; sympy difference {symbolic['max_abs_difference']}"),
        uncertainty="Fitted coefficients carry truncation error of the polynomial model (estimated below 1e-6 "
                    "relative) and RK4 error below 1e-12; kappa0' on the torus comes from a degree-8 fit of exact "
                    "normal-curvature samples.",
        failure_modes_checked=["start-point versus midpoint curvature", "nonzero torsion with constant curvature",
                               "integration drift (speed drift recorded)", "sympy unavailable (analytic fallback)"],
        unresolved_assumptions=["kappa must be known along the path; estimating it from data is not modelled",
                                "Markers are points on the surface; marker thickness and offsets are ignored"],
        recommended_next_task="T047: validate the cylinder chord coefficient cos^4(alpha)/(24 R^2)")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T047
ANGLES_DEG = (0, 15, 30, 45, 60, 75, 90)


@task("T047", changed_files=(MODULE, CHORD_FILE, DOC),
      regression_tests=_tests("test_t047_cylinder_coefficient", "test_chord_derivation_degrades_without_sympy"))
def cylinder_chord_coefficient(ctx):
    basis = {"derivation": f"{DOC}#cylinder-chord-coefficient"}
    symbolic = None
    if ctx.available("module:sympy"):
        symbolic = chord.sympy_cylinder_series(ANGLES_DEG)
        observed = symbolic["max_abs_difference"] if symbolic["symbolic_zero"] else 1.0
        basis["independent_check"] = _independent(
            "sympy series of sqrt((2R sin(s cos(a)/2R))^2 + (s sin(a))^2) against the closed-form c3 and c5 "
            "(symbolic residual zero; evaluated at 7 angles)", observed, 1e-14, "sympy", symbolic["sympy"])
        ctx.artifact_json("sympy-cylinder-series.json", symbolic)
    fits = chord.cylinder_fit(ANGLES_DEG, 1.0) + chord.cylinder_fit(ANGLES_DEG, 0.1)
    integrated = chord.cylinder_integrated(ANGLES_DEG, 1.0) + chord.cylinder_integrated(ANGLES_DEG, 0.1)
    ctx.artifact_json("coefficient-fits.json", {"exact_chord_fits": fits, "integrated_geodesics": integrated})
    unit = [row for row in fits if row["radius"] == 1.0]
    ctx.artifact_text("coefficient-vs-angle.svg", svg.line_plot(
        [("fitted (exact chord)", [r["alpha_deg"] for r in unit], [r["fitted"] for r in unit]),
         ("cos^4(alpha)/24", [r["alpha_deg"] for r in unit], [r["predicted"] for r in unit])],
        title="Cylinder R = 1: chord coefficient", xlabel="alpha from circumferential (deg)",
        ylabel="(s - c)/s^3 at s -> 0"))
    fit_error = max(row["normalized_error"] for row in fits)
    integrated_error = max(row["max_chord_error"] / row["radius"] for row in integrated)
    integrated_fit = max(row["normalized_error"] for row in integrated)
    rulings = [row for row in integrated if row["alpha_deg"] == 90]
    circumferential = [row for row in fits if row["alpha_deg"] == 0]
    ruling_deficit = max(row["max_s_minus_c"] / row["radius"] for row in rulings)
    dominance = max(max(r["fitted"] for r in fits if r["radius"] == row["radius"]) - row["fitted"]
                    for row in circumferential)
    circle_coefficient = next(r["fitted"] for r in circumferential if r["radius"] == 1.0)
    witness_s = 0.1
    witness_gap = float(witness_s - chord.helix_chord(witness_s, 1.0, 0.0))
    findings = [
        finding("The exact helix chord expands as s - cos^4(alpha) s^3/(24 R^2) + (kappa^4/1920 + "
                "kappa^2 tau^2/720) s^5 with kappa = cos^2(alpha)/R, tau = sin(alpha)cos(alpha)/R", "mathematical",
                {"c3": "-cos(alpha)^4/(24 R^2)", "c5": "cos(alpha)^6 (3 + 5 sin(alpha)^2)/(5760 R^4)"}, basis,
                tolerance={"abs": 0, "rel": 0}),
        finding("Small-s fits of exact helix chords recover cos^4(alpha)/(24 R^2) at every angle and radius",
                "numerical", {"fitted_R1": [row["fitted"] for row in unit], "max_normalized_error": fit_error},
                {"generator": _generator("exact helix chord sampled at s in R*[0.02, 0.3]", None,
                                         angles_deg=list(ANGLES_DEG), radii=[1.0, 0.1]),
                 "checks": [_check("|fitted - cos^4(alpha)/(24R^2)| * 24 R^2", fit_error, 1e-9)]},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Geodesics integrated on ciw.lab.surfaces.Cylinder reproduce the exact helix chord and coefficient",
                "numerical", {"max_relative_chord_error": integrated_error, "max_normalized_fit_error": integrated_fit},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Cylinder(R)", None, radii=[1.0, 0.1],
                                         steps=60),
                 "checks": [_check("|c_integrated - c_exact| / R", integrated_error, 1e-13),
                            _check("fitted coefficient from integrated chords, normalized", integrated_fit, 1e-8)]},
                tolerance={"abs": 1e-8, "rel": 0}),
        finding("Axial rulings (alpha = 90 deg) have zero chord correction; circumferential paths have the largest "
                "coefficient 1/(24 R^2)", "numerical",
                {"ruling_max_deficit_over_R": ruling_deficit, "circumferential_coefficient_R1": circle_coefficient},
                {"checks": [_check("max (s - c)/R along integrated rulings", ruling_deficit, 1e-15, kind="invariant"),
                            _check("largest coefficient over angles minus circumferential coefficient", dominance, 0.0,
                                   "le", kind="invariant"),
                            _check("circumferential coefficient relative to 1/24", abs(circle_coefficient * 24 - 1),
                                   1e-9)]},
                tolerance={"abs": 1e-12, "rel": 1e-9},
                counterexample={"statement": "The chord-versus-geodesic correction is a property of the surface "
                                             "alone, independent of path direction",
                                "witness": {"surface": "cylinder", "alpha_deg_zero_correction": 90,
                                            "alpha_deg_max_correction": 0,
                                            "coefficient_at_0_deg_R1": circle_coefficient}}),
        finding("Zero Gaussian curvature does not make chord and geodesic distance equal", "numerical",
                {"gaussian_curvature": 0.0, "s": witness_s, "s_minus_c": witness_gap},
                {"checks": [_check("s - c on the circumferential geodesic of Cylinder(1) at s = 0.1", witness_gap,
                                   1e-5, "ge", kind="analytic")]},
                tolerance={"abs": 1e-15, "rel": 1e-9},
                counterexample={"statement": "On an intrinsically flat surface (K = 0) a chord equals the geodesic "
                                             "distance",
                                "witness": {"surface": "cylinder R = 1", "alpha_deg": 0, "s": witness_s,
                                            "chord": witness_s - witness_gap}}),
        _unestablished("Chords measured between physical markers on a cylindrical part follow the "
                       "cos^4(alpha)/(24 R^2) coefficient", "physical",
                       "No part or marker was measured; radius tolerance, marker height and path deviation are unknown."),
    ]
    fields = _fields(
        hypothesis="On a cylinder of radius R, the geodesic at angle alpha from the circumferential direction has "
                   "chord deficit s - c = cos^4(alpha) s^3/(24 R^2) + O(s^5): largest circumferentially, zero on "
                   "rulings, although the Gaussian curvature is zero everywhere.",
        mathematical_model="Helix X(s) = (R cos(s cos(a)/R), R sin(s cos(a)/R), s sin(a)); kappa = cos^2(a)/R, "
                           "tau = sin(a) cos(a)/R; c^2 = (2R sin(s cos(a)/2R))^2 + (s sin(a))^2 = s^2 - "
                           "cos^4(a) s^4/(12R^2) + cos^6(a) s^6/(360 R^4) + O(s^8).",
        input_data=[f"alpha in {list(ANGLES_DEG)} deg", "R = 1 and R = 0.1 (the camera cylinder)",
                    "40 geometric samples s in R*[0.02, 0.3]; 60-step RK4 geodesics over s in [0, 0.3R]"],
        observation_model="Noise-free chords of embedded points; no instrument.",
        expected_invariant="Fitted coefficient = cos^4(alpha)/(24 R^2) independent of the data source (exact "
                           "formula or integrated geodesic).",
        experiment="Series expansion (hand and sympy), least-squares fit of (s - c)/s^3 = a + b s^2 + d s^4 across "
                   "angles and radii, integration on Cylinder with ciw.lab.jacobi, counterexample search over "
                   "direction and Gaussian curvature.",
        numerical_result=f"max normalized fit error {fit_error:.1e}; integrated chord error {integrated_error:.1e} R; "
                         f"circumferential coefficient {circle_coefficient:.12f} (1/24 = {1 / 24:.12f}); ruling deficit "
                         f"{ruling_deficit:.1e} R"
                         + ("" if symbolic is None else f"; sympy max difference {symbolic['max_abs_difference']:.1e}"),
        uncertainty="Fit truncation and rounding below 1e-9 normalized; the cylinder metric is constant, so RK4 "
                    "integrates helices up to rounding.",
        failure_modes_checked=["alpha = 90 deg (degenerate cos(alpha) = 0)", "small radius R = 0.1",
                               "rounding amplification of (s - c)/s^3 at small s", "sympy unavailable"],
        unresolved_assumptions=["The part is an exact circular cylinder with a known axis",
                                "Markers lie exactly on one geodesic"],
        recommended_next_task="T048: generate synthetic stereo-camera measurements of markers on the cylinder")
    return {"state": "completed", "fields": fields, "findings": findings}
