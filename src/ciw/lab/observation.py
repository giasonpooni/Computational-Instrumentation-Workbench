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
from .evidence import finding, holds as compare
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
    holds = compare(observed, tolerance, comparison)
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


def _independent(reference, observed, tolerance, checker, revision, kind="exact_arithmetic") -> dict:
    return dict(_check(reference, observed, tolerance, kind=kind), producer=dict(PRODUCER),
                checker={"implementation": checker, "revision": revision})


# Per-finding uncertainty (AUTHORING rule 5). ``value`` is a bound or a 95 %
# half-width in the finding's unit, a list or dict of them matching the value.
def _uncertainty(kind, value, basis) -> dict:
    return {"kind": kind, "value": value, "basis": basis}


def _exact(basis="exact: integer counts, refusal codes or dyadic arithmetic; no rounding") -> dict:
    return _uncertainty("exact", 0.0, basis)


def _roundoff(value, basis) -> dict:
    return _uncertainty("roundoff", float(value), basis)


def _truncation(value, basis) -> dict:
    return _uncertainty("truncation_bound", value, basis)


def _mc95(value, basis) -> dict:
    return _uncertainty("monte_carlo_95ci", value, basis)


Z95 = sig.normal_quantile(0.975)


def _half_width(standard_error) -> float:
    """Two-sided 95 % half-width from a standard error."""
    return float(Z95 * standard_error)


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
CONVERSION_ARCS_M = (0.015, 0.06, 0.12)


@task("T045", changed_files=(MODULE, MODES_FILE, CHORD_FILE, DOC), regression_tests=_tests("test_t045_modes_and_refusals"))
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

    # Forward model independent of the inverted helix_chord: embedded points of
    # Cylinder.exact_geodesic (straight lines in the (phi, z) chart).
    conversions, worst = [], 0.0
    for degrees in (0.0, 30.0, 60.0, 90.0):
        model = dict(CYLINDER_MODEL, path_angle_rad=math.radians(degrees))
        embedded = chord.embedded_helix_chords(CONVERSION_ARCS_M, 0.1, math.radians(degrees))
        for arc, measured in zip(CONVERSION_ARCS_M, embedded):
            measured = float(measured)
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
        _check("recovered arc length against the arc length of Cylinder.exact_geodesic embeddings (m)", worst,
               1e-13, kind="analytic")]

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
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("Validation refuses records lacking frame, clock, epoch, calibration or clock-basis references",
                "computational_pipeline", [check["observed_refusal"] for check in record_checks],
                {"checks": record_checks}, uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("No observation mode stands in for another: every ordered substitution is refused",
                "computational_pipeline", {"ordered_pairs": len(substitutions), "refused": refused},
                {"checks": [_check("ordered pairs not refused as mode_substitution", len(substitutions) - refused, 0,
                                   kind="exact_arithmetic")] + conversion_checks[1:3]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("A camera chord becomes a surface distance only through a declared surface model",
                "numerical", worst, {"checks": [conversion_checks[0], conversion_checks[3]]}, unit="m",
                uncertainty=_roundoff(1e-15, "a few ulp of the 0.12 m chord amplified by 1/(dc/ds) <= 1.22; the "
                                      "bisection converges to adjacent doubles"),
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
                    "Chords of Cylinder.exact_geodesic embeddings on a declared cylinder R = 0.1 m at alpha = 0, "
                    "30, 60, 90 deg and s = 0.015, 0.06, 0.12 m"],
        observation_model="No instrument: records are generated with declared (synthetic) calibration references.",
        expected_invariant="Every well-formed record validates; every record with a missing reference or a "
                           "substituted mode is refused with a stable code; conversion needs a declared model.",
        experiment="Validate the registry declarations; strip each reference from a valid record; attempt all 42 "
                   "ordered mode substitutions; convert chords to arc lengths with and without a surface model.",
        numerical_result=f"{len(registry)} modes, {len(missing)} missing declarations; {len(record_checks)} record "
                         f"refusals as expected; {refused}/{len(substitutions)} substitutions refused; chord-to-arc "
                         f"inversion error {worst:.2e} m.",
        uncertainty="Refusals and counts are exact; the chord inversion is a bisection converged to adjacent "
                    "doubles, so the recovered arc carries rounding of about 1e-16 m.",
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
              "c5": "(3 kappa0^4 + 8 kappa0^2 tau0^2 - 72 kappa0 kappa0'' - 64 kappa0'^2)/5760",
              "c5_constant_curvature_torsion": "kappa^4/1920 + kappa^2*tau^2/720"}
    basis = {"derivation": f"{DOC}#chord-versus-geodesic-correction"}
    if ctx.available("module:sympy"):
        symbolic = chord.sympy_versus_closed_form()
        # sympy expands a Frenet-Serret Taylor recursion that ciw.lab.observation_chord sets up; the closed
        # form is the hand derivation. The residuals are simplified as polynomials, not sampled.
        basis["independent_check"] = _independent(
            "sympy series of |gamma(s) - gamma(0)| from the Frenet-Serret Taylor recursion (set up in "
            "ciw.lab.observation_chord.sympy_general_series) against the closed-form c1..c5 and the constant-"
            "curvature c5: number of residuals not simplifying to 0 as polynomials in kappa0, kappa0', kappa0'', "
            "tau0", symbolic["nonzero_residuals"], 0.0, "sympy", symbolic["sympy"])
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
    c3_truncation = max(row["c3_relative_fit_truncation"] for row in rows)
    torus_generator = _generator("ciw.lab.jacobi.transfer RK4 on Torus(2, 1)", None, u0=torus["u0"],
                                 heading_rad=torus["heading_rad"], steps=torus["steps"])
    findings = [
        finding("Chord expansion c = s - kappa0^2 s^3/24 - kappa0 kappa0' s^4/24 + c5 s^5 + O(s^6) with "
                "c5 = (3 kappa0^4 + 8 kappa0^2 tau0^2 - 72 kappa0 kappa0'' - 64 kappa0'^2)/5760, which is "
                "kappa^4/1920 + kappa^2 tau^2/720 for constant curvature and torsion", "mathematical",
                closed, basis, uncertainty=_exact("closed-form coefficients; the symbolic comparison is exact"),
                tolerance={"abs": 0, "rel": 0}),
        finding("RK4 sphere geodesics reproduce the chord 2R sin(s/2R) and the s^3 coefficient kappa^2/24",
                "numerical", {"max_relative_chord_error": chord_error, "max_c3_relative_error": c3_error},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Sphere(R)", None, radii=[0.5, 1.0, 2.0],
                                         steps=400),
                 "checks": [_check("|c_integrated - 2R sin(s/2R)| / R", chord_error, 1e-12),
                            _check("fitted (s - c)/s^3 intercept relative to kappa^2/24", c3_error, 1e-6)]},
                uncertainty=_truncation({"max_c3_relative_error": c3_truncation},
                                        "change of the fitted intercept when one more s^2 term is fitted; the chord "
                                        "error is a direct comparison with the closed form (rounding about 1e-16 R)"),
                tolerance={"abs": 1e-9, "rel": 0}),
        finding("A torus geodesic with varying curvature keeps the s^4 term kappa0 kappa0'/24 at the start point",
                "numerical",
                {"kappa0": torus["kappa0"], "kappa0_prime": torus["kappa0_prime"],
                 "fitted_s4": torus["fitted_start_s4"], "predicted_s4": torus["predicted_c4_residual"]},
                {"generator": torus_generator,
                 "checks": [_check("fitted s^4 coefficient relative to kappa0 kappa0'/24", torus["start_relative_error"],
                                   1e-4),
                            _check("|kappa0 kappa0'/24| is resolved away from zero", abs(torus["predicted_c4_residual"]),
                                   1e-3, "ge")]},
                uncertainty=_truncation({"fitted_s4": torus["start_s4_fit_truncation"],
                                         "kappa0_prime": torus["kappa0_prime_fit_truncation"]},
                                        "change when one more polynomial term is fitted (s^4 remainder model, "
                                        "degree-8 fit of exact normal-curvature samples)"),
                tolerance={"abs": 1e-12, "rel": 1e-5},
                counterexample={"statement": "c = s - kappa(0)^2 s^3/24 + O(s^5) along every surface geodesic",
                                "witness": {"surface": "torus major 2, minor 1", "u0": torus["u0"],
                                            "heading_rad": torus["heading_rad"], "kappa0": torus["kappa0"],
                                            "kappa0_prime": torus["kappa0_prime"],
                                            "s4_coefficient": torus["fitted_start_s4"]}}),
        finding("Evaluating the curvature at the arc midpoint removes the s^4 term on the same torus geodesic",
                "numerical", {"midpoint_ratio": torus["midpoint_ratio"]},
                {"generator": torus_generator,
                 "checks": [_check("midpoint-curvature s^4 coefficient relative to kappa0 kappa0'/24",
                                   torus["midpoint_ratio"], 1e-4)]},
                uncertainty=_truncation(torus["midpoint_ratio_fit_truncation"],
                                        "the exact midpoint s^4 coefficient is 0; the fitted ratio is fit truncation "
                                        "and rounding amplified by 1/s^4, so only the bound is regression-tested"),
                tolerance={"abs": 1e-5, "rel": 0}),
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
                uncertainty=_truncation({"fitted_s5_gap": helix["s5_fit_truncation"]},
                                        "change of the fitted s^5 coefficient when one more s^2 term is fitted"),
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
                   "kappa the absolute normal curvature; the s^4 term -kappa kappa' s^4/24 vanishes only when "
                   "kappa kappa' = 0, and the circle formula 2 sin(kappa s/2)/kappa additionally needs zero torsion.",
        mathematical_model="Taylor expansion of gamma(s) - gamma(0) in the Frenet frame (T' = kappa N, "
                           "N' = -kappa T + tau B, B' = -tau N): |delta|^2 = s^2 - kappa^2 s^4/12 - kappa kappa' s^5/12 "
                           "+ O(s^6), so c = s - kappa^2 s^3/24 - kappa kappa' s^4/24 + c5 s^5 with c5 = (3 kappa^4 + "
                           "8 kappa^2 tau^2 - 72 kappa kappa'' - 64 kappa'^2)/5760. For a geodesic k_g = 0, hence "
                           "kappa = |II(T, T)|. For a circle (tau = 0) c = 2 sin(kappa s/2)/kappa exactly.",
        input_data=["Sphere radii 0.5, 1, 2 (RK4, 400 steps over s in [0, R])",
                    "Torus major 2, minor 1, start (0, pi/4), heading 0.6 rad, 800 RK4 steps over s in [0, 0.4]",
                    "Cylinder R = 1 helix at 45 deg, 160 steps over s in [0, 0.8]",
                    "Symbols kappa0, kappa0', kappa0'', tau0 for the symbolic sympy comparison"],
        observation_model="Chords are Euclidean distances between embedded points of integrated geodesics; no "
                          "instrument noise.",
        expected_invariant="Integrated chords agree with closed forms to integration accuracy; fitted coefficients "
                           "agree with the derived ones.",
        experiment="Derive the series by hand and (when sympy is present) expand the Frenet-Serret recursion with "
                   "sympy and simplify the coefficient residuals symbolically; integrate geodesics with "
                   "ciw.lab.jacobi; fit polynomial models of (s - c)/s^3 and of the remainders; search for "
                   "counterexamples to the O(s^5) remainder and to the constant-curvature circle formula.",
        numerical_result=f"sphere chord error {chord_error:.1e} R, c3 relative error {c3_error:.1e}; torus s^4 "
                         f"coefficient {torus['fitted_start_s4']:.6e} vs predicted {torus['predicted_c4_residual']:.6e} "
                         f"(midpoint ratio {torus['midpoint_ratio']:.1e}); helix s^5 gap {helix['fitted_s5_gap']:.6e} vs "
                         f"{helix['predicted_s5_gap']:.6e}; the symbolic sympy comparison, when sympy is available, "
                         "is recorded on the derivation finding (independent_check).",
        uncertainty=f"Fit truncation, estimated by fitting one more term: c3 {c3_truncation:.1e} relative (sphere), "
                    f"s^4 {torus['start_s4_fit_truncation']:.1e} and kappa0' {torus['kappa0_prime_fit_truncation']:.1e} "
                    f"(torus), s^5 {helix['s5_fit_truncation']:.1e} (helix); RK4 chord error below 1e-12 R. The "
                    "torus midpoint ratio is zero within its fit truncation, so only its bound is regression-tested.",
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
    if ctx.available("module:sympy"):
        symbolic = chord.sympy_cylinder_series(ANGLES_DEG)
        observed = symbolic["max_abs_difference"] if symbolic["symbolic_zero"] else 1.0
        basis["independent_check"] = _independent(
            "sympy series of sqrt((2R sin(s cos(a)/2R))^2 + (s sin(a))^2) against the closed-form c3 and c5 "
            "(symbolic residual zero; 30-digit evaluation at 7 angles)", observed, 1e-14, "sympy", symbolic["sympy"],
            kind="high_precision")
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
    fit_truncation = max(row["normalized_fit_truncation"] for row in fits)
    integrated_error = max(row["max_chord_error"] / row["radius"] for row in integrated)
    integrated_fit = max(row["normalized_error"] for row in integrated)
    integrated_truncation = max(row["normalized_fit_truncation"] for row in integrated)
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
                uncertainty=_exact("closed-form coefficients; the symbolic residuals are exactly zero"),
                tolerance={"abs": 0, "rel": 0}),
        finding("Small-s fits of exact helix chords recover cos^4(alpha)/(24 R^2) at every angle and radius",
                "numerical", {"fitted_R1": [row["fitted"] for row in unit], "max_normalized_error": fit_error},
                {"generator": _generator("exact helix chord sampled at s in R*[0.02, 0.3]", None,
                                         angles_deg=list(ANGLES_DEG), radii=[1.0, 0.1]),
                 "checks": [_check("|fitted - cos^4(alpha)/(24R^2)| * 24 R^2", fit_error, 1e-9)]},
                uncertainty=_truncation({"max_normalized_error": fit_truncation},
                                        "change of the normalized intercept when an s^6 term is added to the fit"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Geodesics integrated on ciw.lab.surfaces.Cylinder reproduce the exact helix chord and coefficient",
                "numerical", {"max_relative_chord_error": integrated_error, "max_normalized_fit_error": integrated_fit},
                {"generator": _generator("ciw.lab.jacobi.transfer RK4 on Cylinder(R)", None, radii=[1.0, 0.1],
                                         steps=60),
                 "checks": [_check("|c_integrated - c_exact| / R", integrated_error, 1e-13),
                            _check("fitted coefficient from integrated chords, normalized", integrated_fit, 1e-8)]},
                uncertainty=_truncation({"max_normalized_fit_error": integrated_truncation,
                                         "max_relative_chord_error": 1e-15},
                                        "fit: change of the normalized intercept when an s^6 term is added; chord: "
                                        "rounding of embedded points (the constant chart metric makes RK4 exact up to "
                                        "rounding)"),
                tolerance={"abs": 1e-8, "rel": 0}),
        finding("Axial rulings (alpha = 90 deg) have zero chord correction; circumferential paths have the largest "
                "coefficient 1/(24 R^2)", "numerical",
                {"ruling_max_deficit_over_R": ruling_deficit, "circumferential_coefficient_R1": circle_coefficient},
                {"checks": [_check("max (s - c)/R along integrated rulings", ruling_deficit, 1e-15, kind="invariant"),
                            _check("largest coefficient over angles minus circumferential coefficient", dominance, 0.0,
                                   "le", kind="invariant"),
                            _check("circumferential coefficient relative to 1/24", abs(circle_coefficient * 24 - 1),
                                   1e-9)]},
                uncertainty=_truncation({"ruling_max_deficit_over_R": 1e-15,
                                         "circumferential_coefficient_R1": fit_truncation / 24},
                                        "ruling deficit: rounding of |z(s) - z(0)|; coefficient: fit truncation"),
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
                uncertainty=_roundoff(1e-16, "closed-form chord 2 sin(s/2) evaluated in double precision"),
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
                         f"{ruling_deficit:.1e} R; the sympy series comparison, when sympy is available, is recorded "
                         "on the derivation finding (independent_check).",
        uncertainty=f"Fit truncation, estimated by adding an s^6 term, is {fit_truncation:.1e} normalized for "
                    f"exact-chord fits and {integrated_truncation:.1e} for integrated-geodesic fits (checked below "
                    "1e-9 and 1e-8); the cylinder metric is constant, so RK4 integrates helices up to rounding.",
        failure_modes_checked=["alpha = 90 deg (degenerate cos(alpha) = 0)", "small radius R = 0.1",
                               "rounding amplification of (s - c)/s^3 at small s", "sympy unavailable"],
        unresolved_assumptions=["The part is an exact circular cylinder with a known axis",
                                "Markers lie exactly on one geodesic"],
        recommended_next_task="T048: generate synthetic stereo-camera measurements of markers on the cylinder")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- camera scene
HELIX_DEG = (0.0, 45.0, 90.0)
ARCS_M = np.linspace(0.0, 0.12, 9)
PIXEL_SIGMA_PX = 0.25


def camera_scene() -> dict:
    """Declared rig and markers on three cylinder geodesics; pairs (0, k) along each helix."""
    cameras = cam.stereo_rig()
    markers = [cam.helix_markers(math.radians(degrees), ARCS_M) for degrees in HELIX_DEG]
    points = np.vstack([m["world"] for m in markers])
    normals = np.vstack([m["normals"] for m in markers])
    n = len(ARCS_M)
    pairs = np.array([(n * i, n * i + k) for i in range(len(HELIX_DEG)) for k in range(1, n)])
    visible = np.ones(len(points), dtype=bool)
    for camera in cameras:
        visible &= cam.visible(camera, points, normals)
    return {"cameras": cameras, "points": points, "normals": normals, "pairs": pairs,
            "arcs": np.array([ARCS_M[k] for _ in HELIX_DEG for k in range(1, n)]),
            "alphas_deg": np.array([degrees for degrees in HELIX_DEG for _ in range(1, n)]),
            "truth": cam.pair_chords(points, pairs), "visible": int(visible.sum())}


def _scene_inputs(scene):
    return [f"Stereo rig {cam.RIG}", f"Cylinder {cam.CYLINDER}",
            f"Markers at s = {_floats(ARCS_M)} m along helices alpha = {list(HELIX_DEG)} deg "
            f"({scene['visible']}/{len(scene['points'])} visible in both images)"]


# --------------------------------------------------------------- T048
def helix_chord_slope(s, radius, alpha):
    """dc/ds of the exact helix chord, used to propagate chord noise to converted arc lengths."""
    theta = s * math.cos(alpha) / radius
    return (radius * math.sin(theta) * math.cos(alpha) + s * math.sin(alpha) ** 2) / float(
        chord.helix_chord(s, radius, alpha))


@task("T048", changed_files=(MODULE, CAMERA_FILE, CHORD_FILE, MODES_FILE, DOC),
      regression_tests=_tests("test_t048_synthetic_camera"))
def synthetic_camera_measurements(ctx):
    scene = ctx.memo("observation:camera-scene", camera_scene)
    cameras, points, pairs, truth = scene["cameras"], scene["points"], scene["pairs"], scene["truth"]
    left, right = cameras
    pixels = (left.project(points), right.project(points))
    dlt, midpoint = cam.triangulate(cameras, pixels), cam.triangulate_midpoint(cameras, pixels)
    measured = cam.pair_chords(dlt, pairs)
    chord_error = float(np.max(np.abs(measured - truth)))
    method_gap = float(np.max(np.abs(dlt - midpoint)))
    alphas = np.radians(scene["alphas_deg"])
    radius = cam.CYLINDER["radius_m"]
    predicted_bias = np.array([arc - float(chord.helix_chord(arc, radius, a)) for arc, a in zip(scene["arcs"], alphas)])
    bias = scene["arcs"] - measured
    bias_mismatch = float(np.max(np.abs(bias - predicted_bias)))
    conversion_error = 0.0
    for value, arc, alpha in zip(measured, scene["arcs"], alphas):
        record = example_observation("camera_chord_distance", value=float(value))
        model = dict(CYLINDER_MODEL, path_angle_rad=float(alpha))
        conversion_error = max(conversion_error, abs(om.chord_to_surface_distance(record, model).value[0] - arc))
    seed, trials = 48, 500
    rng = sig.generator(seed)
    noisy = cam.noisy_chords(cameras, points, pairs, PIXEL_SIGMA_PX, trials, rng)
    arcs_back = np.column_stack([chord.arc_from_chord(noisy[:, j], radius, alphas[j]) for j in range(len(pairs))])
    chord_rms = np.sqrt(np.mean((noisy - truth) ** 2, axis=0))
    arc_rms = np.sqrt(np.mean((arcs_back - scene["arcs"]) ** 2, axis=0))
    # First-order prediction: independent coordinate errors of variance sigma^2 + 1/12 through the pixel
    # Jacobian (T051), and through ds/dc = 1/c'(s) for the converted arc.
    jac = cam.chord_pixel_jacobian(cameras, points[pairs[:, 0]], points[pairs[:, 1]])
    chord_variance = (PIXEL_SIGMA_PX ** 2 + 1 / 12) * np.sum(jac ** 2, axis=1)
    slopes = np.array([helix_chord_slope(arc, radius, a) for arc, a in zip(scene["arcs"], alphas)])
    arc_variance = chord_variance / slopes ** 2
    per_helix, propagation_checks = {}, []
    for degrees in HELIX_DEG:
        on = scene["alphas_deg"] == degrees
        entry = {"max_substitution_bias_m": float(np.max(bias[on]))}
        for name, errors, variance in (("chord", noisy[:, on] - truth[on], chord_variance[on]),
                                       ("arc", arcs_back[:, on] - scene["arcs"][on], arc_variance[on])):
            # Average squared error over the helix's pairs within each trial; trials are independent.
            stats = sig.mean_z(np.mean(errors ** 2, axis=1), float(np.mean(variance)))
            rms = math.sqrt(stats["sample_mean"])
            entry[f"rms_{name}_error_m"] = rms
            entry[f"predicted_rms_{name}_error_m"] = math.sqrt(float(np.mean(variance)))
            entry[f"rms_{name}_error_95ci_m"] = _half_width(stats["standard_error"]) / (2 * rms)
            propagation_checks.append(_z_check(f"{name} mean squared error against first-order propagation, "
                                               f"alpha = {degrees:g} deg", stats["z"]))
            entry[f"{name}_z"] = float(stats["z"])
        per_helix[f"{degrees:g}"] = entry
    rows = [{"alpha_deg": float(d), "arc_m": float(a), "true_chord_m": float(c), "noise_free_chord_m": float(m),
             "substitution_bias_m": float(b), "noisy_chord_rms_m": float(cr), "noisy_arc_rms_m": float(ar),
             "predicted_chord_sd_m": float(math.sqrt(cv)), "predicted_arc_sd_m": float(math.sqrt(av))}
            for d, a, c, m, b, cr, ar, cv, av in zip(scene["alphas_deg"], scene["arcs"], truth, measured, bias,
                                                     chord_rms, arc_rms, chord_variance, arc_variance)]
    ctx.artifact_json("camera-measurements.json", {"rig": {"left": left.describe(), "right": right.describe()},
                                                   "pairs": rows, "per_helix": per_helix,
                                                   "noise": {"pixel_sigma_px": PIXEL_SIGMA_PX, "rounding": "integer",
                                                             "grid_phase": "uniform, independent per marker, camera "
                                                                           "and axis", "trials": trials, "seed": seed}})
    ctx.artifact_text("substitution-bias-vs-noise.svg", svg.line_plot(
        [(f"bias alpha={d:g}", scene["arcs"][scene["alphas_deg"] == d], np.maximum(bias[scene["alphas_deg"] == d], 1e-12))
         for d in HELIX_DEG[:2]]
        + [(f"noise RMS alpha={d:g}", scene["arcs"][scene["alphas_deg"] == d], chord_rms[scene["alphas_deg"] == d])
           for d in HELIX_DEG[:2]],
        title="Chord-for-geodesic substitution bias versus chord noise", xlabel="arc length s (m)",
        ylabel="metres", logy=True))
    worst_bias = float(np.max(bias))
    bias_by_helix = [per_helix[f"{d:g}"]["max_substitution_bias_m"] for d in HELIX_DEG]
    ruling_bias = float(np.max(np.abs(bias[scene["alphas_deg"] == 90.0])))
    circumferential = scene["alphas_deg"] == 0.0
    long_circumferential = int(np.argmax(np.where(circumferential, scene["arcs"], -1.0)))
    bias_to_noise = float(bias[long_circumferential] / chord_rms[long_circumferential])
    long_stats = sig.mean_z((noisy[:, long_circumferential] - truth[long_circumferential]) ** 2, 0.0)
    ratio_95ci = bias_to_noise * _half_width(long_stats["standard_error"]) / (2 * long_stats["sample_mean"])
    noise_rms = float(np.sqrt(np.mean(chord_rms ** 2)))
    findings = [
        finding("Noise-free triangulation of the synthetic rig reproduces ground-truth marker chords",
                "numerical", {"max_chord_error_m": chord_error, "dlt_midpoint_gap_m": method_gap},
                {"generator": _generator("declared stereo rig and cylinder markers", None, pairs=len(pairs)),
                 "checks": [_check("|triangulated chord - true chord| (m)", chord_error, 1e-12),
                            _check("DLT against ray-midpoint triangulation (m)", method_gap, 1e-12,
                                   kind="cross_implementation")]},
                unit="m", uncertainty=_roundoff(1e-15, "double-precision projection and triangulation of 0.6 m "
                                                "coordinates (a few ulp)"),
                tolerance={"abs": 1e-11, "rel": 0}),
        finding("Using the camera chord as geodesic distance underestimates it by exactly s - c(s); the largest "
                "bias is on the circumferential helix", "numerical", worst_bias,
                {"checks": [_check("measured bias against s - helix_chord(s) from T047 (m)", bias_mismatch, 1e-12),
                            _check("largest bias over all helices minus largest circumferential bias (m)",
                                   worst_bias - bias_by_helix[0], 0.0, kind="invariant"),
                            _check("largest 45 deg bias minus largest circumferential bias (m)",
                                   bias_by_helix[1] - bias_by_helix[0], 0.0, "signed_le", kind="invariant"),
                            _check("largest ruling bias minus largest 45 deg bias (m)",
                                   bias_by_helix[2] - bias_by_helix[1], 0.0, "signed_le", kind="invariant")]},
                unit="m", uncertainty=_roundoff(1e-15, "noise-free triangulation rounding (a few ulp)"),
                tolerance={"abs": 1e-12, "rel": 1e-9}),
        finding("The declared cylinder model converts noise-free chords to arc lengths", "numerical",
                conversion_error, {"checks": [_check("|converted arc - true arc| (m)", conversion_error, 1e-11)]},
                unit="m", uncertainty=_roundoff(1e-15, "triangulation rounding amplified by ds/dc <= 1.22"),
                tolerance={"abs": 1e-11, "rel": 0}),
        finding("Synthetic chord and model-converted arc RMS errors under 0.25 px Gaussian noise with integer "
                "rounding match first-order propagation", "numerical",
                {"rms_chord_error_m": noise_rms, "per_helix": per_helix},
                {"generator": _generator("ciw.lab.observation_camera.noisy_chords", seed, trials=trials,
                                         pixel_sigma_px=PIXEL_SIGMA_PX, rounding="integer"),
                 "checks": propagation_checks},
                unit="m", uncertainty=_mc95({name: {"chord": v["rms_chord_error_95ci_m"], "arc": v["rms_arc_error_95ci_m"]}
                                             for name, v in per_helix.items()},
                                            f"95 % half-width of each per-helix RMS from {trials} independent trials"),
                tolerance={"abs": 1e-12, "rel": 1e-4}),
        finding("For the longest circumferential chord the substitution bias exceeds the 0.25 px noise RMS more "
                "than tenfold, while on rulings it vanishes", "numerical",
                {"bias_to_noise_ratio": bias_to_noise, "ruling_max_abs_bias_m": ruling_bias},
                {"checks": [_check("circumferential bias over its chord RMS at s = 0.12 m", bias_to_noise, 10.0, "ge"),
                            _check("largest |bias| on the ruling (m)", ruling_bias, 1e-12, kind="invariant")]},
                uncertainty=_mc95({"bias_to_noise_ratio": ratio_95ci},
                                  f"relative 95 % half-width of that pair's chord RMS ({trials} trials) carried to the "
                                  "ratio; the bias is exact"),
                tolerance={"abs": 1e-12, "rel": 1e-4}),
        _unestablished("A physical stereo rig with this geometry achieves the synthetic chord accuracy",
                       "sensor_performance", "No camera was acquired; lens, sensor, marker detection and calibration "
                       "errors of real hardware are not represented."),
    ]
    fields = _fields(
        hypothesis="A declared pinhole stereo pair observing markers on the cylinder recovers chords exactly without "
                   "noise; the chord differs from the geodesic distance by the T047 correction, which a declared "
                   "surface model removes; for long circumferential chords that bias exceeds the error caused by "
                   "0.25 px pixel noise, while along rulings it vanishes.",
        mathematical_model="x_cam = R X + t, pixels u = f x/z + c; linear DLT triangulation on normalized "
                           "coordinates; chord = |X_i - X_j|; arc from chord by inverting the helix chord on its "
                           "monotone branch. Noise: sigma_c^2 = (sigma^2 + 1/12) sum J^2 over the 8 pixel coordinates "
                           "of a pair (T051) and sigma_s = sigma_c / c'(s) for the converted arc.",
        input_data=_scene_inputs(scene) + [f"Pixel noise sigma = {PIXEL_SIGMA_PX} px, integer rounding, uniform "
                                           f"grid phase per marker, camera and axis, {trials} trials, seed {seed}"],
        observation_model="Synthetic images: ideal projection, then Gaussian pixel noise and integer rounding; the "
                          "camera_chord_distance mode (extrinsic).",
        expected_invariant="Noise-free chords equal truth to rounding; substitution bias equals s - c(s) exactly; "
                           "model conversion recovers s; noisy mean squared errors within 99.9 % Monte Carlo bounds of "
                           "first-order propagation.",
        experiment="Project, triangulate (DLT and ray midpoint), compare chords with truth, convert chords through "
                   "typed records and the declared cylinder model, then repeat with seeded noisy pixels and compare "
                   "per-helix mean squared errors with first-order propagation.",
        numerical_result=f"noise-free chord error {chord_error:.1e} m; substitution bias up to {worst_bias:.4e} m "
                         f"(largest circumferential, {bias_by_helix[1]:.3e} m at 45 deg, {ruling_bias:.1e} m on the "
                         f"ruling); conversion error {conversion_error:.1e} m; noisy chord RMS {noise_rms:.3e} m; "
                         f"longest circumferential bias / noise RMS = {bias_to_noise:.1f}.",
        uncertainty=f"Noise-free results are exact to rounding; noisy RMS values carry about "
                    f"{100 / math.sqrt(2 * trials):.1f} % relative Monte Carlo standard error (95 % half-widths per "
                    "helix in the finding).",
        failure_modes_checked=["markers outside the image or back-facing (visibility check)",
                               "DLT versus ray-midpoint disagreement", "chord substituted for geodesic distance",
                               "conversion without a surface model (refused in T045)",
                               "correlated rounding errors (grid phase drawn per marker)"],
        unresolved_assumptions=["Pinhole cameras with perfectly known calibration (perturbed in T049/T050)",
                                "Marker centres are detected without bias; occlusion is not modelled"],
        recommended_next_task="T049: perturb focal length, principal point and extrinsic rotation")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T049
CALIBRATION_DIRECTION = np.array([2.0, 0.5, -0.4, 2e-4, -1e-4, 3e-4])
CALIBRATION_STEPS = np.array([1e-2, 1e-3, 1e-3, 1e-7, 1e-7, 1e-7])
PARAMETERS = ("focal_px", "right_cx_px", "right_cy_px", "right_rx_rad", "right_ry_rad", "right_rz_rad")


def _believed_chords(true_cameras, delta, points, pairs):
    left, right = true_cameras
    believed = cam.believed_rig(true_cameras, delta)
    return cam.pair_chords(cam.triangulate(believed, (left.project(points), right.project(points))), pairs)


def calibration_jacobian(cameras, points, pairs, steps=CALIBRATION_STEPS) -> np.ndarray:
    """Central-difference d chord / d calibration error, one column per parameter."""
    jac = np.empty((len(pairs), len(steps)))
    for j, step in enumerate(steps):
        delta = np.zeros(len(steps))
        delta[j] = step
        jac[:, j] = (_believed_chords(cameras, delta, points, pairs)
                     - _believed_chords(cameras, -delta, points, pairs)) / (2 * step)
    return jac


def rectified_analytic_jacobian(points, pairs) -> np.ndarray:
    """dc/df = dZ^2/(c f) and dc/dcx_right from Z = f b / disparity on the rectified rig."""
    f, b = cam.RIG["focal_px"], cam.RIG["baseline_m"]
    difference = points[pairs[:, 0]] - points[pairs[:, 1]]
    length = np.linalg.norm(difference, axis=1)
    depth = points[:, 2]
    # Rectified left camera coordinates equal world coordinates; dZ/dcx = -Z^2/(b f), X = x Z, Y = y Z.
    moved = np.column_stack([points[:, 0] / depth, points[:, 1] / depth, np.ones(len(points))]) \
        * (-depth ** 2 / (b * f))[:, None]
    d_difference = moved[pairs[:, 0]] - moved[pairs[:, 1]]
    return np.column_stack([difference[:, 2] ** 2 / (length * f),
                            np.einsum("ij,ij->i", difference, d_difference) / length])


@task("T049", changed_files=(MODULE, CAMERA_FILE, DOC), regression_tests=_tests("test_t049_calibration_perturbations"))
def camera_calibration_perturbations(ctx):
    scene = ctx.memo("observation:camera-scene", camera_scene)
    points, pairs = scene["points"], scene["pairs"]
    selected = pairs[[8, 10, 12, 15]]  # alpha = 45 deg pairs spanning s = 0.015 to 0.12 m
    converged = scene["cameras"]
    rectified = cam.stereo_rig(rectified=True)
    rect_visible = all(bool(np.all(cam.visible(c, points, scene["normals"]))) for c in rectified)
    jac_rect = calibration_jacobian(rectified, points, selected)
    analytic = rectified_analytic_jacobian(points, selected)
    analytic_gap = float(max(np.max(np.abs(jac_rect[:, j] - analytic[:, j])) / np.max(np.abs(analytic[:, j]))
                             for j in range(2)))
    jac = calibration_jacobian(converged, points, selected)
    jac_half = calibration_jacobian(converged, points, selected, CALIBRATION_STEPS / 2)
    convergence = float(np.max(np.abs(jac - jac_half) / np.max(np.abs(jac), axis=0)))
    base = _believed_chords(converged, np.zeros(6), points, selected)
    scales = (0.03, 0.1, 0.3, 1.0, 3.0)
    rows = []
    for scale in scales:
        delta = scale * CALIBRATION_DIRECTION
        direct = _believed_chords(converged, delta, points, selected) - base
        predicted = jac @ delta
        rows.append({"scale": scale, "max_direct_error_m": float(np.max(np.abs(direct))),
                     "max_first_order_residual_m": float(np.max(np.abs(direct - predicted))),
                     "relative_residual": float(np.max(np.abs(direct - predicted)) / np.max(np.abs(direct)))})
    slope = sig.loglog_slope(scales, [row["max_first_order_residual_m"] for row in rows])
    declared = next(row for row in rows if row["scale"] == 1.0)
    # Counterexample: a common focal error leaves same-depth (ruling) chords unchanged on the rectified rig.
    focal_error = 5.0
    # The circumferential pair (0, 4) spans a depth change; (0, 8) would be symmetric about phi = 0.
    ruling, circumferential = np.array([[18, 26]]), np.array([[0, 4]])
    ratio = 1 + focal_error / cam.RIG["focal_px"]
    changes = {}
    for name, pair in (("ruling", ruling), ("circumferential", circumferential)):
        before = cam.pair_chords(points, pair)[0]
        after = _believed_chords(rectified, [focal_error, 0, 0, 0, 0, 0], points, pair)[0]
        changes[name] = float(after / before - 1)
    all_pairs = pairs
    d = points[all_pairs[:, 0]] - points[all_pairs[:, 1]]
    exact_model = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + (d[:, 2] * ratio) ** 2)
    model_gap = float(np.max(np.abs(_believed_chords(rectified, [focal_error, 0, 0, 0, 0, 0], points, all_pairs)
                                    - exact_model)))
    sensitivity = {name: _floats(jac[:, j] * (1e-3 if name.endswith("_rad") else 1.0))
                   for j, name in enumerate(PARAMETERS)}
    longest = {name: abs(values[-1]) for name, values in sensitivity.items()}
    # Compare like units only: horizontal against vertical principal point (m/px), yaw against pitch (m/mrad).
    principal_ratio = longest["right_cx_px"] / longest["right_cy_px"]
    rotation_ratio = longest["right_ry_rad"] / longest["right_rx_rad"]
    ctx.artifact_json("calibration-jacobian.json", {
        "pairs_arc_m": _floats(scene["arcs"][[8, 10, 12, 15]]), "parameters": list(PARAMETERS),
        "sensitivity_m_per_px_or_mrad": sensitivity, "rectified_numeric": jac_rect[:, :2].tolist(),
        "rectified_analytic": analytic.tolist(), "linearization": rows,
        "declared_perturbation": dict(zip(PARAMETERS, _floats(CALIBRATION_DIRECTION))), "focal_counterexample": changes})
    ctx.artifact_text("linearization-residual.svg", svg.line_plot(
        [("direct chord error", scales, [r["max_direct_error_m"] for r in rows]),
         ("first-order residual", scales, [r["max_first_order_residual_m"] for r in rows])],
        title="Calibration perturbation: direct error and first-order residual", xlabel="perturbation scale",
        ylabel="metres", logx=True, logy=True))
    findings = [
        finding("The finite-difference chord Jacobian matches the analytic focal-length and principal-point "
                "derivatives on a rectified rig", "numerical", analytic_gap,
                {"checks": [_check("max |numeric - analytic| / max |analytic| per column", analytic_gap, 1e-6)]},
                uncertainty=_uncertainty("reference_error", 1e-15, "closed-form derivatives evaluated in double "
                                         "precision; the recorded gap is the finite-difference error itself"),
                tolerance={"abs": 1e-6, "rel": 0}),
        finding("First-order calibration-error prediction agrees with direct recomputation with a second-order "
                "residual", "numerical",
                {"relative_residual_at_declared": declared["relative_residual"], "residual_slope": slope,
                 "max_direct_error_m_at_declared": declared["max_direct_error_m"]},
                {"generator": _generator("declared calibration perturbation direction", None,
                                         direction=dict(zip(PARAMETERS, _floats(CALIBRATION_DIRECTION)))),
                 "checks": [_check("relative first-order residual at the declared perturbation",
                                   declared["relative_residual"], 0.01),
                            _check("log-log slope of the residual against scale minus 2", slope - 2, 0.05)]},
                uncertainty=_truncation({"relative_residual_at_declared": convergence},
                                        "Jacobian change when the finite-difference steps are halved, relative"),
                tolerance={"abs": 1e-9, "rel": 1e-5}),
        finding("Chord sensitivity to each calibration parameter (m per px, m per mrad); disparity-changing errors "
                "dominate", "numerical", sensitivity,
                {"checks": [_check("Jacobian change when finite-difference steps are halved (relative)", convergence,
                                   1e-5, kind="self_convergence"),
                            _check("0.12 m chord: |d/d cx| over |d/d cy| (both m per px)", principal_ratio, 10.0,
                                   "ge", kind="invariant"),
                            _check("0.12 m chord: |d/d yaw| over |d/d pitch| (both m per mrad)", rotation_ratio, 10.0,
                                   "ge", kind="invariant")]},
                uncertainty=_truncation(convergence, "relative Jacobian change when the finite-difference steps are "
                                        "halved"),
                tolerance={"abs": 1e-12, "rel": 1e-5}),
        finding("A common focal-length error leaves same-depth chords unchanged and scales only the depth "
                "component", "numerical", {"ruling_relative_change": changes["ruling"],
                                           "circumferential_relative_change": changes["circumferential"],
                                           "exact_model_gap_m": model_gap},
                {"checks": [_check("relative change of the ruling chord under a 5 px focal error",
                                   changes["ruling"], 1e-12),
                            _check("relative change of the circumferential chord (resolved)",
                                   abs(changes["circumferential"]), 1e-5, "ge"),
                            _check("chords against sqrt(dX^2 + dY^2 + (dZ f_b/f)^2) (m)", model_gap, 1e-12)]},
                uncertainty=_roundoff(1e-14, "rectified-rig triangulation rounding relative to the chord (a few "
                                      "ulp of 0.6 m coordinates over a 0.02-0.12 m chord)"),
                tolerance={"abs": 1e-12, "rel": 1e-6},
                counterexample={"statement": "A common focal-length error rescales every measured distance by one "
                                             "factor",
                                "witness": {"rig": "rectified", "focal_error_px": focal_error,
                                            "ruling_relative_change": changes["ruling"],
                                            "circumferential_relative_change": changes["circumferential"]}}),
        _unestablished("The declared perturbation magnitudes bound the calibration error of a real stereo rig",
                       "calibration", "No calibration procedure was run; real calibration uncertainty is unknown."),
    ]
    fields = _fields(
        hypothesis="Chord errors caused by small calibration errors are linear in the errors, with a Jacobian that "
                   "matches closed forms where they exist; errors that change horizontal disparity (right-camera yaw, "
                   "horizontal principal point) dominate, while vertical principal-point and pitch errors barely move "
                   "chords.",
        mathematical_model="Pixels from the true rig are triangulated with a believed rig (focal f + df on both "
                           "cameras, right principal point + (dcx, dcy), right rotation exp([w]) about its centre). "
                           "Rectified rig: Z_b = Z f_b/f and X_b = X, so c_b^2 = dX^2 + dY^2 + (dZ f_b/f)^2 and "
                           "dZ/dcx = -Z^2/(b f).",
        input_data=_scene_inputs(scene) + [f"Declared direction {dict(zip(PARAMETERS, _floats(CALIBRATION_DIRECTION)))}",
                                           f"Scales {list(scales)}", f"Finite-difference steps {_floats(CALIBRATION_STEPS)}",
                                           f"Rectified rig markers visible: {rect_visible}"],
        observation_model="Noise-free synthetic pixels; only the calibration used for triangulation is wrong.",
        expected_invariant="Direct error - J delta = O(|delta|^2); exact closed forms on the rectified rig.",
        experiment="Central-difference Jacobian, step-halving convergence, comparison with analytic derivatives, "
                   "direct recomputation over five perturbation scales, and a counterexample search for uniform "
                   "focal scaling.",
        numerical_result=f"analytic gap {analytic_gap:.1e}; relative first-order residual "
                         f"{declared['relative_residual']:.2e} at the declared perturbation (direct error "
                         f"{declared['max_direct_error_m']:.2e} m); residual slope {slope:.3f}; ruling change "
                         f"{changes['ruling']:.1e} vs circumferential {changes['circumferential']:.2e} for df = 5 px; "
                         f"0.12 m chord ratios cx/cy {principal_ratio:.0f} and yaw/pitch {rotation_ratio:.0f}.",
        uncertainty="Finite-difference truncation below 1e-5 relative (step halving); no stochastic component.",
        failure_modes_checked=["nonlinearity at large perturbations (residual slope)", "finite-difference step size",
                               "rotation about the camera centre versus about the world origin",
                               "uniform-scaling assumption for focal errors"],
        unresolved_assumptions=["Calibration errors are static and small", "Distortion is absent (see T050)"],
        recommended_next_task="T050: add perspective and lens-distortion perturbations")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T050
DISTORTION_SHIFTS = ((0.0, 0.0, 0.0), (0.04, -0.03, 0.0), (0.08, -0.06, 0.0), (0.12, -0.09, 0.0), (0.16, -0.12, 0.0))
DISTORTION_K1 = (-0.1, -0.01, 0.01, 0.1)


def _pixel_offsets(cameras, distorted_cameras, points, pairs):
    """Pixel displacement vectors in the (uL_a, vL_a, uR_a, vR_a, uL_b, vL_b, uR_b, vR_b) order."""
    blocks = []
    for index in (pairs[:, 0], pairs[:, 1]):
        for ideal, distorted in zip(cameras, distorted_cameras):
            blocks.append(distorted.project(points[index]) - ideal.project(points[index]))
    return np.concatenate(blocks, axis=1)


DISTORTION_ARCS_M = (0.0, 0.015, 0.03, 0.045, 0.06)
DISTORTION_PAIRS = ((0, 2), (0, 4), (1, 3))
DISTORTION_MODEL = (0.05, 0.01, 5e-4, -5e-4)
FOLD = {"k1": -0.6, "r_true": 0.85, "undistort_iterations": 400}


def _monotone_preimage(k1, distorted_radius):
    """Radius r below the fold 1/sqrt(-3 k1) with r (1 + k1 r^2) = distorted_radius, by bisection."""
    low, high = 0.0, 1 / math.sqrt(-3 * k1)
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if middle * (1 + k1 * middle ** 2) < distorted_radius else (low, middle)
    return 0.5 * (low + high)


def _fraction_beyond(radius, stride):
    """Fraction of pixel centres (sampled every ``stride`` px) farther than ``radius`` from the principal point."""
    (width, height), f = cam.RIG["image_size_px"], cam.RIG["focal_px"]
    (cx, cy) = cam.RIG["principal_point_px"]
    u, v = np.meshgrid(np.arange(0.5, width, stride), np.arange(0.5, height, stride))
    return float(np.mean(np.hypot(u - cx, v - cy) / f > radius))


def distortion_study() -> dict:
    f = cam.RIG["focal_px"]
    # Implementation consistency: on the x-axis the implemented model is x (1 + k1 r^2 + k2 r^4).
    radii = np.geomspace(0.02, 0.4, 12)
    along_x = np.column_stack([radii, np.zeros_like(radii)])
    formula = {}
    for k1 in (1e-3, 1e-2, 1e-1):
        formula[k1] = f * np.linalg.norm(cam.distort(along_x, (k1, 0.0, 0.0, 0.0)) - along_x, axis=1)
    formula_error = max(float(np.max(np.abs(v / (f * k1 * radii ** 3) - 1))) for k1, v in formula.items())
    k1, k2 = 0.05, 0.02
    both = f * np.linalg.norm(cam.distort(along_x, (k1, k2, 0.0, 0.0)) - along_x, axis=1)
    formula_error = max(formula_error, float(np.max(np.abs(both / (f * (k1 * radii ** 3 + k2 * radii ** 5)) - 1))))

    cameras = cam.stereo_rig()
    pairs = np.array(DISTORTION_PAIRS)
    rows, first_order, odd = [], 0.0, 0.0
    visible_all = True
    for shift in DISTORTION_SHIFTS:
        markers = cam.helix_markers(math.radians(45.0), np.array(DISTORTION_ARCS_M), shift=shift)
        points = markers["world"]
        for camera in cameras:
            visible_all &= bool(np.all(cam.visible(camera, points, markers["normals"])))
        truth = cam.pair_chords(points, pairs)
        jac = cam.chord_pixel_jacobian(cameras, points[pairs[:, 0]], points[pairs[:, 1]])
        radius = float(np.mean([np.linalg.norm(c.normalized(c.project(points)), axis=1).mean() for c in cameras]))
        biases = {}
        for k in DISTORTION_K1:
            distorted = tuple(replace(c, distortion=(k, 0.0, 0.0, 0.0)) for c in cameras)
            left, right = distorted
            measured = cam.pair_chords(cam.triangulate(cameras, (left.project(points), right.project(points))), pairs)
            direct = measured - truth
            predicted = np.einsum("ij,ij->i", jac, _pixel_offsets(cameras, distorted, points, pairs))
            first_order = max(first_order, float(np.max(np.abs(direct - predicted) / np.abs(direct))))
            biases[k] = direct
        odd = max(odd, float(np.max(np.abs(biases[0.01] + biases[-0.01]) / np.abs(biases[0.01]))))
        # Linear in k1: slope of log mean |bias| against log |k1| over the four declared values.
        k1_slope = sig.loglog_slope([abs(k) for k in DISTORTION_K1],
                                    [float(np.mean(np.abs(biases[k]))) for k in DISTORTION_K1])
        rows.append({"shift_m": list(shift), "mean_normalized_radius": radius,
                     "bias_m": {f"{k:g}": _floats(v) for k, v in biases.items()},
                     "mean_abs_bias_k1_0.01_m": float(np.mean(np.abs(biases[0.01]))), "k1_slope": k1_slope})
    # Radius law: bias ~ J_pix (3 f k1 r^2 dr) for chords of fixed image extent, so slope 2 in r.
    mean_radius = [row["mean_normalized_radius"] for row in rows]
    growth = [row["mean_abs_bias_k1_0.01_m"] for row in rows]
    radius_slope = sig.loglog_slope(mean_radius, growth)
    pairwise = [math.log(growth[i + 1] / growth[i]) / math.log(mean_radius[i + 1] / mean_radius[i])
                for i in range(len(rows) - 1)]
    k1_slope_error = max(abs(row["k1_slope"] - 1) for row in rows)

    model = DISTORTION_MODEL
    distorted = tuple(replace(c, distortion=model) for c in cameras)
    markers = cam.helix_markers(math.radians(45.0), np.array(DISTORTION_ARCS_M), shift=DISTORTION_SHIFTS[-1])
    points = markers["world"]
    pixels = tuple(c.project(points) for c in distorted)
    truth = cam.pair_chords(points, pairs)
    uncorrected = float(np.max(np.abs(cam.pair_chords(cam.triangulate(cameras, pixels), pairs) - truth)))
    corrected = float(np.max(np.abs(cam.pair_chords(cam.triangulate(cameras, pixels, undistort_model=model), pairs)
                                    - truth)))

    # Fold of the radial model: r (1 + k1 r^2) peaks at r = 1/sqrt(-3 k1), where the distorted radius is
    # (2/3)/sqrt(-3 k1). Pixel coordinates are distorted coordinates, so the image corner is compared with
    # the distorted fold radius; pixels beyond it have no preimage, pixels inside it have two.
    k_fold, r_true = FOLD["k1"], FOLD["r_true"]
    fold = 1 / math.sqrt(-3 * k_fold)
    fold_distorted = fold * (1 + k_fold * fold ** 2)
    corner = math.hypot(*cam.RIG["image_size_px"]) / 2 / f
    target = r_true * (1 + k_fold * r_true ** 2)
    r_other = _monotone_preimage(k_fold, target)
    diagonal = np.array(cam.RIG["image_size_px"], dtype=float) / math.hypot(*cam.RIG["image_size_px"])
    recovered = float(np.linalg.norm(cam.undistort((target * diagonal)[None, :], (k_fold, 0.0, 0.0, 0.0),
                                                   iterations=FOLD["undistort_iterations"])[0]))
    beyond = _fraction_beyond(fold_distorted, 2)
    return {"radii": _floats(radii), "displacement_px": {f"{k:g}": _floats(v) for k, v in formula.items()},
            "formula_relative_error": formula_error,
            "chord_rows": rows, "first_order_relative_residual": first_order, "odd_symmetry_residual": odd,
            "radius_slope": radius_slope, "pairwise_radius_slopes": pairwise, "k1_slope_max_error": k1_slope_error,
            "all_visible": visible_all, "brown_conrady_model": list(model), "uncorrected_bias_m": uncorrected,
            "corrected_error_m": corrected,
            "fold": {"k1": k_fold, "fold_radius_undistorted": fold, "fold_radius_distorted": fold_distorted,
                     "image_corner_radius": corner, "k1_fold_enters_image": -4 / (27 * corner ** 2),
                     "image_fraction_beyond_fold": beyond,
                     "image_fraction_sampling_change": abs(beyond - _fraction_beyond(fold_distorted, 4)),
                     "r_true": r_true, "r_other": r_other, "distorted_radius": target,
                     "distorted_gap": abs(r_other * (1 + k_fold * r_other ** 2) - target),
                     "non_injectivity_gap_px": abs(r_other - r_true) * f,
                     "undistort_iterations": FOLD["undistort_iterations"], "undistorted_radius": recovered,
                     "undistort_to_other_root": abs(recovered - r_other)}}


@task("T050", changed_files=(MODULE, CAMERA_FILE, DOC), regression_tests=_tests("test_t050_lens_distortion"))
def lens_distortion_perturbations(ctx):
    study = distortion_study()
    fold = study["fold"]
    ctx.artifact_json("distortion-study.json", study)
    ctx.artifact_text("radial-displacement.svg", svg.line_plot(
        [(f"k1 = {k}", study["radii"], v) for k, v in study["displacement_px"].items()],
        title="Radial distortion displacement f k1 r^3", xlabel="normalized image radius r",
        ylabel="displacement (px)", logx=True, logy=True))
    rows = study["chord_rows"]
    radius = [r["mean_normalized_radius"] for r in rows]
    ctx.artifact_text("chord-bias-vs-radius.svg", svg.line_plot(
        [("|bias| k1 = 0.01", radius, [r["mean_abs_bias_k1_0.01_m"] for r in rows]),
         ("r^2 law through the first point", radius,
          [rows[0]["mean_abs_bias_k1_0.01_m"] * (x / radius[0]) ** 2 for x in radius])],
        title="Uncorrected radial distortion: chord bias", xlabel="mean normalized image radius",
        ylabel="mean |chord bias| (m)", logx=True, logy=True))
    slope_spread = max(abs(v - study["radius_slope"]) for v in study["pairwise_radius_slopes"])
    findings = [
        finding("The implemented radial term displaces an on-axis image point by f (k1 r^3 + k2 r^5), as the "
                "Brown-Conrady model defines (implementation consistency check)", "numerical",
                {"formula_relative_error": study["formula_relative_error"]},
                {"checks": [_check("|displacement / f(k1 r^3 + k2 r^5) - 1| on the image x-axis (consistency of "
                                   "observation_camera.distort with its defining formula)",
                                   study["formula_relative_error"], 1e-9, kind="invariant")]},
                uncertainty=_roundoff(1e-9, "cancellation in x (1 + k1 r^2) - x at r = 0.02, k1 = 1e-3 (about "
                                      "eps / (k1 r^2))"),
                tolerance={"abs": 1e-9, "rel": 0}),
        finding("Uncorrected radial distortion biases chords as J_pix delta_pix to first order: odd and linear in "
                "k1 and growing as r^2 with mean image radius", "numerical",
                {"first_order_relative_residual": study["first_order_relative_residual"],
                 "odd_symmetry_residual_k1_0.01": study["odd_symmetry_residual"],
                 "radius_slope": study["radius_slope"], "pairwise_radius_slopes": study["pairwise_radius_slopes"],
                 "k1_slope_max_error": study["k1_slope_max_error"],
                 "mean_abs_bias_k1_0.01_m": [r["mean_abs_bias_k1_0.01_m"] for r in rows],
                 "mean_normalized_radius": radius},
                {"generator": _generator("declared rig, 45 deg helix markers shifted across the image", None,
                                         shifts_m=[list(s) for s in DISTORTION_SHIFTS], k1=list(DISTORTION_K1)),
                 "checks": [_check("max |direct - J_pix delta_pix| / |direct| for |k1| <= 0.1",
                                   study["first_order_relative_residual"], 0.03),
                            _check("|bias(k1) + bias(-k1)| / |bias(k1)| at k1 = 0.01", study["odd_symmetry_residual"],
                                   0.01),
                            _check("log-log slope of mean |bias| (k1 = 0.01) against mean image radius minus the "
                                   "predicted 2 (bias ~ J_pix 3 f k1 r^2 dr)", study["radius_slope"] - 2, 0.1),
                            _check("max over shifts of |log-log slope of mean |bias| against |k1| - 1|",
                                   study["k1_slope_max_error"], 0.01),
                            _check("all shifted markers visible in both images", 0 if study["all_visible"] else 1, 0,
                                   kind="invariant")]},
                uncertainty=_truncation({"radius_slope": slope_spread,
                                         "first_order_relative_residual": study["first_order_relative_residual"]},
                                        "radius slope: largest departure of a pairwise slope from the fitted one "
                                        "(r^4 and perspective terms); first order: the O(k1 r^2) residual itself"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Undistorting with the true Brown-Conrady model (radial and tangential) removes the chord bias",
                "numerical",
                {"uncorrected_bias_m": study["uncorrected_bias_m"], "corrected_error_m": study["corrected_error_m"]},
                {"checks": [_check("chord error after undistortion with the generating model (m)",
                                   study["corrected_error_m"], 1e-12)]},
                unit="m", uncertainty=_roundoff(1e-15, "fixed-point undistortion converged to rounding (60 "
                                                "iterations at contraction below 0.1)"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Strong barrel distortion (k1 = -0.6) folds inside the image, so the radial model is not "
                "invertible there", "numerical",
                {"fold_radius_distorted": fold["fold_radius_distorted"],
                 "image_corner_radius": fold["image_corner_radius"],
                 "image_fraction_beyond_fold": fold["image_fraction_beyond_fold"],
                 "r_true": fold["r_true"], "r_other": fold["r_other"],
                 "non_injectivity_gap_px": fold["non_injectivity_gap_px"],
                 "undistort_to_other_root": fold["undistort_to_other_root"]},
                {"checks": [_check("distorted radius at the fold, (2/3)/sqrt(-3 k1), minus the image-corner radius "
                                   "(pixel coordinates are distorted coordinates; negative: fold inside the image)",
                                   fold["fold_radius_distorted"] - fold["image_corner_radius"], 0.0, "signed_le",
                                   kind="invariant"),
                            _check("image fraction beyond the fold circle (pixels without a preimage)",
                                   fold["image_fraction_beyond_fold"], 1e-3, "ge", kind="invariant"),
                            _check("|r_d(r_other) - r_d(r_true)| (two radii, one distorted radius)",
                                   fold["distorted_gap"], 1e-12),
                            _check("non-injectivity gap |r_other - r_true| f (px)", fold["non_injectivity_gap_px"], 1.0,
                                   "ge"),
                            _check(f"converged fixed-point undistortion ({fold['undistort_iterations']} iterations) "
                                   "against the monotone-branch preimage r_other", fold["undistort_to_other_root"],
                                   1e-12)]},
                uncertainty=_truncation({"image_fraction_beyond_fold": fold["image_fraction_sampling_change"],
                                         "r_other": 1e-16},
                                        "fraction: change between 2 px and 4 px pixel sampling; r_other: bisection "
                                        "to adjacent doubles"),
                tolerance={"abs": 1e-9, "rel": 1e-9},
                counterexample={"statement": "The Brown-Conrady radial model can be inverted everywhere in the image",
                                "witness": {"k1": fold["k1"], "r_true": fold["r_true"], "r_other": fold["r_other"],
                                            "distorted_radius": fold["distorted_radius"],
                                            "fold_radius_distorted": fold["fold_radius_distorted"],
                                            "image_corner_radius": fold["image_corner_radius"]}}),
        _unestablished("A two-term radial plus tangential Brown-Conrady model describes a real lens to the required "
                       "accuracy", "calibration", "No lens was measured; higher-order, decentring and "
                       "temperature effects of real optics are unknown."),
    ]
    fields = _fields(
        hypothesis="Uncorrected radial distortion displaces pixels by f(k1 r^3 + k2 r^5), producing chord biases "
                   "that are first order and linear in k1, grow as r^2 with image radius for chords of fixed image "
                   "extent and vanish when the generating model is inverted; strong barrel distortion makes the model "
                   "non-invertible inside the image.",
        mathematical_model="x_d = x (1 + k1 r^2 + k2 r^4) + [2 p1 x y + p2 (r^2 + 2x^2), p1 (r^2 + 2y^2) + 2 p2 x y]; "
                           "chord bias ~ J_pix delta_pix with J_pix = d chord / d pixels; only displacement differences "
                           "(between endpoints, and between cameras through disparity) matter, and for radii differing "
                           "by dr they are f k1 ((r + dr)^3 - r^3) ~ 3 f k1 r^2 dr (slope 2 in r, 1 in k1). r (1 + k1 r^2) peaks "
                           "at r = 1/sqrt(-3 k1) with distorted radius (2/3)/sqrt(-3 k1); the fold enters an image of "
                           "corner radius rho when k1 < -4/(27 rho^2).",
        input_data=[f"Stereo rig {cam.RIG}", f"Cylinder {cam.CYLINDER}",
                    f"{len(DISTORTION_ARCS_M)} markers at s = {list(DISTORTION_ARCS_M)} m on the 45 deg helix, chord "
                    f"pairs {[list(pair) for pair in DISTORTION_PAIRS]}, cylinder shifted by "
                    f"{[list(shift) for shift in DISTORTION_SHIFTS]} m",
                    f"Radial k1 in {list(DISTORTION_K1)} (k2 = p1 = p2 = 0) for the bias law",
                    f"Brown-Conrady model (k1, k2, p1, p2) = {list(DISTORTION_MODEL)} for the correction",
                    "Displacement scan: 12 radii in [0.02, 0.4], k1 in {1e-3, 1e-2, 1e-1} and (k1, k2) = (0.05, 0.02)",
                    f"Fold at k1 = {FOLD['k1']}, point at undistorted radius {FOLD['r_true']} on the image diagonal"],
        observation_model="Noise-free synthetic pixels from distorted cameras triangulated as if undistorted "
                          "(uncorrected) or after fixed-point undistortion with the true model.",
        expected_invariant="Implemented displacement equals f(k1 r^3 + k2 r^5); first-order bias law with slopes 2 "
                           "(radius) and 1 (k1); zero bias after correct undistortion inside the fold radius.",
        experiment="Implementation consistency scan of the displacement; chord bias against the pixel-Jacobian "
                   "prediction and log-log slopes across image radius and k1; correction with the generating model; "
                   "fold counterexample with converged undistortion.",
        numerical_result=f"formula consistency {study['formula_relative_error']:.1e}; first-order residual "
                         f"{study['first_order_relative_residual']:.2%}; bias at k1 = 0.01 grows from "
                         f"{rows[0]['mean_abs_bias_k1_0.01_m']:.2e} to {rows[-1]['mean_abs_bias_k1_0.01_m']:.2e} m, "
                         f"radius slope {study['radius_slope']:.3f} (pairwise "
                         f"{min(study['pairwise_radius_slopes']):.3f}-{max(study['pairwise_radius_slopes']):.3f}), "
                         f"k1 slope within {study['k1_slope_max_error']:.1e} of 1; corrected error "
                         f"{study['corrected_error_m']:.1e} m; fold at distorted r = {fold['fold_radius_distorted']:.3f} "
                         f"< corner {fold['image_corner_radius']:.3f} ({fold['image_fraction_beyond_fold']:.2%} of the "
                         f"image has no preimage); two preimages {fold['non_injectivity_gap_px']:.1f} px apart.",
        uncertainty="Deterministic; the first-order residual is O(k1 r^2) and bounded by the stated tolerance; the "
                    "radius slope departs from 2 by the r^4 and perspective terms (pairwise slopes up to "
                    f"{max(study['pairwise_radius_slopes']):.2f} at the largest radius).",
        failure_modes_checked=["sign of k1 (odd symmetry)", "markers leaving the image at large shifts",
                               "tangential terms", "non-invertible distortion inside the image",
                               "undistorted versus distorted radius when locating the fold",
                               "fixed-point undistortion stopped before convergence"],
        unresolved_assumptions=["Both cameras share one distortion model", "Principal point equals distortion centre"],
        recommended_next_task="T051: add quantization and pixel noise")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T051
def quantization_study(seed=51, samples=200_000) -> dict:
    rng = sig.generator(seed)
    rows = []
    for sigma in (0.0, 0.1, 0.25, 0.5, 1.0):
        truth = rng.uniform(0.0, 1.0, samples) * 37.0  # uniform sub-pixel phase
        error = np.round(truth + sigma * rng.standard_normal(samples)) - truth
        stats = sig.variance_z(error, sigma ** 2 + 1 / 12)
        rows.append({"sigma_px": sigma, **{k: float(v) for k, v in stats.items()}})
    aligned = np.full(samples, 512.0)
    aligned_error = np.round(aligned + 0.1 * rng.standard_normal(samples)) - aligned
    return {"rows": rows, "aligned_sigma_px": 0.1, "aligned_variance": float(aligned_error.var()),
            "aligned_prediction": 0.01 + 1 / 12, "seed": seed, "samples": samples}


@task("T051", changed_files=(MODULE, CAMERA_FILE, SIGNALS_FILE, DOC),
      regression_tests=_tests("test_t051_quantization_noise"))
def quantization_and_pixel_noise(ctx):
    study = quantization_study()
    scene = ctx.memo("observation:camera-scene", camera_scene)
    cameras, points = scene["cameras"], scene["points"]
    pairs = scene["pairs"][[0, 3, 7, 8, 11, 15]]
    jac = cam.chord_pixel_jacobian(cameras, points[pairs[:, 0]], points[pairs[:, 1]])
    variance = PIXEL_SIGMA_PX ** 2 + 1 / 12
    predicted_sd = np.sqrt(variance * np.sum(jac ** 2, axis=1))
    seed, trials = 5100, 4000
    chords = cam.noisy_chords(cameras, points, pairs, PIXEL_SIGMA_PX, trials, sig.generator(seed))
    errors = chords - cam.pair_chords(points, pairs)
    propagation = []
    for j in range(len(pairs)):
        stats = sig.variance_z(errors[:, j], predicted_sd[j] ** 2)
        low = (trials - 1) * stats["sample_variance"] / sig.chi2_quantile(0.9995, trials - 1)
        high = (trials - 1) * stats["sample_variance"] / sig.chi2_quantile(0.0005, trials - 1)
        propagation.append({"arc_m": float(scene["arcs"][[0, 3, 7, 8, 11, 15]][j]),
                            "alpha_deg": float(scene["alphas_deg"][[0, 3, 7, 8, 11, 15]][j]),
                            "predicted_sd_m": float(predicted_sd[j]), "sample_sd_m": math.sqrt(stats["sample_variance"]),
                            "sd_interval_99.9_m": [math.sqrt(low), math.sqrt(high)], "z": float(stats["z"]),
                            "sd_95ci_m": _half_width(stats["standard_error"]) / (2 * math.sqrt(stats["sample_variance"])),
                            "mean_error_z": float(sig.mean_z(errors[:, j], 0.0)["z"])})
    ctx.artifact_json("quantization.json", {"additivity": study, "propagation": propagation,
                                            "pixel_sigma_px": PIXEL_SIGMA_PX, "trials": trials, "seed": seed,
                                            "grid_phase": "uniform, independent per marker, camera and axis"})
    ctx.artifact_text("error-variance-vs-sigma.svg", svg.line_plot(
        [("sample variance", [r["sigma_px"] for r in study["rows"]], [r["sample_variance"] for r in study["rows"]]),
         ("sigma^2 + 1/12", [r["sigma_px"] for r in study["rows"]], [r["predicted_variance"] for r in study["rows"]])],
        title="Rounded noisy pixels: error variance", xlabel="Gaussian sigma (px)", ylabel="variance (px^2)"))
    max_z = max(abs(r["z"]) for r in study["rows"])
    max_prop_z = max(abs(r["z"]) for r in propagation)
    inside = sum(r["sd_interval_99.9_m"][0] <= r["predicted_sd_m"] <= r["sd_interval_99.9_m"][1] for r in propagation)
    findings = [
        finding("Rounding plus Gaussian pixel noise has error variance sigma^2 + 1/12 px^2 under a uniform grid phase",
                "numerical", {"sigma_px": [r["sigma_px"] for r in study["rows"]], "z": [r["z"] for r in study["rows"]]},
                {"generator": _generator("uniform sub-pixel phase, Gaussian noise, integer rounding", study["seed"],
                                         samples=study["samples"]),
                 "checks": [_z_check(f"sample variance against sigma^2 + 1/12 at sigma = {r['sigma_px']}", r["z"])
                            for r in study["rows"]]},
                uncertainty=_mc95({"sample_variance_px2": [_half_width(r["standard_error"]) for r in study["rows"]]},
                                  f"95 % half-width of each sample variance from {study['samples']} samples "
                                  "(fourth-moment standard error)"),
                tolerance={"abs": 1e-6, "rel": 1e-4}),
        finding("Without a random grid phase the quantization variance is not 1/12: an integer-aligned coordinate "
                "with sigma = 0.1 px has almost no rounding error", "numerical",
                {"sample_variance": study["aligned_variance"], "additive_prediction": study["aligned_prediction"]},
                {"checks": [_check("sample variance for an integer-aligned coordinate, sigma = 0.1 px (px^2)",
                                   study["aligned_variance"], 0.01, "le")]},
                uncertainty=_mc95(3 / study["samples"], "one-sided 95 % upper bound (rule of three) on the variance "
                                  "when no rounding flip occurs in the samples; the exact value is 2 Phi(-5) = 5.7e-7 "
                                  "px^2"),
                tolerance={"abs": 1e-6, "rel": 0},
                counterexample={"statement": "Integer pixel rounding always adds 1/12 px^2 to the noise variance",
                                "witness": {"true_coordinate_px": 512.0, "sigma_px": 0.1,
                                            "sample_variance": study["aligned_variance"]}}),
        finding("Linear propagation J (sigma^2 + 1/12) J^T predicts the chord standard deviation", "numerical",
                {"predicted_sd_m": [r["predicted_sd_m"] for r in propagation],
                 "sample_sd_m": [r["sample_sd_m"] for r in propagation], "z": [r["z"] for r in propagation]},
                {"generator": _generator("ciw.lab.observation_camera.noisy_chords", seed, trials=trials,
                                         pixel_sigma_px=PIXEL_SIGMA_PX),
                 "checks": [_z_check(f"chord variance, pair s = {r['arc_m']:.3f} m, alpha = {r['alpha_deg']:g} deg",
                                     r["z"]) for r in propagation]},
                unit="m", uncertainty=_mc95({"sample_sd_m": [r["sd_95ci_m"] for r in propagation]},
                                            f"95 % half-width of each sample SD from {trials} trials; the 99.9 % "
                                            f"chi-square intervals contain the prediction for {inside}/"
                                            f"{len(propagation)} pairs"),
                tolerance={"abs": 1e-12, "rel": 1e-4}),
        _unestablished("Real image noise is Gaussian with sigma = 0.25 px and marker localization rounds to whole "
                       "pixels", "sensor_performance", "No images were acquired; real noise is signal dependent, "
                       "spatially correlated and marker detectors interpolate below one pixel."),
    ]
    fields = _fields(
        hypothesis="Rounding to integer pixels after Gaussian noise gives error variance sigma^2 + 1/12 when the "
                   "true coordinate is uniformly placed on the pixel grid, and chord standard deviations follow "
                   "linear propagation through the triangulation Jacobian.",
        mathematical_model="e = round(x + n) - x = n + q; with frac(x) ~ U(0,1) independent of n, q ~ U(-1/2, 1/2) is "
                           "independent of n, so Var e = sigma^2 + 1/12. sigma_c^2 = (sigma^2 + 1/12) sum_j J_j^2 "
                           "(8 pixel coordinates per pair).",
        input_data=_scene_inputs(scene) + [f"{study['samples']} scalar samples per sigma (seed {study['seed']})",
                                           f"{trials} stereo trials (seed {seed}), sigma = {PIXEL_SIGMA_PX} px"],
        observation_model="Synthetic pixels with a uniform grid phase drawn independently per marker, camera and "
                          "axis, Gaussian noise and integer rounding.",
        expected_invariant="Sample variances within 99.9 % sampling intervals of the predictions.",
        experiment="Scalar Monte Carlo of the additivity law at five sigmas; aligned-coordinate counterexample; "
                   "seeded stereo Monte Carlo against first-order chord propagation.",
        numerical_result=f"additivity max |z| = {max_z:.2f}; aligned variance {study['aligned_variance']:.2e} vs "
                         f"{study['aligned_prediction']:.4f}; chord propagation max |z| = {max_prop_z:.2f}, predicted SD "
                         f"inside the 99.9 % interval for {inside}/{len(propagation)} pairs.",
        uncertainty="Variance z-scores use a fourth-moment standard error; intervals are 99.9 % two-sided "
                    "(chi-square Wilson-Hilferty quantiles).",
        failure_modes_checked=["no dither (integer-aligned coordinate)", "sigma = 0 (pure quantization)",
                               "nonlinearity bias of chords (mean error z recorded)",
                               "correlated rounding from a grid phase shared by all markers"],
        unresolved_assumptions=["Real pixel noise and rounding errors are independent between markers and cameras "
                                "(the generator draws them so; a shared grid phase would correlate them)",
                                "Grid phase is uniform, which a fixed rig and target do not guarantee"],
        recommended_next_task="T052: add encoder bias, scale and backlash")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T052
ENCODER = {"backlash_m": 2e-5, "scale": 5e-4, "bias_m": 3e-5, "noise_m": 5e-7, "samples": 4000, "duration_s": 20.0}


def encoder_study(seed=52) -> dict:
    rng = sig.generator(seed)
    time = np.linspace(0.0, ENCODER["duration_s"], ENCODER["samples"])
    reference = (0.004 * np.sin(2 * math.pi * 0.23 * time) + 0.0025 * np.sin(2 * math.pi * 0.61 * time + 0.4)
                 + 0.0005 * time)
    width = ENCODER["backlash_m"]
    played = sig.backlash(reference, width)
    error = played - reference
    labels = sig.engagement(reference, width)
    steps = np.sign(np.diff(reference))
    reversals = int(np.sum(steps[1:] * steps[:-1] < 0))
    changed = np.abs(np.diff(error)) > 1e-9 * width
    # A step is inside a post-reversal window if it starts in the deadband or ends the take-up.
    window = (labels[:-1] == 0) | (labels[:-1] != labels[1:])
    reading = (1 + ENCODER["scale"]) * played + ENCODER["bias_m"] + ENCODER["noise_m"] * rng.standard_normal(len(time))
    engaged = labels != 0
    design = np.column_stack([reference[engaged], np.ones(engaged.sum()), (labels[engaged] == -1).astype(float)])
    fit = sig.least_squares(design, reading[engaged])
    truth = np.array([1 + ENCODER["scale"], ENCODER["bias_m"], (1 + ENCODER["scale"]) * width])
    sd = np.sqrt(np.diag(fit["covariance"]))
    naive_design = np.column_stack([reference, np.ones(len(time))])
    naive = sig.least_squares(naive_design, reading)
    naive_sd = np.sqrt(np.diag(naive["covariance"]))
    # Noise-free prediction of the naive fit's error: the OLS projection of (1 + s)(play(x) - x) onto [x, 1].
    projection = np.linalg.lstsq(naive_design, (1 + ENCODER["scale"]) * error, rcond=None)[0]
    # Noise contribution to the naive intercept: sigma_noise sqrt([(A^T A)^-1]_11).
    naive_noise_sd = ENCODER["noise_m"] * math.sqrt(np.linalg.inv(naive_design.T @ naive_design)[1, 1])
    return {"error_min_m": float(error.min()), "error_max_m": float(error.max()), "reversals": reversals,
            "error_changes": int(changed.sum()), "changes_outside_windows": int(np.sum(changed & ~window)),
            "rising_engaged_max_m": float(np.max(np.abs(error[labels == 1]))),
            "falling_engaged_offset_m": float(np.max(np.abs(error[labels == -1] - width))),
            "engaged_fraction": float(engaged.mean()), "falling_fraction": float(np.mean(labels == -1)),
            "estimates": _floats(fit["coefficients"]), "truth": _floats(truth), "standard_errors": _floats(sd),
            "z": _floats((fit["coefficients"] - truth) / sd), "residual_sigma_m": fit["residual_sigma"],
            "naive_estimates": _floats(naive["coefficients"]),
            "naive_z": _floats((naive["coefficients"] - truth[:2]) / naive_sd),
            "naive_bias_error_over_backlash": float((naive["coefficients"][1] - truth[1]) / width),
            "naive_predicted_over_backlash": float(projection[1] / width),
            "naive_prediction_relative_error": float((naive["coefficients"][1] - truth[1]) / projection[1] - 1),
            "naive_noise_relative_sd": float(naive_noise_sd / abs(projection[1])),
            "trace": {"time_s": _floats(time[:1200:6]), "error_m": _floats(error[:1200:6])}}


@task("T052", changed_files=(MODULE, SIGNALS_FILE, DOC), regression_tests=_tests("test_t052_encoder_backlash"))
def encoder_bias_scale_backlash(ctx):
    study = encoder_study()
    b = ENCODER["backlash_m"]
    ctx.artifact_json("encoder-study.json", study)
    ctx.artifact_text("backlash-error.svg", svg.line_plot(
        [("played - reference", study["trace"]["time_s"], study["trace"]["error_m"])],
        title="Encoder backlash error (first 6 s)", xlabel="time (s)", ylabel="metres", markers=False))
    names = ("1 + scale", "bias", "(1 + scale) backlash")
    findings = [
        finding("Backlash error stays within [0, b] for a play operator of width b", "numerical",
                {"min_m": study["error_min_m"], "max_m": study["error_max_m"], "backlash_m": b},
                {"generator": _generator("two-tone reference with drift", 52, **ENCODER),
                 "checks": [_check("minimum of played - reference (m)", study["error_min_m"], 0.0, "ge", "invariant"),
                            _check("maximum minus b (m)", study["error_max_m"] - b, 1e-9 * b, "signed_le",
                                   "invariant")]},
                unit="m", uncertainty=_roundoff(1e-20, "rounding of x + b at |x| <= 1 cm (a few ulp)"),
                tolerance={"abs": 1e-15, "rel": 1e-9}),
        finding("Backlash error changes only while the play is taken up after a direction reversal", "numerical",
                {"reversals": study["reversals"], "error_changes": study["error_changes"],
                 "changes_outside_windows": study["changes_outside_windows"]},
                {"checks": [_check("error changes outside post-reversal windows", study["changes_outside_windows"], 0,
                                   kind="invariant"),
                            _check("rising engaged |error| (m)", study["rising_engaged_max_m"], 1e-15, kind="invariant"),
                            _check("falling engaged |error - b| (m)", study["falling_engaged_offset_m"], 1e-15,
                                   kind="invariant")]},
                uncertainty=_exact("integer counts of reversals and error changes"), tolerance={"abs": 0, "rel": 0}),
        finding("Least squares with a direction term recovers scale, bias and backlash", "numerical",
                {"estimates": study["estimates"], "truth": study["truth"], "z": study["z"]},
                {"generator": _generator("encoder reading (1 + scale) play(x) + bias + noise", 52, **ENCODER),
                 "checks": [_z_check(f"{name} estimate against truth", z) for name, z in zip(names, study["z"])]},
                uncertainty=_mc95({"estimates": [_half_width(v) for v in study["standard_errors"]]},
                                  "95 % half-widths from the least-squares standard errors"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Least squares that ignores backlash biases the offset estimate by the projection of the play error "
                "onto [x, 1]", "numerical",
                {"naive_bias_z": study["naive_z"][1], "bias_error_over_backlash": study["naive_bias_error_over_backlash"],
                 "predicted_over_backlash": study["naive_predicted_over_backlash"],
                 "falling_fraction": study["falling_fraction"]},
                {"checks": [_check("|z| of the naive bias estimate exceeds the 99.9 % bound", abs(study["naive_z"][1]),
                                   Z999, "ge"),
                            _check("naive offset error relative to the OLS projection of (1 + s)(play(x) - x) onto "
                                   "[x, 1], minus 1", study["naive_prediction_relative_error"], 0.01)]},
                uncertainty=_mc95({"bias_error_over_backlash": _half_width(study["naive_noise_relative_sd"])
                                   * study["naive_predicted_over_backlash"]},
                                  "95 % half-width of the noise contribution to the naive offset, in units of b"),
                tolerance={"abs": 1e-9, "rel": 1e-6},
                counterexample={"statement": "Fitting reading = a x + c without a direction term gives unbiased "
                                             "scale and bias under backlash",
                                "witness": {"backlash_m": b, "naive_bias_z": study["naive_z"][1],
                                            "bias_error_over_backlash": study["naive_bias_error_over_backlash"]}}),
        _unestablished("A real encoder drive train behaves as a constant-width play operator with constant scale and "
                       "bias", "physical", "No drive was measured; real backlash varies with load, wear, temperature "
                       "and position."),
    ]
    fields = _fields(
        hypothesis="With reading = (1 + scale) play_b(x) + bias + noise, the backlash error lies in [0, b], changes "
                   "only during take-up after reversals, and a direction-aware least-squares fit recovers scale, "
                   "bias and backlash.",
        mathematical_model="Play operator: y_k = max(min(y_{k-1}, x_k + b), x_k) (engaged on the positive flank at "
                           "start); engaged samples: reading = (1 + s) x + beta + (1 + s) b [falling]. A fit on [x, 1] "
                           "alone absorbs the OLS projection of (1 + s)(play(x) - x) onto [x, 1] into its coefficients.",
        input_data=[f"Declared parameters {ENCODER}", "Reference x(t) = 4 mm sin(2 pi 0.23 t) + 2.5 mm "
                    "sin(2 pi 0.61 t + 0.4) + 0.5 mm/s t", "Seed 52"],
        observation_model="Synthetic encoder_displacement readings with Gaussian noise; no quantization "
                          "(studied in T051).",
        expected_invariant="0 <= play - x <= b; error constant on engaged samples; LS estimates within sampling error.",
        experiment="Generate the reference and readings, label engagement by travel since reversal, check the "
                   "invariants, fit with and without a direction term.",
        numerical_result=f"error in [{study['error_min_m']:.1e}, {study['error_max_m']:.4e}] m for b = {b} m; "
                         f"{study['reversals']} reversals, {study['changes_outside_windows']} changes outside take-up; "
                         f"z = {[round(z, 2) for z in study['z']]}; naive bias z = {study['naive_z'][1]:.1f}, offset "
                         f"error {study['naive_bias_error_over_backlash']:.4f} b against the projected "
                         f"{study['naive_predicted_over_backlash']:.4f} b (falling fraction {study['falling_fraction']:.3f}).",
        uncertainty="Estimates carry LS standard errors from the residual variance; checks use the two-sided 99.9 % "
                    "normal bound.",
        failure_modes_checked=["reversal within one sample of engagement", "rounding at the engaged offset",
                               "fitting without a direction term"],
        unresolved_assumptions=["Backlash width, scale and bias are constant", "The reference position is known "
                                "exactly (a real calibration needs an independent reference instrument)"],
        recommended_next_task="T053: add IMU drift and orientation noise")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T053
IMU = {"runs": 2000, "steps": 500, "dt_s": 0.01, "bias_rad_per_s": 1e-3, "arw_rad_per_sqrt_s": 1e-3}
STRAPDOWN = {"bias_rad_per_s": [2e-3, 0.0, 1e-3], "rotation_rad_per_s": [0.0, 0.0, 2 * math.pi],
             "arw_rad_per_sqrt_s": 2e-4, "runs": 200, "steps": 500, "dt_s": 0.01}


def imu_study(seed=53) -> dict:
    rng = sig.generator(seed)
    dt, bias, arw = IMU["dt_s"], IMU["bias_rad_per_s"], IMU["arw_rad_per_sqrt_s"]
    errors = sig.gyro_heading_error(IMU["runs"], IMU["steps"], dt, bias, arw, rng)
    single = []
    for k in (49, 99, 199, 499):
        t = (k + 1) * dt
        sample = errors[:, k]
        moments = (sig.mean_z(sample, bias * t), sig.variance_z(sample, arw ** 2 * t),
                   sig.mean_z(sample ** 2, arw ** 2 * t + (bias * t) ** 2))
        single.append({"t_s": t, "mean_z": float(moments[0]["z"]), "variance_z": float(moments[1]["z"]),
                       "mse": float(np.mean(sample ** 2)), "mse_predicted": arw ** 2 * t + (bias * t) ** 2,
                       "mse_z": float(moments[2]["z"]),
                       "mean_95ci_rad": _half_width(moments[0]["standard_error"]),
                       "variance_95ci_rad2": _half_width(moments[1]["standard_error"]),
                       "mse_95ci_rad2": _half_width(moments[2]["standard_error"])})
    b3 = np.array(STRAPDOWN["bias_rad_per_s"])
    steps, dt3 = STRAPDOWN["steps"], STRAPDOWN["dt_s"]
    cases = {}
    for name, omega in (("stationary", np.zeros(3)), ("rotating", np.array(STRAPDOWN["rotation_rad_per_s"]))):
        exact = sig.orientation_errors(omega, b3, 0.0, 1, steps, dt3, rng)[:, 0, :]
        predicted = sig.bias_error_prediction(omega, b3, steps, dt3)
        transverse = np.linalg.norm(exact[:, :2], axis=1)
        noisy = sig.orientation_errors(omega, b3, STRAPDOWN["arw_rad_per_sqrt_s"], STRAPDOWN["runs"], steps, dt3, rng)
        squared = (noisy - predicted[:, None, :]) ** 2
        n2 = STRAPDOWN["arw_rad_per_sqrt_s"] ** 2
        # Total spread 3 N^2 t and, for isotropy, each body axis separately at N^2 t.
        total = [sig.mean_z(np.sum(squared[k], axis=1), 3 * n2 * (k + 1) * dt3) for k in (99, 499)]
        axes = [[sig.mean_z(squared[k, :, i], n2 * (k + 1) * dt3) for i in range(3)] for k in (99, 499)]
        cases[name] = {"prediction_relative_error": float(np.max(np.abs(exact - predicted)) / np.max(np.abs(predicted))),
                       "max_transverse_rad": float(transverse.max()), "final_error_rad": _floats(exact[-1]),
                       "arw_z": [float(stats["z"]) for stats in total],
                       "arw_axis_z": [[float(stats["z"]) for stats in row] for row in axes],
                       "arw_95ci_rad2": [_half_width(stats["standard_error"]) for stats in total],
                       "transverse_trace": _floats(transverse[::10])}
    b_perp = float(np.linalg.norm(b3[:2]))
    bound = 2 * b_perp / float(np.linalg.norm(STRAPDOWN["rotation_rad_per_s"]))
    return {"single_axis": single, "strapdown": cases, "transverse_bound_rad": bound,
            "stationary_transverse_at_end_rad": b_perp * steps * dt3, "seed": seed,
            "mean_trace": {"t_s": _floats((np.arange(IMU["steps"]) + 1)[::10] * dt),
                           "mean": _floats(errors.mean(axis=0)[::10]), "variance": _floats(errors.var(axis=0)[::10])}}


@task("T053", changed_files=(MODULE, SIGNALS_FILE, DOC), regression_tests=_tests("test_t053_imu_drift"))
def imu_drift_and_noise(ctx):
    study = imu_study()
    single, cases = study["single_axis"], study["strapdown"]
    ctx.artifact_json("imu-study.json", study)
    trace = study["mean_trace"]
    ctx.artifact_text("heading-error-moments.svg", svg.line_plot(
        [("sample mean", trace["t_s"], trace["mean"]),
         ("b t", trace["t_s"], [IMU["bias_rad_per_s"] * t for t in trace["t_s"]]),
         ("sample variance x 1000", trace["t_s"], [1000 * v for v in trace["variance"]]),
         ("N^2 t x 1000", trace["t_s"], [1000 * IMU["arw_rad_per_sqrt_s"] ** 2 * t for t in trace["t_s"]])],
        title="Single-axis gyro heading error", xlabel="time (s)", ylabel="rad (variance scaled)", markers=False))
    rotating, stationary = cases["rotating"], cases["stationary"]
    findings = [
        finding("Single-axis gyro heading error has mean b t and variance N^2 t", "numerical",
                {"t_s": [r["t_s"] for r in single], "mean_z": [r["mean_z"] for r in single],
                 "variance_z": [r["variance_z"] for r in single]},
                {"generator": _generator("gyro bias plus white rate noise", study["seed"], **IMU),
                 "checks": [_z_check(f"mean at t = {r['t_s']:g} s", r["mean_z"]) for r in single]
                 + [_z_check(f"variance at t = {r['t_s']:g} s", r["variance_z"]) for r in single]},
                uncertainty=_mc95({"mean_rad": [r["mean_95ci_rad"] for r in single],
                                   "variance_rad2": [r["variance_95ci_rad2"] for r in single]},
                                  f"95 % half-widths of the sample moments over {IMU['runs']} runs"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Mean squared heading error is N^2 t + (b t)^2", "numerical",
                {"mse": [r["mse"] for r in single], "predicted": [r["mse_predicted"] for r in single]},
                {"checks": [_z_check(f"mean squared error at t = {r['t_s']:g} s", r["mse_z"]) for r in single]},
                unit="rad^2", uncertainty=_mc95([r["mse_95ci_rad2"] for r in single],
                                                f"95 % half-width of each sample MSE over {IMU['runs']} runs"),
                tolerance={"abs": 1e-15, "rel": 1e-6}),
        finding("Strapdown bias error follows e_{k+1} = exp(-omega dt) e_k + J_r(omega dt) b dt", "numerical",
                {"stationary_relative_error": stationary["prediction_relative_error"],
                 "rotating_relative_error": rotating["prediction_relative_error"]},
                {"checks": [_check("stationary body: integrated error against b t (relative)",
                                   stationary["prediction_relative_error"], 1e-9),
                            _check("rotating body: integrated error against the linearized recursion (relative)",
                                   rotating["prediction_relative_error"], 2e-3)]},
                uncertainty=_truncation(2e-3, "second-order BCH terms of the linearized recursion (relative); the "
                                        "stationary case is exact up to rounding"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("On a body rotating about z, transverse gyro bias produces a bounded orientation error "
                "2 |b_perp| / |omega| instead of |b_perp| t", "numerical",
                {"max_transverse_rad": rotating["max_transverse_rad"], "bound_rad": study["transverse_bound_rad"],
                 "stationary_transverse_rad": stationary["max_transverse_rad"]},
                {"checks": [_check("max transverse error minus the bound (rad)",
                                   rotating["max_transverse_rad"] - study["transverse_bound_rad"],
                                   1e-3 * study["transverse_bound_rad"], "signed_le"),
                            _check("stationary transverse error over the rotating bound",
                                   stationary["max_transverse_rad"] / study["transverse_bound_rad"], 10.0, "ge")]},
                uncertainty=_truncation(rotating["prediction_relative_error"] * study["transverse_bound_rad"],
                                        "linearization error of the recursion carried to the transverse maximum (rad)"),
                tolerance={"abs": 1e-12, "rel": 1e-6},
                counterexample={"statement": "The mean orientation error from a constant gyro bias grows as |bias| t",
                                "witness": {"rotation_rad_per_s": STRAPDOWN["rotation_rad_per_s"],
                                            "bias_rad_per_s": STRAPDOWN["bias_rad_per_s"],
                                            "max_transverse_rad": rotating["max_transverse_rad"],
                                            "stationary_transverse_rad": stationary["max_transverse_rad"]}}),
        finding("Angle random walk stays isotropic, N^2 t per body axis and E|e - m|^2 = 3 N^2 t, on stationary and "
                "rotating bodies", "numerical",
                {"stationary_z": stationary["arw_z"], "rotating_z": rotating["arw_z"],
                 "stationary_axis_z": stationary["arw_axis_z"], "rotating_axis_z": rotating["arw_axis_z"]},
                {"generator": _generator("strapdown SO(3) integration", study["seed"], **STRAPDOWN),
                 "checks": [_z_check(f"{name} total spread at step {k}", z) for name in ("stationary", "rotating")
                            for k, z in zip((100, 500), cases[name]["arw_z"])]
                 + [_z_check(f"{name} spread of body axis {'xyz'[i]} at step {k}", z)
                    for name in ("stationary", "rotating")
                    for k, row in zip((100, 500), cases[name]["arw_axis_z"]) for i, z in enumerate(row)]},
                uncertainty=_mc95({name: cases[name]["arw_95ci_rad2"] for name in ("stationary", "rotating")},
                                  f"95 % half-width of the total spread at steps 100 and 500 over {STRAPDOWN['runs']} "
                                  "runs (rad^2)"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        _unestablished("Real gyroscopes have constant bias and white rate noise with the declared densities",
                       "sensor_performance", "No IMU was acquired; bias instability, scale factor, temperature and "
                       "vibration effects are not modelled."),
    ]
    fields = _fields(
        hypothesis="Integrating a gyro with constant bias b and white rate noise of density N gives heading error "
                   "with mean b t and variance N^2 t; in 3-D the bias error rotates with the body, so transverse "
                   "bias stays bounded while the random walk stays isotropic.",
        mathematical_model="Single axis: e(t) = b t + N W(t). Strapdown: E_{k+1} = exp(-[w] dt) E_k exp([w + b + n] dt); "
                           "linearized e_{k+1} = exp(-[w] dt) e_k + J_r(w dt)(b + n) dt, so transverse components "
                           "circle with radius |b_perp|/|w| (maximum 2|b_perp|/|w|).",
        input_data=[f"Single axis {IMU}", f"Strapdown {STRAPDOWN}", f"Seed {study['seed']}"],
        observation_model="Synthetic imu_orientation outputs from integrated gyro rates; no accelerometer or "
                          "magnetometer aiding.",
        expected_invariant="Moments within 99.9 % Monte Carlo intervals; deterministic bias propagation matches the "
                           "linearized recursion.",
        experiment="Seeded single-axis ensemble; noise-free and noisy SO(3) integrations for a stationary and a "
                   "rotating body; counterexample search for unbounded bias growth.",
        numerical_result=f"single-axis max |z| = {max(max(abs(r['mean_z']), abs(r['variance_z'])) for r in single):.2f}; "
                         f"rotating transverse max {rotating['max_transverse_rad']:.3e} rad vs bound "
                         f"{study['transverse_bound_rad']:.3e} (stationary {stationary['max_transverse_rad']:.3e}); "
                         f"linearization error {rotating['prediction_relative_error']:.1e}.",
        uncertainty="Monte Carlo z-scores with 2000 (single axis) and 200 (strapdown) runs; second-order BCH terms "
                    "limit the linearized recursion to about 1e-3 relative.",
        failure_modes_checked=["noncommuting rotation with a transverse bias", "small-angle log map",
                               "variance versus mean squared error", "anisotropy hidden by a trace-only test "
                               "(per-axis checks)"],
        unresolved_assumptions=["Bias is constant (no bias instability)", "Truth rotation rate is constant and known"],
        recommended_next_task="T054: add asynchronous timestamps")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T054
TIMING = {"rate_a_hz": 100.0, "rate_b_hz": 30.0, "offset_s": 1 / 256, "jitter_s": 5e-4, "runs": 300}


SIGNAL_TONES = ((0.05, 0.7, 0.0), (0.02, 1.9, 0.3))  # amplitude (m), frequency (Hz), phase (rad)


def _signal(t):
    return 0.05 * np.sin(2 * math.pi * 0.7 * t) + 0.02 * np.sin(2 * math.pi * 1.9 * t + 0.3)


def secant_regression_prediction(h: float) -> float:
    """Expected regression of -S delta on -v delta for linear interpolation with knot spacing h.

    For a tone A sin(w t + phi) the interpolant slope on an interval is A w cos(w t_mid + phi) sinc(w h/2),
    and averaging cos(w (t - t_mid)) over a uniform position within the interval gives a second sinc(w h/2);
    tones are weighted by their velocity power (A w)^2. sinc^2(x) = 1 - x^2/3 + ..., i.e. about
    1 - (w h)^2/12.
    """
    weights = [(a * 2 * math.pi * f) ** 2 for a, f, _ in SIGNAL_TONES]
    factors = [(math.sin(math.pi * f * h) / (math.pi * f * h)) ** 2 for _, f, _ in SIGNAL_TONES]
    return sum(w * x for w, x in zip(weights, factors)) / sum(weights)


def _velocity(t):
    return (0.05 * 2 * math.pi * 0.7 * np.cos(2 * math.pi * 0.7 * t)
            + 0.02 * 2 * math.pi * 1.9 * np.cos(2 * math.pi * 1.9 * t + 0.3))


def _acceleration_bound():
    return 0.05 * (2 * math.pi * 0.7) ** 2 + 0.02 * (2 * math.pi * 1.9) ** 2


def timing_study(seed=54) -> dict:
    rng = sig.generator(seed)
    times_a = np.arange(20, 380) / TIMING["rate_a_hz"]
    knots = np.arange(0, 121) / TIMING["rate_b_hz"]
    values = _signal(knots)
    baseline = np.interp(times_a, knots, values) - _signal(times_a)
    # Sample level: attributing x(t) to t + delta.
    deltas = np.geomspace(1e-4, 1e-2, 7)
    residuals, violations = [], 0
    for delta in deltas:
        error = values - _signal(knots + delta)
        residual = np.abs(error + _velocity(knots) * delta)
        violations += int(np.sum(residual > _acceleration_bound() * delta ** 2 / 2 * (1 + 1e-9)))
        residuals.append(float(residual.max()))
    sample_slope = sig.loglog_slope(deltas, residuals)
    # After linear interpolation to the common (A) times.
    delta = 2e-3
    index, weight, width = sig.interpolation_weights(knots, times_a)
    secant = (values[index + 1] - values[index]) / width
    offset_error = np.interp(times_a, knots + delta, values) - _signal(times_a) - baseline
    no_knot = weight * width >= delta
    exact_residual = float(np.max(np.abs(offset_error + secant * delta)[no_knot]))
    reference = -_velocity(times_a) * delta
    regression = float(offset_error @ reference / (reference @ reference))
    regression_prediction = secant_regression_prediction(1 / TIMING["rate_b_hz"])
    # Jitter.
    sigma = TIMING["jitter_s"]
    per_run = np.empty(TIMING["runs"])
    for run in range(TIMING["runs"]):
        stamps = knots + sigma * rng.standard_normal(len(knots))
        per_run[run] = np.mean((np.interp(times_a, stamps, values) - _signal(times_a) - baseline) ** 2)
    predicted = float(np.mean(secant ** 2 * sigma ** 2 * ((1 - weight) ** 2 + weight ** 2)))
    jitter = sig.mean_z(per_run, predicted)
    return {"deltas_s": _floats(deltas), "sample_residual_max": residuals, "sample_slope": sample_slope,
            "bound_violations": violations, "interp_delta_s": delta, "exact_residual": exact_residual,
            "excluded_near_knots": int(np.sum(~no_knot)), "velocity_regression": regression,
            "velocity_regression_predicted": regression_prediction,
            "jitter_mse": float(jitter["sample_mean"]), "jitter_predicted": predicted, "jitter_z": float(jitter["z"]),
            "jitter_mse_95ci": _half_width(jitter["standard_error"]),
            "baseline_rms": float(np.sqrt(np.mean(baseline ** 2))), "seed": seed}


def _encoder_record(value, clock, epoch, time_s, sequence):
    return om.observe("encoder_displacement", value, unit="m", frame_id="axis:x", clock_id=clock,
                      clock_basis="acquisition", epoch=epoch, time_s=float(time_s),
                      calibration_ref="calibration:declared-synthetic", sequence=sequence,
                      raw_ref=f"raw:{clock}:{sequence}")


@task("T054", changed_files=(MODULE, SIGNALS_FILE, MODES_FILE, DOC),
      regression_tests=_tests("test_t054_asynchronous_timestamps"))
def asynchronous_timestamps(ctx):
    study = timing_study()
    offset = TIMING["offset_s"]
    knots = np.arange(0, 121) / TIMING["rate_b_hz"]
    b_records = [_encoder_record(v, "clock:B", "epoch:B-boot", t + offset, k)
                 for k, (t, v) in enumerate(zip(knots, _signal(knots)))]
    a_record = _encoder_record(float(_signal(np.array([1.0]))[0]), "clock:A", "epoch:run-0", 1.0, 0)
    mapping = om.ClockMapping("clock:B", "epoch:B-boot", "acquisition", "clock:A", "epoch:run-0", "acquisition",
                              1.0, -offset, "declared synthetic synchronization")
    mapped = [om.apply_clock(record, mapping) for record in b_records]
    times_a = np.arange(20, 380) / TIMING["rate_a_hz"]
    stamps = np.array([record.time_s for record in mapped])
    values = np.array([record.value[0] for record in mapped])
    mapped_gap = float(np.max(np.abs(np.interp(times_a, stamps, values) - np.interp(times_a, knots, _signal(knots)))))
    unmapped_stamps = np.array([record.time_s for record in b_records])
    unmapped_gap = float(np.max(np.abs(np.interp(times_a, unmapped_stamps, values)
                                       - np.interp(times_a, knots, _signal(knots)))))
    refusals = [_refusal("combining clock A and clock B records without a mapping", "clock_mismatch",
                         lambda: om.combine(a_record, b_records[30])),
                _refusal("combining after the declared mapping", "accepted", lambda: om.combine(a_record, mapped[30])),
                _refusal("applying the mapping to a record of another epoch", "epoch_mismatch",
                         lambda: om.apply_clock(replace(b_records[30], epoch="epoch:B-reboot"), mapping))]
    ctx.artifact_json("timing-study.json", {**study, "declared": TIMING, "mapping": mapping.record(),
                                            "mapped_interpolation_gap": mapped_gap,
                                            "unmapped_interpolation_gap": unmapped_gap})
    ctx.artifact_text("offset-residual.svg", svg.line_plot(
        [("max |e + v delta|", study["deltas_s"], study["sample_residual_max"]),
         ("a_max delta^2 / 2", study["deltas_s"], [_acceleration_bound() * d ** 2 / 2 for d in study["deltas_s"]])],
        title="Clock offset: first-order residual", xlabel="offset delta (s)", ylabel="position residual (m)",
        logx=True, logy=True))
    findings = [
        finding("A clock offset delta produces error -v delta + O(delta^2) at the sample times", "numerical",
                {"residual_slope": study["sample_slope"], "bound_violations": study["bound_violations"]},
                {"checks": [_check("log-log slope of max |e + v delta| minus 2", study["sample_slope"] - 2, 0.02),
                            _check("samples exceeding a_max delta^2 / 2", study["bound_violations"], 0,
                                   kind="invariant")]},
                uncertainty=_truncation({"residual_slope": abs(study["sample_slope"] - 2)},
                                        "the O(delta^3) term tilts the fitted residual slope away from 2"),
                tolerance={"abs": 1e-9, "rel": 1e-9}),
        finding("After linear interpolation to a common time the offset error is exactly -S delta (S the interpolant "
                "slope); its regression on -v delta is the velocity-weighted sinc^2(omega h/2)", "numerical",
                {"exact_residual_m": study["exact_residual"], "velocity_regression": study["velocity_regression"],
                 "predicted_regression": study["velocity_regression_predicted"]},
                {"checks": [_check("|error + S delta| away from knots (m)", study["exact_residual"], 1e-12),
                            _check("regression coefficient on -v delta minus the velocity-weighted sinc^2(omega h/2)",
                                   study["velocity_regression"] - study["velocity_regression_predicted"], 5e-4)]},
                uncertainty=_truncation({"velocity_regression": 5e-4},
                                        "the finite set of interval positions (10 per cycle of 100 Hz against 30 Hz) "
                                        "and samples near knots depart from the uniform-position average"),
                tolerance={"abs": 1e-12, "rel": 1e-9}),
        finding("Timestamp jitter adds mean squared error S^2 sigma_j^2 ((1 - w)^2 + w^2)", "numerical",
                {"sample_mse": study["jitter_mse"], "predicted_mse": study["jitter_predicted"], "z": study["jitter_z"]},
                {"generator": _generator("Gaussian timestamp jitter", study["seed"], jitter_s=TIMING["jitter_s"],
                                         runs=TIMING["runs"]),
                 "checks": [_z_check("per-run mean squared jitter error against the first-order prediction",
                                     study["jitter_z"])]},
                unit="m^2", uncertainty=_mc95({"sample_mse": study["jitter_mse_95ci"]},
                                              f"95 % half-width over {TIMING['runs']} independent runs (m^2)"),
                tolerance={"abs": 1e-18, "rel": 1e-6}),
        finding("A declared clock mapping removes the offset error; combining clocks without one is refused",
                "computational_pipeline", {"mapped_gap_m": mapped_gap, "unmapped_gap_m": unmapped_gap},
                {"checks": refusals + [_check("interpolation with mapped stamps against true stamps (m)", mapped_gap,
                                              1e-15)]},
                uncertainty=_roundoff(1e-16, "dyadic offset 1/256 s; mapped stamps equal the true stamps up to "
                                      "rounding of t + offset"),
                tolerance={"abs": 1e-15, "rel": 1e-9}),
        _unestablished("Real sensor clocks have the declared constant offset and white jitter", "physical",
                       "No clocks were measured; drift, rate error and non-Gaussian latency are not modelled."),
    ]
    fields = _fields(
        hypothesis="An uncorrected clock offset delta between two sensors produces an error -v delta to first order "
                   "at the sample times and exactly -S delta after linear interpolation to a common time, where the "
                   "interpolant slope S equals the velocity up to a sinc(omega h/2) factor; jitter adds a predictable "
                   "variance; a declared clock mapping removes the offset.",
        mathematical_model="x(t) attributed to t + delta: e = x(t) - x(t + delta) = -v delta - a delta^2/2 + ...; "
                           "linear interpolation I(t - delta) - I(t) = -S delta off the knots, with S = v(t_mid) "
                           "sinc(omega h/2) per tone, so regressing on -v delta gives sinc^2(omega h/2) weighted by "
                           "(A omega)^2; jitter j_i: e = -S((1 - w) j_i + w j_{i+1}) to first order.",
        input_data=[f"Declared timing {TIMING}", "Signal 50 mm sin(2 pi 0.7 t) + 20 mm sin(2 pi 1.9 t + 0.3)",
                    f"Seed {study['seed']}"],
        observation_model="Synthetic encoder_displacement records on clock B (30 Hz) interpolated to clock A times "
                          "(100 Hz), acquisition-stamped.",
        expected_invariant="Residual O(delta^2); exact secant law; jitter MSE within the Monte Carlo interval; mapped "
                           "stamps reproduce the true-stamp interpolation.",
        experiment="Offset scan over delta; exact comparison with the interpolant slope; seeded jitter ensemble; "
                   "typed records mapped with a declared ClockMapping and combined.",
        numerical_result=f"residual slope {study['sample_slope']:.4f}; exact residual {study['exact_residual']:.1e} m; "
                         f"velocity regression {study['velocity_regression']:.5f} vs sinc^2 prediction "
                         f"{study['velocity_regression_predicted']:.5f}; jitter MSE {study['jitter_mse']:.3e} "
                         f"vs {study['jitter_predicted']:.3e} (z = {study['jitter_z']:.2f}); mapped gap {mapped_gap:.1e} m "
                         f"vs unmapped {unmapped_gap:.2e} m.",
        uncertainty="The velocity regression departs from 1 by the product of the secant-slope factor and the "
                    "position-within-interval factor, sinc^2(omega h/2), about 1 - (omega h)^2/12; jitter z uses 300 "
                    "independent runs.",
        failure_modes_checked=["offset crossing an interpolation knot (excluded and counted)",
                               "combining clocks without a mapping", "mapping applied to the wrong epoch"],
        unresolved_assumptions=["Clock offset is constant (no drift)", "Jitter is white and much smaller than the "
                                "sample interval"],
        recommended_next_task="T055: add dropped observations")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T055
DROPS = {"probability": 0.2, "good_to_bad": 0.05, "bad_to_good": 0.2, "runs": 400, "samples": 600, "burn_in": 100,
         "walk_variance": 1e-4, "noise_variance": 1e-6}
MEAN_EXPERIMENT = {"mean": 0.5, "sigma": 0.01, "runs": 400, "samples": 200}
STATIONARY_VIBRATION_M = 5e-7  # sub-count vibration of a stationary encoder axis (1 um counts)


def drop_study(seed=55) -> dict:
    rng = sig.generator(seed)
    p = DROPS["probability"]
    n = MEAN_EXPERIMENT["samples"]
    values = MEAN_EXPERIMENT["mean"] + MEAN_EXPERIMENT["sigma"] * rng.standard_normal(n)
    stream = [_encoder_record(v, "clock:daq", "epoch:run-0", 0.01 * k, k) for k, v in enumerate(values)]
    dropped = om.bernoulli_drops(n, p, rng)
    kept = om.with_drops(stream, dropped)
    summary = om.admit_stream(kept)
    position_mismatch = len(set(summary["missing"]) ^ set(np.flatnonzero(dropped).tolist()))
    filled = om.zero_fill(kept, stream[0])
    unbacked = list(kept)
    first_present = next(k for k, item in enumerate(kept) if item is not None)
    unbacked[first_present] = replace(kept[first_present], raw_ref=None)
    stripped = [0.0 if d else float(v) for v, d in zip(values, dropped)]
    flagged = om.detect_zero_fill(stripped, MEAN_EXPERIMENT["sigma"])
    detection_mismatch = len(set(flagged) ^ set(np.flatnonzero(dropped).tolist()))
    # Witness: a stationary encoder axis whose sub-count vibration is quantized to the declared resolution, so
    # genuine readings of exactly 0 counts occur. Filling only the gaps whose genuine reading was 0 counts gives
    # a stream value-identical to the complete acquired stream, which has no fills at all: no rule that sees
    # values alone can recover the fill positions of both.
    resolution = om.MODES["encoder_displacement"].noise_model["resolution_m"]
    readings = np.round(STATIONARY_VIBRATION_M * rng.standard_normal(n) / resolution) * resolution
    genuine_zero = readings == 0.0
    zero_fills = dropped & genuine_zero
    identical_gap = float(np.max(np.abs(np.where(zero_fills, 0.0, readings) - readings)))
    filled_stationary = np.where(dropped, 0.0, readings)
    exact_zero_false = int(np.sum((filled_stationary == 0.0) & ~dropped))
    neighbour_flags = set(om.detect_zero_fill(filled_stationary.tolist(), resolution))
    neighbour_missed = int(np.sum(dropped)) - len(neighbour_flags & set(np.flatnonzero(dropped).tolist()))

    runs, mu, sigma = MEAN_EXPERIMENT["runs"], MEAN_EXPERIMENT["mean"], MEAN_EXPERIMENT["sigma"]
    samples = mu + sigma * rng.standard_normal((runs, n))
    received = ~(rng.random((runs, n)) < p)
    counts = received.sum(axis=1)
    explicit = np.sum(np.where(received, samples, 0.0), axis=1) / counts
    zero_filled = np.sum(np.where(received, samples, 0.0), axis=1) / n
    explicit_bias = sig.mean_z(explicit, mu)
    filled_bias = sig.mean_z(zero_filled, mu * (1 - p))
    explicit_variance = sig.variance_z(explicit, sigma ** 2 * float(np.mean(1.0 / counts)))

    q, r = DROPS["walk_variance"], DROPS["noise_variance"]
    shape = (DROPS["runs"], DROPS["samples"])
    channels = {"bernoulli": (rng.random(shape) < p, p / (1 - p)),
                "gilbert_elliott": (om.gilbert_elliott_drops(DROPS["samples"], DROPS["good_to_bad"], DROPS["bad_to_good"],
                                                             rng, streams=DROPS["runs"]), p / DROPS["bad_to_good"])}
    hold = {}
    for name, (mask, expected_age) in channels.items():
        result = sig.hold_estimates(mask, q, r, rng, DROPS["burn_in"])
        age, mse = sig.mean_z(result["age"], expected_age), sig.mean_z(result["mse"], q * expected_age + r)
        hold[name] = {"drop_rate": float(mask.mean()), "expected_age": expected_age,
                      "mean_age": float(result["age"].mean()), "age_z": float(age["z"]),
                      "mse": float(result["mse"].mean()), "predicted_mse": q * expected_age + r,
                      "mse_z": float(mse["z"]), "age_95ci": _half_width(age["standard_error"]),
                      "mse_95ci": _half_width(mse["standard_error"])}
    return {"stream": {"samples": n, "dropped": int(dropped.sum()), "present": summary["present"],
                       "position_mismatch": position_mismatch,
                       "zero_fill_code": refusal_code(lambda: om.admit_stream(filled)),
                       "unbacked_code": refusal_code(lambda: om.admit_stream(unbacked)),
                       "detection_mismatch": detection_mismatch},
            "stationary": {"resolution_m": resolution, "vibration_sigma_m": STATIONARY_VIBRATION_M,
                           "genuine_zero_readings": int(np.sum(genuine_zero & ~dropped)),
                           "fills_equal_to_genuine_readings": int(zero_fills.sum()),
                           "identical_stream_gap_m": identical_gap, "exact_zero_rule_false_positives": exact_zero_false,
                           "neighbour_rule_flags": len(neighbour_flags), "neighbour_rule_missed": neighbour_missed},
            "mean": {"explicit_bias_z": float(explicit_bias["z"]), "zero_filled_mean": float(filled_bias["sample_mean"]),
                     "zero_filled_bias": float(filled_bias["sample_mean"] - mu), "predicted_bias": -p * mu,
                     "zero_filled_bias_z": float(filled_bias["z"]), "explicit_variance_z": float(explicit_variance["z"]),
                     "variance_inflation": float(explicit_variance["sample_variance"] / (sigma ** 2 / n)),
                     "zero_filled_bias_95ci": _half_width(filled_bias["standard_error"]),
                     "explicit_mean_95ci": _half_width(explicit_bias["standard_error"]),
                     "explicit_variance_95ci": _half_width(explicit_variance["standard_error"])},
            "hold": hold, "seed": seed}


@task("T055", changed_files=(MODULE, MODES_FILE, SIGNALS_FILE, DOC),
      regression_tests=_tests("test_t055_dropped_observations"))
def dropped_observations(ctx):
    study = drop_study()
    stream, stationary, mean, hold = study["stream"], study["stationary"], study["mean"], study["hold"]
    ctx.artifact_json("drop-study.json", {**study, "declared": {"drops": DROPS, "mean_experiment": MEAN_EXPERIMENT}})
    ages = [0.0, 1.1 * max(v["expected_age"] for v in hold.values())]
    ctx.artifact_text("hold-error-by-channel.svg", svg.line_plot(
        [("prediction q E[age] + r", ages, [DROPS["walk_variance"] * a + DROPS["noise_variance"] for a in ages]),
         ("Bernoulli drops (sample)", [hold["bernoulli"]["mean_age"]], [hold["bernoulli"]["mse"]]),
         ("Gilbert-Elliott bursts (sample)", [hold["gilbert_elliott"]["mean_age"]], [hold["gilbert_elliott"]["mse"]])],
        title="Hold-estimate error: Bernoulli versus burst drops at rate 0.2", xlabel="mean age of held sample",
        ylabel="mean squared error"))
    ratio = hold["gilbert_elliott"]["mse"] / hold["bernoulli"]["mse"]
    findings = [
        finding("Dropped observations are retained as explicit gaps at their sequence positions",
                "computational_pipeline", {"dropped": stream["dropped"], "present": stream["present"],
                                           "position_mismatch": stream["position_mismatch"]},
                {"checks": [_check("gap positions differing from the drop mask", stream["position_mismatch"], 0,
                                   kind="exact_arithmetic"),
                            _check("present plus dropped minus samples", stream["present"] + stream["dropped"]
                                   - stream["samples"], 0, kind="exact_arithmetic")]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("A zero-filled stream and a value without a raw reference are refused", "computational_pipeline",
                {"zero_fill": stream["zero_fill_code"], "unbacked": stream["unbacked_code"]},
                {"checks": [{"reference_kind": "refusal", "reference": "admit a zero-filled stream",
                             "expected_refusal": "zero_filled_missing", "observed_refusal": stream["zero_fill_code"],
                             "passed": stream["zero_fill_code"] == "zero_filled_missing"},
                            {"reference_kind": "refusal", "reference": "admit a value whose raw reference was removed",
                             "expected_refusal": "unbacked_value", "observed_refusal": stream["unbacked_code"],
                             "passed": stream["unbacked_code"] == "unbacked_value"}]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("Zero-filling biases the mean by -p mu while explicit gaps leave it unbiased with variance "
                "sigma^2 E[1/N]", "numerical",
                {"zero_filled_bias": mean["zero_filled_bias"], "predicted_bias": mean["predicted_bias"],
                 "explicit_bias_z": mean["explicit_bias_z"], "variance_inflation": mean["variance_inflation"]},
                {"generator": _generator("Bernoulli drops of a constant plus Gaussian noise", study["seed"],
                                         **MEAN_EXPERIMENT, probability=DROPS["probability"]),
                 "checks": [_z_check("explicit-gap mean against mu", mean["explicit_bias_z"]),
                            _z_check("zero-filled mean against (1 - p) mu", mean["zero_filled_bias_z"]),
                            _z_check("explicit-gap variance against sigma^2 E[1/N]", mean["explicit_variance_z"])]},
                uncertainty=_mc95({"zero_filled_bias": mean["zero_filled_bias_95ci"],
                                   "explicit_mean": mean["explicit_mean_95ci"],
                                   "explicit_variance": mean["explicit_variance_95ci"]},
                                  f"95 % half-widths over {MEAN_EXPERIMENT['runs']} independent runs"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Values alone recognize zero fills only in favourable signals: a neighbour-median detector finds every "
                "fill 50 sigma from zero, but on a stationary quantized encoder axis a fill equals a genuine "
                "zero-count reading", "numerical",
                {"far_from_zero_mismatch": stream["detection_mismatch"], "dropped": stream["dropped"],
                 **{key: stationary[key] for key in ("genuine_zero_readings", "fills_equal_to_genuine_readings",
                                                     "identical_stream_gap_m", "exact_zero_rule_false_positives",
                                                     "neighbour_rule_missed")}},
                {"checks": [_check("detector flags differing from true fills (signal at 50 sigma)",
                                   stream["detection_mismatch"], 0, kind="exact_arithmetic"),
                            _check("dropped samples whose genuine reading is 0 counts (stationary axis)",
                                   stationary["fills_equal_to_genuine_readings"], 1, "ge", kind="exact_arithmetic"),
                            _check("max |stream with those gaps zero-filled - complete acquired stream| (m)",
                                   stationary["identical_stream_gap_m"], 0.0, kind="exact_arithmetic"),
                            _check("genuine zero readings an exact-zero rule would flag as fills",
                                   stationary["exact_zero_rule_false_positives"], 1, "ge", kind="exact_arithmetic"),
                            _check("fills the neighbour-median detector misses on the stationary axis",
                                   stationary["neighbour_rule_missed"], 1, "ge", kind="exact_arithmetic")]},
                uncertainty=_exact("integer counts and exact zeros of quantized readings"),
                tolerance={"abs": 0, "rel": 0},
                counterexample={"statement": "Zero-filled gaps can be recognized from the values alone",
                                "witness": {"signal": "stationary encoder axis, 0.5 um vibration quantized to 1 um "
                                                      "counts", "seed": study["seed"], "dropped": stream["dropped"],
                                            "fills_equal_to_genuine_readings":
                                                stationary["fills_equal_to_genuine_readings"],
                                            "identical_stream_gap_m": stationary["identical_stream_gap_m"]}}),
        finding("Hold-estimate error follows q E[age] + r; bursts at the same drop rate raise it", "numerical",
                {name: {"mean_age": v["mean_age"], "mse": v["mse"], "predicted_mse": v["predicted_mse"]}
                 for name, v in hold.items()} | {"burst_to_bernoulli_mse_ratio": ratio},
                {"generator": _generator("random walk through Bernoulli and Gilbert-Elliott channels", study["seed"],
                                         **DROPS),
                 "checks": [_z_check(f"{name} mean age against the analytic E[age]", v["age_z"]) for name, v in hold.items()]
                 + [_z_check(f"{name} hold MSE against q E[age] + r", v["mse_z"]) for name, v in hold.items()]
                 + [_check("burst over Bernoulli MSE ratio at equal drop rate", ratio, 2.0, "ge")]},
                uncertainty=_mc95({name: {"mean_age": v["age_95ci"], "mse": v["mse_95ci"]} for name, v in hold.items()},
                                  f"95 % half-widths over {DROPS['runs']} independent runs"),
                tolerance={"abs": 1e-12, "rel": 1e-6},
                counterexample={"statement": "The drop rate alone determines how much estimates degrade",
                                "witness": {"drop_rate": DROPS["probability"],
                                            "bernoulli_mse": hold["bernoulli"]["mse"],
                                            "burst_mse": hold["gilbert_elliott"]["mse"]}}),
        _unestablished("Real links drop observations as Bernoulli or two-state burst processes with these rates",
                       "sensor_performance", "No link was monitored; real loss depends on load, interference and "
                       "buffering."),
    ]
    fields = _fields(
        hypothesis="Dropped observations must stay explicit gaps: zero-filling biases estimates and cannot be "
                   "detected reliably from values, while explicit gaps keep estimates unbiased; burst losses degrade "
                   "hold estimates more than independent losses at the same rate.",
        mathematical_model="Mean of received samples is unbiased with variance sigma^2 E[1/N]; zero-filled mean has "
                           "expectation (1 - p) mu. Hold error of a random walk: E e^2 = q E[age] + r with "
                           "E[age] = p/(1 - p) (Bernoulli) or pi_B / P(bad -> good) (Gilbert-Elliott, all samples lost "
                           "in the bad state).",
        input_data=[f"Declared drops {DROPS}", f"Mean experiment {MEAN_EXPERIMENT}",
                    f"Stationary encoder axis: vibration sigma {STATIONARY_VIBRATION_M} m quantized to the declared "
                    f"{stationary['resolution_m']} m resolution", f"Seed {study['seed']}"],
        observation_model="Synthetic encoder_displacement records with raw references; drops become None; a "
                          "zero-filled copy fabricates values without raw references.",
        expected_invariant="Gap positions preserved; zero fill refused; moments within 99.9 % Monte Carlo intervals.",
        experiment="Retain a stream with Bernoulli drops, attempt zero fill, run a value-only detector on a signal far "
                   "from zero, construct a quantized stationary-axis stream whose zero-filled version equals a complete "
                   "acquired stream, then Monte Carlo the mean estimator and a sample-and-hold tracker through "
                   "Bernoulli and burst channels.",
        numerical_result=f"{stream['dropped']} of {stream['samples']} dropped, positions exact; zero fill refused "
                         f"({stream['zero_fill_code']}); far-from-zero detector mismatches {stream['detection_mismatch']}; "
                         f"stationary axis: {stationary['fills_equal_to_genuine_readings']} of {stream['dropped']} fills "
                         f"equal genuine 0-count readings (stream identical to a complete one), exact-zero rule "
                         f"{stationary['exact_zero_rule_false_positives']} false positives, neighbour detector misses "
                         f"{stationary['neighbour_rule_missed']}; zero-fill bias {mean['zero_filled_bias']:.4f} vs "
                         f"{mean['predicted_bias']:.4f}; "
                         f"hold MSE {hold['bernoulli']['mse']:.3e} (Bernoulli) vs {hold['gilbert_elliott']['mse']:.3e} "
                         f"(burst), ratio {ratio:.2f}.",
        uncertainty="Monte Carlo z-scores use per-run statistics (runs are independent; samples within a run are "
                    "not).",
        failure_modes_checked=["zero-filled gaps", "value without raw reference",
                               "value-based detection when genuine zero readings occur",
                               "burst versus independent loss", "start-up before the first received sample (burn-in)"],
        unresolved_assumptions=["Drops are independent of the signal value", "Received samples are not delayed "
                                "(delay is T056)"],
        recommended_next_task="T056: add stale-state observations")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T056
STALE = {"limit_s": 0.03, "samples": 400, "rate_hz": 100.0, "latency_range_s": (0.002, 0.05),
         "processing_range_s": (0.0, 0.01), "velocity_m_per_s": 0.5}


def _tracker_record(value, arrival, latency, sequence):
    return om.observe("tracker_measurement", (float(value), 0.0, 0.0), unit="m", frame_id="tracker:room",
                      clock_id="clock:tracker", clock_basis="arrival", epoch="epoch:run-0", time_s=float(arrival),
                      calibration_ref="calibration:declared-synthetic", sequence=sequence,
                      raw_ref=f"raw:tracker:{sequence}", latency_s=None if latency is None else float(latency))


def stale_study(seed=56) -> dict:
    rng = sig.generator(seed)
    n, limit, v = STALE["samples"], STALE["limit_s"], STALE["velocity_m_per_s"]
    acquired = np.arange(n) / STALE["rate_hz"]
    latency = rng.uniform(*STALE["latency_range_s"], n)
    arrival = acquired + latency
    used = arrival + rng.uniform(*STALE["processing_range_s"], n)
    records = [_tracker_record(v * t, a, lat, k) for k, (t, a, lat) in enumerate(zip(acquired, arrival, latency))]
    codes = [refusal_code(lambda rec=rec, now=now: om.admit_fresh(rec, now, limit)) for rec, now in zip(records, used)]
    age = used - acquired
    flag_mismatch = int(np.sum((np.array(codes) == "stale_observation") != (age > limit)))
    # Constant velocity: the error of using each record at its use time, against v times the age computed by
    # the record machinery (arrival stamp minus declared latency).
    record_ages = np.array([om.acquisition_age(rec, now) for rec, now in zip(records, used)])
    stale_errors = np.array([v * now - rec.value[0] for rec, now in zip(records, used)])
    constant_error = float(np.max(np.abs(stale_errors - v * record_ages)))
    amplitude, omega = 0.05, 2 * math.pi * 1.5
    ages = np.geomspace(1e-3, 0.1, 9)
    grid = np.linspace(0.5, 3.5, 601)
    residuals, violations = [], 0
    for a in ages:
        error = amplitude * (np.sin(omega * grid) - np.sin(omega * (grid - a)))
        residual = np.abs(error - amplitude * omega * np.cos(omega * grid) * a)
        violations += int(np.sum(residual > amplitude * omega ** 2 * a ** 2 / 2 * (1 + 1e-9)))
        residuals.append(float(residual.max()))
    return {"stale": int(np.sum(age > limit)), "fresh": int(np.sum(age <= limit)), "flag_mismatch": flag_mismatch,
            "codes": sorted(set(codes)), "constant_velocity_error": constant_error, "ages_s": _floats(ages),
            "sinusoid_residual_max": residuals, "residual_slope": sig.loglog_slope(ages, residuals),
            "bound_violations": violations, "seed": seed,
            "age_trace": {"age_s": _floats(age[::8]), "error_m": _floats((v * age)[::8])}}


@task("T056", changed_files=(MODULE, MODES_FILE, DOC), regression_tests=_tests("test_t056_stale_observations"))
def stale_state_observations(ctx):
    study = stale_study()
    limit = STALE["limit_s"]
    missing_latency = _tracker_record(0.1, 1.0, None, 0)
    future = _tracker_record(0.1, 1.02, 0.01, 1)
    late = _tracker_record(0.1, 1.0, 0.035, 2)
    use_time = 1.005
    late_ages = {"arrival_age_s": use_time - late.time_s, "acquisition_age_s": om.acquisition_age(late, use_time)}
    refusals = [_refusal("arrival-stamped record without a declared latency", "missing_latency",
                         lambda: om.admit_fresh(missing_latency, 1.01, limit)),
                _refusal("record acquired after the use time", "future_observation",
                         lambda: om.admit_fresh(future, 1.0, limit)),
                _refusal("record older than the validity limit", "stale_observation",
                         lambda: om.admit_fresh(late, use_time, limit))]
    ctx.artifact_json("stale-study.json", {**study, "declared": STALE})
    ctx.artifact_text("stale-error-vs-age.svg", svg.line_plot(
        [("max |e - v a| (sinusoid)", study["ages_s"], study["sinusoid_residual_max"]),
         ("A w^2 a^2 / 2", study["ages_s"], [0.05 * (2 * math.pi * 1.5) ** 2 * a ** 2 / 2 for a in study["ages_s"]])],
        title="Stale observation: second-order residual", xlabel="age a (s)", ylabel="m", logx=True, logy=True))
    findings = [
        finding("Observations older than the validity limit are flagged exactly by acquisition-time age",
                "computational_pipeline", {"stale": study["stale"], "fresh": study["fresh"],
                                           "flag_mismatch": study["flag_mismatch"]},
                {"generator": _generator("uniform latency and processing delay", study["seed"], **{
                    k: list(v) if isinstance(v, tuple) else v for k, v in STALE.items()}),
                 "checks": [_check("flags differing from age > limit computed independently", study["flag_mismatch"], 0,
                                   kind="exact_arithmetic"), refusals[2]]},
                uncertainty=_exact("integer flag counts; no sample lies within rounding of the limit"),
                tolerance={"abs": 0, "rel": 0}),
        finding("Stale-state error equals velocity times age for constant velocity and to first order otherwise",
                "numerical", {"constant_velocity_error_m": study["constant_velocity_error"],
                              "residual_slope": study["residual_slope"], "bound_violations": study["bound_violations"]},
                {"checks": [_check("|(v t_use - recorded position) - v acquisition_age(record, t_use)| at constant "
                                   "velocity, over the 400 tracker records (m)", study["constant_velocity_error"], 1e-12),
                            _check("log-log slope of the residual minus 2", study["residual_slope"] - 2, 0.05),
                            _check("residuals above A omega^2 a^2 / 2", study["bound_violations"], 0,
                                   kind="invariant")]},
                uncertainty=_truncation({"constant_velocity_error_m": 1e-15,
                                         "residual_slope": abs(study["residual_slope"] - 2)},
                                        "constant velocity: rounding of arrival - latency (a few ulp of 4 s times "
                                        "0.5 m/s); slope: the O(a^3) term tilts the fit away from 2"),
                tolerance={"abs": 1e-12, "rel": 1e-9}),
        finding("Age is refused without a declared latency for arrival stamps and for observations from the future",
                "computational_pipeline", [check["observed_refusal"] for check in refusals[:2]],
                {"checks": refusals[:2]}, uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("Staleness is decided by acquisition-time age, not arrival age: a record that arrived 5 ms before use "
                "with 35 ms latency is refused under a 30 ms limit", "computational_pipeline",
                {**late_ages, "latency_s": late.latency_s, "limit_s": limit},
                {"checks": [refusals[2],
                            _check("arrival age minus the limit (s; negative: fresh by arrival age)",
                                   late_ages["arrival_age_s"] - limit, 0.0, "signed_le", kind="invariant"),
                            _check("acquisition age minus the limit (s; positive: stale)",
                                   late_ages["acquisition_age_s"] - limit, 0.0, "signed_ge", kind="invariant")]},
                uncertainty=_roundoff(1e-15, "decimal stamps 1.0, 1.005 and 0.035 s are not dyadic (a few ulp)"),
                tolerance={"abs": 1e-12, "rel": 0}),
        _unestablished("Real tracker latencies and target speeds match the declared values", "physical",
                       "No tracker was acquired; latency distributions and target dynamics are declared."),
    ]
    fields = _fields(
        hypothesis="An observation used after its validity age must be flagged, with age measured from acquisition, "
                   "not arrival; using a stale position of a moving target costs velocity x age to first order.",
        mathematical_model="age = t_use - (t_arrival - latency); stale iff age > limit. Error x(t) - x(t - a) = v a - "
                           "x'' a^2/2 + ..., exact v a for constant velocity; |residual| <= max|x''| a^2 / 2.",
        input_data=[f"Declared {STALE}", "Sinusoid 50 mm at 1.5 Hz for the residual law", f"Seed {study['seed']}"],
        observation_model="Synthetic arrival-stamped tracker_measurement records with declared latency.",
        expected_invariant="Flags equal age > limit exactly; residual O(a^2) below the curvature bound.",
        experiment="Admit each record at its use time; compare flags with an independent numpy age computation; "
                   "compare each record's constant-velocity stale error with v times the age acquisition_age "
                   "derives from the record; scan a sinusoid's error against age; refusals for undeclared latency, "
                   "future records and late arrival.",
        numerical_result=f"{study['stale']} stale / {study['fresh']} fresh, {study['flag_mismatch']} mismatches; "
                         f"constant-velocity error {study['constant_velocity_error']:.1e} m; residual slope "
                         f"{study['residual_slope']:.4f}; {study['bound_violations']} bound violations.",
        uncertainty="Deterministic given the seed; flag equality is exact (no sample lies within rounding of the "
                    "limit).",
        failure_modes_checked=["arrival stamp without latency", "future observation", "short arrival age with long "
                               "latency", "second-order motion"],
        unresolved_assumptions=["Latency is known per record", "Use time is on the same clock as the stamps"],
        recommended_next_task="T057: compare raw, filtered and smoothed measurements")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T057
TRACK = {"dt_s": 0.1, "q": 0.5, "r": 0.25, "runs": 300, "steps": 200, "x0": [0.0, 1.0], "p0_diag": [1.0, 0.25]}


def filter_study(seed=57) -> dict:
    rng = sig.generator(seed)
    transition, process = sig.constant_velocity(TRACK["dt_s"], TRACK["q"])
    x0, p0 = np.array(TRACK["x0"]), np.diag(TRACK["p0_diag"])
    truth, measured = sig.simulate_track(TRACK["runs"], TRACK["steps"], transition, process, x0, p0, TRACK["r"], rng)
    result = sig.kalman_rts(measured, transition, process, TRACK["r"], x0, p0)
    difference = result["p_filt"] - result["p_smooth"]
    min_eig = float(min(np.linalg.eigvalsh(0.5 * (d + d.T)).min() for d in difference))
    # P_f - P_s is only semidefinite (rank one at step N-2), so strict reduction is tested on the trace.
    interior_trace = float(min(np.trace(d) for d in difference[:-1]))
    filtered, smoothed = result["x_filt"] - truth, result["x_smooth"] - truth
    position_errors = {"raw": measured - truth[:, :, 0], "filtered": filtered[:, :, 0], "smoothed": smoothed[:, :, 0]}
    rmse = {name: float(np.sqrt(np.mean(e ** 2))) for name, e in position_errors.items()}
    # Runs are independent (steps within a run are not): standard errors come from per-run statistics.
    rmse_95ci = {}
    for name, e in position_errors.items():
        per_run = sig.mean_z(np.mean(e ** 2, axis=1), 0.0)
        rmse_95ci[name] = _half_width(per_run["standard_error"]) / (2 * rmse[name])
    runs = TRACK["runs"]
    low = sig.chi2_quantile(0.025, 2 * runs) / runs
    high = sig.chi2_quantile(0.975, 2 * runs) / runs
    consistency = {}
    for name, errors, covariance in (("filtered", filtered, result["p_filt"]), ("smoothed", smoothed, result["p_smooth"])):
        values = sig.nees(errors, covariance)
        average = values.mean(axis=0)
        consistency[name] = {"inside_fraction": float(np.mean((average >= low) & (average <= high))),
                             "mean_nees": float(average.mean()), "trace": _floats(average[::4]),
                             "mean_nees_95ci": _half_width(sig.mean_z(values.mean(axis=1), 2.0)["standard_error"])}
    steady = sig.riccati_steady_state(transition, process, TRACK["r"])
    worse_by_run = np.mean(np.abs(smoothed[:, :, 0]) > np.abs(filtered[:, :, 0]), axis=1)
    worse = float(worse_by_run.mean())
    return {"min_eigenvalue": min_eig, "interior_min_trace": interior_trace, "rmse": rmse, "rmse_95ci": rmse_95ci,
            "smoothed_worse_95ci": _half_width(sig.mean_z(worse_by_run, worse)["standard_error"]),
            "nees_bounds": [low, high], "consistency": consistency, "steady_state": steady.tolist(),
            "filter_final_prediction": result["p_pred"][-1].tolist(),
            "steady_gap": float(np.max(np.abs(result["p_pred"][-1] - steady)) / np.max(np.abs(steady))),
            "smoothed_worse_fraction": worse, "seed": seed,
            "rmse_trace": {name: _floats(np.sqrt(np.mean(e ** 2, axis=0))[::4]) for name, e in
                           (("raw", measured - truth[:, :, 0]), ("filtered", filtered[:, :, 0]),
                            ("smoothed", smoothed[:, :, 0]))}}


def _scipy_checks(study):
    """Independent steady-state covariance (scipy DARE) and chi-square quantiles (scipy.stats)."""
    import scipy
    from scipy.linalg import solve_discrete_are
    from scipy.stats import chi2

    transition, process = sig.constant_velocity(TRACK["dt_s"], TRACK["q"])
    dare = solve_discrete_are(transition.T, np.array([[1.0], [0.0]]), process, np.array([[TRACK["r"]]]))
    steady = np.array(study["steady_state"])
    dare_gap = float(np.max(np.abs(dare - steady)) / np.max(np.abs(dare)))
    runs = TRACK["runs"]
    exact = [chi2.ppf(0.025, 2 * runs) / runs, chi2.ppf(0.975, 2 * runs) / runs]
    quantile_gap = float(max(abs(a - b) / b for a, b in zip(study["nees_bounds"], exact)))
    return dare_gap, quantile_gap, scipy.__version__


@task("T057", changed_files=(MODULE, SIGNALS_FILE, DOC), regression_tests=_tests("test_t057_filter_and_smoother"))
def raw_filtered_smoothed(ctx):
    study = filter_study()
    rmse, consistency = study["rmse"], study["consistency"]
    ctx.artifact_json("filter-study.json", {**study, "declared": TRACK})
    steps = list(range(0, TRACK["steps"], 4))
    ctx.artifact_text("rmse-by-step.svg", svg.line_plot(
        [(name, steps, trace) for name, trace in study["rmse_trace"].items()],
        title="Position RMSE across the ensemble", xlabel="step", ylabel="RMSE", markers=False))
    ctx.artifact_text("nees-by-step.svg", svg.line_plot(
        [(name, steps, consistency[name]["trace"]) for name in ("filtered", "smoothed")]
        + [("95 % lower bound", [0, TRACK["steps"] - 1], [study["nees_bounds"][0]] * 2),
           ("95 % upper bound", [0, TRACK["steps"] - 1], [study["nees_bounds"][1]] * 2)],
        title="Ensemble-average NEES (2 dof)", xlabel="step", ylabel="NEES", markers=False))
    steady_basis = {"checks": [_check("filter final predicted covariance against the Riccati fixed point (relative)",
                                      study["steady_gap"], 1e-10, kind="self_convergence")]}
    quantile_basis = {"derivation": f"{DOC}#t057-raw-filtered-and-smoothed-estimates "
                                    "(Wilson-Hilferty chi-square quantile)"}
    if ctx.available("module:scipy"):
        dare_gap, quantile_gap, scipy_version = _scipy_checks(study)
        steady_basis["independent_check"] = dict(
            _check("ciw Riccati iteration against scipy.linalg.solve_discrete_are (relative)", dare_gap, 1e-10),
            producer=dict(PRODUCER), checker={"implementation": "scipy.linalg", "revision": scipy_version})
        quantile_basis["independent_check"] = dict(
            _check("Wilson-Hilferty NEES bounds against scipy.stats.chi2.ppf (relative)", quantile_gap, 1e-4),
            producer=dict(PRODUCER), checker={"implementation": "scipy.stats", "revision": scipy_version})
    runs = TRACK["runs"]
    coverage = [sig.chi2_cdf(bound * runs, 2 * runs) for bound in study["nees_bounds"]]
    quantile_basis["checks"] = [_check("chi-square CDF (incomplete-gamma series) at the lower bound minus 0.025",
                                       coverage[0] - 0.025, 1e-4, kind="high_precision"),
                                _check("chi-square CDF (incomplete-gamma series) at the upper bound minus 0.975",
                                       coverage[1] - 0.975, 1e-4, kind="high_precision")]
    findings = [
        finding("The RTS smoothed covariance never exceeds the filtered covariance in matrix order", "numerical",
                {"min_eigenvalue_filtered_minus_smoothed": study["min_eigenvalue"],
                 "interior_min_trace_reduction": study["interior_min_trace"]},
                {"generator": _generator("constant-velocity track", study["seed"], **TRACK),
                 "checks": [_check("min eigenvalue of P_filt - P_smooth over all steps (>= -1e-12: semidefinite up "
                                   "to rounding)", study["min_eigenvalue"], -1e-12, "signed_ge", "invariant"),
                            _check("min trace(P_filt - P_smooth) before the final step (strict reduction)",
                                   study["interior_min_trace"], 1e-6, "ge", "invariant")]},
                uncertainty=_roundoff(1e-15, "eigenvalues of 2 x 2 covariance differences with entries below 1"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Ensemble position RMSE orders smoothed <= filtered <= raw", "numerical", rmse,
                {"generator": _generator("constant-velocity track", study["seed"], **TRACK),
                 "checks": [_check("filtered minus raw RMSE", rmse["filtered"] - rmse["raw"], 0.0, "signed_le",
                                   "invariant"),
                            _check("smoothed minus filtered RMSE", rmse["smoothed"] - rmse["filtered"], 0.0, "signed_le",
                                   "invariant")]},
                uncertainty=_mc95(study["rmse_95ci"], f"95 % half-widths from per-run mean squared errors over "
                                  f"{TRACK['runs']} independent runs"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Filtered and smoothed NEES are chi-square consistent across the ensemble", "numerical",
                {name: {"inside_fraction": v["inside_fraction"], "mean_nees": v["mean_nees"]}
                 for name, v in consistency.items()} | {"bounds_95": study["nees_bounds"]},
                {"checks": [_check(f"{name} fraction of steps inside the 95 % bounds", v["inside_fraction"], 0.9, "ge")
                            for name, v in consistency.items()]
                 + [_check(f"{name} time-averaged NEES minus 2", v["mean_nees"] - 2, 0.1) for name, v in
                    consistency.items()]},
                uncertainty=_mc95({name: {"mean_nees": v["mean_nees_95ci"]} for name, v in consistency.items()},
                                  f"95 % half-width of the time-averaged NEES from per-run averages over "
                                  f"{TRACK['runs']} runs"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("The filter's predicted covariance reaches the discrete Riccati fixed point", "numerical",
                {"steady_state": study["steady_state"], "relative_gap": study["steady_gap"]}, steady_basis,
                uncertainty=_roundoff(1e-15, "relative rounding of the fixed-point iteration and the 200-step filter "
                                      "recursion"),
                tolerance={"abs": 1e-12, "rel": 1e-9}),
        finding("Wilson-Hilferty chi-square quantiles give the NEES consistency bounds", "numerical",
                {"bounds_95": study["nees_bounds"], "cdf_at_bounds": coverage}, quantile_basis,
                uncertainty=_uncertainty("reference_error", max(abs(coverage[0] - 0.025), abs(coverage[1] - 0.975)),
                                         "coverage error of the Wilson-Hilferty bounds at 600 degrees of freedom, "
                                         "measured with the incomplete-gamma series CDF"),
                tolerance={"abs": 1e-12, "rel": 1e-9}),
        finding("Smoothing does not reduce the error of every individual sample", "numerical",
                {"smoothed_worse_fraction": study["smoothed_worse_fraction"]},
                {"checks": [_check("fraction of (run, step) samples where |smoothed error| > |filtered error|",
                                   study["smoothed_worse_fraction"], 0.1, "ge")]},
                uncertainty=_mc95(study["smoothed_worse_95ci"], f"95 % half-width from per-run fractions over "
                                  f"{TRACK['runs']} runs"),
                tolerance={"abs": 1e-12, "rel": 0},
                counterexample={"statement": "The smoothed estimate is closer to truth than the filtered estimate at "
                                             "every sample",
                                "witness": {"seed": study["seed"], "fraction_worse": study["smoothed_worse_fraction"]}}),
        _unestablished("A real tracker's measurement noise and target motion match the constant-velocity model",
                       "sensor_performance", "No tracker data were acquired; the model and noise are declared."),
    ]
    fields = _fields(
        hypothesis="For a linear-Gaussian constant-velocity track, the RTS smoother's covariance is below the "
                   "filter's in matrix order and the ensemble RMSE orders smoothed <= filtered <= raw, with NEES "
                   "consistent with chi-square bounds; the ordering holds in the ensemble, not per sample.",
        mathematical_model="x_{k+1} = F x_k + w, F = [[1, dt], [0, 1]], Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]; "
                           "z_k = x_k[0] + v, R = r. Kalman filter (Joseph form) and RTS: P_s = P_f + C (P_s' - P_p') C^T "
                           "with P_s' <= P_p', so P_s <= P_f. NEES averaged over N runs ~ chi2(2N)/N.",
        input_data=[f"Declared track {TRACK}", f"Seed {study['seed']}"],
        observation_model="Synthetic position measurements (raw), causal filter estimates and non-causal smoothed "
                          "estimates of the same track.",
        expected_invariant="Matrix order P_s <= P_f; RMSE order; NEES inside 95 % bounds at about 95 % of steps.",
        experiment="Seeded ensemble, filter and smoother, eigenvalue test of P_f - P_s, RMSE and NEES statistics, "
                   "Riccati fixed point and Wilson-Hilferty quantiles (each also compared with scipy "
                   "solve_discrete_are and scipy.stats.chi2 when scipy is available), and a per-sample dominance "
                   "counterexample search.",
        numerical_result=f"min eig(P_f - P_s) = {study['min_eigenvalue']:.1e} (semidefinite; exactly 0 at the final "
                         "step where P_s = P_f), min trace reduction before the last "
                         f"step {study['interior_min_trace']:.3e}; "
                         f"RMSE raw {rmse['raw']:.4f}, filtered {rmse['filtered']:.4f}, smoothed {rmse['smoothed']:.4f}; "
                         f"NEES inside bounds {consistency['filtered']['inside_fraction']:.2f} / "
                         f"{consistency['smoothed']['inside_fraction']:.2f}; smoothed worse at "
                         f"{study['smoothed_worse_fraction']:.1%} of samples.",
        uncertainty="NEES bounds are 95 % two-sided per step (about 5 % of steps expected outside); RMSE values carry "
                    "Monte Carlo error of order 1 % for 300 runs.",
        failure_modes_checked=["final step where P_s = P_f (equality allowed)",
                               "rank-one difference at step N-2 (semidefinite, not definite)",
                               "covariance symmetry (Joseph form)",
                               "per-sample versus ensemble ordering", "chi-square quantile approximation"],
        unresolved_assumptions=["Model matches the generator exactly (no mismatch)", "Measurements are synchronous "
                                "and none are dropped"],
        recommended_next_task="T066: verify that filtered residuals use filter covariance, not raw sensor covariance")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T058
def frame_clock_study() -> dict:
    # Exactly representable values make the declared mappings checkable by exact arithmetic.
    quarter_turn = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    exact_map = om.FrameMapping("tracker:room", "tracker:cell", quarter_turn, (0.25, -0.5, 1.0),
                                "calibration:declared-synthetic")
    points = [(0.125, -0.25, 0.625), (1.5, 2.25, -0.75), (-3.0, 0.5, 0.0)]
    records = [example_observation("tracker_measurement", value=p, sequence=k) for k, p in enumerate(points)]
    mapped = [om.apply_frame(r, exact_map) for r in records]
    by_hand = [(-p[1] + 0.25, p[0] - 0.5, p[2] + 1.0) for p in points]
    exact_gap = max(abs(a - b) for m, h in zip(mapped, by_hand) for a, b in zip(m.value, h))
    identity_recorded = all(m.mappings == (exact_map.identity(),) and m.frame_id == "tracker:cell" for m in mapped)
    rotation = cam.rotation_matrix((1.0, 2.0, 2.0), 0.7)
    general = om.FrameMapping("tracker:room", "tracker:cell", tuple(tuple(float(v) for v in row) for row in rotation),
                              (0.1, -0.2, 0.3), "calibration:declared-synthetic")
    round_trip = max(abs(a - b) for r in records
                     for a, b in zip(om.apply_frame(om.apply_frame(r, general), general.inverse()).value, r.value))
    # A general rotation rounds each coordinate; allow a few ulp of the largest coordinate.
    scale = max(abs(v) for point in points for v in point)
    rounding_bound = 16 * float(np.finfo(float).eps) * scale
    # Distances between mapped tracker positions must not change under a rigid mapping.
    general_mapped = [om.apply_frame(r, general) for r in records]
    distance_change = max(abs(math.dist(general_mapped[i].value, general_mapped[j].value)
                              - math.dist(records[i].value, records[j].value))
                          for i in range(len(records)) for j in range(i + 1, len(records)))
    # Distance-mode records are carried unchanged by design (apply_frame copies their value).
    chord_record = example_observation("camera_chord_distance")
    rig_map = om.FrameMapping("camera_rig:stereo-0", "camera_rig:stereo-1", tuple(map(tuple, rotation)),
                              (0.1, 0.0, 0.0), "calibration:declared-synthetic")
    chord_carried = om.apply_frame(chord_record, rig_map).value == chord_record.value

    base = example_observation("tracker_measurement", clock_id="clock:tracker", time_s=0.5)
    shifted = example_observation("tracker_measurement", clock_id="clock:daq", epoch="epoch:boot-7",
                                  clock_basis="arrival", time_s=2.015625, latency_s=0.0078125, sequence=1)
    to_acquisition = om.ClockMapping("clock:daq", "epoch:boot-7", "arrival", "clock:daq", "epoch:boot-7",
                                     "acquisition", 1.0, -0.0078125, "declared latency")
    to_tracker = om.ClockMapping("clock:daq", "epoch:boot-7", "acquisition", "clock:tracker", "epoch:run-0",
                                 "acquisition", 1.0, -1.5, "declared synchronization")
    # A tracker record is stamped on arrival; map base to acquisition too so both share one basis.
    base_acq = om.apply_clock(base, om.ClockMapping("clock:tracker", "epoch:run-0", "arrival", "clock:tracker",
                                                    "epoch:run-0", "acquisition", 1.0, -0.0078125, "declared latency"))
    other = om.apply_clock(om.apply_clock(shifted, to_acquisition), to_tracker)
    expected_time = 2.015625 - 0.0078125 - 1.5
    time_gap = abs(other.time_s - expected_time)
    combined = om.combine(other, base_acq)
    dt_gap = abs(combined["dt_s"] - (expected_time - (0.5 - 0.0078125)))
    refusals = {
        "frame_mismatch": refusal_code(lambda: om.combine(records[0], mapped[1])),
        "clock_mismatch": refusal_code(lambda: om.combine(base, replace(base, clock_id="clock:daq"))),
        "epoch_mismatch": refusal_code(lambda: om.combine(base, replace(base, epoch="epoch:run-1"))),
        "clock_basis_mismatch": refusal_code(lambda: om.combine(base_acq, replace(base_acq, clock_basis="arrival"))),
        "mapping_not_applicable": refusal_code(lambda: om.apply_frame(mapped[0], exact_map)),
        "frame_kind_mismatch": refusal_code(lambda: om.apply_frame(
            records[0], om.FrameMapping("tracker:room", "camera_rig:stereo-0", quarter_turn, (0.0, 0.0, 0.0), "c"))),
        "epoch_mismatch_mapping": refusal_code(lambda: om.apply_clock(replace(shifted, epoch="epoch:boot-8"),
                                                                       to_acquisition)),
    }
    return {"exact_gap": exact_gap, "identity_recorded": identity_recorded, "round_trip_error": round_trip,
            "rounding_bound": rounding_bound, "distance_change": distance_change,
            "chord_records_carried_unchanged": chord_carried, "time_gap": time_gap, "dt_gap": dt_gap,
            "combined": combined, "mapped_records": [m.record() for m in mapped], "refusals": refusals,
            "mappings": {"exact_frame": exact_map.record(), "general_frame": general.record(),
                         "arrival_to_acquisition": to_acquisition.record(), "daq_to_tracker": to_tracker.record()},
            "final_record": other.record()}


@task("T058", changed_files=(MODULE, MODES_FILE, DOC), regression_tests=_tests("test_t058_frame_and_clock_basis"))
def frame_and_clock_basis(ctx):
    study = frame_clock_study()
    ctx.artifact_json("frame-clock-study.json", study)
    expected = {"frame_mismatch": "frame_mismatch", "clock_mismatch": "clock_mismatch",
                "epoch_mismatch": "epoch_mismatch", "clock_basis_mismatch": "clock_basis_mismatch",
                "mapping_not_applicable": "mapping_not_applicable", "frame_kind_mismatch": "frame_kind_mismatch",
                "epoch_mismatch_mapping": "epoch_mismatch"}
    refusal_checks = [{"reference_kind": "refusal", "reference": name.replace("_", " "), "expected_refusal": code,
                       "observed_refusal": study["refusals"][name], "passed": study["refusals"][name] == code}
                      for name, code in expected.items()]
    findings = [
        finding("Combining observations across frames, clocks, epochs or time bases without a declared mapping is "
                "refused", "computational_pipeline", {k: study["refusals"][k] for k in list(expected)[:4]},
                {"checks": refusal_checks[:4]}, uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("A declared rigid frame mapping is applied exactly on dyadic inputs, preserves distances to rounding "
                "and is recorded on the observation",
                "computational_pipeline", {"exact_gap_m": study["exact_gap"], "round_trip_error_m": study["round_trip_error"],
                                           "distance_change_m": study["distance_change"],
                                           "chord_records_carried_unchanged": study["chord_records_carried_unchanged"],
                                           "identity_recorded": study["identity_recorded"]},
                {"checks": [_check("quarter-turn mapping against the hand-written formula (dyadic values)",
                                   study["exact_gap"], 0.0, kind="exact_arithmetic"),
                            _check("general rotation mapping followed by its inverse (m; bound 16 eps max|p|)",
                                   study["round_trip_error"], study["rounding_bound"], kind="invariant"),
                            _check("change of pairwise distances between tracker records under the general rotation "
                                   "(m; bound 16 eps max|p|)", study["distance_change"], study["rounding_bound"],
                                   kind="invariant"),
                            _check("mapping identity missing from mapped records", 0 if study["identity_recorded"] else 1,
                                   0, kind="exact_arithmetic")]},
                uncertainty=_roundoff(study["rounding_bound"], "16 eps times the largest coordinate (3 m) for the "
                                      "general rotation; the dyadic quarter turn is exact"),
                tolerance={"abs": 1e-14, "rel": 0}),
        finding("Declared clock mappings (arrival to acquisition, then clock to clock) are applied exactly",
                "computational_pipeline", {"time_gap_s": study["time_gap"], "combined_dt_gap_s": study["dt_gap"],
                                           "final_clock": study["final_record"]["clock_id"],
                                           "final_basis": study["final_record"]["clock_basis"]},
                {"checks": [_check("mapped time against the declared affine map (dyadic values, s)", study["time_gap"],
                                   0.0, kind="exact_arithmetic"),
                            _check("time difference after mapping both records to one basis (s)", study["dt_gap"], 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty=_exact("dyadic times and offsets; no rounding"), tolerance={"abs": 0, "rel": 0}),
        finding("A mapping declared for another frame, frame kind or epoch is refused", "computational_pipeline",
                {k: study["refusals"][k] for k in list(expected)[4:]}, {"checks": refusal_checks[4:]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        _unestablished("The declared frame and clock mappings equal the real extrinsic calibration and clock "
                       "synchronization", "calibration", "Mappings are declared synthetic values; no calibration or "
                       "synchronization procedure was run."),
    ]
    fields = _fields(
        hypothesis="Observations that carry frame id, clock id, epoch and time basis can be combined only after an "
                   "explicit declared mapping, which is then applied exactly and recorded.",
        mathematical_model="Frame mapping p' = R p + t between frames of one kind (distances invariant, checked on "
                           "tracker positions; distance-mode records are carried unchanged by design); clock mapping "
                           "t' = rate t + offset from (clock, epoch, basis) to another; arrival -> acquisition by the "
                           "declared latency.",
        input_data=["Tracker records at dyadic positions", "Quarter-turn and general rotations",
                    "Clock mappings with dyadic offsets (latency 1/128 s, offset -1.5 s)"],
        observation_model="Synthetic tracker_measurement and camera_chord_distance records.",
        expected_invariant="Exact agreement with hand-written maps on dyadic inputs; inverse round trip to rounding; "
                           "refusal of every undeclared combination.",
        experiment="Combine mismatched records; apply declared frame and clock mappings; compare with independent "
                   "hand-written formulas; attempt mappings with the wrong source frame, kind or epoch.",
        numerical_result=f"exact gap {study['exact_gap']}, round trip {study['round_trip_error']:.1e} m and distance "
                         f"change {study['distance_change']:.1e} m (bound {study['rounding_bound']:.1e} m), time gap "
                         f"{study['time_gap']} s; refusals {sorted(set(study['refusals'].values()))}.",
        uncertainty="Exact arithmetic for dyadic inputs; general rotations within 16 eps of the largest coordinate "
                    "(about 1e-14 m), since each mapped coordinate is rounded.",
        failure_modes_checked=["frame, clock, epoch and basis mismatches", "mapping for another source frame",
                               "mapping across frame kinds", "clock mapping for another epoch",
                               "mapping identity recorded on the result"],
        unresolved_assumptions=["Clock rates are exactly 1 (drift not modelled)", "Mappings are static"],
        recommended_next_task="T059: retain observations without admitting them as state")
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T059
def retention_study() -> dict:
    per_mode = {}
    for name in sorted(om.MODES):
        components = om.MODES[name].components
        store = om.StateStore(name, [0.0] * components, [1.0] * components)
        before = store.digest()
        observations = [example_observation(name, sequence=k, raw_ref=f"raw:{name}:{k}") for k in range(3)]
        records = [store.retain(o) for o in observations]
        after = store.digest()
        per_mode[name] = {
            "state_unchanged": before == after, "retained": len(store.retained()),
            "admission": sorted({r["state_admission"] for r in records}),
            "retention": sorted({r["retention"] for r in records}),
            "update_code": refusal_code(lambda s=store, r=records[0]: s.update(r, 0.5)),
            "state_after_refusal_unchanged": store.digest() == before}
    store = om.StateStore("intrinsic_geodesic_distance", 1.0, 0.5)
    record = store.retain(example_observation("intrinsic_geodesic_distance", value=2.0))
    admission = store.admit(record["observation_digest"], "declared synthetic admission")
    updated = store.update(record, 0.5)
    exact = {"mean": updated["mean"][0] - 1.5, "variance": updated["variance"][0] - 0.25}
    tampered = dict(record, observation=dict(record["observation"], value=[2.5]))
    stranger = example_observation("intrinsic_geodesic_distance", value=3.0, sequence=9)
    chord_store = om.StateStore("intrinsic_geodesic_distance", 0.0, 1.0)
    chord_record = chord_store.retain(example_observation("camera_chord_distance"))
    refusals = {"admission_digest_mismatch": refusal_code(lambda: store.update(tampered, 0.5)),
                "not_retained": refusal_code(lambda: store.admit(stranger.digest(), "declared")),
                "mode_substitution": refusal_code(lambda: chord_store.admit(chord_record["observation_digest"],
                                                                             "declared"))}
    return {"per_mode": per_mode, "admission": admission, "updated_state": updated, "exact_update_gap": exact,
            "refusals": refusals}


@task("T059", changed_files=(MODULE, MODES_FILE, DOC), regression_tests=_tests("test_t059_retained_without_admission"))
def retained_without_admission(ctx):
    study = retention_study()
    per_mode = study["per_mode"]
    ctx.artifact_json("retention-study.json", study)
    unchanged = sum(v["state_unchanged"] and v["state_after_refusal_unchanged"] for v in per_mode.values())
    labelled = sum(v["admission"] == ["not_performed"] and v["retention"] == ["retained"] for v in per_mode.values())
    findings = [
        finding("Every observation mode can be retained without changing estimator state; retained records carry "
                "state_admission not_performed", "computational_pipeline",
                {"modes": len(per_mode), "state_unchanged": unchanged, "labelled_not_performed": labelled},
                {"checks": [_check("modes whose state digest changed on retention or refusal", len(per_mode) - unchanged,
                                   0, kind="exact_arithmetic"),
                            _check("modes whose records lack retention retained / admission not_performed",
                                   len(per_mode) - labelled, 0, kind="exact_arithmetic")]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("Updating state from a retained but unadmitted observation is refused for every mode",
                "computational_pipeline", {name: v["update_code"] for name, v in per_mode.items()},
                {"checks": [{"reference_kind": "refusal", "reference": f"update from unadmitted {name}",
                             "expected_refusal": "not_admitted", "observed_refusal": v["update_code"],
                             "passed": v["update_code"] == "not_admitted"} for name, v in per_mode.items()]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        finding("An admitted, digest-bound observation updates state by the exact Kalman formula", "numerical",
                {"mean": study["updated_state"]["mean"][0], "variance": study["updated_state"]["variance"][0]},
                {"checks": [_check("posterior mean minus 1.5 (prior 1.0/0.5, observation 2.0/0.5)",
                                   study["exact_update_gap"]["mean"], 0.0, kind="exact_arithmetic"),
                            _check("posterior variance minus 0.25", study["exact_update_gap"]["variance"], 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty=_exact("dyadic prior, observation and variances; the update is exact"),
                tolerance={"abs": 0, "rel": 0}),
        finding("Tampered, unretained and mode-substituted admissions are refused", "computational_pipeline",
                study["refusals"],
                {"checks": [{"reference_kind": "refusal", "reference": name.replace("_", " "), "expected_refusal": name,
                             "observed_refusal": code, "passed": code == name}
                            for name, code in study["refusals"].items()]},
                uncertainty=_exact(), tolerance={"abs": 0, "rel": 0}),
        _unestablished("Admission as workbench state confers authority to act on a machine", "actuator_authority",
                       "Admission is bookkeeping inside the workbench; actuator authority is decided outside it."),
    ]
    fields = _fields(
        hypothesis="Retention and admission are separate: any observation can be retained as evidence without "
                   "changing state, and only a retained, validated, admitted observation bound by content digest can "
                   "update state.",
        mathematical_model="State (mean, variance) per component; retain(o) stores (o, digest(o)) with "
                           "state_admission = not_performed; admit(digest) validates mode and references; update "
                           "applies K = P/(P + R), m' = m + K (z - m), P' = (1 - K) P only for admitted digests.",
        input_data=["One synthetic record per mode, three sequences each", "Prior 1.0 / 0.5 and observation "
                    "2.0 / 0.5 for the exact update"],
        observation_model="Synthetic records; no instrument.",
        expected_invariant="State digest unchanged by retention and by refused updates; exact posterior on dyadic "
                           "numbers.",
        experiment="Retain records of all seven modes, attempt updates before admission, admit and update one "
                   "record, then tamper with it, admit an unretained digest and admit a chord into a geodesic-"
                   "distance store.",
        numerical_result=f"{unchanged}/{len(per_mode)} modes unchanged by retention; all unadmitted updates refused; "
                         f"posterior {study['updated_state']['mean'][0]} / {study['updated_state']['variance'][0]}; "
                         f"refusals {study['refusals']}.",
        uncertainty="Exact; no stochastic component.",
        failure_modes_checked=["update before admission (every mode)", "record altered after admission",
                               "admission of an unretained digest", "camera chord admitted as intrinsic distance"],
        unresolved_assumptions=["Admission decisions are declared, not reviewed by an operator",
                                "State is per-component independent (no cross-covariance)"],
        recommended_next_task="T060: build the deterministic multi-sensor synthetic bench")
    return {"state": "completed", "fields": fields, "findings": findings}
