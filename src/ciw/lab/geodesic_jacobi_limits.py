"""Geodesic/Jacobi experiments, part 2: limits, invariance and counterexamples (T010-T018).

Scope: near-focus and post-focus behavior of the first-order (Jacobi)
separation prediction, invariance of geodesic and Jacobi results under
nonlinear chart changes, ambient rotations and tangent-basis changes, the
flat/developable limit, reversal and truncation of numerical geodesics,
long-horizon drift of first integrals, the cost of strongly negative
curvature, validity domains of the first-order approximation, and a
resolvability comparison of curvature signal against integrator error. All
experiments use the ``ciw.lab`` geometry core (``surfaces``, ``integrators``,
``jacobi``) on declared test surfaces; helpers live in
``geodesic_jacobi_limits_core``.

Non-claims: every surface, length, curvature and perturbation is in declared
normalized units. Nothing here measures a physical surface, calibrates a
sensor or certifies an operating envelope; findings that invite such
conclusions are recorded in their physical or authority domain, where they
are ``not_established``. Self-consistency between ``ciw`` computations is
reported as numerical verification against closed forms, invariants or
self-convergence, never as independent verification.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import integrators, jacobi, svg
from . import geodesic_jacobi_limits_core as core
from .evidence import finding
from .registry import task
from .surfaces import (Cylinder, GaussianBump, HyperbolicPlane, Plane, Reparametrized, Rotated, Saddle, Sphere,
                       SurfaceRefusal, Torus, rotation_matrix)

MODULE = "src/ciw/lab/geodesic_jacobi_limits.py"
CORE = "src/ciw/lab/geodesic_jacobi_limits_core.py"
DOC = "docs/lab/GEODESIC_JACOBI_LIMITS.md"
TESTS = "tests/test_lab_geodesic_jacobi_limits.py"
CHANGED = (MODULE, CORE, DOC)

# Heading perturbations (radians) used for every separation-scaling fit.
EPS = (0.005, 0.01, 0.02, 0.04)
# Relative first-order tolerance defining the validity domain.
TAU = 0.01

# Declared test geodesics: (start chart point, heading from the first coordinate direction).
SPHERE_START = ((math.pi / 2, 0.0), 1.0)          # great circle, theta stays in [1.0, 2.14]
TORUS_PATHS = {"equator": ((0.0, 0.0), 0.0),      # outer equator of Torus(2, 1): K = 1/3, symmetric
               "generic": ((0.0, 0.3), 0.5)}      # oscillates about the outer equator, no symmetry
TORUS_M, TORUS_STEPS = 400, 580                   # node TORUS_M is the conjugate point; grid runs to 1.45 s*
SPHERE_STEPS = 384                                # h = pi / 256, grid [0, 1.5 pi]


def _gen(name: str, **params) -> dict:
    return {"name": f"ciw.lab.geodesic_jacobi_limits:{name}", "seed": None, "deterministic": True, **params}


def _derivation(anchor: str) -> str:
    return f"{DOC}#{anchor}"


def _fields(**values) -> dict:
    required = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                "experiment", "numerical_result", "uncertainty", "failure_modes_checked",
                "unresolved_assumptions", "recommended_next_task")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Report fields missing: {missing}")
    return values


def _g(value, digits=6) -> str:
    return format(float(value), f".{digits}g")


# ------------------------------------------------------------------ shared runs
def sphere_family(ctx, lateral_ratio: float):
    """Numerical great-circle families on the unit sphere, eps = 0.02 (pure heading or lateral = heading)."""
    def compute():
        sphere = Sphere(1.0)
        u0, heading = SPHERE_START
        return core.separation_family(sphere, u0, heading, 1.5 * math.pi, SPHERE_STEPS, (0.02,), lateral_ratio)
    return ctx.memo(("gjl-sphere-family", lateral_ratio), compute)


def torus_family(ctx, key: str):
    """Torus(2, 1) heading families on a grid whose node TORUS_M is the conjugate point s*."""
    def compute():
        torus = Torus(2.0, 1.0)
        u0, heading = TORUS_PATHS[key]
        if key == "equator":
            s_star = math.pi * math.sqrt(torus.minor * (torus.major + torus.minor))
        else:
            probe = jacobi.transfer(torus, u0, heading, 8.0, steps=1024)
            s_star = probe.conjugate_points()[0]
        family = core.separation_family(torus, u0, heading, s_star * TORUS_STEPS / TORUS_M, TORUS_STEPS, EPS)
        family["s_star"] = s_star
        family["located"] = family["transfer"].conjugate_points()[0]
        return family
    return ctx.memo(("gjl-torus-family", key), compute)


def torus_family_coarse(ctx, key: str):
    """The same torus family on the same length with half the steps: fine node 2i is coarse node i."""
    def compute():
        fine = torus_family(ctx, key)
        u0, heading = TORUS_PATHS[key]
        return core.separation_family(Torus(2.0, 1.0), u0, heading, fine["length"], TORUS_STEPS // 2, EPS)
    return ctx.memo(("gjl-torus-family-coarse", key), compute)


def grid_halving_change(ctx, key: str, eps: float, nodes, series: str = "signed") -> float:
    """Largest |fine - coarse| of one separation series over fine nodes (all even) when the grid is halved."""
    fine, coarse = torus_family(ctx, key), torus_family_coarse(ctx, key)
    if any(i % 2 for i in nodes):
        raise ValueError("Grid-halving comparisons need even fine nodes")
    return max(float(abs(fine["runs"][eps][series][i] - coarse["runs"][eps][series][i // 2])) for i in nodes)


def equator_grid_check(ctx):
    """Smallest separation (equator, eps = 0.005) at s* on the 400-step grid and on a grid with half the steps."""
    def compute():
        fine, coarse = torus_family(ctx, "equator"), torus_family_coarse(ctx, "equator")
        a, b = coarse["runs"][EPS[0]]["chord"][TORUS_M // 2], fine["runs"][EPS[0]]["chord"][TORUS_M]
        return {"absolute": float(abs(a - b)), "relative": float(abs(a / b - 1.0))}
    return ctx.memo("gjl-equator-grid", compute)


def _divergence(family, eps, offsets=(1, 2, 4, 8), side=-1):
    """Relative first-order error |signed - eps j| / |eps j| at nodes s* + side m h (side -1: before s*)."""
    s, j = family["s"], family["j"]
    rows = []
    for m in offsets:
        i = TORUS_M + side * m
        predicted = eps * j[i]
        rows.append({"offset_steps": m, "distance": float(abs(family["s_star"] - s[i])),
                     "relative_error": float(abs(family["runs"][eps]["signed"][i] - predicted) / abs(predicted))})
    return rows


def _torus_summary(family):
    i = TORUS_M
    chord = [float(family["runs"][e]["chord"][i]) for e in EPS]
    signed = [float(family["runs"][e]["signed"][i]) for e in EPS]
    rate = abs(float(family["dj"][i]))
    window = [abs(sg) / (e * rate) for e, sg in zip(EPS, signed)]
    return {"s_star": float(family["s_star"]), "located_conjugate_point": float(family["located"]),
            "j_at_s_star": float(family["j"][i]), "dj_at_s_star": float(family["dj"][i]),
            "first_order_at_s_star": max(abs(e * float(family["j"][i])) for e in EPS),
            "chord_at_s_star": chord, "signed_at_s_star": signed,
            "chord_exponent": core.loglog_slope(EPS, chord),
            "signed_exponent": core.loglog_slope(EPS, [abs(v) for v in signed]),
            "invalid_window_half_width": window, "window_exponent": core.loglog_slope(EPS, window)}


# ------------------------------------------------------------------ T010
@task("T010", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t010_near_focus_and_post_focus_counterexamples",))
def near_focus_counterexamples(ctx):
    equator, generic = torus_family(ctx, "equator"), torus_family(ctx, "generic")
    eq, gen = _torus_summary(equator), _torus_summary(generic)
    grid = equator_grid_check(ctx)
    grid_difference = grid["relative"]
    torus = Torus(2.0, 1.0)
    analytic_s_star = math.pi * math.sqrt(torus.minor * (torus.major + torus.minor))

    # Divergence of the relative first-order error approaching s* (generic path, eps = 0.01).
    divergence = _divergence(generic, 0.01)
    divergence_slope = core.loglog_slope([r["distance"] for r in divergence], [r["relative_error"] for r in divergence])
    divergence_spread = core.slope_spread([r["distance"] for r in divergence], [r["relative_error"] for r in divergence])
    divergence_eq = _divergence(equator, 0.04)
    divergence_eq_slope = core.loglog_slope([r["distance"] for r in divergence_eq],
                                            [r["relative_error"] for r in divergence_eq])
    witness_rel = _divergence(generic, 0.04, offsets=(1,))[0]
    # Post-focus recovery: the same relative error at s* + m h, falling again like 1/|s - s*|.
    recovery = _divergence(generic, 0.01, side=1)
    recovery_slope = core.loglog_slope([r["distance"] for r in recovery], [r["relative_error"] for r in recovery])
    recovery_spread = core.slope_spread([r["distance"] for r in recovery], [r["relative_error"] for r in recovery])
    recovery_eq = _divergence(equator, 0.04, side=1)
    recovery_eq_slope = core.loglog_slope([r["distance"] for r in recovery_eq],
                                          [r["relative_error"] for r in recovery_eq])

    # Post-focus inversion: signed separation follows eps j through its sign change.
    before, after = TORUS_M // 2, int(1.25 * TORUS_M)
    inversion = {}
    for key, fam in (("equator", equator), ("generic", generic)):
        signed = fam["runs"][0.01]["signed"]
        inversion[key] = {"s_before": float(fam["s"][before]), "s_after": float(fam["s"][after]),
                          "j_before": float(fam["j"][before]), "j_after": float(fam["j"][after]),
                          "signed_before": float(signed[before]), "signed_after": float(signed[after]),
                          "ratio_after": float(signed[after] / (0.01 * fam["j"][after]))}
    # Grid halving measured for the configurations and nodes each finding compares.
    inversion_grid = max(grid_halving_change(ctx, key, 0.01, (before, after)) for key in ("equator", "generic"))
    late = generic["runs"][0.04]["signed"][after]
    late_rel = float(abs(late - 0.04 * generic["j"][after]) / abs(0.04 * generic["j"][after]))
    early_rel = _divergence(generic, 0.04, offsets=(1,), side=1)[0]
    monotone_ratio = float(equator["runs"][0.02]["chord"][TORUS_M] / equator["runs"][0.02]["chord"][TORUS_M // 2])
    coarse_eq = torus_family_coarse(ctx, "equator")
    monotone_coarse = float(coarse_eq["runs"][0.02]["chord"][TORUS_M // 2]
                            / coarse_eq["runs"][0.02]["chord"][TORUS_M // 4])
    monotone_grid = abs(monotone_coarse / monotone_ratio - 1.0)

    # Unit sphere: numerical families against closed-form great circles.
    sphere = Sphere(1.0)
    u0, heading = SPHERE_START
    pure, mixed = sphere_family(ctx, 0.0), sphere_family(ctx, 1.0)
    s = pure["s"]
    closed_pure = core.sphere_separation(sphere, u0, heading, s, 0.02, 0.0)
    closed_mixed = core.sphere_separation(sphere, u0, heading, s, 0.02, 1.0)
    sphere_closed_error = max(float(np.max(np.abs(pure["runs"][0.02]["chord"] - closed_pure["chord"]))),
                              float(np.max(np.abs(pure["runs"][0.02]["signed"] - closed_pure["signed"]))),
                              float(np.max(np.abs(mixed["runs"][0.02]["chord"] - closed_mixed["chord"]))),
                              float(np.max(np.abs(mixed["runs"][0.02]["signed"] - closed_mixed["signed"]))))
    uniform = 2.0 * math.sin(0.01) / 0.02 - 1.0
    interior = [i for i in range(1, len(s)) if i != 256]   # exclude s = 0 and the conjugate node s = pi
    rel_pure = [pure["runs"][0.02]["chord"][i] / (0.02 * abs(pure["j"][i])) - 1.0 for i in interior]
    sphere_uniform_dev = float(max(abs(v - uniform) for v in rel_pure))
    near_pi = float(pure["runs"][0.02]["chord"][255] / (0.02 * abs(pure["j"][255])) - 1.0)
    sphere_flip = float(pure["runs"][0.02]["signed"][320] / pure["runs"][0.02]["signed"][192])
    s0 = 0.75 * math.pi
    mixed_chord = [float(core.sphere_separation(sphere, u0, heading, [s0], e, 1.0)["chord"][0]) for e in EPS]
    mixed_exponent = core.loglog_slope(EPS, mixed_chord)
    mixed_coefficient = mixed_chord[0] / EPS[0] ** 2
    mixed_numeric = float(mixed["runs"][0.02]["chord"][192])

    table = {"torus_equator": eq, "torus_generic": gen, "divergence_generic_eps_0.01": divergence,
             "divergence_equator_eps_0.04": divergence_eq, "recovery_generic_eps_0.01": recovery,
             "recovery_equator_eps_0.04": recovery_eq, "inversion": inversion,
             "grid_halving": {"equator_smallest_chord_at_s_star": grid, "inversion_signed_max_abs": inversion_grid,
                              "equator_chord_ratio_relative": monotone_grid},
             "equator_chord_ratio_s_star_over_half": monotone_ratio,
             "sphere": {"closed_form_max_error": sphere_closed_error, "uniform_relative_error": uniform,
                        "max_deviation_from_uniform": sphere_uniform_dev, "relative_error_at_pi_minus_h": near_pi,
                        "signed_ratio_5pi4_over_3pi4": sphere_flip, "mixed_chord_at_3pi4": mixed_chord,
                        "mixed_exponent": mixed_exponent, "mixed_coefficient": mixed_coefficient,
                        "mixed_numeric_eps_0.02": mixed_numeric}}
    ctx.artifact_json("near-focus.json", core.jsonable(table, 12))
    series = []
    for label, fam in (("torus equator", equator), ("torus generic", generic)):
        rel = np.abs(fam["runs"][0.01]["signed"] - 0.01 * fam["j"]) / np.maximum(np.abs(0.01 * fam["j"]), 1e-300)
        keep = [i for i in range(1, len(fam["s"])) if i != TORUS_M]
        series.append((label, [fam["s"][i] / fam["s_star"] for i in keep], [rel[i] for i in keep]))
    ctx.artifact_text("relative-first-order-error.svg", svg.line_plot(
        series, title="Relative first-order error, eps = 0.01", xlabel="s / s* (conjugate point at 1)",
        ylabel="|d - eps j| / |eps j|", logy=True, markers=False))
    ctx.artifact_text("separation-at-conjugate-point.svg", svg.line_plot(
        [("torus equator (chord)", EPS, eq["chord_at_s_star"]), ("torus generic (chord)", EPS, gen["chord_at_s_star"]),
         ("sphere lateral+heading at 3pi/4", EPS, mixed_chord)],
        title="Separation where the first-order prediction vanishes", xlabel="eps", ylabel="chord", logx=True, logy=True))

    findings = [
        finding("On the torus outer equator the separation at the conjugate point scales as eps^3, not eps^2",
                "numerical", {"chord_exponent": eq["chord_exponent"], "signed_exponent": eq["signed_exponent"],
                              "chord_at_s_star": eq["chord_at_s_star"], "grid_relative_difference": grid_difference},
                {"generator": _gen("torus-equator-family", eps=list(EPS), steps=TORUS_STEPS),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "numerical conjugate point minus pi sqrt(r (R + r))",
                                       eq["located_conjugate_point"] - analytic_s_star, 1e-7),
                            core.check("analytic", "first-order prediction |eps j(s*)| (vanishes at s*)",
                                       eq["first_order_at_s_star"], 1e-9),
                            core.check("analytic", "fitted chord exponent minus 3 (reflection symmetry)",
                                       eq["chord_exponent"] - 3.0, 0.05),
                            core.check("self_convergence", "smallest chord at s*: 200- versus 400-step grid "
                                       "(relative)", grid_difference, 1e-3)]},
                uncertainty=core.uncertainty("truncation_bound", grid["relative"],
                                             "relative change of the smallest separation (eps = 0.005) at s* when "
                                             "the grid is halved; the 400-step error is about a sixteenth of it"),
                tolerance={"abs": 1e-10, "rel": 1e-4},
                counterexample={"statement": "The separation at a conjugate point is of exact order eps^2",
                                "witness": {"surface": "Torus(2, 1)", "path": "outer equator, heading 0",
                                            "s_star": eq["s_star"], "eps": list(EPS),
                                            "chord_exponent": eq["chord_exponent"]}}),
        finding("On a generic torus geodesic the separation at the conjugate point is O(eps^2) while eps j vanishes",
                "numerical", {"s_star": gen["s_star"], "chord_exponent": gen["chord_exponent"],
                              "chord_at_s_star": gen["chord_at_s_star"]},
                {"generator": _gen("torus-generic-family", eps=list(EPS), steps=TORUS_STEPS),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("self_convergence", "first-order prediction |eps j| on the 580-step grid at the "
                                       "conjugate point located on a separate 1024-step probe",
                                       gen["first_order_at_s_star"], 1e-8),
                            core.check("analytic", "fitted chord exponent minus 2 (generic remainder)",
                                       gen["chord_exponent"] - 2.0, 0.1)]},
                uncertainty=core.uncertainty("fit_spread", core.slope_spread(EPS, gen["chord_at_s_star"]),
                                             "largest gap between the fitted exponent and the slopes of "
                                             "consecutive eps pairs"),
                tolerance={"abs": 1e-6, "rel": 1e-5}),
        finding("The relative first-order error diverges like 1/|s - s*| approaching the conjugate point",
                "numerical", {"slope_generic_eps_0.01": divergence_slope, "slope_equator_eps_0.04": divergence_eq_slope,
                              "relative_error_at_s_star_minus_h_eps_0.04": witness_rel["relative_error"]},
                {"generator": _gen("torus-divergence", offsets=[1, 2, 4, 8]),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "generic log-log slope plus 1", divergence_slope + 1.0, 0.15),
                            core.check("analytic", "equator log-log slope plus 1", divergence_eq_slope + 1.0, 0.2),
                            core.check("invariant", "relative first-order error one step before s* (eps = 0.04)",
                                       witness_rel["relative_error"], 1.0, "ge")]},
                uncertainty=core.uncertainty("fit_spread", divergence_spread,
                                             "largest gap between the fitted slope and consecutive-offset slopes "
                                             "(generic path)"),
                tolerance={"abs": 1e-3, "rel": 1e-4},
                counterexample={"statement": "The first-order separation eps j is accurate along the whole path "
                                             "once eps is small",
                                "witness": {"surface": "Torus(2, 1)", "path": "start (0, 0.3), heading 0.5",
                                            "eps": 0.04, "s": gen["s_star"] - witness_rel["distance"],
                                            "relative_error": witness_rel["relative_error"],
                                            "invalid_window_half_width_by_eps": dict(zip([str(e) for e in EPS],
                                                                                         gen["invalid_window_half_width"])),
                                            "window_exponent": gen["window_exponent"]}}),
        finding("After the conjugate point the separation inverts sign and still follows eps j",
                "numerical", inversion,
                {"generator": _gen("torus-inversion", eps=0.01),
                 "checks": [core.check("invariant", "generic: signed / (eps j) after s* minus 1",
                                       inversion["generic"]["ratio_after"] - 1.0, 0.05),
                            core.check("invariant", "equator: signed / (eps j) after s* minus 1",
                                       inversion["equator"]["ratio_after"] - 1.0, 0.05),
                            core.check("invariant", "sign product of separations before and after s* plus 1 (generic)",
                                       math.copysign(1.0, inversion["generic"]["signed_before"])
                                       * math.copysign(1.0, inversion["generic"]["signed_after"]) + 1.0, 0.0),
                            core.check("invariant", "sign product of separations before and after s* plus 1 (equator)",
                                       math.copysign(1.0, inversion["equator"]["signed_before"])
                                       * math.copysign(1.0, inversion["equator"]["signed_after"]) + 1.0, 0.0)]},
                uncertainty=core.uncertainty("truncation_bound", inversion_grid,
                                             "largest absolute change of the compared signed separations (eps = "
                                             "0.01, s*/2 and 1.25 s*, both paths) when the grid is halved; the "
                                             "separations compared here are at least "
                                             + _g(min(abs(v[name]) for v in inversion.values()
                                                      for name in ("signed_before", "signed_after")), 2)),
                tolerance={"abs": 1e-9, "rel": 1e-5},
                counterexample={"statement": "Neighbouring geodesics stay on the side of the base geodesic they start "
                                             "on (the signed separation keeps its sign along the path)",
                                "witness": {"surface": "Torus(2, 1)", "eps": 0.01,
                                            "paths": {key: {"s_before": v["s_before"], "s_after": v["s_after"],
                                                            "s_star": float(fam["s_star"]),
                                                            "signed_before": v["signed_before"],
                                                            "signed_after": v["signed_after"]}
                                                      for (key, v), fam in zip(inversion.items(), (equator, generic))}}}),
        finding("Past the conjugate point the first-order prediction recovers: the relative error falls again like "
                "1/|s - s*|",
                "numerical", {"slope_after_generic_eps_0.01": recovery_slope,
                              "slope_after_equator_eps_0.04": recovery_eq_slope,
                              "relative_error_at_s_star_plus_h_eps_0.04": early_rel["relative_error"],
                              "relative_error_at_1.25_s_star_eps_0.04": late_rel},
                {"generator": _gen("torus-recovery", offsets=[1, 2, 4, 8]),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "generic log-log slope after s* plus 1", recovery_slope + 1.0, 0.15),
                            core.check("analytic", "equator log-log slope after s* plus 1", recovery_eq_slope + 1.0,
                                       0.2),
                            core.check("invariant", "relative first-order error one step after s* (eps = 0.04)",
                                       early_rel["relative_error"], 1.0, "ge"),
                            core.check("invariant", "relative first-order error at 1.25 s* (generic, eps = 0.04)",
                                       late_rel, 0.05, "le")]},
                uncertainty=core.uncertainty("fit_spread", recovery_spread,
                                             "largest gap between the fitted slope and consecutive-offset slopes "
                                             "after s* (generic path)"),
                tolerance={"abs": 1e-3, "rel": 1e-4},
                counterexample={"statement": "Once the first-order prediction has failed at a conjugate point it stays "
                                             "invalid beyond it",
                                "witness": {"surface": "Torus(2, 1)", "path": "start (0, 0.3), heading 0.5",
                                            "eps": 0.04, "s_star": gen["s_star"],
                                            "s_after_h": gen["s_star"] + early_rel["distance"],
                                            "relative_error_after_h": early_rel["relative_error"],
                                            "s_late": inversion["generic"]["s_after"],
                                            "relative_error_late": late_rel}}),
        finding("Separation does not grow monotonically with length: it nearly vanishes at the conjugate point",
                "numerical", monotone_ratio,
                {"generator": _gen("torus-equator-family", eps=0.02),
                 "checks": [core.check("invariant", "chord(s*) / chord(s*/2) on the outer equator",
                                       monotone_ratio, 1e-3, "le")]},
                uncertainty=core.uncertainty("truncation_bound", monotone_grid,
                                             "relative change of chord(s*) / chord(s*/2) (eps = 0.02, equator) when "
                                             "the grid is halved"),
                tolerance={"abs": 1e-9, "rel": 1e-4},
                counterexample={"statement": "The separation of neighbouring geodesics grows monotonically with length",
                                "witness": {"surface": "Torus(2, 1)", "path": "outer equator", "eps": 0.02,
                                            "s_half": float(equator["s"][TORUS_M // 2]), "s_star": eq["s_star"],
                                            "chord_ratio": monotone_ratio}}),
        finding("On the unit sphere a pure heading perturbation refocuses exactly: the relative first-order error "
                "is uniform and does not diverge at the conjugate point",
                "numerical", {"uniform_relative_error": uniform, "relative_error_at_pi_minus_h": near_pi,
                              "max_deviation_from_uniform": sphere_uniform_dev},
                {"generator": _gen("sphere-great-circles", eps=0.02, steps=SPHERE_STEPS),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "numerical chord and signed separation versus great circles",
                                       sphere_closed_error, 1e-9),
                            core.check("analytic", "relative error minus 2 sin(eps/2)/eps - 1 at every node",
                                       sphere_uniform_dev, 1e-6)]},
                uncertainty=core.uncertainty("reference_error", sphere_closed_error,
                                             "max deviation of the numerical separations from exact great circles"),
                tolerance={"abs": 1e-9, "rel": 1e-4},
                counterexample={"statement": "The relative first-order error diverges at every conjugate point",
                                "witness": {"surface": "unit sphere", "perturbation": "pure heading",
                                            "eps": 0.02, "s": float(s[255]), "relative_error": near_pi}}),
        finding("On the unit sphere the image inverts after the conjugate point (signed ratio -1)",
                "numerical", sphere_flip,
                {"generator": _gen("sphere-great-circles", eps=0.02),
                 "checks": [core.check("analytic", "signed separation at 5pi/4 over 3pi/4 plus 1", sphere_flip + 1.0, 1e-8)]},
                uncertainty=core.uncertainty("reference_error",
                                             sphere_closed_error / abs(float(pure["runs"][0.02]["signed"][192])),
                                             "numerical-versus-great-circle deviation relative to the signed "
                                             "separation at 3pi/4"),
                tolerance={"abs": 1e-8, "rel": 0.0},
                counterexample={"statement": "The image of a family of geodesics keeps its orientation beyond a "
                                             "conjugate point",
                                "witness": {"surface": "unit sphere", "perturbation": "pure heading", "eps": 0.02,
                                            "s_before": float(s[192]), "s_after": float(s[320]),
                                            "signed_before": float(pure["runs"][0.02]["signed"][192]),
                                            "signed_after": float(pure["runs"][0.02]["signed"][320]),
                                            "signed_ratio": sphere_flip}}),
        finding("A combined lateral+heading perturbation on the sphere has true separation eps^2/2 where eps j vanishes",
                "numerical", {"exponent": mixed_exponent, "coefficient": mixed_coefficient, "s0": s0},
                {"generator": _gen("sphere-lateral-heading", lateral_ratio=1.0, eps=list(EPS)),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "closed-form exponent minus 2", mixed_exponent - 2.0, 0.01),
                            core.check("analytic", "chord / eps^2 at s0 = 3pi/4 minus 1/2", mixed_coefficient - 0.5, 1e-3),
                            core.check("analytic", "numerical chord at s0 minus closed form (eps = 0.02)",
                                       mixed_numeric - float(closed_mixed["chord"][192]), 1e-9)]},
                uncertainty=core.uncertainty("fit_spread", core.slope_spread(EPS, mixed_chord),
                                             "largest gap between the fitted exponent and consecutive-pair slopes "
                                             "of the closed form"),
                tolerance={"abs": 1e-6, "rel": 1e-6}),
    ]
    fields = _fields(
        hypothesis=("Where the first-order separation eps j(s) vanishes (conjugate or focal points) the true "
                    "separation is of higher order in eps, so the relative first-order error diverges there; past "
                    "the zero the separation changes sign (image inversion) and the first-order prediction recovers "
                    "as |j| grows again; symmetric configurations raise the order of the true separation."),
        mathematical_model=("Unit-speed geodesics with j'' + K j = 0; d(s; eps) = eps j(s) + r(eps, s). At s* with "
                            "j(s*) = 0, d(s*) = r(eps, s*). Torus(2, 1) outer equator: K = 1/3, s* = pi sqrt(3); a "
                            "reflection theta -> -theta makes the normal offset odd in eps and the O(eps^2) along-track "
                            "lag -eps^2 sin(2 w s)/(4 w) vanish at s*, so d(s*) = O(eps^3). Unit sphere: all great "
                            "circles through a point refocus exactly at s = pi; the combined perturbation "
                            "(lateral eps, heading eps) has j = eps (cos s + sin s) with zero at 3pi/4 where the exact "
                            "chord is eps^2/2 + O(eps^3)."),
        input_data=["Torus(2, 1): outer equator start (0, 0) heading 0; generic start (0, 0.3) heading 0.5",
                    f"eps = {list(EPS)} rad; RK4 grid with node {TORUS_M} at s*, {TORUS_STEPS} steps",
                    "Unit sphere: start (pi/2, 0) heading 1.0; grid h = pi/256 to 1.5 pi; eps = 0.02"],
        observation_model=("Embedded chord |X_eps(s) - X(s)| and its signed component along the base geodesic's "
                           "embedded in-surface normal, compared at matched arclength nodes; no renormalization."),
        expected_invariant=("d(s*) = O(eps^2) generically and O(eps^3) on the symmetric equator; relative error "
                            "~ 1/|s - s*| on both sides of s*; sign(d) = sign(j) away from s*; sphere relative chord "
                            "error 2 sin(eps/2)/eps - 1 at every s."),
        experiment=("Integrate base transfer and perturbed geodesics on common grids (RK4), locate s* from j_head, "
                    "fit exponents of d(s*) against eps, fit the divergence slope over s* - m h and the recovery slope "
                    "over s* + m h (m = 1, 2, 4, 8), check sign inversion after s*, compare sphere runs with "
                    "closed-form great circles, and repeat the torus families on a grid with half the steps to "
                    "measure the grid dependence of every compared node."),
        numerical_result=(f"Equator: s* = {_g(eq['s_star'])}, chord(s*) exponent {_g(eq['chord_exponent'], 4)}; "
                          f"generic: s* = {_g(gen['s_star'])}, exponent {_g(gen['chord_exponent'], 4)}; divergence "
                          f"slope {_g(divergence_slope, 4)} (generic), {_g(divergence_eq_slope, 4)} (equator); "
                          f"relative error {_g(witness_rel['relative_error'], 4)} one step before s* at eps = 0.04; "
                          f"recovery slope after s* {_g(recovery_slope, 4)} (generic), {_g(recovery_eq_slope, 4)} "
                          f"(equator), relative error {_g(early_rel['relative_error'], 4)} one step after s* and "
                          f"{_g(late_rel, 3)} at 1.25 s* (eps = 0.04); "
                          f"after s* signed/(eps j) = {_g(inversion['generic']['ratio_after'], 5)} with j < 0; "
                          f"chord(s*)/chord(s*/2) = {_g(monotone_ratio, 3)}; sphere relative error uniform "
                          f"{_g(uniform, 4)} (max deviation {_g(sphere_uniform_dev, 2)}); sphere combined perturbation "
                          f"chord(3pi/4) = {_g(mixed_coefficient, 5)} eps^2."),
        uncertainty=(f"Halving the grid changes the smallest separation (equator, eps = {EPS[0]}, chord "
                     f"{_g(eq['chord_at_s_star'][0], 2)}) by a relative {_g(grid_difference, 2)}, the signed "
                     f"separations compared for the inversion (eps = 0.01, s*/2 and 1.25 s*, both paths) by at most "
                     f"{_g(inversion_grid, 2)} absolute, and chord(s*)/chord(s*/2) by a relative "
                     f"{_g(monotone_grid, 2)}; the finer-grid values are accurate to about a sixteenth of these "
                     f"changes. Exponents are least-squares fits over eps in [{EPS[0]}, {EPS[-1]}] and include "
                     "higher-order terms (largest gap to consecutive-pair slopes "
                     f"{_g(core.slope_spread(EPS, gen['chord_at_s_star']), 2)} for the generic path)."),
        failure_modes_checked=["numerical conjugate point versus pi sqrt(3) on the equator",
                               "equator: first-order prediction at the analytic s* = pi sqrt(3) is at integrator "
                               "level; generic path: s* is located numerically on a separate 1024-step probe, so its "
                               "vanishing first-order prediction on the 580-step grid is a self-convergence check of "
                               "the zero location, not an analytic one",
                               "grid dependence measured for each configuration and node compared, not borrowed "
                               "from another configuration",
                               "sphere numerical separations versus exact great circles at every node",
                               "exact-refocusing (sphere) and symmetric (equator) cases that do not show eps^2",
                               "sign inversion checked with the first-order field itself, not assumed",
                               "post-focus behaviour recorded as counterexamples witnessed after s* (side change, "
                               "image inversion, recovery of the first-order prediction)"],
        unresolved_assumptions=["Chord separation is extrinsic; intrinsic distance differs at O(d^3) and would "
                                "change only third-order coefficients",
                                "The generic torus path is one geodesic; genericity is argued, not sampled",
                                "Heading perturbations only for the torus; lateral families are not fitted there"],
        recommended_next_task="T017 (validity domains, which tabulates C(s) for these families) and T008 "
                              "(conjugate-point location accuracy)",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T011
T011_CASES = (("plane", (0.3, -0.2), 0.7, 2.0), ("sphere", SPHERE_START[0], SPHERE_START[1], 2.0),
              ("torus", (0.0, 0.5), 0.7, 3.0))
T011_STEPS = (64, 128, 256)


def _t011_base(key):
    return {"plane": Plane(), "sphere": Sphere(1.0), "torus": Torus(2.0, 1.0)}[key]


def _t011_reference(key, u0, heading, length):
    """Endpoint X(L) and transfer matrix (j_lat, j_lat', j_head, j_head') of the base-chart geodesic."""
    base = _t011_base(key)
    u0 = np.asarray(u0, dtype=float)
    t0 = base.unit_tangent(u0, heading)
    if key == "plane":
        return np.concatenate([base.embedding(u0 + length * t0), [1.0, 0.0, length, 1.0]]), 0.0, "analytic"
    if key == "sphere":
        point = base.exact_embedded_geodesic(u0, t0, [length])[0]
        return (np.concatenate([point, [math.cos(length), -math.sin(length), math.sin(length), math.cos(length)]]),
                0.0, "analytic")
    fine = jacobi.transfer(base, u0, heading, length, rtol=1e-13, atol=1e-15)
    coarse = jacobi.transfer(base, u0, heading, length, rtol=1e-12, atol=1e-14)
    ref = np.concatenate([base.embedding(fine.states[-1, :2]), fine.states[-1, 4:8]])
    other = np.concatenate([base.embedding(coarse.states[-1, :2]), coarse.states[-1, 4:8]])
    # The same DP45 integrator at a tighter tolerance: a self-convergence reference, not an independent one.
    return ref, float(np.max(np.abs(ref - other))), "self_convergence"


def _chart_run(surface, a0, ta, length, steps=None, rtol=None):
    y0 = np.concatenate([a0, ta, [1.0, 0.0, 0.0, 1.0]])
    f = jacobi.rhs(surface)
    if rtol is not None:
        s, states, stats = integrators.integrate_adaptive(f, y0, length, rtol=rtol, atol=rtol * 1e-2)
    else:
        s, states = integrators.integrate_fixed(f, y0, length, steps, "rk4")
        stats = {"steps": steps}
    speed = np.sqrt([surface.speed_squared(y[:2], y[2:4]) for y in states])
    arclength = float(np.sum(0.5 * (speed[1:] + speed[:-1]) * np.diff(s)))
    observed = np.concatenate([surface.embedding(states[-1, :2]), states[-1, 4:8]])
    return observed, arclength - length, stats, states


def chart_study(ctx):
    """Every chart on every T011 surface: fixed-step errors, adaptive errors and the identity-chart bit check."""
    def compute():
        out = {}
        for key, u0, heading, length in T011_CASES:
            base = _t011_base(key)
            ref, ref_spread, ref_kind = _t011_reference(key, u0, heading, length)
            a0, ta = core.chart_start(base, None, u0, heading)
            _, _, _, base_states = _chart_run(base, a0, ta, length, steps=T011_STEPS[-1])
            middle = base_states[len(base_states) // 2, :2]
            center = np.round(middle, 2)
            axis = int(np.argmax(np.abs(base_states[-1, :2] - base_states[0, :2])))
            charts = {"base": None, "polynomial-warp": core.PolynomialWarp(center, 0.3),
                      "quadratic-shear": core.QuadraticShear(center, 0.8),
                      "exponential-stretch": core.ExponentialStretch(center, 0.6),
                      "near-fold mu=0.2": core.NearFold(center, 0.2, axis),
                      "near-fold mu=0.1": core.NearFold(center, 0.1, axis)}
            rows = {}
            for name, chart in charts.items():
                surface = base if chart is None else Reparametrized(base, chart)
                a0, ta = core.chart_start(base, chart, u0, heading)
                errors, length_errors = [], []
                for steps in T011_STEPS:
                    observed, length_error, _, _ = _chart_run(surface, a0, ta, length, steps=steps)
                    errors.append(float(np.max(np.abs(observed - ref))))
                    length_errors.append(abs(length_error))
                adaptive, adaptive_length, stats, _ = _chart_run(surface, a0, ta, length, rtol=1e-10)
                rows[name] = {"fixed_errors": errors, "fixed_length_errors": length_errors,
                              # Orders from step pairs; roundoff-level errors carry no order.
                              "order_first_pair": -core.loglog_slope(T011_STEPS[:2], errors[:2])
                              if min(errors[:2]) > 1e-13 else None,
                              "order_last_pair": -core.loglog_slope(T011_STEPS[-2:], errors[-2:])
                              if min(errors[-2:]) > 1e-13 else None,
                              "adaptive_error": float(np.max(np.abs(adaptive - ref))),
                              "adaptive_length_error": abs(adaptive_length),
                              "adaptive_accepted_steps": stats["accepted_steps"]}
            identity = Reparametrized(base, core.IdentityChart())
            a0, ta = core.chart_start(base, core.IdentityChart(), u0, heading)
            _, _, _, id_states = _chart_run(identity, a0, ta, length, steps=T011_STEPS[0])
            b0, tb = core.chart_start(base, None, u0, heading)
            _, _, _, direct = _chart_run(base, b0, tb, length, steps=T011_STEPS[0])
            out[key] = {"center": center.tolist(), "fold_axis": axis, "reference_kind": ref_kind,
                        "reference_spread": ref_spread, "charts": rows,
                        "identity_max_difference": float(np.max(np.abs(id_states - direct)))}
        return out
    return ctx.memo("gjl-chart-study", compute)


@task("T011", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t011_chart_invariance_and_fold_amplification",
                                                     f"{TESTS}::test_chart_maps_have_exact_derivatives"))
def coordinate_change_invariance(ctx):
    study = chart_study(ctx)
    ctx.artifact_json("chart-invariance.json", core.jsonable(study, 12))
    lengths = {case[0]: case[3] for case in T011_CASES}
    series = []
    for key in ("sphere", "torus"):
        for name in ("base", "polynomial-warp", "near-fold mu=0.1"):
            series.append((f"{key}: {name}", [lengths[key] / n for n in T011_STEPS],
                           study[key]["charts"][name]["fixed_errors"]))
    ctx.artifact_text("chart-error.svg", svg.line_plot(series, title="RK4 endpoint/Jacobi error by chart",
                                                       xlabel="step h = L / N", ylabel="max error",
                                                       logx=True, logy=True))
    adaptive_max = max(row["adaptive_error"] for case in study.values() for row in case["charts"].values())
    adaptive_by_kind = {kind: max(row["adaptive_error"] for case in study.values() if case["reference_kind"] == kind
                                  for row in case["charts"].values()) for kind in ("analytic", "self_convergence")}
    length_max = max(row["adaptive_length_error"] for case in study.values() for row in case["charts"].values())
    orders = {f"{key}: {name}": row["order_last_pair"] for key, case in study.items()
              for name, row in case["charts"].items() if row["order_last_pair"] is not None}
    min_order = min(orders.values())
    first_orders = {f"{key}: {name}": row["order_first_pair"] for key, case in study.items()
                    for name, row in case["charts"].items() if row["order_first_pair"] is not None}
    min_first = min(first_orders, key=first_orders.get)
    # Near-fold charts are pre-asymptotic at N <= 256 (orders above 4); smooth charts are compared with 4.
    smooth_orders = {name: v for name, v in orders.items() if "near-fold" not in name}
    fold_orders = {name: v for name, v in orders.items() if "near-fold" in name}
    smooth_order_gap = max(abs(v - 4.0) for v in smooth_orders.values())
    smooth_spread = max(abs(row["order_first_pair"] - row["order_last_pair"]) for case in study.values()
                        for name, row in case["charts"].items()
                        if "near-fold" not in name and row["order_first_pair"] is not None
                        and row["order_last_pair"] is not None)
    torus_spread = study["torus"]["reference_spread"]
    factors = {key: {name: row["fixed_errors"][-1] / study[key]["charts"]["base"]["fixed_errors"][-1]
                     for name, row in study[key]["charts"].items() if name != "base"} for key in ("sphere", "torus")}
    fold_factor = min(factors["sphere"]["near-fold mu=0.1"], factors["torus"]["near-fold mu=0.1"])
    smooth = [v for key in factors for name, v in factors[key].items() if not name.startswith("near-fold")]
    plane = study["plane"]["charts"]
    identity = max(case["identity_max_difference"] for case in study.values())
    steps_fold = {key: study[key]["charts"]["near-fold mu=0.1"]["adaptive_accepted_steps"]
                  / study[key]["charts"]["base"]["adaptive_accepted_steps"] for key in study}

    findings = [
        finding("Converged geodesic endpoints, lengths and Jacobi transfer matrices agree in every chart",
                "numerical", {"max_adaptive_error": adaptive_max, "max_length_error": length_max,
                              "by_reference": adaptive_by_kind},
                {"generator": _gen("chart-study", charts=list(study["plane"]["charts"]), rtol=1e-10),
                 "derivation": _derivation("t011-coordinate-change-invariance"),
                 "checks": [core.check("analytic", "adaptive error against exact plane and sphere geodesics",
                                       adaptive_by_kind["analytic"], 1e-8),
                            core.check("self_convergence", "adaptive error (rtol 1e-10) against the same DP45 at "
                                       "rtol 1e-13 (torus)", adaptive_by_kind["self_convergence"], 1e-8),
                            core.check("invariant", "|integral of speed - L| in every chart", length_max, 1e-8)]},
                uncertainty=core.uncertainty("reference_error", torus_spread,
                                             "torus reference spread (rtol 1e-13 against 1e-12); plane and sphere "
                                             "references are exact"),
                tolerance={"abs": 1e-9, "rel": 0.5}),
        finding("Between N = 128 and 256 RK4 error falls at least like h^3.7 in every chart; smooth charts show order "
                "4, near-fold charts are still pre-asymptotic (order above 4) at N = 256",
                "numerical", {"min_order": min_order, "smooth_max_gap_to_4": smooth_order_gap,
                              "min_fold_order": min(fold_orders.values()), "orders": orders,
                              "min_order_N_64_128": first_orders[min_first], "min_order_N_64_128_chart": min_first},
                {"generator": _gen("chart-study", steps=list(T011_STEPS)),
                 "checks": [core.check("invariant", "smallest observed order between N = 128 and 256 (all charts)",
                                       min_order, 3.7, "ge"),
                            core.check("analytic", "max |order - 4| over the smooth charts (N = 128/256)",
                                       smooth_order_gap, 0.1),
                            core.check("invariant", "smallest near-fold order between N = 128 and 256 "
                                       "(pre-asymptotic from above)", min(fold_orders.values()), 4.0, "ge")]},
                uncertainty=core.uncertainty("fit_spread", smooth_spread,
                                             "largest difference between the smooth-chart orders from N = 64/128 "
                                             "and N = 128/256; near-fold orders are not asymptotic and carry no "
                                             "order estimate"),
                tolerance={"abs": 0.05, "rel": 0.0}),
        finding("A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor",
                "numerical", {"error_factor_mu_0.1_at_N_256": {k: factors[k]["near-fold mu=0.1"] for k in factors},
                              "error_factor_mu_0.2_at_N_256": {k: factors[k]["near-fold mu=0.2"] for k in factors},
                              "smooth_chart_factor_range": [min(smooth), max(smooth)]},
                {"generator": _gen("chart-study", steps=T011_STEPS[-1]),
                 "checks": [core.check("invariant", "min over sphere and torus of error(fold 0.1)/error(base)",
                                       fold_factor, 100.0, "ge")]},
                uncertainty=core.uncertainty("reference_error",
                                             torus_spread / study["torus"]["charts"]["base"]["fixed_errors"][-1],
                                             "torus reference spread relative to the torus base-chart error at N = "
                                             "256 (relative uncertainty of the torus factor)"),
                tolerance={"abs": 1e-6, "rel": 0.02},
                counterexample={"statement": "A change of chart that preserves the geometry leaves the fixed-step "
                                             "integration error unchanged",
                                "witness": {"chart": "u_axis = c + mu a + a^3/3, mu = 0.1 (det J >= 0.1)",
                                            "steps": T011_STEPS[-1], "factor_sphere": factors["sphere"]["near-fold mu=0.1"],
                                            "factor_torus": factors["torus"]["near-fold mu=0.1"],
                                            "plane_error_base": plane["base"]["fixed_errors"][-1],
                                            "plane_error_fold": plane["near-fold mu=0.1"]["fixed_errors"][-1],
                                            "adaptive_step_ratio": steps_fold}}),
        finding("The identity chart reproduces the base-chart integration bit for bit", "computational_pipeline",
                identity, {"checks": [core.check("exact_arithmetic", "max |identity-chart state - base state|",
                                                 identity, 0.0)]},
                uncertainty=core.uncertainty("roundoff", 0.0,
                                             "bitwise comparison of identical floating-point operations"),
                tolerance={"abs": 0.0, "rel": 0.0}),
    ]
    torus_rows = study["torus"]["charts"]
    fields = _fields(
        hypothesis=("A geodesic and its Jacobi fields are geometric: the same initial point and unit tangent give the "
                    "same embedded endpoint, length and transfer matrix in any chart, up to integration error; the "
                    "size of that error, however, depends on the chart."),
        mathematical_model=("Pullback metric g'(a) = J^T g(phi(a)) J with exact Hessian terms (Reparametrized); the "
                            "initial tangent is t_a = J^{-1} t_u, so the geometric initial data coincide. RK4 local error "
                            "~ h^5 y^(5); a near fold u = c + mu a + a^3/3 makes the chart velocity ~ 1/mu over an arclength "
                            "window ~ mu^(3/2), so the derivatives entering the error grow as mu decreases."),
        input_data=["Plane start (0.3, -0.2) heading 0.7, L = 2; unit sphere start (pi/2, 0) heading 1.0, L = 2; "
                    "Torus(2, 1) start (0, 0.5) heading 0.7, L = 3",
                    "Charts centered at the rounded chart midpoint of the base path: polynomial warp beta = 0.3, "
                    "quadratic shear sigma = 0.8, exponential stretch lam = 0.6, near-fold mu = 0.2 and 0.1 on the "
                    "coordinate with the largest excursion, identity",
                    f"RK4 with N = {list(T011_STEPS)}; adaptive DP45 rtol 1e-10"],
        observation_model=("Embedded endpoint X(L) = base embedding of phi(a(L)), transfer matrix entries at L, and the "
                           "trapezoid length of sqrt(g(v, v)); error = max absolute deviation from the reference."),
        expected_invariant=("All charts agree to the reference within the integrator tolerance; asymptotic RK4 order 4 "
                            "in every chart (near-fold charts reach it only at steps finer than those used here); the "
                            "error constant is chart dependent."),
        experiment=("Integrate the joint geodesic/Jacobi system in each chart with identical geometric initial data; "
                    "compare with exact (plane, sphere) or rtol 1e-13 (torus) references; fit orders; form error "
                    "ratios chart/base at N = 256."),
        numerical_result=(f"Adaptive max error {_g(adaptive_max, 3)} over all charts; RK4 orders at N = 128/256: min "
                          f"{_g(min_order, 4)} over all charts, smooth charts within {_g(smooth_order_gap, 2)} of 4, "
                          f"near-fold charts {_g(min(fold_orders.values()), 3)}-{_g(max(fold_orders.values()), 3)} "
                          f"(pre-asymptotic); at N = 64/128 the smallest order is {_g(first_orders[min_first], 4)} "
                          f"({min_first}), so the h^3.7 bound is claimed for N = 128/256 only; "
                          f"fold mu = 0.1 multiplies the N = 256 error by {_g(factors['sphere']['near-fold mu=0.1'], 3)} "
                          f"(sphere) and {_g(factors['torus']['near-fold mu=0.1'], 3)} (torus); smooth charts change it by "
                          f"{_g(min(smooth), 3)}-{_g(max(smooth), 3)}x; the plane is exact to "
                          f"{_g(plane['base']['fixed_errors'][-1], 2)} in its base chart but has error "
                          f"{_g(plane['near-fold mu=0.1']['fixed_errors'][-1], 3)} in the fold chart; the adaptive "
                          f"integrator spends {_g(steps_fold['torus'], 3)}x the torus base steps in the fold chart; "
                          f"identity chart max difference {_g(identity, 2)}."),
        uncertainty=(f"Torus reference spread (rtol 1e-13 vs 1e-12) {_g(study['torus']['reference_spread'], 2)}; torus "
                     f"base error at N = 256 is {_g(torus_rows['base']['fixed_errors'][-1], 2)}, so torus factors are "
                     "reliable to a few percent; orders are two-point estimates."),
        failure_modes_checked=["chart inverse reproduces the start point (refused otherwise)",
                               "exponential chart domain (refused outside u > c - 1/lam)",
                               "near-fold kept strictly regular (mu > 0; mu = 0 refused)",
                               "orientation preserved (det J > 0) so signed Jacobi fields are comparable",
                               "pre-asymptotic orders at coarse steps reported, not hidden"],
        unresolved_assumptions=["Chart centers are rounded midpoints of one path; other placements change the factors",
                                "The fold amplification is measured, not derived; its scaling in mu is not fitted",
                                "Embedded observables only; the intrinsic hyperbolic chart is not reparametrized here"],
        recommended_next_task="T012 (frame-change invariance) and a fold-scaling study of error versus mu",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T012
T012_EMBEDDED = (("sphere", lambda: Sphere(1.0), SPHERE_START[0], SPHERE_START[1], 2.0),
                 ("torus", lambda: Torus(2.0, 1.0), (0.0, 0.5), 0.7, 3.0),
                 ("saddle", lambda: Saddle(1.0), (0.1, -0.2), 0.8, 1.5),
                 ("gaussian-bump", lambda: GaussianBump(0.5, 1.0), (-1.2, 0.3), 0.2, 2.5))
T012_ROTATIONS = (((1.0, 2.0, 3.0), 0.7), ((0.0, 0.0, 1.0), math.pi / 3), ((1.0, -1.0, 0.0), 2.5))
T012_BASIS_ANGLES = (0.3, 1.1, 2.5, -2.0)
T012_BASIS_EPS = 1e-4   # heading perturbation stated in the rotated basis (central difference, O(eps^2) error)
T012_STEPS = 64
# Orientation reversal off the sphere: the generic Torus(2, 1) path, no symmetry that makes the separation odd in eps.
T012_TORUS_LENGTH = 3.0
T012_TORUS_EPS = (1e-3, 1e-2, 4e-2)


def _state_with_tangent(u0, tangent):
    return np.concatenate([np.asarray(u0, dtype=float), tangent, [1.0, 0.0, 0.0, 1.0]])


def _orientation_runs(surface, u0, heading, length, eps, steps=T012_STEPS):
    """Signed embedded separations for a heading change +eps stated in (e1, e2) and in the left-handed (e1, -e2).

    (e1, -e2) is orthonormal but left-handed: its heading -h + eps is the right-handed heading h - eps (the
    geometric perturbation -eps) and its +90 degree normal is -N. Returns the base arclength grid and, at every
    node, the right-handed separation along N and the left-handed one along N and along its own normal -N (RK4).
    """
    u0 = np.asarray(u0, dtype=float)
    e1, e2 = surface.orthonormal_frame(u0)
    base = jacobi.transfer(surface, u0, heading, length, steps=steps)
    points = np.array([surface.embedding(y[:2]) for y in base.states])
    normals = np.array([core.embedded_normal(surface, y) for y in base.states])
    delta = {}
    for name, (f1, f2, local) in {"right-handed": (e1, e2, heading), "left-handed": (e1, -e2, -heading)}.items():
        tangent = math.cos(local + eps) * f1 + math.sin(local + eps) * f2
        _, states = integrators.integrate_fixed(surface.geodesic_rhs, np.concatenate([u0, tangent]), length,
                                                steps, "rk4")
        delta[name] = np.array([surface.embedding(y[:2]) for y in states]) - points
    return base.s, {"right-handed along N": np.einsum("ij,ij->i", delta["right-handed"], normals),
                    "left-handed along N": np.einsum("ij,ij->i", delta["left-handed"], normals),
                    "left-handed along its own normal -N": np.einsum("ij,ij->i", delta["left-handed"], -normals)}


def _orientation_pair(surface, u0, heading, length, eps, steps=T012_STEPS):
    """Right-handed separation along N and left-handed separation along its own normal -N, at L."""
    _, series = _orientation_runs(surface, u0, heading, length, eps, steps)
    return float(series["right-handed along N"][-1]), float(series["left-handed along its own normal -N"][-1])


def frame_study():
    """Ambient rotations, rotated reference bases and an orientation-reversing basis."""
    rotation_rows, basis_rows = [], []
    for key, make, u0, heading, length in T012_EMBEDDED:
        base = make()
        reference = jacobi.transfer(base, u0, heading, length, steps=T012_STEPS)
        base_curvature = reference.curvature_along()
        base_end = base.embedding(reference.states[-1, :2])
        for axis, angle in T012_ROTATIONS:
            rot = rotation_matrix(axis, angle)
            rotated = Rotated(base, rot)
            run = jacobi.transfer(rotated, u0, heading, length, steps=T012_STEPS)
            rotation_rows.append({
                "surface": key, "axis": list(axis), "angle": angle,
                "max_state_difference": float(np.max(np.abs(run.states - reference.states))),
                "endpoint_rotation_error": float(np.linalg.norm(rotated.embedding(run.states[-1, :2]) - rot @ base_end)),
                "max_curvature_difference": float(np.max(np.abs(run.curvature_along() - base_curvature)))})
    intrinsic = T012_EMBEDDED[:3] + (("hyperbolic-plane", lambda: HyperbolicPlane(1.0), (0.0, 1.0), 0.6, 1.5),)
    for key, make, u0, heading, length in intrinsic:
        surface = make()
        u0 = np.asarray(u0, dtype=float)
        reference = jacobi.transfer(surface, u0, heading, length, steps=T012_STEPS)
        j_head = reference.states[:, 6]
        e1, e2 = surface.orthonormal_frame(u0)
        for beta in T012_BASIS_ANGLES:
            # The rotated basis goes through ciw's metric: f2 is ciw's +90 degree normal of f1, not a local rotation.
            f1 = math.cos(beta) * e1 + math.sin(beta) * e2
            f2 = surface.normal(u0, f1)
            frame_residual = max(abs(surface.inner(u0, f1, f1) - 1.0), abs(surface.inner(u0, f2, f2) - 1.0),
                                 abs(surface.inner(u0, f1, f2)))
            local = heading - beta
            tangent = math.cos(local) * f1 + math.sin(local) * f2
            _, states = integrators.integrate_fixed(jacobi.rhs(surface), _state_with_tangent(u0, tangent),
                                                    length, T012_STEPS, "rk4")
            # Heading perturbations +/- eps stated in the rotated basis, integrated as geodesics and measured along
            # ciw's normal of the base geodesic; the central difference must reproduce ciw's Jacobi field j_head.
            separations = []
            for sign in (1.0, -1.0):
                angle = local + sign * T012_BASIS_EPS
                start = np.concatenate([u0, math.cos(angle) * f1 + math.sin(angle) * f2])
                _, moved = integrators.integrate_fixed(surface.geodesic_rhs, start, length, T012_STEPS, "rk4")
                separations.append(jacobi.normal_separation(surface, reference.states, moved))
            central = (separations[0] - separations[1]) / (2.0 * T012_BASIS_EPS)
            basis_rows.append({"surface": key, "beta": beta, "frame_residual": float(frame_residual),
                               "tangent_difference": float(np.max(np.abs(tangent - reference.states[0, 2:4]))),
                               "max_state_difference": float(np.max(np.abs(states - reference.states))),
                               "jacobi_relative_gap": float(np.max(np.abs(central - j_head)) / np.max(np.abs(j_head)))})
    # Orientation reversal on the unit sphere, where the separation sin(s) sin(eps) is odd in eps, so the
    # left-handed "+eps" (geometric -eps) measured along its own normal -N equals the right-handed one exactly.
    sphere = Sphere(1.0)
    eps, length = 1e-3, 2.0
    s, series = _orientation_runs(sphere, SPHERE_START[0], SPHERE_START[1], length, eps)
    exact = np.sin(s) * math.sin(eps)
    end = {name: float(values[-1]) for name, values in series.items()}
    # Off the sphere the separation d(eps) = eps j + C2 eps^2 + ... is not odd, and the left-handed run along -N
    # is -d(-eps) = eps j - C2 eps^2 + ...: the two agree only to first order, so own/right - 1 ~ -2 C2 eps / j.
    torus = Torus(2.0, 1.0)
    t_u0, t_heading = TORUS_PATHS["generic"]
    pairs = {n: [_orientation_pair(torus, t_u0, t_heading, T012_TORUS_LENGTH, e, n) for e in T012_TORUS_EPS]
             for n in (T012_STEPS, 2 * T012_STEPS)}
    gaps = {n: [own / right - 1.0 for right, own in rows] for n, rows in pairs.items()}
    refusal = "none"
    try:
        Rotated(sphere, np.diag([1.0, 1.0, -1.0]))
    except SurfaceRefusal as exc:
        refusal = str(exc)
    return {"rotations": rotation_rows, "basis_rotations": basis_rows,
            "orientation": {"eps": eps, "length": length, "signed_separation_at_L": end,
                            "exact_at_L": float(exact[-1]),
                            "right_handed_minus_exact": end["right-handed along N"] - float(exact[-1]),
                            "ratio_along_right_handed_normal": end["left-handed along N"] / end["right-handed along N"],
                            "ratio_along_own_normal":
                                end["left-handed along its own normal -N"] / end["right-handed along N"],
                            "s": s.tolist(), "series": {k: v.tolist() for k, v in series.items()},
                            "exact": exact.tolist()},
            "orientation_torus": {"surface": "Torus(2, 1)", "start": list(t_u0), "heading": t_heading,
                                  "length": T012_TORUS_LENGTH, "eps": list(T012_TORUS_EPS), "steps": T012_STEPS,
                                  "right_handed_along_N": [r for r, _ in pairs[T012_STEPS]],
                                  "left_handed_along_own_normal": [o for _, o in pairs[T012_STEPS]],
                                  "own_over_right_minus_1": gaps[T012_STEPS],
                                  "own_over_right_minus_1_at_2N": gaps[2 * T012_STEPS]},
            "improper_rotation_refusal": refusal}


@task("T012", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t012_frame_invariance_and_refusal",))
def frame_change_invariance(ctx):
    study = ctx.memo("gjl-frame-study", frame_study)
    ctx.artifact_json("frame-invariance.json", core.jsonable(study, 12))
    floor = 1e-18
    surfaces = [case[0] for case in T012_EMBEDDED] + ["hyperbolic-plane"]
    figure = []
    for key in surfaces:
        rotations = [r["max_state_difference"] for r in study["rotations"] if r["surface"] == key]
        if rotations:
            figure.append((f"rotation: {key}", list(range(1, len(rotations) + 1)), [max(v, floor) for v in rotations]))
        basis = [r["max_state_difference"] for r in study["basis_rotations"] if r["surface"] == key]
        if basis:
            figure.append((f"basis angle: {key}", list(range(1, len(basis) + 1)), [max(v, floor) for v in basis]))
    ctx.artifact_text("frame-invariance.svg", svg.line_plot(
        figure, title="Max chart/Jacobi state difference under frame changes (RK4, N = 64)",
        xlabel="case index (rotations 1-3, basis angles 1-4)", ylabel="max |state - base state| (floored at 1e-18)",
        logy=True))
    orient = study["orientation"]
    ctx.artifact_text("orientation-separation.svg", svg.line_plot(
        [(name, orient["s"], values) for name, values in orient["series"].items()]
        + [("exact sin(s) sin(eps)", orient["s"], orient["exact"])],
        title=f"Unit sphere, heading change +eps = {orient['eps']:g} in each basis", xlabel="arclength s",
        ylabel="signed embedded separation", markers=False))
    rot_state = max(r["max_state_difference"] for r in study["rotations"])
    rot_end = max(r["endpoint_rotation_error"] for r in study["rotations"])
    rot_k = max(r["max_curvature_difference"] for r in study["rotations"])
    basis_state = max(r["max_state_difference"] for r in study["basis_rotations"])
    basis_tangent = max(r["tangent_difference"] for r in study["basis_rotations"])
    basis_frame = max(r["frame_residual"] for r in study["basis_rotations"])
    basis_gap = max(r["jacobi_relative_gap"] for r in study["basis_rotations"])
    flip, own = orient["ratio_along_right_handed_normal"], orient["ratio_along_own_normal"]
    residual = abs(flip + 1.0)
    expected = "Frame change requires a proper rotation matrix"
    orientation_basis = ("RK4 truncation (N = 64) of the difference between the +eps and -eps runs; the exact "
                         "sphere separation sin(s) sin(eps) is odd in eps, so no eps^2 term enters the ratio")
    torus = study["orientation_torus"]
    torus_eps, torus_gap = torus["eps"], torus["own_over_right_minus_1"]
    torus_abs = [abs(v) for v in torus_gap]
    torus_slope = core.loglog_slope(torus_eps, torus_abs)
    torus_step_change = max(abs(b / a - 1.0) for a, b in zip(torus_gap, torus["own_over_right_minus_1_at_2N"]))
    torus_path = (f"Torus(2, 1) from ({', '.join(_g(v, 3) for v in torus['start'])}), heading "
                  f"{_g(torus['heading'], 3)}, L = {_g(torus['length'], 3)}")
    findings = [
        finding("Ambient rotations leave chart trajectories and Jacobi fields unchanged to roundoff", "numerical",
                rot_state, {"generator": _gen("frame-study", rotations=len(T012_ROTATIONS), steps=T012_STEPS),
                            "derivation": _derivation("t012-frame-change-invariance"),
                            "checks": [core.check("invariant", "max |rotated state - base state| over all nodes",
                                                  rot_state, 1e-11)]},
                uncertainty=core.uncertainty("roundoff", rot_state,
                                             "the observed difference is itself the roundoff of rotated dot "
                                             "products"),
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("Embedded endpoints rotate exactly with the ambient frame", "numerical", rot_end,
                {"generator": _gen("frame-study"),
                 "checks": [core.check("invariant", "max |X_rotated(L) - R X(L)|", rot_end, 1e-12)]},
                uncertainty=core.uncertainty("roundoff", rot_end, "the observed difference is itself roundoff"),
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Gaussian curvature from the rotated second fundamental form is unchanged", "numerical", rot_k,
                {"generator": _gen("frame-study"),
                 "checks": [core.check("invariant", "max |K_rotated - K| along every path", rot_k, 1e-12)]},
                uncertainty=core.uncertainty("roundoff", rot_k, "the observed difference is itself roundoff"),
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("A reference tangent basis rotated in ciw's metric (heading measured from e1') gives the same start "
                "tangent, and heading perturbations stated in it separate as ciw's Jacobi field predicts",
                "numerical", {"max_frame_residual": basis_frame, "max_tangent_difference": basis_tangent,
                              "max_state_difference": basis_state, "max_jacobi_relative_gap": basis_gap},
                {"generator": _gen("frame-study", basis_angles=list(T012_BASIS_ANGLES), eps=T012_BASIS_EPS,
                                   steps=T012_STEPS),
                 "derivation": _derivation("t012-frame-change-invariance"),
                 "checks": [core.check("invariant", "max |g(f_a, f_b) - delta_ab| for f1 rotated by beta and f2 = "
                                       "ciw normal of f1", basis_frame, 1e-13),
                            core.check("invariant", "max |cos(h - beta) f1 + sin(h - beta) f2 - unit_tangent(h)|",
                                       basis_tangent, 1e-13),
                            core.check("analytic", "max |(d(+eps) - d(-eps)) / (2 eps) - j_head| / max |j_head| for "
                                       "perturbations stated in the rotated basis (ciw normal separation)",
                                       basis_gap, 1e-6)]},
                uncertainty=core.uncertainty("truncation_bound", basis_gap,
                                             "central-difference O(eps^2) and RK4 (N = 64) mismatch between the "
                                             "perturbed geodesics and the integrated Jacobi field, relative to "
                                             "max |j_head|"),
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("A heading change +eps stated in a left-handed basis (e1, -e2) is the geometric perturbation -eps: "
                "along the right-handed normal its separation has the opposite sign",
                "numerical", {"ratio_along_right_handed_normal": flip,
                              "ratio_along_own_normal": own,
                              "right_handed_minus_exact": orient["right_handed_minus_exact"]},
                {"generator": _gen("frame-study", eps=orient["eps"], length=orient["length"]),
                 "derivation": _derivation("t012-frame-change-invariance"),
                 "checks": [core.check("analytic", "right-handed signed separation at L minus sin(L) sin(eps)",
                                       orient["right_handed_minus_exact"], 1e-10),
                            core.check("invariant", "left-handed / right-handed separation along N on the unit sphere, plus 1",
                                       flip + 1.0, 1e-9)]},
                uncertainty=core.uncertainty("truncation_bound", abs(flip + 1.0), orientation_basis),
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("On a non-symmetric surface the orientation-reversed separation agrees only to first order: "
                "own/right - 1 is proportional to eps (Torus(2, 1) generic path)",
                "numerical", {"eps": torus_eps, "own_over_right_minus_1": torus_gap, "loglog_slope": torus_slope},
                {"generator": _gen("frame-study", surface=torus["surface"], start=torus["start"],
                                   heading=torus["heading"], length=torus["length"], eps=torus_eps,
                                   steps=torus["steps"]),
                 "derivation": _derivation("t012-frame-change-invariance"),
                 "checks": [core.check("analytic", "log-log slope of |own/right - 1| against eps minus 1",
                                       torus_slope - 1.0, 0.05),
                            core.check("invariant", "|own/right - 1| at eps = 1e-3", torus_abs[0], 1e-5, "ge"),
                            core.check("self_convergence", "max relative change of own/right - 1 from N = "
                                       f"{torus['steps']} to {2 * torus['steps']} (the gap is not truncation)",
                                       torus_step_change, 1e-4, "le")]},
                uncertainty=core.uncertainty("fit_spread", core.slope_spread(torus_eps, torus_abs),
                                             "largest gap between the fitted slope and consecutive-eps slopes "
                                             "(eps^3 terms of the separation)"),
                tolerance={"abs": 1e-6, "rel": 1e-3},
                counterexample={"statement": "Finite-eps signed separations are invariant under an "
                                             "orientation-reversing basis change expressed consistently",
                                "witness": {"path": torus_path, "eps": torus_eps,
                                            "own_over_right_minus_1": torus_gap}}),
        finding("An improper rotation (reflection) is refused as a frame change", "computational_pipeline",
                study["improper_rotation_refusal"],
                {"checks": [core.refusal_check("Rotated(sphere, diag(1, 1, -1))", expected,
                                               study["improper_rotation_refusal"])]}),
    ]
    rotations_text = "; ".join(f"axis ({', '.join(_g(v, 3) for v in axis)}) angle {_g(angle, 5)}"
                               for axis, angle in T012_ROTATIONS)
    fields = _fields(
        hypothesis=("Geodesics and Jacobi fields are intrinsic: an ambient rotation changes only the embedded "
                    "coordinates (which rotate exactly), and the choice of reference basis for headings is a "
                    "relabeling; an orientation-reversing basis changes the meaning of '+eps' and of the normal "
                    "together, so a consistently expressed first-order (Jacobi) separation is unchanged; the "
                    "finite-eps separation is unchanged exactly only where it is odd in eps."),
        mathematical_model=("Rotated(base, R): X' = R X, so X'_i . X'_j = X_i . X_j and the second fundamental form is "
                            "unchanged; in exact arithmetic the chart ODE is identical. A basis with f1 = cos(beta) e1 "
                            "+ sin(beta) e2 and f2 = N(f1), the metric +90 degree rotation, is orthonormal, and heading "
                            "h - beta in it is the unit tangent of heading h; a heading change +/- eps stated in it is "
                            "the geometric heading perturbation, so (d(+eps) - d(-eps)) / (2 eps) = j_head + O(eps^2) "
                            "with d measured along N by g(delta u, N). In the left-handed basis "
                            "(e1, -e2) the heading -h + eps is the right-handed heading h - eps, and its +90 degree "
                            "normal is -N. With d(eps) = J_eps . N = eps j + C2 eps^2 + ..., the left-handed run "
                            "measured along -N is -d(-eps) = eps j - C2 eps^2 + ..., so (J_eps . (-N)) equals the "
                            "right-handed J_eps . N to first order in eps; the eps^2 parts have opposite signs, so the "
                            "finite separations agree exactly only where the separation is odd in eps, as on the unit "
                            "sphere (sin(s) sin(eps)), and otherwise own/right - 1 = -2 C2 eps / j + O(eps^2). "
                            "Measured along -N the left-handed separation is exactly minus its value along N, so "
                            "its ratio to the right-handed one is minus the ratio along N by construction."),
        input_data=["Sphere, Torus(2, 1), Saddle(1), GaussianBump(0.5, 1) on declared paths; HyperbolicPlane(1) for "
                    "basis changes", f"Rotations: {rotations_text}",
                    f"Basis angles beta = {', '.join(_g(b, 3) for b in T012_BASIS_ANGLES)} with heading "
                    f"perturbations +/- {_g(T012_BASIS_EPS, 2)} stated in the rotated basis; RK4 with N = {T012_STEPS}; "
                    f"orientation test on the unit sphere with eps = {orient['eps']:g}, L = {orient['length']:g}, and "
                    f"on {torus_path} with eps = {', '.join(_g(e, 3) for e in torus_eps)} (RK4 N = {torus['steps']}, "
                    f"checked at {2 * torus['steps']})"],
        observation_model=("Chart states (u, v, Jacobi columns) at every node, embedded endpoints, curvature along "
                           "the path, the metric inner products of the rotated basis, the chart normal separation "
                           "g(delta u, N) of basis-stated perturbations at matched nodes, and the signed embedded "
                           "separation along the base geodesic's in-surface normal (N, or -N for the left-handed "
                           "basis)."),
        expected_invariant=("Differences at roundoff level; exact rotation of X(L); an orthonormal rotated basis "
                            "whose perturbations separate like eps j_head; separation ratio -1 when a left-handed "
                            "'+eps' is measured along N on the unit sphere; along its own normal -N the ratio is +1 on "
                            "the unit sphere and 1 + O(eps) on a non-symmetric surface."),
        experiment=("Integrate each path in the base and rotated surfaces and from the rotated reference bases; "
                    "compare node by node; integrate heading perturbations +/- eps stated in each rotated basis and "
                    "compare their central-difference separation along ciw's normal with ciw's Jacobi field; "
                    "integrate +eps heading perturbations in right- and left-handed bases and project each onto N "
                    "and -N, on the unit sphere and, for three eps, on a generic torus path."),
        numerical_result=(f"Rotation: max state difference {_g(rot_state, 2)}, endpoint rotation error {_g(rot_end, 2)}, "
                          f"curvature difference {_g(rot_k, 2)}; basis rotation: frame residual {_g(basis_frame, 2)}, "
                          f"start-tangent difference {_g(basis_tangent, 2)}, max state difference {_g(basis_state, 2)}, "
                          f"basis-stated perturbations match j_head to {_g(basis_gap, 2)} relative; unit sphere: "
                          f"left-handed '+eps' along N: ratio {_g(flip, 13)} (along its own normal {_g(own, 13)}, the "
                          f"same number negated); right-handed separation minus sin(L) sin(eps) "
                          f"{_g(orient['right_handed_minus_exact'], 2)}; torus: own/right - 1 = "
                          f"{', '.join(_g(v, 4) for v in torus_gap)} for eps = {', '.join(_g(e, 3) for e in torus_eps)} "
                          f"(log-log slope {_g(torus_slope, 4)}); reflection refused: "
                          f"'{study['improper_rotation_refusal']}'."),
        uncertainty=(f"Roundoff differences depend on platform arithmetic (observed up to {_g(max(rot_state, basis_state), 2)}); "
                     f"the regression tolerances allow 1e-11. The basis-stated perturbations reproduce j_head to "
                     f"{_g(basis_gap, 2)} relative, the central-difference O(eps^2) and RK4 mismatch between perturbed "
                     "geodesics and the integrated Jacobi field. On the unit sphere the orientation ratio differs from "
                     f"-1 by {_g(residual, 2)}: RK4 truncation of the +eps and -eps runs, not a "
                     "second-order eps term, because the exact sphere separation is odd in eps. On the torus the gap "
                     f"own/right - 1 is a second-order term of the separation, not truncation: it changes by "
                     f"{_g(torus_step_change, 2)} relative from N = {torus['steps']} to {2 * torus['steps']}, and its "
                     f"slope in eps is within {_g(core.slope_spread(torus_eps, torus_abs), 2)} of the consecutive-eps "
                     "slopes."),
        failure_modes_checked=["bitwise equality is not assumed (rotation changes rounding of dot products)",
                               "the rotated basis is built with ciw's metric normal and checked for orthonormality; "
                               "its perturbations are compared with ciw's Jacobi field, so a wrong metric, normal or "
                               "geodesic equation fails the basis finding (a relabeled start tangent alone would "
                               "integrate the same ODE twice)",
                               "improper rotation refused by the core",
                               "curvature recomputed from the rotated embedding, not reused",
                               "orientation conventions separated: sign flip under mixed conventions, invariance "
                               "under consistent ones",
                               "right-handed separation compared with its closed form sin(L) sin(eps)",
                               "orientation invariance not generalized from the sphere, whose separation is odd in "
                               "eps: a non-symmetric torus path shows the first-order-only agreement, checked against "
                               "a doubled step count"],
        unresolved_assumptions=["Rotations only; translations and reflections of the embedding are not exercised",
                                "Hyperbolic-plane isometries (Mobius maps) are not tested, only basis changes",
                                "The orientation test uses one sphere path and one torus path; the saddle is covered "
                                "by the ambient-rotation and basis-rotation tests and the bump by the ambient-rotation "
                                "test only, and the coefficient C2 of the torus gap is measured, not derived"],
        recommended_next_task="T013 (flat/developable limit) and an isometry test for HyperbolicPlane under Mobius maps",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T013
T013_RADII = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
T013_FIT_FROM = 16.0
T013_LENGTH, T013_STEPS = 2.0, 64


def flat_limit_study():
    """Plane versus cylinder (intrinsically identical) and tori with growing major radius."""
    u0, heading, length = (0.2, 0.1), 0.6, 3.0
    plane, cylinder = Plane(), Cylinder(1.0)
    run_plane = jacobi.transfer(plane, u0, heading, length, steps=60)
    run_cyl = jacobi.transfer(cylinder, u0, heading, length, steps=60)
    jacobi_difference = float(np.max(np.abs(run_plane.states[:, 4:8] - run_cyl.states[:, 4:8])))
    # Both core classes return a literal K = 0, so the comparison above tests pipeline arithmetic only. The
    # geometric statement is tested with K recomputed from the cylinder's second fundamental form.
    geometric = core.SecondFormCurvature(cylinder)
    run_geo = jacobi.transfer(geometric, u0, heading, length, steps=60)
    second_form_k = float(np.max(np.abs(run_geo.curvature_along())))
    j_minus_s = float(np.max(np.abs(run_geo.states[:, 6] - run_geo.s)))
    helix = cylinder.exact_geodesic(np.asarray(u0, dtype=float), run_geo.states[0, 2:4], run_geo.s)
    helix_error = float(np.max(np.abs(run_geo.states[:, :2] - helix)))
    chord_plane = float(np.linalg.norm(plane.embedding(run_plane.states[-1, :2]) - plane.embedding(np.array(u0))))
    chord_cyl = float(np.linalg.norm(cylinder.embedding(run_cyl.states[-1, :2]) - cylinder.embedding(np.array(u0))))
    radius = cylinder.radius
    helix_chord = math.hypot(2 * radius * math.sin(length * math.cos(heading) / (2 * radius)), length * math.sin(heading))
    tori = []
    for major in T013_RADII:
        torus = Torus(major, 1.0)
        run = jacobi.transfer(torus, (0.0, 0.0), 0.0, T013_LENGTH, steps=T013_STEPS)
        rho = major + torus.minor
        curvature = 1.0 / (torus.minor * rho)
        w = math.sqrt(curvature)
        exact_j = math.sin(w * T013_LENGTH) / w
        chord = float(np.linalg.norm(torus.embedding(run.states[-1, :2]) - torus.embedding(np.zeros(2))))
        tori.append({"major": major, "rho": rho, "curvature": curvature,
                     "j_head_relative_error": abs(float(run.states[-1, 6]) - exact_j) / exact_j,
                     "deviation": T013_LENGTH - float(run.states[-1, 6]),
                     # L - sin(wL)/w = x_minus_sin(wL) / w, free of cancellation.
                     "exact_deviation": core.x_minus_sin(w * T013_LENGTH) / w,
                     "leading_order": curvature * T013_LENGTH ** 3 / 6,
                     "chord_deficit": T013_LENGTH - chord,
                     # The equator is a circle of radius rho: L - 2 rho sin(L / (2 rho)) = 2 rho (x - sin x).
                     "exact_chord_deficit": 2 * rho * core.x_minus_sin(T013_LENGTH / (2 * rho))})
    fit = [row for row in tori if row["major"] >= T013_FIT_FROM]
    radii = [row["major"] for row in fit]
    rhos = [row["rho"] for row in fit]
    return {"plane_cylinder": {"jacobi_max_difference": jacobi_difference, "second_form_max_abs_curvature": second_form_k,
                               "j_head_minus_s_max": j_minus_s, "helix_chart_error": helix_error,
                               "chord_plane": chord_plane, "chord_cylinder": chord_cyl, "chord_cylinder_exact": helix_chord,
                               "length": length, "heading": heading},
            "tori": tori,
            "deviation_exponent": core.loglog_slope(radii, [r["deviation"] for r in fit]),
            "deviation_exponent_exact": core.loglog_slope(radii, [r["exact_deviation"] for r in fit]),
            "leading_order_exponent": core.loglog_slope(radii, [r["leading_order"] for r in fit]),
            "deviation_exponent_in_rho": core.loglog_slope(rhos, [r["deviation"] for r in fit]),
            "chord_deficit_exponent": core.loglog_slope(radii, [r["chord_deficit"] for r in fit]),
            "chord_deficit_exponent_in_rho": core.loglog_slope(rhos, [r["chord_deficit"] for r in fit]),
            "deviation_spread": core.slope_spread(radii, [r["deviation"] for r in fit]),
            "chord_deficit_spread": core.slope_spread(radii, [r["chord_deficit"] for r in fit]),
            "max_j_head_relative_error": max(r["j_head_relative_error"] for r in tori),
            "max_relative_error_vs_closed_form": max(abs(r["deviation"] / r["exact_deviation"] - 1) for r in tori),
            "max_chord_deficit_relative_error": max(abs(r["chord_deficit"] / r["exact_chord_deficit"] - 1) for r in tori),
            "max_leading_order_relative_gap": max(abs(r["deviation"] / r["leading_order"] - 1) for r in fit)}


@task("T013", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t013_flat_limit",))
def flat_developable_limit(ctx):
    study = ctx.memo("gjl-flat-limit", flat_limit_study)
    ctx.artifact_json("flat-limit.json", core.jsonable(study, 12))
    radii = [r["major"] for r in study["tori"]]
    ctx.artifact_text("flat-limit.svg", svg.line_plot(
        [("L - j_head(L) (Jacobi deviation)", radii, [r["deviation"] for r in study["tori"]]),
         ("K L^3 / 6", radii, [r["leading_order"] for r in study["tori"]]),
         ("L - chord (extrinsic)", radii, [r["chord_deficit"] for r in study["tori"]])],
        title="Torus outer equator, L = 2: flat limit", xlabel="major radius R (r = 1)", ylabel="deviation",
        logx=True, logy=True))
    pc = study["plane_cylinder"]
    fit_bound = next(r for r in study["tori"] if r["major"] == T013_FIT_FROM)
    # Split the finite-R offset of the exponent: K = 1/(R + 1) versus the K L^2/20 correction.
    offset = study["deviation_exponent"] + 1.0
    rho_share = study["leading_order_exponent"] + 1.0
    correction_share = study["deviation_exponent_in_rho"] + 1.0
    findings = [
        finding("The cylinder is intrinsically flat: K from its second fundamental form vanishes along the helix and "
                "the Jacobi field integrated with that K is j_head(s) = s", "numerical",
                {"second_form_max_abs_curvature": pc["second_form_max_abs_curvature"],
                 "j_head_minus_s_max": pc["j_head_minus_s_max"], "helix_chart_error": pc["helix_chart_error"],
                 # Context only: both core classes return a literal K = 0, so equal Jacobi columns test nothing.
                 "literal_K_plane_cylinder_jacobi_difference": pc["jacobi_max_difference"]},
                {"generator": _gen("cylinder-second-form", steps=60),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("analytic", "max |K| from the second fundamental form along the path",
                                       pc["second_form_max_abs_curvature"], 1e-14),
                            core.check("analytic", "max |j_head(s) - s| with the second-form K",
                                       pc["j_head_minus_s_max"], 1e-13),
                            core.check("analytic", "max chart distance to the exact helix (straight line in (phi, z))",
                                       pc["helix_chart_error"], 1e-12)]},
                uncertainty=core.uncertainty("roundoff", pc["j_head_minus_s_max"],
                                             "accumulated rounding of j_head = s over 60 RK4 steps"),
                tolerance={"abs": 1e-13, "rel": 0.0}),
        finding("Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord",
                "numerical", {"chord_plane": pc["chord_plane"], "chord_cylinder": pc["chord_cylinder"]},
                {"generator": _gen("plane-cylinder", length=pc["length"]),
                 "checks": [core.check("analytic", "cylinder chord minus helix closed form",
                                       pc["chord_cylinder"] - pc["chord_cylinder_exact"], 1e-9),
                            core.check("analytic", "plane chord minus L", pc["chord_plane"] - pc["length"], 1e-12),
                            core.check("invariant", "plane chord minus cylinder chord",
                                       pc["chord_plane"] - pc["chord_cylinder"], 0.1, "signed_ge")]},
                uncertainty=core.uncertainty("reference_error",
                                             abs(pc["chord_cylinder"] - pc["chord_cylinder_exact"]),
                                             "numerical helix chord against its closed form"),
                tolerance={"abs": 1e-9, "rel": 1e-9},
                counterexample={"statement": "Surfaces with identical Jacobi fields (intrinsic geometry) have identical "
                                             "chords between corresponding points",
                                "witness": {"surfaces": ["Plane", "Cylinder(1)"], "start": [0.2, 0.1], "heading": 0.6,
                                            "length": pc["length"], "chord_plane": pc["chord_plane"],
                                            "chord_cylinder": pc["chord_cylinder"]}}),
        finding("Torus outer-equator Jacobi deviation L - j_head(L) matches the closed form for every major radius",
                "numerical", study["max_relative_error_vs_closed_form"],
                {"generator": _gen("torus-flat-limit", radii=list(T013_RADII), steps=T013_STEPS),
                 "checks": [core.check("analytic", "max relative error of L - j_head(L) against (wL - sin wL)/w",
                                       study["max_relative_error_vs_closed_form"], 1e-7)]},
                uncertainty=core.uncertainty("truncation_bound", study["max_relative_error_vs_closed_form"],
                                             "the RK4 relative error itself bounds the deviation error"),
                tolerance={"abs": 1e-9, "rel": 0.5}),
        finding("The Jacobi deviation from flat decays like 1/R: exponent -1 in R + 1 = 1/K, with the finite-R offset "
                "explained", "numerical",
                {"fitted_exponent": study["deviation_exponent"], "closed_form_exponent": study["deviation_exponent_exact"],
                 "fitted_exponent_in_R_plus_1": study["deviation_exponent_in_rho"],
                 "offset_from_K_equals_1_over_R_plus_1": rho_share, "offset_from_K_L2_over_20": correction_share,
                 "max_gap_to_K_L3_over_6": study["max_leading_order_relative_gap"]},
                {"generator": _gen("torus-flat-limit", fit_radii_from=T013_FIT_FROM),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("analytic", "fitted minus closed-form exponent over the same radii",
                                       study["deviation_exponent"] - study["deviation_exponent_exact"], 1e-4),
                            core.check("analytic", "exponent in R + 1 plus 1 (leaves only the K L^2/20 correction)",
                                       study["deviation_exponent_in_rho"] + 1.0, 0.01),
                            core.check("analytic", "|dev / (K L^3/6) - 1| <= K L^2/20 at the smallest fitted R",
                                       study["max_leading_order_relative_gap"],
                                       1.05 * fit_bound["curvature"] * T013_LENGTH ** 2 / 20, "le")]},
                uncertainty=core.uncertainty("fit_spread", study["deviation_spread"],
                                             "largest gap between the fitted exponent in R and consecutive-radius "
                                             "slopes (R >= 16)"),
                tolerance={"abs": 1e-6, "rel": 0.0}),
        finding("The chord deficit vanishes faster (exponent -2 in R + 1) than the Jacobi deviation (-1)",
                "numerical", {"exponent_in_R": study["chord_deficit_exponent"],
                              "exponent_in_R_plus_1": study["chord_deficit_exponent_in_rho"],
                              "max_relative_error_vs_closed_form": study["max_chord_deficit_relative_error"]},
                {"generator": _gen("torus-flat-limit"),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("analytic", "numerical chord deficit against 2 rho (x - sin x), max relative",
                                       study["max_chord_deficit_relative_error"], 1e-7),
                            core.check("analytic", "chord-deficit exponent in R + 1 plus 2",
                                       study["chord_deficit_exponent_in_rho"] + 2.0, 1e-3)]},
                uncertainty=core.uncertainty("roundoff", study["max_chord_deficit_relative_error"],
                                             "largest relative deviation of the numerical deficit from its "
                                             "closed form: rounding of embedded coordinates of size R + 1 against a "
                                             "deficit of order L^3/(24 (R + 1)^2)"),
                tolerance={"abs": 1e-4, "rel": 0.0},
                counterexample={"statement": "Intrinsic (Jacobi) and extrinsic (chord) signatures of curvature vanish "
                                             "at the same rate in the flat limit",
                                "witness": {"jacobi_exponent_in_R_plus_1": study["deviation_exponent_in_rho"],
                                            "chord_exponent_in_R_plus_1": study["chord_deficit_exponent_in_rho"]}}),
        finding("A physical cylinder or large-radius torus workpiece shows these separations", "physical", None, {}),
    ]
    fields = _fields(
        hypothesis=("Intrinsic flatness (K = 0) makes Jacobi fields identical to the plane's even when the surface is "
                    "curved in space, while chords (extrinsic) differ; along the outer equator of a torus with "
                    "growing major radius the Jacobi deviation from flat vanishes like K L^3/6 ~ 1/R."),
        mathematical_model=("Cylinder: the second fundamental form has only the phi-phi entry, so K = 0 and j_head = s "
                            "exactly, but a helix of angle alpha has chord sqrt((2R sin(L cos(alpha)/(2R)))^2 + "
                            "(L sin(alpha))^2). Torus(R, 1) outer equator: K = 1/(R + 1), j_head = sin(wL)/w, "
                            "w = sqrt(K), L - j_head = K L^3/6 - K^2 L^5/120 + ...; the equator is a circle of radius "
                            "R + 1 with chord deficit 2 (R + 1)(x - sin x), x = L/(2 (R + 1)), = L^3/(24 (R + 1)^2) + ..."),
        input_data=["Plane and Cylinder(1): start (0.2, 0.1), heading 0.6, L = 3, RK4 N = 60; the cylinder also with "
                    "K recomputed from its second fundamental form",
                    f"Torus(R, 1), R in {', '.join(_g(r, 4) for r in T013_RADII)}, outer equator start (0, 0) heading 0, "
                    f"L = {T013_LENGTH}, RK4 N = {T013_STEPS}; exponents fitted for R >= {_g(T013_FIT_FROM, 3)}"],
        observation_model="Jacobi columns at every node; embedded chord |X(L) - X(0)|; no renormalization.",
        expected_invariant=("Cylinder K = 0 from the embedding and j_head = s; exponent of L - j_head(L) in R + 1 "
                            "equal to -1 up to the K L^2/20 correction, chord-deficit exponent -2."),
        experiment=("Integrate both flat surfaces on identical grids (their literal K = 0 makes equal Jacobi columns "
                    "a pipeline consistency fact, recorded as context, not a finding); integrate the cylinder again "
                    "with K from its second fundamental form; integrate the torus equator for each "
                    "R, compare with cancellation-free closed forms and fit log-log exponents in R and in R + 1."),
        numerical_result=(f"Plane/cylinder Jacobi difference with the literal K = 0 {_g(pc['jacobi_max_difference'], 2)} "
                          f"(pipeline consistency only); second-form cylinder "
                          f"max |K| {_g(pc['second_form_max_abs_curvature'], 2)}, max |j_head - s| "
                          f"{_g(pc['j_head_minus_s_max'], 2)}; chords {_g(pc['chord_plane'], 8)} (plane) versus "
                          f"{_g(pc['chord_cylinder'], 8)} (cylinder); torus deviation exponent "
                          f"{_g(study['deviation_exponent'], 5)} in R (closed form {_g(study['deviation_exponent_exact'], 5)}), "
                          f"{_g(study['deviation_exponent_in_rho'], 5)} in R + 1; chord-deficit exponent "
                          f"{_g(study['chord_deficit_exponent'], 5)} in R, {_g(study['chord_deficit_exponent_in_rho'], 6)} "
                          f"in R + 1; max relative error versus closed form {_g(study['max_relative_error_vs_closed_form'], 2)} "
                          f"(deviation), {_g(study['max_chord_deficit_relative_error'], 2)} (chord deficit)."),
        uncertainty=(f"RK4 relative error of j_head(L) at h = {_g(T013_LENGTH / T013_STEPS, 3)} is at most "
                     f"{_g(study['max_j_head_relative_error'], 2)} (largest at R = 2). The finite-R exponent in R, "
                     f"{_g(study['deviation_exponent'], 5)}, sits {_g(offset, 2)} above -1: {_g(rho_share, 2)} of that "
                     f"comes from K = 1/(R + 1) rather than 1/R (the fit of K L^3/6 alone) and {_g(correction_share, 2)} "
                     "from the K L^2/20 correction; the chord-deficit offset from -2 in R is the same R/(R + 1) effect. "
                     "The numerical chord deficit is limited by rounding of embedded coordinates of size R + 1."),
        failure_modes_checked=["cylinder flatness tested with K recomputed from its second fundamental form, not the "
                               "closed-form K = 0 that both core classes return",
                               "numerical deviation compared with its closed form, not only with K L^3/6",
                               "closed forms evaluated without cancellation (series for x - sin x below 0.1)",
                               "exponent offset attributed quantitatively (R + 1 versus K L^2/20)"],
        unresolved_assumptions=["Other developable surfaces (cones, tangent developables) are not in the core catalogue",
                                "Only the outer equator (constant K) is fitted; inclined torus geodesics average K and "
                                "are not studied",
                                "Physical workpieces are not measured: the physical finding is not established"],
        recommended_next_task="T046 (chord versus geodesic distance) and T018 (resolvability of weak curvature)",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T014
T014_CASES = (("sphere", lambda: Sphere(1.0), (1.1, 0.4), 0.9, 2.0),
              ("torus", lambda: Torus(2.0, 1.0), (0.0, 0.5), 0.7, 3.0),
              ("hyperbolic-plane", lambda: HyperbolicPlane(1.0), (0.0, 1.0), 0.6, 1.5))
T014_STEPS = (20, 40, 80, 160)
# Reversal predicts order p for odd p and p + 1 for even p (leading errors of the h and -h steps cancel).
T014_PREDICTED = {"euler": 1, "midpoint": 3, "rk4": 5}
# Time reversal of the joint state: velocities and Jacobi derivatives change sign.
FLIP = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def reversal_study():
    rows, adaptive = [], []
    for key, make, u0, heading, length in T014_CASES:
        surface = make()
        f = jacobi.rhs(surface)
        y0 = jacobi.initial_state(surface, u0, heading)
        for method in ("euler", "midpoint", "rk4"):
            errors = []
            for steps in T014_STEPS:
                _, forward = integrators.integrate_fixed(f, y0, length, steps, method)
                _, backward = integrators.integrate_fixed(f, forward[-1] * FLIP, length, steps, method)
                errors.append(float(np.max(np.abs(backward[-1] * FLIP - y0))))
            rows.append({"surface": key, "method": method, "errors": errors,
                         "order": -core.loglog_slope(T014_STEPS, errors)})
        rtol = 1e-9
        _, forward, stats_f = integrators.integrate_adaptive(f, y0, length, rtol=rtol, atol=1e-12)
        _, backward, stats_b = integrators.integrate_adaptive(f, forward[-1] * FLIP, length, rtol=rtol, atol=1e-12)
        adaptive.append({"surface": key, "rtol": rtol, "return_error": float(np.max(np.abs(backward[-1] * FLIP - y0))),
                         "forward_steps": stats_f["accepted_steps"], "backward_steps": stats_b["accepted_steps"]})
    return {"fixed": rows, "adaptive": adaptive}


def truncation_study():
    """Truncate at L1 and continue to L2 versus direct integration (torus geodesic)."""
    torus = Torus(2.0, 1.0)
    y0 = jacobi.initial_state(torus, (0.0, 0.5), 0.7)
    f = jacobi.rhs(torus)
    dyadic = {}
    for method in ("euler", "midpoint", "rk4"):
        _, direct = integrators.integrate_fixed(f, y0, 2.0, 256, method)
        _, first = integrators.integrate_fixed(f, y0, 1.25, 160, method)
        _, rest = integrators.integrate_fixed(f, first[-1], 0.75, 96, method)
        joined = np.vstack([first, rest[1:]])
        dyadic[method] = float(np.max(np.abs(joined - direct)))
    # Decimal truncation lengths whose step sizes are not all the same double.
    total, steps_total = 3.0, 300
    h = total / steps_total
    _, direct = integrators.integrate_fixed(f, y0, total, steps_total, "rk4")
    witness, differing = None, 0
    for n1 in range(100, 200):
        cut = round(n1 * 0.01, 2)
        steps_rest = steps_total - n1
        if cut / n1 == h and (total - cut) / steps_rest == h:
            continue
        _, first = integrators.integrate_fixed(f, y0, cut, n1, "rk4")
        _, rest = integrators.integrate_fixed(f, first[-1], total - cut, steps_rest, "rk4")
        joined = np.vstack([first, rest[1:]])
        differing = int(np.count_nonzero(joined[-1] != direct[-1]))
        if differing:
            # The count of differing components depends on last-bit rounding (libm); it stays in the artifact only.
            witness = {"L1": cut, "N1": n1, "L2": total, "N2": steps_total, "h_direct": h, "h_first": cut / n1,
                       "h_rest": (total - cut) / steps_rest,
                       "step_mismatch": cut / n1 != h or (total - cut) / steps_rest != h, "differs": True,
                       "max_abs_difference": float(np.max(np.abs(joined - direct)))}
            break
    rtol = 1e-9
    _, whole, _ = integrators.integrate_adaptive(f, y0, 2.0, rtol=rtol, atol=1e-12)
    _, part, _ = integrators.integrate_adaptive(f, y0, 1.25, rtol=rtol, atol=1e-12)
    _, cont, _ = integrators.integrate_adaptive(f, part[-1], 0.75, rtol=rtol, atol=1e-12)
    return {"dyadic_max_difference": dyadic, "non_dyadic_witness": witness,
            "non_dyadic_differing_final_components": differing,
            "adaptive_restart": {"rtol": rtol, "difference": float(np.max(np.abs(cont[-1] - whole[-1])))}}


@task("T014", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t014_reversal_and_truncation",))
def reversal_and_truncation(ctx):
    rev = ctx.memo("gjl-reversal", reversal_study)
    trunc = ctx.memo("gjl-truncation", truncation_study)
    ctx.artifact_json("reversal.json", core.jsonable(rev, 12))
    ctx.artifact_json("truncation.json", core.jsonable(trunc, 17))
    ctx.artifact_text("reversal.svg", svg.line_plot(
        [(f"{r['surface']} {r['method']}", [1.0 / n for n in T014_STEPS], r["errors"]) for r in rev["fixed"]],
        title="Forward-then-reversed return error", xlabel="1/N", ylabel="max |return - start|", logx=True, logy=True))
    orders = {f"{r['surface']}: {r['method']}": r["order"] for r in rev["fixed"]}
    checks = [core.check("analytic", f"{r['surface']} {r['method']} order minus {T014_PREDICTED[r['method']]}",
                         r["order"] - T014_PREDICTED[r["method"]], 0.2) for r in rev["fixed"]]
    adaptive_ratio = max(r["return_error"] / r["rtol"] for r in rev["adaptive"])
    witness = trunc["non_dyadic_witness"]
    dyadic = max(trunc["dyadic_max_difference"].values())
    restart_ratio = trunc["adaptive_restart"]["difference"] / trunc["adaptive_restart"]["rtol"]
    by_method = {m: float(np.mean([r["order"] for r in rev["fixed"] if r["method"] == m])) for m in T014_PREDICTED}
    findings = [
        finding("Reversal error orders are 1 (Euler), 3 (midpoint) and 5 (RK4): even-order methods gain one order",
                "numerical", orders,
                {"generator": _gen("reversal", steps=list(T014_STEPS)),
                 "derivation": _derivation("t014-geodesic-reversal-and-path-truncation"), "checks": checks},
                uncertainty=core.uncertainty("fit_spread",
                                             max(core.slope_spread(T014_STEPS, r["errors"]) for r in rev["fixed"]),
                                             "largest gap between a fitted order and its consecutive-step-pair "
                                             "orders"),
                tolerance={"abs": 0.05, "rel": 0.0},
                counterexample={"statement": "Forward-then-reversed integration with a method of order p returns to "
                                             "the start with error proportional to h^p",
                                "witness": {"mean_orders": by_method, "surfaces": [c[0] for c in T014_CASES]}}),
        finding("Adaptive forward-then-reversed integration returns to the start at tolerance level", "numerical",
                adaptive_ratio,
                {"generator": _gen("reversal-adaptive", rtol=1e-9),
                 "checks": [core.check("invariant", "max return error / rtol", adaptive_ratio, 10.0, "le")]},
                uncertainty=core.uncertainty("step_sequence",
                                             adaptive_ratio - min(r["return_error"] / r["rtol"] for r in rev["adaptive"]),
                                             "spread of return error / rtol over the three surfaces (set by the "
                                             "accepted step sequences)"),
                tolerance={"abs": 0.5, "rel": 0.5}),
        finding("Truncating at L1 and continuing on the same dyadic grid reproduces direct integration bit for bit",
                "computational_pipeline", trunc["dyadic_max_difference"],
                {"generator": _gen("truncation", L1=1.25, L2=2.0, steps=256),
                 "checks": [core.check("exact_arithmetic", "max |continued - direct| (Euler, midpoint, RK4)",
                                       dyadic, 0.0)]},
                uncertainty=core.uncertainty("roundoff", 0.0,
                                             "bitwise comparison of identical floating-point operations"),
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails",
                "computational_pipeline",
                {"differs": witness is not None, "max_abs_difference": witness["max_abs_difference"] if witness else 0.0},
                {"generator": _gen("truncation-search", L2=3.0, N2=300),
                 "checks": [core.check("exact_arithmetic", "differing final state components",
                                       trunc["non_dyadic_differing_final_components"], 1, "ge"),
                            core.check("invariant", "max difference stays at roundoff level",
                                       witness["max_abs_difference"] if witness else 1.0, 1e-12, "le")]},
                uncertainty=core.uncertainty("roundoff", witness["max_abs_difference"] if witness else 0.0,
                                             "the difference is one-ulp step-size rounding propagated through the "
                                             "run"),
                tolerance={"abs": 1e-12, "rel": 0.0},
                counterexample={"statement": "Truncate-and-continue reproduces fixed-step integration bit for bit "
                                             "for any truncation length", "witness": witness}),
        finding("Adaptive restart at L1 reproduces direct adaptive integration only to tolerance level", "numerical",
                restart_ratio,
                {"generator": _gen("truncation-adaptive", rtol=1e-9),
                 "checks": [core.check("invariant", "difference / rtol", restart_ratio, 10.0, "le"),
                            core.check("invariant", "restart is not bitwise: absolute difference",
                                       trunc["adaptive_restart"]["difference"], 1e-15, "ge")]},
                uncertainty=core.uncertainty("step_sequence", restart_ratio,
                                             "the whole difference comes from different accepted step sequences "
                                             "after the restart"),
                tolerance={"abs": 0.5, "rel": 0.5}),
    ]
    fields = _fields(
        hypothesis=("The geodesic flow is reversible, so integrating forward, flipping velocities and Jacobi "
                    "derivatives, and integrating again returns to the start up to the method's global error; "
                    "truncate-and-continue on an identical grid is the same arithmetic as direct integration."),
        mathematical_model=("With (u, v, j, j') -> (u, -v, j, -j') the flow over L is inverted. For a one-step method "
                            "with local error C h^(p+1), the step with -h has local error C (-h)^(p+1); the composition "
                            "cancels at order h^(p+1) when p is even, so the return error is O(h^(p+1)) for even p and "
                            "O(h^p) for odd p. Linear check: RK4 R(z) R(-z) = 1 + z^6/72 + ..., midpoint "
                            "1 + z^4/4, Euler 1 - z^2."),
        input_data=["Unit sphere (1.1, 0.4) heading 0.9 L = 2; Torus(2, 1) (0, 0.5) heading 0.7 L = 3; "
                    "HyperbolicPlane(1) (0, 1) heading 0.6 L = 1.5",
                    f"Fixed steps N = {list(T014_STEPS)}; DP45 rtol 1e-9, atol 1e-12",
                    "Truncation: torus, L1 = 1.25 / N1 = 160 and L2 = 2 / N2 = 256 (h = 2^-7); decimal cuts of L2 = 3, N2 = 300"],
        observation_model="Max absolute difference of the full joint state (geodesic and both Jacobi columns).",
        expected_invariant="Return error orders 1, 3, 5; bitwise continuation for identical step doubles.",
        experiment=("Forward/backward fixed-step and adaptive runs; truncated and continued runs compared with direct "
                    "runs bit for bit; a deterministic search over decimal truncation lengths for a step-size mismatch."),
        numerical_result=(f"Mean reversal orders {', '.join(f'{m} {_g(v, 4)}' for m, v in by_method.items())}; adaptive "
                          f"return error up to {_g(adaptive_ratio, 3)} x rtol; dyadic continuation difference "
                          f"{_g(dyadic, 2)}; decimal witness L1 = {witness['L1'] if witness else None} with a "
                          f"step-size mismatch and a bitwise different result (max difference "
                          f"{_g(witness['max_abs_difference'], 2) if witness else 0}); adaptive restart "
                          f"difference {_g(restart_ratio, 3)} x rtol."),
        uncertainty=("Orders are least-squares fits over four step sizes; the smallest RK4 return error ("
                     f"{_g(min(min(r['errors']) for r in rev['fixed'] if r['method'] == 'rk4'), 2)}) stays far above "
                     "roundoff. Adaptive ratios depend on the accepted step sequence."),
        failure_modes_checked=["Jacobi derivatives flipped along with velocities (otherwise the Jacobi state does not return)",
                               "roundoff floor kept below the smallest fitted error",
                               "bitwise claims checked with exact equality, not tolerances",
                               "decimal witness search is deterministic and bounded"],
        unresolved_assumptions=["The order gain for even p is derived for smooth problems; it can fail near "
                                "chart singularities",
                                "Adaptive step sequences are platform-sensitive at the last-bit level"],
        recommended_next_task="T015 (long-horizon drift) and a symmetric integrator (implicit midpoint) for exact reversal",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T015
T015_H = 0.125                     # dyadic, so every checkpoint is a node
T015_CHECKPOINTS = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
T015_METHODS = ("euler", "midpoint", "rk4", "adaptive")
T015_RTOL = 1e-6
T015_CLAIRAUT_FROM = 40.0          # adaptive torus Clairaut local slopes agree from here on


def _t015_case(key):
    if key == "torus":
        return Torus(2.0, 1.0), (0.0, 0.5), 0.7
    return Sphere(1.0), SPHERE_START[0], SPHERE_START[1]


def drift_run(key: str, method: str) -> dict:
    """One long run; drift envelopes max_{s' <= L} |I(s') - I(0)| at the checkpoints the run reached."""
    surface, u0, heading = _t015_case(key)
    u0 = np.asarray(u0, dtype=float)
    y0 = np.concatenate([u0, surface.unit_tangent(u0, heading)])
    horizon = T015_CHECKPOINTS[-1]
    failed, stats = None, {}
    if method == "adaptive":
        s, states, stats = integrators.integrate_adaptive(surface.geodesic_rhs, y0, horizon, rtol=T015_RTOL,
                                                          atol=T015_RTOL * 1e-3)
    else:
        # The polar chart of the sphere ends at the poles; stop there instead of integrating through them.
        stop = (lambda y: not 1e-6 < y[0] < math.pi - 1e-6) if key == "sphere" else None
        s, states, failed = core.fixed_march(surface.geodesic_rhs, y0, T015_H, int(horizon / T015_H), method, stop)
    series = {"energy": np.array([abs(surface.speed_squared(y[:2], y[2:4]) - 1.0) for y in states])}
    extra = {}
    if key == "torus":
        clairaut = np.array([surface.clairaut(y[:2], y[2:4]) for y in states])
        series["clairaut"] = np.abs(clairaut - clairaut[0])
        turning = math.acos((clairaut[0] - surface.major) / surface.minor)
        extra = {"theta_turning": turning, "theta_max_abs": float(np.max(np.abs(states[:, 1]))),
                 "clairaut_initial": float(clairaut[0])}
    else:
        points = np.array([surface.embedding(y[:2]) for y in states])
        velocity = np.array([surface.embedding_jacobian(y[:2]) @ y[2:4] for y in states])
        moment = np.cross(points, velocity)
        size = np.linalg.norm(moment, axis=1)
        series["angular_momentum"] = np.linalg.norm(moment - moment[0], axis=1)
        # Split the angular-momentum drift into a tilt of the orbit plane and a change of its size (speed).
        series["plane_tilt"] = np.linalg.norm(moment / size[:, None] - moment[0] / size[0], axis=1)
        series["momentum_size"] = np.abs(size - size[0])
        exact = surface.exact_embedded_geodesic(u0, surface.unit_tangent(u0, heading), s)
        error = points - exact
        series["position"] = np.linalg.norm(error, axis=1)
        # Unit sphere: the exact orbit plane has the fixed normal n0; the along-track direction is n0 x X_exact.
        plane_normal = moment[0] / size[0]
        series["cross_track"] = np.abs(error @ plane_normal)
        along = np.einsum("ij,ij->i", error, np.cross(plane_normal, exact))
        speed = np.sqrt([surface.speed_squared(y[:2], y[2:4]) for y in states])
        # A speed error moves the point along its circle by the integrated excess arclength.
        excess = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1] - 2.0) * np.diff(s))])
        nodes = [int(np.searchsorted(s, c - 1e-9)) for c in T015_CHECKPOINTS if c <= s[-1] + 1e-9]
        extra = {"along_track": [float(along[i]) for i in nodes],
                 "integrated_speed_error": [float(excess[i]) for i in nodes],
                 "checkpoint_arclength": [float(s[i]) for i in nodes]}
    envelopes, exponents, spreads = {}, {}, {}
    for name, values in series.items():
        running = np.maximum.accumulate(values)
        reached = [c for c in T015_CHECKPOINTS if c <= s[-1] + 1e-9]
        envelopes[name] = [float(running[np.searchsorted(s, c - 1e-9)]) for c in reached]
        fitted = len(reached) >= 3 and min(envelopes[name]) > 0.0
        exponents[name] = core.loglog_slope(reached, envelopes[name]) if fitted else None
        spreads[name] = core.slope_spread(reached, envelopes[name]) if fitted else None
    return {"surface": key, "method": method, "failed_at": failed, "reached": float(s[-1]),
            "envelopes": envelopes, "exponents": exponents, "exponent_spreads": spreads,
            "max_energy_error": float(np.max(series["energy"])),
            "accepted_steps": stats.get("accepted_steps"), **extra}


def _local_slopes(xs, ys):
    """Log-log slopes of consecutive checkpoint pairs (None where an envelope value is zero)."""
    return [None if a <= 0 or b <= 0 else math.log(b / a) / math.log(x1 / x0)
            for x0, x1, a, b in zip(xs, xs[1:], ys, ys[1:])]


def _slopes_text(slopes, digits=2) -> str:
    return ", ".join("-" if v is None else _g(v, digits) for v in slopes)


def drift_study():
    return {f"{key}:{method}": drift_run(key, method) for key in ("torus", "sphere") for method in T015_METHODS}


@task("T015", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t015_long_horizon_drift",))
def long_horizon_drift(ctx):
    study = ctx.memo("gjl-drift", drift_study)
    ctx.artifact_json("drift.json", core.jsonable(study, 12))
    series = []
    for label, row in study.items():
        values = row["envelopes"]["energy"]
        series.append((label, list(T015_CHECKPOINTS[:len(values)]), values))
    ctx.artifact_text("energy-drift.svg", svg.line_plot(series, title="Energy-error envelope |g(v, v) - 1|",
                                                        xlabel="length L", ylabel="max error up to L",
                                                        logx=True, logy=True))
    ctx.artifact_text("sphere-position-drift.svg", svg.line_plot(
        [(m, list(T015_CHECKPOINTS[:len(study[f"sphere:{m}"]["envelopes"]["position"])]),
          study[f"sphere:{m}"]["envelopes"]["position"]) for m in T015_METHODS],
        title="Sphere: distance to the exact great circle", xlabel="length L", ylabel="max error up to L",
        logx=True, logy=True))
    adaptive_energy = {k: study[f"{k}:adaptive"]["exponents"]["energy"] for k in ("torus", "sphere")}
    pos = {m: study[f"sphere:{m}"]["exponents"]["position"] for m in ("rk4", "adaptive")}
    euler_t, euler_s = study["torus:euler"], study["sphere:euler"]
    # RK4 energy: the torus envelope is flat to L = 160 and then turns secular; the sphere stays flat to 320.
    plateau = T015_CHECKPOINTS.index(160.0) + 1
    torus_rk4 = study["torus:rk4"]["envelopes"]["energy"]
    sphere_rk4 = study["sphere:rk4"]["envelopes"]["energy"]
    rk4_plateau = {"torus_exponent_L_10_to_160": core.loglog_slope(T015_CHECKPOINTS[:plateau], torus_rk4[:plateau]),
                   "sphere_exponent_L_10_to_320": core.loglog_slope(T015_CHECKPOINTS, sphere_rk4),
                   "torus_local_slope_L_160_to_320": _local_slopes(T015_CHECKPOINTS, torus_rk4)[-1],
                   "torus_growth_factor_10_to_160": torus_rk4[plateau - 1] / torus_rk4[0],
                   "sphere_growth_factor_10_to_320": sphere_rk4[-1] / sphere_rk4[0]}
    local = {}
    for key, row in study.items():
        for name in ("energy", "clairaut"):
            values = row["envelopes"].get(name)
            if values and len(values) >= 2:
                local[f"{key} {name}"] = _local_slopes(T015_CHECKPOINTS[:len(values)], values)
    clair_envelope = study["torus:adaptive"]["envelopes"]["clairaut"]
    clair_from = T015_CHECKPOINTS.index(T015_CLAIRAUT_FROM)
    clair_adaptive = core.loglog_slope(T015_CHECKPOINTS[clair_from:], clair_envelope[clair_from:])
    clair_spread = core.slope_spread(T015_CHECKPOINTS[clair_from:], clair_envelope[clair_from:])
    # Sphere position error split into cross-track (orbit-plane precession) and along-track (phase) parts at L = 320.
    rk4_s, ad_s = study["sphere:rk4"], study["sphere:adaptive"]
    rk4_tilt = {"plane_tilt_exponent": rk4_s["exponents"]["plane_tilt"],
                "plane_tilt_spread": rk4_s["exponent_spreads"]["plane_tilt"],
                "plane_tilt_at_320": rk4_s["envelopes"]["plane_tilt"][-1],
                "momentum_size_at_320": rk4_s["envelopes"]["momentum_size"][-1],
                "angular_momentum_exponent": rk4_s["exponents"]["angular_momentum"]}
    rk4_mech = {"position_at_320": rk4_s["envelopes"]["position"][-1],
                "cross_track_at_320": rk4_s["envelopes"]["cross_track"][-1],
                "along_track_at_320": rk4_s["along_track"][-1],
                "integrated_speed_error_at_320": rk4_s["integrated_speed_error"][-1]}
    rk4_mech["cross_track_share"] = rk4_mech["cross_track_at_320"] / rk4_mech["position_at_320"]
    rk4_mech["cross_track_over_plane_tilt"] = rk4_mech["cross_track_at_320"] / rk4_tilt["plane_tilt_at_320"]
    rk4_mech["along_over_integrated_speed_error"] = (rk4_mech["along_track_at_320"]
                                                     / rk4_mech["integrated_speed_error_at_320"])
    ad_excess = [abs(v) for v in ad_s["integrated_speed_error"]]
    ad_mech = {"position_at_320": ad_s["envelopes"]["position"][-1], "along_track_at_320": ad_s["along_track"][-1],
               "integrated_speed_error_at_320": ad_s["integrated_speed_error"][-1],
               "cross_track_at_320": ad_s["envelopes"]["cross_track"][-1],
               "integrated_speed_error_exponent": core.loglog_slope(T015_CHECKPOINTS, ad_excess)}
    ad_mech["along_track_share"] = abs(ad_mech["along_track_at_320"]) / ad_mech["position_at_320"]
    ad_mech["along_over_integrated_speed_error"] = (ad_mech["along_track_at_320"]
                                                    / ad_mech["integrated_speed_error_at_320"])
    findings = [
        finding("Adaptive DP45 energy error grows linearly with length on the torus and the sphere", "numerical",
                adaptive_energy,
                {"generator": _gen("drift", rtol=T015_RTOL, checkpoints=list(T015_CHECKPOINTS)),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("invariant", f"{k} energy-envelope exponent minus 1", v - 1.0, 0.2)
                            for k, v in adaptive_energy.items()]},
                uncertainty=core.uncertainty("fit_spread",
                                             max(study[f"{k}:adaptive"]["exponent_spreads"]["energy"] for k in ("torus", "sphere")),
                                             "largest gap between the fitted exponent and consecutive-checkpoint "
                                             "slopes"),
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Sphere position error grows like L for fixed-step RK4 and like L^2 for adaptive DP45", "numerical",
                pos,
                {"generator": _gen("drift", h=T015_H),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("analytic", "RK4 position exponent minus 1", pos["rk4"] - 1.0, 0.2),
                            core.check("analytic", "adaptive position exponent minus 2", pos["adaptive"] - 2.0, 0.2)]},
                uncertainty=core.uncertainty("fit_spread",
                                             max(study[f"sphere:{m}"]["exponent_spreads"]["position"] for m in ("rk4", "adaptive")),
                                             "largest gap between the fitted exponent and consecutive-checkpoint "
                                             "slopes"),
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Sphere angular-momentum drift of fixed-step RK4 grows linearly with length and is a tilt of the "
                "orbit plane, not a change of its size", "numerical", rk4_tilt,
                {"generator": _gen("drift", method="rk4", h=T015_H),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("invariant", "RK4 orbit-plane tilt envelope exponent minus 1",
                                       rk4_tilt["plane_tilt_exponent"] - 1.0, 0.1),
                            core.check("invariant", "largest gap between the tilt exponent and consecutive-checkpoint "
                                       "slopes", rk4_tilt["plane_tilt_spread"], 0.1, "le"),
                            core.check("invariant", "angular-momentum size drift / plane tilt at L = 320",
                                       rk4_tilt["momentum_size_at_320"] / rk4_tilt["plane_tilt_at_320"], 0.05, "le")]},
                uncertainty=core.uncertainty("fit_spread", rk4_tilt["plane_tilt_spread"],
                                             "largest gap between the fitted tilt exponent and consecutive-checkpoint "
                                             "slopes"),
                tolerance={"abs": 0.02, "rel": 0.02}),
        finding("Fixed-step RK4 sphere position error is cross-track: the precessing orbit plane, not the speed "
                "error, sets it", "numerical", rk4_mech,
                {"generator": _gen("drift", method="rk4", h=T015_H),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("invariant", "cross-track envelope / position envelope at L = 320",
                                       rk4_mech["cross_track_share"], 0.99, "ge"),
                            core.check("invariant", "cross-track envelope / orbit-plane tilt at L = 320, minus 1",
                                       rk4_mech["cross_track_over_plane_tilt"] - 1.0, 0.05),
                            core.check("invariant", "along-track error / integrated speed error at L = 320 "
                                       "(nonpositive: opposite signs)", rk4_mech["along_over_integrated_speed_error"],
                                       0.0, "signed_le")]},
                uncertainty=core.uncertainty("model_truncation", abs(rk4_mech["cross_track_over_plane_tilt"] - 1.0),
                                             "gap between the cross-track envelope and the orbit-plane tilt at L = "
                                             "320 (the tilt bounds the cross-track excursion)"),
                tolerance={"abs": 1e-6, "rel": 0.05},
                counterexample={"statement": "The fixed-step RK4 position error on the sphere is the phase error of "
                                             "its speed error (a constant speed error gives position error ~ L)",
                                "witness": {"method": "rk4", "h": T015_H, "L": T015_CHECKPOINTS[-1], **rk4_mech}}),
        finding("Adaptive DP45 sphere position error is along-track and equals its integrated speed error, which "
                "grows like L^2", "numerical", ad_mech,
                {"generator": _gen("drift", method="adaptive", rtol=T015_RTOL),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("analytic", "along-track error / integrated speed error at L = 320, minus 1",
                                       ad_mech["along_over_integrated_speed_error"] - 1.0, 0.02),
                            core.check("invariant", "|along-track error| / position envelope at L = 320",
                                       ad_mech["along_track_share"], 0.99, "ge"),
                            core.check("analytic", "exponent of |integrated speed error| minus 2",
                                       ad_mech["integrated_speed_error_exponent"] - 2.0, 0.2)]},
                uncertainty=core.uncertainty("model_truncation", abs(ad_mech["along_over_integrated_speed_error"] - 1.0),
                                             "relative gap between the along-track error and the integrated speed "
                                             "error at L = 320"),
                tolerance={"abs": 1e-6, "rel": 0.05}),
        finding("Fixed-step RK4 energy error has a flat (oscillation-dominated) envelope up to L = 160 on the torus "
                "and L = 320 on the sphere; on the torus a secular term emerges between L = 160 and 320",
                "numerical", rk4_plateau,
                {"generator": _gen("drift", h=T015_H),
                 "checks": [core.check("invariant", "torus RK4 energy-envelope exponent over L = 10-160",
                                       rk4_plateau["torus_exponent_L_10_to_160"], 0.5, "le"),
                            core.check("invariant", "sphere RK4 energy-envelope exponent over L = 10-320",
                                       rk4_plateau["sphere_exponent_L_10_to_320"], 0.5, "le"),
                            core.check("invariant", "torus RK4 local envelope slope from L = 160 to 320",
                                       rk4_plateau["torus_local_slope_L_160_to_320"], 0.5, "ge")]},
                uncertainty=core.uncertainty("fit_spread",
                                             max(core.slope_spread(T015_CHECKPOINTS[:plateau], torus_rk4[:plateau]),
                                                 core.slope_spread(T015_CHECKPOINTS, sphere_rk4)),
                                             "largest gap between a plateau exponent and its consecutive-checkpoint "
                                             "slopes"),
                tolerance={"abs": 0.02, "rel": 0.01},
                counterexample={"statement": "The energy error of a non-symplectic fixed-step integrator grows "
                                             "linearly with length at every horizon",
                                "witness": {"method": "rk4", "h": T015_H, "surface": "Torus(2, 1)",
                                            "lengths": list(T015_CHECKPOINTS[:plateau]),
                                            "energy_envelope": torus_rk4[:plateau],
                                            "exponent": rk4_plateau["torus_exponent_L_10_to_160"]}}),
        finding("Torus Clairaut drift of adaptive DP45 grows linearly with length from L = 40 on", "numerical",
                {"exponent_L_40_to_320": clair_adaptive,
                 "local_slopes": _local_slopes(T015_CHECKPOINTS, clair_envelope)},
                {"generator": _gen("drift", method="adaptive", rtol=T015_RTOL, fit_from=T015_CLAIRAUT_FROM),
                 "checks": [core.check("invariant", "adaptive Clairaut-envelope exponent over L = 40-320 minus 1",
                                       clair_adaptive - 1.0, 0.1),
                            core.check("invariant", "largest gap between that exponent and its consecutive-checkpoint "
                                       "slopes", clair_spread, 0.1, "le")]},
                uncertainty=core.uncertainty("fit_spread", clair_spread,
                                             "largest gap between the adaptive fitted exponent and consecutive-"
                                             "checkpoint slopes over L = 40-320"),
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Euler on the torus keeps a bounded energy error but changes the orbit type (Clairaut drift)",
                "numerical", {"max_energy_error": euler_t["max_energy_error"],
                              "clairaut_drift_at_320": euler_t["envelopes"]["clairaut"][-1],
                              "clairaut_initial": euler_t["clairaut_initial"], "theta_max_abs": euler_t["theta_max_abs"],
                              "theta_turning": euler_t["theta_turning"]},
                {"generator": _gen("drift", method="euler", h=T015_H),
                 "checks": [core.check("invariant", "max energy error of the Euler run", euler_t["max_energy_error"],
                                       0.1, "le"),
                            core.check("invariant", "max |theta| minus exact turning latitude",
                                       euler_t["theta_max_abs"] - euler_t["theta_turning"], 1.0, "signed_ge")]},
                uncertainty=core.uncertainty("method_error", euler_t["max_energy_error"],
                                             "the Euler run's own speed error; the orbit change is a property of "
                                             "this discretization, not of the geodesic"),
                tolerance={"abs": 1e-6, "rel": 1e-2},
                counterexample={"statement": "A bounded energy (speed) error implies a qualitatively correct "
                                             "long-horizon geodesic",
                                "witness": {"surface": "Torus(2, 1)", "method": "euler", "h": T015_H,
                                            "max_energy_error": euler_t["max_energy_error"],
                                            "theta_turning": euler_t["theta_turning"],
                                            "theta_max_abs": euler_t["theta_max_abs"]}}),
        finding("Euler on the sphere leaves the polar chart before the horizon", "numerical",
                euler_s["failed_at"] if euler_s["failed_at"] is not None else euler_s["reached"],
                {"generator": _gen("drift", method="euler", h=T015_H),
                 "checks": [core.check("invariant", "arclength at which the Euler run was stopped",
                                       euler_s["failed_at"] if euler_s["failed_at"] is not None else 1e9,
                                       T015_CHECKPOINTS[-1], "le")]},
                unit="arclength",
                uncertainty=core.uncertainty("step_quantization", T015_H, "the stop is located to one fixed step"),
                tolerance={"abs": T015_H, "rel": 0.0}),
    ]
    fields = _fields(
        hypothesis=("Non-symplectic integrators drift in the first integrals of the geodesic flow (speed, Clairaut "
                    "constant, angular momentum); adaptive local-error control accumulates a linear secular drift, "
                    "while a fixed step on these closed or quasi-periodic orbits can keep the speed drift bounded "
                    "over long stretches; bounded speed error does not guarantee the right orbit."),
        mathematical_model=("Energy g(v, v) = 1, torus Clairaut rho^2 phi' and sphere angular momentum M = X x X' are "
                            "exact first integrals. On the unit sphere |M| is the speed and M/|M| the normal of the "
                            "orbit plane, so a position error splits into a cross-track part, bounded by the tilt of "
                            "M/|M|, and an along-track (phase) part. A speed error advances the point along its circle "
                            "by the integrated excess arclength int (|X'| - 1) ds; a speed error growing ~ L therefore "
                            "gives along-track error ~ L^2 (adaptive DP45). A tilt of the orbit plane growing ~ L gives "
                            "cross-track error ~ L (fixed-step RK4, whose local error rotates the plane secularly while "
                            "its speed error stays bounded). On Torus(2, 1) "
                            "the orbit oscillates between |theta| <= arccos((c - R)/r); a Clairaut drift across the "
                            "separatrix changes it into an orbit winding around the tube."),
        input_data=["Torus(2, 1) start (0, 0.5) heading 0.7 (bounded oscillation about the outer equator)",
                    "Unit sphere start (pi/2, 0) heading 1.0 (great circle)",
                    f"Fixed step h = {T015_H} for Euler, midpoint, RK4; DP45 rtol {T015_RTOL}, atol {T015_RTOL * 1e-3}; "
                    f"checkpoints L = {', '.join(_g(c, 4) for c in T015_CHECKPOINTS)}"],
        observation_model=("Running maxima (envelopes) of |I(s) - I(0)| at checkpoints; exponents are log-log slopes "
                           "over the reached checkpoints; no renormalization at any step."),
        expected_invariant=("Adaptive drift exponent 1; sphere position exponent 1 (fixed RK4, cross-track, from "
                            "plane precession) and 2 (adaptive, along-track, from the integrated speed error)."),
        experiment=("One run per surface and method to L = 320 (fixed steps stop at a nonfinite state or at the "
                    "sphere chart's poles); envelopes and fits of energy, Clairaut, angular momentum (split into "
                    "orbit-plane tilt and size) and position error (split into cross-track and along-track parts, "
                    "the latter compared with the integrated speed error)."),
        numerical_result=(f"Energy exponents: adaptive {', '.join(f'{k} {_g(v, 3)}' for k, v in adaptive_energy.items())}; "
                          f"RK4 envelope exponent {_g(rk4_plateau['torus_exponent_L_10_to_160'], 3)} (torus, L = 10-160) and "
                          f"{_g(rk4_plateau['sphere_exponent_L_10_to_320'], 3)} (sphere, L = 10-320), torus RK4 local slope "
                          f"{_g(rk4_plateau['torus_local_slope_L_160_to_320'], 3)} from L = 160 to 320; sphere position "
                          f"exponent RK4 {_g(pos['rk4'], 3)}, adaptive {_g(pos['adaptive'], 3)}; at L = 320 the RK4 "
                          f"position error {_g(rk4_mech['position_at_320'], 3)} is cross-track "
                          f"{_g(rk4_mech['cross_track_at_320'], 3)} (orbit-plane tilt {_g(rk4_tilt['plane_tilt_at_320'], 3)}, "
                          f"tilt exponent {_g(rk4_tilt['plane_tilt_exponent'], 3)}), its along-track part "
                          f"{_g(rk4_mech['along_track_at_320'], 2)} has the opposite sign to the integrated speed error "
                          f"{_g(rk4_mech['integrated_speed_error_at_320'], 2)}; the adaptive position error "
                          f"{_g(ad_mech['position_at_320'], 3)} is along-track {_g(ad_mech['along_track_at_320'], 4)} "
                          f"against an integrated speed error {_g(ad_mech['integrated_speed_error_at_320'], 4)}; torus "
                          f"adaptive Clairaut exponent {_g(clair_adaptive, 3)} over L = 40-320 (local slopes "
                          f"{_slopes_text(local['torus:adaptive clairaut'], 3)}); fixed-step Clairaut envelopes are not "
                          "described by one "
                          f"exponent (local slopes: Euler {_slopes_text(local['torus:euler clairaut'])}, RK4 "
                          f"{_slopes_text(local['torus:rk4 clairaut'])}); torus Euler max energy error "
                          f"{_g(euler_t['max_energy_error'], 3)} but max |theta| {_g(euler_t['theta_max_abs'], 4)} versus "
                          f"turning latitude {_g(euler_t['theta_turning'], 4)}; sphere Euler stopped at s = "
                          f"{_g(euler_s['failed_at'], 4) if euler_s['failed_at'] is not None else 'never'}."),
        uncertainty=("Envelope exponents mix oscillatory and secular parts, so single-exponent fits are reported only "
                     "where the local slopes agree; the torus RK4 energy local slopes are "
                     f"{_slopes_text(local['torus:rk4 energy'])} over L = 10-320, so its secular part appears between "
                     "L = 160 and 320 and may dominate beyond; the adaptive torus Clairaut slopes settle only from "
                     "L = 40, so its exponent is fitted there. Fits use the six checkpoints or the stated subrange. "
                     "The sphere position decomposition is linear in errors below "
                     f"{_g(max(rk4_mech['position_at_320'], ad_mech['position_at_320']), 2)} (RK4 and adaptive), so "
                     "second-order cross terms are negligible."),
        failure_modes_checked=["no hidden renormalization of speed",
                               "position error decomposed into cross-track and along-track parts before a mechanism "
                               "is attributed (the speed-error mechanism is refuted for fixed-step RK4)",
                               "chart exit at the sphere poles detected and "
                               "reported instead of integrating through the singularity",
                               "nonfinite states stop the run", "dyadic step so every checkpoint is a node"],
        unresolved_assumptions=["Two geodesics only; resonant or chaotic geodesics are not sampled",
                                "Horizon 320 is limited by the run budget; asymptotic drift laws are not established",
                                "Symplectic or symmetric integrators are not in the core and are not compared"],
        recommended_next_task="T016 (negative curvature) and a symmetric-integrator comparison on the same horizons",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T016
T016_K = (1.0, 2.0, 4.0, 8.0)
T016_LENGTH, T016_STEPS, T016_TAU = 2.0, 128, 1e-6
T016_ADAPTIVE_RTOL = 1e-10
T016_SADDLE_C = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0)
T016_SADDLE_FIT_FROM = 16.0
# Implicit one-step methods compared with RK4 on the constant-curvature Jacobi system: (name, order,
# leading error constant c with R(z) = e^z (1 + c z^(p+1) + ...) on the growing mode).
T016_IMPLICIT = (("implicit-midpoint", 2, 1.0 / 12.0), ("gauss-legendre-2", 4, 1.0 / 720.0))
RK4_STABILITY = 2.785293563405282   # |R(-x)| <= 1 for the classical RK4 polynomial exactly when x <= this


def _saddle_start(c):
    """x0 < 0 with arclength 1 from (x0, 0) to the saddle point along the ridge geodesic y = 0."""
    def arclength(x):
        return 0.5 * x * math.sqrt(1 + (c * x) ** 2) + math.asinh(c * x) / (2 * c)
    low, high = 0.0, 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if arclength(mid) < 1.0 else (low, mid)
    return -0.5 * (low + high)


def _required_steps(method, k, length, exact):
    def error(n):
        return abs(core.constant_curvature_transfer(method, -k * k, length, n)[0, 1] - exact) / exact
    steps = core.minimal_steps(error, T016_TAU)
    # The same method evaluated through its scalar stability function on the eigenmodes +k and -k.
    modes = core.hyperbolic_j_head_by_modes(method, k, length, steps)
    matrix = core.constant_curvature_transfer(method, -k * k, length, steps)[0, 1]
    return steps, abs(matrix - modes) / abs(modes)


def _predicted_steps(order, constant, k, length):
    """Smallest N with relative error N |c| (kh)^(p+1) = tau on the growing mode: L (L |c| k^(p+1) / tau)^(1/p)."""
    return length * (length * constant * k ** (order + 1) / T016_TAU) ** (1.0 / order)


def negative_curvature_study():
    u0, heading, length = (0.0, 1.0), 0.6, T016_LENGTH
    rows = []
    for k in T016_K:
        plane = HyperbolicPlane(k)
        exact = math.sinh(k * length) / k
        adaptive = jacobi.transfer(plane, u0, heading, length, rtol=T016_ADAPTIVE_RTOL, atol=1e-13)
        fixed = jacobi.transfer(plane, u0, heading, length, steps=T016_STEPS)
        h = length / T016_STEPS
        start = np.asarray(u0, dtype=float)
        end_exact = plane.exact_geodesic(start, plane.unit_tangent(start, heading), [length])[0]
        linear = core.constant_curvature_transfer("rk4", -k * k, length, T016_STEPS)
        # Growth and decay rates measured from the integrated transfer matrix Phi(L) (both Jacobi columns): its
        # eigenvalues are exp(+/- kL) for constant K = -k^2. The decaying one is resolvable only while it exceeds the
        # absolute integration error of Phi, about rtol times its largest entry.
        phi = adaptive.matrix()
        low, high = np.sort(np.abs(np.linalg.eigvals(phi)))
        decay_resolved = bool(low >= 100.0 * T016_ADAPTIVE_RTOL * float(np.max(np.abs(phi))))
        row = {
            "k": k, "exact_j_head": exact, "adaptive_j_head": float(adaptive.states[-1, 6]),
            "adaptive_relative_error": abs(adaptive.states[-1, 6] - exact) / exact,
            "adaptive_accepted_steps": adaptive.stats["accepted_steps"],
            "growth_rate": math.log(high) / length,
            "decay_rate": -math.log(low) / length if decay_resolved else None,
            "decay_resolved": decay_resolved,
            "rk4_relative_error": abs(fixed.states[-1, 6] - exact) / exact,
            "rk4_predicted_relative_error": length * k ** 5 * h ** 4 / 120,
            "linear_system_matches_full_system": abs(linear[0, 1] - fixed.states[-1, 6]) / exact,
            "geodesic_endpoint_distance_error": core.hyperbolic_distance(k, fixed.states[-1, :2], end_exact),
            "rk4_steps_predicted": _predicted_steps(4, 1.0 / 120.0, k, length),
            "rk4_stability_steps": math.ceil(k * length / RK4_STABILITY)}
        row["rk4_steps_required"], row["rk4_modes_mismatch"] = _required_steps("rk4", k, length, exact)
        for method, order, constant in T016_IMPLICIT:
            row[f"{method}_steps_required"], row[f"{method}_modes_mismatch"] = _required_steps(method, k, length, exact)
            row[f"{method}_steps_predicted"] = _predicted_steps(order, constant, k, length)
        rows.append(row)
    # Implicit midpoint beyond its pole kh = 2: the growing mode's factor turns negative.
    k, steps = 8.0, 7
    h = T016_LENGTH / steps
    trajectory = [np.eye(2)[:, 1]]
    trajectory_rk4 = [np.eye(2)[:, 1]]
    for _ in range(steps):
        trajectory.append(core.step_matrix("implicit-midpoint", -k * k, h) @ trajectory[-1])
        trajectory_rk4.append(core.step_matrix("rk4", -k * k, h) @ trajectory_rk4[-1])
    j_im = [float(v[0]) for v in trajectory]
    j_rk4 = [float(v[0]) for v in trajectory_rk4]
    sign_changes = sum(1 for a, b in zip(j_im[1:], j_im[2:]) if a * b < 0)
    beyond_pole = {"k": k, "h": h, "kh": k * h, "j_implicit_midpoint": j_im, "j_rk4": j_rk4,
                   "j_exact": [math.sinh(k * h * n) / k for n in range(steps + 1)],
                   "implicit_midpoint_sign_changes": sign_changes}
    saddle = []
    for c in T016_SADDLE_C:
        surface = Saddle(c)
        x0 = _saddle_start(c)
        run = jacobi.transfer(surface, (x0, 0.0), 0.0, 2.0, rtol=1e-9, atol=1e-12)
        tight = jacobi.transfer(surface, (x0, 0.0), 0.0, 2.0, rtol=1e-11, atol=1e-14)
        saddle.append({"c": c, "x0": x0, "peak_abs_curvature": c * c, "j_head": float(run.states[-1, 6]),
                       "self_convergence": abs(run.states[-1, 6] - tight.states[-1, 6]) / abs(tight.states[-1, 6]),
                       "accepted_steps": run.stats["accepted_steps"],
                       # y = 0 and v_y = 0 make the y-equations vanish: the ridge is kept by construction.
                       "max_abs_y": float(np.max(np.abs(run.states[:, 1]))),
                       "end_symmetry": abs(run.states[-1, 0] + x0)})
    return {"hyperbolic": rows, "beyond_pole": beyond_pole, "saddle": saddle}


@task("T016", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t016_negative_curvature_is_not_stiffness",))
def negative_curvature(ctx):
    study = ctx.memo("gjl-negative-curvature", negative_curvature_study)
    ctx.artifact_json("negative-curvature.json", core.jsonable(study, 12))
    rows, saddle = study["hyperbolic"], study["saddle"]
    ks = [r["k"] for r in rows]
    ctx.artifact_text("steps-versus-k.svg", svg.line_plot(
        [("RK4 steps for rel. error 1e-6", ks, [r["rk4_steps_required"] for r in rows]),
         ("implicit midpoint steps", ks, [r["implicit-midpoint_steps_required"] for r in rows]),
         ("2-stage Gauss-Legendre steps", ks, [r["gauss-legendre-2_steps_required"] for r in rows]),
         ("DP45 accepted steps (rtol 1e-10)", ks, [r["adaptive_accepted_steps"] for r in rows]),
         ("RK4 stability limit", ks, [r["rk4_stability_steps"] for r in rows])],
        title="HyperbolicPlane(k), L = 2: steps versus k", xlabel="k (K = -k^2)", ylabel="steps", logx=True, logy=True))
    cs = [r["c"] for r in saddle]
    ctx.artifact_text("saddle-growth.svg", svg.line_plot(
        [("j_head(L)", cs, [r["j_head"] for r in saddle]), ("peak |K| = c^2", cs, [r["peak_abs_curvature"] for r in saddle]),
         ("DP45 accepted steps", cs, [r["accepted_steps"] for r in saddle])],
        title="Saddle(c): ridge geodesic crossing the saddle point", xlabel="c", ylabel="value", logx=True, logy=True))
    # Exponential growth measured from the integrated (adaptive) Jacobi field, not from the closed form.
    growth = float(np.polyfit([k * T016_LENGTH for k in ks], [math.log(2 * r["k"] * r["adaptive_j_head"]) for r in rows],
                              1)[0])
    growth_rate_error = max(abs(r["growth_rate"] / r["k"] - 1.0) for r in rows)
    decay_rows = [r for r in rows if r["decay_resolved"]]
    decay_rate_error = max(abs(r["decay_rate"] / r["k"] - 1.0) for r in decay_rows)
    rk4_exp = core.loglog_slope(ks, [r["rk4_relative_error"] for r in rows])
    ratio_pred = [r["rk4_relative_error"] / r["rk4_predicted_relative_error"] for r in rows]
    steps_exp = core.loglog_slope(ks, [r["rk4_steps_required"] for r in rows])
    adaptive_exp = core.loglog_slope(ks, [r["adaptive_accepted_steps"] for r in rows])
    implicit = {}
    for method, order, _ in T016_IMPLICIT:
        required = [r[f"{method}_steps_required"] for r in rows]
        implicit[method] = {"steps_required": required, "exponent": core.loglog_slope(ks, required),
                            "predicted_exponent": (order + 1) / order,
                            "required_over_predicted": [r[f"{method}_steps_required"] / r[f"{method}_steps_predicted"]
                                                        for r in rows],
                            "over_rk4_steps": [r[f"{method}_steps_required"] / r["rk4_steps_required"] for r in rows],
                            "spread": core.slope_spread(ks, required)}
    modes_mismatch = max(r[f"{m}_modes_mismatch"] for r in rows for m in ("rk4", "implicit-midpoint", "gauss-legendre-2"))
    linear_match = max(r["linear_system_matches_full_system"] for r in rows)
    stiffness = [r["rk4_steps_required"] / r["rk4_stability_steps"] for r in rows]
    geo_err = [r["geodesic_endpoint_distance_error"] for r in rows]
    large = [r for r in saddle if r["c"] >= T016_SADDLE_FIT_FROM]
    saddle_exp = core.loglog_slope([r["c"] for r in large], [r["j_head"] for r in large])
    saddle_local = _local_slopes(cs, [r["j_head"] for r in saddle])
    saddle_decrease = max(b - a for a, b in zip(saddle_local[2:], saddle_local[3:]))
    increments = [b["accepted_steps"] - a["accepted_steps"] for a, b in zip(saddle, saddle[1:])]
    beyond = study["beyond_pole"]
    im, gl = implicit["implicit-midpoint"], implicit["gauss-legendre-2"]
    # Same order p = 4, so N_GL / N_RK4 = (|c_GL| / |c_RK4|)^(1/4) = (120/720)^(1/4) from N(tau) ~ |c|^(1/p).
    gl_over_rk4_predicted = (120.0 / 720.0) ** 0.25
    gl_over_rk4 = max(abs(v / gl_over_rk4_predicted - 1.0) for v in gl["over_rk4_steps"])
    step_checks = [core.check("cross_implementation", "max relative difference of j_head between matrix powers and "
                              "the scalar stability function on the eigenmodes (RK4, implicit midpoint, Gauss-Legendre, "
                              "at the required N)", modes_mismatch, 1e-10),
                   core.check("cross_implementation", "max |RK4 matrix-power j_head - integrated j_head| / j_head "
                              f"(full geodesic/Jacobi system, N = {T016_STEPS})", linear_match, 1e-12)]
    findings = [
        finding("Jacobi fields on HyperbolicPlane(k) grow like sinh(kL)/k and adaptive integration resolves them",
                "numerical", {"max_adaptive_relative_error": max(r["adaptive_relative_error"] for r in rows),
                              "j_head_at_k_8": rows[-1]["adaptive_j_head"]},
                {"generator": _gen("hyperbolic-k", k=list(T016_K), length=T016_LENGTH, rtol=1e-10),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "max relative error of j_head(L) against sinh(kL)/k",
                                       max(r["adaptive_relative_error"] for r in rows), 1e-8)]},
                uncertainty=core.uncertainty("reference_error", max(r["adaptive_relative_error"] for r in rows),
                                             "closed-form reference; the adaptive relative error is the whole "
                                             "uncertainty"),
                tolerance={"abs": 1e-8, "rel": 1e-9}),
        finding("Fixed-step RK4 relative error grows like k^5 and matches L k^5 h^4 / 120", "numerical",
                {"exponent": rk4_exp, "ratio_to_prediction": ratio_pred},
                {"generator": _gen("hyperbolic-k", steps=T016_STEPS),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "fitted exponent minus 5", rk4_exp - 5.0, 0.2),
                            core.check("analytic", "max |measured / predicted - 1|",
                                       max(abs(v - 1.0) for v in ratio_pred), 0.2)]},
                uncertainty=core.uncertainty("fit_spread",
                                             core.slope_spread(ks, [r["rk4_relative_error"] for r in rows]),
                                             "largest gap between the fitted exponent and consecutive-k slopes "
                                             "(higher powers of kh)"),
                tolerance={"abs": 1e-3, "rel": 1e-4}),
        finding("RK4 steps for relative accuracy 1e-6 grow like k^(5/4); DP45 accepted steps grow about linearly in k",
                "numerical", {"rk4_exponent": steps_exp, "adaptive_exponent": adaptive_exp,
                              "rk4_required_over_predicted": [r["rk4_steps_required"] / r["rk4_steps_predicted"]
                                                              for r in rows]},
                {"generator": _gen("hyperbolic-k", tau=T016_TAU),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "RK4 step exponent minus 5/4", steps_exp - 1.25, 0.1),
                            core.check("analytic", "max |required / predicted - 1|",
                                       max(abs(r["rk4_steps_required"] / r["rk4_steps_predicted"] - 1) for r in rows),
                                       0.2),
                            core.check("invariant", "DP45 step exponent minus 1 (local-error model h ~ 1/k)",
                                       adaptive_exp - 1.0, 0.2)] + step_checks},
                uncertainty=core.uncertainty("fit_spread",
                                             core.slope_spread(ks, [r["rk4_steps_required"] for r in rows]),
                                             "largest gap between the fitted exponent and consecutive-k slopes of "
                                             "integer step counts"),
                tolerance={"abs": 0.03, "rel": 0.0}),
        finding("This is intrinsic exponential instability, not stiffness: the integrated Jacobi flow grows and "
                "decays at the same rate k and accuracy, not stability, sets the step",
                "numerical", {"steps_required_over_stability_limit": stiffness,
                              "measured_rates": {_g(r["k"], 3): {"growth": r["growth_rate"], "decay": r["decay_rate"]}
                                                 for r in rows},
                              "log_growth_slope_in_kL": growth},
                {"generator": _gen("hyperbolic-k", rtol=T016_ADAPTIVE_RTOL),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "max over k of |log(largest eigenvalue of the integrated Phi(L)) "
                                       "/ (k L) - 1|", growth_rate_error, 1e-8),
                            core.check("analytic", "max over resolvable k of |-log(smallest eigenvalue of the "
                                       "integrated Phi(L)) / (k L) - 1|", decay_rate_error, 1e-8),
                            core.check("invariant", "min over k of accuracy steps / RK4 stability steps", min(stiffness),
                                       5.0, "ge"),
                            core.check("analytic", "slope of log(2k j_head) from the adaptive runs in kL, minus 1",
                                       growth - 1.0, 0.01)]},
                uncertainty=core.uncertainty("quantization", 1.0 / min(r["rk4_stability_steps"] for r in rows),
                                             "integer stability step counts (ceil) limit the ratio to one step"),
                tolerance={"abs": 1e-8, "rel": 0.05}),
        finding("A-stable implicit methods do not remove the growth of the required steps with k (implicit midpoint "
                "~ k^(3/2), 2-stage Gauss-Legendre ~ k^(5/4)); implicit midpoint is qualitatively wrong beyond kh = 2",
                "numerical",
                {"implicit_midpoint": {"steps_required": im["steps_required"], "exponent": im["exponent"]},
                 "gauss_legendre_2": {"steps_required": gl["steps_required"], "exponent": gl["exponent"],
                                      "over_rk4_steps": gl["over_rk4_steps"]},
                 "sign_changes_at_kh": [beyond["kh"], beyond["implicit_midpoint_sign_changes"]]},
                {"generator": _gen("implicit-methods", methods=[m for m, _, _ in T016_IMPLICIT], k=list(T016_K),
                                   tau=T016_TAU),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "implicit-midpoint step exponent minus 3/2", im["exponent"] - 1.5,
                                       0.15),
                            core.check("analytic", "Gauss-Legendre step exponent minus 5/4", gl["exponent"] - 1.25, 0.1),
                            core.check("analytic", "max |required / predicted - 1| over both implicit methods",
                                       max(abs(v - 1.0) for m in implicit.values() for v in m["required_over_predicted"]),
                                       0.2),
                            core.check("analytic", "max |GL/RK4 steps / (120/720)^(1/4) - 1| (same order p = 4, "
                                       "error constants 1/720 and 1/120)", gl_over_rk4, 0.05),
                            core.check("invariant", "sign changes of the implicit-midpoint j at kh > 2",
                                       beyond["implicit_midpoint_sign_changes"], 1, "ge")] + step_checks},
                uncertainty=core.uncertainty("fit_spread", max(m["spread"] for m in implicit.values()),
                                             "largest gap between a fitted implicit step exponent and its "
                                             "consecutive-k slopes"),
                tolerance={"abs": 0.0, "rel": 0.05},
                counterexample={"statement": "An implicit (A-stable) integrator removes the growth of the step count "
                                             "with k on strongly negatively curved surfaces",
                                "witness": {"k": list(T016_K), "implicit_midpoint_steps": im["steps_required"],
                                            "gauss_legendre_2_steps": gl["steps_required"],
                                            "rk4_steps": [r["rk4_steps_required"] for r in rows],
                                            "beyond_pole_kh": beyond["kh"],
                                            "j_implicit_midpoint": beyond["j_implicit_midpoint"],
                                            "j_exact": beyond["j_exact"]}}),
        finding("Saddle(c): peak |K| = c^2 but Jacobi growth is polynomial in c; the local exponent of j_head(L) "
                "decreases toward sqrt(2)", "numerical",
                {"fitted_exponent_c_16_to_16384": saddle_exp, "local_exponents": saddle_local,
                 "j_head": [r["j_head"] for r in saddle], "heuristic_exponent": math.sqrt(2.0)},
                {"generator": _gen("saddle-ridge", c=list(T016_SADDLE_C), rtol=1e-9),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("self_convergence", "max relative change of j_head(L), rtol 1e-9 -> 1e-11",
                                       max(r["self_convergence"] for r in saddle), 1e-6),
                            core.check("invariant", "max |x(L) + x0| on the ridge geodesic (mirror symmetry)",
                                       max(r["end_symmetry"] for r in saddle), 1e-9),
                            core.check("invariant", "log j_head(L) / (sqrt(peak |K|) L) at the largest c (exponential "
                                       "growth would give about 1)", math.log(saddle[-1]["j_head"])
                                       / (saddle[-1]["c"] * 2.0), 0.1, "le"),
                            core.check("invariant", "largest change of consecutive local exponents from c = 16 on "
                                       "(nonpositive: decreasing)", saddle_decrease, 0.0, "signed_le"),
                            core.check("analytic", "last local exponent (c = 4096 -> 16384) minus the matched-"
                                       "asymptotics value sqrt(2)", saddle_local[-1] - math.sqrt(2.0), 0.01)]},
                uncertainty=core.uncertainty("truncation_bound", max(r["self_convergence"] for r in saddle),
                                             "relative change of j_head(L) when rtol is tightened from 1e-9 to "
                                             "1e-11; local exponents inherit about twice this"),
                tolerance={"abs": 0.0, "rel": 1e-4},
                counterexample={"statement": "Jacobi growth is exponential in sqrt(peak |K|) times the length",
                                "witness": {"c": saddle[-1]["c"], "peak_abs_curvature": saddle[-1]["peak_abs_curvature"],
                                            "log_j_head": math.log(saddle[-1]["j_head"]),
                                            "sqrt_peak_times_L": 2.0 * saddle[-1]["c"]}}),
    ]
    fields = _fields(
        hypothesis=("On K = -k^2 the Jacobi fields grow like e^(kL), so absolute errors are amplified by e^(kL) and a "
                    "fixed relative accuracy needs steps growing with k; this is intrinsic instability of the flow "
                    "(eigenvalues +k and -k of the Jacobi linearization), not classical stiffness, so A-stable "
                    "implicit methods do not remove it. Concentrated negative curvature (Saddle with large c) does "
                    "not produce exponential growth."),
        mathematical_model=("j'' = k^2 j, j_head = sinh(kL)/k. A one-step method of order p with R(z) = e^z (1 + c "
                            "z^(p+1) + ...) on the growing mode has relative error N |c| (kh)^(p+1), so N(tau) = L (L |c| "
                            "k^(p+1) / tau)^(1/p) ~ k^((p+1)/p): RK4 c = -1/120 (k^(5/4)), implicit midpoint c = 1/12 "
                            "(k^(3/2), pole at kh = 2 and negative factor beyond), 2-stage Gauss-Legendre (the (2, 2) "
                            "Pade approximant) c = -1/720 (k^(5/4), no real pole). RK4 stability for the decaying mode "
                            "needs kh <= 2.785. Saddle ridge y = 0: K = -c^2/(1 + c^2 x^2)^2 ~ -1/(4 s^2) away from the "
                            "saddle point, where j'' = j/(4 s^2) has solutions |s|^((1 +/- sqrt(2))/2); matching the "
                            "inbound and outbound power laws through the core of width 1/c gives j_head(L) ~ c^sqrt(2) "
                            "(matched asymptotics, not a proof)."),
        input_data=[f"HyperbolicPlane(k), k in {', '.join(_g(k, 3) for k in T016_K)}, start (0, 1), heading 0.6, "
                    f"L = {T016_LENGTH}",
                    f"RK4 N = {T016_STEPS}; DP45 rtol 1e-10; relative accuracy target {T016_TAU}; implicit midpoint "
                    "and 2-stage Gauss-Legendre as exact step matrices",
                    f"Saddle(c), c in {', '.join(_g(c, 5) for c in T016_SADDLE_C)}, ridge geodesic y = 0 from "
                    "arclength 1 before the saddle point, L = 2, DP45 rtol 1e-9 (checked at 1e-11)"],
        observation_model=("Relative error of j_head(L); required steps by doubling and bisection on the exact step "
                           "matrices of RK4, implicit midpoint and 2-stage Gauss-Legendre (matrix powers, checked "
                           "against the scalar stability function on the eigenmodes, and for RK4 against the full "
                           "geodesic/Jacobi integration); hyperbolic distance of the geodesic endpoint to the exact "
                           "semicircle."),
        expected_invariant=("Exponents 5 (error), 5/4 (RK4 and Gauss-Legendre steps), 3/2 (implicit midpoint), ~1 "
                            "(DP45 steps); Gauss-Legendre/RK4 step ratio (120/720)^(1/4) = 0.639 at every k; saddle "
                            "local exponents approaching sqrt(2)."),
        experiment=("Integrate the joint geodesic/Jacobi system per k (adaptive and RK4), measure the growth and "
                    "decay rates from the eigenvalues of the integrated transfer matrix, search the minimal step "
                    "counts, compare with "
                    "stability limits, iterate implicit midpoint beyond its pole, and integrate the saddle ridge "
                    "geodesic for c up to 16384."),
        numerical_result=(f"RK4 error exponent {_g(rk4_exp, 4)} (measured/predicted {_g(min(ratio_pred), 3)}-"
                          f"{_g(max(ratio_pred), 3)}); required steps for k = {', '.join(_g(k, 2) for k in ks)}: RK4 "
                          f"{', '.join(str(r['rk4_steps_required']) for r in rows)} (exponent {_g(steps_exp, 3)}), "
                          f"implicit midpoint {', '.join(str(v) for v in im['steps_required'])} (exponent "
                          f"{_g(im['exponent'], 3)}), 2-stage Gauss-Legendre {', '.join(str(v) for v in gl['steps_required'])} "
                          f"(exponent {_g(gl['exponent'], 3)}, {_g(min(gl['over_rk4_steps']), 3)} to "
                          f"{_g(max(gl['over_rk4_steps']), 3)} times the RK4 steps, predicted (120/720)^(1/4) = "
                          f"{_g(gl_over_rk4_predicted, 3)}), DP45 "
                          f"{', '.join(str(r['adaptive_accepted_steps']) for r in rows)} (exponent {_g(adaptive_exp, 3)}); "
                          f"accuracy/stability step ratio {_g(min(stiffness), 3)}-{_g(max(stiffness), 3)}; growth and "
                          f"decay rates of the integrated Phi(L) equal k to {_g(growth_rate_error, 2)} and "
                          f"{_g(decay_rate_error, 2)} relative (decay resolvable for k <= "
                          f"{_g(max(r['k'] for r in decay_rows), 3)}); RK4 geodesic endpoint error {_g(geo_err[0], 2)} "
                          f"(k = 1) to {_g(geo_err[-1], 3)} (k = 8); implicit midpoint at kh = {_g(beyond['kh'], 3)} "
                          f"changes sign {beyond['implicit_midpoint_sign_changes']} times; saddle j_head(L) "
                          f"{', '.join(_g(r['j_head'], 4) for r in saddle)} for c = {', '.join(_g(c, 5) for c in cs)} "
                          f"(fitted exponent {_g(saddle_exp, 3)} for c >= {_g(T016_SADDLE_FIT_FROM, 3)}, local exponents "
                          f"{_slopes_text(saddle_local[2:], 4)} from c = 16), DP45 steps grow by "
                          f"{', '.join(str(v) for v in increments)} per factor 4 in c."),
        uncertainty=("Step counts are integers found by bisection assuming monotone error in N; the saddle local "
                     f"exponents ({_slopes_text(saddle_local[2:], 4)} for c = 16 ... 16384) come from adaptive runs "
                     f"self-consistent to {_g(max(r['self_convergence'] for r in saddle), 2)} relative, so each "
                     "local exponent is accurate to about 1e-8; the sqrt(2) limit is a matched-asymptotics argument "
                     "supported by these slopes, not a proof."),
        failure_modes_checked=["matrix-power step search checked against the scalar stability functions on the "
                               "eigenmodes and, for RK4, against the full geodesic/Jacobi integration",
                               "implicit-midpoint singular step (kh = 2) refused rather than divided by zero",
                               "saddle ridge symmetry x(L) = -x0 checked (y = 0 holds by construction of the start)",
                               "adaptive saddle results checked by tightening rtol",
                               "exponential growth fitted from the integrated Jacobi field, not the closed form",
                               "growth and decay rates measured from the integrated transfer matrix, not from the "
                               "generator at the literal K = -k^2 (which would only recompute its eigenvalues); the "
                               "decay rate is claimed only where the decaying eigenvalue exceeds 100 rtol times the "
                               "largest entry of Phi(L)",
                               "implicit methods of order 2 and 4 both compared, so the step growth is not an order "
                               "artefact"],
        unresolved_assumptions=["The sqrt(2) saddle exponent is a matched-asymptotics argument, not a proof",
                                "Only the ridge geodesic of the saddle is studied; oblique geodesics sample other K",
                                "Implicit methods are evaluated on the constant-curvature Jacobi system through their "
                                "exact step matrices; no nonlinear implicit geodesic integrator is in the core",
                                "Error amplification of the geodesic in the half-plane chart mixes chart compression near "
                                "y = 0 with intrinsic instability"],
        recommended_next_task="T017 (validity domains, which shrink like 1/cosh(ks) on the hyperbolic plane) and T018",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T017
T017_NODE_FRACTIONS = (0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.01, 1.1, 1.25)
T017_SPHERE_FRACTIONS = (0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1.001, 1.01, 1.1, 1.25)
T017_HYPERBOLIC_S = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)
EPS_CAP = 1.0   # validity bounds are reported up to this heading perturbation


def remainder_fit(eps, remainder):
    """Least-squares r(eps) = C2 eps^2 + C3 eps^3 over the declared eps values."""
    eps = np.asarray(eps, dtype=float)
    design = np.column_stack([np.ones_like(eps), eps])
    (c2, c3), *_ = np.linalg.lstsq(design, np.asarray(remainder, dtype=float) / eps ** 2, rcond=None)
    return float(c2), float(c3)


def validity_bound(c2, c3, j, tau=TAU):
    """Smallest eps > 0 with |C2 eps + C3 eps^2| = tau |j|, capped at EPS_CAP."""
    target = tau * abs(j)
    if target == 0.0:
        return 0.0
    roots = []
    for sign in (1.0, -1.0):
        for root in np.roots([c3, c2, -sign * target]) if c3 != 0.0 else ([sign * target / c2] if c2 else []):
            if abs(np.imag(root)) < 1e-12 * max(1.0, abs(root)) and np.real(root) > 0:
                roots.append(float(np.real(root)))
    return min([EPS_CAP] + roots)


def _validity_row(s, j, distances):
    remainder = [d - e * abs(j) for d, e in zip(distances, EPS)]
    c2, c3 = remainder_fit(EPS, remainder)
    exponent = core.loglog_slope(EPS, [abs(r) for r in remainder]) if all(abs(r) > 0 for r in remainder) else None
    scale = max(abs(r) for r in remainder)
    residual = max(abs(r - (c2 * e ** 2 + c3 * e ** 3)) for r, e in zip(remainder, EPS)) / scale if scale else 0.0
    return {"s": float(s), "j": float(j), "C2": c2, "C3": c3, "remainder_exponent": exponent,
            "eps_max": validity_bound(c2, c3, j), "fit_residual": float(residual)}


def validity_study(ctx):
    sphere = Sphere(1.0)
    u0, heading = SPHERE_START
    cases = {}
    rows = []
    for f in T017_SPHERE_FRACTIONS:
        s = f * math.pi
        distances = [float(core.sphere_separation(sphere, u0, heading, [s], e)["distance"][0]) for e in EPS]
        row = _validity_row(s, core.sphere_first_order(s), distances)
        row.update(fraction=f, C3_analytic=-abs(math.sin(s)) * math.cos(s) ** 2 / 24,
                   eps_max_analytic=min(EPS_CAP, math.sqrt(24 * TAU) / abs(math.cos(s))))
        rows.append(row)
    cases["sphere-heading"] = {"singular_s": math.pi, "measure": "great-circle distance (closed form)", "rows": rows}
    rows = []
    s0 = 0.75 * math.pi
    for f in T017_NODE_FRACTIONS:
        s = f * s0
        distances = [float(core.sphere_separation(sphere, u0, heading, [s], e, 1.0)["distance"][0]) for e in EPS]
        row = _validity_row(s, core.sphere_first_order(s, 1.0), distances)
        row["fraction"] = f
        rows.append(row)
    cases["sphere-lateral-heading"] = {"singular_s": s0, "measure": "great-circle distance (closed form)", "rows": rows}
    rows = []
    for s in T017_HYPERBOLIC_S:
        distances = [float(core.hyperbolic_heading_distance(1.0, s, e)) for e in EPS]
        row = _validity_row(s, math.sinh(s), distances)
        row.update(C3_analytic=-math.sinh(s) * math.cosh(s) ** 2 / 24,
                   eps_max_analytic=min(EPS_CAP, math.sqrt(24 * TAU) / math.cosh(s)))
        rows.append(row)
    cases["hyperbolic-heading"] = {"singular_s": None, "measure": "hyperbolic distance (closed form)", "rows": rows}
    for key in ("generic", "equator"):
        family, coarse = torus_family(ctx, key), torus_family_coarse(ctx, key)
        rows = []
        for f in T017_NODE_FRACTIONS:
            i = int(round(f * TORUS_M))
            row = _validity_row(family["s"][i], family["j"][i], [family["runs"][e]["chord"][i] for e in EPS])
            # The same node on the grid with half the steps (every tabulated fine node is even).
            half = _validity_row(coarse["s"][i // 2], coarse["j"][i // 2],
                                 [coarse["runs"][e]["chord"][i // 2] for e in EPS])
            row.update(fraction=f, node=i, dj=float(family["dj"][i]), C2_half_grid=half["C2"],
                       eps_max_half_grid=half["eps_max"])
            rows.append(row)
        cases[f"torus-{key}"] = {"singular_s": float(family["s_star"]), "measure": "embedded chord (RK4)", "rows": rows}
    # Search for a violation of the predicted boundary at 0.9 s* on the generic torus path.
    family = torus_family(ctx, "generic")
    torus = Torus(2.0, 1.0)
    u0t, heading_t = TORUS_PATHS["generic"]
    node = int(round(0.9 * TORUS_M))
    row = next(r for r in cases["torus-generic"]["rows"] if r["fraction"] == 0.9)
    coarse = torus_family_coarse(ctx, "generic")
    probes = {}
    for label, factor in (("half", 0.5), ("double", 2.0)):
        eps = factor * row["eps_max"]
        start = jacobi.perturbed_start(torus, u0t, heading_t, heading_change=eps)
        relative = []
        # The probe on the family's grid and, for its grid dependence, on the grid with half the steps.
        for grid, steps in ((family, node), (coarse, node // 2)):
            step = grid["length"] / grid["steps"]
            _, states = integrators.integrate_fixed(torus.geodesic_rhs, start, step * steps, steps, "rk4")
            chord = float(np.linalg.norm(torus.embedding(states[-1, :2]) - grid["points"][steps]))
            first = eps * abs(float(grid["j"][steps]))
            relative.append(abs(chord - first) / first)
        model = abs(row["C2"] * eps + row["C3"] * eps ** 2) / abs(row["j"])
        probes[label] = {"eps": eps, "j": float(family["j"][node]), "relative_error": relative[0],
                         "relative_error_half_grid": relative[1], "model_relative_error": model}
    return {"tau": TAU, "eps": list(EPS), "cases": cases, "boundary_probe": {"s": float(family["s"][node]), **probes}}


@task("T017", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t017_validity_domains",))
def validity_domains(ctx):
    study = ctx.memo("gjl-validity", lambda: validity_study(ctx))
    ctx.artifact_json("validity-domains.json", core.jsonable(study, 12))
    series = []
    for key in ("sphere-heading", "sphere-lateral-heading", "torus-generic", "torus-equator"):
        case = study["cases"][key]
        series.append((key, [r["s"] / case["singular_s"] for r in case["rows"]],
                       [max(r["eps_max"], 1e-12) for r in case["rows"]]))
    ctx.artifact_text("validity-domains.svg", svg.line_plot(
        series, title=f"First-order validity bound eps_max (relative tolerance {TAU})",
        xlabel="s / s0 (first-order zero at 1)", ylabel="eps_max (floored at 1e-12)", logy=True))
    cases = study["cases"]
    gen = {r["fraction"]: r for r in cases["torus-generic"]["rows"]}
    eq = {r["fraction"]: r for r in cases["torus-equator"]["rows"]}
    mixed = {r["fraction"]: r for r in cases["sphere-lateral-heading"]["rows"]}
    sph = cases["sphere-heading"]["rows"]
    hyp = cases["hyperbolic-heading"]["rows"]
    sph_c2 = max(abs(r["C2"]) / abs(r["j"]) for r in sph)
    # At s = pi/2 the closed form has C3 = 0 and the remainder is roundoff; those rows carry no model residual.
    sph_modelled = [r for r in sph if abs(r["C3_analytic"]) > 1e-6]
    sph_c3 = max(abs(r["C3"] / r["C3_analytic"] - 1) for r in sph_modelled)
    sph_eps = max(abs(r["eps_max"] / r["eps_max_analytic"] - 1) for r in sph)
    near = {r["fraction"]: r["eps_max"] for r in sph if r["fraction"] in (0.999, 1.001)}
    hyp_eps = max(abs(r["eps_max"] / r["eps_max_analytic"] - 1) for r in hyp)
    hyp_c3_row = max(hyp, key=lambda r: abs(r["C3"] / r["C3_analytic"] - 1))
    hyp_c3 = abs(hyp_c3_row["C3"] / hyp_c3_row["C3_analytic"] - 1)
    # Higher-order (eps^5) terms leak into the fitted C2; compare it with the cubic term at the smallest eps.
    hyp_c2 = max(abs(r["C2"]) / (abs(r["C3"]) * EPS[0]) for r in hyp)
    eq_c2 = max(abs(r["C2"]) for r in eq.values())
    eq_c2_grid = max(abs(r["C2"] - r["C2_half_grid"]) for r in eq.values())
    gen_c2_grid = max(abs(r["C2"] - r["C2_half_grid"]) for r in gen.values())
    # Linear collapse of eps_max at a first-order zero s0: eps_max ~ slope |s - s0|. The two nodes one percent on
    # either side of s0 are averaged, which cancels the first-order variation of j' and C2 across s0.
    def collapse_slope(rows, s0):
        return (rows[0.99]["eps_max"] + rows[1.01]["eps_max"]) / (abs(rows[0.99]["s"] - s0) + abs(rows[1.01]["s"] - s0))
    s_star = cases["torus-generic"]["singular_s"]
    gen_slope = collapse_slope(gen, s_star)
    # Generic torus: the eps^2 remainder is normal, so tau |j| = |C2| eps gives slope tau |j'(s*)| / |C2(s*)|.
    gen_slope_predicted = TAU * abs(gen[1.0]["dj"]) / abs(gen[1.0]["C2"])
    s0_mixed = cases["sphere-lateral-heading"]["singular_s"]
    mixed_slope = collapse_slope(mixed, s0_mixed)
    # Sphere lateral+heading: the eps^2/2 offset at s0 is along-track, orthogonal to eps j N, so near s0
    # d = sqrt((eps j)^2 + (eps^2/2)^2) and tau = (1 + tau) - 1 gives eps_max = 2 |j| sqrt(2 tau + tau^2).
    mixed_slope_predicted = 2.0 * math.sqrt(2.0 * TAU + TAU ** 2) * abs(math.cos(s0_mixed) - math.sin(s0_mixed))
    probe = study["boundary_probe"]
    # Linear predictions tau/2 and 2 tau (context) and the full C2/C3 remainder model (checked).
    probe_gap = {"half": probe["half"]["relative_error"] / (TAU / 2) - 1.0,
                 "double": probe["double"]["relative_error"] / (2 * TAU) - 1.0}
    model_gap = {label: probe[label]["relative_error"] / probe[label]["model_relative_error"] - 1.0
                 for label in ("half", "double")}
    probe_grid = max(abs(probe[label]["relative_error"] - probe[label]["relative_error_half_grid"])
                     for label in ("half", "double"))
    probe_row = gen[0.9]
    findings = [
        finding("On a generic torus geodesic C2(s*) != 0 and the validity domain shrinks to zero linearly at the "
                "conjugate point",
                "numerical", {"C2_at_s_star": gen[1.0]["C2"], "eps_max_at_s_star": gen[1.0]["eps_max"],
                              "collapse_slope": gen_slope, "collapse_slope_predicted": gen_slope_predicted,
                              "eps_max_by_fraction": {str(f): r["eps_max"] for f, r in gen.items()}},
                {"generator": _gen("validity-torus-generic", eps=list(EPS), tau=TAU),
                 "derivation": _derivation("t017-validity-domains-of-the-first-order-approximation"),
                 "checks": [core.check("invariant", "|C2(s*)|", abs(gen[1.0]["C2"]), 0.1, "ge"),
                            core.check("analytic", "eps_max / |s - s*| at 0.99 s* and 1.01 s* against tau |j'(s*)| / "
                                       "|C2(s*)|, relative", gen_slope / gen_slope_predicted - 1.0, 0.01),
                            core.check("invariant", "eps_max at s*/2", gen[0.5]["eps_max"], 0.01, "ge")]},
                uncertainty=core.uncertainty("truncation_bound", gen_c2_grid,
                                             "largest change of the fitted C2 over the tabulated nodes when the "
                                             "grid is halved (the two-term fit residual at s* is "
                                             + _g(gen[1.0]["fit_residual"], 2) + " relative)"),
                tolerance={"abs": 1e-8, "rel": 1e-3}),
        finding("The predicted validity boundary holds: new integrations at eps_max/2 and 2 eps_max fall on either "
                "side of tau and match the C2/C3 remainder model",
                "numerical", {**probe, "relative_gap_to_linear_prediction": probe_gap,
                              "relative_gap_to_remainder_model": model_gap},
                {"generator": _gen("validity-probe", s_fraction=0.9, tau=TAU),
                 "derivation": _derivation("t017-validity-domains-of-the-first-order-approximation"),
                 "checks": [core.check("invariant", "relative error at eps_max / 2 minus tau",
                                       probe["half"]["relative_error"] - TAU, 0.0, "signed_le"),
                            core.check("invariant", "relative error at 2 eps_max minus tau",
                                       probe["double"]["relative_error"] - TAU, 0.0, "signed_ge"),
                            core.check("analytic", "max relative gap of the probes to |C2 eps + C3 eps^2| / |j|",
                                       max(abs(v) for v in model_gap.values()), 1e-3)]},
                uncertainty=core.uncertainty("truncation_bound", probe_grid,
                                             "largest change of the probe relative errors when the probes and the "
                                             "base path are integrated with half the steps (the two-term fit "
                                             "residual at 0.9 s* is " + _g(probe_row["fit_residual"], 2) + ")"),
                tolerance={"abs": 1e-8, "rel": 1e-3}),
        finding("Unit sphere, pure heading: C2 = 0, C3 = -|sin s| cos^2 s / 24, and the domain does not shrink at s = pi",
                "numerical", {"max_C2_over_j": sph_c2, "max_C3_relative_error": sph_c3,
                              "max_eps_max_relative_error": sph_eps, "eps_max_near_pi": near},
                {"generator": _gen("validity-sphere", eps=list(EPS), tau=TAU),
                 "derivation": _derivation("t017-validity-domains-of-the-first-order-approximation"),
                 "checks": [core.check("analytic", "max |C2| / |j|", sph_c2, 1e-5),
                            core.check("analytic", "max |C3 / C3_analytic - 1|", sph_c3, 1e-2),
                            core.check("analytic", "max |eps_max / (sqrt(24 tau)/|cos s|) - 1|", sph_eps, 2e-2),
                            core.check("invariant", "min eps_max at s = 0.999 pi and 1.001 pi", min(near.values()),
                                       0.4, "ge")]},
                uncertainty=core.uncertainty("fit_residual", max(r["fit_residual"] for r in sph_modelled),
                                             "largest relative residual of the C2/C3 model over the tabulated s "
                                             "where the closed-form C3 is nonzero (at s = pi/2 the remainder is "
                                             "roundoff)"),
                tolerance={"abs": 1e-7, "rel": 1e-3},
                counterexample={"statement": "The validity domain of the first-order approximation shrinks to zero at "
                                             "every conjugate point",
                                "witness": {"surface": "unit sphere", "perturbation": "pure heading", "tau": TAU,
                                            "eps_max_near_pi": near}}),
        finding("Hyperbolic plane, pure heading: C2 = 0 and eps_max = sqrt(24 tau)/cosh(s) shrinks exponentially",
                "numerical", {"max_C2_over_C3_eps_min": hyp_c2, "max_eps_max_relative_error": hyp_eps,
                              "eps_max": [r["eps_max"] for r in hyp]},
                {"generator": _gen("validity-hyperbolic", eps=list(EPS), tau=TAU),
                 "derivation": _derivation("t017-validity-domains-of-the-first-order-approximation"),
                 "checks": [core.check("analytic", "max |C2| / (|C3| eps_min): quadratic term negligible", hyp_c2, 0.05),
                            core.check("analytic", "max |eps_max / (sqrt(24 tau)/cosh s) - 1|", hyp_eps, 5e-2)]},
                uncertainty=core.uncertainty("model_truncation", hyp_c3,
                                             f"eps^5 terms absorbed by the fitted C3 (largest at s = {_g(hyp_c3_row['s'], 3)})"),
                tolerance={"abs": 1e-7, "rel": 1e-3}),
        finding("Sphere lateral+heading perturbation: C2(s0) = 1/2 at the first-order zero s0 = 3pi/4, and eps_max "
                "shrinks to zero linearly there",
                "numerical", {"C2_at_s0": mixed[1.0]["C2"], "eps_max_at_s0": mixed[1.0]["eps_max"],
                              "eps_max_at_0.9_s0": mixed[0.9]["eps_max"], "collapse_slope": mixed_slope,
                              "collapse_slope_predicted": mixed_slope_predicted},
                {"generator": _gen("validity-sphere-mixed", eps=list(EPS), tau=TAU),
                 "derivation": _derivation("t017-validity-domains-of-the-first-order-approximation"),
                 "checks": [core.check("analytic", "|C2(s0)| - 1/2", abs(mixed[1.0]["C2"]) - 0.5, 1e-2),
                            core.check("analytic", "eps_max / |s - s0| at 0.99 s0 and 1.01 s0 against "
                                       "2 sqrt(2 tau + tau^2) |j'(s0)| (along-track eps^2/2), relative",
                                       mixed_slope / mixed_slope_predicted - 1.0, 0.01)]},
                uncertainty=core.uncertainty("fit_residual", mixed[1.0]["fit_residual"],
                                             "relative residual of the C2/C3 model at s0"),
                tolerance={"abs": 1e-8, "rel": 1e-3}),
        finding("Torus outer equator: C2 vanishes along the whole path (reflection symmetry); the remainder is cubic",
                "numerical", {"max_abs_C2": eq_c2, "remainder_exponent_at_s_star": eq[1.0]["remainder_exponent"],
                              "eps_max_at_s_star": eq[1.0]["eps_max"]},
                {"generator": _gen("validity-torus-equator", eps=list(EPS)),
                 "checks": [core.check("invariant", "max |C2| over the tabulated nodes", eq_c2, 1e-4),
                            core.check("analytic", "remainder exponent at s* minus 3 (reflection symmetry)",
                                       (eq[1.0]["remainder_exponent"] or 0.0) - 3.0, 0.05)]},
                uncertainty=core.uncertainty("model_truncation", eq_c2, "the fitted C2 is not exactly zero because eps^5 terms leak "
                                             "into the two-term fit; halving the grid changes it by at most "
                                             f"{_g(eq_c2_grid, 2)} over the tabulated nodes"),
                tolerance={"abs": 1e-6, "rel": 1e-3}),
        finding("These validity domains certify first-order path corrections as safe on real machines",
                "machine_safety", None, {}),
    ]
    fields = _fields(
        hypothesis=("The first-order remainder r(eps, s) = d(s) - eps |j(s)| is C2(s) eps^2 + C3(s) eps^3 + ...; the "
                    "first-order prediction is within relative tolerance tau while |C2 eps + C3 eps^2| <= tau |j|, a "
                    "domain that shrinks to zero where j vanishes unless the remainder vanishes there too."),
        mathematical_model=("Unsigned separation d at matched arclength. Unit sphere, pure heading: d = 2 arcsin(|sin s| "
                            "sin(eps/2)), so C2 = 0, C3 = -|sin s| cos^2 s / 24 and eps_max = sqrt(24 tau)/|cos s| "
                            "(no collapse at s = pi). Hyperbolic plane: d = 2 asinh(sinh s sin(eps/2)), C2 = 0, "
                            "eps_max = sqrt(24 tau)/cosh s. Generic paths: a normal eps^2 term C2(s*) != 0, so "
                            "eps_max ~ tau |j'(s*)| |s - s*| / |C2(s*)| -> 0; symmetric paths: C2 = 0 and eps_max ~ "
                            "sqrt(|s - s*|). Sphere lateral+heading: the eps^2/2 offset at s0 = 3pi/4 is along-track, "
                            "orthogonal to eps j N, so d = sqrt((eps j)^2 + eps^4/4) near s0 and eps_max = 2 |j| "
                            "sqrt(2 tau + tau^2) ~ 2 sqrt(2 tau) |j'(s0)| |s - s0|."),
        input_data=[f"eps = {list(EPS)}; tau = {TAU}; bounds capped at eps = {EPS_CAP}",
                    "Closed forms: unit sphere (pure heading; lateral = heading), HyperbolicPlane(1) (pure heading)",
                    "Numerical: Torus(2, 1) generic and outer-equator families from T010 (RK4 chords at grid nodes)"],
        observation_model=("Great-circle or hyperbolic distance (closed form) or embedded chord (torus) versus "
                           "eps |j(s)|; C2 and C3 by least squares over eps; eps_max from the fitted quadratic."),
        expected_invariant=("C2 = 0 for isotropic/symmetric configurations; eps_max -> 0 at first-order zeros only "
                            "when the remainder does not vanish there."),
        experiment=("Tabulate C2, C3 and eps_max over s for five configurations (the torus ones also on a grid with "
                    "half the steps); compare the collapse of eps_max next to each first-order zero with its linear "
                    "law; probe the predicted boundary at 0.9 s* on the generic torus path with new integrations at "
                    "eps_max/2 and 2 eps_max and compare them with the fitted C2/C3 remainder model."),
        numerical_result=(f"Generic torus: C2(s*) = {_g(gen[1.0]['C2'], 4)}, eps_max(s*/2) = {_g(gen[0.5]['eps_max'], 3)}, "
                          f"eps_max / |s - s*| next to s* = {_g(gen_slope, 4)} against tau |j'(s*)| / |C2(s*)| = "
                          f"{_g(gen_slope_predicted, 4)}; boundary probe relative errors "
                          f"{_g(probe['half']['relative_error'], 3)} (eps_max/2) and {_g(probe['double']['relative_error'], 3)} "
                          f"(2 eps_max) against tau = {TAU}, within {_g(max(abs(v) for v in model_gap.values()), 2)} "
                          f"(relative) of the C2/C3 model and {_g(probe_gap['half'], 2)}, {_g(probe_gap['double'], 2)} of "
                          f"the linear values tau/2, 2 tau; sphere pure heading eps_max near pi = "
                          f"{_g(min(near.values()), 4)} (C3 relative error {_g(sph_c3, 2)}); hyperbolic eps_max within "
                          f"{_g(hyp_eps, 2)} (relative) of sqrt(24 tau)/cosh s; sphere "
                          f"lateral+heading C2(s0) = {_g(mixed[1.0]['C2'], 4)}, eps_max / |s - s0| next to s0 = "
                          f"{_g(mixed_slope, 4)} against {_g(mixed_slope_predicted, 4)}; equator max |C2| = {_g(eq_c2, 2)}; "
                          f"hyperbolic eps_max = {', '.join(_g(r['eps_max'], 4) for r in hyp)}."),
        uncertainty=(f"C2 and C3 absorb higher-order terms from eps up to {EPS[-1]} (on the hyperbolic plane the fitted "
                     f"C3 differs from its closed form by up to {_g(100 * hyp_c3, 2)} percent at s = "
                     f"{_g(hyp_c3_row['s'], 3)}, and the fitted C2 is not exactly zero); eps_max beyond the fitted eps "
                     "range is an extrapolation of the quadratic model; halving the torus grids changes the fitted C2 "
                     f"by at most {_g(gen_c2_grid, 2)} (generic) and {_g(eq_c2_grid, 2)} (equator) over the tabulated "
                     f"nodes and the probe relative errors by {_g(probe_grid, 2)}; the finer-grid values are "
                     "accurate to about a sixteenth of these changes. At a first-order zero eps_max itself is set "
                     "by the numerical |j| there (about 1e-10 on the torus, where the grid node is placed on the "
                     "located zero), so the collapse is checked through its linear law next to the zero, not by "
                     "the value at the zero."),
        failure_modes_checked=["fitted C2 compared with its analytic zero on isotropic surfaces",
                               "predicted boundary probed by new integrations on both sides and compared with the "
                               "full C2/C3 model at the fit-residual level, not only with the linear values",
                               "collapse at first-order zeros checked through the linear law next to the zero (eps_max "
                               "at the zero is zero by construction of the grid)",
                               "grid dependence of C2 and of the probes measured on a grid with half the steps",
                               "exact first-order zeros handled (eps_max = 0 when j = 0)",
                               "chord versus intrinsic distance difference is O(eps^3) and cannot change C2"],
        unresolved_assumptions=["The quadratic remainder model is extrapolated to eps_max up to the cap",
                                "Heading perturbations (plus one lateral case on the sphere) only",
                                "Machine-safety use of these domains is outside what the computation establishes"],
        recommended_next_task="T018 (curvature signal versus integrator error) and a sampled survey of C2(s*) over "
                              "random torus geodesics",
    )
    return {"state": "completed", "fields": fields, "findings": findings}


# ------------------------------------------------------------------ T018
T018_LENGTH = 2.0
T018_STEPS = (4, 8, 16, 32, 64, 128)
T018_METHODS = ("euler", "midpoint", "rk4")
RESOLVED = 10.0          # curvature signal at least ten times the integrator error
# An error is rounding-affected when its measured rounding part exceeds this share of its truncation part: a
# different rounding sequence could then move the observed ratio by more than about ten percent.
ROUNDING_SHARE = 0.1


def _t018_cases():
    """(key, surface, start, heading, constant curvature along the path or None).

    The flat surfaces use K recomputed from their second fundamental form, so a zero signal there tests the
    embedding rather than the closed-form K = 0 of the core classes.
    """
    sphere_start, sphere_heading = SPHERE_START
    return (("plane", core.SecondFormCurvature(Plane()), (0.3, -0.2), 0.7, 0.0),
            ("cylinder", core.SecondFormCurvature(Cylinder(1.0)), (0.2, 0.1), 0.6, 0.0),
            ("sphere R=1", Sphere(1.0), sphere_start, sphere_heading, 1.0),
            ("sphere R=10", Sphere(10.0), sphere_start, sphere_heading, 1e-2),
            ("sphere R=100", Sphere(100.0), sphere_start, sphere_heading, 1e-4),
            ("sphere R=1e4", Sphere(1e4), sphere_start, sphere_heading, 1e-8),
            ("sphere R=1e7", Sphere(1e7), sphere_start, sphere_heading, 1e-14),
            ("sphere R=1e8", Sphere(1e8), sphere_start, sphere_heading, 1e-16),
            ("torus R=2", Torus(2.0, 1.0), (0.0, 0.5), 0.7, None),
            ("torus R=64 equator", Torus(64.0, 1.0), (0.0, 0.0), 0.0, 1.0 / 65.0),
            ("saddle c=1", Saddle(1.0), (0.1, -0.2), 0.8, None),
            ("bump h=0.5", GaussianBump(0.5, 1.0), (-1.2, 0.3), 0.2, None),
            ("bump h=0.05", GaussianBump(0.05, 1.0), (-1.2, 0.3), 0.2, None),
            ("hyperbolic k=1", HyperbolicPlane(1.0), (0.0, 1.0), 0.6, -1.0))


def flat_deviation(curvature, length):
    """j_head(L) - L for constant K, evaluated without cancellation (series for small sqrt|K| L)."""
    if curvature == 0.0:
        return 0.0
    w = math.sqrt(abs(curvature))
    x = w * length
    if x < 1e-2:
        # x - sin x = x^3/6 - x^5/120 + x^7/5040 - ...; sinh x - x has the same terms with + signs.
        sign = -1.0 if curvature > 0 else 1.0
        return sign * (x ** 3 / 6 + sign * x ** 5 / 120 + x ** 7 / 5040) / w
    return (math.sin(x) if curvature > 0 else math.sinh(x)) / w - length


def rounding_affected(truncation, rounding):
    """Per step: True where the measured rounding part exceeds ROUNDING_SHARE of the truncation part."""
    return [bool(r > ROUNDING_SHARE * e) for e, r in zip(truncation, rounding)]


def resolved_from(ratios):
    """Smallest N from which every finer step keeps the ratio >= RESOLVED (a None ratio is an exact zero error)."""
    for i in range(len(T018_STEPS)):
        if all(r is None or r >= RESOLVED for r in ratios[i:]):
            return T018_STEPS[i]
    return None


def error_decomposition(surface, u0, heading, method, steps, curvature, true_deviation, computed):
    """(truncation part, rounding part) of |computed - true deviation| for one fixed-step run.

    Constant K: the method's step matrix evaluated in exact rational arithmetic gives the truncation-only
    deviation, and the rounding part is the computed deviation minus it, both exact. Variable K: no exact
    reference exists, so the truncation part is the observed error and the rounding part is estimated by
    rerunning with the heading Jacobi column scaled by 3 (identical in exact arithmetic, rounded differently).
    """
    if curvature is not None:
        exact = core.exact_arithmetic_transfer(method, curvature, T018_LENGTH, steps)[0][1] - Fraction(T018_LENGTH)
        return float(abs(exact - Fraction(true_deviation))), float(abs(Fraction(computed) - exact))
    y0 = jacobi.initial_state(surface, u0, heading)
    y0[7] = 3.0
    _, scaled = integrators.integrate_fixed(jacobi.rhs(surface), y0, T018_LENGTH, steps, method)
    rounding = abs((float(scaled[-1, 6]) / 3.0 - T018_LENGTH) - computed)
    return abs(computed - true_deviation), float(rounding)


def _scipy_reference(surface, u0, heading, length, reference_j_head):
    """Optional cross-check with scipy's DOP853 (artifact only; findings never depend on scipy).

    SciPy supplies only the time stepper: the right-hand side is ciw's own geodesic/Jacobi model, so agreement
    checks the integrator, not the geometry.
    """
    try:
        from scipy.integrate import solve_ivp
        import scipy
    except ImportError:
        return None
    f = jacobi.rhs(surface)
    solution = solve_ivp(lambda _s, y: f(y), (0.0, length), jacobi.initial_state(surface, u0, heading),
                         method="DOP853", rtol=1e-12, atol=1e-14)
    record = {"implementation": "scipy.integrate.solve_ivp(DOP853)", "revision": scipy.__version__,
              "right_hand_side": "ciw.lab.jacobi.rhs (shared model; independent stepper only)",
              "success": bool(solution.success), "status": int(solution.status)}
    if not solution.success:
        return {**record, "j_head": None, "j_head_minus_reference": None, "reason": str(solution.message)}
    j_head = float(solution.y[6, -1])
    return {**record, "j_head": j_head, "j_head_minus_reference": j_head - reference_j_head}


def resolvability_study():
    rows = []
    for key, surface, u0, heading, curvature in _t018_cases():
        if curvature is not None:
            true_deviation, spread, kind = flat_deviation(curvature, T018_LENGTH), 0.0, "analytic"
        else:
            fine = jacobi.transfer(surface, u0, heading, T018_LENGTH, rtol=1e-12, atol=1e-14)
            coarse = jacobi.transfer(surface, u0, heading, T018_LENGTH, rtol=1e-11, atol=1e-13)
            # The same DP45 at a tighter tolerance: a self-convergence reference.
            true_deviation, kind = float(fine.states[-1, 6]) - T018_LENGTH, "self_convergence"
            spread = abs(float(fine.states[-1, 6]) - float(coarse.states[-1, 6]))
        signal = abs(true_deviation)
        methods, path_curvature = {}, 0.0
        for method in T018_METHODS:
            errors, computed, truncation, rounding = [], [], [], []
            for steps in T018_STEPS:
                run = jacobi.transfer(surface, u0, heading, T018_LENGTH, steps=steps, method=method)
                # j_head(L) is within a factor 2 of L, so this subtraction is exact (Sterbenz).
                deviation = float(run.states[-1, 6]) - T018_LENGTH
                computed.append(deviation)
                errors.append(abs(deviation - true_deviation))
                parts = error_decomposition(surface, u0, heading, method, steps, curvature, true_deviation, deviation)
                truncation.append(parts[0])
                rounding.append(parts[1])
                if curvature == 0.0:
                    path_curvature = max(path_curvature, float(np.max(np.abs(run.curvature_along()))))
            ratios = [None if e == 0.0 else signal / e for e in errors]
            truncation_ratios = [None if e == 0.0 else signal / e for e in truncation]
            methods[method] = {"computed_deviation": computed, "errors": errors, "ratios": ratios,
                               "truncation_errors": truncation, "rounding_errors": rounding,
                               "truncation_ratios": truncation_ratios,
                               "rounding_affected": rounding_affected(truncation, rounding) if signal > 0
                               else [False] * len(errors),
                               "resolved_from_steps": resolved_from(ratios) if signal > 0 else None,
                               "truncation_resolved_from_steps": resolved_from(truncation_ratios) if signal > 0
                               else None}
        reference_j_head = T018_LENGTH + true_deviation
        rows.append({"surface": key, "curvature": curvature, "reference_kind": kind, "true_deviation": true_deviation,
                     "error_decomposition": "exact rational step matrix" if curvature is not None
                     else "observed error; rounding from a rerun with the heading column scaled by 3",
                     "reference_spread": spread, "curvature_signal": signal, "methods": methods,
                     "max_abs_path_curvature": path_curvature if curvature == 0.0 else None,
                     "scipy": _scipy_reference(surface, u0, heading, T018_LENGTH, reference_j_head)
                     if curvature is None else None,
                     "reference_j_head": reference_j_head})
    return rows


@task("T018", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t018_resolvability_report",
                                                     f"{TESTS}::test_optional_scipy_cross_check_agrees_with_adaptive_references"))
def curvature_versus_integrator_error(ctx):
    rows = ctx.memo("gjl-resolvability", resolvability_study)
    by_name = {r["surface"]: r for r in rows}
    ctx.artifact_json("resolvability.json", core.jsonable(rows, 12))
    shown_steps = (4, 16, 128)
    table = ["| surface | K | signal abs(j_head(L) - L) | "
             + " | ".join(f"{m}: ratio at N = {' / '.join(str(n) for n in shown_steps)}; resolved from N "
                          "(truncation alone)" for m in T018_METHODS) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in T018_METHODS) + " |"]

    def resolved_label(value, row):
        return str(value) if value else ("no signal" if row["curvature_signal"] == 0 else "never")

    for row in rows:
        cells = []
        for m in T018_METHODS:
            entry = row["methods"][m]
            shown = []
            for n in shown_steps:
                i = T018_STEPS.index(n)
                ratio = entry["ratios"][i]
                shown.append("-" if ratio is None else _g(ratio, 3) + (" (r)" if entry["rounding_affected"][i] else ""))
            cells.append(f"{' / '.join(shown)}; {resolved_label(entry['resolved_from_steps'], row)} "
                         f"({resolved_label(entry['truncation_resolved_from_steps'], row)})")
        curvature = "varies" if row["curvature"] is None else _g(row["curvature"], 3)
        table.append(f"| {row['surface']} | {curvature} | {_g(row['curvature_signal'], 4)} | " + " | ".join(cells) + " |")
    ctx.artifact_text("resolvability.md", "# Curvature signal versus integrator error (L = 2)\n\n"
                      f"Ratio = signal / abs(computed - true deviation); resolved when >= {RESOLVED:g} for every finer "
                      "step. Each error is split into a truncation part and a rounding part: for constant K the "
                      "truncation part is the method's step matrix evaluated in exact rational arithmetic and the "
                      "rounding part is the computed deviation minus that exact result; for variable K the truncation "
                      "part is the observed error and the rounding part is estimated by a rerun with the heading "
                      f"column scaled by 3. (r) marks a rounding part above {ROUNDING_SHARE:g} of the truncation part. "
                      "In parentheses: the step from which the truncation part alone would be resolved.\n\n"
                      + "\n".join(table) + "\n")
    curved = [r for r in rows if r["curvature_signal"] > 0]
    for method in ("rk4", "euler"):
        ctx.artifact_text(f"resolvability-{method}.svg", svg.line_plot(
            [(r["surface"], list(T018_STEPS), [math.nan if v is None else v for v in r["methods"][method]["ratios"]])
             for r in curved],
            title=f"{'RK4' if method == 'rk4' else 'Euler'} resolvability ratio (signal / error), L = 2",
            xlabel="steps N (h = 2/N)", ylabel="signal / error", logx=True, logy=True))
    at16 = T018_STEPS.index(16)

    def truncation(row, method, index):
        return row["curvature_signal"] > 0 and not row["methods"][method]["rounding_affected"][index]

    # RK4 on every (surface, N >= 16) whose error is truncation-dominated.
    rk4_entries = [(r["surface"], n, r["methods"]["rk4"]["ratios"][i]) for r in curved
                   for i, n in enumerate(T018_STEPS) if n >= 16 and truncation(r, "rk4", i)]
    rk4_min = min(entry[2] for entry in rk4_entries)
    hardest = min(rk4_entries, key=lambda entry: entry[2])
    rk4_surfaces = sorted({entry[0] for entry in rk4_entries}, key=[r["surface"] for r in rows].index)
    rk4_excluded = [r["surface"] for r in curved if r["surface"] not in rk4_surfaces]
    weakest = min((by_name[name] for name in rk4_surfaces), key=lambda r: r["curvature_signal"])
    # Weak-curvature comparison at N = 16 on truncation-dominated entries only.
    compare = {"euler": ("sphere R=1e4", "sphere R=10"), "midpoint": ("sphere R=1e4", "sphere R=10"),
               "rk4": ("sphere R=100", "sphere R=10")}
    used_rounding = sum(1 for m, names in compare.items() for name in names
                        if by_name[name]["methods"][m]["rounding_affected"][at16])
    weak_pair = {m: by_name["sphere R=1e4"]["methods"][m]["ratios"][at16] / by_name["sphere R=10"]["methods"][m]["ratios"][at16]
                 for m in ("euler", "midpoint")}
    strong_pair = {m: by_name["sphere R=10"]["methods"][m]["ratios"][at16] / by_name["sphere R=1"]["methods"][m]["ratios"][at16]
                   for m in ("euler", "midpoint")}
    rk4_gain = by_name["sphere R=100"]["methods"]["rk4"]["ratios"][at16] / by_name["sphere R=10"]["methods"]["rk4"]["ratios"][at16]
    witness_ratios = {m: {name: by_name[name]["methods"][m]["ratios"][at16]
                          for name in ("sphere R=1", "sphere R=10", "sphere R=100", "sphere R=1e4")
                          if truncation(by_name[name], m, at16)} for m in T018_METHODS}
    # K = 1e-16: the computed deviation is rounded away although the truncation part alone would be resolved.
    floor = by_name["sphere R=1e8"]
    floor_ratio = max(v for m in T018_METHODS for v in floor["methods"][m]["ratios"] if v is not None)
    floor_nonzero = sum(1 for m in T018_METHODS for v in floor["methods"][m]["computed_deviation"] if v != 0.0)
    floor_truncation = min(floor["methods"][m]["truncation_ratios"][-1] for m in T018_METHODS)
    # K = 1e-14 (signal about 30 ulp(L)): reported from the decomposition, not claimed.
    near_floor = by_name["sphere R=1e7"]
    near = {m: near_floor["methods"][m] for m in T018_METHODS}
    near_text = (f"K = 1e-14 (signal {_g(near_floor['curvature_signal'] / math.ulp(T018_LENGTH), 3)} ulp(L)): the "
                 f"Euler and midpoint errors at N = 4 are {_g(near['euler']['errors'][0] / near['euler']['truncation_errors'][0], 3)} "
                 f"and {_g(near['midpoint']['errors'][0] / near['midpoint']['truncation_errors'][0], 3)} times their "
                 "exact-arithmetic truncation errors, so they are truncation errors, while every RK4 error is rounding "
                 f"(exact-arithmetic truncation at most {_g(max(near['rk4']['truncation_errors']), 2)})")
    # Rounding-affected entries by method (counts only; which entries they are is in resolvability.md).
    affected = {m: sum(sum(r["methods"][m]["rounding_affected"]) for r in curved) for m in T018_METHODS}
    total_entries = len(curved) * len(T018_STEPS)
    changed = [(r, m) for r in curved for m in T018_METHODS
               if r["methods"][m]["resolved_from_steps"] != r["methods"][m]["truncation_resolved_from_steps"]]
    changed_text = (f"rounding changes the outcome for {len(changed)} (surface, method) pairs, all with |K| <= "
                    + _g(max(math.inf if r["curvature"] is None else abs(r["curvature"]) for r, _ in changed), 2)
                    if changed else "rounding changes no outcome")
    sphere = by_name["sphere R=1"]
    orders = {m: -core.loglog_slope(T018_STEPS[2:], sphere["methods"][m]["errors"][2:]) for m in T018_METHODS}
    flat = [r for r in rows if r["curvature_signal"] == 0.0]
    flat_error = max(max(r["methods"][m]["errors"]) for r in flat for m in T018_METHODS)
    flat_curvature = max(r["max_abs_path_curvature"] for r in flat)
    spread = max(r["reference_spread"] for r in rows)
    resolved_text = []
    for m in T018_METHODS:
        groups = {}
        for r in curved:
            groups.setdefault(r["methods"][m]["truncation_resolved_from_steps"], []).append(r["surface"])
        resolved_text.append(f"{m}: " + "; ".join(f"N = {n} ({', '.join(names)})" for n, names in sorted(
            groups.items(), key=lambda item: (item[0] is None, item[0] or 0))))
    scipy_rows = [r for r in rows if r["scipy"] is not None]
    scipy_text = ("not run (SciPy absent)" if not scipy_rows else
                  "max |DOP853 - DP45 reference| = "
                  + _g(max(abs(r["scipy"]["j_head_minus_reference"]) for r in scipy_rows
                           if r["scipy"]["j_head_minus_reference"] is not None), 2)
                  if all(r["scipy"]["success"] for r in scipy_rows) else "a SciPy run failed (see artifact)")
    findings = [
        finding("RK4 resolves every truncation-dominated curvature signal from N = 16 (h = 1/8)", "numerical",
                {"min_ratio_rk4_N_ge_16": rk4_min, "min_ratio_surface": hardest[0], "min_ratio_steps": hardest[1],
                 "weakest_signal_surface": weakest["surface"], "weakest_signal": weakest["curvature_signal"],
                 "rounding_affected_surfaces_excluded": rk4_excluded},
                {"generator": _gen("resolvability", steps=list(T018_STEPS), length=T018_LENGTH),
                 "derivation": _derivation("t018-curvature-signal-versus-integrator-error"),
                 "checks": [core.check("invariant", "min over truncation-dominated (surface, N >= 16) of signal / RK4 "
                                       "error", rk4_min, RESOLVED, "ge"),
                            core.check("self_convergence", "max reference spread (DP45 rtol 1e-12 vs 1e-11)",
                                       spread, 1e-9)]},
                uncertainty=core.uncertainty("reference_error", spread,
                                             "largest adaptive reference spread (rtol 1e-12 against 1e-11); "
                                             "closed-form references are exact"),
                tolerance={"abs": 1e-9, "rel": 0.05}),
        finding("Weak curvature is harder to resolve for Euler and midpoint only down to K ~ 1e-2: for K <= 1e-2 "
                "their ratios no longer depend on K, and the RK4 ratio grows like 1/K",
                "numerical", {"ratio_K_1e-8_over_K_1e-2_at_N16": weak_pair,
                              "ratio_K_1e-2_over_K_1_at_N16": strong_pair,
                              "rk4_ratio_K_1e-4_over_K_1e-2_at_N16": rk4_gain},
                {"generator": _gen("resolvability", surfaces=["sphere R=1", "sphere R=10", "sphere R=100",
                                                              "sphere R=1e4"]),
                 "derivation": _derivation("t018-curvature-signal-versus-integrator-error"),
                 "checks": [core.check("analytic", f"{m}: ratio(K = 1e-8) / ratio(K = 1e-2) minus 1", v - 1.0, 0.05)
                            for m, v in weak_pair.items()]
                 + [core.check("analytic", "RK4 ratio grows like 1/K: ratio(K = 1e-4)/ratio(K = 1e-2) in [50, 200] "
                               "(distance from 100)", abs(math.log10(rk4_gain) - 2.0), math.log10(2.0), "le"),
                    core.check("invariant", "Euler and midpoint: max ratio(K = 1e-2) / ratio(K = 1) (weaker is harder "
                               "above K = 1e-2)", max(strong_pair.values()), 1.0, "le"),
                    core.check("invariant", "rounding-affected entries among the compared ratios", used_rounding, 0.0)]},
                uncertainty=core.uncertainty("model_truncation", max(abs(v - 1.0) for v in weak_pair.values()),
                                             "O(K L^2) corrections to K-independence at K = 1e-2"),
                tolerance={"abs": 1e-6, "rel": 0.02},
                counterexample={"statement": "Weaker intrinsic curvature is harder to resolve at a fixed step size",
                                "witness": {"truncation_dominated_ratios_at_N16": witness_ratios}}),
        finding("Below the floating-point resolution of L no step size resolves the curvature signal, although the "
                "methods' truncation errors alone would resolve it", "numerical",
                {"max_ratio_K_1e-16": floor_ratio, "nonzero_computed_deviations_K_1e-16": floor_nonzero,
                 "min_truncation_only_ratio_K_1e-16_at_N128": floor_truncation},
                {"generator": _gen("resolvability", surface="sphere R=1e8"),
                 "derivation": _derivation("t018-curvature-signal-versus-integrator-error"),
                 "checks": [core.check("invariant", "max ratio over methods and N for K = 1e-16", floor_ratio, 1.0, "le"),
                            core.check("exact_arithmetic", "nonzero computed deviations for K = 1e-16", floor_nonzero,
                                       0.0),
                            core.check("exact_arithmetic", "min over methods of signal / exact-arithmetic truncation "
                                       "error at N = 128 for K = 1e-16", floor_truncation, RESOLVED, "ge")]},
                uncertainty=core.uncertainty("roundoff", math.ulp(T018_LENGTH),
                                             "one unit in the last place of L = 2"),
                tolerance={"abs": 1e-12, "rel": 1e-6},
                counterexample={"statement": "Refining the step size always makes a nonzero curvature effect resolvable",
                                "witness": {"surface": "Sphere(1e8), K = 1e-16", "signal": floor["curvature_signal"],
                                            "computed_deviation": 0.0, "steps": list(T018_STEPS)}}),
        finding("Resolvability ratios improve like h^-p with p = 1, 2, 4 (sphere R = 1)", "numerical", orders,
                {"generator": _gen("resolvability", surface="sphere R=1"),
                 "checks": [core.check("analytic", f"{m} order minus {integrators.ORDERS[m]}",
                                       orders[m] - integrators.ORDERS[m], 0.3) for m in T018_METHODS]},
                uncertainty=core.uncertainty("fit_spread",
                                             max(core.slope_spread(T018_STEPS[2:], sphere["methods"][m]["errors"][2:]) for m in T018_METHODS),
                                             "largest gap between a fitted order and consecutive-step-pair orders"),
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Flat surfaces (K from the second fundamental form) show no spurious curvature signal for any method "
                "or step", "numerical", {"max_abs_deviation": flat_error, "max_abs_path_curvature": flat_curvature},
                {"generator": _gen("resolvability", surfaces=[r["surface"] for r in flat]),
                 "checks": [core.check("analytic", "max |K| from the second fundamental form along the plane and "
                                       "cylinder paths", flat_curvature, 1e-14),
                            core.check("analytic", "max |j_head(L) - L| on plane and cylinder", flat_error, 1e-13)]},
                uncertainty=core.uncertainty("roundoff", flat_error,
                                             "with K = 0 from the embedding, j'' = 0 is integrated exactly by every "
                                             "method"),
                tolerance={"abs": 1e-13, "rel": 0.0}),
        finding("Curvature signals resolvable here would be resolvable in measured sensor data", "sensor_performance",
                None, {}),
    ]
    fields = _fields(
        hypothesis=("A curvature effect in a numerical Jacobi field is meaningful only where it exceeds the integrator "
                    "error; the working expectation was that weak curvature (small |K| L^3/6) is the hard case for "
                    "coarse steps and low-order methods."),
        mathematical_model=("Signal S = |j_head(L) - L| (zero on flat surfaces; |K| L^3/6 for small constant K). "
                            "Truncation error E(h) ~ C h^p. For j'' + K j = 0 the Euler and midpoint errors are "
                            "themselves proportional to K (the K-free part j = s is integrated exactly), so S/E is "
                            "independent of K as K -> 0; the RK4 error enters at O(K^2 h^4), so S/E grows like 1/K. "
                            "For constant K the Jacobi part of every explicit Runge-Kutta stage is linear, so a run's "
                            "deviation is the method's step matrix to the power N plus rounding; evaluating that matrix "
                            "in exact rational arithmetic separates the truncation part from the rounding part exactly. "
                            "The limit is floating point: when S approaches ulp(L) = 4.4e-16 the computed deviation is "
                            f"rounded away. Resolved when S/E >= {RESOLVED:g} for every finer step."),
        input_data=[f"{len(rows)} declared paths of length {T018_LENGTH}: " + ", ".join(r["surface"] for r in rows),
                    "Plane and cylinder with K recomputed from their second fundamental form",
                    f"Euler, midpoint, RK4 with N = {', '.join(str(n) for n in T018_STEPS)}",
                    "References: cancellation-free closed forms for constant curvature, DP45 at rtol 1e-12 (checked "
                    "at 1e-11) otherwise",
                    "Error decomposition: exact rational step matrices for constant K; for variable K the observed "
                    "error with a rounding estimate from a rerun with the heading column scaled by 3; an entry is "
                    f"rounding-affected when its rounding part exceeds {ROUNDING_SHARE:g} of its truncation part "
                    "(declared)"],
        observation_model=("Computed deviation j_head,h(L) - L (an exact floating-point subtraction) against the true "
                           "deviation, split into the method's truncation part and the run's rounding part; no "
                           "renormalization."),
        expected_invariant=("Flat surfaces: S = E = 0; truncation-dominated curved surfaces: ratio grows like h^-p; "
                            "rounding parts of a few ulp(L)."),
        experiment=("Integrate each path with each method and step count; form ratios; split each error into its "
                    "truncation and rounding parts; find the smallest N from which every finer step keeps the ratio "
                    "above the threshold, with and without rounding; compare curvature scales 1 ... 1e-16."),
        numerical_result=(f"Min RK4 ratio over truncation-dominated entries with N >= 16: {_g(rk4_min, 4)} "
                          f"({hardest[0]}, N = {hardest[1]}); the weakest signal with truncation-dominated RK4 errors "
                          f"({weakest['surface']}, S = {_g(weakest['curvature_signal'], 3)}) is resolved from N = "
                          f"{weakest['methods']['rk4']['resolved_from_steps']}; RK4 errors are rounding-affected at "
                          f"every N >= 16 for {', '.join(rk4_excluded)}. At N = 16, ratio(K = 1e-2)/ratio(K = 1): Euler "
                          f"{_g(strong_pair['euler'], 3)}, midpoint {_g(strong_pair['midpoint'], 3)} (weaker is harder "
                          f"there); ratio(K = 1e-8)/ratio(K = 1e-2): Euler {_g(weak_pair['euler'], 4)}, midpoint "
                          f"{_g(weak_pair['midpoint'], 4)}; RK4 ratio gain from K = 1e-2 to 1e-4: {_g(rk4_gain, 3)}; "
                          f"K = 1e-16: computed deviation 0 for all runs (max ratio {_g(floor_ratio, 3)}) while the "
                          f"exact-arithmetic truncation errors alone give ratios of at least {_g(floor_truncation, 3)} at "
                          f"N = 128; {near_text}; orders {', '.join(f'{m} {_g(v, 3)}' for m, v in orders.items())}; flat "
                          f"max |j - L| {_g(flat_error, 2)} with second-form max |K| {_g(flat_curvature, 2)}. Resolved "
                          f"from N by the truncation part alone: {' | '.join(resolved_text)}; with rounding included, "
                          f"{changed_text}. Optional SciPy DOP853 cross-check of the adaptive references: {scipy_text} "
                          "(artifact only; it shares ciw's right-hand side, so it checks the integrator, not the "
                          "geometry)."),
        uncertainty=(f"Reference spread up to {_g(spread, 2)}. Rounding parts are exact for constant K and estimated "
                     "by a rescaled rerun for variable K; a rounding part above "
                     f"{ROUNDING_SHARE:g} of the truncation part (marked (r) in resolvability.md) occurs in "
                     + ", ".join(f"{affected[m]} {m}" for m in T018_METHODS)
                     + f" of {total_entries} entries per method. Those ratios depend on the rounding sequence; they are "
                     "kept in the artifacts and out of every checked ratio and witness except the K = 1e-16 floor "
                     "finding, whose claim is about roundoff and whose computed deviations are exactly zero. "
                     f"The threshold {RESOLVED:g} and the rounding share {ROUNDING_SHARE:g} are declared conventions."),
        failure_modes_checked=["exact zero errors on flat surfaces kept as 'no signal', not as infinite ratios",
                               "flat surfaces tested with K from the second fundamental form, not the closed-form "
                               "K = 0 of the core classes",
                               "resolution required to persist for every finer step, not just one lucky step",
                               "rounding separated from truncation by the exact-arithmetic step matrix (or a rescaled "
                               "rerun), not by a fixed ulp threshold on the observed error, which would class "
                               "truncation errors of signals near ulp(L) as rounding",
                               "true deviation computed without cancellation (series for small sqrt|K| L)",
                               "adaptive references checked by tightening rtol",
                               "the a priori expectation (weak curvature is harder) tested: it holds for Euler and "
                               "midpoint between K = 1 and 1e-2 and fails below K = 1e-2 and for RK4"],
        unresolved_assumptions=["Only j_head(L) is compared; the lateral column and conjugate-point locations are not",
                                "Variable-curvature paths add geodesic-position error to K(gamma(s)); its K-scaling is "
                                "not separated here, and their rounding part is an estimate, not an exact split",
                                "The SciPy cross-check shares ciw's right-hand side and is independent in the "
                                "integrator only",
                                "Measurement noise of any real observation is absent; sensor-level resolvability is "
                                "not established"],
        recommended_next_task="T005 (Jacobi separation law) with resolvability-aware step selection, and T046",
    )
    return {"state": "completed", "fields": fields, "findings": findings}
