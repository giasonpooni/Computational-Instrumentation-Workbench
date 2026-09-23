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


def _divergence(family, eps, offsets=(1, 2, 4, 8)):
    """Relative first-order error at nodes s* - m h: |signed - eps j| / |eps j|."""
    s, j = family["s"], family["j"]
    rows = []
    for m in offsets:
        i = TORUS_M - m
        predicted = eps * j[i]
        rows.append({"offset_steps": m, "distance": float(family["s_star"] - s[i]),
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
    torus = Torus(2.0, 1.0)
    analytic_s_star = math.pi * math.sqrt(torus.minor * (torus.major + torus.minor))

    # Divergence of the relative first-order error approaching s* (generic path, eps = 0.01).
    divergence = _divergence(generic, 0.01)
    divergence_slope = core.loglog_slope([r["distance"] for r in divergence], [r["relative_error"] for r in divergence])
    divergence_eq = _divergence(equator, 0.04)
    divergence_eq_slope = core.loglog_slope([r["distance"] for r in divergence_eq],
                                            [r["relative_error"] for r in divergence_eq])
    witness_rel = _divergence(generic, 0.04, offsets=(1,))[0]

    # Post-focus inversion: signed separation follows eps j through its sign change.
    before, after = TORUS_M // 2, int(1.25 * TORUS_M)
    inversion = {}
    for key, fam in (("equator", equator), ("generic", generic)):
        signed = fam["runs"][0.01]["signed"]
        inversion[key] = {"s_before": float(fam["s"][before]), "s_after": float(fam["s"][after]),
                          "j_before": float(fam["j"][before]), "j_after": float(fam["j"][after]),
                          "signed_before": float(signed[before]), "signed_after": float(signed[after]),
                          "ratio_after": float(signed[after] / (0.01 * fam["j"][after]))}
    monotone_ratio = float(equator["runs"][0.02]["chord"][TORUS_M] / equator["runs"][0.02]["chord"][TORUS_M // 2])

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
             "divergence_equator_eps_0.04": divergence_eq, "inversion": inversion,
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
                              "chord_at_s_star": eq["chord_at_s_star"]},
                {"generator": _gen("torus-equator-family", eps=list(EPS), steps=TORUS_STEPS),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "numerical conjugate point minus pi sqrt(r (R + r))",
                                       eq["located_conjugate_point"] - analytic_s_star, 1e-7),
                            core.check("analytic", "first-order prediction |eps j(s*)| (vanishes at s*)",
                                       eq["first_order_at_s_star"], 1e-9),
                            core.check("self_convergence", "fitted chord exponent minus 3 (reflection symmetry)",
                                       eq["chord_exponent"] - 3.0, 0.05)]},
                tolerance={"abs": 1e-3, "rel": 1e-5},
                counterexample={"statement": "The separation at a conjugate point is of exact order eps^2",
                                "witness": {"surface": "Torus(2, 1)", "path": "outer equator, heading 0",
                                            "s_star": eq["s_star"], "eps": list(EPS),
                                            "chord_exponent": eq["chord_exponent"]}}),
        finding("On a generic torus geodesic the separation at the conjugate point is O(eps^2) while eps j vanishes",
                "numerical", {"s_star": gen["s_star"], "chord_exponent": gen["chord_exponent"],
                              "chord_at_s_star": gen["chord_at_s_star"]},
                {"generator": _gen("torus-generic-family", eps=list(EPS), steps=TORUS_STEPS),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "first-order prediction |eps j(s*)| (vanishes at s*)",
                                       gen["first_order_at_s_star"], 1e-8),
                            core.check("self_convergence", "fitted chord exponent minus 2",
                                       gen["chord_exponent"] - 2.0, 0.1)]},
                tolerance={"abs": 1e-6, "rel": 1e-5}),
        finding("The relative first-order error diverges like 1/|s - s*| approaching the conjugate point",
                "numerical", {"slope_generic_eps_0.01": divergence_slope, "slope_equator_eps_0.04": divergence_eq_slope,
                              "relative_error_at_s_star_minus_h_eps_0.04": witness_rel["relative_error"]},
                {"generator": _gen("torus-divergence", offsets=[1, 2, 4, 8]),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("self_convergence", "generic log-log slope plus 1", divergence_slope + 1.0, 0.15),
                            core.check("self_convergence", "equator log-log slope plus 1", divergence_eq_slope + 1.0, 0.2),
                            core.check("invariant", "relative first-order error one step before s* (eps = 0.04)",
                                       witness_rel["relative_error"], 1.0, "ge")]},
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
                            core.check("invariant", "sign product of separations before and after s* (generic)",
                                       math.copysign(1.0, inversion["generic"]["signed_before"])
                                       * math.copysign(1.0, inversion["generic"]["signed_after"]), 0.0, "le"),
                            core.check("invariant", "sign product of separations before and after s* (equator)",
                                       math.copysign(1.0, inversion["equator"]["signed_before"])
                                       * math.copysign(1.0, inversion["equator"]["signed_after"]), 0.0, "le")]},
                tolerance={"abs": 1e-9, "rel": 1e-5}),
        finding("Separation does not grow monotonically with length: it nearly vanishes at the conjugate point",
                "numerical", monotone_ratio,
                {"generator": _gen("torus-equator-family", eps=0.02),
                 "checks": [core.check("invariant", "chord(s*) / chord(s*/2) on the outer equator",
                                       monotone_ratio, 1e-3, "le")]},
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
                tolerance={"abs": 1e-9, "rel": 1e-4},
                counterexample={"statement": "The relative first-order error diverges at every conjugate point",
                                "witness": {"surface": "unit sphere", "perturbation": "pure heading",
                                            "eps": 0.02, "s": float(s[255]), "relative_error": near_pi}}),
        finding("On the unit sphere the image inverts after the conjugate point (signed ratio -1)",
                "numerical", sphere_flip,
                {"generator": _gen("sphere-great-circles", eps=0.02),
                 "checks": [core.check("analytic", "signed separation at 5pi/4 over 3pi/4 plus 1", sphere_flip + 1.0, 1e-8)]},
                tolerance={"abs": 1e-8, "rel": 0.0}),
        finding("A combined lateral+heading perturbation on the sphere has true separation eps^2/2 where eps j vanishes",
                "numerical", {"exponent": mixed_exponent, "coefficient": mixed_coefficient, "s0": s0},
                {"generator": _gen("sphere-lateral-heading", lateral_ratio=1.0, eps=list(EPS)),
                 "derivation": _derivation("t010-near-focus-and-post-focus-counterexamples"),
                 "checks": [core.check("analytic", "closed-form exponent minus 2", mixed_exponent - 2.0, 0.01),
                            core.check("analytic", "chord / eps^2 at s0 = 3pi/4 minus 1/2", mixed_coefficient - 0.5, 1e-3),
                            core.check("analytic", "numerical chord at s0 minus closed form (eps = 0.02)",
                                       mixed_numeric - float(closed_mixed["chord"][192]), 1e-9)]},
                tolerance={"abs": 1e-6, "rel": 1e-6}),
    ]
    fields = _fields(
        hypothesis=("Where the first-order separation eps j(s) vanishes (conjugate or focal points) the true "
                    "separation is of higher order in eps, so the relative first-order error diverges there; past "
                    "the zero the separation changes sign (image inversion); symmetric configurations raise the "
                    "order of the true separation."),
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
                            "~ 1/|s - s*|; sign(d) = sign(j) away from s*; sphere relative chord error "
                            "2 sin(eps/2)/eps - 1 at every s."),
        experiment=("Integrate base transfer and perturbed geodesics on common grids (RK4), locate s* from j_head, "
                    "fit exponents of d(s*) against eps, fit the divergence slope over s* - m h (m = 1, 2, 4, 8), "
                    "check sign inversion after s*, and compare sphere runs with closed-form great circles."),
        numerical_result=(f"Equator: s* = {_g(eq['s_star'])}, chord(s*) exponent {_g(eq['chord_exponent'], 4)}; "
                          f"generic: s* = {_g(gen['s_star'])}, exponent {_g(gen['chord_exponent'], 4)}; divergence "
                          f"slope {_g(divergence_slope, 4)} (generic), {_g(divergence_eq_slope, 4)} (equator); "
                          f"relative error {_g(witness_rel['relative_error'], 4)} one step before s* at eps = 0.04; "
                          f"after s* signed/(eps j) = {_g(inversion['generic']['ratio_after'], 5)} with j < 0; "
                          f"chord(s*)/chord(s*/2) = {_g(monotone_ratio, 3)}; sphere relative error uniform "
                          f"{_g(uniform, 4)} (max deviation {_g(sphere_uniform_dev, 2)}); sphere combined perturbation "
                          f"chord(3pi/4) = {_g(mixed_coefficient, 5)} eps^2."),
        uncertainty=("RK4 grid errors in the separations are below 1e-11 (the ratio of separations at two grid "
                     "resolutions agrees to that level); exponents are least-squares fits over eps in "
                     "[0.005, 0.04] and include higher-order terms (few 1e-2 for the generic path)."),
        failure_modes_checked=["numerical conjugate point versus pi sqrt(3) on the equator",
                               "first-order prediction at s* is at integrator noise level, not zero by construction",
                               "sphere numerical separations versus exact great circles at every node",
                               "exact-refocusing (sphere) and symmetric (equator) cases that do not show eps^2",
                               "sign inversion checked with the first-order field itself, not assumed"],
        unresolved_assumptions=["Chord separation is extrinsic; intrinsic distance differs at O(d^3) and would "
                                "change only third-order coefficients",
                                "The generic torus path is one geodesic; genericity is argued, not sampled",
                                "Heading perturbations only for the torus; lateral families are not fitted there"],
        recommended_next_task="T017 (validity domains, which tabulates C(s) for these families) and T008 "
                              "(conjugate-point location accuracy)",
    )
    return {"state": "completed", "fields": fields, "findings": findings}
