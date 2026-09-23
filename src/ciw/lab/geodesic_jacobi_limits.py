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
    return ref, float(np.max(np.abs(ref - other))), "high_precision"


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
                              # Order from the last step pair; roundoff-level errors carry no order.
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
    series = []
    for key in ("sphere", "torus"):
        for name in ("base", "polynomial-warp", "near-fold mu=0.1"):
            series.append((f"{key}: {name}", [T011_CASES[0][3] / n for n in T011_STEPS],
                           study[key]["charts"][name]["fixed_errors"]))
    ctx.artifact_text("chart-error.svg", svg.line_plot(series, title="RK4 endpoint/Jacobi error by chart",
                                                       xlabel="step h (plane/sphere units)", ylabel="max error",
                                                       logx=True, logy=True))
    adaptive_max = max(row["adaptive_error"] for case in study.values() for row in case["charts"].values())
    adaptive_by_kind = {kind: max(row["adaptive_error"] for case in study.values() if case["reference_kind"] == kind
                                  for row in case["charts"].values()) for kind in ("analytic", "high_precision")}
    length_max = max(row["adaptive_length_error"] for case in study.values() for row in case["charts"].values())
    orders = {f"{key}: {name}": row["order_last_pair"] for key, case in study.items()
              for name, row in case["charts"].items() if row["order_last_pair"] is not None}
    min_order = min(orders.values())
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
                            core.check("high_precision", "adaptive error against the rtol 1e-13 torus reference",
                                       adaptive_by_kind["high_precision"], 1e-8),
                            core.check("invariant", "|integral of speed - L| in every chart", length_max, 1e-8)]},
                tolerance={"abs": 1e-9, "rel": 0.5}),
        finding("RK4 keeps fourth-order convergence in every chart, including the near-fold charts",
                "numerical", {"min_order": min_order, "orders": orders},
                {"generator": _gen("chart-study", steps=list(T011_STEPS)),
                 "checks": [core.check("self_convergence", "smallest observed order between N = 128 and 256",
                                       min_order, 3.7, "ge")]},
                tolerance={"abs": 0.05, "rel": 0.0}),
        finding("A geometry-preserving near-fold chart multiplies the fixed-step error by a large factor",
                "numerical", {"error_factor_mu_0.1_at_N_256": {k: factors[k]["near-fold mu=0.1"] for k in factors},
                              "error_factor_mu_0.2_at_N_256": {k: factors[k]["near-fold mu=0.2"] for k in factors},
                              "smooth_chart_factor_range": [min(smooth), max(smooth)]},
                {"generator": _gen("chart-study", steps=T011_STEPS[-1]),
                 "checks": [core.check("self_convergence", "min over sphere and torus of error(fold 0.1)/error(base)",
                                       fold_factor, 100.0, "ge")]},
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
        expected_invariant=("All charts agree to the reference within the integrator tolerance; fixed-step order 4 in "
                            "every chart; the error constant is chart dependent."),
        experiment=("Integrate the joint geodesic/Jacobi system in each chart with identical geometric initial data; "
                    "compare with exact (plane, sphere) or rtol 1e-13 (torus) references; fit orders; form error "
                    "ratios chart/base at N = 256."),
        numerical_result=(f"Adaptive max error {_g(adaptive_max, 3)} over all charts; min RK4 order {_g(min_order, 4)}; "
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
T012_STEPS = 64


def _state_with_tangent(surface, u0, tangent):
    return np.concatenate([np.asarray(u0, dtype=float), tangent, [1.0, 0.0, 0.0, 1.0]])


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
        reference = jacobi.transfer(surface, u0, heading, length, steps=T012_STEPS)
        e1, e2 = surface.orthonormal_frame(np.asarray(u0, dtype=float))
        for beta in T012_BASIS_ANGLES:
            f1 = math.cos(beta) * e1 + math.sin(beta) * e2
            f2 = -math.sin(beta) * e1 + math.cos(beta) * e2
            local = heading - beta
            tangent = math.cos(local) * f1 + math.sin(local) * f2
            _, states = integrators.integrate_fixed(jacobi.rhs(surface), _state_with_tangent(surface, u0, tangent),
                                                    length, T012_STEPS, "rk4")
            basis_rows.append({"surface": key, "beta": beta,
                               "tangent_difference": float(np.max(np.abs(tangent - reference.states[0, 2:4]))),
                               "max_state_difference": float(np.max(np.abs(states - reference.states)))})
    # Orientation reversal: (e1, -e2) is also orthonormal but left-handed; "+eps" turns the other way.
    sphere = Sphere(1.0)
    u0, heading = np.asarray(SPHERE_START[0], dtype=float), SPHERE_START[1]
    e1, e2 = sphere.orthonormal_frame(u0)
    eps, length = 1e-3, 2.0
    base = jacobi.transfer(sphere, u0, heading, length, steps=T012_STEPS)
    normal_end = core.embedded_normal(sphere, base.states[-1])
    end = sphere.embedding(base.states[-1, :2])
    signed = {}
    for name, (f1, f2, local) in {"right-handed": (e1, e2, heading), "left-handed": (e1, -e2, -heading)}.items():
        tangent = math.cos(local + eps) * f1 + math.sin(local + eps) * f2
        _, states = integrators.integrate_fixed(sphere.geodesic_rhs, np.concatenate([u0, tangent]), length,
                                                T012_STEPS, "rk4")
        signed[name] = float((sphere.embedding(states[-1, :2]) - end) @ normal_end)
    refusal = None
    try:
        Rotated(sphere, np.diag([1.0, 1.0, -1.0]))
    except SurfaceRefusal as exc:
        refusal = str(exc)
    return {"rotations": rotation_rows, "basis_rotations": basis_rows,
            "orientation": {"eps": eps, "signed_separation": signed, "j_head_L": float(base.states[-1, 6]),
                            "ratio": signed["left-handed"] / signed["right-handed"]},
            "improper_rotation_refusal": refusal}


@task("T012", changed_files=CHANGED, regression_tests=(f"{TESTS}::test_t012_frame_invariance_and_refusal",))
def frame_change_invariance(ctx):
    study = ctx.memo("gjl-frame-study", frame_study)
    ctx.artifact_json("frame-invariance.json", core.jsonable(study, 12))
    rot_state = max(r["max_state_difference"] for r in study["rotations"])
    rot_end = max(r["endpoint_rotation_error"] for r in study["rotations"])
    rot_k = max(r["max_curvature_difference"] for r in study["rotations"])
    basis_state = max(r["max_state_difference"] for r in study["basis_rotations"])
    basis_tangent = max(r["tangent_difference"] for r in study["basis_rotations"])
    orient = study["orientation"]
    expected = "Frame change requires a proper rotation matrix"
    findings = [
        finding("Ambient rotations leave chart trajectories and Jacobi fields unchanged to roundoff", "numerical",
                rot_state, {"generator": _gen("frame-study", rotations=len(T012_ROTATIONS), steps=T012_STEPS),
                            "derivation": _derivation("t012-frame-change-invariance"),
                            "checks": [core.check("invariant", "max |rotated state - base state| over all nodes",
                                                  rot_state, 1e-11)]},
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("Embedded endpoints rotate exactly with the ambient frame", "numerical", rot_end,
                {"generator": _gen("frame-study"),
                 "checks": [core.check("invariant", "max |X_rotated(L) - R X(L)|", rot_end, 1e-12)]},
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Gaussian curvature from the rotated second fundamental form is unchanged", "numerical", rot_k,
                {"generator": _gen("frame-study"),
                 "checks": [core.check("invariant", "max |K_rotated - K| along every path", rot_k, 1e-12)]},
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Rotating the reference tangent basis (heading measured from e1') leaves every result unchanged",
                "numerical", {"max_state_difference": basis_state, "max_tangent_difference": basis_tangent},
                {"generator": _gen("frame-study", basis_angles=list(T012_BASIS_ANGLES)),
                 "checks": [core.check("invariant", "max |state(beta) - state(0)|", basis_state, 1e-11)]},
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("An orientation-reversing tangent basis flips the sign of the heading perturbation's separation",
                "numerical", orient["ratio"],
                {"generator": _gen("frame-study", eps=orient["eps"]),
                 "checks": [core.check("invariant", "left-handed / right-handed signed separation plus 1",
                                       orient["ratio"] + 1.0, 1e-2)]},
                tolerance={"abs": 1e-6, "rel": 0.0},
                counterexample={"statement": "Signed Jacobi separations are invariant under every change of "
                                             "orthonormal tangent basis",
                                "witness": {"surface": "unit sphere", "basis": "(e1, -e2)", "eps": orient["eps"],
                                            "signed_right_handed": orient["signed_separation"]["right-handed"],
                                            "signed_left_handed": orient["signed_separation"]["left-handed"]}}),
        finding("An improper rotation (reflection) is refused as a frame change", "computational_pipeline",
                study["improper_rotation_refusal"],
                {"checks": [core.refusal_check("Rotated(sphere, diag(1, 1, -1))", expected,
                                               study["improper_rotation_refusal"])]}),
    ]
    fields = _fields(
        hypothesis=("Geodesics and Jacobi fields are intrinsic: an ambient rotation changes only the embedded "
                    "coordinates (which rotate exactly), and the choice of reference basis for headings is a "
                    "relabeling; only the orientation of the basis enters, through the sign of the normal."),
        mathematical_model=("Rotated(base, R): X' = R X, so X'_i . X'_j = X_i . X_j and the second fundamental form is "
                            "unchanged; in exact arithmetic the chart ODE is identical. A basis (e1', e2') rotated by "
                            "beta with heading h - beta yields the same unit tangent; a left-handed basis (e1, -e2) "
                            "maps the heading change +eps to -eps in the right-handed convention, so J = j N flips sign."),
        input_data=["Sphere, Torus(2, 1), Saddle(1), GaussianBump(0.5, 1) on declared paths; HyperbolicPlane(1) for "
                    "basis changes", f"Rotations (axis, angle) = {[(list(a), g) for a, g in T012_ROTATIONS]}",
                    f"Basis angles beta = {list(T012_BASIS_ANGLES)}; RK4 with N = {T012_STEPS}"],
        observation_model=("Chart states (u, v, Jacobi columns) at every node, embedded endpoints, curvature along "
                           "the path, and the signed embedded separation at L for eps = 1e-3."),
        expected_invariant="Differences at roundoff level; exact rotation of X(L); separation ratio -1 under reflection.",
        experiment=("Integrate each path in the base and rotated surfaces and with rotated reference bases; compare "
                    "node by node; integrate +eps heading perturbations in right- and left-handed bases."),
        numerical_result=(f"Rotation: max state difference {_g(rot_state, 2)}, endpoint rotation error {_g(rot_end, 2)}, "
                          f"curvature difference {_g(rot_k, 2)}; basis rotation: max state difference "
                          f"{_g(basis_state, 2)}; orientation reversal ratio {_g(orient['ratio'], 8)}; reflection "
                          f"refused: {study['improper_rotation_refusal']!r}."),
        uncertainty=("Roundoff differences depend on platform arithmetic (few 1e-15); the regression tolerances "
                     "allow 1e-11. The reversal ratio differs from -1 by O(eps) second-order terms."),
        failure_modes_checked=["bitwise equality is not assumed (rotation changes rounding of dot products)",
                               "improper rotation refused by the core",
                               "curvature recomputed from the rotated embedding, not reused",
                               "orientation dependence of signed quantities made explicit"],
        unresolved_assumptions=["Rotations only; translations and reflections of the embedding are not exercised",
                                "Hyperbolic-plane isometries (Mobius maps) are not tested, only basis changes"],
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
    j_minus_s = float(np.max(np.abs(run_cyl.states[:, 6] - run_cyl.s)))
    chord_plane = float(np.linalg.norm(plane.embedding(run_plane.states[-1, :2]) - plane.embedding(np.array(u0))))
    chord_cyl = float(np.linalg.norm(cylinder.embedding(run_cyl.states[-1, :2]) - cylinder.embedding(np.array(u0))))
    radius = cylinder.radius
    helix = math.hypot(2 * radius * math.sin(length * math.cos(heading) / (2 * radius)), length * math.sin(heading))
    tori = []
    for major in T013_RADII:
        torus = Torus(major, 1.0)
        run = jacobi.transfer(torus, (0.0, 0.0), 0.0, T013_LENGTH, steps=T013_STEPS)
        curvature = 1.0 / (torus.minor * (major + torus.minor))
        w = math.sqrt(curvature)
        exact_j = math.sin(w * T013_LENGTH) / w
        deviation = T013_LENGTH - float(run.states[-1, 6])
        chord = float(np.linalg.norm(torus.embedding(run.states[-1, :2]) - torus.embedding(np.zeros(2))))
        rho = major + torus.minor
        tori.append({"major": major, "curvature": curvature, "deviation": deviation,
                     "exact_deviation": T013_LENGTH - exact_j, "leading_order": curvature * T013_LENGTH ** 3 / 6,
                     "chord_deficit": T013_LENGTH - chord,
                     "exact_chord_deficit": T013_LENGTH - 2 * rho * math.sin(T013_LENGTH / (2 * rho))})
    fit = [row for row in tori if row["major"] >= T013_FIT_FROM]
    radii = [row["major"] for row in fit]
    return {"plane_cylinder": {"jacobi_max_difference": jacobi_difference, "j_head_minus_s_max": j_minus_s,
                               "chord_plane": chord_plane, "chord_cylinder": chord_cyl, "chord_cylinder_exact": helix,
                               "length": length, "heading": heading},
            "tori": tori,
            "deviation_exponent": core.loglog_slope(radii, [r["deviation"] for r in fit]),
            "deviation_exponent_exact": core.loglog_slope(radii, [r["exact_deviation"] for r in fit]),
            "chord_deficit_exponent": core.loglog_slope(radii, [r["chord_deficit"] for r in fit]),
            "max_relative_error_vs_closed_form": max(abs(r["deviation"] / r["exact_deviation"] - 1) for r in tori),
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
    findings = [
        finding("Cylinder and plane Jacobi fields coincide exactly: j_head(s) = s", "numerical",
                {"jacobi_max_difference": pc["jacobi_max_difference"], "j_head_minus_s_max": pc["j_head_minus_s_max"]},
                {"generator": _gen("plane-cylinder", steps=60),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("exact_arithmetic", "max |Jacobi(cylinder) - Jacobi(plane)|",
                                       pc["jacobi_max_difference"], 0.0),
                            core.check("analytic", "max |j_head(s) - s|", pc["j_head_minus_s_max"], 1e-13)]},
                tolerance={"abs": 1e-13, "rel": 0.0}),
        finding("Equal Jacobi fields do not imply equal chords: the helix chord is shorter than the plane chord",
                "numerical", {"chord_plane": pc["chord_plane"], "chord_cylinder": pc["chord_cylinder"]},
                {"generator": _gen("plane-cylinder", length=pc["length"]),
                 "checks": [core.check("analytic", "cylinder chord minus helix closed form",
                                       pc["chord_cylinder"] - pc["chord_cylinder_exact"], 1e-9),
                            core.check("analytic", "plane chord minus L", pc["chord_plane"] - pc["length"], 1e-12),
                            core.check("invariant", "plane chord minus cylinder chord",
                                       pc["chord_plane"] - pc["chord_cylinder"], 0.1, "ge")]},
                tolerance={"abs": 1e-9, "rel": 1e-9},
                counterexample={"statement": "Surfaces with identical Jacobi fields (intrinsic geometry) have identical "
                                             "chords between corresponding points",
                                "witness": {"surfaces": ["Plane", "Cylinder(1)"], "start": [0.2, 0.1], "heading": 0.6,
                                            "length": pc["length"], "chord_plane": pc["chord_plane"],
                                            "chord_cylinder": pc["chord_cylinder"]}}),
        finding("Torus outer-equator Jacobi deviation L - j_head(L) matches the closed form for every major radius",
                "numerical", study["max_relative_error_vs_closed_form"],
                {"generator": _gen("torus-flat-limit", radii=list(T013_RADII), steps=T013_STEPS),
                 "checks": [core.check("analytic", "max relative error of L - j_head(L) against L - sin(wL)/w",
                                       study["max_relative_error_vs_closed_form"], 1e-7)]},
                tolerance={"abs": 1e-9, "rel": 0.5}),
        finding("The Jacobi deviation from flat decays like 1/R (fitted exponent near -1)", "numerical",
                {"fitted_exponent": study["deviation_exponent"], "closed_form_exponent": study["deviation_exponent_exact"],
                 "max_gap_to_K_L3_over_6": study["max_leading_order_relative_gap"]},
                {"generator": _gen("torus-flat-limit", fit_radii_from=T013_FIT_FROM),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("analytic", "fitted minus closed-form exponent over the same radii",
                                       study["deviation_exponent"] - study["deviation_exponent_exact"], 1e-4),
                            core.check("analytic", "fitted exponent plus 1 (asymptotic)",
                                       study["deviation_exponent"] + 1.0, 0.05),
                            core.check("analytic", "|dev / (K L^3/6) - 1| <= K L^2/20 at the smallest fitted R",
                                       study["max_leading_order_relative_gap"],
                                       1.05 * fit_bound["curvature"] * T013_LENGTH ** 2 / 20, "le")]},
                tolerance={"abs": 1e-6, "rel": 0.0}),
        finding("The chord deficit vanishes faster (exponent near -2) than the Jacobi deviation (near -1)",
                "numerical", study["chord_deficit_exponent"],
                {"generator": _gen("torus-flat-limit"),
                 "derivation": _derivation("t013-flat-and-developable-limit"),
                 "checks": [core.check("analytic", "chord-deficit exponent plus 2", study["chord_deficit_exponent"] + 2.0,
                                       0.1)]},
                tolerance={"abs": 1e-4, "rel": 0.0},
                counterexample={"statement": "Intrinsic (Jacobi) and extrinsic (chord) signatures of curvature vanish "
                                             "at the same rate in the flat limit",
                                "witness": {"jacobi_exponent": study["deviation_exponent"],
                                            "chord_exponent": study["chord_deficit_exponent"]}}),
        finding("A physical cylinder or large-radius torus workpiece shows these separations", "physical", None, {}),
    ]
    fields = _fields(
        hypothesis=("Intrinsic flatness (K = 0) makes Jacobi fields identical to the plane's even when the surface is "
                    "curved in space, while chords (extrinsic) differ; along the outer equator of a torus with "
                    "growing major radius the Jacobi deviation from flat vanishes like K L^3/6 ~ 1/R."),
        mathematical_model=("Cylinder: K = 0 so j_head = s exactly, but a helix of angle alpha has chord "
                            "sqrt((2R sin(L cos(alpha)/(2R)))^2 + (L sin(alpha))^2). Torus(R, 1) outer equator: "
                            "K = 1/(R + 1), j_head = sin(wL)/w, w = sqrt(K), L - j_head = K L^3/6 - K^2 L^5/120 + ...; "
                            "the equator is a circle of radius R + 1, chord deficit = L^3/(24 (R + 1)^2) + ..."),
        input_data=["Plane and Cylinder(1): start (0.2, 0.1), heading 0.6, L = 3, RK4 N = 60",
                    f"Torus(R, 1), R in {list(T013_RADII)}, outer equator start (0, 0) heading 0, L = {T013_LENGTH}, "
                    f"RK4 N = {T013_STEPS}; exponents fitted for R >= {T013_FIT_FROM}"],
        observation_model="Jacobi columns at every node; embedded chord |X(L) - X(0)|; no renormalization.",
        expected_invariant=("Cylinder and plane Jacobi columns identical; exponent of L - j_head(L) in R tends to -1, "
                            "of the chord deficit to -2."),
        experiment=("Integrate both flat surfaces on identical grids and compare Jacobi columns bit for bit; integrate "
                    "the torus equator for each R, compare with closed forms and fit log-log exponents."),
        numerical_result=(f"Plane/cylinder Jacobi difference {_g(pc['jacobi_max_difference'], 2)}; chords "
                          f"{_g(pc['chord_plane'], 8)} (plane) versus {_g(pc['chord_cylinder'], 8)} (cylinder); torus "
                          f"deviation exponent {_g(study['deviation_exponent'], 5)} (closed form "
                          f"{_g(study['deviation_exponent_exact'], 5)}), chord-deficit exponent "
                          f"{_g(study['chord_deficit_exponent'], 5)}; max relative error versus closed form "
                          f"{_g(study['max_relative_error_vs_closed_form'], 2)}."),
        uncertainty=("RK4 error of j_head at h = 1/32 is below 1e-10 relative; fitted exponents include the "
                     "O(K L^2/20) correction, which is why the finite-R exponent is slightly above -1."),
        failure_modes_checked=["flat surfaces checked for spurious curvature (K = 0 exactly, j'' = 0 integrated exactly)",
                               "numerical deviation compared with its closed form, not only with K L^3/6",
                               "chord deficit computed from embedded points without cancellation below 1e-9 relative"],
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
    witness = None
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
            witness = {"L1": cut, "N1": n1, "L2": total, "N2": steps_total, "h_direct": h, "h_first": cut / n1,
                       "h_rest": (total - cut) / steps_rest, "differing_final_components": differing,
                       "max_abs_difference": float(np.max(np.abs(joined - direct)))}
            break
    rtol = 1e-9
    _, whole, _ = integrators.integrate_adaptive(f, y0, 2.0, rtol=rtol, atol=1e-12)
    _, part, _ = integrators.integrate_adaptive(f, y0, 1.25, rtol=rtol, atol=1e-12)
    _, cont, _ = integrators.integrate_adaptive(f, part[-1], 0.75, rtol=rtol, atol=1e-12)
    return {"dyadic_max_difference": dyadic, "non_dyadic_witness": witness,
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
                tolerance={"abs": 0.05, "rel": 0.0},
                counterexample={"statement": "Forward-then-reversed integration with a method of order p returns to "
                                             "the start with error proportional to h^p",
                                "witness": {"mean_orders": by_method, "surfaces": [c[0] for c in T014_CASES]}}),
        finding("Adaptive forward-then-reversed integration returns to the start at tolerance level", "numerical",
                adaptive_ratio,
                {"generator": _gen("reversal-adaptive", rtol=1e-9),
                 "checks": [core.check("invariant", "max return error / rtol", adaptive_ratio, 10.0, "le")]},
                tolerance={"abs": 0.5, "rel": 0.5}),
        finding("Truncating at L1 and continuing on the same dyadic grid reproduces direct integration bit for bit",
                "computational_pipeline", trunc["dyadic_max_difference"],
                {"generator": _gen("truncation", L1=1.25, L2=2.0, steps=256),
                 "checks": [core.check("exact_arithmetic", "max |continued - direct| (Euler, midpoint, RK4)",
                                       dyadic, 0.0)]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("With decimal truncation lengths the step sizes differ in the last bit and bitwise reproduction fails",
                "computational_pipeline",
                {"differs": witness is not None, "max_abs_difference": witness["max_abs_difference"] if witness else 0.0},
                {"generator": _gen("truncation-search", L2=3.0, N2=300),
                 "checks": [core.check("exact_arithmetic", "differing final state components",
                                       witness["differing_final_components"] if witness else 0, 1, "ge"),
                            core.check("invariant", "max difference stays at roundoff level",
                                       witness["max_abs_difference"] if witness else 1.0, 1e-12, "le")]},
                tolerance={"abs": 1e-12, "rel": 0.0},
                counterexample={"statement": "Truncate-and-continue reproduces fixed-step integration bit for bit "
                                             "for any truncation length", "witness": witness}),
        finding("Adaptive restart at L1 reproduces direct adaptive integration only to tolerance level", "numerical",
                restart_ratio,
                {"generator": _gen("truncation-adaptive", rtol=1e-9),
                 "checks": [core.check("invariant", "difference / rtol", restart_ratio, 10.0, "le"),
                            core.check("invariant", "restart is not bitwise: absolute difference",
                                       trunc["adaptive_restart"]["difference"], 1e-15, "ge")]},
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
                          f"{_g(dyadic, 2)}; decimal witness L1 = {witness['L1'] if witness else None} with "
                          f"{witness['differing_final_components'] if witness else 0} differing final components "
                          f"(max {_g(witness['max_abs_difference'], 2) if witness else 0}); adaptive restart "
                          f"difference {_g(restart_ratio, 3)} x rtol."),
        uncertainty=("Orders are least-squares fits over four step sizes; the RK4 return errors stay above 1e-12, "
                     "far from roundoff. Adaptive ratios depend on the accepted step sequence."),
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
        series["angular_momentum"] = np.linalg.norm(moment - moment[0], axis=1)
        exact = surface.exact_embedded_geodesic(u0, surface.unit_tangent(u0, heading), s)
        series["position"] = np.linalg.norm(points - exact, axis=1)
    envelopes, exponents = {}, {}
    for name, values in series.items():
        running = np.maximum.accumulate(values)
        reached = [c for c in T015_CHECKPOINTS if c <= s[-1] + 1e-9]
        envelopes[name] = [float(running[np.searchsorted(s, c - 1e-9)]) for c in reached]
        exponents[name] = core.loglog_slope(reached, envelopes[name]) if len(reached) >= 3 else None
    return {"surface": key, "method": method, "failed_at": failed, "reached": float(s[-1]),
            "envelopes": envelopes, "exponents": exponents, "max_energy_error": float(np.max(series["energy"])),
            "accepted_steps": stats.get("accepted_steps"), **extra}


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
    exps = {label: row["exponents"] for label, row in study.items()}
    adaptive_energy = {k: study[f"{k}:adaptive"]["exponents"]["energy"] for k in ("torus", "sphere")}
    rk4_energy = {k: study[f"{k}:rk4"]["exponents"]["energy"] for k in ("torus", "sphere")}
    rk4_growth = {k: study[f"{k}:rk4"]["envelopes"]["energy"][-1] / study[f"{k}:rk4"]["envelopes"]["energy"][0]
                  for k in ("torus", "sphere")}
    pos = {m: study[f"sphere:{m}"]["exponents"]["position"] for m in ("rk4", "adaptive")}
    clair = {m: study[f"torus:{m}"]["exponents"]["clairaut"] for m in T015_METHODS}
    euler_t, euler_s = study["torus:euler"], study["sphere:euler"]
    findings = [
        finding("Adaptive DP45 energy error grows linearly with length on the torus and the sphere", "numerical",
                adaptive_energy,
                {"generator": _gen("drift", rtol=T015_RTOL, checkpoints=list(T015_CHECKPOINTS)),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("self_convergence", f"{k} energy-envelope exponent minus 1", v - 1.0, 0.2)
                            for k, v in adaptive_energy.items()]},
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Sphere position error grows like L for fixed-step RK4 and like L^2 for adaptive DP45", "numerical",
                pos,
                {"generator": _gen("drift", h=T015_H),
                 "derivation": _derivation("t015-long-horizon-drift"),
                 "checks": [core.check("analytic", "RK4 position exponent minus 1", pos["rk4"] - 1.0, 0.2),
                            core.check("analytic", "adaptive position exponent minus 2", pos["adaptive"] - 2.0, 0.2)]},
                tolerance={"abs": 0.02, "rel": 0.0}),
        finding("Fixed-step RK4 energy error stays oscillation-dominated over lengths 10-320 (envelope exponent < 0.5)",
                "numerical", {"exponents": rk4_energy, "envelope_growth_factor_10_to_320": rk4_growth},
                {"generator": _gen("drift", h=T015_H),
                 "checks": [core.check("invariant", f"{k} RK4 energy-envelope exponent", v, 0.5, "le")
                            for k, v in rk4_energy.items()]},
                tolerance={"abs": 0.02, "rel": 0.01},
                counterexample={"statement": "The energy error of a non-symplectic fixed-step integrator grows "
                                             "linearly with length at every horizon",
                                "witness": {"method": "rk4", "h": T015_H, "lengths": [10.0, 320.0],
                                            "exponents": rk4_energy, "growth_factors": rk4_growth}}),
        finding("Torus Clairaut drift: adaptive grows linearly; drift exponents by method", "numerical", clair,
                {"generator": _gen("drift"),
                 "checks": [core.check("self_convergence", "adaptive Clairaut exponent minus 1",
                                       clair["adaptive"] - 1.0, 0.2)]},
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
                                       euler_t["theta_max_abs"] - euler_t["theta_turning"], 1.0, "ge")]},
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
                unit="arclength", tolerance={"abs": T015_H, "rel": 0.0}),
    ]
    fields = _fields(
        hypothesis=("Non-symplectic integrators drift in the first integrals of the geodesic flow (speed, Clairaut "
                    "constant, angular momentum); adaptive local-error control accumulates a linear secular drift, "
                    "while a fixed step on these closed or quasi-periodic orbits can keep the drift bounded over "
                    "long stretches; bounded speed error does not guarantee the right orbit."),
        mathematical_model=("Energy g(v, v) = 1, torus Clairaut rho^2 phi' and sphere angular momentum X x X' are exact "
                            "first integrals. A constant speed error gives a constant frequency error on the sphere, hence "
                            "position error ~ L; a speed error growing ~ L gives position error ~ L^2. On Torus(2, 1) "
                            "the orbit oscillates between |theta| <= arccos((c - R)/r); a Clairaut drift across the "
                            "separatrix changes it into an orbit winding around the tube."),
        input_data=["Torus(2, 1) start (0, 0.5) heading 0.7 (bounded oscillation about the outer equator)",
                    "Unit sphere start (pi/2, 0) heading 1.0 (great circle)",
                    f"Fixed step h = {T015_H} for Euler, midpoint, RK4; DP45 rtol {T015_RTOL}, atol {T015_RTOL * 1e-3}; "
                    f"checkpoints {list(T015_CHECKPOINTS)}"],
        observation_model=("Running maxima (envelopes) of |I(s) - I(0)| at checkpoints; exponents are log-log slopes "
                           "over the reached checkpoints; no renormalization at any step."),
        expected_invariant="Adaptive drift exponent 1; sphere position exponent 1 (fixed RK4) and 2 (adaptive).",
        experiment=("One run per surface and method to L = 320 (fixed steps stop at a nonfinite state or at the "
                    "sphere chart's poles); envelopes and fits of energy, Clairaut, angular momentum and position error."),
        numerical_result=(f"Energy exponents: adaptive {', '.join(f'{k} {_g(v, 3)}' for k, v in adaptive_energy.items())}; "
                          f"RK4 {', '.join(f'{k} {_g(v, 3)}' for k, v in rk4_energy.items())}; sphere position exponent "
                          f"RK4 {_g(pos['rk4'], 3)}, adaptive {_g(pos['adaptive'], 3)}; torus Clairaut exponents "
                          f"{', '.join(f'{m} {_g(v, 3)}' for m, v in clair.items() if v is not None)}; torus Euler max "
                          f"energy error {_g(euler_t['max_energy_error'], 3)} but max |theta| "
                          f"{_g(euler_t['theta_max_abs'], 4)} versus turning latitude {_g(euler_t['theta_turning'], 4)}; "
                          f"sphere Euler stopped at s = {euler_s['failed_at']}."),
        uncertainty=("Envelope exponents mix oscillatory and secular parts; a late secular RK4 component is visible on "
                     "the torus after L ~ 100 and may dominate beyond L = 320. Fits use six checkpoints."),
        failure_modes_checked=["no hidden renormalization of speed", "chart exit at the sphere poles detected and "
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
T016_SADDLE_C = (1.0, 4.0, 16.0, 64.0, 256.0)
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


def negative_curvature_study():
    u0, heading, length = (0.0, 1.0), 0.6, T016_LENGTH
    rows = []
    for k in T016_K:
        plane = HyperbolicPlane(k)
        exact = math.sinh(k * length) / k
        adaptive = jacobi.transfer(plane, u0, heading, length, rtol=1e-10, atol=1e-13)
        fixed = jacobi.transfer(plane, u0, heading, length, steps=T016_STEPS)
        h = length / T016_STEPS
        start = np.asarray(u0, dtype=float)
        end_exact = plane.exact_geodesic(start, plane.unit_tangent(start, heading), [length])[0]
        linear = core.constant_curvature_transfer("rk4", -k * k, length, T016_STEPS)

        def rk4_error(n, k=k, exact=exact):
            return abs(core.constant_curvature_transfer("rk4", -k * k, length, n)[0, 1] - exact) / exact

        def midpoint_error(n, k=k, exact=exact):
            return abs(core.constant_curvature_transfer("implicit-midpoint", -k * k, length, n)[0, 1] - exact) / exact

        predicted_n = length * (length * k ** 5 / (120 * T016_TAU)) ** 0.25
        rows.append({
            "k": k, "exact_j_head": exact, "adaptive_relative_error": abs(adaptive.states[-1, 6] - exact) / exact,
            "adaptive_accepted_steps": adaptive.stats["accepted_steps"],
            "rk4_relative_error": abs(fixed.states[-1, 6] - exact) / exact,
            "rk4_predicted_relative_error": length * k ** 5 * h ** 4 / 120,
            "linear_system_matches_full_system": abs(linear[0, 1] - fixed.states[-1, 6]) / exact,
            "geodesic_endpoint_distance_error": core.hyperbolic_distance(k, fixed.states[-1, :2], end_exact),
            "rk4_steps_required": core.minimal_steps(rk4_error, T016_TAU),
            "rk4_steps_predicted": predicted_n,
            "implicit_midpoint_steps_required": core.minimal_steps(midpoint_error, T016_TAU),
            "rk4_stability_steps": math.ceil(k * length / RK4_STABILITY)})
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
         ("implicit midpoint steps", ks, [r["implicit_midpoint_steps_required"] for r in rows]),
         ("DP45 accepted steps (rtol 1e-10)", ks, [r["adaptive_accepted_steps"] for r in rows]),
         ("RK4 stability limit", ks, [r["rk4_stability_steps"] for r in rows])],
        title="HyperbolicPlane(k), L = 2: steps versus k", xlabel="k (K = -k^2)", ylabel="steps", logx=True, logy=True))
    cs = [r["c"] for r in saddle]
    ctx.artifact_text("saddle-growth.svg", svg.line_plot(
        [("j_head(L)", cs, [r["j_head"] for r in saddle]), ("peak |K| = c^2", cs, [r["peak_abs_curvature"] for r in saddle]),
         ("DP45 accepted steps", cs, [r["accepted_steps"] for r in saddle])],
        title="Saddle(c): ridge geodesic crossing the saddle point", xlabel="c", ylabel="value", logx=True, logy=True))
    growth = core.loglog_slope([k * T016_LENGTH for k in ks], [math.log(r["exact_j_head"] * r["k"] * 2) for r in rows])
    rk4_exp = core.loglog_slope(ks, [r["rk4_relative_error"] for r in rows])
    ratio_pred = [r["rk4_relative_error"] / r["rk4_predicted_relative_error"] for r in rows]
    steps_exp = core.loglog_slope(ks, [r["rk4_steps_required"] for r in rows])
    im_exp = core.loglog_slope(ks, [r["implicit_midpoint_steps_required"] for r in rows])
    adaptive_exp = core.loglog_slope(ks, [r["adaptive_accepted_steps"] for r in rows])
    stiffness = [r["rk4_steps_required"] / r["rk4_stability_steps"] for r in rows]
    im_over_rk4 = [r["implicit_midpoint_steps_required"] / r["rk4_steps_required"] for r in rows]
    geo_err = [r["geodesic_endpoint_distance_error"] for r in rows]
    large = [r for r in saddle if r["c"] >= 16.0]
    saddle_exp = core.loglog_slope([r["c"] for r in large], [r["j_head"] for r in large])
    increments = [b["accepted_steps"] - a["accepted_steps"] for a, b in zip(saddle, saddle[1:])]
    beyond = study["beyond_pole"]
    findings = [
        finding("Jacobi fields on HyperbolicPlane(k) grow like sinh(kL)/k and adaptive integration resolves them",
                "numerical", {"max_adaptive_relative_error": max(r["adaptive_relative_error"] for r in rows),
                              "j_head_at_k_8": rows[-1]["exact_j_head"]},
                {"generator": _gen("hyperbolic-k", k=list(T016_K), length=T016_LENGTH, rtol=1e-10),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "max relative error of j_head(L) against sinh(kL)/k",
                                       max(r["adaptive_relative_error"] for r in rows), 1e-8)]},
                tolerance={"abs": 1e-8, "rel": 1e-9}),
        finding("Fixed-step RK4 relative error grows like k^5 and matches L k^5 h^4 / 120", "numerical",
                {"exponent": rk4_exp, "ratio_to_prediction": ratio_pred},
                {"generator": _gen("hyperbolic-k", steps=T016_STEPS),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("analytic", "fitted exponent minus 5", rk4_exp - 5.0, 0.2),
                            core.check("analytic", "max |measured / predicted - 1|",
                                       max(abs(v - 1.0) for v in ratio_pred), 0.2)]},
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
                            core.check("self_convergence", "DP45 step exponent minus 1", adaptive_exp - 1.0, 0.2)]},
                tolerance={"abs": 0.03, "rel": 0.0}),
        finding("This is intrinsic exponential instability, not stiffness: accuracy, not stability, sets the step",
                "numerical", {"steps_required_over_stability_limit": stiffness,
                              "jacobian_eigenvalues": "+k and -k (ratio 1)",
                              "geodesic_endpoint_distance_error": geo_err, "log_growth_slope_in_kL": growth},
                {"generator": _gen("hyperbolic-k"),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("invariant", "min over k of accuracy steps / RK4 stability steps", min(stiffness),
                                       5.0, "ge"),
                            core.check("analytic", "d log(2k j_head) / d(kL) minus 1", growth - 1.0, 0.01)]},
                tolerance={"abs": 1e-8, "rel": 0.05}),
        finding("Implicit midpoint does not remove the cost and is qualitatively wrong beyond kh = 2", "numerical",
                {"implicit_over_rk4_steps": im_over_rk4, "implicit_step_exponent": im_exp,
                 "sign_changes_at_kh": [beyond["kh"], beyond["implicit_midpoint_sign_changes"]]},
                {"generator": _gen("implicit-midpoint", k=beyond["k"], steps=7),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("invariant", "min over k of implicit-midpoint / RK4 required steps",
                                       min(im_over_rk4), 1.0, "ge"),
                            core.check("analytic", "implicit-midpoint step exponent minus 3/2", im_exp - 1.5, 0.15),
                            core.check("invariant", "sign changes of the implicit-midpoint j at kh > 2",
                                       beyond["implicit_midpoint_sign_changes"], 1, "ge")]},
                tolerance={"abs": 0.0, "rel": 0.05},
                counterexample={"statement": "An implicit (A-stable) integrator removes the step restriction on "
                                             "strongly negatively curved surfaces",
                                "witness": {"k": beyond["k"], "kh": beyond["kh"],
                                            "j_implicit_midpoint": beyond["j_implicit_midpoint"],
                                            "j_exact": beyond["j_exact"],
                                            "implicit_over_rk4_steps": im_over_rk4}}),
        finding("Saddle(c): peak |K| = c^2 but Jacobi growth is polynomial, j_head(L) ~ c^sqrt(2)", "numerical",
                {"exponent_c_16_to_256": saddle_exp, "j_head": [r["j_head"] for r in saddle]},
                {"generator": _gen("saddle-ridge", c=list(T016_SADDLE_C), rtol=1e-9),
                 "derivation": _derivation("t016-strongly-negative-curvature"),
                 "checks": [core.check("self_convergence", "max relative change rtol 1e-9 -> 1e-11",
                                       max(r["self_convergence"] for r in saddle), 1e-6),
                            core.check("invariant", "max |y| on the ridge geodesic (symmetry)",
                                       max(r["max_abs_y"] for r in saddle), 1e-12),
                            core.check("analytic", "fitted exponent minus sqrt(2) (far-field K ~ -1/(4 s^2))",
                                       saddle_exp - math.sqrt(2.0), 0.1)]},
                tolerance={"abs": 0.0, "rel": 1e-4},
                counterexample={"statement": "Jacobi growth is exponential in sqrt(peak |K|) times the length",
                                "witness": {"c": saddle[-1]["c"], "peak_abs_curvature": saddle[-1]["peak_abs_curvature"],
                                            "log_j_head": math.log(saddle[-1]["j_head"]),
                                            "sqrt_peak_times_L": 2.0 * saddle[-1]["c"]}}),
    ]
    fields = _fields(
        hypothesis=("On K = -k^2 the Jacobi fields grow like e^(kL), so absolute errors are amplified by e^(kL) and a "
                    "fixed relative accuracy needs steps growing with k; this is intrinsic instability of the flow "
                    "(eigenvalues +k and -k of the Jacobi linearization), not classical stiffness, so implicit methods "
                    "do not remove it. Concentrated negative curvature (Saddle with large c) does not produce "
                    "exponential growth."),
        mathematical_model=("j'' = k^2 j, j_head = sinh(kL)/k. RK4 on the growing mode: R(z) = e^z (1 - z^5/120 + ...), "
                            "relative error ~ L k^5 h^4 / 120, so N(tau) = L (L k^5 / (120 tau))^(1/4) ~ k^(5/4). Implicit "
                            "midpoint: R(z) = (1 + z/2)/(1 - z/2) = e^z (1 + z^3/12 + ...), N ~ k^(3/2), pole at z = 2, "
                            "negative factor beyond. RK4 stability for the decaying mode needs kh <= 2.785. Saddle ridge "
                            "y = 0: K = -c^2/(1 + c^2 x^2)^2 ~ -1/(4 s^2) away from the saddle point, so j grows like "
                            "s^((1 + sqrt(2))/2) after a kick ~ c at the saddle point; heuristic total ~ c^sqrt(2)."),
        input_data=[f"HyperbolicPlane(k), k in {list(T016_K)}, start (0, 1), heading 0.6, L = {T016_LENGTH}",
                    f"RK4 N = {T016_STEPS}; DP45 rtol 1e-10; relative accuracy target {T016_TAU}",
                    f"Saddle(c), c in {list(T016_SADDLE_C)}, ridge geodesic y = 0 from arclength 1 before the saddle "
                    "point, L = 2, DP45 rtol 1e-9 (checked at 1e-11)"],
        observation_model=("Relative error of j_head(L); required steps by doubling and bisection on the exact RK4 and "
                           "implicit-midpoint transfer matrices (matrix powers, verified equal to the full integration); "
                           "hyperbolic distance of the geodesic endpoint to the exact semicircle."),
        expected_invariant="Exponents 5 (error), 5/4 (RK4 steps), 3/2 (implicit midpoint), ~1 (DP45 steps).",
        experiment=("Integrate the joint geodesic/Jacobi system per k (adaptive and RK4), search the minimal step "
                    "counts, compare with stability limits, iterate implicit midpoint beyond its pole, and integrate "
                    "the saddle ridge geodesic for growing c."),
        numerical_result=(f"RK4 error exponent {_g(rk4_exp, 4)} (measured/predicted {_g(min(ratio_pred), 3)}-"
                          f"{_g(max(ratio_pred), 3)}); required RK4 steps {[r['rk4_steps_required'] for r in rows]} "
                          f"(exponent {_g(steps_exp, 3)}), implicit midpoint {[r['implicit_midpoint_steps_required'] for r in rows]} "
                          f"(exponent {_g(im_exp, 3)}), DP45 {[r['adaptive_accepted_steps'] for r in rows]} (exponent "
                          f"{_g(adaptive_exp, 3)}); accuracy/stability step ratio {_g(min(stiffness), 3)}-{_g(max(stiffness), 3)}; "
                          f"RK4 geodesic endpoint error {_g(geo_err[0], 2)} (k = 1) to {_g(geo_err[-1], 3)} (k = 8); implicit "
                          f"midpoint at kh = {_g(beyond['kh'], 3)} changes sign {beyond['implicit_midpoint_sign_changes']} "
                          f"times; saddle j_head(L) {[round(r['j_head'], 3) for r in saddle]} (exponent {_g(saddle_exp, 3)}), "
                          f"DP45 steps grow by {increments} per factor 4 in c."),
        uncertainty=("Step counts are integers found by bisection assuming monotone error in N; the saddle exponent is "
                     "a finite-c fit of a heuristic asymptotic law (the local slopes are about 1.43-1.45)."),
        failure_modes_checked=["linear transfer-matrix search checked against the full geodesic/Jacobi integration",
                               "implicit-midpoint singular step (kh = 2) refused rather than divided by zero",
                               "saddle ridge symmetry (y = 0, x(L) = -x0) verified",
                               "adaptive saddle results checked by tightening rtol"],
        unresolved_assumptions=["The sqrt(2) saddle exponent is a heuristic far-field argument, not a proof",
                                "Only the ridge geodesic of the saddle is studied; oblique geodesics sample other K",
                                "Error amplification of the geodesic in the half-plane chart mixes chart compression near "
                                "y = 0 with intrinsic instability"],
        recommended_next_task="T017 (validity domains, which shrink like 1/cosh(ks) on the hyperbolic plane) and T018",
    )
    return {"state": "completed", "fields": fields, "findings": findings}
