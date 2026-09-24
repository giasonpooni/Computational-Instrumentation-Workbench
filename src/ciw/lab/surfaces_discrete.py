"""Surface interface, derivative checks, charts and singularities (T033-T037).

Scope: validates and extends the reusable surface interface of
``ciw.lab.surfaces`` without modifying it. T033 runs a conformance suite over
every catalogue surface, two reparametrized charts and a rigid rotation, and
shows that the suite rejects seeded defects. T034 checks the hand-coded
derivatives against sympy (distinct origin, optional) and against nested
forward-mode dual numbers (same origin). T035 maps the central-difference
error of metric derivatives over twelve decades of step size. T036 builds
chart atlases with exact transitions (two polar charts of the sphere, and the
Monge and polar charts of a graph surface) and integrates geodesics through
chart singularities with chart switching. T037 scans approaches to candidate
singular points and separates coordinate, conical and curvature singularities
and an infinite-distance boundary within stated detection limits, with
pointwise refusal codes.

Non-claims: all surfaces, coordinates and curvatures are normalized
mathematical objects. Findings establish agreement between computations on
sampled points of declared domains; they do not establish correctness between
samples, outside the domains, or for any measured physical surface. The
singularity rules are validated on the declared examples and misclassify
cases beyond their thresholds, which T037 records as counterexamples.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from . import svg
from .evidence import finding, holds as compare
from .integrators import integrate_fixed
from .registry import task
from .runner import builtin_identity
from .surfaces import GaussianBump, HyperbolicPlane, Plane, Reparametrized, Saddle, Sphere
from .surfaces_discrete_ad import DualMath, DualSurface, diffgeom_reference, formulas, partial, symbolic_reference
from .surfaces_discrete_charts import (CONICAL_TOLERANCE, CURVATURE_BLOWUP, DEGENERACY, DIVERGENCE_TOLERANCE,
                                       FIT_RESIDUAL, FIT_WINDOW, SWITCH_THRESHOLD, Approach, SphereAtlas, graph_atlas,
                                       great_circle, integrate_atlas, integrate_single, pole_passing_great_circle,
                                       refusal_code, require_regular, scan)
from .surfaces_discrete_geometry import (DOMAINS, EPS, SEED, STENCIL_STEP, THRESHOLDS, ConformalHalfPlane, Cone,
                                         CubeRootChart, DroppedCrossTermBump, PolarChart, PowerGraph,
                                         central_difference, conformance, conformance_surfaces, mutant_surfaces)

MODULE = "src/ciw/lab/surfaces_discrete.py"
GEOMETRY = "src/ciw/lab/surfaces_discrete_geometry.py"
AD = "src/ciw/lab/surfaces_discrete_ad.py"
CHARTS = "src/ciw/lab/surfaces_discrete_charts.py"
DOC = "docs/lab/SURFACE_INTERFACE.md"
# Core modules the numbers depend on: recorded in the runtime identity, not as changed files.
CORE = "src/ciw/lab/surfaces.py"
INTEGRATORS = "src/ciw/lab/integrators.py"
TESTS = "tests/test_lab_surfaces_discrete.py"
POINTS = 32
AD_POINTS = 12
# Each task's next step is its own deferred research question. T042 defines refusal states for triangle meshes
# (MESH_CODES, TRACE_CODES, QUERY_CODES) and T043 propagates vertex noise; neither delivers T033's, T035's or
# T037's question, so those are named as queue extensions after T168. Context sits in parentheses so that the
# planner's clause split (at ';' and before a joined task pointer) keeps each question whole.
MESH_REFUSALS = ("the refusal states that T042 defines (MESH_CODES, TRACE_CODES and QUERY_CODES in "
                 "surfaces_discrete_mesh_geometry.py) are for triangle meshes")
NEXT_STEPS = {
    "T033": ("Deferred research question (queue extension after T168): make this conformance suite the admission "
             "gate for smooth surface data, run on every Surface before a lab task integrates on it and refusing a "
             "nonconforming one with a named code per failed identity (the seeded defects are refused only inside "
             "T033 today, " + MESH_REFUSALS + " and do not use this suite)."),
    "T034": ("Deferred research question: an independent assembly of the connection and curvature for gaussian-bump, "
             "gaussian-bump-shear and rotated-torus (for example sympy.diffgeom as for the other seven surfaces, or a "
             "second computer algebra system), where ciw-written assembly of sympy derivatives gives only same-origin "
             "evidence, and complex-step derivatives once the core surfaces accept complex coordinates."),
    "T035": ("Deferred research question (queue extension after T168): with metric samples carrying noise of "
             "standard deviation sigma, does the optimal central-difference step move to about "
             "(sigma / |d^3 g|)^(1/3) and the smallest derivative error grow like sigma^(2/3)? (T043 propagates "
             "vertex noise to mesh lengths, normals and curvature but tests no difference-based metric derivative, "
             "so this law is not tested anywhere in the queue.)"),
    "T036": ("Deferred research question (queue extension after T168): locate the chart-switch crossing with event "
             "detection in the adaptive integrator (switching now happens between steps), and drive switching from "
             "the T037 require_regular refusal codes instead of a fixed regularity threshold."),
    "T037": ("Deferred research question (queue extension after T168): adopt these refusal codes (nonfinite_point, "
             "nonfinite_metric, degenerate_metric, curvature_blowup, outside_chart, curvature_singularity, "
             "conical_singularity) as the admission refusal states for smooth surface data, and add a removability "
             "test (metric regularity in radial arclength coordinates) so the cube-root chart's removable "
             "singularity is classified (" + MESH_REFUSALS + " and contain none of these codes)."),
}


def _check(reference, observed, tolerance, comparison="abs_le", kind="analytic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = compare(observed, tolerance, comparison)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _source_identity(name):
    data = Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _sig(value, digits=3):
    return float(f"{float(value):.{digits}g}")


def _fmt(value):
    return format(float(value), ".3g")


# ---------------------------------------------------------------- T033
CONFORMANCE_FAILURE_MODES = [
    "asymmetric metric matrix (seeded mutant)", "indefinite metric (seeded mutant)",
    "transposed derivative index order (seeded mutant)", "sign-flipped metric derivatives (seeded mutant)",
    "dropped mixed height derivative (seeded mutant)", "curvature misscaled by a factor R (seeded mutant)",
    "NaN metric derivatives on part of the domain (seeded mutant)",
    "points refused by the core Surface.check inside a declared domain"]
RESIDUAL_FLOOR = 1e-17  # log-axis floor for exact-zero residuals in the conformance figure
# Per-finding uncertainty (AUTHORING rule 5). Algebraic identities are exact up
# to rounding; the stencil-based identities carry the fourth-order stencil
# error at h = 1e-3 l: h^4 ~ 1e-12 truncation and eps / h ~ 2e-13 rounding,
# scaled by higher metric derivatives (largest on the hyperbolic plane, whose
# length scale y reaches 0.3 in the declared domain).
ALGEBRAIC_UNCERTAINTY = {"kind": "roundoff", "value": 1e-15,
                         "basis": "algebraic identity evaluated in floating point (einsum and 2x2 products)"}
EXACT_UNCERTAINTY = {"kind": "reference_error", "value": 0, "basis": "exact counts of identities and mutants"}
STENCIL_UNCERTAINTY = {
    name: {"kind": "truncation_bound", "value": value,
           "basis": f"fourth-order stencil (h = 1e-3 l) in the {what}: h^4 truncation and eps/h rounding scaled by "
                    "higher metric derivatives; the residual measures this stencil error, not a defect"}
    for name, value, what in (("gauss_equation", 1e-10, "second metric derivatives of the Brioschi curvature"),
                              ("derivative_consistency", 1e-11, "differences of the metric"),
                              ("mixed_partials", 1e-12, "differences of the exact metric derivatives"))}
# Rounding bounds the conformance figure records per point (svg.line_plot(..., rounding=...)): the largest change
# rounding (another BLAS kernel or platform) makes to a plotted residual between two runs, twice its roundoff in one
# run. An algebraic identity's roundoff is ALGEBRAIC_UNCERTAINTY; a stencil identity's is the stencil's eps / h at
# h = 1e-3 l, times 4 for the stencil weights (their absolute values sum to 1.5) and the metric-derivative scale.
ALGEBRAIC_ROUNDING = 2 * ALGEBRAIC_UNCERTAINTY["value"]
STENCIL_ROUNDING = 8 * EPS / STENCIL_STEP


@task("T033", changed_files=(MODULE, GEOMETRY, DOC),
      regression_tests=(f"{TESTS}::test_conformance_suite_accepts_every_surface",
                        f"{TESTS}::test_conformance_suite_rejects_seeded_mutants",
                        f"{TESTS}::test_conformance_reports_nonfinite_residuals",
                        f"{TESTS}::test_brioschi_recovers_supplied_curvature",
                        f"{TESTS}::test_t033_report",
                        f"{TESTS}::test_next_steps_do_not_hand_work_to_tasks_that_did_not_deliver_it"))
def surface_interface(ctx):
    surfaces = conformance_surfaces()
    table = {key: conformance(surface, DOMAINS[key], POINTS, SEED) for key, surface in surfaces.items()}
    mutants = {key: conformance(surface, domain, POINTS, SEED) for key, (surface, domain) in mutant_surfaces().items()}
    worst = {}
    for name in THRESHOLDS:
        # A nonfinite residual already makes its surface nonconforming; the
        # worst finite value is still reported for the others.
        values = [row["worst"][name] for row in table.values() if isinstance(row["worst"][name], float)]
        worst[name] = min(values) if name == "min_eigenvalue_ratio" else max(values)
    nonconforming = sorted(key for key, row in table.items() if not row["conforms"])
    refused = sum(row["refused_by_core_check"] for row in table.values())
    undetected = sorted(key for key, row in mutants.items() if row["conforms"])
    rotated, base = surfaces["rotated-torus"], surfaces["torus"]
    rigid = max(float(np.max(np.abs(rotated.metric(u) - base.metric(u)))) / float(np.max(np.abs(base.metric(u))))
                for u in DOMAINS["torus"].sample(POINTS, SEED))
    ctx.artifact_json("conformance.json", {"thresholds": THRESHOLDS, "points_per_surface": POINTS, "seed": SEED,
                                           "order": list(table), "domains": {k: {"low": list(DOMAINS[k].low),
                                                                                 "high": list(DOMAINS[k].high)}
                                                                             for k in table},
                                           "surfaces": {k: dict(row, describe=surfaces[k].describe())
                                                        for k, row in table.items()}})
    ctx.artifact_json("mutants.json", {"detection": {k: {"failed": row["failed"], "not_evaluated": row["not_evaluated"],
                                                         "nonfinite": row["nonfinite"], "errors": row["errors"],
                                                         "worst": row["worst"]} for k, row in mutants.items()}})
    keys = list(table)
    plotted = ("gauss_equation", "derivative_consistency", "mixed_partials", "compatibility")
    series = [(name, list(range(1, len(keys) + 1)), [max(table[k]["worst"][name], RESIDUAL_FLOOR) for k in keys])
              for name in plotted]
    short = {"plane": "plane", "sphere": "sphere", "cylinder": "cyl", "saddle": "saddle", "torus": "torus",
             "gaussian-bump": "bump", "hyperbolic-plane": "hyp", "plane-polar": "polar",
             "gaussian-bump-shear": "shear", "rotated-torus": "rot-torus"}
    ctx.artifact_text("conformance-residuals.svg", svg.line_plot(
        series, title="Worst residual per surface; values < 1e-17 (and zeros) drawn at 1e-17",
        xlabel=" ".join(f"{i} {short.get(k, k)}" for i, k in enumerate(keys, 1)), ylabel="normalized residual",
        logy=True, rounding=[ALGEBRAIC_ROUNDING if name == "compatibility" else STENCIL_ROUNDING for name in plotted]),
        rounding_level=True)
    misscaled, flipped = mutants["misscaled-curvature"], mutants["sign-flipped-derivatives"]
    nan_mutant = mutants["nan-derivatives"]
    fields = {
        "hypothesis": ("Every catalogue surface, the reparametrized charts plane-polar and gaussian-bump-shear and a "
                       "rigidly rotated torus satisfy the interface identities on their declared domains, and the "
                       "Gauss equation holds: curvature recomputed from the metric alone equals the supplied K."),
        "mathematical_model": ("g symmetric positive definite; Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il - d_l g_ij) "
                               "symmetric in ij; d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il; d_l d_k g_ij symmetric "
                               "in kl; Brioschi K(E, F, G and their first and second derivatives) = supplied K."),
        "input_data": [f"{len(surfaces)} surfaces (catalogue + plane-polar, gaussian-bump-shear, rotated-torus)",
                       f"{POINTS} seeded points per surface (PCG64 seed {SEED}) in declared domains "
                       "(docs/lab/SURFACE_INTERFACE.md)",
                       f"{len(mutants)} seeded defect mutants",
                       "interface under test: ciw.lab.surfaces at the source digest recorded in "
                       "provider_runtime_identity.sources"],
        "observation_model": ("Normalized residuals: metric terms by max|g|, derivative terms by max|dg| + max|g|/l, "
                              "Christoffel terms by max|Gamma| + 1/l, curvature by |K| + 1/l^2, with l the local "
                              "length scale (y on the hyperbolic plane, 1 elsewhere). Second metric derivatives come "
                              "from fourth-order central differences (step 1e-3 l) of the exact first derivatives."),
        "expected_invariant": ("Algebraic identities at rounding level; stencil-based identities at <= 1e-8 (dg) "
                               "and <= 1e-7 (mixed partials, Gauss)."),
        "experiment": ("Run the conformance suite on every surface and every mutant; record worst residuals, failed "
                       "identities and whether the core Surface.check refuses any sampled point."),
        "numerical_result": (f"All {len(surfaces)} surfaces conform ({len(nonconforming)} nonconforming). Worst: Gauss "
                             f"{_fmt(worst['gauss_equation'])}, dg consistency {_fmt(worst['derivative_consistency'])}, "
                             f"mixed partials {_fmt(worst['mixed_partials'])}, compatibility {_fmt(worst['compatibility'])}, "
                             f"Christoffel asymmetry {_fmt(worst['christoffel_asymmetry'])}, min eigenvalue ratio "
                             f"{_fmt(worst['min_eigenvalue_ratio'])}. {len(mutants) - len(undetected)}/{len(mutants)} "
                             f"mutants rejected; the misscaled-curvature mutant fails only the Gauss equation "
                             f"({_fmt(misscaled['worst']['gauss_equation'])}); the sign-flipped mutant passes "
                             f"compatibility ({_fmt(flipped['worst']['compatibility'])})."),
        "uncertainty": ("Residuals are worst cases over 32 seeded points per surface; a defect confined between "
                        "samples or outside the declared domains would not be seen. The Gauss residual includes the "
                        "fourth-order stencil error (~h^4) of the second derivatives."),
        "failure_modes_checked": CONFORMANCE_FAILURE_MODES,
        "unresolved_assumptions": [
            "Declared domains avoid coordinate singularities; behaviour near them is T037's subject.",
            "Metric compatibility is an algebraic consequence of the Christoffel formula for any symmetric dg; it "
            "checks index conventions, not that dg is the derivative of g (derivative consistency does that).",
            "Christoffel symmetry is exact whenever dg is symmetric in its last two indices, because the core einsum "
            "is then symmetric term by term; the check adds evidence only for surfaces that override christoffel().",
            "Conformance on sampled points is evidence, not proof, of correctness on the whole domain."],
        "recommended_next_task": NEXT_STEPS["T033"],
        "provider_runtime_identity": builtin_identity((MODULE, GEOMETRY, DOC, CORE)),
    }
    findings = [
        finding("Brioschi curvature from the metric alone equals the supplied Gaussian curvature at every sampled point",
                "numerical", worst["gauss_equation"],
                {"derivation": "Theorema Egregium, Brioschi form; docs/lab/SURFACE_INTERFACE.md#conformance-suite",
                 "checks": [_check("Brioschi K (fourth-order differences of exact dg) vs supplied K, worst normalized",
                                   worst["gauss_equation"], THRESHOLDS["gauss_equation"], kind="invariant")]},
                unit="normalized residual", uncertainty=STENCIL_UNCERTAINTY["gauss_equation"],
                tolerance={"abs": THRESHOLDS["gauss_equation"], "rel": 0.0}),
        finding("Metrics are symmetric positive definite at every sampled point of the declared domains", "numerical",
                {"max_asymmetry": worst["metric_asymmetry"], "min_eigenvalue_ratio": worst["min_eigenvalue_ratio"],
                 "points_refused_by_core_check": refused},
                {"checks": [_check("|g_12 - g_21| / max|g|", worst["metric_asymmetry"], THRESHOLDS["metric_asymmetry"],
                                   kind="invariant"),
                            _check("min lambda_min / lambda_max over samples", worst["min_eigenvalue_ratio"],
                                   THRESHOLDS["min_eigenvalue_ratio"], comparison="ge", kind="invariant"),
                            _check("sampled points refused by core Surface.check", refused, 0, kind="exact_arithmetic")]},
                uncertainty={"kind": "roundoff", "value": 1e-15,
                             "basis": "metric entries and eigenvalues to a few ulps; the refusal count is exact"},
                tolerance={"abs": 1e-13, "rel": 1e-9}),
        finding("Christoffel symbols are symmetric in their lower indices at every sampled point", "numerical",
                worst["christoffel_asymmetry"],
                {"checks": [_check("max |Gamma^k_ij - Gamma^k_ji|, normalized", worst["christoffel_asymmetry"],
                                   THRESHOLDS["christoffel_asymmetry"], kind="invariant")]},
                uncertainty=ALGEBRAIC_UNCERTAINTY, tolerance={"abs": THRESHOLDS["christoffel_asymmetry"], "rel": 0.0}),
        finding("The connection is metric compatible at every sampled point: d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il",
                "numerical",
                worst["compatibility"],
                {"checks": [_check("max compatibility residual, normalized", worst["compatibility"],
                                   THRESHOLDS["compatibility"], kind="invariant")]},
                uncertainty=ALGEBRAIC_UNCERTAINTY, tolerance={"abs": THRESHOLDS["compatibility"], "rel": 0.0}),
        finding("Supplied metric derivatives agree with fourth-order differences of the metric at every sampled point",
                "numerical",
                worst["derivative_consistency"],
                {"checks": [_check("max |D4 g - dg|, normalized (h = 1e-3 l)", worst["derivative_consistency"],
                                   THRESHOLDS["derivative_consistency"], kind="self_convergence")]},
                uncertainty=STENCIL_UNCERTAINTY["derivative_consistency"],
                tolerance={"abs": THRESHOLDS["derivative_consistency"], "rel": 0.0}),
        finding("Differences of the exact metric derivatives have symmetric mixed partials (dg is a gradient field)",
                "numerical", worst["mixed_partials"],
                {"checks": [_check("max |d_u d_v g - d_v d_u g|, normalized", worst["mixed_partials"],
                                   THRESHOLDS["mixed_partials"], kind="invariant")]},
                uncertainty=STENCIL_UNCERTAINTY["mixed_partials"],
                tolerance={"abs": THRESHOLDS["mixed_partials"], "rel": 0.0}),
        finding("A rigid rotation leaves the torus metric unchanged at every sampled point", "numerical", rigid,
                {"checks": [_check("max |g_rotated - g_torus| / max|g|", rigid, 1e-14, kind="invariant")]},
                uncertainty=ALGEBRAIC_UNCERTAINTY, tolerance={"abs": 1e-14, "rel": 0.0}),
        finding("The conformance suite rejects every seeded defect mutant", "computational_pipeline",
                {"mutants": len(mutants), "undetected": len(undetected)},
                {"checks": [_check("mutants passing every identity", len(undetected), 0, kind="exact_arithmetic"),
                            _check("identities flagged nonfinite for the NaN-derivative mutant",
                                   len(nan_mutant["nonfinite"]), 1, comparison="ge", kind="exact_arithmetic")]},
                uncertainty=EXACT_UNCERTAINTY, tolerance={"abs": 0, "rel": 0}),
        finding("A curvature-misscaled sphere passes every identity except the Gauss equation", "computational_pipeline",
                {"failed": misscaled["failed"], "gauss_residual": _sig(misscaled["worst"]["gauss_equation"])},
                {"checks": [_check("identities other than the Gauss equation failed by the misscaled mutant",
                                   len(set(misscaled["failed"]) - {"gauss_equation"}), 0, kind="exact_arithmetic"),
                            _check("Gauss equation failed by the misscaled mutant (1 = yes)",
                                   int("gauss_equation" in misscaled["failed"]), 1, comparison="ge",
                                   kind="exact_arithmetic"),
                            _check("misscaled mutant Gauss residual", misscaled["worst"]["gauss_equation"], 0.1,
                                   comparison="ge", kind="invariant")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-10,
                             "basis": "stencil error of the Brioschi curvature (h = 1e-3 l); the misscaling residual "
                                      "itself is exact"},
                tolerance={"abs": 1e-6, "rel": 1e-6},
                counterexample={"statement": ("Metric symmetry, positive definiteness, Christoffel symmetry, metric "
                                              "compatibility and derivative consistency together certify a surface "
                                              "implementation"),
                                "witness": {"mutant": "misscaled-curvature (sphere R = 2 returning K = 1/R)",
                                            "failed": misscaled["failed"],
                                            "gauss_residual": _sig(misscaled["worst"]["gauss_equation"])}}),
        finding("Metric compatibility cannot detect wrong metric derivatives", "mathematical",
                {"compatibility_residual_passes": flipped["worst"]["compatibility"] <= THRESHOLDS["compatibility"],
                 "derivative_consistency_residual": _sig(flipped["worst"]["derivative_consistency"])},
                {"derivation": "Gamma^l_ki g_lj + Gamma^l_kj g_il = D_kij for any D symmetric in ij",
                 "checks": [_check("sign-flipped mutant compatibility residual", flipped["worst"]["compatibility"],
                                   THRESHOLDS["compatibility"], kind="invariant"),
                            _check("sign-flipped mutant derivative-consistency residual",
                                   flipped["worst"]["derivative_consistency"], 0.5, comparison="ge", kind="invariant")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-11,
                             "basis": "stencil error of the metric differences (h = 1e-3 l); the sign-flip residual "
                                      "itself is the defect"},
                tolerance={"abs": 1e-6, "rel": 1e-6},
                counterexample={"statement": "A metric-compatible connection certifies the metric derivatives",
                                "witness": {"mutant": "saddle with dg negated", "failed": flipped["failed"]}}),
        finding("The conformance suite certifies surfaces reconstructed from physical measurements", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T034
# Surfaces whose sympy.diffgeom Riemann tensor is cheap to assemble here; the
# three heavier ones (gaussian-bump, gaussian-bump-shear, rotated-torus) get
# sympy derivatives with the connection and curvature assembled in ciw code.
DIFFGEOM_KEYS = ("plane", "sphere", "cylinder", "saddle", "torus", "hyperbolic-plane", "plane-polar")


def _declared_curvature(key, surface, sp, u, v):
    """Closed forms of ciw.lab.surfaces restated with the exact rationals of their float parameters."""
    q = sp.Rational
    if key in ("plane", "cylinder", "plane-polar"):
        return sp.Integer(0)
    if key == "sphere":
        return 1 / q(surface.radius) ** 2
    if key == "saddle":
        c = q(surface.c)
        return -c ** 2 / (1 + c ** 2 * u ** 2 + c ** 2 * v ** 2) ** 2
    if key == "torus":
        big, small = q(surface.major), q(surface.minor)
        return sp.cos(v) / (small * (big + small * sp.cos(v)))
    if key == "hyperbolic-plane":
        return -q(surface.k) ** 2
    raise KeyError(key)


def _residuals(reference, surface, u, length):
    """Normalized residuals of whichever quantities ``reference`` supplies."""
    g, dg = surface.metric(u), surface.metric_derivatives(u)
    gamma, curvature = surface.christoffel(u), float(surface.gaussian_curvature(u))
    dg_scale = float(np.max(np.abs(dg))) + float(np.max(np.abs(g))) / length
    out = {}
    if "metric" in reference:
        out["metric"] = float(np.max(np.abs(reference["metric"] - g))) / float(np.max(np.abs(g)))
    if "metric_derivatives" in reference:
        out["metric_derivatives"] = float(np.max(np.abs(reference["metric_derivatives"] - dg))) / dg_scale
    if "christoffel" in reference:
        out["christoffel"] = (float(np.max(np.abs(reference["christoffel"] - gamma)))
                              / (float(np.max(np.abs(gamma))) + 1.0 / length))
    if "gaussian_curvature" in reference:
        out["gaussian_curvature"] = abs(reference["gaussian_curvature"] - curvature) / (abs(curvature) + length ** -2)
    return out


def dual_self_test() -> dict:
    """Dual-number derivatives of closed-form functions against hand derivatives, including nesting."""
    x0, y0 = 0.7, -0.4
    sin, cos, exp, sqrt = DualMath.sin, DualMath.cos, DualMath.exp, DualMath.sqrt

    def third(f, x):
        return partial(lambda w: partial(lambda z: partial(f, z, 0), w, 0), x, 0)

    cases = {
        "d/dx sin x": (partial(lambda w: sin(w[0]), [x0, y0], 0), math.cos(x0)),
        "d3/dx3 sin x": (third(lambda w: sin(w[0]), [x0, y0]), -math.cos(x0)),
        "d/dx 1/(1+x^2)": (partial(lambda w: 1.0 / (1.0 + w[0] * w[0]), [x0, y0], 0), -2 * x0 / (1 + x0 * x0) ** 2),
        "d/dx sqrt(1+x^2)": (partial(lambda w: sqrt(1.0 + w[0] * w[0]), [x0, y0], 0), x0 / math.sqrt(1 + x0 * x0)),
        "d2/dxdy sin x exp y": (partial(lambda w: partial(lambda z: sin(z[0]) * exp(z[1]), w, 1), [x0, y0], 0),
                                math.cos(x0) * math.exp(y0)),
        "d3/dx3 exp(x^2)": (third(lambda w: exp(w[0] * w[0]), [x0, y0]),
                            (8 * x0 ** 3 + 12 * x0) * math.exp(x0 * x0)),
        "d2/dx2 cos(x)/(2+cos(x))": (partial(lambda w: partial(lambda z: cos(z[0]) / (2.0 + cos(z[0])), w, 0), [x0, y0], 0),
                                     (-2 * math.cos(x0) * (2 + math.cos(x0)) - 4 * math.sin(x0) ** 2)
                                     / (2 + math.cos(x0)) ** 3),
        # Perturbation confusion (Siskind-Pearlmutter): d/dx [x d/dy (x + y)] = 1, not 2.
        "d/dx [x d/dy (x + y)]": (partial(lambda w: w[0] * partial(lambda z: w[0] + z[1], [w[0], w[1]], 1), [x0, y0], 0),
                                  1.0),
    }
    return {name: {"dual": float(got), "closed_form": float(want), "error": abs(float(got) - float(want))}
            for name, (got, want) in cases.items()}


@task("T034", changed_files=(MODULE, AD, GEOMETRY, DOC),
      regression_tests=(f"{TESTS}::test_dual_numbers_match_closed_forms",
                        f"{TESTS}::test_dual_derivatives_match_surface_interface",
                        f"{TESTS}::test_sympy_references_match_surface_interface",
                        f"{TESTS}::test_t034_report", f"{TESTS}::test_t034_degrades_without_sympy"))
def derivative_checks(ctx):
    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    have_sympy = ctx.available("module:sympy")
    if have_sympy:
        import sympy as sp
    dual_rows, sympy_rows, diffgeom_rows, exact = {}, {}, {}, {}
    for key, surface in surfaces.items():
        dual = DualSurface(*forms[key])
        reference = symbolic_reference(*forms[key]) if have_sympy else None
        geometric = None
        if have_sympy and key in DIFFGEOM_KEYS:
            geometric, expression, (u_sym, v_sym) = diffgeom_reference(*forms[key])
            simplified = sp.simplify(expression)
            declared = _declared_curvature(key, surface, sp, u_sym, v_sym)
            exact[key] = {"sympy_diffgeom": str(simplified), "declared": str(declared),
                          "difference_simplifies_to_zero": bool(sp.simplify(simplified - declared) == 0)}
        domain = DOMAINS[key]
        dual_worst, sym_worst, geo_worst = {}, {}, {}
        for u in domain.sample(AD_POINTS, SEED + 34):
            length = domain.length(u)
            row = _residuals({"metric": dual.metric(u), "metric_derivatives": dual.metric_derivatives(u),
                              "christoffel": dual.christoffel(u), "gaussian_curvature": dual.intrinsic_curvature(u)},
                             surface, u, length)
            extrinsic = dual.extrinsic_curvature(u)
            if extrinsic is not None:
                k = float(surface.gaussian_curvature(u))
                row["extrinsic_curvature"] = abs(extrinsic - k) / (abs(k) + length ** -2)
                row["embedding"] = float(np.max(np.abs(dual.point(u) - surface.embedding(u))))
            for name, value in row.items():
                dual_worst[name] = max(dual_worst.get(name, 0.0), value)
            for source, worst in ((reference, sym_worst), (geometric, geo_worst)):
                if source is not None:
                    for name, value in _residuals(source(u), surface, u, length).items():
                        worst[name] = max(worst.get(name, 0.0), value)
        dual_rows[key] = dual_worst
        if have_sympy:
            sympy_rows[key] = sym_worst
        if geo_worst:
            diffgeom_rows[key] = geo_worst
    dual_derivs = max(max(r["metric"], r["metric_derivatives"], r["christoffel"]) for r in dual_rows.values())
    dual_curv = max(max(r["gaussian_curvature"], r.get("extrinsic_curvature", 0.0)) for r in dual_rows.values())
    dual_embed = max(r.get("embedding", 0.0) for r in dual_rows.values())
    self_test = dual_self_test()
    self_error = max(case["error"] for case in self_test.values())
    # A seeded defect: the dropped mixed height derivative must be visible to the dual numbers.
    mutant, bump = DroppedCrossTermBump(0.5, 1.0), DualSurface(*forms["gaussian-bump"])
    defect = max(float(np.max(np.abs(mutant.metric_derivatives(u) - bump.metric_derivatives(u))))
                 / (float(np.max(np.abs(bump.metric_derivatives(u)))) + float(np.max(np.abs(bump.metric(u)))))
                 for u in DOMAINS["gaussian-bump"].sample(AD_POINTS, SEED + 34))
    ctx.artifact_json("dual-vs-ciw.json", {"points_per_surface": AD_POINTS, "seed": SEED + 34, "surfaces": dual_rows,
                                           "self_test": self_test, "dropped_cross_term_defect": defect})
    # The compared ciw values come from the core surfaces and, for plane-polar and
    # gaussian-bump-shear, from the chart maps of the section's geometry module.
    producer = {"implementation": "ciw.lab.surfaces",
                "revision": (f"surfaces.py sha256 {_source_identity('surfaces.py')}; surfaces_discrete_geometry.py "
                             f"sha256 {_source_identity('surfaces_discrete_geometry.py')}")}
    restated = {"implementation": "ciw.lab.surfaces_discrete._declared_curvature",
                "revision": f"surfaces_discrete.py sha256 {_source_identity('surfaces_discrete.py')}"}
    sympy_claims = (
        "sympy-differentiated metric and metric derivatives match the ciw surface interface on every conformance surface",
        "sympy.diffgeom Christoffel symbols match the ciw surface interface on seven surfaces",
        "sympy.diffgeom Riemann curvature R_1212 / det g matches the supplied Gaussian curvature on seven surfaces",
        "sympy.diffgeom curvature of seven surfaces simplifies exactly to the closed forms restated from ciw.lab.surfaces",
        "Christoffel symbols and curvature assembled in ciw code from sympy derivatives match the interface on every "
        "conformance surface")
    findings = []
    if have_sympy:
        checker = {"implementation": "sympy", "revision": sp.__version__}
        diffgeom_checker = {"implementation": "sympy.diffgeom", "revision": sp.__version__}
        sym_derivs = max(max(r["metric"], r["metric_derivatives"]) for r in sympy_rows.values())
        assembled = max(max(r["christoffel"], r["gaussian_curvature"]) for r in sympy_rows.values())
        geo_gamma = max(r["christoffel"] for r in diffgeom_rows.values())
        geo_curv = max(r["gaussian_curvature"] for r in diffgeom_rows.values())
        mismatches = sum(not row["difference_simplifies_to_zero"] for row in exact.values())
        ctx.artifact_json("sympy-vs-ciw.json", {"sympy": sp.__version__, "points_per_surface": AD_POINTS,
                                                "sympy_derivatives": sympy_rows, "sympy_diffgeom": diffgeom_rows,
                                                "exact_curvature": exact})
        findings += [
            finding(sympy_claims[0], "numerical", sym_derivs,
                    {"independent_check": dict(_check("max normalized residual over g and dg (sympy differentiation "
                                                      "of the re-expressed formula)", sym_derivs, 1e-12),
                                               producer=producer, checker=checker)},
                    unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12,
                                                             "basis": "floating evaluation of lambdified expressions"},
                    tolerance={"abs": 1e-12, "rel": 0.0}),
            finding(sympy_claims[1], "numerical", geo_gamma,
                    {"independent_check": dict(_check("max |Gamma_diffgeom - Gamma| / (max|Gamma| + 1/l) "
                                                      "(metric_to_Christoffel_2nd)", geo_gamma, 1e-12),
                                               producer=producer, checker=diffgeom_checker)},
                    unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12,
                                                             "basis": "floating evaluation of lambdified expressions"},
                    tolerance={"abs": 1e-12, "rel": 0.0}),
            finding(sympy_claims[2], "numerical", geo_curv,
                    {"independent_check": dict(_check("max |K_diffgeom - K| / (|K| + 1/l^2) "
                                                      "(metric_to_Riemann_components)", geo_curv, 1e-12),
                                               producer=producer, checker=diffgeom_checker)},
                    unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12,
                                                             "basis": "floating evaluation of lambdified expressions"},
                    tolerance={"abs": 1e-12, "rel": 0.0}),
            finding(sympy_claims[3], "mathematical", {"surfaces": len(exact), "mismatches": mismatches},
                    {"independent_check": dict(_check("restated closed forms whose difference from the diffgeom "
                                                      "curvature does not simplify to 0", mismatches, 0,
                                                      kind="exact_arithmetic"),
                                               producer=restated, checker=diffgeom_checker)},
                    uncertainty={"kind": "reference_error", "value": 0,
                                 "basis": "exact rational parameters; symbolic identity"},
                    tolerance={"abs": 0, "rel": 0}),
            finding(sympy_claims[4], "numerical", assembled,
                    {"checks": [_check("max normalized residual over Gamma and K (sympy differentiation; Levi-Civita "
                                       "and Riemann assembly written in ciw, same origin)", assembled, 1e-12,
                                       kind="cross_implementation")]},
                    unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12,
                                                             "basis": "floating evaluation of lambdified expressions"},
                    tolerance={"abs": 1e-12, "rel": 0.0}),
        ]
        state = "completed"
    else:
        findings += [finding(claim, "mathematical" if index == 3 else "numerical", None, {},
                             expected_not_established=True) for index, claim in enumerate(sympy_claims)]
        state = "partial"
    findings += [
        finding("Nested dual-number derivatives of re-expressed embeddings match the metric, dg and Christoffel symbols",
                "numerical", dual_derivs,
                {"checks": [_check("max normalized residual over metric, dg and Gamma (same ciw origin)", dual_derivs,
                                   1e-12, kind="cross_implementation"),
                            _check("max |X_dual - X| (formula re-expresses the same embedding)", dual_embed, 1e-13,
                                   kind="cross_implementation")]},
                unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "dual arithmetic"},
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Dual-number curvature (Brioschi with exact second derivatives, and LN - M^2) matches the supplied K",
                "numerical", dual_curv,
                {"checks": [_check("max normalized curvature residual", dual_curv, 1e-12,
                                   kind="cross_implementation")]},
                unit="normalized residual", uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "dual arithmetic"},
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Dual numbers reproduce closed-form first, mixed and third derivatives without perturbation confusion",
                "numerical", self_error,
                {"checks": [_check("max |dual - closed form| over 8 cases", self_error, 1e-13, kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-13, "basis": "dual arithmetic"},
                tolerance={"abs": 1e-13, "rel": 0.0}),
        finding("Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see",
                "numerical", _sig(defect, 6),
                {"checks": [_check("dropped-cross-term mutant: max normalized |dg_mutant - dg_dual|", defect, 1e-3,
                                   comparison="ge", kind="cross_implementation")]},
                uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "dual arithmetic"},
                tolerance={"abs": 1e-6, "rel": 1e-4},
                counterexample={"statement": "Finite, index-symmetric hand-coded metric derivatives are correct",
                                "witness": {"mutant": "gaussian-bump with f_xy dropped", "normalized_error": _sig(defect)}}),
        finding("Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from "
                "physical measurements", "physical", None, {}),
    ]
    fields = {
        "hypothesis": ("At seeded points of every conformance surface, the hand-coded metric, metric derivatives, "
                       "Christoffel symbols and Gaussian curvature equal those of its closed-form embedding (or "
                       "metric) computed by an independent symbolic system (sympy differentiation, and sympy.diffgeom "
                       "for the connection and curvature of seven surfaces) and by forward-mode automatic "
                       "differentiation; for those seven surfaces the symbolic curvature equals the declared closed "
                       "form exactly."),
        "mathematical_model": ("Re-expressed X(u) per surface with exact rational parameters; g = X_i . X_j; dg by "
                               "differentiation; sympy.diffgeom Gamma from metric_to_Christoffel_2nd and K = g_0m "
                               "R^m_101 / det g from metric_to_Riemann_components; ciw-assembled Gamma and R_1212 / det g "
                               "from sympy derivatives; dual-number K from Brioschi with exact second derivatives and "
                               "from LN - M^2."),
        "input_data": [f"{len(surfaces)} conformance surfaces, {AD_POINTS} points each (PCG64 seed {SEED + 34})",
                       f"sympy {'available' if have_sympy else 'unavailable'}; sympy.diffgeom and exact closed forms "
                       f"for {len(DIFFGEOM_KEYS)} surfaces ({', '.join(DIFFGEOM_KEYS)})"],
        "observation_model": ("Same normalization as T033; exact symbolic comparison by "
                              "sympy.simplify(simplify(K_diffgeom) - declared) == 0."),
        "expected_invariant": "Residuals at rounding level (<= 1e-12); exact closed forms identical.",
        "experiment": ("Evaluate sympy-lambdified, sympy.diffgeom and dual-number quantities at seeded points, compare "
                       "with the ciw interface, simplify the diffgeom curvature exactly, self-test the dual numbers, "
                       "and confirm a seeded derivative defect is exposed."),
        "numerical_result": (f"dual numbers: derivatives {_fmt(dual_derivs)}, curvature {_fmt(dual_curv)}, self-test "
                             f"{_fmt(self_error)}, defect exposed at {_fmt(defect)}"
                             + (f"; sympy: g and dg {_fmt(sym_derivs)}; sympy.diffgeom: Gamma {_fmt(geo_gamma)}, "
                                f"K {_fmt(geo_curv)}, exact closed forms {len(exact) - mismatches}/{len(exact)}; "
                                f"ciw assembly from sympy derivatives {_fmt(assembled)}"
                                if have_sympy else "; sympy unavailable, symbolic checks not run")),
        "uncertainty": ("Rounding only for the pointwise comparisons (floating evaluation of lambdified expressions "
                        "and dual arithmetic); agreement is shown at sampled points, and the exact symbolic identity "
                        "for seven closed forms with the declared parameters."),
        "failure_modes_checked": ["perturbation confusion in nested dual numbers",
                                  "formula re-expression not equal to the core embedding",
                                  "dropped mixed derivative (seeded defect)",
                                  "sign and index convention of the sympy.diffgeom Riemann components",
                                  "float parameters rounded by simplification (parameters are exact rationals)"],
        "unresolved_assumptions": [
            "The re-expressed formulas and the restated closed forms are written by hand from the same definitions "
            "as the core; a shared misunderstanding of a surface definition would pass both.",
            "sympy.diffgeom assembles the connection and curvature only for seven surfaces; for gaussian-bump, "
            "gaussian-bump-shear and rotated-torus they come from ciw-written assembly of sympy derivatives, which is "
            "same-origin evidence.",
            "Dual-number agreement is same-origin (ciw) evidence."]
            + ([] if have_sympy else ["sympy is not installed here, so the independent symbolic comparison did not run."]),
        "recommended_next_task": NEXT_STEPS["T034"],
        "provider_runtime_identity": builtin_identity((MODULE, AD, GEOMETRY, DOC, CORE)),
    }
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T035
FD_SURFACES = ("sphere", "torus", "gaussian-bump", "hyperbolic-plane")
FD_CONTROLS = ("saddle", "plane-polar", "plane")
TRUNCATION_WINDOW = (10 ** -3.5, 1e-2)
ROUNDING_WINDOW = (1e-13, 1e-10)
# Rounding bound the V-shape figure records per measured point: the largest change rounding (another BLAS kernel or
# platform) makes to a plotted error between two runs, twice the 2 eps / h that bounds the rounding branch (every
# error times h is within 2 eps in the rounding window). The predicted curve is closed-form and has none.
FD_ROUNDING = 4 * EPS
LEADING_STEP = -3.0  # log10 of the relative step where the h^2/6 d^3 g term is compared
# The signed leading-term comparison uses only points where the truncation
# term exceeds the rounding floor eps|g|/h by this factor, so rounding moves
# the relative deviation by at most ~1e-4 and the check measures truncation.
LEADING_MARGIN = 1e4


def fd_steps():
    return np.array([10.0 ** (-k / 4) for k in range(4, 53)])


def fd_study(points=12, seed=SEED + 35) -> dict:
    """Median central-difference error of dg over seeded points for relative steps 1e-1..1e-13.

    The prediction per point and component is h^2/6 |d^3_k g_ij| + eps |g_ij| / h
    with third derivatives from nested dual numbers. Per point, the smallest
    error over all steps is also compared with that point's smallest
    predicted error, so the pointwise statement does not rest on the median.
    """
    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    hs = fd_steps()
    out = {}
    for key in FD_SURFACES + FD_CONTROLS:
        surface, dual, domain = surfaces[key], DualSurface(*forms[key]), DOMAINS[key]
        errors = np.zeros((points, len(hs)))
        predicted = np.zeros_like(errors)
        leading, skipped = [], 0
        for a, u in enumerate(domain.sample(points, seed)):
            length = domain.length(u)
            g, dg = surface.metric(u), surface.metric_derivatives(u)
            scale = float(np.max(np.abs(g))) / length
            third = np.stack([dual.metric_third_derivative(u, k) for k in range(2)])
            for b, h in enumerate(hs):
                step = h * length
                fd = np.stack([central_difference(surface.metric, u, k, step) for k in range(2)])
                errors[a, b] = float(np.max(np.abs(fd - dg))) / scale
                predicted[a, b] = float(np.max(step ** 2 / 6 * np.abs(third) + EPS * np.abs(g)[None] / step)) / scale
                if abs(math.log10(h) - LEADING_STEP) < 1e-9 and np.max(np.abs(third)) > 0:
                    # Signed leading-term check: FD - dg = h^2/6 d^3 g + O(h^4) + rounding.
                    term = step ** 2 / 6 * third
                    floor = float(np.max(EPS * np.abs(g))) / step
                    if float(np.max(np.abs(term))) >= LEADING_MARGIN * floor:
                        leading.append(float(np.max(np.abs(fd - dg - term)) / np.max(np.abs(term))))
                    else:
                        skipped += 1
        median, pmedian = np.median(errors, axis=0), np.median(predicted, axis=0)
        pointwise_min = np.min(errors, axis=1)
        row = {"median_error": [float(v) for v in median], "median_predicted": [float(v) for v in pmedian],
               "leading_term_deviation": max(leading) if leading else None,
               "leading_term_points": len(leading), "leading_term_skipped": skipped,
               "pointwise_min_error": float(np.max(pointwise_min)),
               "pointwise_min_ratio": float(np.max(pointwise_min / np.min(predicted, axis=1)))}
        for name, (lo, hi) in (("truncation_slope", TRUNCATION_WINDOW), ("rounding_slope", ROUNDING_WINDOW)):
            mask = (hs >= lo * (1 - 1e-9)) & (hs <= hi * (1 + 1e-9))
            values = median[mask]
            row[name] = row[f"{name}_stderr"] = None
            if np.all(values > 0):
                x, y = np.log(hs[mask]), np.log(values)
                slope, intercept = np.polyfit(x, y, 1)
                residual = y - (slope * x + intercept)
                row[name] = float(slope)
                # Standard error of the least-squares slope from the fit residuals.
                row[f"{name}_stderr"] = float(math.sqrt(float(residual @ residual) / (len(x) - 2)
                                                        / float(np.sum((x - x.mean()) ** 2))))
        best, pbest = int(np.argmin(median)), int(np.argmin(pmedian))
        row.update({"h_opt": float(hs[best]), "h_opt_predicted": float(hs[pbest]), "min_error": float(median[best]),
                    "min_error_predicted": float(pmedian[pbest]),
                    "error_at_1e-2": float(median[np.argmin(np.abs(np.log(hs / 1e-2)))]),
                    "error_at_1e-12": float(median[np.argmin(np.abs(np.log(hs / 1e-12)))])})
        out[key] = row
    return {"steps": [float(h) for h in hs], "points": points, "seed": seed, "surfaces": out}


@task("T035", changed_files=(MODULE, GEOMETRY, AD, DOC),
      regression_tests=(f"{TESTS}::test_finite_difference_error_is_v_shaped", f"{TESTS}::test_t035_report",
                        f"{TESTS}::test_next_steps_do_not_hand_work_to_tasks_that_did_not_deliver_it"))
def finite_difference_derivatives(ctx):
    study = fd_study()
    rows = study["surfaces"]
    main = [rows[k] for k in FD_SURFACES]
    trunc = max(abs(r["truncation_slope"] - 2.0) for r in main)
    rounding = max(abs(r["rounding_slope"] + 1.0) for r in main)
    rounding_stderr = max(r["rounding_slope_stderr"] for r in main)
    leading = max(r["leading_term_deviation"] for r in main)
    leading_points = sum(r["leading_term_points"] for r in main)
    pointwise_min = max(r["pointwise_min_error"] for r in main)
    pointwise_ratio = max(r["pointwise_min_ratio"] for r in main)
    hopt_log = max(abs(math.log10(r["h_opt"] / r["h_opt_predicted"])) for r in main)
    bracket = [min(r["h_opt"] for r in main), max(r["h_opt"] for r in main)]
    min_ratio = max(r["min_error"] / r["min_error_predicted"] for r in main)
    worst_min = max(r["min_error"] for r in main)
    sphere = rows["sphere"]
    growth = sphere["error_at_1e-12"] / sphere["min_error"]
    control = max(rows[k]["error_at_1e-2"] for k in ("saddle", "plane-polar"))
    plane_zero = max(rows["plane"]["median_error"])
    ctx.artifact_json("fd-scan.json", study)
    hs = study["steps"]
    series = [(k, hs, rows[k]["median_error"]) for k in FD_SURFACES + ("saddle",)]
    series.append(("predicted (sphere)", hs, rows["sphere"]["median_predicted"]))
    ctx.artifact_text("fd-v-shape.svg", svg.line_plot(
        series, title="Central-difference error of metric derivatives vs step",
        xlabel="relative step h", ylabel="median max |D_h g - dg| / (max|g|/l)", logx=True, logy=True,
        rounding=[[FD_ROUNDING / h for h in hs]] * (len(series) - 1) + [0.0]), rounding_level=True)
    fields = {
        "hypothesis": ("Central differences of the metric approach the analytic dg as h^2 until rounding, which grows "
                       "as eps/h, takes over; the optimum lies near h* = (3 eps |g| / |d^3 g|)^(1/3) ~ eps^(1/3)."),
        "mathematical_model": ("D_h g = (g(u + h e_k) - g(u - h e_k)) / 2h = d_k g + h^2/6 d_k^3 g + O(h^4); rounding "
                               "error ~ eps |g| / h; total minimized at h* with error ~ eps^(2/3)."),
        "input_data": [f"{len(FD_SURFACES)} curved surfaces and {len(FD_CONTROLS)} controls, {study['points']} "
                       f"points each (seed {study['seed']})", f"{len(hs)} relative steps from 1e-1 to 1e-13 (4 per decade)"],
        "observation_model": ("Median over points of max_kij |D_h g_ij - d_k g_ij| / (max|g| / l); prediction uses "
                              "dual-number third derivatives."),
        "expected_invariant": "Slope 2 on the truncation branch, slope -1 on the rounding branch, V-bottom near eps^(1/3).",
        "experiment": "Scan h, fit both branches, locate the minimum, compare with the prediction and with controls.",
        "numerical_result": (f"truncation slopes within {_fmt(trunc)} of 2, leading term within {_fmt(leading)} "
                             f"relative at h = 1e-3 ({leading_points} points where truncation exceeds rounding by "
                             f"1e4); "
                             f"rounding slopes within {_fmt(rounding)} of -1; observed h* in [{_fmt(bracket[0])}, "
                             f"{_fmt(bracket[1])}] (eps^(1/3) = {_fmt(EPS ** (1 / 3))}), within 10^{_fmt(hopt_log)} of "
                             f"prediction; median minimum error <= {_fmt(worst_min)}, pointwise best error <= "
                             f"{_fmt(pointwise_min)} ({_fmt(pointwise_ratio)} of each point's predicted minimum); "
                             f"sphere error at h = 1e-12 is "
                             f"{_fmt(growth)} times the minimum; quadratic-metric controls {_fmt(control)} at h = 1e-2."),
        "uncertainty": ("Rounding-branch values depend on libm and summation details and scatter by tens of percent; "
                        "h* is resolved only on a grid of 4 points per decade (factor 1.78)."),
        "failure_modes_checked": ["rounding-dominated steps (h <= 1e-10)", "quadratic metrics with no truncation branch",
                                  "constant metric (exact zero difference)", "local length scale on the hyperbolic plane"],
        "unresolved_assumptions": [
            "Complex-step differentiation was not run: the core surfaces evaluate with math.* and do not accept "
            "complex coordinates.",
            "Measured surface samples would add noise sigma >> eps, moving the optimum to ~(sigma / |d^3 g|)^(1/3); "
            "nothing here measures that."],
        "recommended_next_task": NEXT_STEPS["T035"],
        "provider_runtime_identity": builtin_identity((MODULE, GEOMETRY, AD, DOC, CORE)),
    }
    findings = [
        finding("Central-difference error of the analytic metric derivatives falls as h^2 on the truncation branch",
                "numerical", {k: _sig(rows[k]["truncation_slope"], 6) for k in FD_SURFACES},
                {"checks": [_check("max |slope - 2| over h in [10^-3.5, 1e-2]", trunc, 0.02, kind="analytic"),
                            _check("max |FD - dg - h^2/6 d^3 g| / max|h^2/6 d^3 g| at h = 1e-3 over points where "
                                   "the term exceeds 1e4 eps|g|/h (dual-number third derivatives)", leading, 1e-3,
                                   kind="analytic"),
                            _check("points entering the leading-term comparison", leading_points, 40, comparison="ge",
                                   kind="exact_arithmetic")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-3,
                             "basis": "O(h^4) remainder plus rounding below 1e-4 of the term at h = 1e-3"},
                tolerance={"abs": 0.01, "rel": 0.0}),
        finding("Rounding error of central differences grows as 1/h for small steps", "numerical",
                {k: _sig(rows[k]["rounding_slope"], 4) for k in FD_SURFACES},
                {"checks": [_check("max |slope + 1| over h in [1e-13, 1e-10]", rounding, 0.2, kind="analytic")]},
                uncertainty={"kind": "fit", "value": _sig(rounding_stderr, 2),
                             "basis": "largest standard error of the least-squares slope over the rounding window "
                                      "(13 steps of the median error curve)"},
                tolerance={"abs": 0.2, "rel": 0.0}),
        finding("The optimal step lies within a factor 4 of the predicted h* = (3 eps |g| / |d^3 g|)^(1/3)", "numerical",
                {k: _sig(math.log10(rows[k]["h_opt"]), 4) for k in FD_SURFACES},
                {"derivation": "minimize h^2/6 |d^3 g| + eps |g| / h",
                 "checks": [_check("max |log10(h_opt / h_pred)|", hopt_log, math.log10(4.0), kind="analytic"),
                            _check("smallest observed h_opt", bracket[0], 1e-7, comparison="ge", kind="analytic"),
                            _check("largest observed h_opt", bracket[1], 1e-4, comparison="le", kind="analytic")]},
                unit="log10(relative step)",
                uncertainty={"kind": "reference_error", "value": 0.25,
                             "basis": "h_opt is located on a grid of 4 steps per decade and the flat V-bottom of the "
                                      "median curve can move it by one grid step (0.25 decade)"},
                tolerance={"abs": 0.75, "rel": 0.0}),
        finding("The median central-difference error curve bottoms out at or below the predicted minimum error",
                "numerical", _sig(worst_min, 2),
                {"checks": [_check("max over surfaces of median E(h_opt) / median E_pred(h_pred)", min_ratio, 1.0,
                                   comparison="le", kind="analytic"),
                            _check("max over surfaces of median E(h_opt)", worst_min, 1e-10, comparison="le",
                                   kind="analytic")]},
                unit="normalized error", uncertainty={"kind": "roundoff", "value": 1e-10,
                                                      "basis": "rounding floor eps^(2/3) of the V-bottom"},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("At every sampled point the best central difference agrees with the analytic metric derivatives to "
                "within twice that point's predicted minimum error",
                "numerical", _sig(pointwise_min, 2),
                {"checks": [_check("max over points of min_h E / min_h E_pred", pointwise_ratio, 2.0, comparison="le",
                                   kind="analytic"),
                            _check("max over points of min_h E (normalized)", pointwise_min, 2e-10, comparison="le",
                                   kind="analytic")]},
                unit="normalized error", uncertainty={"kind": "roundoff", "value": 2e-10,
                                                      "basis": "rounding of the difference quotient at its best step"},
                tolerance={"abs": 2e-10, "rel": 0.0}),
        finding("Smaller finite-difference steps can be far less accurate", "numerical", _sig(math.log10(growth), 3),
                {"checks": [_check("sphere E(1e-12) / E(h_opt)", growth, 1e3, comparison="ge", kind="analytic")]},
                unit="log10 error ratio",
                uncertainty={"kind": "reference_error", "value": 0.25,
                             "basis": "h_opt on a grid of 4 steps per decade and rounding-branch scatter of tens of "
                                      "percent (about 0.1 decade) in E(1e-12)"},
                tolerance={"abs": 0.5, "rel": 0.0},
                counterexample={"statement": ("Decreasing the finite-difference step always improves agreement with "
                                              "the analytic derivative"),
                                "witness": {"surface": "sphere", "log10_h_opt": _sig(math.log10(sphere["h_opt"]), 4),
                                            "log10_error_at_h_opt": _sig(math.log10(sphere["min_error"]), 4),
                                            "log10_error_at_1e-12": _sig(math.log10(sphere["error_at_1e-12"]), 4)}}),
        finding("Quadratic metrics have no truncation branch and a constant metric differences to exactly zero",
                "numerical", {"quadratic_error_at_1e-2": _sig(control, 2), "plane_max_error": plane_zero},
                {"checks": [_check("saddle and plane-polar E(1e-2)", control, 1e-13, comparison="le", kind="analytic"),
                            _check("plane E(h) for every h", plane_zero, 0.0, kind="exact_arithmetic")]},
                uncertainty={"kind": "roundoff", "value": 1e-13,
                             "basis": "rounding of the difference quotient at h = 1e-2 (eps |g| / h ~ 2e-14 "
                                      "normalized, times a few ulps of g); the plane value is exact"},
                tolerance={"abs": 1e-13, "rel": 0.0},
                counterexample={"statement": "Every smooth metric shows an O(h^2) truncation branch in central-difference error",
                                "witness": {"surface": "saddle, E = 1 + c^2 x^2 (degree-2 metric)",
                                            "error_at_1e-2": _sig(rows["saddle"]["error_at_1e-2"], 2)}}),
        finding("The optimal-step law derived here applies to derivatives of measured surface samples", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T036
DELTAS = (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
AZIMUTH = 1.3
STEPS = 400
# Meridian scan in chart A alone. Its angular momentum L = sin^2(theta) v_phi is
# 0 in exact arithmetic; rounding of g_12 seeds a platform-dependent one (0 to
# about 1e-18 on the OpenBLAS kernels measured) that each pole crossing
# amplifies. The scan therefore starts from a declared seed that dominates it,
# v_phi = MERIDIAN_SEED / sin^2(theta0); NATURAL_SEED_BOUND bounds the rounding
# seed at 1e-4 of the declared one. The STEPS grid is rerun at MERIDIAN_SEED /
# SEED_RATIO (natural share at most 1e-3 there), and the seed is swept over
# half decades on the step counts whose closest RK4 stage point lies within
# SWEEP_BAND failure radii of a pole; the failure boundary is checked against
# the band [BAND_INNER, BAND_OUTER] d* across the sweep.
MERIDIAN_STEPS = tuple(range(350, 451))
MERIDIAN_SEED = 1e-12
NATURAL_SEED_BOUND = 1e-16
SEED_RATIO = 10
SWEEP_SEEDS = (1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11, 3e-11, 1e-10)
SWEEP_BAND = 2.0
BAND_INNER, BAND_OUTER = 0.9, 1.5
# Graph atlas of the Gaussian bump (Monge chart plus its polar chart): starts
# at x = -GRAPH_START, offset delta from the apex, heading in +x; RK4 over
# GRAPH_LENGTH; the reference is the Monge chart alone at GRAPH_REFINE times the steps.
GRAPH_DELTAS = (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
GRAPH_START, GRAPH_LENGTH, GRAPH_STEPS, GRAPH_REFINE = 1.5, 3.0, 200, 4
GRAPH_DISC = 2.0  # radius of the Monge-chart disc sampled for graph-atlas transitions


# Numerical breakdowns that count as a single-chart failure; any other
# exception (a bad step count, an unknown method, a shape error) is a bug and
# propagates instead of strengthening the counterexample.
SINGLE_CHART_FAILURES = ("rk4 produced a nonfinite state at step", "math domain error", "math range error")


def _single(chart, u0, v0, length, steps):
    with np.errstate(all="ignore"):
        try:
            return integrate_single(chart, u0, v0, length, steps), None
        except (FloatingPointError, ValueError, OverflowError) as exc:
            message = str(exc)
            if not any(message.startswith(prefix) for prefix in SINGLE_CHART_FAILURES):
                raise
            return None, f"{type(exc).__name__}: {message}"


def _path_error(points, reference) -> float:
    return float(np.max(np.linalg.norm(np.asarray(points) - np.asarray(reference), axis=1)))


def atlas_study(radius=1.0) -> dict:
    atlas = SphereAtlas(radius)
    length = 2 * math.pi * radius
    rows = []
    for delta in DELTAS:
        start, tangent = pole_passing_great_circle(radius, delta, azimuth=AZIMUTH)
        u0 = atlas.to_chart("A", start)
        v0 = atlas.lift("A", u0, tangent)
        run = integrate_atlas(atlas, "A", u0, v0, length, STEPS)
        error = _path_error(run["points"], great_circle(start, tangent, radius, run["s"]))
        single, failure = _single(atlas.charts["A"], u0, v0, length, STEPS)
        single_error = None if single is None else _path_error(single["points"],
                                                               great_circle(start, tangent, radius, single["s"]))
        rows.append({"delta": delta, "atlas_error": error, "switches": len(run["switches"]),
                     "switch_path": [f"{w['from']}->{w['to']}" for w in run["switches"]],
                     "min_det_after_switch": min(w["det_after"] for w in run["switches"]) if run["switches"] else None,
                     "min_active_det": run["min_active_det"], "single_chart_error": single_error,
                     "single_chart_failure": failure,
                     "single_chart_min_det": None if single is None else single["min_det"]})
    convergence = []
    start, tangent = pole_passing_great_circle(radius, 0.0, azimuth=AZIMUTH)
    u0 = atlas.to_chart("A", start)
    v0 = atlas.lift("A", u0, tangent)
    for steps in (STEPS // 2, STEPS, 2 * STEPS):
        run = integrate_atlas(atlas, "A", u0, v0, length, steps)
        convergence.append({"steps": steps, "error": _path_error(run["points"],
                                                                 great_circle(start, tangent, radius, run["s"]))})
    orders = [math.log2(convergence[i]["error"] / convergence[i + 1]["error"]) for i in range(2)]
    # Pre-asymptotic spread of the 400-step error: its distance from the fourth-order
    # predictions e(200) / 16 and 16 e(800).
    coarse, middle, fine = (row["error"] for row in convergence)
    richardson = max(abs(middle - coarse / 16.0), abs(middle - 16.0 * fine))
    # Regularity along the delta = 1e-3 path: active chart versus chart A alone.
    start, tangent = pole_passing_great_circle(radius, 1e-3, azimuth=AZIMUTH)
    u0 = atlas.to_chart("A", start)
    run = integrate_atlas(atlas, "A", u0, atlas.lift("A", u0, tangent), length, STEPS)
    exact = great_circle(start, tangent, radius, run["s"])
    chart_a = [atlas.regularity("A", atlas.to_chart("A", x)) for x in exact]
    active = [max(atlas.regularity(name, atlas.to_chart(name, x)) for name in ("A", "B")) for x in exact]
    return {"atlas": atlas.describe(), "azimuth": AZIMUTH, "steps": STEPS, "length": length, "runs": rows,
            "convergence": convergence, "orders": orders, "richardson_spread": richardson,
            "regularity_profile": {"s": [float(v) for v in run["s"][::4]], "chart_A": chart_a[::4],
                                   "best_chart": active[::4]}}


def _stage_distance(steps, length, crossings) -> float:
    """Smallest distance between an RK4 stage abscissa (step point or half step) and a pole-crossing arclength."""
    s = np.arange(2 * steps + 1) * (length / (2 * steps))
    return float(min(np.min(np.abs(s - crossing)) for crossing in crossings))


def failure_radius(seed, step, radius=1.0) -> float:
    """d* = R (R L0 (h/R)^4)^(1/5): the seeded meridian fails when an RK4 stage point lands this close to a pole.

    Near a pole the chart is the polar chart of the tangent plane, which is
    scale invariant, so on the unit sphere the outcome depends on d/h and
    L0/h only. A crossing whose closest stage point lies at d multiplies L by
    a factor proportional to (h/d)^2 and turns nonlinear once L h^2/d^3
    reaches order 1; two crossings at d therefore fail for L0 h^4/d^5 >~ 1.
    The exponent follows from this argument; the constant 1 is fitted, and the
    boundary is not sharp (T036 checks it against a band of d* across a seed sweep).
    """
    return radius * (radius * seed * (step / radius) ** 4) ** 0.2


def blas_kernel() -> dict:
    """NumPy's BLAS build and the inputs of OpenBLAS's kernel choice: the OPENBLAS_CORETYPE override (None when
    the kernel is detected from the CPU) and the CPU features NumPy found.

    The runtime core name is not read: the package loads native code only in its declared hardware probes (T144).
    """
    info = {"openblas_coretype": os.environ.get("OPENBLAS_CORETYPE") or None}  # empty: detected from the CPU
    try:
        config = np.show_config(mode="dicts")
    except (TypeError, ValueError, AttributeError):  # older NumPy without mode="dicts"
        return info
    blas = config.get("Build Dependencies", {}).get("blas", {})
    info["blas"] = f"{blas.get('name')} {blas.get('version')}"
    info["simd_found"] = sorted(config.get("SIMD Extensions", {}).get("found", []))
    return info


def meridian_study(radius=1.0, step_counts=MERIDIAN_STEPS, seed=MERIDIAN_SEED, sweep_seeds=SWEEP_SEEDS) -> dict:
    """Chart A alone on the meridian (delta = 0) from a declared angular-momentum seed, for many step counts.

    The path crosses the north pole at arclength R and the south pole half a
    circle later. It starts from the lifted meridian tangent with v_phi
    replaced by seed / sin^2(theta0), so its angular momentum L = sin^2(theta)
    v_phi is the declared seed, not the rounding of g_12, and the reference is
    the exact great circle of that initial state. Each crossing amplifies L,
    the more the closer an RK4 stage point lands to the pole (failure_radius).
    Every other seed of ``sweep_seeds`` runs the step counts within SWEEP_BAND
    failure radii; STEPS is rerun at seed / SEED_RATIO. The unseeded run at
    STEPS records the platform's natural seed: g_12 and v_phi at the start,
    and L before the first crossing.
    """
    atlas = SphereAtlas(radius)
    chart = atlas.charts["A"]
    length = 2 * math.pi * radius
    start, tangent = pole_passing_great_circle(radius, 0.0, azimuth=AZIMUTH)
    u0 = atlas.to_chart("A", start)
    lifted = atlas.lift("A", u0, tangent)
    crossings = (radius, radius * (1.0 + math.pi))

    def initial(momentum):
        return np.array([lifted[0], momentum / math.sin(u0[0]) ** 2])

    def scan(momentum, counts):
        v0 = initial(momentum)
        velocity = chart.embedding_jacobian(u0) @ v0
        speed = float(np.linalg.norm(velocity))
        rows = []
        for steps in counts:
            single, failure = _single(chart, u0, v0, length, steps)
            rows.append({"steps": int(steps), "stage_distance": _stage_distance(steps, length, crossings),
                         "failure_radius": failure_radius(momentum, length / steps, radius),
                         "error": None if single is None else _path_error(
                             single["points"], great_circle(start, velocity / speed, radius, speed * single["s"])),
                         "failure": failure})
        return rows

    def momentum_along(v0):
        s, states = integrate_fixed(chart.geodesic_rhs, np.concatenate([u0, v0]), length, STEPS, "rk4")
        return s, states, np.sin(states[:, 0]) ** 2 * states[:, 3]

    def near(steps, momentum):
        return _stage_distance(steps, length, crossings) < SWEEP_BAND * failure_radius(momentum, length / steps, radius)

    rows = scan(seed, step_counts)
    sweep = [{"seed": other, "rows": scan(other, [n for n in step_counts if near(n, other)])}
             for other in sweep_seeds if other != seed]
    at_steps = next((row for row in rows if row["steps"] == STEPS), None) or scan(seed, [STEPS])[0]
    _, _, momentum = momentum_along(initial(seed))
    seeded = dict(at_steps, rerun_seed=seed / SEED_RATIO, rerun_error=scan(seed / SEED_RATIO, [STEPS])[0]["error"],
                  momentum_after_crossings=float(momentum[-1]), max_abs_momentum=float(np.max(np.abs(momentum))))
    s, states, momentum = momentum_along(lifted)
    points = np.array([chart.embedding(y[:2]) for y in states])
    natural = {"steps": STEPS, "blas_kernel": blas_kernel(), "g12_at_start": float(chart.metric(u0)[0, 1]),
               "v_phi_at_start": float(lifted[1]),
               "momentum_before_first_crossing": float(np.max(np.abs(momentum[s < crossings[0]]))),
               "max_abs_momentum": float(np.max(np.abs(momentum))), "max_abs_v_phi": float(np.max(np.abs(states[:, 3]))),
               "error": _path_error(points, great_circle(start, tangent, radius, s))}
    return {"seed": seed, "step_counts": [int(n) for n in step_counts], "failure_radius": "(L0 h^4)^(1/5)",
            "rows": rows, "sweep_band": SWEEP_BAND, "sweep": sweep, "at_steps": seeded, "natural": natural}


def transition_defects(atlas, source, target, points, rng, length=1.0, floor=0.05) -> dict:
    """Round trip, speed preservation and difference Jacobian of source -> target at ambient points.

    Every point enters the covering minimum (the better chart's regularity);
    the transition is evaluated only where both charts have regularity >= floor.
    """
    roundtrip = metric_defect = jacobian_defect = 0.0
    covering, used = math.inf, 0
    for x in points:
        ua, ub = atlas.to_chart(source, x), atlas.to_chart(target, x)
        ra, rb = atlas.regularity(source, ua), atlas.regularity(target, ub)
        covering = min(covering, max(ra, rb))
        if min(ra, rb) < floor:
            continue
        used += 1
        back = atlas.transition(target, source, atlas.transition(source, target, ua))
        roundtrip = max(roundtrip, float(np.linalg.norm(atlas.embedding(source, back) - x)) / length)
        va = rng.standard_normal(2)
        ub2, vb = atlas.pushforward(source, target, ua, va)
        speed_a, speed_b = atlas.charts[source].speed_squared(ua, va), atlas.charts[target].speed_squared(ub2, vb)
        metric_defect = max(metric_defect, abs(speed_b - speed_a) / speed_a)
        jac = np.column_stack([atlas.pushforward(source, target, ua, e)[1] for e in np.eye(2)])
        h = 1e-6
        fd = np.column_stack([
            _angle_difference(atlas.transition(source, target, ua + h * e),
                              atlas.transition(source, target, ua - h * e)) / (2 * h)
            for e in np.eye(2)])
        jacobian_defect = max(jacobian_defect, float(np.max(np.abs(fd - jac))) / float(np.max(np.abs(jac))))
    return {"used": used, "roundtrip": roundtrip, "metric_defect": metric_defect,
            "jacobian_defect": jacobian_defect, "covering_min": covering}


def transition_study(radius=1.0, count=256, dense=4096, seed=SEED + 36) -> dict:
    """Sphere atlas: transition defects at ``count`` points; covering bound through the atlas at count + dense points."""
    atlas = SphereAtlas(radius)
    rng = np.random.Generator(np.random.PCG64(seed))
    normals = rng.standard_normal((count, 3))
    points = radius * normals / np.linalg.norm(normals, axis=1)[:, None]
    result = transition_defects(atlas, "A", "B", points, rng, length=radius)
    # Dense covering check, evaluated through each chart's inverse and metric:
    # the better chart's regularity (sin^2 theta = det g / R^4) is at least 1/2.
    extra = rng.standard_normal((dense, 3))
    extra = radius * extra / np.linalg.norm(extra, axis=1)[:, None]
    covering = min([result["covering_min"]] + [max(atlas.regularity(name, atlas.to_chart(name, x)) for name in ("A", "B"))
                                               for x in extra])
    return dict(result, points=count, dense_points=dense, seed=seed, covering_min=covering)


def graph_transition_study(count=256, seed=SEED + 360, disc=GRAPH_DISC) -> dict:
    """Graph atlas of the Gaussian bump: transition defects at seeded points of a Monge-chart disc."""
    surface = GaussianBump(0.5, 1.0)
    atlas = graph_atlas(surface)
    rng = np.random.Generator(np.random.PCG64(seed))
    radii, angles = disc * np.sqrt(rng.random(count)), 2 * math.pi * rng.random(count)
    points = [atlas.embedding("monge", np.array([r * math.cos(t), r * math.sin(t)])) for r, t in zip(radii, angles)]
    result = transition_defects(atlas, "polar", "monge", points, rng)
    # The Monge chart has g = I + grad f grad f^T, so its regularity is
    # 1 / (1 + |grad f|^2) >= 1 / (1 + h^2 / (e sigma^2)) on the whole plane.
    bound = 1.0 / (1.0 + surface.h ** 2 / (math.e * surface.sigma ** 2))
    return dict(result, points=count, seed=seed, disc_radius=disc, monge_regularity_bound=bound)


def graph_study(deltas=GRAPH_DELTAS) -> dict:
    """Geodesics from the polar chart of the Gaussian bump through and near its apex, with and without switching."""
    atlas = graph_atlas(GaussianBump(0.5, 1.0))
    monge, polar = atlas.charts["monge"], atlas.charts["polar"]
    rows = []
    for delta in deltas:
        base = np.array([-GRAPH_START, delta])
        start = atlas.embedding("monge", base)
        tangent = monge.embedding_jacobian(base) @ np.array([1.0, 0.0])
        tangent = tangent / np.linalg.norm(tangent)
        vm = atlas.lift("monge", base, tangent)
        up = atlas.to_chart("polar", start)
        vp = atlas.lift("polar", up, tangent)
        _, fine = integrate_fixed(monge.geodesic_rhs, np.concatenate([base, vm]), GRAPH_LENGTH,
                                  GRAPH_REFINE * GRAPH_STEPS, "rk4")
        reference = np.array([monge.embedding(y[:2]) for y in fine[::GRAPH_REFINE]])
        run = integrate_atlas(atlas, "polar", up, vp, GRAPH_LENGTH, GRAPH_STEPS)
        alone = integrate_single(monge, base, vm, GRAPH_LENGTH, GRAPH_STEPS)
        single, failure = _single(polar, up, vp, GRAPH_LENGTH, GRAPH_STEPS)
        rows.append({"delta": delta, "atlas_error": _path_error(run["points"], reference),
                     "switches": len(run["switches"]), "switch_path": [f"{w['from']}->{w['to']}" for w in run["switches"]],
                     "min_active_regularity": run["min_active_det"],
                     "monge_error": _path_error(alone["points"], reference),
                     "polar_error": None if single is None else _path_error(single["points"], reference),
                     "polar_failure": failure})
    return {"atlas": atlas.describe(), "start": [-GRAPH_START, "delta"], "length": GRAPH_LENGTH, "steps": GRAPH_STEPS,
            "reference_steps": GRAPH_REFINE * GRAPH_STEPS, "runs": rows}


def _angle_difference(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (d + math.pi) % (2 * math.pi) - math.pi


def _log10(value) -> float:
    return _sig(math.log10(value), 4)


@task("T036", changed_files=(MODULE, CHARTS, GEOMETRY, DOC),
      regression_tests=(f"{TESTS}::test_atlas_transitions_are_exact",
                        f"{TESTS}::test_atlas_geodesic_through_pole_matches_great_circle",
                        f"{TESTS}::test_single_chart_near_pole_counterexample",
                        f"{TESTS}::test_meridian_depends_on_the_step_grid",
                        f"{TESTS}::test_graph_atlas_transitions_and_apex_geodesics", f"{TESTS}::test_t036_report"))
def chart_transitions(ctx):
    study = atlas_study()
    transitions = transition_study()
    meridian = meridian_study()
    graph = graph_study()
    graph_transitions = graph_transition_study()
    runs = {row["delta"]: row for row in study["runs"]}
    through = runs[0.0]
    atlas_errors = [row["atlas_error"] for row in study["runs"]]
    spread = max(atlas_errors) / min(atlas_errors)
    order_defect = max(abs(order - 4.0) for order in study["orders"])
    relative_richardson = study["richardson_spread"] / through["atlas_error"]
    min_after = min(row["min_det_after_switch"] for row in study["runs"])
    # Recorded after any switch, so it restates the threshold and covering bound; kept as a value only.
    min_active = min(row["min_active_det"] for row in study["runs"])
    degraded = [row for row in study["runs"] if row["delta"] > 0 and (
        row["single_chart_failure"] is not None or row["single_chart_error"] > 10 * row["atlas_error"])]
    failed = [row for row in study["runs"] if row["single_chart_failure"] is not None]
    # Seeded meridian in chart A alone: the step grid and the seed decide the outcome.
    natural, seeded = meridian["natural"], meridian["at_steps"]
    ceiling = 2.0  # a failed run counts as the largest possible error on the unit sphere, 2R

    def within(row, factor=1.0):
        return row["stage_distance"] < factor * row["failure_radius"]

    def ratio(row):
        return row["stage_distance"] / row["failure_radius"]

    def extremes(rows):
        # Closest approaches of the two outcomes to d*: the largest d/d* of a failed run and the smallest of any other.
        return (max((ratio(row) for row in rows if row["error"] is None), default=0.0),
                min((ratio(row) for row in rows if row["error"] is not None), default=0.0))

    near = [row for row in meridian["rows"] if within(row)]
    meridian_failed = [row["steps"] for row in meridian["rows"] if row["error"] is None]
    other_errors = [row["error"] for row in meridian["rows"] if not within(row) and row["error"] is not None] or [ceiling]
    declared_failed_ratio, declared_kept_ratio = extremes(meridian["rows"])
    # The declared seed's full scan and, at the other seeds, the step counts within SWEEP_BAND d*.
    swept = meridian["rows"] + [row for entry in meridian["sweep"] for row in entry["rows"]]
    swept_failed = [row for row in swept if row["error"] is None]
    inner_kept = [row for row in swept if within(row, BAND_INNER) and row["error"] is not None]
    outer_failed = [row for row in swept_failed if not within(row, BAND_OUTER)]
    failed_ratio, kept_ratio = extremes(swept)
    # Step counts that fail at the declared seed but not at a smaller seed: d* shrinks with the seed.
    moved = sorted({row["steps"] for entry in meridian["sweep"] if entry["seed"] < MERIDIAN_SEED
                    for row in entry["rows"] if row["error"] is not None and row["steps"] in meridian_failed})
    linearity = abs(seeded["error"] / (SEED_RATIO * seeded["rerun_error"]) - 1.0)
    natural_share = NATURAL_SEED_BOUND / MERIDIAN_SEED
    # Graph atlas of the Gaussian bump.
    graph_rows = graph["runs"]
    graph_atlas_error = max(row["atlas_error"] for row in graph_rows)
    graph_monge_error = max(row["monge_error"] for row in graph_rows)
    graph_switches = min(row["switches"] for row in graph_rows)
    graph_positive = [row for row in graph_rows if row["delta"] > 0]
    graph_degraded = [row for row in graph_positive if row["polar_failure"] is not None
                      or row["polar_error"] > 10 * row["atlas_error"]]
    graph_failed = [row for row in graph_positive if row["polar_failure"] is not None]
    ctx.artifact_json("atlas-runs.json", study)
    ctx.artifact_json("transitions.json", {"sphere": transitions, "graph": graph_transitions})
    ctx.artifact_json("meridian-steps.json", meridian)
    ctx.artifact_json("graph-atlas-runs.json", graph)
    positive = [row for row in study["runs"] if row["delta"] > 0]
    # Each run of consecutive successful single-chart deltas is its own series,
    # so the line never bridges a failure; failures sit at a fixed ceiling.
    single_series, current = [], []
    for row in positive:
        if row["single_chart_error"] is None:
            if current:
                single_series.append(current)
            current = []
        else:
            current.append(row)
    if current:
        single_series.append(current)
    series = [("atlas (chart switching)", [r["delta"] for r in positive], [r["atlas_error"] for r in positive])]
    series += [(f"chart A alone{'' if i == 0 else ' (cont.)'}", [r["delta"] for r in part],
                [r["single_chart_error"] for r in part]) for i, part in enumerate(single_series)]
    if failed:
        series.append(("chart A failed (at 1)", [r["delta"] for r in failed],
                       [1.0] * len(failed)))
    ctx.artifact_text("atlas-vs-single-chart.svg", svg.line_plot(
        series, title=f"Great-circle error after 2 pi R, RK4 with {STEPS} steps", xlabel="closest approach to the pole",
        ylabel="max |X - X_exact|", logx=True, logy=True))
    profile = study["regularity_profile"]
    ctx.artifact_text("regularity-profile.svg", svg.line_plot(
        [("chart A alone: sin^2 theta_A", profile["s"], profile["chart_A"]),
         ("best chart of the atlas", profile["s"], profile["best_chart"])],
        title="Chart regularity along the delta = 1e-3 great circle", xlabel="arclength s",
        ylabel="1/cond(g) = det g / R^4 = sin^2 theta", logy=True, markers=False))
    counts = [row["steps"] for row in meridian["rows"]]
    ctx.artifact_text("meridian-steps.svg", svg.line_plot(
        [(f"chart A alone, seed {MERIDIAN_SEED:g}", counts,
          [ceiling if row["error"] is None else row["error"] for row in meridian["rows"]]),
         ("closest RK4 stage point to a pole", counts, [row["stage_distance"] for row in meridian["rows"]]),
         ("failure radius d* = (L0 h^4)^(1/5)", counts, [row["failure_radius"] for row in meridian["rows"]])],
        title="Seeded meridian in one polar chart: error (failures at 2), stage distance to the poles and d*",
        xlabel="RK4 steps over 2 pi", ylabel="length", logy=True))
    fields = {
        "hypothesis": ("An atlas of charts with exact transitions through the embedding integrates geodesics through "
                       "or near a chart's coordinate singularity with an accuracy that does not depend on how close "
                       "they pass, while a single chart loses accuracy or fails near its singular point. Shown for "
                       "the two-chart polar atlas of the sphere (great circles through and near a pole) and for the "
                       "Monge-plus-polar atlas of the Gaussian bump built from a ChartMap (geodesics through and near "
                       "the apex)."),
        "mathematical_model": ("Atlas: charts X_c with closed-form inverses; u_B = X_B^{-1}(X_A(u_A)), "
                               "v_B = g_B^{-1} J_B^T J_A v_A; regularity lambda_min / lambda_max of g (scale free); "
                               "switch between RK4 steps when the active chart's regularity < 1/4. Sphere: chart A "
                               "X = R(sin t cos p, sin t sin p, cos t), chart B = R_y(pi/2) X_A; regularity sin^2 theta "
                               "= det g / R^4 in each chart and sin^2 theta_A + sin^2 theta_B = 1 + y^2/R^2 >= 1, so the "
                               "better chart always has regularity >= 1/2. Meridian in chart A: angular momentum "
                               "L = sin^2(theta) v_phi (0 on the meridian, sin(delta) on a unit-speed great circle "
                               "passing delta from the pole) starts at a declared seed L0; near a pole the chart is the "
                               "plane's polar chart, so the outcome depends on d/h and L0/h only (d the closest RK4 "
                               "stage point's distance to the pole); a crossing multiplies L by a factor proportional "
                               "to (h/d)^2 and turns nonlinear once L h^2/d^3 reaches order 1, so two crossings fail "
                               "within about d* = (L0 h^4)^(1/5) (exponent from this argument, constant fitted). Graph "
                               "z = f(x, y): "
                               "Monge chart g = I + grad f grad f^T with regularity 1/(1 + |grad f|^2), and the polar "
                               "chart (r, t) pulled back through PolarChart, singular at r = 0."),
        "input_data": [f"unit sphere; great circles with closest approach delta in {list(DELTAS)} to the north pole, "
                       f"azimuth {AZIMUTH}, full length 2 pi", f"RK4, {STEPS} steps (convergence: 200/400/800)",
                       f"meridian (delta = 0) in chart A alone from the declared angular-momentum seed "
                       f"L0 = sin^2(theta0) v_phi0 = {MERIDIAN_SEED:g} at {len(MERIDIAN_STEPS)} step counts "
                       f"({MERIDIAN_STEPS[0]}..{MERIDIAN_STEPS[-1]}); at the other half-decade seeds from "
                       f"{SWEEP_SEEDS[0]:g} to {SWEEP_SEEDS[-1]:g} the step counts whose closest RK4 stage point lies "
                       f"within {SWEEP_BAND:g} d* of a pole; {STEPS} steps also at L0 / {SEED_RATIO}; unseeded at "
                       f"{STEPS} steps",
                       f"switch threshold: regularity < {SWITCH_THRESHOLD}",
                       f"{transitions['points']} + {transitions['dense_points']} seeded sphere points for transitions "
                       f"and covering (seed {transitions['seed']})",
                       f"Gaussian bump h = 0.5, sigma = 1: geodesics from x = -{GRAPH_START} at offset delta in "
                       f"{list(GRAPH_DELTAS)} from the apex, length {GRAPH_LENGTH}, RK4 {GRAPH_STEPS} steps, reference "
                       f"Monge chart alone at {GRAPH_REFINE * GRAPH_STEPS} steps; {graph_transitions['points']} seeded "
                       f"points of the Monge disc of radius {GRAPH_DISC} (seed {graph_transitions['seed']})",
                       "interface under test: ciw.lab.surfaces and ciw.lab.integrators at the source digests recorded "
                       "in provider_runtime_identity.sources"],
        "observation_model": ("Ambient error max_s |X_num(s) - X_ref(s)|: exact great circle cos(s) X0 + sin(s) T0 on "
                              "the sphere, the refined Monge-chart run on the bump."),
        "expected_invariant": ("Atlas error independent of delta and fourth order in the step; transitions exact to "
                               "rounding; single-chart error grows or the integration fails for small delta > 0; on "
                               "the seeded meridian the single chart fails when an RK4 stage point lands within about "
                               "d* of a pole, and on the 400-step grid, far from d*, its error is proportional to the "
                               "seed."),
        "experiment": ("Integrate each geodesic with chart switching and in the singular chart alone; verify "
                       "transitions by round trip, speed preservation and difference Jacobians; convergence at "
                       "delta = 0; scan the step count of the seeded meridian in chart A alone, sweep the seed over "
                       "half decades on the grids near a pole, rerun the 400-step grid at a tenth of the seed, and "
                       "record the natural seed of the unseeded meridian."),
        "numerical_result": (f"sphere atlas through the pole: error {_fmt(through['atlas_error'])}, "
                             f"{through['switches']} switches ({', '.join(through['switch_path'])}); atlas error spread "
                             f"over delta {_fmt(spread)}x; orders {', '.join(_fmt(o) for o in study['orders'])}; single "
                             f"chart failed for {len(failed)} and degraded (>10x atlas) for {len(degraded)} of "
                             f"{len(positive)} delta > 0; meridian in chart A alone from the seed {MERIDIAN_SEED:g}: "
                             f"{STEPS}-step error {_fmt(seeded['error'])} ({SEED_RATIO} times the error from a seed "
                             f"of {seeded['rerun_seed']:g} is {_fmt(SEED_RATIO * seeded['rerun_error'])}), the two "
                             f"crossings amplifying L to {_fmt(seeded['momentum_after_crossings'])}; over "
                             f"{len(MERIDIAN_STEPS)} step counts {len(near)} runs have an RK4 stage point within d* of a "
                             f"pole and {len(meridian_failed)} fail, the largest d/d* of a failure "
                             f"{_fmt(declared_failed_ratio)} and the smallest of any other run "
                             f"{_fmt(declared_kept_ratio)}; the runs beyond d* err {_fmt(min(other_errors))} to "
                             f"{_fmt(max(other_errors))}; across {1 + len(meridian['sweep'])} seeds from "
                             f"{SWEEP_SEEDS[0]:g} to {SWEEP_SEEDS[-1]:g} ({len(swept)} runs, {len(swept_failed)} "
                             f"failures) the largest d/d* of a failure is {_fmt(failed_ratio)} and the smallest of a "
                             f"run that did not fail {_fmt(kept_ratio)}; unseeded, g_12 = "
                             f"{_fmt(natural['g12_at_start'])} and v_phi = {_fmt(natural['v_phi_at_start'])} at the "
                             f"start and |L| <= {_fmt(natural['momentum_before_first_crossing'])} before the first "
                             f"crossing, amplified to {_fmt(natural['max_abs_momentum'])} with error "
                             f"{_fmt(natural['error'])} at {STEPS} steps; transitions round trip "
                             f"{_fmt(transitions['roundtrip'])}, "
                             f"speed {_fmt(transitions['metric_defect'])}, Jacobian "
                             f"{_fmt(transitions['jacobian_defect'])}; covering bound {_fmt(transitions['covering_min'])}. "
                             f"Gaussian-bump atlas: error <= {_fmt(graph_atlas_error)} for every delta (Monge chart "
                             f"alone {_fmt(graph_monge_error)}), polar chart alone failed for {len(graph_failed)} and "
                             f"degraded for {len(graph_degraded)} of {len(graph_positive)} delta > 0; transitions round "
                             f"trip {_fmt(graph_transitions['roundtrip'])}, speed "
                             f"{_fmt(graph_transitions['metric_defect'])}, Jacobian "
                             f"{_fmt(graph_transitions['jacobian_defect'])}."),
        "uncertainty": ("Atlas errors are RK4 truncation errors (order 4); the 400-step sphere error differs from the "
                        "fourth-order predictions of the 200- and 800-step runs by "
                        f"{_fmt(100 * relative_richardson)}%. Single-chart errors for small delta come from the "
                        "unresolved azimuthal rate 1/sin(delta) at a fixed step; the failure steps and the errors for "
                        "delta <= 1e-8 depend on rounding and may differ across platforms. On the meridian the "
                        "single-chart result depends on the step grid and on the angular momentum L = sin^2(theta) "
                        "v_phi that each pole crossing amplifies, so the 400-step success is a property of that grid "
                        f"(closest RK4 stage point {_fmt(seeded['stage_distance'])} from a pole) and of the seed, not "
                        "of the chart. Unseeded, L comes from the rounding of g_12, which depends on the BLAS kernel "
                        "(the BLAS build and the inputs of the OpenBLAS kernel choice are recorded in the runtime "
                        "identity and in meridian-steps.json), so the "
                        f"meridian runs from a declared seed of {MERIDIAN_SEED:g}: the natural seed is at most "
                        f"{NATURAL_SEED_BOUND:g}, {natural_share:g} of the declared one. It moves an error linear in the "
                        f"seed (the {STEPS}-step grid) by at most {natural_share:g} relative, an error quadratic in it "
                        "(grids within a few d*, where perturbing the seed gave exponents up to 2.01) by at most about "
                        f"{2 * natural_share:g}, and the {STEPS}-step error from a seed of {seeded['rerun_seed']:g} by "
                        f"at most {SEED_RATIO * natural_share:g}; that error differs from proportionality by "
                        f"{_fmt(linearity)}. At the smallest swept seeds the bound is a larger share (up to "
                        f"{NATURAL_SEED_BOUND / SWEEP_SEEDS[0]:g} at {SWEEP_SEEDS[0]:g}), so of the swept runs only the "
                        "outcomes are compared. The failure radius d* takes its exponent from the scaling argument and "
                        "its constant 1 from a fit; across the seed sweep the largest d/d* of a failure is "
                        f"{_fmt(failed_ratio)} and the smallest of a run that did not fail {_fmt(kept_ratio)}."),
        "failure_modes_checked": ["geodesic exactly through a pole or the apex", "geodesics 1e-1..1e-12 from a pole",
                                  "chattering between charts (bound 1/2 > threshold 1/4)",
                                  "longitude wrap-around in transition differences",
                                  "RK4 stage points (step points and half steps) landing near a pole on the meridian",
                                  "a failure radius that holds at one seed only (swept over five decades of the seed)",
                                  "rounding-level angular momentum on the unseeded meridian (bounded, and dominated by "
                                  "the declared seed)",
                                  "nonfinite single-chart states and math domain errors (recorded with their message; "
                                  "any other exception propagates instead of counting as a failure)"],
        "unresolved_assumptions": [
            "An atlas needs an embedding and a closed-form inverse for each chart; atlases are built here for the "
            "sphere (rotated polar charts) and for graph surfaces (Monge chart and its polar ChartMap); intrinsic "
            "charts such as the hyperbolic plane are not covered.",
            "Switching happens between steps; an adaptive integrator would need event location at the threshold.",
            "The recorded minimum regularity of the active chart restates the switch threshold and the covering "
            "bound; it is reported, not checked.",
            f"The failure radius d* = (L0 h^4)^(1/5) takes its exponent from the scaling argument and its constant 1 "
            f"from a fit. It is checked on one path at half-decade seeds from {SWEEP_SEEDS[0]:g} to "
            f"{SWEEP_SEEDS[-1]:g}, beyond {SWEEP_BAND:g} d* at the declared seed only; finer seed steps or other paths "
            "may widen the band of d/d* in which failed and completed runs overlap. Near d* a run that does not fail "
            "can still err by order 1."],
        "recommended_next_task": NEXT_STEPS["T036"],
        # The BLAS kernel sets the unseeded meridian's natural seed; recorded here and in meridian-steps.json only.
        "provider_runtime_identity": dict(builtin_identity((MODULE, CHARTS, GEOMETRY, DOC, CORE, INTEGRATORS)),
                                          blas_kernel=natural["blas_kernel"]),
    }
    findings = [
        finding("Atlas integration of the great circle through the north pole matches the exact great circle",
                "numerical", _sig(through["atlas_error"], 3),
                {"checks": [_check("max |X - X_exact| over 2 pi, RK4 400 steps", through["atlas_error"], 1e-6,
                                   kind="analytic"),
                            _check("chart switches along the path", through["switches"], 1, comparison="ge",
                                   kind="exact_arithmetic")]},
                unit="length", uncertainty={"kind": "truncation_bound", "value": _sig(study["richardson_spread"], 2),
                                            "basis": "Richardson: largest distance of the 400-step error from the "
                                                     "fourth-order predictions e(200)/16 and 16 e(800)"},
                tolerance={"abs": 1e-8, "rel": 0.05}),
        finding("Atlas accuracy is independent of the distance of closest approach to a pole", "numerical",
                _sig(spread, 3),
                {"checks": [_check("max / min atlas error over delta", spread, 1.5, comparison="le", kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": _sig(2 * relative_richardson * spread, 2),
                             "basis": "each atlas error carries the relative Richardson spread of the delta = 0 run; "
                                      "a ratio of two errors carries at most about twice it"},
                tolerance={"abs": 0.1, "rel": 0.0}),
        finding("Atlas integration keeps fourth-order convergence across chart switches", "numerical",
                [_sig(order, 4) for order in study["orders"]],
                {"checks": [_check("max |order - 4| (200/400/800 steps)", order_defect, 0.25, kind="self_convergence")]},
                uncertainty={"kind": "fit", "value": _sig(abs(study["orders"][0] - study["orders"][1]) / 2, 2),
                             "basis": "half the difference between the two pairwise orders (pre-asymptotic drift)"},
                tolerance={"abs": 0.05, "rel": 0.0}),
        finding("Chart transitions are exact: round trip, speed preservation and difference Jacobian", "numerical",
                {"roundtrip": transitions["roundtrip"], "speed": transitions["metric_defect"],
                 "jacobian": transitions["jacobian_defect"]},
                {"checks": [_check("A -> B -> A round trip, max |X' - X| / R", transitions["roundtrip"], 1e-13,
                                   kind="invariant"),
                            _check("|g_B(v_B, v_B) - g_A(v_A, v_A)| / g_A(v_A, v_A)", transitions["metric_defect"],
                                   1e-12, kind="invariant"),
                            _check("pushforward vs central difference of the point transition (h = 1e-6)",
                                   transitions["jacobian_defect"], 1e-7, kind="self_convergence")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-9,
                             "basis": "the Jacobian comparison carries the central-difference error at h = 1e-6 "
                                      "(h^2 truncation ~1e-12, eps/h rounding ~2e-10); round trip and speed are "
                                      "rounding level (~1e-15)"},
                tolerance={"abs": 1e-7, "rel": 0.0}),
        finding("The better chart of the atlas always has det g / R^4 >= 1/2, so switching never chatters",
                "mathematical", {"covering_min": _sig(transitions["covering_min"], 6), "min_det_after_switch":
                                 _sig(min_after, 6), "min_active_det": _sig(min_active, 6)},
                {"derivation": "sin^2 theta_A + sin^2 theta_B = 1 + y^2 / R^2 >= 1",
                 "checks": [_check(f"min over {transitions['points'] + transitions['dense_points']} sphere points of "
                                   "the better chart's regularity, evaluated through each chart's inverse and metric",
                                   transitions["covering_min"], 0.5 - 1e-12, comparison="ge", kind="analytic"),
                            _check("min regularity right after a switch", min_after, 0.5 - 1e-12, comparison="ge",
                                   kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-15,
                             "basis": "regularity from the chart inverse and the eigenvalues of g at sampled points; "
                                      "the bound itself is analytic"},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("A single polar chart fails or loses accuracy on great circles passing near its pole", "numerical",
                {"cases": len(positive), "failed": len(failed), "degraded": len(degraded)},
                {"checks": [_check("delta > 0 cases where chart A alone fails or errs > 10x the atlas", len(degraded),
                                   5, comparison="ge", kind="analytic"),
                            _check("delta > 0 cases where chart A alone fails outright", len(failed), 1,
                                   comparison="ge", kind="analytic")]},
                uncertainty={"kind": "reference_error", "value": 1,
                             "basis": "counts at a fixed 400-step grid; which small-delta runs fail or degrade depends "
                                      "on rounding and may move by one case across platforms"},
                tolerance={"abs": 1, "rel": 0.0},
                counterexample={"statement": ("Fixed-step RK4 in a single polar chart integrates every great circle "
                                              "as accurately as a chart-switching atlas at the same step count"),
                                "witness": {"log10_delta_failed": [round(math.log10(r["delta"])) for r in failed],
                                            "log10_delta_0.1_single_error": None
                                            if runs[0.1]["single_chart_error"] is None
                                            else _log10(runs[0.1]["single_chart_error"]),
                                            "log10_delta_0.1_atlas_error": _log10(runs[0.1]["atlas_error"])}}),
        finding(f"From a declared angular-momentum seed of {MERIDIAN_SEED:g}, chart A alone crosses both poles of the "
                f"meridian at {STEPS} RK4 steps to within 1e-6", "numerical",
                {"error": _sig(seeded["error"], 6), "closest_stage_to_pole": _sig(seeded["stage_distance"], 6)},
                {"checks": [_check(f"chart A alone, seed {MERIDIAN_SEED:g}, {STEPS} steps: max |X - X_exact| against the "
                                   "great circle of the seeded initial state", seeded["error"], 1e-6, comparison="le",
                                   kind="analytic")]},
                unit="length", uncertainty={"kind": "roundoff", "value": _sig(seeded["error"] * natural_share, 2),
                                            "basis": "the error is linear in the seed on this grid, so the natural seed "
                                                     f"(at most {NATURAL_SEED_BOUND:g}, {natural_share:g} of the declared "
                                                     f"one) moves it by at most {natural_share:g} relative. Across the "
                                                     "SkylakeX, Haswell and Sandybridge OpenBLAS kernels it agreed to "
                                                     "2.0e-7 relative, since their natural seeds (at most 1.1e-18) lie "
                                                     "far below the bound; the regression tolerance is the derived "
                                                     f"bound, {natural_share:g} relative, so any kernel within it "
                                                     "reproduces the value"},
                tolerance={"abs": 0.0, "rel": natural_share},
                counterexample={"statement": "Fixed-step integration in a single polar chart across its pole always fails",
                                "witness": {"steps": STEPS, "seed": MERIDIAN_SEED, "error": _sig(seeded["error"], 6)}}),
        finding(f"At {STEPS} RK4 steps the meridian error of chart A alone is proportional to the declared seed: "
                f"{SEED_RATIO} times the error from a seed of {seeded['rerun_seed']:g} matches the error from "
                f"{MERIDIAN_SEED:g} to within 1e-2", "numerical",
                {"seed_ratio": SEED_RATIO, "error_ratio": _sig(seeded["error"] / seeded["rerun_error"], 6)},
                {"checks": [_check(f"|e({MERIDIAN_SEED:g}) / ({SEED_RATIO} e({seeded['rerun_seed']:g})) - 1| at {STEPS} "
                                   "steps (first order in the seed)", linearity, 1e-2, comparison="le",
                                   kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": _sig((1 + SEED_RATIO) * natural_share, 2),
                             "basis": f"the natural seed (at most {NATURAL_SEED_BOUND:g}) is at most "
                                      f"{SEED_RATIO * natural_share:g} of the smaller seed and {natural_share:g} of the "
                                      "larger, so it moves the ratio of the two errors, each linear in its seed, by at "
                                      f"most {(1 + SEED_RATIO) * natural_share:g} relative. Across the SkylakeX, Haswell "
                                      "and Sandybridge OpenBLAS kernels the ratio agreed to 1.7e-6 relative, since "
                                      "their natural seeds (at most 1.1e-18) lie far below the bound; the regression "
                                      "tolerance is the derived bound, so any kernel within it reproduces the value"},
                tolerance={"abs": 0.0, "rel": (1 + SEED_RATIO) * natural_share}),
        finding(f"Across declared seeds from {SWEEP_SEEDS[0]:g} to {SWEEP_SEEDS[-1]:g}, chart A alone fails on the "
                f"meridian whenever an RK4 stage point lands within {BAND_INNER:g} d* of a pole and never beyond "
                f"{BAND_OUTER:g} d*, d* = (L0 h^4)^(1/5), but no single multiple of d* separates the outcomes at every "
                f"seed ({MERIDIAN_STEPS[0]} to {MERIDIAN_STEPS[-1]} RK4 steps at {MERIDIAN_SEED:g}, the step counts "
                f"within {SWEEP_BAND:g} d* at the other half-decade seeds)", "numerical",
                {"step_counts": len(MERIDIAN_STEPS), "seeds": 1 + len(meridian["sweep"]), "runs": len(swept),
                 "failed": len(swept_failed), "largest_ratio_failed": _sig(failed_ratio, 6),
                 "smallest_ratio_not_failed": _sig(kept_ratio, 6),
                 "at_declared_seed": {"within_d_star": len(near), "failed": len(meridian_failed),
                                      "largest_ratio_failed": _sig(declared_failed_ratio, 6),
                                      "smallest_ratio_not_failed": _sig(declared_kept_ratio, 6)},
                 "failed_at_declared_seed_not_at_smaller_seed": moved},
                {"checks": [_check(f"swept runs with an RK4 stage point within {BAND_INNER:g} d* of a pole that did not "
                                   "fail (d*: exponent from the scaling argument, constant 1 fitted)",
                                   len(inner_kept), 0, comparison="le", kind="analytic"),
                            _check(f"swept runs with no RK4 stage point within {BAND_OUTER:g} d* of a pole that failed",
                                   len(outer_failed), 0, comparison="le", kind="analytic"),
                            _check("smallest d/d* of a swept run that did not fail minus the largest of a failure "
                                   "(negative when the outcomes overlap)", kept_ratio - failed_ratio, 0.0,
                                   comparison="signed_le", kind="analytic"),
                            _check(f"step counts with an RK4 stage point within d* of a pole at {MERIDIAN_SEED:g}",
                                   len(near), 1, comparison="ge", kind="exact_arithmetic"),
                            _check(f"step counts that fail at {MERIDIAN_SEED:g} but not at a smaller swept seed (d* "
                                   "shrinks with the seed)", len(moved), 1, comparison="ge",
                                   kind="analytic")]},
                uncertainty={"kind": "fit", "value": _sig(failed_ratio - kept_ratio, 3),
                             "basis": "the constant of d* is fitted, and failed and completed runs overlap "
                                      "over this width in d/d*. Stage distances, d* and their ratios are exact "
                                      "arithmetic on the step grid, and the outcomes are set by the declared seed: "
                                      f"moving every swept seed by +-{NATURAL_SEED_BOUND:g}, the natural seed's bound, "
                                      f"changed no outcome, even at {SWEEP_SEEDS[0]:g} where the bound is "
                                      f"{NATURAL_SEED_BOUND / SWEEP_SEEDS[0]:g} of the seed, and across the SkylakeX, "
                                      "Haswell and Sandybridge OpenBLAS kernels they were identical. The swept runs' "
                                      "errors are recorded, not compared: at the smallest seeds the natural seed is a "
                                      "larger share, and they spread by 1.2e-3 relative across those kernels. The regression "
                                      f"tolerance, {natural_share:g} relative, is the natural seed's largest share of the "
                                      f"{STEPS}-step error in the witness, which is linear in the seed; the ratios, "
                                      "counts and step lists must match"},
                tolerance={"abs": 0.0, "rel": natural_share},
                counterexample={"statement": ("A single-chart integration that crosses a pole accurately at one step "
                                              "count stays accurate at nearby step counts"),
                                "witness": {"failed_steps": meridian_failed, f"error_at_{STEPS}": _sig(seeded["error"], 6)}}),
        finding(f"Without a declared seed, rounding seeds at most {NATURAL_SEED_BOUND:g} of angular momentum on the "
                f"meridian before its first pole crossing ({STEPS} RK4 steps)", "numerical",
                {"g12_at_start": _sig(natural["g12_at_start"], 3), "v_phi_at_start": _sig(natural["v_phi_at_start"], 3),
                 "momentum_before_first_crossing": _sig(natural["momentum_before_first_crossing"], 3)},
                {"checks": [_check("|g_12| at the start (0 in exact arithmetic)", abs(natural["g12_at_start"]),
                                   NATURAL_SEED_BOUND, comparison="le", kind="analytic"),
                            _check("max |sin^2(theta) v_phi| over the step points before the first pole crossing (0 "
                                   "in exact arithmetic on the meridian)", natural["momentum_before_first_crossing"],
                                   NATURAL_SEED_BOUND, comparison="le", kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": NATURAL_SEED_BOUND,
                             "basis": "the values are rounding residues of quantities that vanish exactly (g_12 = "
                                      "x_theta . x_phi sums two products of size up to R^2/2) and depend on the BLAS "
                                      "kernel, whose selection the runtime identity records. Across the SkylakeX, "
                                      "Haswell and Sandybridge kernels they spread by up to 1.1e-18 (v_phi 0 or "
                                      "1.1e-18 at the start; momentum 5.8e-35 to 7.8e-19); the regression tolerance "
                                      f"is the claimed bound {NATURAL_SEED_BOUND:g}, about 90 times that spread"},
                tolerance={"abs": NATURAL_SEED_BOUND, "rel": 0.0}),
        finding("The Monge-plus-polar atlas of a graph surface has exact transitions and covers the plane with "
                "regularity above the analytic Monge bound", "numerical",
                {"roundtrip": graph_transitions["roundtrip"], "speed": graph_transitions["metric_defect"],
                 "jacobian": graph_transitions["jacobian_defect"],
                 "covering_min": _sig(graph_transitions["covering_min"], 6),
                 "monge_bound": _sig(graph_transitions["monge_regularity_bound"], 6)},
                {"derivation": "Monge chart g = I + grad f grad f^T: regularity 1 / (1 + |grad f|^2), and |grad f| "
                               "<= h / (sqrt(e) sigma) for the Gaussian bump",
                 "checks": [_check("polar -> Monge -> polar round trip, max |X' - X|", graph_transitions["roundtrip"],
                                   1e-13, kind="invariant"),
                            _check("relative speed change under the velocity pushforward",
                                   graph_transitions["metric_defect"], 1e-12, kind="invariant"),
                            _check("pushforward vs central difference of the point transition (h = 1e-6)",
                                   graph_transitions["jacobian_defect"], 1e-7, kind="self_convergence"),
                            _check("min over sampled points of the better chart's regularity minus the Monge bound",
                                   graph_transitions["covering_min"] - graph_transitions["monge_regularity_bound"],
                                   -1e-12, comparison="signed_ge", kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-9,
                             "basis": "the Jacobian comparison carries the central-difference error at h = 1e-6; "
                                      "round trip, speed and regularities are rounding level"},
                tolerance={"abs": 1e-7, "rel": 0.0}),
        finding("The graph atlas integrates geodesics through and near the Gaussian-bump apex, where its polar chart "
                "alone fails or loses accuracy", "numerical",
                {"atlas_error": _sig(graph_atlas_error, 3), "monge_alone_error": _sig(graph_monge_error, 3),
                 "min_switches": graph_switches, "polar_failed": len(graph_failed),
                 "polar_degraded": len(graph_degraded), "cases": len(graph_positive)},
                {"checks": [_check(f"max over delta of |X_atlas - X_ref| (RK4 {GRAPH_STEPS} steps from the polar chart; "
                                   f"reference Monge chart alone, {GRAPH_REFINE * GRAPH_STEPS} steps)",
                                   graph_atlas_error, 1e-8, comparison="le", kind="cross_implementation"),
                            _check("chart switches per run (polar -> Monge)", graph_switches, 1, comparison="ge",
                                   kind="exact_arithmetic"),
                            _check("delta > 0 cases where the polar chart alone fails or errs > 10x the atlas",
                                   len(graph_degraded), len(graph_positive) - 1, comparison="ge", kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": _sig(graph_monge_error / GRAPH_REFINE ** 4, 2),
                             "basis": "fourth-order error of the refined Monge reference, estimated as the "
                                      f"{GRAPH_STEPS}-step Monge error / {GRAPH_REFINE}^4"},
                tolerance={"abs": 1e-9, "rel": 0.1}),
        finding("Chart-switching geodesic integration is ready for tool paths over physical parts", "industrial_readiness",
                None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T037
def approaches() -> list:
    """Declared approaches; ``polar`` is set from knowledge of each chart (loops are geodesic-circle preimages)."""
    return [Approach("sphere-north-pole", Sphere(1.0), (0.0, 0.0), True),
            Approach("plane-polar-origin", Reparametrized(Plane(), PolarChart()), (0.0, 0.0), True),
            Approach("cone-apex", Cone(math.pi / 6), (0.0, 0.0), True),
            Approach("power-graph-apex", PowerGraph(1.0, 1.5), (0.0, 0.0), False),
            Approach("power-graph-1.8-apex", PowerGraph(1.0, 1.8), (0.0, 0.0), False),
            Approach("hyperbolic-boundary", HyperbolicPlane(1.0), (0.2, 0.0), False, angle=math.pi / 2),
            Approach("conformal-0.9-boundary", ConformalHalfPlane(0.9), (0.2, 0.0), False, angle=math.pi / 2),
            Approach("plane-cube-root-chart", Reparametrized(Plane(), CubeRootChart()), (0.0, 0.3), False),
            Approach("saddle-origin", Saddle(1.0), (0.0, 0.0), False),
            Approach("sphere-equator", Sphere(1.0), (math.pi / 2, 0.3), False)]


# True type of each declared approach. The plane in the cube-root chart has a
# removable coordinate singularity (cond g -> infinity at finite distance with
# K = 0); rule 4 of the scan leaves every finite-distance metric blow-up with
# bounded K unclassified, so the scan misses it. T037 counts that miss and
# records it as a counterexample rather than expecting it.
TRUE_CLASS = {"sphere-north-pole": "coordinate_singularity", "plane-polar-origin": "coordinate_singularity",
              "cone-apex": "conical_singularity", "power-graph-apex": "curvature_singularity",
              "power-graph-1.8-apex": "curvature_singularity",
              "hyperbolic-boundary": "infinite_distance_boundary",
              "conformal-0.9-boundary": "curvature_singularity", "plane-cube-root-chart": "coordinate_singularity",
              "saddle-origin": "regular", "sphere-equator": "regular"}
RULE4_MISS = "plane-cube-root-chart"

# Cases just beyond each detection threshold, with their true type: the scan
# is expected to misclassify them, and T037 records that as a counterexample.
SMALL_DEFICIT = 5e-7  # 1 - sin(alpha) of the nearly flat cone


def limit_approaches() -> dict:
    return {
        "power-graph-1.99-apex": (Approach("power-graph-1.99-apex", PowerGraph(1.0, 1.99), (0.0, 0.0), False),
                                  "curvature_singularity"),
        "conformal-0.9995-boundary": (Approach("conformal-0.9995-boundary", ConformalHalfPlane(0.9995), (0.2, 0.0),
                                               False, angle=math.pi / 2), "curvature_singularity"),
        "cone-small-deficit-apex": (Approach("cone-small-deficit-apex", Cone(math.asin(1.0 - SMALL_DEFICIT)),
                                             (0.0, 0.0), True), "conical_singularity"),
    }


def refusal_cases() -> dict:
    """Guard and core-check outcomes computed from the metric and curvature: name -> (expected, observed)."""
    sphere, graph = Sphere(1.0), PowerGraph(1.0, 1.5)
    return {
        "sphere theta = 3e-5 (guard)": ("degenerate_metric", refusal_code(require_regular, sphere, np.array([3e-5, 0.3]))),
        "sphere theta = 1e-7 (core Surface.check)": ("degenerate_metric",
                                                     refusal_code(sphere.check, np.array([1e-7, 0.3]))),
        "cone apex r = 0 (guard)": ("degenerate_metric", refusal_code(require_regular, Cone(), np.array([0.0, 0.3]))),
        "power graph rho = 1e-6 (guard)": ("curvature_blowup", refusal_code(require_regular, graph, np.array([1e-6, 0.0]))),
        "power graph p = 1.8 at rho = 1e-8, |K| below the bound (guard)": (
            "accepted", refusal_code(require_regular, PowerGraph(1.0, 1.8), np.array([1e-8, 0.0]))),
        "hyperbolic y = -1 (guard, core outside_chart propagated)": (
            "outside_chart", refusal_code(require_regular, HyperbolicPlane(), np.array([0.0, -1.0]))),
        "nonfinite coordinates (guard)": ("nonfinite_point", refusal_code(require_regular, sphere, np.array([math.nan, 0.0]))),
        "sphere equator (guard)": ("accepted", refusal_code(require_regular, sphere, np.array([math.pi / 2, 0.3]))),
    }


def declared_refusal_codes() -> dict:
    """Codes that singular surfaces raise themselves at their apex: author labels, recorded but not checked."""
    return {"cone curvature at r = 0": refusal_code(Cone().gaussian_curvature, np.array([0.0, 0.3])),
            "power graph metric at rho = 0": refusal_code(PowerGraph(1.0, 1.5).metric, np.array([0.0, 0.0]))}


def propagated_refusal_cases() -> dict:
    """A surface's own apex refusal passed through the pointwise guard: name -> (expected, observed).

    The guard must neither swallow nor re-code a refusal raised by the surface.
    """
    return {"power graph apex through the guard": (
        "curvature_singularity", refusal_code(require_regular, PowerGraph(1.0, 1.5), np.array([0.0, 0.0])))}


def core_check_leniency() -> dict:
    """Largest metric condition number the core Surface.check accepts along the sphere pole approach."""
    sphere = Sphere(1.0)
    accepted = []
    for theta in [10.0 ** (-k / 4) for k in range(4, 33)]:
        u = np.array([theta, 0.3])
        if refusal_code(sphere.check, u) == "accepted":
            g = sphere.metric(u)
            accepted.append((float(g[0, 0] / g[1, 1]), theta))
    condition, theta = max(accepted)
    return {"max_accepted_condition": condition, "at_theta": theta}


@task("T037", changed_files=(MODULE, CHARTS, GEOMETRY, DOC),
      regression_tests=(f"{TESTS}::test_singularity_scans_classify_every_case",
                        f"{TESTS}::test_singularity_detection_limits",
                        f"{TESTS}::test_singularity_refusal_codes", f"{TESTS}::test_t037_report",
                        f"{TESTS}::test_next_steps_do_not_hand_work_to_tasks_that_did_not_deliver_it"))
def coordinate_singularities(ctx):
    scans = {a.name: scan(a) for a in approaches()}
    classes = {name: result["classification"] for name, result in scans.items()}
    misclassified = sorted(name for name in classes if classes[name] != TRUE_CLASS[name])
    unexpected = sorted(set(misclassified) - {RULE4_MISS})
    correct = len(classes) - len(misclassified)
    false_positives = sorted(name for name in classes if TRUE_CLASS[name] == "regular" and classes[name] != "regular")
    cube = scans[RULE4_MISS]
    limits = {name: (scan(approach), truth) for name, (approach, truth) in limit_approaches().items()}
    limit_classes = {name: {"true": truth, "observed": result["classification"]}
                     for name, (result, truth) in limits.items()}
    missed = sorted(name for name, row in limit_classes.items() if row["observed"] != row["true"])
    cartesian_pole = scan(Approach("sphere-north-pole-cartesian-loops", Sphere(1.0), (0.0, 0.3), False))
    refusals, propagated, declared = refusal_cases(), propagated_refusal_cases(), declared_refusal_codes()
    leniency = core_check_leniency()
    ctx.artifact_json("singularity-scans.json", {
        "thresholds": {"fit_window": FIT_WINDOW, "fit_residual": FIT_RESIDUAL, "curvature_blowup": CURVATURE_BLOWUP,
                       "degeneracy": DEGENERACY, "divergence_tolerance": DIVERGENCE_TOLERANCE,
                       "conical_tolerance": CONICAL_TOLERANCE},
        "true_class": TRUE_CLASS, "scans": scans,
        "detection_limits": {name: dict(limit_classes[name], scan=result) for name, (result, _) in limits.items()},
        "approach_loop_dependence": cartesian_pole})
    ctx.artifact_json("refusals.json", {"computed": {name: {"expected": e, "observed": o}
                                                     for name, (e, o) in refusals.items()},
                                        "propagated_by_guard": {name: {"expected": e, "observed": o}
                                                                for name, (e, o) in propagated.items()},
                                        "declared_by_surface": declared,
                                        "core_check_leniency": leniency})
    r = scans["sphere-north-pole"]["distances"]
    ctx.artifact_text("singularity-scan.svg", svg.line_plot(
        [("sphere pole: cond(g)", r, scans["sphere-north-pole"]["table"]["condition"]),
         ("sphere pole: |K|", r, [abs(v) for v in scans["sphere-north-pole"]["table"]["curvature"]]),
         ("cone apex: cond(g)", r, scans["cone-apex"]["table"]["condition"]),
         ("r^(3/2) graph: |K|", r, [abs(v) for v in scans["power-graph-apex"]["table"]["curvature"]]),
         ("r^(3/2) graph: max|Gamma|", r, scans["power-graph-apex"]["table"]["christoffel"]),
         ("sphere pole: max|Gamma|", r, scans["sphere-north-pole"]["table"]["christoffel"])],
        title="Approach scans: coordinate versus curvature singularities", xlabel="distance parameter r",
        ylabel="value", logx=True, logy=True))
    pole, cone, polar = scans["sphere-north-pole"], scans["cone-apex"], scans["plane-polar-origin"]
    graph, boundary, conformal = scans["power-graph-apex"], scans["hyperbolic-boundary"], scans["conformal-0.9-boundary"]
    e_pole, e_graph = pole["exponents"], graph["exponents"]
    atlas = SphereAtlas(1.0)
    pole_in_b = atlas.regularity("B", atlas.to_chart("B", np.array([0.0, 0.0, 1.0])))
    rmin = min(graph["distances"])
    graph_prefactor = abs(graph["table"]["curvature"][-1]) * rmin
    graph_expected = 1.125 * PowerGraph(1.0, 1.5).c ** 2
    signature = max(abs(polar["exponents"][k] - cone["exponents"][k]) for k in ("det", "condition", "christoffel"))
    cone_obj = Cone(math.pi / 6)
    slow = limits["power-graph-1.99-apex"][0]["exponents"]["curvature"]
    near_critical = limits["conformal-0.9995-boundary"][0]["exponents"]["radial_speed"]
    small_ratio = limits["cone-small-deficit-apex"][0]["circumference_ratio"]
    fields = {
        "hypothesis": ("A scan into a candidate point, along approach loops that are preimages of geodesic circles, "
                       "separates coordinate singularities (det g -> 0 or cond g -> infinity with bounded K and "
                       "circumference ratio 1) from conical points (circumference deficit above 1e-6), curvature "
                       "singularities (|K| ~ r^a with a <= -0.05) and boundaries at infinite distance (radial speed "
                       "~ r^b with b <= -1 + 1e-3); beyond these detection limits it misclassifies, and it leaves a "
                       "metric blow-up at finite distance with bounded K unclassified even when that is a removable "
                       "coordinate singularity. A pointwise guard refuses points whose metric condition number or |K| "
                       "exceeds its declared bounds, with codes."),
        "mathematical_model": ("Fit power laws r^a on r in [1e-8, 1e-5] for det g, cond(g), max|Gamma|, |K| and the "
                               "radial speed |dX/dr|; a fit is clean when its max log residual is <= 0.05. Radial "
                               "distance int r^b dr diverges iff b <= -1; circumference ratio C(r) / (2 pi rho(r)) -> 1 "
                               "at smooth points and sin(alpha) at a cone apex."),
        "input_data": [f"{len(TRUE_CLASS)} declared approaches: " + ", ".join(TRUE_CLASS),
                       f"{len(limit_classes)} detection-limit cases: " + ", ".join(limit_classes),
                       "the sphere pole with Cartesian loops in chart A", "29 log-spaced distances 1e-1..1e-8",
                       "interface under test: ciw.lab.surfaces at the source digest recorded in "
                       "provider_runtime_identity.sources"],
        "observation_model": ("Pointwise invariants of the chart; loop length by 64-point trapezoid; radial length "
                              "by 16-point Gauss-Legendre."),
        "expected_invariant": ("Sphere pole: det ~ r^2, cond ~ r^-2, Gamma ~ r^-1, K ~ r^0; r^(3/2) graph: K ~ r^-1 "
                               "with det g -> 1; cone: ratio sin(alpha) = 1/2; hyperbolic: radial speed ~ y^-1."),
        "experiment": ("Scan, fit and classify every declared approach and every detection-limit case; repeat the "
                       "sphere pole with Cartesian loops; exercise the pointwise guard and the core check."),
        "numerical_result": (f"{correct}/{len(classes)} declared approaches classified as their true type (missed: "
                             f"{', '.join(misclassified) or 'none'}, read as "
                             f"{', '.join(classes[name] for name in misclassified) or 'n/a'}); "
                             f"{len(missed)}/{len(limit_classes)} detection-limit cases misclassified as "
                             f"predicted; sphere pole exponents det {_fmt(e_pole['det'])}, cond "
                             f"{_fmt(e_pole['condition'])}, Gamma {_fmt(e_pole['christoffel'])}, K "
                             f"{_fmt(e_pole['curvature'])}; r^(3/2) graph K exponent {e_graph['curvature']:.7g}, K r -> "
                             f"{graph_prefactor:.6g} (9/8 expected); cone circumference ratio "
                             f"{_fmt(cone['circumference_ratio'])}; hyperbolic radial-speed exponent "
                             f"{_fmt(boundary['exponents']['radial_speed'])}; Cartesian loops at the sphere pole give "
                             f"ratio {_fmt(cartesian_pole['circumference_ratio'])} "
                             f"({cartesian_pole['classification']}); core check accepts cond(g) up to "
                             f"{_fmt(leniency['max_accepted_condition'])}; "
                             f"{sum(e == o for e, o in refusals.values())}/{len(refusals)} computed refusal codes as "
                             f"expected, and {sum(e == o for e, o in propagated.values())}/{len(propagated)} "
                             f"surface-raised apex code propagated unchanged by the guard."),
        "uncertainty": ("Fits on r <= 1e-5 keep the leading smooth correction of each quantity: for z = r^(3/2), "
                        "K = (9/8) r^-1 (1 + 9r/4)^-2, so the K exponent is -1 - O(r) (observed "
                        f"{e_graph['curvature']:.7g}) and the det exponent O(r) ({_fmt(e_graph['det'])}); exact power "
                        "laws (sphere pole, cone, hyperbolic boundary) fit to rounding. The circumference ratio at "
                        "r = 1e-8 is exact to rounding for these examples."),
        "failure_modes_checked": ["false positive at regular points (saddle origin, sphere equator)",
                                  "coordinate singularity mistaken for a conical point (identical pointwise signature)",
                                  "Christoffel blow-up without curvature blow-up and vice versa",
                                  "metric blow-up at an infinite-distance boundary",
                                  "slow curvature blow-up (K ~ r^-0.4) and a finite-distance boundary with K blow-up",
                                  "removable metric blow-up at finite distance (missed: left unclassified, recorded "
                                  "as a counterexample)",
                                  "cases just beyond each detection threshold", "approach loops that are not "
                                  "geodesic-circle preimages", "apex evaluation (refused)"],
        "unresolved_assumptions": [
            "The rules assume rotationally symmetric approaches whose radial chart lines are geodesics and whose loops "
            "are preimages of geodesic circles; Approach.polar encodes that knowledge of the chart and is chosen by "
            "the caller, and a wrong choice misclassifies.",
            "Detection limits: |K| blow-up slower than r^0.05 over the fit window, radial-speed exponents within 1e-3 "
            "of -1 and circumference deficits below 1e-6 are misclassified; logarithmic corrections are read through "
            "their fitted exponent.",
            "A metric blow-up at finite distance with bounded K is left unclassified, so the removable coordinate "
            "singularity of the plane in the cube-root chart is missed (counted as misclassified); telling it apart "
            "from a genuine singularity would need a removability test, for example regularity of the metric in "
            "radial arclength coordinates.",
            "Non-rotationally-symmetric singular points (edges, cusps along curves) are not covered.",
            "The pointwise guard reports a conical point and a coordinate singularity alike as degenerate_metric, "
            "accepts curvature below its declared bound however fast it grows, and does not flag the hyperbolic "
            "boundary at any finite y (cond g = 1, K = -1); only the scan separates them.",
            "The core Surface.check threshold is scale free (det(g / tr g) = det g / (tr g)^2 <= 1e-12, roughly "
            "cond g > 1e12) and is left unchanged; a tighter condition bound (1e8, as in require_regular) is "
            "proposed as a core change.",
            "The apex codes that the cone and the power graph raise themselves (conical_singularity, "
            "curvature_singularity) are author labels, recorded in refusals.json but not counted as detections."],
        "recommended_next_task": NEXT_STEPS["T037"],
        "provider_runtime_identity": builtin_identity((MODULE, CHARTS, GEOMETRY, DOC, CORE)),
    }
    findings = [
        finding("The sphere pole in the polar chart is a coordinate singularity: det g -> 0 while K stays 1",
                "numerical", {"exponents": {k: _sig(v, 6) for k, v in e_pole.items()},
                              "classification": classes["sphere-north-pole"],
                              "regularity_in_chart_B": _sig(pole_in_b, 6)},
                {"checks": [_check("|det exponent - 2|", e_pole["det"] - 2, 1e-3, kind="analytic"),
                            _check("|cond exponent + 2|", e_pole["condition"] + 2, 1e-3, kind="analytic"),
                            _check("|Gamma exponent + 1|", e_pole["christoffel"] + 1, 1e-3, kind="analytic"),
                            _check("|K exponent|", e_pole["curvature"], 1e-3, kind="analytic"),
                            _check("chart B det g / R^4 at the pole of chart A", pole_in_b, 1 - 1e-12, comparison="ge",
                                   kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-9, "basis": "least-squares fit of exact power laws"},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("The scan classifies 9 of the 10 declared approaches as their true type, with no false positive at "
                "regular points", "computational_pipeline", {"classes": classes, "correct": correct},
                {"checks": [_check("declared approaches classified as their true type", correct, 9, comparison="ge",
                                   kind="exact_arithmetic"),
                            _check(f"declared approaches misclassified other than {RULE4_MISS}", len(unexpected), 0,
                                   kind="exact_arithmetic"),
                            _check("regular approaches classified as singular", len(false_positives), 0,
                                   kind="exact_arithmetic")]},
                uncertainty={"kind": "reference_error", "value": 0, "basis": "exact classification outcome per scan"},
                tolerance={"abs": 0, "rel": 0}),
        finding("The scan misses the removable coordinate singularity of the plane in the cube-root chart", "numerical",
                {"true": TRUE_CLASS[RULE4_MISS], "observed": cube["classification"],
                 "det_exponent": _sig(cube["exponents"]["det"], 6),
                 "condition_exponent": _sig(cube["exponents"]["condition"], 6),
                 "radial_speed_exponent": _sig(cube["exponents"]["radial_speed"], 6),
                 "max_abs_curvature": max(abs(k) for k in cube["table"]["curvature"])},
                {"checks": [_check("classified as other than its true type, coordinate_singularity (1 = yes)",
                                   int(cube["classification"] != TRUE_CLASS[RULE4_MISS]), 1, comparison="ge",
                                   kind="exact_arithmetic"),
                            _check("|cond exponent + 4/3| (cond g -> infinity)", cube["exponents"]["condition"] + 4 / 3,
                                   1e-6, kind="analytic"),
                            _check("|radial-speed exponent + 2/3| (the point is at finite distance)",
                                   cube["exponents"]["radial_speed"] + 2 / 3, 1e-6, kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-6,
                             "basis": "least-squares fit on r <= 1e-5; the radial speed carries an O(r^(4/3)) "
                                      "correction from the unscaled coordinate"},
                tolerance={"abs": 1e-6, "rel": 0.0},
                counterexample={"statement": ("The scan detects every coordinate singularity, that is every det g -> 0 "
                                              "or cond g -> infinity at finite distance with bounded K"),
                                "witness": {"approach": RULE4_MISS, "true": TRUE_CLASS[RULE4_MISS],
                                            "observed": cube["classification"]}}),
        finding("The graph z = r^(3/2) has a curvature singularity although its Monge metric is regular", "numerical",
                {"curvature_exponent": _sig(e_graph["curvature"], 6), "det_exponent": _sig(e_graph["det"], 3),
                 "christoffel_exponent": _sig(e_graph["christoffel"], 3), "K_times_r": _sig(graph_prefactor, 8)},
                {"checks": [_check("|K exponent + 1|", e_graph["curvature"] + 1, 1e-3, kind="analytic"),
                            _check("|det exponent|", e_graph["det"], 1e-3, kind="analytic"),
                            _check("|K r - 9 c^2 / 8| at r = 1e-8", graph_prefactor - graph_expected, 1e-6, kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-4,
                             "basis": "O(r) correction (1 + 9r/4)^-2 over the fit window r <= 1e-5"},
                tolerance={"abs": 1e-4, "rel": 1e-6},
                counterexample={"statement": "A nondegenerate metric chart implies bounded Gaussian curvature",
                                "witness": {"surface": "z = r^(3/2)", "det_g_at_1e-8": _sig(graph["table"]["det"][-1], 6),
                                            "K_at_1e-8": _sig(graph["table"]["curvature"][-1], 6)}}),
        finding("Christoffel blow-up and curvature blow-up are independent near singular points", "numerical",
                {"sphere_pole": {"christoffel": _sig(e_pole["christoffel"], 4), "curvature": _sig(e_pole["curvature"], 4)},
                 "power_graph": {"christoffel": _sig(e_graph["christoffel"], 4), "curvature": _sig(e_graph["curvature"], 4)}},
                {"checks": [_check("minus the sphere pole Gamma exponent (blow-up rate)", -e_pole["christoffel"], 0.9,
                                   comparison="ge"),
                            _check("sphere pole |K exponent| (bounded)", e_pole["curvature"], 1e-3),
                            _check("power graph |Gamma exponent| (bounded)", e_graph["christoffel"], 1e-2),
                            _check("minus the power graph K exponent (blow-up rate)", -e_graph["curvature"], 0.9,
                                   comparison="ge")]},
                uncertainty={"kind": "truncation_bound", "value": 1e-4, "basis": "O(r) corrections over r <= 1e-5"},
                tolerance={"abs": 1e-2, "rel": 0.0},
                counterexample={"statement": "Christoffel-symbol blow-up indicates a curvature singularity",
                                "witness": {"sphere_pole_max_gamma_at_1e-8": _sig(pole["table"]["christoffel"][-1]),
                                            "sphere_pole_K": 1.0}}),
        finding("A cone apex and the polar-chart origin share every pointwise exponent; only the circumference ratio "
                "separates them",
                "numerical", {"signature_difference": _sig(signature, 3), "cone_ratio": _sig(cone["circumference_ratio"], 8),
                              "polar_ratio": _sig(polar["circumference_ratio"], 8),
                              "angle_deficit": _sig(cone_obj.angle_deficit(), 8)},
                {"checks": [_check("max exponent difference (det, cond, Gamma)", signature, 1e-6, kind="analytic"),
                            _check("|cone ratio - sin(pi/6)|", cone["circumference_ratio"] - 0.5, 1e-9, kind="analytic"),
                            _check("|polar ratio - 1|", polar["circumference_ratio"] - 1.0, 1e-9, kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "trapezoid and Gauss-Legendre quadrature"},
                tolerance={"abs": 1e-6, "rel": 0.0},
                counterexample={"statement": ("Pointwise metric and curvature invariants distinguish a removable "
                                              "coordinate singularity from a conical point"),
                                "witness": {"removable": "plane in polar chart at r = 0", "conical": "cone alpha = pi/6 apex",
                                            "angle_deficit": _sig(cone_obj.angle_deficit())}}),
        finding("The hyperbolic chart boundary y -> 0 is at infinite distance, not a singular point", "numerical",
                {"det_exponent": _sig(boundary["exponents"]["det"], 6),
                 "radial_speed_exponent": _sig(boundary["exponents"]["radial_speed"], 8)},
                {"checks": [_check("|det exponent + 4|", boundary["exponents"]["det"] + 4, 1e-3, kind="analytic"),
                            _check("|radial-speed exponent + 1| (distance int dy / y diverges)",
                                   boundary["exponents"]["radial_speed"] + 1, 1e-6, kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-9, "basis": "least-squares fit of exact power laws"},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("The boundary of g = y^-1.8 I lies at finite distance and carries a curvature singularity", "numerical",
                {"radial_speed_exponent": _sig(conformal["exponents"]["radial_speed"], 8),
                 "curvature_exponent": _sig(conformal["exponents"]["curvature"], 8),
                 "classification": conformal["classification"]},
                {"checks": [_check("|radial-speed exponent + 0.9| (distance y^0.1 / 0.1 is finite)",
                                   conformal["exponents"]["radial_speed"] + 0.9, 1e-6, kind="analytic"),
                            _check("|K exponent + 0.2| (K = -0.9 y^-0.2)", conformal["exponents"]["curvature"] + 0.2,
                                   1e-6, kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-9, "basis": "least-squares fit of exact power laws"},
                tolerance={"abs": 1e-6, "rel": 0.0}),
        finding("Cases just beyond each classification threshold are misclassified", "computational_pipeline",
                {"cases": limit_classes, "slow_curvature_exponent": _sig(slow, 6),
                 "near_critical_speed_exponent": _sig(near_critical, 8),
                 "small_deficit_ratio_defect": _sig(1.0 - small_ratio, 6)},
                {"checks": [_check("detection-limit cases misclassified", len(missed), len(limit_classes),
                                   comparison="ge", kind="exact_arithmetic"),
                            _check("z = r^1.99: K exponent + 0.02 (just above the -0.05 threshold)", slow + 0.02, 1e-3,
                                   kind="analytic"),
                            _check("g = y^-1.999 I: radial-speed exponent + 0.9995 (within 1e-3 of -1)",
                                   near_critical + 0.9995, 1e-6, kind="analytic"),
                            _check("cone with 1 - sin(alpha) = 5e-7: ratio defect - 5e-7 (below the 1e-6 threshold)",
                                   1.0 - small_ratio - SMALL_DEFICIT, 1e-12, kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-9, "basis": "least-squares fit of exact power laws"},
                tolerance={"abs": 1e-3, "rel": 0.0},
                counterexample={"statement": "The approach scan classifies every singular point correctly",
                                "witness": limit_classes}),
        finding("Cartesian loops around the sphere pole in the polar chart make a coordinate singularity read as a "
                "conical point", "computational_pipeline",
                {"circumference_ratio": _sig(cartesian_pole["circumference_ratio"], 6),
                 "classification": cartesian_pole["classification"],
                 "exponents": {k: (None if v is None else _sig(v, 6)) for k, v in cartesian_pole["exponents"].items()}},
                {"checks": [_check("classified conical (1 = yes)",
                                   int(cartesian_pole["classification"] == "conical_singularity"), 1, comparison="ge",
                                   kind="exact_arithmetic"),
                            _check("|circumference ratio - 1| with Cartesian loops",
                                   abs(cartesian_pole["circumference_ratio"] - 1.0), 0.1, comparison="ge",
                                   kind="analytic")]},
                uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "trapezoid and Gauss-Legendre quadrature"},
                tolerance={"abs": 1e-6, "rel": 1e-6},
                counterexample={"statement": ("The scan classifies a degenerate point independently of the approach "
                                              "loops chosen by the caller"),
                                "witness": {"point": "sphere north pole, chart A", "polar_loops": classes["sphere-north-pole"],
                                            "cartesian_loops": cartesian_pole["classification"]}}),
        finding("The pointwise guard and the core check refuse degenerate, blown-up, nonfinite and out-of-chart points "
                "with computed codes, and accept curvature below the declared bound", "computational_pipeline",
                {name: observed for name, (_, observed) in refusals.items()},
                {"checks": [_refusal(name, expected, observed) for name, (expected, observed) in refusals.items()]},
                uncertainty={"kind": "reference_error", "value": 0, "basis": "exact refusal codes"},
                tolerance={"abs": 0, "rel": 0}),
        finding("The pointwise guard propagates a refusal code that a surface raises at its own apex unchanged",
                "computational_pipeline", {name: observed for name, (_, observed) in propagated.items()},
                {"checks": [_refusal(name, expected, observed) for name, (expected, observed) in propagated.items()]},
                uncertainty={"kind": "reference_error", "value": 0, "basis": "exact refusal codes"},
                tolerance={"abs": 0, "rel": 0}),
        finding("The core Surface.check accepts sphere-chart points with metric condition number above 1e10",
                "numerical", _sig(math.log10(leniency["max_accepted_condition"]), 4),
                {"checks": [_check("max cond(g) accepted by Surface.check on the pole approach",
                                   leniency["max_accepted_condition"], 1e10, comparison="ge", kind="analytic")]},
                unit="log10 condition number", uncertainty={"kind": "reference_error", "value": 0.25,
                                                            "basis": "theta grid of 4 points per decade"},
                tolerance={"abs": 0.3, "rel": 0.0}),
        finding("The singularity classification applies to scanned physical parts", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}
