"""Surface interface, derivative checks, charts and singularities (T033-T037).

Scope: validates and extends the reusable surface interface of
``ciw.lab.surfaces`` without modifying it. T033 runs a conformance suite over
every catalogue surface, two reparametrized charts and a rigid rotation, and
shows that the suite rejects seeded defects. T034 checks the hand-coded
derivatives against sympy (distinct origin, optional) and against nested
forward-mode dual numbers (same origin). T035 maps the central-difference
error of metric derivatives over twelve decades of step size. T036 builds a
two-chart sphere atlas with exact transitions and integrates geodesics through
the poles with chart switching. T037 scans approaches to candidate singular
points and separates coordinate, conical and curvature singularities and an
infinite-distance boundary, with pointwise refusal codes.

Non-claims: all surfaces, coordinates and curvatures are normalized
mathematical objects. Findings establish agreement between computations on
sampled points of declared domains; they do not establish correctness between
samples, outside the domains, or for any measured physical surface, and the
singularity rules are validated only on the declared examples.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from . import svg
from .evidence import finding
from .registry import task
from .surfaces import HyperbolicPlane, Plane, Reparametrized, Saddle, Sphere
from .surfaces_discrete_ad import DualMath, DualSurface, formulas, partial, symbolic_exact_curvature, \
    symbolic_reference
from .surfaces_discrete_charts import (SWITCH_THRESHOLD, Approach, SphereAtlas, great_circle, integrate_atlas,
                                       integrate_single, pole_passing_great_circle, refusal_code, require_regular,
                                       scan)
from .surfaces_discrete_geometry import (DOMAINS, EPS, SEED, THRESHOLDS, Cone, DroppedCrossTermBump, PolarChart,
                                         PowerGraph, central_difference, conformance, conformance_surfaces,
                                         mutant_surfaces)

MODULE = "src/ciw/lab/surfaces_discrete.py"
GEOMETRY = "src/ciw/lab/surfaces_discrete_geometry.py"
AD = "src/ciw/lab/surfaces_discrete_ad.py"
CHARTS = "src/ciw/lab/surfaces_discrete_charts.py"
DOC = "docs/lab/SURFACE_INTERFACE.md"
TESTS = "tests/test_lab_surfaces_discrete.py"
POINTS = 32
AD_POINTS = 12


def _check(reference, observed, tolerance, comparison="abs_le", kind="analytic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
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
    "points refused by the core Surface.check inside a declared domain"]


@task("T033", changed_files=(MODULE, GEOMETRY, DOC),
      regression_tests=(f"{TESTS}::test_conformance_suite_accepts_every_surface",
                        f"{TESTS}::test_conformance_suite_rejects_seeded_mutants",
                        f"{TESTS}::test_brioschi_recovers_supplied_curvature",
                        f"{TESTS}::test_t033_report"))
def surface_interface(ctx):
    surfaces = conformance_surfaces()
    table = {key: conformance(surface, DOMAINS[key], POINTS, SEED) for key, surface in surfaces.items()}
    mutants = {key: conformance(surface, domain, POINTS, SEED) for key, (surface, domain) in mutant_surfaces().items()}
    worst = {}
    for name in THRESHOLDS:
        values = [row["worst"][name] for row in table.values()]
        worst[name] = min(values) if name == "min_eigenvalue_ratio" else max(values)
    nonconforming = sorted(key for key, row in table.items() if not row["conforms"])
    refused = sum(row["refused_by_core_check"] for row in table.values())
    undetected = sorted(key for key, row in mutants.items() if row["conforms"])
    rotated, base = surfaces["rotated-torus"], surfaces["torus"]
    rigid = max(float(np.max(np.abs(rotated.metric(u) - base.metric(u)))) / float(np.max(np.abs(base.metric(u))))
                for u in DOMAINS["torus"].sample(POINTS, SEED))
    ctx.artifact_json("conformance.json", {"thresholds": THRESHOLDS, "points_per_surface": POINTS, "seed": SEED,
                                           "surfaces": {k: dict(row, describe=surfaces[k].describe())
                                                        for k, row in table.items()}})
    ctx.artifact_json("mutants.json", {"detection": {k: {"failed": row["failed"], "not_evaluated": row["not_evaluated"],
                                                         "worst": row["worst"]} for k, row in mutants.items()}})
    keys = list(table)
    series = [(name, list(range(1, len(keys) + 1)), [table[k]["worst"][name] for k in keys])
              for name in ("gauss_equation", "derivative_consistency", "mixed_partials", "compatibility")]
    ctx.artifact_text("conformance-residuals.svg", svg.line_plot(
        series, title="Worst normalized residual per surface (1..10 as in conformance.json)",
        xlabel="surface index", ylabel="normalized residual", logy=True))
    misscaled, flipped = mutants["misscaled-curvature"], mutants["sign-flipped-derivatives"]
    fields = {
        "hypothesis": ("Every catalogue surface, the reparametrized charts plane-polar and gaussian-bump-shear and a "
                       "rigidly rotated torus satisfy the interface identities on their declared domains, and the "
                       "Gauss equation holds: curvature recomputed from the metric alone equals the supplied K."),
        "mathematical_model": ("g symmetric positive definite; Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il - d_l g_ij) "
                               "symmetric in ij; d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il; d_l d_k g_ij symmetric "
                               "in kl; Brioschi K(E, F, G and their first and second derivatives) = supplied K."),
        "input_data": [f"{len(surfaces)} surfaces (catalogue + plane-polar, gaussian-bump-shear, rotated-torus)",
                       f"{POINTS} seeded points per surface (PCG64 seed {SEED}) in declared domains (docs/lab/SURFACE_INTERFACE.md)",
                       f"{len(mutants)} seeded defect mutants"],
        "observation_model": ("Normalized residuals: metric terms by max|g|, derivative terms by max|dg| + max|g|/l, "
                              "Christoffel terms by max|Gamma| + 1/l, curvature by |K| + 1/l^2, with l the local "
                              "length scale (y on the hyperbolic plane, 1 elsewhere). Second metric derivatives come "
                              "from fourth-order central differences (step 1e-3 l) of the exact first derivatives."),
        "expected_invariant": "Algebraic identities at rounding level; stencil-based identities at <= 1e-8 (dg) and <= 1e-7 (mixed partials, Gauss).",
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
            "Conformance on sampled points is evidence, not proof, of correctness on the whole domain."],
        "recommended_next_task": ("T042: turn the conformance suite into the admission gate for invalid or "
                                  "incomplete surface data, reusing its refusal of the seeded defects."),
    }
    findings = [
        finding("Brioschi curvature from the metric alone equals the supplied Gaussian curvature at every sampled point",
                "numerical", worst["gauss_equation"],
                {"derivation": "Theorema Egregium, Brioschi form; docs/lab/SURFACE_INTERFACE.md#conformance-suite",
                 "checks": [_check("Brioschi K (fourth-order differences of exact dg) vs supplied K, worst normalized",
                                   worst["gauss_equation"], THRESHOLDS["gauss_equation"], kind="invariant")]},
                unit="normalized residual", tolerance={"abs": THRESHOLDS["gauss_equation"], "rel": 0.0}),
        finding("Metrics are symmetric positive definite at every sampled point of the declared domains", "numerical",
                {"max_asymmetry": worst["metric_asymmetry"], "min_eigenvalue_ratio": worst["min_eigenvalue_ratio"],
                 "points_refused_by_core_check": refused},
                {"checks": [_check("|g_12 - g_21| / max|g|", worst["metric_asymmetry"], THRESHOLDS["metric_asymmetry"],
                                   kind="invariant"),
                            _check("min lambda_min / lambda_max over samples", worst["min_eigenvalue_ratio"],
                                   THRESHOLDS["min_eigenvalue_ratio"], comparison="ge", kind="invariant"),
                            _check("sampled points refused by core Surface.check", refused, 0, kind="exact_arithmetic")]},
                tolerance={"abs": 1e-13, "rel": 1e-9}),
        finding("Christoffel symbols are symmetric in their lower indices at every sampled point", "numerical", worst["christoffel_asymmetry"],
                {"checks": [_check("max |Gamma^k_ij - Gamma^k_ji|, normalized", worst["christoffel_asymmetry"],
                                   THRESHOLDS["christoffel_asymmetry"], kind="invariant")]},
                tolerance={"abs": THRESHOLDS["christoffel_asymmetry"], "rel": 0.0}),
        finding("The connection is metric compatible at every sampled point: d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il",
                "numerical",
                worst["compatibility"],
                {"checks": [_check("max compatibility residual, normalized", worst["compatibility"],
                                   THRESHOLDS["compatibility"], kind="invariant")]},
                tolerance={"abs": THRESHOLDS["compatibility"], "rel": 0.0}),
        finding("Supplied metric derivatives agree with fourth-order differences of the metric at every sampled point",
                "numerical",
                worst["derivative_consistency"],
                {"checks": [_check("max |D4 g - dg|, normalized (h = 1e-3 l)", worst["derivative_consistency"],
                                   THRESHOLDS["derivative_consistency"], kind="self_convergence")]},
                tolerance={"abs": THRESHOLDS["derivative_consistency"], "rel": 0.0}),
        finding("Differences of the exact metric derivatives have symmetric mixed partials (dg is a gradient field)",
                "numerical", worst["mixed_partials"],
                {"checks": [_check("max |d_u d_v g - d_v d_u g|, normalized", worst["mixed_partials"],
                                   THRESHOLDS["mixed_partials"], kind="invariant")]},
                tolerance={"abs": THRESHOLDS["mixed_partials"], "rel": 0.0}),
        finding("A rigid rotation leaves the torus metric unchanged at every sampled point", "numerical", rigid,
                {"checks": [_check("max |g_rotated - g_torus| / max|g|", rigid, 1e-14, kind="invariant")]},
                tolerance={"abs": 1e-14, "rel": 0.0}),
        finding("The conformance suite rejects every seeded defect mutant", "computational_pipeline",
                {"mutants": len(mutants), "undetected": len(undetected)},
                {"checks": [_check("mutants passing every identity", len(undetected), 0, kind="exact_arithmetic")]},
                tolerance={"abs": 0, "rel": 0},
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
                tolerance={"abs": 1e-6, "rel": 1e-6},
                counterexample={"statement": "A metric-compatible connection certifies the metric derivatives",
                                "witness": {"mutant": "saddle with dg negated", "failed": flipped["failed"]}}),
        finding("The conformance suite certifies surfaces reconstructed from physical measurements", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T034
EXACT_CURVATURE_KEYS = ("plane", "sphere", "cylinder", "saddle", "torus", "hyperbolic-plane", "plane-polar")


def _declared_curvature(key, surface, sp, u, v):
    """Closed forms declared in ciw.lab.surfaces, with parameters made exact."""
    q = sp.nsimplify
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
    g, dg = surface.metric(u), surface.metric_derivatives(u)
    gamma, curvature = surface.christoffel(u), float(surface.gaussian_curvature(u))
    dg_scale = float(np.max(np.abs(dg))) + float(np.max(np.abs(g))) / length
    return {"metric": float(np.max(np.abs(reference["metric"] - g))) / float(np.max(np.abs(g))),
            "metric_derivatives": float(np.max(np.abs(reference["metric_derivatives"] - dg))) / dg_scale,
            "christoffel": float(np.max(np.abs(reference["christoffel"] - gamma)))
            / (float(np.max(np.abs(gamma))) + 1.0 / length),
            "gaussian_curvature": abs(reference["gaussian_curvature"] - curvature) / (abs(curvature) + length ** -2)}


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


@task("T034", changed_files=(MODULE, AD, GEOMETRY),
      regression_tests=(f"{TESTS}::test_dual_numbers_match_closed_forms",
                        f"{TESTS}::test_dual_derivatives_match_surface_interface",
                        f"{TESTS}::test_sympy_references_match_surface_interface",
                        f"{TESTS}::test_t034_report", f"{TESTS}::test_t034_degrades_without_sympy"))
def derivative_checks(ctx):
    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    have_sympy = ctx.available("module:sympy")
    dual_rows, sympy_rows = {}, {}
    for key, surface in surfaces.items():
        dual = DualSurface(*forms[key])
        reference = symbolic_reference(*forms[key]) if have_sympy else None
        domain = DOMAINS[key]
        dual_worst, sym_worst = {}, {}
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
            if reference is not None:
                for name, value in _residuals(reference(u), surface, u, length).items():
                    sym_worst[name] = max(sym_worst.get(name, 0.0), value)
        dual_rows[key], sympy_rows[key] = dual_worst, sym_worst
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
    producer = {"implementation": "ciw.lab.surfaces", "revision": _source_identity("surfaces.py")}
    findings = []
    if have_sympy:
        import sympy as sp

        checker = {"implementation": "sympy", "revision": sp.__version__}
        sym_derivs = max(max(r["metric"], r["metric_derivatives"], r["christoffel"]) for r in sympy_rows.values())
        sym_curv = max(r["gaussian_curvature"] for r in sympy_rows.values())
        u, v = sp.symbols("u v", real=True)
        exact = {}
        for key in EXACT_CURVATURE_KEYS:
            expression = symbolic_exact_curvature(*forms[key])
            declared = _declared_curvature(key, surfaces[key], sp, u, v)
            exact[key] = {"sympy": str(expression), "declared": str(declared),
                          "difference_simplifies_to_zero": bool(sp.simplify(expression - declared) == 0)}
        mismatches = sum(not row["difference_simplifies_to_zero"] for row in exact.values())
        ctx.artifact_json("sympy-vs-ciw.json", {"sympy": sp.__version__, "points_per_surface": AD_POINTS,
                                                "surfaces": sympy_rows, "exact_curvature": exact})
        findings += [
            finding("sympy symbolic metric, metric derivatives and Christoffel symbols match the ciw surface interface",
                    "numerical", sym_derivs,
                    {"independent_check": dict(_check("max normalized residual over metric, dg and Gamma",
                                                      sym_derivs, 1e-12, kind="analytic"),
                                               producer=producer, checker=checker)},
                    unit="normalized residual", tolerance={"abs": 1e-12, "rel": 0.0}),
            finding("sympy Riemann-tensor curvature R_1212 / det g matches the supplied Gaussian curvature",
                    "numerical", sym_curv,
                    {"independent_check": dict(_check("max |K_sympy - K| / (|K| + 1/l^2)", sym_curv, 1e-12,
                                                      kind="analytic"), producer=producer, checker=checker)},
                    unit="normalized residual", tolerance={"abs": 1e-12, "rel": 0.0}),
            finding("sympy simplifies the Brioschi curvature of seven surfaces exactly to the declared closed forms",
                    "mathematical", {"surfaces": len(exact), "mismatches": mismatches},
                    {"independent_check": dict(_check("closed forms whose difference does not simplify to 0",
                                                      mismatches, 0, kind="exact_arithmetic"),
                                               producer=producer, checker=checker)},
                    tolerance={"abs": 0, "rel": 0}),
        ]
        state = "completed"
    else:
        findings.append(finding("sympy symbolic metric, metric derivatives and Christoffel symbols match the ciw "
                                "surface interface", "numerical", None, {}, expected_not_established=True))
        state = "partial"
    findings += [
        finding("Nested dual-number derivatives of re-expressed embeddings match the metric, dg and Christoffel symbols",
                "numerical", dual_derivs,
                {"checks": [_check("max normalized residual over metric, dg and Gamma (same ciw origin)", dual_derivs,
                                   1e-12, kind="analytic"),
                            _check("max |X_dual - X| (formula re-expresses the same embedding)", dual_embed, 1e-13,
                                   kind="analytic")]},
                unit="normalized residual", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Dual-number curvature (Brioschi with exact second derivatives, and LN - M^2) matches the supplied K",
                "numerical", dual_curv,
                {"checks": [_check("max normalized curvature residual", dual_curv, 1e-12, kind="analytic")]},
                unit="normalized residual", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Dual numbers reproduce closed-form first, mixed and third derivatives without perturbation confusion",
                "numerical", self_error,
                {"checks": [_check("max |dual - closed form| over 8 cases", self_error, 1e-13, kind="analytic")]},
                tolerance={"abs": 1e-13, "rel": 0.0}),
        finding("Dual-number checks expose a hand-coded derivative defect that symmetry checks cannot see",
                "numerical", _sig(defect, 6),
                {"checks": [_check("dropped-cross-term mutant: max normalized |dg_mutant - dg_dual|", defect, 1e-3,
                                   comparison="ge", kind="analytic")]},
                tolerance={"abs": 1e-6, "rel": 1e-4},
                counterexample={"statement": "Finite, index-symmetric hand-coded metric derivatives are correct",
                                "witness": {"mutant": "gaussian-bump with f_xy dropped", "normalized_error": _sig(defect)}}),
    ]
    fields = {
        "hypothesis": ("The hand-coded metric, metric derivatives, Christoffel symbols and Gaussian curvature of every "
                       "conformance surface equal the derivatives of its closed-form embedding (or metric) computed "
                       "by an independent symbolic system and by forward-mode automatic differentiation."),
        "mathematical_model": ("Re-expressed X(u) per surface; g = X_i . X_j; dg by differentiation; Gamma from g and "
                               "dg; sympy K = R_1212 / det g from the Riemann tensor of its own Christoffel symbols; "
                               "dual-number K from Brioschi with exact second derivatives and from LN - M^2."),
        "input_data": [f"{len(surfaces)} conformance surfaces, {AD_POINTS} points each (PCG64 seed {SEED + 34})",
                       f"sympy {'available' if have_sympy else 'unavailable'}; closed forms for {len(EXACT_CURVATURE_KEYS)} surfaces"],
        "observation_model": "Same normalization as T033; exact symbolic comparison by sympy.simplify(expr - declared) == 0.",
        "expected_invariant": "Residuals at rounding level (<= 1e-12); exact closed forms identical.",
        "experiment": ("Evaluate sympy-lambdified and dual-number quantities at seeded points, compare with the ciw "
                       "interface, simplify symbolic curvature exactly, self-test the dual numbers, and confirm a "
                       "seeded derivative defect is exposed."),
        "numerical_result": (f"dual numbers: derivatives {_fmt(dual_derivs)}, curvature {_fmt(dual_curv)}, self-test "
                             f"{_fmt(self_error)}, defect exposed at {_fmt(defect)}"
                             + (f"; sympy: derivatives {_fmt(sym_derivs)}, curvature {_fmt(sym_curv)}, exact closed "
                                f"forms {len(EXACT_CURVATURE_KEYS) - mismatches}/{len(EXACT_CURVATURE_KEYS)}"
                                if have_sympy else "; sympy unavailable, symbolic check not run")),
        "uncertainty": ("Rounding only for the pointwise comparisons (floating evaluation of lambdified expressions "
                        "and dual arithmetic); agreement is shown at sampled points, and the exact symbolic identity "
                        "for seven closed forms."),
        "failure_modes_checked": ["perturbation confusion in nested dual numbers",
                                  "formula re-expression not equal to the core embedding",
                                  "dropped mixed derivative (seeded defect)", "sign convention of the Riemann tensor"],
        "unresolved_assumptions": [
            "The re-expressed formulas are written by hand from the same definitions as the core; a shared "
            "misunderstanding of a surface definition would pass both.",
            "Dual-number agreement is same-origin (ciw) evidence; only the sympy comparison is independent."]
            + ([] if have_sympy else ["sympy is not installed here, so the independent symbolic comparison did not run."]),
        "recommended_next_task": ("T035: use the dual-number third derivatives to predict the finite-difference "
                                  "error curve of the metric derivatives."),
    }
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T035
FD_SURFACES = ("sphere", "torus", "gaussian-bump", "hyperbolic-plane")
FD_CONTROLS = ("saddle", "plane-polar", "plane")
TRUNCATION_WINDOW = (10 ** -3.5, 1e-2)
ROUNDING_WINDOW = (1e-13, 1e-10)
LEADING_STEP = -3.0  # log10 of the relative step where the h^2/6 d^3 g term is compared


def fd_steps():
    return np.array([10.0 ** (-k / 4) for k in range(4, 53)])


def fd_study(points=12, seed=SEED + 35) -> dict:
    """Median central-difference error of dg over seeded points for relative steps 1e-1..1e-13.

    The prediction per point and component is h^2/6 |d^3_k g_ij| + eps |g_ij| / h
    with third derivatives from nested dual numbers.
    """
    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    hs = fd_steps()
    out = {}
    for key in FD_SURFACES + FD_CONTROLS:
        surface, dual, domain = surfaces[key], DualSurface(*forms[key]), DOMAINS[key]
        errors = np.zeros((points, len(hs)))
        predicted = np.zeros_like(errors)
        leading = []
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
                    leading.append(float(np.max(np.abs(fd - dg - term))
                                         / np.max(np.abs(term) + EPS * np.abs(g)[None] / step)))
        median, pmedian = np.median(errors, axis=0), np.median(predicted, axis=0)
        row = {"median_error": [float(v) for v in median], "median_predicted": [float(v) for v in pmedian],
               "leading_term_deviation": max(leading) if leading else None}
        for name, (lo, hi) in (("truncation_slope", TRUNCATION_WINDOW), ("rounding_slope", ROUNDING_WINDOW)):
            mask = (hs >= lo * (1 - 1e-9)) & (hs <= hi * (1 + 1e-9))
            values = median[mask]
            row[name] = None if np.any(values <= 0) else float(np.polyfit(np.log(hs[mask]), np.log(values), 1)[0])
        best, pbest = int(np.argmin(median)), int(np.argmin(pmedian))
        row.update({"h_opt": float(hs[best]), "h_opt_predicted": float(hs[pbest]), "min_error": float(median[best]),
                    "min_error_predicted": float(pmedian[pbest]),
                    "error_at_1e-2": float(median[np.argmin(np.abs(np.log(hs / 1e-2)))]),
                    "error_at_1e-12": float(median[np.argmin(np.abs(np.log(hs / 1e-12)))])})
        out[key] = row
    return {"steps": [float(h) for h in hs], "points": points, "seed": seed, "surfaces": out}


@task("T035", changed_files=(MODULE, GEOMETRY, AD),
      regression_tests=(f"{TESTS}::test_finite_difference_error_is_v_shaped", f"{TESTS}::test_t035_report"))
def finite_difference_derivatives(ctx):
    study = fd_study()
    rows = study["surfaces"]
    main = [rows[k] for k in FD_SURFACES]
    trunc = max(abs(r["truncation_slope"] - 2.0) for r in main)
    rounding = max(abs(r["rounding_slope"] + 1.0) for r in main)
    leading = max(r["leading_term_deviation"] for r in main)
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
        xlabel="relative step h", ylabel="median max |D_h g - dg| / (max|g|/l)", logx=True, logy=True))
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
        "numerical_result": (f"truncation slopes within {_fmt(trunc)} of 2, leading term within {_fmt(leading)}; "
                             f"rounding slopes within {_fmt(rounding)} of -1; observed h* in [{_fmt(bracket[0])}, "
                             f"{_fmt(bracket[1])}] (eps^(1/3) = {_fmt(EPS ** (1 / 3))}), within 10^{_fmt(hopt_log)} of "
                             f"prediction; minimum error <= {_fmt(worst_min)}; sphere error at h = 1e-12 is "
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
        "recommended_next_task": ("T043: propagate vertex and normal uncertainty through difference-based metric "
                                  "derivatives, where noise replaces eps in the step-size law."),
    }
    findings = [
        finding("Central-difference error of the analytic metric derivatives falls as h^2 on the truncation branch",
                "numerical", {k: _sig(rows[k]["truncation_slope"], 6) for k in FD_SURFACES},
                {"checks": [_check("max |slope - 2| over h in [10^-3.5, 1e-2]", trunc, 0.02, kind="analytic"),
                            _check("max |FD - dg - h^2/6 d^3 g| / (|h^2/6 d^3 g| + eps|g|/h) at h = 1e-3 "
                                   "(dual-number third derivatives)", leading, 1e-3, kind="analytic")]},
                tolerance={"abs": 0.01, "rel": 0.0}),
        finding("Rounding error of central differences grows as 1/h for small steps", "numerical",
                {k: _sig(rows[k]["rounding_slope"], 4) for k in FD_SURFACES},
                {"checks": [_check("max |slope + 1| over h in [1e-13, 1e-10]", rounding, 0.2, kind="analytic")]},
                tolerance={"abs": 0.2, "rel": 0.0}),
        finding("The optimal step lies within a factor 4 of the predicted h* = (3 eps |g| / |d^3 g|)^(1/3)", "numerical",
                {k: _sig(math.log10(rows[k]["h_opt"]), 4) for k in FD_SURFACES},
                {"derivation": "minimize h^2/6 |d^3 g| + eps |g| / h",
                 "checks": [_check("max |log10(h_opt / h_pred)|", hopt_log, math.log10(4.0), kind="analytic"),
                            _check("smallest observed h_opt", bracket[0], 1e-7, comparison="ge", kind="analytic"),
                            _check("largest observed h_opt", bracket[1], 1e-4, comparison="le", kind="analytic")]},
                unit="log10(relative step)", tolerance={"abs": 0.75, "rel": 0.0}),
        finding("Analytic metric derivatives agree with central differences to within the predicted minimum error",
                "numerical", _sig(worst_min, 2),
                {"checks": [_check("max E(h_opt) / E_pred(h_pred)", min_ratio, 1.0, comparison="le", kind="analytic"),
                            _check("max normalized E(h_opt)", worst_min, 1e-10, comparison="le", kind="analytic")]},
                unit="normalized error", tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("Smaller finite-difference steps can be far less accurate", "numerical", _sig(math.log10(growth), 3),
                {"checks": [_check("sphere E(1e-12) / E(h_opt)", growth, 1e3, comparison="ge", kind="analytic")]},
                unit="log10 error ratio", tolerance={"abs": 1.0, "rel": 0.0},
                counterexample={"statement": "Decreasing the finite-difference step always improves agreement with the analytic derivative",
                                "witness": {"surface": "sphere", "h_opt": sphere["h_opt"], "error_at_h_opt": _sig(sphere["min_error"]),
                                            "error_at_1e-12": _sig(sphere["error_at_1e-12"])}}),
        finding("Quadratic metrics have no truncation branch and a constant metric differences to exactly zero",
                "numerical", {"quadratic_error_at_1e-2": _sig(control, 2), "plane_max_error": plane_zero},
                {"checks": [_check("saddle and plane-polar E(1e-2)", control, 1e-13, comparison="le", kind="analytic"),
                            _check("plane E(h) for every h", plane_zero, 0.0, kind="exact_arithmetic")]},
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


def _single(chart, u0, v0, length, steps):
    with np.errstate(all="ignore"):
        try:
            return integrate_single(chart, u0, v0, length, steps), None
        except (FloatingPointError, ValueError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            return None, type(exc).__name__


def atlas_study(radius=1.0) -> dict:
    atlas = SphereAtlas(radius)
    length = 2 * math.pi * radius
    rows = []
    for delta in DELTAS:
        start, tangent = pole_passing_great_circle(radius, delta, azimuth=AZIMUTH)
        u0 = atlas.to_chart("A", start)
        v0 = atlas.lift("A", u0, tangent)
        run = integrate_atlas(atlas, "A", u0, v0, length, STEPS)
        exact = great_circle(start, tangent, radius, run["s"])
        error = float(np.max(np.linalg.norm(run["points"] - exact, axis=1)))
        single, failure = _single(atlas.charts["A"], u0, v0, length, STEPS)
        single_error = None if single is None else float(np.max(np.linalg.norm(
            single["points"] - great_circle(start, tangent, radius, single["s"]), axis=1)))
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
        convergence.append({"steps": steps, "error": float(np.max(np.linalg.norm(
            run["points"] - great_circle(start, tangent, radius, run["s"]), axis=1)))})
    orders = [math.log2(convergence[i]["error"] / convergence[i + 1]["error"]) for i in range(2)]
    # Regularity along the delta = 1e-3 path: active chart versus chart A alone.
    start, tangent = pole_passing_great_circle(radius, 1e-3, azimuth=AZIMUTH)
    u0 = atlas.to_chart("A", start)
    run = integrate_atlas(atlas, "A", u0, atlas.lift("A", u0, tangent), length, STEPS)
    exact = great_circle(start, tangent, radius, run["s"])
    chart_a = [atlas.regularity("A", atlas.to_chart("A", x)) for x in exact]
    active = [max(atlas.regularity(name, atlas.to_chart(name, x)) for name in ("A", "B")) for x in exact]
    return {"atlas": atlas.describe(), "azimuth": AZIMUTH, "steps": STEPS, "length": length, "runs": rows,
            "convergence": convergence, "orders": orders,
            "regularity_profile": {"s": [float(v) for v in run["s"][::4]], "chart_A": chart_a[::4],
                                   "best_chart": active[::4]}}


def transition_study(radius=1.0, count=256, seed=SEED + 36) -> dict:
    """Round trips, metric preservation of the velocity pushforward and its difference Jacobian."""
    atlas = SphereAtlas(radius)
    rng = np.random.Generator(np.random.PCG64(seed))
    normals = rng.standard_normal((count, 3))
    points = radius * normals / np.linalg.norm(normals, axis=1)[:, None]
    roundtrip = metric_defect = jacobian_defect = 0.0
    covering = math.inf
    used = 0
    for index, x in enumerate(points):
        ua, ub = atlas.to_chart("A", x), atlas.to_chart("B", x)
        ra, rb = atlas.regularity("A", ua), atlas.regularity("B", ub)
        covering = min(covering, max(ra, rb))
        if min(ra, rb) < 0.05:
            continue  # the transition is evaluated only where both charts are regular
        used += 1
        back = atlas.transition("B", "A", atlas.transition("A", "B", ua))
        roundtrip = max(roundtrip, float(np.linalg.norm(atlas.embedding("A", back) - x)) / radius)
        va = rng.standard_normal(2)
        ub2, vb = atlas.pushforward("A", "B", ua, va)
        speed_a, speed_b = atlas.charts["A"].speed_squared(ua, va), atlas.charts["B"].speed_squared(ub2, vb)
        metric_defect = max(metric_defect, abs(speed_b - speed_a) / speed_a)
        jac = np.column_stack([atlas.pushforward("A", "B", ua, e)[1] for e in np.eye(2)])
        h = 1e-6
        fd = np.column_stack([
            _angle_difference(atlas.transition("A", "B", ua + h * e), atlas.transition("A", "B", ua - h * e)) / (2 * h)
            for e in np.eye(2)])
        jacobian_defect = max(jacobian_defect, float(np.max(np.abs(fd - jac))) / float(np.max(np.abs(jac))))
    # Dense covering check: max(det_A, det_B) / R^4 = max(x^2 + y^2, y^2 + z^2) / R^2 >= 1/2.
    dense = rng.standard_normal((4096, 3))
    dense /= np.linalg.norm(dense, axis=1)[:, None]
    covering = min(covering, float(np.min(np.maximum(dense[:, 0] ** 2 + dense[:, 1] ** 2,
                                                     dense[:, 1] ** 2 + dense[:, 2] ** 2))))
    return {"points": count, "used": used, "seed": seed, "roundtrip": roundtrip, "metric_defect": metric_defect,
            "jacobian_defect": jacobian_defect, "covering_min": covering}


def _angle_difference(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (d + math.pi) % (2 * math.pi) - math.pi


@task("T036", changed_files=(MODULE, CHARTS, DOC),
      regression_tests=(f"{TESTS}::test_atlas_transitions_are_exact",
                        f"{TESTS}::test_atlas_geodesic_through_pole_matches_great_circle",
                        f"{TESTS}::test_single_chart_near_pole_counterexample", f"{TESTS}::test_t036_report"))
def chart_transitions(ctx):
    study = atlas_study()
    transitions = transition_study()
    runs = {row["delta"]: row for row in study["runs"]}
    through = runs[0.0]
    atlas_errors = [row["atlas_error"] for row in study["runs"]]
    spread = max(atlas_errors) / min(atlas_errors)
    order_defect = max(abs(order - 4.0) for order in study["orders"])
    min_after = min(row["min_det_after_switch"] for row in study["runs"])
    min_active = min(row["min_active_det"] for row in study["runs"])
    degraded = [row for row in study["runs"] if row["delta"] > 0 and (
        row["single_chart_failure"] is not None or row["single_chart_error"] > 10 * row["atlas_error"])]
    failed = [row for row in study["runs"] if row["single_chart_failure"] is not None]
    ctx.artifact_json("atlas-runs.json", study)
    ctx.artifact_json("transitions.json", transitions)
    positive = [row for row in study["runs"] if row["delta"] > 0]
    ctx.artifact_text("atlas-vs-single-chart.svg", svg.line_plot(
        [("atlas (chart switching)", [r["delta"] for r in positive], [r["atlas_error"] for r in positive]),
         ("single polar chart (failures omitted)",
          [r["delta"] for r in positive if r["single_chart_error"] is not None],
          [r["single_chart_error"] for r in positive if r["single_chart_error"] is not None])],
        title=f"Great-circle error after 2 pi R, RK4 with {STEPS} steps", xlabel="closest approach to the pole",
        ylabel="max |X - X_exact|", logx=True, logy=True))
    profile = study["regularity_profile"]
    ctx.artifact_text("regularity-profile.svg", svg.line_plot(
        [("chart A alone: sin^2 theta_A", profile["s"], profile["chart_A"]),
         ("best chart of the atlas", profile["s"], profile["best_chart"])],
        title="Normalized det g along the delta = 1e-3 great circle", xlabel="arclength s",
        ylabel="det g / R^4", logy=True, markers=False))
    fields = {
        "hypothesis": ("A two-chart polar atlas of the sphere with exact transitions integrates great circles through "
                       "or near a pole with an accuracy that does not depend on how close they pass, while a single "
                       "polar chart loses accuracy or fails near its poles."),
        "mathematical_model": ("Chart A: X = R(sin t cos p, sin t sin p, cos t); chart B = R_y(pi/2) X_A, poles on A's "
                               "equator. det g / R^4 = sin^2 theta in each chart and sin^2 theta_A + sin^2 theta_B = "
                               "1 + y^2/R^2 >= 1, so the better chart always has det g / R^4 >= 1/2. Transitions: "
                               "u_B = X_B^{-1}(X_A(u_A)), v_B = g_B^{-1} J_B^T J_A v_A."),
        "input_data": [f"unit sphere; great circles with closest approach delta in {list(DELTAS)} to the north pole, "
                       f"azimuth {AZIMUTH}, full length 2 pi", f"RK4, {STEPS} steps (convergence: 200/400/800)",
                       f"switch threshold det g / R^4 < {SWITCH_THRESHOLD}",
                       f"{transitions['points']} seeded sphere points for transitions (seed {transitions['seed']})"],
        "observation_model": "Ambient error max_s |X_num(s) - X_exact(s)|; exact great circle cos(s) X0 + sin(s) T0.",
        "expected_invariant": ("Atlas error independent of delta and fourth order in the step; transitions exact to "
                               "rounding; single-chart error grows or the integration fails for small delta > 0."),
        "experiment": ("Integrate each great circle with chart switching and in chart A alone; verify transitions "
                       "by round trip, speed preservation and difference Jacobians; convergence at delta = 0."),
        "numerical_result": (f"atlas through the pole: error {_fmt(through['atlas_error'])}, {through['switches']} "
                             f"switches ({', '.join(through['switch_path'])}); atlas error spread over delta "
                             f"{_fmt(spread)}x; orders {', '.join(_fmt(o) for o in study['orders'])}; single chart "
                             f"failed for {len(failed)} and degraded (>10x atlas) for {len(degraded)} of "
                             f"{len(positive)} delta > 0; exact meridian in chart A alone: error "
                             f"{_fmt(through['single_chart_error'])}; transitions round trip "
                             f"{_fmt(transitions['roundtrip'])}, speed {_fmt(transitions['metric_defect'])}, Jacobian "
                             f"{_fmt(transitions['jacobian_defect'])}; covering bound {_fmt(transitions['covering_min'])}."),
        "uncertainty": ("Atlas errors are RK4 truncation errors (order 4); single-chart failure steps and the error "
                        "of the knife-edge cases (delta <= 1e-8) depend on rounding and may differ across platforms."),
        "failure_modes_checked": ["geodesic exactly through a pole", "geodesics 1e-1..1e-12 from a pole",
                                  "chattering between charts (bound 1/2 > threshold 1/4)",
                                  "longitude wrap-around in transition differences",
                                  "nonfinite single-chart states and math domain errors"],
        "unresolved_assumptions": [
            "Only the sphere has an atlas here; other surfaces need their own charts and transitions.",
            "Switching happens between steps; an adaptive integrator would need event location at the threshold."],
        "recommended_next_task": ("T037: detect coordinate singularities automatically so that chart switching can "
                                  "be triggered by a classified singularity rather than a fixed threshold."),
    }
    single_through = through["single_chart_error"]
    findings = [
        finding("Atlas integration of the great circle through the north pole matches the exact great circle",
                "numerical", _sig(through["atlas_error"], 3),
                {"checks": [_check("max |X - X_exact| over 2 pi, RK4 400 steps", through["atlas_error"], 1e-6,
                                   kind="analytic"),
                            _check("chart switches along the path", through["switches"], 1, comparison="ge",
                                   kind="exact_arithmetic")]},
                unit="length", tolerance={"abs": 1e-8, "rel": 0.05}),
        finding("Atlas accuracy is independent of the distance of closest approach to a pole", "numerical",
                _sig(spread, 3),
                {"checks": [_check("max / min atlas error over delta", spread, 1.5, comparison="le", kind="analytic")]},
                tolerance={"abs": 0.1, "rel": 0.0}),
        finding("Atlas integration keeps fourth-order convergence across chart switches", "numerical",
                [_sig(order, 4) for order in study["orders"]],
                {"checks": [_check("max |order - 4| (200/400/800 steps)", order_defect, 0.25, kind="self_convergence")]},
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
                tolerance={"abs": 1e-7, "rel": 0.0}),
        finding("The better chart of the atlas always has det g / R^4 >= 1/2, so switching never chatters",
                "mathematical", {"covering_min": _sig(transitions["covering_min"], 6), "min_det_after_switch":
                                 _sig(min_after, 6), "min_active_det": _sig(min_active, 6)},
                {"derivation": "sin^2 theta_A + sin^2 theta_B = 1 + y^2 / R^2 >= 1",
                 "checks": [_check("min over 4352 sphere points of max(det_A, det_B) / R^4", transitions["covering_min"],
                                   0.5 - 1e-12, comparison="ge", kind="analytic"),
                            _check("min det g / R^4 right after a switch", min_after, 0.5 - 1e-12, comparison="ge",
                                   kind="analytic"),
                            _check("min det g / R^4 of the active chart along every run", min_active,
                                   SWITCH_THRESHOLD, comparison="ge", kind="analytic")]},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("A single polar chart fails or loses accuracy on great circles passing near its pole", "numerical",
                {"cases": len(positive), "failed": len(failed), "degraded": len(degraded)},
                {"checks": [_check("delta > 0 cases where chart A alone fails or errs > 10x the atlas", len(degraded),
                                   5, comparison="ge", kind="analytic"),
                            _check("delta > 0 cases where chart A alone fails outright", len(failed), 1,
                                   comparison="ge", kind="analytic")]},
                tolerance={"abs": 1, "rel": 0.0},
                counterexample={"statement": ("Fixed-step RK4 in a single polar chart integrates every great circle "
                                              "as accurately as a chart-switching atlas at the same step count"),
                                "witness": {"delta_failed": [r["delta"] for r in failed],
                                            "delta_0.1_single_error": _sig(runs[0.1]["single_chart_error"]) if runs[0.1]["single_chart_error"] is not None else None,
                                            "delta_0.1_atlas_error": _sig(runs[0.1]["atlas_error"])}}),
        finding("Along the exact meridian (v_phi = 0 exactly) chart A alone crosses the pole accurately", "numerical",
                _sig(single_through, 2),
                {"checks": [_check("chart A alone, delta = 0: max |X - X_exact|", single_through, 1e-10, comparison="le",
                                   kind="analytic")]},
                unit="length", tolerance={"abs": 1e-10, "rel": 0.0},
                counterexample={"statement": "Single-chart integration exactly through a coordinate pole always fails",
                                "witness": {"delta": 0.0, "single_chart_error": _sig(single_through, 2),
                                            "delta_1e-12_error": _sig(runs[1e-12]["single_chart_error"], 2)
                                            if runs[1e-12]["single_chart_error"] is not None else None}}),
        finding("Chart-switching geodesic integration is ready for tool paths over physical parts", "industrial_readiness",
                None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ---------------------------------------------------------------- T037
def approaches() -> list:
    return [Approach("sphere-north-pole", Sphere(1.0), (0.0, 0.0), True),
            Approach("plane-polar-origin", Reparametrized(Plane(), PolarChart()), (0.0, 0.0), True),
            Approach("cone-apex", Cone(math.pi / 6), (0.0, 0.0), True),
            Approach("power-graph-apex", PowerGraph(1.0, 1.5), (0.0, 0.0), False),
            Approach("hyperbolic-boundary", HyperbolicPlane(1.0), (0.2, 0.0), False, angle=math.pi / 2),
            Approach("saddle-origin", Saddle(1.0), (0.0, 0.0), False),
            Approach("sphere-equator", Sphere(1.0), (math.pi / 2, 0.3), False)]


EXPECTED_CLASS = {"sphere-north-pole": "coordinate_singularity", "plane-polar-origin": "coordinate_singularity",
                  "cone-apex": "conical_singularity", "power-graph-apex": "curvature_singularity",
                  "hyperbolic-boundary": "infinite_distance_boundary", "saddle-origin": "regular",
                  "sphere-equator": "regular"}


def refusal_cases() -> dict:
    """Pointwise guard outcomes: (description, expected code, observed code)."""
    sphere, graph = Sphere(1.0), PowerGraph(1.0, 1.5)
    return {
        "sphere theta = 3e-5 (guard)": ("degenerate_metric", refusal_code(require_regular, sphere, np.array([3e-5, 0.3]))),
        "sphere theta = 1e-7 (core Surface.check)": ("surface_refusal", refusal_code(sphere.check, np.array([1e-7, 0.3]))),
        "cone apex r = 0 (guard)": ("degenerate_metric", refusal_code(require_regular, Cone(), np.array([0.0, 0.3]))),
        "cone apex r = 0 (curvature)": ("conical_singularity", refusal_code(Cone().gaussian_curvature, np.array([0.0, 0.3]))),
        "power graph rho = 1e-6 (guard)": ("curvature_blowup", refusal_code(require_regular, graph, np.array([1e-6, 0.0]))),
        "power graph apex (metric)": ("curvature_singularity", refusal_code(require_regular, graph, np.array([0.0, 0.0]))),
        "hyperbolic y = -1 (guard)": ("chart_refused", refusal_code(require_regular, HyperbolicPlane(), np.array([0.0, -1.0]))),
        "nonfinite coordinates (guard)": ("nonfinite_point", refusal_code(require_regular, sphere, np.array([math.nan, 0.0]))),
        "sphere equator (guard)": ("accepted", refusal_code(require_regular, sphere, np.array([math.pi / 2, 0.3]))),
    }


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
                        f"{TESTS}::test_singularity_refusal_codes", f"{TESTS}::test_t037_report"))
def coordinate_singularities(ctx):
    scans = {a.name: scan(a) for a in approaches()}
    classes = {name: result["classification"] for name, result in scans.items()}
    misclassified = sorted(name for name in classes if classes[name] != EXPECTED_CLASS[name])
    refusals = refusal_cases()
    leniency = core_check_leniency()
    ctx.artifact_json("singularity-scans.json", {"expected": EXPECTED_CLASS, "scans": scans})
    ctx.artifact_json("refusals.json", {name: {"expected": e, "observed": o} for name, (e, o) in refusals.items()}
                      | {"core_check_leniency": leniency})
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
    graph, boundary = scans["power-graph-apex"], scans["hyperbolic-boundary"]
    e_pole, e_graph = pole["exponents"], graph["exponents"]
    atlas = SphereAtlas(1.0)
    pole_in_b = atlas.regularity("B", atlas.to_chart("B", np.array([0.0, 0.0, 1.0])))
    rmin = min(graph["distances"])
    graph_prefactor = abs(graph["table"]["curvature"][-1]) * rmin
    graph_expected = 1.125 * PowerGraph(1.0, 1.5).c ** 2
    signature = max(abs(polar["exponents"][k] - cone["exponents"][k]) for k in ("det", "condition", "christoffel"))
    cone_obj = Cone(math.pi / 6)
    fields = {
        "hypothesis": ("A scan into a candidate point separates coordinate singularities (det g -> 0 with bounded K, "
                       "removable by a chart change) from conical points (bounded K but a circumference deficit) and "
                       "curvature singularities (K unbounded), and pointwise guards refuse all of them with codes."),
        "mathematical_model": ("Fit power laws r^a for det g, cond(g), max|Gamma| and |K| over r in [1e-8, 1e-3]; "
                               "circumference ratio C(r) / (2 pi rho(r)) -> 1 at smooth points and sin(alpha) at a cone "
                               "apex; radial distance diverging (log) marks an infinite-distance boundary."),
        "input_data": ["sphere pole (chart A), plane in polar chart, cone alpha = pi/6, z = r^(3/2), hyperbolic y -> 0, "
                       "saddle origin, sphere equator", "29 log-spaced distances 1e-1..1e-8"],
        "observation_model": "Pointwise invariants of the chart; loop length by 64-point trapezoid; radial length by 16-point Gauss-Legendre.",
        "expected_invariant": ("Sphere pole: det ~ r^2, cond ~ r^-2, Gamma ~ r^-1, K ~ r^0; r^(3/2) graph: K ~ r^-1 "
                               "with det g -> 1; cone: ratio sin(alpha) = 1/2."),
        "experiment": "Scan, fit, classify every approach; exercise the pointwise guard and the core check.",
        "numerical_result": (f"{len(classes) - len(misclassified)}/{len(classes)} approaches classified as expected; "
                             f"sphere pole exponents det {_fmt(e_pole['det'])}, cond {_fmt(e_pole['condition'])}, Gamma "
                             f"{_fmt(e_pole['christoffel'])}, K {_fmt(e_pole['curvature'])}; r^(3/2) graph K exponent "
                             f"{_fmt(e_graph['curvature'])}, K r -> {graph_prefactor:.6g} (9/8 expected); cone circumference ratio "
                             f"{_fmt(cone['circumference_ratio'])}; polar-plane and cone exponent signatures differ by "
                             f"{_fmt(signature)}; core check accepts cond(g) up to {_fmt(leniency['max_accepted_condition'])}; "
                             f"{sum(e == o for e, o in refusals.values())}/{len(refusals)} refusal codes as expected."),
        "uncertainty": ("Exponent fits are exact power laws up to rounding for these closed forms; the circumference "
                        "test assumes radial chart lines are geodesics (true for these rotationally symmetric "
                        "examples) and a small-r limit taken at r = 1e-8."),
        "failure_modes_checked": ["false positive at regular points (saddle origin, sphere equator)",
                                  "coordinate singularity mistaken for a conical point (identical pointwise signature)",
                                  "Christoffel blow-up without curvature blow-up and vice versa",
                                  "metric blow-up at an infinite-distance boundary", "apex evaluation (refused)"],
        "unresolved_assumptions": [
            "The classification rules are validated on seven declared examples, not proven for general surfaces.",
            "Non-rotationally-symmetric singular points (edges, cusps along curves) are not covered.",
            "The pointwise guard reports both a conical point and a coordinate singularity as degenerate_metric and "
            "does not flag the hyperbolic boundary at any finite y (cond g = 1, K = -1); only the scan separates them.",
            "The core Surface.check threshold (det g <= 1e-12 (tr g)^2) is left unchanged; a condition-number "
            "threshold is proposed as a core change."],
        "recommended_next_task": ("T042: adopt the refusal codes (degenerate_metric, curvature_blowup, "
                                  "curvature_singularity, chart_refused, nonfinite_point) as the refusal states for "
                                  "invalid or incomplete surface data."),
    }
    findings = [
        finding("The sphere pole in the polar chart is a coordinate singularity: det g -> 0 while K stays 1",
                "numerical", {"exponents": {k: _sig(v, 6) for k, v in e_pole.items()}, "classification": classes["sphere-north-pole"],
                              "regularity_in_chart_B": _sig(pole_in_b, 6)},
                {"checks": [_check("|det exponent - 2|", e_pole["det"] - 2, 1e-3, kind="analytic"),
                            _check("|cond exponent + 2|", e_pole["condition"] + 2, 1e-3, kind="analytic"),
                            _check("|Gamma exponent + 1|", e_pole["christoffel"] + 1, 1e-3, kind="analytic"),
                            _check("|K exponent|", e_pole["curvature"], 1e-3, kind="analytic"),
                            _check("chart B det g / R^4 at the pole of chart A", pole_in_b, 1 - 1e-12, comparison="ge",
                                   kind="analytic")]},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("Every approach is classified as expected, with no false positive at regular points",
                "computational_pipeline", classes,
                {"checks": [_check("misclassified approaches", len(misclassified), 0, kind="exact_arithmetic")]},
                tolerance={"abs": 0, "rel": 0}),
        finding("The graph z = r^(3/2) has a curvature singularity although its Monge metric is regular", "numerical",
                {"curvature_exponent": _sig(e_graph["curvature"], 6), "det_exponent": _sig(e_graph["det"], 3),
                 "christoffel_exponent": _sig(e_graph["christoffel"], 3), "K_times_r": _sig(graph_prefactor, 8)},
                {"checks": [_check("|K exponent + 1|", e_graph["curvature"] + 1, 5e-3, kind="analytic"),
                            _check("|det exponent|", e_graph["det"], 1e-3, kind="analytic"),
                            _check("|K r - 9 c^2 / 8| at r = 1e-8", graph_prefactor - graph_expected, 1e-6, kind="analytic")]},
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
                tolerance={"abs": 1e-2, "rel": 0.0},
                counterexample={"statement": "Christoffel-symbol blow-up indicates a curvature singularity",
                                "witness": {"sphere_pole_max_gamma_at_1e-8": _sig(pole["table"]["christoffel"][-1]),
                                            "sphere_pole_K": 1.0}}),
        finding("A cone apex and the polar-chart origin share every pointwise exponent; only the circumference ratio separates them",
                "numerical", {"signature_difference": _sig(signature, 3), "cone_ratio": _sig(cone["circumference_ratio"], 8),
                              "polar_ratio": _sig(polar["circumference_ratio"], 8),
                              "angle_deficit": _sig(cone_obj.angle_deficit(), 8)},
                {"checks": [_check("max exponent difference (det, cond, Gamma)", signature, 1e-6, kind="analytic"),
                            _check("|cone ratio - sin(pi/6)|", cone["circumference_ratio"] - 0.5, 1e-9, kind="analytic"),
                            _check("|polar ratio - 1|", polar["circumference_ratio"] - 1.0, 1e-9, kind="analytic")]},
                tolerance={"abs": 1e-6, "rel": 0.0},
                counterexample={"statement": ("Pointwise metric and curvature invariants distinguish a removable "
                                              "coordinate singularity from a conical point"),
                                "witness": {"removable": "plane in polar chart at r = 0", "conical": "cone alpha = pi/6 apex",
                                            "angle_deficit": _sig(cone_obj.angle_deficit())}}),
        finding("The hyperbolic chart boundary y -> 0 is at infinite distance, not a singular point", "numerical",
                {"det_exponent": _sig(boundary["exponents"]["det"], 6), "near_over_far_distance": _sig(boundary["distance_near_over_far"], 6)},
                {"checks": [_check("|det exponent + 4|", boundary["exponents"]["det"] + 4, 1e-3, kind="analytic"),
                            _check("radial length of the last two decades / first two decades",
                                   boundary["distance_near_over_far"], 0.9, comparison="ge", kind="analytic")]},
                tolerance={"abs": 1e-3, "rel": 0.0}),
        finding("The pointwise guard refuses singular points with the expected codes and accepts a regular point",
                "computational_pipeline", {name: observed for name, (_, observed) in refusals.items()},
                {"checks": [_refusal(name, expected, observed) for name, (expected, observed) in refusals.items()]},
                tolerance={"abs": 0, "rel": 0}),
        finding("The core Surface.check accepts sphere-chart points with metric condition number above 1e10",
                "numerical", _sig(math.log10(leniency["max_accepted_condition"]), 4),
                {"checks": [_check("max cond(g) accepted by Surface.check on the pole approach",
                                   leniency["max_accepted_condition"], 1e10, comparison="ge", kind="analytic")]},
                unit="log10 condition number", tolerance={"abs": 0.3, "rel": 0.0}),
        finding("The singularity classification applies to scanned physical parts", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}
