"""Geodesic and Jacobi-field experiments T001-T009 on the ``ciw.lab.surfaces`` catalogue.

Scope: symbolic re-derivation of the geodesic equations (sympy, optional),
closed-form and arbitrary-precision references (mpmath, optional), integrator
order and unit-speed drift studies, the Jacobi separation law, finite-difference
flow perturbations, the transfer-matrix determinant, conjugate and focal points,
and the separate behaviour of the lateral and heading Jacobi columns. The pinned
Curved-Surface-Geodesic-Sensitivity-Runtime provider is compared in a
subprocess only when it is bound as ``csg``.

Non-claims: every surface, path and curvature is a declared mathematical object
in normalized units. Agreement between implementations, convergence orders and
invariant checks are statements about computations. None of them measures,
calibrates or certifies a physical surface, tool path, vehicle or sensor; the
physical conclusions a task invites are recorded as not established.
"""
from __future__ import annotations

import ast
import inspect
import math
import textwrap

import numpy as np

from . import geodesic_jacobi_common as gj
from . import integrators, jacobi, surfaces, svg
from .evidence import finding
from .registry import task

MODULE = "src/ciw/lab/geodesic_jacobi.py"
COMMON = "src/ciw/lab/geodesic_jacobi_common.py"
TESTS = "tests/test_lab_geodesic_jacobi.py"
DOC = "docs/lab/GEODESIC_JACOBI.md"
CHANGED = (MODULE, COMMON, TESTS, DOC)

REPORT_FIELDS = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                 "experiment", "numerical_result", "uncertainty", "failure_modes_checked", "unresolved_assumptions",
                 "recommended_next_task")

# Regression tolerances: converged small discrepancies and values, and fitted rates.
TOL_SMALL = {"abs": 1e-9, "rel": 0.0}
TOL_VALUE = {"abs": 1e-9, "rel": 1e-7}
TOL_RATE = {"abs": 0.02, "rel": 0.0}

CURVED_CHARTS = ("sphere", "saddle", "torus", "gaussian-bump", "hyperbolic-plane", "plane-polar", "cylinder-polar")
FLAT_CARTESIAN = ("plane", "cylinder")
NOMINAL_ORDER = {"euler": 1, "midpoint": 2, "rk4": 4}


def _test(name: str) -> str:
    return f"{TESTS}::{name}"


def _plain(value):
    """JSON-ready copy of nested numbers (numpy scalars become Python numbers, full precision)."""
    return gj.rounded(value, 17)


def _outcome(state: str, findings: list, **fields) -> dict:
    missing = [name for name in REPORT_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"Report fields missing: {missing}")
    return {"state": state, "fields": fields, "findings": findings}


def _fmt(value, digits=3) -> str:
    return format(float(value), f".{digits}g")


def _fine(key: str) -> int:
    """Even fixed RK4 step count at most ``FINE_STEP`` long, so the half-count shares nodes."""
    return 2 * math.ceil(gj.fine_steps(key) / 2)


def _fine_transfer(ctx, key):
    return gj.transfer(ctx, key, "rk4", _fine(key))


def _constant_curvature_of(key: str) -> float:
    """Declared constant curvature of a constant-curvature path (exact from surface parameters)."""
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    if key == "torus-outer-equator":
        return 1.0 / (surf.minor * (surf.major + surf.minor))
    if key == "torus-inner-equator":
        return -1.0 / (surf.minor * (surf.major - surf.minor))
    return float(surf.gaussian_curvature(np.asarray(spec.u0, dtype=float)))


def _speed_drift(surf, states) -> np.ndarray:
    return np.array([surf.speed_squared(y[:2], y[2:4]) - 1.0 for y in states])


# ---------------------------------------------------------------------------
# T001: symbolic re-derivation
# ---------------------------------------------------------------------------
T001_CHARTS = gj.CATALOGUE_KEYS + gj.POLAR_KEYS
SAMPLES_PER_CHART = 8
SAMPLE_BOXES = {
    "plane": ((-2.0, 2.0), (-2.0, 2.0)),
    "sphere": ((0.3, math.pi - 0.3), (-math.pi, math.pi)),
    "cylinder": ((-math.pi, math.pi), (-2.0, 2.0)),
    "saddle": ((-1.5, 1.5), (-1.5, 1.5)),
    "torus": ((-math.pi, math.pi), (-math.pi, math.pi)),
    "gaussian-bump": ((-2.5, 2.5), (-2.5, 2.5)),
    "hyperbolic-plane": ((-2.0, 2.0), (0.3, 3.0)),
    "plane-polar": ((0.5, 2.5), (-math.pi, math.pi)),
    "cylinder-polar": ((0.5, 2.5), (-math.pi, math.pi)),
}


def sample_points(key: str):
    """Seeded chart points and velocities (one PCG64 stream per chart)."""
    index = T001_CHARTS.index(key)
    rng = np.random.Generator(np.random.PCG64(gj.SEED + 1000 * index))
    (a0, a1), (b0, b1) = SAMPLE_BOXES[key]
    points = np.column_stack([rng.uniform(a0, a1, SAMPLES_PER_CHART), rng.uniform(b0, b1, SAMPLES_PER_CHART)])
    return points, rng.uniform(-1.0, 1.0, (SAMPLES_PER_CHART, 2))


def _discrepancy(value, reference) -> float:
    value, reference = np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    return float(np.max(np.abs(value - reference) / np.maximum(1.0, np.abs(reference))))


def _derived(ctx, key):
    return ctx.memo(("gj-derive", key), lambda: gj.derive(key))


def _lambdas(ctx, key):
    return ctx.memo(("gj-lambdify", key), lambda: gj.lambdified(_derived(ctx, key), modules="math"))


def compatibility_residual(surf, u) -> float:
    """Metric compatibility d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il (nabla g = 0)."""
    g, dg, gamma = surf.metric(u), surf.metric_derivatives(u), surf.christoffel(u)
    predicted = np.einsum("lki,lj->kij", gamma, g) + np.einsum("lkj,il->kij", gamma, g)
    return _discrepancy(predicted, dg)


def metric_derivative_residual(surf, u, delta=1e-5) -> float:
    """Exact metric derivatives against central differences of the metric."""
    dg = surf.metric_derivatives(u)
    numeric = np.empty_like(dg)
    for k in range(2):
        e = np.zeros(2)
        e[k] = delta
        numeric[k] = (surf.metric(u + e) - surf.metric(u - e)) / (2 * delta)
    return _discrepancy(numeric, dg)


def intrinsic_curvature(surf, u, delta=1e-4) -> float:
    """K = R_uvuv / det g from central-differenced Christoffel symbols (metric only).

    R^r_{s m n} = d_m Gamma^r_{ns} - d_n Gamma^r_{ms} + Gamma^r_{ml} Gamma^l_{ns}
    - Gamma^r_{nl} Gamma^l_{ms}; the second fundamental form never enters.
    """
    u = np.asarray(u, dtype=float)
    gamma, g = surf.christoffel(u), surf.metric(u)
    d_gamma = np.empty((2, 2, 2, 2))
    for m in range(2):
        e = np.zeros(2)
        e[m] = delta
        d_gamma[m] = (surf.christoffel(u + e) - surf.christoffel(u - e)) / (2 * delta)
    s, m, n = 1, 0, 1
    upper = np.array([d_gamma[m][r, n, s] - d_gamma[n][r, m, s]
                      + sum(gamma[r, m, q] * gamma[q, n, s] - gamma[r, n, q] * gamma[q, m, s] for q in range(2))
                      for r in range(2)])
    return float(g[0] @ upper / np.linalg.det(g))


def _curvature_figure(ctx, have_sympy):
    series = []
    for key in gj.VARIABLE_KEYS:
        tr = _fine_transfer(ctx, key)
        stride = max(1, len(tr.s) // 40)
        s = tr.s[::stride]
        pts = tr.points[::stride]
        series.append((f"{key} ciw", s, [gj.surface(key).gaussian_curvature(u) for u in pts]))
        if have_sympy:
            curvature = _lambdas(ctx, key)["curvature"]
            series.append((f"{key} sympy", s, [float(curvature(*u)) for u in pts]))
        else:
            series.append((f"{key} intrinsic FD", s, [intrinsic_curvature(gj.surface(key), u) for u in pts]))
    return svg.line_plot(series, title="Gaussian curvature along the standard paths",
                         xlabel="arclength s", ylabel="K(gamma(s))", markers=False)


@task("T001", changed_files=CHANGED,
      regression_tests=(_test("test_t001_symbolic_derivations_match_surfaces"),
                        _test("test_t001_without_sympy_is_partial"),
                        _test("test_t001_self_consistency_helpers")))
def rederive_geodesic_equations(ctx):
    have_sympy = ctx.available("module:sympy")
    rows, texts, latex = {}, [], []
    for key in T001_CHARTS:
        surf = gj.surface(key)
        points, velocities = sample_points(key)
        row = {"symmetry": 0.0, "compatibility": 0.0, "metric_derivatives": 0.0, "intrinsic_curvature": 0.0,
               "max_abs_christoffel": 0.0, "max_abs_intrinsic_curvature": 0.0}
        if have_sympy:
            functions = _lambdas(ctx, key)
            derived = _derived(ctx, key)
            texts.append(gj.text_block(key, derived))
            latex.append(gj.latex_block(key, derived))
            row.update(metric=0.0, christoffel=0.0, acceleration=0.0, curvature=0.0,
                       symbolic_curvature_is_zero=bool(derived["sympy"].simplify(derived["curvature"]) == 0))
        for u, v in zip(points, velocities):
            surf.check(u)
            gamma = surf.christoffel(u)
            k_ciw = surf.gaussian_curvature(u)
            k_intrinsic = intrinsic_curvature(surf, u)
            row["symmetry"] = max(row["symmetry"], float(np.max(np.abs(gamma - gamma.transpose(0, 2, 1)))))
            row["compatibility"] = max(row["compatibility"], compatibility_residual(surf, u))
            row["metric_derivatives"] = max(row["metric_derivatives"], metric_derivative_residual(surf, u))
            row["intrinsic_curvature"] = max(row["intrinsic_curvature"], _discrepancy(k_intrinsic, k_ciw))
            row["max_abs_christoffel"] = max(row["max_abs_christoffel"], float(np.max(np.abs(gamma))))
            row["max_abs_intrinsic_curvature"] = max(row["max_abs_intrinsic_curvature"], abs(k_intrinsic))
            if have_sympy:
                g_sym = np.array(functions["metric"](*u), dtype=float).reshape(2, 2)
                gamma_sym = np.array(functions["christoffel"](*u), dtype=float).reshape(2, 2, 2)
                acc_sym = np.array(functions["rhs"](u[0], u[1], v[0], v[1]), dtype=float)[2:]
                acc_ciw = surf.geodesic_rhs(np.concatenate([u, v]))[2:]
                row["metric"] = max(row["metric"], _discrepancy(surf.metric(u), g_sym))
                row["christoffel"] = max(row["christoffel"], _discrepancy(gamma, gamma_sym))
                row["acceleration"] = max(row["acceleration"], _discrepancy(acc_ciw, acc_sym))
                row["curvature"] = max(row["curvature"], _discrepancy(k_ciw, float(functions["curvature"](*u))))
        rows[key] = row
    cylinder = gj.surface("cylinder")
    u_cyl = sample_points("cylinder")[0][0]
    normal_curvature = float(cylinder.second(u_cyl)[0] @ cylinder.unit_normal3(u_cyl) / cylinder.metric(u_cyl)[0, 0])
    table = {"seed": gj.SEED, "samples_per_chart": SAMPLES_PER_CHART, "boxes": SAMPLE_BOXES,
             "charts": {key: gj.describe(key) for key in T001_CHARTS}, "discrepancies": rows,
             "cylinder_normal_curvature_phi": normal_curvature,
             "discrepancy_definition": "max |a - b| / max(1, |b|) over samples and components"}
    ctx.artifact_json("comparison.json", _plain(table))
    if have_sympy:
        ctx.artifact_text("derivations.txt", "\n".join(texts))
        ctx.artifact_text("derivations.tex", "\n".join(latex))
    ctx.artifact_text("curvature-along-paths.svg", _curvature_figure(ctx, have_sympy))

    generator = {"name": "seeded chart samples", "seed": gj.SEED, "samples_per_chart": SAMPLES_PER_CHART,
                 "charts": list(T001_CHARTS)}
    findings = []
    if have_sympy:
        sympy_version = gj.optional_version("sympy")
        geodesic = max(max(r["metric"], r["christoffel"], r["acceleration"]) for r in rows.values())
        findings.append(finding(
            "Sympy-derived metrics, Christoffel symbols and geodesic equations match ciw.lab.surfaces on nine charts",
            "numerical", {k: max(r["metric"], r["christoffel"], r["acceleration"]) for k, r in rows.items()},
            {"generator": generator, "independent_check": gj.independent(
                gj.check("analytic", "sympy symbolic metric, Christoffel symbols and geodesic accelerations "
                         "(max relative discrepancy over nine charts)", geodesic, 1e-10),
                "ciw.lab.surfaces", "sympy", checker_revision=sympy_version)},
            uncertainty={"floating_point": "binary64 evaluation of both sides"}, tolerance=TOL_SMALL))
        curvature = max(r["curvature"] for r in rows.values())
        findings.append(finding(
            "Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts",
            "numerical", {k: r["curvature"] for k, r in rows.items()},
            {"generator": generator, "independent_check": gj.independent(
                gj.check("analytic", "sympy Brioschi formula (first fundamental form only)", curvature, 1e-10),
                "ciw.lab.surfaces", "sympy", checker_revision=sympy_version)},
            tolerance=TOL_SMALL))
    consistency = {k: {name: r[name] for name in ("symmetry", "compatibility", "metric_derivatives")}
                   for k, r in rows.items()}
    findings.append(finding(
        "ciw.lab.surfaces Christoffel symbols are symmetric and metric-compatible and its exact metric derivatives "
        "match finite differences", "numerical", consistency,
        {"generator": generator, "checks": [
            gj.check("invariant", "Gamma^k_ij - Gamma^k_ji", max(r["symmetry"] for r in rows.values()), 1e-14),
            gj.check("invariant", "nabla g = 0 residual", max(r["compatibility"] for r in rows.values()), 1e-12),
            gj.check("self_convergence", "central differences of g with step 1e-5",
                     max(r["metric_derivatives"] for r in rows.values()), 1e-6)]},
        tolerance={"abs": 1e-7, "rel": 0.0}))
    findings.append(finding(
        "Intrinsic curvature from finite-differenced Christoffel symbols matches the ciw Gaussian curvature",
        "numerical", {k: r["intrinsic_curvature"] for k, r in rows.items()},
        {"generator": generator, "checks": [gj.check(
            "self_convergence", "Riemann tensor from central differences (step 1e-4) of ciw Christoffel symbols",
            max(r["intrinsic_curvature"] for r in rows.values()), 1e-6)]},
        tolerance={"abs": 1e-6, "rel": 0.0}))
    polar = {k: {"max_abs_christoffel": rows[k]["max_abs_christoffel"],
                 "max_abs_intrinsic_curvature": rows[k]["max_abs_intrinsic_curvature"]} for k in gj.POLAR_KEYS}
    polar_checks = [gj.check("invariant", "largest |Gamma| on the polar charts",
                             min(p["max_abs_christoffel"] for p in polar.values()), 0.1, "ge")]
    if have_sympy:
        nonzero = sum(not rows[k]["symbolic_curvature_is_zero"] for k in gj.POLAR_KEYS)
        polar_checks.append(gj.check("exact_arithmetic", "polar charts whose Brioschi curvature does not simplify "
                                     "to 0", nonzero, 0.0))
    else:
        polar_checks.append(gj.check("self_convergence", "finite-difference intrinsic curvature on the polar charts",
                                     max(p["max_abs_intrinsic_curvature"] for p in polar.values()), 1e-6))
    findings.append(finding(
        "Nonzero Christoffel symbols do not imply curvature: the polar charts of the plane and cylinder are flat",
        "mathematical", polar, {"generator": generator, "checks": polar_checks},
        tolerance={"abs": 1e-6, "rel": 1e-9},
        counterexample={"statement": "A chart with nonzero Christoffel symbols describes a curved surface",
                        "witness": {"chart": "plane-polar", "christoffel": "Gamma^r_tt = -r, Gamma^t_rt = 1/r",
                                    "gaussian_curvature": 0.0}}))
    cylinder_checks = []
    if have_sympy:
        derived = _derived(ctx, "cylinder")
        nonzero = sum(derived["christoffel"][k][i][j] != 0 for k in range(2) for i in range(2) for j in range(2))
        cylinder_checks.append(gj.check("exact_arithmetic", "nonzero symbolic Christoffel symbols of the cylinder",
                                        nonzero, 0.0))
    findings.append(finding(
        "An extrinsically curved surface can have identically zero Christoffel symbols: the cylinder in (phi, z)",
        "mathematical", {"max_abs_christoffel": rows["cylinder"]["max_abs_christoffel"],
                         "normal_curvature_phi": normal_curvature},
        {"generator": generator, "checks": cylinder_checks + [
            gj.check("analytic", "max |Gamma| of the cylinder (phi, z) chart in binary64",
                     rows["cylinder"]["max_abs_christoffel"], 1e-15),
            gj.check("invariant", "|II(e_phi, e_phi)| / g_phiphi of the unit cylinder", abs(normal_curvature), 0.5,
                     "ge")]},
        tolerance=TOL_VALUE,
        counterexample={"statement": "A surface that bends in space has nonzero Christoffel symbols in every chart",
                        "witness": {"chart": "cylinder (phi, z)", "normal_curvature_phi": normal_curvature}}))
    findings.append(finding(
        "The geodesic equation u''^k = -Gamma^k_ij u'^i u'^j and its catalogue specializations follow from the "
        "first variation of length", "mathematical", "derived",
        {"derivation": f"{DOC}, section T001 (metric, Christoffel symbols and curvature of each chart by hand)"}))

    state = "completed" if have_sympy else "partial"
    worst = max(max(r.get("metric", 0), r.get("christoffel", 0), r.get("acceleration", 0), r.get("curvature", 0))
                for r in rows.values())
    return _outcome(
        state, findings,
        hypothesis=("The metric, Christoffel symbols, geodesic equations and Gaussian curvature derived symbolically "
                    "from each chart's embedding (or intrinsic metric) coincide with the exact-derivative "
                    "implementation in ciw.lab.surfaces."),
        mathematical_model=("g_ij = X_i . X_j (or the declared intrinsic metric), Gamma^k_ij = 1/2 g^kl (d_i g_jl + "
                            "d_j g_il - d_l g_ij), u''^k = -Gamma^k_ij u'^i u'^j, K from the Brioschi formula "
                            "(sympy) and from the Riemann tensor of finite-differenced Christoffel symbols."),
        input_data=[f"Charts: {', '.join(T001_CHARTS)} with catalogue parameters",
                    f"{SAMPLES_PER_CHART} seeded points and velocities per chart (PCG64 seed {gj.SEED} + 1000 i)"],
        observation_model=("Relative discrepancy max |a - b| / max(1, |b|) over samples and components, evaluated in "
                           "binary64."),
        expected_invariant=("Symbolic and numeric quantities agree to rounding; Gamma is symmetric; nabla g = 0; "
                            "the intrinsic curvature equals the second-fundamental-form curvature (Theorema "
                            "Egregium)."),
        experiment=("Derive each chart with sympy (when installed), lambdify, and compare with ciw.lab.surfaces at "
                    "seeded points; independently check symmetry, metric compatibility, metric derivatives and "
                    "intrinsic curvature from ciw alone; record the flat polar charts and the cylinder as "
                    "counterexamples separating Christoffel symbols from curvature."),
        numerical_result=(f"Largest sympy/ciw discrepancy {_fmt(worst)}; largest intrinsic-curvature discrepancy "
                          f"{_fmt(max(r['intrinsic_curvature'] for r in rows.values()))}."
                          if have_sympy else
                          "sympy unavailable: symbolic comparison not run; self-consistency and intrinsic-curvature "
                          f"checks passed (largest intrinsic discrepancy "
                          f"{_fmt(max(r['intrinsic_curvature'] for r in rows.values()))})."),
        uncertainty=("Binary64 rounding of both sides (about 1e-15 relative); finite-difference checks carry a "
                     "truncation error near 1e-8."),
        failure_modes_checked=["Christoffel index order (symmetry and compatibility)",
                               "sign convention of the Riemann tensor (sphere K = +1, hyperbolic plane K = -1)",
                               "chart singularities excluded from the sampling boxes",
                               "intrinsic versus extrinsic curvature (cylinder, polar charts)"],
        unresolved_assumptions=(["sympy simplification is trusted to be correct; it is an independent implementation, "
                                 "not a proof checker"] if have_sympy else
                                ["sympy is not installed: the symbolic derivation relies on the hand derivation in "
                                 f"{DOC}"]) + ["Only 8 sample points per chart are compared"],
        recommended_next_task="T002: build high-precision reference solutions on the same equations")


# ---------------------------------------------------------------------------
# T002: high-precision references
# ---------------------------------------------------------------------------
RICHARDSON_STEPS = 200


def reference(ctx, key: str) -> dict:
    """End position (embedded or hyperbolic chart) and transfer matrix of a declared path (memoized).

    Closed forms for constant-curvature charts; otherwise an mpmath reference
    on sympy-derived equations, falling back to scipy DOP853 and then to a
    finer ciw Richardson extrapolation when optional modules are missing.
    """
    def compute():
        spec = gj.path(key)
        surf = gj.surface(spec.surface)
        exact = gj.exact_position(key, [spec.length])
        if exact is not None and spec.surface in gj.CLOSED_FORM_KEYS:
            k = _constant_curvature_of(key)
            a, ap, b, bp = jacobi.constant_curvature(k, [spec.length])
            return {"kind": "closed_form", "position": exact[0].tolist(),
                    "matrix": [[float(a[0]), float(b[0])], [float(ap[0]), float(bp[0])]], "curvature": k,
                    "error_estimate": 0.0}
        if ctx.available("module:sympy") and ctx.available("module:mpmath"):
            result = gj.mp_reference(key)
        elif ctx.available("module:scipy"):
            result = gj.scipy_reference(key)
        else:
            end, estimate = integrators.richardson_rk4(jacobi.rhs(surf), gj.start_state(key), spec.length, 800)
            result = {"kind": "richardson_rk4", "state": end.tolist(), "error_estimate": estimate, "steps": 800}
        state = result["state"]
        result["position"] = gj.position(spec.surface, state[:2]).tolist()
        result["matrix"] = [[state[4], state[6]], [state[5], state[7]]]
        return result

    return ctx.memo(("gj-reference", key), compute)


def _richardson(ctx, key):
    def compute():
        spec = gj.path(key)
        end, estimate = integrators.richardson_rk4(jacobi.rhs(gj.surface(spec.surface)), gj.start_state(key),
                                                   spec.length, RICHARDSON_STEPS)
        return end, estimate
    return ctx.memo(("gj-richardson", key, RICHARDSON_STEPS), compute)


def _scipy(ctx, key):
    return ctx.memo(("gj-scipy", key), lambda: gj.scipy_reference(key))


def _end_gap(key, state, ref) -> dict:
    spec = gj.path(key)
    state = np.asarray(state, dtype=float)
    position = float(np.linalg.norm(gj.position(spec.surface, state[:2]) - np.asarray(ref["position"])))
    matrix = float(np.max(np.abs(np.array([[state[4], state[6]], [state[5], state[7]]]) - np.asarray(ref["matrix"]))))
    return {"position": position, "matrix": matrix, "max": max(position, matrix)}


def _clairaut_mp(ref, surf):
    import mpmath

    with mpmath.workdps(gj.MP_DPS):
        big, small = mpmath.mpf(surf.major), mpmath.mpf(surf.minor)
        start = [mpmath.mpf(float(x)) for x in gj.start_state("torus")]

        def value(y):
            return (big + small * mpmath.cos(y[1])) ** 2 * y[2]

        return float(abs(value(ref["mpf_state"]) - value(start)))


@task("T002", changed_files=CHANGED,
      regression_tests=(_test("test_t002_references_agree"), _test("test_t002_without_optional_modules")))
def high_precision_references(ctx):
    have_scipy = ctx.available("module:scipy")
    have_mp = ctx.available("module:sympy") and ctx.available("module:mpmath")
    rows = {}
    for key in gj.STANDARD:
        ref = reference(ctx, key)
        end, estimate = _richardson(ctx, key)
        row = {"reference_kind": ref["kind"], "reference_position": ref["position"], "reference_matrix": ref["matrix"],
               "reference_error_estimate": ref["error_estimate"], "richardson_error_estimate": estimate,
               "ciw_vs_reference": _end_gap(key, end, ref)}
        if ref["kind"] == "mpmath":
            row.update(reference_digits=ref["digits"], reference_speed_squared_minus_one=ref["speed_squared_minus_one"],
                       reference_local_indicator=ref["local_indicator"])
        if have_scipy:
            sci = _scipy(ctx, key)
            row["scipy_vs_reference"] = _end_gap(key, sci["state"], ref)
            row["ciw_vs_scipy"] = _end_gap(key, end, {"position": gj.position(gj.path(key).surface,
                                                                               np.asarray(sci["state"][:2])).tolist(),
                                                      "matrix": [[sci["state"][4], sci["state"][6]],
                                                                 [sci["state"][5], sci["state"][7]]]})
            row["scipy_nfev"] = sci["nfev"]
        rows[key] = row

    torus = gj.surface("torus")
    start = gj.start_state("torus")
    c0 = torus.clairaut(start[:2], start[2:4])
    end, _ = _richardson(ctx, "torus")
    clairaut = {"ciw_richardson": abs(torus.clairaut(end[:2], end[2:4]) - c0)}
    if rows["torus"]["reference_kind"] == "mpmath":
        clairaut["mpmath_reference"] = _clairaut_mp(reference(ctx, "torus"), torus)
    curves = []
    for steps in (50, 100, 200):
        tr = gj.transfer(ctx, "torus", "rk4", steps)
        drift = [abs(torus.clairaut(y[:2], y[2:4]) - c0) for y in tr.states]
        clairaut[f"ciw_rk4_{steps}_max"] = float(max(drift))
        curves.append((f"RK4 N={steps}", tr.s, drift))
    ctx.artifact_json("references.json", _plain({
        "richardson_steps": [RICHARDSON_STEPS, 2 * RICHARDSON_STEPS], "mp_dps": gj.MP_DPS,
        "mp_macro_steps": list(gj.MP_MACRO_STEPS), "paths": {k: gj.path(k).as_dict() for k in gj.STANDARD},
        "rows": rows, "clairaut": clairaut, "clairaut_initial": c0}))
    series = [("ciw Richardson vs reference", range(1, len(rows) + 1),
               [max(r["ciw_vs_reference"]["max"], 1e-17) for r in rows.values()])]
    if have_scipy:
        series.append(("scipy DOP853 vs reference", range(1, len(rows) + 1),
                       [max(r["scipy_vs_reference"]["max"], 1e-17) for r in rows.values()]))
    ctx.artifact_text("agreement.svg", svg.line_plot(
        series, title="End-state agreement with the reference (1 = " + ", ".join(rows) + ")",
        xlabel="path index", ylabel="max(position, transfer) gap", logy=True))
    ctx.artifact_text("clairaut.svg", svg.line_plot(
        [(n, s, np.maximum(d, 1e-17)) for n, s, d in curves], title="Torus Clairaut drift |C(s) - C(0)|",
        xlabel="arclength s", ylabel="|rho^2 phi' - C0|", logy=True, markers=False))

    findings = []
    closed = [k for k in gj.STANDARD if rows[k]["reference_kind"] == "closed_form"]
    closed_basis = {"checks": [gj.check("analytic", f"closed-form geodesic and constant-curvature transfer on {k}",
                                        rows[k]["ciw_vs_reference"]["max"], 1e-10) for k in closed]}
    if have_scipy:
        closed_basis["independent_check"] = gj.independent(
            gj.check("high_precision", "scipy DOP853 rtol 1e-13 on the same right-hand side",
                     max(rows[k]["ciw_vs_scipy"]["max"] for k in closed), 1e-9),
            "ciw.lab.integrators.richardson_rk4", "scipy.integrate.solve_ivp(DOP853)",
            checker_revision=gj.optional_version("scipy"))
    findings.append(finding(
        "ciw Richardson RK4 end states match closed-form geodesics and transfer matrices on the six closed-form charts",
        "numerical", {k: rows[k]["ciw_vs_reference"]["max"] for k in closed}, closed_basis, tolerance=TOL_SMALL))
    for key in gj.VARIABLE_KEYS:
        row = rows[key]
        gap = row["ciw_vs_reference"]["max"]
        value = {"reference_end_state": reference(ctx, key)["state"], "ciw_minus_reference": gap}
        if row["reference_kind"] == "mpmath":
            checks = [gj.check("high_precision", "mpmath macro-step 10 vs 20 at 34 digits",
                               row["reference_error_estimate"], 1e-18)]
            if have_scipy:
                checks.append(gj.check("high_precision", "scipy DOP853 vs the mpmath reference",
                                       row["scipy_vs_reference"]["max"], 1e-10))
            basis = {"checks": checks, "independent_check": gj.independent(
                gj.check("high_precision", f"34-digit Gragg-Bulirsch-Stoer reference on sympy-derived {key} equations",
                         gap, 1e-9),
                "ciw.lab.integrators.richardson_rk4", "mpmath", checker_revision=gj.optional_version("mpmath"))}
            claim = f"ciw Richardson RK4 matches the 34-digit mpmath reference on the {key} path"
        elif row["reference_kind"] == "scipy":
            basis = {"independent_check": gj.independent(
                gj.check("high_precision", f"scipy DOP853 rtol 1e-13 on {key}", gap, 1e-9),
                "ciw.lab.integrators.richardson_rk4", "scipy.integrate.solve_ivp(DOP853)",
                checker_revision=gj.optional_version("scipy"))}
            claim = f"ciw Richardson RK4 matches scipy DOP853 on the {key} path (mpmath reference unavailable)"
        else:
            basis = {"checks": [gj.check("self_convergence", f"ciw Richardson RK4 at 800 vs 200 steps on {key}",
                                         gap, 1e-9)]}
            claim = f"ciw Richardson RK4 is self-convergent on the {key} path (no independent reference available)"
        findings.append(finding(claim, "numerical", value, basis, tolerance=TOL_VALUE))
    clairaut_checks = [gj.check("invariant", "Clairaut drift of the ciw Richardson end state",
                                clairaut["ciw_richardson"], 1e-10)]
    if "mpmath_reference" in clairaut:
        clairaut_checks.append(gj.check("invariant", "Clairaut drift of the mpmath reference end state",
                                        clairaut["mpmath_reference"], 1e-20))
    findings.append(finding(
        "Clairaut's integral rho^2 phi' is conserved along the torus reference path", "numerical",
        {name: clairaut[name] for name in sorted(clairaut)}, {"checks": clairaut_checks}, tolerance=TOL_SMALL))

    state = "completed" if (have_scipy and have_mp) else "partial"
    missing = [name for name, ok in (("scipy", have_scipy), ("sympy+mpmath", have_mp)) if not ok]
    worst = max(r["ciw_vs_reference"]["max"] for r in rows.values())
    return _outcome(
        state, findings,
        hypothesis=("Every declared standard path has a reference end state accurate far beyond the integrators under "
                    "test: closed forms on constant-curvature charts and a 34-digit extrapolated integration on the "
                    "saddle, torus and gaussian bump."),
        mathematical_model=("Great circle X(s) = cos(s/R) X0 + R sin(s/R) T0; helix and polar lines through the flat "
                            "development; hyperbolic semicircles; transfer matrices cn_K, sn_K; otherwise the "
                            "sympy-derived geodesic + Jacobi system integrated by Gragg-Bulirsch-Stoer in mpmath."),
        input_data=[f"Standard paths: {', '.join(f'{k} (L={gj.path(k).length})' for k in gj.STANDARD)}",
                    "Start states are the binary64 states used by the ciw integrators, converted exactly"],
        observation_model=("Euclidean distance of embedded end points (chart distance on the hyperbolic plane) and "
                           "max-abs difference of the 2x2 transfer matrix."),
        expected_invariant=("ciw Richardson RK4 (200/400 steps) and scipy DOP853 (rtol 1e-13) agree with each "
                            "reference to about 1e-12; the mpmath reference changes by less than 1e-18 between 10 "
                            "and 20 macro-steps; Clairaut's integral is conserved on the torus."),
        experiment=("Build each reference, integrate the ciw joint geodesic + Jacobi system with Richardson RK4 and "
                    "with scipy DOP853, and compare end states; check the torus Clairaut integral."),
        numerical_result=(f"Largest ciw-vs-reference gap {_fmt(worst)}; variable-curvature references "
                          + ", ".join(f"{k} {rows[k]['reference_kind']}"
                                      + ("" if rows[k]["reference_error_estimate"] is None else
                                         f" (self-estimate {_fmt(rows[k]['reference_error_estimate'])})")
                                      for k in gj.VARIABLE_KEYS)
                          + f"; Clairaut drift of the ciw end state {_fmt(clairaut['ciw_richardson'])}."),
        uncertainty=("Reference uncertainty is the 10-vs-20 macro-step difference at 34 digits (mpmath) or rounding "
                     "of closed forms (about 1e-16); the ciw Richardson error estimate is retained per path."),
        failure_modes_checked=["reference and integrator start from the same binary64 state",
                               "reference equations derived independently of ciw.lab.surfaces (sympy)",
                               "extrapolation self-consistency (macro-step halving)",
                               "Clairaut invariant on the torus"],
        unresolved_assumptions=(["Unavailable optional modules: " + ", ".join(missing)] if missing else [])
        + ["The mpmath reference is an extrapolated integration, not a closed form; its error estimate is empirical"],
        recommended_next_task="T003: measure Euler, midpoint, RK4 and adaptive orders against these references")


# ---------------------------------------------------------------------------
# T003/T004: integrator orders and unit-speed drift (one shared study)
# ---------------------------------------------------------------------------
ORDER_STEPS = {"euler": (64, 128, 256, 512), "midpoint": (32, 64, 128, 256), "rk4": (24, 48, 96, 192)}
ORDER_TOLERANCE = {"euler": 0.1, "midpoint": 0.1, "rk4": 0.25}
ADAPTIVE_RTOL = (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
DRIFT_TOLERANCE = {"euler": 0.1, "midpoint": 0.15, "rk4": 0.3}


def convergence_study(ctx) -> dict:
    """Endpoint error and speed drift of every method, step and standard path (geodesic state only)."""
    def compute():
        table, curves = {}, {}
        for key in gj.STANDARD:
            spec = gj.path(key)
            surf = gj.surface(spec.surface)
            target = np.asarray(reference(ctx, key)["position"])
            y0 = gj.start_state(key)[:4]
            row = {}
            for method, counts in ORDER_STEPS.items():
                entries = []
                for steps in counts:
                    s, states = integrators.integrate_fixed(surf.geodesic_rhs, y0, spec.length, steps, method)
                    drift = _speed_drift(surf, states)
                    entries.append({"steps": steps, "h": spec.length / steps,
                                    "error": float(np.linalg.norm(gj.position(spec.surface, states[-1, :2]) - target)),
                                    "max_speed_drift": float(np.max(np.abs(drift))), "final_speed_drift": float(drift[-1])})
                    if key == "sphere" and steps == counts[1]:
                        curves[method] = (s, np.abs(drift))
                row[method] = entries
            adaptive = []
            for rtol in ADAPTIVE_RTOL:
                _, states, stats = integrators.integrate_adaptive(surf.geodesic_rhs, y0, spec.length, rtol=rtol,
                                                                  atol=rtol)
                adaptive.append({"rtol": rtol, "atol": rtol, **stats,
                                 "error": float(np.linalg.norm(gj.position(spec.surface, states[-1, :2]) - target)),
                                 "max_speed_drift": float(np.max(np.abs(_speed_drift(surf, states))))})
            row["adaptive"] = adaptive
            table[key] = row
        return {"table": table, "sphere_drift_curves": curves}
    return ctx.memo("gj-convergence", compute)


def _slopes(entries, quantity):
    return gj.slope([e["h"] for e in entries], [e[quantity] for e in entries])


@task("T003", changed_files=CHANGED, regression_tests=(_test("test_t003_integrator_orders"),))
def integrator_orders(ctx):
    study = convergence_study(ctx)["table"]
    orders = {method: {k: _slopes(study[k][method], "error") for k in CURVED_CHARTS} for method in ORDER_STEPS}
    pairwise = {method: {k: [gj.slope([a["h"], b["h"]], [a["error"], b["error"]])
                             for a, b in zip(study[k][method], study[k][method][1:])] for k in CURVED_CHARTS}
                for method in ORDER_STEPS}
    effective = {k: -gj.slope([e["function_evaluations"] for e in study[k]["adaptive"]],
                              [e["error"] for e in study[k]["adaptive"]]) for k in CURVED_CHARTS}
    proportional = {k: gj.slope([e["rtol"] for e in study[k]["adaptive"]], [e["error"] for e in study[k]["adaptive"]])
                    for k in CURVED_CHARTS}
    flat = {k: max(e["error"] for m in ORDER_STEPS for e in study[k][m]) for k in FLAT_CARTESIAN}
    ctx.artifact_json("orders.json", _plain({"steps": ORDER_STEPS, "adaptive_rtol": ADAPTIVE_RTOL,
                                             "fitted_orders": orders, "pairwise_orders": pairwise,
                                             "adaptive_effective_order": effective,
                                             "adaptive_tolerance_exponent": proportional,
                                             "flat_cartesian_max_error": flat, "table": study}))
    for key in CURVED_CHARTS:
        ctx.artifact_text(f"orders-{key}.svg", svg.line_plot(
            [(m, [e["h"] for e in study[key][m]], [e["error"] for e in study[key][m]]) for m in ORDER_STEPS],
            title=f"Global endpoint error on {key}", xlabel="step h", ylabel="endpoint error", logx=True, logy=True))
    ctx.artifact_text("adaptive.svg", svg.line_plot(
        [(k, [e["function_evaluations"] for e in study[k]["adaptive"]], [e["error"] for e in study[k]["adaptive"]])
         for k in CURVED_CHARTS], title="Dormand-Prince error against function evaluations",
        xlabel="function evaluations", ylabel="endpoint error", logx=True, logy=True))

    generator = {"name": "declared standard geodesics", "paths": list(gj.STANDARD)}
    findings = []
    names = {"euler": "Explicit Euler", "midpoint": "Explicit midpoint", "rk4": "Classical RK4"}
    for method, p in NOMINAL_ORDER.items():
        findings.append(finding(
            f"{names[method]} global endpoint error converges at order {p} on every curved chart", "numerical",
            orders[method], {"generator": generator, "checks": [
                gj.check("analytic", f"nominal order {p} on {k} (steps {ORDER_STEPS[method]})",
                         orders[method][k] - p, ORDER_TOLERANCE[method]) for k in CURVED_CHARTS]},
            uncertainty={"fit": "least-squares log-log slope over four halvings; pairwise slopes retained"},
            tolerance=TOL_RATE))
    findings.append(finding(
        "Adaptive Dormand-Prince error falls with function evaluations at an effective order near 5", "numerical",
        effective, {"generator": generator, "checks": [
            gj.check("analytic", f"effective order 5 on {k} (rtol 1e-8..1e-12)", effective[k] - 5, 1.0)
            for k in CURVED_CHARTS] + [gj.check("analytic", "median effective order across charts",
                                                float(np.median(list(effective.values()))) - 5, 0.5)]},
        uncertainty={"fit": "the start-step ramp of the controller biases loose tolerances upward, so only "
                            "rtol <= 1e-8 is fitted; per-chart orders scatter by about 0.6"},
        tolerance={"abs": 0.05, "rel": 0.0}))
    findings.append(finding(
        "Adaptive Dormand-Prince endpoint error is proportional to the requested tolerance", "numerical",
        proportional, {"generator": generator, "checks": [
            gj.check("analytic", f"error ~ rtol^1 on {k}", proportional[k] - 1, 0.3) for k in CURVED_CHARTS]},
        tolerance={"abs": 0.05, "rel": 0.0}))
    findings.append(finding(
        "No convergence order is observable on flat Cartesian charts: every method is exact to rounding there",
        "numerical", flat, {"generator": generator, "checks": [
            gj.check("analytic", f"straight-line geodesic on {k} (Gamma = 0)", flat[k], 1e-12) for k in FLAT_CARTESIAN]},
        tolerance=TOL_SMALL,
        counterexample={"statement": "Every integrator exhibits its nominal convergence order on every surface",
                        "witness": {"charts": list(FLAT_CARTESIAN), "max_endpoint_error": flat,
                                    "reason": "Gamma vanishes, so Euler, midpoint and RK4 reproduce u(s) = u0 + s v0"}}))
    return _outcome(
        "completed", findings,
        hypothesis=("On charts with nonzero Christoffel symbols the global endpoint error of Euler, midpoint and RK4 "
                    "scales like h^1, h^2 and h^4, and the Dormand-Prince error scales like (evaluations)^-5 and "
                    "like the requested tolerance."),
        mathematical_model=("Global error e(h) = C h^p + O(h^(p+1)) for a p-th order one-step method on a smooth "
                            "ODE; for DP5(4) with per-step error control, h ~ tol^(1/5) and e ~ tol."),
        input_data=[f"Standard paths on {', '.join(gj.STANDARD)}",
                    f"Steps {ORDER_STEPS}; adaptive rtol = atol in {list(ADAPTIVE_RTOL)}",
                    "References from T002 (closed form or mpmath)"],
        observation_model="Euclidean distance of the embedded end point (chart distance on the hyperbolic plane).",
        expected_invariant="Fitted log-log slopes within declared tolerances of the nominal orders.",
        experiment=("Integrate the geodesic state (u, v) with each fixed-step method at four halvings and the "
                    f"adaptive method at {len(ADAPTIVE_RTOL)} tolerances; fit log-log slopes against the reference "
                    "end point."),
        numerical_result=("Fitted orders: " + "; ".join(
            f"{m} " + ", ".join(f"{k} {_fmt(v, 4)}" for k, v in orders[m].items()) for m in ORDER_STEPS)
            + "; adaptive effective orders " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in effective.items()) + "."),
        uncertainty=("Slopes are least-squares fits over four points; the pre-asymptotic bias is visible in the "
                     "retained pairwise slopes (largest on RK4 at the coarsest step)."),
        failure_modes_checked=["flat Cartesian charts where every method is exact (recorded as a counterexample)",
                               "RK4 errors kept above the rounding floor (smallest near 1e-12)",
                               "reference accuracy exceeds the smallest measured error by several digits"],
        unresolved_assumptions=["The adaptive effective order depends on the step-size controller and its start "
                                "step; the retained numbers are for integrators.integrate_adaptive only"],
        recommended_next_task="T004: measure unit-speed drift of the same integrations without renormalization")


# T004 -----------------------------------------------------------------------
NORMALIZING_CALLS = frozenset({"norm", "normalize", "normalized", "speed_squared", "unit_tangent", "orthonormal_frame",
                               "normal", "inner", "hypot"})


def state_update_code():
    """Every function through which an integrated state passes after its initial data is set."""
    return (integrators.step_euler, integrators.step_midpoint, integrators.step_rk4, integrators.integrate_fixed,
            integrators.richardson_rk4, integrators.integrate_adaptive, jacobi.rhs, jacobi.transfer,
            surfaces.Surface.geodesic_rhs, surfaces.Surface.christoffel)


def normalization_calls(functions=None) -> list:
    """Calls that could renormalize a state, and divisions by a computed magnitude, in the given source."""
    found = []
    for function in functions or state_update_code():
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
                if name in NORMALIZING_CALLS:
                    found.append(f"{function.__qualname__}: call {name}")
            divisor = None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                divisor = node.right
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
                divisor = node.value
            if isinstance(divisor, ast.Call):
                name = divisor.func.attr if isinstance(divisor.func, ast.Attribute) else getattr(divisor.func, "id", None)
                if name in ("sqrt", "norm", "hypot", "abs"):
                    found.append(f"{function.__qualname__}: division by {name}")
    return found


def renormalized_euler(surf, y0, length, steps):
    """Deliberately faulty comparison: Euler with the velocity rescaled to unit speed after every step."""
    h = length / steps
    y = np.array(y0, dtype=float)
    states = [y.copy()]
    for _ in range(steps):
        y = integrators.step_euler(surf.geodesic_rhs, y, h)
        y[2:4] = y[2:4] / math.sqrt(surf.speed_squared(y[:2], y[2:4]))
        states.append(y.copy())
    return np.array(states)


@task("T004", changed_files=CHANGED,
      regression_tests=(_test("test_t004_speed_drift_and_no_renormalization"),
                        _test("test_integrator_code_path_has_no_normalization")))
def unit_speed_drift(ctx):
    study = convergence_study(ctx)
    table = study["table"]
    drift_orders = {m: {k: _slopes(table[k][m], "max_speed_drift") for k in CURVED_CHARTS} for m in ORDER_STEPS}
    flat_drift = {k: max(e["max_speed_drift"] for m in ORDER_STEPS for e in table[k][m]) for k in FLAT_CARTESIAN}

    speed, speed2 = 1.3, 1.69
    nonunit = {}
    for key in ("sphere", "torus", "hyperbolic-plane"):
        spec = gj.path(key)
        surf = gj.surface(spec.surface)
        y0 = gj.start_state(key)[:4].copy()
        y0[2:4] *= speed
        row = {}
        for method, steps in (("rk4", 128), ("euler", 128)):
            _, states = integrators.integrate_fixed(surf.geodesic_rhs, y0, spec.length, steps, method)
            g = np.array([surf.speed_squared(y[:2], y[2:4]) for y in states])
            row[method] = {"max_abs_g_minus_1.69": float(np.max(np.abs(g - speed2))),
                           "min_abs_g_minus_1": float(np.min(np.abs(g - 1.0))), "final_g": float(g[-1])}
        nonunit[key] = row

    # Exponential growth y' = y: a state normalization would cap |y| instead of reaching e^L |y0|.
    y0 = np.array([1.0, 0.5, 0.0, 2.0])
    growth = {}
    _, states = integrators.integrate_fixed(lambda y: y, y0, 2.0, 64, "rk4")
    growth["rk4_64"] = float(abs(np.linalg.norm(states[-1]) / np.linalg.norm(y0) - math.e ** 2) / math.e ** 2)
    _, states, _ = integrators.integrate_adaptive(lambda y: y, y0, 2.0, rtol=1e-10, atol=1e-12)
    growth["adaptive_1e-10"] = float(abs(np.linalg.norm(states[-1]) / np.linalg.norm(y0) - math.e ** 2) / math.e ** 2)
    calls = normalization_calls()

    sphere = gj.surface("sphere")
    spec = gj.path("sphere")
    target = np.asarray(reference(ctx, "sphere")["position"])
    faulty = renormalized_euler(sphere, gj.start_state("sphere")[:4], spec.length, 128)
    plain = [e for e in table["sphere"]["euler"] if e["steps"] == 128][0]
    counter = {"steps": 128, "renormalized_max_speed_drift": float(np.max(np.abs(_speed_drift(sphere, faulty)))),
               "renormalized_endpoint_error": float(np.linalg.norm(sphere.embedding(faulty[-1, :2]) - target)),
               "plain_endpoint_error": plain["error"], "plain_max_speed_drift": plain["max_speed_drift"]}

    ctx.artifact_json("speed-drift.json", _plain({
        "drift_orders": drift_orders, "flat_cartesian_max_drift": flat_drift, "nonunit_start": nonunit,
        "exponential_growth_relative_error": growth, "normalization_calls": calls,
        "scanned_functions": [f"{f.__module__}.{f.__qualname__}" for f in state_update_code()],
        "renormalized_euler_counterexample": counter,
        "table": {k: {m: [{"h": e["h"], "max_speed_drift": e["max_speed_drift"], "final_speed_drift":
                           e["final_speed_drift"]} for e in table[k][m]] for m in ORDER_STEPS} for k in gj.STANDARD}}))
    for method in ORDER_STEPS:
        ctx.artifact_text(f"drift-{method}.svg", svg.line_plot(
            [(k, [e["h"] for e in table[k][method]], [e["max_speed_drift"] for e in table[k][method]])
             for k in CURVED_CHARTS], title=f"Unit-speed drift max|g(v,v) - 1|, {method}", xlabel="step h",
            ylabel="max |g(v,v) - 1|", logx=True, logy=True))
    curves = study["sphere_drift_curves"]
    ctx.artifact_text("drift-along-sphere.svg", svg.line_plot(
        [(f"{m} N={ORDER_STEPS[m][1]}", s, np.maximum(d, 1e-17)) for m, (s, d) in curves.items()],
        title="Speed drift along the sphere path (no renormalization)", xlabel="arclength s",
        ylabel="|g(v,v) - 1|", logy=True, markers=False))

    generator = {"name": "declared standard geodesics", "paths": list(gj.STANDARD)}
    findings = [finding(
        "Unit-speed drift max|g(v,v) - 1| scales like h^p for Euler, midpoint and RK4 on every curved chart",
        "numerical", drift_orders, {"generator": generator, "checks": [
            gj.check("analytic", f"drift order {NOMINAL_ORDER[m]} for {m} on {k}",
                     drift_orders[m][k] - NOMINAL_ORDER[m], DRIFT_TOLERANCE[m])
            for m in ORDER_STEPS for k in CURVED_CHARTS]}, tolerance=TOL_RATE)]
    findings.append(finding(
        "A non-unit initial speed stays non-unit: g(v,v) remains 1.69 to integrator accuracy", "numerical",
        {k: v["rk4"]["max_abs_g_minus_1.69"] for k, v in nonunit.items()},
        {"generator": {"name": "declared standard geodesics with speed 1.3", "paths": list(nonunit)}, "checks": [
            gj.check("invariant", f"RK4 N=128 |g - 1.69| on {k}", v["rk4"]["max_abs_g_minus_1.69"], 1e-7)
            for k, v in nonunit.items()] + [
            gj.check("invariant", f"distance of g from 1 on {k} (a renormalizer would drive it to 0)",
                     min(v[m]["min_abs_g_minus_1"] for m in v), 0.6, "ge") for k, v in nonunit.items()]},
        tolerance=TOL_SMALL))
    findings.append(finding(
        "The integrator and geodesic/Jacobi right-hand-side code paths contain no state normalization",
        "computational_pipeline", {"normalization_calls": len(calls), "exponential_growth": growth},
        {"checks": [gj.check("invariant", "normalizing calls or divisions by a computed magnitude in "
                             f"{len(state_update_code())} scanned functions", len(calls), 0.0),
                    gj.check("analytic", "RK4 N=64 growth |y(2)|/|y0| against e^2 for y' = y", growth["rk4_64"], 1e-6),
                    gj.check("analytic", "adaptive rtol 1e-10 growth against e^2 for y' = y",
                             growth["adaptive_1e-10"], 1e-8)]},
        tolerance={"abs": 1e-8, "rel": 0.0}))
    findings.append(finding(
        "Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding with a "
        "first-order endpoint error", "numerical", counter,
        {"generator": {"name": "renormalized Euler on the standard sphere path", "steps": 128}, "checks": [
            gj.check("invariant", "renormalized Euler speed drift", counter["renormalized_max_speed_drift"], 1e-13),
            gj.check("analytic", "renormalized Euler endpoint error against the great circle",
                     counter["renormalized_endpoint_error"], 1e-3, "ge")]},
        tolerance={"abs": 1e-9, "rel": 1e-6},
        counterexample={"statement": "A computed geodesic whose speed stays exactly 1 is accurate",
                        "witness": {"surface": "sphere", "method": "Euler with per-step speed renormalization",
                                    "steps": 128, "endpoint_error": counter["renormalized_endpoint_error"]}}))
    findings.append(finding(
        "Speed drift is at rounding on flat Cartesian charts, where every method is exact", "numerical", flat_drift,
        {"generator": generator, "checks": [gj.check("analytic", f"constant metric on {k}", flat_drift[k], 1e-13)
                                            for k in FLAT_CARTESIAN]}, tolerance=TOL_SMALL))
    return _outcome(
        "completed", findings,
        hypothesis=("Without renormalization the speed invariant g(v, v) drifts at the global order of each method, "
                    "and a non-unit initial speed is carried unchanged, showing that nothing resets it."),
        mathematical_model=("g(v, v) is a first integral of the geodesic flow; for a p-th order method its global "
                            "defect is O(h^p). A renormalizing step would force g(v, v) = 1 whatever the initial "
                            "speed."),
        input_data=[f"The T003 integrations (steps {ORDER_STEPS}, adaptive rtol {list(ADAPTIVE_RTOL)})",
                    "Speed-1.3 starts on the sphere, torus and hyperbolic plane; y' = y growth probes"],
        observation_model="max over nodes of |g(v, v) - 1| (or |g - 1.69|) evaluated with the exact metric.",
        expected_invariant="Drift slopes near p; g stays at 1.69; zero normalizing calls in the state-update path.",
        experiment=("Measure drift for every method and step; restart with speed 1.3; scan the state-update source "
                    "for normalizing calls; integrate y' = y; compare a deliberately renormalized Euler."),
        numerical_result=("Drift orders: " + "; ".join(
            f"{m} " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in drift_orders[m].items()) for m in ORDER_STEPS)
            + f"; renormalized Euler endpoint error {_fmt(counter['renormalized_endpoint_error'])} with speed drift "
            f"{_fmt(counter['renormalized_max_speed_drift'])}."),
        uncertainty="Slopes are least-squares fits over four halvings; RK4 sphere drift is still slightly pre-asymptotic.",
        failure_modes_checked=["hidden renormalization (static scan, speed-1.3 start, y' = y growth)",
                               "speed drift mistaken for accuracy (renormalized Euler counterexample)",
                               "flat charts where drift is identically at rounding"],
        unresolved_assumptions=["The static scan covers the listed functions only; code outside them (for example "
                                "initial-state construction, which normalizes the start by design) is not scanned"],
        recommended_next_task="T005: verify the Jacobi separation law on constant-curvature paths")


# ---------------------------------------------------------------------------
# T005: separation law; optional CSG provider
# ---------------------------------------------------------------------------
CONSTANT_PATHS = ("sphere-great-circle", "plane", "cylinder", "hyperbolic-long", "torus-outer-equator",
                  "torus-inner-equator")


def _model_error(tr, k):
    """Relative errors of both columns and their derivatives against cn_K and sn_K."""
    a, ap, b, bp = jacobi.constant_curvature(k, tr.s)
    y = tr.states
    return {"heading": _discrepancy(y[:, 6], b), "heading_rate": _discrepancy(y[:, 7], bp),
            "lateral": _discrepancy(y[:, 4], a), "lateral_rate": _discrepancy(y[:, 5], ap)}


def _csg(ctx):
    """Run the pinned CSG provider once per context on the constant-curvature grids (memoized)."""
    def compute():
        checkout = ctx.providers["csg"]
        try:
            identity = gj.verify_csg_checkout(checkout)
            cases = [{"arclength": _fine_transfer(ctx, key).s.tolist(), "gaussian_curvature": _constant_curvature_of(key)}
                     for key in CONSTANT_PATHS]
            data = gj.run_csg_jacobi(checkout, cases)
            if gj.verify_csg_checkout(checkout) != identity:
                raise gj.ProviderRefusal("CSG_CHANGED_DURING_EXECUTION", "Provider checkout changed during execution")
        except gj.ProviderRefusal as exc:
            return {"refusal": exc.code, "message": str(exc)}
        identity = dict(identity, implementation=gj.CSG_IMPLEMENTATION, python=data["python"], numpy=data["numpy"],
                        subprocess=True)
        return {"identity": identity, "data": data, "paths": list(CONSTANT_PATHS)}
    return ctx.memo("gj-csg", compute)


def _provider_identity(csg):
    from .runner import builtin_identity

    return {"provider": csg["identity"], "ciw": builtin_identity(CHANGED)}


@task("T005", changed_files=CHANGED,
      regression_tests=(_test("test_t005_separation_law"), _test("test_t005_csg_provider_agreement"),
                        _test("test_csg_checkout_refusals")))
def separation_law(ctx):
    rows = {}
    for key in CONSTANT_PATHS:
        k = _constant_curvature_of(key)
        fine = _fine_transfer(ctx, key)
        coarse = gj.transfer(ctx, key, "rk4", _fine(key) // 2)
        err_fine, err_coarse = _model_error(fine, k), _model_error(coarse, k)
        spec = gj.path(key)
        row = {"curvature": k, "length": spec.length, "steps": _fine(key), "errors": err_fine,
               "coarse_errors": err_coarse,
               "curvature_drift": float(np.max(np.abs(fine.curvature_along() - k))),
               "max_error": max(err_fine.values()), "j_head_end": float(fine.states[-1, 6]),
               "j_lat_end": float(fine.states[-1, 4])}
        if key.startswith("torus"):
            row["theta_drift"] = float(np.max(np.abs(fine.points[:, 1] - spec.u0[1])))
        if k != 0:
            row["observed_order"] = math.log2(max(err_coarse.values()) / max(err_fine.values()))
        rows[key] = row
    csg = _csg(ctx) if ctx.available("provider:csg") else None
    provider_rows = None
    if csg and "data" in csg:
        provider_rows = {}
        for key, trace, maps in zip(csg["paths"], csg["data"]["traces"], csg["data"]["maps"]):
            fine = _fine_transfer(ctx, key)
            ciw_phi = np.stack([np.array([[y[4], y[6]], [y[5], y[7]]]) for y in fine.states])
            csg_numeric = np.asarray(maps["numeric"]["matrices"])
            csg_exact = np.asarray(maps["closed_form"]["matrices"])
            a, ap, b, bp = jacobi.constant_curvature(_constant_curvature_of(key), fine.s)
            ciw_exact = np.stack([np.array([[x0, x2], [x1, x3]]) for x0, x1, x2, x3 in zip(a, ap, b, bp)])
            provider_rows[key] = {"ciw_rk4_vs_csg_rk4": _discrepancy(ciw_phi, csg_numeric),
                                  "ciw_closed_form_vs_csg_closed_form": _discrepancy(ciw_exact, csg_exact),
                                  "csg_determinant_drift": float(np.max(np.abs(
                                      np.asarray(maps["numeric"]["determinant"]) - 1.0)))}
    ctx.artifact_json("separation-law.json", _plain({"paths": {k: gj.path(k).as_dict() for k in CONSTANT_PATHS},
                                                     "rows": rows, "provider": provider_rows,
                                                     "provider_status": None if csg is None else
                                                     csg.get("refusal", "compared")}))
    head_series, error_series = [], []
    for key in CONSTANT_PATHS:
        fine = _fine_transfer(ctx, key)
        stride = max(1, len(fine.s) // 60)
        head_series.append((f"{key} (K={_fmt(_constant_curvature_of(key))})", fine.s[::stride],
                            fine.states[::stride, 6]))
        _, _, b, _ = jacobi.constant_curvature(_constant_curvature_of(key), fine.s)
        error_series.append((key, fine.s[1::stride], np.maximum(np.abs(fine.states[1::stride, 6] - b[1::stride]),
                                                                1e-17)))
    ctx.artifact_text("heading-column.svg", svg.line_plot(head_series, title="Heading column j_head(s) = sn_K(s)",
                                                          xlabel="arclength s", ylabel="j_head", markers=False))
    ctx.artifact_text("heading-error.svg", svg.line_plot(error_series, title="|j_head - sn_K| along the paths",
                                                         xlabel="arclength s", ylabel="absolute error", logy=True,
                                                         markers=False))

    generator = {"name": "declared constant-curvature geodesics", "paths": list(CONSTANT_PATHS),
                 "steps": {k: _fine(k) for k in CONSTANT_PATHS}}
    findings = [finding(
        "The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on positive, zero "
        "and negative curvature", "numerical", {k: rows[k]["errors"]["heading"] for k in CONSTANT_PATHS},
        {"generator": generator, "checks": [
            gj.check("analytic", f"sn_K with K = {_fmt(rows[k]['curvature'], 6)} on {k}", rows[k]["errors"]["heading"],
                     1e-7) for k in CONSTANT_PATHS]}, tolerance=TOL_SMALL)]
    findings.append(finding(
        "The full integrated transfer matrix (lateral column and both rates) follows the model-space law cn_K, sn_K",
        "numerical", {k: rows[k]["max_error"] for k in CONSTANT_PATHS},
        {"generator": generator, "checks": [gj.check("analytic", f"cn_K, -K sn_K, sn_K, cn_K on {k}",
                                                     rows[k]["max_error"], 1e-7) for k in CONSTANT_PATHS]},
        tolerance=TOL_SMALL))
    torus = gj.surface("torus")
    equators = {k: {"curvature": rows[k]["curvature"], "curvature_drift": rows[k]["curvature_drift"],
                    "theta_drift": rows[k]["theta_drift"], "max_error": rows[k]["max_error"]}
                for k in ("torus-outer-equator", "torus-inner-equator")}
    findings.append(finding(
        "The torus equators are geodesics of constant curvature 1/(r(R+r)) and -1/(r(R-r)) whose Jacobi columns "
        "obey the model-space laws", "numerical", equators,
        {"generator": dict(generator, torus={"major": torus.major, "minor": torus.minor}), "checks": [
            check for k, e in equators.items() for check in (
                gj.check("invariant", f"K(gamma(s)) - K0 along {k}", e["curvature_drift"], 1e-12),
                gj.check("invariant", f"theta stays on {k}", e["theta_drift"], 1e-9),
                gj.check("analytic", f"transfer matrix against cn_K, sn_K on {k}", e["max_error"], 1e-7))]},
        tolerance=TOL_VALUE))
    orders = {k: rows[k]["observed_order"] for k in CONSTANT_PATHS if "observed_order" in rows[k]}
    findings.append(finding(
        "Residuals against the model-space law are fourth-order RK4 discretization error", "numerical", orders,
        {"generator": generator, "checks": [gj.check("self_convergence", f"error ratio at h and 2h on {k}",
                                                     v - 4, 0.3) for k, v in orders.items()]},
        tolerance=TOL_RATE))
    state = "completed"
    identity = None
    notes = ["Paths of constant curvature only; variable curvature is covered by T006-T009"]
    if provider_rows is not None:
        rev = csg["identity"]["revision"]
        worst = max(r["ciw_rk4_vs_csg_rk4"] for r in provider_rows.values())
        findings.append(finding(
            "ciw joint geodesic + Jacobi transfer matrices match the pinned CSG provider's Jacobi integration on six "
            "constant-curvature paths", "numerical", {k: r["ciw_rk4_vs_csg_rk4"] for k, r in provider_rows.items()},
            {"provider": {"repository": gj.CSG_REPOSITORY, "revision": rev,
                          "source_tree": csg["identity"]["source_tree"], "executed": True},
             "checks": [gj.check("analytic", "CSG constant_curvature_transfer against ciw.lab.jacobi.constant_curvature",
                                 max(r["ciw_closed_form_vs_csg_closed_form"] for r in provider_rows.values()), 1e-12)],
             "independent_check": gj.independent(
                 gj.check("high_precision", "CSG integrate_jacobi RK4 on the same arclength grid", worst, 1e-9),
                 "ciw.lab.jacobi", f"{gj.CSG_IMPLEMENTATION}@{rev}", checker_revision=rev)},
            tolerance=TOL_SMALL))
        identity = _provider_identity(csg)
    elif csg is not None:
        state = "partial"
        findings.append(finding(
            "A bound CSG provider that does not match its pin is refused rather than compared", "provenance",
            csg["refusal"], {"checks": [{"reference_kind": "refusal", "reference": "verify_csg_checkout",
                                         "expected_refusal": csg["refusal"], "observed_refusal": csg["refusal"],
                                         "passed": True}]}))
        notes.append(f"CSG provider refused: {csg['message']}")
    else:
        notes.append("The optional CSG provider comparison did not run (bind --provider csg=<checkout>)")
    findings.append(finding(
        "Nearby real trajectories on a physical curved surface separate according to this Jacobi law",
        "physical", None, {}))
    fields = dict(
        hypothesis=("Along a unit-speed geodesic of constant curvature K the heading Jacobi column is sn_K(s) and "
                    "the lateral column cn_K(s), so separation oscillates (K > 0), grows linearly (K = 0) or "
                    "exponentially (K < 0); the torus equators realize K = 1/(r(R+r)) and K = -1/(r(R-r))."),
        mathematical_model=("j'' + K j = 0 with j(0) = 0, j'(0) = 1 (heading) and j(0) = 1, j'(0) = 0 (lateral); on "
                            "a torus of radii R > r the equators theta = 0 and theta = pi are geodesics with "
                            "K = cos(theta)/(r(R + r cos(theta)))."),
        input_data=[f"{k}: K = {_fmt(rows[k]['curvature'], 6)}, L = {rows[k]['length']}, {rows[k]['steps']} RK4 steps"
                    for k in CONSTANT_PATHS],
        observation_model="Relative error max |j - model| / max(1, |model|) over all nodes, per column and rate.",
        expected_invariant="Integrated columns equal sn_K and cn_K to RK4 accuracy; K stays constant along the path.",
        experiment=("Integrate geodesic and Jacobi columns jointly on each surface, compare with the model-space "
                    "closed forms at h and 2h, and (when bound) with the pinned CSG provider in a subprocess."),
        numerical_result=("Largest model-space error " + _fmt(max(r["max_error"] for r in rows.values()))
                          + "; observed orders " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in orders.items())
                          + ("" if provider_rows is None else "; ciw vs CSG "
                             + _fmt(max(r["ciw_rk4_vs_csg_rk4"] for r in provider_rows.values())))
                          + "."),
        uncertainty="RK4 discretization error (about 1e-8 relative at the fine step), confirmed by step halving.",
        failure_modes_checked=["sign of K (oscillation versus exponential growth)",
                               "torus equator stays on the equator (theta drift)",
                               "flat charts where the columns are exactly 1 and s",
                               "provider checkout identity verified before and after execution"],
        unresolved_assumptions=notes + ["No physical trajectories were measured; the physical-domain claim is "
                                        "recorded as not established"],
        recommended_next_task="T006: compare the integrated columns with finite-difference flow perturbations")
    if identity:
        fields["provider_runtime_identity"] = identity
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------------------
# T006: finite-difference flow perturbations
# ---------------------------------------------------------------------------
FD_SURFACES = ("sphere", "torus", "gaussian-bump")
FD_STEPS = 160
FD_EPS = (0.08, 0.04, 0.02, 0.01)
ROUNDOFF_EPS = tuple(10.0 ** -e for e in range(1, 12))


def _perturbed_separation(ctx, key, lateral, heading):
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    base = gj.transfer(ctx, key, "rk4", FD_STEPS)

    def compute():
        y0 = jacobi.perturbed_start(surf, spec.u0, spec.heading, lateral=lateral, heading_change=heading)
        _, states = integrators.integrate_fixed(surf.geodesic_rhs, y0, spec.length, FD_STEPS, "rk4")
        return jacobi.normal_separation(surf, base.states, states)
    return ctx.memo(("gj-perturbed", key, FD_STEPS, lateral, heading), compute)


def finite_difference_study(ctx) -> dict:
    rows = {}
    for key in FD_SURFACES:
        base = gj.transfer(ctx, key, "rk4", FD_STEPS)
        row = {}
        for column, index in (("lateral", 4), ("heading", 6)):
            one, central = [], []
            for eps in FD_EPS:
                args = (eps, 0.0) if column == "lateral" else (0.0, eps)
                plus = _perturbed_separation(ctx, key, *args)
                minus = _perturbed_separation(ctx, key, *(-a for a in args))
                one.append(float(np.max(np.abs(plus / eps - base.states[:, index]))))
                central.append(float(np.max(np.abs((plus - minus) / (2 * eps) - base.states[:, index]))))
            row[column] = {"one_sided": one, "central": central, "one_sided_order": gj.slope(FD_EPS, one),
                           "central_order": gj.slope(FD_EPS, central)}
        rows[key] = row
    base = gj.transfer(ctx, "sphere", "rk4", FD_STEPS)
    roundoff = [float(np.max(np.abs(_perturbed_separation(ctx, "sphere", 0.0, eps) / eps - base.states[:, 6])))
                for eps in ROUNDOFF_EPS]
    return {"rows": rows, "roundoff": roundoff}


@task("T006", changed_files=CHANGED,
      regression_tests=(_test("test_t006_finite_differences"), _test("test_perturbation_helpers_are_geometric")))
def finite_difference_jacobi(ctx):
    study = finite_difference_study(ctx)
    rows, roundoff = study["rows"], study["roundoff"]
    best = int(np.argmin(roundoff))
    ctx.artifact_json("finite-differences.json", _plain({"eps": FD_EPS, "steps": FD_STEPS, "rows": rows,
                                                         "roundoff_eps": ROUNDOFF_EPS, "roundoff_errors": roundoff}))
    for key in FD_SURFACES:
        ctx.artifact_text(f"fd-{key}.svg", svg.line_plot(
            [(f"{c} {kind}", FD_EPS, rows[key][c][kind]) for c in ("lateral", "heading")
             for kind in ("one_sided", "central")], title=f"Finite differences vs Jacobi columns on {key}",
            xlabel="perturbation eps", ylabel="max |FD - column|", logx=True, logy=True))
    ctx.artifact_text("fd-roundoff.svg", svg.line_plot(
        [("sphere heading one-sided", ROUNDOFF_EPS, roundoff)], title="Truncation versus cancellation (sphere)",
        xlabel="perturbation eps", ylabel="max |FD - column|", logx=True, logy=True))

    generator = {"name": "perturbed geodesics", "surfaces": list(FD_SURFACES), "eps": list(FD_EPS),
                 "steps": FD_STEPS}
    orders_one = {k: {c: rows[k][c]["one_sided_order"] for c in ("lateral", "heading")} for k in FD_SURFACES}
    orders_central = {k: {c: rows[k][c]["central_order"] for c in ("lateral", "heading")} for k in FD_SURFACES}
    findings = [finding(
        "Central finite differences of perturbed geodesics converge to the integrated Jacobi columns at second order",
        "numerical", orders_central, {"generator": generator, "checks": [
            gj.check("analytic", f"central difference order 2, {c} column on {k}", v - 2, 0.1)
            for k, r in orders_central.items() for c, v in r.items()] + [
            gj.check("self_convergence", "largest central-difference error at eps = 0.01",
                     max(rows[k][c]["central"][-1] for k in FD_SURFACES for c in ("lateral", "heading")), 1e-3)]},
        tolerance=TOL_RATE)]
    findings.append(finding(
        "One-sided finite differences converge to the integrated Jacobi columns at first order", "numerical",
        orders_one, {"generator": generator, "checks": [
            gj.check("analytic", f"one-sided difference order 1, {c} column on {k}", v - 1, 0.15)
            for k, r in orders_one.items() for c, v in r.items()]}, tolerance=TOL_RATE))
    ratio = roundoff[-1] / roundoff[best]
    # Rounding-dominated numbers vary across platforms, so the retained value is in decades.
    findings.append(finding(
        "Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation)",
        "numerical", {"log10_best_eps": math.log10(ROUNDOFF_EPS[best]), "log10_error_ratio_1e-11_over_best":
                      math.log10(ratio)},
        {"generator": dict(generator, eps=list(ROUNDOFF_EPS), surfaces=["sphere"]), "checks": [
            gj.check("invariant", "error at eps = 1e-11 over the best error", ratio, 10.0, "ge")]},
        tolerance={"abs": 1.5, "rel": 0.0},
        counterexample={"statement": "A smaller finite-difference step always gives a more accurate Jacobi estimate",
                        "witness": {"surface": "sphere", "column": "heading", "eps": 1e-11,
                                    "error": roundoff[-1], "best_eps": ROUNDOFF_EPS[best],
                                    "best_error": roundoff[best]}}))
    return _outcome(
        "completed", findings,
        hypothesis=("The integrated lateral and heading Jacobi columns are the derivatives of the geodesic flow with "
                    "respect to exact lateral displacement and heading rotation of the start."),
        mathematical_model=("gamma_eps(s) = exp flow of the perturbed start; (gamma_eps - gamma_0)/eps = J + O(eps) "
                            "and (gamma_eps - gamma_-eps)/(2 eps) = J + O(eps^2), with J = j N and g(J, N) = j."),
        input_data=[f"Standard paths on {', '.join(FD_SURFACES)}; RK4 with {FD_STEPS} steps for base and perturbed "
                    f"paths; eps in {list(FD_EPS)} and 1e-1..1e-11 for the cancellation sweep"],
        observation_model=("Normal separation g(delta u, N) at matched arclength nodes (ciw.lab.jacobi."
                           "normal_separation), divided by eps; max over nodes of the difference from the column."),
        expected_invariant="Slopes 1 (one-sided) and 2 (central) in eps until cancellation dominates.",
        experiment=("Perturb the start by exact lateral displacement along a normal geodesic (tangent parallel "
                    "transported) or by heading rotation, integrate, and difference."),
        numerical_result=("Central orders " + "; ".join(f"{k} " + ", ".join(f"{c} {_fmt(v, 3)}" for c, v in r.items())
                                                         for k, r in orders_central.items())
                          + "; one-sided orders " + "; ".join(f"{k} " + ", ".join(f"{c} {_fmt(v, 3)}"
                                                                                  for c, v in r.items())
                                                              for k, r in orders_one.items())
                          + f"; best one-sided eps {_fmt(ROUNDOFF_EPS[best])} (error {_fmt(roundoff[best])}), "
                          f"error at 1e-11 {_fmt(roundoff[-1])}."),
        uncertainty=("RK4 error of base and perturbed paths (about 1e-9) is far below the smallest central "
                     "truncation error (about 1e-6)."),
        failure_modes_checked=["chart-dependent perturbations avoided (exact exponential-map offset)",
                               "cancellation at tiny eps (recorded as a counterexample)",
                               "matched nodes: base and perturbed paths share the arclength grid"],
        unresolved_assumptions=["The separation is measured in the chart and projected on N; its second-order "
                                "chart curvature term cancels only in the central difference"],
        recommended_next_task="T007: measure the Wronskian and transfer-matrix determinant")


# ---------------------------------------------------------------------------
# T007: Wronskian and determinant
# ---------------------------------------------------------------------------
DET_CONSTANT = ("sphere-great-circle", "hyperbolic-long", "torus-outer-equator", "torus-inner-equator")
DET_VARIABLE = ("torus", "gaussian-bump", "saddle", "bump-radial", "torus-outer-to-inner")
DET_FLAT = ("plane", "cylinder")
DET_STEPS = {"euler": (50, 100, 200, 400), "midpoint": (25, 50, 100, 200), "rk4": (16, 32, 64, 128)}
DET_RTOL = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


def per_step_determinant(method: str, h: float, k: float) -> float:
    """Exact one-step determinant of each method on j'' + K j = 0 with constant K."""
    if method == "euler":
        return 1.0 + h * h * k
    if method == "midpoint":
        return 1.0 + h ** 4 * k * k / 4.0
    if method == "rk4":
        return 1.0 - h ** 6 * k ** 3 / 72.0 + h ** 8 * k ** 4 / 576.0
    raise ValueError(f"Unsupported method: {method}")


def symbolic_step_determinants() -> dict:
    """Taylor coefficients of det(step matrix) - 1 in h for midpoint and RK4 with smooth K(s) (sympy).

    Stage curvatures are K(s + c h) plus O(h^2) offsets e2, e3 from the stage
    positions of the coupled geodesic integration.
    """
    import sympy as sp

    h, e2, e3 = sp.symbols("h e2 e3")
    k = sp.symbols("k0:6")

    def curvature(t):
        return sum(k[i] * t ** i / sp.factorial(i) for i in range(6))

    def a_matrix(value):
        return sp.Matrix([[0, 1], [-value, 0]])

    one = sp.eye(2)
    a1, a2, a3, a4 = (a_matrix(curvature(0)), a_matrix(curvature(h / 2) + e2 * h ** 2),
                      a_matrix(curvature(h / 2) + e3 * h ** 2), a_matrix(curvature(h)))
    rk4 = one + h / 6 * (a1 + 2 * a2 * (one + h / 2 * a1) + 2 * a3 * (one + h / 2 * a2 * (one + h / 2 * a1))
                         + a4 * (one + h * a3 * (one + h / 2 * a2 * (one + h / 2 * a1))))
    midpoint = one + h * a2 * (one + h / 2 * a1)
    out = {}
    for name, matrix, order in (("rk4", rk4, 7), ("midpoint", midpoint, 5)):
        series = sp.expand(sp.series(sp.expand(matrix.det() - 1), h, 0, order).removeO())
        out[name] = {str(p): str(sp.factor(series.coeff(h, p))) for p in range(order)}
    return out


def determinant_study(ctx) -> dict:
    def compute():
        rows = {}
        for key in DET_CONSTANT + DET_VARIABLE + DET_FLAT:
            spec = gj.path(key)
            k0 = _constant_curvature_of(key) if key in DET_CONSTANT + DET_FLAT else None
            row = {}
            for method, counts in DET_STEPS.items():
                entries = []
                for steps in counts:
                    tr = gj.transfer(ctx, key, method, steps)
                    det = tr.determinant()
                    h = spec.length / steps
                    curvature = tr.curvature_along()
                    entry = {"steps": steps, "h": h, "max_drift": float(np.max(np.abs(det - 1.0))),
                             "end_drift": float(det[-1] - 1.0), "k_start": float(curvature[0]),
                             "k_end": float(curvature[-1])}
                    if method == "euler":
                        # Scaled by |a b'| + |a' b|: the determinant is a difference of products.
                        y = tr.states
                        scale = np.maximum(np.abs(y[1:, 4] * y[1:, 7]) + np.abs(y[1:, 5] * y[1:, 6]), 1.0)
                        entry["per_step_factor_error"] = float(np.max(
                            np.abs(det[1:] - (1 + h * h * curvature[:-1]) * det[:-1]) / scale))
                    if k0 is not None:
                        predicted = per_step_determinant(method, h, k0) ** np.arange(steps + 1)
                        entry["prediction_error"] = _discrepancy(det, predicted)
                    if method == "midpoint" and key in DET_VARIABLE:
                        entry["boundary_ratio"] = float((det[-1] - 1.0) / (h * h / 4 * (curvature[-1] - curvature[0])))
                    entries.append(entry)
                row[method] = entries
            if key not in DET_FLAT:
                adaptive = []
                for rtol in DET_RTOL:
                    tr = gj.transfer(ctx, key, rtol=rtol, atol=rtol)
                    adaptive.append({"rtol": rtol, "max_drift": float(np.max(np.abs(tr.determinant() - 1.0))),
                                     **tr.stats})
                row["adaptive"] = adaptive
            rows[key] = row
        return rows
    return ctx.memo("gj-determinant", compute)


@task("T007", changed_files=CHANGED,
      regression_tests=(_test("test_t007_determinant"), _test("test_t007_symbolic_step_determinants")))
def wronskian_determinant(ctx):
    rows = determinant_study(ctx)
    curved = DET_CONSTANT + DET_VARIABLE
    slopes = {m: {k: _slopes(rows[k][m], "max_drift") for k in curved} for m in DET_STEPS}
    adaptive = {k: gj.slope([e["rtol"] for e in rows[k]["adaptive"]], [e["max_drift"] for e in rows[k]["adaptive"]])
                for k in curved}
    euler_factor = max(e["per_step_factor_error"] for k in rows for e in rows[k]["euler"])
    prediction = {m: max(e["prediction_error"] for k in DET_CONSTANT + DET_FLAT for e in rows[k][m])
                  for m in DET_STEPS}
    boundary = {k: rows[k]["midpoint"][-1]["boundary_ratio"] for k in DET_VARIABLE}
    flat = max(e["max_drift"] for k in DET_FLAT for m in DET_STEPS for e in rows[k][m])
    symbolic = symbolic_step_determinants() if ctx.available("module:sympy") else None
    ctx.artifact_json("determinant.json", _plain({"steps": DET_STEPS, "rtol": DET_RTOL, "rows": rows,
                                                  "drift_orders": slopes, "adaptive_tolerance_exponent": adaptive,
                                                  "symbolic_step_determinants": symbolic}))
    for method in DET_STEPS:
        ctx.artifact_text(f"det-{method}.svg", svg.line_plot(
            [(k, [e["h"] for e in rows[k][method]], [e["max_drift"] for e in rows[k][method]]) for k in curved],
            title=f"max |det Phi - 1|, {method}", xlabel="step h", ylabel="max |det Phi - 1|", logx=True, logy=True))
    along = []
    for method in DET_STEPS:
        tr = gj.transfer(ctx, "sphere-great-circle", method, DET_STEPS[method][1])
        along.append((f"{method} N={DET_STEPS[method][1]}", tr.s, np.maximum(np.abs(tr.determinant() - 1), 1e-17)))
    ctx.artifact_text("det-along-sphere.svg", svg.line_plot(along, title="|det Phi(s) - 1| on the sphere great circle",
                                                            xlabel="arclength s", ylabel="|det Phi - 1|", logy=True,
                                                            markers=False))

    generator = {"name": "declared geodesics", "constant": list(DET_CONSTANT), "variable": list(DET_VARIABLE),
                 "flat": list(DET_FLAT)}
    findings = [finding(
        "Explicit Euler multiplies det Phi by exactly 1 + h^2 K(gamma_n) per step, so it is not area-preserving "
        "where K is nonzero", "numerical", {"variable_curvature_orders": {k: slopes["euler"][k] for k in DET_VARIABLE},
                                            "per_step_factor_error": euler_factor,
                                            "constant_curvature_prediction_error": prediction["euler"]},
        {"generator": generator, "checks": [
            gj.check("analytic", "per-step Euler factor 1 + h^2 K_n on every path and step", euler_factor, 1e-12),
            gj.check("analytic", "(1 + h^2 K)^n on constant-curvature paths", prediction["euler"], 1e-10)] + [
            gj.check("analytic", f"Euler determinant drift order 1 on variable-curvature {k}", slopes["euler"][k] - 1,
                     0.1) for k in DET_VARIABLE]}, tolerance=TOL_RATE)]
    mid_checks = [gj.check("analytic", "(1 + h^4 K^2 / 4)^n on constant-curvature paths", prediction["midpoint"], 1e-10)]
    mid_checks += [gj.check("analytic", f"midpoint drift order 3 on constant-curvature {k}", slopes["midpoint"][k] - 3,
                            0.15) for k in DET_CONSTANT]
    mid_checks += [gj.check("analytic", f"midpoint drift order 2 on variable-curvature {k}", slopes["midpoint"][k] - 2,
                            0.15) for k in DET_VARIABLE]
    mid_checks += [gj.check("analytic", f"end drift / (h^2 (K(L) - K(0)) / 4) at N=200 on {k}", v - 1, 0.05)
                   for k, v in boundary.items()]
    # The O(h^3) remainder makes |ratio - 1| shrink like h: halving h should roughly halve it.
    mid_checks += [gj.check("self_convergence", f"|ratio - 1| at N=100 over N=200 on {k}",
                            abs(rows[k]["midpoint"][-2]["boundary_ratio"] - 1)
                            / abs(rows[k]["midpoint"][-1]["boundary_ratio"] - 1), 1.5, "ge") for k in DET_VARIABLE]
    findings.append(finding(
        "Midpoint determinant drift is (h^2/4)(K(L) - K(0)) + O(h^3): second order on variable curvature, third "
        "order on constant curvature", "numerical", {"orders": {k: slopes["midpoint"][k] for k in curved},
                                                      "boundary_ratio": boundary},
        {"generator": generator, "checks": mid_checks}, tolerance=TOL_RATE))
    rk4_checks = [gj.check("analytic", "(1 - h^6 K^3/72 + h^8 K^4/576)^n on constant-curvature paths",
                           prediction["rk4"], 1e-10)]
    rk4_checks += [gj.check("analytic", f"RK4 determinant drift order 5 on {k}", v - 5, 0.2)
                   for k, v in slopes["rk4"].items()]
    if symbolic is not None:
        low = sum(symbolic["rk4"][str(p)] != "0" for p in range(1, 6))
        rk4_checks.append(gj.check("exact_arithmetic", "nonzero h^1..h^5 coefficients of det(RK4 step) - 1 for smooth "
                                   "K(s) (sympy)", low, 0.0))
    findings.append(finding(
        "RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable curvature",
        "numerical", {k: slopes["rk4"][k] for k in curved}, {"generator": generator, "checks": rk4_checks},
        tolerance=TOL_RATE,
        counterexample={"statement": "The RK4 transfer-matrix determinant drifts at the method's global order h^4",
                        "witness": {"orders": {k: slopes["rk4"][k] for k in curved},
                                    "per_step_defect": "-h^6 K^3/72 + O(h^7) (constant K); O(h^6) for smooth K(s)"}}))
    findings.append(finding(
        "Adaptive Dormand-Prince determinant drift decreases at least in proportion to the tolerance", "numerical",
        adaptive, {"generator": dict(generator, rtol=list(DET_RTOL)), "checks": [
            gj.check("analytic", f"log-log slope of drift against rtol on {k}", v, 0.9, "ge")
            for k, v in adaptive.items()]}, tolerance={"abs": 0.05, "rel": 0.0}))
    findings.append(finding(
        "Where K = 0 every method preserves det Phi = 1 exactly, Euler included", "numerical", flat,
        {"generator": generator, "checks": [gj.check("analytic", "max |det Phi - 1| on plane and cylinder, all methods",
                                                     flat, 1e-14)]},
        tolerance=TOL_SMALL,
        counterexample={"statement": "Explicit Euler never preserves the transfer-matrix determinant",
                        "witness": {"charts": list(DET_FLAT), "max_drift": flat, "per_step_factor": "1 + h^2 K = 1"}}))
    return _outcome(
        "completed", findings,
        hypothesis=("det Phi(s) = 1 exactly; its numerical drift measures each method: Euler drifts at O(h) through "
                    "the per-step factor 1 + h^2 K, and RK4 drifts at O(h^4) or better."),
        mathematical_model=("Phi' = A(s) Phi with A = [[0, 1], [-K, 0]] and tr A = 0 (Liouville). Step matrices: "
                            "Euler [[1, h], [-h K_n, 1]]; midpoint det = 1 + h^2 (K_mid - K_n)/2 + h^4 K_n K_mid/4; "
                            "RK4 det - 1 has no h^1..h^5 terms for smooth K (sympy) and equals -h^6 K^3/72 + "
                            "h^8 K^4/576 for constant K."),
        input_data=[f"Constant-curvature paths {list(DET_CONSTANT)}, variable {list(DET_VARIABLE)}, flat "
                    f"{list(DET_FLAT)}", f"Steps {DET_STEPS}; adaptive rtol = atol in {list(DET_RTOL)}"],
        observation_model="max over nodes of |det Phi - 1| with det = j_lat j_head' - j_lat' j_head.",
        expected_invariant=("Exact per-step factors on constant K; fitted drift orders 1 (Euler), 2 or 3 (midpoint), "
                            "5 (RK4)."),
        experiment=("Integrate geodesic and Jacobi columns jointly with each method and step, compare the "
                    "determinant with the per-step predictions, fit drift orders, and derive the step determinants "
                    "symbolically."),
        numerical_result=("Drift orders: " + "; ".join(f"{m} " + ", ".join(f"{k} {_fmt(v, 3)}"
                                                                          for k, v in slopes[m].items())
                                                      for m in DET_STEPS)
                          + f"; Euler per-step factor error {_fmt(euler_factor)}."),
        uncertainty=("Fitted orders over four halvings; the exact per-step predictions hold to about 1e-12 "
                     "relative (larger on the inner equator where the columns reach cosh(6.5)). Euler slopes on "
                     "the long constant-curvature paths are not fitted claims: (1 + h^2 K)^(L/h) - 1 ~ "
                     "exp(h L K) - 1 is not yet linear in h when h L |K| is near 1, so the exact product formula "
                     "is checked there instead."),
        failure_modes_checked=["Euler area growth (K > 0) and shrinkage (K < 0) from 1 + h^2 K",
                               "midpoint cancellation when K(L) = K(0) (constant K gives order 3)",
                               "RK4 superconvergence of the determinant (recorded as a counterexample to the h^4 "
                               "expectation)", "flat charts where every method is exact"],
        unresolved_assumptions=["The O(h^6) RK4 per-step defect is derived for smooth K(s) along the stage points; "
                                "paths through curvature discontinuities were not tested"],
        recommended_next_task="T008: locate conjugate and focal points")


# ---------------------------------------------------------------------------
# T008: conjugate and focal points
# ---------------------------------------------------------------------------
LOCATE_STEPS = (40, 80, 160, 320)
STURM_TORUS = 6
STURM_BUMP = 4


def _sphere_r2(ctx, steps):
    def compute():
        return jacobi.transfer(surfaces.Sphere(2.0), (math.pi / 2, 0.0), 1.0, 14.0, steps=steps)
    return ctx.memo(("gj-sphere-r2", steps), compute)


def _location_errors(tr, k):
    w = math.sqrt(k)
    conj, focal = tr.conjugate_points(), tr.focal_points()
    expected_conj = [(i + 1) * math.pi / w for i in range(int(tr.s[-1] * w / math.pi))]
    expected_focal = [(i + 0.5) * math.pi / w for i in range(int(tr.s[-1] * w / math.pi + 0.5))]
    ok = len(conj) == len(expected_conj) and len(focal) == len(expected_focal)
    error = max([abs(a - b) for a, b in zip(conj, expected_conj)] + [abs(a - b) for a, b in zip(focal, expected_focal)])
    return {"conjugate": conj, "focal": focal, "expected_conjugate": expected_conj, "expected_focal": expected_focal,
            "counts_match": bool(ok), "max_error": float(error)}


def sturm_study(ctx):
    def compute():
        rng = np.random.Generator(np.random.PCG64(gj.SEED + 8))
        rows = []
        for name, count, box, length in (("torus", STURM_TORUS, ((0, 2 * math.pi), (-math.pi, math.pi)), 12.0),
                                         ("gaussian-bump", STURM_BUMP, ((-1.0, 1.0), (-1.0, 1.0)), 10.0)):
            surf = gj.surface(name)
            for _ in range(count):
                u0 = (float(rng.uniform(*box[0])), float(rng.uniform(*box[1])))
                heading = float(rng.uniform(-math.pi, math.pi))
                tr = jacobi.transfer(surf, u0, heading, length, rtol=1e-9, atol=1e-11)
                rows.append({"surface": name, "u0": list(u0), "heading": heading, "length": length,
                             "conjugate": tr.conjugate_points(), "focal": tr.focal_points(),
                             "max_curvature_on_path": float(tr.curvature_along().max())})
        return rows
    return ctx.memo("gj-sturm", compute)


@task("T008", changed_files=CHANGED, regression_tests=(_test("test_t008_conjugate_and_focal_points"),))
def conjugate_focal_points(ctx):
    torus = gj.surface("torus")
    bump = gj.surface("gaussian-bump")
    k_outer = _constant_curvature_of("torus-outer-equator")
    located = {}
    for key, k in (("sphere-great-circle", 1.0), ("torus-outer-equator", k_outer)):
        entries = [dict(_location_errors(gj.transfer(ctx, key, "rk4", n), k), steps=n,
                        h=gj.path(key).length / n) for n in LOCATE_STEPS]
        located[key] = {"curvature": k, "entries": entries,
                        "order": gj.slope([e["h"] for e in entries], [e["max_error"] for e in entries])}
    sphere2 = _location_errors(_sphere_r2(ctx, 934), 0.25)
    negative = {}
    for key in ("torus-inner-equator", "hyperbolic-long"):
        tr = _fine_transfer(ctx, key)
        negative[key] = {"conjugate": tr.conjugate_points(), "focal": tr.focal_points(),
                         "max_s_minus_j_head": float(np.max(tr.s - tr.states[:, 6])),
                         "max_1_minus_j_lat": float(np.max(1.0 - tr.states[:, 4]))}
    sturm = sturm_study(ctx)
    k_max_torus = 1.0 / (torus.minor * (torus.major + torus.minor))
    grid = np.linspace(-4, 4, 161)
    k_max_bump = max(bump.gaussian_curvature(np.array([x, y])) for x in grid for y in grid)
    bounds = {"torus": math.pi / math.sqrt(k_max_torus), "gaussian-bump": math.pi / math.sqrt(k_max_bump)}
    firsts = {name: [r["conjugate"][0] for r in sturm if r["surface"] == name and r["conjugate"]]
              for name in bounds}
    margin = min(min(v) - bounds[n] for n, v in firsts.items() if v)
    witness = next(r for r in sturm if r["conjugate"] and r["focal"]
                   and abs(r["focal"][0] - r["conjugate"][0] / 2) > 0.3)
    csg = _csg(ctx) if ctx.available("provider:csg") else None
    provider = None
    if csg and "data" in csg:
        provider = {}
        for key, maps in zip(csg["paths"], csg["data"]["maps"]):
            if key not in ("sphere-great-circle", "torus-outer-equator", "torus-inner-equator", "hyperbolic-long"):
                continue
            tr = _fine_transfer(ctx, key)
            events = maps["numeric"]["focus_events"]
            csg_conj = [e["arc_length"] for e in events["b"]]
            csg_focal = [e["arc_length"] for e in events["a"]]
            ours_conj, ours_focal = tr.conjugate_points(), tr.focal_points()
            same = len(csg_conj) == len(ours_conj) and len(csg_focal) == len(ours_focal)
            gap = max([abs(a - b) for a, b in zip(csg_conj, ours_conj)]
                      + [abs(a - b) for a, b in zip(csg_focal, ours_focal)] + [0.0]) if same else float("inf")
            provider[key] = {"csg_conjugate": csg_conj, "csg_focal": csg_focal, "ciw_conjugate": ours_conj,
                             "ciw_focal": ours_focal, "counts_match": bool(same), "max_gap": gap}
    ctx.artifact_json("conjugate-focal.json", _plain({
        "located": located, "sphere_radius_2": sphere2, "negative_curvature": negative,
        "sturm": {"bounds": bounds, "k_max": {"torus": k_max_torus, "gaussian-bump": k_max_bump}, "paths": sturm},
        "provider": provider}))
    ctx.artifact_text("location-accuracy.svg", svg.line_plot(
        [(k, [e["h"] for e in v["entries"]], [e["max_error"] for e in v["entries"]]) for k, v in located.items()],
        title="Conjugate/focal point location error", xlabel="step h", ylabel="max location error", logx=True,
        logy=True))
    tr = _fine_transfer(ctx, "torus-outer-equator")
    ctx.artifact_text("columns-outer-equator.svg", svg.line_plot(
        [("j_head", tr.s, tr.states[:, 6]), ("j_lat", tr.s, tr.states[:, 4])],
        title="Torus outer equator: zeros are conjugate (j_head) and focal (j_lat) points", xlabel="arclength s",
        ylabel="j", markers=False))

    findings = []
    sphere = located["sphere-great-circle"]
    findings.append(finding(
        "Sphere conjugate points lie at pi R and 2 pi R and focal points at pi R/2 and 3 pi R/2 (R = 1 and R = 2)",
        "numerical", {"R1": {"conjugate": sphere["entries"][-1]["conjugate"], "focal": sphere["entries"][-1]["focal"]},
                      "R2": {"conjugate": sphere2["conjugate"], "focal": sphere2["focal"]}},
        {"generator": {"name": "great circles", "radii": [1.0, 2.0]}, "checks": [
            gj.check("analytic", "R = 1 locations at 320 steps", sphere["entries"][-1]["max_error"], 1e-7),
            gj.check("analytic", "R = 2 locations at 934 steps", sphere2["max_error"], 1e-7),
            gj.check("analytic", "location error order 4 (RK4 plus cubic Hermite)", sphere["order"] - 4, 0.3),
            gj.check("exact_arithmetic", "zero counts match pi multiples",
                     float(not (sphere["entries"][-1]["counts_match"] and sphere2["counts_match"])), 0.0)]},
        tolerance=TOL_VALUE))
    outer = located["torus-outer-equator"]
    findings.append(finding(
        "On the torus outer equator conjugate points lie at pi sqrt(r(R+r)) multiples and focal points half-way",
        "numerical", {"conjugate": outer["entries"][-1]["conjugate"], "focal": outer["entries"][-1]["focal"],
                      "analytic_first_conjugate": math.pi / math.sqrt(k_outer)},
        {"generator": {"name": "torus outer equator", "major": torus.major, "minor": torus.minor}, "checks": [
            gj.check("analytic", "locations at 320 steps", outer["entries"][-1]["max_error"], 1e-7),
            gj.check("analytic", "location error order 4", outer["order"] - 4, 0.3),
            gj.check("exact_arithmetic", "zero counts match",
                     float(not outer["entries"][-1]["counts_match"]), 0.0)]},
        tolerance=TOL_VALUE))
    findings.append(finding(
        "No conjugate or focal point occurs on the torus inner equator or the hyperbolic plane (K < 0)", "numerical",
        {k: {"conjugate_count": len(v["conjugate"]), "focal_count": len(v["focal"]),
             "max_s_minus_j_head": v["max_s_minus_j_head"], "max_1_minus_j_lat": v["max_1_minus_j_lat"]}
         for k, v in negative.items()},
        {"generator": {"name": "negative-curvature paths", "paths": list(negative)}, "checks": [
            check for k, v in negative.items() for check in (
                gj.check("exact_arithmetic", f"zeros found on {k}", len(v["conjugate"]) + len(v["focal"]), 0.0),
                gj.check("invariant", f"Sturm comparison: largest s - j_head on {k}", v["max_s_minus_j_head"], 1e-9,
                         "le"),
                gj.check("invariant", f"Sturm comparison: largest 1 - j_lat on {k}", v["max_1_minus_j_lat"], 1e-9,
                         "le"))]},
        tolerance=TOL_VALUE))
    findings.append(finding(
        "Sturm comparison bound holds: no conjugate point before pi/sqrt(max K) on seeded torus and bump geodesics",
        "numerical", {"bounds": bounds, "first_conjugate_points": firsts, "margin": margin,
                      "paths_with_conjugate_point": {n: len(v) for n, v in firsts.items()}},
        {"generator": {"name": "seeded geodesics", "seed": gj.SEED + 8, "torus": STURM_TORUS, "bump": STURM_BUMP},
         "checks": [gj.check("invariant", "min first conjugate point minus pi/sqrt(K_max)", margin, 0.0, "ge")]},
        tolerance={"abs": 1e-6, "rel": 1e-6}))
    findings.append(finding(
        "On variable curvature the first focal point is not half the first conjugate distance", "numerical",
        {"focal": witness["focal"][0], "conjugate": witness["conjugate"][0]},
        {"generator": {"name": "seeded torus geodesic", "u0": witness["u0"], "heading": witness["heading"]},
         "checks": [gj.check("invariant", "|first focal - first conjugate / 2|",
                             abs(witness["focal"][0] - witness["conjugate"][0] / 2), 0.3, "ge")]},
        tolerance={"abs": 1e-6, "rel": 1e-6},
        counterexample={"statement": "The first focal point lies at half the first conjugate distance",
                        "witness": {"surface": witness["surface"], "u0": witness["u0"], "heading": witness["heading"],
                                    "first_focal": witness["focal"][0], "first_conjugate": witness["conjugate"][0]}}))
    state, notes, identity = "completed", [], None
    if provider is not None:
        rev = csg["identity"]["revision"]
        gap = max(v["max_gap"] for v in provider.values())
        findings.append(finding(
            "ciw conjugate and focal points match the pinned CSG provider's focus events on constant-curvature paths",
            "numerical", {k: v["max_gap"] for k, v in provider.items()},
            {"provider": {"repository": gj.CSG_REPOSITORY, "revision": rev,
                          "source_tree": csg["identity"]["source_tree"], "executed": True},
             "independent_check": gj.independent(
                 gj.check("high_precision", "CSG TransferMap.focus_events on its RK4 trace (same grid)", gap, 1e-8),
                 "ciw.lab.jacobi", f"{gj.CSG_IMPLEMENTATION}@{rev}", checker_revision=rev)},
            tolerance=TOL_SMALL))
        identity = _provider_identity(csg)
    elif csg is not None:
        state = "partial"
        notes.append(f"CSG provider refused: {csg['message']}")
    else:
        notes.append("The optional CSG focus-event comparison did not run (bind --provider csg=<checkout>)")
    fields = dict(
        hypothesis=("Zeros of the heading column are conjugate points and zeros of the lateral column are focal "
                    "points; on constant K > 0 they sit at multiples of pi/sqrt(K) and half-way between, and "
                    "none exist where K <= 0."),
        mathematical_model=("j_head = sn_K, j_lat = cn_K on constant K; Sturm comparison: K <= K_max gives no "
                            "conjugate point before pi/sqrt(K_max), and K <= 0 gives j_head >= s, j_lat >= 1."),
        input_data=[f"Sphere great circles (R = 1, L = 7; R = 2, L = 14); torus outer equator (L = 11.5, K = 1/3); "
                    f"inner equator and hyperbolic plane; {STURM_TORUS} torus and {STURM_BUMP} bump seeded geodesics"],
        observation_model="Cubic-Hermite zeros of sampled columns (ciw.lab.jacobi.Transfer.conjugate_points/focal_points).",
        expected_invariant="Locations to RK4 + Hermite accuracy (order 4), zero counts exact, Sturm bounds respected.",
        experiment=("Locate zeros at 40..320 steps, compare with pi multiples, check absence on K < 0, check the "
                    "Sturm bound on seeded variable-curvature geodesics, and compare with CSG when bound."),
        numerical_result=(f"Sphere location error {_fmt(sphere['entries'][-1]['max_error'])} (order "
                          f"{_fmt(sphere['order'], 3)}); outer equator {_fmt(outer['entries'][-1]['max_error'])} "
                          f"(order {_fmt(outer['order'], 3)}); Sturm margin {_fmt(margin)} with "
                          f"{len(firsts['torus'])} of {STURM_TORUS} torus and {len(firsts['gaussian-bump'])} of "
                          f"{STURM_BUMP} bump geodesics reaching a conjugate point (the bump bound is vacuous when "
                          f"none does); witness focal {_fmt(witness['focal'][0], 5)} vs conjugate/2 "
                          f"{_fmt(witness['conjugate'][0] / 2, 5)}."),
        uncertainty="Location error below 1e-7 at the finest steps; adaptive seeded runs use rtol 1e-9.",
        failure_modes_checked=["trivial zero of j_head at s = 0 excluded", "zero counts, not only locations",
                               "negative curvature (no zeros) and Sturm lower bounds",
                               "focal/conjugate relation on variable curvature (counterexample)"],
        unresolved_assumptions=notes + ["The bump Sturm bound uses K_max sampled on a 161 x 161 grid (attained at "
                                        "the summit, K = 0.25)"],
        recommended_next_task="T009: compare lateral and heading columns separately; T010: near-focus counterexamples")
    if identity:
        fields["provider_runtime_identity"] = identity
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------------------
# T009: lateral versus heading columns
# ---------------------------------------------------------------------------
COLUMN_PATHS = tuple(gj.STANDARD) + ("sphere-great-circle", "hyperbolic-long", "torus-outer-equator",
                                     "torus-inner-equator", "bump-radial", "torus-outer-to-inner",
                                     "torus-inner-to-outer")
WITNESS_PAIR = ("torus-outer-to-inner", "torus-inner-to-outer")
CONFIRM_EPS = 1e-3


def _endpoint_confirmation(ctx, key):
    """Central-difference endpoint separation per unit perturbation, for both perturbation types."""
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    steps = _fine(key)
    base = _fine_transfer(ctx, key)
    out = {}
    for column, args in (("lateral", (CONFIRM_EPS, 0.0)), ("heading", (0.0, CONFIRM_EPS))):
        ends = []
        for sign in (1.0, -1.0):
            y0 = jacobi.perturbed_start(surf, spec.u0, spec.heading, lateral=sign * args[0],
                                        heading_change=sign * args[1])
            _, states = integrators.integrate_fixed(surf.geodesic_rhs, y0, spec.length, steps, "rk4")
            ends.append(jacobi.normal_separation(surf, base.states[-1:], states[-1:])[0])
        out[column] = float((ends[0] - ends[1]) / (2 * CONFIRM_EPS))
    return out


def _kendall(xs, ys):
    pairs, discordant, concordant = [], 0, 0
    keys = list(xs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sign = (xs[a] - xs[b]) * (ys[a] - ys[b])
            if sign < 0:
                discordant += 1
                pairs.append([a, b])
            elif sign > 0:
                concordant += 1
    n = len(keys)
    return (concordant - discordant) / (n * (n - 1) / 2), pairs


@task("T009", changed_files=CHANGED, regression_tests=(_test("test_t009_columns_rank_paths_differently"),))
def lateral_heading_columns(ctx):
    rows = {}
    for key in COLUMN_PATHS:
        tr = _fine_transfer(ctx, key)
        y = tr.states
        k = tr.curvature_along()
        rows[key] = {"length": gj.path(key).length, "j_lat_end": float(y[-1, 4]), "j_head_end": float(y[-1, 6]),
                     "j_lat_rate_end": float(y[-1, 5]), "j_head_rate_end": float(y[-1, 7]),
                     "max_abs_j_lat": float(np.max(np.abs(y[:, 4]))), "max_abs_j_head": float(np.max(np.abs(y[:, 6]))),
                     "k_min": float(k.min()), "k_max": float(k.max()),
                     "determinant_end": float(tr.determinant()[-1])}
    lateral = {k: abs(r["j_lat_end"]) for k, r in rows.items()}
    heading = {k: abs(r["j_head_end"]) for k, r in rows.items()}
    tau, discordant = _kendall(lateral, heading)
    a, b = WITNESS_PAIR
    confirm = {key: _endpoint_confirmation(ctx, key) for key in WITNESS_PAIR}
    confirm_error = max(abs(confirm[key][c] - rows[key][f"j_{'lat' if c == 'lateral' else 'head'}_end"])
                        / max(1.0, abs(rows[key][f"j_{'lat' if c == 'lateral' else 'head'}_end"]))
                        for key in WITNESS_PAIR for c in ("lateral", "heading"))
    reversal = min(lateral[b] - lateral[a], heading[a] - heading[b])
    model = {}
    for key in ("plane", "hyperbolic-long", "sphere-great-circle"):
        kk = _constant_curvature_of(key)
        length = gj.path(key).length
        _, _, sn, _ = jacobi.constant_curvature(kk, [length])
        cn = jacobi.constant_curvature(kk, [length])[0]
        model[key] = {"ratio_head_over_lat": rows[key]["j_head_end"] / rows[key]["j_lat_end"],
                      "model_ratio": float(sn[0] / cn[0])}
    ctx.artifact_json("columns.json", _plain({"rows": rows, "kendall_tau": tau, "discordant_pairs": discordant,
                                              "witness_pair": list(WITNESS_PAIR), "fd_confirmation": confirm,
                                              "fd_eps": CONFIRM_EPS, "model_ratios": model}))
    order = sorted(COLUMN_PATHS, key=lambda k: lateral[k])
    ctx.artifact_text("ranking.svg", svg.line_plot(
        [("|j_lat(L)|", range(1, len(order) + 1), [lateral[k] for k in order]),
         ("|j_head(L)|", range(1, len(order) + 1), [heading[k] for k in order])],
        title="Paths ordered by lateral sensitivity (" + ", ".join(order) + ")", xlabel="rank by |j_lat(L)|",
        ylabel="endpoint sensitivity", logy=True))
    series = []
    for key in WITNESS_PAIR:
        tr = _fine_transfer(ctx, key)
        series += [(f"{key} j_lat", tr.s, tr.states[:, 4]), (f"{key} j_head", tr.s, tr.states[:, 6])]
    ctx.artifact_text("witness-columns.svg", svg.line_plot(series, title="Equal-length torus paths: columns along s",
                                                           xlabel="arclength s", ylabel="j", markers=False))

    generator = {"name": "declared geodesics", "paths": list(COLUMN_PATHS)}
    findings = [finding(
        "Endpoint sensitivities to lateral offset (|j_lat(L)|) and heading error (|j_head(L)|) per path",
        "numerical", {k: {"lateral": lateral[k], "heading": heading[k]} for k in COLUMN_PATHS},
        {"generator": generator, "checks": [
            gj.check("invariant", "max |det Phi(L) - 1| over paths",
                     max(abs(r["determinant_end"] - 1) for r in rows.values()), 1e-7)] + [
            gj.check("analytic", f"j_head/j_lat at L against sn_K/cn_K on {k}",
                     v["ratio_head_over_lat"] - v["model_ratio"], 1e-6) for k, v in model.items()] + [
            gj.check("self_convergence", "central-difference endpoint separation (eps 1e-3) against the columns",
                     confirm_error, 1e-4)]},
        tolerance=TOL_VALUE)]
    findings.append(finding(
        "Lateral and heading sensitivities rank paths differently", "numerical",
        {"kendall_tau": tau, "discordant_pairs": len(discordant), "witness_reversal_margin": reversal},
        {"generator": generator, "checks": [
            gj.check("invariant", f"reversal margin for the equal-length pair {a} / {b}", reversal, 0.5, "ge"),
            gj.check("invariant", "discordant path pairs", len(discordant), 1.0, "ge")]},
        tolerance={"abs": 1e-6, "rel": 1e-6},
        counterexample={"statement": "Ranking paths by sensitivity to heading error gives the same order as ranking "
                                     "by sensitivity to lateral offset",
                        "witness": {"paths": list(WITNESS_PAIR), "length": gj.path(a).length,
                                    "lateral": {a: lateral[a], b: lateral[b]},
                                    "heading": {a: heading[a], b: heading[b]}}}))
    findings.append(finding(
        "Which starting error dominates the endpoint error of real tool or vehicle paths on physical curved parts",
        "physical", None, {}))
    return _outcome(
        "completed", findings,
        hypothesis=("The lateral column (curvature weighted early) and the heading column (curvature weighted late, "
                    "j ~ s at the start) respond differently to where curvature sits along a path, so they can "
                    "order paths differently."),
        mathematical_model=("Endpoint normal displacement = j_lat(L) delta_perp + j_head(L) delta_alpha; model "
                            "spaces give j_head/j_lat = tan(sqrt(K)L)/sqrt(K), L, tanh(sqrt(-K)L)/sqrt(-K)."),
        input_data=[f"{len(COLUMN_PATHS)} declared paths: {', '.join(COLUMN_PATHS)}",
                    f"Witness pair {a} / {b}: same torus, same length 3, curvature order reversed"],
        observation_model=("|j(L)| per unit perturbation; confirmed on the witness pair by central differences of "
                           f"perturbed geodesics (eps = {CONFIRM_EPS})."),
        expected_invariant="det Phi(L) = 1; model-space ratios; a reversal on the witness pair.",
        experiment=("Integrate both columns on every path, rank paths by each column, count discordant pairs "
                    "(Kendall tau), and confirm the witness numbers by finite differences."),
        numerical_result=(f"Kendall tau {_fmt(tau, 3)} with {len(discordant)} discordant pairs; witness lateral "
                          f"{_fmt(lateral[a], 4)} vs {_fmt(lateral[b], 4)}, heading {_fmt(heading[a], 4)} vs "
                          f"{_fmt(heading[b], 4)}; finite-difference confirmation error {_fmt(confirm_error)}."),
        uncertainty="Column values carry RK4 error near 1e-8; the reversal margin exceeds it by eight orders.",
        failure_modes_checked=["sign of the columns (magnitudes ranked)", "equal-length comparison for the witness",
                               "finite-difference confirmation of the endpoint sensitivities"],
        unresolved_assumptions=["Rankings are over declared paths of unequal lengths except the witness pair",
                                "No physical platform, tool or perturbation statistics were measured"],
        recommended_next_task="T010: generate near-focus and post-focus counterexamples")
