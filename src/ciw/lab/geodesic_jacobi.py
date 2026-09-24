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

# Charts with nonzero Christoffel symbols along their standard paths (five curved surfaces and the two flat polar
# charts, whose declared paths differ); on the flat Cartesian charts every method is exact.
GAMMA_CHARTS = ("sphere", "saddle", "torus", "gaussian-bump", "hyperbolic-plane", "plane-polar", "cylinder-polar")
FLAT_CARTESIAN = ("plane", "cylinder")
NOMINAL_ORDER = {"euler": 1, "midpoint": 2, "rk4": 4}


def _test(name: str) -> str:
    return f"{TESTS}::{name}"


# Section-wide contract tests (per-finding uncertainty, figure legibility) cover every task.
SECTION_TESTS = (_test("test_every_numerical_finding_declares_uncertainty_and_tolerance"),
                 _test("test_figures_fit_their_legend_and_title_space"),
                 _test("test_every_next_step_is_a_deferred_research_question"))


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
    """Declared constant curvature of a constant-curvature path, from surface parameters only.

    Never evaluated through ``gaussian_curvature``: the integrated Jacobi
    equation uses that method, so a reference built from it could not detect
    a wrong curvature.
    """
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    if key == "torus-outer-equator":
        return 1.0 / (surf.minor * (surf.major + surf.minor))
    if key == "torus-inner-equator":
        return -1.0 / (surf.minor * (surf.major - surf.minor))
    if spec.surface == "sphere":
        return 1.0 / surf.radius ** 2
    if spec.surface == "hyperbolic-plane":
        return -surf.k ** 2
    if spec.surface in ("plane", "cylinder") + gj.POLAR_KEYS:
        return 0.0  # the plane, and the cylinder's development, are flat
    raise ValueError(f"Path {key} has no declared constant curvature")


def _unc(kind: str, value, basis: str) -> dict:
    """Per-finding uncertainty object (AUTHORING rule 5)."""
    return {"kind": kind, "value": float(value), "basis": basis}


def _fit_spread(pairwise: dict, fitted: dict) -> float:
    """Largest distance of a pairwise log-log slope from its least-squares fit."""
    return max(abs(v - fitted[k]) for k, values in pairwise.items() for v in values)


def _pairwise(entries, x, y) -> list:
    return [gj.slope([a[x], b[x]], [a[y], b[y]]) for a, b in zip(entries, entries[1:])]


def _speed_drift(surf, states) -> np.ndarray:
    return np.array([surf.speed_squared(y[:2], y[2:4]) - 1.0 for y in states])


# ---------------------------------------------------------------------------
# T001: symbolic re-derivation
# ---------------------------------------------------------------------------
T001_CHARTS = gj.CATALOGUE_KEYS + gj.POLAR_KEYS
EMBEDDED_CHARTS = ("plane", "sphere", "cylinder", "saddle", "torus", "gaussian-bump")
SAMPLES_PER_CHART = 8
# The catalogue's declared boxes plus the polar charts (r >= 0.5 keeps clear of r = 0).
SAMPLE_BOXES = {**surfaces.SAMPLING_DOMAINS, "plane-polar": ((0.5, 2.5), (-math.pi, math.pi)),
                "cylinder-polar": ((0.5, 2.5), (-math.pi, math.pi))}


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


def _monge(fx, fy, fxx, fxy, fyy):
    """Monge-patch metric and Christoffel symbols: g = I + grad f grad f^T, Gamma^k_ij = f_k f_ij / W^2."""
    grad, hess = np.array([fx, fy]), np.array([[fxx, fxy], [fxy, fyy]])
    w2 = 1.0 + fx * fx + fy * fy
    return np.eye(2) + np.outer(grad, grad), np.einsum("k,ij->kij", grad, hess) / w2, w2


def hand_geometry(key: str, u, v) -> dict:
    """Metric, Christoffel symbols, geodesic acceleration and K from the hand table of the section doc (T001).

    A Python transcription of the closed forms written in
    docs/lab/GEODESIC_JACOBI.md, using only the surface parameters, so a
    comparison with ``ciw.lab.surfaces`` tests this transcription and the
    implementation against each other. The Markdown table itself is not
    parsed. ``gamma[k, i, j]`` is Gamma^k_ij, as in ``ciw.lab.surfaces``.
    """
    surf = gj.surface(key)
    x, y = float(u[0]), float(u[1])
    gamma = np.zeros((2, 2, 2))
    acceleration = None
    if key == "plane":
        g, curvature = np.eye(2), 0.0
    elif key == "cylinder":
        g, curvature = np.diag([surf.radius ** 2, 1.0]), 0.0
    elif key == "sphere":
        g = surf.radius ** 2 * np.diag([1.0, math.sin(x) ** 2])
        gamma[0, 1, 1] = -math.sin(x) * math.cos(x)
        gamma[1, 0, 1] = gamma[1, 1, 0] = math.cos(x) / math.sin(x)
        curvature = 1.0 / surf.radius ** 2
    elif key == "saddle":
        c = surf.c
        g, gamma, w2 = _monge(c * x, -c * y, c, 0.0, -c)
        curvature = -c * c / (1.0 + c * c * (x * x + y * y)) ** 2
    elif key == "gaussian-bump":
        h, s2 = surf.h, surf.sigma ** 2
        rho2 = x * x + y * y
        f = h * math.exp(-rho2 / (2 * s2))
        g, gamma, w2 = _monge(-x * f / s2, -y * f / s2, (x * x / s2 - 1) * f / s2, x * y * f / s2 ** 2,
                              (y * y / s2 - 1) * f / s2)
        curvature = h * h * math.exp(-rho2 / s2) * (1 - rho2 / s2) / (s2 ** 2 * w2 ** 2)
    elif key == "torus":
        big, r = surf.major, surf.minor
        rho = big + r * math.cos(y)
        g = np.diag([rho ** 2, r ** 2])
        gamma[0, 0, 1] = gamma[0, 1, 0] = -r * math.sin(y) / rho
        gamma[1, 0, 0] = rho * math.sin(y) / r
        curvature = math.cos(y) / (r * rho)
        # The doc's explicit geodesic equations, written out rather than read from Gamma.
        acceleration = np.array([2 * r * math.sin(y) * v[0] * v[1] / rho, -rho * math.sin(y) * v[0] ** 2 / r])
    elif key == "hyperbolic-plane":
        g = np.eye(2) / (surf.k * y) ** 2
        gamma[0, 0, 1] = gamma[0, 1, 0] = -1.0 / y
        gamma[1, 0, 0] = 1.0 / y
        gamma[1, 1, 1] = -1.0 / y
        curvature = -surf.k ** 2
        acceleration = np.array([2 * v[0] * v[1] / y, (v[1] ** 2 - v[0] ** 2) / y])
    elif key in gj.POLAR_KEYS:
        if key == "cylinder-polar" and surf.base.radius != 1.0:
            raise ValueError("The hand table covers the polar chart of the unit cylinder only")
        g = np.diag([1.0, x * x])
        gamma[0, 1, 1] = -x
        gamma[1, 0, 1] = gamma[1, 1, 0] = 1.0 / x
        curvature = 0.0
    else:
        raise KeyError(f"No hand derivation for {key}")
    if acceleration is None:
        acceleration = -np.einsum("kij,i,j->k", gamma, v, v)
    return {"metric": g, "christoffel": gamma, "acceleration": acceleration, "curvature": curvature}


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


SHORT = {"saddle": "saddle", "torus": "torus", "gaussian-bump": "bump"}


def _curvature_figure(ctx, have_sympy):
    series = []
    for key in gj.VARIABLE_KEYS:
        tr = _fine_transfer(ctx, key)
        stride = max(1, len(tr.s) // 40)
        s = tr.s[::stride]
        pts = tr.points[::stride]
        series.append((f"{SHORT[key]} ciw", s, [gj.surface(key).gaussian_curvature(u) for u in pts]))
        if have_sympy:
            curvature = _lambdas(ctx, key)["curvature"]
            series.append((f"{SHORT[key]} sympy", s, [float(curvature(*u)) for u in pts]))
        else:
            series.append((f"{SHORT[key]} intrinsic FD", s, [intrinsic_curvature(gj.surface(key), u) for u in pts]))
    return svg.line_plot(series, title="Gaussian curvature along the standard paths",
                         xlabel="arclength s", ylabel="K(gamma(s))", markers=False)


def extrinsic_curvature(surf, u) -> float:
    """(LN - M^2) / det g through the generic EmbeddedSurface route (the catalogue overrides it with closed forms)."""
    return surfaces.EmbeddedSurface.gaussian_curvature(surf, u)


@task("T001", changed_files=CHANGED,
      regression_tests=(_test("test_t001_symbolic_derivations_match_surfaces"),
                        _test("test_t001_without_sympy_is_partial"),
                        _test("test_t001_self_consistency_helpers")) + SECTION_TESTS)
def rederive_geodesic_equations(ctx):
    have_sympy = ctx.available("module:sympy")
    rows, texts, latex = {}, [], []
    for key in T001_CHARTS:
        surf = gj.surface(key)
        points, velocities = sample_points(key)
        row = {"symmetry": 0.0, "compatibility": 0.0, "metric_derivatives": 0.0, "intrinsic_curvature": 0.0,
               "max_abs_christoffel": 0.0, "max_abs_intrinsic_curvature": 0.0, "hand_metric": 0.0,
               "hand_christoffel": 0.0, "hand_acceleration": 0.0, "hand_curvature": 0.0}
        if key in EMBEDDED_CHARTS:
            row.update(extrinsic_vs_closed_form=0.0, extrinsic_vs_intrinsic=0.0)
        if have_sympy:
            functions = _lambdas(ctx, key)
            derived = _derived(ctx, key)
            texts.append(gj.text_block(key, derived))
            latex.append(gj.latex_block(key, derived))
            row.update(metric=0.0, christoffel=0.0, acceleration=0.0, curvature=0.0,
                       symbolic_curvature_is_zero=bool(derived["sympy"].simplify(derived["curvature"]) == 0))
            if key in EMBEDDED_CHARTS:
                row["extrinsic_vs_brioschi"] = 0.0
        for u, v in zip(points, velocities):
            surf.check(u)
            gamma = surf.christoffel(u)
            k_ciw = surf.gaussian_curvature(u)
            k_intrinsic = intrinsic_curvature(surf, u)
            acc_ciw = surf.geodesic_rhs(np.concatenate([u, v]))[2:]
            hand = hand_geometry(key, u, v)
            row["symmetry"] = max(row["symmetry"], float(np.max(np.abs(gamma - gamma.transpose(0, 2, 1)))))
            row["compatibility"] = max(row["compatibility"], compatibility_residual(surf, u))
            row["metric_derivatives"] = max(row["metric_derivatives"], metric_derivative_residual(surf, u))
            row["intrinsic_curvature"] = max(row["intrinsic_curvature"], _discrepancy(k_intrinsic, k_ciw))
            row["max_abs_christoffel"] = max(row["max_abs_christoffel"], float(np.max(np.abs(gamma))))
            row["max_abs_intrinsic_curvature"] = max(row["max_abs_intrinsic_curvature"], abs(k_intrinsic))
            for name, ours in (("metric", surf.metric(u)), ("christoffel", gamma), ("acceleration", acc_ciw),
                               ("curvature", k_ciw)):
                row[f"hand_{name}"] = max(row[f"hand_{name}"], _discrepancy(ours, hand[name]))
            if key in EMBEDDED_CHARTS:
                k_ext = extrinsic_curvature(surf, u)
                row["extrinsic_vs_closed_form"] = max(row["extrinsic_vs_closed_form"], _discrepancy(k_ext, k_ciw))
                row["extrinsic_vs_intrinsic"] = max(row["extrinsic_vs_intrinsic"], _discrepancy(k_ext, k_intrinsic))
            if have_sympy:
                g_sym = np.array(functions["metric"](*u), dtype=float).reshape(2, 2)
                gamma_sym = np.array(functions["christoffel"](*u), dtype=float).reshape(2, 2, 2)
                acc_sym = np.array(functions["rhs"](u[0], u[1], v[0], v[1]), dtype=float)[2:]
                k_sym = float(functions["curvature"](*u))
                row["metric"] = max(row["metric"], _discrepancy(surf.metric(u), g_sym))
                row["christoffel"] = max(row["christoffel"], _discrepancy(gamma, gamma_sym))
                row["acceleration"] = max(row["acceleration"], _discrepancy(acc_ciw, acc_sym))
                row["curvature"] = max(row["curvature"], _discrepancy(k_ciw, k_sym))
                if key in EMBEDDED_CHARTS:
                    row["extrinsic_vs_brioschi"] = max(row["extrinsic_vs_brioschi"],
                                                       _discrepancy(extrinsic_curvature(surf, u), k_sym))
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
        ctx.artifact_text("derivations.tex", gj.latex_document(latex))
    ctx.artifact_text("curvature-along-paths.svg", _curvature_figure(ctx, have_sympy))

    generator = {"name": "seeded chart samples", "seed": gj.SEED, "samples_per_chart": SAMPLES_PER_CHART,
                 "charts": list(T001_CHARTS)}
    roundoff = _unc("roundoff", 1e-15, "binary64 evaluation of both sides, relative to max(1, |value|)")
    fd_intrinsic = max(r["intrinsic_curvature"] for r in rows.values())
    fd_basis = "central differences of Christoffel symbols with step 1e-4 (O(step^2) truncation plus rounding)"
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
            uncertainty=roundoff, tolerance=TOL_SMALL))
        curvature = max(r["curvature"] for r in rows.values())
        findings.append(finding(
            "Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts",
            "numerical", {k: r["curvature"] for k, r in rows.items()},
            {"generator": generator, "independent_check": gj.independent(
                gj.check("analytic", "sympy Brioschi formula (first fundamental form only)", curvature, 1e-10),
                "ciw.lab.surfaces", "sympy", checker_revision=sympy_version)},
            uncertainty=roundoff, tolerance=TOL_SMALL))
    hand = {k: {name: r[f"hand_{name}"] for name in ("metric", "christoffel", "acceleration", "curvature")}
            for k, r in rows.items()}
    findings.append(finding(
        "The hand-derived metrics, Christoffel symbols, geodesic equations and curvatures of the section doc, as "
        "transcribed in hand_geometry, match ciw.lab.surfaces on nine charts", "mathematical", hand,
        {"derivation": f"{DOC}, section T001 (metric, Christoffel symbols and curvature of each chart by hand)",
         "generator": generator, "checks": [
             gj.check("analytic", f"hand-derived {name} (hand_geometry, the Python transcription of the doc "
                      "table) against ciw.lab.surfaces, all charts",
                      max(h[name] for h in hand.values()), 1e-12)
             for name in ("metric", "christoffel", "acceleration", "curvature")]},
        uncertainty=roundoff, tolerance=TOL_SMALL))
    consistency = {k: {name: r[name] for name in ("symmetry", "compatibility", "metric_derivatives")}
                   for k, r in rows.items()}
    fd_metric = max(r["metric_derivatives"] for r in rows.values())
    findings.append(finding(
        "ciw.lab.surfaces Christoffel symbols are symmetric and metric-compatible and its exact metric derivatives "
        "match finite differences", "numerical", consistency,
        {"generator": generator, "checks": [
            gj.check("invariant", "Gamma^k_ij - Gamma^k_ji", max(r["symmetry"] for r in rows.values()), 1e-14),
            gj.check("invariant", "nabla g = 0 residual", max(r["compatibility"] for r in rows.values()), 1e-12),
            gj.check("self_convergence", "central differences of g with step 1e-5", fd_metric, 1e-6)]},
        uncertainty=_unc("truncation_bound", fd_metric, "the metric-derivative residual is the truncation and "
                         "rounding of central differences with step 1e-5; symmetry and compatibility are rounding"),
        tolerance={"abs": 1e-7, "rel": 0.0}))
    findings.append(finding(
        "Intrinsic curvature from finite-differenced Christoffel symbols matches the ciw Gaussian curvature",
        "numerical", {k: r["intrinsic_curvature"] for k, r in rows.items()},
        {"generator": generator, "checks": [gj.check(
            "self_convergence", "Riemann tensor from central differences (step 1e-4) of ciw Christoffel symbols",
            fd_intrinsic, 1e-6)]},
        uncertainty=_unc("truncation_bound", fd_intrinsic, fd_basis), tolerance={"abs": 1e-6, "rel": 0.0}))
    egregium = {k: {name: rows[k][name] for name in ("extrinsic_vs_closed_form", "extrinsic_vs_intrinsic",
                                                     "extrinsic_vs_brioschi") if name in rows[k]}
                for k in EMBEDDED_CHARTS}
    egregium_basis = {"generator": dict(generator, charts=list(EMBEDDED_CHARTS)), "checks": [
        gj.check("analytic", "generic (LN - M^2)/det g against the closed-form curvatures",
                 max(e["extrinsic_vs_closed_form"] for e in egregium.values()), 1e-12),
        gj.check("self_convergence", "generic (LN - M^2)/det g against the finite-difference Riemann curvature",
                 max(e["extrinsic_vs_intrinsic"] for e in egregium.values()), 1e-6)]}
    if have_sympy:
        egregium_basis["independent_check"] = gj.independent(
            gj.check("analytic", "sympy Brioschi curvature (first fundamental form only) against the generic "
                     "second-fundamental-form curvature", max(e["extrinsic_vs_brioschi"] for e in egregium.values()),
                     1e-10), "ciw.lab.surfaces.EmbeddedSurface.gaussian_curvature", "sympy",
            checker_revision=gj.optional_version("sympy"))
    findings.append(finding(
        "The second-fundamental-form curvature (LN - M^2)/det g equals the intrinsic curvature on the six embedded "
        "charts (Theorema Egregium)", "numerical", egregium, egregium_basis,
        uncertainty=_unc("truncation_bound", max(e["extrinsic_vs_intrinsic"] for e in egregium.values()), fd_basis),
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
        uncertainty=_unc("truncation_bound", max(p["max_abs_intrinsic_curvature"] for p in polar.values()),
                         "finite-difference curvature of a flat chart is pure truncation and rounding"),
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
        uncertainty=roundoff, tolerance=TOL_VALUE,
        counterexample={"statement": "A surface that bends in space has nonzero Christoffel symbols in every chart",
                        "witness": {"chart": "cylinder (phi, z)", "normal_curvature_phi": normal_curvature}}))

    state = "completed" if have_sympy else "partial"
    worst = max(max(r.get("metric", 0), r.get("christoffel", 0), r.get("acceleration", 0), r.get("curvature", 0))
                for r in rows.values())
    worst_hand = max(max(h.values()) for h in hand.values())
    return _outcome(
        state, findings,
        hypothesis=("The metric, Christoffel symbols, geodesic equations and Gaussian curvature derived by hand (doc "
                    "table) and symbolically from each chart's embedding (or intrinsic metric) coincide with the "
                    "exact-derivative implementation in ciw.lab.surfaces, and the extrinsic and intrinsic "
                    "curvatures agree."),
        mathematical_model=("g_ij = X_i . X_j (or the declared intrinsic metric), Gamma^k_ij = 1/2 g^kl (d_i g_jl + "
                            "d_j g_il - d_l g_ij), u''^k = -Gamma^k_ij u'^i u'^j (first variation of length), K from "
                            "the second fundamental form (LN - M^2)/det g, from the Brioschi formula (sympy) and from "
                            "the Riemann tensor of finite-differenced Christoffel symbols."),
        input_data=[f"Charts: {', '.join(T001_CHARTS)} with catalogue parameters",
                    f"{SAMPLES_PER_CHART} seeded points and velocities per chart (PCG64 seed {gj.SEED} + 1000 i) in "
                    "ciw.lab.surfaces.SAMPLING_DOMAINS plus r in [0.5, 2.5] on the polar charts"],
        observation_model=("Relative discrepancy max |a - b| / max(1, |b|) over samples and components, evaluated in "
                           "binary64."),
        expected_invariant=("Hand, symbolic and numeric quantities agree to rounding; Gamma is symmetric; nabla g = 0; "
                            "the generic second-fundamental-form curvature equals the intrinsic (finite-difference "
                            "Riemann and Brioschi) curvature on every embedded chart (Theorema Egregium)."),
        experiment=("Evaluate the hand table and (when installed) the sympy derivation at seeded points and compare "
                    "with ciw.lab.surfaces; check symmetry, metric compatibility and metric derivatives of ciw alone; "
                    "compare the generic extrinsic curvature with the intrinsic ones; record the flat polar charts and "
                    "the cylinder as counterexamples separating Christoffel symbols from curvature."),
        numerical_result=((f"Largest sympy/ciw discrepancy {_fmt(worst)}; " if have_sympy else
                           "sympy unavailable: symbolic comparison not run; ")
                          + f"largest hand-table/ciw discrepancy {_fmt(worst_hand)}; largest intrinsic-curvature "
                          f"discrepancy {_fmt(fd_intrinsic)}; largest extrinsic-intrinsic discrepancy "
                          f"{_fmt(max(e['extrinsic_vs_intrinsic'] for e in egregium.values()))}."),
        uncertainty=("Binary64 rounding of both sides (about 1e-15 relative) for the hand and symbolic comparisons; "
                     "finite-difference checks carry a truncation error near 1e-8."),
        failure_modes_checked=["Christoffel index order (symmetry and compatibility)",
                               "sign convention of the Riemann tensor (sphere K = +1, hyperbolic plane K = -1)",
                               "chart singularities excluded from the sampling boxes",
                               "intrinsic versus extrinsic curvature (cylinder, polar charts, Theorema Egregium)",
                               "transcription errors in hand_geometry, the Python transcription of the doc's hand "
                               "table (every row evaluated against the implementation)"],
        unresolved_assumptions=(["sympy simplification is trusted to be correct; it is an independent implementation, "
                                 "not a proof checker"] if have_sympy else
                                ["sympy is not installed: the derivation is checked only through the hand table in "
                                 f"{DOC} and ciw self-consistency"])
        + ["Only 8 sample points per chart are compared", "The hand table covers the unit-radius polar cylinder only",
           f"The Markdown table in {DOC} is not parsed: what is checked is its Python transcription hand_geometry, "
           "and the correspondence between the two rests on review"],
        recommended_next_task=("Deferred research question: parse the hand table in docs/lab/GEODESIC_JACOBI.md "
                               "itself (not its Python transcription hand_geometry), evaluate it at more than 8 "
                               "points per chart and add polar cylinders of radius other than 1, so that a table "
                               "edit that diverges from ciw.lab.surfaces fails a check instead of resting on review"))


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
      regression_tests=(_test("test_t002_references_agree"), _test("test_t002_without_optional_modules"))
      + SECTION_TESTS)
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
    order = list(rows)
    ctx.artifact_json("references.json", _plain({
        "richardson_steps": [RICHARDSON_STEPS, 2 * RICHARDSON_STEPS], "mp_dps": gj.MP_DPS,
        "mp_macro_steps": list(gj.MP_MACRO_STEPS), "paths": {k: gj.path(k).as_dict() for k in gj.STANDARD},
        "figure_path_index": {str(i + 1): k for i, k in enumerate(order)},
        "rows": rows, "clairaut": clairaut, "clairaut_initial": c0}))
    series = [("ciw Richardson", range(1, len(rows) + 1), [max(rows[k]["ciw_vs_reference"]["max"], 1e-17)
                                                           for k in order])]
    if have_scipy:
        series.append(("scipy DOP853", range(1, len(rows) + 1),
                       [max(rows[k]["scipy_vs_reference"]["max"], 1e-17) for k in order]))
    ctx.artifact_text("agreement.svg", svg.line_plot(
        series, title="End-state gap to the reference (paths: references.json)",
        xlabel="path index (figure_path_index)", ylabel="max(position, transfer) gap", logy=True,
        rounding=64 * float(np.finfo(float).eps)), rounding_level=True)
    ctx.artifact_text("clairaut.svg", svg.line_plot(
        [(n, s, np.maximum(d, 1e-17)) for n, s, d in curves], title="Torus Clairaut drift |C(s) - C(0)|",
        xlabel="arclength s", ylabel="|rho^2 phi' - C0|", logy=True, markers=False))

    findings = []
    # Without extrapolation the RK4 end state misses the reference by about the Richardson estimate itself, so a
    # gap far below it shows that the extrapolation (not just a fine step) produced the agreement.
    extrapolated = {k: rows[k]["ciw_vs_reference"]["max"] / rows[k]["richardson_error_estimate"]
                    for k in gj.STANDARD if rows[k]["richardson_error_estimate"] >= 1e-13}
    closed = [k for k in gj.STANDARD if rows[k]["reference_kind"] == "closed_form"]
    closed_basis = {"checks": [gj.check("analytic", f"closed-form geodesic and constant-curvature transfer on {k}",
                                        rows[k]["ciw_vs_reference"]["max"], 1e-12) for k in closed]
                    + [gj.check("self_convergence", "gap over the Richardson error estimate on the curved "
                                "closed-form charts (about 1 without extrapolation)",
                                max(v for k, v in extrapolated.items() if k in closed), 0.1, "le")]}
    if have_scipy:
        # Not an independent check: scipy supplies only the integrator; the right-hand side and the closed forms
        # the claim is about are ciw code.
        closed_basis["checks"].append(gj.check(
            "high_precision", "scipy DOP853 rtol 1e-13 on the ciw right-hand side (only the integrator is "
            "independent; the equations and closed forms are ciw code)",
            max(rows[k]["ciw_vs_scipy"]["max"] for k in closed), 1e-12))
    findings.append(finding(
        "ciw Richardson RK4 end states match closed-form geodesics and transfer matrices on the six closed-form charts",
        "numerical", {k: rows[k]["ciw_vs_reference"]["max"] for k in closed}, closed_basis,
        uncertainty=_unc("reference_error", 1e-15, "closed forms evaluated in binary64; the retained ciw Richardson "
                         "error estimates are up to "
                         + _fmt(max(rows[k]["richardson_error_estimate"] for k in closed))),
        tolerance=TOL_SMALL))
    for key in gj.VARIABLE_KEYS:
        row = rows[key]
        gap = row["ciw_vs_reference"]["max"]
        value = {"reference_end_state": reference(ctx, key)["state"], "ciw_minus_reference": gap}
        extrapolation = gj.check("self_convergence", f"gap over the Richardson error estimate on {key} (about 1 "
                                 "without extrapolation)", gap / row["richardson_error_estimate"], 0.1, "le")
        if row["reference_kind"] == "mpmath":
            checks = [gj.check("high_precision", "Gragg-Bulirsch-Stoer macro-step 10 vs 20 at 34 digits",
                               row["reference_error_estimate"], 1e-18), extrapolation]
            if have_scipy:
                checks.append(gj.check("high_precision", "scipy DOP853 vs the 34-digit reference",
                                       row["scipy_vs_reference"]["max"], 1e-11))
            # The independent part is the equations: sympy derives the metric, Christoffel symbols and Brioschi
            # curvature from the embedding. The extrapolation integrator (gbs_integrate) is ciw-authored and mpmath
            # supplies only the arithmetic, so neither is named as the checker.
            basis = {"checks": checks, "independent_check": gj.independent(
                gj.check("high_precision", f"34-digit integration of the sympy-derived {key} geodesic and Jacobi "
                         "equations by the ciw-authored Gragg-Bulirsch-Stoer integrator in mpmath arithmetic (the "
                         "equations are independent; the integrator is not)", gap, 1e-12),
                "ciw.lab.integrators.richardson_rk4", "sympy.lambdify(derived geodesic and Jacobi equations)",
                checker_revision=gj.optional_version("sympy"))}
            claim = (f"ciw Richardson RK4 matches a 34-digit integration of the sympy-derived equations on the {key} "
                     "path")
            # The reported end state and gap are binary64: rounding bounds their uncertainty from below.
            scale = max(1.0, float(np.max(np.abs(row["reference_position"]))),
                        float(np.max(np.abs(row["reference_matrix"]))))
            rounding = 4 * float(np.finfo(float).eps) * scale
            estimate = row["reference_error_estimate"]
            uncertainty = _unc("roundoff" if rounding >= estimate else "reference_error", max(rounding, estimate),
                               f"larger of the binary64 rounding of the reference end state and of the gap evaluation "
                               f"(4 eps times the end-state scale {_fmt(scale)}: {_fmt(rounding)}) and the "
                               f"Gragg-Bulirsch-Stoer 10 vs 20 macro-step difference at 34 digits ({_fmt(estimate)}); "
                               f"the ciw Richardson estimate on this path is {_fmt(row['richardson_error_estimate'])}")
        elif row["reference_kind"] == "scipy":
            basis = {"checks": [extrapolation], "independent_check": gj.independent(
                gj.check("high_precision", f"scipy DOP853 rtol 1e-13 on {key} (ciw right-hand side: only the "
                         "integrator is independent)", gap, 1e-12),
                "ciw.lab.integrators.richardson_rk4", "scipy.integrate.solve_ivp(DOP853)",
                checker_revision=gj.optional_version("scipy"))}
            claim = (f"ciw Richardson RK4 matches scipy DOP853 integrating the same ciw equations on the {key} path "
                     "(sympy-derived reference unavailable)")
            uncertainty = _unc("reference_error", 1e-13, "requested DOP853 tolerance; scipy returns no global error "
                               "estimate, and it integrates the ciw equations, so only the integrator is independent")
        else:
            basis = {"checks": [gj.check("self_convergence", f"ciw Richardson RK4 at 800 vs 200 steps on {key}",
                                         gap, 1e-12)]}
            claim = f"ciw Richardson RK4 is self-convergent on the {key} path (no independent reference available)"
            uncertainty = _unc("truncation_bound", reference(ctx, key)["error_estimate"],
                               "Richardson error estimate of the 800-step reference")
        findings.append(finding(claim, "numerical", value, basis, uncertainty=uncertainty, tolerance=TOL_VALUE))
    clairaut_checks = [gj.check("invariant", "Clairaut drift of the ciw Richardson end state",
                                clairaut["ciw_richardson"], 1e-12)]
    if "mpmath_reference" in clairaut:
        clairaut_checks.append(gj.check("invariant", "Clairaut drift of the mpmath reference end state",
                                        clairaut["mpmath_reference"], 1e-20))
    findings.append(finding(
        "Clairaut's integral rho^2 phi' is conserved along the torus reference path", "numerical",
        {name: clairaut[name] for name in sorted(clairaut)}, {"checks": clairaut_checks},
        uncertainty=_unc("roundoff", 1e-15 * max(1.0, abs(c0)), "binary64 evaluation of rho^2 phi'"),
        tolerance=TOL_SMALL))

    state = "completed" if (have_scipy and have_mp) else "partial"
    missing = [name for name, ok in (("scipy", have_scipy), ("sympy+mpmath", have_mp)) if not ok]
    worst = max(r["ciw_vs_reference"]["max"] for r in rows.values())
    mp_estimates = [r["reference_error_estimate"] for r in rows.values() if r["reference_kind"] == "mpmath"]
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
        uncertainty=((f"The 34-digit references change by at most {_fmt(max(mp_estimates))} between 10 and 20 "
                      "macro-steps, but the reported end states and gaps are binary64, so each per-finding "
                      "uncertainty is at least their binary64 rounding (about 1e-15); " if mp_estimates else
                      "Without the 34-digit references the variable-curvature uncertainty is the scipy tolerance or "
                      "the Richardson estimate; ")
                     + "closed forms carry rounding only; the ciw Richardson error estimate is retained per path."),
        failure_modes_checked=["reference and integrator start from the same binary64 state",
                               "reference equations derived independently of ciw.lab.surfaces (sympy)",
                               "extrapolation self-consistency (macro-step halving)",
                               "missing or wrong Richardson extrapolation (gap must be far below the RK4 estimate)",
                               "Clairaut invariant on the torus"],
        unresolved_assumptions=(["Unavailable optional modules: " + ", ".join(missing)] if missing else [])
        + ([] if have_mp else ["Without sympy the variable-curvature references integrate the ciw equations "
                               "(scipy) or reuse ciw RK4: the integrator may be independent, the equations are not"])
        + ["The 34-digit reference is an extrapolated integration, not a closed form; its error estimate is "
           "empirical",
           "Only the equations of the 34-digit reference are independent of ciw (sympy derives them); its "
           "Gragg-Bulirsch-Stoer integrator is ciw-authored and mpmath supplies the arithmetic. The scipy DOP853 "
           "comparison integrates the ciw equations, so only its integrator is independent"],
        recommended_next_task=("Deferred research question: an integrator-independent 34-digit reference: integrate "
                               "the sympy-derived equations with mpmath's own Taylor-series solver (mpmath.odefun) "
                               "instead of the ciw-authored Gragg-Bulirsch-Stoer integrator and check that it agrees "
                               "with the retained references within their empirical error estimates, so that both "
                               "the equations and the integrator of the reference are independent of ciw"))


# ---------------------------------------------------------------------------
# T003/T004: integrator orders and unit-speed drift (one shared study)
# ---------------------------------------------------------------------------
ORDER_STEPS = {"euler": (64, 128, 256, 512), "midpoint": (32, 64, 128, 256), "rk4": (24, 48, 96, 192)}
ORDER_TOLERANCE = {"euler": 0.1, "midpoint": 0.1, "rk4": 0.25}
ADAPTIVE_RTOL = (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
DRIFT_TOLERANCE = {"euler": 0.1, "midpoint": 0.15, "rk4": 0.3}


def convergence_study(ctx) -> dict:
    """Endpoint error and speed drift of every fixed-step method, step and standard path (geodesic state only)."""
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
            table[key] = row
        return {"table": table, "sphere_drift_curves": curves}
    return ctx.memo("gj-convergence", compute)


def _slopes(entries, quantity):
    return gj.slope([e["h"] for e in entries], [e[quantity] for e in entries])


# Order 5 (local extrapolation, error ~ tol) and order 4 (advancing with y4, error ~ tol^0.8) are the two
# hypotheses the adaptive checks must separate; the thresholds sit half-way between them.
ADAPTIVE_ORDER_THRESHOLD = 4.5
ADAPTIVE_EXPONENT_THRESHOLD = 0.9


def dormand_prince_y4(f, y0, length, rtol=1e-8, atol=1e-10):
    """Deliberately faulty comparison: DP5(4) that advances with its fourth-order solution.

    Same tableau, error norm and controller as ``integrators.integrate_adaptive``;
    only the accepted state differs (y4 instead of y5, so no local
    extrapolation and no first-same-as-last reuse).
    """
    tableau, b5, b4 = integrators._A, integrators._B5, integrators._B4
    y = np.array(y0, dtype=float)
    s, h = 0.0, min(length, 0.01 * max(length, 1e-12))
    nodes, states = [0.0], [y.copy()]
    accepted = rejected = 0
    k1 = f(y)
    evaluations = 1
    while s < length:
        h = min(h, length - s)
        k = [k1]
        for stage in range(1, 7):
            k.append(f(y + h * sum(a * kj for a, kj in zip(tableau[stage], k))))
        evaluations += 6
        y5 = y + h * sum(b * kj for b, kj in zip(b5, k))
        y4 = y + h * sum(b * kj for b, kj in zip(b4, k))
        error = math.sqrt(float(np.mean(((y5 - y4) / (atol + rtol * np.maximum(np.abs(y), np.abs(y5)))) ** 2)))
        if error <= 1.0:
            s = length if length - (s + h) < 1e-14 * max(1.0, length) else s + h
            y = y4
            k1 = f(y)
            evaluations += 1
            nodes.append(s)
            states.append(y.copy())
            accepted += 1
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** -0.2))
        else:
            rejected += 1
            factor = max(0.2, 0.9 * error ** -0.2)
        h *= factor
    stats = {"accepted_steps": accepted, "rejected_steps": rejected, "function_evaluations": evaluations}
    return np.array(nodes), np.array(states), stats


def adaptive_summary(ctx, name: str, integrate) -> dict:
    """Per-chart effective order (error vs evaluations) and tolerance exponent (error vs rtol), and their medians."""
    def compute():
        charts = {}
        for key in GAMMA_CHARTS:
            spec = gj.path(key)
            surf = gj.surface(spec.surface)
            target = np.asarray(reference(ctx, key)["position"])
            rows = []
            for rtol in ADAPTIVE_RTOL:
                _, states, stats = integrate(surf.geodesic_rhs, gj.start_state(key)[:4], spec.length, rtol=rtol,
                                             atol=rtol)
                rows.append({"rtol": rtol, **stats,
                             "error": float(np.linalg.norm(gj.position(spec.surface, states[-1, :2]) - target))})
            charts[key] = {"rows": rows,
                           "effective_order": -gj.slope([r["function_evaluations"] for r in rows],
                                                        [r["error"] for r in rows]),
                           "tolerance_exponent": gj.slope([r["rtol"] for r in rows], [r["error"] for r in rows])}
        orders = [c["effective_order"] for c in charts.values()]
        exponents = [c["tolerance_exponent"] for c in charts.values()]
        return {"charts": charts, "median_effective_order": float(np.median(orders)),
                "median_tolerance_exponent": float(np.median(exponents)),
                "order_mad": float(np.median(np.abs(np.array(orders) - np.median(orders)))),
                "exponent_mad": float(np.median(np.abs(np.array(exponents) - np.median(exponents))))}
    return ctx.memo(("gj-adaptive-summary", name), compute)


@task("T003", changed_files=CHANGED, regression_tests=(_test("test_t003_integrator_orders"),
                                                       _test("test_adaptive_checks_reject_fourth_order_variant"))
      + SECTION_TESTS)
def integrator_orders(ctx):
    study = convergence_study(ctx)["table"]
    orders = {method: {k: _slopes(study[k][method], "error") for k in GAMMA_CHARTS} for method in ORDER_STEPS}
    pairwise = {method: {k: _pairwise(study[k][method], "h", "error") for k in GAMMA_CHARTS}
                for method in ORDER_STEPS}
    adaptive = adaptive_summary(ctx, "dp54", integrators.integrate_adaptive)
    variant = adaptive_summary(ctx, "dp54-y4", dormand_prince_y4)
    effective = {k: c["effective_order"] for k, c in adaptive["charts"].items()}
    flat = {k: max(e["error"] for m in ORDER_STEPS for e in study[k][m]) for k in FLAT_CARTESIAN}
    ctx.artifact_json("orders.json", _plain({
        "steps": ORDER_STEPS, "adaptive_rtol": ADAPTIVE_RTOL, "fitted_orders": orders, "pairwise_orders": pairwise,
        "adaptive": adaptive,
        "adaptive_y4_variant": variant, "flat_cartesian_max_error": flat, "table": study}))
    for key in GAMMA_CHARTS:
        ctx.artifact_text(f"orders-{key}.svg", svg.line_plot(
            [(m, [e["h"] for e in study[key][m]], [e["error"] for e in study[key][m]]) for m in ORDER_STEPS],
            title=f"Global endpoint error on {key}", xlabel="step h", ylabel="endpoint error", logx=True, logy=True))
    ctx.artifact_text("adaptive.svg", svg.line_plot(
        [(k, [r["function_evaluations"] for r in c["rows"]], [r["error"] for r in c["rows"]])
         for k, c in adaptive["charts"].items()], title="Dormand-Prince error against function evaluations",
        xlabel="function evaluations", ylabel="endpoint error", logx=True, logy=True,
        rounding=64 * float(np.finfo(float).eps)), rounding_level=True)

    generator = {"name": "declared standard geodesics", "paths": list(gj.STANDARD)}
    adaptive_generator = dict(generator, rtol=list(ADAPTIVE_RTOL), charts=list(GAMMA_CHARTS))
    findings = []
    names = {"euler": "Explicit Euler", "midpoint": "Explicit midpoint", "rk4": "Classical RK4"}
    for method, p in NOMINAL_ORDER.items():
        findings.append(finding(
            f"{names[method]} global endpoint error converges at order {p} on every chart with nonzero Christoffel "
            "symbols", "numerical",
            orders[method], {"generator": generator, "checks": [
                gj.check("analytic", f"nominal order {p} on {k} (steps {ORDER_STEPS[method]})",
                         orders[method][k] - p, ORDER_TOLERANCE[method]) for k in GAMMA_CHARTS]},
            uncertainty=_unc("fit", _fit_spread(pairwise[method], orders[method]),
                             "largest distance of a pairwise (halving) slope from the least-squares slope"),
            tolerance=TOL_RATE))
    # Per-chart effective orders move by up to about 0.3 when one step is accepted or rejected differently,
    # so the claims are about the medians over the seven charts; per-chart values stay in orders.json.
    median_order, median_exponent = adaptive["median_effective_order"], adaptive["median_tolerance_exponent"]
    findings.append(finding(
        "Adaptive Dormand-Prince error falls with function evaluations at a median effective order near 5",
        "numerical", median_order, {"generator": adaptive_generator, "checks": [
            gj.check("analytic", "median effective order over seven charts, at least half-way from 4 to 5 (rtol "
                     "1e-8..1e-12)", median_order, ADAPTIVE_ORDER_THRESHOLD, "signed_ge"),
            gj.check("analytic", "median effective order over seven charts, at most 6", median_order, 6.0,
                     "signed_le")]},
        uncertainty=_unc("fit", adaptive["order_mad"], "median absolute deviation of the seven per-chart effective "
                         "orders; the controller's start-step ramp biases loose tolerances upward"),
        tolerance={"abs": 0.35, "rel": 0.0}))
    findings.append(finding(
        "Adaptive Dormand-Prince endpoint error is proportional to the requested tolerance (median over charts)",
        "numerical", median_exponent, {"generator": adaptive_generator, "checks": [
            gj.check("analytic", "median exponent of error ~ rtol^q, at least half-way from 0.8 to 1",
                     median_exponent, ADAPTIVE_EXPONENT_THRESHOLD, "signed_ge"),
            gj.check("analytic", "median exponent of error ~ rtol^q, at most 1.1", median_exponent, 1.1,
                     "signed_le")]},
        uncertainty=_unc("fit", adaptive["exponent_mad"], "median absolute deviation of the seven per-chart "
                         "tolerance exponents"),
        tolerance={"abs": 0.1, "rel": 0.0}))
    findings.append(finding(
        "The adaptive-order checks reject a Dormand-Prince variant that advances with its fourth-order solution",
        "numerical", {"median_effective_order": variant["median_effective_order"],
                      "median_tolerance_exponent": variant["median_tolerance_exponent"]},
        {"generator": dict(adaptive_generator, variant="accepted state y4, no local extrapolation"), "checks": [
            gj.check("analytic", "variant median effective order below the acceptance threshold",
                     variant["median_effective_order"], ADAPTIVE_ORDER_THRESHOLD, "signed_le"),
            gj.check("analytic", "variant median tolerance exponent below the acceptance threshold",
                     variant["median_tolerance_exponent"], ADAPTIVE_EXPONENT_THRESHOLD, "signed_le")]},
        uncertainty=_unc("fit", max(variant["order_mad"], variant["exponent_mad"]),
                         "median absolute deviation over the seven charts"),
        tolerance={"abs": 0.35, "rel": 0.0},
        counterexample={"statement": "An effective order near 5 is observed whatever solution a DP5(4) pair advances",
                        "witness": {"variant": "advance with y4", "median_effective_order":
                                    variant["median_effective_order"], "median_tolerance_exponent":
                                    variant["median_tolerance_exponent"]}}))
    findings.append(finding(
        "No convergence order is observable on flat Cartesian charts: every method is exact to rounding there",
        "numerical", flat, {"generator": generator, "checks": [
            gj.check("analytic", f"straight-line geodesic on {k} (Gamma = 0)", flat[k], 1e-12) for k in FLAT_CARTESIAN]},
        uncertainty=_unc("roundoff", max(flat.values()), "accumulated binary64 rounding over at most 512 steps"),
        tolerance=TOL_SMALL,
        counterexample={"statement": "Every integrator exhibits its nominal convergence order on every surface",
                        "witness": {"charts": list(FLAT_CARTESIAN), "max_endpoint_error": flat,
                                    "reason": "Gamma vanishes, so Euler, midpoint and RK4 reproduce u(s) = u0 + s v0"}}))
    return _outcome(
        "completed", findings,
        hypothesis=("On charts with nonzero Christoffel symbols the global endpoint error of Euler, midpoint and RK4 "
                    "scales like h^1, h^2 and h^4, and the Dormand-Prince error scales like (evaluations)^-5 and "
                    "like the requested tolerance, which distinguishes it from a pair advancing with y4 "
                    "((evaluations)^-4, tol^0.8)."),
        mathematical_model=("Global error e(h) = C h^p + O(h^(p+1)) for a p-th order one-step method on a smooth "
                            "ODE; for DP5(4) with per-step error control and local extrapolation, h ~ tol^(1/5) and "
                            "e ~ h^5 ~ tol; advancing with y4 gives e ~ h^4 ~ tol^(4/5)."),
        input_data=[f"Standard paths on {', '.join(gj.STANDARD)}",
                    f"Steps {ORDER_STEPS}; adaptive rtol = atol in {list(ADAPTIVE_RTOL)}",
                    "References from T002 (closed form or mpmath)"],
        observation_model="Euclidean distance of the embedded end point (chart distance on the hyperbolic plane).",
        expected_invariant=("Fitted log-log slopes within declared tolerances of the nominal orders; adaptive medians "
                            f"above the half-way thresholds {ADAPTIVE_ORDER_THRESHOLD} (order) and "
                            f"{ADAPTIVE_EXPONENT_THRESHOLD} (tolerance exponent), which the y4 variant fails."),
        experiment=("Integrate the geodesic state (u, v) with each fixed-step method at four halvings and the "
                    f"adaptive method and its y4 variant at {len(ADAPTIVE_RTOL)} tolerances; fit log-log slopes "
                    "against the reference end point."),
        numerical_result=("Fitted orders: " + "; ".join(
            f"{m} " + ", ".join(f"{k} {_fmt(v, 4)}" for k, v in orders[m].items()) for m in ORDER_STEPS)
            + "; adaptive effective orders " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in effective.items())
            + f" (median {_fmt(median_order, 3)}), tolerance-exponent median {_fmt(median_exponent, 3)}; y4 variant "
            f"medians {_fmt(variant['median_effective_order'], 3)} and "
            f"{_fmt(variant['median_tolerance_exponent'], 3)}."),
        uncertainty=("Fixed-step slopes are least-squares fits over four points; the pre-asymptotic bias is the "
                     "retained spread of pairwise slopes. Adaptive slopes depend on accept/reject decisions, so "
                     "only their medians over seven charts are claimed."),
        failure_modes_checked=["flat Cartesian charts where every method is exact (recorded as a counterexample)",
                               "RK4 errors kept above the rounding floor (smallest near 1e-12)",
                               "reference accuracy exceeds the smallest measured error by several digits",
                               "a DP5(4) pair advancing with y4 (rejected by the median checks)",
                               "single accept/reject flips (per-chart orders are not claimed)"],
        unresolved_assumptions=["The adaptive effective order depends on the step-size controller and its start "
                                "step; the retained numbers are for integrators.integrate_adaptive only",
                                "The half-way thresholds separate order 5 from order 4; they do not certify the "
                                "exact order of the pair",
                                "plane-polar and cylinder-polar carry the same flat metric (unit cylinder); they "
                                "count as two charts because their declared paths differ (start, heading, length), "
                                "so each contributes its own data point"],
        recommended_next_task=("Deferred research question: separate the adaptive effective order from the step-size "
                               "controller by re-measuring it with a second controller and start step on the same "
                               "references (for example scipy's RK45 and DOP853 at matched tolerances), since the "
                               "retained effective orders hold for integrators.integrate_adaptive only and the "
                               "half-way thresholds do not certify the exact order of the pair"))


# T004 -----------------------------------------------------------------------
NORMALIZING_CALLS = frozenset({"norm", "normalize", "normalized", "speed_squared", "unit_tangent", "orthonormal_frame",
                               "normal", "inner", "hypot", "sqrt", "rsqrt"})
# Operations that produce a magnitude: a name bound to one taints later divisions by that name.
MAGNITUDE_CALLS = frozenset({"sqrt", "norm", "hypot"})
# Justified uses, by function: the adaptive controller's RMS error norm scales the step size, never the state.
ALLOWED_CALLS = frozenset({("integrate_adaptive", "sqrt")})
RENORM_STEPS = (64, 128, 256, 512)


def state_update_code():
    """Every function through which an integrated state passes after its initial data is set."""
    return (integrators.step_euler, integrators.step_midpoint, integrators.step_rk4, integrators.integrate_fixed,
            integrators.richardson_rk4, integrators.integrate_adaptive, jacobi.rhs, jacobi.transfer,
            surfaces.Surface.geodesic_rhs, surfaces.Surface.christoffel)


def _called(node) -> str | None:
    return node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)


def _constant(node):
    """Value of a numeric constant expression (+, -, *, / of literals), else None."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _constant(node.operand)
        return None if value is None else (-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _constant(node.left), _constant(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right if right else None
    return None


def _half_power(node) -> bool:
    """True for an exponent that is a constant expression equal to +0.5 or -0.5."""
    value = _constant(node)
    return value is not None and abs(abs(value) - 0.5) < 1e-12


def _is_magnitude(node) -> bool:
    return any((isinstance(n, ast.Call) and _called(n) in MAGNITUDE_CALLS)
               or (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) and _half_power(n.right))
               for n in ast.walk(node))


def normalization_calls(functions=None) -> list:
    """Operations in the given source that could renormalize a state.

    Flags norm-like and square-root calls (except the allow-listed error
    norm), any power of +-0.5, divisions by sqrt/norm/hypot/abs expressions,
    and divisions by names previously bound to a sqrt, norm, hypot or
    half-power expression.
    """
    found = []
    for function in functions or state_update_code():
        where = function.__qualname__
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        tainted = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None \
                    and _is_magnitude(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                tainted |= {n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called(node) in NORMALIZING_CALLS \
                    and (function.__name__, _called(node)) not in ALLOWED_CALLS:
                found.append(f"{where}: call {_called(node)}")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and _half_power(node.right):
                found.append(f"{where}: power of 0.5 or -0.5")
            divisor = None
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
                divisor = node.right
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
                divisor = node.value
            if divisor is None:
                continue
            if isinstance(divisor, ast.Call) and _called(divisor) in MAGNITUDE_CALLS | {"abs"}:
                found.append(f"{where}: division by {_called(divisor)}")
            elif isinstance(divisor, ast.Name) and divisor.id in tainted:
                found.append(f"{where}: division by the magnitude {divisor.id}")
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


def speed_probe(surf, y0, length, method: str) -> np.ndarray:
    """g(v, v) along a path started at the given (possibly non-unit) speed, for a fixed or adaptive method."""
    if method == "adaptive":
        _, states, _ = integrators.integrate_adaptive(surf.geodesic_rhs, y0, length, rtol=1e-10, atol=1e-12)
    else:
        _, states = integrators.integrate_fixed(surf.geodesic_rhs, y0, length, 128, method)
    return np.array([surf.speed_squared(y[:2], y[2:4]) for y in states])


@task("T004", changed_files=CHANGED,
      regression_tests=(_test("test_t004_speed_drift_and_no_renormalization"),
                        _test("test_integrator_code_path_has_no_normalization")) + SECTION_TESTS)
def unit_speed_drift(ctx):
    study = convergence_study(ctx)
    table = study["table"]
    drift_orders = {m: {k: _slopes(table[k][m], "max_speed_drift") for k in GAMMA_CHARTS} for m in ORDER_STEPS}
    drift_pairwise = {f"{m}/{k}": _pairwise(table[k][m], "h", "max_speed_drift")
                      for m in ORDER_STEPS for k in GAMMA_CHARTS}
    drift_fit = _fit_spread(drift_pairwise, {f"{m}/{k}": v for m, r in drift_orders.items() for k, v in r.items()})
    flat_drift = {k: max(e["max_speed_drift"] for m in ORDER_STEPS for e in table[k][m]) for k in FLAT_CARTESIAN}

    speed, speed2 = 1.3, 1.69
    nonunit = {}
    for key in ("sphere", "torus", "hyperbolic-plane"):
        spec = gj.path(key)
        surf = gj.surface(spec.surface)
        y0 = gj.start_state(key)[:4].copy()
        y0[2:4] *= speed
        row = {}
        for method in ("rk4", "adaptive", "midpoint", "euler"):
            g = speed_probe(surf, y0, spec.length, method)
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
    renorm = []
    for steps in RENORM_STEPS:
        faulty = renormalized_euler(sphere, gj.start_state("sphere")[:4], spec.length, steps)
        renorm.append({"steps": steps, "h": spec.length / steps,
                       "max_speed_drift": float(np.max(np.abs(_speed_drift(sphere, faulty)))),
                       "error": float(np.linalg.norm(sphere.embedding(faulty[-1, :2]) - target))})
    renorm_order = _slopes(renorm, "error")
    at128 = next(e for e in renorm if e["steps"] == 128)
    plain = next(e for e in table["sphere"]["euler"] if e["steps"] == 128)
    counter = {"steps": 128, "renormalized_max_speed_drift": max(e["max_speed_drift"] for e in renorm),
               "renormalized_endpoint_error": at128["error"], "renormalized_error_order": renorm_order,
               "plain_endpoint_error": plain["error"], "plain_max_speed_drift": plain["max_speed_drift"]}

    ctx.artifact_json("speed-drift.json", _plain({
        "drift_orders": drift_orders, "pairwise_drift_orders": drift_pairwise, "flat_cartesian_max_drift": flat_drift,
        "nonunit_start": nonunit, "exponential_growth_relative_error": growth, "normalization_calls": calls,
        "scanned_functions": [f"{f.__module__}.{f.__qualname__}" for f in state_update_code()],
        "allowed_calls": sorted(map(list, ALLOWED_CALLS)), "renormalized_euler": renorm,
        "renormalized_euler_counterexample": counter,
        "table": {k: {m: [{"h": e["h"], "max_speed_drift": e["max_speed_drift"], "final_speed_drift":
                           e["final_speed_drift"]} for e in table[k][m]] for m in ORDER_STEPS} for k in gj.STANDARD}}))
    for method in ORDER_STEPS:
        ctx.artifact_text(f"drift-{method}.svg", svg.line_plot(
            [(k, [e["h"] for e in table[k][method]], [e["max_speed_drift"] for e in table[k][method]])
             for k in GAMMA_CHARTS], title=f"Unit-speed drift max|g(v,v) - 1|, {method}", xlabel="step h",
            ylabel="max |g(v,v) - 1|", logx=True, logy=True))
    curves = study["sphere_drift_curves"]
    ctx.artifact_text("drift-along-sphere.svg", svg.line_plot(
        [(f"{m} N={ORDER_STEPS[m][1]}", s, np.maximum(d, 1e-17)) for m, (s, d) in curves.items()],
        title="Speed drift along the sphere path (no renormalization)", xlabel="arclength s",
        ylabel="|g(v,v) - 1|", logy=True, markers=False))

    generator = {"name": "declared standard geodesics", "paths": list(gj.STANDARD)}
    findings = [finding(
        "Unit-speed drift max|g(v,v) - 1| scales like h^p for Euler, midpoint and RK4 on every chart with nonzero "
        "Christoffel symbols",
        "numerical", drift_orders, {"generator": generator, "checks": [
            gj.check("analytic", f"drift order {NOMINAL_ORDER[m]} for {m} on {k}",
                     drift_orders[m][k] - NOMINAL_ORDER[m], DRIFT_TOLERANCE[m])
            for m in ORDER_STEPS for k in GAMMA_CHARTS]},
        uncertainty=_unc("fit", drift_fit, "largest distance of a pairwise (halving) slope from the least-squares "
                         "slope"), tolerance=TOL_RATE)]
    accurate = ("rk4", "adaptive")
    nonunit_drift = max(v[m]["max_abs_g_minus_1.69"] for v in nonunit.values() for m in accurate)
    findings.append(finding(
        "A non-unit initial speed stays non-unit: g(v,v) remains 1.69 to integrator accuracy", "numerical",
        {k: {m: v[m]["max_abs_g_minus_1.69"] for m in accurate} for k, v in nonunit.items()},
        {"generator": {"name": "declared standard geodesics with speed 1.3", "paths": list(nonunit),
                       "methods": ["rk4 N=128", "adaptive rtol 1e-10", "midpoint N=128", "euler N=128"]}, "checks": [
            gj.check("invariant", f"{m} |g - 1.69| on {k}", v[m]["max_abs_g_minus_1.69"], 1e-7)
            for k, v in nonunit.items() for m in accurate] + [
            gj.check("invariant", f"distance of g from 1 on {k}, all four methods (a renormalizer would drive it "
                     "to 0)", min(v[m]["min_abs_g_minus_1"] for m in v), 0.6, "ge") for k, v in nonunit.items()]},
        uncertainty=_unc("truncation_bound", nonunit_drift, "RK4 (N = 128) and adaptive (rtol 1e-10) truncation of "
                         "the speed invariant"), tolerance=TOL_SMALL))
    findings.append(finding(
        "The integrator and geodesic/Jacobi right-hand-side code paths contain no state normalization",
        "computational_pipeline", {"normalization_calls": len(calls), "exponential_growth": growth},
        {"checks": [gj.check("invariant", "normalizing calls, half powers or divisions by a computed magnitude in "
                             f"{len(state_update_code())} scanned functions", len(calls), 0.0),
                    gj.check("analytic", "RK4 N=64 growth |y(2)|/|y0| against e^2 for y' = y", growth["rk4_64"], 1e-6),
                    gj.check("analytic", "adaptive rtol 1e-10 growth against e^2 for y' = y",
                             growth["adaptive_1e-10"], 1e-8)]},
        uncertainty=_unc("truncation_bound", max(growth.values()), "global error of the y' = y probes; the "
                         "normalization count is exact"),
        tolerance={"abs": 1e-8, "rel": 0.0}))
    findings.append(finding(
        "Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding with a "
        "first-order endpoint error", "numerical", counter,
        {"generator": {"name": "renormalized Euler on the standard sphere path", "steps": list(RENORM_STEPS)},
         "checks": [
            gj.check("invariant", "renormalized Euler speed drift, all steps", counter["renormalized_max_speed_drift"],
                     1e-13),
            gj.check("analytic", "renormalized Euler endpoint error against the great circle at N = 128",
                     counter["renormalized_endpoint_error"], 1e-3, "ge"),
            gj.check("analytic", f"renormalized Euler endpoint error order 1 (N = {RENORM_STEPS[0]}.."
                     f"{RENORM_STEPS[-1]})", renorm_order - 1, 0.1)]},
        uncertainty=_unc("fit", max(abs(v - renorm_order) for v in _pairwise(renorm, "h", "error")),
                         "largest distance of a pairwise slope from the fitted order"),
        tolerance={"abs": 1e-9, "rel": 1e-6},
        counterexample={"statement": "A computed geodesic whose speed stays exactly 1 is accurate",
                        "witness": {"surface": "sphere", "method": "Euler with per-step speed renormalization",
                                    "steps": 128, "endpoint_error": counter["renormalized_endpoint_error"]}}))
    findings.append(finding(
        "Speed drift is at rounding on flat Cartesian charts, where every method is exact", "numerical", flat_drift,
        {"generator": generator, "checks": [gj.check("analytic", f"constant metric on {k}", flat_drift[k], 1e-13)
                                            for k in FLAT_CARTESIAN]},
        uncertainty=_unc("roundoff", max(flat_drift.values()), "accumulated binary64 rounding"), tolerance=TOL_SMALL))
    return _outcome(
        "completed", findings,
        hypothesis=("Without renormalization the speed invariant g(v, v) drifts at the global order of each method, "
                    "and a non-unit initial speed is carried unchanged, showing that nothing resets it."),
        mathematical_model=("g(v, v) is a first integral of the geodesic flow; for a p-th order method its global "
                            "defect is O(h^p). A renormalizing step would force g(v, v) = 1 whatever the initial "
                            "speed."),
        input_data=[f"The T003 fixed-step integrations (steps {ORDER_STEPS})",
                    "Speed-1.3 starts on the sphere, torus and hyperbolic plane (RK4, midpoint and Euler with 128 "
                    "steps, adaptive rtol 1e-10); y' = y growth probes",
                    f"Renormalized Euler on the sphere path with {list(RENORM_STEPS)} steps"],
        observation_model="max over nodes of |g(v, v) - 1| (or |g - 1.69|) evaluated with the exact metric.",
        expected_invariant=("Drift slopes near p; g stays at 1.69; zero normalizing operations in the state-update "
                            "path."),
        experiment=("Measure drift for every method and step; restart with speed 1.3 under all four methods; scan "
                    "the state-update source for normalizing calls, half powers and divisions by magnitudes; "
                    "integrate y' = y; compare a deliberately renormalized Euler at four step counts."),
        numerical_result=("Drift orders: " + "; ".join(
            f"{m} " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in drift_orders[m].items()) for m in ORDER_STEPS)
            + f"; speed-1.3 starts keep |g - 1.69| <= {_fmt(nonunit_drift)} (RK4, adaptive); renormalized Euler "
            f"endpoint error {_fmt(counter['renormalized_endpoint_error'])} at N = 128 (order "
            f"{_fmt(renorm_order, 3)}) with speed drift {_fmt(counter['renormalized_max_speed_drift'])}."),
        uncertainty="Slopes are least-squares fits over four halvings; RK4 sphere drift is still slightly pre-asymptotic.",
        failure_modes_checked=["hidden renormalization (static scan, speed-1.3 start under every method, y' = y "
                               "growth)",
                               "normalization through an intermediate name or a -0.5 power (scanner probes)",
                               "speed drift mistaken for accuracy (renormalized Euler counterexample)",
                               "flat charts where drift is identically at rounding"],
        unresolved_assumptions=["The static scan covers the listed functions only; code outside them (for example "
                                "initial-state construction, which normalizes the start by design, and the metric "
                                "evaluations called from the right-hand side) is not scanned"],
        recommended_next_task=("Deferred research question: extend the no-renormalization scan from the listed "
                               "functions to every function reachable from the integrator step (the metric "
                               "evaluations called from the right-hand side included), for example by walking the "
                               "call graph with the ast module, with the start-state normalization declared as the "
                               "one allowed exception"))


# ---------------------------------------------------------------------------
# T005: separation law; optional CSG provider
# ---------------------------------------------------------------------------
CONSTANT_PATHS = ("sphere-great-circle", "plane", "cylinder", "hyperbolic-long", "torus-outer-equator",
                  "torus-inner-equator")
# Figure legends stay within the 20 characters the SVG legend column shows.
CONSTANT_SHORT = {"sphere-great-circle": "sphere", "plane": "plane", "cylinder": "cylinder",
                  "hyperbolic-long": "hyperbolic", "torus-outer-equator": "outer eq",
                  "torus-inner-equator": "inner eq"}
# K = +1, 0 and -1. The K = 0 case uses straight lines seen through the polar chart of the plane, where the
# perturbed starts are built with nonzero Christoffel symbols (on the Cartesian charts j'' = 0 is exact for RK4).
SEPARATION_PATHS = ("sphere-great-circle", "plane-polar", "hyperbolic-long")
SEPARATION_EPS = (0.04, 0.02, 0.01)
SEPARATION_NODES = 141
CSG_REFUSALS = frozenset(gj.PIN_REFUSALS + gj.EXECUTION_REFUSALS)
# After pin verification the refusal code follows from where it was raised: run_csg_jacobi refuses only
# with CSG_EXECUTION_FAILED, and the re-verification after it only with CSG_CHANGED_DURING_EXECUTION.
STAGE_REFUSALS = {"execution": "CSG_EXECUTION_FAILED", "post-execution": "CSG_CHANGED_DURING_EXECUTION"}


def _model_error(tr, k):
    """Relative errors of both columns and their derivatives against cn_K and sn_K."""
    a, ap, b, bp = jacobi.constant_curvature(k, tr.s)
    y = tr.states
    return {"heading": _discrepancy(y[:, 6], b), "heading_rate": _discrepancy(y[:, 7], bp),
            "lateral": _discrepancy(y[:, 4], a), "lateral_rate": _discrepancy(y[:, 5], ap)}


def geodesic_distance(key: str, a, b) -> np.ndarray:
    """Exact intrinsic distance between matched points of two closed-form geodesics (sphere, plane, half-plane).

    Sphere: 2 R asin(chord / 2R) of embedded points; polar chart of the plane:
    |a - b| of the Cartesian points; half-plane with g = I/(k^2 y^2):
    (2/k) asinh(|a - b| / (2 sqrt(y_a y_b))) of chart points.
    """
    surf = gj.surface(gj.path(key).surface)
    if isinstance(surf, surfaces.Sphere):
        chord = np.linalg.norm(a - b, axis=1)
        return 2 * surf.radius * np.arcsin(np.minimum(1.0, chord / (2 * surf.radius)))
    if isinstance(surf, surfaces.Reparametrized) and isinstance(surf.base, surfaces.Plane):
        return np.linalg.norm(a - b, axis=1)
    if isinstance(surf, surfaces.HyperbolicPlane):
        return 2 / surf.k * np.arcsinh(np.linalg.norm(a - b, axis=1) / (2 * np.sqrt(a[:, 1] * b[:, 1])))
    raise ValueError(f"No closed-form distance on {key}")


def separation_study(key: str) -> dict:
    """Distance of perturbed closed-form geodesics per unit perturbation against |cn_K| and |sn_K|.

    Uses no integrated Jacobi field: the base and perturbed geodesics are the
    surface's closed forms (the lateral start is the exact normal-geodesic
    offset of ``jacobi.perturbed_start``), and K comes from the parameters.
    """
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    s = np.linspace(0.0, spec.length, SEPARATION_NODES)
    u0 = np.asarray(spec.u0, dtype=float)
    t0 = surf.unit_tangent(u0, spec.heading)

    def curve(u, t):
        if isinstance(surf, surfaces.Sphere):
            return surf.exact_embedded_geodesic(u, t, s)
        if isinstance(surf, surfaces.Reparametrized):
            # A straight line of the plane, from the polar start mapped to Cartesian position and velocity.
            return surf.chart.forward(u) + np.outer(s, surf.chart.jacobian(u) @ t)
        return surf.exact_geodesic(u, t, s)

    base = curve(u0, t0)
    k = _constant_curvature_of(key)
    cn, _, sn, _ = jacobi.constant_curvature(k, s)
    out = {"curvature": k}
    for column, model in (("lateral", np.abs(cn)), ("heading", np.abs(sn))):
        errors = []
        for eps in SEPARATION_EPS:
            args = {"lateral": eps} if column == "lateral" else {"heading_change": eps}
            y = jacobi.perturbed_start(surf, spec.u0, spec.heading, **args)
            ratio = geodesic_distance(key, curve(y[:2], y[2:]), base) / eps
            errors.append(_discrepancy(ratio, model))
        # On K = 0 the lateral law has no remainder (parallel lines stay eps apart), so no order is defined.
        exact = k == 0 and column == "lateral"
        out[column] = {"errors": errors, "order": None if exact else gj.slope(SEPARATION_EPS, errors),
                       "exact_law": exact}
    return out


def _csg(ctx):
    """Run the pinned CSG provider once per context on the constant-curvature grids (memoized).

    The expected pin-stage refusal is predicted from direct git queries before
    verification, so a refusal finding compares a prediction with the outcome.
    Later refusals are expected from the stage that raised them (see
    ``STAGE_REFUSALS``), never copied from the observed code.
    """
    def compute():
        checkout = ctx.providers["csg"]
        predicted = gj.predict_csg_refusal(checkout)
        stage = "pin"
        try:
            identity = gj.verify_csg_checkout(checkout)
            stage = "execution"
            cases = [{"arclength": _fine_transfer(ctx, key).s.tolist(), "gaussian_curvature": _constant_curvature_of(key)}
                     for key in CONSTANT_PATHS]
            data = gj.run_csg_jacobi(checkout, cases)
            stage = "post-execution"
            try:
                unchanged = gj.verify_csg_checkout(checkout) == identity
            except gj.ProviderRefusal:
                unchanged = False
            if not unchanged:
                raise gj.ProviderRefusal("CSG_CHANGED_DURING_EXECUTION", "Provider checkout changed during execution")
        except gj.ProviderRefusal as exc:
            # The checkout path is machine-specific; retained messages name it generically.
            message = str(exc).replace(str(checkout), "<checkout>")
            return {"refusal": exc.code, "stage": stage, "predicted": predicted, "message": message}
        identity = dict(identity, implementation=gj.CSG_IMPLEMENTATION, python=data["python"], numpy=data["numpy"],
                        subprocess=True)
        return {"identity": identity, "data": data, "paths": list(CONSTANT_PATHS), "predicted": predicted}
    return ctx.memo("gj-csg", compute)


def _provider_identity(csg):
    from .runner import builtin_identity

    return {"provider": csg["identity"], "ciw": builtin_identity(CHANGED)}


def _refusal_record(ctx, csg) -> tuple:
    """Refusal finding, fixed report sentence and retained detail for a refused CSG binding."""
    ctx.artifact_json("provider-refusal.json", {key: csg[key] for key in ("refusal", "stage", "predicted", "message")})
    code = csg["refusal"]
    if csg["stage"] == "pin":
        record = finding(
            "A bound CSG checkout whose git state differs from the pin is refused with the code that state predicts",
            "provenance", {"predicted": csg["predicted"], "observed": code},
            {"checks": [{"reference_kind": "refusal", "reference": "code predicted from direct git queries "
                         "(rev-parse, status, ls-files) before verify_csg_checkout ran",
                         "expected_refusal": csg["predicted"], "observed_refusal": code,
                         "passed": code == csg["predicted"]}]},
            uncertainty=_unc("exact", 0.0, "refusal codes are exact strings"))
    else:
        expected = STAGE_REFUSALS[csg["stage"]]
        record = finding(
            "A pinned CSG provider whose execution fails or whose checkout changes during execution is refused "
            "rather than compared", "provenance",
            {"predicted_pin_stage": csg["predicted"], "stage": csg["stage"], "expected": expected, "observed": code},
            {"checks": [
                {"reference_kind": "refusal", "reference": "pin stage predicted clean by direct git queries and "
                 "verified clean", "expected_refusal": csg["predicted"], "observed_refusal": "none",
                 "passed": csg["predicted"] == "none"},
                {"reference_kind": "refusal", "reference": "code implied by the stage that refused: a failed provider "
                 "subprocess or malformed output (execution) or a failed re-verification (post-execution)",
                 "expected_refusal": expected, "observed_refusal": code, "passed": code == expected}]},
            uncertainty=_unc("exact", 0.0, "refusal codes are exact strings"))
    return record, f"CSG provider refused ({code}); the comparison did not run (detail in provider-refusal.json)"


@task("T005", changed_files=CHANGED,
      regression_tests=(_test("test_t005_separation_law"), _test("test_t005_csg_provider_agreement"),
                        _test("test_csg_checkout_refusals"), _test("test_csg_output_is_refused_unless_complete"),
                        _test("test_csg_execution_refusals_are_expected_from_their_stage"))
      + SECTION_TESTS)
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
               "j_lat_end": float(fine.states[-1, 4]),
               "halving_estimate": abs(max(err_coarse.values()) - max(err_fine.values())) / 15}
        if key.startswith("torus"):
            row["theta_drift"] = float(np.max(np.abs(fine.points[:, 1] - spec.u0[1])))
        if k != 0:
            row["observed_order"] = math.log2(max(err_coarse.values()) / max(err_fine.values()))
        rows[key] = row
    separation = {key: separation_study(key) for key in SEPARATION_PATHS}
    csg = _csg(ctx) if ctx.available("provider:csg") else None
    provider_rows = None
    if csg and "data" in csg:
        provider_rows = {}
        for key, trace, maps in zip(csg["paths"], csg["data"]["traces"], csg["data"]["maps"], strict=True):
            fine = _fine_transfer(ctx, key)
            ciw_phi = np.stack([np.array([[y[4], y[6]], [y[5], y[7]]]) for y in fine.states])
            csg_numeric = np.asarray(maps["numeric"]["matrices"])
            csg_exact = np.asarray(maps["closed_form"]["matrices"])
            a, ap, b, bp = jacobi.constant_curvature(_constant_curvature_of(key), fine.s)
            ciw_exact = np.stack([np.array([[x0, x2], [x1, x3]]) for x0, x1, x2, x3 in zip(a, ap, b, bp)])
            provider_rows[key] = {"ciw_rk4_vs_csg_closed_form": _discrepancy(ciw_phi, csg_exact),
                                  "ciw_rk4_vs_csg_rk4": _discrepancy(ciw_phi, csg_numeric),
                                  "ciw_closed_form_vs_csg_closed_form": _discrepancy(ciw_exact, csg_exact),
                                  "csg_determinant_drift": float(np.max(np.abs(
                                      np.asarray(maps["numeric"]["determinant"]) - 1.0)))}
    ctx.artifact_json("separation-law.json", _plain({"paths": {k: gj.path(k).as_dict() for k in CONSTANT_PATHS},
                                                     "rows": rows, "closed_form_separation": separation,
                                                     "separation_eps": SEPARATION_EPS, "provider": provider_rows,
                                                     "provider_status": None if csg is None else
                                                     csg.get("refusal", "compared")}))
    head_series, error_series = [], []
    for key in CONSTANT_PATHS:
        fine = _fine_transfer(ctx, key)
        stride = max(1, len(fine.s) // 60)
        head_series.append((f"{CONSTANT_SHORT[key]} K={_fmt(_constant_curvature_of(key), 2)}", fine.s[::stride],
                            fine.states[::stride, 6]))
        _, _, b, _ = jacobi.constant_curvature(_constant_curvature_of(key), fine.s)
        error_series.append((CONSTANT_SHORT[key], fine.s[1::stride],
                             np.maximum(np.abs(fine.states[1::stride, 6] - b[1::stride]), 1e-17)))
    ctx.artifact_text("heading-column.svg", svg.line_plot(head_series, title="Heading column j_head(s) = sn_K(s)",
                                                          xlabel="arclength s", ylabel="j_head", markers=False))
    ctx.artifact_text("heading-error.svg", svg.line_plot(error_series, title="|j_head - sn_K| along the paths",
                                                         xlabel="arclength s", ylabel="absolute error", logy=True,
                                                         markers=False))

    generator = {"name": "declared constant-curvature geodesics", "paths": list(CONSTANT_PATHS),
                 "steps": {k: _fine(k) for k in CONSTANT_PATHS}}
    halving = max(r["halving_estimate"] for r in rows.values())
    truncation = _unc("truncation_bound", halving, "RK4 step halving: |error(h) - error(h/2)| / 15, largest over "
                      "the paths; curvature references come from the surface parameters")
    findings = [finding(
        "The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on positive, zero "
        "and negative curvature", "numerical", {k: rows[k]["errors"]["heading"] for k in CONSTANT_PATHS},
        {"generator": generator, "checks": [
            gj.check("analytic", f"sn_K with K = {_fmt(rows[k]['curvature'], 6)} on {k}", rows[k]["errors"]["heading"],
                     1e-7) for k in CONSTANT_PATHS]}, uncertainty=truncation, tolerance=TOL_SMALL)]
    findings.append(finding(
        "The full integrated transfer matrix (lateral column and both rates) follows the model-space law cn_K, sn_K",
        "numerical", {k: rows[k]["max_error"] for k in CONSTANT_PATHS},
        {"generator": generator, "checks": [gj.check("analytic", f"cn_K, -K sn_K, sn_K, cn_K on {k}",
                                                     rows[k]["max_error"], 1e-7) for k in CONSTANT_PATHS]},
        uncertainty=truncation, tolerance=TOL_SMALL))
    sep_value = {k: {c: {"error_at_smallest_eps": v[c]["errors"][-1], "order": v[c]["order"]}
                     for c in ("lateral", "heading")} for k, v in separation.items()}
    sep_checks = []
    for key, v in separation.items():
        for column in ("lateral", "heading"):
            model = "|cn_K|" if column == "lateral" else "|sn_K|"
            if v[column]["exact_law"]:
                sep_checks.append(gj.check("analytic", f"{column} d(gamma_eps, gamma_0)/eps against {model} = 1 on "
                                           f"{key} (K = 0: parallel geodesics, no remainder), every eps",
                                           max(v[column]["errors"]), 1e-9, "le"))
                continue
            sep_checks.append(gj.check("self_convergence", f"{column} d(gamma_eps, gamma_0)/eps against {model} on "
                                       f"{key}: remainder order 2 in eps", v[column]["order"] - 2, 0.15))
            sep_checks.append(gj.check("analytic", f"{column} d(gamma_eps, gamma_0)/eps against {model} on {key} at "
                                       f"eps = {SEPARATION_EPS[-1]}", v[column]["errors"][-1], 5e-3, "le"))
    findings.append(finding(
        "Neighbouring closed-form geodesics on the sphere, the plane (seen in its polar chart) and the hyperbolic "
        "plane separate as |sn_K| (heading) and |cn_K| (lateral) per unit perturbation (K = 1, 0, -1)",
        "numerical", sep_value,
        {"generator": {"name": "closed-form perturbed geodesics", "paths": list(SEPARATION_PATHS),
                       "eps": list(SEPARATION_EPS), "nodes": SEPARATION_NODES}, "checks": sep_checks},
        uncertainty=_unc("truncation_bound", max(v[c]["errors"][-1] for v in separation.values()
                                                 for c in ("lateral", "heading")),
                         f"O(eps^2) remainder of the separation ratio at eps = {SEPARATION_EPS[-1]}"),
        tolerance={"abs": 1e-6, "rel": 1e-4}))
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
        uncertainty=truncation, tolerance=TOL_VALUE))
    orders = {k: rows[k]["observed_order"] for k in CONSTANT_PATHS if "observed_order" in rows[k]}
    findings.append(finding(
        "Residuals against the model-space law are fourth-order RK4 discretization error", "numerical", orders,
        {"generator": generator, "checks": [gj.check("self_convergence", f"error ratio at h and 2h on {k}",
                                                     v - 4, 0.3) for k, v in orders.items()]},
        uncertainty=_unc("fit", max(orders.values()) - min(orders.values()), "spread of the two-point step-halving "
                         "orders across the curved paths"), tolerance=TOL_RATE))
    state = "completed"
    identity = None
    notes = ["Paths of constant curvature only; variable curvature is covered by T006-T009"]
    if provider_rows is not None:
        rev = csg["identity"]["revision"]
        findings.append(finding(
            "ciw joint geodesic + Jacobi transfer matrices match the pinned CSG provider on six constant-curvature "
            "paths", "numerical", {k: {name: r[name] for name in ("ciw_rk4_vs_csg_closed_form", "ciw_rk4_vs_csg_rk4")}
                                   for k, r in provider_rows.items()},
            {"provider": {"repository": gj.CSG_REPOSITORY, "revision": rev,
                          "source_tree": csg["identity"]["source_tree"], "executed": True},
             "checks": [
                 gj.check("analytic", "CSG constant_curvature_transfer against ciw.lab.jacobi.constant_curvature",
                          max(r["ciw_closed_form_vs_csg_closed_form"] for r in provider_rows.values()), 1e-12),
                 # Two origins running the same method on the same grid: not cross_implementation, which the
                 # contract reserves for same-origin pairs; the independent check below sets the label.
                 gj.check("high_precision", "CSG integrate_jacobi RK4 trace against the ciw RK4 transfer on the same "
                          "grid: same method, different origin, so the ciw discrete solution is reproduced to "
                          "rounding", max(r["ciw_rk4_vs_csg_rk4"] for r in provider_rows.values()), 1e-9)],
             "independent_check": gj.independent(
                 gj.check("analytic", "CSG closed-form constant_curvature_transfer at the ciw nodes",
                          max(r["ciw_rk4_vs_csg_closed_form"] for r in provider_rows.values()), 1e-7),
                 "ciw.lab.jacobi", f"{gj.CSG_IMPLEMENTATION}@{rev}", checker_revision=rev)},
            uncertainty=_unc("truncation_bound", max(r["ciw_rk4_vs_csg_closed_form"] for r in provider_rows.values()),
                             "ciw RK4 truncation at the fine step; the provider closed form is exact to rounding"),
            tolerance=TOL_SMALL))
        identity = _provider_identity(csg)
    elif csg is not None:
        state = "partial"
        record, sentence = _refusal_record(ctx, csg)
        findings.append(record)
        notes.append(sentence)
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
                            "K = cos(theta)/(r(R + r cos(theta))); on constant curvature d(gamma_eps(s), gamma_0(s)) = "
                            "|eps| |j(s)| (1 + O(eps^2)), e.g. sin(d/2) = sin(eps/2) |sin s| on the unit sphere."),
        input_data=[f"{k}: K = {_fmt(rows[k]['curvature'], 6)}, L = {rows[k]['length']}, {rows[k]['steps']} RK4 steps"
                    for k in CONSTANT_PATHS] + [f"Closed-form perturbed geodesics on {', '.join(SEPARATION_PATHS)} "
                                                f"with eps in {list(SEPARATION_EPS)}"],
        observation_model=("Relative error max |j - model| / max(1, |model|) over all nodes, per column and rate; "
                           "exact intrinsic distance of matched points of closed-form geodesics."),
        expected_invariant=("Integrated columns equal sn_K and cn_K to RK4 accuracy with K from the surface "
                            "parameters; K stays constant along the path; closed-form separations approach "
                            "eps |sn_K| and eps |cn_K|."),
        experiment=("Integrate geodesic and Jacobi columns jointly on each surface, compare with the model-space "
                    "closed forms at h and 2h; measure the separation of closed-form perturbed geodesics on the "
                    "sphere, the plane (polar chart) and the hyperbolic plane; compare (when bound) with the "
                    "pinned CSG provider in a subprocess."),
        numerical_result=("Largest model-space error " + _fmt(max(r["max_error"] for r in rows.values()))
                          + "; observed orders " + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in orders.items())
                          + "; closed-form separation error at eps = 0.01 "
                          + ", ".join(f"{k} {_fmt(max(v[c]['errors'][-1] for c in ('lateral', 'heading')))}"
                                      for k, v in separation.items())
                          + ("" if provider_rows is None else "; ciw vs CSG closed form "
                             + _fmt(max(r["ciw_rk4_vs_csg_closed_form"] for r in provider_rows.values()))
                             + ", ciw vs CSG RK4 "
                             + _fmt(max(r["ciw_rk4_vs_csg_rk4"] for r in provider_rows.values())))
                          + "."),
        uncertainty="RK4 discretization error (about 1e-8 relative at the fine step), confirmed by step halving.",
        failure_modes_checked=["sign of K (oscillation versus exponential growth)",
                               "curvature reference taken from the surface parameters, not from gaussian_curvature",
                               "separation law on actual geodesics, not only on the scalar equation, for K = 1, "
                               "0 and -1 (K = 0 on straight lines seen through the polar chart, whose Christoffel "
                               "symbols are nonzero)",
                               "torus equator stays on the equator (theta drift)",
                               "flat charts where the columns are exactly 1 and s",
                               "provider checkout identity verified before and after execution"],
        unresolved_assumptions=notes + ["On the flat Cartesian charts (plane, cylinder) RK4 integrates j'' = 0 "
                                        "exactly, so those scalar-equation checks test only that K = 0 is "
                                        "integrated; the K = 0 separation law itself is measured on neighbouring "
                                        "straight lines through the polar chart of the plane",
                                        "No physical trajectories were measured; the physical-domain claim is "
                                        "recorded as not established"],
        recommended_next_task=("Deferred research question (hardware-gated): measure the separation law on real "
                               "neighbouring trajectories (for example two tracked markers on great circles of a "
                               "sphere and on a saddle, with a calibrated position sensor) and compare the measured "
                               "separation with eps j(s) within the sensor's calibrated uncertainty. Route: T005 "
                               "would read the tracker export as an operator capture (ctx.capture('trajectory-log'), "
                               "bound with ciw lab run T005 --capture trajectory-log=PATH) and fit the separation "
                               "from it as a computational finding; the physical finding needs, besides an "
                               "acquisition record (device, raw_sha256 of the captured bytes, acquired_at, "
                               "calibration), a probe of the tracker on the analysing host that succeeds in T005 "
                               "(runner.CAPTURE_INSTRUMENTS has no entry for trajectory-log) or a signed-capture "
                               "trust anchor, and the run is retained with ciw lab hardware retain under "
                               "lab/hardware/<run-id>. Neither the capture reader nor a tracker probe exists, so the "
                               "physical-domain finding stays not_established even when such data exist"))
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
                           "central_order": gj.slope(FD_EPS, central),
                           "one_sided_pairwise": [gj.slope(FD_EPS[i:i + 2], one[i:i + 2]) for i in range(3)],
                           "central_pairwise": [gj.slope(FD_EPS[i:i + 2], central[i:i + 2]) for i in range(3)]}
        rows[key] = row
    base = gj.transfer(ctx, "sphere", "rk4", FD_STEPS)
    roundoff = [float(np.max(np.abs(_perturbed_separation(ctx, "sphere", 0.0, eps) / eps - base.states[:, 6])))
                for eps in ROUNDOFF_EPS]
    return {"rows": rows, "roundoff": roundoff}


@task("T006", changed_files=CHANGED,
      regression_tests=(_test("test_t006_finite_differences"), _test("test_perturbation_helpers_are_geometric"))
      + SECTION_TESTS)
def finite_difference_jacobi(ctx):
    study = finite_difference_study(ctx)
    rows, roundoff = study["rows"], study["roundoff"]
    best = int(np.argmin(roundoff))
    ctx.artifact_json("finite-differences.json", _plain({"eps": FD_EPS, "steps": FD_STEPS, "rows": rows,
                                                         "roundoff_eps": ROUNDOFF_EPS, "roundoff_errors": roundoff}))
    for key in FD_SURFACES:
        ctx.artifact_text(f"fd-{key}.svg", svg.line_plot(
            [(f"{c} {kind.replace('_', '-')}", FD_EPS, rows[key][c][kind]) for c in ("lateral", "heading")
             for kind in ("one_sided", "central")], title=f"Finite differences vs Jacobi columns on {key}",
            xlabel="perturbation eps", ylabel="max |FD - column|", logx=True, logy=True))
    ctx.artifact_text("fd-roundoff.svg", svg.line_plot(
        [("heading one-sided", ROUNDOFF_EPS, roundoff)], title="Truncation versus cancellation (sphere)",
        xlabel="perturbation eps", ylabel="max |FD - column|", logx=True, logy=True))

    generator = {"name": "perturbed geodesics", "surfaces": list(FD_SURFACES), "eps": list(FD_EPS),
                 "steps": FD_STEPS}
    orders_one = {k: {c: rows[k][c]["one_sided_order"] for c in ("lateral", "heading")} for k in FD_SURFACES}
    orders_central = {k: {c: rows[k][c]["central_order"] for c in ("lateral", "heading")} for k in FD_SURFACES}
    fit = {kind: max(abs(v - rows[k][c][f"{kind}_order"]) for k in FD_SURFACES for c in ("lateral", "heading")
                     for v in rows[k][c][f"{kind}_pairwise"]) for kind in ("one_sided", "central")}
    fit_basis = "largest distance of a pairwise (eps-halving) slope from the least-squares slope"
    findings = [finding(
        "Central finite differences of perturbed geodesics converge to the integrated Jacobi columns at second order",
        "numerical", orders_central, {"generator": generator, "checks": [
            gj.check("analytic", f"central difference order 2, {c} column on {k}", v - 2, 0.1)
            for k, r in orders_central.items() for c, v in r.items()] + [
            gj.check("self_convergence", "largest central-difference error at eps = 0.01",
                     max(rows[k][c]["central"][-1] for k in FD_SURFACES for c in ("lateral", "heading")), 1e-3)]},
        uncertainty=_unc("fit", fit["central"], fit_basis), tolerance=TOL_RATE)]
    findings.append(finding(
        "One-sided finite differences converge to the integrated Jacobi columns at first order", "numerical",
        orders_one, {"generator": generator, "checks": [
            gj.check("analytic", f"one-sided difference order 1, {c} column on {k}", v - 1, 0.15)
            for k, r in orders_one.items() for c, v in r.items()]},
        uncertainty=_unc("fit", fit["one_sided"], fit_basis), tolerance=TOL_RATE))
    ratio = roundoff[-1] / roundoff[best]
    # Rounding-dominated numbers vary across platforms, so the retained value is in decades.
    findings.append(finding(
        "Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate (cancellation)",
        "numerical", {"log10_best_eps": math.log10(ROUNDOFF_EPS[best]), "log10_error_ratio_1e-11_over_best":
                      math.log10(ratio)},
        {"generator": dict(generator, eps=list(ROUNDOFF_EPS), surfaces=["sphere"]), "checks": [
            gj.check("invariant", "error at eps = 1e-11 over the best error", ratio, 10.0, "ge")]},
        uncertainty=_unc("roundoff", 1.0, "cancellation errors depend on platform rounding; values are decades "
                         "(log10) and are uncertain to about one decade"),
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
        recommended_next_task=("Deferred research question: repeat the comparison with an intrinsic separation "
                               "(geodesic distance between perturbed and base points at matched arclength, from the "
                               "34-digit references) instead of the chart difference projected on N, and measure how "
                               "much of the one-sided difference error is the chart's second-order term"))


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
                    # Errors are scaled by |a b'| + |a' b|: the determinant is a difference of products,
                    # so its rounding grows with the columns (cosh(6.5) on the inner equator).
                    y = tr.states
                    scale = np.maximum(np.abs(y[:, 4] * y[:, 7]) + np.abs(y[:, 5] * y[:, 6]), 1.0)
                    if method == "euler":
                        entry["per_step_factor_error"] = float(np.max(
                            np.abs(det[1:] - (1 + h * h * curvature[:-1]) * det[:-1]) / scale[1:]))
                    if k0 is not None:
                        predicted = per_step_determinant(method, h, k0) ** np.arange(steps + 1)
                        entry["prediction_error"] = float(np.max(np.abs(det - predicted) / scale))
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
      regression_tests=(_test("test_t007_determinant"), _test("test_t007_symbolic_step_determinants"))
      + SECTION_TESTS)
def wronskian_determinant(ctx):
    rows = determinant_study(ctx)
    curved = DET_CONSTANT + DET_VARIABLE
    slopes = {m: {k: _slopes(rows[k][m], "max_drift") for k in curved} for m in DET_STEPS}
    adaptive = {k: gj.slope([e["rtol"] for e in rows[k]["adaptive"]], [e["max_drift"] for e in rows[k]["adaptive"]])
                for k in curved}
    median_adaptive = float(np.median(list(adaptive.values())))
    pairwise = {m: {k: _pairwise(rows[k][m], "h", "max_drift") for k in curved} for m in DET_STEPS}
    fit_basis = "largest distance of a pairwise (halving) slope from the least-squares slope"
    euler_factor = max(e["per_step_factor_error"] for k in rows for e in rows[k]["euler"])
    prediction = {m: max(e["prediction_error"] for k in DET_CONSTANT + DET_FLAT for e in rows[k][m])
                  for m in DET_STEPS}
    boundary = {k: rows[k]["midpoint"][-1]["boundary_ratio"] for k in DET_VARIABLE}
    flat = max(e["max_drift"] for k in DET_FLAT for m in DET_STEPS for e in rows[k][m])
    symbolic = symbolic_step_determinants() if ctx.available("module:sympy") else None
    ctx.artifact_json("determinant.json", _plain({"steps": DET_STEPS, "rtol": DET_RTOL, "rows": rows,
                                                  "drift_orders": slopes, "adaptive_tolerance_exponent": adaptive,
                                                  "symbolic_step_determinants": symbolic}))
    # One figure per method and curvature class keeps each within the palette's eight distinct colours.
    for method in DET_STEPS:
        for kind, keys in (("constant", DET_CONSTANT), ("variable", DET_VARIABLE)):
            ctx.artifact_text(f"det-{method}-{kind}.svg", svg.line_plot(
                [(k, [e["h"] for e in rows[k][method]], [e["max_drift"] for e in rows[k][method]]) for k in keys],
                title=f"max |det Phi - 1|, {method}, {kind} curvature", xlabel="step h", ylabel="max |det Phi - 1|",
                logx=True, logy=True))
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
                     0.1) for k in DET_VARIABLE]},
        uncertainty=_unc("fit", _fit_spread({k: pairwise["euler"][k] for k in DET_VARIABLE}, slopes["euler"]),
                         fit_basis + " (variable-curvature paths); the per-step factors hold to rounding"),
        tolerance=TOL_RATE)]
    mid_checks = [gj.check("analytic", "(1 + h^4 K^2 / 4)^n on constant-curvature paths", prediction["midpoint"], 1e-10)]
    mid_checks += [gj.check("analytic", f"midpoint drift order 3 on constant-curvature {k}", slopes["midpoint"][k] - 3,
                            0.15) for k in DET_CONSTANT]
    mid_checks += [gj.check("analytic", f"midpoint drift order 2 on variable-curvature {k}", slopes["midpoint"][k] - 2,
                            0.15) for k in DET_VARIABLE]
    mid_checks += [gj.check("analytic", f"end drift / (h^2 (K(L) - K(0)) / 4) at N=200 on {k}", v - 1, 0.05)
                   for k, v in boundary.items()]
    # The O(h^3) remainder makes |ratio - 1| shrink like h: halving h should roughly halve it, and the
    # Richardson combination 2 r(h) - r(2h) should then sit much closer to 1.
    mid_checks += [gj.check("self_convergence", f"|ratio - 1| at N=100 over N=200 on {k}",
                            abs(rows[k]["midpoint"][-2]["boundary_ratio"] - 1)
                            / abs(rows[k]["midpoint"][-1]["boundary_ratio"] - 1), 1.5, "ge") for k in DET_VARIABLE]
    extrapolated = {k: 2 * rows[k]["midpoint"][-1]["boundary_ratio"] - rows[k]["midpoint"][-2]["boundary_ratio"]
                    for k in DET_VARIABLE}
    mid_checks += [gj.check("self_convergence", f"extrapolated ratio 2 r(N=200) - r(N=100) on {k}", v - 1, 1e-3)
                   for k, v in extrapolated.items()]
    findings.append(finding(
        "Midpoint determinant drift is (h^2/4)(K(L) - K(0)) + O(h^3): second order on variable curvature, third "
        "order on constant curvature", "numerical", {"orders": {k: slopes["midpoint"][k] for k in curved},
                                                      "boundary_ratio": boundary,
                                                      "extrapolated_boundary_ratio": extrapolated},
        {"generator": generator, "checks": mid_checks},
        uncertainty=_unc("fit", _fit_spread(pairwise["midpoint"], slopes["midpoint"]), fit_basis),
        tolerance=TOL_RATE))
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
        uncertainty=_unc("fit", _fit_spread(pairwise["rk4"], slopes["rk4"]), fit_basis), tolerance=TOL_RATE,
        counterexample={"statement": "The RK4 transfer-matrix determinant drifts at the method's global order h^4",
                        "witness": {"orders": {k: slopes["rk4"][k] for k in curved},
                                    "per_step_defect": "-h^6 K^3/72 + O(h^7) (constant K); O(h^6) for smooth K(s)"}}))
    # Per-path slopes shift with single accept/reject decisions; the claim is about their median.
    findings.append(finding(
        "Adaptive Dormand-Prince determinant drift decreases in proportion to the tolerance or faster (median over "
        "paths)", "numerical", median_adaptive, {"generator": dict(generator, rtol=list(DET_RTOL)), "checks": [
            gj.check("analytic", "median log-log slope of drift against rtol, at least half-way from 0.8 (a y4-"
                     "advancing pair) to 1", median_adaptive, ADAPTIVE_EXPONENT_THRESHOLD, "signed_ge")]},
        uncertainty=_unc("fit", float(np.median(np.abs(np.array(list(adaptive.values())) - median_adaptive))),
                         "median absolute deviation of the nine per-path slopes"),
        tolerance={"abs": 0.15, "rel": 0.0}))
    findings.append(finding(
        "Where K = 0 every method preserves det Phi = 1 exactly, Euler included", "numerical", flat,
        {"generator": generator, "checks": [gj.check("analytic", "max |det Phi - 1| on plane and cylinder, all methods",
                                                     flat, 1e-14)]},
        uncertainty=_unc("roundoff", flat, "binary64 rounding of j_lat j_head' - j_lat' j_head"), tolerance=TOL_SMALL,
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
                            "5 (RK4); adaptive drift proportional to the tolerance or faster (median slope >= "
                            f"{ADAPTIVE_EXPONENT_THRESHOLD})."),
        experiment=("Integrate geodesic and Jacobi columns jointly with each method and step, compare the "
                    "determinant with the per-step predictions, fit drift orders, and derive the step determinants "
                    "symbolically."),
        numerical_result=("Drift orders: " + "; ".join(f"{m} " + ", ".join(f"{k} {_fmt(v, 3)}"
                                                                          for k, v in slopes[m].items())
                                                      for m in DET_STEPS)
                          + f"; Euler per-step factor error {_fmt(euler_factor)}; midpoint boundary ratio "
                          f"{_fmt(min(boundary.values()), 4)}..{_fmt(max(boundary.values()), 4)} at N = 200; "
                          f"adaptive median drift slope {_fmt(median_adaptive, 3)}."),
        uncertainty=("Fitted orders over four halvings; the exact per-step predictions hold to rounding once "
                     "scaled by |a b'| + |a' b| (the columns reach cosh(6.5) on the inner equator). Euler slopes on "
                     "the long constant-curvature paths are not fitted claims: (1 + h^2 K)^(L/h) - 1 ~ "
                     "exp(h L K) - 1 is not yet linear in h when h L |K| is near 1, so the exact product formula "
                     "is checked there instead."),
        failure_modes_checked=["Euler area growth (K > 0) and shrinkage (K < 0) from 1 + h^2 K",
                               "midpoint cancellation when K(L) = K(0) (constant K gives order 3)",
                               "RK4 superconvergence of the determinant (recorded as a counterexample to the h^4 "
                               "expectation)", "flat charts where every method is exact"],
        unresolved_assumptions=["The O(h^6) RK4 per-step defect is derived for smooth K(s) along the stage points; "
                                "paths through curvature discontinuities were not tested"],
        recommended_next_task=("Deferred research question: characterize the leading RK4 determinant coefficient "
                               "-(4 k0^3 - k0 k2 + 2 k1^2 + 4 k0 (e2 + e3))/288 along whole paths (stage offsets "
                               "included), and measure the determinant defect on a path through a curvature "
                               "discontinuity (for example a cylinder joined to a spherical cap), where the O(h^6) "
                               "per-step derivation for smooth K(s) does not apply"))


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


# Declared bump chords through the summit region: long enough for the positive-curvature cap to focus them.
# They meet K > 0 only after s = 7, beyond the global bound pi/sqrt(max K) = 2 pi, so on the bump that bound
# holds whatever the Jacobi integration does; the chords are tested by the two-sided comparison instead.
BUMP_CHORDS = ((-8.0, 0.0), (-8.0, 0.2), (-10.0, 0.0), (-10.0, 0.2))
BUMP_CHORD_LENGTH = 30.0
STURM_RTOL = (1e-9, 1e-10)
# Two-sided comparison: piecewise-constant curvature envelopes on a fixed-step copy of each geodesic.
ENVELOPE_STEP = 0.01
PATH_ALLOWANCE = 1e-6   # widening of every coordinate interval, covering the error of the fixed-step nodes
ANGLE_ALLOWANCE = 1e-7  # integration error allowed when the ciw Pruefer angle is compared with the envelopes
CONTROL_SCALE = 2.0     # negative control: the heading column integrated with CONTROL_SCALE * K


def _scaled_transfer(surf, u0, heading, length, scale, rtol, atol):
    """Geodesic with Jacobi columns of j'' + scale K j = 0 (scale 1 is ``jacobi.transfer``); a negative control."""
    y0 = jacobi.initial_state(surf, u0, heading)

    def f(y):
        k = scale * surf.gaussian_curvature(y[:2])
        return np.concatenate([surf.geodesic_rhs(y[:4]), [y[5], -k * y[4], y[7], -k * y[6]]])

    s, states, stats = integrators.integrate_adaptive(f, y0, length, rtol=rtol, atol=atol)
    return jacobi.Transfer(surf, s, states, stats)


def curvature_profile(name: str) -> dict:
    """The scalar coordinate q that K depends on, its Lipschitz constant in arclength, and K's range on q-intervals.

    Torus: K = cos(theta) / (r (R + r cos(theta))) increases with cos(theta), and r^2 theta'^2 <= 1 at unit speed,
    so |dtheta/ds| <= 1/r. Gaussian bump: K depends on rho = |(x, y)| only; the Monge metric is at least the
    identity, so |drho/ds| <= 1; K decreases on [0, rho*] and increases beyond (checked on a grid by the caller).
    K values come from ``gaussian_curvature``, the function the Jacobi integration uses.
    """
    surf = gj.surface(name)
    if name == "torus":
        def k(t):
            return surf.gaussian_curvature(np.array([0.0, t]))

        def k_range(lo, hi):
            top = 2 * math.pi * math.ceil(lo / (2 * math.pi))
            bottom = math.pi + 2 * math.pi * math.ceil((lo - math.pi) / (2 * math.pi))
            ends = (k(lo), k(hi))
            return (k(math.pi) if bottom <= hi else min(ends)), (k(0.0) if top <= hi else max(ends))

        return {"coordinate": lambda u: u[:, 1], "lipschitz": 1.0 / surf.minor, "range": k_range, "k": k}
    if name == "gaussian-bump":
        def k(r):
            return surf.gaussian_curvature(np.array([r, 0.0]))

        a, b = surf.sigma, 3.0 * surf.sigma  # golden section for the flank minimum rho*
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(80):
            c, d = b - ratio * (b - a), a + ratio * (b - a)
            a, b = (a, d) if k(c) < k(d) else (c, b)
        rho_star = (a + b) / 2

        def k_range(lo, hi):
            lo = max(lo, 0.0)
            ends = (k(lo), k(hi))
            return (k(rho_star) if lo <= rho_star <= hi else min(ends)), max(ends)

        return {"coordinate": lambda u: np.hypot(u[:, 0], u[:, 1]), "lipschitz": 1.0, "range": k_range, "k": k,
                "rho_star": rho_star}
    raise ValueError(f"No curvature profile for {name}")


def _step_matrix(k: float, h: float) -> np.ndarray:
    """Exact transfer matrix of j'' + k j = 0 over a length h."""
    if k > 0:
        w = math.sqrt(k)
        return np.array([[math.cos(w * h), math.sin(w * h) / w], [-w * math.sin(w * h), math.cos(w * h)]])
    if k < 0:
        w = math.sqrt(-k)
        return np.array([[math.cosh(w * h), math.sinh(w * h) / w], [w * math.sinh(w * h), math.cosh(w * h)]])
    return np.array([[1.0, h], [0.0, 1.0]])


def comparison_solution(nodes, curvatures) -> dict:
    """Heading data (0, 1) propagated exactly through piecewise-constant K on [nodes[i], nodes[i+1]].

    Returns the node states, the unwrapped Pruefer angles atan2(j, j') and the first zero after s = 0 (None if
    there is none), located by bisection on the closed form inside its interval.
    """
    states, angles = [np.array([0.0, 1.0])], [0.0]
    for i, k in enumerate(curvatures):
        state = _step_matrix(k, nodes[i + 1] - nodes[i]) @ states[-1]
        angles.append(angles[-1] + math.remainder(math.atan2(state[0], state[1]) - angles[-1], 2 * math.pi))
        states.append(state)
    angles = np.array(angles)
    zero = None
    crossing = np.nonzero(angles >= math.pi)[0]
    if len(crossing):
        i = int(crossing[0]) - 1
        a, b = 0.0, nodes[i + 1] - nodes[i]
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if (_step_matrix(curvatures[i], m) @ states[i])[0] > 0 else (a, m)
        zero = float(nodes[i] + (a + b) / 2)
    return {"nodes": np.asarray(nodes), "curvatures": np.asarray(curvatures), "states": states, "angles": angles,
            "first_zero": zero}


def comparison_angles(solution: dict, s_eval) -> np.ndarray:
    """Pruefer angle of a :func:`comparison_solution` at arbitrary arclengths inside its node range."""
    nodes, ks = solution["nodes"], solution["curvatures"]
    index = np.clip(np.searchsorted(nodes, s_eval, side="right") - 1, 0, len(ks) - 1)
    out = []
    for s, i in zip(s_eval, index):
        state = _step_matrix(ks[i], s - nodes[i]) @ solution["states"][i]
        base = solution["angles"][i]
        out.append(base + math.remainder(math.atan2(state[0], state[1]) - base, 2 * math.pi))
    return np.array(out)


def pruefer_angle(tr) -> np.ndarray:
    """Unwrapped Pruefer angle atan2(j_head, j_head') of an integrated heading column."""
    return np.unwrap(np.arctan2(tr.states[:, 6], tr.states[:, 7]))


def two_sided_comparison(surf_name, u0, heading, length, tr, control) -> dict:
    """Bracket the heading column between the solutions for the upper and lower curvature envelopes.

    Sturm (Pruefer) comparison: theta' = cos^2 theta + K sin^2 theta increases with K, so K_lo <= K <= K_hi along
    the geodesic gives theta_lo <= theta <= theta_hi at every s, and the first conjugate point lies between the
    first zeros of the two envelope solutions. The envelopes bound K over each interval of a fixed-step copy of
    the geodesic: q moves by at most c h / 2 from the mean of its end values (c the Lipschitz constant).
    """
    profile = curvature_profile(surf_name)
    surf = gj.surface(surf_name)
    steps = int(math.ceil(length / ENVELOPE_STEP))
    nodes, path_states = integrators.integrate_fixed(surf.geodesic_rhs, jacobi.initial_state(surf, u0, heading)[:4],
                                                     length, steps, "rk4")
    q = profile["coordinate"](path_states[:, :2])
    h = length / steps
    reach = profile["lipschitz"] * h / 2
    mean = (q[:-1] + q[1:]) / 2
    lows = np.minimum(mean - reach, np.minimum(q[:-1], q[1:])) - PATH_ALLOWANCE
    highs = np.maximum(mean + reach, np.maximum(q[:-1], q[1:])) + PATH_ALLOWANCE
    ranges = [profile["range"](lo, hi) for lo, hi in zip(lows, highs)]
    lower = comparison_solution(nodes, [r[0] for r in ranges])
    upper = comparison_solution(nodes, [r[1] for r in ranges])

    def gaps(transfer):
        theta, s = pruefer_angle(transfer)[1:], transfer.s[1:]
        return float(np.min(theta - comparison_angles(lower, s))), float(np.min(comparison_angles(upper, s) - theta))

    below, above = gaps(tr)
    control_below, control_above = gaps(control)
    positive = np.nonzero(np.array([profile["k"](x) for x in q]) > 0)[0]
    return {"lower_gap": below, "upper_gap": above, "control_gap": min(control_below, control_above),
            "zero_bracket": [upper["first_zero"], lower["first_zero"]],
            "endpoint_path_gap": float(abs(q[-1] - profile["coordinate"](tr.states[-1:, :2])[0])),
            "first_positive_curvature_s": float(nodes[positive[0]]) if len(positive) else None,
            "coordinate_range": [float(q.min()), float(q.max())], "envelope_steps": steps}


def sturm_study(ctx):
    """Seeded torus and bump geodesics plus declared bump chords: zeros at two tolerances and the comparisons."""
    def compute():
        rng = np.random.Generator(np.random.PCG64(gj.SEED + 8))
        starts = []
        for name, count, box, length in (("torus", STURM_TORUS, ((0, 2 * math.pi), (-math.pi, math.pi)), 12.0),
                                         ("gaussian-bump", STURM_BUMP, ((-1.0, 1.0), (-1.0, 1.0)), 10.0)):
            for _ in range(count):
                u0 = (float(rng.uniform(*box[0])), float(rng.uniform(*box[1])))
                starts.append((name, "seeded", u0, float(rng.uniform(-math.pi, math.pi)), length))
        starts += [("gaussian-bump", "declared chord", u0, 0.0, BUMP_CHORD_LENGTH) for u0 in BUMP_CHORDS]
        rows = []
        for name, origin, u0, heading, length in starts:
            surf = gj.surface(name)
            tr = jacobi.transfer(surf, u0, heading, length, rtol=STURM_RTOL[0], atol=1e-11)
            control = _scaled_transfer(surf, u0, heading, length, CONTROL_SCALE, STURM_RTOL[0], 1e-11)
            row = {"surface": name, "origin": origin, "u0": list(u0), "heading": heading, "length": length,
                   "conjugate": tr.conjugate_points(), "focal": tr.focal_points(),
                   "max_curvature_on_path": float(tr.curvature_along().max()),
                   "control_conjugate": control.conjugate_points(),
                   "comparison": two_sided_comparison(name, u0, heading, length, tr, control)}
            if row["conjugate"] or row["focal"]:
                # Zero locations at a ten times tighter tolerance bound their integration error.
                tight = jacobi.transfer(surf, u0, heading, length, rtol=STURM_RTOL[1], atol=1e-12)
                pairs = list(zip(row["conjugate"], tight.conjugate_points())) + list(zip(row["focal"],
                                                                                          tight.focal_points()))
                row["location_change"] = max(abs(a - b) for a, b in pairs) if pairs else 0.0
                row["count_change"] = int(len(tight.conjugate_points()) != len(row["conjugate"])
                                          or len(tight.focal_points()) != len(row["focal"]))
            rows.append(row)
        return rows
    return ctx.memo("gj-sturm", compute)


@task("T008", changed_files=CHANGED, regression_tests=(_test("test_t008_conjugate_and_focal_points"),
                                                       _test("test_t008_envelope_comparison_rejects_wrong_curvature"),
                                                       _test("test_t008_missing_witness_is_recorded_not_raised"),
                                                       _test("test_csg_checkout_refusals"),
                                                       _test("test_csg_execution_refusals_are_expected_from_their_stage"))
      + SECTION_TESTS)
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
                         "max_s_minus_j_head": float(np.max(tr.s[1:] - tr.states[1:, 6])),
                         "max_1_minus_j_lat": float(np.max(1.0 - tr.states[1:, 4])),
                         "model_error": max(_model_error(tr, _constant_curvature_of(key)).values())}
    sturm = sturm_study(ctx)
    torus_rows = [r for r in sturm if r["surface"] == "torus"]
    # Global upper-curvature bound on the torus: K_max = 1/(r(R+r)) on the outer equator, from the parameters.
    k_max = 1.0 / (torus.minor * (torus.major + torus.minor))
    bound = math.pi / math.sqrt(k_max)
    firsts = [r["conjugate"][0] for r in torus_rows if r["conjugate"]]
    margin = min(firsts) - bound if firsts else None
    # Negative control: the column integrated with 2K must violate the bound; with no zero before L the margin is at
    # least L - bound, so the control check fails rather than passing vacuously.
    control_margin = min(r["control_conjugate"][0] - bound if r["control_conjugate"] else r["length"] - bound
                         for r in torus_rows)
    # On the bump the global bound 2 pi = pi/sqrt(h^2/sigma^4) is not exercised: K > 0 only for rho < sigma, and
    # every conjugate point found lies far beyond it (recorded, not claimed).
    bump_rows = [r for r in sturm if r["surface"] == "gaussian-bump"]
    bump_bound = math.pi / math.sqrt(bump.h ** 2 / bump.sigma ** 4)
    profile = curvature_profile("gaussian-bump")
    radii = np.arange(0.0, max(r["comparison"]["coordinate_range"][1] for r in bump_rows) + 1.0, 2e-3)
    k_radial = np.array([profile["k"](x) for x in radii])
    split = int(np.searchsorted(radii, profile["rho_star"]))
    # Grid points below rho* must decrease and those at or beyond it increase (the straddling pair is not compared).
    unimodal_violations = int(np.sum(np.diff(k_radial[:split]) > 0) + np.sum(np.diff(k_radial[split:]) < 0))
    comparison = {"lower_gap": min(r["comparison"]["lower_gap"] for r in sturm),
                  "upper_gap": min(r["comparison"]["upper_gap"] for r in sturm),
                  "control_gap": max(r["comparison"]["control_gap"] for r in sturm),
                  "endpoint_path_gap": max(r["comparison"]["endpoint_path_gap"] for r in sturm)}
    chords = [r for r in sturm if r["origin"] == "declared chord"]
    names = [f"{r['surface']} {r['origin']} {i}" for i, r in enumerate(sturm)]
    brackets = {n: {"conjugate": r["conjugate"][0], "bracket": r["comparison"]["zero_bracket"]}
                for n, r in zip(names, sturm) if r["conjugate"]}
    located_change = max([r.get("location_change", 0.0) for r in sturm])
    count_changes = sum(r.get("count_change", 0) for r in sturm)
    witness = next((r for r in sturm if r["surface"] == "torus" and r["conjugate"] and r["focal"]
                    and abs(r["focal"][0] - r["conjugate"][0] / 2) > 0.3), None)
    csg = _csg(ctx) if ctx.available("provider:csg") else None
    provider = None
    if csg and "data" in csg:
        provider = {}
        for key, maps in zip(csg["paths"], csg["data"]["maps"], strict=True):
            if key not in ("sphere-great-circle", "torus-outer-equator", "torus-inner-equator", "hyperbolic-long"):
                continue
            tr = _fine_transfer(ctx, key)
            ours = {"b": tr.conjugate_points(), "a": tr.focal_points()}
            row = {"ciw_conjugate": ours["b"], "ciw_focal": ours["a"]}
            for source in ("numeric", "closed_form"):
                events = {c: [e["arc_length"] for e in maps[source]["focus_events"][c]] for c in ("a", "b")}
                # Gaps over paired zeros; a count mismatch is checked separately and never hidden.
                row[f"csg_{source}"] = events
                row[f"{source}_count_mismatch"] = sum(len(events[c]) != len(ours[c]) for c in ("a", "b"))
                row[f"{source}_gap"] = max([abs(x - y) for c in ("a", "b") for x, y in zip(events[c], ours[c])]
                                           + [0.0])
            provider[key] = row
    ctx.artifact_json("conjugate-focal.json", _plain({
        "located": located, "sphere_radius_2": sphere2, "negative_curvature": negative,
        "sturm": {"torus_bound": bound, "torus_k_max": k_max, "rtol": STURM_RTOL, "paths": sturm,
                  "bump_global_bound_not_exercised": {
                      "bound": bump_bound, "first_conjugate_points": [r["conjugate"][0] for r in bump_rows
                                                                      if r["conjugate"]],
                      "first_positive_curvature_s": {str(r["u0"]): r["comparison"]["first_positive_curvature_s"]
                                                     for r in chords}},
                  "two_sided": {"envelope_step": ENVELOPE_STEP, "path_allowance": PATH_ALLOWANCE,
                                "angle_allowance": ANGLE_ALLOWANCE, "control_scale": CONTROL_SCALE,
                                "summary": comparison, "zero_brackets": brackets,
                                "bump_rho_star": profile["rho_star"],
                                "bump_unimodality_violations": unimodal_violations}},
        "provider": provider}))
    ctx.artifact_text("location-accuracy.svg", svg.line_plot(
        [(k, [e["h"] for e in v["entries"]], [e["max_error"] for e in v["entries"]]) for k, v in located.items()],
        title="Conjugate/focal point location error", xlabel="step h", ylabel="max location error", logx=True,
        logy=True))
    tr = _fine_transfer(ctx, "torus-outer-equator")
    ctx.artifact_text("columns-outer-equator.svg", svg.line_plot(
        [("j_head", tr.s, tr.states[:, 6]), ("j_lat", tr.s, tr.states[:, 4])],
        title="Torus outer equator: conjugate (j_head) and focal (j_lat) zeros", xlabel="arclength s",
        ylabel="j", markers=False))

    findings = []
    sphere = located["sphere-great-circle"]
    outer = located["torus-outer-equator"]

    def location_uncertainty(entries):
        change = abs(entries[-2]["max_error"] - entries[-1]["max_error"])
        return _unc("truncation_bound", entries[-1]["max_error"], "location error against the pi multiples at the "
                    f"finest step; halving the step changed it by {_fmt(change)}")

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
        uncertainty=location_uncertainty(sphere["entries"]), tolerance=TOL_VALUE))
    findings.append(finding(
        "On the torus outer equator conjugate points lie at pi sqrt(r(R+r)) multiples and focal points half-way",
        "numerical", {"conjugate": outer["entries"][-1]["conjugate"], "focal": outer["entries"][-1]["focal"],
                      "analytic_first_conjugate": math.pi / math.sqrt(k_outer)},
        {"generator": {"name": "torus outer equator", "major": torus.major, "minor": torus.minor}, "checks": [
            gj.check("analytic", "locations at 320 steps", outer["entries"][-1]["max_error"], 1e-7),
            gj.check("analytic", "location error order 4", outer["order"] - 4, 0.3),
            gj.check("exact_arithmetic", "zero counts match",
                     float(not outer["entries"][-1]["counts_match"]), 0.0)]},
        uncertainty=location_uncertainty(outer["entries"]), tolerance=TOL_VALUE))
    findings.append(finding(
        "No conjugate or focal point occurs on the torus inner equator or the hyperbolic plane (K < 0)", "numerical",
        {k: {"conjugate_count": len(v["conjugate"]), "focal_count": len(v["focal"]),
             "max_s_minus_j_head": v["max_s_minus_j_head"], "max_1_minus_j_lat": v["max_1_minus_j_lat"]}
         for k, v in negative.items()},
        {"generator": {"name": "negative-curvature paths", "paths": list(negative)}, "checks": [
            check for k, v in negative.items() for check in (
                gj.check("exact_arithmetic", f"zeros found on {k}", len(v["conjugate"]) + len(v["focal"]), 0.0),
                gj.check("invariant", f"Sturm comparison: largest s - j_head for s > 0 on {k}",
                         v["max_s_minus_j_head"], 1e-9, "signed_le"),
                gj.check("invariant", f"Sturm comparison: largest 1 - j_lat for s > 0 on {k}",
                         v["max_1_minus_j_lat"], 1e-9, "signed_le"))]},
        uncertainty=_unc("truncation_bound", max(v["model_error"] for v in negative.values()),
                         "RK4 error of the columns against cosh and sinh at the fine step; counts are exact"),
        tolerance=TOL_VALUE))
    sturm_generator = {"name": "seeded geodesics and declared bump chords", "seed": gj.SEED + 8,
                       "torus": STURM_TORUS, "bump": STURM_BUMP, "bump_chords": [list(c) for c in BUMP_CHORDS],
                       "chord_length": BUMP_CHORD_LENGTH, "rtol": STURM_RTOL[0], "control_scale": CONTROL_SCALE}
    findings.append(finding(
        "Sturm comparison bound holds on every seeded torus geodesic that reaches a conjugate point (none occurs "
        "before pi/sqrt(max K)) and rejects a heading column integrated with 2K", "numerical",
        {"bound": bound, "first_conjugate_points": firsts, "margin": margin, "control_margin": control_margin,
         "paths_with_conjugate_point": len(firsts)},
        {"generator": {"name": "seeded torus geodesics", "seed": gj.SEED + 8, "count": STURM_TORUS, "length": 12.0,
                       "rtol": STURM_RTOL[0], "control_scale": CONTROL_SCALE}, "checks": [
            gj.check("invariant", "seeded torus geodesics that reach a conjugate point", len(firsts), 1.0, "ge"),
            gj.check("invariant", "first conjugate point minus pi/sqrt(K_max) on the torus",
                     -1.0 if margin is None else margin, 0.0, "signed_ge"),
            gj.check("invariant", "negative control: first zero of the column integrated with 2K minus "
                     "pi/sqrt(K_max) (must be negative on some geodesic)", control_margin, 0.0, "signed_le")]},
        uncertainty=_unc("truncation_bound", located_change, "largest change of a zero location between adaptive "
                         "rtol 1e-9 and 1e-10"),
        tolerance={"abs": 1e-6, "rel": 1e-6}))
    findings.append(finding(
        "Two-sided Sturm comparison with piecewise-constant curvature envelopes brackets the heading column on "
        "every seeded torus and bump geodesic and declared bump chord, and rejects the column integrated with 2K",
        "numerical",
        {"smallest_gap_above_lower_envelope": comparison["lower_gap"],
         "smallest_gap_below_upper_envelope": comparison["upper_gap"],
         "largest_control_gap": comparison["control_gap"], "zero_brackets": brackets,
         "chords_with_conjugate_point": sum(1 for r in chords if r["conjugate"])},
        {"generator": dict(sturm_generator, envelope_step=ENVELOPE_STEP, path_allowance=PATH_ALLOWANCE), "checks": [
            gj.check("invariant", "smallest Pruefer angle minus the lower-envelope angle for s > 0, all paths",
                     comparison["lower_gap"], -ANGLE_ALLOWANCE, "signed_ge"),
            gj.check("invariant", "smallest upper-envelope angle minus the Pruefer angle for s > 0, all paths",
                     comparison["upper_gap"], -ANGLE_ALLOWANCE, "signed_ge"),
            gj.check("invariant", "negative control: largest envelope gap of the column integrated with 2K over all "
                     "paths (negative means rejected)", comparison["control_gap"], -ANGLE_ALLOWANCE, "signed_le"),
            gj.check("invariant", "declared bump chords that reach a conjugate point",
                     sum(1 for r in chords if r["conjugate"]), 1.0, "ge"),
            gj.check("self_convergence", "coordinate gap between the fixed-step envelope path and the adaptive path at "
                     "s = L", comparison["endpoint_path_gap"], PATH_ALLOWANCE, "le"),
            gj.check("analytic", "bump curvature at the summit minus h^2/sigma^4",
                     profile["k"](0.0) - bump.h ** 2 / bump.sigma ** 4, 1e-15),
            gj.check("exact_arithmetic", "grid violations of K(rho) decreasing up to rho* and increasing beyond",
                     unimodal_violations, 0.0),
            gj.check("self_convergence", "zero-count changes between rtol 1e-9 and 1e-10", count_changes, 0.0)]},
        uncertainty=_unc("truncation_bound", max(located_change, comparison["endpoint_path_gap"]),
                         "largest change of a zero location between adaptive rtol 1e-9 and 1e-10, and the "
                         "fixed-step path error covered by the coordinate allowance"),
        tolerance={"abs": 1e-6, "rel": 1e-6}))
    if witness is not None:
        findings.append(finding(
            "On variable curvature the first focal point is not half the first conjugate distance", "numerical",
            {"focal": witness["focal"][0], "conjugate": witness["conjugate"][0]},
            {"generator": {"name": "seeded torus geodesic", "u0": witness["u0"], "heading": witness["heading"]},
             "checks": [gj.check("invariant", "|first focal - first conjugate / 2|",
                                 abs(witness["focal"][0] - witness["conjugate"][0] / 2), 0.3, "ge")]},
            uncertainty=_unc("truncation_bound", witness["location_change"], "change of the zero locations between "
                             "adaptive rtol 1e-9 and 1e-10"),
            tolerance={"abs": 1e-6, "rel": 1e-6},
            counterexample={"statement": "The first focal point lies at half the first conjugate distance",
                            "witness": {"surface": witness["surface"], "u0": witness["u0"],
                                        "heading": witness["heading"], "first_focal": witness["focal"][0],
                                        "first_conjugate": witness["conjugate"][0]}}))
    else:
        # Recorded, not raised: a generator basis would label it synthetic, so the sample is declared as inputs
        # (which add no label) and the claim stays not_established.
        separations = [abs(r["focal"][0] - r["conjugate"][0] / 2) for r in sturm
                       if r["surface"] == "torus" and r["conjugate"] and r["focal"]]
        findings.append(finding(
            "On variable curvature the first focal point is not half the first conjugate distance", "numerical",
            {"torus_geodesics": STURM_TORUS, "with_focal_and_conjugate": len(separations),
             "largest_separation": max(separations, default=None)},
            {"inputs": {"name": "seeded torus geodesics", "seed": gj.SEED + 8, "count": STURM_TORUS}},
            uncertainty=_unc("truncation_bound", located_change, "largest change of a zero location between "
                             "adaptive rtol 1e-9 and 1e-10; no witness separated by more than 0.3"),
            tolerance={"abs": 1e-6, "rel": 1e-6}, expected_not_established=True))
    state, notes, identity = "completed", [], None
    if witness is None:
        notes.append("No seeded torus geodesic separated its first focal point from half its first conjugate "
                     "distance by more than 0.3; the counterexample was not found in this sample")
    if provider is not None:
        rev = csg["identity"]["revision"]
        findings.append(finding(
            "ciw conjugate and focal points match the pinned CSG provider's focus events on constant-curvature paths",
            "numerical", {k: {"closed_form_gap": v["closed_form_gap"], "numeric_gap": v["numeric_gap"]}
                          for k, v in provider.items()},
            {"provider": {"repository": gj.CSG_REPOSITORY, "revision": rev,
                          "source_tree": csg["identity"]["source_tree"], "executed": True},
             "checks": [
                 gj.check("exact_arithmetic", "zero-count mismatches against CSG (closed form and RK4 trace)",
                          sum(v["closed_form_count_mismatch"] + v["numeric_count_mismatch"] for v in provider.values()),
                          0.0),
                 # Two origins running the same method on the same grid: not cross_implementation, which the
                 # contract reserves for same-origin pairs.
                 gj.check("high_precision", "CSG focus_events on its own RK4 trace: same method and grid, different "
                          "origin, so the ciw zeros are reproduced to Hermite root-finding accuracy",
                          max(v["numeric_gap"] for v in provider.values()), 1e-8)],
             "independent_check": gj.independent(
                 gj.check("analytic", "CSG focus_events of the closed-form constant_curvature_transfer",
                          max(v["closed_form_gap"] for v in provider.values()), 1e-7),
                 "ciw.lab.jacobi", f"{gj.CSG_IMPLEMENTATION}@{rev}", checker_revision=rev)},
            uncertainty=_unc("truncation_bound", max(v["closed_form_gap"] for v in provider.values()),
                             "ciw RK4 + Hermite location error at the fine step; the provider closed form is exact "
                             "to rounding"), tolerance=TOL_SMALL))
        identity = _provider_identity(csg)
    elif csg is not None:
        state = "partial"
        record, sentence = _refusal_record(ctx, csg)
        findings.append(record)
        notes.append(sentence)
    else:
        notes.append("The optional CSG focus-event comparison did not run (bind --provider csg=<checkout>)")
    seeded_bump = sum(1 for r in bump_rows if r["origin"] == "seeded" and r["conjugate"])
    chord_bump = [r["comparison"]["first_positive_curvature_s"] for r in chords]
    fields = dict(
        hypothesis=("Zeros of the heading column are conjugate points and zeros of the lateral column are focal "
                    "points; on constant K > 0 they sit at multiples of pi/sqrt(K) and half-way between, none exist "
                    "where K <= 0, and on variable curvature none occurs before pi/sqrt(max K) and each lies "
                    "between the first zeros of the comparison equations for the upper and lower curvature "
                    "envelopes along the geodesic."),
        mathematical_model=("j_head = sn_K, j_lat = cn_K on constant K; Sturm comparison: K <= K_max gives no "
                            "conjugate point before pi/sqrt(K_max), and K <= 0 gives j_head >= s, j_lat >= 1; the "
                            "Pruefer angle theta = atan2(j, j') obeys theta' = cos^2 theta + K sin^2 theta, which "
                            "increases with K, so K_lo(s) <= K(s) <= K_hi(s) gives theta_lo <= theta <= theta_hi."),
        input_data=[f"Sphere great circles (R = 1, L = 7; R = 2, L = 14); torus outer equator (L = 11.5, K = 1/3); "
                    f"inner equator and hyperbolic plane; {STURM_TORUS} torus and {STURM_BUMP} bump seeded geodesics "
                    f"(seed {gj.SEED + 8}); {len(BUMP_CHORDS)} bump chords from {[list(c) for c in BUMP_CHORDS]} "
                    f"heading 0, L = {BUMP_CHORD_LENGTH}",
                    f"Curvature envelopes on fixed-step RK4 copies of those geodesics (step <= {ENVELOPE_STEP}, "
                    f"coordinate allowance {PATH_ALLOWANCE}); negative control integrated with {CONTROL_SCALE} K"],
        observation_model=("Cubic-Hermite zeros of sampled columns (ciw.lab.jacobi.Transfer.conjugate_points/"
                           "focal_points); unwrapped Pruefer angle of the adaptive heading column against the exact "
                           "piecewise-constant envelope solutions at the same arclengths."),
        expected_invariant=("Locations to RK4 + Hermite accuracy (order 4), zero counts exact, Sturm bounds "
                            "respected, and both bounds violated by the column integrated with 2K."),
        experiment=("Locate zeros at 40..320 steps, compare with pi multiples, check absence on K < 0, check the "
                    "global Sturm bound on the seeded torus geodesics, bracket the heading column between the "
                    "upper- and lower-envelope comparison solutions on every seeded and declared geodesic, repeat "
                    "both with the column integrated with 2K (which must be rejected), and compare with CSG when "
                    "bound."),
        numerical_result=(f"Sphere location error {_fmt(sphere['entries'][-1]['max_error'])} (order "
                          f"{_fmt(sphere['order'], 3)}); outer equator {_fmt(outer['entries'][-1]['max_error'])} "
                          f"(order {_fmt(outer['order'], 3)}); torus Sturm margin "
                          + ("none" if margin is None else _fmt(margin, 3))
                          + f" with {len(firsts)} of {STURM_TORUS} seeded torus geodesics reaching a conjugate point, "
                          f"2K-control margin {_fmt(control_margin, 3)}; envelope brackets hold to "
                          f"{_fmt(min(comparison['lower_gap'], comparison['upper_gap']))} rad and reject the 2K "
                          f"control by at least {_fmt(-comparison['control_gap'])} rad; "
                          f"{sum(1 for r in chords if r['conjugate'])} of {len(chords)} bump chords and "
                          f"{seeded_bump} of {STURM_BUMP} seeded bump geodesics reach a conjugate point"
                          + ("" if witness is None else f"; witness focal {_fmt(witness['focal'][0], 5)} vs "
                             f"conjugate/2 {_fmt(witness['conjugate'][0] / 2, 5)}") + "."),
        uncertainty=("Location error below 1e-7 at the finest fixed steps; adaptive zero locations move by at most "
                     f"{_fmt(located_change)} between rtol 1e-9 and 1e-10; the envelopes are widened by "
                     f"{PATH_ALLOWANCE} in the curvature coordinate for the error of the fixed-step path."),
        failure_modes_checked=["trivial zero of j_head at s = 0 excluded (also from the Sturm lower bounds)",
                               "zero counts, not only locations",
                               "negative curvature (no zeros) and Sturm lower bounds",
                               "vacuous Sturm bound: the torus bound must be reached by at least one geodesic and "
                               "must reject a column integrated with 2K; the two-sided envelope comparison must "
                               "reject the same control on every path",
                               "focal/conjugate relation on variable curvature (counterexample)"],
        unresolved_assumptions=notes + [
            f"The global bound pi/sqrt(max K) = {_fmt(bump_bound, 4)} is not exercised on the gaussian bump and is "
            "not claimed there: K > 0 only for rho < sigma, the declared chords first meet K > 0 at s = "
            + ", ".join(_fmt(v, 3) for v in chord_bump if v is not None)
            + ", beyond the bound, so it would hold whatever the Jacobi integration did; the bump is covered by the "
            "two-sided envelope comparison instead",
            "The envelope comparison assumes K(rho) on the bump decreases up to its flank minimum and increases "
            "beyond (checked on a grid, not proved)"],
        recommended_next_task=("Deferred research question: prove the monotonicity the bump's envelope comparison "
                               "assumes (K(rho) decreasing up to its flank minimum and increasing beyond; now checked "
                               "on a grid), for example by a sympy sign analysis of dK/drho, and exercise the global "
                               "bound pi/sqrt(max K) on a variable-curvature path that meets K > 0 before the bound"))
    if identity:
        fields["provider_runtime_identity"] = identity
    return {"state": state, "fields": fields, "findings": findings}


# ---------------------------------------------------------------------------
# T009: lateral versus heading columns
# ---------------------------------------------------------------------------
# The Jacobi columns are intrinsic, so the polar charts (flat paths re-expressed in another chart) are left out:
# they would repeat flat rows (|j_lat| = 1, |j_head| = L) already represented by the plane and the cylinder.
COLUMN_PATHS = tuple(k for k in gj.STANDARD if k not in gj.POLAR_KEYS) + (
    "sphere-great-circle", "hyperbolic-long", "torus-outer-equator", "torus-inner-equator", "bump-radial",
    "torus-outer-to-inner", "torus-inner-to-outer")
WITNESS_PAIR = ("torus-outer-to-inner", "torus-inner-to-outer")
WITNESS_SHORT = {"torus-outer-to-inner": "out>in", "torus-inner-to-outer": "in>out"}
REVERSAL_PATHS = WITNESS_PAIR + ("gaussian-bump", "saddle")
CONFIRM_EPS = 1e-3
# First-order kernel experiment: a small Gaussian curvature bump on a flat background of length 3.
KERNEL_LENGTH, KERNEL_AMPLITUDE, KERNEL_WIDTH = 3.0, 1e-3, 0.3
KERNEL_CENTERS = (0.5, 1.5, 2.5)
KERNEL_STEPS = 600


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


def kernel_study() -> dict:
    """Endpoint response of both columns to a small curvature bump at early, middle and late arclength.

    Direct: RK4 on j'' + dK(s) j = 0 (flat background). First order:
    delta j(L) = -int (L - s) j0(s) dK(s) ds with j0 = 1 (lateral) or s
    (heading), i.e. the kernels sn(L - s) cn(s) and sn(L - s) sn(s) at K = 0.
    """
    length = KERNEL_LENGTH
    nodes = np.linspace(0.0, length, 3001)
    out = {}
    for center in KERNEL_CENTERS:
        def bump(s, c=center):
            return KERNEL_AMPLITUDE * math.exp(-((s - c) / KERNEL_WIDTH) ** 2)

        def f(y, bump=bump):
            k = bump(y[0])
            return np.array([1.0, y[2], -k * y[1], y[4], -k * y[3]])

        _, states = integrators.integrate_fixed(f, np.array([0.0, 1.0, 0.0, 0.0, 1.0]), length, KERNEL_STEPS, "rk4")
        dk = np.array([bump(s) for s in nodes])
        weights = np.full(len(nodes), nodes[1] - nodes[0])
        weights[[0, -1]] /= 2  # trapezoid rule
        out[str(center)] = {"direct": {"lateral": float(states[-1, 1] - 1.0), "heading": float(states[-1, 3] - length)},
                            "first_order": {"lateral": float(-np.sum(weights * (length - nodes) * dk)),
                                            "heading": float(-np.sum(weights * (length - nodes) * nodes * dk))}}
    return out


def reversal_study(ctx, key) -> dict:
    """Transfer matrix of the reversed geodesic against the reciprocity prediction D Phi(L)^-1 D, D = diag(1, -1).

    The reversed path starts at the forward end point with the velocity
    negated, so it sees the curvature profile K(L - s).
    """
    spec = gj.path(key)
    surf = gj.surface(spec.surface)
    forward = _fine_transfer(ctx, key)
    end = forward.states[-1]
    start = np.concatenate([end[:2], -end[2:4], [1.0, 0.0, 0.0, 1.0]])
    _, states = integrators.integrate_fixed(jacobi.rhs(surf), start, spec.length, _fine(key), "rk4")
    reversed_phi = np.array([[states[-1, 4], states[-1, 6]], [states[-1, 5], states[-1, 7]]])
    flip = np.diag([1.0, -1.0])
    predicted = flip @ np.linalg.inv(forward.matrix()) @ flip
    return {"forward": forward.matrix().tolist(), "reversed": reversed_phi.tolist(),
            "prediction_error": float(np.max(np.abs(reversed_phi - predicted))),
            "heading_change": float(abs(reversed_phi[0, 1] - forward.matrix()[0, 1])),
            "return_gap": float(np.linalg.norm(states[-1, :2] - np.asarray(spec.u0)))}


@task("T009", changed_files=CHANGED,
      regression_tests=(_test("test_t009_columns_rank_paths_differently"),) + SECTION_TESTS)
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
        cn, _, sn, _ = jacobi.constant_curvature(_constant_curvature_of(key), [gj.path(key).length])
        model[key] = {"ratio_head_over_lat": rows[key]["j_head_end"] / rows[key]["j_lat_end"],
                      "model_ratio": float(sn[0] / cn[0])}
    kernels = kernel_study()
    kernel_error = max(abs(v["direct"][c] / v["first_order"][c] - 1) for v in kernels.values()
                       for c in ("lateral", "heading"))
    early, middle, late = (kernels[str(c)]["direct"] for c in KERNEL_CENTERS)
    heading_asymmetry = abs(early["heading"] - late["heading"]) / abs(middle["heading"])
    lateral_ratio = early["lateral"] / late["lateral"]
    reversals = {key: reversal_study(ctx, key) for key in REVERSAL_PATHS}
    # Witness-path kernels from the integrated columns: G(L, s) = j_lat(s) j_head(L) - j_head(s) j_lat(L).
    witness_kernels = {}
    for key in WITNESS_PAIR:
        tr = _fine_transfer(ctx, key)
        green = tr.states[:, 4] * tr.states[-1, 6] - tr.states[:, 6] * tr.states[-1, 4]
        stride = max(1, len(tr.s) // 50)
        witness_kernels[key] = {"s": tr.s[::stride].tolist(), "lateral": (green * tr.states[:, 4])[::stride].tolist(),
                                "heading": (green * tr.states[:, 6])[::stride].tolist(),
                                "curvature": tr.curvature_along()[::stride].tolist()}
    rank_order = sorted(COLUMN_PATHS, key=lambda k: lateral[k])
    ctx.artifact_json("columns.json", _plain({"rows": rows, "kendall_tau": tau, "discordant_pairs": discordant,
                                              "rank_order_by_lateral": rank_order,
                                              "witness_pair": list(WITNESS_PAIR), "fd_confirmation": confirm,
                                              "fd_eps": CONFIRM_EPS, "model_ratios": model,
                                              "kernel_experiment": {"length": KERNEL_LENGTH,
                                                                    "amplitude": KERNEL_AMPLITUDE,
                                                                    "width": KERNEL_WIDTH, "responses": kernels},
                                              "reversal": reversals, "witness_kernels": witness_kernels}))
    ctx.artifact_text("ranking.svg", svg.line_plot(
        [("|j_lat(L)|", range(1, len(rank_order) + 1), [lateral[k] for k in rank_order]),
         ("|j_head(L)|", range(1, len(rank_order) + 1), [heading[k] for k in rank_order])],
        title="Paths ordered by |j_lat(L)| (order: columns.json)", xlabel="rank by |j_lat(L)| (rank_order_by_lateral)",
        ylabel="endpoint sensitivity", logy=True))
    series = []
    for key in WITNESS_PAIR:
        tr = _fine_transfer(ctx, key)
        series += [(f"{WITNESS_SHORT[key]} lat", tr.s, tr.states[:, 4]), (f"{WITNESS_SHORT[key]} head", tr.s,
                                                                         tr.states[:, 6])]
    ctx.artifact_text("witness-columns.svg", svg.line_plot(
        series, title="Witness pair out>in, in>out (torus, L = 3): columns along s", xlabel="arclength s",
        ylabel="j", markers=False))
    s = np.linspace(0.0, KERNEL_LENGTH, 61)
    kernel_series = [("flat lat (L-s)", s, KERNEL_LENGTH - s), ("flat head s(L-s)", s, s * (KERNEL_LENGTH - s))]
    for key in WITNESS_PAIR:
        w = witness_kernels[key]
        kernel_series += [(f"{WITNESS_SHORT[key]} lat", w["s"], w["lateral"]),
                          (f"{WITNESS_SHORT[key]} head", w["s"], w["heading"])]
    ctx.artifact_text("kernels.svg", svg.line_plot(
        kernel_series, title="First-order weight of curvature at s on j(L)", xlabel="arclength s",
        ylabel="G(L, s) j(s)", markers=False))

    generator = {"name": "declared geodesics", "paths": list(COLUMN_PATHS)}
    fd_unc = _unc("truncation_bound", confirm_error, "central-difference confirmation (eps 1e-3) of the witness "
                  "sensitivities; columns carry RK4 error near 1e-8")
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
        uncertainty=fd_unc, tolerance=TOL_VALUE)]
    findings.append(finding(
        "Lateral and heading sensitivities rank paths differently", "numerical",
        {"kendall_tau": tau, "discordant_pairs": len(discordant), "witness_reversal_margin": reversal},
        {"generator": generator, "checks": [
            gj.check("invariant", f"reversal margin for the equal-length pair {a} / {b}", reversal, 0.5, "signed_ge"),
            gj.check("invariant", "discordant path pairs", len(discordant), 1.0, "ge")]},
        uncertainty=fd_unc, tolerance={"abs": 1e-6, "rel": 1e-6},
        counterexample={"statement": "Ranking paths by sensitivity to heading error gives the same order as ranking "
                                     "by sensitivity to lateral offset",
                        "witness": {"paths": list(WITNESS_PAIR), "length": gj.path(a).length,
                                    "lateral": {a: lateral[a], b: lateral[b]},
                                    "heading": {a: heading[a], b: heading[b]}}}))
    findings.append(finding(
        "On a flat background, to first order, curvature at arclength s moves j_lat(L) with weight L - s "
        "(early-weighted) and j_head(L) with weight s(L - s) (symmetric about mid-path)", "mathematical",
        {"responses": {c: v["direct"] for c, v in kernels.items()}, "first_order_relative_error": kernel_error,
         "heading_early_late_asymmetry": heading_asymmetry, "lateral_early_over_late": lateral_ratio},
        {"derivation": f"{DOC}, section T009 (variation of j'' + K j = 0 with its Green function)",
         "generator": {"name": "curvature bump on a flat background", "length": KERNEL_LENGTH,
                       "amplitude": KERNEL_AMPLITUDE, "width": KERNEL_WIDTH, "centers": list(KERNEL_CENTERS),
                       "steps": KERNEL_STEPS}, "checks": [
            gj.check("analytic", "direct RK4 response against the first-order kernel integral (relative)",
                     kernel_error, 1e-3),
            gj.check("analytic", "heading response to a bump at s = 0.5 versus s = 2.5, over the mid-path response",
                     heading_asymmetry, 1e-6, "le"),
            gj.check("analytic", "lateral response to a bump at s = 0.5 over s = 2.5 (first order about 4.9)",
                     lateral_ratio, 3.0, "ge")]},
        uncertainty=_unc("truncation_bound", kernel_error, "second-order terms in the bump amplitude 1e-3"),
        tolerance={"abs": 1e-9, "rel": 1e-6}))
    reversal_error = max(v["prediction_error"] for v in reversals.values())
    findings.append(finding(
        "Reversing a geodesic leaves j_head(L) unchanged and exchanges j_lat(L) with j_head'(L) (transfer matrix "
        "D Phi(L)^-1 D)", "numerical",
        {k: {"prediction_error": v["prediction_error"], "heading_change": v["heading_change"]}
         for k, v in reversals.items()},
        {"derivation": f"{DOC}, section T009 (reciprocity of j'' + K(s) j = 0 under s -> L - s)",
         "generator": {"name": "declared geodesics and their reversals", "paths": list(REVERSAL_PATHS)}, "checks": [
            gj.check("analytic", f"reversed transfer matrix against D Phi^-1 D on {k}", v["prediction_error"], 1e-8)
            for k, v in reversals.items()] + [
            gj.check("invariant", "reversed path returns to the start",
                     max(v["return_gap"] for v in reversals.values()), 1e-8)]},
        uncertainty=_unc("truncation_bound", reversal_error, "RK4 error of forward and reversed integrations"),
        tolerance=TOL_SMALL))
    findings.append(finding(
        "The ranking of lateral versus heading start errors computed here predicts which error dominates the "
        "endpoint error of real tool or vehicle paths on physical curved parts", "physical", None, {}))
    return _outcome(
        "completed", findings,
        hypothesis=("On a constant-K background a curvature change at arclength s moves j_lat(L) with weight "
                    "sn(L-s) cn(s), which decreases along the path when K <= 0 or sqrt(K) L <= pi/2 (before the "
                    "first focal distance), and j_head(L) with weight sn(L-s) sn(s), which is symmetric about "
                    "mid-path for every K and vanishes at both ends; exactly, reversing the curvature profile leaves "
                    "j_head(L) unchanged. The two columns therefore respond differently to where curvature sits "
                    "along a path and can order paths differently."),
        mathematical_model=("Endpoint normal displacement = j_lat(L) delta_perp + j_head(L) delta_alpha; "
                            "delta j(L) = -int G(L, s) j(s) delta K(s) ds with G(L, s) = j_lat(s) j_head(L) - "
                            "j_head(s) j_lat(L); reversal maps Phi(L) to D Phi(L)^-1 D; model spaces give "
                            "j_head/j_lat = tan(sqrt(K)L)/sqrt(K), L, tanh(sqrt(-K)L)/sqrt(-K)."),
        input_data=[f"{len(COLUMN_PATHS)} declared paths: {', '.join(COLUMN_PATHS)}",
                    f"Witness pair {a} / {b}: same torus, same length 3, curvature order reversed but not mirror "
                    "profiles", f"Curvature bump of amplitude {KERNEL_AMPLITUDE}, width {KERNEL_WIDTH} at s in "
                    f"{list(KERNEL_CENTERS)} on a flat path of length {KERNEL_LENGTH}"],
        observation_model=("|j(L)| per unit perturbation; confirmed on the witness pair by central differences of "
                           f"perturbed geodesics (eps = {CONFIRM_EPS}); reversed paths start at the end point with "
                           "negated velocity."),
        expected_invariant=("det Phi(L) = 1; model-space ratios; first-order kernels; reversal reciprocity; a ranking "
                            "reversal on the witness pair."),
        experiment=("Integrate both columns on every path, rank paths by each column, count discordant pairs "
                    "(Kendall tau), confirm the witness numbers by finite differences, measure the response to a "
                    "curvature bump at three positions, and integrate reversed paths."),
        numerical_result=(f"Kendall tau {_fmt(tau, 3)} with {len(discordant)} discordant pairs; witness lateral "
                          f"{_fmt(lateral[a], 4)} vs {_fmt(lateral[b], 4)}, heading {_fmt(heading[a], 4)} vs "
                          f"{_fmt(heading[b], 4)}; finite-difference confirmation error {_fmt(confirm_error)}; bump "
                          f"response lateral early/late {_fmt(lateral_ratio, 3)}, heading early/late asymmetry "
                          f"{_fmt(heading_asymmetry)}; reversal reciprocity error {_fmt(reversal_error)}."),
        uncertainty="Column values carry RK4 error near 1e-8; the reversal margin exceeds it by eight orders.",
        failure_modes_checked=["sign of the columns (magnitudes ranked)", "equal-length comparison for the witness",
                               "finite-difference confirmation of the endpoint sensitivities",
                               "mechanism: heading is not late-weighted (bump symmetry and exact reversal "
                               "reciprocity)"],
        unresolved_assumptions=["Rankings are over declared paths of unequal lengths except the witness pair",
                                "The polar charts are excluded from the ranking: they re-express flat paths, and the "
                                "Jacobi columns do not depend on the chart",
                                "The witness heading difference comes from the two curvature profiles not being "
                                "mirror images; the first-order kernels explain it only qualitatively",
                                "No physical platform, tool or perturbation statistics were measured"],
        recommended_next_task=("Deferred research question: rank lateral against heading sensitivity over "
                               "equal-length paths (only the witness pair has equal lengths now) and predict the "
                               "witness pair's heading difference quantitatively from its two curvature profiles "
                               "through the first-order kernels, which now explain it only qualitatively"))
