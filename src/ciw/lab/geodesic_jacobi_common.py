"""Shared helpers for the geodesic/Jacobi lab experiments.

Scope: declared test geodesics on the ``ciw.lab.surfaces`` catalogue, their
closed-form references where one exists, embedded and chart distances, a
flat polar chart that gives the flat surfaces nonzero Christoffel symbols,
symbolic derivations with sympy (optional), arbitrary-precision references
with mpmath (optional), and check builders whose ``passed`` flag is always
computed from the recorded numbers.

Non-claims: every surface, length and curvature is in declared normalized
units. Nothing here represents, measures or calibrates a physical surface, and
agreement between computations is not agreement with the physical world.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math

import numpy as np

from .. import __version__
from . import integrators, jacobi
from .surfaces import ChartMap, Cylinder, Plane, Reparametrized, catalogue

SEED = 20260923

# The catalogue order is the reporting order of every per-surface table.
CATALOGUE_KEYS = ("plane", "sphere", "cylinder", "saddle", "torus", "gaussian-bump", "hyperbolic-plane")
POLAR_KEYS = ("plane-polar", "cylinder-polar")
VARIABLE_KEYS = ("saddle", "torus", "gaussian-bump")
CLOSED_FORM_KEYS = ("plane", "sphere", "cylinder", "hyperbolic-plane", "plane-polar", "cylinder-polar")


class PolarChart(ChartMap):
    """a = (r, t) -> u = (r cos t, r sin t); singular only at r = 0."""

    name = "polar"

    def forward(self, a):
        r, t = a
        return np.array([r * math.cos(t), r * math.sin(t)])

    def jacobian(self, a):
        r, t = a
        return np.array([[math.cos(t), -r * math.sin(t)], [math.sin(t), r * math.cos(t)]])

    def hessian(self, a):
        r, t = a
        c, s = math.cos(t), math.sin(t)
        # hessian[p, i, k] = d^2 phi^p / (da^i da^k)
        return np.array([[[0.0, -s], [-s, -r * c]], [[0.0, c], [c, -r * s]]])

    def inverse(self, u):
        return np.array([math.hypot(u[0], u[1]), math.atan2(u[1], u[0])])


_SURFACES: dict = {}


def surface(key: str):
    """One shared instance per catalogue key (surfaces are immutable after construction)."""
    if not _SURFACES:
        _SURFACES.update(catalogue())
        _SURFACES["plane-polar"] = Reparametrized(Plane(), PolarChart())
        _SURFACES["cylinder-polar"] = Reparametrized(Cylinder(1.0), PolarChart())
    if key not in _SURFACES:
        raise KeyError(f"Unknown lab surface: {key}")
    return _SURFACES[key]


def describe(key: str) -> dict:
    return dict(surface(key).describe(), key=key)


@dataclass(frozen=True)
class PathSpec:
    """A declared unit-speed geodesic: start chart point, heading from the first coordinate, length."""

    key: str
    surface: str
    u0: tuple
    heading: float
    length: float
    note: str = ""

    def as_dict(self) -> dict:
        return {"key": self.key, "surface": self.surface, "u0": list(self.u0), "heading": self.heading,
                "length": self.length, "note": self.note}


# Standard paths stay inside their charts: the sphere path keeps theta in
# [0.77, 1.9] (Clairaut), the polar paths keep r >= 1.39, the hyperbolic path
# keeps y > 0.5.
STANDARD = {
    "plane": PathSpec("plane", "plane", (0.3, -0.2), 0.7, 2.0),
    "sphere": PathSpec("sphere", "sphere", (1.1, 0.4), 0.9, 2.0, "inclined great-circle arc"),
    "cylinder": PathSpec("cylinder", "cylinder", (0.2, 0.1), 0.6, 3.0, "helix"),
    "saddle": PathSpec("saddle", "saddle", (0.1, -0.2), 0.8, 1.5),
    "torus": PathSpec("torus", "torus", (0.0, 0.5), 0.7, 3.0, "crosses the K > 0 and K < 0 regions"),
    "gaussian-bump": PathSpec("gaussian-bump", "gaussian-bump", (-1.2, 0.3), 0.2, 2.5, "passes over the bump"),
    "hyperbolic-plane": PathSpec("hyperbolic-plane", "hyperbolic-plane", (0.0, 1.0), 0.6, 1.5, "semicircle arc"),
    "plane-polar": PathSpec("plane-polar", "plane-polar", (1.5, 0.4), 1.2, 2.0, "straight line seen in polar chart"),
    "cylinder-polar": PathSpec("cylinder-polar", "cylinder-polar", (1.5, 0.4), 1.2, 2.0,
                               "helix seen in a polar chart of the (phi, z) development"),
}

# Paths along which the curvature is constant by construction.
SPECIAL = {
    "sphere-great-circle": PathSpec("sphere-great-circle", "sphere", (math.pi / 2, 0.0), 1.0, 7.0,
                                    "inclined great circle through (pi/2, 0); theta stays in [1.0, 2.14]"),
    "torus-outer-equator": PathSpec("torus-outer-equator", "torus", (0.0, 0.0), 0.0, 11.5,
                                    "theta = 0 parallel; K = 1/(r(R+r))"),
    "torus-inner-equator": PathSpec("torus-inner-equator", "torus", (0.0, math.pi), 0.0, 6.5,
                                    "theta = pi parallel; K = -1/(r(R-r))"),
    "bump-radial": PathSpec("bump-radial", "gaussian-bump", (0.0, 0.0), 0.0, 3.0,
                            "radial geodesic from the summit: K > 0 first, then K < 0"),
}


def path(key: str) -> PathSpec:
    return STANDARD[key] if key in STANDARD else SPECIAL[key]


def transfer(ctx, key: str, method: str = "rk4", steps: int | None = None, rtol: float | None = None,
             atol: float | None = None):
    """Memoized joint geodesic + Jacobi integration of a declared path."""
    spec = path(key)

    def compute():
        s = surface(spec.surface)
        if rtol is not None:
            return jacobi.transfer(s, spec.u0, spec.heading, spec.length, rtol=rtol,
                                   atol=rtol * 1e-2 if atol is None else atol)
        return jacobi.transfer(s, spec.u0, spec.heading, spec.length, steps=steps, method=method)

    return ctx.memo(("gj-transfer", key, method, steps, rtol, atol), compute)


# Geometry of positions ---------------------------------------------------
def position(key: str, u) -> np.ndarray:
    """Embedded point for embedded surfaces, chart point for the hyperbolic plane."""
    s = surface(key)
    point = s.embedding(np.asarray(u, dtype=float))
    return np.asarray(u, dtype=float) if point is None else np.asarray(point, dtype=float)


def distance(key: str, a, b) -> float:
    """Euclidean distance in R^3 (embedded) or in the chart (hyperbolic plane)."""
    return float(np.linalg.norm(position(key, a) - position(key, b)))


def exact_position(key: str, s_values) -> np.ndarray | None:
    """Closed-form geodesic positions (as returned by :func:`position`), or None."""
    spec = path(key)
    surf = surface(spec.surface)
    u0 = np.asarray(spec.u0, dtype=float)
    t0 = surf.unit_tangent(u0, spec.heading)
    s_values = np.atleast_1d(np.asarray(s_values, dtype=float))
    name = spec.surface
    if name == "sphere":
        return surf.exact_embedded_geodesic(u0, t0, s_values)
    if name in ("plane", "cylinder", "hyperbolic-plane"):
        chart = surf.exact_geodesic(u0, t0, s_values)
        return np.array([position(name, u) for u in chart])
    if name in POLAR_KEYS:
        base, chart = surf.base, surf.chart
        base_u0, base_t0 = chart.forward(u0), chart.jacobian(u0) @ t0
        return np.array([base.embedding(base_u0 + s * base_t0) for s in s_values])
    return None


# Checks whose pass flag is derived from their numbers ----------------------
def check(kind: str, reference: str, observed: float, tolerance: float, comparison: str = "abs_le") -> dict:
    observed, tolerance = float(observed), float(tolerance)
    if comparison == "abs_le":
        passed = abs(observed) <= tolerance
    elif comparison == "le":
        passed = observed <= tolerance
    elif comparison == "ge":
        passed = observed >= tolerance
    else:
        raise ValueError(f"Unsupported comparison: {comparison}")
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": bool(passed)}


def independent(base: dict, producer: str, checker: str, producer_revision: str | None = None,
                checker_revision: str | None = None) -> dict:
    return dict(base, producer={"implementation": producer, "revision": producer_revision or f"ciw {__version__}"},
                checker={"implementation": checker, "revision": checker_revision or "unversioned"})


def optional_version(name: str) -> str | None:
    """Version of an optional module when it is importable, else None (imports lazily)."""
    if importlib.util.find_spec(name) is None:
        return None
    return str(__import__(name).__version__)


def slope(xs, ys) -> float:
    """Least-squares log-log slope; refuses nonpositive data rather than dropping it."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.any(xs <= 0) or np.any(ys <= 0) or not np.all(np.isfinite(ys)):
        raise ValueError("Log-log slopes need positive finite data")
    return integrators.observed_order(xs, ys)


def rounded(value, digits: int = 12):
    """JSON-friendly nested floats rounded to significant digits (artifact tables only)."""
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if value == 0 or not math.isfinite(value) else float(f"{value:.{digits}g}")
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# Symbolic derivations (sympy, optional) -----------------------------------
def symbolic_model(key: str, sp):
    """Coordinates and embedding (or intrinsic metric) of a surface from its declared parameters."""
    surf = surface(key)
    exact = sp.nsimplify
    if key == "hyperbolic-plane":
        u = sp.Symbol("u", real=True)
        v = sp.Symbol("v", positive=True)
        k = exact(surf.k)
        return (u, v), None, sp.eye(2) / (k ** 2 * v ** 2)
    u, v = sp.symbols("u v", real=True)
    if key == "plane":
        X = sp.Matrix([u, v, 0])
    elif key == "sphere":
        R = exact(surf.radius)
        X = R * sp.Matrix([sp.sin(u) * sp.cos(v), sp.sin(u) * sp.sin(v), sp.cos(u)])
    elif key == "cylinder":
        R = exact(surf.radius)
        X = sp.Matrix([R * sp.cos(u), R * sp.sin(u), v])
    elif key == "saddle":
        X = sp.Matrix([u, v, exact(surf.c) * (u ** 2 - v ** 2) / 2])
    elif key == "torus":
        a, b = exact(surf.major), exact(surf.minor)
        X = sp.Matrix([(a + b * sp.cos(v)) * sp.cos(u), (a + b * sp.cos(v)) * sp.sin(u), b * sp.sin(v)])
    elif key == "gaussian-bump":
        h, sigma = exact(surf.h), exact(surf.sigma)
        X = sp.Matrix([u, v, h * sp.exp(-(u ** 2 + v ** 2) / (2 * sigma ** 2))])
    elif key == "plane-polar":
        X = sp.Matrix([u * sp.cos(v), u * sp.sin(v), 0])
    elif key == "cylinder-polar":
        R = exact(surf.base.radius)
        X = sp.Matrix([R * sp.cos(u * sp.cos(v)), R * sp.sin(u * sp.cos(v)), u * sp.sin(v)])
    else:
        raise KeyError(f"No symbolic model for {key}")
    jac = X.jacobian([u, v])
    return (u, v), X, sp.simplify(jac.T * jac)


def derive(key: str) -> dict:
    """Metric, Christoffel symbols, geodesic equations and Brioschi curvature, all symbolic.

    The curvature uses the Brioschi formula (first fundamental form only), so
    it is an intrinsic route distinct from the second-fundamental-form
    curvature that ``ciw.lab.surfaces`` evaluates for embedded surfaces.
    """
    import sympy as sp

    (u, v), X, g = symbolic_model(key, sp)
    coords = (u, v)
    ginv = sp.simplify(g.inv())
    gamma = [[[sp.simplify(sum(ginv[k, l] * (sp.diff(g[j, l], coords[i]) + sp.diff(g[i, l], coords[j])
                                               - sp.diff(g[i, j], coords[l])) for l in range(2)) / 2)
               for j in range(2)] for i in range(2)] for k in range(2)]
    du, dv = sp.symbols("du dv", real=True)
    velocity = (du, dv)
    acceleration = [sp.expand(-sum(gamma[k][i][j] * velocity[i] * velocity[j] for i in range(2) for j in range(2)))
                    for k in range(2)]
    E, F, G = g[0, 0], g[0, 1], g[1, 1]
    d = sp.diff
    m1 = sp.Matrix([[-d(E, v, 2) / 2 + d(F, u, v) - d(G, u, 2) / 2, d(E, u) / 2, d(F, u) - d(E, v) / 2],
                    [d(F, v) - d(G, u) / 2, E, F], [d(G, v) / 2, F, G]])
    m2 = sp.Matrix([[0, d(E, v) / 2, d(G, u) / 2], [d(E, v) / 2, E, F], [d(G, u) / 2, F, G]])
    curvature = sp.simplify((m1.det() - m2.det()) / (E * G - F ** 2) ** 2)
    return {"sympy": sp, "coords": coords, "velocity": velocity, "embedding": X, "metric": g,
            "christoffel": gamma, "acceleration": acceleration, "curvature": curvature}


def lambdified(derived: dict, modules="math") -> dict:
    """Numeric callables from a derivation: metric(u, v), christoffel(u, v), rhs(u, v, du, dv), K(u, v)."""
    sp = derived["sympy"]
    u, v = derived["coords"]
    du, dv = derived["velocity"]
    g, gamma = derived["metric"], derived["christoffel"]
    flat_gamma = [gamma[k][i][j] for k in range(2) for i in range(2) for j in range(2)]
    return {
        "metric": sp.lambdify((u, v), [g[0, 0], g[0, 1], g[1, 0], g[1, 1]], modules=modules),
        "christoffel": sp.lambdify((u, v), flat_gamma, modules=modules),
        "rhs": sp.lambdify((u, v, du, dv), [du, dv, *derived["acceleration"]], modules=modules, cse=True),
        "curvature": sp.lambdify((u, v), derived["curvature"], modules=modules),
    }


def latex_block(key: str, derived: dict) -> str:
    """LaTeX for the derived metric, nonzero Christoffel symbols, geodesic equations and curvature."""
    sp = derived["sympy"]
    names = {derived["coords"][0]: "u", derived["coords"][1]: "v", derived["velocity"][0]: r"\dot{u}",
             derived["velocity"][1]: r"\dot{v}"}
    idx = ("u", "v")
    lines = [f"% {key}: coordinates (u, v) = {describe(key).get('chart', '(u, v)')}",
             r"g = " + sp.latex(derived["metric"], symbol_names=names)]
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                value = derived["christoffel"][k][i][j]
                if value != 0:
                    lines.append(rf"\Gamma^{{{idx[k]}}}_{{{idx[i]}{idx[j]}}} = " + sp.latex(value, symbol_names=names))
    for k in range(2):
        lines.append(rf"\ddot{{{idx[k]}}} = " + sp.latex(derived["acceleration"][k], symbol_names=names))
    lines.append("K = " + sp.latex(derived["curvature"], symbol_names=names))
    return "\n".join(lines) + "\n"


def text_block(key: str, derived: dict) -> str:
    sp = derived["sympy"]
    idx = ("u", "v")
    lines = [f"[{key}] {describe(key)}"]
    if derived["embedding"] is not None:
        lines.append(f"  X(u, v) = {sp.sstr(list(derived['embedding']))}")
    g = derived["metric"]
    lines.append(f"  g = [[{sp.sstr(g[0, 0])}, {sp.sstr(g[0, 1])}], [{sp.sstr(g[1, 0])}, {sp.sstr(g[1, 1])}]]")
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                value = derived["christoffel"][k][i][j]
                if value != 0:
                    lines.append(f"  Gamma^{idx[k]}_{idx[i]}{idx[j]} = {sp.sstr(value)}")
    for k in range(2):
        lines.append(f"  {idx[k]}'' = {sp.sstr(derived['acceleration'][k])}")
    lines.append(f"  K (Brioschi) = {sp.sstr(derived['curvature'])}")
    return "\n".join(lines) + "\n"


# Arbitrary-precision references (mpmath, optional) ------------------------
def gbs_integrate(f, y0, length, macro_steps, sequence=(2, 4, 6, 8, 10, 12, 14, 16)):
    """Gragg-Bulirsch-Stoer extrapolated modified midpoint in the current mpmath precision.

    Returns the final state and the largest per-step difference between the
    two highest extrapolation orders (a local error indicator).
    """
    import mpmath

    y = [mpmath.mpf(x) for x in y0]
    big_h = mpmath.mpf(length) / macro_steps
    indicator = mpmath.mpf(0)
    for _ in range(macro_steps):
        table = []
        for j, n in enumerate(sequence):
            h = big_h / n
            z0 = y
            z1 = [a + h * b for a, b in zip(z0, f(z0))]
            for _ in range(1, n):
                z0, z1 = z1, [a + 2 * h * b for a, b in zip(z0, f(z1))]
            row = [[(a + b + h * c) / 2 for a, b, c in zip(z1, z0, f(z1))]]
            for k in range(1, j + 1):
                ratio = mpmath.mpf(sequence[j]) ** 2 / mpmath.mpf(sequence[j - k]) ** 2
                row.append([a + (a - b) / (ratio - 1) for a, b in zip(row[k - 1], table[j - 1][k - 1])])
            table.append(row)
        indicator = max(indicator, max(abs(a - b) for a, b in zip(table[-1][-1], table[-1][-2])))
        y = table[-1][-1]
    return y, indicator


def mp_speed_squared(derived: dict, state, dps: int):
    import mpmath

    sp = derived["sympy"]
    u, v = derived["coords"]
    with mpmath.workdps(dps):
        e, f, g = sp.lambdify((u, v), [derived["metric"][0, 0], derived["metric"][0, 1], derived["metric"][1, 1]],
                              modules="mpmath")(state[0], state[1])
        return e * state[2] ** 2 + 2 * f * state[2] * state[3] + g * state[3] ** 2
