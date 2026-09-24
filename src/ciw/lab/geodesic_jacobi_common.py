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
from .evidence import holds

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
# [0.77, 1.9] (Clairaut), the polar paths keep r >= 1.5 (plane-polar, moving
# outward) and r >= 1.61 (cylinder-polar, passing its closest approach to the
# pole), the hyperbolic path keeps y > 0.5. The two polar charts carry the same
# flat metric, so their paths differ: identical paths would integrate one ODE
# twice and count one data point twice in per-chart statistics.
STANDARD = {
    "plane": PathSpec("plane", "plane", (0.3, -0.2), 0.7, 2.0),
    "sphere": PathSpec("sphere", "sphere", (1.1, 0.4), 0.9, 2.0, "inclined great-circle arc"),
    "cylinder": PathSpec("cylinder", "cylinder", (0.2, 0.1), 0.6, 3.0, "helix"),
    "saddle": PathSpec("saddle", "saddle", (0.1, -0.2), 0.8, 1.5),
    "torus": PathSpec("torus", "torus", (0.0, 0.5), 0.7, 3.0, "crosses the K > 0 and K < 0 regions"),
    "gaussian-bump": PathSpec("gaussian-bump", "gaussian-bump", (-1.2, 0.3), 0.2, 2.5, "passes over the bump"),
    "hyperbolic-plane": PathSpec("hyperbolic-plane", "hyperbolic-plane", (0.0, 1.0), 0.6, 1.5, "semicircle arc"),
    "plane-polar": PathSpec("plane-polar", "plane-polar", (1.5, 0.4), 1.2, 2.0, "straight line seen in polar chart"),
    "cylinder-polar": PathSpec("cylinder-polar", "cylinder-polar", (2.0, -0.6), 2.2, 2.5,
                               "helix seen in a polar chart of the (phi, z) development, passing r = 1.62"),
}

# Paths along which the curvature is constant by construction.
SPECIAL = {
    "sphere-great-circle": PathSpec("sphere-great-circle", "sphere", (math.pi / 2, 0.0), 1.0, 7.0,
                                    "inclined great circle through (pi/2, 0); theta stays in [1.0, 2.14]"),
    "torus-outer-equator": PathSpec("torus-outer-equator", "torus", (0.0, 0.0), 0.0, 11.5,
                                    "theta = 0 parallel; K = 1/(r(R+r))"),
    "torus-inner-equator": PathSpec("torus-inner-equator", "torus", (0.0, math.pi), 0.0, 6.5,
                                    "theta = pi parallel; K = -1/(r(R-r))"),
    "hyperbolic-long": PathSpec("hyperbolic-long", "hyperbolic-plane", (0.0, 1.0), 0.6, 3.0,
                                "semicircle arc; K = -1"),
    "bump-radial": PathSpec("bump-radial", "gaussian-bump", (0.0, 0.0), 0.0, 3.0,
                            "radial geodesic from the summit: K > 0 first, then K < 0"),
    "torus-outer-to-inner": PathSpec("torus-outer-to-inner", "torus", (0.0, math.pi / 6), 1.2, 3.0,
                                     "starts where K > 0 and crosses K < 0 mid-path"),
    "torus-inner-to-outer": PathSpec("torus-inner-to-outer", "torus", (0.0, 5 * math.pi / 6), -0.8, 3.0,
                                     "starts where K < 0 and crosses K > 0 mid-path"),
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


# A fixed RK4 step of at most this arclength keeps Jacobi-column errors near
# 1e-9 on every declared path; shared so tasks reuse one memoized integration.
FINE_STEP = 0.015


def fine_steps(key: str) -> int:
    return max(100, int(math.ceil(path(key).length / FINE_STEP)))


def start_state(key: str) -> np.ndarray:
    """Binary64 geodesic + Jacobi initial state of a declared path (shared by every reference)."""
    spec = path(key)
    return jacobi.initial_state(surface(spec.surface), spec.u0, spec.heading)


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
    passed = holds(observed, tolerance, comparison)
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
    it is an intrinsic route. ``ciw.lab.surfaces`` returns closed-form
    curvatures for the catalogue; its generic second-fundamental-form route
    is ``EmbeddedSurface.gaussian_curvature``, which T001 calls unbound.
    """
    import sympy as sp

    (u, v), X, g = symbolic_model(key, sp)
    coords = (u, v)
    ginv = sp.simplify(g.inv())
    gamma = [[[sp.simplify(sum(ginv[k, q] * (sp.diff(g[j, q], coords[i]) + sp.diff(g[i, q], coords[j])
                                               - sp.diff(g[i, j], coords[q])) for q in range(2)) / 2)
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
    """LaTeX section with the derived metric, nonzero Christoffel symbols, geodesic equations and curvature.

    Every equation is in display math, so the blocks joined by
    :func:`latex_document` compile as one document.
    """
    sp = derived["sympy"]
    names = {derived["coords"][0]: "u", derived["coords"][1]: "v", derived["velocity"][0]: r"\dot{u}",
             derived["velocity"][1]: r"\dot{v}"}
    idx = ("u", "v")
    equations = [r"g = " + sp.latex(derived["metric"], symbol_names=names)]
    for k in range(2):
        for i in range(2):
            for j in range(i, 2):
                value = derived["christoffel"][k][i][j]
                if value != 0:
                    equations.append(rf"\Gamma^{{{idx[k]}}}_{{{idx[i]}{idx[j]}}} = "
                                     + sp.latex(value, symbol_names=names))
    for k in range(2):
        equations.append(rf"\ddot{{{idx[k]}}} = " + sp.latex(derived["acceleration"][k], symbol_names=names))
    equations.append("K = " + sp.latex(derived["curvature"], symbol_names=names))
    chart = str(describe(key).get("chart", "(u, v)")).replace("_", r"\_")
    lines = [rf"\section*{{{key}}}", rf"Coordinates $(u, v)$: \texttt{{{chart}}}."]
    lines += [rf"\[ {equation} \]" for equation in equations]
    return "\n".join(lines) + "\n"


def latex_document(blocks) -> str:
    """A minimal standalone LaTeX document around :func:`latex_block` sections."""
    return ("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
            + "\n".join(blocks) + "\\end{document}\n")


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


MP_DPS = 34
MP_MACRO_STEPS = (10, 20)


def mp_reference(key: str, dps: int = MP_DPS, macro_steps=MP_MACRO_STEPS, derived: dict | None = None) -> dict:
    """Arbitrary-precision geodesic + Jacobi end state of a declared path (needs sympy and mpmath).

    The equations are the sympy derivation (Christoffel symbols and Brioschi
    curvature), not ``ciw.lab.surfaces``; the start is the same binary64 state
    the ciw integrators use, so only the integration differs. The error
    estimate is the largest component difference between the two macro-step
    counts at ``dps`` digits. ``derived`` reuses a :func:`derive` of the
    path's surface.
    """
    import mpmath

    spec = path(key)
    derived = derive(spec.surface) if derived is None else derived
    functions = lambdified(derived, modules="mpmath")
    rhs, curvature = functions["rhs"], functions["curvature"]

    def f(y):
        k = curvature(y[0], y[1])
        return [*rhs(y[0], y[1], y[2], y[3]), y[5], -k * y[4], y[7], -k * y[6]]

    y0 = [float(x) for x in start_state(key)]
    with mpmath.workdps(dps):
        coarse, _ = gbs_integrate(f, y0, spec.length, macro_steps[0])
        fine, indicator = gbs_integrate(f, y0, spec.length, macro_steps[1])
        estimate = max(abs(a - b) for a, b in zip(coarse, fine))
        speed = mp_speed_squared(derived, fine, dps)
        return {"kind": "mpmath", "state": [float(x) for x in fine], "digits": [mpmath.nstr(x, 30) for x in fine],
                "error_estimate": float(estimate), "local_indicator": float(indicator),
                "speed_squared_minus_one": float(speed - 1), "dps": dps, "macro_steps": list(macro_steps),
                "method": "Gragg-Bulirsch-Stoer modified midpoint, sequence 2..16", "mpf_state": fine}


# mpmath.odefun takes each step's Taylor coefficients from finite differences of Euler steps carried at about
# (degree + 1) times the working precision, so a higher degree buys fewer steps at a steeper price per step. Its
# default at 34 digits is 3 + 3 * 34 // 2 = 54; degree 30 gives the same 34-digit end states on the three
# variable-curvature paths in about 12 s instead of 17 s (one core, mpmath's pure-Python backend).
ODEFUN_DEGREE = 30


def mp_system(derived: dict):
    """The sympy-derived geodesic + Jacobi system (8 states) as one mpmath callable f(y).

    The same expressions as :func:`mp_reference` integrates (Christoffel
    symbols and Brioschi curvature), with products of exponentials combined
    by sympy (``powsimp``) and lambdified jointly, so that the transcendental
    subexpressions the geodesic and Jacobi parts share are evaluated once.
    State order: u, v, u', v', j_lat, j_lat', j_head, j_head'.
    """
    sp = derived["sympy"]
    u, v = derived["coords"]
    du, dv = derived["velocity"]
    lat, lat_rate, head, head_rate = sp.symbols("j_lat j_lat_rate j_head j_head_rate", real=True)
    k = derived["curvature"]
    system = [du, dv, *derived["acceleration"], lat_rate, -k * lat, head_rate, -k * head]
    joint = sp.lambdify((u, v, du, dv, lat, lat_rate, head, head_rate),
                        [sp.powsimp(entry, combine="exp", deep=True) for entry in system], modules="mpmath", cse=True)
    return lambda y: joint(*y)


def odefun_reference(key: str, derived: dict | None = None, dps: int = MP_DPS, degree: int = ODEFUN_DEGREE,
                     length: float | None = None) -> dict:
    """Geodesic + Jacobi end state from mpmath's own Taylor-series solver (needs sympy and mpmath).

    ``mpmath.odefun`` integrates the sympy-derived equations of
    :func:`mp_system` from the same binary64 start as every other reference,
    so it differs from :func:`mp_reference` only in the integrator: mpmath's
    Taylor series, not the ciw-authored :func:`gbs_integrate`. It reports no
    error estimate. Two first integrals of the exact flow are returned as
    diagnostics: the drift of g(u', u') from its start value and det(Phi) - 1
    of the transfer matrix (1 at the start). ``length`` defaults to the path
    length.
    """
    import mpmath

    spec = path(key)
    derived = derive(spec.surface) if derived is None else derived
    system = mp_system(derived)
    evaluations = [0]

    def f(s, y):  # autonomous: the arclength odefun passes is not used
        evaluations[0] += 1
        return system(y)

    with mpmath.workdps(dps):
        start = [mpmath.mpf(float(x)) for x in start_state(key)]
        end = mpmath.odefun(f, 0, start, degree=degree)(spec.length if length is None else length)
        drift = mp_speed_squared(derived, end, dps) - mp_speed_squared(derived, start, dps)
        determinant = end[4] * end[7] - end[6] * end[5] - 1
        # odefun evaluates the system `degree` times per Taylor step, and nowhere else.
        return {"kind": "mpmath.odefun", "state": [float(x) for x in end], "digits": [mpmath.nstr(x, 30) for x in end],
                "speed_squared_drift": float(drift), "determinant_minus_one": float(determinant),
                "evaluations": evaluations[0], "taylor_steps": evaluations[0] // degree, "dps": dps,
                "degree": degree, "method": "mpmath.odefun Taylor series", "mpf_state": end}


def mp_difference(a, b, dps: int = MP_DPS) -> float:
    """Largest component difference of two arbitrary-precision states, evaluated at ``dps`` digits."""
    import mpmath

    with mpmath.workdps(dps):
        return float(max(abs(x - y) for x, y in zip(a, b, strict=True)))


def scipy_reference(key: str, rtol: float = 1e-13, atol: float = 1e-15) -> dict:
    """End state from scipy's DOP853 applied to the ciw geodesic + Jacobi right-hand side (needs scipy)."""
    from scipy.integrate import solve_ivp

    spec = path(key)
    f = jacobi.rhs(surface(spec.surface))
    solution = solve_ivp(lambda s, y: f(y), (0.0, spec.length), start_state(key), method="DOP853",
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise FloatingPointError(f"scipy DOP853 failed on {key}: {solution.message}")
    # DOP853 reports no global error estimate; None records that rather than inventing one.
    return {"kind": "scipy", "state": [float(x) for x in solution.y[:, -1]], "nfev": int(solution.nfev),
            "rtol": rtol, "atol": atol, "error_estimate": None}


# Pinned constant-curvature Jacobi provider (optional) ---------------------
CSG_REPOSITORY = "giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime"
CSG_IMPLEMENTATION = "Curved-Surface-Geodesic-Sensitivity-Runtime"
CSG_ENTRY = "geodesic_testbed.jacobi.integrate_jacobi"

# Runs in a separate interpreter so the provider never shares this process's imports.
# Besides the RK4 traces it returns, per case, the provider's TransferMap
# (matrices, determinant, focus events) built from that trace and from its
# closed-form constant_curvature_transfer.
_CSG_BOOTSTRAP = r'''
import dataclasses, json, sys
sys.path.insert(0, sys.argv[1])
import numpy
from geodesic_testbed.jacobi import integrate_jacobi
from geodesic_testbed.engine.transfer import TransferMap, constant_curvature_transfer
def summary(transfer):
    return {"matrices": transfer.matrices().tolist(), "determinant": transfer.determinant.tolist(),
            "focus_events": {column: [dataclasses.asdict(event) for event in transfer.focus_events(component=column)]
                             for column in ("a", "b")}}
cases = json.loads(sys.stdin.read())
traces, maps = [], []
for case in cases:
    trace = integrate_jacobi(case["arclength"], case["gaussian_curvature"])
    traces.append(trace.as_dict())
    numeric = TransferMap(arc_length=trace.arclength, a=trace.position_basis, a_rate=trace.position_rate,
                          b=trace.angle_basis, b_rate=trace.angle_rate)
    exact = constant_curvature_transfer(numpy.asarray(case["arclength"], dtype=float), case["gaussian_curvature"])
    maps.append({"numeric": summary(numeric), "closed_form": summary(exact)})
print(json.dumps({"python": sys.version.split()[0], "numpy": numpy.__version__, "traces": traces, "maps": maps},
                 sort_keys=True, allow_nan=False))
'''


class ProviderRefusal(ValueError):
    """A provider checkout or execution does not match its pin."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# Refusal codes by stage: pin verification before execution, then execution itself.
PIN_REFUSALS = ("CSG_CHECKOUT_UNREADABLE", "CSG_REVISION_MISMATCH", "CSG_TREE_MISMATCH", "CSG_CHECKOUT_DIRTY")
EXECUTION_REFUSALS = ("CSG_EXECUTION_FAILED", "CSG_CHANGED_DURING_EXECUTION")


def csg_pin() -> dict:
    """The pin shared with ciw.geodesic_reference (single source of truth)."""
    from ..geodesic_reference import PINS

    pin = PINS["curved-path-transfer"]
    return {"revision": pin["revision"], "source_tree": pin["source_tree"], "source_root": pin["source_root"]}


def _git(checkout, *args, raw: bool = False) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout
    return out if raw else out.strip()


def untracked_sources(checkout) -> list:
    """Untracked files under the provider source root, ignored ones included, other than bytecode caches.

    The bootstrap puts ``<checkout>/<source_root>`` first on sys.path, so any
    such file (an ignored ``numpy.py``, a stray module) could shadow pinned
    code on import even when ``git status`` reads clean.
    """
    listing = _git(checkout, "ls-files", "--others", "--", csg_pin()["source_root"]).splitlines()
    return sorted(path for path in listing if path and "__pycache__" not in path.split("/"))


def _status_paths(checkout) -> list:
    """Paths ``git status`` reports (modified, untracked and ignored), parsed from NUL-separated porcelain v1."""
    entries = _git(checkout, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored",
                   raw=True).split("\0")
    paths, index = [], 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        paths.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            # A rename or copy is followed by its origin path.
            paths.append(entries[index])
            index += 1
    return paths


def predict_csg_refusal(checkout) -> str:
    """Pin-stage refusal code a checkout calls for, or "none", from direct git queries.

    Computed before and independently of :func:`verify_csg_checkout` (which
    goes through ``ciw.lab.runner.git_identity``), so a refusal finding can
    compare an expected code with the observed one instead of copying it. The
    dirtiness rule is the core one: any modified, untracked or ignored path
    outside a runtime-cache directory, or a tracked file flagged skip-worktree
    or assume-unchanged (``ls-files -v`` tags ``S`` or lower case), whose bytes
    ``git status`` never compares.
    """
    import subprocess

    from .runner import RUNTIME_CACHES

    pin = csg_pin()
    try:
        head = _git(checkout, "rev-parse", "--verify", "HEAD")
        tree = _git(checkout, "rev-parse", "HEAD^{tree}")
        status = _status_paths(checkout)
        flagged = [line for line in _git(checkout, "ls-files", "-v", "--", ":(top)").splitlines()
                   if line[:1] == "S" or line[:1].islower()]
        stray = untracked_sources(checkout)
    except (OSError, subprocess.CalledProcessError):
        return "CSG_CHECKOUT_UNREADABLE"
    if head != pin["revision"]:
        return "CSG_REVISION_MISMATCH"
    if tree != pin["source_tree"]:
        return "CSG_TREE_MISMATCH"
    # Runtime caches are directories: a path is exempt when one of its parent directories is one.
    changed = [path for path in status if not set(path.rstrip("/").split("/")[:-1]) & set(RUNTIME_CACHES)]
    return "CSG_CHECKOUT_DIRTY" if changed or flagged or stray else "none"


def verify_csg_checkout(checkout) -> dict:
    """Refuse a checkout that is unreadable, dirty, shadowable or not at the pinned revision and tree."""
    from pathlib import Path
    import subprocess

    from .runner import git_identity

    pin = csg_pin()
    try:
        identity = git_identity(Path(checkout))
        stray = untracked_sources(checkout)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProviderRefusal("CSG_CHECKOUT_UNREADABLE",
                              f"Provider checkout is not a readable git repository ({type(exc).__name__})")
    if identity["revision"] != pin["revision"]:
        raise ProviderRefusal("CSG_REVISION_MISMATCH",
                              f"Provider revision {identity['revision']} differs from pin {pin['revision']}")
    if identity["source_tree"] != pin["source_tree"]:
        raise ProviderRefusal("CSG_TREE_MISMATCH", "Provider source tree differs from the pinned tree")
    if identity["dirty"] or stray:
        causes = (["uncommitted, untracked, ignored or index-flagged files outside the runtime caches"]
                  if identity["dirty"] else [])
        causes += [f"untracked files under its source root ({len(stray)})"] if stray else []
        raise ProviderRefusal("CSG_CHECKOUT_DIRTY", "Provider checkout has " + ", including ".join(causes))
    return {"repository": CSG_REPOSITORY, "revision": identity["revision"], "source_tree": identity["source_tree"],
            "dirty": False, "entry": CSG_ENTRY}


def _check_csg_output(data, cases) -> None:
    """Refuse provider output that does not answer every requested case in the expected shape."""
    if not isinstance(data, dict) or not isinstance(data["python"], str) or not isinstance(data["numpy"], str):
        raise TypeError("provider identity fields")
    traces, maps = data["traces"], data["maps"]
    if not (isinstance(traces, list) and isinstance(maps, list) and len(traces) == len(maps) == len(cases)):
        raise ValueError("provider returned a different number of cases")
    for case, entry in zip(cases, maps, strict=True):
        nodes = len(case["arclength"])
        for source in ("numeric", "closed_form"):
            summary = entry[source]
            matrices = np.asarray(summary["matrices"], dtype=float)
            determinant = np.asarray(summary["determinant"], dtype=float)
            if matrices.shape != (nodes, 2, 2) or determinant.shape != (nodes,):
                raise ValueError("provider transfer matrices do not match the requested grid")
            if not (np.all(np.isfinite(matrices)) and np.all(np.isfinite(determinant))):
                raise ValueError("provider transfer matrices are not finite")
            for column in ("a", "b"):
                for event in summary["focus_events"][column]:
                    float(event["arc_length"])


def run_csg_jacobi(checkout, cases, timeout: float = 120.0) -> dict:
    """Integrate declared (arclength grid, constant K) cases with the pinned provider in a subprocess.

    Anything other than a complete, well-formed answer is refused as
    ``CSG_EXECUTION_FAILED`` so callers never compare truncated output.
    """
    from pathlib import Path
    import json
    import subprocess
    import sys

    root = Path(checkout) / csg_pin()["source_root"]
    try:
        result = subprocess.run([sys.executable, "-c", _CSG_BOOTSTRAP, str(root)], input=json.dumps(cases),
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderRefusal("CSG_EXECUTION_FAILED", f"Provider subprocess did not complete ({type(exc).__name__})")
    if result.returncode != 0:
        raise ProviderRefusal("CSG_EXECUTION_FAILED", "Provider exited with a nonzero status: "
                              + (result.stderr.strip().splitlines() or ["no error output"])[-1][:300])
    try:
        data = json.loads(result.stdout)
        _check_csg_output(data, cases)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ProviderRefusal("CSG_EXECUTION_FAILED",
                              f"Provider output is not the expected document ({type(exc).__name__}: {exc})")
    return data
