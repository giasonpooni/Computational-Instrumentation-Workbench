"""Forward-mode dual numbers and symbolic (sympy) derivative references for surfaces.

Scope: each conformance surface is re-expressed as a closed-form embedding
X(u) (or, for the intrinsic hyperbolic chart, a closed-form metric) written
once against a small math namespace. The same formula then feeds

* nested forward-mode dual numbers implemented here (ciw origin): exact
  first, second and third derivatives up to rounding, with tags that keep
  nested perturbations apart; and
* sympy (optional, imported lazily): symbolic differentiation, Christoffel
  symbols and the Riemann tensor, evaluated numerically with lambdify.

Both are compared with the hand-coded derivatives of ``ciw.lab.surfaces``.
sympy is an implementation of distinct origin; the dual numbers are not and
count as a same-origin cross-check only.

Non-claims: the formulas define normalized mathematical surfaces. Agreement
shows that hand-coded derivatives match the derivatives of the declared
formula at the sampled points; it says nothing about whether the formula
describes any physical object.
"""
from __future__ import annotations

from itertools import count
import math

import numpy as np

from .surfaces_discrete_geometry import brioschi, christoffel_from

_TAGS = count(1)


class Dual:
    """a + b eps with eps^2 = 0 for one tagged perturbation; a and b may be nested duals.

    Arithmetic between duals of different tags treats the lower tag as a
    constant with respect to the higher one, which keeps nested directional
    derivatives from confusing their perturbations.
    """

    __slots__ = ("tag", "a", "b")

    def __init__(self, tag, a, b):
        self.tag, self.a, self.b = tag, a, b

    def __repr__(self):
        return f"Dual({self.tag}, {self.a!r}, {self.b!r})"

    def __add__(self, other):
        t = _top(self, other)
        a, b = _split(self, t)
        c, d = _split(other, t)
        return Dual(t, a + c, b + d)

    __radd__ = __add__

    def __sub__(self, other):
        t = _top(self, other)
        a, b = _split(self, t)
        c, d = _split(other, t)
        return Dual(t, a - c, b - d)

    def __rsub__(self, other):
        t = _top(self, other)
        a, b = _split(self, t)
        c, d = _split(other, t)
        return Dual(t, c - a, d - b)

    def __neg__(self):
        return Dual(self.tag, -self.a, -self.b)

    def __mul__(self, other):
        t = _top(self, other)
        a, b = _split(self, t)
        c, d = _split(other, t)
        return Dual(t, a * c, _sum(_prod(a, d), _prod(b, c)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        t = _top(self, other)
        a, b = _split(self, t)
        c, d = _split(other, t)
        return Dual(t, a / c, _prod(_sum(_prod(b, c), -_prod(a, d)), 1.0 / (c * c)))

    def __rtruediv__(self, other):
        t = _top(self, other)
        a, b = _split(other, t)
        c, d = _split(self, t)
        return Dual(t, a / c, _prod(_sum(_prod(b, c), -_prod(a, d)), 1.0 / (c * c)))

    def __pow__(self, n):
        if isinstance(n, Dual):
            raise TypeError("Dual exponents are not supported")
        return Dual(self.tag, self.a ** n, _prod(n * self.a ** (n - 1), self.b))


def _zero(x):
    return isinstance(x, float) and x == 0.0


def _prod(x, y):
    # Skip exact zeros so that nested duals stay small.
    if _zero(x) or _zero(y):
        return 0.0
    return x * y


def _sum(x, y):
    if _zero(x):
        return y
    if _zero(y):
        return x
    return x + y


def _top(x, y):
    return max(x.tag if isinstance(x, Dual) else 0, y.tag if isinstance(y, Dual) else 0)


def _split(x, tag):
    if isinstance(x, Dual) and x.tag == tag:
        return x.a, x.b
    return x, 0.0


def _unary(function, derivative):
    def apply(x):
        if isinstance(x, Dual):
            return Dual(x.tag, apply(x.a), _prod(derivative(x.a), x.b))
        return function(float(x))
    return apply


class DualMath:
    """Math namespace for formulas evaluated on (nested) dual numbers."""

    sin = staticmethod(_unary(math.sin, lambda a: DualMath.cos(a)))
    cos = staticmethod(_unary(math.cos, lambda a: -DualMath.sin(a)))
    exp = staticmethod(_unary(math.exp, lambda a: DualMath.exp(a)))
    sqrt = staticmethod(_unary(math.sqrt, lambda a: 0.5 / DualMath.sqrt(a)))


def value(x) -> float:
    """Real part of a (nested) dual number."""
    while isinstance(x, Dual):
        x = x.a
    return float(x)


def _map(function, tree):
    if isinstance(tree, (list, tuple)):
        return [_map(function, item) for item in tree]
    return function(tree)


def directional(f, u, direction):
    """Exact directional derivative of f at u along ``direction`` (nested lists allowed)."""
    tag = next(_TAGS)
    x = [Dual(tag, ui, float(di)) for ui, di in zip(u, direction)]
    return _map(lambda y: _split(y, tag)[1], f(x))


def partial(f, u, k):
    return directional(f, u, (1.0, 0.0) if k == 0 else (0.0, 1.0))


def _floats(tree):
    return np.array(_map(value, tree), dtype=float)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ------------------------------------------------------------- formulas
def formulas(surfaces: dict) -> dict:
    """Closed-form re-expression of each conformance surface: key -> (kind, formula(ns, u))."""
    out = {}
    for key, surface in surfaces.items():
        out[key] = _formula(key, surface)
    return out


def _formula(key, surface):
    if key == "plane":
        return "embedding", lambda ns, u: [u[0], u[1], 0.0 * u[0]]
    if key == "sphere":
        r = surface.radius
        return "embedding", lambda ns, u: [r * ns.sin(u[0]) * ns.cos(u[1]), r * ns.sin(u[0]) * ns.sin(u[1]),
                                           r * ns.cos(u[0])]
    if key == "cylinder":
        r = surface.radius
        return "embedding", lambda ns, u: [r * ns.cos(u[0]), r * ns.sin(u[0]), u[1]]
    if key == "saddle":
        c = surface.c
        return "embedding", lambda ns, u: [u[0], u[1], 0.5 * c * (u[0] * u[0] - u[1] * u[1])]
    if key in ("torus", "rotated-torus"):
        base = surface if key == "torus" else surface.base
        big, small = base.major, base.minor

        def torus(ns, u):
            rho = big + small * ns.cos(u[1])
            return [rho * ns.cos(u[0]), rho * ns.sin(u[0]), small * ns.sin(u[1])]
        if key == "torus":
            return "embedding", torus
        rotation = [[float(v) for v in row] for row in surface.rotation]
        return "embedding", lambda ns, u: [sum(rotation[i][j] * x for j, x in enumerate(torus(ns, u)))
                                           for i in range(3)]
    if key in ("gaussian-bump", "gaussian-bump-shear"):
        base = surface if key == "gaussian-bump" else surface.base
        h, s2 = base.h, base.sigma ** 2
        shear = 0.0 if key == "gaussian-bump" else surface.chart.c

        def bump(ns, u):
            x = u[0] + shear * u[1] * u[1]
            y = u[1]
            return [x, y, h * ns.exp(-(x * x + y * y) / (2.0 * s2))]
        return "embedding", bump
    if key == "hyperbolic-plane":
        k2 = surface.k ** 2
        return "metric", lambda ns, u: [[1.0 / (k2 * u[1] * u[1]), 0.0 * u[0]], [0.0 * u[0], 1.0 / (k2 * u[1] * u[1])]]
    if key == "plane-polar":
        return "embedding", lambda ns, u: [u[0] * ns.cos(u[1]), u[0] * ns.sin(u[1]), 0.0 * u[0]]
    raise KeyError(f"No closed-form formula for surface {key}")


class DualSurface:
    """Metric data of a re-expressed formula, differentiated with nested dual numbers."""

    def __init__(self, kind, formula):
        self.kind, self.formula = kind, formula

    def point(self, u):
        return _floats(self.formula(DualMath, list(u))) if self.kind == "embedding" else None

    def _metric(self, u):
        if self.kind == "metric":
            return self.formula(DualMath, u)
        x = lambda w: self.formula(DualMath, w)  # noqa: E731
        xu, xv = partial(x, u, 0), partial(x, u, 1)
        return [[_dot(xu, xu), _dot(xu, xv)], [_dot(xv, xu), _dot(xv, xv)]]

    def metric(self, u):
        return _floats(self._metric(list(u)))

    def metric_derivatives(self, u):
        return _floats([partial(self._metric, list(u), k) for k in range(2)])

    def metric_second_derivatives(self, u):
        return _floats([[partial(lambda w, k=k: partial(self._metric, w, k), list(u), m) for k in range(2)]
                        for m in range(2)])

    def metric_third_derivative(self, u, k):
        """d_k^3 g_ij along one coordinate direction."""
        def d1(w):
            return partial(self._metric, w, k)

        def d2(w):
            return partial(d1, w, k)
        return _floats(partial(d2, list(u), k))

    def christoffel(self, u):
        return christoffel_from(self.metric(u), self.metric_derivatives(u))

    def intrinsic_curvature(self, u):
        """Brioschi curvature with exact dual-number second metric derivatives."""
        return brioschi(self.metric(u), self.metric_derivatives(u), self.metric_second_derivatives(u))

    def extrinsic_curvature(self, u):
        """(LN - M^2) / det g from dual-number embedding derivatives; None for an intrinsic chart."""
        if self.kind != "embedding":
            return None
        x = lambda w: self.formula(DualMath, w)  # noqa: E731
        u = list(u)
        first = [_floats(partial(x, u, k)) for k in range(2)]
        second = [[_floats(partial(lambda w, j=j: partial(x, w, j), u, i)) for j in range(2)] for i in range(2)]
        normal = np.cross(first[0], first[1])
        normal = normal / math.sqrt(float(normal @ normal))
        g = np.array([[first[i] @ first[j] for j in range(2)] for i in range(2)])
        big_l, big_m, big_n = second[0][0] @ normal, second[0][1] @ normal, second[1][1] @ normal
        return float((big_l * big_n - big_m * big_m) / (g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]))


# ------------------------------------------------------------- sympy
def symbolic_reference(kind, formula):
    """Lambdified sympy metric, dg, Christoffel symbols and Riemann-tensor curvature.

    Returns a function u -> dict of float arrays. Curvature is
    K = R_{1212} / det g with R^l_{ijk} = d_j G^l_ik - d_k G^l_ij + G^l_jm G^m_ik - G^l_km G^m_ij,
    a formula distinct from both the second fundamental form and Brioschi.
    """
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    coords = (u, v)
    if kind == "embedding":
        x = [sp.sympify(c) for c in formula(sp, [u, v])]
        xd = [[sp.diff(c, q) for c in x] for q in coords]
        g = [[sum(a * b for a, b in zip(xd[i], xd[j])) for j in range(2)] for i in range(2)]
    else:
        g = [[sp.sympify(c) for c in row] for row in formula(sp, [u, v])]
    dg = [[[sp.diff(g[i][j], coords[k]) for j in range(2)] for i in range(2)] for k in range(2)]
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    ginv = [[g[1][1] / det, -g[0][1] / det], [-g[1][0] / det, g[0][0] / det]]
    gamma = [[[sum(ginv[k][m] * (dg[i][j][m] + dg[j][i][m] - dg[m][i][j]) for m in range(2)) / 2
               for j in range(2)] for i in range(2)] for k in range(2)]

    def riemann_up(m):  # R^m_{101}
        return (sp.diff(gamma[m][1][1], u) - sp.diff(gamma[m][1][0], v)
                + sum(gamma[m][0][n] * gamma[n][1][1] - gamma[m][1][n] * gamma[n][1][0] for n in range(2)))

    curvature = sum(g[0][m] * riemann_up(m) for m in range(2)) / det
    flat = ([g[i][j] for i in range(2) for j in range(2)]
            + [dg[k][i][j] for k in range(2) for i in range(2) for j in range(2)]
            + [gamma[k][i][j] for k in range(2) for i in range(2) for j in range(2)] + [curvature])
    function = sp.lambdify((u, v), flat, modules="math", cse=True)

    def evaluate(point):
        values = np.array(function(float(point[0]), float(point[1])), dtype=float)
        return {"metric": values[:4].reshape(2, 2), "metric_derivatives": values[4:12].reshape(2, 2, 2),
                "christoffel": values[12:20].reshape(2, 2, 2), "gaussian_curvature": float(values[20])}
    return evaluate


def symbolic_exact_curvature(kind, formula):
    """sympy-simplified curvature expression as a string (small formulas only)."""
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    coords = (u, v)
    if kind == "embedding":
        x = [sp.sympify(c) for c in formula(sp, [u, v])]
        xd = [[sp.diff(c, q) for c in x] for q in coords]
        g = [[sp.simplify(sum(a * b for a, b in zip(xd[i], xd[j]))) for j in range(2)] for i in range(2)]
    else:
        g = [[sp.sympify(c) for c in row] for row in formula(sp, [u, v])]
    e, f, gg = g[0][0], g[0][1], g[1][1]
    eu, ev, fu, fv, gu, gv = (sp.diff(e, u), sp.diff(e, v), sp.diff(f, u), sp.diff(f, v), sp.diff(gg, u),
                              sp.diff(gg, v))
    first = sp.Matrix([[-sp.diff(e, v, 2) / 2 + sp.diff(f, u, v) - sp.diff(gg, u, 2) / 2, eu / 2, fu - ev / 2],
                       [fv - gu / 2, e, f], [gv / 2, f, gg]])
    second = sp.Matrix([[0, ev / 2, gu / 2], [ev / 2, e, f], [gu / 2, f, gg]])
    return sp.nsimplify(sp.simplify((first.det() - second.det()) / (e * gg - f * f) ** 2))
