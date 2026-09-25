"""Forward-mode dual numbers and symbolic (sympy) derivative references for surfaces.

Scope: each conformance surface is re-expressed as a closed-form embedding
X(u) (or, for the intrinsic hyperbolic chart, a closed-form metric) written
once against a small math namespace; parameters enter through ``ns.const`` so
that the symbolic namespace sees them as exact rationals. The same formula
then feeds

* nested forward-mode dual numbers implemented here (ciw origin): exact
  first, second and third derivatives up to rounding, with tags that keep
  nested perturbations apart;
* sympy differentiation (optional, imported lazily): symbolic metric and
  metric derivatives, plus Christoffel symbols and Riemann curvature
  assembled here in ciw code from those derivatives; and
* sympy.diffgeom (optional): Christoffel symbols and Riemann components
  assembled by sympy itself, so both differentiation and assembly have a
  distinct origin. For the surfaces whose symbolic Riemann tensor is cheap it
  assembles the surface's own metric (:func:`diffgeom_reference`); for the
  others it assembles a generic metric (E, F; F, G) once
  (:class:`GenericDiffgeom`), whose expressions are evaluated at the metric
  2-jet that sympy differentiation gives at each point
  (:func:`pointwise_diffgeom_reference`). An exact identity on a rational
  Taylor metric checks that evaluation (:func:`taylor_jet_check`).

All are compared with the hand-coded derivatives of ``ciw.lab.surfaces``.
Only sympy-differentiated and sympy.diffgeom-assembled quantities have a
distinct origin; the dual numbers and the ciw assembly count as same-origin
cross-checks. :func:`complex_step_derivatives` differentiates the core
interface itself with a complex step, which works only where the core
formulas carry complex coordinates.

Non-claims: the formulas define normalized mathematical surfaces. Agreement
shows that hand-coded derivatives match the derivatives of the declared
formula at the sampled points; it says nothing about whether the formula
describes any physical object.
"""
from __future__ import annotations

from itertools import count
import math
import warnings

import numpy as np

from .surfaces import SurfaceRefusal
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
    const = staticmethod(float)


class _SympyMath:
    """Math namespace for sympy: parameters become the exact rationals of their binary floats."""

    def __init__(self, sp):
        self.sin, self.cos, self.exp, self.sqrt = sp.sin, sp.cos, sp.exp, sp.sqrt
        self.const = sp.Rational


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
        return "embedding", lambda ns, u: [u[0], u[1], ns.const(0.0) * u[0]]
    if key == "sphere":
        radius = surface.radius

        def sphere(ns, u):
            r = ns.const(radius)
            return [r * ns.sin(u[0]) * ns.cos(u[1]), r * ns.sin(u[0]) * ns.sin(u[1]), r * ns.cos(u[0])]
        return "embedding", sphere
    if key == "cylinder":
        radius = surface.radius
        return "embedding", lambda ns, u: [ns.const(radius) * ns.cos(u[0]), ns.const(radius) * ns.sin(u[0]), u[1]]
    if key == "saddle":
        c = surface.c
        return "embedding", lambda ns, u: [u[0], u[1], ns.const(c) / 2 * (u[0] * u[0] - u[1] * u[1])]
    if key in ("torus", "rotated-torus"):
        base = surface if key == "torus" else surface.base
        big, small = base.major, base.minor

        def torus(ns, u):
            rho = ns.const(big) + ns.const(small) * ns.cos(u[1])
            return [rho * ns.cos(u[0]), rho * ns.sin(u[0]), ns.const(small) * ns.sin(u[1])]
        if key == "torus":
            return "embedding", torus
        rotation = [[float(v) for v in row] for row in surface.rotation]
        return "embedding", lambda ns, u: [sum(ns.const(rotation[i][j]) * x for j, x in enumerate(torus(ns, u)))
                                           for i in range(3)]
    if key in ("gaussian-bump", "gaussian-bump-shear"):
        base = surface if key == "gaussian-bump" else surface.base
        height, s2 = base.h, base.sigma ** 2
        shear = 0.0 if key == "gaussian-bump" else surface.chart.c

        def bump(ns, u):
            x = u[0] + ns.const(shear) * u[1] * u[1]
            y = u[1]
            return [x, y, ns.const(height) * ns.exp(-(x * x + y * y) / (2 * ns.const(s2)))]
        return "embedding", bump
    if key == "hyperbolic-plane":
        k = surface.k

        def hyperbolic(ns, u):
            diagonal = 1 / (ns.const(k) ** 2 * u[1] * u[1])
            return [[diagonal, ns.const(0.0) * u[0]], [ns.const(0.0) * u[0], diagonal]]
        return "metric", hyperbolic
    if key == "plane-polar":
        return "embedding", lambda ns, u: [u[0] * ns.cos(u[1]), u[0] * ns.sin(u[1]), ns.const(0.0) * u[0]]
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
def _symbolic_metric(sp, kind, formula, u, v):
    """Metric of a formula with exact parameters: g_ij = X_i . X_j by sympy differentiation."""
    ns = _SympyMath(sp)
    if kind == "embedding":
        x = [sp.sympify(c) for c in formula(ns, [u, v])]
        xd = [[sp.diff(c, q) for c in x] for q in (u, v)]
        return [[sum(a * b for a, b in zip(xd[i], xd[j])) for j in range(2)] for i in range(2)]
    return [[sp.sympify(c) for c in row] for row in formula(ns, [u, v])]


def symbolic_reference(kind, formula):
    """Lambdified sympy metric and dg, with Christoffel symbols and curvature assembled in ciw code.

    Returns a function u -> dict of float arrays. Differentiation is sympy's;
    the Levi-Civita formula and the Riemann contraction
    K = R_{1212} / det g with R^l_{ijk} = d_j G^l_ik - d_k G^l_ij + G^l_jm G^m_ik - G^l_km G^m_ij
    are written here, so only ``metric`` and ``metric_derivatives`` have a
    distinct origin (see :func:`diffgeom_reference` for sympy's own assembly).
    """
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    coords = (u, v)
    g = _symbolic_metric(sp, kind, formula, u, v)
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


def _diffgeom_assembly(sp, g, u, v):
    """sympy.diffgeom's connection and curvature of the metric ``g`` (2x2 sympy expressions in the symbols u, v).

    Returns nine expressions in u, v: ``Gamma^k_ij`` in (k, i, j) order from
    ``metric_to_Christoffel_2nd``, then ``K = g_{0m} R^m_{101} / det g`` from
    ``metric_to_Riemann_components`` (R[rho, sigma, mu, nu] = R^rho_{sigma mu nu}).
    """
    from sympy.diffgeom import (CoordSystem, Manifold, Patch, TensorProduct, metric_to_Christoffel_2nd,
                                metric_to_Riemann_components)

    a, b = sp.symbols("a b", real=True)
    chart = CoordSystem("C", Patch("P", Manifold("M", 2)), [a, b])
    fields, forms = chart.coord_functions(), chart.base_oneforms()
    to_fields = {u: fields[0], v: fields[1]}
    metric = sum(g[i][j].subs(to_fields) * TensorProduct(forms[i], forms[j]) for i in range(2) for j in range(2))
    back = {fields[0]: u, fields[1]: v}
    christoffel = metric_to_Christoffel_2nd(metric)
    riemann = metric_to_Riemann_components(metric)
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    curvature = sum(g[0][m] * sp.sympify(riemann[m, 1, 0, 1]).subs(back) for m in range(2)) / det
    return [sp.sympify(christoffel[k, i, j]).subs(back) for k in range(2) for i in range(2) for j in range(2)] + [
        curvature]


def _connection(values) -> dict:
    values = np.array(values, dtype=float)
    return {"christoffel": values[:8].reshape(2, 2, 2), "gaussian_curvature": float(values[8])}


def diffgeom_reference(kind, formula):
    """Christoffel symbols and curvature assembled by sympy.diffgeom from the symbolic metric.

    Returns (evaluate, curvature expression): ``evaluate(u)`` gives
    ``christoffel[k, i, j] = Gamma^k_ij`` from ``metric_to_Christoffel_2nd``
    and ``gaussian_curvature = g_{0m} R^m_{101} / det g`` from
    ``metric_to_Riemann_components`` (R[rho, sigma, mu, nu] = R^rho_{sigma mu nu});
    the expression is the unsimplified symbolic K in the symbols u, v.
    """
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    flat = _diffgeom_assembly(sp, _symbolic_metric(sp, kind, formula, u, v), u, v)
    function = sp.lambdify((u, v), flat, modules="math", cse=True)

    def evaluate(point):
        return _connection(function(float(point[0]), float(point[1])))
    return evaluate, flat[8], (u, v)


# The metric 2-jet at a point: the independent components g_00, g_01 = g_10 and
# g_11 (E, F, G), each with its derivatives of these orders (in u, in v). The
# Christoffel symbols at a point depend on g and its first derivatives there,
# the Riemann tensor also on its second derivatives, and on nothing else.
JET_COMPONENTS = ((0, 0), (0, 1), (1, 1))
JET_ORDERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
# An exact 2-jet (E, F, G, each in JET_ORDERS order) with 18 distinct nonzero
# entries and E G - F^2 = 23/4 > 0, for taylor_jet_check.
TAYLOR_JET = ((2, 1), (1, 3), (-1, 5), (1, 7), (2, 9), (-1, 11),
              (1, 2), (-2, 13), (3, 17), (-1, 19), (4, 23), (1, 29),
              (3, 1), (-1, 31), (2, 37), (1, 41), (-3, 43), (1, 47))


class GenericDiffgeom:
    """sympy.diffgeom's connection and curvature of a generic metric (E, F; F, G), evaluated at a metric 2-jet.

    ``metric_to_Christoffel_2nd`` and ``metric_to_Riemann_components`` run once
    on a metric whose components are undetermined functions E(u, v), F(u, v)
    and G(u, v). Every term of the result is one of those functions or one of
    their first or second partial derivatives; each is replaced by a jet symbol
    (``symbols``, in JET_COMPONENTS x JET_ORDERS order), so evaluating the
    expressions at a point needs only the metric's 2-jet there. The assembly
    is sympy.diffgeom's; this class only names its terms, and a term it cannot
    name is refused.
    """

    def __init__(self):
        import sympy as sp
        from sympy.core.function import AppliedUndef

        u, v = sp.symbols("u v", real=True)
        functions = {pair: sp.Function(name)(u, v) for pair, name in zip(JET_COMPONENTS, "EFG")}
        g = [[functions[(0, 0)], functions[(0, 1)]], [functions[(0, 1)], functions[(1, 1)]]]
        flat = [expression.doit() for expression in _diffgeom_assembly(sp, g, u, v)]
        jet, self.symbols = {}, []
        for pair, name in zip(JET_COMPONENTS, "EFG"):
            for p, q in JET_ORDERS:
                symbol = sp.Symbol(f"{name}_{p}{q}", real=True)
                term = functions[pair] if p + q == 0 else sp.Derivative(functions[pair], *([u] * p + [v] * q))
                jet[term] = symbol
                self.symbols.append(symbol)
        self.expressions = [expression.xreplace(jet) for expression in flat]
        unnamed = set().union(*(e.atoms(sp.Derivative, sp.Subs, AppliedUndef) for e in self.expressions))
        unnamed |= set().union(*(e.free_symbols for e in self.expressions)) - set(self.symbols)
        if unnamed:
            raise ValueError(f"Generic assembly has terms outside the metric 2-jet: {sorted(map(str, unnamed))}")
        self._function = sp.lambdify(self.symbols, self.expressions, modules="math", cse=True)

    def __call__(self, jet) -> dict:
        return _connection(self._function(*[float(value) for value in jet]))


def _symbolic_jet(sp, kind, formula, u, v):
    """The metric 2-jet of a formula by sympy differentiation, 18 expressions in GenericDiffgeom.symbols order."""
    g = _symbolic_metric(sp, kind, formula, u, v)
    return [sp.diff(g[i][j], u, p, v, q) for i, j in JET_COMPONENTS for p, q in JET_ORDERS]


def metric_jet(kind, formula):
    """u -> the metric 2-jet of a formula by sympy differentiation, 18 floats in GenericDiffgeom.symbols order."""
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    function = sp.lambdify((u, v), _symbolic_jet(sp, kind, formula, u, v), modules="math", cse=True)
    return lambda point: function(float(point[0]), float(point[1]))


def generic_curvature(assembly: GenericDiffgeom, kind, formula):
    """The generic assembly's curvature K at a formula's symbolic 2-jet, exact in its parameters, and (u, v).

    For an exact identity with a closed form through the generic route; its
    cost is the simplification, which grows with the jet expressions
    (docs/lab/SURFACE_INTERFACE.md records it for the two gaussian bumps).
    """
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    jet = dict(zip(assembly.symbols, _symbolic_jet(sp, kind, formula, u, v)))
    return assembly.expressions[8].xreplace(jet), (u, v)


def pointwise_diffgeom_reference(kind, formula, assembly: GenericDiffgeom):
    """Christoffel symbols and curvature of a formula from sympy.diffgeom's generic assembly at the sympy 2-jet.

    The same quantities as :func:`diffgeom_reference` without assembling the
    surface's own symbolic Riemann tensor, which is slow for large metric
    expressions (docs/lab/SURFACE_INTERFACE.md records the measured cost).
    """
    jet = metric_jet(kind, formula)
    return lambda point: assembly(jet(point))


def taylor_jet_check(assembly: GenericDiffgeom) -> dict:
    """Exact check of the generic assembly's jet evaluation against sympy.diffgeom's own assembly.

    sympy.diffgeom assembles the quadratic Taylor metric of the exact jet
    TAYLOR_JET directly; at the origin its Christoffel symbols and curvature
    depend only on that jet, so they must equal the generic expressions at
    the jet exactly, in rational arithmetic. Every jet entry is nonzero and
    distinct, so a term named with the wrong entry changes a value unless the
    result cannot tell the two entries apart. ``used`` lists the jet entries
    the generic expressions contain.
    """
    import sympy as sp

    u, v = sp.symbols("u v", real=True)
    values = [sp.Rational(n, d) for n, d in TAYLOR_JET]
    taylor = {pair: sum(values[6 * index + order] * u ** p * v ** q / (sp.factorial(p) * sp.factorial(q))
                        for order, (p, q) in enumerate(JET_ORDERS))
              for index, pair in enumerate(JET_COMPONENTS)}
    g = [[taylor[(0, 0)], taylor[(0, 1)]], [taylor[(0, 1)], taylor[(1, 1)]]]
    direct = [expression.subs({u: 0, v: 0}) for expression in _diffgeom_assembly(sp, g, u, v)]
    generic = [expression.subs(dict(zip(assembly.symbols, values))) for expression in assembly.expressions]
    used = set().union(*(expression.free_symbols for expression in assembly.expressions))
    return {"jet": [str(value) for value in values], "direct": [str(value) for value in direct],
            "generic": [str(value) for value in generic], "components": len(direct),
            "used": sorted(str(symbol) for symbol in used),
            "nonzero_components": sum(value != 0 for value in direct),
            "rational": all(value.is_Rational for value in direct + generic),
            # Indices in the order of the assembly: Gamma^k_ij in (k, i, j) order, then K.
            "mismatched_components": [index for index, (a, b) in enumerate(zip(direct, generic))
                                      if sp.simplify(a - b) != 0]}


# ------------------------------------------------------------- complex step
COMPLEX_STEP = 1e-30
COMPLEX_STEP_TOLERANCE = 1e-13  # normalized; the complex step has no cancellation, so rounding is all that remains
_COMPLEX_WARNING = np.exceptions.ComplexWarning


def complex_step_derivatives(surface, u, h=COMPLEX_STEP):
    """Complex-step metric derivatives Im g(u + i h e_k) / h through the core interface, and how the interface ran.

    Returns ``(dg, outcome)``. ``outcome`` is ``"refused"`` when the metric
    raised for complex coordinates (``dg`` is None), ``"imaginary_part_discarded"``
    when NumPy cast them to real with a ComplexWarning (``dg`` is then whatever
    imaginary part survived), else ``"carried"``. Warnings
    are recorded here whatever the caller's filters, so the outcome does not
    depend on them.
    """
    dg, outcomes = np.zeros((2, 2, 2)), []
    for k in range(2):
        z = np.array(u, dtype=complex)
        z[k] += 1j * h
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                g = np.asarray(surface.metric(z))
            except (TypeError, ValueError, SurfaceRefusal):
                return None, "refused"
        outcomes.append("imaginary_part_discarded" if any(issubclass(w.category, _COMPLEX_WARNING) for w in caught)
                        else "carried")
        dg[k] = np.imag(g) / h
    return dg, ("imaginary_part_discarded" if "imaginary_part_discarded" in outcomes else "carried")
