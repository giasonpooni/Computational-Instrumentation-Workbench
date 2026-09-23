"""Conformance suite, intrinsic curvature and derivative stencils for the surface interface.

Scope: validates and extends the chart-level interface of ``ciw.lab.surfaces``
(metric ``g_ij``, exact first derivatives ``dg[k, i, j] = d_k g_ij``,
Christoffel symbols and supplied Gaussian curvature) without modifying it.
It adds three concrete chart maps (polar, a polynomial shear and a
cube-root chart), three surfaces with singular points or boundaries (a cone,
the graph z = c r^p and the conformal half-plane g = y^(-2a) I), the Brioschi
formula for curvature from the metric alone, fourth-order difference
stencils, a conformance suite run over declared domains, and seeded defect
mutants that the suite must reject.

Non-claims: every surface, coordinate and curvature is in declared normalized
units. Passing the suite shows that one implementation is self-consistent and
consistent with the Gauss equation on the sampled points of its declared
domain; it does not show correctness outside that domain, between samples, or
for any measured physical surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .surfaces import (ChartMap, EmbeddedSurface, GaussianBump, MongeSurface, Reparametrized, Rotated, Saddle,
                       Sphere, Surface, SurfaceRefusal, Torus, catalogue, rotation_matrix, sampling_domain)

SEED = 3301
EPS = float(np.finfo(float).eps)

# Relative step of the fourth-order stencils; truncation ~ h^4, rounding ~ eps/h.
STENCIL_STEP = 1e-3
STENCIL5 = ((-2, 1.0 / 12.0), (-1, -8.0 / 12.0), (1, 8.0 / 12.0), (2, -1.0 / 12.0))

# Conformance thresholds on normalized residuals. Symmetry, compatibility and
# Christoffel symmetry are algebraic identities (rounding only); derivative
# consistency and the Gauss equation carry fourth-order stencil error.
THRESHOLDS = {
    "metric_asymmetry": 1e-13,
    "min_eigenvalue_ratio": 1e-3,
    "derivative_asymmetry": 1e-13,
    "christoffel_asymmetry": 1e-12,
    "compatibility": 1e-12,
    "derivative_consistency": 1e-8,
    "mixed_partials": 1e-7,
    "gauss_equation": 1e-7,
}


# ---------------------------------------------------------------- chart maps
class PolarChart(ChartMap):
    """a = (r, t) -> u = (r cos t, r sin t); the chart degenerates at r = 0."""

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
        # hessian[p, i, k] = d^2 u^p / (da^i da^k)
        return np.array([[[0.0, -s], [-s, -r * c]], [[0.0, c], [c, -r * s]]])

    def inverse(self, u):
        return np.array([math.hypot(u[0], u[1]), math.atan2(u[1], u[0])])


class ShearChart(ChartMap):
    """a -> u = (a1 + c a2^2, a2): a global polynomial diffeomorphism of the plane."""

    name = "shear"

    def __init__(self, c=0.3):
        self.c = float(c)

    def forward(self, a):
        return np.array([a[0] + self.c * a[1] ** 2, a[1]])

    def jacobian(self, a):
        return np.array([[1.0, 2.0 * self.c * a[1]], [0.0, 1.0]])

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        hess[0, 1, 1] = 2.0 * self.c
        return hess

    def inverse(self, u):
        return np.array([u[0] - self.c * u[1] ** 2, u[1]])


class CubeRootChart(ChartMap):
    """a -> u = (cbrt(a1), a2): a homeomorphism of the plane whose Jacobian blows up on a1 = 0.

    Pulling back a smooth surface through it gives a metric that blows up at
    finite distance while the surface itself stays smooth there.
    """

    name = "cube-root"

    def forward(self, a):
        return np.array([math.copysign(abs(a[0]) ** (1.0 / 3.0), a[0]), a[1]])

    def jacobian(self, a):
        if a[0] == 0.0:
            raise SurfaceRefusal("Cube-root chart: the Jacobian is unbounded on a1 = 0", "degenerate_metric")
        return np.array([[abs(a[0]) ** (-2.0 / 3.0) / 3.0, 0.0], [0.0, 1.0]])

    def hessian(self, a):
        if a[0] == 0.0:
            raise SurfaceRefusal("Cube-root chart: the Hessian is unbounded on a1 = 0", "degenerate_metric")
        hess = np.zeros((2, 2, 2))
        hess[0, 0, 0] = -2.0 / 9.0 * math.copysign(abs(a[0]) ** (-5.0 / 3.0), a[0])
        return hess

    def inverse(self, u):
        return np.array([u[0] ** 3, u[1]])


# ------------------------------------------------------- singular surfaces
class Cone(EmbeddedSurface):
    """Cone of half-angle alpha in the chart (r, phi), r the slant distance to the apex.

    K = 0 for r > 0; the apex carries the concentrated curvature (angle
    deficit) 2 pi (1 - sin alpha) and is not a point of any smooth chart.
    """

    name = "cone"

    def __init__(self, half_angle=math.pi / 6):
        if not 0 < half_angle < math.pi / 2:
            raise SurfaceRefusal("Cone half-angle must lie in (0, pi/2)", "invalid_parameter")
        self.alpha = float(half_angle)
        self.sa, self.ca = math.sin(self.alpha), math.cos(self.alpha)

    def describe(self):
        return {"name": self.name, "half_angle": self.alpha, "chart": "(r, phi), r slant distance"}

    def angle_deficit(self) -> float:
        return 2.0 * math.pi * (1.0 - self.sa)

    def embedding(self, u):
        r, p = u
        return r * np.array([self.sa * math.cos(p), self.sa * math.sin(p), self.ca])

    def first(self, u):
        r, p = u
        return (np.array([self.sa * math.cos(p), self.sa * math.sin(p), self.ca]),
                r * np.array([-self.sa * math.sin(p), self.sa * math.cos(p), 0.0]))

    def second(self, u):
        r, p = u
        return (np.zeros(3), np.array([-self.sa * math.sin(p), self.sa * math.cos(p), 0.0]),
                r * np.array([-self.sa * math.cos(p), -self.sa * math.sin(p), 0.0]))

    def gaussian_curvature(self, u):
        if not u[0] > 0:
            raise SurfaceRefusal("Cone apex: curvature is concentrated at the vertex", "conical_singularity")
        return 0.0


class PowerGraph(MongeSurface):
    """Graph z = c rho^p, rho = |(x, y)|, 1 < p < 2: C^1 at the origin with K ~ rho^(2p-4).

    The Monge chart is regular (det g >= 1); the curvature singularity at the
    origin is intrinsic and is refused there.
    """

    name = "power-graph"

    def __init__(self, c=1.0, power=1.5):
        if not 1.0 < power < 2.0:
            raise SurfaceRefusal("Power graph exponent must lie in (1, 2)", "invalid_parameter")
        self.c, self.p = float(c), float(power)

    def describe(self):
        return {"name": self.name, "c": self.c, "power": self.p, "graph": "z = c (x^2 + y^2)^(p/2)"}

    def height(self, u):
        return self.c * math.hypot(u[0], u[1]) ** self.p

    def height_derivatives(self, u):
        x, y = float(u[0]), float(u[1])
        rho = math.hypot(x, y)
        if rho == 0.0:
            raise SurfaceRefusal("Power graph apex: second derivatives and curvature diverge",
                                 "curvature_singularity")
        c, p = self.c, self.p
        a = c * p * rho ** (p - 2)
        b = c * p * (p - 2) * rho ** (p - 4)
        return a * x, a * y, a + b * x * x, b * x * y, a + b * y * y

    def radial_curvature(self, rho) -> float:
        """Closed form K(rho) = f' f'' / (rho (1 + f'^2)^2) for the surface of revolution."""
        c, p = self.c, self.p
        f1, f2 = c * p * rho ** (p - 1), c * p * (p - 1) * rho ** (p - 2)
        return f1 * f2 / (rho * (1 + f1 * f1) ** 2)


class ConformalHalfPlane(Surface):
    """g = y^(-2a) I on y > 0: K = -a y^(2a - 2).

    a = 1 is the hyperbolic plane (boundary at infinite distance, K = -1). For
    0 < a < 1 the boundary y = 0 lies at finite distance y^(1-a) / (1 - a)
    and K diverges there, ever more slowly as a approaches 1.
    """

    name = "conformal-half-plane"

    def __init__(self, a=0.9):
        if not a > 0:
            raise SurfaceRefusal("Conformal exponent must be positive", "invalid_parameter")
        self.a = float(a)

    def describe(self):
        return {"name": self.name, "a": self.a, "metric": "y^(-2a) I on y > 0"}

    def check(self, u):
        if not (np.all(np.isfinite(u)) and u[1] > 0):
            raise SurfaceRefusal("Conformal half-plane requires y > 0", "outside_chart")

    def metric(self, u):
        return np.eye(2) * u[1] ** (-2.0 * self.a)

    def metric_derivatives(self, u):
        dg = np.zeros((2, 2, 2))
        dg[1] = -2.0 * self.a * u[1] ** (-2.0 * self.a - 1.0) * np.eye(2)
        return dg

    def gaussian_curvature(self, u):
        # g = exp(2 w) I with w = -a ln y: K = -exp(-2 w) Laplacian(w) = -a y^(2a - 2).
        return -self.a * u[1] ** (2.0 * self.a - 2.0)


# ---------------------------------------------------------------- mutants
class MisscaledSphere(Sphere):
    """Defect: K returned as 1/R instead of 1/R^2 (only the Gauss equation can see it)."""

    name = "mutant-misscaled-curvature"

    def gaussian_curvature(self, u):
        return 1.0 / self.radius


class TransposedTorus(Torus):
    """Defect: metric derivatives returned as dg[i, k, j] instead of dg[k, i, j]."""

    name = "mutant-transposed-derivatives"

    def metric_derivatives(self, u):
        return np.ascontiguousarray(np.transpose(super().metric_derivatives(u), (1, 0, 2)))


class SignFlippedSaddle(Saddle):
    """Defect: metric derivatives with the wrong sign."""

    name = "mutant-sign-flipped-derivatives"

    def metric_derivatives(self, u):
        return -super().metric_derivatives(u)


class DroppedCrossTermBump(GaussianBump):
    """Defect: the mixed height derivative f_xy dropped from the exact derivatives."""

    name = "mutant-dropped-cross-term"

    def height_derivatives(self, u):
        fx, fy, fxx, _, fyy = super().height_derivatives(u)
        return fx, fy, fxx, 0.0, fyy


class LorentzianChart(Surface):
    """Defect: an indefinite 'metric' diag(1, -(1 + y^2)) that no surface can have."""

    name = "mutant-indefinite-metric"

    def metric(self, u):
        return np.array([[1.0, 0.0], [0.0, -(1.0 + u[1] ** 2)]])

    def metric_derivatives(self, u):
        dg = np.zeros((2, 2, 2))
        dg[1, 1, 1] = -2.0 * u[1]
        return dg

    def gaussian_curvature(self, u):
        return 0.0


class NaNDerivativeTorus(Torus):
    """Defect: metric derivatives that turn NaN on half of the domain (u1 > 0)."""

    name = "mutant-nan-derivatives"

    def metric_derivatives(self, u):
        dg = super().metric_derivatives(u)
        if u[0] > 0:
            dg[0, 0, 0] = math.nan
        return dg


class NonsymmetricMetric(Surface):
    """Defect: an asymmetric metric matrix with a small off-diagonal slip."""

    name = "mutant-nonsymmetric-metric"

    def metric(self, u):
        return np.array([[1.0, 0.1], [0.0, 1.0]])

    def metric_derivatives(self, u):
        return np.zeros((2, 2, 2))

    def gaussian_curvature(self, u):
        return 0.0


# ------------------------------------------------------------ domains
@dataclass(frozen=True)
class Domain:
    """Axis-aligned sampling box and the local length scale used to normalize residuals."""

    low: tuple
    high: tuple
    scale: Callable = None

    def length(self, u) -> float:
        return 1.0 if self.scale is None else float(self.scale(u))

    def sample(self, count: int, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        low, high = np.array(self.low, dtype=float), np.array(self.high, dtype=float)
        return low + (high - low) * rng.random((count, 2))


def _hyperbolic_scale(u):
    return u[1]


def _core_domain(key, scale=None) -> Domain:
    """A catalogue sampling box as declared by ciw.lab.surfaces.SAMPLING_DOMAINS."""
    (low1, high1), (low2, high2) = sampling_domain(key)
    return Domain((low1, low2), (high1, high2), scale)


BOX = Domain((-2.0, -2.0), (2.0, 2.0))
# Catalogue boxes come from the core; only the local length scale of the
# hyperbolic plane and the section's derived surfaces are declared here.
DOMAINS = {key: _core_domain(key, _hyperbolic_scale if key == "hyperbolic-plane" else None)
           for key in ("plane", "sphere", "cylinder", "saddle", "torus", "gaussian-bump", "hyperbolic-plane")}
DOMAINS.update({
    "plane-polar": Domain((0.3, -math.pi), (2.0, math.pi)),
    "gaussian-bump-shear": BOX,
    "rotated-torus": DOMAINS["torus"],
})
TORUS_ROTATION = rotation_matrix([1.0, 2.0, 3.0], 0.7)


def conformance_surfaces() -> dict:
    """Every catalogue surface plus two reparametrized charts and a rigid rotation, in report order."""
    surfaces = dict(catalogue())
    surfaces["plane-polar"] = Reparametrized(surfaces["plane"], PolarChart())
    surfaces["gaussian-bump-shear"] = Reparametrized(surfaces["gaussian-bump"], ShearChart(0.3))
    surfaces["rotated-torus"] = Rotated(surfaces["torus"], TORUS_ROTATION)
    return surfaces


def mutant_surfaces() -> dict:
    """Seeded defects, each paired with the declared domain it is sampled on."""
    return {
        "misscaled-curvature": (MisscaledSphere(2.0), DOMAINS["sphere"]),
        "transposed-derivatives": (TransposedTorus(2.0, 1.0), DOMAINS["torus"]),
        "sign-flipped-derivatives": (SignFlippedSaddle(1.0), DOMAINS["saddle"]),
        "dropped-cross-term": (DroppedCrossTermBump(0.5, 1.0), DOMAINS["gaussian-bump"]),
        "nan-derivatives": (NaNDerivativeTorus(2.0, 1.0), DOMAINS["torus"]),
        "indefinite-metric": (LorentzianChart(), BOX),
        "nonsymmetric-metric": (NonsymmetricMetric(), BOX),
    }


# ---------------------------------------------------------- stencils
def stencil_derivative(function, u, k, h):
    """Fourth-order central difference d_k function(u) with step h (arrays allowed)."""
    u = np.asarray(u, dtype=float)
    step = np.zeros(2)
    step[k] = h
    total = None
    for multiple, weight in STENCIL5:
        term = weight * np.asarray(function(u + multiple * step), dtype=float)
        total = term if total is None else total + term
    return total / h


def central_difference(function, u, k, h):
    """Second-order central difference (f(u + h e_k) - f(u - h e_k)) / 2h."""
    u = np.asarray(u, dtype=float)
    step = np.zeros(2)
    step[k] = h
    return (np.asarray(function(u + step), dtype=float) - np.asarray(function(u - step), dtype=float)) / (2.0 * h)


def metric_second_derivatives(surface: Surface, u, h: float) -> np.ndarray:
    """d2g[l, k, i, j] = d_l d_k g_ij by fourth-order differences of the exact first derivatives."""
    return np.stack([stencil_derivative(surface.metric_derivatives, u, m, h) for m in range(2)])


def _det3(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def brioschi(g, dg, d2g) -> float:
    """Gaussian curvature from the metric alone (Theorema Egregium, Brioschi form).

    ``dg[k, i, j] = d_k g_ij`` and ``d2g[l, k, i, j] = d_l d_k g_ij``; the mixed
    derivative F_uv is the average of both orders.
    """
    e, f, g2 = g[0][0], g[0][1], g[1][1]
    eu, ev = dg[0][0][0], dg[1][0][0]
    fu, fv = dg[0][0][1], dg[1][0][1]
    gu, gv = dg[0][1][1], dg[1][1][1]
    evv = d2g[1][1][0][0]
    guu = d2g[0][0][1][1]
    fuv = 0.5 * (d2g[0][1][0][1] + d2g[1][0][0][1])
    first = [[-0.5 * evv + fuv - 0.5 * guu, 0.5 * eu, fu - 0.5 * ev],
             [fv - 0.5 * gu, e, f],
             [0.5 * gv, f, g2]]
    second = [[0.0, 0.5 * ev, 0.5 * gu], [0.5 * ev, e, f], [0.5 * gu, f, g2]]
    return float((_det3(first) - _det3(second)) / (e * g2 - f * f) ** 2)


def christoffel_from(g, dg) -> np.ndarray:
    """Gamma[k, i, j] from explicit loops (a second implementation of the core einsum)."""
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    ginv = [[g[1][1] / det, -g[0][1] / det], [-g[1][0] / det, g[0][0] / det]]
    gamma = np.zeros((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gamma[k, i, j] = 0.5 * sum(ginv[k][m] * (dg[i][j][m] + dg[j][i][m] - dg[m][i][j]) for m in range(2))
    return gamma


# ------------------------------------------------------- conformance
def conformance_point(surface: Surface, u, length: float) -> dict:
    """Normalized residuals of every interface identity at one point."""
    u = np.asarray(u, dtype=float)
    g = np.asarray(surface.metric(u), dtype=float)
    dg = np.asarray(surface.metric_derivatives(u), dtype=float)
    g_scale = float(np.max(np.abs(g)))
    dg_scale = float(np.max(np.abs(dg))) + g_scale / length
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    result = {"metric_asymmetry": abs(g[0, 1] - g[1, 0]) / g_scale,
              "min_eigenvalue_ratio": float(eig[0] / eig[1]) if eig[1] > 0 else -math.inf,
              "derivative_asymmetry": float(np.max(np.abs(dg[:, 0, 1] - dg[:, 1, 0]))) / dg_scale}
    if not eig[0] > 0:
        # An indefinite metric has no Levi-Civita connection to compare: the
        # remaining identities are reported as not evaluated, not as passed.
        result.update({name: None for name in ("christoffel_asymmetry", "compatibility",
                                               "derivative_consistency", "mixed_partials", "gauss_equation")})
        return result
    gamma = surface.christoffel(u)
    gamma_scale = float(np.max(np.abs(gamma))) + 1.0 / length
    result["christoffel_asymmetry"] = float(np.max(np.abs(gamma - np.transpose(gamma, (0, 2, 1))))) / gamma_scale
    # d_k g_ij = Gamma^l_ki g_lj + Gamma^l_kj g_il (Levi-Civita compatibility)
    compat = dg - np.einsum("lki,lj->kij", gamma, g) - np.einsum("lkj,il->kij", gamma, g)
    result["compatibility"] = float(np.max(np.abs(compat))) / dg_scale
    h = STENCIL_STEP * length
    fd = np.stack([stencil_derivative(surface.metric, u, k, h) for k in range(2)])
    result["derivative_consistency"] = float(np.max(np.abs(fd - dg))) / dg_scale
    d2g = metric_second_derivatives(surface, u, h)
    d2_scale = float(np.max(np.abs(d2g))) + dg_scale / length
    result["mixed_partials"] = float(np.max(np.abs(d2g[0, 1] - d2g[1, 0]))) / d2_scale
    curvature = float(surface.gaussian_curvature(u))
    result["gauss_equation"] = abs(brioschi(g, dg, d2g) - curvature) / (abs(curvature) + length ** -2)
    result["curvature"] = curvature
    return result


def conformance(surface: Surface, domain: Domain, count: int = 32, seed: int = SEED) -> dict:
    """Worst normalized residual of each identity over seeded domain points, and failed checks.

    An identity that could not be evaluated at some point (indefinite metric)
    is listed under ``not_evaluated`` and never counts as passed. A NaN or
    infinite residual at any point fails its identity (``nonfinite``): Python
    comparisons with NaN are false, so a running max would silently drop it.
    An exception while evaluating a point fails the surface (``errors``).
    """
    points = domain.sample(count, seed)
    worst = {name: None for name in THRESHOLDS}
    nonfinite, errors = set(), []
    refused = 0
    for u in points:
        try:
            surface.check(u)
        except SurfaceRefusal:
            refused += 1
        try:
            row = conformance_point(surface, u, domain.length(u))
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        for name in THRESHOLDS:
            value = row[name]
            if value is None:
                continue
            if not math.isfinite(value):
                nonfinite.add(name)
                continue
            pick = min if name == "min_eigenvalue_ratio" else max
            worst[name] = value if worst[name] is None else pick(worst[name], value)
    failed = [name for name, bound in THRESHOLDS.items() if name in nonfinite or (
        worst[name] is not None
        and (worst[name] < bound if name == "min_eigenvalue_ratio" else not worst[name] <= bound))]
    not_evaluated = [name for name in THRESHOLDS if worst[name] is None and name not in nonfinite]
    return {"points": int(count), "seed": int(seed), "refused_by_core_check": refused,
            "worst": {k: "nonfinite" if k in nonfinite else (None if v is None else float(v))
                      for k, v in worst.items()},
            "failed": failed, "not_evaluated": not_evaluated, "nonfinite": sorted(nonfinite),
            "errors": sorted(set(errors)), "conforms": not failed and not not_evaluated and not errors}
