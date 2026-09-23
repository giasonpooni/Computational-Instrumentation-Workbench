"""Reusable metric, Christoffel and curvature interface for two-dimensional surfaces.

A surface is described in one chart by its metric ``g_ij(u)`` and the exact
first derivatives ``dg[k, i, j] = d_k g_ij``. Christoffel symbols and the
geodesic equation follow generically. Embedded surfaces derive both from exact
embedding derivatives, so no finite differences enter the equations of motion.
Gaussian curvature is supplied exactly (second fundamental form or a closed
form) so that Jacobi integration does not inherit derivative error.

Coordinates, lengths and curvature are in declared normalized units; none of
these objects represents a measured physical surface.
"""
from __future__ import annotations

import math

import numpy as np

# Coordinate regularity threshold on det(g), relative to the surface scale.
SINGULAR_DET = 1e-12


class SurfaceRefusal(ValueError):
    """A point is outside the chart or the chart is singular there; ``code`` names the reason."""

    def __init__(self, message: str, code: str = "surface_refused"):
        super().__init__(message)
        self.code = code


class Surface:
    """Chart-level interface. Subclasses implement metric, metric_derivatives and gaussian_curvature."""

    name = "surface"
    exact_geodesics = False

    def metric(self, u) -> np.ndarray:
        raise NotImplementedError

    def metric_derivatives(self, u) -> np.ndarray:
        raise NotImplementedError

    def gaussian_curvature(self, u) -> float:
        raise NotImplementedError

    def embedding(self, u):
        """Point in R^3, or None for an intrinsic (non-embedded) chart."""
        return None

    def embedding_jacobian(self, u):
        return None

    def describe(self) -> dict:
        return {"name": self.name}

    # Generic geometry -------------------------------------------------
    def check(self, u) -> None:
        """Refuse nonfinite points and coordinate singularities (degenerate metric)."""
        u = np.asarray(u, dtype=float)
        if u.shape != (2,) or not np.all(np.isfinite(u)):
            raise SurfaceRefusal("Surface coordinates must be two finite numbers", "nonfinite_point")
        g = self.metric(u)
        if not np.all(np.isfinite(g)) or np.linalg.det(g) <= SINGULAR_DET * max(1.0, float(np.trace(g))) ** 2:
            raise SurfaceRefusal(f"{self.name}: coordinate singularity or degenerate metric at {u.tolist()}",
                                 "degenerate_metric")

    def christoffel(self, u) -> np.ndarray:
        """Gamma[k, i, j] = 1/2 g^{kl} (d_i g_jl + d_j g_il - d_l g_ij)."""
        g = self.metric(u)
        dg = self.metric_derivatives(u)
        ginv = np.linalg.inv(g)
        # lowered[l, i, j] = d_i g_jl + d_j g_il - d_l g_ij
        lowered = np.einsum("ijl->lij", dg) + np.einsum("jil->lij", dg) - dg
        return 0.5 * np.einsum("kl,lij->kij", ginv, lowered)

    def geodesic_rhs(self, y) -> np.ndarray:
        """State y = (u1, u2, v1, v2): u' = v, v'^k = -Gamma^k_ij v^i v^j."""
        u, v = y[:2], y[2:4]
        gamma = self.christoffel(u)
        return np.concatenate([v, -np.einsum("kij,i,j->k", gamma, v, v)])

    def speed_squared(self, u, v) -> float:
        return float(v @ self.metric(u) @ v)

    def normal(self, u, t) -> np.ndarray:
        """Unit tangent vector rotated +90 degrees by the metric orientation."""
        g = self.metric(u)
        lowered = g @ t
        return np.array([-lowered[1], lowered[0]]) / math.sqrt(np.linalg.det(g))

    def orthonormal_frame(self, u) -> tuple[np.ndarray, np.ndarray]:
        """(e1, e2): e1 along the first coordinate direction, e2 its +90 degree normal."""
        g = self.metric(u)
        e1 = np.array([1.0, 0.0]) / math.sqrt(g[0, 0])
        return e1, self.normal(u, e1)

    def unit_tangent(self, u, heading: float) -> np.ndarray:
        """Unit tangent at angle ``heading`` (radians) from the first coordinate direction."""
        e1, e2 = self.orthonormal_frame(u)
        return math.cos(heading) * e1 + math.sin(heading) * e2

    def inner(self, u, a, b) -> float:
        return float(a @ self.metric(u) @ b)


class EmbeddedSurface(Surface):
    """Surface X(u) in R^3 with exact first and second embedding derivatives."""

    def first(self, u) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def second(self, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def embedding_jacobian(self, u):
        xu, xv = self.first(u)
        return np.column_stack([xu, xv])

    def metric(self, u):
        xu, xv = self.first(u)
        return np.array([[xu @ xu, xu @ xv], [xv @ xu, xv @ xv]])

    def metric_derivatives(self, u):
        xu, xv = self.first(u)
        xuu, xuv, xvv = self.second(u)
        first = (xu, xv)
        second = ((xuu, xuv), (xuv, xvv))
        dg = np.empty((2, 2, 2))
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    dg[k, i, j] = second[k][i] @ first[j] + first[i] @ second[k][j]
        return dg

    def unit_normal3(self, u):
        xu, xv = self.first(u)
        n = np.cross(xu, xv)
        return n / np.linalg.norm(n)

    def gaussian_curvature(self, u):
        xuu, xuv, xvv = self.second(u)
        n = self.unit_normal3(u)
        g = self.metric(u)
        return float(((xuu @ n) * (xvv @ n) - (xuv @ n) ** 2) / np.linalg.det(g))


class Plane(EmbeddedSurface):
    name = "plane"
    exact_geodesics = True

    def embedding(self, u):
        return np.array([u[0], u[1], 0.0])

    def first(self, u):
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    def second(self, u):
        zero = np.zeros(3)
        return zero, zero, zero

    def gaussian_curvature(self, u):
        return 0.0

    def exact_geodesic(self, u0, t0, s):
        s = np.asarray(s, dtype=float)
        return u0[None, :] + s[:, None] * t0[None, :]


class Sphere(EmbeddedSurface):
    """Polar chart u = (theta, phi); singular at theta = 0 and pi."""

    name = "sphere"
    exact_geodesics = True

    def __init__(self, radius=1.0):
        self.radius = float(radius)

    def describe(self):
        return {"name": self.name, "radius": self.radius, "chart": "polar (theta, phi)"}

    def embedding(self, u):
        t, p = u
        return self.radius * np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])

    def first(self, u):
        t, p = u
        r = self.radius
        return (r * np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), -math.sin(t)]),
                r * np.array([-math.sin(t) * math.sin(p), math.sin(t) * math.cos(p), 0.0]))

    def second(self, u):
        t, p = u
        r = self.radius
        return (r * np.array([-math.sin(t) * math.cos(p), -math.sin(t) * math.sin(p), -math.cos(t)]),
                r * np.array([-math.cos(t) * math.sin(p), math.cos(t) * math.cos(p), 0.0]),
                r * np.array([-math.sin(t) * math.cos(p), -math.sin(t) * math.sin(p), 0.0]))

    def gaussian_curvature(self, u):
        return 1.0 / self.radius ** 2

    def exact_embedded_geodesic(self, u0, t0, s):
        """Great circle X(s) = cos(s/R) X0 + R sin(s/R) T0 in R^3."""
        s = np.asarray(s, dtype=float)
        x0 = self.embedding(u0)
        tangent = self.embedding_jacobian(u0) @ t0
        r = self.radius
        return np.cos(s / r)[:, None] * x0[None, :] + (r * np.sin(s / r))[:, None] * tangent[None, :]


class Cylinder(EmbeddedSurface):
    """Chart u = (phi, z); intrinsically flat, extrinsically curved."""

    name = "cylinder"
    exact_geodesics = True

    def __init__(self, radius=1.0):
        self.radius = float(radius)

    def describe(self):
        return {"name": self.name, "radius": self.radius, "chart": "(phi, z)"}

    def embedding(self, u):
        p, z = u
        return np.array([self.radius * math.cos(p), self.radius * math.sin(p), z])

    def first(self, u):
        p, _ = u
        return np.array([-self.radius * math.sin(p), self.radius * math.cos(p), 0.0]), np.array([0, 0, 1.0])

    def second(self, u):
        p, _ = u
        zero = np.zeros(3)
        return np.array([-self.radius * math.cos(p), -self.radius * math.sin(p), 0.0]), zero, zero

    def gaussian_curvature(self, u):
        return 0.0

    def exact_geodesic(self, u0, t0, s):
        """Helices are straight lines in the (phi, z) chart, whose metric is constant."""
        s = np.asarray(s, dtype=float)
        return u0[None, :] + s[:, None] * t0[None, :]


class MongeSurface(EmbeddedSurface):
    """Graph z = f(x, y) with exact derivatives supplied by subclasses."""

    def height(self, u):
        raise NotImplementedError

    def height_derivatives(self, u):
        """Return (f_x, f_y, f_xx, f_xy, f_yy)."""
        raise NotImplementedError

    def embedding(self, u):
        return np.array([u[0], u[1], self.height(u)])

    def first(self, u):
        fx, fy, *_ = self.height_derivatives(u)
        return np.array([1.0, 0.0, fx]), np.array([0.0, 1.0, fy])

    def second(self, u):
        _, _, fxx, fxy, fyy = self.height_derivatives(u)
        return np.array([0, 0, fxx]), np.array([0, 0, fxy]), np.array([0, 0, fyy])

    def gaussian_curvature(self, u):
        fx, fy, fxx, fxy, fyy = self.height_derivatives(u)
        return float((fxx * fyy - fxy ** 2) / (1 + fx ** 2 + fy ** 2) ** 2)


class Saddle(MongeSurface):
    """Hyperbolic paraboloid z = c (x^2 - y^2) / 2; K(0) = -c^2."""

    name = "saddle"

    def __init__(self, c=1.0):
        self.c = float(c)

    def describe(self):
        return {"name": self.name, "c": self.c, "graph": "z = c (x^2 - y^2) / 2"}

    def height(self, u):
        return 0.5 * self.c * (u[0] ** 2 - u[1] ** 2)

    def height_derivatives(self, u):
        c = self.c
        return c * u[0], -c * u[1], c, 0.0, -c


class GaussianBump(MongeSurface):
    """z = h exp(-(x^2 + y^2) / (2 sigma^2)): positive curvature on top, negative on the flank."""

    name = "gaussian-bump"

    def __init__(self, height=0.5, sigma=1.0):
        self.h, self.sigma = float(height), float(sigma)

    def describe(self):
        return {"name": self.name, "height": self.h, "sigma": self.sigma}

    def height(self, u):
        return self.h * math.exp(-(u[0] ** 2 + u[1] ** 2) / (2 * self.sigma ** 2))

    def height_derivatives(self, u):
        x, y = u
        s2 = self.sigma ** 2
        f = self.height(u)
        fx, fy = -x / s2 * f, -y / s2 * f
        fxx = (x * x / s2 - 1) / s2 * f
        fyy = (y * y / s2 - 1) / s2 * f
        fxy = x * y / (s2 * s2) * f
        return fx, fy, fxx, fxy, fyy


class Torus(EmbeddedSurface):
    """Chart u = (phi, theta): K = cos(theta) / (r (R + r cos(theta))) changes sign."""

    name = "torus"

    def __init__(self, major=2.0, minor=1.0):
        if not major > minor > 0:
            raise SurfaceRefusal("Torus requires major > minor > 0")
        self.major, self.minor = float(major), float(minor)

    def describe(self):
        return {"name": self.name, "major": self.major, "minor": self.minor, "chart": "(phi, theta)"}

    def _rho(self, theta):
        return self.major + self.minor * math.cos(theta)

    def embedding(self, u):
        p, t = u
        rho = self._rho(t)
        return np.array([rho * math.cos(p), rho * math.sin(p), self.minor * math.sin(t)])

    def first(self, u):
        p, t = u
        rho, r = self._rho(t), self.minor
        return (np.array([-rho * math.sin(p), rho * math.cos(p), 0.0]),
                np.array([-r * math.sin(t) * math.cos(p), -r * math.sin(t) * math.sin(p), r * math.cos(t)]))

    def second(self, u):
        p, t = u
        rho, r = self._rho(t), self.minor
        return (np.array([-rho * math.cos(p), -rho * math.sin(p), 0.0]),
                np.array([r * math.sin(t) * math.sin(p), -r * math.sin(t) * math.cos(p), 0.0]),
                np.array([-r * math.cos(t) * math.cos(p), -r * math.cos(t) * math.sin(p), -r * math.sin(t)]))

    def gaussian_curvature(self, u):
        t = u[1]
        return math.cos(t) / (self.minor * self._rho(t))

    def clairaut(self, u, v) -> float:
        """rho^2 dphi/ds, conserved along every geodesic of a surface of revolution."""
        return self._rho(u[1]) ** 2 * v[0]


class HyperbolicPlane(Surface):
    """Upper half-plane with g = I / (k^2 y^2): constant curvature -k^2, exact geodesics."""

    name = "hyperbolic-plane"
    exact_geodesics = True

    def __init__(self, k=1.0):
        if not k > 0:
            raise SurfaceRefusal("Hyperbolic curvature scale must be positive")
        self.k = float(k)

    def describe(self):
        return {"name": self.name, "k": self.k, "chart": "upper half-plane (x, y > 0)"}

    def check(self, u):
        if not (np.all(np.isfinite(u)) and u[1] > 0):
            raise SurfaceRefusal("Hyperbolic chart requires y > 0", "outside_chart")

    def metric(self, u):
        return np.eye(2) / (self.k * u[1]) ** 2

    def metric_derivatives(self, u):
        dg = np.zeros((2, 2, 2))
        dg[1] = -2.0 * np.eye(2) / (self.k ** 2 * u[1] ** 3)
        return dg

    def gaussian_curvature(self, u):
        return -self.k ** 2

    def exact_geodesic(self, u0, t0, s):
        """Semicircles orthogonal to y = 0 (or vertical lines), parameterized by arclength."""
        s = np.asarray(s, dtype=float)
        x0, y0 = float(u0[0]), float(u0[1])
        direction = t0 / np.linalg.norm(t0)
        cos_a, sin_a = float(direction[0]), float(direction[1])
        if abs(cos_a) < 1e-15:
            y = y0 * np.exp(math.copysign(1.0, sin_a) * self.k * s)
            return np.column_stack([np.full_like(s, x0), y])
        center = x0 + y0 * sin_a / cos_a
        radius = y0 / abs(cos_a)
        t_start = math.atanh((x0 - center) / radius)
        t = t_start + math.copysign(1.0, cos_a) * self.k * s
        return np.column_stack([center + radius * np.tanh(t), radius / np.cosh(t)])


class ChartMap:
    """Nonlinear parameterization a -> u = phi(a) with exact Jacobian and Hessian.

    ``hessian(a)[p, i, k]`` is d^2 phi^p / (da^i da^k).
    """

    name = "chart"

    def forward(self, a):
        raise NotImplementedError

    def jacobian(self, a):
        raise NotImplementedError

    def hessian(self, a):
        raise NotImplementedError

    def inverse(self, u):
        raise NotImplementedError


class Reparametrized(Surface):
    """The same surface seen through a chart map: g'(a) = J^T g(phi(a)) J."""

    def __init__(self, base: Surface, chart: ChartMap):
        self.base, self.chart = base, chart
        self.name = f"{base.name}∘{chart.name}"

    def describe(self):
        return {"name": self.name, "base": self.base.describe(), "chart": self.chart.name}

    def metric(self, a):
        jac = self.chart.jacobian(a)
        return jac.T @ self.base.metric(self.chart.forward(a)) @ jac

    def metric_derivatives(self, a):
        u = self.chart.forward(a)
        jac, hess = self.chart.jacobian(a), self.chart.hessian(a)
        g, dg = self.base.metric(u), self.base.metric_derivatives(u)
        # d_k g_pq(phi(a)) = sum_r dg[r, p, q] J[r, k]
        dg_a = np.einsum("rpq,rk->kpq", dg, jac)
        out = np.empty((2, 2, 2))
        for k in range(2):
            djk = hess[:, :, k]  # d_k J[p, i]
            out[k] = djk.T @ g @ jac + jac.T @ dg_a[k] @ jac + jac.T @ g @ djk
        return out

    def gaussian_curvature(self, a):
        return self.base.gaussian_curvature(self.chart.forward(a))

    def embedding(self, a):
        return self.base.embedding(self.chart.forward(a))

    def embedding_jacobian(self, a):
        base = self.base.embedding_jacobian(self.chart.forward(a))
        return None if base is None else base @ self.chart.jacobian(a)


class Rotated(EmbeddedSurface):
    """Rigid rotation of an embedded surface: intrinsic quantities are unchanged."""

    def __init__(self, base: EmbeddedSurface, rotation):
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) \
                or np.linalg.det(rotation) <= 0:
            raise SurfaceRefusal("Frame change requires a proper rotation matrix")
        self.base, self.rotation = base, rotation
        self.name = f"rotated-{base.name}"

    def embedding(self, u):
        return self.rotation @ self.base.embedding(u)

    def first(self, u):
        return tuple(self.rotation @ x for x in self.base.first(u))

    def second(self, u):
        return tuple(self.rotation @ x for x in self.base.second(u))


def rotation_matrix(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


# Declared sampling boxes (u1, u2 ranges) inside each catalogue chart, away from
# coordinate singularities, for experiments that draw seeded sample points.
SAMPLING_DOMAINS = {
    "plane": ((-2.0, 2.0), (-2.0, 2.0)),
    "sphere": ((0.3, math.pi - 0.3), (-math.pi, math.pi)),
    "cylinder": ((-math.pi, math.pi), (-2.0, 2.0)),
    "saddle": ((-1.5, 1.5), (-1.5, 1.5)),
    "torus": ((-math.pi, math.pi), (-math.pi, math.pi)),
    "gaussian-bump": ((-2.5, 2.5), (-2.5, 2.5)),
    "hyperbolic-plane": ((-2.0, 2.0), (0.3, 3.0)),
}


def sampling_domain(name: str):
    """The declared sampling box for a catalogue surface."""
    if name not in SAMPLING_DOMAINS:
        raise SurfaceRefusal(f"No declared sampling domain for {name}", "unknown_surface")
    return SAMPLING_DOMAINS[name]


def catalogue() -> dict:
    """The surfaces used by the geodesic/Jacobi experiments, with declared parameters."""
    return {"plane": Plane(), "sphere": Sphere(1.0), "cylinder": Cylinder(1.0), "saddle": Saddle(1.0),
            "torus": Torus(2.0, 1.0), "gaussian-bump": GaussianBump(0.5, 1.0),
            "hyperbolic-plane": HyperbolicPlane(1.0)}
