"""Native geometry kernel for smooth two-dimensional surfaces.

Supported objects: plane, sphere (two-chart atlas), cylinder, torus, graph
surfaces ``z = f(x, y)`` and conformal metrics ``g = exp(2 sigma) I`` with
declared polynomials, and the hyperbolic half-plane (a metric without an
embedding). Each surface supplies its metric, Christoffel symbols and Gaussian
curvature analytically; ``curvature_from_christoffel`` recomputes curvature by
finite differences as an independent check.

Geodesics integrate ``u''^k = -Gamma^k_ij u'^i u'^j`` with classical RK4 at a
fixed step. Chart quality is monitored; on the sphere the integrator changes
charts through the embedding before the pole singularity rather than stepping
through it. Surfaces with a single chart refuse a trajectory that reaches a
singular metric. ``extrinsic_geodesic`` integrates the same curve in R^3 from
the implicit surface equation, without Christoffel symbols, as an independent
implementation. Scalar Jacobi fields solve ``j'' + K j = 0`` along unit-speed
geodesics and report the first conjugate point.

Limits: binary64 only; fixed-step RK4 carries O(h^4) global error that the
oracle layer estimates by step halving; log maps use Newton shooting and are
refused near conjugate points; mesh surfaces live in ``mesh.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from ._common import Refusal, finite, integer, mapping, plain, require_keys, text, vector
from .units import Quantity, require_dimension

MAX_STEPS = 200_000
QUALITY_SWITCH = 0.3
SINGULAR_QUALITY = 1e-6
MAX_POLYNOMIAL_TERMS = 32


# ---------------------------------------------------------------- charts
class Chart:
    """A coordinate chart: metric, Christoffel symbols and (optionally) an embedding."""

    name = "chart"

    def metric(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def christoffel(self, u: np.ndarray) -> np.ndarray:
        """``Gamma[k, i, j]``."""
        raise NotImplementedError

    def curvature(self, u: np.ndarray) -> float:
        return curvature_from_christoffel(self, u)

    def embed(self, u: np.ndarray) -> np.ndarray | None:
        return None

    def jacobian(self, u: np.ndarray) -> np.ndarray | None:
        return None

    def invert(self, x: np.ndarray) -> np.ndarray | None:
        return None

    def quality(self, u: np.ndarray) -> float:
        """A scale-free measure of how far ``u`` is from a coordinate singularity (1 = good)."""
        g = self.metric(u)
        trace = float(np.trace(g))
        return 0.0 if trace <= 0 else float(2.0 * math.sqrt(max(np.linalg.det(g), 0.0)) / trace)


def curvature_from_christoffel(chart: Chart, u: np.ndarray, h: float = 1e-5) -> float:
    """Gaussian curvature from Riemann ``R^l_{ijk}`` with finite-differenced Christoffels."""
    u = np.asarray(u, dtype=float)
    gamma = chart.christoffel(u)
    derivative = np.zeros((2, 2, 2, 2))  # [m, k, i, j] = d_m Gamma^k_ij
    for m in range(2):
        step = np.zeros(2)
        step[m] = h * max(1.0, abs(u[m]))
        derivative[m] = (chart.christoffel(u + step) - chart.christoffel(u - step)) / (2 * step[m])
    # R^l_{ijk} = d_j G^l_{ik} - d_k G^l_{ij} + G^l_{jm} G^m_{ik} - G^l_{km} G^m_{ij}; K = g_{0l} R^l_{101}/det g
    riemann = np.zeros(2)
    i, j, k = 1, 0, 1
    for l in range(2):
        riemann[l] = (derivative[j, l, i, k] - derivative[k, l, i, j]
                      + sum(gamma[l, j, m] * gamma[m, i, k] - gamma[l, k, m] * gamma[m, i, j] for m in range(2)))
    g = chart.metric(u)
    return float(g[0] @ riemann / np.linalg.det(g))


class PlaneChart(Chart):
    name = "cartesian"

    def metric(self, u):
        return np.eye(2)

    def christoffel(self, u):
        return np.zeros((2, 2, 2))

    def curvature(self, u):
        return 0.0

    def embed(self, u):
        return np.array([u[0], u[1], 0.0])

    def jacobian(self, u):
        return np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def invert(self, x):
        return np.array([x[0], x[1]])


class SphereChart(Chart):
    """Polar angle ``theta`` and azimuth ``phi`` about the rotated axis ``rotation @ e_z``."""

    def __init__(self, radius: float, rotation: np.ndarray | None = None, name: str = "polar-z"):
        self.radius = radius
        self.rotation = np.eye(3) if rotation is None else rotation
        self.name = name

    def metric(self, u):
        return self.radius ** 2 * np.diag([1.0, math.sin(u[0]) ** 2])

    def christoffel(self, u):
        s, c = math.sin(u[0]), math.cos(u[0])
        gamma = np.zeros((2, 2, 2))
        gamma[0, 1, 1] = -s * c
        if abs(s) > 0:
            gamma[1, 0, 1] = gamma[1, 1, 0] = c / s
        else:
            gamma[1, 0, 1] = gamma[1, 1, 0] = math.inf
        return gamma

    def curvature(self, u):
        return 1.0 / self.radius ** 2

    def embed(self, u):
        t, p = u
        return self.radius * self.rotation @ np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])

    def jacobian(self, u):
        t, p = u
        local = np.array([[math.cos(t) * math.cos(p), -math.sin(t) * math.sin(p)],
                          [math.cos(t) * math.sin(p), math.sin(t) * math.cos(p)],
                          [-math.sin(t), 0.0]])
        return self.radius * self.rotation @ local

    def invert(self, x):
        local = self.rotation.T @ np.asarray(x, dtype=float) / self.radius
        return np.array([math.acos(max(-1.0, min(1.0, local[2]))), math.atan2(local[1], local[0])])

    def quality(self, u):
        return abs(math.sin(u[0]))


class CylinderChart(Chart):
    """Azimuth ``phi`` (rad) and axial ``z``; intrinsically flat (K = 0)."""

    name = "phi-z"

    def __init__(self, radius: float):
        self.radius = radius

    def metric(self, u):
        return np.diag([self.radius ** 2, 1.0])

    def christoffel(self, u):
        return np.zeros((2, 2, 2))

    def curvature(self, u):
        return 0.0

    def embed(self, u):
        return np.array([self.radius * math.cos(u[0]), self.radius * math.sin(u[0]), u[1]])

    def jacobian(self, u):
        return np.array([[-self.radius * math.sin(u[0]), 0.0], [self.radius * math.cos(u[0]), 0.0], [0.0, 1.0]])

    def invert(self, x):
        return np.array([math.atan2(x[1], x[0]), x[2]])

    def quality(self, u):
        return 1.0


class TorusChart(Chart):
    """``phi`` about the z axis and ``theta`` about the tube; ``K = cos(theta)/(r (R + r cos(theta)))``."""

    name = "phi-theta"

    def __init__(self, major: float, minor: float):
        self.major, self.minor = major, minor

    def _rho(self, u):
        return self.major + self.minor * math.cos(u[1])

    def metric(self, u):
        return np.diag([self._rho(u) ** 2, self.minor ** 2])

    def christoffel(self, u):
        rho, s = self._rho(u), math.sin(u[1])
        gamma = np.zeros((2, 2, 2))
        gamma[0, 0, 1] = gamma[0, 1, 0] = -self.minor * s / rho
        gamma[1, 0, 0] = rho * s / self.minor
        return gamma

    def curvature(self, u):
        return math.cos(u[1]) / (self.minor * self._rho(u))

    def embed(self, u):
        rho = self._rho(u)
        return np.array([rho * math.cos(u[0]), rho * math.sin(u[0]), self.minor * math.sin(u[1])])

    def jacobian(self, u):
        rho, (p, t) = self._rho(u), u
        return np.array([[-rho * math.sin(p), -self.minor * math.sin(t) * math.cos(p)],
                         [rho * math.cos(p), -self.minor * math.sin(t) * math.sin(p)],
                         [0.0, self.minor * math.cos(t)]])

    def invert(self, x):
        rho = math.hypot(x[0], x[1])
        return np.array([math.atan2(x[1], x[0]), math.atan2(x[2], rho - self.major)])

    def quality(self, u):
        return 1.0


@dataclass(frozen=True)
class Polynomial:
    """``sum c * x^px * y^py`` with analytic derivatives up to second order."""

    terms: tuple[tuple[float, int, int], ...]

    @classmethod
    def from_json(cls, value: Any, name: str) -> "Polynomial":
        if not isinstance(value, list) or not value or len(value) > MAX_POLYNOMIAL_TERMS:
            raise Refusal("malformed_record", f"{name} must list 1..{MAX_POLYNOMIAL_TERMS} terms")
        terms = []
        for index, term in enumerate(value):
            require_keys(term, f"{name}[{index}]", {"c", "px", "py"})
            terms.append((finite(term["c"], f"{name}[{index}].c"), integer(term["px"], "px", minimum=0, maximum=8),
                          integer(term["py"], "py", minimum=0, maximum=8)))
        return cls(tuple(terms))

    def derivative(self, x: float, y: float, dx: int = 0, dy: int = 0) -> float:
        total = 0.0
        for c, px, py in self.terms:
            if px < dx or py < dy:
                continue
            coefficient = c * math.perm(px, dx) * math.perm(py, dy)
            total += coefficient * x ** (px - dx) * y ** (py - dy)
        return total

    def to_json(self) -> list:
        return [{"c": c, "px": px, "py": py} for c, px, py in self.terms]


class GraphChart(Chart):
    """``X(x, y) = (x, y, f(x, y))``: a non-constant-curvature embedded surface."""

    name = "graph-xy"

    def __init__(self, height: Polynomial):
        self.height = height

    def _derivatives(self, u):
        f = self.height.derivative
        x, y = u
        return (f(x, y, 1, 0), f(x, y, 0, 1), f(x, y, 2, 0), f(x, y, 1, 1), f(x, y, 0, 2))

    def metric(self, u):
        fx, fy, *_ = self._derivatives(u)
        return np.array([[1 + fx * fx, fx * fy], [fx * fy, 1 + fy * fy]])

    def christoffel(self, u):
        fx, fy, fxx, fxy, fyy = self._derivatives(u)
        w = 1.0 + fx * fx + fy * fy
        second = np.array([[fxx, fxy], [fxy, fyy]])
        gradient = np.array([fx, fy])
        # Gamma^k_ij = f_ij f_k / (1 + |grad f|^2)
        return np.einsum("k,ij->kij", gradient, second) / w

    def curvature(self, u):
        fx, fy, fxx, fxy, fyy = self._derivatives(u)
        return (fxx * fyy - fxy * fxy) / (1 + fx * fx + fy * fy) ** 2

    def embed(self, u):
        return np.array([u[0], u[1], self.height.derivative(u[0], u[1])])

    def jacobian(self, u):
        fx, fy, *_ = self._derivatives(u)
        return np.array([[1.0, 0.0], [0.0, 1.0], [fx, fy]])

    def invert(self, x):
        return np.array([x[0], x[1]])


class ConformalChart(Chart):
    """``g = exp(2 sigma(x, y)) I``; no embedding is claimed. ``K = -exp(-2 sigma) Laplacian(sigma)``."""

    name = "conformal-xy"

    def __init__(self, sigma: Polynomial):
        self.sigma = sigma

    def metric(self, u):
        return math.exp(2 * self.sigma.derivative(u[0], u[1])) * np.eye(2)

    def christoffel(self, u):
        sx, sy = self.sigma.derivative(u[0], u[1], 1, 0), self.sigma.derivative(u[0], u[1], 0, 1)
        gamma = np.zeros((2, 2, 2))
        gamma[0, 0, 0], gamma[0, 1, 1], gamma[0, 0, 1], gamma[0, 1, 0] = sx, -sx, sy, sy
        gamma[1, 1, 1], gamma[1, 0, 0], gamma[1, 0, 1], gamma[1, 1, 0] = sy, -sy, sx, sx
        return gamma

    def curvature(self, u):
        x, y = u
        laplacian = self.sigma.derivative(x, y, 2, 0) + self.sigma.derivative(x, y, 0, 2)
        return -math.exp(-2 * self.sigma.derivative(x, y)) * laplacian


class HalfPlaneChart(Chart):
    """Poincaré half-plane ``g = I / y^2`` (``y > 0``), constant ``K = -1``; no embedding."""

    name = "upper-half-plane"

    def metric(self, u):
        if u[1] <= 0:
            return np.zeros((2, 2))
        return np.eye(2) / u[1] ** 2

    def christoffel(self, u):
        y = u[1]
        gamma = np.zeros((2, 2, 2))
        gamma[0, 0, 1] = gamma[0, 1, 0] = -1.0 / y
        gamma[1, 0, 0] = 1.0 / y
        gamma[1, 1, 1] = -1.0 / y
        return gamma

    def curvature(self, u):
        return -1.0

    def quality(self, u):
        return 1.0 if u[1] > 0 else 0.0


# ---------------------------------------------------------------- surfaces
@dataclass
class Surface:
    """A declared surface: its charts, length unit and optional implicit equation."""

    kind: str
    parameters: dict
    length_unit: str
    charts: list[Chart]
    constant_curvature: float | None = None
    implicit: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]] | None = None
    axis_of_revolution: tuple[int, Callable[[np.ndarray], float]] | None = None  # (phi index, rho(u)) in chart 0
    notes: list[str] = field(default_factory=list)

    @property
    def embedded(self) -> bool:
        return self.charts[0].embed(np.zeros(2) + 0.5) is not None

    def describe(self) -> dict:
        return plain({"kind": self.kind, "parameters": self.parameters, "length_unit": self.length_unit,
                      "charts": [chart.name for chart in self.charts], "embedded": self.embedded,
                      "constant_curvature": self.constant_curvature, "notes": self.notes})


def _length(value: Any, name: str, unit: str) -> float:
    quantity = Quantity.from_json(value, name)
    require_dimension(quantity.unit, "m", name)
    result = quantity.magnitude(unit)
    if result <= 0:
        raise Refusal("out_of_domain", f"{name} must be positive")
    return result


def surface_from_json(record: Any, length_unit: str = "m") -> Surface:
    """Build a surface from ``{"type": ..., <parameters with units>}``."""
    record = mapping(record, "surface")
    kind = text(record.get("type"), "surface.type", 64)
    require_dimension(length_unit, "m", "working length unit")
    if kind == "plane":
        require_keys(record, "surface", {"type"})
        return Surface(kind, {}, length_unit, [PlaneChart()], 0.0, _implicit_plane(), None)
    if kind == "sphere":
        require_keys(record, "surface", {"type", "radius"})
        radius = _length(record["radius"], "surface.radius", length_unit)
        swap = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # e_z -> e_x
        charts = [SphereChart(radius, None, "polar-z"), SphereChart(radius, swap, "polar-x")]
        return Surface(kind, {"radius": radius}, length_unit, charts, 1.0 / radius ** 2, _implicit_sphere(radius),
                       (1, lambda u: radius * math.sin(u[0])))
    if kind == "cylinder":
        require_keys(record, "surface", {"type", "radius"})
        radius = _length(record["radius"], "surface.radius", length_unit)
        return Surface(kind, {"radius": radius}, length_unit, [CylinderChart(radius)], 0.0,
                       _implicit_cylinder(radius), (0, lambda u: radius),
                       ["developable: intrinsically flat, extrinsically curved"])
    if kind == "torus":
        require_keys(record, "surface", {"type", "major_radius", "minor_radius"})
        major = _length(record["major_radius"], "surface.major_radius", length_unit)
        minor = _length(record["minor_radius"], "surface.minor_radius", length_unit)
        if minor >= major:
            raise Refusal("out_of_domain", "A ring torus requires minor_radius < major_radius")
        return Surface(kind, {"major_radius": major, "minor_radius": minor}, length_unit, [TorusChart(major, minor)],
                       None, _implicit_torus(major, minor), (0, lambda u: major + minor * math.cos(u[1])))
    if kind == "graph":
        require_keys(record, "surface", {"type", "height"})
        polynomial = Polynomial.from_json(record["height"], "surface.height")
        return Surface(kind, {"height": polynomial.to_json()}, length_unit, [GraphChart(polynomial)], None,
                       _implicit_graph(polynomial), None, ["coordinates and height share the working length unit"])
    if kind == "conformal":
        require_keys(record, "surface", {"type", "sigma"})
        polynomial = Polynomial.from_json(record["sigma"], "surface.sigma")
        return Surface(kind, {"sigma": polynomial.to_json()}, length_unit, [ConformalChart(polynomial)], None, None,
                       None, ["abstract metric: no embedding is claimed"])
    if kind == "hyperbolic-half-plane":
        require_keys(record, "surface", {"type"})
        return Surface(kind, {}, length_unit, [HalfPlaneChart()], -1.0, None, None,
                       ["abstract metric with curvature -1 in its own normalized length"])
    raise Refusal("unsupported_surface", f"Surface type {kind!r} is not declared",
                  allowed=["plane", "sphere", "cylinder", "torus", "graph", "conformal", "hyperbolic-half-plane"])


def _implicit_plane():
    return lambda x: (x[2], np.array([0.0, 0.0, 1.0]), np.zeros((3, 3)))


def _implicit_sphere(radius):
    return lambda x: (x @ x - radius ** 2, 2 * x, 2 * np.eye(3))


def _implicit_cylinder(radius):
    return lambda x: (x[0] ** 2 + x[1] ** 2 - radius ** 2, np.array([2 * x[0], 2 * x[1], 0.0]),
                      np.diag([2.0, 2.0, 0.0]))


def _implicit_torus(major, minor):
    def implicit(x):
        rho = math.hypot(x[0], x[1])
        value = (rho - major) ** 2 + x[2] ** 2 - minor ** 2
        factor = 2 * (rho - major) / rho
        gradient = np.array([factor * x[0], factor * x[1], 2 * x[2]])
        hessian = np.zeros((3, 3))
        planar = np.array([x[0], x[1]])
        hessian[:2, :2] = factor * np.eye(2) + 2 * major / rho ** 3 * np.outer(planar, planar)
        hessian[2, 2] = 2.0
        return value, gradient, hessian
    return implicit


def _implicit_graph(polynomial):
    def implicit(x):
        f = polynomial.derivative
        value = x[2] - f(x[0], x[1])
        gradient = np.array([-f(x[0], x[1], 1, 0), -f(x[0], x[1], 0, 1), 1.0])
        hessian = np.zeros((3, 3))
        hessian[0, 0], hessian[0, 1] = -f(x[0], x[1], 2, 0), -f(x[0], x[1], 1, 1)
        hessian[1, 0], hessian[1, 1] = hessian[0, 1], -f(x[0], x[1], 0, 2)
        return value, gradient, hessian
    return implicit


# ---------------------------------------------------------------- geodesics
@dataclass
class Trajectory:
    s: np.ndarray
    u: np.ndarray
    du: np.ndarray
    chart: np.ndarray
    transitions: list[dict]
    embedded: np.ndarray | None
    speed: np.ndarray
    jacobi: np.ndarray | None = None
    djacobi: np.ndarray | None = None
    curvature: np.ndarray | None = None

    def endpoint(self) -> dict:
        record = {"u": self.u[-1], "du": self.du[-1], "chart": int(self.chart[-1]), "length": float(self.s[-1])}
        if self.embedded is not None:
            record["x"] = self.embedded[-1]
        if self.jacobi is not None:
            record["jacobi"] = float(self.jacobi[-1])
            record["djacobi"] = float(self.djacobi[-1])
        return plain(record)


def _acceleration(chart: Chart, u: np.ndarray, du: np.ndarray) -> np.ndarray:
    return -np.einsum("kij,i,j->k", chart.christoffel(u), du, du)


def _rk4_step(chart: Chart, surface: Surface, state: np.ndarray, h: float, jacobi: bool) -> np.ndarray:
    def rate(y):
        u, du = y[:2], y[2:4]
        out = np.empty_like(y)
        out[:2], out[2:4] = du, _acceleration(chart, u, du)
        if jacobi:
            out[4], out[5] = y[5], -chart.curvature(u) * y[4]
        return out
    k1 = rate(state)
    k2 = rate(state + 0.5 * h * k1)
    k3 = rate(state + 0.5 * h * k2)
    k4 = rate(state + h * k3)
    return state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def speed(chart: Chart, u: np.ndarray, du: np.ndarray) -> float:
    return float(math.sqrt(max(du @ chart.metric(u) @ du, 0.0)))


def change_chart(surface: Surface, source: int, target: int, u: np.ndarray, du: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a point and tangent vector between charts through the embedding."""
    first, second = surface.charts[source], surface.charts[target]
    x, velocity = first.embed(u), first.jacobian(u) @ du
    mapped = second.invert(x)
    if second.quality(mapped) < SINGULAR_QUALITY:
        raise Refusal("chart_singularity", f"Point is singular in chart {second.name}")
    jac = second.jacobian(mapped)
    return mapped, np.linalg.solve(jac.T @ jac, jac.T @ velocity)


def geodesic(surface: Surface, u0: Any, du0: Any, length: float, steps: int, *, chart: int = 0,
             transitions: bool = True, jacobi: bool = False, stop_at_conjugate: bool = False) -> Trajectory:
    """Integrate a geodesic over parameter ``[0, length]`` with fixed-step RK4.

    With ``jacobi=True`` the initial velocity must have unit speed so the
    parameter is arclength, and the scalar normal Jacobi field with ``j(0)=0``,
    ``j'(0)=1`` is integrated alongside.
    """
    steps = integer(steps, "steps", minimum=1, maximum=MAX_STEPS)
    length = finite(length, "length", minimum=0.0, exclusive_minimum=True)
    current = integer(chart, "chart", minimum=0, maximum=len(surface.charts) - 1)
    u, du = vector(u0, "initial point", 2), vector(du0, "initial velocity", 2)
    chart_object = surface.charts[current]
    if chart_object.quality(u) < SINGULAR_QUALITY:
        raise Refusal("chart_singularity", f"Initial point is singular in chart {chart_object.name}")
    initial_speed = speed(chart_object, u, du)
    if initial_speed == 0:
        raise Refusal("out_of_domain", "A geodesic needs a nonzero initial velocity")
    if jacobi and abs(initial_speed - 1.0) > 1e-9:
        raise Refusal("not_unit_speed", "Jacobi fields require a unit-speed initial velocity", speed=initial_speed)
    h = length / steps
    state = np.concatenate([u, du, [0.0, 1.0] if jacobi else []])
    rows_u, rows_du, charts, s_values, jac, djac, speeds, curvature = [u], [du], [current], [0.0], [0.0], [1.0], [], []
    speeds.append(initial_speed)
    curvature.append(chart_object.curvature(u))
    switched: list[dict] = []
    for index in range(steps):
        if transitions and len(surface.charts) > 1 and surface.charts[current].quality(state[:2]) < QUALITY_SWITCH:
            best = max(range(len(surface.charts)), key=lambda i: surface.charts[i].quality(
                surface.charts[i].invert(surface.charts[current].embed(state[:2]))))
            if best != current:
                mapped_u, mapped_du = change_chart(surface, current, best, state[:2], state[2:4])
                switched.append({"step": index, "s": index * h, "from": surface.charts[current].name,
                                 "to": surface.charts[best].name})
                state = np.concatenate([mapped_u, mapped_du, state[4:]])
                current = best
        chart_object = surface.charts[current]
        state = _rk4_step(chart_object, surface, state, h, jacobi)
        if not np.all(np.isfinite(state)) or chart_object.quality(state[:2]) < SINGULAR_QUALITY:
            raise Refusal("chart_singularity", "Trajectory reached a coordinate singularity",
                          chart=chart_object.name, s=(index + 1) * h, transitions_allowed=transitions)
        rows_u.append(state[:2].copy())
        rows_du.append(state[2:4].copy())
        charts.append(current)
        s_values.append((index + 1) * h)
        speeds.append(speed(chart_object, state[:2], state[2:4]))
        curvature.append(chart_object.curvature(state[:2]))
        if jacobi:
            jac.append(state[4])
            djac.append(state[5])
            if stop_at_conjugate and jac[-2] > 0 >= jac[-1]:
                break
    u_array = np.array(rows_u)
    chart_array = np.array(charts)
    embedded = None
    if surface.charts[0].embed(u_array[0]) is not None:
        embedded = np.array([surface.charts[c].embed(point) for c, point in zip(chart_array, u_array)])
    return Trajectory(np.array(s_values), u_array, np.array(rows_du), chart_array, switched, embedded,
                      np.array(speeds), np.array(jac) if jacobi else None, np.array(djac) if jacobi else None,
                      np.array(curvature))


def first_conjugate_point(trajectory: Trajectory) -> float | None:
    """Arclength of the first zero of the scalar Jacobi field after ``s = 0`` (linear interpolation)."""
    if trajectory.jacobi is None:
        raise Refusal("jacobi_not_integrated", "The trajectory carries no Jacobi field")
    j, s = trajectory.jacobi, trajectory.s
    for index in range(1, len(j) - 1):
        if j[index] > 0 >= j[index + 1]:
            return float(s[index] + (s[index + 1] - s[index]) * j[index] / (j[index] - j[index + 1]))
    return None


def jacobi_closed_form(curvature: float, s: np.ndarray) -> np.ndarray:
    """``j(s)`` with ``j(0)=0, j'(0)=1`` on constant curvature ``K``."""
    if curvature > 0:
        root = math.sqrt(curvature)
        return np.sin(root * s) / root
    if curvature < 0:
        root = math.sqrt(-curvature)
        return np.sinh(root * s) / root
    return np.asarray(s, dtype=float).copy()


def unit_direction(surface: Surface, u: Any, heading: float, chart: int = 0) -> np.ndarray:
    """Unit tangent at ``u`` making angle ``heading`` with the first coordinate direction (orthonormalized)."""
    u = vector(u, "point", 2)
    if surface.charts[chart].quality(u) < SINGULAR_QUALITY:
        raise Refusal("chart_singularity", f"No tangent frame at a singular point of chart {surface.charts[chart].name}")
    g = surface.charts[chart].metric(u)
    e1 = np.array([1.0, 0.0]) / math.sqrt(g[0, 0])
    e2 = np.array([0.0, 1.0]) - (g[0, 1] / g[0, 0]) * np.array([1.0, 0.0])
    e2 = e2 / math.sqrt(e2 @ g @ e2)
    return math.cos(heading) * e1 + math.sin(heading) * e2


def exp_map(surface: Surface, u: Any, v: Any, steps: int = 400, chart: int = 0) -> np.ndarray:
    trajectory = geodesic(surface, u, v, 1.0, steps, chart=chart, transitions=False)
    return trajectory.u[-1]


def log_map(surface: Surface, u: Any, target: Any, *, guess: Any = None, steps: int = 400, tolerance: float = 1e-11,
            max_iterations: int = 30, chart: int = 0, condition_limit: float = 1e5) -> dict:
    """Newton shooting for ``v`` with ``exp_u(v) = target``; refused near conjugate points.

    The exponential-map Jacobian is evaluated at every iterate, including the
    accepted one: a converged residual beside a singular Jacobian means the
    geodesic is not locally unique, so the result is refused, not returned.
    """
    u, target = vector(u, "point", 2), vector(target, "target", 2)
    v = target - u if guess is None else vector(guess, "guess", 2)
    history = []
    for iteration in range(max_iterations + 1):
        residual = exp_map(surface, u, v, steps, chart) - target
        error = float(np.linalg.norm(residual))
        history.append(error)
        jacobian = np.zeros((2, 2))
        epsilon = 1e-6 * max(1.0, float(np.linalg.norm(v)))
        for column in range(2):
            step = np.zeros(2)
            step[column] = epsilon
            jacobian[:, column] = (exp_map(surface, u, v + step, steps, chart)
                                   - exp_map(surface, u, v - step, steps, chart)) / (2 * epsilon)
        condition = float(np.linalg.cond(jacobian))
        if not math.isfinite(condition) or condition > condition_limit:
            raise Refusal("near_conjugate", "Exponential-map Jacobian is ill-conditioned; the log map is not "
                          "resolved near a conjugate point", condition=condition if math.isfinite(condition) else None)
        if error <= tolerance * max(1.0, float(np.linalg.norm(target))):
            return plain({"v": v, "length": speed(surface.charts[chart], u, v), "iterations": iteration,
                          "residual": error, "history": history, "jacobian_condition": condition})
        if iteration == max_iterations:
            break
        v = v - np.linalg.solve(jacobian, residual)
    raise Refusal("log_map_not_converged", "Newton shooting did not converge", history=history)


def geodesic_curvature(surface: Surface, path: Any, chart: int = 0) -> np.ndarray:
    """Signed geodesic curvature of a sampled parameter path (central differences, interior samples)."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise Refusal("malformed_record", "A path needs at least five samples of two coordinates")
    chart_object = surface.charts[chart]
    result = []
    for index in range(1, len(points) - 1):
        u = points[index]
        g = chart_object.metric(u)
        first = (points[index + 1] - points[index - 1]) / 2.0
        second = points[index + 1] - 2 * points[index] + points[index - 1]
        norm = math.sqrt(first @ g @ first)
        covariant = second + np.einsum("kij,i,j->k", chart_object.christoffel(u), first, first)
        # R90(g u') is g-orthogonal to u' with g-norm sqrt(det g)|u'|_g; kappa_g = <A, N>_g / |u'|_g^2
        normal = np.array([-(g[1] @ first), g[0] @ first]) / math.sqrt(np.linalg.det(g))
        result.append(float(covariant @ g @ normal) / norm ** 3)
    return np.array(result)


def extrinsic_geodesic(surface: Surface, u0: Any, du0: Any, length: float, steps: int) -> np.ndarray:
    """Integrate ``x'' = -(x'^T H x') grad F / |grad F|^2`` in R^3 without Christoffel symbols."""
    if surface.implicit is None:
        raise Refusal("no_implicit_form", f"{surface.kind} has no implicit embedding for the extrinsic route")
    steps = integer(steps, "steps", minimum=1, maximum=MAX_STEPS)
    chart = surface.charts[0]
    u, du = vector(u0, "initial point", 2), vector(du0, "initial velocity", 2)
    state = np.concatenate([chart.embed(u), chart.jacobian(u) @ du])
    implicit = surface.implicit

    def rate(y):
        x, v = y[:3], y[3:]
        _, gradient, hessian = implicit(x)
        return np.concatenate([v, -(v @ hessian @ v) / (gradient @ gradient) * gradient])

    h = length / steps
    path = [state[:3].copy()]
    for _ in range(steps):
        k1 = rate(state)
        k2 = rate(state + 0.5 * h * k1)
        k3 = rate(state + 0.5 * h * k2)
        k4 = rate(state + h * k3)
        state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(state[:3].copy())
    return np.array(path)


# ---------------------------------------------------------------- closed forms
def closed_form_distance(surface: Surface, u: Any, w: Any) -> float | None:
    """Minimal geodesic distance where a closed form exists (chart-0 coordinates)."""
    u, w = np.asarray(u, dtype=float), np.asarray(w, dtype=float)
    if surface.kind == "plane":
        return float(np.linalg.norm(w - u))
    if surface.kind == "sphere":
        radius = surface.parameters["radius"]
        a, b = surface.charts[0].embed(u) / radius, surface.charts[0].embed(w) / radius
        return float(radius * math.atan2(np.linalg.norm(np.cross(a, b)), a @ b))
    if surface.kind == "cylinder":
        radius = surface.parameters["radius"]
        delta = (w[0] - u[0] + math.pi) % (2 * math.pi) - math.pi
        return float(math.hypot(radius * delta, w[1] - u[1]))
    if surface.kind == "hyperbolic-half-plane":
        return float(math.acosh(1 + ((w[0] - u[0]) ** 2 + (w[1] - u[1]) ** 2) / (2 * u[1] * w[1])))
    return None


def chord_distance(surface: Surface, u: Any, w: Any, chart: int = 0) -> float:
    """Straight-line distance between two surface points in the embedding space."""
    chart_object = surface.charts[chart]
    a, b = chart_object.embed(np.asarray(u, dtype=float)), chart_object.embed(np.asarray(w, dtype=float))
    if a is None:
        raise Refusal("no_embedding", f"{surface.kind} declares no embedding; a chord is undefined")
    return float(np.linalg.norm(b - a))


def great_circle(surface: Surface, u0: Any, du0: Any, s: float) -> np.ndarray:
    """Closed-form embedded point at arclength ``s`` on a sphere (unit-speed start)."""
    radius = surface.parameters["radius"]
    chart = surface.charts[0]
    x0 = chart.embed(np.asarray(u0, dtype=float))
    t0 = chart.jacobian(np.asarray(u0, dtype=float)) @ np.asarray(du0, dtype=float)
    t0 = t0 / np.linalg.norm(t0)
    return math.cos(s / radius) * x0 + radius * math.sin(s / radius) * t0
