"""Geometry models for the manufacturing and robotic use cases (T126-T141).

Scope: declared nominal specimens in millimetres (a flat plate, a rolled
cylinder, a domed coupon and a toroidal winding mandrel), geodesic routes and
their Jacobi transfer, geodesic and normal curvature of chart curves by two
independent routes (chart Christoffel symbols and the embedded 3D curve),
standoff and offset tool paths, and coverage of scan rows.

Non-claims: every specimen is a declared nominal model. A predicted
separation, curvature, standoff or coverage is a property of that model, not
of a manufactured part; it becomes a statement about hardware only through the
measurement protocols, which nothing here executes. Nothing here decides
production acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from . import integrators, jacobi
from .surfaces import Cylinder, GaussianBump, Plane, Torus

# Declared nominal specimens (millimetres). The coupon dome has crest principal
# radius sigma^2 / height = 40 mm and sits 60 mm from the station edge, so a
# straight route across it passes through a lateral focal point.
PLATE = Plane()
CYLINDER_RADIUS = 100.0
CYLINDER = Cylinder(CYLINDER_RADIUS)
DOME_HEIGHT, DOME_SIGMA = 10.0, 20.0
COUPON = GaussianBump(DOME_HEIGHT, DOME_SIGMA)
COUPON_X = (-60.0, 140.0)
COUPON_Y = (-100.0, 100.0)
STATION = (-60.0, 0.0)
TORUS_MAJOR, TORUS_MINOR = 150.0, 50.0
TORUS = Torus(TORUS_MAJOR, TORUS_MINOR)

STEP_MM = 1.0


def coupon(height=DOME_HEIGHT, sigma=DOME_SIGMA) -> GaussianBump:
    return GaussianBump(height, sigma)


def embed(surface, points) -> np.ndarray:
    return np.array([surface.embedding(np.asarray(u, dtype=float)) for u in points])


def polyline_length(points3d) -> float:
    points3d = np.asarray(points3d, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points3d, axis=0), axis=1)))


# Routes ---------------------------------------------------------------------
@dataclass
class Route:
    """A geodesic route from a start point until it reaches the chart line x = x_end."""

    name: str
    heading: float
    length: float
    transfer: jacobi.Transfer          # integrated exactly to ``length``
    focal: list = field(default_factory=list)       # j_lat zeros within the horizon
    conjugate: list = field(default_factory=list)   # j_head zeros within the horizon
    horizon: float = 0.0
    exits_side: bool = False

    @property
    def end(self) -> np.ndarray:
        return self.transfer.points[-1]

    def nearest_focus(self):
        """(distance, kind) of the nearest focal or conjugate point, or (None, 'none')."""
        candidates = [(s, "focal") for s in self.focal[:1]] + [(s, "conjugate") for s in self.conjugate[:1]]
        return min(candidates) if candidates else (None, "none")

    def margin(self) -> tuple[float, bool]:
        """Focus margin s_focus / L; the bound flag is True when no focus lies within the horizon."""
        distance, _ = self.nearest_focus()
        if distance is None:
            return self.horizon / self.length, True
        return distance / self.length, False

    def summary(self) -> dict:
        states = self.transfer.states
        distance, kind = self.nearest_focus()
        margin, bound = self.margin()
        return {"route": self.name, "heading_deg": round(math.degrees(self.heading), 10),
                "length_mm": self.length, "end_u_mm": [float(v) for v in self.end],
                "max_abs_j_lat": float(np.max(np.abs(states[:, 4]))),
                "max_abs_j_head_mm": float(np.max(np.abs(states[:, 6]))),
                "j_lat_end": float(states[-1, 4]), "j_lat_prime_end_per_mm": float(states[-1, 5]),
                "j_head_end_mm": float(states[-1, 6]), "j_head_prime_end": float(states[-1, 7]),
                "first_focal_mm": self.focal[0] if self.focal else None,
                "first_conjugate_mm": self.conjugate[0] if self.conjugate else None,
                "nearest_focus_kind": kind, "nearest_focus_mm": distance,
                "focus_margin": margin, "margin_is_lower_bound": bound, "horizon_mm": self.horizon,
                "exits_side": self.exits_side}


def route_to_edge(surface, u0, heading, x_end, name="route", step=STEP_MM, horizon_factor=2.5,
                  y_limits=COUPON_Y) -> Route:
    """Integrate a geodesic with its Jacobi columns until the chart coordinate x reaches ``x_end``.

    The first pass runs to a horizon beyond the edge so focal and conjugate
    points just past the route end are found; the second pass integrates
    exactly to the crossing arclength so end values carry no interpolation.
    """
    u0 = np.asarray(u0, dtype=float)
    reach = (x_end - u0[0]) / max(math.cos(heading), 0.2)
    horizon = horizon_factor * reach
    steps = int(math.ceil(horizon / step))
    probe = jacobi.transfer(surface, u0, heading, horizon, steps=steps)
    x = probe.states[:, 0] - x_end
    crossings = integrators.hermite_zeros(probe.s, x, probe.states[:, 2])
    if not crossings:
        raise ValueError(f"Route {name} does not reach x = {x_end} within its horizon")
    length = crossings[0]
    inside = probe.s <= length
    y = probe.states[inside, 1]
    exits = bool(np.any(y < y_limits[0]) or np.any(y > y_limits[1]))
    exact = jacobi.transfer(surface, u0, heading, length, steps=int(math.ceil(length / step)))
    return Route(name, float(heading), float(length), exact, probe.focal_points(), probe.conjugate_points(),
                 float(horizon), exits)


def fan(surface=COUPON, headings_deg=(0, 5, 10, 15, 20, 25), step=STEP_MM, start=STATION, x_end=COUPON_X[1]):
    return [route_to_edge(surface, start, math.radians(a), x_end, name=f"fan{a:+d}deg", step=step)
            for a in headings_deg]


def separation_linear(transfer, lateral, dheading) -> np.ndarray:
    """First-order normal separation lateral * j_lat + dheading * j_head along the route."""
    return lateral * transfer.states[:, 4] + dheading * transfer.states[:, 6]


def separation_nonlinear(surface, u0, heading, length, steps, lateral, dheading, base=None) -> np.ndarray:
    """Normal separation of the exactly perturbed geodesic at matched arclength nodes."""
    if base is None:
        base = jacobi.transfer(surface, u0, heading, length, steps=steps)
    start = jacobi.perturbed_start(surface, u0, heading, lateral=lateral, heading_change=dheading)
    _, states = integrators.integrate_fixed(surface.geodesic_rhs, start, length, steps, "rk4")
    return jacobi.normal_separation(surface, base.states[:, :4], states)


# Path curvature by two routes ----------------------------------------------
def second_fundamental_form(surface, u) -> np.ndarray:
    xuu, xuv, xvv = surface.second(u)
    n = surface.unit_normal3(u)
    return np.array([[xuu @ n, xuv @ n], [xuv @ n, xvv @ n]])


def chart_curvatures(surface, u, du, ddu) -> tuple[float, float]:
    """(kappa_g, kappa_n) of a chart curve from Christoffel symbols and the second fundamental form.

    kappa_g = <u'' + Gamma(u', u'), N> / |u'|^2 with N the unit tangent rotated
    +90 degrees; kappa_n = II(u', u') / I(u', u') with the chart normal Xu x Xv.
    """
    u, du, ddu = (np.asarray(v, dtype=float) for v in (u, du, ddu))
    speed2 = surface.speed_squared(u, du)
    acceleration = ddu + np.einsum("kij,i,j->k", surface.christoffel(u), du, du)
    normal = surface.normal(u, du / math.sqrt(speed2))
    return (surface.inner(u, acceleration, normal) / speed2,
            float(du @ second_fundamental_form(surface, u) @ du) / speed2)


def embedded_curvatures(surface, u, du, ddu) -> tuple[float, float]:
    """(kappa_g, kappa_n) from the embedded curve X(u(t)) in R^3 (no Christoffel symbols)."""
    u, du, ddu = (np.asarray(v, dtype=float) for v in (u, du, ddu))
    xu, xv = surface.first(u)
    xuu, xuv, xvv = surface.second(u)
    velocity = du[0] * xu + du[1] * xv
    acceleration = (ddu[0] * xu + ddu[1] * xv + du[0] ** 2 * xuu + 2 * du[0] * du[1] * xuv
                    + du[1] ** 2 * xvv)
    speed2 = float(velocity @ velocity)
    tangent = velocity / math.sqrt(speed2)
    curvature = (acceleration - (acceleration @ tangent) * tangent) / speed2
    n = surface.unit_normal3(u)
    return float(curvature @ np.cross(n, tangent)), float(curvature @ n)


# Offset (standoff) tool paths and ray casting -------------------------------
def ray_to_graph(surface, origin, direction, t0=0.0, iterations=60) -> float:
    """Parameter t where origin + t direction meets the graph z = f(x, y) (Newton)."""
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    t = float(t0)
    for _ in range(iterations):
        point = origin + t * direction
        fx, fy, *_ = surface.height_derivatives(point[:2])
        residual = point[2] - surface.height(point[:2])
        slope = direction[2] - fx * direction[0] - fy * direction[1]
        update = residual / slope
        t -= update
        if abs(update) < 1e-14 * max(1.0, abs(t)):
            break
    return t


def standoff_error_exact(surface, u, lateral_direction3, lateral, standoff) -> float:
    """Distance along the programmed tool axis to the surface, minus the standoff.

    The tool point is X(u) + standoff n(u) displaced by ``lateral`` along the unit
    tangent-plane direction; its axis stays -n(u), as it would under a rigid
    registration offset.
    """
    n = surface.unit_normal3(u)
    origin = surface.embedding(u) + standoff * n + lateral * np.asarray(lateral_direction3, dtype=float)
    return ray_to_graph(surface, origin, -n, t0=standoff) - standoff


def lateral_direction3(surface, u, tangent_chart) -> np.ndarray:
    normal_chart = surface.normal(u, tangent_chart)
    return surface.embedding_jacobian(u) @ normal_chart


# Surface area sampling and coverage ----------------------------------------
def area_grid(surface, x_range, y_range, count):
    """Chart grid points, their embedding and trapezoid area weights sqrt(det g) dx dy."""
    xs = np.linspace(*x_range, count)
    ys = np.linspace(*y_range, count)
    wx = np.full(count, xs[1] - xs[0])
    wx[[0, -1]] *= 0.5
    wy = np.full(count, ys[1] - ys[0])
    wy[[0, -1]] *= 0.5
    points, weights = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            u = np.array([x, y])
            points.append(surface.embedding(u))
            weights.append(math.sqrt(np.linalg.det(surface.metric(u))) * wx[i] * wy[j])
    return np.array(points), np.array(weights)


def coverage_fraction(points3d, weights, rows3d, swath) -> float:
    """Area fraction within swath/2 (3D distance to the nearest row sample) of some row."""
    nearest = np.full(len(points3d), np.inf)
    for row in rows3d:
        for chunk in range(0, len(row), 256):
            block = row[chunk:chunk + 256]
            d = np.sqrt(((points3d[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            np.minimum(nearest, d, out=nearest)
    covered = nearest <= 0.5 * swath + 1e-12
    return float(weights[covered].sum() / weights.sum())


def clip_to_extent(points_u, x_range=COUPON_X, y_range=COUPON_Y) -> np.ndarray:
    points_u = np.asarray(points_u, dtype=float)
    keep = ((points_u[:, 0] >= x_range[0] - 1e-9) & (points_u[:, 0] <= x_range[1] + 1e-9)
            & (points_u[:, 1] >= y_range[0] - 1e-9) & (points_u[:, 1] <= y_range[1] + 1e-9))
    # Keep the leading run only: a row that leaves the coupon is not scanned after it.
    if not keep.all():
        first_out = int(np.argmin(keep))
        keep[first_out:] = False
    return points_u[keep]
