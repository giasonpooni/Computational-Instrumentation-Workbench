"""Chart atlas with exact transitions, and coordinate-versus-curvature singularity scans.

Scope: a two-chart atlas of the sphere of radius R built from the core polar
chart A (poles on the z axis) and the same chart rigidly rotated so that its
poles B lie on the equator of A (``ciw.lab.surfaces.Rotated``). Transition
maps go through the embedding: a point is mapped by the closed-form inverse of
the target chart and a velocity by v_B = g_B^{-1} J_B^T J_A v_A, which is
exact because J_A v_A lies in the common tangent plane. Geodesics are
integrated with the core RK4 step, switching charts between steps when the
active chart's normalized det g falls below a threshold.

The singularity scan follows a path into a candidate point and fits power
laws for det g, the metric condition number, the largest Christoffel symbol
and |K|; a geodesic-circle circumference ratio separates removable coordinate
singularities from conical points, and the radial distance integral separates
boundaries at infinite distance. ``require_regular`` is the pointwise guard; it
refuses with a :class:`SingularityRefusal` code.

Non-claims: normalized mathematical surfaces only. The classification rules
are heuristics validated on the declared examples (rotationally symmetric
approach paths, radial chart lines that are geodesics); they are not a proof
of the singularity type of an arbitrary surface or of a measured one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .integrators import integrate_fixed, step_rk4
from .surfaces import Rotated, Sphere, Surface, SurfaceRefusal
from .surfaces_discrete_geometry import SingularityRefusal

# Rotation about the y axis by +pi/2: maps e_z to e_x, so chart B's poles are (+-R, 0, 0).
ROTATION_B = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
SWITCH_THRESHOLD = 0.25


class SphereAtlas:
    """Charts A (core polar chart) and B (A rotated so its poles lie on A's equator)."""

    def __init__(self, radius=1.0, threshold=SWITCH_THRESHOLD):
        if not 0 < threshold < 0.5:
            raise SurfaceRefusal("Chart-switch threshold must lie in (0, 1/2), below the atlas covering bound")
        self.radius, self.threshold = float(radius), float(threshold)
        self.charts = {"A": Sphere(radius), "B": Rotated(Sphere(radius), ROTATION_B)}
        self.rotations = {"A": np.eye(3), "B": ROTATION_B}

    def describe(self) -> dict:
        return {"radius": self.radius, "threshold": self.threshold,
                "charts": {"A": "polar (theta, phi), poles (0, 0, +-R)",
                           "B": "polar chart rotated by +pi/2 about y, poles (+-R, 0, 0)"}}

    def embedding(self, name, u) -> np.ndarray:
        return self.charts[name].embedding(u)

    def to_chart(self, name, point) -> np.ndarray:
        """Closed-form inverse of chart ``name``: theta in [0, pi], phi in (-pi, pi]."""
        p = self.rotations[name].T @ np.asarray(point, dtype=float)
        return np.array([math.atan2(math.hypot(p[0], p[1]), p[2]), math.atan2(p[1], p[0])])

    def regularity(self, name, u) -> float:
        """det g / R^4 = sin^2(theta): 1 on the chart's equator, 0 at its poles."""
        g = self.charts[name].metric(u)
        return float((g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]) / self.radius ** 4)

    def transition(self, source, target, u) -> np.ndarray:
        return self.to_chart(target, self.embedding(source, u))

    def pushforward(self, source, target, u, v) -> tuple[np.ndarray, np.ndarray]:
        """Exact point and velocity transition between charts."""
        tangent = self.charts[source].embedding_jacobian(u) @ np.asarray(v, dtype=float)
        target_u = self.transition(source, target, u)
        return target_u, self.lift(target, target_u, tangent)

    def lift(self, name, u, tangent) -> np.ndarray:
        """Chart components of an ambient tangent vector: g^{-1} J^T w."""
        jac = self.charts[name].embedding_jacobian(u)
        return np.linalg.solve(jac.T @ jac, jac.T @ np.asarray(tangent, dtype=float))

    def best_chart(self, point) -> str:
        scores = {name: self.regularity(name, self.to_chart(name, point)) for name in sorted(self.charts)}
        return max(sorted(scores), key=lambda name: scores[name])


def integrate_atlas(atlas: SphereAtlas, chart, u0, v0, length, steps) -> dict:
    """RK4 geodesic with chart switches between steps when regularity drops below the threshold.

    A switch changes only coordinates (exact transition), so the sequence of
    embedded states is the one a chart-free integrator would see up to the
    local truncation error of each step.
    """
    h = length / steps
    y = np.concatenate([np.asarray(u0, dtype=float), np.asarray(v0, dtype=float)])
    active = chart
    points = [atlas.embedding(active, y[:2])]
    tangents = [atlas.charts[active].embedding_jacobian(y[:2]) @ y[2:]]
    active_det = [atlas.regularity(active, y[:2])]
    switches = []
    for n in range(steps):
        y = step_rk4(atlas.charts[active].geodesic_rhs, y, h)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"Atlas integration produced a nonfinite state at step {n + 1}")
        det = atlas.regularity(active, y[:2])
        if det < atlas.threshold:
            candidates = {name: atlas.regularity(name, atlas.transition(active, name, y[:2]))
                          for name in sorted(atlas.charts) if name != active}
            best = max(sorted(candidates), key=lambda name: candidates[name])
            if candidates[best] > det:
                u_new, v_new = atlas.pushforward(active, best, y[:2], y[2:])
                switches.append({"step": n + 1, "s": (n + 1) * h, "from": active, "to": best,
                                 "det_before": det, "det_after": candidates[best]})
                active, y = best, np.concatenate([u_new, v_new])
                det = candidates[best]
        points.append(atlas.embedding(active, y[:2]))
        tangents.append(atlas.charts[active].embedding_jacobian(y[:2]) @ y[2:])
        active_det.append(det)
    return {"s": np.linspace(0.0, length, steps + 1), "points": np.array(points), "tangents": np.array(tangents),
            "switches": switches, "min_active_det": float(min(active_det))}


def integrate_single(surface: Surface, u0, v0, length, steps) -> dict:
    """The same RK4 integration confined to one chart, with no switching."""
    y0 = np.concatenate([np.asarray(u0, dtype=float), np.asarray(v0, dtype=float)])
    s, states = integrate_fixed(surface.geodesic_rhs, y0, length, steps, "rk4")
    points = np.array([surface.embedding(y[:2]) for y in states])
    dets = [float(np.linalg.det(surface.metric(y[:2]))) for y in states]
    return {"s": s, "points": points, "min_det": float(min(dets))}


def pole_passing_great_circle(radius, delta, azimuth=0.3, lead=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Start point and unit tangent of the great circle whose closest approach to (0, 0, R) is delta.

    The circle reaches the closest-approach point after arclength ``lead * R``.
    """
    closest = np.array([math.sin(delta) * math.cos(azimuth), math.sin(delta) * math.sin(azimuth), math.cos(delta)])
    across = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    start = radius * (math.cos(lead) * closest - math.sin(lead) * across)
    tangent = math.sin(lead) * closest + math.cos(lead) * across
    return start, tangent


def great_circle(start, tangent, radius, s) -> np.ndarray:
    """Exact great circle X(s) = cos(s/R) X0 + R sin(s/R) T0 (T0 unit)."""
    s = np.asarray(s, dtype=float)
    return np.cos(s / radius)[:, None] * start[None, :] + (radius * np.sin(s / radius))[:, None] * tangent[None, :]


# ------------------------------------------------------------ singularities
@dataclass(frozen=True)
class Approach:
    """A path into a candidate point: polar charts use (r, t); Cartesian charts use p + r (cos t, sin t)."""

    name: str
    surface: Surface
    center: tuple
    polar: bool
    angle: float = 0.7

    def point(self, r, t=None):
        t = self.angle if t is None else t
        if self.polar:
            return np.array([r, t])
        return np.array([self.center[0] + r * math.cos(t), self.center[1] + r * math.sin(t)])

    def loop_velocity(self, r, t):
        return np.array([0.0, 1.0]) if self.polar else np.array([-r * math.sin(t), r * math.cos(t)])

    def radial_velocity(self, t=None):
        t = self.angle if t is None else t
        return np.array([1.0, 0.0]) if self.polar else np.array([math.cos(t), math.sin(t)])


def scan_distances(first=4, last=32, per_decade=4) -> np.ndarray:
    """Log-spaced approach distances 10^(-k/per_decade), from 1e-1 to 1e-8 by default."""
    return np.array([10.0 ** (-k / per_decade) for k in range(first, last + 1)])


def point_invariants(surface: Surface, u) -> dict:
    g = surface.metric(u)
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    gamma = surface.christoffel(u)
    return {"det": float(g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]), "condition": float(eig[1] / eig[0]),
            "christoffel": float(np.max(np.abs(gamma))), "curvature": float(surface.gaussian_curvature(u))}


def _slope(r, values):
    """Least-squares power-law exponent; None when values vanish (identically zero quantity)."""
    values = np.abs(np.asarray(values, dtype=float))
    if np.any(values == 0) or not np.all(np.isfinite(values)):
        return None
    slope, _ = np.polyfit(np.log(r), np.log(values), 1)
    return float(slope)


def circumference_ratio(approach: Approach, r, nodes=64) -> float:
    """C(r) / (2 pi rho(r)): loop length over 2 pi times radial distance (trapezoid / Gauss-Legendre)."""
    ts = 2.0 * math.pi * np.arange(nodes) / nodes
    speeds = [math.sqrt(float(w @ approach.surface.metric(approach.point(r, t)) @ w))
              for t, w in ((t, approach.loop_velocity(r, t)) for t in ts)]
    circumference = 2.0 * math.pi * float(np.mean(speeds))
    x, weights = np.polynomial.legendre.leggauss(16)
    w = approach.radial_velocity()
    radial = 0.0
    for xi, wi in zip(x, weights):
        rr = 0.5 * r * (xi + 1.0)
        radial += 0.5 * r * wi * math.sqrt(float(w @ approach.surface.metric(approach.point(rr)) @ w))
    return circumference / (2.0 * math.pi * radial)


def scan(approach: Approach, distances=None) -> dict:
    """Invariants along the approach, fitted exponents and the classification."""
    r = scan_distances() if distances is None else np.asarray(distances, dtype=float)
    rows = [point_invariants(approach.surface, approach.point(ri)) for ri in r]
    table = {key: [row[key] for row in rows] for key in ("det", "condition", "christoffel", "curvature")}
    small = r <= 1e-3
    exponents = {key: _slope(r[small], np.array(table[key])[small]) for key in table}
    w = approach.radial_velocity()
    radial_speed = [math.sqrt(float(w @ approach.surface.metric(approach.point(ri)) @ w)) for ri in r]
    # Radial distance over the last two and the first two decades of the scan (trapezoid in r).
    order = np.argsort(r)
    rs, speeds = r[order], np.array(radial_speed)[order]
    pieces = 0.5 * (speeds[1:] + speeds[:-1]) * np.diff(rs)
    edges = rs[1:]
    near = float(np.sum(pieces[edges <= 100 * rs[0]]))
    far = float(np.sum(pieces[edges > rs[-1] / 100]))
    ratio = circumference_ratio(approach, float(r.min())) if exponents["det"] is None or exponents["det"] >= -0.5 else None
    result = {"approach": approach.name, "distances": [float(v) for v in r], "table": table, "exponents": exponents,
              "circumference_ratio": ratio, "distance_near_over_far": near / far if far > 0 else None}
    result["classification"] = classify(result)
    return result


def classify(result) -> str:
    """regular | coordinate_singularity | conical_singularity | curvature_singularity | infinite_distance_boundary."""
    e = result["exponents"]
    det, cond, curvature = e["det"], e["condition"], e["curvature"]
    if curvature is not None and curvature <= -0.5:
        return "curvature_singularity"
    if det is not None and det <= -0.5 and (result["distance_near_over_far"] or 0.0) > 0.1:
        return "infinite_distance_boundary"
    degenerate = (det is not None and det >= 0.5) or (cond is not None and cond <= -0.5)
    if degenerate:
        ratio = result["circumference_ratio"]
        if ratio is not None and abs(ratio - 1.0) > 1e-3:
            return "conical_singularity"
        return "coordinate_singularity"
    return "regular"


def require_regular(surface: Surface, u, *, max_condition=1e8, curvature_bound=1e6, length=1.0) -> dict:
    """Pointwise guard: refuse degenerate metrics and curvature blow-up with a refusal code.

    Codes: ``nonfinite_point``, ``nonfinite_metric``, ``degenerate_metric``
    (indefinite or condition number above ``max_condition``: a coordinate or
    conical singularity), ``curvature_blowup`` (|K| length^2 above
    ``curvature_bound``), a code raised by the surface itself (for example
    ``curvature_singularity``), or ``chart_refused`` from the core check.
    """
    u = np.asarray(u, dtype=float)
    if u.shape != (2,) or not np.all(np.isfinite(u)):
        raise SingularityRefusal("nonfinite_point", "Surface coordinates must be two finite numbers")
    g = surface.metric(u)
    if not np.all(np.isfinite(g)):
        raise SingularityRefusal("nonfinite_metric", f"{surface.name}: metric is not finite at {u.tolist()}")
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    if not eig[0] > 0:
        raise SingularityRefusal("degenerate_metric", f"{surface.name}: metric is not positive definite at {u.tolist()}")
    condition = float(eig[1] / eig[0])
    if condition > max_condition:
        raise SingularityRefusal("degenerate_metric", f"{surface.name}: metric condition number {condition:.3g} "
                                 f"exceeds {max_condition:.3g}; change chart or treat as a singular point")
    curvature = float(surface.gaussian_curvature(u))
    if not math.isfinite(curvature) or abs(curvature) * length ** 2 > curvature_bound:
        raise SingularityRefusal("curvature_blowup", f"{surface.name}: |K| = {abs(curvature):.3g} exceeds the "
                                 f"declared bound {curvature_bound:.3g}")
    try:
        surface.check(u)
    except SingularityRefusal:
        raise
    except SurfaceRefusal as exc:
        raise SingularityRefusal("chart_refused", str(exc)) from exc
    return {"det": float(eig[0] * eig[1]), "condition": condition, "curvature": curvature}


def refusal_code(function, *args, **kwargs) -> str:
    """Run a guarded call and return its refusal code, or 'accepted'."""
    try:
        function(*args, **kwargs)
    except SingularityRefusal as exc:
        return exc.code
    except SurfaceRefusal:
        return "surface_refusal"
    return "accepted"
