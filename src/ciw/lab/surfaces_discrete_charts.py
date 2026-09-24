"""Chart atlases with exact transitions, and coordinate-versus-curvature singularity scans.

Scope: ``Atlas`` holds charts of one embedded surface, each with a closed-form
inverse of its embedding. Transition maps go through the embedding: a point
is mapped by the inverse of the target chart and a velocity by
v_B = g_B^{-1} J_B^T J_A v_A, which is exact because J_A v_A lies in the
common tangent plane. A chart's regularity is the scale-free inverse
condition number lambda_min / lambda_max of its metric. ``Atlas.from_chart_maps``
builds an atlas from ``ChartMap`` reparametrizations of a base chart (their
``inverse`` composed with the base inverse); ``graph_atlas`` is the Monge chart
of a graph surface with its polar reparametrization, and ``SphereAtlas`` is
the core polar chart A of the sphere of radius R with the same chart rigidly
rotated so that its poles B lie on the equator of A
(``ciw.lab.surfaces.Rotated``). Geodesics are integrated with the core RK4
step, switching charts between steps when the active chart's regularity falls
below a threshold.

The singularity scan follows a path into a candidate point and fits power
laws for det g, the metric condition number, the largest Christoffel symbol,
|K| and the radial speed; a geodesic-circle circumference ratio separates
removable coordinate singularities from conical points, and the radial-speed
exponent separates boundaries at infinite distance. ``require_regular`` is the
pointwise guard; it refuses with a coded core ``SurfaceRefusal``.

Detection limits (stated, and exercised as counterexamples in T037): a
curvature blow-up slower than r^0.05 over the fit window reads as bounded; a
radial-speed exponent within 1e-3 of -1 reads as divergent; a circumference
deficit below 1e-6 reads as removable; a metric blow-up at finite distance
with bounded K is left unclassified even when it is a removable coordinate
singularity; and the approach loops must be preimages of geodesic circles
about the candidate point, which the caller chooses (``Approach.polar``) from
knowledge of the chart.

Non-claims: normalized mathematical surfaces only. Atlases need an embedding
and a closed-form inverse per chart; intrinsic charts are not covered. The
singularity rules assume rotationally symmetric approaches whose radial chart
lines are geodesics; they are not a proof of the singularity type of an
arbitrary surface or of a measured one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .integrators import integrate_fixed, step_rk4
from .surfaces import MongeSurface, Reparametrized, Rotated, Sphere, Surface, SurfaceRefusal
from .surfaces_discrete_geometry import PolarChart

# Rotation about the y axis by +pi/2: maps e_z to e_x, so chart B's poles are (+-R, 0, 0).
ROTATION_B = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
SWITCH_THRESHOLD = 0.25


def inverse_condition(g) -> float:
    """lambda_min / lambda_max of a metric: scale free, 1 for a conformal chart, 0 where it degenerates."""
    g = np.asarray(g, dtype=float)
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    return float(eig[0] / eig[1]) if eig[1] > 0 else 0.0


class Atlas:
    """Charts of one embedded surface, each with a closed-form inverse of its embedding."""

    def __init__(self, charts: dict, inverses: dict, threshold=SWITCH_THRESHOLD):
        if len(charts) < 2 or set(charts) != set(inverses):
            raise SurfaceRefusal("An atlas needs at least two charts, each with an inverse", "invalid_parameter")
        if not 0 < threshold < 1:
            raise SurfaceRefusal("Chart-switch threshold must lie in (0, 1)", "invalid_parameter")
        self.charts, self.inverses, self.threshold = dict(charts), dict(inverses), float(threshold)

    @classmethod
    def from_chart_maps(cls, base: Surface, base_inverse: Callable, maps: dict, threshold=SWITCH_THRESHOLD):
        """Charts Reparametrized(base, map) (the base itself for None), inverted by map.inverse(base_inverse(X))."""
        charts, inverses = {}, {}
        for name, chart_map in maps.items():
            if chart_map is None:
                charts[name], inverses[name] = base, base_inverse
            else:
                charts[name] = Reparametrized(base, chart_map)
                inverses[name] = (lambda m: lambda point: m.inverse(np.asarray(base_inverse(point), dtype=float)))(
                    chart_map)
        return cls(charts, inverses, threshold)

    def describe(self) -> dict:
        return {"threshold": self.threshold, "regularity": "lambda_min / lambda_max of g",
                "charts": {name: self.charts[name].describe() for name in sorted(self.charts)}}

    def embedding(self, name, u) -> np.ndarray:
        return self.charts[name].embedding(u)

    def to_chart(self, name, point) -> np.ndarray:
        return np.asarray(self.inverses[name](np.asarray(point, dtype=float)), dtype=float)

    def regularity(self, name, u) -> float:
        """Scale-free inverse condition number of the chart metric at u."""
        return inverse_condition(self.charts[name].metric(u))

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


class SphereAtlas(Atlas):
    """Charts A (core polar chart) and B (A rotated so its poles lie on A's equator).

    In each chart g = R^2 diag(1, sin^2 theta), so the regularity is
    sin^2 theta = det g / R^4, and sin^2 theta_A + sin^2 theta_B = 1 + y^2 / R^2
    gives the better chart a regularity of at least 1/2 everywhere.
    """

    def __init__(self, radius=1.0, threshold=SWITCH_THRESHOLD):
        if not 0 < threshold < 0.5:
            raise SurfaceRefusal("Chart-switch threshold must lie in (0, 1/2), below the atlas covering bound",
                                 "invalid_parameter")
        self.radius = float(radius)
        self.rotations = {"A": np.eye(3), "B": ROTATION_B}
        charts = {"A": Sphere(radius), "B": Rotated(Sphere(radius), ROTATION_B)}
        inverses = {name: (lambda rotation: lambda point: self._polar_inverse(rotation, point))(self.rotations[name])
                    for name in charts}
        super().__init__(charts, inverses, threshold)

    @staticmethod
    def _polar_inverse(rotation, point) -> np.ndarray:
        """Closed-form inverse of a rotated polar chart: theta in [0, pi], phi in (-pi, pi]."""
        p = rotation.T @ np.asarray(point, dtype=float)
        return np.array([math.atan2(math.hypot(p[0], p[1]), p[2]), math.atan2(p[1], p[0])])

    def describe(self) -> dict:
        return {"radius": self.radius, "threshold": self.threshold, "regularity": "lambda_min / lambda_max of g",
                "charts": {"A": "polar (theta, phi), poles (0, 0, +-R)",
                           "B": "polar chart rotated by +pi/2 about y, poles (+-R, 0, 0)"}}


def graph_atlas(surface: MongeSurface, threshold=SWITCH_THRESHOLD) -> Atlas:
    """The Monge chart (x, y) of a graph surface and its polar reparametrization (r, t), singular at r = 0."""
    return Atlas.from_chart_maps(surface, lambda point: np.asarray(point, dtype=float)[:2],
                                 {"monge": None, "polar": PolarChart()}, threshold)


def integrate_atlas(atlas: Atlas, chart, u0, v0, length, steps) -> dict:
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


# Classification thresholds. Power laws are fitted on r <= FIT_WINDOW, where a
# smooth quantity q(r) = q0 (1 + c r) has log-slope at most |c| FIT_WINDOW, so
# a regular point reads as a blow-up only if K varies on chart scales below
# FIT_WINDOW / |CURVATURE_BLOWUP|, i.e. about 2e-4.
FIT_WINDOW = 1e-5
FIT_RESIDUAL = 0.05          # max |log q - fit| for a clean power law
CURVATURE_BLOWUP = -0.05     # |K| ~ r^a with a <= this counts as unbounded
DEGENERACY = 0.5             # det ~ r^a with a >= this, or cond ~ r^-a
DIVERGENCE_TOLERANCE = 1e-3  # radial speed ~ r^b: distance diverges iff b <= -1
CONICAL_TOLERANCE = 1e-6     # |C / (2 pi rho) - 1| above this is a cone deficit
CLASSES = ("regular", "coordinate_singularity", "conical_singularity", "curvature_singularity",
           "infinite_distance_boundary", "unclassified")


def point_invariants(surface: Surface, u) -> dict:
    g = surface.metric(u)
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    gamma = surface.christoffel(u)
    return {"det": float(g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]), "condition": float(eig[1] / eig[0]),
            "christoffel": float(np.max(np.abs(gamma))), "curvature": float(surface.gaussian_curvature(u))}


def _fit(r, values) -> tuple:
    """Least-squares power-law exponent and max log residual.

    (None, 0.0) for an identically zero quantity; (None, inf) when only some
    values vanish or any is nonfinite, which is not a power law.
    """
    values = np.abs(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(values)):
        return None, math.inf
    if np.all(values == 0):
        return None, 0.0
    if np.any(values == 0):
        return None, math.inf
    x, y = np.log(r), np.log(values)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(np.max(np.abs(y - (slope * x + intercept))))


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
    """Invariants along the approach, fitted exponents and residuals, and the classification."""
    r = scan_distances() if distances is None else np.asarray(distances, dtype=float)
    rows = [point_invariants(approach.surface, approach.point(ri)) for ri in r]
    table = {key: [row[key] for row in rows] for key in ("det", "condition", "christoffel", "curvature")}
    w = approach.radial_velocity()
    table["radial_speed"] = [math.sqrt(float(w @ approach.surface.metric(approach.point(ri)) @ w)) for ri in r]
    window = r <= FIT_WINDOW * (1 + 1e-9)
    fits = {key: _fit(r[window], np.array(values)[window]) for key, values in table.items()}
    exponents = {key: slope for key, (slope, _) in fits.items()}
    residuals = {key: residual for key, (_, residual) in fits.items()}
    ratio = (circumference_ratio(approach, float(r.min()))
             if exponents["det"] is None or exponents["det"] >= -DEGENERACY else None)
    result = {"approach": approach.name, "distances": [float(v) for v in r], "fit_window": FIT_WINDOW,
              "table": table, "exponents": exponents,
              "fit_residuals": {k: (v if math.isfinite(v) else "not_a_power_law") for k, v in residuals.items()},
              "circumference_ratio": ratio}
    result["classification"] = classify(result)
    return result


def classify(result) -> str:
    """One of CLASSES, by these rules in order.

    1. Any rule quantity that is not a clean power law: ``unclassified``.
    2. |K| ~ r^a with a <= CURVATURE_BLOWUP: ``curvature_singularity``.
    3. Radial speed ~ r^b with b <= -1 + DIVERGENCE_TOLERANCE (the radial
       length diverges): ``infinite_distance_boundary``.
    4. det g blows up at finite distance with bounded K: ``unclassified`` (a
       removable blow-up chart and a genuine singularity look alike here).
    5. det g -> 0 or cond g -> infinity: ``conical_singularity`` if the
       circumference ratio differs from 1 by more than CONICAL_TOLERANCE,
       else ``coordinate_singularity``.
    6. Otherwise ``regular``.
    """
    e, residuals = result["exponents"], result["fit_residuals"]
    if any(residuals[key] == "not_a_power_law" or residuals[key] > FIT_RESIDUAL
           for key in ("det", "condition", "curvature", "radial_speed")):
        return "unclassified"
    det, cond, curvature, speed = e["det"], e["condition"], e["curvature"], e["radial_speed"]
    if curvature is not None and curvature <= CURVATURE_BLOWUP:
        return "curvature_singularity"
    if speed is not None and speed <= -1.0 + DIVERGENCE_TOLERANCE:
        return "infinite_distance_boundary"
    if det is not None and det <= -DEGENERACY:
        return "unclassified"
    if (det is not None and det >= DEGENERACY) or (cond is not None and cond <= -DEGENERACY):
        ratio = result["circumference_ratio"]
        if ratio is not None and abs(ratio - 1.0) > CONICAL_TOLERANCE:
            return "conical_singularity"
        return "coordinate_singularity"
    return "regular"


def require_regular(surface: Surface, u, *, max_condition=1e8, curvature_bound=1e6, length=1.0) -> dict:
    """Pointwise guard: refuse degenerate metrics and curvature above a bound with a coded SurfaceRefusal.

    Codes computed here: ``nonfinite_point``, ``nonfinite_metric``,
    ``degenerate_metric`` (indefinite, or condition number above
    ``max_condition``: a coordinate or conical singularity) and
    ``curvature_blowup`` (|K| length^2 above ``curvature_bound``; a slower
    blow-up passes). A refusal raised by the surface itself (for example
    ``curvature_singularity`` at a declared apex) or by the core
    ``Surface.check`` (``outside_chart``, ``degenerate_metric``) propagates
    unchanged with its own code.
    """
    u = np.asarray(u, dtype=float)
    if u.shape != (2,) or not np.all(np.isfinite(u)):
        raise SurfaceRefusal("Surface coordinates must be two finite numbers", "nonfinite_point")
    g = surface.metric(u)
    if not np.all(np.isfinite(g)):
        raise SurfaceRefusal(f"{surface.name}: metric is not finite at {u.tolist()}", "nonfinite_metric")
    eig = np.linalg.eigvalsh(0.5 * (g + g.T))
    if not eig[0] > 0:
        raise SurfaceRefusal(f"{surface.name}: metric is not positive definite at {u.tolist()}", "degenerate_metric")
    condition = float(eig[1] / eig[0])
    if condition > max_condition:
        raise SurfaceRefusal(f"{surface.name}: metric condition number {condition:.3g} exceeds "
                             f"{max_condition:.3g}; change chart or treat as a singular point", "degenerate_metric")
    curvature = float(surface.gaussian_curvature(u))
    if not math.isfinite(curvature) or abs(curvature) * length ** 2 > curvature_bound:
        raise SurfaceRefusal(f"{surface.name}: |K| = {abs(curvature):.3g} exceeds the declared bound "
                             f"{curvature_bound:.3g}", "curvature_blowup")
    surface.check(u)
    return {"det": float(eig[0] * eig[1]), "condition": condition, "curvature": curvature}


def refusal_code(function, *args, **kwargs) -> str:
    """Run a guarded call and return its refusal code, or 'accepted'."""
    try:
        function(*args, **kwargs)
    except SurfaceRefusal as exc:
        return exc.code
    return "accepted"
