"""Geodesics with their normal Jacobi fields and transfer matrices.

Along a unit-speed geodesic on a surface, a normal Jacobi field J = j N obeys
j'' + K(gamma(s)) j = 0. The transfer matrix

    Phi(s) = [[j_lat(s),  j_head(s)],
              [j_lat'(s), j_head'(s)]]

has a lateral-displacement column (j(0) = 1, j'(0) = 0) and a heading column
(j(0) = 0, j'(0) = 1). Its determinant is the Wronskian, identically one.
Conjugate points are zeros of j_head; focal points of the initial normal
geodesic are zeros of j_lat.
"""
from __future__ import annotations

import math

import numpy as np

from .integrators import hermite_zeros, integrate_adaptive, integrate_fixed


def initial_state(surface, u0, heading):
    """Geodesic state plus Jacobi initial data for both columns."""
    u0 = np.asarray(u0, dtype=float)
    surface.check(u0)
    t0 = surface.unit_tangent(u0, heading)
    return np.concatenate([u0, t0, [1.0, 0.0, 0.0, 1.0]])


def rhs(surface):
    """Augmented system: geodesic (4) + lateral column (2) + heading column (2)."""
    def f(y):
        geodesic = surface.geodesic_rhs(y[:4])
        curvature = surface.gaussian_curvature(y[:2])
        return np.concatenate([geodesic, [y[5], -curvature * y[4], y[7], -curvature * y[6]]])
    return f


class Transfer:
    """Sampled geodesic and transfer matrix along [0, length]."""

    def __init__(self, surface, s, states, stats=None):
        self.surface, self.s, self.states, self.stats = surface, s, states, stats or {}

    @property
    def points(self):
        return self.states[:, :2]

    @property
    def velocities(self):
        return self.states[:, 2:4]

    def matrix(self, index=-1) -> np.ndarray:
        y = self.states[index]
        return np.array([[y[4], y[6]], [y[5], y[7]]])

    def determinant(self) -> np.ndarray:
        y = self.states
        return y[:, 4] * y[:, 7] - y[:, 6] * y[:, 5]

    def speed_drift(self) -> np.ndarray:
        """|g(v, v) - 1| at every node, with no renormalization applied."""
        return np.array([abs(self.surface.speed_squared(y[:2], y[2:4]) - 1.0) for y in self.states])

    def _in_travel_order(self, zeros):
        """Zeros ordered as the geodesic meets them (descending s for a backward transfer)."""
        return sorted(zeros, key=lambda z: abs(z - self.s[0]))

    def conjugate_points(self):
        """Zeros of the heading column other than the trivial zero at s = 0, in travel order."""
        return self._in_travel_order(z for z in hermite_zeros(self.s, self.states[:, 6], self.states[:, 7])
                                     if z != self.s[0])

    def focal_points(self):
        return self._in_travel_order(hermite_zeros(self.s, self.states[:, 4], self.states[:, 5]))

    def curvature_along(self) -> np.ndarray:
        return np.array([self.surface.gaussian_curvature(y[:2]) for y in self.states])


def transfer(surface, u0, heading, length, steps=None, method="rk4", rtol=None, atol=1e-12):
    """Integrate a geodesic and both Jacobi columns; fixed-step unless ``rtol`` is given."""
    y0 = initial_state(surface, u0, heading)
    f = rhs(surface)
    if rtol is not None:
        s, states, stats = integrate_adaptive(f, y0, length, rtol=rtol, atol=atol)
        return Transfer(surface, s, states, stats)
    s, states = integrate_fixed(f, y0, length, steps, method)
    return Transfer(surface, s, states, {"steps": steps, "method": method})


def constant_curvature(curvature, s):
    """Exact (j_lat, j_lat', j_head, j_head') for constant K."""
    s = np.asarray(s, dtype=float)
    if curvature > 0:
        w = math.sqrt(curvature)
        return np.cos(w * s), -w * np.sin(w * s), np.sin(w * s) / w, np.cos(w * s)
    if curvature < 0:
        w = math.sqrt(-curvature)
        return np.cosh(w * s), w * np.sinh(w * s), np.sinh(w * s) / w, np.cosh(w * s)
    return np.ones_like(s), np.zeros_like(s), s.copy(), np.ones_like(s)


def perturbed_start(surface, u0, heading, lateral=0.0, heading_change=0.0):
    """Start state displaced laterally along a normal geodesic and/or rotated in heading.

    The lateral start is exp_{u0}(lateral N0) with the tangent parallel
    transported along that short normal geodesic, so both perturbations are
    exact geometric operations rather than chart-dependent shifts.
    """
    u0 = np.asarray(u0, dtype=float)
    t0 = surface.unit_tangent(u0, heading)
    n0 = surface.normal(u0, t0)
    if lateral:
        steps = max(8, int(math.ceil(abs(lateral) / 1e-3)))
        # Carry the tangent as a parallel-transported vector along the normal geodesic.
        def f(y):
            u, v, w = y[:2], y[2:4], y[4:6]
            gamma = surface.christoffel(u)
            return np.concatenate([v, -np.einsum("kij,i,j->k", gamma, v, v), -np.einsum("kij,i,j->k", gamma, v, w)])
        y0 = np.concatenate([u0, n0 * math.copysign(1.0, lateral), t0])
        _, states = integrate_fixed(f, y0, abs(lateral), steps, "rk4")
        u0, t0 = states[-1, :2], states[-1, 4:6]
    if heading_change:
        n0 = surface.normal(u0, t0)
        t0 = math.cos(heading_change) * t0 + math.sin(heading_change) * n0
    return np.concatenate([u0, t0])


def normal_separation(surface, base_states, perturbed_states):
    """First-order normal separation g(delta u, N) at matched arclength nodes."""
    out = np.empty(len(base_states))
    for i, (y, z) in enumerate(zip(base_states, perturbed_states)):
        u, v = y[:2], y[2:4]
        out[i] = surface.inner(u, z[:2] - u, surface.normal(u, v))
    return out
