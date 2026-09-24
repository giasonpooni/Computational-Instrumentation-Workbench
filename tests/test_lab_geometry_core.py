"""Regression tests for the shared geometry core: surfaces, integrators and Jacobi transfers.

Each test pins one reviewed defect of ``ciw.lab.surfaces``, ``ciw.lab.integrators``
or ``ciw.lab.jacobi``: closed forms are checked against an independent numerical
integration, refusals by their SurfaceRefusal code, and the integrators against
their own fixed-step or time-reversed counterparts.
"""
import math

import numpy as np
import pytest

from ciw.lab import integrators, jacobi
from ciw.lab.surfaces import (ChartMap, HyperbolicPlane, Plane, Reparametrized, Rotated, Sphere, Surface,
                              SurfaceRefusal, Torus, rotation_matrix)


class IdentityChart(ChartMap):
    name = "identity"

    def forward(self, a):
        return np.asarray(a, dtype=float)

    def jacobian(self, a):
        return np.eye(2)

    def hessian(self, a):
        return np.zeros((2, 2, 2))


class PolarChart(ChartMap):
    name = "polar"

    def forward(self, a):
        return np.array([a[0] * math.cos(a[1]), a[0] * math.sin(a[1])])

    def jacobian(self, a):
        r, t = a
        return np.array([[math.cos(t), -r * math.sin(t)], [math.sin(t), r * math.cos(t)]])

    def hessian(self, a):
        r, t = a
        c, s = math.cos(t), math.sin(t)
        return np.array([[[0.0, -s], [-s, -r * c]], [[0.0, c], [c, -r * s]]])


class PoleNormalChart(ChartMap):
    """a = (p, q) -> (theta, phi) = (|a|, atan2(q, p)): normal coordinates about the sphere's north pole.

    Over the unit sphere the pullback metric is close to I near a = 0, where the
    base (theta, phi) chart has its coordinate singularity.
    """

    name = "pole-normal"

    def forward(self, a):
        return np.array([np.hypot(a[0], a[1]), np.arctan2(a[1], a[0])])

    def jacobian(self, a):
        p, q = a
        r = np.hypot(p, q)
        return np.array([[p / r, q / r], [-q / r ** 2, p / r ** 2]])

    def hessian(self, a):
        p, q = a
        r = np.hypot(p, q)
        return np.array([[[q * q / r ** 3, -p * q / r ** 3], [-p * q / r ** 3, p * p / r ** 3]],
                         [[2 * p * q / r ** 4, (q * q - p * p) / r ** 4],
                          [(q * q - p * p) / r ** 4, -2 * p * q / r ** 4]]])


class ExpChart(ChartMap):
    """a -> (a1, exp(a2)): finite everywhere in exact arithmetic, but overflows to inf for a2 > ~709."""

    name = "exp"

    def forward(self, a):
        with np.errstate(over="ignore"):
            return np.array([a[0], np.exp(a[1])])

    def jacobian(self, a):
        with np.errstate(over="ignore"):
            return np.array([[1.0, 0.0], [0.0, np.exp(a[1])]])

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        with np.errstate(over="ignore"):
            hess[1, 1, 1] = np.exp(a[1])
        return hess


class ConstantMetric(Surface):
    name = "constant-metric"

    def __init__(self, g):
        self.g = np.asarray(g, dtype=float)

    def metric(self, u):
        return self.g

    def metric_derivatives(self, u):
        return np.zeros((2, 2, 2))

    def gaussian_curvature(self, u):
        return 0.0


def refusal_code(function, *args):
    try:
        function(*args)
    except SurfaceRefusal as refusal:
        return refusal.code
    return "accepted"


# ---------------------------------------------------------------- HyperbolicPlane.exact_geodesic
NEAR_VERTICAL = [math.pi / 2 - 1e-6, math.pi / 2 - 1e-9, math.pi / 2 - 1e-12, math.pi / 2 + 1e-8,
                 -math.pi / 2 + 1e-7, -math.pi / 2 - 1e-10]


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("heading", [0.0, 0.6, 2.5, -2.0] + NEAR_VERTICAL)
def test_hyperbolic_closed_form_matches_integration_for_every_heading(k, heading):
    surf = HyperbolicPlane(k)
    u0 = np.array([0.3, 0.7])
    t0 = surf.unit_tangent(u0, heading)
    s_values = [0.0, 0.4, 1.3, -0.9]
    exact = surf.exact_geodesic(u0, t0, s_values)
    # The curve passes through its own start point (to rounding) for every heading.
    assert abs(exact[0, 0] - u0[0]) <= 1e-15 * u0[1] and exact[0, 1] == pytest.approx(u0[1], rel=1e-15)
    for s, point in zip(s_values[1:], exact[1:]):
        # gamma(-s; t0) = gamma(s; -t0): the reference always integrates forward.
        start = np.concatenate([u0, math.copysign(1.0, s) * t0])
        _, states, _ = integrators.integrate_adaptive(surf.geodesic_rhs, start, abs(s), rtol=1e-13, atol=1e-15)
        assert abs(point[0] - states[-1, 0]) <= 1e-10 * u0[1]
        assert point[1] == pytest.approx(states[-1, 1], rel=1e-10)


def test_hyperbolic_closed_form_saturates_on_long_arcs():
    surf = HyperbolicPlane(1.0)
    u0 = np.array([0.0, 1.0])
    for heading in (0.6, math.pi / 2 - 1e-12, -math.pi / 2 + 1e-9):
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            far = surf.exact_geodesic(u0, surf.unit_tangent(u0, heading), [-900.0, 400.0, 900.0])
        assert np.all(np.isfinite(far)) and np.all(far[:, 1] >= 0)
    vertical = surf.exact_geodesic(u0, np.array([0.0, 1.0]), [2.0, -2.0])
    np.testing.assert_allclose(vertical, [[0.0, math.exp(2.0)], [0.0, math.exp(-2.0)]], rtol=1e-15)


# ---------------------------------------------------------------- Surface.check
def test_singular_metric_test_is_scale_relative():
    # A millimetre sphere is regular away from its poles, like the unit sphere.
    for radius in (1e-6, 1e-3, 1.0, 1e3):
        sphere = Sphere(radius)
        assert refusal_code(sphere.check, np.array([1.0, 0.3])) == "accepted"
        assert refusal_code(sphere.check, np.array([1e-7, 0.3])) == "degenerate_metric"
    transfer = jacobi.transfer(Sphere(1e-3), [1.0, 0.3], 0.4, 1e-3, steps=10)
    np.testing.assert_allclose(transfer.matrix(), [[math.cos(1.0), math.sin(1.0) * 1e-3],
                                                   [-math.sin(1.0) / 1e-3, math.cos(1.0)]], rtol=1e-5)
    # The same decisions at every scale along the pole approach.
    thetas = [10.0 ** (-k / 4) for k in range(4, 33)]
    decisions = {radius: [refusal_code(Sphere(radius).check, np.array([t, 0.3])) for t in thetas]
                 for radius in (1e-4, 1.0, 1e4)}
    assert decisions[1e-4] == decisions[1.0] == decisions[1e4]
    assert set(decisions[1.0]) == {"accepted", "degenerate_metric"}
    # Genuine degeneracies stay refused at any scale: near rank one, indefinite, negative definite, zero, nonfinite.
    for g in ([[1e-9, 0.0], [0.0, 1e-9 * 1e-13]], [[1.0, 0.0], [0.0, -1.0]], [[-1.0, 0.0], [0.0, -2.0]],
              [[0.0, 0.0], [0.0, 0.0]], [[math.nan, 0.0], [0.0, 1.0]]):
        assert refusal_code(ConstantMetric(g).check, np.zeros(2)) == "degenerate_metric"
    assert refusal_code(ConstantMetric([[1e-9, 0.0], [0.0, 1e-9]]).check, np.zeros(2)) == "accepted"


# ---------------------------------------------------------------- Reparametrized.check
def test_reparametrized_check_delegates_to_the_base_chart():
    base = HyperbolicPlane(1.0)
    through_identity = Reparametrized(base, IdentityChart())
    for point in ([0.0, -1.0], [0.0, 0.0], [0.0, 2000.0], [0.5, 1e-4]):
        point = np.array(point)
        assert refusal_code(through_identity.check, point) == refusal_code(base.check, point)
    assert refusal_code(through_identity.check, np.array([0.0, -1.0])) == "outside_chart"
    assert refusal_code(through_identity.check, np.array([math.nan, 1.0])) == "nonfinite_point"
    # The chart map itself is still checked: polar coordinates degenerate at r = 0 over a regular plane.
    polar = Reparametrized(Plane(), PolarChart())
    assert refusal_code(polar.check, np.array([0.0, 0.3])) == "degenerate_metric"
    assert refusal_code(polar.check, np.array([1.0, 0.3])) == "accepted"
    # A regular chart over a base singularity is refused by the base.
    assert refusal_code(Reparametrized(Sphere(1.0), IdentityChart()).check, np.array([1e-7, 0.3])) == \
        "degenerate_metric"
    with pytest.raises(SurfaceRefusal) as refused:
        jacobi.initial_state(through_identity, [0.0, -1.0], 0.3)
    assert refused.value.code == "outside_chart"
    # A finite chart point whose chart image overflows is an unbounded chart map, named as such.
    with pytest.raises(SurfaceRefusal) as refused:
        Reparametrized(Plane(), ExpChart()).check(np.array([0.0, 800.0]))
    assert refused.value.code == "degenerate_metric" and "plane∘exp" in str(refused.value) \
        and "chart map is not finite at [0.0, 800.0]" in str(refused.value)
    assert refusal_code(Reparametrized(Plane(), ExpChart()).check, np.array([0.0, 1.0])) == "accepted"


def test_chart_that_removes_a_base_coordinate_singularity_is_regular():
    # The base (theta, phi) chart refuses 1e-7 from the pole; normal coordinates there are regular (J^T g J ~ I).
    sphere = Sphere(1.0)
    pole = Reparametrized(sphere, PoleNormalChart())
    length, heading = 1.0, 0.3
    for distance in (1e-3, 1e-7, 1e-9):
        a0 = distance * np.array([math.cos(0.4), math.sin(0.4)])
        assert refusal_code(sphere.check, PoleNormalChart().forward(a0)) == ("accepted" if distance > 1e-5
                                                                             else "degenerate_metric")
        np.testing.assert_allclose(pole.metric(a0), np.eye(2), atol=1e-6)
        assert refusal_code(pole.check, a0) == "accepted"
        # Accurate there: the transfer matches the great-circle closed form.
        run = jacobi.transfer(pole, a0, heading, length, steps=200)
        x0, t0 = pole.embedding(a0), pole.embedding_jacobian(a0) @ run.states[0, 2:4]
        exact_end = math.cos(length) * x0 + math.sin(length) * t0
        assert np.linalg.norm(pole.embedding(run.states[-1, :2]) - exact_end) < 1e-9
        np.testing.assert_allclose(run.matrix(), [[math.cos(length), math.sin(length)],
                                                  [-math.sin(length), math.cos(length)]], atol=1e-10)
    # The chart's own singularity at its centre is still refused, by the pullback test.
    with np.errstate(divide="ignore", invalid="ignore"):
        assert refusal_code(pole.check, np.zeros(2)) == "degenerate_metric"
    # The base domain still propagates through such a chart.
    assert refusal_code(Reparametrized(HyperbolicPlane(1.0), PoleNormalChart()).check,
                        np.array([1.0, -0.5])) == "outside_chart"


# ---------------------------------------------------------------- Rotated
def test_rotated_requires_an_orthogonal_proper_matrix():
    rotation = rotation_matrix([0.3, -0.5, 0.8], 1.1)
    Rotated(Torus(2.0, 1.0), rotation)
    Rotated(Torus(2.0, 1.0), np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    skew = np.eye(3)
    skew[0, 1] = 1e-9
    for bad in (1.000004 * rotation, 0.99 * rotation, rotation @ skew, np.diag([1.0, 1.0, -1.0]),
                np.full((3, 3), math.nan), np.eye(2)):
        with pytest.raises(SurfaceRefusal):
            Rotated(Torus(2.0, 1.0), bad)


# ---------------------------------------------------------------- integrate_adaptive
def test_adaptive_negative_length_integrates_backward_like_fixed_step():
    def f(y):
        return np.array([y[1], -y[0]])

    y0 = np.array([0.3, 1.0])
    s, states, stats = integrators.integrate_adaptive(f, y0, -2.0, rtol=1e-11, atol=1e-13)
    assert s[0] == 0.0 and s[-1] == -2.0 and np.all(np.diff(s) < 0) and stats["accepted_steps"] > 1
    exact = np.array([0.3 * math.cos(2.0) - math.sin(2.0), 0.3 * math.sin(2.0) + math.cos(2.0)])
    np.testing.assert_allclose(states[-1], exact, atol=1e-9)
    _, fixed = integrators.integrate_fixed(f, y0, -2.0, 400)
    np.testing.assert_allclose(states[-1], fixed[-1], atol=1e-9)
    # Backward in s is forward for the reversed field, step for step.
    s_rev, reversed_states, reversed_stats = integrators.integrate_adaptive(lambda y: -f(y), y0, 2.0,
                                                                            rtol=1e-11, atol=1e-13)
    assert np.array_equal(s, -s_rev) and np.array_equal(states, reversed_states) and stats == reversed_stats
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            integrators.integrate_adaptive(f, y0, bad)


def test_backward_transfer_matches_fixed_step_and_orders_zeros_by_travel():
    sphere = Sphere(1.0)
    adaptive = jacobi.transfer(sphere, [1.0, 0.3], 0.4, -1.0, rtol=1e-10)
    fixed = jacobi.transfer(sphere, [1.0, 0.3], 0.4, -1.0, steps=200)
    assert adaptive.s[-1] == -1.0 and len(adaptive.s) > 2
    np.testing.assert_allclose(adaptive.matrix(), fixed.matrix(), atol=1e-8)
    np.testing.assert_allclose(adaptive.matrix(), [[math.cos(1.0), -math.sin(1.0)], [math.sin(1.0), math.cos(1.0)]],
                               atol=1e-8)
    start = [math.pi / 2, 0.0]
    backward = jacobi.transfer(sphere, start, 0.3, -7.0, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(backward.conjugate_points(), [-math.pi, -2 * math.pi], atol=1e-7)
    np.testing.assert_allclose(backward.focal_points(), [-math.pi / 2, -1.5 * math.pi], atol=1e-7)
    forward = jacobi.transfer(sphere, start, 0.3, 7.0, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(forward.conjugate_points(), [math.pi, 2 * math.pi], atol=1e-7)
    np.testing.assert_allclose(forward.focal_points(), [math.pi / 2, 1.5 * math.pi], atol=1e-7)


# ---------------------------------------------------------------- observed_order
def test_observed_order_refuses_data_without_a_logarithm():
    assert integrators.observed_order([0.1, 0.05, 0.025], [1e-4, 6.25e-6, 3.90625e-7]) == pytest.approx(4.0)
    # As documented, a repeated step size is fitted once two distinct step sizes are present.
    assert integrators.observed_order([0.1, 0.1, 0.05], [1e-4, 1e-4, 6.25e-6]) == pytest.approx(4.0)
    for steps, errors in (([0.1, 0.05, 0.025], [1e-4, 6.25e-6, 0.0]),
                          ([0.1, 0.05, 0.025], [1e-4, -6.25e-6, 1e-7]),
                          ([0.1, 0.05, 0.025], [1e-4, math.nan, 1e-7]),
                          ([0.1, 0.05, 0.025], [1e-4, math.inf, 1e-7]),
                          ([0.1, 0.0, 0.025], [1e-4, 6.25e-6, 1e-7]),
                          ([0.1, 0.05], [1e-4, 6.25e-6, 1e-7]),
                          ([0.1, 0.1], [1e-4, 2e-4]),
                          ([0.1], [1e-4])):
        with pytest.raises(ValueError):
            integrators.observed_order(steps, errors)
