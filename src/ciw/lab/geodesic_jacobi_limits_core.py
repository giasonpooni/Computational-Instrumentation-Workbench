"""Helpers for the geodesic/Jacobi limit, invariance and counterexample experiments (T010-T018).

Scope: nonlinear chart maps with exact Jacobians and Hessians, heading and
lateral perturbation families evaluated on one common arclength grid,
closed-form separations on the unit sphere and the hyperbolic plane,
constant-curvature Jacobi transfer matrices of explicit and implicit one-step
methods (in floating point and, for the explicit methods, in exact rational
arithmetic), log-log fits and check builders whose ``passed`` flag is computed
from the recorded numbers.

Non-claims: every surface, length, curvature and perturbation is in declared
normalized units. Nothing here measures or represents a physical surface or
sensor, and agreement between two ``ciw`` computations is self-consistency,
not independent verification.
"""
from __future__ import annotations

import math

import numpy as np

from . import evidence, integrators, jacobi
from .surfaces import ChartMap, EmbeddedSurface, SurfaceRefusal


# Checks whose pass flag is derived from their numbers ---------------------
def check(kind: str, reference: str, observed, tolerance, comparison: str = "abs_le") -> dict:
    """A check object whose ``passed`` flag is computed by :func:`ciw.lab.evidence.holds`.

    ``le``/``ge`` bound a nonnegative magnitude; signed differences use
    ``signed_le``/``signed_ge``. The evidence validator refuses a check whose
    comparison does not fit its observed value.
    """
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": bool(evidence.holds(observed, tolerance, comparison))}


def refusal_check(reference: str, expected: str, observed: str | None) -> dict:
    """Refusal check; ``observed`` is the refusal message, or 'none' when nothing was refused."""
    observed = "none" if observed is None else observed
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def uncertainty(kind: str, value, basis: str) -> dict:
    """Per-finding uncertainty: a number with its kind and the basis it was derived from."""
    return {"kind": kind, "value": float(value), "basis": basis}


def slope_spread(xs, ys) -> float:
    """Largest gap between the global log-log slope and the slopes of consecutive point pairs."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    fit = loglog_slope(xs, ys)
    local = np.diff(np.log(ys)) / np.diff(np.log(xs))
    return float(np.max(np.abs(local - fit)))


def loglog_slope(xs, ys) -> float:
    """Least-squares slope of log(y) against log(x); refuses nonpositive or nonfinite data."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or len(xs) < 2:
        raise ValueError("A log-log slope needs at least two paired points")
    if np.any(xs <= 0) or np.any(ys <= 0) or not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise ValueError("Log-log slopes need positive finite data")
    return integrators.observed_order(xs, ys)


def jsonable(value, digits: int | None = None):
    """Nested Python floats/ints/lists for findings and artifacts; optional significant-digit rounding."""
    if isinstance(value, dict):
        return {str(k): jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v, digits) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Findings and artifacts hold finite numbers only")
        if digits is None or value == 0.0:
            return value
        return float(f"{value:.{digits}g}")
    return value


class SecondFormCurvature(EmbeddedSurface):
    """An embedded surface whose K is recomputed from its second fundamental form.

    Core surfaces such as Plane and Cylinder return a closed-form K = 0; this
    wrapper delegates only the embedding derivatives, so a Jacobi integration on
    it tests the geometry of the embedding instead of a literal constant.
    """

    def __init__(self, base: EmbeddedSurface):
        self.base = base
        self.name = f"{base.name}(second-form K)"

    def embedding(self, u):
        return self.base.embedding(u)

    def first(self, u):
        return self.base.first(u)

    def second(self, u):
        return self.base.second(u)


def x_minus_sin(x: float) -> float:
    """x - sin(x) without cancellation (Taylor series below 0.1, where the direct form loses digits)."""
    if abs(x) >= 0.1:
        return x - math.sin(x)
    # Alternating terms x^(2n+1) / (2n+1)! from x^3/6 to x^13/13!; the first omitted one is < 1e-23 x^3.
    total, term = 0.0, x ** 3 / 6.0
    for n in range(1, 7):
        total += term
        term *= -x * x / ((2 * n + 2) * (2 * n + 3))
    return total


# Chart maps a -> u = phi(a), orientation preserving (det J > 0) -------------
class IdentityChart(ChartMap):
    """u = a; the pullback metric must reproduce the base chart bit for bit."""

    name = "identity"

    def forward(self, a):
        return np.array(a, dtype=float)

    def jacobian(self, a):
        return np.eye(2)

    def hessian(self, a):
        return np.zeros((2, 2, 2))

    def inverse(self, u):
        return np.array(u, dtype=float)


class PolynomialWarp(ChartMap):
    """u_i = c_i + a_i + beta a_i^3 (beta > 0, monotone in each coordinate)."""

    name = "polynomial-warp"

    def __init__(self, center, beta=0.3):
        if not beta > 0:
            raise SurfaceRefusal("Polynomial warp requires beta > 0")
        self.center, self.beta = np.array(center, dtype=float), float(beta)

    def forward(self, a):
        a = np.asarray(a, dtype=float)
        return self.center + a + self.beta * a ** 3

    def jacobian(self, a):
        a = np.asarray(a, dtype=float)
        return np.diag(1.0 + 3.0 * self.beta * a ** 2)

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        hess[0, 0, 0], hess[1, 1, 1] = 6.0 * self.beta * a[0], 6.0 * self.beta * a[1]
        return hess

    def inverse(self, u):
        w = np.asarray(u, dtype=float) - self.center
        # Both starts lie beyond the root on the convex side, so Newton converges monotonically.
        a = np.where(np.abs(w) > 1.0, np.cbrt(w / self.beta), w)
        for _ in range(60):
            a = a - (a + self.beta * a ** 3 - w) / (1.0 + 3.0 * self.beta * a ** 2)
        return a


class QuadraticShear(ChartMap):
    """u = c + (a1 + sigma a2^2, a2): a nonlinear shear with an exact inverse."""

    name = "quadratic-shear"

    def __init__(self, center, sigma=0.8):
        self.center, self.sigma = np.array(center, dtype=float), float(sigma)

    def forward(self, a):
        return self.center + np.array([a[0] + self.sigma * a[1] ** 2, a[1]])

    def jacobian(self, a):
        return np.array([[1.0, 2.0 * self.sigma * a[1]], [0.0, 1.0]])

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        hess[0, 1, 1] = 2.0 * self.sigma
        return hess

    def inverse(self, u):
        w = np.asarray(u, dtype=float) - self.center
        return np.array([w[0] - self.sigma * w[1] ** 2, w[1]])


class ExponentialStretch(ChartMap):
    """u_i = c_i + (exp(lam a_i) - 1) / lam; defined for u_i > c_i - 1/lam."""

    name = "exponential-stretch"

    def __init__(self, center, lam=0.6):
        if not lam > 0:
            raise SurfaceRefusal("Exponential stretch requires lam > 0")
        self.center, self.lam = np.array(center, dtype=float), float(lam)

    def forward(self, a):
        return self.center + np.expm1(self.lam * np.asarray(a, dtype=float)) / self.lam

    def jacobian(self, a):
        return np.diag(np.exp(self.lam * np.asarray(a, dtype=float)))

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        grow = self.lam * np.exp(self.lam * np.asarray(a, dtype=float))
        hess[0, 0, 0], hess[1, 1, 1] = grow[0], grow[1]
        return hess

    def inverse(self, u):
        w = self.lam * (np.asarray(u, dtype=float) - self.center)
        if np.any(w <= -1.0):
            raise SurfaceRefusal("Point is outside the exponential-stretch chart domain")
        return np.log1p(w) / self.lam


class NearFold(ChartMap):
    """u_axis = c + mu a + a^3 / 3: det J = mu + a^2 is small (but positive) near a = 0.

    The geometry is unchanged; only the chart velocity near the fold grows like 1/mu.
    """

    name = "near-fold"

    def __init__(self, center, mu=0.1, axis=0):
        if not mu > 0:
            raise SurfaceRefusal("Near-fold chart requires mu > 0 (mu = 0 is a true fold)")
        if axis not in (0, 1):
            raise ValueError("Fold axis must be 0 or 1")
        self.center, self.mu, self.axis = np.array(center, dtype=float), float(mu), axis

    def forward(self, a):
        u = self.center + np.asarray(a, dtype=float)
        x = a[self.axis]
        u[self.axis] = self.center[self.axis] + self.mu * x + x ** 3 / 3.0
        return u

    def jacobian(self, a):
        jac = np.eye(2)
        jac[self.axis, self.axis] = self.mu + a[self.axis] ** 2
        return jac

    def hessian(self, a):
        hess = np.zeros((2, 2, 2))
        hess[self.axis, self.axis, self.axis] = 2.0 * a[self.axis]
        return hess

    def inverse(self, u):
        a = np.asarray(u, dtype=float) - self.center
        w = a[self.axis]
        # Monotone cubic: Newton from the larger of the linear and cubic estimates converges.
        x = w / self.mu if abs(w) < self.mu ** 1.5 else math.copysign(abs(3.0 * w) ** (1 / 3), w)
        for _ in range(100):
            x -= (self.mu * x + x ** 3 / 3.0 - w) / (self.mu + x * x)
        a[self.axis] = x
        return a


def chart_start(base, chart, u0, heading):
    """Chart point and velocity carrying the same geometric initial data as (u0, heading) on ``base``."""
    u0 = np.asarray(u0, dtype=float)
    tangent = base.unit_tangent(u0, heading)
    if chart is None:
        return u0, tangent
    a0 = chart.inverse(u0)
    if np.max(np.abs(chart.forward(a0) - u0)) > 1e-12:
        raise SurfaceRefusal(f"{chart.name}: inverse did not reproduce the start point")
    return a0, np.linalg.solve(chart.jacobian(a0), tangent)


# Perturbation families on a common grid -----------------------------------
def embedded_normal(surface, state) -> np.ndarray:
    """Embedded unit normal of the geodesic inside the surface (tangent plane, +90 degrees)."""
    u, v = state[:2], state[2:4]
    return surface.embedding_jacobian(u) @ surface.normal(u, v)


def separation_family(surface, u0, heading, length, steps, eps_list, lateral_ratio=0.0):
    """Base transfer plus perturbed geodesics (lateral = ratio * eps, heading change = eps).

    All runs use RK4 on the same arclength grid, so node ``i`` of every run is
    the same arclength. Separations are embedded: the chord |X_eps - X| and its
    signed component along the base geodesic's in-surface normal.
    """
    base = jacobi.transfer(surface, u0, heading, length, steps=steps)
    points = np.array([surface.embedding(y[:2]) for y in base.states])
    normals = np.array([embedded_normal(surface, y) for y in base.states])
    runs = {}
    for eps in eps_list:
        start = jacobi.perturbed_start(surface, u0, heading, lateral=lateral_ratio * eps, heading_change=eps)
        _, states = integrators.integrate_fixed(surface.geodesic_rhs, start, length, steps, "rk4")
        delta = np.array([surface.embedding(y[:2]) for y in states]) - points
        runs[eps] = {"chord": np.linalg.norm(delta, axis=1), "signed": np.einsum("ij,ij->i", delta, normals)}
    first_order = lateral_ratio * base.states[:, 4] + base.states[:, 6]
    first_order_rate = lateral_ratio * base.states[:, 5] + base.states[:, 7]
    return {"s": base.s, "transfer": base, "j": first_order, "dj": first_order_rate, "points": points,
            "normals": normals, "runs": runs, "lateral_ratio": lateral_ratio, "steps": steps, "length": length}


def sphere_separation(sphere, u0, heading, s, eps, lateral_ratio=0.0) -> dict:
    """Closed-form separation of two great circles on a unit sphere.

    The perturbed start is exp_p0(delta N0) with the tangent parallel
    transported and then rotated by eps toward the new normal, exactly the
    construction of :func:`ciw.lab.jacobi.perturbed_start`.
    """
    if abs(sphere.radius - 1.0) > 0:
        raise SurfaceRefusal("Closed-form separations are declared for the unit sphere only")
    u0 = np.asarray(u0, dtype=float)
    s = np.atleast_1d(np.asarray(s, dtype=float))
    p0 = sphere.embedding(u0)
    t0 = sphere.embedding_jacobian(u0) @ sphere.unit_tangent(u0, heading)
    n0 = sphere.embedding_jacobian(u0) @ sphere.normal(u0, sphere.unit_tangent(u0, heading))
    delta = lateral_ratio * eps
    p1 = math.cos(delta) * p0 + math.sin(delta) * n0
    n1 = -math.sin(delta) * p0 + math.cos(delta) * n0
    t1 = math.cos(eps) * t0 + math.sin(eps) * n1
    base = np.cos(s)[:, None] * p0 + np.sin(s)[:, None] * t0
    moved = np.cos(s)[:, None] * p1 + np.sin(s)[:, None] * t1
    diff = moved - base
    chord = np.linalg.norm(diff, axis=1)
    # The in-surface normal of a great circle is the constant vector n0.
    return {"chord": chord, "signed": diff @ n0, "distance": 2.0 * np.arcsin(np.minimum(chord / 2.0, 1.0))}


def sphere_first_order(s, lateral_ratio=0.0):
    """First-order normal separation per unit eps on the unit sphere: ratio cos s + sin s."""
    s = np.asarray(s, dtype=float)
    return lateral_ratio * np.cos(s) + np.sin(s)


def hyperbolic_heading_distance(k, s, eps):
    """Distance between gamma(s) and gamma_eps(s) from a common start in curvature -k^2."""
    s = np.asarray(s, dtype=float)
    return (2.0 / k) * np.arcsinh(np.sinh(k * s) * math.sin(eps / 2.0))


def hyperbolic_distance(k, a, b) -> float:
    """Upper-half-plane distance for g = I / (k^2 y^2), stable for nearby points."""
    gap = math.hypot(a[0] - b[0], a[1] - b[1])
    return (2.0 / k) * math.asinh(gap / (2.0 * math.sqrt(a[1] * b[1])))


# Constant-curvature Jacobi transfer of one-step methods --------------------
def jacobi_generator(curvature) -> np.ndarray:
    """A with (j, j')' = A (j, j') for j'' + K j = 0."""
    return np.array([[0.0, 1.0], [-float(curvature), 0.0]])


def step_matrix(method: str, curvature, h) -> np.ndarray:
    """One-step transfer matrix of a method on the linear Jacobi system."""
    z = h * jacobi_generator(curvature)
    eye = np.eye(2)
    if method in integrators.ORDERS:
        order = integrators.ORDERS[method]
        out, term = eye.copy(), eye.copy()
        for j in range(1, order + 1):
            term = term @ z / j
            out = out + term
        return out
    if method == "implicit-midpoint":
        left = eye - 0.5 * z
        if abs(np.linalg.det(left)) < 1e-14:
            raise FloatingPointError("Implicit midpoint step is singular at this step size")
        return np.linalg.solve(left, eye + 0.5 * z)
    if method == "gauss-legendre-2":
        # Two-stage Gauss collocation: the (2, 2) Pade approximant of exp, order 4 and A-stable;
        # 1 - z/2 + z^2/12 has no real zero, so the step is never singular for real eigenvalues.
        quadratic = z @ z / 12.0
        return np.linalg.solve(eye - 0.5 * z + quadratic, eye + 0.5 * z + quadratic)
    raise ValueError(f"Unsupported Jacobi step method: {method}")


# Scalar stability functions R(z) of the same one-step methods (y' = lambda y, z = h lambda).
IMPLICIT_ORDERS = {"implicit-midpoint": 2, "gauss-legendre-2": 4}


def stability_function(method: str, z: float) -> float:
    """R(z) with y_{n+1} = R(h lambda) y_n; an independent evaluation path to :func:`step_matrix`."""
    if method in integrators.ORDERS:
        return sum(z ** j / math.factorial(j) for j in range(integrators.ORDERS[method] + 1))
    if method == "implicit-midpoint":
        return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    if method == "gauss-legendre-2":
        return (1.0 + 0.5 * z + z * z / 12.0) / (1.0 - 0.5 * z + z * z / 12.0)
    raise ValueError(f"Unsupported Jacobi step method: {method}")


def hyperbolic_j_head_by_modes(method: str, k: float, length: float, steps: int) -> float:
    """Numerical j_head(L) on K = -k^2 from the eigenmodes (1, +k) and (1, -k) of the Jacobi generator.

    (0, 1) = ((1, k) - (1, -k)) / (2k), so after N steps j = (R(kh)^N - R(-kh)^N) / (2k).
    """
    z = k * length / steps
    return (stability_function(method, z) ** steps - stability_function(method, -z) ** steps) / (2.0 * k)


def constant_curvature_transfer(method: str, curvature, length, steps) -> np.ndarray:
    """Transfer matrix after ``steps`` equal steps: the method's step matrix to the power ``steps``."""
    return np.linalg.matrix_power(step_matrix(method, curvature, length / steps), int(steps))


def _fraction_product(a, b):
    return [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]


def exact_arithmetic_transfer(method: str, curvature: float, length: float, steps: int):
    """Transfer matrix of an explicit method on j'' + K j = 0 in exact rational arithmetic.

    The float inputs are taken at their exact binary values (``Fraction``), so
    the result is the method's truncation-only answer with no rounding at all:
    for constant K the Jacobi part of every stage of an explicit Runge-Kutta
    step is linear, and the step matrix is the truncated exponential series
    sum_{j <= p} (h A)^j / j!. The difference between a floating-point run and
    this matrix is therefore the run's rounding contribution.
    """
    from fractions import Fraction

    if method not in integrators.ORDERS:
        raise ValueError(f"Exact-arithmetic transfer is defined for the explicit methods only, not {method}")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    one, zero = Fraction(1), Fraction(0)
    h, k = Fraction(length) / int(steps), Fraction(curvature)
    z = [[zero, h], [-k * h, zero]]
    step = [[one, zero], [zero, one]]
    term = [[one, zero], [zero, one]]
    for j in range(1, integrators.ORDERS[method] + 1):
        term = [[entry / j for entry in row] for row in _fraction_product(term, z)]
        step = [[step[r][c] + term[r][c] for c in range(2)] for r in range(2)]
    result, power, n = [[one, zero], [zero, one]], step, int(steps)
    while n:
        if n & 1:
            result = _fraction_product(result, power)
        power, n = _fraction_product(power, power), n >> 1
    return result


def minimal_steps(error_of, tolerance, limit=2 ** 24) -> int:
    """Smallest N with error_of(N) <= tolerance by doubling then bisection.

    Assumes the error is monotone between the last failing and the first
    passing power of two; a nonfinite or refused evaluation counts as failing.
    """
    def fails(n):
        try:
            value = error_of(n)
        except (FloatingPointError, np.linalg.LinAlgError):
            return True
        return not (math.isfinite(value) and value <= tolerance)

    n = 1
    while fails(n):
        n *= 2
        if n > limit:
            raise FloatingPointError("Step search exceeded its limit")
    if n == 1:
        return 1
    low, high = n // 2, n
    while high - low > 1:
        mid = (low + high) // 2
        if fails(mid):
            low = mid
        else:
            high = mid
    return high


def fixed_march(f, y0, h, steps, method, stop=None):
    """Fixed-step march that stops at the first nonfinite state or when ``stop(y)`` is true.

    Returns (s, states, failed_at) where ``failed_at`` is the arclength of the
    first rejected state (None when the march completed).
    """
    step = integrators.STEPS[method]
    y = np.array(y0, dtype=float)
    states = [y.copy()]
    for n in range(steps):
        y = step(f, y, h)
        if not np.all(np.isfinite(y)) or (stop is not None and stop(y)):
            return np.arange(len(states)) * h, np.array(states), (n + 1) * h
        states.append(y.copy())
    return np.arange(len(states)) * h, np.array(states), None
