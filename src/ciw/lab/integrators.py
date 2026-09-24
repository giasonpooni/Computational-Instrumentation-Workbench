"""Fixed-step and adaptive integrators for first-order systems y' = f(y).

Explicit methods: Euler, explicit midpoint, classical RK4 and adaptive
Dormand-Prince 5(4). Symmetric implicit methods: the Gauss collocation
methods with one stage (implicit midpoint, order 2) and two stages
(Gauss-Legendre, order 4), whose stage equations are solved by fixed-point
iteration to a declared tolerance; a step whose iteration does not converge
is refused (:class:`ImplicitSolveRefusal`), never returned unconverged.

The integrators never renormalize the state: any drift of speed, energy or
first integrals is part of the measured numerical result.
"""
from __future__ import annotations

import math

import numpy as np

ORDERS = {"euler": 1, "midpoint": 2, "rk4": 4}


def step_euler(f, y, h):
    return y + h * f(y)


def step_midpoint(f, y, h):
    return y + h * f(y + 0.5 * h * f(y))


def step_rk4(f, y, h):
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


STEPS = {"euler": step_euler, "midpoint": step_midpoint, "rk4": step_rk4}


def integrate_fixed(f, y0, length, steps, method="rk4"):
    """Integrate over [0, length] with ``steps`` equal steps; return (s, states)."""
    if method not in STEPS:
        raise ValueError(f"Unsupported fixed-step method: {method}")
    if steps < 1:
        raise ValueError("steps must be positive")
    step = STEPS[method]
    h = length / steps
    states = np.empty((steps + 1, len(y0)))
    states[0] = y0
    y = np.array(y0, dtype=float)
    for n in range(steps):
        y = step(f, y, h)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"{method} produced a nonfinite state at step {n + 1}")
        states[n + 1] = y
    return np.linspace(0.0, length, steps + 1), states


# Gauss collocation methods (symmetric, A-stable, order 2s): Butcher matrix A and weights b. Stage sums are
# formed elementwise (no BLAS), like the explicit steps, so the method adds no kernel-dependent rounding.
_R3 = math.sqrt(3.0) / 6.0
GAUSS_TABLEAUS = {"implicit-midpoint": (((0.5,),), (1.0,)),
                  "gauss-legendre-2": (((0.25, 0.25 - _R3), (0.25 + _R3, 0.25)), (0.5, 0.5))}
IMPLICIT_ORDERS = {"implicit-midpoint": 2, "gauss-legendre-2": 4}
# Declared stage-equation tolerance: the fixed-point iteration stops at the first iterate whose stage values
# change by at most SOLVE_TOL times the magnitude of the terms they are summed from (componentwise). That is
# about 450 units of roundoff, so rounding noise cannot keep a contracting iteration from meeting it. The test
# bounds the last change, not the remainder: for a contraction factor q the stage values then differ from the
# exact ones by at most q/(1 - q) times that change (1.5 times it at q = 0.6, about the largest q that
# SOLVE_MAX_ITERATIONS admits), and the returned step, formed from f of the previous iterate, by at most
# |h| Lip(f)/(1 - q) times it (2q/(1 - q), 3 at q = 0.6, for implicit midpoint). On y' = y both bounds are attained.
SOLVE_TOL = 1e-13
SOLVE_MAX_ITERATIONS = 60


class ImplicitSolveRefusal(FloatingPointError):
    """The stage equations of an implicit step did not converge; nothing unconverged is returned."""

    code = "implicit_solve_not_converged"


def step_gauss(f, y, h, method="gauss-legendre-2", tol=SOLVE_TOL, max_iterations=SOLVE_MAX_ITERATIONS):
    """One Gauss collocation step by fixed-point iteration; returns (next state, iterations).

    With stage derivatives K (one row per stage) the stage values are
    Y_i = y + h sum_j a_ij K_j and K_i = f(Y_i). Starting from K_i = f(y), the
    iteration K <- f(y + h A K) stops at the first iterate m with
    |Y^(m) - Y^(m-1)| <= tol (|y| + |h| sum_j |a_ij| |K_j|) in every
    component, the scale of the terms Y is summed from, so the test is
    independent of the units of each component and reachable by rounding. It
    contracts when |h| times the Lipschitz constant of f times the spectral
    radius of A (1/2 for implicit midpoint, 1/sqrt(12) for two-stage
    Gauss-Legendre) is below one; otherwise, or when an iterate is nonfinite
    or outside the domain of f, the step is refused with
    :class:`ImplicitSolveRefusal`.
    """
    if method not in GAUSS_TABLEAUS:
        raise ValueError(f"Unsupported implicit method: {method}")
    if not (tol > 0 and math.isfinite(tol)) or max_iterations < 1:
        raise ValueError("An implicit step needs a positive finite tolerance and at least one iteration")
    a, b = GAUSS_TABLEAUS[method]
    y = np.asarray(y, dtype=float)
    size = np.abs(y)

    def stage_values(stages):
        return [y + h * sum(aij * kj for aij, kj in zip(row, stages)) for row in a]

    previous = stage_values([f(y)] * len(b))
    # A diverging iteration may overflow inside f or leave its domain; it ends in the refusal below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for iteration in range(1, max_iterations + 1):
            try:
                stages = [f(value) for value in previous]
            except (ArithmeticError, ValueError):
                break
            values = stage_values(stages)
            if not all(np.all(np.isfinite(value)) for value in values):
                break
            magnitude = [abs(h) * np.abs(kj) for kj in stages]
            if all(np.all(np.abs(new - old) <= tol * (size + sum(abs(aij) * mj for aij, mj in zip(row, magnitude))))
                   for new, old, row in zip(values, previous, a)):
                return y + h * sum(bi * kj for bi, kj in zip(b, stages)), iteration
            previous = values
    raise ImplicitSolveRefusal(f"{method} stage equations did not converge within {max_iterations} fixed-point "
                               f"iterations at step size {h!r}")


def integrate_implicit(f, y0, length, steps, method="gauss-legendre-2", tol=SOLVE_TOL,
                       max_iterations=SOLVE_MAX_ITERATIONS):
    """Gauss collocation over [0, length] with ``steps`` equal steps; returns (s, states, stats).

    ``stats`` reports the fixed-point iterations (total, largest and mean per
    step) and the function evaluations. A step whose stage equations do not
    converge raises :class:`ImplicitSolveRefusal` naming the step; no
    unconverged state is returned. A negative ``length`` integrates backward.
    """
    if method not in GAUSS_TABLEAUS:
        raise ValueError(f"Unsupported implicit method: {method}")
    if steps < 1:
        raise ValueError("steps must be positive")
    h = length / steps
    stages = len(GAUSS_TABLEAUS[method][1])
    states = np.empty((steps + 1, len(y0)))
    states[0] = y0
    y = np.array(y0, dtype=float)
    iterations = []
    for n in range(steps):
        try:
            y, count = step_gauss(f, y, h, method, tol, max_iterations)
        except ImplicitSolveRefusal as refusal:
            raise ImplicitSolveRefusal(f"{refusal} (step {n + 1} of {steps})") from None
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"{method} produced a nonfinite state at step {n + 1}")
        iterations.append(count)
        states[n + 1] = y
    stats = {"method": method, "steps": steps, "solve_tol": tol, "iterations_total": int(sum(iterations)),
             "iterations_max": int(max(iterations)), "iterations_mean": float(np.mean(iterations)),
             "function_evaluations": int(steps + stages * sum(iterations))}
    return np.linspace(0.0, length, steps + 1), states, stats


def richardson_rk4(f, y0, length, steps):
    """RK4 at ``steps`` and ``2*steps``; extrapolated final state and its error estimate."""
    _, coarse = integrate_fixed(f, y0, length, steps, "rk4")
    _, fine = integrate_fixed(f, y0, length, 2 * steps, "rk4")
    difference = fine[-1] - coarse[-1]
    return fine[-1] + difference / 15.0, float(np.max(np.abs(difference)) / 15.0)


# Dormand-Prince 5(4) tableau.
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def integrate_adaptive(f, y0, length, rtol=1e-8, atol=1e-10, h0=None, max_steps=200000):
    """Dormand-Prince 5(4) with local extrapolation; returns (s, states, stats).

    The error norm is the RMS of (y5 - y4) / (atol + rtol * max|y|). Steps end
    exactly at ``length``; a negative ``length`` integrates backward, as
    :func:`integrate_fixed` does, with nodes running from 0 down to ``length``.
    """
    if not math.isfinite(length):
        raise ValueError("Adaptive integration needs a finite length")
    direction, span = math.copysign(1.0, length), abs(length)
    y = np.array(y0, dtype=float)
    s = 0.0  # distance covered, in the direction of ``length``
    h = abs(h0) if h0 else min(span, 0.01 * max(span, 1e-12))
    nodes, states = [0.0], [y.copy()]
    accepted = rejected = evaluations = 0
    k1 = f(y)
    evaluations += 1
    while s < span:
        if accepted + rejected > max_steps:
            raise FloatingPointError("Adaptive integration exceeded its step budget")
        h = min(h, span - s)
        step = direction * h
        k = [k1]
        for stage in range(1, 7):
            k.append(f(y + step * sum(a * kj for a, kj in zip(_A[stage], k))))
        evaluations += 6
        y5 = y + step * sum(b * kj for b, kj in zip(_B5, k))
        y4 = y + step * sum(b * kj for b, kj in zip(_B4, k))
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error = math.sqrt(float(np.mean(((y5 - y4) / scale) ** 2)))
        if not math.isfinite(error):
            raise FloatingPointError("Adaptive integration produced a nonfinite error estimate")
        if error <= 1.0:
            s = span if span - (s + h) < 1e-14 * max(1.0, span) else s + h
            y = y5
            k1 = k[6]  # first-same-as-last
            nodes.append(direction * s)
            states.append(y.copy())
            accepted += 1
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** -0.2))
        else:
            rejected += 1
            factor = max(0.2, 0.9 * error ** -0.2)
        h *= factor
    stats = {"accepted_steps": accepted, "rejected_steps": rejected, "function_evaluations": evaluations}
    return np.array(nodes), np.array(states), stats


def observed_order(step_sizes, errors) -> float:
    """Least-squares slope of log(error) against log(step size).

    Refuses (ValueError) unless the pairs are positive and finite with at least
    two distinct step sizes: an error of exactly zero (a bit-exact result) has
    no logarithm, and dropping it silently would change the fit, so the caller
    must decide; with fewer than two distinct step sizes the slope is
    undetermined. A step size may repeat once two distinct ones are present.
    """
    h, e = np.asarray(step_sizes, dtype=float), np.asarray(errors, dtype=float)
    if h.ndim != 1 or h.shape != e.shape or len(np.unique(h)) < 2:
        raise ValueError("An observed order needs paired step sizes and errors with at least two distinct "
                         "step sizes")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(e)) and np.all(h > 0) and np.all(e > 0)):
        raise ValueError(f"An observed order needs positive finite step sizes and errors; got {h.tolist()} "
                         f"and {e.tolist()}")
    slope, _ = np.polyfit(np.log(h), np.log(e), 1)
    return float(slope)


def hermite_zeros(s, values, derivatives):
    """Zeros of a sampled function using cubic Hermite interpolation on sign-change intervals."""
    zeros = []
    for i in range(len(s) - 1):
        a, b = values[i], values[i + 1]
        if a == 0.0:
            zeros.append(float(s[i]))
            continue
        if b == 0.0 or a * b > 0:
            continue  # an exact zero at s[i + 1] is recorded by the next interval
        h = s[i + 1] - s[i]
        da, db = derivatives[i] * h, derivatives[i + 1] * h

        def p(t):
            return ((2 * t ** 3 - 3 * t ** 2 + 1) * a + (t ** 3 - 2 * t ** 2 + t) * da
                    + (-2 * t ** 3 + 3 * t ** 2) * b + (t ** 3 - t ** 2) * db)

        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if p(lo) * p(mid) <= 0:
                hi = mid
            else:
                lo = mid
        zeros.append(float(s[i] + 0.5 * (lo + hi) * h))
    if values[-1] == 0.0:
        zeros.append(float(s[-1]))
    return sorted(set(zeros))
