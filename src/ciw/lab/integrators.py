"""Fixed-step and adaptive explicit integrators for first-order systems y' = f(y).

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
    exactly at ``length``.
    """
    y = np.array(y0, dtype=float)
    s = 0.0
    h = h0 or min(length, 0.01 * max(length, 1e-12))
    nodes, states = [0.0], [y.copy()]
    accepted = rejected = evaluations = 0
    k1 = f(y)
    evaluations += 1
    while s < length:
        if accepted + rejected > max_steps:
            raise FloatingPointError("Adaptive integration exceeded its step budget")
        h = min(h, length - s)
        k = [k1]
        for stage in range(1, 7):
            k.append(f(y + h * sum(a * kj for a, kj in zip(_A[stage], k))))
        evaluations += 6
        y5 = y + h * sum(b * kj for b, kj in zip(_B5, k))
        y4 = y + h * sum(b * kj for b, kj in zip(_B4, k))
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        error = math.sqrt(float(np.mean(((y5 - y4) / scale) ** 2)))
        if not math.isfinite(error):
            raise FloatingPointError("Adaptive integration produced a nonfinite error estimate")
        if error <= 1.0:
            s = length if length - (s + h) < 1e-14 * max(1.0, length) else s + h
            y = y5
            k1 = k[6]  # first-same-as-last
            nodes.append(s)
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
    """Least-squares slope of log(error) against log(step size)."""
    x = np.log(np.asarray(step_sizes, dtype=float))
    y = np.log(np.asarray(errors, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
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
