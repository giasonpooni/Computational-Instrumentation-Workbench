"""Synthetic signal models for encoder, IMU, timing and filtering experiments.

Scope: the generators and estimators used by T052-T057: a backlash (play)
operator with scale and bias, gyro bias plus angle random walk (single axis
and strapdown SO(3)), sample-and-hold tracking through a lossy link, sampling
with clock offset and jitter, a constant-velocity Kalman filter with a
Rauch-Tung-Striebel smoother, and the confidence-interval helpers used to
compare seeded Monte Carlo ensembles with predictions.

Non-claims: every signal is generated from declared parameters. Nothing here
characterizes a real encoder, gyroscope, clock or tracked object.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


# Statistics -------------------------------------------------------------------

def normal_quantile(p: float) -> float:
    return NormalDist().inv_cdf(p)


def chi2_quantile(p: float, dof: float) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile."""
    z = normal_quantile(p)
    c = 2.0 / (9.0 * dof)
    return dof * (1 - c + z * math.sqrt(c)) ** 3


def chi2_cdf(x: float, dof: float, terms: int = 20000) -> float:
    """Chi-square CDF by the series of the regularized lower incomplete gamma P(dof/2, x/2)."""
    a, y = dof / 2.0, x / 2.0
    if y <= 0:
        return 0.0
    term = total = 1.0
    for n in range(1, terms):
        term *= y / (a + n)
        total += term
        if term < 1e-17 * total:
            break
    return math.exp(a * math.log(y) - y - math.lgamma(a + 1) + math.log(total))


def variance_z(samples, predicted: float) -> dict:
    """Sample variance against a prediction, with a fourth-moment standard error."""
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    centered = x - x.mean()
    variance = float(centered @ centered / (n - 1))
    m4 = float(np.mean(centered ** 4))
    se = math.sqrt(max(m4 - variance ** 2 * (n - 3) / (n - 1), 0.0) / n)
    return {"n": n, "sample_variance": variance, "predicted_variance": float(predicted), "standard_error": se,
            "z": (variance - predicted) / se}


def mean_z(samples, predicted: float) -> dict:
    x = np.asarray(samples, dtype=float).ravel()
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"n": len(x), "sample_mean": float(x.mean()), "predicted_mean": float(predicted), "standard_error": se,
            "z": (float(x.mean()) - predicted) / se}


def loglog_slope(x, y) -> float:
    slope, _ = np.polyfit(np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float)), 1)
    return float(slope)


# Encoder ----------------------------------------------------------------------

def backlash(x, width: float) -> np.ndarray:
    """Play operator engaged on the positive flank at the start: output - input lies in [0, width]."""
    out = np.empty(len(x))
    held = float(x[0])
    for k, value in enumerate(x):
        if value > held:
            held = float(value)
        elif value < held - width:
            held = float(value) + width
        out[k] = held
    return out


def engagement(x, width_bound: float) -> np.ndarray:
    """+1 / -1 where the reference has travelled at least ``width_bound`` since its last reversal, else 0."""
    labels = np.zeros(len(x), dtype=int)
    direction, extreme = 0, float(x[0])
    for k in range(1, len(x)):
        step = x[k] - x[k - 1]
        if step > 0 and direction != 1:
            direction, extreme = 1, float(x[k - 1])
        elif step < 0 and direction != -1:
            direction, extreme = -1, float(x[k - 1])
        if direction and abs(x[k] - extreme) >= width_bound:
            labels[k] = direction
    return labels


def least_squares(design, target) -> dict:
    coefficients, residual, _, _ = np.linalg.lstsq(design, target, rcond=None)
    dof = len(target) - design.shape[1]
    sigma2 = float(np.sum((target - design @ coefficients) ** 2) / dof)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    return {"coefficients": coefficients, "covariance": covariance, "residual_sigma": math.sqrt(sigma2)}


# IMU --------------------------------------------------------------------------

def gyro_heading_error(runs: int, steps: int, dt: float, bias: float, arw: float, rng) -> np.ndarray:
    """Integrated single-axis heading error: bias t plus a random walk of variance arw^2 t."""
    rate_noise = arw / math.sqrt(dt)
    increments = (bias + rate_noise * rng.standard_normal((runs, steps))) * dt
    return np.cumsum(increments, axis=1)


def rotation_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of a @ b^T for stacks of rotation matrices."""
    relative = np.einsum("nij,nkj->nik", a, b)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.arccos(cosine)


def rotation_exp(vectors: np.ndarray) -> np.ndarray:
    """Rodrigues formula for a stack of rotation vectors."""
    vectors = np.atleast_2d(vectors)
    angle = np.linalg.norm(vectors, axis=1)
    safe = np.where(angle > 0, angle, 1.0)
    axis = vectors / safe[:, None]
    k = np.zeros((len(vectors), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    sin, cos = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + sin * k + (1 - cos) * (k @ k)


def rotation_log(matrices: np.ndarray) -> np.ndarray:
    """Rotation vectors of a stack of rotation matrices (angles below pi)."""
    cosine = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cosine)
    vee = np.stack([matrices[:, 2, 1] - matrices[:, 1, 2], matrices[:, 0, 2] - matrices[:, 2, 0],
                    matrices[:, 1, 0] - matrices[:, 0, 1]], axis=1)
    safe = np.where(angle > 1e-8, angle, 1.0)
    factor = np.where(angle > 1e-8, angle / (2 * np.sin(safe)), 0.5)
    return vee * factor[:, None]


def right_jacobian(phi) -> np.ndarray:
    """Right Jacobian of SO(3): exp(phi + d) = exp(phi) exp(J_r(phi) d) to first order in d."""
    phi = np.asarray(phi, dtype=float)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        return np.eye(3)
    k = np.array([[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]])
    return np.eye(3) - (1 - math.cos(angle)) / angle ** 2 * k + (angle - math.sin(angle)) / angle ** 3 * (k @ k)


def orientation_errors(omega, bias, arw: float, runs: int, steps: int, dt: float, rng) -> np.ndarray:
    """Body-frame error log(R_true^T R_est) per step for a strapdown gyro integration.

    The true body turns at constant rate ``omega``; the gyro reads omega +
    bias + white noise of density ``arw``. Returns shape (steps, runs, 3).
    """
    omega, bias = np.asarray(omega, dtype=float), np.asarray(bias, dtype=float)
    true_step = rotation_exp(omega[None, :] * dt)[0]
    truth, estimate = np.eye(3), np.repeat(np.eye(3)[None], runs, axis=0)
    out = np.empty((steps, runs, 3))
    for k in range(steps):
        noise = (arw / math.sqrt(dt)) * rng.standard_normal((runs, 3)) if arw > 0 else 0.0
        estimate = estimate @ rotation_exp((omega + bias + noise) * dt)
        truth = truth @ true_step
        out[k] = rotation_log(np.einsum("ji,njk->nik", truth, estimate))
    return out


def bias_error_prediction(omega, bias, steps: int, dt: float) -> np.ndarray:
    """Linearized mean error e_{k+1} = exp(-omega dt) e_k + J_r(omega dt) bias dt."""
    omega, bias = np.asarray(omega, dtype=float), np.asarray(bias, dtype=float)
    back = rotation_exp(-omega[None, :] * dt)[0]
    injected = right_jacobian(omega * dt) @ bias * dt
    mean, out = np.zeros(3), np.empty((steps, 3))
    for k in range(steps):
        mean = back @ mean + injected
        out[k] = mean
    return out


# Dropped samples and hold estimates -------------------------------------------

def hold_estimates(dropped: np.ndarray, q: float, r: float, rng, burn: int) -> dict:
    """Sample-and-hold tracking of a random walk through a lossy link.

    Truth x_k is a random walk with increment variance ``q``; each received
    sample is x_k plus noise of variance ``r``. Returns per-run means (after
    ``burn`` steps) of the squared hold error and of the age of the held sample.
    """
    runs, n = dropped.shape
    truth = np.cumsum(math.sqrt(q) * rng.standard_normal((runs, n)), axis=1)
    measured = truth + math.sqrt(r) * rng.standard_normal((runs, n))
    held, held_at = np.full(runs, np.nan), np.full(runs, -1)
    squared, ages, counts = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for k in range(n):
        received = ~dropped[:, k]
        held = np.where(received, measured[:, k], held)
        held_at = np.where(received, k, held_at)
        if k >= burn:
            valid = held_at >= 0
            squared += np.where(valid, (held - truth[:, k]) ** 2, 0.0)
            ages += np.where(valid, k - held_at, 0)
            counts += valid
    return {"mse": squared / counts, "age": ages / counts, "evaluated": counts}


# Timing -------------------------------------------------------------------------

def interpolation_weights(knots, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interval index, fractional position w in [0, 1) and interval length for linear interpolation."""
    index = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, len(knots) - 2)
    width = knots[index + 1] - knots[index]
    return index, (times - knots[index]) / width, width


# Kalman filter and RTS smoother ------------------------------------------------

def constant_velocity(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition and process covariance for white-noise acceleration of spectral density q."""
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    return transition, process


def simulate_track(runs, steps, transition, process, x0, p0, r, rng):
    """Truth x_1..x_N from x_0 ~ N(x0, P0) and position measurements z_k = x_k[0] + v_k."""
    chol_q, chol_p = np.linalg.cholesky(process), np.linalg.cholesky(p0)
    state = x0[None, :] + rng.standard_normal((runs, 2)) @ chol_p.T
    truth = np.empty((runs, steps, 2))
    for k in range(steps):
        state = state @ transition.T + rng.standard_normal((runs, 2)) @ chol_q.T
        truth[:, k] = state
    measurements = truth[:, :, 0] + math.sqrt(r) * rng.standard_normal((runs, steps))
    return truth, measurements


def kalman_rts(measurements, transition, process, r, x0, p0) -> dict:
    """Filter and smooth an ensemble; covariances are data independent and shared by all runs."""
    runs, steps = measurements.shape
    h = np.array([1.0, 0.0])
    p_pred, p_filt = np.empty((steps, 2, 2)), np.empty((steps, 2, 2))
    x_pred, x_filt = np.empty((runs, steps, 2)), np.empty((runs, steps, 2))
    mean = np.repeat(x0[None, :], runs, axis=0)
    cov = p0.copy()
    for k in range(steps):
        mean = mean @ transition.T
        cov = transition @ cov @ transition.T + process
        x_pred[:, k], p_pred[k] = mean, cov
        gain = cov @ h / (h @ cov @ h + r)
        mean = mean + (measurements[:, k] - mean[:, 0])[:, None] * gain[None, :]
        # Joseph form keeps the filtered covariance symmetric positive semidefinite.
        a = np.eye(2) - np.outer(gain, h)
        cov = a @ cov @ a.T + r * np.outer(gain, gain)
        x_filt[:, k], p_filt[k] = mean, cov
    x_smooth, p_smooth = x_filt.copy(), p_filt.copy()
    for k in range(steps - 2, -1, -1):
        c = p_filt[k] @ transition.T @ np.linalg.inv(p_pred[k + 1])
        x_smooth[:, k] = x_filt[:, k] + (x_smooth[:, k + 1] - x_pred[:, k + 1]) @ c.T
        p_smooth[k] = p_filt[k] + c @ (p_smooth[k + 1] - p_pred[k + 1]) @ c.T
    return {"x_pred": x_pred, "p_pred": p_pred, "x_filt": x_filt, "p_filt": p_filt,
            "x_smooth": x_smooth, "p_smooth": p_smooth}


def nees(errors, covariances) -> np.ndarray:
    """Normalized estimation error squared per run and step."""
    inverse = np.linalg.inv(covariances)
    return np.einsum("rki,kij,rkj->rk", errors, inverse, errors)


def riccati_steady_state(transition, process, r, iterations=20000, tolerance=1e-15) -> np.ndarray:
    """Predicted-covariance fixed point of the filter Riccati recursion by plain iteration."""
    h = np.array([1.0, 0.0])
    cov = process.copy()
    for _ in range(iterations):
        gain = cov @ h / (h @ cov @ h + r)
        updated = transition @ (cov - np.outer(gain, h @ cov)) @ transition.T + process
        if np.max(np.abs(updated - cov)) <= tolerance * np.max(np.abs(cov)):
            return updated
        cov = updated
    return cov
