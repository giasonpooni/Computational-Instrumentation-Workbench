"""Deterministic synthetic multi-sensor bench and linear-Gaussian filter algebra.

Scope: a planar constant-velocity truth driven by continuous white-noise
acceleration, four synthetic sensors (camera position, encoder speed, IMU
heading rate, tracker position) sampled at declared rates from independent
seeded PCG64 streams, and the filter algebra used by the sensor-fusion
experiments T060-T076: gain schedules, batched Kalman filters, exact second
moments of mismatched filters, the batch information-form posterior, an exact
rational scalar filter and chi-square/binomial bounds.

Non-claims: every reading is drawn from a declared distribution. Nothing here
models a real camera, encoder, IMU, tracker or clock; agreement with the
declared covariances says nothing about real sensor performance, calibration
validity or timing behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
import math
from statistics import NormalDist

import numpy as np

BENCH_SEED = 60_2026
H_POS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_NORMAL = NormalDist()


# Motion model --------------------------------------------------------------
def cv_model(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of planar white-noise acceleration (spectral density ``q``) over ``dt``."""
    eye, zero = np.eye(2), np.zeros((2, 2))
    F = np.block([[eye, dt * eye], [zero, eye]])
    Q = q * np.block([[dt ** 3 / 3 * eye, dt ** 2 / 2 * eye], [dt ** 2 / 2 * eye, dt * eye]])
    return F, Q


def factor(cov) -> np.ndarray:
    """Lower factor L with L L^T = cov; Cholesky, or a zero factor for a zero matrix."""
    cov = np.asarray(cov, dtype=float)
    if not np.any(cov):
        return np.zeros_like(cov)
    return np.linalg.cholesky(cov)


def gaussian(rng, cov, shape) -> np.ndarray:
    """Draws of N(0, cov) with leading ``shape``; Cholesky keeps them platform-stable."""
    L = factor(cov)
    return rng.standard_normal(tuple(shape) + (L.shape[0],)) @ L.T


def generator(seed) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def simulate_truth(rng, F, Q, mu0, P0, runs, ticks) -> np.ndarray:
    """Truth trajectories (runs, ticks + 1, n) of x_{k+1} = F x_k + w_k."""
    mu0 = np.asarray(mu0, dtype=float)
    x = mu0 + gaussian(rng, P0, (runs,))
    noise = gaussian(rng, Q, (runs, ticks))
    out = np.empty((runs, ticks + 1, len(mu0)))
    out[:, 0] = x
    for k in range(ticks):
        x = x @ F.T + noise[:, k]
        out[:, k + 1] = x
    return out


def measure(rng, truth, H, R, ticks) -> np.ndarray:
    """Linear readings H x + v at the listed ticks, shape (runs, len(ticks), m)."""
    ticks = np.asarray(ticks, dtype=int)
    clean = truth[:, ticks] @ np.asarray(H).T
    return clean + gaussian(rng, R, clean.shape[:2])


# Four-sensor bench (T060) -----------------------------------------------------
@dataclass(frozen=True)
class SensorSpec:
    """A declared synthetic sensor: kind, sampling period in base ticks and noise covariance."""

    name: str
    kind: str
    every: int
    covariance: tuple
    frame_id: str
    unit: str

    @property
    def R(self) -> np.ndarray:
        return np.array(self.covariance, dtype=float)

    def ticks(self, horizon: int) -> np.ndarray:
        """Reading ticks: every positive multiple of ``every`` up to the horizon."""
        return np.arange(self.every, horizon + 1, self.every)


DEFAULT_SENSORS = (
    SensorSpec("camera", "position", 2, ((0.04, 0.012), (0.012, 0.04)), "world", "m"),
    SensorSpec("encoder", "speed", 1, ((0.01,),), "body", "m/s"),
    SensorSpec("imu", "heading_rate", 1, ((0.0004,),), "body", "rad/s"),
    SensorSpec("tracker", "position", 10, ((0.0025, 0.0), (0.0, 0.0025)), "world", "m"),
)


@dataclass(frozen=True)
class BenchConfig:
    dt: float = 0.05
    ticks: int = 400
    q: float = 0.05
    x0: tuple = (0.0, 0.0, 1.0, 0.5)
    P0: tuple = ((0.25, 0, 0, 0), (0, 0.25, 0, 0), (0, 0, 0.04, 0), (0, 0, 0, 0.04))
    sensors: tuple = field(default=DEFAULT_SENSORS)

    def describe(self) -> dict:
        return {"motion": "planar constant velocity, white-noise acceleration", "dt_s": self.dt,
                "ticks": self.ticks, "q_m2_per_s3": self.q, "x0": list(self.x0),
                "P0": [list(row) for row in self.P0],
                "sensors": [{"name": s.name, "kind": s.kind, "every_ticks": s.every,
                             "rate_hz": 1.0 / (s.every * self.dt), "covariance": [list(r) for r in s.covariance],
                             "frame_id": s.frame_id, "unit": s.unit} for s in self.sensors]}


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def sensor_function(kind: str, truth: np.ndarray, ticks: np.ndarray, dt: float) -> np.ndarray:
    """Noise-free sensor output (runs, len(ticks), m) from the truth trajectories."""
    if kind == "position":
        return truth[:, ticks, :2]
    if kind == "speed":
        return np.linalg.norm(truth[:, ticks, 2:4], axis=-1)[..., None]
    if kind == "heading_rate":
        # Integrating gyro: heading increment over the preceding tick divided by dt.
        now = np.arctan2(truth[:, ticks, 3], truth[:, ticks, 2])
        before = np.arctan2(truth[:, ticks - 1, 3], truth[:, ticks - 1, 2])
        return (wrap(now - before) / dt)[..., None]
    raise ValueError(f"Unsupported sensor kind: {kind}")


def generate_bench(config: BenchConfig, seed: int, runs: int) -> dict:
    """Truth, raw readings, noise draws and declared covariances for ``runs`` seeded replicas.

    The truth and every sensor draw from their own spawned PCG64 stream, so
    changing one sensor's rate or covariance leaves the other streams intact.
    """
    children = np.random.SeedSequence(seed).spawn(1 + len(config.sensors))
    F, Q = cv_model(config.dt, config.q)
    truth_rng = generator(children[0])
    x0 = np.asarray(config.x0, dtype=float) + gaussian(truth_rng, config.P0, (runs,))
    process = gaussian(truth_rng, Q, (runs, config.ticks))
    truth = np.empty((runs, config.ticks + 1, 4))
    truth[:, 0] = x0
    for k in range(config.ticks):
        truth[:, k + 1] = truth[:, k] @ F.T + process[:, k]
    readings = {}
    for spec, child in zip(config.sensors, children[1:]):
        ticks = spec.ticks(config.ticks)
        clean = sensor_function(spec.kind, truth, ticks, config.dt)
        noise = gaussian(generator(child), spec.R, clean.shape[:2])
        readings[spec.name] = {"ticks": ticks, "clean": clean, "noise": noise, "values": clean + noise}
    return {"config": config, "seed": seed, "runs": runs, "F": F, "Q": Q, "truth": truth,
            "process": process, "readings": readings}


def bench_digest(bench: dict) -> str:
    """SHA-256 over truth and raw readings in a fixed order (retained, not a finding value)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(bench["truth"], dtype="<f8").tobytes())
    for name in sorted(bench["readings"]):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(bench["readings"][name]["values"], dtype="<f8").tobytes())
    return digest.hexdigest()


def with_sensor(config: BenchConfig, name: str, **changes) -> BenchConfig:
    return replace(config, sensors=tuple(replace(s, **changes) if s.name == name else s for s in config.sensors))


# Kalman filter algebra ------------------------------------------------------------
@dataclass
class Step:
    """One filter tick: prior and posterior covariance and, when updated, H, R, S and K."""

    prior: np.ndarray
    post: np.ndarray
    H: np.ndarray | None = None
    R: np.ndarray | None = None
    S: np.ndarray | None = None
    K: np.ndarray | None = None


def gain_schedule(F, Q, P0, plan) -> list:
    """Data-independent covariance recursion; ``plan[k-1]`` is None or (H, R) for tick k.

    The posterior uses the Joseph form so that it stays symmetric positive
    definite for any gain; with the optimal gain it equals (I - K H) P.
    """
    n = F.shape[0]
    eye = np.eye(n)
    P = np.array(P0, dtype=float)
    steps = []
    for item in plan:
        prior = F @ P @ F.T + Q
        if item is None:
            P = prior
            steps.append(Step(prior, prior))
            continue
        H, R = (np.asarray(a, dtype=float) for a in item)
        S = H @ prior @ H.T + R
        K = np.linalg.solve(S, H @ prior).T
        A = eye - K @ H
        P = A @ prior @ A.T + K @ R @ K.T
        P = 0.5 * (P + P.T)
        steps.append(Step(prior, P, H, R, S, K))
    return steps


def run_shared(F, x0_hat, steps, readings) -> tuple[np.ndarray, list]:
    """Apply one gain schedule to every run: estimates (runs, K+1, n) and innovations per tick."""
    runs = next(z.shape[0] for z in readings if z is not None)
    x = np.broadcast_to(np.asarray(x0_hat, dtype=float), (runs, F.shape[0])).copy()
    estimates = np.empty((runs, len(steps) + 1, F.shape[0]))
    estimates[:, 0] = x
    innovations = []
    for k, (step, z) in enumerate(zip(steps, readings)):
        x = x @ F.T
        if step.H is not None:
            nu = z - x @ step.H.T
            x = x + nu @ step.K.T
            innovations.append(nu)
        else:
            innovations.append(None)
        estimates[:, k + 1] = x
    return estimates, innovations


def quadratic(vectors, cov) -> np.ndarray:
    """v^T cov^{-1} v for each row (shared cov) or each batch entry (stacked cov)."""
    vectors = np.asarray(vectors, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:
        flat = vectors.reshape(-1, cov.shape[0])
        solved = np.linalg.solve(cov, flat.T).T.reshape(vectors.shape)
    else:
        solved = np.linalg.solve(cov, vectors[..., None])[..., 0]
    return np.einsum("...i,...i->...", vectors, solved)


def nees_series(estimates, truth, steps) -> np.ndarray:
    """NEES (runs, K) at ticks 1..K against each tick's posterior covariance."""
    errors = estimates[:, 1:] - truth[:, 1:]
    return np.stack([quadratic(errors[:, k], step.post) for k, step in enumerate(steps)], axis=1)


def batched_update(x, P, nu, H, R):
    """Per-run Kalman update with innovation ``nu``; H is (m, n) or (runs, m, n).

    Returns the updated mean and Joseph-form covariance, S and the NIS.
    """
    H = np.broadcast_to(H, (x.shape[0],) + np.shape(H)[-2:])
    Ht = np.swapaxes(H, 1, 2)
    S = H @ P @ Ht + R
    K = np.swapaxes(np.linalg.solve(S, H @ P), 1, 2)
    nis = np.einsum("ri,ri->r", nu, np.linalg.solve(S, nu[..., None])[..., 0])
    A = np.eye(x.shape[1]) - K @ H
    P_new = A @ P @ np.swapaxes(A, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)
    P_new = 0.5 * (P_new + np.swapaxes(P_new, 1, 2))
    return x + (K @ nu[..., None])[..., 0], P_new, S, nis


def run_gated(F, Q, x0_hat, P0, plan, readings, threshold=None) -> dict:
    """Per-run filter whose covariance depends on gating decisions (NIS > threshold is rejected)."""
    runs = next(z.shape[0] for z in readings if z is not None)
    n = F.shape[0]
    x = np.broadcast_to(np.asarray(x0_hat, dtype=float), (runs, n)).copy()
    P = np.broadcast_to(np.asarray(P0, dtype=float), (runs, n, n)).copy()
    estimates = np.empty((runs, len(plan) + 1, n))
    covariances = np.empty((runs, len(plan) + 1, n, n))
    estimates[:, 0], covariances[:, 0] = x, P
    nis_all, accepted_all = [], []
    for k, (item, z) in enumerate(zip(plan, readings)):
        x = x @ F.T
        P = F @ P @ F.T + Q
        if item is not None:
            H, R = item
            nu = z - x @ H.T
            x_upd, P_upd, _, nis = batched_update(x, P, nu, H, R)
            accept = np.ones(runs, dtype=bool) if threshold is None else nis <= threshold
            x = np.where(accept[:, None], x_upd, x)
            P = np.where(accept[:, None, None], P_upd, P)
            nis_all.append(nis)
            accepted_all.append(accept)
        else:
            nis_all.append(None)
            accepted_all.append(None)
        estimates[:, k + 1], covariances[:, k + 1] = x, P
    return {"estimates": estimates, "covariances": covariances, "nis": nis_all, "accepted": accepted_all}


def mismatch_moments(F, Q_true, mu0, P0_true, x0_hat, steps, truth_plan) -> dict:
    """Exact first and second moments of a linear filter run against a different linear truth.

    The joint vector y = [x; x_hat] is linear-Gaussian: truth x_{k+1} = F x_k + w
    with w ~ N(0, Q_true); a reading used by the filter at tick k is
    z = H_true x + c + v with v ~ N(0, R_true) (``truth_plan[k-1]``), while the
    filter applies its own gain K_k from ``steps``. Returns, per tick, the
    expected NEES tr(P_f^{-1} E[e e^T]), the expected NIS tr(S_f^{-1} E[nu nu^T])
    and the expected squared position error. With truth equal to the filter
    model they reduce to n, m and tr(P_pos).
    """
    n = F.shape[0]
    eye, zero = np.eye(n), np.zeros((n, n))
    A = np.block([[F, zero], [zero, F]])
    mean = np.concatenate([np.asarray(mu0, dtype=float), np.asarray(x0_hat, dtype=float)])
    cov = np.block([[np.asarray(P0_true, dtype=float), zero], [zero, zero]])
    selector = np.hstack([-eye, eye])
    nees, nis, mse, bias = [], [], [], []
    for step, truth_item in zip(steps, truth_plan):
        mean = A @ mean
        cov = A @ cov @ A.T
        cov[:n, :n] += Q_true
        if step.H is not None:
            H_true, c, R_true = (np.asarray(a, dtype=float) for a in truth_item)
            T = np.hstack([H_true, -step.H])
            nu_mean = T @ mean + c
            nu_second = T @ cov @ T.T + R_true + np.outer(nu_mean, nu_mean)
            nis.append(float(np.trace(np.linalg.solve(step.S, nu_second))))
            B = np.block([[eye, zero], [step.K @ H_true, eye - step.K @ step.H]])
            gain = np.vstack([np.zeros((n, step.K.shape[1])), step.K])
            mean = B @ mean + gain @ c
            cov = B @ cov @ B.T + gain @ R_true @ gain.T
        else:
            nis.append(None)
        e_mean = selector @ mean
        e_second = selector @ cov @ selector.T + np.outer(e_mean, e_mean)
        nees.append(float(np.trace(np.linalg.solve(step.post, e_second))))
        mse.append(float(e_second[0, 0] + e_second[1, 1]))
        bias.append(e_mean)
    return {"nees": nees, "nis": nis, "position_mse": mse, "error_mean": bias}


# Batch posterior and exact arithmetic (T074) -----------------------------------------
def batch_posterior(F, Q, mu0, P0, plan, readings) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gaussian posterior over x_0..x_K from the information (normal-equation) form.

    Minimizes |x_0 - mu0|^2_{P0} + sum |x_k - F x_{k-1}|^2_Q + sum |z_k - H x_k|^2_R;
    no recursion is used, so it is an independent route to the filter's answer.
    """
    n = F.shape[0]
    size = n * (len(plan) + 1)
    information = np.zeros((size, size))
    vector = np.zeros(size)
    P0_inv, Q_inv = np.linalg.inv(P0), np.linalg.inv(Q)
    information[:n, :n] += P0_inv
    vector[:n] += P0_inv @ np.asarray(mu0, dtype=float)
    for k, (item, z) in enumerate(zip(plan, readings), start=1):
        i, j = slice(n * (k - 1), n * k), slice(n * k, n * (k + 1))
        information[j, j] += Q_inv
        information[i, i] += F.T @ Q_inv @ F
        information[i, j] -= F.T @ Q_inv
        information[j, i] -= Q_inv @ F
        if item is not None:
            H, R = item
            R_inv = np.linalg.inv(R)
            information[j, j] += H.T @ R_inv @ H
            vector[j] += H.T @ R_inv @ z
    covariance = np.linalg.inv(information)
    mean = np.linalg.solve(information, vector)
    return mean.reshape(len(plan) + 1, n), covariance


def exact_scalar_filter(q: Fraction, r: Fraction, p0: Fraction, m0: Fraction, readings) -> dict:
    """Random-walk Kalman filter and batch posterior in exact rational arithmetic.

    The batch side solves the tridiagonal normal equations by elimination; the
    final-time mean and variance must equal the recursion's exactly.
    """
    x, P = m0, p0
    for z in readings:
        prior = P + q
        gain = prior / (prior + r)
        x, P = x + gain * (z - x), (1 - gain) * prior
    size = len(readings) + 1
    diagonal = [1 / p0 + 1 / q] + [2 / q + 1 / r] * (size - 2) + [1 / q + 1 / r]
    if size == 1:
        diagonal = [1 / p0]
    off = -1 / q
    rhs_mean = [m0 / p0] + [z / r for z in readings]
    rhs_unit = [Fraction(0)] * (size - 1) + [Fraction(1)]

    def solve(rhs):
        c, d = [Fraction(0)] * size, [Fraction(0)] * size
        c[0], d[0] = off / diagonal[0], rhs[0] / diagonal[0]
        for i in range(1, size):
            denominator = diagonal[i] - off * c[i - 1]
            c[i] = off / denominator
            d[i] = (rhs[i] - off * d[i - 1]) / denominator
        solution = [Fraction(0)] * size
        solution[-1] = d[-1]
        for i in range(size - 2, -1, -1):
            solution[i] = d[i] - c[i] * solution[i + 1]
        return solution

    return {"filter_mean": x, "filter_variance": P, "batch_mean": solve(rhs_mean)[-1],
            "batch_variance": solve(rhs_unit)[-1]}


# Statistical bounds -------------------------------------------------------------------
def normal_quantile(p: float) -> float:
    return _NORMAL.inv_cdf(p)


def chi2_cdf(x: float, dof: int) -> float:
    """Chi-square CDF from its closed-form series (exact up to roundoff for every dof).

    Even dof: 1 - exp(-x/2) sum_{i<k} (x/2)^i / i!. Odd dof 2k+1:
    2 Phi(sqrt x) - 1 - 2 phi(sqrt x) sum_{i=1..k} x^{i-1/2} / (2i-1)!!.
    Terms are summed in log space so large dof neither overflows nor underflows.
    """
    if x <= 0:
        return 0.0
    if dof % 2 == 0:
        k = dof // 2
        i = np.arange(k)
        log_factorial = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, k)))])
        logs = i * math.log(x / 2) - log_factorial - x / 2
    else:
        k = (dof - 1) // 2
        head = 2 * _NORMAL.cdf(math.sqrt(x)) - 1
        if k == 0:
            return head
        i = np.arange(1, k + 1)
        log_double_factorial = np.cumsum(np.log(2 * i - 1.0))
        logs = math.log(2.0) - x / 2 - 0.5 * math.log(2 * math.pi) + (i - 0.5) * math.log(x) - log_double_factorial
    peak = float(np.max(logs))
    tail = math.exp(peak) * float(np.sum(np.exp(logs - peak)))
    return 1.0 - tail if dof % 2 == 0 else head - tail


def chi2_quantile(p: float, dof: int) -> float:
    """Chi-square quantile by bisection on :func:`chi2_cdf` (closed form for dof 1 and 2)."""
    if not 0 < p < 1:
        raise ValueError("Quantile probability must lie strictly between 0 and 1")
    if dof == 1:
        return normal_quantile(0.5 + p / 2) ** 2
    if dof == 2:
        return -2.0 * math.log1p(-p)
    lo, hi = 0.0, dof + 10 * math.sqrt(2 * dof) + 10
    while chi2_cdf(hi, dof) < p:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return 0.5 * (lo + hi)


def mean_interval(dof: int, samples: int, confidence: float = 0.99) -> tuple[float, float]:
    """Two-sided interval for the mean of ``samples`` independent chi-square(dof) values."""
    tail = (1 - confidence) / 2
    total = dof * samples
    return chi2_quantile(tail, total) / samples, chi2_quantile(1 - tail, total) / samples


def wilson_interval(successes: int, trials: int, confidence: float = 0.999) -> tuple[float, float]:
    z = normal_quantile(0.5 + confidence / 2)
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def noncentral_chi2_2_cdf(x: float, noncentrality: float, terms: int = 400) -> float:
    """CDF of a noncentral chi-square with two degrees of freedom as a Poisson mixture."""
    total, weight = 0.0, math.exp(-noncentrality / 2)
    for j in range(terms):
        if j:
            weight *= (noncentrality / 2) / j
        total += weight * chi2_cdf(x, 2 + 2 * j)
        if j > noncentrality and weight < 1e-18:
            break
    return total


def consistency(values: np.ndarray, dof: int, confidence: float = 0.99) -> dict:
    """Per-tick averaged NEES/NIS over independent runs against the chi-square interval.

    ``values`` is (runs, ticks). The run-average at a tick is chi-square(dof * runs)
    / runs under a consistent filter. Returns the interval, the fraction of ticks
    inside it and the fractions above and below.
    """
    runs = values.shape[0]
    lo, hi = mean_interval(dof, runs, confidence)
    per_tick = values.mean(axis=0)
    return {"runs": runs, "ticks": int(values.shape[1]), "interval": [lo, hi],
            "fraction_inside": float(np.mean((per_tick >= lo) & (per_tick <= hi))),
            "fraction_above": float(np.mean(per_tick > hi)), "fraction_below": float(np.mean(per_tick < lo)),
            "grand_mean": float(values.mean())}
