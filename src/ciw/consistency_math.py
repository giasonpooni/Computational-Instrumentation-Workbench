"""Finite-sample consistency statistics for declared estimates and innovations.

The regularized incomplete gamma and beta functions below give chi-square and
binomial quantiles without a SciPy dependency; both are the standard series and
continued-fraction evaluations with explicit convergence bounds.  Every band is
a two-sided interval at a declared confidence.  Nothing here repairs, reweights
or reinterprets a declared covariance: a statistic that falls outside its band
is reported as such.
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-15
_TINY = 1e-300
_MAX_ITERATIONS = 100_000
CONSISTENT = "consistent"
TOO_SMALL = "covariance_too_small"
TOO_LARGE = "covariance_too_large"
UNDER = "under_covering"
OVER = "over_covering"
UNBIASED = "unbiased_at_declared_uncertainty"
BIASED = "biased_at_declared_uncertainty"


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def regularized_gamma_p(a, x):
    """P(a, x), the regularized lower incomplete gamma function, for a > 0 and x >= 0."""
    a = _positive(a, "shape")
    if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
        raise ValueError("argument must be a finite nonnegative number")
    if x == 0:
        return 0.0
    scale = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(_MAX_ITERATIONS):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * _EPS:
                return min(1.0, total * scale)
        raise ValueError("Incomplete gamma series did not converge")
    b = x + 1 - a
    c = 1 / _TINY
    d = 1 / b
    h = d
    for i in range(1, _MAX_ITERATIONS + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < _EPS:
            return max(0.0, 1.0 - scale * h)
    raise ValueError("Incomplete gamma continued fraction did not converge")


def _beta_continued_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = _TINY if abs(d) < _TINY else d
    d = 1 / d
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = _TINY if abs(d) < _TINY else d
        c = 1 + aa / c
        c = _TINY if abs(c) < _TINY else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < _EPS:
            return h
    raise ValueError("Incomplete beta continued fraction did not converge")


def regularized_beta(a, b, x):
    """I_x(a, b), the regularized incomplete beta function, for a, b > 0 and 0 <= x <= 1."""
    a, b = _positive(a, "first shape"), _positive(b, "second shape")
    if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("argument must lie in the unit interval")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return min(1.0, front * _beta_continued_fraction(a, b, x) / a)
    return max(0.0, 1 - front * _beta_continued_fraction(b, a, 1 - x) / b)


def _bisect(function, target, lower, upper):
    """Root of a monotone increasing function on [lower, upper] to full binary64 resolution."""
    low, high = lower, upper
    for _ in range(400):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if function(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def chi_square_cdf(x, dof):
    return regularized_gamma_p(_positive(dof, "degrees of freedom") / 2, x / 2)


def chi_square_quantile(p, dof):
    dof = _positive(dof, "degrees of freedom")
    if type(p) not in (int, float) or not 0 < p < 1:
        raise ValueError("probability must lie strictly inside the unit interval")
    upper = max(dof, 1.0)
    while chi_square_cdf(upper, dof) < p:
        upper *= 2
    return _bisect(lambda value: chi_square_cdf(value, dof), p, 0.0, upper)


def chi_square_band(dof, confidence):
    """Two-sided chi-square acceptance interval at the declared confidence."""
    alpha = 1 - confidence
    return chi_square_quantile(alpha / 2, dof), chi_square_quantile(1 - alpha / 2, dof)


def normal_quantile(confidence):
    """z such that a standard normal lies in [-z, z] with the declared probability."""
    return math.sqrt(chi_square_quantile(confidence, 1))


def _beta_quantile(p, a, b):
    return _bisect(lambda value: regularized_beta(a, b, value), p, 0.0, 1.0)


def binomial_interval(count, total, confidence):
    """Clopper-Pearson two-sided interval for a binomial proportion."""
    if type(count) is not int or type(total) is not int or not 0 <= count <= total or total < 1:
        raise ValueError("count and total must be integers with 0 <= count <= total")
    alpha = 1 - confidence
    lower = 0.0 if count == 0 else _beta_quantile(alpha / 2, count, total - count + 1)
    upper = 1.0 if count == total else _beta_quantile(1 - alpha / 2, count + 1, total - count)
    return lower, upper


def normalized_squares(errors, covariances):
    """e_k^T P_k^{-1} e_k for each retained pair; covariances must be positive definite."""
    errors = np.asarray(errors, dtype=np.float64)
    values = []
    for error, covariance in zip(errors, covariances):
        solved = np.linalg.solve(np.asarray(covariance, dtype=np.float64), error)
        values.append(float(error @ solved))
    return values


def mean_square_status(values, dimension, confidence):
    """Compare the mean normalized square with its chi-square band for K*n degrees of freedom."""
    samples = len(values)
    dof = samples * dimension
    lower, upper = chi_square_band(dof, confidence)
    mean = float(sum(values) / samples)
    band = [lower / samples, upper / samples]
    status = CONSISTENT if band[0] <= mean <= band[1] else (TOO_SMALL if mean > band[1] else TOO_LARGE)
    return {"values": values, "mean": mean, "dof": dof, "mean_band": band, "status": status}


def coverage_status(errors, covariances, confidence):
    """Per-component interval coverage against the binomial band around the nominal probability."""
    errors = np.asarray(errors, dtype=np.float64)
    z = normal_quantile(confidence)
    components = []
    total = errors.shape[0]
    for index in range(errors.shape[1]):
        sigma = np.sqrt(np.asarray([covariance[index][index] for covariance in covariances], dtype=np.float64))
        covered = int(np.count_nonzero(np.abs(errors[:, index]) <= z * sigma))
        lower, upper = binomial_interval(covered, total, confidence)
        status = CONSISTENT if lower <= confidence <= upper else (UNDER if upper < confidence else OVER)
        components.append({"covered": covered, "total": total, "fraction": covered / total,
                           "interval": [lower, upper], "status": status})
    return {"probability": confidence, "z": z, "components": components}


def bias_status(errors, covariances, confidence):
    """Mean error per component against its standard error under the declared covariances."""
    errors = np.asarray(errors, dtype=np.float64)
    z = normal_quantile(confidence)
    samples = errors.shape[0]
    components = []
    for index in range(errors.shape[1]):
        variance = sum(float(covariance[index][index]) for covariance in covariances) / samples ** 2
        mean = float(np.mean(errors[:, index]))
        standard_error = math.sqrt(variance)
        score = mean / standard_error
        components.append({"mean_error": mean, "standard_error": standard_error, "z": score,
                           "status": UNBIASED if abs(score) <= z else BIASED})
    return {"z": z, "components": components}
