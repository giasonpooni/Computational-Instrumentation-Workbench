"""Shared evidence helpers for the sensor-fusion experiments (T060-T076).

Scope: check objects, generator bases, JSON conversion and the completed or
partial outcome rule used by every sensor-fusion task module, plus the shared
regression tolerances. Labels are never assigned here; every finding is built
by :func:`ciw.lab.evidence.finding`.

Non-claims: nothing in this module measures, calibrates or authorizes
anything; it only packages synthetic computations for the evidence validator.
"""
from __future__ import annotations

import math

import numpy as np

from .. import __version__
from .evidence import COMPUTATIONAL_DOMAINS, finding, holds as compare
from .sensor_fusion_bench import normal_quantile, wilson_interval
from .sensor_fusion_objects import FusionRefusal

TESTS = "tests/test_lab_sensor_fusion.py"
BENCH = "src/ciw/lab/sensor_fusion_bench.py"
OBJECTS = "src/ciw/lab/sensor_fusion_objects.py"
COMMON = "src/ciw/lab/sensor_fusion_common.py"
PRODUCER = {"implementation": "ciw.lab.sensor_fusion", "revision": __version__}
FAMILY_ALPHA = 1e-3

# Shared linear-Gaussian bench constants: prior mean and covariance of the
# planar constant-velocity state (x, y, vx, vy) and the declared camera noise.
MU0 = np.array([0.0, 0.0, 1.0, 0.5])
P0_BENCH = np.diag([0.25, 0.25, 0.04, 0.04])
R_CAMERA = np.array([[0.04, 0.012], [0.012, 0.04]])

# Regression tolerances. Seeded Monte Carlo statistics reproduce to roundoff
# (BLAS summation order moves the last bits only); exact counts and refusal
# codes must match exactly; rates derived from counts get one-sample slack.
TOL_MC = {"abs": 1e-9, "rel": 1e-6}
TOL_EXACT = {"abs": 0.0, "rel": 0.0}
TOL_RATE = {"abs": 2e-3, "rel": 0.0}
TOL_ROUNDOFF = {"abs": 1e-9, "rel": 1e-6}
TOL_TINY = {"abs": 1e-9, "rel": 0.0}


def files(*modules) -> tuple:
    return tuple(f"src/ciw/lab/{name}.py" for name in modules) + (COMMON, BENCH)


def check(kind, reference, observed, tolerance, comparison="abs_le") -> dict:
    """A reference check whose ``passed`` flag is computed exactly as the validator does."""
    observed, tolerance = float(observed), float(tolerance)
    holds = compare(observed, tolerance, comparison)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def refusal(reference, expected, observed) -> dict:
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def generator_basis(seed, **extra) -> dict:
    return {"generator": {"name": "ciw.lab.sensor_fusion_bench", "bit_generator": "PCG64", "seed": seed, **extra}}


def refusal_code(call) -> str:
    """The FusionRefusal code raised by ``call``, or ``no_refusal``."""
    try:
        call()
    except FusionRefusal as exc:
        return exc.code
    return "no_refusal"


def as_json(value):
    """Plain JSON values (numpy scalars and arrays converted, floats kept finite)."""
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return as_json(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Finding values must be finite")
        return value
    return value


def unreal(claim, domain, seed, statement) -> dict:
    """A real-world conclusion the synthetic bench invites but cannot support.

    Physical and authority domains are ``not_established`` whatever the basis;
    the generator is recorded so the report shows where the numbers came from.
    """
    return finding(claim, domain, statement, {**generator_basis(seed), "notes": "synthetic draws only; no acquisition"})


def outcome(fields: dict, findings: list) -> dict:
    """Completed when every computational check passed; otherwise partial with the failures named."""
    failed = [f["claim"] for f in findings if f["evidence_status"] == "not_established"
              and f["domain"] in COMPUTATIONAL_DOMAINS and not f.get("expected_not_established")]
    fields = dict(fields)
    if failed:
        fields["unresolved_assumptions"] = list(fields["unresolved_assumptions"]) + [
            f"Check failed in this run: {claim}" for claim in failed]
    return {"state": "partial" if failed else "completed", "fields": fields, "findings": findings}


def bonferroni(family: int, alpha: float = FAMILY_ALPHA) -> float:
    """Two-sided normal critical value for ``family`` simultaneous z-tests at family error ``alpha``."""
    return normal_quantile(1 - alpha / (2 * max(1, family)))


def covariance_z(samples, reference, *, known_mean=True):
    """Standardized deviations of a sample covariance from ``reference`` (unique entries).

    For Gaussian samples with known zero mean, Var(S_ij) = (C_ij^2 + C_ii C_jj) / N.
    Returns (sample covariance, z-values over the upper triangle).
    """
    samples = np.asarray(samples, dtype=float)
    reference = np.asarray(reference, dtype=float)
    N = samples.shape[0]
    if known_mean:
        S = samples.T @ samples / N
    else:
        centred = samples - samples.mean(axis=0)
        S = centred.T @ centred / (N - 1)
    diag = np.diag(reference)
    z = (S - reference) / np.sqrt((reference ** 2 + np.outer(diag, diag)) / N)
    return S, z[np.triu_indices(reference.shape[0])]


def rate_interval(count: int, trials: int, confidence: float = 0.999) -> dict:
    lo, hi = wilson_interval(count, trials, confidence)
    return {"count": int(count), "trials": int(trials), "rate": count / trials, "wilson": [lo, hi],
            "confidence": confidence}


def run_mean_z(values):
    """z of a grand mean against zero using run-level means (robust to within-run correlation).

    ``values`` is (runs, samples, m); returns per-component z and the grand mean.
    """
    per_run = values.mean(axis=1)
    mean = per_run.mean(axis=0)
    se = per_run.std(axis=0, ddof=1) / math.sqrt(per_run.shape[0])
    return mean / se, mean, se
