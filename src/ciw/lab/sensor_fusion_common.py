"""Shared evidence helpers for the sensor-fusion experiments (T060-T076).

Scope: check objects, generator bases, per-finding uncertainty records,
runtime identities, JSON conversion, a thread-independent Gram product, a
markers-only categorical figure and the completed or partial outcome rule used
by every sensor-fusion task module, plus the shared regression tolerances.
Labels are never assigned here; every finding is built by
:func:`ciw.lab.evidence.finding`.

Non-claims: nothing in this module measures, calibrates or authorizes
anything; it only packages synthetic computations for the evidence validator.
"""
from __future__ import annotations

from html import escape
import math

import numpy as np

from .. import __version__
from . import svg
from .evidence import COMPUTATIONAL_DOMAINS, finding, holds as compare
from .runner import builtin_identity
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
# (platform libm and BLAS kernels move the last bits only); exact counts and
# refusal codes must match exactly.
TOL_MC = {"abs": 1e-9, "rel": 1e-6}
TOL_EXACT = {"abs": 0.0, "rel": 0.0}
TOL_ROUNDOFF = {"abs": 1e-9, "rel": 1e-6}
TOL_TINY = {"abs": 1e-9, "rel": 0.0}

# Core modules some tasks compute with; recorded in the runtime identity, not as changed files.
CORE_GEOMETRY = ("src/ciw/lab/jacobi.py", "src/ciw/lab/integrators.py", "src/ciw/lab/surfaces.py")
DECLARED_WORKLOAD = "src/ciw/declared_workload.py"


def files(*modules) -> tuple:
    """Changed files of a task: its own modules plus the shared bench, objects and helpers."""
    names = tuple(f"src/ciw/lab/{name}.py" for name in modules) + (COMMON, BENCH, OBJECTS)
    return tuple(dict.fromkeys(names))


def identity(changed, *dependencies) -> dict:
    """Runtime identity whose source digests also cover the core modules the numbers depend on."""
    return builtin_identity(tuple(dict.fromkeys(tuple(changed) + tuple(dependencies))))


# Per-finding uncertainty records (AUTHORING rule 5) -------------------------------------
def uncertainty(kind: str, value, basis: str) -> dict:
    return {"kind": kind, "value": float(value), "basis": basis}


def mc95(standard_error, basis: str) -> dict:
    """Monte Carlo 95% half-width from a standard error."""
    return uncertainty("monte_carlo_95ci", 1.96 * float(standard_error), basis)


def max_abs_z_spread(family: int, what: str) -> dict:
    """Null sampling spread of a reported max |z| over ``family`` standardized moments.

    Half-width of the central 95% interval of max |Z_i| for ``family``
    independent standard normals, whose CDF is (2 Phi(x) - 1)^family: how far
    the reported maximum moves between seeds when the model holds. It depends
    on the family size, not on a fixed 1.96.
    """
    def quantile(u):
        return normal_quantile(0.5 + 0.5 * u ** (1.0 / family))
    return uncertainty("monte_carlo_95ci", 0.5 * (quantile(0.975) - quantile(0.025)),
                       f"null sampling spread of the reported max |z| over {family} {what}: half-width of the central "
                       "95% interval of the maximum of that many independent |N(0, 1)| (dependence among the "
                       "moments changes it)")


def roundoff(value, basis: str = "observed floating-point residual of an exact identity") -> dict:
    return uncertainty("roundoff", abs(float(value)), basis)


def exact(basis: str = "deterministic: exact counts, refusal codes or rational arithmetic") -> dict:
    return uncertainty("reference_error", 0.0, basis)


def gram(a, b=None) -> np.ndarray:
    """a^T b summed over rows with einsum's own loops (no BLAS), so thread count cannot change the bits."""
    a = np.asarray(a, dtype=float)
    b = a if b is None else np.asarray(b, dtype=float)
    return np.einsum("ni,nj->ij", a, b, optimize=False)


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
    """The FusionRefusal code raised by ``call``, or ``none`` (the evidence convention) when nothing was refused."""
    try:
        call()
    except FusionRefusal as exc:
        return exc.code
    return "none"


def is_not(observed, expected) -> float:
    """1.0 when a recorded outcome differs from the expected one, for invariant checks on equality."""
    return float(observed != expected)


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


def unreal(claim, domain, seed, statement, *, source=None) -> dict:
    """A real-world conclusion the synthetic bench invites but cannot support.

    Physical and authority domains are ``not_established`` whatever the basis.
    With a ``seed`` the generator is recorded so the report shows where the
    numbers came from; a task that ran no generator passes ``seed=None`` and
    names what it did examine in ``source``.
    """
    if seed is None:
        if not source:
            raise ValueError("A finding without a generator must name its source")
        return finding(claim, domain, statement, {"derivation": source, "notes": "no acquisition"})
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
        S = gram(samples) / N
    else:
        centred = samples - samples.mean(axis=0)
        S = gram(centred) / (N - 1)
    diag = np.diag(reference)
    z = (S - reference) / np.sqrt((reference ** 2 + np.outer(diag, diag)) / N)
    return S, z[np.triu_indices(reference.shape[0])]


def rate_interval(count: int, trials: int, confidence: float = 0.999) -> dict:
    lo, hi = wilson_interval(count, trials, confidence)
    return {"count": int(count), "trials": int(trials), "rate": count / trials, "wilson": [lo, hi],
            "confidence": confidence}


def dot_chart(categories, series, *, title, ylabel) -> str:
    """Grouped markers over categorical x positions with no connecting lines.

    ``series`` = [(name, {category: value}), ...]; a missing category is left
    blank. Uses the palette and canvas of :mod:`ciw.lab.svg` so figures match.
    """
    values = [v for _, row in series for v in row.values()]
    low, high = min(0.0, min(values)), max(values)
    span = (high - low) or 1.0
    high += 0.06 * span
    plot_w, plot_h = svg.WIDTH - svg.LEFT - svg.RIGHT, svg.HEIGHT - svg.TOP - svg.BOTTOM
    slot = plot_w / len(categories)

    def py(y):
        return svg.TOP + (high - y) / (high - low) * plot_h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg.WIDTH}" height="{svg.HEIGHT}" '
           f'viewBox="0 0 {svg.WIDTH} {svg.HEIGHT}" font-family="sans-serif" font-size="11">',
           f'<rect width="{svg.WIDTH}" height="{svg.HEIGHT}" fill="#ffffff"/>',
           f'<text x="{svg.LEFT}" y="22" font-size="13" font-weight="bold">{escape(title)}</text>',
           f'<rect x="{svg.LEFT}" y="{svg.TOP}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>']
    step = 10 ** math.floor(math.log10(span / 5))
    step *= next((m for m in (1, 2, 5, 10) if m * step >= span / 5), 10)
    tick = math.ceil(low / step) * step
    while tick <= high:
        y = py(tick)
        out.append(f'<line x1="{svg.LEFT}" y1="{y:.2f}" x2="{svg.LEFT + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{svg.LEFT - 6}" y="{y + 4:.2f}" text-anchor="end">{tick:.3g}</text>')
        tick += step
    for index, name in enumerate(categories):
        x = svg.LEFT + slot * (index + 0.5)
        out.append(f'<text x="{x:.2f}" y="{svg.TOP + plot_h + 16}" text-anchor="middle">{escape(name)}</text>')
    out.append(f'<text x="16" y="{svg.TOP + plot_h / 2:.2f}" text-anchor="middle" '
               f'transform="rotate(-90 16 {svg.TOP + plot_h / 2:.2f})">{escape(ylabel)}</text>')
    for index, (name, row) in enumerate(series):
        color = svg.PALETTE[index % len(svg.PALETTE)]
        offset = (index - (len(series) - 1) / 2) * min(14.0, slot / (len(series) + 1))
        for position, category in enumerate(categories):
            if category in row:
                x = svg.LEFT + slot * (position + 0.5) + offset
                out.append(f'<circle cx="{x:.2f}" cy="{py(row[category]):.2f}" r="4" fill="{color}"/>')
        ly = svg.TOP + 12 + 18 * index
        out.append(f'<circle cx="{svg.LEFT + plot_w + 22}" cy="{ly}" r="4" fill="{color}"/>')
        out.append(f'<text x="{svg.LEFT + plot_w + 32}" y="{ly + 4}">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def run_mean_z(values):
    """z of a grand mean against zero using run-level means (robust to within-run correlation).

    ``values`` is (runs, samples, m); returns per-component z and the grand mean.
    """
    per_run = values.mean(axis=1)
    mean = per_run.mean(axis=0)
    se = per_run.std(axis=0, ddof=1) / math.sqrt(per_run.shape[0])
    return mean / se, mean, se
