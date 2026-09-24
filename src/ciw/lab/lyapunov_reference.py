"""Independent references for the PLSR Lyapunov experiments (T101-T114).

Everything here is CIW code and never imports the PLSR runtime: float64
re-derivations of the documented PLSR decrease form, resolution bound and
decision order; exact dyadic-rational classification of the declared binary64
inputs; optional SciPy Lyapunov solvers imported lazily; and seeded input
generators, some built in exact rational arithmetic without BLAS or LAPACK so
that every kernel declares the same inputs. Exact arithmetic decides what the
declared numbers imply mathematically. It says nothing about a physical plant.
The float64 re-derivations transcribe the runtime's documented procedure, so
agreement with them is a same-specification check (recorded as an ordinary
check, never as an independent one): it shows that the runtime implements its
own documentation, not that the documentation is sufficient.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from .. import __version__

U = 2.0 ** -53                      # binary64 unit roundoff
TINY = 2.0 ** -1074                 # smallest positive subnormal
# LAPACK dsyevd rescales by a factor that is not a power of two when the
# largest entry lies outside [RMIN, RMAX] = [2^-485, 2^485]; inside that window
# its arithmetic commutes with exact power-of-two scaling.
LAPACK_RMIN, LAPACK_RMAX = 2.0 ** -485, 2.0 ** 485

RUNTIME_CODES = ("CERTIFIED_WITH_MARGIN", "MARGIN_LOW", "NOT_CERTIFIED", "DECREASE_NOT_DEFINITE",
                 "NUMERICAL_INCONCLUSIVE", "NUMERICAL_OVERFLOW", "OUTSIDE_PARAMETER_BOX", "OUTSIDE_LEVEL_SET",
                 "CERTIFICATE_NOT_POSITIVE")
HOST_OWNED = ("MODEL_MISMATCH", "STALE_STATE", "INVALID_SENSOR_DATA", "CERTIFICATE_EXPIRED", "RUNTIME_FAULT")
CERTIFYING = frozenset({"CERTIFIED_WITH_MARGIN", "MARGIN_LOW"})


def generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def jsonable(value):
    """Plain JSON values; non-finite floats become strings so retention never fails."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else ("nan" if math.isnan(value) else ("inf" if value > 0 else "-inf"))
    return value


def hexed(matrix) -> list:
    """Exact binary64 spelling of a vector or matrix for counterexample witnesses."""
    array = np.asarray(matrix, dtype=float)
    return [hexed(row) for row in array] if array.ndim > 1 else [float(v).hex() for v in array]


# Float64 re-derivations of the documented PLSR arithmetic ------------------

def decrease_matrix(A, P, time="continuous", P_rate=None):
    """Continuous A^T P + P A (+ P_rate), discrete A^T P A - P, symmetrised as documented."""
    A, P = np.asarray(A, dtype=float), np.asarray(P, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        if time == "continuous":
            form = A.T @ P + P @ A
            if P_rate is not None:
                form = form + np.asarray(P_rate, dtype=float)
        elif time == "discrete":
            form = A.T @ P @ A - P
        else:
            raise ValueError("time must be continuous or discrete")
        return 0.5 * form + 0.5 * form.T


def resolution(A, P, time="continuous", P_rate=None, form=None) -> float:
    """The documented float64 resolution n*|E|max + n^3*u*max|M| (Higham, Weyl, LAPACK)."""
    A, P = np.asarray(A, dtype=float), np.asarray(P, dtype=float)
    n = float(A.shape[0])
    k = n + 3.0
    gamma = k * U / (1.0 - k * U) if 1.0 - k * U > 0.0 else math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        a, p = float(np.max(np.abs(A))), float(np.max(np.abs(P)))
        if time == "continuous":
            formation = 2.0 * gamma * n * a * p
            if P_rate is not None:
                formation += U * float(np.max(np.abs(np.asarray(P_rate, dtype=float))))
        else:
            formation = 2.0 * gamma * n * n * a * a * p + U * p
        if form is None:
            form = decrease_matrix(A, P, time, P_rate)
        eigensolver = n * n * n * U * float(np.max(np.abs(form)))
        return n * formation + eigensolver


def state_scale_exponent(x) -> int:
    peak = float(np.max(np.abs(np.asarray(x, dtype=float))))
    if peak == 0.0 or not math.isfinite(peak):
        return 0
    return int(np.frexp(peak)[1]) - 1


def documented_level_exceeded(scaled_value: float, exponent: int, level: float) -> bool:
    """The documented level gate, including its treatment of an unrepresentable s^2."""
    if level < 0.0:
        return True
    if level == 0.0:
        return scaled_value > 0.0
    scale = float(np.ldexp(1.0, exponent))
    with np.errstate(over="ignore", under="ignore"):
        squared = scale * scale
    if not math.isfinite(squared):
        return scaled_value > 0.0
    if squared == 0.0:
        return False
    return scaled_value > level / squared


# The conditions of the documented decision order, in order. The first true
# one of the first five decides the code; then ``certified`` (MARGIN_LOW when
# ``margin_low`` holds as well: a certified margin at or below the declared
# one), then ``not_definite``; otherwise NUMERICAL_INCONCLUSIVE.
GATES = ("box", "overflow", "not_positive", "level", "scalar_positive", "certified", "margin_low", "not_definite")
LATER_GATES = GATES[2:]
_GATE_CODES = (("box", "OUTSIDE_PARAMETER_BOX"), ("overflow", "NUMERICAL_OVERFLOW"),
               ("not_positive", "CERTIFICATE_NOT_POSITIVE"), ("level", "OUTSIDE_LEVEL_SET"),
               ("scalar_positive", "NOT_CERTIFIED"))


def gate_code(gates) -> str:
    """The code the documented order assigns to a vector of gate conditions."""
    for gate, code in _GATE_CODES:
        if gates[gate]:
            return code
    if gates["certified"]:
        return "MARGIN_LOW" if gates["margin_low"] else "CERTIFIED_WITH_MARGIN"
    return "DECREASE_NOT_DEFINITE" if gates["not_definite"] else "NUMERICAL_INCONCLUSIVE"


def _documented_quantities(A, P, x, time):
    """The documented float64 quantities at one sample, or None where the arithmetic leaves binary64."""
    form = decrease_matrix(A, P, time)
    exponent = state_scale_exponent(x)
    unit = x / float(np.ldexp(1.0, exponent))
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.array([unit @ P @ unit, unit @ form @ unit, unit @ unit])
    if not (np.all(np.isfinite(form)) and np.all(np.isfinite(scaled))):
        return None
    max_decrease = float(np.max(np.linalg.eigvalsh(form)))
    return {"exponent": exponent, "scaled_value": float(scaled[0]), "scaled_decrease": float(scaled[1]),
            "unit_norm2": float(scaled[2]), "resolution": resolution(A, P, time, form=form),
            "min_P": float(np.min(np.linalg.eigvalsh(P))), "max_decrease": max_decrease, "margin": -max_decrease}


def documented_gates(A, P, x, time="continuous", level=None, required_margin=0.0, in_box=True,
                     rescale=True) -> dict:
    """Every condition of the documented decision order, evaluated whether or not an earlier one decides.

    Returns the gate booleans, the code they give (``gate_code``) and the documented quantities. Where the
    arithmetic overflows in continuous time and ``rescale`` is set, the later conditions are read at
    ``2^-s A`` (and ``2^-s required_margin``) for the smallest ``s`` that fits: each of them is homogeneous
    in that power-of-two scale, so these are the values the order would read at the declared A if the
    arithmetic could hold them. Otherwise they are None. The level gate does not involve A.
    """
    A, P, x = np.asarray(A, dtype=float), np.asarray(P, dtype=float), np.asarray(x, dtype=float)
    gates = dict.fromkeys(GATES, False)
    gates["box"] = not in_box
    facts, shift = _documented_quantities(A, P, x, time), 0
    if facts is None:
        gates["overflow"] = True
        if rescale and time == "continuous":
            for shift in range(1, 2200):
                facts = _documented_quantities(np.ldexp(A, -shift), P, x, time)
                if facts is not None:
                    break
    if facts is None:
        gates.update(dict.fromkeys(LATER_GATES))
        return {"gates": gates, "code": gate_code(gates), "shift": None, "facts": None}
    res, margin = facts["resolution"], facts["margin"]
    gates.update(
        not_positive=facts["min_P"] <= 0.0 or facts["scaled_value"] < 0.0,
        level=level is not None and documented_level_exceeded(facts["scaled_value"], facts["exponent"],
                                                               float(level)),
        scalar_positive=facts["scaled_decrease"] > res * facts["unit_norm2"],
        certified=margin > max(res, 0.0),
        not_definite=facts["max_decrease"] > res)
    # MARGIN_LOW refines a certified margin, so its condition is read only on that branch.
    gates["margin_low"] = gates["certified"] and margin <= float(np.ldexp(float(required_margin), -shift))
    return {"gates": gates, "code": gate_code(gates), "shift": shift, "facts": facts}


def documented_code(A, P, x, time="continuous", level=None, required_margin=0.0, in_box=True) -> dict:
    """Re-derive the runtime-status-v1 decision order in CIW from A, P and x.

    Order: outside box; overflow; certificate not positive; level; resolvably
    positive scalar decrease; margin beyond resolution (MARGIN_LOW against the
    declared margin); resolvably indefinite form; otherwise inconclusive.
    """
    if not in_box:
        return {"code": "OUTSIDE_PARAMETER_BOX"}
    result = documented_gates(A, P, x, time, level, required_margin, rescale=False)
    if result["gates"]["overflow"]:
        return {"code": "NUMERICAL_OVERFLOW"}
    facts = result["facts"]
    return {"resolution": facts["resolution"], "margin": facts["margin"],
            "scaled_decrease": facts["scaled_decrease"], "code": result["code"]}


# Exact dyadic-rational arithmetic on the declared binary64 inputs ------------

def fractions(matrix) -> list:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        return [Fraction(float(v)) for v in array]
    return [[Fraction(float(v)) for v in row] for row in array]


def exact_form(A, P, time="continuous") -> list:
    """The decrease form of the declared float inputs in exact rational arithmetic."""
    a, p = fractions(A), fractions(P)
    n = len(a)
    if time == "continuous":
        ap = [[sum(a[k][i] * p[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        return [[ap[i][j] + ap[j][i] for j in range(n)] for i in range(n)]
    pa = [[sum(p[i][k] * a[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[sum(a[k][i] * pa[k][j] for k in range(n)) - p[i][j] for j in range(n)] for i in range(n)]


def representable(q) -> bool:
    """q is zero or rounds (to nearest) to a finite nonzero binary64 number."""
    if q == 0:
        return True
    try:
        value = float(Fraction(q))  # correctly rounded; OverflowError when it rounds beyond the largest double
    except OverflowError:
        return False
    return value != 0.0 and math.isfinite(value)


def exact_quadratic(x, M) -> Fraction:
    v = [Fraction(float(t)) for t in np.asarray(x, dtype=float)] if not isinstance(x[0], Fraction) else x
    n = len(v)
    return sum(v[i] * M[i][j] * v[j] for i in range(n) for j in range(n))


def _integers(M) -> list:
    """Scale a dyadic rational matrix by one power of two to integers (signs of minors unchanged)."""
    shift = max(entry.denominator.bit_length() - 1 for row in M for entry in row)
    return [[entry.numerator << (shift - (entry.denominator.bit_length() - 1)) for entry in row] for row in M]


def _leading_minors(M) -> list:
    """Leading principal minors by fraction-free Bareiss elimination (no pivoting)."""
    m = _integers(M)
    n, previous, minors = len(m), 1, []
    for k in range(n):
        minors.append(m[k][k])
        if m[k][k] == 0:
            # Later minors are not needed by the callers once one is zero.
            return minors + [0] * (n - k - 1)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // previous
        previous = m[k][k]
    return minors


def exact_det(M) -> int:
    """Sign-exact determinant of a dyadic rational matrix (scaled integer, pivoting Bareiss)."""
    m = _integers(M)
    n, previous, sign = len(m), 1, 1
    for k in range(n):
        pivot = next((r for r in range(k, n) if m[r][k] != 0), None)
        if pivot is None:
            return 0
        if pivot != k:
            m[k], m[pivot], sign = m[pivot], m[k], -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // previous
        previous = m[k][k]
    return sign * m[n - 1][n - 1]


def positive_definite(S) -> bool:
    """Sylvester's criterion in exact arithmetic."""
    return all(minor > 0 for minor in _leading_minors(S))


def negate(M) -> list:
    return [[-v for v in row] for row in M]


def exact_class(M) -> str:
    """negative_definite, negative_semidefinite (singular) or has_positive_eigenvalue, exactly."""
    S = negate(M)
    if positive_definite(S):
        return "negative_definite"
    n = len(S)
    for size in range(1, n + 1):
        for index in combinations(range(n), size):
            if exact_det([[S[i][j] for j in index] for i in index]) < 0:
                return "has_positive_eigenvalue"
    return "negative_semidefinite"


def lambda_max_below(M, t) -> bool:
    """Exactly decide max eig(M) < t for a dyadic threshold t."""
    t = Fraction(t)
    n = len(M)
    return positive_definite([[(t if i == j else 0) - M[i][j] for j in range(n)] for i in range(n)])


def resolution_bin(M, res: float) -> str:
    """Exact position of max eig(M) relative to multiples of the resolution."""
    if res == 0.0:
        return "zero resolution"
    for bound, label in ((-2.0, "below -2 res"), (-1.0, "[-2, -1) res"), (0.0, "[-1, 0) res"),
                         (1.0, "[0, 1) res"), (2.0, "[1, 2) res")):
        if lambda_max_below(M, Fraction(bound) * Fraction(res)):
            return label
    return "at or above 2 res"


def lambda_bracket(M, res: float, low=-16, high=16, steps=24) -> tuple:
    """Exact bracket [lo, hi) of max eig(M) / res by bisection (width (high - low) 2^-steps; M in rationals)."""
    low, high, res = Fraction(low), Fraction(high), Fraction(res)
    for _ in range(steps):
        middle = (low + high) / 2
        if lambda_max_below(M, middle * res):
            high = middle
        else:
            low = middle
    return low, high


# Lyapunov solvers independent of PLSR ---------------------------------------

def kron_lyapunov(A, Q, time="continuous"):
    """CIW Kronecker solve of A^T P + P A = -Q or A^T P A - P = -Q (column-major vec)."""
    A, Q = np.asarray(A, dtype=float), np.asarray(Q, dtype=float)
    n = A.shape[0]
    identity = np.eye(n)
    if time == "continuous":
        operator = np.kron(identity, A.T) + np.kron(A.T, identity)
    else:
        operator = np.kron(A.T, A.T) - np.eye(n * n)
    P = np.linalg.solve(operator, -Q.reshape(-1, order="F")).reshape((n, n), order="F")
    return 0.5 * P + 0.5 * P.T


def scipy_lyapunov(A, Q, time="continuous"):
    """SciPy Bartels-Stewart solution, or None when SciPy is absent."""
    try:
        from scipy import linalg
    except ImportError:
        return None
    A, Q = np.asarray(A, dtype=float), np.asarray(Q, dtype=float)
    if time == "continuous":
        P = linalg.solve_continuous_lyapunov(A.T, -Q)
    else:
        P = linalg.solve_discrete_lyapunov(A.T, Q)
    return 0.5 * P + 0.5 * P.T


def independent_lyapunov(A, Q, time="continuous"):
    """SciPy when installed, otherwise the CIW Kronecker solve; returns (P, implementation, revision)."""
    P = scipy_lyapunov(A, Q, time)
    if P is not None:
        import scipy

        return P, f"scipy.linalg.solve_{time}_lyapunov@{scipy.__version__}", scipy.__version__
    return kron_lyapunov(A, Q, time), "ciw.lab.lyapunov_reference.kron_lyapunov", __version__


# Seeded generators ----------------------------------------------------------

def random_orthogonal(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def random_spd(rng, n, condition=10.0):
    q = random_orthogonal(rng, n)
    P = q @ np.diag(np.geomspace(1.0, condition, n)) @ q.T
    return 0.5 * P + 0.5 * P.T


def _threshold_builder(rng, n, P, skew):
    """A(top) = P^{-1}(N/2 + K): A^T P + P A = N before rounding, N = Q diag(top, rest) Q^T.

    A large skew K makes the resolution large while N alone sets the spectrum
    of the decrease form; the exact spectrum of the rounded A is decided
    separately by exact_form.
    """
    P = np.eye(n) if P is None else np.asarray(P, dtype=float)
    q = random_orthogonal(rng, n)
    rest = -rng.uniform(0.5, 2.0, size=n - 1)
    G = rng.normal(size=(n, n))
    K = skew * 0.5 * (G - G.T)

    def build(top):
        N = q @ np.diag(np.concatenate(([top], rest))) @ q.T
        N = 0.5 * N + 0.5 * N.T
        return np.linalg.solve(P, 0.5 * N + K)

    return P, build


def near_threshold(rng, n, kappa, P=None, skew=100.0, time="continuous"):
    """A continuous-time A whose decrease form has max eig close to kappa * resolution."""
    P, build = _threshold_builder(rng, n, P, skew)
    return build(kappa * resolution(build(0.0), P, time)), P


def razor_edge(rng, n, P=None, side=-1.0, skew=100.0, iterations=48):
    """A with |max eig(M)| / resolution just above 1 as computed in float64 (side -1: certifiable side)."""
    P, build = _threshold_builder(rng, n, P, skew)
    base = resolution(build(0.0), P)

    def ratio(kappa):
        A = build(side * kappa * base)
        form = decrease_matrix(A, P)
        return side * float(np.max(np.linalg.eigvalsh(form))) / resolution(A, P, form=form), A

    low, high = 0.5, 2.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if ratio(middle)[0] > 1.0:
            high = middle
        else:
            low = middle
    value, A = ratio(high)
    return A, P, value


def razor_edge_discrete(rng, n, side=-1.0, iterations=48):
    """Discrete-time A = Q diag(s, r) with P = I whose computed |max eig(A^T A - I)| / resolution is just above 1.

    A^T A - I = diag(s^2 - 1, r^2 - 1) before rounding (r in [0.3, 0.8]), so s sets the top eigenvalue; the
    bisection places it about one resolution from zero on the certifiable (side -1) or indefinite side.
    """
    q = random_orthogonal(rng, n)
    rest = rng.uniform(0.3, 0.8, size=n - 1)
    P = np.eye(n)

    def build(top):
        return q @ np.diag(np.concatenate(([math.sqrt(1.0 + top)], rest)))

    base = resolution(build(0.0), P, "discrete")

    def ratio(kappa):
        A = build(side * kappa * base)
        form = decrease_matrix(A, P, "discrete")
        return side * float(np.max(np.linalg.eigvalsh(form))) / resolution(A, P, "discrete", form=form), A

    low, high = 0.5, 2.0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if ratio(middle)[0] > 1.0:
            high = middle
        else:
            low = middle
    value, A = ratio(high)
    return A, P, value


# Platform-independent generators: integer draws and exact rational arithmetic, rounded once to binary64, so
# the declared inputs never pass through BLAS or LAPACK and every kernel declares the same matrices.

def solve_fractions(M, B) -> list:
    """Exact M^-1 B for rational matrices by Gauss-Jordan elimination (M nonsingular)."""
    n = len(M)
    rows = [list(row) + list(rhs) for row, rhs in zip(M, B)]
    for column in range(n):
        pivot = next(r for r in range(column, n) if rows[r][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for r in range(n):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [row[n:] for row in rows]


def _product(a, b) -> list:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a) -> list:
    return [list(row) for row in zip(*a)]


def rounded(M):
    """Each rational entry correctly rounded to binary64."""
    return np.array([[float(v) for v in row] for row in M])


def cayley_orthogonal(rng, n) -> list:
    """Exactly orthogonal rational Q = (I - S)(I + S)^-1 for a seeded skew S with entries in sixteenths."""
    S = [[Fraction(0)] * n for _ in range(n)]
    for i, j in combinations(range(n), 2):
        S[i][j] = Fraction(int(rng.integers(-16, 17)), 16)
        S[j][i] = -S[i][j]
    minus = [[Fraction(i == j) - S[i][j] for j in range(n)] for i in range(n)]
    plus = [[Fraction(i == j) + S[i][j] for j in range(n)] for i in range(n)]
    return _transpose(solve_fractions(_transpose(plus), _transpose(minus)))


def rounded_resolution(A, P, time="continuous") -> float:
    """The documented resolution with max|M| read from the correctly rounded exact form instead of a BLAS product.

    It is the same number on every kernel; the runtime's own resolution reads max|M| from its formed matrix and
    can differ from it only in the last bits of that one term.
    """
    return resolution(A, P, time, form=rounded(exact_form(A, P, time)))


def exact_threshold_builder(rng, n, spd=False, skew=100.0):
    """A platform-independent (P, A(top)) with A^T P + P A = N(top) before one final rounding of A.

    N(top) = Q diag(top, rest) Q^T with an exactly orthogonal rational Q, rest in [-2, -1/2] in 64ths, the skew
    part K = skew (G - G^T) / 2 with G in 64ths, and P = I or Q2 diag(1 .. 10) Q2^T rounded entrywise (SPD).
    A(top) = P^-1 (N/2 + K) is solved in rationals and rounded once, so max eig of the exact form of the declared
    A is top up to that rounding (a few hundredths of a resolution); callers locate it exactly.
    """
    Q = cayley_orthogonal(rng, n)
    rest = [-(Fraction(1, 2) + Fraction(int(rng.integers(0, 97)), 64)) for _ in range(n - 1)]
    G = [[Fraction(int(rng.integers(-64, 65)), 64) for _ in range(n)] for _ in range(n)]
    K = [[Fraction(skew) * (G[i][j] - G[j][i]) / 2 for j in range(n)] for i in range(n)]
    if spd:
        Q2 = cayley_orthogonal(rng, n)
        D = [[Fraction(1) + Fraction(9 * i, n - 1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
        P = rounded(_product(_product(Q2, D), _transpose(Q2)))
    else:
        P = np.eye(n)
    exact_P = fractions(P)

    def build(top):
        D = [[(Fraction(top) if i == 0 else rest[i - 1]) if i == j else Fraction(0) for j in range(n)]
             for i in range(n)]
        N = _product(_product(Q, D), _transpose(Q))
        return rounded(solve_fractions(exact_P, [[N[i][j] / 2 + K[i][j] for j in range(n)] for i in range(n)]))

    return P, build


def dyadic_state(rng, n) -> np.ndarray:
    """A nonzero seeded state with entries in 32nds of [-2, 2]."""
    while True:
        x = rng.integers(-64, 65, size=n)
        if np.any(x != 0):
            return x.astype(float) / 32.0
