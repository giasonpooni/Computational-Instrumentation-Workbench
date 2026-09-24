"""Kernel operation counts, reduction policies and a CPU/GPU comparison harness (T142, T147, T148).

Operation counts come from scalar re-statements of the core kernels executed
on a counting number type, so the count is exact and independent of timing.
Each scalar kernel is checked against the core NumPy implementation it
re-states; the scalar form is the specification a Rust port would follow.
Interpreter dispatch counts (calls issued by ``ciw`` code) are measured with a
profile hook and wall-clock timings are retained only as artifacts.

Reductions: sequential, fixed-tree pairwise, Kahan, Neumaier and a correctly
rounded exact accumulation, with rigorous error bounds (Higham; Ogita, Rump
and Oishi) except Kahan's, whose second-order constant is a declared policy
allowance. The comparison
harness compares a candidate output against a reference under a bitwise,
analytic-bound or absolute/relative policy. No GPU is used here; float32 and
reordered float64 CPU reductions stand in for a device to show the harness
detects differences, and injected faults are judged by the harness itself.
Nothing here measures GPU behaviour.
"""
from __future__ import annotations

from fractions import Fraction
import math
import os
import sys
import time

import numpy as np

U64 = 2.0 ** -53
U32 = 2.0 ** -24
# Unresolved on purpose: code objects carry the path the package was imported from.
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


# ------------------------------------------------------ counted scalars
class _Tally:
    active: dict | None = None


def _tick(kind: str) -> None:
    if _Tally.active is not None:
        _Tally.active[kind] += 1


class Scalar:
    """A float that counts its own arithmetic while a tally is active."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v.v if isinstance(v, Scalar) else float(v)

    @staticmethod
    def _value(other):
        return other.v if isinstance(other, Scalar) else float(other)

    def __add__(self, other):
        _tick("add")
        return Scalar(self.v + self._value(other))

    def __radd__(self, other):
        _tick("add")
        return Scalar(self._value(other) + self.v)

    def __sub__(self, other):
        _tick("add")
        return Scalar(self.v - self._value(other))

    def __rsub__(self, other):
        _tick("add")
        return Scalar(self._value(other) - self.v)

    def __mul__(self, other):
        _tick("mul")
        return Scalar(self.v * self._value(other))

    def __rmul__(self, other):
        _tick("mul")
        return Scalar(self._value(other) * self.v)

    def __truediv__(self, other):
        _tick("div")
        return Scalar(self.v / self._value(other))

    def __rtruediv__(self, other):
        _tick("div")
        return Scalar(self._value(other) / self.v)

    def __neg__(self):
        _tick("neg")
        return Scalar(-self.v)


def csin(x):
    _tick("trig")
    return Scalar(math.sin(Scalar._value(x)))


def ccos(x):
    _tick("trig")
    return Scalar(math.cos(Scalar._value(x)))


def count_ops(function, *args):
    """Run ``function`` with a fresh tally; return (result, counts)."""
    tally = {"add": 0, "mul": 0, "div": 0, "neg": 0, "trig": 0}
    previous, _Tally.active = _Tally.active, tally
    try:
        result = function(*args)
    finally:
        _Tally.active = previous
    tally["flops"] = tally["add"] + tally["mul"] + tally["div"] + tally["neg"]
    return result, tally


def values(items):
    return np.array([Scalar._value(x) for x in items])


# ------------------------------------------------ scalar kernel restatements
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def torus_frames(u, major, minor):
    """First and second embedding derivatives of the torus with shared trigonometry."""
    p, t = u
    sp, cp, st, ct = csin(p), ccos(p), csin(t), ccos(t)
    mst, mct = minor * st, minor * ct
    rho = major + mct
    rsp, rcp = rho * sp, rho * cp
    xu = (-rsp, rcp, Scalar(0.0))
    xv = (-(mst * cp), -(mst * sp), mct)
    xuu = (-rcp, -rsp, Scalar(0.0))
    xuv = (mst * sp, -(mst * cp), Scalar(0.0))
    xvv = (-(mct * cp), -(mct * sp), -mst)
    return (xu, xv), ((xuu, xuv), (xuv, xvv)), ct, rho


def geodesic_rhs_embedded(first, second, v):
    """Generic embedded-surface geodesic acceleration using metric symmetry."""
    g00, g01, g11 = _dot(first[0], first[0]), _dot(first[0], first[1]), _dot(first[1], first[1])
    # dg[k][i][j] = d_k g_ij, computed once per symmetric (i, j) pair.
    dg = [[[None, None], [None, None]] for _ in range(2)]
    for k in range(2):
        for i, j in ((0, 0), (0, 1), (1, 1)):
            dg[k][i][j] = dg[k][j][i] = _dot(second[k][i], first[j]) + _dot(first[i], second[k][j])
    det = g00 * g11 - g01 * g01
    off = -(g01 / det)
    ginv = ((g11 / det, off), (off, g00 / det))
    vv = (v[0] * v[0], v[0] * v[1], v[1] * v[1])
    accel = []
    for k in range(2):
        gamma = {}
        for i, j in ((0, 0), (0, 1), (1, 1)):
            lowered = [dg[i][j][l] + dg[j][i][l] - dg[l][i][j] for l in range(2)]
            gamma[i, j] = 0.5 * (ginv[k][0] * lowered[0] + ginv[k][1] * lowered[1])
        accel.append(-(gamma[0, 0] * vv[0] + 2.0 * (gamma[0, 1] * vv[1]) + gamma[1, 1] * vv[2]))
    return accel


def torus_geodesic_rhs(y, major=2.0, minor=1.0):
    first, second, _, _ = torus_frames(y[:2], major, minor)
    return [y[2], y[3], *geodesic_rhs_embedded(first, second, y[2:4])]


def torus_jacobi_rhs(y, major=2.0, minor=1.0):
    """Geodesic plus both Jacobi columns; curvature K = cos(t) / (r (R + r cos(t)))."""
    first, second, ct, rho = torus_frames(y[:2], major, minor)
    accel = geodesic_rhs_embedded(first, second, y[2:4])
    minus_k = -(ct / (minor * rho))
    return [y[2], y[3], *accel, y[5], minus_k * y[4], y[7], minus_k * y[6]]


def rk4_step(f, y, h):
    """Same association order as ciw.lab.integrators.step_rk4."""
    k1 = f(y)
    half = 0.5 * h
    k2 = f([a + half * b for a, b in zip(y, k1)])
    half = 0.5 * h
    k3 = f([a + half * b for a, b in zip(y, k2)])
    k4 = f([a + h * b for a, b in zip(y, k3)])
    sixth = h / 6.0
    return [a + sixth * (((b + 2 * c) + 2 * d) + e) for a, b, c, d, e in zip(y, k1, k2, k3, k4)]


def rk4_combination_ops(n: int) -> int:
    """Flops outside the right-hand side in one RK4 step on an n-vector: 13 n + 3."""
    return 2 * (1 + 2 * n) + 2 * n + 1 + 7 * n


def transfer(f, y0, length, steps):
    h = Scalar(length) / steps
    y = list(y0)
    for _ in range(steps):
        y = rk4_step(f, y, h)
    return y


def kalman_update(x, P, z, H, R):
    """Joseph-form measurement update (reference kernel, NumPy)."""
    innovation = z - H @ x
    S = H @ P @ H.T + R
    gain = np.linalg.solve(S, H @ P).T
    A = np.eye(len(x)) - gain @ H
    return x + gain @ innovation, A @ P @ A.T + gain @ R @ gain.T


def kalman_update_scalar(x, P, z, H, R):
    """Scalar restatement for m = 2 with a closed-form 2x2 inverse and symmetric products."""
    n, m = len(x), len(z)
    if m != 2:
        raise ValueError("The scalar Kalman restatement supports two measurements")
    innovation = [z[i] - _sum([H[i][j] * x[j] for j in range(n)]) for i in range(m)]
    PHt = [[_sum([P[i][k] * H[j][k] for k in range(n)]) for j in range(m)] for i in range(n)]
    S = [[None] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            S[i][j] = S[j][i] = _sum([H[i][k] * PHt[k][j] for k in range(n)]) + R[i][j]
    det = S[0][0] * S[1][1] - S[0][1] * S[0][1]
    off = -(S[0][1] / det)
    Sinv = [[S[1][1] / det, off], [off, S[0][0] / det]]
    K = [[_sum([PHt[i][k] * Sinv[k][j] for k in range(m)]) for j in range(m)] for i in range(n)]
    x_new = [x[i] + _sum([K[i][j] * innovation[j] for j in range(m)]) for i in range(n)]
    A = [[(1.0 if i == j else 0.0) - _sum([K[i][k] * H[k][j] for k in range(m)]) for j in range(n)] for i in range(n)]
    AP = [[_sum([A[i][k] * P[k][j] for k in range(n)]) for j in range(n)] for i in range(n)]
    KR = [[_sum([K[i][k] * R[k][j] for k in range(m)]) for j in range(m)] for i in range(n)]
    P_new = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            P_new[i][j] = P_new[j][i] = (_sum([AP[i][k] * A[j][k] for k in range(n)])
                                         + _sum([KR[i][k] * K[j][k] for k in range(m)]))
    return x_new, P_new


def _sum(terms):
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def kalman_case(seed: int = 142, n: int = 4, m: int = 2):
    rng = np.random.Generator(np.random.PCG64(seed))
    L = rng.normal(size=(n, n))
    P = L @ L.T + n * np.eye(n)
    M = rng.normal(size=(m, m))
    return rng.normal(size=n), P, rng.normal(size=m), rng.normal(size=(m, n)), M @ M.T + np.eye(m)


# ------------------------------------------------- interpreter dispatches
def dispatch_count(function, *args) -> int:
    """Python and C calls issued from frames inside the ciw package during one invocation."""
    count = 0

    def profile(frame, event, arg):
        nonlocal count
        if event == "call":
            caller = frame.f_back
            if caller is not None and caller.f_code.co_filename.startswith(PACKAGE_DIR):
                count += 1
        elif event == "c_call" and frame.f_code.co_filename.startswith(PACKAGE_DIR):
            count += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        function(*args)
    finally:
        sys.setprofile(previous)
    return count


def time_per_call(function, *args, budget: float = 0.05) -> float:
    """Median seconds per call over repeated batches; retained only as an artifact."""
    samples, calls = [], 1
    start = time.perf_counter()
    function(*args)
    first = time.perf_counter() - start
    calls = max(1, int(budget / 5 / max(first, 1e-7)))
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(calls):
            function(*args)
        samples.append((time.perf_counter() - start) / calls)
    return float(sorted(samples)[2])


# ------------------------------------------------------------- reductions
def sum_sequential(xs) -> float:
    total = 0.0
    for x in xs:
        total += x
    return total


def sum_pairwise(xs) -> float:
    """Fixed binary tree: split at n // 2, so the order depends only on n and the input order."""
    xs = list(xs)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    half = len(xs) // 2
    return sum_pairwise(xs[:half]) + sum_pairwise(xs[half:])


def sum_pairwise_levels(xs) -> float:
    """The same fixed tree as sum_pairwise, built level by level with NumPy vector additions.

    For a power-of-two length, splitting at n // 2 recursively pairs x[2k] with
    x[2k + 1] at the leaves and adjacent partial sums above them, so adding the
    even and odd entries level by level performs the same additions in a
    different implementation (NumPy's elementwise binary64 add instead of
    recursive Python floats). Other lengths are refused: their recursive tree
    is not level-wise.
    """
    values = np.asarray(xs, dtype=np.float64)
    n = values.size
    if n == 0 or n & (n - 1):
        raise ValueError("Level-wise pairwise summation needs a power-of-two length")
    while values.size > 1:
        values = values[0::2] + values[1::2]
    return float(values[0])


def sum_kahan(xs) -> float:
    total = compensation = 0.0
    for x in xs:
        y = x - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def sum_neumaier(xs) -> float:
    total = compensation = 0.0
    for x in xs:
        t = total + x
        if abs(total) >= abs(x):
            compensation += (total - t) + x
        else:
            compensation += (x - t) + total
        total = t
    return total + compensation


def exact_fraction(xs) -> Fraction:
    """Exact rational sum of binary64 values by integer accumulation at scale 2**1074."""
    scale = 1074
    total = 0
    for x in xs:
        numerator, denominator = float(x).as_integer_ratio()
        total += numerator * ((1 << scale) // denominator)
    return Fraction(total, 1 << scale)


def sum_exact(xs) -> float:
    """Correctly rounded sum: exact integer accumulation, one final rounding (int / int)."""
    value = exact_fraction(xs)
    return value.numerator / value.denominator


REDUCTIONS = {"sequential": sum_sequential, "pairwise": sum_pairwise, "kahan": sum_kahan,
              "neumaier": sum_neumaier, "exact": sum_exact}


def gamma(k: int, u: float = U64) -> float:
    return k * u / (1 - k * u)


# Declared second-order allowance for Kahan summation. Higham (2002, eq. 4.8) gives 2u + O(n u^2) with no
# explicit constant; this constant is a policy choice, not a proven bound.
KAHAN_SECOND_ORDER = 4


def error_bound(algorithm: str, n: int, abs_sum: float, exact: float, u: float = U64) -> float:
    """Absolute error bound of each reduction for n terms with S = sum|x| and exact sum s.

    Rigorous: sequential gamma_{n-1} S and pairwise gamma_{ceil(log2 n)} S
    (Higham 2002, ch. 4); Neumaier u|s| + gamma_{n-1}^2 S, because its
    compensation terms are exactly those of Sum2 (Ogita, Rump and Oishi 2005,
    Prop. 4.5); exact accumulation u|s| plus half the smallest subnormal.
    Kahan: first-order 2u S plus the declared allowance KAHAN_SECOND_ORDER n u^2 S.
    """
    if algorithm == "sequential":
        return gamma(max(n - 1, 0), u) * abs_sum
    if algorithm == "pairwise":
        return gamma(math.ceil(math.log2(n)) if n > 1 else 0, u) * abs_sum
    if algorithm == "kahan":
        return 2 * u * abs_sum + KAHAN_SECOND_ORDER * n * u * u * abs_sum
    if algorithm == "neumaier":
        return u * abs(exact) + gamma(max(n - 1, 0), u) ** 2 * abs_sum
    if algorithm == "exact":
        return u * abs(exact) + 2.0 ** -1075
    raise ValueError(f"Unsupported reduction algorithm: {algorithm}")


def superseded_neumaier_bound(n: int, abs_sum: float, exact: float, u: float = U64) -> float:
    """The earlier stated bound 2u|s| + 4 n u^2 S, kept to record where it fails."""
    return 2 * u * abs(exact) + 4 * n * u * u * abs_sum


def absorbed_tiny_terms(count: int = 1000) -> list:
    """1, then ``count`` terms just above 0.7u (each absorbed by 1 when added), then -1.

    The exact sum is count * 0.7u(1 + 2^-20); the compensation accumulates them
    sequentially, so the second-order error grows like n^2 u^2 S.
    """
    return [1.0] + [0.7 * U64 * (1 + 2.0 ** -20)] * count + [-1.0]


def reduction_datasets(seed: int = 148, n: int = 1024) -> dict:
    rng = np.random.Generator(np.random.PCG64(seed))
    uniform = rng.uniform(-1.0, 1.0, n)
    spread = 10.0 ** rng.uniform(-8.0, 8.0, n)
    big = 10.0 ** rng.uniform(0.0, 16.0, n // 2)
    cancelling = np.concatenate([big, -big]) + rng.uniform(-1.0, 1.0, n)
    return {"uniform": [float(x) for x in uniform], "positive-wide-range": [float(x) for x in spread],
            "cancelling": [float(x) for x in cancelling], "kahan-counterexample": [1.0, 1e100, 1.0, -1e100],
            "absorbed-tiny-terms": absorbed_tiny_terms()}


def permutation_study(datasets: dict, permutations: int = 24, seed: int = 1480) -> dict:
    """Distinct results and worst error/bound ratio of each algorithm over seeded permutations."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {}
    for name, xs in datasets.items():
        exact = exact_fraction(xs)
        exact_float = exact.numerator / exact.denominator
        abs_sum = float(exact_fraction([abs(x) for x in xs]))
        orders = [list(xs)] + [[xs[i] for i in rng.permutation(len(xs))] for _ in range(permutations)]
        per = {}
        for algorithm, reduce in REDUCTIONS.items():
            results = [reduce(order) for order in orders]
            bound = error_bound(algorithm, len(xs), abs_sum, exact_float)
            errors = [abs(Fraction(r) - exact) for r in results]
            per[algorithm] = {"distinct_results": len(set(results)), "first_order_result": results[0],
                              "max_abs_error": float(max(errors)), "bound": bound,
                              "max_error_over_bound": float(max(errors) / Fraction(bound)) if bound > 0 else 0.0,
                              "results": results}
        per["fsum"] = {"results": [math.fsum(order) for order in orders]}
        out[name] = {"n": len(xs), "exact": exact_float, "abs_sum": abs_sum, "orders": len(orders),
                     "algorithms": per, "orders_data": orders}
    return out


# -------------------------------------------------- comparison harness
def ulp_distance(a, b) -> np.ndarray:
    """Units in the last place between equal-dtype float arrays (sign-magnitude to monotone map)."""
    a, b = np.asarray(a), np.asarray(b)
    signed, magnitude = {np.dtype(np.float64): (np.int64, (1 << 63) - 1),
                         np.dtype(np.float32): (np.int32, (1 << 31) - 1)}[a.dtype]

    def monotone(v):
        bits = v.view(signed).astype(np.int64).ravel().tolist()
        return [-(x & magnitude) if x < 0 else x for x in bits]

    # Python integers: the difference of two 64-bit monotone keys can exceed int64.
    return np.array([float(abs(x - y)) for x, y in zip(monotone(a), monotone(b))]).reshape(a.shape)


def compare_outputs(reference, candidate, policy: dict) -> dict:
    """Compare candidate to reference elementwise under a declared tolerance policy.

    ``policy`` is {"mode": "bitwise"} or {"mode": "bound", "tolerance": array}
    or {"mode": "abs_rel", "abs": a, "rel": r}. Shapes must match and every
    value must be finite; otherwise the comparison is refused.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("Comparison requires equal shapes")
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate))):
        raise ValueError("Comparison requires finite outputs")
    difference = np.abs(candidate - reference)
    mode = policy["mode"]
    if mode == "bitwise":
        tolerance = np.zeros_like(reference)
        violating = candidate.view(np.int64) != reference.view(np.int64)
    elif mode == "bound":
        tolerance = np.broadcast_to(np.asarray(policy["tolerance"], dtype=np.float64), reference.shape)
        if not np.all(tolerance > 0):
            raise ValueError("Bound policy tolerances must be positive")
        violating = difference > tolerance
    elif mode == "abs_rel":
        tolerance = policy["abs"] + policy["rel"] * np.abs(reference)
        if not np.all(tolerance > 0):
            raise ValueError("Absolute/relative policy tolerances must be positive")
        violating = difference > tolerance
    else:
        raise ValueError(f"Unsupported comparison policy: {mode}")
    # Ratios to the tolerance are defined only for tolerance policies; bitwise mode reports ULPs instead.
    ratio = None if mode == "bitwise" else difference / tolerance
    return {"elements": int(reference.size), "bitwise_differences": int(np.sum(candidate != reference)),
            "violations": int(np.sum(violating)), "max_abs": float(np.max(difference)),
            "max_ulp": float(np.max(ulp_distance(reference, candidate))),
            "max_ratio": None if ratio is None else float(np.max(ratio)), "ratios": ratio,
            "violating": np.asarray(violating, dtype=bool), "passed": not bool(np.any(violating))}


def two_product(a, b):
    """Error-free product a*b = p + e (Dekker split; valid for |a|, |b| < 2**996)."""
    p = a * b
    factor = 134217729.0
    ca, cb = factor * a, factor * b
    ah = ca - (ca - a)
    bh = cb - (cb - b)
    al, bl = a - ah, b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def blocked_rows(products: np.ndarray, width: int) -> np.ndarray:
    """GPU-like order: sequential within lanes of ``width``, then a pairwise tree over lanes."""
    rows, n = products.shape
    lanes = np.cumsum(products.reshape(rows, n // width, width), axis=2)[:, :, -1]
    while lanes.shape[1] > 1:
        half = lanes.shape[1] // 2
        lanes = lanes[:, :half] + lanes[:, half:]
    return lanes[:, 0]


def pairwise_rows(products: np.ndarray) -> np.ndarray:
    values = products
    while values.shape[1] > 1:
        half = values.shape[1] // 2
        values = values[:, :half] + values[:, half:]
    return values[:, 0]


def batched_dot_study(seed: int = 147, rows: int = 128, n: int = 1024, width: int = 32) -> dict:
    """Batched dot products under several reduction orders and precisions, with exact references."""
    rng = np.random.Generator(np.random.PCG64(seed))
    A = rng.uniform(-1.0, 1.0, (rows, n))
    x = rng.uniform(-1.0, 1.0, n)
    products = A * x
    p, e = two_product(A, x)
    exact = np.array([math.fsum(np.concatenate([p[i], e[i]])) for i in range(rows)])
    abs_dot = np.array([math.fsum(np.abs(np.concatenate([p[i], e[i]]))) for i in range(rows)])
    outputs = {
        "f64-sequential": np.cumsum(products, axis=1)[:, -1],
        "f64-pairwise": pairwise_rows(products),
        "f64-blocked": blocked_rows(products, width),
    }
    products32 = A.astype(np.float32) * x.astype(np.float32)
    outputs["f32-blocked"] = blocked_rows(products32, width).astype(np.float64)
    depth_blocked = width - 1 + int(math.log2(n // width))
    # Each term passes through its product rounding plus the additions on its path.
    depths = {"f64-sequential": (n, U64), "f64-pairwise": (int(math.log2(n)) + 1, U64),
              "f64-blocked": (depth_blocked + 1, U64), "f32-blocked": (depth_blocked + 3, U32)}
    bounds = {name: gamma(k, u) * abs_dot for name, (k, u) in depths.items()}
    return {"rows": rows, "n": n, "width": width, "exact": exact, "abs_dot": abs_dot, "outputs": outputs,
            "bounds": bounds, "depths": {k: v[0] for k, v in depths.items()}, "A": A, "x": x}


DETECTION_BINS = (0.0, 0.5, 0.9, 0.99, 1.01, 1.1, 2.0, math.inf)


def dropped_product_study(reference, candidate, products, tolerance, column_scale) -> dict:
    """Every single dropped partial product, judged by compare_outputs (the harness under test).

    Fault (i, j) replaces candidate[i] by candidate[i] - products[i, j]. All
    faults go through one bound-policy comparison against the reference
    broadcast over columns, so detection is what the harness reports; this
    function only tallies it against predictions that use the policy
    tolerance and the fault size, never the fault-free deviation of a row:

    - operational guarantee: when the fault-free candidate passes the policy
      (|c_i - r_i| <= tol_i, reported by the harness), a fault with
      |p| > 2 tol_i is detected (up to rounding far below tol_i);
    - threshold: with m = the harness's largest fault-free |c - r| / tol,
      faults below (1 - m) tol are missed and faults above (1 + m) tol are
      detected, so detection switches at the bound;
    - miss count: for products a_ij s_j with a_ij uniform on [-1, 1) and
      column scale s_j, P(|p_ij| <= tol_i) = min(1, tol_i / |s_j|), whose sum
      (with the Bernoulli variance) predicts the number of missed faults.
    """
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    products, tolerance = np.asarray(products, float), np.asarray(tolerance, float)
    scale = np.abs(np.asarray(column_scale, float))
    policy = {"mode": "bound", "tolerance": tolerance}
    clean = compare_outputs(reference, candidate, policy)
    faulty = candidate[:, None] - products
    judged = compare_outputs(np.broadcast_to(reference[:, None], faulty.shape), faulty,
                             {"mode": "bound", "tolerance": tolerance[:, None]})
    detected = np.asarray(judged["violating"], dtype=bool)
    magnitude = np.abs(products)
    ratio = magnitude / tolerance[:, None]
    m = float(clean["max_ratio"])
    miss = np.minimum(1.0, tolerance[:, None] / scale[None, :])
    expected, sigma = float(np.sum(miss)), float(math.sqrt(np.sum(miss * (1.0 - miss))))
    undetected = int(np.sum(~detected))

    def witness(cells):
        if not len(cells):
            return None
        i, j = max(cells.tolist(), key=lambda ij: (magnitude[ij[0], ij[1]], -ij[0], -ij[1]))
        return {"row": int(i), "column": int(j), "magnitude": float(magnitude[i, j]),
                "row_tolerance": float(tolerance[i]), "ratio": float(ratio[i, j])}

    bins = []
    for low, high in zip(DETECTION_BINS[:-1], DETECTION_BINS[1:]):
        inside = (ratio >= low) & (ratio < high)
        bins.append({"ratio_from": low, "ratio_to": None if math.isinf(high) else high, "faults": int(np.sum(inside)),
                     "detected": int(np.sum(inside & detected))})
    return {"faults": int(products.size), "undetected": undetected,
            "fault_free_violations": clean["violations"], "fault_free_max_ratio": m,
            "above_twice_tolerance": int(np.sum(ratio > 2.0)),
            "undetected_above_twice_tolerance": int(np.sum((ratio > 2.0) & ~detected)),
            "undetected_above_bound": int(np.sum((ratio > 1.0) & ~detected)),
            "detected_below_band": int(np.sum((ratio < 1.0 - m) & detected)),
            "undetected_above_band": int(np.sum((ratio > 1.0 + m) & ~detected)),
            "expected_undetected": expected, "sigma_undetected": sigma,
            "undetected_z": (undetected - expected) / sigma if sigma > 0 else float(undetected - expected),
            "max_undetected_ratio": float(np.max(ratio[~detected], initial=0.0)),
            "largest_undetected": witness(np.argwhere(~detected)),
            "above_bound_witness": witness(np.argwhere((ratio > 1.0) & ~detected)),
            "detection_by_ratio": bins,
            "max_tolerance": float(np.max(tolerance)), "min_tolerance": float(np.min(tolerance))}
