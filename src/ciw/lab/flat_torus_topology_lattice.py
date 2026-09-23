"""Exact lattice algebra for flat tori C / Lambda (tasks T019-T026, T031).

A lattice is carried by its Gram matrix G = B^T B of an oriented basis
(w1, w2), so a lattice with irrational coordinates (the hexagonal lattice)
still has an integer Gram matrix and every basis change, reduction and length
comparison is exact in integers or :class:`fractions.Fraction`. A basis change
B' = B M acts as G' = M^T G M; lattice coordinates transform as c' = M^{-1} c.
The shape is tau = w2 / w1 = (G12 + i sqrt(det G)) / G11.

Floating-point entry points (``tau`` inputs) exist only for comparisons with
float providers and are marked as such. Nothing here measures a surface.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import math

import numpy as np

S = ((0, -1), (1, 0))


def T(k: int = 1):
    return ((1, k), (0, 1))


class LatticeRefusal(ValueError):
    """A basis change or form outside the declared exact domain, with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def exact(value):
    """Integers stay integers; everything else becomes an exact Fraction."""
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise LatticeRefusal("FLOAT_ENTRY", "Exact lattice arithmetic refuses floating-point entries")
    return Fraction(value)


def gram(G):
    """Normalize a symmetric positive-definite 2x2 Gram matrix to exact entries (a, b, c)."""
    (a, b), (b2, c) = G
    a, b, b2, c = (exact(x) for x in (a, b, b2, c))
    if b != b2:
        raise LatticeRefusal("GRAM_NOT_SYMMETRIC", "Gram matrix must be symmetric")
    if not (a > 0 and a * c - b * b > 0):
        raise LatticeRefusal("GRAM_NOT_POSITIVE_DEFINITE", "Gram matrix must be positive definite")
    return (a, b, c)


def gram_of_basis(w1, w2):
    """Exact Gram entries of a basis with rational coordinates."""
    w1, w2 = [tuple(exact(x) for x in w) for w in (w1, w2)]
    return gram(((w1[0] ** 2 + w1[1] ** 2, w1[0] * w2[0] + w1[1] * w2[1]),
                 (w1[0] * w2[0] + w1[1] * w2[1], w2[0] ** 2 + w2[1] ** 2)))


def det_form(form):
    a, b, c = form
    return a * c - b * b


def quad(form, m, n):
    """Q(m, n) = |m w1 + n w2|^2, exact for exact inputs."""
    a, b, c = form
    return a * m * m + 2 * b * m * n + c * n * n


def bilinear(form, x, y):
    a, b, c = form
    return a * x[0] * y[0] + b * (x[0] * y[1] + x[1] * y[0]) + c * x[1] * y[1]


def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)) for i in range(2))


def det2(M) -> int:
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def inverse_sl2(M):
    """Integer inverse of an SL(2, Z) matrix; refuses anything else."""
    check_sl2z(M)
    (a, b), (c, d) = M
    return ((d, -b), (-c, a))


def check_sl2z(M):
    if not all(isinstance(x, int) and not isinstance(x, bool) for row in M for x in row):
        raise LatticeRefusal("BASIS_CHANGE_NOT_INTEGER", "Basis change must have integer entries")
    determinant = det2(M)
    if determinant == -1:
        raise LatticeRefusal("BASIS_CHANGE_REVERSES_ORIENTATION",
                             "Basis change reverses orientation (det -1): same lattice, not an SL(2,Z) change")
    if determinant != 1:
        raise LatticeRefusal("BASIS_CHANGE_NOT_UNIMODULAR",
                             f"Basis change is not unimodular (det {determinant}): it generates a sublattice")
    return M


def transform(form, M):
    """G' = M^T G M for M in SL(2, Z); refuses non-unimodular or orientation-reversing M."""
    check_sl2z(M)
    a, b, c = form
    (p, q), (r, s) = M
    return (a * p * p + 2 * b * p * r + c * r * r,
            a * p * q + b * (p * s + q * r) + c * r * s,
            a * q * q + 2 * b * q * s + c * s * s)


def transform_any(form, M):
    """M^T G M for any integer matrix (used only to exhibit refused counterexamples)."""
    a, b, c = form
    (p, q), (r, s) = M
    return (a * p * p + 2 * b * p * r + c * r * r,
            a * p * q + b * (p * s + q * r) + c * r * s,
            a * q * q + 2 * b * q * s + c * s * s)


def is_reduced(form) -> bool:
    """Non-strict Gauss conditions |2b| <= a <= c (closed fundamental domain)."""
    a, b, c = form
    return abs(2 * b) <= a <= c


def is_canonical(form) -> bool:
    """Gauss canonical form: reduced, and b >= 0 whenever |2b| = a or a = c."""
    a, b, c = form
    return is_reduced(form) and (b >= 0 or (abs(2 * b) < a and a < c))


def _round_half_up(x) -> int:
    return math.floor(x + Fraction(1, 2)) if isinstance(x, (int, Fraction)) else math.floor(x + 0.5)


def gauss_reduce(form):
    """Lagrange/Gauss reduction with the unimodular matrix that achieves it.

    Returns (canonical form, M) with transform(form, M) == canonical form, and
    the number of S steps. Exact for integer or Fraction forms.
    """
    form = gram(((form[0], form[1]), (form[1], form[2])))
    M = ((1, 0), (0, 1))
    steps = 0
    while True:
        a, b, c = form
        q = _round_half_up(Fraction(b) / Fraction(a))
        if q:
            step = T(-q)
            form, M = transform(form, step), matmul(M, step)
        a, b, c = form
        if c < a:
            form, M = transform(form, S), matmul(M, S)
            steps += 1
            continue
        break
    a, b, c = form
    if 2 * b == -a:
        form, M = transform(form, T(1)), matmul(M, T(1))
    a, b, c = form
    if a == c and b < 0:
        form, M = transform(form, S), matmul(M, S)
    if not is_canonical(form):
        raise LatticeRefusal("REDUCTION_FAILED", "Gauss reduction did not reach the canonical domain")
    return form, M, steps


def sl2z_matrices(bound: int):
    """Every SL(2, Z) matrix with entries in [-bound, bound], in lexicographic order."""
    out = []
    for a, b, c, d in product(range(-bound, bound + 1), repeat=4):
        if a * d - b * c == 1:
            out.append(((a, b), (c, d)))
    return out


def random_word(rng, letters: int, max_power: int = 3):
    """A seeded random product of S and T^k letters; returns (matrix, word)."""
    M = ((1, 0), (0, 1))
    word = []
    for _ in range(letters):
        if rng.random() < 0.5:
            M, word = matmul(M, S), word + ["S"]
        else:
            k = int(rng.integers(1, max_power + 1)) * (1 if rng.random() < 0.5 else -1)
            M, word = matmul(M, T(k)), word + [f"T^{k}"]
    return M, word


def reduced_basis_count(form, bound: int = 2):
    """Number of SL(2, Z) bases of the lattice satisfying the closed reduction conditions."""
    canonical, _, _ = gauss_reduce(form)
    return sum(1 for M in sl2z_matrices(bound) if is_reduced(transform(canonical, M)))


def lattice_vectors(form, radius_sq, include_zero=False):
    """All (m, n) with Q(m, n) <= radius_sq, exact comparison, sorted by (Q, m, n).

    Rows n are scanned Fincke-Pohst style: Q = a (m + b n / a)^2 + (det / a) n^2,
    so each row needs only the m-interval around -b n / a. Float bounds are
    widened by one and every candidate is filtered exactly, so skewed
    (unreduced) bases cost O(sqrt(a)) rows rather than a full box.
    """
    a, b, c = form
    af, bf, detf, r2 = float(a), float(b), float(det_form(form)), float(radius_sq)
    rows = int(math.floor(math.sqrt(r2 * af / detf))) + 1
    out = []
    for n in range(-rows, rows + 1):
        center = -bf * n / af
        half = math.sqrt(max(0.0, (r2 - detf * n * n / af) / af))
        for m in range(int(math.floor(center - half)) - 1, int(math.ceil(center + half)) + 2):
            if not include_zero and m == 0 and n == 0:
                continue
            value = quad(form, m, n)
            if value <= radius_sq:
                out.append((value, m, n))
    out.sort()
    return out


def length_spectrum(form, radius_sq):
    """Exact multiset of squared lengths as sorted (Q, multiplicity) pairs."""
    spectrum = {}
    for value, _, _ in lattice_vectors(form, radius_sq):
        spectrum[value] = spectrum.get(value, 0) + 1
    return sorted(spectrum.items())


def systole_sq(form):
    """Squared length of the shortest nonzero vector (first entry of the reduced form)."""
    return gauss_reduce(form)[0][0]


def tau_of(form) -> complex:
    """Shape w2 / w1 in the upper half-plane (float, for reporting and providers only)."""
    a, b, c = (float(x) for x in form)
    return complex(b / a, math.sqrt(a * c - b * b) / a)


def area_one_form(tau: complex):
    """Float Gram entries of the area-one lattice y^{-1/2} (Z + tau Z)."""
    x, y = float(tau.real), float(tau.imag)
    if not y > 0:
        raise LatticeRefusal("TAU_NOT_IN_UPPER_HALF_PLANE", "tau must lie in the upper half-plane")
    return (1.0 / y, x / y, (x * x + y * y) / y)


def float_reduce(tau: complex, tol: float = 1e-12):
    """Float Gauss reduction of tau with the same canonical boundary rule (b >= 0 on boundaries)."""
    a, b, c = area_one_form(tau)
    M = ((1, 0), (0, 1))
    for _ in range(10000):
        q = math.floor(b / a + 0.5)
        if q:
            a, b, c = transform_any((a, b, c), T(-q))
            M = matmul(M, T(-q))
        if c < a * (1 - tol):
            a, b, c = transform_any((a, b, c), S)
            M = matmul(M, S)
            continue
        break
    else:
        raise LatticeRefusal("REDUCTION_FAILED", "Float reduction did not terminate")
    if abs(2 * b + a) <= tol * a:
        a, b, c = transform_any((a, b, c), T(1))
        M = matmul(M, T(1))
    if abs(a - c) <= tol * a and b < 0:
        a, b, c = transform_any((a, b, c), S)
        M = matmul(M, S)
    return tau_of((a, b, c)), M


def mobius(M, tau: complex) -> complex:
    """Shape of the basis B M.

    For B' = B M with columns w1' = M00 w1 + M10 w2 and w2' = M01 w1 + M11 w2,
    tau' = (M01 + M11 tau) / (M00 + M10 tau).
    """
    (p, q), (r, s) = M
    return (q + s * tau) / (p + r * tau)


# ---------------------------------------------------------------- nearest translates
def nearest_translates(form, z, window: int = 3):
    """Exact nearest lattice translates of z (lattice coordinates, rational).

    Returns (minimum Q(z + lambda), list of minimizing lambda). The window is
    sufficient for a reduced form and z in the unit cell.
    """
    z = tuple(exact(x) for x in z)
    best, arg = None, []
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            d = (z[0] + m, z[1] + n)
            value = quad(form, d[0], d[1])
            if best is None or value < best:
                best, arg = value, [(m, n)]
            elif value == best:
                arg.append((m, n))
    return best, arg


def float_multiplicity(form, z, rel_tol: float = 0.0, window: int = 3):
    """Multiplicity of the shortest translate in binary64 with a declared relative tolerance."""
    a, b, c = (float(x) for x in form)
    zx, zy = float(z[0]), float(z[1])
    values = []
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            x, y = zx + m, zy + n
            values.append(a * x * x + 2 * b * x * y + c * y * y)
    low = min(values)
    return sum(1 for v in values if v <= low * (1 + rel_tol))


def cut_locus_vertices(form, span: int = 2):
    """Points of the torus with at least three nearest lattice translates (Voronoi vertices).

    Candidates are circumcenters of triangles (0, e, f) of small lattice vectors,
    solved exactly; each is reduced into the unit cell and kept when its exact
    multiplicity is at least three.
    """
    vectors = [(m, n) for m in range(-span, span + 1) for n in range(-span, span + 1) if (m, n) != (0, 0)]
    found = {}
    for i, e in enumerate(vectors):
        for f in vectors[i + 1:]:
            # 2 B(z, e) = Q(e), 2 B(z, f) = Q(f): a 2x2 linear system in z.
            a, b, c = form
            row_e = (2 * (a * e[0] + b * e[1]), 2 * (b * e[0] + c * e[1]))
            row_f = (2 * (a * f[0] + b * f[1]), 2 * (b * f[0] + c * f[1]))
            det = row_e[0] * row_f[1] - row_e[1] * row_f[0]
            if det == 0:
                continue
            rhs_e, rhs_f = quad(form, *e), quad(form, *f)
            zx = Fraction(rhs_e * row_f[1] - rhs_f * row_e[1]) / det
            zy = Fraction(row_e[0] * rhs_f - row_f[0] * rhs_e) / det
            cell = (zx - math.floor(zx), zy - math.floor(zy))
            if cell in found:
                continue
            _, arg = nearest_translates(form, cell)
            if len(arg) >= 3:
                found[cell] = len(arg)
    return sorted(found.items())


# ---------------------------------------------------------------- windings and flow
def classify_winding(m: int, n: int) -> dict:
    """Primitive classes are simple closed geodesics; (k m', k n') is a k-fold cover."""
    if m == 0 and n == 0:
        raise LatticeRefusal("ZERO_WINDING", "Winding (0, 0) is the constant loop, not a closed geodesic")
    g = math.gcd(m, n)
    return {"winding": [m, n], "gcd": g, "primitive": g == 1, "cover_degree": g,
            "primitive_class": [m // g, n // g]}


def trace_lattice_flow(m: int, n: int, start, max_time=Fraction(1), max_segments: int = 100000):
    """Exact straight-line flow on R^2 / Z^2 in lattice coordinates, walked segment by segment.

    The start (strictly inside the unit cell) moves with velocity (m, n); each
    segment runs to the next cell wall, where the coordinate wraps. A corner
    pass counts as two crossings (both walls). A return is observed when the
    start point lies on the current segment, found by an exact solve, so the
    return times are measured from the flow rather than assumed. Returns every
    return time in (0, max_time] and the crossings before the first return.
    """
    classify_winding(m, n)
    s = tuple(exact(x) for x in start)
    s = (s[0] - math.floor(s[0]), s[1] - math.floor(s[1]))
    if s[0] == 0 or s[1] == 0:
        raise LatticeRefusal("START_ON_CELL_WALL", "Flow start must lie strictly inside the unit cell")
    x, y = s
    t = Fraction(0)
    returns, events = [], []
    for _ in range(max_segments):
        if t >= max_time:
            break
        wall_a = (1 - x) / m if m > 0 else (x / -m if m < 0 else None)
        wall_b = (1 - y) / n if n > 0 else (y / -n if n < 0 else None)
        dt = min(w for w in (wall_a, wall_b) if w is not None)
        # Does the segment pass through the start point? Solve along a coordinate that moves.
        tau = (s[0] - x) / m if m else (s[1] - y) / n
        if 0 < tau <= dt and x + m * tau == s[0] and y + n * tau == s[1] and t + tau <= max_time:
            returns.append(t + tau)
        x, y, t = x + m * dt, y + n * dt, t + dt
        if wall_a == dt:
            events.append((t, "alpha"))
            x = Fraction(0) if m > 0 else Fraction(1)
        if wall_b == dt:
            events.append((t, "beta"))
            y = Fraction(0) if n > 0 else Fraction(1)
    else:
        raise LatticeRefusal("FLOW_SEGMENT_LIMIT", "Flow exceeded its segment limit before max_time")
    first = returns[0] if returns else None
    before = [e for e in events if first is not None and e[0] < first]
    return {"first_return": first, "return_times": returns, "returns": len(returns), "crossings": len(before),
            "alpha_crossings": sum(1 for e in before if e[1] == "alpha"),
            "beta_crossings": sum(1 for e in before if e[1] == "beta"),
            "displacement_at_first_return": None if first is None else (m * first, n * first)}


def intersection_count(v1, v2, p1=(Fraction(1, 7), Fraction(2, 11)), p2=(Fraction(3, 13), Fraction(5, 17))):
    """Exact count of transverse intersections of two closed geodesics (lattice coordinates).

    Counts the solutions of p1 + t v1 = p2 + s v2 + k with integer k and
    t, s in [0, 1): with r = p2 - p1 + k, t v1 - s v2 = r gives
    t = (r_x v2_y - r_y v2_x) / det and s = (r_x v1_y - r_y v1_x) / det.
    """
    solutions = intersection_solutions(v1, v2, p1, p2)
    return len(solutions)


def intersection_solutions(v1, v2, p1=(Fraction(1, 7), Fraction(2, 11)), p2=(Fraction(3, 13), Fraction(5, 17))):
    """The (t, s, k) solutions counted by :func:`intersection_count`."""
    det = v1[0] * v2[1] - v1[1] * v2[0]
    if det == 0:
        return []
    out = []
    span = abs(v1[0]) + abs(v1[1]) + abs(v2[0]) + abs(v2[1]) + 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    for kx in range(-span, span + 1):
        for ky in range(-span, span + 1):
            rx, ry = dx + kx, dy + ky
            t = Fraction(rx * v2[1] - ry * v2[0], det)
            s = Fraction(rx * v1[1] - ry * v1[0], det)
            if 0 <= t < 1 and 0 <= s < 1:
                out.append((t, s, (kx, ky)))
    return out


def box_lattice_count(form, radius_sq):
    """Brute-force count of (m, n) with Q(m, n) <= radius_sq over a box proven to contain them.

    Q = ((c n + b m)^2 + det m^2) / c >= det m^2 / c, and symmetrically
    Q >= det n^2 / a, so |m| <= sqrt(R^2 c / det) and |n| <= sqrt(R^2 a / det).
    Integer forms and radii only. Returns (all points including the origin,
    primitive points).
    """
    a, b, c = form
    d = det_form(form)
    if not all(isinstance(x, int) for x in (a, b, c, radius_sq)):
        raise LatticeRefusal("BOX_COUNT_NOT_INTEGER", "Box count needs an integer form and radius")
    mmax, nmax = math.isqrt(radius_sq * c // d), math.isqrt(radius_sq * a // d)
    total = primitive = 0
    for m in range(-mmax, mmax + 1):
        for n in range(-nmax, nmax + 1):
            if a * m * m + 2 * b * m * n + c * n * n <= radius_sq:
                total += 1
                primitive += math.gcd(m, n) == 1
    return total, primitive


def return_distance(tau: complex, headings, length: float, s_min: float = 0.05):
    """Distance of the flat geodesic segment {s u(theta): s_min <= s <= length} to the nearest nonzero lattice point.

    Vectorized over headings on the area-one lattice of ``tau``; float by design.
    Returns (distance, index of the minimizing lattice vector, lattice vectors).
    """
    form = area_one_form(tau)
    vectors = lattice_vectors(form, (length + 1.0) ** 2)
    w1 = complex(1 / math.sqrt(tau.imag), 0.0)
    w2 = tau / math.sqrt(tau.imag)
    points = np.array([m * w1 + n * w2 for _, m, n in vectors])
    ux, uy = np.cos(headings)[:, None], np.sin(headings)[:, None]
    px, py = points.real[None, :], points.imag[None, :]
    t = np.clip(px * ux + py * uy, s_min, length)
    dist = np.hypot(px - t * ux, py - t * uy)
    index = np.argmin(dist, axis=1)
    return dist[np.arange(len(headings)), index], index, vectors
