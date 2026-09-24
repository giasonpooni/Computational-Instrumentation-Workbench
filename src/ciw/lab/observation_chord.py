"""Chord versus geodesic length along surface geodesics (T046, T047).

Scope: the closed-form chord expansion derived in docs/lab/OBSERVATION.md,
two optional sympy references (a symbolic expansion of the Frenet-Serret
Taylor recursion, and an exact rational expansion of explicit polynomial
space curves that does not use the Frenet recursion), and the numerical
studies that compare the expansion with geodesics integrated by
``ciw.lab.jacobi`` on ``ciw.lab.surfaces`` (sphere, torus, cylinder). The
curve models of the sympy references are written in this module; sympy
performs their series expansion, simplification and exact arithmetic.

For a unit-speed curve with curvature kappa(s) and torsion tau(s) the chord
c(s) = |gamma(s) - gamma(0)| satisfies

    c = s - kappa0^2 s^3 / 24 - kappa0 kappa0' s^4 / 24 + c5 s^5 + O(s^6),
    c5 = (3 kappa0^4 + 8 kappa0^2 tau0^2 - 72 kappa0 kappa0'' - 64 kappa0'^2) / 5760,

which is kappa^4 / 1920 + kappa^2 tau^2 / 720 when kappa and tau are
constant. Along a surface geodesic the geodesic curvature vanishes, so kappa
is the absolute normal curvature II(T, T).

Non-claims: every surface here is a declared mathematical surface in
normalized units. Agreement between the expansion, sympy and integrated
geodesics says nothing about a measured part, whose shape, curvature and
marker placement are not modelled.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from . import jacobi
from .surfaces import Cylinder, Sphere, Torus


def closed_form_coefficients(kappa0, kappa1, tau0, kappa2=0):
    """Hand-derived chord coefficients c3, c4, c5 (docs/lab/OBSERVATION.md).

    ``kappa1`` and ``kappa2`` are kappa'(0) and kappa''(0). The arguments may
    be numbers or sympy symbols, so the same expressions feed the numerical
    studies and the symbolic check.
    """
    c3 = -kappa0 ** 2 / 24
    c4 = -kappa0 * kappa1 / 24
    c5 = (3 * kappa0 ** 4 + 8 * kappa0 ** 2 * tau0 ** 2 - 72 * kappa0 * kappa2 - 64 * kappa1 ** 2) / 5760
    return c3, c4, c5


def circle_chord(kappa: float, s):
    """Chord of a planar circle of curvature ``kappa``: 2 sin(kappa s / 2) / kappa."""
    s = np.asarray(s, dtype=float)
    return s.copy() if kappa == 0 else 2 * np.sin(kappa * s / 2) / kappa


def helix_chord(s, radius: float, alpha: float):
    """Exact chord of the cylinder geodesic at angle ``alpha`` from the circumferential direction."""
    s = np.asarray(s, dtype=float)
    theta = s * math.cos(alpha) / radius
    return np.sqrt((2 * radius * np.sin(theta / 2)) ** 2 + (s * math.sin(alpha)) ** 2)


def arc_from_chord(chords, radius: float, alpha: float, iterations: int = 80) -> np.ndarray:
    """Vectorized inverse of :func:`helix_chord` on its monotone branch s cos(alpha)/R <= pi."""
    chords = np.asarray(chords, dtype=float)
    if abs(math.cos(alpha)) < 1e-15:
        return chords.copy()
    low, high = chords.copy(), np.full_like(chords, math.pi * radius / abs(math.cos(alpha)))
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        below = helix_chord(middle, radius, alpha) < chords
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    return 0.5 * (low + high)


def embedded_helix_chords(s_values, radius: float, alpha: float, u0=(0.3, -0.2)) -> np.ndarray:
    """Chords of embedded points of ``Cylinder.exact_geodesic``, a forward model independent of :func:`helix_chord`."""
    cylinder = Cylinder(radius)
    start = np.asarray(u0, dtype=float)
    chart = cylinder.exact_geodesic(start, cylinder.unit_tangent(start, alpha),
                                    np.concatenate([[0.0], np.asarray(s_values, dtype=float)]))
    points = np.array([cylinder.embedding(u) for u in chart])
    return np.linalg.norm(points[1:] - points[0], axis=1)


def helix_curvature_torsion(radius: float, alpha: float) -> tuple[float, float]:
    """kappa = cos^2(alpha) / R and tau = sin(alpha) cos(alpha) / R for the cylinder geodesic."""
    return math.cos(alpha) ** 2 / radius, math.sin(alpha) * math.cos(alpha) / radius


# Independent symbolic derivation ------------------------------------------------

@lru_cache(maxsize=1)
def sympy_general_series():
    """Chord series of a Frenet curve with polynomial kappa(s), tau(s), derived by sympy.

    Returns the sympy coefficient expressions of s^1..s^5 in the symbols
    kappa0, kappa1, kappa2, tau0 (tau1, kappa3 do not reach order five).
    """
    import sympy as sp

    s = sp.Symbol("s", positive=True)
    k0, k1, k2, k3, t0, t1 = sp.symbols("kappa0 kappa1 kappa2 kappa3 tau0 tau1", real=True)
    kappa = k0 + k1 * s + k2 * s ** 2 / 2 + k3 * s ** 3 / 6
    tau = t0 + t1 * s

    def derivative(v):
        # Frenet-Serret: T' = kappa N, N' = -kappa T + tau B, B' = -tau N.
        a, b, c = v
        return (sp.diff(a, s) - kappa * b, sp.diff(b, s) + kappa * a - tau * c, sp.diff(c, s) + tau * b)

    v = (sp.Integer(1), sp.Integer(0), sp.Integer(0))
    delta = [sp.Integer(0)] * 3
    # Orders one to five of the Taylor expansion fix |delta|^2 through s^6 and c through s^5.
    for n in range(1, 6):
        for i in range(3):
            delta[i] += v[i].subs(s, 0) * s ** n / sp.factorial(n)
        v = tuple(sp.expand(x) for x in derivative(v))
    squared = sp.expand(sum(x ** 2 for x in delta))
    chord = sp.expand(sp.series(sp.sqrt(squared), s, 0, 6).removeO())
    return {n: sp.factor(chord.coeff(s, n)) for n in range(1, 6)}, (k0, k1, k2, t0)


def sympy_versus_closed_form() -> dict:
    """Symbolic identity check of sympy's coefficients against the closed form.

    Each residual sympy(c_n) - closed(c_n) is simplified as a polynomial in
    (kappa0, kappa0', kappa0'', tau0); the check passes only if every residual
    is identically zero, not merely zero at sample points.
    """
    import sympy as sp

    coefficients, (k0, k1, k2, t0) = sympy_general_series()
    c3, c4, c5 = closed_form_coefficients(k0, k1, t0, k2)
    closed = {1: sp.Integer(1), 2: sp.Integer(0), 3: sp.sympify(c3), 4: sp.sympify(c4), 5: sp.sympify(c5)}
    residuals = {n: sp.simplify(sp.expand(coefficients[n] - closed[n])) for n in range(1, 6)}
    # The constant-curvature, constant-torsion specialisation quoted for the circle and helix.
    constant = sp.simplify(coefficients[5].subs({k1: 0, k2: 0}) - (k0 ** 4 / 1920 + k0 ** 2 * t0 ** 2 / 720))
    nonzero = sum(residual != 0 for residual in residuals.values()) + int(constant != 0)
    return {"expressions": {f"c{n}": str(coefficients[n]) for n in range(1, 6)},
            "closed_form": {f"c{n}": str(closed[n]) for n in range(1, 6)},
            "residuals": {f"c{n}": str(residuals[n]) for n in range(1, 6)},
            "constant_curvature_c5_residual": str(constant), "nonzero_residuals": int(nonzero),
            "sympy": sp.__version__}


# Explicit polynomial space curves g(t) = (t, a2 t^2 + a3 t^3 + a4 t^4, b3 t^3 + b4 t^4) with exact
# rational coefficients (a2, a3, a4, b3, b4); together they give generic kappa0, kappa0', kappa0'', tau0.
EXPLICIT_CURVES = (("1/2", "1/3", "-1/5", "2/7", "3/11"), ("3/4", "-2/9", "1/6", "-5/8", "1/13"),
                   ("-2/5", "1/7", "3/10", "4/9", "-1/3"))


def sympy_explicit_curve_check(curves=EXPLICIT_CURVES, order: int = 7) -> dict:
    """Chord coefficients of explicit polynomial space curves in exact rational arithmetic (no Frenet recursion).

    For each curve g(t), sympy expands the arc length s(t) = int |g'| dt,
    reverts it to t(s), expands the chord |g(t(s)) - g(0)| in s, and expands
    the curvature |g' x g''| / |g'|^3 and torsion (g' x g'') . g''' / |g' x g''|^2
    to obtain kappa0, kappa0', kappa0'' and tau0 with respect to arc length.
    The closed-form c1..c5 at those values must equal the chord coefficients
    exactly (rational residual 0). This route shares neither the Frenet-Serret
    Taylor recursion of :func:`sympy_general_series` nor the hand derivation;
    the curve model is written here and sympy performs the series and the
    exact arithmetic.
    """
    import sympy as sp

    t, s = sp.symbols("t s")

    def truncated(expr, var, n=order):
        poly = sp.Poly(sp.expand(expr), var)
        return sum((c * var ** m[0] for m, c in zip(poly.monoms(), poly.coeffs()) if m[0] < n), sp.Integer(0))

    rows, nonzero = [], 0
    for coefficients in curves:
        a2, a3, a4, b3, b4 = (sp.Rational(value) for value in coefficients)
        g = sp.Matrix([t, a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4, b3 * t ** 3 + b4 * t ** 4])
        d1 = g.diff(t)
        d2, d3 = d1.diff(t), d1.diff(t, 2)
        arc = sp.integrate(sp.series(sp.sqrt(d1.dot(d1)), t, 0, order).removeO(), t)
        inverse = s
        for _ in range(order):  # fixed-point series reversion of s = arc(t)
            inverse = truncated(s - (arc.subs(t, inverse) - inverse), s)
        chord = sp.series(sp.sqrt(sp.expand(g.dot(g))), t, 0, order).removeO()
        chord_s = truncated(chord.subs(t, inverse), s, 6)
        cross = d1.cross(d2)
        kappa = sp.series(sp.sqrt(sp.expand(cross.dot(cross))) / sp.sqrt(sp.expand(d1.dot(d1))) ** 3,
                          t, 0, 4).removeO()
        tau = sp.series(sp.expand(cross.dot(d3)) / sp.expand(cross.dot(cross)), t, 0, 2).removeO()
        kappa_s = truncated(kappa.subs(t, inverse), s, 3)
        k0, k1, k2 = (sp.diff(kappa_s, s, n).subs(s, 0) for n in range(3))
        tau0 = truncated(tau.subs(t, inverse), s, 1).subs(s, 0)
        c3, c4, c5 = closed_form_coefficients(k0, k1, tau0, k2)
        closed = {1: sp.Integer(1), 2: sp.Integer(0), 3: c3, 4: c4, 5: c5}
        residuals = {n: sp.nsimplify(chord_s.coeff(s, n) - closed[n]) for n in range(1, 6)}
        nonzero += sum(residual != 0 for residual in residuals.values())
        rows.append({"a2_a3_a4_b3_b4": list(coefficients), "kappa0": str(k0), "kappa0_prime": str(k1),
                     "kappa0_second": str(k2), "tau0": str(tau0),
                     "chord_coefficients": {f"c{n}": str(chord_s.coeff(s, n)) for n in range(1, 6)},
                     "residuals": {f"c{n}": str(residuals[n]) for n in range(1, 6)}})
    return {"curves": rows, "residuals_checked": 5 * len(curves), "nonzero_residuals": int(nonzero),
            "sympy": sp.__version__}


def sympy_cylinder_series() -> dict:
    """sympy expansion of the exact helix chord against the closed-form coefficients (symbolic in R, alpha)."""
    import sympy as sp

    s, radius = sp.symbols("s R", positive=True)
    alpha = sp.Symbol("alpha", real=True)
    chord = sp.sqrt((2 * radius * sp.sin(s * sp.cos(alpha) / (2 * radius))) ** 2 + (s * sp.sin(alpha)) ** 2)
    series = sp.series(chord, s, 0, 6).removeO()
    c3, c5 = sp.simplify(series.coeff(s, 3)), sp.simplify(series.coeff(s, 5))
    kappa, tau = sp.cos(alpha) ** 2 / radius, sp.sin(alpha) * sp.cos(alpha) / radius
    closed3, _, closed5 = closed_form_coefficients(kappa, 0, tau)
    symbolic_residual = [sp.simplify(c3 - closed3), sp.simplify(c5 - closed5)]
    return {"c3": str(c3), "c5": str(c5), "closed_c3": str(sp.simplify(closed3)),
            "closed_c5": str(sp.simplify(closed5)), "symbolic_residuals": [str(r) for r in symbolic_residual],
            "nonzero_residuals": int(sum(r != 0 for r in symbolic_residual)), "sympy": sp.__version__}


# Integrated geodesics ------------------------------------------------------------

def _embedded_chords(surface, transfer):
    points = np.array([surface.embedding(u) for u in transfer.points])
    return np.linalg.norm(points - points[0], axis=1)


def _fit(x, y, degree):
    design = np.vander(np.asarray(x, dtype=float), degree + 1, increasing=True)
    return np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]


def _intercept_with_truncation(x, y, degree) -> tuple[float, float]:
    """Fitted intercept and its change when one more polynomial term is fitted (truncation estimate)."""
    intercept = float(_fit(x, y, degree)[0])
    return intercept, abs(float(_fit(x, y, degree + 1)[0]) - intercept)


def sphere_study(radii=(0.5, 1.0, 2.0), steps=400) -> dict:
    """Great circles integrated with RK4: chords against 2R sin(s/2R) and the s^3 coefficient."""
    rows = []
    for radius in radii:
        sphere = Sphere(radius)
        transfer = jacobi.transfer(sphere, (math.pi / 2 - 0.3, 0.2), 0.7, radius, steps=steps)
        s, chord = transfer.s, _embedded_chords(sphere, transfer)
        exact = circle_chord(1.0 / radius, s)
        small = (s > 0) & (s <= 0.25 * radius)
        # (s - c) / s^3 = kappa^2/24 - kappa^4 s^2/1920 + ...
        coefficients = _fit(s[small] ** 2, (s[small] - chord[small]) / s[small] ** 3, 2)
        _, truncation = _intercept_with_truncation(s[small] ** 2, (s[small] - chord[small]) / s[small] ** 3, 2)
        predicted = 1.0 / (24 * radius ** 2)
        rows.append({"radius": radius, "kappa": 1.0 / radius, "steps": steps,
                     "max_chord_error": float(np.max(np.abs(chord - exact))),
                     "relative_chord_error": float(np.max(np.abs(chord - exact)) / radius),
                     "fitted_c3": float(-coefficients[0]), "predicted_c3": -predicted,
                     "c3_relative_error": float(abs(coefficients[0] / predicted - 1)),
                     "c3_relative_fit_truncation": float(truncation / predicted),
                     "fitted_c5": float(-coefficients[1]), "predicted_c5": 1.0 / (1920 * radius ** 4),
                     "speed_drift": float(transfer.speed_drift().max()),
                     "curve": {"s": s[small][::4].tolist(), "s_minus_c": (s - chord)[small][::4].tolist()}})
    return {"rows": rows}


def normal_curvature(surface, state) -> float:
    """II(v, v) with exact second embedding derivatives; |II| is the space curvature of a geodesic."""
    u, v = state[:2], state[2:4]
    xuu, xuv, xvv = surface.second(u)
    n = surface.unit_normal3(u)
    return float((xuu @ n) * v[0] ** 2 + 2 * (xuv @ n) * v[0] * v[1] + (xvv @ n) * v[1] ** 2)


def torus_counterexample(length=0.4, steps=800, low=0.02, degree=4) -> dict:
    """A torus geodesic with varying curvature: the start-point expansion has an s^4 term."""
    torus = Torus(2.0, 1.0)
    u0, heading = np.array([0.0, math.pi / 4]), 0.6
    transfer = jacobi.transfer(torus, u0, heading, length, steps=steps)
    s, chord = transfer.s, _embedded_chords(torus, transfer)
    kappa = np.abs([normal_curvature(torus, y) for y in transfer.states])
    poly = np.polynomial.polynomial.polyfit(s, kappa, 8)
    kappa0, kappa1 = float(kappa[0]), float(poly[1])
    kappa1_truncation = abs(float(np.polynomial.polynomial.polyfit(s, kappa, 9)[1]) - kappa1)
    predicted = kappa0 * kappa1 / 24
    selected = s >= low
    start_residual = s - chord - kappa0 ** 2 * s ** 3 / 24
    start_fit = _fit(s[selected], start_residual[selected] / s[selected] ** 4, degree)
    _, start_truncation = _intercept_with_truncation(s[selected], start_residual[selected] / s[selected] ** 4, degree)
    # Curvature at the arc midpoint cancels the s^4 term by symmetry.
    even = np.arange(2, steps + 1, 2)
    s_even, kappa_mid = s[even], kappa[even // 2]
    mid_residual = s_even - chord[even] - kappa_mid ** 2 * s_even ** 3 / 24
    keep = s_even >= low
    mid_fit = _fit(s_even[keep], mid_residual[keep] / s_even[keep] ** 4, degree)
    _, mid_truncation = _intercept_with_truncation(s_even[keep], mid_residual[keep] / s_even[keep] ** 4, degree)
    slope_start = float(np.polyfit(np.log(s_even[keep]), np.log(np.abs(start_residual[even][keep])), 1)[0])
    slope_mid = float(np.polyfit(np.log(s_even[keep]), np.log(np.abs(mid_residual[keep])), 1)[0])
    return {"surface": torus.describe(), "u0": u0.tolist(), "heading_rad": heading, "length": length,
            "steps": steps, "kappa0": kappa0, "kappa0_prime": kappa1,
            "predicted_c4_residual": predicted, "fitted_start_s4": float(start_fit[0]),
            "start_relative_error": float(abs(start_fit[0] / predicted - 1)),
            "start_s4_fit_truncation": start_truncation, "kappa0_prime_fit_truncation": kappa1_truncation,
            "midpoint_ratio_fit_truncation": float(mid_truncation / abs(predicted)),
            "fitted_midpoint_s4": float(mid_fit[0]),
            "midpoint_ratio": float(abs(mid_fit[0] / predicted)),
            "loglog_slope_start": slope_start, "loglog_slope_midpoint": slope_mid,
            "speed_drift": float(transfer.speed_drift().max()),
            "curve": {"s": s_even[keep][::8].tolist(), "start": np.abs(start_residual[even][keep])[::8].tolist(),
                      "midpoint": np.abs(mid_residual[keep])[::8].tolist()}}


def helix_torsion_counterexample(radius=1.0, alpha_deg=45.0, length=0.8, steps=160) -> dict:
    """Constant curvature with torsion: the circle chord 2 sin(kappa s/2)/kappa is not exact."""
    cylinder = Cylinder(radius)
    alpha = math.radians(alpha_deg)
    transfer = jacobi.transfer(cylinder, (0.3, -0.2), alpha, length, steps=steps)
    s, chord = transfer.s, _embedded_chords(cylinder, transfer)
    kappa, tau = helix_curvature_torsion(radius, alpha)
    gap = chord - circle_chord(kappa, s)
    selected = s >= 0.1 * length
    fit = _fit(s[selected] ** 2, gap[selected] / s[selected] ** 5, 2)
    _, truncation = _intercept_with_truncation(s[selected] ** 2, gap[selected] / s[selected] ** 5, 2)
    predicted = kappa ** 2 * tau ** 2 / 720
    return {"radius": radius, "alpha_deg": alpha_deg, "kappa": kappa, "tau": tau, "length": length,
            "max_gap": float(np.max(np.abs(gap))), "predicted_s5_gap": predicted, "fitted_s5_gap": float(fit[0]),
            "relative_error": float(abs(fit[0] / predicted - 1)), "s5_fit_truncation": truncation,
            "max_exact_chord_error": float(np.max(np.abs(chord - helix_chord(s, radius, alpha))))}


def cylinder_fit(angles_deg, radius: float, low=0.02, high=0.3, points=40) -> list:
    """Fit (s - c) / s^3 = a + b s^2 + d s^4 to exact helix chords at small s."""
    rows = []
    for degrees in angles_deg:
        alpha = math.radians(degrees)
        s = np.geomspace(low, high, points) * radius
        chord = helix_chord(s, radius, alpha)
        coefficients = _fit(s ** 2, (s - chord) / s ** 3, 2)
        _, truncation = _intercept_with_truncation(s ** 2, (s - chord) / s ** 3, 2)
        kappa, _ = helix_curvature_torsion(radius, alpha)
        predicted = kappa ** 2 / 24
        rows.append({"alpha_deg": degrees, "radius": radius, "fitted": float(coefficients[0]),
                     "predicted": predicted, "abs_error": float(abs(coefficients[0] - predicted)),
                     "normalized_error": float(abs(coefficients[0] - predicted) * 24 * radius ** 2),
                     "normalized_fit_truncation": float(truncation * 24 * radius ** 2)})
    return rows


def cylinder_integrated(angles_deg, radius: float, length_factor=0.3, steps=60) -> list:
    """Geodesics integrated on ciw.lab.surfaces.Cylinder against the exact helix chord."""
    cylinder = Cylinder(radius)
    rows = []
    for degrees in angles_deg:
        alpha = math.radians(degrees)
        transfer = jacobi.transfer(cylinder, (0.3, -0.2), alpha, length_factor * radius, steps=steps)
        s, chord = transfer.s, _embedded_chords(cylinder, transfer)
        selected = s > 0
        coefficients = _fit(s[selected] ** 2, (s[selected] - chord[selected]) / s[selected] ** 3, 2)
        _, truncation = _intercept_with_truncation(s[selected] ** 2, (s[selected] - chord[selected]) / s[selected] ** 3, 2)
        kappa, _ = helix_curvature_torsion(radius, alpha)
        rows.append({"alpha_deg": degrees, "radius": radius,
                     "max_chord_error": float(np.max(np.abs(chord - helix_chord(s, radius, alpha)))),
                     # Over s > 0 only: s = 0 gives s - c = 0 exactly and would anchor a one-sided maximum.
                     "max_abs_s_minus_c": float(np.max(np.abs(s[selected] - chord[selected]))),
                     "fitted": float(coefficients[0]),
                     "predicted": kappa ** 2 / 24,
                     "normalized_error": float(abs(coefficients[0] - kappa ** 2 / 24) * 24 * radius ** 2),
                     "normalized_fit_truncation": float(truncation * 24 * radius ** 2)})
    return rows
