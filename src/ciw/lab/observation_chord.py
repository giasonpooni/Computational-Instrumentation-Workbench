"""Chord versus geodesic length along surface geodesics (T046, T047).

Scope: the closed-form chord expansion derived in docs/lab/OBSERVATION.md,
an optional sympy re-derivation used as an independent check, and the
numerical studies that compare both with geodesics integrated by
``ciw.lab.jacobi`` on ``ciw.lab.surfaces`` (sphere, torus, cylinder).

For a unit-speed curve with curvature kappa(s) and torsion tau(s) the chord
c(s) = |gamma(s) - gamma(0)| satisfies

    c = s - kappa0^2 s^3 / 24 - kappa0 kappa0' s^4 / 24 + c5 s^5 + O(s^6),

with c5 = kappa^4 / 1920 + kappa^2 tau^2 / 720 when kappa and tau are
constant. Along a surface geodesic the geodesic curvature vanishes, so kappa
is the absolute normal curvature II(T, T).

Non-claims: every surface here is a declared mathematical surface in
normalized units. Agreement between the expansion, sympy and integrated
geodesics says nothing about a measured part, whose shape, curvature and
marker placement are not modelled.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from . import jacobi
from .surfaces import Cylinder, Sphere, Torus

# Rational sample points (kappa0, kappa1, kappa2, tau0) for exact coefficient comparison.
SAMPLE_POINTS = ((Fraction(1, 2), Fraction(1, 3), Fraction(0), Fraction(1, 5)),
                 (Fraction(3), Fraction(-2), Fraction(5, 4), Fraction(7, 4)),
                 (Fraction(5, 7), Fraction(0), Fraction(0), Fraction(2)),
                 (Fraction(2, 3), Fraction(0), Fraction(0), Fraction(0)))


def closed_form_coefficients(kappa0, kappa1, tau0):
    """Hand-derived chord coefficients (c3, c4, c5 for constant kappa and tau)."""
    c3 = -kappa0 ** 2 / 24
    c4 = -kappa0 * kappa1 / 24
    c5_constant = kappa0 ** 4 / 1920 + kappa0 ** 2 * tau0 ** 2 / 720
    return c3, c4, c5_constant


def circle_chord(kappa: float, s):
    """Chord of a planar circle of curvature ``kappa``: 2 sin(kappa s / 2) / kappa."""
    s = np.asarray(s, dtype=float)
    return s.copy() if kappa == 0 else 2 * np.sin(kappa * s / 2) / kappa


def helix_chord(s, radius: float, alpha: float):
    """Exact chord of the cylinder geodesic at angle ``alpha`` from the circumferential direction."""
    s = np.asarray(s, dtype=float)
    theta = s * math.cos(alpha) / radius
    return np.sqrt((2 * radius * np.sin(theta / 2)) ** 2 + (s * math.sin(alpha)) ** 2)


def helix_curvature_torsion(radius: float, alpha: float) -> tuple[float, float]:
    """kappa = cos^2(alpha) / R and tau = sin(alpha) cos(alpha) / R for the cylinder geodesic."""
    return math.cos(alpha) ** 2 / radius, math.sin(alpha) * math.cos(alpha) / radius


# Independent symbolic derivation ------------------------------------------------

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
    for n in range(1, 7):
        for i in range(3):
            delta[i] += v[i].subs(s, 0) * s ** n / sp.factorial(n)
        v = tuple(sp.expand(x) for x in derivative(v))
    squared = sp.expand(sum(x ** 2 for x in delta))
    chord = sp.expand(sp.series(sp.sqrt(squared), s, 0, 6).removeO())
    return {n: sp.factor(chord.coeff(s, n)) for n in range(1, 6)}, (k0, k1, k2, t0)


def _fraction(value) -> Fraction:
    import sympy as sp

    rational = sp.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def sympy_versus_closed_form() -> dict:
    """Exact rational comparison of sympy's coefficients with the closed form."""
    import sympy as sp

    coefficients, (k0, k1, k2, t0) = sympy_general_series()
    rows, worst = [], Fraction(0)
    for point in SAMPLE_POINTS:
        kappa0, kappa1, kappa2, tau0 = point
        c3, c4, c5 = closed_form_coefficients(kappa0, kappa1, tau0)
        subs = {k0: sp.Rational(kappa0.numerator, kappa0.denominator),
                k1: sp.Rational(kappa1.numerator, kappa1.denominator),
                k2: sp.Rational(kappa2.numerator, kappa2.denominator),
                t0: sp.Rational(tau0.numerator, tau0.denominator)}
        symbolic = {n: _fraction(coefficients[n].subs(subs)) for n in (1, 2, 3, 4)}
        # The closed-form c5 is claimed only for constant curvature and torsion.
        constant = {k0: subs[k0], k1: 0, k2: 0, t0: subs[t0]}
        symbolic_c5 = _fraction(coefficients[5].subs(constant))
        differences = [symbolic[1] - 1, symbolic[2], symbolic[3] - c3, symbolic[4] - c4, symbolic_c5 - c5]
        worst = max([worst] + [abs(d) for d in differences])
        rows.append({"point": [str(x) for x in point], "sympy": {f"c{n}": str(symbolic[n]) for n in (3, 4)},
                     "sympy_c5_constant": str(symbolic_c5),
                     "closed_form": {"c3": str(c3), "c4": str(c4), "c5_constant": str(c5)}})
    return {"expressions": {f"c{n}": str(coefficients[n]) for n in range(1, 6)}, "points": rows,
            "max_abs_difference": float(worst), "sympy": sp.__version__}


def sympy_cylinder_series(angles_deg) -> dict:
    """sympy expansion of the exact helix chord against the closed-form coefficients."""
    import sympy as sp

    s, radius = sp.symbols("s R", positive=True)
    alpha = sp.Symbol("alpha", real=True)
    chord = sp.sqrt((2 * radius * sp.sin(s * sp.cos(alpha) / (2 * radius))) ** 2 + (s * sp.sin(alpha)) ** 2)
    series = sp.series(chord, s, 0, 6).removeO()
    c3, c5 = sp.simplify(series.coeff(s, 3)), sp.simplify(series.coeff(s, 5))
    kappa, tau = sp.cos(alpha) ** 2 / radius, sp.sin(alpha) * sp.cos(alpha) / radius
    symbolic_residual = [sp.simplify(c3 + kappa ** 2 / 24),
                         sp.simplify(c5 - (kappa ** 4 / 1920 + kappa ** 2 * tau ** 2 / 720))]
    rows, worst = [], 0.0
    for degrees in angles_deg:
        a = math.radians(degrees)
        k, t = helix_curvature_torsion(1.0, a)
        closed = closed_form_coefficients(k, 0.0, t)
        exact_alpha = sp.pi * sp.Rational(degrees, 180)
        values = [float(c.subs({alpha: exact_alpha, radius: 1}).evalf(30)) for c in (c3, c5)]
        difference = max(abs(values[0] - closed[0]), abs(values[1] - closed[2]))
        worst = max(worst, difference)
        rows.append({"alpha_deg": degrees, "sympy_c3": values[0], "sympy_c5": values[1],
                     "closed_c3": closed[0], "closed_c5": closed[2], "abs_difference": difference})
    return {"c3": str(c3), "c5": str(c5), "symbolic_residuals": [str(r) for r in symbolic_residual],
            "symbolic_zero": all(r == 0 for r in symbolic_residual), "angles": rows,
            "max_abs_difference": worst, "sympy": sp.__version__}


# Integrated geodesics ------------------------------------------------------------

def _embedded_chords(surface, transfer):
    points = np.array([surface.embedding(u) for u in transfer.points])
    return np.linalg.norm(points - points[0], axis=1)


def _fit(x, y, degree):
    design = np.vander(np.asarray(x, dtype=float), degree + 1, increasing=True)
    return np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)[0]


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
        predicted = 1.0 / (24 * radius ** 2)
        rows.append({"radius": radius, "kappa": 1.0 / radius, "steps": steps,
                     "max_chord_error": float(np.max(np.abs(chord - exact))),
                     "relative_chord_error": float(np.max(np.abs(chord - exact)) / radius),
                     "fitted_c3": float(-coefficients[0]), "predicted_c3": -predicted,
                     "c3_relative_error": float(abs(coefficients[0] / predicted - 1)),
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
    predicted = kappa0 * kappa1 / 24
    selected = s >= low
    start_residual = s - chord - kappa0 ** 2 * s ** 3 / 24
    start_fit = _fit(s[selected], start_residual[selected] / s[selected] ** 4, degree)
    # Curvature at the arc midpoint cancels the s^4 term by symmetry.
    even = np.arange(2, steps + 1, 2)
    s_even, kappa_mid = s[even], kappa[even // 2]
    mid_residual = s_even - chord[even] - kappa_mid ** 2 * s_even ** 3 / 24
    keep = s_even >= low
    mid_fit = _fit(s_even[keep], mid_residual[keep] / s_even[keep] ** 4, degree)
    slope_start = float(np.polyfit(np.log(s_even[keep]), np.log(np.abs(start_residual[even][keep])), 1)[0])
    slope_mid = float(np.polyfit(np.log(s_even[keep]), np.log(np.abs(mid_residual[keep])), 1)[0])
    return {"surface": torus.describe(), "u0": u0.tolist(), "heading_rad": heading, "length": length,
            "steps": steps, "kappa0": kappa0, "kappa0_prime": kappa1,
            "predicted_c4_residual": predicted, "fitted_start_s4": float(start_fit[0]),
            "start_relative_error": float(abs(start_fit[0] / predicted - 1)),
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
    predicted = kappa ** 2 * tau ** 2 / 720
    return {"radius": radius, "alpha_deg": alpha_deg, "kappa": kappa, "tau": tau, "length": length,
            "max_gap": float(np.max(np.abs(gap))), "predicted_s5_gap": predicted, "fitted_s5_gap": float(fit[0]),
            "relative_error": float(abs(fit[0] / predicted - 1)),
            "max_exact_chord_error": float(np.max(np.abs(chord - helix_chord(s, radius, alpha))))}


def cylinder_fit(angles_deg, radius: float, low=0.02, high=0.3, points=40) -> list:
    """Fit (s - c) / s^3 = a + b s^2 + d s^4 to exact helix chords at small s."""
    rows = []
    for degrees in angles_deg:
        alpha = math.radians(degrees)
        s = np.geomspace(low, high, points) * radius
        chord = helix_chord(s, radius, alpha)
        coefficients = _fit(s ** 2, (s - chord) / s ** 3, 2)
        kappa, _ = helix_curvature_torsion(radius, alpha)
        predicted = kappa ** 2 / 24
        rows.append({"alpha_deg": degrees, "radius": radius, "fitted": float(coefficients[0]),
                     "predicted": predicted, "abs_error": float(abs(coefficients[0] - predicted)),
                     "normalized_error": float(abs(coefficients[0] - predicted) * 24 * radius ** 2)})
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
        kappa, _ = helix_curvature_torsion(radius, alpha)
        rows.append({"alpha_deg": degrees, "radius": radius,
                     "max_chord_error": float(np.max(np.abs(chord - helix_chord(s, radius, alpha)))),
                     "max_s_minus_c": float(np.max(s - chord)), "fitted": float(coefficients[0]),
                     "predicted": kappa ** 2 / 24,
                     "normalized_error": float(abs(coefficients[0] - kappa ** 2 / 24) * 24 * radius ** 2)})
    return rows
