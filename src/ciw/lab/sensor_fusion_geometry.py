"""Covariance through frame transforms and Jacobi transfer matrices (T063, T064).

Scope: T063 propagates a declared body-frame state covariance through
rotation, translation, a metre-to-millimetre unit change and their composite
as P' = J P J^T, verifies it by Monte Carlo, and checks Mahalanobis-distance
invariance. T064 propagates a declared [lateral, heading] covariance along
geodesics of the unit sphere and the hyperbolic plane with the Jacobi
transfer matrix Phi from :mod:`ciw.lab.jacobi`, and compares Phi P Phi^T with
Monte Carlo over perturbed geodesics (:func:`ciw.lab.jacobi.perturbed_start`
integrated by :mod:`ciw.lab.integrators`), including the lateral-variance
collapse at the sphere's conjugate point and the breakdown of the first-order
propagation as the perturbation grows.

Non-claims: the covariances, transforms and surfaces are declared synthetic
objects in normalized units. Nothing here establishes a real sensor mounting,
extrinsic calibration or the uncertainty of a physical path on a physical
surface.
"""
from __future__ import annotations

import math

import numpy as np

from . import svg
from .evidence import finding
from .integrators import integrate_fixed
from .jacobi import constant_curvature, perturbed_start, transfer
from .registry import task
from .sensor_fusion_bench import chi2_quantile, factor, generator
from .sensor_fusion_common import (CORE_GEOMETRY, TESTS, TOL_MC, TOL_ROUNDOFF, TOL_TINY, as_json, bonferroni,
                                   check, covariance_z, files, generator_basis, gram, identity, max_abs_z_spread,
                                   mc95, outcome, rate_interval, roundoff, uncertainty, unreal)
from .sensor_fusion_objects import FrameTransform, Observation
from .surfaces import HyperbolicPlane, Sphere

T063_SEED = 63_2026
T064_SEED = 64_2026


# T063 ------------------------------------------------------------------------------
def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_covariance() -> np.ndarray:
    """Declared body-frame covariance of (x, y, vx, vy): elongated along-track, correlated."""
    std = np.array([0.30, 0.05, 0.10, 0.06])
    correlation = np.array([[1.0, 0.3, 0.2, 0.0], [0.3, 1.0, 0.0, 0.25],
                            [0.2, 0.0, 1.0, 0.4], [0.0, 0.25, 0.4, 1.0]])
    return correlation * np.outer(std, std)


def frame_transforms(theta: float, translation, scale: float) -> dict:
    """Affine maps y = J x + b of the planar state; translation moves positions only."""
    rot = np.kron(np.eye(2), rotation(theta))
    shift = np.concatenate([np.asarray(translation, dtype=float), [0.0, 0.0]])
    return {"rotation": (rot, np.zeros(4)), "translation": (np.eye(4), shift),
            "unit_m_to_mm": (scale * np.eye(4), np.zeros(4)),
            "composite": (scale * rot, scale * shift)}


def mahalanobis(vectors, cov) -> np.ndarray:
    L = np.linalg.cholesky(cov)
    white = np.linalg.solve(L, np.asarray(vectors, dtype=float).T)
    return np.sum(white ** 2, axis=0)


def frame_study(seed: int = T063_SEED, samples: int = 100_000) -> dict:
    theta, translation, scale = math.radians(35.0), (12.5, -4.0), 1000.0
    mean = np.array([2.0, 0.5, 1.0, 0.2])
    P = body_covariance()
    rng = generator(seed)
    x = mean + rng.standard_normal((samples, 4)) @ factor(P).T
    maps = frame_transforms(theta, translation, scale)
    per_map, z_all, exact, roundtrip, asymmetry = {}, [], 0.0, 0.0, 0.0
    d_body = mahalanobis(x - mean, P)
    invariance = {}
    S_x = gram(x - mean) / samples
    # Expected transformed means from the declared parameters, not from b: the translation moves positions
    # only, so a translation leaking into the velocity block shifts the sample mean of the velocities.
    R4 = np.kron(np.eye(2), rotation(theta))
    shift = np.concatenate([np.asarray(translation, dtype=float), np.zeros(2)])
    expected = {"rotation": R4 @ mean, "translation": mean + shift, "unit_m_to_mm": scale * mean,
                "composite": scale * (R4 @ mean + shift)}
    mean_z = []
    for name, (J, b) in maps.items():
        y = x @ J.T + b
        mu_y = J @ mean + b
        P_y = J @ P @ J.T
        asymmetry = max(asymmetry, float(np.max(np.abs(P_y - P_y.T)) / np.max(np.abs(P_y))))
        mean_z.append((y.mean(axis=0) - expected[name]) / np.sqrt(np.diag(P_y) / samples))
        S_y, z = covariance_z(y - mu_y, P_y)
        per_map[name] = {"max_abs_z": float(np.max(np.abs(z))), "predicted": P_y, "sample": S_y}
        z_all.append(z)
        # Linear maps carry the sample covariance exactly; only roundoff separates the two sides.
        exact = max(exact, float(np.max(np.abs(S_y - J @ S_x @ J.T)) / np.max(np.abs(P_y))))
        J_inv = np.linalg.inv(J)
        roundtrip = max(roundtrip, float(np.max(np.abs(J_inv @ P_y @ J_inv.T - P)) / np.max(np.abs(P))))
        d_y = mahalanobis(y - mu_y, P_y)
        invariance[name] = float(np.max(np.abs(d_y - d_body) / d_body))
    z_all = np.concatenate(z_all)
    z_crit = bonferroni(len(z_all))
    mean_z = np.concatenate(mean_z)

    # Counterexamples: transform the vector but not its covariance.
    pos = P[:2, :2]
    R2 = rotation(theta)
    e = (x - mean)[:, :2]
    rotated_only = mahalanobis(e @ R2.T, pos)
    unit_only = mahalanobis(scale * e, pos)
    d_pos = mahalanobis(e, pos)
    predicted_rotated = float(np.trace(np.linalg.solve(pos, R2 @ pos @ R2.T)))
    gate = chi2_quantile(0.99, 2)
    rejected = int(np.count_nonzero(rotated_only > gate))
    rejected_correct = int(np.count_nonzero(mahalanobis(e @ R2.T, R2 @ pos @ R2.T) > gate))
    var_rotated = float(2 * np.trace(np.linalg.matrix_power(np.linalg.solve(pos, R2 @ pos @ R2.T), 2)))

    # API path: FrameTransform.apply on a typed position observation.
    obs = Observation("tracker", "body", 1, tuple(mean[:2]), tuple(map(tuple, pos)), "synthetic-cal")
    moved = FrameTransform("body", "world", tuple(map(tuple, R2)), translation).apply(obs)
    api = max(float(np.max(np.abs(np.asarray(moved.covariance) - R2 @ pos @ R2.T))),
              float(np.max(np.abs(np.asarray(moved.value) - (R2 @ mean[:2] + translation)))))
    moved_cov = np.asarray(moved.covariance)
    asymmetry = max(asymmetry, float(np.max(np.abs(moved_cov - moved_cov.T)) / np.max(np.abs(moved_cov))))
    return {"seed": seed, "samples": samples, "theta_deg": 35.0, "translation_m": list(translation),
            "scale": scale, "mean": mean, "P_body": P, "per_map": per_map, "max_abs_z": float(np.max(np.abs(z_all))),
            "z_critical": z_crit, "moments_tested": int(len(z_all)), "exact_transport": exact,
            "roundtrip": roundtrip, "invariance": invariance, "relative_asymmetry": asymmetry,
            "mean_max_abs_z": float(np.max(np.abs(mean_z))), "mean_velocity_max_abs_z":
                float(np.max(np.abs(mean_z.reshape(-1, 4)[:, 2:]))), "mean_z_critical": bonferroni(len(mean_z)),
            "means_tested": int(len(mean_z)),
            "rotated_only": {"mean_d2": float(rotated_only.mean()), "predicted_mean_d2": predicted_rotated,
                             "standard_error": math.sqrt(var_rotated / samples),
                             "z": float((rotated_only.mean() - predicted_rotated) / math.sqrt(var_rotated / samples)),
                             "gate_99": gate, "rejection": rate_interval(rejected, samples),
                             "rejection_with_rotated_covariance": rate_interval(rejected_correct, samples)},
            "unit_only": {"mean_d2": float(unit_only.mean()), "mean_d2_consistent": float(d_pos.mean()),
                          "max_relative_deviation_from_scale_squared":
                              float(np.max(np.abs(unit_only / d_pos / scale ** 2 - 1.0))),
                          "exact_ratio": scale ** 2},
            "api_max_difference": api}


@task("T063", changed_files=files("sensor_fusion_geometry"), regression_tests=(
    f"{TESTS}::test_frame_transform_covariance_and_mahalanobis_invariance",
    f"{TESTS}::test_section_reports_labels_and_states[T063]"))
def frame_transform_covariance(ctx):
    study = frame_study()
    ctx.artifact_json("frame_transforms.json", as_json(study))
    names = list(study["per_map"])
    ctx.artifact_text("transform_z.svg", svg.line_plot(
        [("max |z| per transform", list(range(len(names))), [study["per_map"][n]["max_abs_z"] for n in names]),
         ("Bonferroni bound", [0, len(names) - 1], [study["z_critical"]] * 2)],
        title="T063 Monte Carlo J P J^T check (0 rot, 1 trans, 2 unit, 3 composite)",
        xlabel="transform index", ylabel="max |z|"))
    seed, rot = study["seed"], study["rotated_only"]
    findings = [
        finding("Monte Carlo covariances of rotated, translated, metre-to-millimetre and composite-transformed "
                "states match P' = J P J^T within a Bonferroni-corrected 99.9% sampling bound", "numerical",
                {"max_abs_z": study["max_abs_z"], "z_critical": study["z_critical"],
                 "moments_tested": study["moments_tested"],
                 "per_transform_max_abs_z": {n: study["per_map"][n]["max_abs_z"] for n in names}},
                {**generator_basis(seed, samples=study["samples"]), "checks": [
                    check("analytic", "Gaussian sampling law Var(S_ij) = (P'_ij^2 + P'_ii P'_jj)/N", study["max_abs_z"],
                          study["z_critical"], "le")]},
                uncertainty=max_abs_z_spread(study["moments_tested"], "standardized covariance entries"),
                tolerance=TOL_MC),
        finding("Roundoff sanity: the sample covariance of transformed samples equals J S J^T, J^-1 (J P J^T) "
                "J^-T returns P, and every transformed covariance (J P J^T and the FrameTransform output) is "
                "symmetric, to roundoff (algebraic identities that check the arithmetic, not the propagation law)",
                "numerical",
                {"relative_transport_error": study["exact_transport"], "relative_roundtrip_error": study["roundtrip"],
                 "relative_asymmetry": study["relative_asymmetry"]},
                {**generator_basis(seed), "checks": [
                    check("invariant", "sample covariance of J x + b against J S J^T", study["exact_transport"], 1e-12),
                    check("invariant", "J^-1 (J P J^T) J^-T against P", study["roundtrip"], 1e-12),
                    check("invariant", "max |P' - P'^T| relative to max |P'| over the transforms and the API output",
                          study["relative_asymmetry"], 1e-14)]},
                uncertainty=roundoff(max(study["exact_transport"], study["roundtrip"], study["relative_asymmetry"])),
                tolerance=TOL_TINY),
        finding("Transformed sample means match the declared maps: rotation and unit change act on the whole state, "
                "and the translation moves positions only, so the transformed velocities average to J_v mu_v",
                "numerical",
                {"max_abs_z": study["mean_max_abs_z"], "velocity_max_abs_z": study["mean_velocity_max_abs_z"],
                 "z_critical": study["mean_z_critical"], "means_tested": study["means_tested"]},
                {**generator_basis(seed, samples=study["samples"]), "checks": [
                    check("analytic", "sample means of J x + b against means built from the declared rotation, "
                                      "translation and scale (max |z|)", study["mean_max_abs_z"],
                          study["mean_z_critical"], "le")]},
                uncertainty=max_abs_z_spread(study["means_tested"], "standardized transformed means"),
                tolerance=TOL_MC),
        finding("Mahalanobis distance is invariant under every transform, including the metre-to-millimetre "
                "unit change, when the covariance is transformed with the vector", "numerical",
                study["invariance"],
                {**generator_basis(seed), "checks": [
                    check("invariant", "max relative change of d^2 over all samples and transforms",
                          max(study["invariance"].values()), 1e-9)]},
                unit="max relative difference", uncertainty=roundoff(max(study["invariance"].values())),
                tolerance=TOL_TINY),
        finding("FrameTransform.apply moves a typed position observation to value R z + t and covariance R Sigma "
                "R^T, matching J P J^T to roundoff", "computational_pipeline", study["api_max_difference"],
                {"derivation": "FrameTransform.apply in ciw.lab.sensor_fusion_objects", "checks": [
                    check("invariant", "API output against J P J^T and J mu + b", study["api_max_difference"], 1e-12)]},
                unit="max abs difference", uncertainty=roundoff(study["api_max_difference"]), tolerance=TOL_TINY),
        finding("Rotating a position measurement by 35 degrees without rotating its covariance inflates the mean "
                "Mahalanobis distance to tr(P^-1 R P R^T) and multiplies the 99% gate rejection rate", "numerical",
                {"mean_d2": rot["mean_d2"], "predicted_mean_d2": rot["predicted_mean_d2"], "z": rot["z"],
                 "rejection_rate": rot["rejection"]["rate"],
                 "rejection_rate_with_rotated_covariance": rot["rejection_with_rotated_covariance"]["rate"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "mean d^2 against tr(P^-1 R P R^T) (z)", rot["z"], 4.0),
                    check("analytic", "Wilson lower bound of the rejection rate above the nominal 1%",
                          rot["rejection"]["wilson"][0], 0.05, "ge"),
                    check("analytic", "rejection with the rotated covariance inside its Wilson interval around 1%",
                          float(rot["rejection_with_rotated_covariance"]["wilson"][0] <= 0.01
                                <= rot["rejection_with_rotated_covariance"]["wilson"][1]), 1.0, "ge")]},
                uncertainty=mc95(rot["standard_error"], "sampling standard error of the mean d^2 (exact variance "
                                                        "2 tr((P^-1 R P R^T)^2) / N)"),
                tolerance=TOL_MC, counterexample={
                    "statement": "Rotating a measurement into another frame without rotating its covariance is harmless",
                    "witness": {"rotation_deg": study["theta_deg"], "mean_d2": rot["mean_d2"], "nominal_mean_d2": 2.0,
                                "rejection_rate_at_99pct_gate": rot["rejection"]["rate"]}}),
        finding("Converting a state to millimetres while keeping its covariance in metres multiplies every squared "
                "Mahalanobis distance by exactly 10^6", "numerical", study["unit_only"],
                {**generator_basis(seed), "checks": [
                    check("invariant", "per-sample d^2 ratio against scale^2 (max relative deviation)",
                          study["unit_only"]["max_relative_deviation_from_scale_squared"], 1e-9)]},
                uncertainty=roundoff(study["unit_only"]["max_relative_deviation_from_scale_squared"]),
                tolerance=TOL_ROUNDOFF, counterexample={
                    "statement": "A unit change of the measurement vector alone leaves gating decisions unchanged",
                    "witness": {"mean_d2": study["unit_only"]["mean_d2"],
                                "consistent_mean_d2": study["unit_only"]["mean_d2_consistent"]}}),
        unreal("The declared 35 degree rotation, translation and body covariance describe a real sensor mounting "
               "or extrinsic calibration", "calibration", seed,
               "not established: the transform and covariance are declared synthetic values"),
    ]
    fields = {
        "hypothesis": "For affine frame maps y = J x + b (rotation, translation, unit scale and their composite) "
                      "the covariance transforms as J P J^T, translations do not change it, and Mahalanobis "
                      "distance is invariant when vector and covariance are transformed together.",
        "mathematical_model": "State (x, y, vx, vy) with declared body covariance (std 0.30, 0.05 m, 0.10, 0.06 "
                              "m/s, correlations 0.2-0.4). Rotation J = blockdiag(R, R) with R = R(35 deg); "
                              "translation (12.5, -4.0) m acts on positions; unit change J = 1000 I; composite "
                              "J = 1000 blockdiag(R, R). d^2 = e^T P^-1 e is invariant because "
                              "(J e)^T (J P J^T)^-1 (J e) = e^T P^-1 e for invertible J.",
        "input_data": [f"seed {seed} (PCG64)", f"{study['samples']} Gaussian samples of the body state"],
        "observation_model": "Samples x = mu + L xi with L the Cholesky factor of P; each map applied exactly.",
        "expected_invariant": "Sample covariance of J x + b within sampling error of J P J^T; exact J S J^T "
                              "identity; d^2 unchanged; vector-only transforms change d^2 by the predicted amount.",
        "experiment": "Transform 100000 samples by each map, compare sample covariance with J P J^T via z-scores "
                      "(Var(S_ij) = (P'_ij^2 + P'_ii P'_jj)/N), round-trip through J^-1, compare d^2 before and "
                      "after, then drop the covariance transform (rotation-only and unit-only counterexamples) "
                      "and route one observation through FrameTransform.apply.",
        "numerical_result": f"max |z| = {study['max_abs_z']:.3f} vs {study['z_critical']:.3f} over "
                            f"{study['moments_tested']} moments; max relative d^2 change "
                            f"{max(study['invariance'].values()):.1e}; rotation without covariance: mean d^2 "
                            f"{rot['mean_d2']:.3f} (predicted {rot['predicted_mean_d2']:.3f}, nominal 2), 99% gate "
                            f"rejects {rot['rejection']['rate']:.3f}; unit-only d^2 ratio "
                            f"{study['unit_only']['mean_d2'] / study['unit_only']['mean_d2_consistent']:.6g}.",
        "uncertainty": "Monte Carlo sampling error enters only the z-scores; the invariance and round-trip "
                       "statements are exact up to floating-point roundoff (relative 1e-12 or better).",
        "failure_modes_checked": ["translation leaking into covariance", "rotation applied to vector only",
                                  "unit change applied to vector only", "velocity block translated",
                                  "API transform differing from J P J^T", "loss of symmetry in the transformed P"],
        "unresolved_assumptions": ["The transforms are exact and known; an uncertain extrinsic (rotation with its own "
                                   "covariance) adds a J_theta Sigma_theta J_theta^T term not modelled here.",
                                   "Only affine maps are covered; nonlinear maps (range-bearing) need a "
                                   "linearization check such as T064's."],
        "recommended_next_task": "T064: propagate covariance through the Jacobi transfer matrix, where the map is "
                                 "only first-order.",
    }
    return outcome(fields, findings)


# T064 ------------------------------------------------------------------------------
SPHERE_START = (math.pi / 2, 0.0)
HYPERBOLIC_START = (0.0, 1.0)
HEADING = math.pi / 2  # along the equator / the vertical geodesic x = 0


def sphere_rhs(y):
    """Batched geodesic equations on the unit sphere in the polar chart (rows of y are (theta, phi, v, w))."""
    y = y.reshape(-1, 4)
    theta, v, w = y[:, 0], y[:, 2], y[:, 3]
    s, c = np.sin(theta), np.cos(theta)
    return np.column_stack([v, w, s * c * w * w, -2.0 * c / s * v * w]).ravel()


def hyperbolic_rhs(y):
    """Batched geodesic equations on the upper half-plane with g = I / y^2."""
    y = y.reshape(-1, 4)
    x2, a, b = y[:, 1], y[:, 2], y[:, 3]
    return np.column_stack([a, b, 2.0 * a * b / x2, (b * b - a * a) / x2]).ravel()


def sphere_lateral(states):
    """Signed geodesic distance to the equator (latitude) and its arclength derivative."""
    return math.pi / 2 - states[..., 0], -states[..., 2]


def hyperbolic_lateral(states):
    """Signed distance to the geodesic x = 0 (positive toward -x, the Jacobi normal) and its derivative."""
    x, y, a, b = states[..., 0], states[..., 1], states[..., 2], states[..., 3]
    w = -x / y
    return np.arcsinh(w), -(a * y - x * b) / (y * y) / np.sqrt(1.0 + w * w)


CASES = {
    "sphere": {"surface": lambda: Sphere(1.0), "start": SPHERE_START, "curvature": 1.0, "length": 1.2 * math.pi,
               "steps": 480, "rhs": sphere_rhs, "lateral": sphere_lateral, "sweep_s": 0.9 * math.pi,
               "exact_heading": lambda alpha, s: np.arcsin(np.sin(alpha) * math.sin(s))},
    "hyperbolic": {"surface": lambda: HyperbolicPlane(1.0), "start": HYPERBOLIC_START, "curvature": -1.0,
                   "length": 3.0, "steps": 300, "rhs": hyperbolic_rhs, "lateral": hyperbolic_lateral,
                   "sweep_s": 3.0, "exact_heading": lambda alpha, s: np.arcsinh(np.sin(alpha) * math.sinh(s))},
}
P0_JACOBI = {"lateral_std": 0.002, "heading_std": 0.02, "correlation": 0.25}
MC_GEODESICS = 600
SWEEP_SIGMAS = (0.003, 0.01, 0.03, 0.1, 0.3)
SWEEP_SAMPLES = 1500


def jacobi_prior() -> np.ndarray:
    a, b, r = P0_JACOBI["lateral_std"], P0_JACOBI["heading_std"], P0_JACOBI["correlation"]
    return np.array([[a * a, r * a * b], [r * a * b, b * b]])


def batched_geodesics(case, starts, length, steps):
    """Integrate many geodesics at once with the fixed-step RK4 of ciw.lab.integrators."""
    s, states = integrate_fixed(case["rhs"], np.asarray(starts, dtype=float).ravel(), length, steps, "rk4")
    return s, states.reshape(len(s), -1, 4)


def rhs_consistency(case, surface, rng) -> float:
    """Max difference between the batched RHS and the generic Christoffel RHS of the surface."""
    worst = 0.0
    u0 = np.asarray(case["start"], dtype=float)
    for _ in range(8):
        u = u0 + rng.uniform(-0.3, 0.3, 2)
        v = rng.standard_normal(2)
        y = np.concatenate([u, v])
        worst = max(worst, float(np.max(np.abs(case["rhs"](y) - surface.geodesic_rhs(y)))))
    return worst


def gauss_hermite_moments(function, sigma, s, nodes=96):
    """E[d^2] and E[d^4] for d = function(sigma xi, s), xi ~ N(0, 1), by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / math.sqrt(2 * math.pi)
    d = function(sigma * x, s)
    return float(w @ d ** 2), float(w @ d ** 4)


def linear_variance_error(case, sigma, s):
    """Relative error of the first-order lateral variance, sigma^2 j_head(s)^2 / E[d^2] - 1."""
    j_head = constant_curvature(case["curvature"], s)[2]
    exact, _ = gauss_hermite_moments(case["exact_heading"], sigma, s)
    return float(sigma ** 2 * j_head ** 2 / exact - 1.0)


def breakdown_sigma(case, level=0.1):
    """Heading std at which the first-order variance error reaches ``level`` (bisection), and the final bracket width.

    Returns (None, None) when the error stays below ``level`` up to 1 rad.
    """
    s = case["sweep_s"]
    lo, hi = 1e-4, 1.0
    if linear_variance_error(case, hi, s) < level:
        return None, None
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if linear_variance_error(case, mid, s) < level:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi), hi - lo


def jacobi_case(name: str, seed: int) -> dict:
    case = CASES[name]
    surface = case["surface"]()
    u0 = np.asarray(case["start"], dtype=float)
    rng = generator(seed)
    rhs_error = rhs_consistency(case, surface, rng)
    flow = transfer(surface, u0, HEADING, case["length"], steps=case["steps"])
    s = flow.s
    exact = np.stack(constant_curvature(case["curvature"], s), axis=1)  # j_lat, j_lat', j_head, j_head'
    computed = np.stack([flow.states[:, 4], flow.states[:, 5], flow.states[:, 6], flow.states[:, 7]], axis=1)
    phi_error = float(np.max(np.abs(computed - exact)))
    det_error = float(np.max(np.abs(flow.determinant() - 1.0)))
    P0 = jacobi_prior()

    # Monte Carlo over perturbed geodesics: exact normal-geodesic displacement and heading rotation.
    draws = rng.standard_normal((MC_GEODESICS, 2)) @ factor(P0).T
    starts = np.array([perturbed_start(surface, u0, HEADING, lateral=float(a), heading_change=float(b))
                       for a, b in draws])
    _, states = batched_geodesics(case, starts, case["length"], case["steps"])
    lateral, rate = case["lateral"](states)
    nodes = np.arange(0, len(s), case["steps"] // 12)
    predicted, sample, z_all = [], [], []
    for index in nodes:
        Phi = flow.matrix(index)
        P_s = Phi @ P0 @ Phi.T
        values = np.column_stack([lateral[index], rate[index]])
        S, z = covariance_z(values, P_s, known_mean=False)
        predicted.append(P_s)
        sample.append(S)
        z_all.append(z)
    z_all = np.concatenate(z_all)
    lateral_pred = np.array([p[0, 0] for p in predicted])
    lateral_mc = np.array([m[0, 0] for m in sample])
    full_lateral = np.array([(flow.matrix(i) @ P0 @ flow.matrix(i).T)[0, 0] for i in range(len(s))])

    # Integrator accuracy against the closed-form heading-only geodesic, and step-halving.
    alphas = rng.standard_normal(SWEEP_SAMPLES)
    s_eval = case["sweep_s"]
    sweep_steps = max(60, int(round(case["steps"] * s_eval / case["length"])))
    sweep = []
    closed_form_error = 0.0
    for sigma in SWEEP_SIGMAS:
        heading = sigma * alphas
        starts = np.array([perturbed_start(surface, u0, HEADING, heading_change=float(a)) for a in heading])
        _, final = batched_geodesics(case, starts, s_eval, sweep_steps)
        d = case["lateral"](final[-1])[0]
        closed_form_error = max(closed_form_error, float(np.max(np.abs(d - case["exact_heading"](heading, s_eval)))))
        exact2, exact4 = gauss_hermite_moments(case["exact_heading"], sigma, s_eval)
        coarse2, _ = gauss_hermite_moments(case["exact_heading"], sigma, s_eval, nodes=64)
        j_head = constant_curvature(case["curvature"], s_eval)[2]
        linear = float(sigma ** 2 * j_head ** 2)
        mc = float(np.mean(d ** 2))
        sweep.append({"sigma_heading": sigma, "linear_variance": linear, "exact_variance": exact2,
                      "monte_carlo_variance": mc,
                      "z_mc_vs_exact": float((mc - exact2) / math.sqrt((exact4 - exact2 ** 2) / SWEEP_SAMPLES)),
                      "linear_relative_error": linear / exact2 - 1.0,
                      "monte_carlo_relative_error": linear / mc - 1.0,
                      "quadrature_relative_change_64_to_96_nodes": abs(exact2 / coarse2 - 1.0)})
    second_order = {"sphere": math.cos(s_eval) ** 2, "hyperbolic": math.cosh(s_eval) ** 2}[name]
    leading = sweep[0]["linear_relative_error"] / SWEEP_SIGMAS[0] ** 2
    breakdown, bracket = breakdown_sigma(case)
    j_lat, _, j_hd, _ = constant_curvature(case["curvature"], np.array([0.0, case["length"]]))
    closed_lateral = j_lat ** 2 * P0[0, 0] + 2 * j_lat * j_hd * P0[0, 1] + j_hd ** 2 * P0[1, 1]
    return {"surface": surface.describe(), "start": u0, "heading": HEADING, "length": case["length"],
            "steps": case["steps"], "rhs_consistency": rhs_error, "phi_error": phi_error, "det_error": det_error,
            "conjugate_points": flow.conjugate_points(), "nodes_s": s[nodes], "predicted": predicted,
            "sample": sample, "max_abs_z": float(np.max(np.abs(z_all))), "moments_tested": int(len(z_all)),
            "lateral_predicted": lateral_pred, "lateral_monte_carlo": lateral_mc, "s": s,
            "full_lateral": full_lateral, "sweep_s": s_eval, "sweep": sweep,
            "closed_form_error": closed_form_error, "second_order_coefficient": second_order,
            "observed_leading_coefficient": leading, "breakdown_sigma_10pct": breakdown,
            "breakdown_bracket": bracket, "closed_form_growth": float(closed_lateral[1] / closed_lateral[0]),
            "lateral_mc_relative_se": math.sqrt(2.0 / (MC_GEODESICS - 1))}


def jacobi_study(seed: int = T064_SEED) -> dict:
    cases = {name: jacobi_case(name, seed + index) for index, name in enumerate(CASES)}
    z_crit = bonferroni(sum(c["moments_tested"] for c in cases.values()))
    sweep_crit = bonferroni(sum(len(c["sweep"]) for c in cases.values()))
    return {"seed": seed, "P0": jacobi_prior(), "cases": cases, "z_critical": z_crit, "sweep_z_critical": sweep_crit,
            "monte_carlo_geodesics": MC_GEODESICS, "sweep_samples": SWEEP_SAMPLES}


@task("T064", changed_files=files("sensor_fusion_geometry"), regression_tests=(
    f"{TESTS}::test_jacobi_transfer_covariance_collapse_and_breakdown",
    f"{TESTS}::test_section_reports_labels_and_states[T064]"))
def jacobi_transfer_covariance(ctx):
    study = ctx.memo("sensor_fusion.jacobi", jacobi_study)
    sphere, hyper = study["cases"]["sphere"], study["cases"]["hyperbolic"]
    seed = study["seed"]
    ctx.artifact_json("jacobi_covariance.json", as_json({k: v for k, v in study.items() if k != "cases"} | {
        "cases": {name: {k: v for k, v in case.items() if k not in ("s", "full_lateral")}
                  for name, case in study["cases"].items()}}))
    for name, case in study["cases"].items():
        ctx.artifact_text(f"{name}_lateral_variance.svg", svg.line_plot(
            [("Phi P Phi^T", case["s"], case["full_lateral"]),
             ("Monte Carlo", case["nodes_s"], case["lateral_monte_carlo"])],
            title=f"T064 {name}: lateral variance along the geodesic", xlabel="arclength s",
            ylabel="lateral variance", logy=True))
        ctx.artifact_text(f"{name}_linearization.svg", svg.line_plot(
            [("|linear/exact - 1|", [r["sigma_heading"] for r in case["sweep"]],
              [abs(r["linear_relative_error"]) for r in case["sweep"]]),
             ("10% level", [SWEEP_SIGMAS[0], SWEEP_SIGMAS[-1]], [0.1, 0.1])],
            title=f"T064 {name}: first-order variance error at s = {case['sweep_s']:.3f}",
            xlabel="heading std (rad)", ylabel="relative variance error", logx=True, logy=True))
    conjugate = sphere["conjugate_points"][0] if sphere["conjugate_points"] else float("nan")
    s_index = int(np.argmin(np.abs(sphere["nodes_s"] - math.pi)))
    collapse_pred = float(sphere["lateral_predicted"][s_index] / np.max(sphere["full_lateral"]))
    collapse_mc = float(sphere["lateral_monte_carlo"][s_index] / np.max(sphere["lateral_monte_carlo"]))
    sigma_l2 = float(study["P0"][0, 0])
    growth = float(hyper["lateral_predicted"][-1] / hyper["lateral_predicted"][0])
    growth_error = abs(growth / hyper["closed_form_growth"] - 1.0)
    sweep_z = max(abs(r["z_mc_vs_exact"]) for c in study["cases"].values() for r in c["sweep"])
    coefficient_error = max(abs(c["observed_leading_coefficient"] / c["second_order_coefficient"] - 1.0)
                            for c in study["cases"].values())
    table = {name: [{k: r[k] for k in ("sigma_heading", "linear_relative_error", "monte_carlo_relative_error")}
                    for r in case["sweep"]] for name, case in study["cases"].items()}
    breakdown = {name: case["breakdown_sigma_10pct"] for name, case in study["cases"].items()}
    found = all(value is not None for value in breakdown.values())
    quadrature = max(r["quadrature_relative_change_64_to_96_nodes"] for c in study["cases"].values() for r in c["sweep"])
    bracket = max(case["breakdown_bracket"] or 0.0 for case in study["cases"].values())

    def shown(value, digits):
        return "not found below 1 rad" if value is None else f"{value:.{digits}f} rad"
    findings = [
        finding("The Jacobi transfer matrix integrated by ciw.lab.jacobi.transfer matches the constant-curvature "
                "closed forms (cos s, sin s; cosh s, sinh s) with unit Wronskian on both surfaces", "numerical",
                {name: {"max_phi_error": c["phi_error"], "max_det_error": c["det_error"]}
                 for name, c in study["cases"].items()},
                {"derivation": "j'' + K j = 0 with constant K (ciw.lab.jacobi.constant_curvature)", "checks": [
                    check("analytic", "max |Phi - closed form| over both surfaces",
                          max(sphere["phi_error"], hyper["phi_error"]), 1e-7),
                    check("invariant", "max |det Phi - 1|", max(sphere["det_error"], hyper["det_error"]), 1e-7)]},
                uncertainty=uncertainty("truncation_bound", max(sphere["phi_error"], hyper["phi_error"]),
                                        "observed RK4 global error of Phi against the closed forms"),
                tolerance={"abs": 1e-9, "rel": 1e-3}),
        finding("Phi P Phi^T predicts the Monte Carlo covariance of (lateral offset, lateral rate) over perturbed "
                "geodesics on the sphere and the hyperbolic plane within a Bonferroni 99.9% bound", "numerical",
                {"max_abs_z": max(sphere["max_abs_z"], hyper["max_abs_z"]), "z_critical": study["z_critical"],
                 "moments_tested": sphere["moments_tested"] + hyper["moments_tested"],
                 "batched_rhs_vs_christoffel": max(sphere["rhs_consistency"], hyper["rhs_consistency"])},
                {**generator_basis(seed, geodesics=MC_GEODESICS), "checks": [
                    check("analytic", "sampling law of the 2x2 covariance at 13 arclength nodes per surface",
                          max(sphere["max_abs_z"], hyper["max_abs_z"]), study["z_critical"], "le"),
                    check("invariant", "batched geodesic RHS against Surface.geodesic_rhs",
                          max(sphere["rhs_consistency"], hyper["rhs_consistency"]), 1e-12)]},
                uncertainty=max_abs_z_spread(sphere["moments_tested"] + hyper["moments_tested"],
                                             "standardized covariance entries"),
                tolerance=TOL_MC),
        finding("Lateral variance collapses at the sphere's conjugate point s = pi: the heading contribution "
                "vanishes because j_head(pi) = 0, leaving only the initial lateral variance", "numerical",
                {"conjugate_point": conjugate, "predicted_ratio_to_peak": collapse_pred,
                 "monte_carlo_ratio_to_peak": collapse_mc,
                 "predicted_variance_at_pi": float(sphere["lateral_predicted"][s_index]),
                 "monte_carlo_variance_at_pi": float(sphere["lateral_monte_carlo"][s_index]),
                 "initial_lateral_variance": sigma_l2, "hyperbolic_conjugate_points": hyper["conjugate_points"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "first conjugate point minus pi", conjugate - math.pi, 1e-6),
                    check("analytic", "predicted variance at pi relative to sigma_lateral^2 minus one",
                          float(sphere["lateral_predicted"][s_index]) / sigma_l2 - 1.0, 1e-6),
                    check("analytic", "Monte Carlo ratio of variance at pi to its peak", collapse_mc, 0.05, "le")]},
                uncertainty=mc95(collapse_mc * sphere["lateral_mc_relative_se"],
                                 "Gaussian relative standard error sqrt(2/(N-1)) of a sample variance, applied to "
                                 "the Monte Carlo ratio"),
                tolerance=TOL_MC, counterexample={
                    "statement": "Lateral position uncertainty grows monotonically with distance travelled",
                    "witness": {"surface": "unit sphere", "s": math.pi, "variance_ratio_to_peak": collapse_mc}}),
        finding("On the hyperbolic plane (K = -1) there is no conjugate point and the lateral variance grows "
                "as the closed form cosh^2 s sigma_l^2 + 2 cosh s sinh s c + sinh^2 s sigma_h^2", "numerical",
                {"growth_factor_s0_to_s3": growth, "closed_form_growth_factor": hyper["closed_form_growth"],
                 "predicted_final_lateral_variance": float(hyper["lateral_predicted"][-1]),
                 "monte_carlo_final_lateral_variance": float(hyper["lateral_monte_carlo"][-1])},
                {**generator_basis(seed), "checks": [
                    check("analytic", "number of conjugate points on [0, 3]", len(hyper["conjugate_points"]), 0),
                    check("analytic", "integrated growth factor over 3 units of arclength against the closed form "
                                      "(relative)", growth_error, 1e-6)]},
                uncertainty=uncertainty("truncation_bound", growth_error,
                                        "relative RK4 error of the integrated growth factor"),
                tolerance=TOL_MC),
        finding("The first-order variance sigma^2 j_head^2 overestimates the exact lateral variance by a relative "
                "error that grows like cos^2(s) sigma^2 on the sphere and cosh^2(s) sigma^2 on the hyperbolic "
                "plane; integrated geodesics agree with the exact nonlinear expectation", "numerical",
                {"table": table, "leading_coefficient_relative_error": coefficient_error,
                 "max_abs_z_monte_carlo_vs_exact": sweep_z, "z_critical": study["sweep_z_critical"],
                 "closed_form_integration_error": max(sphere["closed_form_error"], hyper["closed_form_error"])},
                {**generator_basis(seed, samples=SWEEP_SAMPLES), "checks": [
                    check("analytic", "second-order coefficient from Gauss-Hermite against cos^2 s / cosh^2 s",
                          coefficient_error, 0.02),
                    check("analytic", "Monte Carlo variance against the Gauss-Hermite exact variance (max |z|)",
                          sweep_z, study["sweep_z_critical"], "le"),
                    check("analytic", "integrated lateral against asin(sin a sin s) / asinh(sin a sinh s)",
                          max(sphere["closed_form_error"], hyper["closed_form_error"]), 1e-6)]},
                uncertainty=uncertainty("reference_error", quadrature,
                                        "largest relative change of the Gauss-Hermite exact variance from 64 to 96 "
                                        "nodes"),
                tolerance=TOL_MC),
        finding("The heading standard deviation at which the first-order variance is 10% too large is about "
                "ten times smaller on the hyperbolic plane at s = 3 than on the sphere near its conjugate point "
                "(s = 0.9 pi)",
                "numerical", breakdown,
                {**generator_basis(seed), "checks": [
                    # A breakdown not found below 1 rad fails explicitly instead of passing a bound vacuously.
                    check("analytic", "10% breakdown found on both surfaces", float(found), 1.0, "ge"),
                    check("analytic", "hyperbolic breakdown sigma at the 10% level",
                          1.0 if breakdown["hyperbolic"] is None else breakdown["hyperbolic"], 0.05, "le"),
                    check("analytic", "sphere breakdown sigma at the 10% level",
                          0.0 if breakdown["sphere"] is None else breakdown["sphere"], 0.2, "ge")]},
                unit="rad", uncertainty=uncertainty("truncation_bound", bracket,
                                                    "final bisection bracket width on sigma (quadrature error "
                                                    "below the linearization finding's reference error)"),
                tolerance=TOL_ROUNDOFF, counterexample={
                    "statement": "A first-order (Phi P Phi^T) covariance is accurate for any heading uncertainty "
                                 "below 0.1 rad",
                    "witness": {"surface": "hyperbolic plane", "s": 3.0, "sigma_heading": 0.1,
                                "relative_variance_error": table["hyperbolic"][3]["linear_relative_error"]}}),
        unreal("The declared [lateral, heading] covariance and these surfaces predict the path uncertainty of a "
               "real vehicle or tool on a real curved part", "physical", seed,
               "not established: normalized synthetic surfaces and declared perturbations only"),
    ]
    fields = {
        "hypothesis": "A declared [lateral, heading] covariance propagates along a geodesic as Phi(s) P Phi(s)^T, "
                      "collapses in the heading direction at a conjugate point, and the first-order propagation "
                      "fails in a curvature-dependent way as the perturbation grows.",
        "mathematical_model": "Normal Jacobi field j'' + K j = 0; Phi(s) = [[j_lat, j_head], [j_lat', j_head']]. "
                              "Unit sphere (K = 1) along the equator from (theta, phi) = (pi/2, 0); hyperbolic "
                              "plane (K = -1, g = I/y^2) along x = 0 from (0, 1). P0: lateral std 0.002, heading "
                              "std 0.02 rad, correlation 0.25. Exact heading-only laterals: asin(sin a sin s) "
                              "(sphere), asinh(sin a sinh s) (hyperbolic); second-order relative variance error "
                              "cos^2 s sigma^2 and cosh^2 s sigma^2.",
        "input_data": [f"seed {seed} (PCG64)", f"{MC_GEODESICS} perturbed geodesics per surface",
                       f"heading sweep sigma in {list(SWEEP_SIGMAS)} with {SWEEP_SAMPLES} common random numbers"],
        "observation_model": "Lateral offset = signed geodesic distance from the perturbed geodesic's point at "
                             "arclength s to the base geodesic (latitude, or asinh(-x/y)); lateral rate = its "
                             "arclength derivative. First order, these equal j and j'.",
        "expected_invariant": "Covariance of (offset, rate) ~ Phi P Phi^T for small P; lateral variance at s = pi on "
                              "the sphere equals sigma_lateral^2; Monte Carlo equals the Gauss-Hermite exact "
                              "variance at every sigma.",
        "experiment": "Integrate Phi with jacobi.transfer (RK4); draw starts with jacobi.perturbed_start; integrate "
                      "all perturbed geodesics with integrators.integrate_fixed on a batched RHS checked against "
                      "Surface.geodesic_rhs; compare covariances at 13 nodes; sweep the heading std and compare "
                      "linear, Monte Carlo and quadrature variances.",
        "numerical_result": f"max |z| {max(sphere['max_abs_z'], hyper['max_abs_z']):.2f} vs "
                            f"{study['z_critical']:.2f}; conjugate point {conjugate:.9f}; variance at pi / peak "
                            f"{collapse_mc:.4f} (predicted {collapse_pred:.4f}); hyperbolic growth {growth:.0f}x "
                            f"(closed form {hyper['closed_form_growth']:.0f}x); 10% breakdown sigma: sphere "
                            f"{shown(breakdown['sphere'], 3)}, hyperbolic {shown(breakdown['hyperbolic'], 4)}.",
        "uncertainty": "Monte Carlo sampling error (600 geodesics) is covered by the Bonferroni bound; the "
                       "linearization errors come from 96-node Gauss-Hermite quadrature of closed forms and are "
                       "deterministic (64 against 96 nodes changes them by at most "
                       f"{quadrature:.0e} relative); RK4 integration error is below 1e-6 against the closed forms.",
        "failure_modes_checked": ["transfer matrix vs closed form", "Wronskian drift", "batched RHS vs Christoffel RHS",
                                  "chart singularity (sweep limited to sigma <= 0.3 on the sphere)",
                                  "linearization breakdown", "matched-arclength vs closest-point lateral offset "
                                  "(equal to first order)"],
        "unresolved_assumptions": ["Only constant-curvature surfaces were used, where exact nonlinear references "
                                   "exist; variable-curvature surfaces need a numerical reference.",
                                   "The perturbation is Gaussian in (lateral, heading); real path errors may have "
                                   "other shapes and correlations with speed."],
        "recommended_next_task": "T065: model the correlation that a filter itself induces between successive "
                                 "estimates.",
        # The numbers depend on the core Jacobi, integrator and surface modules; their digests are recorded.
        "provider_runtime_identity": identity(files("sensor_fusion_geometry"), *CORE_GEOMETRY),
    }
    return outcome(fields, findings)
