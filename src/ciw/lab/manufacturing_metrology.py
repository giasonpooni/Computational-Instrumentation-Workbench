"""Metrology models for the manufacturing use cases: sampling, registration, artifacts, frames, Gage R&R.

Scope: closed-form design rules for resolving curvature from sampled profiles,
rigid registration (Kabsch) and its residual statistics, a least-squares fit of
the as-built dome (height, width, centre, base plane) with its linearized
covariance, least-squares sphere and step-gauge fits with linearized
covariance, 3-2-1 datum frames, first-order and sigma-point covariance
propagation along a chain of rigid frames, and the ANOVA method for a balanced
crossed Gage R&R study. Each is exercised on seeded synthetic data with known
truth.

Non-claims: instrument noise figures are declared inputs, not measured
properties of any device. Recovering known synthetic components shows that the
estimators behave as derived; it says nothing about the capability of a real
gage, the validity of a real calibration certificate or a real frame chain.
"""
from __future__ import annotations

import math

import numpy as np


class MetrologyRefusal(ValueError):
    """An estimator was given data outside its declared design."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


# Curvature from sampled profiles -------------------------------------------
def quadratic_design(window, spacing) -> np.ndarray:
    """Sample abscissae: odd count, symmetric about the vertex, spacing <= requested."""
    half = int(math.ceil(0.5 * window / spacing))
    return np.linspace(-0.5 * window, 0.5 * window, 2 * half + 1)


def quadratic_curvature(x, z) -> float:
    """Curvature 2c of the least-squares fit z = a + b x + c x^2 (vertex slope assumed small)."""
    design = np.column_stack([np.ones_like(x), x, x * x])
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    return 2.0 * float(coefficients[2])


def curvature_noise_std(x, noise) -> float:
    """Exact standard deviation of 2c for independent noise ``noise`` on every sample."""
    design = np.column_stack([np.ones_like(x), x, x * x])
    return 2.0 * noise * math.sqrt(float(np.linalg.inv(design.T @ design)[2, 2]))


def curvature_noise_std_continuum(window, spacing, noise) -> float:
    """Large-n form: std(2c) = noise sqrt(720 spacing) / window^(5/2)."""
    return noise * math.sqrt(720.0 * spacing) / window ** 2.5


def curvature_bias_series(window, quartic) -> float:
    """Bias of 2c from a quartic profile term q x^4 over a uniform window: (3/7) q W^2."""
    return 3.0 * quartic * window ** 2 / 7.0


def sampling_design(curvature, quartic, tolerance, noise, coverage_factor=2.0, max_window=100.0) -> dict:
    """Largest window and sample spacing that resolve curvature to ``tolerance``.

    Half the tolerance is allotted to the quartic bias (3/7)|q| W^2 and half to
    k standard deviations of noise, k noise sqrt(720 d) W^(-5/2).
    """
    if quartic:
        window = min(max_window, math.sqrt(0.5 * tolerance * 7.0 / (3.0 * abs(quartic))))
    else:
        window = max_window
    spacing = (0.5 * tolerance * window ** 2.5 / (coverage_factor * noise)) ** 2 / 720.0
    return {"curvature_per_mm": curvature, "quartic_per_mm3": quartic, "tolerance_per_mm": tolerance,
            "noise_mm": noise, "coverage_factor": coverage_factor, "window_mm": window,
            "max_spacing_mm": spacing, "points_per_window": int(math.ceil(window / spacing)) + 1}


# Rigid registration ----------------------------------------------------------
def kabsch(reference, measured) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation minimizing sum |R reference + t - measured|^2 (proper rotation)."""
    reference, measured = np.asarray(reference, dtype=float), np.asarray(measured, dtype=float)
    if reference.shape != measured.shape or reference.shape[0] < 3:
        raise MetrologyRefusal("registration_underdetermined", "Registration needs at least three matched points")
    cr, cm = reference.mean(axis=0), measured.mean(axis=0)
    h = (reference - cr).T @ (measured - cm)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, cm - rotation @ cr


def rotation_vector(rotation) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (small and moderate angles)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    if angle < 1e-12:
        return 0.5 * np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                               rotation[1, 0] - rotation[0, 1]])
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(angle))
    return angle * axis


def registration_rotation_covariance(reference, noise) -> np.ndarray:
    """First-order rotation covariance noise^2 (sum |p|^2 I - p p^T)^-1 about the centroid."""
    p = np.asarray(reference, dtype=float) - np.mean(reference, axis=0)
    inertia = sum(float(q @ q) * np.eye(3) - np.outer(q, q) for q in p)
    return noise ** 2 * np.linalg.inv(inertia)


# Rigid frames ----------------------------------------------------------------
def hat(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(phi) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    angle = float(np.linalg.norm(phi))
    k = hat(phi)
    if angle < 1e-12:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + math.sin(angle) / angle * k + (1 - math.cos(angle)) / angle ** 2 * k @ k


def exp_se3(xi) -> np.ndarray:
    """4x4 transform of xi = (rho, phi) with the SO(3) left Jacobian on rho."""
    rho, phi = np.asarray(xi[:3], dtype=float), np.asarray(xi[3:], dtype=float)
    angle = float(np.linalg.norm(phi))
    k = hat(phi)
    if angle < 1e-12:
        left = np.eye(3) + 0.5 * k
    else:
        left = (np.eye(3) + (1 - math.cos(angle)) / angle ** 2 * k
                + (angle - math.sin(angle)) / angle ** 3 * k @ k)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = exp_so3(phi), left @ rho
    return out


def transform(rotation, translation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, translation
    return out


def adjoint(pose) -> np.ndarray:
    """Adjoint of a 4x4 pose for xi = (rho, phi): [[R, t^ R], [0, R]]."""
    r, t = pose[:3, :3], pose[:3, 3]
    out = np.zeros((6, 6))
    out[:3, :3], out[:3, 3:], out[3:, 3:] = r, hat(t) @ r, r
    return out


def compose_chain(links):
    """Compose [(pose, covariance), ...] parent-to-child; left perturbations exp(xi) T.

    Returns the composed pose and its first-order covariance
    sum_k Ad(T_1 ... T_{k-1}) C_k Ad(...)^T.
    """
    pose, covariance = np.eye(4), np.zeros((6, 6))
    for link_pose, link_covariance in links:
        ad = adjoint(pose)
        covariance = covariance + ad @ np.asarray(link_covariance) @ ad.T
        pose = pose @ link_pose
    return pose, covariance


def point_covariance(pose, covariance, point) -> np.ndarray:
    """Covariance of pose * point under a left perturbation: J C J^T with J = [I, -(T p)^]."""
    q = pose[:3, :3] @ point + pose[:3, 3]
    jac = np.hstack([np.eye(3), -hat(q)])
    return jac @ covariance @ jac.T


def sigma_point_chain_covariance(links, point) -> np.ndarray:
    """Point covariance from symmetric sigma points pushed through the exact chain of left-perturbed links.

    Every link covariance is factored by its eigenvectors; the 2n points
    +-sqrt(n) sqrt(lambda) v (n positive eigenvalues over all links, weights
    1 / 2n) reproduce the first-order covariance exactly for a linear map, so
    their difference from :func:`point_covariance` comes from the nonlinearity
    of the SE(3) products. The n scaling overweights fourth-order terms, which
    makes the difference a conservative linearization-error estimate.
    """
    directions = []
    for index, (_, covariance) in enumerate(links):
        covariance = np.asarray(covariance, dtype=float)
        values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        directions += [(index, math.sqrt(float(value)) * vectors[:, k]) for k, value in enumerate(values) if value > 0.0]
    scale = math.sqrt(len(directions))
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    outputs = []
    for index, direction in directions:
        for sign in (1.0, -1.0):
            pose = np.eye(4)
            for k, (link_pose, _) in enumerate(links):
                pose = pose @ (exp_se3(sign * scale * direction) @ link_pose if k == index else link_pose)
            outputs.append((pose @ homogeneous)[:3])
    centred = np.array(outputs) - np.mean(outputs, axis=0)
    return centred.T @ centred / (2 * len(directions))


def sample_chain_point(links, point, generator, count) -> np.ndarray:
    """Monte Carlo: perturb every link by exp(xi), xi ~ N(0, C), and transform the point."""
    factors = [np.linalg.cholesky(np.asarray(c) + 1e-30 * np.eye(6)) for _, c in links]
    out = np.empty((count, 3))
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    for n in range(count):
        pose = np.eye(4)
        for (link_pose, _), factor in zip(links, factors):
            pose = pose @ (exp_se3(factor @ generator.standard_normal(6)) @ link_pose)
        out[n] = (pose @ homogeneous)[:3]
    return out


def datum_frame_321(a_points, b_points, c_point) -> np.ndarray:
    """3-2-1 datum reference frame as a 4x4 pose (datum -> measurement frame).

    Primary A: plane through three points, z along its normal (right-handed
    with the listed order). Secondary B: direction of the two B points
    projected into A gives x. Tertiary C: the origin lies on plane A, on the
    B plane (normal y through B1) and on the C plane (normal x through C).
    """
    a, b, c = (np.asarray(v, dtype=float) for v in (a_points, b_points, c_point))
    z = np.cross(a[1] - a[0], a[2] - a[0])
    if np.linalg.norm(z) < 1e-12:
        raise MetrologyRefusal("datum_degenerate", "Primary datum points are collinear")
    z /= np.linalg.norm(z)
    direction = b[1] - b[0]
    x = direction - (direction @ z) * z
    if np.linalg.norm(x) < 1e-12:
        raise MetrologyRefusal("datum_degenerate", "Secondary datum direction is normal to the primary plane")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    origin = x * (x @ c) + y * (y @ b[0]) + z * (z @ a.mean(axis=0))
    return transform(np.column_stack([x, y, z]), origin)


def pose_difference(pose, nominal) -> np.ndarray:
    """xi with pose = exp(xi) nominal (first order)."""
    delta = pose @ np.linalg.inv(nominal)
    return np.concatenate([delta[:3, 3], rotation_vector(delta[:3, :3])])


# Calibration artifacts -------------------------------------------------------
def sphere_fit(points) -> tuple[np.ndarray, float, np.ndarray]:
    """Geometric least-squares sphere (algebraic start, Gauss-Newton); returns centre, radius, Jacobian."""
    p = np.asarray(points, dtype=float)
    if len(p) < 4:
        raise MetrologyRefusal("sphere_underdetermined", "A sphere fit needs at least four points")
    design = np.column_stack([2 * p, np.ones(len(p))])
    solution, *_ = np.linalg.lstsq(design, (p * p).sum(axis=1), rcond=None)
    centre = solution[:3]
    radius = math.sqrt(solution[3] + centre @ centre)
    for _ in range(50):
        offsets = p - centre
        distances = np.linalg.norm(offsets, axis=1)
        residual = distances - radius
        jac = np.column_stack([-offsets / distances[:, None], -np.ones(len(p))])
        step, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        centre, radius = centre + step[:3], radius + step[3]
        if np.max(np.abs(step)) < 1e-15 * max(1.0, radius):
            break
    offsets = p - centre
    distances = np.linalg.norm(offsets, axis=1)
    jac = np.column_stack([-offsets / distances[:, None], -np.ones(len(p))])
    return centre, float(radius), jac


def cap_points(centre, radius, count, max_polar, generator=None, noise=0.0) -> np.ndarray:
    """Deterministic spiral probe pattern on a spherical cap, optionally with radial noise."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    polar = np.arccos(1 - (1 - math.cos(max_polar)) * (np.arange(count) + 0.5) / count)
    azimuth = golden * np.arange(count)
    directions = np.column_stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])
    radial = radius + (generator.normal(0.0, noise, count) if generator is not None and noise else 0.0)
    return np.asarray(centre) + directions * np.reshape(radial, (-1, 1))


def step_gauge_fit(nominal, measured) -> tuple[np.ndarray, np.ndarray]:
    """Fit measured = (1 + e) nominal + b; returns (e, b) and the unscaled covariance (X^T X)^-1."""
    nominal = np.asarray(nominal, dtype=float)
    design = np.column_stack([nominal, np.ones_like(nominal)])
    solution, *_ = np.linalg.lstsq(design, np.asarray(measured, dtype=float) - nominal, rcond=None)
    return solution, np.linalg.inv(design.T @ design)


# As-built surface identification ---------------------------------------------
DOME_PARAMETERS = ("height_mm", "sigma_mm", "x0_mm", "y0_mm", "z0_mm", "tilt_x", "tilt_y")


def dome_surface(params, x, y) -> np.ndarray:
    """As-built dome z = z0 + a_x x + a_y y + h exp(-((x - x0)^2 + (y - y0)^2) / (2 sigma^2)).

    ``params`` follow :data:`DOME_PARAMETERS`; the base plane absorbs the
    registration tilt and offset of the scan.
    """
    h, sigma, x0, y0, z0, ax, ay = (float(v) for v in params)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return z0 + ax * x + ay * y + h * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def dome_jacobian(params, x, y) -> np.ndarray:
    """Exact derivatives of :func:`dome_surface` with respect to its seven parameters (columns)."""
    h, sigma, x0, y0 = (float(v) for v in params[:4])
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = x - x0, y - y0
    bump = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    return np.column_stack([bump, h * bump * (dx ** 2 + dy ** 2) / sigma ** 3, h * bump * dx / sigma ** 2,
                            h * bump * dy / sigma ** 2, np.ones_like(x), x, y])


def fit_dome(x, y, z, start, iterations=50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Newton least-squares fit of the as-built dome; returns parameters, (J^T J)^-1 and residuals.

    The parameter covariance for independent point noise sigma is
    sigma^2 (J^T J)^-1 at the solution (first order).
    """
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    if not (x.shape == y.shape == z.shape) or len(z) <= len(DOME_PARAMETERS):
        raise MetrologyRefusal("fit_underdetermined", "A dome fit needs more points than its seven parameters")
    params = np.asarray(start, dtype=float).copy()
    for _ in range(iterations):
        step, *_ = np.linalg.lstsq(dome_jacobian(params, x, y), z - dome_surface(params, x, y), rcond=None)
        params = params + step
        if np.max(np.abs(step)) < 1e-12 * max(1.0, float(np.max(np.abs(params)))):
            break
    jac = dome_jacobian(params, x, y)
    return params, np.linalg.inv(jac.T @ jac), z - dome_surface(params, x, y)


# Gage R&R (ANOVA method, balanced crossed design) ----------------------------
def gage_rr_anova(data) -> dict:
    """Variance components of y[part, operator, replicate] by the ANOVA method.

    Expected mean squares: E[MS_E] = s_e^2, E[MS_PO] = s_e^2 + r s_po^2,
    E[MS_O] = s_e^2 + r s_po^2 + p r s_o^2, E[MS_P] = s_e^2 + r s_po^2 + o r s_p^2.
    Negative moment estimates are reported raw and truncated at zero.
    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 3 or min(y.shape) < 2 or not np.all(np.isfinite(y)):
        raise MetrologyRefusal("unbalanced_design",
                               "Gage R&R ANOVA needs a complete balanced parts x operators x replicates array")
    p, o, r = y.shape
    grand = y.mean()
    part_means, operator_means, cell_means = y.mean(axis=(1, 2)), y.mean(axis=(0, 2)), y.mean(axis=2)
    ss_part = o * r * float(((part_means - grand) ** 2).sum())
    ss_operator = p * r * float(((operator_means - grand) ** 2).sum())
    ss_interaction = r * float(((cell_means - part_means[:, None] - operator_means[None, :] + grand) ** 2).sum())
    ss_error = float(((y - cell_means[:, :, None]) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    df = {"part": p - 1, "operator": o - 1, "interaction": (p - 1) * (o - 1), "error": p * o * (r - 1)}
    ms = {"part": ss_part / df["part"], "operator": ss_operator / df["operator"],
          "interaction": ss_interaction / df["interaction"], "error": ss_error / df["error"]}
    raw = {"repeatability": ms["error"], "interaction": (ms["interaction"] - ms["error"]) / r,
           "operator": (ms["operator"] - ms["interaction"]) / (p * r),
           "part": (ms["part"] - ms["interaction"]) / (o * r)}
    var = {key: max(0.0, value) for key, value in raw.items()}
    reproducibility = var["operator"] + var["interaction"]
    grr = var["repeatability"] + reproducibility
    total = grr + var["part"]
    return {"shape": [p, o, r], "ss": {"part": ss_part, "operator": ss_operator, "interaction": ss_interaction,
                                       "error": ss_error, "total": ss_total},
            "df": df, "ms": ms, "raw_components": raw, "components": var,
            "reproducibility": reproducibility, "grr": grr, "total": total,
            "percent_grr": 100.0 * math.sqrt(grr / total) if total > 0 else float("nan"),
            "ndc": 1.41 * math.sqrt(var["part"] / grr) if grr > 0 else float("inf"),
            "truncated": sorted(key for key, value in raw.items() if value < 0)}


def simulate_gage_study(generator, parts, operators, replicates, sigma) -> np.ndarray:
    """y = mu + P_i + O_j + (PO)_ij + e_ijk with declared standard deviations."""
    part = generator.normal(0.0, sigma["part"], parts)[:, None, None]
    operator = generator.normal(0.0, sigma["operator"], operators)[None, :, None]
    interaction = generator.normal(0.0, sigma["interaction"], (parts, operators))[:, :, None]
    error = generator.normal(0.0, sigma["repeatability"], (parts, operators, replicates))
    return 10.0 + part + operator + interaction + error
