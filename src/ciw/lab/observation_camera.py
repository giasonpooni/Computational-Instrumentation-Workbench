"""Synthetic pinhole stereo cameras observing markers on a declared cylinder.

Scope: the camera model used by the observation experiments T048-T051. A
camera maps world points to pixels by x_cam = R X + t, normalized coordinates
(x, y) = (x_cam / z_cam, y_cam / z_cam), optional Brown-Conrady distortion and
pixels u = f x + cx, v = f y + cy. The declared camera frame has z forward,
x right and y up (so image v grows upward); it is a convention, not a claim
about any device. Triangulation is linear (DLT on normalized coordinates) with
a ray-midpoint method kept as a second implementation for comparison.
Circular markers are flat discs whose image-ellipse centre is given in closed
form (dual conic) and by a conic fit through projected rim points; rounding
with a grid phase shared by all markers has a closed-form covariance.

Non-claims: every camera, marker and pixel here is generated. Agreement with
generated ground truth says nothing about the accuracy of a real camera, lens
or calibration procedure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .surfaces import Cylinder, rotation_matrix


@dataclass(frozen=True)
class Camera:
    focal_px: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    distortion: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    def describe(self) -> dict:
        return {"focal_px": self.focal_px, "principal_point_px": [self.cx, self.cy],
                "rotation_world_to_camera": self.rotation.tolist(), "translation_m": self.translation.tolist(),
                "center_m": self.center.tolist(), "distortion_k1_k2_p1_p2": list(self.distortion)}

    def to_camera(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points @ self.rotation.T + self.translation

    def project(self, points) -> np.ndarray:
        xc = self.to_camera(points)
        normalized = xc[..., :2] / xc[..., 2:3]
        if any(self.distortion):
            normalized = distort(normalized, self.distortion)
        return self.focal_px * normalized + np.array([self.cx, self.cy])

    def normalized(self, pixels, undistort_model=None) -> np.ndarray:
        x = (np.asarray(pixels, dtype=float) - np.array([self.cx, self.cy])) / self.focal_px
        if undistort_model is not None and any(undistort_model):
            x = undistort(x, undistort_model)
        return x

    def perturbed(self, *, focal=0.0, cx=0.0, cy=0.0, rotation_vector=(0.0, 0.0, 0.0),
                  center_shift=(0.0, 0.0, 0.0)) -> "Camera":
        """The same camera with calibration errors.

        The rotation error is applied about the camera center; ``center_shift``
        moves the center (world coordinates) without rotating the camera.
        """
        rotation = self.rotation
        vector = np.asarray(rotation_vector, dtype=float)
        angle = float(np.linalg.norm(vector))
        if angle > 0:
            rotation = rotation_matrix(vector / angle, angle) @ rotation
        center = self.center + np.asarray(center_shift, dtype=float)
        return replace(self, focal_px=self.focal_px + focal, cx=self.cx + cx, cy=self.cy + cy,
                       rotation=rotation, translation=-rotation @ center)


def look_at(center, target, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation and translation with z toward ``target``, y toward ``up``."""
    center, target, up = (np.asarray(v, dtype=float) for v in (center, target, up))
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.vstack([x, y, z])
    return rotation, -rotation @ center


# Declared synthetic rig: 2400 px focal length, 2048 x 1536 images, 0.2 m
# baseline, cameras converging on a point 0.6 m in front of the left camera.
RIG = {"focal_px": 2400.0, "principal_point_px": (1024.0, 768.0), "image_size_px": (2048, 1536),
       "baseline_m": 0.2, "convergence_distance_m": 0.6}
CYLINDER = {"radius_m": 0.1, "axis": "world +y", "front_point_m": (0.1, 0.0, 0.6),
            "chart_origin_direction": "toward the cameras (world -z)"}


def stereo_rig(distortion=(0.0, 0.0, 0.0, 0.0), rectified=False) -> tuple[Camera, Camera]:
    f, (cx, cy), b, d = RIG["focal_px"], RIG["principal_point_px"], RIG["baseline_m"], RIG["convergence_distance_m"]
    cameras = []
    for center in (np.zeros(3), np.array([b, 0.0, 0.0])):
        if rectified:
            rotation, translation = np.eye(3), -center
        else:
            rotation, translation = look_at(center, np.array([b / 2, 0.0, d]))
        cameras.append(Camera(f, cx, cy, rotation, translation, tuple(distortion)))
    return cameras[0], cameras[1]


def cylinder_to_world() -> tuple[np.ndarray, np.ndarray]:
    """Rigid map of the Cylinder chart embedding into the world: axis to +y, phi = 0 facing -z."""
    columns = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation = columns.T
    radius = CYLINDER["radius_m"]
    front = np.array(CYLINDER["front_point_m"])
    return rotation, front + np.array([0.0, 0.0, radius])


def helix_markers(alpha: float, s_values, phi0: float = -0.6, z0: float = -0.03, shift=(0.0, 0.0, 0.0)) -> dict:
    """Markers along the exact cylinder geodesic at angle ``alpha`` from the circumferential direction.

    ``shift`` translates the whole cylinder in the world; chords and arc
    lengths are unchanged, only where the markers fall in the images.
    """
    cylinder = Cylinder(CYLINDER["radius_m"])
    u0 = np.array([phi0, z0])
    tangent = cylinder.unit_tangent(u0, alpha)
    chart = cylinder.exact_geodesic(u0, tangent, np.asarray(s_values, dtype=float))
    rotation, offset = cylinder_to_world()
    local = np.array([cylinder.embedding(u) for u in chart])
    normals = np.array([[math.cos(u[0]), math.sin(u[0]), 0.0] for u in chart])
    return {"s": np.asarray(s_values, dtype=float), "chart": chart,
            "world": local @ rotation.T + offset + np.asarray(shift, dtype=float), "normals": normals @ rotation.T}


def visible(camera: Camera, points, normals) -> np.ndarray:
    """Front-facing markers in front of the camera and inside the declared image."""
    toward = camera.center[None, :] - points
    facing = np.einsum("ij,ij->i", toward, normals) > 0
    pixels = camera.project(points)
    width, height = RIG["image_size_px"]
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] <= width) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height)
    return facing & inside & (camera.to_camera(points)[:, 2] > 0)


# Distortion -------------------------------------------------------------------

def distort(normalized, model) -> np.ndarray:
    """Brown-Conrady radial (k1, k2) and tangential (p1, p2) distortion of normalized coordinates."""
    k1, k2, p1, p2 = model
    x, y = normalized[..., 0], normalized[..., 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def undistort(distorted, model, iterations: int = 60) -> np.ndarray:
    """Fixed-point inverse of :func:`distort`; converges for the moderate models used here."""
    k1, k2, p1, p2 = model
    xd, yd = distorted[..., 0], distorted[..., 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial
        y = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
    return np.stack([x, y], axis=-1)


# Triangulation ----------------------------------------------------------------

def triangulate(cameras, pixel_sets, undistort_model=None) -> np.ndarray:
    """Linear DLT triangulation on normalized coordinates; batched over points."""
    return triangulate_normalized(cameras, [camera.normalized(pixels, undistort_model)
                                            for camera, pixels in zip(cameras, pixel_sets)])


def triangulate_normalized(cameras, normalized_sets) -> np.ndarray:
    """DLT on already normalized coordinates; only the camera poses are used."""
    rows = []
    for camera, x in zip(cameras, normalized_sets):
        projection = np.hstack([camera.rotation, camera.translation[:, None]])
        rows.append(x[:, 0:1] * projection[2][None, :] - projection[0][None, :])
        rows.append(x[:, 1:2] * projection[2][None, :] - projection[1][None, :])
    system = np.stack(rows, axis=1)
    _, _, vt = np.linalg.svd(system)
    homogeneous = vt[:, -1, :]
    return homogeneous[:, :3] / homogeneous[:, 3:4]


def triangulate_midpoint(cameras, pixel_sets) -> np.ndarray:
    """Midpoint of the shortest segment between the two viewing rays."""
    (a, b), (pa, pb) = cameras, pixel_sets
    xa, xb = a.normalized(pa), b.normalized(pb)
    da = np.column_stack([xa, np.ones(len(xa))]) @ a.rotation
    db = np.column_stack([xb, np.ones(len(xb))]) @ b.rotation
    w = a.center - b.center
    aa, bb, ab = (np.einsum("ij,ij->i", u, v) for u, v in ((da, da), (db, db), (da, db)))
    aw, bw = da @ w, db @ w
    denominator = aa * bb - ab * ab
    ta = (ab * bw - bb * aw) / denominator
    tb = (aa * bw - ab * aw) / denominator
    return 0.5 * ((a.center + ta[:, None] * da) + (b.center + tb[:, None] * db))


CALIBRATION_PARAMETERS = ("focal_px", "right_cx_px", "right_cy_px", "right_rx_rad", "right_ry_rad", "right_rz_rad",
                          "right_focal_px", "left_cx_px", "left_cy_px", "right_tx_m", "right_ty_m", "right_tz_m")


def believed_rig(cameras, delta) -> tuple[Camera, Camera]:
    """Calibration used for triangulation when it differs from the true rig.

    ``delta`` follows :data:`CALIBRATION_PARAMETERS`: focal error of both
    cameras [px], right principal point errors cx, cy [px], right-camera
    rotation vector [rad] about its center, an additional right-camera focal
    error [px], left principal point errors cx, cy [px] and the right-camera
    center shift [m] (its x component changes the baseline). A six-entry
    ``delta`` perturbs only the first six. Chords are invariant under a rigid
    motion of the whole believed rig, so a left-camera pose error is
    equivalent to a right-camera pose error and is not listed separately.
    Aspect ratio and skew are not modelled.
    """
    values = [float(v) for v in delta] + [0.0] * (len(CALIBRATION_PARAMETERS) - len(delta))
    focal, cx, cy, wx, wy, wz, right_focal, left_cx, left_cy, tx, ty, tz = values
    left, right = cameras
    return (replace(left, focal_px=left.focal_px + focal, cx=left.cx + left_cx, cy=left.cy + left_cy),
            right.perturbed(focal=focal + right_focal, cx=cx, cy=cy, rotation_vector=(wx, wy, wz),
                            center_shift=(tx, ty, tz)))


def pair_chords(points, pairs) -> np.ndarray:
    pairs = np.asarray(pairs)
    return np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)


def noisy_chords(cameras, points, pairs, sigma_px: float, trials: int, rng, quantize: bool = True,
                 shared_phase: bool = False) -> np.ndarray:
    """Chords triangulated from Gaussian-noisy, optionally integer-rounded pixels (distortion-free cameras).

    Each trial draws an independent uniform pixel-grid phase per marker,
    camera and axis (the unknown sub-pixel position of each marker image) and
    removes it again after rounding, so the rounding errors are uniform on
    [-1/2, 1/2) px and independent between coordinates, as the linear
    propagation sigma_c^2 = (sigma^2 + 1/12) sum J^2 assumes. With
    ``shared_phase`` one phase per trial, camera and axis is shared by all
    markers, which correlates their rounding errors
    (:func:`shared_phase_rounding_covariance`).
    """
    normalized = []
    for camera in cameras:
        ideal = camera.project(points)
        shape = (trials, 1 if shared_phase else len(points), 2)
        phase = rng.uniform(0.0, 1.0, shape) if quantize else np.zeros(shape)
        pixels = ideal[None] + phase + sigma_px * rng.standard_normal((trials, len(points), 2))
        if quantize:
            pixels = np.round(pixels)
        normalized.append(((pixels - phase - np.array([camera.cx, camera.cy])) / camera.focal_px).reshape(-1, 2))
    world = triangulate_normalized(cameras, normalized).reshape(trials, len(points), 3)
    pairs = np.asarray(pairs)
    return np.linalg.norm(world[:, pairs[:, 0]] - world[:, pairs[:, 1]], axis=2)


def shared_phase_rounding_covariance(offset_px: float, sigma_px: float) -> float:
    """Covariance of the total errors of two coordinates rounded with one shared uniform grid phase.

    With e = round(x + phi + n) - phi - x, independent Gaussian n of standard
    deviation sigma and d = frac(x_1 - x_2), the Fourier series of the
    sawtooth gives Cov(e_1, e_2) = sum_k cos(2 pi k d) exp(-4 pi^2 k^2 sigma^2)
    / (2 pi^2 k^2), which is 1/12 - d (1 - d)/2 for sigma = 0
    (docs/lab/OBSERVATION.md, T051).
    """
    d = float(offset_px) % 1.0
    if sigma_px == 0:
        return 1 / 12 - d * (1 - d) / 2
    # Terms beyond exp(-4 pi^2 k^2 sigma^2) < 1e-18 are below double precision.
    terms = int(math.ceil(math.sqrt(math.log(1e18) / (4 * math.pi ** 2 * sigma_px ** 2)))) + 1
    k = np.arange(1, terms + 1)
    return float(np.sum(np.cos(2 * math.pi * k * d) * np.exp(-4 * math.pi ** 2 * k ** 2 * sigma_px ** 2)
                        / (2 * math.pi ** 2 * k ** 2)))


def shared_phase_chord_variance(cameras, points, pairs, jacobian, sigma_px: float) -> np.ndarray:
    """Chord variance J Sigma J^T when all markers of a camera share one grid phase per axis.

    Coordinates of different cameras or axes have independent phases; the
    two markers of a pair on the same camera and axis are correlated by
    :func:`shared_phase_rounding_covariance`. The Jacobian columns follow
    :func:`chord_pixel_jacobian`.
    """
    pairs = np.asarray(pairs)
    left, right = cameras
    ends = [np.concatenate([left.project(points[pairs[:, i]]), right.project(points[pairs[:, i]])], axis=1)
            for i in (0, 1)]
    variance = (sigma_px ** 2 + 1 / 12) * np.sum(jacobian ** 2, axis=1)
    for row in range(len(pairs)):
        for axis in range(4):
            covariance = shared_phase_rounding_covariance(ends[0][row, axis] - ends[1][row, axis], sigma_px)
            variance[row] += 2 * jacobian[row, axis] * jacobian[row, 4 + axis] * covariance
    return variance


# Circular markers under perspective ------------------------------------------------

def disc_frame(normal) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane axes of a disc with unit ``normal``."""
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def disc_image_centre(camera: Camera, centre, normal, radius: float) -> np.ndarray:
    """Pixel centre of the image ellipse of a flat circular marker (closed form, distortion-free camera).

    The image conic's dual is H diag(rho^2, rho^2, -1) H^T with H = [e1 e2 C]
    in camera coordinates; the ellipse centre, pole of the line at infinity,
    is therefore Z C - rho^2 t with t = e_z - n_z n, the in-plane component of
    the optical axis: x_e = (Z X - rho^2 t_x) / (Z^2 - rho^2 t_z), and likewise
    for y. It equals the projected marker centre X/Z only when t = 0
    (fronto-parallel marker).
    """
    c = camera.to_camera(np.asarray(centre, dtype=float)[None])[0]
    n = camera.rotation @ (np.asarray(normal, dtype=float) / np.linalg.norm(normal))
    t = np.array([0.0, 0.0, 1.0]) - n[2] * n
    x = (c[2] * c[:2] - radius ** 2 * t[:2]) / (c[2] ** 2 - radius ** 2 * t[2])
    return camera.focal_px * x + np.array([camera.cx, camera.cy])


def conic_centre(pixels) -> np.ndarray:
    """Centre of the conic through >= 5 image points (algebraic fit, exact for points on one conic)."""
    pixels = np.asarray(pixels, dtype=float)
    mean = pixels.mean(axis=0)
    scale = float(np.max(np.abs(pixels - mean)))
    x, y = ((pixels - mean) / scale).T
    design = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    a, b, c, d, e, _ = np.linalg.svd(design)[2][-1]
    return mean + scale * np.linalg.solve(np.array([[2 * a, b], [b, 2 * c]]), [-d, -e])


def disc_rim(centre, normal, radius: float, samples: int = 64) -> np.ndarray:
    """World points on the rim of a flat circular marker."""
    first, second = disc_frame(normal)
    angle = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    return (np.asarray(centre, dtype=float)[None]
            + radius * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second))


def weak_perspective_project(camera: Camera, points, depth: float) -> np.ndarray:
    """Affine (weak-perspective) pixels: camera x, y divided by one fixed ``depth`` instead of each z."""
    xc = camera.to_camera(points)
    return camera.focal_px * xc[..., :2] / depth + np.array([camera.cx, camera.cy])


def chord_from_pixels(cameras, pixels_left, pixels_right, pairs, undistort_model=None) -> np.ndarray:
    return pair_chords(triangulate(cameras, (pixels_left, pixels_right), undistort_model), pairs)


def chord_pixel_jacobian(cameras, points_a, points_b, step=1e-3) -> np.ndarray:
    """d chord / d (uL_a, vL_a, uR_a, vR_a, uL_b, vL_b, uR_b, vR_b) by central differences, per row."""
    left, right = cameras
    base = np.concatenate([left.project(points_a), right.project(points_a),
                           left.project(points_b), right.project(points_b)], axis=1)
    n = len(base)
    jac = np.empty((n, 8))
    for j in range(8):
        chords = []
        for sign in (1.0, -1.0):
            pix = base.copy()
            pix[:, j] += sign * step
            pa = triangulate(cameras, (pix[:, 0:2], pix[:, 2:4]))
            pb = triangulate(cameras, (pix[:, 4:6], pix[:, 6:8]))
            chords.append(np.linalg.norm(pa - pb, axis=1))
        jac[:, j] = (chords[0] - chords[1]) / (2 * step)
    return jac
