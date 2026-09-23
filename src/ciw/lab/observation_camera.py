"""Synthetic pinhole stereo cameras observing markers on a declared cylinder.

Scope: the camera model used by the observation experiments T048-T051. A
camera maps world points to pixels by x_cam = R X + t, normalized coordinates
(x, y) = (x_cam / z_cam, y_cam / z_cam), optional Brown-Conrady distortion and
pixels u = f x + cx, v = f y + cy. The declared camera frame has z forward,
x right and y up (so image v grows upward); it is a convention, not a claim
about any device. Triangulation is linear (DLT on normalized coordinates) with
a ray-midpoint method kept as a second implementation for comparison.

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

    def perturbed(self, *, focal=0.0, cx=0.0, cy=0.0, rotation_vector=(0.0, 0.0, 0.0)) -> "Camera":
        """The same camera with calibration errors; the rotation error is applied about the camera center."""
        rotation = self.rotation
        vector = np.asarray(rotation_vector, dtype=float)
        angle = float(np.linalg.norm(vector))
        if angle > 0:
            rotation = rotation_matrix(vector / angle, angle) @ rotation
        center = self.center
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


def believed_rig(cameras, delta) -> tuple[Camera, Camera]:
    """Calibration used for triangulation when it differs from the true rig.

    ``delta`` = (focal error of both cameras [px], right principal point
    errors cx, cy [px], right-camera rotation vector [rad] about its center).
    """
    focal, cx, cy, wx, wy, wz = (float(v) for v in delta)
    left, right = cameras
    return (replace(left, focal_px=left.focal_px + focal),
            right.perturbed(focal=focal, cx=cx, cy=cy, rotation_vector=(wx, wy, wz)))


def pair_chords(points, pairs) -> np.ndarray:
    pairs = np.asarray(pairs)
    return np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)


def noisy_chords(cameras, points, pairs, sigma_px: float, trials: int, rng, quantize: bool = True) -> np.ndarray:
    """Chords triangulated from Gaussian-noisy, optionally integer-rounded pixels (distortion-free cameras).

    Each trial draws a uniform pixel-grid phase per camera (the unknown
    sub-pixel position of the principal point); triangulation uses the
    matching principal point, so the phase adds no calibration error but
    makes the rounding error uniform on [-1/2, 1/2) px.
    """
    normalized = []
    for camera in cameras:
        ideal = camera.project(points)
        phase = rng.uniform(0.0, 1.0, (trials, 1, 2)) if quantize else np.zeros((trials, 1, 2))
        pixels = ideal[None] + phase + sigma_px * rng.standard_normal((trials, len(points), 2))
        if quantize:
            pixels = np.round(pixels)
        normalized.append(((pixels - phase - np.array([camera.cx, camera.cy])) / camera.focal_px).reshape(-1, 2))
    world = triangulate_normalized(cameras, normalized).reshape(trials, len(points), 3)
    pairs = np.asarray(pairs)
    return np.linalg.norm(world[:, pairs[:, 0]] - world[:, pairs[:, 1]], axis=2)


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
