"""Triangle meshes, their refusal states and intrinsic geodesics (NumPy only).

Scope: deterministic mesh generators (icosphere, latitude-longitude sphere,
prism cylinder, Schwarz lantern, planar grid, torus grid), a validator that
names every structural defect it refuses, straightest geodesics traced by
unfolding across edges (Polthier-Schmies: straight inside a face, equal angles
on both sides of an edge), graph distances on the edge graph and on a
Steiner-point graph, a dense heat-method distance for small meshes,
angle-defect Gaussian curvature, vertex normals and the planar unfolding of a
face strip.

Declared rules: a traced geodesic that reaches a vertex (within a relative
edge-parameter tolerance) is refused with ``vertex_hit`` rather than continued
by the vertex angle-bisection rule; a geodesic that reaches a boundary edge is
refused with ``boundary_reached`` and keeps its partial path.

Non-claims: every mesh is generated from a declared formula in normalized
units. Nothing here reads, represents or validates a scanned physical surface,
and agreement with a smooth surface is agreement between two computations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

# Relative thresholds; all meshes here have unit-scale edge lengths or smaller.
DEGENERATE_RATIO = 1e-12      # 2 * area / longest_edge^2 at or below this is a zero-area face
VERTEX_TOLERANCE = 1e-9       # edge parameter distance to an endpoint that counts as a vertex hit
ON_FACE_TOLERANCE = 1e-9      # distance to a face (relative to its longest edge) for a start point
FOLD_COSINE = -0.9            # adjacent unit normals with dot at or below this (bend > 154 degrees) are folded

# Named refusal codes in the order the validator reports them.
MESH_CODES = ("invalid_shape", "empty_mesh", "nonfinite_vertex", "invalid_face_index", "degenerate_face",
              "non_manifold_edge", "inconsistent_orientation", "non_manifold_vertex", "folded_face", "unreferenced_vertex",
              "disconnected_components", "open_boundary")
TRACE_CODES = ("point_outside_face", "invalid_direction", "boundary_reached", "vertex_hit",
               "step_budget_exceeded")
QUERY_CODES = ("unreachable_target", "boundary_vertex_curvature")


class MeshRefusal(ValueError):
    """Invalid or incomplete surface data, identified by a named code."""

    def __init__(self, code: str, message: str, issues=None):
        super().__init__(message)
        self.code = code
        self.issues = list(issues or [(code, message)])


# ---------------------------------------------------------------- validation
def inspect(vertices, faces, *, require_connected=True, require_closed=False) -> list:
    """Every structural defect as (code, message), in MESH_CODES order; empty when valid."""
    issues = []
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        return [("invalid_shape", "Vertices must be (n, 3) and faces (m, 3) arrays")]
    if len(vertices) < 3 or len(faces) < 1:
        return [("empty_mesh", "A mesh needs at least three vertices and one face")]
    if not np.issubdtype(faces.dtype, np.integer):
        return [("invalid_shape", "Face indices must be integers")]
    bad = np.flatnonzero(~np.all(np.isfinite(vertices), axis=1))
    if len(bad):
        issues.append(("nonfinite_vertex", f"Vertices {bad[:5].tolist()} have nonfinite coordinates"))
    out = np.flatnonzero(np.any((faces < 0) | (faces >= len(vertices)), axis=1))
    if len(out):
        issues.append(("invalid_face_index", f"Faces {out[:5].tolist()} reference vertices outside the mesh"))
        return issues
    faces = faces.astype(np.int64)
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    if not len(bad):
        cross, longest = _cross_and_longest(vertices, faces)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.linalg.norm(cross, axis=1) / np.maximum(longest ** 2, 1e-300)
        repeated |= ratio <= DEGENERATE_RATIO
    degenerate = np.flatnonzero(repeated)
    if len(degenerate):
        issues.append(("degenerate_face", f"Faces {degenerate[:5].tolist()} have zero area or repeated vertices"))
    edges = _edge_table(faces, len(vertices))
    if len(edges["non_manifold"]):
        issues.append(("non_manifold_edge", f"{len(edges['non_manifold'])} edges are shared by more than two faces"))
    if edges["misoriented"]:
        issues.append(("inconsistent_orientation",
                       f"{edges['misoriented']} interior edges are traversed in the same direction by both faces"))
    fans = _fan_count(faces, edges)
    pinched = np.flatnonzero(fans > 1)
    if len(pinched):
        issues.append(("non_manifold_vertex", f"Vertices {pinched[:5].tolist()} join separate face fans"))
    if not len(bad) and not len(degenerate) and not edges["misoriented"] and len(edges["pair_first"]):
        # Consistently oriented neighbours with nearly opposite normals: a fold-over
        # that structural checks accept but that inverts local geometry.
        unit = cross / np.linalg.norm(cross, axis=1)[:, None]
        dots = np.einsum("ij,ij->i", unit[edges["pair_first"] // 3], unit[edges["pair_second"] // 3])
        folded = int(np.sum(dots <= FOLD_COSINE))
        if folded:
            issues.append(("folded_face", f"{folded} interior edges join faces with nearly opposite normals"))
    unused = np.flatnonzero(np.bincount(faces.ravel(), minlength=len(vertices)) == 0)
    if len(unused):
        issues.append(("unreferenced_vertex", f"Vertices {unused[:5].tolist()} belong to no face"))
    if require_connected:
        components = face_components(len(faces), edges)
        if components.max() > 0:
            issues.append(("disconnected_components", f"The mesh has {components.max() + 1} face components"))
    if require_closed and edges["boundary_count"]:
        issues.append(("open_boundary", f"{edges['boundary_count']} boundary edges on a mesh declared closed"))
    return issues


def _cross_and_longest(vertices, faces):
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    longest = np.max(np.stack([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1),
                               np.linalg.norm(p0 - p2, axis=1)]), axis=0)
    return cross, longest


def _edge_table(faces, n_vertices):
    """Unique edges, half-edge pairing, face neighbours and orientation defects (sort based)."""
    m = len(faces)
    tail = faces.reshape(-1)
    head = np.roll(faces, -1, axis=1).reshape(-1)
    lo, hi = np.minimum(tail, head), np.maximum(tail, head)
    key = lo * np.int64(n_vertices) + hi
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    counts = np.diff(np.r_[starts, len(sorted_key)])
    edge_of_half = np.empty(3 * m, dtype=np.int64)
    edge_of_half[order] = np.repeat(np.arange(len(starts)), counts)
    neighbors = np.full(3 * m, -1, dtype=np.int64)
    pairs = starts[counts == 2]
    first, second = order[pairs], order[pairs + 1]
    neighbors[first], neighbors[second] = second // 3, first // 3
    misoriented = int(np.sum(tail[first] == tail[second]))
    return {"lo": lo[order[starts]], "hi": hi[order[starts]], "counts": counts,
            "edge_of_half": edge_of_half.reshape(m, 3), "neighbors": neighbors.reshape(m, 3),
            "pair_first": first, "pair_second": second, "misoriented": misoriented,
            "non_manifold": np.flatnonzero(counts > 2), "boundary_count": int(np.sum(counts == 1)),
            "tail": tail, "head": head}


def _propagate_min(labels, a, b):
    """Connected-component labels by min-label propagation with pointer jumping."""
    while True:
        new = labels.copy()
        np.minimum.at(new, a, labels[b])
        np.minimum.at(new, b, labels[a])
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, labels):
            return labels
        labels = new


def _fan_count(faces, edges):
    """Number of edge-connected face fans around every vertex (one for a manifold vertex)."""
    m = len(faces)
    first, second = edges["pair_first"], edges["pair_second"]
    if len(first) == 0:
        incidences_a = incidences_b = np.zeros(0, dtype=np.int64)
    else:
        f1, k1 = first // 3, first % 3
        f2, k2 = second // 3, second % 3
        tails = []
        heads = []
        for local in (0, 1):
            vertex = faces[f1, (k1 + local) % 3]
            match = np.where(faces[f2, k2] == vertex, k2, (k2 + 1) % 3)
            tails.append(3 * f1 + (k1 + local) % 3)
            heads.append(3 * f2 + match)
        incidences_a, incidences_b = np.concatenate(tails), np.concatenate(heads)
    labels = _propagate_min(np.arange(3 * m, dtype=np.int64), incidences_a, incidences_b)
    vertex = faces.reshape(-1)
    unique = np.unique(np.stack([vertex, labels]), axis=1)
    return np.bincount(unique[0], minlength=int(faces.max()) + 1)


def face_components(m, edges) -> np.ndarray:
    first, second = edges["pair_first"], edges["pair_second"]
    labels = _propagate_min(np.arange(m, dtype=np.int64), first // 3, second // 3)
    _, compact = np.unique(labels, return_inverse=True)
    return compact


@dataclass
class TriMesh:
    """A validated, consistently oriented triangle mesh with face adjacency."""

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=float)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        table = _edge_table(self.faces, len(self.vertices))
        self.neighbors = table["neighbors"]
        self.edge_of_half = table["edge_of_half"]
        self.edges = np.stack([table["lo"], table["hi"]], axis=1)
        self.edge_counts = table["counts"]
        p0, p1, p2 = (self.vertices[self.faces[:, k]] for k in range(3))
        cross = np.cross(p1 - p0, p2 - p0)
        self.face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        self.face_normals = cross / (2.0 * self.face_areas[:, None])
        boundary = self.edges[self.edge_counts == 1]
        self.boundary_vertices = np.zeros(len(self.vertices), dtype=bool)
        self.boundary_vertices[boundary.ravel()] = True

    @classmethod
    def build(cls, vertices, faces, name="mesh", *, require_connected=True, require_closed=False, params=None):
        """Validate and construct; refuse with the first named defect (all defects are attached)."""
        issues = inspect(vertices, faces, require_connected=require_connected, require_closed=require_closed)
        if issues:
            code, message = issues[0]
            raise MeshRefusal(code, message, issues)
        return cls(np.asarray(vertices, dtype=float), np.asarray(faces), name, dict(params or {}))

    # Metrics ----------------------------------------------------------
    @property
    def edge_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], axis=1)

    def mean_edge(self) -> float:
        return float(np.mean(self.edge_lengths))

    def area(self) -> float:
        return float(np.sum(self.face_areas))

    def corner_angles(self) -> np.ndarray:
        """Interior angle at every face corner, shape (m, 3), via atan2 for accuracy."""
        v = self.vertices[self.faces]
        angles = np.empty(self.faces.shape)
        for k in range(3):
            a = v[:, (k + 1) % 3] - v[:, k]
            b = v[:, (k + 2) % 3] - v[:, k]
            angles[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
        return angles

    def quality(self) -> dict:
        """Minimum angle (degrees) and radius ratio R_circ / (2 r_in) (1 for equilateral)."""
        v = self.vertices[self.faces]
        a = np.linalg.norm(v[:, 1] - v[:, 2], axis=1)
        b = np.linalg.norm(v[:, 2] - v[:, 0], axis=1)
        c = np.linalg.norm(v[:, 0] - v[:, 1], axis=1)
        area = self.face_areas
        s = 0.5 * (a + b + c)
        ratio = (a * b * c / (4 * area)) / (2 * area / s)
        return {"min_angle_deg": float(np.degrees(self.corner_angles().min())),
                "max_radius_ratio": float(ratio.max()), "mean_radius_ratio": float(ratio.mean())}

    def locate(self, point, tolerance=ON_FACE_TOLERANCE):
        """(face, barycentric) of a point lying on the mesh; refuses a point on no face."""
        point = np.asarray(point, dtype=float)
        p0 = self.vertices[self.faces[:, 0]]
        offset = np.einsum("ij,ij->i", point[None, :] - p0, self.face_normals)
        bary = self._barycentric(np.arange(len(self.faces)), point[None, :] - offset[:, None] * self.face_normals)
        scale = np.sqrt(self.face_areas)
        inside = (np.abs(offset) <= tolerance * np.maximum(scale, 1.0)) & np.all(bary >= -tolerance, axis=1)
        hits = np.flatnonzero(inside)
        if not len(hits):
            raise MeshRefusal("point_outside_face", "The point does not lie on any face of the mesh")
        best = hits[np.argmax(np.min(bary[hits], axis=1))]
        return int(best), bary[best]

    def locate_ray(self, direction):
        """Face and point where the ray from the origin along ``direction`` meets the mesh (star-shaped meshes)."""
        direction = np.asarray(direction, dtype=float)
        p0, p1, p2 = (self.vertices[self.faces[:, k]] for k in range(3))
        e1, e2 = p1 - p0, p2 - p0
        pvec = np.cross(direction[None, :], e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            tvec = -p0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            qvec = np.cross(tvec, e1)
            v = (qvec @ direction) * inv
            t = np.einsum("ij,ij->i", e2, qvec) * inv
        ok = (np.abs(det) > 1e-300) & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 0)
        hits = np.flatnonzero(ok)
        if not len(hits):
            raise MeshRefusal("point_outside_face", "The ray from the origin does not meet the mesh")
        margin = np.minimum(np.minimum(u[hits], v[hits]), 1 - u[hits] - v[hits])
        face = int(hits[np.argmax(margin)])
        return face, float(t[face]) * direction

    def _barycentric(self, face_ids, points):
        v = self.vertices[self.faces[face_ids]]
        e1, e2, d = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], points - v[:, 0]
        d11, d12, d22 = (np.einsum("ij,ij->i", x, y) for x, y in ((e1, e1), (e1, e2), (e2, e2)))
        d1, d2 = np.einsum("ij,ij->i", d, e1), np.einsum("ij,ij->i", d, e2)
        den = d11 * d22 - d12 * d12
        b1 = (d22 * d1 - d12 * d2) / den
        b2 = (d11 * d2 - d12 * d1) / den
        return np.stack([1 - b1 - b2, b1, b2], axis=1)

    def barycentric(self, face, point) -> np.ndarray:
        return self._barycentric(np.array([face]), np.asarray(point, dtype=float)[None, :])[0]

    def point(self, face, bary) -> np.ndarray:
        return np.asarray(bary, dtype=float) @ self.vertices[self.faces[face]]

    def tangent(self, face, vector) -> np.ndarray:
        """Declared projection of a 3D vector onto a face plane, normalized."""
        n = self.face_normals[face]
        projected = np.asarray(vector, dtype=float) - (np.dot(vector, n)) * n
        norm = np.linalg.norm(projected)
        if not norm > 1e-12:
            raise MeshRefusal("invalid_direction", "The vector has no component in the face plane")
        return projected / norm

    def rotate_in_face(self, face, direction, angle) -> np.ndarray:
        n = self.face_normals[face]
        return math.cos(angle) * direction + math.sin(angle) * np.cross(n, direction)


# ---------------------------------------------------------------- generators
def icosahedron():
    t = (1 + math.sqrt(5)) / 2
    vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                         [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                      [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return vertices / np.linalg.norm(vertices, axis=1)[:, None], faces


def icosphere(level: int, radius: float = 1.0) -> TriMesh:
    """Loop-style 1:4 midpoint subdivision of the icosahedron, projected to the sphere."""
    vertices, faces = icosahedron()
    for _ in range(level):
        n = len(vertices)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        key = np.minimum(edges[:, 0], edges[:, 1]) * n + np.maximum(edges[:, 0], edges[:, 1])
        unique, inverse = np.unique(key, return_inverse=True)
        lo, hi = unique // n, unique % n
        mid = vertices[lo] + vertices[hi]
        vertices = np.concatenate([vertices, mid / np.linalg.norm(mid, axis=1)[:, None]])
        m = len(faces)
        ab, bc, ca = (n + inverse[i * m:(i + 1) * m] for i in range(3))
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        faces = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1), np.stack([c, ca, bc], 1),
                                np.stack([ab, bc, ca], 1)])
    return TriMesh(radius * vertices, faces, f"icosphere-{level}", {"level": level, "radius": radius})


def uv_sphere(n_lat: int, n_lon: int, radius: float = 1.0, twist: float = 0.0) -> TriMesh:
    """Latitude-longitude sphere with poles; ``twist`` rotates ring j by j * twist radians."""
    rings = []
    for j in range(1, n_lat + 1):
        theta = math.pi * j / (n_lat + 1)
        phi = 2 * math.pi * np.arange(n_lon) / n_lon + twist * j
        rings.append(np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                               np.full(n_lon, math.cos(theta))], 1))
    vertices = np.concatenate([[[0.0, 0.0, 1.0]], *rings, [[0.0, 0.0, -1.0]]])
    south = len(vertices) - 1

    def ring(j, i):
        return 1 + j * n_lon + (i % n_lon)

    faces = [[0, ring(0, i), ring(0, i + 1)] for i in range(n_lon)]
    for j in range(n_lat - 1):
        for i in range(n_lon):
            faces.append([ring(j, i), ring(j + 1, i), ring(j + 1, i + 1)])
            faces.append([ring(j, i), ring(j + 1, i + 1), ring(j, i + 1)])
    faces += [[south, ring(n_lat - 1, i + 1), ring(n_lat - 1, i)] for i in range(n_lon)]
    return TriMesh(radius * vertices, np.array(faces), f"uv-sphere-{n_lat}x{n_lon}",
                   {"n_lat": n_lat, "n_lon": n_lon, "twist": twist, "radius": radius})


def cylinder_mesh(n: int, m: int, radius: float = 1.0, height: float = 1.0, lantern: bool = False) -> TriMesh:
    """Open cylinder: m + 1 rings of n vertices. ``lantern`` offsets odd rings by pi / n (Schwarz lantern).

    Without the offset each band is split into planar rectangles, so the mesh is
    intrinsically a flat strip of circumference 2 n R sin(pi / n).
    """
    j = np.arange(m + 1)
    i = np.arange(n)
    phi = 2 * math.pi * i[None, :] / n + (lantern * (j[:, None] % 2)) * math.pi / n
    z = np.broadcast_to((height * j / m)[:, None], phi.shape)
    vertices = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=-1).reshape(-1, 3)
    band = np.repeat(np.arange(m), n)[:, None]
    k = np.tile(np.arange(n), m)[:, None]
    base, top, k1 = band * n, (band + 1) * n, (k + 1) % n
    # Lantern even bands: apex of the upward triangle sits between two base vertices.
    offset = (lantern & (band % 2 == 0))
    first = np.where(offset, np.hstack([base + k, base + k1, top + k]), np.hstack([base + k, base + k1, top + k1]))
    second = np.where(offset, np.hstack([base + k1, top + k1, top + k]), np.hstack([base + k, top + k1, top + k]))
    faces = np.stack([first, second], axis=1).reshape(-1, 3)
    name = "schwarz-lantern" if lantern else "prism-cylinder"
    return TriMesh(vertices, faces, f"{name}-{n}x{m}",
                   {"n": n, "m": m, "radius": radius, "height": height, "lantern": lantern})


def plane_mesh(nx: int, ny: int, width: float = 1.0, height: float = 1.0, shear: float = 0.0) -> TriMesh:
    x, y = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    vertices = np.stack([x + shear * y, y, np.zeros_like(x)], axis=-1).reshape(-1, 3)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i
            faces += [[a, b, c], [a, c, d]]
    return TriMesh(vertices, np.array(faces), f"plane-{nx}x{ny}",
                   {"nx": nx, "ny": ny, "width": width, "height": height, "shear": shear})


def torus_mesh(n_phi: int, n_theta: int, major: float = 2.0, minor: float = 1.0) -> TriMesh:
    """Closed torus grid in (phi, theta); vertex (i, j) sits at phi_i = 2 pi i / n_phi, theta_j = 2 pi j / n_theta."""
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    theta = 2 * math.pi * np.arange(n_theta) / n_theta
    p, t = np.meshgrid(phi, theta, indexing="ij")
    rho = major + minor * np.cos(t)
    vertices = np.stack([rho * np.cos(p), rho * np.sin(p), minor * np.sin(t)], axis=-1).reshape(-1, 3)

    def v(i, j):
        return (i % n_phi) * n_theta + (j % n_theta)

    faces = []
    for i in range(n_phi):
        for j in range(n_theta):
            faces += [[v(i, j), v(i + 1, j), v(i + 1, j + 1)], [v(i, j), v(i + 1, j + 1), v(i, j + 1)]]
    return TriMesh(vertices, np.array(faces), f"torus-{n_phi}x{n_theta}",
                   {"n_phi": n_phi, "n_theta": n_theta, "major": major, "minor": minor})


def jittered(mesh: TriMesh, amplitude: float, seed: int, radius: float = 1.0) -> TriMesh:
    """Tangential Gaussian jitter of a sphere mesh (std amplitude * mean edge), re-projected to the sphere."""
    rng = np.random.Generator(np.random.PCG64(seed))
    x = mesh.vertices / radius
    noise = rng.standard_normal(x.shape) * amplitude * mesh.mean_edge() / radius
    noise -= np.einsum("ij,ij->i", noise, x)[:, None] * x
    moved = x + noise
    moved = radius * moved / np.linalg.norm(moved, axis=1)[:, None]
    return TriMesh.build(moved, mesh.faces, f"{mesh.name}-jitter{amplitude:g}-s{seed}",
                         params={**mesh.params, "jitter": amplitude, "seed": seed})


# ---------------------------------------------------------------- tracing
@dataclass
class Trace:
    """A straightest geodesic: polyline, visited faces and its outcome."""

    status: str
    message: str
    points: list
    faces: list
    length: float
    end_face: int
    end_point: np.ndarray
    end_direction: np.ndarray
    min_vertex_margin: float

    @property
    def completed(self) -> bool:
        return self.status == "completed"


def trace(mesh: TriMesh, face: int, point, direction, length: float, *, vertex_tolerance=VERTEX_TOLERANCE,
          max_steps: int | None = None, strict: bool = False) -> Trace:
    """Straightest geodesic of ``length`` from ``point`` in ``face`` along a tangent ``direction``.

    Straight inside a face; across an edge the direction keeps its component
    along the edge and its perpendicular magnitude (equal angles on both sides,
    i.e. rotation about the edge onto the next face). ``strict`` raises the
    refusal instead of returning a partial trace.
    """
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    ids = mesh.faces[face]
    corners = mesh.vertices[ids]
    longest = max(np.linalg.norm(corners[k] - corners[(k + 1) % 3]) for k in range(3))
    normal = mesh.face_normals[face]
    bary = mesh.barycentric(face, point)
    if abs(np.dot(point - corners[0], normal)) > ON_FACE_TOLERANCE * longest or np.min(bary) < -ON_FACE_TOLERANCE:
        return _refuse(strict, "point_outside_face", "The start point is not on the start face",
                       [point], [], 0.0, face, point, direction, math.inf)
    norm = np.linalg.norm(direction)
    if not (norm > 1e-12 and np.all(np.isfinite(direction))) or abs(np.dot(direction, normal)) > 1e-9 * norm:
        return _refuse(strict, "invalid_direction", "The start direction must be a nonzero tangent of the start face",
                       [point], [], 0.0, face, point, direction, math.inf)
    direction = direction / norm
    max_steps = max_steps or 50 * len(mesh.faces) + 100
    remaining, travelled = float(length), 0.0
    points, faces = [point.copy()], [int(face)]
    entry = -1
    margin = math.inf
    for _ in range(max_steps):
        ids = mesh.faces[face]
        corners = mesh.vertices[ids]
        origin = corners[0]
        e1 = corners[1] - origin
        e1 /= np.linalg.norm(e1)
        nx, ny, nz = mesh.face_normals[face]
        e2 = np.array([ny * e1[2] - nz * e1[1], nz * e1[0] - nx * e1[2], nx * e1[1] - ny * e1[0]])
        local = np.stack([(corners - origin) @ e1, (corners - origin) @ e2], axis=1)
        q = np.array([(point - origin) @ e1, (point - origin) @ e2])
        w = np.array([direction @ e1, direction @ e2])
        best = None
        for k in range(3):
            if k == entry:
                continue
            a, edge = local[k], local[(k + 1) % 3] - local[k]
            den = w[0] * edge[1] - w[1] * edge[0]
            if den <= 0.0:  # counterclockwise face: only edges the ray leaves through
                continue
            rel = a - q
            t = (rel[0] * edge[1] - rel[1] * edge[0]) / den
            s = (rel[0] * w[1] - rel[1] * w[0]) / den
            if best is None or t < best[0]:
                best = (max(t, 0.0), k, min(max(s, 0.0), 1.0))
        if best is None:
            return _refuse(strict, "invalid_direction", "No exit edge: the direction is not inside the face",
                           points, faces, travelled, face, point, direction, margin)
        t, k, s = best
        if t >= remaining:
            end = origin + (q[0] + remaining * w[0]) * e1 + (q[1] + remaining * w[1]) * e2
            points.append(end)
            return Trace("completed", "", points, faces, travelled + remaining, int(face), end,
                         w[0] * e1 + w[1] * e2, margin)
        a3, b3 = corners[k], corners[(k + 1) % 3]
        crossing = a3 + s * (b3 - a3)
        travelled += t
        remaining -= t
        points.append(crossing)
        margin = min(margin, s, 1.0 - s)
        if min(s, 1.0 - s) <= vertex_tolerance:
            return _refuse(strict, "vertex_hit", f"The geodesic reaches vertex {int(ids[k] if s < 0.5 else ids[(k + 1) % 3])}"
                           " where the straightest continuation is not unique", points, faces, travelled, face,
                           crossing, direction, margin)
        nxt = int(mesh.neighbors[face, k])
        if nxt < 0:
            return _refuse(strict, "boundary_reached", "The geodesic reached a boundary edge before its length",
                           points, faces, travelled, face, crossing, direction, margin)
        axis = (b3 - a3) / np.linalg.norm(b3 - a3)
        d3 = w[0] * e1 + w[1] * e2
        along = float(d3 @ axis)
        across = math.sqrt(max(1.0 - along * along, 0.0))
        shared = {int(ids[k]), int(ids[(k + 1) % 3])}
        nxt_ids = mesh.faces[nxt]
        apex = mesh.vertices[[v for v in nxt_ids if int(v) not in shared][0]] - a3
        inward = apex - (apex @ axis) * axis
        inward /= np.linalg.norm(inward)
        direction = along * axis + across * inward
        direction /= np.linalg.norm(direction)
        entry = int(np.flatnonzero(mesh.neighbors[nxt] == face)[0])
        face, point = nxt, crossing
        faces.append(face)
    return _refuse(strict, "step_budget_exceeded", "The trace exceeded its face-crossing budget",
                   points, faces, travelled, face, point, direction, margin)


def _refuse(strict, code, message, points, faces, travelled, face, point, direction, margin):
    if strict:
        raise MeshRefusal(code, message)
    return Trace(code, message, list(points), list(faces), float(travelled), int(face), np.asarray(point),
                 np.asarray(direction), margin)


# ---------------------------------------------------------------- graph distances
def _csr(n, a, b, w):
    a, b, w = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([w, w])
    order = np.lexsort((b, a))
    a, b, w = a[order], b[order], w[order]
    indptr = np.searchsorted(a, np.arange(n + 1))
    return indptr, b, w


def dijkstra(indptr, indices, weights, source, target=None):
    """Heap-based single-source shortest paths (all nonnegative weights)."""
    n = len(indptr) - 1
    dist = [math.inf] * n
    dist[source] = 0.0
    done = [False] * n
    ptr, idx, wts = indptr.tolist(), indices.tolist(), weights.tolist()
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == target:
            break
        for e in range(ptr[u], ptr[u + 1]):
            v = idx[e]
            nd = d + wts[e]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return np.array(dist)


def edge_graph(mesh: TriMesh):
    return _csr(len(mesh.vertices), mesh.edges[:, 0], mesh.edges[:, 1], mesh.edge_lengths)


def edge_distances(mesh: TriMesh, source: int) -> np.ndarray:
    """Shortest edge-path lengths: an upper bound on the polyhedral geodesic distance."""
    return dijkstra(*edge_graph(mesh), source)


def steiner_graph(mesh: TriMesh, k: int, extra=()):
    """Graph on vertices plus k equally spaced points per edge, complete inside every face.

    ``extra`` = [(face, point)] adds off-vertex nodes joined to their face's
    boundary nodes. Every graph edge is a straight segment inside one face, so
    every graph path is a surface path and graph distances are upper bounds on
    polyhedral distances. With k = 2^j - 1 the node sets are nested, so the
    distances cannot increase as j grows.
    """
    n, n_edges = len(mesh.vertices), len(mesh.edges)
    fractions = (np.arange(1, k + 1) / (k + 1)) if k else np.zeros(0)
    a, b = mesh.vertices[mesh.edges[:, 0]], mesh.vertices[mesh.edges[:, 1]]
    steiner = (a[:, None, :] + fractions[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)
    nodes = np.concatenate([mesh.vertices, steiner])
    ids = [mesh.faces]
    for local in range(3):
        # Node order inside a face is irrelevant: every pair of its nodes is joined.
        ids.append(n + mesh.edge_of_half[:, local][:, None] * k + np.arange(k)[None, :])
    face_nodes = np.concatenate(ids, axis=1)
    i, j = np.triu_indices(face_nodes.shape[1], 1)
    src, dst = face_nodes[:, i].ravel(), face_nodes[:, j].ravel()
    extra_ids = []
    for face, point in extra:
        new = len(nodes)
        nodes = np.concatenate([nodes, np.asarray(point, dtype=float)[None, :]])
        src = np.concatenate([src, np.full(face_nodes.shape[1], new)])
        dst = np.concatenate([dst, face_nodes[face]])
        extra_ids.append(new)
    weights = np.linalg.norm(nodes[src] - nodes[dst], axis=1)
    keep = src != dst
    return _csr(len(nodes), src[keep], dst[keep], weights[keep]), extra_ids, n_edges * k


def point_distance(mesh: TriMesh, k: int, start, end) -> float:
    """Steiner-graph distance between two (face, point) locations; refuses unreachable targets."""
    (indptr, indices, weights), (s, t), _ = steiner_graph(mesh, k, extra=(start, end))
    value = float(dijkstra(indptr, indices, weights, s, t)[t])
    if not math.isfinite(value):
        raise MeshRefusal("unreachable_target", "The target lies in another component of the mesh")
    return value


def vertex_distance(mesh: TriMesh, source: int, target: int) -> float:
    value = float(edge_distances(mesh, source)[target])
    if not math.isfinite(value):
        raise MeshRefusal("unreachable_target", "The target lies in another component of the mesh")
    return value


# ---------------------------------------------------------------- operators
def cotangent_weights(mesh: TriMesh):
    """Per face corner cot(angle); the corner opposite edge (k+1, k+2)."""
    v = mesh.vertices[mesh.faces]
    cot = np.empty(mesh.faces.shape)
    for k in range(3):
        a = v[:, (k + 1) % 3] - v[:, k]
        b = v[:, (k + 2) % 3] - v[:, k]
        cot[:, k] = np.einsum("ij,ij->i", a, b) / np.linalg.norm(np.cross(a, b), axis=1)
    return cot


def heat_distance(mesh: TriMesh, source: int, t_factor: float = 1.0, max_vertices: int = 3000) -> np.ndarray:
    """Heat-method distance (Crane, Weischedel and Wardetzky 2013) with dense solves; t = t_factor h^2."""
    n = len(mesh.vertices)
    if n > max_vertices:
        raise MeshRefusal("invalid_shape", f"Dense heat method limited to {max_vertices} vertices, mesh has {n}")
    cot = cotangent_weights(mesh)
    lap = np.zeros((n, n))
    for k in range(3):
        i, j = mesh.faces[:, (k + 1) % 3], mesh.faces[:, (k + 2) % 3]
        w = 0.5 * cot[:, k]
        np.add.at(lap, (i, j), -w)
        np.add.at(lap, (j, i), -w)
        np.add.at(lap, (i, i), w)
        np.add.at(lap, (j, j), w)
    mass = np.bincount(mesh.faces.ravel(), weights=np.repeat(mesh.face_areas / 3, 3), minlength=n)
    t = t_factor * mesh.mean_edge() ** 2
    delta = np.zeros(n)
    delta[source] = 1.0
    u = np.linalg.solve(np.diag(mass) + t * lap, delta)
    v = mesh.vertices[mesh.faces]
    grad = np.zeros((len(mesh.faces), 3))
    for k in range(3):
        opposite = v[:, (k + 2) % 3] - v[:, (k + 1) % 3]
        grad += u[mesh.faces[:, k]][:, None] * np.cross(mesh.face_normals, opposite)
    grad /= (2 * mesh.face_areas)[:, None]
    field_ = -grad / np.linalg.norm(grad, axis=1)[:, None]
    div = np.zeros(n)
    for k in range(3):
        i = mesh.faces[:, k]
        e1 = v[:, (k + 1) % 3] - v[:, k]
        e2 = v[:, (k + 2) % 3] - v[:, k]
        # cot of the angle opposite e1 sits at corner k+2, opposite e2 at corner k+1
        contribution = 0.5 * (cot[:, (k + 2) % 3] * np.einsum("ij,ij->i", e1, field_)
                              + cot[:, (k + 1) % 3] * np.einsum("ij,ij->i", e2, field_))
        np.add.at(div, i, contribution)
    keep = np.arange(n) != source
    phi = np.zeros(n)
    phi[keep] = np.linalg.solve(lap[np.ix_(keep, keep)], -div[keep])
    return phi - phi[source]


# ---------------------------------------------------------------- curvature and normals
def angle_defect(mesh: TriMesh):
    """(2 pi - angle sum) and barycentric area A/3 at every vertex; boundary vertices flagged."""
    angles = mesh.corner_angles()
    n = len(mesh.vertices)
    total = np.bincount(mesh.faces.ravel(), weights=angles.ravel(), minlength=n)
    area = np.bincount(mesh.faces.ravel(), weights=np.repeat(mesh.face_areas / 3, 3), minlength=n)
    return 2 * math.pi - total, area, ~mesh.boundary_vertices


def angle_defect_curvature(mesh: TriMesh, vertex: int | None = None):
    """K_v = (2 pi - sum of angles) / (A_v / 3). Boundary vertices are refused (or masked for arrays)."""
    defect, area, interior = angle_defect(mesh)
    if vertex is not None:
        if not interior[vertex]:
            raise MeshRefusal("boundary_vertex_curvature",
                              f"Vertex {vertex} is on the boundary where the angle defect is not a curvature")
        return float(defect[vertex] / area[vertex])
    return defect / area, interior


def mixed_voronoi_area(mesh: TriMesh) -> np.ndarray:
    """Meyer-Desbrun-Schroeder-Barr mixed area (Voronoi for non-obtuse faces)."""
    v = mesh.vertices[mesh.faces]
    cot = cotangent_weights(mesh)
    angles = mesh.corner_angles()
    obtuse = angles > math.pi / 2
    contribution = np.zeros(mesh.faces.shape)
    for k in range(3):
        e_next = np.sum((v[:, (k + 1) % 3] - v[:, k]) ** 2, axis=1)
        e_prev = np.sum((v[:, (k + 2) % 3] - v[:, k]) ** 2, axis=1)
        contribution[:, k] = (e_prev * cot[:, (k + 1) % 3] + e_next * cot[:, (k + 2) % 3]) / 8
    any_obtuse = obtuse.any(axis=1)
    contribution[any_obtuse] = np.where(obtuse[any_obtuse], mesh.face_areas[any_obtuse, None] / 2,
                                        mesh.face_areas[any_obtuse, None] / 4)
    return np.bincount(mesh.faces.ravel(), weights=contribution.ravel(), minlength=len(mesh.vertices))


def vertex_normals(mesh: TriMesh) -> np.ndarray:
    """Area-weighted vertex normals (sum of face cross products)."""
    weighted = mesh.face_normals * (2 * mesh.face_areas)[:, None]
    normals = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(normals, mesh.faces[:, k], weighted)
    return normals / np.linalg.norm(normals, axis=1)[:, None]


# ---------------------------------------------------------------- batched local functions
def batch_angle_defect_curvature(positions, center_index, ring_faces):
    """Angle-defect curvature at one vertex for a batch of one-ring positions (N, r, 3).

    ``ring_faces`` lists the incident faces as local index triples.
    """
    total = np.zeros(positions.shape[0])
    area = np.zeros(positions.shape[0])
    for tri in ring_faces:
        p = positions[:, tri]
        local = list(tri).index(center_index)
        a = p[:, (local + 1) % 3] - p[:, local]
        b = p[:, (local + 2) % 3] - p[:, local]
        cross = np.linalg.norm(np.cross(a, b), axis=1)
        total += np.arctan2(cross, np.einsum("ij,ij->i", a, b))
        area += cross / 6.0
    return (2 * math.pi - total) / area


def batch_vertex_normal(positions, ring_faces):
    normal = np.zeros((positions.shape[0], 3))
    for tri in ring_faces:
        p = positions[:, tri]
        normal += np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return normal / np.linalg.norm(normal, axis=1)[:, None]


def one_ring(mesh: TriMesh, vertex: int):
    """Local vertex ids (center first) and incident faces as local index triples."""
    incident = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
    ids = [vertex] + sorted(set(mesh.faces[incident].ravel().tolist()) - {vertex})
    local = {v: i for i, v in enumerate(ids)}
    return np.array(ids), [tuple(local[int(v)] for v in mesh.faces[f]) for f in incident]


def strip_unfold_distance(positions, strip, start_bary, end_bary):
    """Planar distance between two markers after unfolding a face strip, batched over positions (N, nv, 3).

    ``strip`` is a list of local vertex-index triples, consecutive faces sharing
    an edge. Returns (distance, margin): margin is the smallest edge-parameter
    distance of the straight unfolded segment from the shared edges' endpoints
    (negative when the segment leaves the strip, where the unfolded distance is
    not a surface path length).
    """
    batch = positions.shape[0]

    def length(i, j):
        return np.linalg.norm(positions[:, i] - positions[:, j], axis=1)

    a, b, c = strip[0]
    placed = {a: np.zeros((batch, 2))}
    lab = length(a, b)
    placed[b] = np.stack([lab, np.zeros(batch)], 1)
    placed[c] = _third(placed[a], placed[b], length(a, c), length(b, c), side=1.0)
    layers = [dict(placed)]
    shared_edges = []
    for previous, current in zip(strip[:-1], strip[1:]):
        common = [v for v in current if v in previous]
        if len(common) != 2:
            raise MeshRefusal("invalid_shape", "Consecutive strip faces must share exactly one edge")
        old = [v for v in previous if v not in common][0]
        new = [v for v in current if v not in common][0]
        prev_layer = layers[-1]
        u, v = common
        pu, pv, po = prev_layer[u], prev_layer[v], prev_layer[old]
        side_old = np.sign(_cross2(pv - pu, po - pu))
        layer = {u: pu, v: pv, new: _third(pu, pv, length(u, new), length(v, new), side=-side_old)}
        layers.append(layer)
        shared_edges.append((pu, pv))
    start = sum(start_bary[k] * layers[0][v] for k, v in enumerate(strip[0]))
    end = sum(end_bary[k] * layers[-1][v] for k, v in enumerate(strip[-1]))
    distance = np.linalg.norm(end - start, axis=1)
    margin = np.full(batch, np.inf)
    direction = end - start
    for pu, pv in shared_edges:
        edge = pv - pu
        den = _cross2(direction, edge)
        s = _cross2(pu - start, direction) / den
        margin = np.minimum(margin, np.minimum(s, 1 - s))
    return distance, margin


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _third(pa, pb, ra, rb, side):
    """Point at distances ra from pa and rb from pb, on the given side of pa->pb."""
    base = pb - pa
    d = np.linalg.norm(base, axis=1)
    x = (d * d + ra * ra - rb * rb) / (2 * d)
    y = np.sqrt(np.maximum(ra * ra - x * x, 0.0))
    ex = base / d[:, None]
    ey = np.stack([-ex[:, 1], ex[:, 0]], 1)
    side = np.broadcast_to(np.asarray(side, dtype=float), d.shape)
    return pa + x[:, None] * ex + (side * y)[:, None] * ey
