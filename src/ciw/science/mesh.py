"""Triangle-mesh surfaces and geodesic distance on them (capability ``mesh_surfaces``).

Scope
-----
``TriangleMesh`` admits an immutable mesh only if every coordinate is finite,
every face names three distinct in-range vertices, no face repeats another's
vertex set, no edge is shared by more than two faces, every vertex is referenced
and no face is degenerate (area <= ``DEGENERATE_RELATIVE_AREA`` x its longest
edge squared). Invalid input is refused with a stable code; it is never welded,
re-indexed, re-oriented or re-triangulated. Orientation is not required (a
Moebius band is admitted). Generators sample a plane patch, an open cylinder
patch or closed tube, an icosphere and a torus, and return each vertex's
intrinsic parameters so analytic references can be evaluated.

Two estimators give the distance from one source vertex to every vertex:

* ``edge_graph_distance``: Dijkstra over edges with Euclidean lengths. Every edge
  path lies on the polyhedral surface, so it bounds the *polyhedral* geodesic
  distance from above. It is not a bound on the smooth surface (chords are
  shorter than arcs) and it does not converge under refinement: on a grid with
  one diagonal per cell the metrication error reaches 41 % (sqrt 2).
* ``heat_method_distance``: Crane, Weischedel & Wardetzky, "Geodesics in heat"
  (ACM TOG 2013), with the conventions stated on the function.

Limits
------
* Dense numpy solves: O(n^2) memory and O(n^3) time, refused above
  ``MAX_DENSE_VERTICES`` vertices in the source's component.
* The heat method is an approximation, neither an upper nor a lower bound, and
  values are not clipped (they may dip below zero next to the source). Its
  error shrinks under refinement on regular meshes but it smooths the distance
  near the source and the cut locus; averaged Neumann/Dirichlet boundary
  conditions reduce boundary bias without removing it. Poor (obtuse) triangles
  degrade it; no intrinsic Delaunay retriangulation is done.
* No exact polyhedral geodesics (MMP / Chen-Han) and no geodesic paths: only
  distances from a single vertex.
* Analytic references describe the smooth surface; the mesh is inscribed, so
  even an exact polyhedral solver differs from them by O(h^2). The ``polyhedral``
  reference is exact for the developable (plane and cylinder) meshes only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Any

import numpy as np

from ._common import Refusal, content_identity, finite, integer, plain, require_keys
from .topology import _admit, edge_table, face_array
from .units import require_dimension

SCHEMA = "ciw.triangle-mesh.v1"
CAPABILITY = "mesh_surfaces"
MAX_VERTICES = 20_000
MAX_FACES = 3 * MAX_VERTICES
MAX_DENSE_VERTICES = 3_000
MAX_COORDINATE = 1e100
DEGENERATE_RELATIVE_AREA = 1e-12
MAX_T_FACTOR = 1e4
MAX_SUBDIVISIONS = 8
BOUNDARY_CONDITIONS = ("average", "neumann", "dirichlet")
REFERENCE_MODELS = ("smooth", "polyhedral")
SURFACE_KINDS = ("plane", "cylinder_patch", "cylinder_tube", "sphere", "torus")
TWO_PI = 2.0 * math.pi


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class TriangleMesh:
    """Validated, immutable triangle mesh. ``vertices`` (n, 3) float, ``faces`` (m, 3) int64."""

    def __init__(self, vertices: Any, faces: Any, *, length_unit: str = "m"):
        self.length_unit = require_dimension(length_unit, "m", "length_unit").expression
        try:
            coordinates = _admit(vertices, "vertices", "malformed_mesh", MAX_VERTICES, integral=False)
        except Refusal as refusal:
            if refusal.code == "oversized_input":
                raise Refusal("mesh_too_large", f"meshes are bounded to {MAX_VERTICES} vertices",
                              limit=MAX_VERTICES) from refusal
            raise
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 3:
            raise Refusal("malformed_mesh", "vertices must be an (n, 3) array with n >= 3",
                          shape=list(coordinates.shape))
        if len(coordinates) > MAX_VERTICES:
            raise Refusal("mesh_too_large", f"meshes are bounded to {MAX_VERTICES} vertices",
                          vertices=len(coordinates), limit=MAX_VERTICES)
        bad = ~np.isfinite(coordinates).all(axis=1)
        if bad.any():
            raise Refusal("non_finite_vertex", f"vertex {int(np.argmax(bad))} has a non-finite coordinate",
                          vertex=int(np.argmax(bad)))
        if np.abs(coordinates).max() > MAX_COORDINATE:
            raise Refusal("out_of_domain", f"coordinates must satisfy |x| <= {MAX_COORDINATE}")
        try:
            triangles = face_array(faces, len(coordinates), limit=MAX_FACES)
        except Refusal as refusal:
            if refusal.code == "oversized_input":
                raise Refusal("mesh_too_large", f"meshes are bounded to {MAX_FACES} faces", limit=MAX_FACES) from refusal
            raise
        unused = np.bincount(triangles.ravel(), minlength=len(coordinates)) == 0
        if unused.any():
            raise Refusal("unreferenced_vertex", f"vertex {int(np.argmax(unused))} belongs to no face",
                          vertex=int(np.argmax(unused)), count=int(unused.sum()))
        _, first, repeats = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True, return_counts=True)
        if (repeats > 1).any():
            raise Refusal("duplicate_face", "two faces share the same three vertices",
                          face=int(first[np.argmax(repeats > 1)]))
        edges, inverse, counts = edge_table(triangles)
        if (counts > 2).any():
            bad_edge = edges[int(np.argmax(counts > 2))]
            raise Refusal("non_manifold_edge", f"edge {bad_edge.tolist()} is shared by more than two faces",
                          edge=bad_edge, faces=int(counts.max()))
        p0, p1, p2 = (coordinates[triangles[:, k]] for k in range(3))
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        longest = np.max([((p1 - p0) ** 2).sum(1), ((p2 - p1) ** 2).sum(1), ((p0 - p2) ** 2).sum(1)], axis=0)
        thin = areas <= DEGENERATE_RELATIVE_AREA * longest
        if thin.any():
            index = int(np.argmax(thin))
            raise Refusal("degenerate_face", f"face {index} has (near-)zero area", face=index,
                          area=float(areas[index]), longest_edge=float(math.sqrt(longest[index])))
        self.vertices = _read_only(np.array(coordinates, dtype=float))
        self.faces = _read_only(triangles)
        self.edges = _read_only(edges)
        self.edge_face_counts = _read_only(counts)
        self.face_areas = _read_only(areas)
        self.edge_lengths = _read_only(np.linalg.norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1))
        boundary = np.zeros(len(coordinates), dtype=bool)
        boundary[edges[counts == 1].ravel()] = True
        self.boundary_vertices = _read_only(boundary)
        self._adjacency: tuple[list[int], list[int], list[float]] | None = None

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))

    @property
    def mean_edge_length(self) -> float:
        return float(self.edge_lengths.mean())

    def adjacency(self) -> tuple[list[int], list[int], list[float]]:
        """Compressed adjacency (offsets, neighbours, edge lengths) as Python lists."""
        if self._adjacency is None:
            tails = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
            heads = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
            weights = np.concatenate([self.edge_lengths, self.edge_lengths])
            order = np.argsort(tails, kind="stable")
            offsets = np.concatenate([[0], np.cumsum(np.bincount(tails, minlength=self.vertex_count))])
            self._adjacency = (offsets.tolist(), heads[order].tolist(), weights[order].tolist())
        return self._adjacency

    # -------------------------------------------------------------- persistence
    def to_json(self) -> dict:
        return plain({"schema": SCHEMA, "length_unit": self.length_unit, "vertices": self.vertices,
                      "faces": self.faces})

    @classmethod
    def from_json(cls, record: dict) -> "TriangleMesh":
        record = require_keys(record, "triangle mesh", {"schema", "length_unit", "vertices", "faces"})
        if record["schema"] != SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {SCHEMA}", schema=str(record["schema"]))
        return cls(record["vertices"], record["faces"], length_unit=record["length_unit"])

    def identity(self) -> str:
        return content_identity(self.to_json())


def _require_mesh(mesh: Any) -> TriangleMesh:
    if not isinstance(mesh, TriangleMesh):
        raise Refusal("malformed_mesh", "expected a TriangleMesh")
    return mesh


def _vertex(mesh: TriangleMesh, source: Any) -> int:
    return integer(source, "source", minimum=0, maximum=mesh.vertex_count - 1)


# ------------------------------------------------------------------ generators
@dataclass(frozen=True, eq=False)
class SampledSurface:
    """A generated mesh with the intrinsic parameters of every vertex.

    ``developed`` is an isometric development of the *polyhedral* surface for the
    developable kinds (plane, cylinder patch/tube), with period ``period`` along its
    first axis for the closed tube; ``None`` otherwise.
    """

    mesh: TriangleMesh
    kind: str
    parameters: np.ndarray
    parameter_names: tuple[str, str]
    constants: dict = field(default_factory=dict)
    developed: np.ndarray | None = None
    period: float | None = None

    def reference_distance(self, source: Any, model: str = "smooth") -> np.ndarray:
        """Exact geodesic distance from vertex ``source`` to every vertex.

        ``smooth``: on the sampled smooth surface (plane: Euclidean; cylinder patch:
        unrolled ``sqrt((R dphi)^2 + dz^2)``, valid because the unrolled rectangle is
        convex; closed tube: the same with ``dphi`` taken the short way round; sphere:
        ``R * central angle``). ``polyhedral``: on the mesh itself, from the flat
        development (plane and cylinder kinds only). Torus: refused, no closed form.
        """
        index = _vertex(self.mesh, source)
        if model not in REFERENCE_MODELS:
            raise Refusal("unknown_reference_model", f"model must be one of {list(REFERENCE_MODELS)}")
        if self.kind == "torus" or (model == "polyhedral" and self.developed is None):
            raise Refusal("no_closed_form_reference", f"no exact {model} distance is available for a {self.kind}")
        if model == "polyhedral":
            delta = self.developed - self.developed[index]
            if self.period is not None:
                delta[:, 0] = (delta[:, 0] + self.period / 2) % self.period - self.period / 2
            return np.hypot(delta[:, 0], delta[:, 1])
        delta = self.parameters - self.parameters[index]
        if self.kind == "plane":
            return np.hypot(delta[:, 0], delta[:, 1])
        radius = self.constants["radius"]
        if self.kind in ("cylinder_patch", "cylinder_tube"):
            dphi = delta[:, 0] if self.kind == "cylinder_patch" else (delta[:, 0] + math.pi) % TWO_PI - math.pi
            return np.hypot(radius * dphi, delta[:, 1])
        unit = self.mesh.vertices / radius
        cosine = unit @ unit[index]
        sine = np.linalg.norm(np.cross(unit, unit[index]), axis=1)
        return radius * np.arctan2(sine, cosine)


def _vertex_budget(count: int) -> None:
    if count > MAX_VERTICES:
        raise Refusal("mesh_too_large", f"the requested sampling needs {count} vertices (limit {MAX_VERTICES})",
                      vertices=count, limit=MAX_VERTICES)


def _grid_faces(nu: int, nv: int, wrap_u: bool = False, wrap_v: bool = False) -> np.ndarray:
    """Two counter-clockwise (in the (u, v) plane) triangles per cell; vertex (i, j) has index j * columns + i."""
    columns, rows = (nu if wrap_u else nu + 1), (nv if wrap_v else nv + 1)
    i, j = (grid.ravel() for grid in np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij"))
    i1, j1 = (i + 1) % columns, (j + 1) % rows
    a, b, c, d = j * columns + i, j * columns + i1, j1 * columns + i1, j1 * columns + i
    return np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])


def _cells(value: Any, name: str, minimum: int) -> int:
    return integer(value, name, minimum=minimum, maximum=MAX_VERTICES)


def _positive(value: Any, name: str) -> float:
    return finite(value, name, minimum=0.0, exclusive_minimum=True, maximum=MAX_COORDINATE)


def plane_patch(width: float, height: float, nx: int, ny: int, *, length_unit: str = "m") -> SampledSurface:
    """Rectangle [0, width] x [0, height] in z = 0 with nx x ny cells (two triangles each).
    Parameters are (x, y); normals point to +z."""
    width, height = _positive(width, "width"), _positive(height, "height")
    nx, ny = _cells(nx, "nx", 1), _cells(ny, "ny", 1)
    _vertex_budget((nx + 1) * (ny + 1))
    x, y = np.meshgrid(np.linspace(0.0, width, nx + 1), np.linspace(0.0, height, ny + 1))
    parameters = np.column_stack([x.ravel(), y.ravel()])
    mesh = TriangleMesh(np.column_stack([parameters, np.zeros(len(parameters))]), _grid_faces(nx, ny),
                        length_unit=length_unit)
    return SampledSurface(mesh, "plane", parameters, ("x", "y"), {"width": width, "height": height, "nx": nx, "ny": ny},
                          developed=parameters.copy())


def cylinder_patch(radius: float, height: float, angle_span: float, nu: int, nv: int, *,
                   length_unit: str = "m") -> SampledSurface:
    """Cylinder x = R cos phi, y = R sin phi, z in [0, height], phi in [0, angle_span].

    ``nu`` cells around, ``nv`` along the axis. ``angle_span < 2 pi`` gives an open
    patch (kind ``cylinder_patch``, nu + 1 vertex columns); ``angle_span == 2 pi``
    (to 1e-12 relative) gives a closed tube (kind ``cylinder_tube``, nu >= 3 columns,
    seam vertices shared). Parameters are (phi, z); normals point outward.
    """
    radius, height = _positive(radius, "radius"), _positive(height, "height")
    span = finite(angle_span, "angle_span", minimum=0.0, exclusive_minimum=True)
    closed = math.isclose(span, TWO_PI, rel_tol=1e-12, abs_tol=0.0)
    if span > TWO_PI and not closed:
        raise Refusal("out_of_domain", "angle_span must be <= 2 pi (2 pi gives a closed tube)", angle_span=span)
    nu, nv = _cells(nu, "nu", 3 if closed else 1), _cells(nv, "nv", 1)
    columns = nu if closed else nu + 1
    _vertex_budget(columns * (nv + 1))
    step = (TWO_PI if closed else span) / nu
    phi, z = np.meshgrid(np.arange(columns) * step, np.linspace(0.0, height, nv + 1))
    parameters = np.column_stack([phi.ravel(), z.ravel()])
    vertices = np.column_stack([radius * np.cos(parameters[:, 0]), radius * np.sin(parameters[:, 0]), parameters[:, 1]])
    chord = 2.0 * radius * math.sin(step / 2.0)
    developed = np.column_stack([np.rint(parameters[:, 0] / step) * chord, parameters[:, 1]])
    mesh = TriangleMesh(vertices, _grid_faces(nu, nv, wrap_u=closed), length_unit=length_unit)
    return SampledSurface(mesh, "cylinder_tube" if closed else "cylinder_patch", parameters, ("phi", "z"),
                          {"radius": radius, "height": height, "angle_span": TWO_PI if closed else span, "nu": nu, "nv": nv},
                          developed=developed, period=nu * chord if closed else None)


_ICOSAHEDRON_FACES = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
    [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
    [8, 6, 7], [9, 8, 1]], dtype=np.int64)


def icosphere(radius: float, subdivisions: int, *, length_unit: str = "m") -> SampledSurface:
    """Icosahedron split ``subdivisions`` times (each triangle into four, midpoints projected to
    the sphere): 10 * 4^s + 2 vertices. Parameters are (colatitude, longitude) in rad;
    longitude is arbitrary at the poles. Normals point outward."""
    radius = _positive(radius, "radius")
    subdivisions = integer(subdivisions, "subdivisions", minimum=0, maximum=MAX_SUBDIVISIONS)
    _vertex_budget(10 * 4 ** subdivisions + 2)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    points = np.array([[-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0], [0, -1, golden],
                       [0, 1, golden], [0, -1, -golden], [0, 1, -golden], [golden, 0, -1], [golden, 0, 1],
                       [-golden, 0, -1], [-golden, 0, 1]], dtype=float)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    faces = _ICOSAHEDRON_FACES
    for _ in range(subdivisions):
        edges, inverse, _ = edge_table(faces)
        middle = points[edges[:, 0]] + points[edges[:, 1]]
        points = np.vstack([points, middle / np.linalg.norm(middle, axis=1, keepdims=True)])
        ab, bc, ca = (len(points) - len(edges) + inverse[:, k] for k in range(3))
        a, b, c = faces.T
        faces = np.concatenate([np.stack(tri, axis=1) for tri in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    vertices = radius * points
    parameters = np.column_stack([np.arccos(np.clip(points[:, 2], -1.0, 1.0)), np.arctan2(points[:, 1], points[:, 0])])
    mesh = TriangleMesh(vertices, faces, length_unit=length_unit)
    return SampledSurface(mesh, "sphere", parameters, ("colatitude", "longitude"),
                          {"radius": radius, "subdivisions": subdivisions})


def torus_mesh(R: float, r: float, nu: int, nv: int, *, length_unit: str = "m") -> SampledSurface:
    """Torus ((R + r cos v) cos u, (R + r cos v) sin u, r sin v) with nu x nv cells (nu, nv >= 3),
    closed in both directions. Requires R > r > 0 (no self-intersection). Parameters are
    (u, v); normals point outward. No closed-form distance reference is offered."""
    R, r = _positive(R, "R"), _positive(r, "r")
    if r >= R:
        raise Refusal("out_of_domain", "a torus needs R > r (otherwise it self-intersects)", R=R, r=r)
    nu, nv = _cells(nu, "nu", 3), _cells(nv, "nv", 3)
    _vertex_budget(nu * nv)
    u, v = np.meshgrid(np.arange(nu) * TWO_PI / nu, np.arange(nv) * TWO_PI / nv)
    u, v = u.ravel(), v.ravel()
    ring = R + r * np.cos(v)
    mesh = TriangleMesh(np.column_stack([ring * np.cos(u), ring * np.sin(u), r * np.sin(v)]),
                        _grid_faces(nu, nv, wrap_u=True, wrap_v=True), length_unit=length_unit)
    return SampledSurface(mesh, "torus", np.column_stack([u, v]), ("u", "v"), {"R": R, "r": r, "nu": nu, "nv": nv})


# ------------------------------------------------------------------ distances
def edge_graph_distance(mesh: TriangleMesh, source: int) -> np.ndarray:
    """Dijkstra shortest edge-path length from ``source`` (inf for other components).

    An upper bound on the polyhedral geodesic distance (every edge path is a surface
    path); not a bound on the smooth surface's distance and not convergent under
    refinement.
    """
    mesh = _require_mesh(mesh)
    start = _vertex(mesh, source)
    offsets, neighbours, weights = mesh.adjacency()
    distance = [math.inf] * mesh.vertex_count
    distance[start] = 0.0
    done = [False] * mesh.vertex_count
    heap = [(0.0, start)]
    while heap:
        current, vertex = heapq.heappop(heap)
        if done[vertex]:
            continue
        done[vertex] = True
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            other, candidate = neighbours[slot], current + weights[slot]
            if candidate < distance[other]:
                distance[other] = candidate
                heapq.heappush(heap, (candidate, other))
    return np.array(distance)


def _cotangents(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-face cotangent of the angle at each corner (m, 3) and twice the face area (m,)."""
    p = [points[faces[:, k]] for k in range(3)]
    doubled = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]), axis=1)
    cot = np.stack([((p[(k + 1) % 3] - p[k]) * (p[(k + 2) % 3] - p[k])).sum(1) / doubled for k in range(3)], axis=1)
    return cot, doubled


def cotangent_laplacian(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Dense cotangent Laplacian, negative semidefinite: ``L_ij = (cot a_ij + cot b_ij) / 2`` for
    each edge, ``L_ii = -sum_j L_ij``; ``(L u)_i`` approximates the integral of the Laplacian of
    u over vertex i's dual cell. Natural (Neumann) boundary behaviour; constants span its
    null space on a connected mesh."""
    count = len(points)
    cot, _ = _cotangents(points, faces)
    tails = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])      # edge opposite corner 0, 1, 2
    heads = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    weight = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])
    index = np.concatenate([tails * count + heads, heads * count + tails, tails * (count + 1), heads * (count + 1)])
    return np.bincount(index, np.concatenate([weight, weight, -weight, -weight]), count * count).reshape(count, count)


def _solve(matrix: np.ndarray, rhs: np.ndarray, stage: str) -> np.ndarray:
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise Refusal("solver_failed", f"the {stage} system is singular", stage=stage) from exc
    if not np.isfinite(result).all():
        raise Refusal("solver_failed", f"the {stage} solve produced non-finite values", stage=stage)
    return result


def heat_method_distance(mesh: TriangleMesh, source: int, t_factor: float = 1.0, *,
                         boundary: str = "average") -> np.ndarray:
    """Approximate geodesic distance from vertex ``source`` by the heat method (Crane et al. 2013).

    1. Heat flow: solve ``(M - t L) u = delta_source`` with ``L`` the negative
       semidefinite cotangent Laplacian, ``M`` the lumped (barycentric: a third of
       each incident face area) mass and ``t = t_factor * h^2``, ``h`` the mesh's mean
       edge length. ``boundary``: ``"neumann"`` (natural), ``"dirichlet"`` (u = 0 on
       boundary vertices) or ``"average"`` of both (the paper's recommendation, the
       default). A Dirichlet solve for a boundary source is identically zero, so
       ``average`` then equals half the Neumann solution and ``dirichlet`` is refused
       (``heat_gradient_vanished``).
    2. Per face ``X = -grad u / |grad u|``; faces where ``grad u`` is exactly zero get
       ``X = 0`` (their direction is undefined, as on a cut locus).
    3. Integrated divergence ``div_i = 1/2 sum cot(theta_1)(e_1 . X) + cot(theta_2)(e_2 . X)``,
       which equals ``(L phi)_i`` whenever ``X`` is the gradient of a piecewise-linear phi.
    4. Solve ``L phi = div``. The constant null space is removed by pinning
       ``phi(source) = 0`` (dropping the source's row and column; the dropped equation
       is implied because ``div`` sums to zero), so no regularizer is added.

    Only the source's connected component is solved; other vertices get ``inf``.
    Refuses ``mesh_too_large_for_dense_solver`` when that component exceeds
    ``MAX_DENSE_VERTICES``.
    """
    mesh = _require_mesh(mesh)
    start = _vertex(mesh, source)
    t_factor = finite(t_factor, "t_factor", minimum=0.0, exclusive_minimum=True, maximum=MAX_T_FACTOR)
    if boundary not in BOUNDARY_CONDITIONS:
        raise Refusal("unknown_boundary_condition", f"boundary must be one of {list(BOUNDARY_CONDITIONS)}",
                      boundary=str(boundary))
    reached = np.isfinite(edge_graph_distance(mesh, start))
    keep = np.flatnonzero(reached)
    if len(keep) > MAX_DENSE_VERTICES:
        raise Refusal("mesh_too_large_for_dense_solver", f"the dense heat method is bounded to {MAX_DENSE_VERTICES} "
                      "vertices per component", vertices=int(len(keep)), limit=MAX_DENSE_VERTICES)
    local = np.full(mesh.vertex_count, -1, dtype=np.int64)
    local[keep] = np.arange(len(keep))
    faces = local[mesh.faces[reached[mesh.faces[:, 0]]]]
    points, count, s = mesh.vertices[keep], len(keep), int(local[start])

    laplacian = cotangent_laplacian(points, faces)
    cot, doubled = _cotangents(points, faces)
    mass = np.bincount(faces.ravel(), np.repeat(doubled / 6.0, 3), count)
    step = t_factor * mesh.mean_edge_length ** 2
    heat = -step * laplacian
    heat.flat[:: count + 1] += mass
    impulse = np.zeros(count)
    impulse[s] = 1.0
    on_boundary = mesh.boundary_vertices[keep]
    constrained = boundary != "neumann" and bool(on_boundary.any())   # closed surfaces: all three coincide
    u = _solve(heat, impulse, "heat") if boundary != "dirichlet" or not constrained else None
    if constrained:
        dirichlet = np.zeros(count)
        inner = np.flatnonzero(~on_boundary)
        if not on_boundary[s]:
            dirichlet[inner] = _solve(heat[np.ix_(inner, inner)], impulse[inner], "dirichlet heat")
        u = dirichlet if u is None else 0.5 * (u + dirichlet)

    p = [points[faces[:, k]] for k in range(3)]
    normal = np.cross(p[1] - p[0], p[2] - p[0]) / doubled[:, None]
    opposite = [p[2] - p[1], p[0] - p[2], p[1] - p[0]]
    gradient = sum(u[faces[:, k], None] * np.cross(normal, opposite[k]) for k in range(3)) / doubled[:, None]
    norm = np.linalg.norm(gradient, axis=1)
    if not (norm > 0).any():
        raise Refusal("heat_gradient_vanished", "the heat solution is constant (e.g. a Dirichlet source on the boundary)")
    field_x = -gradient / np.where(norm > 0, norm, 1.0)[:, None]
    divergence = np.zeros(count)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        contribution = 0.5 * (cot[:, j] * ((p[i] - p[k]) * field_x).sum(1) + cot[:, i] * ((p[j] - p[k]) * field_x).sum(1))
        divergence += np.bincount(faces[:, k], contribution, count)
    free = np.flatnonzero(np.arange(count) != s)
    phi = np.zeros(count)
    phi[free] = _solve(laplacian[np.ix_(free, free)], divergence[free], "poisson")
    result = np.full(mesh.vertex_count, math.inf)
    result[keep] = phi
    return result


def _distances(value: Any, name: str, size: int | None = None) -> np.ndarray:
    array = _admit(value, name, "malformed_distances", MAX_VERTICES, integral=False)
    if array.ndim != 1 or len(array) == 0 or (size is not None and len(array) != size):
        raise Refusal("malformed_distances", f"{name} must be a nonempty 1-D array"
                      + ("" if size is None else f" of length {size}"), shape=list(array.shape))
    if not np.isfinite(array).all():
        raise Refusal("non_finite_distance", f"{name} contains non-finite values (e.g. unreachable vertices); "
                      "select the compared vertices with mask", index=int(np.argmax(~np.isfinite(array))))
    return array


def compare_to_reference(distances: Any, exact: Any, *, mask: Any = None) -> dict:
    """Absolute and relative error statistics of ``distances`` against ``exact``.

    Relative errors use only vertices with ``exact > 0`` (so the source is excluded).
    ``mask`` (boolean, same length) selects the compared vertices; non-finite values
    among them are refused, never skipped.
    """
    estimate = _admit(distances, "distances", "malformed_distances", MAX_VERTICES, integral=False)
    reference = _admit(exact, "exact", "malformed_distances", MAX_VERTICES, integral=False)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.dtype != bool or selected.shape != estimate.shape or selected.shape != reference.shape:
            raise Refusal("malformed_distances", "mask must be a boolean array matching distances and exact")
        estimate, reference = estimate[selected], reference[selected]
    estimate = _distances(estimate, "distances")
    reference = _distances(reference, "exact", len(estimate))
    if (reference < 0).any():
        raise Refusal("negative_reference", "exact distances must be non-negative")
    error = np.abs(estimate - reference)
    positive = reference > 0
    relative = error[positive] / reference[positive]
    stats = lambda values: (float(values.max()), float(values.mean()), float(np.sqrt(np.mean(values ** 2))))
    result = dict(zip(("max_abs_error", "mean_abs_error", "rms_abs_error"), stats(error)))
    result.update(dict(zip(("max_rel_error", "mean_rel_error", "rms_rel_error"),
                           stats(relative) if relative.size else (None, None, None))))
    result.update(count=int(len(error)), relative_count=int(relative.size))
    return result


__all__ = [
    "SCHEMA", "CAPABILITY", "MAX_VERTICES", "MAX_DENSE_VERTICES", "BOUNDARY_CONDITIONS", "TriangleMesh",
    "SampledSurface", "plane_patch", "cylinder_patch", "icosphere", "torus_mesh", "edge_graph_distance",
    "cotangent_laplacian", "heat_method_distance", "compare_to_reference",
]
