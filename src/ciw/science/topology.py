"""Combinatorial topology and homotopy classes that govern geodesic decisions.

Capability ``winding_classes``. A shortest-path question is only well posed once
the topology is known: on a simply connected surface every pair of points has one
homotopy class of paths, on a cylinder the classes are indexed by an integer
winding, on a torus by a pair of integers, and on anything else a winding number
does not describe them at all. This module computes that information and refuses
the questions it cannot answer.

Scope
-----
* Triangle-mesh topology from face lists alone: Euler characteristic, boundary
  loops, vertex-connected components, orientability (propagation of a consistent
  face orientation) and genus, plus ``classify_surface`` which names the homotopy
  model a geodesic solver must respect.
* Angle unwrapping with an explicit ambiguity margin, winding classes of sampled
  paths on the cylinder and the torus, path equivalence rel endpoints, and the
  closed-form geodesics of every homotopy class between two points on a cylinder.
* 0-dimensional sublevel-set persistence on a graph (union-find, elder rule) and
  the exact bottleneck distance between small diagrams.

Limits
------
* Topology is purely combinatorial: geometry (self-intersection, degenerate
  faces) is not examined here. Non-manifold edges, pinched vertices and
  disconnected input are refused where they would make an answer ambiguous.
* Winding classes are only as good as the sampling: a successive angular step
  within ``UNWRAP_MARGIN`` of a half-turn is refused, never guessed. A path that
  jumps a full turn between samples is indistinguishable from one that does not.
* Class labels are relative to the principal lift of the endpoint difference in
  ``[-pi, pi)``; for endpoints an exact half-turn apart the label depends on
  that convention (``paths_equivalent`` compares lifts directly and is immune).
* Bottleneck distances are exact but use binary search plus bipartite matching,
  so they are bounded to ``MAX_DIAGRAM_POINTS`` points per diagram.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ._common import Refusal, finite, integer, vector

CAPABILITY = "winding_classes"
MAX_FACES = 100_000
MAX_INDEX = 2**31 - 1
MAX_PATH_SAMPLES = 100_000
MAX_ABS_ANGLE = 1e6          # rad; beyond this float64 angles lose sub-nanoradian resolution
UNWRAP_MARGIN = math.radians(5.0)
MAX_CLASS_RANGE = 10_001
MAX_GRAPH_VERTICES = 100_000
MAX_GRAPH_EDGES = 500_000
MAX_DIAGRAM_POINTS = 200
SURFACE_KINDS = ("plane", "sphere", "cylinder", "torus")
TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------- validation
def _is_number(item: Any, integral: bool) -> bool:
    kinds = (int, np.integer) if integral else (int, float, np.integer, np.floating)
    return isinstance(item, kinds) and not isinstance(item, (bool, np.bool_))


def _admit(value: Any, name: str, code: str, limit: int, integral: bool) -> np.ndarray:
    """Numeric array from an ndarray or nested lists, refusing strings, bools, ragged input and
    more than ``limit`` rows before any conversion. Integers are kept exact (object dtype)."""
    kinds = "iu" if integral else "iuf"
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in kinds:
            raise Refusal(code, f"{name} must hold {'integers' if integral else 'numbers'}")
        if value.ndim and len(value) > limit:
            raise Refusal("oversized_input", f"{name} exceeds {limit} entries", entries=len(value))
        return value if integral else value.astype(float)
    if not isinstance(value, (list, tuple)):
        raise Refusal(code, f"{name} must be an array")
    if len(value) > limit:
        raise Refusal("oversized_input", f"{name} exceeds {limit} entries", entries=len(value))
    try:
        raw = np.array(value, dtype=object)
        if not all(_is_number(item, integral) for item in raw.flat):
            raise TypeError(name)
        return raw if integral else raw.astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise Refusal(code, f"{name} must hold only {'integers' if integral else 'numbers'}") from exc


def _index_pairs(value: Any, name: str, code: str, limit: int, width: int, bound: int) -> np.ndarray:
    array = _admit(value, name, code, limit, integral=True)
    if array.ndim != 2 or array.shape[1] != width:
        raise Refusal(code, f"{name} must have shape (k, {width})", shape=list(array.shape))
    if array.size and (array.min() < 0 or array.max() >= bound):   # exact ints: no overflow before this check
        bad = next(i for i, row in enumerate(array.tolist()) if min(row) < 0 or max(row) >= bound)
        raise Refusal("face_index_out_of_range" if width == 3 else "edge_index_out_of_range",
                      f"{name}[{bad}] names a vertex outside [0, {bound})", row=bad, bound=bound)
    return np.asarray(array, dtype=np.int64).reshape(-1, width)


def face_array(faces: Any, vertex_count: int | None = None, *, limit: int = MAX_FACES) -> np.ndarray:
    """Admit a nonempty (m, 3) array of integer vertex indices; refuse anything else.

    Indices must be non-negative, below ``vertex_count`` when given (else below
    ``MAX_INDEX``), and distinct within each face. Nothing is cast from floats.
    """
    bound = MAX_INDEX if vertex_count is None else integer(vertex_count, "vertex_count", minimum=3, maximum=MAX_INDEX)
    result = _index_pairs(faces, "faces", "malformed_mesh", limit, 3, bound)
    if len(result) == 0:
        raise Refusal("malformed_mesh", "faces must not be empty")
    repeated = (result[:, 0] == result[:, 1]) | (result[:, 1] == result[:, 2]) | (result[:, 0] == result[:, 2])
    if repeated.any():
        bad = int(np.argmax(repeated))
        raise Refusal("repeated_face_vertex", f"face {bad} repeats a vertex", face=bad, vertices=result[bad])
    return result


def edge_table(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges ``(E, 2)`` (sorted pairs), the edge index of each face slot
    ``(m, 3)`` for the directed sides (f0,f1), (f1,f2), (f2,f0), and each edge's face count."""
    base = int(faces.max()) + 1
    heads = np.roll(faces, -1, axis=1)
    keys = np.minimum(faces, heads) * base + np.maximum(faces, heads)
    unique, inverse, counts = np.unique(keys.ravel(), return_inverse=True, return_counts=True)
    return np.stack([unique // base, unique % base], axis=1), inverse.reshape(faces.shape), counts


def _manifold_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges, inverse, counts = edge_table(faces)
    if (counts > 2).any():
        bad = int(np.argmax(counts > 2))
        raise Refusal("non_manifold_edge", f"edge {edges[bad].tolist()} is shared by {int(counts[bad])} faces",
                      edge=edges[bad], faces=int(counts[bad]))
    return edges, inverse, counts


class _Sets:
    """Union-find with path halving; ``union`` keeps the smaller root."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _interior_pairs(faces: np.ndarray, inverse: np.ndarray, counts: np.ndarray):
    """For each edge used by two faces: the two face slots (face, side) holding it."""
    slots = np.flatnonzero(counts[inverse.ravel()] == 2)
    order = slots[np.argsort(inverse.ravel()[slots], kind="stable")]
    first, second = order[0::2], order[1::2]
    return first // 3, first % 3, second // 3, second % 3


# ----------------------------------------------------------------------------- mesh topology
def euler_characteristic(faces: Any, vertex_count: int | None = None) -> int:
    """chi = V - E + F where V counts only vertices referenced by a face.

    ``vertex_count`` (optional) bounds the admissible indices; unreferenced
    vertices below it are isolated points and are deliberately not counted, so
    chi describes the surface the faces span.
    """
    array = face_array(faces, vertex_count)
    return int(np.unique(array).size - len(edge_table(array)[0]) + len(array))


def connected_components(faces: Any) -> list[list[int]]:
    """Vertex sets of the components (faces joined through shared vertices), each sorted,
    ordered by their smallest vertex."""
    array = face_array(faces)
    labels, compact = np.unique(array, return_inverse=True)
    compact = compact.reshape(array.shape)
    sets = _Sets(len(labels))
    for a, b, c in compact.tolist():
        sets.union(a, b)
        sets.union(a, c)
    roots = np.array([sets.find(i) for i in range(len(labels))])
    return [labels[roots == root].tolist() for root in np.unique(roots)]


def boundary_loops(faces: Any) -> list[list[int]]:
    """Edges used by exactly one face, chained into closed vertex loops.

    Each loop starts at its smallest vertex and, when that vertex has a unique
    outgoing boundary half-edge, follows the face orientation (so for
    counter-clockwise faces the surface lies to the left). Loops are ordered by
    their first vertex and do not repeat it at the end. Refuses
    ``non_manifold_edge`` (an edge in more than two faces) and
    ``non_manifold_vertex`` (a vertex on more than two boundary edges, where the
    chaining would be ambiguous).
    """
    array = face_array(faces)
    _, inverse, counts = _manifold_edges(array)
    single = counts[inverse] == 1
    tails, heads = array[single].tolist(), np.roll(array, -1, axis=1)[single].tolist()
    neighbours: dict[int, list[int]] = {}
    for a, b in zip(tails, heads):
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    for vertex, adjacent in neighbours.items():
        if len(adjacent) != 2:
            raise Refusal("non_manifold_vertex", f"vertex {vertex} lies on {len(adjacent)} boundary edges",
                          vertex=vertex, boundary_edges=len(adjacent))
    outgoing: dict[int, list[int]] = {}
    for a, b in zip(tails, heads):
        outgoing.setdefault(a, []).append(b)
    loops, visited = [], set()
    for start in sorted(neighbours):
        if start in visited:
            continue
        forward = outgoing.get(start, [])
        previous, current = start, forward[0] if len(forward) == 1 else min(neighbours[start])
        loop = [start]
        visited.add(start)
        while current != start:
            loop.append(current)
            visited.add(current)
            first, second = neighbours[current]
            previous, current = current, (second if first == previous else first)
        loops.append(loop)
    return loops


def is_consistently_oriented(faces: Any) -> bool:
    """True when every interior edge is traversed in opposite directions by its two faces."""
    array = face_array(faces)
    _, inverse, counts = _manifold_edges(array)
    f, s, g, t = _interior_pairs(array, inverse, counts)
    return bool(np.all(array[f, s] != array[g, t]))


def is_orientable(faces: Any) -> bool:
    """Whether face orientations can be flipped so every interior edge is traversed oppositely.

    Propagates orientation across interior edges (a 2-colouring search of the face
    graph, one flip parity per edge); a contradiction proves non-orientability.
    Refuses non-manifold edges.
    """
    array = face_array(faces)
    _, inverse, counts = _manifold_edges(array)
    f, s, g, t = _interior_pairs(array, inverse, counts)
    flip = (array[f, s] == array[g, t]).astype(int)     # same direction -> exactly one must flip
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(array))]
    for a, b, parity in zip(f.tolist(), g.tolist(), flip.tolist()):
        adjacency[a].append((b, parity))
        adjacency[b].append((a, parity))
    colour = [-1] * len(array)
    for seed in range(len(array)):
        if colour[seed] != -1:
            continue
        colour[seed], queue = 0, [seed]
        while queue:
            face = queue.pop()
            for other, parity in adjacency[face]:
                wanted = colour[face] ^ parity
                if colour[other] == -1:
                    colour[other] = wanted
                    queue.append(other)
                elif colour[other] != wanted:
                    return False
    return True


def _fans(faces: np.ndarray, inverse: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per referenced vertex: number of face fans around it (1 for a manifold vertex)."""
    corners = _Sets(3 * len(faces))
    f, s, g, t = _interior_pairs(faces, inverse, counts)
    same = faces[f, s] == faces[g, t]
    for a, i, b, j, parallel in zip(f.tolist(), s.tolist(), g.tolist(), t.tolist(), same.tolist()):
        i1, j1 = (i + 1) % 3, (j + 1) % 3
        if parallel:
            corners.union(3 * a + i, 3 * b + j)
            corners.union(3 * a + i1, 3 * b + j1)
        else:
            corners.union(3 * a + i, 3 * b + j1)
            corners.union(3 * a + i1, 3 * b + j)
    roots = np.array([corners.find(c) for c in range(3 * len(faces))])
    pairs = np.unique(np.stack([faces.ravel(), roots], axis=1), axis=0)
    vertices, fan_counts = np.unique(pairs[:, 0], return_counts=True)
    return np.stack([vertices, fan_counts], axis=1)


def _require_surface(array: np.ndarray) -> None:
    _, inverse, counts = _manifold_edges(array)
    fans = _fans(array, inverse, counts)
    pinched = fans[fans[:, 1] > 1]
    if len(pinched):
        raise Refusal("non_manifold_vertex", f"vertex {int(pinched[0, 0])} joins {int(pinched[0, 1])} separate fans",
                      vertex=int(pinched[0, 0]), fans=int(pinched[0, 1]))


def genus(faces: Any) -> int:
    """Genus g = (2 - chi - b) / 2 of a connected, orientable, manifold triangle surface.

    Refuses ``disconnected_surface``, ``non_manifold_edge``, ``non_manifold_vertex``
    and ``non_orientable_surface`` (whose crosscap number ``classify_surface`` reports).
    """
    summary = classify_surface(faces)
    if not summary["orientable"]:
        raise Refusal("non_orientable_surface", "genus (2 - chi - b)/2 applies only to orientable surfaces",
                      crosscaps=summary["crosscaps"])
    return summary["genus"]


def classify_surface(faces: Any) -> dict:
    """Topological summary of a connected manifold triangle surface and the homotopy model
    a geodesic solver must respect.

    ``homotopy_model`` is ``"trivial"`` (one class per endpoint pair: plane patch, disk,
    sphere), ``"Z"`` (integer winding: annulus/tube, Moebius band), ``"Z^2"`` (torus),
    ``"Z/2"`` (projective plane) or ``"nonabelian"`` (winding numbers do not classify
    paths; this module's class tools do not apply). ``fundamental_group_rank`` is the
    free rank for surfaces with boundary and the generator count otherwise.
    """
    array = face_array(faces)
    if len(connected_components(array)) != 1:
        raise Refusal("disconnected_surface", "topology is summarized per connected surface; split the components first")
    _require_surface(array)
    chi, loops, orientable = euler_characteristic(array), len(boundary_loops(array)), is_orientable(array)
    handles = 2 - chi - loops                      # 2g if orientable, crosscap number k otherwise
    if handles < 0 or (orientable and handles % 2):
        raise Refusal("inconsistent_topology", "2 - chi - b must be non-negative (and even if orientable)")
    rank = (handles + loops - 1) if loops else handles
    if loops:
        model = {0: "trivial", 1: "Z"}.get(rank, "nonabelian")
    elif orientable:
        model = {0: "trivial", 2: "Z^2"}.get(rank, "nonabelian")
    else:
        model = "Z/2" if handles == 1 else "nonabelian"
    return {
        "vertices": int(np.unique(array).size), "edges": int(len(edge_table(array)[0])), "faces": int(len(array)),
        "euler_characteristic": chi, "boundary_loops": loops, "orientable": orientable,
        "genus": handles // 2 if orientable else None, "crosscaps": None if orientable else handles,
        "fundamental_group_rank": rank, "simply_connected": model == "trivial", "homotopy_model": model,
    }


# ----------------------------------------------------------------------------- winding classes
def _wrap(angle: Any) -> Any:
    """Principal representative in [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + math.pi) % TWO_PI - math.pi


def _samples(value: Any, name: str, columns: int | None, minimum: int = 1, code: str = "malformed_path",
             limit: int = MAX_PATH_SAMPLES) -> np.ndarray:
    array = _admit(value, name, code, limit, integral=False)
    shape_ok = array.ndim == 1 if columns is None else (array.ndim == 2 and array.shape[1] == columns)
    if not shape_ok or len(array) < minimum:
        raise Refusal(code, f"{name} must have shape ({'k' if columns is None else f'k, {columns}'}) with k >= {minimum}",
                      shape=list(array.shape))
    if not np.isfinite(array).all():
        raise Refusal(code, f"{name} must be finite")
    return array


def unwrap_angles(angles: Any, max_step: float = math.pi, *, margin: float = UNWRAP_MARGIN) -> np.ndarray:
    """Continuous lift of sampled angles (rad), starting at ``angles[0]``.

    Each successive difference is reduced to [-pi, pi) and accumulated. The lift
    is only trustworthy if the true change between samples is smaller than a
    half-turn; a reduced step with ``|d| >= pi - margin`` (default margin 5 deg,
    the angular uncertainty the lift must survive) or ``|d| > max_step`` (a declared
    physical bound on the per-sample change) is refused as ``undersampled_path``
    rather than resolved by guessing a direction. Angles must satisfy
    ``|angle| <= MAX_ABS_ANGLE``.
    """
    values = _samples(angles, "angles", None)
    max_step = finite(max_step, "max_step", minimum=0.0, exclusive_minimum=True, maximum=math.pi)
    margin = finite(margin, "margin", minimum=0.0, maximum=math.pi / 2)
    if np.abs(values).max() > MAX_ABS_ANGLE:
        raise Refusal("angle_out_of_range", f"angles must satisfy |angle| <= {MAX_ABS_ANGLE} rad")
    steps = _wrap(np.diff(values))
    bad = (np.abs(steps) > max_step) | (np.abs(steps) >= math.pi - margin)
    if bad.any():
        index = int(np.argmax(bad))
        raise Refusal("undersampled_path", f"step {index} -> {index + 1} ({float(steps[index]):.6g} rad) cannot be "
                      "unwrapped unambiguously", step_index=index, step_rad=float(steps[index]),
                      max_step=max_step, margin=margin)
    return values[0] + np.concatenate([[0.0], np.cumsum(steps)])


def _lifted(angles: np.ndarray, margin: float) -> float:
    lift = unwrap_angles(angles, margin=margin)
    return float(lift[-1] - lift[0])


def _net_turns(angles: np.ndarray, margin: float) -> int:
    """Class = (lifted displacement - principal difference) / 2pi, principal in [-pi, pi)."""
    return int(round((_lifted(angles, margin) - float(_wrap(angles[-1] - angles[0]))) / TWO_PI))


def cylinder_homotopy_class(path_phi_z: Any, *, margin: float = UNWRAP_MARGIN) -> int:
    """Winding class of a sampled path ``[(phi, z), ...]`` on a cylinder.

    With the lifted angular displacement ``D`` (via ``unwrap_angles``) and the
    principal endpoint difference ``d = wrap(phi_end - phi_start)`` in [-pi, pi),
    the class is the integer ``k = (D - d) / 2pi``: the number of extra
    counter-clockwise full turns relative to the shortest-way lift. A closed loop
    has ``d = 0`` and ``k`` equals its turn count. Two paths with the same
    endpoints are homotopic (rel endpoints) exactly when their classes agree, and
    the geodesic of class ``k`` is the one ``cylinder_geodesic_classes`` lists
    under ``k``. ``z`` does not affect the class but must be finite.
    """
    path = _samples(path_phi_z, "path_phi_z", 2, minimum=2)
    return _net_turns(path[:, 0], margin)


def torus_homotopy_class(path_uv: Any, *, margin: float = UNWRAP_MARGIN) -> tuple[int, int]:
    """Class ``(m, n)`` of a sampled path ``[(u, v), ...]`` on a torus, each angle treated as in
    ``cylinder_homotopy_class``. The torus fundamental group is abelian (Z^2), so this pair
    classifies paths with fixed endpoints completely."""
    path = _samples(path_uv, "path_uv", 2, minimum=2)
    return _net_turns(path[:, 0], margin), _net_turns(path[:, 1], margin)


def paths_equivalent(path_a: Any, path_b: Any, surface_kind: str, *, tolerance: float = 1e-9,
                     margin: float = UNWRAP_MARGIN) -> bool:
    """Whether two sampled paths are homotopic with fixed endpoints.

    ``plane`` paths are ``(k, 2)`` points and ``sphere`` paths ``(k, 3)`` points:
    both surfaces are simply connected, so paths are equivalent iff their
    endpoints agree. ``cylinder`` paths are ``(phi, z)`` and ``torus`` paths
    ``(u, v)``: endpoints must agree (angles modulo 2pi) and the lifted angular
    displacements must agree, which is compared directly rather than through class
    labels. Endpoints agree when every coordinate differs by at most ``tolerance``.
    Paths with different endpoints are never equivalent (False, not a refusal).
    Sphere samples are not checked to lie on a common sphere.
    """
    if surface_kind not in SURFACE_KINDS:
        raise Refusal("unknown_surface_kind", f"surface_kind must be one of {list(SURFACE_KINDS)}",
                      surface_kind=str(surface_kind))
    tolerance = finite(tolerance, "tolerance", minimum=0.0, maximum=0.1)
    columns = 3 if surface_kind == "sphere" else 2
    a = _samples(path_a, "path_a", columns, minimum=2)
    b = _samples(path_b, "path_b", columns, minimum=2)
    angular = {"plane": (), "sphere": (), "cylinder": (0,), "torus": (0, 1)}[surface_kind]
    for end in (0, -1):
        gap = np.abs(a[end] - b[end])
        for axis in angular:
            gap[axis] = abs(float(_wrap(a[end, axis] - b[end, axis])))
        if gap.max() > tolerance:
            return False
    # With endpoints equal to within tolerance (<= 0.1 rad), lifted displacements differ by
    # 2 pi j plus at most 2 * tolerance, so |difference| < pi decides j = 0.
    return all(abs(_lifted(a[:, axis], margin) - _lifted(b[:, axis], margin)) < math.pi for axis in angular)


def _class_range(k_range: Any) -> list[int]:
    if isinstance(k_range, range):
        values = k_range
    elif isinstance(k_range, (list, tuple, np.ndarray)):
        values = list(k_range)
    else:
        raise Refusal("malformed_record", "k_range must be a range or a sequence of integers")
    try:
        size = len(values)
    except OverflowError:
        size = MAX_CLASS_RANGE + 1
    if size == 0 or size > MAX_CLASS_RANGE:
        raise Refusal("oversized_input" if size else "malformed_record",
                      f"k_range must hold between 1 and {MAX_CLASS_RANGE} classes")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise Refusal("malformed_record", "k_range must hold integers")
        if abs(int(value)) > MAX_CLASS_RANGE:
            raise Refusal("out_of_domain", f"winding classes must satisfy |k| <= {MAX_CLASS_RANGE}")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise Refusal("malformed_record", "k_range must not repeat a class")
    return result


def _cylinder_class(dphi: float, dz: float, radius: float, k: int) -> dict:
    angle = dphi + TWO_PI * k
    along = radius * angle
    length = math.hypot(along, dz)
    return {"k": k, "length": length, "angular_displacement_rad": angle, "axial_displacement": dz,
            "helix_angle_rad": math.atan2(abs(along), abs(dz)) if length > 0 else None, "turns": angle / TWO_PI}


def cylinder_geodesic_classes(p: Any, q: Any, radius: float, k_range: Any) -> list[dict]:
    """The geodesic of each winding class between ``p = (phi, z)`` and ``q = (phi, z)``.

    Unrolling the cylinder of radius ``R`` onto its universal cover (the plane with
    coordinates ``(R phi, z)``), the lifts of ``q`` are ``q + (2 pi R k, 0)`` and
    each class ``k`` contains exactly one geodesic: the straight segment to that
    lift, a helix on the cylinder. With ``dphi = wrap(phi_q - phi_p)`` in [-pi, pi)
    it has ``length = sqrt((R (dphi + 2 pi k))^2 + dz^2)``, ``turns = (dphi + 2 pi k)
    / 2pi`` (signed, fractional) and ``helix_angle_rad`` = angle between the helix
    and the cylinder's generators (0 = axial line, pi/2 = circle; None for the
    zero-length path). Class labels match ``cylinder_homotopy_class``.
    """
    p, q = vector(p, "p", 2), vector(q, "q", 2)
    radius = finite(radius, "radius", minimum=0.0, exclusive_minimum=True)
    dphi, dz = float(_wrap(q[0] - p[0])), float(q[1] - p[1])
    return [_cylinder_class(dphi, dz, radius, k) for k in _class_range(k_range)]


def shortest_class(p: Any, q: Any, radius: float) -> dict:
    """The shortest geodesic class between ``p`` and ``q`` on a cylinder (always k = 0 under the
    principal-lift convention), with ``tied_classes`` listing every class of equal length
    (two classes tie exactly when the points are a half-turn apart)."""
    classes = cylinder_geodesic_classes(p, q, radius, range(-1, 2))
    best = min(item["length"] for item in classes)
    tied = [item["k"] for item in classes if item["length"] <= best * (1 + 1e-12)]
    result = dict(next(item for item in classes if item["length"] == best))
    result["tied_classes"] = tied
    return result


# ----------------------------------------------------------------------------- persistence
def _graph_edges(edges: Any, vertex_count: int) -> np.ndarray:
    if isinstance(edges, (list, tuple)) and len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    result = _index_pairs(edges, "edges", "malformed_graph", MAX_GRAPH_EDGES, 2, vertex_count)
    loops = result[:, 0] == result[:, 1]
    if loops.any():
        raise Refusal("self_loop", "edges must join two distinct vertices", edge=int(np.argmax(loops)))
    return result


def persistence_0d(vertex_values: Any, edges: Any, *, keep_zero: bool = False) -> list[tuple[float, float]]:
    """0-dimensional persistence diagram of the sublevel-set filtration of a graph.

    Vertex ``v`` enters at ``f(v)`` and edge ``(u, v)`` at ``max(f(u), f(v))``.
    Edges are processed in increasing order (ties by position); when an edge joins
    two components the younger one (larger birth value; equal births: larger
    vertex index) dies there (elder rule), giving ``(birth, death)``. Each
    component that never merges gives ``(birth, inf)``. Zero-persistence pairs are
    dropped unless ``keep_zero``. Points are returned sorted.
    """
    values = _samples(vertex_values, "vertex_values", None, code="malformed_graph", limit=MAX_GRAPH_VERTICES)
    graph = _graph_edges(edges, len(values))
    weights = np.maximum(values[graph[:, 0]], values[graph[:, 1]])
    order = np.argsort(weights, kind="stable")
    parent = list(range(len(values)))
    key = [(float(v), i) for i, v in enumerate(values)]      # elder = smaller key; roots are always elders

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    diagram: list[tuple[float, float]] = []
    for index in order.tolist():
        a, b = find(int(graph[index, 0])), find(int(graph[index, 1]))
        if a == b:
            continue
        elder, younger = (a, b) if key[a] < key[b] else (b, a)
        parent[younger] = elder
        death = float(weights[index])
        if keep_zero or death > key[younger][0]:
            diagram.append((key[younger][0], death))
    diagram.extend((key[root][0], math.inf) for root in range(len(values)) if parent[root] == root)
    return sorted(diagram)


def _diagram(value: Any, name: str) -> np.ndarray:
    array = _admit(value, name, "malformed_diagram", MAX_DIAGRAM_POINTS, integral=False)
    if array.size == 0:
        return np.zeros((0, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        raise Refusal("malformed_diagram", f"{name} must be a list of (birth, death) pairs")
    births, deaths = array[:, 0], array[:, 1]
    if not np.isfinite(births).all() or np.isnan(deaths).any() or (deaths == -np.inf).any():
        raise Refusal("malformed_diagram", f"{name} births must be finite and deaths finite or +inf")
    if (deaths < births).any():
        raise Refusal("malformed_diagram", f"{name} has a point with death < birth")
    return array


def _max_matching(adjacency: list[list[int]], right_size: int) -> int:
    """Hopcroft-Karp maximum matching size for a bipartite graph given as left adjacency lists."""
    match_left, match_right = [-1] * len(adjacency), [-1] * right_size
    size = 0
    while True:
        layer = [-1] * len(adjacency)
        queue = [u for u in range(len(adjacency)) if match_left[u] == -1]
        for u in queue:
            layer[u] = 0
        reachable, head = False, 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adjacency[u]:
                w = match_right[v]
                if w == -1:
                    reachable = True
                elif layer[w] == -1:
                    layer[w] = layer[u] + 1
                    queue.append(w)
        if not reachable:
            return size

        def augment(u: int) -> bool:
            for v in adjacency[u]:
                w = match_right[v]
                if w == -1 or (layer[w] == layer[u] + 1 and augment(w)):
                    match_left[u], match_right[v] = v, u
                    return True
            layer[u] = -2
            return False

        for u in range(len(adjacency)):
            if match_left[u] == -1 and augment(u):
                size += 1


def _finite_bottleneck(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) + len(b) == 0:
        return 0.0
    half_a, half_b = (a[:, 1] - a[:, 0]) / 2, (b[:, 1] - b[:, 0]) / 2
    cost = np.maximum(np.abs(a[:, None, 0] - b[None, :, 0]), np.abs(a[:, None, 1] - b[None, :, 1]))

    def feasible(eps: float) -> bool:
        # A perfect matching of the diagonal-augmented graph exists iff some matching of
        # A-B pairs within eps leaves only points within eps of the diagonal unmatched. By the
        # Mendelsohn-Dulmage theorem that holds iff one matching covers every far A point and
        # another covers every far B point, so two ordinary maximum matchings decide it.
        far_a, far_b = np.flatnonzero(half_a > eps), np.flatnonzero(half_b > eps)
        return (_max_matching([np.flatnonzero(cost[i] <= eps).tolist() for i in far_a], len(b)) == len(far_a)
                and _max_matching([np.flatnonzero(cost[:, j] <= eps).tolist() for j in far_b], len(a)) == len(far_b))

    candidates = np.unique(np.concatenate([[0.0], cost.ravel(), half_a, half_b]))
    low, high = 0, len(candidates) - 1          # matching everything to the diagonal is feasible at the maximum
    while low < high:
        middle = (low + high) // 2
        if feasible(float(candidates[middle])):
            high = middle
        else:
            low = middle + 1
    return float(candidates[low])


def bottleneck_distance(diagram_a: Any, diagram_b: Any) -> float:
    """Exact bottleneck distance between persistence diagrams (L-infinity ground metric).

    Points may be matched to each other or to their diagonal projection (cost
    ``(death - birth) / 2``). Points at infinity can only be matched to points at
    infinity, at cost ``|birth_a - birth_b|`` (sorted matching, optimal in 1-D); if
    their counts differ the distance is ``inf``. The finite part is the smallest
    candidate value (pairwise costs and half-persistences) whose matching problem is
    feasible, found by binary search with Hopcroft-Karp matchings.
    """
    a, b = _diagram(diagram_a, "diagram_a"), _diagram(diagram_b, "diagram_b")
    a_inf, b_inf = np.isinf(a[:, 1]), np.isinf(b[:, 1])
    if a_inf.sum() != b_inf.sum():
        return math.inf
    essential = float(np.max(np.abs(np.sort(a[a_inf, 0]) - np.sort(b[b_inf, 0])), initial=0.0))
    return max(essential, _finite_bottleneck(a[~a_inf], b[~b_inf]))


__all__ = [
    "CAPABILITY", "MAX_FACES", "UNWRAP_MARGIN", "MAX_DIAGRAM_POINTS", "SURFACE_KINDS", "face_array", "edge_table",
    "euler_characteristic", "connected_components", "boundary_loops", "is_consistently_oriented", "is_orientable",
    "genus", "classify_surface", "unwrap_angles", "cylinder_homotopy_class", "torus_homotopy_class",
    "paths_equivalent", "cylinder_geodesic_classes", "shortest_class", "persistence_0d", "bottleneck_distance",
]
