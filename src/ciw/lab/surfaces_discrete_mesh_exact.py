"""Exact polyhedral geodesic distances on triangle meshes by window propagation (NumPy setup, Python loop).

Scope: single-source distances along the surface of a triangle mesh that are
exact up to floating-point rounding: the length of the globally shortest path
on the polyhedron, not of a locally straight one. Geodesic rays from the
source are propagated across edges as windows (an interval of an edge
together with the unfolded position of the source that reaches it), in the
manner of Chen and Han (1990) as improved by Xin and Wang (2009, "ICH"):

- a priority queue ordered by the smallest distance a window carries, so
  vertex distances settle in increasing order as in Dijkstra's algorithm;
- pseudo-sources at saddle vertices (angle sum above 2 pi) and at reflex
  boundary vertices (angle sum above pi), the only places where a shortest
  path may bend;
- pruning against the best distance known at vertices: the part of a window
  on an edge whose path is longer than the path through an endpoint of that
  edge is cut off, and a window is dropped when a vertex of one of its two
  faces beats every point of it. A subpath of a shortest path is shortest,
  so a point that a vertex path beats strictly carries no shortest path, and
  pruning only by a strict margin never discards one.

Sources and targets at arbitrary surface points are inserted as vertices by a
planar 1-to-3 face split, or a 1-to-2 split of the two faces at a point on an
edge (``insert_points``). The new faces lie in the old face planes, so the
polyhedral metric, and every distance, is unchanged.

Non-claims: exactness holds in exact arithmetic; computed distances carry the
rounding of the unfoldings. The loop is plain Python, for meshes of a few
thousand vertices. It returns distances, not the shortest path polyline.
Every mesh here is generated from a declared formula in normalized units, and
nothing measures a physical surface.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from . import surfaces_discrete_mesh_geometry as G

# Angle sums within this of 2 pi (pi at a boundary vertex) are flat: rays pass such a vertex on both sides.
ANGLE_TOLERANCE = 1e-9
# Relative edge-parameter slack: a ray this close to an apex passes through it.
PARAMETER_TOLERANCE = 1e-12
# Relative length slack (times the mean edge): pruning cuts only what a vertex path beats by more than this.
PRUNE_TOLERANCE = 1e-10
# Barycentric coordinate below which an inserted point lies on an edge (or, twice, at a vertex).
ON_EDGE = 1e-9


class ExactGeodesic:
    """Half-edge layout of a triangle mesh and exact single-source distances on it.

    Half-edge ``h = 3 f + k`` runs from ``faces[f, k]`` to ``faces[f, k + 1]``;
    its frame puts the tail at the origin, the head at ``(L, 0)`` and the
    apex of face ``f`` at ``(cx, cy)`` with ``cy > 0``. A window on ``h``
    propagates into face ``f`` and holds ``(b0, b1, sx, sy, sigma)``: the edge
    interval, its source unfolded to ``sy < 0`` and the source's own distance
    (nonzero for a pseudo-source).
    """

    def __init__(self, mesh: G.TriMesh):
        vertices, faces = mesh.vertices, mesh.faces
        m = len(faces)
        table = G._edge_table(faces, len(vertices))
        twin = np.full(3 * m, -1, dtype=np.int64)
        twin[table["pair_first"]] = table["pair_second"]
        twin[table["pair_second"]] = table["pair_first"]
        tail = faces.reshape(-1)
        head = np.roll(faces, -1, axis=1).reshape(-1)
        apex = np.roll(faces, -2, axis=1).reshape(-1)
        edge = vertices[head] - vertices[tail]
        side = vertices[apex] - vertices[tail]
        length = np.linalg.norm(edge, axis=1)
        cx = np.einsum("ij,ij->i", edge, side) / length
        cy = np.linalg.norm(np.cross(edge, side), axis=1) / length
        base = 3 * (np.arange(3 * m) // 3)
        local = np.arange(3 * m) % 3
        left = base + (local + 2) % 3    # apex -> tail: rays passing the apex on the tail side leave through it
        right = base + (local + 1) % 3   # head -> apex: rays passing the apex on the head side
        self.n = len(vertices)
        self.scale = float(np.mean(length))
        angles = mesh.corner_angles()
        total = np.bincount(faces.ravel(), weights=angles.ravel(), minlength=self.n)
        boundary = mesh.boundary_vertices
        self.pseudo = ((~boundary & (total > 2 * math.pi + ANGLE_TOLERANCE))
                       | (boundary & (total > math.pi + ANGLE_TOLERANCE))).tolist()
        # Opposite half-edges of every corner: the edges a pseudo-source at that vertex sees first.
        corners = np.argsort(faces.ravel(), kind="stable")
        counts = np.bincount(faces.ravel(), minlength=self.n)
        starts = np.r_[0, np.cumsum(counts)]
        opposite = (base[corners] + (local[corners] + 1) % 3).tolist()
        self.ring = [opposite[starts[v]:starts[v + 1]] for v in range(self.n)]
        back = np.where(twin >= 0, twin, 0)
        # Per half-edge: length, apex (cx, cy), tail, head, apex vertex, the twins and lengths of the edges
        # apex -> tail and head -> apex, the apex of the face across (vertex, x, y in this frame), own twin.
        self.half = list(zip(
            length.tolist(), cx.tolist(), cy.tolist(), tail.tolist(), head.tolist(), apex.tolist(),
            twin[left].tolist(), length[left].tolist(), twin[right].tolist(), length[right].tolist(),
            np.where(twin >= 0, apex[back], -1).tolist(), (length - cx[back]).tolist(), (-cy[back]).tolist(),
            twin.tolist()))

    def distances(self, source: int, targets=None) -> np.ndarray:
        """Exact distances from vertex ``source`` to every vertex (inf when unreachable).

        With ``targets``, propagation stops once every target distance is
        final (no queued window can carry less); other entries are then upper
        bounds only.
        """
        return self.solve(source, targets)[0]

    def solve(self, source: int, targets=None):
        """(distances, counters) with counters of windows created, propagated and pseudo-sources expanded."""
        half, ring, pseudo = self.half, self.ring, self.pseudo
        slack = PRUNE_TOLERANCE * self.scale
        hypot, inf = math.hypot, math.inf
        dist = [inf] * self.n
        dist[source] = 0.0
        heap = [(0.0, 0, -1 - source, 0.0, 0.0, 0.0, 0.0, 0.0)]
        counters = {"windows": 0, "propagated": 0, "pseudo_sources": 0}
        sequence = 1
        wanted = None if targets is None else sorted({int(t) for t in targets})

        def relax(vertex, value):
            nonlocal sequence
            if value < dist[vertex]:
                dist[vertex] = value
                if pseudo[vertex]:
                    heapq.heappush(heap, (value, sequence, -1 - vertex, 0.0, 0.0, 0.0, 0.0, 0.0))
                    sequence += 1

        def push(h, b0, b1, sx, sy, sigma):
            nonlocal sequence
            window = _prune(half[h], b0, b1, sx, sy, sigma, dist, slack)
            if window is None:
                return
            b0, b1 = window
            if sx < b0:
                key = sigma + hypot(b0 - sx, sy)
            elif sx > b1:
                key = sigma + hypot(b1 - sx, sy)
            else:
                key = sigma - sy
            heapq.heappush(heap, (key, sequence, h, b0, b1, sx, sy, sigma))
            sequence += 1
            counters["windows"] += 1

        while heap:
            key, _, h, b0, b1, sx, sy, sigma = heapq.heappop(heap)
            if wanted is not None and key > max(dist[t] for t in wanted) + slack:
                break
            if h < 0:
                vertex = -1 - h
                if key > dist[vertex]:
                    continue  # superseded by a shorter path to the same pseudo-source
                counters["pseudo_sources"] += 1
                for o in ring[vertex]:
                    # The vertex is the apex of o; it sees o's endpoints directly and the face across o
                    # through the whole edge, from its position unfolded below the twin's frame.
                    record = half[o]
                    length, cx, cy, tail, head = record[:5]
                    relax(tail, key + record[7])
                    relax(head, key + record[9])
                    if record[13] >= 0:
                        push(record[13], 0.0, length, length - cx, -cy, key)
                continue
            record = half[h]
            window = _prune(record, b0, b1, sx, sy, sigma, dist, slack)
            if window is None:
                continue
            b0, b1 = window
            counters["propagated"] += 1
            length, cx, cy, _, _, apex, twin_left, left_length, twin_right, right_length = record[:10]
            tolerance = PARAMETER_TOLERANCE * length
            # Where the ray from the source through the apex crosses this edge.
            cross = sx - (cx - sx) * sy / (cy - sy)
            if b0 - tolerance <= cross <= b1 + tolerance:
                relax(apex, sigma + hypot(cx - sx, cy - sy))
            if cross > b0 + tolerance and twin_left >= 0:
                # Rays through [b0, min(b1, cross)] leave through the edge tail -> apex (child frame: tail, apex).
                ex, ey = cx / left_length, cy / left_length
                u0 = b0 * sy / ((b0 - sx) * ey + sy * ex) if b0 > 0.0 else 0.0
                u1 = left_length if cross <= b1 + tolerance else b1 * sy / ((b1 - sx) * ey + sy * ex)
                push(twin_left, max(u0, 0.0), min(u1, left_length), sx * ex + sy * ey, sy * ex - sx * ey, sigma)
            if cross < b1 - tolerance and twin_right >= 0:
                # Rays through [max(b0, cross), b1] leave through the edge apex -> head (child frame: apex, head).
                ex, ey = (length - cx) / right_length, -cy / right_length
                dx, dy = sx - cx, sy - cy
                v0 = 0.0 if cross >= b0 - tolerance else _right_parameter(b0, sx, sy, dx, dy, ex, ey)
                v1 = right_length if b1 >= length else _right_parameter(b1, sx, sy, dx, dy, ex, ey)
                push(twin_right, max(v0, 0.0), min(v1, right_length), dx * ex + dy * ey, dy * ex - dx * ey, sigma)
        return np.array(dist), counters


def insert_points(mesh: G.TriMesh, points) -> tuple:
    """(mesh, vertex ids) with every surface point of ``points`` = [(face, xyz)] made a vertex.

    A point inside a face splits it 1-to-3 about the point; a point within
    ``ON_EDGE`` (barycentric) of an edge splits both faces at that edge
    1-to-2; a point at a vertex is that vertex. New faces lie in the plane of
    the face they replace and keep its orientation, so the polyhedral metric
    is unchanged. The face of a later point is looked up again in the refined
    mesh. A point on no face is refused with ``point_outside_face``.
    """
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    ids = []
    for face, point in points:
        current = G.TriMesh(vertices, faces, mesh.name, mesh.params)
        point = np.asarray(point, dtype=float)
        bary = current.barycentric(face, point)
        if np.min(bary) < -ON_EDGE or not _on_plane(current, face, point):
            face, bary = current.locate(point)  # an earlier insertion split the declared face
        small = np.flatnonzero(bary < ON_EDGE)
        if len(small) >= 2:
            ids.append(int(faces[face, 3 - small.sum()]))
            continue
        new = len(vertices)
        vertices = np.concatenate([vertices, point[None]])
        if len(small) == 0:
            a, b, c = faces[face]
            faces = np.concatenate([faces, [[b, c, new], [c, a, new]]])
            faces[face] = [a, b, new]
        else:
            # The point lies on the edge opposite corner k; split this face and the one across.
            k = int(small[0])
            a, b, c = faces[face, (k + 1) % 3], faces[face, (k + 2) % 3], faces[face, k]
            faces[face] = [a, new, c]
            faces = np.concatenate([faces, [[new, b, c]]])
            across = np.flatnonzero(np.any(faces == a, axis=1) & np.any(faces == b, axis=1))
            for g in across:
                if g == face or new in faces[g]:
                    continue
                j = int(np.flatnonzero(faces[g] == b)[0])  # the face across runs b -> a
                d = faces[g, (j + 2) % 3]
                faces[g] = [b, new, d]
                faces = np.concatenate([faces, [[new, a, d]]])
        ids.append(new)
    return G.TriMesh(vertices, faces, mesh.name, mesh.params), ids


def _on_plane(mesh, face, point) -> bool:
    corners = mesh.vertices[mesh.faces[face]]
    return abs(float((point - corners[0]) @ mesh.face_normals[face])) <= G.ON_FACE_TOLERANCE * max(
        1.0, float(np.max(np.linalg.norm(corners - corners[0], axis=1))))


def _right_parameter(x, sx, sy, dx, dy, ex, ey) -> float:
    """Distance from the apex along apex -> head where the ray from the source through (x, 0) crosses it."""
    return -((x - sx) * (-dy) + sy * (-dx)) / ((x - sx) * ey + sy * ex)


def _prune(record, b0, b1, sx, sy, sigma, dist, slack):
    """The part of [b0, b1] not beaten by more than ``slack`` through the edge's endpoints, or None.

    Along the edge, (sigma + |S X|) - |P X| never increases away from the tail
    P and (sigma + |S X|) - |Q X| never decreases towards the head Q, so each
    endpoint cuts one end of the interval at a single root. The apex of either
    face drops the window only when it beats both ends by a margin c >= 0,
    because the points it beats by more than c >= 0 form a convex set.
    """
    length, cx, cy, tail, head, apex = record[:6]
    hypot = math.hypot
    d = dist[tail]
    if d < math.inf:
        c = d + slack - sigma
        if hypot(b1 - sx, sy) - b1 > c:
            return None
        if hypot(b0 - sx, sy) - b0 > c:
            den = 2.0 * (sx + c)
            b0 = min(max((sx * sx + sy * sy - c * c) / den, b0), b1) if den > 0.0 else b0
    d = dist[head]
    if d < math.inf:
        c = d + slack - sigma + length
        if hypot(b0 - sx, sy) + b0 > c:
            return None
        if hypot(b1 - sx, sy) + b1 > c:
            den = 2.0 * (c - sx)
            b1 = max(min((c * c - sx * sx - sy * sy) / den, b1), b0) if den > 0.0 else b1
    if b1 - b0 < -PARAMETER_TOLERANCE * length:
        return None
    for vertex, wx, wy in ((apex, cx, cy), (record[10], record[11], record[12])):
        if vertex < 0 or dist[vertex] == math.inf:
            continue
        c = dist[vertex] + slack - sigma
        if c >= 0.0 and all(hypot(x - sx, sy) - hypot(x - wx, wy) > c for x in (b0, b1)):
            return None
    return b0, b1
