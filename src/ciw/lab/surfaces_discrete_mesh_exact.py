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
  path may bend, and at vertices within ``ANGLE_TOLERANCE`` of flat, whose
  rounded angle sums cannot tell a flat vertex from a slight saddle;
- pruning against the best distance known at vertices: the part of a window
  on an edge whose path is longer than the path through an endpoint of that
  edge is cut off, and a window is dropped when a vertex of one of its two
  faces beats every point of it. A subpath of a shortest path is shortest,
  so a point that a vertex path beats strictly carries no shortest path, and
  pruning only by a strict margin never discards one.

Sources and targets at arbitrary surface points are inserted as vertices by a
planar 1-to-3 face split, or a 1-to-2 split of the two faces at a point on an
edge (``insert_points``), after moving the point onto that face or edge. The
new faces lie in the old face planes, so the polyhedral metric, and every
distance, is unchanged.

Every window records the window it was propagated from (or the pseudo-source
that emitted it), and every vertex the window or pseudo-source that last
lowered its distance. ``Propagation.path`` back-traces the shortest path
polyline from a target through that chain: from the target along the ray to
the window's unfolded source, edge crossing by edge crossing, to the
pseudo-source, and on from there to the source. ``Propagation.distance_at``
evaluates the exact distance at any surface point from the recorded windows
of its face and the distances of its vertices.

Non-claims: exactness holds in exact arithmetic; computed distances and paths
carry the rounding of the unfoldings. Where several shortest paths tie, the
back-trace returns the one whose window reached the target first. The loop
is plain Python, for meshes of a few thousand vertices. Every mesh here is
generated from a declared formula in normalized units, and nothing measures
a physical surface.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from . import surfaces_discrete_mesh_geometry as G

# Vertices whose angle sum exceeds 2 pi (pi at a boundary vertex) by more than -ANGLE_TOLERANCE are pseudo-sources.
# The margin leans towards more of them: a slight saddle taken for flat would leave the wedge behind it, as wide
# as its excess, to no window, while a flat vertex taken for a pseudo-source only adds windows.
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
        self.mesh = mesh
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
        # Angle sum minus that of a flat vertex: positive at saddles and reflex boundary corners.
        self.excess = total - np.where(mesh.boundary_vertices, math.pi, 2 * math.pi)
        self.pseudo = (self.excess > -ANGLE_TOLERANCE).tolist()
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
        propagation = self.propagate(source, targets)
        return propagation.distances, propagation.counters

    def propagate(self, source: int, targets=None, limit=None) -> "Propagation":
        """Propagate windows from vertex ``source``, recording what shortest paths need to be back-traced.

        With ``targets`` propagation stops once every target distance is final;
        with ``limit`` once no queued window carries a distance up to ``limit``,
        so that every distance up to it is final. Other distances are then upper
        bounds only.
        """
        half, ring, pseudo = self.half, self.ring, self.pseudo
        slack = PRUNE_TOLERANCE * self.scale
        hypot, inf = math.hypot, math.inf
        dist = [inf] * self.n
        dist[source] = 0.0
        # How each vertex's distance was last lowered: a window id (>= 0) or -1 - pseudo-source vertex.
        via = [None] * self.n
        # Per heap sequence number: (half-edge, b0, b1, sx, sy, sigma, parent) for a window, None for a vertex entry.
        windows = [None]
        heap = [(0.0, 0, -1 - source, 0.0, 0.0, 0.0, 0.0, 0.0)]
        counters = {"windows": 0, "propagated": 0, "pseudo_sources": 0}
        wanted = None if targets is None else sorted({int(t) for t in targets})
        stop = inf if limit is None else float(limit) + slack

        def relax(vertex, value, parent):
            if value < dist[vertex]:
                dist[vertex] = value
                via[vertex] = parent
                if pseudo[vertex]:
                    heapq.heappush(heap, (value, len(windows), -1 - vertex, 0.0, 0.0, 0.0, 0.0, 0.0))
                    windows.append(None)

        def push(h, b0, b1, sx, sy, sigma, parent):
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
            heapq.heappush(heap, (key, len(windows), h, b0, b1, sx, sy, sigma))
            windows.append((h, b0, b1, sx, sy, sigma, parent))
            counters["windows"] += 1

        while heap:
            key, sequence, h, b0, b1, sx, sy, sigma = heapq.heappop(heap)
            if key > stop or (wanted is not None and key > max(dist[t] for t in wanted) + slack):
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
                    relax(tail, key + record[7], h)
                    relax(head, key + record[9], h)
                    if record[13] >= 0:
                        push(record[13], 0.0, length, length - cx, -cy, key, h)
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
                relax(apex, sigma + hypot(cx - sx, cy - sy), sequence)
            if cross > b0 + tolerance and twin_left >= 0:
                # Rays through [b0, min(b1, cross)] leave through the edge tail -> apex (child frame: tail, apex).
                ex, ey = cx / left_length, cy / left_length
                u0 = b0 * sy / ((b0 - sx) * ey + sy * ex) if b0 > 0.0 else 0.0
                u1 = left_length if cross <= b1 + tolerance else b1 * sy / ((b1 - sx) * ey + sy * ex)
                push(twin_left, max(u0, 0.0), min(u1, left_length), sx * ex + sy * ey, sy * ex - sx * ey, sigma,
                     sequence)
            if cross < b1 - tolerance and twin_right >= 0:
                # Rays through [max(b0, cross), b1] leave through the edge apex -> head (child frame: apex, head).
                ex, ey = (length - cx) / right_length, -cy / right_length
                dx, dy = sx - cx, sy - cy
                v0 = 0.0 if cross >= b0 - tolerance else _right_parameter(b0, sx, sy, dx, dy, ex, ey)
                v1 = right_length if b1 >= length else _right_parameter(b1, sx, sy, dx, dy, ex, ey)
                push(twin_right, max(v0, 0.0), min(v1, right_length), dx * ex + dy * ey, dy * ex - dx * ey, sigma,
                     sequence)
        return Propagation(self, source, np.array(dist), via, windows, counters)


class Propagation:
    """One source's recorded propagation: distances, windows and how every vertex was reached.

    ``via[v]`` is the window (id >= 0) or pseudo-source (-1 - vertex) that
    last lowered vertex v's distance, and each window records its parent the
    same way. Following that chain from a vertex back-traces a shortest path:
    each window's ray runs straight to its unfolded source through the
    windows it was propagated from, and ends at the pseudo-source that
    emitted the first of them.
    """

    def __init__(self, solver: ExactGeodesic, source: int, distances, via, windows, counters):
        self.solver, self.source, self.distances, self.via = solver, source, distances, via
        self.windows, self.counters = windows, counters
        self._frames = {}
        self._by_half = None

    def _frame(self, h):
        """Tail position and in-plane unit axes of half-edge h's frame (x along the edge, y towards the apex)."""
        if h not in self._frames:
            record = self.solver.half[h]
            vertices = self.solver.mesh.vertices
            tail = vertices[record[3]]
            ex = (vertices[record[4]] - tail) / record[0]
            ey = vertices[record[5]] - tail
            ey = ey - float(np.dot(ey, ex)) * ex
            self._frames[h] = (tail, ex, ey / np.linalg.norm(ey))
        return self._frames[h]

    def _ray_crossing(self, h, point, sx, sy):
        """Parameter along half-edge h where the ray from the unfolded source (sx, sy) to ``point`` crosses it."""
        tail, ex, ey = self._frame(h)
        px, py = float(np.dot(point - tail, ex)), float(np.dot(point - tail, ey))
        return sx + (px - sx) * (-sy) / (py - sy), px, py

    def path(self, target: int) -> dict:
        """The shortest path polyline from vertex ``target`` back to the source.

        Returns the points (target first, source last), the face holding each
        segment, the mesh vertex at each point (-1 elsewhere), the path length,
        the target's distance and the largest distance by which a crossing fell
        outside its window's edge interval (rounding only; the crossing is
        clamped to the edge).
        """
        if not math.isfinite(self.distances[target]):
            raise G.MeshRefusal("unreachable_target", f"Vertex {target} was not reached from the source")
        vertices = self.solver.mesh.vertices
        return self._back_trace([vertices[target].copy()], [], [int(target)], int(target), None,
                                float(self.distances[target]))

    def path_to_point(self, face: int, point) -> dict:
        """The shortest path polyline from a surface point of ``face`` back to the source (as ``path``).

        The path's last segment is the one ``distance_at`` found shortest: from
        a vertex of the point's face, or along a window's ray into it.
        """
        point = np.asarray(point, dtype=float)
        distance, face, code = self._best_at(face, point)
        if code < 0:
            vertex = -1 - code
            return self._back_trace([point.copy(), self.solver.mesh.vertices[vertex].copy()], [face], [-1, vertex],
                                    vertex, None, distance)
        return self._back_trace([point.copy()], [], [-1], None, (code, point), distance)

    def _back_trace(self, points, faces, at_vertex, vertex, ray, distance) -> dict:
        """Follow the recorded chain from ``vertex`` (or from a point on a window's ray) to the source."""
        solver, vertices = self.solver, self.solver.mesh.vertices
        faces_of = solver.mesh.faces
        outside = 0.0
        while ray is not None or self.via[vertex] is not None:
            if ray is not None:
                code, point = ray
                ray = None
            else:
                code = self.via[vertex]
                point = vertices[vertex]
            if code < 0:  # straight along an edge from the pseudo-source that relaxed this vertex
                q = -1 - code
                faces.append(next(o // 3 for o in solver.ring[q] if vertex in faces_of[o // 3]))
                points.append(vertices[q].copy())
                at_vertex.append(q)
                vertex = q
                continue
            while True:
                h, b0, b1, sx, sy, _, parent = self.windows[code]
                x, _, _ = self._ray_crossing(h, point, sx, sy)
                outside = max(outside, b0 - x, x - b1)
                length = solver.half[h][0]
                x = min(max(x, 0.0), length)
                tail, ex, _ = self._frame(h)
                point = tail + x * ex
                faces.append(h // 3)
                points.append(point.copy())
                at_vertex.append(-1)
                if parent >= 0:
                    code = parent
                    continue
                q = -1 - parent  # the emitting pseudo-source lies in the face across this edge
                faces.append(solver.half[h][13] // 3)
                points.append(vertices[q].copy())
                at_vertex.append(q)
                vertex = q
                break
        return {"points": np.array(points), "faces": faces, "vertices": at_vertex, "outside_interval": outside,
                "length": float(np.sum(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1))),
                "distance": distance}

    def distance_at(self, face: int, point) -> float:
        """Exact distance from the source to ``point`` on ``face`` (inside it or on its boundary).

        A shortest path's last segment lies in the point's face, or for a point
        on an edge possibly in the face across it: it starts at a vertex of that
        face or enters it through a window on one of its edges. The windows
        pushed are all kept, so this needs a propagation without a target stop
        and with a limit, if any, at least the point's distance.
        """
        return self._best_at(face, np.asarray(point, dtype=float))[0]

    def _best_at(self, face, point):
        """(distance, face holding the last segment, window id or -1 - vertex it starts from)."""
        solver = self.solver
        mesh = solver.mesh
        if self._by_half is None:
            self._by_half = {}
            for index, window in enumerate(self.windows):
                if window is not None:
                    self._by_half.setdefault(window[0], []).append(index)
        bary = mesh.barycentric(face, point)
        candidates = [int(face)] + [int(mesh.neighbors[face, (k + 1) % 3]) for k in range(3)
                                    if bary[k] <= ON_EDGE and mesh.neighbors[face, (k + 1) % 3] >= 0]
        best = (math.inf, int(face), None)
        for f in candidates:
            for vertex in mesh.faces[f]:
                value = float(self.distances[vertex]) + float(np.linalg.norm(mesh.vertices[vertex] - point))
                if value < best[0]:
                    best = (value, f, -1 - int(vertex))
            for h in range(3 * f, 3 * f + 3):
                tolerance = PARAMETER_TOLERANCE * solver.half[h][0]
                for index in self._by_half.get(h, ()):
                    _, b0, b1, sx, sy, sigma, _ = self.windows[index]
                    x, px, py = self._ray_crossing(h, point, sx, sy)
                    if b0 - tolerance <= x <= b1 + tolerance:
                        value = sigma + math.hypot(px - sx, py - sy)
                        if value < best[0]:
                            best = (value, f, index)
        return best


def insert_points(mesh: G.TriMesh, points) -> tuple:
    """(mesh, vertex ids) with every surface point of ``points`` = [(face, xyz)] made a vertex.

    A point inside a face splits it 1-to-3 about the point; a point within
    ``ON_EDGE`` (barycentric) of an edge splits both faces at that edge
    1-to-2; a point at a vertex is that vertex. The point is first projected
    onto the face plane and, within ``ON_EDGE`` of an edge, moved onto that
    edge from the opposite corner (by about ``ON_EDGE`` times an edge length
    at most), so the new faces lie in the plane of the face they replace,
    keep its orientation, and leave the polyhedral metric unchanged. The face
    of a later point is looked up again in the refined mesh. A point on no
    face is refused with ``point_outside_face``.
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
        bary[small] = 0.0
        point = current.point(face, bary / bary.sum())  # on the face plane, and on the edge when near one
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
