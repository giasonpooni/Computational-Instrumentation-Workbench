"""Deterministic studies behind tasks T038-T044 (triangle-mesh geodesics and geometry uncertainty).

Each study returns plain numbers and lists; the task module turns them into
findings. Smooth references come from ``ciw.lab.surfaces`` (great circles,
helices, closed-form curvature) and ``ciw.lab.jacobi`` (the smooth transfer
matrix); exact polyhedral distances come from
``ciw.lab.surfaces_discrete_mesh_exact`` and are compared with pygeodesic and
potpourri3d when they import. Random inputs use seeded PCG64 generators only.

Non-claims: the meshes are generated from declared formulas in normalized
units and the noise models are declared Gaussian models. Nothing here is a
scanned surface, a sensor or a calibration, and a Monte Carlo agreement is
agreement between two computations on the same declared model.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math

import numpy as np

from . import jacobi
from .integrators import observed_order
from .surfaces import Sphere, Torus
from . import surfaces_discrete_mesh_exact as E
from . import surfaces_discrete_mesh_geometry as G

SEED = 20260938
UNIT_SPHERE = Sphere(1.0)
# Test geodesics on the unit sphere: (chart point (theta, phi), heading) with generic values
# so that no trace is aimed along a lattice row or through a vertex.
STARTS = (((1.1, 0.3), 0.7), ((0.8, 2.0), 2.1), ((1.9, 4.0), -1.3), ((1.4, 5.5), 0.2), ((0.6, 1.0), 2.9),
          ((2.3, 3.1), -2.4))
TRACE_LENGTH = 2.0
TRACE_LEVELS = (1, 2, 3, 4, 5, 6, 7)
VALENCE5_LIMIT = 4.5 - 1.5 * math.sqrt(5.0)  # 3 / (4 cos^2(pi / 5)) for a regular valence-5 star
EDGE_GRAPH_FLOOR = math.sqrt(5.0) - 2.0      # 1 / cos(pi / 5) - 1: two hops between rows 72 degrees apart
# Declared marker geodesic for T043/T044: STARTS[5] has the largest vertex margin of the six on
# icosphere-3 and stays in its face corridor for sigma <= 1e-3. The largest margin does not make it the
# most robust strip at larger noise (T043 records a smaller-margin strip that leaves its corridor less
# often at sigma >= 3e-3); corridor_study reports all six.
MARKER_START = 5
ORIGIN = np.zeros(3)
# Seed offsets (from SEED) of the jitter fold-rate study; disjoint from quality_study's seeds 1-3.
FOLD_SEEDS = tuple(range(1001, 1061))
# Declared jittered icosphere-3 with an inverted face whose bends stay below the fold threshold (T042).
INVERTED_JITTER = (0.2, 1002)


def optional_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    return str(__import__(name).__version__)


def fitted_order(hs, errors) -> float:
    return float(observed_order(hs, errors))


def _angle(a, b) -> float:
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b))))


# ---------------------------------------------------------------- sphere traces
def sphere_start(mesh, u0, heading, surface=UNIT_SPHERE):
    """Mesh start on the ray through the smooth start; direction = smooth tangent projected to the face."""
    u0 = np.asarray(u0, dtype=float)
    t0 = surface.unit_tangent(u0, heading)
    x0 = surface.embedding(u0)
    tangent = surface.embedding_jacobian(u0) @ t0
    face, point = mesh.locate_ray(x0)
    return {"u0": u0, "heading": float(heading), "chart_tangent": t0, "x0": x0, "tangent": tangent,
            "face": face, "point": point, "direction": mesh.tangent(face, tangent)}


def great_circle_errors(start, end_point, length):
    """Endpoint angle error, cross-track and along-track components, and the length defect."""
    x0, t0 = start["x0"], start["tangent"]
    smooth = math.cos(length) * x0 + math.sin(length) * t0
    ahead = -math.sin(length) * x0 + math.cos(length) * t0
    binormal = np.cross(x0, t0)
    e = end_point / np.linalg.norm(end_point)
    return {"endpoint": _angle(e, smooth), "cross_track": float(math.asin(np.clip(e @ binormal, -1, 1))),
            "along_track": float(math.atan2(e @ ahead, e @ smooth)), "length_defect": _angle(x0, e) - length}


def sphere_trace_study(levels=TRACE_LEVELS, starts=STARTS, length=TRACE_LENGTH):
    rows = []
    for level in levels:
        mesh = G.icosphere(level)
        per = []
        for u0, heading in starts:
            start = sphere_start(mesh, u0, heading)
            tr = G.trace(mesh, start["face"], start["point"], start["direction"], length)
            record = {"status": tr.status, "faces": len(tr.faces), "min_vertex_margin": float(tr.min_vertex_margin)}
            if tr.completed:
                record.update(great_circle_errors(start, tr.end_point, length))
                ids = sorted(set(mesh.faces[tr.faces].ravel().tolist()))
                local = {v: i for i, v in enumerate(ids)}
                strip = [tuple(local[int(v)] for v in mesh.faces[f]) for f in tr.faces]
                a = mesh.barycentric(tr.faces[0], start["point"])
                b = mesh.barycentric(tr.end_face, tr.end_point)
                distance, margin = G.strip_unfold_distance(mesh.vertices[ids][None], strip, a, b)
                record["unfold_minus_trace"] = float(distance[0] - tr.length)
                record["unfold_margin"] = float(margin[0])
            per.append(record)
        done = [r for r in per if r["status"] == "completed"]
        rows.append({"level": level, "h": mesh.mean_edge(), "vertices": len(mesh.vertices), "traces": per,
                     "completed": len(done),
                     "mean_endpoint": float(np.mean([r["endpoint"] for r in done])),
                     "max_endpoint": float(np.max([r["endpoint"] for r in done])),
                     "mean_cross_track": float(np.mean([abs(r["cross_track"]) for r in done])),
                     "mean_along_track": float(np.mean([abs(r["along_track"]) for r in done])),
                     "mean_length_defect": float(np.mean([abs(r["length_defect"]) for r in done])),
                     "max_unfold_minus_trace": float(np.max([abs(r["unfold_minus_trace"]) for r in done]))})
    return {"length": length, "starts": [list(s[0]) + [s[1]] for s in starts], "rows": rows}


# ---------------------------------------------------------------- cylinder helices
def cylinder_study(ns=(8, 16, 32, 64, 128), radius=1.0, height=4.0, alpha=0.5, length=3.0, z0=0.5, sector=1):
    """Prism cylinder helices against the exact mesh development and the smooth helix.

    The endpoint lies on the development at chord i, fraction t, so its azimuth
    is 2 a i + a + atan((2t - 1) tan a) with a = pi / n. Against the smooth
    azimuth this gives the exact error R |(L cos(alpha) / R)(a / sin a - 1) +
    atan(s tan a) - s a|, s = 2t - 1: a development deficit ~ L cos(alpha) a^2 / 6
    plus a chord-position term bounded by 2 a^3 / (9 sqrt 3) (1 + O(a^2)).
    """
    rows = []
    leading = length * math.cos(alpha) * math.pi ** 2 / 6.0
    for n in ns:
        m = max(4, int(round(height / (2 * radius * math.sin(math.pi / n)))))
        mesh = G.cylinder_mesh(n, m, radius, height)
        mid = 2 * math.pi * (sector + 0.5) / n
        # Chord midpoint: the chord is parallel to the smooth tangent there, so the heading is exact.
        inset = radius * math.cos(math.pi / n)
        point = np.array([inset * math.cos(mid), inset * math.sin(mid), z0])
        face, _ = mesh.locate(point)
        direction = np.array([-math.cos(alpha) * math.sin(mid), math.cos(alpha) * math.cos(mid), math.sin(alpha)])
        tr = G.trace(mesh, face, point, direction, length)
        chord = 2 * radius * math.sin(math.pi / n)
        x = ((sector + 0.5) * chord + length * math.cos(alpha)) % (n * chord)
        i = int(x // chord)
        t = x / chord - i

        def ring(k):
            return radius * np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)])

        xy = (1 - t) * ring(i) + t * ring(i + 1)
        developed = np.array([xy[0], xy[1], z0 + length * math.sin(alpha)])
        phi = mid + length * math.cos(alpha) / radius
        end = tr.end_point
        dphi = (math.atan2(end[1], end[0]) - phi + math.pi) % (2 * math.pi) - math.pi
        error = math.hypot(radius * dphi, end[2] - (z0 + length * math.sin(alpha)))
        a = math.pi / n
        deficit = length * math.cos(alpha) * (a / math.sin(a) - 1)
        s = 2 * t - 1
        chord_term = radius * (math.atan(s * math.tan(a)) - s * a)
        rows.append({"n": n, "rings": m + 1, "h": mesh.mean_edge(), "status": tr.status,
                     "development_error": float(np.max(np.abs(end - developed))), "helix_error": error,
                     "circumference_prediction": deficit, "chord_position_term": chord_term,
                     "exact_prediction": abs(deficit + chord_term), "chord_fraction": t,
                     "chord_term_bound": radius * 2 * a ** 3 / (9 * math.sqrt(3)),
                     "scaled_error": error * n * n, "z_error": abs(end[2] - (z0 + length * math.sin(alpha)))})
    return {"radius": radius, "alpha": alpha, "length": length, "leading_constant": leading, "rows": rows}


# ---------------------------------------------------------------- planar meshes
def plane_study(shears=(0.0, 0.5, 1.0, 1.5), size=8, length=0.2, angles=(0.3, 1.9, 3.7, 5.1)):
    """Straightest geodesics on sheared planar grids are exact lines; edge-graph distances are not."""
    rows = []
    for shear in shears:
        mesh = G.plane_mesh(size, size, shear=shear)
        start = np.array([0.5 + 0.5 * shear + 0.013, 0.5 + 0.007, 0.0])
        face, _ = mesh.locate(start)
        errors, statuses = [], []
        for angle in angles:
            d = np.array([math.cos(angle), math.sin(angle), 0.0])
            tr = G.trace(mesh, face, start, d, length)
            statuses.append(tr.status)
            if tr.completed:
                errors.append(float(np.max(np.abs(tr.end_point - (start + length * d)))))
        # The sheared domain is convex, so the intrinsic distance from corner 0 is Euclidean.
        dist = G.edge_distances(mesh, 0)[1:]
        euclid = np.linalg.norm(mesh.vertices[1:] - mesh.vertices[0], axis=1)
        quality = mesh.quality()
        rows.append({"shear": shear, "min_angle_deg": quality["min_angle_deg"],
                     "max_radius_ratio": quality["max_radius_ratio"], "statuses": statuses,
                     "max_trace_error": max(errors) if errors else None,
                     "max_graph_excess": float(np.max(dist / euclid - 1)),
                     "mean_graph_excess": float(np.mean(dist / euclid - 1))})
    return {"size": size, "length": length, "rows": rows}


# ---------------------------------------------------------------- graph and heat distances
def distance_study(levels=(1, 2, 3, 4), ks=(1, 3), heat_levels=(1, 2, 3)):
    """Graph and heat-method distances from vertex 0 against great-circle distances.

    The dense heat method stops at level 3 (642 vertices) so that a direct solve
    stays cheap even when the machine's BLAS threads are contended.
    """
    rows = []
    for level in levels:
        mesh = G.icosphere(level)
        n = len(mesh.vertices)
        true = np.arccos(np.clip(mesh.vertices @ mesh.vertices[0], -1, 1))
        others = np.arange(n) != 0
        row = {"level": level, "h": mesh.mean_edge(), "vertices": n}
        # Signed relative errors can be negative (the inscribed mesh is shorter than the sphere);
        # the *_max_abs_rel fields are magnitudes for log plots and floors.
        edge = G.edge_distances(mesh, 0)[others] / true[others] - 1
        row["edge_max_signed_rel"] = float(np.max(edge))
        row["edge_max_abs_rel"] = float(np.max(np.abs(edge)))
        row["edge_mean_signed_rel"] = float(np.mean(edge))
        for k in ks:
            (indptr, indices, weights), _, _ = G.steiner_graph(mesh, k)
            d = G.dijkstra(indptr, indices, weights, 0)[:n][others] / true[others] - 1
            row[f"steiner{k}_max_signed_rel"] = float(np.max(d))
            row[f"steiner{k}_min_signed_rel"] = float(np.min(d))
            row[f"steiner{k}_max_abs_rel"] = float(np.max(np.abs(d)))
        if level in heat_levels:
            heat = G.heat_distance(mesh, 0)
            row["heat_max_abs"] = float(np.max(np.abs(heat - true)))
            row["heat_rms"] = float(np.sqrt(np.mean((heat - true) ** 2)))
        rows.append(row)
    return {"source": "vertex 0 (valence 5)", "rows": rows}


def face_membership(mesh, points, tolerance=1e-12) -> np.ndarray:
    """Boolean (points, faces): the point lies on the closed face (plane offset and barycentric coordinates)."""
    points = np.asarray(points, dtype=float)
    p0 = mesh.vertices[mesh.faces[:, 0]]
    offset = np.einsum("pfi,fi->pf", points[:, None, :] - p0[None], mesh.face_normals)
    v = mesh.vertices[mesh.faces]
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    d = points[:, None, :] - p0[None]
    d11, d12, d22 = (np.einsum("fi,fi->f", x, y) for x, y in ((e1, e1), (e1, e2), (e2, e2)))
    d1, d2 = np.einsum("pfi,fi->pf", d, e1), np.einsum("pfi,fi->pf", d, e2)
    den = d11 * d22 - d12 * d12
    b1, b2 = (d22 * d1 - d12 * d2) / den, (d11 * d2 - d12 * d1) / den
    return (np.abs(offset) <= tolerance) & (b1 >= -tolerance) & (b2 >= -tolerance) & (1 - b1 - b2 >= -tolerance)


def edges_outside_faces(mesh, nodes, rows, cols) -> int:
    """Graph edges whose two end nodes lie on no common face, by a geometric test independent of the construction.

    A segment between two points of one (convex, planar) face lies in that face,
    so zero means every graph edge is a surface path.
    """
    packed = np.packbits(face_membership(mesh, nodes), axis=1)
    return int(np.sum(~np.any(packed[rows] & packed[cols], axis=1)))


def _graph_edges(indptr, indices):
    return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr)), np.asarray(indices)


def steiner_nested_study(level=2, ks=(0, 1, 3, 7), trace_length=1.0, starts=STARTS):
    """Nested Steiner graphs: distances cannot increase with k, every graph edge lies in a face, and traced
    geodesics are lower sandwiches."""
    mesh = G.icosphere(level)
    n = len(mesh.vertices)
    previous, increases, table = None, [], {}
    for k in ks:
        (indptr, indices, weights), _, _ = G.steiner_graph(mesh, k)
        d = G.dijkstra(indptr, indices, weights, 0)[:n]
        table[k] = d
        if previous is not None:
            increases.append(float(np.max(d - previous)))
        previous = d
    traced = []
    locations = []
    for u0, heading in starts:
        start = sphere_start(mesh, u0, heading)
        tr = G.trace(mesh, start["face"], start["point"], start["direction"], trace_length)
        if tr.completed:
            traced.append(tr.length)
            locations.append(((start["face"], start["point"]), (tr.end_face, tr.end_point)))
    gaps, outside, checked = {}, 0, 0
    extra = [loc for pair in locations for loc in pair]
    for k in ks:
        (indptr, indices, weights), ids, _ = G.steiner_graph(mesh, k, extra=extra)
        nodes = np.concatenate([G.steiner_nodes(mesh, k)] + [np.asarray(p, dtype=float)[None] for _, p in extra])
        rows, cols = _graph_edges(indptr, indices)
        outside += edges_outside_faces(mesh, nodes, rows, cols)
        checked += len(rows) // 2
        if k in ks[1:]:
            gaps[k] = [float(G.dijkstra(indptr, indices, weights, ids[2 * i], ids[2 * i + 1])[ids[2 * i + 1]] - ell)
                       for i, ell in enumerate(traced)]
    return {"level": level, "ks": list(ks), "max_increase_with_k": max(increases),
            "edges_outside_faces": outside, "edges_checked": checked, "traced_lengths": traced,
            "gap_to_traced": {str(k): v for k, v in gaps.items()},
            "min_gap": min(min(v) for v in gaps.values()), "vertex0_to_last": {str(k): float(table[k][-1]) for k in ks}}


def dijkstra_independent(level=3):
    """ciw heap Dijkstra against scipy.sparse.csgraph (when installed) and a dense Floyd-Warshall."""
    mesh = G.icosphere(level)
    indptr, indices, weights = G.edge_graph(mesh)
    ours = G.dijkstra(indptr, indices, weights, 0)
    result = {"level": level, "vertices": len(mesh.vertices), "scipy": optional_version("scipy")}
    if result["scipy"]:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra as sp_dijkstra
        graph = csr_matrix((weights, indices, indptr), shape=(len(ours), len(ours)))
        result["scipy_max_abs"] = float(np.max(np.abs(sp_dijkstra(graph, indices=0) - ours)))
    small = G.icosphere(min(level, 2))
    ip, ix, w = G.edge_graph(small)
    n = len(small.vertices)
    dense = np.full((n, n), np.inf)
    np.fill_diagonal(dense, 0.0)
    rows = np.repeat(np.arange(n), np.diff(ip))
    dense[rows, ix] = w
    for k in range(n):
        dense = np.minimum(dense, dense[:, k:k + 1] + dense[k:k + 1, :])
    result["floyd_level"] = min(level, 2)
    result["floyd_max_abs"] = float(np.max(np.abs(dense[0] - G.dijkstra(ip, ix, w, 0))))
    return result


# ---------------------------------------------------------------- exact polyhedral distances
# Meshes of the exact-distance study, three sources each (vertex 0 of an icosphere has valence 5). The
# torus has 120 saddle vertices (angle sum above 2 pi), where shortest paths may bend; its 48 flat vertices are
# pseudo-sources too (surfaces_discrete_mesh_exact.ANGLE_TOLERANCE).
EXACT_LEVELS = (1, 2, 3, 4)
EXACT_TORUS = (24, 12)
FLIPOUT_MESHES = ("icosphere-2", "icosphere-3", "torus-24x12")
STEINER_LEVELS = (2, 3)
HEAT_LEVELS = (1, 2, 3)
# Traced geodesics compared with the exact distance between their endpoints, as (level, length): length 2 is the
# convergence study's trace length, and the length-1 traces on level 2 are those of the Steiner comparison.
TRACED_EXACT = ((1, 2.0), (2, 2.0), (3, 2.0), (4, 2.0), (2, 1.0))
# A path within this of the exact distance is a shortest path; rounding stays below 1e-14 on these meshes.
SHORTEST = 1e-10


def package_version(name: str) -> str | None:
    """Distribution version of an optional package that imports here (potpourri3d has no ``__version__``)."""
    if importlib.util.find_spec(name) is None:
        return None
    try:
        importlib.import_module(name)
        return importlib.metadata.version(name)
    except Exception:  # installed but not importable here: absent
        return None


def exact_sources(mesh) -> tuple:
    n = len(mesh.vertices)
    return 0, n // 3, 2 * n // 3


def exact_meshes(levels=EXACT_LEVELS, torus=EXACT_TORUS) -> list:
    return [G.icosphere(level) for level in levels] + [G.torus_mesh(*torus)]


def exact_distance_study(levels=EXACT_LEVELS, torus=EXACT_TORUS, flipout=FLIPOUT_MESHES):
    """ciw exact distances from three sources to every vertex, with the checks that need no other implementation.

    Symmetry compares d(a -> b) with d(b -> a) between the sources, which propagate different windows; the edge
    excess max(|d(u) - d(v)| - |uv|) over edges is at most 0 for a distance (the edge is a surface path); and on
    the ``flipout`` meshes the edge-graph path from each source to every other vertex is never shorter.
    """
    rows = []
    for mesh in exact_meshes(levels, torus):
        solver = E.ExactGeodesic(mesh)
        sources = exact_sources(mesh)
        solved = [solver.solve(source) for source in sources]
        d = np.array([distances for distances, _ in solved])
        a, b = mesh.edges[:, 0], mesh.edges[:, 1]
        row = {"mesh": mesh.name, "vertices": len(mesh.vertices), "h": mesh.mean_edge(), "sources": list(sources),
               "saddle_vertices": int(np.sum(solver.excess > E.ANGLE_TOLERANCE)),
               "pseudo_source_vertices": int(sum(solver.pseudo)), "distances": d,
               "windows": [counters["windows"] for _, counters in solved],
               "symmetry_max_abs": float(max(abs(d[i, sources[j]] - d[j, sources[i]])
                                             for i in range(3) for j in range(i + 1, 3))),
               "edge_excess": float(np.max(np.abs(d[:, a] - d[:, b]) - mesh.edge_lengths[None])),
               "unreached": int(np.sum(~np.isfinite(d)))}
        if mesh.name in flipout:
            gaps = [np.delete(G.edge_distances(mesh, source) - distances, source)
                    for source, distances in zip(sources, d)]
            row["pairs"] = int(sum(len(g) for g in gaps))
            row["edge_graph_min_excess"] = float(min(np.min(g) for g in gaps))
        rows.append(row)
    return {"levels": list(levels), "torus": list(torus), "flipout": list(flipout), "rows": rows}


def exact_independent_study(exact):
    """pygeodesic (Kirsanov's exact MMP) and potpourri3d (geometry-central's FlipOut) on the same meshes.

    A package is compared only when it imports here. pygeodesic returns exact distances from each source to
    every vertex. FlipOut shortens the edge-graph path between two vertices to a locally shortest geodesic,
    whose length is at least the exact distance and equals it when that geodesic is globally shortest.
    """
    result = {"pygeodesic": package_version("pygeodesic"), "potpourri3d": package_version("potpourri3d")}
    meshes = exact_meshes(exact["levels"], exact["torus"])
    if result["pygeodesic"]:
        from pygeodesic.geodesic import PyGeodesicAlgorithmExact
        worst, compared = 0.0, 0
        for mesh, row in zip(meshes, exact["rows"]):
            algorithm = PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces.astype(np.int32))
            for source, distances in zip(row["sources"], row["distances"]):
                reference, _ = algorithm.geodesicDistances(np.array([source], dtype=np.int32), None)
                worst = max(worst, float(np.max(np.abs(np.asarray(reference) - distances))))
                compared += len(distances)
        result.update(pygeodesic_max_abs=worst, pygeodesic_compared=compared)
    if result["potpourri3d"] and exact["flipout"]:
        from potpourri3d import EdgeFlipGeodesicSolver
        per_mesh = []
        for mesh, row in zip(meshes, exact["rows"]):
            if row["mesh"] not in exact["flipout"]:
                continue
            solver = EdgeFlipGeodesicSolver(mesh.vertices, mesh.faces)
            excess = np.array([float(np.sum(np.linalg.norm(np.diff(solver.find_geodesic_path(source, target), axis=0),
                                                           axis=1))) - distances[target]
                               for source, distances in zip(row["sources"], row["distances"])
                               for target in range(len(mesh.vertices)) if target != source])
            per_mesh.append({"mesh": row["mesh"], "pairs": len(excess), "min_excess": float(excess.min()),
                             "max_excess": float(excess.max()), "shortest": int(np.sum(np.abs(excess) <= SHORTEST))})
        result["flipout"] = per_mesh
        result["flipout_min_excess"] = min(r["min_excess"] for r in per_mesh)
    return result


def exact_comparison_study(exact, steiner_levels=STEINER_LEVELS, ks=(1, 3, 7), heat_levels=HEAT_LEVELS):
    """Edge-graph, Steiner-graph and heat-method distances from vertex 0 against the exact distances."""
    spheres = {mesh.params["level"]: (mesh, row["distances"][0])
               for mesh, row in zip(exact_meshes(exact["levels"], exact["torus"]), exact["rows"])
               if mesh.name.startswith("icosphere")}
    edge_rows = []
    for level, (mesh, d) in sorted(spheres.items()):
        edge = G.edge_distances(mesh, 0)[1:]
        edge_rows.append({"level": level, "h": mesh.mean_edge(), "min_excess": float(np.min(edge - d[1:])),
                          "max_relative_excess": float(np.max(edge / d[1:] - 1))})
    steiner_rows = []
    for level in steiner_levels:
        mesh, d = spheres[level]
        row = {"level": level, "ks": list(ks), "min_excess": [], "mean_excess": [], "max_relative_excess": []}
        for k in ks:
            (indptr, indices, weights), _, _ = G.steiner_graph(mesh, k)
            steiner = G.dijkstra(indptr, indices, weights, 0)[1:len(d)]
            row["min_excess"].append(float(np.min(steiner - d[1:])))
            row["mean_excess"].append(float(np.mean(steiner - d[1:])))
            row["max_relative_excess"].append(float(np.max(steiner / d[1:] - 1)))
        steiner_rows.append(row)
    heat_rows = []
    for level in heat_levels:
        mesh, d = spheres[level]
        heat = G.heat_distance(mesh, 0)
        smooth = np.arccos(np.clip(mesh.vertices @ mesh.vertices[0], -1, 1))
        heat_rows.append({"level": level, "h": mesh.mean_edge(), "max_abs_error": float(np.max(np.abs(heat - d))),
                          "rms_error": float(np.sqrt(np.mean((heat - d) ** 2))),
                          "exact_minus_smooth_max": float(np.max(np.abs(d - smooth)))})
    return {"source": "vertex 0 (valence 5)", "edge_graph": edge_rows, "steiner": steiner_rows, "heat": heat_rows}


def traced_exact_study(configs=TRACED_EXACT, starts=STARTS):
    """Straightest geodesics against the exact distance between their endpoints, both inserted as vertices.

    The propagation from the start runs to the traced length, so the exact distance from the start is also known
    at every edge crossing of the trace (``cut_point``): where the trace stops being a shortest path.
    """
    rows = []
    for level, length in configs:
        mesh = G.icosphere(level)
        defect = G.angle_defect(mesh)[0]
        for index, (u0, heading) in enumerate(starts):
            start = sphere_start(mesh, u0, heading)
            tr = G.trace(mesh, start["face"], start["point"], start["direction"], length)
            row = {"level": level, "length": length, "start": index, "status": tr.status, "h": mesh.mean_edge()}
            if tr.completed:
                refined, (a, b) = E.insert_points(mesh, [(start["face"], start["point"]),
                                                         (tr.end_face, tr.end_point)])
                propagation = E.ExactGeodesic(refined).propagate(a, limit=tr.length)
                exact = float(propagation.distances[b])
                row.update(traced=tr.length, exact=exact, excess=tr.length - exact,
                           shortest=bool(abs(tr.length - exact) <= SHORTEST))
                row["cut"] = cut_point(mesh, defect, refined, propagation, tr, b)
            rows.append(row)
    return {"configs": [list(c) for c in configs], "shortest_tolerance": SHORTEST, "rows": rows}


def arclengths(points) -> np.ndarray:
    """Cumulative polyline length at every point."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1))]


def _nearest_on_polyline(points, polyline):
    """(distance, arclength of the nearest polyline point) for each point, by projection onto every segment."""
    points, polyline = np.asarray(points, dtype=float), np.asarray(polyline, dtype=float)
    a, ab = polyline[:-1], np.diff(polyline, axis=0)
    length = np.linalg.norm(ab, axis=1)
    t = np.clip(np.einsum("psi,si->ps", points[:, None] - a[None], ab) / np.maximum(length ** 2, 1e-300), 0.0, 1.0)
    gap = np.linalg.norm(points[:, None] - (a[None] + t[..., None] * ab[None]), axis=2)
    best = np.argmin(gap, axis=1)
    rows = np.arange(len(points))
    return gap[rows, best], arclengths(polyline)[best] + t[rows, best] * length[best]


# Cut points: an excess of traced length over the exact distance above CUT_EXCESS (rounding stays below 1e-14 here)
# marks a point past the cut point; bisection locates it; the other shortest path is traced from CUT_PAST mean
# edges past it.
CUT_EXCESS = 1e-12
CUT_BISECTIONS = 40
CUT_PAST = 0.02


def cut_point(mesh, defect, refined, propagation, tr, end) -> dict:
    """Where a straightest geodesic on a sphere mesh stops being a shortest path from its start (its cut point).

    The excess e(s) = s - d(s) of traced length over the exact distance from the start never decreases along
    the trace (a subpath of a shortest path is shortest). It is sampled at every edge crossing and at the
    endpoint, ``d`` coming from the recorded windows (at the endpoint it must equal the vertex distance). The
    first sample whose excess passes CUT_EXCESS brackets the cut point with the sample before it; bisection
    inside that face segment finds where the excess passes CUT_EXCESS, and since the excess rises linearly past
    the cut point, moving back by CUT_EXCESS over its slope gives the cut point (``arclength``), with that move
    plus the bisection width as its ``resolution``. Just past it the other shortest path is back-traced from
    the recorded windows; with the trace it bounds a digon, whose ``enclosed`` vertices (a gnomonic
    point-in-polygon test, exact for polylines on a mesh inscribed in a sphere about the origin) carry the
    curvature that closes it. ``trigger`` is the passed vertex whose isolated-cone cut ray the trace would
    cross first (``single_cone_cut``); ``nearest`` is the passed vertex closest to the trace before the cut.
    """
    points = np.asarray(tr.points)
    s = arclengths(points)

    def at(arclength, j):
        """The trace point at an arclength inside segment j (from point j to point j + 1)."""
        t = 0.0 if s[j + 1] == s[j] else (arclength - s[j]) / (s[j + 1] - s[j])
        return points[j] + t * (points[j + 1] - points[j])

    def excess_at(arclength, j):
        x = at(arclength, j)
        return float(arclength - propagation.distance_at(refined.locate(x)[0], x))

    excess = [excess_at(s[j + 1], j) for j in range(len(points) - 1)]
    end_gap = float(s[-1] - excess[-1] - propagation.distances[end])
    first = next((j for j, e in enumerate(excess) if e > CUT_EXCESS), None)
    h = mesh.mean_edge()
    result = {"samples": len(excess), "max_excess_decrease": max((a - b for a, b in zip(excess, excess[1:])),
                                                                  default=0.0),
              "endpoint_window_minus_vertex": end_gap, "bracket": None, "arclength": None, "resolution": None,
              "enclosed": None, "enclosed_defect": None, "probes": None}
    if first is not None:
        lo, hi = float(s[first]), float(s[first + 1])
        for _ in range(CUT_BISECTIONS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if excess_at(mid, first) <= CUT_EXCESS else (lo, mid)
        # The excess rises linearly past the cut point: extrapolate its zero from the bisected CUT_EXCESS crossing
        # with the secant slope to the sample past it; the correction bounds the error.
        slope = excess[first] / max(float(s[first + 1]) - hi, 1e-300)
        correction = CUT_EXCESS / slope
        past = min(hi + CUT_PAST * h, float(s[-1]))
        j = min(int(np.searchsorted(s, past)) - 1, len(points) - 2)
        x = at(past, j)
        other = propagation.path_to_point(refined.locate(x)[0], x)
        loop = np.vstack([points[:j + 1], [x], other["points"][1:-1]])
        enclosed = enclosed_vertices(mesh.vertices, loop)
        # Probe points CUT_PAST mean edges before and after the cut point, for an independent exact solver.
        probes = {"arclengths": [], "segments": [], "points": [], "distances": []}
        for arclength in (max(hi - correction - CUT_PAST * h, 0.0), past):
            k = min(max(int(np.searchsorted(s, arclength)) - 1, 0), len(points) - 2)
            x = at(arclength, k)
            probes["arclengths"].append(float(arclength))
            probes["segments"].append(k)
            probes["points"].append(x)
            probes["distances"].append(float(propagation.distance_at(refined.locate(x)[0], x)))
        result.update(bracket=[float(s[first]), float(s[first + 1])], arclength=hi - correction,
                      resolution=correction + (s[first + 1] - s[first]) / 2 ** CUT_BISECTIONS,
                      enclosed=enclosed, enclosed_defect=float(sum(defect[v] for v in enclosed)), probes=probes)
    upto = len(points) if first is None else first + 2  # the polyline up to the sample past the cut point
    gap, along = _nearest_on_polyline(mesh.vertices, points[:upto])
    passed = np.flatnonzero((along > 0.0) & (along < s[upto - 1]))
    nearest = int(passed[np.argmin(gap[passed])])
    predictions = np.array([single_cone_cut(along[v], gap[v], defect[v]) for v in passed])
    trigger = int(passed[np.argmin(predictions)]) if np.isfinite(predictions).any() else None
    result["nearest"] = {"vertex": nearest, "distance_over_h": float(gap[nearest] / h),
                         "arclength": float(along[nearest]), "angle_defect": float(defect[nearest])}
    result["trigger"] = None if trigger is None else {
        "vertex": trigger, "prediction": float(np.min(predictions)), "distance_over_h": float(gap[trigger] / h),
        "arclength": float(along[trigger]), "angle_defect": float(defect[trigger])}
    return result


def enclosed_vertices(vertices, loop) -> list:
    """Vertices inside a closed polyline loop drawn on a mesh inscribed in a sphere about the origin.

    Central (gnomonic) projection about the loop's mean direction maps every
    straight segment to a straight segment (it lies in a plane through the
    origin), so an even-odd test in the projection plane is exact for the
    polyline; the loop must lie in the open hemisphere about that direction.
    Of a loop's two sides the one in that hemisphere is taken.
    """
    loop = np.asarray(loop, dtype=float)
    center = loop.mean(axis=0)
    center /= np.linalg.norm(center)
    t1 = np.cross(center, [0.0, 0.0, 1.0] if abs(center[2]) < 0.9 else [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(center, t1)
    if np.min(loop @ center) <= 0.0:
        raise ValueError("The loop leaves the hemisphere about its mean direction")
    polygon = np.stack([loop @ t1, loop @ t2], axis=1) / (loop @ center)[:, None]
    front = np.flatnonzero(np.asarray(vertices) @ center > 0.0)
    q = np.stack([vertices[front] @ t1, vertices[front] @ t2], axis=1) / (vertices[front] @ center)[:, None]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    straddles = (a[None, :, 1] > q[:, None, 1]) != (b[None, :, 1] > q[:, None, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = a[None, :, 0] + (q[:, None, 1] - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (
            b[None, :, 1] - a[None, :, 1])
    inside = np.sum(straddles & (x > q[:, None, 0]), axis=1) % 2 == 1
    return [int(v) for v in front[inside]]


def single_cone_cut(along, lateral, defect) -> float:
    """Arclength at which a straight line crosses the cut ray of one cone vertex of angle defect ``defect``.

    With every other vertex flat, the cut locus of the start is the ray from the vertex directly away from it.
    A line from the start that passes the vertex, seen at angle phi from the line at distance r (in the
    development, ``along`` = r cos phi and ``lateral`` = r sin phi), crosses that ray at arclength
    r sin(defect / 2) / sin(defect / 2 - phi) when phi < defect / 2, and never otherwise.
    """
    phi, r = math.atan2(lateral, along), math.hypot(along, lateral)
    if not 0.0 < defect or phi >= 0.5 * defect:
        return math.inf
    return r * math.sin(0.5 * defect) / math.sin(0.5 * defect - phi)


def _l_shape(size=8) -> G.TriMesh:
    """The unit-square grid without its upper-right quadrant: one reflex boundary corner, at (0.5, 0.5)."""
    plane = G.plane_mesh(size, size)
    centroid = plane.vertices[plane.faces].mean(axis=1)
    faces = plane.faces[~((centroid[:, 0] > 0.5) & (centroid[:, 1] > 0.5))]
    used = np.unique(faces)
    index = np.full(len(plane.vertices), -1)
    index[used] = np.arange(len(used))
    return G.TriMesh.build(plane.vertices[used], index[faces], f"l-shape-{size}", params={"size": size})


def _l_distance(source, target, corner=(0.5, 0.5)) -> tuple:
    """(distance, bent) in the L-shape: the segment when it avoids the open removed quadrant, else via the corner."""
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        a, b = source[axis], target[axis]
        if a == b:
            lo, hi = (lo, hi) if a > corner[axis] else (1.0, 0.0)
        elif b > a:
            lo = max(lo, (corner[axis] - a) / (b - a))
        else:
            hi = min(hi, (corner[axis] - a) / (b - a))
    if lo >= hi:
        return float(np.linalg.norm(target[:2] - source[:2])), False
    c = np.asarray(corner, dtype=float)
    return float(np.linalg.norm(c - source[:2]) + np.linalg.norm(target[:2] - c)), True


def exact_analytic_study(shears=(0.0, 0.5, 1.0, 1.5), size=8, ns=(8, 16, 32), radius=1.0, height=4.0, z0=0.5,
                         sector=1, cube=4):
    """Exact distances where the polyhedral distance has a closed form.

    Sheared planar grids are convex, so distances are Euclidean; in the L-shape a hidden target is reached
    through the reflex corner; the prism cylinder develops to a flat strip of circumference 2 n R sin(pi / n),
    so a distance is the shortest segment to the periodic images; and on the unit cube the corner distances are
    1, sqrt 2 and sqrt 5 (two faces unfolded). Sources are a vertex and a point inserted inside a face.
    """
    planes = []
    for shear in shears:
        mesh = G.plane_mesh(size, size, shear=shear)
        point = np.array([0.5 + 0.5 * shear + 0.013, 0.5 + 0.007, 0.0])  # plane_study's start
        refined, (inserted,) = E.insert_points(mesh, [(mesh.locate(point)[0], point)])
        solver = E.ExactGeodesic(refined)
        planes.append({"shear": shear, "max_error": max(
            float(np.max(np.abs(solver.distances(s) - np.linalg.norm(refined.vertices - refined.vertices[s], axis=1))))
            for s in (0, inserted))})
    shape = _l_shape(size)
    solver = E.ExactGeodesic(shape)
    l_errors, bent = [], 0
    for corner in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        source = int(np.argmin(np.linalg.norm(shape.vertices - corner, axis=1)))
        reference = [_l_distance(shape.vertices[source], v) for v in shape.vertices]
        bent += sum(b for _, b in reference)
        l_errors.append(float(np.max(np.abs(solver.distances(source) - np.array([d for d, _ in reference])))))
    cylinders = []
    for n in ns:
        m = max(4, int(round(height / (2 * radius * math.sin(math.pi / n)))))  # as in cylinder_study
        mesh = G.cylinder_mesh(n, m, radius, height)
        chord = 2 * radius * math.sin(math.pi / n)
        mid = 2 * math.pi * (sector + 0.5) / n
        inset = radius * math.cos(math.pi / n)
        point = np.array([inset * math.cos(mid), inset * math.sin(mid), z0])
        refined, (inserted,) = E.insert_points(mesh, [(mesh.locate(point)[0], point)])
        # Development: ring j, column i at (i chord, j H / m); the inserted chord midpoint at
        # ((sector + 1/2) chord, z0).
        index = np.arange(len(mesh.vertices))
        along = np.r_[(index % n) * chord, (sector + 0.5) * chord]
        up = np.r_[(index // n) * height / m, z0]
        solver = E.ExactGeodesic(refined)
        errors = []
        for source in (0, inserted):
            reference = np.min([np.hypot(along - along[source] + k * n * chord, up - up[source]) for k in (-1, 0, 1)],
                               axis=0)
            errors.append(float(np.max(np.abs(solver.distances(source) - reference))))
        cylinders.append({"n": n, "rings": m + 1, "max_error": max(errors)})
    box = G.cube_mesh(cube)
    corners = [int(np.flatnonzero(np.all(box.vertices == c, axis=1))[0])
               for c in np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T]
    closed = [(0.0, 1.0, math.sqrt(2.0), math.sqrt(5.0))[int(box.vertices[c].sum())] for c in corners]
    d = E.ExactGeodesic(box).distances(corners[0])
    return {"planes": planes, "plane_max_error": max(r["max_error"] for r in planes),
            "l_shape": {"vertices": len(shape.vertices), "bent_targets": int(bent), "max_error": max(l_errors)},
            "cylinders": cylinders, "cylinder_max_error": max(r["max_error"] for r in cylinders),
            "cube": {"k": cube, "corner_distances": [float(d[c]) for c in corners], "closed_forms": closed,
                     "max_error": float(max(abs(d[c] - r) for c, r in zip(corners, closed))),
                     "edge_graph_far_corner": float(G.edge_distances(box, corners[0])[corners[-1]])}}


def insertion_study(level=2, torus=(12, 6)):
    """Exact distances between original vertices before and after inserting a face point and an edge point."""
    rows = []
    for mesh in (G.icosphere(level), G.torus_mesh(*torus)):
        f = len(mesh.faces) // 2
        points = [(f, mesh.point(f, [0.2, 0.3, 0.5])), (f + 1, mesh.point(f + 1, [0.35, 0.65, 0.0]))]
        refined, ids = E.insert_points(mesh, points)
        n = len(mesh.vertices)
        before, after = E.ExactGeodesic(mesh), E.ExactGeodesic(refined)
        worst = max(float(np.max(np.abs(after.distances(s)[:n] - before.distances(s)))) for s in (0, n // 2))
        rows.append({"mesh": mesh.name, "inserted": ids, "faces_added": len(refined.faces) - len(mesh.faces),
                     "issues": [code for code, _ in G.inspect(refined.vertices, refined.faces, require_closed=True)],
                     "max_abs_change": worst})
    return {"rows": rows, "max_abs_change": max(r["max_abs_change"] for r in rows)}


# ---------------------------------------------------------------- vertex continuation and shortest paths
PS = "polthier_schmies"
# Cube-corner traces: start (u, w, 0) on the face z = 0 aimed at the corner, then CUBE_RUN past it.
CUBE_STARTS = ((0.61, 0.23), (0.19, 0.53))
CUBE_RUN = 0.5
# One-vertex fans: total angles / pi (cones, flat, saddles), start and end radii, the start's polar angle as a
# fraction of the first apex angle, end polar angles from the start / pi, and the lateral offset (times the start
# radius) of the rays that pass the centre on either side.
FAN_ANGLES = (1.5, 1.8, 2.0, 2.2, 2.5)
FAN_TRIANGLES = 6
FAN_RADII = (0.45, 0.35)
FAN_START = 0.37
FAN_ENDS = (0.3, 0.6, 0.9, 1.1, 1.3)
FAN_OFFSET = 1e-6
# Vertex hits of generic traces: tangentially jittered icospheres (validated with the centre declared), seeded
# uniform start points and headings, and the edge-parameter distances to a vertex that are counted.
HIT_LEVELS = (2, 3, 4, 5)
HIT_JITTER = 0.1
HIT_TRACES = 200
HIT_LENGTH = 1.0
HIT_TAUS = (1e-1, 1e-2, 1e-3, 1e-4)
# Shortest-path back-tracing: seeded sources and targets per mesh.
PATH_SOURCES = 3
PATH_TARGETS = 6
# A path side angle this far below pi, or a turn this large at an edge, is a bend (rounding stays below 1e-12).
BEND = 1e-9


def _unit(vector) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _development_point(n, radius, height_step, x, z) -> np.ndarray:
    """The prism cylinder's point at development coordinates (x along the circumference, z up)."""
    chord = 2 * radius * math.sin(math.pi / n)
    i = math.floor(x / chord)
    t = x / chord - i
    ring = [radius * np.array([math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)]) for k in (i, i + 1)]
    xy = (1 - t) * ring[0] + t * ring[1]
    return np.array([xy[0], xy[1], z])


def _leaving_face(mesh, point, direction) -> int:
    """The face through which a ray from ``point`` (possibly a vertex or on an edge) leaves along ``direction``."""
    return mesh.locate(point + 1e-7 * mesh.mean_edge() * np.asarray(direction))[0]


def continuation_study(size=8, n=16, radius=1.0, height=4.0, cube=4):
    """Straightest geodesics continued through vertices by the Polthier-Schmies rule, against closed forms.

    Flat vertices (planar grids, prism cylinders): the trace must stay a straight line of the development,
    whether it passes one vertex or runs along a row of vertices. Cube corner (total angle 3 pi / 2, a cone): a
    ray on the face z = 0 aimed at the corner at angle gamma < pi / 4 from the x axis leaves on the face x = 0 at
    pi / 4 + gamma from the y axis (3 pi / 4 on both sides), and the exact distance between its endpoints is
    sqrt(r1^2 + r2^2 - 2 r1 r2 cos(3 pi / 4)) (the endpoint lies on the start's cut ray, reached around both
    sides), shorter than the continued path r1 + r2.
    """
    rows = []
    for shear, start, targets in ((0.0, (0.43, 0.61), ((0.75, 0.75), (0.25, 0.25))),
                                  (0.5, (0.763, 0.507), ((0.9375, 0.625), (0.4375, 0.375)))):
        mesh = G.plane_mesh(size, size, shear=shear)
        point = np.array([start[0], start[1], 0.0])
        face = mesh.locate(point)[0]
        for target in targets:
            direction = _unit(np.array([target[0], target[1], 0.0]) - point)
            tr = G.trace(mesh, face, point, direction, 0.5, vertex_rule=PS)
            rows.append({"mesh": mesh.name, "shear": shear, "kind": "aimed at a vertex", "status": tr.status,
                         "vertices_passed": len(tr.vertices),
                         "error": float(np.max(np.abs(tr.end_point - (point + 0.5 * direction))))})
    plane = G.plane_mesh(size, size)
    for vertex, direction, length, kind in (((1, 2), (1.0, 1.0), 0.8, "along a diagonal row of vertices"),
                                            ((1, 3), (1.0, 0.0), 0.7, "along a grid row of vertices")):
        point = plane.vertices[vertex[1] * (size + 1) + vertex[0]]
        direction = _unit(np.array([direction[0], direction[1], 0.0]))
        tr = G.trace(plane, _leaving_face(plane, point, direction), point, direction, length, vertex_rule=PS)
        rows.append({"mesh": plane.name, "shear": 0.0, "kind": kind, "status": tr.status,
                     "vertices_passed": len(tr.vertices),
                     "error": float(np.max(np.abs(tr.end_point - (point + length * direction))))})
    m = max(4, int(round(height / (2 * radius * math.sin(math.pi / n)))))  # as in cylinder_study
    cylinder = G.cylinder_mesh(n, m, radius, height)
    chord, step = 2 * radius * math.sin(math.pi / n), height / m
    for x0, z0, dx, dz, length, kind in (
            (1.5 * chord, 0.5, 2.5 * chord, 2.0 - 0.5, None, "aimed at a vertex"),
            (3 * chord, step, 0.0, 1.0, 2.5, "along a column of vertices"),
            (5 * chord, step, chord, step, 1.5, "along a diagonal row of vertices")):
        norm = math.hypot(dx, dz)
        dx, dz = dx / norm, dz / norm
        length = norm + 0.6 if length is None else length
        point = _development_point(n, radius, step, x0, z0)
        sector = int(math.floor(x0 / chord + 1e-9))
        tangent = _unit(_development_point(n, radius, step, (sector + 1) * chord, 0.0)
                        - _development_point(n, radius, step, sector * chord, 0.0))
        direction = dx * tangent + dz * np.array([0.0, 0.0, 1.0])
        tr = G.trace(cylinder, _leaving_face(cylinder, point, direction), point, direction, length, vertex_rule=PS)
        expected = _development_point(n, radius, step, x0 + length * dx, z0 + length * dz)
        rows.append({"mesh": cylinder.name, "shear": None, "kind": kind, "status": tr.status,
                     "vertices_passed": len(tr.vertices), "error": float(np.max(np.abs(tr.end_point - expected)))})
    box = G.cube_mesh(cube)
    corners = []
    for u, w in CUBE_STARTS:
        point = np.array([u, w, 0.0])
        face = box.locate(point)[0]
        r1 = float(np.linalg.norm(point))
        tr = G.trace(box, face, point, -point / r1, r1 + CUBE_RUN, vertex_rule=PS)
        gamma = math.atan2(w, u)
        turn = math.pi / 4 + min(gamma, math.pi / 2 - gamma)
        side = (0.0, CUBE_RUN * math.cos(turn), CUBE_RUN * math.sin(turn))
        expected = np.array(side if gamma < math.pi / 4 else (side[1], 0.0, side[2]))
        refined, (a, b) = E.insert_points(box, [(face, point), (tr.end_face, tr.end_point)])
        exact = float(E.ExactGeodesic(refined).distances(a, targets=[b])[b])
        closed = math.sqrt(r1 ** 2 + CUBE_RUN ** 2 - 2 * r1 * CUBE_RUN * math.cos(0.75 * math.pi))
        corners.append({"start": [u, w], "status": tr.status, "vertices_passed": len(tr.vertices),
                        "endpoint_error": float(np.max(np.abs(tr.end_point - expected))), "traced": tr.length,
                        "exact": exact, "closed_form": closed, "exact_minus_closed_form": exact - closed,
                        "traced_minus_exact": tr.length - exact})
    return {"flat": rows, "flat_max_error": max(r["error"] for r in rows),
            "flat_vertices_passed": [r["vertices_passed"] for r in rows], "cube": {"k": cube, "rows": corners}}


def fan_point(mesh, polar, radius):
    """(face, point) at a polar angle (around vertex 0, from ring vertex 1) and a distance from it on a fan."""
    n = mesh.params["n"]
    apex = mesh.params["total_angle"] / n
    turns = math.floor(polar / apex)
    face = int(turns) % n
    local = polar - turns * apex
    center = mesh.vertices[0]
    edge = _unit(mesh.vertices[1 + face] - center)
    return face, center + radius * (math.cos(local) * edge + math.sin(local) * np.cross(mesh.face_normals[face],
                                                                                        edge))


def fan_polar(mesh, face, point):
    """Polar angle (around vertex 0, from ring vertex 1) and distance from vertex 0 of a point of fan face ``face``."""
    apex = mesh.params["total_angle"] / mesh.params["n"]
    center = mesh.vertices[0]
    edge = _unit(mesh.vertices[1 + face] - center)
    offset = np.asarray(point) - center
    local = math.atan2(float(np.dot(np.cross(edge, offset), mesh.face_normals[face])), float(np.dot(edge, offset)))
    return face * apex + local, float(np.linalg.norm(offset))


def fan_study(angles=FAN_ANGLES, n=FAN_TRIANGLES, radii=FAN_RADII, start=FAN_START, ends=FAN_ENDS,
              offset=FAN_OFFSET):
    """The Polthier-Schmies continuation through one cone, flat or saddle vertex of total angle theta.

    Every fan is flat except at its centre, so a point at distance r2 and polar angle phi from a start at
    distance r1 is sqrt(r1^2 + r2^2 - 2 r1 r2 cos(min(phi, theta - phi, pi))) away: around the centre on the
    nearer side when that side's angle is below pi, through the centre (r1 + r2) otherwise. The continued trace
    ends at phi = theta / 2, so it is shortest exactly when theta >= 2 pi; rays passing the centre at a small
    lateral offset on either side end at phi = pi and phi = theta - pi, whose bisector the continuation is.
    """
    r1, r2 = radii
    rows = []
    for fraction in angles:
        theta = fraction * math.pi
        mesh = G.fan_mesh(n, theta)
        issues = [code for code, _ in G.inspect(mesh.vertices, mesh.faces)]
        polar0 = start * theta / n
        face, point = fan_point(mesh, polar0, r1)
        inward = _unit(mesh.vertices[0] - point)
        tr = G.trace(mesh, face, point, inward, r1 + r2, vertex_rule=PS)
        back = G.trace(mesh, tr.end_face, tr.end_point, -tr.end_direction, r1 + r2, vertex_rule=PS)
        polar, radius = fan_polar(mesh, tr.end_face, tr.end_point)
        side = np.cross(mesh.face_normals[face], inward)
        one_sided = []
        for sign in (1.0, -1.0):
            aim = _unit(mesh.vertices[0] + sign * offset * r1 * side - point)
            passing = G.trace(mesh, face, point, aim, r1 + r2, vertex_rule=PS)
            one_sided.append((fan_polar(mesh, passing.end_face, passing.end_point)[0] - polar0) % theta)
        limits = min(max(abs(one_sided[0] - math.pi), abs(one_sided[1] - (theta - math.pi))),
                     max(abs(one_sided[1] - math.pi), abs(one_sided[0] - (theta - math.pi))))
        grid = [fan_point(mesh, polar0 + phi * math.pi, r2) for phi in ends]
        refined, ids = E.insert_points(mesh, [(face, point), (tr.end_face, tr.end_point)] + grid)
        propagation = E.ExactGeodesic(refined).propagate(ids[0])
        continued = (polar - polar0) % theta
        closed = [math.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * math.cos(min(phi, theta - phi, math.pi)))
                  for phi in [continued] + [e * math.pi for e in ends]]
        exact = [float(propagation.distances[v]) for v in ids[1:]]
        through = [min(e * math.pi, theta - e * math.pi) >= math.pi for e in ends]
        measured = [abs(d - (r1 + r2)) <= SHORTEST for d in exact[1:]]
        path = propagation.path(ids[1])
        rows.append({"total_angle_over_pi": fraction, "issues": issues, "status": tr.status,
                     "vertices_passed": len(tr.vertices), "polar_minus_half": continued - 0.5 * theta,
                     "radius_error": radius - r2, "reverse_error": float(np.max(np.abs(back.end_point - point))),
                     "one_sided_polar": one_sided, "one_sided_limit_error": limits,
                     "traced_minus_exact": tr.length - exact[0], "exact_minus_closed_form": [
                         d - c for d, c in zip(exact, closed)],
                     "through_expected": through, "through_measured": measured,
                     "path_through_centre": 0 in path["vertices"],
                     "refined": {"vertices": refined.vertices, "faces": refined.faces, "start": ids[0],
                                 "ends": ids[1:], "closed_form": closed}})
    return {"triangles": n, "radii": list(radii), "start_fraction": start, "ends_over_pi": list(ends),
            "offset": offset, "rows": rows}


def vertex_hit_study(levels=HIT_LEVELS, amplitude=HIT_JITTER, traces=HIT_TRACES, length=HIT_LENGTH, taus=HIT_TAUS,
                     seed=SEED + 200):
    """How often generic straightest geodesics on irregular meshes pass within a tolerance of a vertex.

    Jittered icospheres are refined at fixed jitter amplitude (in units of h); seeded traces start at uniform
    points of uniformly chosen faces with uniform headings and run ``length`` with the Polthier-Schmies rule.
    For every edge crossing the edge-parameter distance to the nearer endpoint is recorded; a crossing within
    ``VERTEX_TOLERANCE`` is a vertex hit that the rule continues.
    """
    rows = []
    for level in levels:
        jittered = _jitter_unvalidated(G.icosphere(level), amplitude, seed + level)
        mesh = G.TriMesh.build(jittered.vertices, jittered.faces, jittered.name, require_closed=True, center=ORIGIN)
        rng = np.random.Generator(np.random.PCG64(seed + 100 + level))
        faces = rng.integers(len(mesh.faces), size=traces)
        uniform = rng.random((traces, 3))
        margins, passes, statuses = [], 0, set()
        for face, (u, v, heading) in zip(faces, uniform):
            root = math.sqrt(u)
            point = mesh.point(face, [1 - root, root * (1 - v), root * v])
            corners = mesh.vertices[mesh.faces[face]]
            e1 = _unit(corners[1] - corners[0])
            direction = math.cos(2 * math.pi * heading) * e1 + math.sin(2 * math.pi * heading) * np.cross(
                mesh.face_normals[face], e1)
            tr = G.trace(mesh, int(face), point, direction, length, vertex_rule=PS)
            statuses.add(tr.status)
            passes += len(tr.vertices)
            margins += crossing_margins(mesh, tr)
        margins = np.array(margins)
        near = [int(np.sum(margins < tau)) for tau in taus]
        rows.append({"level": level, "h": mesh.mean_edge(), "vertices": len(mesh.vertices), "traces": traces,
                     "crossings": len(margins), "crossings_per_trace": len(margins) / traces, "near": near,
                     "near_over_2tau": [c / (2 * tau * len(margins)) for c, tau in zip(near, taus)],
                     "vertex_hits": passes, "min_margin": float(np.min(margins)), "statuses": sorted(statuses)})
    crossings = sum(r["crossings"] for r in rows)
    pooled = [sum(r["near"][k] for r in rows) / (2 * tau * crossings) for k, tau in enumerate(taus)]
    return {"amplitude": amplitude, "length": length, "taus": list(taus), "tolerance": G.VERTEX_TOLERANCE,
            "rows": rows, "crossings": crossings, "pooled_near_over_2tau": pooled,
            "vertex_hits": sum(r["vertex_hits"] for r in rows),
            "crossing_order": fitted_order([r["h"] for r in rows], [r["crossings_per_trace"] for r in rows])}


def crossing_margins(mesh, tr) -> list:
    """Edge-parameter distance to the nearer endpoint at every edge crossing of a trace (vertex passes excluded)."""
    margins = []
    for i, (f, g) in enumerate(zip(tr.faces, tr.faces[1:])):
        shared = sorted(set(mesh.faces[f].tolist()) & set(mesh.faces[g].tolist()))
        if len(shared) != 2:
            continue
        a, b = mesh.vertices[shared[0]], mesh.vertices[shared[1]]
        t = float(np.linalg.norm(np.asarray(tr.points[i + 1]) - a) / np.linalg.norm(b - a))
        margins.append(min(t, 1.0 - t))
    return margins


def _jitter_torus(n_phi, n_theta, amplitude, seed, major=2.0, minor=1.0) -> G.TriMesh:
    """Torus grid with every vertex moved along the smooth torus by seeded Gaussian noise of amplitude * step in its
    two angles."""
    rng = np.random.Generator(np.random.PCG64(seed))
    base = G.torus_mesh(n_phi, n_theta, major, minor)
    i, j = np.divmod(np.arange(n_phi * n_theta), n_theta)
    phi = 2 * math.pi * (i + amplitude * rng.standard_normal(len(i))) / n_phi
    theta = 2 * math.pi * (j + amplitude * rng.standard_normal(len(j))) / n_theta
    rho = major + minor * np.cos(theta)
    vertices = np.stack([rho * np.cos(phi), rho * np.sin(phi), minor * np.sin(theta)], axis=1)
    return G.TriMesh.build(vertices, base.faces, f"{base.name}-jitter{amplitude:g}-s{seed}", require_closed=True,
                           params={"amplitude": amplitude, "seed": seed})


def path_meshes(seed=SEED + 300) -> list:
    """Meshes of the back-tracing study: a convex and a saddle-bearing irregular closed mesh, a reflex boundary
    corner and a saddle fan with points inserted (so that paths through its centre run between vertices)."""
    sphere = _jitter_unvalidated(G.icosphere(2), 0.1, seed)
    fan = G.fan_mesh(FAN_TRIANGLES, 2.5 * math.pi)
    points = [fan_point(fan, (0.1 + 0.37 * k) * fan.params["total_angle"] / 3.0, 0.3 + 0.1 * (k % 3))
              for k in range(7)]
    fan_refined, _ = E.insert_points(fan, points)
    return [G.TriMesh.build(sphere.vertices, sphere.faces, sphere.name, require_closed=True, center=ORIGIN),
            _jitter_torus(24, 12, 0.1, seed + 1), _l_shape(8),
            G.TriMesh(fan_refined.vertices, fan_refined.faces, fan.name + "-inserted", fan.params)]


def vertex_classes(mesh) -> np.ndarray:
    """Per vertex: 'boundary', 'saddle', 'flat' or 'cone' by its angle sum against 2 pi (BEND tolerance)."""
    excess = -G.angle_defect(mesh)[0]  # angle sum minus 2 pi
    kind = np.where(excess > BEND, "saddle", np.where(excess < -BEND, "cone", "flat")).astype(object)
    kind[mesh.boundary_vertices] = "boundary"
    return kind


def _fan_positions(mesh, vertex):
    """Faces around a vertex in counterclockwise order: {face: (corner, angle position of its first edge)}, total."""
    incident = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
    face = int(incident[0])
    if mesh.boundary_vertices[vertex]:
        # Start at the face whose clockwise neighbour (across the edge v -> first) is missing.
        for f in incident:
            k = int(np.flatnonzero(mesh.faces[f] == vertex)[0])
            if mesh.neighbors[f, k] < 0:
                face = int(f)
    positions, total, f = {}, 0.0, face
    while f >= 0 and f not in positions:
        k = int(np.flatnonzero(mesh.faces[f] == vertex)[0])
        ids = mesh.faces[f]
        a = mesh.vertices[ids[(k + 1) % 3]] - mesh.vertices[vertex]
        b = mesh.vertices[ids[(k + 2) % 3]] - mesh.vertices[vertex]
        positions[f] = (k, total)
        total += math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))
        f = int(mesh.neighbors[f, (k + 2) % 3])
    return positions, total


def side_angles(mesh, vertex, before, face_before, after, face_after) -> tuple:
    """Angles between a path's two segments at a vertex, measured around it on each side (one side at the boundary).

    ``before`` and ``after`` are the neighbouring path points, in faces ``face_before`` and ``face_after``.
    """
    positions, total = _fan_positions(mesh, vertex)
    center = mesh.vertices[vertex]

    def position(face, point):
        k, start = positions[face]
        first = mesh.vertices[mesh.faces[face, (k + 1) % 3]] - center
        offset = np.asarray(point) - center
        return start + math.atan2(float(np.dot(np.cross(first, offset), mesh.face_normals[face])),
                                  float(np.dot(first, offset)))

    a, b = position(face_before, before), position(face_after, after)
    if mesh.boundary_vertices[vertex]:
        return (abs(b - a),)
    turn = (b - a) % total
    return turn, total - turn


def edge_turn(mesh, before, point, after, face_before, face_after) -> float:
    """|a1 + a2 - pi| at a path point on the edge shared by two faces: zero when the unfolded path is straight."""
    shared = sorted(set(mesh.faces[face_before].tolist()) & set(mesh.faces[face_after].tolist()))
    axis = _unit(mesh.vertices[shared[1]] - mesh.vertices[shared[0]])
    return abs(_angle(np.asarray(before) - point, axis) + _angle(axis, np.asarray(after) - point) - math.pi)


def point_to_polyline(points, polyline) -> float:
    """Largest distance from the points to the polyline (point-to-segment distances)."""
    return float(np.max(_nearest_on_polyline(points, polyline)[0]))


def path_checks(mesh, path, classes) -> dict:
    """Checks of one back-traced path that use no part of the solver: surface membership, straightness, bends."""
    points, faces, at_vertex = path["points"], path["faces"], path["vertices"]
    member = face_membership(mesh, points)
    off_surface = sum(not (member[k, f] and member[k + 1, f]) for k, f in enumerate(faces))
    turns, sides, bends = [], [], {"saddle": 0, "boundary": 0, "flat": 0, "cone": 0}
    for k in range(1, len(points) - 1):
        if min(np.linalg.norm(points[k - 1] - points[k]), np.linalg.norm(points[k + 1] - points[k])) <= 1e-14:
            continue  # a repeated point (a crossing at the end of a window's own edge)
        if at_vertex[k] < 0:
            turns.append(edge_turn(mesh, points[k - 1], points[k], points[k + 1], faces[k - 1], faces[k]))
            continue
        angles = side_angles(mesh, at_vertex[k], points[k - 1], faces[k - 1], points[k + 1], faces[k])
        sides.append(min(angles) - math.pi)
        if max(abs(a - math.pi) for a in angles) > BEND:
            bends[classes[at_vertex[k]]] += 1
    return {"off_surface": int(off_surface), "max_turn": max(turns, default=0.0),
            "min_side_minus_pi": min(sides, default=0.0), "bends": bends,
            "vertex_points": int(sum(v >= 0 for v in at_vertex[1:-1]))}


def path_study(meshes=None, sources=PATH_SOURCES, targets=PATH_TARGETS, seed=SEED + 310):
    """Shortest paths back-traced from the exact solver's windows, checked without the solver's own geometry.

    Per seeded pair: the polyline's length against the exact distance, every segment inside the face it is
    assigned to (a point-in-face test), straight across every edge it crosses (equal angles on both sides), and
    at every vertex it passes an angle of at least pi on each side (on the one side at a boundary vertex), so that
    it bends only at saddle and reflex boundary vertices; and the path back-traced from the other end, reversed.
    """
    meshes = path_meshes() if meshes is None else meshes
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for mesh in meshes:
        solver = E.ExactGeodesic(mesh)
        classes = vertex_classes(mesh)
        n = len(mesh.vertices)
        pairs, worst = [], {"length_error": 0.0, "outside_interval": 0.0, "off_surface": 0, "max_turn": 0.0,
                            "min_side_minus_pi": math.inf, "reverse_distance": 0.0}
        bends = {"saddle": 0, "boundary": 0, "flat": 0, "cone": 0}
        vertex_points = 0
        for source in rng.choice(n, size=sources, replace=False):
            forward = solver.propagate(int(source))
            chosen = rng.choice(np.delete(np.arange(n), source), size=targets, replace=False)
            for target in chosen:
                path = forward.path(int(target))
                checks = path_checks(mesh, path, classes)
                backward = solver.propagate(int(target)).path(int(source))
                reverse = max(point_to_polyline(path["points"], backward["points"]),
                              point_to_polyline(backward["points"], path["points"]))
                worst["length_error"] = max(worst["length_error"], abs(path["length"] - path["distance"]))
                worst["outside_interval"] = max(worst["outside_interval"], path["outside_interval"])
                worst["off_surface"] += checks["off_surface"]
                worst["max_turn"] = max(worst["max_turn"], checks["max_turn"])
                worst["min_side_minus_pi"] = min(worst["min_side_minus_pi"], checks["min_side_minus_pi"])
                worst["reverse_distance"] = max(worst["reverse_distance"], reverse)
                vertex_points += checks["vertex_points"]
                for kind, count in checks["bends"].items():
                    bends[kind] += count
                pairs.append({"source": int(source), "target": int(target), "distance": path["distance"],
                              "points": path["points"]})
        rows.append({"mesh": mesh.name, "vertices": n, "pairs": len(pairs), "vertex_points": vertex_points,
                     "bends": bends, "saddle_vertices": int(np.sum(classes == "saddle")), **worst,
                     "mesh_arrays": {"vertices": mesh.vertices, "faces": mesh.faces}, "paths": pairs})
    return {"sources": sources, "targets": targets, "rows": rows}


def _polyline_distance(a, b) -> float:
    return max(point_to_polyline(a, b), point_to_polyline(b, a))


def continuation_independent_study(fans, paths, traced):
    """pygeodesic (Kirsanov's exact MMP) on the fans, the back-traced paths and the cut points, when it imports.

    Fans: exact distances from the start to the continued endpoint and the grid points, against the closed form.
    Paths: pygeodesic's own path between the same vertices (``geodesicDistance`` returns it, target first),
    against the back-traced polyline, and its distance. Cut points: exact distances from the start to the two
    probe points on either side of each located cut point, inserted as vertices, against the recorded windows.
    """
    result = {"pygeodesic": package_version("pygeodesic")}
    if not result["pygeodesic"]:
        return result
    from pygeodesic.geodesic import PyGeodesicAlgorithmExact
    fan_gap = 0.0
    for row in fans["rows"]:
        refined = row["refined"]
        algorithm = PyGeodesicAlgorithmExact(refined["vertices"], refined["faces"].astype(np.int32))
        reference, _ = algorithm.geodesicDistances(np.array([refined["start"]], dtype=np.int32), None)
        fan_gap = max(fan_gap, max(abs(float(reference[v]) - c) for v, c in zip(refined["ends"],
                                                                                refined["closed_form"])))
    path_distance, path_length = 0.0, 0.0
    for row in paths["rows"]:
        arrays = row["mesh_arrays"]
        algorithm = PyGeodesicAlgorithmExact(arrays["vertices"], arrays["faces"].astype(np.int32))
        for pair in row["paths"]:
            distance, polyline = algorithm.geodesicDistance(pair["source"], pair["target"])
            path_distance = max(path_distance, _polyline_distance(pair["points"], np.asarray(polyline)))
            path_length = max(path_length, abs(float(distance) - pair["distance"]))
    cut_gap, probes = 0.0, 0
    for row in traced["rows"]:
        cut = row.get("cut")
        if not cut or cut["probes"] is None:
            continue
        mesh = G.icosphere(row["level"])
        u0, heading = STARTS[row["start"]]
        start = sphere_start(mesh, u0, heading)
        tr = G.trace(mesh, start["face"], start["point"], start["direction"], row["length"])
        inserted = [(start["face"], start["point"])] + [(tr.faces[j], p) for j, p in zip(cut["probes"]["segments"],
                                                                                          cut["probes"]["points"])]
        refined, ids = E.insert_points(mesh, inserted)
        algorithm = PyGeodesicAlgorithmExact(refined.vertices, refined.faces.astype(np.int32))
        reference, _ = algorithm.geodesicDistances(np.array([ids[0]], dtype=np.int32), None)
        for v, d in zip(ids[1:], cut["probes"]["distances"]):
            cut_gap = max(cut_gap, abs(float(reference[v]) - d))
            probes += 1
    result.update(fan_max_abs=fan_gap, path_max_distance=path_distance, path_length_max_abs=path_length,
                  cut_probe_max_abs=cut_gap, cut_probes=probes)
    return result


# ---------------------------------------------------------------- Jacobi fields and curvature
def smooth_jacobi(starts=STARTS, length=TRACE_LENGTH, steps=200):
    values = []
    for u0, heading in starts:
        tr = jacobi.transfer(UNIT_SPHERE, np.array(u0, dtype=float), heading, length, steps=steps)
        values.append(float(tr.matrix()[0, 1]))
    return {"j_head": values, "closed_form": math.sin(length),
            "max_error": float(max(abs(v - math.sin(length)) for v in values))}


def jacobi_study(levels=TRACE_LEVELS[1:], deltas=(0.1, 0.03, 0.01, 1e-5), starts=STARTS, length=TRACE_LENGTH):
    """Central finite-difference heading Jacobi field |X+ - X-| / (2 sin delta) from paired traces.

    On the smooth unit sphere this estimator equals sin(L) exactly for every delta.
    ``swept_vertices`` = 2 delta (1 - cos L) V / (4 pi) is the expected number of
    vertices inside the thin wedge between a pair (a heuristic for uniformly spread
    vertices); a pair shares its face sequence only when no vertex lies in the wedge.
    """
    reference = smooth_jacobi(starts, length)
    rows = []
    for level in levels:
        mesh = G.icosphere(level)
        for delta in deltas:
            errors, identical, flat, statuses = [], 0, [], []
            for (u0, heading), smooth in zip(starts, reference["j_head"]):
                s = sphere_start(mesh, u0, heading)
                plus, minus = (G.trace(mesh, s["face"], s["point"],
                                       mesh.rotate_in_face(s["face"], s["direction"], angle), length)
                               for angle in (delta, -delta))
                statuses += [plus.status, minus.status]
                if not (plus.completed and minus.completed):
                    continue
                j = float(np.linalg.norm(plus.end_point - minus.end_point) / (2 * math.sin(delta)))
                errors.append(j - smooth)
                if plus.faces == minus.faces:
                    identical += 1
                    flat.append(abs(j - length))
            rows.append({"level": level, "h": mesh.mean_edge(), "delta": delta, "pairs": len(errors),
                         "swept_vertices": 2 * delta * (1 - math.cos(length)) * len(mesh.vertices) / (4 * math.pi),
                         "refused": sum(s != "completed" for s in statuses),
                         "mean_abs_error": float(np.mean(np.abs(errors))),
                         "max_abs_error": float(np.max(np.abs(errors))),
                         "identical_face_sequences": identical,
                         "max_flat_deviation": float(max(flat)) if flat else None})
    return {"length": length, "smooth": reference, "flat_value": length, "rows": rows}


def icosahedral_mirror_classes(vertices, tolerance=1e-12):
    """Masks of unit-sphere vertices on the projected base-icosahedron edges and on the rest of its mirror planes.

    The 15 mirror planes each contain two opposite base edges; inside a base face
    they run along its medians. Midpoint subdivision keeps vertices exactly on
    these great circles (up to rounding), so a small absolute tolerance suffices.
    """
    base, faces = G.icosahedron()
    arcs = sorted({tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in faces for k in range(3)})
    on_edge = np.zeros(len(vertices), dtype=bool)
    on_mirror = np.zeros(len(vertices), dtype=bool)
    for a, b in arcs:
        pa, pb = base[a], base[b]
        normal = np.cross(pa, pb)
        normal /= np.linalg.norm(normal)
        on = np.abs(vertices @ normal) <= tolerance
        span = np.arccos(np.clip(vertices @ pa, -1, 1)) + np.arccos(np.clip(vertices @ pb, -1, 1))
        on_mirror |= on
        on_edge |= on & (np.abs(span - math.acos(float(pa @ pb))) <= 1e-9)
    return on_edge, on_mirror & ~on_edge


def curvature_study(levels=TRACE_LEVELS, torus_sizes=(8, 16, 32, 64)):
    sphere_rows = []
    for level in levels:
        mesh = G.icosphere(level)
        defect, area, _ = G.angle_defect(mesh)
        k = defect / area
        voronoi = defect / G.mixed_voronoi_area(mesh)
        valence = np.bincount(mesh.faces.ravel())
        five, six = valence == 5, valence == 6
        edge_arc, median = icosahedral_mirror_classes(mesh.vertices)
        off = six & ~edge_arc & ~median
        error = np.abs(k - 1)

        def worst(mask, values=error):
            return float(np.max(values[mask])) if mask.any() else None
        sphere_rows.append({
            "level": level, "h": mesh.mean_edge(), "vertices": len(mesh.vertices),
            "gauss_bonnet_residual": float(defect.sum() - 4 * math.pi),
            "valence5_value": float(np.mean(k[five])), "valence5_spread": float(np.ptp(k[five])),
            "valence6_max_error": worst(six),
            "valence6_base_edge_max_error": worst(six & edge_arc), "valence6_median_max_error": worst(six & median),
            "valence6_off_mirror_max_error": worst(off),
            "voronoi_valence6_mirror_max_error": worst(six & (edge_arc | median), np.abs(voronoi - 1)),
            "rms_error": float(np.sqrt(np.mean((k - 1) ** 2))), "max_error": float(np.max(error)),
            "voronoi_valence5_error": float(np.max(np.abs(voronoi[five] - 1))),
            "voronoi_max_error": float(np.max(np.abs(voronoi - 1)))})
    torus_rows = []
    torus = Torus(2.0, 1.0)
    for size in torus_sizes:
        mesh = G.torus_mesh(2 * size, size, torus.major, torus.minor)
        defect, area, _ = G.angle_defect(mesh)
        theta = 2 * math.pi * (np.arange(len(mesh.vertices)) % size) / size
        smooth = np.array([torus.gaussian_curvature((0.0, t)) for t in theta])
        k = defect / area
        torus_rows.append({"n_theta": size, "n_phi": 2 * size, "h": mesh.mean_edge(),
                           "gauss_bonnet_residual": float(defect.sum()),
                           "max_error": float(np.max(np.abs(k - smooth))),
                           "rms_error": float(np.sqrt(np.mean((k - smooth) ** 2)))})
    return {"valence5_limit": VALENCE5_LIMIT, "sphere": sphere_rows, "torus": torus_rows}


# ---------------------------------------------------------------- mesh quality
def inverted_faces(mesh, center=ORIGIN) -> np.ndarray:
    """Faces whose unit normal points towards the centre at their centroid (the fold indicator of T041)."""
    centroid = mesh.vertices[mesh.faces].mean(axis=1) - center
    return np.flatnonzero(np.einsum("ij,ij->i", mesh.face_normals, centroid) < 0)


def mesh_errors(mesh, starts=STARTS, length=TRACE_LENGTH):
    """Quality metrics and errors of a unit-sphere mesh, measured whether or not it validates.

    ``issues`` come from the validator with the sphere centre declared (so an
    inverted face is refused); ``structural_issues`` from the structural and
    dihedral checks alone.
    """
    structural = [code for code, _ in G.inspect(mesh.vertices, mesh.faces, require_closed=True)]
    issues = [code for code, _ in G.inspect(mesh.vertices, mesh.faces, require_closed=True, center=ORIGIN)]
    with np.errstate(divide="ignore", invalid="ignore"):
        defect, area, _ = G.angle_defect(mesh)
        k = defect / area
        voronoi = defect / G.mixed_voronoi_area(mesh)
    quality = mesh.quality()
    geodesic, statuses = [], []
    for u0, heading in starts:
        s = sphere_start(mesh, u0, heading)
        tr = G.trace(mesh, s["face"], s["point"], s["direction"], length)
        statuses.append(tr.status)
        if tr.completed:
            geodesic.append(great_circle_errors(s, tr.end_point, length)["endpoint"])
    inverted = len(inverted_faces(mesh))
    return {"mesh": mesh.name, "vertices": len(mesh.vertices), "issues": issues, "structural_issues": structural,
            "inverted_faces": inverted,
            "min_angle_deg": quality["min_angle_deg"], "max_radius_ratio": quality["max_radius_ratio"],
            "curvature_rms": float(np.sqrt(np.mean((k - 1) ** 2))), "curvature_max": float(np.max(np.abs(k - 1))),
            "voronoi_curvature_rms": float(np.sqrt(np.mean((voronoi - 1) ** 2))),
            "area_error": float(mesh.area() / (4 * math.pi) - 1),
            "geodesic_mean": float(np.mean(geodesic)) if geodesic else None,
            "geodesic_max": float(np.max(geodesic)) if geodesic else None, "trace_statuses": statuses}


def quality_study(level=3, amplitudes=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3), seeds=(1, 2, 3),
                  uv=((20, 32, 0.0), (10, 64, 0.0), (40, 16, 0.0), (20, 32, 0.08), (20, 32, 0.16))):
    base = G.icosphere(level)
    jitter = []
    for amplitude in amplitudes:
        for seed in (seeds if amplitude > 0 else (0,)):
            mesh = base if amplitude == 0 else _jitter_unvalidated(base, amplitude, SEED + seed)
            row = mesh_errors(mesh)
            row.update({"amplitude": amplitude, "seed": seed})
            jitter.append(row)
    family = []
    for n_lat, n_lon, twist in uv:
        row = mesh_errors(G.uv_sphere(n_lat, n_lon, twist=twist))
        row.update({"n_lat": n_lat, "n_lon": n_lon, "twist": twist})
        family.append(row)
    return {"base": base.name, "vertex_count": len(base.vertices), "jitter": jitter, "latitude_longitude": family}


def _jitter_unvalidated(mesh, amplitude, seed):
    """Jittered copy built without validation so that folded meshes can be measured and then refused."""
    rng = np.random.Generator(np.random.PCG64(seed))
    x = mesh.vertices
    noise = rng.standard_normal(x.shape) * amplitude * mesh.mean_edge()
    noise -= np.einsum("ij,ij->i", noise, x)[:, None] * x
    moved = x + noise
    moved /= np.linalg.norm(moved, axis=1)[:, None]
    return G.TriMesh(moved, mesh.faces, f"{mesh.name}-jitter{amplitude:g}-s{seed}",
                     {"amplitude": amplitude, "seed": seed})


def fold_rate_study(level=3, amplitudes=(0.1, 0.15, 0.2, 0.3), seeds=FOLD_SEEDS):
    """Dihedral fold check against inverted faces over many seeds of the tangential jitter generator.

    Categories per amplitude: inverted (a face normal points into the sphere) or
    not, refused by the structural and dihedral checks (no centre declared) or
    not; plus refusals with the sphere centre declared. The witness is the
    clearest miss: the accepted inverted mesh with the largest
    min(-normal . radial, smallest adjacent normal dot - FOLD_COSINE), i.e.
    both clearly inverted and clearly below the fold threshold.
    """
    base = G.icosphere(level)
    rows, witness = [], None
    for amplitude in amplitudes:
        row = {"amplitude": amplitude, "meshes": len(seeds), "inverted_refused": 0, "inverted_accepted": 0,
               "clean_refused": 0, "clean_accepted": 0, "refused_with_centre": 0, "accepted_inverted_seeds": []}
        for offset in seeds:
            mesh = _jitter_unvalidated(base, amplitude, SEED + offset)
            inverted = inverted_faces(mesh)
            structural = [c for c, _ in G.inspect(mesh.vertices, mesh.faces, require_closed=True)]
            declared = [c for c, _ in G.inspect(mesh.vertices, mesh.faces, require_closed=True, center=ORIGIN)]
            key = ("inverted" if len(inverted) else "clean") + ("_refused" if structural else "_accepted")
            row[key] += 1
            row["refused_with_centre"] += bool(declared)
            if len(inverted) and not structural:
                row["accepted_inverted_seeds"].append(SEED + offset)
                candidate = _inverted_witness(mesh, inverted, amplitude, SEED + offset, declared)
                if witness is None or _miss_margin(candidate) > _miss_margin(witness):
                    witness = candidate
        rows.append(row)
    return {"level": level, "seeds": [SEED + s for s in seeds], "fold_cosine": G.FOLD_COSINE, "rows": rows,
            "witness": witness}


def _miss_margin(witness) -> float:
    return min(-witness["normal_radial"], witness["min_adjacent_normal_dot"] - G.FOLD_COSINE)


def _inverted_witness(mesh, inverted, amplitude, seed, declared_issues):
    centroid = mesh.vertices[mesh.faces].mean(axis=1)
    radial = np.einsum("ij,ij->i", mesh.face_normals, centroid / np.linalg.norm(centroid, axis=1)[:, None])
    face = int(inverted[np.argmin(radial[inverted])])
    with np.errstate(divide="ignore", invalid="ignore"):
        defect, area, _ = G.angle_defect(mesh)
    table = G._edge_table(mesh.faces, len(mesh.vertices))
    dots = np.einsum("ij,ij->i", mesh.face_normals[table["pair_first"] // 3],
                     mesh.face_normals[table["pair_second"] // 3])
    return {"amplitude": amplitude, "seed": seed, "inverted_faces": len(inverted), "face": face,
            "normal_radial": float(radial[face]), "corner_angles_deg": np.degrees(mesh.corner_angles()[face]).tolist(),
            "neighbour_normal_dots": [float(mesh.face_normals[face] @ mesh.face_normals[g])
                                      for g in mesh.neighbors[face]],
            "min_adjacent_normal_dot": float(np.min(dots)),
            "unguarded_curvature_rms": float(np.sqrt(np.mean((defect / area - 1) ** 2))),
            "issues_with_centre": declared_issues}


def lantern_study(q=0.25, ns=(4, 8, 16, 32, 64), radius=1.0, height=1.0, folded_q=1.0, folded_n=8, samples=4):
    """Schwarz lantern with m = q n^2 bands: Hausdorff distance -> 0 while area and height do not converge."""
    rows = []
    for n in ns:
        m = max(1, int(round(q * n * n)))
        mesh = G.cylinder_mesh(n, m, radius, height, lantern=True)
        sag = radius * (1 - math.cos(math.pi / n))
        closed_area = 2 * n * radius * math.sin(math.pi / n) * math.sqrt(height ** 2 + (m * sag) ** 2)
        closed_height = math.sqrt(height ** 2 + (m * sag) ** 2)
        # Barycentric sample grid (includes edge midpoints) for the one-sided distance to the cylinder.
        i, j = np.meshgrid(np.arange(samples + 1), np.arange(samples + 1), indexing="ij")
        keep = (i + j) <= samples
        bary = np.stack([i[keep], j[keep], samples - i[keep] - j[keep]], 1) / samples
        points = np.einsum("sk,fkd->fsd", bary, mesh.vertices[mesh.faces]).reshape(-1, 3)
        gap = float(np.max(radius - np.hypot(points[:, 0], points[:, 1])))
        start = 0.7 * mesh.vertices[0] + 0.3 * mesh.vertices[1]
        face, _ = mesh.locate(start)
        tr = G.trace(mesh, face, start, mesh.tangent(face, np.array([0.0, 0.0, 1.0])), 100.0 * height)
        k, interior = G.angle_defect_curvature(mesh)
        bend = _dihedral(mesh)
        centroid = mesh.vertices[mesh.faces].mean(axis=1)
        radial = centroid * np.array([1.0, 1.0, 0.0])
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        tilt = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", mesh.face_normals, radial), -1, 1)))
        rows.append({"n": n, "bands": m, "faces": len(mesh.faces),
                     "area_ratio": mesh.area() / (2 * math.pi * radius * height),
                     "area_closed_form_error": float(mesh.area() - closed_area), "hausdorff_sampled": gap,
                     "hausdorff_closed_form": sag, "trace_status": tr.status, "traced_height": tr.length,
                     "height_closed_form": closed_height, "max_interior_curvature": float(np.max(np.abs(k[interior]))),
                     "total_abs_mean_curvature": bend, "max_normal_tilt_deg": float(np.max(tilt)),
                     "issues": [c for c, _ in G.inspect(mesh.vertices, mesh.faces)]})
    folded = G.cylinder_mesh(folded_n, int(round(folded_q * folded_n ** 2)), radius, height, lantern=True)
    limit = math.sqrt(1 + (math.pi ** 2 * radius * q / (2 * height)) ** 2)
    tilt = math.degrees(math.atan(math.pi ** 2 * radius * q / (2 * height)))
    folded_issues = [c for c, _ in G.inspect(folded.vertices, folded.faces)]
    centroid = folded.vertices[folded.faces].mean(axis=1) * np.array([1.0, 1.0, 0.0])
    radial = np.einsum("ij,ij->i", folded.face_normals, centroid / np.linalg.norm(centroid, axis=1)[:, None])
    return {"q": q, "limit_area_ratio": limit, "limit_tilt_deg": tilt,
            "smooth_total_abs_mean_curvature": math.pi * height, "rows": rows,
            "folded": {"q": folded_q, "n": folded_n, "issues": folded_issues,
                       "min_normal_radial": float(np.min(radial))}}


def _dihedral(mesh) -> float:
    """Total absolute mean curvature 1/2 sum |e| |bend angle| over interior edges."""
    table = G._edge_table(mesh.faces, len(mesh.vertices))
    first, second = table["pair_first"], table["pair_second"]
    n1, n2 = mesh.face_normals[first // 3], mesh.face_normals[second // 3]
    bend = np.arctan2(np.linalg.norm(np.cross(n1, n2), axis=1), np.einsum("ij,ij->i", n1, n2))
    length = np.linalg.norm(mesh.vertices[table["head"][first]] - mesh.vertices[table["tail"][first]], axis=1)
    return float(0.5 * np.sum(length * bend))


# ---------------------------------------------------------------- refusal catalogue
def _octahedron():
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return vertices, faces


def _code(call):
    try:
        call()
    except G.MeshRefusal as exc:
        return exc.code
    return None


def refusal_study():
    octa_v, octa_f = _octahedron()
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    cases = []

    def build_case(name, expected, vertices, faces, **kwargs):
        cases.append({"case": name, "stage": "validation", "expected": expected,
                      "observed": _code(lambda: G.TriMesh.build(vertices, faces, **kwargs))})

    nan_v = octa_v.copy()
    nan_v[3, 1] = np.nan
    build_case("nan-vertex", "nonfinite_vertex", nan_v, octa_f)
    build_case("index-out-of-range", "invalid_face_index", octa_v, np.vstack([octa_f[:-1], [[0, 3, 99]]]))
    collinear = np.vstack([square, [[2, 0, 0]]])
    build_case("zero-area-face", "degenerate_face", collinear, np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]]))
    build_case("repeated-index", "degenerate_face", square, np.array([[0, 1, 2], [0, 2, 2]]))
    book = np.vstack([square, [[0.5, 0.5, 1.0]]])
    build_case("three-faces-on-one-edge", "non_manifold_edge", book, np.array([[0, 1, 2], [0, 3, 1], [1, 0, 4]]))
    flipped = octa_f.copy()
    flipped[0] = flipped[0][::-1]
    build_case("flipped-face", "inconsistent_orientation", octa_v, flipped)
    bowtie = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    build_case("bowtie-vertex", "non_manifold_vertex", bowtie, np.array([[0, 1, 2], [0, 3, 4]]))
    fold = square.copy()
    fold[3] = [0.8, 0.2, 0.0]
    build_case("fold-over", "folded_face", fold, np.array([[0, 1, 2], [0, 2, 3]]))
    build_case("inward-oriented-octahedron", "inverted_face", octa_v, octa_f[:, ::-1], center=ORIGIN)
    amplitude, offset = INVERTED_JITTER
    jittered = _jitter_unvalidated(G.icosphere(3), amplitude, SEED + offset)
    build_case("jittered-inverted-face", "inverted_face", jittered.vertices, jittered.faces, require_closed=True,
               center=ORIGIN)
    build_case("unreferenced-vertex", "unreferenced_vertex", np.vstack([octa_v, [[3, 3, 3]]]), octa_f)
    build_case("two-components", "disconnected_components", np.vstack([octa_v, octa_v + 5]),
               np.vstack([octa_f, octa_f + 6]))
    build_case("hole-when-closed-required", "open_boundary", octa_v, octa_f[1:], require_closed=True)
    build_case("empty", "empty_mesh", octa_v, np.zeros((0, 3), dtype=int))
    build_case("planar-coordinates", "invalid_shape", octa_v[:, :2], octa_f)

    plane = G.plane_mesh(4, 4)
    start = np.array([0.43, 0.61, 0.0])
    face, _ = plane.locate(start)
    east = np.array([1.0, 0.0, 0.0])

    def trace_case(name, expected, mesh, face_, point, direction, length, **kwargs):
        tr = G.trace(mesh, face_, point, direction, length, **kwargs)
        cases.append({"case": name, "stage": "tracing", "expected": expected, "observed": tr.status,
                      "partial_length": tr.length})
        return tr

    boundary = trace_case("leave-planar-patch", "boundary_reached", plane, face, start, east, 5.0)
    vertex = plane.vertices[3 * 5 + 3]
    toward = (vertex - start) / np.linalg.norm(vertex - start)
    trace_case("aimed-at-vertex", "vertex_hit", plane, face, start, toward, 5.0)
    trace_case("point-off-face", "point_outside_face", plane, face, start + np.array([0.0, 0.0, 0.1]), east, 1.0)
    trace_case("normal-direction", "invalid_direction", plane, face, start, np.array([0.3, 0.0, 1.0]), 1.0)
    trace_case("step-budget", "step_budget_exceeded", plane, face, start, east, 5.0, max_steps=2)
    holed = G.TriMesh.build(G.icosphere(2).vertices, np.delete(G.icosphere(2).faces, 0, axis=0), "icosphere-2-hole")
    hole_center = G.icosphere(2).vertices[G.icosphere(2).faces[0]].mean(axis=0)
    target = hole_center / np.linalg.norm(hole_center)
    neighbor = int(G.icosphere(2).neighbors[0, 0]) - 1  # index shifts by one after deleting face 0
    hole_start = holed.vertices[holed.faces[neighbor]].mean(axis=0)
    toward_hole = holed.tangent(neighbor, target - hole_start)
    trace_case("trace-into-hole", "boundary_reached", holed, neighbor, hole_start, toward_hole, 1.0)
    split = G.TriMesh.build(np.vstack([octa_v, octa_v + 5]), np.vstack([octa_f, octa_f + 6]), require_connected=False)
    cases.append({"case": "distance-across-components", "stage": "query", "expected": "unreachable_target",
                  "observed": _code(lambda: G.vertex_distance(split, 0, 7))})
    open_plane = G.plane_mesh(3, 3)
    cases.append({"case": "curvature-at-boundary-vertex", "stage": "query", "expected": "boundary_vertex_curvature",
                  "observed": _code(lambda: G.angle_defect_curvature(open_plane, 0))})
    small = G.icosphere(1)
    cases.append({"case": "heat-method-above-size-limit", "stage": "query", "expected": "mesh_too_large",
                  "observed": _code(lambda: G.heat_distance(small, 0, max_vertices=12))})
    # Faces 0 and 2 of the 4 x 4 grid share no edge, so they cannot be unfolded as a strip.
    cases.append({"case": "strip-faces-not-adjacent", "stage": "query", "expected": "invalid_strip",
                  "observed": _code(lambda: G.strip_unfold_distance(
                      plane.vertices[None], [tuple(plane.faces[0]), tuple(plane.faces[2])],
                      np.full(3, 1 / 3), np.full(3, 1 / 3)))})

    controls = []
    for mesh, closed, center in ((G.icosphere(2), True, ORIGIN), (G.uv_sphere(10, 16, twist=0.1), True, ORIGIN),
                                 (G.cylinder_mesh(12, 6), False, None), (G.plane_mesh(5, 5, shear=0.4), False, None),
                                 (G.torus_mesh(16, 8), True, None)):
        controls.append({"mesh": mesh.name, "center_declared": center is not None,
                         "issues": [c for c, _ in G.inspect(mesh.vertices, mesh.faces, require_closed=closed,
                                                            center=center)]})
    several = octa_v.copy()
    several[1, 0] = np.inf
    several_f = octa_f.copy()
    several_f[2] = several_f[2][::-1]
    multiple = [c for c, _ in G.inspect(np.vstack([several, [[9, 9, 9]]]), several_f)]
    # Unguarded evaluation: bypass validation and evaluate curvature on the zero-area mesh.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = G.TriMesh(collinear, np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]]), "unvalidated")
        defect, area, _ = G.angle_defect(raw)
        unguarded = defect / area
        normals_nonfinite = int(np.sum(~np.isfinite(raw.face_normals)))
    undetected = {"amplitude": amplitude, "seed": SEED + offset, "inverted_faces": len(inverted_faces(jittered)),
                  "structural_issues": [c for c, _ in G.inspect(jittered.vertices, jittered.faces,
                                                                require_closed=True)]}
    return {"cases": cases, "controls": controls, "multiple_defects": multiple, "undetected_inversion": undetected,
            "boundary_partial_length": boundary.length, "boundary_analytic_length": 1.0 - start[0],
            "unguarded_nonfinite_curvatures": int(np.sum(~np.isfinite(unguarded))),
            "unguarded_nonfinite_normals": normals_nonfinite}


# ---------------------------------------------------------------- uncertainty propagation
def _fd_jacobian(function, base, tau=1e-6):
    """Central differences of a batched function over every coordinate of base (nv, 3)."""
    size = base.size
    steps = np.zeros((2 * size,) + base.shape)
    flat = steps.reshape(2 * size, -1)
    flat[np.arange(size) * 2, np.arange(size)] = tau
    flat[np.arange(size) * 2 + 1, np.arange(size)] = -tau
    values = np.asarray(function(base[None] + steps))
    return (values[0::2] - values[1::2]) / (2 * tau)


def marker_strip(mesh, index=MARKER_START, length=1.0, starts=STARTS):
    """Markers A (start) and B (end) of a declared traced geodesic, attached barycentrically to their faces."""
    u0, heading = starts[index]
    s = sphere_start(mesh, u0, heading)
    tr = G.trace(mesh, s["face"], s["point"], s["direction"], length)
    if not tr.completed:
        raise G.MeshRefusal(tr.status, f"The declared marker geodesic {index} did not complete: {tr.message}")
    ids = np.array(sorted(set(mesh.faces[tr.faces].ravel().tolist())))
    local = {int(v): i for i, v in enumerate(ids)}
    strip = [tuple(local[int(v)] for v in mesh.faces[f]) for f in tr.faces]
    return {"start_index": index, "ids": ids, "strip": strip, "length": tr.length,
            "margin": float(tr.min_vertex_margin), "faces": len(tr.faces),
            "a": mesh.barycentric(tr.faces[0], s["point"]), "b": mesh.barycentric(tr.end_face, tr.end_point)}


def strip_distance(strip):
    """Batched fixed-corridor marker distance and corridor margin for a marker strip."""
    def distance(positions):
        return G.strip_unfold_distance(positions, strip["strip"], strip["a"], strip["b"])[0]

    def margin(positions):
        return G.strip_unfold_distance(positions, strip["strip"], strip["a"], strip["b"])[1]
    return distance, margin


def gradient_split(strip, base, gradient):
    """|grad d| split by direction (normal / tangential to the unit sphere) and by vertex group.

    Marker vertices are those of the first and last strip faces (the barycentric
    markers move with them); the others are interior strip vertices.
    """
    gradient = gradient.reshape(base.shape)
    normal = base / np.linalg.norm(base, axis=1)[:, None]
    along = np.einsum("ij,ij->i", gradient, normal)
    tangential = gradient - along[:, None] * normal
    marker = np.zeros(len(base), dtype=bool)
    marker[sorted(set(strip["strip"][0]) | set(strip["strip"][-1]))] = True

    def norm(values):
        return float(np.sqrt(np.sum(values ** 2)))
    return {"total": norm(gradient), "normal": norm(along), "tangential": norm(tangential),
            "marker": norm(gradient[marker]), "interior": norm(gradient[~marker]),
            "interior_normal": norm(along[~marker]), "marker_vertices": int(marker.sum()),
            "interior_vertices": int((~marker).sum())}


def far_valence6_vertex(mesh) -> int:
    valence = np.bincount(mesh.faces.ravel())
    special = mesh.vertices[valence == 5]
    separation = np.min(np.arccos(np.clip(mesh.vertices @ special.T, -1, 1)), axis=1)
    separation[valence != 6] = -1.0
    # Symmetric vertices tie mathematically; rounding makes the lowest index win on every platform.
    return int(np.argmax(np.round(separation, 9)))


def _normal_observable(ring_faces, reference):
    t1 = np.cross(reference, [0.0, 0.0, 1.0] if abs(reference[2]) < 0.9 else [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(reference, t1)

    def tangential(positions):
        n = G.batch_vertex_normal(positions, ring_faces)
        return np.stack([n @ t1, n @ t2], axis=1)

    def alignment(positions):
        return G.batch_vertex_normal(positions, ring_faces) @ reference
    return tangential, alignment


def propagate(mesh, observable, sigmas, samples, seed):
    """Linearized (finite-difference Jacobian) against seeded Monte Carlo for one observable."""
    kind = observable["kind"]
    base = mesh.vertices[observable["ids"]]
    function = observable["function"]
    jac = _fd_jacobian(function, base)
    value0 = np.asarray(function(base[None]))[0]
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for sigma in sigmas:
        noise = rng.standard_normal((samples,) + base.shape) * sigma
        values = np.asarray(function(base[None] + noise))
        if kind == "vector":
            linear = sigma ** 2 * float(np.sum(jac ** 2))
            empirical = float(np.mean(np.sum((values - value0) ** 2, axis=1)))
            bias = float(np.linalg.norm(values.mean(axis=0) - value0))
        else:
            linear = sigma ** 2 * float(np.sum(jac ** 2))
            empirical = float(np.var(values, ddof=1))
            bias = float(np.mean(values) - value0)
        row = {"sigma": sigma, "linear_variance": linear, "mc_variance": empirical, "ratio": empirical / linear,
               "bias": bias, "linear_sd": math.sqrt(linear)}
        if "validity" in observable:
            row["invalid_fraction"] = float(np.mean(observable["validity"](base[None] + noise) < 0))
        if "alignment" in observable:
            # A tangential-component observable cannot see a normal that flipped sign; count flips directly.
            dots = observable["alignment"](base[None] + noise)
            row["min_normal_dot"] = float(np.min(dots))
            row["flipped_normals"] = int(np.sum(dots < 0))
        rows.append(row)
    return {"observable": observable["name"], "value": float(np.ravel(value0)[0]) if kind != "vector" else 0.0,
            "gradient_norm": float(np.sqrt(np.sum(jac ** 2))), "rows": rows}


def observables(mesh, index=MARKER_START, strip_length=1.0):
    strip = marker_strip(mesh, index, strip_length)
    distance, margin = strip_distance(strip)
    result = [{"name": "marker geodesic distance", "kind": "scalar", "ids": strip["ids"], "function": distance,
               "validity": margin}]
    far = far_valence6_vertex(mesh)
    for label, vertex in (("valence-5 vertex 0", 0), ("valence-6 vertex far from valence 5", far)):
        ids, ring = G.one_ring(mesh, vertex)
        normal0 = G.batch_vertex_normal(mesh.vertices[ids][None], ring)[0]
        function, alignment = _normal_observable(ring, normal0)
        result.append({"name": f"vertex normal at {label}", "kind": "vector", "ids": ids,
                       "function": function, "alignment": alignment, "vertex": vertex})
        result.append({"name": f"angle-defect curvature at {label}", "kind": "scalar", "ids": ids,
                       "function": (lambda r: (lambda p: G.batch_angle_defect_curvature(p, 0, r)))(ring),
                       "vertex": vertex})
    return strip, result


def uncertainty_study(level=3, sigmas=(1e-4, 1e-3, 3e-3, 1e-2), samples=4000, seed=SEED):
    mesh = G.icosphere(level)
    strip, items = observables(mesh)
    rows = [propagate(mesh, item, sigmas, samples, seed + index) for index, item in enumerate(items)]
    return {"level": level, "h": mesh.mean_edge(), "h_squared": mesh.mean_edge() ** 2, "samples": samples,
            "strip": {"start_index": strip["start_index"], "faces": strip["faces"], "length": strip["length"],
                      "vertex_margin": strip["margin"], "vertices": len(strip["ids"])},
            "observables": rows}


def corridor_study(level=3, sigmas=(1e-4, 1e-3, 3e-3, 1e-2), samples=4000, length=1.0, seed=SEED + 30):
    """Fraction of noisy samples whose unfolded marker segment leaves its corridor, for all six declared strips."""
    mesh = G.icosphere(level)
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for index in range(len(STARTS)):
        strip = marker_strip(mesh, index, length)
        _, margin = strip_distance(strip)
        base = mesh.vertices[strip["ids"]]
        fractions = []
        for sigma in sigmas:
            noise = rng.standard_normal((samples,) + base.shape) * sigma
            fractions.append(float(np.mean(margin(base[None] + noise) < 0)))
        rows.append({"start_index": index, "vertex_margin": strip["margin"],
                     "margin_times_h": strip["margin"] * mesh.mean_edge(), "faces": strip["faces"],
                     "left_fraction": fractions})
    return {"level": level, "h": mesh.mean_edge(), "sigmas": list(sigmas), "samples": samples, "rows": rows}


def vertex_field_study(level=3, sigma=1e-4, normal_sigma=3e-4, tangential_sigma=1e-4, samples=2000,
                       seed=SEED + 60, chunk=250):
    """Per-vertex linearized normal and curvature standard deviations against Monte Carlo at every vertex.

    Two declared vertex covariances: isotropic sigma^2 I, and normal-dominant
    sigma_t^2 (I - u u^T) + sigma_n^2 u u^T with u the unit-sphere normal. The
    linearization is ``G.vertex_uncertainty`` (one-ring finite differences); the
    Monte Carlo evaluates the whole mesh with ``G.batch_vertex_fields``.
    """
    mesh = G.icosphere(level)
    n = len(mesh.vertices)
    unit = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1)[:, None]
    outer = np.einsum("ni,nj->nij", unit, unit)
    covariances = {"isotropic": np.broadcast_to(sigma ** 2 * np.eye(3), (n, 3, 3)).copy(),
                   "normal-dominant": tangential_sigma ** 2 * (np.eye(3)[None] - outer) + normal_sigma ** 2 * outer}
    normal0, curvature0 = (a[0] for a in G.batch_vertex_fields(mesh.vertices[None], mesh.faces))
    valence = np.bincount(mesh.faces.ravel())
    se = math.sqrt(2.0 / (samples - 1))
    rng = np.random.Generator(np.random.PCG64(seed))
    models = []
    for name, covariance in covariances.items():
        linear = G.vertex_uncertainty(mesh, covariance)
        root = np.linalg.cholesky(covariance)
        s1, s2, tilt = np.zeros(n), np.zeros(n), np.zeros(n)
        for start in range(0, samples, chunk):
            size = min(chunk, samples - start)
            noise = np.einsum("nij,bnj->bni", root, rng.standard_normal((size, n, 3)))
            normals, curvature = G.batch_vertex_fields(mesh.vertices[None] + noise, mesh.faces)
            shift = curvature - curvature0
            s1 += shift.sum(axis=0)
            s2 += (shift ** 2).sum(axis=0)
            tilt += np.sum((normals - normal0) ** 2, axis=2).sum(axis=0)
        curvature_ratio = ((s2 - s1 ** 2 / samples) / (samples - 1)) / linear["curvature_sd"] ** 2
        normal_ratio = (tilt / samples) / linear["normal_sd"] ** 2
        models.append({
            "covariance": name, "curvature_sd": linear["curvature_sd"], "normal_sd": linear["normal_sd"],
            "curvature_ratio": curvature_ratio, "normal_ratio": normal_ratio,
            "max_abs_z_curvature": float(np.max(np.abs(curvature_ratio - 1)) / se),
            "max_abs_z_normal": float(np.max(np.abs(normal_ratio - 1)) / se),
            "median_curvature_ratio": float(np.median(curvature_ratio)),
            "median_normal_ratio": float(np.median(normal_ratio)),
            "curvature_sd_range": [float(np.min(linear["curvature_sd"])), float(np.max(linear["curvature_sd"]))],
            "normal_sd_range": [float(np.min(linear["normal_sd"])), float(np.max(linear["normal_sd"]))],
            "valence5_curvature_sd": float(np.mean(linear["curvature_sd"][valence == 5])),
            "valence6_curvature_sd_median": float(np.median(linear["curvature_sd"][valence == 6]))})
    return {"level": level, "h": mesh.mean_edge(), "vertices": n, "samples": samples, "sigma": sigma,
            "normal_sigma": normal_sigma, "tangential_sigma": tangential_sigma, "far_vertex": far_valence6_vertex(mesh),
            "models": models}


def scaling_study(levels=(2, 3, 4, 5), index=MARKER_START):
    """Linearized sensitivity |d observable / d vertices| against mean edge length.

    The marker distance follows one declared geodesic at every level, and its
    gradient is split into marker-face and interior strip vertices.
    """
    rows = []
    for level in levels:
        mesh = G.icosphere(level)
        strip, items = observables(mesh, index)
        row = {"level": level, "h": mesh.mean_edge(), "strip_vertex_margin": strip["margin"]}
        for item in items:
            base = mesh.vertices[item["ids"]]
            jac = _fd_jacobian(item["function"], base)
            row[item["name"]] = float(np.sqrt(np.sum(jac ** 2)))
            if item["name"] == "marker geodesic distance":
                split = gradient_split(strip, base, jac)
                row["marker-face part"], row["interior part"] = split["marker"], split["interior"]
        rows.append(row)
    names = [k for k in rows[0] if k not in ("level", "h", "strip_vertex_margin")]
    hs = [r["h"] for r in rows]
    slopes = {name: fitted_order(hs, [r[name] for r in rows]) for name in names}
    local = {name: [math.log(a[name] / b[name]) / math.log(a["h"] / b["h"]) for a, b in zip(rows, rows[1:])]
             for name in names}
    return {"start_index": index, "rows": rows, "slopes": slopes, "local_slopes": local}


def refinement_noise_study(levels=(2, 3, 4, 5), sigma=1e-3, samples=2000, seed=SEED + 50):
    """Total curvature error under fixed vertex noise as the mesh is refined (valence-6 vertex)."""
    rows = []
    rng = np.random.Generator(np.random.PCG64(seed))
    for level in levels:
        mesh = G.icosphere(level)
        vertex = far_valence6_vertex(mesh)
        ids, ring = G.one_ring(mesh, vertex)
        base = mesh.vertices[ids]
        clean = float(G.batch_angle_defect_curvature(base[None], 0, ring)[0])
        noise = rng.standard_normal((samples,) + base.shape) * sigma
        noisy = G.batch_angle_defect_curvature(base[None] + noise, 0, ring)
        rows.append({"level": level, "h": mesh.mean_edge(), "discretization_error": clean - 1.0,
                     "total_rms_error": float(np.sqrt(np.mean((noisy - 1.0) ** 2))),
                     "noise_sd": float(np.std(noisy, ddof=1))})
    return {"sigma": sigma, "samples": samples, "rows": rows}


# ---------------------------------------------------------------- geometry versus sensor variance
def variance_split_study(level=3, geometry_sigmas=(1e-4, 1e-3), sensor_sigmas=(1e-4, 5e-4, 2e-3), outer=1000,
                         inner=16, fresh=20000, repeats=(1, 4, 16, 64), repeat_case=(1e-3, 2e-3),
                         normal_case=(1e-3, 5e-4), seed=SEED + 100):
    """Residual r = y - d(V_nominal), y = d(V_nominal + eta) + eps: nested and fresh Monte Carlo.

    eta is isotropic per vertex coordinate unless stated; the normal-only case
    moves each vertex along its unit-sphere normal (shape noise without the
    tangential re-parameterisation that drags barycentric markers).
    """
    mesh = G.icosphere(level)
    strip = marker_strip(mesh)
    base = mesh.vertices[strip["ids"]]
    distance, corridor = strip_distance(strip)
    nominal = float(distance(base[None])[0])
    jac = _fd_jacobian(distance, base)
    split = gradient_split(strip, base, jac)
    gain = split["total"]
    rng = np.random.Generator(np.random.PCG64(seed))
    scenarios = []
    for sigma_g in geometry_sigmas:
        for sigma_s in sensor_sigmas:
            built = distance(base[None] + rng.standard_normal((outer,) + base.shape) * sigma_g) - nominal
            y = built[:, None] + rng.standard_normal((outer, inner)) * sigma_s
            within = float(np.mean(np.var(y, axis=1, ddof=1)))
            between = float(np.var(y.mean(axis=1), ddof=1))
            total = float(np.var(y, ddof=1))
            n_all = outer * inner
            sst = (n_all - 1) * total
            ssb = inner * (outer - 1) * between
            ssw = outer * (inner - 1) * within
            perturbed = base[None] + rng.standard_normal((fresh,) + base.shape) * sigma_g
            fresh_y = distance(perturbed) - nominal + rng.standard_normal(fresh) * sigma_s
            outside = float(np.mean(corridor(perturbed) < 0))
            geometry_linear = (gain * sigma_g) ** 2
            predicted = geometry_linear + sigma_s ** 2
            scenarios.append({"sigma_geometry": sigma_g, "sigma_sensor": sigma_s,
                              "geometry_variance_linear": geometry_linear, "sensor_variance": sigma_s ** 2,
                              "within_variance": within, "between_corrected": between - within / inner,
                              "nested_total": total, "anova_residual": (sst - ssb - ssw) / sst,
                              "fresh_total": float(np.var(fresh_y, ddof=1)), "predicted_total": predicted,
                              "fresh_ratio": float(np.var(fresh_y, ddof=1)) / predicted,
                              "within_ratio": within / sigma_s ** 2,
                              "between_ratio": (between - within / inner) / geometry_linear,
                              "geometry_share": geometry_linear / predicted,
                              "crossover_sensor_sigma": gain * sigma_g, "corridor_left_fraction": outside})
    sigma_g, sigma_s = repeat_case
    averaging = []
    for count in repeats:
        built = distance(base[None] + rng.standard_normal((fresh,) + base.shape) * sigma_g) - nominal
        mean_sensor = rng.standard_normal((fresh, count)).mean(axis=1) * sigma_s
        predicted = (gain * sigma_g) ** 2 + sigma_s ** 2 / count
        variance = float(np.var(built + mean_sensor, ddof=1))
        averaging.append({"repeats": count, "variance": variance, "predicted": predicted, "ratio": variance / predicted,
                          "geometry_share": (gain * sigma_g) ** 2 / predicted})
    sigma_g, sigma_s = normal_case
    normals = base / np.linalg.norm(base, axis=1)[:, None]
    shape = distance(base[None] + rng.standard_normal((fresh, len(base)))[:, :, None] * sigma_g * normals) - nominal
    readings = shape + rng.standard_normal(fresh) * sigma_s
    geometry_normal = (split["normal"] * sigma_g) ** 2
    normal_only = {"sigma_geometry": sigma_g, "sigma_sensor": sigma_s, "gain": split["normal"],
                   "geometry_variance_linear": geometry_normal, "geometry_variance_mc": float(np.var(shape, ddof=1)),
                   "total_mc": float(np.var(readings, ddof=1)), "predicted_total": geometry_normal + sigma_s ** 2,
                   "geometry_share": geometry_normal / (geometry_normal + sigma_s ** 2),
                   "crossover_sensor_sigma": split["normal"] * sigma_g}
    return {"level": level, "nominal_distance": nominal, "gain": gain, "gain_split": split, "outer": outer,
            "inner": inner, "fresh": fresh, "strip_start_index": strip["start_index"],
            "strip_vertex_margin": strip["margin"], "scenarios": scenarios,
            "averaging": {"sigma_geometry": repeat_case[0], "sigma_sensor": repeat_case[1], "rows": averaging},
            "normal_only": normal_only}
