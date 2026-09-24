"""Deterministic studies behind tasks T038-T044 (triangle-mesh geodesics and geometry uncertainty).

Each study returns plain numbers and lists; the task module turns them into
findings. Smooth references come from ``ciw.lab.surfaces`` (great circles,
helices, closed-form curvature) and ``ciw.lab.jacobi`` (the smooth transfer
matrix). Random inputs use seeded PCG64 generators only.

Non-claims: the meshes are generated from declared formulas in normalized
units and the noise models are declared Gaussian models. Nothing here is a
scanned surface, a sensor or a calibration, and a Monte Carlo agreement is
agreement between two computations on the same declared model.
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np

from . import jacobi
from .integrators import observed_order
from .surfaces import Sphere, Torus
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
