"""Mesh topology, winding classes, cylinder geodesic classes and 0-D persistence."""
import itertools
import math

import numpy as np
import pytest

from ciw.science import topology as t
from ciw.science._common import Refusal
from ciw.science.mesh import cylinder_patch, icosphere, plane_patch, torus_mesh
from ciw.science.vocabulary import CAPABILITIES

TAU = 2 * math.pi


def refused(code, call, *args, **kwargs):
    with pytest.raises(Refusal) as info:
        call(*args, **kwargs)
    assert info.value.code == code, info.value.to_dict()
    return info.value


def moebius_faces(nu=12, nv=3):
    """Strip [0, 2pi) x [-w, w] whose seam glues (2pi, v) to (0, -v); vertex (i, j) = i * (nv + 1) + j."""
    index = lambda i, j: i * (nv + 1) + j
    faces = []
    for i in range(nu):
        for j in range(nv):
            b, c = ((index(i + 1, j), index(i + 1, j + 1)) if i < nu - 1 else (index(0, nv - j), index(0, nv - j - 1)))
            faces += [[index(i, j), b, c], [index(i, j), c, index(i, j + 1)]]
    return faces


def test_capability_is_declared():
    assert t.CAPABILITY in CAPABILITIES


@pytest.mark.parametrize("surface, chi, loops, genus, model", [
    (icosphere(1.0, 2), 2, 0, 0, "trivial"),
    (torus_mesh(2.0, 0.5, 12, 8), 0, 0, 1, "Z^2"),
    (cylinder_patch(1.0, 2.0, TAU, 10, 3), 0, 2, 0, "Z"),
    (cylinder_patch(1.0, 2.0, math.pi, 10, 3), 1, 1, 0, "trivial"),
    (plane_patch(1.0, 1.0, 5, 4), 1, 1, 0, "trivial"),
])
def test_surface_invariants(surface, chi, loops, genus, model):
    faces = surface.mesh.faces
    assert t.euler_characteristic(faces, surface.mesh.vertex_count) == chi
    assert len(t.boundary_loops(faces)) == loops
    assert t.genus(faces) == genus
    assert t.is_orientable(faces) and t.is_consistently_oriented(faces)
    assert len(t.connected_components(faces)) == 1
    summary = t.classify_surface(faces)
    assert (summary["euler_characteristic"], summary["boundary_loops"], summary["genus"]) == (chi, loops, genus)
    assert summary["homotopy_model"] == model and summary["simply_connected"] == (model == "trivial")


def test_moebius_strip_is_detected_as_non_orientable():
    faces = moebius_faces()
    assert t.euler_characteristic(faces) == 0
    loops = t.boundary_loops(faces)
    assert len(loops) == 1 and len(loops[0]) == 2 * 12        # one boundary circle running round twice
    assert not t.is_orientable(faces) and not t.is_consistently_oriented(faces)
    refused("non_orientable_surface", t.genus, faces)
    summary = t.classify_surface(faces)
    assert summary["crosscaps"] == 1 and summary["genus"] is None and summary["homotopy_model"] == "Z"


def test_orientability_is_distinct_from_consistent_orientation():
    faces = icosphere(1.0, 1).mesh.faces.copy()
    faces[7] = faces[7, ::-1]
    assert t.is_orientable(faces) and not t.is_consistently_oriented(faces)
    assert t.genus(faces) == 0


def test_boundary_loop_is_ordered_and_follows_face_orientation():
    surface = plane_patch(1.0, 1.0, 4, 3)
    (loop,) = t.boundary_loops(surface.mesh.faces)
    assert len(loop) == 2 * (4 + 3) and loop[0] == min(loop)
    edges = {tuple(sorted(edge)) for edge in surface.mesh.edges[surface.mesh.edge_face_counts == 1].tolist()}
    assert {tuple(sorted(pair)) for pair in zip(loop, loop[1:] + loop[:1])} == edges
    x, y = surface.parameters[loop, 0], surface.parameters[loop, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) == pytest.approx(1.0)   # counter-clockwise
    tube = cylinder_patch(1.0, 2.0, TAU, 9, 2)
    rings = t.boundary_loops(tube.mesh.faces)
    assert [len(ring) for ring in rings] == [9, 9]
    assert sorted({float(tube.parameters[ring[0], 1]) for ring in rings}) == [0.0, 2.0]


def test_non_manifold_and_disconnected_input_is_refused():
    fin = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]
    for call in (t.boundary_loops, t.is_orientable, t.genus):
        refused("non_manifold_edge", call, fin)
    bowtie = [[0, 1, 2], [0, 3, 4]]                              # two triangles pinched at vertex 0
    assert len(t.connected_components(bowtie)) == 1
    refused("non_manifold_vertex", t.boundary_loops, bowtie)
    refused("non_manifold_vertex", t.genus, bowtie)
    double_cone = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2], [0, 5, 4], [0, 4, 6], [4, 5, 6], [0, 6, 5]]
    assert t.boundary_loops(double_cone) == [] and t.euler_characteristic(double_cone) == 3
    refused("non_manifold_vertex", t.genus, double_cone)
    apart = [[0, 1, 2], [3, 4, 5], [5, 4, 6]]
    assert t.connected_components(apart) == [[0, 1, 2], [3, 4, 5, 6]]
    refused("disconnected_surface", t.genus, apart)
    refused("disconnected_surface", t.classify_surface, apart)


def test_face_validation(monkeypatch):
    assert t.euler_characteristic([[0, 1, 2]], 10) == 1         # unreferenced vertices are not counted
    refused("face_index_out_of_range", t.euler_characteristic, [[0, 1, 12]], 10)
    refused("face_index_out_of_range", t.euler_characteristic, [[0, 1, -1]])
    refused("face_index_out_of_range", t.euler_characteristic, [[0, 1, 2**70]])
    refused("repeated_face_vertex", t.euler_characteristic, [[0, 1, 1]])
    refused("malformed_mesh", t.euler_characteristic, [[0, 1, 2.0]])
    refused("malformed_mesh", t.euler_characteristic, np.array([[0.0, 1.0, 2.0]]))
    refused("malformed_mesh", t.euler_characteristic, [[0, 1]])
    refused("malformed_mesh", t.euler_characteristic, [])
    monkeypatch.setattr(t, "MAX_FACES", 1)
    refused("oversized_input", t.face_array, [[0, 1, 2], [0, 2, 3]], limit=t.MAX_FACES)


# ----------------------------------------------------------------------------- winding
def test_unwrap_recovers_a_sampled_continuous_angle():
    truth = 0.3 + np.linspace(0.0, 5.5 * TAU, 400) + 0.4 * np.sin(np.linspace(0.0, 9.0, 400))
    wrapped = np.angle(np.exp(1j * truth))
    lifted = t.unwrap_angles(wrapped)
    assert np.allclose(lifted, truth - truth[0] + wrapped[0], atol=1e-9)
    assert np.allclose(lifted, np.unwrap(wrapped), atol=1e-9)


def test_unwrap_refuses_ambiguous_steps():
    error = refused("undersampled_path", t.unwrap_angles, [0.0, 0.1, math.radians(178.0) + 0.1])
    assert error.detail["step_index"] == 1
    refused("undersampled_path", t.unwrap_angles, [0.0, math.pi])                      # exact half-turn
    refused("undersampled_path", t.unwrap_angles, [0.0, -math.pi + 0.01], margin=0.0 + 0.02)
    assert t.unwrap_angles([0.0, math.radians(170.0)])[-1] == pytest.approx(math.radians(170.0))
    assert t.unwrap_angles([0.0, math.radians(178.0)], margin=0.0)[-1] == pytest.approx(math.radians(178.0))
    refused("undersampled_path", t.unwrap_angles, [0.0, 1.0], max_step=0.5)
    refused("malformed_path", t.unwrap_angles, [0.0, float("nan")])
    refused("malformed_path", t.unwrap_angles, [[0.0, 1.0]])
    refused("angle_out_of_range", t.unwrap_angles, [0.0, 1e7])
    refused("out_of_domain", t.unwrap_angles, [0.0, 1.0], max_step=4.0)


def helix(p, q, k, radius=None, samples=400):
    """Straight segment in the unrolled (phi, z) plane from p to the class-k lift of q."""
    dphi = (q[0] - p[0] + math.pi) % TAU - math.pi + TAU * k
    s = np.linspace(0.0, 1.0, samples)
    return np.column_stack([p[0] + s * dphi, p[1] + s * (q[1] - p[1])])


def test_cylinder_classes_of_sampled_paths():
    p, q = (0.4, 0.0), (2.9, 1.5)
    for k in range(-3, 4):
        path = helix(p, q, k)
        path[:, 0] = np.angle(np.exp(1j * path[:, 0]))              # the sensor only reports wrapped angles
        assert t.cylinder_homotopy_class(path) == k
    loop = np.column_stack([np.linspace(0.0, -2 * TAU, 50), np.zeros(50)])
    assert t.cylinder_homotopy_class(loop) == -2
    refused("malformed_path", t.cylinder_homotopy_class, [[0.0, 0.0]])
    refused("undersampled_path", t.cylinder_homotopy_class, [[0.0, 0.0], [3.1, 0.0]])


def test_torus_classes_of_sampled_paths():
    s = np.linspace(0.0, 1.0, 300)
    path = np.column_stack([0.2 + s * (2 * TAU + 0.5), 1.0 - s * (TAU + 0.3)])
    assert t.torus_homotopy_class(np.angle(np.exp(1j * path))) == (2, -1)
    assert t.torus_homotopy_class([[0.0, 0.0], [0.5, 0.5]]) == (0, 0)


def test_paths_equivalent():
    straight = [[0.0, 0.0], [1.0, 1.0]]
    detour = [[0.0, 0.0], [3.0, -2.0], [1.0, 1.0]]
    assert t.paths_equivalent(straight, detour, "plane")
    assert not t.paths_equivalent(straight, [[0.0, 0.0], [1.0, 1.1]], "plane")
    arc = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    other = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    assert t.paths_equivalent(arc, other, "sphere")
    p, q = (0.4, 0.0), (2.9, 1.5)
    assert t.paths_equivalent(helix(p, q, 1), helix(p, q, 1, samples=37), "cylinder")
    assert not t.paths_equivalent(helix(p, q, 1), helix(p, q, 0), "cylinder")
    shifted = helix(p, q, 1)
    shifted[:, 0] += TAU                                                 # same points, other lift
    assert t.paths_equivalent(helix(p, q, 1), shifted, "cylinder")
    wiggle = helix(p, q, 1)
    wiggle[:, 0] += 0.8 * np.sin(np.linspace(0.0, math.pi, len(wiggle)))  # homotopic deformation
    assert t.paths_equivalent(helix(p, q, 1), wiggle, "cylinder")
    s = np.linspace(0.0, 1.0, 200)[:, None]
    a = np.hstack([s * TAU, s * 0.0])
    assert t.paths_equivalent(a, np.hstack([s * TAU, np.sin(s * math.pi)]), "torus")
    assert not t.paths_equivalent(a, np.hstack([s * TAU, s * TAU]), "torus")
    refused("unknown_surface_kind", t.paths_equivalent, straight, straight, "klein")
    refused("malformed_path", t.paths_equivalent, straight, straight, "sphere")


def polyline_length_3d(path_phi_z, radius):
    xyz = np.column_stack([radius * np.cos(path_phi_z[:, 0]), radius * np.sin(path_phi_z[:, 0]), path_phi_z[:, 1]])
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def test_cylinder_geodesic_classes_against_brute_force():
    radius, p, q = 0.7, (5.9, -0.2), (0.6, 1.1)
    classes = t.cylinder_geodesic_classes(p, q, radius, range(-4, 5))
    assert [item["k"] for item in classes] == list(range(-4, 5))
    for item in classes:
        path = helix(p, q, item["k"], samples=20001)                     # chord error ~ (R dphi)^2 / 24
        assert polyline_length_3d(path, radius) == pytest.approx(item["length"], rel=1e-6)
        assert t.cylinder_homotopy_class(path) == item["k"]
        assert item["turns"] == pytest.approx(item["angular_displacement_rad"] / TAU)
        assert math.tan(item["helix_angle_rad"]) == pytest.approx(radius * abs(item["angular_displacement_rad"]) / 1.3)
        bump = path.copy()                                               # any same-class deformation is longer
        bump[:, 1] += 0.05 * np.sin(np.linspace(0.0, math.pi, len(bump)))
        assert polyline_length_3d(bump, radius) > item["length"]
    brute = min(range(-50, 51), key=lambda k: polyline_length_3d(helix(p, q, k, samples=2001), radius))
    best = t.shortest_class(p, q, radius)
    assert best["k"] == brute == 0 and best["tied_classes"] == [0]
    assert best["length"] == pytest.approx(min(item["length"] for item in classes))


def test_shortest_class_reports_half_turn_ties():
    best = t.shortest_class((0.0, 0.0), (math.pi, 1.0), 1.0)
    assert best["tied_classes"] == [0, 1] and best["length"] == pytest.approx(math.hypot(math.pi, 1.0))
    same = t.cylinder_geodesic_classes((1.0, 2.0), (1.0, 2.0), 1.0, [0, 1])
    assert same[0]["length"] == 0.0 and same[0]["helix_angle_rad"] is None
    assert same[1]["helix_angle_rad"] == pytest.approx(math.pi / 2)
    refused("malformed_record", t.cylinder_geodesic_classes, (0, 0), (1, 1), 1.0, [0, 0])
    refused("oversized_input", t.cylinder_geodesic_classes, (0, 0), (1, 1), 1.0, range(20000))
    refused("out_of_domain", t.cylinder_geodesic_classes, (0, 0), (1, 1), 0.0, [0])


# ----------------------------------------------------------------------------- persistence
def path_edges(n):
    return [[i, i + 1] for i in range(n - 1)]


def grid_edges(rows, cols):
    index = lambda r, c: r * cols + c
    return ([[index(r, c), index(r, c + 1)] for r in range(rows) for c in range(cols - 1)]
            + [[index(r, c), index(r + 1, c)] for r in range(rows - 1) for c in range(cols)])


def test_persistence_on_small_graphs():
    assert t.persistence_0d([0, 3, 1, 4, 2], path_edges(5)) == [(0.0, math.inf), (1.0, 3.0), (2.0, 4.0)]
    full = t.persistence_0d([0, 3, 1, 4, 2], path_edges(5), keep_zero=True)
    assert len(full) == 5 and full.count((3.0, 3.0)) == 1 and full.count((4.0, 4.0)) == 1
    assert t.persistence_0d([2.0, 1.0, 5.0], [[0, 1]]) == [(1.0, math.inf), (5.0, math.inf)]
    assert t.persistence_0d([1.0, 1.0], [[0, 1]], keep_zero=True) == [(1.0, 1.0), (1.0, math.inf)]
    refused("self_loop", t.persistence_0d, [0.0, 1.0], [[1, 1]])
    refused("edge_index_out_of_range", t.persistence_0d, [0.0, 1.0], [[0, 2]])
    refused("malformed_graph", t.persistence_0d, [0.0, float("inf")], [[0, 1]])
    refused("malformed_graph", t.persistence_0d, [0.0, 1.0], [[0, 1.0]])


def brute_bottleneck(a, b):
    """Minimum over all perfect matchings of the diagonal-augmented cost matrix."""
    a, b = np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2)
    size = len(a) + len(b)
    if size == 0:
        return 0.0
    cost = np.full((size, size), np.inf)
    for i, j in itertools.product(range(len(a)), range(len(b))):
        cost[i, j] = np.abs(a[i] - b[j]).max()
    for i in range(len(a)):
        cost[i, len(b) + i] = (a[i, 1] - a[i, 0]) / 2
    for j in range(len(b)):
        cost[len(a) + j, j] = (b[j, 1] - b[j, 0]) / 2
        cost[len(a) + j, len(b):] = 0.0
    return min(max(cost[row, col] for row, col in enumerate(perm)) for perm in itertools.permutations(range(size)))


def test_bottleneck_matches_brute_force_and_hand_values():
    assert t.bottleneck_distance([], []) == 0.0
    assert t.bottleneck_distance([(0.0, 2.0)], []) == 1.0
    assert t.bottleneck_distance([(0.0, 2.0)], [(0.5, 2.25)]) == 0.5
    assert t.bottleneck_distance([(0.0, math.inf)], [(0.75, math.inf), (1.0, 1.5)]) == 0.75
    assert t.bottleneck_distance([(0.0, math.inf)], []) == math.inf
    rng = np.random.default_rng(7)
    for _ in range(40):
        a = [tuple(sorted(rng.uniform(0, 3, 2))) for _ in range(rng.integers(0, 4))]
        b = [tuple(sorted(rng.uniform(0, 3, 2))) for _ in range(rng.integers(0, 4))]
        expected = brute_bottleneck(a, b)
        assert t.bottleneck_distance(a, b) == pytest.approx(expected, abs=1e-15)
        assert t.bottleneck_distance(b, a) == pytest.approx(expected, abs=1e-15)
    refused("malformed_diagram", t.bottleneck_distance, [(2.0, 1.0)], [])
    refused("malformed_diagram", t.bottleneck_distance, [(math.inf, math.inf)], [])
    refused("oversized_input", t.bottleneck_distance, [(0.0, 1.0)] * (t.MAX_DIAGRAM_POINTS + 1), [])


@pytest.mark.parametrize("seed", range(6))
def test_stability_theorem_holds_numerically(seed):
    rng = np.random.default_rng(seed)
    for vertex_count, edges in ((60, path_edges(60)), (49, grid_edges(7, 7))):
        f = rng.normal(size=vertex_count)
        for g in (f + rng.uniform(-0.2, 0.2, vertex_count), rng.normal(size=vertex_count)):
            distance = t.bottleneck_distance(t.persistence_0d(f, edges), t.persistence_0d(g, edges))
            assert distance <= np.abs(f - g).max() + 1e-12
        assert t.bottleneck_distance(t.persistence_0d(f, edges), t.persistence_0d(f + 0.3, edges)) == pytest.approx(0.3)
