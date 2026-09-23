"""Mesh-surface geodesics: validation, generators, Dijkstra bounds and heat-method convergence."""
import json
import math

import numpy as np
import pytest

from ciw.science import mesh as m
from ciw.science._common import Refusal
from ciw.science.vocabulary import CAPABILITIES


def nearest(surface, target):
    return int(np.argmin(np.linalg.norm(surface.parameters - np.asarray(target, dtype=float), axis=1)))


def rms_rel(estimate, exact):
    return m.compare_to_reference(estimate, exact)["rms_rel_error"]


def refused(code, call, *args, **kwargs):
    with pytest.raises(Refusal) as info:
        call(*args, **kwargs)
    assert info.value.code == code, info.value.to_dict()
    return info.value


SQUARE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]


def test_capability_is_declared():
    assert m.CAPABILITY in CAPABILITIES


def test_generators_have_expected_counts_and_outward_orientation():
    plane = m.plane_patch(2.0, 1.0, 4, 3)
    assert (plane.mesh.vertex_count, plane.mesh.face_count) == (20, 24)
    assert np.allclose(plane.mesh.face_areas.sum(), 2.0)
    normals = np.cross(*(plane.mesh.vertices[plane.mesh.faces[:, k]] - plane.mesh.vertices[plane.mesh.faces[:, 0]]
                         for k in (1, 2)))
    assert (normals[:, 2] > 0).all()

    sphere = m.icosphere(2.0, 2)
    assert sphere.mesh.vertex_count == 10 * 4 ** 2 + 2 and sphere.mesh.face_count == 20 * 4 ** 2
    assert np.allclose(np.linalg.norm(sphere.mesh.vertices, axis=1), 2.0)
    p = [sphere.mesh.vertices[sphere.mesh.faces[:, k]] for k in range(3)]
    volume = np.einsum("ij,ij->i", p[0], np.cross(p[1], p[2])).sum() / 6.0
    assert 0.9 * (4 / 3) * math.pi * 8 < volume < (4 / 3) * math.pi * 8      # inscribed and outward

    tube = m.cylinder_patch(1.5, 2.0, 2 * math.pi, 12, 4)
    assert tube.kind == "cylinder_tube" and tube.mesh.vertex_count == 12 * 5   # seam vertices shared
    patch = m.cylinder_patch(1.5, 2.0, math.pi, 12, 4)
    assert patch.kind == "cylinder_patch" and patch.mesh.vertex_count == 13 * 5
    for surface in (tube, patch):
        v, f = surface.mesh.vertices, surface.mesh.faces
        normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        radial = v[f].mean(axis=1) * np.array([1.0, 1.0, 0.0])
        assert (np.einsum("ij,ij->i", normal, radial) > 0).all()
        assert np.allclose(np.hypot(v[:, 0], v[:, 1]), 1.5)

    torus = m.torus_mesh(2.0, 0.5, 16, 8)
    assert torus.mesh.vertex_count == 128 and torus.mesh.face_count == 256
    assert not torus.mesh.boundary_vertices.any()


def test_json_round_trip_identity_and_immutability():
    surface = m.cylinder_patch(1.0, 1.0, 1.0, 3, 2)
    record = json.loads(json.dumps(surface.mesh.to_json()))
    assert record["schema"] == m.SCHEMA
    again = m.TriangleMesh.from_json(record)
    assert again.identity() == surface.mesh.identity() and again.identity().startswith("sha256:")
    moved = dict(record, vertices=[list(row) for row in record["vertices"]])
    moved["vertices"][0][2] += 1e-9
    assert m.TriangleMesh.from_json(moved).identity() != surface.mesh.identity()
    with pytest.raises(ValueError):
        surface.mesh.vertices[0, 0] = 5.0
    refused("unsupported_schema", m.TriangleMesh.from_json, dict(record, schema="ciw.triangle-mesh.v0"))
    refused("malformed_record", m.TriangleMesh.from_json, dict(record, extra=1))
    refused("dimension_mismatch", m.TriangleMesh, SQUARE, [[0, 1, 2], [0, 2, 3]], length_unit="s")


@pytest.mark.parametrize("vertices, faces, code", [
    ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], "degenerate_face"),                       # collinear
    ([[0, 0, 0], [1, 0, 0], [1, 1e-13, 0]], [[0, 1, 2]], "degenerate_face"),                   # sliver
    (SQUARE, [[0, 1, 2], [0, 2, 4]], "face_index_out_of_range"),
    (SQUARE, [[0, 1, 2], [0, 2, -1]], "face_index_out_of_range"),
    ([[0, 0, 0], [1, 0, float("nan")], [0, 1, 0]], [[0, 1, 2]], "non_finite_vertex"),
    ([[0, 0, 0], [1, 0, math.inf], [0, 1, 0]], [[0, 1, 2]], "non_finite_vertex"),
    (SQUARE, [[0, 1, 2], [0, 2, 2]], "repeated_face_vertex"),
    (SQUARE, [[0, 1, 2], [0, 2, 3.0]], "malformed_mesh"),                                      # float index
    (SQUARE, [[0, 1, 2], [0, 2, True]], "malformed_mesh"),
    ([[0, 0, "0"], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], "malformed_mesh"),
    ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], "malformed_mesh"),
    (SQUARE, [[0, 1, 2]], "unreferenced_vertex"),
    (SQUARE, [[0, 1, 2], [0, 2, 3], [2, 1, 0]], "duplicate_face"),
    (SQUARE + [[0.5, 0.5, 1.0]], [[0, 1, 2], [0, 2, 3], [0, 2, 4]], "non_manifold_edge"),
])
def test_invalid_meshes_are_refused_not_repaired(vertices, faces, code):
    refused(code, m.TriangleMesh, vertices, faces)


def test_size_bounds(monkeypatch):
    refused("mesh_too_large", m.icosphere, 1.0, 6)            # 40962 vertices, refused before allocation
    refused("mesh_too_large", m.plane_patch, 1.0, 1.0, 200, 200)
    surface = m.plane_patch(1.0, 1.0, 8, 8)
    monkeypatch.setattr(m, "MAX_DENSE_VERTICES", 50)
    error = refused("mesh_too_large_for_dense_solver", m.heat_method_distance, surface.mesh, 0)
    assert error.detail["vertices"] == 81
    assert np.isfinite(m.edge_graph_distance(surface.mesh, 0)).all()   # graph ops keep the larger bound
    monkeypatch.setattr(m, "MAX_VERTICES", 10)
    refused("mesh_too_large", m.TriangleMesh, surface.mesh.vertices, surface.mesh.faces)


def test_generator_argument_refusals():
    refused("out_of_domain", m.plane_patch, 0.0, 1.0, 2, 2)
    refused("malformed_record", m.plane_patch, 1.0, 1.0, 2.0, 2)
    refused("out_of_domain", m.cylinder_patch, 1.0, 1.0, 7.0, 4, 2)
    refused("out_of_domain", m.cylinder_patch, 1.0, 1.0, 2 * math.pi, 2, 2)     # a tube needs nu >= 3
    refused("out_of_domain", m.torus_mesh, 1.0, 1.0, 8, 8)
    refused("malformed_record", m.icosphere, float("nan"), 1)


def test_dijkstra_is_an_upper_bound_on_polyhedral_distance():
    plane = m.plane_patch(1.0, 1.5, 9, 13)
    for source in (0, nearest(plane, (0.5, 0.7))):
        graph, exact = m.edge_graph_distance(plane.mesh, source), plane.reference_distance(source)
        assert (graph >= exact - 1e-12).all()
        row = np.isclose(plane.parameters[:, 1], plane.parameters[source, 1])
        assert np.allclose(graph[row], exact[row], atol=1e-12)                 # exact along grid lines
    for surface in (m.cylinder_patch(1.0, 2.0, 2.5, 10, 6), m.cylinder_patch(1.0, 2.0, 2 * math.pi, 16, 6)):
        source = nearest(surface, (1.2, 1.0))
        graph = m.edge_graph_distance(surface.mesh, source)
        assert (graph >= surface.reference_distance(source, "polyhedral") - 1e-12).all()
        # Chords are shorter than arcs: edge paths along a ring undercut the smooth distance,
        # which is why the bound is stated for the polyhedral surface only.
        assert (graph - surface.reference_distance(source, "smooth")).min() < -1e-3


def test_heat_method_converges_on_cylinder_patch():
    errors = []
    for nu, nv in ((8, 6), (16, 12), (32, 24)):
        surface = m.cylinder_patch(1.0, 2.0, math.pi, nu, nv)
        source = nearest(surface, (math.pi / 2, 1.0))
        distance = m.heat_method_distance(surface.mesh, source)
        assert distance[source] == 0.0
        errors.append(rms_rel(distance, surface.reference_distance(source)))
    assert errors[0] > errors[1] > errors[2] and errors[2] < 0.04


def test_heat_method_converges_on_icosphere():
    errors = []
    for level in (1, 2, 3):
        surface = m.icosphere(2.0, level)
        errors.append(m.compare_to_reference(m.heat_method_distance(surface.mesh, 0), surface.reference_distance(0)))
    assert errors[0]["rms_rel_error"] > errors[1]["rms_rel_error"] > errors[2]["rms_rel_error"]
    assert errors[0]["rms_abs_error"] > errors[1]["rms_abs_error"] > errors[2]["rms_abs_error"]
    assert errors[2]["rms_rel_error"] < 0.04


def test_heat_method_beats_dijkstra_on_plane_patch():
    surface = m.plane_patch(1.0, 1.0, 20, 20)
    source = nearest(surface, (0.5, 0.5))
    exact = surface.reference_distance(source)
    heat = rms_rel(m.heat_method_distance(surface.mesh, source), exact)
    graph = rms_rel(m.edge_graph_distance(surface.mesh, source), exact)
    assert heat < 0.05 and graph > 0.15 and heat < graph / 4


def test_heat_method_on_closed_tube_respects_wrapping():
    surface = m.cylinder_patch(1.0, 3.0, 2 * math.pi, 32, 12)
    source = nearest(surface, (0.0, 1.5))
    distance = m.heat_method_distance(surface.mesh, source)
    assert rms_rel(distance, surface.reference_distance(source)) < 0.05
    mirror = nearest(surface, (math.pi / 2, 1.5)), nearest(surface, (3 * math.pi / 2, 1.5))
    assert distance[mirror[0]] == pytest.approx(distance[mirror[1]], rel=1e-9)


def test_heat_method_scales_with_the_mesh_and_handles_components():
    surface = m.icosphere(1.0, 2)
    base = m.heat_method_distance(surface.mesh, 5)
    scaled = m.heat_method_distance(m.TriangleMesh(3.0 * surface.mesh.vertices, surface.mesh.faces), 5)
    assert np.allclose(scaled, 3.0 * base, rtol=1e-8, atol=1e-10)
    far = SQUARE + [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]
    split = m.TriangleMesh(far, [[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    heat, graph = m.heat_method_distance(split, 0), m.edge_graph_distance(split, 0)
    assert np.isinf(heat[4:]).all() and np.isinf(graph[4:]).all() and np.isfinite(heat[:4]).all()


def test_boundary_condition_options():
    surface = m.plane_patch(1.0, 1.0, 12, 12)
    centre, corner = nearest(surface, (0.5, 0.5)), 0
    exact = surface.reference_distance(centre)
    results = {mode: m.heat_method_distance(surface.mesh, centre, boundary=mode) for mode in m.BOUNDARY_CONDITIONS}
    assert all(rms_rel(value, exact) < 0.1 for value in results.values())
    assert not np.allclose(results["neumann"], results["dirichlet"])
    refused("unknown_boundary_condition", m.heat_method_distance, surface.mesh, centre, boundary="robin")
    refused("heat_gradient_vanished", m.heat_method_distance, surface.mesh, corner, boundary="dirichlet")
    assert rms_rel(m.heat_method_distance(surface.mesh, corner), surface.reference_distance(corner)) < 0.1
    for bad in (0.0, -1.0, float("nan"), 1e9):
        with pytest.raises(Refusal):
            m.heat_method_distance(surface.mesh, centre, t_factor=bad)
    refused("out_of_domain", m.heat_method_distance, surface.mesh, surface.mesh.vertex_count)


def test_cotangent_laplacian_properties():
    surface = m.plane_patch(1.0, 1.0, 5, 4)
    laplacian = m.cotangent_laplacian(surface.mesh.vertices, surface.mesh.faces)
    assert np.allclose(laplacian, laplacian.T) and np.allclose(laplacian.sum(axis=1), 0.0)
    assert np.linalg.eigvalsh(laplacian).max() < 1e-10
    linear = 2.0 * surface.parameters[:, 0] - 3.0 * surface.parameters[:, 1]
    interior = ~surface.mesh.boundary_vertices
    assert np.allclose((laplacian @ linear)[interior], 0.0, atol=1e-12)


def test_non_orientable_meshes_are_admitted():
    nu, nv, width = 24, 3, 0.3
    u, v = np.meshgrid(np.arange(nu) * 2 * math.pi / nu, np.linspace(-width, width, nv + 1), indexing="ij")
    u, v = u.ravel(), v.ravel()
    vertices = np.column_stack([(1 + v * np.cos(u / 2)) * np.cos(u), (1 + v * np.cos(u / 2)) * np.sin(u),
                                v * np.sin(u / 2)])
    index = lambda i, j: i * (nv + 1) + j
    faces = []
    for i in range(nu):
        for j in range(nv):
            b, c = ((index(i + 1, j), index(i + 1, j + 1)) if i < nu - 1 else (index(0, nv - j), index(0, nv - j - 1)))
            faces += [[index(i, j), b, c], [index(i, j), c, index(i, j + 1)]]
    band = m.TriangleMesh(vertices, faces)
    distance = m.heat_method_distance(band, index(0, 1))
    assert np.isfinite(distance).all() and distance.max() > 1.0


def test_references_and_comparison():
    refused("no_closed_form_reference", m.torus_mesh(2.0, 0.5, 8, 6).reference_distance, 0)
    refused("no_closed_form_reference", m.icosphere(1.0, 1).reference_distance, 0, "polyhedral")
    refused("unknown_reference_model", m.plane_patch(1.0, 1.0, 2, 2).reference_distance, 0, "chord")
    sphere = m.icosphere(1.0, 1)
    north = int(np.argmax(sphere.mesh.vertices[:, 2]))
    assert sphere.reference_distance(north).max() == pytest.approx(math.pi)
    stats = m.compare_to_reference([0.0, 1.1, 1.8], [0.0, 1.0, 2.0])
    assert stats["max_abs_error"] == pytest.approx(0.2) and stats["relative_count"] == 2
    assert stats["max_rel_error"] == pytest.approx(0.1) and stats["rms_rel_error"] == pytest.approx(0.1)
    assert stats["mean_abs_error"] == pytest.approx(0.1)
    masked = m.compare_to_reference([0.0, 1.1, math.inf], [0.0, 1.0, 2.0], mask=np.array([True, True, False]))
    assert masked["count"] == 2 and masked["max_abs_error"] == pytest.approx(0.1)
    refused("non_finite_distance", m.compare_to_reference, [0.0, math.inf], [0.0, 1.0])
    refused("negative_reference", m.compare_to_reference, [0.0, 1.0], [0.0, -1.0])
    refused("malformed_distances", m.compare_to_reference, [0.0, 1.0], [0.0, 1.0, 2.0])
    assert m.compare_to_reference([0.0], [0.0])["max_rel_error"] is None
