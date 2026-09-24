"""Analytical offline fixtures; these tests make no provider execution claim."""
from copy import deepcopy
import math

import pytest

from ciw.geometry_mesh_contract import AUTHORITY, BUDGET, CLAIM, validate_request, validate_result
from ciw.core.canonical import canonical, digest


def request():
    value = {
        "schema":"isgt.edge-geodesic-request.v1",
        "mesh":{"vertices":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]], "triangles":[[0,1,3],[1,2,3]],
                "units":"m", "coordinate_frame":"declared Cartesian axes",
                "provenance":{"source":"analytical unit square", "preprocessing":[]}},
        "source_vertex":0, "target_vertex":2,
        "settings":{"algorithm":"edge_dijkstra", "minimum_triangle_quality":0,
                    "distance_absolute_tolerance":1e-12}, "tolerance_budget":deepcopy(BUDGET),
    }
    reseal_request(value)
    return value


def reseal_request(value):
    mesh = value["mesh"]
    mesh["mesh_digest"] = digest({k:v for k,v in mesh.items() if k != "mesh_digest"})


def reseal_result(value):
    value["request_digest"] = digest(value["request"])
    value["mesh_digest"] = value["request"]["mesh"]["mesh_digest"]
    value["artifact_digest"] = digest({k:v for k,v in value.items() if k != "artifact_digest"})


def result():
    req = request()
    value = {
        "schema":"isgt.edge-geodesic-result.v1", "algorithm":"edge_dijkstra.v1", "claim_scope":CLAIM,
        "request":req, "request_digest":digest(req), "mesh_digest":req["mesh"]["mesh_digest"],
        "mesh_quality":{"vertex_count":4, "triangle_count":2, "edge_count":5, "component_count":1,
            "vertex_components":[0,0,0,0], "boundary_edge_count":4, "orientation_conflict_count":0,
            "minimum_triangle_quality":0.5, "triangle_quality_definition":"twice_area_over_longest_edge_squared",
            "degenerate_triangle_count":0, "duplicate_triangle_count":0, "coincident_vertex_count":0,
            "nonmanifold_edge_count":0, "nonmanifold_vertex_count":0, "isolated_vertex_count":0,
            "self_intersections":"not_checked", "repair":"not_performed", "flags":["boundary"]},
        "solution":{"source_vertex":0, "target_vertex":2, "vertex_distances":[0.0,1.0,2.0,1.0],
            "target_path":[0,1,2], "target_distance":2.0, "reachable":True},
        "bounds":{"euclidean_lower_bounds":[0.0,1.0,math.sqrt(2),1.0], "target_lower_bound":math.sqrt(2),
            "target_upper_bound":2.0, "target_gap":2-math.sqrt(2), "triangle_inequality_residual_max":0.0,
            "roundoff_certification":"not_established"},
        "tolerance_budget":deepcopy(BUDGET), "authority":deepcopy(AUTHORITY),
    }
    reseal_result(value)
    return value


def test_analytical_square_and_detached_request():
    data = result()
    saved = deepcopy(data)
    assert validate_result(data["request"],data) is None
    detached = validate_request(data["request"])
    assert canonical(detached) == canonical(data["request"])
    detached["mesh"]["vertices"][0][0] = 99
    assert data == saved


@pytest.mark.parametrize("mutation", [
    lambda r:r["settings"].update(algorithm="continuous_exact"),
    lambda r:r["settings"].update(extra="ignored"),
    lambda r:r["settings"].update(minimum_triangle_quality=True),
    lambda r:r["settings"].update(minimum_triangle_quality=-1),
    lambda r:r["settings"].update(distance_absolute_tolerance=.002),
    lambda r:r["tolerance_budget"].update(calibration="established"),
    lambda r:r["mesh"]["provenance"].update(preprocessing="none"),
    lambda r:r["mesh"]["provenance"].update(source=""),
    lambda r:r["mesh"]["provenance"].update(preprocessing=["step"]*33),
    lambda r:r["mesh"]["vertices"][0].__setitem__(0,1e6+1),
    lambda r:r["mesh"]["vertices"][0].__setitem__(0,True),
    lambda r:r["mesh"].update(units="radians"),
    lambda r:r.update(source_vertex=True),
])
def test_request_profile_refuses_resealed_unsupported_declarations(mutation):
    value = request()
    mutation(value)
    reseal_request(value)
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize("vertices,triangles", [
    ([[0,0,0],[1,3,7],[3,9,21]], [[0,1,2]]),
    ([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2],[2,1,0]]),
    ([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], [[0,1,2],[1,0,3],[0,1,4]]),
    ([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], [[0,1,2],[0,3,4]]),
    ([[0,0,0],[1,0,0],[0,1,0],[2,2,2]], [[0,1,2]]),
    ([[0,0,0],[1,0,0],[0,0,0]], [[0,1,2]]),
])
def test_invalid_topology_and_exact_degeneracy_are_refused(vertices, triangles):
    value = request()
    value["mesh"].update(vertices=vertices,triangles=triangles)
    reseal_request(value)
    with pytest.raises(ValueError):
        validate_request(value)


@pytest.mark.parametrize("mutation", [
    lambda d:d["bounds"].update(target_gap="certified_zero"),
    lambda d:d["bounds"].update(target_gap=True),
    lambda d:d["bounds"].update(target_gap=-1),
    lambda d:d["bounds"].update(target_gap=0),
    lambda d:d["bounds"].update(triangle_inequality_residual_max=-1),
    lambda d:d["bounds"].update(triangle_inequality_residual_max=1e-6),
    lambda d:d["bounds"].update(target_upper_bound=1),
    lambda d:d["bounds"].update(target_lower_bound=True),
    lambda d:d["bounds"]["euclidean_lower_bounds"].__setitem__(1,0),
    lambda d:d["bounds"].update(roundoff_certification="certified"),
    lambda d:d.update(mesh_quality={}),
    lambda d:d["mesh_quality"].update(component_count=True),
    lambda d:d["mesh_quality"].update(vertex_components=[0,0,1,0]),
    lambda d:d["mesh_quality"].update(self_intersections="checked"),
    lambda d:d["mesh_quality"].update(repair="projected"),
    lambda d:d["mesh_quality"].update(minimum_triangle_quality=.1),
    lambda d:d["mesh_quality"].update(flags=[]),
    lambda d:d["solution"].update(target_path=[0,2]),
    lambda d:d["solution"].update(target_path=[0,1,0,2]),
    lambda d:d["solution"].update(target_distance=True),
    lambda d:d["solution"].update(reachable=1),
    lambda d:d["solution"]["vertex_distances"].__setitem__(1,None),
    lambda d:d["solution"]["vertex_distances"].__setitem__(1,100),
    lambda d:d["authority"].update(physical_accuracy="established"),
])
def test_resealed_bad_path_bound_and_quality_evidence_is_refused(mutation):
    value = result()
    mutation(value)
    reseal_result(value)
    with pytest.raises(ValueError):
        validate_result(value["request"],value)


def test_resealed_shorter_distance_cannot_name_a_longer_path():
    value = result()
    value["solution"]["target_distance"] = .1
    value["solution"]["vertex_distances"][2] = .1
    value["bounds"]["target_upper_bound"] = .1
    reseal_result(value)
    with pytest.raises(ValueError):
        validate_result(value["request"],value)


def test_nontarget_distance_requires_a_predecessor_witness():
    value = result()
    value["request"]["target_vertex"] = 1
    value["solution"].update(target_vertex=1,target_path=[0,1],target_distance=1.0)
    value["bounds"].update(target_lower_bound=1.0,target_upper_bound=1.0,target_gap=0.0)
    reseal_result(value)
    validate_result(value["request"],value)
    # This remains above the chord, satisfies every edge inequality, and leaves
    # the target path unchanged, but vertex 2 has mesh-edge distance 2, not sqrt2.
    value["solution"]["vertex_distances"][2] = math.sqrt(2)
    reseal_result(value)
    with pytest.raises(ValueError, match="predecessor chain"):
        validate_result(value["request"],value)


def test_tight_predecessor_chain_accepts_binary64_distance_plateaus():
    value = result()
    value["request"]["mesh"].update(vertices=[[0,0,0],[1,0,0],[1,1e-20,0],[1,1e-20,1e-20]],
                                    triangles=[[0,1,2],[1,3,2]])
    value["request"]["target_vertex"] = 1
    reseal_request(value["request"])
    value["mesh_quality"]["minimum_triangle_quality"] = 1e-20
    value["solution"].update(target_vertex=1,target_path=[0,1],target_distance=1.0,
                             vertex_distances=[0.0,1.0,1.0,1.0])
    value["bounds"].update(euclidean_lower_bounds=[0.0,1.0,1.0,1.0],
                           target_lower_bound=1.0,target_upper_bound=1.0,target_gap=0.0)
    reseal_result(value)
    # Vertex 3 has no source edge: its positive final edge is smaller than one
    # ulp of the accumulated distance, so its valid predecessor value is equal.
    validate_result(value["request"],value)


def test_reachable_topology_cannot_be_reported_unreachable():
    value = result()
    value["solution"].update(reachable=False,target_path=[],target_distance=None)
    value["solution"]["vertex_distances"][2] = None
    value["bounds"].update(target_upper_bound=None,target_gap=None)
    reseal_result(value)
    with pytest.raises(ValueError):
        validate_result(value["request"],value)


def test_source_equals_target_has_zero_bound_and_single_vertex_path():
    value = result()
    value["request"]["target_vertex"] = 0
    value["solution"].update(target_vertex=0,target_path=[0],target_distance=0.0)
    value["bounds"].update(target_lower_bound=0.0,target_upper_bound=0.0,target_gap=0.0)
    reseal_result(value)
    validate_result(value["request"],value)


def test_disconnected_mesh_requires_null_upper_bound_gap_and_component_distances():
    value = result()
    value["request"]["mesh"].update(vertices=[[0,0,0],[1,0,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]],
                                   triangles=[[0,1,2],[3,4,5]])
    value["request"]["target_vertex"] = 5
    reseal_request(value["request"])
    value["mesh_quality"].update(vertex_count=6,edge_count=6,component_count=2,vertex_components=[0,0,0,1,1,1],
                                 boundary_edge_count=6,flags=["boundary","disconnected"])
    value["solution"].update(target_vertex=5,vertex_distances=[0.0,1.0,1.0,None,None,None],
                             target_path=[],target_distance=None,reachable=False)
    value["bounds"].update(euclidean_lower_bounds=[0.0,1.0,1.0,3.0,4.0,math.sqrt(10)],
                           target_lower_bound=math.sqrt(10),target_upper_bound=None,target_gap=None)
    reseal_result(value)
    validate_result(value["request"],value)
    for field in ("target_upper_bound","target_gap"):
        changed = deepcopy(value)
        changed["bounds"][field] = 0
        reseal_result(changed)
        with pytest.raises(ValueError):
            validate_result(changed["request"],changed)


def test_request_identity_preserves_integer_and_float_distinction():
    value = result()
    different = deepcopy(value["request"])
    different["settings"]["minimum_triangle_quality"] = 0.0
    with pytest.raises(ValueError):
        validate_result(different,value)


@pytest.mark.parametrize("number", [float("nan"),float("inf"),10**400])
def test_nonfinite_and_unrepresentable_inputs_refuse_cleanly(number):
    value = request()
    value["mesh"]["vertices"][0][0] = number
    with pytest.raises(ValueError):
        validate_request(value)
