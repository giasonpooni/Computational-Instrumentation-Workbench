"""Offline checks for the bounded mesh-edge record, without a provider import.

These checks establish internal consistency of retained declarations and path
evidence. They do not authenticate an execution, provenance, physical accuracy,
or certified floating-point bounds. No shortest-path solver is executed here.
"""
from copy import deepcopy
from fractions import Fraction
import math

from .core.canonical import canonical, digest

BUDGET = {"discretization": "not_estimated", "scan_noise": "not_established", "calibration": "not_established"}
AUTHORITY = {"physical_accuracy": "not_established", "calibration": "not_performed",
             "state_estimation": "not_performed", "continuous_surface_solver": "not_implemented"}
CLAIM = "edge_constrained_upper_bound_on_declared_piecewise_flat_mesh"


def _keys(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Unexpected mesh contract fields")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Mesh numerical evidence requires finite numbers, not Booleans")
    return value


def _text(value):
    if type(value) is not str or not value.strip() or len(value) > 512:
        raise ValueError("Require a nonempty mesh declaration of at most 512 characters")


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Retained mesh declaration or identity differs")


def _near(actual, expected, operations=1):
    """Allow bounded binary64 accumulation differences, not geometric error."""
    _number(actual)
    _number(expected)
    scale = max(abs(actual), abs(expected))
    budget = 64 * max(1, operations) * math.ulp(scale)
    if abs(actual - expected) > budget:
        raise ValueError("Retained mesh numerical evidence is inconsistent")


def _geometry(mesh, minimum_quality):
    vertices = [tuple(map(float, point)) for point in mesh["vertices"]]
    if len(set(vertices)) != len(vertices):
        raise ValueError("Coincident mesh vertices are unsupported")
    adjacency = [dict() for _ in vertices]
    links = [dict() for _ in vertices]
    edges, seen, qualities = {}, set(), []
    for face in mesh["triangles"]:
        key = tuple(sorted(face))
        if key in seen:
            raise ValueError("Duplicate mesh triangle")
        seen.add(key)
        a, b, c = (vertices[i] for i in face)
        u = [Fraction.from_float(b[i]) - Fraction.from_float(a[i]) for i in range(3)]
        v = [Fraction.from_float(c[i]) - Fraction.from_float(a[i]) for i in range(3)]
        if all(u[i]*v[j] == u[j]*v[i] for i,j in ((0,1), (0,2), (1,2))):
            raise ValueError("Degenerate mesh triangle")
        lengths = [math.dist(a,b), math.dist(b,c), math.dist(c,a)]
        if min(lengths) <= 0:
            raise ValueError("Zero-length mesh edge")
        longest = max(lengths)
        u = [(b[i]-a[i])/longest for i in range(3)]
        v = [(c[i]-a[i])/longest for i in range(3)]
        quality = math.hypot(u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if quality <= minimum_quality:
            raise ValueError("Mesh triangle is below its quality threshold")
        qualities.append(quality)
        for i,j in zip(face, face[1:]+face[:1]):
            edges.setdefault(tuple(sorted((i,j))), []).append((i,j))
            adjacency[i][j] = adjacency[j][i] = math.dist(vertices[i], vertices[j])
        for position, vertex in enumerate(face):
            x,y = face[(position+1)%3], face[(position+2)%3]
            links[vertex].setdefault(x,set()).add(y)
            links[vertex].setdefault(y,set()).add(x)
    if any(len(incident) > 2 for incident in edges.values()):
        raise ValueError("Nonmanifold mesh edge")
    if any(not neighbors for neighbors in adjacency):
        raise ValueError("Isolated mesh vertex")
    for link in links:
        pending, reached = [min(link)], set()
        while pending:
            vertex = pending.pop()
            if vertex not in reached:
                reached.add(vertex)
                pending.extend(link[vertex] - reached)
        degrees = [len(neighbors) for neighbors in link.values()]
        if len(reached) != len(link) or any(d not in (1,2) for d in degrees) or degrees.count(1) not in (0,2):
            raise ValueError("Nonmanifold mesh vertex link")
    labels, components = [-1]*len(vertices), 0
    for vertex in range(len(vertices)):
        if labels[vertex] >= 0:
            continue
        labels[vertex] = components
        pending = [vertex]
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if labels[neighbor] < 0:
                    labels[neighbor] = components
                    pending.append(neighbor)
        components += 1
    boundary = sum(len(incident) == 1 for incident in edges.values())
    orientation = sum(len(incident) == 2 and incident[0] == incident[1] for incident in edges.values())
    quality = {
        "vertex_count":len(vertices), "triangle_count":len(mesh["triangles"]), "edge_count":len(edges),
        "component_count":components, "vertex_components":labels, "boundary_edge_count":boundary,
        "orientation_conflict_count":orientation, "minimum_triangle_quality":min(qualities),
        "triangle_quality_definition":"twice_area_over_longest_edge_squared", "degenerate_triangle_count":0,
        "duplicate_triangle_count":0, "coincident_vertex_count":0, "nonmanifold_edge_count":0,
        "nonmanifold_vertex_count":0, "isolated_vertex_count":0, "self_intersections":"not_checked",
        "repair":"not_performed", "flags":(["boundary"] if boundary else []) +
        (["disconnected"] if components > 1 else []) + (["inconsistent_face_orientation"] if orientation else []),
    }
    return vertices, adjacency, quality


def _request(request):
    if len(canonical(request)) > 128*1024:
        raise ValueError("Mesh request exceeds its byte budget")
    _keys(request, {"schema", "mesh", "source_vertex", "target_vertex", "settings", "tolerance_budget"})
    if request["schema"] != "isgt.edge-geodesic-request.v1":
        raise ValueError("Unsupported mesh request schema")
    mesh = request["mesh"]
    _keys(mesh, {"vertices", "triangles", "units", "coordinate_frame", "provenance", "mesh_digest"})
    vertices, triangles = mesh["vertices"], mesh["triangles"]
    if type(vertices) is not list or not 3 <= len(vertices) <= 256:
        raise ValueError("Require 3..256 mesh vertices")
    for point in vertices:
        if type(point) is not list or len(point) != 3:
            raise ValueError("Mesh vertices require three coordinates")
        if any(abs(_number(value)) > 1e6 for value in point):
            raise ValueError("Mesh coordinate magnitude exceeds 1e6")
    if type(triangles) is not list or not 1 <= len(triangles) <= 512:
        raise ValueError("Require 1..512 mesh triangles")
    for face in triangles:
        if (type(face) is not list or len(face) != 3 or
            any(type(i) is not int or not 0 <= i < len(vertices) for i in face) or len(set(face)) != 3):
            raise ValueError("Require three distinct bounded integer triangle indices")
    for field in ("source_vertex", "target_vertex"):
        if type(request[field]) is not int or not 0 <= request[field] < len(vertices):
            raise ValueError("Invalid mesh endpoint")
    if mesh["units"] not in ("m", "mm", "normalized_length"):
        raise ValueError("Unsupported mesh length unit")
    _text(mesh["coordinate_frame"])
    _keys(mesh["provenance"], {"source", "preprocessing"})
    _text(mesh["provenance"]["source"])
    operations = mesh["provenance"]["preprocessing"]
    if type(operations) is not list or len(operations) > 32:
        raise ValueError("Require at most 32 preprocessing descriptions")
    for operation in operations:
        _text(operation)
    if mesh["mesh_digest"] != digest({k:v for k,v in mesh.items() if k != "mesh_digest"}):
        raise ValueError("Mesh digest differs from its declaration")
    settings = request["settings"]
    _keys(settings, {"algorithm", "minimum_triangle_quality", "distance_absolute_tolerance"})
    if settings["algorithm"] != "edge_dijkstra":
        raise ValueError("Only the declared edge algorithm is supported")
    if not 0 <= _number(settings["minimum_triangle_quality"]) <= .01:
        raise ValueError("Invalid mesh quality threshold")
    if not 0 <= _number(settings["distance_absolute_tolerance"]) <= .001:
        raise ValueError("Invalid mesh distance tolerance")
    _same(request["tolerance_budget"], BUDGET)
    return _geometry(mesh, settings["minimum_triangle_quality"])


def validate_request(request):
    """Return a detached, unchanged supported declaration or raise ValueError."""
    try:
        _request(request)
        return deepcopy(request)
    except (TypeError, KeyError, IndexError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Malformed bounded mesh declaration") from exc


def validate_result(request, data):
    """Check retained path/bound/quality consistency; never execute a provider."""
    try:
        _result(request, data)
    except (TypeError, KeyError, IndexError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("Malformed retained mesh result") from exc


def _result(request, data):
    vertices, adjacency, quality = _request(request)
    _keys(data, {"schema", "algorithm", "claim_scope", "request", "request_digest", "mesh_digest", "mesh_quality",
                "solution", "bounds", "tolerance_budget", "authority", "artifact_digest"})
    if (data["schema"] != "isgt.edge-geodesic-result.v1" or data["algorithm"] != "edge_dijkstra.v1" or
        data["claim_scope"] != CLAIM or data["mesh_digest"] != request["mesh"]["mesh_digest"] or
        data["request_digest"] != digest(request) or
        data["artifact_digest"] != digest({k:v for k,v in data.items() if k != "artifact_digest"})):
        raise ValueError("Mesh profile or content identity differs")
    _same(data["request"], request)
    _same(data["tolerance_budget"], BUDGET)
    _same(data["authority"], AUTHORITY)
    _keys(data["mesh_quality"], set(quality))
    actual_quality = data["mesh_quality"]
    _near(actual_quality["minimum_triangle_quality"], quality["minimum_triangle_quality"])
    if actual_quality["minimum_triangle_quality"] <= request["settings"]["minimum_triangle_quality"]:
        raise ValueError("Retained triangle quality violates the declaration")
    _same({k:v for k,v in actual_quality.items() if k != "minimum_triangle_quality"},
          {k:v for k,v in quality.items() if k != "minimum_triangle_quality"})
    solution = data["solution"]
    _keys(solution, {"source_vertex", "target_vertex", "vertex_distances", "target_path", "target_distance", "reachable"})
    source, target, count = request["source_vertex"], request["target_vertex"], len(vertices)
    for field in ("source_vertex", "target_vertex"):
        if type(solution[field]) is not int or solution[field] != request[field]:
            raise ValueError("Mesh endpoint differs")
    distances, path = solution["vertex_distances"], solution["target_path"]
    if type(distances) is not list or len(distances) != count:
        raise ValueError("Mesh distance count differs")
    components = quality["vertex_components"]
    for vertex, distance in enumerate(distances):
        if components[vertex] == components[source]:
            if _number(distance) < 0:
                raise ValueError("Reachable distance must be nonnegative")
        elif distance is not None:
            raise ValueError("Unreachable vertex requires a null distance")
    if distances[source] != 0:
        raise ValueError("Mesh source must have zero distance")
    reachable = components[source] == components[target]
    if type(solution["reachable"]) is not bool or solution["reachable"] != reachable:
        raise ValueError("Reachability differs from declared topology")
    if (type(path) is not list or len(path) > count or
        any(type(v) is not int or not 0 <= v < count for v in path) or len(set(path)) != len(path)):
        raise ValueError("Require a simple bounded mesh edge path")
    if reachable:
        _number(solution["target_distance"])
        if solution["target_distance"] != distances[target] or not path or path[0] != source or path[-1] != target:
            raise ValueError("Path endpoint or distance binding differs")
        if any(b not in adjacency[a] for a,b in zip(path,path[1:])):
            raise ValueError("Retained path leaves the declared edge graph")
        _near(distances[target], math.fsum(adjacency[a][b] for a,b in zip(path,path[1:])), count)
    elif path or solution["target_distance"] is not None:
        raise ValueError("Unreachable endpoint cannot contain a finite path")
    # Every retained finite distance must be a potential on the edge graph and
    # have a tight-edge predecessor chain to the source. A potential alone is
    # insufficient: a non-target distance could otherwise be underestimated.
    # Equal-distance plateaus can arise when adding a positive edge rounds to
    # the same binary64 value, so certify decreasing hop rank to the source
    # instead of demanding a strictly decreasing floating value at every hop.
    tight_successors = [[] for _ in vertices]
    for a, neighbors in enumerate(adjacency):
        if distances[a] is None:
            continue
        for b, weight in neighbors.items():
            difference = abs(distances[a] - distances[b])
            roundoff = 64 * count * math.ulp(max(distances[a], distances[b], weight))
            if difference > weight + roundoff:
                raise ValueError("Retained distances violate an edge inequality")
            if distances[a] <= distances[b] and abs(distances[b] - (distances[a] + weight)) <= roundoff:
                tight_successors[a].append(b)
    witnessed, pending = {source}, [source]
    while pending:
        for vertex in tight_successors[pending.pop()]:
            if vertex not in witnessed:
                witnessed.add(vertex)
                pending.append(vertex)
    if witnessed != {vertex for vertex, distance in enumerate(distances) if distance is not None}:
        raise ValueError("Every reachable distance requires a tight predecessor chain to the source")
    bounds = data["bounds"]
    _keys(bounds, {"euclidean_lower_bounds", "target_lower_bound", "target_upper_bound", "target_gap",
                   "triangle_inequality_residual_max", "roundoff_certification"})
    lower = bounds["euclidean_lower_bounds"]
    if type(lower) is not list or len(lower) != count or bounds["roundoff_certification"] != "not_established":
        raise ValueError("Invalid mesh lower-bound shape or certification claim")
    for vertex, bound in enumerate(lower):
        if _number(bound) < 0:
            raise ValueError("Mesh lower bound must be nonnegative")
        _near(bound, math.dist(vertices[source], vertices[vertex]))
    if _number(bounds["target_lower_bound"]) != lower[target]:
        raise ValueError("Target lower-bound binding differs")
    residual = max([0.0] + [low-distance for low,distance in zip(lower, distances) if distance is not None])
    reported = _number(bounds["triangle_inequality_residual_max"])
    if not 0 <= reported <= request["settings"]["distance_absolute_tolerance"]:
        raise ValueError("Mesh inequality residual exceeds its numerical budget")
    _near(reported, residual, count)
    if reachable:
        if _number(bounds["target_upper_bound"]) != distances[target]:
            raise ValueError("Target upper-bound binding differs")
        if _number(bounds["target_gap"]) < 0:
            raise ValueError("Mesh numerical gap must be nonnegative")
        _near(bounds["target_gap"], max(0.0, distances[target]-lower[target]), count)
    elif bounds["target_upper_bound"] is not None or bounds["target_gap"] is not None:
        raise ValueError("Unreachable upper bound and gap must be null")
