import copy
import importlib.metadata
import importlib.util
import math

import numpy as np
import pytest

from ciw.lab import runner
from ciw.lab import surfaces_discrete_mesh as M
from ciw.lab import surfaces_discrete_mesh_exact as E
from ciw.lab import surfaces_discrete_mesh_geometry as G
from ciw.lab import surfaces_discrete_mesh_studies as S
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, validate_finding
from ciw.lab.registry import load_queue, module_implementations
from ciw.lab.report import validate_report

TASKS = ("T038", "T039", "T040", "T041", "T042", "T043", "T044")
IMPLEMENTATIONS = module_implementations("surfaces_discrete_mesh")
DIJKSTRA_CLAIM = ("Heap Dijkstra edge-graph distances agree with a dense Floyd-Warshall recomputation (and "
                  "scipy.sparse.csgraph when installed)")
PYGEODESIC_CLAIM = ("Exact polyhedral distances from three sources are symmetric between the sources (and agree at "
                    "every vertex with pygeodesic's exact MMP implementation when installed) on icosphere levels 1-4 "
                    "and a torus with saddle vertices")
FLIPOUT_CLAIM = ("FlipOut edge-flip geodesics between vertex pairs (potpourri3d; without it, the edge-graph paths FlipOut "
                 "starts from) are never shorter than the exact distance")
FAN_CLAIM = ("Through one vertex of total angle theta the Polthier-Schmies continuation ends at polar angle theta / 2 "
             "on both sides, where the exact distance from the start takes its closed form (and agrees with "
             "pygeodesic's exact MMP when installed), so the continued geodesic is shortest exactly when theta is at "
             "least 2 pi")
PATH_CLAIM = ("Shortest paths back-traced from the exact solver's windows have the exact length, lie on the surface, "
              "run straight across every edge and bend only at saddle and reflex boundary vertices (and match "
              "pygeodesic's paths when installed)")
CUT_CLAIM = ("The first cut point of every declared straightest geodesic, where its length first exceeds the exact "
             "distance from its start, comes no later than the smallest isolated-cone prediction r sin(delta / 2) / "
             "sin(delta / 2 - phi) over the vertices it passes, and equals it when the digon between the trace and the "
             "other shortest path encloses one vertex")
# The memoized studies T038 reads (surfaces_discrete_mesh:<name>).
T038_STUDIES = ("sphere-traces", "cylinder", "plane", "steiner-nested", "dijkstra-independent", "exact-distances",
                "exact-independent", "exact-comparison", "traced-exact", "exact-analytic", "exact-insertion",
                "continuation", "fans", "vertex-hits", "paths", "continuation-independent")
RANK_CLAIM = ("Maximum radius ratio is positively rank-correlated with the mixed-Voronoi curvature error and the "
              "geodesic error across the pooled valid 642-vertex meshes, but not within the latitude-longitude family")


def _run(task_id, ctx):
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    return validate_report(runner.run_task(queue[task_id], IMPLEMENTATIONS[task_id], ctx, {}))


@pytest.fixture(scope="module")
def mesh_ctx(tmp_path_factory):
    """The shared context whose memoized studies the section run computes."""
    return runner.Context(tmp_path_factory.mktemp("lab-mesh"))


@pytest.fixture(scope="module")
def reports(mesh_ctx):
    """Run the section once through the runner with one shared context (memoized studies)."""
    return {tid: _run(tid, mesh_ctx) for tid in TASKS}


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _value(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def _completed_with_unestablished_physics(report, state="completed"):
    assert report["state"] == state
    assert report["physical_validation_status"]["status"] == "not_established"
    assert report["evidence_status"]["primary"] in ("numerically_verified", "independently_verified")
    for record in report["findings"]:
        validate_finding(record)
        if record["domain"] not in COMPUTATIONAL_DOMAINS:
            assert record["evidence_status"] == "not_established"
            continue
        assert record["evidence_status"] in ("numerically_verified", "independently_verified"), record["claim"]
        assert record["uncertainty"] is not None, record["claim"]
        assert "regression_tolerance" in record, record["claim"]
        for check in record["basis"].get("checks", []):
            assert check["reference_kind"] in ("analytic", "invariant", "self_convergence", "exact_arithmetic",
                                               "refusal", "cross_implementation")


def test_every_task_is_registered_in_this_module():
    assert sorted(IMPLEMENTATIONS) == list(TASKS)


def test_primary_labels(reports):
    for task_id, report in reports.items():
        assert report["evidence_status"]["primary"] == "numerically_verified", task_id


# ---------------------------------------------------------------- T038
@pytest.mark.lab_task("T038")
def test_tracer_is_exact_on_developable_meshes():
    plane = S.plane_study(shears=(0.0, 1.5), size=4, length=0.2)
    assert all(r["statuses"] == ["completed"] * 4 for r in plane["rows"])
    assert max(r["max_trace_error"] for r in plane["rows"]) <= 1e-12
    cylinder = S.cylinder_study(ns=(8, 32))
    assert max(r["development_error"] for r in cylinder["rows"]) <= 1e-11
    assert max(r["z_error"] for r in cylinder["rows"]) <= 1e-12
    traces = S.sphere_trace_study(levels=(2,), starts=S.STARTS[:2])
    assert traces["rows"][0]["completed"] == 2
    assert traces["rows"][0]["max_unfold_minus_trace"] <= 1e-11


@pytest.mark.lab_task("T038")
def test_graph_distances_and_steiner_sandwich():
    nested = S.steiner_nested_study(level=1, ks=(0, 1, 3), starts=S.STARTS[:3])
    assert nested["max_increase_with_k"] <= 1e-12
    assert nested["edges_outside_faces"] == 0 and nested["edges_checked"] > 1000
    assert nested["min_gap"] >= -1e-12
    assert np.mean(nested["gap_to_traced"]["3"]) <= np.mean(nested["gap_to_traced"]["1"])
    independent = S.dijkstra_independent(level=2)
    assert independent["floyd_max_abs"] <= 1e-12
    if independent["scipy"]:
        assert independent["scipy_max_abs"] <= 1e-12
    # The face test can fail: an edge between Steiner nodes of two faces that share no edge is detected.
    mesh = G.icosphere(1)
    (indptr, indices, _), _, _ = G.steiner_graph(mesh, 1)
    nodes = G.steiner_nodes(mesh, 1)
    rows, cols = S._graph_edges(indptr, indices)
    assert S.edges_outside_faces(mesh, nodes, rows, cols) == 0
    far = int(np.argmax(np.linalg.norm(nodes - nodes[0], axis=1)))
    assert S.edges_outside_faces(mesh, nodes, np.r_[rows, 0], np.r_[cols, far]) == 1


@pytest.mark.lab_task("T038")
def test_solver_task_report(reports):
    report = reports["T038"]
    # Completed: the exact polyhedral distance is delivered and every computational finding is established.
    _completed_with_unestablished_physics(report)
    # Back-tracing and the vertex rule are delivered; what stays open is named.
    assumptions = " ".join(report["unresolved_assumptions"])
    assert "back-tracing the path" not in assumptions and "refused rather than continued" not in assumptions
    assert "several vertices" in assumptions and "ties are not enumerated" in assumptions
    steiner = _value(report, "Nested Steiner-graph distances never increase with k")
    assert steiner["value"]["edges_outside_faces"] == 0
    assert "chord" not in " ".join(c["reference"] for c in steiner["basis"]["checks"])
    labels = _labels(report)
    scipy_present = S.optional_version("scipy") is not None
    assert labels[DIJKSTRA_CLAIM] == ("independently_verified" if scipy_present else "numerically_verified")
    assert labels[PYGEODESIC_CLAIM] == ("independently_verified" if S.package_version("pygeodesic")
                                        else "numerically_verified")
    assert labels[FLIPOUT_CLAIM] == ("independently_verified" if S.package_version("potpourri3d")
                                     else "numerically_verified")
    floyd = _value(report, "Heap Dijkstra")["basis"]["checks"][0]
    assert floyd["reference_kind"] == "cross_implementation"
    plane = "Straightest geodesics on sheared planar meshes coincide with straight lines"
    assert labels[plane] == "numerically_verified"
    for prefix in ("Exact polyhedral distances on sheared planar", "On an L-shaped planar mesh",
                   "Exact polyhedral distances on prism-cylinder", "Exact distances between the corners of a refined"):
        record = _value(report, prefix)
        assert [c["reference_kind"] for c in record["basis"]["checks"]] == ["analytic"], prefix
        assert record["evidence_status"] == "numerically_verified"
    cube = _value(report, "Exact distances between the corners of a refined")["value"]
    assert cube["corner_distances"][-1] == pytest.approx(math.sqrt(5), abs=1e-12)
    assert cube["edge_graph_far_corner"] > cube["corner_distances"][-1] + 0.1  # the edge graph misses the unfolding
    # Traced geodesics below pi that are not shortest paths: a counterexample on every level, shrinking.
    traced = _value(report, "Traced straightest geodesics are never shorter than the exact distance")
    assert traced["counterexample"]["witness"]["excess"] > 1e-6
    assert all(s < t for s, t in zip(traced["value"]["shortest"], traced["value"]["traces"]))
    assert traced["value"]["min_excess"] >= -1e-12
    assert traced["value"]["max_excess"] == sorted(traced["value"]["max_excess"], reverse=True)
    heat = _value(report, "The largest heat-method error")["value"]["max_abs_error"]
    assert heat == sorted(heat, reverse=True)
    assert "Complete T038" not in report["recommended_next_task"]
    physical = [f for f in report["findings"] if f["domain"] == "physical"]
    assert physical and physical[0]["evidence_status"] == "not_established"
    for name in ("sphere-traces.json", "exact-distances.json", "vertex-continuation.json", "shortest-paths.json"):
        assert any(a["path"].endswith(name) for a in report["generated_artifacts"]), name
    # Vertex continuation, back-traced paths and cut points.
    external = "independently_verified" if S.package_version("pygeodesic") else "numerically_verified"
    assert labels[FAN_CLAIM] == labels[PATH_CLAIM] == labels[CUT_CLAIM] == external
    for prefix in ("Straightest geodesics continued through flat vertices", "At a cube corner",
                   "At a saddle vertex every end direction", "Generic straightest geodesics on jittered icospheres",
                   "A straightest geodesic can stop being shortest"):
        assert _value(report, prefix)["evidence_status"] == "numerically_verified", prefix
    hits = _value(report, "Generic straightest geodesics on jittered icospheres")["value"]
    assert hits["vertex_hits"] == 0 and sum(hits["crossings"]) > 10000 and hits["expected_hits"] < 1e-3
    limits = _value(report, "At a saddle vertex every end direction")
    assert limits["counterexample"]["statement"].startswith("A straightest geodesic through a vertex is the limit")
    paths = _value(report, PATH_CLAIM)["value"]
    assert paths["bends"]["saddle"] > 0 and paths["bends"]["boundary"] > 0 and paths["max_path_distance"] <= 1e-9
    cut = _value(report, CUT_CLAIM)["value"]
    assert cut["one_vertex_digons"] >= 10 and cut["several_vertex_digons"] >= 1 and cut["cut"] < cut["traces"]
    several = _value(report, "A straightest geodesic can stop being shortest")["counterexample"]["witness"]
    assert len(several["enclosed_vertices"]) >= 2 and several["prediction"] > several["cut_point"] + 0.1


@pytest.mark.lab_task("T038")
def test_ciw_producers_of_t038_independent_checks_carry_a_revision(reports):
    """Every ciw producer of an independent check names the package version and its module digest."""
    from ciw import __version__
    for record in reports["T038"]["findings"]:
        check = record["basis"].get("independent_check")
        if check is None:
            continue
        assert check["producer"]["revision"] == f"ciw {__version__}", record["claim"]
        assert len(check["producer"]["source_sha256"]) == 64, record["claim"]
        assert check["checker"]["revision"] not in ("", "working tree", "unknown"), record["claim"]


def _t038_without(monkeypatch, tmp_path, mesh_ctx, recomputed, absent=()):
    """T038 in a fresh context seeded with the section run's studies except ``recomputed``.

    ``absent`` names version helpers of ``S`` that report their optional packages as not installed.
    """
    for helper in absent:
        monkeypatch.setattr(S, helper, lambda package: None)
    ctx = runner.Context(tmp_path)
    for name in T038_STUDIES:
        if name not in recomputed:
            key = f"surfaces_discrete_mesh:{name}"
            value = mesh_ctx.memo(key, lambda: pytest.fail(f"{key} was not computed by the section run"))
            ctx.memo(key, lambda value=value: value)
    return _run("T038", ctx)


@pytest.mark.lab_task("T038")
def test_dijkstra_check_falls_back_without_scipy(monkeypatch, tmp_path, reports, mesh_ctx):
    monkeypatch.setattr(S, "optional_version", lambda name: None)
    result = S.dijkstra_independent(level=1)
    assert "scipy_max_abs" not in result and result["floyd_max_abs"] <= 1e-12
    report = _t038_without(monkeypatch, tmp_path, mesh_ctx, ("dijkstra-independent",))
    assert _labels(report)[DIJKSTRA_CLAIM] == "numerically_verified"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    # The prose must not change with the optional checker, or the regression gate flags wording.
    assert runner._skeleton(report["numerical_result"]) == runner._skeleton(reports["T038"]["numerical_result"])


@pytest.mark.lab_task("T038")
def test_exact_checks_fall_back_without_external_packages(monkeypatch, tmp_path, reports, mesh_ctx):
    """Without pygeodesic and potpourri3d the same claims rest on same-origin checks; claims and prose are unchanged."""
    report = _t038_without(monkeypatch, tmp_path, mesh_ctx, ("exact-independent", "continuation-independent"),
                           absent=("package_version",))
    assert S.package_version("pygeodesic") is None and S.package_version("potpourri3d") is None
    labels = _labels(report)
    external = (PYGEODESIC_CLAIM, FLIPOUT_CLAIM, FAN_CLAIM, PATH_CLAIM, CUT_CLAIM)
    assert {labels[claim] for claim in external} == {"numerically_verified"}
    for claim in external:
        record = _value(report, claim)
        assert "independent_check" not in record["basis"] and record["basis"]["checks"]
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"
    assert set(labels) == set(_labels(reports["T038"]))
    for field in ("numerical_result", "uncertainty", "unresolved_assumptions"):
        assert runner._skeleton(report[field]) == runner._skeleton(reports["T038"][field]), field
    # Values keep their shape and stay within tolerance, so `ciw lab verify` reports only the label changes, and
    # names the packages behind them.
    for record in reports["T038"]["findings"]:
        tolerance = record.get("regression_tolerance", {"abs": 0.0, "rel": 1e-9})
        assert runner._close(record["value"], _value(report, record["claim"])["value"], tolerance), record["claim"]
    installed = {"provider_runtime_identity": {"implementation": "ciw.lab", "pygeodesic": "0.1.11",
                                               "potpourri3d": "1.4.0"}}
    bare = {"provider_runtime_identity": {"implementation": "ciw.lab"}}
    assert runner._environment_note(installed, bare) == " (optional modules differ: -potpourri3d, -pygeodesic)"
    identity = runner.builtin_identity([])
    for package in ("pygeodesic", "potpourri3d"):  # potpourri3d has no __version__: its distribution names it
        version = importlib.metadata.version(package) if importlib.util.find_spec(package) else None
        assert identity.get(package) == version, package
    # The origins of the two external checkers are recognised, so an installed checker yields independence.
    from ciw.lab.evidence import supported_label
    base = {"reference_kind": "exact_arithmetic", "reference": "r", "observed": 0.0, "tolerance": 1e-12,
            "comparison": "abs_le", "passed": True}
    for checker in ("pygeodesic.geodesic.PyGeodesicAlgorithmExact", "potpourri3d.EdgeFlipGeodesicSolver"):
        independent = dict(base, producer=M._ciw_producer("ciw.lab.surfaces_discrete_mesh_exact.ExactGeodesic",
                                                          M.SOLVER),
                           checker={"implementation": checker, "revision": "0.1"})
        assert supported_label({"independent_check": independent}, "numerical") == "independently_verified"


@pytest.mark.lab_task("T038")
def test_exact_distances_match_closed_forms():
    study = S.exact_analytic_study(shears=(0.0, 1.5), size=4, ns=(8,), height=2.0)
    assert study["plane_max_error"] <= 1e-12 and study["cylinder_max_error"] <= 1e-12
    assert study["l_shape"]["max_error"] <= 1e-12 and study["l_shape"]["bent_targets"] > 0
    cube = study["cube"]
    assert cube["max_error"] <= 1e-12
    assert sorted(cube["closed_forms"]) == pytest.approx([0, 1, 1, 1, math.sqrt(2)] + [math.sqrt(2)] * 2
                                                         + [math.sqrt(5)])
    # The L-shape needs its reflex corner as a pseudo-source: it is the only vertex whose angle sum exceeds flat,
    # and paths bend there. Flat vertices are pseudo-sources too (they only add windows); its convex corners are not.
    shape = S._l_shape(4)
    solver = E.ExactGeodesic(shape)
    assert [shape.vertices[v].tolist() for v in np.flatnonzero(solver.excess > E.ANGLE_TOLERANCE)] == [[0.5, 0.5, 0.0]]
    flat = np.flatnonzero(np.abs(solver.excess) < E.ANGLE_TOLERANCE)
    assert len(flat) and all(solver.pseudo[v] for v in flat)
    assert not any(solver.pseudo[v] for v in np.flatnonzero(solver.excess < -E.ANGLE_TOLERANCE))
    far = int(np.argmin(np.linalg.norm(shape.vertices - [0.5, 1.0, 0.0], axis=1)))
    source = int(np.argmin(np.linalg.norm(shape.vertices - [1.0, 0.0, 0.0], axis=1)))
    assert solver.distances(source)[far] == pytest.approx(math.sqrt(0.5) + 0.5, abs=1e-12)
    # Globally shortest, not locally straight: every edge-graph distance on the cube is an upper bound.
    box = G.cube_mesh(2)
    assert G.inspect(box.vertices, box.faces, require_closed=True, center=np.full(3, 0.5)) == []
    exact = E.ExactGeodesic(box).distances(0)
    assert np.all(G.edge_distances(box, 0) >= exact - 1e-12)


@pytest.mark.lab_task("T038")
def test_point_insertion_leaves_distances_unchanged():
    study = S.insertion_study()
    assert study["max_abs_change"] <= 1e-12
    assert all(r["issues"] == [] and r["faces_added"] == 4 for r in study["rows"])
    mesh = G.icosphere(1)
    corner = mesh.faces[3]
    # A point at a vertex is that vertex; a point on an edge splits both faces at the edge.
    same, ids = E.insert_points(mesh, [(3, mesh.vertices[corner[1]])])
    assert ids == [int(corner[1])] and len(same.faces) == len(mesh.faces)
    edge_point = mesh.point(3, [0.5, 0.5, 0.0])
    split, (new,) = E.insert_points(mesh, [(3, edge_point)])
    assert len(split.faces) == len(mesh.faces) + 2 and np.sum(np.any(split.faces == new, axis=1)) == 4
    assert G.inspect(split.vertices, split.faces, require_closed=True) == []
    before, after = E.ExactGeodesic(mesh).distances(0), E.ExactGeodesic(split).distances(0)
    assert np.max(np.abs(after[:len(mesh.vertices)] - before)) <= 1e-12
    # From the inserted point: symmetric with the distances to it.
    from_point = E.ExactGeodesic(split).distances(new)
    assert from_point[0] == pytest.approx(after[new], abs=1e-12)
    # A point within ON_EDGE of an edge is moved onto it, so the face across keeps its plane.
    for f in range(0, len(mesh.faces), 7):
        near = mesh.point(f, [0.3, 0.7 - 0.5 * E.ON_EDGE, 0.5 * E.ON_EDGE])
        snapped, (new,) = E.insert_points(mesh, [(f, near)])
        assert np.sum(np.any(snapped.faces == new, axis=1)) == 4
        a, b = mesh.vertices[mesh.faces[f, :2]]
        assert np.linalg.norm(np.cross(snapped.vertices[new] - a, b - a)) / np.linalg.norm(b - a) <= 1e-15
        assert np.linalg.norm(snapped.vertices[new] - near) <= E.ON_EDGE
        moved = E.ExactGeodesic(snapped).distances(0)[:len(mesh.vertices)]
        assert np.max(np.abs(moved - before)) <= 1e-12
    with pytest.raises(G.MeshRefusal) as refused:
        E.insert_points(mesh, [(3, 2 * edge_point)])
    assert refused.value.code == "point_outside_face"


@pytest.mark.lab_task("T038")
def test_exact_distances_agree_with_pygeodesic():
    geodesic = pytest.importorskip("pygeodesic.geodesic")
    exact = S.exact_distance_study(levels=(1, 2), torus=(12, 6), flipout=())
    for mesh, row in zip(S.exact_meshes((1, 2), (12, 6)), exact["rows"]):
        algorithm = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces.astype(np.int32))
        for source, distances in zip(row["sources"], row["distances"]):
            reference, _ = algorithm.geodesicDistances(np.array([source], dtype=np.int32), None)
            assert np.max(np.abs(reference - distances)) <= 1e-12, (row["mesh"], source)
    assert exact["rows"][-1]["saddle_vertices"] > 0  # the torus exercises saddle pseudo-sources
    external = S.exact_independent_study(exact)
    assert external["pygeodesic_max_abs"] <= 1e-12
    assert external["pygeodesic_compared"] == sum(3 * r["vertices"] for r in exact["rows"])


@pytest.mark.lab_task("T038")
def test_nearly_flat_saddles_leave_no_shadow():
    """A saddle whose angle excess is below the tolerance is still a pseudo-source, so no vertex behind it is missed."""
    plane = G.plane_mesh(4, 4)
    vertices = plane.vertices.copy()
    vertices[:, 2] = 1e-6 * np.random.default_rng(0).standard_normal(len(vertices))
    assert G.inspect(vertices, plane.faces) == []
    solver = E.ExactGeodesic(G.TriMesh(vertices, plane.faces))
    assert np.any((solver.excess > 0) & (solver.excess < E.ANGLE_TOLERANCE))  # slight saddles
    distances = solver.distances(0)
    assert np.all(np.isfinite(distances))
    assert np.max(np.abs(distances - np.linalg.norm(vertices[:, :2] - vertices[0, :2], axis=1))) <= 1e-9


@pytest.mark.lab_task("T038")
def test_exact_distances_on_a_perturbed_cube_agree_with_pygeodesic():
    """A closed cube with every vertex moved by about 1e-6 from the centre: slight saddles and cones, no boundary."""
    geodesic = pytest.importorskip("pygeodesic.geodesic")
    box = G.cube_mesh(3)
    vertices = box.vertices + 1e-6 * np.random.default_rng(1).standard_normal(len(box.vertices))[:, None] * (
        box.vertices - 0.5)
    assert G.inspect(vertices, box.faces, require_closed=True) == []
    solver = E.ExactGeodesic(G.TriMesh(vertices, box.faces))
    assert np.any((solver.excess > 0) & (solver.excess < E.ANGLE_TOLERANCE))
    reference, _ = geodesic.PyGeodesicAlgorithmExact(vertices, box.faces.astype(np.int32)).geodesicDistances(
        np.array([0], dtype=np.int32), None)
    assert np.max(np.abs(solver.distances(0) - reference)) <= 1e-10


@pytest.mark.lab_task("T038")
def test_flipout_geodesics_are_never_shorter():
    pytest.importorskip("potpourri3d")
    exact = S.exact_distance_study(levels=(2,), torus=(12, 6), flipout=("icosphere-2", "torus-12x6"))
    flipout = S.exact_independent_study(exact)["flipout"]
    assert [r["mesh"] for r in flipout] == ["icosphere-2", "torus-12x6"]
    for row in flipout:
        assert row["min_excess"] >= -1e-12, row
        assert 0 < row["shortest"] <= row["pairs"]
    # Locally shortest is not globally shortest: some FlipOut geodesics on the icosphere are longer.
    assert flipout[0]["shortest"] < flipout[0]["pairs"] and flipout[0]["max_excess"] > 1e-6


@pytest.mark.lab_task("T038")
def test_paths_and_approximations_never_beat_the_exact_distance():
    exact = S.exact_distance_study(levels=(1, 2), torus=(12, 6), flipout=("icosphere-2",))
    assert all(r["symmetry_max_abs"] <= 1e-12 and r["edge_excess"] <= 1e-12 for r in exact["rows"])
    assert exact["rows"][1]["edge_graph_min_excess"] >= -1e-12
    comparison = S.exact_comparison_study(exact, steiner_levels=(2,), ks=(1, 3), heat_levels=(1, 2))
    assert min(r["min_excess"] for r in comparison["edge_graph"]) >= -1e-12
    steiner = comparison["steiner"][0]
    assert min(steiner["min_excess"]) >= -1e-12 and steiner["mean_excess"][1] < steiner["mean_excess"][0]
    heat = comparison["heat"]
    assert heat[1]["max_abs_error"] < heat[0]["max_abs_error"]
    traced = S.traced_exact_study(configs=((1, 2.0), (2, 2.0)))
    rows = [r for r in traced["rows"] if r["status"] == "completed"]
    assert len(rows) == 12 and min(r["excess"] for r in rows) >= -1e-12
    for level in (1, 2):  # some traces below pi are shortest paths and some are not
        shortest = [r["shortest"] for r in rows if r["level"] == level]
        assert 0 < sum(shortest) < len(shortest)


@pytest.mark.lab_task("T038")
def test_vertex_continuation_on_flat_and_cone_vertices():
    study = S.continuation_study()
    assert all(r["status"] == "completed" and r["vertices_passed"] >= 1 for r in study["flat"])
    assert study["flat_max_error"] <= 1e-12
    assert max(study["flat_vertices_passed"]) >= 5  # rows of vertices, one at every step
    for row in study["cube"]["rows"]:  # a cone vertex: 3 pi / 4 on both sides
        assert row["vertices_passed"] == 1 and row["endpoint_error"] <= 1e-12
        assert abs(row["exact_minus_closed_form"]) <= 1e-12 and row["traced_minus_exact"] > 0.05
    plane = G.plane_mesh(4, 4)
    start = np.array([0.43, 0.61, 0.0])
    face = plane.locate(start)[0]
    toward = S._unit(plane.vertices[18] - start)  # vertex 18 = (0.75, 0.75), interior and flat
    # The default rule still refuses a vertex hit; the continuation is asked for.
    assert G.trace(plane, face, start, toward, 1.0).status == "vertex_hit"
    continued = G.trace(plane, face, start, toward, 0.5, vertex_rule=S.PS)
    assert continued.completed and continued.vertices == [18]
    assert np.max(np.abs(continued.end_point - (start + 0.5 * toward))) <= 1e-12
    assert continued.length == pytest.approx(0.5, abs=1e-15)
    # A start at a vertex leaves through the declared face; a direction outside that face's wedge is refused.
    out = S._unit(np.array([1.0, 0.3, 0.0]))
    leaving = S._leaving_face(plane, plane.vertices[18], out)
    from_vertex = G.trace(plane, leaving, plane.vertices[18], out, 0.2, vertex_rule=S.PS)
    assert from_vertex.completed and np.max(np.abs(from_vertex.end_point - (plane.vertices[18] + 0.2 * out))) <= 1e-12
    wrong = G.trace(plane, S._leaving_face(plane, plane.vertices[18], -out), plane.vertices[18], out, 0.2,
                    vertex_rule=S.PS)
    assert wrong.status == "invalid_direction"
    # At a boundary vertex no continuation is declared: the trace stops there.
    stopped = G.trace(plane, face, start, S._unit(plane.vertices[24] - start), 1.0, vertex_rule=S.PS)
    assert stopped.status == "boundary_reached" and np.allclose(stopped.points[-1], plane.vertices[24])
    with pytest.raises(ValueError):
        G.trace(plane, face, start, toward, 0.5, vertex_rule="bisect")


@pytest.mark.lab_task("T038")
def test_fan_continuation_bisects_the_total_angle():
    study = S.fan_study()
    offset = study["offset"] * sum(study["radii"]) / study["radii"][1]
    for row in study["rows"]:
        assert row["issues"] == [] and row["status"] == "completed" and row["vertices_passed"] == 1
        assert abs(row["polar_minus_half"]) <= 1e-12 and abs(row["radius_error"]) <= 1e-12
        assert row["reverse_error"] <= 1e-12
        assert max(abs(d) for d in row["exact_minus_closed_form"]) <= 1e-12
        assert row["through_measured"] == row["through_expected"]
        shortest = abs(row["traced_minus_exact"]) <= S.SHORTEST
        assert shortest == (row["total_angle_over_pi"] >= 2.0) == row["path_through_centre"]
        # The two one-sided limits, pi and theta - pi, are the continuation's first-order neighbours.
        assert abs(row["one_sided_limit_error"] - offset) <= 1e-9
    saddle = next(r for r in study["rows"] if r["total_angle_over_pi"] == 2.5)
    assert saddle["through_expected"] == [False, False, False, True, True]  # the fan between pi and theta - pi
    with pytest.raises(ValueError):
        G.fan_mesh(5, 2.5 * math.pi)  # a saddle fan zigzags, so it needs an even number of triangles


@pytest.mark.lab_task("T038")
def test_generic_traces_do_not_hit_vertices():
    study = S.vertex_hit_study(levels=(2, 3), traces=40)
    assert study["vertex_hits"] == 0
    assert all(r["statuses"] == ["completed"] and r["min_margin"] > G.VERTEX_TOLERANCE for r in study["rows"])
    assert study["rows"][1]["crossings_per_trace"] > 1.5 * study["rows"][0]["crossings_per_trace"]
    # The margins are those of the edge crossings: a trace across a planar grid row by row.
    plane = G.plane_mesh(4, 4)
    tr = G.trace(plane, plane.locate([0.1, 0.05, 0.0])[0], np.array([0.1, 0.05, 0.0]), np.array([0.0, 1.0, 0.0]),
                 0.9)
    margins = S.crossing_margins(plane, tr)
    assert len(margins) == len(tr.faces) - 1 and min(margins) == pytest.approx(0.4, abs=1e-12)


@pytest.mark.lab_task("T038")
def test_back_traced_paths_are_shortest_and_bend_only_at_saddles_and_boundaries():
    study = S.path_study(sources=2, targets=3)
    for row in study["rows"]:
        assert row["length_error"] <= 1e-12 and row["off_surface"] == 0 and row["outside_interval"] <= 1e-12
        assert row["max_turn"] <= S.BEND and row["min_side_minus_pi"] >= -S.BEND
        assert row["reverse_distance"] <= 1e-9 and row["bends"]["cone"] == row["bends"]["flat"] == 0
    # The L-shape's hidden target is reached around the reflex corner, where the path bends.
    shape = S._l_shape(8)
    nearest = [int(np.argmin(np.linalg.norm(shape.vertices - p, axis=1)))
               for p in ([1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 0.0])]
    source, target, corner = nearest
    path = E.ExactGeodesic(shape).propagate(source).path(target)
    assert corner in path["vertices"] and path["length"] == pytest.approx(math.sqrt(0.5) + 0.5, abs=1e-12)
    assert np.allclose(path["points"][0], shape.vertices[target]) and np.allclose(path["points"][-1],
                                                                                    shape.vertices[source])
    assert S.side_angles(shape, corner, path["points"][path["vertices"].index(corner) - 1],
                         path["faces"][path["vertices"].index(corner) - 1],
                         path["points"][path["vertices"].index(corner) + 1],
                         path["faces"][path["vertices"].index(corner)])[0] > math.pi + 1e-3
    # Distances at surface points from the recorded windows equal the distances to the points inserted as vertices.
    mesh = G.icosphere(2)
    inside, on_edge = mesh.point(7, [0.2, 0.5, 0.3]), mesh.point(40, [0.4, 0.6, 0.0])
    refined, ids = E.insert_points(mesh, [(7, inside), (40, on_edge)])
    exact = E.ExactGeodesic(refined).distances(0)
    propagation = E.ExactGeodesic(mesh).propagate(0)
    assert np.array_equal(propagation.distances, E.ExactGeodesic(mesh).distances(0))
    assert propagation.distance_at(7, inside) == pytest.approx(exact[ids[0]], abs=1e-12)
    assert propagation.distance_at(40, on_edge) == pytest.approx(exact[ids[1]], abs=1e-12)
    assert propagation.path_to_point(7, inside)["length"] == pytest.approx(exact[ids[0]], abs=1e-12)
    vertices, faces = S._octahedron()
    split = G.TriMesh.build(np.vstack([vertices, vertices + 5]), np.vstack([faces, faces + 6]), require_connected=False)
    with pytest.raises(G.MeshRefusal) as refused:
        E.ExactGeodesic(split).propagate(0).path(7)
    assert refused.value.code == "unreachable_target"


@pytest.mark.lab_task("T038")
def test_cut_points_match_the_isolated_cone_prediction():
    traced = S.traced_exact_study(configs=((1, 2.0), (2, 2.0)))
    summary = M.cut_summary(traced)
    assert summary["max_decrease"] <= 1e-12 and summary["endpoint_gap"] <= 1e-12 and summary["mismatched"] == 0
    assert len(summary["single"]) >= 5 and summary["single_error"] <= 0.0
    assert summary["later_than_predicted"] <= 0.0 and summary["uncut_margin"] >= 0.0
    # Level 2, trace 5: the digon encloses two vertices, and the cut comes before either one alone predicts.
    witness = next(r for r in summary["several"] if (r["level"], r["start"]) == (2, 5))
    assert len(witness["cut"]["enclosed"]) == 2
    assert summary["prediction"](witness) - witness["cut"]["arclength"] > 0.1
    # The isolated-cone formula: the vertex itself when aimed at it, no crossing from half the defect on.
    assert S.single_cone_cut(1.0, 0.0, 0.2) == pytest.approx(1.0, abs=1e-15)
    assert S.single_cone_cut(1.0, math.tan(0.1), 0.2) == math.inf
    assert S.single_cone_cut(1.0, math.tan(0.05), 0.2) == pytest.approx(
        math.hypot(1.0, math.tan(0.05)) * math.sin(0.1) / math.sin(0.05), rel=1e-12)
    # A loop through the spoke midpoints of a vertex's one-ring encloses exactly that vertex.
    mesh = G.icosphere(2)
    center = int(mesh.faces[0, 0])
    loop = [0.5 * (mesh.vertices[center] + mesh.vertices[mesh.faces[f, (k + 1) % 3]])
            for f, k, _ in G.vertex_fan(mesh, 0, 0)]
    assert S.enclosed_vertices(mesh.vertices, np.array(loop)) == [center]


@pytest.mark.lab_task("T038")
def test_continuation_paths_and_cut_points_agree_with_pygeodesic():
    pytest.importorskip("pygeodesic.geodesic")
    fans = S.fan_study(angles=(1.5, 2.5))
    paths = S.path_study(sources=2, targets=2)
    traced = S.traced_exact_study(configs=((1, 2.0),))
    result = S.continuation_independent_study(fans, paths, traced)
    assert result["fan_max_abs"] <= 1e-12 and result["path_max_distance"] <= 1e-9
    assert result["path_length_max_abs"] <= 1e-12
    assert result["cut_probes"] >= 2 and result["cut_probe_max_abs"] <= 1e-12


# ---------------------------------------------------------------- T039
@pytest.mark.lab_task("T039")
def test_convergence_orders_on_small_levels():
    traces = S.sphere_trace_study(levels=(2, 3, 4), starts=S.STARTS[:3])
    order = S.fitted_order([r["h"] for r in traces["rows"]], [r["mean_length_defect"] for r in traces["rows"]])
    assert abs(order - 2.0) <= 0.3
    cylinder = S.cylinder_study(ns=(16, 32, 64))
    for row in cylinder["rows"]:
        assert row["helix_error"] == pytest.approx(row["exact_prediction"], abs=1e-12)
        assert abs(row["chord_position_term"]) <= row["chord_term_bound"] * 1.01
    helix = S.fitted_order([r["h"] for r in cylinder["rows"]], [r["helix_error"] for r in cylinder["rows"]])
    assert abs(helix - 2.0) <= 0.1
    distances = S.distance_study(levels=(1, 2, 3), ks=(1,), heat_levels=(1, 2, 3))
    edge = [r["edge_max_signed_rel"] for r in distances["rows"]]
    assert edge[-1] >= 0.2 and edge[-1] >= edge[0]  # no convergence of edge-graph distance
    assert all(r["steiner1_max_abs_rel"] > 0 for r in distances["rows"])
    heat = [r["heat_max_abs"] for r in distances["rows"]]
    assert heat[2] < heat[1] < heat[0]


@pytest.mark.lab_task("T039")
def test_convergence_task_report(reports):
    report = reports["T039"]
    _completed_with_unestablished_physics(report)
    length = _value(report, "Traced-geodesic length defect")
    assert abs(length["value"]["order"] - 2.0) <= 0.2
    endpoint = _value(report, "Traced-geodesic endpoint error")
    assert min(endpoint["value"]["endpoint_local_orders"]) >= 1.0
    floor = _value(report, "Edge-graph Dijkstra distance")
    assert floor["counterexample"]["statement"].startswith("Edge-graph shortest paths converge")
    assert floor["value"]["aitken_extrapolation"] == pytest.approx(math.sqrt(5) - 2, abs=1e-3)
    helix = _value(report, "Prism-cylinder helix")
    assert helix["value"]["max_abs_minus_exact"] <= 1e-9
    assert abs(helix["value"]["scaled_to_leading_constant"][-1] - 1.0) <= 0.01
    svg = next(a for a in report["generated_artifacts"] if a["path"].endswith("convergence.svg"))
    assert svg["sha256"]


# ---------------------------------------------------------------- T040
@pytest.mark.lab_task("T040")
def test_flat_jacobi_counterexample_and_valence_limit():
    study = S.jacobi_study(levels=(2, 3), deltas=(0.1, 1e-5), starts=S.STARTS[:2])
    tiny = [r for r in study["rows"] if r["delta"] == 1e-5]
    assert all(r["identical_face_sequences"] == r["pairs"] == 2 for r in tiny)
    assert max(r["max_flat_deviation"] for r in tiny) <= 1e-6
    assert tiny[-1]["mean_abs_error"] == pytest.approx(study["length"] - math.sin(study["length"]), abs=1e-6)
    wide = [r for r in study["rows"] if r["delta"] == 0.1]
    assert wide[1]["mean_abs_error"] < wide[0]["mean_abs_error"]
    curvature = S.curvature_study(levels=(4, 5), torus_sizes=(8, 16))
    assert curvature["valence5_limit"] == pytest.approx(3 / (4 * math.cos(math.pi / 5) ** 2), rel=1e-14)
    assert abs(curvature["sphere"][-1]["valence5_value"] - curvature["valence5_limit"]) <= 2e-4
    assert curvature["sphere"][-1]["valence5_value"] - 1 >= 0.14  # the pointwise estimate does not tend to K = 1
    assert all(abs(r["gauss_bonnet_residual"]) <= 1e-9 for r in curvature["sphere"] + curvature["torus"])
    for row in curvature["sphere"]:  # mirror-plane plateau with the barycentric area, convergence with Voronoi
        assert row["valence6_base_edge_max_error"] >= 1.5e-3
        assert row["voronoi_valence6_mirror_max_error"] <= 2e-3


def test_icosahedral_mirror_classes():
    mesh = G.icosphere(2)
    edge, median = S.icosahedral_mirror_classes(mesh.vertices)
    assert not (edge & median).any()
    assert edge[:12].all()  # base vertices lie on base-edge arcs
    assert edge.sum() == 12 + 30 * 3  # three interior subdivision points on each of the 30 base edges


@pytest.mark.lab_task("T040")
def test_jacobi_task_report(reports):
    report = reports["T040"]
    _completed_with_unestablished_physics(report)
    flat = _value(report, "At fixed mesh a 1e-5 heading offset")
    assert "counterexample" in flat
    assert flat["value"]["identical_face_sequences"] == flat["value"]["pairs"] == 30
    assert flat["value"]["error_vs_smooth"] == pytest.approx(2.0 - math.sin(2.0), rel=1e-12)
    assert flat["value"]["crossover"]["pairs"] > flat["value"]["crossover"]["identical_face_sequences"]
    assert flat["regression_tolerance"]["abs"] >= 1e-8
    valence = _value(report, "Angle-defect curvature at valence-5")
    assert valence["value"]["valence5_values"][-1] == pytest.approx(4.5 - 1.5 * math.sqrt(5), abs=1e-4)
    mirror = _value(report, "Barycentric angle-defect curvature does not converge pointwise")
    assert mirror["counterexample"]["statement"].startswith("Angle-defect curvature with the barycentric area")
    # Gauss-Bonnet holds for any closed mesh, so it is a sanity value in the prose, not a finding.
    assert not any("Gauss-Bonnet" in f["claim"] for f in report["findings"])
    assert "sanity check" in report["numerical_result"]


# ---------------------------------------------------------------- T041
@pytest.mark.lab_task("T041")
def test_schwarz_lantern_counterexample():
    study = S.lantern_study(ns=(4, 8, 16), folded_n=4)
    for row in study["rows"]:
        assert row["hausdorff_sampled"] == pytest.approx(row["hausdorff_closed_form"], abs=1e-12)
        assert abs(row["area_closed_form_error"]) <= 1e-9
        assert row["trace_status"] == "boundary_reached"
        assert row["traced_height"] == pytest.approx(row["height_closed_form"], abs=1e-10)
        assert row["max_interior_curvature"] <= 1e-9
        assert row["issues"] == []
    assert study["rows"][-1]["area_ratio"] > 1.5 and study["rows"][-1]["hausdorff_sampled"] < 0.02
    assert abs(study["rows"][-1]["max_normal_tilt_deg"] - study["limit_tilt_deg"]) <= 0.2
    assert study["folded"]["issues"][0] == "folded_face"
    assert study["folded"]["min_normal_radial"] > 0  # refused as folded with no face pointing inward


@pytest.mark.lab_task("T041")
def test_rank_correlation_matches_average_ranks():
    assert M._average_ranks([3.0, 1.0, 3.0, 2.0]).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    stats = pytest.importorskip("scipy.stats")
    x = [0.3, 1.2, 0.3, 5.0, 2.2, 2.2, 0.9]
    y = [1.0, 0.5, 0.7, 3.0, 3.0, 2.0, 0.1]
    assert M.spearman(x, y) == pytest.approx(float(stats.spearmanr(x, y).statistic), abs=1e-12)


@pytest.mark.lab_task("T041")
def test_quality_task_report(reports):
    report = reports["T041"]
    _completed_with_unestablished_physics(report)
    labels = _labels(report)
    assert labels[RANK_CLAIM] == ("independently_verified" if S.optional_version("scipy") else "numerically_verified")
    counterexamples = [f for f in report["findings"] if "counterexample" in f]
    assert len(counterexamples) >= 6
    folded = _value(report, "For the three declared seeds per amplitude, the dihedral fold check refuses exactly")
    refusals = [c for c in folded["basis"]["checks"] if c["reference_kind"] == "refusal"]
    inverted = folded["value"]["inverted_faces"]
    assert len(refusals) == len(inverted) == len(folded["value"]["meshes"])
    for check, count in zip(refusals, inverted):
        assert check["expected_refusal"] == ("folded_face" if count > 0 else "none")
        assert check["observed_refusal"] == check["expected_refusal"]
    missed = _value(report, "Over 60 further seeds per amplitude the dihedral fold check accepts some")
    assert missed["counterexample"]["statement"].startswith("The dihedral fold check (folded_face) refuses every")
    assert missed["value"]["witness"]["normal_radial"] < 0
    assert missed["value"]["witness"]["issues_with_centre"] == ["inverted_face"]
    rates = {r["amplitude"]: r for r in missed["value"]["rates"]}
    assert rates[0.2]["inverted_accepted"] > 0 and rates[0.2]["clean_accepted"] > 0
    assert "counterexample" in _value(report, "A tangential jitter of 0.2 h inverts a face for some but not all")
    assert _value(report, "Over 60 further seeds per amplitude the dihedral fold check refuses no")["value"]["rates"]
    assert "fold refusal exactly when" not in report["expected_invariant"]
    ranked = _value(report, "Maximum radius ratio is positively rank-correlated")
    assert max(ranked["value"]["spearman_within_family"]["latitude_longitude"].values()) <= 1e-12
    within = _value(report, "Within the latitude-longitude family")
    assert within["value"]["better_quality"]["mesh"] == "uv-sphere-40x16"
    voronoi = _value(report, "With the mixed Voronoi area the regular icosphere")
    assert voronoi["evidence_status"] == "numerically_verified"
    lantern = _value(report, "A strongly pleated lantern")
    assert lantern["value"]["min_normal_radial"] > 0 and "counterexample" in lantern
    authority = [f for f in report["findings"] if f["domain"] == "production_acceptance"]
    assert authority and authority[0]["evidence_status"] == "not_established"


def test_rank_finding_without_scipy(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "optional_version", lambda name: None)
    report = _run("T041", runner.Context(tmp_path))
    assert _labels(report)[RANK_CLAIM] == "numerically_verified"
    assert report["evidence_status"]["primary"] == "numerically_verified"


@pytest.mark.lab_task("T041", "T042")
def test_fold_check_misses_small_bend_inversions():
    amplitude, offset = S.INVERTED_JITTER
    mesh = S._jitter_unvalidated(G.icosphere(3), amplitude, S.SEED + offset)
    inverted = S.inverted_faces(mesh)
    assert len(inverted) == 1
    assert G.inspect(mesh.vertices, mesh.faces, require_closed=True) == []  # the dihedral check misses it
    table = G._edge_table(mesh.faces, len(mesh.vertices))
    dots = np.einsum("ij,ij->i", mesh.face_normals[table["pair_first"] // 3],
                     mesh.face_normals[table["pair_second"] // 3])
    assert dots.min() > G.FOLD_COSINE
    with pytest.raises(G.MeshRefusal) as refused:
        G.TriMesh.build(mesh.vertices, mesh.faces, require_closed=True, center=np.zeros(3))
    assert refused.value.code == "inverted_face"
    vertices, faces = S._octahedron()
    assert [c for c, _ in G.inspect(vertices, faces[:, ::-1], center=np.zeros(3))] == ["inverted_face"]
    for valid in (G.icosphere(2), G.uv_sphere(10, 16, twist=0.1)):
        assert G.inspect(valid.vertices, valid.faces, require_closed=True, center=np.zeros(3)) == []
    rates = S.fold_rate_study(amplitudes=(0.2,), seeds=(offset, 1001))
    row = rates["rows"][0]
    assert row["inverted_accepted"] == 1 and row["refused_with_centre"] == row["meshes"] - row["clean_accepted"]
    assert rates["witness"]["seed"] == S.SEED + offset and rates["witness"]["issues_with_centre"] == ["inverted_face"]


def test_uv_sphere_names_include_twist():
    assert G.uv_sphere(4, 6).name == "uv-sphere-4x6"
    assert G.uv_sphere(4, 6, twist=0.08).name == "uv-sphere-4x6-t0.08"


# ---------------------------------------------------------------- T042
@pytest.mark.lab_task("T042")
def test_every_defect_has_a_named_refusal():
    study = S.refusal_study()
    assert all(c["observed"] == c["expected"] for c in study["cases"]), study["cases"]
    assert {c["expected"] for c in study["cases"]} >= set(G.MESH_CODES) | set(G.TRACE_CODES) | set(G.QUERY_CODES)
    assert all(c["issues"] == [] for c in study["controls"])
    assert study["multiple_defects"] == ["nonfinite_vertex", "inconsistent_orientation", "unreferenced_vertex"]
    assert study["unguarded_nonfinite_curvatures"] >= 1
    vertices, faces = S._octahedron()
    with pytest.raises(G.MeshRefusal) as refused:
        G.TriMesh.build(vertices, faces[1:], require_closed=True)
    assert refused.value.code == "open_boundary"
    plane = G.plane_mesh(2, 2)
    with pytest.raises(G.MeshRefusal) as refused:
        G.trace(plane, 0, [0.3, 0.1, 0.0], [1.0, 0.0, 0.0], 5.0, strict=True)
    assert refused.value.code == "boundary_reached"
    with pytest.raises(G.MeshRefusal) as refused:
        G.heat_distance(G.icosphere(1), 0, max_vertices=10)
    assert refused.value.code == "mesh_too_large"


@pytest.mark.lab_task("T042")
def test_refusal_task_report(reports):
    report = reports["T042"]
    _completed_with_unestablished_physics(report)
    refusals = _value(report, "Every declared surface-data defect")
    checks = refusals["basis"]["checks"]
    assert len(checks) >= 23 and all(c["reference_kind"] == "refusal" and c["passed"] for c in checks)
    assert refusals["value"]["aimed-at-vertex"] == "vertex_hit"
    assert refusals["value"]["strip-faces-not-adjacent"] == "invalid_strip"
    assert refusals["value"]["jittered-inverted-face"] == "inverted_face"
    assert any("Self-intersections" in a or "self-intersections" in a for a in report["unresolved_assumptions"])
    assert any("inverted faces whose bend to every neighbour" in a for a in report["unresolved_assumptions"])
    undetected = _value(report, "Without a declared centre the validator accepts a jittered icosphere")
    assert undetected["value"]["structural_issues"] == [] and undetected["value"]["inverted_faces"] >= 1
    assert "counterexample" in undetected


# ---------------------------------------------------------------- T043
@pytest.mark.lab_task("T043")
def test_linearization_matches_monte_carlo_and_breaks():
    study = S.uncertainty_study(level=3, sigmas=(1e-3, 1e-2), samples=2000)
    assert study["strip"]["start_index"] == S.MARKER_START
    rows = {o["observable"]: o["rows"] for o in study["observables"]}
    assert abs(rows["marker geodesic distance"][0]["ratio"] - 1) <= 0.12
    assert rows["marker geodesic distance"][0]["invalid_fraction"] == 0.0
    assert rows["marker geodesic distance"][1]["invalid_fraction"] > 0.1
    for name, values in rows.items():
        if name.startswith("vertex normal"):
            assert all(abs(r["ratio"] - 1) <= 0.12 and r["flipped_normals"] == 0 for r in values)
        if name.startswith("angle-defect"):
            assert abs(values[0]["ratio"] - 1) <= 0.12 and values[1]["ratio"] >= 1.25


@pytest.mark.lab_task("T043")
def test_corridor_threshold_depends_on_the_strip():
    study = S.corridor_study(sigmas=(1e-3, 1e-2), samples=1000)
    fractions = {r["start_index"]: r["left_fraction"] for r in study["rows"]}
    margins = {r["start_index"]: r["vertex_margin"] for r in study["rows"]}
    # What is claimed: the declared strip has the largest margin and stays in its corridor at sigma = 1e-3 ...
    assert max(margins, key=margins.get) == S.MARKER_START and fractions[S.MARKER_START][0] == 0.0
    assert fractions[min(margins, key=margins.get)][0] > 0.1
    # ... but it is not the most robust strip at larger sigma: a smaller-margin strip leaves less often.
    declared = fractions[S.MARKER_START][1]
    best = min((i for i in fractions if i != S.MARKER_START), key=lambda i: fractions[i][1])
    assert margins[best] < margins[S.MARKER_START]
    assert declared - fractions[best][1] > 4 * math.sqrt(0.5 / 1000)  # well beyond binomial noise


@pytest.mark.lab_task("T043")
def test_per_vertex_uncertainty_matches_monte_carlo():
    mesh = G.icosphere(2)
    n, sigma = len(mesh.vertices), 1e-4
    field = G.vertex_uncertainty(mesh, sigma ** 2)
    same = [G.vertex_uncertainty(mesh, np.full(n, sigma ** 2)),
            G.vertex_uncertainty(mesh, np.broadcast_to(sigma ** 2 * np.eye(3), (n, 3, 3)))]
    for other in same:
        assert np.allclose(other["curvature_sd"], field["curvature_sd"], rtol=1e-12)
        assert np.allclose(other["normal_sd"], field["normal_sd"], rtol=1e-12)
    # Against the two-vertex propagation (separate one-ring code) at vertex 0.
    _, items = S.observables(mesh)
    gradients = {o["name"]: np.sqrt(np.sum(S._fd_jacobian(o["function"], mesh.vertices[o["ids"]]) ** 2))
                 for o in items}
    assert field["curvature_sd"][0] == pytest.approx(sigma * gradients["angle-defect curvature at valence-5 vertex 0"],
                                                     rel=1e-6)
    assert field["normal_sd"][0] == pytest.approx(sigma * gradients["vertex normal at valence-5 vertex 0"], rel=1e-6)
    # Against whole-mesh Monte Carlo at every vertex.
    rng = np.random.Generator(np.random.PCG64(7))
    samples = 1000
    normals, curvature = G.batch_vertex_fields(mesh.vertices[None] + rng.standard_normal((samples, n, 3)) * sigma,
                                               mesh.faces)
    normal0, curvature0 = (a[0] for a in G.batch_vertex_fields(mesh.vertices[None], mesh.faces))
    se = math.sqrt(2 / (samples - 1))
    assert np.max(np.abs(np.var(curvature, axis=0, ddof=1) / field["curvature_sd"] ** 2 - 1)) <= 5 * se
    tilt = np.mean(np.sum((normals - normal0) ** 2, axis=2), axis=0)
    assert np.max(np.abs(tilt / field["normal_sd"] ** 2 - 1)) <= 5 * se
    assert np.allclose(curvature0, field["curvature"], atol=1e-12)
    # Boundary vertices have no curvature; invalid covariances and meshes are refused.
    plane = G.vertex_uncertainty(G.plane_mesh(3, 3), 1e-8)
    assert np.isnan(plane["curvature_sd"][~plane["interior"]]).all()
    assert np.isfinite(plane["curvature_sd"][plane["interior"]]).all()
    with pytest.raises(ValueError):
        G.vertex_uncertainty(mesh, -np.eye(3)[None].repeat(n, axis=0))
    bad = G.TriMesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.8, 0.2, 0.0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    with pytest.raises(G.MeshRefusal) as refused:
        G.vertex_uncertainty(bad, 1e-8)
    assert refused.value.code == "folded_face"


@pytest.mark.lab_task("T043")
def test_uncertainty_task_report(reports):
    report = reports["T043"]
    _completed_with_unestablished_physics(report)
    robust = _value(report, "A strip with a smaller vertex margin than the declared strip")
    assert robust["counterexample"]["statement"].startswith("The strip with the largest vertex margin")
    assert robust["value"]["more_robust"]["vertex_margin"] < robust["value"]["declared"]["vertex_margin"]
    field = _value(report, "Linearized per-vertex standard deviations of vertex normals and angle-defect curvature")
    assert field["value"]["vertices"] == 642 and field["value"]["two_vertex_cross_check_max_rel"] <= 1e-6
    kinds = {c["reference_kind"] for c in field["basis"]["checks"]}
    assert kinds == {"self_convergence", "cross_implementation"}
    assert "best case" not in " ".join(report["input_data"])
    assert any("independently measured normals" in a for a in report["unresolved_assumptions"])
    slopes = _value(report, "Sensitivity to vertex noise")["value"]["slopes"]
    assert slopes["angle-defect curvature at valence-5 vertex 0"] == pytest.approx(-2.0, abs=0.25)
    assert abs(slopes["marker geodesic distance"]) <= 0.2
    split = _value(report, "The marker-distance sensitivity is carried by the marker-face vertices")
    assert split["value"]["interior_slope"] > 0.25
    corridors = _value(report, "The noise level at which a fixed face corridor fails")
    assert "counterexample" in corridors
    normals = _value(report, "Linearized vertex-noise propagation matches Monte Carlo for vertex normals")
    assert normals["value"]["flipped_normals"] == 0
    growth = _value(report, "Under fixed vertex noise")
    assert growth["value"]["total_rms_error"][-1] > 10 * growth["value"]["total_rms_error"][0]
    calibration = [f for f in report["findings"] if f["domain"] == "calibration"]
    assert calibration and calibration[0]["evidence_status"] == "not_established"
    legend = next(a for a in report["generated_artifacts"] if a["path"].endswith("linearization-ratio.svg"))
    assert legend["sha256"]


@pytest.mark.lab_task("T043")
def test_t043_reconciles_the_two_declared_strip_estimates(reports):
    """Two seeded studies estimate the declared strip's leave fraction; the report checks their agreement."""
    report = reports["T043"]
    record = _value(report, "The declared marker segment stays in its face corridor")
    other = record["value"]["corridor_study"]
    assert record["value"]["seed"] == S.SEED and other["seed"] == S.SEED + 30
    check = next(c for c in record["basis"]["checks"] if "independent six-strip corridor study" in c["reference"])
    assert check["passed"] and check["tolerance"] == 1.96 and check["comparison"] == "le"
    assert other["max_abs_difference"] <= other["interval_95_at_max"]
    text = report["numerical_result"]
    assert f"seed {S.SEED})" in text and f"seed {S.SEED + 30}, 4000 samples" in text
    assert "within the 95% interval of a difference of two independent fractions" in text
    # The retained estimates at sigma = 1e-2 (0.39525 and 0.41075) differ by more than one estimate's 95% half-width
    # (0.015) but lie within the 95% interval of their difference (about 0.0215).
    retained = M.declared_strip_agreement({"0.0001": 0.0, "0.001": 0.0, "0.003": 0.03075, "0.01": 0.39525}, 4000,
                                          [0.0, 0.0, 0.02825, 0.41075], [1e-4, 1e-3, 3e-3, 1e-2], 4000)
    assert retained["max_abs"] == pytest.approx(0.0155) and retained["sigma_at_max"] == 1e-2
    assert retained["interval_at_max"] == pytest.approx(0.0215, abs=5e-4) and retained["max_z"] < 1.96
    apart = M.declared_strip_agreement({"0.01": 0.39525}, 4000, [0.45], [1e-2], 4000)
    assert apart["max_z"] > 1.96
    assert M.agreement_text(retained).startswith("the two independent estimates agree within")
    assert M.agreement_text(apart).startswith("the two independent estimates differ beyond")


@pytest.mark.lab_task("T043")
def test_t043_says_when_the_two_declared_strip_estimates_differ(reports, mesh_ctx, tmp_path):
    """When the corridor study disagrees with the propagation study, the report text does not claim agreement."""
    ctx = runner.Context(tmp_path)
    for name in ("uncertainty", "scaling", "refinement-noise", "vertex-field", "corridors"):
        key = f"surfaces_discrete_mesh:{name}"
        value = copy.deepcopy(mesh_ctx.memo(key, lambda: pytest.fail(f"{key} was not computed by the section run")))
        if name == "corridors":
            # The corridor study's declared strip leaves its corridor at sigma = 1e-2 in 50% of samples, far from
            # the propagation study's 0.395 (z about 9.5).
            declared = _value(reports["T043"], "The declared marker segment stays")["value"]["start_index"]
            row = next(r for r in value["rows"] if r["start_index"] == declared)
            row["left_fraction"][value["sigmas"].index(1e-2)] = 0.5
        ctx.memo(key, lambda value=value: value)
    report = _run("T043", ctx)
    record = _value(report, "The declared marker segment stays")
    check = next(c for c in record["basis"]["checks"] if "independent six-strip corridor study" in c["reference"])
    assert check["observed"] > 1.96 and check["passed"] is False
    assert record["evidence_status"] == "not_established"
    assert report["evidence_status"]["primary"] == "not_established"
    text = report["numerical_result"]
    assert "agree within" not in text
    assert "differ beyond the 95% interval of a difference of two independent fractions" in text
    assert f"largest z {check['observed']:.2f} against 1.96" in text


@pytest.mark.lab_task("T038", "T039", "T040", "T041", "T042", "T043", "T044")
def test_next_steps_name_forward_work(reports):
    """A completed task's next step is its own open question, never a queue task that already ran."""
    for task_id, report in reports.items():
        text = report["recommended_next_task"]
        assert text == M.NEXT_STEPS[task_id], task_id
        if report["state"] == "completed":
            assert text.startswith("Deferred research question: "), (task_id, text)
    # T038 delivered the exact solver, the vertex rule, back-traced paths and located cut points; its next step names
    # the work they leave open, not the delivered work again.
    step = reports["T038"]["recommended_next_task"]
    assert reports["T038"]["state"] == "completed"
    assert "several vertices" in step and "cut locus" in step and "saddle-bearing" in step and "exp(-K L^2 / 6)" in step
    assert "T039" not in step and "exact two-point" not in step
    assert "refused as vertex_hit" not in step and "back-trace the shortest path" not in step
    # T045 now carries the split for model-derived distances on parametric surfaces; T044 hands out only the
    # part still open there (the mesh form of geometry_m2), naming T045 and T140 as partial deliverers.
    step = M.NEXT_STEPS["T044"]
    assert "Partly delivered by T045" in step and "T140" in step
    assert "does not carry" not in step
    assert "geometry_m2" in step and "T043" in step and "mesh" in step
    # A hardware-gated part names the route by which acquired bytes could enter the task.
    hardware = M.NEXT_STEPS["T043"]
    for fragment in ("hardware-gated", "ctx.capture('scan-export')", "ciw lab run T043 --capture scan-export=PATH",
                     "raw_sha256", "calibration", "runner.CAPTURE_INSTRUMENTS", "signed-capture trust anchor",
                     "lab/hardware/<run-id>", "ciw lab hardware retain", "stay not_established even when such data"):
        assert fragment in hardware, fragment
    # The step says no scanner probe exists: true while the physical gate maps no instrument to that role.
    assert "scan-export" not in runner.CAPTURE_INSTRUMENTS


@pytest.mark.lab_task("T044")
def test_t044_next_step_names_what_t045_leaves_open(reports, tmp_path):
    """T045 carries the split T044 asked for; T044's next step names only the mesh form, which T045 leaves open."""
    pytest.importorskip("ciw.lab.observation")
    modes = pytest.importorskip("ciw.lab.observation_modes")
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    t045 = validate_report(runner.run_task(queue["T045"], module_implementations("observation")["T045"],
                                           runner.Context(tmp_path), {}))
    split = [f for f in t045["findings"] if "geometry and sensor variance" in f["claim"] and "T044" in f["claim"]]
    assert len(split) == 1 and split[0]["evidence_status"] == "numerically_verified"
    surface = modes.MODES["reconstructed_surface_distance"]
    assert set(surface.variance_components) == {"geometry_m2", "sensor_m2"} and surface.variance_required
    # What remains open: intrinsic readings keep one sensor sigma by design, and no mesh conversion fills geometry_m2.
    assert not modes.MODES["intrinsic_geodesic_distance"].variance_components
    assert "mesh" in t045["recommended_next_task"] and "geometry_m2" in t045["recommended_next_task"]
    step = reports["T044"]["recommended_next_task"]
    assert step == M.NEXT_STEPS["T044"] and "Partly delivered by T045" in step and "does not carry" not in step


# ---------------------------------------------------------------- T044
@pytest.mark.lab_task("T044")
def test_law_of_total_variance_split():
    study = S.variance_split_study(geometry_sigmas=(1e-3,), sensor_sigmas=(5e-4,), outer=300, inner=8, fresh=4000,
                                   repeats=(1, 64))
    scenario = study["scenarios"][0]
    assert abs(scenario["anova_residual"]) <= 1e-10
    assert abs(scenario["fresh_ratio"] - 1) <= 4 * math.sqrt(2 / 3999)
    assert scenario["geometry_share"] > 0.5
    floor = (study["gain"] * 1e-3) ** 2
    assert study["averaging"]["rows"][-1]["variance"] >= 0.9 * floor
    split = study["gain_split"]
    assert split["tangential"] ** 2 + split["normal"] ** 2 == pytest.approx(split["total"] ** 2, rel=1e-12)
    assert split["tangential"] > split["normal"] and split["marker"] > split["interior"]
    normal = study["normal_only"]
    assert normal["geometry_share"] < 0.5
    assert normal["geometry_variance_mc"] == pytest.approx(normal["geometry_variance_linear"], rel=0.1)


@pytest.mark.lab_task("T044")
def test_variance_split_task_report(reports):
    report = reports["T044"]
    _completed_with_unestablished_physics(report)
    dominance = _value(report, "Under isotropic vertex noise with barycentric markers")
    assert dominance["value"]["baseline"]["geometry_share"] > 0.5
    normal = _value(report, "Under normal-only (shape) vertex noise")
    assert normal["value"]["geometry_share"] < 0.5 and "counterexample" in normal
    averaging = _value(report, "Averaging repeated sensor readings")
    assert averaging["counterexample"]["statement"].startswith("Averaging repeated measurements")
    domains = {f["domain"] for f in report["findings"] if f["evidence_status"] == "not_established"}
    assert domains == {"sensor_performance", "physical"}
    # Identities that hold for any data are sanity values in the prose, not evidence.
    assert not any("sums of squares" in f["claim"] for f in report["findings"])
    references = " ".join(c["reference"] for f in report["findings"] for c in f["basis"].get("checks", []))
    assert "tangential^2 + normal^2" not in references and "best case" not in references
    corridors = S.corridor_study()
    narrowest = min(corridors["rows"], key=lambda r: r["vertex_margin"])
    assert f"{narrowest['left_fraction'][corridors['sigmas'].index(1e-3)]:.1%}" in report["uncertainty"]
