import copy
import math

import numpy as np
import pytest

from ciw.lab import runner
from ciw.lab import surfaces_discrete_mesh as M
from ciw.lab import surfaces_discrete_mesh_geometry as G
from ciw.lab import surfaces_discrete_mesh_studies as S
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, validate_finding
from ciw.lab.registry import load_queue, module_implementations
from ciw.lab.report import validate_report

TASKS = ("T038", "T039", "T040", "T041", "T042", "T043", "T044")
IMPLEMENTATIONS = module_implementations("surfaces_discrete_mesh")
DIJKSTRA_CLAIM = ("Heap Dijkstra edge-graph distances agree with a dense Floyd-Warshall recomputation (and "
                  "scipy.sparse.csgraph when installed)")
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


def test_solver_task_report(reports):
    report = reports["T038"]
    # Partial: an exact two-point polyhedral geodesic (MMP/ICH) is not delivered, and the report says so.
    _completed_with_unestablished_physics(report, state="partial")
    assert any(a.startswith("Partial delivery") for a in report["unresolved_assumptions"])
    steiner = _value(report, "Nested Steiner-graph distances never increase with k")
    assert steiner["value"]["edges_outside_faces"] == 0
    assert "chord" not in " ".join(c["reference"] for c in steiner["basis"]["checks"])
    labels = _labels(report)
    scipy_present = S.optional_version("scipy") is not None
    assert labels[DIJKSTRA_CLAIM] == ("independently_verified" if scipy_present else "numerically_verified")
    floyd = _value(report, "Heap Dijkstra")["basis"]["checks"][0]
    assert floyd["reference_kind"] == "cross_implementation"
    plane = "Straightest geodesics on sheared planar meshes coincide with straight lines"
    assert labels[plane] == "numerically_verified"
    physical = [f for f in report["findings"] if f["domain"] == "physical"]
    assert physical and physical[0]["evidence_status"] == "not_established"
    assert any(a["path"].endswith("sphere-traces.json") for a in report["generated_artifacts"])


def test_dijkstra_check_falls_back_without_scipy(monkeypatch, tmp_path, reports):
    monkeypatch.setattr(S, "optional_version", lambda name: None)
    result = S.dijkstra_independent(level=1)
    assert "scipy_max_abs" not in result and result["floyd_max_abs"] <= 1e-12
    report = _run("T038", runner.Context(tmp_path))
    assert _labels(report)[DIJKSTRA_CLAIM] == "numerically_verified"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    # The prose must not change with the optional checker, or the regression gate flags wording.
    assert runner._skeleton(report["numerical_result"]) == runner._skeleton(reports["T038"]["numerical_result"])


# ---------------------------------------------------------------- T039
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


def test_rank_correlation_matches_average_ranks():
    assert M._average_ranks([3.0, 1.0, 3.0, 2.0]).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    stats = pytest.importorskip("scipy.stats")
    x = [0.3, 1.2, 0.3, 5.0, 2.2, 2.2, 0.9]
    y = [1.0, 0.5, 0.7, 3.0, 3.0, 2.0, 0.1]
    assert M.spearman(x, y) == pytest.approx(float(stats.spearmanr(x, y).statistic), abs=1e-12)


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


def test_next_steps_name_forward_work(reports):
    """A completed task's next step is its own open question, never a queue task that already ran."""
    for task_id, report in reports.items():
        text = report["recommended_next_task"]
        assert text == M.NEXT_STEPS[task_id], task_id
        if report["state"] == "completed":
            assert text.startswith("Deferred research question: "), (task_id, text)
    assert reports["T038"]["recommended_next_task"].startswith("Complete T038: ")
    assert "T039" not in reports["T038"]["recommended_next_task"]
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
