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
RANK_CLAIM = ("Maximum radius ratio ranks the mixed-Voronoi curvature error and the geodesic error across the valid "
              "642-vertex meshes")


def _run(task_id, ctx):
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    return validate_report(runner.run_task(queue[task_id], IMPLEMENTATIONS[task_id], ctx, {}))


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Run the section once through the runner with one shared context (memoized studies)."""
    ctx = runner.Context(tmp_path_factory.mktemp("lab-mesh"))
    return {tid: _run(tid, ctx) for tid in TASKS}


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _value(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def _completed_with_unestablished_physics(report):
    assert report["state"] == "completed"
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
    assert nested["max_increase_with_k"] <= 1e-12 and nested["max_chord_excess"] <= 1e-12
    assert nested["min_gap"] >= -1e-12
    assert np.mean(nested["gap_to_traced"]["3"]) <= np.mean(nested["gap_to_traced"]["1"])
    independent = S.dijkstra_independent(level=2)
    assert independent["floyd_max_abs"] <= 1e-12
    if independent["scipy"]:
        assert independent["scipy_max_abs"] <= 1e-12
    mesh = G.icosphere(1)
    chord = np.linalg.norm(mesh.vertices - mesh.vertices[0], axis=1)
    assert np.all(G.edge_distances(mesh, 0) >= chord - 1e-12)  # every surface path is at least the chord


def test_solver_task_report(reports):
    report = reports["T038"]
    _completed_with_unestablished_physics(report)
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
    folded = _value(report, "Jittered meshes are refused as folded exactly when a face is inverted")
    refusals = [c for c in folded["basis"]["checks"] if c["reference_kind"] == "refusal"]
    inverted = folded["value"]["inverted_faces"]
    assert len(refusals) == len(inverted) == 16
    for check, count in zip(refusals, inverted):
        assert check["expected_refusal"] == ("folded_face" if count > 0 else "none")
        assert check["observed_refusal"] == check["expected_refusal"]
    assert sum(count > 0 for count in inverted) == 6
    within = _value(report, "Within the latitude-longitude family")
    assert within["value"]["better_quality"]["mesh"] == "uv-sphere-40x16"
    voronoi = _value(report, "With the mixed Voronoi area the regular icosphere")
    assert voronoi["evidence_status"] == "numerically_verified"
    lantern = _value(report, "A strongly pleated lantern")
    assert lantern["regression_tolerance"] == {"abs": 0, "rel": 0}
    authority = [f for f in report["findings"] if f["domain"] == "production_acceptance"]
    assert authority and authority[0]["evidence_status"] == "not_established"


def test_rank_finding_without_scipy(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "optional_version", lambda name: None)
    report = _run("T041", runner.Context(tmp_path))
    assert _labels(report)[RANK_CLAIM] == "numerically_verified"
    assert report["evidence_status"]["primary"] == "numerically_verified"


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
    assert any("Self-intersections" in a or "self-intersections" in a for a in report["unresolved_assumptions"])


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
    study = S.corridor_study(sigmas=(1e-3,), samples=1000)
    fractions = {r["start_index"]: r["left_fraction"][0] for r in study["rows"]}
    margins = {r["start_index"]: r["vertex_margin"] for r in study["rows"]}
    assert max(margins, key=margins.get) == S.MARKER_START and fractions[S.MARKER_START] == 0.0
    assert fractions[min(margins, key=margins.get)] > 0.1


def test_uncertainty_task_report(reports):
    report = reports["T043"]
    _completed_with_unestablished_physics(report)
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
