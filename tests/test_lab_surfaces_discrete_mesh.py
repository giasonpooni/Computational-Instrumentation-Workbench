import math

import numpy as np
import pytest

from ciw.lab import runner
from ciw.lab import surfaces_discrete_mesh  # noqa: F401  (registers T038-T044)
from ciw.lab import surfaces_discrete_mesh_geometry as G
from ciw.lab import surfaces_discrete_mesh_studies as S
from ciw.lab.evidence import validate_finding
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.report import validate_report

TASKS = ("T038", "T039", "T040", "T041", "T042", "T043", "T044")


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Run the section once through the runner with one shared context (memoized studies)."""
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    ctx = runner.Context(tmp_path_factory.mktemp("lab-mesh"))
    return {tid: validate_report(runner.run_task(queue[tid], _REGISTRY[tid], ctx, {})) for tid in TASKS}


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _value(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def _completed_with_unestablished_physics(report):
    assert report["state"] == "completed"
    assert report["physical_validation_status"]["status"] == "not_established"
    for record in report["findings"]:
        validate_finding(record)
        if record["domain"] in ("physical", "calibration", "sensor_performance", "production_acceptance"):
            assert record["evidence_status"] == "not_established"
        else:
            assert record["evidence_status"] != "not_established"
            if isinstance(record["value"], (int, float)) and not isinstance(record["value"], bool):
                assert "regression_tolerance" in record


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
    assert labels["Heap Dijkstra edge-graph distances agree with an independent shortest-path implementation"] == (
        "independently_verified" if scipy_present else "numerically_verified")
    plane_claim = "Straightest geodesics on sheared planar meshes coincide with straight lines"
    assert labels[plane_claim] == "numerically_verified"
    physical = [f for f in report["findings"] if f["domain"] == "physical"]
    assert physical and physical[0]["evidence_status"] == "not_established"
    assert any(a["path"].endswith("sphere-traces.json") for a in report["generated_artifacts"])


def test_dijkstra_check_falls_back_without_scipy(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "optional_version", lambda name: None)
    result = S.dijkstra_independent(level=1)
    assert "scipy_max_abs" not in result and result["floyd_max_abs"] <= 1e-12
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    report = runner.run_task(queue["T038"], _REGISTRY["T038"], runner.Context(tmp_path), {})
    claim = "Heap Dijkstra edge-graph distances agree with an independent shortest-path implementation"
    assert _labels(validate_report(report))[claim] == "numerically_verified"


# ---------------------------------------------------------------- T039
def test_convergence_orders_on_small_levels():
    traces = S.sphere_trace_study(levels=(2, 3, 4), starts=S.STARTS[:3])
    order = S.fitted_order([r["h"] for r in traces["rows"]], [r["mean_length_defect"] for r in traces["rows"]])
    assert abs(order - 2.0) <= 0.3
    cylinder = S.cylinder_study(ns=(16, 32, 64))
    helix = S.fitted_order([r["h"] for r in cylinder["rows"]], [r["helix_error"] for r in cylinder["rows"]])
    assert abs(helix - 2.0) <= 0.1
    assert cylinder["rows"][-1]["scaled_error"] == pytest.approx(cylinder["leading_constant"], rel=0.03)
    distances = S.distance_study(levels=(1, 2, 3), ks=(1,), heat_levels=(1, 2, 3))
    edge = [r["edge_max_rel"] for r in distances["rows"]]
    assert edge[-1] >= 0.2 and edge[-1] >= edge[0]  # no convergence of edge-graph distance
    heat = [r["heat_max_abs"] for r in distances["rows"]]
    assert heat[2] < heat[1] < heat[0]


def test_convergence_task_report(reports):
    report = reports["T039"]
    _completed_with_unestablished_physics(report)
    length = _value(report, "Traced-geodesic length defect")
    assert length["evidence_status"] == "numerically_verified"
    assert abs(length["value"] - 2.0) <= 0.2
    floor = _value(report, "Edge-graph Dijkstra distance keeps")
    assert floor["counterexample"]["statement"].startswith("Edge-graph shortest paths converge")
    assert min(floor["value"]["max_relative_error"][1:]) >= 0.2
    helix = _value(report, "Prism-cylinder helix")
    assert abs(helix["value"]["scaled_to_leading_constant"] - 1.0) <= 0.02


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


def test_jacobi_task_report(reports):
    report = reports["T040"]
    _completed_with_unestablished_physics(report)
    flat = _value(report, "At fixed mesh a 1e-5 heading offset")
    assert flat["evidence_status"] == "numerically_verified" and "counterexample" in flat
    assert flat["value"]["identical_face_sequences"] == flat["value"]["pairs"] == 30
    assert flat["value"]["error_vs_smooth"] == pytest.approx(2.0 - math.sin(2.0), rel=1e-12)
    valence = _value(report, "Angle-defect curvature at valence-5")
    assert valence["value"]["valence5_values"][-1] == pytest.approx(4.5 - 1.5 * math.sqrt(5), abs=1e-4)


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
    assert study["folded"]["issues"][0] == "folded_face"


def test_quality_task_report(reports):
    report = reports["T041"]
    _completed_with_unestablished_physics(report)
    labels = _labels(report)
    spearman = "Spearman rank correlation of minimum angle with each error over valid 642-vertex meshes"
    assert labels[spearman] == "synthetic"  # descriptive statistic without a reference check
    counterexamples = [f for f in report["findings"] if "counterexample" in f]
    assert len(counterexamples) >= 5
    folded = _value(report, "Jittered meshes with folded faces")
    refusals = [c for c in folded["basis"]["checks"] if c["reference_kind"] == "refusal"]
    assert refusals and all(c["observed_refusal"] == "folded_face" for c in refusals)
    authority = [f for f in report["findings"] if f["domain"] == "production_acceptance"]
    assert authority and authority[0]["evidence_status"] == "not_established"


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


def test_refusal_task_report(reports):
    report = reports["T042"]
    _completed_with_unestablished_physics(report)
    refusals = _value(report, "Every declared surface-data defect")
    checks = refusals["basis"]["checks"]
    assert len(checks) >= 20 and all(c["reference_kind"] == "refusal" and c["passed"] for c in checks)
    assert refusals["evidence_status"] == "numerically_verified"
    assert refusals["value"]["aimed-at-vertex"] == "vertex_hit"


# ---------------------------------------------------------------- T043
def test_linearization_matches_monte_carlo_and_breaks():
    study = S.uncertainty_study(level=3, sigmas=(1e-3, 1e-2), samples=2000)
    rows = {o["observable"]: o["rows"] for o in study["observables"]}
    assert abs(rows["marker geodesic distance"][0]["ratio"] - 1) <= 0.12
    assert rows["marker geodesic distance"][0]["invalid_fraction"] == 0.0
    assert rows["marker geodesic distance"][1]["invalid_fraction"] > 0.1
    for name, values in rows.items():
        if name.startswith("vertex normal"):
            assert all(abs(r["ratio"] - 1) <= 0.12 for r in values)
        if name.startswith("angle-defect"):
            assert abs(values[0]["ratio"] - 1) <= 0.12 and values[1]["ratio"] >= 1.25


def test_uncertainty_task_report(reports):
    report = reports["T043"]
    _completed_with_unestablished_physics(report)
    slopes = _value(report, "Sensitivity to vertex noise")["value"]
    assert slopes["angle-defect curvature at valence-5 vertex 0"] == pytest.approx(-2.0, abs=0.25)
    growth = _value(report, "Under fixed vertex noise")
    assert growth["value"]["total_rms_error"][-1] > 10 * growth["value"]["total_rms_error"][0]
    calibration = [f for f in report["findings"] if f["domain"] == "calibration"]
    assert calibration and calibration[0]["evidence_status"] == "not_established"


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


def test_variance_split_task_report(reports):
    report = reports["T044"]
    _completed_with_unestablished_physics(report)
    dominance = _value(report, "Geometry uncertainty dominates")
    assert dominance["value"]["baseline"]["geometry_share"] > 0.5
    averaging = _value(report, "Averaging repeated sensor readings")
    assert averaging["counterexample"]["statement"].startswith("Averaging repeated measurements")
    domains = {f["domain"] for f in report["findings"] if f["evidence_status"] == "not_established"}
    assert domains == {"sensor_performance", "physical"}
