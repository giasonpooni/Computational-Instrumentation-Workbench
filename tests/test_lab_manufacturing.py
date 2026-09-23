import importlib.util
import json
import math

import numpy as np
import pytest

from ciw.lab import manufacturing as mfg
from ciw.lab import manufacturing_geometry as geo
from ciw.lab import manufacturing_metrology as met
from ciw.lab import manufacturing_records as rec
from ciw.lab import runner
from ciw.lab.evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, finding
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report

TASK_IDS = [f"T{n}" for n in range(126, 142)]


@pytest.fixture(scope="module")
def section(tmp_path_factory):
    """Run T126-T141 once in queue order with one shared context (memoized studies)."""
    directory = tmp_path_factory.mktemp("lab-manufacturing")
    implementations = section_implementations("manufacturing")
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    ctx = runner.Context(directory)
    reports = {task_id: validate_report(runner.run_task(queue[task_id], implementations[task_id], ctx, {}))
               for task_id in TASK_IDS}
    return directory, reports


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert matches, f"no finding starting with {prefix!r} in {report['task_id']}"
    return matches[0]


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def test_every_task_is_registered_and_reports_honestly(section):
    _, reports = section
    assert set(section_implementations("manufacturing")) == set(TASK_IDS)
    for task_id, report in reports.items():
        expected = "partial" if task_id == "T138" else "completed"
        assert report["state"] == expected, (task_id, report["experiment"])
        assert report["physical_validation_status"]["status"] == "not_established"
        assert not report.get("tests_failed")
        for record in report["findings"]:
            if record["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS:
                assert record["evidence_status"] == "not_established"
            if record["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS:
                assert "regression_tolerance" in record, record["claim"]
                assert record["uncertainty"] is not None, record["claim"]
        assert any(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS for f in report["findings"]), task_id
        for name in ("hypothesis", "mathematical_model", "experiment", "numerical_result", "recommended_next_task"):
            assert isinstance(report[name], str) and report[name].strip()


def test_flat_plate_protocol_is_a_zero_curvature_control(section):
    directory, reports = section
    report = reports["T126"]
    gap = _finding(report, "Flat-plate control: chord and geodesic")
    assert gap["evidence_status"] == "numerically_verified" and gap["value"] <= 1e-9
    phi = _finding(report, "Flat-plate Jacobi transfer")
    assert np.allclose(phi["value"], [[1.0, 240.0], [0.0, 1.0]], atol=1e-9)
    assert _labels(report)["Measured marker chords on the physical plate equal the predicted geodesic distances "
                           "within instrument uncertainty"] == "not_established"
    protocol = rec.validate_protocol(json.loads(
        (directory / "artifacts" / "T126" / "protocol-flat-plate.json").read_text(encoding="utf-8")))
    assert protocol["hardware_measured"] == {"status": "not_acquired", "records": []}
    assert protocol["production_acceptance"] == "outside_system"
    heading = next(q for q in protocol["predicted_quantities"] if q["id"] == "P3")
    assert heading["value"][-1] == pytest.approx(1.2, abs=1e-12)
    assert heading["evidence_status"] == phi["evidence_status"]


def test_protocols_refuse_filled_slots_and_decisions(section):
    directory, reports = section
    for task_id, name in (("T126", "protocol-flat-plate.json"), ("T127", "protocol-rolled-cylinder.json"),
                          ("T128", "protocol-domed-coupon.json")):
        protocol = json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))
        matrix = rec.protocol_refusal_matrix(protocol)
        assert all(case["observed"] == case["expected"] for case in matrix.values()), matrix
        validated = [f for f in reports[task_id]["findings"] if f["claim"].endswith("refuses malformed variants")]
        assert validated[0]["value"] == len(matrix) == 7
        assert validated[0]["evidence_status"] == "numerically_verified"
    acquired = dict(protocol, hardware_measured={"status": "acquired", "records": [{"acquisition": {
        "device": "camera:1", "raw_sha256": "a" * 64, "acquired_at": "2026-09-23T00:00:00Z", "calibration": "CERT"}}]})
    assert rec.validate_protocol(acquired)["hardware_measured"]["status"] == "acquired"


def test_cylinder_protocol_predicts_chord_geodesic_gaps(section):
    _, reports = section
    report = reports["T127"]
    gaps = _finding(report, "Rolled-cylinder chord-geodesic gaps")["value"]
    radius = geo.CYLINDER_RADIUS
    assert gaps["circumferential 90 deg"] == pytest.approx(radius * math.pi / 2 - 2 * radius * math.sin(math.pi / 4), rel=1e-12)
    assert gaps["axial 100 mm"] == pytest.approx(0.0, abs=1e-12)
    counter = _finding(report, "A marker chord differs from the surface distance")
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    resolvable = _finding(report, "Minimum circumferential marker separation")["value"]
    assert resolvable["camera"] == pytest.approx(23.86, abs=0.01)
    assert resolvable["cmm"] < resolvable["tracker"] < resolvable["camera"]
    assert _finding(report, "Rolled-cylinder Jacobi transfer equals")["value"] <= 1e-9


def test_coupon_protocol_predicts_focal_crossing(section):
    _, reports = section
    report = reports["T128"]
    focal = _finding(report, "Laterally offset routes cross the nominal route")
    assert focal["evidence_status"] == "numerically_verified"
    assert focal["value"] == pytest.approx(153.407, abs=2e-3)
    study = mfg.nominal_study()
    assert study["nonlinear_crossing_mm"] == pytest.approx(154.36, abs=0.01)
    assert study["crossing_extrapolated_mm"] == pytest.approx(focal["value"], abs=0.02)
    orders = _finding(report, "Linearized separation remainder")["value"]
    assert orders["on_axis"] == pytest.approx(3.0, abs=0.1) and orders["off_axis"] == pytest.approx(2.0, abs=0.1)
    signature = _finding(report, "Curvature signature")["value"]
    assert signature["coupon_lateral_2mm"] < 0 < signature["plate_lateral_2mm"]
    assert _labels(report)["The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance"] \
        == "not_established"
    has = {name: importlib.util.find_spec(name) is not None for name in ("scipy", "sympy")}
    integrator = _finding(report, "The RK4 coupon transfer agrees")
    geometry = _finding(report, "Coupon Christoffel symbols and Gaussian curvature agree")
    assert integrator["evidence_status"] == ("independently_verified" if has["scipy"] else "numerically_verified")
    assert geometry["evidence_status"] == ("independently_verified" if has["sympy"] else "numerically_verified")
    assert integrator["value"] < 1e-4


def test_coupon_independent_checks_fall_back_to_same_origin_labels(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name in ("scipy", "sympy") else real(name, *args))
    result = mfg.independent_coupon_checks(mfg.nominal_study()["length_mm"])
    assert result["integrator"]["checker"] is None and result["geometry"]["checker"] is None
    basis = mfg._independent_basis(result["integrator"], "fallback", result["integrator"]["max_difference"], 1e-4)
    record = finding("fallback agreement", "numerical", 0.0, basis)
    assert record["evidence_status"] == "numerically_verified"
    assert result["geometry"]["max_christoffel_difference"] < 1e-7


def test_metrology_sampling_design_and_counterexamples(section):
    _, reports = section
    report = reports["T129"]
    design = _finding(report, "Sampling design resolving profile curvature")
    assert design["evidence_status"] == "numerically_verified"
    assert design["value"]["coupon crest"]["window_mm"] == pytest.approx(13.66, abs=0.01)
    assert _finding(report, "Repeat scans needed")["value"] == {"5%": 1, "2%": 5, "1%": 109}
    circle = _finding(report, "The osculating-circle window rule")
    assert circle["counterexample"]["witness"]["quartic_ratio_gauss_over_circle"] == pytest.approx(4.0)
    assert circle["value"] > 1.0
    window = _finding(report, "A smaller fitting window")
    assert window["value"]["rms_at_one_third_window"] > 2 * window["value"]["rms_at_design_window"]
    assert _finding(report, "Monte Carlo curvature error")["evidence_status"] == "numerically_verified"
    assert _finding(report, "A real laser line scanner")["evidence_status"] == "not_established"
    # The design rule is explicit: halve the tolerance between bias and k = 2 noise.
    rule = met.sampling_design(0.01, 1.25e-7, 5e-4, 0.01)
    assert met.curvature_bias_series(rule["window_mm"], 1.25e-7) == pytest.approx(2.5e-4)
    assert 2 * met.curvature_noise_std_continuum(rule["window_mm"], rule["max_spacing_mm"], 0.01) == pytest.approx(2.5e-4)


def test_artifacts_datum_frames_and_chain_covariance(section):
    _, reports = section
    report = reports["T130"]
    for prefix in ("First-order covariance of the INSTRUMENT->CAD", "The 3-2-1 datum frame", "Gauge-sphere fit",
                   "Step-gauge fit"):
        assert _finding(report, prefix)["evidence_status"] == "numerically_verified", prefix
    assert _finding(report, "The physical gauge sphere")["domain"] == "calibration"
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.datum_frame_321([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0]], [0, 1, 0])
    assert refused.value.code == "datum_degenerate"
    pose = met.transform(met.exp_so3([0.1, -0.2, 0.3]), [1.0, 2.0, 3.0])
    assert np.allclose(met.pose_difference(met.exp_se3(np.array([1e-4, 0, 0, 0, 0, 2e-5])) @ pose, pose),
                       [1e-4, 0, 0, 0, 0, 2e-5], atol=1e-8)  # first order: the left Jacobian adds 1e-9


def test_gage_rr_recovers_components_and_refuses_unbalanced(section):
    _, reports = section
    report = reports["T131"]
    recovery = _finding(report, "ANOVA Gage R&R recovers")
    assert recovery["evidence_status"] == "numerically_verified"
    assert recovery["value"]["repeatability"] == pytest.approx(1e-4, rel=0.02)
    spread = _finding(report, "Sampling spread of %GRR")["value"]
    assert spread["0.05"] < spread["true"] < spread["0.95"] and spread["true"] == pytest.approx(23.94, abs=0.01)
    assert _finding(report, "Gage R&R refuses")["value"] == 2
    assert _labels(report)["The measurement system is approved for production use"] == "not_established"
    exact = np.zeros((2, 2, 2))
    exact[1] += 1.0
    result = met.gage_rr_anova(exact)
    assert result["ss"]["part"] == pytest.approx(2.0) and result["ss"]["error"] == 0.0


def test_placement_curvature_stack_and_radius_counterexample(section):
    _, reports = section
    report = reports["T132"]
    kg = _finding(report, "Geodesic curvature of a 30-to-60 degree")["value"]
    assert kg["min_steering_radius_mm"] == pytest.approx(300 / (math.pi / 6 * math.cos(math.pi / 6)), rel=1e-9)
    stack = _finding(report, "Tolerance stack of the placed course")["value"]
    assert stack["length_max_mm"] == pytest.approx(409.83, abs=0.01)
    counter = _finding(report, "Programming a helix in machine angles")
    assert counter["value"] == pytest.approx(1000 * math.sin(math.atan(1.002) - math.pi / 4), rel=1e-12)
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    assert mfg.placement_deviation(500.0, math.pi / 4, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_winding_clairaut_sensitivity_and_slippage(section):
    _, reports = section
    report = reports["T133"]
    assert _finding(report, "The Clairaut constant is conserved")["evidence_status"] == "numerically_verified"
    amplification = _finding(report, "Heading-error amplification")["value"]
    assert amplification["psi70_passing"] > 2.5 and amplification["psi50_bounded"] < 1.0
    slip = _finding(report, "Slippage tendency")["value"]
    assert slip["psi50"]["max_ratio"] == pytest.approx(0.4436, abs=1e-3) and slip["psi70"]["fraction_above_mu"] == 0.0
    counter = _finding(report, "A constant winding angle is not geodesic")
    assert counter["evidence_status"] == "numerically_verified" and "counterexample" in counter
    turn = _finding(report, "Clairaut sensitivity predicts")["value"]
    assert math.degrees(turn["turn_rad"]) == pytest.approx(115.39, abs=0.01)


def test_coating_standoff_and_offset_cusp(section):
    _, reports = section
    report = reports["T134"]
    envelope = _finding(report, "Lateral-error envelope")["value"]
    assert envelope["max_mm"] == pytest.approx(0.4201, abs=1e-4)
    cusp = _finding(report, "A spray standoff beyond the concave radius")
    assert cusp["value"]["min_concave_radius_mm"] < cusp["value"]["spray_standoff_mm"]
    assert cusp["value"]["reversed_segments"] > 0 and "counterexample" in cusp
    assert _labels(report)["The trajectory is safe to execute on a welding or coating robot cell"] == "not_established"
    u = np.array([-20.0, 5.0])
    t = geo.COUPON.unit_tangent(u, 0.3)
    exact = geo.standoff_error_exact(geo.COUPON, u, geo.lateral_direction3(geo.COUPON, u, t), 0.01, 15.0)
    n = geo.COUPON.normal(u, t)
    assert exact == pytest.approx(-0.5 * float(n @ geo.second_fundamental_form(geo.COUPON, u) @ n) * 1e-4, rel=1e-3)


def test_scan_plans_coverage_and_counterexample(section):
    _, reports = section
    report = reports["T135"]
    plans = _finding(report, "Coverage and path length")["value"]
    assert plans["geodesic x1"]["coverage"] < 0.99 < plans["geodesic x0.6"]["coverage"]
    counter = _finding(report, "Geodesic rows at the swath spacing leave gaps")
    assert counter["counterexample"]["witness"]["plate_coverage"] == 1.0
    shortest = _finding(report, "Shortest evaluated scan plan")["value"]
    assert shortest["plan"] == "chart-parallel x0.9"
    # Two abutting 2 mm swaths cover a 10 x 4 mm plate exactly; one covers half of it.
    rows = [np.column_stack([np.linspace(0, 10, 11), np.full(11, y), np.zeros(11)]) for y in (0.0, 2.0)]
    sample = geo.area_sample(geo.PLATE, (0.0, 10.0), (-1.0, 3.0), 400, columns=5)
    assert geo.coverage_fraction(sample, rows, 2.0) == pytest.approx(1.0)
    assert geo.coverage_fraction(sample, rows[:1], 2.0) == pytest.approx(0.5, abs=0.02)


def test_rankings_by_calibration_tolerance_and_focus_margin(section):
    _, reports = section
    calibration = _finding(reports["T136"], "Candidate coupon routes ranked by the heading calibration tolerance")
    assert calibration["evidence_status"] == "numerically_verified"
    assert calibration["value"]["ranking"][0] == "fan+0deg"
    focus = _finding(reports["T137"], "Candidate coupon routes ranked by focus margin")
    assert focus["value"]["ranking"][-1] == "fan+0deg"
    assert focus["value"]["margin"]["fan+0deg"] == pytest.approx(0.7588, abs=1e-4)
    counter = _finding(reports["T137"], "The shortest candidate route has the worst focus margin")
    assert counter["evidence_status"] == "numerically_verified"
    assert counter["counterexample"]["witness"]["route"] == "fan+0deg"
    variation = _finding(reports["T137"], "The second variation of route length")["value"]
    assert variation["finite_difference_mm"] == pytest.approx(variation["index_form_mm"], rel=2e-3)
    assert _finding(reports["T136"], "The robot, fixture and frame calibration")["evidence_status"] == "not_established"


def test_predicted_separation_has_no_measured_counterpart(section):
    _, reports = section
    report = reports["T138"]
    assert report["state"] == "partial"
    predicted = _finding(report, "Predicted separation of the 2 mm offset route")
    assert predicted["evidence_status"] == "numerically_verified"
    assert predicted["value"]["separation_mm"][0] == pytest.approx(2.0, abs=1e-6)
    assert predicted["value"]["separation_mm"][-1] == pytest.approx(-1.0374, abs=1e-4)
    assert _finding(report, "The comparison refuses")["value"] == 3
    assert _finding(report, "Measured separation on the coupon agrees")["evidence_status"] == "not_established"
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_separation({"separation_mm": [1.0], "expanded_uncertainty_mm": [0.1]}, None)
    assert refused.value.code == "measurement_absent"
    assert rec.normalized_error([1.1], [1.0], [0.06], [0.08])[0] == pytest.approx(1.0)


def test_retention_schema_refusals_and_fixture_boundary(section):
    _, reports = section
    report = reports["T139"]
    schema = _finding(report, "The retention schema refuses")
    assert schema["value"] == 9 and schema["evidence_status"] == "numerically_verified"
    assert _finding(report, "A schema fixture, or a record without its raw bytes")["value"] == 2
    assert _finding(report, "A real measurement with raw bytes")["evidence_status"] == "not_established"
    fixture, raw = mfg._schema_fixture()
    assert rec.refusal_code(rec.to_acquisition, fixture, raw) == "fixture_is_not_measurement"
    measured = dict(fixture, record_kind="measurement")
    acquisition = rec.to_acquisition(measured, raw)
    # Only a real measurement record could make a physical finding hardware_measured; a fixture never does.
    assert finding("probe", "physical", 1.0, {"acquisition": acquisition})["evidence_status"] == "hardware_measured"
    assert rec.refusal_code(rec.validate_retention, dict(fixture, clock=dict(fixture["clock"], acquired_at="2026-09-23 00:00"))) \
        == "clock_without_timezone"


def test_uncertainty_budget_classifies_limiting_terms(section):
    _, reports = section
    report = reports["T140"]
    budget = _finding(report, "Uncertainty budget per predicted quantity")["value"]
    assert budget["cylinder gap, 90 deg pair"]["dominant"] == "instrument"
    assert budget["coupon focal distance"]["dominant"] == "geometry"
    assert budget["coarse-solver control (8 RK4 steps)"]["dominant"] == "solver"
    counter = _finding(report, "The coupon focal-distance prediction is geometry-limited")
    assert counter["value"] > 0.5 and "counterexample" in counter
    assert mfg._classify({"instrument": 1.0, "geometry": 1.0, "solver": 0.0})[0] == "mixed (largest: instrument)"


def test_production_acceptance_stays_outside_the_system(section):
    _, reports = section
    report = reports["T141"]
    api = _finding(report, "The lab API cannot mark production acceptance")
    assert api["evidence_status"] == "numerically_verified" and api["value"] == 64 * len(AUTHORITY_DOMAINS)
    assert _finding(report, "Production acceptance of the coupon")["evidence_status"] == "not_established"
    policy = rec.AcceptancePolicy()
    with pytest.raises(rec.RecordRefusal) as refused:
        policy.decide({"part": "coupon-001", "decision": "accept"})
    assert refused.value.code == "production_acceptance_outside_system"
    assert policy.record({"part": "x"})["decision"] == "not_performed"
    for task_id, report in reports.items():
        for record in report["findings"]:
            if record["domain"] == "production_acceptance":
                assert record["evidence_status"] == "not_established", task_id
