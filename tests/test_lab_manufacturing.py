from copy import deepcopy
import hashlib
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
from ciw.lab.evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, finding
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


def _gap_at(row, radius):
    """Chord-geodesic gap of a cylinder marker pair at another radius (same angles and heights)."""
    return math.hypot(radius * row["dphi_rad"], row["dz_mm"]) - math.hypot(2 * radius * math.sin(row["dphi_rad"] / 2),
                                                                         row["dz_mm"])


def _measurement_record(data):
    """A retention record of kind measurement that is not the schema fixture (synthetic bytes, never retained)."""
    fixture, _ = mfg._schema_fixture()
    record = deepcopy(fixture)
    record.update(record_kind="measurement", instrument=dict(fixture["instrument"], serial="SN-1"),
                  raw=[{"name": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value),
                        "media_type": "application/octet-stream"} for name, value in data.items()])
    record["calibration"] = dict(fixture["calibration"], sha256="c" * 64)
    return record


def test_every_task_is_registered_and_reports_honestly(section):
    _, reports = section
    assert set(section_implementations("manufacturing")) == set(TASK_IDS)
    for task_id, report in reports.items():
        # T138 has no measurement to compare and T139 none to retain: both are partial until hardware exists.
        expected = "partial" if task_id in ("T138", "T139") else "completed"
        assert report["state"] == expected, (task_id, report["experiment"])
        assert report["physical_validation_status"]["status"] == "not_established"
        assert not report.get("tests_failed")
        for record in report["findings"]:
            if record["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS:
                assert record["evidence_status"] == "not_established"
            if record["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and not record.get("expected_not_established"):
                assert "regression_tolerance" in record, record["claim"]
                assert record["uncertainty"] is not None, record["claim"]
            assert "|" not in record["claim"], record["claim"]  # claims are table cells in the rendered report
        assert any(f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS for f in report["findings"]), task_id
        # Every section task passes its findings through the acceptance-language screen.
        assert rec.screen_acceptance_language(report["findings"])
        assert hasattr(section_implementations("manufacturing")[task_id].run, "__wrapped__")
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
    # The geodesic distance is solved for: Newton starts 0.1 rad off the chord direction and converges onto it.
    rows = mfg.plate_study()["pairs"]
    assert min(r["shooting_iterations"] for r in rows) >= 3
    assert max(r["heading_minus_chord_direction_rad"] for r in rows) < 1e-12
    shot = mfg.shoot_geodesic(geo.PLATE, [0.0, 0.0], [30.0, 40.0], 0.0, 100.0)
    assert shot["arclength_mm"] == pytest.approx(50.0, abs=1e-9)
    assert shot["heading_rad"] == pytest.approx(math.atan2(40.0, 30.0), abs=1e-12)


def test_protocols_refuse_filled_slots_and_decisions(section):
    directory, reports = section
    for task_id, name in (("T126", "protocol-flat-plate.json"), ("T127", "protocol-rolled-cylinder.json"),
                          ("T128", "protocol-domed-coupon.json"), ("T129", "protocol-surface-scan.json")):
        protocol = json.loads((directory / "artifacts" / task_id / name).read_text(encoding="utf-8"))
        matrix = rec.protocol_refusal_matrix(protocol)
        assert all(case["observed"] == case["expected"] for case in matrix.values()), matrix
        validated = [f for f in reports[task_id]["findings"] if f["claim"].endswith("refuses malformed variants")]
        assert validated[0]["value"] == len(matrix) == 9
        assert validated[0]["evidence_status"] == "numerically_verified"
        assert all("unsteered" in path["realization"] for path in protocol["paths"] if "realization" in path)
    forged = dict(protocol, hardware_measured={"status": "acquired", "records": [{"acquisition": {
        "device": "camera:1", "raw_sha256": "not-a-digest", "acquired_at": "yesterday", "calibration": "CERT"},
        "retention_identity": "sha256:" + "a" * 64}]})
    assert rec.refusal_code(rec.validate_protocol, forged) == "acquisition_malformed"
    uncited = dict(protocol, hardware_measured={"status": "acquired", "records": [{"acquisition": {
        "device": "camera:1", "raw_sha256": "a" * 64, "acquired_at": "2026-09-23T00:00:00Z", "calibration": "CERT"}}]})
    assert rec.refusal_code(rec.validate_protocol, uncited) == "retention_record_missing"
    # A well-formed slot passes the schema check only; the hardware gate is the runner's (T139).
    cited = deepcopy(uncited)
    cited["hardware_measured"]["records"][0]["retention_identity"] = "sha256:" + "a" * 64
    assert rec.validate_protocol(cited)["hardware_measured"]["status"] == "acquired"


def test_cylinder_protocol_predicts_chord_geodesic_gaps(section):
    _, reports = section
    report = reports["T127"]
    gap_finding = _finding(report, "Rolled-cylinder chord-geodesic gaps")
    gaps = gap_finding["value"]
    # The series check is a signed margin: negative when the alternating-series bound holds with room.
    series = [c for c in gap_finding["basis"]["checks"] if "alternating-series bound" in c["reference"]]
    assert len(series) == 1 and series[0]["comparison"] == "signed_le" and series[0]["observed"] < 0.0
    radius = geo.CYLINDER_RADIUS
    for row in mfg.cylinder_study()["pairs"]:
        step = 1e-4
        numeric = (_gap_at(row, radius + step) - _gap_at(row, radius - step)) / (2 * step)
        assert mfg.gap_radius_derivative(row["dphi_rad"], row["dz_mm"], radius) == pytest.approx(numeric, abs=1e-8)
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
    # Same-origin second derivations are always present and always numerically_verified.
    integrator = _finding(report, "The RK4 coupon transfer agrees with ciw's adaptive")
    geometry = _finding(report, "Coupon Christoffel symbols and Gaussian curvature agree with finite-difference")
    assert integrator["evidence_status"] == geometry["evidence_status"] == "numerically_verified"
    assert integrator["value"] < 1e-5 and 0.0 < geometry["value"]["curvature"] < 1e-9
    signature = _finding(report, "Curvature signature")
    assert signature["basis"]["checks"][0]["observed"] > 5.0  # plate - coupon separation over the combined U
    has = {name: importlib.util.find_spec(name) is not None for name in ("scipy", "sympy")}
    for prefix, module in (("scipy DOP853 integrating", "scipy"), ("A sympy derivation", "sympy")):
        record = _finding(report, prefix)
        if has[module]:
            assert record["evidence_status"] == "independently_verified"
        else:
            assert record["evidence_status"] == "not_established" and record["expected_not_established"] is True


def test_coupon_report_wording_does_not_depend_on_optional_modules(section, monkeypatch, tmp_path):
    _, reports = section
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name in ("scipy", "sympy", "mpmath") else real(name, *args))
    result = mfg.independent_coupon_checks(mfg.nominal_study()["length_mm"])
    assert result["scipy"] is None and result["sympy"] is None
    # The fallback curvature comes from the height values, not from the curvature it checks.
    assert 0.0 < result["ciw_geometry"]["max_curvature_difference"] < 1e-9
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    bare = validate_report(runner.run_task(queue["T128"], section_implementations("manufacturing")["T128"],
                                           runner.Context(tmp_path), {}))
    full = reports["T128"]
    assert bare["state"] == "completed" and bare["evidence_status"]["primary"] == "numerically_verified"
    assert [f["claim"] for f in bare["findings"]] == [f["claim"] for f in full["findings"]]
    for name in runner.PROSE_FIELDS:
        assert runner._skeleton(bare[name]) == runner._skeleton(full[name]), name
    changed = {f["claim"] for f, g in zip(bare["findings"], full["findings"]) if f["evidence_status"] != g["evidence_status"]}
    optional = {f["claim"] for f in full["findings"] if f["claim"].startswith(("scipy DOP853", "A sympy derivation"))}
    assert changed <= optional


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


def test_scan_protocol_and_as_built_fit(section):
    directory, reports = section
    report = reports["T129"]
    fit = _finding(report, "The as-built dome fit")
    assert fit["evidence_status"] == "numerically_verified" and fit["uncertainty"]["value"] >= 0.0
    model = _finding(report, "The fit residual flags an elliptical as-built dome")
    assert model["evidence_status"] == "numerically_verified" and "counterexample" in model
    assert _finding(report, "Surface-scan protocol record validates")["value"] == 9
    assert _labels(report)["The formed coupon passes the Gaussian model test and its fitted height and width lie within "
                           "the declared forming tolerances"] == "not_established"
    protocol = rec.validate_protocol(json.loads(
        (directory / "artifacts" / "T129" / "protocol-surface-scan.json").read_text(encoding="utf-8")))
    assert protocol["protocol_id"] == "MFG-SCAN-01" and protocol["hardware_measured"]["records"] == []
    plan = protocol["scan_plan"]
    assert plan["instrument_setup"]["coupon_max_slope_deg"] < plan["instrument_setup"]["max_incidence_deg"]
    assert "model_test" in plan["surface_fit"] and plan["retained_raw_data"]
    study = mfg.as_built_study()
    assert study["recovery_error"] < 1e-8
    covariance = np.array(study["covariance_mm2"])
    assert np.allclose(covariance, covariance.T) and np.linalg.eigvalsh(covariance).min() > 0.0
    # The scan identifies height and width far better than the declared forming tolerances do.
    u = study["standard_uncertainty_mm"]
    assert u["height_mm"] < 0.1 * 0.2 / math.sqrt(3.0) and u["sigma_mm"] < 0.1 * 0.5 / math.sqrt(3.0)
    assert study["elliptical_chi2"] > study["chi2_threshold"] > 1.0 and study["false_alarm_fraction"] <= 0.02
    # The fit itself: exact on noise-free data from the nominal start, refused when underdetermined.
    x, y = np.meshgrid(np.linspace(-60, 140, 21), np.linspace(-100, 100, 21))
    truth = [10.1, 19.8, 0.2, -0.1, 0.0, 0.0, 1e-4]
    params, _, residual = met.fit_dome(x, y, met.dome_surface(truth, x, y), [10.0, 20.0, 0, 0, 0, 0, 0])
    assert np.allclose(params, truth, atol=1e-9) and float(np.max(np.abs(residual))) < 1e-9
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.fit_dome([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [10.0, 20.0, 0, 0, 0, 0, 0])
    assert refused.value.code == "fit_underdetermined"
    # T138 carries the scan-conditioned uncertainty; it is smaller than the tolerance-based one.
    prediction = mfg.separation_prediction()
    assert max(prediction["scan_conditioned_expanded_mm"]) < max(prediction["conditioned_expanded_mm"])


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
    with pytest.raises(met.MetrologyRefusal) as refused:
        met.datum_frame_321([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]], [0, 1, 0])
    assert refused.value.code == "datum_degenerate"
    # Both degenerate datums are refusal checks of the task, not only of this test.
    datum_checks = {c["reference"]: c for c in _finding(report, "The 3-2-1 datum frame")["basis"]["checks"]}
    assert datum_checks["collinear primary datum points (A)"]["observed_refusal"] == "datum_degenerate"
    assert datum_checks["secondary datum direction (B) normal to the primary plane (A)"]["passed"]
    # The chain std's uncertainty is a linearization estimate in mm, far below the std itself.
    chain = _finding(report, "First-order covariance of the INSTRUMENT->CAD")
    assert chain["uncertainty"]["kind"] == "truncation_bound"
    assert 0.0 <= chain["uncertainty"]["value"] < 1e-3 * min(chain["value"]["coupon_far_corner"])
    links = [(met.transform(np.eye(3), [100.0, 0.0, 0.0]), np.diag([1e-4] * 3 + [1e-10] * 3))]
    linear = met.point_covariance(*met.compose_chain(links), np.array([10.0, 0.0, 0.0]))
    assert np.allclose(met.sigma_point_chain_covariance(links, [10.0, 0.0, 0.0]), linear, rtol=1e-6, atol=1e-15)
    pose = met.transform(met.exp_so3([0.1, -0.2, 0.3]), [1.0, 2.0, 3.0])
    assert np.allclose(met.pose_difference(met.exp_se3(np.array([1e-4, 0, 0, 0, 0, 2e-5])) @ pose, pose),
                       [1e-4, 0, 0, 0, 0, 2e-5], atol=1e-8)  # first order: the left Jacobian adds 1e-9


def test_gage_rr_recovers_components_and_refuses_unbalanced(section):
    directory, reports = section
    report = reports["T131"]
    recovery = _finding(report, "ANOVA Gage R&R recovers")
    assert recovery["evidence_status"] == "numerically_verified"
    assert recovery["value"]["repeatability"] == pytest.approx(1e-4, rel=0.02)
    spread = _finding(report, "Sampling spread of %GRR")["value"]
    assert spread["0.05"] < spread["true"] < spread["0.95"] and spread["true"] == pytest.approx(23.94, abs=0.01)
    refused = _finding(report, "Gage R&R refuses")
    assert refused["value"] == 2 and refused["evidence_status"] == "numerically_verified"
    spread_finding = _finding(report, "Sampling spread of %GRR")
    spread_checks = spread_finding["basis"]["checks"]
    assert all(check["passed"] for check in spread_checks) and len(spread_checks) == 3
    # The uncertainty is in % (order-statistic intervals of the quantiles), like the value.
    halfwidth = spread_finding["uncertainty"]["value"]
    assert set(halfwidth) == {"0.05", "0.5", "0.95"} and all(0.0 < v < 5.0 for v in halfwidth.values())
    assert mfg._quantile_halfwidth(np.arange(1.0, 2001.0), 0.5) == pytest.approx(0.5 * (1044 - 956), abs=1.0)
    # The procedure of a real study: parts are features, every cell once per replicate round, re-fixturing between.
    procedure = json.loads((directory / "artifacts" / "T131" / "gage-rr-procedure.json").read_text(encoding="utf-8"))
    assert procedure["status"].startswith("plan") and procedure["type1_study"]["readings"] == 25
    for protocol_id, study in procedure["studies"].items():
        parts = [p["feature"] for p in study["parts"]]
        assert len(parts) == 10 and len(study["rounds"]) == 3, protocol_id
        for round_ in study["rounds"]:
            assert "re-seat" in round_["before"]
            assert sorted(b["operator"] for b in round_["blocks"]) == ["O1", "O2", "O3"]
            assert all(sorted(b["order"]) == sorted(parts) for b in round_["blocks"])
    plate_pairs = {"-".join(r["pair"]) for r in mfg.plate_study()["pairs"]}
    assert set(mfg.GAGE_PARTS["MFG-FLAT-PLATE-01"]) <= plate_pairs
    assert set(mfg.GAGE_PARTS["MFG-CYLINDER-01"]) == {r["pair"] for r in mfg.cylinder_study()["pairs"]}
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
    # Both regimes grow secularly (linearly): the rate is |M01| / P of the parabolic one-period monodromy.
    rate = _finding(report, "Secular growth rate of the heading-error Jacobi field")
    assert rate["evidence_status"] == "numerically_verified"
    assert rate["value"]["psi50_librating"] == pytest.approx(1.0417, abs=1e-3)
    assert rate["value"]["psi70_circulating"] == pytest.approx(2.9540, abs=1e-3)
    study = mfg.winding_study()
    for regime in study["regimes"].values():
        assert abs(regime["monodromy_trace_minus_2"]) < 1e-6
        assert regime["three_period_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert regime["m01_mm"] == pytest.approx(regime["m01_clairaut_mm"], rel=1e-5)
    assert _finding(report, "The 50 deg winding librates")["evidence_status"] == "numerically_verified"
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
    study = mfg.coating_study()
    assert study["cylinder_control"]["max_difference_mm"] < 1e-12
    # The series check is tight: ray casting and -kappa e^2 / 2 agree to well within 0.1% at every station.
    assert study["standoff_excess"] <= 0.0
    for row in study["standoff"]:
        exact, series = row["standoff_error_exact_mm"], row["standoff_error_series_mm"]
        assert abs(exact - series) <= 1e-3 * abs(series) + 1e-9
    standoff = _finding(report, "Standoff error from a lateral tool offset")
    tight = [c for c in standoff["basis"]["checks"] if c["reference"].startswith("max over stations of abs(ray-cast")]
    assert len(tight) == 1 and "1e-3 abs(series)" in tight[0]["reference"] and tight[0]["passed"]
    # The TCP speed factor is checked pointwise against |t + H dn/ds| and per polyline segment, on the symmetry
    # axis (tau_g = 0) and off it, where the tau_g term is large enough for the pointwise check to resolve.
    offset = _finding(report, "Tool-centre-point path length element")
    assert offset["evidence_status"] == "numerically_verified" and len(offset["basis"]["checks"]) == 3
    assert study["tools"]["welding torch"]["max_H_tau_g"] < 1e-12
    off = study["off_axis_tools"]["welding torch"]
    assert off["max_H_tau_g"] > 1e-2 and off["tau_term_max"] > 100 * 1e-7
    for tools in (study["tools"], study["off_axis_tools"]):
        for tool in tools.values():
            assert tool["pointwise_max_difference"] < 1e-7
        assert tools["welding torch"]["segment_max_difference"] < 2e-3
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
    table = mfg.calibration_table()
    for row in table["rows"]:
        assert len(row["corner_ratios"]) == 4 and row["derating_converged"]
        assert max(row["corner_ratios"].values()) <= mfg.DERATE_TARGET + 1e-9, row["route"]
        assert row["realized_max_error_mm"] <= mfg.LATERAL_SPEC_MM
        # The evidence: a second computation of the derated corners agrees and stays within the spec.
        assert max(row["independent_corner_ratios"].values()) <= 1.0, row["route"]
        assert all(abs(row["independent_corner_ratios"][c] - v) < 1e-4 for c, v in row["corner_ratios"].items())
    kinds = {c["reference_kind"] for c in calibration["basis"]["checks"]}
    assert "cross_implementation" in kinds
    first = _finding(reports["T136"], "The first-order tolerance allocation exceeds the spec")
    assert first["evidence_status"] == "numerically_verified" and "counterexample" in first
    assert first["value"]["fan+25deg"] > 1.0 and first["value"]["fan+0deg"] < 1.0
    focus = _finding(reports["T137"], "Candidate coupon routes ranked by focus margin")
    # Routes without a focus inside the horizon have lower-bound margins only: one tied tier, not an order.
    assert focus["value"]["ranking"][0] == ["fan+15deg", "fan+20deg", "fan+25deg"]
    assert focus["value"]["ranking"][-1] == ["fan+0deg"] and all(len(t) == 1 for t in focus["value"]["ranking"][1:])
    assert focus["value"]["unresolved_tier"] == ["fan+15deg", "fan+20deg", "fan+25deg"]
    assert focus["value"]["margin"]["fan+0deg"] == pytest.approx(0.7588, abs=1e-4)
    counter = _finding(reports["T137"], "The shortest candidate route has the worst focus margin")
    assert counter["evidence_status"] == "numerically_verified"
    assert counter["counterexample"]["witness"]["route"] == "fan+0deg"
    variation_finding = _finding(reports["T137"], "The second variation of route length")
    variation = variation_finding["value"]
    assert variation["finite_difference_mm"] == pytest.approx(variation["index_form_mm"], rel=2e-4)
    assert variation_finding["uncertainty"]["value"] == pytest.approx(
        abs(variation["finite_difference_mm"] - variation["index_form_mm"]), abs=1e-12)
    assert _finding(reports["T136"], "The robot, fixture and frame calibration")["evidence_status"] == "not_established"


def test_ranking_evidence_detects_a_wrong_exact_perturbation(monkeypatch):
    """A wrong exact perturbation still satisfies the derating loop's stop condition but disagrees with the re-evaluation."""
    fan = mfg.fan_study()
    real = geo.separation_nonlinear
    monkeypatch.setattr(geo, "separation_nonlinear", lambda *args, **kwargs: 1.08 * real(*args, **kwargs))
    row = mfg.calibration_ranking({"routes": fan["routes"][:1], "coarse": fan["coarse"]})["rows"][0]
    assert max(row["corner_ratios"].values()) <= mfg.DERATE_TARGET + 1e-9
    assert max(abs(row["independent_corner_ratios"][c] - v) for c, v in row["corner_ratios"].items()) > 1e-2


def test_predicted_separation_has_no_measured_counterpart(section):
    _, reports = section
    report = reports["T138"]
    assert report["state"] == "partial"
    predicted = _finding(report, "Predicted separation of the 2 mm offset route")
    assert predicted["evidence_status"] == "numerically_verified"
    assert predicted["value"]["separation_mm"][0] == pytest.approx(2.0, abs=1e-6)
    assert predicted["value"]["separation_mm"][-1] == pytest.approx(-1.0374, abs=1e-4)
    refused = _finding(report, "The comparison refuses")
    assert refused["value"] == 7 and refused["evidence_status"] == "numerically_verified"
    assert "scan_conditioned_expanded_mm" in predicted["value"]
    assert _finding(report, "Measured separation on the coupon agrees")["evidence_status"] == "not_established"
    start = _finding(report, "A 2-sigma start offset of the declared jig")
    assert start["value"]["max_en_without_execution"] > 1.0 >= start["value"]["max_en_open_loop"]
    # The measured branch: prediction re-integrated from a 2-sigma CMM start-pose estimate, conditioned U_p.
    assert 0.0 < start["value"]["max_en_conditioned"] <= 1.0
    conditioned = [c for c in start["basis"]["checks"] if "re-integrated from a start pose" in c["reference"]]
    assert len(conditioned) == 1 and conditioned[0]["passed"] and conditioned[0]["comparison"] == "le"
    assert start["evidence_status"] == "numerically_verified"
    assert "counterexample" in start
    prediction = mfg.separation_prediction()
    assert max(prediction["conditioned_expanded_mm"]) < max(prediction["open_loop_expanded_mm"])
    assert all(value <= 0.1 for value in prediction["linearity"].values())
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_separation({"separation_mm": [1.0], "expanded_uncertainty_mm": [0.1]}, None)
    assert refused.value.code == "measurement_absent"
    assert rec.normalized_error([1.1], [1.0], [0.06], [0.08])[0] == pytest.approx(1.0)


def test_comparators_compute_normalized_errors_for_a_measurement_record():
    """The success path of both comparators on a synthetic measurement-kind record (never retained, never cited)."""
    data = {"targets.csv": b"station,separation\n"}
    record = _measurement_record(data)
    predicted = {"stations_mm": [0.0, 100.0, 200.0], "separation_mm": [2.0, 1.0, -1.0],
                 "expanded_uncertainty_mm": [0.08, 0.08, 0.08]}
    result = rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.03, 1.08, -1.0],
                                                "expanded_uncertainty_mm": [0.06, 0.06, 0.06]})
    assert result["normalized_error"] == pytest.approx([0.3, 0.8, 0.0], abs=1e-9) and result["agrees"] is True
    assert result["acquisition"]["raw_sha256"] == hashlib.sha256(rec.raw_manifest(record)).hexdigest()
    disagree = rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.0, 1.2, -1.0],
                                                  "expanded_uncertainty_mm": [0.06, 0.06, 0.06]})
    assert disagree["normalized_error"][1] == pytest.approx(2.0, abs=1e-9) and disagree["agrees"] is False
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_separation(predicted, {"record": record, "raw_bytes": data, "values_mm": [2.0, 1.0],
                                           "expanded_uncertainty_mm": [0.06, 0.06]})
    assert refused.value.code == "stations_mismatch"
    pairs = mfg.pair_predictions(mfg.plate_study(), mfg.cylinder_study())
    plate = pairs["MFG-FLAT-PLATE-01"]
    measured = {"record": record, "raw_bytes": data, "labels": plate["pairs"],
                "values_mm": [v + 0.01 for v in plate["values_mm"]], "expanded_uncertainty_mm": 0.05}
    result = rec.compare_pair_distances(plate, measured)
    assert result["normalized_error"] == pytest.approx([0.2] * len(plate["pairs"]), abs=1e-9) and result["agrees"]
    with pytest.raises(rec.RecordRefusal) as refused:
        rec.compare_pair_distances(plate, dict(measured, labels=list(reversed(plate["pairs"]))))
    assert refused.value.code == "pairs_mismatch"
    cylinder = pairs["MFG-CYLINDER-01"]
    ninety = cylinder["pairs"].index("circumferential 90 deg")
    assert cylinder["expanded_uncertainty_mm"][ninety] == pytest.approx(
        2 * abs(math.pi / 2 - 2 * math.sin(math.pi / 4)) * 0.1 / math.sqrt(3.0), rel=1e-9)
    assert cylinder["expanded_uncertainty_mm"][cylinder["pairs"].index("axial 100 mm")] == 0.0


def test_retention_schema_refusals_and_fixture_boundary(section, tmp_path):
    _, reports = section
    report = reports["T139"]
    schema = _finding(report, "The retention schema refuses")
    assert schema["value"] == 11 and schema["evidence_status"] == "numerically_verified"
    boundary = _finding(report, "A schema fixture (as is or relabelled as a measurement)")
    assert boundary["value"] == 3 and boundary["evidence_status"] == "numerically_verified"
    assert _finding(report, "The retention schema keeps a rank-deficient")["evidence_status"] == "numerically_verified"
    assert _finding(report, "A real measurement with raw bytes")["evidence_status"] == "not_established"
    assert report["state"] == "partial" and "no acquisition exists" in report["experiment"]
    assert not any(a["path"].endswith("fixture.txt") for a in report["generated_artifacts"])
    fixture, raw = mfg._schema_fixture()
    assert rec.refusal_code(rec.to_acquisition, fixture, raw) == "fixture_is_not_measurement"
    # Relabelling the fixture does not help: its bytes, serial and zero calibration digest are refused.
    assert rec.refusal_code(rec.to_acquisition, dict(fixture, record_kind="measurement"), raw) == "fixture_is_not_measurement"
    assert rec.refusal_code(rec.validate_retention, dict(fixture, clock=dict(fixture["clock"], acquired_at="2026-09-23 00:00"))) \
        == "clock_without_timezone"
    assert rec.refusal_code(rec.validate_retention, dict(fixture, clock=dict(fixture["clock"], acquired_at="2031-01-01T00:00:00Z"))) \
        == "calibration_expired"
    # A record that is not the fixture yields acquisition fields bound to every raw file, yet the runner still refuses
    # a physical finding without a hardware probe and retained bytes: validators alone cannot mint hardware evidence.
    data = {"a.bin": b"bytes of file a", "b.bin": b"bytes of file b"}
    record = deepcopy(fixture)
    record.update(record_kind="measurement", instrument=dict(fixture["instrument"], serial="SN-1"),
                  raw=[{"name": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value),
                        "media_type": "application/octet-stream"} for name, value in data.items()])
    record["calibration"] = dict(fixture["calibration"], sha256="c" * 64)
    acquisition = rec.to_acquisition(record, data)
    assert acquisition["raw_sha256"] == hashlib.sha256(rec.raw_manifest(record)).hexdigest()
    physical = finding("probe", "physical", 1.0, {"acquisition": acquisition})
    with pytest.raises(EvidenceRefusal):
        runner._gate_physical([physical], runner.Context(tmp_path))
    jac = np.random.Generator(np.random.PCG64(1)).normal(0.0, 1000.0, (6, 3))
    lever = deepcopy(fixture)
    lever["frame_chain"][0]["covariance"] = (jac @ np.diag([1e-4] * 3) @ jac.T).tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) is None
    # The scale test: kept by the relative rule although an absolute 1e-12 rule would refuse it; a negative
    # eigenvalue at the same scale is still refused.
    covariance = mfg._rank_deficient_covariance()
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    assert eigenvalues[0] < -mfg.ABSOLUTE_THRESHOLD and eigenvalues[-1] == pytest.approx(1e6, rel=1e-9)
    lever["frame_chain"][0]["covariance"] = covariance.tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) is None
    lever["frame_chain"][0]["covariance"] = mfg._rank_deficient_covariance(-1e3).tolist()
    assert rec.refusal_code(rec.validate_retention, lever, raw) == "frame_covariance_invalid"


def test_uncertainty_budget_classifies_limiting_terms(section):
    _, reports = section
    report = reports["T140"]
    budget = _finding(report, "Uncertainty budget per predicted quantity")["value"]
    assert budget["cylinder gap, 90 deg pair"]["dominant"] == "instrument"
    assert budget["coupon focal distance (conditioned on the measured start pose)"]["dominant"] == "geometry"
    assert budget["coupon separation at L, 5 mrad heading offset (open-loop start)"]["dominant"] == "execution"
    assert budget["coarse-solver control (6 RK4 steps)"]["dominant"] == "solver"
    plate = budget["plate separation at 240 mm, 5 mrad heading offset (open-loop start)"]
    assert plate["value"] == pytest.approx(1.2, abs=1e-12) and 0.0 < plate["geometry"] < 1e-5
    assert _finding(report, "On the coupon, the heading-offset separation")["evidence_status"] == "numerically_verified"
    # Richardson: a fine-step value carries |Q(h) - Q(2h)| / 15, the same as T128 uses for the focal point.
    focal = budget["coupon focal distance (conditioned on the measured start pose)"]
    nominal = mfg.nominal_study()
    assert focal["solver"] == pytest.approx(abs(nominal["focal_mm"] - nominal["focal_h2_mm"]) / 15.0, rel=1e-12)
    counter = _finding(report, "The coupon focal-distance prediction is geometry-limited")
    assert counter["value"] > 0.5 and "counterexample" in counter and "declared" in counter["claim"]
    assert mfg._classify({"instrument": 1.0, "geometry": 1.0, "solver": 0.0})[0] == "mixed (largest: instrument)"
    # The lateral T138 quantity is budgeted open loop too: geometry, not the start pose, limits it at L.
    assert budget["coupon separation at L, 2 mm lateral offset (open-loop start)"]["dominant"] == "geometry"
    assert "declared instrument and start-pose uncertainties" in _finding(report, "On the coupon, the heading-offset")["claim"]
    # Conditioning on the as-built scan (T129) replaces the dome tolerances and shrinks the geometry term.
    scan = _finding(report, "Conditioning on the as-built scan")
    assert scan["evidence_status"] == "numerically_verified"
    for key in ("coupon focal distance", "coupon separation at L, 2 mm lateral offset"):
        declared = budget[f"{key} (conditioned on the measured start pose)"]["geometry"]
        scanned = budget[f"{key} (conditioned on the start pose and the as-built scan)"]["geometry"]
        assert 0.0 < scanned < 0.1 * declared, key


def test_production_acceptance_stays_outside_the_system(section):
    _, reports = section
    report = reports["T141"]
    api = _finding(report, "No basis establishes a claim filed in an authority domain")
    assert api["evidence_status"] == "numerically_verified"
    assert api["value"]["basis_domain_cases"] == 64 * len(AUTHORITY_DOMAINS) and api["value"]["violations"] == 0
    loophole = _finding(report, "evidence.finding establishes an acceptance statement")
    assert loophole["value"] == "numerically_verified" and "counterexample" in loophole
    note = _finding(report, "Domain assignment of free-text claims is machine-checked")
    assert note["evidence_status"] == "not_established" and note["expected_not_established"] is True
    passing = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
    statement = finding("Coupon lot accepted for production", "computational_pipeline", "accepted", {"checks": [passing]})
    assert rec.refusal_code(rec.screen_acceptance_language, [statement]) == "acceptance_outside_authority_domain"
    refs = {c["reference"]: c for c in api["basis"]["checks"]}
    rejection_check = refs["rejection statement filed in a computational domain (section screen)"]
    assert rejection_check["passed"] and rejection_check["observed_refusal"] == "acceptance_outside_authority_domain"
    assert api["value"]["refusals"] == len(api["basis"]["checks"]) - 1
    for claim in ("Coupon lot rejected for production", "Coupon lot rejected", "Coupon lot scrapped",
                  "Coupon lot quarantined", "Coupon lot passes acceptance", "Coupon lot passed inspection",
                  "Coupon lot conforms and is released to production"):
        decision = finding(claim, "computational_pipeline", "ok", {"checks": [passing]})
        assert decision["evidence_status"] == "numerically_verified"
        assert rec.refusal_code(rec.screen_acceptance_language, [decision]) == "acceptance_outside_authority_domain", claim
    topic = finding("Acceptance criteria are hypotheses", "computational_pipeline", 1.0, {"checks": [passing]})
    assert rec.screen_acceptance_language([topic]) == ["Acceptance criteria are hypotheses"]
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
