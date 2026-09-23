import dataclasses
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from ciw.lab import runner
from ciw.lab import sensor_fusion_bench as bench
from ciw.lab import sensor_fusion_filtering as filtering
from ciw.lab import sensor_fusion_objects as objects
from ciw.lab.evidence import AUTHORITY_DOMAINS, COMPUTATIONAL_DOMAINS, PHYSICAL_DOMAINS
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report

QUEUE = {t["id"]: t for t in load_queue()["tasks"]}
# Only this section is imported; the other sections are not needed here.
IMPLEMENTATIONS = section_implementations("sensor-fusion")
TASK_IDS = [f"T{n:03d}" for n in range(60, 77)]
COUNTEREXAMPLE_TASKS = {"T061", "T062", "T063", "T064", "T065", "T066", "T067", "T068", "T069", "T070", "T071",
                        "T072", "T073"}


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Run the whole section once (about ten seconds) and share the validated reports."""
    ctx = runner.Context(tmp_path_factory.mktemp("sensor-fusion"))
    return {task_id: validate_report(runner.run_task(QUEUE[task_id], IMPLEMENTATIONS[task_id], ctx, {}))
            for task_id in TASK_IDS}


def _find(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def _labels(report):
    return [f["evidence_status"] for f in report["findings"]]


def test_section_reports_labels_and_states(reports):
    assert set(IMPLEMENTATIONS) == set(TASK_IDS)
    for task_id, report in reports.items():
        assert report["state"] == "completed", (task_id, report["unresolved_assumptions"])
        # T061 derives its steady-state gain analytically; the primary label is the weakest established one.
        assert report["evidence_status"]["primary"] == ("analytic" if task_id == "T061" else "numerically_verified"), task_id
        assert report["physical_validation_status"]["status"] == "not_established"
        unreal = [f for f in report["findings"] if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS]
        assert unreal and all(f["evidence_status"] == "not_established" for f in unreal), task_id
        for record in report["findings"]:
            if record["domain"] in COMPUTATIONAL_DOMAINS:
                assert record["evidence_status"] in ("numerically_verified", "analytic"), (task_id, record["claim"])
                assert "regression_tolerance" in record, (task_id, record["claim"])
        for name in ("hypothesis", "mathematical_model", "experiment", "numerical_result", "uncertainty",
                     "recommended_next_task"):
            assert isinstance(report[name], str) and report[name].strip(), (task_id, name)
        assert report["failure_modes_checked"] and report["unresolved_assumptions"], task_id
        assert report["recommended_next_task"].startswith("T"), task_id
        assert report["generated_artifacts"], task_id
        has_counterexample = any("counterexample" in f for f in report["findings"])
        assert has_counterexample == (task_id in COUNTEREXAMPLE_TASKS), task_id
    # Tasks invoking real-world conclusions record them in the proper domain.
    assert _find(reports["T061"], "Real sensors")["domain"] == "sensor_performance"
    assert _find(reports["T072"], "The validity interval")["domain"] == "calibration"
    assert _find(reports["T073"], "A 1 m")["domain"] == "machine_safety"
    assert _find(reports["T075"], "An admitted")["domain"] == "actuator_authority"
    assert _find(reports["T076"], "Synthetic fusion")["domain"] == "production_acceptance"


def test_findings_are_deterministic(tmp_path):
    first = runner.run_task(QUEUE["T066"], IMPLEMENTATIONS["T066"], runner.Context(tmp_path / "a"), {})
    second = runner.run_task(QUEUE["T066"], IMPLEMENTATIONS["T066"], runner.Context(tmp_path / "b"), {})
    assert json.dumps(first["findings"], sort_keys=True) == json.dumps(second["findings"], sort_keys=True)


def test_bench_is_deterministic_rate_exact_and_stream_independent(reports):
    report = reports["T060"]
    regen = _find(report, "Regenerating the bench")["value"]
    assert regen["same_seed_arrays_differing"] == 0 and regen["other_seed_fraction_changed"] == 1.0
    rates = _find(report, "Every sensor reports")["value"]
    assert rates["max_count_mismatch"] == 0 and rates["off_grid_ticks"] == 0
    assert {k: v["readings_per_run"] for k, v in rates["per_sensor"].items()} == {
        "camera": 200, "encoder": 400, "imu": 400, "tracker": 40}
    assert _find(report, "Every raw reading")["value"] < 1e-12
    assert _find(report, "Changing the tracker rate")["value"] == 0
    # The generator itself: same seed, same bytes; readings retain their declared covariances.
    small = bench.BenchConfig(ticks=20)
    a, b = bench.generate_bench(small, 5, 3), bench.generate_bench(small, 5, 3)
    assert bench.bench_digest(a) == bench.bench_digest(b)
    assert bench.bench_digest(a) != bench.bench_digest(bench.generate_bench(small, 6, 3))


def test_declared_covariance_matches_and_low_rate_misstatement_escapes(reports):
    report = reports["T061"]
    moments = _find(report, "Sample means and covariances")["value"]
    assert moments["max_abs_z"] < moments["z_critical"] and moments["moments_tested"] == 28
    detect = _find(report, "A 10% understatement")
    assert detect["value"]["tracker"] < moments["z_critical"] < min(detect["value"][k] for k in ("camera", "encoder",
                                                                                                  "imu"))
    assert detect["counterexample"]["witness"]["sensor"] == "tracker"
    minimal = _find(report, "Minimal relative variance")
    assert minimal["evidence_status"] == "analytic"
    assert minimal["value"]["tracker"] == pytest.approx(moments["z_critical"] * math.sqrt(2 / 2000))


def test_correlated_noise_consistency_and_ignored_correlation_counterexample(reports):
    report = reports["T062"]
    good = _find(report, "With the correct cross-correlated")["value"]
    assert good["nees"]["fraction_inside"] >= 0.9 and good["whitened_max_z"] < good["z_critical"]
    bad = _find(report, "Ignoring the camera-tracker")
    assert bad["value"]["nees"]["fraction_above"] >= 0.9
    assert bad["counterexample"]["statement"] == "Ignoring correlation between sensor noises is harmless"
    blind = _find(report, "The ignored-correlation filter still passes")["value"]
    assert blind["nis"]["fraction_inside"] >= 0.9 and blind["whitened_max_z"] > blind["z_critical"]
    predicted = _find(report, "Monte Carlo grand-mean NEES")["value"]
    assert predicted["predicted"]["correct"]["nees"] == pytest.approx(4.0, abs=1e-9)
    assert predicted["predicted"]["ignored"]["nees"] > 5.0


def test_frame_transform_covariance_and_mahalanobis_invariance(reports):
    report = reports["T063"]
    mc = _find(report, "Monte Carlo covariances of rotated")["value"]
    assert mc["max_abs_z"] < mc["z_critical"] and mc["moments_tested"] == 40
    assert max(_find(report, "Mahalanobis distance is invariant")["value"].values()) < 1e-9
    rotated = _find(report, "Rotating a position measurement")
    assert rotated["value"]["mean_d2"] > 10 and rotated["value"]["rejection_rate"] > 0.3
    assert rotated["value"]["rejection_rate_with_rotated_covariance"] == pytest.approx(0.01, abs=0.003)
    unit = _find(report, "Converting a state to millimetres")["value"]
    assert unit["max_relative_deviation_from_scale_squared"] < 1e-12
    assert _find(report, "The declared 35 degree")["domain"] == "calibration"


def test_jacobi_transfer_covariance_collapse_and_breakdown(reports):
    report = reports["T064"]
    phi = _find(report, "The Jacobi transfer matrix")["value"]
    assert phi["sphere"]["max_phi_error"] < 1e-8 and phi["hyperbolic"]["max_det_error"] < 1e-9
    mc = _find(report, "Phi P Phi^T predicts")["value"]
    assert mc["max_abs_z"] < mc["z_critical"] and mc["batched_rhs_vs_christoffel"] < 1e-12
    collapse = _find(report, "Lateral variance collapses")
    assert collapse["value"]["conjugate_point"] == pytest.approx(math.pi, abs=1e-6)
    assert collapse["value"]["predicted_variance_at_pi"] == pytest.approx(0.002 ** 2, rel=1e-6)
    assert collapse["value"]["monte_carlo_ratio_to_peak"] < 0.05
    assert collapse["value"]["hyperbolic_conjugate_points"] == []
    breakdown = _find(report, "The heading standard deviation")
    assert breakdown["value"]["hyperbolic"] < 0.05 < 0.2 < breakdown["value"]["sphere"]
    table = _find(report, "The first-order variance")["value"]["table"]
    errors = [row["linear_relative_error"] for row in table["hyperbolic"]]
    assert errors == sorted(errors) and errors[0] < 2e-3 and errors[-1] > 1.0


def test_filter_induced_correlation_and_naive_average(reports):
    exact = filtering.rational_steady_state(Fraction(1), Fraction(30))
    assert (exact["M"], exact["P"], exact["K"], exact["a"]) == (6, 5, Fraction(1, 6), Fraction(5, 6))
    assert exact["riccati_residual"] == 0
    with pytest.raises(ValueError, match="irrational"):
        filtering.rational_steady_state(Fraction(1), Fraction(1))
    # Two outputs: Var((e1 + e2)/2) = P/4 (2 + 2a).
    assert filtering.average_variance(Fraction(5), Fraction(5, 6), 2) == Fraction(5, 4) * (2 + Fraction(5, 3))
    report = reports["T065"]
    naive = _find(report, "Treating 20 successive")
    assert naive["value"]["ratio"] > 7.5 and naive["value"]["naive_coverage_exact"] < 0.6
    assert abs(naive["value"]["z_monte_carlo_vs_exact"]) < 4
    lag = _find(report, "On the planar constant-velocity bench")["value"]
    assert lag["max_abs_z"] < lag["z_critical"] and lag["position_correlation_by_lag"][1] > 0.8


def test_residuals_need_filter_covariance(reports):
    report = reports["T066"]
    assert _find(report, "Innovations normalized by S")["value"]["fraction_inside"] >= 0.9
    raw = _find(report, "Normalizing innovations by the raw")["value"]
    assert raw["fraction_above"] >= 0.9 and raw["grand_mean"] == pytest.approx(raw["predicted_grand_mean"], rel=0.05)
    post = _find(report, "Normalizing post-fit residuals")["value"]
    assert post["fraction_below"] >= 0.9 and post["predicted_grand_mean"] < 2.0
    assert _find(report, "Post-fit residuals z - H x+")["value"]["identity_error"] < 1e-12


def test_gating_rates_open_and_closed_loop(reports):
    report = reports["T067"]
    open_loop = _find(report, "Open loop")["value"]
    for p, row in open_loop.items():
        assert row["wilson"][0] <= 1 - float(p) <= row["wilson"][1]
    closed = _find(report, "Closed loop")
    assert closed["value"]["0.9"]["wilson"][0] > 0.1
    assert closed["value"]["0.9"]["rejection_rate_after_a_rejection"] > 0.2
    assert _find(report, "Gating the NIS computed with the raw")["value"]["rate"] > 0.02
    assert bench.chi2_quantile(0.99, 2) == pytest.approx(-2 * math.log(0.01), rel=1e-14)
    with pytest.raises(ValueError, match="strictly between"):
        bench.chi2_quantile(1.0, 2)


def test_chi2_and_noncentral_laws_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    for dof in (1, 2, 3, 4, 7, 12):
        for p in (0.9, 0.99, 0.999):
            assert bench.chi2_quantile(p, dof) == pytest.approx(stats.chi2.ppf(p, dof), rel=1e-9)
    for lam in (0.5, 4.0, 30.0):
        assert bench.noncentral_chi2_2_cdf_many(9.21, [lam])[0] == pytest.approx(stats.ncx2.cdf(9.21, 2, lam), abs=1e-9)
    lo, hi = bench.wilson_interval(50, 1000, 0.999)
    assert lo < 0.05 < hi


def test_outlier_rejection_detection_lockout_and_cost(reports):
    report = reports["T068"]
    gross = _find(report, "Gross 1.5 m outliers")["value"]
    assert gross["detection_rate"] > 0.98 and abs(gross["detection_z"]) < 3
    kept = _find(report, "Outside the lock-out runs")["value"]["rmse_without_lockout_runs"]
    assert kept["gated"] < 1.05 * kept["oracle"] < kept["ungated"]
    lockout = _find(report, "Cold-start lock-out")
    assert lockout["value"]["lockout_runs"] == lockout["value"]["lockout_runs_with_accepted_first_outlier"] > 0
    assert lockout["counterexample"]["witness"]["rmse_gated"] > 1.0
    clean = _find(report, "On clean data the 99% gate")["value"]
    assert clean["rmse_gated"] > clean["rmse_ungated"] and clean["paired_mse_z_gated_minus_ungated"] > 3
    assert _find(report, "Subtle 0.3 m outliers")["value"]["detection_rate"] < 0.2


def test_missing_data_prediction_only_and_zero_fill_refusal(reports):
    report = reports["T069"]
    assert _find(report, "Prediction-only steps grow")["value"]["max_relative_error"] < 1e-12
    refusals = _find(report, "The session refuses zero-filling")["value"]
    assert refusals == {"zero_fill": "zero_fill_refused", "hold_last": "gap_strategy_refused",
                        "nan_reading": "nonfinite_observation", "absent_value": "missing_reading",
                        "state_unchanged": True}
    assert _find(report, "Zero-filling missing readings")["value"]["zero_fill"]["grand_mean_nees"] > 100
    # Direct API: a zero-fill request is refused before any arithmetic.
    session = objects.FusionSession(read_only=False)
    session.initialize(np.zeros(4), np.eye(4), 0)
    with pytest.raises(objects.FusionRefusal) as caught:
        session.handle_gap("camera", 3, strategy="zero_fill")
    assert caught.value.code == "zero_fill_refused" and session.tick == 0


def test_stale_clock_bias_detection_and_augmented_offset(reports):
    report = reports["T070"]
    detected = _find(report, "With a correctly clocked tracker")["value"]
    assert detected["max_abs_mean_z"] > 10 > detected["z_critical"] > detected["max_abs_z_vs_predicted"]
    augmented = _find(report, "A filter that estimates the clock offset")["value"]
    assert augmented["tau_mean"] == pytest.approx(0.1, abs=0.01) and augmented["max_abs_mean_z"] < 4.5
    alone = _find(report, "With the stale camera as the only")
    assert max(abs(z) for z in alone["value"]["camera_mean_z"]) < 3
    assert alone["value"]["position_error_mean"][0] == pytest.approx(-0.1, abs=0.02)
    assert _find(report, "The recovered offset")["domain"] == "calibration"


def test_frame_mismatch_inflation_blind_spot_and_refusal(reports):
    report = reports["T071"]
    mixed = _find(report, "Fusing a tracker expressed")["value"]
    assert mixed["late_fraction_above"] >= 0.9 and mixed["late_grand_nis"] > 6
    assert _find(report, "Near the rotation centre")["value"]["early_fraction_inside"] >= 0.9
    alone = _find(report, "A rotated sensor fused alone")["value"]
    assert alone["late_fraction_inside"] >= 0.9 and alone["final_nees"] > 100
    api = _find(report, "The session refuses an observation whose frame id")["value"]
    assert api["fuse_in_wrong_frame"] == api["transform_from_wrong_frame"] == "frame_mismatch"
    assert api["fuse_after_transform"] == "no_refusal"


def test_calibration_expiry_retains_but_refuses(reports):
    report = reports["T072"]
    counts = _find(report, "Readings inside the half-open")["value"]
    assert (counts["fused"], counts["refused_expired"], counts["first_refused"]) == (59, 41, 60)
    retained = _find(report, "Refused readings are retained")["value"]
    assert retained["retained"] == 100 and retained["state_equals_prediction_only_shadow"] is True
    assert retained["dispositions"] == {"fused": 59, "refused:calibration_expired": 41}
    other = _find(report, "Revoked, unknown")["value"]
    assert other["revoked"] == "calibration_revoked" and other["unknown"] == "calibration_unknown"
    record = objects.CalibrationRecord("c", "camera", "world", 10, 20)
    assert (record.covers(10), record.covers(19), record.covers(20)) == (True, True, False)


def test_track_lost_prediction_refusals_and_reacquisition(reports):
    report = reports["T073"]
    lost = _find(report, "The track is declared lost")["value"]
    assert lost["observed_lost_tick"] == lost["predicted_lost_tick"]
    codes = _find(report, "A lost track is not admissible")["value"]
    assert codes["admit_lost"] == "track_lost" and codes["fuse_lost"] == "track_lost_requires_reacquisition"
    assert codes["reacquire_gap"] == "reacquisition_needs_consecutive_readings"
    assert codes["reacquire_while_tracking"] == "reacquisition_not_needed"
    two = _find(report, "The two-point reacquisition covariance")["value"]
    assert two["session_vs_formula"] == 0.0 and abs(two["nees_z"]) < 4
    turn = _find(report, "Under an unmodelled 0.2 rad/s turn")
    assert turn["value"]["first_inconsistent_gap_ticks"] > turn["value"]["gap_ticks_before_track_loss"]


def test_fused_state_equals_batch_posterior(reports):
    report = reports["T074"]
    batch = _find(report, "The recursive Kalman estimate")["value"]
    assert all(row["max_relative_mean_difference"] < 1e-9 for row in batch["comparisons"])
    assert batch["dense_vs_banded_at_K10"] < 1e-9
    exact = _find(report, "In exact rational arithmetic")
    assert exact["value"]["identical"] is True and exact["value"]["filter_mean"] == exact["value"]["batch_mean"]
    # Independent small case: dense and banded batch solvers agree with the filter.
    F, Q = bench.cv_model(0.1, 0.05)
    plan = [(bench.H_POS, np.eye(2) * 0.04)] * 6
    readings = [np.array([0.1 * k, 0.05 * k]) for k in range(1, 7)]
    dense, cov = bench.batch_posterior(F, Q, np.zeros(4), np.eye(4), plan, readings)
    banded, last = bench.batch_posterior_banded(F, Q, np.zeros(4), np.eye(4), plan, readings)
    steps = bench.gain_schedule(F, Q, np.eye(4), plan)
    estimates, _ = bench.run_shared(F, np.zeros(4), steps, [r[None] for r in readings])
    assert np.allclose(dense, banded, atol=1e-12) and np.allclose(cov[-4:, -4:], last, atol=1e-12)
    assert np.allclose(estimates[0, -1], banded[-1], atol=1e-12) and np.allclose(steps[-1].post, last, atol=1e-12)


def test_typed_objects_and_admission_mutations(reports):
    report = reports["T075"]
    refused = _find(report, "Every admission check refuses")["value"]
    assert len(refused) == len(objects.ADMISSION_CHECKS)
    assert all(row["expected"] == row["full_gate"] for row in refused.values())
    mutation = _find(report, "Mutation analysis")["value"]
    assert mutation["mutants_killed"] == mutation["checks"] == 13 and mutation["sole_guard_checks"] == 12
    typed = _find(report, "Observations, candidates and admitted")["value"]
    assert typed["auto_admitted_before_gate"] == 0 and typed["admitted_after_gate"] == 1
    assert typed["construct_admitted_directly"] == "admission_requires_gate"
    # Direct refusals at the type boundary.
    with pytest.raises(objects.FusionRefusal, match="gate"):
        objects.AdmittedState(None, (), {}, {})
    observation = objects.Observation("camera", "world", 1, (0.0, 0.0), np.eye(2), "cal")
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.tick = 2
    with pytest.raises(objects.FusionRefusal) as caught:
        objects.Observation("camera", "world", 1, (0.0, 0.0), -np.eye(2), "cal")
    assert caught.value.code == "covariance_not_positive_definite"


def test_defaults_are_read_only_and_not_performed(reports):
    from ciw.declared_workload import AUTHORITY

    report = reports["T076"]
    vocabulary = _find(report, "The CIW authority vocabulary")["value"]
    assert vocabulary["ciw_authority"] == AUTHORITY == vocabulary["session_authority"]
    assert AUTHORITY["sensor_fusion"] == AUTHORITY["state_admission"] == "not_performed"
    defaults = _find(report, "FusionSession defaults to read-only")["value"]
    assert set(defaults["codes"].values()) == {"read_only_session"} and defaults["admitted"] == 0
    session = objects.FusionSession()
    assert session.read_only is True and session.authority == AUTHORITY
    with pytest.raises(objects.FusionRefusal) as caught:
        session.initialize(np.zeros(4), np.eye(4), 0)
    assert caught.value.code == "read_only_session"
    assert _find(report, "Enabling fusion explicitly")["value"]["sensor_fusion"] == "synthetic_only"
