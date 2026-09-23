import importlib.util
import math

import numpy as np
import pytest

from ciw.lab import observation, runner
from ciw.lab import observation_chord as chord
from ciw.lab import observation_modes as om
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, primary_label
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report

QUEUE = {t["id"]: t for t in load_queue()["tasks"]}
# Only this section is imported; the other sections are not needed here.
IMPLEMENTATIONS = section_implementations("observation")


class _Without(runner.Context):
    """A run context that reports the named optional modules as unavailable."""

    def __init__(self, output_dir, missing):
        super().__init__(output_dir)
        self.missing = set(missing)

    def available(self, requirement):
        return False if requirement in self.missing else super().available(requirement)


def _run(task_id, directory, ctx=None):
    report = runner.run_task(QUEUE[task_id], IMPLEMENTATIONS[task_id], ctx or runner.Context(directory), {})
    return validate_report(report)


def _find(report, prefix):
    return next(f for f in report["findings"] if f["claim"].startswith(prefix))


def _physical_unestablished(report):
    physical = [f for f in report["findings"] if f["domain"] not in
                ("mathematical", "numerical", "computational_pipeline", "provenance")]
    assert physical and all(f["evidence_status"] == "not_established" for f in physical)
    assert report["physical_validation_status"]["status"] == "not_established"


def _completed(report):
    """Completed, headline derived from the findings, physical claims unestablished, rule-5 fields present."""
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == primary_label(report["findings"])
    _physical_unestablished(report)
    for record in report["findings"]:
        if record["domain"] in COMPUTATIONAL_DOMAINS:
            assert record["evidence_status"] != "not_established", record["claim"]
            assert record["uncertainty"] is not None and "regression_tolerance" in record, record["claim"]


def _sympy():
    return importlib.util.find_spec("sympy") is not None


def test_every_section_task_is_registered_with_tests():
    assert set(IMPLEMENTATIONS) == {f"T0{n}" for n in range(45, 60)}
    for implementation in IMPLEMENTATIONS.values():
        assert implementation.regression_tests
        assert all(node.startswith("tests/test_lab_observation.py::") for node in implementation.regression_tests)
        assert "src/ciw/lab/observation.py" in implementation.changed_files
        assert "docs/lab/OBSERVATION.md" in implementation.changed_files
    # T045 converts chords with the helix-chord inverse in observation_chord.
    assert "src/ciw/lab/observation_chord.py" in IMPLEMENTATIONS["T045"].changed_files


def test_t045_modes_and_refusals(tmp_path):
    report = _run("T045", tmp_path)
    _completed(report)
    assert _find(report, "Seven typed")["value"]["intrinsic"] == ["intrinsic_geodesic_distance",
                                                                  "reconstructed_surface_distance"]
    assert _find(report, "Validation refuses")["value"][:4] == ["missing_frame", "missing_clock", "missing_epoch",
                                                                "missing_calibration"]
    assert _find(report, "No observation mode")["value"] == {"ordered_pairs": 42, "refused": 42}
    record = observation.example_observation("camera_chord_distance")
    for field, code in (("frame_id", "missing_frame"), ("clock_id", "missing_clock"),
                        ("calibration_ref", "missing_calibration")):
        with pytest.raises(om.ObservationRefusal) as refused:
            om.validate(om.observe(record.mode, record.value, **{**_fields(record), field: None}))
        assert refused.value.code == code
    with pytest.raises(om.ObservationRefusal, match="stand in"):
        om.require_mode(record, "intrinsic_geodesic_distance")
    with pytest.raises(om.ObservationRefusal) as refused:
        om.chord_to_surface_distance(record)
    assert refused.value.code == "surface_model_required"
    converted = om.chord_to_surface_distance(record, observation.CYLINDER_MODEL)
    assert converted.mode == "reconstructed_surface_distance" and converted.value[0] == pytest.approx(0.12, abs=1e-14)


def _fields(record):
    fields = record.record()
    for name in ("mode", "value"):
        fields.pop(name)
    fields["derived_from"] = tuple(fields["derived_from"])
    fields["mappings"] = tuple(fields["mappings"])
    return fields


def test_t046_chord_expansion(tmp_path):
    report = _run("T046", tmp_path)
    _completed(report)
    assert _find(report, "Chord expansion")["evidence_status"] == ("independently_verified" if _sympy() else "analytic")
    torus = _find(report, "A torus geodesic")
    assert torus["evidence_status"] == "numerically_verified" and torus["counterexample"]
    assert torus["value"]["fitted_s4"] == pytest.approx(torus["value"]["predicted_s4"], rel=1e-4)
    midpoint = _find(report, "Evaluating the curvature at the arc midpoint")
    assert midpoint["value"]["midpoint_ratio"] < 1e-4 and midpoint["regression_tolerance"]["abs"] >= 1e-5
    helix = _find(report, "Constant curvature alone")
    assert helix["counterexample"]["statement"].startswith("For constant space curvature")
    assert helix["value"]["fitted_s5_gap"] == pytest.approx(0.5 ** 2 * 0.5 ** 2 / 720, rel=1e-5)
    assert _find(report, "RK4 sphere")["value"]["max_c3_relative_error"] < 1e-6


def test_sympy_series_agrees_exactly_with_closed_form(monkeypatch):
    pytest.importorskip("sympy")
    comparison = chord.sympy_versus_closed_form()
    assert comparison["nonzero_residuals"] == 0
    assert set(comparison["residuals"].values()) == {"0"}
    assert comparison["expressions"]["c3"] == "-kappa0**2/24"
    assert comparison["expressions"]["c4"] == "-kappa0*kappa1/24"
    assert comparison["expressions"]["c5"] == "(3*kappa0**4 + 8*kappa0**2*tau0**2 - 72*kappa0*kappa2 - 64*kappa1**2)/5760"
    # The symbolic check detects a closed form that is right only at special points (kappa0' = 0).
    right = chord.closed_form_coefficients
    monkeypatch.setattr(chord, "closed_form_coefficients",
                        lambda k0, k1, t0, k2=0: (*right(k0, k1, t0, k2)[:2], right(k0, 0, t0, k2)[2]))
    assert chord.sympy_versus_closed_form()["nonzero_residuals"] == 1


def test_chord_derivation_degrades_without_sympy(tmp_path):
    ctx = _Without(tmp_path, {"module:sympy"})
    for task_id in ("T046", "T047"):
        report = _run(task_id, tmp_path, ctx)
        assert report["state"] == "completed"
        derivation = report["findings"][0]
        assert derivation["evidence_status"] == "analytic" and "independent_check" not in derivation["basis"]
        assert report["evidence_status"]["primary"] == "analytic"


def test_t047_cylinder_coefficient(tmp_path):
    report = _run("T047", tmp_path)
    _completed(report)
    assert _find(report, "The exact helix chord")["evidence_status"] == ("independently_verified" if _sympy()
                                                                         else "analytic")
    fitted = _find(report, "Small-s fits")["value"]["fitted_R1"]
    predicted = [math.cos(math.radians(a)) ** 4 / 24 for a in observation.ANGLES_DEG]
    assert fitted == pytest.approx(predicted, abs=1e-10)
    rulings = _find(report, "Axial rulings")
    assert rulings["value"]["ruling_max_deficit_over_R"] <= 1e-15
    assert rulings["value"]["circumferential_coefficient_R1"] == pytest.approx(1 / 24, rel=1e-9)
    flat = _find(report, "Zero Gaussian curvature")
    assert flat["counterexample"] and flat["value"]["s_minus_c"] == pytest.approx(0.1 - 2 * math.sin(0.05), rel=1e-12)


def test_t048_synthetic_camera(tmp_path):
    report = _run("T048", tmp_path)
    _completed(report)
    assert _find(report, "Noise-free triangulation")["value"]["max_chord_error_m"] < 1e-12
    bias = _find(report, "Using the camera chord")
    assert bias["value"] == pytest.approx(0.12 - 0.2 * math.sin(0.6), abs=1e-12)
    assert len(bias["basis"]["checks"]) == 4
    noisy = _find(report, "Synthetic chord")
    assert noisy["evidence_status"] == "numerically_verified"
    assert 5e-5 < noisy["value"]["rms_chord_error_m"] < bias["value"] / 10
    for helix in noisy["value"]["per_helix"].values():
        assert abs(helix["chord_z"]) <= observation.Z999 and abs(helix["arc_z"]) <= observation.Z999
    ratio = _find(report, "For the longest circumferential chord")["value"]
    assert ratio["bias_to_noise_ratio"] >= 10 and ratio["ruling_max_abs_bias_m"] < 1e-12
    sensor = _find(report, "A physical stereo rig")
    assert sensor["domain"] == "sensor_performance" and sensor["evidence_status"] == "not_established"


def test_t049_calibration_perturbations(tmp_path):
    report = _run("T049", tmp_path)
    _completed(report)
    ratios = [c for c in _find(report, "Chord sensitivity")["basis"]["checks"] if c["comparison"] == "ge"]
    assert [c["reference"].split(":")[1].split("(")[0].strip() for c in ratios] == ["|d/d cx| over |d/d cy|",
                                                                                   "|d/d yaw| over |d/d pitch|"]
    assert all(c["observed"] >= 10 for c in ratios)
    assert _find(report, "The finite-difference chord Jacobian")["value"] < 1e-6
    linear = _find(report, "First-order calibration")["value"]
    assert linear["residual_slope"] == pytest.approx(2.0, abs=0.05) and linear["relative_residual_at_declared"] < 0.01
    focal = _find(report, "A common focal-length")
    assert abs(focal["value"]["ruling_relative_change"]) < 1e-12
    assert abs(focal["value"]["circumferential_relative_change"]) > 1e-5 and focal["counterexample"]
    assert _find(report, "The declared perturbation")["domain"] == "calibration"


def test_t050_lens_distortion(tmp_path):
    report = _run("T050", tmp_path)
    _completed(report)
    consistency = _find(report, "The implemented radial term")
    assert consistency["basis"]["checks"][0]["reference_kind"] == "invariant"
    law = _find(report, "Uncorrected radial")["value"]
    assert law["mean_abs_bias_k1_0.01_m"] == sorted(law["mean_abs_bias_k1_0.01_m"])
    assert law["radius_slope"] == pytest.approx(2.0, abs=0.1) and law["k1_slope_max_error"] < 0.01
    assert _find(report, "Undistorting")["value"]["corrected_error_m"] < 1e-12
    fold = _find(report, "Strong barrel")
    value = fold["value"]
    # Pixel coordinates are distorted coordinates: the distorted fold radius (2/3)/sqrt(-3 k1) is compared.
    assert value["fold_radius_distorted"] == pytest.approx(2 / 3 / math.sqrt(1.8), rel=1e-12)
    assert value["fold_radius_distorted"] < value["image_corner_radius"] and fold["counterexample"]
    assert value["non_injectivity_gap_px"] > 1.0 and value["undistort_to_other_root"] < 1e-12
    assert 0 < value["image_fraction_beyond_fold"] < 0.05
    # The fold enters this image only below k1 = -4 / (27 rho^2) = -0.521.
    corner = value["image_corner_radius"]
    assert (2 / 3) / math.sqrt(3 * 0.5) > corner > (2 / 3) / math.sqrt(3 * 0.6)


def test_t051_quantization_noise(tmp_path):
    report = _run("T051", tmp_path)
    _completed(report)
    additivity = _find(report, "Rounding plus Gaussian")
    assert max(abs(z) for z in additivity["value"]["z"]) <= observation.Z999
    aligned = _find(report, "Without a random grid phase")
    assert aligned["value"]["sample_variance"] <= 0.01 and aligned["counterexample"]
    propagation = _find(report, "Linear propagation")["value"]
    assert max(abs(z) for z in propagation["z"]) <= observation.Z999


def test_t052_encoder_backlash(tmp_path):
    report = _run("T052", tmp_path)
    _completed(report)
    bounded = _find(report, "Backlash error stays")["value"]
    assert bounded["min_m"] >= 0 and bounded["max_m"] <= bounded["backlash_m"] * (1 + 1e-9)
    assert _find(report, "Backlash error changes")["value"]["changes_outside_windows"] == 0
    assert max(abs(z) for z in _find(report, "Least squares with")["value"]["z"]) <= observation.Z999
    naive = _find(report, "Least squares that ignores")
    assert naive["counterexample"] and abs(naive["value"]["naive_bias_z"]) > 10
    assert naive["value"]["bias_error_over_backlash"] == pytest.approx(naive["value"]["predicted_over_backlash"],
                                                                       rel=0.01)
    assert _find(report, "Backlash error stays")["basis"]["checks"][1]["comparison"] == "signed_le"


def test_t053_imu_drift(tmp_path):
    report = _run("T053", tmp_path)
    _completed(report)
    bounded = _find(report, "On a body rotating")["value"]
    assert bounded["max_transverse_rad"] <= bounded["bound_rad"] * 1.001
    assert bounded["stationary_transverse_rad"] == pytest.approx(0.01, rel=1e-9)
    assert _find(report, "Strapdown bias")["value"]["stationary_relative_error"] < 1e-9
    isotropy = _find(report, "Angle random walk")["value"]
    for name in ("stationary_axis_z", "rotating_axis_z"):
        assert max(abs(z) for row in isotropy[name] for z in row) <= observation.Z999


def test_t054_asynchronous_timestamps(tmp_path):
    report = _run("T054", tmp_path)
    _completed(report)
    assert _find(report, "A clock offset")["value"]["residual_slope"] == pytest.approx(2.0, abs=0.02)
    interpolated = _find(report, "After linear interpolation")["value"]
    assert interpolated["exact_residual_m"] < 1e-12
    assert interpolated["velocity_regression"] == pytest.approx(interpolated["predicted_regression"], abs=5e-4)
    # The sinc^2 factor, not the secant factor alone (about 1 - (omega h)^2/24), explains the regression.
    h, tones = 1 / observation.TIMING["rate_b_hz"], observation.SIGNAL_TONES
    weights = [(a * 2 * math.pi * f) ** 2 for a, f, _ in tones]
    secant_only = sum(w * (1 - (2 * math.pi * f * h) ** 2 / 24) for w, (_, f, _) in zip(weights, tones)) / sum(weights)
    assert abs(interpolated["velocity_regression"] - secant_only) > 2e-3
    mapping = _find(report, "A declared clock mapping")
    assert mapping["value"]["mapped_gap_m"] <= 1e-15 < mapping["value"]["unmapped_gap_m"]
    codes = [c["observed_refusal"] for c in mapping["basis"]["checks"] if c["reference_kind"] == "refusal"]
    assert codes == ["clock_mismatch", "accepted", "epoch_mismatch"]


def test_t055_dropped_observations(tmp_path):
    report = _run("T055", tmp_path)
    _completed(report)
    assert _find(report, "A zero-filled stream")["value"] == {"zero_fill": "zero_filled_missing",
                                                               "unbacked": "unbacked_value"}
    detection = _find(report, "Values alone recognize zero fills")
    value = detection["value"]
    assert detection["counterexample"] and value["far_from_zero_mismatch"] == 0
    # A zero-filled stream that equals a complete acquired stream: fills and genuine zeros are indistinguishable.
    assert value["fills_equal_to_genuine_readings"] > 0 and value["identical_stream_gap_m"] == 0.0
    assert value["exact_zero_rule_false_positives"] > 0 and value["neighbour_rule_missed"] > 0
    hold = _find(report, "Hold-estimate error")["value"]
    assert hold["burst_to_bernoulli_mse_ratio"] > 2
    stream = [observation._encoder_record(0.5, "clock:daq", "epoch:run-0", 0.01 * k, k) for k in range(4)]
    kept = om.with_drops(stream, [False, True, False, True])
    assert om.admit_stream(kept) == {"present": 2, "missing": [1, 3]}
    with pytest.raises(om.ObservationRefusal) as refused:
        om.admit_stream(om.zero_fill(kept, stream[0]))
    assert refused.value.code == "zero_filled_missing"


def test_t056_stale_observations(tmp_path):
    report = _run("T056", tmp_path)
    _completed(report)
    assert _find(report, "Observations older")["value"]["flag_mismatch"] == 0
    assert _find(report, "Age is refused")["value"] == ["missing_latency", "future_observation"]
    late = _find(report, "Staleness is decided by acquisition-time age")["value"]
    assert late["arrival_age_s"] < late["limit_s"] < late["acquisition_age_s"]
    assert late["acquisition_age_s"] == pytest.approx(late["arrival_age_s"] + late["latency_s"], abs=1e-12)
    constant = _find(report, "Stale-state error")["basis"]["checks"][0]
    assert "acquisition_age(record" in constant["reference"] and constant["observed"] < 1e-12


def test_t057_filter_and_smoother(tmp_path):
    report = _run("T057", tmp_path)
    _completed(report)
    rmse = _find(report, "Ensemble position RMSE")["value"]
    assert rmse["smoothed"] <= rmse["filtered"] <= rmse["raw"]
    assert _find(report, "The RTS smoothed")["value"]["min_eigenvalue_filtered_minus_smoothed"] >= -1e-12
    steady = _find(report, "The filter's predicted")
    expected = "independently_verified" if importlib.util.find_spec("scipy") else "numerically_verified"
    assert steady["evidence_status"] == expected
    assert _find(report, "Smoothing does not")["value"]["smoothed_worse_fraction"] > 0.1
    degraded = _run("T057", tmp_path / "noscipy", _Without(tmp_path / "noscipy", {"module:scipy"}))
    assert degraded["state"] == "completed"
    assert _find(degraded, "The filter's predicted")["evidence_status"] == "numerically_verified"


def test_t058_frame_and_clock_basis(tmp_path):
    report = _run("T058", tmp_path)
    _completed(report)
    rigid = _find(report, "A declared rigid frame")
    assert rigid["value"]["exact_gap_m"] == 0.0 and rigid["value"]["distance_change_m"] <= 1.1e-14
    assert rigid["regression_tolerance"]["abs"] >= 1e-14
    assert _find(report, "Declared clock mappings")["value"]["time_gap_s"] == 0.0
    refused = _find(report, "Combining observations")["value"]
    assert refused == {"frame_mismatch": "frame_mismatch", "clock_mismatch": "clock_mismatch",
                       "epoch_mismatch": "epoch_mismatch", "clock_basis_mismatch": "clock_basis_mismatch"}


def test_t059_retained_without_admission(tmp_path):
    report = _run("T059", tmp_path)
    _completed(report)
    assert set(_find(report, "Updating state")["value"].values()) == {"not_admitted"}
    assert _find(report, "Admission as workbench")["domain"] == "actuator_authority"
    store = om.StateStore("encoder_displacement", 0.0, 1.0)
    before = store.digest()
    record = store.retain(observation.example_observation("encoder_displacement"))
    assert store.digest() == before and record["state_admission"] == "not_performed"
    with pytest.raises(om.ObservationRefusal) as refused:
        store.update(record, 0.5)
    assert refused.value.code == "not_admitted" and store.digest() == before
    store.admit(record["observation_digest"], "test")
    assert store.update(record, 1.0)["mean"] == [0.02125]


def test_reports_regenerate_identically(tmp_path):
    for task_id in ("T045", "T052", "T058"):
        first = _run(task_id, tmp_path / "a")
        second = _run(task_id, tmp_path / "b")
        assert first["report_id"] == second["report_id"]
