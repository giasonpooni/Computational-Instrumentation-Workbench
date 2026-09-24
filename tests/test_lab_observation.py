import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ciw.lab import observation, runner
from ciw.lab import observation_chord as chord
from ciw.lab import observation_modes as om
from ciw.lab import sensor_fusion_intake as intake
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


NV, IV, NE = "numerically_verified", "independently_verified", "not_established"


def _completed(report, labels, primary=NV):
    """Completed with the expected label on every finding and headline, physical claims unestablished.

    ``labels`` lists the expected evidence label of each finding in order, so a
    finding whose checks were lost (falling to ``synthetic`` or ``analytic``)
    fails here, and so does a primary label other than ``primary``.
    """
    assert report["state"] == "completed"
    assert [f["evidence_status"] for f in report["findings"]] == labels
    assert report["evidence_status"]["primary"] == primary == primary_label(report["findings"])
    _physical_unestablished(report)
    for record in report["findings"]:
        if record["domain"] in COMPUTATIONAL_DOMAINS:
            assert record["uncertainty"] is not None and "regression_tolerance" in record, record["claim"]


def _sympy():
    return importlib.util.find_spec("sympy") is not None


def _scipy():
    return importlib.util.find_spec("scipy") is not None


def _derivation_label():
    return IV if _sympy() else "analytic"


def _derivation_primary():
    """Without sympy the derivation finding is analytic, the weakest established label, so it is the primary."""
    return NV if _sympy() else "analytic"


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
    _completed(report, [NV, NV, NV, NV, NV, NE])
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
    # The runner retains operator captures; what is missing is a section-4 parser that reads one, and an
    # authenticated route for these instruments (a capture alone never supports hardware_measured).
    step = report["recommended_next_task"]
    assert "operator capture" in step and "capture parser" in step and "the runner does not have" not in step
    assert "trust anchor" in step and "No hardware_measured observation exists" in step


def test_reconstructed_distance_carries_the_t044_split(tmp_path):
    report = _run("T045", tmp_path)
    split = _find(report, "A model-derived surface distance carries separate geometry and sensor")
    assert split["evidence_status"] == NV and set(split["value"]["components"]) == {"geometry_m2", "sensor_m2"}
    assert all(abs(z) < observation.Z999 for z in split["value"]["z"].values())
    assert split["value"]["finite_difference_relative_gap"] < 1e-6
    assert om.MODES["reconstructed_surface_distance"].variance_components == ("geometry_m2", "sensor_m2")
    assert om.MODES["reconstructed_surface_distance"].variance_required is True
    # The conversion fills both components by first-order propagation of the declared uncertainties.
    chord_record = observation.example_observation("camera_chord_distance")
    derived = om.chord_to_surface_distance(chord_record, observation.CYLINDER_MODEL)
    gains = om.arc_length_sensitivities(chord_record.value[0], observation.CYLINDER_MODEL)
    sigma = observation.CYLINDER_MODEL["parameter_sigma"]
    assert derived.variance_components["sensor_m2"] == pytest.approx(gains["chord"] ** 2 * observation.CHORD_SENSOR_M2)
    assert derived.variance_components["geometry_m2"] == pytest.approx(
        (gains["radius"] * sigma["radius"]) ** 2 + (gains["path_angle_rad"] * sigma["path_angle_rad"]) ** 2)
    # A surface distance without both components, or a conversion without the inputs of either, is refused.
    for record, code in ((dataclasses.replace(derived, variance_components={"sensor_m2": 1e-8}),
                          "variance_split_mismatch"),
                         (dataclasses.replace(derived, variance_components=None), "variance_split_mismatch"),
                         (dataclasses.replace(derived, variance_components={"geometry_m2": -1.0, "sensor_m2": 0.0}),
                          "invalid_variance"),
                         (dataclasses.replace(observation.example_observation("tracker_measurement"),
                                              variance_components={"sensor_m2": 1e-8}), "variance_split_mismatch")):
        with pytest.raises(om.ObservationRefusal) as refused:
            om.validate(record)
        assert refused.value.code == code
    with pytest.raises(om.ObservationRefusal) as refused:
        om.chord_to_surface_distance(dataclasses.replace(chord_record, variance_components=None),
                                     observation.CYLINDER_MODEL)
    assert refused.value.code == "variance_split_required"
    bare = {k: v for k, v in observation.CYLINDER_MODEL.items() if k != "parameter_sigma"}
    with pytest.raises(om.ObservationRefusal) as refused:
        om.chord_to_surface_distance(chord_record, bare)
    assert refused.value.code == "geometry_uncertainty_required"
    # A plane has no parameters: its geometry component is zero only because the model states so.
    plane = om.chord_to_surface_distance(chord_record, {"kind": "plane", "parameter_sigma": {}})
    assert plane.variance_components == {"geometry_m2": 0.0, "sensor_m2": observation.CHORD_SENSOR_M2}
    # Components that are not a mapping are refused with a code, not a crash, also when a retained record is admitted.
    listed = dataclasses.replace(chord_record, variance_components=["sensor_m2"])
    with pytest.raises(om.ObservationRefusal) as refused:
        om.validate(listed)
    assert refused.value.code == "invalid_variance"
    ledger = om.ObservationLedger(read_only=False)
    kept = ledger.retain(listed)
    with pytest.raises(om.ObservationRefusal) as refused:
        ledger.admit(kept["observation_digest"], "test")
    assert refused.value.code == "invalid_variance"
    # An empty mapping declares nothing: it validates and has the digest of the record without components, so one
    # measurement cannot be retained under two digests. A required split is not met by it.
    for record in (observation.example_observation("encoder_displacement"),
                   dataclasses.replace(chord_record, variance_components=None)):
        empty = dataclasses.replace(record, variance_components={})
        assert om.validate(empty) is empty and empty.digest() == record.digest()
        assert "variance_components" not in empty.record()
    with pytest.raises(om.ObservationRefusal) as refused:
        om.validate(dataclasses.replace(derived, variance_components={}))
    assert refused.value.code == "variance_split_mismatch"


def _fields(record):
    fields = record.record()
    for name in ("mode", "value"):
        fields.pop(name)
    fields["derived_from"] = tuple(fields["derived_from"])
    fields["mappings"] = tuple(fields["mappings"])
    return fields


def test_t046_chord_expansion(tmp_path):
    report = _run("T046", tmp_path)
    _completed(report, [_derivation_label(), NV, NV, NV, NV, NE], primary=_derivation_primary())
    derivation = _find(report, "Chord expansion")
    if _sympy():
        # The Frenet recursion shares the Frenet model, so it is an ordinary check; the explicit-curve route
        # (no Frenet recursion) is the independent comparison.
        assert [c["reference_kind"] for c in derivation["basis"]["checks"]] == ["exact_arithmetic"]
        independent = derivation["basis"]["independent_check"]
        assert independent["checker"]["implementation"] == "sympy" and independent["observed"] == 0.0
        assert "explicit polynomial space curves" in independent["reference"]
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


def test_explicit_curve_route_detects_a_wrong_closed_form(monkeypatch):
    pytest.importorskip("sympy")
    one_curve = chord.EXPLICIT_CURVES[:1]
    assert chord.sympy_explicit_curve_check(one_curve)["nonzero_residuals"] == 0
    # Dropping the -64 kappa0'^2 term of c5 is caught without the Frenet recursion.
    right = chord.closed_form_coefficients
    monkeypatch.setattr(chord, "closed_form_coefficients",
                        lambda k0, k1, t0, k2=0: (*right(k0, k1, t0, k2)[:2],
                                                  right(k0, k1, t0, k2)[2] + 64 * k1 ** 2 / 5760))
    assert chord.sympy_explicit_curve_check(one_curve)["nonzero_residuals"] == 1


def test_chord_derivation_degrades_without_sympy(tmp_path):
    ctx = _Without(tmp_path, {"module:sympy"})
    for task_id in ("T046", "T047"):
        report = _run(task_id, tmp_path, ctx)
        _completed(report, ["analytic", NV, NV, NV, NV, NE], primary="analytic")
        derivation = report["findings"][0]
        assert "independent_check" not in derivation["basis"] and "checks" not in derivation["basis"]


def test_t047_cylinder_coefficient(tmp_path):
    report = _run("T047", tmp_path)
    _completed(report, [_derivation_label(), NV, NV, NV, NV, NE], primary=_derivation_primary())
    if _sympy():
        independent = _find(report, "The exact helix chord")["basis"]["independent_check"]
        # A symbolic residual count, not a float comparison of 30-digit values.
        assert independent["reference_kind"] == "exact_arithmetic" and independent["observed"] == 0.0
    fitted = _find(report, "Small-s fits")["value"]["fitted_R1"]
    predicted = [math.cos(math.radians(a)) ** 4 / 24 for a in observation.ANGLES_DEG]
    assert fitted == pytest.approx(predicted, abs=1e-10)
    rulings = _find(report, "Axial rulings")
    # Two-sided over s > 0: the s = 0 sample (s - c = 0 exactly) cannot anchor the bound.
    ruling_check = rulings["basis"]["checks"][0]
    assert ruling_check["comparison"] == "abs_le" and "s > 0" in ruling_check["reference"]
    assert 0 < rulings["value"]["ruling_max_abs_deficit_over_R"] <= 1e-15
    assert rulings["value"]["circumferential_coefficient_R1"] == pytest.approx(1 / 24, rel=1e-9)
    flat = _find(report, "Zero Gaussian curvature")
    assert flat["counterexample"] and flat["value"]["s_minus_c"] == pytest.approx(0.1 - 2 * math.sin(0.05), rel=1e-12)


def test_t048_synthetic_camera(tmp_path):
    report = _run("T048", tmp_path)
    _completed(report, [NV, NV, NV, NV, NV, NE])
    noise_free = _find(report, "Noise-free triangulation")
    assert noise_free["value"]["max_chord_error_m"] < 1e-12
    visibility = noise_free["basis"]["checks"][0]
    assert visibility["reference"].startswith("markers not visible") and visibility["observed"] == 0
    assert not any("correlated rounding" in mode for mode in report["failure_modes_checked"])
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
    _completed(report, [NV, NV, NV, NV, NV, NE])
    sensitivity = _find(report, "Chord sensitivity to 12")["value"]
    assert list(sensitivity) == list(observation.PARAMETERS) and len(sensitivity) == 12
    # The baseline column: about c/b per metre, i.e. 0.12 m chord / 0.2 m per mm of baseline error.
    assert abs(sensitivity["right_tx_m"][-1]) == pytest.approx(0.1175 / 0.2 * 1e-3, rel=0.02)
    baseline = _find(report, "A baseline error")["value"]
    assert baseline["model_gap_m"] < 1e-12 and baseline["min_relative_change"] == pytest.approx(0.005, rel=1e-9)
    assert any("Aspect ratio and skew" in item for item in report["unresolved_assumptions"])
    ratios = [c for c in _find(report, "Chord sensitivity")["basis"]["checks"] if c["comparison"] == "ge"]
    assert [c["reference"].split(":")[1].split("(")[0].strip() for c in ratios] == ["|d/d cx| over |d/d cy|",
                                                                                   "|d/d yaw| over |d/d pitch|"]
    assert all(c["observed"] >= 10 for c in ratios)
    analytic = _find(report, "The finite-difference chord Jacobian")
    assert analytic["value"] < 1e-6 and "baseline derivatives" in analytic["claim"]
    linear = _find(report, "First-order calibration")["value"]
    assert linear["residual_slope"] == pytest.approx(2.0, abs=0.05) and linear["relative_residual_at_declared"] < 0.01
    focal = _find(report, "A common focal-length")
    assert abs(focal["value"]["ruling_relative_change"]) < 1e-12
    assert abs(focal["value"]["circumferential_relative_change"]) > 1e-5 and focal["counterexample"]
    assert _find(report, "The declared perturbation")["domain"] == "calibration"


def test_t050_lens_distortion(tmp_path):
    report = _run("T050", tmp_path)
    _completed(report, [NV, NV, NV, NV, NV, NV, NE, NE])
    consistency = _find(report, "The implemented radial term")
    assert consistency["basis"]["checks"][0]["reference_kind"] == "invariant"
    law = _find(report, "Uncorrected radial")["value"]
    assert law["mean_abs_bias_k1_0.01_m"] == sorted(law["mean_abs_bias_k1_0.01_m"])
    # J_pix by central differences with the declared step: its truncation and rounding stay far below the residual.
    assert observation.JACOBIAN_STEP_PX == 1e-2 and "0.01 px step" in report["experiment"]
    assert law["first_order_relative_residual"] == pytest.approx(0.0030165, rel=1e-5)
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


def test_t050_perspective_markers(tmp_path):
    from ciw.lab import observation_camera as cam

    camera = cam.stereo_rig(rectified=True)[0]
    # On the optical axis, a disc tilted by beta about y: offset f rho^2 sin(beta) cos(beta) / (Z^2 - rho^2 sin^2 beta).
    z, rho, beta = 0.6, 0.01, math.radians(35.0)
    normal = np.array([math.sin(beta), 0.0, -math.cos(beta)])
    centre = np.array([0.0, 0.0, z])
    expected = camera.focal_px * rho ** 2 * math.sin(beta) * math.cos(beta) / (z ** 2 - rho ** 2 * math.sin(beta) ** 2)
    closed = cam.disc_image_centre(camera, centre, normal, rho) - camera.project(centre[None])[0]
    assert abs(closed[0]) == pytest.approx(expected, rel=1e-12) and abs(closed[1]) < 1e-12
    fitted = cam.conic_centre(camera.project(cam.disc_rim(centre, normal, rho)))
    assert np.allclose(fitted, cam.disc_image_centre(camera, centre, normal, rho), atol=1e-9)
    fronto = cam.conic_centre(camera.project(cam.disc_rim(centre, (0.0, 0.0, -1.0), rho)))
    assert np.allclose(fronto, camera.project(centre[None])[0], atol=1e-9)

    report = _run("T050", tmp_path)
    offset = _find(report, "Under full perspective")
    value = offset["value"]
    assert offset["evidence_status"] == NV and offset["counterexample"]
    assert value["closed_form_gap_px"] < 1e-9 and value["fronto_parallel_offset_px"] < 1e-9
    assert value["weak_perspective_offset_px"] < 1e-9 and value["max_offset_px"][-1] > 0.05
    assert value["offset_radius_slope"] == pytest.approx(2.0, abs=1e-3)
    bias = _find(report, "Perspective displacement of circular-marker")
    assert bias["evidence_status"] == NV
    assert bias["value"]["first_order_relative_residual"] < 0.01
    assert bias["value"]["bias_radius_slope"] == pytest.approx(2.0, abs=1e-3)
    assert 1e-5 < bias["value"]["max_abs_chord_bias_m"][-1] < 1e-3
    assert "perspective" in report["hypothesis"].lower() and "perspective" in report["experiment"].lower()


def _under_another_openblas_kernel(code):
    """JSON printed by ``code`` in a subprocess forcing another OpenBLAS kernel than this process runs.

    Sandybridge (no FMA), or Haswell when this process already forces Sandybridge. Skips where NumPy's BLAS is not
    a DYNAMIC_ARCH x86-64 OpenBLAS, whose kernel OPENBLAS_CORETYPE selects.
    """
    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    if "openblas" not in str(blas.get("name")) or "DYNAMIC_ARCH" not in str(blas.get("openblas configuration")) \
            or platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("NumPy's BLAS is not a DYNAMIC_ARCH x86-64 OpenBLAS")
    kernel = "Haswell" if os.environ.get("OPENBLAS_CORETYPE", "").lower() == "sandybridge" else "Sandybridge"
    source = str(Path(observation.__file__).resolve().parents[2])
    environment = dict(os.environ, OPENBLAS_CORETYPE=kernel,
                       PYTHONPATH=os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH")))))
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_t050_first_order_residuals_stay_within_tolerance_on_another_blas_kernel(tmp_path):
    """The first-order residuals carry J_pix's rounding, which depends on the BLAS kernel (camera products and the
    triangulation's SVD); another kernel's residuals stay within the findings' regression tolerance."""
    other = _under_another_openblas_kernel(
        "import json; from ciw.lab import observation as o; print(json.dumps("
        "[o.distortion_study()['first_order_relative_residual'], "
        "o.perspective_study(o.camera_scene())['first_order_relative_residual']]))")
    report = _run("T050", tmp_path)
    for prefix, value in zip(("Uncorrected radial", "Perspective displacement of circular-marker"), other):
        record = _find(report, prefix)
        assert runner._close(record["value"]["first_order_relative_residual"], value, record["regression_tolerance"])


def test_shared_phase_rounding_covariance():
    from ciw.lab import observation_camera as cam

    assert cam.shared_phase_rounding_covariance(0.0, 0.0) == pytest.approx(1 / 12)
    assert cam.shared_phase_rounding_covariance(0.5, 0.0) == pytest.approx(-1 / 24)
    for d in (0.1, 0.37, 0.8):
        assert cam.shared_phase_rounding_covariance(d, 1e-3) == pytest.approx(1 / 12 - d * (1 - d) / 2, abs=1e-4)
    # Gaussian noise decorrelates the rounding: the covariance decays towards 0.
    assert abs(cam.shared_phase_rounding_covariance(0.0, 0.5)) < 1e-4


def test_t051_quantization_noise(tmp_path):
    report = _run("T051", tmp_path)
    _completed(report, [NV, NV, NV, NV, NE])
    additivity = _find(report, "Rounding plus Gaussian")
    assert max(abs(z) for z in additivity["value"]["z"]) <= observation.Z999
    aligned = _find(report, "Without a random grid phase")
    assert "almost no total error" in aligned["claim"] and aligned["counterexample"]
    assert aligned["value"]["sample_variance"] <= aligned["basis"]["checks"][0]["tolerance"] == pytest.approx(1e-4)
    propagation = _find(report, "Linear propagation")["value"]
    assert max(abs(z) for z in propagation["z"]) <= observation.Z999
    shared = _find(report, "A grid phase shared")
    assert shared["counterexample"] and max(abs(z) for z in shared["value"]["z_shared"]) <= observation.Z999
    assert shared["value"]["independent_law_max_abs_z_sigma0"] > observation.Z999
    departures = shared["value"]["predicted_departure"]
    assert min(departures["0"]) < -0.3 and -0.05 < min(departures["0.25"]) < 0


def test_t052_encoder_backlash(tmp_path):
    report = _run("T052", tmp_path)
    _completed(report, [NV, NV, NV, NV, NE])
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
    _completed(report, [NV, NV, NV, NV, NV, NE])
    bounded = _find(report, "On a body rotating")["value"]
    assert bounded["max_transverse_rad"] <= bounded["bound_rad"] * 1.001
    assert bounded["stationary_transverse_rad"] == pytest.approx(0.01, rel=1e-9)
    assert _find(report, "Strapdown bias")["value"]["stationary_relative_error"] < 1e-9
    isotropy = _find(report, "Angle random walk")["value"]
    for name in ("stationary_axis_z", "rotating_axis_z"):
        assert max(abs(z) for row in isotropy[name] for z in row) <= observation.Z999


def test_t054_asynchronous_timestamps(tmp_path):
    report = _run("T054", tmp_path)
    _completed(report, [NV, NV, NV, NV, NE])
    assert any("geomspace(0.0001, 0.01, 7)" in item for item in report["input_data"])
    assert any("Interpolation offset delta = 0.002 s" in item for item in report["input_data"])
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
    assert codes == ["clock_mismatch", "none", "epoch_mismatch"]


def test_t055_dropped_observations(tmp_path):
    report = _run("T055", tmp_path)
    _completed(report, [NV, NV, NV, NV, NV, NE])
    assert _find(report, "A zero-filled stream")["value"] == {"zero_fill": "zero_filled_missing",
                                                               "unbacked": "unbacked_value"}
    detection = _find(report, "Values alone recognize zero fills")
    value = detection["value"]
    assert detection["counterexample"] and value["far_from_zero_mismatch"] == 0
    # A zero-filled stream that equals a complete acquired stream: fills and genuine zeros are indistinguishable.
    assert value["fills_equal_to_genuine_readings"] > 0 and value["identical_stream_gap_m"] == 0.0
    assert value["exact_zero_rule_false_positives"] > 0 and value["neighbour_rule_missed"] > 0
    # Values cannot tell the two streams apart; provenance does.
    assert value["complete_stream_code"] == "accepted" and value["refilled_stream_code"] == "zero_filled_missing"
    refusals = [c for c in detection["basis"]["checks"] if c["reference_kind"] == "refusal"]
    assert [c["observed_refusal"] for c in refusals] == ["none", "zero_filled_missing"]
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
    _completed(report, [NV, NV, NV, NV, NE])
    assert _find(report, "Observations older")["value"]["flag_mismatch"] == 0
    assert _find(report, "Age is refused")["value"] == ["missing_latency", "future_observation"]
    late = _find(report, "Staleness is decided by acquisition-time age")["value"]
    assert late["arrival_age_s"] < late["limit_s"] < late["acquisition_age_s"]
    assert late["acquisition_age_s"] == pytest.approx(late["arrival_age_s"] + late["latency_s"], abs=1e-12)
    constant = _find(report, "Stale-state error")["basis"]["checks"][0]
    assert "acquisition_age(record" in constant["reference"] and constant["observed"] < 1e-12


def test_t057_filter_and_smoother(tmp_path):
    report = _run("T057", tmp_path)
    scipy_label = IV if _scipy() else NV
    _completed(report, [NV, NV, NV, scipy_label, scipy_label, NV, NE])
    rmse = _find(report, "Ensemble position RMSE")["value"]
    assert rmse["smoothed"] <= rmse["filtered"] <= rmse["raw"]
    assert _find(report, "The RTS smoothed")["value"]["min_eigenvalue_filtered_minus_smoothed"] >= -1e-12
    steady = _find(report, "The filter's predicted")
    assert steady["evidence_status"] == scipy_label
    assert _find(report, "Smoothing does not")["value"]["smoothed_worse_fraction"] > 0.1
    degraded = _run("T057", tmp_path / "noscipy", _Without(tmp_path / "noscipy", {"module:scipy"}))
    _completed(degraded, [NV, NV, NV, NV, NV, NV, NE])


def test_t058_frame_and_clock_basis(tmp_path):
    report = _run("T058", tmp_path)
    _completed(report, [NV, NV, NV, NV, NV, NE])
    rigid = _find(report, "A declared rigid frame")
    assert rigid["value"]["exact_gap_m"] == 0.0 and rigid["value"]["distance_change_m"] <= 1.1e-14
    assert rigid["regression_tolerance"]["abs"] >= 1e-14
    assert _find(report, "Declared clock mappings")["value"]["time_gap_s"] == 0.0
    refused = _find(report, "Combining observations")["value"]
    assert refused == {"frame_mismatch": "frame_mismatch", "clock_mismatch": "clock_mismatch",
                       "epoch_mismatch": "epoch_mismatch", "clock_basis_mismatch": "clock_basis_mismatch"}
    # The intake applies no acquisition-age limit (T056); the report says so rather than implying it.
    assert any(assumption.startswith("Acquisition-age staleness (T056) is not applied at the fusion intake")
               for assumption in report["unresolved_assumptions"])


def test_t059_retained_without_admission(tmp_path):
    report = _run("T059", tmp_path)
    _completed(report, [NV, NV, NV, NV, NV, NV, NV, NE])
    assert set(_find(report, "Updating state")["value"].values()) == {"not_admitted"}
    assert _find(report, "Admission as workbench")["domain"] == "actuator_authority"
    assert _find(report, "On a writable store")["value"]["state_admission"] == "synthetic_only"
    store = om.StateStore("encoder_displacement", 0.0, 1.0, read_only=False)
    before = store.digest()
    record = store.retain(observation.example_observation("encoder_displacement"))
    assert store.digest() == before and record["state_admission"] == "not_performed"
    with pytest.raises(om.ObservationRefusal) as refused:
        store.update(record, 0.5)
    assert refused.value.code == "not_admitted" and store.digest() == before
    assert store.admit(record["observation_digest"], "test")["state_admission"] == "synthetic_only"
    assert store.update(record, 1.0)["mean"] == [0.02125]


def test_state_store_defaults_to_read_only(tmp_path):
    report = _run("T059", tmp_path)
    defaults = _find(report, "A state store defaults to read-only")
    assert defaults["evidence_status"] == NV and defaults["value"]["authority_state_admission"] == ["not_performed"]
    assert {code for codes in defaults["value"]["refusals"].values() for code in codes.values()} == {
        "read_only_session"}
    store = om.StateStore("encoder_displacement", 0.0, 1.0)
    assert store.read_only is True and store.authority["state_admission"] == "not_performed"
    record = store.retain(observation.example_observation("encoder_displacement"))
    before = store.digest()
    for action in (lambda: store.admit(record["observation_digest"], "test"), lambda: store.update(record, 0.5)):
        with pytest.raises(om.ObservationRefusal) as refused:
            action()
        assert refused.value.code == "read_only_session"
    assert store.digest() == before
    ledger = om.ObservationLedger()
    with pytest.raises(om.ObservationRefusal) as refused:
        ledger.read_only = False
    assert refused.value.code == "read_only_session" and ledger.read_only is True
    # The store's flag is fixed too, including the ledger that carries it.
    for name, value in (("read_only", False), ("authority", {"state_admission": "synthetic_only"}),
                        ("_ledger", om.ObservationLedger(read_only=False))):
        with pytest.raises(om.ObservationRefusal) as refused:
            setattr(store, name, value)
        assert refused.value.code == "read_only_session", name
    assert store.read_only is True and store.authority["state_admission"] == "not_performed"
    with pytest.raises(om.ObservationRefusal) as refused:
        store.admit(record["observation_digest"], "test")
    assert refused.value.code == "read_only_session" and store.digest() == before
    # Nothing in this section writes a state_admission value outside the CIW vocabulary.
    assert not hasattr(om, "ADMITTED") and om.ADMISSION_VOCABULARY == ("not_performed", "synthetic_only")
    writable = om.StateStore("encoder_displacement", 0.0, 1.0, read_only=False)
    assert writable.authority["state_admission"] == writable.authority["sensor_fusion"] == "synthetic_only"


def test_variance_split_survives_retention_and_digesting(tmp_path):
    report = _run("T059", tmp_path)
    split = _find(report, "The geometry and sensor variance components")["value"]
    assert split["digest_preserved"] is True and split["tampered_update"] == "admission_digest_mismatch"
    assert split["altered_component_changes_digest"] == {"geometry_m2": True, "sensor_m2": True}
    derived = om.chord_to_surface_distance(observation.example_observation("camera_chord_distance"),
                                           observation.CYLINDER_MODEL)
    ledger = om.ObservationLedger(read_only=False)
    kept = ledger.retain(derived)
    assert kept["observation"]["variance_components"] == derived.variance_components
    ledger.admit(kept["observation_digest"], "test")
    assert ledger.admitted(kept["observation_digest"]).variance_components == derived.variance_components
    # Records of modes without components keep the key out of their content, so their digests are unchanged.
    assert "variance_components" not in observation.example_observation("tracker_measurement").record()


def test_state_update_takes_the_declared_variance(tmp_path):
    report = _run("T059", tmp_path)
    split = _find(report, "The geometry and sensor variance components")
    assert split["evidence_status"] == NV and split["value"]["update_posterior_variance_gap_m2"] == 0.0
    assert split["value"]["update_variance_refusal"] == "variance_split_mismatch"
    assert _find(report, "Tampered, unretained")["value"]["invalid_variance"] == "invalid_variance"
    derived = om.chord_to_surface_distance(observation.example_observation("camera_chord_distance"),
                                           observation.CYLINDER_MODEL)
    total = om.declared_variance(derived)
    assert total == pytest.approx(sum(derived.variance_components.values()), rel=1e-15) and total > 0

    def admitted(record, prior_variance=1.0):
        store = om.StateStore(record.mode, 0.12, prior_variance, read_only=False)
        kept = store.retain(record)
        store.admit(kept["observation_digest"], "test")
        return store, kept

    # The declared total is the update variance, whether or not the caller repeats it.
    for given in (None, total):
        store, kept = admitted(derived)
        assert store.update(kept, given)["variance"][0] == pytest.approx(total / (1.0 + total), rel=1e-6)
    # Any other caller variance, zero included, is refused and leaves the state unchanged.
    for given in (1e-12, 0.0, 2 * total):
        store, kept = admitted(derived)
        before = store.digest()
        with pytest.raises(om.ObservationRefusal) as refused:
            store.update(kept, given)
        assert refused.value.code == "variance_split_mismatch" and store.digest() == before
    # A record without a split needs a finite, positive caller variance of the right shape.
    for given in (None, 0.0, -1.0, float("nan"), float("inf"), [1.0, 2.0]):
        store, kept = admitted(observation.example_observation("encoder_displacement"))
        before = store.digest()
        with pytest.raises(om.ObservationRefusal) as refused:
            store.update(kept, given)
        assert refused.value.code == "invalid_variance" and store.digest() == before, given
    # The section-4 store and the fusion intake treat one record's noise the same way.
    assert intake.declared_covariance(derived)[0, 0] == total


def test_intake_import_failure_blocks_only_its_tasks(tmp_path):
    """An intake that cannot be imported blocks T058 and T059, not the other observation tasks."""
    script = textwrap.dedent("""
        import json, sys
        sys.modules["ciw.lab.sensor_fusion_intake"] = None  # any import of the intake now fails
        from ciw.lab import runner
        from ciw.lab.registry import load_queue, section_implementations
        queue = {t["id"]: t for t in load_queue()["tasks"]}
        implementations = section_implementations("observation")
        ctx = runner.Context(sys.argv[1])
        reports = {t: runner.run_task(queue[t], implementations[t], ctx, {}) for t in ("T054", "T058", "T059")}
        print(json.dumps({"registered": sorted(implementations),
                          "states": {t: r["state"] for t, r in reports.items()},
                          "reasons": {t: " ".join(r["failure_modes_checked"]) for t, r in reports.items()}}))
    """)
    source = str(Path(observation.__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")])))
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["registered"] == sorted(IMPLEMENTATIONS)
    assert outcome["states"] == {"T054": "completed", "T058": "blocked", "T059": "blocked"}
    assert all("sensor_fusion_intake" in outcome["reasons"][t] for t in ("T058", "T059"))


def test_intake_refuses_unadmitted_records(tmp_path):
    report = _run("T059", tmp_path)
    fused = _find(report, "The fusion intake refuses a retained but unadmitted record")
    assert fused["evidence_status"] == NV
    assert {k: v["intake"] for k, v in fused["value"]["refusals"].items()} == {
        "not_retained": "not_retained", "not_admitted": "not_admitted", "missing_calibration": "not_admitted"}
    assert fused["value"]["refusals"]["missing_calibration"]["admission"] == "missing_calibration"
    session = intake.FusionSession(read_only=False, dt=intake.DT, q=intake.Q_SPECTRAL)
    session.register_calibration(intake.CalibrationRecord("trk-cal", "tracker", "tracker:cell", 0, 100))
    session.initialize(intake.PRIOR_MEAN, intake.PRIOR_COV, 0)
    route = intake.ObservationIntake(session, intake.FUSION_CLOCK, (intake.TRACKER_CHANNEL,),
                                     frame_mappings=(intake.ROOM_TO_CELL,),
                                     clock_mappings=(intake.ARRIVAL_TO_ACQUISITION, intake.TRACKER_TO_FUSION))
    ledger = om.ObservationLedger(read_only=False)
    record = intake.tracker_records(ticks=1)[1][0]
    digest = ledger.retain(record)["observation_digest"]
    with pytest.raises(intake.IntakeRefusal) as refused:
        route.fuse(ledger, digest)
    assert refused.value.code == "not_admitted" and session.tick == 0 and session.log == []
    assert route.log[-1]["disposition"] == "refused:not_admitted"
    ledger.admit(digest, "test")
    candidate = route.fuse(ledger, digest)
    assert candidate.tick == 1 and route.trace()[0]["observation_digest"] == digest
    assert route.trace()[0]["state_admission"] == "synthetic_only"


def test_intake_maps_frames_and_clocks_or_refuses(tmp_path):
    report = _run("T058", tmp_path)
    mapped = _find(report, "A section-4 record reaches the fusion session only through declared")
    assert mapped["evidence_status"] == NV and mapped["value"]["value_gap_m"] == 0.0
    assert mapped["value"]["tick"] == intake.DEMO_TICKS and mapped["value"]["mappings_recorded"] == 3
    assert mapped["value"]["refusals"] == {
        "unmapped_frame": "unmapped_frame", "unmapped_clock": "unmapped_clock", "unmapped_epoch": "unmapped_clock",
        "arrival_without_latency_mapping": "unmapped_clock", "latency_mismatch": "latency_mismatch",
        "combined_latency_and_synchronization": "unmapped_clock", "off_tick_grid": "off_tick_grid"}
    # The arrival stamp reaches the fusion tick only through the declared latency and synchronization.
    session = intake.FusionSession(read_only=False, dt=intake.DT, q=intake.Q_SPECTRAL)
    session.register_calibration(intake.CalibrationRecord("trk-cal", "tracker", "tracker:cell", 0, 100))
    route = intake.ObservationIntake(session, intake.FUSION_CLOCK, (intake.TRACKER_CHANNEL,),
                                     frame_mappings=(intake.ROOM_TO_CELL,),
                                     clock_mappings=(intake.ARRIVAL_TO_ACQUISITION, intake.TRACKER_TO_FUSION))
    ledger = om.ObservationLedger(read_only=False)
    record = intake.tracker_records(ticks=3)[1][2]
    digest = ledger.retain(record)["observation_digest"]
    ledger.admit(digest, "test")
    reading, link = route.convert(ledger, digest)
    assert reading.tick == 3 and reading.frame_id == "world" and reading.origin_frame_id == "tracker:cell"
    assert reading.value == (-record.value[1] + 0.25, record.value[0] - 0.5)
    assert link["mappings"] == [intake.ROOM_TO_CELL.identity(), intake.ARRIVAL_TO_ACQUISITION.identity(),
                                intake.TRACKER_TO_FUSION.identity()]
    # A basis change must be a latency mapping on one clock and epoch, checked against the record. A record with a
    # one-tick latency reaches its tick 3 through the two-step declaration; one mapping that changes clock and basis
    # together is refused whether it forgets the latency (it would land one tick late) or folds it into its offset.
    late = dataclasses.replace(record, latency_s=intake.DT, time_s=record.time_s - intake.LATENCY_S + intake.DT,
                               sequence=77, raw_ref="raw:tracker:late")
    late_digest = ledger.retain(late)["observation_digest"]
    ledger.admit(late_digest, "test")

    def route_with(*clock_mappings):
        return intake.ObservationIntake(session, intake.FUSION_CLOCK, (intake.TRACKER_CHANNEL,),
                                        frame_mappings=(intake.ROOM_TO_CELL,), clock_mappings=clock_mappings)

    def clock(source, target, offset, reference="declared"):
        return om.ClockMapping(*source, *target, 1.0, offset, reference)

    tracker_arrival = ("clock:tracker", "epoch:run-0", "arrival")
    tracker_acquisition = ("clock:tracker", "epoch:run-0", "acquisition")
    fusion = ("clock:fusion", "epoch:fusion-0", "acquisition")
    latency = clock(tracker_arrival, tracker_acquisition, -intake.DT)
    assert route_with(latency, intake.TRACKER_TO_FUSION).convert(ledger, late_digest)[0].tick == 3
    for combined in (clock(tracker_arrival, fusion, intake.SYNC_OFFSET_S),
                     clock(tracker_arrival, fusion, intake.SYNC_OFFSET_S - intake.DT)):
        with pytest.raises(intake.IntakeRefusal) as refused:
            route_with(combined).convert(ledger, late_digest)
        assert refused.value.code == "unmapped_clock"
    # A detour back to the arrival basis cannot add an unchecked offset between two checked latency mappings.
    detour = (intake.ARRIVAL_TO_ACQUISITION, clock(tracker_acquisition, tracker_arrival, intake.LATENCY_S + intake.DT),
              clock(tracker_arrival, tracker_acquisition, -intake.LATENCY_S, "declared latency, again"),
              intake.TRACKER_TO_FUSION)
    with pytest.raises(intake.IntakeRefusal) as refused:
        route_with(*detour).convert(ledger, digest)
    assert refused.value.code == "unmapped_clock"
    # An intake whose clock does not match the session step, or a channel of another geometry, is refused.
    with pytest.raises(intake.IntakeRefusal) as refused:
        intake.ObservationIntake(intake.FusionSession(read_only=False), intake.FUSION_CLOCK)
    assert refused.value.code == "malformed_intake"
    with pytest.raises(intake.IntakeRefusal) as refused:
        intake.ObservationIntake(session, intake.FUSION_CLOCK, (intake.TRACKER_CHANNEL,), geometry="intrinsic")
    assert refused.value.code == "channel_geometry_mismatch"


def test_next_steps_are_forward_research_questions(tmp_path):
    """No completed task in this section points at a queue task that has already run (R04/R09)."""
    ctx = runner.Context(tmp_path)
    for task_id in sorted(IMPLEMENTATIONS):
        report = runner.run_task(QUEUE[task_id], IMPLEMENTATIONS[task_id], ctx, {})
        assert report["state"] == "completed", task_id
        assert report["recommended_next_task"].startswith("Deferred research question: "), task_id


def test_reports_regenerate_identically(tmp_path):
    for task_id in ("T045", "T052", "T058"):
        first = _run(task_id, tmp_path / "a")
        second = _run(task_id, tmp_path / "b")
        assert first["report_id"] == second["report_id"]
