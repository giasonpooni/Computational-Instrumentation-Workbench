"""Lyapunov runtime experiments T101-T114.

Provider-free parts (exact references, the documented-rule re-derivation, the
ISS branch, the residual adapter and the servo specification) always run.
Parts that execute the pinned PLSR runtime need a Python 3.12+ interpreter with
it installed: set CIW_LAB_PLSR_PYTHON, or run the tests under such an
interpreter; otherwise they are skipped.
"""
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ciw.lab import lyapunov as L
from ciw.lab import lyapunov_reference as R
from ciw.lab import lyapunov_research as X
from ciw.lab.lyapunov_provider import PLSR_ROLE, ProviderRefusal, manifest, run_plsr
from ciw.lab.registry import load_implementations, load_queue
from ciw.lab.report import validate_report
from ciw.lab.runner import Context, run_queue, run_task

PROVIDER_TASKS = [f"T1{n:02d}" for n in range(1, 12)] + ["T113"]


def _plsr_python():
    configured = os.environ.get("CIW_LAB_PLSR_PYTHON")
    if configured:
        return configured if Path(configured).exists() else None
    if sys.version_info >= (3, 12) and importlib.util.find_spec("lyapunov") is not None:
        return sys.executable
    return None


PLSR_PYTHON = _plsr_python()
needs_provider = pytest.mark.skipif(PLSR_PYTHON is None, reason="set CIW_LAB_PLSR_PYTHON to a Python 3.12+ "
                                                                "interpreter with the pinned PLSR runtime")


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Run every provider-dependent task once with the bound runtime."""
    if PLSR_PYTHON is None:
        pytest.skip("PLSR provider interpreter not configured")
    directory = tmp_path_factory.mktemp("lyapunov")
    run_queue(directory, task_ids=PROVIDER_TASKS, providers={PLSR_ROLE: PLSR_PYTHON})
    return {task_id: validate_report(json.loads((directory / "reports" / f"{task_id}.json").read_text()))
            for task_id in PROVIDER_TASKS}


def _run(task_id, tmp_path, providers=None):
    implementations, _ = load_implementations()
    item = next(t for t in load_queue()["tasks"] if t["id"] == task_id)
    return run_task(item, implementations[task_id], Context(tmp_path, providers), {})


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert matches, f"no finding starting with {prefix!r}"
    return matches[0]


# Provider-free references ------------------------------------------------------

def test_reference_witness_is_exactly_indefinite():
    exact = L._witness_exact()
    assert exact["exact_form_units"] == [[-4.0, 5.0], [5.0, -6.0]]
    assert exact["exact_det_units2"] == -1.0 and exact["exact_class"] == "has_positive_eigenvalue"
    assert exact["float_form_units"] == [[-4.0, 4.0], [4.0, -6.0]]
    assert exact["float_class"] == "negative_definite"
    assert R.resolution(*L._subnormal_witness()[:2]) == 0.0


def test_exact_classification_and_resolution_formula():
    assert R.exact_class(R.fractions(np.diag([-1.0, -2.0]))) == "negative_definite"
    assert R.exact_class(R.fractions(np.diag([0.0, -2.0]))) == "negative_semidefinite"
    assert R.exact_class(R.fractions(np.array([[-1.0, 2.0], [2.0, -1.0]]))) == "has_positive_eigenvalue"
    assert R.lambda_max_below(R.fractions(np.diag([-1.0, -2.0])), Fraction(-1, 2))
    assert not R.lambda_max_below(R.fractions(np.diag([-1.0, -2.0])), -1.0)
    # n = 2, A = P = I: resolution = 2 (2 gamma_5 2) + 8 u 2 with gamma_5 = 5u / (1 - 5u).
    gamma = 5 * R.U / (1 - 5 * R.U)
    assert R.resolution(np.eye(2), np.eye(2)) == pytest.approx(8 * gamma + 16 * R.U, rel=1e-15)
    assert L.EPSILON_STAR == pytest.approx(10 * np.finfo(float).eps, rel=1e-14)


def test_eigvalsh_scaling_window():
    inside, _ = L._eigvalsh_window()
    assert inside and all(inside)


def test_level_gate_prediction():
    rows = L.level_scan()
    missed, spurious = L._level_counts(rows, "documented_exceeded")
    assert (len(rows), missed, spurious) == (68, 5, 9)
    first = next(r for r in rows if r["exact_exceeded"] and not r["documented_exceeded"])
    assert 2 * first["e"] < -1074  # s^2 underflows to zero while V = 2^(p + 2e) is representable
    assert R.documented_level_exceeded(1.0, 0, 0.5) and not R.documented_level_exceeded(1.0, 0, 2.0)


def test_edge_case_exact_classes():
    cases = {c["name"]: c for c in L.edge_cases()}
    assert all(c["exact_zero"] and c["exact_class"] == "negative_semidefinite"
               for c in cases.values() if c["group"] == "skew P=I")
    assert all(c["exact_class"] == "has_positive_eigenvalue" for c in cases.values() if c["group"] == "skew P=SPD")
    jordan = {name: c["exact_class"] for name, c in cases.items() if c["group"] == "Jordan P=I"}
    assert jordan["Jordan lambda=0.49, P=I"] == "has_positive_eigenvalue"
    assert jordan["Jordan lambda=0.5, P=I"] == "negative_semidefinite"
    assert jordan["Jordan lambda=0.51, P=I"] == "negative_definite"
    assert L.expected_codes("negative_semidefinite", "[0, 1) res") == {"NUMERICAL_INCONCLUSIVE"}
    assert L.expected_codes("negative_definite", "below -2 res") == {"CERTIFIED_WITH_MARGIN"}


def test_conversion_scan():
    scan = L.conversion_scan(400)
    collision = scan["witnesses"].get("collision 0.001")
    assert scan["neighbour_collisions"]["0.001"] >= 1 and collision is not None
    bound = collision["bound"]
    assert math.nextafter(bound, math.inf) * 1e-3 == bound * 1e-3
    assert scan["formula_disagreements"]["0.001"] >= 1


def test_documented_decision_order():
    predictions = L._path_cases(L.status_paths(), [])[1]
    assert [predictions[f"required_margin (margin 2)#{j}"] for j in range(6)] == (
        ["CERTIFIED_WITH_MARGIN"] * 3 + ["MARGIN_LOW"] * 3)
    assert predictions["matrix scale k, A = -2^k I#3"] == "NUMERICAL_OVERFLOW"
    assert predictions["theta, box [-1, 1]#4"] == "OUTSIDE_PARAMETER_BOX"
    assert predictions["stability a, A = [[a, 1], [-1, a]]#3"] == "NUMERICAL_INCONCLUSIVE"
    assert predictions["direction phi, A = diag(-1, 1)#4"] == "NOT_CERTIFIED"
    assert {p for p in predictions.values()} >= set(R.RUNTIME_CODES) - {"CERTIFICATE_NOT_POSITIVE"}


def test_documented_rule_is_monotone():
    for member in L.near_threshold_family(108, 12):
        info = R.documented_code(member["A"], member["P"], member["x"])
        steps = [dict(code=R.documented_code(member["A"], member["P"], member["x"], required_margin=r)["code"],
                      meets_required_margin=info["margin"] > max(r, info["resolution"]),
                      inequality_certified=info["margin"] > info["resolution"])
                 for r in L.margin_grid(info["margin"], info["resolution"])]
        assert sum(L.monotonicity_violations(steps).values()) == 0


def test_numpy_misreads_exact_jordan_block():
    cases = [c for c in L.adversarial_cases() if c["group"] == "Jordan"]
    largest = max(cases, key=lambda c: c["A"].shape[0])
    lam = -largest["exact_spectrum"][0]
    # A = T J T^-1 holds exactly (asserted while building); the float spectrum moves by about eps^(1/n).
    error = abs(largest["numpy_abscissa"] + lam)
    assert error > 1e3 * np.finfo(float).eps * np.max(np.abs(largest["A"]))


def test_bridge_refuses_unusable_interpreter():
    if importlib.util.find_spec("lyapunov") is not None and sys.version_info >= (3, 12):
        pytest.skip("this interpreter hosts the runtime")
    with pytest.raises(ProviderRefusal) as caught:
        run_plsr(sys.executable, [])
    assert caught.value.code in ("PLSR_PYTHON_UNSUPPORTED", "PLSR_UNAVAILABLE")


def test_provider_tasks_are_partial_without_the_provider(tmp_path):
    report = _run("T106", tmp_path)
    assert report["state"] == "partial"
    assert "not bound" in report["experiment"]
    labels = {f["claim"]: f["evidence_status"] for f in report["findings"]}
    assert labels["A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation"] == "not_established"
    assert labels["The documented decision order, re-derived in CIW, assigns all nine codes to the constructed inputs"] \
        == "numerically_verified"
    assert report["provider_runtime_identity"]["provider"]["executed"] is False


def test_t112_iss_branch(tmp_path):
    report = _run("T112", tmp_path)
    assert report["state"] == "completed"
    ratios = _finding(report, "Simulated sup sqrt(V)")
    assert ratios["evidence_status"] == "numerically_verified" and max(ratios["value"].values()) < 1.0
    assert _finding(report, "Quadratic ISS-Lyapunov bound")["evidence_status"] == "analytic"
    assert _finding(report, "In one dimension")["value"]["ratio"] == pytest.approx(1.0, abs=1e-9)
    assert _finding(report, "The series matrix exponential")["evidence_status"] == "independently_verified"
    assert _finding(report, "The disturbance bound")["evidence_status"] == "not_established"
    assert _finding(report, "The ISS bound defines a safe")["domain"] == "machine_safety"
    assert report["physical_validation_status"]["status"] == "not_established"


def test_adapter_keeps_metadata_outside():
    windows = X.adapter_windows()
    outcomes = [X.adapt(w["envelope"]) for w in windows]
    host = [o["host_status"] for o in outcomes if "host_status" in o]
    assert sorted(host) == ["CERTIFICATE_EXPIRED", "INVALID_SENSOR_DATA", "MODEL_MISMATCH", "STALE_STATE"]
    samples = [o["sample"] for o in outcomes if "sample" in o]
    assert len(samples) == 3 and all(tuple(s) == X.SAMPLE_FIELDS for s in samples)
    envelope = windows[0]["envelope"]
    assert X.metadata_leaks({"x": samples[0]["x"], "theta": samples[0]["theta"]}, envelope) == []
    assert X.metadata_leaks({"x": [0.0], "note": f"from {envelope['sensor_id']}"}, envelope) == [
        "value encoder-axis-7"]
    assert "key calibration_ref" in X.metadata_leaks({"calibration_ref": 1.0}, envelope)


def test_t114_servo_pilot_spec(tmp_path):
    report = _run("T114", tmp_path)
    assert report["state"] == "completed"
    grid = _finding(report, "The unit-balanced nominal P")
    assert grid["evidence_status"] == "numerically_verified" and grid["value"]["not_negative_definite"] == 0
    assert _finding(report, "With Q = I the nominal-model P")["counterexample"]
    for domain in ("machine_safety", "actuator_authority", "production_acceptance", "industrial_readiness",
                   "physical", "calibration"):
        assert [f["evidence_status"] for f in report["findings"] if f["domain"] == domain] == ["not_established"]
    spec = json.loads((tmp_path / "artifacts" / "T114" / "servo-pilot-spec.json").read_text())
    assert set(X.SERVO_SPEC_SECTIONS) <= set(spec)
    assert spec["authority_and_safety"]["actuator_authority"].startswith("none")


# Provider-backed ---------------------------------------------------------------

@needs_provider
def test_bridge_verifies_the_source_pin():
    tampered = manifest()
    tampered["files"]["runtime.py"] = "0" * 64
    with pytest.raises(ProviderRefusal) as caught:
        run_plsr(PLSR_PYTHON, [], pin=tampered)
    assert caught.value.code == "PLSR_SOURCE_MISMATCH" and "runtime.py" in str(caught.value)
    wrong = dict(manifest(), package_version="0.0.0")
    with pytest.raises(ProviderRefusal, match="PLSR_VERSION_MISMATCH"):
        run_plsr(PLSR_PYTHON, [], pin=wrong)
    identity = run_plsr(PLSR_PYTHON, [{"id": "c", "op": "constants"}])["identity"]
    assert identity["files_verified"] == identity["files_pinned"] == 20
    assert identity["commit"] == "19ea6967060166ba09db6cd4563bd87bd6b3d196"


@needs_provider
def test_t101_resolution_floor(reports):
    report = reports["T101"]
    assert report["state"] == "completed"
    assert _finding(report, "PLSR decrease_resolution equals")["evidence_status"] == "independently_verified"
    inside = _finding(report, "Inside the binary64 normal range")
    assert inside["value"]["threshold_mismatches"] == 0 and inside["value"]["max_normalised_resolution_deviation"] == 0.0
    witness = _finding(report, "Below the normal range")
    assert witness["value"]["code"] == "CERTIFIED_WITH_MARGIN" and witness["value"]["exact_det_units2"] == -1.0
    assert witness["counterexample"]["statement"].startswith("The float64 resolution floor")


@needs_provider
def test_t102_power_of_two_scaling(reports):
    report = reports["T102"]
    inside = _finding(report, "Power-of-two scaling of (A, P, x) inside")
    assert inside["value"]["code_flips"] == 0 and inside["value"]["ratio_changes"] == 0
    assert _finding(report, "Scaling the subnormal witness")["value"] == {
        "scaled_code": "CERTIFIED_WITH_MARGIN", "unit_code": "DECREASE_NOT_DEFINITE"}
    assert all(f["evidence_status"] != "not_established" for f in report["findings"])


@needs_provider
def test_t103_overflow_underflow(reports):
    report = reports["T103"]
    assert _finding(report, "PLSR decides the level gate")["value"]["disagreements"] == 0
    assert _finding(report, "The PLSR level gate misses")["value"]["missed"] == 5
    assert _finding(report, "The PLSR level gate reports")["value"]["spurious"] == 9
    assert _finding(report, "Non-finite states")["value"] == {"inf": "raises ValueError", "nan": "raises ValueError"}
    theta = _finding(report, "A finite in-box theta")
    assert theta["value"]["theta:+1e308,c=2"] == "raises ValueError"
    assert theta["value"]["theta:+1e308,c=1"] == "NUMERICAL_OVERFLOW"
    assert _finding(report, "A subnormal plant matrix")["counterexample"]


@needs_provider
def test_t104_semidefinite_edges(reports):
    report = reports["T104"]
    sound = _finding(report, "PLSR never certifies a semidefinite")
    assert sound["evidence_status"] == "independently_verified" and sound["value"]["violations"] == 0
    assert _finding(report, "Every edge-case code")["value"]["unexpected_codes"] == 0
    refusals = _finding(report, "PLSR's Lyapunov solver and certificate constructor")["value"]
    assert refusals["solve:psdQ"] == refusals["quadratic:psd"] == "raises ValueError"


@needs_provider
def test_t105_unit_scales(reports):
    report = reports["T105"]
    assert _finding(report, "Interior and boundary stiffness samples")["value"]["mismatches"] == 0
    assert _finding(report, "A parameter just above the SI bound")["value"] == {
        "SI": "OUTSIDE_PARAMETER_BOX", "x1e-3": "CERTIFIED_WITH_MARGIN"}
    assert _finding(report, "A parameter exactly on the SI bound")["value"] == {
        "SI": "CERTIFIED_WITH_MARGIN", "x1e-3": "OUTSIDE_PARAMETER_BOX"}
    light = _finding(report, "The light-damping plant's verdict")
    assert light["value"]["m, s, N/m"] == "CERTIFIED_WITH_MARGIN"
    assert light["value"]["m, ms, N/m"] == "NUMERICAL_INCONCLUSIVE"
    assert _finding(report, "The declared stiffness box")["evidence_status"] == "not_established"


@needs_provider
def test_t106_status_coverage(reports):
    report = reports["T106"]
    coverage = _finding(report, "All nine runtime-status-v1 codes")
    assert coverage["evidence_status"] == "independently_verified"
    assert sorted(coverage["value"]["codes"]) == sorted(R.RUNTIME_CODES)
    host = _finding(report, "The runtime refuses to emit")["value"]
    assert all(v == {"Verdict": "raises ValueError", "require_status": "raises ValueError"} for v in host.values())
    constants = _finding(report, "Pinned runtime constants")
    assert constants["value"]["DECREASE_RESOLUTION_FACTOR"] == 1.0
    assert constants["evidence_status"] == "numerically_verified"


@needs_provider
def test_t107_inconclusive_band(reports):
    report = reports["T107"]
    assert _finding(report, "No near-boundary case receives")["value"]["violations"] == 0
    assert _finding(report, "Beyond two resolutions")["value"]["unresolved"] == 0
    assert _finding(report, "With a declared margin of three")["value"]["certified"] == 0
    assert _finding(report, "MARGIN_LOW appears exactly")["value"]["margin_low_observed"] is True


@needs_provider
def test_t108_margin_monotonicity(reports):
    report = reports["T108"]
    totals = _finding(report, "Increasing required_margin")["value"]
    assert sum(totals[k] for k in ("passing_regained", "meets_regained", "noncertifying_code_changed",
                                   "inequality_changed")) == 0
    assert _finding(report, "The switch from CERTIFIED_WITH_MARGIN")["value"]["mismatches"] == 0
    assert set(_finding(report, "Negative and non-finite")["value"].values()) == {"raises ValueError"}


@needs_provider
def test_t109_adversarial_eigenvalues(reports):
    report = reports["T109"]
    assert _finding(report, "Every certifying PLSR verdict")["value"]["violations"] == 0
    agreement = _finding(report, "PLSR Lyapunov solutions agree")
    assert agreement["evidence_status"] == "independently_verified"
    assert agreement["value"]["max_relative_difference"] < 1e-6
    assert _finding(report, "Certified non-normal plants")["value"]["max_ratio"] <= 1.0
    assert _finding(report, "With P = I the non-normal plants")["value"]["mismatches"] == 0


@needs_provider
def test_t110_time_interpretation(reports):
    report = reports["T110"]
    assert _finding(report, "PLSR certifies each matrix")["value"]["mismatches"] == 0
    differing = _finding(report, "The two time interpretations")["value"]
    assert differing["differing"] == differing["off_quadrant_matrices"] == 20
    table = _finding(report, "Diagonal plants with P = I")["value"]
    assert table["diag(-1.5, -0.25) continuous"] == "CERTIFIED_WITH_MARGIN"
    assert table["diag(-1.5, -0.25) discrete"] == "NOT_CERTIFIED"
    assert _finding(report, "A discrete plant refuses")["value"]["code"] == "raises ValueError"


@needs_provider
def test_t111_routes(reports):
    report = reports["T111"]
    assert _finding(report, "PLSR Lyapunov solutions agree")["value"]["max_relative_difference"] < 1e-9
    assert _finding(report, "The PLSR matrix route certifies")["value"]["mismatches"] == 0
    thin = _finding(report, "The scalar route sees decrease")
    assert thin["value"]["plsr_codes"] == {"DECREASE_NOT_DEFINITE": 64} and thin["counterexample"]


@needs_provider
def test_t113_residual_adapter(reports):
    report = reports["T113"]
    assert report["state"] == "completed"
    codes = _finding(report, "Forwarded samples receive")["value"]
    assert codes["nominal theta 0.1"] == "CERTIFIED_WITH_MARGIN"
    assert codes["estimate outside box theta 0.9"] == "OUTSIDE_PARAMETER_BOX"
    assert set(_finding(report, "The kernel refuses every host-owned code")["value"].values()) == {"raises ValueError"}
    assert _finding(report, "Kernel payloads built by the adapter carry only")["value"]["metadata_leaks"] == []
    assert _finding(report, "The synthetic residual statistics")["evidence_status"] == "not_established"
