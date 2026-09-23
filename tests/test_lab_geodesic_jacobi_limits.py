"""Regression tests for the geodesic/Jacobi limit, invariance and counterexample tasks (T010-T018).

The nine tasks run once per module through ``ciw.lab.runner.run_task`` with one
shared context, exactly as a queue run does; the tests assert labels, key
numbers, counterexamples and refusals from the retained reports.
"""
import json
import math

import numpy as np
import pytest

from ciw.lab import geodesic_jacobi_limits as gjl
from ciw.lab import geodesic_jacobi_limits_core as core
from ciw.lab import registry, runner
from ciw.lab.report import FIELD_NAMES, validate_report
from ciw.lab.surfaces import Plane, Reparametrized, Sphere, SurfaceRefusal, Torus

TASKS = ("T010", "T011", "T012", "T013", "T014", "T015", "T016", "T017", "T018")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lab-gjl")
    queue = {t["id"]: t for t in registry.load_queue()["tasks"]}
    ctx = runner.Context(directory)
    reports = {}
    for task_id in TASKS:
        # Only this section's registrations are used; other sections are never imported here.
        reports[task_id] = validate_report(runner.run_task(queue[task_id], registry._REGISTRY[task_id], ctx, {}))
    return directory, reports


def _findings(report):
    return {f["claim"]: f for f in report["findings"]}


def _labelled(report, fragment):
    matches = [f for f in report["findings"] if fragment in f["claim"]]
    assert len(matches) == 1, fragment
    return matches[0]


def test_reports_answer_every_question_and_never_claim_physical_validation(run):
    directory, reports = run
    for task_id, report in reports.items():
        assert report["state"] == "completed", (task_id, report["experiment"])
        assert all(name in report for name in FIELD_NAMES)
        for name in ("hypothesis", "mathematical_model", "observation_model", "expected_invariant", "experiment",
                     "numerical_result", "uncertainty", "recommended_next_task"):
            assert isinstance(report[name], str) and len(report[name]) > 20, (task_id, name)
        assert report["failure_modes_checked"] and report["unresolved_assumptions"] and report["input_data"]
        assert report["physical_validation_status"]["status"] == "not_established"
        assert report["generated_artifacts"], task_id
        for record in report["findings"]:
            assert record["assigned_by"] == "ciw.lab.evidence"
            if record["evidence_status"] != "not_established" and not isinstance(record["value"], (str, type(None))):
                assert "regression_tolerance" in record, (task_id, record["claim"])
        for artifact in report["generated_artifacts"]:
            assert (directory / artifact["path"]).is_file()


def test_t010_near_focus_and_post_focus_counterexamples(run):
    report = run[1]["T010"]
    equator = _labelled(report, "outer equator the separation at the conjugate point scales as eps^3")
    assert equator["evidence_status"] == "numerically_verified"
    assert equator["value"]["chord_exponent"] == pytest.approx(3.0, abs=0.01)
    assert equator["counterexample"]["witness"]["s_star"] == pytest.approx(math.pi * math.sqrt(3.0), abs=1e-7)
    generic = _labelled(report, "generic torus geodesic")
    assert generic["value"]["chord_exponent"] == pytest.approx(2.0, abs=0.05)
    divergence = _labelled(report, "diverges like 1/|s - s*|")
    assert divergence["value"]["slope_generic_eps_0.01"] == pytest.approx(-1.0, abs=0.1)
    assert divergence["value"]["relative_error_at_s_star_minus_h_eps_0.04"] > 1.0
    assert "whole path" in divergence["counterexample"]["statement"]
    inversion = _labelled(report, "inverts sign")["value"]
    for key in ("equator", "generic"):
        assert inversion[key]["j_before"] > 0 > inversion[key]["j_after"]
        assert inversion[key]["signed_before"] > 0 > inversion[key]["signed_after"]
        assert inversion[key]["ratio_after"] == pytest.approx(1.0, abs=0.05)
    monotone = _labelled(report, "does not grow monotonically")
    assert monotone["value"] < 1e-3 and "monotonically" in monotone["counterexample"]["statement"]
    sphere = _labelled(report, "refocuses exactly")
    assert sphere["value"]["uniform_relative_error"] == pytest.approx(2 * math.sin(0.01) / 0.02 - 1, rel=1e-12)
    assert abs(sphere["value"]["relative_error_at_pi_minus_h"]) < 1e-4
    mixed = _labelled(report, "combined lateral+heading")
    assert mixed["value"]["coefficient"] == pytest.approx(0.5, abs=1e-3)
    assert all(f["evidence_status"] == "numerically_verified" for f in report["findings"])


def test_t011_chart_invariance_and_fold_amplification(run):
    report = run[1]["T011"]
    converged = _labelled(report, "agree in every chart")
    assert converged["evidence_status"] == "numerically_verified"
    assert converged["value"]["max_adaptive_error"] < 1e-8
    assert _labelled(report, "fourth-order convergence")["value"]["min_order"] >= 3.7
    fold = _labelled(report, "near-fold chart multiplies")
    assert fold["evidence_status"] == "numerically_verified"
    assert min(fold["value"]["error_factor_mu_0.1_at_N_256"].values()) > 1e3
    assert fold["counterexample"]["witness"]["plane_error_base"] < 1e-12 < fold["counterexample"]["witness"]["plane_error_fold"]
    assert _labelled(report, "identity chart")["value"] == 0.0


def test_chart_maps_have_exact_derivatives():
    charts = [core.PolynomialWarp((0.4, -0.2), 0.3), core.QuadraticShear((0.4, -0.2), 0.8),
              core.ExponentialStretch((0.4, -0.2), 0.6), core.NearFold((0.4, -0.2), 0.1, 0),
              core.NearFold((0.4, -0.2), 0.1, 1), core.IdentityChart()]
    a = np.array([0.23, -0.31])
    step = 1e-6
    for chart in charts:
        numeric_j = np.column_stack([(chart.forward(a + step * e) - chart.forward(a - step * e)) / (2 * step)
                                     for e in np.eye(2)])
        assert np.allclose(chart.jacobian(a), numeric_j, atol=1e-8), chart.name
        for k, e in enumerate(np.eye(2)):
            numeric_h = (chart.jacobian(a + step * e) - chart.jacobian(a - step * e)) / (2 * step)
            assert np.allclose(chart.hessian(a)[:, :, k], numeric_h, atol=1e-7), chart.name
        assert np.allclose(chart.inverse(chart.forward(a)), a, atol=1e-12), chart.name
    with pytest.raises(SurfaceRefusal, match="outside the exponential-stretch chart domain"):
        core.ExponentialStretch((0.0, 0.0), 0.6).inverse(np.array([-2.0, 0.0]))
    with pytest.raises(SurfaceRefusal, match="mu > 0"):
        core.NearFold((0.0, 0.0), 0.0)
    # Same geometric start in a warped chart: the pullback tangent has unit length.
    torus = Torus(2.0, 1.0)
    chart = core.NearFold((0.5, 1.26), 0.1, 0)
    a0, ta = core.chart_start(torus, chart, (0.0, 0.5), 0.7)
    assert Reparametrized(torus, chart).speed_squared(a0, ta) == pytest.approx(1.0, abs=1e-12)


def test_t012_frame_invariance_and_refusal(run):
    report = run[1]["T012"]
    assert _labelled(report, "Ambient rotations")["value"] < 1e-11
    assert _labelled(report, "rotate exactly")["value"] < 1e-12
    assert _labelled(report, "reference tangent basis")["value"]["max_state_difference"] < 1e-11
    orientation = _labelled(report, "orientation-reversing")
    assert orientation["value"] == pytest.approx(-1.0, abs=1e-6)
    assert "every change of orthonormal tangent basis" in orientation["counterexample"]["statement"]
    refusal = _labelled(report, "improper rotation")
    assert refusal["evidence_status"] == "numerically_verified"
    assert refusal["basis"]["checks"][0]["observed_refusal"] == "Frame change requires a proper rotation matrix"


def test_t013_flat_limit(run):
    report = run[1]["T013"]
    exact = _labelled(report, "coincide exactly")
    assert exact["value"]["jacobi_max_difference"] == 0.0
    chords = _labelled(report, "do not imply equal chords")["value"]
    assert chords["chord_plane"] == pytest.approx(3.0, abs=1e-12)
    assert chords["chord_cylinder"] < 2.6
    decay = _labelled(report, "decays like 1/R")["value"]
    assert decay["fitted_exponent"] == pytest.approx(decay["closed_form_exponent"], abs=1e-6)
    assert -1.0 < decay["fitted_exponent"] < -0.95
    assert _labelled(report, "chord deficit")["value"] == pytest.approx(-2.0, abs=0.05)
    assert _labelled(report, "physical cylinder")["evidence_status"] == "not_established"


def test_t014_reversal_and_truncation(run):
    report = run[1]["T014"]
    orders = _labelled(report, "Reversal error orders")
    assert orders["evidence_status"] == "numerically_verified"
    for label, order in orders["value"].items():
        expected = {"euler": 1, "midpoint": 3, "rk4": 5}[label.split(": ")[1]]
        assert order == pytest.approx(expected, abs=0.15), label
    assert _labelled(report, "dyadic grid")["value"] == {"euler": 0.0, "midpoint": 0.0, "rk4": 0.0}
    decimal = _labelled(report, "decimal truncation")
    assert decimal["value"]["differs"] is True and decimal["value"]["max_abs_difference"] < 1e-12
    witness = decimal["counterexample"]["witness"]
    assert witness["h_first"] != witness["h_direct"] or witness["h_rest"] != witness["h_direct"]
    assert _labelled(report, "Adaptive restart")["value"] <= 10.0


def test_t015_long_horizon_drift(run):
    report = run[1]["T015"]
    adaptive = _labelled(report, "grows linearly with length")["value"]
    assert all(abs(v - 1.0) < 0.2 for v in adaptive.values())
    position = _labelled(report, "position error grows")["value"]
    assert position["rk4"] == pytest.approx(1.0, abs=0.1) and position["adaptive"] == pytest.approx(2.0, abs=0.1)
    rk4 = _labelled(report, "oscillation-dominated")
    assert max(rk4["value"]["exponents"].values()) < 0.5 and rk4["counterexample"]
    euler = _labelled(report, "changes the orbit type")["value"]
    assert euler["max_energy_error"] < 0.1 and euler["theta_max_abs"] > euler["theta_turning"] + 1.0
    assert _labelled(report, "leaves the polar chart")["value"] < 320.0


def test_t016_negative_curvature_is_not_stiffness(run):
    report = run[1]["T016"]
    assert _labelled(report, "grow like sinh(kL)/k")["value"]["max_adaptive_relative_error"] < 1e-8
    assert _labelled(report, "grows like k^5")["value"]["exponent"] == pytest.approx(5.0, abs=0.2)
    steps = _labelled(report, "k^(5/4)")["value"]
    assert steps["rk4_exponent"] == pytest.approx(1.25, abs=0.1)
    stiffness = _labelled(report, "not stiffness")["value"]
    assert min(stiffness["steps_required_over_stability_limit"]) >= 5.0
    implicit = _labelled(report, "Implicit midpoint")
    assert min(implicit["value"]["implicit_over_rk4_steps"]) > 1.0
    assert implicit["value"]["sign_changes_at_kh"][1] >= 1 and "A-stable" in implicit["counterexample"]["statement"]
    saddle = _labelled(report, "Saddle(c)")
    assert saddle["value"]["exponent_c_16_to_256"] == pytest.approx(math.sqrt(2.0), abs=0.1)
    assert saddle["counterexample"]["witness"]["log_j_head"] < 0.1 * saddle["counterexample"]["witness"]["sqrt_peak_times_L"]


def test_t017_validity_domains(run):
    report = run[1]["T017"]
    generic = _labelled(report, "shrinks to zero at the conjugate point")["value"]
    assert abs(generic["C2_at_s_star"]) > 0.1 and generic["eps_max_at_s_star"] < 1e-6
    probe = _labelled(report, "predicted validity boundary")["value"]
    assert probe["half"]["relative_error"] <= gjl.TAU <= probe["double"]["relative_error"]
    sphere = _labelled(report, "does not shrink at s = pi")
    assert min(sphere["value"]["eps_max_near_pi"].values()) == pytest.approx(math.sqrt(24 * gjl.TAU), rel=1e-3)
    assert _labelled(report, "Sphere lateral+heading")["value"]["C2_at_s0"] == pytest.approx(0.5, abs=1e-2)
    assert _labelled(report, "real machines")["evidence_status"] == "not_established"
    assert _labelled(report, "real machines")["domain"] == "machine_safety"


def test_validity_bound_and_step_search_helpers():
    # r = C2 eps^2 exactly: the fit returns C2 and the bound is tau |j| / C2.
    c2, c3 = gjl.remainder_fit(gjl.EPS, [0.7 * e ** 2 for e in gjl.EPS])
    assert c2 == pytest.approx(0.7, rel=1e-12) and abs(c3) < 1e-9
    assert gjl.validity_bound(0.7, 0.0, 0.5) == pytest.approx(gjl.TAU * 0.5 / 0.7, rel=1e-12)
    assert gjl.validity_bound(0.0, -1.0 / 24, 1.0) == pytest.approx(math.sqrt(24 * gjl.TAU), rel=1e-12)
    assert gjl.validity_bound(1.0, 0.0, 0.0) == 0.0
    assert core.minimal_steps(lambda n: 1.0 / n, 0.01) == 100
    # Implicit midpoint is singular at kh = 2 and refused there.
    with pytest.raises(FloatingPointError, match="singular"):
        core.step_matrix("implicit-midpoint", -4.0, 1.0)
    assert np.allclose(core.constant_curvature_transfer("rk4", 1.0, 1.0, 64), [[math.cos(1), math.sin(1)],
                                                                                [-math.sin(1), math.cos(1)]], atol=1e-9)
    with pytest.raises(ValueError, match="positive finite"):
        core.loglog_slope([1.0, 2.0], [0.0, 1.0])
    closed = core.sphere_separation(Sphere(1.0), (math.pi / 2, 0.0), 1.0, [math.pi / 2], 0.02)
    assert closed["chord"][0] == pytest.approx(2 * math.sin(0.01), rel=1e-12)
    assert core.hyperbolic_distance(1.0, (0.0, 1.0), (0.0, math.e)) == pytest.approx(1.0, rel=1e-12)
    assert gjl.flat_deviation(1e-16, 2.0) == pytest.approx(-8e-16 / 6, rel=1e-12)
    assert gjl.flat_deviation(1.0, 2.0) == pytest.approx(math.sin(2.0) - 2.0, rel=1e-14)
    assert gjl.flat_deviation(-1e-6, 2.0) == pytest.approx(math.sinh(2e-3) / 1e-3 - 2.0, rel=1e-6)
    assert gjl.flat_deviation(0.0, 2.0) == 0.0 and Plane().gaussian_curvature(np.zeros(2)) == 0.0


def test_t018_resolvability_report(run):
    directory, reports = run
    report = reports["T018"]
    assert _labelled(report, "resolves every truncation-limited")["value"]["min_ratio_rk4_N16"] >= 10.0
    weak = _labelled(report, "Weak curvature is not harder")
    assert weak["value"]["ratio_K_1e-8_over_K_1e-2_at_N16"]["euler"] == pytest.approx(1.0, abs=0.05)
    assert weak["value"]["rk4_ratio_K_1e-4_over_K_1e-2_at_N16"] == pytest.approx(100.0, rel=0.5)
    floor = _labelled(report, "floating-point resolution")
    assert floor["value"] == {"max_ratio_K_1e-16": 1.0, "nonzero_computed_deviations_K_1e-16": 0}
    assert _labelled(report, "no spurious curvature")["value"] == 0.0
    sensor = _labelled(report, "measured sensor data")
    assert sensor["domain"] == "sensor_performance" and sensor["evidence_status"] == "not_established"
    rows = json.loads((directory / "artifacts" / "T018" / "resolvability.json").read_text(encoding="utf-8"))
    assert {r["surface"] for r in rows} >= {"plane", "sphere R=1e8", "torus R=2"}


def test_optional_scipy_cross_check_agrees_with_adaptive_references(run):
    pytest.importorskip("scipy")
    rows = json.loads((run[0] / "artifacts" / "T018" / "resolvability.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["scipy"] is not None]
    assert len(rows) == 4
    for row in rows:
        assert row["scipy"]["j_head"] == pytest.approx(row["reference_j_head"], abs=1e-9), row["surface"]
