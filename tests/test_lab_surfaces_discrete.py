"""Surface interface conformance, derivative checks, chart atlas and singularity scans (T033-T037)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ciw.lab import surfaces_discrete as sd
from ciw.lab.evidence import validate_finding
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.runner import Context, run_task
from ciw.lab.surfaces import Sphere, SurfaceRefusal, Torus
from ciw.lab.surfaces_discrete_ad import DualSurface, formulas
from ciw.lab.surfaces_discrete_charts import (SphereAtlas, great_circle, integrate_atlas, pole_passing_great_circle,
                                              require_regular, scan)
from ciw.lab.surfaces_discrete_geometry import (DOMAINS, SEED, THRESHOLDS, PowerGraph, SingularityRefusal, brioschi,
                                                conformance, conformance_surfaces, metric_second_derivatives,
                                                mutant_surfaces)

QUEUE = {item["id"]: item for item in load_queue()["tasks"]}


def _run(task_id, tmp_path, context=Context):
    return run_task(QUEUE[task_id], _REGISTRY[task_id], context(tmp_path), {})


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _common_report_checks(report):
    assert report["physical_validation_status"]["status"] == "not_established"
    for record in report["findings"]:
        validate_finding(record)
        if record["domain"] in ("physical", "industrial_readiness"):
            assert record["evidence_status"] == "not_established"
        elif record["value"] is not None:
            assert "regression_tolerance" in record
    for name in ("hypothesis", "mathematical_model", "experiment", "numerical_result", "uncertainty",
                 "recommended_next_task"):
        assert isinstance(report[name], str) and report[name]
    assert report["failure_modes_checked"] and report["unresolved_assumptions"]
    assert report["generated_artifacts"]


# ------------------------------------------------------------------ T033
def test_conformance_suite_accepts_every_surface():
    surfaces = conformance_surfaces()
    assert {"plane-polar", "gaussian-bump-shear", "rotated-torus"} <= set(surfaces)
    for key, surface in surfaces.items():
        result = conformance(surface, DOMAINS[key], 12, SEED)
        assert result["conforms"], (key, result["failed"])
        assert result["refused_by_core_check"] == 0
        assert result["worst"]["gauss_equation"] <= THRESHOLDS["gauss_equation"]


def test_conformance_suite_rejects_seeded_mutants():
    results = {key: conformance(surface, domain, 12, SEED) for key, (surface, domain) in mutant_surfaces().items()}
    assert not any(result["conforms"] for result in results.values())
    # A misscaled curvature is invisible to every identity except the Gauss equation.
    assert results["misscaled-curvature"]["failed"] == ["gauss_equation"]
    # Compatibility holds for any symmetric dg: only differences expose the sign flip.
    flipped = results["sign-flipped-derivatives"]["failed"]
    assert "derivative_consistency" in flipped and "compatibility" not in flipped
    assert "christoffel_asymmetry" in results["transposed-derivatives"]["failed"]
    assert results["indefinite-metric"]["failed"] == ["min_eigenvalue_ratio"]
    assert "gauss_equation" in results["indefinite-metric"]["not_evaluated"]
    assert results["nonsymmetric-metric"]["failed"] == ["metric_asymmetry"]


def test_brioschi_recovers_supplied_curvature():
    surfaces = conformance_surfaces()
    for key in ("torus", "gaussian-bump", "hyperbolic-plane", "plane-polar"):
        surface = surfaces[key]
        for u in DOMAINS[key].sample(4, SEED + 1):
            length = DOMAINS[key].length(u)
            d2g = metric_second_derivatives(surface, u, 1e-3 * length)
            k = brioschi(surface.metric(u), surface.metric_derivatives(u), d2g)
            assert k == pytest.approx(surface.gaussian_curvature(u), abs=1e-8 / length ** 2)
    # Closed-form second derivatives on the unit sphere: E = 1, F = 0, G = sin^2 theta.
    t = 0.8
    dg = np.zeros((2, 2, 2))
    dg[0, 1, 1] = math.sin(2 * t)
    d2g = np.zeros((2, 2, 2, 2))
    d2g[0, 0, 1, 1] = 2 * math.cos(2 * t)
    assert brioschi(np.diag([1.0, math.sin(t) ** 2]), dg, d2g) == pytest.approx(1.0, abs=1e-14)


def test_t033_report(tmp_path):
    report = _run("T033", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["numerically_verified"] == 9
    labels = _labels(report)
    assert labels["The conformance suite certifies surfaces reconstructed from physical measurements"] == "not_established"
    mutants = next(f for f in report["findings"] if f["claim"].startswith("The conformance suite rejects"))
    assert mutants["value"] == {"mutants": 6, "undetected": 0}
    assert mutants["counterexample"]["witness"]["failed"] == ["gauss_equation"]
    assert report["findings"][0]["value"] <= 1e-7
    _common_report_checks(report)


# ------------------------------------------------------------------ T034
def test_dual_numbers_match_closed_forms():
    cases = sd.dual_self_test()
    assert len(cases) == 8
    assert max(case["error"] for case in cases.values()) <= 1e-13
    assert cases["d/dx [x d/dy (x + y)]"]["dual"] == pytest.approx(1.0, abs=1e-15)


def test_dual_derivatives_match_surface_interface():
    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    for key in ("sphere", "torus", "hyperbolic-plane", "gaussian-bump-shear", "rotated-torus"):
        surface, dual = surfaces[key], DualSurface(*forms[key])
        for u in DOMAINS[key].sample(3, SEED + 2):
            np.testing.assert_allclose(dual.metric(u), surface.metric(u), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(dual.metric_derivatives(u), surface.metric_derivatives(u), rtol=1e-12,
                                       atol=1e-12 * float(np.max(np.abs(surface.metric(u)))))
            np.testing.assert_allclose(dual.christoffel(u), surface.christoffel(u), rtol=1e-12, atol=1e-12)
            assert dual.intrinsic_curvature(u) == pytest.approx(surface.gaussian_curvature(u), rel=1e-11, abs=1e-12)


def test_sympy_references_match_surface_interface():
    sp = pytest.importorskip("sympy")
    from ciw.lab.surfaces_discrete_ad import symbolic_exact_curvature, symbolic_reference

    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    for key in ("sphere", "torus", "hyperbolic-plane"):
        reference = symbolic_reference(*forms[key])
        for u in DOMAINS[key].sample(3, SEED + 3):
            values = reference(u)
            np.testing.assert_allclose(values["metric_derivatives"], surfaces[key].metric_derivatives(u), atol=1e-12)
            assert values["gaussian_curvature"] == pytest.approx(surfaces[key].gaussian_curvature(u), abs=1e-12)
    assert sp.simplify(symbolic_exact_curvature(*forms["sphere"]) - 1) == 0


def test_t034_report(tmp_path):
    pytest.importorskip("sympy")
    report = _run("T034", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["independently_verified"] == 3
    assert report["evidence_status"]["counts"]["numerically_verified"] == 4
    exact = next(f for f in report["findings"] if f["claim"].startswith("sympy simplifies"))
    assert exact["value"] == {"surfaces": 7, "mismatches": 0}
    assert exact["basis"]["independent_check"]["checker"]["implementation"] == "sympy"
    _common_report_checks(report)


class _NoSympy(Context):
    def available(self, requirement):
        return False if requirement == "module:sympy" else super().available(requirement)


def test_t034_degrades_without_sympy(tmp_path):
    report = _run("T034", tmp_path, _NoSympy)
    assert report["state"] == "partial"
    labels = _labels(report)
    assert labels["sympy symbolic metric, metric derivatives and Christoffel symbols match the ciw surface interface"] \
        == "not_established"
    assert report["evidence_status"]["counts"]["numerically_verified"] == 4
    assert report["evidence_status"]["counts"]["independently_verified"] == 0


# ------------------------------------------------------------------ T035
def test_finite_difference_error_is_v_shaped():
    study = sd.fd_study(points=4)
    for key in sd.FD_SURFACES:
        row = study["surfaces"][key]
        assert row["truncation_slope"] == pytest.approx(2.0, abs=0.02)
        assert row["rounding_slope"] == pytest.approx(-1.0, abs=0.3)
        assert 1e-7 <= row["h_opt"] <= 1e-4
        assert abs(math.log10(row["h_opt"] / row["h_opt_predicted"])) <= math.log10(4.0)
        assert row["error_at_1e-12"] > 1e3 * row["min_error"]
        assert row["leading_term_deviation"] <= 1e-3
    assert study["surfaces"]["saddle"]["error_at_1e-2"] <= 1e-13
    assert max(study["surfaces"]["plane"]["median_error"]) == 0.0


def test_t035_report(tmp_path):
    report = _run("T035", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    counter = [f for f in report["findings"] if f.get("counterexample")]
    assert len(counter) == 2
    assert _labels(report)["The optimal-step law derived here applies to derivatives of measured surface samples"] \
        == "not_established"
    _common_report_checks(report)


# ------------------------------------------------------------------ T036
def test_atlas_transitions_are_exact():
    study = sd.transition_study(count=48)
    assert study["used"] > 20
    assert study["roundtrip"] <= 1e-13
    assert study["metric_defect"] <= 1e-12
    assert study["jacobian_defect"] <= 1e-7
    assert study["covering_min"] >= 0.5 - 1e-12
    with pytest.raises(SurfaceRefusal, match="threshold"):
        SphereAtlas(threshold=0.6)


def _great_circle_run(delta, steps=200):
    atlas = SphereAtlas(1.0)
    start, tangent = pole_passing_great_circle(1.0, delta, azimuth=sd.AZIMUTH)
    u0 = atlas.to_chart("A", start)
    v0 = atlas.lift("A", u0, tangent)
    return atlas, start, tangent, u0, v0, integrate_atlas(atlas, "A", u0, v0, 2 * math.pi, steps)


def test_atlas_geodesic_through_pole_matches_great_circle():
    for delta in (0.0, 1e-3):
        atlas, start, tangent, _, _, run = _great_circle_run(delta)
        error = np.max(np.linalg.norm(run["points"] - great_circle(start, tangent, 1.0, run["s"]), axis=1))
        assert error <= 2e-6
        assert len(run["switches"]) == 4
        assert min(w["det_after"] for w in run["switches"]) >= 0.5
        assert run["min_active_det"] >= atlas.threshold


def test_single_chart_near_pole_counterexample():
    atlas, start, tangent, u0, v0, run = _great_circle_run(1e-3)
    single, failure = sd._single(atlas.charts["A"], u0, v0, 2 * math.pi, 200)
    assert single is None and failure in ("FloatingPointError", "ValueError", "OverflowError")
    atlas, start, tangent, u0, v0, run = _great_circle_run(0.1)
    single, failure = sd._single(atlas.charts["A"], u0, v0, 2 * math.pi, 200)
    atlas_error = np.max(np.linalg.norm(run["points"] - great_circle(start, tangent, 1.0, run["s"]), axis=1))
    single_error = np.max(np.linalg.norm(single["points"] - great_circle(start, tangent, 1.0, single["s"]), axis=1))
    assert failure is None and single_error > 10 * atlas_error


def test_t036_report(tmp_path):
    report = _run("T036", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    findings = {f["claim"]: f for f in report["findings"]}
    through = findings["Atlas integration of the great circle through the north pole matches the exact great circle"]
    assert through["value"] <= 1e-6
    single = findings["A single polar chart fails or loses accuracy on great circles passing near its pole"]
    assert single["value"]["failed"] >= 1 and single["value"]["degraded"] >= 5
    assert single["counterexample"]["statement"].startswith("Fixed-step RK4")
    assert findings["Chart-switching geodesic integration is ready for tool paths over physical parts"][
        "evidence_status"] == "not_established"
    _common_report_checks(report)


# ------------------------------------------------------------------ T037
def test_singularity_scans_classify_every_case():
    results = {a.name: scan(a) for a in sd.approaches()}
    assert {name: r["classification"] for name, r in results.items()} == sd.EXPECTED_CLASS
    pole = results["sphere-north-pole"]["exponents"]
    assert pole["det"] == pytest.approx(2.0, abs=1e-3) and pole["curvature"] == pytest.approx(0.0, abs=1e-3)
    assert results["power-graph-apex"]["exponents"]["curvature"] == pytest.approx(-1.0, abs=5e-3)
    assert results["cone-apex"]["circumference_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert results["plane-polar-origin"]["circumference_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_singularity_refusal_codes():
    cases = sd.refusal_cases()
    assert {name: observed for name, (_, observed) in cases.items()} == \
        {name: expected for name, (expected, _) in cases.items()}
    with pytest.raises(SingularityRefusal) as info:
        PowerGraph().metric(np.array([0.0, 0.0]))
    assert info.value.code == "curvature_singularity" and isinstance(info.value, SurfaceRefusal)
    # The core check is lenient where the guard is not.
    Sphere(1.0).check(np.array([3e-5, 0.3]))
    with pytest.raises(SingularityRefusal, match="condition number"):
        require_regular(Sphere(1.0), np.array([3e-5, 0.3]))
    assert require_regular(Torus(2.0, 1.0), np.array([0.1, 0.2]))["condition"] < 10


def test_t037_report(tmp_path):
    report = _run("T037", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    findings = {f["claim"]: f for f in report["findings"]}
    classes = findings["Every approach is classified as expected, with no false positive at regular points"]["value"]
    assert classes == sd.EXPECTED_CLASS
    refusals = findings["The pointwise guard refuses singular points with the expected codes and accepts a regular point"]
    assert all(check["reference_kind"] == "refusal" and check["passed"] for check in refusals["basis"]["checks"])
    assert sum(1 for f in report["findings"] if f.get("counterexample")) == 3
    _common_report_checks(report)
