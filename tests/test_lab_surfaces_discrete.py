"""Surface interface conformance, derivative checks, chart atlas and singularity scans (T033-T037)."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from ciw.lab import surfaces_discrete as sd
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS, validate_finding
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.runner import Context, _close, _witness, run_task
from ciw.lab.surfaces import SAMPLING_DOMAINS, GaussianBump, Sphere, SurfaceRefusal, Torus
from ciw.lab.surfaces_discrete_ad import DualSurface, formulas
from ciw.lab.surfaces_discrete_charts import (Atlas, SphereAtlas, graph_atlas, great_circle, integrate_atlas,
                                              pole_passing_great_circle, refusal_code, require_regular, scan)
from ciw.lab.surfaces_discrete_geometry import (DOMAINS, SEED, THRESHOLDS, PowerGraph, brioschi, conformance,
                                                conformance_surfaces, metric_second_derivatives, mutant_surfaces)

QUEUE = {item["id"]: item for item in load_queue()["tasks"]}


def _run(task_id, tmp_path, context=Context):
    return run_task(QUEUE[task_id], _REGISTRY[task_id], context(tmp_path), {})


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _numeric_leaves(value, path=()):
    """Paths of the numeric (non-boolean) leaves of a JSON value."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [path]
    items = value.items() if isinstance(value, dict) else enumerate(value)
    return [leaf for key, item in items for leaf in _numeric_leaves(item, path + (key,))]


def _set_leaf(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


def _witnesses_guarded(record):
    """Every numeric witness value is bounded by the finding's regression tolerance (the gate compares with it)."""
    counter = record.get("counterexample")
    if not counter:
        return
    tolerance = record["regression_tolerance"]
    for path in _numeric_leaves(counter["witness"]):
        tampered = deepcopy(counter)
        _set_leaf(tampered["witness"], path, 0.5)
        assert not _close(_witness(counter), _witness(tampered), tolerance), (record["claim"], path)


def _common_report_checks(report):
    assert report["physical_validation_status"]["status"] == "not_established"
    assert "src/ciw/lab/surfaces.py" in report["provider_runtime_identity"]["sources"]
    for record in report["findings"]:
        validate_finding(record)
        if record["domain"] in ("physical", "industrial_readiness"):
            assert record["evidence_status"] == "not_established"
        elif record["value"] is not None:
            assert "regression_tolerance" in record
            # AUTHORING rule 5: every numerical finding states its uncertainty.
            assert record["domain"] in COMPUTATIONAL_DOMAINS
            uncertainty = record["uncertainty"]
            assert isinstance(uncertainty, dict) and set(uncertainty) == {"kind", "value", "basis"}, record["claim"]
            assert isinstance(uncertainty["value"], (int, float)) and uncertainty["basis"]
            _witnesses_guarded(record)
    for name in ("hypothesis", "mathematical_model", "experiment", "numerical_result", "uncertainty",
                 "recommended_next_task"):
        assert isinstance(report[name], str) and report[name]
    assert report["failure_modes_checked"] and report["unresolved_assumptions"]
    assert report["generated_artifacts"]


# ------------------------------------------------------------------ T033
def test_conformance_suite_accepts_every_surface():
    surfaces = conformance_surfaces()
    assert {"plane-polar", "gaussian-bump-shear", "rotated-torus"} <= set(surfaces)
    # Catalogue sampling boxes come from the core declaration, not a restatement.
    for key, ((low1, high1), (low2, high2)) in SAMPLING_DOMAINS.items():
        assert (DOMAINS[key].low, DOMAINS[key].high) == ((low1, low2), (high1, high2))
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
    # NaN derivatives on half of the domain are failures, not silently dropped maxima.
    nan = results["nan-derivatives"]
    assert not nan["conforms"] and "derivative_asymmetry" in nan["nonfinite"]
    assert set(nan["nonfinite"]) <= set(nan["failed"])
    assert nan["worst"]["derivative_consistency"] == "nonfinite"


class _NaNAtFirstPoint(Sphere):
    """Curvature NaN at the first sampled point only (a running max would keep or drop it by position)."""

    first_sample = tuple(DOMAINS["sphere"].sample(12, SEED)[0])

    def gaussian_curvature(self, u):
        return math.nan if tuple(u) == self.first_sample else super().gaussian_curvature(u)


def test_conformance_reports_nonfinite_residuals():
    result = conformance(_NaNAtFirstPoint(1.0), DOMAINS["sphere"], 12, SEED)
    assert result["failed"] == ["gauss_equation"] and result["nonfinite"] == ["gauss_equation"]
    assert result["worst"]["gauss_equation"] == "nonfinite" and not result["conforms"]


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
    assert report["evidence_status"]["counts"]["numerically_verified"] == 10
    labels = _labels(report)
    assert labels["The conformance suite certifies surfaces reconstructed from physical measurements"] == "not_established"
    findings = {f["claim"]: f for f in report["findings"]}
    assert findings["The conformance suite rejects every seeded defect mutant"]["value"] == {"mutants": 7, "undetected": 0}
    misscaled = findings["A curvature-misscaled sphere passes every identity except the Gauss equation"]
    assert misscaled["evidence_status"] == "numerically_verified"
    assert misscaled["value"]["failed"] == ["gauss_equation"] and misscaled["counterexample"]["witness"]["failed"] == [
        "gauss_equation"]
    assert all(check["passed"] for check in misscaled["basis"]["checks"])
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
    from ciw.lab.surfaces_discrete_ad import diffgeom_reference, symbolic_reference

    surfaces = conformance_surfaces()
    forms = formulas(surfaces)
    for key in ("sphere", "hyperbolic-plane"):
        reference = symbolic_reference(*forms[key])
        geometric, expression, (u_sym, v_sym) = diffgeom_reference(*forms[key])
        for u in DOMAINS[key].sample(3, SEED + 3):
            values = reference(u)
            np.testing.assert_allclose(values["metric_derivatives"], surfaces[key].metric_derivatives(u), atol=1e-12)
            assert values["gaussian_curvature"] == pytest.approx(surfaces[key].gaussian_curvature(u), abs=1e-12)
            np.testing.assert_allclose(geometric(u)["christoffel"], surfaces[key].christoffel(u), atol=1e-12)
            assert geometric(u)["gaussian_curvature"] == pytest.approx(surfaces[key].gaussian_curvature(u), abs=1e-12)
        declared = {"sphere": 1, "hyperbolic-plane": -1}[key]
        assert sp.simplify(expression - declared) == 0
    # Parameters enter the symbolic formulas exactly: no float survives in the curvature.
    _, expression, _ = diffgeom_reference(*forms["sphere"])
    assert not sp.simplify(expression).atoms(sp.Float)


def test_t034_report(tmp_path):
    pytest.importorskip("sympy")
    report = _run("T034", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["independently_verified"] == 4
    assert report["evidence_status"]["counts"]["numerically_verified"] == 5
    labels = _labels(report)
    # The independent evidence is per finding: every sympy-origin claim is independently verified,
    # while ciw assembly of sympy derivatives and the dual numbers stay same-origin.
    for claim, label in labels.items():
        if claim.startswith("sympy"):
            assert label == "independently_verified", claim
    assert labels["Christoffel symbols and curvature assembled in ciw code from sympy derivatives match the interface "
                  "on every conformance surface"] == "numerically_verified"
    assert labels["Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from "
                  "physical measurements"] == "not_established"
    findings = {f["claim"]: f for f in report["findings"]}
    exact = findings["sympy.diffgeom curvature of seven surfaces simplifies exactly to the closed forms restated from "
                     "ciw.lab.surfaces"]
    assert exact["value"] == {"surfaces": 7, "mismatches": 0}
    assert exact["basis"]["independent_check"]["checker"]["implementation"] == "sympy.diffgeom"
    assert exact["basis"]["independent_check"]["producer"]["implementation"].endswith("_declared_curvature")
    derivs = findings["sympy-differentiated metric and metric derivatives match the ciw surface interface on every "
                      "conformance surface"]
    assert "surfaces_discrete_geometry.py" in derivs["basis"]["independent_check"]["producer"]["revision"]
    # ciw dual numbers against ciw surfaces are same-origin comparisons; only the
    # self-test against hand-derived closed forms is analytic.
    for claim, record in findings.items():
        if claim.startswith(("Nested dual-number", "Dual-number curvature", "Dual-number checks expose")):
            assert {c["reference_kind"] for c in record["basis"]["checks"]} == {"cross_implementation"}, claim
    self_test = findings["Dual numbers reproduce closed-form first, mixed and third derivatives without perturbation "
                         "confusion"]
    assert {c["reference_kind"] for c in self_test["basis"]["checks"]} == {"analytic"}
    _common_report_checks(report)


class _NoSympy(Context):
    def available(self, requirement):
        return False if requirement == "module:sympy" else super().available(requirement)


def test_t034_degrades_without_sympy(tmp_path):
    report = _run("T034", tmp_path, _NoSympy)
    assert report["state"] == "partial"
    labels = _labels(report)
    sympy_claims = [claim for claim in labels if claim.startswith("sympy")]
    assert len(sympy_claims) == 4 and all(labels[claim] == "not_established" for claim in sympy_claims)
    assert report["evidence_status"]["primary"] == "numerically_verified"
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
        # Signed leading-term comparison only where truncation dominates rounding by 1e4.
        assert row["leading_term_points"] >= 3 and row["leading_term_deviation"] <= 1e-3
        assert row["pointwise_min_ratio"] <= 2.0 and row["pointwise_min_error"] <= 2e-10
    assert study["surfaces"]["saddle"]["error_at_1e-2"] <= 1e-13
    assert max(study["surfaces"]["plane"]["median_error"]) == 0.0


def test_t035_report(tmp_path):
    report = _run("T035", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    counter = [f for f in report["findings"] if f.get("counterexample")]
    assert len(counter) == 2
    smaller = {f["claim"]: f for f in counter}["Smaller finite-difference steps can be far less accurate"]
    assert set(smaller["counterexample"]["witness"]) == {"surface", "log10_h_opt", "log10_error_at_h_opt",
                                                         "log10_error_at_1e-12"}
    assert _labels(report)["The optimal-step law derived here applies to derivatives of measured surface samples"] \
        == "not_established"
    _common_report_checks(report)


# ------------------------------------------------------------------ T036
def test_atlas_transitions_are_exact():
    study = sd.transition_study(count=48, dense=256)
    assert study["used"] > 20
    assert study["roundtrip"] <= 1e-13
    assert study["metric_defect"] <= 1e-12
    assert study["jacobian_defect"] <= 1e-7
    assert study["covering_min"] >= 0.5 - 1e-12
    with pytest.raises(SurfaceRefusal, match="threshold"):
        SphereAtlas(threshold=0.6)
    with pytest.raises(SurfaceRefusal, match="two charts"):
        Atlas({"only": Sphere(1.0)}, {"only": lambda x: x[:2]})
    # The scale-free regularity (1/cond g) equals det g / R^4 = sin^2 theta on the sphere, for any radius.
    atlas = SphereAtlas(3.0)
    for theta in (0.2, 0.9, 1.5):
        u = np.array([theta, 0.4])
        assert atlas.regularity("A", u) == pytest.approx(math.sin(theta) ** 2, rel=1e-13)


def test_covering_check_goes_through_the_atlas(monkeypatch):
    """A wrong chart inverse must make the covering bound fail: the dense points are not a closed form."""
    healthy = sd.transition_study(count=16, dense=512)["covering_min"]
    assert healthy >= 0.5 - 1e-12
    broken = staticmethod(lambda rotation, point: np.array([0.05, 0.0]))  # every point mapped next to a pole
    monkeypatch.setattr(SphereAtlas, "_polar_inverse", broken)
    assert sd.transition_study(count=16, dense=512)["covering_min"] < 0.01


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
    assert single is None and failure.startswith(("FloatingPointError: rk4 produced a nonfinite state",
                                                  "ValueError: math domain error"))
    # A programming error is not a single-chart failure: it propagates.
    with pytest.raises(ValueError, match="steps must be positive"):
        sd._single(atlas.charts["A"], u0, v0, 2 * math.pi, 0)
    atlas, start, tangent, u0, v0, run = _great_circle_run(0.1)
    single, failure = sd._single(atlas.charts["A"], u0, v0, 2 * math.pi, 200)
    atlas_error = np.max(np.linalg.norm(run["points"] - great_circle(start, tangent, 1.0, run["s"]), axis=1))
    single_error = np.max(np.linalg.norm(single["points"] - great_circle(start, tangent, 1.0, single["s"]), axis=1))
    assert failure is None and single_error > 10 * atlas_error


def test_meridian_depends_on_the_step_grid():
    study = sd.meridian_study(step_counts=(355, 399, 400))
    rows = {row["steps"]: row for row in study["rows"]}
    # v_phi starts at exactly 0 but rounding in g_12 seeds it.
    assert study["drift"]["v_phi_at_start"] == 0.0 and study["drift"]["max_abs_v_phi"] > 0.0
    assert rows[400]["grid_distance"] > 5e-3 and rows[400]["error"] <= 1e-10
    assert rows[399]["grid_distance"] < sd.NEAR_POLE and rows[399]["error"] > 1e-8
    assert rows[355]["grid_distance"] < 1e-6 and rows[355]["error"] is None
    assert rows[355]["failure"].startswith("FloatingPointError: rk4 produced a nonfinite state")


def test_graph_atlas_transitions_and_apex_geodesics():
    bump = GaussianBump(0.5, 1.0)
    atlas = graph_atlas(bump)
    assert atlas.regularity("polar", np.array([1e-3, 0.4])) < 1e-5 < atlas.regularity("monge", np.array([0.0, 0.0]))
    transitions = sd.graph_transition_study(count=48)
    assert transitions["used"] > 40 and transitions["roundtrip"] <= 1e-13 and transitions["metric_defect"] <= 1e-12
    assert transitions["jacobian_defect"] <= 1e-7
    assert transitions["covering_min"] >= transitions["monge_regularity_bound"] - 1e-12
    study = sd.graph_study(deltas=(0.0, 1e-3))
    for row in study["runs"]:
        assert row["switch_path"] == ["polar->monge"] and row["atlas_error"] <= 1e-8
    assert study["runs"][1]["polar_failure"] is not None or study["runs"][1]["polar_error"] > 10 * study["runs"][1][
        "atlas_error"]


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
    meridian = findings["At 400 RK4 steps chart A alone crosses both poles on the exact meridian to within 1e-10"]
    assert meridian["evidence_status"] == "numerically_verified" and meridian["value"]["log10_error"] <= -10
    grid = findings["On the exact meridian chart A alone fails or loses accuracy whenever a step point lands within "
                    "1e-4 of a pole (350 to 450 RK4 steps)"]
    assert grid["evidence_status"] == "numerically_verified"
    assert 399 in grid["counterexample"]["witness"]["steps_near_pole"]
    assert grid["counterexample"]["witness"]["failed_steps"] == [355]
    graph = findings["The graph atlas integrates geodesics through and near the Gaussian-bump apex, where its polar "
                     "chart alone fails or loses accuracy"]
    assert graph["evidence_status"] == "numerically_verified" and graph["value"]["min_switches"] >= 1
    assert report["evidence_status"]["counts"]["numerically_verified"] == 10
    assert "src/ciw/lab/integrators.py" in report["provider_runtime_identity"]["sources"]
    _common_report_checks(report)


# ------------------------------------------------------------------ T037
def test_singularity_scans_classify_every_case():
    results = {a.name: scan(a) for a in sd.approaches()}
    # Every declared approach is read as its true type except the removable cube-root chart blow-up (rule 4).
    assert {name: r["classification"] for name, r in results.items()} == dict(
        sd.TRUE_CLASS, **{sd.RULE4_MISS: "unclassified"})
    assert sd.TRUE_CLASS[sd.RULE4_MISS] == "coordinate_singularity"
    pole = results["sphere-north-pole"]["exponents"]
    assert pole["det"] == pytest.approx(2.0, abs=1e-3) and pole["curvature"] == pytest.approx(0.0, abs=1e-3)
    assert results["power-graph-apex"]["exponents"]["curvature"] == pytest.approx(-1.0, abs=1e-3)
    # The slow blow-up K ~ r^-0.4 and the finite-distance conformal boundary are curvature singularities.
    assert results["power-graph-1.8-apex"]["exponents"]["curvature"] == pytest.approx(-0.4, abs=1e-3)
    assert results["conformal-0.9-boundary"]["exponents"]["radial_speed"] == pytest.approx(-0.9, abs=1e-6)
    assert results["hyperbolic-boundary"]["exponents"]["radial_speed"] == pytest.approx(-1.0, abs=1e-6)
    assert results["cone-apex"]["circumference_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert results["plane-polar-origin"]["circumference_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_singularity_detection_limits():
    for name, (approach, truth) in sd.limit_approaches().items():
        observed = scan(approach)["classification"]
        assert observed != truth, name
    limits = {name: scan(approach)["classification"] for name, (approach, _) in sd.limit_approaches().items()}
    assert limits == {"power-graph-1.99-apex": "regular", "conformal-0.9995-boundary": "infinite_distance_boundary",
                      "cone-small-deficit-apex": "coordinate_singularity"}
    # The loops must be geodesic-circle preimages: Cartesian loops at the sphere pole read as a cone.
    cartesian = scan(sd.Approach("pole-cartesian", Sphere(1.0), (0.0, 0.3), False))
    assert cartesian["classification"] == "conical_singularity"
    assert cartesian["circumference_ratio"] == pytest.approx(0.8317, abs=1e-3)


def test_singularity_refusal_codes():
    for cases in (sd.refusal_cases(), sd.propagated_refusal_cases()):
        assert {name: observed for name, (_, observed) in cases.items()} == \
            {name: expected for name, (expected, _) in cases.items()}
    # Surface-declared apex codes are recorded as author labels, not checked as detections.
    assert sd.declared_refusal_codes() == {"cone curvature at r = 0": "conical_singularity",
                                           "power graph metric at rho = 0": "curvature_singularity"}
    with pytest.raises(SurfaceRefusal) as info:
        PowerGraph().metric(np.array([0.0, 0.0]))
    assert info.value.code == "curvature_singularity"
    # Core refusals keep their own codes through the guard.
    assert refusal_code(Sphere(1.0).check, np.array([1e-7, 0.3])) == "degenerate_metric"
    assert refusal_code(require_regular, sd.HyperbolicPlane(), np.array([0.0, -1.0])) == "outside_chart"
    # The core check is lenient where the guard is not.
    Sphere(1.0).check(np.array([3e-5, 0.3]))
    with pytest.raises(SurfaceRefusal, match="condition number") as info:
        require_regular(Sphere(1.0), np.array([3e-5, 0.3]))
    assert info.value.code == "degenerate_metric"
    # The guard refuses only above its declared curvature bound, however fast K grows.
    assert refusal_code(require_regular, PowerGraph(1.0, 1.8), np.array([1e-8, 0.0])) == "accepted"
    assert require_regular(Torus(2.0, 1.0), np.array([0.1, 0.2]))["condition"] < 10


def test_t037_report(tmp_path):
    report = _run("T037", tmp_path)
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    findings = {f["claim"]: f for f in report["findings"]}
    classes = findings["The scan classifies 9 of the 10 declared approaches as their true type, with no false "
                       "positive at regular points"]["value"]
    assert classes["correct"] == 9 and classes["classes"][sd.RULE4_MISS] == "unclassified"
    missed = findings["The scan misses the removable coordinate singularity of the plane in the cube-root chart"]
    assert missed["evidence_status"] == "numerically_verified"
    assert missed["counterexample"]["witness"] == {"approach": sd.RULE4_MISS, "true": "coordinate_singularity",
                                                   "observed": "unclassified"}
    computed = findings["The pointwise guard and the core check refuse degenerate, blown-up, nonfinite and "
                        "out-of-chart points with computed codes, and accept curvature below the declared bound"]
    propagated = findings["The pointwise guard propagates a refusal code that a surface raises at its own apex "
                          "unchanged"]
    for record, count in ((computed, 8), (propagated, 1)):
        assert len(record["basis"]["checks"]) == count
        assert all(check["reference_kind"] == "refusal" and check["passed"] for check in record["basis"]["checks"])
    assert "curvature_singularity" not in computed["value"].values()
    limits = findings["Cases just beyond each classification threshold are misclassified"]
    assert limits["evidence_status"] == "numerically_verified"
    assert all(row["observed"] != row["true"] for row in limits["value"]["cases"].values())
    assert sum(1 for f in report["findings"] if f.get("counterexample")) == 6
    assert report["evidence_status"]["counts"]["numerically_verified"] == 13
    assert "max(1, tr g)" not in " ".join(report["unresolved_assumptions"])
    _common_report_checks(report)
