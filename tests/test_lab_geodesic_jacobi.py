"""Regression tests for the geodesic/Jacobi lab tasks T001-T009.

All tasks share one module-scoped context so memoized integrations run once.
Optional modules are skipped with importorskip; the numpy-only fallbacks are
exercised by hiding them from the context. The pinned provider comparison runs
only when CIW_LAB_CSG_REPO names a checkout.
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ciw.lab import geodesic_jacobi as gjt
from ciw.lab import geodesic_jacobi_common as gj
from ciw.lab import integrators, jacobi, runner, surfaces, svg
from ciw.lab.registry import load_queue, module_implementations
from ciw.lab.report import validate_report

SECTION = tuple(f"T00{i}" for i in range(1, 10))
# Only this module is imported: the other geodesic/Jacobi module belongs to another task set.
IMPLEMENTATIONS = module_implementations("geodesic_jacobi")
OPTIONAL = frozenset({"module:sympy", "module:mpmath", "module:scipy"})
HAND_CLAIM = ("The hand-derived metrics, Christoffel symbols, geodesic equations and curvatures of the section doc "
              "match ciw.lab.surfaces on nine charts")
EGREGIUM_CLAIM = ("The second-fundamental-form curvature (LN - M^2)/det g equals the intrinsic curvature on the six "
                  "embedded charts (Theorema Egregium)")


class BareContext(runner.Context):
    """A context that reports the optional modules as missing (the numpy-only CI job)."""

    def available(self, requirement):
        return False if requirement in OPTIONAL else super().available(requirement)


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _run(ctx, task_id):
    return validate_report(runner.run_task(_queue()[task_id], IMPLEMENTATIONS[task_id], ctx, {}))


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    ctx = runner.Context(tmp_path_factory.mktemp("geodesic-jacobi"))
    reports = {}

    def run(task_id):
        if task_id not in reports:
            reports[task_id] = _run(ctx, task_id)
        return reports[task_id]

    run.ctx = ctx
    return run


def _findings(report):
    return {f["claim"]: f for f in report["findings"]}


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _artifact(ctx, task_id, name):
    return json.loads((ctx.output_dir / "artifacts" / task_id / name).read_text(encoding="utf-8"))


def test_registrations_name_existing_tests():
    names = set(globals())
    assert set(IMPLEMENTATIONS) == set(SECTION)
    for task_id in SECTION:
        implementation = IMPLEMENTATIONS[task_id]
        assert implementation.regression_tests, task_id
        for node in implementation.regression_tests:
            path, _, name = node.partition("::")
            assert path == "tests/test_lab_geodesic_jacobi.py" and name in names, node
        assert "src/ciw/lab/geodesic_jacobi.py" in implementation.changed_files


def test_t001_symbolic_derivations_match_surfaces(lab):
    pytest.importorskip("sympy")
    report = lab("T001")
    assert report["state"] == "completed"
    # The primary label is the weakest established one; the sympy findings are independent, the rest checked.
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["independently_verified"] == 3
    assert report["evidence_status"]["counts"]["analytic"] == 0
    labels = _labels(report)
    geodesic = _findings(report)["Sympy-derived metrics, Christoffel symbols and geodesic equations match "
                                 "ciw.lab.surfaces on nine charts"]
    assert geodesic["basis"]["independent_check"]["checker"]["implementation"] == "sympy"
    assert max(geodesic["value"].values()) < 1e-12
    assert labels["Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts"] \
        == "independently_verified"
    counter = [f for f in report["findings"] if f.get("counterexample")]
    assert len(counter) == 2 and all(f["evidence_status"] == "numerically_verified" for f in counter)
    hand = _findings(report)[HAND_CLAIM]
    assert hand["evidence_status"] == "numerically_verified" and hand["basis"]["derivation"].startswith("docs/lab/")
    assert max(max(v.values()) for v in hand["value"].values()) < 1e-12
    egregium = _findings(report)[EGREGIUM_CLAIM]
    assert egregium["evidence_status"] == "independently_verified"
    assert set(egregium["value"]) == set(gjt.EMBEDDED_CHARTS)
    text = (lab.ctx.output_dir / "artifacts" / "T001" / "derivations.txt").read_text(encoding="utf-8")
    # Printed forms vary between sympy versions; the retained text must name every derived object.
    assert "[torus]" in text and "Gamma^u_uv" in text and "Gamma^v_uu" in text and "K (Brioschi)" in text
    tex = (lab.ctx.output_dir / "artifacts" / "T001" / "derivations.tex").read_text(encoding="utf-8")
    assert tex.startswith("\\documentclass") and tex.rstrip().endswith("\\end{document}")
    body = [line for line in tex.splitlines() if line.startswith(("g =", "K =", "\\Gamma", "\\ddot"))]
    assert not body and tex.count("\\[") == tex.count("\\]") > 9
    table = _artifact(lab.ctx, "T001", "comparison.json")
    assert table["discrepancies"]["plane-polar"]["symbolic_curvature_is_zero"] is True


def test_t001_without_sympy_is_partial(tmp_path):
    report = _run(BareContext(tmp_path), "T001")
    assert report["state"] == "partial"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["independently_verified"] == 0
    assert set(_labels(report).values()) == {"numerically_verified"}
    assert _findings(report)[EGREGIUM_CLAIM]["evidence_status"] == "numerically_verified"
    assert any("sympy is not installed" in item for item in report["unresolved_assumptions"])
    polar = _findings(report)["Nonzero Christoffel symbols do not imply curvature: the polar charts of the plane "
                              "and cylinder are flat"]
    assert polar["evidence_status"] == "numerically_verified"
    assert all(p["max_abs_intrinsic_curvature"] < 1e-6 for p in polar["value"].values())


def test_t001_self_consistency_helpers():
    sphere, hyperbolic = gj.surface("sphere"), gj.surface("hyperbolic-plane")
    assert gjt.intrinsic_curvature(sphere, np.array([1.1, 0.3])) == pytest.approx(1.0, abs=1e-7)
    assert gjt.intrinsic_curvature(hyperbolic, np.array([0.2, 1.3])) == pytest.approx(-1.0, abs=1e-6)
    assert gjt.compatibility_residual(gj.surface("torus"), np.array([0.4, 2.0])) < 1e-13

    class Transposed(surfaces.Sphere):
        """Christoffel symbols with the upper and first lower index swapped (a wrong convention)."""

        def christoffel(self, u):
            return super().christoffel(u).transpose(1, 0, 2)

    assert gjt.compatibility_residual(Transposed(), np.array([1.1, 0.3])) > 0.1
    # The hand table and the generic second-fundamental-form route agree with the catalogue.
    u, v = np.array([0.4, 2.0]), np.array([0.3, -0.7])
    torus = gj.surface("torus")
    hand = gjt.hand_geometry("torus", u, v)
    assert np.allclose(hand["christoffel"], torus.christoffel(u), atol=1e-14)
    assert np.allclose(hand["acceleration"], torus.geodesic_rhs(np.concatenate([u, v]))[2:], atol=1e-14)
    assert gjt.extrinsic_curvature(torus, u) == pytest.approx(torus.gaussian_curvature(u), abs=1e-14)
    assert gjt.SAMPLE_BOXES["torus"] == surfaces.SAMPLING_DOMAINS["torus"]
    points, velocities = gjt.sample_points("torus")
    again, _ = gjt.sample_points("torus")
    assert points.shape == (gjt.SAMPLES_PER_CHART, 2) and np.array_equal(points, again)


def test_t002_references_agree(lab):
    for name in ("sympy", "mpmath", "scipy"):
        pytest.importorskip(name)
    report = lab("T002")
    assert report["state"] == "completed"
    labels = _labels(report)
    for key in gj.VARIABLE_KEYS:
        claim = f"ciw Richardson RK4 matches the 34-digit mpmath reference on the {key} path"
        assert labels[claim] == "independently_verified"
        record = _findings(report)[claim]
        assert record["basis"]["independent_check"]["checker"]["implementation"] == "mpmath"
        assert record["basis"]["independent_check"]["tolerance"] == 1e-12
        assert record["value"]["ciw_minus_reference"] < 1e-12
        # Extrapolation, not only a fine step, produced the agreement: the gap is far below the RK4 estimate.
        ratio = [c for c in record["basis"]["checks"] if c["reference"].startswith("gap over")][0]
        assert ratio["passed"] and ratio["observed"] < 0.1
        assert record["uncertainty"]["kind"] == "reference_error" and record["uncertainty"]["value"] < 1e-18
    rows = _artifact(lab.ctx, "T002", "references.json")["rows"]
    assert all(rows[k]["reference_error_estimate"] < 1e-18 for k in gj.VARIABLE_KEYS)
    assert all(rows[k]["scipy_vs_reference"]["max"] < 1e-11 for k in rows)
    closed = _findings(report)["ciw Richardson RK4 end states match closed-form geodesics and transfer matrices on "
                               "the six closed-form charts"]
    assert closed["evidence_status"] == "independently_verified" and len(closed["value"]) == 6
    assert labels["Clairaut's integral rho^2 phi' is conserved along the torus reference path"] == "numerically_verified"


def test_t002_without_optional_modules(tmp_path):
    report = _run(BareContext(tmp_path), "T002")
    assert report["state"] == "partial"
    assert set(_labels(report).values()) == {"numerically_verified"}
    assert "ciw Richardson RK4 is self-convergent on the torus path (no independent reference available)" \
        in _labels(report)
    assert any("Unavailable optional modules" in item for item in report["unresolved_assumptions"])


def test_adaptive_checks_reject_fourth_order_variant(lab):
    lab("T003")
    real = gjt.adaptive_summary(lab.ctx, "dp54", integrators.integrate_adaptive)
    variant = gjt.adaptive_summary(lab.ctx, "dp54-y4", gjt.dormand_prince_y4)
    assert real["median_effective_order"] > gjt.ADAPTIVE_ORDER_THRESHOLD > variant["median_effective_order"]
    assert real["median_tolerance_exponent"] > gjt.ADAPTIVE_EXPONENT_THRESHOLD > variant["median_tolerance_exponent"]
    # The variant is the same pair: on y' = y at one tolerance it only loses local extrapolation.
    y0 = np.array([1.0])
    _, fifth, _ = integrators.integrate_adaptive(lambda y: y, y0, 1.0, rtol=1e-8, atol=1e-8)
    _, fourth, _ = gjt.dormand_prince_y4(lambda y: y, y0, 1.0, rtol=1e-8, atol=1e-8)
    assert abs(fourth[-1, 0] - math.e) > abs(fifth[-1, 0] - math.e)


def test_t003_integrator_orders(lab):
    report = lab("T003")
    assert report["state"] == "completed"
    assert set(_labels(report).values()) == {"numerically_verified"}
    found = _findings(report)
    for name, p, tol in (("Explicit Euler", 1, 0.1), ("Explicit midpoint", 2, 0.1), ("Classical RK4", 4, 0.25)):
        orders = found[f"{name} global endpoint error converges at order {p} on every curved chart"]["value"]
        assert set(orders) == set(gjt.CURVED_CHARTS)
        assert all(abs(v - p) <= tol for v in orders.values())
    effective = found["Adaptive Dormand-Prince error falls with function evaluations at a median effective order "
                      "near 5"]
    assert gjt.ADAPTIVE_ORDER_THRESHOLD < effective["value"] < 6
    exponent = found["Adaptive Dormand-Prince endpoint error is proportional to the requested tolerance (median over "
                     "charts)"]
    assert gjt.ADAPTIVE_EXPONENT_THRESHOLD < exponent["value"] < 1.1
    variant = found["The adaptive-order checks reject a Dormand-Prince variant that advances with its fourth-order "
                    "solution"]
    assert variant["value"]["median_effective_order"] < gjt.ADAPTIVE_ORDER_THRESHOLD
    assert variant["value"]["median_tolerance_exponent"] < gjt.ADAPTIVE_EXPONENT_THRESHOLD
    assert variant["counterexample"]
    flat = found["No convergence order is observable on flat Cartesian charts: every method is exact to rounding "
                 "there"]
    assert flat["counterexample"]["witness"]["charts"] == ["plane", "cylinder"]
    assert max(flat["value"].values()) < 1e-12


def test_t004_speed_drift_and_no_renormalization(lab):
    report = lab("T004")
    assert report["state"] == "completed"
    found = _findings(report)
    orders = found["Unit-speed drift max|g(v,v) - 1| scales like h^p for Euler, midpoint and RK4 on every curved "
                   "chart"]["value"]
    assert all(abs(v - 1) < 0.1 for v in orders["euler"].values())
    assert all(abs(v - 4) < 0.3 for v in orders["rk4"].values())
    nonunit = found["A non-unit initial speed stays non-unit: g(v,v) remains 1.69 to integrator accuracy"]
    assert nonunit["evidence_status"] == "numerically_verified"
    assert max(v for row in nonunit["value"].values() for v in row.values()) < 1e-7
    counter = found["Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding "
                    "with a first-order endpoint error"]
    assert counter["value"]["renormalized_max_speed_drift"] < 1e-13
    assert abs(counter["value"]["renormalized_error_order"] - 1) < 0.1
    assert counter["value"]["renormalized_endpoint_error"] > 1e-3 and counter["counterexample"]
    assert set(_labels(report).values()) == {"numerically_verified"}
    scan = found["The integrator and geodesic/Jacobi right-hand-side code paths contain no state normalization"]
    assert scan["value"]["normalization_calls"] == 0 and scan["domain"] == "computational_pipeline"


def test_integrator_code_path_has_no_normalization():
    assert gjt.normalization_calls() == []

    def renormalizing_step(f, y, h):
        y = y + h * f(y)
        return y / np.linalg.norm(y)

    def rescaled(y):
        y[2:] /= np.sqrt(y[2:] @ y[2:])
        return y

    def through_a_name(y):
        s = np.sqrt(y[2:] @ y[2:])
        y[2:] = y[2:] / s
        return y

    def half_power(y):
        y[2:] = y[2:] * (y[2:] @ y[2:]) ** -0.5
        return y

    def reciprocal(y):
        y[2:] *= 1.0 / math.sqrt(y[2:] @ y[2:])
        return y

    def harmless(y):
        return y ** 2 / (1.0 + y)

    for probe in (renormalizing_step, rescaled, through_a_name, half_power, reciprocal, gjt.renormalized_euler):
        assert gjt.normalization_calls([probe]), probe.__name__
    assert gjt.normalization_calls([harmless]) == []
    # A speed-1.3 start keeps its speed under every method, which a renormalizing integrator could not do.
    sphere = gj.surface("sphere")
    y0 = gj.start_state("sphere")[:4].copy()
    y0[2:] *= 1.3
    for method, tolerance in (("rk4", 1e-7), ("adaptive", 1e-9), ("midpoint", 1e-3), ("euler", 0.1)):
        g = gjt.speed_probe(sphere, y0, 1.0, method)
        assert np.max(np.abs(g - 1.69)) < tolerance and np.min(np.abs(g - 1.0)) > 0.6, method


def test_t005_separation_law(lab):
    report = lab("T005")
    assert report["state"] == "completed"
    found = _findings(report)
    heading = found["The heading Jacobi column follows sin(sqrt(K)s)/sqrt(K), s and sinh(sqrt(-K)s)/sqrt(-K) on "
                    "positive, zero and negative curvature"]
    assert heading["evidence_status"] == "numerically_verified"
    assert set(heading["value"]) == set(gjt.CONSTANT_PATHS) and max(heading["value"].values()) < 1e-7
    equators = found["The torus equators are geodesics of constant curvature 1/(r(R+r)) and -1/(r(R-r)) whose "
                     "Jacobi columns obey the model-space laws"]["value"]
    assert equators["torus-outer-equator"]["curvature"] == pytest.approx(1 / 3)
    assert equators["torus-inner-equator"]["curvature"] == pytest.approx(-1.0)
    separation = found["Neighbouring closed-form geodesics on the sphere and hyperbolic plane separate as |sn_K| "
                       "(heading) and |cn_K| (lateral) per unit perturbation"]
    assert separation["evidence_status"] == "numerically_verified"
    assert all(abs(c["order"] - 2) < 0.15 and c["error_at_smallest_eps"] < 5e-3
               for row in separation["value"].values() for c in row.values())
    # Reference curvatures come from surface parameters, never from gaussian_curvature.
    assert gjt._constant_curvature_of("sphere-great-circle") == 1.0 / gj.surface("sphere").radius ** 2
    assert gjt._constant_curvature_of("hyperbolic-long") == -gj.surface("hyperbolic-plane").k ** 2
    with pytest.raises(ValueError):
        gjt._constant_curvature_of("saddle")
    physical = found["Nearby real trajectories on a physical curved surface separate according to this Jacobi law"]
    assert physical["evidence_status"] == "not_established" and physical["domain"] == "physical"
    assert report["physical_validation_status"]["status"] == "not_established"
    if not lab.ctx.available("provider:csg"):
        assert any("CSG provider comparison did not run" in item for item in report["unresolved_assumptions"])


@pytest.mark.skipif(not os.environ.get("CIW_LAB_CSG_REPO"), reason="CIW_LAB_CSG_REPO names no CSG checkout")
def test_t005_csg_provider_agreement(tmp_path):
    ctx = runner.Context(tmp_path, {"csg": Path(os.environ["CIW_LAB_CSG_REPO"])})
    report = _run(ctx, "T005")
    claim = ("ciw joint geodesic + Jacobi transfer matrices match the pinned CSG provider on six constant-curvature "
             "paths")
    record = _findings(report)[claim]
    assert record["evidence_status"] == "independently_verified"
    assert record["basis"]["independent_check"]["checker"]["implementation"].startswith(gj.CSG_IMPLEMENTATION + "@")
    assert max(v["ciw_rk4_vs_csg_rk4"] for v in record["value"].values()) < 1e-9
    assert max(v["ciw_rk4_vs_csg_closed_form"] for v in record["value"].values()) < 1e-7
    pin = gj.csg_pin()
    assert report["provider_runtime_identity"]["provider"]["revision"] == pin["revision"]
    assert report["provider_runtime_identity"]["provider"]["source_tree"] == pin["source_tree"]
    focus = _run(ctx, "T008")
    labels = _labels(focus)
    assert labels["ciw conjugate and focal points match the pinned CSG provider's focus events on constant-curvature "
                  "paths"] == "independently_verified"


def _git_prefix(repo, tmp_path):
    # Signing and hooks from the developer's configuration must not affect the throwaway repository.
    return ["git", "-C", str(repo), "-c", "user.email=lab@example.invalid", "-c", "user.name=lab",
            "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={tmp_path / 'no-hooks'}"]


def test_csg_checkout_refusals(tmp_path, monkeypatch):
    # No repository above tmp_path can be discovered, so a plain directory is unreadable by construction.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    for path in (tmp_path / "missing", plain):
        with pytest.raises(gj.ProviderRefusal) as unreadable:
            gj.verify_csg_checkout(path)
        assert unreadable.value.code == "CSG_CHECKOUT_UNREADABLE" == gj.predict_csg_refusal(path)
        assert str(path) not in str(unreadable.value)
    if shutil.which("git"):
        repo = tmp_path / "other"
        repo.mkdir()
        git = _git_prefix(repo, tmp_path)
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "unpinned"], check=True)
        assert gj.predict_csg_refusal(repo) == "CSG_REVISION_MISMATCH"
        with pytest.raises(gj.ProviderRefusal) as mismatch:
            gj.verify_csg_checkout(repo)
        assert mismatch.value.code == "CSG_REVISION_MISMATCH"
        # An ignored module under the source root could shadow pinned code on import.
        (repo / ".gitignore").write_text("src/numpy.py\n", encoding="utf-8")
        (repo / "src" / "__pycache__").mkdir(parents=True)
        (repo / "src" / "numpy.py").write_text("", encoding="utf-8")
        (repo / "src" / "__pycache__" / "x.pyc").write_bytes(b"")
        assert gj.untracked_sources(repo) == ["src/numpy.py"]
    ctx = runner.Context(tmp_path / "out", {"csg": plain})
    for task_id in ("T005", "T008"):
        report = _run(ctx, task_id)
        assert report["state"] == "partial", task_id
        refused = _findings(report)["A bound CSG checkout whose git state differs from the pin is refused with the "
                                    "code that state predicts"]
        assert refused["evidence_status"] == "numerically_verified"
        assert refused["value"] == {"predicted": "CSG_CHECKOUT_UNREADABLE", "observed": "CSG_CHECKOUT_UNREADABLE"}
        check = refused["basis"]["checks"][0]
        assert check["reference_kind"] == "refusal" and check["expected_refusal"] == "CSG_CHECKOUT_UNREADABLE"
        # Report prose names the code only; the machine-specific path stays out of the report.
        assert "CSG provider refused (CSG_CHECKOUT_UNREADABLE); the comparison did not run (detail in " \
               "provider-refusal.json)" in report["unresolved_assumptions"]
        assert str(tmp_path) not in json.dumps(report)
        detail = _artifact(ctx, task_id, "provider-refusal.json")
        assert detail["stage"] == "pin" and str(tmp_path) not in detail["message"]


def test_csg_output_is_refused_unless_complete():
    cases = [{"arclength": [0.0, 0.5, 1.0], "gaussian_curvature": 1.0}]
    summary = {"matrices": [[[1, 0], [0, 1]]] * 3, "determinant": [1.0] * 3,
               "focus_events": {"a": [{"arc_length": 0.9}], "b": []}}
    good = {"python": "3.11", "numpy": "2", "traces": [{}], "maps": [{"numeric": summary, "closed_form": summary}]}
    gj._check_csg_output(good, cases)
    for broken in ({**good, "maps": []}, {**good, "traces": [{}, {}]},
                   {**good, "maps": [{"numeric": dict(summary, matrices=[]), "closed_form": summary}]},
                   {key: value for key, value in good.items() if key != "numpy"}):
        with pytest.raises((ValueError, KeyError, TypeError)):
            gj._check_csg_output(broken, cases)


def test_t006_finite_differences(lab):
    report = lab("T006")
    assert report["state"] == "completed"
    assert set(_labels(report).values()) == {"numerically_verified"}
    found = _findings(report)
    central = found["Central finite differences of perturbed geodesics converge to the integrated Jacobi columns at "
                    "second order"]["value"]
    one_sided = found["One-sided finite differences converge to the integrated Jacobi columns at first order"]["value"]
    assert set(central) == set(gjt.FD_SURFACES)
    assert all(abs(v - 2) <= 0.1 for r in central.values() for v in r.values())
    assert all(abs(v - 1) <= 0.15 for r in one_sided.values() for v in r.values())
    counter = found["Shrinking the finite-difference step far below its optimum degrades the Jacobi estimate "
                    "(cancellation)"]
    assert counter["counterexample"] and counter["value"]["log10_error_ratio_1e-11_over_best"] >= 1


def test_t007_determinant(lab):
    report = lab("T007")
    assert report["state"] == "completed"
    assert set(_labels(report).values()) == {"numerically_verified"}
    found = _findings(report)
    rk4 = found["RK4 determinant drift is O(h^5), one order above its O(h^4) global error, on constant and variable "
                "curvature"]
    assert all(abs(v - 5) <= 0.2 for v in rk4["value"].values()) and rk4["counterexample"]
    euler = found["Explicit Euler multiplies det Phi by exactly 1 + h^2 K(gamma_n) per step, so it is not "
                  "area-preserving where K is nonzero"]["value"]
    assert euler["per_step_factor_error"] < 1e-12
    assert found["Where K = 0 every method preserves det Phi = 1 exactly, Euler included"]["value"] < 1e-14
    adaptive = found["Adaptive Dormand-Prince determinant drift decreases in proportion to the tolerance or faster "
                     "(median over paths)"]
    assert adaptive["value"] >= gjt.ADAPTIVE_EXPONENT_THRESHOLD
    midpoint = found["Midpoint determinant drift is (h^2/4)(K(L) - K(0)) + O(h^3): second order on variable "
                     "curvature, third order on constant curvature"]["value"]
    assert all(abs(v - 1) < 1e-3 for v in midpoint["extrapolated_boundary_ratio"].values())
    # The exact one-step determinants for constant K.
    h, k = 0.1, 1.0
    euler_step = np.array([[1, h], [-h * k, 1]])
    assert np.linalg.det(euler_step) == pytest.approx(gjt.per_step_determinant("euler", h, k), rel=1e-14)
    assert gjt.per_step_determinant("rk4", h, 0.0) == 1.0


def test_t007_symbolic_step_determinants():
    pytest.importorskip("sympy")
    table = gjt.symbolic_step_determinants()
    assert all(table["rk4"][str(p)] == "0" for p in range(6))
    assert table["rk4"]["6"] != "0" and "k0**3" in table["rk4"]["6"]
    assert table["midpoint"]["2"] == "0" and table["midpoint"]["3"] == "k1/4"


def test_t008_conjugate_and_focal_points(lab):
    report = lab("T008")
    assert report["state"] == "completed"
    assert set(_labels(report).values()) == {"numerically_verified"}
    found = _findings(report)
    sphere = found["Sphere conjugate points lie at pi R and 2 pi R and focal points at pi R/2 and 3 pi R/2 "
                   "(R = 1 and R = 2)"]["value"]
    assert sphere["R1"]["conjugate"] == pytest.approx([math.pi, 2 * math.pi], abs=1e-7)
    assert sphere["R2"]["focal"] == pytest.approx([math.pi, 3 * math.pi], abs=1e-7)
    outer = found["On the torus outer equator conjugate points lie at pi sqrt(r(R+r)) multiples and focal points "
                  "half-way"]["value"]
    assert outer["conjugate"][0] == pytest.approx(math.pi * math.sqrt(3), abs=1e-7)
    negative = found["No conjugate or focal point occurs on the torus inner equator or the hyperbolic plane (K < 0)"]
    assert all(v["conjugate_count"] == 0 for v in negative["value"].values())
    sturm = found["Sturm comparison bound holds on every seeded torus geodesic and declared bump chord that reaches "
                  "a conjugate point: none occurs before pi/sqrt(max K)"]
    # The bound is exercised on both surfaces, not vacuous.
    assert all(n >= 1 for n in sturm["value"]["paths_with_conjugate_point"].values())
    assert all(m >= 0 for m in sturm["value"]["margins"].values())
    assert sturm["value"]["bounds"]["torus"] == pytest.approx(math.pi * math.sqrt(3))
    assert sturm["value"]["bounds"]["gaussian-bump"] == pytest.approx(2 * math.pi)
    counter = found["On variable curvature the first focal point is not half the first conjugate distance"]
    assert abs(counter["value"]["focal"] - counter["value"]["conjugate"] / 2) > 0.3 and counter["counterexample"]


def test_t009_columns_rank_paths_differently(lab):
    report = lab("T009")
    assert report["state"] == "completed"
    found = _findings(report)
    ranking = found["Lateral and heading sensitivities rank paths differently"]
    witness = ranking["counterexample"]["witness"]
    a, b = gjt.WITNESS_PAIR
    assert witness["lateral"][b] > witness["lateral"][a] and witness["heading"][a] > witness["heading"][b]
    assert ranking["value"]["kendall_tau"] < 1 and ranking["value"]["discordant_pairs"] >= 1
    assert found["Endpoint sensitivities to lateral offset (|j_lat(L)|) and heading error (|j_head(L)|) per "
                 "path"]["evidence_status"] == "numerically_verified"
    physical = [f for f in report["findings"] if f["domain"] == "physical"]
    assert physical and all(f["evidence_status"] == "not_established" for f in physical)
    kernel = found["To first order, curvature at arclength s moves j_lat(L) with weight sn(L-s) cn(s) (early-weighted) "
                   "and j_head(L) with weight sn(L-s) sn(s) (symmetric about mid-path)"]
    assert kernel["evidence_status"] == "numerically_verified"
    assert kernel["value"]["heading_early_late_asymmetry"] < 1e-6 and kernel["value"]["lateral_early_over_late"] > 3
    reversal = found["Reversing a geodesic leaves j_head(L) unchanged and exchanges j_lat(L) with j_head'(L) (transfer "
                     "matrix D Phi(L)^-1 D)"]
    assert all(v["heading_change"] < 1e-8 for v in reversal["value"].values())


def _numeric(value) -> bool:
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    items = value.values() if isinstance(value, dict) else value
    return any(_numeric(item) for item in items)


def test_every_numerical_finding_declares_uncertainty_and_tolerance(lab):
    for task_id in SECTION:
        for record in lab(task_id)["findings"]:
            if record["domain"] not in ("mathematical", "numerical") or not _numeric(record["value"]):
                continue
            uncertainty = record.get("uncertainty")
            assert isinstance(uncertainty, dict) and set(uncertainty) == {"kind", "value", "basis"}, record["claim"]
            assert math.isfinite(uncertainty["value"]) and uncertainty["basis"], record["claim"]
            assert set(record["regression_tolerance"]) == {"abs", "rel"}, record["claim"]


def test_figures_fit_their_legend_and_title_space(lab):
    legend_x = str(svg.WIDTH - svg.RIGHT + 38)
    for task_id in SECTION:
        lab(task_id)
        for path in sorted((lab.ctx.output_dir / "artifacts" / task_id).glob("*.svg")):
            texts = list(ET.parse(path).getroot().iter("{http://www.w3.org/2000/svg}text"))
            title = [t.text for t in texts if t.get("font-weight") == "bold"][0]
            legends = [t.text for t in texts if t.get("x") == legend_x]
            assert len(title) <= 72, (path.name, title)
            assert legends and all(len(name) <= 20 for name in legends), (path.name, legends)
            assert len(legends) <= len(svg.PALETTE), path.name


def test_perturbation_helpers_are_geometric():
    sphere = gj.surface("sphere")
    spec = gj.path("sphere")
    shifted = jacobi.perturbed_start(sphere, spec.u0, spec.heading, lateral=1e-3)
    assert sphere.speed_squared(shifted[:2], shifted[2:]) == pytest.approx(1.0, abs=1e-12)
    base = gj.start_state("sphere")
    separation = jacobi.normal_separation(sphere, base[None, :4], shifted[None, :])
    assert separation[0] == pytest.approx(1e-3, rel=1e-3)
