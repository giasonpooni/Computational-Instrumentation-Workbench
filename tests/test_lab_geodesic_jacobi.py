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

import numpy as np
import pytest

from ciw.lab import geodesic_jacobi as gjt
from ciw.lab import geodesic_jacobi_common as gj
from ciw.lab import integrators, jacobi, runner, surfaces
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.report import validate_report

SECTION = tuple(f"T00{i}" for i in range(1, 10))
OPTIONAL = frozenset({"module:sympy", "module:mpmath", "module:scipy"})


class BareContext(runner.Context):
    """A context that reports the optional modules as missing (the numpy-only CI job)."""

    def available(self, requirement):
        return False if requirement in OPTIONAL else super().available(requirement)


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _run(ctx, task_id):
    return validate_report(runner.run_task(_queue()[task_id], _REGISTRY[task_id], ctx, {}))


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
    for task_id in SECTION:
        implementation = _REGISTRY[task_id]
        assert implementation.regression_tests, task_id
        for node in implementation.regression_tests:
            path, _, name = node.partition("::")
            assert path == "tests/test_lab_geodesic_jacobi.py" and name in names, node
        assert "src/ciw/lab/geodesic_jacobi.py" in implementation.changed_files


def test_t001_symbolic_derivations_match_surfaces(lab):
    pytest.importorskip("sympy")
    report = lab("T001")
    assert report["state"] == "completed"
    assert report["evidence_status"]["primary"] == "independently_verified"
    labels = _labels(report)
    geodesic = _findings(report)["Sympy-derived metrics, Christoffel symbols and geodesic equations match "
                                 "ciw.lab.surfaces on nine charts"]
    assert geodesic["basis"]["independent_check"]["checker"]["implementation"] == "sympy"
    assert max(geodesic["value"].values()) < 1e-12
    assert labels["Intrinsic Brioschi curvature from sympy matches the ciw Gaussian curvature on nine charts"] \
        == "independently_verified"
    counter = [f for f in report["findings"] if f.get("counterexample")]
    assert len(counter) == 2 and all(f["evidence_status"] == "numerically_verified" for f in counter)
    assert labels["The geodesic equation u''^k = -Gamma^k_ij u'^i u'^j and its catalogue specializations follow "
                  "from the first variation of length"] == "analytic"
    text = (lab.ctx.output_dir / "artifacts" / "T001" / "derivations.txt").read_text(encoding="utf-8")
    # Printed forms vary between sympy versions; the retained text must name every derived object.
    assert "[torus]" in text and "Gamma^u_uv" in text and "Gamma^v_uu" in text and "K (Brioschi)" in text
    table = _artifact(lab.ctx, "T001", "comparison.json")
    assert table["discrepancies"]["plane-polar"]["symbolic_curvature_is_zero"] is True


def test_t001_without_sympy_is_partial(tmp_path):
    report = _run(BareContext(tmp_path), "T001")
    assert report["state"] == "partial"
    assert report["evidence_status"]["primary"] == "numerically_verified"
    assert report["evidence_status"]["counts"]["independently_verified"] == 0
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
        assert record["value"]["ciw_minus_reference"] < 1e-12
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


def test_t003_integrator_orders(lab):
    report = lab("T003")
    assert report["state"] == "completed"
    assert set(_labels(report).values()) == {"numerically_verified"}
    found = _findings(report)
    for name, p, tol in (("Explicit Euler", 1, 0.1), ("Explicit midpoint", 2, 0.1), ("Classical RK4", 4, 0.25)):
        orders = found[f"{name} global endpoint error converges at order {p} on every curved chart"]["value"]
        assert set(orders) == set(gjt.CURVED_CHARTS)
        assert all(abs(v - p) <= tol for v in orders.values())
    effective = found["Adaptive Dormand-Prince error falls with function evaluations at an effective order near 5"]
    assert abs(float(np.median(list(effective["value"].values()))) - 5) <= 0.5
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
    assert nonunit["evidence_status"] == "numerically_verified" and max(nonunit["value"].values()) < 1e-7
    counter = found["Unit speed does not certify an accurate path: renormalized Euler keeps |g - 1| at rounding "
                    "with a first-order endpoint error"]
    assert counter["value"]["renormalized_max_speed_drift"] < 1e-13
    assert counter["value"]["renormalized_endpoint_error"] > 1e-3 and counter["counterexample"]
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

    assert gjt.normalization_calls([renormalizing_step]) and gjt.normalization_calls([rescaled])
    assert gjt.normalization_calls([gjt.renormalized_euler])
    # A speed-1.3 start keeps its speed, which a renormalizing integrator could not do.
    sphere = gj.surface("sphere")
    y0 = gj.start_state("sphere")[:4].copy()
    y0[2:] *= 1.3
    _, states = integrators.integrate_fixed(sphere.geodesic_rhs, y0, 1.0, 64, "rk4")
    assert sphere.speed_squared(states[-1, :2], states[-1, 2:]) == pytest.approx(1.69, abs=1e-8)


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


def test_csg_checkout_refusals(tmp_path):
    with pytest.raises(gj.ProviderRefusal) as unreadable:
        gj.verify_csg_checkout(tmp_path / "missing")
    assert unreadable.value.code == "CSG_CHECKOUT_UNREADABLE"
    if shutil.which("git"):
        repo = tmp_path / "other"
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.email=lab@example.invalid", "-c", "user.name=lab"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "unpinned"], check=True)
        with pytest.raises(gj.ProviderRefusal) as mismatch:
            gj.verify_csg_checkout(repo)
        assert mismatch.value.code == "CSG_REVISION_MISMATCH"
    ctx = runner.Context(tmp_path / "out", {"csg": tmp_path})
    report = _run(ctx, "T005")
    assert report["state"] == "partial"
    refused = _findings(report)["A bound CSG provider that fails pin verification or execution is refused rather "
                                "than compared"]
    assert refused["evidence_status"] == "numerically_verified"
    # A temporary directory is normally no repository; inside one, git would report another revision.
    assert refused["value"] in ("CSG_CHECKOUT_UNREADABLE", "CSG_REVISION_MISMATCH")
    assert refused["basis"]["checks"][0]["reference_kind"] == "refusal"


def test_t006_finite_differences(lab):
    report = lab("T006")
    assert report["state"] == "completed"
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
    sturm = found["Sturm comparison bound holds: no conjugate point before pi/sqrt(max K) on seeded torus and bump "
                  "geodesics"]
    assert sturm["value"]["margin"] >= 0 and sturm["value"]["bounds"]["torus"] == pytest.approx(math.pi * math.sqrt(3))
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


def test_perturbation_helpers_are_geometric():
    sphere = gj.surface("sphere")
    spec = gj.path("sphere")
    shifted = jacobi.perturbed_start(sphere, spec.u0, spec.heading, lateral=1e-3)
    assert sphere.speed_squared(shifted[:2], shifted[2:]) == pytest.approx(1.0, abs=1e-12)
    base = gj.start_state("sphere")
    separation = jacobi.normal_separation(sphere, base[None, :4], shifted[None, :])
    assert separation[0] == pytest.approx(1e-3, rel=1e-3)
