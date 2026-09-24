"""Lyapunov runtime experiments T101-T114.

Provider-free parts (exact references, the documented-rule transcription, the
ISS branch, the residual adapter and the servo specification) always run.
Parts that execute the pinned PLSR runtime need a Python 3.12+ interpreter with
it installed: set CIW_LAB_PLSR_PYTHON, or run the tests under such an
interpreter; otherwise they are skipped. A CIW_LAB_PLSR_PYTHON that names a
missing interpreter fails the provider tests instead of skipping them.
"""
import ast
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
from ciw.lab import runner
from ciw.lab import lyapunov_reference as R
from ciw.lab import lyapunov_research as X
from ciw.lab.evidence import COMPUTATIONAL_DOMAINS
from ciw.lab.lyapunov_provider import PLSR_ROLE, ProviderRefusal, manifest, run_plsr
from ciw.lab.registry import load_implementations, load_queue
from ciw.lab.report import validate_report
from ciw.lab.runner import Context, run_queue, run_task

PROVIDER_TASKS = [f"T1{n:02d}" for n in range(1, 12)] + ["T113", "T114"]


def _plsr_python():
    configured = os.environ.get("CIW_LAB_PLSR_PYTHON")
    if configured:
        return configured  # checked by _provider(): a missing path fails rather than skips
    if sys.version_info >= (3, 12) and importlib.util.find_spec("lyapunov") is not None:
        return sys.executable
    return None


PLSR_PYTHON = _plsr_python()
needs_provider = pytest.mark.skipif(PLSR_PYTHON is None, reason="set CIW_LAB_PLSR_PYTHON to a Python 3.12+ "
                                                                "interpreter with the pinned PLSR runtime")


def _provider():
    if not Path(PLSR_PYTHON).exists():
        pytest.fail(f"CIW_LAB_PLSR_PYTHON names a missing interpreter: {PLSR_PYTHON}")
    return PLSR_PYTHON


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """Run every provider-dependent task once with the bound runtime."""
    if PLSR_PYTHON is None:
        pytest.skip("PLSR provider interpreter not configured")
    python = _provider()
    directory = tmp_path_factory.mktemp("lyapunov")
    run_queue(directory, task_ids=PROVIDER_TASKS, providers={PLSR_ROLE: python})
    return {task_id: validate_report(json.loads((directory / "reports" / f"{task_id}.json").read_text(encoding="utf-8")))
            for task_id in PROVIDER_TASKS}


def _run(task_id, tmp_path, providers=None):
    implementations, _ = load_implementations()
    item = next(t for t in load_queue()["tasks"] if t["id"] == task_id)
    return validate_report(run_task(item, implementations[task_id], Context(tmp_path, providers), {}))


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert matches, f"no finding starting with {prefix!r}"
    return matches[0]


def _label(report, prefix):
    return _finding(report, prefix)["evidence_status"]


def _regression_ready(report):
    """Every computational finding carries a regression tolerance and an uncertainty; none is refuted."""
    for record in report["findings"]:
        if record["domain"] in COMPUTATIONAL_DOMAINS:
            assert "regression_tolerance" in record, record["claim"]
            assert record.get("uncertainty") is not None, record["claim"]
            assert record["evidence_status"] != "not_established" or record.get("expected_not_established"), \
                record["claim"]


def _completed(report, primary):
    assert report["state"] == "completed", report["unresolved_assumptions"]
    assert report["evidence_status"]["primary"] == primary
    assert report["provider_runtime_identity"]["provider"]["files_verified"] == 20
    _regression_ready(report)


# Provider-free references ------------------------------------------------------

@pytest.mark.lab_task("T101")
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


@pytest.mark.lab_task("T101")
def test_t101_threshold_is_analytic():
    _, meta = L._t101_family()
    normal = [m for m in meta if m["family"] == "F1" and m["normal"]]
    assert len(normal) == 112
    # The analytic code depends on eps only: certified iff eps > eps*, whatever the scale 2^k.
    assert {L._t101_expected(m["eps"]) for m in normal if m["eps"] > L.EPSILON_STAR} == {"CERTIFIED_WITH_MARGIN"}
    mismatches = [m for m in normal
                  if R.documented_code(m["A"], m["P"], (1.0, 0.5))["code"] != L._t101_expected(m["eps"])]
    assert mismatches == []
    offline = {f["claim"]: f for f in L._t101_offline(meta)}
    assert offline["The documented decision order, transcribed in CIW, places the family's threshold at eps* in "
                   "the normal range"]["evidence_status"] == "numerically_verified"


@pytest.mark.lab_task("T102")
def test_eigvalsh_scaling_window():
    inside, _ = L._eigvalsh_window()
    assert inside and all(inside)


@pytest.mark.lab_task("T102")
def test_discrete_razor_edge_sits_at_the_threshold():
    rng = R.generator(4242)
    for side in (-1.0, 1.0):
        A, P, ratio = R.razor_edge_discrete(rng, 2, side)
        assert 1.0 < ratio < 1.1
        code = R.documented_code(A, P, np.ones(2), "discrete")["code"]
        assert code == ("CERTIFIED_WITH_MARGIN" if side < 0 else "DECREASE_NOT_DEFINITE")
        assert R.resolution_bin(R.exact_form(A, P, "discrete"), R.resolution(A, P, "discrete")) in L.BAND


@pytest.mark.lab_task("T103")
def test_level_gate_prediction():
    rows = L.level_scan()
    missed, spurious = L._level_counts(rows, "documented_exceeded")
    assert (len(rows), missed, spurious) == (68, 5, 9)
    first = next(r for r in rows if r["exact_exceeded"] and not r["documented_exceeded"])
    assert 2 * first["e"] < -1074  # s^2 underflows to zero while V = 2^(p + 2e) is representable
    assert R.documented_level_exceeded(1.0, 0, 0.5) and not R.documented_level_exceeded(1.0, 0, 2.0)


@pytest.mark.lab_task("T104")
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


@pytest.mark.lab_task("T105")
def test_conversion_scan():
    scan = L.conversion_scan(400)
    collision = scan["witnesses"].get("collision 0.001")
    assert scan["neighbour_collisions"]["0.001"] >= 1 and collision is not None
    bound = collision["bound"]
    assert math.nextafter(bound, math.inf) * 1e-3 == bound * 1e-3
    assert scan["formula_disagreements"]["0.001"] >= 1


@pytest.mark.lab_task("T106")
def test_documented_decision_order():
    predictions = L._path_cases(L.status_paths(), [])[1]
    assert [predictions[f"required_margin (margin 2)#{j}"] for j in range(6)] == (
        ["CERTIFIED_WITH_MARGIN"] * 3 + ["MARGIN_LOW"] * 3)
    assert predictions["matrix scale k, A = -2^k I#3"] == "NUMERICAL_OVERFLOW"
    assert predictions["theta, box [-1, 1]#4"] == "OUTSIDE_PARAMETER_BOX"
    assert predictions["stability a, A = [[a, 1], [-1, a]]#3"] == "NUMERICAL_INCONCLUSIVE"
    assert predictions["direction phi, A = diag(-1, 1)#4"] == "NOT_CERTIFIED"
    assert set(predictions.values()) == set(L.ROUNDING_FREE_CODES)


@pytest.mark.lab_task("T108")
def test_documented_rule_is_monotone():
    for member in L.near_threshold_family(108, 12):
        info = R.documented_code(member["A"], member["P"], member["x"])
        steps = [dict(code=R.documented_code(member["A"], member["P"], member["x"], required_margin=r)["code"])
                 for r in L.margin_grid(info["margin"], info["resolution"])]
        violations = L.monotonicity_violations(steps)
        assert set(violations) == {"passing_regained", "noncertifying_code_changed"}  # codes only
        assert sum(violations.values()) == 0
    # The detector itself fails on a sequence that regains a passing code.
    regained = [{"code": "MARGIN_LOW"}, {"code": "CERTIFIED_WITH_MARGIN"}]
    assert L.monotonicity_violations(regained)["passing_regained"] == 1


@pytest.mark.lab_task("T106")
def test_transition_graph():
    graph = L.transition_graph()
    statuses = {key: entry["status"] for key, entry in graph.items()}
    assert sorted(statuses.values()).count("allowed") == 24
    assert sorted(statuses.values()).count("rounding only") == 8
    assert sorted(k for k, v in statuses.items() if v == "excluded") == sorted([
        "CERTIFIED_WITH_MARGIN <-> NOT_CERTIFIED", "CERTIFIED_WITH_MARGIN <-> DECREASE_NOT_DEFINITE",
        "MARGIN_LOW <-> NOT_CERTIFIED", "MARGIN_LOW <-> DECREASE_NOT_DEFINITE"])
    assert graph["NOT_CERTIFIED <-> NUMERICAL_INCONCLUSIVE"]["crossings"] == [["scalar_positive", "not_definite"]]
    # Every allowed pair occurs directly between consecutive steps of the transcribed paths, at its crossing.
    paths = L.status_paths()
    _, predictions, gates = L._path_cases(paths, [])
    rows = L.path_transitions(paths, predictions, gates, graph)
    coverage = L.transition_coverage(graph, rows)
    assert all(coverage[k]["exercised_on"] for k, v in statuses.items() if v == "allowed")
    assert all(row["direct"] for row in rows)
    # A step across two thresholds at once (CERTIFIED_WITH_MARGIN straight to NOT_CERTIFIED) is not direct.
    skip = {"skip": [dict(A=-np.eye(2), P=np.eye(2), x=np.array([1.0, 0.0])),
                     dict(A=np.eye(2), P=np.eye(2), x=np.array([1.0, 0.0]))]}
    _, codes, skip_gates = L._path_cases(skip, [])
    assert [r["direct"] for r in L.path_transitions(skip, codes, skip_gates, graph)] == [False]
    for steps in paths.values():
        for step in steps:
            evaluated = L.step_gates(step)
            A, P, x, in_box, options = L._step_inputs(step)
            assert evaluated["code"] == R.documented_code(A, P, x, in_box=in_box, **options)["code"]
    assert "x" in L.coverage_markdown(coverage) and "!" not in L.coverage_markdown(coverage).split("\n\n")[1]


@pytest.mark.lab_task("T103")
def test_representability_reference():
    assert R.representable(Fraction(0)) and R.representable(Fraction(1, 3))
    assert R.representable(Fraction(R.TINY)) and not R.representable(Fraction(R.TINY) / 3)
    assert not R.representable(Fraction(2) ** 1024) and R.representable(Fraction(L.DBL_MAX))
    # 0.5625 * 2^-1074 rounds to 2^-1074: representable, although PLSR's two-step product reports 0.
    assert R.representable(Fraction(9, 16) * Fraction(R.TINY))


@pytest.mark.lab_task("T101", "T102", "T103", "T104", "T106", "T109")
def test_checks_are_unconditional_and_observed_values_are_computed():
    """No check is added only after its outcome was observed, and none records a literal observed value.

    A check built inside an ``if`` on an observed outcome cannot fail, and a literal observed value is a number
    chosen after the fact. The only conditional checks allowed depend on which optional reference module is
    installed (``if independent is not None``), not on a result.
    """
    tree = ast.parse(Path(L.__file__).read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    allowed_conditions = {"independent is not None"}
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("_check",
                                                                                                  "_refusal")):
            continue
        observed = node.args[1] if node.func.id == "_check" and len(node.args) > 1 else None
        if isinstance(observed, ast.Constant):
            problems.append(f"line {node.lineno}: literal observed value {observed.value!r}")
        ancestor = parents.get(node)
        while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.Lambda)):
            if isinstance(ancestor, (ast.If, ast.IfExp)) and ast.unparse(ancestor.test) not in allowed_conditions:
                problems.append(f"line {node.lineno}: check built under 'if {ast.unparse(ancestor.test)}'")
            ancestor = parents.get(ancestor)
    assert problems == []


@pytest.mark.lab_task("T109")
def test_numpy_misreads_exact_jordan_block():
    cases = [c for c in L.adversarial_cases() if c["group"] == "Jordan"]
    largest = max(cases, key=lambda c: c["A"].shape[0])
    lam = -largest["exact_spectrum"][0]
    # A = T J T^-1 with T Ti = I holds exactly (asserted while building); the float spectrum moves by eps^(1/n).
    error = abs(largest["numpy_abscissa"] + lam)
    assert error > 1e3 * np.finfo(float).eps * np.max(np.abs(largest["A"]))
    assert all(abs(c["numpy_abscissa"] + (-c["exact_spectrum"][0])) > 1e3 * np.finfo(float).eps
               * np.max(np.abs(c["A"])) for c in cases)


def test_bridge_refuses_unusable_interpreter():
    if importlib.util.find_spec("lyapunov") is not None and sys.version_info >= (3, 12):
        pytest.skip("this interpreter hosts the runtime")
    with pytest.raises(ProviderRefusal) as caught:
        run_plsr(sys.executable, [])
    assert caught.value.code in ("PLSR_PYTHON_UNSUPPORTED", "PLSR_UNAVAILABLE")


@pytest.mark.lab_task("T101", "T102", "T103", "T104", "T105", "T106", "T107", "T108", "T109", "T110", "T111", "T113",
                      "T114")
@pytest.mark.parametrize("task_id", PROVIDER_TASKS)
def test_provider_tasks_are_partial_without_the_provider(task_id, tmp_path):
    report = _run(task_id, tmp_path)
    assert report["state"] == "partial"
    assert "not bound" in report["experiment"] and "not bound" in report["unresolved_assumptions"][0]
    assert report["provider_runtime_identity"]["provider"]["executed"] is False
    assert report["evidence_status"]["primary"] != "not_established"
    assert all("provider" not in f["basis"] for f in report["findings"])
    _regression_ready(report)
    assert report["recommended_next_task"] == L.NEXT_STEPS[task_id]


@pytest.mark.lab_task("T101", "T102", "T103", "T104", "T105", "T106", "T107", "T108", "T109", "T110", "T111", "T112",
                      "T113", "T114")
def test_next_steps_name_forward_work(tmp_path):
    """Each next step is the task's own open question, never the next queue task, which has already run."""
    assert sorted(L.NEXT_STEPS) == [f"T1{n:02d}" for n in range(1, 15)]
    for task_id, text in L.NEXT_STEPS.items():
        assert text.startswith("Deferred research question"), task_id
        assert not text.split(": ", 1)[1].startswith("T1"), task_id
    # A hardware-gated step names the route by which acquired bytes could enter the task, what the physical gate
    # needs beyond them, and where the run is retained.
    gated = sorted(task_id for task_id, text in L.NEXT_STEPS.items()
                   if text.startswith("Deferred research question (hardware-gated)"))
    assert gated == ["T113", "T114"] == sorted(L.CAPTURE_ROLES)
    for task_id in gated:
        text, role = L.NEXT_STEPS[task_id], L.CAPTURE_ROLES[task_id]
        for fragment in (f"ctx.capture('{role}')", f"ciw lab run {task_id} --capture {role}=PATH", "raw_sha256",
                         "calibration", f"runner.CAPTURE_INSTRUMENTS has no entry for {role}",
                         "signed-capture trust anchor", "ciw lab hardware retain under lab/hardware/<run-id>",
                         "stay not_established even when such data exist"):
            assert fragment in text, (task_id, fragment)
        assert role not in runner.CAPTURE_INSTRUMENTS  # the step's claim that no instrument probe exists
    # T112 runs without the provider and completes: its report carries the same next step.
    assert _run("T112", tmp_path)["recommended_next_task"] == L.NEXT_STEPS["T112"]


@pytest.mark.lab_task("T101", "T102")
@pytest.mark.parametrize("task_id", ["T101", "T102"])
def test_t101_t102_defer_cross_platform_reproduction_as_one_question(task_id, tmp_path):
    assumptions = _run(task_id, tmp_path)["unresolved_assumptions"]
    assert assumptions.count(L.PLATFORM_QUESTION) == 1
    assert not [a for a in assumptions if "BLAS" in a and a != L.PLATFORM_QUESTION]
    for fragment in ("Windows x86-64", "macOS arm64", "Linux x86-64", "OpenBLAS", "case for case",
                     "regression tolerance"):
        assert fragment in L.PLATFORM_QUESTION, fragment


@pytest.mark.lab_task("T106")
def test_t106_offline_findings_without_the_provider(tmp_path):
    report = _run("T106", tmp_path)
    assert _label(report, "A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation") \
        == "not_established"
    assert _label(report, "The documented decision order, transcribed in CIW, assigns the eight") \
        == "numerically_verified"


@pytest.mark.lab_task("T112")
def test_t112_iss_branch(tmp_path):
    report = _run("T112", tmp_path)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "analytic"
    _regression_ready(report)
    simulated = _finding(report, "For the four simulated disturbance classes")
    assert simulated["evidence_status"] == "numerically_verified"
    assert max(simulated["value"]["to_sharp_supremum"].values()) <= 1.0 + 1e-5
    assert simulated["value"]["to_sharp_supremum"]["worst-case switching"] >= 0.98
    sharp = _finding(report, "The sharp reachable-set supremum")
    assert sharp["evidence_status"] == "numerically_verified"
    assert sharp["value"]["sharp_supremum"] == pytest.approx(0.411754, rel=1e-5)
    assert sharp["value"]["to_iss_bound"] < 0.26
    assert _label(report, "Quadratic ISS-Lyapunov bound") == "analytic"
    approached = _finding(report, "In one dimension")
    assert approached["value"]["ratio"] == pytest.approx(1.0 - math.exp(-30.0), abs=1e-13)
    # The oscillator's augmented matrix is diagonalisable, so an independent exponential always applies.
    assert _label(report, "The series matrix exponential") == "independently_verified"
    assert _label(report, "The disturbance bound") == "not_established"
    assert _finding(report, "The ISS bound defines a safe")["domain"] == "machine_safety"
    assert report["physical_validation_status"]["status"] == "not_established"


@pytest.mark.lab_task("T113")
def test_adapter_keeps_metadata_outside():
    windows = X.adapter_windows()
    outcomes = {w["name"]: X.adapt(w["envelope"]) for w in windows}
    host = [o["host_status"] for o in outcomes.values() if "host_status" in o]
    assert sorted(host) == ["CERTIFICATE_EXPIRED", "INVALID_SENSOR_DATA", "MODEL_MISMATCH", "STALE_STATE"]
    forwarded = {name: o for name, o in outcomes.items() if "samples" in o}
    assert len(forwarded) == 4
    for outcome in forwarded.values():
        samples, stats = outcome["samples"], outcome["statistics"]
        assert tuple(samples) == X.SAMPLE_ROLES and all(tuple(s) == X.SAMPLE_FIELDS for s in samples.values())
        assert samples["upper"]["theta"][0] == stats["theta"] + X.GUARD_SE * stats["theta_se"]
    near = forwarded["estimate near bound theta 0.48"]
    assert near["samples"]["estimate"]["theta"][0] <= X.ADAPTER_BOX[1] < near["samples"]["upper"]["theta"][0]
    envelope = windows[0]["envelope"]
    sample = forwarded["nominal theta 0.1"]["samples"]["estimate"]
    assert X.metadata_leaks({"x": sample["x"], "theta": sample["theta"]}, envelope) == []
    assert X.metadata_leaks({"x": [0.0], "note": f"from {envelope['sensor_id']}"}, envelope) == [
        "value encoder-axis-7"]
    assert "key calibration_ref" in X.metadata_leaks({"calibration_ref": 1.0}, envelope)


def _solver_label():
    """Agreement with SciPy's Lyapunov solvers is independent; the CIW Kronecker fallback is not."""
    return "independently_verified" if importlib.util.find_spec("scipy") is not None else "numerically_verified"


def _exponential_label():
    optional = any(importlib.util.find_spec(name) is not None for name in ("scipy", "mpmath"))
    return "independently_verified" if optional else "numerically_verified"


@pytest.mark.lab_task("T114")
def test_t114_servo_pilot_spec(tmp_path):
    report = _run("T114", tmp_path)
    # Without the provider the monitor scan runs only in the CIW transcription: partial, not completed.
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "numerically_verified"
    _regression_ready(report)
    grid = _finding(report, "The unit-balanced nominal P")
    assert grid["evidence_status"] == "numerically_verified" and grid["value"]["not_negative_definite"] == 0
    assert _finding(report, "With Q = I the nominal-model P")["counterexample"]
    assert _label(report, "The ZOH exponential") == _exponential_label()
    monitor = _finding(report, "By the CIW transcription of the documented decision order, the monitor's code")
    assert monitor["evidence_status"] == "numerically_verified"
    assert monitor["value"]["codes_without_level"] == ["CERTIFIED_WITH_MARGIN"]
    assert monitor["value"]["codes_with_level"] == ["CERTIFIED_WITH_MARGIN", "OUTSIDE_LEVEL_SET"]
    assert monitor["value"]["level_mismatches"] == 0
    assert _label(report, "By the CIW transcription of the documented decision order, every runtime code") \
        == "numerically_verified"
    envelope = _finding(report, "The declared level set {V <= c} lies inside the operating envelope")
    assert envelope["evidence_status"] == "numerically_verified"
    assert 0.99 < envelope["value"]["sampled_boundary_extent_ratio"] <= 1.0
    assert 0.99 < envelope["value"]["exact_largest_squared_extent_ratio"] <= 1.0
    for domain in ("machine_safety", "actuator_authority", "production_acceptance", "industrial_readiness",
                   "physical", "calibration"):
        assert [f["evidence_status"] for f in report["findings"] if f["domain"] == domain] == ["not_established"]
    spec = json.loads((tmp_path / "artifacts" / "T114" / "servo-pilot-spec.json").read_text(encoding="utf-8"))
    assert set(X.SERVO_SPEC_SECTIONS) <= set(spec)  # schema guard for the retained specification
    assert spec["authority_and_safety"]["actuator_authority"].startswith("none")
    criteria = " ".join(spec["abort_criteria"])
    assert "OUTSIDE_LEVEL_SET" in criteria and "OUTSIDE_PARAMETER_BOX" not in criteria
    assert "NOT_CERTIFIED or DECREASE_NOT_DEFINITE" not in criteria
    assert spec["lyapunov_check_scope"]["runtime_codes"]["abort_on"] == ["OUTSIDE_LEVEL_SET"]
    assert spec["lyapunov_check_scope"]["runtime_codes"]["scan_evaluated_by"].startswith("CIW transcription")


@pytest.mark.lab_task("T114")
def test_level_set_extent_is_checked_independently():
    _, models = X.servo_models()
    P, _ = X.servo_certificate(models)
    level = X.servo_level(P)
    assert X.level_set_extent(P, level) <= 1
    assert X.level_set_boundary_extent(P, level) <= 1.0
    # A level four times larger doubles the ellipsoid: both checks must then fail.
    assert X.level_set_extent(P, 4.0 * level) > 1
    assert X.level_set_boundary_extent(P, 4.0 * level) > 1.0


@pytest.mark.lab_task("T112", "T114")
def test_research_tasks_without_optional_modules(tmp_path, monkeypatch):
    # The plain CI job has NumPy only: neither exponential check may depend on SciPy or mpmath. SymPy is
    # blocked too: an unimported SymPy would import the blocked mpmath when the runner records module versions.
    for name in ("scipy", "scipy.linalg", "mpmath", "sympy"):
        monkeypatch.setitem(sys.modules, name, None)
    augmented = np.zeros((3, 3))
    _, models = X.servo_models()
    augmented[:2, :2], augmented[:2, 2:] = models[0]["A"], models[0]["B"]
    assert X.independent_expm(augmented * X.SERVO["Ts_s"]) is None  # defective: no eigendecomposition
    servo = _run("T114", tmp_path / "servo")
    assert servo["state"] == "partial"  # the provider is not bound here
    exponential = _finding(servo, "The ZOH exponential")
    assert exponential["evidence_status"] == "numerically_verified"
    assert exponential["value"]["max_abs_difference_independent"] is None
    iss = _run("T112", tmp_path / "iss")
    assert iss["state"] == "completed"
    assert _label(iss, "The series matrix exponential") == "independently_verified"
    assert "numpy.linalg.eig" in _finding(iss, "The series matrix exponential")["basis"]["independent_check"][
        "checker"]["implementation"]


# Provider-backed ---------------------------------------------------------------

@needs_provider
def test_bridge_verifies_the_source_pin():
    python = _provider()
    tampered = manifest()
    tampered["files"]["runtime.py"] = "0" * 64
    with pytest.raises(ProviderRefusal) as caught:
        run_plsr(python, [], pin=tampered)
    assert caught.value.code == "PLSR_SOURCE_MISMATCH" and "runtime.py" in str(caught.value)
    wrong = dict(manifest(), package_version="0.0.0")
    with pytest.raises(ProviderRefusal, match="PLSR_VERSION_MISMATCH"):
        run_plsr(python, [], pin=wrong)
    identity = run_plsr(python, [{"id": "c", "op": "constants"}])["identity"]
    assert identity["files_verified"] == identity["files_pinned"] == 20
    assert identity["commit"] == "19ea6967060166ba09db6cd4563bd87bd6b3d196"


@pytest.mark.lab_task("T101")
@needs_provider
def test_t101_resolution_floor(reports):
    report = reports["T101"]
    _completed(report, "analytic")
    assert _label(report, "PLSR decrease_resolution equals") == "numerically_verified"
    inside = _finding(report, "Inside the binary64 normal range")
    assert inside["evidence_status"] == "numerically_verified"
    assert inside["value"]["analytic_mismatches"] == inside["value"]["unit_scale_mismatches"] == 0
    assert inside["value"]["max_normalised_resolution_deviation"] == 0.0
    assert _label(report, "NUMERICAL_OVERFLOW first appears") == "numerically_verified"
    witness = _finding(report, "PLSR's resolution of a subnormal plant is zero")
    assert witness["evidence_status"] == "numerically_verified"
    assert witness["value"]["exact_det_units2"] == -1.0 and witness["value"]["resolution"] == 0.0
    assert witness["value"]["plsr_form_units"] == [[-4.0, 4.0], [4.0, -6.0]]
    assert witness["value"]["formation_error_units"] == 1.0
    assert witness["counterexample"]["statement"].startswith("The float64 resolution floor")
    # The witness's code depends on LAPACK's subnormal handling: retained in the artifact, never in a finding.
    assert all("code" not in f["value"] for f in report["findings"] if isinstance(f["value"], dict)
               and "formation_error_units" in f["value"])


@pytest.mark.lab_task("T102")
@needs_provider
def test_t102_power_of_two_scaling(reports):
    report = reports["T102"]
    _completed(report, "numerically_verified")
    inside = _finding(report, "Power-of-two scaling of (A, P, x) inside")
    assert inside["value"]["code_flips"] == 0 and inside["value"]["ratio_changes"] == 0
    assert inside["evidence_status"] == "numerically_verified"
    discrete = _finding(report, "Scaling P and x by powers of two")
    assert discrete["value"] == {"code_flips": 0, "evaluations": 36}
    outside = _finding(report, "Outside LAPACK's scaling window")
    assert outside["value"]["unsound"] == 0 and outside["evidence_status"] == "numerically_verified"
    assert _label(report, "No unscaled PLSR verdict certifies") == "independently_verified"
    witness = _finding(report, "Scaling the witness by 2^-1074")
    assert witness["value"] == {"unit_code": "DECREASE_NOT_DEFINITE", "scaled_resolution": 0.0}
    assert "counterexample" not in witness and "counterexample" not in outside
    assert set(outside["value"]) == {"evaluations", "unsound"}


@pytest.mark.lab_task("T103")
@needs_provider
def test_t103_overflow_underflow(reports):
    report = reports["T103"]
    _completed(report, "numerically_verified")
    gate = _finding(report, "PLSR decides the level gate")
    assert gate["value"]["disagreements"] == 0 and gate["evidence_status"] == "numerically_verified"
    missed = _finding(report, "The PLSR level gate misses")
    assert missed["value"]["missed"] == 5 and missed["evidence_status"] == "numerically_verified"
    spurious = _finding(report, "The PLSR level gate reports")
    assert spurious["value"]["spurious"] == 9 and spurious["counterexample"]
    refused = _finding(report, "Non-finite states")
    assert refused["value"] == {"inf": "raises ValueError", "nan": "raises ValueError"}
    assert refused["evidence_status"] == "numerically_verified"
    theta = _finding(report, "A finite in-box theta")
    assert theta["value"]["theta:+1e308,c=2"] == "raises ValueError"
    assert theta["value"]["theta:+1e308,c=1"] == "NUMERICAL_OVERFLOW"
    reported = _finding(report, "PLSR's reported V and x^T M x equal the exact values")
    assert reported["evidence_status"] == "independently_verified"
    assert reported["value"]["missed_flags"] == 0 and reported["value"]["states"] == 37
    conservative = _finding(report, "PLSR sets value_out_of_range and reports V = 0")
    assert conservative["evidence_status"] == "numerically_verified"
    assert conservative["value"]["representable_but_flagged"] == 2 and conservative["counterexample"]


@pytest.mark.lab_task("T101", "T103")
@needs_provider
def test_t101_t103_retain_the_subnormal_witness_once(reports):
    """T101 keeps the subnormal-witness finding; T103 cites it and re-evaluates the witness for its artifact only."""
    claims = {task_id: [f["claim"] for f in reports[task_id]["findings"]] for task_id in ("T101", "T103")}
    assert claims["T101"].count(L.WITNESS_CLAIM) == 1 and L.WITNESS_CLAIM not in claims["T103"]
    result = reports["T103"]["numerical_result"]
    assert f"T101 retains its finding '{L.WITNESS_CLAIM}'" in result
    assert "formed decrease matrix off the exact form by 1 x 2^-1074" in result and "resolution 0," in result
    assert "subnormal plant's decrease form" not in result  # the refutation is T101's
    assert "T101's finding" in reports["T103"]["hypothesis"]
    # Deduplication is report bookkeeping guarded by this test, not a failure mode T103's experiment checks.
    assert not [m for m in reports["T103"]["failure_modes_checked"] if "witness" in m or "T101" in m]


@pytest.mark.lab_task("T104")
@needs_provider
def test_t104_semidefinite_edges(reports):
    report = reports["T104"]
    _completed(report, "numerically_verified")
    sound = _finding(report, "PLSR never certifies a semidefinite")
    assert sound["evidence_status"] == "independently_verified" and sound["value"]["violations"] == 0
    assert _finding(report, "Every edge-case code")["value"]["unexpected_codes"] == 0
    refusals = _finding(report, "PLSR's Lyapunov solver and certificate constructor")["value"]
    assert refusals["solve:psdQ"] == refusals["quadratic:psd"] == "raises ValueError"
    candidates = _finding(report, "No verdict certifies with a P that quadratic() accepts")
    assert candidates["value"]["certifying_verdicts"] == 0
    assert candidates["evidence_status"] == "numerically_verified"


@pytest.mark.lab_task("T105")
@needs_provider
def test_t105_unit_scales(reports):
    report = reports["T105"]
    _completed(report, "numerically_verified")
    assert _finding(report, "Interior and boundary stiffness samples")["value"]["mismatches"] == 0
    assert _label(report, "check_vertices passes") == "numerically_verified"
    collision = _finding(report, "A parameter just above the SI bound")
    assert collision["value"] == {"SI": "OUTSIDE_PARAMETER_BOX", "x1e-3": "CERTIFIED_WITH_MARGIN"}
    assert collision["evidence_status"] == "numerically_verified" and collision["counterexample"]
    formula = _finding(report, "A parameter exactly on the SI bound")
    assert formula["value"] == {"SI": "CERTIFIED_WITH_MARGIN", "x1e-3": "OUTSIDE_PARAMETER_BOX"}
    assert _label(report, "The declared box bounds 8 and 12 N/m") == "numerically_verified"
    neighbour = _finding(report, "A binary64 neighbour just outside the declared stiffness box")
    assert neighbour["evidence_status"] == "numerically_verified" and neighbour["counterexample"]
    assert neighbour["value"]["neighbours_admitted_by_second_formula"] == {
        "um, ms, N/um|k just below 8": "CERTIFIED_WITH_MARGIN"}
    light = _finding(report, "The light-damping plant's verdict")
    assert light["value"]["m, s, N/m"] == "CERTIFIED_WITH_MARGIN"
    assert light["value"]["m, ms, N/m"] == "NUMERICAL_INCONCLUSIVE"
    assert light["evidence_status"] == "numerically_verified"
    assert _label(report, "The declared stiffness box") == "not_established"


@pytest.mark.lab_task("T106")
@needs_provider
def test_t106_status_coverage(reports):
    report = reports["T106"]
    _completed(report, "numerically_verified")
    coverage = _finding(report, "The eight rounding-free runtime-status-v1 codes")
    assert coverage["evidence_status"] == "numerically_verified"
    assert sorted(coverage["value"]["codes"]) == sorted(L.ROUNDING_FREE_CODES)
    assert _label(report, "Codes along each one-parameter path") == "numerically_verified"
    witnesses = _finding(report, "The exactly indefinite P witnesses")
    assert witnesses["value"] == {"witnesses": 8, "certifying": 0}
    assert witnesses["evidence_status"] == "numerically_verified"
    transitions = _finding(report, "Every transition the decision order allows")
    assert transitions["evidence_status"] == "numerically_verified"
    assert transitions["value"]["allowed"] == transitions["value"]["exercised"] == 24
    assert transitions["value"]["undeclared"] == 0
    singular = _finding(report, "An affine certificate that is singular at an in-box theta")
    assert singular["value"] == {"theta = -1": "raises ValueError", "theta = -0.5": "CERTIFIED_WITH_MARGIN"}
    assert singular["evidence_status"] == "numerically_verified" and singular["counterexample"]
    assert any("CERTIFICATE_NOT_POSITIVE are not exercised" in text for text in report["unresolved_assumptions"])
    host = _finding(report, "The runtime refuses to emit")["value"]
    assert all(v == {"Verdict": "raises ValueError", "require_status": "raises ValueError"} for v in host.values())
    constants = _finding(report, "Pinned runtime constants")
    assert constants["value"]["DECREASE_RESOLUTION_FACTOR"] == 1.0
    assert constants["evidence_status"] == "numerically_verified"
    assert _label(report, "A CERTIFIED_WITH_MARGIN verdict") == "not_established"


@pytest.mark.lab_task("T107")
@needs_provider
def test_t107_inconclusive_band(reports):
    report = reports["T107"]
    _completed(report, "provider_backed")
    for prefix in ("No near-boundary case receives", "Beyond two resolutions", "With a declared margin of three"):
        assert _label(report, prefix) == "independently_verified"
    assert _finding(report, "No near-boundary case receives")["value"]["violations"] == 0
    assert _finding(report, "Beyond two resolutions")["value"]["unresolved"] == 0
    assert _finding(report, "With a declared margin of three")["value"]["certified"] == 0
    band = _finding(report, "At required_margin 0 near-boundary spectra")
    assert band["evidence_status"] == "numerically_verified" and band["counterexample"]
    assert band["value"]["certified"] >= 1 and band["value"]["unsound"] == 0
    assert "candidate hypothesis" in band["counterexample"]["statement"]
    assert band["counterexample"]["witness"]["exact_bin"] == "[-2, -1) res"
    assert _finding(report, "MARGIN_LOW appears exactly")["value"]["margin_low_observed"] is True
    assert _label(report, "Share of exactly") == "provider_backed"
    assert set(_finding(report, "Share of exactly")["value"]) == {"inconclusive_share"}


@pytest.mark.lab_task("T108")
@needs_provider
def test_t108_margin_monotonicity(reports):
    report = reports["T108"]
    _completed(report, "analytic")
    monotone = _finding(report, "Increasing required_margin")
    assert monotone["evidence_status"] == "numerically_verified"
    assert sum(monotone["value"][k] for k in ("passing_regained", "meets_regained", "noncertifying_code_changed",
                                               "inequality_changed")) == 0
    assert _finding(report, "The switch from CERTIFIED_WITH_MARGIN")["value"]["mismatches"] == 0
    assert set(_finding(report, "Negative and non-finite")["value"].values()) == {"raises ValueError"}
    assert _label(report, "Monotonicity of the verdict") == "analytic"


@pytest.mark.lab_task("T109")
@needs_provider
def test_t109_adversarial_eigenvalues(reports):
    report = reports["T109"]
    _completed(report, "numerically_verified")
    sound = _finding(report, "Every certifying PLSR verdict")
    assert sound["value"]["violations"] == 0 and sound["evidence_status"] == "independently_verified"
    agreement = _finding(report, "PLSR Lyapunov solutions agree")
    assert agreement["evidence_status"] == _solver_label()
    assert "within 10 n^2 u cond(P)" in agreement["claim"]  # the claim states the bound the check applies
    assert agreement["value"]["beyond_bound"] == 0 and agreement["value"]["cases"] >= 1
    assert agreement["regression_tolerance"] == {"abs": 0.0, "rel": 0.0}
    check = agreement["basis"].get("independent_check") or agreement["basis"]["checks"][0]
    assert check["reference_kind"] == "analytic" and check["observed"] <= 10.0
    numpy_misplaced = _finding(report, "numpy.linalg.eigvals misplaces")
    assert set(numpy_misplaced["value"]) == {"cases", "misplaced_beyond_1e3_eps"}
    assert "counterexample" not in numpy_misplaced
    assert _finding(report, "Certified non-normal plants")["value"]["max_ratio"] <= 1.0
    threshold = _finding(report, "With P = I the non-normal plants")
    assert threshold["value"]["mismatches"] == 0
    assert threshold["value"]["codes"]["non-normal K=2.82"] == "CERTIFIED_WITH_MARGIN"
    assert threshold["value"]["codes"]["non-normal K=2.83"] != "CERTIFIED_WITH_MARGIN"
    solver = _finding(report, "PLSR's solve_lyapunov returns only exactly valid")
    assert solver["evidence_status"] == "independently_verified"
    refused = _finding(report, "solve_lyapunov refuses an exactly Hurwitz plant")
    assert refused["evidence_status"] == "numerically_verified"
    assert "Jordan n=4, lambda=2^-6" in refused["value"]["plants"] and refused["counterexample"]
    assert _label(report, "numpy.linalg.eigvals misplaces") == "numerically_verified"


@pytest.mark.lab_task("T110")
@needs_provider
def test_t110_time_interpretation(reports):
    report = reports["T110"]
    _completed(report, "numerically_verified")
    own = _finding(report, "PLSR certifies each matrix")
    assert own["value"]["mismatches"] == 0 and own["evidence_status"] == "independently_verified"
    cross = _finding(report, "No Lyapunov P solved for one convention")
    assert cross["value"]["unsound"] == 0 and cross["evidence_status"] == "independently_verified"
    differing = _finding(report, "The two time interpretations")["value"]
    assert differing["differing"] == differing["off_quadrant_matrices"] == 20
    assert differing["pattern_mismatches"] == 0
    table = _finding(report, "Diagonal plants with P = I")["value"]
    assert table["diag(-1.5, -0.25) continuous"] == "CERTIFIED_WITH_MARGIN"
    assert table["diag(-1.5, -0.25) discrete"] == "NOT_CERTIFIED"
    assert _finding(report, "A discrete plant refuses")["value"]["code"] == "raises ValueError"


@pytest.mark.lab_task("T111")
@needs_provider
def test_t111_routes(reports):
    report = reports["T111"]
    _completed(report, "numerically_verified")
    for time in ("continuous", "discrete"):
        agreement = _finding(report, f"PLSR {time}-time Lyapunov solutions agree")
        assert agreement["evidence_status"] == _solver_label()
        assert agreement["value"]["max_relative_difference"] < 1e-9
        if "independent_check" in agreement["basis"]:
            checker = agreement["basis"]["independent_check"]["checker"]["implementation"]
            assert checker.startswith(f"scipy.linalg.solve_{time}_lyapunov")
            assert agreement["basis"]["independent_check"]["reference_kind"] == "analytic"
    gate = _finding(report, "PLSR's scalar NOT_CERTIFIED gate fires only where")
    assert gate["evidence_status"] == "independently_verified"
    assert set(gate["value"]["violations"].values()) == {0}
    assert gate["value"]["samples"] == {"route family": 400, "near threshold": 408}
    assert gate["value"]["not_certified_share"]["route family"] > 0.0
    assert gate["value"]["not_certified_share"]["near threshold"] > 0.0
    assert gate["value"]["agreement"]["route family"]["scalar_vs_exact_sample_sign"] == 1.0
    solver = _finding(report, "PLSR's solve_lyapunov returns a P exactly")
    assert solver["value"]["mismatches"] == 0 and solver["evidence_status"] == "independently_verified"
    assert solver["value"]["refused"] == len(solver["basis"]["checks"]) == 20
    verdicts = _finding(report, "PLSR's verdict certifies every numpy-stable plant")
    assert verdicts["value"] == {"stable_not_certified": 0, "unstable_certified_with_identity": 0}
    assert verdicts["evidence_status"] == "independently_verified"
    thin = _finding(report, "The scalar route sees decrease")
    assert thin["value"]["plsr_codes"] == {"DECREASE_NOT_DEFINITE": 64} and thin["counterexample"]
    assert thin["value"]["exactly_indefinite"] is True and thin["evidence_status"] == "numerically_verified"
    assert len(thin["basis"]["checks"]) == 4  # indefiniteness, scalar decrease, PLSR code, no certificate
    assert _label(report, "For n = 1 the PLSR verdict") == "numerically_verified"


@pytest.mark.lab_task("T114")
@needs_provider
def test_t114_level_set_and_monitor(reports):
    report = reports["T114"]
    _completed(report, "numerically_verified")
    monitor = _finding(report, "PLSR's monitor code on the declared configuration")
    assert monitor["evidence_status"] == "independently_verified"
    assert monitor["value"]["codes_without_level"] == ["CERTIFIED_WITH_MARGIN"]
    assert monitor["value"]["codes_with_level"] == ["CERTIFIED_WITH_MARGIN", "OUTSIDE_LEVEL_SET"]
    assert monitor["value"]["level_mismatches"] == 0
    assert _label(report, "PLSR produces every runtime code named as an abort trigger") == "numerically_verified"
    assert _label(report, "By the CIW transcription of the documented decision order, the monitor's code") \
        == "numerically_verified"
    assert _label(report, "The declared level set {V <= c} lies inside") == "numerically_verified"


@pytest.mark.lab_task("T113")
@needs_provider
def test_t113_residual_adapter(reports):
    report = reports["T113"]
    _completed(report, "numerically_verified")
    codes = _finding(report, "Forwarded samples receive")
    assert codes["evidence_status"] == "numerically_verified"
    assert set(codes["value"]["nominal theta 0.1"].values()) == {"CERTIFIED_WITH_MARGIN"}
    assert codes["value"]["estimate near bound theta 0.48"]["upper"] == "OUTSIDE_PARAMETER_BOX"
    assert codes["value"]["estimate outside box theta 0.9"]["estimate"] == "OUTSIDE_PARAMETER_BOX"
    accepted = _finding(report, "The host accepts a window only when")
    assert accepted["value"]["accepted"] == {"nominal theta 0.1": True, "nominal theta -0.3": True,
                                             "estimate near bound theta 0.48": False,
                                             "estimate outside box theta 0.9": False}
    assert accepted["evidence_status"] == "numerically_verified"
    assert set(_finding(report, "The kernel refuses every host-owned code")["value"].values()) == {"raises ValueError"}
    assert _finding(report, "Samples the adapter emits for the kernel carry only")["value"]["metadata_leaks"] == []
    assert _label(report, "The synthetic residual statistics") == "not_established"
    assert _label(report, "The EKF standard error of theta covers") == "not_established"
