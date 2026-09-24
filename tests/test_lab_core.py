import hashlib
import json
import os
import pathlib
import re
import sys

import pytest

from ciw.lab import evidence, report, runner
from ciw.lab.evidence import EvidenceRefusal, finding, supported_label, validate_finding
from ciw.lab.registry import load_queue

CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 1e-10, "tolerance": 1e-8, "passed": True}
PROVIDER = {"repository": "owner/provider", "revision": "a" * 40, "source_tree": "b" * 40, "executed": True}
ACQUISITION = {"device": "camera-1", "raw_sha256": "c" * 64, "acquired_at": "2026-09-23T00:00:00Z",
               "calibration": "not_applied"}


def test_queue_has_168_ordered_tasks_in_eleven_sections():
    queue = load_queue()
    assert len(queue["tasks"]) == 168
    assert [t["number"] for t in queue["tasks"]] == list(range(1, 169))
    assert len(queue["sections"]) == 11


@pytest.mark.parametrize("basis,domain,label", [
    ({}, "numerical", "not_established"),
    ({"derivation": "docs"}, "mathematical", "analytic"),
    ({"generator": {"name": "seeded"}}, "numerical", "synthetic"),
    ({"generator": {"name": "seeded"}, "checks": [CHECK]}, "numerical", "numerically_verified"),
    ({"provider": PROVIDER}, "numerical", "provider_backed"),
    ({"acquisition": ACQUISITION}, "physical", "hardware_measured"),
    ({"generator": {"name": "seeded"}, "checks": [CHECK]}, "physical", "not_established"),
    ({"provider": PROVIDER}, "sensor_performance", "not_established"),
    ({"acquisition": ACQUISITION, "checks": [CHECK]}, "machine_safety", "not_established"),
    ({"derivation": "docs"}, "actuator_authority", "not_established"),
])
def test_label_is_the_strongest_supported_by_the_basis(basis, domain, label):
    assert supported_label(basis, domain) == label


def test_failed_check_leaves_claim_not_established_even_with_generator():
    failed = dict(CHECK, observed=1.0, passed=False)
    assert supported_label({"generator": {"name": "g"}, "checks": [CHECK, failed]}, "numerical") == "not_established"


def test_check_passed_flag_must_match_numbers():
    with pytest.raises(EvidenceRefusal, match="passed does not match"):
        supported_label({"checks": [dict(CHECK, observed=1.0)]}, "numerical")


def test_independence_requires_distinct_implementations():
    check = dict(CHECK, producer={"implementation": "ciw.lab"}, checker={"implementation": "scipy.integrate"})
    assert supported_label({"independent_check": check}, "numerical") == "independently_verified"
    for sibling in ("ciw.lab", "ciw.lab.analytic", "ciw@other-revision", "CIW:reference"):
        same = dict(check, checker={"implementation": sibling, "revision": "other"})
        with pytest.raises(EvidenceRefusal, match="share an implementation origin"):
            supported_label({"independent_check": same}, "numerical")
    assert evidence.origin("Curved-Surface-Geodesic-Sensitivity-Runtime@bbc535a") != evidence.origin("ciw.lab")


def test_upgraded_or_hand_assigned_label_is_refused():
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}})
    assert record["evidence_status"] == "synthetic"
    for forged in ("numerically_verified", "hardware_measured", "independently_verified"):
        with pytest.raises(EvidenceRefusal, match="refused"):
            validate_finding(dict(record, evidence_status=forged))
    with pytest.raises(EvidenceRefusal, match="deterministic validator"):
        validate_finding(dict(record, assigned_by="assistant"))


def test_computational_claim_cannot_cite_hardware():
    with pytest.raises(EvidenceRefusal, match="cannot cite hardware"):
        supported_label({"acquisition": ACQUISITION}, "numerical")


def test_derived_physical_status_never_exceeds_weakest_input():
    measured = finding("m", "physical", 1.0, {"acquisition": ACQUISITION})
    simulated = finding("s", "physical", 1.0, {"generator": {"name": "g"}})
    assert evidence.physical_status([measured]) == "hardware_measured"
    assert evidence.physical_status([measured, simulated]) == "not_established"
    assert evidence.physical_status([]) == "not_established"


def test_report_answers_all_nineteen_questions_and_rejects_forged_status():
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    built = report.build_report(task, "completed", {"hypothesis": "h"}, [record])
    assert [name for name, _ in report.FIELDS] == list(report.FIELD_NAMES)
    assert len(report.FIELDS) == 19
    assert all(name in built for name in report.FIELD_NAMES)
    assert built["physical_validation_status"]["status"] == "not_established"
    report.validate_report(built)
    forged = json.loads(json.dumps(built))
    forged["physical_validation_status"]["status"] = "hardware_measured"
    forged["report_id"] = report.report_identity(forged)
    with pytest.raises(EvidenceRefusal, match="Physical validation status"):
        report.validate_report(forged)
    tampered = dict(built, hypothesis="changed")
    with pytest.raises(EvidenceRefusal, match="identity"):
        report.validate_report(tampered)
    with pytest.raises(EvidenceRefusal, match="derived"):
        report.build_report(task, "completed", {"evidence_status": "analytic"}, [record])


def test_completed_task_cannot_hide_an_unestablished_computational_finding():
    task = load_queue()["tasks"][0]
    unsupported = finding("claim", "numerical", 1.0, {})
    built = report.build_report(task, "completed", {}, [unsupported])
    with pytest.raises(EvidenceRefusal, match="unexpectedly unestablished"):
        report.validate_report(built)
    flagged = finding("claim", "numerical", 1.0, {}, expected_not_established=True)
    report.validate_report(report.build_report(task, "completed", {}, [flagged]))


def test_unimplemented_and_blocked_tasks_still_report(tmp_path):
    summary = runner.run_queue(tmp_path, task_ids=["T116"])
    assert summary["tasks"] == 1
    saved = json.loads((tmp_path / "reports" / "T116.json").read_text(encoding="utf-8"))
    assert saved["state"] in ("deferred", "blocked", "completed", "partial")
    assert saved["physical_validation_status"]["status"] == "not_established"
    report.validate_report(saved)


def test_regression_comparison_detects_value_and_label_changes(tmp_path):
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]},
                     tolerance={"abs": 1e-6, "rel": 0.0})
    for name, value in (("old", 4.0), ("new", 4.5)):
        directory = tmp_path / name / "reports"
        directory.mkdir(parents=True)
        built = report.build_report(task, "completed", {}, [dict(record, value=value)])
        (directory / "T001.json").write_text(runner.dumps(built))
    result = runner.compare(tmp_path / "old", tmp_path / "new")
    assert not result["passed"] and "regression tolerance" in result["problems"][0]
    assert runner.compare(tmp_path / "old", tmp_path / "old")["passed"]


def test_oversized_artifacts_are_refused(tmp_path):
    ctx = runner.Context(tmp_path)
    ctx.begin("T001")
    ctx.artifact_text("small.txt", "x")
    with pytest.raises(ValueError, match="exceeds"):
        ctx.artifact_text("large.txt", "x" * (runner.MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="single file names"):
        ctx.artifact_text("../escape.txt", "x")


def test_regression_comparison_flags_unretained_tasks_and_changed_findings(tmp_path):
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    extra = finding("extra", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    for name, findings in (("old", [record]), ("new", [record, extra])):
        directory = tmp_path / name / "reports"
        directory.mkdir(parents=True)
        (directory / "T001.json").write_text(runner.dumps(report.build_report(task, "completed", {}, findings)))
    second = load_queue()["tasks"][1]
    (tmp_path / "new" / "reports" / "T002.json").write_text(
        runner.dumps(report.build_report(second, "completed", {}, [record])))
    problems = runner.compare(tmp_path / "old", tmp_path / "new")["problems"]
    assert "T002: not retained" in problems
    assert any("finding claims differ" in p and "extra" in p for p in problems)
    assert any("finding count 1 -> 2" in p for p in problems)
    # Comparing against nothing verifies nothing: a missing or empty retained directory fails.
    (tmp_path / "empty" / "reports").mkdir(parents=True)
    for retained in (tmp_path / "empty", tmp_path / "missing"):
        result = runner.compare(retained, tmp_path / "new")
        assert not result["passed"] and "no retained reports" in result["problems"][0]
        assert {"T001: not retained", "T002: not retained"} <= set(result["problems"])


def test_repository_root_honours_the_clean_room_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path))
    assert runner.repository_path("examples", "x.json") == tmp_path / "examples" / "x.json"
    monkeypatch.delenv("CIW_LAB_REPOSITORY_ROOT")
    root = runner.repository_root()
    assert root is None or (root / "examples").is_dir()


def test_blocked_plan_can_record_unestablished_claims(tmp_path):
    from ciw.lab.registry import Implementation

    def run(ctx):
        raise AssertionError("a blocked task must not run")
    run.plan = {"hypothesis": "h", "experiment": "measure on hardware",
                "findings": [finding("GPU energy per batch", "physical", None, {})]}
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    built = runner.run_task(item, Implementation("T116", run, requires=("hardware:no-such-device",)),
                            runner.Context(tmp_path), {})
    assert built["state"] == "blocked" and built["findings"][0]["evidence_status"] == "not_established"


def test_cross_implementation_agreement_is_verified_not_independent():
    check = dict(CHECK, reference_kind="cross_implementation", reference="ciw Rust kernel vs ciw Python")
    assert supported_label({"checks": [check]}, "numerical") == "numerically_verified"


def test_section_implementations_loads_one_section():
    from ciw.lab.registry import section_implementations
    loaded = section_implementations("research-portfolio")
    assert set(loaded) == {f"T{n}" for n in range(155, 169)}
    with pytest.raises(ValueError, match="Unknown lab section"):
        section_implementations("no-such-section")


def test_signed_thresholds_are_allowed_for_directional_checks():
    ge = dict(CHECK, observed=-1e-12, tolerance=-1e-9, comparison="ge")
    assert supported_label({"checks": [ge]}, "numerical") == "numerically_verified"
    with pytest.raises(EvidenceRefusal, match="nonnegative for abs_le"):
        supported_label({"checks": [dict(CHECK, tolerance=-1.0, passed=False)]}, "numerical")


def test_label_changes_name_differing_optional_modules(tmp_path):
    task = load_queue()["tasks"][0]
    independent = dict(CHECK, producer={"implementation": "ciw.lab"}, checker={"implementation": "sympy"})
    runs = (("full", {"independent_check": independent}, {"implementation": "ciw.lab", "sympy": "1.14.0"}),
            ("bare", {"derivation": "docs"}, {"implementation": "ciw.lab"}))
    for name, basis, identity in runs:
        built = report.build_report(task, "completed", {"provider_runtime_identity": identity},
                                    [finding("series", "mathematical", 1.0, basis)])
        (tmp_path / name / "reports").mkdir(parents=True)
        (tmp_path / name / "reports" / "T001.json").write_text(runner.dumps(built))
    problems = runner.compare(tmp_path / "full", tmp_path / "bare")["problems"]
    assert problems == ["T001: 'series' label independently_verified -> analytic (optional modules differ: -sympy)"]


def test_module_implementations_returns_only_that_module():
    from ciw.lab.registry import module_implementations
    assert set(module_implementations("research_portfolio")) == {f"T{n}" for n in range(155, 169)}


@pytest.mark.parametrize("checker", ["ciw-rust", "python:ciw", "rust ciw", "https://github.com/x/ciw",
                                     "сiw.lab", "ciw​.lab", "ｃｉｗ.lab", "homemade.solver"])
def test_independence_cannot_be_minted_by_spelling(checker):
    check = dict(CHECK, producer={"implementation": "ciw.lab"}, checker={"implementation": checker})
    with pytest.raises(EvidenceRefusal):
        supported_label({"independent_check": check}, "numerical")


@pytest.mark.parametrize("checker", ["scipy.integrate.solve_ivp(DOP853)", "sympy", "git rev-parse HEAD^{tree}",
                                     "Curved-Surface-Geodesic-Sensitivity-Runtime@bbc535a", "cpython.math.fsum"])
def test_recognised_external_origins_are_independent(checker):
    check = dict(CHECK, producer={"implementation": "ciw.lab"}, checker={"implementation": checker})
    assert supported_label({"independent_check": check}, "numerical") == "independently_verified"
    same_origin = dict(check, reference_kind="cross_implementation")
    with pytest.raises(EvidenceRefusal, match="same-origin"):
        supported_label({"independent_check": same_origin}, "numerical")


def test_check_passed_flags_and_comparisons_are_strict():
    refusal = {"reference_kind": "refusal", "reference": "parser", "expected_refusal": "E_RANGE",
               "observed_refusal": "E_OTHER", "passed": True}
    with pytest.raises(EvidenceRefusal, match="expected and observed refusal"):
        supported_label({"checks": [refusal]}, "numerical")
    with pytest.raises(EvidenceRefusal, match="must be a string"):
        supported_label({"checks": [dict(refusal, observed_refusal=None, passed=False)]}, "numerical")
    with pytest.raises(EvidenceRefusal, match="nonnegative magnitude"):
        supported_label({"checks": [dict(CHECK, observed=-10.0, tolerance=1e-9, comparison="le", passed=True)]}, "numerical")
    signed = dict(CHECK, observed=-10.0, tolerance=1e-9, comparison="signed_le", passed=True)
    assert supported_label({"checks": [signed]}, "numerical") == "numerically_verified"
    with pytest.raises(EvidenceRefusal, match="vacuous"):
        supported_label({"checks": [dict(CHECK, observed=3.0, tolerance=-1e308, comparison="ge", passed=True)]}, "numerical")


def test_acquisition_needs_a_raw_digest_and_flags_are_strict():
    with pytest.raises(EvidenceRefusal, match="SHA-256"):
        supported_label({"acquisition": dict(ACQUISITION, raw_sha256="n/a")}, "physical")
    failed = dict(CHECK, observed=5.0, passed=False)
    with pytest.raises(EvidenceRefusal, match="supported or refuted"):
        finding("refuted", "numerical", 5.0, {"checks": [failed]}, expected_not_established=True)
    with pytest.raises(EvidenceRefusal, match="True or False"):
        finding("flag", "numerical", 1.0, {}, expected_not_established="false")
    record = finding("flag", "numerical", 1.0, {}, expected_not_established=True)
    with pytest.raises(EvidenceRefusal, match="exactly true"):
        validate_finding(dict(record, expected_not_established="false"))


def test_primary_label_is_order_independent_and_conservative():
    strong = finding("independent", "numerical", 1.0, {"independent_check": dict(
        CHECK, producer={"implementation": "ciw.lab"}, checker={"implementation": "sympy"})})
    checked = finding("checked", "numerical", 1.0, {"checks": [CHECK]})
    honest = finding("unsupported", "provenance", None, {}, expected_not_established=True)
    physical = finding("physical", "physical", None, {})
    for order in ([strong, checked, honest, physical], [physical, honest, checked, strong]):
        assert evidence.primary_label(order) == "numerically_verified"
    assert evidence.primary_label([honest, physical]) == "not_established"
    assert evidence.primary_label([checked, finding("refuted", "numerical", 1.0, {})]) == "not_established"


def test_reports_refuse_duplicates_edited_statements_and_established_blocked_findings():
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    with pytest.raises(EvidenceRefusal, match="unique"):
        report.validate_report(report.build_report(task, "completed", {}, [record, dict(record)]))
    built = report.build_report(task, "completed", {}, [record])
    edited = json.loads(json.dumps(built))
    edited["physical_validation_status"]["statement"] = "Validated on the production line."
    edited["report_id"] = report.report_identity(edited)
    with pytest.raises(EvidenceRefusal, match="derived statement"):
        report.validate_report(edited)
    with pytest.raises(EvidenceRefusal, match="blocked or deferred"):
        report.validate_report(report.build_report(task, "blocked", {}, [record]))


def _implementation(run, requires=()):
    from ciw.lab.registry import Implementation
    return Implementation("T116", run, requires=requires)


def test_invented_hardware_results_are_refused_and_real_ones_need_retained_bytes(tmp_path, monkeypatch):
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    raw = b"timestamp,energy_mj\n0,1\n"
    acquisition = dict(ACQUISITION, raw_sha256=hashlib.sha256(raw).hexdigest())

    def invented(ctx):
        return {"findings": [finding("GPU energy", "physical", 1.0, {"acquisition": acquisition})]}
    built = runner.run_task(item, _implementation(invented), runner.Context(tmp_path / "a"), {})
    assert built["state"] == "blocked" and "no hardware probe succeeded" in built["experiment"]

    monkeypatch.setattr(runner, "_probe_hardware", lambda name: name == "nvidia-gpu")

    def unretained(ctx):
        assert ctx.available("hardware:nvidia-gpu")
        return {"findings": [finding("GPU energy", "physical", 1.0, {"acquisition": acquisition})]}
    built = runner.run_task(item, _implementation(unretained), runner.Context(tmp_path / "b"), {})
    assert built["state"] == "blocked" and "not a retained artifact" in built["experiment"]

    def measured(ctx):
        assert ctx.available("hardware:nvidia-gpu")
        ctx.artifact_text("raw.csv", raw.decode())
        return {"findings": [finding("GPU energy", "physical", 1.0, {"acquisition": acquisition})]}
    built = runner.run_task(item, _implementation(measured), runner.Context(tmp_path / "c"), {})
    assert built["state"] == "completed" and built["physical_validation_status"]["status"] == "hardware_measured"


def test_contract_violations_become_blocked_reports_and_the_queue_continues(tmp_path):
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]

    def careless(ctx):
        return {"state": "completed", "findings": [finding("unsupported", "numerical", 1.0, {})]}
    built = runner.run_task(item, _implementation(careless), runner.Context(tmp_path), {})
    assert built["state"] == "blocked" and "evidence contract" in built["experiment"]

    def run(ctx):
        raise AssertionError("never runs")
    run.plan = {"findings": [finding("claimed", "numerical", 1.0, {"checks": [CHECK]})]}
    built = runner.run_task(item, _implementation(run, ("hardware:no-such-device",)), runner.Context(tmp_path), {})
    assert built["state"] == "blocked" and not built["findings"] and "Plan findings refused" in built["experiment"]


def test_verification_catches_wording_units_and_artifact_edits(tmp_path):
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]}, unit="rad",
                     tolerance={"abs": 1e-6, "rel": 0.0})
    for name, fields, unit in (("old", {"numerical_result": "rate 4.000 rad"}, "rad"),
                               ("digits", {"numerical_result": "rate 4.001 rad"}, "rad"),
                               ("words", {"numerical_result": "rate 4.000 rad, validated on hardware"}, "rad"),
                               ("units", {"numerical_result": "rate 4.000 rad"}, "mm")):
        directory = tmp_path / name
        (directory / "reports").mkdir(parents=True)
        ctx = runner.Context(directory)
        ctx.begin("T001")
        ctx.artifact_text("table.csv", "a,b\n")
        built = report.build_report(task, "completed", dict(fields, generated_artifacts=ctx.artifacts),
                                    [dict(record, unit=unit)])
        (directory / "reports" / "T001.json").write_text(runner.dumps(built))
    assert runner.compare(tmp_path / "old", tmp_path / "digits")["passed"]
    assert any("wording differs" in p for p in runner.compare(tmp_path / "old", tmp_path / "words")["problems"])
    assert any("unit or domain" in p for p in runner.compare(tmp_path / "old", tmp_path / "units")["problems"])
    (tmp_path / "old" / "artifacts" / "T001" / "table.csv").write_text("edited\n")
    assert any("recorded digest" in p for p in runner.compare(tmp_path / "old", tmp_path / "digits")["problems"])


def _retain(directory, findings, task_index=0, **fields):
    task = load_queue()["tasks"][task_index]
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    built = report.build_report(task, "completed", fields, findings)
    (directory / "reports" / f"{task['id']}.json").write_text(runner.dumps(built))
    return built


def test_verification_of_a_task_subset_needs_each_task_on_both_sides(tmp_path):
    record = finding("rate", "numerical", 4.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    for index in (0, 1):
        _retain(tmp_path / "retained", [record], index)
    _retain(tmp_path / "fresh", [record], 1)
    assert runner.compare(tmp_path / "retained", tmp_path / "fresh")["problems"] == ["T001: not regenerated"]
    subset = runner.compare(tmp_path / "retained", tmp_path / "fresh", tasks=["T002"])
    assert subset == {"compared": 1, "problems": [], "passed": True}
    problems = runner.compare(tmp_path / "retained", tmp_path / "fresh", tasks=["T002", "T003"])["problems"]
    assert problems == ["T003: not retained", "T003: not regenerated"]
    _retain(tmp_path / "fresh", [dict(record, value=5.0)], 0)
    assert runner.compare(tmp_path / "retained", tmp_path / "fresh", tasks=["T002"])["passed"]
    assert not runner.compare(tmp_path / "retained", tmp_path / "fresh", tasks=["T001"])["passed"]
    # An empty selection compares nothing, so it verifies nothing, even where the directories disagree.
    for empty in ([], ()):
        assert runner.compare(tmp_path / "retained", tmp_path / "fresh", tasks=empty) == {
            "compared": 0, "problems": ["no tasks selected: nothing to verify"], "passed": False}


def test_verification_distinguishes_booleans_counterexamples_and_flags(tmp_path):
    def differs(old, new):
        _retain(tmp_path / "old", [old])
        _retain(tmp_path / "new", [new])
        return runner.compare(tmp_path / "old", tmp_path / "new")["problems"]
    checked = {"checks": [CHECK]}
    assert differs(finding("flag", "numerical", True, checked), finding("flag", "numerical", 1, checked)) == [
        "T001: 'flag' value outside regression tolerance"]
    refuted = {"statement": "separation grows with length", "witness": {"s": 4.0}}
    assert differs(finding("c", "numerical", 1.0, checked, counterexample=refuted),
                   finding("c", "numerical", 1.0, checked, counterexample=dict(refuted, statement="it shrinks"))) == [
        "T001: 'c' counterexample differs"]
    assert differs(finding("c", "numerical", 1.0, checked, counterexample=refuted),
                   finding("c", "numerical", 1.0, checked, counterexample=dict(refuted, witness={"s": 9.0}))) == [
        "T001: 'c' counterexample differs"]
    assert differs(finding("c", "numerical", 1.0, checked, counterexample=refuted),
                   finding("c", "numerical", 1.0, checked)) == ["T001: 'c' counterexample differs"]
    unmeasured = finding("u", "physical", None, {})
    assert differs(unmeasured, dict(unmeasured, expected_not_established=True)) == [
        "T001: 'u' expected_not_established differs"]


def _junit(path, *cases):
    body = "".join(f'<testcase classname="{classname}" name="{name}">{child}</testcase>'
                   for classname, name, child in cases)
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>", encoding="utf-8")
    return path


def test_junit_outcomes_fold_by_severity_and_map_class_node_ids(tmp_path):
    from ciw.lab.registry import Implementation
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("")
    junit = runner.read_junit(_junit(tmp_path / "junit.xml",
                                     ("tests.test_demo", "test_p[1]", "<skipped/>"),
                                     ("tests.test_demo", "test_p[2]", "<failure/>"),
                                     ("tests.test_demo", "test_q[1]", "<skipped/>"),
                                     ("tests.test_demo", "test_q[2]", ""),
                                     ("tests.test_demo.TestC", "test_m", "<failure/>"),
                                     ("tests.old_test_demo", "test_r", "<failure/>")), root=tmp_path)
    assert junit["tests/test_demo.py::TestC::test_m"] == "failed"
    nodes = ("tests/test_demo.py::test_p", "tests/test_demo.py::test_q", "tests/test_demo.py::TestC::test_m",
             "test_demo.py::test_r")
    passed, skipped, failed = runner._test_fields(Implementation("T116", None, regression_tests=nodes), [], junit)
    assert failed == ["pytest: tests/test_demo.py::test_p", "pytest: tests/test_demo.py::TestC::test_m"]
    assert passed == ["pytest: tests/test_demo.py::test_q"]
    # A suffix of another file's node id is not the registered test.
    assert skipped == ["pytest: test_demo.py::test_r (not run in this invocation)"]
    # Without the sources, classes start at pytest's Test prefix.
    assert runner._node_id("tests.test_demo.TestC", "test_m", None) == "tests/test_demo.py::TestC::test_m"
    assert runner._node_id("tests.test_demo", "test_m", tmp_path / "elsewhere") == "tests/test_demo.py::test_m"


def test_unsupported_requirements_are_refused_and_never_abort_the_queue(tmp_path):
    from ciw.lab.registry import task
    with pytest.raises(ValueError, match="Unsupported lab requirement"):
        task("T999", requires=("modules:scipy",))

    def run(ctx):
        raise AssertionError("an unprobeable requirement is never met")
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    built = runner.run_task(item, _implementation(run, ("modules:scipy",)), runner.Context(tmp_path), {})
    assert built["state"] == "blocked" and "modules:scipy" in built["experiment"]


def test_blocked_plan_checks_are_not_reported_as_tests(tmp_path):
    def run(ctx):
        raise AssertionError("never runs")
    run.plan = {"findings": [finding("GPU agrees with CPU", "physical", 1.0, {"generator": {"name": "g"},
                                                                          "checks": [CHECK]})]}
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    built = runner.run_task(item, _implementation(run, ("hardware:no-such-device",)), runner.Context(tmp_path), {})
    assert built["state"] == "blocked" and built["findings"][0]["evidence_status"] == "not_established"
    assert built["tests_passed"] == [] and "tests_failed" not in built


def test_executed_task_without_answers_says_they_are_not_stated(tmp_path):
    def terse(ctx):
        return {"findings": [finding("rate", "numerical", 4.0, {"checks": [CHECK]})]}
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    built = runner.run_task(item, _implementation(terse), runner.Context(tmp_path), {})
    assert built["state"] == "completed"
    for name in ("hypothesis", "experiment", "observation_model", "recommended_next_task"):
        assert built[name] == runner.NOT_STATED
    assert built["unresolved_assumptions"] == [runner.NOT_STATED]
    assert "not been executed" not in json.dumps(built)


def test_rewritten_artifact_keeps_one_entry_with_the_bytes_on_disk(tmp_path):
    ctx = runner.Context(tmp_path)
    ctx.begin("T001")
    ctx.artifact_text("t.json", "first")
    ctx.artifact_text("other.txt", "x")
    ctx.artifact_text("t.json", "second")
    entries = [a for a in ctx.artifacts if a["path"] == "artifacts/T001/t.json"]
    assert len(ctx.artifacts) == 2 and len(entries) == 1
    assert entries[0]["sha256"] == hashlib.sha256((tmp_path / "artifacts" / "T001" / "t.json").read_bytes()).hexdigest()


def test_completed_report_cannot_record_failed_tests():
    task = load_queue()["tasks"][0]
    record = finding("rate", "numerical", 4.0, {"checks": [CHECK]})
    failed = {"tests_failed": ["pytest: tests/test_lab_core.py::test_x"]}
    with pytest.raises(EvidenceRefusal, match="failed tests"):
        report.validate_report(report.build_report(task, "completed", {}, [record], extra=failed))
    report.validate_report(report.build_report(task, "partial", {}, [record], extra=failed))


def test_untracked_files_make_a_provider_checkout_dirty(tmp_path):
    import shutil
    import subprocess
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=lab@example.invalid", "-c", "user.name=lab"]
    subprocess.run(git + ["init", "-q"], check=True)
    (tmp_path / "engine.py").write_text("VALUE = 1\n")
    subprocess.run(git + ["add", "engine.py"], check=True)
    subprocess.run(git + ["commit", "-qm", "pinned"], check=True)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "engine.cpython-311.pyc").write_bytes(b"cache")
    assert runner.git_identity(tmp_path)["dirty"] is False
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "__init__.py").write_text("VALUE = 999\n")  # shadows engine.py on import
    assert runner.git_identity(tmp_path)["dirty"] is True


@pytest.mark.parametrize("hide", ["pinned_gitignore", "info_exclude", "skip_worktree", "assume_unchanged"])
def test_code_git_is_told_to_overlook_makes_a_provider_checkout_dirty(tmp_path, hide):
    import shutil
    import subprocess
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=lab@example.invalid", "-c", "user.name=lab",
           "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={tmp_path / 'no-hooks'}"]
    subprocess.run(git + ["init", "-q"], check=True)
    (tmp_path / "engine.py").write_text("VALUE = 1\n")
    (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")  # as pinned: ignores bytecode
    subprocess.run(git + ["add", "engine.py", ".gitignore"], check=True)
    subprocess.run(git + ["commit", "-qm", "pinned"], check=True)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "engine.cpython-311.pyc").write_bytes(b"cache")
    assert runner.git_identity(tmp_path)["dirty"] is False
    if hide == "pinned_gitignore":  # a sourceless package the pinned .gitignore hides shadows engine.py
        (tmp_path / "engine").mkdir()
        (tmp_path / "engine" / "__init__.pyc").write_bytes(b"shadow")
    elif hide == "info_exclude":  # a local, untracked exclude hides a shadowing package
        (tmp_path / ".git" / "info").mkdir(exist_ok=True)
        (tmp_path / ".git" / "info" / "exclude").write_text("engine/\n")
        (tmp_path / "engine").mkdir()
        (tmp_path / "engine" / "__init__.py").write_text("VALUE = 999\n")
    else:  # git status never compares the bytes of a tracked file flagged this way
        flag = "--skip-worktree" if hide == "skip_worktree" else "--assume-unchanged"
        subprocess.run(git + ["update-index", flag, "engine.py"], check=True)
        (tmp_path / "engine.py").write_text("VALUE = 999\n")
    assert runner.git_identity(tmp_path)["dirty"] is True


@pytest.mark.skipif(os.name == "nt", reason="powercap zone names contain ':'")
def test_rapl_probe_needs_a_readable_counter_not_a_present_one(tmp_path, monkeypatch):
    zone = tmp_path / "intel-rapl:0"
    zone.mkdir()
    (zone / "energy_uj").write_text("123456\n", encoding="utf-8")
    monkeypatch.setattr(runner, "POWERCAP", tmp_path)
    assert runner._probe_hardware("rapl") is True
    read_text = pathlib.Path.read_text

    def root_only(path, *args, **kwargs):  # the counter exists, but only root may read it
        if path.name == "energy_uj":
            raise PermissionError(13, "Permission denied", str(path))
        return read_text(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "read_text", root_only)
        assert runner._probe_hardware("rapl") is False
        ctx = runner.Context(tmp_path / "out")
        ctx.begin("T115")
        assert ctx.available("hardware:rapl") is False and not ctx.hardware
    (zone / "energy_uj").write_text("", encoding="utf-8")
    assert runner._probe_hardware("rapl") is False


def test_builtin_identity_records_optional_modules_that_fail_to_identify(tmp_path, monkeypatch):
    (tmp_path / "scipy.py").write_text("raise ImportError('compiled against another numpy')\n", encoding="utf-8")
    (tmp_path / "sympy.py").write_text("", encoding="utf-8")  # importable, states no __version__
    (tmp_path / "mpmath.py").write_text("__version__ = '9.9.9'\n", encoding="utf-8")
    names = ("scipy", "sympy", "mpmath")
    for name in names:  # start without them, as a session that never imported them (the CI test job has none)
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    with monkeypatch.context() as patch:
        for name in names:
            patch.setitem(sys.modules, name, None)  # records the entry, or its absence, for the undo
            patch.delitem(sys.modules, name)
        patch.syspath_prepend(str(tmp_path))
        identity = runner.builtin_identity([])
    assert not set(names) & set(sys.modules)  # the fakes never outlive the test
    assert identity["mpmath"] == "9.9.9"
    assert identity["sympy"] == "unknown (AttributeError: module 'sympy' has no attribute '__version__')"
    assert "scipy" not in identity  # a module that fails on import is not present
    assert identity["optional_module_errors"] == {"scipy": "ImportError: compiled against another numpy"}


def _probing(ctx):
    cargo, scr = ctx.available("tool:cargo"), ctx.available("provider:scr")
    basis = {"checks": [CHECK]} if cargo and scr else {"derivation": "docs"}
    return {"state": "completed" if cargo else "partial", "findings": [finding("kernel", "numerical", 1.0, basis)]}


def test_reports_record_requirement_probes_and_comparison_names_their_differences(tmp_path, monkeypatch):
    import shutil
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    checkout = tmp_path / "scr-checkout"
    checkout.mkdir()
    which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/opt/cargo" if name == "cargo" else which(name, *a, **k))
    built = {}
    for name, providers in (("with", {"scr": checkout}), ("again", {"scr": checkout}), ("unbound", {})):
        ctx = runner.Context(tmp_path / name, providers)
        built[name] = runner.run_task(item, _implementation(_probing), ctx, {})
    assert built["with"]["provider_runtime_identity"]["requirement_probes"] == {"provider:scr": True, "tool:cargo": True}
    assert built["with"]["report_id"] == built["again"]["report_id"]  # deterministic in one environment
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "cargo" else which(name, *a, **k))
    built["no-cargo"] = runner.run_task(item, _implementation(_probing), runner.Context(tmp_path / "x", {"scr": checkout}), {})
    blocked = runner.run_task(item, _implementation(_probing, ("tool:cargo",)), runner.Context(tmp_path / "y"), {})
    assert blocked["state"] == "blocked" and blocked["provider_runtime_identity"]["requirement_probes"] == {
        "tool:cargo": False}
    # A task that probes nothing keeps the plain built-in identity.
    quiet = runner.run_task(item, _implementation(lambda ctx: {"findings": [finding("k", "numerical", 1.0, {
        "checks": [CHECK]})]}), runner.Context(tmp_path / "z"), {})
    assert "requirement_probes" not in quiet["provider_runtime_identity"]
    for name, value in built.items():
        (tmp_path / name / "reports").mkdir(parents=True, exist_ok=True)
        (tmp_path / name / "reports" / "T116.json").write_text(runner.dumps(value))
    assert runner.compare(tmp_path / "with", tmp_path / "again")["passed"]
    assert runner.compare(tmp_path / "with", tmp_path / "unbound")["problems"] == [
        "T116: 'kernel' label numerically_verified -> analytic (requirement probes differ: provider:scr available -> "
        "unavailable)"]
    assert runner.compare(tmp_path / "with", tmp_path / "no-cargo")["problems"] == [
        "T116: state completed -> partial (requirement probes differ: tool:cargo available -> unavailable)",
        "T116: 'kernel' label numerically_verified -> analytic (requirement probes differ: tool:cargo available -> "
        "unavailable)"]


def test_tasks_sharing_a_memo_record_the_probes_its_computation_made(tmp_path, monkeypatch):
    item = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    monkeypatch.setattr(runner, "_probe_hardware", lambda name: True)
    computed = []

    def reference(ctx):
        computed.append(ctx.task_id)
        return 1.0 if ctx.available("module:json") and ctx.available("hardware:nvidia-gpu") else 0.0

    def shared(ctx):
        value = ctx.memo("shared", lambda: reference(ctx))
        return {"findings": [finding("k", "numerical", value, {"checks": [CHECK]})]}

    def outer(ctx):  # a memo computation that reuses another memo records its probes too
        value = ctx.memo("outer", lambda: ctx.memo("shared", lambda: reference(ctx)) + 1.0)
        return {"findings": [finding("k", "numerical", value, {"checks": [CHECK]})]}
    ctx = runner.Context(tmp_path / "run")
    first, second = (runner.run_task(item, _implementation(shared), ctx, {}) for _ in range(2))
    assert computed == ["T116"] and not ctx.hardware  # a replayed hardware probe is not the task's own
    alone = runner.run_task(item, _implementation(shared), runner.Context(tmp_path / "alone"), {})
    probes = {"hardware:nvidia-gpu": True, "module:json": True}
    assert first["provider_runtime_identity"]["requirement_probes"] == probes
    assert second["report_id"] == alone["report_id"] == first["report_id"]
    nested, again = (runner.run_task(item, _implementation(outer), ctx, {}) for _ in range(2))
    assert computed == ["T116", "T116"] and nested["report_id"] == again["report_id"]  # "alone" computed once more
    assert again["provider_runtime_identity"]["requirement_probes"] == probes


def test_a_self_blocked_task_that_probed_is_not_proposed_again(tmp_path):
    import dataclasses
    from ciw.lab import planner
    from ciw.lab.registry import load_implementations
    registered = load_implementations()[0]["T098"]
    assert registered.requires == ()  # blocked without a declared requirement
    item = {t["id"]: t for t in load_queue()["tasks"]}["T098"]

    def self_blocked(ctx):
        ctx.available("tool:git")
        return {"state": "blocked", "findings": []}

    def refused(ctx):
        ctx.available("tool:git")
        return {"state": "completed", "findings": []}  # the contract refuses it; it is retained as blocked
    (tmp_path / "reports").mkdir()
    for run in (self_blocked, refused):
        built = runner.run_task(item, dataclasses.replace(registered, run=run), runner.Context(tmp_path / "work"), {})
        assert built["state"] == "blocked"
        (tmp_path / "reports" / "T098.json").write_text(runner.dumps(built))
        plan = planner.next_tasks(tmp_path, limit=200)
        # Re-running it unchanged reproduces the same report, so the plan must not propose it.
        assert "T098" not in [row["task_id"] for row in plan["next"]]
        assert "T098" in [row["task_id"] for row in plan["still_blocked"]]


def test_physical_statement_counts_findings_on_acquired_hardware():
    none = "Physical validation requires acquired hardware evidence; none was acquired for this task."
    task = {t["id"]: t for t in load_queue()["tasks"]}["T116"]
    energy = finding("GPU energy", "physical", 1.0, {"acquisition": ACQUISITION})
    power = finding("GPU power", "physical", 2.0, {"acquisition": ACQUISITION})
    accuracy = finding("NVML accuracy", "physical", None, {})
    checked = finding("rate", "numerical", 4.0, {"checks": [CHECK]})
    cases = (([checked], "not_established", none),
             ([checked, accuracy], "not_established", none),
             ([energy, accuracy], "not_established",
              "1 of 2 physical-domain findings rests on acquired hardware evidence; the rest are not_established."),
             ([energy, power, accuracy], "not_established",
              "2 of 3 physical-domain findings rest on acquired hardware evidence; the rest are not_established."),
             ([energy, power, checked], "hardware_measured",
              "2 of 2 physical-domain findings rest on acquired hardware evidence."))
    for findings, status, statement in cases:
        built = report.validate_report(report.build_report(task, "completed", {}, findings))
        assert built["physical_validation_status"] == {"status": status, "statement": statement}
    # A GPU-host report that measured one of its two claims may neither deny it nor claim validation.
    partly = report.build_report(task, "completed", {}, [energy, accuracy])
    for forged in ({"status": "not_established", "statement": none},
                   {"status": "not_established", "statement": cases[4][2]},
                   {"status": "hardware_measured", "statement": cases[2][2]}):
        edited = dict(json.loads(json.dumps(partly)), physical_validation_status=forged)
        edited["report_id"] = report.report_identity(edited)
        with pytest.raises(EvidenceRefusal, match="derived statement: 1 of 2"):
            report.validate_report(edited)


def test_independence_is_symmetric_between_distinct_known_origins():
    for producer, checker in (("ciw.lab.jacobi", "scipy"), ("scipy.integrate", "ciw.lab.jacobi"),
                              ("parameterized-lyapunov-stability-runtime", "ciw.lab.lyapunov_reference"),
                              ("sympy", "mpmath")):
        check = dict(CHECK, producer={"implementation": producer}, checker={"implementation": checker})
        assert supported_label({"independent_check": check}, "numerical") == "independently_verified"
    for producer, checker in (("ciw.lab.a", "ciw.lab.b"), ("scipy.linalg", "scipy.integrate"), ("homemade", "ciw")):
        check = dict(CHECK, producer={"implementation": producer}, checker={"implementation": checker})
        with pytest.raises(EvidenceRefusal):
            supported_label({"independent_check": check}, "numerical")


def test_markdown_rows_keep_the_label_column_for_any_claim_text():
    task = load_queue()["tasks"][0]
    record = finding("a | b\nsplit claim", "numerical", 1.0, {"checks": [CHECK]}, unit="m|s")
    lines = report.render_markdown(report.build_report(task, "completed", {}, [record])).splitlines()
    table = lines[lines.index("| Finding | Value | Evidence status |"):]
    assert len(table) == 3  # header, separator and exactly one row: the newline did not split it
    row = table[2]
    assert "a \\| b split claim" in row and "m\\|s" in row
    assert len(re.findall(r"(?<!\\)\|", row)) == 4 and row.endswith("| `numerically_verified` |")
