import hashlib
import json

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
    assert runner.compare(tmp_path / "empty", tmp_path / "new")["passed"]


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
