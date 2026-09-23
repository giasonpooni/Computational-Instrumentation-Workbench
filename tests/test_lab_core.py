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
    saved = json.loads((tmp_path / "reports" / "T116.json").read_text())
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
