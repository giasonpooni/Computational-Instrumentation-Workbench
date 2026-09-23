import json

import pytest

from ciw.lab import research_portfolio, runner
from ciw.lab.evidence import DOMAINS, finding
from ciw.lab.registry import Implementation, load_implementations, load_queue
from ciw.lab.report import build_report, validate_report

CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _retain(directory, task_id, state, findings, **fields):
    built = build_report(_queue()[task_id], state, {"hypothesis": f"h {task_id}", "experiment": "e",
                                                    "unresolved_assumptions": [f"assumption of {task_id}"], **fields},
                         findings)
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(built), encoding="utf-8")
    return built


@pytest.fixture
def retained(tmp_path):
    counter = finding("Separation is not monotone past the focus", "numerical", -0.5,
                      {"generator": {"name": "g"}, "checks": [CHECK]}, uncertainty={"abs": 1e-9},
                      counterexample={"statement": "separation grows with length", "witness": {"s": 4.0}})
    _retain(tmp_path, "T010", "completed", [counter])
    _retain(tmp_path, "T116", "blocked", [finding("GPU energy per batch", "physical", None, {})],
            experiment="Blocked: unavailable requirement(s) hardware:nvidia-gpu.")
    return tmp_path


def _run(task_id, directory, junit=None):
    implementations, _ = load_implementations()
    ctx = runner.Context(directory)
    return validate_report(runner.run_task(_queue()[task_id], implementations[task_id], ctx, junit or {}))


def test_label_invariants_hold_exhaustively():
    result = research_portfolio.label_invariants()
    assert result["cases"] == 2 * 2 * 3 * 2 * 2 * 2 * len(DOMAINS)
    assert result["violations"] == []
    assert result["label_counts"]["not_established"] > 0 and "hardware_measured" in result["label_counts"]


def test_textbook_and_contribution_ledger(tmp_path):
    report = _run("T156", tmp_path)
    labels = {f["claim"]: f["evidence_status"] for f in report["findings"]}
    assert labels["Contributions are novel relative to the literature"] == "not_established"
    ledger = json.loads((tmp_path / "artifacts" / "T156" / "attribution-ledger.json").read_text())
    assert len(ledger["textbook"]) >= 10 and ledger["contributions"]


def test_catalogue_tasks_aggregate_retained_reports(retained):
    catalogue = _run("T157", retained)
    assert catalogue["findings"][0]["value"] == 1
    assert catalogue["findings"][0]["evidence_status"] == "numerically_verified"
    entries = json.loads((retained / "artifacts" / "T157" / "counterexamples.json").read_text())
    assert entries[0]["statement"] == "separation grows with length" and entries[0]["task_id"] == "T010"
    budget = _run("T159", retained)
    assert budget["findings"][0]["value"] == 1
    unmeasured = _run("T167", retained)
    document = json.loads((retained / "artifacts" / "T167" / "unmeasured.json").read_text())
    assert document["not_established_claims"][0]["task_id"] == "T116"
    assert document["blocked_or_deferred_tasks"][0]["state"] == "blocked"
    assert {f["evidence_status"] for f in unmeasured["findings"]} == {"numerically_verified", "not_established"}
    assert unmeasured["physical_validation_status"]["status"] == "not_established"
    ledger = _run("T166", retained)
    assert ledger["findings"][0]["value"] == 2
    release = _run("T165", retained)
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text())
    assert record["physical_validation"] == "not_established" and record["tasks"] == 2
    assert release["state"] == "completed"
    portfolio = _run("T163", retained)
    assert portfolio["state"] == "partial"


def test_aggregates_block_without_prior_reports(tmp_path):
    for task_id in ("T157", "T159", "T165", "T166", "T167"):
        assert _run(task_id, tmp_path)["state"] == "blocked"


def test_paper_drafts_trace_to_reports(retained):
    report = _run("T161", retained)
    assert report["state"] == "partial"
    text = (retained / "artifacts" / "T161" / "geometry-methods-draft.md").read_text()
    assert "Not peer reviewed" in text and "T010" in text
    labels = {f["claim"]: f["evidence_status"] for f in report["findings"]}
    assert labels["Draft has passed external peer review"] == "not_established"


def test_figures_are_reproducible(tmp_path, monkeypatch):
    from ciw.lab import svg

    def figure(ctx):
        ctx.artifact_text("plot.svg", svg.line_plot([("a", [1, 2, 3], [1, 4, 9])], title="t", xlabel="x", ylabel="y"))
        return {"fields": {}, "findings": [finding("f", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]})]}

    fake = Implementation("T010", figure)
    ctx = runner.Context(tmp_path)
    saved = runner.run_task(_queue()["T010"], fake, ctx, {})
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "T010.json").write_text(runner.dumps(saved))
    real, errors = load_implementations()
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, "T010": fake}, errors))
    report = _run("T158", tmp_path)
    values = {f["claim"]: f["value"] for f in report["findings"]}
    assert values["Regenerated figures are byte-identical to retained figures"] == 0
    assert report["state"] == "completed"


def test_clean_room_marker_is_recognized(tmp_path, monkeypatch):
    monkeypatch.delenv("CIW_LAB_CLEAN_ROOM", raising=False)
    assert _run("T164", tmp_path)["state"] == "partial"
    monkeypatch.setenv("CIW_LAB_CLEAN_ROOM", json.dumps({"wheel_sha256": "a" * 64, "python": "3.12"}))
    report = _run("T164", tmp_path)
    assert report["state"] == "completed" and report["findings"][0]["evidence_status"] == "numerically_verified"


def test_regression_coverage_is_checked(retained, monkeypatch):
    real, errors = load_implementations()
    covered = Implementation("T010", None, regression_tests=(
        "tests/test_lab_core.py::test_queue_has_168_ordered_tasks_in_eleven_sections",))
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, "T010": covered}, errors))
    report = _run("T168", retained)
    assert report["state"] == "completed"
    rows = json.loads((retained / "artifacts" / "T168" / "regression-coverage.json").read_text())
    assert {row["task_id"] for row in rows} == {"T010", "T116"} and all(not row["missing"] for row in rows)
    dangling = Implementation("T010", None, regression_tests=("tests/test_lab_core.py::test_does_not_exist",))
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, "T010": dangling}, errors))
    report = _run("T168", retained)
    assert report["state"] == "partial" and report["tests_failed"]
    uncovered = Implementation("T010", None)
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, "T010": uncovered}, errors))
    report = _run("T168", retained)
    values = {f["claim"]: f["value"] for f in report["findings"]}
    assert values["Completed or partial tasks lacking a regression test"] == 1 and report["state"] == "partial"
