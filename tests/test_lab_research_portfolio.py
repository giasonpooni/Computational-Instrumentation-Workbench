import json
from pathlib import Path
import platform

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
    ledger = json.loads((tmp_path / "artifacts" / "T156" / "attribution-ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["textbook"]) >= 10 and ledger["contributions"]


def test_catalogue_tasks_aggregate_retained_reports(retained):
    catalogue = _run("T157", retained)
    assert catalogue["findings"][0]["value"] == 1
    assert catalogue["findings"][0]["evidence_status"] == "numerically_verified"
    entries = json.loads((retained / "artifacts" / "T157" / "counterexamples.json").read_text(encoding="utf-8"))
    assert entries[0]["statement"] == "separation grows with length" and entries[0]["task_id"] == "T010"
    budget = _run("T159", retained)
    assert budget["findings"][0]["value"] == 1
    unmeasured = _run("T167", retained)
    document = json.loads((retained / "artifacts" / "T167" / "unmeasured.json").read_text(encoding="utf-8"))
    assert document["not_established_claims"][0]["task_id"] == "T116"
    assert document["blocked_or_deferred_tasks"][0]["state"] == "blocked"
    assert {f["evidence_status"] for f in unmeasured["findings"]} == {"numerically_verified", "not_established"}
    assert unmeasured["physical_validation_status"]["status"] == "not_established"
    ledger = _run("T166", retained)
    assert ledger["findings"][0]["value"] == 2
    release = _run("T165", retained)
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
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
    text = (retained / "artifacts" / "T161" / "geometry-methods-draft.md").read_text(encoding="utf-8")
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
    monkeypatch.setenv("CIW_LAB_CLEAN_ROOM", json.dumps({"wheel_sha256": "a" * 64, "python": "2.7.18"}))
    report = _run("T164", tmp_path)
    # An asserted marker without a matching wheel and isolated install is not evidence.
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    # The interpreter is observed, never copied from the marker.
    assert report["provider_runtime_identity"]["python"] == platform.python_version()


def _wheel(path, package_dir, drop=(), extra=None, record=True):
    """A wheel-shaped zip of ``package_dir`` (bytecode caches aside), optionally altered."""
    import zipfile
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(package_dir.rglob("*")):
            name = file.relative_to(package_dir.parent).as_posix()
            if file.is_file() and "__pycache__" not in file.parts and name not in drop:
                archive.writestr(name, file.read_bytes())
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
        if record:
            archive.writestr("ciw-0.dist-info/RECORD", "")
    return path


def _clean_room(monkeypatch, wheel_path):
    import hashlib
    import sys
    import ciw
    # An isolated interpreter whose prefix holds the imported package, as in the clean room.
    monkeypatch.setattr(sys, "prefix", str(Path(ciw.__file__).resolve().parents[1]))
    monkeypatch.setattr(sys, "base_prefix", "/nonexistent-base-interpreter")
    monkeypatch.setenv("CIW_LAB_CLEAN_ROOM", json.dumps({
        "wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(), "wheel_path": str(wheel_path)}))


def test_clean_room_needs_the_installed_package_to_be_the_named_wheel(tmp_path, monkeypatch):
    import ciw
    package = Path(ciw.__file__).resolve().parent
    # A forged marker names any file with its own digest; nothing ties it to the installed code.
    notes = tmp_path / "notes.txt"
    notes.write_text("not a wheel at all\n")
    _clean_room(monkeypatch, notes)
    report = _run("T164", tmp_path / "forged")
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    assert "'installed_from_wheel': False" in report["numerical_result"]
    # The wheel the installed package came from: every file present with the same bytes.
    _clean_room(monkeypatch, _wheel(tmp_path / "ciw-0-py3-none-any.whl", package))
    report = _run("T164", tmp_path / "genuine")
    assert report["state"] == "completed" and report["findings"][0]["evidence_status"] == "numerically_verified"


def test_installed_package_is_compared_file_by_file_with_the_wheel(tmp_path):
    package = tmp_path / "site" / "ciw"
    (package / "lab" / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    (package / "lab" / "task.py").write_bytes(b"def run():\n    pass\n")
    (package / "lab" / "__pycache__" / "task.cpython-311.pyc").write_bytes(b"bytecode")
    matches = research_portfolio._installed_from
    assert matches(_wheel(tmp_path / "same.whl", package).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "no-record.whl", package, record=False).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "missing.whl", package, drop=("ciw/lab/task.py",)).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "extra.whl", package, extra={"ciw/lab/more.py": b""}).read_bytes(), package)
    assert not matches(b"not a zip", package)
    wheel = _wheel(tmp_path / "before.whl", package).read_bytes()
    (package / "lab" / "task.py").write_bytes(b"def run():\n    return 999\n")  # edited after installation
    assert not matches(wheel, package)


def test_regression_coverage_is_checked(retained, monkeypatch):
    existing = "tests/test_lab_core.py::test_queue_has_168_ordered_tasks_in_eleven_sections"

    def registry(t010):
        # Only the fixture's tasks, so real section registrations cannot leak in.
        fakes = {"T010": t010, "T116": Implementation("T116", None, regression_tests=(existing,))}
        monkeypatch.setattr(research_portfolio, "load_implementations", lambda: (fakes, {}))

    registry(Implementation("T010", None, regression_tests=(existing,)))
    report = _run("T168", retained)
    assert report["state"] == "completed"
    rows = json.loads((retained / "artifacts" / "T168" / "regression-coverage.json").read_text(encoding="utf-8"))
    assert {row["task_id"] for row in rows} == {"T010", "T116"} and all(not row["missing"] for row in rows)
    registry(Implementation("T010", None, regression_tests=("tests/test_lab_core.py::test_does_not_exist",)))
    report = _run("T168", retained)
    assert report["state"] == "partial" and report["tests_failed"]
    registry(Implementation("T010", None))
    report = _run("T168", retained)
    values = {f["claim"]: f["value"] for f in report["findings"]}
    assert values["Completed or partial tasks lacking a regression test"] == 1 and report["state"] == "partial"


def test_regression_node_ids_resolve_class_methods_and_async_tests(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_demo.py").write_text(
        "def test_plain():\n    pass\n\n"
        "async def test_async():\n    pass\n\n"
        "class TestGroup:\n    def test_method(self):\n        pass\n\n"
        "    class TestNested:\n        async def test_inner(self):\n            pass\n\n"
        "def helper():\n    pass\n", encoding="utf-8")
    assert research_portfolio._test_names(tests) == {
        "tests/test_demo.py::test_plain", "tests/test_demo.py::test_async",
        "tests/test_demo.py::TestGroup::test_method", "tests/test_demo.py::TestGroup::TestNested::test_inner"}


def test_regression_coverage_never_reads_the_current_directory(retained, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tests").mkdir(parents=True)
    (elsewhere / "tests" / "test_unrelated.py").write_text("def test_other():\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path / "installed-without-tests"))
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({"T010": Implementation(
        "T010", None, regression_tests=("tests/test_unrelated.py::test_other",))}, {}))
    report = _run("T168", retained)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert [f["evidence_status"] for f in report["findings"]] == ["not_established"]
