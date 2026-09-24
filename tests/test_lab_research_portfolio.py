import json
import math
from pathlib import Path
import platform

import pytest

from ciw.lab import research_portfolio, runner
from ciw.lab.evidence import DOMAINS, EvidenceRefusal, finding
from ciw.lab.registry import Implementation, load_implementations, load_queue
from ciw.lab.report import build_report, validate_report

SECTION = ("T155", "T156", "T157", "T158", "T159", "T160", "T161", "T162", "T163", "T164", "T165", "T166",
           "T167", "T168")
CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}
PIPE_CLAIM = "Unit-speed drift max|g(v,v) - 1| stays below the bound"
TINY = 1.1122324405657753e-10
ROUNDOFF = 1.7763568394002505e-15
ORACLE_CLAIM = ("The evidence-label function agrees with the reference oracle for rules 1-5 on the exhaustive basis "
                "grammar")


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _retain(directory, task_id, state, findings, extra=None, **fields):
    answers = {"hypothesis": f"h {task_id}", "experiment": "e", "unresolved_assumptions": [f"assumption of {task_id}"],
               "changed_files": ["src/ciw/lab/geodesic_jacobi.py"], **fields}
    built = build_report(_queue()[task_id], state, answers, findings, extra=extra)
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(built), encoding="utf-8")
    return built


@pytest.fixture
def retained(tmp_path):
    """Five retained reports: a counterexample, a pipe claim, an analytic, a provider-backed and a hardware-blocked task."""
    counter = finding("Separation is not monotone past the focus", "numerical", -0.5,
                      {"generator": {"name": "g"}, "checks": [CHECK]},
                      uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "binary64"}, tolerance={"abs": 1e-9, "rel": 0},
                      counterexample={"statement": "separation grows with length", "witness": {"s": 4.0}})
    drift = finding(PIPE_CLAIM, "numerical", {"drift": TINY, "orders": [4.16, 3.962, 4.0, 4.01, 3.99]},
                    {"checks": [CHECK]}, unit="m|s", tolerance={"abs": 0, "rel": 1e-6},
                    uncertainty={"kind": "roundoff", "value": ROUNDOFF, "basis": "binary64 | eps"})
    orders = finding("Atlas integration keeps fourth-order convergence", "numerical", [4.16, 3.962], {"checks": [CHECK]},
                     tolerance={"abs": 0.1, "rel": 0})
    _retain(tmp_path, "T010", "completed", [counter, drift, orders],
            mathematical_model="Jacobi equation integrated with RK4")
    _retain(tmp_path, "T021", "completed",
            [finding("Heading sensitivity equals path length on a flat torus", "mathematical", 1.0,
                     {"derivation": "K = 0"}, uncertainty={"kind": "exact", "value": 0, "basis": "closed form"},
                     tolerance={"abs": 0, "rel": 0})])
    provider = {"repository": "giasonpooni/Scientific-Computation-Runtime", "revision": "a" * 40,
                "source_tree": "b" * 40, "executed": True}
    _retain(tmp_path, "T098", "completed",
            [finding("Bundle digests agree with the pinned runtime", "provenance", "sha256:x", {"provider": provider})],
            provider_runtime_identity={"ciw": {"implementation": "ciw.lab"},
                                       "scr": {"head": "a" * 40, "tree": "b" * 40, "engine_sha256": "c" * 64},
                                       "requirement_probes": {"provider:scr": True}})
    _retain(tmp_path, "T116", "blocked", [finding("GPU energy per batch", "physical", None, {})],
            experiment="Blocked: unavailable requirement(s) hardware:nvidia-gpu.")
    _retain(tmp_path, "T147", "partial",
            [finding("CPU and GPU outputs agree under the tolerance policy on GPU hardware", "numerical", None, {},
                     expected_not_established=True),
             finding("Kernel is ready for industrial deployment", "industrial_readiness", None, {})],
            provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {"hardware:nvidia-gpu": False}},
            unresolved_assumptions=["The GPU half did not run: no GPU was probed."])
    return tmp_path


def _run(task_id, directory, junit=None, keep=False):
    implementations, _ = load_implementations()
    report = validate_report(runner.run_task(_queue()[task_id], implementations[task_id], runner.Context(directory),
                                             junit or {}))
    if keep:  # retained like a queue run, for the section tasks that follow
        (directory / "reports").mkdir(parents=True, exist_ok=True)
        (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(report), encoding="utf-8")
    return report


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


# --------------------------------------------------------------- T155
def test_label_function_matches_the_reference_oracle_exhaustively():
    result = research_portfolio.label_invariants()
    assert result["grammar"] == {"derivation": 2, "generator": 2, "checks": 4, "provider": 3, "independent_check": 8,
                                 "acquisition": 2}
    assert result["cases"] == 2 * 2 * 4 * 3 * 8 * 2 * len(DOMAINS)
    assert result["violations"] == [] and result["violation_count"] == 0
    # Every rule branch of the oracle is exercised, refusals and precedence included.
    assert result["unexercised_branches"] == []
    assert set(result["rule_branches"]) == set(research_portfolio.RULE_BRANCHES)
    counts = result["label_counts"]
    assert all(counts[label] for label in ("refused", "hardware_measured", "provider_backed", "synthetic", "analytic",
                                           "independently_verified", "numerically_verified", "not_established"))


@pytest.mark.parametrize("mutation", ["swap_precedence", "ignore_independent", "accept_same_origin", "physical_without_acquisition"])
def test_label_oracle_catches_a_label_function_that_departs_from_the_rules(monkeypatch, mutation):
    original = research_portfolio.supported_label

    def mutated(basis, domain):
        if mutation == "swap_precedence":
            label = original(basis, domain)
            return {"numerically_verified": "synthetic", "analytic": "independently_verified"}.get(label, label)
        if mutation == "ignore_independent":
            return original({k: v for k, v in basis.items() if k != "independent_check"}, domain)
        if mutation == "accept_same_origin":
            try:
                return original(basis, domain)
            except EvidenceRefusal:
                return "independently_verified"
        if domain in research_portfolio.SPEC_PHYSICAL and basis.get("checks") and "acquisition" not in basis:
            return "numerically_verified"
        return original(basis, domain)

    monkeypatch.setattr(research_portfolio, "supported_label", mutated)
    assert research_portfolio.label_invariants()["violation_count"] > 0


def test_formal_specifications_cover_the_queue(retained, tmp_path):
    report = _run("T155", retained)
    labels = _labels(report)
    assert labels[ORACLE_CLAIM] == "numerically_verified"
    assert labels["Every computational queue task is named by a specification document"] == "numerically_verified"
    assert _finding(report, "Every computational queue task")["value"] == 0
    # Five retained reports cannot exercise every specification unit: the coverage claim is refuted, not vacuous.
    assert labels["Specification documents are exercised by retained established findings"] == "not_established"
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert "14 domains" not in report["experiment"] and f"x {len(DOMAINS)} domains" in report["experiment"]
    # Without earlier reports the coverage finding is honestly unestablished, never passed by a zero threshold.
    empty = _run("T155", tmp_path / "empty")
    coverage = _finding(empty, "Specification documents are exercised")
    assert coverage["expected_not_established"] is True and coverage["value"] is None and coverage["basis"] == {}
    assert empty["state"] == "partial" and empty["evidence_status"]["primary"] == "numerically_verified"


def test_formal_specifications_complete_when_every_unit_is_exercised(retained, monkeypatch):
    queue = load_queue()
    monkeypatch.setattr(research_portfolio, "load_queue",
                        lambda: dict(queue, tasks=[t for t in queue["tasks"] if t["id"] in ("T010", "T021")]))
    monkeypatch.setattr(research_portfolio, "specification_documents",
                        lambda: {"units": {"SPECIFICATIONS.md: A": {"T010"}, "B.md": {"T021"}}, "missing_pages": []})
    report = _run("T155", retained)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"
    assert set(_labels(report).values()) == {"numerically_verified"}
    assert _finding(report, "Specification documents are exercised")["value"] == 2


# --------------------------------------------------------------- T156
def test_textbook_and_contribution_ledger(retained, monkeypatch):
    report = _run("T156", retained)
    labels = _labels(report)
    assert labels["Contributions are novel relative to the literature"] == "not_established"
    assert labels["Every retained task is attributed to a textbook result or a present implementation source file"] \
        == "numerically_verified"
    # Most ledger entries name results these five tasks never use: that claim is refuted here.
    assert labels["Every textbook result in the ledger is named by a retained task"] == "not_established"
    assert report["state"] == "partial"
    ledger = json.loads((retained / "artifacts" / "T156" / "attribution-ledger.json").read_text(encoding="utf-8"))
    rows = {row["result"]: row["tasks"] for row in ledger["textbook"]}
    assert "T010" in rows["Jacobi equation j'' + K j = 0; conjugate and focal points"]
    # Methods retained tasks name (compensated summation, fast marching, Bartels-Stewart, Monte Carlo) are in the ledger.
    names = " ".join(result + reference for result, reference, _ in research_portfolio.TEXTBOOK)
    assert all(name in names for name in ("Kahan", "Dijkstra", "Bartels", "Monte Carlo", "Cholesky"))
    # A task with no textbook name and no package source is unattributed and refutes the hypothesis.
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], changed_files=[])
    small = (("Jacobi equation", "do Carmo", r"\bJacobi\b"), ("Flat torus sensitivity", "folklore", r"(?i:flat torus)"))
    monkeypatch.setattr(research_portfolio, "TEXTBOOK", small)
    report = _run("T156", retained)
    assert _finding(report, "Every retained task is attributed")["value"] == 1
    assert _labels(report)["Every retained task is attributed to a textbook result or a present implementation source file"] \
        == "not_established"
    (retained / "reports" / "T023.json").unlink()
    report = _run("T156", retained)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"


# --------------------------------------------------------------- T157
def test_counterexample_catalogue(retained):
    catalogue = _run("T157", retained)
    record = catalogue["findings"][0]
    assert record["value"] == 1 and record["evidence_status"] == "numerically_verified"
    assert catalogue["state"] == "completed" and catalogue["unresolved_assumptions"]
    entries = json.loads((retained / "artifacts" / "T157" / "counterexamples.json").read_text(encoding="utf-8"))
    assert entries[0]["statement"] == "separation grows with length" and entries[0]["task_id"] == "T010"
    # The raw-text recount is a second path: a counterexample key the traversal does not see refutes the catalogue.
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})],
            input_data=[{"counterexample": {"statement": "outside any finding"}}])
    refuted = _run("T157", retained)
    assert refuted["findings"][0]["evidence_status"] == "not_established" and refuted["state"] == "partial"


# --------------------------------------------------------------- T158
def _figure_task(task_id, timing=False, varying=False):
    from ciw.lab import svg
    calls = iter(range(1, 100))

    def figure(ctx):
        scale = next(calls) if varying else 1
        if timing:
            ctx.artifact_json("timings.json", {"note": "Wall-clock timings of this run; not reproducible.", "seconds": scale})
        ctx.artifact_text("plot.svg", svg.line_plot([("a", [1, 2, 3], [1, 4, 9 * scale])], title="t", xlabel="x",
                                                    ylabel="y"))
        return {"fields": {}, "findings": [finding("f", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]},
                                                   uncertainty={"kind": "exact", "value": 0, "basis": "b"},
                                                   tolerance={"abs": 0, "rel": 0})]}
    return Implementation(task_id, figure)


def _retain_figures(directory, monkeypatch, fakes):
    for task_id, fake in fakes.items():
        saved = runner.run_task(_queue()[task_id], fake, runner.Context(directory), {})
        (directory / "reports").mkdir(exist_ok=True)
        (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(saved))
    real, errors = load_implementations()
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, **fakes}, errors))
    monkeypatch.setattr(research_portfolio, "REGENERATED", tuple(fakes))


def test_figures_are_reproducible(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010")})
    report = _run("T158", tmp_path)
    values = {f["claim"]: f["value"] for f in report["findings"]}
    assert values["Re-executed figure tasks without wall-clock timings regenerate byte-identical figures"] == 0
    assert set(_labels(report).values()) == {"numerically_verified"}
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"


def test_a_changed_figure_is_a_mismatch_and_a_timing_figure_a_counterexample(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010", varying=True),
                                            "T013": _figure_task("T013", timing=True, varying=True)})
    report = _run("T158", tmp_path)
    labels = _labels(report)
    # A timing-free figure that changes refutes byte reproducibility.
    assert labels["Re-executed figure tasks without wall-clock timings regenerate byte-identical figures"] == "not_established"
    assert _finding(report, "Re-executed figure tasks without")["value"] == 1
    # A timing figure that changes is a counterexample to the hypothesis, established by its check.
    counter = _finding(report, "Re-executed figures that plot wall-clock timings differ")
    assert counter["evidence_status"] == "numerically_verified" and counter["value"] == 1
    assert counter["counterexample"]["witness"] == {"figures": ["artifacts/T013/plot.svg"]}
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    # An edited retained figure fails its digest.
    (tmp_path / "artifacts" / "T010" / "plot.svg").write_text("<svg></svg>")
    edited = _run("T158", tmp_path)
    assert _labels(edited)["Retained figures hash to the digests their reports record"] == "not_established"


def test_figures_not_reexecuted_leave_the_task_partial(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010")})
    monkeypatch.setattr(research_portfolio, "REGENERATED", ())
    report = _run("T158", tmp_path)
    assert report["state"] == "partial"
    assert _finding(report, "Re-executed figure tasks without")["expected_not_established"] is True
    assert any("not re-executed" in a and "T010" in a for a in report["unresolved_assumptions"])


# --------------------------------------------------------------- T159
def test_uncertainty_table_restates_every_numerical_finding(retained):
    report = _run("T159", retained)
    assert set(_labels(report).values()) == {"numerically_verified"} and report["state"] == "completed"
    # Scalar, object and list values all count; the list without an uncertainty is identified.
    assert _finding(report, "Uncertainty table restates")["value"] == 4
    assert _finding(report, "Numerical findings that declare no")["value"] == 1
    text = (retained / "artifacts" / "T159" / "uncertainty-budget.md").read_text(encoding="utf-8")
    rows = research_portfolio._table_rows(text, research_portfolio.UNCERTAINTY_HEADER)
    assert all(len(cells) == 8 for cells in rows)
    drift = next(cells for cells in rows if cells[1] == PIPE_CLAIM)
    assert float(drift[5]) == ROUNDOFF and drift[5] == repr(ROUNDOFF) and drift[6] == "binary64 | eps"
    assert str(TINY) in drift[2] and drift[3] == "m|s"
    rows_json = json.loads((retained / "artifacts" / "T159" / "uncertainty-budget.json").read_text(encoding="utf-8"))
    assert research_portfolio._uncertainty_table_problems(text, rows_json) == []
    # A cell cut inside a number is caught by the parse-back.
    cut = text.replace(repr(ROUNDOFF), repr(ROUNDOFF)[:9])
    assert research_portfolio._uncertainty_table_problems(cut, rows_json)


def test_headline_never_cuts_a_number():
    value = {"b": [TINY, 2.5e300, -3.0, 4, 5], "a": 1.7763568394002505e-15, "c": "text 12", "d": True, "e": 7}
    cell = research_portfolio._headline(value)
    assert cell == '{"a": 1.7763568394002505e-15, "c": "text 12", "d": true, "e": 7 …(+5)}'
    assert research_portfolio._stated_numbers(cell) == [1.7763568394002505e-15, 7.0]
    assert not research_portfolio._headline_problem(cell, value)
    assert research_portfolio._headline_problem(cell.replace("e-15", "e-1"), value)
    assert research_portfolio._headline([TINY, 1, 2, 3, 4, 5]) == f"[{TINY!r}, 1, 2, 3 …(+2)]"
    # Nested values show their leading numbers with paths, never an empty placeholder.
    nested = {"euler": {"order": 1.02, "errors": [0.5, 0.25]}, "rk4": {"order": 3.99}}
    assert research_portfolio._headline(nested) == \
        '{"euler.order": 1.02, "euler.errors[0]": 0.5, "euler.errors[1]": 0.25, "rk4.order": 3.99}'
    assert research_portfolio._headline([{"x": 1}, {"x": 2e-300}]) == '{"[0].x": 1, "[1].x": 2e-300}'
    assert research_portfolio._headline({}) == "{}" and research_portfolio._headline([]) == "[]"


# --------------------------------------------------------------- T160-T162
@pytest.mark.parametrize("task_id", ["T160", "T161", "T162"])
def test_paper_drafts_trace_to_reports(retained, task_id):
    if task_id == "T162":
        _run("T155", retained, keep=True)
    report = _run(task_id, retained)
    assert report["state"] == "partial"
    labels = _labels(report)
    assert labels["Draft has passed external peer review"] == "not_established"
    assert labels["Draft results table restates every retained finding of its sections with its retained label"] \
        == "numerically_verified"
    slug = research_portfolio.PAPERS[task_id][0]
    text = (retained / "artifacts" / task_id / f"{slug}-draft.md").read_text(encoding="utf-8")
    assert "Not peer reviewed" in text and report["hypothesis"].startswith(("An ", "A "))
    assert "physical validation is not established for any of them" in text
    for cells in research_portfolio._table_rows(text, research_portfolio.RESULTS_HEADER):
        assert cells is not None and len(cells) == 6 and cells[4].startswith("`")
    if task_id == "T161":
        assert "max\\|g(v,v) - 1\\|" in text and PIPE_CLAIM in [c[1] for c in research_portfolio._table_rows(
            text, research_portfolio.RESULTS_HEADER)]
    if task_id == "T162":
        assert "weakest" in text and "supported_label" in text and "raw_sha256" in text
        assert labels["Draft methods state the evidence-label rules that T155 found the label function to follow"] \
            == "numerically_verified"


def test_a_corrupted_draft_row_fails_the_citation_check(retained):
    _run("T161", retained)
    text = (retained / "artifacts" / "T161" / "geometry-methods-draft.md").read_text(encoding="utf-8")
    reports = [r for r in runner.load_reports(retained) if r["section"] in research_portfolio.PAPERS["T161"][2]]
    cited = [(r, f) for r in reports for f in r["findings"]]
    assert research_portfolio._draft_problems(text, cited) == []
    unescaped = text.replace("max\\|g(v,v) - 1\\|", "max|g(v,v) - 1|")
    relabelled = text.replace("`numerically_verified`", "`independently_verified`", 1)
    dropped = "\n".join(line for line in text.splitlines() if "Heading sensitivity" not in line)
    for corrupted in (unescaped, relabelled, dropped):
        assert research_portfolio._draft_problems(corrupted, cited)


# --------------------------------------------------------------- T163
def test_portfolio_shows_every_label_in_use(retained):
    report = _run("T163", retained)
    record = report["findings"][0]
    assert record["evidence_status"] == "numerically_verified" and report["evidence_status"]["primary"] == "numerically_verified"
    # Only T010 of the curated panels is retained: panels are added for analytic, provider_backed and not_established.
    assert record["value"] == 4 and report["state"] == "partial"
    text = (retained / "artifacts" / "T163" / "PORTFOLIO.md").read_text(encoding="utf-8")
    shown = set(research_portfolio.SHOWN_LABEL.findall(text))
    assert shown == {"analytic", "numerically_verified", "provider_backed", "not_established"}
    assert "- GPU energy per batch: " in text


# --------------------------------------------------------------- T164
def test_clean_room_marker_is_recognized(tmp_path, monkeypatch):
    monkeypatch.delenv("CIW_LAB_CLEAN_ROOM", raising=False)
    report = _run("T164", tmp_path)
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    assert any("--no-compare" in a for a in report["unresolved_assumptions"])
    assert "Retained reports" not in json.dumps(report["input_data"])
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
    observation = json.loads((tmp_path / "genuine" / "artifacts" / "T164" / "clean-room-observation.json").read_text())
    assert observation["conditions"]["installed_from_wheel"] is True and "wheel_sha256" not in json.dumps(observation)


def test_clean_room_prose_does_not_carry_the_wheel_digest(tmp_path, monkeypatch):
    import hashlib
    import shutil
    import zipfile
    import ciw
    package = Path(ciw.__file__).resolve().parent
    first = _wheel(tmp_path / "ciw-0-py3-none-any.whl", package)
    # Another build of the same files: equal members, different bytes and digest (as file times do).
    second = tmp_path / "rebuilt" / first.name
    second.parent.mkdir()
    shutil.copy2(first, second)
    with zipfile.ZipFile(second, "a") as archive:
        archive.comment = b"second build"
    assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(second.read_bytes()).digest()
    for wheel, run in ((first, "run1"), (second, "run2")):
        _clean_room(monkeypatch, wheel)
        report = _run("T164", tmp_path / run)
        assert report["state"] == "completed" and _labels(report) == {research_portfolio.CLEAN_ROOM_CLAIM: "numerically_verified"}
        assert report["provider_runtime_identity"]["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
        (tmp_path / run / "reports").mkdir(parents=True)
        (tmp_path / run / "reports" / "T164.json").write_text(runner.dumps(report), encoding="utf-8")
    # Compared prose is identical across builds; the digest stays in the runtime identity.
    assert runner.compare(tmp_path / "run1", tmp_path / "run2")["problems"] == []


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


# --------------------------------------------------------------- T165
def test_release_report_inventories_nested_runtimes(retained, tmp_path):
    release = _run("T165", retained)
    assert release["state"] == "completed"
    labels = _labels(release)
    assert labels["Release state and label totals match a raw-text recount of the retained report files"] == "numerically_verified"
    assert labels["Every report whose provider probe succeeded contributes a runtime identity to the release inventory"] \
        == "numerically_verified"
    assert labels["The release digest is signed by a project key"] == "not_established"
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    # The SCR identity sits below the top level of T098's runtime identity.
    assert {r["runtime"] for r in record["runtimes"]} == {"scr"} and record["runtimes"][0]["tasks"] == ["T098"]
    assert record["physical_validation"] == "not_established" and record["reports"] == 5
    assert record["queue"]["tasks"] == 168 and "T165" in record["queue"]["not_reported"]
    # Artifact bytes (timing figures) stay out of the digest: a rerun with other artifacts gives the same digest.
    other = tmp_path / "other"
    for path in (retained / "reports").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["generated_artifacts"] = [{"path": f"artifacts/{report['task_id']}/t.svg", "sha256": "d" * 64, "bytes": 1}]
        report["report_id"] = __import__("ciw.lab.report", fromlist=["report_identity"]).report_identity(report)
        (other / "reports").mkdir(parents=True, exist_ok=True)
        (other / "reports" / path.name).write_text(runner.dumps(report), encoding="utf-8")
    _run("T165", other)
    again = json.loads((other / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    assert again["release_digest"] == record["release_digest"]
    # A report that ran a provider without recording its identity refutes the inventory.
    _retain(retained, "T099", "completed", [finding("g", "numerical", 1, {"checks": [CHECK]},
                                                    uncertainty={"kind": "exact", "value": 0, "basis": "b"})],
            provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {"provider:scr": True}})
    refuted = _run("T165", retained)
    assert refuted["state"] == "partial" and refuted["evidence_status"]["primary"] == "not_established"


# --------------------------------------------------------------- T166
def test_unresolved_assumption_ledger(retained):
    _run("T155", retained, keep=True)  # this section's earlier reports are in the ledger too
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], unresolved_assumptions=[])
    ledger = _run("T166", retained)
    assert ledger["findings"][0]["evidence_status"] == "numerically_verified" and ledger["state"] == "completed"
    rows = json.loads((retained / "artifacts" / "T166" / "unresolved-assumptions.json").read_text(encoding="utf-8"))
    assert {task for row in rows for task in row["tasks"]} == {"T010", "T021", "T098", "T116", "T147", "T155"}
    assert ledger["findings"][0]["value"] == len(rows)
    assert "Reports stating no unresolved assumption: T023" in ledger["unresolved_assumptions"]
    text = (retained / "artifacts" / "T166" / "UNRESOLVED_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert research_portfolio._ledger_problems(text, rows) == []
    assert research_portfolio._ledger_problems(text.replace("(T010)", "(T011)"), rows)


# --------------------------------------------------------------- T167
def test_unmeasured_ledger(retained):
    unmeasured = _run("T167", retained)
    document = json.loads((retained / "artifacts" / "T167" / "unmeasured.json").read_text(encoding="utf-8"))
    assert [c["task_id"] for c in document["not_established_claims"]] == ["T116", "T147"]
    assert {t["task_id"]: t["state"] for t in document["unfinished_tasks"]} == {"T116": "blocked", "T147": "partial"}
    # A hardware claim filed under a computational domain is listed through its task's failed hardware probe.
    assert "CPU and GPU outputs agree under the tolerance policy on GPU hardware" in {
        h["claim"] for h in document["hardware_unavailable_findings"]}
    text = (retained / "artifacts" / "T167" / "UNMEASURED.md").read_text(encoding="utf-8")
    assert "The GPU half did not run" in text
    labels = _labels(unmeasured)
    assert labels["Unmeasured ledger lists every not-established physical or authority claim"] == "numerically_verified"
    assert labels["Unmeasured ledger lists every blocked, deferred or partial task"] == "numerically_verified"
    assert labels["Physical validity of the lab's computational results"] == "not_established"
    assert unmeasured["physical_validation_status"]["status"] == "not_established" and unmeasured["state"] == "completed"


def test_aggregates_block_without_prior_reports(tmp_path):
    for task_id in ("T157", "T158", "T159", "T160", "T161", "T162", "T163", "T165", "T166", "T167"):
        report = _run(task_id, tmp_path)
        assert report["state"] == "blocked" and report["findings"] == [], task_id
    # T156 checks its curated half without reports; attribution stays unestablished and the task partial.
    ledger = _run("T156", tmp_path)
    assert ledger["state"] == "partial"
    assert [f["evidence_status"] for f in ledger["findings"]] == [
        "not_established", "not_established", "numerically_verified", "not_established"]
    assert all(f.get("expected_not_established") for f in ledger["findings"] if f["evidence_status"] == "not_established")


# --------------------------------------------------------------- T168
TIED_TESTS = '''
def test_t010_runs_and_labels(lab):
    assert lab("T010")["findings"][0]["evidence_status"] == "numerically_verified"

def test_values_only(lab):
    assert lab("T116")["findings"][0]["value"] is None

def test_unrelated():
    assert 1 + 1 == 2
'''


def _fake_repository(root, monkeypatch):
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_fake.py").write_text(TIED_TESTS, encoding="utf-8")
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(root))


def test_regression_coverage_is_checked(retained, tmp_path, monkeypatch):
    _fake_repository(tmp_path / "repo", monkeypatch)
    tied, values = "tests/test_fake.py::test_t010_runs_and_labels", "tests/test_fake.py::test_values_only"

    def registry(t010, t116=(values,)):
        # Only the fixture's tasks, so real section registrations cannot leak in.
        fakes = {"T010": Implementation("T010", None, regression_tests=t010),
                 "T116": Implementation("T116", None, regression_tests=t116)}
        monkeypatch.setattr(research_portfolio, "load_implementations", lambda: (fakes, {}))

    registry((tied,))
    report = _run("T168", retained)
    labels = _labels(report)
    rows = json.loads((retained / "artifacts" / "T168" / "regression-coverage.json").read_text(encoding="utf-8"))
    assert {row["task_id"]: row["tied"] for row in rows}["T010"] == [tied]
    # T116's test asserts values only, and T021, T098 and T147 register no test at all.
    assert _finding(report, "Tasks without a registered test that both")["value"] == 1
    assert labels["Tasks without a registered test that both names the task and asserts an evidence label"] == "numerically_verified"
    assert _finding(report, "Completed or partial tasks lacking")["value"] == 3
    assert labels["Completed or partial tasks lacking a regression test"] == "not_established"
    # No JUnit outcomes were recorded in these reports: the pass/fail finding is honestly unestablished.
    assert _finding(report, "Registered regression tests failing")["expected_not_established"] is True
    assert report["state"] == "partial"
    # A registered test that fails in the recorded JUnit outcomes refutes the task.
    _retain(retained, "T010", "partial", [finding("f", "numerical", 1, {"checks": [CHECK]})],
            tests_passed=[], extra={"tests_failed": [f"pytest: {tied}"]})
    failing = _run("T168", retained)
    assert _labels(failing)["Registered regression tests failing in the JUnit record of this run"] == "not_established"
    registry(("tests/test_fake.py::test_does_not_exist",))
    report = _run("T168", retained)
    assert report["state"] == "partial" and _finding(report, "Registered regression node ids")["value"] == 1


def test_regression_tie_analysis_is_checked_on_probe_cases():
    assert research_portfolio._tie_probe_errors() == []
    index = {}
    research_portfolio._index_source("tests/test_fake.py", TIED_TESTS, index)
    assert research_portfolio._tie("T010", "", "tests/test_fake.py::test_t010_runs_and_labels", index) == (True, True)
    assert research_portfolio._tie("T010", "", "tests/test_fake.py::test_unrelated", index) == (False, False)
    assert research_portfolio._tie("T116", "", "tests/test_fake.py::test_values_only", index) == (True, False)


def test_every_section_registration_is_tied_to_its_task():
    implementations, _ = load_implementations()
    index = research_portfolio._test_index(Path(__file__).resolve().parent)
    for task_id in SECTION:
        implementation = implementations[task_id]
        function = getattr(implementation.run, "__name__", "")
        assert any(research_portfolio._tie(task_id, function, node, index) == (True, True)
                   for node in implementation.regression_tests), task_id


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


# --------------------------------------------------------------- whole section
def test_every_numerical_finding_declares_uncertainty_and_tolerance(retained, monkeypatch):
    monkeypatch.delenv("CIW_LAB_CLEAN_ROOM", raising=False)
    for task_id in SECTION:
        report = _run(task_id, retained, keep=True)
        assert report["unresolved_assumptions"], task_id
        for record in report["findings"]:
            if not research_portfolio._numbers(record["value"]):
                continue
            uncertainty = record["uncertainty"]
            assert isinstance(uncertainty, dict) and set(uncertainty) == {"kind", "value", "basis"}, (task_id, record["claim"])
            assert math.isfinite(uncertainty["value"]) and uncertainty["basis"], (task_id, record["claim"])
            assert record["regression_tolerance"] == {"abs": 0, "rel": 0}, (task_id, record["claim"])
            assert record["basis"].get("checks"), (task_id, record["claim"])
