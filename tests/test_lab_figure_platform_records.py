"""Second-platform figure records (lab/figure-platforms/) and T158's binding to them.

The record tests start from a small synthetic record under
tests/fixtures/lab/figure-platform/ (no figure task ran for it, and its digests
name no real figure) and change one thing at a time, resealing the manifest
where the defect is not the manifest's own. T158's tests retain fake figure
tasks, as tests/test_lab_research_portfolio.py does, and bind a record built
from the fixture whose entries name those figures. A record is refused when it
was made on this process's operating system, so those tests fix the host's as
Linux (the fixture's platform is Windows). Tests of the scripts, workflows and
the repository's records skip where those files are not beside the tests (the
clean room).
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

from ciw import cli
from ciw.lab import figure_platform_records as records
from ciw.lab import research_portfolio, runner, svg
from ciw.lab.evidence import finding
from ciw.lab.registry import Implementation, load_implementations, load_queue
from ciw.lab.report import validate_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lab" / "figure-platform" / "windows-1001"
RECORD_ID = "windows-1001"
QUEUE = {t["id"]: t for t in load_queue()["tasks"]}
CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}
RECORD_CLAIM = research_portfolio.PLATFORM_RECORD_CLAIM
PLATFORM_CLAIM = research_portfolio.SECOND_PLATFORM_CLAIM
FIGURES = research_portfolio.FIGURE_CLAIM
PROVIDER_REASON = "provider csg not bound here; the retained run used it"
IDENTITY = {"os": "Windows", "platform": "Windows-2025Server-10.0.26100-SP0", "machine": "AMD64", "python": "3.12.10",
            "numpy": "2.4.3", "openblas_core": "Haswell"}
BOUND = 1e-9  # the rounding bound a fake rounding-level figure records for every point


@pytest.fixture(autouse=True)
def linux_host(monkeypatch):
    """The fixture record was made on Windows; the run reading it is on Linux, wherever these tests run."""
    monkeypatch.setattr(records, "host_system", lambda: "Linux")


def _digest(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _edit(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    _write(path, value)


def _copy(root: Path, name: str = RECORD_ID) -> Path:
    target = root / records.DIRECTORY / name
    shutil.copytree(FIXTURE, target)
    return target


def _reseal(directory: Path) -> None:
    """Record the text files' digests and every file's in the manifest again, so an edit is the only defect."""
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    for name in records.TEXT_FILES:
        record["artifact_files"][name] = {**_digest((directory / name).read_bytes()), "line_endings": "lf"}
    _write(directory / "record.json", record)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {path.relative_to(directory).as_posix(): _digest(path.read_bytes())
                         for path in sorted(directory.rglob("*"))
                         if path.is_file() and path.relative_to(directory).as_posix() != "manifest.json"}
    _write(directory / "manifest.json", manifest)


def _summary(check: dict) -> dict:
    counts = records.recount(check, len)  # one figure per synthetic task
    return {key: counts[key] for key in records.SUMMARY_KEYS}


# ---------------------------------------------------------------- the record and its verification

def test_the_synthetic_record_verifies(tmp_path):
    directory = _copy(tmp_path / "lab")
    inspected = records.inspect_record(directory)
    assert inspected["problems"] == [] and inspected["files"] == 3
    summary = inspected["summary"]
    assert summary["system"] == "Windows" and summary["identity"] == IDENTITY
    assert summary["not_reexecuted"] == {"T030": PROVIDER_REASON} and summary["counts"]["compared_figures"] == 3
    assert summary["source"]["artifact"]["name"] == "figure-check-windows"
    result = records.verify_records(tmp_path / "lab")
    assert result["passed"] and result["verified"] == 1 and "integrity only" in result["note"]
    # A record made on the reading host's operating system is no second platform. The OS is platform.system()'s,
    # or else read from the platform string, which spells Darwin as macOS.
    assert records.second_platform_problem(summary, host="windows") == (
        "the record was made on Windows, the operating system of this run, so it is no second platform")
    assert records.second_platform_problem(summary, host="Linux") is None
    for text, system in (("Windows-2025Server-10.0.26100-SP0", "Windows"), ("Linux-6.8.0-x86_64-with-glibc2.39", "Linux"),
                         ("macOS-14.5-arm64-arm-64bit", "Darwin")):
        assert records.record_system({"platform": text}) == system
    assert records.record_system({"platform": "macOS-14.5", "system": "Darwin"}) == "Darwin"


def _crlf(directory: Path) -> None:
    path = directory / "figure-check.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    _edit(directory / "manifest.json", lambda m: m["files"].update({"figure-check.json": _digest(path.read_bytes())}))


def _renamed(directory: Path) -> Path:
    target = directory.parent / "windows-1002"
    directory.rename(target)
    return target


@pytest.mark.parametrize("defect, expected", [
    (lambda d: (d / "figure-check.md").write_bytes((d / "figure-check.md").read_bytes() + b" "),
     "integrity: figure-check.md differs from its manifest digest"),
    (lambda d: (d / "figure-check.md").unlink(), "integrity: figure-check.md is missing or is a link, not retained bytes"),
    (lambda d: (d / "notes.txt").write_text("x"), "integrity: notes.txt is not recorded in manifest.json"),
    (lambda d: _edit(d / "manifest.json", lambda m: m["files"].update({"run/reports/T010.json": _digest(b"{}")})),
     "integrity: manifest.json lists 'run/reports/T010.json', which is not a file of a figure-platform record"),
    (_crlf, "integrity: figure-check.json holds a carriage return; the record stores it with LF line endings"),
    (_renamed, "integrity: manifest.json names record 'windows-1001', retained as 'windows-1002'"),
])
def test_a_record_with_a_changed_missing_or_unrecorded_file_is_refused(tmp_path, defect, expected):
    directory = _copy(tmp_path / "lab")
    moved = defect(directory)
    inspected = records.inspect_record(moved if isinstance(moved, Path) else directory)
    assert expected in inspected["problems"] and inspected["summary"] is None


def _mismatch(check: dict) -> None:
    check["figures"][0].update(outcome="differs", fresh_sha256="0" * 64)
    check["summary"].update(identical=0, mismatched=1)


@pytest.mark.parametrize("path, change, expected", [
    ("figure-check.json", lambda c: c["summary"].update(identical=2),
     "figure_check: figure-check.json summary identical is 2; its figure list gives 1"),
    ("figure-check.json", lambda c: c["summary"].update(figures=5),
     "figure_check: figure-check.json summary counts 5 figures, not the 4 compared, not re-executed and not comparable"),
    ("figure-check.json", lambda c: c["figures"][0].update(fresh_sha256="0" * 64),
     "figure_check: figure-check.json figures[0] outcome 'identical' disagrees with its retained and fresh digests"),
    ("figure-check.json", lambda c: c["figures"][0].update(outcome="same structure"),
     "figure_check: figure-check.json figures[0] outcome 'same structure' is not an outcome of an undeclared figure"),
    ("figure-check.json", lambda c: c["figures"][0].update(retained_values=[]),
     "figure_check: figure-check.json figures[0] retained_values are not the recorded values of a rounding-level figure"),
    ("figure-check.json", lambda c: c["figures"][0].update(task_sources={"svg.py": "0" * 64}),
     "figure_check: figure-check.json figures[0] task_sources are not the source digests of its task's retained report"),
    ("figure-check.json", lambda c: c["figures"][1].update(path="artifacts/T010/plot.svg"),
     "figure_check: figure-check.json figures[1] path 'artifacts/T010/plot.svg' is not an SVG artifact of its task"),
    ("figure-check.json", lambda c: c["not_reexecuted"].update(T010="sources differ"),
     "figure_check: figure-check.json lists T010 as compared and left out, or left out twice"),
    ("figure-check.json", lambda c: c.update(retained="/home/operator/lab"),
     "figure_check: figure-check.json.retained holds a host path"),
    ("figure-check.json", lambda c: c["platform"]["ciw"].update(package_digest="unknown"),
     "figure_check: figure-check.json does not name the digest of the CIW package the run executed"),
    ("figure-check.json", _mismatch,
     "integrity: the fresh SVG of mismatched figure artifacts/T010/plot.svg is not retained"),
    ("record.json", lambda r: r["platform"].update(python="3.11.9"),
     "record: record.json platform differs from the platform figure-check.json records"),
    ("record.json", lambda r: r["source"].update(head_sha="main"),
     "record: record.json source head_sha is not a 40-digit commit"),
    ("record.json", lambda r: r["source"]["artifact"].update(digest="9b29f55e"),
     "record: record.json source artifact digest is not sha256:<64 hex digits>"),
    ("record.json", lambda r: r["source"].update(workflow="figures.yml"),
     "record: record.json source workflow is not a path under .github/workflows"),
    ("record.json", lambda r: r.update(record_id="windows-1002"),
     "record: record.json names record 'windows-1002', retained as 'windows-1001'"),
])
def test_a_resealed_record_that_contradicts_itself_is_refused(tmp_path, path, change, expected):
    directory = _copy(tmp_path / "lab")
    _edit(directory / path, change)
    _reseal(directory)
    inspected = records.inspect_record(directory)
    assert expected in inspected["problems"], inspected["problems"]
    assert inspected["summary"] is None


def test_the_text_files_give_back_the_bytes_the_artifact_held(tmp_path):
    # record.json records figure-check.json as the zip held it (CRLF, as a Windows run writes it); the retained LF copy
    # must give those bytes back.
    directory = _copy(tmp_path / "lab")
    raw = (directory / "figure-check.json").read_bytes()
    _edit(directory / "record.json", lambda r: r["artifact_files"].update(
        {"figure-check.json": {**_digest(raw.replace(b"\n", b"\r\n")), "line_endings": "crlf"}}))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["record.json"] = _digest((directory / "record.json").read_bytes())
    _write(directory / "manifest.json", manifest)
    assert records.inspect_record(directory)["problems"] == []
    _edit(directory / "record.json", lambda r: r["artifact_files"]["figure-check.json"].update(line_endings="lf"))
    manifest["files"]["record.json"] = _digest((directory / "record.json").read_bytes())
    _write(directory / "manifest.json", manifest)
    assert records.inspect_record(directory)["problems"] == [
        "integrity: figure-check.json does not give back the bytes record.json records for it in the artifact"]


def _zip(path: Path, members: dict) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 9, 24, 22, 56, 0)), data)
    path.write_bytes(buffer.getvalue())
    return "sha256:" + hashlib.sha256(buffer.getvalue()).hexdigest()


PROVENANCE = {"repository": "example-owner/example-workbench", "workflow": ".github/workflows/figures.yml",
              "run_id": 1001, "head_sha": "a" * 40, "artifact_id": 2002, "artifact_name": "figure-check-windows"}


def test_retention_keeps_only_the_record_files_and_refuses_another_digest(tmp_path):
    check, summary = (FIXTURE / "figure-check.json").read_bytes(), (FIXTURE / "figure-check.md").read_bytes()
    # As the Windows run uploads it: CRLF text files beside the fresh reports and SVGs.
    members = {"figure-check.json": check.replace(b"\n", b"\r\n"), "figure-check.md": summary.replace(b"\n", b"\r\n"),
               "run/reports/T010.json": b"{}", "run/artifacts/T010/plot.svg": b"<svg/>"}
    digest = _zip(tmp_path / "artifact.zip", members)
    lab = tmp_path / "lab"
    with pytest.raises(ValueError, match="not the declared artifact digest"):
        records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest="sha256:" + "0" * 64, **PROVENANCE)
    with pytest.raises(ValueError, match="sha256:<64 hex digits>"):
        records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest[7:], **PROVENANCE)
    assert not (lab / records.DIRECTORY).exists()
    result = records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest, run_attempt=1, **PROVENANCE)
    directory = lab / records.DIRECTORY / RECORD_ID
    assert result["record_id"] == RECORD_ID and result["date"] == "2026-09-24" and result["files"] == 3
    assert sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*")) == [
        "figure-check.json", "figure-check.md", "manifest.json", "record.json"]
    # Stored with LF line endings, as the repository keeps them, and recorded as the zip held them.
    assert (directory / "figure-check.json").read_bytes() == check
    record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
    assert record["artifact_files"]["figure-check.json"] == {**_digest(members["figure-check.json"]),
                                                             "line_endings": "crlf"}
    assert record["source"]["artifact"]["digest"] == digest and record["source"]["run_attempt"] == 1
    assert runner._host_paths(record, "record.json") == [] and records.inspect_record(directory)["problems"] == []
    with pytest.raises(ValueError, match="already retained"):
        records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest, **PROVENANCE)
    # A later attempt is named apart; a bad provenance or record identity is refused before anything is written.
    assert records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest, run_attempt=2,
                                 **PROVENANCE)["record_id"] == "windows-1001-2"
    with pytest.raises(ValueError, match="head_sha is not a 40-digit commit"):
        records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest,
                              **{**PROVENANCE, "head_sha": "main"}, record_id="other")
    with pytest.raises(ValueError, match="kebab-case"):
        records.retain_record(tmp_path / "artifact.zip", lab, artifact_digest=digest, **PROVENANCE, record_id="Bad Id")
    assert sorted(p.name for p in (lab / records.DIRECTORY).iterdir()) == ["windows-1001", "windows-1001-2"]


def test_retention_keeps_the_fresh_svg_of_a_mismatch_and_refuses_what_it_cannot_retain(tmp_path):
    # As a Windows run might write it: the record keeps its bytes, CRLF included, since they must hash to fresh_sha256.
    fresh = b"<svg xmlns='http://www.w3.org/2000/svg'>\r\n<path/>\r\n</svg>\r\n"
    check = json.loads((FIXTURE / "figure-check.json").read_text(encoding="utf-8"))
    check["figures"][0].update(outcome="differs", fresh_sha256=hashlib.sha256(fresh).hexdigest())
    check["summary"] = _summary(check)
    text = (json.dumps(check, indent=1, sort_keys=True) + "\n").encode()
    summary = (FIXTURE / "figure-check.md").read_bytes()
    digest = _zip(tmp_path / "mismatch.zip", {"figure-check.json": text, "figure-check.md": summary,
                                              "run/artifacts/T010/plot.svg": fresh,
                                              "run/artifacts/T013/plot.svg": b"<svg/>"})
    records.retain_record(tmp_path / "mismatch.zip", tmp_path / "lab", artifact_digest=digest, **PROVENANCE)
    directory = tmp_path / "lab" / records.DIRECTORY / RECORD_ID
    assert (directory / "fresh" / "artifacts" / "T010" / "plot.svg").read_bytes() == fresh
    assert not (directory / "fresh" / "artifacts" / "T013").exists()
    assert records.inspect_record(directory)["summary"]["fresh_figures"] == ["artifacts/T010/plot.svg"]
    # Its line endings normalized (what .gitattributes keeps git from doing): it differs, and is not "not retained".
    (directory / "fresh" / "artifacts" / "T010" / "plot.svg").write_bytes(fresh.replace(b"\r\n", b"\n"))
    assert records.inspect_record(directory)["problems"] == [
        "integrity: fresh/artifacts/T010/plot.svg differs from its manifest digest"]
    for name, members, message in (
            ("no-svg", {"figure-check.json": text, "figure-check.md": summary}, "no fresh SVG run/artifacts/T010/plot.svg"),
            ("other-svg", {"figure-check.json": text, "figure-check.md": summary,
                           "run/artifacts/T010/plot.svg": b"<svg/>"}, "is not the fresh figure"),
            ("no-summary", {"figure-check.json": text}, "holds no figure-check.md"),
            ("mixed", {"figure-check.json": text.replace(b"\n", b"\r\n", 3), "figure-check.md": summary},
             "mixes LF and CRLF line endings"),
            ("nothing-compared", {"figure-check.json": json.dumps({**check, "figures": [], "summary": _summary(
                {**check, "figures": []})}).encode(), "figure-check.md": summary}, "compared no figure"),
            ("not-a-check", {"figure-check.json": b'{"schema": "other"}', "figure-check.md": summary},
             "is not ciw.lab-figure-check.v1")):
        digest = _zip(tmp_path / f"{name}.zip", members)
        with pytest.raises(ValueError, match=re.escape(message)):
            records.retain_record(tmp_path / f"{name}.zip", tmp_path / name, artifact_digest=digest, **PROVENANCE)
        assert not (tmp_path / name).exists()


def test_git_keeps_the_bytes_of_a_records_fresh_svgs():
    """A retained fresh SVG must hash to its recorded digest: .gitattributes exempts it from the line-ending
    normalization every other text file of a record gets."""
    if not (ROOT / ".gitattributes").is_file() or shutil.which("git") is None:
        pytest.skip(".gitattributes is not beside the tests")
    paths = ["lab/figure-platforms/windows-1/fresh/artifacts/T010/plot.svg",
             "lab/figure-platforms/windows-1/record.json"]
    result = subprocess.run(["git", "check-attr", "text", "--", *paths], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        pytest.skip("the tests are not in a git work tree")
    assert result.stdout.splitlines() == [f"{paths[0]}: text: unset", f"{paths[1]}: text: auto"]


def test_lab_verify_checks_every_figure_platform_record(tmp_path, capsys):
    directory = _copy(tmp_path / "lab")
    result = runner.verify_retained(tmp_path / "lab", tmp_path / "fresh")
    assert result["figure_platform_records"]["verified"] == 1
    assert not any(problem.startswith("figure-platforms/") for problem in result["problems"])
    assert cli.main(["lab", "figure-platform", "verify", "--retained", str(tmp_path / "lab")]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    (directory / "figure-check.md").write_bytes(b"edited\n")
    result = runner.verify_retained(tmp_path / "lab", tmp_path / "fresh")
    assert (f"figure-platforms/{RECORD_ID}: integrity: figure-check.md differs from its manifest digest"
            in result["problems"]) and not result["passed"]
    assert cli.main(["lab", "figure-platform", "verify", "--retained", str(tmp_path / "lab")]) == 3
    assert json.loads(capsys.readouterr().out)["passed"] is False


# ---------------------------------------------------------------- T158 with and without a record

def _figure_task(task_id, timing=False, rounding=False, provider=None, varying=False, states=("completed",)):
    """A fake figure task with one figure: a wall-clock ``timing`` figure changes on every call (same structure), as
    an undeclared ``varying`` one does, a ``rounding``-level one records its values with :data:`BOUND`; ``provider``
    is probed as the task's provider; call ``n`` ends in ``states[n - 1]`` (the last from there on). Its report
    records the digest of ``src/ciw/lab/svg.py`` as its task sources."""
    calls = iter(range(1, 100))

    def figure(ctx):
        call = next(calls)
        if provider:
            ctx.available(f"provider:{provider}")
        ys = [1.0, 4.0 + (call if timing or varying else 0)]
        ctx.artifact_text("plot.svg", svg.line_plot([("a", [1, 2], ys)], title="t", xlabel="x", ylabel="y",
                                                    rounding=BOUND if rounding else None),
                          wall_clock_timing=timing, rounding_level=rounding)
        return {"state": states[min(call, len(states)) - 1], "fields": {},
                "findings": [finding("f", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]},
                                     uncertainty={"kind": "exact", "value": 0, "basis": "b"},
                                     tolerance={"abs": 0, "rel": 0})]}
    return Implementation(task_id, figure, changed_files=("src/ciw/lab/svg.py",))


def _retain(directory, monkeypatch, fakes, regenerated=(), providers=None):
    """Retain the fake figure tasks' reports in ``directory``; T158 re-executes ``regenerated`` (and timing tasks)."""
    for task_id, fake in fakes.items():
        saved = runner.run_task(QUEUE[task_id], fake, runner.Context(directory, providers), {})
        (directory / "reports").mkdir(parents=True, exist_ok=True)
        (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(saved), encoding="utf-8")
    real, errors = load_implementations()
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, **fakes}, errors))
    monkeypatch.setattr(research_portfolio, "REGENERATED", tuple(regenerated))


def _record(root, run, outcomes=None, not_reexecuted=None, edits=None) -> Path:
    """A record from the fixture whose entries are the retained figures of ``run`` (except the tasks listed as not
    re-executed there), with their reports' task sources: identical, same structure or within rounding bounds by
    declaration unless ``outcomes`` says otherwise, each changed by its task's ``edits`` before the record is
    sealed."""
    directory = _copy(root)
    check = json.loads((directory / "figure-check.json").read_text(encoding="utf-8"))
    figures = []
    for path in sorted((run / "reports").glob("T*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        task_id = report["task_id"]
        for artifact in report["generated_artifacts"]:
            if not artifact["path"].endswith(".svg") or task_id in (not_reexecuted or {}):
                continue
            timing, rounding = artifact.get("wall_clock_timing") is True, artifact.get("rounding_level") is True
            outcome = (outcomes or {}).get(task_id) or (
                "same structure" if timing else "within rounding bounds" if rounding else "identical")
            data = (run / artifact["path"]).read_bytes()
            fresh = data if outcome == "identical" else data + b"\n<!-- regenerated there -->"
            entry = {"task_id": task_id, "path": artifact["path"], "wall_clock_timing": timing,
                     "rounding_level": rounding, "outcome": outcome, "retained_sha256": artifact["sha256"],
                     "fresh_sha256": hashlib.sha256(fresh).hexdigest()}
            if timing or rounding:
                entry["retained_structure"] = research_portfolio.figure_structure(data)
            if rounding:
                entry["retained_values"] = svg.recorded_values(data)
            if research_portfolio.task_sources(report):  # as scripts/check_figures.py names them
                entry["task_sources"] = research_portfolio.task_sources(report)
            (edits or {}).get(task_id, lambda e: None)(entry)
            if outcome in records.mismatch_outcomes():
                (directory / "fresh" / artifact["path"]).parent.mkdir(parents=True, exist_ok=True)
                (directory / "fresh" / artifact["path"]).write_bytes(fresh)
            figures.append(entry)
    check.update(figures=figures, not_reexecuted=dict(not_reexecuted or {}), not_comparable={})
    check["summary"] = _summary(check)
    _write(directory / "figure-check.json", check)
    _reseal(directory)
    assert records.inspect_record(directory)["problems"] == []
    return directory


def _run(directory, record=None):
    implementations, _ = load_implementations()
    providers = {research_portfolio.FIGURE_PLATFORM_RECORD: record} if record is not None else None
    return validate_report(runner.run_task(QUEUE["T158"], implementations["T158"],
                                           runner.Context(directory, providers), {}))


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _finding(report, claim):
    matches = [f for f in report["findings"] if f["claim"] == claim]
    assert len(matches) == 1, claim
    return matches[0]


@pytest.mark.lab_task("T158")
def test_t158_reads_a_second_platform_record_as_provider_backed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    plsr_reason = "provider plsr-python not bound here; the retained run used it"
    _retain(run, monkeypatch, {"T010": _figure_task("T010"), "T013": _figure_task("T013", timing=True),
                               "T020": _figure_task("T020", rounding=True),
                               "T030": _figure_task("T030", provider="csg"),
                               "T101": _figure_task("T101", provider="plsr-python")},
            regenerated=("T010",), providers={"csg": tmp_path, "plsr-python": tmp_path})
    record = _record(tmp_path / "records", run, not_reexecuted={"T030": PROVIDER_REASON, "T101": plsr_reason})
    report = _run(run, record)
    labels = _labels(report)
    # The record's integrity, platform and currency are CIW's recomputations; that the figures regenerated on
    # Windows rests on the CI run that wrote the record, with its provenance as the basis.
    assert labels[RECORD_CLAIM] == "numerically_verified" and labels[PLATFORM_CLAIM] == "provider_backed"
    assert labels[FIGURES] == "numerically_verified"
    claim = _finding(report, PLATFORM_CLAIM)
    assert claim["value"] == {
        "record": RECORD_ID, "platform": IDENTITY, "compared": 3, "mismatched": 0,
        "outcomes": {"identical": 1, "same structure": 1, "within rounding bounds": 1},
        "figures_not_compared_there": {"not re-executed there": 2},
        "not_reexecuted": {"T030": PROVIDER_REASON, "T101": plsr_reason}, "not_comparable": {}}
    fixture = json.loads((FIXTURE / "record.json").read_text(encoding="utf-8"))
    assert claim["origin"] == ["provider"] and claim["basis"]["provider"] == {
        "repository": "example-owner/example-workbench", "revision": fixture["source"]["head_sha"],
        "runtime_digest": "sha256:" + fixture["platform"]["ciw"]["package_digest"], "executed": True}
    assert claim["basis"]["notes"]["artifact"] == fixture["source"]["artifact"]
    assert _finding(report, RECORD_CLAIM)["value"] == {"record": RECORD_ID, "retained_files": 3, "problems": 0,
                                                        "current_entries": 3, "entries_not_current": {}}
    # The provider tasks' figures were compared neither here (outside the time budget) nor on Windows (their
    # providers were not bound there): a stated gap with its own next steps, so the task stays partial. The CI
    # workflow binds plsr-python, so its next run re-executes that task; csg needs a host with its private checkout.
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "provider_backed"
    assert any(a.startswith("2 figures of 2 tasks are compared neither here nor by the second-platform record: "
                            f"T030 ({PROVIDER_REASON}); T101 ({plsr_reason})") for a in report["unresolved_assumptions"])
    step = report["recommended_next_task"]
    assert step.startswith("Run the second-platform comparison again against this lab/")
    assert "does not compare 1 of this run's figures that the workflow can re-execute there" in step
    assert "Re-execute on Windows the figure tasks that used providers the CI workflow cannot bind (csg):" in step
    assert "scripts/retain_figure_check.py" in step and not re.search(r"\bT\d{3}\b", step)
    identity = report["provider_runtime_identity"]
    assert identity["figure_platform_record"]["state"] == "valid"
    assert identity["figure_platform_record"]["revision"] == fixture["source"]["head_sha"]
    assert identity["requirement_probes"][f"provider:{research_portfolio.FIGURE_PLATFORM_RECORD}"] is True
    assert "No second-platform record" not in report["numerical_result"]
    assert "Windows, Python 3.12.10, NumPy 2.4.3, OpenBLAS kernel Haswell" in report["numerical_result"]
    written = json.loads((run / "artifacts" / "T158" / "figure-platform-record.json").read_text(encoding="utf-8"))
    assert written["coverage"]["artifacts/T030/plot.svg"] == "not re-executed there"
    # With every figure compared here or there, the question is answered: completed, and the next step is research.
    everything = _record(tmp_path / "everything", run)
    report = _run(run, everything)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "provider_backed"
    assert report["recommended_next_task"].startswith("Compare the figures on a third platform")


@pytest.mark.lab_task("T158")
@pytest.mark.parametrize("defect, problem", [
    ("tampered", "integrity: figure-check.json differs from its manifest digest"),
    ("same platform", "platform: the record was made on Windows, the operating system of this run, so it is no "
                      "second platform"),
    ("missing", "integrity: the record directory does not exist")])
def test_t158_refuses_a_tampered_or_same_platform_record_by_name(tmp_path, monkeypatch, defect, problem):
    run = tmp_path / "run"
    _retain(run, monkeypatch, {"T010": _figure_task("T010")}, regenerated=("T010",))
    record = _record(tmp_path / "records", run)
    if defect == "tampered":
        path = record / "figure-check.json"
        path.write_bytes(path.read_bytes().replace(b"Synthetic", b"Edited"))
    elif defect == "same platform":
        monkeypatch.setattr(records, "host_system", lambda: "Windows")
    else:
        shutil.rmtree(record)
    report = _run(run, record)
    refused = _finding(report, RECORD_CLAIM)
    assert refused["evidence_status"] == "not_established" and problem in refused["value"]["first_problems"]
    claim = _finding(report, PLATFORM_CLAIM)
    assert claim["evidence_status"] == "not_established" and claim["expected_not_established"] is True
    assert claim["value"].startswith(f"not established: the bound second-platform record {RECORD_ID} is refused")
    # The comparison on this platform stands; the refused record refutes the report's claim to a second platform.
    assert _labels(report)[FIGURES] == "numerically_verified"
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert report["recommended_next_task"].startswith("Retain a valid second-platform record")
    assert report["provider_runtime_identity"]["figure_platform_record"] == {"state": "refused", "record_id": RECORD_ID}


@pytest.mark.lab_task("T158")
def test_t158_counts_only_record_entries_whose_figure_is_this_runs(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _retain(run, monkeypatch, {"T010": _figure_task("T010"), "T013": _figure_task("T013", timing=True),
                               "T020": _figure_task("T020", rounding=True), "T023": _figure_task("T023"),
                               "T025": _figure_task("T025")})

    def other_kernel(entry):
        # Made against the same figure on another kernel: other bytes, values within their rounding bounds.
        entry["retained_sha256"] = "e" * 64
        entry["retained_values"][0]["y"] = [y + 0.5 * BOUND for y in entry["retained_values"][0]["y"]]

    record = _record(tmp_path / "records", run, outcomes={"T010": "differs"}, edits={
        "T010": lambda e: e.update(retained_sha256="0" * 64),         # made against another T010 figure
        "T013": lambda e: e.pop("retained_structure"),                  # a record older than the declared identities
        "T020": other_kernel, "T025": lambda e: None})
    _edit(record / "figure-check.json", lambda c: c.update(figures=[f for f in c["figures"] if f["task_id"] != "T025"]))
    _edit(record / "figure-check.json", lambda c: c.update(summary=_summary(c)))
    _reseal(record)
    report = _run(run, record)
    value = _finding(report, PLATFORM_CLAIM)["value"]
    # Only T020 (matched by its values, on any kernel) and T023 (by its digest) are this run's figures; the stale T010
    # entry's mismatch refutes nothing.
    assert value["compared"] == 2 and value["mismatched"] == 0
    assert value["outcomes"] == {"identical": 1, "within rounding bounds": 1}
    assert value["figures_not_compared_there"] == {"not in the record": 1, "record entry not current": 2}
    assert _finding(report, RECORD_CLAIM)["value"]["entries_not_current"] == {
        "no kernel-independent identity recorded": 1, "retained digest differs": 1}
    assert _labels(report)[PLATFORM_CLAIM] == "provider_backed"
    # T013 was re-executed here; T010 and T025 were compared neither here nor there.
    assert report["state"] == "partial"
    assert report["recommended_next_task"].startswith("Run the second-platform comparison again against this lab/")
    assert "does not compare 2 of this run's figures" in report["recommended_next_task"]
    # A rounding-level entry whose values moved beyond their rounding bounds is another figure's: not current.
    moved = _record(tmp_path / "moved", run, edits={
        "T020": lambda e: e["retained_values"][0].update(y=[1.0, 4.0 + 1e-6])})
    assert _finding(_run(run, moved), RECORD_CLAIM)["value"]["entries_not_current"] == {
        "retained values differ beyond their rounding bounds": 1}
    # An entry regenerated there by other code than this run's is not current whatever its figure: its task's sources
    # differ from those this run's report records (the task changed since the record, its figure bytes did not), or
    # the record names none.
    recoded = _record(tmp_path / "recoded", run, edits={
        "T023": lambda e: e["task_sources"].update({"src/ciw/lab/svg.py": "f" * 64}),
        "T025": lambda e: e.pop("task_sources")})
    report = _run(run, recoded)
    assert _finding(report, RECORD_CLAIM)["value"]["entries_not_current"] == {
        "no task sources recorded": 1, "task sources differ from the record's": 1}
    assert _finding(report, PLATFORM_CLAIM)["value"]["compared"] == 3


@pytest.mark.lab_task("T158")
def test_t158_is_refuted_by_a_mismatch_the_record_reports(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _retain(run, monkeypatch, {"T010": _figure_task("T010"), "T020": _figure_task("T020", rounding=True)},
            regenerated=("T010", "T020"))
    record = _record(tmp_path / "records", run, outcomes={"T020": "values differ"})
    assert (record / "fresh" / "artifacts" / "T020" / "plot.svg").is_file()
    report = _run(run, record)
    claim = _finding(report, PLATFORM_CLAIM)
    assert claim["evidence_status"] == "not_established" and not claim.get("expected_not_established")
    assert claim["value"]["mismatched"] == 1 and claim["value"]["outcomes"] == {"identical": 1, "values differ": 1}
    assert _labels(report)[RECORD_CLAIM] == "numerically_verified" and _labels(report)[FIGURES] == "numerically_verified"
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert report["recommended_next_task"].startswith(
        "Fix the figures the second-platform record reports as mismatched on Windows")


@pytest.mark.lab_task("T158")
def test_t158_names_what_keeps_it_partial_before_a_third_platform(tmp_path, monkeypatch):
    """With every figure compared by the record, a third platform is the next step only once the task completes."""
    run = tmp_path / "run"
    _retain(run, monkeypatch, {"T010": _figure_task("T010"),
                               "T023": _figure_task("T023", states=("completed", "partial"))},
            regenerated=("T010", "T023"))
    report = _run(run, _record(tmp_path / "records", run))
    # T023's re-execution here ended partial: its figure is not compared here, so the task stays partial.
    assert _finding(report, PLATFORM_CLAIM)["value"]["compared"] == 2 and report["state"] == "partial"
    step = report["recommended_next_task"]
    assert step.startswith("Compare the figures of the re-executed tasks that ended in another state here than in "
                           "their retained reports") and "third platform" not in step
    # A figure that mismatched when re-executed here refutes the comparison here: its fix comes first.
    run = tmp_path / "mismatched"
    _retain(run, monkeypatch, {"T010": _figure_task("T010"), "T023": _figure_task("T023", varying=True)},
            regenerated=("T010", "T023"))
    report = _run(run, _record(tmp_path / "mismatched-records", run))
    assert _labels(report)[FIGURES] == "not_established" and report["state"] == "partial"
    step = report["recommended_next_task"]
    assert step.startswith("Fix the figures that mismatched when re-executed in this run")
    assert research_portfolio.DECLARE_ONLY in step and "third platform" not in step


@pytest.mark.lab_task("T158")
def test_t158_without_a_record_keeps_its_comparison_and_says_so(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _retain(run, monkeypatch, {"T010": _figure_task("T010")}, regenerated=("T010",))
    report = _run(run)
    assert RECORD_CLAIM not in _labels(report) and PLATFORM_CLAIM not in _labels(report)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"
    assert report["numerical_result"].endswith("No second-platform record is bound.")
    assert any(a.startswith("Byte identity is established on this platform and kernel only: no second-platform "
                            "record is bound") for a in report["unresolved_assumptions"])
    identity = report["provider_runtime_identity"]
    assert "figure_platform_record" not in identity
    assert identity["requirement_probes"][f"provider:{research_portfolio.FIGURE_PLATFORM_RECORD}"] is False
    step = report["recommended_next_task"]
    assert step.startswith("Compare the figures on Windows, the second platform")
    assert "retain the run's figure-check.json as a second-platform record with scripts/retain_figure_check.py" in step
    assert not (run / "artifacts" / "T158" / "figure-platform-record.json").exists()


def test_t158_registers_the_record_tests_in_this_file():
    from ciw.lab import registry
    nodes = [node for node in registry._REGISTRY["T158"].regression_tests
             if node.startswith(research_portfolio.RECORD_TESTS)]
    assert len(nodes) == 6
    for node in nodes:
        assert node.split("::")[1] in globals(), node


def test_a_bound_figure_platform_record_verifies():
    """The record the clean-room gate binds (CIW_LAB_FIGURE_PLATFORM_RECORD) verifies and names another OS."""
    bound = os.environ.get("CIW_LAB_FIGURE_PLATFORM_RECORD")
    if not bound:
        pytest.skip("no figure-platform record is bound")
    inspected = records.inspect_record(Path(bound))
    assert inspected["problems"] == []


# ---------------------------------------------------------------- repository files (skipped in the clean room)

def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        pytest.skip("scripts/ is not beside the tests")
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"lab_script_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_the_repository_figure_platform_records_verify():
    if not (ROOT / "lab").is_dir():
        pytest.skip("lab/ is not beside the tests")
    result = records.verify_records(ROOT / "lab")
    assert result["passed"], result["problems"]
    for directory in records.record_directories(ROOT / "lab"):
        for name in ("record.json", "manifest.json"):
            assert runner._host_paths(json.loads((directory / name).read_text(encoding="utf-8")), name) == []


def test_the_retention_script_retains_an_artifact_zip(tmp_path, monkeypatch, capsys):
    retain = _script("retain_figure_check")
    digest = _zip(tmp_path / "artifact.zip", {"figure-check.json": (FIXTURE / "figure-check.json").read_bytes(),
                                              "figure-check.md": (FIXTURE / "figure-check.md").read_bytes()})
    arguments = ["retain_figure_check.py", str(tmp_path / "artifact.zip"), "--retained", str(tmp_path / "lab"),
                 "--repository", PROVENANCE["repository"], "--workflow", PROVENANCE["workflow"], "--run-id", "1001",
                 "--head-sha", "a" * 40, "--artifact-id", "2002", "--artifact-name", "figure-check-windows"]
    monkeypatch.setattr(sys, "argv", arguments + ["--artifact-digest", "sha256:" + "0" * 64])
    with pytest.raises(SystemExit, match="not the declared artifact digest"):
        retain.main()
    monkeypatch.setattr(sys, "argv", arguments + ["--artifact-digest", digest, "--date", "2026-09-25"])
    assert retain.main() == 0
    assert json.loads(capsys.readouterr().out)["record_id"] == RECORD_ID
    assert records.verify_records(tmp_path / "lab")["records"][0]["date"] == "2026-09-25"


def test_the_figure_check_script_writes_what_the_record_reads():
    check = _script("check_figures")
    assert check.RECORD_SCHEMA == records.CHECK_SCHEMA and check.NOT_RETAINED == records.NOT_RETAINED
    assert records.OUTCOMES["undeclared"][:2] == ("identical", "differs")
    # The Windows job binds its own interpreter, where the plsr extra is installed, as plsr-python.
    assert check.bindings(["plsr-python=@python", "csg=checkouts/csg"]) == {
        "plsr-python": Path(sys.executable), "csg": Path("checkouts/csg")}
    workflow = ROOT / ".github" / "workflows" / "figures.yml"
    if not workflow.is_file():
        pytest.skip(".github/workflows is not beside the tests")
    text = workflow.read_text(encoding="utf-8")
    # The artifact the retention script reads: both text files at the root of the upload.
    assert "name: figure-check-windows" in text and "results/figures-windows/figure-check.json" in text
    assert "results/figures-windows/figure-check.md" in text and "runs-on: windows-latest" in text
    # The providers T158 counts on the workflow to bind are the ones it binds.
    assert 'pip install -e ".[dev,lab,plsr]"' in text and "--provider plsr-python=@python" in text
    assert re.findall(r"--provider ([a-z-]+)=", text) == list(research_portfolio.FIGURES_WORKFLOW_PROVIDERS)


def test_lab_scripts_bind_and_preserve_the_figure_platform_record(tmp_path):
    check, reproduce, refresh = _script("check_lab"), _script("reproduce_lab"), _script("refresh_lab")
    root = tmp_path / records.DIRECTORY
    assert check.figure_platform_record(root) is None
    for name, date, run_id in (("windows-9", "2026-09-24", 9), ("windows-10", "2026-09-24", 10),
                               ("windows-8", "2026-09-01", 8)):
        (root / name).mkdir(parents=True)
        _write(root / name / "record.json", {"date": date, "source": {"run_id": run_id}})
    (root / "README.md").write_text("not a record", encoding="utf-8")
    # The latest by date, then by CI run id (not by name: windows-9 sorts after windows-10).
    assert check.figure_platform_record(root) == root / "windows-10"
    # Then by run attempt, a record without one being the first (not by name: windows-10-9 sorts after windows-10-10).
    for attempt in (9, 10):
        (root / f"windows-10-{attempt}").mkdir()
        _write(root / f"windows-10-{attempt}" / "record.json",
               {"date": "2026-09-24", "source": {"run_id": 10, "run_attempt": attempt}})
    assert check.figure_platform_record(root) == root / "windows-10-10"
    role = research_portfolio.FIGURE_PLATFORM_RECORD
    assert reproduce.TEST_VARIABLES[role] == "CIW_LAB_FIGURE_PLATFORM_RECORD"
    assert "CIW_LAB_FIGURE_PLATFORM_RECORD" in reproduce.INHERITED_EXCLUDED
    assert records.DIRECTORY in refresh.PRESERVED and records.DIRECTORY not in refresh.RETAINED


def test_check_lab_binds_the_latest_figure_platform_record(tmp_path, monkeypatch):
    check = _script("check_lab")
    commands = []
    monkeypatch.setattr(check, "call", lambda command, **kwargs: commands.append([str(part) for part in command]))
    monkeypatch.setattr(check, "validate_checkout", lambda path, revision: Path(path))
    monkeypatch.setattr(check, "figure_platform_record", lambda: tmp_path / "lab" / records.DIRECTORY / RECORD_ID)
    monkeypatch.setattr(sys, "argv", ["check_lab.py", "--no-compare", "--stack-root", str(tmp_path / "stack"),
                                      "--output-dir", str(tmp_path / "out")])
    assert check.main() == 0
    command, = commands
    assert f"figure-platform-record={tmp_path / 'lab' / records.DIRECTORY / RECORD_ID}" in command


def test_refresh_keeps_figure_platform_records_and_requires_their_binding(tmp_path, monkeypatch):
    refresh = _script("refresh_lab")
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    kept = _copy(tmp_path / "lab") / "record.json"
    before = kept.read_bytes()
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "artifacts").mkdir()
    (run / "reports" / "T001.json").write_text(json.dumps({"task_id": "T001"}), encoding="utf-8")
    (run / "reports" / "T077.json").write_text(json.dumps({"task_id": "T077", "provider_runtime_identity": {
        "telemetry-stack": {"gsie": {"state": "ready"}}, "executed_runtimes": {"gsie": {}}}}), encoding="utf-8")
    for name in ("queue-state.json", "REPORTS.md", "index.html"):
        (run / name).write_text("", encoding="utf-8")
    gate = {"schema": "ciw.lab-clean-room-gate.v1", "python": "3.12.3",
            "providers": [f"{role}=/p" for role in refresh.REQUIRED_PROVIDERS]}
    (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["refresh_lab.py", "--from-run", str(run)])
    with pytest.raises(SystemExit, match="figure-platform-record"):
        refresh.main()
    gate["providers"].append("figure-platform-record=/r")
    (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    assert refresh.main() == 0
    assert kept.read_bytes() == before
