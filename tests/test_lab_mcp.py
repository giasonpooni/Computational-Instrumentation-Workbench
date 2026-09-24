import asyncio
import json
import threading
import time

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from ciw.lab import mcp_server, planner, runner  # noqa: E402
from ciw.lab.evidence import finding  # noqa: E402
from ciw.lab.mcp_server import build_server, serve  # noqa: E402
from ciw.lab.registry import Implementation, load_queue  # noqa: E402
from ciw.lab.report import build_report  # noqa: E402

EXPECTED = {"ciw_lab_list_tasks", "ciw_lab_get_report", "ciw_lab_plan_next", "ciw_lab_run_tasks",
            "ciw_lab_verify_run", "ciw_lab_classify_workspace", "ciw_lab_explain_labels"}


def _call(server, name, arguments=None):
    async def go():
        async with Client(server) as client:
            result = await client.call_tool(name, arguments or {})
            return result.is_error, "".join(part.text for part in result.content if hasattr(part, "text"))
    return asyncio.run(go())


def test_tools_are_declared_and_no_tool_accepts_labels(tmp_path):
    server = build_server(None, tmp_path / "work")

    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools
    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == EXPECTED
    for tool in tools.values():
        properties = set(tool.input_schema.get("properties", {}))
        assert not properties & {"label", "labels", "evidence_status", "finding", "findings", "report", "basis"}
    assert tools["ciw_lab_run_tasks"].annotations.read_only_hint is False
    assert tools["ciw_lab_run_tasks"].annotations.destructive_hint is True   # replaces reports, deletes artifacts
    assert all(tools[name].annotations.read_only_hint for name in EXPECTED - {"ciw_lab_run_tasks"})


def test_run_read_plan_and_verify_through_mcp(tmp_path):
    retained = tmp_path / "retained"
    runner.run_queue(retained, ["T156"])
    server = build_server(retained, tmp_path / "work")
    error, text = _call(server, "ciw_lab_list_tasks", {"section": "research-portfolio", "response_format": "json"})
    listing = json.loads(text)
    assert not error and listing["total"] == 14 and listing["items"][1]["task_id"] == "T156"
    error, text = _call(server, "ciw_lab_run_tasks", {"task_ids": ["T156"]})
    assert not error and "T156" in text and "physical validation not_established" in text
    error, text = _call(server, "ciw_lab_get_report", {"task_id": "T156"})
    assert not error and "**Evidence status:**" in text and str(tmp_path / "work") in text
    error, text = _call(server, "ciw_lab_verify_run")
    assert not error and json.loads(text)["passed"] is True
    error, text = _call(server, "ciw_lab_plan_next", {"limit": 3, "response_format": "json"})
    assert not error and json.loads(text)["schema"] == "ciw.lab-next.v2"
    error, text = _call(server, "ciw_lab_explain_labels")
    assert "independently_verified" in text and "Machine safety" in text


def test_errors_are_actionable(tmp_path):
    server = build_server(None, tmp_path / "work")
    error, text = _call(server, "ciw_lab_run_tasks", {"task_ids": ["T1"]})
    assert error and "T003" in text
    error, text = _call(server, "ciw_lab_get_report", {"task_id": "T010"})
    assert error and "ciw_lab_run_tasks" in text
    error, text = _call(server, "ciw_lab_verify_run")
    assert error and "retained" in text
    error, text = _call(server, "ciw_lab_list_tasks", {"limit": 0})
    assert error and "limit" in text


def test_stdio_entry_point_serves_the_queue(tmp_path):
    import os
    import sys
    from pathlib import Path

    from mcp import StdioServerParameters

    source = str(Path(__file__).resolve().parents[1] / "src")
    environment = {**os.environ, "PYTHONPATH": source + os.pathsep + os.environ.get("PYTHONPATH", "")}
    parameters = StdioServerParameters(command=sys.executable, env=environment,
                                       args=["-m", "ciw", "lab", "mcp", "--workdir", str(tmp_path / "work")])

    async def go():
        async with Client(parameters) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool("ciw_lab_list_tasks", {"limit": 2, "response_format": "json"})
            return names, json.loads(result.content[0].text)
    names, listing = asyncio.run(go())
    assert names == EXPECTED and listing["total"] == 168 and listing["has_more"] is True


def test_evaluation_answers_hold_through_the_tools(tmp_path):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    pairs = ET.parse(Path(__file__).resolve().parents[1] / "docs" / "lab" / "mcp_evaluation.xml").getroot()
    answers = [pair.find("answer").text for pair in pairs.iter("qa_pair")]
    server = build_server(None, tmp_path / "work")
    tasks = []
    for offset in (0, 50, 100, 150):
        error, text = _call(server, "ciw_lab_list_tasks", {"offset": offset, "limit": 50, "response_format": "json"})
        assert not error
        tasks += json.loads(text)["items"]
    assert len(tasks) == 168
    titles = {t["task_id"]: t for t in tasks}
    _, labels = _call(server, "ciw_lab_explain_labels")
    derived = [
        next(t["task_id"] for t in tasks if "filter covariance" in t["title"]),
        str(sum(t["section"] == "exchange-provenance" for t in tasks)),
        next(f"{t['section']}:{t['task_id']}" for t in tasks if "cargo build --locked" in t["title"]),
        str(sum("FPGA" in t["title"] for t in tasks)),
        "independently_verified" if "``independently_verified``\n    Agreement between two computations whose implementations have distinct" in labels else "?",
        [t["task_id"] for t in tasks if t["section"] == "flat-torus-topology"][-1],
        "T%03d" % (int(next(t["task_id"] for t in tasks if "glued edges" in t["title"])[1:]) + 1),
        "Industrial readiness" if "| GPU/CPU agreement | Industrial readiness |" in labels else "?",
        str(sum(t["section"] == "energy-gpu" and ("GPU" in t["title"] or "RTX" in t["title"]) for t in tasks)),
        next(t["task_id"] for t in tasks if "counterexample" in t["title"].lower()
             and t["section"] not in ("geodesic-jacobi", "flat-torus-topology")),
    ]
    assert derived == answers and titles["T029"]["title"].startswith("Detect cone")


CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}


def _retain(directory, task_id, state, value=1.0, **fields):
    """Write a valid report without running the task."""
    task = {t["id"]: t for t in load_queue()["tasks"]}[task_id]
    record = finding("claim", "numerical", value, {"generator": {"name": "g"}, "checks": [CHECK]})
    report = build_report(task, state, {"recommended_next_task": f"after {task_id}", "experiment": "e", **fields},
                          [record] if state in ("completed", "partial") else [])
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(report))
    return report


def test_work_directory_must_be_separate_from_retained(tmp_path):
    retained = tmp_path / "retained"
    _retain(retained, "T002", "completed")
    (tmp_path / "alias").symlink_to(retained)
    for workdir in (retained, retained / "work", tmp_path, tmp_path / "alias", tmp_path / "alias" / "work"):
        with pytest.raises(ValueError, match="separate"):
            build_server(retained, workdir)
    with pytest.raises(ValueError, match="separate"):
        serve(retained, retained / "work")
    assert not (retained / "work").exists()
    build_server(retained, tmp_path / "work")


def test_list_tasks_names_valid_states_and_sections(tmp_path):
    server = build_server(None, tmp_path / "work")
    error, text = _call(server, "ciw_lab_list_tasks", {"state": "complete"})
    assert error and "'completed'" not in text and "completed, partial, blocked, deferred, not_run" in text
    error, text = _call(server, "ciw_lab_list_tasks", {"section": "geodesics"})
    assert error and "geodesic-jacobi" in text and "research-portfolio" in text
    error, text = _call(server, "ciw_lab_list_tasks", {"state": "not_run", "section": "research-portfolio",
                                                       "response_format": "json"})
    assert not error and json.loads(text)["total"] == 14


@pytest.fixture
def small_registry(monkeypatch):
    fakes = {tid: Implementation(tid, None, requires=requires) for tid, requires in (
        ("T001", ()), ("T002", ()), ("T003", ()), ("T004", ("hardware:no-such-device",)), ("T005", ()))}
    monkeypatch.setattr(planner, "load_implementations", lambda: (fakes, {}))


def test_plan_next_merges_the_run_directory_over_retained_reports(tmp_path, small_registry):
    retained, work = tmp_path / "retained", tmp_path / "work"
    _retain(retained, "T002", "partial")
    _retain(retained, "T003", "completed")
    _retain(retained, "T005", "blocked", unresolved_assumptions=["Blocked by unexpected TimeoutError"])
    server = build_server(retained, work)
    expected = {"T001": "ready", "T005": "retry", "T002": "refinement", "T003": "research"}
    error, text = _call(server, "ciw_lab_plan_next", {"response_format": "json"})
    assert not error and {r["task_id"]: r["kind"] for r in json.loads(text)["next"]} == expected
    _retain(work, "T002", "completed")                      # a run through the server completes T002
    error, text = _call(server, "ciw_lab_plan_next", {"response_format": "json"})
    plan = json.loads(text)
    assert {r["task_id"]: r["kind"] for r in plan["next"]} == dict(expected, T002="research")
    assert plan["retained"] == [str(work), str(retained)]
    assert [r["task_id"] for r in plan["still_blocked"]] == ["T004"]


def _verify(server):
    error, text = _call(server, "ciw_lab_verify_run")
    return error, (text if error else json.loads(text))


@pytest.mark.parametrize("accepts_tasks", [False, True, None])
def test_verify_run_compares_only_tasks_in_the_run_directory(tmp_path, monkeypatch, accepts_tasks):
    retained, work = tmp_path / "retained", tmp_path / "work"
    _retain(retained, "T002", "completed")
    _retain(retained, "T003", "partial")
    _retain(work, "T002", "completed")
    passed = []
    if accepts_tasks is False:      # compare(retained, fresh): problems of other tasks are filtered out
        monkeypatch.setattr(mcp_server, "compare", lambda retained_dir, fresh_dir: runner.compare(retained_dir, fresh_dir))
    elif accepts_tasks is True:     # compare(retained, fresh, tasks=...): the subset is passed through
        def compare(retained_dir, fresh_dir, tasks=None):
            passed.append(tasks)
            return runner.compare(retained_dir, fresh_dir)
        monkeypatch.setattr(mcp_server, "compare", compare)
    server = build_server(retained, work)
    error, result = _verify(server)
    assert not error and result["passed"] is True and result["problems"] == []
    assert result["tasks"] == ["T002"] and result["not_regenerated"] == ["T003"] and result["compared"] == 1
    assert passed == ([["T002"]] if accepts_tasks else [])
    _retain(work, "T002", "completed", value=2.0)
    error, result = _verify(server)
    assert not error and result["passed"] is False
    assert [p.split(":")[0] for p in result["problems"]] == ["T002"] and "tolerance" in result["problems"][0]
    _retain(work, "T001", "completed")
    error, result = _verify(server)
    assert any(p.startswith("T001: not retained") for p in result["problems"])


def test_verify_run_refuses_empty_directories(tmp_path):
    retained, work = tmp_path / "retained", tmp_path / "work"
    (retained / "reports").mkdir(parents=True)
    error, text = _call(build_server(retained, work), "ciw_lab_verify_run")
    assert error and "holds no reports" in text
    _retain(retained, "T002", "completed")
    error, text = _call(build_server(retained, work), "ciw_lab_verify_run")
    assert error and "ciw_lab_run_tasks" in text


def test_tools_touching_the_work_directory_are_serialized(tmp_path, monkeypatch):
    work = tmp_path / "work"
    report = _retain(tmp_path / "source", "T002", "completed")
    text = runner.dumps(report)
    started, active, overlap = threading.Event(), [0], []

    def slow_run_queue(output_dir, task_ids, providers=None):
        active[0] += 1
        overlap.append(active[0])
        path = output_dir / "reports" / "T002.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text[: len(text) // 2])             # mid-write, as a non-atomic writer leaves it
        started.set()
        time.sleep(0.4)
        path.write_text(text)
        active[0] -= 1
        return {"tasks": 1, "states": {"completed": 1}}

    monkeypatch.setattr(mcp_server, "run_queue", slow_run_queue)
    server = build_server(None, work)

    async def go():
        async with Client(server) as client:
            runs = [asyncio.create_task(client.call_tool("ciw_lab_run_tasks", {"task_ids": ["T002"]})) for _ in range(2)]
            while not started.is_set():
                await asyncio.sleep(0.01)
            listing = await client.call_tool("ciw_lab_list_tasks", {"state": "completed", "response_format": "json"})
            return listing, await asyncio.gather(*runs)
    listing, runs = asyncio.run(go())
    assert not listing.is_error and json.loads(listing.content[0].text)["items"][0]["task_id"] == "T002"
    assert not any(result.is_error for result in runs) and max(overlap) == 1


def test_a_run_directory_sharing_files_with_retained_is_refused(tmp_path):
    import os

    retained = tmp_path / "retained"
    before = _retain(retained, "T002", "completed")
    linked, hard = tmp_path / "linked", tmp_path / "hard"
    linked.mkdir()
    (linked / "reports").symlink_to(retained / "reports")              # a run would write through the link
    (hard / "reports").mkdir(parents=True)
    os.link(retained / "reports" / "T002.json", hard / "reports" / "T002.json")   # as `cp -al` leaves it
    for workdir in (linked, hard):
        with pytest.raises(ValueError, match="separate"):
            build_server(retained, workdir)
    work = tmp_path / "work"
    server = build_server(retained, work)
    (work / "reports").mkdir(parents=True)
    os.link(retained / "reports" / "T002.json", work / "reports" / "T002.json")   # linked after startup
    error, text = _call(server, "ciw_lab_run_tasks", {"task_ids": ["T002"]})
    assert error and "separate" in text
    assert json.loads((retained / "reports" / "T002.json").read_text()) == before


def test_a_second_mount_point_of_retained_is_refused(tmp_path):
    import subprocess

    retained, alias = tmp_path / "retained", tmp_path / "alias"
    _retain(retained, "T002", "completed")
    alias.mkdir()
    try:
        mounted = subprocess.run(["mount", "--bind", str(retained), str(alias)], capture_output=True).returncode == 0
    except OSError:
        mounted = False
    if not mounted:
        pytest.skip("bind mounts need privileges")
    try:
        with pytest.raises(ValueError, match="separate"):
            build_server(retained, alias)
        with pytest.raises(ValueError, match="separate"):
            build_server(retained, alias / "work")
    finally:
        subprocess.run(["umount", str(alias)], check=True)


def test_tools_are_serialized_across_server_processes(tmp_path):
    # Two clients may each start a server on the documented fixed work directory.
    import os
    import subprocess
    import sys
    from pathlib import Path

    work, source = tmp_path / "work", tmp_path / "T002.json"
    source.write_text(runner.dumps(_retain(tmp_path / "source", "T002", "completed")))
    holder = ("import sys, time\nfrom pathlib import Path\nfrom ciw.lab.mcp_server import workdir_lock\n"
              "work, text = Path(sys.argv[1]), Path(sys.argv[2]).read_text()\n"
              "with workdir_lock(work):\n"
              "    path = work / 'reports' / 'T002.json'\n"
              "    path.parent.mkdir(parents=True, exist_ok=True)\n"
              "    path.write_text(text[: len(text) // 2])\n"
              "    print('locked', flush=True)\n"
              "    time.sleep(0.5)\n"
              "    path.write_text(text)\n")
    src = str(Path(__file__).resolve().parents[1] / "src")
    other = subprocess.Popen([sys.executable, "-c", holder, str(work), str(source)], stdout=subprocess.PIPE,
                             env={**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")})
    try:
        assert other.stdout.readline().strip() == b"locked"
        error, text = _call(build_server(None, work), "ciw_lab_list_tasks", {"state": "completed", "response_format": "json"})
    finally:
        other.wait(timeout=30)
    assert not error and json.loads(text)["items"][0]["task_id"] == "T002"


def test_plan_next_marks_hardware_runs_and_counts_research(tmp_path, small_registry, monkeypatch):
    retained = tmp_path / "retained"
    _retain(retained, "T003", "completed", recommended_next_task="T002: refine; then bind a signed capture")
    _retain(retained, "T004", "blocked")
    measured = _retain(tmp_path / "hardware-run", "T004", "completed", recommended_next_task="T005 (compare)")
    entry = {"run_id": "rtx2080-2026-10-01", "date": "2026-10-01", "host": "h", "state": "completed",
             "evidence_status": "numerically_verified", "physical_validation_status": "not_established",
             "counts": {}, "hardware_measured": 0, "report": measured}
    monkeypatch.setattr(planner, "latest_hardware", lambda directories, problems: {"T004": entry})
    server = build_server(retained, tmp_path / "work")
    error, text = _call(server, "ciw_lab_plan_next", {"limit": 50})
    assert not error
    # T004 is blocked in the main run and ranked from its hardware run, which the row names.
    assert "- T004 (follow_up -> T005) [hardware run rtx2080-2026-10-01]:" in text
    assert "- T003 (research):" in text and "bind a signed capture" in text
    assert "follow-ups: 2; research questions: 1; stale pointers: 0" in text
    # A follow-up is listed whatever the limit, however many rows rank above it.
    error, text = _call(server, "ciw_lab_plan_next", {"limit": 1})
    assert not error and "- T004 (follow_up -> T005)" in text and "- T003 (follow_up -> T002)" in text
    error, text = _call(server, "ciw_lab_plan_next", {"response_format": "json"})
    plan = json.loads(text)
    assert "T004" not in [r["task_id"] for r in plan["still_blocked"]] and plan["research"][0]["task_id"] == "T003"
