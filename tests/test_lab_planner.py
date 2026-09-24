import json

import pytest

from ciw import cli
from ciw.lab import planner, runner
from ciw.lab.evidence import finding
from ciw.lab.registry import Implementation, load_implementations, load_queue
from ciw.lab.report import build_report

CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}


def _retain(directory, task_id, state, **fields):
    task = {t["id"]: t for t in load_queue()["tasks"]}[task_id]
    record = finding("claim", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]})
    built = build_report(task, state, {"recommended_next_task": f"after {task_id}", "experiment": "e", **fields},
                         [record] if state in ("completed", "partial") else [])
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(built))


@pytest.fixture
def fake_registry(monkeypatch):
    fakes = {tid: Implementation(tid, None, requires=requires) for tid, requires in (
        ("T001", ()), ("T002", ()), ("T003", ()), ("T004", ("hardware:no-such-device",)),
        ("T005", ("tool:python",)), ("T006", ()))}
    monkeypatch.setattr(planner, "load_implementations", lambda: (fakes, {"lyapunov": "section module not implemented"}))
    return fakes


def test_next_ranks_ready_unblocked_refinement_then_follow_up(tmp_path, fake_registry):
    _retain(tmp_path, "T002", "completed")
    _retain(tmp_path, "T003", "partial")
    _retain(tmp_path, "T004", "blocked")
    _retain(tmp_path, "T005", "blocked")
    _retain(tmp_path, "T006", "deferred")
    plan = planner.next_tasks(tmp_path)
    kinds = [(row["task_id"], row["kind"]) for row in plan["next"]]
    assert kinds == [("T001", "ready"), ("T006", "ready"), ("T005", "unblocked"),
                     ("T003", "refinement"), ("T002", "follow_up")]
    assert plan["still_blocked"][0]["task_id"] == "T004" and "no-such-device" in plan["still_blocked"][0]["reason"]
    assert plan["next"][3]["reason"] == "after T003"
    assert "T168" in plan["unimplemented"] and plan["section_import_errors"]
    assert len(planner.next_tasks(tmp_path, limit=2)["next"]) == 2


def test_next_without_retained_reports_lists_ready_tasks(fake_registry):
    plan = planner.next_tasks(None)
    assert [row["task_id"] for row in plan["next"]] == ["T001", "T002", "T003", "T005", "T006"]
    assert [(row["task_id"], row["state"]) for row in plan["still_blocked"]] == [("T004", "not_run")]


def test_blocked_without_a_declared_requirement_is_proposed_for_retry(tmp_path, fake_registry):
    # A provider timeout or a self-checked precondition leaves no unmet requirement to wait for.
    _retain(tmp_path, "T001", "blocked", unresolved_assumptions=["Blocked by unexpected TimeoutError: provider"])
    _retain(tmp_path, "T004", "blocked")
    plan = planner.next_tasks(tmp_path)
    retry = [row for row in plan["next"] if row["kind"] == "retry"]
    assert [row["task_id"] for row in retry] == ["T001"] and "TimeoutError: provider" in retry[0]["reason"]
    assert [row["task_id"] for row in plan["still_blocked"]] == ["T004"]


def test_retry_ranks_after_unblocked_and_before_refinement(tmp_path, fake_registry):
    _retain(tmp_path, "T001", "blocked", unresolved_assumptions=["Blocked by unexpected OSError: reset"])
    _retain(tmp_path, "T003", "partial")
    _retain(tmp_path, "T005", "blocked")
    kinds = [(row["task_id"], row["kind"]) for row in planner.next_tasks(tmp_path)["next"]]
    assert kinds[-3:] == [("T005", "unblocked"), ("T001", "retry"), ("T003", "refinement")]


def test_next_merges_directories_with_the_first_winning(tmp_path, fake_registry):
    work, retained = tmp_path / "work", tmp_path / "retained"
    _retain(retained, "T002", "completed")
    _retain(retained, "T003", "partial")
    _retain(retained, "T006", "blocked", unresolved_assumptions=["Blocked by unexpected OSError: reset"])
    _retain(work, "T003", "completed")
    plan = planner.next_tasks([work, retained])
    kinds = {row["task_id"]: row["kind"] for row in plan["next"]}
    assert kinds == {"T001": "ready", "T005": "ready", "T006": "retry", "T002": "follow_up", "T003": "follow_up"}
    assert plan["retained"] == [str(work), str(retained)]
    assert planner.next_tasks(retained)["retained"] == str(retained)


def test_a_self_blocked_task_is_retried_only_when_its_outcome_could_differ(tmp_path, fake_registry):
    # Re-running a task whose code and runtime are unchanged reproduces the same block, so
    # proposing it would loop; it stays blocked with its own reason.
    same = runner.builtin_identity(fake_registry["T001"].changed_files)
    reason = "No readable provider checkout is bound."
    _retain(tmp_path, "T001", "blocked", unresolved_assumptions=[reason], provider_runtime_identity=same)
    _retain(tmp_path, "T002", "blocked", unresolved_assumptions=[reason], provider_runtime_identity=dict(
        same, sources={"src/ciw/lab/edited.py": "0" * 64}))
    _retain(tmp_path, "T003", "blocked", unresolved_assumptions=[reason])      # no built-in identity recorded
    plan = planner.next_tasks(tmp_path)
    assert [(row["task_id"], row["kind"]) for row in plan["next"] if row["state"] == "blocked"] == [("T002", "retry")]
    assert "changed" in plan["next"][-1]["reason"] and reason in plan["next"][-1]["reason"]
    blocked = {row["task_id"]: row["reason"] for row in plan["still_blocked"]}
    assert blocked["T001"] == reason and blocked["T003"] == reason


def test_following_the_plan_does_not_loop_on_a_self_blocked_task(tmp_path):
    runner.run_queue(tmp_path, ["T098"])                  # blocked here: no provider checkout is bound
    report = runner.load_reports(tmp_path)[0]
    assert report["state"] == "blocked"
    plan = planner.next_tasks(tmp_path, limit=200)
    assert "T098" not in [row["task_id"] for row in plan["next"]]
    assert "T098" in [row["task_id"] for row in plan["still_blocked"]]


def test_run_log_keeps_timing_out_of_reports(tmp_path):
    summary = runner.run_queue(tmp_path, ["T156"], budget_seconds=0.0)
    log = json.loads((tmp_path / "run-log.json").read_text(encoding="utf-8"))
    assert log["tasks"][0]["task_id"] == "T156" and log["tasks"][0]["seconds"] >= 0
    assert summary["over_budget"] and summary["slowest"][0]["task_id"] == "T156"
    assert "seconds" not in (tmp_path / "reports" / "T156.json").read_text(encoding="utf-8")


def test_reports_satisfy_the_structural_schema(tmp_path):
    pytest.importorskip("jsonschema")
    runner.run_queue(tmp_path, ["T155", "T156", "T116"])
    for report in runner.load_reports(tmp_path):
        assert runner.schema_errors(report) == []
    broken = runner.load_reports(tmp_path)[0]
    broken["findings"].append({"claim": "x"})
    broken["state"] = "done"
    problems = runner.schema_errors(broken)
    assert any("state" in p for p in problems) and any("findings/" in p for p in problems)


def test_cli_next_filters_and_schema(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T156", "--output-dir", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["lab", "queue", "--retained", str(out), "--section", "research-portfolio"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 14 and all(line[:4] in {f"T{n}" for n in range(155, 169)} for line in lines)
    assert cli.main(["lab", "queue", "--retained", str(out), "--state", "not_run"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 167
    assert cli.main(["lab", "next", "--retained", str(out), "--limit", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "ciw.lab-next.v1"
    if pytest.importorskip("jsonschema"):
        assert cli.main(["lab", "report", "T156", "--retained", str(out), "--schema"]) == 0
