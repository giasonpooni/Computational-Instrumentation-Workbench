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


def test_run_log_keeps_timing_out_of_reports(tmp_path):
    summary = runner.run_queue(tmp_path, ["T156"], budget_seconds=0.0)
    log = json.loads((tmp_path / "run-log.json").read_text())
    assert log["tasks"][0]["task_id"] == "T156" and log["tasks"][0]["seconds"] >= 0
    assert summary["over_budget"] and summary["slowest"][0]["task_id"] == "T156"
    assert "seconds" not in (tmp_path / "reports" / "T156.json").read_text()


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
