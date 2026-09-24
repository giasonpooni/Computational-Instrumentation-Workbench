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


def test_next_ranks_ready_unblocked_refinement_then_research(tmp_path, fake_registry):
    _retain(tmp_path, "T002", "completed")
    _retain(tmp_path, "T003", "partial")
    _retain(tmp_path, "T004", "blocked")
    _retain(tmp_path, "T005", "blocked")
    _retain(tmp_path, "T006", "deferred")
    plan = planner.next_tasks(tmp_path)
    kinds = [(row["task_id"], row["kind"]) for row in plan["next"]]
    # A completed task's next step that names no queue task is an open research question.
    assert kinds == [("T001", "ready"), ("T006", "ready"), ("T005", "unblocked"),
                     ("T003", "refinement"), ("T002", "research")]
    assert plan["still_blocked"][0]["task_id"] == "T004" and "no-such-device" in plan["still_blocked"][0]["reason"]
    assert plan["next"][3]["reason"] == "after T003"
    assert [row["task_id"] for row in plan["research"]] == ["T002"]
    assert "T168" in plan["unimplemented"] and plan["section_import_errors"]
    # The limit cuts the ranking, never a refinement.
    assert [row["task_id"] for row in planner.next_tasks(tmp_path, limit=2)["next"]] == ["T001", "T006", "T003"]


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
    assert kinds == {"T001": "ready", "T005": "ready", "T006": "retry", "T002": "research", "T003": "research"}
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
    assert json.loads(capsys.readouterr().out)["schema"] == "ciw.lab-next.v2"
    if pytest.importorskip("jsonschema"):
        assert cli.main(["lab", "report", "T156", "--retained", str(out), "--schema"]) == 0


def test_next_step_items_keep_questions_and_drop_completed_pointers():
    done = {"T002", "T005", "T046", "T148", "T100"}
    # A colon pointer describes the pointed task; the clause after ';' is the task's own question.
    kept, stale = planner.next_step_items("T002: build references; then propose an upstream subnormal floor",
                                          "T001", done)
    assert kept == [("question", "propose an upstream subnormal floor", None)]
    assert stale == [{"points_to": "T002", "text": "T002: build references"}]
    # Text after a parenthetical is a separate question when joined by a connective, and a variant of the
    # completed experiment when it modifies it; a short modifier is only a pointer.
    kept, stale = planner.next_step_items("T005 (Jacobi separation law) with resolvability-aware step selection, "
                                          "and T046", "T018", done)
    assert kept == [("question", "T005 (Jacobi separation law) with resolvability-aware step selection", "T005")]
    assert [item["points_to"] for item in stale] == ["T046"]
    kept, stale = planner.next_step_items("T002 (frame change) and a fold-scaling study of error", "T011", done)
    assert kept == [("question", "a fold-scaling study of error", None)]
    assert stale == [{"points_to": "T002", "text": "T002 (frame change)"}]
    kept, stale = planner.next_step_items("T147 (compare a ported kernel), then T148 for its reductions", "T142", done)
    assert kept == [("pointer", "T147", "T147 (compare a ported kernel)")] and stale[0]["points_to"] == "T148"
    # A later pointer inside a colon description starts its own clause; pointers inside parentheses never split.
    kept, _ = planner.next_step_items("T148: adopt a policy (fixed tree) and T147 to compare against RTX outputs",
                                      "T121", done)
    assert kept == [("pointer", "T147", "T147 to compare against RTX outputs")]
    kept, stale = planner.next_step_items("Execute the protocols (T126-T128, then T100) on hardware; then T100.",
                                          "T167", done)
    assert kept == [("question", "Execute the protocols (T126-T128, then T100) on hardware", None)]
    assert stale == [{"points_to": "T100", "text": "T100"}]


def test_next_never_proposes_a_completed_pointer_and_deduplicates(tmp_path, fake_registry):
    _retain(tmp_path, "T001", "completed",
            recommended_next_task="Deferred research question: propose an upstream floor for subnormal inputs.")
    _retain(tmp_path, "T002", "completed",
            recommended_next_task="T003: measure orders; then propose an upstream floor for subnormal inputs")
    _retain(tmp_path, "T003", "completed", recommended_next_task="T006 (drift) and T005 (Jacobi separation)")
    _retain(tmp_path, "T004", "blocked", unresolved_assumptions=[
        "Blocked: unavailable requirement(s) hardware:no-such-device.",
        "Deferred research question: bind a signed capture to the device."])
    _retain(tmp_path, "T005", "partial", recommended_next_task="T002: references; finish the Jacobi comparison")
    _retain(tmp_path, "T006", "completed", recommended_next_task="T002 (references)")
    plan = planner.next_tasks(tmp_path, limit=50)
    completed = {"T001", "T002", "T003", "T006"}
    follow = [row for row in plan["next"] if row["kind"] == "follow_up"]
    assert [(row["task_id"], row["points_to"]) for row in follow] == [("T003", "T005")]
    assert not [row for row in follow if row["points_to"] in completed]
    research = {row["task_id"]: row for row in plan["research"]}
    # One question from T001 and T002 (prefix, case and punctuation ignored), and the blocked task's own.
    assert set(research) == {"T001", "T004"} and research["T001"]["also_from"] == ["T002"]
    assert research["T001"]["reason"] == "propose an upstream floor for subnormal inputs"
    assert research["T004"]["origin"] == "unresolved_assumptions"
    assert research["T004"]["reason"] == "bind a signed capture to the device."
    refinement = next(row for row in plan["next"] if row["kind"] == "refinement")
    assert refinement["task_id"] == "T005" and refinement["reason"] == "finish the Jacobi comparison"
    stale = {(row["task_id"], row["points_to"]) for row in plan["stale_pointers"]}
    assert stale == {("T002", "T003"), ("T003", "T006"), ("T005", "T002"), ("T006", "T002")}
    assert "T006" not in {row["task_id"] for row in plan["next"]}


def test_reasons_are_cut_at_word_boundaries():
    text = "alpha " * 80
    cut = planner.clip(text)
    assert cut.endswith("…") and len(cut) <= planner.REASON_LIMIT and cut[:-1].split() == ["alpha"] * len(cut[:-1].split())
    assert planner.clip("short   reason") == "short reason"


def test_every_refinement_is_listed_whatever_the_limit(tmp_path, fake_registry):
    for task_id in ("T001", "T002", "T003", "T006"):
        _retain(tmp_path, task_id, "partial")
    plan = planner.next_tasks(tmp_path, limit=1)
    assert [row["kind"] for row in plan["next"]] == ["ready"] + ["refinement"] * 4


def test_the_retained_run_proposes_no_pointer_to_a_completed_task():
    from pathlib import Path
    lab = Path(__file__).resolve().parents[1] / "lab"
    if not (lab / "reports").is_dir():
        pytest.skip("the retained run is not part of the clean-room copy of the tests")
    reports = {r["task_id"]: r for r in runner.load_reports(lab)}
    plan = planner.next_tasks(lab)
    partial = sorted(task_id for task_id, r in reports.items() if r["state"] == "partial")
    assert sorted(row["task_id"] for row in plan["next"] if row["kind"] == "refinement") == partial
    for row in plan["next"] + plan["research"]:
        assert not (row["kind"] == "follow_up" and reports[row["points_to"]]["state"] == "completed")
        assert len(row["reason"]) <= planner.REASON_LIMIT


def test_sentences_after_a_completed_colon_pointer_stay_research_questions():
    """R04: a completed pointer in colon form hides only its own sentence, never a later one (T034)."""
    done = {"T040", "T075", "T130", "T140"}
    kept, stale = planner.next_step_items(
        "T040: compare smooth and mesh Jacobi approximations using these verified smooth derivatives as the "
        "reference. Complex-step derivatives remain open until the core surfaces accept complex coordinates.",
        "T034", done)
    assert kept == [("question", "Complex-step derivatives remain open until the core surfaces accept complex "
                                 "coordinates", None)]
    assert stale == [{"points_to": "T040", "text": "T040: compare smooth and mesh Jacobi approximations using these "
                                                   "verified smooth derivatives as the reference"}]
    # A later sentence opening with a pointer is read as one.
    kept, stale = planner.next_step_items("T040: compare the approximations. T075 gates the estimate. Then T034 again.",
                                          "T034", done)
    assert kept == [("question", "T034 again", None)]
    assert [item["points_to"] for item in stale] == ["T040", "T075"]
    # Abbreviations, decimals and text inside brackets or backticks never end a sentence.
    assert planner.sentences("Use e.g. a 1.5 mm step (see T. Smith. Vol 2) and `x. Y` here. Then stop.") == [
        "Use e.g. a 1.5 mm step (see T. Smith. Vol 2) and `x. Y` here.", "Then stop."]


def test_a_completed_pointer_conditioned_on_a_physical_execution_stays_research():
    """R04: a step to run once real data exists revisits the completed task instead of being dropped (T129, T131)."""
    done = {"T075", "T130", "T140"}
    kept, stale = planner.next_step_items("T140: feed measured repeatability into the uncertainty budget once a real "
                                          "study exists (retained via T139).", "T131", done)
    assert kept == [("question", "T140: feed measured repeatability into the uncertainty budget once a real study "
                                 "exists (retained via T139)", "T140")] and stale == []
    kept, stale = planner.next_step_items("T130: calibrate the scanner scale and frames on artifacts before executing "
                                          "MFG-SCAN-01 and the T128 coupon measurement.", "T129", done)
    assert [(item[0], item[2]) for item in kept] == [("question", "T130")] and stale == []
    # A condition that is not a physical execution leaves the pointer stale.
    kept, stale = planner.next_step_items("T075: keep the fused estimate a candidate until an explicit admission "
                                          "gate.", "T074", done)
    assert kept == [] and [item["points_to"] for item in stale] == ["T075"]


def test_every_follow_up_is_listed_whatever_the_limit(tmp_path, fake_registry):
    """R04: follow-ups are never hidden behind more refinements than the limit."""
    for task_id in ("T001", "T002", "T003"):
        _retain(tmp_path, task_id, "partial")
    _retain(tmp_path, "T006", "completed", recommended_next_task="T005: run the tool-backed comparison")
    plan = planner.next_tasks(tmp_path, limit=1)
    assert [(row["task_id"], row["kind"]) for row in plan["next"]] == [
        ("T005", "ready"), ("T001", "refinement"), ("T002", "refinement"), ("T003", "refinement")]
    assert [(row["task_id"], row["points_to"]) for row in plan["follow_ups"]] == [("T006", "T005")]
