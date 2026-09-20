"""Bounded batch evaluation: accounting, retained evidence, interruption and resume."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


L = pytest.importorskip("lyapunov", reason="install the optional plsr extra")

from ciw import plsr
from ciw import plsr_batch
from ciw import plsr_engine as engine


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "plsr"
SWEEP = EXAMPLES / "batch-parameter-sweep.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cli(*args, exit_code=0):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "ciw", *map(str, args)], cwd=ROOT,
        env=environment, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == exit_code, completed.stderr
    if exit_code == 2:
        assert not completed.stdout and completed.stderr
        return completed
    assert not completed.stderr
    return json.loads(completed.stdout)


def sample(x, theta=None, theta_dot=None):
    return {"sample_schema": "plsr-sample-v1", "x": x, "theta": theta, "theta_dot": theta_dot}


@pytest.fixture
def workspace(tmp_path):
    """A writable copy of every example model, ready to hold a plan."""
    for path in EXAMPLES.glob("*.json"):
        shutil.copy(path, tmp_path / path.name)
    return tmp_path


def write_plan(directory, samples, *, batch_id="test-batch", model="continuous-affine.json"):
    plan = {"batch_schema": "ciw-plsr-batch-v1", "batch_id": batch_id,
            "description": "A declared collection written by the test suite.",
            "model_file": model, "samples": samples}
    path = Path(directory) / "plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def interior(count):
    return [{"sample_id": f"s-{number:03d}",
             "sample": sample([0.1, 0.2], [number / count - 0.5], [0.1])}
            for number in range(count)]


def entries_of(index):
    return {entry["sample_id"]: entry for entry in index["entries"]}


def test_every_requested_sample_is_accounted_for(tmp_path):
    index = plsr_batch.run_batch(SWEEP, tmp_path / "out")
    plan = load(SWEEP)
    assert index["requested"] == len(plan["samples"]) == len(index["entries"])
    assert [entry["sample_id"] for entry in index["entries"]] == [
        item["sample_id"] for item in plan["samples"]]
    assert sum(index["outcomes"].values()) == index["requested"]
    assert index["status"] == "complete" and index["outcomes"]["unfinished"] == 0
    assert index["batch_digest"] == plsr_batch.batch_digest(plan)
    assert set(index["outcomes"]) == set(plsr_batch.OUTCOMES)


def test_the_example_sweep_reaches_every_outcome_the_accounting_distinguishes(tmp_path):
    index = plsr_batch.run_batch(SWEEP, tmp_path / "out")
    assert index["outcomes"]["completed"] and index["outcomes"]["refused"]
    assert index["outcomes"]["violated"] and not index["outcomes"]["errored"]
    assert set(index["codes"]) >= {
        "CERTIFIED_WITH_MARGIN", "MARGIN_LOW", "NOT_CERTIFIED",
        "DECREASE_NOT_DEFINITE", "OUTSIDE_PARAMETER_BOX", "OUTSIDE_LEVEL_SET"}


def test_completed_is_accounting_and_is_not_the_acceptable_count(tmp_path):
    index = plsr_batch.run_batch(SWEEP, tmp_path / "out")
    low = [entry for entry in index["entries"] if entry["code"] == "MARGIN_LOW"]
    assert low and all(entry["outcome"] == "completed" for entry in low)
    assert all(entry["operationally_acceptable"] is False for entry in low)
    assert all(entry["inequality_certified"] is True for entry in low)
    assert index["operationally_acceptable"] == index["outcomes"]["completed"] - len(low)
    assert index["codes"]["MARGIN_LOW"] == len(low)


def test_the_outcome_map_covers_every_presentation_category_the_adapter_emits():
    assert set(plsr_batch.OUTCOME_OF_CATEGORY) == set(engine._CATEGORIES.values())
    assert set(plsr_batch.OUTCOME_OF_CATEGORY.values()) <= set(plsr_batch.OUTCOMES)


def test_each_result_keeps_its_own_inspectable_bundle(tmp_path):
    directory = tmp_path / "out"
    index = plsr_batch.run_batch(SWEEP, directory)
    saved = sorted(path.name for path in (directory / "runs").iterdir())
    evaluated = [entry for entry in index["entries"] if entry["saved_file"]]
    assert len(saved) == len(evaluated) == index["requested"]
    for entry in evaluated:
        bundle = plsr.inspect_run(directory / "runs" / entry["saved_file"])["bundle"]
        assert bundle["record"]["record_digest"] == entry["record_digest"]
        assert bundle["result_id"] == entry["result_id"]
        assert bundle["evidence_id"] == entry["evidence_id"]
        assert bundle["record"]["code"] == entry["code"]


def test_identical_samples_share_an_evidence_identity_and_keep_distinct_results(workspace):
    plan = write_plan(workspace, [
        {"sample_id": "first", "sample": sample([0.1, 0.2], [0.5], [0.1])},
        {"sample_id": "second", "sample": sample([0.1, 0.2], [0.5], [0.1])}])
    index = plsr_batch.run_batch(plan, workspace / "out")
    first, second = index["entries"]
    assert first["evidence_id"] == second["evidence_id"]
    assert first["result_id"] != second["result_id"]
    assert first["saved_file"] != second["saved_file"]
    assert first["record_digest"] == second["record_digest"]


def test_a_bounded_run_leaves_the_rest_unfinished_and_resume_completes_them(workspace):
    plan = write_plan(workspace, interior(8))
    directory = workspace / "out"
    first = plsr_batch.run_batch(plan, directory, limit=3)
    assert first["status"] == "incomplete" and first["evaluated_this_run"] == 3
    assert first["outcomes"]["unfinished"] == 5 and first["resumed"] is False
    unfinished = [entry for entry in first["entries"] if entry["outcome"] == "unfinished"]
    assert all(entry["saved_file"] is None and entry["code"] is None for entry in unfinished)

    retained = {path.name: path.read_bytes() for path in (directory / "runs").iterdir()}
    journal = (directory / plsr_batch.JOURNAL_NAME).read_bytes()

    second = plsr_batch.run_batch(plan, directory, resume=True)
    assert second["status"] == "complete" and second["evaluated_this_run"] == 5
    assert second["resumed"] is True and second["outcomes"]["unfinished"] == 0
    assert (directory / plsr_batch.JOURNAL_NAME).read_bytes().startswith(journal)
    for name, content in retained.items():
        assert (directory / "runs" / name).read_bytes() == content
    before = entries_of(first)
    for sample_id, entry in before.items():
        if entry["outcome"] != "unfinished":
            assert entries_of(second)[sample_id] == entry


def test_resuming_after_every_sample_finished_changes_nothing(workspace):
    plan = write_plan(workspace, interior(4))
    directory = workspace / "out"
    first = plsr_batch.run_batch(plan, directory)
    retained = {path.name: path.read_bytes() for path in (directory / "runs").iterdir()}
    second = plsr_batch.run_batch(plan, directory, resume=True)
    assert second["evaluated_this_run"] == 0 and second["status"] == "complete"
    assert entries_of(second) == entries_of(first)
    assert {path.name: path.read_bytes() for path in (directory / "runs").iterdir()} == retained


def test_a_second_run_will_not_append_to_a_saved_batch_without_resume(workspace):
    plan = write_plan(workspace, interior(2))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory)
    with pytest.raises(plsr_batch.BatchError, match="already holds a batch"):
        plsr_batch.run_batch(plan, directory)


def test_resuming_a_different_collection_is_refused(workspace):
    plan = write_plan(workspace, interior(3))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory, limit=1)
    write_plan(workspace, interior(4))
    with pytest.raises(plsr_batch.BatchError, match="different declared collection"):
        plsr_batch.run_batch(plan, directory, resume=True)


def test_resuming_against_a_different_runtime_identity_is_refused(workspace, monkeypatch):
    plan = write_plan(workspace, interior(3))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory, limit=1)
    identity = dict(engine.runtime_identity())
    identity["commit"] = "0" * 40
    monkeypatch.setattr(engine, "runtime_identity", lambda: identity)
    with pytest.raises(plsr_batch.BatchError, match="different runtime identity"):
        plsr_batch.run_batch(plan, directory, resume=True)


def test_a_torn_final_journal_line_is_discarded_exactly_and_the_rest_survives(workspace):
    plan = write_plan(workspace, interior(6))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory, limit=3)
    journal = directory / plsr_batch.JOURNAL_NAME
    intact = journal.read_bytes()
    torn = b'{"entry_schema": "ciw-plsr-batch-entry-v1", "sample_id": "s-0'
    journal.write_bytes(intact + torn)

    index = plsr_batch.run_batch(plan, directory, resume=True)
    assert index["discarded_partial_bytes"] == len(torn)
    assert index["status"] == "complete" and index["evaluated_this_run"] == 3
    assert journal.read_bytes().startswith(intact)


def test_a_complete_but_invalid_journal_line_is_corruption_not_a_torn_write(workspace):
    plan = write_plan(workspace, interior(3))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory, limit=1)
    journal = directory / plsr_batch.JOURNAL_NAME
    journal.write_bytes(journal.read_bytes() + b'{"entry_schema": "something-else"}\n')
    with pytest.raises(plsr_batch.BatchError, match="journal line 3"):
        plsr_batch.run_batch(plan, directory, resume=True)


def test_a_journal_without_a_header_is_refused(workspace):
    plan = write_plan(workspace, interior(2))
    directory = workspace / "out"
    directory.mkdir()
    (directory / plsr_batch.JOURNAL_NAME).write_bytes(b'{"entry_schema": "x"}\n')
    with pytest.raises(plsr_batch.BatchError, match="must begin with its batch header"):
        plsr_batch.run_batch(plan, directory, resume=True)


def test_a_refused_sample_is_accounted_as_errored_with_its_reason(workspace):
    plan = write_plan(workspace, [
        {"sample_id": "good", "sample": sample([0.1, 0.2], [0.5], [0.1])},
        {"sample_id": "wrong-dimension", "sample": sample([0.1], [0.5], [0.1])},
        {"sample_id": "missing-model", "model_file": "absent.json",
         "sample": sample([0.1, 0.2], [0.5], [0.1])}])
    index = plsr_batch.run_batch(plan, workspace / "out")
    entries = entries_of(index)
    assert index["outcomes"] == {"completed": 1, "refused": 0, "violated": 0,
                                 "errored": 2, "unfinished": 0}
    assert index["status"] == "complete"
    assert "x must be an explicit JSON array of 2 numbers" in entries["wrong-dimension"]["reason"]
    assert entries["wrong-dimension"]["saved_file"] is None
    assert entries["wrong-dimension"]["timings_s"] is None
    assert "absent.json" in entries["missing-model"]["reason"]
    assert len(list((workspace / "out" / "runs").iterdir())) == 1


def test_an_interrupted_run_still_writes_the_accounting_it_reached(workspace, monkeypatch):
    plan = write_plan(workspace, interior(6))
    directory = workspace / "out"
    real = plsr.evaluate_sample
    calls = []

    def stop_after_two(*args, **kwargs):
        if len(calls) >= 2:
            raise KeyboardInterrupt
        calls.append(None)
        return real(*args, **kwargs)

    monkeypatch.setattr(plsr, "evaluate_sample", stop_after_two)
    index = plsr_batch.run_batch(plan, directory)
    assert index["interrupted"] is True and index["status"] == "incomplete"
    assert index["outcomes"]["completed"] == 2 and index["outcomes"]["unfinished"] == 4
    monkeypatch.undo()
    resumed = plsr_batch.run_batch(plan, directory, resume=True)
    assert resumed["status"] == "complete" and resumed["evaluated_this_run"] == 4
    assert resumed["interrupted"] is False


def test_a_killed_process_loses_no_finished_evidence_and_resumes(workspace):
    plan = write_plan(workspace, interior(120))
    directory = workspace / "out"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-m", "ciw", "plsr", "batch", "run", str(plan),
         "--output-dir", str(directory)], cwd=ROOT, env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    journal = directory / plsr_batch.JOURNAL_NAME
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            if journal.exists() and journal.read_bytes().count(b"\n") >= 4:
                break
            if process.poll() is not None:
                break
            time.sleep(0.02)
    finally:
        process.terminate()
        process.wait(timeout=60)
    assert journal.exists(), "the journal must reach disk before the first sample finishes"
    retained = {path.name: path.read_bytes() for path in (directory / "runs").iterdir()}
    survived = journal.read_bytes()

    index = plsr_batch.run_batch(plan, directory, resume=True)
    assert index["status"] == "complete"
    # The kill landed mid-run, so resuming really had outstanding work to do.
    assert index["evaluated_this_run"] > 0
    assert sum(index["outcomes"].values()) == 120
    assert index["outcomes"]["unfinished"] == 0
    assert journal.read_bytes().startswith(survived.rpartition(b"\n")[0])
    for name, content in retained.items():
        assert (directory / "runs" / name).read_bytes() == content


def test_status_reads_the_saved_index_without_evaluating_anything(workspace, monkeypatch):
    plan = write_plan(workspace, interior(3))
    directory = workspace / "out"
    index = plsr_batch.run_batch(plan, directory)
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: pytest.fail("no evaluation"))
    assert plsr_batch.read_index(directory) == index


def test_a_tampered_index_is_refused(workspace):
    plan = write_plan(workspace, interior(2))
    directory = workspace / "out"
    plsr_batch.run_batch(plan, directory)
    path = directory / plsr_batch.INDEX_NAME
    index = load(path)
    index["outcomes"]["completed"] = 99
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    with pytest.raises(plsr_batch.BatchError, match="index_digest does not match"):
        plsr_batch.read_index(directory)


def test_the_three_measured_segments_are_separate_and_add_up(workspace):
    plan = write_plan(workspace, interior(5))
    index = plsr_batch.run_batch(plan, workspace / "out")
    for entry in index["entries"]:
        timings = entry["timings_s"]
        assert set(timings) == {"engine", "adapter", "evidence", "total"}
        assert all(value >= 0.0 for value in timings.values())
        parts = timings["engine"] + timings["adapter"] + timings["evidence"]
        assert abs(parts - timings["total"]) <= 1e-9
    for key in ("engine", "adapter", "evidence", "total"):
        statistics = index["timings_s"][key]
        assert statistics["count"] == 5
        assert statistics["min"] <= statistics["mean"] <= statistics["max"]
    assert index["timings_s"]["wall"] >= index["timings_s"]["total"]["total"]


@pytest.mark.parametrize("change,message", [
    (lambda p: p["samples"].append(dict(p["samples"][0])), "duplicate sample_id"),
    (lambda p: p.update(samples=[]), "at least one sample"),
    (lambda p: p.update(model_file="/etc/passwd"), "relative path beside the plan"),
    (lambda p: p["samples"][0].update(model_file="../escape.json"), "relative path beside the plan"),
    (lambda p: p.update(batch_schema="ciw-plsr-batch-v2"), "batch_schema must be"),
    (lambda p: p["samples"][0].update(extra=1), "each sample needs sample_id and sample"),
    (lambda p: p.update(batch_id="  "), "batch_id must be a nonempty string"),
    (lambda p: p.pop("description"), "plan fields"),
])
def test_a_malformed_plan_is_refused_with_a_named_reason(workspace, change, message):
    path = write_plan(workspace, interior(2))
    plan = load(path)
    change(plan)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with pytest.raises(plsr_batch.BatchError, match=message):
        plsr_batch.run_batch(path, workspace / "out")


def test_a_non_positive_limit_is_refused(workspace):
    path = write_plan(workspace, interior(2))
    with pytest.raises(plsr_batch.BatchError, match="limit must be a positive integer"):
        plsr_batch.run_batch(path, workspace / "out", limit=0)


def test_the_terminal_reports_the_index_and_distinguishes_unfinished_from_errored(tmp_path):
    directory = tmp_path / "out"
    index = cli("plsr", "batch", "run", SWEEP, "--output-dir", directory)
    assert index["index_schema"] == "ciw-plsr-batch-index-v1"
    assert index["status"] == "complete"
    assert cli("plsr", "batch", "status", directory) == index

    bounded = tmp_path / "bounded"
    partial = cli("plsr", "batch", "run", SWEEP, "--output-dir", bounded,
                  "--limit", 2, exit_code=7)
    assert partial["outcomes"]["unfinished"] == partial["requested"] - 2
    cli("plsr", "batch", "status", bounded, exit_code=7)
    finished = cli("plsr", "batch", "run", SWEEP, "--output-dir", bounded, "--resume")
    assert finished["status"] == "complete"


def test_the_terminal_reports_an_errored_sample_with_a_distinct_exit_code(workspace):
    plan = write_plan(workspace, [{"sample_id": "bad", "sample": sample([0.1], [0.5], [0.1])}])
    index = cli("plsr", "batch", "run", plan, "--output-dir", workspace / "out", exit_code=6)
    assert index["outcomes"]["errored"] == 1 and index["status"] == "complete"


def test_the_published_sweep_plan_names_only_models_beside_it():
    plan = plsr_batch.load_plan(SWEEP)
    named = {plan["model_file"]} | {item["model_file"] for item in plan["samples"]
                                    if "model_file" in item}
    for name in named:
        assert (EXAMPLES / name).is_file()
        assert engine.load_model(EXAMPLES / name)
