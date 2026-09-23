import json
import sys

import pytest

from ciw import cli
from ciw.lab import registry

MODULE = '''
from ciw.lab.evidence import finding
from ciw.lab.registry import task

@task("T169", changed_files=("extension_tasks_under_test.py",))
def follow_up(ctx):
    ctx.artifact_json("result.json", {"value": 2.0})
    check = {"reference_kind": "analytic", "reference": "1 + 1", "observed": 0.0, "tolerance": 0.0, "passed": True}
    return {"fields": {"hypothesis": "1 + 1 = 2", "recommended_next_task": "T170"},
            "findings": [finding("Sum", "mathematical", 2.0, {"derivation": "arithmetic", "checks": [check]})]}
'''


@pytest.fixture
def extension(tmp_path, monkeypatch):
    path = tmp_path / "queue-extension.json"
    path.write_text(json.dumps({"schema": "ciw.lab-queue-extension.v1",
                                "section": {"key": "follow-ups", "name": "Follow-up experiments"},
                                "tasks": [{"id": "T169", "title": "Check a follow-up."},
                                          {"id": "T170", "title": "Defer a follow-up."}]}))
    (tmp_path / "extension_tasks_under_test.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CIW_LAB_EXTENSIONS", raising=False)
    monkeypatch.delenv("CIW_LAB_MODULES", raising=False)
    yield path
    registry.configure()
    registry._REGISTRY.pop("T169", None)
    sys.modules.pop("extension_tasks_under_test", None)


def test_extension_tasks_join_the_queue_and_run(extension, tmp_path, capsys):
    out = tmp_path / "run"
    base = ["lab", "--extension", str(extension), "--module", "extension_tasks_under_test"]
    assert cli.main(base + ["run", "T169", "T170", "--output-dir", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["states"] == {"completed": 1, "partial": 0, "deferred": 1, "blocked": 0}
    report = json.loads((out / "reports" / "T169.json").read_text(encoding="utf-8"))
    assert report["section"] == "follow-ups" and report["evidence_status"]["primary"] == "numerically_verified"
    assert cli.main(base + ["queue", "--retained", str(out), "--section", "follow-ups"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line[:4] for line in lines] == ["T169", "T170"]
    registry.configure()
    assert len(registry.load_queue()["tasks"]) == 168


def test_environment_selects_extensions_for_subprocesses(extension, monkeypatch):
    monkeypatch.setenv("CIW_LAB_EXTENSIONS", str(extension))
    monkeypatch.setenv("CIW_LAB_MODULES", "extension_tasks_under_test")
    queue = registry.load_queue()
    assert queue["tasks"][-1]["id"] == "T170" and queue["sections"][-1]["section"] == 12
    implementations, errors = registry.load_implementations()
    assert "T169" in implementations and "extension_tasks_under_test" not in errors


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["tasks"].__setitem__(0, {"id": "T168", "title": "x"}), "must follow"),
    (lambda d: d["section"].__setitem__("key", "geodesic-jacobi"), "already in the queue"),
    (lambda d: d.__setitem__("schema", "other"), "Require"),
    (lambda d: d["tasks"].__setitem__(0, {"id": "T169", "title": "x", "state": "completed"}), "exactly an id"),
    (lambda d: d.__setitem__("tasks", []), "no tasks"),
    (lambda d: d["tasks"].reverse(), "must follow"),
])
def test_malformed_extensions_are_refused(extension, mutate, message):
    data = json.loads(extension.read_text(encoding="utf-8"))
    mutate(data)
    extension.write_text(json.dumps(data))
    registry.configure([extension])
    with pytest.raises(ValueError, match=message):
        registry.load_queue()
