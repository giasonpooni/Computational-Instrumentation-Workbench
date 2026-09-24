import importlib.util
import json
from pathlib import Path
import sys

import pytest

from ciw import cli
from ciw.lab.report import FIELDS


def test_lab_run_report_queue_and_verify(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T156", "--output-dir", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["tasks"] == 1
    assert cli.main(["lab", "report", "T156", "--retained", str(out)]) == 0
    rendered = capsys.readouterr().out
    for _, label in FIELDS:
        assert f"**{label}:**" in rendered
    assert cli.main(["lab", "report", "T156", "--retained", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["task_id"] == "T156"
    assert cli.main(["lab", "queue", "--retained", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 168 and lines[155].startswith("T156") and "not_run" in lines[0]
    assert cli.main(["lab", "verify", "--retained", str(out), "--fresh", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_lab_verify_exits_3_on_drift(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T156", "--output-dir", str(out)]) == 0
    capsys.readouterr()
    fresh = tmp_path / "fresh"
    (fresh / "reports").mkdir(parents=True)
    report = json.loads((out / "reports" / "T156.json").read_text(encoding="utf-8"))
    report["state"] = "partial" if report["state"] != "partial" else "completed"
    from ciw.lab.report import report_identity
    report["report_id"] = report_identity(report)
    (fresh / "reports" / "T156.json").write_text(json.dumps(report))
    assert cli.main(["lab", "verify", "--retained", str(out), "--fresh", str(fresh)]) == 3
    assert f"state {json.loads((out / 'reports' / 'T156.json').read_text(encoding='utf-8'))['state']} -> " in capsys.readouterr().out


def test_lab_cli_refuses_ambiguous_selection_and_bad_bindings(tmp_path, capsys):
    assert cli.main(["lab", "run", "--output-dir", str(tmp_path)]) == 2
    assert "Name task identities or pass --all" in capsys.readouterr().err
    assert cli.main(["lab", "run", "T001", "--all", "--output-dir", str(tmp_path)]) == 2
    capsys.readouterr()
    assert cli.main(["lab", "run", "T001", "--provider", "csg", "--output-dir", str(tmp_path)]) == 2
    assert "ROLE=PATH" in capsys.readouterr().err
    assert cli.main(["lab", "run", "T999", "--output-dir", str(tmp_path)]) == 2
    assert "Unknown lab task" in capsys.readouterr().err


def test_tampered_retained_report_is_refused(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T156", "--output-dir", str(out)]) == 0
    capsys.readouterr()
    path = out / "reports" / "T156.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["findings"][0]["evidence_status"] = "hardware_measured"
    path.write_text(json.dumps(report))
    assert cli.main(["lab", "report", "T156", "--retained", str(out)]) == 2
    assert "refused" in capsys.readouterr().err


def test_lab_verify_fails_without_retained_reports(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["lab", "run", "T156", "--output-dir", str(out)]) == 0
    capsys.readouterr()
    (tmp_path / "empty").mkdir()
    for retained in (tmp_path / "empty", tmp_path / "no-such-directory"):
        assert cli.main(["lab", "verify", "--retained", str(retained), "--fresh", str(out)]) == 3
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is False and result["compared"] == 0
        assert "T156: not retained" in result["problems"]


def _script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    if not path.is_file():
        pytest.skip("scripts/ is not part of the clean-room copy of the tests")
    spec = importlib.util.spec_from_file_location(f"lab_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_room_refuses_an_empty_comparison_and_ignores_extensions(tmp_path, monkeypatch):
    reproduce = _script("reproduce_lab")
    (tmp_path / "retained" / "reports").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["reproduce_lab.py", "--output-dir", str(tmp_path / "out"),
                                      "--retained", str(tmp_path / "retained")])
    with pytest.raises(SystemExit, match="No retained reports .*--no-compare"):
        reproduce.main()
    assert not (tmp_path / "out").exists()  # refused before building anything
    monkeypatch.setenv("CIW_LAB_EXTENSIONS", str(tmp_path / "follow-ups.json"))
    monkeypatch.setenv("CIW_LAB_MODULES", "my_lab_tasks")
    environment = reproduce.clean_room_environment(tmp_path)
    assert not {"CIW_LAB_EXTENSIONS", "CIW_LAB_MODULES", "PYTHONPATH"} & set(environment)
    assert environment["CIW_LAB_REPOSITORY_ROOT"] == str(tmp_path)
