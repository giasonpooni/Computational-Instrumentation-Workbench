import json

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
    report = json.loads((out / "reports" / "T156.json").read_text())
    report["state"] = "blocked"
    from ciw.lab.report import report_identity
    report["report_id"] = report_identity(report)
    (fresh / "reports" / "T156.json").write_text(json.dumps(report))
    assert cli.main(["lab", "verify", "--retained", str(out), "--fresh", str(fresh)]) == 3
    assert "-> blocked" in capsys.readouterr().out


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
    report = json.loads(path.read_text())
    report["findings"][0]["evidence_status"] = "hardware_measured"
    path.write_text(json.dumps(report))
    assert cli.main(["lab", "report", "T156", "--retained", str(out)]) == 2
    assert "refused" in capsys.readouterr().err
