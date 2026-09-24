"""Covariance argument parsing and refusal exit codes at the terminal boundary."""
import json
from pathlib import Path

import pytest

from ciw import cli
from ciw.adapters.protocol import AdapterRefusal


def _stub_covariance(monkeypatch):
    calls = []
    import ciw.covariance_workflow as workflow
    monkeypatch.setattr(workflow, "execute_covariance", lambda path, **kwargs: calls.append(("execute", path, kwargs)) or {})
    monkeypatch.setattr(workflow, "replay_covariance", lambda path, **kwargs: calls.append(("replay", path, kwargs)) or {})
    monkeypatch.setattr(cli, "print_json", lambda value: None)
    return calls


def test_covariance_requires_parameters_and_replay_refuses_them(tmp_path, monkeypatch):
    _stub_covariance(monkeypatch)
    base = [str(tmp_path / "ws.json"), "--jspt-repo", str(tmp_path), "--output-dir", str(tmp_path), "--json"]
    with pytest.raises(SystemExit) as missing:
        cli.main(["covariance", *base])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as extra:
        cli.main(["covariance-replay", *base, "--parameters", str(tmp_path / "p.json")])
    assert extra.value.code == 2


@pytest.mark.parametrize("alias", ["--adapter-python", "--python"])
def test_covariance_interpreter_aliases_bind_the_same_option(tmp_path, monkeypatch, alias):
    calls = _stub_covariance(monkeypatch)
    parameters = tmp_path / "p.json"
    parameters.write_text(json.dumps({"map": "declared"}))
    code = cli.main(["covariance", str(tmp_path / "ws.json"), "--parameters", str(parameters), "--jspt-repo", str(tmp_path),
                     "--output-dir", str(tmp_path), alias, "/trusted/python", "--json"])
    assert code == 0
    (action, _, kwargs), = calls
    assert action == "execute" and kwargs["parameters"] == {"map": "declared"}
    assert kwargs["python_executable"] == Path("/trusted/python") and kwargs["jspt_repo"] == tmp_path


def test_non_object_covariance_parameters_exit_two_before_execution(tmp_path, monkeypatch, capsys):
    calls = _stub_covariance(monkeypatch)
    parameters = tmp_path / "p.json"
    parameters.write_text("[1, 2]")
    code = cli.main(["covariance", str(tmp_path / "ws.json"), "--parameters", str(parameters),
                     "--jspt-repo", str(tmp_path), "--output-dir", str(tmp_path)])
    assert code == 2 and calls == [] and "must be a JSON object" in capsys.readouterr().err


def test_calibration_status_refusal_exits_two_with_its_reason(tmp_path, monkeypatch, capsys):
    import ciw.investigation as investigation
    from ciw.calibration_status import _timestamp

    def inspect(path, evaluated_at=None):
        _timestamp(evaluated_at, "evaluated_at", strict=True)
        return {}

    monkeypatch.setattr(investigation, "inspect_investigation", inspect)
    code = cli.main(["investigation", "inspect", str(tmp_path / "ws.json"), "--evaluated-at", "2026-01-01T00:00:00", "--json"])
    assert code == 2 and "timezone" in capsys.readouterr().err


def test_a_provider_refusal_exits_two_with_its_code(tmp_path, monkeypatch, capsys):
    import ciw.covariance_workflow as workflow

    def refuse(path, **kwargs):
        raise AdapterRefusal("replay_unavailable", "No JSPT covariance execution retained")

    monkeypatch.setattr(workflow, "replay_covariance", refuse)
    code = cli.main(["covariance-replay", str(tmp_path / "ws.json"), "--jspt-repo", str(tmp_path), "--output-dir", str(tmp_path)])
    assert code == 2 and "ciw: replay_unavailable:" in capsys.readouterr().err
