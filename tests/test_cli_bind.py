"""serve --bind-role KIND:ROLE=PATH binds any declared kind through the workbench's own role and pin checks."""
from pathlib import Path

import pytest

from ciw import cli


def test_bindings_group_by_kind_and_refuse_malformed_or_repeated_roles():
    grouped = cli.parse_bindings(["mesh-path:isgt=/a", "variational-free-energy:csg=/c", "variational-free-energy:gsie=/g"])
    assert grouped == {"mesh-path": {"isgt": Path("/a")},
                       "variational-free-energy": {"csg": Path("/c"), "gsie": Path("/g")}}
    for bad in ("mesh-path", "mesh-path:isgt", "mesh-path:=/a", ":isgt=/a", "mesh-path:isgt= "):
        with pytest.raises(ValueError, match="KIND:ROLE=PATH"):
            cli.parse_bindings([bad])
    with pytest.raises(ValueError, match="more than once"):
        cli.parse_bindings(["mesh-path:isgt=/a", "mesh-path:isgt=/b"])


def _serve(monkeypatch, tmp_path, *arguments):
    calls = []
    monkeypatch.setattr(cli, "run_server", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(cli.asyncio, "run", lambda coroutine: None)
    return cli.main(["serve", "--output-dir", str(tmp_path), *arguments]), calls


def test_serve_bind_reaches_the_workbench_role_checks(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr("ciw.workbench.Workbench.bind_workflow", lambda self, kind, repositories: seen.append((kind, repositories)))
    _serve(monkeypatch, tmp_path, "--bind-role", "mesh-path:isgt=" + str(tmp_path / "isgt"))
    assert seen == [("mesh-path", {"isgt": tmp_path / "isgt"})]


def test_serve_bind_refuses_an_undeclared_role_or_kind(monkeypatch, tmp_path, capsys):
    for value, message in (("mesh-path:unknown=/x", "Bind exactly the repositories declared by the workflow"),
                           ("not-a-kind:role=/x", "no executable operation")):
        code, calls = _serve(monkeypatch, tmp_path, "--bind-role", value)
        assert code != 0 and calls == [] and message in capsys.readouterr().err


def test_serve_bind_role_refuses_a_kind_another_option_already_bound(monkeypatch, tmp_path, capsys):
    def bind(self, kind, repositories):
        self._bindings[kind] = repositories

    monkeypatch.setattr("ciw.workbench.Workbench.bind_workflow", bind)
    code, calls = _serve(monkeypatch, tmp_path, "--intrinsic-surface-repo", str(tmp_path / "isgt"),
                         "--bind-role", "mesh-path:isgt=" + str(tmp_path / "other"))
    assert code != 0 and calls == [] and "repeats a binding another option already made" in capsys.readouterr().err
