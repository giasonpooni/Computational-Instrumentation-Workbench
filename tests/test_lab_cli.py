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
    # An operator's hardware captures (for example after a T115 capture) are not the packaged queue's.
    captures = {"CIW_LAB_RAPL_LOG", "CIW_LAB_ENERGY_LOG", "CIW_LAB_NVIDIA_SMI_CSV", "CIW_LAB_NVIDIA_SMI_UTC_OFFSET"}
    for name in captures:
        monkeypatch.setenv(name, str(tmp_path / "operator-capture"))
    environment = reproduce.clean_room_environment(tmp_path)
    assert not {"CIW_LAB_EXTENSIONS", "CIW_LAB_MODULES", "PYTHONPATH", *captures} & set(environment)
    assert environment["CIW_LAB_REPOSITORY_ROOT"] == str(tmp_path)
    # The queue reads exactly these capture variables.
    from ciw.lab import energy_gpu_telemetry as telemetry
    assert {telemetry.LOG_ENV, telemetry.SMI_ENV, telemetry.SMI_OFFSET_ENV, telemetry.RAPL_ENV} == captures


def test_clean_room_builds_with_build_isolation_under_a_relative_temporary_root(tmp_path, monkeypatch):
    import os
    import shutil
    import tempfile
    from types import SimpleNamespace
    reproduce = _script("reproduce_lab")

    class Relative:
        """A temporary directory named relative to the current directory, as mkdtemp returns it on Python 3.11."""
        def __init__(self, prefix, dir):
            self.name = os.path.relpath(tempfile.mkdtemp(prefix=prefix, dir=dir))

        def __enter__(self):
            return self.name

        def __exit__(self, *exc):
            shutil.rmtree(self.name)

    class Built(Exception):
        pass

    commands = []

    def run(command, **kwargs):
        commands.append([str(part) for part in command])
        raise Built

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reltmp").mkdir()
    monkeypatch.setattr(reproduce, "tempfile", SimpleNamespace(TemporaryDirectory=Relative))
    monkeypatch.setattr(reproduce, "run", run)
    monkeypatch.setattr(sys, "argv", ["reproduce_lab.py", "--no-compare", "--output-dir", "out",
                                      "--temporary-root", "reltmp"])
    with pytest.raises(Built):
        reproduce.main()
    wheel = commands[0]
    # pip's build isolation provides setuptools>=77; the host interpreter need not have it.
    assert wheel[1:4] == ["-m", "pip", "wheel"] and "--no-build-isolation" not in wheel
    # Every clean-room path is absolute: later steps run with the clean room as working directory.
    paths = [part for part in wheel if "ciw-lab-clean-room-" in part]
    assert paths and all(Path(part).is_absolute() for part in paths)


def test_gate_record_names_the_bindings_the_queue_received(tmp_path, monkeypatch):
    from types import SimpleNamespace
    reproduce = _script("reproduce_lab")
    commands, configurations = [], []

    def run(command, **kwargs):
        command = [str(part) for part in command]
        commands.append(command)
        if command[1:4] == ["-m", "pip", "wheel"]:  # stands in for the wheel build
            dist = Path(command[command.index("--wheel-dir") + 1])
            dist.mkdir(parents=True)
            (dist / "ciw-0-py3-none-any.whl").write_bytes(b"wheel")
        if "pytest" in command:
            configurations.append((Path(command[command.index("--rootdir") + 1]) / "pytest.ini").read_text())

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reproduce, "run", run)
    monkeypatch.setattr(reproduce, "venv", SimpleNamespace(EnvBuilder=lambda **kwargs: SimpleNamespace(
        create=lambda path: None)))
    monkeypatch.setattr(reproduce, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout=str(tmp_path / "site" / "ciw" / "__init__.py"))))
    monkeypatch.setattr(sys, "argv", ["reproduce_lab.py", "--no-compare", "--output-dir", "out",
                                      "--temporary-root", str(tmp_path), "--provider", "csg=stack/csg",
                                      "--provider", "plsr-python=venv312/bin/python", "--provider", "ftr-python=@venv"])
    assert reproduce.main() == 0
    queue = next(command for command in commands if command[1:5] == ["-m", "ciw", "lab", "run"])
    passed = [queue[index + 1] for index, part in enumerate(queue) if part == "--provider"]
    assert passed[:2] == [f"csg={Path.cwd() / 'stack' / 'csg'}", f"plsr-python={Path.cwd() / 'venv312' / 'bin' / 'python'}"]
    record = json.loads((tmp_path / "out" / "gate.json").read_text(encoding="utf-8"))
    # The record shows what the queue and the tests were bound to, and the clean-room Python.
    assert record["providers"] == passed and record["python"] == sys.version.split()[0]
    # The copied tests run under the checkout's markers (lab_task), although pyproject.toml is not copied.
    assert len(configurations) == 1 and "\n    lab_task(*task_ids): " in configurations[0]


def test_clean_room_tests_see_the_bound_providers(tmp_path, monkeypatch):
    reproduce = _script("reproduce_lab")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CIW_LAB_SET_REPO", str(tmp_path / "unbound-set"))  # the calling shell's, not bound
    python = tmp_path / "venv" / "bin" / "python"
    providers = reproduce.bindings(["csg=stack/csg", "ftr=stack/ftr", "scr=stack/scr",
                                    "plsr-python=@venv", "ftr-python=@venv"], python)
    stack = Path.cwd() / "stack"
    assert providers == [("csg", str(stack / "csg")), ("ftr", str(stack / "ftr")), ("scr", str(stack / "scr")),
                         ("plsr-python", str(python)), ("ftr-python", str(python))]
    environment = reproduce.clean_room_environment(tmp_path, providers)
    assert {name: environment.get(name) for name in reproduce.TEST_VARIABLES.values()} == {
        "CIW_LAB_CSG_REPO": str(stack / "csg"), "CIW_LAB_FTR_REPO": str(stack / "ftr"),
        "CIW_LAB_SCR_REPO": str(stack / "scr"), "CIW_LAB_PLSR_PYTHON": str(python),
        "CIW_LAB_FTR_PYTHON": str(python), "CIW_LAB_SET_REPO": None, "CIW_LAB_PPDA_REPO": None,
        "CIW_LAB_SCR_EXCHANGE_REPO": None, "CIW_LAB_SCR_ENGINE": None}
    # A virtual environment's python is a symlink to a base interpreter that lacks the environment's
    # packages (PLSR): the binding is made absolute, never resolved to that interpreter.
    link = Path("plsr-venv") / "bin" / "python"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(tmp_path / "base" / "python3")
    except OSError:  # symlinks need a privilege on Windows
        link = None
    if link is not None:
        [(role, path)] = reproduce.bindings([f"plsr-python={link}"], python)
        assert path == str(Path.cwd() / link) != str(link.resolve())
        assert reproduce.clean_room_environment(tmp_path, [(role, path)])["CIW_LAB_PLSR_PYTHON"] == path
    # The variables are the ones the provider-gated tests read, and every provider variable they read is bound.
    import re
    tests = "".join(path.read_text(encoding="utf-8") for path in Path(__file__).parent.glob("test_lab_*.py"))
    assert all(name in tests for name in reproduce.TEST_VARIABLES.values())
    assert set(re.findall(r"CIW_LAB_[A-Z_]+_(?:REPO|PYTHON|ENGINE)\b", tests)) == set(reproduce.TEST_VARIABLES.values())


def test_gate_compares_only_under_python_312_and_resolves_its_temporary_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    check = _script("check_lab")
    commands = []
    monkeypatch.setattr(check, "call", lambda command, **kwargs: commands.append([str(part) for part in command]))
    monkeypatch.setattr(check, "sys", SimpleNamespace(version_info=(3, 11, 9), version="3.11.9",
                                                      executable=sys.executable))
    monkeypatch.setattr(sys, "argv", ["check_lab.py", "--output-dir", str(tmp_path / "out")])
    # The retained run binds PLSR and FTR interpreters, which need Python 3.12.
    with pytest.raises(SystemExit, match="needs Python 3.12"):
        check.main()
    assert commands == []
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reltmp").mkdir()
    monkeypatch.setattr(check, "pins", lambda: {role: "0" * 40 for role in check.REPOSITORIES})
    monkeypatch.setattr(check, "validate_checkout", lambda path, revision: Path(path).resolve())
    monkeypatch.setattr(sys, "argv", ["check_lab.py", "--no-compare", "--stack-root", "stack",
                                      "--temporary-root", "reltmp", "--output-dir", "out"])
    assert check.main() == 0
    command = commands[-1]
    assert Path(command[command.index("--temporary-root") + 1]) == (tmp_path / "reltmp").resolve()
    assert "plsr-python=@venv" not in command


def test_refresh_retains_only_a_run_that_bound_every_provider(tmp_path, monkeypatch):
    refresh = _script("refresh_lab")
    run = tmp_path / "run"
    run.mkdir()
    before = sorted((Path(refresh.ROOT) / "lab").rglob("*"))

    bound = ["csg=/c", "ftr=/f", "scr=/s", "set=/e", "ppda=/p", "scr-exchange=/x", "plsr-python=/v/bin/python",
             "ftr-python=/v/bin/python"]

    def refresh_from(providers, python="3.12.3"):
        (run / "gate.json").write_text(json.dumps({"schema": "ciw.lab-clean-room-gate.v1", "providers": providers,
                                                   "python": python}))
        monkeypatch.setattr(sys, "argv", ["refresh_lab.py", "--from-run", str(run)])
        with pytest.raises(SystemExit) as refused:
            refresh.main()
        return str(refused.value)

    # A run without the PLSR and FTR interpreters (Python 3.11) differs from what CI reproduces.
    assert "bound no plsr-python, ftr-python" in refresh_from(["csg=/c", "ftr=/f", "scr=/s", "set=/e", "ppda=/p",
                                                               "scr-exchange=/x"])
    assert "bound no set, ppda, scr-exchange" in refresh_from(["csg=/c", "ftr=/f", "scr=/s",
                                                                "plsr-python=/v/bin/python", "ftr-python=/v/bin/python"])
    assert "Python 3.11.9" in refresh_from(bound, "3.11.9")
    assert "Python unrecorded" in refresh_from(bound, None)
    # check_lab.py's bindings pass; this run then lacks its reports.
    assert "incomplete" in refresh_from(bound)
    # Every role bound, but the PLSR interpreter could not run PLSR: its tasks are not CI's.
    (run / "reports").mkdir()
    (run / "artifacts").mkdir()
    (run / "queue-state.json").write_text("{}")
    (run / "REPORTS.md").write_text("")
    (run / "index.html").write_text("")
    (run / "reports" / "T100.json").write_text(json.dumps({"task_id": "T100", "state": "partial"}))
    (run / "reports" / "T101.json").write_text(json.dumps({"task_id": "T101", "state": "partial", "unresolved_assumptions": [
        "Provider plsr-python refused: PLSR_UNAVAILABLE: PackageNotFoundError: No package metadata"]}))
    (run / "reports" / "T019.json").write_text(json.dumps({"task_id": "T019", "findings": [
        {"value": {"refusal": "FTR_EXECUTION_FAILED", "message": "Provider subprocess did not complete"}}]}))
    assert "refused to run in T019, T101" in refresh_from(bound)
    assert sorted((Path(refresh.ROOT) / "lab").rglob("*")) == before


def test_lab_docs_describe_the_gate_as_implemented(monkeypatch):
    import re
    docs = Path(__file__).resolve().parents[1] / "docs"
    if not (docs / "lab" / "AUTHORING.md").is_file():
        pytest.skip("docs/ is not available")

    def text(*parts):
        return " ".join(docs.joinpath(*parts).read_text(encoding="utf-8").split())

    for guide in (text("DEVELOPMENT.md"), text("LAB.md")):
        # Retained evidence comes from a clean-room run with every provider, never a local run into lab/.
        assert not re.search(r"--output-dir\s+lab(?![\w/.-])", guide)
        assert "scripts/refresh_lab.py" in guide and "scripts/check_lab.py" in guide and "Python 3.12+" in guide
    authoring, lab = text("lab", "AUTHORING.md"), text("LAB.md")
    assert "removes both variables" in authoring
    # Provider-gated tests run in CI only for the roles check_lab.py binds; a new role needs provisioning there.
    assert "add a new role there" not in authoring and "`scripts/check_lab.py` (`REPOSITORIES` and its pin)" in authoring
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    check, reproduce = _script("check_lab"), _script("reproduce_lab")
    gate_roles = [*check.REPOSITORIES, "plsr-python", "ftr-python"]  # the latter two on Python 3.12+
    documented = re.search(r"The bindings `check_lab.py` makes set (.*?);", lab).group(1)
    assert set(re.findall(r"CIW_LAB_\w+", documented)) == {reproduce.TEST_VARIABLES[role] for role in gate_roles}
    assert all(f"`{name}`" in lab for name in reproduce.OPERATOR_CAPTURES)
