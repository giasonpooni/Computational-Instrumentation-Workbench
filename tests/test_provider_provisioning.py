"""Offline gates reuse exact operator-owned checkouts without rewriting them."""
import importlib
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def git(path, *arguments):
    return subprocess.run(["git", "-C", str(path), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def repository(path):
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "core.autocrlf", "false")
    git(path, "config", "user.name", "Provider test")
    git(path, "config", "user.email", "provider@example.invalid")
    (path / "source.txt").write_bytes(b"pinned provider\n")
    git(path, "add", "source.txt")
    git(path, "commit", "-qm", "provider fixture")
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("provider_checkouts")


def configured_gate(name, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    gate = importlib.import_module("check_" + name)
    root = tmp_path / "project"
    manifests = root / "src" / "ciw"
    manifests.mkdir(parents=True)
    stack = tmp_path / "stack"
    path = stack / ("Example-Provider" if name == "telemetry" else "engine")
    previous = repository(path)
    historical = None
    pin = {"revision": previous, "repository": "https://example.invalid/provider", "module": "provider"}
    if name == "adapters":
        historical = stack / "engine-legacy"
        subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--quiet", "--no-hardlinks", str(path), str(historical)], check=True)
        (path / "source.txt").write_bytes(b"current pinned provider\n")
        git(path, "add", "source.txt")
        git(path, "commit", "-qm", "current fixture")
        pin["revision"] = git(path, "rev-parse", "HEAD")
        pin["historical"] = [{"revision": previous, "module": "provider"}]
        manifest = "adapter-runtimes.json"
        monkeypatch.setattr(gate, "ROOT", root)
    else:
        monkeypatch.setattr(gate, "__file__", str(root / "scripts" / ("check_" + name + ".py")))
        monkeypatch.setattr(gate, "REPOSITORIES", {"engine": "Example-Provider"})
        manifest = "telemetry-runtimes.json" if name == "telemetry" else "calibrated-observable-runtimes.json"
    (manifests / manifest).write_text(json.dumps({"engine": pin}), encoding="utf-8")
    return gate, stack, path, historical, previous


def guard_processes(monkeypatch):
    """Run only local read-only Git probes; capture the final pytest process."""
    original = subprocess.run
    invocations = []

    def run(arguments, **kwargs):
        if arguments[0] == "git":
            assert not {"clone", "checkout", "config", "fetch", "pull"}.intersection(arguments)
            return original(arguments, **kwargs)
        assert arguments[1:3] == ["-m", "pytest"]
        invocations.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0)

    def call(arguments, **kwargs):
        return run(arguments, **kwargs).returncode

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "call", call)
    return invocations


@pytest.mark.parametrize("name", ["adapters", "telemetry", "calibrated_observable"])
def test_local_stack_uses_exact_pins_without_rewriting_checkouts(name, tmp_path, monkeypatch):
    gate, stack, path, historical, _ = configured_gate(name, tmp_path, monkeypatch)
    checkouts = [path] + ([historical] if historical else [])
    before = {(checkout, filename): (checkout / ".git" / filename).read_bytes()
              for checkout in checkouts for filename in ("HEAD", "config", "index")}
    invocations = guard_processes(monkeypatch)
    gate.main(["--stack-root", str(stack)])
    assert len(invocations) == 1
    environment = invocations[0][1]["env"]
    if name == "adapters":
        assert environment["CIW_ENGINE_REPO"] == str(path.resolve())
        assert environment["CIW_ENGINE_LEGACY_REPO"] == str(historical.resolve())
    else:
        key = "CIW_TELEMETRY_STACK_ROOT" if name == "telemetry" else "CIW_CALIBRATED_STACK_ROOT"
        assert environment[key] == str(stack.resolve())
    assert all((checkout / ".git" / filename).read_bytes() == value
               for (checkout, filename), value in before.items())


@pytest.mark.parametrize("name", ["adapters", "telemetry", "calibrated_observable"])
@pytest.mark.parametrize("failure", ["wrong_pin", "dirty", "missing"])
def test_local_stack_refuses_before_starting_integration_tests(name, failure, tmp_path, monkeypatch):
    gate, stack, path, _, _ = configured_gate(name, tmp_path, monkeypatch)
    if failure == "wrong_pin":
        (path / "source.txt").write_bytes(b"unapproved revision\n")
        git(path, "add", "source.txt")
        git(path, "commit", "-qm", "unapproved change")
    elif failure == "dirty":
        (path / "source.txt").write_bytes(b"uncommitted change\n")
    else:
        stack = tmp_path / "unavailable"
    invocations = guard_processes(monkeypatch)
    with pytest.raises(ValueError, match="wrong pin|dirty|unavailable"):
        gate.main(["--stack-root", str(stack)])
    assert not invocations


def test_separate_historical_stack_is_explicitly_bound(tmp_path, monkeypatch):
    gate, stack, path, historical, previous = configured_gate("adapters", tmp_path, monkeypatch)
    historic_root = tmp_path / "historical"
    historic = historic_root / "engine"
    historic_root.mkdir()
    subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--quiet", "--no-hardlinks", str(historical), str(historic)], check=True)
    invocations = guard_processes(monkeypatch)
    gate.main(["--stack-root", str(stack), "--historical-stack-root", str(historic_root)])
    assert invocations[0][1]["env"]["CIW_ENGINE_LEGACY_REPO"] == str(historic.resolve())
    assert git(historic, "rev-parse", "HEAD") == previous


def test_historical_checkout_must_match_its_own_pin(tmp_path, monkeypatch):
    gate, stack, _, historical, _ = configured_gate("adapters", tmp_path, monkeypatch)
    (historical / "source.txt").write_bytes(b"modified historical evidence\n")
    invocations = guard_processes(monkeypatch)
    with pytest.raises(ValueError, match="dirty"):
        gate.main(["--stack-root", str(stack)])
    assert not invocations


def test_assume_unchanged_does_not_hide_raw_byte_drift(tmp_path, helpers):
    path = tmp_path / "provider"
    revision = repository(path)
    (path / ".git" / "info" / "attributes").write_text("*.txt text\n", encoding="utf-8")
    (path / "source.txt").write_bytes(b"pinned provider\r\n")
    git(path, "update-index", "--assume-unchanged", "source.txt")
    assert git(path, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="tracked bytes"):
        helpers.validate_checkout(path, revision)


def test_ignored_noncache_source_is_not_an_approved_pin(tmp_path, helpers):
    path = tmp_path / "provider"
    revision = repository(path)
    (path / ".git" / "info" / "exclude").write_text("shadow.py\n", encoding="utf-8")
    (path / "shadow.py").write_text("raise RuntimeError('unapproved')\n", encoding="utf-8")
    assert git(path, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="untracked"):
        helpers.validate_checkout(path, revision)


def test_uninitialized_pinned_gitlink_needs_no_symlink_or_download(tmp_path, helpers):
    path = tmp_path / "provider"
    linked_revision = repository(path)
    git(path, "update-index", "--add", "--cacheinfo", f"160000,{linked_revision},vendor/scout")
    git(path, "commit", "-qm", "retain submodule boundary")
    (path / "vendor" / "scout").mkdir(parents=True)
    revision = git(path, "rev-parse", "HEAD")
    assert helpers.validate_checkout(path, revision) == path.resolve()
