"""Saved adapter runtimes outside the current-or-historical pin allowlist refuse before binding."""

from importlib.resources import files
import json
import sys

import pytest

from ciw import investigation
from ciw.adapters.protocol import AdapterRefusal


def _pins():
    return json.loads(files("ciw").joinpath("adapter-runtimes.json").read_text())


def _saved(entry, **changes):
    return {"revision": entry["revision"], "module": entry["module"], "python_sha256": "0" * 64,
            "python_version": "3.12.0", "dependencies": {"numpy": None, "scipy": None}, **changes}


class _Recorder:
    calls = []

    def __init__(self, repo, revision, module, **kwargs):
        _Recorder.calls.append((repo, revision, module, kwargs))


@pytest.mark.parametrize("name", ["rci", "fsrt"])
@pytest.mark.parametrize("drift", ["revision", "module"])
def test_saved_runtime_off_the_allowlist_is_refused_before_any_binding(tmp_path, monkeypatch, name, drift):
    spec = _pins()[name]
    monkeypatch.setattr(investigation, "PinnedSubprocessAdapter",
                        lambda *args, **kwargs: pytest.fail("an off-allowlist runtime must not bind an adapter"))
    changes = {"revision": "0" * 40} if drift == "revision" else {"module": spec["module"] + ".other"}
    with pytest.raises(AdapterRefusal) as refusal:
        investigation._runtime(name, tmp_path, expected=_saved(spec, **changes))
    assert refusal.value.code == "runtime_mismatch"


@pytest.mark.parametrize("name", ["rci", "fsrt"])
def test_current_and_historical_pins_bind_with_the_saved_identity(tmp_path, monkeypatch, name):
    spec = _pins()[name]
    monkeypatch.setattr(investigation, "PinnedSubprocessAdapter", _Recorder)
    monkeypatch.setattr(_Recorder, "calls", [])
    entries = [spec, *spec.get("historical", [])]
    assert len(entries) >= 2, "these adapters carry a historical pin"
    for entry in entries:
        saved = _saved(entry)
        assert isinstance(investigation._runtime(name, tmp_path, expected=saved), _Recorder)
        repo, revision, module, kwargs = _Recorder.calls[-1]
        assert repo == tmp_path
        assert (revision, module) == (entry["revision"], entry["module"])
        assert kwargs == {"python_executable": sys.executable,
                          "expected_python_sha256": saved["python_sha256"],
                          "expected_python_version": saved["python_version"],
                          "expected_dependencies": saved["dependencies"]}
    # A fresh binding without a saved identity uses the current pin and no expectations.
    assert isinstance(investigation._runtime(name, tmp_path), _Recorder)
    repo, revision, module, kwargs = _Recorder.calls[-1]
    assert (revision, module, kwargs) == (spec["revision"], spec["module"], {"python_executable": sys.executable})
