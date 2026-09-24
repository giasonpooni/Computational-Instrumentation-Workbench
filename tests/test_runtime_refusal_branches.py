"""Dedicated assertions for adapter runtime refusal branches that other tests reach only indirectly."""
import os
from pathlib import Path
import sys

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.adapters import subprocess as adapter


def _code(call):
    with pytest.raises(AdapterRefusal) as caught:
        call()
    return caught.value.code


def test_a_runtime_that_cannot_start_is_unavailable(tmp_path):
    missing = str(tmp_path / "no-such-runtime")
    assert _code(lambda: adapter._bounded_process([missing], cwd=tmp_path, timeout=5, limit=1024)) == "RUNTIME_UNAVAILABLE"


@pytest.mark.skipif(os.name != "posix", reason="escaping the runtime's process group is a POSIX session behavior")
def test_a_runtime_that_leaves_an_output_pipe_open_is_refused(tmp_path):
    # The grandchild leaves the runtime's session, so stopping the runtime's
    # process group cannot close the stdout it inherited.
    script = ("import os, sys, time\n"
              "if os.fork() == 0:\n"
              "    os.setsid(); time.sleep(4); os._exit(0)\n"
              "sys.exit(0)\n")
    code = _code(lambda: adapter._bounded_process([sys.executable, "-c", script], cwd=tmp_path, timeout=10, limit=1024))
    assert code == "RUNTIME_IO"


def _bare_adapter(python_executable):
    bound = object.__new__(adapter.PinnedSubprocessAdapter)
    bound.python_executable = Path(python_executable)
    return bound


def test_an_unreadable_interpreter_is_unavailable(tmp_path):
    bound = _bare_adapter(tmp_path / "missing-python")
    assert _code(bound._executable_digest) == "RUNTIME_UNAVAILABLE"


@pytest.mark.parametrize("probe", [(1, b""), (0, b'{"python_version": "3.12.0"}'), (0, b"[]")])
def test_a_failed_or_malformed_runtime_probe_is_unavailable(tmp_path, monkeypatch, probe):
    bound = _bare_adapter(sys.executable)
    monkeypatch.setattr(bound, "_run", lambda *args, **kwargs: probe, raising=False)
    assert _code(bound._probe) == "RUNTIME_UNAVAILABLE"


def test_a_saved_runtime_off_the_pin_allowlist_is_a_runtime_mismatch(tmp_path):
    from ciw.investigation import _runtime
    saved = {"revision": "0" * 40, "module": "not.the.adapter", "python_sha256": "0" * 64,
             "python_version": "3.12.0", "dependencies": {}}
    assert _code(lambda: _runtime("rci", tmp_path, expected=saved)) == "runtime_mismatch"
