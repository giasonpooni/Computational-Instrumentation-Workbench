"""Pinned, operator-bound subprocess adapters for external scientific repositories.

Persisted investigations contain identity metadata, never executable bindings.
An operator must create this binding again when opening an investigation. Source
and interpreter pins detect drift; they do not sandbox a trusted domain runtime
or authenticate its installed dependency binaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import new as new_hash, sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any

from .protocol import AdapterRefusal

ADAPTER_VERSION = "ciw-pinned-subprocess-v1"
_CACHE_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_BOOTSTRAP = """import importlib.util, pathlib, runpy, sys
source = pathlib.Path(sys.argv[1]).resolve()
module = sys.argv[2]
sys.path.insert(0, str(source))
spec = importlib.util.find_spec(module)
if spec is not None and spec.submodule_search_locations is not None:
    spec = importlib.util.find_spec(module + '.__main__')
if spec is None or spec.origin is None or not pathlib.Path(spec.origin).resolve().is_relative_to(source):
    raise RuntimeError('Adapter entry point resolved outside the bound source root')
runpy.run_module(module, run_name='__main__', alter_sys=True)
"""
_RUNTIME_PROBE = """import importlib.metadata as metadata, json, platform
dependencies = {}
for name in ('numpy', 'scipy'):
    try:
        dependencies[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        dependencies[name] = None
print(json.dumps({'python_version': platform.python_version(), 'dependencies': dependencies}, allow_nan=False))
"""


def _refuse(code: str, message: str) -> None:
    raise AdapterRefusal(code, message)


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _bounded_process(
    command: list[str], *, cwd: Path, timeout: float, limit: int, stdin: bytes = b""
) -> tuple[int, bytes]:
    """Drain both pipes concurrently, retaining at most ``limit`` bytes in total."""
    output: list[bytes] = []
    total = 0
    lock = threading.Lock()
    exceeded = threading.Event()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryFile() as request:
        request.write(stdin)
        request.seek(0)
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdin=request,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                start_new_session=os.name == "posix",
            )
        except (OSError, ValueError) as exc:
            raise AdapterRefusal("RUNTIME_UNAVAILABLE", "Cannot start the bound runtime") from exc

        def drain(pipe: Any, retain: bool) -> None:
            nonlocal total
            try:
                while chunk := pipe.read1(8192):
                    with lock:
                        remaining = max(0, limit - total)
                        total += len(chunk)
                        if retain and remaining:
                            output.append(chunk[:remaining])
                        if total > limit:
                            exceeded.set()
                            _stop(process)
                            return
            finally:
                pipe.close()

        readers = [
            threading.Thread(target=drain, args=(process.stdout, True), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, False), daemon=True),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(process)
            process.wait(timeout=5)
        finally:
            for reader in readers:
                reader.join(timeout=0.5)
            if any(reader.is_alive() for reader in readers):
                _stop(process)
                for reader in readers:
                    reader.join(timeout=0.5)
        if exceeded.is_set():
            _refuse("OUTPUT_LIMIT", "The bound runtime exceeded its combined output limit")
        if timed_out:
            _refuse("TIMEOUT", "The bound runtime exceeded its execution deadline")
        if any(reader.is_alive() for reader in readers):
            _refuse("RUNTIME_IO", "The bound runtime left an output pipe open")
        return process.returncode, b"".join(output)


def _json(payload: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise ValueError("nonfinite JSON number")

    def finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    try:
        result = json.loads(payload.decode("utf-8"), parse_constant=constant, object_pairs_hook=pairs)
        finite(result)
        return result
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise AdapterRefusal("MALFORMED_RESPONSE", "The bound runtime did not return finite, unambiguous JSON") from exc


class PinnedSubprocessAdapter:
    """An explicit local binding, which must never be loaded from workspace JSON.

    ``revision`` must be a full Git commit ID. ``source_root`` is relative to the
    checkout and is the only added import root. Isolated Python mode removes cwd,
    PYTHONPATH and user-site imports. Installed system/venv dependencies remain
    part of the trusted environment; NumPy/SciPy versions are recorded and pinned.
    """

    def __init__(
        self, repository_root: str | Path, revision: str, module: str, *,
        source_root: str = "src", python_executable: str | Path = sys.executable,
        timeout_seconds: float = 30.0, max_output_bytes: int = 4 * 1024 * 1024,
        expected_python_sha256: str | None = None,
        expected_python_version: str | None = None,
        expected_dependencies: Mapping[str, str | None] | None = None,
    ) -> None:
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
            raise ValueError("revision must be a full lowercase Git commit ID")
        if not isinstance(module, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise ValueError("module must be a dotted Python module name")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(max_output_bytes) is not int or max_output_bytes < 256:
            raise ValueError("max_output_bytes must be an integer of at least 256")
        self.repository_root = Path(repository_root).expanduser().resolve()
        relative_source = Path(source_root)
        if relative_source.is_absolute() or ".." in relative_source.parts:
            raise ValueError("source_root must be relative to the bound repository")
        self.source_root = (self.repository_root / relative_source).resolve()
        if not self.source_root.is_relative_to(self.repository_root) or not self.source_root.is_dir():
            raise ValueError("source_root must be a directory within the bound repository")
        self.source_root_relative = self.source_root.relative_to(self.repository_root).as_posix()
        self.revision = revision
        self.module = module
        # Keep the venv entry point, rather than resolving its interpreter symlink.
        self.python_executable = Path(python_executable).expanduser().absolute()
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self._source_tree = self._verify_source()
        self._python_sha256 = self._executable_digest()
        if expected_python_sha256 is not None and expected_python_sha256 != self._python_sha256:
            _refuse("RUNTIME_PIN_MISMATCH", "Python executable digest differs from the declared pin")
        probe = self._probe()
        self._python_version = probe["python_version"]
        self._dependencies = probe["dependencies"]
        if expected_python_version is not None and expected_python_version != self._python_version:
            _refuse("RUNTIME_PIN_MISMATCH", "Python version differs from the declared pin")
        if expected_dependencies is not None and dict(expected_dependencies) != self._dependencies:
            _refuse("RUNTIME_PIN_MISMATCH", "Dependency versions differ from the declared pin")

    def _git(self, *arguments: str) -> bytes:
        code, data = _bounded_process(
            ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", *arguments], cwd=self.repository_root,
            timeout=min(self.timeout_seconds, 10.0), limit=max(self.max_output_bytes, 4 * 1024 * 1024),
        )
        if code:
            _refuse("SOURCE_PIN_MISMATCH", "Cannot verify the bound Git checkout")
        return data

    def _verify_source(self) -> str:
        actual_root = self._git("rev-parse", "--show-toplevel").decode().strip()
        if Path(actual_root).resolve() != self.repository_root:
            _refuse("SOURCE_PIN_MISMATCH", "The bound directory is not a repository root")
        if self._git("rev-parse", "HEAD").decode().strip() != self.revision:
            _refuse("SOURCE_PIN_MISMATCH", "The bound checkout is not at the declared revision")
        # Include ignored files: an ignored Python module can still shadow a pin.
        exclusions = [f":(exclude,glob)**/{name}/**" for name in sorted(_CACHE_DIRS)]
        untracked = self._git("ls-files", "--others", "-z", "--", ".", *exclusions)
        if untracked:
            _refuse("SOURCE_PIN_MISMATCH", "The bound repository contains untracked files outside permitted caches")
        entries = self._git("ls-tree", "-r", "-z", "HEAD")
        object_format = self._git("rev-parse", "--show-object-format").decode().strip()
        if object_format not in {"sha1", "sha256"}:
            _refuse("SOURCE_PIN_MISMATCH", "Unsupported Git object format")
        tracked: set[str] = set()
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            mode, kind, expected = metadata.decode("ascii").split()
            name = os.fsdecode(raw_name)
            path = self.repository_root / name
            tracked.add(name)
            if kind != "blob":
                _refuse("SOURCE_PIN_MISMATCH", "Submodules require a separately verified runtime binding")
            try:
                if mode == "120000":
                    # Source links can resolve to unpinned external modules.
                    if path.is_relative_to(self.source_root):
                        _refuse("SOURCE_PIN_MISMATCH", "Import-root symlinks are not accepted")
                    data = os.fsencode(os.readlink(path))
                else:
                    if path.is_symlink() or not path.is_file():
                        _refuse("SOURCE_PIN_MISMATCH", "A tracked source file changed type or disappeared")
                    data = path.read_bytes()
                    if os.name == "posix" and bool(path.stat().st_mode & 0o111) != (mode == "100755"):
                        _refuse("SOURCE_PIN_MISMATCH", "A tracked source file changed executable mode")
                digest = new_hash(object_format, b"blob " + str(len(data)).encode("ascii") + b"\0" + data)
            except OSError as exc:
                raise AdapterRefusal("SOURCE_PIN_MISMATCH", "Cannot read the pinned source tree") from exc
            if digest.hexdigest() != expected:
                _refuse("SOURCE_PIN_MISMATCH", "Tracked file bytes differ from the declared revision")
        # Detect staged/index-only edits even if working bytes match HEAD.
        self._git("diff", "--cached", "--quiet", "--no-ext-diff", "--no-textconv", "HEAD", "--")
        prefix = "" if self.source_root_relative == "." else self.source_root_relative + "/"
        module_path = prefix + self.module.replace(".", "/")
        if module_path + ".py" not in tracked and module_path + "/__main__.py" not in tracked:
            _refuse("SOURCE_PIN_MISMATCH", "The adapter entry point is not part of the pinned source tree")
        return self._git("rev-parse", "HEAD^{tree}").decode().strip()

    def _executable_digest(self) -> str:
        try:
            with self.python_executable.open("rb") as stream:
                digest = sha256()
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                return digest.hexdigest()
        except OSError as exc:
            raise AdapterRefusal("RUNTIME_UNAVAILABLE", "Cannot read the bound Python executable") from exc

    def _run(self, code: str, arguments: list[str], stdin: bytes = b"") -> tuple[int, bytes]:
        with tempfile.TemporaryDirectory(prefix="ciw-adapter-") as directory:
            return _bounded_process(
                # -B disables writes only. A fresh prefix also prevents reads of
                # forged/stale repository or installation bytecode caches.
                [str(self.python_executable), "-I", "-B", "-X",
                 f"pycache_prefix={directory}/bytecode", "-c", code, *arguments],
                cwd=Path(directory), timeout=self.timeout_seconds, limit=self.max_output_bytes, stdin=stdin,
            )

    def _probe(self) -> dict[str, Any]:
        code, data = self._run(_RUNTIME_PROBE, [])
        if code:
            _refuse("RUNTIME_UNAVAILABLE", "Cannot inspect the bound Python runtime")
        result = _json(data)
        if not isinstance(result, dict) or set(result) != {"python_version", "dependencies"}:
            _refuse("RUNTIME_UNAVAILABLE", "The Python runtime identity probe failed")
        return result

    def runtime_identity(self) -> dict[str, Any]:
        tree = self._verify_source()
        if tree != self._source_tree or self._executable_digest() != self._python_sha256:
            _refuse("RUNTIME_PIN_MISMATCH", "The bound source or executable identity changed")
        probe = self._probe()
        if probe != {"python_version": self._python_version, "dependencies": self._dependencies}:
            _refuse("RUNTIME_PIN_MISMATCH", "The Python or dependency versions changed")
        return {
            "schema": "ciw.subprocess-runtime.v1", "adapter_version": ADAPTER_VERSION,
            "repository_root": str(self.repository_root), "revision": self.revision,
            "source_tree": tree, "module": self.module, "source_root": self.source_root_relative,
            "python_executable": str(self.python_executable), "python_sha256": self._python_sha256,
            "python_version": self._python_version, "dependencies": dict(self._dependencies),
        }

    def invoke(self, operation_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id or not isinstance(inputs, Mapping):
            _refuse("INVALID_INPUT", "An operation ID and JSON input object are required")
        def check_keys(value: Any) -> None:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("JSON object keys must be strings")
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    check_keys(child)
        try:
            check_keys(dict(inputs))
            request = json.dumps(
                {"schema": "ciw.adapter-request.v1", "operation_id": operation_id, "inputs": dict(inputs)},
                allow_nan=False, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii")
        except (TypeError, ValueError, RecursionError) as exc:
            raise AdapterRefusal("INVALID_INPUT", "Adapter inputs must be finite JSON") from exc
        if len(request) > self.max_output_bytes:
            _refuse("INPUT_LIMIT", "Adapter input exceeds the configured request limit")
        self.runtime_identity()
        code, payload = self._run(_BOOTSTRAP, [str(self.source_root), self.module], request)
        if code:
            _refuse("RUNTIME_FAILED", "The bound runtime exited unsuccessfully")
        response = _json(payload)
        self.runtime_identity()
        if not isinstance(response, dict) or response.get("schema") != "ciw.adapter-response.v1":
            _refuse("MALFORMED_RESPONSE", "The runtime returned an unknown response envelope")
        if response.get("status") == "ok" and set(response) == {"schema", "status", "data"}:
            if isinstance(response["data"], dict):
                return response["data"]
        elif response.get("status") == "refused" and set(response) == {"schema", "status", "refusal"}:
            refusal = response["refusal"]
            if (isinstance(refusal, dict)
                    and {"code", "message"} <= refusal.keys() <= {"code", "message", "reason_code"}
                    and all(isinstance(value, str) and value.strip() for value in refusal.values())):
                raise AdapterRefusal(
                    refusal["code"], refusal["message"], reason_code=refusal.get("reason_code"),
                )
        _refuse("MALFORMED_RESPONSE", "The runtime returned an invalid success/refusal envelope")
