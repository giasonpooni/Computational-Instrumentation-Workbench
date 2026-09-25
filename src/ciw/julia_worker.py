"""Host-owned persistent Julia worker sessions over a framed binary stdio protocol.

The Julia executable is an explicit operator binding and is never read from a
saved workspace. The worker environment (project, manifest and worker source)
ships inside the ``ciw`` package and is pinned by ``julia-runtime.json``. A
worker session has its own identity and a monotonically increasing occurrence
number; both are separate from content identities. Any timeout, malformed
response, unexpected EOF or process failure ends the session: the active
occurrence fails, the process is terminated and reaped, and later work starts a
fresh session with a new identity. Nothing is retried silently.
"""

from __future__ import annotations

import atexit
from collections import deque
from hashlib import sha256
from importlib import resources
import json
import os
from pathlib import Path
import queue
import re
import struct
import subprocess
import threading
import time
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json

ADAPTER_VERSION = "ciw-julia-worker-v1"
RUNTIME_SCHEMA = "ciw.julia-worker-runtime.v1"
HELLO_SCHEMA = "ciw.julia-worker-hello.v1"
OCCURRENCE_SCHEMA = "ciw.julia-worker-occurrence.v1"
PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 4 << 20
MAX_HELLO_BYTES = 64 * 1024
DEFAULT_TIMEOUT = 60.0
HELLO_TIMEOUT = 600.0
QUEUE_LIMIT = 8
STDERR_LIMIT = 64 * 1024
PIN_RESOURCE = "julia-runtime.json"
ENVIRONMENT_RESOURCE = "julia"
_HEX64 = re.compile(r"[a-f0-9]{64}")
_HOST_PATH_KEYS = ("julia_executable", "environment_root")


def _refuse(code: str, message: str) -> None:
    raise AdapterRefusal(code, message)


def _digest_file(path: Path) -> str:
    try:
        with Path(path).open("rb") as stream:
            digest = sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError as exc:
        raise AdapterRefusal("RUNTIME_UNAVAILABLE", f"Cannot read {path}") from exc


def runtime_pin() -> dict:
    """The packaged Julia environment pin; saved workspaces cannot alter it."""
    pin = json.loads(resources.files("ciw").joinpath(PIN_RESOURCE).read_text(encoding="utf-8"))
    for key in ("schema", "julia_version", "protocol_version", "operations", "worker", "project", "manifest", "packages"):
        if key not in pin:
            raise ValueError("Packaged Julia runtime pin is incomplete: " + key)
    if pin["schema"] != "ciw.julia-runtime-pin.v1" or pin["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported packaged Julia runtime pin")
    return pin


def environment_root() -> Path:
    root = Path(str(resources.files("ciw").joinpath(ENVIRONMENT_RESOURCE)))
    if not root.is_dir():
        raise AdapterRefusal("RUNTIME_UNAVAILABLE", "The packaged Julia environment directory is unavailable")
    return root


def verify_environment(root: Path, pin: dict) -> dict:
    """Digest the packaged worker files and require the committed pin."""
    files = {"worker": root / pin["worker"]["path"], "project": root / pin["project"]["path"],
             "manifest": root / pin["manifest"]["path"]}
    digests = {}
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            _refuse("RUNTIME_PIN_MISMATCH", f"Packaged Julia {name} file is missing or is a link")
        digests[name] = _digest_file(path)
        if digests[name] != pin[name]["sha256"]:
            _refuse("RUNTIME_PIN_MISMATCH", f"Packaged Julia {name} bytes differ from the committed pin")
    return {"paths": files, "digests": digests}


def _string(value, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"Worker hello field {name} must be a bounded nonempty string")
    return value


def check_hello(hello, pin: dict, digests: dict) -> dict:
    """Compare the worker's self-reported identities with the host's expectations."""
    try:
        if not isinstance(hello, dict) or hello.get("schema") != HELLO_SCHEMA:
            raise ValueError("Unknown worker hello schema")
        if set(hello) != {"schema", "protocol_version", "operations", "julia_version", "platform", "threads",
                          "options", "worker_source_sha256", "project_sha256", "manifest_sha256", "sysimage",
                          "packages", "solver"}:
            raise ValueError("Worker hello fields differ from protocol version 1")
        if hello["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("Worker protocol version mismatch")
        if _string(hello["julia_version"], "julia_version") != pin["julia_version"]:
            raise ValueError(f"Julia {hello['julia_version']} is not the pinned {pin['julia_version']}")
        for name in ("worker_source_sha256", "project_sha256", "manifest_sha256"):
            reported = _string(hello[name], name)
            expected = digests[name.split("_")[0]] if name != "worker_source_sha256" else digests["worker"]
            if not _HEX64.fullmatch(reported) or reported != expected:
                raise ValueError(f"Worker loaded a different {name.rsplit('_', 1)[0]} than the packaged pin")
        operations = hello["operations"]
        if not isinstance(operations, list) or not set(pin["operations"]) <= set(operations) or any(not isinstance(o, str) for o in operations):
            raise ValueError("Worker operation allowlist differs from the pin")
        if hello["threads"] != 1:
            raise ValueError("Worker must run with exactly one thread")
        options = hello["options"]
        if not isinstance(options, dict) or set(options) != {"startup_file", "fast_math", "check_bounds", "opt_level"}:
            raise ValueError("Worker options are incomplete")
        if options["startup_file"] != "disabled" or options["fast_math"] != "default" or options["check_bounds"] != 0:
            raise ValueError("Worker started with an unapproved startup file, fast-math or bounds-check setting")
        if type(options["opt_level"]) is not int or not 0 <= options["opt_level"] <= 3:
            raise ValueError("Worker optimization level is malformed")
        platform = hello["platform"]
        if not isinstance(platform, dict) or set(platform) != {"machine", "arch", "kernel", "word_size"} or platform["word_size"] != 64:
            raise ValueError("Worker platform identity is incomplete or not 64-bit")
        for key in ("machine", "arch", "kernel"):
            _string(platform[key], "platform." + key)
        packages = hello["packages"]
        if not isinstance(packages, dict) or set(packages) != set(pin["packages"]):
            raise ValueError("Worker package set differs from the pin")
        for name, expected in pin["packages"].items():
            reported = packages[name]
            if not isinstance(reported, dict) or set(reported) != {"uuid", "version", "git_tree_sha1"} or reported != expected:
                raise ValueError(f"Package {name} differs from the pinned identity")
        sysimage = hello["sysimage"]
        if not isinstance(sysimage, dict) or set(sysimage) != {"name", "sha256"} or not _HEX64.fullmatch(_string(sysimage["sha256"], "sysimage")):
            raise ValueError("Worker system image identity is malformed")
        _string(sysimage["name"], "sysimage.name")
        solver = hello["solver"]
        if (not isinstance(solver, dict) or set(solver) != {"algorithm", "package", "version", "arithmetic"} or
                solver["algorithm"] != "Tsit5" or solver["package"] != "OrdinaryDiffEqTsit5" or
                solver["version"] != pin["packages"]["OrdinaryDiffEqTsit5"]["version"] or solver["arithmetic"] != "binary64"):
            raise ValueError("Worker solver identity differs from the pin")
        return hello
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed worker hello") from exc


def runtime_projection(runtime: dict) -> dict:
    """Identity without host-specific paths, for replay comparison."""
    return {key: value for key, value in runtime.items() if key not in _HOST_PATH_KEYS}


def check_runtime(runtime, pin: dict | None = None) -> None:
    """Structural check of a retained runtime identity; no process is started."""
    pin = pin or runtime_pin()
    if not isinstance(runtime, dict) or set(runtime) != {
            "schema", "adapter_version", "protocol_version", "julia_executable", "environment_root", "julia_sha256",
            "julia_version", "platform", "threads", "options", "worker_source_sha256", "project_sha256",
            "manifest_sha256", "sysimage", "packages", "operations", "solver"}:
        raise ValueError("Invalid Julia worker runtime identity fields")
    if runtime["schema"] != RUNTIME_SCHEMA or runtime["adapter_version"] != ADAPTER_VERSION:
        raise ValueError("Unapproved Julia worker adapter identity")
    if not isinstance(runtime["julia_sha256"], str) or not _HEX64.fullmatch(runtime["julia_sha256"]):
        raise ValueError("Invalid Julia executable digest")
    for key in ("julia_executable", "environment_root"):
        _string(runtime[key], key)
    hello = {key: runtime[key] for key in ("protocol_version", "operations", "julia_version", "platform", "threads", "options",
                                          "worker_source_sha256", "project_sha256", "manifest_sha256", "sysimage", "packages", "solver")}
    hello["schema"] = HELLO_SCHEMA
    digests = {"worker": pin["worker"]["sha256"], "project": pin["project"]["sha256"], "manifest": pin["manifest"]["sha256"]}
    check_hello(hello, pin, digests)


class WorkerFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class JuliaWorkerSession:
    """One worker process: explicit executable, packaged environment, one request in flight."""

    def __init__(self, executable: str | Path, *, root: Path | None = None, pin: dict | None = None,
                 timeout_seconds: float = DEFAULT_TIMEOUT, hello_timeout: float = HELLO_TIMEOUT) -> None:
        if type(timeout_seconds) not in (int, float) or not timeout_seconds > 0:
            raise ValueError("timeout_seconds must be positive")
        self.pin = pin or runtime_pin()
        self.root = Path(root) if root is not None else environment_root()
        # A launcher link (for example juliaup's) is followed once here; the
        # digest and identity describe the resolved executable file.
        self.executable = Path(executable).expanduser().resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            _refuse("RUNTIME_UNAVAILABLE", "The bound Julia executable is not an executable regular file")
        self.environment = verify_environment(self.root, self.pin)
        self.julia_sha256 = _digest_file(self.executable)
        self.session_id = "julia-worker-" + uuid.uuid4().hex
        self.timeout_seconds = float(timeout_seconds)
        self.occurrence = 0
        self.failure: WorkerFailure | None = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(QUEUE_LIMIT)
        self._frames: queue.Queue = queue.Queue()
        self._stderr: deque = deque()
        self._stderr_bytes = 0
        self._process = self._start()
        self._reader = threading.Thread(target=self._read_frames, daemon=True)
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self._drain.start()
        self.identity = self._handshake(hello_timeout)

    # -- process ------------------------------------------------------------

    def _command(self) -> list[str]:
        return [str(self.executable), "--startup-file=no", "--history-file=no", "--color=no", "--threads=1",
                "--project=" + str(self.root), str(self.environment["paths"]["worker"])]

    def _start(self) -> subprocess.Popen:
        environment = dict(os.environ)
        for key in ("JULIA_PROJECT", "JULIA_LOAD_PATH", "JULIA_NUM_THREADS", "JULIA_EXCLUSIVE", "JULIA_CPU_TARGET"):
            environment.pop(key, None)
        environment.update({"JULIA_LOAD_PATH": "@", "JULIA_PKG_OFFLINE": "true", "JULIA_PKG_PRECOMPILE_AUTO": "0",
                            "JULIA_NUM_THREADS": "1"})
        try:
            return subprocess.Popen(self._command(), cwd=str(self.root), env=environment, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                                    start_new_session=os.name == "posix")
        except (OSError, ValueError) as exc:
            raise AdapterRefusal("RUNTIME_UNAVAILABLE", "Cannot start the bound Julia worker") from exc

    def _read_frames(self) -> None:
        stream = self._process.stdout
        try:
            while True:
                header = stream.read(4)
                if len(header) < 4:
                    self._frames.put(("eof", None))
                    return
                count = struct.unpack("<I", header)[0]
                if count > MAX_RESPONSE_BYTES:
                    self._frames.put(("oversize", None))
                    return
                payload = stream.read(count)
                if len(payload) < count:
                    self._frames.put(("eof", None))
                    return
                self._frames.put(("frame", payload))
        except (OSError, ValueError):
            self._frames.put(("eof", None))

    def _drain_stderr(self) -> None:
        stream = self._process.stderr
        try:
            while chunk := stream.read1(8192):
                self._stderr.append(chunk)
                self._stderr_bytes += len(chunk)
                while self._stderr_bytes > STDERR_LIMIT and len(self._stderr) > 1:
                    self._stderr_bytes -= len(self._stderr.popleft())
        except (OSError, ValueError):
            pass

    @property
    def alive(self) -> bool:
        return self.failure is None and self._process.poll() is None

    def diagnostics(self) -> str:
        return b"".join(self._stderr).decode("utf-8", errors="replace")[-4096:]

    def terminate(self, code: str = "WORKER_CLOSED", message: str = "Worker session ended by the host") -> None:
        """End this session; reap the process; keep the first recorded failure."""
        if self.failure is None:
            self.failure = WorkerFailure(code, message)
        process = self._process
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, 9)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin,):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    # -- protocol -----------------------------------------------------------

    def _next_frame(self, deadline: float) -> bytes:
        remaining = deadline - time.monotonic()
        try:
            kind, payload = self._frames.get(timeout=max(remaining, 0.0))
        except queue.Empty:
            raise WorkerFailure("TIMEOUT", "The Julia worker exceeded its execution deadline") from None
        if kind == "frame":
            return payload
        if kind == "oversize":
            raise WorkerFailure("OUTPUT_LIMIT", "The Julia worker sent a frame above the response bound")
        raise WorkerFailure("WORKER_FAILED", "The Julia worker ended its output stream: " + self.diagnostics())

    def _handshake(self, hello_timeout: float) -> dict:
        try:
            payload = self._next_frame(time.monotonic() + hello_timeout)
            if len(payload) > MAX_HELLO_BYTES:
                raise WorkerFailure("MALFORMED_RESPONSE", "Worker hello exceeds its byte bound")
            try:
                parsed = _json(payload)
            except AdapterRefusal as exc:
                raise WorkerFailure(exc.code, "Worker hello is not finite JSON") from exc
            try:
                hello = check_hello(parsed, self.pin, self.environment["digests"])
            except ValueError as exc:
                raise WorkerFailure("RUNTIME_PIN_MISMATCH", str(exc)) from exc
        except WorkerFailure as failure:
            self.terminate(failure.code, str(failure))
            raise AdapterRefusal(failure.code, str(failure)) from failure
        identity = {"schema": RUNTIME_SCHEMA, "adapter_version": ADAPTER_VERSION, "protocol_version": PROTOCOL_VERSION,
                    "julia_executable": str(self.executable), "environment_root": str(self.root),
                    "julia_sha256": self.julia_sha256}
        for key in ("julia_version", "platform", "threads", "options", "worker_source_sha256", "project_sha256",
                    "manifest_sha256", "sysimage", "packages", "operations", "solver"):
            identity[key] = hello[key]
        return identity

    def execute(self, payload, *, timeout_seconds: float | None = None) -> tuple[int, bytes, dict]:
        """Send one request frame; return (occurrence, response payload, occurrence metadata).

        ``payload`` is the exact request bytes or a callable that builds them
        from the occurrence number allocated under the in-flight lock. A failed
        occurrence ends the session; the caller must obtain a fresh session for
        later work. Nothing is retried here.
        """
        if not callable(payload) and (not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_REQUEST_BYTES):
            _refuse("INPUT_LIMIT", "Julia worker request must be 1..1048576 exact bytes")
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if not self._slots.acquire(blocking=False):
            _refuse("WORKER_BUSY", "The Julia worker request queue is full")
        try:
            with self._lock:
                if not self.alive:
                    _refuse("WORKER_UNAVAILABLE", "The Julia worker session has ended; start a fresh session")
                self.occurrence += 1
                number = self.occurrence
                if callable(payload):
                    payload = payload(number)
                    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_REQUEST_BYTES:
                        _refuse("INPUT_LIMIT", "Julia worker request must be 1..1048576 exact bytes")
                try:
                    try:
                        self._process.stdin.write(struct.pack("<I", len(payload)) + payload)
                        self._process.stdin.flush()
                    except (OSError, ValueError) as exc:
                        raise WorkerFailure("WORKER_FAILED", "Cannot deliver the request to the Julia worker: " + self.diagnostics()) from exc
                    deadline = time.monotonic() + timeout
                    response = self._next_frame(deadline)
                    metadata_bytes = self._next_frame(deadline)
                    if len(metadata_bytes) > 4096:
                        raise WorkerFailure("MALFORMED_RESPONSE", "Worker occurrence metadata exceeds its bound")
                    try:
                        metadata = _json(metadata_bytes)
                    except AdapterRefusal as exc:
                        raise WorkerFailure("MALFORMED_RESPONSE", "Worker occurrence metadata is not finite JSON") from exc
                    if (not isinstance(metadata, dict) or set(metadata) != {"schema", "request_number", "worker_occurrence", "elapsed_ns"} or
                            metadata["schema"] != OCCURRENCE_SCHEMA or metadata["request_number"] != number or
                            metadata["worker_occurrence"] != number or type(metadata["elapsed_ns"]) is not int or metadata["elapsed_ns"] < 0):
                        raise WorkerFailure("MALFORMED_RESPONSE", "Worker response does not bind the request occurrence")
                    return number, response, metadata
                except WorkerFailure as failure:
                    self.terminate(failure.code, str(failure))
                    raise AdapterRefusal(failure.code, str(failure)) from failure
        finally:
            self._slots.release()

    def fail(self, code: str, message: str) -> None:
        """Let the protocol consumer end the session on a content-level violation."""
        self.terminate(code, message)


class JuliaWorkerPool:
    """Process-local registry of live sessions keyed by the bound executable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, JuliaWorkerSession] = {}

    def session(self, executable: str | Path, **options) -> JuliaWorkerSession:
        key = str(Path(executable).expanduser().absolute())
        with self._lock:
            current = self._sessions.get(key)
            if current is not None and current.alive:
                return current
            if current is not None:
                current.terminate()
            session = JuliaWorkerSession(executable, **options)
            self._sessions[key] = session
            return session

    def restart(self, executable: str | Path, **options) -> JuliaWorkerSession:
        key = str(Path(executable).expanduser().absolute())
        with self._lock:
            current = self._sessions.pop(key, None)
        if current is not None:
            current.terminate()
        return self.session(executable, **options)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.terminate()


POOL = JuliaWorkerPool()
atexit.register(POOL.shutdown)
