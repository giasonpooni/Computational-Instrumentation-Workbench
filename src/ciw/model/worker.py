"""Host side of the persistent Julia model worker, below the SCR execution seam.

The host owns the worker process. Each request is an SCR specification triple
(program, configuration, input) whose program bytes are a registered operation
descriptor bound to the worker's runtime digest. The host recomputes every
identity the worker echoes and never trusts one it did not derive itself.

Lifecycle rules (docs/JULIA_SP1.md): one request in flight; framed messages
with size limits; diagnostics only on stderr; a handshake that must match the
pinned runtime before any work; a timeout, malformed or truncated response,
unexpected EOF or process failure ends the session, fails that occurrence and
is never retried silently. Later work starts a new session with a new identity.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from importlib import resources
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import uuid

from . import codec
from .commitments import Specification, commit_hex, computation_identity, output_identity

PROTOCOL = "ciw.julia-model-worker.v1"
FRAME_MAGIC = b"CIWF"
MAX_FRAME = 64 * 1024 * 1024
SOURCE_TAG = "ciw.julia-worker.source.v1"
STARTUP_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 300.0
STDERR_LINES = 400
PROFILES = {"core": ("ciw.model.simulate.v1", "ciw.model.linearize.v1", "ciw.model.measurement-selection.v1"),
            "symbolic": ("ciw.model.symbolic.v1",)}
OPERATION_PROFILE = {operation: profile for profile, operations in PROFILES.items() for operation in operations}

DESCRIPTORS = {
    "ciw.model.simulate.v1": (
        "ciw.julia.model-simulate.v1\n"
        "input: CIWB v1 {initial_state: f64[n] declared units, inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, sample_times: f64[m]}\n"
        "configuration: CIWB v1 {algorithm: Tsit5, abstol: f64[n] state units, controller: algorithm_default, dtmax: f64|null, initial_dt: solver_automatic, internalnorm: ODE_DEFAULT_NORM, maxiters: i64, reltol: f64, save_policy: interpolated_saveat, span: first_to_last_sample}\n"
        "semantics: x_si = scale .* x; derived in declared order; dx/dtau = rhs_si * scale(tau) ./ scale(x); binary64\n"
        "output: CIWB v1 {derived: {symbol: f64[m]}, observations: {symbol: f64[m]}, retcode: str, stats: {naccept, nf, nreject}, t: f64[m], x: f64[m,n]}\n"
        "faults: 2=malformed, 3=solver unsuccessful, 4=nonfinite or domain error, 5=incomplete sample coverage\n"),
    "ciw.model.linearize.v1": (
        "ciw.julia.model-linearize.v1\n"
        "input: CIWB v1 {inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, state: f64[n], time: f64}\n"
        "configuration: CIWB v1 {analysis: ControlSystemsBase ss/poles/ctrb/obsv, differentiation: ForwardDiff.jacobian, rank_rtol: f64}\n"
        "semantics: A=d(dx/dtau)/dx, B=d(dx/dtau)/du, C=dy/dx, D=dy/du in declared units at the operating point\n"
        "output: CIWB v1 {A, B, C, D: f64[.,.], controllability_rank: i64, observability_rank: i64, poles_imag: f64[n], poles_real: f64[n], rhs: f64[n]}\n"
        "faults: 2=malformed, 4=nonfinite or domain error\n"),
    "ciw.model.measurement-selection.v1": (
        "ciw.julia.model-measurement-selection.v1\n"
        "input: CIWB v1 {budget: f64, candidates: [{cost: f64, observation: str, time: f64}], design_parameters: [str], initial_state: f64[n], inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, max_count: i64, noise_variance: {observation: f64}, prior_variance: {parameter: f64}, start_time: f64}\n"
        "configuration: CIWB v1 {abstol: f64[n], algorithm: Tsit5, maxiters: i64, mip_rel_gap: 0.0, objective: maximin_prior_normalized_fisher_diagonal, random_seed: 0, reltol: f64, sensitivity: ForwardDiff_through_Tsit5, solver: HiGHS, threads: 1, time_limit_s: f64}\n"
        "semantics: G[i,j] = (dy_i/dtheta_j)^2 * prior_variance[j] / noise_variance[obs_i]; maximize s subject to sum_i w_i G[i,j] >= s, sum w_i cost_i <= budget, sum w_i <= max_count, w binary\n"
        "output: CIWB v1 {information: f64[c,p], objective: f64, primal_status: str, selected: [i64], sensitivities: f64[c,p], termination_status: str}\n"
        "faults: 2=malformed, 3=solver unsuccessful, 4=nonfinite or domain error\n"),
    "ciw.model.symbolic.v1": (
        "ciw.julia.model-symbolic.v1\n"
        "input: CIWB v1 {inputs: {symbol: f64}, lowered: ciw.model-lowered.v1, state: f64[n], time: f64}\n"
        "configuration: CIWB v1 {compile: ModelingToolkit.mtkcompile, jacobian: Symbolics.jacobian, latex: Latexify.latexify}\n"
        "semantics: build ModelingToolkit System from the lowered model in declared units; evaluate symbolic Jacobians at the operating point in declared state order\n"
        "output: CIWB v1 {compiled_unknowns: [str], declared_unknowns: [str], jacobian_compiled_to_declared: f64[n,n], jacobian_declared: f64[n,n], latex_equations: str, permutation: [i64]}\n"
        "faults: 2=malformed, 3=compilation failed, 4=nonfinite or domain error\n"),
}


class WorkerError(RuntimeError):
    """A worker session failed; the active occurrence has no result."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ExecutionRefused(WorkerError):
    """The worker refused to start the program (SCR ``unrunnable``); nothing ran."""


def worker_root() -> Path:
    return Path(resources.files("ciw.model") / "julia")


def pinned_runtime() -> dict:
    """The worker's pin, defined by its provider descriptor."""
    from ..pipelines import provider_descriptor
    return provider_descriptor("julia-model-worker")["pin"]


def provider_binding() -> dict:
    """What this worker executes, for ``pipelines.check_providers``."""
    root = worker_root()
    return {"pin": {"project_sha256": file_sha256(root / "Project.toml"),
                    "manifest_sha256": file_sha256(root / "Manifest.toml"),
                    "worker_source_sha256": source_sha256(root)},
            "operations": {profile: list(operations) for profile, operations in PROFILES.items()}}


def file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path | None = None) -> str:
    """Mirror of the worker's own source digest (bin/worker.jl and src/*.jl)."""
    root = Path(root or worker_root())
    names = sorted(["bin/worker.jl"] + [f"src/{path.name}" for path in (root / "src").glob("*.jl")])
    fields = []
    for name in names:
        fields += [name.encode("utf-8"), (root / name).read_bytes()]
    return commit_hex(SOURCE_TAG, fields)


def descriptor_sha256(operation: str) -> str:
    return hashlib.sha256(DESCRIPTORS[operation].encode("utf-8")).hexdigest()


def program_bytes(operation: str, runtime_digest: str) -> bytes:
    return (DESCRIPTORS[operation] + f"runtime: sha256:{runtime_digest}\n").encode("utf-8")


@dataclass(frozen=True)
class JuliaBinding:
    """An explicit host binding; never reconstructed from saved workspaces."""

    julia: Path
    profile: str = "core"
    root: Path | None = None
    depot: Path | None = None
    startup_timeout_s: float = STARTUP_TIMEOUT_S
    command_prefix: tuple = ()  # test-only protocol doubles run as [*prefix, profile]

    def resolved_root(self) -> Path:
        return Path(self.root or worker_root()).resolve()


@dataclass
class WorkerResult:
    """One checked execution, shaped like SCR's ``ExecutionResult`` plus session context."""

    operation: str
    specification: Specification
    specification_identity: str
    program_identity: str
    input_identity: str
    status: str
    exit_code: int
    output: bytes | None
    output_identity: str | None
    computation_identity: str | None
    detail: str | None
    session_id: str
    occurrence: int
    elapsed_ns: int
    runtime_digest: str
    wall_time_s: float = field(default=0.0)


def expected_identity(profile: str, root: Path) -> dict:
    """The handshake fields the host requires, from the pinned runtime and local files."""
    pin = pinned_runtime()
    return {
        "protocol": PROTOCOL, "profile": profile, "julia_version": pin["julia_version"],
        "threads": 1, "blas_threads": 1,
        "worker_source_sha256": source_sha256(root),
        "project_sha256": file_sha256(root / "Project.toml"),
        "manifest_sha256": file_sha256(root / "Manifest.toml"),
        "preferences_sha256": file_sha256(root / "LocalPreferences.toml"),
        "packages": {name: pin["packages"][name] for name in pin["profiles"][profile]},
        "operations": {operation: descriptor_sha256(operation) for operation in PROFILES[profile]},
    }


class JuliaWorker:
    """A persistent worker session manager. Not thread-safe: one request in flight."""

    def __init__(self, binding: JuliaBinding, *, verify_pins: bool = True):
        if binding.profile not in PROFILES:
            raise ValueError(f"Unknown worker profile: {binding.profile}")
        self.binding, self.verify_pins = binding, verify_pins
        self.process: subprocess.Popen | None = None
        self.session_id: str | None = None
        self.identity: dict | None = None
        self.context: dict | None = None
        self.runtime_digest: str | None = None
        self.occurrence = 0
        self.sessions: list[dict] = []
        self._frames: queue.Queue = queue.Queue()
        self._stderr: deque = deque(maxlen=STDERR_LINES)
        self.startup_s: float | None = None

    # -- lifecycle -----------------------------------------------------------------
    def _command(self) -> list[str]:
        if self.binding.command_prefix:
            return [*map(str, self.binding.command_prefix), self.binding.profile]
        root = self.binding.resolved_root()
        return [str(self.binding.julia), "--startup-file=no", "--history-file=no", "--threads=1", "--color=no",
                f"--project={root}", str(root / "bin" / "worker.jl"), self.binding.profile]

    def _environment(self) -> dict:
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("JULIA_") and key not in {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"}}
        environment.update(JULIA_LOAD_PATH="@" + os.pathsep + "@stdlib", JULIA_PKG_OFFLINE="true",
                           JULIA_PKG_PRECOMPILE_AUTO="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        if self.binding.depot is not None:
            environment["JULIA_DEPOT_PATH"] = str(self.binding.depot)
        return environment

    def start(self) -> dict:
        if self.alive:
            return self.identity
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=self._environment(), bufsize=0)
        except OSError as exc:
            raise WorkerError("runtime_unavailable", f"Cannot start the Julia worker: {exc}") from exc
        self.session_id = "julia-session-" + uuid.uuid4().hex
        self.occurrence = 0
        self._frames = queue.Queue()
        self._stderr = deque(maxlen=STDERR_LINES)
        threading.Thread(target=self._read_frames, args=(self.process, self._frames), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.process, self._stderr), daemon=True).start()
        hello = self._receive(self.binding.startup_timeout_s, "handshake")
        try:
            self._accept_handshake(codec.decode(hello))
        except (codec.CodecError, KeyError, TypeError, ValueError) as exc:
            self._terminate("handshake_rejected")
            raise WorkerError("runtime_mismatch", f"Julia worker handshake refused: {exc}") from exc
        self.startup_s = time.perf_counter() - started
        self.sessions.append({"session_id": self.session_id, "runtime_digest": self.runtime_digest,
                              "startup_s": self.startup_s, "ended": None})
        return self.identity

    def _accept_handshake(self, hello: dict) -> None:
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            raise ValueError("first frame is not a hello")
        identity, context, digest = hello["identity"], hello["context"], hello["runtime_digest"]
        if hashlib.sha256(codec.encode(identity)).hexdigest() != digest:
            raise ValueError("runtime digest does not match the reported identity")
        if self.verify_pins:
            root = self.binding.resolved_root()
            expected = expected_identity(self.binding.profile, root)
            for key, value in expected.items():
                if identity.get(key) != value:
                    raise ValueError(f"{key}: worker reports {identity.get(key)!r}, host requires {value!r}")
            executable = file_sha256(Path(self.binding.julia))
            if identity.get("julia_executable_sha256") != executable:
                raise ValueError("the worker's Julia executable differs from the bound executable")
            platform = f"{identity['kernel']}-{identity['arch']}"
            pins = pinned_runtime()["platforms"].get(platform)
            if pins is None:
                raise ValueError(f"no verified Julia runtime pin for platform {platform}")
            for key in ("julia_executable_sha256", "sysimage_sha256"):
                if identity.get(key) != pins[key]:
                    raise ValueError(f"{key} differs from the pinned {platform} runtime")
        self.identity, self.context, self.runtime_digest = identity, context, digest

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.runtime_digest is not None

    def _terminate(self, reason: str) -> None:
        process, self.process = self.process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream and stream.close()
                except OSError:
                    pass
        if self.sessions and self.sessions[-1]["session_id"] == self.session_id and self.sessions[-1]["ended"] is None:
            self.sessions[-1]["ended"] = reason
        self.runtime_digest = None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None and self.process.stdin:
            try:
                self._send(codec.encode({"type": "shutdown"}))
                self.process.wait(timeout=30)
            except (OSError, subprocess.SubprocessError, WorkerError):
                pass
        self._terminate("closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- framing -------------------------------------------------------------------
    @staticmethod
    def _read_frames(process, frames):
        stream = process.stdout
        try:
            while True:
                header = _read_exact(stream, 8)
                if not header:
                    frames.put(("eof", None))
                    return
                if len(header) != 8 or header[:4] != FRAME_MAGIC:
                    frames.put(("protocol", "malformed frame header"))
                    return
                size = struct.unpack("<I", header[4:])[0]
                if size > MAX_FRAME:
                    frames.put(("protocol", "frame exceeds its bound"))
                    return
                payload = _read_exact(stream, size)
                if len(payload) != size:
                    frames.put(("protocol", "truncated frame payload"))
                    return
                frames.put(("frame", payload))
        except (OSError, ValueError) as exc:
            frames.put(("protocol", f"stdout read failed: {exc}"))

    @staticmethod
    def _read_stderr(process, lines):
        try:
            for line in iter(process.stderr.readline, b""):
                lines.append(line.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def stderr_tail(self, count: int = 40) -> str:
        return "\n".join(list(self._stderr)[-count:])

    def _send(self, payload: bytes) -> None:
        if len(payload) > MAX_FRAME:
            raise WorkerError("oversized_request", "Request frame exceeds its bound; nothing was sent")
        try:
            self.process.stdin.write(FRAME_MAGIC + struct.pack("<I", len(payload)) + payload)
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self._terminate("write_failed")
            raise WorkerError("worker_exited", f"Julia worker input closed: {exc}") from exc

    def _receive(self, timeout: float, what: str) -> bytes:
        try:
            kind, payload = self._frames.get(timeout=timeout)
        except queue.Empty:
            self._terminate("timeout")
            raise WorkerError("worker_timeout", f"Julia worker {what} exceeded {timeout} s; session ended") from None
        if kind == "frame":
            return payload
        tail = self.stderr_tail()
        self._terminate(kind)
        if kind == "eof":
            raise WorkerError("worker_exited", f"Julia worker exited during {what}\n{tail}")
        raise WorkerError("worker_protocol", f"Julia worker {what}: {payload}\n{tail}")

    # -- execution -----------------------------------------------------------------
    def execute(self, operation: str, configuration: bytes, input_payload: bytes, *,
                timeout: float = REQUEST_TIMEOUT_S) -> WorkerResult:
        if OPERATION_PROFILE.get(operation) != self.binding.profile:
            raise ExecutionRefused("unsupported_operation", f"{operation} is not served by the {self.binding.profile} profile")
        if not self.alive:
            self.start()
        specification = Specification(program_bytes(operation, self.runtime_digest), configuration, input_payload)
        expected_spec = specification.identity()
        self.occurrence += 1
        occurrence, session = self.occurrence, self.session_id
        request = codec.encode({"type": "execute", "occurrence": occurrence, "operation": operation,
                                "program": specification.program, "configuration": configuration,
                                "input": input_payload, "specification_identity": expected_spec})
        started = time.perf_counter()
        self._send(request)
        raw = self._receive(timeout, f"occurrence {occurrence}")
        wall = time.perf_counter() - started
        try:
            response = codec.decode(raw)
            return self._check(operation, specification, expected_spec, response, occurrence, session, wall)
        except ExecutionRefused:
            raise
        except (codec.CodecError, KeyError, TypeError, ValueError) as exc:
            self._terminate("malformed_response")
            raise WorkerError("worker_protocol", f"Malformed worker response: {exc}") from exc

    def _check(self, operation, specification, expected_spec, response, occurrence, session, wall) -> WorkerResult:
        if response.get("type") != "result" or response.get("occurrence") != occurrence:
            raise ValueError("response does not answer the active occurrence")
        if response["specification_identity"] != expected_spec:
            raise ValueError("worker answered a different specification")
        status = response["status"]
        if status == "unrunnable":
            raise ExecutionRefused("execution_refused", str(response.get("detail")))
        for key, expected in (("program_identity", specification.program_identity()),
                              ("input_identity", specification.input_identity())):
            if response[key] != expected:
                raise ValueError(f"worker echoed {key} {response[key]!r}; recomputed {expected}")
        exit_code = response["exit_code"]
        if type(exit_code) is not int or not 0 <= exit_code < 2**32:
            raise ValueError("invalid exit code")
        common = dict(operation=operation, specification=specification, specification_identity=expected_spec,
                      program_identity=specification.program_identity(), input_identity=specification.input_identity(),
                      exit_code=exit_code, detail=response.get("detail"), session_id=session, occurrence=occurrence,
                      elapsed_ns=int(response.get("elapsed_ns", 0)), runtime_digest=self.runtime_digest, wall_time_s=wall)
        if status == "halted":
            if any(response[key] is not None for key in ("output", "output_identity", "computation_identity")) or exit_code == 0:
                raise ValueError("a halted run has no output, output identity or computation identity")
            return WorkerResult(status="halted", output=None, output_identity=None, computation_identity=None, **common)
        if status != "completed" or exit_code != 0:
            raise ValueError(f"unknown status {status!r}")
        output = response["output"]
        if not isinstance(output, bytes):
            raise ValueError("a completed run must return output bytes")
        out_id = output_identity(output)
        if response["output_identity"] != out_id:
            raise ValueError("worker output identity does not match the returned bytes")
        computation = computation_identity(common["program_identity"], common["input_identity"], out_id, exit_code)
        if response["computation_identity"] != computation:
            raise ValueError("worker computation identity does not match its recomputation")
        return WorkerResult(status="completed", output=output, output_identity=out_id,
                            computation_identity=computation, **common)

    def runtime_record(self) -> dict:
        return {"protocol": PROTOCOL, "profile": self.binding.profile, "runtime_digest": self.runtime_digest,
                "identity": self.identity, "context": self.context,
                "host_checked": {"julia_executable_sha256": file_sha256(Path(self.binding.julia))
                                 if not self.binding.command_prefix else None,
                                 "pins_verified": self.verify_pins},
                "startup_s": self.startup_s}


def _read_exact(stream, count: int) -> bytes:
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
