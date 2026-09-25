"""Host side of the CIW Julia worker (protocol ciw.julia.worker-protocol.v1; docs/JULIA_SP1.md).

Scope: the pinned Julia runtime (``julia/julia-runtime.json``: Julia 1.10.12 LTS,
the official archive and its checksum) and worker environment (``julia/Project.toml``
and the machine-generated ``julia/Manifest.toml``), the versioned binary encodings
of the one allowlisted operation (``ciw.julia.damped-oscillator.v1``: its
configuration, input and output), framed messages with request identifiers and
size limits, and a host-owned worker session: started with an explicit project,
depot, disabled startup file and declared thread count; accepted only when its
handshake matches the expected runtime identity (Julia version, platform,
executable and worker source digests, project and manifest digests, every loaded
package's version and source tree, threads, load path and numerical settings);
one request in flight; a session identity and a monotonically increasing
occurrence number per request. A timeout, unexpected end of stream, malformed or
oversized response, mismatched request identifier or worker exit ends the
session: the process is killed and reaped and the occurrence fails. Nothing is
retried and no result is ever made up for a failed, refused or halted request.

The module imports the standard library only, so SCR's interpreter can host it
behind ``execution.dispatcher.SpecificationDispatcher`` (see
``ciw.lab.implementation_targets_julia``). The fault-injection arguments of
:meth:`WorkerSession.request` exist for the acceptance set's failure fixtures;
they break the channel and never produce a result.

Non-claims: the handshake is a declaration by the process that sent it (a
process replaying a recorded handshake passes it), so it identifies the
environment the worker reports, not an attested one; digests here are unkeyed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid

WORKER_DIR = Path(__file__).resolve().parent / "julia"
WORKER_SOURCE = WORKER_DIR / "oscillator_worker.jl"
RUNTIME_PIN = json.loads((WORKER_DIR / "julia-runtime.json").read_text(encoding="utf-8"))

PROTOCOL = "ciw.julia.worker-protocol.v1"
HANDSHAKE_SCHEMA = "ciw.julia.worker-handshake.v1"
REFUSAL_SCHEMA = "ciw.julia.worker-refusal.v1"
HALT_SCHEMA = "ciw.julia.worker-halt.v1"
OPERATION = "ciw.julia.damped-oscillator.v1"
INPUT_SCHEMA = "ciw.julia.oscillator-input.v1"
CONFIGURATION_SCHEMA = "ciw.julia.tsit5-configuration.v1"
OUTPUT_SCHEMA = "ciw.julia.oscillator-output.v1"
# The program bytes of every request, byte for byte the worker's DESCRIPTOR (its sha256 is in the handshake).
OPERATION_DESCRIPTOR = (
    "ciw.julia.damped-oscillator.v1\n"
    "state [q, v] in m and m/s, binary64; q' = v; v' = -2*gamma*v - omega_0^2*q; E = 0.5*mass*(v^2 + omega_0^2*q^2) in J\n"
    "input: ciw.julia.oscillator-input.v1; configuration: ciw.julia.tsit5-configuration.v1; output: ciw.julia.oscillator-output.v1\n"
    "solver: OrdinaryDiffEqTsit5.Tsit5() on ODEProblem{true, SciMLBase.AutoSpecialize} over [0, t_last]; adaptive; "
    "PIController with the configured gains and limits; internalnorm ODE_DEFAULT_NORM; values at the requested "
    "times from the Tsit5 free interpolant (saveat, save_everystep=false, save_start=true, save_end=true, dense=false)\n"
    "completed only on ReturnCode.Success with every requested time saved exactly and finite; otherwise halted without output"
).encode("ascii")
# The v1 controller profile: OrdinaryDiffEqCore's Tsit5 defaults made explicit, in wire order. The worker refuses others.
CONTROLLER = (("qmin", 0.2), ("qmax", 10.0), ("qmax_first_step", 10000.0), ("gamma", 0.9), ("qsteady_min", 1.0),
              ("qsteady_max", 1.0), ("beta1", 0.14), ("beta2", 0.08), ("qoldinit", 1.0e-4), ("failfactor", 2.0))
# Proposed software limits of docs/JULIA_SP1.md (not physical validity limits).
LIMITS = {"omega_0": (0.0, 20.0), "mass": (0.0, 100.0), "q0": (-10.0, 10.0), "v0": (-100.0, 100.0),
          "duration": (0.0, 12.0), "samples": (2, 4096), "tolerance": (1e-14, 1e-2), "step": (0.0, 12.0),
          "maxiters": (1, 10_000_000)}

MAGIC = b"CIWJ"
VERSION = 1
KINDS = {"handshake": 1, "request": 2, "completed": 3, "refused": 4, "halted": 5, "shutdown": 6}
KIND_NAMES = {code: name for name, code in KINDS.items()}
HEADER = struct.Struct("<4sBBHQI")  # magic, version, kind, reserved, request id, payload length
MAX_REQUEST_PAYLOAD = 65536
MAX_RESPONSE_PAYLOAD = 262144
MAX_HANDSHAKE_PAYLOAD = 65536
# Every handshake field, exactly: compared with the expected identity, checked against the bound files or pin
# (machine, sysimage, project, depot, packages) or recorded only (julia_commit, opt_level, cpu_target, cpu_name).
HANDSHAKE_FIELDS = ("protocol", "operations", "operation_descriptor_sha256", "julia_version", "julia_commit",
                    "machine", "word_size", "executable_sha256", "sysimage", "worker_source_sha256", "project",
                    "project_sha256", "manifest_sha256", "local_preferences", "depot", "depot_count", "load_path",
                    "packages", "threads", "interactive_threads", "rounding", "zero_subnormals", "opt_level",
                    "check_bounds", "fast_math", "cpu_target", "cpu_name", "controller_profile")


class RequestRefusal(ValueError):
    """A request the host refuses to encode or send: nothing is dispatched and no occurrence is used."""

    def __init__(self, code: str, field: str = ""):
        super().__init__(f"{code}: {field}" if field else code)
        self.code, self.field = code, field


class WorkerFailure(RuntimeError):
    """The worker session ended: the active occurrence (if any) failed without a result."""

    def __init__(self, code: str, detail: str = "", occurrence: int | None = None):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail, self.occurrence = code, detail, occurrence
        self.request_frame: bytes | None = None   # the bytes sent for the failed occurrence
        self.response: bytes | None = None        # a complete frame received before the session ended, if any


# ------------------------------------------------------------------ encodings

def _tag(schema: str) -> bytes:
    raw = schema.encode("ascii")
    return struct.pack("<H", len(raw)) + raw


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RequestRefusal("not_a_number", name)
    value = float(value)
    if not math.isfinite(value):
        raise RequestRefusal("nonfinite_number", name)
    return value


def _within(value, low, high, name, low_open=False) -> float:
    value = _number(value, name)
    if not ((value > low if low_open else value >= low) and value <= high):
        raise RequestRefusal("out_of_bounds", name)
    return value


def encode_configuration(abstol: float, reltol: float, dt: float, dtmax: float, maxiters: int) -> bytes:
    """``ciw.julia.tsit5-configuration.v1``: tolerances, initial and maximum step, iteration limit, controller profile."""
    low, high = LIMITS["tolerance"]
    values = [_within(abstol, low, high, "abstol"), _within(reltol, low, high, "reltol"),
              _within(dt, *LIMITS["step"], "dt", low_open=True), _within(dtmax, *LIMITS["step"], "dtmax", low_open=True)]
    if isinstance(maxiters, bool) or not isinstance(maxiters, int):
        raise RequestRefusal("not_an_integer", "maxiters")
    if not LIMITS["maxiters"][0] <= maxiters <= LIMITS["maxiters"][1]:
        raise RequestRefusal("out_of_bounds", "maxiters")
    return (_tag(CONFIGURATION_SCHEMA) + struct.pack("<4d", *values) + struct.pack("<Q", maxiters)
            + struct.pack(f"<{len(CONTROLLER)}d", *(value for _, value in CONTROLLER)))


def encode_input(omega_0, gamma, mass, q0, v0, duration, times) -> bytes:
    """``ciw.julia.oscillator-input.v1``: parameters, initial state [q0, v0] at t = 0 and the requested times.

    The grid starts at the initial time 0, is strictly increasing and lies in
    the half-open interval [0, duration). Booleans, nonfinite and out-of-bound
    numbers and invalid grids are refused here, before any dispatch.
    """
    omega_0 = _within(omega_0, *LIMITS["omega_0"], "omega_0", low_open=True)
    gamma = _within(gamma, 0.0, 0.5 * omega_0, "gamma")
    mass = _within(mass, *LIMITS["mass"], "mass", low_open=True)
    q0, v0 = _within(q0, *LIMITS["q0"], "q0"), _within(v0, *LIMITS["v0"], "v0")
    duration = _within(duration, *LIMITS["duration"], "duration", low_open=True)
    times = [_number(value, "time") for value in times]
    if not LIMITS["samples"][0] <= len(times) <= LIMITS["samples"][1]:
        raise RequestRefusal("sample_count_out_of_bounds", "times")
    if times[0] != 0.0:
        raise RequestRefusal("grid_origin", "times")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise RequestRefusal("grid_not_increasing", "times")
    if times[-1] >= duration:
        raise RequestRefusal("grid_outside_interval", "times")
    return (_tag(INPUT_SCHEMA) + struct.pack("<6d", omega_0, gamma, mass, q0, v0, duration)
            + struct.pack("<I", len(times)) + struct.pack(f"<{len(times)}d", *times))


def encode_request(program: bytes, configuration: bytes, input_payload: bytes) -> bytes:
    """The request payload: SCR's three specification fields, each as u64 LE length then bytes."""
    return b"".join(struct.pack("<Q", len(field)) + field for field in (program, configuration, input_payload))


def frame(kind: str, request_id: int, payload: bytes, limit: int = MAX_REQUEST_PAYLOAD) -> bytes:
    if len(payload) > limit:
        raise RequestRefusal("frame_too_large", f"{len(payload)} bytes over the {limit}-byte limit")
    return HEADER.pack(MAGIC, VERSION, KINDS[kind], 0, request_id, len(payload)) + payload


def decode_output(payload: bytes) -> dict:
    """Decode ``ciw.julia.oscillator-output.v1`` strictly (exact length, schema tag, finite values)."""
    def take(count):
        nonlocal at
        if at + count > len(payload):
            raise ValueError("Julia output is truncated")
        chunk = payload[at:at + count]
        at += count
        return chunk

    at = 0
    (size,) = struct.unpack("<H", take(2))
    if take(size) != OUTPUT_SCHEMA.encode("ascii"):
        raise ValueError(f"Julia output is not {OUTPUT_SCHEMA}")
    (size,) = struct.unpack("<H", take(2))
    retcode = take(size).decode("ascii")
    naccept, nreject, nf, count = struct.unpack("<QQQI", take(28))
    series = [list(struct.unpack(f"<{count}d", take(8 * count))) for _ in range(4)]
    if at != len(payload):
        raise ValueError("Julia output has trailing bytes")
    if not all(math.isfinite(value) for values in series for value in values):
        raise ValueError("Julia output holds a nonfinite value")
    return {"retcode": retcode, "naccept": naccept, "nreject": nreject, "nf": nf, "count": count,
            "t": series[0], "q": series[1], "v": series[2], "energy": series[3]}


def decode_lines(payload: bytes, schema: str) -> dict:
    """A refusal, halt or handshake payload: its schema line, then unique ``key value`` lines."""
    lines = payload.decode("utf-8").split("\n")
    if lines[0] != schema or lines[-1] != "":
        raise ValueError(f"Not a {schema} payload")
    fields = {}
    for line in lines[1:-1]:
        key, separator, value = line.partition(" ")
        if not separator or not key or key in fields:
            raise ValueError(f"Malformed {schema} line")
        fields[key] = value
    return fields


# ---------------------------------------------------------------- identities

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_CRC32C = []
for _byte in range(256):
    _crc = _byte
    for _ in range(8):
        _crc = (_crc >> 1) ^ (0x82F63B78 if _crc & 1 else 0)
    _CRC32C.append(_crc)


def crc32c(data: bytes, crc: int = 0) -> int:
    crc ^= 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def version_slug(package_uuid: str, tree_sha1: str, length: int = 5) -> str:
    """Julia's ``Base.version_slug``: the package directory name Pkg installs a (uuid, git tree) pair under."""
    value = crc32c(bytes.fromhex(tree_sha1), crc32c(uuid.UUID(package_uuid).int.to_bytes(16, "little")))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    slug = ""
    for _ in range(length):
        value, digit = divmod(value, len(alphabet))
        slug += alphabet[digit]
    return slug


def manifest_packages(manifest: Path) -> dict:
    """Non-standard-library packages of a Julia manifest: name -> uuid, version, git tree and extension names."""
    data = tomllib.loads(Path(manifest).read_text(encoding="utf-8"))
    packages = {}
    for name, entries in data.get("deps", {}).items():
        for entry in entries:
            if "git-tree-sha1" in entry:
                packages[name] = {"uuid": entry["uuid"], "version": entry["version"],
                                  "tree": entry["git-tree-sha1"], "extensions": sorted(entry.get("extensions", {}))}
    return {"julia_version": data.get("julia_version"), "packages": packages}


@dataclass(frozen=True)
class JuliaRuntime:
    """An explicit host binding: the Julia executable and the depot the worker environment was instantiated in."""

    executable: Path
    depot: Path
    project: Path = WORKER_DIR
    worker: Path = WORKER_SOURCE


def platform_archive(machine: str) -> dict | None:
    return next((dict(entry, platform=name) for name, entry in RUNTIME_PIN["archives"].items()
                 if entry["machine"] == machine), None)


def expected_identity(runtime: JuliaRuntime, threads: int = 1) -> dict:
    """The handshake values the host requires, computed from the bound files (never read from a saved workspace)."""
    return {"protocol": PROTOCOL, "operations": OPERATION,
            "operation_descriptor_sha256": hashlib.sha256(OPERATION_DESCRIPTOR).hexdigest(),
            "julia_version": RUNTIME_PIN["version"], "word_size": "64",
            "executable_sha256": sha256_file(Path(runtime.executable).resolve()),
            "worker_source_sha256": sha256_file(runtime.worker),
            "project_sha256": sha256_file(Path(runtime.project) / "Project.toml"),
            "manifest_sha256": sha256_file(Path(runtime.project) / "Manifest.toml"),
            "local_preferences": "false", "depot_count": "1", "load_path": "@|@stdlib",
            "threads": str(threads), "interactive_threads": "0", "rounding": "RoundingMode{:Nearest}()",
            "zero_subnormals": "false", "check_bounds": "0", "fast_math": "0",
            "controller_profile": ",".join(f"{name}={_julia_float(value)}" for name, value in CONTROLLER)}


def _julia_float(value: float) -> str:
    """Julia's shortest ``string(::Float64)`` for the controller constants (1.0e-4, 10000.0, 0.2)."""
    text = repr(float(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        mantissa = mantissa if "." in mantissa else mantissa + ".0"
        return f"{mantissa}e{int(exponent)}"
    return text


def compare_handshake(handshake: dict, runtime: JuliaRuntime, threads: int = 1) -> dict:
    """Field-by-field comparison with the expected identity; ``mismatches`` lists field names only."""
    expected = expected_identity(runtime, threads)
    mismatches = sorted(name for name, value in expected.items() if handshake.get(name) != value)
    if set(handshake) != set(HANDSHAKE_FIELDS):
        mismatches.append("fields")
    archive = platform_archive(handshake.get("machine", ""))
    if archive is None:
        mismatches.append("machine")
    bindir = Path(runtime.executable).resolve().parent
    sysimage = handshake.get("sysimage", "")
    if archive is None or Path(sysimage) != Path(archive["sysimage"]):
        mismatches.append("sysimage")
    for name, bound in (("project", Path(runtime.project) / "Project.toml"), ("depot", Path(runtime.depot))):
        try:
            same = os.path.samefile(handshake.get(name, ""), bound)
        except OSError:
            same = False
        if not same:
            mismatches.append(name)
    manifest = manifest_packages(Path(runtime.project) / "Manifest.toml")
    loaded, problems = parse_packages(handshake.get("packages", ""), manifest["packages"])
    if problems:
        mismatches.append("packages")
    direct = tomllib.loads((Path(runtime.project) / "Project.toml").read_text(encoding="utf-8"))["deps"]
    missing = sorted(name for name in direct if name in manifest["packages"] and name not in loaded)
    if missing:
        mismatches.append("packages_loaded")
    return {"accepted": not mismatches, "mismatches": sorted(set(mismatches)), "compared": sorted(expected),
            "package_problems": problems, "packages_verified": len(loaded),
            "sysimage_sha256": _cached_sha256(bindir / sysimage) if archive is not None and not mismatches else None}


_DIGESTS: dict = {}


def _cached_sha256(path: Path) -> str:
    """The digest of a large runtime file (the system image), computed once per file state in this process."""
    status = path.stat()
    key = (str(path), status.st_size, status.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = sha256_file(path)
    return _DIGESTS[key]


def parse_packages(text: str, manifest: dict) -> tuple:
    """Each loaded package ``name=uuid=version=<package>/<slug>/<file>`` against the manifest's uuid, version and tree."""
    loaded, problems = {}, []
    for item in [part for part in text.split(";") if part]:
        try:
            name, package_uuid, version, where = item.split("=")
            package, slug = where.split("/")[:2]
        except ValueError:
            problems.append(f"malformed:{item[:60]}")
            continue
        entry = manifest.get(package)
        if entry is None:
            problems.append(f"not_in_manifest:{name}")
            continue
        if slug != version_slug(entry["uuid"], entry["tree"]):
            problems.append(f"tree:{name}")
        if version != entry["version"]:
            problems.append(f"version:{name}")
        if name == package:
            if package_uuid != entry["uuid"]:
                problems.append(f"uuid:{name}")
            loaded[name] = version
        elif name not in entry["extensions"]:
            problems.append(f"not_an_extension:{name}")
    return loaded, problems


# ------------------------------------------------------------------ sessions

@dataclass
class Occurrence:
    """One request's exchange: its session and occurrence number, exact frames and outcome."""

    session_id: str
    occurrence: int
    request_id: int
    request_frame: bytes
    response_frame: bytes
    kind: str
    payload: bytes
    elapsed_s: float


class WorkerSession:
    """One worker process. ``start`` accepts or refuses it on its handshake; ``request`` serves one occurrence.

    ``command`` and ``environment`` start the process; ``accept`` maps the
    parsed handshake to a comparison whose ``accepted`` decides. A failure of
    the channel ends the session for good; a new session must be started.
    """

    def __init__(self, command, environment=None, accept=None, *, startup_timeout: float = 180.0,
                 request_timeout: float = 120.0, cwd=None):
        self.command, self.environment, self.accept = [str(part) for part in command], environment, accept
        self.startup_timeout, self.request_timeout, self.cwd = startup_timeout, request_timeout, cwd
        self.session_id = uuid.uuid4().hex  # a fresh identity per session, never a content identity
        self.occurrences = 0
        self.process = None
        self.handshake_payload = None
        self.handshake = None
        self.comparison = None
        self.ended = None                   # the failure code that ended the session, or "closed"
        self.startup_s = None
        self._chunks: queue.Queue = queue.Queue()
        self._buffer = bytearray()
        self._stderr = bytearray()
        self._stderr_bytes = 0

    @classmethod
    def for_runtime(cls, runtime: JuliaRuntime, threads: int = 1, expected_threads: int | None = None,
                    **options) -> "WorkerSession":
        """The pinned worker under ``runtime``, started with ``threads`` and accepted only if its handshake matches
        the expected identity for ``expected_threads`` (default ``threads``; another value is a wrong environment)."""
        expected = threads if expected_threads is None else expected_threads
        environment = {key: value for key, value in os.environ.items() if not key.startswith("JULIA_")}
        environment.update({"JULIA_DEPOT_PATH": str(runtime.depot), "JULIA_LOAD_PATH": os.pathsep.join(["@", "@stdlib"]),
                            "JULIA_PKG_OFFLINE": "true"})
        command = [runtime.executable, f"--project={runtime.project}", "--startup-file=no", "--history-file=no",
                   f"--threads={threads}", "--color=no", runtime.worker]
        return cls(command, environment, lambda handshake: compare_handshake(handshake, runtime, expected), **options)

    def start(self) -> dict:
        began = time.perf_counter()
        directory = self.cwd or tempfile.gettempdir()
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=self.environment, cwd=directory)
        except OSError as exc:
            self.ended = "start_failed"
            raise WorkerFailure("start_failed", type(exc).__name__) from exc
        threading.Thread(target=self._pump, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self._drain, args=(self.process.stderr,), daemon=True).start()
        header, payload = self._read_frame(time.monotonic() + self.startup_timeout, MAX_HANDSHAKE_PAYLOAD, None)
        if header[2] != KINDS["handshake"] or header[4] != 0:
            self._end("malformed_handshake", "first frame is not a handshake")
        try:
            self.handshake = decode_lines(payload, HANDSHAKE_SCHEMA)
        except (UnicodeDecodeError, ValueError):
            self._end("malformed_handshake", "handshake payload")
        self.handshake_payload = payload
        self.startup_s = time.perf_counter() - began
        self.comparison = self.accept(self.handshake) if self.accept else {"accepted": True, "mismatches": []}
        if not self.comparison["accepted"]:
            self._end("environment_mismatch", ",".join(self.comparison["mismatches"]))
        return self.handshake

    def request(self, program: bytes, configuration: bytes, input_payload: bytes, timeout: float | None = None,
                fault: str | None = None) -> Occurrence:
        """Serve one request; returns its completed, refused or halted occurrence or raises WorkerFailure.

        ``fault`` injects a channel failure for the acceptance fixtures:
        ``oversized_header`` declares a payload over the worker's limit,
        ``stall`` sends half the payload and waits, ``crash`` kills the worker
        once the request is sent. Each ends the session without a result.
        """
        if self.ended is not None or self.process is None:
            raise WorkerFailure("session_ended", self.ended or "not started")
        payload = encode_request(program, configuration, input_payload)
        occurrence = self.occurrences + 1
        data = frame("request", occurrence, payload)   # an oversized request is refused here and uses no occurrence
        self.occurrences = occurrence
        if fault == "oversized_header":
            data = HEADER.pack(MAGIC, VERSION, KINDS["request"], 0, occurrence, MAX_REQUEST_PAYLOAD + 1)
        elif fault == "stall":
            data = data[:HEADER.size + len(payload) // 2]
        elif fault not in (None, "crash"):
            raise ValueError(f"Unknown fault injection: {fault}")
        try:
            return self._exchange(occurrence, data, timeout, fault)
        except WorkerFailure as failure:
            failure.request_frame = data
            raise

    def _exchange(self, occurrence: int, data: bytes, timeout, fault) -> Occurrence:
        began = time.perf_counter()
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except OSError:
            self._end("worker_exited", "request could not be written", occurrence)
        if fault == "crash":
            self.process.kill()
        deadline = time.monotonic() + (self.request_timeout if timeout is None else timeout)
        header, response = self._read_frame(deadline, MAX_RESPONSE_PAYLOAD, occurrence)
        elapsed = time.perf_counter() - began
        kind = KIND_NAMES.get(header[2])
        if header[4] != occurrence:
            self._end("request_id_mismatch", f"response names request {header[4]}", occurrence)
        if kind not in ("completed", "refused", "halted"):
            self._end("malformed_response", f"response kind {header[2]}", occurrence)
        received = HEADER.pack(*header) + response
        if fault == "oversized_header":
            # The worker refuses an oversized frame without reading it, then exits: the session is over either way.
            self._end("frame_too_large", _refusal_code(response), occurrence, response=received)
        return Occurrence(self.session_id, occurrence, occurrence, data, received, kind, response, elapsed)

    def close(self) -> None:
        if self.process is not None and self.ended is None:
            try:
                self.process.stdin.write(frame("shutdown", 0, b""))
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
            self.ended = "closed"

    def diagnostics(self) -> dict:
        """Stderr volume only: compilation and log messages are diagnostics, never protocol data or evidence."""
        return {"stderr_bytes": self._stderr_bytes, "exit_code": None if self.process is None else self.process.poll()}

    # ------------------------------------------------------------ internals
    def _pump(self, stream) -> None:
        try:
            while True:
                chunk = stream.read1(65536)
                if not chunk:
                    break
                self._chunks.put(chunk)
        except (OSError, ValueError):
            pass
        self._chunks.put(b"")

    def _drain(self, stream) -> None:
        try:
            for chunk in iter(lambda: stream.read1(65536), b""):
                self._stderr_bytes += len(chunk)
                self._stderr.extend(chunk)
                del self._stderr[:-65536]
        except (OSError, ValueError):
            pass

    def _read_exact(self, count: int, deadline: float, occurrence, within_frame: bool) -> bytes:
        while len(self._buffer) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._end("response_timeout", "no complete frame before the deadline", occurrence)
            try:
                chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                continue
            if not chunk:
                self._chunks.put(b"")
                # End of stream inside a frame is a truncated frame; between frames, the worker exited.
                code = "unexpected_eof" if within_frame or self._buffer else "worker_exited"
                self._end(code, f"stream ended after {len(self._buffer)} of {count} bytes", occurrence)
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:count])
        del self._buffer[:count]
        return data

    def _read_frame(self, deadline: float, limit: int, occurrence) -> tuple:
        header = HEADER.unpack(self._read_exact(HEADER.size, deadline, occurrence, False))
        if header[0] != MAGIC or header[1] != VERSION or header[3] != 0:
            self._end("malformed_response", "frame header", occurrence)
        if header[5] > limit:
            self._end("response_too_large", f"{header[5]} bytes over the {limit}-byte limit", occurrence)
        return header, self._read_exact(header[5], deadline, occurrence, True)

    def _end(self, code: str, detail: str, occurrence=None, response: bytes | None = None):
        """End the session: kill and reap the worker, fail the occurrence. Never returns."""
        self.ended = code
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            for stream in (self.process.stdin,):
                try:
                    stream.close()
                except OSError:
                    pass
        failure = WorkerFailure(code, detail, occurrence)
        failure.response = response
        raise failure


def _refusal_code(payload: bytes) -> str:
    try:
        return decode_lines(payload, REFUSAL_SCHEMA).get("code", "")
    except (UnicodeDecodeError, ValueError):
        return "unparsed_refusal"


# ---------------------------------------------------------------- plan runner

def direct_dispatch(session: WorkerSession, step: dict, program: bytes, configuration: bytes,
                    input_payload: bytes) -> dict:
    """One request straight to the worker (no SCR): the occurrence's frames and outcome, or the failure that ended it."""
    try:
        occurrence = session.request(program, configuration, input_payload, step.get("timeout"), step.get("fault"))
    except WorkerFailure as failure:
        return failure_record(failure)
    return occurrence_record(occurrence)


def occurrence_record(occurrence: Occurrence) -> dict:
    return {"outcome": occurrence.kind, "occurrence": occurrence.occurrence, "request_id": occurrence.request_id,
            "request_frame": occurrence.request_frame.hex(), "response_frame": occurrence.response_frame.hex(),
            "elapsed_s": occurrence.elapsed_s}


def failure_record(failure: WorkerFailure) -> dict:
    return {"outcome": failure.code, "failure_detail": failure.detail, "occurrence": failure.occurrence,
            "request_frame": None if failure.request_frame is None else failure.request_frame.hex(),
            "response_frame": None if failure.response is None else failure.response.hex(), "elapsed_s": None}


def run_plan(runtime: JuliaRuntime | None, steps, dispatch=direct_dispatch) -> dict:
    """Execute acceptance steps in order on named worker sessions and record every exchange; nothing is retried.

    Steps: ``start`` (a session under ``runtime`` with ``threads`` and
    ``expected_threads``, or a protocol ``mock`` replaying the handshake of the
    named ``replay`` session), ``request`` (hex ``program``, ``configuration``
    and ``input``, optional ``timeout`` and ``fault``) through ``dispatch`` and
    ``close``. A request on a session that has ended fails as ``session_ended``.
    """
    sessions, records = {}, []
    try:
        for step in steps:
            record = {"label": step["label"], "op": step["op"], "session": step["session"]}
            if step["op"] == "start":
                if step.get("mock"):
                    # Without a recorded handshake to replay the mock's own start fails (malformed_handshake).
                    source = sessions[step["replay"]]
                    session = mock_session(step["mock"], source.handshake_payload or b"", source.accept,
                                           request_timeout=step.get("timeout", 30.0))
                else:
                    session = WorkerSession.for_runtime(runtime, step.get("threads", 1), step.get("expected_threads"))
                sessions[step["session"]] = session
                try:
                    session.start()
                    record["outcome"] = "accepted"
                except WorkerFailure as failure:
                    record.update(outcome=failure.code, failure_detail=failure.detail)
                record.update(session_id=session.session_id, startup_s=session.startup_s, comparison=session.comparison,
                              handshake=None if session.handshake_payload is None else session.handshake_payload.hex())
            elif step["op"] == "request":
                session = sessions[step["session"]]
                record.update(session_id=session.session_id, fixture=step.get("fixture"), fault=step.get("fault"))
                record.update(dispatch(session, step, bytes.fromhex(step["program"]),
                                       bytes.fromhex(step["configuration"]), bytes.fromhex(step["input"])))
            elif step["op"] == "close":
                sessions[step["session"]].close()
                record["outcome"] = "closed"
            else:
                raise ValueError(f"Unknown plan step: {step['op']}")
            record["session_ended"] = sessions[step["session"]].ended
            records.append(record)
    finally:
        for session in sessions.values():
            if session.ended is None:
                session.close()
    return {"records": records,
            "sessions": {name: {"session_id": session.session_id, "occurrences": session.occurrences,
                                "ended": session.ended, **session.diagnostics()} for name, session in sessions.items()}}


# --------------------------------------------------------------- protocol mock

# A stand-in worker for the channel-failure fixtures (never for numerical gates): it replays the handshake it is
# given, reads one request frame and answers it wrongly as its mode says.
MOCK_WORKER = r'''
import struct, sys, time
mode, handshake = sys.argv[1], bytes.fromhex(sys.argv[2])
header = struct.Struct("<4sBBHQI")
out = sys.stdout.buffer
out.write(header.pack(b"CIWJ", 1, 1, 0, 0, len(handshake)) + handshake)
out.flush()
if mode == "exit_after_handshake":
    sys.exit(3)
raw = sys.stdin.buffer.read(header.size)
magic, version, kind, reserved, request_id, length = header.unpack(raw)
sys.stdin.buffer.read(length)
if mode == "truncated":
    out.write(header.pack(b"CIWJ", 1, 3, 0, request_id, 1000) + b"\0" * 400)
elif mode == "malformed":
    out.write(header.pack(b"XIWJ", 1, 3, 0, request_id, 0))
elif mode == "oversized":
    out.write(header.pack(b"CIWJ", 1, 3, 0, request_id, 262145))
elif mode == "wrong_request_id":
    out.write(header.pack(b"CIWJ", 1, 3, 0, request_id + 1, 0))
elif mode == "unknown_kind":
    out.write(header.pack(b"CIWJ", 1, 9, 0, request_id, 0))
elif mode == "silent":
    time.sleep(60)
out.flush()
'''


def mock_session(mode: str, handshake_payload: bytes, accept=None, **options) -> WorkerSession:
    """A protocol mock replaying ``handshake_payload`` (for failure fixtures; its outputs are never numerical evidence)."""
    command = [sys.executable, "-I", "-B", "-c", MOCK_WORKER, mode, handshake_payload.hex()]
    return WorkerSession(command, None, accept, **options)
