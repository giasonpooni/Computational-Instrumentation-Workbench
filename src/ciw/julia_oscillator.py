"""Julia numerical oscillator integration on the SCR execution seam.

The Julia worker is a simulation provider below CIW's execution boundary. Each
execution is retained as an SCR execution specification (program descriptor,
configuration bytes and input payload) with SCR's byte commitments, the exact
worker request and response bytes, and a decoded view derived from those bytes.
The closed-form oscillator in ``ciw.adapters.oscillator`` is the independent
numerical oracle; its comparison is a verification record with measured
errors, not a physical claim. Nothing here is an observation, a calibrated
measurement, an inferred covariance or authority over equipment. The Julia
executable binding is host configuration and never enters a saved workspace.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import math
import re
import struct
import uuid

import numpy as np

from .adapters.oscillator import analytic_trajectory
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .core.identities import evidence_id as run_evidence_id
from .declared_workload import AUTHORITY, RESULT_SCHEMA, _commit, _text
from .exchange import _identity
from .julia_worker import POOL, check_runtime, runtime_projection
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _now, _keys

KIND = "julia-oscillator"
ROLE = "julia"
OPERATION = "ciw.julia-oscillator.v1"
WORKER_OPERATION = "ciw.julia-oscillator.integrate.v1"
SOURCE_SCHEMA = "ciw.julia-oscillator-source.v1"
SESSION_SCHEMA = "ciw.julia-oscillator-session.v1"
VERIFY_SCHEMA = "ciw.julia-oscillator-verification.v1"
REPLAY_SCHEMA = "ciw.julia-oscillator-replay.v1"
REPLAY_VERIFY_SCHEMA = "ciw.julia-oscillator-replay-verification.v1"
INSTRUMENT = "julia-oscillator-trajectory.v1"
ORACLE_REFERENCE = "analytic-damped-oscillator.v1"
ORIGIN = "simulation"
MAX_BYTES = 4 * 1024 * 1024
SOURCE_LIMIT = 64 * 1024
MAX_SAMPLES = 4096
MIN_SAMPLES = 2
MAX_DURATION_S = 12.0
OUTPUT_POLICIES = {"saveat_interpolated": 1, "tstops_stepped": 2}
UNITS = {"q": "m", "v": "m/s", "energy": "J"}
FRAME = "oscillator-state"
CONFIGURATION_MAGIC = b"OSCC"
INPUT_MAGIC = b"OSCI"
OUTPUT_MAGIC = b"OSCO"
REQUEST_MAGIC = b"CIWQ"
PROGRAM = (
    b"ciw.julia-oscillator.integrate.v1\n"
    b"model: q' = v; v' = -2*gamma*v - omega_0^2*q; energy = 0.5*mass*(v^2 + omega_0^2*q^2); state order [q, v]; initial state at t = 0 s\n"
    b"configuration: ciw.julia-oscillator-configuration.v1 little-endian [OSCC][u32 version=1][u32 len][algorithm ascii]"
    b"[f64 abstol][f64 reltol][f64 dt_initial, 0 = automatic][f64 dtmax, 0 = solver default][u64 maxiters][u8 adaptive=1]"
    b"[u8 output_policy: 1 = saveat requested times only through the free 4th-order interpolant, 2 = tstops stepping onto every requested time]\n"
    b"input: ciw.julia-oscillator-input.v1 little-endian [OSCI][u32 version=1][f64 omega_0 rad/s][f64 gamma 1/s][f64 mass kg]"
    b"[f64 q0 m][f64 v0 m/s][u32 n][n x f64 sample times s, strictly increasing, 0 <= t <= 12]; 2 <= n <= 4096\n"
    b"output: ciw.julia-oscillator-output.v1 little-endian; response frame = [OSCO][u32 version=1][u64 request_number] header followed by the committed output bytes: [u32 status]"
    b"[status 0: u32 len, return code ascii, u32 n, n x f64 t, n x f64 q, n x f64 v, n x f64 energy, u64 accepted steps, u64 rejected steps, u64 function evaluations]"
    b"[status != 0: u32 len, refusal message ascii]; the output commitment covers the bytes after the header\n"
    b"arithmetic: binary64; OrdinaryDiffEqTsit5.Tsit5 explicit Runge-Kutta 5(4) with adaptive PI step control; save_start and save_end retained\n"
    b"bounds: 0 < omega_0 <= 20; 0 <= gamma <= 0.5*omega_0; 0 < mass <= 100; |q0| <= 10; |v0| <= 100; abstol, reltol in [1e-14, 1e-2]; maxiters in [1000, 1e8]\n"
    b"faults: 2 = malformed, 3 = unsupported, 4 = bound, 5 = solver did not succeed, 6 = nonfinite output, 7 = incomplete time coverage"
)
_SOLVER_KEYS = {"algorithm", "abstol", "reltol", "dt_initial", "dtmax", "maxiters", "output_policy"}
_MODEL_KEYS = {"omega_0_rad_s", "gamma_s_inv", "mass_kg", "initial_q_m", "initial_v_m_s"}
_GRID_KEYS = {"start_s", "sample_rate_hz", "sample_count", "endpoint"}
_NATIVE_KEYS = {"specification", "specification_identity", "program_identity", "input_identity", "status", "exit_code",
                "output", "output_identity", "computation_identity", "detail", "decoded"}
_OCCURRENCE_KEYS = {"worker_session_id", "engine_occurrence", "request", "request_sha256", "response", "response_sha256", "worker_metadata"}
RESPONSE_HEADER = struct.Struct("<4sIQ")
_SOLVER_STATS = ("accepted_steps", "rejected_steps", "function_evaluations")


# --- source contract ---------------------------------------------------------

def _real(value, name, low=None, high=None, *, exclusive_low=False):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if low is not None and (result < low or (exclusive_low and result == low)):
        raise ValueError(f"{name} is below its declared bound")
    if high is not None and result > high:
        raise ValueError(f"{name} is above its declared bound")
    return result


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer within [{low}, {high}]")
    return value


def _source(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Julia oscillator source requires 1..65536 exact bytes")
    value = _json(raw)
    canonical(value)
    _keys(value, {"schema", "experiment_id", "configuration", "model", "grid"})
    if value["schema"] != SOURCE_SCHEMA:
        raise ValueError("Unsupported Julia oscillator source")
    _text(value["experiment_id"])
    configuration = value["configuration"]
    _keys(configuration, {"solver", "oracle_policy", "replay_policy", "origin", "covariance_status"})
    if configuration["origin"] != ORIGIN or configuration["covariance_status"] != "not_applicable":
        raise ValueError("Julia oscillator sources declare a simulation origin without covariance")
    solver = configuration["solver"]
    _keys(solver, _SOLVER_KEYS)
    if solver["algorithm"] != "Tsit5":
        raise ValueError("Only the Tsit5 profile is registered")
    _real(solver["abstol"], "abstol", 1e-14, 1e-2)
    _real(solver["reltol"], "reltol", 1e-14, 1e-2)
    for key in ("dt_initial", "dtmax"):
        step = _real(solver[key], key, 0.0, MAX_DURATION_S)
        if step != 0.0 and step < 1e-9:
            raise ValueError(f"{key} must be 0 or within [1e-9, 12]")
    _integer(solver["maxiters"], "maxiters", 1000, 100_000_000)
    if solver["output_policy"] not in OUTPUT_POLICIES:
        raise ValueError("Unknown solver output policy")
    oracle = configuration["oracle_policy"]
    _keys(oracle, {"reference", "max_normalized_error"})
    if oracle["reference"] != ORACLE_REFERENCE:
        raise ValueError("The oracle must be the analytic damped oscillator reference")
    _keys(oracle["max_normalized_error"], set(UNITS))
    for name in UNITS:
        _real(oracle["max_normalized_error"][name], "oracle threshold " + name, 0.0, 1.0, exclusive_low=True)
    replay = configuration["replay_policy"]
    _keys(replay, {"max_normalized_difference"})
    _real(replay["max_normalized_difference"], "replay tolerance", 0.0, 1.0)
    model = value["model"]
    _keys(model, _MODEL_KEYS)
    omega_0 = _real(model["omega_0_rad_s"], "omega_0_rad_s", 0.0, 20.0, exclusive_low=True)
    _real(model["gamma_s_inv"], "gamma_s_inv", 0.0, 0.5 * omega_0)
    _real(model["mass_kg"], "mass_kg", 0.0, 100.0, exclusive_low=True)
    _real(model["initial_q_m"], "initial_q_m", -10.0, 10.0)
    _real(model["initial_v_m_s"], "initial_v_m_s", -100.0, 100.0)
    grid = value["grid"]
    _keys(grid, _GRID_KEYS)
    if _real(grid["start_s"], "start_s", 0.0, 0.0) != 0.0 or grid["endpoint"] != "excluded":
        raise ValueError("The v1 grid starts at the initial state and excludes its endpoint")
    rate = _real(grid["sample_rate_hz"], "sample_rate_hz", 0.0, None, exclusive_low=True)
    count = _integer(grid["sample_count"], "sample_count", MIN_SAMPLES, MAX_SAMPLES)
    duration = count / rate
    if not math.isfinite(duration) or duration > MAX_DURATION_S:
        raise ValueError("sample_count / sample_rate_hz must be at most 12 s")
    times = sample_times(grid)
    if not np.all(np.diff(times) > 0.0) or times[-1] > MAX_DURATION_S:
        raise ValueError("The requested grid must be strictly increasing within [0, 12] s")
    return value


def sample_times(grid):
    """Exact requested sample times: k / sample_rate_hz for k below sample_count."""
    return np.arange(int(grid["sample_count"]), dtype=np.float64) / float(grid["sample_rate_hz"])


# --- byte encodings -----------------------------------------------------------

def encode_configuration(solver):
    algorithm = solver["algorithm"].encode("ascii")
    return (CONFIGURATION_MAGIC + struct.pack("<I", 1) + struct.pack("<I", len(algorithm)) + algorithm
            + struct.pack("<dddd", float(solver["abstol"]), float(solver["reltol"]), float(solver["dt_initial"]), float(solver["dtmax"]))
            + struct.pack("<Q", int(solver["maxiters"])) + bytes([1, OUTPUT_POLICIES[solver["output_policy"]]]))


def encode_input(model, times):
    times = np.asarray(times, dtype=np.float64)
    return (INPUT_MAGIC + struct.pack("<I", 1)
            + struct.pack("<ddddd", float(model["omega_0_rad_s"]), float(model["gamma_s_inv"]), float(model["mass_kg"]),
                          float(model["initial_q_m"]), float(model["initial_v_m_s"]))
            + struct.pack("<I", len(times)) + times.astype("<f8").tobytes())


def encode_request(number, configuration, input_payload, operation=WORKER_OPERATION):
    name = operation.encode("ascii")
    return (REQUEST_MAGIC + struct.pack("<I", 1) + struct.pack("<Q", int(number)) + struct.pack("<I", len(name)) + name
            + struct.pack("<I", len(configuration)) + configuration + struct.pack("<I", len(input_payload)) + input_payload)


class _Reader:
    def __init__(self, payload):
        self.payload, self.offset = payload, 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        if self.offset + size > len(self.payload):
            raise ValueError("Worker output ended inside a field")
        values = struct.unpack_from(fmt, self.payload, self.offset)
        self.offset += size
        return values

    def text(self, limit):
        count, = self.take("<I")
        if count > limit:
            raise ValueError("Worker output string exceeds its bound")
        raw, = self.take(f"<{count}s")
        text = raw.decode("ascii", errors="strict") if all(32 <= b <= 126 for b in raw) else None
        if text is None:
            raise ValueError("Worker output string must be printable ASCII")
        return text

    def floats(self, count):
        size = 8 * count
        if self.offset + size > len(self.payload):
            raise ValueError("Worker output ended inside an array")
        values = np.frombuffer(self.payload, dtype="<f8", count=count, offset=self.offset).astype(np.float64)
        self.offset += size
        if not np.all(np.isfinite(values)):
            raise ValueError("Worker output contains a nonfinite value")
        return values

    def finished(self):
        if self.offset != len(self.payload):
            raise ValueError("Worker output has trailing bytes")


def split_response(payload):
    """Separate the occurrence header from the committed output bytes."""
    if not isinstance(payload, bytes) or len(payload) < RESPONSE_HEADER.size:
        raise ValueError("Worker response is shorter than its header")
    magic, version, number = RESPONSE_HEADER.unpack_from(payload, 0)
    if magic != OUTPUT_MAGIC or version != 1:
        raise ValueError("Worker output magic or version mismatch")
    return number, payload[RESPONSE_HEADER.size:]


def decode_output(output):
    """Strictly decode committed output bytes; malformed bytes raise ValueError."""
    if not isinstance(output, bytes):
        raise ValueError("Worker output must be bytes")
    reader = _Reader(output)
    status, = reader.take("<I")
    if status != 0:
        message = reader.text(512)
        reader.finished()
        return {"status": status, "message": message}
    code = reader.text(64)
    count, = reader.take("<I")
    if not MIN_SAMPLES <= count <= MAX_SAMPLES:
        raise ValueError("Worker output sample count is outside the profile")
    arrays = {name: reader.floats(count) for name in ("time_s", "q", "v", "energy")}
    accepted, rejected, evaluations = reader.take("<QQQ")
    reader.finished()
    return {"status": 0, "return_code": code,
            **{name: values.tolist() for name, values in arrays.items()},
            "solver": {"accepted_steps": accepted, "rejected_steps": rejected, "function_evaluations": evaluations}}


def _energy(model, q, v):
    """Same operation order as the worker and the analytical reference."""
    mass, omega_0 = float(model["mass_kg"]), float(model["omega_0_rad_s"])
    return 0.5 * mass * (v * v + omega_0 * omega_0 * q * q)


def _check_completed(decoded, source, times):
    """A completed output must answer the exact request with finite, consistent values."""
    if decoded["status"] != 0 or decoded["return_code"] != "Success":
        raise ValueError("Worker output is not a successful Tsit5 solve")
    if decoded["time_s"] != times.tolist():
        raise ValueError("Worker output does not cover the exact requested sample times")
    q, v, energy = (np.asarray(decoded[name], dtype=np.float64) for name in ("q", "v", "energy"))
    if not np.array_equal(_energy(source["model"], q, v), energy):
        raise ValueError("Worker energy derivation differs from the declared definition")
    solver = decoded["solver"]
    if set(solver) != set(_SOLVER_STATS) or any(type(solver[k]) is not int or solver[k] < 0 for k in _SOLVER_STATS):
        raise ValueError("Worker solver diagnostics are malformed")
    if solver["accepted_steps"] < 1 or solver["function_evaluations"] < solver["accepted_steps"]:
        raise ValueError("Worker solver diagnostics are inconsistent with a completed solve")


# --- oracle comparison --------------------------------------------------------

def oracle_reference(model, times):
    """Closed-form q, v and energy at the requested times, retained as verification input."""
    q, v, energy = analytic_trajectory(float(model["omega_0_rad_s"]), float(model["gamma_s_inv"]), float(model["mass_kg"]),
                                       float(model["initial_q_m"]), float(model["initial_v_m_s"]), np.asarray(times, dtype=np.float64))
    reference = {"reference": ORACLE_REFERENCE, "generator": "ciw.adapters.oscillator.analytic_trajectory",
                 "generator_version": 1, "dtype": "float64", "numpy_version": np.__version__,
                 "q": q.tolist(), "v": v.tolist(), "energy": energy.tolist()}
    reference["reference_digest"] = digest({name: reference[name] for name in UNITS})
    return reference


def measure_errors(decoded, reference, model):
    """Exactly recomputable componentwise error measures against retained reference values.

    Only subtraction, absolute value, maximum and one division are used, so a
    retained record can be rechecked bitwise on any IEEE binary64 host.
    """
    measured = {}
    for name in UNITS:
        values = np.asarray(decoded[name], dtype=np.float64)
        expected = np.asarray(reference[name], dtype=np.float64)
        if values.shape != expected.shape:
            raise ValueError("Reference and decoded trajectories differ in length")
        error = np.abs(values - expected)
        index = int(np.argmax(error))
        scale = float(np.max(np.abs(expected)))
        normalization = scale if scale > 0.0 else 1.0
        measured[name] = {"unit": UNITS[name], "max_abs_error": float(error[index]), "at_sample_index": index,
                          "reference_scale": scale, "normalized_max_error": float(error[index]) / normalization}
    energy = np.asarray(decoded["energy"], dtype=np.float64)
    if float(model["gamma_s_inv"]) != 0.0:
        conservation = {"status": "not_applicable_damped"}
    elif energy[0] <= 0.0:
        conservation = {"status": "not_applicable_zero_energy"}
    else:
        conservation = {"status": "evaluated", "initial_energy_j": float(energy[0]),
                        "max_abs_change_j": float(np.max(np.abs(energy - energy[0]))),
                        "relative_drift": float(np.max(np.abs(energy - energy[0]))) / float(energy[0])}
    measured["numerical_energy_conservation"] = conservation
    return measured


def _outcome(measured, thresholds):
    return "passed" if all(measured[name]["normalized_max_error"] <= float(thresholds[name]) for name in UNITS) else "failed"


def _identify(report):
    report["verification_id"] = byte_digest(report["schema"].encode() + b"\0" + canonical(report))
    return report


# --- workflow -----------------------------------------------------------------

class JuliaOscillatorWorkflow:
    MAX_BYTES = MAX_BYTES
    ROLES = {ROLE}
    SOURCE_SCHEMA = SOURCE_SCHEMA

    def __init__(self, kind=KIND):
        if kind != KIND:
            raise ValueError("Unsupported Julia operation")
        self.kind, self.role = kind, ROLE
        self.schema, self.operation = SESSION_SCHEMA, OPERATION

    def _source(self, raw):
        return _source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        return runtime_projection(runtime)

    def _adapters(self, repositories, expected=None):
        """Obtain the live host-owned worker for the bound executable; refuse a mismatch."""
        if set(repositories) != self.ROLES:
            raise ValueError("Bind exactly the Julia executable role")
        session = POOL.session(repositories[ROLE])
        runtime = deepcopy(session.identity)
        check_runtime(runtime)
        if expected is not None and self._runtime_projection(runtime) != self._runtime_projection(expected[ROLE]):
            raise ValueError("Julia worker runtime differs from the retained execution")
        return session, runtime

    def _step(self, source, evidence_id, bound):
        session, runtime = bound
        if self._runtime_projection(session.identity) != self._runtime_projection(runtime):
            raise ValueError("Julia worker identity changed before execution")
        configuration = encode_configuration(source["configuration"]["solver"])
        times = sample_times(source["grid"])
        input_payload = encode_input(source["model"], times)
        sent = {}

        def build(number):
            sent["bytes"] = encode_request(number, configuration, input_payload)
            return sent["bytes"]

        number, response, metadata = session.execute(build)
        try:
            bound_number, output = split_response(response)
            decoded = decode_output(output)
        except ValueError as exc:
            session.fail("MALFORMED_RESPONSE", str(exc))
            raise AdapterRefusal("MALFORMED_RESPONSE", "The Julia worker returned an undecodable oscillator output") from exc
        if bound_number != number:
            session.fail("MALFORMED_RESPONSE", "Worker output does not bind the request occurrence")
            raise AdapterRefusal("MALFORMED_RESPONSE", "Worker output does not bind the request occurrence")
        if decoded["status"] != 0:
            raise AdapterRefusal("JULIA_OSCILLATOR_REFUSED",
                                 f"The Julia worker refused the request (status {decoded['status']}): {decoded['message']}")
        try:
            _check_completed(decoded, source, times)
        except ValueError as exc:
            raise AdapterRefusal("INVALID_WORKER_OUTPUT", str(exc)) from exc
        specification = {"program": PROGRAM.hex(), "configuration": configuration.hex(), "input_payload": input_payload.hex()}
        native = {"specification": specification,
                  "specification_identity": _commit("specification", [PROGRAM, configuration, input_payload]),
                  "program_identity": _commit("program", [PROGRAM]), "input_identity": _commit("input", [input_payload]),
                  "status": "completed", "exit_code": 0, "output": output.hex(),
                  "output_identity": _commit("output", [output]), "detail": None, "decoded": decoded}
        native["computation_identity"] = _commit("computation", [bytes.fromhex(native[k]) for k in ("program_identity", "input_identity", "output_identity")] + [struct.pack("<I", 0)])
        occurrence_record = {"worker_session_id": session.session_id, "engine_occurrence": number,
                             "request": sent["bytes"].hex(), "request_sha256": byte_digest(sent["bytes"]),
                             "response": response.hex(), "response_sha256": byte_digest(response), "worker_metadata": metadata}
        data = {"native": native, "occurrence": occurrence_record}
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": deepcopy(AUTHORITY)}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": deepcopy(native)}
        return {"runtime_ref": ROLE, "operation_id": self.operation, "execution_id": occurrence,
                "input_refs": [evidence_id], "request": deepcopy(source), "request_sha256": digest(source),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    @staticmethod
    def _check_data(source, data):
        _keys(data, {"native", "occurrence"})
        native, occurrence = data["native"], data["occurrence"]
        _keys(native, _NATIVE_KEYS)
        _keys(occurrence, _OCCURRENCE_KEYS)
        configuration = encode_configuration(source["configuration"]["solver"])
        times = sample_times(source["grid"])
        input_payload = encode_input(source["model"], times)
        if native["specification"] != {"program": PROGRAM.hex(), "configuration": configuration.hex(), "input_payload": input_payload.hex()}:
            raise ValueError("Retained specification differs from the declared source")
        if native["status"] != "completed" or native["exit_code"] != 0 or native["detail"] is not None:
            raise ValueError("Retained Julia occurrence is not a completed execution")
        try:
            output = bytes.fromhex(native["output"]) if isinstance(native["output"], str) else None
        except ValueError:
            output = None
        if output is None or not output:
            raise ValueError("Retained output bytes are malformed")
        decoded = decode_output(output)
        if canonical(decoded) != canonical(native["decoded"]):
            raise ValueError("Decoded view differs from the retained output bytes")
        number = occurrence["engine_occurrence"]
        if type(number) is not int or number < 1:
            raise ValueError("Invalid worker occurrence number")
        try:
            response = bytes.fromhex(occurrence["response"]) if isinstance(occurrence["response"], str) else None
        except ValueError:
            response = None
        if response != RESPONSE_HEADER.pack(OUTPUT_MAGIC, 1, number) + output or occurrence["response_sha256"] != byte_digest(response):
            raise ValueError("Retained response frame does not bind the committed output to its worker occurrence")
        _check_completed(decoded, source, times)
        expected = {"program_identity": _commit("program", [PROGRAM]), "input_identity": _commit("input", [input_payload]),
                    "output_identity": _commit("output", [output]),
                    "specification_identity": _commit("specification", [PROGRAM, configuration, input_payload])}
        expected["computation_identity"] = _commit("computation", [bytes.fromhex(expected[k]) for k in ("program_identity", "input_identity", "output_identity")] + [struct.pack("<I", 0)])
        if any(native[k] != v for k, v in expected.items()):
            raise ValueError("SCR execution commitment mismatch")
        if not isinstance(occurrence["worker_session_id"], str) or not re.fullmatch(r"julia-worker-[a-f0-9]{32}", occurrence["worker_session_id"]):
            raise ValueError("Invalid worker session identity")
        try:
            request = bytes.fromhex(occurrence["request"]) if isinstance(occurrence["request"], str) else None
        except ValueError:
            request = None
        if request != encode_request(number, configuration, input_payload) or occurrence["request_sha256"] != byte_digest(request):
            raise ValueError("Retained request bytes differ from the specification and occurrence")
        metadata = occurrence["worker_metadata"]
        _keys(metadata, {"schema", "request_number", "worker_occurrence", "elapsed_ns"})
        if (metadata["schema"] != "ciw.julia-worker-occurrence.v1" or metadata["request_number"] != number or
                metadata["worker_occurrence"] != number or type(metadata["elapsed_ns"]) is not int or metadata["elapsed_ns"] < 0):
            raise ValueError("Worker occurrence metadata does not bind the retained request")

    def _validate_step(self, step, source, evidence):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                     "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if (step["runtime_ref"] != ROLE or step["operation_id"] != self.operation or step["input_refs"] != [evidence] or
                not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"])):
            raise ValueError("Invalid Julia oscillator execution or source binding")
        if canonical(step["request"]) != canonical(source):
            raise ValueError("Retained request differs from the exact source")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        self._check_data(source, result["data"])
        if canonical(result["authority"]) != canonical(AUTHORITY):
            raise ValueError("A simulation cannot confer physical or admission authority")
        if (result["schema"] != RESULT_SCHEMA or result["operation_id"] != self.operation or
                result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence] or
                result["result_id"] != step["result_id"] or result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"})):
            raise ValueError("Julia oscillator result binding mismatch")
        if canonical(step["numerical_result"]) != canonical({"operation_id": self.operation, "data": result["data"]["native"]}):
            raise ValueError("Numerical identity must cover the committed bytes and exclude occurrence details")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("Julia oscillator step content identity mismatch")

    def _verification(self, bundle, reference, occurrence=None):
        step = bundle["steps"][0]
        source = step["request"]
        decoded = step["result"]["data"]["native"]["decoded"]
        thresholds = source["configuration"]["oracle_policy"]["max_normalized_error"]
        measured = measure_errors(decoded, reference, source["model"])
        report = {"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": _outcome(measured, thresholds),
                  "independent": False, "method": "analytical_oscillator_oracle_comparison",
                  "runtime_digest": digest(bundle["runtimes"]), "result_id": step["result_id"], "execution_id": step["execution_id"],
                  "verification_operation_id": occurrence or ("verification-" + uuid.uuid4().hex),
                  "oracle": deepcopy(reference), "thresholds": {"kind": "normalized_max_error", **{k: float(thresholds[k]) for k in UNITS}},
                  "measured": measured,
                  "scope": "numerical agreement with the closed-form model at the requested samples; not a physical, calibration or uncertainty claim",
                  "authority": deepcopy(AUTHORITY)}
        return _identify(report)

    def _check_verification(self, bundle, report):
        _keys(report, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "result_id", "execution_id",
                       "verification_operation_id", "oracle", "thresholds", "measured", "scope", "authority", "verification_id"})
        occurrence = report["verification_operation_id"]
        if not isinstance(occurrence, str) or not re.fullmatch(r"verification-[a-f0-9]{32}", occurrence):
            raise ValueError("Invalid oracle verification occurrence")
        reference = report["oracle"]
        _keys(reference, {"reference", "generator", "generator_version", "dtype", "numpy_version", "q", "v", "energy", "reference_digest"})
        step = bundle["steps"][0]
        count = int(step["request"]["grid"]["sample_count"])
        for name in UNITS:
            values = reference[name]
            if (not isinstance(values, list) or len(values) != count or
                    any(isinstance(x, bool) or type(x) not in (int, float) or not math.isfinite(x) for x in values)):
                raise ValueError("Retained oracle reference is malformed")
        if (reference["reference"] != ORACLE_REFERENCE or reference["generator"] != "ciw.adapters.oscillator.analytic_trajectory" or
                reference["generator_version"] != 1 or reference["dtype"] != "float64" or not isinstance(reference["numpy_version"], str) or
                reference["reference_digest"] != digest({name: reference[name] for name in UNITS})):
            raise ValueError("Retained oracle reference identity mismatch")
        if canonical(report) != canonical(self._verification(bundle, reference, occurrence)):
            raise ValueError("Oracle verification differs from its retained measured statement")
        _identity(report, "verification_id")

    def _execute(self, raw, bound):
        source, evidence = self._source(raw), byte_digest(raw)
        step = self._step(source, evidence, bound)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(source["configuration"]), "runtimes": {ROLE: bound[1]}, "steps": [step]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = self._verification(bundle, oracle_reference(source["model"], sample_times(source["grid"])))
        self._validate(bundle)
        return bundle

    def _validate(self, bundle):
        """Validate retained bindings and recompute exact error measures; never execute Julia."""
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != _bundle_digest(bundle):
                raise ValueError("Julia oscillator bundle exceeds budget or content binding differs")
            if not isinstance(bundle["session_id"], str) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Invalid Julia oscillator session occurrence")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            if canonical(evidence) != canonical({"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}):
                raise ValueError("Exact source binding mismatch")
            if canonical(bundle["source"]) != canonical({"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]}):
                raise ValueError("Source identity mismatch")
            if canonical(bundle["configuration"]) != canonical(source["configuration"]):
                raise ValueError("Source configuration mismatch")
            _keys(bundle["runtimes"], {ROLE})
            check_runtime(bundle["runtimes"][ROLE])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"])
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError("A Julia oscillator occurrence retains at most one replay receipt")
            for receipt in receipts:
                self._check_receipt(bundle, receipt)
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed Julia oscillator session") from exc

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories))

    # -- replay -------------------------------------------------------------

    @staticmethod
    def _agreement(original, fresh):
        """Normalized maximum differences between two retained decoded trajectories."""
        old, new = original["steps"][0], fresh["steps"][0]
        before, after = old["result"]["data"]["native"], new["result"]["data"]["native"]
        threshold = float(fresh["configuration"]["replay_policy"]["max_normalized_difference"])
        agreement = {"threshold_normalized": threshold}
        for name in UNITS:
            previous = np.asarray(before["decoded"][name], dtype=np.float64)
            current = np.asarray(after["decoded"][name], dtype=np.float64)
            if previous.shape != current.shape:
                raise ValueError("Replay produced a different sample count")
            difference = float(np.max(np.abs(current - previous)))
            scale = float(np.max(np.abs(previous)))
            normalized = difference / (scale if scale > 0.0 else 1.0)
            agreement[name] = {"unit": UNITS[name], "max_abs_difference": difference, "normalized_max_difference": normalized}
            if normalized > threshold:
                raise ValueError(f"Fresh Julia execution differs from the retained {name} trajectory beyond the declared replay tolerance")
        return agreement, before["output"] == after["output"]

    def _replay_verification(self, original, fresh):
        old, new = original["steps"][0], fresh["steps"][0]
        if old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"]:
            raise ValueError("Replay requires a distinct execution occurrence")
        agreement, identical = self._agreement(original, fresh)
        return _identify({"schema": REPLAY_VERIFY_SCHEMA, "subject_ref": original["bundle_digest"], "outcome": "passed",
                          "independent": False, "method": "same_runtime_fresh_occurrence_declared_tolerance_agreement",
                          "runtime_digest": digest(original["runtimes"]), "fresh_runtime_digest": digest(fresh["runtimes"]),
                          "original_execution_id": old["execution_id"], "fresh_execution_id": new["execution_id"],
                          "fresh_result_id": new["result_id"], "original_numerical_result_id": old["numerical_result_id"],
                          "fresh_numerical_result_id": new["numerical_result_id"], "byte_identical": identical,
                          "agreement": agreement, "fresh_oracle_outcome": fresh["verification"]["outcome"],
                          "authority": deepcopy(AUTHORITY)})

    def _check_receipt(self, fresh, receipt):
        _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
        source_id = receipt["source_bundle_digest"]
        if (receipt["schema"] != REPLAY_SCHEMA or not isinstance(source_id, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", source_id) or
                source_id == fresh["bundle_digest"] or receipt["replayed_bundle_digest"] != fresh["bundle_digest"] or
                receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                receipt["replay_id"] != digest({k: v for k, v in receipt.items() if k != "replay_id"})):
            raise ValueError("Invalid Julia oscillator replay receipt")
        verification = receipt["verification"]
        _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "fresh_runtime_digest",
                             "original_execution_id", "fresh_execution_id", "fresh_result_id", "original_numerical_result_id",
                             "fresh_numerical_result_id", "byte_identical", "agreement", "fresh_oracle_outcome", "authority", "verification_id"})
        new = fresh["steps"][0]
        threshold = float(fresh["configuration"]["replay_policy"]["max_normalized_difference"])
        agreement = verification["agreement"]
        _keys(agreement, {"threshold_normalized"} | set(UNITS))
        for name in UNITS:
            entry = agreement[name]
            _keys(entry, {"unit", "max_abs_difference", "normalized_max_difference"})
            if (entry["unit"] != UNITS[name] or _real(entry["max_abs_difference"], name, 0.0) < 0 or
                    _real(entry["normalized_max_difference"], name, 0.0, threshold) < 0):
                raise ValueError("Replay agreement exceeds the declared tolerance")
        if (verification["schema"] != REPLAY_VERIFY_SCHEMA or verification["subject_ref"] != source_id or verification["outcome"] != "passed" or
                verification["independent"] is not False or verification["method"] != "same_runtime_fresh_occurrence_declared_tolerance_agreement" or
                verification["fresh_runtime_digest"] != digest(fresh["runtimes"]) or verification["fresh_execution_id"] != new["execution_id"] or
                verification["fresh_result_id"] != new["result_id"] or verification["fresh_numerical_result_id"] != new["numerical_result_id"] or
                verification["original_execution_id"] == new["execution_id"] or agreement["threshold_normalized"] != threshold or
                not isinstance(verification["byte_identical"], bool) or
                verification["byte_identical"] != (verification["original_numerical_result_id"] == new["numerical_result_id"]) or
                verification["fresh_oracle_outcome"] != fresh["verification"]["outcome"] or canonical(verification["authority"]) != canonical(AUTHORITY)):
            raise ValueError("Replay verification does not bind the exact fresh occurrence")
        _identity(verification, "verification_id")

    def validate_replay(self, original, fresh, receipt):
        self._validate(original)
        self._validate(fresh)
        self._check_receipt(fresh, receipt)
        if canonical(original["source"]) != canonical(fresh["source"]):
            raise ValueError("Replay must retain exact source bytes")
        if self._runtime_projection(original["runtimes"][ROLE]) != self._runtime_projection(fresh["runtimes"][ROLE]):
            raise ValueError("Replay runtime changed")
        if original["session_id"] == fresh["session_id"]:
            raise ValueError("Replay requires a fresh session occurrence")
        if receipt["source_bundle_digest"] != original["bundle_digest"]:
            raise ValueError("Replay receipt names another original bundle")
        if canonical(receipt["verification"]) != canonical(self._replay_verification(original, fresh)):
            raise ValueError("Replay differs from retained historical context")

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute(raw, self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": REPLAY_SCHEMA, "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": self._replay_verification(bundle, fresh), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self.validate_replay(bundle, fresh, receipt)
        return {"session": fresh, "replay_receipt": receipt}


# --- terminal helpers ----------------------------------------------------------

def read_bundle(path):
    """Read a retained bundle within the byte budget and validate it offline."""
    from .exchange import _read
    bundle = _json(_read(path, MAX_BYTES))
    JuliaOscillatorWorkflow()._validate(bundle)
    return bundle


def write_bundle(bundle, output_dir):
    """Write bundle.json and its run.v1 projection into a new or empty directory."""
    from pathlib import Path
    from .session import write_json
    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Output directory must be new or empty; retained occurrences are never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    return {"bundle_file": str(write_json(directory / "bundle.json", bundle)),
            "recording_file": str(write_json(directory / "recording.json", project_run(bundle)))}


def inspect_bundle(bundle):
    """Offline summary of retained identities, solver work and measured oracle errors."""
    JuliaOscillatorWorkflow()._validate(bundle)
    step = bundle["steps"][0]
    data = step["result"]["data"]
    verification = bundle["verification"]
    return {"schema": bundle["schema"], "session_id": bundle["session_id"], "bundle_digest": bundle["bundle_digest"],
            "experiment_id": bundle["source"]["experiment_id"], "evidence_id": step["input_refs"][0],
            "execution_id": step["execution_id"], "result_id": step["result_id"], "numerical_result_id": step["numerical_result_id"],
            "specification_identity": data["native"]["specification_identity"], "computation_identity": data["native"]["computation_identity"],
            "worker_session_id": data["occurrence"]["worker_session_id"], "engine_occurrence": data["occurrence"]["engine_occurrence"],
            "runtime_digest": digest(bundle["runtimes"]), "julia_version": bundle["runtimes"][ROLE]["julia_version"],
            "solver": deepcopy(bundle["configuration"]["solver"]), "solver_return_code": data["native"]["decoded"]["return_code"],
            "solver_statistics": deepcopy(data["native"]["decoded"]["solver"]), "sample_count": len(data["native"]["decoded"]["time_s"]),
            "verification_id": verification["verification_id"], "oracle_outcome": verification["outcome"],
            "thresholds": deepcopy(verification["thresholds"]), "measured": deepcopy(verification["measured"]),
            "replay_receipts": [{"replay_id": r["replay_id"], "source_bundle_digest": r["source_bundle_digest"],
                                 "byte_identical": r["verification"]["byte_identical"], "agreement": r["verification"]["agreement"]}
                                for r in bundle.get("replay_receipts", [])],
            "origin": ORIGIN, "physical_validation": "not_established", "state_admission": "not_performed",
            "numerical_replay": "not_performed_by_inspection"}


# --- projections --------------------------------------------------------------

def _validated_source(bundle):
    evidence, = bundle["source"]["evidence"]
    return _source(base64.b64decode(evidence["bytes_b64"], validate=True))


def project_run(bundle):
    """Derive a ``run.v1`` recording for the existing state-space viewport.

    The recording is a projection of retained bytes: its provenance names the
    bundle, result, execution and commitments it came from. It is simulated
    evidence, never sensor acquisition, and render geometry is display only.
    """
    workflow = JuliaOscillatorWorkflow()
    workflow._validate(bundle)
    source = _validated_source(bundle)
    step = bundle["steps"][0]
    native = step["result"]["data"]["native"]
    decoded = native["decoded"]
    model, grid, solver = source["model"], source["grid"], source["configuration"]["solver"]
    time = np.asarray(decoded["time_s"], dtype=np.float64)
    q, v, energy = (np.asarray(decoded[name], dtype=np.float64) for name in ("q", "v", "energy"))
    omega_0, mass = float(model["omega_0_rad_s"]), float(model["mass_kg"])
    amplitude_q = float(np.max(np.abs(q))) or 1.0
    amplitude_v = float(np.max(np.abs(v))) or 1.0
    q_axis = np.linspace(-1.1 * amplitude_q, 1.1 * amplitude_q, 25, dtype=np.float64)
    v_axis = np.linspace(-1.1 * amplitude_v, 1.1 * amplitude_v, 25, dtype=np.float64)
    vertices = [[float(qi), float(_energy(model, np.float64(qi), np.float64(vi))), float(vi)] for vi in v_axis for qi in q_axis]
    surface_max = max(vertex[1] for vertex in vertices) or 1.0
    indices = []
    width = len(q_axis)
    for row in range(len(v_axis) - 1):
        for column in range(width - 1):
            a = row * width + column
            indices.extend([a, a + width, a + 1, a + 1, a + width, a + width + 1])
    channels = {name: {"unit": UNITS[name], "values": values.tolist()} for name, values in (("q", q), ("v", v), ("energy", energy))}
    metadata = {
        "duration_s": int(grid["sample_count"]) / float(grid["sample_rate_hz"]), "sample_rate_hz": float(grid["sample_rate_hz"]),
        "sample_count": len(time), "coordinate_frame": FRAME,
        "model": {"equation": "q'' + 2*gamma*q' + omega_0^2*q = 0", "mass_kg": mass, "omega_0_rad_s": omega_0,
                  "gamma_s_inv": float(model["gamma_s_inv"]), "initial_q_m": float(model["initial_q_m"]),
                  "initial_v_m_s": float(model["initial_v_m_s"]), "energy_definition": "0.5*mass*(v^2 + omega_0^2*q^2)",
                  "solver": deepcopy(solver), "solver_return_code": decoded["return_code"], "solver_statistics": deepcopy(decoded["solver"])},
        "provenance": {
            "source": "numerical simulation; Julia Tsit5 integration of the declared model; synthetic evidence, not sensor acquisition",
            "origin": ORIGIN, "generator": "ciw.julia_oscillator.project_run", "generator_version": 1, "dtype": "float64",
            "time_reference": "seconds since run start", "sampling": "uniform; endpoint excluded",
            "bundle_digest": bundle["bundle_digest"], "result_id": step["result_id"], "execution_id": step["execution_id"],
            "specification_identity": native["specification_identity"], "computation_identity": native["computation_identity"],
            "worker_runtime_digest": digest(bundle["runtimes"]), "verification_id": bundle["verification"]["verification_id"],
            "oracle_outcome": bundle["verification"]["outcome"],
            "oracle_normalized_max_error": {name: bundle["verification"]["measured"][name]["normalized_max_error"] for name in UNITS},
        },
    }
    run = {"run_id": "run-julia-oscillator-" + step["execution_id"][len("execution-"):], "instrument": INSTRUMENT,
           "metadata": metadata, "time_s": time.tolist(), "channels": channels}
    run["evidence_id"] = run_evidence_id(run)
    run["render"] = {
        "coordinate_frame": FRAME, "axis_labels": ["q (m)", "energy (J)", "v (m/s)"],
        "trajectory": np.column_stack((q, energy, v)).tolist(), "sample_indices": list(range(len(time))),
        "surface": {"vertices": vertices, "indices": indices},
        "transform": {"origin": [0.0, 0.0, 0.0], "scale": [2.5 / amplitude_q, 4.6 / surface_max, 2.5 / amplitude_v],
                      "note": "Visual display scaling only; source coordinates and units stay physical."},
    }
    return run


def state_trajectory_view(bundle):
    """Retained state-trajectory projection of the Julia result and its oracle."""
    from .state_trajectory import projection
    source = _validated_source(bundle)
    step = bundle["steps"][0]
    native = step["result"]["data"]["native"]
    decoded, reference = native["decoded"], bundle["verification"]["oracle"]
    provenance = {"bundle_id": bundle["bundle_digest"], "result_id": step["result_id"], "execution_id": step["execution_id"],
                  "evidence_id": step["input_refs"][0], "specification_identity": native["specification_identity"],
                  "computation_identity": native["computation_identity"]}
    basis = {"coordinate_frame": FRAME, "state_order": ["q", "v"], "model": deepcopy(source["model"])}
    simulated = projection(
        origin=ORIGIN, label="Julia Tsit5 numerical integration",
        independent_axis={"name": "time", "unit": "s", "values": decoded["time_s"]},
        state_axes=[{"name": name, "unit": UNITS[name], "values": decoded[name]} for name in ("q", "v")],
        derived_quantities=[{"name": "energy", "unit": "J", "definition": "0.5*mass*(v^2 + omega_0^2*q^2)", "values": decoded["energy"]}],
        coordinate_basis=basis, provenance=provenance, verification_ref=bundle["verification"]["verification_id"])
    oracle = projection(
        origin="reference", label="Analytic damped oscillator reference",
        independent_axis={"name": "time", "unit": "s", "values": decoded["time_s"]},
        state_axes=[{"name": name, "unit": UNITS[name], "values": reference[name]} for name in ("q", "v")],
        derived_quantities=[{"name": "energy", "unit": "J", "definition": "0.5*mass*(v^2 + omega_0^2*q^2)", "values": reference["energy"]}],
        coordinate_basis=basis,
        provenance={**provenance, "reference_digest": reference["reference_digest"], "generator": reference["generator"]},
        verification_ref=bundle["verification"]["verification_id"])
    return {"simulation": simulated, "reference": oracle}
