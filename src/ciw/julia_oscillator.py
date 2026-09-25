"""Julia Tsit5 oscillator provider on the CIW/SCR retention seam.

The provider is deliberately narrow.  CIW owns source, execution and result
identities; Julia receives only a bounded data request and returns framed JSON.
The retained record keeps the exact request/response bodies, while the Python
analytic solution remains the independent oracle used for comparison.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import uuid

import numpy as np

from .adapters.oscillator import validate_run
from .adapters.subprocess import _bounded_process, _json
from .exchange import _identity
from .telemetry import _bundle_digest, _now, byte_digest, canonical, digest, _keys

KIND = "julia-oscillator"
SCHEMA = "ciw.julia-oscillator-session.v1"
SOURCE_SCHEMA = "ciw.julia-oscillator-source.v1"
REQUEST_SCHEMA = "ciw.julia-oscillator-request.v1"
RESPONSE_SCHEMA = "ciw.julia-worker-response.v1"
RESULT_SCHEMA = "ciw.julia-oscillator-workbench-result.v1"
VERIFY_SCHEMA = "ciw.julia-oscillator-verification.v1"
OPERATION = "ciw.julia-oscillator.v1"
ROLE = "julia"
RUNTIME_ROLE = "julia_runtime"
ROLES = {ROLE, RUNTIME_ROLE}
MAX_BYTES = 16 * 1024 * 1024
SOURCE_LIMIT = 256 * 1024
FRAME_LIMIT = 4 * 1024 * 1024
DEFAULT_ABSTOL = 1e-10
DEFAULT_RELTOL = 1e-10
ORACLE_ATOL = 2e-8
ORACLE_RTOL = 2e-8
AUTHORITY = {
    "claim_scope": "simulated_numerical_trajectory_against_independent_analytic_oracle",
    "physical_validation": "not_established",
    "measurement_uncertainty": "not_declared",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}


def _number(value, name, *, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{name} is outside its declared finite bounds")
    return value


def _array(value, name, *, minimum=1, maximum=4096):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must be a bounded list")
    result = [_number(item, f"{name}[]") for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _path(value, name, *, directory=None):
    path = Path(value).expanduser().absolute()
    if directory is True and not path.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    if directory is False and not path.is_file():
        raise ValueError(f"{name} must be an existing executable file")
    return path


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Julia runtime binding must be a pinned Git checkout") from exc
    head = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head):
        raise ValueError("Julia runtime checkout does not expose a full revision")
    clean = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, timeout=10, check=True)
    if clean.stdout.strip():
        raise ValueError("Julia runtime checkout contains uncommitted or untracked files")
    return head


def _worker_path(root: Path) -> Path:
    candidates = (root / "runtimes" / "julia-oscillator" / "oscillator_worker.jl",
                  root / "oscillator_worker.jl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("Julia runtime binding has no oscillator_worker.jl")


def _runtime_files(root: Path):
    worker = _worker_path(root)
    project = worker.with_name("Project.toml")
    manifest = worker.with_name("Manifest.toml")
    if not project.is_file() or not manifest.is_file():
        raise ValueError("Julia oscillator environment must contain an instantiated Project.toml and Manifest.toml")
    return worker, project, manifest


def _sha_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def runtime_identity(repositories):
    _keys(repositories, {ROLE, RUNTIME_ROLE})
    executable = _path(repositories[ROLE], ROLE, directory=False)
    root = _path(repositories[RUNTIME_ROLE], RUNTIME_ROLE, directory=True)
    worker, project, manifest = _runtime_files(root)
    revision = _git_head(root)
    version_code, version_bytes = _bounded_process([str(executable), "--version"], cwd=root,
                                                    timeout=10, limit=64 * 1024)
    if version_code:
        raise ValueError("Julia executable did not report its version")
    version = version_bytes.decode("utf-8", "replace").strip()
    # Julia's launcher reports a lowercase ``julia version`` string on the
    # Windows distribution while some builds use title case.  The version
    # pin is semantic; casing is not part of the runtime identity.
    if not re.search(r"Julia Version 1\.10\.(?:1[0-2])\b", version, re.IGNORECASE):
        raise ValueError("Julia runtime must be the declared 1.10 LTS line")
    return {
        "schema": "ciw.julia-oscillator-runtime.v1", "role": ROLE,
        "profile": "ordinarydiffeqtsit5", "execution_scope": "bounded_simulated_oscillator",
        "julia_version": version, "platform": platform.platform(), "threads": 1,
        "startup_file": "disabled", "runtime_revision": revision,
        "worker_sha256": _sha_file(worker), "project_sha256": _sha_file(project),
        "manifest_sha256": _sha_file(manifest), "executable_sha256": _sha_file(executable),
        "worker_relative_path": worker.relative_to(root).as_posix(),
    }


def _runtime_projection(value):
    return deepcopy(value)


def _adapters(repositories, expected=None):
    if not isinstance(repositories, dict):
        raise ValueError("Julia oscillator bindings must be an object")
    runtime = runtime_identity(repositories)
    if expected is not None:
        expected_runtime = expected.get(RUNTIME_ROLE, expected)
        if _runtime_projection(runtime) != _runtime_projection(expected_runtime):
            raise ValueError("Julia runtime identity differs from the retained execution")
    return repositories, runtime


def validate_source(raw: bytes) -> dict:
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Julia oscillator source must contain bounded exact bytes")
    source = _json(raw)
    if canonical(source) != raw:
        raise ValueError("Julia oscillator source bytes must use canonical JSON")
    _keys(source, {"schema", "experiment_id", "operation_id", "model", "initial_state", "time_s", "solver", "claim_scope"})
    if source["schema"] != SOURCE_SCHEMA or source["operation_id"] != OPERATION:
        raise ValueError("Unsupported Julia oscillator source")
    if not isinstance(source["experiment_id"], str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", source["experiment_id"]):
        raise ValueError("experiment_id must be bounded and stable")
    model = source["model"]
    _keys(model, {"omega_0_rad_s", "gamma_s_inv", "mass_kg"})
    omega = _number(model["omega_0_rad_s"], "omega_0_rad_s", lo=np.finfo(float).eps, hi=20.0)
    gamma = _number(model["gamma_s_inv"], "gamma_s_inv", lo=0.0, hi=10.0)
    _number(model["mass_kg"], "mass_kg", lo=np.finfo(float).eps, hi=100.0)
    if gamma > 0.5 * omega:
        raise ValueError("initial profile accepts only underdamped or undamped cases")
    initial = source["initial_state"]
    _keys(initial, {"q0_m", "v0_m_s"})
    _number(initial["q0_m"], "q0_m", lo=-10.0, hi=10.0)
    _number(initial["v0_m_s"], "v0_m_s", lo=-100.0, hi=100.0)
    times = _array(source["time_s"], "time_s", minimum=2)
    if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("time_s must start at zero and be strictly increasing")
    if times[-1] > 12.0 or (times[-1] == 0.0):
        raise ValueError("time_s is outside the twelve-second profile")
    solver = source["solver"]
    _keys(solver, {"abstol", "reltol", "maxiters"})
    _number(solver["abstol"], "solver.abstol", lo=1e-14, hi=1e-3)
    _number(solver["reltol"], "solver.reltol", lo=1e-14, hi=1e-3)
    if isinstance(solver["maxiters"], bool) or not isinstance(solver["maxiters"], int) or not 1 <= solver["maxiters"] <= 10_000_000:
        raise ValueError("solver.maxiters is outside its declared bound")
    if source["claim_scope"] != AUTHORITY["claim_scope"]:
        raise ValueError("source claim scope is not the bounded simulated profile")
    return source


def request_from_source(source: dict) -> dict:
    return {"schema": REQUEST_SCHEMA, "operation_id": OPERATION,
            "model": deepcopy(source["model"]), "initial_state": deepcopy(source["initial_state"]),
            "time_s": deepcopy(source["time_s"]), "solver": deepcopy(source["solver"])}


def analytic_oracle(source: dict) -> dict:
    omega = float(source["model"]["omega_0_rad_s"])
    gamma = float(source["model"]["gamma_s_inv"])
    mass = float(source["model"]["mass_kg"])
    q0 = float(source["initial_state"]["q0_m"])
    v0 = float(source["initial_state"]["v0_m_s"])
    t = np.asarray(source["time_s"], dtype=np.float64)
    if gamma == 0.0:
        q = q0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        v = -q0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    else:
        wd = math.sqrt(omega * omega - gamma * gamma)
        b = (v0 + gamma * q0) / wd
        env, c, s = np.exp(-gamma * t), np.cos(wd * t), np.sin(wd * t)
        q = env * (q0 * c + b * s)
        v = env * ((b * wd - gamma * q0) * c + (-q0 * wd - gamma * b) * s)
    energy = 0.5 * mass * (v * v + omega * omega * q * q)
    return {"time_s": t.tolist(), "q_m": q.tolist(), "v_m_s": v.tolist(), "energy_j": energy.tolist()}


def _compare(source, output):
    oracle = analytic_oracle(source)
    _keys(output, {"schema", "operation_id", "request_id", "time_s", "q_m", "v_m_s", "energy_j", "solver"})
    if output["schema"] != "ciw.julia-oscillator-result.v1" or output["operation_id"] != OPERATION:
        raise ValueError("Julia output schema or operation differs")
    arrays = {}
    for key in ("time_s", "q_m", "v_m_s", "energy_j"):
        values = np.asarray(output[key], dtype=np.float64)
        reference = np.asarray(oracle[key], dtype=np.float64)
        if values.shape != reference.shape or not np.all(np.isfinite(values)):
            raise ValueError("Julia output arrays do not match the requested finite grid")
        error = np.abs(values - reference)
        scale = np.maximum(np.abs(reference), 1.0)
        arrays[key] = {"max_abs": float(error.max()), "max_rel": float((error / scale).max())}
        if not np.allclose(values, reference, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
            raise ValueError(f"Julia trajectory failed the analytic oracle for {key}")
    solver = output["solver"]
    _keys(solver, {"algorithm", "retcode", "abstol", "reltol", "accepted_steps", "rejected_steps"})
    if solver["algorithm"] != "Tsit5" or solver["retcode"] != "Success" or solver["accepted_steps"] < 1:
        raise ValueError("Julia solver did not report successful Tsit5 completion")
    return {"status": "passed", "absolute_tolerance": ORACLE_ATOL,
            "relative_tolerance": ORACLE_RTOL, "components": arrays}


def _frame(value: bytes) -> bytes:
    if len(value) > FRAME_LIMIT:
        raise ValueError("worker frame exceeds the bounded protocol limit")
    return struct.pack(">I", len(value)) + value


def _frames(value: bytes):
    frames, offset = [], 0
    while offset < len(value):
        if len(value) - offset < 4:
            raise ValueError("truncated worker frame header")
        length = struct.unpack(">I", value[offset:offset + 4])[0]
        offset += 4
        if length > FRAME_LIMIT or len(value) - offset < length:
            raise ValueError("truncated or oversized worker frame")
        frames.append(value[offset:offset + length])
        offset += length
    if not frames:
        raise ValueError("worker returned no framed messages")
    return frames


def _invoke(repositories, request, execution_id):
    executable = Path(repositories[ROLE]).resolve()
    root = Path(repositories[RUNTIME_ROLE]).resolve()
    worker, _, _ = _runtime_files(root)
    handshake = {"schema": "ciw.julia-worker-handshake-request.v1", "request_id": "handshake-" + execution_id,
                 "operation_id": OPERATION}
    solve = {**request, "request_id": execution_id}
    request_bytes = canonical(solve)
    stdin = _frame(canonical(handshake)) + _frame(request_bytes)
    code, stdout = _bounded_process([str(executable), "--project=" + str(worker.parent), "--startup-file=no",
                                     "--threads=1", str(worker)], cwd=root, timeout=120.0,
                                    limit=FRAME_LIMIT * 2, stdin=stdin)
    if code:
        raise ValueError(f"Julia worker exited with status {code}")
    messages = [_json(frame) for frame in _frames(stdout)]
    if len(messages) != 2:
        raise ValueError("Julia worker must return exactly handshake and result frames")
    identity, response = messages
    _keys(identity, {"schema", "status", "request_id", "operation_id", "identity"})
    if (identity["schema"] != "ciw.julia-worker-handshake-response.v1" or
            identity["status"] != "ok" or identity["request_id"] != handshake["request_id"] or
            identity["operation_id"] != OPERATION):
        raise ValueError("Julia worker handshake was not accepted")
    worker_identity = identity["identity"]
    _keys(worker_identity, {"schema", "profile", "operation_id", "julia_version", "platform",
                            "threads", "startup_file", "worker_sha256", "project_sha256", "manifest_sha256"})
    if (worker_identity["schema"] != "ciw.julia-worker-identity.v1" or
            worker_identity["profile"] != "ordinarydiffeqtsit5" or
            worker_identity["operation_id"] != OPERATION or worker_identity["threads"] != 1 or
            worker_identity["startup_file"] != "disabled"):
        raise ValueError("Julia worker identity was not accepted")
    _keys(response, {"schema", "status", "request_id", "operation_id"}, {"data", "refusal"})
    if response["schema"] != RESPONSE_SCHEMA or response["request_id"] != execution_id:
        raise ValueError("Julia worker response did not bind the request")
    if response["status"] == "refused":
        refusal = response.get("refusal", {})
        raise ValueError("Julia worker refused request: " + str(refusal.get("message", "unknown refusal")))
    if response["status"] != "ok" or "data" not in response:
        raise ValueError("Julia worker returned an unsupported response status")
    return request_bytes, canonical(response), response["data"], worker_identity


def _step(source, evidence_id, repositories, runtime, execution_id=None):
    occurrence = execution_id or "execution-" + uuid.uuid4().hex
    request = request_from_source(source)
    request_bytes, response_bytes, output, worker_identity = _invoke(repositories, request, occurrence)
    for key in ("profile", "threads", "startup_file", "worker_sha256", "project_sha256", "manifest_sha256"):
        if worker_identity.get(key) != runtime.get(key):
            raise ValueError("Julia worker identity differs from the pinned runtime")
    comparison = _compare(source, output)
    data = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "request": request,
            "output": output, "oracle_comparison": comparison, "oracle": analytic_oracle(source),
            "authority": deepcopy(AUTHORITY), "worker_identity": worker_identity}
    result = {"schema": RESULT_SCHEMA, "operation_id": OPERATION, "execution_ref": occurrence,
              "input_refs": [evidence_id], "data": data, "request_bytes_sha256": byte_digest(request_bytes),
              "response_bytes_sha256": byte_digest(response_bytes), "authority": deepcopy(AUTHORITY)}
    result["result_id"] = digest(result)
    numerical = {"operation_id": OPERATION, "data": data}
    return {"runtime_ref": ROLE, "operation_id": OPERATION, "execution_id": occurrence,
            "input_refs": [evidence_id], "request": request, "request_sha256": digest(request),
            "request_bytes_b64": base64.b64encode(request_bytes).decode(),
            "response_bytes_b64": base64.b64encode(response_bytes).decode(),
            "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
            "numerical_result": numerical, "numerical_result_id": digest(numerical),
            "runtime_identity": deepcopy(runtime)}


def _validate_step(step, source, evidence_id, runtime):
    _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                 "request_bytes_b64", "response_bytes_b64", "result", "result_sha256", "result_id",
                 "numerical_result", "numerical_result_id", "runtime_identity"})
    if step["runtime_ref"] != ROLE or step["operation_id"] != OPERATION or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]):
        raise ValueError("Julia occurrence identity differs")
    request = request_from_source(source)
    if step["request"] != request or step["input_refs"] != [evidence_id] or step["request_sha256"] != digest(request):
        raise ValueError("Julia request binding differs")
    try:
        request_bytes = base64.b64decode(step["request_bytes_b64"], validate=True)
        response_bytes = base64.b64decode(step["response_bytes_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Julia raw protocol bytes are not canonical base64") from exc
    if canonical(_json(request_bytes)) != request_bytes or byte_digest(request_bytes) != step["result"]["request_bytes_sha256"]:
        raise ValueError("Julia request bytes are not retained exactly")
    response = _json(response_bytes)
    _keys(response, {"schema", "status", "request_id", "operation_id", "data"})
    if response["request_id"] != step["execution_id"] or response["operation_id"] != OPERATION or response["status"] != "ok":
        raise ValueError("Julia response binding differs")
    if byte_digest(response_bytes) != step["result"]["response_bytes_sha256"]:
        raise ValueError("Julia response bytes are not retained exactly")
    if step["runtime_identity"] != runtime:
        raise ValueError("Julia runtime identity differs")
    expected_comparison = _compare(source, response["data"])
    data = step["result"]["data"]
    if data["output"] != response["data"] or data["oracle_comparison"] != expected_comparison:
        raise ValueError("Julia result data differs from the retained response")
    result = step["result"]
    unsigned = {key: result[key] for key in ("schema", "operation_id", "execution_ref", "input_refs", "data",
                                              "request_bytes_sha256", "response_bytes_sha256", "authority")}
    if result["result_id"] != digest(unsigned) or step["result_id"] != result["result_id"] or step["result_sha256"] != digest(result):
        raise ValueError("Julia result identity differs")
    numerical = {"operation_id": OPERATION, "data": data}
    if step["numerical_result"] != numerical or step["numerical_result_id"] != digest(numerical):
        raise ValueError("Julia numerical result identity differs")


def _verification(bundle, reproduction):
    if bundle["steps"][0]["numerical_result"] != reproduction["numerical_result"]:
        raise ValueError("Julia replay numerical result differs")
    value = {"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
             "independent": True, "method": "python_analytic_oracle_componentwise",
             "runtime_digest": digest(bundle["runtimes"]), "reproduction": deepcopy(reproduction),
             "authority": deepcopy(AUTHORITY)}
    value["verification_id"] = byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(value))
    return value


class JuliaOscillatorWorkflow:
    MAX_BYTES = MAX_BYTES
    kind, role, ROLES = KIND, ROLE, ROLES
    SOURCE_SCHEMA, schema, operation = SOURCE_SCHEMA, SCHEMA, OPERATION

    def _source(self, raw):
        return validate_source(raw)

    def _adapters(self, repositories, expected=None):
        return _adapters(repositories, expected)

    @staticmethod
    def _runtime_projection(value):
        return _runtime_projection(value)

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if bundle["schema"] != SCHEMA or bundle["bundle_digest"] != _bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"]):
                raise ValueError("Julia bundle identity differs")
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            expected_source = {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                               "evidence": [{"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}]}
            if bundle["source"] != expected_source or len(bundle["steps"]) != 1:
                raise ValueError("Julia source evidence differs")
            _keys(bundle["runtimes"], {RUNTIME_ROLE})
            runtime = bundle["runtimes"][RUNTIME_ROLE]
            _keys(runtime, {"schema", "role", "profile", "execution_scope", "julia_version", "platform", "threads", "startup_file", "runtime_revision", "worker_sha256", "project_sha256", "manifest_sha256", "executable_sha256", "worker_relative_path"})
            _validate_step(bundle["steps"][0], source, evidence["artifact_ref"], runtime)
            verification = bundle["verification"]
            _keys(verification, {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest", "reproduction", "authority", "verification_id"})
            _validate_step(verification["reproduction"], source, evidence["artifact_ref"], runtime)
            if verification != _verification(bundle, verification["reproduction"]):
                raise ValueError("Julia verification differs")
            _identity(verification, "verification_id")
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError("At most one Julia replay receipt is retained")
            for receipt in receipts:
                _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
                if receipt["schema"] != "ciw." + KIND + "-replay.v1" or receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or receipt["source_bundle_digest"] == bundle["bundle_digest"] or receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or receipt["replay_id"] != digest({key: value for key, value in receipt.items() if key != "replay_id"}):
                    raise ValueError("Invalid Julia replay receipt")
                _identity(receipt["verification"], "verification_id")
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed retained Julia oscillator session") from exc
        return raw

    def _execute(self, raw, repositories, runtime):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = _step(source, evidence, repositories, runtime)
        bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex, "created_at": _now(),
                  "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(AUTHORITY), "runtimes": {RUNTIME_ROLE: deepcopy(runtime)}, "steps": [step]}
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = _verification(bundle, _step(source, evidence, repositories, runtime))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        source = self._source(raw)
        bound, runtime = self._adapters(repositories)
        return self._execute(canonical(source), bound, runtime)

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        bound, runtime = self._adapters(repositories, bundle["runtimes"])
        fresh = self._execute(raw, bound, runtime)
        receipt = {"schema": "ciw." + KIND + "-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": _verification(bundle, fresh["steps"][0]), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}


workflow = JuliaOscillatorWorkflow()


def to_run(result_data: dict) -> dict:
    """Project a retained Julia trajectory into the existing read-only run view."""
    output = result_data["output"]
    times = output["time_s"]
    if len(times) < 2:
        raise ValueError("trajectory requires at least two samples")
    dt = times[1] - times[0]
    run = {"run_id": "run-julia-oscillator-" + digest(result_data)[7:23],
           "evidence_id": digest(result_data), "instrument": "julia-tsit5-oscillator.v1",
           "metadata": {"duration_s": float(times[-1] + dt), "sample_rate_hz": float(1.0 / dt),
                        "sample_count": len(times), "coordinate_frame": "oscillator-state",
                        "model": result_data["request"]["model"], "provenance": {
                            "source": "Julia OrdinaryDiffEqTsit5 trajectory; simulated evidence",
                            "generator": OPERATION, "dtype": "float64", "sampling": "uniform; endpoint excluded"}},
           "time_s": times, "channels": {"q": {"unit": "m", "values": output["q_m"]},
                                           "v": {"unit": "m/s", "values": output["v_m_s"]},
                                           "energy": {"unit": "J", "values": output["energy_j"]}}}
    # The standard viewport only needs a declared run; its surface is optional
    # for this provider and must never be reconstructed from rendered pixels.
    run["render"] = {"coordinate_frame": "oscillator-state", "axis_labels": ["q (m)", "energy (J)", "v (m/s)"],
                      "trajectory": [[q, e, v] for q, e, v in zip(output["q_m"], output["energy_j"], output["v_m_s"])],
                      "sample_indices": list(range(len(times))), "surface": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "indices": [0, 1, 2]},
                      "transform": {"origin": [0, 0, 0], "scale": [1, 1, 1], "note": "Derived display scaling only."}}
    validate_run(run)
    return run
