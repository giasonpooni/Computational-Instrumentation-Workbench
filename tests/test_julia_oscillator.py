"""Structural, protocol and offline boundaries of the Julia oscillator operation.

These tests never run Julia. Worker replies come from an explicitly labelled
Python double that copies the analytic reference; they exercise persistence,
identities, refusals and projections, not numerical integration. The real
Julia acceptance gate is tests/test_julia_oscillator_session.py.
"""
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

from ciw import julia_oscillator as module
from ciw import julia_worker
from ciw.adapters.oscillator import analytic_trajectory, make_demo_run
from ciw.adapters.protocol import AdapterRefusal
from ciw.adapters.registry import default_registry
from ciw.declared_workload import _commit
from ciw.julia_oscillator import (JuliaOscillatorWorkflow, OPERATION, PROGRAM, UNITS, decode_output, encode_configuration,
                                  encode_input, encode_request, inspect_bundle, measure_errors, oracle_reference,
                                  project_run, sample_times, split_response, state_trajectory_view)
from ciw.julia_worker import JuliaWorkerSession, JuliaWorkerPool, check_runtime, runtime_pin
from ciw.instruments import make_demo_run as demo_run
from ciw.session import Session
from ciw.state_trajectory import projection
from ciw.telemetry import canonical, digest, byte_digest, _bundle_digest
from ciw.workbench import Workbench

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/julia-oscillator"
FAKE = Path(__file__).resolve().parent / "fake_julia_worker.py"


def source(name="default"):
    return json.loads((EXAMPLES / (name + ".json")).read_bytes())


def call(session, kind, payload=None, *, error=False):
    reply = session.handle({"protocol_version": 1, "request_id": "julia-test", "type": kind, "payload": payload or {}})
    assert reply["type"] == ("error" if error else "response"), reply
    return reply["payload"]


def fake_runtime():
    """Synthetic runtime identity consistent with the packaged pin; offline only."""
    pin = runtime_pin()
    return {"schema": "ciw.julia-worker-runtime.v1", "adapter_version": "ciw-julia-worker-v1", "protocol_version": 1,
            "julia_executable": "/unit-test/julia", "environment_root": "/unit-test/env", "julia_sha256": "a" * 64,
            "julia_version": pin["julia_version"], "platform": {"machine": "unit-test", "arch": "x86_64", "kernel": "Test", "word_size": 64},
            "threads": 1, "options": {"startup_file": "disabled", "fast_math": "default", "check_bounds": 0, "opt_level": 2},
            "worker_source_sha256": pin["worker"]["sha256"], "project_sha256": pin["project"]["sha256"],
            "manifest_sha256": pin["manifest"]["sha256"], "sysimage": {"name": "sys.so", "sha256": "b" * 64},
            "packages": deepcopy(pin["packages"]), "operations": list(pin["operations"]),
            "solver": {"algorithm": "Tsit5", "package": "OrdinaryDiffEqTsit5",
                       "version": pin["packages"]["OrdinaryDiffEqTsit5"]["version"], "arithmetic": "binary64"}}


class FakeWorker:
    """In-process double for JuliaWorkerSession: analytic values, no ODE solve."""

    def __init__(self, identity=None, perturb=0.0):
        self.identity = identity or fake_runtime()
        self.session_id = "julia-worker-" + "f" * 32
        self.occurrence = 0
        self.perturb = perturb
        self.alive = True
        self.failed = []

    def execute(self, build, *, timeout_seconds=None):
        self.occurrence += 1
        number = self.occurrence
        request = build(number)
        offset = 16
        length, = struct.unpack_from("<I", request, offset)
        offset += 4 + length
        length, = struct.unpack_from("<I", request, offset)
        offset += 4 + length
        length, = struct.unpack_from("<I", request, offset)
        payload = request[offset + 4:offset + 4 + length]
        omega_0, gamma, mass, q0, v0 = struct.unpack_from("<ddddd", payload, 8)
        count, = struct.unpack_from("<I", payload, 48)
        times = np.frombuffer(payload, dtype="<f8", count=count, offset=52)
        q, v, _ = analytic_trajectory(omega_0, gamma, mass, q0, v0, times)
        q = q + self.perturb
        energy = 0.5 * mass * (v * v + omega_0 * omega_0 * q * q)
        body = struct.pack("<I", 0) + struct.pack("<I", 7) + b"Success" + struct.pack("<I", count)
        body += times.astype("<f8").tobytes() + q.astype("<f8").tobytes() + v.astype("<f8").tobytes() + energy.astype("<f8").tobytes()
        body += struct.pack("<QQQ", 600, 0, 3600)
        response = struct.pack("<4sIQ", b"OSCO", 1, number) + body
        return number, response, {"schema": "ciw.julia-worker-occurrence.v1", "request_number": number,
                                  "worker_occurrence": number, "elapsed_ns": 12345}

    def fail(self, code, message):
        self.alive = False
        self.failed.append((code, message))


@pytest.fixture
def fake_workflow(monkeypatch):
    worker = FakeWorker()
    monkeypatch.setattr(JuliaOscillatorWorkflow, "_adapters", lambda self, repositories, expected=None: (worker, deepcopy(worker.identity)))
    return worker


def fake_bundle(name="default", worker=None):
    worker = worker or FakeWorker()
    workflow = JuliaOscillatorWorkflow()
    raw = canonical(source(name))
    return workflow._execute(raw, (worker, deepcopy(worker.identity)))


# --- source contract ---------------------------------------------------------

@pytest.mark.parametrize("path,value", [
    ("model.omega_0_rad_s", 25.0), ("model.omega_0_rad_s", 0.0), ("model.omega_0_rad_s", True), ("model.gamma_s_inv", 3.0),
    ("model.gamma_s_inv", -0.1), ("model.mass_kg", 0.0), ("model.mass_kg", 101.0), ("model.initial_q_m", 10.5),
    ("model.initial_v_m_s", -100.5), ("model.initial_q_m", "1"), ("grid.sample_count", 1), ("grid.sample_count", 4097),
    ("grid.sample_count", 768.0), ("grid.sample_count", True), ("grid.sample_rate_hz", 0.0), ("grid.sample_rate_hz", -64.0),
    ("grid.sample_rate_hz", 60.0), ("grid.start_s", 1.0), ("grid.endpoint", "included"),
    ("configuration.solver.algorithm", "Vern7"), ("configuration.solver.abstol", 1e-15), ("configuration.solver.reltol", 0.1),
    ("configuration.solver.maxiters", 10), ("configuration.solver.maxiters", 1000.0), ("configuration.solver.dt_initial", 1e-12),
    ("configuration.solver.dtmax", 13.0), ("configuration.solver.output_policy", "dense"),
    ("configuration.origin", "observation"), ("configuration.covariance_status", "declared"),
    ("configuration.oracle_policy.reference", "julia-self-check"), ("configuration.oracle_policy.max_normalized_error.q", 0.0),
    ("configuration.oracle_policy.max_normalized_error.energy", 2.0), ("configuration.replay_policy.max_normalized_difference", -1.0),
    ("schema", "ciw.julia-oscillator-source.v2"), ("experiment_id", ""), ("julia_executable", "/client/selected/julia"),
])
def test_source_refuses_undeclared_or_out_of_bound_values(path, value):
    declaration = source()
    node = declaration
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    with pytest.raises(ValueError):
        JuliaOscillatorWorkflow()._source(canonical(declaration))


@pytest.mark.parametrize("raw", [b'{"schema": "ciw.julia-oscillator-source.v1", "model": {"omega_0_rad_s": NaN}}',
                                 b'{"schema": "ciw.julia-oscillator-source.v1", "model": {"omega_0_rad_s": Infinity}}',
                                 b'{"schema":1,"schema":2}', b"", b"\xff"])
def test_source_refuses_nonfinite_or_malformed_json(raw):
    with pytest.raises((ValueError, AdapterRefusal)):
        JuliaOscillatorWorkflow()._source(raw)


def test_example_sources_are_valid_and_declare_simulation():
    for path in sorted(EXAMPLES.glob("*.json")):
        declaration = JuliaOscillatorWorkflow()._source(path.read_bytes())
        assert declaration["configuration"]["origin"] == "simulation"
        assert declaration["grid"]["sample_count"] == 768
    default = source()
    assert default["model"]["omega_0_rad_s"] == 2.0 * math.pi * 0.8
    assert sample_times(default["grid"]).tolist() == make_demo_run()["time_s"]


# --- byte contract -------------------------------------------------------------

def test_specification_identity_binds_solver_configuration():
    declaration = source()
    times = sample_times(declaration["grid"])
    configuration = encode_configuration(declaration["configuration"]["solver"])
    payload = encode_input(declaration["model"], times)
    tighter = encode_configuration({**declaration["configuration"]["solver"], "reltol": 1e-12})
    assert configuration != tighter
    assert _commit("specification", [PROGRAM, configuration, payload]) != _commit("specification", [PROGRAM, tighter, payload])
    assert _commit("input", [payload]) == _commit("input", [payload])
    assert configuration.startswith(b"OSCC") and payload.startswith(b"OSCI")
    assert struct.unpack_from("<I", payload, 48)[0] == 768
    assert np.frombuffer(payload, dtype="<f8", count=768, offset=52).tolist() == times.tolist()
    request = encode_request(7, configuration, payload)
    assert request.startswith(b"CIWQ") and struct.unpack_from("<Q", request, 8)[0] == 7
    assert b"ciw.julia-oscillator.integrate.v1" in request and configuration in request and payload in request


def test_decode_output_is_strict():
    number, response, _ = FakeWorker().execute(lambda n: encode_request(n, encode_configuration(source()["configuration"]["solver"]),
                                                                         encode_input(source()["model"], sample_times(source()["grid"]))))
    bound, output = split_response(response)
    assert bound == number == 1
    decoded = decode_output(output)
    assert decoded["status"] == 0 and decoded["return_code"] == "Success" and len(decoded["q"]) == 768
    assert decoded["solver"] == {"accepted_steps": 600, "rejected_steps": 0, "function_evaluations": 3600}
    refusal = decode_output(struct.pack("<II", 4, 5) + b"bound")
    assert refusal == {"status": 4, "message": "bound"}
    for bad in (output[:-1], output + b"\0", b"OSCO" + output, struct.pack("<II", 0, 3) + b"\xff\xff\xff" + output[15:],
                output[:8] + struct.pack("<I", 1) + output[12:], struct.pack("<II", 4, 600) + b"x" * 600):
        with pytest.raises(ValueError):
            decode_output(bad)
    nonfinite = bytearray(output)
    struct.pack_into("<d", nonfinite, 4 + 4 + 7 + 4 + 8 * 768 + 8, float("nan"))
    with pytest.raises(ValueError):
        decode_output(bytes(nonfinite))
    with pytest.raises(ValueError):
        split_response(b"XXXX" + response[4:])


# --- oracle --------------------------------------------------------------------

def test_analytic_trajectory_is_the_demo_recording_bitwise():
    run = make_demo_run()
    model = run["metadata"]["model"]
    q, v, energy = analytic_trajectory(model["omega_0_rad_s"], model["gamma_s_inv"], model["mass_kg"],
                                       model["initial_q_m"], model["initial_v_m_s"], np.asarray(run["time_s"]))
    assert q.tolist() == run["channels"]["q"]["values"]
    assert v.tolist() == run["channels"]["v"]["values"]
    assert energy.tolist() == run["channels"]["energy"]["values"]
    assert run["evidence_id"] == "sha256:824a01978910ebcf738dffb75370c931e7fffc0eef6ee97bc11ec3223f1323c2"


def test_measured_errors_are_exact_and_thresholds_decide_outcome():
    declaration = source("undamped")
    times = sample_times(declaration["grid"])
    reference = oracle_reference(declaration["model"], times)
    decoded = {name: list(reference[name]) for name in UNITS}
    decoded["q"][10] += 0.5
    expected = abs(decoded["q"][10] - reference["q"][10])
    measured = measure_errors(decoded, reference, declaration["model"])
    assert measured["q"]["max_abs_error"] == expected and measured["q"]["at_sample_index"] == 10
    assert measured["q"]["normalized_max_error"] == expected / measured["q"]["reference_scale"]
    assert measured["v"]["max_abs_error"] == 0.0
    assert measured["numerical_energy_conservation"]["status"] == "evaluated"
    assert module._outcome(measured, {"q": 1e-3, "v": 1e-3, "energy": 1e-3}) == "failed"
    assert module._outcome(measured, {"q": 1.0, "v": 1e-3, "energy": 1e-3}) == "passed"
    damped = measure_errors(decoded, reference, {**declaration["model"], "gamma_s_inv": 0.1})
    assert damped["numerical_energy_conservation"] == {"status": "not_applicable_damped"}


# --- retained bundle --------------------------------------------------------------

def test_fake_bundle_retains_scr_specification_bytes_and_identities():
    bundle = fake_bundle()
    step, = bundle["steps"]
    data = step["result"]["data"]
    native, occurrence = data["native"], data["occurrence"]
    configuration = bytes.fromhex(native["specification"]["configuration"])
    payload = bytes.fromhex(native["specification"]["input_payload"])
    output = bytes.fromhex(native["output"])
    assert bytes.fromhex(native["specification"]["program"]) == PROGRAM
    assert native["specification_identity"] == _commit("specification", [PROGRAM, configuration, payload])
    assert native["computation_identity"] == _commit("computation", [bytes.fromhex(native["program_identity"]), bytes.fromhex(native["input_identity"]), bytes.fromhex(native["output_identity"]), struct.pack("<I", 0)])
    assert bytes.fromhex(occurrence["request"]) == encode_request(1, configuration, payload)
    assert bytes.fromhex(occurrence["response"]) == struct.pack("<4sIQ", b"OSCO", 1, 1) + output
    assert occurrence["worker_session_id"].startswith("julia-worker-") and occurrence["engine_occurrence"] == 1
    assert step["numerical_result"] == {"operation_id": OPERATION, "data": native}
    assert bundle["verification"]["outcome"] == "passed" and bundle["verification"]["independent"] is False
    assert bundle["verification"]["measured"]["q"]["max_abs_error"] == 0.0
    assert bundle["verification"]["oracle"]["reference_digest"] == digest({k: bundle["verification"]["oracle"][k] for k in UNITS})
    assert len({step["execution_id"], step["result_id"], step["numerical_result_id"], bundle["verification"]["verification_id"],
                bundle["bundle_digest"], step["input_refs"][0]}) == 6
    summary = inspect_bundle(bundle)
    assert summary["oracle_outcome"] == "passed" and summary["physical_validation"] == "not_established"


def test_failed_oracle_outcome_is_retained_not_fabricated():
    bundle = fake_bundle("tight-tolerance", FakeWorker(perturb=1e-6))
    assert bundle["verification"]["outcome"] == "failed"
    assert bundle["verification"]["measured"]["q"]["max_abs_error"] == pytest.approx(1e-6)
    JuliaOscillatorWorkflow()._validate(bundle)


@pytest.mark.parametrize("fault", ["output-byte", "decoded-value", "oracle-value", "measured", "outcome", "threshold",
                                   "runtime-version", "occurrence-number", "response", "request", "authority",
                                   "numerical", "worker-session", "verification-subject", "metadata", "pin-worker"])
def test_retained_faults_are_refused_offline(fault):
    bundle = fake_bundle()
    step = bundle["steps"][0]
    data = step["result"]["data"]
    if fault == "output-byte":
        raw = bytearray(bytes.fromhex(data["native"]["output"]))
        raw[-1] ^= 1
        data["native"]["output"] = bytes(raw).hex()
    if fault == "decoded-value":
        data["native"]["decoded"]["q"][5] += 1e-9
    if fault == "oracle-value":
        bundle["verification"]["oracle"]["v"][5] += 1e-9
    if fault == "measured":
        bundle["verification"]["measured"]["q"]["max_abs_error"] = 1e-9
    if fault == "outcome":
        bundle["verification"]["outcome"] = "failed"
    if fault == "threshold":
        bundle["verification"]["thresholds"]["q"] = 1e-3
    if fault == "runtime-version":
        bundle["runtimes"]["julia"]["julia_version"] = "1.11.0"
    if fault == "occurrence-number":
        data["occurrence"]["engine_occurrence"] = 2
    if fault == "response":
        raw = bytearray(bytes.fromhex(data["occurrence"]["response"]))
        raw[12] ^= 1
        data["occurrence"]["response"] = bytes(raw).hex()
    if fault == "request":
        data["occurrence"]["request"] = encode_request(1, b"OSCC", b"OSCI").hex()
    if fault == "authority":
        step["result"]["authority"]["physical_truth"] = "established"
    if fault == "numerical":
        step["numerical_result"]["data"]["exit_code"] = 1
    if fault == "worker-session":
        data["occurrence"]["worker_session_id"] = "session-" + "0" * 32
    if fault == "verification-subject":
        bundle["verification"]["subject_ref"] = "sha256:" + "0" * 64
    if fault == "metadata":
        data["occurrence"]["worker_metadata"]["request_number"] = 9
    if fault == "pin-worker":
        bundle["runtimes"]["julia"]["worker_source_sha256"] = "c" * 64
    if fault not in {"verification-subject", "outcome", "measured", "threshold", "oracle-value"}:
        bundle["bundle_digest"] = _bundle_digest(bundle)
    with pytest.raises(ValueError):
        JuliaOscillatorWorkflow()._validate(bundle)


def test_shared_session_flow_offline_restore_and_replay_binding(tmp_path, fake_workflow, monkeypatch):
    session = Session(demo_run(), tmp_path)
    operation = next(o for o in call(session, "operation.list")["operations"] if o["operation_id"] == OPERATION)
    assert operation["role"] == "numerical_simulation" and operation["available"] is False
    raw = b"\n" + canonical(source()) + b"\n"
    retained = call(session, "source.add", {"kind": "julia-oscillator", "label": "default", "bytes_b64": base64.b64encode(raw).decode()})
    call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": retained["source_id"]}}, error=True)
    session.workbench.bind_workflow("julia-oscillator", {"julia": tmp_path / "julia"})
    summary = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": retained["source_id"]}})
    assert summary["retained_verification_outcome"] == "passed" and summary["operation_id"] == OPERATION
    bundle = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    assert base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"]) == raw
    view = call(session, "experiment.inspect", {"bundle_id": summary["bundle_id"]})
    assert [p["panel_id"] for p in view["panels"]] == ["initial-state", "final-state", "oracle-error", "solver-work"]
    assert view["object_context"]["origin"] == "simulation" and view["fusion_context"] is None
    assert view["state_trajectories"]["simulation"]["origin"] == "simulation"
    assert view["state_trajectories"]["reference"]["origin"] == "reference"
    assert view["state_trajectories"]["simulation"]["state_axes"][0]["values"] == bundle["steps"][0]["result"]["data"]["native"]["decoded"]["q"]
    assert "q" not in view["verification"]["oracle"]
    replay = call(session, "bundle.replay", {"bundle_id": summary["bundle_id"]})
    receipt = replay["replay_receipt"]
    assert receipt["verification"]["byte_identical"] is True and receipt["numerical_match"] is True
    fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    assert fresh["steps"][0]["execution_id"] != bundle["steps"][0]["execution_id"]
    assert fresh["steps"][0]["numerical_result_id"] == bundle["steps"][0]["numerical_result_id"]
    assert fresh["steps"][0]["result"]["data"]["occurrence"]["engine_occurrence"] == 2
    assert {row["instrument"] for row in call(session, "instrument.list")["instruments"]} == {"julia"}
    assert call(session, "fusion.list")["contexts"] == []
    path = session.save_workspace(tmp_path / "workspace.json")
    before = session.workbench.serialize()

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not start or bind a Julia worker")

    monkeypatch.setattr(JuliaWorkerSession, "__init__", forbidden)
    monkeypatch.setattr(JuliaOscillatorWorkflow, "_adapters", forbidden)
    restored = Session.from_workspace(path, tmp_path / "restored")
    assert restored.workbench.serialize() == before
    assert not next(o for o in restored.workbench.describe_operations() if o["operation_id"] == OPERATION)["available"]
    restored_view = call(restored, "experiment.inspect", {"bundle_id": summary["bundle_id"]})
    assert restored_view["panels"] == view["panels"]
    call(restored, "bundle.replay", {"bundle_id": summary["bundle_id"]}, error=True)
    assert restored.workbench.pending_operations == 0


def test_replay_receipt_faults_are_refused(fake_workflow, tmp_path):
    session = Session(demo_run(), tmp_path)
    session.workbench.bind_workflow("julia-oscillator", {"julia": tmp_path / "julia"})
    retained = call(session, "source.add", {"kind": "julia-oscillator", "label": "default", "bytes_b64": base64.b64encode(canonical(source())).decode()})
    original = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": retained["source_id"]}})
    call(session, "bundle.replay", {"bundle_id": original["bundle_id"]})
    saved = session.workbench.serialize()
    fresh = next(r for r in saved["bundles"] if r["native"].get("replay_receipts"))
    for fault in ("byte_identical", "agreement", "source"):
        tampered = deepcopy(saved)
        record = next(r for r in tampered["bundles"] if r["bundle_id"] == fresh["bundle_id"])
        receipt = record["native"]["replay_receipts"][0]
        if fault == "byte_identical":
            receipt["verification"]["byte_identical"] = False
        if fault == "agreement":
            receipt["verification"]["agreement"]["q"]["normalized_max_difference"] = 1e-3
        if fault == "source":
            receipt["source_bundle_digest"] = "sha256:" + "1" * 64
        with pytest.raises(ValueError):
            Workbench.restore(tampered)
    replay_disagrees = deepcopy(saved)
    record = next(r for r in replay_disagrees["bundles"] if r["bundle_id"] == fresh["bundle_id"])
    record["native"]["steps"][0]["result"]["data"]["native"]["decoded"]["q"][0] += 1.0
    with pytest.raises(ValueError):
        Workbench.restore(replay_disagrees)


def test_replay_beyond_declared_tolerance_is_refused_without_retention(tmp_path, monkeypatch):
    workers = [FakeWorker(), FakeWorker(perturb=1e-6)]
    monkeypatch.setattr(JuliaOscillatorWorkflow, "_adapters", lambda self, repositories, expected=None: (workers[0], deepcopy(workers[0].identity)))
    session = Session(demo_run(), tmp_path)
    session.workbench.bind_workflow("julia-oscillator", {"julia": tmp_path / "julia"})
    retained = call(session, "source.add", {"kind": "julia-oscillator", "label": "default", "bytes_b64": base64.b64encode(canonical(source())).decode()})
    original = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": retained["source_id"]}})
    monkeypatch.setattr(JuliaOscillatorWorkflow, "_adapters", lambda self, repositories, expected=None: (workers[1], deepcopy(workers[1].identity)))
    reply = call(session, "bundle.replay", {"bundle_id": original["bundle_id"]}, error=True)
    assert "replay tolerance" in reply["message"]
    assert len(call(session, "bundle.list")["bundles"]) == 1


def test_runtime_identity_mismatch_refuses_replay_before_execution(tmp_path, monkeypatch):
    bundle = fake_bundle("default")
    other = FakeWorker(identity={**fake_runtime(), "julia_sha256": "d" * 64})
    monkeypatch.setattr(module.POOL, "session", lambda executable, **options: other)
    with pytest.raises(ValueError, match="runtime differs"):
        JuliaOscillatorWorkflow().replay_session(bundle, {"julia": tmp_path / "julia"})
    assert other.occurrence == 0


# --- projections -----------------------------------------------------------------

def test_run_projection_is_derived_and_drives_the_legacy_session(tmp_path):
    bundle = fake_bundle("mixed-initial-state")
    run = project_run(bundle)
    default_registry().validate_run(run)
    assert run["instrument"] == "julia-oscillator-trajectory.v1" and run["metadata"]["provenance"]["origin"] == "simulation"
    assert run["metadata"]["provenance"]["result_id"] == bundle["steps"][0]["result_id"]
    assert run["render"]["sample_indices"] == list(range(768)) and len(run["render"]["trajectory"]) == 768
    assert run["channels"]["q"]["values"][0] == -0.6 and run["channels"]["v"]["values"][0] == 2.5
    session = Session(run, tmp_path)
    sample = call(session, "sample.get", {"time_s": 0.0})
    assert sample["values"] == {"q": -0.6, "v": 2.5, "energy": run["channels"]["energy"]["values"][0]}
    assert call(session, "analysis.stats")["data"]["sample_count"] == 768
    assert call(session, "run.get")["render"]["axis_labels"] == ["q (m)", "energy (J)", "v (m/s)"]
    changed = deepcopy(bundle)
    changed["steps"][0]["result"]["data"]["native"]["decoded"]["q"][1] += 1e-6
    with pytest.raises(ValueError):
        project_run(changed)
    misdeclared = deepcopy(run)
    misdeclared["metadata"]["provenance"]["origin"] = "observation"
    from ciw.core.identities import evidence_id
    misdeclared["evidence_id"] = evidence_id(misdeclared)
    with pytest.raises(ValueError, match="simulation origin"):
        default_registry().validate_run(misdeclared)


def test_state_trajectory_projection_contract():
    bundle = fake_bundle()
    views = state_trajectory_view(bundle)
    for name, view in views.items():
        assert view["schema"] == "ciw.state-trajectory-projection.v1" and view["sample_count"] == 768
        assert [axis["name"] for axis in view["state_axes"]] == ["q", "v"] and view["authority"]["state_estimate"] is False
    assert views["simulation"]["origin"] == "simulation" and views["reference"]["origin"] == "reference"
    good = dict(origin="simulation", label="x", independent_axis={"name": "t", "unit": "s", "values": [0.0, 1.0]},
                state_axes=[{"name": "q", "unit": "m", "values": [1.0, 2.0]}], derived_quantities=[],
                coordinate_basis={}, provenance={"result_id": "r", "execution_id": "e"})
    projection(**good)
    for change in ({"origin": "estimate"}, {"independent_axis": {"name": "t", "unit": "s", "values": [1.0, 0.0]}},
                   {"state_axes": [{"name": "q", "unit": "m", "values": [1.0]}]}, {"provenance": {}},
                   {"derived_quantities": [{"name": "e", "unit": "J", "values": [1.0, 2.0]}]},
                   {"state_axes": [{"name": "q", "unit": "m", "values": [1.0, float("inf")]}]}, {"selected_index": 2}):
        with pytest.raises(ValueError):
            projection(**{**good, **change})


# --- worker protocol doubles -------------------------------------------------------

@pytest.fixture
def fake_process(monkeypatch, tmp_path):
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(runtime_pin()))
    monkeypatch.setenv("CIW_FAKE_JULIA_ROOT", str(julia_worker.environment_root()))
    monkeypatch.setenv("CIW_FAKE_JULIA_PIN", str(pin_path))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def start(mode, **options):
        monkeypatch.setattr(JuliaWorkerSession, "_command", lambda self: [sys.executable, str(FAKE), mode])
        return JuliaWorkerSession(sys.executable, **{"hello_timeout": 20, "timeout_seconds": 5, **options})

    return start


def build_request():
    declaration = source()
    return lambda number: encode_request(number, encode_configuration(declaration["configuration"]["solver"]),
                                         encode_input(declaration["model"], sample_times(declaration["grid"])))


def test_fake_worker_handshake_and_execution_bind_occurrences(fake_process):
    session = fake_process("ok")
    try:
        assert session.identity["julia_version"] == runtime_pin()["julia_version"]
        check_runtime(session.identity)
        assert session.session_id.startswith("julia-worker-")
        number, response, metadata = session.execute(build_request())
        assert number == 1 and metadata["request_number"] == 1
        assert split_response(response)[0] == 1
        number, _, _ = session.execute(build_request())
        assert number == 2 and session.alive
        with pytest.raises(AdapterRefusal, match="1..1048576"):
            session.execute(b"")
    finally:
        session.terminate()
    assert not session.alive


@pytest.mark.parametrize("mode,code", [("hello-wrong-version", "RUNTIME_PIN_MISMATCH"), ("hello-tampered-worker", "RUNTIME_PIN_MISMATCH"),
                                       ("hello-bad-json", "MALFORMED_RESPONSE"), ("hello-oversized", "OUTPUT_LIMIT"),
                                       ("crash-after-hello", None)])
def test_fake_worker_wrong_environment_or_broken_hello_never_binds(fake_process, mode, code):
    if mode == "crash-after-hello":
        session = fake_process(mode)
        with pytest.raises(AdapterRefusal) as failure:
            session.execute(build_request())
        assert failure.value.code in {"WORKER_FAILED", "WORKER_UNAVAILABLE"}
        return
    with pytest.raises(AdapterRefusal) as failure:
        fake_process(mode)
    assert failure.value.code == code


def test_fake_worker_hello_timeout_is_a_refusal(fake_process):
    with pytest.raises(AdapterRefusal) as failure:
        fake_process("hello-silent", hello_timeout=0.5)
    assert failure.value.code == "TIMEOUT"


@pytest.mark.parametrize("mode,code", [("timeout", "TIMEOUT"), ("crash-mid-request", "WORKER_FAILED"),
                                       ("truncated-response", "WORKER_FAILED"), ("oversized-response", "OUTPUT_LIMIT"),
                                       ("wrong-number-metadata", "MALFORMED_RESPONSE"), ("junk-metadata", "MALFORMED_RESPONSE")])
def test_fake_worker_failures_end_the_session_without_a_result(fake_process, mode, code):
    session = fake_process(mode, timeout_seconds=0.5)
    with pytest.raises(AdapterRefusal) as failure:
        session.execute(build_request())
    assert failure.value.code == code
    assert not session.alive
    with pytest.raises(AdapterRefusal) as again:
        session.execute(build_request())
    assert again.value.code == "WORKER_UNAVAILABLE"


@pytest.mark.parametrize("mode,code", [("wrong-number-header", "MALFORMED_RESPONSE"), ("nonfinite", "MALFORMED_RESPONSE"),
                                       ("bad-energy", "INVALID_WORKER_OUTPUT"), ("wrong-coverage", "INVALID_WORKER_OUTPUT"),
                                       ("refuse", "JULIA_OSCILLATOR_REFUSED")])
def test_fake_worker_content_violations_never_become_results(fake_process, mode, code, monkeypatch):
    session = fake_process(mode)
    try:
        workflow = JuliaOscillatorWorkflow()
        monkeypatch.setattr(JuliaOscillatorWorkflow, "_adapters", lambda self, repositories, expected=None: (session, deepcopy(session.identity)))
        with pytest.raises(AdapterRefusal) as failure:
            workflow.create_session(canonical(source()), {"julia": sys.executable})
        assert failure.value.code == code
        assert session.alive == (mode in {"refuse", "bad-energy", "wrong-coverage"})
    finally:
        session.terminate()


def test_pool_restarts_a_dead_session_with_a_new_identity(fake_process, monkeypatch):
    pool = JuliaWorkerPool()
    monkeypatch.setattr(JuliaWorkerSession, "_command", lambda self: [sys.executable, str(FAKE), "ok"])
    first = pool.session(sys.executable, hello_timeout=20, timeout_seconds=5)
    assert pool.session(sys.executable) is first
    first.execute(build_request())
    first.fail("TEST", "ended by the test")
    second = pool.session(sys.executable, hello_timeout=20, timeout_seconds=5)
    try:
        assert second is not first and second.session_id != first.session_id and second.occurrence == 0
        assert second.execute(build_request())[0] == 1
    finally:
        pool.shutdown()
    assert not second.alive


def test_packaged_pin_matches_the_packaged_environment_files():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from pin_julia_runtime import build_pin
    finally:
        sys.path.pop(0)
    pin = runtime_pin()
    assert build_pin(julia_worker.environment_root()) == pin
    assert pin["julia_version"] == "1.10.12"
    assert pin["packages"]["OrdinaryDiffEqTsit5"]["version"] == "2.1.4"
    root = julia_worker.environment_root()
    assert (root / "Manifest.toml").read_text(encoding="utf-8").startswith("# This file is machine-generated")
    julia_worker.verify_environment(root, pin)
    with pytest.raises(AdapterRefusal, match="differ"):
        julia_worker.verify_environment(root, {**pin, "worker": {**pin["worker"], "sha256": "0" * 64}})
