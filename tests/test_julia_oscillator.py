import base64
import copy

import pytest

from ciw import julia_oscillator as jo
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import canonical


def source():
    return {
        "schema": jo.SOURCE_SCHEMA,
        "experiment_id": "julia-oscillator-test-v1",
        "operation_id": jo.OPERATION,
        "model": {"omega_0_rad_s": 2.0, "gamma_s_inv": 0.1, "mass_kg": 1.0},
        "initial_state": {"q0_m": 1.0, "v0_m_s": -0.25},
        "time_s": [0.0, 0.1, 0.2, 0.3],
        "solver": {"abstol": jo.DEFAULT_ABSTOL, "reltol": jo.DEFAULT_RELTOL, "maxiters": 100000},
        "claim_scope": jo.AUTHORITY["claim_scope"],
    }


RUNTIME = {
    "schema": "ciw.julia-oscillator-runtime.v1", "role": jo.ROLE,
    "profile": "ordinarydiffeqtsit5", "execution_scope": "bounded_simulated_oscillator",
    "julia_version": "julia version 1.10.12", "platform": "test", "threads": 1,
    "startup_file": "disabled", "runtime_revision": "a" * 40,
    "worker_sha256": "sha256:" + "1" * 64, "project_sha256": "sha256:" + "2" * 64,
    "manifest_sha256": "sha256:" + "3" * 64, "executable_sha256": "sha256:" + "4" * 64,
    "worker_relative_path": "runtimes/julia-oscillator/oscillator_worker.jl",
}


def fake_invoke(_repositories, request, execution_id):
    solve_request = {**request, "request_id": execution_id}
    src = {**source(), "model": request["model"], "initial_state": request["initial_state"],
           "time_s": request["time_s"], "solver": request["solver"]}
    output = jo.analytic_oracle(src)
    output.update({"schema": "ciw.julia-oscillator-result.v1", "operation_id": jo.OPERATION,
                   "request_id": execution_id,
                   "solver": {"algorithm": "Tsit5", "retcode": "Success",
                               "abstol": request["solver"]["abstol"], "reltol": request["solver"]["reltol"],
                               "accepted_steps": 4, "rejected_steps": 0}})
    response = {"schema": jo.RESPONSE_SCHEMA, "status": "ok", "request_id": execution_id,
                "operation_id": jo.OPERATION, "data": output}
    return canonical(solve_request), canonical(response), output, {
        "schema": "ciw.julia-worker-identity.v1", "profile": RUNTIME["profile"],
        "operation_id": jo.OPERATION, "julia_version": "1.10.12", "platform": "test",
        "threads": RUNTIME["threads"], "startup_file": RUNTIME["startup_file"],
        "worker_sha256": RUNTIME["worker_sha256"], "project_sha256": RUNTIME["project_sha256"],
        "manifest_sha256": RUNTIME["manifest_sha256"],
    }


@pytest.fixture
def fake_worker(monkeypatch):
    monkeypatch.setattr(jo, "_invoke", fake_invoke)


def test_source_allowlist_and_refusal():
    raw = canonical(source())
    assert jo.validate_source(raw)["operation_id"] == jo.OPERATION
    invalid = copy.deepcopy(source())
    invalid["model"]["gamma_s_inv"] = 2.0
    with pytest.raises(ValueError):
        jo.validate_source(canonical(invalid))
    with pytest.raises(ValueError):
        jo.validate_source(canonical({**source(), "unexpected": 1}))


def test_framing_rejects_truncation():
    payload = canonical({"x": 1})
    encoded = jo._frame(payload)
    assert jo._frames(encoded) == [payload]
    with pytest.raises(ValueError):
        jo._frames(encoded[:-1])


def test_oracle_and_retained_raw_bytes(fake_worker):
    bundle = jo.workflow._execute(canonical(source()), {}, RUNTIME)
    step = bundle["steps"][0]
    assert base64.b64decode(step["request_bytes_b64"]) == canonical({**jo.request_from_source(source()), "request_id": step["execution_id"]})
    assert step["result"]["data"]["oracle_comparison"]["status"] == "passed"
    assert bundle["verification"]["independent"] is True
    jo.workflow._validate(bundle)


def test_replay_keeps_fresh_occurrence_and_reuses_numerics(fake_worker, monkeypatch):
    original = jo.workflow._execute(canonical(source()), {}, RUNTIME)
    monkeypatch.setattr(jo.workflow, "_adapters", lambda repositories, expected=None: ({}, RUNTIME))
    replay = jo.workflow.replay_session(original, {"julia": "unused", "julia_runtime": "unused"})
    assert replay["session"]["steps"][0]["execution_id"] != original["steps"][0]["execution_id"]
    assert replay["session"]["steps"][0]["numerical_result_id"] == original["steps"][0]["numerical_result_id"]
    assert replay["replay_receipt"]["admission"] == "not_performed"


def test_offline_projection_is_the_existing_run_contract(fake_worker):
    bundle = jo.workflow._execute(canonical(source()), {}, RUNTIME)
    run = jo.to_run(bundle["steps"][0]["result"]["data"])
    assert run["instrument"] == "julia-tsit5-oscillator.v1"
    assert run["channels"]["q"]["values"] == bundle["steps"][0]["result"]["data"]["output"]["q_m"]


def test_session_registers_and_retains_the_julia_operation(fake_worker, monkeypatch, tmp_path):
    monkeypatch.setattr(jo.JuliaOscillatorWorkflow, "_adapters", lambda self, repositories, expected=None: ({}, RUNTIME))
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("julia-oscillator", {"julia": "unused", "julia_runtime": "unused"})
    descriptor = session.handle({"protocol_version": 1, "request_id": "source", "type": "source.add",
                                 "payload": {"kind": jo.KIND, "label": "Julia test", "bytes_b64": base64.b64encode(canonical(source())).decode()}})
    operation = next(item for item in session.handle({"protocol_version": 1, "request_id": "ops", "type": "operation.list", "payload": {}})["operations"] if item["operation_id"] == jo.OPERATION)
    assert operation["available"] is True
    completed = session.handle({"protocol_version": 1, "request_id": "execute", "type": "operation.execute",
                                "payload": {"operation_id": jo.OPERATION, "parameters": {"source_id": descriptor["source_id"]}}})
    assert completed["type"] == "response"
    assert completed["payload"]["bundle"]["kind"] == jo.KIND
    session.save_workspace(tmp_path / "workspace.json")
    reopened = Session.from_workspace(tmp_path / "workspace.json", tmp_path / "reopened")
    assert reopened.handle({"protocol_version": 1, "request_id": "list", "type": "bundle.list", "payload": {}})["bundles"]


@pytest.mark.integration
def test_real_julia_provider_is_explicitly_gated(tmp_path):
    """This test is only enabled when an operator supplies a real Julia pin."""
    executable = tmp_path / "julia"
    runtime = tmp_path / "runtime"
    if not executable.is_file() or not runtime.is_dir():
        pytest.skip("Julia 1.10 runtime and instantiated project are not provisioned")
