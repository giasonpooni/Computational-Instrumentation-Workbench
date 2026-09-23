"""Genuine Julia worker acceptance set (docs/JULIA_SP1.md, oscillator increment and model core).

Requires ``CIW_JULIA`` naming the pinned Julia 1.10.12 executable with the
worker environment instantiated. Protocol doubles cannot satisfy these tests;
``scripts/check_model_core.py`` runs them and fails on any skip.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from ciw.adapters.oscillator import make_demo_run
from ciw.model import codec
from ciw.model.compose import compose
from ciw.model.providers import committed_bytes
from ciw.model.simulation import simulate_reference, uniform_request
from ciw.model.spec import SpecificationError, validate_spec
from ciw.model.transform import rescale, rescale_request
from ciw.model.worker import FRAME_MAGIC, ExecutionRefused, JuliaBinding, JuliaWorker, WorkerError
from ciw.model.workflow import binding_for_run, execute_run, inspect_run, read_run, replay_run, save_run

JULIA = os.environ.get("CIW_JULIA")
DEPOT = os.environ.get("CIW_JULIA_DEPOT")
pytestmark = pytest.mark.skipif(not JULIA, reason="CIW_JULIA names the pinned Julia 1.10.12 executable")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
SIMULATE = "ciw.model.simulate.v1"
# Declared before execution: componentwise absolute thresholds against the analytic oracle.
ORACLE = {"q": 1e-9, "v": 5e-9, "E": 1e-8}


def binding(profile="core"):
    return JuliaBinding(julia=Path(JULIA), profile=profile, depot=Path(DEPOT) if DEPOT else None)


@pytest.fixture(scope="module")
def core():
    with JuliaWorker(binding()) as worker:
        worker.start()
        yield worker


def load(name):
    return json.loads((EXAMPLES / name).read_text())


def oscillator(**changes):
    spec = load("damped-oscillator.json")
    for entry in spec["parameters"]:
        if entry["symbol"] in changes:
            entry["value"] = changes[entry["symbol"]]
    return validate_spec(spec)


def simulate(worker, model, request):
    configuration, payload = committed_bytes(SIMULATE, model, request)
    result = worker.execute(SIMULATE, configuration, payload)
    assert result.status == "completed", result.detail
    return result, codec.decode(result.output)


def analytic(t, q0, v0, omega, gamma, mass=1.0):
    wd = math.sqrt(omega * omega - gamma * gamma)
    b = (v0 + gamma * q0) / wd
    envelope = np.exp(-gamma * t)
    q = envelope * (q0 * np.cos(wd * t) + b * np.sin(wd * t))
    v = envelope * ((b * wd - gamma * q0) * np.cos(wd * t) + (-q0 * wd - gamma * b) * np.sin(wd * t))
    return q, v, 0.5 * mass * (v * v + omega * omega * q * q)


def errors(output, q, v, energy):
    x = output["x"].array
    return {"q": float(np.max(np.abs(x[:, 0] - q))), "v": float(np.max(np.abs(x[:, 1] - v))),
            "E": float(np.max(np.abs(output["derived"]["E"].array - energy)))}


def test_handshake_matches_the_pinned_runtime(core):
    identity = core.identity
    assert identity["julia_version"] == "1.10.12" and identity["threads"] == 1 and identity["blas_threads"] == 1
    assert identity["packages"]["OrdinaryDiffEqTsit5"] == "2.1.4"
    assert core.runtime_record()["host_checked"]["pins_verified"] is True


def test_default_fixture_matches_all_768_analytic_samples(core):
    model = oscillator()
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0})
    _, output = simulate(core, model, request)
    run = make_demo_run()
    assert output["t"].array.tolist() == run["time_s"] and len(run["time_s"]) == 768
    measured = errors(output, run["channels"]["q"]["values"], run["channels"]["v"]["values"],
                      run["channels"]["energy"]["values"])
    assert all(measured[key] <= ORACLE[key] for key in ORACLE), measured
    assert output["retcode"] == "Success" and output["stats"]["nreject"] >= 0


def test_mixed_initial_state_catches_order_and_sign_errors(core):
    model = oscillator()
    request = uniform_request(model, initial_state=[0.7, -1.3], inputs={"F": 0.0})
    _, output = simulate(core, model, request)
    t = output["t"].array
    measured = errors(output, *analytic(t, 0.7, -1.3, 2 * math.pi * 0.8, 0.15))
    assert all(measured[key] <= ORACLE[key] for key in ORACLE), measured
    swapped = errors(output, *analytic(t, -1.3, 0.7, 2 * math.pi * 0.8, 0.15))
    assert swapped["q"] > 0.1  # the check would detect a state-order error


def test_undamped_limit_keeps_phase_and_bounded_energy_drift(core):
    model = oscillator(gamma=0.0)
    request = uniform_request(model, initial_state=[1.0, 0.5], inputs={"F": 0.0})
    _, output = simulate(core, model, request)
    t = output["t"].array
    measured = errors(output, *analytic(t, 1.0, 0.5, 2 * math.pi * 0.8, 0.0))
    energy = output["derived"]["E"].array
    drift = float(np.max(np.abs(energy - energy[0])) / energy[0])
    assert measured["q"] <= ORACLE["q"] and measured["v"] <= ORACLE["v"] and drift <= 1e-9, (measured, drift)


def test_tightened_tolerances_record_error_and_cost_changes(core):
    model = oscillator()
    rows = []
    for reltol, abstol in ((1e-6, 1e-8), (1e-9, 1e-11), (1e-12, 1e-14)):
        request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, reltol=reltol,
                                  abstol=[abstol, abstol])
        result, output = simulate(core, model, request)
        run = make_demo_run()
        measured = errors(output, run["channels"]["q"]["values"], run["channels"]["v"]["values"],
                          run["channels"]["energy"]["values"])
        rows.append((reltol, measured["q"], output["stats"]["naccept"], output["stats"]["nf"], result.elapsed_ns))
    assert rows[0][1] > rows[1][1] > rows[2][1]  # error falls with tighter tolerance ...
    assert rows[0][2] < rows[1][2] < rows[2][2] and rows[0][3] < rows[2][3]  # ... and the cost rises
    # The measured global error is recorded against the oracle, never inferred
    # from the requested tolerance (which bounds local, not global, error).
    print("reltol, max |q error| m, accepted steps, f evaluations, elapsed ns:", rows)


def test_repeated_interleaved_and_restarted_runs_share_no_state(core):
    model = oscillator()
    a = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=256)
    b = uniform_request(model, initial_state=[-0.4, 2.0], inputs={"F": 3.0}, count=256)
    first, _ = simulate(core, model, a)
    other, _ = simulate(core, model, b)
    again, _ = simulate(core, model, a)
    assert first.output == again.output and first.output != other.output
    assert first.occurrence < other.occurrence < again.occurrence and first.session_id == again.session_id
    with JuliaWorker(binding()) as restarted:
        fresh, _ = simulate(restarted, model, a)
    assert fresh.output == first.output and fresh.session_id != first.session_id
    assert fresh.specification_identity == first.specification_identity


def test_invalid_numbers_and_grids_are_refused_before_dispatch():
    model = oscillator()
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=8)
    for field, value in (("initial_state", [True, 0.0]), ("sample_times", [0.0, float("nan")] + [1.0] * 6),
                         ("inputs", {"F": 1e9})):
        broken = deepcopy(request)
        broken[field] = value
        with pytest.raises(SpecificationError):
            committed_bytes(SIMULATE, model, broken)


def test_wrong_environment_program_is_refused_by_the_worker(monkeypatch):
    import ciw.model.worker as module
    genuine = module.program_bytes
    monkeypatch.setattr(module, "program_bytes", lambda operation, digest: genuine(operation, "0" * 64))
    model = oscillator()
    configuration, payload = committed_bytes(SIMULATE, model, uniform_request(
        model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=8))
    with JuliaWorker(binding()) as worker:
        with pytest.raises(ExecutionRefused, match="not registered in this worker runtime"):
            worker.execute(SIMULATE, configuration, payload)
        assert worker.alive  # a refusal is an answer, not a session failure


def test_worker_side_frame_bound_and_out_of_order_requests_end_the_session():
    with JuliaWorker(binding()) as worker:
        worker.start()
        worker.process.stdin.write(FRAME_MAGIC + struct.pack("<I", 64 * 1024 * 1024 + 1))
        worker.process.stdin.flush()
        with pytest.raises(WorkerError) as failure:
            worker._receive(120.0, "oversized frame")
        assert failure.value.code == "worker_exited" and "exceeds" in failure.value.args[0]
        worker.start()
        worker._send(codec.encode({"type": "execute", "occurrence": 7, "operation": SIMULATE, "program": b"",
                                   "configuration": b"", "input": b"", "specification_identity": ""}))
        with pytest.raises(WorkerError) as failure:
            worker._receive(120.0, "out-of-order occurrence")
        assert failure.value.code == "worker_exited"


def test_timeout_kills_the_session_and_later_work_starts_fresh():
    model = oscillator()
    configuration, payload = committed_bytes(SIMULATE, model, uniform_request(
        model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, count=4096, rate=4096 / 12, reltol=1e-13,
        abstol=[1e-15, 1e-15]))
    with JuliaWorker(binding()) as worker:
        worker.start()
        timed_out = worker.session_id
        with pytest.raises(WorkerError) as failure:
            worker.execute(SIMULATE, configuration, payload, timeout=0.001)
        assert failure.value.code == "worker_timeout" and not worker.alive
        result = worker.execute(SIMULATE, configuration, payload)
        assert result.status == "completed" and result.session_id != timed_out and result.occurrence == 1


def test_solver_failure_halts_without_output(core):
    model = oscillator()
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, maxiters=5)
    configuration, payload = committed_bytes(SIMULATE, model, request)
    result = core.execute(SIMULATE, configuration, payload)
    assert result.status == "halted" and result.exit_code == 3 and "MaxIters" in result.detail
    assert result.output is None and result.computation_identity is None


def test_millimetre_model_is_invariant_through_julia(core):
    original = oscillator()
    mm = rescale(original, {"q": "mm", "v": "mm/s", "y": "mm"})
    request = uniform_request(original, initial_state=[0.3, -0.4], inputs={"F": 0.5}, abstol=[1e-12, 1e-12])
    _, a = simulate(core, original, request)
    _, b = simulate(core, mm, rescale_request(original, mm, request))
    assert np.allclose(b["x"].array / 1000.0, a["x"].array, rtol=1e-9, atol=1e-15)
    assert np.allclose(b["observations"]["y"].array / 1000.0, a["observations"]["y"].array, rtol=1e-9, atol=1e-15)
    naive = uniform_request(mm, initial_state=[300.0, -400.0], inputs={"F": 0.5}, abstol=[1e-12, 1e-12])
    _, c = simulate(core, mm, naive)  # tolerances not transformed: a different numerical problem
    assert c["stats"]["naccept"] != b["stats"]["naccept"]


def test_retained_simulation_run_inspects_offline_and_replays_freshly(core, tmp_path):
    spec, request = load("damped-oscillator.json"), load("oscillator-simulate.json")
    bundle = execute_run(SIMULATE, spec, request, core)
    assert bundle["execution"]["status"] == "completed" and bundle["diagnostics"]["passed"]
    assert bundle["physical_measurements"]["status"] == "not_acquired"
    path = save_run(bundle, tmp_path)
    environment = {key: value for key, value in os.environ.items() if key not in {"CIW_JULIA", "PATH"}}
    environment["PATH"] = os.path.dirname(sys.executable)
    completed = subprocess.run([sys.executable, "-m", "ciw", "model", "inspect", str(path)], capture_output=True,
                               text=True, env=environment, timeout=120)
    assert completed.returncode == 0, completed.stderr
    offline = json.loads(completed.stdout)
    assert offline["offline_checks"]["derived_view"] and offline["computation_identity"]
    replay = replay_run(read_run(path), core)
    comparison = replay["replay_of"]["comparison"]
    assert comparison["output_bytes_identical"] and comparison["fresh_execution_identity"]
    assert replay["result"]["result_id"] != bundle["result"]["result_id"]
    view = binding_for_run(bundle)
    assert view["estimates"]["omega_0"]["standard_uncertainty"] == pytest.approx(0.01)


def test_linearization_matches_analytic_matrices_and_poles(core):
    spec, request = load("damped-oscillator.json"), load("oscillator-linearize.json")
    bundle = execute_run("ciw.model.linearize.v1", spec, request, core)
    assert bundle["diagnostics"]["passed"], bundle["diagnostics"]
    view = bundle["result"]["view"]
    omega, gamma = 2 * math.pi * 0.8, 0.15
    assert np.allclose(view["A"], [[0.0, 1.0], [-omega ** 2, -2 * gamma]], rtol=1e-14, atol=1e-14)
    assert np.allclose(view["B"], [[0.0], [1.0]]) and view["C"] == [[1.0, 0.0]] and view["D"] == [[0.0]]
    wd = math.sqrt(omega ** 2 - gamma ** 2)
    assert np.allclose(view["poles_real"], [-gamma, -gamma]) and np.allclose(view["poles_imag"], [-wd, wd])
    assert view["controllability_rank"] == 2 and view["observability_rank"] == 2


def test_measurement_selection_is_optimal_and_unit_invariant(core):
    spec, request = load("damped-oscillator.json"), load("oscillator-design.json")
    bundle = execute_run("ciw.model.measurement-selection.v1", spec, request, core)
    diagnostics = bundle["diagnostics"]
    assert diagnostics["passed"] and diagnostics["enumeration"]["selected_is_optimal"], diagnostics
    original = validate_spec(spec)
    scaled = rescale(original, {"omega_0": "rad/ms", "gamma": "ms^-1", "y": "mm", "q": "mm", "v": "mm/s"})
    moved = deepcopy(request)
    moved.update(spec_digest=scaled.digest, initial_state=[1000.0, 0.0])
    moved["solver"]["abstol"] = [1e-9, 1e-9]
    again = execute_run("ciw.model.measurement-selection.v1", scaled.spec, moved, core)
    assert again["result"]["view"]["selected"] == bundle["result"]["view"]["selected"]
    assert again["result"]["view"]["objective"] == pytest.approx(bundle["result"]["view"]["objective"], rel=1e-6)


def test_composed_closed_loop_runs_in_julia_against_the_python_reference(core):
    plant, controller = oscillator(), validate_spec(load("pd-controller.json"))
    loop = compose("closed-loop", "closed loop", [{"name": "plant", "model": plant}, {"name": "ctrl", "model": controller}],
                   [{"from": "plant.position", "to": "ctrl.measurement"}, {"from": "ctrl.command", "to": "plant.force"}])
    request = uniform_request(loop, initial_state=[0.3, 0.0, 300.0], inputs={}, count=384,
                              abstol=[1e-12, 1e-12, 1e-9], state_atol=[1e-8, 1e-8, 1e-5])
    bundle = execute_run(SIMULATE, loop.spec, request, core)
    assert bundle["diagnostics"]["passed"], bundle["diagnostics"]
    reference = simulate_reference(loop, request)
    assert np.allclose(bundle["result"]["view"]["derived"]["plant.F"], reference["derived"]["plant.F"], atol=1e-6)


def test_modelingtoolkit_route_maps_its_state_order_back():
    spec, request = load("damped-oscillator.json"), load("oscillator-linearize.json")
    with JuliaWorker(binding("symbolic")) as worker:
        bundle = execute_run("ciw.model.symbolic.v1", spec, request, worker)
    diagnostics = bundle["diagnostics"]
    assert diagnostics["passed"], diagnostics
    assert diagnostics["state_order"]["declared"] == ["q", "v"]
    view = bundle["result"]["view"]
    assert sorted(view["compiled_unknowns"]) == ["q", "v"]
    assert [view["compiled_unknowns"][i] for i in view["permutation"]] == ["q", "v"]
    assert r"\frac{\mathrm{d}" in view["latex_equations"]
    assert inspect_run(bundle)["offline_checks"]["derived_view"]
