"""Real Julia acceptance gate: numerical integration, oracle errors, isolation, failures and replay.

Set CIW_JULIA_EXECUTABLE to a Julia 1.10.12 executable whose depot holds the
packaged environment (see docs/JULIA_OSCILLATOR.md). Every test here runs the
actual pinned worker; a skipped run is not evidence of integration.
"""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.cli import main
from ciw.instruments import make_demo_run
from ciw.julia_oscillator import (JuliaOscillatorWorkflow, OPERATION, UNITS, encode_configuration, encode_input,
                                  encode_request, project_run, sample_times, split_response, decode_output)
from ciw.julia_worker import POOL, JuliaWorkerSession, environment_root, runtime_pin
from ciw.session import Session, write_json
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/julia-oscillator"
FIXTURES = ("default", "mixed-initial-state", "undamped", "loose-tolerance", "tight-tolerance", "stepped-output")


def call(session, kind, payload=None, *, error=False):
    reply = session.handle({"protocol_version": 1, "request_id": "julia-gate", "type": kind, "payload": payload or {}})
    assert reply["type"] == ("error" if error else "response"), reply
    return reply["payload"]


def source(name):
    return json.loads((EXAMPLES / (name + ".json")).read_bytes())


def add(session, name):
    raw = b"\n" + canonical(source(name)) + b"\n"
    return call(session, "source.add", {"kind": "julia-oscillator", "label": name, "bytes_b64": base64.b64encode(raw).decode()})


def native(bundle):
    return bundle["steps"][0]["result"]["data"]["native"]


def occurrence(bundle):
    return bundle["steps"][0]["result"]["data"]["occurrence"]


@pytest.fixture(scope="module")
def julia():
    path = os.environ.get("CIW_JULIA_EXECUTABLE")
    if not path:
        pytest.skip("set CIW_JULIA_EXECUTABLE to the pinned Julia 1.10.12 executable")
    return Path(path)


@pytest.fixture(scope="module")
def retained(julia, tmp_path_factory):
    directory = tmp_path_factory.mktemp("julia-workbench")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("julia-oscillator", {"julia": julia})
    bundles = {}
    for name in FIXTURES:
        declared = add(session, name)
        summary = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": declared["source_id"]}})
        bundles[name] = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    replay = call(session, "bundle.replay", {"bundle_id": bundles["default"]["bundle_digest"]})
    fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    output = os.environ.get("CIW_JULIA_FIXTURE_DIR")
    if output:
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        # Preserve declared key order: the viewport reads channels positionally.
        for name, bundle in bundles.items():
            write_json(target / (name + ".json"), bundle)
        write_json(target / "default-replay.json", fresh)
        write_json(target / "default-recording.json", project_run(bundles["default"]))
        write_json(target / "default-view.json", call(session, "experiment.inspect", {"bundle_id": bundles["default"]["bundle_digest"]}))
    return session, bundles, fresh, session.save_workspace(directory / "workspace.json")


def test_native_runtime_identity_is_the_pinned_environment(retained):
    session, bundles, _, _ = retained
    runtime = bundles["default"]["runtimes"]["julia"]
    pin = runtime_pin()
    assert runtime["julia_version"] == pin["julia_version"] == "1.10.12"
    assert runtime["packages"] == pin["packages"] and runtime["worker_source_sha256"] == pin["worker"]["sha256"]
    assert runtime["threads"] == 1 and runtime["options"]["startup_file"] == "disabled" and runtime["options"]["fast_math"] == "default"
    assert runtime["solver"] == {"algorithm": "Tsit5", "package": "OrdinaryDiffEqTsit5", "version": "2.1.4", "arithmetic": "binary64"}
    assert len(runtime["sysimage"]["sha256"]) == 64 and len(runtime["julia_sha256"]) == 64
    assert all(b["runtimes"]["julia"] == runtime for b in bundles.values())


def test_native_default_fixture_matches_the_analytic_reference_at_every_sample(retained):
    _, bundles, _, _ = retained
    bundle = bundles["default"]
    decoded = native(bundle)["decoded"]
    demo = make_demo_run()
    assert decoded["time_s"] == demo["time_s"]
    assert decoded["return_code"] == "Success" and decoded["solver"]["accepted_steps"] > 0
    measured = bundle["verification"]["measured"]
    for name in UNITS:
        actual = np.max(np.abs(np.asarray(decoded[name]) - np.asarray(demo["channels"][name]["values"])))
        assert measured[name]["max_abs_error"] == actual
        assert measured[name]["normalized_max_error"] <= 1e-7
    assert bundle["verification"]["outcome"] == "passed"
    assert bundle["verification"]["oracle"]["q"] == demo["channels"]["q"]["values"]
    assert measured["q"]["max_abs_error"] < 1e-8 and measured["v"]["max_abs_error"] < 1e-7 and measured["energy"]["max_abs_error"] < 1e-7


def test_native_mixed_initial_state_keeps_state_order_and_signs(retained):
    _, bundles, _, _ = retained
    decoded = native(bundles["mixed-initial-state"])["decoded"]
    assert decoded["q"][0] == -0.6 and decoded["v"][0] == 2.5
    assert bundles["mixed-initial-state"]["verification"]["outcome"] == "passed"
    assert all(bundles["mixed-initial-state"]["verification"]["measured"][name]["normalized_max_error"] <= 1e-7 for name in UNITS)


def test_native_undamped_limit_conserves_energy_within_tolerance(retained):
    _, bundles, _, _ = retained
    bundle = bundles["undamped"]
    conservation = bundle["verification"]["measured"]["numerical_energy_conservation"]
    assert conservation["status"] == "evaluated" and conservation["relative_drift"] < 1e-6
    assert bundle["verification"]["outcome"] == "passed"
    assert bundle["verification"]["measured"]["q"]["normalized_max_error"] <= 1e-7


def test_native_tolerance_profiles_record_error_and_work_without_proportionality_claims(retained):
    _, bundles, _, _ = retained
    rows = {name: (bundles[name]["verification"]["measured"]["q"]["normalized_max_error"],
                   native(bundles[name])["decoded"]["solver"]["accepted_steps"]) for name in ("loose-tolerance", "default", "tight-tolerance")}
    assert rows["loose-tolerance"][0] > rows["default"][0] > rows["tight-tolerance"][0]
    assert rows["loose-tolerance"][1] < rows["default"][1] < rows["tight-tolerance"][1]
    assert all(bundles[name]["verification"]["outcome"] == "passed" for name in rows)
    stepped = bundles["stepped-output"]["verification"]["measured"]["q"]["normalized_max_error"]
    assert stepped < rows["default"][0]
    assert native(bundles["stepped-output"])["decoded"]["solver"]["accepted_steps"] > rows["default"][1]


def test_native_worker_isolation_a_b_a_and_restart(retained, julia):
    session, bundles, _, _ = retained
    a_source = call(session, "source.list")["sources"]
    default_id = next(s["source_id"] for s in a_source if s["label"] == "default")
    mixed_id = next(s["source_id"] for s in a_source if s["label"] == "mixed-initial-state")
    runs = []
    for source_id in (default_id, mixed_id, default_id):
        summary = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source_id}})
        runs.append(call(session, "bundle.get", {"bundle_id": summary["bundle_id"]}))
    first, second, third = runs
    same_worker = {occurrence(b)["worker_session_id"] for b in runs}
    assert len(same_worker) == 1
    assert occurrence(first)["engine_occurrence"] < occurrence(second)["engine_occurrence"] < occurrence(third)["engine_occurrence"]
    assert first["steps"][0]["execution_id"] != third["steps"][0]["execution_id"]
    assert first["steps"][0]["numerical_result_id"] == third["steps"][0]["numerical_result_id"]
    assert second["steps"][0]["numerical_result_id"] != first["steps"][0]["numerical_result_id"]
    assert native(second)["decoded"]["q"][0] == -0.6
    restarted = POOL.restart(julia)
    assert restarted.session_id not in same_worker
    summary = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": default_id}})
    fourth = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    assert occurrence(fourth)["worker_session_id"] == restarted.session_id and occurrence(fourth)["engine_occurrence"] == 1
    assert fourth["steps"][0]["numerical_result_id"] == first["steps"][0]["numerical_result_id"]


def test_native_refusals_before_and_at_the_worker(retained, julia):
    session, _, _, _ = retained
    for change in ({"model": {"omega_0_rad_s": 25.0}}, {"grid": {"sample_count": 4097}}):
        declaration = source("default")
        for key, inner in change.items():
            declaration[key].update(inner)
        call(session, "source.add", {"kind": "julia-oscillator", "label": "bad", "bytes_b64": base64.b64encode(canonical(declaration)).decode()}, error=True)
    worker = POOL.session(julia)
    declaration = source("default")
    unsafe = dict(declaration["model"], omega_0_rad_s=25.0)
    _, response, _ = worker.execute(lambda n: encode_request(n, encode_configuration(declaration["configuration"]["solver"]),
                                                             encode_input(unsafe, sample_times(declaration["grid"]))))
    decoded = decode_output(split_response(response)[1])
    assert decoded["status"] == 4 and "omega_0" in decoded["message"] and worker.alive
    _, response, _ = worker.execute(lambda n: encode_request(n, encode_configuration(declaration["configuration"]["solver"]),
                                                             encode_input(declaration["model"], sample_times(declaration["grid"])), operation="ciw.other.v1"))
    assert decode_output(split_response(response)[1])["status"] == 3 and worker.alive


def test_native_timeout_and_crash_end_the_session_and_a_fresh_one_serves(retained, julia):
    session, _, _, _ = retained
    default_id = next(s["source_id"] for s in call(session, "source.list")["sources"] if s["label"] == "default")
    worker = POOL.session(julia)
    declaration = source("tight-tolerance")
    request = lambda n: encode_request(n, encode_configuration(declaration["configuration"]["solver"]),
                                       encode_input(declaration["model"], sample_times(declaration["grid"])))
    with pytest.raises(AdapterRefusal) as failure:
        worker.execute(request, timeout_seconds=1e-4)
    assert failure.value.code == "TIMEOUT" and not worker.alive
    replacement = POOL.session(julia)
    assert replacement.session_id != worker.session_id and replacement.alive
    os.kill(replacement._process.pid, signal.SIGKILL)
    replacement._process.wait(timeout=10)
    with pytest.raises(AdapterRefusal) as crashed:
        replacement.execute(request)
    assert crashed.value.code in {"WORKER_FAILED", "WORKER_UNAVAILABLE"} and not replacement.alive
    summary = call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": default_id}})
    bundle = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    assert occurrence(bundle)["worker_session_id"] not in {worker.session_id, replacement.session_id}
    assert bundle["verification"]["outcome"] == "passed"


def test_native_wrong_environment_is_refused_explicitly(julia, tmp_path):
    root = tmp_path / "environment"
    shutil.copytree(environment_root(), root)
    worker = root / "worker/oscillator_worker.jl"
    worker.write_text(worker.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(AdapterRefusal) as failure:
        JuliaWorkerSession(julia, root=root)
    assert failure.value.code == "RUNTIME_PIN_MISMATCH"
    wrong_version = deepcopy(runtime_pin())
    wrong_version["julia_version"] = "1.11.0"
    with pytest.raises(AdapterRefusal) as mismatch:
        JuliaWorkerSession(julia, pin=wrong_version)
    assert mismatch.value.code == "RUNTIME_PIN_MISMATCH" and "1.10.12" in str(mismatch.value)


def test_native_offline_restore_then_explicit_replay(retained, julia, tmp_path, monkeypatch):
    session, bundles, fresh, path = retained
    original = bundles["default"]
    receipt = fresh["replay_receipts"][0]
    assert receipt["verification"]["byte_identical"] is True
    assert fresh["steps"][0]["execution_id"] != original["steps"][0]["execution_id"]
    saved = json.loads(path.read_text(encoding="utf-8"))["workbench"]
    with monkeypatch.context() as patched:
        def forbidden(*args, **kwargs):
            raise AssertionError("Inspection must not start Julia")
        patched.setattr(JuliaWorkerSession, "__init__", forbidden)
        restored = Session.from_workspace(path, tmp_path / "restored")
        assert restored.workbench.serialize() == saved
        assert {r["kind"] for r in saved["bundles"]} == {"julia-oscillator"} and len(saved["bundles"]) == len(FIXTURES) + 1
        view = call(restored, "experiment.inspect", {"bundle_id": original["bundle_digest"]})
        assert view["object_context"]["oracle_outcome"] == "passed"
        call(restored, "bundle.replay", {"bundle_id": original["bundle_digest"]}, error=True)
    restored.workbench.bind_workflow("julia-oscillator", {"julia": julia})
    replayed = call(restored, "bundle.replay", {"bundle_id": original["bundle_digest"]})
    assert replayed["replay_receipt"]["verification"]["byte_identical"] is True
    assert replayed["bundle"]["bundle_id"] not in {original["bundle_digest"], fresh["bundle_digest"]}


def test_native_recording_projection_drives_the_legacy_viewport_session(retained, tmp_path):
    _, bundles, _, _ = retained
    run = project_run(bundles["default"])
    session = Session(run, tmp_path)
    snapshot = session.snapshot()
    assert snapshot["run"]["instrument"] == "julia-oscillator-trajectory.v1" and snapshot["selection"]["channel"] == "q"
    sample = call(session, "sample.get", {"time_s": 1.0})
    assert sample["sample_index"] == 64 and sample["values"]["q"] == native(bundles["default"])["decoded"]["q"][64]
    assert call(session, "analysis.spectrum")["data"]["peak_frequency_hz"] == pytest.approx(0.8, abs=0.1)
    full = call(session, "run.get")
    assert len(full["render"]["trajectory"]) == 768 and full["metadata"]["provenance"]["origin"] == "simulation"


def test_native_terminal_run_inspect_replay_recording(julia, tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["julia-oscillator", "run", "--source", str(EXAMPLES / "undamped.json"), "--julia-executable", str(julia), "--output-dir", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inspection"]["oracle_outcome"] == "passed" and Path(report["recording_file"]).is_file()
    assert main(["julia-oscillator", "inspect", str(output / "bundle.json")]) == 0
    assert json.loads(capsys.readouterr().out)["measured"]["numerical_energy_conservation"]["status"] == "evaluated"
    assert main(["julia-oscillator", "replay", str(output / "bundle.json"), "--julia-executable", str(julia), "--output-dir", str(tmp_path / "replay")]) == 0
    assert json.loads(capsys.readouterr().out)["replay_receipt"]["numerical_match"] is True
    assert main(["julia-oscillator", "recording", str(output / "bundle.json"), "--output", str(tmp_path / "rec.json")]) == 0
    assert json.loads(capsys.readouterr().out)["instrument"] == "julia-oscillator-trajectory.v1"
    assert main(["julia-oscillator", "run", "--source", str(EXAMPLES / "default.json"), "--julia-executable", str(julia), "--output-dir", str(output)]) == 2
