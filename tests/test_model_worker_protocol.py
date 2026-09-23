"""Worker lifecycle, refusal and failure paths with a protocol double.

The double performs no numerical work. These tests cover what a real worker
cannot be made to do on demand; they never satisfy the numerical Julia gate.
"""
import json
from pathlib import Path
import sys

import pytest

from ciw.model import codec
from ciw.model.worker import (DESCRIPTORS, ExecutionRefused, JuliaBinding, JuliaWorker, WorkerError, pinned_runtime,
                              program_bytes, source_sha256, worker_root)
from ciw.model.workflow import execute_run, inspect_run, replay_run

DOUBLE = Path(__file__).resolve().parent / "fixtures" / "fake_model_worker.py"
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
OPERATION = "ciw.model.simulate.v1"


def worker(mode, *extra, timeout=30.0):
    binding = JuliaBinding(julia=Path(sys.executable), command_prefix=(sys.executable, DOUBLE, mode, *extra),
                           startup_timeout_s=timeout)
    return JuliaWorker(binding, verify_pins=False)


def request():
    return codec.encode({"c": 1}), codec.encode({"i": 2})


def test_completed_execution_recomputes_every_identity():
    with worker("ok") as w:
        result = w.execute(OPERATION, *request())
        assert result.status == "completed" and result.occurrence == 1
        assert result.specification.program == program_bytes(OPERATION, w.runtime_digest)
        second = w.execute(OPERATION, *request())
        assert second.occurrence == 2 and second.session_id == result.session_id
        assert second.computation_identity == result.computation_identity  # same request, new occurrence


@pytest.mark.parametrize("mode, code", [
    ("hang", "worker_timeout"), ("crash", "worker_exited"), ("truncate", "worker_protocol"),
    ("wrong_occurrence", "worker_protocol"), ("wrong_output_identity", "worker_protocol"),
    ("wrong_input_identity", "worker_protocol"), ("halted_with_output", "worker_protocol"),
])
def test_failures_end_the_session_without_fabricating_a_result(mode, code):
    w = worker(mode)
    try:
        w.start()
        first_session = w.session_id
        with pytest.raises(WorkerError) as failure:
            w.execute(OPERATION, *request(), timeout=2.0)
        assert failure.value.code == code
        assert not w.alive and w.sessions[-1]["ended"] is not None
        assert w.sessions[-1]["session_id"] == first_session
    finally:
        w.close()


def test_a_failed_occurrence_is_not_retried_and_later_work_uses_a_new_session(tmp_path):
    marker = tmp_path / "crashed-once"
    with worker("fail_first", str(marker)) as w:
        w.start()
        crashed = w.session_id
        with pytest.raises(WorkerError, match="exited"):
            w.execute(OPERATION, *request())
        result = w.execute(OPERATION, *request())
        assert result.session_id != crashed and result.occurrence == 1
        assert [item["ended"] for item in w.sessions] == ["eof", None]


def test_refusal_and_halting_are_distinct_outcomes():
    with worker("unrunnable") as w:
        with pytest.raises(ExecutionRefused, match="not registered"):
            w.execute(OPERATION, *request())
    with worker("halted") as w:
        result = w.execute(OPERATION, *request())
        assert result.status == "halted" and result.output is None and result.computation_identity is None
        assert result.exit_code == 3 and "MaxIters" in result.detail


def test_handshake_must_bind_its_runtime_digest():
    with pytest.raises(WorkerError) as failure:
        worker("bad_digest").start()
    assert failure.value.code == "runtime_mismatch"


def test_garbage_on_stdout_is_a_protocol_failure():
    with pytest.raises(WorkerError) as failure:
        worker("garbage").start()
    assert failure.value.code == "worker_protocol"


def test_oversized_requests_are_refused_before_sending(monkeypatch):
    import ciw.model.worker as module
    monkeypatch.setattr(module, "MAX_FRAME", 1024)
    with worker("ok") as w:
        with pytest.raises(WorkerError) as failure:
            w.execute(OPERATION, codec.encode({}), codec.encode({"blob": b"x" * 4096}))
        assert failure.value.code == "oversized_request"


def test_operations_are_bound_to_their_profile():
    with worker("ok") as w:
        with pytest.raises(ExecutionRefused, match="not served"):
            w.execute("ciw.model.symbolic.v1", *request())


def test_pinned_handshake_rejects_an_unverified_runtime():
    binding = JuliaBinding(julia=Path(sys.executable), command_prefix=(sys.executable, DOUBLE, "ok"))
    with pytest.raises(WorkerError) as failure:
        JuliaWorker(binding, verify_pins=True).start()
    assert failure.value.code == "runtime_mismatch"


def test_descriptors_and_worker_sources_are_consistent():
    julia_descriptors = (worker_root() / "src" / "descriptors.jl").read_text()
    for operation, text in DESCRIPTORS.items():
        assert f'"{operation}" => """\n{text}"""' in julia_descriptors
    pin = pinned_runtime()
    assert pin["julia_version"] == "1.10.12"
    assert set(pin["profiles"]["core"]) < set(pin["profiles"]["symbolic"])
    assert len(source_sha256()) == 64
    manifest = (worker_root() / "Manifest.toml").read_text()
    assert 'julia_version = "1.10.12"' in manifest
    for name, version in pin["packages"].items():
        assert f'[[deps.{name}]]' in manifest and f'version = "{version}"' in manifest


def spec_and_request():
    spec = json.loads((EXAMPLES / "damped-oscillator.json").read_text())
    return spec, json.loads((EXAMPLES / "oscillator-simulate.json").read_text())


def test_failed_runs_are_retained_restorable_and_have_no_result(tmp_path):
    spec, simulation = spec_and_request()
    with worker("crash") as w:
        bundle = execute_run(OPERATION, spec, simulation, w)
    assert bundle["execution"]["status"] == "failed" and bundle["result"] is None
    assert bundle["execution"]["refusal"]["code"] == "worker_exited"
    assert bundle["physical_measurements"]["status"] == "not_acquired"
    restored = json.loads(json.dumps(bundle))
    summary = inspect_run(restored)
    assert summary["status"] == "failed" and summary["offline_checks"]["scr_identities"]
    restored["scr"]["input_b64"] = restored["scr"]["configuration_b64"]
    with pytest.raises(ValueError):
        inspect_run(restored)


def test_unbound_attempts_cannot_be_replayed():
    spec, simulation = spec_and_request()
    with worker("bad_digest") as w:
        bundle = execute_run(OPERATION, spec, simulation, w)
    assert bundle["execution"]["status"] == "refused" and bundle["runtime"] is None
    assert bundle["scr"]["program_b64"] is None
    inspect_run(bundle)
    with pytest.raises(ValueError, match="not bound to a runtime"):
        replay_run(bundle, worker("ok"))


def test_replay_requires_the_retained_runtime_digest(tmp_path):
    spec, simulation = spec_and_request()
    with worker("halted") as w:
        bundle = execute_run(OPERATION, spec, simulation, w)
    assert bundle["execution"]["status"] == "halted" and bundle["scr"]["output_b64"] is None
    tampered = json.loads(json.dumps(bundle))
    tampered["runtime"]["runtime_digest"] = "1" * 64
    with pytest.raises(ValueError):
        inspect_run(tampered)  # sealed record and program bytes both bind the runtime
    other = JuliaWorker(JuliaBinding(julia=Path(sys.executable), profile="core",
                                     command_prefix=(sys.executable, DOUBLE, "halted", "--variant")), verify_pins=False)
    with other:
        other.start()
        other.runtime_digest = "2" * 64
        with pytest.raises(WorkerError, match="retained runtime digest"):
            replay_run(bundle, other)
