"""Recorder integration with explicit test doubles; no fake hardware claims.

The recorder's host-only factory seam is exercised with synthetic counters and
CPU answers. Files written by these tests live only in pytest temporary paths.
The public JSON examples separately declare origin='synthetic_fixture'.
"""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.server import serve

from ciw import energy_bench as bench, energy_records as records, energy_cuda as cuda
from ciw import energy_nvml as nvml, free_energy_math as mathematics
from ciw import cli
from ciw.adapters import subprocess as adapter_subprocess
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.core.canonical import canonical, digest
from test_energy_records import make_log

ROOT = Path(__file__).resolve().parents[1]


def specification():
    return json.loads((ROOT/"examples/energy-accuracy/problem.json").read_text(encoding="utf-8"))


class SyntheticClock:
    def __init__(self):
        self.value = 10**15

    def perf_counter_ns(self):
        self.value += 1_000_000
        return self.value

    def perf_counter(self):
        return self.perf_counter_ns()/1e9

    def sleep(self,seconds):
        self.value += int(seconds*1e9)


@pytest.fixture
def rig(monkeypatch):
    clock = SyntheticClock()
    original_clock_info = bench.time.get_clock_info
    monkeypatch.setattr(bench,"time",SimpleNamespace(perf_counter_ns=clock.perf_counter_ns,
        perf_counter=clock.perf_counter,sleep=clock.sleep,get_clock_info=original_clock_info))
    template = make_log()
    state = SimpleNamespace(counter=None,worker=None,counter_start_error=False,worker_start_error=False,
        counter_error_on=0,worker_error_on=0,failed_counter_row=False,failed_row_emitted=False,
        mismatched_device=False,cleanup_errors=False)

    class Counter:
        def __init__(self,device_index=0):
            if state.counter_start_error:
                raise RuntimeError("synthetic counter initialization failed")
            self.closed,self.read_count = 0,0
            state.counter = self

        def identity(self):
            return deepcopy(template["sensor"])

        def read(self):
            self.read_count += 1
            if self.read_count == state.counter_error_on:
                raise RuntimeError("synthetic counter transport failed")
            a,b = clock.perf_counter_ns(),clock.perf_counter_ns()
            row = {"read_start_ns":a,"read_end_ns":b,"utc_start_ns":10**18+a,"utc_end_ns":10**18+b,
                "status":"ok","energy_mj":2**63+self.read_count*500,"error_code":None,
                "power_mw":1000,"temperature_c":40,"graphics_clock_mhz":300,"context_errors":{}}
            if (state.failed_counter_row and not state.failed_row_emitted and
                    state.worker is not None and state.worker.solve_count >= 2):
                state.failed_row_emitted = True
                row.update(status="error",energy_mj=None,error_code=15)
            return row

        def close(self):
            self.closed += 1
            if state.cleanup_errors:
                raise RuntimeError("synthetic counter cleanup failed")

    class Worker:
        def __init__(self,problem,solver,iterations,*,replicas,device_index):
            if state.worker_start_error:
                raise RuntimeError("synthetic worker initialization failed")
            self.closed,self.solve_count = 0,0
            self.problem,self.replicas = deepcopy(problem),replicas
            self.metadata = deepcopy(template["runtime"]["workload"])
            self.metadata.update(device_index=device_index,iterations=iterations,replicas=replicas,
                solver_settings=deepcopy(solver),problem_sha256=digest(problem)[7:],
                prepared_input_sha256=sha256(cuda._prepare(problem,solver,iterations,replicas,device_index).astype("<f8").tobytes()).hexdigest())
            if state.mismatched_device:
                self.metadata["device_uuid"] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
            state.worker = self

        def identity(self):
            return deepcopy(self.metadata)

        def solve(self):
            self.solve_count += 1
            if self.solve_count == state.worker_error_on:
                raise RuntimeError("synthetic worker solve failed")
            clock.value += 600_000_000
            # Deliberately CPU-built test output, not a CUDA implementation.
            reference = mathematics.gaussian_reference(self.problem)
            return np.tile(np.r_[reference["mean"],np.asarray(reference["covariance"]).ravel()],(self.replicas,1))

        def close(self):
            self.closed += 1
            if state.cleanup_errors:
                raise RuntimeError("synthetic worker cleanup failed")

    state.counter_factory,state.worker_factory,state.clock = Counter,Worker,clock
    return state


def capture(rig,path,**changes):
    arguments = {"minimum_duration_s":1,"idle_duration_s":.2,"warmup_batches":1,
        "replicas":4,"max_batches":4,"counter_factory":rig.counter_factory,"worker_factory":rig.worker_factory}
    arguments.update(changes)
    return bench.capture(specification(),path,**arguments)


def journal(path):
    return [json.loads(line) for line in (path/"journal.jsonl").read_text(encoding="utf-8").splitlines()]


def test_capture_retains_lossless_counter_rows_outputs_and_phase_boundaries(rig,tmp_path):
    path = tmp_path/"capture"
    log,report = capture(rig,path)
    records.validate_log(log)
    assert records.analyze(log) == report
    assert json.loads((path/"log.json").read_text()) == log
    assert json.loads((path/"report.json").read_text()) == report
    assert [p["name"] for p in log["phases"]] == list(records.PHASE_NAMES)
    assert len(log["phases"][1]["batches"]) == 1
    assert len(log["phases"][3]["batches"]) == 2
    events = journal(path)
    samples = [entry["value"] for entry in events if entry["event"] == "sample"]
    assert samples == [{"phase":phase["name"],**row} for phase in log["phases"] for row in phase["samples"]]
    assert all(type(row["energy_mj"]) is str and int(row["energy_mj"]) > 2**63 for row in samples)
    assert all(type(row["utc_start_ns"]) is str for row in samples)
    assert events[-1] == {"event":"complete","value":{"log_digest":log["log_digest"]}}
    assert rig.counter.closed == rig.worker.closed == 1
    assert report["hardware_provenance"] == "retained_operator_record_not_authenticated"


@pytest.mark.parametrize("mode", ["counter_start_error","worker_start_error","counter_error_on","worker_error_on"])
def test_failed_capture_keeps_durable_partial_journal_and_closes_resources(rig,tmp_path,mode):
    setattr(rig,mode,3 if mode == "counter_error_on" else 2 if mode == "worker_error_on" else True)
    path = tmp_path/"failed"
    with pytest.raises(RuntimeError,match="synthetic"):
        capture(rig,path)
    failure = json.loads((path/"failure.json").read_text())
    events = journal(path)
    assert failure["error_type"] == "RuntimeError" and "synthetic" in failure["error"]
    assert events[0]["event"] == "capture" and events[-1]["event"] == "failure"
    assert not (path/"log.json").exists() and not (path/"report.json").exists()
    if rig.counter is not None:
        assert rig.counter.closed == 1
    if rig.worker is not None:
        assert rig.worker.closed == 1
    if mode == "worker_error_on":
        assert any(e["event"] == "batch" for e in events)  # Completed warmup survives later failure.


def test_raw_failed_counter_row_remains_null_and_invalidates_comparison(rig,tmp_path):
    rig.failed_counter_row = True
    log,report = capture(rig,tmp_path/"missing-counter")
    failed = [r for p in log["phases"] for r in p["samples"] if r["status"] == "error"]
    assert len(failed) == 1 and failed[0]["energy_mj"] is None and failed[0]["error_code"] == 15
    assert not report["comparison"]["eligible"]
    assert "counter_read_error" in report["comparison"]["reasons"]
    assert report["measurement"]["gross_energy_j"] is None


def test_different_gpu_uuid_refuses_work_before_warmup(rig,tmp_path):
    rig.mismatched_device = True
    path = tmp_path/"mismatch"
    with pytest.raises(ValueError,match="different UUIDs"):
        capture(rig,path)
    assert rig.worker.solve_count == 0
    assert rig.counter.closed == rig.worker.closed == 1
    assert any(e["event"] == "failure" for e in journal(path))


def test_cleanup_errors_remain_in_journal_and_both_resources_close(rig,tmp_path):
    rig.cleanup_errors = True
    path = tmp_path/"cleanup"
    capture(rig,path)
    errors = [e["value"] for e in journal(path) if e["event"] == "cleanup_error"]
    assert {e["resource"] for e in errors} == {"counter","worker"}
    assert rig.counter.closed == rig.worker.closed == 1


def test_existing_output_is_not_replaced_and_invalid_plan_never_opens_hardware(rig,tmp_path):
    existing = tmp_path/"existing"
    existing.mkdir()
    (existing/"journal.jsonl").write_text("retained previous run\n")
    with pytest.raises(FileExistsError):
        capture(rig,existing)
    assert (existing/"journal.jsonl").read_text() == "retained previous run\n"
    with pytest.raises(ValueError):
        capture(rig,tmp_path/"bad-plan",replicas=True)
    assert rig.counter is rig.worker is None
    assert not (tmp_path/"bad-plan").exists()


def test_plan_keeps_correlated_problem_and_explicit_underconverged_iteration():
    source = specification()
    before = deepcopy(source)
    planned = bench.prepare_plan(source)
    assert planned["problem"]["noise_covariance"][0][1] != 0
    assert 0 < planned["iterations"] <= source["solver"]["max_iterations"]
    assert bench.prepare_plan(source,iterations=0)["iterations"] == 0
    assert source == before


def test_cli_replay_rejects_oversize_whitespace_before_json_parse(tmp_path,monkeypatch,capsys):
    path = tmp_path/"oversize.json"
    # This is valid JSON with a small parsed object: the transport bytes matter.
    path.write_bytes(canonical(make_log()) + b" " * records.MAX_BYTES)
    output = tmp_path/"report.json"
    def forbidden(*_args,**_kwargs):
        pytest.fail("Oversize energy input reached parsing or analysis")
    monkeypatch.setattr(adapter_subprocess,"_json",forbidden)
    monkeypatch.setattr(records,"analyze",forbidden)
    assert cli.main(["energy","replay",str(path),"--output",str(output)]) == 2
    captured = capsys.readouterr()
    assert "exact-byte size bound" in captured.err and captured.out == ""
    assert not output.exists()


def test_cli_interrupted_record_returns_130_and_retains_partial_journal(rig,tmp_path,monkeypatch,capsys):
    original_capture = bench.capture
    class InterruptedWorker(rig.worker_factory):
        def solve(self):
            if self.solve_count == 1:
                raise KeyboardInterrupt("synthetic operator interrupt")
            return super().solve()
    def fake_capture(spec,path,**options):
        return original_capture(spec,path,counter_factory=rig.counter_factory,
            worker_factory=InterruptedWorker,**options)
    monkeypatch.setattr(bench,"capture",fake_capture)
    output = tmp_path/"interrupted"
    code = cli.main(["energy","record","--problem",str(ROOT/"examples/energy-accuracy/problem.json"),
        "--output-dir",str(output),"--duration","1","--replicas","4","--max-batches","4",
        "--warmup-batches","1","--idle-duration","0.2"])
    assert code == 130
    failure = json.loads((output/"failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    events = journal(output)
    assert events[-1]["event"] == "failure"
    assert any(e["event"] == "batch" and e["value"]["phase"] == "warmup" for e in events)
    assert not (output/"log.json").exists() and not (output/"report.json").exists()
    assert rig.worker.closed == rig.counter.closed == 1
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_cli_interrupted_probe_preserves_existing_nonrecord_exit_semantics(monkeypatch,capsys):
    def interrupted(*_args,**_kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(bench,"probe",interrupted)
    assert cli.main(["energy","probe"]) == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("name,reason", [("baseline",None),("reset","reset_or_wrap_ambiguous"),
    ("missing","missing_endpoint_brackets"),("under-target","accuracy_target_not_met_by_every_batch")])
def test_bundled_fixture_claims_remain_synthetic_and_failure_modes_are_visible(name,reason):
    log = json.loads((ROOT/"examples/energy-accuracy"/(name+".json")).read_text())
    report = records.analyze(log)
    assert log["origin"] == "synthetic_fixture"
    assert report["hardware_provenance"] == "synthetic_fixture_no_physical_verification"
    assert report["comparison"]["eligible"] is False
    if reason:
        assert reason in report["comparison"]["reasons"]
        assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None
    else:
        assert report["comparison"]["classification"] == "synthetic_only"
        assert report["measurement"]["target_met"]


def test_example_client_retains_and_replays_logs_and_restores_without_hardware(tmp_path_factory,monkeypatch):
    def forbidden(*_args,**_kwargs):
        pytest.fail("Retained log workflow attempted hardware execution")
    monkeypatch.setattr(nvml,"_load_nvml",forbidden)
    monkeypatch.setattr(cuda,"_Driver",forbidden)
    monkeypatch.setattr(bench,"capture",forbidden)
    client = runpy.run_path(str(ROOT/"examples/energy-accuracy/run.py"))["run"]
    # Keep generated recording filenames below legacy Windows path limits.
    tmp_path = tmp_path_factory.mktemp("energy")
    session = Session(make_demo_run(),tmp_path/"session")
    async def live():
        async with serve(WorkbenchServer(session).handler,"127.0.0.1",0) as listener:
            url = "ws://127.0.0.1:"+str(listener.sockets[0].getsockname()[1])
            results = [await client(url,ROOT/"examples/energy-accuracy"/(name+".json"),tmp_path/name)
                       for name in ("baseline","reset")]
            with pytest.raises(FileExistsError):
                await client(url,ROOT/"examples/energy-accuracy/baseline.json",tmp_path/"baseline")
            return results
    results = asyncio.run(live())
    assert len(session.workbench.list_sources()) == 2 and len(session.workbench.list_bundles()) == 4
    for name,result in zip(("baseline","reset"),results):
        assert json.loads((tmp_path/name/"report.json").read_text()) == result
        assert result["original"] != result["replay"]
        assert result["view"]["authority"]["state_admission"] == "not_performed"
    restored = Session.from_workspace(Path(results[-1]["workspace"]["workspace_file"]),tmp_path/"restored")
    assert canonical(restored.workbench.serialize()) == canonical(session.workbench.serialize())
    for result in results:
        for key in ("original","replay"):
            payload = {"bundle_id":result[key]}
            assert restored.workbench.inspect_experiment(payload) == session.workbench.inspect_experiment(payload)
