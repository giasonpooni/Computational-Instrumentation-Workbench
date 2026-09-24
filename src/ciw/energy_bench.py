"""Host-only capture of GPU-domain energy and retained Gaussian accuracy.

Reference planning and NVML initialization precede the five measured phases.
The raw journal is flushed to durable storage at each event. Its cost, sampling,
host validation and background device work remain in each phase's gross window.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

from . import energy_records, free_energy_math
from .energy_cuda import CudaGaussianWorker
from .energy_nvml import NVMLEnergyCounter
from .core.canonical import canonical, digest

SPEC_SCHEMA = "ciw.energy-problem.v1"


def code_identity():
    files = ("energy_bench.py", "energy_cuda.py", "energy_nvml.py", "energy_records.py", "free_energy_math.py")
    value = {name: Path(__file__).with_name(name).read_text(encoding="utf-8").replace("\r\n", "\n") for name in files}
    return {"profile": "ciw.fixed-gaussian-gpu-energy.v1", "code_sha256": sha256(canonical(value)).hexdigest()}


def prepare_plan(spec, *, minimum_duration_s=3, replicas=4096, iterations=None,
                 max_batches=2048, warmup_batches=2, idle_duration_s=1):
    if type(spec) is not dict or set(spec) != {"schema", "problem", "solver", "target_kl_nats"} or spec["schema"] != SPEC_SCHEMA:
        raise ValueError("Require a ciw.energy-problem.v1 problem specification")
    spec = deepcopy(spec)
    solver, problem = spec["solver"], spec["problem"]
    # Validate resource limits before executing even the bounded CPU planner.
    if type(solver) is not dict or type(solver.get("max_iterations")) is not int or not 1 <= solver["max_iterations"] <= 256:
        raise ValueError("Energy workload max_iterations must lie in [1,256]")
    plan = {"problem": problem, "solver": solver, "iterations": 0 if iterations is None else iterations,
            "replicas": replicas, "target_kl_nats": spec["target_kl_nats"], "minimum_duration_s": minimum_duration_s,
            "max_batches": max_batches, "warmup_batches": warmup_batches, "idle_duration_s": idle_duration_s,
            "problem_digest": digest(problem),
            "model_digest": digest({key: value for key, value in problem.items() if key != "observations"}),
            "observations_digest": digest(problem.get("observations"))}
    energy_records._plan(plan)
    if iterations is None:
        fit = free_energy_math.variational_fit(problem, **solver)
        accepted = [row["iteration"] for row in fit["trace"] if row["kl_to_reference"] <= plan["target_kl_nats"]]
        if not accepted:
            raise ValueError("CPU planning did not attain the target; declare --iterations to measure a failure case")
        plan["iterations"] = accepted[0]
    return plan


def _publish(path, value):
    # The caller owns a newly created directory. Exclusive creation never
    # replaces a prior run; a journal remains if this final write is interrupted.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture(spec, output_dir, *, minimum_duration_s=3, replicas=4096, iterations=None,
            device_index=0, max_batches=2048, warmup_batches=2, idle_duration_s=1,
            counter_factory=NVMLEnergyCounter, worker_factory=CudaGaussianWorker):
    """Capture one new physical occurrence; never overwrite an existing run.

    A failed acquisition leaves journal.jsonl and failure.json for diagnosis.
    A later report-write failure can also leave an already sealed valid log.
    Replay uses energy_records.analyze and never enters this capture function.
    """
    if type(device_index) is not int or not 0 <= device_index <= 63:
        raise ValueError("device_index must lie in [0,63]")
    plan = prepare_plan(spec, minimum_duration_s=minimum_duration_s, replicas=replicas, iterations=iterations,
                        max_batches=max_batches, warmup_batches=warmup_batches, idle_duration_s=idle_duration_s)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    origin = time.perf_counter_ns()
    run_id = "energy-run-" + uuid.uuid4().hex
    counter = worker = None
    phases = []
    runtime_python = {"version": platform.python_version(), "numpy_version": np.__version__,
                      "executable_sha256": sha256(Path(sys.executable).read_bytes()).hexdigest()}
    implementation = code_identity()
    with (directory / "journal.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        def event(kind, value):
            journal.write(json.dumps({"event": kind, "value": value}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        def now():
            return time.perf_counter_ns() - origin

        def reading(phase):
            row = deepcopy(counter.read())
            for key in ("read_start_ns", "read_end_ns"):
                row[key] -= origin
            for key in ("utc_start_ns", "utc_end_ns", "energy_mj"):
                if row[key] is not None:
                    row[key] = str(row[key])
            phase["samples"].append(row)
            event("sample", {"phase": phase["name"], **row})

        def begin(name):
            phase = {"name": name, "start_ns": 0, "end_ns": 0, "samples": [], "batches": [], "error": None}
            phases.append(phase)
            reading(phase)
            phase["start_ns"] = now()
            event("phase_start", {"name": name, "start_ns": phase["start_ns"]})
            return phase

        def finish(phase):
            phase["end_ns"] = now()
            reading(phase)
            event("phase_end", {"name": phase["name"], "end_ns": phase["end_ns"], "error": phase["error"]})

        def batch(phase):
            start = now()
            result = worker.solve()
            end = now()
            reading(phase)
            encoded = energy_records.encode_result(result)
            row = {"batch_index": len(phase["batches"]), "start_ns": start, "end_ns": end, "result": encoded}
            phase["batches"].append(row)
            event("batch", {"phase": phase["name"], **row})

        def idle(name):
            phase = begin(name)
            deadline = time.perf_counter() + plan["idle_duration_s"]
            while (remaining := deadline - time.perf_counter()) > 0:
                time.sleep(min(.1, remaining))
                reading(phase)
            finish(phase)

        try:
            clock = {"epoch_id": uuid.uuid4().hex, "monotonic_origin_ns": str(origin),
                     "implementation": time.get_clock_info("perf_counter").implementation,
                     "resolution_s": time.get_clock_info("perf_counter").resolution, "utc_unit": "unix_ns_decimal_string"}
            event("capture", {"schema": energy_records.SCHEMA, "run_id": run_id, "origin": "physical_measurement",
                              "clock": clock, "plan": plan, "python": runtime_python, "implementation": implementation})
            counter = counter_factory(device_index=device_index)
            sensor = counter.identity()
            event("sensor", sensor)
            phase = begin("startup")
            worker = worker_factory(plan["problem"], plan["solver"], plan["iterations"],
                                    replicas=plan["replicas"], device_index=device_index)
            finish(phase)
            workload = worker.identity()
            event("workload", workload)
            if workload["device_uuid"].lower() != sensor["device_uuid"].lower():
                raise ValueError("CUDA and NVML device indices selected different UUIDs")
            phase = begin("warmup")
            for _ in range(plan["warmup_batches"]):
                batch(phase)
            finish(phase)
            idle("idle_before")
            phase = begin("measurement")
            for _ in range(plan["max_batches"]):
                batch(phase)
                if (now() - phase["start_ns"]) / 1e9 >= plan["minimum_duration_s"]:
                    break
            finish(phase)
            idle("idle_after")
            log = energy_records.seal({"schema": energy_records.SCHEMA, "run_id": run_id, "origin": "physical_measurement",
                "clock": clock, "sensor": sensor, "runtime": {"python": runtime_python, "implementation": implementation,
                "workload": workload}, "plan": plan, "phases": phases})
            report = energy_records.analyze(log)
            _publish(directory / "log.json", log)
            _publish(directory / "report.json", report)
            event("complete", {"log_digest": log["log_digest"]})
            return log, report
        except BaseException as exc:
            if counter is not None and phases and phases[-1]["end_ns"] == 0:
                phase = phases[-1]
                phase["error"] = (type(exc).__name__ + ": " + str(exc))[:1024]
                try:
                    finish(phase)
                except Exception as closing:
                    event("failure_endpoint_unavailable", {"phase": phase["name"], "error": str(closing)[:4096]})
            failure = {"schema": "ciw.energy-capture-failure.v1", "run_id": run_id,
                       "error_type": type(exc).__name__, "error": str(exc)[:4096], "phases": phases}
            event("failure", failure)
            _publish(directory / "failure.json", failure)
            raise
        finally:
            # Close both resources even if the first close fails. Cleanup
            # errors are visible in the journal without masking capture errors.
            for name, resource in (("worker", worker), ("counter", counter)):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        event("cleanup_error", {"resource": name, "error": str(exc)[:4096]})


def probe(device_index=0):
    with NVMLEnergyCounter(device_index=device_index) as counter:
        return {"sensor": counter.identity(), "reading": counter.read()}
