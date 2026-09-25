"""Every check of the energy log validator refuses a log that is otherwise the valid example.

An AST mutation probe dropped each ``if ...: raise`` of ``energy_records`` in
turn; 51 of 75 survived because the suite only reached the structural
refusals. Each case here starts from the sealed example log, changes exactly
one thing, and expects the refusal that names it, so a dropped check either
accepts the log or fails with a different message.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw import energy_records

ROOT = Path(__file__).resolve().parents[1]
SEALED = json.loads((ROOT / "examples" / "energy-accuracy" / "baseline.json").read_text(encoding="utf-8"))
NAMES = ("startup", "warmup", "idle_before", "measurement", "idle_after")


def unsealed():
    log = deepcopy(SEALED)
    del log["log_digest"]
    return log


def phase(log, name):
    return log["phases"][NAMES.index(name)]


def sample(log):
    return phase(log, "startup")["samples"][0]


def batch(log):
    return phase(log, "warmup")["batches"][0]


def work(log):
    return log["runtime"]["workload"]


def more_batches(log, count):
    measurement = phase(log, "measurement")
    template = measurement["batches"][0]
    step = (measurement["end_ns"] - measurement["start_ns"]) // count
    assert step >= 2
    batches = []
    for index in range(count):
        item = deepcopy(template)
        item["batch_index"] = index
        item["start_ns"] = measurement["start_ns"] + index * step
        item["end_ns"] = item["start_ns"] + step // 2
        batches.append(item)
    measurement["batches"] = batches


def more_samples(log, count):
    startup = phase(log, "startup")
    last = startup["samples"][-1]
    extra = deepcopy(last)
    extra["read_start_ns"] = last["read_end_ns"]
    startup["samples"] += [deepcopy(extra) for _ in range(count - len(startup["samples"]))]


def set_in(target, key, value):
    def apply(log):
        target(log)[key] = value
    return apply


def clock(log):
    return log["clock"]


def sensor(log):
    return log["sensor"]


def plan(log):
    return log["plan"]


def solver(log):
    return log["plan"]["solver"]


def result(log):
    return batch(log)["result"]


def startup_batch(log):
    startup = phase(log, "startup")
    item = deepcopy(batch(log))
    item["start_ns"], item["end_ns"] = startup["start_ns"], startup["start_ns"] + 1000
    startup["batches"] = [item]


def asymmetric(log):
    result(log)["values"][3] += 1e-6


def two_batches_one_allowed(log):
    more_batches(log, 2)
    log["plan"]["max_batches"] = 1


def top(log):
    return log


CASES = {
    "object field given as a list of its names": (set_in(top, "clock", ["epoch_id", "monotonic_origin_ns", "implementation", "resolution_s", "utc_unit"]), "declared energy-log fields"),
    "text field given a number": (set_in(sensor, "name", 7), "bounded nonempty text"),
    "text field empty": (set_in(sensor, "name", ""), "bounded nonempty text"),
    "integer field given a float": (set_in(plan, "max_batches", 2.0), "Integer outside"),
    "integer field above its bound": (set_in(plan, "max_batches", 4096), "Integer outside"),
    "number field given text": (set_in(clock, "resolution_s", "1e-9"), "finite real number"),
    "number field above its bound": (set_in(clock, "resolution_s", 2.0), "Number outside"),
    "decimal field given an integer": (set_in(clock, "monotonic_origin_ns", 123), "canonical unsigned decimal string"),
    "digest field given a number": (set_in(sensor, "library_sha256", 7), "lowercase SHA256 identity"),
    "digest field not hexadecimal": (set_in(sensor, "library_sha256", "Z" * 64), "lowercase SHA256 identity"),
    "result encoding renamed": (set_in(result, "encoding", "other"), "Result encoding or replica count differs"),
    "result replicas given as a float": (set_in(result, "replicas", float(SEALED["plan"]["replicas"])), "Result encoding or replica count differs"),
    "result values not a list": (set_in(result, "values", tuple(SEALED["phases"][1]["batches"][0]["result"]["values"])), "exactly six retained binary64 values"),
    "result values with a seventh entry": (set_in(result, "values", SEALED["phases"][1]["batches"][0]["result"]["values"] + [0.0]), "exactly six retained binary64 values"),
    "result covariance asymmetric": (asymmetric, "exactly symmetric"),
    "precision relaxation not strict": (set_in(solver, "beta", 1), "strict precision relaxation"),
    "sensor boolean given as an integer": (set_in(sensor, "background_inclusive", 1), "Unsupported sensor scope"),
    "sensor uuid given a number": (set_in(sensor, "device_uuid", 7), "physical GPU UUID"),
    "workload profile field changed": (set_in(work, "ptx_version", "7.0"), "Unsupported CUDA workload profile"),
    "workload uuid given a number": (set_in(work, "device_uuid", 7), "workload GPU UUID"),
    "workload uuid malformed": (set_in(work, "device_uuid", "GPU-nope"), "workload GPU UUID"),
    "compute capability given as text": (set_in(work, "compute_capability", "50"), "CUDA compute capability"),
    "compute capability with one entry": (set_in(work, "compute_capability", [5]), "CUDA compute capability"),
    "workload problem digest differs": (set_in(work, "problem_sha256", "c" * 64), "Workload problem differs"),
    "successful sample with an error code": (set_in(sample, "error_code", 5), "cannot have an error code"),
    "context errors given as a list": (set_in(sample, "context_errors", []), "Unknown context telemetry errors"),
    "context errors with an unknown name": (set_in(sample, "context_errors", {"fan": 1}), "Unknown context telemetry errors"),
    "missing context value without its error": (set_in(sample, "power_mw", None), "requires its raw error code"),
    "context value beside its error": (set_in(sample, "context_errors", {"power_mw": 3}), "cannot coexist"),
    "schema renamed": (set_in(top, "schema", "ciw.other.v1"), "Unsupported energy log schema"),
    "run id given a number": (set_in(top, "run_id", 7), "Unsupported energy log schema"),
    "run id malformed": (set_in(top, "run_id", "energy-run-nope"), "Unsupported energy log schema"),
    "utc unit renamed": (set_in(clock, "utc_unit", "iso8601"), "UTC timestamp representation differs"),
    "phases not a list": (set_in(top, "phases", tuple(SEALED["phases"])), "all five ordered measurement phases"),
    "four phases": (set_in(top, "phases", SEALED["phases"][:4]), "all five ordered measurement phases"),
    "samples not a list": (lambda log: phase(log, "startup").__setitem__("samples", tuple(phase(log, "startup")["samples"])), "counter/batch budget"),
    "more than 2050 samples": (lambda log: more_samples(log, 2051), "counter/batch budget"),
    "batches not a list": (lambda log: phase(log, "warmup").__setitem__("batches", tuple(phase(log, "warmup")["batches"])), "counter/batch budget"),
    "more than 2048 batches": (lambda log: more_batches(log, 2049), "counter/batch budget"),
    "batch index given as a float": (lambda log: batch(log).__setitem__("batch_index", 0.0), "Batch order or measured phase containment"),
    "batch inside the startup phase": (startup_batch, "Only warmup and measurement phases execute batches"),
    "more batches than the plan allows": (two_batches_one_allowed, "Executed batches exceed declared budget"),
}


def test_the_example_log_reseals_to_its_retained_digest():
    assert energy_records.seal(unsealed())["log_digest"] == SEALED["log_digest"]
    energy_records.validate_log(deepcopy(SEALED))


@pytest.mark.parametrize("mutate, message", CASES.values(), ids=list(CASES))
def test_one_change_to_the_example_log_is_refused_by_its_own_check(mutate, message):
    log = unsealed()
    mutate(log)
    with pytest.raises(ValueError, match=message):
        energy_records.seal(log)


def test_the_byte_budget_is_enforced_before_validation(monkeypatch):
    monkeypatch.setattr(energy_records, "MAX_BYTES", 1024)
    with pytest.raises(ValueError, match="byte budget"):
        energy_records.validate_log(deepcopy(SEALED))
    with pytest.raises(ValueError, match="byte budget"):
        energy_records.seal(unsealed())


@pytest.mark.parametrize("array", [
    [[1.0] * 6], np.ones((2, 6), dtype=np.int64), np.ones(6), np.ones((2, 5)), np.ones((2, 6), dtype=np.float32),
], ids=["list", "integer dtype", "one dimension", "five columns", "binary32"])
def test_encode_result_accepts_only_binary64_n_by_6_arrays(array):
    with pytest.raises(ValueError, match="N-by-6 batch"):
        energy_records.encode_result(array)


def test_encode_result_refuses_nonfinite_rows():
    with pytest.raises(ValueError, match="nonfinite"):
        energy_records.encode_result(np.full((2, 6), np.nan))
