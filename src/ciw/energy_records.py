"""Bounded NVML-counter logs and hardware-free energy/accuracy derivation.

Sealing establishes internal integrity, not hardware or execution authenticity.
Only gross counter differences are supported; no modulo correction, power
integration, idle subtraction, or first-attainment claim is made.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import math
import re

import numpy as np

from . import free_energy_math as mathematics
from .telemetry import canonical, digest

SCHEMA = "ciw.energy-accuracy-log.v1"
RESULT_ENCODING = "ciw.constant-float64-row.v1"
PHASE_NAMES = ("startup", "warmup", "idle_before", "measurement", "idle_after")
MAX_NS = 3600 * 10**9
MAX_BYTES = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]{0,19})\Z")
_CONTEXT = {"power_mw", "temperature_c", "graphics_clock_mhz"}
_LAYOUT = ["mean_0", "mean_1", "covariance_00", "covariance_01", "covariance_10", "covariance_11"]


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Require exactly the declared energy-log fields")


def _text(value, limit=256):
    if type(value) is not str or not 1 <= len(value) <= limit:
        raise ValueError("Require bounded nonempty text")


def _integer(value, lower=0, upper=MAX_NS):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Integer outside the energy-log profile")
    return value


def _number(value, lower=0, upper=1e12, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError("Require a finite real number")
    try:
        valid = math.isfinite(value) and lower <= value <= upper
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise ValueError("Number outside the energy-log profile")
    return float(value)


def _decimal(value, upper=2**64 - 1):
    if type(value) is not str or not _DECIMAL.fullmatch(value) or int(value) > upper:
        raise ValueError("Require a canonical unsigned decimal string")
    return int(value)


def _hash(value):
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ValueError("Require a lowercase SHA256 identity")


def _expanded_digest(row, replicas):
    # Stream the repeated row; validation never expands a multi-megabyte batch.
    raw = np.asarray(row, dtype="<f8").tobytes(order="C")
    return _repeated_bytes_digest(raw, replicas)


@lru_cache(maxsize=256)
def _repeated_bytes_digest(raw, replicas):
    h = sha256()
    block = raw * min(replicas, 1024)
    whole, remainder = divmod(replicas, min(replicas, 1024))
    for _ in range(whole):
        h.update(block)
    h.update(raw * remainder)
    return "sha256:" + h.hexdigest()


def encode_result(array):
    """Losslessly encode a finite binary64 N-by-6 array of identical rows.

    Signed-zero differences are significant. The expanded digest is of the
    C-order little-endian float64 bytes, independent of host byte order.
    """
    if (type(array) is not np.ndarray or array.dtype.kind != "f" or array.dtype.itemsize != 8 or
            array.ndim != 2 or array.shape[1] != 6 or not 1 <= array.shape[0] <= 65536):
        raise ValueError("Require a binary64 N-by-6 batch with 1..65536 replicas")
    if not np.all(np.isfinite(array)):
        raise ValueError("Cannot encode nonfinite GPU results")
    little = np.ascontiguousarray(array, dtype="<f8")
    bits = little.view("<u8")
    if not np.all(bits == bits[0]):
        raise ValueError("Constant-row profile requires bitwise-identical replicas")
    row = little[0].tolist()
    result = {"encoding": RESULT_ENCODING, "replicas": len(array), "values": row,
              "expanded_sha256": _expanded_digest(row, len(array))}
    _result(result, len(array))
    return result


def _result(result, replicas):
    _keys(result, {"encoding", "replicas", "values", "expanded_sha256"})
    if result["encoding"] != RESULT_ENCODING or type(result["replicas"]) is not int or result["replicas"] != replicas:
        raise ValueError("Result encoding or replica count differs")
    row = result["values"]
    if type(row) is not list or len(row) != 6:
        raise ValueError("Require exactly six retained binary64 values")
    for value in row:
        _number(value, -1e20, 1e20)
    mean = mathematics._array(row[:2], (2,), "retained mean", bound=1e12)
    covariance = np.asarray(row[2:], dtype=np.float64).reshape(2, 2)
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("Retained full covariance must be exactly symmetric")
    covariance = mathematics._positive(covariance, "retained covariance")
    if result["expanded_sha256"] != _expanded_digest(row, replicas):
        raise ValueError("Expanded result bytes differ from their digest")
    return mean, covariance


def _plan(plan):
    _keys(plan, {"problem", "solver", "iterations", "replicas", "target_kl_nats", "minimum_duration_s",
                 "max_batches", "warmup_batches", "idle_duration_s", "problem_digest", "model_digest", "observations_digest"})
    mathematics._problem(plan["problem"])
    # Validate the reference's numerical domain before considering measurements.
    mathematics.gaussian_reference(plan["problem"])
    solver = plan["solver"]
    _keys(solver, {"initial_mean", "initial_covariance", "alpha", "beta", "max_iterations",
                   "gradient_tolerance", "precision_tolerance"})
    mathematics._array(solver["initial_mean"], (2,), "initial_mean")
    mathematics._covariance(solver["initial_covariance"], "initial_covariance", input_matrix=True)
    _number(solver["alpha"], upper=1e6, positive=True)
    if not 0 < _number(solver["beta"], upper=1) < 1:
        raise ValueError("Require strict precision relaxation")
    _integer(solver["max_iterations"], 1, 256)
    for name in ("gradient_tolerance", "precision_tolerance"):
        _number(solver[name], 1e-12, 1e-3)
    _integer(plan["iterations"], 0, solver["max_iterations"])
    _integer(plan["replicas"], 1, 65536)
    _number(plan["target_kl_nats"], upper=1e6, positive=True)
    _number(plan["minimum_duration_s"], 1, 30)
    _integer(plan["max_batches"], 1, 2048)
    _integer(plan["warmup_batches"], 1, 8)
    _number(plan["idle_duration_s"], .2, 10)
    model = {key: value for key, value in plan["problem"].items() if key != "observations"}
    for name, value in (("problem_digest", plan["problem"]), ("model_digest", model),
                        ("observations_digest", plan["problem"]["observations"])):
        if plan[name] != digest(value):
            raise ValueError("Problem/model/observation digest differs")


def _sensor(sensor):
    _keys(sensor, {"backend", "device_uuid", "name", "driver_version", "nvml_version", "library_sha256",
                   "counter_unit", "counter_scope", "power_unit", "timestamp_semantics", "update_interval_s",
                   "resolution_j", "accuracy_j", "background_inclusive"})
    fixed = {"backend": "nvml", "counter_unit": "mJ", "counter_scope": "gpu_device", "power_unit": "mW",
             "timestamp_semantics": "host_call_brackets", "background_inclusive": True,
             "update_interval_s": None, "resolution_j": None, "accuracy_j": None}
    for name, value in fixed.items():
        if type(sensor[name]) is not type(value) or sensor[name] != value:
            raise ValueError("Unsupported sensor scope or invented calibration metadata")
    if type(sensor["device_uuid"]) is not str or not _UUID.fullmatch(sensor["device_uuid"]):
        raise ValueError("Require a physical GPU UUID")
    for name in ("name", "driver_version", "nvml_version"):
        _text(sensor[name])
    _hash(sensor["library_sha256"])


def _runtime(runtime, plan):
    _keys(runtime, {"workload", "python", "implementation"})
    python = runtime["python"]
    _keys(python, {"version", "numpy_version", "executable_sha256"})
    _text(python["version"])
    _text(python["numpy_version"])
    _hash(python["executable_sha256"])
    implementation = runtime["implementation"]
    _keys(implementation, {"profile", "code_sha256"})
    _text(implementation["profile"])
    _hash(implementation["code_sha256"])
    work = runtime["workload"]
    _keys(work, {"schema", "execution_device", "device_index", "device_name", "device_uuid", "compute_capability",
                 "cuda_driver_version", "kernel_sha256", "ptx_version", "ptx_target", "arithmetic", "algorithm",
                 "replica_semantics", "iterations", "replicas", "threads_per_block", "output_layout",
                 "prepared_input_sha256", "problem_sha256", "solver_settings", "measurement_boundary"})
    fixed = {"schema": "ciw.cuda-gaussian-worker.v1", "execution_device": "gpu", "ptx_version": "6.0",
             "ptx_target": "sm_50", "arithmetic": "binary64_explicit_round_to_nearest_no_fma_contraction",
             "algorithm": "fixed_step_mean_gradient_and_full_precision_relaxation",
             "replica_semantics": "identical_declared_problem_independent_thread_work", "threads_per_block": 128,
             "output_layout": _LAYOUT,
             "measurement_boundary": "constructor_preparation_and_jit_excluded;solve_includes_launch_sync_copy_and_output_check"}
    for name, value in fixed.items():
        if canonical(work[name]) != canonical(value):
            raise ValueError("Unsupported CUDA workload profile")
    _integer(work["device_index"], 0, 63)
    _text(work["device_name"])
    if type(work["device_uuid"]) is not str or not _UUID.fullmatch(work["device_uuid"]):
        raise ValueError("Require a workload GPU UUID")
    if type(work["compute_capability"]) is not list or len(work["compute_capability"]) != 2:
        raise ValueError("Require CUDA compute capability")
    _integer(work["compute_capability"][0], 5, 100)
    _integer(work["compute_capability"][1], 0, 100)
    _integer(work["cuda_driver_version"], 1, 10**8)
    for name in ("kernel_sha256", "prepared_input_sha256", "problem_sha256"):
        _hash(work[name])
    for name in ("iterations", "replicas"):
        if canonical(work[name]) != canonical(plan[name]):
            raise ValueError("Workload differs from declared iteration/replica plan")
    if canonical(work["solver_settings"]) != canonical(plan["solver"]):
        raise ValueError("Workload solver differs from plan")
    if work["problem_sha256"] != plan["problem_digest"][7:]:
        raise ValueError("Workload problem differs from retained problem")
    # Reconstruct only the initial device input, not the GPU iteration.
    info = mathematics.information_system(plan["problem"])
    solver = plan["solver"]
    q = mathematics._positive(np.linalg.solve(np.asarray(solver["initial_covariance"]), np.eye(2)), "initial precision")
    values = [*np.asarray(info["precision"]).ravel(), *info["information_vector"], *solver["initial_mean"],
              *q.ravel(), solver["alpha"], solver["beta"], 1 - solver["beta"]]
    if sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest() != work["prepared_input_sha256"]:
        raise ValueError("Prepared device input differs from declared initial problem")


def _sample(sample):
    _keys(sample, {"read_start_ns", "read_end_ns", "utc_start_ns", "utc_end_ns", "status", "energy_mj",
                   "error_code", "power_mw", "temperature_c", "graphics_clock_mhz", "context_errors"})
    if _integer(sample["read_start_ns"]) > _integer(sample["read_end_ns"]):
        raise ValueError("Reversed monotonic hardware call bracket")
    _decimal(sample["utc_start_ns"])
    _decimal(sample["utc_end_ns"])
    if sample["status"] == "ok":
        _decimal(sample["energy_mj"])
        if sample["error_code"] is not None:
            raise ValueError("Successful counter reading cannot have an error code")
    elif sample["status"] == "error":
        if sample["energy_mj"] is not None:
            raise ValueError("Failed counter reading cannot contain fabricated energy")
        _integer(sample["error_code"], 1, 2**32 - 1)
    else:
        raise ValueError("Unknown counter reading status")
    context = sample["context_errors"]
    if type(context) is not dict or not set(context) <= _CONTEXT:
        raise ValueError("Unknown context telemetry errors")
    for name in _CONTEXT:
        if sample[name] is None:
            if name not in context:
                raise ValueError("Missing context value requires its raw error code")
            _integer(context[name], 1, 2**32 - 1)
        elif name in context:
            raise ValueError("Context value and failure cannot coexist")
        else:
            _integer(sample[name], 0, 2**32 - 1)


def _validate(log, *, sealed):
    _keys(log, {"schema", "run_id", "origin", "clock", "sensor", "runtime", "plan", "phases"} |
          ({"log_digest"} if sealed else set()))
    if log["schema"] != SCHEMA or type(log["run_id"]) is not str or not re.fullmatch(r"energy-run-[0-9a-f]{32}", log["run_id"]):
        raise ValueError("Unsupported energy log schema or occurrence identity")
    if log["origin"] not in ("physical_measurement", "synthetic_fixture"):
        raise ValueError("Declare actual measurement versus synthetic fixture")
    clock = log["clock"]
    _keys(clock, {"epoch_id", "monotonic_origin_ns", "implementation", "resolution_s", "utc_unit"})
    _text(clock["epoch_id"], 128)
    _decimal(clock["monotonic_origin_ns"])
    _text(clock["implementation"])
    _number(clock["resolution_s"], upper=1, positive=True)
    if clock["utc_unit"] != "unix_ns_decimal_string":
        raise ValueError("UTC timestamp representation differs")
    _plan(log["plan"])
    _sensor(log["sensor"])
    _runtime(log["runtime"], log["plan"])
    phases = log["phases"]
    if type(phases) is not list or len(phases) != len(PHASE_NAMES):
        raise ValueError("Require all five ordered measurement phases")
    previous_end, previous_read = 0, 0
    for expected, phase in zip(PHASE_NAMES, phases):
        _keys(phase, {"name", "start_ns", "end_ns", "samples", "batches", "error"})
        start, end = _integer(phase["start_ns"]), _integer(phase["end_ns"])
        if phase["name"] != expected or not previous_end <= start <= end:
            raise ValueError("Phase names/order or nonoverlap invariant differs")
        previous_end = end
        if phase["error"] is not None:
            _text(phase["error"], 2048)
        samples, batches = phase["samples"], phase["batches"]
        if type(samples) is not list or len(samples) > 2050 or type(batches) is not list or len(batches) > 2048:
            raise ValueError("Phase exceeds retained counter/batch budget")
        for sample in samples:
            _sample(sample)
            if sample["read_start_ns"] < previous_read:
                raise ValueError("Hardware sample order overlaps or goes backwards")
            previous_read = sample["read_end_ns"]
        previous_batch_end = start
        for index, batch in enumerate(batches):
            _keys(batch, {"batch_index", "start_ns", "end_ns", "result"})
            a, b = _integer(batch["start_ns"]), _integer(batch["end_ns"])
            if type(batch["batch_index"]) is not int or batch["batch_index"] != index or not previous_batch_end <= a < b <= end:
                raise ValueError("Batch order or measured phase containment differs")
            previous_batch_end = b
            _result(batch["result"], log["plan"]["replicas"])
        if expected not in ("warmup", "measurement") and batches:
            raise ValueError("Only warmup and measurement phases execute batches")
        limit = log["plan"]["warmup_batches"] if expected == "warmup" else log["plan"]["max_batches"]
        if len(batches) > limit:
            raise ValueError("Executed batches exceed declared budget")
    if sealed:
        unsigned = {key: value for key, value in log.items() if key != "log_digest"}
        if log["log_digest"] != digest(unsigned):
            raise ValueError("Retained log digest differs")


def validate_log(log):
    """Validate bounded structure and identities; failed measurements may persist."""
    try:
        if len(canonical(log)) > MAX_BYTES:
            raise ValueError("Energy log exceeds its byte budget")
        _validate(log, sealed=True)
    except (TypeError, OverflowError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed or unsupported energy log") from exc


def seal(log_without_log_digest):
    """Return a detached validated log; do not repair or reseal an existing one."""
    try:
        if len(canonical(log_without_log_digest)) > MAX_BYTES - 100:
            raise ValueError("Energy log exceeds its byte budget")
        _validate(log_without_log_digest, sealed=False)
        result = deepcopy(log_without_log_digest)
        result["log_digest"] = digest(result)
        return result
    except (TypeError, OverflowError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed or unsupported energy log") from exc


def _phase_analysis(phase, plan, reference):
    samples, batches, reasons = phase["samples"], phase["batches"], []
    if phase["error"] is not None:
        reasons.append("phase_error")
    bracketed = (len(samples) >= 2 and samples[0]["read_end_ns"] <= phase["start_ns"] and
                 samples[-1]["read_start_ns"] >= phase["end_ns"])
    if not bracketed:
        reasons.append("missing_endpoint_brackets")
    if any(sample["status"] != "ok" for sample in samples):
        reasons.append("counter_read_error")
    energies = [int(sample["energy_mj"]) for sample in samples if sample["status"] == "ok"]
    if any(b < a for a, b in zip(energies, energies[1:])):
        reasons.append("reset_or_wrap_ambiguous")
    # One retained intermediate reading per completed batch, before the next
    # batch starts. The final endpoint sample is additional, not a substitute.
    if batches:
        intermediate = samples[1:-1]
        if len(intermediate) != len(batches) or any(
                not (batch["end_ns"] <= sample["read_start_ns"] <= sample["read_end_ns"] <=
                     (batches[index+1]["start_ns"] if index+1 < len(batches) else phase["end_ns"]))
                for index, (batch, sample) in enumerate(zip(batches, intermediate))):
            reasons.append("missing_or_misaligned_batch_counter_read")
    delta = energies[-1] - energies[0] if bracketed and not reasons else None
    if delta == 0:
        reasons.append("below_resolution_or_no_reported_increment")
    elapsed = (phase["end_ns"] - phase["start_ns"]) / 1e9
    bracket_elapsed = (samples[-1]["read_end_ns"] - samples[0]["read_start_ns"]) / 1e9 if bracketed else None
    metrics = []
    for batch in batches:
        mean, covariance = _result(batch["result"], plan["replicas"])
        kl = mathematics.gaussian_kl(mean, covariance, reference["mean"], reference["covariance"])
        metrics.append({"batch_index": batch["batch_index"], "kl_nats": kl,
            "mean_max_abs_error": float(np.max(np.abs(mean - reference["mean"]))),
            "covariance_max_abs_error": float(np.max(np.abs(covariance - reference["covariance"]))),
            "target_met": kl <= plan["target_kl_nats"], "replicas": plan["replicas"]})
    utc_reversal = any(int(s["utc_end_ns"]) < int(s["utc_start_ns"]) for s in samples) or any(
        int(b["utc_start_ns"]) < int(a["utc_end_ns"]) for a, b in zip(samples, samples[1:]))
    return {"name": phase["name"], "elapsed_s": elapsed, "bracket_elapsed_s": bracket_elapsed,
            "alignment_overhead_s": None if bracket_elapsed is None else bracket_elapsed-elapsed,
            "counter_delta_mj": None if delta is None else str(delta),
            "gross_energy_j": None if delta is None else delta/1000,
            "energy_status": "ok" if not reasons else reasons[0], "reasons": reasons,
            "utc_reversal_detected": utc_reversal, "batches": metrics}


def analyze(log):
    """Recompute Gaussian accuracy and gross energy from retained data only."""
    validate_log(log)
    plan = log["plan"]
    reference = mathematics.gaussian_reference(plan["problem"])
    phases = [_phase_analysis(phase, plan, reference) for phase in log["phases"]]
    measured = phases[3]
    metrics = measured["batches"]
    target_met = bool(metrics) and all(batch["target_met"] for batch in metrics)
    duration_met = measured["elapsed_s"] >= plan["minimum_duration_s"]
    same_device = log["sensor"]["device_uuid"].lower() == log["runtime"]["workload"]["device_uuid"].lower()
    reasons = list(measured["reasons"])
    counters = [int(sample["energy_mj"]) for phase in log["phases"] for sample in phase["samples"]
                if sample["status"] == "ok"]
    continuity_lost = any(b < a for a, b in zip(counters, counters[1:]))
    if continuity_lost:
        reasons.append("observed_counter_continuity_loss")
    if not target_met:
        reasons.append("accuracy_target_not_met_by_every_batch")
    if not duration_met:
        reasons.append("minimum_measurement_duration_not_met")
    if not same_device:
        reasons.append("workload_sensor_device_mismatch")
    if len(log["phases"][1]["batches"]) != plan["warmup_batches"] or log["phases"][1]["error"] is not None:
        reasons.append("declared_warmup_not_completed")
    if any(phase["error"] is not None for phase in log["phases"]):
        if "phase_error" not in reasons:
            reasons.append("phase_error")
    for phase in phases:
        if phase["name"] != "measurement" and any(
                reason != "below_resolution_or_no_reported_increment" for reason in phase["reasons"]):
            reasons.append(phase["name"] + "_measurement_invalid")
        if phase["name"] in ("idle_before", "idle_after") and phase["elapsed_s"] < plan["idle_duration_s"]:
            reasons.append(phase["name"] + "_duration_not_met")
    total = len(metrics) * plan["replicas"]
    eligible = not reasons
    measurement = {"batch_count": len(metrics), "total_solves": total,
        "qualified_solves": sum(row["replicas"] for row in metrics if row["target_met"]),
        "target_kl_nats": plan["target_kl_nats"], "target_met": target_met,
        "max_kl_nats": max((row["kl_nats"] for row in metrics), default=None),
        "max_mean_abs_error": max((row["mean_max_abs_error"] for row in metrics), default=None),
        "max_covariance_abs_error": max((row["covariance_max_abs_error"] for row in metrics), default=None),
        "minimum_duration_met": duration_met, "gross_energy_j": measured["gross_energy_j"],
        "amortized_domain_energy_j_per_qualified_solve": measured["gross_energy_j"]/total if eligible else None}
    return {"schema": "ciw.energy-accuracy-analysis.v1", "log_digest": log["log_digest"], "origin": log["origin"],
        "measurement_scope": {"domain": "gpu_device", "device_uuid": log["sensor"]["device_uuid"],
            "workload_device_matches": same_device, "background_inclusive": True,
            "estimator": "gross_counter_difference_over_host_call_brackets", "idle_subtracted": False,
            "counter_continuity": "observed_decrease" if continuity_lost else "no_observed_decrease",
            "update_interval_s": None, "resolution_j": None, "accuracy_j": None},
        "hardware_provenance": "retained_operator_record_not_authenticated" if log["origin"] == "physical_measurement" else "synthetic_fixture_no_physical_verification",
        "reference": reference, "phases": phases, "measurement": measurement,
        "comparison": {"eligible": eligible and log["origin"] == "physical_measurement",
            "classification": ("physical_domain_measurement" if log["origin"] == "physical_measurement" else "synthetic_only") if eligible else "ineligible",
            "reasons": reasons, "accuracy_basis": "all_retained_fixed_iteration_batch_outputs_meet_same_declared_KL_target",
            "claim": "amortized_background_inclusive_energy_not_first_attainment_or_minimum_energy"}}
