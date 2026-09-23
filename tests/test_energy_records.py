"""Synthetic retained logs exercise replay mathematics, never hardware evidence."""
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from ciw import energy_records as records
from ciw import free_energy_math as mathematics
from ciw.telemetry import digest


def make_log():
    """A small, explicitly synthetic log; values are constructed on the CPU."""
    problem = {"coordinate_system": "normalized_dimensionless", "prior_mean": [0., 0.],
        "prior_covariance": [[1., 0.], [0., 1.]], "observation_matrix": [[1., .2], [0., 1.]],
        "observations": [1., 2.], "noise_covariance": [[1., .1], [.1, 1.]]}
    solver = {"initial_mean": [0., 0.], "initial_covariance": [[1., 0.], [0., 1.]],
        "alpha": .1, "beta": .2, "max_iterations": 256, "gradient_tolerance": 1e-9, "precision_tolerance": 1e-9}
    plan = {"problem": problem, "solver": solver, "iterations": 128, "replicas": 4,
        "target_kl_nats": 1e-8, "minimum_duration_s": 1., "max_batches": 2, "warmup_batches": 1,
        "idle_duration_s": .2, "problem_digest": digest(problem),
        "model_digest": digest({k: v for k, v in problem.items() if k != "observations"}),
        "observations_digest": digest(problem["observations"])}
    reference = mathematics.gaussian_reference(problem)
    row = [*reference["mean"], *np.asarray(reference["covariance"]).ravel()]
    result = records.encode_result(np.tile(row, (plan["replicas"], 1)))
    info = mathematics.information_system(problem)
    q = mathematics._positive(np.linalg.solve(solver["initial_covariance"], np.eye(2)), "initial precision")
    prepared = [*np.asarray(info["precision"]).ravel(), *info["information_vector"], *solver["initial_mean"],
                *q.ravel(), solver["alpha"], solver["beta"], 1-solver["beta"]]
    device_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    workload = {"schema": "ciw.cuda-gaussian-worker.v1", "execution_device": "gpu", "device_index": 0,
        "device_name": "synthetic fixture device", "device_uuid": device_uuid, "compute_capability": [8, 6],
        "cuda_driver_version": 12000, "kernel_sha256": "a"*64, "ptx_version": "6.0", "ptx_target": "sm_50",
        "arithmetic": "binary64_explicit_round_to_nearest_no_fma_contraction",
        "algorithm": "fixed_step_mean_gradient_and_full_precision_relaxation",
        "replica_semantics": "identical_declared_problem_independent_thread_work", "iterations": plan["iterations"],
        "replicas": plan["replicas"], "threads_per_block": 128, "output_layout": list(records._LAYOUT),
        "prepared_input_sha256": sha256(np.asarray(prepared, dtype="<f8").tobytes()).hexdigest(),
        "problem_sha256": plan["problem_digest"][7:], "solver_settings": deepcopy(solver),
        "measurement_boundary": "constructor_preparation_and_jit_excluded;solve_includes_launch_sync_copy_and_output_check"}
    sensor = {"backend": "nvml", "device_uuid": device_uuid, "name": "synthetic fixture device",
        "driver_version": "fixture", "nvml_version": "fixture", "library_sha256": "b"*64,
        "counter_unit": "mJ", "counter_scope": "gpu_device", "power_unit": "mW",
        "timestamp_semantics": "host_call_brackets", "update_interval_s": None, "resolution_j": None,
        "accuracy_j": None, "background_inclusive": True}

    def reading(start, value):
        return {"read_start_ns": start, "read_end_ns": start+10, "utc_start_ns": str(10**18+start),
            "utc_end_ns": str(10**18+start+10), "status": "ok", "energy_mj": str(value), "error_code": None,
            "power_mw": 1000, "temperature_c": 40, "graphics_clock_mhz": 1000, "context_errors": {}}

    phases = []
    cursor, energy = 100, 1000
    for name in records.PHASE_NAMES:
        duration = 10**9 if name == "measurement" else 2*10**8
        end = cursor+duration
        samples = [reading(cursor-20, energy)]
        batches = []
        if name in ("warmup", "measurement"):
            batch_end = end-40
            batches = [{"batch_index": 0, "start_ns": cursor+20, "end_ns": batch_end, "result": deepcopy(result)}]
            samples.append(reading(batch_end+10, energy+100))
        samples.append(reading(end+10, energy+200))
        phases.append({"name": name, "start_ns": cursor, "end_ns": end, "samples": samples,
                       "batches": batches, "error": None})
        cursor, energy = end+100, energy+300
    return records.seal({"schema": records.SCHEMA, "run_id": "energy-run-"+"0"*32, "origin": "synthetic_fixture",
        "clock": {"epoch_id": "synthetic-clock-0", "monotonic_origin_ns": "123456789",
                  "implementation": "synthetic monotonic", "resolution_s": 1e-9, "utc_unit": "unix_ns_decimal_string"},
        "sensor": sensor, "runtime": {"workload": workload,
            "python": {"version": "3.12.0", "numpy_version": np.__version__, "executable_sha256": "c"*64},
            "implementation": {"profile": "synthetic-fixture", "code_sha256": "d"*64}}, "plan": plan, "phases": phases})


def reseal(log):
    unsigned = {key: deepcopy(value) for key, value in log.items() if key != "log_digest"}
    return records.seal(unsigned)


def test_synthetic_replay_reports_accuracy_and_gross_bracket_energy():
    log = make_log()
    original = deepcopy(log)
    report = records.analyze(log)
    assert log == original
    assert report["origin"] == "synthetic_fixture"
    assert report["comparison"]["classification"] == "synthetic_only"
    assert not report["comparison"]["eligible"]
    assert report["measurement"]["target_met"]
    assert report["measurement"]["total_solves"] == 4
    assert report["measurement"]["gross_energy_j"] == .2
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] == .05
    assert report["phases"][3]["alignment_overhead_s"] == pytest.approx(40e-9)
    assert report["measurement_scope"]["resolution_j"] is None
    assert not report["measurement_scope"]["idle_subtracted"]


def test_physical_origin_is_a_declaration_not_authentication():
    log = make_log()
    log["origin"] = "physical_measurement"
    report = records.analyze(reseal(log))
    assert report["comparison"]["eligible"]
    assert report["hardware_provenance"] == "retained_operator_record_not_authenticated"


def test_seal_detaches_and_refuses_existing_seal():
    log = make_log()
    with pytest.raises(ValueError):
        records.seal(log)
    unsigned = {key: value for key, value in log.items() if key != "log_digest"}
    sealed = records.seal(unsigned)
    unsigned["plan"]["problem"]["observations"][0] += 1
    records.validate_log(sealed)


def test_digest_tampering_is_refused():
    log = make_log()
    log["phases"][3]["samples"][-1]["energy_mj"] = "9999"
    with pytest.raises(ValueError, match="digest"):
        records.validate_log(log)


def test_rle_roundtrip_is_little_endian_lossless():
    row = np.asarray([1., -0., 2., .25, .25, 1.])
    a = np.tile(row, (13, 1)).astype(">f8")
    encoded = records.encode_result(a)
    assert encoded["expanded_sha256"] == "sha256:"+sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()
    assert np.signbit(encoded["values"][1])


@pytest.mark.parametrize("change", ["signed_zero", "one_ulp", "float32", "nan", "singular", "asymmetric", "shape", "empty"])
def test_rle_refuses_loss_or_invalid_posterior(change):
    a = np.tile([0., 0., 1., 0., 0., 1.], (2, 1))
    if change == "signed_zero": a[1, 0] = -0.
    elif change == "one_ulp": a[1, 2] = np.nextafter(1., 2.)
    elif change == "float32": a = a.astype(np.float32)
    elif change == "nan": a[:, 0] = np.nan
    elif change == "singular": a[:, 2] = 0
    elif change == "asymmetric": a[:, 3] = .1
    elif change == "shape": a = a[:, :5]
    elif change == "empty": a = a[:0]
    with pytest.raises(ValueError):
        records.encode_result(a)


@pytest.mark.parametrize("change,reason", [
    ("reset", "reset_or_wrap_ambiguous"), ("error", "counter_read_error"),
    ("empty", "missing_endpoint_brackets"), ("no_final", "missing_endpoint_brackets"),
    ("no_batch_read", "missing_or_misaligned_batch_counter_read"),
    ("zero", "below_resolution_or_no_reported_increment"), ("phase_error", "phase_error"),
])
def test_counter_failure_is_retained_but_not_qualified(change, reason):
    log = make_log()
    phase = log["phases"][3]
    if change == "reset": phase["samples"][-1]["energy_mj"] = "1"
    elif change == "error": phase["samples"][1].update(status="error", energy_mj=None, error_code=3)
    elif change == "empty": phase["samples"] = []
    elif change == "no_final": phase["samples"].pop()
    elif change == "no_batch_read": phase["samples"].pop(1)
    elif change == "zero":
        for sample in phase["samples"]: sample["energy_mj"] = "7"
    elif change == "phase_error": phase["error"] = "operator interrupted batch"
    report = records.analyze(reseal(log))
    assert reason in report["phases"][3]["reasons"]
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None
    assert not report["comparison"]["eligible"]
    if change != "zero": assert report["measurement"]["gross_energy_j"] is None
    else: assert report["measurement"]["gross_energy_j"] == 0


def test_context_telemetry_error_does_not_destroy_successful_energy_counter():
    log = make_log()
    for sample in log["phases"][3]["samples"]:
        sample["power_mw"] = None
        sample["context_errors"] = {"power_mw": 3}
    report = records.analyze(reseal(log))
    assert report["measurement"]["gross_energy_j"] == .2
    assert report["comparison"]["classification"] == "synthetic_only"


def test_utc_reversal_does_not_change_monotonic_energy():
    log = make_log()
    sample = log["phases"][3]["samples"][-1]
    sample["utc_start_ns"] = "1"
    sample["utc_end_ns"] = "2"
    report = records.analyze(reseal(log))
    assert report["phases"][3]["utc_reversal_detected"]
    assert report["measurement"]["gross_energy_j"] == .2
    assert report["measurement"]["minimum_duration_met"]


def test_wrong_device_prevents_attribution():
    log = make_log()
    log["sensor"]["device_uuid"] = "GPU-99999999-89ab-cdef-0123-456789abcdef"
    report = records.analyze(reseal(log))
    assert "workload_sensor_device_mismatch" in report["comparison"]["reasons"]
    assert report["measurement"]["gross_energy_j"] == .2
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None


def test_cross_phase_counter_reset_prevents_comparison():
    log = make_log()
    for sample in log["phases"][3]["samples"]:
        sample["energy_mj"] = str(int(sample["energy_mj"])-1000)
    report = records.analyze(reseal(log))
    assert report["measurement_scope"]["counter_continuity"] == "observed_decrease"
    assert "observed_counter_continuity_loss" in report["comparison"]["reasons"]
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None


def test_cross_phase_sample_reuse_is_refused():
    log = make_log()
    log["phases"][2]["samples"][-1] = deepcopy(log["phases"][3]["samples"][0])
    with pytest.raises(ValueError, match="sample order"):
        reseal(log)


def test_large_counter_delta_is_subtracted_before_float_conversion():
    log = make_log()
    for phase in log["phases"]:
        for sample in phase["samples"]:
            sample["energy_mj"] = str(int(sample["energy_mj"])+2**63)
    assert records.analyze(reseal(log))["measurement"]["gross_energy_j"] == .2


def test_every_batch_must_meet_target_and_failed_runs_retain_consumption():
    log = make_log()
    phase = log["phases"][3]
    row = np.asarray(phase["batches"][0]["result"]["values"])
    row[0] += .1
    phase["batches"][0]["result"] = records.encode_result(np.tile(row, (4, 1)))
    report = records.analyze(reseal(log))
    assert report["measurement"]["max_kl_nats"] > log["plan"]["target_kl_nats"]
    assert not report["measurement"]["target_met"]
    assert report["measurement"]["qualified_solves"] == 0
    assert report["measurement"]["gross_energy_j"] == .2
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None


def test_partial_accuracy_success_is_not_silently_averaged_over_passing_batches():
    log = make_log()
    phase = log["phases"][3]
    middle = (phase["start_ns"]+phase["end_ns"])//2
    first = deepcopy(phase["batches"][0])
    first["end_ns"] = middle-20
    phase["batches"][0].update(batch_index=1, start_ns=middle+20)
    phase["batches"].insert(0, first)
    read = deepcopy(phase["samples"][1])
    read.update(read_start_ns=middle-10, read_end_ns=middle, energy_mj="1950")
    phase["samples"].insert(1, read)
    row = np.asarray(phase["batches"][1]["result"]["values"])
    row[0] += .1
    phase["batches"][1]["result"] = records.encode_result(np.tile(row, (4, 1)))
    report = records.analyze(reseal(log))
    assert report["measurement"]["qualified_solves"] == 4
    assert report["measurement"]["total_solves"] == 8
    assert not report["measurement"]["target_met"]
    assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None


def test_missing_warmup_or_insufficient_duration_prevents_qualification():
    for changed in ("warmup", "duration", "idle"):
        log = make_log()
        if changed == "warmup":
            log["phases"][1]["batches"] = []
            log["phases"][1]["samples"].pop(1)
        elif changed == "duration": log["plan"]["minimum_duration_s"] = 2.
        else: log["plan"]["idle_duration_s"] = 1.
        report = records.analyze(reseal(log))
        assert report["measurement"]["amortized_domain_energy_j_per_qualified_solve"] is None


@pytest.mark.parametrize("change", ["extra_top", "origin", "boolean_time", "negative_time", "phase_order",
    "overlap", "batch_outside", "batch_index", "sample_reverse", "sample_order", "result_digest", "replicas",
    "problem_digest", "model_digest", "observations_digest", "solver_binding", "iteration_binding",
    "prepared_input", "scope", "invented_accuracy", "error_value", "missing_context_error", "decimal_leading_zero",
    "counter_overflow", "runtime_extra", "bad_gpu_uuid", "zero_target", "infinite_target"])
def test_resealed_structural_or_binding_tampering_is_rejected(change):
    log = make_log()
    phase = log["phases"][3]
    sample = phase["samples"][1]
    if change == "extra_top": log["extra"] = 1
    elif change == "origin": log["origin"] = "measured"
    elif change == "boolean_time": phase["start_ns"] = True
    elif change == "negative_time": phase["start_ns"] = -1
    elif change == "phase_order": log["phases"][0]["name"] = "measurement"
    elif change == "overlap": phase["start_ns"] = 1
    elif change == "batch_outside": phase["batches"][0]["end_ns"] = phase["end_ns"]+1
    elif change == "batch_index": phase["batches"][0]["batch_index"] = 1
    elif change == "sample_reverse": sample["read_end_ns"] = sample["read_start_ns"]-1
    elif change == "sample_order": phase["samples"].reverse()
    elif change == "result_digest": phase["batches"][0]["result"]["expanded_sha256"] = "sha256:"+"0"*64
    elif change == "replicas": phase["batches"][0]["result"]["replicas"] = 5
    elif change in ("problem_digest", "model_digest", "observations_digest"): log["plan"][change] = "sha256:"+"0"*64
    elif change == "solver_binding": log["runtime"]["workload"]["solver_settings"]["alpha"] = .3
    elif change == "iteration_binding": log["runtime"]["workload"]["iterations"] += 1
    elif change == "prepared_input": log["runtime"]["workload"]["prepared_input_sha256"] = "0"*64
    elif change == "scope": log["sensor"]["counter_scope"] = "whole_computer"
    elif change == "invented_accuracy": log["sensor"]["accuracy_j"] = .01
    elif change == "error_value": sample.update(status="error", error_code=3)
    elif change == "missing_context_error": sample["power_mw"] = None
    elif change == "decimal_leading_zero": sample["energy_mj"] = "01"
    elif change == "counter_overflow": sample["energy_mj"] = str(2**64)
    elif change == "runtime_extra": log["runtime"]["workload"]["unsupported"] = 1
    elif change == "bad_gpu_uuid": log["sensor"]["device_uuid"] = "0"
    elif change == "zero_target": log["plan"]["target_kl_nats"] = 0
    elif change == "infinite_target": log["plan"]["target_kl_nats"] = float("inf")
    with pytest.raises(ValueError):
        reseal(log)


def test_offline_analysis_does_not_call_solver_or_hardware(monkeypatch):
    log = make_log()
    monkeypatch.setattr(mathematics, "variational_fit", lambda *a, **k: pytest.fail("must not rerun optimizer"))
    assert records.analyze(log)["measurement"]["target_met"]
