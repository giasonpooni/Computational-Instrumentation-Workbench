"""Energy and GPU experiments T115-T125 of the computational-experimentalist queue.

Scope: deterministic work proxies for geodesic trajectories, a cross-language
(Python/Rust) RK4 comparison, float32/float64 accuracy against operation
counts, CPU emulation of GPU-style reduction orders, a bounded Gaussian
variational free-energy example, typed separation of nats from joules, and
validation, tampering and Session replay of the retained synthetic
energy/accuracy fixtures in ``examples/energy-accuracy``.

Non-claims: this environment has no RAPL counters, no NVIDIA GPU or NVML and
no Julia. No energy, power, temperature, utilization or kernel duration was
measured; every such claim is recorded as a physical-domain finding and stays
``not_established`` unless an operator supplies a capture that passes the
acquisition gate. The lab runner itself never reads an energy counter: RAPL
and NVML captures are operator actions whose retained bytes T115, T116 and
T118 analyze read-only. The fixtures are synthetic, so every joule figure
derived from them is a synthetic value. Emulated reduction orders are not the
orders of any GPU library, and Python/Rust agreement is agreement of two ciw
implementations, not independent verification. Wall-clock and CPU times are
retained only as artifacts; they are not reproducible and never enter a
finding.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from . import energy_gpu_kernels as kernels
from . import energy_gpu_telemetry as telemetry
from . import runner, svg
from .evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, finding, holds, supported_label
from .integrators import integrate_adaptive, integrate_fixed
from .registry import task
from .surfaces import Sphere

MODULE = "src/ciw/lab/energy_gpu.py"
KERNELS = "src/ciw/lab/energy_gpu_kernels.py"
TELEMETRY = "src/ciw/lab/energy_gpu_telemetry.py"
DOC = "docs/lab/ENERGY_GPU.md"
TESTS = "tests/test_lab_energy_gpu.py"
FILES = (MODULE, KERNELS, TELEMETRY, DOC)
UNCERTAINTY_TEST = f"{TESTS}::test_every_numerical_finding_declares_uncertainty_and_tolerance"
FIXTURELESS_TEST = f"{TESTS}::test_fixture_tasks_without_repository_files"

STEPS = kernels.WORKLOAD_STEPS
FIXTURE_NOTE = ("examples/energy-accuracy/{baseline,reset,missing,under-target}.json: synthetic fixtures "
                "(origin synthetic_fixture) constructed on the CPU, not hardware captures")

# Per-finding uncertainty statements (AUTHORING rule 5).
EXACT = {"kind": "exact", "value": 0.0, "basis": "integer counts, refusal codes or exact comparisons"}
SYNTHETIC = {"kind": "reference_error", "value": 0.0,
             "basis": "synthetic fixture with exact decimal counter readings; no physical uncertainty model"}
EXACT_TOL = {"abs": 0.0, "rel": 0.0}
IEEE = {"kind": "roundoff", "value": 0.0, "basis": "deterministic IEEE-754 arithmetic in the declared order; "
                                                   "exact reference by math.fsum"}


def _check(reference, observed, tolerance, comparison="abs_le", kind="analytic"):
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed if observed is not None else "accepted", "passed": observed == expected}


def _state(findings, planned):
    """Downgrade to partial when a computational check failed instead of hiding it."""
    unexpected = [f for f in findings if f["evidence_status"] == "not_established"
                  and f["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and not f.get("expected_not_established")]
    return "partial" if unexpected and planned == "completed" else planned


REQUIRED = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
            "experiment", "numerical_result", "uncertainty", "failure_modes_checked",
            "unresolved_assumptions", "recommended_next_task")


def _fields(**values):
    missing = [name for name in REQUIRED if name not in values]
    if missing:
        raise ValueError(f"Report fields missing: {missing}")
    return values


def _no_measurement(claim, reason, unit=None, domain="physical"):
    """A physical claim this environment cannot support: recorded, never omitted."""
    return finding(claim, domain, None, {"notes": [reason]}, unit=unit)


def _fixtures_blocked(static, physical):
    """Blocked report for a fixture task: the static answers stay, only the run-dependent ones say not run."""
    fields = _fields(**dict(static, numerical_result="not run: examples/energy-accuracy is not reachable",
                            uncertainty="not run: fixtures absent",
                            experiment="Blocked: examples/energy-accuracy is not reachable (no source checkout and no "
                                       "CIW_LAB_REPOSITORY_ROOT). Planned: " + static["experiment"],
                            failure_modes_checked=["repository fixtures reachable through "
                                                   "ciw.lab.runner.repository_path (they are not)"],
                            unresolved_assumptions=list(static["unresolved_assumptions"]) + [
                                "examples/energy-accuracy is outside an installed package; set "
                                "CIW_LAB_REPOSITORY_ROOT to a checkout"]))
    return {"state": "blocked", "fields": fields, "findings": [physical()]}


def _retain_raw(ctx, name, raw: bytes) -> str | None:
    """Retain exact raw bytes as an artifact; return why they could not be retained, or None."""
    if len(raw) > runner.MAX_ARTIFACT_BYTES:
        return f"raw file exceeds the {runner.MAX_ARTIFACT_BYTES}-byte artifact limit and cannot be retained"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "raw file is not UTF-8 text and cannot be retained byte-exactly"
    ctx.artifact_text(name, text)
    if ctx.artifacts[-1]["sha256"] != hashlib.sha256(raw).hexdigest():
        return "retained artifact bytes differ from the raw file"
    return None


# ----------------------------------------------------------------- T115
GROSS_CPU = "Gross CPU package energy (background-inclusive, idle not subtracted) per geodesic trajectory"
IDLE_CPU = "Idle-subtracted CPU package energy per geodesic trajectory (equal-length idle bracket after the workload)"


def _cpu_workload():
    """Fixed-step and adaptive integration of the equatorial sphere geodesics with counted work."""
    surface = Sphere(1.0)
    states = kernels.initial_states()
    counter = {"n": 0}

    def rhs(y):
        counter["n"] += 1
        return surface.geodesic_rhs(y)

    fixed, fixed_counts, wall, cpu = [], [], [], []
    for y0 in states:
        before, wall0, cpu0 = counter["n"], time.perf_counter(), time.process_time()
        _, path = integrate_fixed(rhs, y0, kernels.LENGTH, STEPS, "rk4")
        wall.append(time.perf_counter() - wall0)
        cpu.append(time.process_time() - cpu0)
        fixed.append(path[-1])
        fixed_counts.append(counter["n"] - before)
    adaptive, adaptive_stats = [], []
    for y0 in states:
        _, path, stats = integrate_adaptive(surface.geodesic_rhs, y0, kernels.LENGTH, rtol=1e-9, atol=1e-12)
        adaptive.append(path[-1])
        adaptive_stats.append(stats)
    return {"states": states, "fixed": np.array(fixed), "fixed_counts": fixed_counts,
            "fixed_errors": kernels.endpoint_errors(states, fixed, kernels.LENGTH),
            "adaptive": np.array(adaptive), "adaptive_stats": adaptive_stats,
            "adaptive_errors": kernels.endpoint_errors(states, adaptive, kernels.LENGTH),
            "wall_s": wall, "cpu_s": cpu}


def _rapl_findings(ctx):
    """Physical findings from an operator RAPL capture (CIW_LAB_RAPL_LOG), or not_established without one."""
    path = telemetry.environment_log_path(telemetry.RAPL_ENV)
    if path is None:
        reason = ("no RAPL capture was supplied; the lab runner reads no energy counter (capture with `python -m "
                  f"ciw.lab.energy_gpu_telemetry rapl-capture` and set {telemetry.RAPL_ENV})")
        ctx.artifact_json("rapl-probe.json", {"capture_supplied": False, "note": reason,
                                              "package_domains_present": [d["zone"] for d in telemetry.rapl_domains()]})
        return [_no_measurement(GROSS_CPU, reason, unit="J/trajectory"),
                _no_measurement(IDLE_CPU, reason, unit="J/trajectory")], None
    raw = Path(path).read_bytes()
    reasons = []
    try:
        record = json.loads(raw)
    except ValueError:
        record = None
    reasons += telemetry.rapl_capture_reasons(record, telemetry.rapl_host_identity(), kernels.WORKLOAD)
    if not ctx.available("hardware:rapl"):
        reasons.append("no readable intel-rapl counter answered the probe on this analyzing host")
    retention = _retain_raw(ctx, "rapl-capture.json", raw)
    if retention:
        reasons.append(retention)
    energy = None
    if not reasons:
        try:
            energy = telemetry.rapl_capture_energy(record)
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"capture counters are inconsistent: {exc}")
    ctx.artifact_json("rapl-probe.json", {"capture_supplied": True, "withheld_reasons": reasons, "energy": energy})
    if reasons:
        return [finding(GROSS_CPU, "physical", None, {"notes": reasons}, unit="J/trajectory"),
                finding(IDLE_CPU, "physical", None, {"notes": reasons}, unit="J/trajectory")], energy
    bracket = record["workload_bracket"]
    acquisition = {"device": "RAPL package domains " + ", ".join(record["host"]["zones"]) + " on "
                             + str(record["host"]["cpu_model"]) + " (operator-captured, unauthenticated)",
                   "raw_sha256": hashlib.sha256(raw).hexdigest(),
                   "acquired_at": telemetry._utc(bracket["start_utc_ns"]),
                   "calibration": "not_applied: RAPL model-based counter, package scope, background-inclusive"}
    basis = {"acquisition": acquisition, "notes": [telemetry.OPERATOR_NOTE]}
    uncertainty = {"kind": "reference_error", "value": None,
                   "basis": "RAPL model error, counter unit and background load are not characterized"}
    return [finding(GROSS_CPU, "physical", energy["gross_j_per_trajectory"], basis, unit="J/trajectory",
                    uncertainty=uncertainty, tolerance={"abs": 0.0, "rel": 1.0}),
            finding(IDLE_CPU, "physical", energy["idle_subtracted_j_per_trajectory"], basis, unit="J/trajectory",
                    uncertainty=uncertainty, tolerance={"abs": 0.0, "rel": 1.0})], energy


@task("T115", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_cpu_energy_task_counts_work_and_leaves_energy_unestablished",
    f"{TESTS}::test_rapl_helpers_read_counters_and_one_wrap",
    f"{TESTS}::test_rapl_capture_is_analyzed_read_only_and_gated",
    f"{TESTS}::test_lab_run_never_reads_energy_counters", UNCERTAINTY_TEST))
def cpu_energy_per_trajectory(ctx):
    work = ctx.memo("energy-gpu-cpu-workload", _cpu_workload)
    trajectories = len(work["states"])
    counted = sum(work["fixed_counts"])
    evaluations = [s["function_evaluations"] for s in work["adaptive_stats"]]
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "seed": None}
    findings = [
        finding(f"Fixed-step RK4 spends exactly 4N = {4 * STEPS} right-hand-side evaluations per trajectory "
                f"(N = {STEPS})", "numerical", list(work["fixed_counts"]),
                {"generator": dict(generator, steps=STEPS),
                 "checks": [_check("counted evaluations minus 4 N T", counted - 4 * STEPS * trajectories, 0.0,
                                   kind="exact_arithmetic")]},
                unit="evaluations/trajectory", uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("The fixed-step trajectories are accepted results: embedded endpoint error against the exact "
                "great circle is below 1e-8", "numerical", float(np.max(work["fixed_errors"])),
                {"generator": dict(generator, steps=STEPS),
                 "checks": [_check("great circle cos(L) X0 + sin(L) T0", float(np.max(work["fixed_errors"])), 1e-8)]},
                unit="normalized length",
                uncertainty={"kind": "reference_error", "value": 1e-15,
                             "basis": "closed-form great-circle endpoint evaluated in binary64 (a few ulp)"},
                tolerance={"abs": 1e-12, "rel": 1e-3}),
        finding("Adaptive Dormand-Prince 5(4) (rtol 1e-9) reaches every great-circle endpoint within 1e-7 with "
                "these right-hand-side evaluation counts per trajectory",
                "numerical", evaluations,
                {"generator": dict(generator, rtol=1e-9, atol=1e-12),
                 "checks": [_check("great circle endpoint", float(np.max(work["adaptive_errors"])), 1e-7)]},
                unit="evaluations/trajectory",
                uncertainty={"kind": "exact", "value": 0.0, "basis": "integer counts; step acceptance near the "
                                                                     "tolerance may shift a count by a few stages "
                                                                     "across platforms"},
                tolerance={"abs": 14.0, "rel": 0.05}),
    ]
    physical, energy = _rapl_findings(ctx)
    findings += physical
    measured = all(f["evidence_status"] == "hardware_measured" for f in physical)
    ctx.artifact_json("work-proxies.json", {
        "steps": STEPS, "trajectories": trajectories, "fixed_step_evaluations": work["fixed_counts"],
        "fixed_step_endpoint_errors": work["fixed_errors"].tolist(), "adaptive_stats": work["adaptive_stats"],
        "adaptive_endpoint_errors": work["adaptive_errors"].tolist()})
    ctx.artifact_json("timing.json", {
        "note": "Elapsed and process CPU time of this run only; not reproducible and not an energy measurement.",
        "clock": "time.perf_counter / time.process_time", "wall_s_per_trajectory": work["wall_s"],
        "cpu_s_per_trajectory": work["cpu_s"]})
    energy_text = ("gross and idle-subtracted package energy from the supplied capture, see findings" if measured
                   else "not measured (no accepted RAPL capture; the runner reads no counters)")
    fields = _fields(
        hypothesis="The deterministic work of a fixed-step RK4 geodesic trajectory is exactly 4N right-hand-side "
                   "evaluations; its CPU energy can be read only from package counters bracketing a batch, which "
                   "an operator captures outside the lab runner.",
        mathematical_model="E_gross = sum over package domains of (E(after) - E(before)) / (repeats * trajectories), "
                           "one wrap allowed; E_idle-subtracted = (E_gross_total - E_idle * t_work / t_idle) / "
                           "(repeats * trajectories); work proxy W = 4N evaluations (RK4) or the Dormand-Prince count.",
        input_data=[f"{trajectories} unit-speed geodesics on the unit sphere from the equator, headings "
                    f"{list(kernels.HEADINGS)} rad, length {kernels.LENGTH}",
                    f"optional operator capture named by {telemetry.RAPL_ENV}"],
        observation_model="Evaluations counted by wrapping Surface.geodesic_rhs; a supplied capture is read-only "
                          "input whose raw bytes are retained; wall/CPU time retained only as an artifact.",
        expected_invariant="Counted RK4 evaluations equal 4 N per trajectory; endpoints match the great circle; a "
                           "capture must name this workload and this host's CPU model and RAPL zones.",
        experiment=f"Integrate each geodesic with RK4 (N = {STEPS}) and adaptive DP5(4); count evaluations; when "
                   f"{telemetry.RAPL_ENV} names a capture from `python -m ciw.lab.energy_gpu_telemetry rapl-capture` "
                   "(repeated fixed-step batches, three by default, then an idle interval of equal length), gate and "
                   "analyze it.",
        numerical_result=f"{4 * STEPS} evaluations per RK4 trajectory; max endpoint error "
                         f"{float(np.max(work['fixed_errors'])):.3g}; adaptive evaluations {evaluations}; energy: "
                         + energy_text,
        uncertainty="Work counts are exact; a RAPL energy would carry model error, background load and a "
                    "model-dependent counter unit (commonly 2^-16 J), none characterized here.",
        failure_modes_checked=["evaluation count differs from 4N", "endpoint error exceeds 1e-8",
                               "capture absent, malformed, for another workload or another host (regression tests)",
                               "RAPL counter wrap (one wrap allowed; regression test)"],
        unresolved_assumptions=["Evaluation counts are a work proxy, not an energy measurement",
                                "Package energy is not attributed to this process; idle subtraction assumes the idle "
                                "interval after the workload represents the background during it",
                                "A capture is an operator record: identity binding is checked, authenticity is not"],
        recommended_next_task="Capture T115 on a Linux host with readable intel-rapl counters, then T120 to relate "
                              "precision to measured energy")
    return {"state": "completed" if measured else "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T116 / T118 (GPU host only)
RUN = "runs/rtx2080-<date>"
RECORD = (f"ciw energy record --problem examples/energy-accuracy/problem.json --output-dir {RUN}/capture "
          "--duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0")
SMI = ("TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,"
       f"power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f {RUN}/smi.csv")
NO_GPU = "no NVIDIA GPU or NVML in this environment"

GPU_ENERGY = "GPU-domain gross energy per measured batch"
NVML_ACCURACY = "The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution"
NVML_ACCURACY_NOTE = ("NVML declares no accuracy or resolution for this counter and no external power meter was "
                      "compared; the counter is used uncalibrated")
POWER = "RTX 2080 power draw during the measurement phase (NVML)"
TEMPERATURE = "RTX 2080 temperature during the measurement phase (NVML)"
CLOCK = "RTX 2080 graphics clock during the measurement phase (NVML)"
DURATION = "Host-bracketed batch solve duration (launch, sync and copy included)"
UTILIZATION = "RTX 2080 GPU utilization during the measurement phase (nvidia-smi rows inside the measurement window)"
POWER_STEADY = "RTX 2080 power draw is steady over the measurement phase (coefficient of variation <= 0.10)"
TEMPERATURE_STEADY = "RTX 2080 temperature drifts by at most 5 C over the measurement phase"
KERNEL_ONLY = "RTX 2080 kernel-only duration of the Gaussian VI kernel"
KERNEL_NOTE = ("log.json brackets launch, synchronization and copy; kernel spans need an Nsight Systems report, "
               "which this section does not ingest")
POWER_CV_LIMIT, TEMPERATURE_DRIFT_LIMIT = 0.10, 5.0
T118_UNITS = {POWER: "W", TEMPERATURE: "C", CLOCK: "MHz", DURATION: "ms", UTILIZATION: "%", POWER_STEADY: "ratio",
              TEMPERATURE_STEADY: "C", KERNEL_ONLY: "ms"}


def _gpu_plan_findings(task_id):
    """The physical claims a GPU task exists to test, recorded as not_established where no GPU answers."""
    if task_id == "T116":
        return [_no_measurement(GPU_ENERGY, NO_GPU, unit="J/batch"),
                _no_measurement(NVML_ACCURACY, NVML_ACCURACY_NOTE, domain="sensor_performance")]
    return [_no_measurement(claim, KERNEL_NOTE if claim == KERNEL_ONLY else NO_GPU, unit=unit)
            for claim, unit in T118_UNITS.items()]


PLAN_T116 = {
    "hypothesis": "Gross GPU-device energy per batch of the fixed binary64 Gaussian VI workload is measurable from "
                  "NVML total-energy counter differences bracketing each batch.",
    "mathematical_model": "E_batch = (E(end of measurement) - E(start)) / n_batches, gross, background-inclusive, "
                          "no idle subtraction; per-batch increments from the counter read after each batch.",
    "input_data": ["examples/energy-accuracy/problem.json", "log.json written by `ciw energy record` on the GPU host"],
    "observation_model": "nvmlDeviceGetTotalEnergyConsumption (mJ, whole device) read between host call brackets; "
                         "batch outputs retained bitwise and scored by KL against the exact posterior.",
    "expected_invariant": "Counter monotone across phases; phases non-overlapping; sensor UUID equals workload UUID; "
                          "every batch meets the declared KL target.",
    "experiment": f"On the RTX 2080 host: (0) `mkdir -p {RUN}`; (1) `ciw energy probe --gpu-index 0` must return a "
                  f"reading with status ok; (2) `{RECORD}`; (3) `ciw energy replay {RUN}/capture/log.json`; (4) "
                  f"`CIW_LAB_ENERGY_LOG={RUN}/capture/log.json ciw lab run T116 T118 --output-dir <dir>`.",
    "numerical_result": "none: " + NO_GPU,
    "uncertainty": "NVML counter resolution, update interval and accuracy are undeclared; energy is device-wide and "
                   "includes background work; host brackets add call overhead.",
    "failure_modes_checked": ["hardware:nvidia-gpu probe (nvidia-smi -L): no device here",
                              "acquisition gate exercised by regression tests on synthetic logs: synthetic origin, "
                              "absent or differing host NVML identity, raw bytes retained",
                              "planned on a captured log (energy_records.analyze): counter reset or wrap, missing "
                              "endpoint brackets, sensor/workload UUID mismatch, KL target, minimum duration"],
    "unresolved_assumptions": ["Blocked here: no NVIDIA GPU or NVML",
                               "NVML documents the total-energy counter for Volta-or-newer fully supported devices; "
                               "GeForce support is not documented, so `ciw energy probe` must confirm it on this "
                               "RTX 2080",
                               "An operator log is an unauthenticated record; the gate binds it to this host's NVML "
                               "identity but cannot prove the capture genuine"],
    "recommended_next_task": "T118: record utilization, temperature and power in the same session",
    "findings": _gpu_plan_findings("T116"),
}
PLAN_T118 = {
    "hypothesis": "During the T116 workload the RTX 2080 power draw is steady (coefficient of variation <= 0.10 over "
                  "the measurement phase) and its temperature drifts by at most 5 C; kernel-only time is out of "
                  "scope until an Nsight Systems report is ingested.",
    "mathematical_model": "Measurement-phase min/mean/max of NVML power, temperature and graphics clock; power CV = "
                          "population standard deviation / mean; temperature drift = max - min; utilization from "
                          "nvidia-smi rows whose UTC timestamps fall inside the measurement window.",
    "input_data": ["log.json from `ciw energy record`", "smi.csv from the nvidia-smi sidecar (run with TZ=UTC)",
                   "nsys report from a separate profiling pass (retained by the operator, not ingested)"],
    "observation_model": "NVML context readings sampled after each energy read (not simultaneous with it); "
                         "nvidia-smi sampling every 100 ms in local wall time, converted with the declared UTC "
                         "offset; CUDA kernel spans from nsys.",
    "expected_invariant": "Device name contains RTX 2080; the same UUID appears in log.json, every in-window smi.csv "
                          "row and this host's NVML identity; power and temperature readings are present for every "
                          "sample.",
    "experiment": f"On the RTX 2080 host: (0) `mkdir -p {RUN}`; (1) start `{SMI}` in the background; (2) `{RECORD}`; "
                  f"(3) stop the sidecar; (4) in a separate profiling pass `nsys profile --trace=cuda -o {RUN}/nsys "
                  f"python -m ciw energy record --problem examples/energy-accuracy/problem.json --output-dir "
                  f"{RUN}/capture-nsys --duration 10 --gpu-index 0` and `nsys stats --report cuda_gpu_kern_sum "
                  f"{RUN}/nsys.nsys-rep`; (5) `CIW_LAB_ENERGY_LOG={RUN}/capture/log.json "
                  f"CIW_LAB_NVIDIA_SMI_CSV={RUN}/smi.csv {telemetry.SMI_OFFSET_ENV}=+00:00 ciw lab run T118 "
                  "--output-dir <dir>`.",
    "numerical_result": "none: " + NO_GPU,
    "uncertainty": "NVML power is a vendor estimate with undeclared averaging; 100 ms sidecar sampling aliases short "
                   "batches; profiling perturbs timing and energy, so it is a separate pass.",
    "failure_modes_checked": ["hardware:nvidia-gpu probe (nvidia-smi -L): no device here",
                              "acquisition gate exercised by regression tests on synthetic logs: device is not an "
                              "RTX 2080, synthetic origin, sidecar rows outside the window or without a declared "
                              "UTC offset",
                              "planned on a captured log: context readings unavailable (retained error codes), "
                              "sidecar missing or naming another UUID"],
    "unresolved_assumptions": ["Blocked here: no NVIDIA GPU", "Batch windows in log.json include launch, "
                               "synchronization and copy; they are not kernel durations",
                               "The steady-state limits (CV 0.10, 5 C) are declared protocol criteria, not derived"],
    "recommended_next_task": "T147: compare CPU and GPU outputs of the same workload on the RTX 2080 host",
    "findings": _gpu_plan_findings("T118"),
}


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return {"min": min(values), "mean": math.fsum(values) / len(values), "max": max(values), "count": len(values)}


def _physical(claim, value, basis, reasons, unit, checks=()):
    """A physical finding: the acquisition basis only when nothing withholds it and a value exists."""
    if value is None or reasons:
        return finding(claim, "physical", value, {"notes": list(reasons) or ["no reading available"]}, unit=unit,
                       tolerance={"abs": 0.0, "rel": 1.0})
    return finding(claim, "physical", value, dict(basis, checks=list(checks)) if checks else basis, unit=unit,
                   uncertainty={"kind": "reference_error", "value": None,
                                "basis": "vendor sensor; accuracy, resolution and averaging undeclared"},
                   tolerance={"abs": 0.0, "rel": 1.0})


def _t118_findings(ctx, log, basis, reasons, smi_raw, offset_ns):
    samples = log["phases"][3]["samples"]
    watts = [s["power_mw"] / 1000 if s["power_mw"] is not None else None for s in samples]
    temperatures = [s["temperature_c"] for s in samples]
    power, temperature = _stats(watts), _stats(temperatures)
    clock = _stats(s["graphics_clock_mhz"] for s in samples)
    durations = _stats((b["end_ns"] - b["start_ns"]) / 1e6 for b in log["phases"][3]["batches"])
    present = [w for w in watts if w is not None]
    cv = float(np.std(present) / np.mean(present)) if len(present) >= 2 and np.mean(present) > 0 else None
    known = [t for t in temperatures if t is not None]
    drift = float(max(known) - min(known)) if len(known) >= 2 else None
    findings = [_physical(POWER, power, basis, reasons, "W"), _physical(TEMPERATURE, temperature, basis, reasons, "C"),
                _physical(CLOCK, clock, basis, reasons, "MHz"), _physical(DURATION, durations, basis, reasons, "ms")]
    smi_reasons, utilization, smi_basis = [], None, None
    if smi_raw is None:
        smi_reasons.append("no nvidia-smi sidecar was supplied")
    else:
        retention = _retain_raw(ctx, "nvidia-smi.csv", smi_raw)
        if retention:
            smi_reasons.append(retention)
        else:
            rows, window_reasons = telemetry.smi_window(smi_raw.decode("utf-8"), int(samples[0]["utc_start_ns"]),
                                                        int(samples[-1]["utc_end_ns"]), offset_ns)
            smi_reasons += window_reasons
            if rows and {row.get("uuid", "").lower() for row in rows} != {log["sensor"]["device_uuid"].lower()}:
                smi_reasons.append("sidecar rows in the window do not all name the log's device UUID")
            key = next((k for k in (rows[0] if rows else {}) if k.startswith("utilization.gpu")), None)
            utilization = _stats(telemetry.number(row[key]) for row in rows) if key else None
            if "acquisition" in basis:
                smi_basis = {"acquisition": dict(basis["acquisition"], raw_sha256=hashlib.sha256(smi_raw).hexdigest(),
                                                 device=basis["acquisition"]["device"] + " via nvidia-smi sidecar"),
                             "notes": [telemetry.OPERATOR_NOTE]}
    findings.append(_physical(UTILIZATION, utilization, smi_basis or {}, reasons + smi_reasons, "%"))
    findings.append(_physical(POWER_STEADY, cv, basis, reasons, "ratio",
                              [_check("declared steady-state limit on the power coefficient of variation", cv or 0.0,
                                      POWER_CV_LIMIT, "le", kind="analytic")]))
    findings.append(_physical(TEMPERATURE_STEADY, drift, basis, reasons, "C",
                              [_check("declared steady-state limit on the temperature range", drift or 0.0,
                                      TEMPERATURE_DRIFT_LIMIT, "le", kind="analytic")]))
    findings.append(_no_measurement(KERNEL_ONLY, KERNEL_NOTE, unit="ms"))
    return findings, smi_reasons


def operator_log_outcome(ctx, task_id, raw, log, analysis, host_identity, smi_raw=None, smi_offset_ns=None):
    """Findings from an operator-captured NVML log; physical labels only through the acquisition gate."""
    from ciw import energy_records
    from ciw.telemetry import canonical
    recomputed = canonical(energy_records.analyze(json.loads(raw))) == canonical(analysis)
    measurement = analysis["measurement"]
    pipeline = finding("The operator log validates and its analysis recomputes identically from the retained bytes",
                       "computational_pipeline", recomputed,
                       {"inputs": [hashlib.sha256(raw).hexdigest()],
                        "checks": [_check("second energy_records.analyze of the same bytes", 0.0 if recomputed else 1.0,
                                          0.0, kind="exact_arithmetic")]}, uncertainty=EXACT, tolerance=EXACT_TOL)
    findings = [pipeline]
    basis, reasons = telemetry.physical_basis(raw, log, analysis, host_identity,
                                              required_name="RTX 2080" if task_id == "T118" else None)
    retention = _retain_raw(ctx, "operator-log.json", raw)
    if retention:
        reasons = reasons + [retention]
    smi_reasons = []
    if task_id == "T116":
        batches = measurement["batch_count"]
        value = (measurement["gross_energy_j"] / batches) if measurement["gross_energy_j"] is not None and batches else None
        findings.append(_physical(GPU_ENERGY, value, basis, reasons + ([] if value is not None else
                                                                        ["gross energy withheld by the analysis"]),
                                  "J/batch"))
        findings.append(_no_measurement(NVML_ACCURACY, NVML_ACCURACY_NOTE, domain="sensor_performance"))
        if measurement["max_kl_nats"] is not None:
            findings.append(finding("Every measured batch output meets the declared KL target", "numerical",
                                    measurement["max_kl_nats"],
                                    {"inputs": [log["log_digest"]],
                                     "checks": [_check("declared target_kl_nats", measurement["max_kl_nats"],
                                                       measurement["target_kl_nats"], comparison="le")]},
                                    unit="nat", uncertainty={"kind": "roundoff", "value": 1e-15,
                                                             "basis": "binary64 Gaussian KL of retained outputs"},
                                    tolerance={"abs": 1e-12, "rel": 1e-6}))
    else:
        t118, smi_reasons = _t118_findings(ctx, log, basis, reasons, smi_raw, smi_offset_ns)
        findings += t118
    ctx.artifact_json("operator-log-analysis.json", {"analysis": analysis, "withheld_reasons": reasons,
                                                     "sidecar_withheld_reasons": smi_reasons,
                                                     "host_sensor_identity": host_identity})
    plan = PLAN_T116 if task_id == "T116" else PLAN_T118
    # Claims no capture can support here (calibration, kernel-only time) do not decide the state.
    structural = {NVML_ACCURACY, KERNEL_ONLY}
    measured = all(f["evidence_status"] == "hardware_measured" for f in findings
                   if f["domain"] in PHYSICAL_DOMAINS and f["claim"] not in structural)
    fields = _fields(**{k: v for k, v in plan.items() if k != "findings"})
    fields["experiment"] = "Analyzed the operator-captured log named by CIW_LAB_ENERGY_LOG. Protocol: " + plan["experiment"]
    batches = measurement["batch_count"]
    fields["numerical_result"] = (f"gross measurement energy {measurement['gross_energy_j']} J over {batches} measured "
                                  f"batch{'es' if batches != 1 else ''}; physical findings withheld: {reasons or 'none'}")
    fields["failure_modes_checked"] = [
        "energy_records.validate_log structure and digest", "analysis recomputed from the retained bytes",
        "analysis eligibility: counter reset or wrap, endpoint brackets, sensor/workload UUID, KL target, durations",
        "acquisition gate: declared origin and host NVML identity (UUID, name, driver, NVML version, library digest)",
        "raw log bytes retained byte-exactly"]
    if task_id == "T118":
        fields["failure_modes_checked"] += ["device name contains RTX 2080", "sidecar rows restricted to the "
                                            "measurement window with a declared UTC offset",
                                            "in-window sidecar rows name the same UUID"]
    fields["unresolved_assumptions"] = ["A sealed log proves internal integrity, not that the capture was genuine",
                                        *[item for item in plan["unresolved_assumptions"]
                                          if not item.startswith("Blocked here")], *reasons, *smi_reasons]
    state = "completed" if measured and task_id == "T116" else "partial"
    return {"state": state, "fields": fields, "findings": findings}


def _operator_log_task(ctx, task_id):
    plan = PLAN_T116 if task_id == "T116" else PLAN_T118
    fields = _fields(**{k: v for k, v in plan.items() if k != "findings"})
    path = telemetry.environment_log_path()
    reason = None
    if path is None:
        reason = (f"a GPU is present but the lab runner never acquires hardware data; set {telemetry.LOG_ENV} to a "
                  "log.json captured by `ciw energy record`")
    else:
        try:
            raw, log, analysis = telemetry.read_operator_log(path)
        except (OSError, ValueError) as exc:
            reason = f"the operator log named by {telemetry.LOG_ENV} is unreadable or invalid: {exc}"
    if reason:
        fields["experiment"] = f"Blocked: {reason}. Planned: " + plan["experiment"]
        fields["unresolved_assumptions"] = [reason, *plan["unresolved_assumptions"]]
        return {"state": "blocked", "fields": fields, "findings": _gpu_plan_findings(task_id)}
    smi_path = telemetry.environment_log_path(telemetry.SMI_ENV)
    smi_raw = Path(smi_path).read_bytes() if smi_path else None
    offset = telemetry.utc_offset_ns(telemetry.environment_log_path(telemetry.SMI_OFFSET_ENV))
    host = telemetry.host_sensor_identity(log["sensor"]["device_uuid"])
    return operator_log_outcome(ctx, task_id, raw, log, analysis, host, smi_raw, offset)


@task("T116", changed_files=FILES, requires=("hardware:nvidia-gpu",), plan=PLAN_T116, regression_tests=(
    f"{TESTS}::test_gpu_tasks_are_blocked_with_the_recording_protocol",
    f"{TESTS}::test_operator_log_gate_withholds_physical_labels_from_synthetic_logs",
    f"{TESTS}::test_operator_log_gate_trust_boundary_is_the_host_identity", UNCERTAINTY_TEST))
def gpu_energy_per_batch(ctx):
    return _operator_log_task(ctx, "T116")


@task("T118", changed_files=FILES, requires=("hardware:nvidia-gpu",), plan=PLAN_T118, regression_tests=(
    f"{TESTS}::test_gpu_tasks_are_blocked_with_the_recording_protocol",
    f"{TESTS}::test_operator_log_gate_withholds_physical_labels_from_synthetic_logs",
    f"{TESTS}::test_t118_sidecar_rows_are_restricted_to_the_measurement_window", UNCERTAINTY_TEST))
def rtx2080_telemetry(ctx):
    return _operator_log_task(ctx, "T118")


# ----------------------------------------------------------------- T117
RUST_REFUSALS = {"malformed": ([[1.0, 0.0, 1.0]], "Rust kernel refused its input: states must be a list of 4-vectors"),
                 # theta = 0 is the chart pole: cot(theta) is infinite and the state becomes NaN.
                 "nonfinite": ([[0.0, 0.0, 1.0, 0.0]], "Rust kernel refused its input: nonfinite state")}


def _rust_run(states):
    with tempfile.TemporaryDirectory() as scratch:
        try:
            wall0 = time.perf_counter()
            identity = kernels.build_rust_kernel(scratch)
            build_s = time.perf_counter() - wall0
            executable = identity.pop("executable")
            wall0 = time.perf_counter()
            result = kernels.run_rust_kernel(executable, states, kernels.LENGTH, STEPS)
            run_s = time.perf_counter() - wall0
        except kernels.NativeKernelUnavailable as exc:
            return {"unavailable": str(exc)}
        refusals = {}
        for name, (bad, _) in RUST_REFUSALS.items():
            try:
                kernels.run_rust_kernel(executable, bad, kernels.LENGTH, 4)
                refusals[name] = None
            except kernels.NativeKernelUnavailable as exc:
                refusals[name] = str(exc)
    return {"identity": identity, "states": np.array(result["states"]), "evaluations": result["evaluations"],
            "refusals": refusals, "build_wall_s": build_s, "run_wall_s_including_process_start": run_s}


def _python_runs(states):
    surface = Sphere(1.0)
    out = {}
    for name, rhs in (("python-closed-form", kernels.sphere_rhs), ("python-generic-christoffel", surface.geodesic_rhs)):
        wall0 = time.perf_counter()
        out[name] = np.array([integrate_fixed(rhs, y0, kernels.LENGTH, STEPS, "rk4")[1][-1] for y0 in states])
        out[name + "-wall_s"] = time.perf_counter() - wall0
    return out


def independence_refusal() -> str | None:
    """Try to declare the Rust kernel an independent checker of the Python kernel."""
    check = _check("Python RK4 endpoint", 0.0, 1e-12, kind="high_precision")
    check.update(producer={"implementation": "ciw.lab.integrators", "revision": "working-tree"},
                 checker={"implementation": "ciw.lab.energy_gpu_kernels.RUST_SOURCE", "revision": kernels.rust_source_sha256()})
    try:
        supported_label({"independent_check": check}, "numerical")
    except EvidenceRefusal as exc:
        return str(exc)
    return None


AGREE = "Rust and Python closed-form RK4 kernels agree on every endpoint to 1e-12"
RUST_ERROR = "Rust kernel endpoint error against the exact great circle is below 1e-8"
RUST_REFUSES = "The compiled Rust kernel refuses malformed and nonfinite inputs"
ANGLE_UNIT = "radians or radians per unit length"


@task("T117", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_rust_kernel_matches_python_kernel",
    f"{TESTS}::test_cross_language_agreement_is_not_independent", UNCERTAINTY_TEST))
def compare_implementations(ctx):
    states = kernels.initial_states()
    python = ctx.memo("energy-gpu-python-kernels", lambda: _python_runs(states))
    rust = ctx.memo("energy-gpu-rust-kernel", lambda: _rust_run(states))
    closed, generic = python["python-closed-form"], python["python-generic-christoffel"]
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "steps": STEPS, "seed": None}
    generic_gap = float(np.max(np.abs(generic - closed)))
    roundoff = {"kind": "roundoff", "value": 1e-15, "basis": "libm sin/cos may differ by an ulp across platforms; "
                                                              "differences propagate through 256 steps"}
    findings = []
    if "unavailable" in rust:
        note = {"notes": [rust["unavailable"]]}
        findings += [finding(AGREE, "numerical", None, note, unit=ANGLE_UNIT, expected_not_established=True),
                     finding(RUST_ERROR, "numerical", None, note, unit="normalized length",
                             expected_not_established=True),
                     finding(RUST_REFUSES, "computational_pipeline", None, note, expected_not_established=True)]
        rust_row = {"available": False, "reason": rust["unavailable"]}
    else:
        gap = float(np.max(np.abs(rust["states"] - closed)))
        rust_errors = kernels.endpoint_errors(states, rust["states"], kernels.LENGTH)
        findings.append(finding(
            AGREE, "numerical", gap,
            {"generator": generator,
             "checks": [_check("Python ciw.lab.integrators.step_rk4 with the same closed-form right-hand side and "
                               "operation order", gap, 1e-12, kind="cross_implementation"),
                        _check("right-hand-side calls counted inside the Rust rhs() minus 4 N T",
                               rust["evaluations"] - 4 * STEPS * len(states), 0.0, kind="exact_arithmetic")]},
            unit=ANGLE_UNIT, uncertainty=roundoff, tolerance={"abs": 1e-12, "rel": 0.0}))
        findings.append(finding(
            RUST_ERROR, "numerical", float(np.max(rust_errors)),
            {"generator": generator, "checks": [_check("great circle cos(L) X0 + sin(L) T0", float(np.max(rust_errors)), 1e-8)]},
            unit="normalized length",
            uncertainty={"kind": "reference_error", "value": 1e-15, "basis": "closed-form endpoint in binary64"},
            tolerance={"abs": 1e-12, "rel": 1e-3}))
        findings.append(finding(
            RUST_REFUSES, "computational_pipeline", rust["refusals"],
            {"checks": [_refusal(f"Rust kernel on the {name} input {bad}", expected, rust["refusals"][name])
                        for name, (bad, expected) in RUST_REFUSALS.items()]}, uncertainty=EXACT, tolerance=EXACT_TOL))
        rust_row = {"available": True, "identity": rust["identity"], "evaluations": rust["evaluations"],
                    "endpoints": rust["states"].tolist(), "max_abs_difference_to_python": gap,
                    "max_ulp_distance_to_python": kernels.ulp_distance(rust["states"], closed),
                    "bitwise_identical_to_python": bool(np.array_equal(rust["states"], closed)),
                    "endpoint_errors": rust_errors.tolist(), "refusals": rust["refusals"]}
    findings.append(finding(
        "Generic Christoffel-symbol RK4 (ciw.lab.surfaces) and the closed-form kernel agree to 1e-12", "numerical",
        generic_gap, {"generator": generator, "checks": [_check("closed-form sphere Christoffel symbols",
                                                                generic_gap, 1e-12, kind="cross_implementation")]},
        unit=ANGLE_UNIT, uncertainty=roundoff, tolerance={"abs": 1e-12, "rel": 0.0}))
    refusal = independence_refusal()
    expected = "independent_check producer and checker share an implementation origin"
    findings.append(finding(
        "Declaring the Rust kernel an independent check of the Python kernel is refused (both are ciw code)",
        "provenance", {"producer_origin": "ciw", "checker_origin": "ciw"},
        {"checks": [_refusal("ciw.lab.evidence.supported_label with producer ciw.lab.integrators and checker "
                             "ciw.lab.energy_gpu_kernels.RUST_SOURCE (the origin is read from the declared "
                             "implementation name, so the declaration must name the kernel's true origin)",
                             expected, refusal)]}, uncertainty=EXACT, tolerance=EXACT_TOL))
    julia = ctx.available("tool:julia")
    findings.append(finding("A Julia implementation agrees with the Python kernel", "numerical", None,
                            {"notes": ["julia is not on PATH" if not julia else
                                       "no Julia kernel is implemented in this section"]},
                            expected_not_established=True))
    findings.append(finding("A GPU implementation agrees with the Python kernel", "numerical", None,
                            {"notes": ["no NVIDIA GPU or CUDA toolchain in this environment"]},
                            expected_not_established=True))
    findings.append(_no_measurement("The Rust kernel uses less energy per trajectory than the Python kernel on real "
                                    "hardware", "no energy counter was read; elapsed time is not energy"))
    ctx.artifact_json("implementations.json", {
        "initial_states": states.tolist(), "steps": STEPS, "length": kernels.LENGTH,
        "python_closed_form_endpoints": closed.tolist(), "python_generic_endpoints": generic.tolist(),
        "rust": rust_row, "julia": {"available": julia, "implemented": False},
        "gpu": {"available": ctx.available("hardware:nvidia-gpu"), "implemented": False}})
    ctx.artifact_text("sphere_rk4.rs", kernels.RUST_SOURCE)
    ctx.artifact_json("timing.json", {
        "note": "Elapsed times of this run only; the Rust figure includes process start and JSON exchange.",
        "python_closed_form_wall_s": python["python-closed-form-wall_s"],
        "python_generic_wall_s": python["python-generic-christoffel-wall_s"],
        "rust_build_wall_s": rust.get("build_wall_s"), "rust_run_wall_s": rust.get("run_wall_s_including_process_start")})
    rust_text = (f"Rust-Python max difference {rust_row['max_abs_difference_to_python']:.3g} "
                 f"(bitwise identical: {rust_row['bitwise_identical_to_python']}); Rust refusals "
                 f"{sorted(rust_row['refusals'])}" if rust_row["available"] else "Rust skipped: " + rust_row["reason"])
    identity = dict(runner.builtin_identity(FILES), rust=(
        {"rustc": rust_row["identity"]["rustc"], "flags": rust_row["identity"]["flags"],
         "source_sha256": rust_row["identity"]["source_sha256"], "binary_sha256": rust_row["identity"]["binary_sha256"]}
        if rust_row["available"] else {"available": False, "reason": rust_row["reason"]}))
    fields = _fields(
        hypothesis="Implementations of the same RK4 geodesic kernel in Python and Rust agree to rounding error, and "
                   "all agree with the exact great circle to the RK4 truncation error.",
        mathematical_model="Unit sphere, theta'' = sin cos phi'^2, phi'' = -2 cot(theta) theta' phi'; RK4 with h = L/N "
                           "and the operation order of ciw.lab.integrators.step_rk4.",
        input_data=[f"{len(states)} equatorial geodesics, headings {list(kernels.HEADINGS)} rad, L = {kernels.LENGTH}, "
                    f"N = {STEPS}"],
        observation_model="Endpoints exchanged as JSON with shortest round-trip float text; the Rust source is "
                          "embedded in energy_gpu_kernels and compiled with rustc -O (one codegen unit, scratch path "
                          "remapped) into a temporary directory; the Rust rhs() counts its own calls.",
        expected_invariant="Python closed-form and Rust endpoints agree to 1e-12 (bitwise where libm agrees); "
                           "generic Christoffel path agrees to 1e-12; every endpoint within 1e-8 of the great circle.",
        experiment="Integrate the same initial states with the generic Python path, the closed-form Python path and "
                   "the compiled Rust kernel; compare endpoints, counted evaluations and exact endpoints; send the "
                   "Rust kernel a malformed and a nonfinite input; probe Julia and GPU availability.",
        numerical_result=f"{rust_text}; generic-closed-form max difference {generic_gap:.3g}.",
        uncertainty="Agreement is limited by libm sin/cos differences across platforms; both kernels are ciw code, "
                    "so shared modelling errors would not be detected.",
        failure_modes_checked=["rustc absent or failing (reported, not hidden)",
                               "Rust refuses malformed and nonfinite inputs (run in the task when rustc works)",
                               "evaluation count mismatch (counted inside the Rust rhs)",
                               "independence declaration between ciw implementations"],
        unresolved_assumptions=["Julia and GPU implementations were not run (toolchain/hardware unavailable)",
                                "Cross-platform bitwise identity is not claimed"],
        recommended_next_task="T147: compare CPU and GPU outputs of this kernel on the RTX 2080 host; T146 for "
                              "one canonical float serialization across languages",
        provider_runtime_identity=identity)
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T119
def _textbook_posterior(problem):
    """Exact Gaussian posterior in information form, written here independently of free_energy_math."""
    prior = np.asarray(problem["prior_covariance"], dtype=float)
    h = np.asarray(problem["observation_matrix"], dtype=float)
    noise = np.asarray(problem["noise_covariance"], dtype=float)
    precision = np.linalg.inv(prior) + h.T @ np.linalg.solve(noise, h)
    covariance = np.linalg.inv(precision)
    mean = covariance @ (np.linalg.solve(prior, np.asarray(problem["prior_mean"], dtype=float))
                         + h.T @ np.linalg.solve(noise, np.asarray(problem["observations"], dtype=float)))
    return mean, covariance


def _textbook_kl(mean_q, cov_q, mean_p, cov_p):
    """KL(q || p) = (tr(P^-1 Q) + d^T P^-1 d - k + ln det P - ln det Q) / 2 for Gaussians."""
    delta = mean_p - mean_q
    solve = np.linalg.solve(cov_p, cov_q)
    return 0.5 * (float(np.trace(solve)) + float(delta @ np.linalg.solve(cov_p, delta)) - len(mean_q)
                  + float(np.linalg.slogdet(cov_p)[1] - np.linalg.slogdet(cov_q)[1]))


def _accepted_energy(log, analysis):
    """Recompute the measurement-phase counter delta and accepted solves from raw readings and raw outputs."""
    phase = log["phases"][3]
    readings = [int(s["energy_mj"]) for s in phase["samples"] if s["status"] == "ok"]
    mean_p, cov_p = _textbook_posterior(log["plan"]["problem"])
    accepted_replicas, accepted_batches, kl_gap = 0, 0, 0.0
    reported = {row["batch_index"]: row["kl_nats"] for row in analysis["phases"][3]["batches"]}
    for batch in phase["batches"]:
        # ciw.constant-float64-row.v1: one row [mean_0, mean_1, cov_00, cov_01, cov_10, cov_11] for every replica.
        values = np.asarray(batch["result"]["values"], dtype=float)
        kl = _textbook_kl(values[:2], values[2:].reshape(2, 2), mean_p, cov_p)
        kl_gap = max(kl_gap, abs(kl - reported[batch["batch_index"]]))
        if kl <= log["plan"]["target_kl_nats"]:
            accepted_replicas += batch["result"]["replicas"]
            accepted_batches += 1
    delta_j = (readings[-1] - readings[0]) / 1000 if len(readings) >= 2 else None
    all_readings = [int(s["energy_mj"]) for p in log["phases"] for s in p["samples"] if s["status"] == "ok"]
    return {"delta_j": delta_j, "accepted": accepted_replicas, "accepted_batches": accepted_batches,
            "executed": sum(b["result"]["replicas"] for b in phase["batches"]),
            "replicas_per_batch": log["plan"]["replicas"], "max_kl_difference_to_analysis": kl_gap,
            "span_j": (all_readings[-1] - all_readings[0]) / 1000,
            "analysis_value": analysis["measurement"]["amortized_domain_energy_j_per_qualified_solve"],
            "reasons": analysis["comparison"]["reasons"], "gross_energy_j": analysis["measurement"]["gross_energy_j"]}


STATIC_T119 = {
    "hypothesis": "Energy per accepted result is well defined only when counter brackets are valid and every "
                  "counted solve meets the declared accuracy target; otherwise it must be withheld. Its denominator "
                  "must say whether it counts replica solves or distinct results.",
    "mathematical_model": "E_acc = Delta E_measurement / #{replica solves with KL(q || p) <= target}; replicas in a "
                          "batch are bitwise-identical copies of one result (constant-row encoding), so E per distinct "
                          "accepted result = Delta E / #{accepted batch outputs}; boundary-dependent (measurement "
                          "phase only, gross, no idle subtraction).",
    "input_data": [FIXTURE_NOTE],
    "observation_model": "Counter readings and retained batch outputs from each fixture; outputs decoded directly "
                         "from the constant-row values and scored with a textbook Gaussian KL against an "
                         "information-form posterior written in this module.",
    "expected_invariant": "Recomputation equals energy_records.analyze; the metric is withheld for reset, missing "
                          "bracket and under-target fixtures.",
    "experiment": "Analyze all four fixtures; recompute the baseline metric from raw readings and raw outputs; compare "
                  "with the naive gross/executed ratio, with the per-distinct-result denominator and with a whole-run "
                  "boundary.",
    "failure_modes_checked": ["counter reset", "missing endpoint brackets", "accuracy target not met",
                              "naive division by executed solves", "replica copies counted as distinct results",
                              "boundary choice"],
    "unresolved_assumptions": ["The physical measurement part (T116) could not run here",
                               "Idle subtraction and first-attainment accounting are intentionally not applied",
                               "The recomputation shares its origin (ciw) with energy_records, so it is a "
                               "cross-implementation check, not independent verification"],
    "recommended_next_task": "T116 on the RTX 2080 host, then rerun T119 on the captured log",
}


def _t119_physical():
    return _no_measurement("Physical GPU energy per accepted numerical result",
                           "the fixtures are synthetic and no GPU counter was read here", unit="J/accepted solve")


@task("T119", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_energy_per_accepted_result_on_fixtures", f"{TESTS}::test_textbook_kl_matches_closed_form_values",
    FIXTURELESS_TEST, UNCERTAINTY_TEST))
def energy_per_accepted_result(ctx):
    from ciw import energy_records
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked(STATIC_T119, _t119_physical)
    rows = {}
    for name, raw in raws.items():
        log = json.loads(raw)
        rows[name] = _accepted_energy(log, energy_records.analyze(log))
    base = rows["baseline"]
    recomputed = base["delta_j"] / base["accepted"]
    per_distinct = base["delta_j"] / base["accepted_batches"]
    generator = {"name": "examples/energy-accuracy synthetic fixtures", "origin": "synthetic_fixture", "seed": None}
    withheld = {name: row["reasons"] for name, row in rows.items() if row["analysis_value"] is None}
    under = rows["under-target"]
    naive = under["gross_energy_j"] / under["executed"]
    ratio = (base["span_j"] / base["accepted"]) / recomputed
    findings = [
        finding("Energy per accepted replica solve of the synthetic baseline fixture, recomputed from raw counter "
                "readings, directly decoded outputs and a textbook Gaussian KL, equals the CIW analysis value",
                "numerical", recomputed,
                {"generator": dict(generator, fixture="baseline"),
                 "checks": [_check("energy_records.analyze amortized_domain_energy_j_per_qualified_solve",
                                   recomputed - base["analysis_value"], 1e-15, kind="cross_implementation"),
                            _check("largest per-batch KL difference to energy_records.analyze",
                                   base["max_kl_difference_to_analysis"], 1e-12, kind="cross_implementation")]},
                unit="J per accepted replica solve (synthetic fixture values)", uncertainty=SYNTHETIC,
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Energy per distinct accepted result (one batch output; its replicas are bitwise copies) of the "
                "synthetic baseline fixture", "numerical", per_distinct,
                {"generator": dict(generator, fixture="baseline"),
                 "checks": [_check("accepted replica solves minus replicas per batch times accepted batch outputs",
                                   base["accepted"] - base["replicas_per_batch"] * base["accepted_batches"], 0.0,
                                   kind="exact_arithmetic")]},
                unit="J per distinct accepted result (synthetic fixture values)", uncertainty=SYNTHETIC,
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("The metric is withheld for the reset, missing-bracket and under-target fixtures", "computational_pipeline",
                withheld, {"generator": generator,
                           "checks": [_check("three fixtures with a declared defect", len(withheld) - 3, 0.0,
                                             kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Dividing gross energy by executed solves reports a finite energy per result for the under-target "
                "fixture although no result is accepted", "numerical",
                {"gross_energy_j": under["gross_energy_j"], "executed_solves": under["executed"],
                 "accepted_solves": under["accepted"], "naive_j_per_solve": naive},
                {"generator": dict(generator, fixture="under-target"),
                 "checks": [_check("accepted solves in the under-target fixture", under["accepted"], 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty=SYNTHETIC, tolerance={"abs": 1e-12, "rel": 0.0},
                counterexample={"statement": "Gross energy divided by executed solves is an energy per accepted "
                                             "numerical result",
                                "witness": {"fixture": "under-target", "naive_j_per_solve": naive, "accepted_solves": 0}}),
        finding("Widening the boundary from the measurement phase to the whole run multiplies the baseline energy per "
                "accepted result by 7", "numerical", ratio,
                {"generator": dict(generator, fixture="baseline"),
                 "checks": [_check("(2400 - 1000) mJ / (2100 - 1900) mJ", ratio - 7.0, 1e-12, kind="exact_arithmetic")]},
                unit="ratio", uncertainty=SYNTHETIC, tolerance={"abs": 1e-12, "rel": 0.0}),
        _t119_physical(),
    ]
    ctx.artifact_json("energy-per-accepted-result.json", {
        "definition": "E_acc = (counter(last measurement read) - counter(first measurement read)) / accepted replica "
                      "solves; a replica solve is accepted when its batch output has KL <= target_kl_nats; undefined "
                      "when the analysis is ineligible or no solve is accepted. The denominator counts replica solves "
                      "(identical thread work on one declared problem), not distinct numerical results: every "
                      "replica of a batch is a bitwise copy of one output. E per distinct accepted result divides by "
                      "accepted batch outputs instead.",
        "fixtures": rows, "origin": "synthetic_fixture"})
    fields = _fields(**dict(
        STATIC_T119,
        numerical_result=f"baseline E_acc = {recomputed} J per accepted replica solve and {per_distinct} J per distinct "
                         f"accepted result (synthetic); withheld for {sorted(withheld)}; naive under-target ratio "
                         f"{naive} J/solve with 0 accepted; whole-run boundary ratio {ratio:.6g}.",
        uncertainty="Synthetic values carry no physical uncertainty model; a real NVML counter has undeclared "
                    "resolution and background-inclusive scope."))
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T120
PRECISION_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048)
TARGETS = (1e-4, 1e-5, 1e-7, 1e-10, 1e-11)


def _precision_study():
    states = kernels.initial_states()
    table = {}
    for dtype in (np.float32, np.float64):
        errors = []
        for steps in PRECISION_GRID:
            final = kernels.rk4_batch(states, kernels.LENGTH, steps, dtype)
            errors.append(float(np.max(kernels.endpoint_errors(states, final.astype(np.float64), kernels.LENGTH))))
        table[np.dtype(dtype).name] = errors
    return table


def _slope(steps, errors):
    x = np.log(np.asarray(steps, dtype=float))
    coefficients, residuals, *_ = np.polyfit(x, np.log(np.asarray(errors)), 1, full=True)
    rms = float(np.sqrt(residuals[0] / len(x))) if len(residuals) else 0.0
    return float(-coefficients[0]), rms


@task("T120", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_precision_study_float32_floor_and_counterexample", UNCERTAINTY_TEST))
def precision_versus_cost(ctx):
    table = ctx.memo("energy-gpu-precision", _precision_study)
    f32, f64 = table["float32"], table["float64"]
    fit = [i for i, n in enumerate(PRECISION_GRID) if n <= 256]
    order64, fit_rms = _slope([PRECISION_GRID[i] for i in fit], [f64[i] for i in fit])
    best = int(np.argmin(f32))
    min32 = f32[best]
    plateau = float(np.median([e for n, e in zip(PRECISION_GRID, f32) if n >= 128]))
    ops = kernels.rk4_operation_count()
    counted = kernels.counted_operations_per_step()
    reach = {}
    for target in TARGETS:
        reach[f"{target:g}"] = {name: next((n for n, e in zip(PRECISION_GRID, errs) if e <= target), None)
                                for name, errs in table.items()}
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "grid": list(PRECISION_GRID), "seed": None}
    reached64 = sum(row["float64"] is not None for row in reach.values())
    float32_scatter = {"kind": "roundoff", "value": None,
                       "basis": "float32 sin/cos implementations differ by platform; plateau errors vary by tens of "
                                "percent (regression tolerance rel 1.0)"}
    findings = [
        finding("float64 RK4 endpoint error converges at order 4 on the sphere geodesics (N = 16..256)", "numerical",
                order64, {"generator": generator,
                          "checks": [_check("RK4 global order 4", order64 - 4.0, 0.3)]},
                unit="order", uncertainty={"kind": "fit_residual", "value": fit_rms,
                                           "basis": "rms residual of the least-squares log-log fit (natural log)"},
                tolerance={"abs": 0.05, "rel": 0.0}),
        finding("float32 RK4 endpoint error stops improving: its minimum is a truncation/roundoff crossover inside the "
                "step grid, followed by a roundoff-dominated plateau above it", "numerical",
                {"crossover_min_error": min32, "crossover_steps": PRECISION_GRID[best],
                 "plateau_median_error_n_ge_128": plateau, "error_at_2048": f32[-1], "float64_error_at_2048": f64[-1]},
                {"generator": generator,
                 "checks": [_check("grid positions after the float32 minimum", len(f32) - 1 - best, 1.0, "ge",
                                   kind="exact_arithmetic"),
                            _check("float32 error at N = 2048 over its minimum", f32[-1] / min32, 2.0, "ge",
                                   kind="self_convergence"),
                            _check("float32 over float64 error at N = 2048 (roundoff, not truncation)",
                                   f32[-1] / f64[-1], 1e4, "ge", kind="high_precision"),
                            _check("float32 minimum above the unit roundoff 2^-24 = 6.0e-8", min32, 1e-7, "ge")]},
                unit="normalized length", uncertainty=float32_scatter, tolerance={"abs": 1e-12, "rel": 1.0}),
        finding("Smallest grid N (>= 16) meeting each accuracy target, by precision (None: not reached for N <= 2048; "
                "16 is censored at the grid minimum)", "numerical", reach,
                {"generator": generator,
                 "checks": [_check("float64 reaches all five targets", reached64 - len(TARGETS), 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty={"kind": "grid_resolution", "value": 2.0,
                             "basis": "the true minimal N lies within a factor 2 below the tabulated grid value"},
                tolerance={"abs": 0.0, "rel": 1.0}),
        finding("Lowering precision to float32 cannot reach a 1e-7 endpoint accuracy at any step count up to 2048, "
                "while float64 reaches it", "numerical", {"float32_min_error": min32, "float64_steps": reach["1e-07"]["float64"]},
                {"generator": generator,
                 "checks": [_check("grid points where float32 meets 1e-7", sum(e <= 1e-7 for e in f32), 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty=float32_scatter, tolerance={"abs": 0.0, "rel": 1.0},
                counterexample={"statement": "Halving floating-point precision reaches every accuracy target at no "
                                             "greater operation count",
                                "witness": {"target": 1e-7, "float32_min_error": min32,
                                            "float64_steps": reach["1e-07"]["float64"]}}),
        finding("Per RK4 step the kernel performs the same counted arithmetic in both precisions (the float32 and "
                "float64 transcendental implementations differ)",
                "numerical", dict(ops, state_bytes={"float32": 16, "float64": 32}, instrumented=counted),
                {"generator": dict(generator, instrumented_trajectories=1),
                 "checks": [_check("instrumented flops minus the count derived from the source: 4 x (6 mul + 1 div) "
                                   "+ 3 x 4 x 2 stage + 4 x 5 + 4 x 2 combine", counted["flops"] - ops["flops_per_step"],
                                   0.0, kind="exact_arithmetic"),
                            _check("instrumented sin/cos calls minus 4 x 2", counted["transcendentals"]
                                   - ops["transcendentals_per_step"], 0.0, kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        _no_measurement("float32 lowers the energy per accepted trajectory relative to float64 on real CPU or GPU "
                        "hardware", "no energy counter is available; operation counts are only a proxy",
                        unit="J/trajectory"),
    ]
    ctx.artifact_json("precision.json", {"grid": list(PRECISION_GRID), "max_endpoint_error": table,
                                         "targets": reach, "operation_count": ops,
                                         "float64_order_fit": {"order": order64, "rms_residual": fit_rms}})
    ctx.artifact_text("precision.svg", svg.line_plot(
        [(name, list(PRECISION_GRID), errs) for name, errs in table.items()],
        title="RK4 sphere geodesic: endpoint error vs steps", xlabel="steps N", ylabel="max endpoint error",
        logx=True, logy=True))
    fields = _fields(
        hypothesis="float32 RK4 matches float64 until truncation error reaches float32 roundoff, after which more "
                   "steps cannot buy accuracy; the counted operations per step are precision-independent.",
        mathematical_model="Global error ~ C h^4 + c N u with u = 2^-24 (float32) or 2^-53 (float64); cost ~ 4 N "
                           "right-hand-side evaluations of fixed counted arithmetic.",
        input_data=[f"{len(kernels.HEADINGS)} equatorial sphere geodesics, L = {kernels.LENGTH}, N in {list(PRECISION_GRID)}"],
        observation_model="Endpoints computed entirely in the declared dtype, mapped to R^3 and compared in float64 "
                          "with the exact great circle; maximum over trajectories.",
        expected_invariant="float64 order near 4; float32 minimum inside the grid, above float32 roundoff and followed "
                           "by a roundoff plateau; counted operations per step equal.",
        experiment="Vectorized RK4 in float32 and float64 across the step grid; smallest grid N per accuracy target; "
                   "operation counts derived from the kernel source and checked by running one step of the same code "
                   "on counting scalars.",
        numerical_result=f"float64 order {order64:.3f}; float32 crossover minimum {min32:.3g} at N = "
                         f"{PRECISION_GRID[best]}, plateau median {plateau:.3g} for N >= 128; smallest grid N per "
                         f"target {reach}.",
        uncertainty="float32 errors depend on the platform's float32 sin/cos and vary by tens of percent; "
                    "float64 order fit is stable to about 0.05.",
        failure_modes_checked=["dtype promotion to float64 (asserted)", "roundoff floor mistaken for convergence "
                               "(minimum must lie inside the grid)", "target unreachable at any N (counterexample)"],
        unresolved_assumptions=["Energy cost was not measured; operation counts do not capture memory traffic, "
                                "vector width, transcendental cost or GPU float32 throughput"],
        recommended_next_task="T115/T116 energy measurement of both precisions on hardware; T148 for deterministic "
                              "reduction policies")
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T121
REDUCTION_N = 1 << 14
REDUCTION_SEED = 20260923
# The float32 recipe shifted by float64's 29 extra mantissa bits, so partial sums again swamp 2^-8.
NATIVE64_SCALE = 2048.0 * 2.0 ** 29
ACCEPTANCE_TOLERANCE = 0.01


def _atomic_orders(generator, count):
    return [generator.permutation(REDUCTION_N // kernels.BLOCK) for _ in range(count)]


def _cancellation_data(seed, dtype=np.float32, scale=2048.0):
    """Pairs +a, -a plus 64 terms of 2^-8: the exact sum is 0.25, the partial sums are large."""
    generator = np.random.Generator(np.random.PCG64(seed))
    half = (REDUCTION_N - 64) // 2
    a = ((generator.random(half) - 0.5) * scale).astype(dtype)
    values = np.concatenate([a, -a[generator.permutation(half)], np.full(64, 2.0 ** -8, dtype=dtype)])
    return values[generator.permutation(REDUCTION_N)]


def _first_sign_flip(dtype, scale):
    """First seed whose sequential, strided-tree and adjacent-tree sums disagree in sign (only these are searched)."""
    for seed in range(1, 400):
        candidate = _cancellation_data(seed, dtype, scale)
        signs = {np.sign(kernels.sum_sequential(candidate)), np.sign(kernels.sum_tree_strided(candidate)),
                 np.sign(kernels.sum_tree_adjacent(candidate))}
        if len(signs) > 1:
            return seed, candidate
    raise ValueError("No reduction-order sign counterexample was found in the searched seeds")


def _entry(data, dtype, atomic):
    x = data.astype(dtype)
    exact = math.fsum(float(v) for v in data)
    sums = kernels.reduction_orders(x, atomic)
    bounds = kernels.order_bounds(x, dtype)
    errors = {k: v - exact for k, v in sums.items()}
    fractions = {k: abs(e) / kernels.bound_for(k, bounds) for k, e in errors.items()}
    worst = max(fractions, key=lambda k: (fractions[k], k))
    return {"exact": exact, "sums": sums, "errors": errors, "bound": kernels.summation_bound(x, dtype),
            "order_bounds": bounds, "fraction_of_order_bound": fractions, "worst_order": worst,
            "worst_fraction": fractions[worst], "exact_zero_errors": sum(e == 0 for e in errors.values())}


def _reduction_study():
    generator = np.random.Generator(np.random.PCG64(REDUCTION_SEED))
    u = generator.random(REDUCTION_N)
    positive = (u * u * u * 1000.0 + 1e-3).astype(np.float32)
    atomic = _atomic_orders(generator, 8)
    seed32, cancellation = _first_sign_flip(np.float32, 2048.0)
    seed64, native = _first_sign_flip(np.float64, NATIVE64_SCALE)
    study = {}
    for label, data in (("positive", positive), ("cancellation", cancellation)):
        for dtype in (np.float32, np.float64):
            study[f"{label}-{np.dtype(dtype).name}"] = _entry(data, dtype, atomic)
    study["cancellation-native-float64"] = _entry(native, np.float64, atomic)
    maxima = {float(np.max(positive[generator.permutation(REDUCTION_N)])) for _ in range(8)}
    zeros = [np.array([0.0, -0.0]), np.array([-0.0, 0.0])]
    signed_zero = [bool(np.signbit(np.max(pair))) for pair in zeros]
    return {"study": study, "witness_seed": seed32, "native_float64_seed": seed64, "max_values": sorted(maxima),
            "signed_zero_max_signbits": signed_zero, "n": REDUCTION_N, "block": kernels.BLOCK,
            "atomic_orders": len(atomic), "order_depths": kernels.order_depths(REDUCTION_N)}


def guarded_sign(entry) -> dict:
    """Decide the sign of S only when |S| exceeds the order's own a priori bound; count decided orders."""
    decided = {k: abs(s) > kernels.bound_for(k, entry["order_bounds"]) for k, s in entry["sums"].items()}
    return {"decided_orders": sum(decided.values()), "orders": len(decided),
            "smallest_order_bound": min(kernels.bound_for(k, entry["order_bounds"]) for k in decided),
            "largest_abs_sum": max(abs(s) for s in entry["sums"].values())}


def _bound_finding(claim, entry, generator):
    values = list(entry["sums"].values())
    value = {"relative_spread": (max(values) - min(values)) / abs(entry["exact"]),
             "worst_fraction_of_order_bound": entry["worst_fraction"], "worst_order": entry["worst_order"]}
    return finding(claim, "numerical", value,
                   {"generator": generator,
                    "checks": [_check("order-specific a priori bounds gamma_d sum|x| (Higham 4.2; Kahan 4.3, "
                                      "leading order)", entry["worst_fraction"], 1.0, "le")]},
                   unit="relative", uncertainty=IEEE, tolerance={"abs": 1e-15, "rel": 0.5})


@task("T121", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_reduction_orders_bound_and_sign_counterexample", UNCERTAINTY_TEST))
def reduction_order_conclusions(ctx):
    result = ctx.memo("energy-gpu-reductions", _reduction_study)
    study = result["study"]
    generator = {"name": "seeded reduction datasets", "seed": REDUCTION_SEED, "n": REDUCTION_N,
                 "cancellation_seed": result["witness_seed"], "native_float64_seed": result["native_float64_seed"],
                 "bit_generator": "PCG64"}
    cancel32, cancel64 = study["cancellation-float32"], study["cancellation-float64"]
    native = study["cancellation-native-float64"]
    signs32 = {k: int(np.sign(v)) for k, v in cancel32["sums"].items()}
    signs_native = {k: int(np.sign(v)) for k, v in native["sums"].items()}
    atomic_cancel = {k: v for k, v in cancel32["sums"].items() if k.startswith("atomic-")}
    decisions = {k: abs(v - cancel32["exact"]) <= ACCEPTANCE_TOLERANCE for k, v in atomic_cancel.items()}
    atomic32 = {v for k, v in study["positive-float32"]["sums"].items() if k.startswith("atomic-")}
    guards = {key: guarded_sign(study[key]) for key in ("cancellation-float32", "cancellation-native-float64")}
    findings = [
        _bound_finding("Every emulated float32 summation order of the positive dataset stays within its "
                       "order-specific a priori error bound; relative spread across orders",
                       study["positive-float32"], generator),
        _bound_finding("Every emulated float64 summation order of the same data stays within its order-specific "
                       "bound; relative spread", study["positive-float64"], generator),
        finding("Reduction order alone flips the sign of a float32 sum whose exact value is +0.25 (the one-thread "
                "sequential CPU fold against the GPU-style orders)", "numerical",
                signs32, {"generator": generator,
                          "checks": [_check("distinct signs across emulated orders", len(set(signs32.values())), 2.0,
                                            "ge", kind="exact_arithmetic")]},
                uncertainty=IEEE, tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "The sign of a float32 reduction (a pass/fail decision at threshold 0) "
                                             "does not depend on the reduction order",
                                "witness": {"seed": result["witness_seed"], "found_by": "first flipping seed of 1..399 "
                                            "comparing sequential, strided-tree and adjacent-tree sums",
                                            "exact_sum": cancel32["exact"], "sums": cancel32["sums"]}}),
        finding("float64 accumulation of these float32-valued cancellation inputs is exact in every emulated order "
                "(all errors 0), hence sign-correct", "numerical",
                {"max_abs_error": max(abs(e) for e in cancel64["errors"].values()),
                 "exact_zero_errors": cancel64["exact_zero_errors"]},
                {"generator": generator,
                 "checks": [_check("largest |float64 sum - exact| over orders",
                                   max(abs(e) for e in cancel64["errors"].values()), 0.0, kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Reduction order alone flips the sign of a float64 sum of float64-native cancellation data whose "
                "exact value is +0.25 (the sequential fold against the GPU-style orders)", "numerical", signs_native,
                {"generator": generator,
                 "checks": [_check("distinct signs across emulated float64 orders", len(set(signs_native.values())),
                                   2.0, "ge", kind="exact_arithmetic")]},
                uncertainty=IEEE, tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "float64 reductions are order-robust for sign decisions",
                                "witness": {"seed": result["native_float64_seed"], "scale": NATIVE64_SCALE,
                                            "exact_sum": native["exact"], "sums": native["sums"]}}),
        finding(f"Atomic completion order alone changes a float32 pass/fail test |S - 0.25| <= {ACCEPTANCE_TOLERANCE} "
                "on identical inputs", "numerical", {k: bool(v) for k, v in decisions.items()},
                {"generator": generator,
                 "checks": [_check("distinct pass/fail outcomes over 8 atomic completion orders",
                                   len(set(decisions.values())), 2.0, "ge", kind="exact_arithmetic")]},
                uncertainty=IEEE, tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "Atomic completion order cannot change a float32 pass/fail decision on "
                                             "identical inputs",
                                "witness": {"tolerance": ACCEPTANCE_TOLERANCE, "exact_sum": cancel32["exact"],
                                            "failing": {k: v for k, v in atomic_cancel.items() if not decisions[k]},
                                            "passing": {k: v for k, v in atomic_cancel.items() if decisions[k]}}}),
        finding("Emulated atomicAdd completion orders of identical float32 block partials give distinct sums",
                "numerical", len(atomic32),
                {"generator": generator,
                 "checks": [_check("distinct float32 results over 8 completion orders", len(atomic32), 2.0, "ge",
                                   kind="exact_arithmetic")]},
                unit="distinct results", uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "Accumulating the same block partials yields the same float32 result "
                                             "whatever the atomic completion order",
                                "witness": {"distinct_results": sorted(atomic32)}}),
        finding("The bound-guarded sign test (decide only when |S| exceeds the order's own a priori bound) cannot "
                "contradict across orders but leaves both cancellation sums undecided in every emulated order",
                "numerical", guards,
                {"generator": generator,
                 "checks": [_check("orders whose guarded sign test decides",
                                   sum(g["decided_orders"] for g in guards.values()), 0.0, kind="exact_arithmetic")]},
                uncertainty=IEEE, tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Max reductions of this NaN-free, zero-free positive data are bitwise order-invariant over 8 "
                "permutations", "numerical", len(result["max_values"]),
                {"generator": generator,
                 "checks": [_check("distinct maxima minus one", len(result["max_values"]) - 1, 0.0,
                                   kind="exact_arithmetic")]},
                unit="distinct results", uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("The IEEE maximum of +0.0 and -0.0 depends on operand order (numpy max)", "numerical",
                result["signed_zero_max_signbits"],
                {"checks": [_check("distinct sign bits of max over the two operand orders",
                                   len(set(result["signed_zero_max_signbits"])), 2.0, "ge", kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "Max reductions are bitwise order-invariant for all IEEE inputs",
                                "witness": {"orders": ["[0.0, -0.0]", "[-0.0, 0.0]"],
                                            "signbits": result["signed_zero_max_signbits"]}}),
        _no_measurement("Reductions on the RTX 2080 (CUB, cuBLAS or atomicAdd) reproduce these emulated spreads and "
                        "sign flips", "no GPU was available; the orders are CPU emulations"),
    ]
    names = list(study["positive-float32"]["sums"])
    ctx.artifact_json("reductions.json", {**result, "figure_order": names, "guarded_sign": guards,
                                          "atomic_pass_fail": decisions})
    ctx.artifact_text("reduction-errors.svg", svg.line_plot(
        [(key, list(range(len(names))), [abs(entry["errors"][n]) for n in names]) for key, entry in study.items()],
        title="Absolute error by emulated reduction order (exact zeros omitted)",
        xlabel="order index (figure_order in reductions.json)", ylabel="|sum - exact|", logy=True))
    spread = findings[0]["value"]["relative_spread"], findings[1]["value"]["relative_spread"]
    fields = _fields(
        hypothesis="GPU-style reduction orders keep float32 sums within their order-specific a priori bounds, yet the "
                   "order differences suffice to flip sign and pass/fail decisions near their boundary; float64 "
                   "suffers the same with data scaled to its precision.",
        mathematical_model="|fl(sum x) - sum x| <= gamma_d sum|x_i|, gamma_d = d u / (1 - d u), where d is the "
                           "largest number of additions any input passes through (n - 1 sequential, log2 n trees, "
                           "block height plus partial fold otherwise); Kahan (2u + n u^2) sum|x| to leading order.",
        input_data=[f"n = {REDUCTION_N}: positive data (u^3 * 1000 + 1e-3 from PCG64 seed {REDUCTION_SEED}); float32 "
                    f"cancellation data (+a, -a pairs plus 64 x 2^-8, first flipping seed {result['witness_seed']}); "
                    f"float64-native cancellation data (scale 2^40, first flipping seed "
                    f"{result['native_float64_seed']})"],
        observation_model="Sums in sequential, adjacent tree, strided tree, two-pass blocked, block-sequential, "
                          "Kahan and 8 atomic completion orders; exact reference by math.fsum.",
        expected_invariant="Every error within its order-specific bound; guarded decisions cannot contradict (if "
                           "|S_i - s| <= b and |S_j - s| <= b, then S_i - T > b and T - S_j > b would need S_i - S_j > "
                           "2b, which the triangle inequality excludes); max of NaN-free, zero-free data "
                           "order-invariant.",
        experiment="Emulate the orders in float32 and float64; search seeds for a sign flip among the sequential and "
                   "two tree orders; test a pass/fail tolerance across atomic orders; test the guarded sign "
                   "decision; compare max reductions, including signed zeros.",
        numerical_result=f"float32 relative spread {spread[0]:.3g}; float64 {spread[1]:.3g}; float32 cancellation "
                         f"signs {signs32}; float64-native cancellation signs {signs_native}; atomic pass/fail "
                         f"{decisions}; {len(atomic32)} distinct atomic-order results on positive data.",
        uncertainty="Deterministic IEEE arithmetic; the emulated orders are a sample of plausible GPU orders, not "
                    "those of a specific library; the witnesses were found by search, so they show existence, not "
                    "frequency.",
        failure_modes_checked=["error exceeds its order-specific bound", "no sign counterexample in 399 seeds "
                               "(would raise)", "atomic-order pass/fail flip", "guarded sign test deciding wrongly",
                               "max reduction order dependence (including signed zeros)"],
        unresolved_assumptions=["Real GPU reduction orders, FMA contraction and warp-shuffle trees were not observed",
                                "The pass/fail tolerance 0.01 was chosen inside the observed atomic spread; it shows "
                                "that such flips exist, not how often they occur",
                                "The a priori guard is sound but so conservative that it decides none of the "
                                "cancellation signs"],
        recommended_next_task="T148: adopt a deterministic reduction policy (fixed tree or compensated) and T147 to "
                              "compare against RTX 2080 outputs")
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T122
RAW_PROBLEM = {"prior_mean": [0.4, -1.0], "prior_covariance": [[4.0, 0.1], [0.1, 0.25]],
               "observation_matrix": [[3.0, 0.2], [-0.1, 8.0]], "observations": [12.0, -0.9],
               "noise_covariance": [[9.0, 0.06], [0.06, 0.04]]}
DECLARED_SCALES = {"latent_scales": [0.7, 0.05], "observation_scales": [3.0, 0.2]}
UNIT_SCALES = {"latent_scales": [1.0, 1.0], "observation_scales": [1.0, 1.0]}
ALTERNATE_SCALES = {"latent_scales": [2.0, 0.2], "observation_scales": [1.0, 0.5]}
FIT = {"initial_mean": [0.0, 0.0], "initial_covariance": [[1.0, 0.0], [0.0, 1.0]], "beta": 0.3,
       "max_iterations": 512, "gradient_tolerance": 1e-10, "precision_tolerance": 1e-10}


def _fit(scales, alpha_rule="optimal", max_iterations=None):
    from ciw import free_energy_math
    normalized = free_energy_math.normalize_problem(**RAW_PROBLEM, **scales)
    problem = normalized["problem"]
    eigen = np.linalg.eigvalsh(np.asarray(free_energy_math.information_system(problem)["precision"]))
    alpha = 2.0 / (eigen[0] + eigen[-1]) if alpha_rule == "optimal" else 1.2 * 2.0 / eigen[-1]
    settings = dict(FIT, alpha=float(alpha))
    if max_iterations is not None:
        settings["max_iterations"] = max_iterations
    fit = free_energy_math.variational_fit(problem, **settings)
    return {"normalized": normalized, "fit": fit, "alpha": float(alpha), "condition": float(eigen[-1] / eigen[0])}


def _raw_posterior(run):
    scales = np.diag(run["normalized"]["normalization"]["latent_scales"])
    reference = run["fit"]["reference"]
    return (scales @ np.asarray(reference["mean"]), scales @ np.asarray(reference["covariance"]) @ scales,
            reference["log_evidence"] - run["normalized"]["normalization"]["log_observation_jacobian"])


@task("T122", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_bounded_free_energy_identity_and_counterexamples", UNCERTAINTY_TEST))
def bounded_free_energy(ctx):
    runs = ctx.memo("energy-gpu-free-energy", lambda: {
        "declared": _fit(DECLARED_SCALES), "unit": _fit(UNIT_SCALES), "alternate": _fit(ALTERNATE_SCALES),
        "unstable": _fit(DECLARED_SCALES, alpha_rule="unstable", max_iterations=60)})
    fit = runs["declared"]["fit"]
    log_z = fit["reference"]["log_evidence"]
    trace = fit["trace"]
    residual = max(abs(row["free_energy"] + log_z - row["kl_to_reference"]) for row in trace)
    kl = [row["kl_to_reference"] for row in trace]
    increase = max(b - a for a, b in zip(kl, kl[1:]))
    posteriors = [_raw_posterior(runs[name]) for name in ("declared", "unit", "alternate")]
    invariance = max(max(float(np.max(np.abs(p[0] - posteriors[0][0]))), float(np.max(np.abs(p[1] - posteriors[0][1]))),
                         abs(p[2] - posteriors[0][2])) for p in posteriors[1:])
    unstable = runs["unstable"]
    unstable_kl = [row["kl_to_reference"] for row in unstable["fit"]["trace"]]
    unit = runs["unit"]
    generator = {"name": "declared two-latent linear-Gaussian problem", "raw_problem": RAW_PROBLEM,
                 "scales": DECLARED_SCALES, "seed": None}
    roundoff = {"kind": "roundoff", "value": residual,
                "basis": "largest binary64 residual of F + log Z - KL over the retained trace"}
    findings = [
        finding("The variational free-energy identity F + log Z = KL(q || p) holds at every iterate to 1e-10 nats",
                "numerical", residual,
                {"generator": generator,
                 "checks": [_check("F from free_energy, log Z from observation-space conditioning, KL from the "
                                   "whitened Gaussian formula", residual, 1e-10, kind="invariant")]},
                unit="nat", uncertainty=roundoff, tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("KL to the exact posterior decreases monotonically and ends below 1e-12 nats within the "
                "512-iteration bound", "numerical",
                {"iterations": fit["iterations"], "final_kl_nats": kl[-1], "initial_kl_nats": kl[0],
                 "status": fit["status"], "condition_number": runs["declared"]["condition"]},
                {"generator": generator,
                 "checks": [_check("final KL", kl[-1], 1e-12, "le"),
                            _check("largest KL increase between iterates", increase, 1e-15, "signed_le", kind="invariant"),
                            _check("iterations within the declared bound", fit["iterations"], 512, "le",
                                   kind="exact_arithmetic")]},
                uncertainty=roundoff, tolerance={"abs": 1e-12, "rel": 0.05}),
        finding("The raw-unit posterior and log evidence are invariant under the choice of normalization scales",
                "numerical", invariance,
                {"generator": generator,
                 "checks": [_check("posterior mapped back by x_raw = D x_norm, log Z_raw = log Z_norm - sum log T",
                                   invariance, 1e-10, kind="invariant")]},
                uncertainty={"kind": "roundoff", "value": invariance, "basis": "binary64 back-transformation"},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("A mean step 1.2 times the stability bound makes KL grow instead of converge", "numerical",
                {"alpha": unstable["alpha"], "spectral_radius": unstable["fit"]["stability"]["spectral_radius"],
                 "kl_initial": unstable_kl[0], "kl_final": unstable_kl[-1]},
                {"generator": generator,
                 "checks": [_check("KL growth factor over 60 iterations", unstable_kl[-1] / unstable_kl[0], 1e3, "ge")]},
                uncertainty={"kind": "roundoff", "value": None, "basis": "geometric growth amplifies roundoff; the "
                                                                        "final KL is reproducible to about 1e-3 "
                                                                        "relative"},
                tolerance={"abs": 0.0, "rel": 1e-3},
                counterexample={"statement": "Gradient descent on the variational free energy converges for every "
                                             "positive step size",
                                "witness": {"alpha": unstable["alpha"],
                                            "stable_bound": unstable["fit"]["stability"]["stable_step_size_upper_bound"]}}),
        finding("Unit scales leave the same problem unconverged after 512 iterations (condition number 1.3e3)",
                "numerical", {"final_kl_nats": unit["fit"]["trace"][-1]["kl_to_reference"],
                              "iterations": unit["fit"]["iterations"], "condition_number": unit["condition"]},
                {"generator": dict(generator, scales=UNIT_SCALES),
                 "checks": [_check("final KL with unit scales", unit["fit"]["trace"][-1]["kl_to_reference"], 1e-3, "ge")]},
                uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "binary64 iteration of a contraction"},
                tolerance={"abs": 1e-9, "rel": 1e-4},
                counterexample={"statement": "Coordinate normalization changes only units, not bounded convergence",
                                "witness": {"declared_iterations": fit["iterations"],
                                            "unit_final_kl_nats": unit["fit"]["trace"][-1]["kl_to_reference"]}}),
        _no_measurement("The variational free energy of this model equals a thermodynamic free energy of a physical "
                        "system", "the model is a normalized, dimensionless inference problem; no physical system "
                        "or temperature is declared", unit="nat"),
    ]
    ctx.artifact_json("free-energy-trace.json", {
        "raw_problem": RAW_PROBLEM, "scales": {"declared": DECLARED_SCALES, "unit": UNIT_SCALES,
                                               "alternate": ALTERNATE_SCALES},
        "alpha": {k: v["alpha"] for k, v in runs.items()}, "log_evidence_normalized": log_z,
        "normalization": runs["declared"]["normalized"]["normalization"],
        "trace": [{"iteration": r["iteration"], "free_energy": r["free_energy"], "kl_to_reference": r["kl_to_reference"],
                   "identity_residual": r["free_energy"] + log_z - r["kl_to_reference"]} for r in trace]})
    ctx.artifact_text("free-energy.svg", svg.line_plot(
        [(f"{name} scales", [r["iteration"] for r in runs[name]["fit"]["trace"]],
          [r["kl_to_reference"] for r in runs[name]["fit"]["trace"]]) for name in ("declared", "unit", "alternate")],
        title="KL(q || posterior) during bounded variational descent", xlabel="iteration", ylabel="KL (nat)",
        logy=True, markers=False))
    fields = _fields(
        hypothesis="Bounded natural-gradient/Euclidean descent on the Gaussian variational free energy reaches the "
                   "exact posterior, with F + log Z = KL at every iterate; convergence speed depends on normalization.",
        mathematical_model="F(q) = E_q[-log p(y|x)] + KL(q || p(x)) = KL(q || p(x|y)) - log p(y); mean step "
                           "m <- m - alpha (Lambda m - b), precision Q <- (1 - beta) Q + beta Lambda.",
        input_data=[f"raw problem {RAW_PROBLEM}", f"declared scales {DECLARED_SCALES}, unit scales, alternate "
                    f"{ALTERNATE_SCALES}"],
        observation_model="Trace of F, KL and log Z from ciw.free_energy_math.variational_fit; identity residual "
                          "recomputed here from the retained trace.",
        expected_invariant="|F + log Z - KL| <= 1e-10; KL nonincreasing; raw posterior independent of scales.",
        experiment="normalize_problem with three scale choices; variational_fit with alpha = 2 / (lambda_min + "
                   "lambda_max), beta = 0.3, at most 512 iterations; one unstable step size.",
        numerical_result=f"identity residual {residual:.3g}; {fit['iterations']} iterations to KL {kl[-1]:.3g}; "
                         f"invariance gap {invariance:.3g}; unit-scale KL after 512 iterations "
                         f"{unit['fit']['trace'][-1]['kl_to_reference']:.4g}.",
        uncertainty="Roundoff only (the problem is exact); KL below 1e-15 is at the roundoff level.",
        failure_modes_checked=["unstable step size (counterexample)", "ill-conditioned normalization "
                               "(counterexample)", "identity violation", "iteration bound exceeded"],
        unresolved_assumptions=["Gaussian model with known noise; no claim about non-Gaussian free energies"],
        recommended_next_task="T123: keep these nats separate from the joules of T116/T119")
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T123
def numeric_fields(value, key=None):
    """Names of fields holding numbers (decimal counter strings included), recursively."""
    if isinstance(value, dict):
        for name, child in value.items():
            yield from numeric_fields(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from numeric_fields(child, key)
    elif key is not None and ((isinstance(value, (int, float)) and not isinstance(value, bool))
                              or (isinstance(value, str) and value.isdigit())):
        yield key


def classify_field(name: str) -> str | None:
    """Dimension of a field from its unit token: energy (J), power (W) or information (nat)."""
    tokens = set(name.lower().split("_"))
    if {"j", "mj", "kj", "joule", "joules"} & tokens:
        return "energy"
    if {"w", "mw", "watt", "watts"} & tokens:
        return "power"
    if {"nats", "nat", "kl"} & tokens or name in ("free_energy", "log_evidence", "negative_log_evidence") \
            or name.endswith("_entropy") or name.startswith("free_energy"):
        return "information"
    return None


REFUSAL_ATTEMPTS = {
    "add": (lambda Q: Q(8.5, "nat") + Q(0.2, "J"), "Cannot add information and energy"),
    "compare": (lambda Q: Q(8.5, "nat") < Q(0.2, "J"), "Cannot compare information and energy"),
    "equal": (lambda Q: Q(8.5, "nat") == Q(0.2, "J"), "Cannot compare information and energy"),
    "convert": (lambda Q: Q(0.2, "J").to("nat"), "Cannot express energy in nat"),
    "untyped": (lambda Q: Q(8.5, "nat") + 3.0, "Cannot add a quantity and an untyped number"),
    "untyped_left": (lambda Q: 3.0 + Q(8.5, "nat"), "Cannot add a quantity and an untyped number"),
    "power_plus_energy": (lambda Q: Q(1.0, "W") + Q(0.2, "J"), "Cannot add energy/time and energy"),
}


def typed_refusals() -> dict:
    """Attempt forbidden combinations; return each refusal message (None when accepted)."""
    out = {}
    for name, (attempt, _) in REFUSAL_ATTEMPTS.items():
        try:
            attempt(kernels.Quantity)
            out[name] = None
        except kernels.QuantityRefusal as exc:
            out[name] = str(exc)
    return out


@task("T123", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_typed_quantities_refuse_nats_plus_joules", FIXTURELESS_TEST, UNCERTAINTY_TEST))
def free_energy_distinct_from_energy(ctx):
    from ciw import energy_records, free_energy_math
    Q = kernels.Quantity
    refusals = typed_refusals()
    run = ctx.memo("energy-gpu-free-energy-declared", lambda: _fit(DECLARED_SCALES))
    last = run["fit"]["trace"][-1]
    log_z = run["fit"]["reference"]["log_evidence"]
    identity = (Q(last["free_energy"], "nat") + Q(log_z, "nat") - Q(last["kl_to_reference"], "nat")).to("nat")
    joules = (Q(0.2, "J") + Q(100, "mJ")).to("J") - 0.3
    watts = Q(1000, "mW").to("W") - 1.0
    bits = Q(1, "bit").to("nat") - math.log(2.0)
    equal_across_units = Q(1, "J") == Q(1000, "mJ")
    per_solve = (Q(0.2, "J") / Q(4, "solve")).describe()
    raws = telemetry.fixture_bytes()
    energy_sources = [energy_records.analyze(json.loads(raws["baseline"])), json.loads(raws["baseline"])] if raws else []
    vi_sources = [run["fit"]["trace"], free_energy_math.free_energy(run["normalized"]["problem"], last["mean"],
                                                                    last["covariance"])]
    energy_numbers = {k for source in energy_sources for k in numeric_fields(source)}
    vi_numbers = {k for source in vi_sources for k in numeric_fields(source)}
    joule_fields = sorted(k for k in energy_numbers if classify_field(k) == "energy")
    power_fields = sorted(k for k in energy_numbers if classify_field(k) == "power")
    nat_fields = sorted(k for k in energy_numbers if classify_field(k) == "information")
    vi_fields = sorted(k for k in vi_numbers if classify_field(k) == "information")
    # An energy-named number without a joule-family unit token is ambiguous in an energy record.
    ambiguous = sorted(k for k in energy_numbers if "energy" in k and classify_field(k) != "energy")
    free_in_energy = sorted(k for k in energy_numbers if k.startswith("free_energy"))
    physical_in_vi = sorted(k for k in vi_numbers if classify_field(k) in ("energy", "power"))
    vi_energy_named = sorted(k for k in vi_numbers if "energy" in k and classify_field(k) != "information")
    panels = {}
    if raws:
        view = ctx.memo("energy-gpu-session", lambda: telemetry.session_replay(raws))["baseline_view"]
        panels = {p["panel_id"]: sorted(set(p["units"])) for p in view["panels"]}
    mixed_panels = sum(len(units) != 1 for units in panels.values())
    f_nats, e_joules = last["free_energy"], 0.2
    with_joules, with_millijoules = f_nats + e_joules, f_nats + 1000 * e_joules
    audit_claim = "CIW energy records and free-energy records keep joules, watts and nats in distinct unit-bearing fields"
    audit_value = {"energy_record_joule_fields": joule_fields, "energy_record_power_fields": power_fields,
                   "energy_record_nat_fields": nat_fields, "variational_nat_fields": vi_fields, "panel_units": panels}
    if raws:
        audit = finding(audit_claim, "provenance", audit_value, {"checks": [
            _check("energy-named numbers in energy records without a J/mJ unit token", len(ambiguous), 0.0,
                   kind="exact_arithmetic"),
            _check("free_energy fields in energy records", len(free_in_energy), 0.0, kind="exact_arithmetic"),
            _check("J/mJ or W/mW fields in variational records", len(physical_in_vi), 0.0, kind="exact_arithmetic"),
            _check("energy-named variational numbers outside the free-energy (nat) family", len(vi_energy_named), 0.0,
                   kind="exact_arithmetic"),
            _check("experiment view panels mixing units", mixed_panels, 0.0, kind="exact_arithmetic")]},
            uncertainty=EXACT, tolerance=EXACT_TOL)
    else:
        audit = finding(audit_claim, "provenance", audit_value,
                        {"notes": ["examples/energy-accuracy is not reachable, so no energy record was audited"]},
                        expected_not_established=True)
    findings = [
        finding("Typed arithmetic refuses to add, compare (including ==) or convert nats of variational free energy "
                "with joules, watts with joules, and quantities with untyped numbers",
                "computational_pipeline", refusals,
                {"checks": [_refusal(f"Quantity {name}", expected, refusals[name])
                            for name, (_, expected) in REFUSAL_ATTEMPTS.items()]}, uncertainty=EXACT, tolerance=EXACT_TOL),
        finding("Typed arithmetic within one dimension agrees with hand-converted values to binary64 rounding: "
                "0.2 J + 100 mJ = 0.3 J, 1000 mW = 1 W, 1 bit = ln 2 nat, 1 J == 1000 mJ, F + log Z - KL = 0 nat",
                "numerical", {"joules": joules, "watts": watts, "bits": bits, "identity_nats": identity,
                              "one_joule_equals_1000_millijoules": equal_across_units,
                              "energy_per_solve_dimension": per_solve},
                {"checks": [_check("0.3 J", joules, 1e-15), _check("1 W", watts, 1e-15), _check("ln 2", bits, 1e-15),
                            _check("1 J == 1000 mJ in base units (0 = equal)", 0.0 if equal_across_units else 1.0,
                                   0.0, kind="exact_arithmetic"),
                            _check("F + log Z = KL", identity, 1e-10, kind="invariant")]},
                uncertainty={"kind": "roundoff", "value": 1e-15, "basis": "binary64 unit scales such as 1e-3 are "
                                                                          "inexact"},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        audit,
        finding("An untyped sum of free energy and physical energy changes when the energy unit changes", "numerical",
                {"free_energy_nats": f_nats, "energy_j": e_joules, "sum_with_joules": with_joules,
                 "sum_with_millijoules": with_millijoules},
                {"checks": [_check("difference between the two computed untyped sums", with_millijoules - with_joules,
                                   1.0, "ge", kind="exact_arithmetic")]},
                uncertainty={"kind": "roundoff", "value": 1e-13, "basis": "binary64 sums"},
                tolerance={"abs": 1e-9, "rel": 0.0},
                counterexample={"statement": "Adding variational free energy to physical energy yields a "
                                             "unit-independent quantity",
                                "witness": {"free_energy_nats": f_nats, "energy": "0.2 J = 200 mJ"}}),
        _no_measurement("A decrease of variational free energy corresponds to a decrease of physical energy consumed "
                        "by the computation", "relating nats to joules needs a declared physical system and "
                        "temperature (for example k_B T per nat) and a measured device"),
    ]
    ctx.artifact_json("field-audit.json", {**audit_value, "ambiguous_energy_numbers": ambiguous,
                                           "free_energy_in_energy_records": free_in_energy,
                                           "physical_units_in_variational_records": physical_in_vi,
                                           "refusals": refusals, "fixtures_reachable": raws is not None})
    fields = _fields(
        hypothesis="Variational free energy (nats, information) and device energy (joules) are different dimensions; "
                   "a typed check refuses to combine them, and CIW records already keep them in disjoint fields.",
        mathematical_model="Quantities carry a dimension vector over {information, energy, time, count}; power is "
                           "energy/time; addition, subtraction and comparison (including equality) require equal "
                           "vectors; multiplication and division combine them.",
        input_data=["ciw.free_energy_math trace of the T122 declared problem", FIXTURE_NOTE],
        observation_model="Refusal messages of the typed algebra; field names of energy_records logs/analyses, "
                          "variational traces and the workbench energy view.",
        expected_invariant="nat + J, nat < J, nat == J, J -> nat, W + J and untyped mixing refused; same-dimension "
                           "arithmetic agrees to binary64 rounding; no field or panel mixes dimensions.",
        experiment="Exercise the typed algebra; classify every field name by unit token (J, W or nat family); "
                   "inspect experiment panels.",
        numerical_result=f"refusals {refusals}; joule fields {joule_fields}; power fields {power_fields}; nat fields "
                         f"in energy records {nat_fields}; variational nat fields {vi_fields}; panel units {panels}."
                         + ("" if raws else " Energy-record audit not run: fixtures unreachable."),
        uncertainty="Field classification is by naming convention (unit tokens); a field without a unit token is "
                    "not classified.",
        failure_modes_checked=["implicit nat/J addition", "cross-dimension comparison and equality",
                               "silent conversion", "untyped number on either side of a quantity",
                               "power added to energy", "unit-dependent untyped sum"],
        unresolved_assumptions=["The Landauer relation (k_B T ln 2 per bit) is a physical bound on erasure, not an "
                                "estimate of what this computation dissipated"]
        + ([] if raws else ["examples/energy-accuracy is not reachable; the energy-record field audit did not run"]),
        recommended_next_task="T146: carry units in one canonical serialization across languages")
    return {"state": _state(findings, "completed" if raws else "partial"), "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T124
STATIC_T124 = {
    "hypothesis": "Every retained energy log keeps raw timestamped counter readings with device and runtime identity "
                  "fields, and the validator refuses edits to them unless the log is deliberately resealed.",
    "mathematical_model": "log_digest = sha256(canonical(log without digest)); structural profile of "
                          "energy_records.validate_log; analysis eligibility rules of energy_records.analyze.",
    "input_data": [FIXTURE_NOTE],
    "observation_model": "Field inventory of each sealed fixture (identity fields present, and how many hold "
                         "placeholder values); validation outcome of each mutated copy.",
    "expected_invariant": "All readings carry monotonic and UTC brackets; identity fields present; each tampering "
                          "class refused with its specific message.",
    "experiment": f"Validate the four fixtures; apply {len(telemetry.TAMPERING)} mutations (with or without "
                  "resealing); validate a resealed doubling, an origin relabelling and a UUID mismatch.",
    "failure_modes_checked": [name for name, *_ in telemetry.TAMPERING] + [
        "resealed modification", "origin relabelling", "sensor/workload UUID mismatch",
        "placeholder identity values counted as identification"],
    "unresolved_assumptions": ["Sealing is integrity, not authenticity: no signature binds a log to the device",
                               "Hardware provenance of every fixture is not established",
                               "The fixtures' identity fields are present but mostly placeholders"],
    "recommended_next_task": "T125: replay the retained logs through a Session; later, signed capture on the RTX "
                             "2080 host (T116)",
}


def _t124_physical():
    return _no_measurement("The fixtures' counter readings were produced by a physical GPU and NVML counter",
                           "the fixtures declare origin synthetic_fixture and carry placeholder library and "
                           "executable digests")


def _validation_outcome(log) -> str | None:
    from ciw import energy_records
    try:
        energy_records.validate_log(log)
    except ValueError as exc:
        return str(exc)
    return None


@task("T124", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_raw_telemetry_retention_and_tampering", FIXTURELESS_TEST, UNCERTAINTY_TEST))
def retain_raw_telemetry(ctx):
    from ciw import energy_records
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked(STATIC_T124, _t124_physical)
    logs = {name: json.loads(raw) for name, raw in raws.items()}
    for log in logs.values():
        energy_records.validate_log(log)
    inventory = {name: telemetry.raw_inventory(log) for name, log in logs.items()}
    identity_fields = len(next(iter(inventory.values()))["identity_fields"])
    base = logs["baseline"]
    outcomes = {name: telemetry.tamper_outcome(base, mutation, resealed)
                for name, mutation, resealed, _ in telemetry.TAMPERING}

    def doubled(log):
        for phase in log["phases"]:
            for sample in phase["samples"]:
                sample["energy_mj"] = str(2 * int(sample["energy_mj"]))

    resealed = telemetry.reseal(_mutated(base, doubled))
    resealed_outcome = _validation_outcome(resealed)
    relabelled = telemetry.reseal(_mutated(base, lambda log: log.update(origin="physical_measurement")))
    mismatch = telemetry.reseal(_mutated(base, lambda log: log["sensor"].update(
        device_uuid="GPU-11111111-2222-3333-4444-555555555555")))
    relabelled_analysis = energy_records.analyze(relabelled)
    mismatch_reasons = energy_records.analyze(mismatch)["comparison"]["reasons"]
    generator = {"name": "examples/energy-accuracy synthetic fixtures", "origin": "synthetic_fixture", "seed": None}
    missing_time = sum(row["samples"] - row["timestamped_samples"] for row in inventory.values())
    missing_identity = sum(identity_fields - row["identity_fields_present"] for row in inventory.values())
    findings = [
        finding(f"Every fixture retains raw timestamped counter readings and the {identity_fields} device/runtime "
                "identity fields (placeholder values in these synthetic fixtures)", "provenance",
                {name: {"samples": row["samples"], "raw_counter_readings": row["raw_counter_readings"],
                        "batches": row["batches"], "identity_fields_present": row["identity_fields_present"],
                        "placeholder_values": row["placeholder_values"]} for name, row in inventory.items()},
                {"generator": generator,
                 "checks": [_check("readings without monotonic and UTC brackets", missing_time, 0.0, kind="exact_arithmetic"),
                            _check("missing identity fields", missing_identity, 0.0, kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding(f"The energy-log validator refuses {len(telemetry.TAMPERING)} classes of tampering", "provenance",
                sorted(outcomes),
                {"generator": generator,
                 "checks": [_refusal(name, expected, outcomes[name]) for name, _, _, expected in telemetry.TAMPERING]},
                uncertainty=EXACT, tolerance=EXACT_TOL),
        finding("A log whose counter readings were doubled and then resealed passes validation", "provenance",
                {"gross_energy_j_original": energy_records.analyze(base)["measurement"]["gross_energy_j"],
                 "gross_energy_j_resealed": energy_records.analyze(resealed)["measurement"]["gross_energy_j"]},
                {"generator": generator,
                 "checks": [_check("energy_records.validate_log refusals of the resealed log (0 = accepted)",
                                   0.0 if resealed_outcome is None else 1.0, 0.0, kind="exact_arithmetic")]},
                uncertainty=SYNTHETIC, tolerance={"abs": 1e-12, "rel": 0.0},
                counterexample={"statement": "A valid log digest shows that the retained readings are unmodified "
                                             "hardware output",
                                "witness": {"mutation": "every energy_mj doubled, then energy_records.seal",
                                            "log_digest": resealed["log_digest"]}}),
        finding("Relabelling a synthetic fixture as physical_measurement makes its analysis eligible for physical "
                "comparison", "provenance",
                {"classification": relabelled_analysis["comparison"]["classification"],
                 "hardware_provenance": relabelled_analysis["hardware_provenance"]},
                {"generator": generator,
                 "checks": [_check("eligible after relabelling (1 = yes)",
                                   1.0 if relabelled_analysis["comparison"]["eligible"] else 0.0, 1.0, "ge",
                                   kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOL,
                counterexample={"statement": "The declared origin of a retained energy log authenticates a physical "
                                             "measurement",
                                "witness": {"origin": "physical_measurement", "run_id": base["run_id"]}}),
        finding("A resealed sensor/workload UUID mismatch validates but is withheld by the analysis", "provenance",
                mismatch_reasons,
                {"generator": generator,
                 "checks": [_check("workload_sensor_device_mismatch reported (1 = yes)",
                                   1.0 if "workload_sensor_device_mismatch" in mismatch_reasons else 0.0, 1.0, "ge",
                                   kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOL),
        _t124_physical(),
    ]
    ctx.artifact_json("inventory.json", inventory)
    ctx.artifact_json("tampering.json", {"outcomes": outcomes, "resealed_doubled_digest": resealed["log_digest"],
                                         "resealed_doubled_validation": resealed_outcome or "accepted",
                                         "relabelled_analysis_comparison": relabelled_analysis["comparison"],
                                         "uuid_mismatch_reasons": mismatch_reasons})
    ctx.artifact_json("baseline-measurement-samples.json", base["phases"][3]["samples"])
    placeholders = sum(row["placeholder_values"] for row in inventory.values())
    fields = _fields(**dict(
        STATIC_T124,
        numerical_result=f"{sum(r['samples'] for r in inventory.values())} raw readings retained; {identity_fields} "
                         f"identity fields per fixture, {placeholders} placeholder values over the four fixtures; "
                         f"refusals {outcomes}; resealed doubling "
                         f"{'accepted' if resealed_outcome is None else 'refused'}; relabelled fixture classified "
                         f"{relabelled_analysis['comparison']['classification']}.",
        uncertainty="Exact (structural checks)."))
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}


def _mutated(log, mutation):
    from copy import deepcopy
    changed = deepcopy(log)
    mutation(changed)
    return changed


# ----------------------------------------------------------------- T125
STATIC_T125 = {
    "hypothesis": "Replaying a retained energy analysis through a ciw Session recomputes the same numerical result "
                  "(stable numerical_result_id) under fresh execution and result identities.",
    "mathematical_model": "numerical_result_id = digest({operation_id, data = energy_records.analyze(log)}); "
                          "execution and result identities include a fresh occurrence UUID.",
    "input_data": [FIXTURE_NOTE],
    "observation_model": "Session protocol: source.add, operation.execute, bundle.replay, bundle.get, "
                         "experiment.inspect, workspace.save and Session.from_workspace.",
    "expected_invariant": "One numerical_result_id per fixture; four execution ids, two result ids and two bundle "
                          "digests per fixture; restored workspace equal; edited bundles refused.",
    "experiment": "Run every fixture through one Session, replay it, compare identities, save and restore the "
                  "workspace, and replay an edited copy of the baseline bundle with stale and with recomputed digests.",
    "failure_modes_checked": ["numerical id drift between original and replay", "reused execution identity",
                              "receipt claims independence or admission", "edited retained result",
                              "workspace restore mismatch"],
    "unresolved_assumptions": ["Replay determinism is not independent verification by another party",
                               "The retained readings are synthetic",
                               "Resealing an edited bundle uses ciw.telemetry._bundle_digest (internal; pinned by "
                               "the regression test)"],
    "recommended_next_task": "T146: canonical serialization so numerical ids agree across platforms and languages",
}


def _t125_physical():
    return _no_measurement("Replayed energy values are physically valid measurements",
                           "replay recomputes the same retained synthetic readings; it acquires nothing")


@task("T125", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_session_replay_keeps_numerical_result_id", FIXTURELESS_TEST, UNCERTAINTY_TEST))
def replay_energy_reports(ctx):
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked(STATIC_T125, _t125_physical)
    replay = ctx.memo("energy-gpu-session", lambda: telemetry.session_replay(raws))
    rows = replay["fixtures"]
    distinct_numerical = {name: len(set(row["numerical_result_ids"])) for name, row in rows.items()}
    direct_mismatch = sum(row["direct_numerical_result_id"] != row["numerical_result_ids"][0] for row in rows.values())
    fresh = {name: {"execution_ids": len(set(row["execution_ids"])), "result_ids": len(set(row["result_ids"])),
                    "bundle_digests": len(set(row["bundle_digests"]))} for name, row in rows.items()}
    fresh_deficit = sum((4 - v["execution_ids"]) + (2 - v["result_ids"]) + (2 - v["bundle_digests"])
                        for v in fresh.values())
    receipts_bad = sum(not (row["receipt_numerical_match"] is True and row["receipt_admission"] == "not_performed"
                            and row["verification_independent"] == [False, False]
                            and row["fresh_hardware_measurement"] is False) for row in rows.values())
    generator = {"name": "examples/energy-accuracy synthetic fixtures", "origin": "synthetic_fixture", "seed": None}
    findings = [
        finding("Session replay keeps numerical_result_id identical across original, reproduction and replay for "
                "every fixture", "computational_pipeline", distinct_numerical,
                {"generator": generator,
                 "checks": [_check("distinct numerical_result_id per fixture minus one",
                                   sum(distinct_numerical.values()) - len(distinct_numerical), 0.0, kind="exact_arithmetic"),
                            _check("ids differing from digest(operation, energy_records.analyze(log))", direct_mismatch,
                                   0.0, kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Each replay is a fresh analysis occurrence with new execution, result and bundle identities",
                "computational_pipeline", fresh,
                {"generator": generator,
                 "checks": [_check("missing fresh identities", fresh_deficit, 0.0, kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Replay receipts record a numerical match, no state admission, a non-independent method and no fresh "
                "hardware measurement", "provenance", sorted({row["verification_method"] for row in rows.values()}),
                {"generator": generator,
                 "checks": [_check("fixtures with a receipt or view outside policy", receipts_bad, 0.0,
                                   kind="exact_arithmetic")]}, uncertainty=EXACT, tolerance=EXACT_TOL),
        finding("Saving and restoring the Session workspace reproduces the retained workbench exactly", "provenance",
                {"bundles": replay["bundle_count"], "restored_equal": replay["restored_equal"]},
                {"generator": generator,
                 "checks": [_check("restored workbench differs (1 = yes)", 0.0 if replay["restored_equal"] else 1.0, 0.0,
                                   kind="exact_arithmetic"),
                            _check("bundles retained minus 2 per fixture", replay["bundle_count"] - 2 * len(rows), 0.0,
                                   kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Replaying a retained bundle whose numerical result was edited is refused, with or without "
                "resealed envelopes", "provenance", replay["refusals"],
                {"generator": generator,
                 "checks": [_refusal("replay_session on an edited bundle (stale digests)",
                                     "Energy analysis bundle identity, schema or size differs",
                                     replay["refusals"].get("edited_bundle")),
                            _refusal("replay_session on an edited bundle with recomputed digests",
                                     "Retained energy analysis binding differs",
                                     replay["refusals"].get("edited_and_resealed_bundle"))]}, uncertainty=EXACT, tolerance=EXACT_TOL),
        _t125_physical(),
    ]
    ctx.artifact_json("session-replay.json", {"fixtures": rows, "refusals": replay["refusals"],
                                              "restored_equal": replay["restored_equal"],
                                              "bundle_count": replay["bundle_count"]})
    fields = _fields(**dict(
        STATIC_T125,
        numerical_result=f"distinct numerical ids {distinct_numerical}; fresh identities {fresh}; restored equal "
                         f"{replay['restored_equal']}; edited replays refused: {replay['refusals']}.",
        uncertainty="Exact; the numerical ids themselves may differ across platforms if float results differ in the "
                    "last bit, so only their stability within a run is asserted."))
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}
