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
``not_established`` unless a host supplies acquired evidence. The fixtures
are synthetic, so every joule figure derived from them is a synthetic value.
Emulated reduction orders are not the orders of any GPU library, and
Python/Rust agreement is agreement of two ciw implementations, not
independent verification. Wall-clock and CPU times are retained only as
artifacts; they are not reproducible and never enter a finding.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time

import numpy as np

from . import energy_gpu_kernels as kernels
from . import energy_gpu_telemetry as telemetry
from . import svg
from .evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, finding, supported_label
from .integrators import integrate_adaptive, integrate_fixed
from .registry import task
from .surfaces import Sphere

MODULE = "src/ciw/lab/energy_gpu.py"
KERNELS = "src/ciw/lab/energy_gpu_kernels.py"
TELEMETRY = "src/ciw/lab/energy_gpu_telemetry.py"
DOC = "docs/lab/ENERGY_GPU.md"
TESTS = "tests/test_lab_energy_gpu.py"
FILES = (MODULE, KERNELS, TELEMETRY, DOC)

STEPS = 256
FIXTURE_NOTE = ("examples/energy-accuracy/{baseline,reset,missing,under-target}.json: synthetic fixtures "
                "(origin synthetic_fixture) constructed on the CPU, not hardware captures")


def _check(reference, observed, tolerance, comparison="abs_le", kind="analytic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed if observed is not None else "accepted", "passed": observed == expected}


def _state(findings, planned):
    """Downgrade to partial when a computational check failed instead of hiding it."""
    unexpected = [f for f in findings if f["evidence_status"] == "not_established"
                  and f["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and not f.get("expected_not_established")]
    return "partial" if unexpected and planned == "completed" else planned


def _fields(**values):
    required = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                "experiment", "numerical_result", "uncertainty", "failure_modes_checked",
                "unresolved_assumptions", "recommended_next_task")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Report fields missing: {missing}")
    return values


def _no_measurement(claim, reason, unit=None):
    """A physical claim this environment cannot support: recorded, never omitted."""
    return finding(claim, "physical", None, {"notes": [reason]}, unit=unit)


def _fixtures_blocked(task_id, hypothesis):
    fields = _fields(
        hypothesis=hypothesis, mathematical_model="Not evaluated.", input_data=[FIXTURE_NOTE],
        observation_model="No observation was made.", expected_invariant="Not evaluated.",
        experiment="Blocked: the synthetic energy fixtures are not present (run from a source checkout).",
        numerical_result="none", uncertainty="not quantified", failure_modes_checked=[],
        unresolved_assumptions=["examples/energy-accuracy is outside an installed package"],
        recommended_next_task=f"Rerun {task_id} from a source checkout containing examples/energy-accuracy")
    return {"state": "blocked", "fields": fields, "findings": []}


# ----------------------------------------------------------------- T115
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


def _rapl_measurement(ctx):
    """Background-inclusive package energy over repeated batches, or the reason it is unavailable."""
    if not ctx.available("hardware:rapl"):
        return None, "no readable /sys/class/powercap/intel-rapl:* energy counters (RAPL) on this host"
    domains = telemetry.rapl_domains()
    try:
        repeats = 3
        stamp = time.time_ns()
        before = telemetry.rapl_read(domains)
        for _ in range(repeats):
            _cpu_workload()
        after = telemetry.rapl_read(domains)
    except (OSError, ValueError) as exc:
        return None, f"RAPL counters present but unreadable: {type(exc).__name__}: {exc}"
    deltas = [telemetry.rapl_delta_uj(b, a, d["max_energy_range_uj"]) for b, a, d in zip(before, after, domains)]
    raw = {"domains": domains, "before_uj": before, "after_uj": after, "repeats": repeats,
           "trajectories_per_repeat": len(kernels.HEADINGS), "acquired_unix_ns": stamp}
    total_uj = sum(delta for delta, _ in deltas)
    return {"raw": raw, "wrapped": [w for _, w in deltas],
            "joules_per_trajectory": total_uj * 1e-6 / (repeats * len(kernels.HEADINGS))}, None


@task("T115", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_cpu_energy_task_counts_work_and_leaves_energy_unestablished",
    f"{TESTS}::test_rapl_helpers_read_counters_and_one_wrap"))
def cpu_energy_per_trajectory(ctx):
    work = ctx.memo("energy-gpu-cpu-workload", _cpu_workload)
    trajectories = len(work["states"])
    counted = sum(work["fixed_counts"])
    evaluations = [s["function_evaluations"] for s in work["adaptive_stats"]]
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "seed": None}
    findings = [
        finding(f"Fixed-step RK4 spends exactly 4N = {4 * STEPS} right-hand-side evaluations per trajectory "
                f"(N = {STEPS})", "numerical", 4 * STEPS,
                {"generator": dict(generator, steps=STEPS),
                 "checks": [_check("counted evaluations minus 4 N T", counted - 4 * STEPS * trajectories, 0.0,
                                   kind="exact_arithmetic")]},
                unit="evaluations/trajectory", tolerance={"abs": 0.0, "rel": 0.0}),
        finding("The fixed-step trajectories are accepted results: embedded endpoint error against the exact "
                "great circle is below 1e-8", "numerical", float(np.max(work["fixed_errors"])),
                {"generator": dict(generator, steps=STEPS),
                 "checks": [_check("great circle cos(L) X0 + sin(L) T0", float(np.max(work["fixed_errors"])), 1e-8)]},
                unit="normalized length", tolerance={"abs": 1e-12, "rel": 1e-3}),
        finding("Adaptive Dormand-Prince 5(4) (rtol 1e-9) right-hand-side evaluations per trajectory",
                "numerical", evaluations,
                {"generator": dict(generator, rtol=1e-9, atol=1e-12),
                 "checks": [_check("great circle endpoint", float(np.max(work["adaptive_errors"])), 1e-7)]},
                unit="evaluations/trajectory", tolerance={"abs": 14.0, "rel": 0.05}),
    ]
    measured, reason = _rapl_measurement(ctx)
    if measured is None:
        findings.append(_no_measurement("CPU package energy per geodesic trajectory", reason, unit="J/trajectory"))
    else:
        raw_path = ctx.artifact_json("rapl-raw.json", measured["raw"])
        acquisition = {"device": "RAPL package domains " + ",".join(d["zone"] for d in measured["raw"]["domains"]),
                       "raw_sha256": ctx.artifacts[-1]["sha256"],
                       "acquired_at": telemetry._utc(str(measured["raw"]["acquired_unix_ns"])),
                       "calibration": "not_applied: RAPL model-based counter, background-inclusive"}
        findings.append(finding("CPU package energy per geodesic trajectory", "physical",
                                measured["joules_per_trajectory"], {"acquisition": acquisition, "inputs": [raw_path]},
                                unit="J/trajectory", tolerance={"abs": 0.0, "rel": 1.0}))
    ctx.artifact_json("work-proxies.json", {
        "steps": STEPS, "trajectories": trajectories, "fixed_step_evaluations": work["fixed_counts"],
        "fixed_step_endpoint_errors": work["fixed_errors"].tolist(), "adaptive_stats": work["adaptive_stats"],
        "adaptive_endpoint_errors": work["adaptive_errors"].tolist()})
    ctx.artifact_json("timing.json", {
        "note": "Elapsed and process CPU time of this run only; not reproducible and not an energy measurement.",
        "clock": "time.perf_counter / time.process_time", "wall_s_per_trajectory": work["wall_s"],
        "cpu_s_per_trajectory": work["cpu_s"]})
    ctx.artifact_json("rapl-probe.json", {"available": measured is not None, "reason": reason,
                                          "domains": telemetry.rapl_domains() if measured else []})
    fields = _fields(
        hypothesis="CPU energy per geodesic trajectory is proportional to the deterministic work of the integrator "
                   "(right-hand-side evaluations) and can be read from package energy counters bracketing a batch.",
        mathematical_model="E_traj = (E_pkg(after) - E_pkg(before)) / (repeats * trajectories), gross and "
                           "background-inclusive; work proxy W = 4N evaluations (RK4) or the Dormand-Prince count.",
        input_data=[f"{trajectories} unit-speed geodesics on the unit sphere from the equator, headings "
                    f"{list(kernels.HEADINGS)} rad, length {kernels.LENGTH}"],
        observation_model="Evaluations counted by wrapping Surface.geodesic_rhs; RAPL energy_uj read from "
                          "/sys/class/powercap when present; wall/CPU time retained only as an artifact.",
        expected_invariant="Counted RK4 evaluations equal 4 N per trajectory; endpoints match the great circle.",
        experiment=f"Integrate each geodesic with RK4 (N = {STEPS}) and adaptive DP5(4); count evaluations; "
                   "probe RAPL; bracket three repeated batches with counter reads when RAPL is readable.",
        numerical_result=f"{4 * STEPS} evaluations per RK4 trajectory; max endpoint error "
                         f"{float(np.max(work['fixed_errors'])):.3g}; adaptive evaluations {evaluations}; energy: "
                         + ("measured, see findings" if measured else "not measured (" + reason + ")"),
        uncertainty="Work counts are exact; energy would carry RAPL model error, background load and a "
                    "counter resolution of about 15 uJ that are not characterized here.",
        failure_modes_checked=["evaluation count differs from 4N", "endpoint error exceeds 1e-8",
                               "RAPL absent or unreadable (permission)", "RAPL counter wrap (one wrap allowed)"],
        unresolved_assumptions=["Evaluation counts are a work proxy, not an energy measurement",
                                "Package energy is not attributed to this process and idle power is not subtracted"],
        recommended_next_task="Run T115 on a Linux host with readable intel-rapl counters, then T120 to relate "
                              "precision to measured energy")
    return {"state": "completed" if measured else "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T116 / T118 (GPU host only)
RECORD = ("ciw energy record --problem examples/energy-accuracy/problem.json --output-dir runs/rtx2080-<date> "
          "--duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0")
SMI = ("nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,"
       "clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f runs/rtx2080-<date>/smi.csv")

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
    "experiment": "On the RTX 2080 host: (1) `ciw energy probe --gpu-index 0` must return status ok; (2) " + RECORD
                  + "; (3) `ciw energy replay runs/rtx2080-<date>/log.json`; (4) `CIW_LAB_ENERGY_LOG=runs/rtx2080-"
                  "<date>/log.json ciw lab run T116 T118 --output-dir <dir>`.",
    "numerical_result": "none: no NVIDIA GPU or NVML in this environment",
    "uncertainty": "NVML counter resolution, update interval and accuracy are undeclared; energy is device-wide and "
                   "includes background work; host brackets add call overhead.",
    "failure_modes_checked": ["counter reset or wrap", "missing endpoint brackets", "sensor/workload UUID mismatch",
                              "accuracy target not met", "minimum duration not met"],
    "unresolved_assumptions": ["Blocked here: no NVIDIA GPU or NVML", "The RTX 2080 driver exposes the total-energy "
                               "counter (NVML documents it for Volta and newer); confirm with `ciw energy probe`"],
    "recommended_next_task": "T118: record utilization, temperature, power and kernel duration in the same session",
}
PLAN_T118 = {
    "hypothesis": "During the T116 workload the RTX 2080 power, temperature and clocks stay in a steady state and "
                  "kernel time is a stable fraction of the host-bracketed batch time.",
    "mathematical_model": "Per-phase min/mean/max of NVML power (mW), temperature (C) and graphics clock (MHz); "
                          "utilization from an nvidia-smi sidecar; kernel duration from an Nsight Systems pass.",
    "input_data": ["log.json from `ciw energy record`", "smi.csv from the nvidia-smi sidecar",
                   "nsys report from a separate profiling pass"],
    "observation_model": "NVML context readings sampled after each energy read (not simultaneous with it); "
                         "nvidia-smi sampling every 100 ms; CUDA kernel spans from nsys.",
    "expected_invariant": "Device name contains RTX 2080; the same UUID appears in log.json, smi.csv and "
                          "`nvidia-smi -L`; power and temperature readings are present for every sample.",
    "experiment": "On the RTX 2080 host: start `" + SMI + "` in the background; run `" + RECORD + "`; stop the "
                  "sidecar; separately run `nsys profile --trace=cuda -o runs/rtx2080-<date>/nsys python -m ciw energy "
                  "record ... --output-dir runs/rtx2080-<date>-nsys` and `nsys stats --report cuda_gpu_kern_sum`; then "
                  "`CIW_LAB_ENERGY_LOG=... CIW_LAB_NVIDIA_SMI_CSV=... ciw lab run T118`.",
    "numerical_result": "none: no NVIDIA GPU in this environment",
    "uncertainty": "NVML power is a vendor estimate with undeclared averaging; 100 ms sidecar sampling aliases short "
                   "batches; profiling perturbs timing and energy, so it is a separate pass.",
    "failure_modes_checked": ["context reading unavailable (retained error code)", "device is not an RTX 2080",
                              "UUID absent from the host", "sidecar missing"],
    "unresolved_assumptions": ["Blocked here: no NVIDIA GPU", "Batch windows in log.json include launch, "
                               "synchronization and copy; they are not kernel durations"],
    "recommended_next_task": "T147: compare CPU and GPU outputs of the same workload on the RTX 2080 host",
}


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return {"min": min(values), "mean": math.fsum(values) / len(values), "max": max(values), "count": len(values)}


def operator_log_outcome(ctx, task_id, raw, log, analysis, listing, smi_text=None):
    """Findings from an operator-captured NVML log; physical labels only through the acquisition gate."""
    from ciw import energy_records
    from ciw.telemetry import canonical
    recomputed = canonical(energy_records.analyze(json.loads(raw))) == canonical(analysis)
    measurement = analysis["measurement"]
    pipeline = finding("The operator log validates and its analysis recomputes identically from the retained bytes",
                       "computational_pipeline", recomputed,
                       {"inputs": [hashlib.sha256(raw).hexdigest()],
                        "checks": [_check("second energy_records.analyze of the same bytes", 0.0 if recomputed else 1.0,
                                          0.0, kind="exact_arithmetic")]})
    findings = [pipeline]
    if task_id == "T116":
        basis, reasons = telemetry.physical_basis(raw, log, analysis, listing)
        batches = measurement["batch_count"]
        value = (measurement["gross_energy_j"] / batches) if measurement["gross_energy_j"] is not None and batches else None
        if value is None:
            basis, reasons = {"notes": reasons + ["gross energy withheld by the analysis"]}, reasons
        findings.append(finding("GPU-domain gross energy per measured batch", "physical", value, basis,
                                unit="J/batch", tolerance={"abs": 0.0, "rel": 1.0}))
        if measurement["max_kl_nats"] is not None:
            findings.append(finding("Every measured batch output meets the declared KL target", "numerical",
                                    measurement["max_kl_nats"],
                                    {"inputs": [log["log_digest"]],
                                     "checks": [_check("declared target_kl_nats", measurement["max_kl_nats"],
                                                       measurement["target_kl_nats"], comparison="le")]},
                                    unit="nat", tolerance={"abs": 1e-12, "rel": 1e-6}))
    else:
        basis, reasons = telemetry.physical_basis(raw, log, analysis, listing, required_name="RTX 2080")
        samples = log["phases"][3]["samples"]
        power = _stats(s["power_mw"] / 1000 if s["power_mw"] is not None else None for s in samples)
        temperature = _stats(s["temperature_c"] for s in samples)
        clock = _stats(s["graphics_clock_mhz"] for s in samples)
        durations = _stats((b["end_ns"] - b["start_ns"]) / 1e6 for b in log["phases"][3]["batches"])
        for claim, value, unit in (("RTX 2080 power draw during the measurement phase (NVML)", power, "W"),
                                   ("RTX 2080 temperature during the measurement phase (NVML)", temperature, "C"),
                                   ("RTX 2080 graphics clock during the measurement phase (NVML)", clock, "MHz"),
                                   ("Host-bracketed batch solve duration (launch, sync and copy included)", durations, "ms")):
            findings.append(finding(claim, "physical", value, basis if value is not None else {"notes": reasons},
                                    unit=unit, tolerance={"abs": 0.0, "rel": 1.0}))
        utilization = None
        if smi_text:
            columns = telemetry.smi_columns(smi_text)
            key = next((k for k in columns if k.startswith("utilization.gpu")), None)
            utilization = _stats(columns[key]) if key else None
        smi_basis = {"notes": reasons + ([] if smi_text else ["no nvidia-smi sidecar was supplied"])}
        if utilization is not None and not reasons:
            smi_basis = {"acquisition": dict(basis["acquisition"],
                                             raw_sha256=hashlib.sha256(smi_text.encode()).hexdigest(),
                                             device=basis["acquisition"]["device"] + " via nvidia-smi sidecar")}
        findings.append(finding("RTX 2080 GPU utilization during the measurement phase (nvidia-smi)", "physical",
                                utilization, smi_basis, unit="%", tolerance={"abs": 0.0, "rel": 1.0}))
        findings.append(_no_measurement("RTX 2080 kernel-only duration of the Gaussian VI kernel",
                                        "log.json brackets launch, synchronization and copy; kernel spans need an "
                                        "Nsight Systems pass", unit="ms"))
    ctx.artifact_json("operator-log-analysis.json", {"analysis": analysis, "withheld_reasons": reasons,
                                                     "gpu_listing": listing})
    plan = PLAN_T116 if task_id == "T116" else PLAN_T118
    measured = all(f["evidence_status"] == "hardware_measured" for f in findings if f["domain"] == "physical")
    fields = _fields(**{k: v for k, v in plan.items()})
    fields["experiment"] = "Analyzed the operator-captured log named by CIW_LAB_ENERGY_LOG. Protocol: " + plan["experiment"]
    fields["numerical_result"] = f"gross measurement energy {measurement['gross_energy_j']} J over " \
                                 f"{measurement['batch_count']} batches; physical findings withheld: {reasons or 'none'}"
    fields["unresolved_assumptions"] = ["A sealed log proves internal integrity, not that the capture was genuine",
                                        *reasons]
    return {"state": "completed" if measured else "partial", "fields": fields, "findings": findings}


def _operator_log_task(ctx, task_id):
    plan = PLAN_T116 if task_id == "T116" else PLAN_T118
    path = telemetry.environment_log_path()
    if path is None:
        fields = _fields(**plan)
        fields["experiment"] = ("Blocked: a GPU is present but the lab runner never acquires hardware data; set "
                                f"{telemetry.LOG_ENV} to a log.json captured by `ciw energy record`. Planned: "
                                + plan["experiment"])
        return {"state": "blocked", "fields": fields, "findings": []}
    raw, log, analysis = telemetry.read_operator_log(path)
    smi_path = telemetry.environment_log_path(telemetry.SMI_ENV)
    smi_text = open(smi_path, encoding="utf-8").read() if smi_path else None
    return operator_log_outcome(ctx, task_id, raw, log, analysis, telemetry.gpu_listing(), smi_text)


@task("T116", changed_files=FILES, requires=("hardware:nvidia-gpu",), plan=PLAN_T116, regression_tests=(
    f"{TESTS}::test_gpu_tasks_are_blocked_with_the_recording_protocol",
    f"{TESTS}::test_operator_log_gate_withholds_physical_labels_from_synthetic_logs"))
def gpu_energy_per_batch(ctx):
    return _operator_log_task(ctx, "T116")


@task("T118", changed_files=FILES, requires=("hardware:nvidia-gpu",), plan=PLAN_T118, regression_tests=(
    f"{TESTS}::test_gpu_tasks_are_blocked_with_the_recording_protocol",
    f"{TESTS}::test_operator_log_gate_withholds_physical_labels_from_synthetic_logs"))
def rtx2080_telemetry(ctx):
    return _operator_log_task(ctx, "T118")


# ----------------------------------------------------------------- T117
def _rust_run(states):
    with tempfile.TemporaryDirectory() as scratch:
        try:
            wall0 = time.perf_counter()
            identity = kernels.build_rust_kernel(scratch)
            build_s = time.perf_counter() - wall0
            wall0 = time.perf_counter()
            result = kernels.run_rust_kernel(identity.pop("executable"), states, kernels.LENGTH, STEPS)
            run_s = time.perf_counter() - wall0
        except kernels.NativeKernelUnavailable as exc:
            return {"unavailable": str(exc)}
    return {"identity": identity, "states": np.array(result["states"]), "evaluations": result["evaluations"],
            "build_wall_s": build_s, "run_wall_s_including_process_start": run_s}


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


@task("T117", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_rust_kernel_matches_python_kernel",
    f"{TESTS}::test_cross_language_agreement_is_not_independent"))
def compare_implementations(ctx):
    states = kernels.initial_states()
    python = ctx.memo("energy-gpu-python-kernels", lambda: _python_runs(states))
    rust = ctx.memo("energy-gpu-rust-kernel", lambda: _rust_run(states))
    closed, generic = python["python-closed-form"], python["python-generic-christoffel"]
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "steps": STEPS, "seed": None}
    generic_gap = float(np.max(np.abs(generic - closed)))
    findings = []
    if "unavailable" in rust:
        findings.append(finding("Rust and Python closed-form RK4 kernels agree on every endpoint to 1e-12",
                                "numerical", None, {"notes": [rust["unavailable"]]}, expected_not_established=True))
        findings.append(finding("Rust kernel endpoint error against the exact great circle is below 1e-8",
                                "numerical", None, {"notes": [rust["unavailable"]]}, expected_not_established=True))
        rust_row = {"available": False, "reason": rust["unavailable"]}
    else:
        gap = float(np.max(np.abs(rust["states"] - closed)))
        rust_errors = kernels.endpoint_errors(states, rust["states"], kernels.LENGTH)
        findings.append(finding(
            "Rust and Python closed-form RK4 kernels agree on every endpoint to 1e-12", "numerical", gap,
            {"generator": generator,
             "checks": [_check("Python ciw.lab.integrators.step_rk4 with the same closed-form right-hand side and "
                               "operation order", gap, 1e-12, kind="invariant"),
                        _check("Rust evaluation count minus 4 N T", rust["evaluations"] - 4 * STEPS * len(states), 0.0,
                               kind="exact_arithmetic")]},
            unit="radians or radians per unit length", tolerance={"abs": 1e-12, "rel": 0.0}))
        findings.append(finding(
            "Rust kernel endpoint error against the exact great circle is below 1e-8", "numerical",
            float(np.max(rust_errors)),
            {"generator": generator, "checks": [_check("great circle cos(L) X0 + sin(L) T0", float(np.max(rust_errors)), 1e-8)]},
            unit="normalized length", tolerance={"abs": 1e-12, "rel": 1e-3}))
        rust_row = {"available": True, "identity": rust["identity"], "evaluations": rust["evaluations"],
                    "endpoints": rust["states"].tolist(), "max_abs_difference_to_python": gap,
                    "max_ulp_distance_to_python": kernels.ulp_distance(rust["states"], closed),
                    "bitwise_identical_to_python": bool(np.array_equal(rust["states"], closed)),
                    "endpoint_errors": rust_errors.tolist()}
    findings.append(finding(
        "Generic Christoffel-symbol RK4 (ciw.lab.surfaces) and the closed-form kernel agree to 1e-12", "numerical",
        generic_gap, {"generator": generator, "checks": [_check("closed-form sphere Christoffel symbols",
                                                                generic_gap, 1e-12, kind="invariant")]},
        unit="radians or radians per unit length", tolerance={"abs": 1e-12, "rel": 0.0}))
    refusal = independence_refusal()
    expected = "independent_check producer and checker share an implementation origin"
    findings.append(finding(
        "Declaring the Rust kernel an independent check of the Python kernel is refused (both are ciw code)",
        "provenance", {"producer_origin": "ciw", "checker_origin": "ciw"},
        {"checks": [_refusal("ciw.lab.evidence.supported_label with a ciw producer and a ciw checker",
                             expected, refusal)]}))
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
                 f"(bitwise identical: {rust_row['bitwise_identical_to_python']})" if rust_row["available"]
                 else "Rust skipped: " + rust_row["reason"])
    fields = _fields(
        hypothesis="Implementations of the same RK4 geodesic kernel in Python and Rust agree to rounding error, and "
                   "all agree with the exact great circle to the RK4 truncation error.",
        mathematical_model="Unit sphere, theta'' = sin cos phi'^2, phi'' = -2 cot(theta) theta' phi'; RK4 with h = L/N "
                           "and the operation order of ciw.lab.integrators.step_rk4.",
        input_data=[f"{len(states)} equatorial geodesics, headings {list(kernels.HEADINGS)} rad, L = {kernels.LENGTH}, "
                    f"N = {STEPS}"],
        observation_model="Endpoints exchanged as JSON with shortest round-trip float text; the Rust source is "
                          "embedded in energy_gpu_kernels and compiled with rustc -O into a temporary directory.",
        expected_invariant="Python closed-form and Rust endpoints agree to 1e-12 (bitwise where libm agrees); "
                           "generic Christoffel path agrees to 1e-12; every endpoint within 1e-8 of the great circle.",
        experiment="Integrate the same initial states with the generic Python path, the closed-form Python path and "
                   "the compiled Rust kernel; compare endpoints, evaluation counts and exact endpoints; probe Julia "
                   "and GPU availability.",
        numerical_result=f"{rust_text}; generic-closed-form max difference {generic_gap:.3g}.",
        uncertainty="Agreement is limited by libm sin/cos differences across platforms; both kernels are ciw code, "
                    "so shared modelling errors would not be detected.",
        failure_modes_checked=["rustc absent or failing (reported, not hidden)", "Rust refuses malformed input",
                               "nonfinite state", "evaluation count mismatch",
                               "independence declaration between ciw implementations"],
        unresolved_assumptions=["Julia and GPU implementations were not run (toolchain/hardware unavailable)",
                                "Cross-platform bitwise identity is not claimed"],
        recommended_next_task="T147: compare CPU and GPU outputs of this kernel on the RTX 2080 host; T146 for "
                              "one canonical float serialization across languages")
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T119
def _accepted_energy(log, analysis):
    """Recompute the measurement-phase counter delta and accepted solves from raw readings."""
    from ciw import energy_records, free_energy_math
    phase = log["phases"][3]
    readings = [int(s["energy_mj"]) for s in phase["samples"] if s["status"] == "ok"]
    reference = free_energy_math.gaussian_reference(log["plan"]["problem"])
    accepted = 0
    for batch in phase["batches"]:
        mean, covariance = energy_records._result(batch["result"], log["plan"]["replicas"])
        kl = free_energy_math.gaussian_kl(mean, covariance, reference["mean"], reference["covariance"])
        accepted += batch["result"]["replicas"] if kl <= log["plan"]["target_kl_nats"] else 0
    delta_j = (readings[-1] - readings[0]) / 1000 if len(readings) >= 2 else None
    all_readings = [int(s["energy_mj"]) for p in log["phases"] for s in p["samples"] if s["status"] == "ok"]
    return {"delta_j": delta_j, "accepted": accepted, "executed": sum(b["result"]["replicas"] for b in phase["batches"]),
            "span_j": (all_readings[-1] - all_readings[0]) / 1000,
            "analysis_value": analysis["measurement"]["amortized_domain_energy_j_per_qualified_solve"],
            "reasons": analysis["comparison"]["reasons"], "gross_energy_j": analysis["measurement"]["gross_energy_j"]}


@task("T119", changed_files=FILES, regression_tests=(f"{TESTS}::test_energy_per_accepted_result_on_fixtures",))
def energy_per_accepted_result(ctx):
    from ciw import energy_records
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked("T119", "Energy per accepted result is computable from retained fixtures.")
    rows = {}
    for name, raw in raws.items():
        log = json.loads(raw)
        rows[name] = _accepted_energy(log, energy_records.analyze(log))
    base = rows["baseline"]
    recomputed = base["delta_j"] / base["accepted"]
    generator = {"name": "examples/energy-accuracy synthetic fixtures", "origin": "synthetic_fixture", "seed": None}
    withheld = {name: row["reasons"] for name, row in rows.items() if row["analysis_value"] is None}
    under = rows["under-target"]
    naive = under["gross_energy_j"] / under["executed"]
    ratio = (base["span_j"] / base["accepted"]) / recomputed
    findings = [
        finding("Energy per accepted result of the synthetic baseline fixture, recomputed from its raw counter "
                "readings and re-scored outputs, equals the CIW analysis value", "numerical", recomputed,
                {"generator": dict(generator, fixture="baseline"),
                 "checks": [_check("energy_records.analyze amortized_domain_energy_j_per_qualified_solve",
                                   recomputed - base["analysis_value"], 1e-15, kind="exact_arithmetic")]},
                unit="J per accepted solve (synthetic fixture values)", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("The metric is withheld for the reset, missing-bracket and under-target fixtures", "computational_pipeline",
                withheld, {"generator": generator,
                           "checks": [_check("three fixtures with a declared defect", len(withheld) - 3, 0.0,
                                             kind="exact_arithmetic")]}, tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Dividing gross energy by executed solves reports a finite energy per result for the under-target "
                "fixture although no result is accepted", "numerical",
                {"gross_energy_j": under["gross_energy_j"], "executed_solves": under["executed"],
                 "accepted_solves": under["accepted"], "naive_j_per_solve": naive},
                {"generator": dict(generator, fixture="under-target"),
                 "checks": [_check("accepted solves in the under-target fixture", under["accepted"], 0.0,
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 1e-12, "rel": 0.0},
                counterexample={"statement": "Gross energy divided by executed solves is an energy per accepted "
                                             "numerical result",
                                "witness": {"fixture": "under-target", "naive_j_per_solve": naive, "accepted_solves": 0}}),
        finding("Widening the boundary from the measurement phase to the whole run multiplies the baseline energy per "
                "accepted result by 7", "numerical", ratio,
                {"generator": dict(generator, fixture="baseline"),
                 "checks": [_check("(2400 - 1000) mJ / (2100 - 1900) mJ", ratio - 7.0, 1e-12, kind="exact_arithmetic")]},
                unit="ratio", tolerance={"abs": 1e-12, "rel": 0.0}),
        _no_measurement("Physical GPU energy per accepted numerical result",
                        "the fixtures are synthetic and no GPU counter was read here", unit="J/accepted solve"),
    ]
    ctx.artifact_json("energy-per-accepted-result.json", {
        "definition": "E_acc = (counter(last measurement read) - counter(first measurement read)) / accepted solves; "
                      "a solve is accepted when its batch output has KL <= target_kl_nats; undefined when the analysis "
                      "is ineligible or no solve is accepted.",
        "fixtures": rows, "origin": "synthetic_fixture"})
    fields = _fields(
        hypothesis="Energy per accepted result is well defined only when counter brackets are valid and every "
                   "counted solve meets the declared accuracy target; otherwise it must be withheld.",
        mathematical_model="E_acc = Delta E_measurement / #{solves with KL(q || p) <= target}; boundary-dependent "
                           "(measurement phase only, gross, no idle subtraction).",
        input_data=[FIXTURE_NOTE],
        observation_model="Counter readings and retained batch outputs from each fixture; KL rescored with "
                          "free_energy_math.gaussian_kl against the exact Gaussian reference.",
        expected_invariant="Recomputation equals energy_records.analyze; the metric is withheld for reset, missing "
                           "bracket and under-target fixtures.",
        experiment="Analyze all four fixtures; recompute the baseline metric from raw readings; compare with the naive "
                   "gross/executed ratio and with a whole-run boundary.",
        numerical_result=f"baseline E_acc = {recomputed} J per accepted solve (synthetic); withheld for "
                         f"{sorted(withheld)}; naive under-target ratio {naive} J/solve with 0 accepted; "
                         f"whole-run boundary ratio {ratio}.",
        uncertainty="Synthetic values carry no physical uncertainty model; a real NVML counter has undeclared "
                    "resolution and background-inclusive scope.",
        failure_modes_checked=["counter reset", "missing endpoint brackets", "accuracy target not met",
                               "naive division by executed solves", "boundary choice"],
        unresolved_assumptions=["The physical measurement part (T116) could not run here",
                                "Idle subtraction and first-attainment accounting are intentionally not applied"],
        recommended_next_task="T116 on the RTX 2080 host, then rerun T119 on the captured log")
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T120
PRECISION_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048)
TARGETS = (1e-4, 1e-5, 1e-7, 1e-9, 1e-11)


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
    return float(-np.polyfit(x, np.log(np.asarray(errors)), 1)[0])


@task("T120", changed_files=FILES, regression_tests=(f"{TESTS}::test_precision_study_float32_floor_and_counterexample",))
def precision_versus_cost(ctx):
    table = ctx.memo("energy-gpu-precision", _precision_study)
    f32, f64 = table["float32"], table["float64"]
    fit = [i for i, n in enumerate(PRECISION_GRID) if n <= 256]
    order64 = _slope([PRECISION_GRID[i] for i in fit], [f64[i] for i in fit])
    floor32 = min(f32)
    ops = kernels.rk4_operation_count()
    reach = {}
    for target in TARGETS:
        reach[f"{target:g}"] = {name: next((n for n, e in zip(PRECISION_GRID, errs) if e <= target), None)
                                for name, errs in table.items()}
    generator = {"name": "equatorial unit-sphere geodesics", "headings_rad": list(kernels.HEADINGS),
                 "length": kernels.LENGTH, "grid": list(PRECISION_GRID), "seed": None}
    reached64 = sum(row["float64"] is not None for row in reach.values())
    findings = [
        finding("float64 RK4 endpoint error converges at order 4 on the sphere geodesics (N = 16..256)", "numerical",
                order64, {"generator": generator,
                          "checks": [_check("RK4 global order 4", order64 - 4.0, 0.3)]},
                unit="order", tolerance={"abs": 0.05, "rel": 0.0}),
        finding("float32 RK4 endpoint error saturates at a roundoff floor between 1e-7 and 1e-5", "numerical", floor32,
                {"generator": generator,
                 "checks": [_check("upper bound: truncation error at N = 64 in float64 is 2e-7", floor32, 1e-5, "le"),
                            _check("lower bound: unit roundoff 2^-24 = 6e-8 per operation", floor32, 1e-7, "ge")]},
                unit="normalized length", tolerance={"abs": 0.0, "rel": 1.0}),
        finding("Steps needed to reach each accuracy target, by precision (None: not reached for N <= 2048)",
                "numerical", reach,
                {"generator": generator,
                 "checks": [_check("float64 reaches all five targets", reached64 - len(TARGETS), 0.0,
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 1.0}),
        finding("Lowering precision to float32 cannot reach a 1e-7 endpoint accuracy at any step count up to 2048, "
                "while float64 reaches it", "numerical", {"float32_min_error": floor32, "float64_steps": reach["1e-07"]["float64"]},
                {"generator": generator,
                 "checks": [_check("grid points where float32 meets 1e-7", sum(e <= 1e-7 for e in f32), 0.0,
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 1.0},
                counterexample={"statement": "Halving floating-point precision reaches every accuracy target at no "
                                             "greater operation count",
                                "witness": {"target": 1e-7, "float32_min_error": floor32,
                                            "float64_steps": reach["1e-07"]["float64"]}}),
        finding("Per RK4 step the kernel performs the same arithmetic in both precisions; only the state width differs",
                "numerical", dict(ops, state_bytes={"float32": 16, "float64": 32}),
                {"derivation": "Counted from energy_gpu_kernels.sphere_rhs_batch and rk4_batch: 4 x (6 mul + 1 div + "
                               "2 trig) + 3 x 4 x 2 stage flops + 4 x 5 + 4 x 2 combine flops"},
                tolerance={"abs": 0.0, "rel": 0.0}),
        _no_measurement("float32 lowers the energy per accepted trajectory relative to float64 on real CPU or GPU "
                        "hardware", "no energy counter is available; operation counts are only a proxy",
                        unit="J/trajectory"),
    ]
    ctx.artifact_json("precision.json", {"grid": list(PRECISION_GRID), "max_endpoint_error": table,
                                         "targets": reach, "operation_count": ops})
    ctx.artifact_text("precision.svg", svg.line_plot(
        [(name, list(PRECISION_GRID), errs) for name, errs in table.items()],
        title="RK4 sphere geodesic: endpoint error vs steps", xlabel="steps N", ylabel="max endpoint error",
        logx=True, logy=True))
    fields = _fields(
        hypothesis="float32 RK4 matches float64 until truncation error reaches float32 roundoff, after which more "
                   "steps cannot buy accuracy; operation count per step is precision-independent.",
        mathematical_model="Global error ~ C h^4 + c N u with u = 2^-24 (float32) or 2^-53 (float64); cost ~ 4 N "
                           "right-hand-side evaluations of fixed arithmetic.",
        input_data=[f"{len(kernels.HEADINGS)} equatorial sphere geodesics, L = {kernels.LENGTH}, N in {list(PRECISION_GRID)}"],
        observation_model="Endpoints computed entirely in the declared dtype, mapped to R^3 and compared in float64 "
                          "with the exact great circle; maximum over trajectories.",
        expected_invariant="float64 order near 4; float32 error floor above float32 roundoff; op count per step equal.",
        experiment="Vectorized RK4 in float32 and float64 across the step grid; minimal N per accuracy target; "
                   "operation counts derived from the kernel source.",
        numerical_result=f"float64 order {order64:.3f}; float32 floor {floor32:.3g}; steps to target {reach}.",
        uncertainty="float32 errors depend on the platform's float32 sin/cos and vary by tens of percent; "
                    "float64 order fit is stable to about 0.05.",
        failure_modes_checked=["dtype promotion to float64 (asserted)", "roundoff floor mistaken for convergence",
                               "target unreachable at any N (counterexample)"],
        unresolved_assumptions=["Energy cost was not measured; operation counts do not capture memory traffic, "
                                "vector width or GPU float32 throughput"],
        recommended_next_task="T115/T116 energy measurement of both precisions on hardware; T148 for deterministic "
                              "reduction policies")
    return {"state": "partial", "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T121
REDUCTION_N = 1 << 14
REDUCTION_SEED = 20260923


def _atomic_orders(generator, count):
    return [generator.permutation(REDUCTION_N // kernels.BLOCK) for _ in range(count)]


def _cancellation_data(seed):
    """Pairs +a, -a (float32) plus 64 terms of 2^-8: the exact sum is 0.25, the partial sums are large."""
    generator = np.random.Generator(np.random.PCG64(seed))
    half = (REDUCTION_N - 64) // 2
    a = ((generator.random(half) - 0.5) * 2048.0).astype(np.float32)
    values = np.concatenate([a, -a[generator.permutation(half)], np.full(64, 2.0 ** -8, dtype=np.float32)])
    return values[generator.permutation(REDUCTION_N)]


def _reduction_study():
    generator = np.random.Generator(np.random.PCG64(REDUCTION_SEED))
    u = generator.random(REDUCTION_N)
    positive = (u * u * u * 1000.0 + 1e-3).astype(np.float32)
    atomic = _atomic_orders(generator, 8)
    witness_seed, cancellation = None, None
    for seed in range(1, 400):
        candidate = _cancellation_data(seed)
        signs = {np.sign(kernels.sum_sequential(candidate)), np.sign(kernels.sum_tree_strided(candidate)),
                 np.sign(kernels.sum_tree_adjacent(candidate))}
        if len(signs) > 1:
            witness_seed, cancellation = seed, candidate
            break
    if cancellation is None:
        raise ValueError("No reduction-order sign counterexample was found in the searched seeds")
    study = {}
    for label, data in (("positive", positive), ("cancellation", cancellation)):
        exact = math.fsum(float(v) for v in data)
        for dtype in (np.float32, np.float64):
            x = data.astype(dtype)
            sums = kernels.reduction_orders(x, atomic)
            bound = kernels.summation_bound(x, dtype)
            study[f"{label}-{np.dtype(dtype).name}"] = {"exact": exact, "sums": sums, "bound": bound,
                                                        "errors": {k: v - exact for k, v in sums.items()}}
    maxima = {float(np.max(positive[generator.permutation(REDUCTION_N)])) for _ in range(8)}
    return {"study": study, "witness_seed": witness_seed, "max_values": sorted(maxima),
            "n": REDUCTION_N, "block": kernels.BLOCK, "atomic_orders": len(atomic)}


def guard_contradictions(entry, steps=17):
    """Decide S > T only when |S - T| exceeds the a priori bound; count contradicting order pairs."""
    exact, bound, sums = entry["exact"], entry["bound"], list(entry["sums"].values())
    contradictions, undecided, naive_flips = 0, 0, 0
    for k in range(steps):
        threshold = exact + (k - steps // 2) * bound / 4
        decisions = [1 if s - threshold > bound else -1 if threshold - s > bound else 0 for s in sums]
        contradictions += 1 in decisions and -1 in decisions
        undecided += decisions.count(0)
        naive = {s > threshold for s in sums}
        naive_flips += len(naive) > 1
    return {"thresholds": steps, "contradictions": int(contradictions),
            "undecided_fraction": undecided / (steps * len(sums)), "naive_flipping_thresholds": naive_flips}


@task("T121", changed_files=FILES, regression_tests=(f"{TESTS}::test_reduction_orders_bound_and_sign_counterexample",))
def reduction_order_conclusions(ctx):
    result = ctx.memo("energy-gpu-reductions", _reduction_study)
    study = result["study"]
    generator = {"name": "seeded reduction datasets", "seed": REDUCTION_SEED, "n": REDUCTION_N,
                 "cancellation_seed": result["witness_seed"], "bit_generator": "PCG64"}

    def spread(entry):
        values = list(entry["sums"].values())
        return (max(values) - min(values)) / abs(entry["exact"])

    def worst(entry):
        return max(abs(e) for e in entry["errors"].values()) / entry["bound"]

    cancel32, cancel64 = study["cancellation-float32"], study["cancellation-float64"]
    signs32 = {k: int(np.sign(v)) for k, v in cancel32["sums"].items()}
    wrong64 = sum(np.sign(v) != np.sign(cancel64["exact"]) for v in cancel64["sums"].values())
    guards = {key: guard_contradictions(entry) for key, entry in study.items()}
    atomic32 = {v for k, v in study["positive-float32"]["sums"].items() if k.startswith("atomic-")}
    findings = [
        finding("Every emulated float32 summation order of the positive dataset stays within gamma_(n-1) sum|x|; "
                "relative spread across orders", "numerical", spread(study["positive-float32"]),
                {"generator": generator,
                 "checks": [_check("a priori bound for any summation order (Higham 4.2)", worst(study["positive-float32"]),
                                   1.0, "le")]}, unit="relative", tolerance={"abs": 0.0, "rel": 0.5}),
        finding("Every emulated float64 summation order of the same data stays within its bound; relative spread",
                "numerical", spread(study["positive-float64"]),
                {"generator": generator,
                 "checks": [_check("a priori bound for any summation order (Higham 4.2)", worst(study["positive-float64"]),
                                   1.0, "le")]}, unit="relative", tolerance={"abs": 1e-15, "rel": 1.0}),
        finding("Reduction order alone flips the sign of a float32 sum whose exact value is +0.25", "numerical",
                signs32, {"generator": generator,
                          "checks": [_check("distinct signs across emulated orders", len(set(signs32.values())), 2.0,
                                            "ge", kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "The sign of a float32 reduction (a pass/fail decision at threshold 0) "
                                             "does not depend on the reduction order",
                                "witness": {"seed": result["witness_seed"], "exact_sum": cancel32["exact"],
                                            "sums": cancel32["sums"]}}),
        finding("float64 reductions of the same cancellation data give the correct sign in every emulated order",
                "numerical", {k: int(np.sign(v)) for k, v in cancel64["sums"].items()},
                {"generator": generator,
                 "checks": [_check("orders with the wrong sign", wrong64, 0.0, kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Threshold decisions guarded by the a priori bound never contradict across emulated orders",
                "numerical", {k: {"contradictions": g["contradictions"], "undecided_fraction": g["undecided_fraction"]}
                              for k, g in guards.items()},
                {"generator": generator,
                 "checks": [_check("contradicting decisions over 17 thresholds x 4 datasets",
                                   sum(g["contradictions"] for g in guards.values()), 0.0, kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Emulated atomicAdd completion orders of identical float32 block partials give distinct sums",
                "numerical", len(atomic32),
                {"generator": generator,
                 "checks": [_check("distinct float32 results over 8 completion orders", len(atomic32), 2.0, "ge",
                                   kind="exact_arithmetic")]},
                unit="distinct results", tolerance={"abs": 0.0, "rel": 0.0},
                counterexample={"statement": "Accumulating the same block partials yields the same float32 result "
                                             "whatever the atomic completion order",
                                "witness": {"distinct_results": sorted(atomic32)}}),
        finding("Max reductions are order-invariant (bitwise identical over 8 permutations)", "numerical",
                len(result["max_values"]),
                {"generator": generator,
                 "checks": [_check("distinct maxima minus one", len(result["max_values"]) - 1, 0.0,
                                   kind="exact_arithmetic")]},
                unit="distinct results", tolerance={"abs": 0.0, "rel": 0.0}),
        _no_measurement("Reductions on the RTX 2080 (CUB, cuBLAS or atomicAdd) reproduce these emulated spreads and "
                        "sign flips", "no GPU was available; the orders are CPU emulations"),
    ]
    ctx.artifact_json("reductions.json", {**result, "guards": guards})
    names = list(study["positive-float32"]["sums"])
    ctx.artifact_text("reduction-errors.svg", svg.line_plot(
        [(key, list(range(len(names))), [max(abs(entry["errors"][n]), 1e-30) for n in names])
         for key, entry in study.items()],
        title="Absolute error by emulated reduction order", xlabel="order index (see reductions.json)",
        ylabel="|sum - exact|", logy=True))
    fields = _fields(
        hypothesis="GPU-style reduction orders change float32 sums by up to the a priori bound, enough to flip a "
                   "threshold decision near its boundary, while float64 or guard-banded decisions are order-robust.",
        mathematical_model="|fl(sum x) - sum x| <= gamma_(n-1) sum|x_i|, gamma_k = k u / (1 - k u), for every "
                           "summation order; max is exactly associative and commutative.",
        input_data=[f"n = {REDUCTION_N}: positive data (u^3 * 1000 + 1e-3 from PCG64 seed {REDUCTION_SEED}); "
                    f"cancellation data (+a, -a pairs plus 64 x 2^-8, first flipping seed {result['witness_seed']})"],
        observation_model="Sums in sequential, adjacent tree, strided tree, two-pass blocked, block-sequential, "
                          "Kahan and 8 atomic completion orders; exact reference by math.fsum.",
        expected_invariant="Every error within gamma_(n-1) sum|x|; guard-banded decisions consistent; max invariant.",
        experiment="Emulate the orders in float32 and float64; search seeds for a sign flip; sweep 17 thresholds.",
        numerical_result=f"float32 relative spread {spread(study['positive-float32']):.3g}; float64 "
                         f"{spread(study['positive-float64']):.3g}; cancellation float32 signs {signs32}; "
                         f"{len(atomic32)} distinct atomic-order results.",
        uncertainty="Deterministic IEEE arithmetic; the emulated orders are a sample of plausible GPU orders, not "
                    "those of a specific library.",
        failure_modes_checked=["error exceeds the a priori bound", "no sign counterexample in 399 seeds (would raise)",
                               "guard-band contradiction", "max reduction order dependence"],
        unresolved_assumptions=["Real GPU reduction orders, FMA contraction and warp-shuffle trees were not observed",
                                "The guard band is conservative (worst case), so many decisions stay undecided"],
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


@task("T122", changed_files=FILES, regression_tests=(f"{TESTS}::test_bounded_free_energy_identity_and_counterexamples",))
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
    findings = [
        finding("The variational free-energy identity F + log Z = KL(q || p) holds at every iterate to 1e-10 nats",
                "numerical", residual,
                {"generator": generator,
                 "checks": [_check("F from free_energy, log Z from observation-space conditioning, KL from the "
                                   "whitened Gaussian formula", residual, 1e-10, kind="invariant")]},
                unit="nat", tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("KL to the exact posterior decreases monotonically and ends below 1e-12 nats within the "
                "512-iteration bound", "numerical",
                {"iterations": fit["iterations"], "final_kl_nats": kl[-1], "initial_kl_nats": kl[0],
                 "status": fit["status"], "condition_number": runs["declared"]["condition"]},
                {"generator": generator,
                 "checks": [_check("final KL", kl[-1], 1e-12, "le"),
                            _check("largest KL increase between iterates", increase, 1e-15, "le", kind="invariant"),
                            _check("iterations within the declared bound", fit["iterations"], 512, "le",
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 1e-12, "rel": 0.05}),
        finding("The raw-unit posterior and log evidence are invariant under the choice of normalization scales",
                "numerical", invariance,
                {"generator": generator,
                 "checks": [_check("posterior mapped back by x_raw = D x_norm, log Z_raw = log Z_norm - sum log T",
                                   invariance, 1e-10, kind="invariant")]},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("A mean step 1.2 times the stability bound makes KL grow instead of converge", "numerical",
                {"alpha": unstable["alpha"], "spectral_radius": unstable["fit"]["stability"]["spectral_radius"],
                 "kl_initial": unstable_kl[0], "kl_final": unstable_kl[-1]},
                {"generator": generator,
                 "checks": [_check("KL growth factor over 60 iterations", unstable_kl[-1] / unstable_kl[0], 1e3, "ge")]},
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
def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def classify_field(name: str) -> str | None:
    tokens = name.lower().split("_")
    if {"j", "mj", "mw", "joule", "joules"} & set(tokens):
        return "energy"
    if {"nats", "nat", "kl"} & set(tokens) or name in ("free_energy", "log_evidence", "negative_log_evidence") \
            or name.endswith("_entropy") or name.startswith("free_energy"):
        return "information"
    return None


def typed_refusals() -> dict:
    """Attempt forbidden combinations; return each refusal message (None when accepted)."""
    free = kernels.Quantity(8.5, "nat")
    energy = kernels.Quantity(0.2, "J")
    attempts = {"add": lambda: free + energy, "compare": lambda: free < energy,
                "convert": lambda: energy.to("nat"), "untyped": lambda: free + 3.0}
    out = {}
    for name, attempt in attempts.items():
        try:
            attempt()
            out[name] = None
        except kernels.QuantityRefusal as exc:
            out[name] = str(exc)
    return out


@task("T123", changed_files=FILES, regression_tests=(f"{TESTS}::test_typed_quantities_refuse_nats_plus_joules",))
def free_energy_distinct_from_energy(ctx):
    from ciw import energy_records, free_energy_math
    Q = kernels.Quantity
    refusals = typed_refusals()
    expected = {"add": "Cannot add information and energy", "compare": "Cannot compare information and energy",
                "convert": "Cannot express energy in nat", "untyped": "Cannot add a quantity and an untyped number"}
    run = ctx.memo("energy-gpu-free-energy-declared", lambda: _fit(DECLARED_SCALES))
    last = run["fit"]["trace"][-1]
    log_z = run["fit"]["reference"]["log_evidence"]
    identity = (Q(last["free_energy"], "nat") + Q(log_z, "nat") - Q(last["kl_to_reference"], "nat")).to("nat")
    joules = (Q(0.2, "J") + Q(100, "mJ")).to("J") - 0.3
    bits = Q(1, "bit").to("nat") - math.log(2.0)
    per_solve = (Q(0.2, "J") / Q(4, "solve")).describe()
    raws = telemetry.fixture_bytes()
    energy_sources = [energy_records.analyze(json.loads(raws["baseline"])), json.loads(raws["baseline"])] if raws else []
    vi_sources = [run["fit"]["trace"], free_energy_math.free_energy(run["normalized"]["problem"], last["mean"],
                                                                    last["covariance"])]
    energy_fields = sorted({k for source in energy_sources for k in _keys(source) if classify_field(k) == "energy"})
    info_in_energy = sorted({k for source in energy_sources for k in _keys(source) if classify_field(k) == "information"})
    vi_fields = sorted({k for source in vi_sources for k in _keys(source) if classify_field(k) == "information"})
    energy_in_vi = sorted({k for source in vi_sources for k in _keys(source) if classify_field(k) == "energy"})
    panels = {}
    if raws:
        view = ctx.memo("energy-gpu-session", lambda: telemetry.session_replay(raws))["baseline_view"]
        panels = {p["panel_id"]: sorted(set(p["units"])) for p in view["panels"]}
    mixed_panels = sum(len(units) != 1 for units in panels.values())
    f_nats, e_joules = last["free_energy"], 0.2
    findings = [
        finding("Typed arithmetic refuses to add, compare or convert nats of variational free energy with joules",
                "computational_pipeline", refusals,
                {"checks": [_refusal(f"Quantity {name}", expected[name], refusals[name]) for name in expected]}),
        finding("Typed arithmetic within one dimension is exact: 0.2 J + 100 mJ = 0.3 J, 1 bit = ln 2 nat, "
                "F + log Z - KL = 0 nat", "numerical", {"joules": joules, "bits": bits, "identity_nats": identity,
                                                         "energy_per_solve_dimension": per_solve},
                {"checks": [_check("0.3 J", joules, 1e-15, kind="exact_arithmetic"),
                            _check("ln 2", bits, 1e-15, kind="exact_arithmetic"),
                            _check("F + log Z = KL", identity, 1e-10, kind="invariant")]},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("CIW energy records and free-energy records use disjoint unit-bearing fields", "provenance",
                {"energy_fields": energy_fields, "information_fields": vi_fields, "panel_units": panels},
                {"checks": [_check("fixtures available", 0.0 if raws else 1.0, 0.0, kind="exact_arithmetic"),
                            _check("information-class fields in energy logs other than kl_*_nats",
                                   len([k for k in info_in_energy if not k.endswith("_nats")]), 0.0,
                                   kind="exact_arithmetic"),
                            _check("energy-class fields in variational records", len(energy_in_vi), 0.0,
                                   kind="exact_arithmetic"),
                            _check("experiment view panels mixing units", mixed_panels, 0.0, kind="exact_arithmetic")]}),
        finding("An untyped sum of free energy and physical energy changes when the energy unit changes", "numerical",
                {"free_energy_nats": f_nats, "energy_j": e_joules, "sum_with_joules": f_nats + e_joules,
                 "sum_with_millijoules": f_nats + 1000 * e_joules},
                {"checks": [_check("difference between the two untyped sums", 999 * e_joules, 1.0, "ge",
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 1e-9, "rel": 0.0},
                counterexample={"statement": "Adding variational free energy to physical energy yields a "
                                             "unit-independent quantity",
                                "witness": {"free_energy_nats": f_nats, "energy": "0.2 J = 200 mJ"}}),
        _no_measurement("A decrease of variational free energy corresponds to a decrease of physical energy consumed "
                        "by the computation", "relating nats to joules needs a declared physical system and "
                        "temperature (for example k_B T per nat) and a measured device"),
    ]
    ctx.artifact_json("field-audit.json", {"energy_fields": energy_fields, "information_fields_in_energy_records":
                                           info_in_energy, "information_fields": vi_fields,
                                           "energy_fields_in_variational_records": energy_in_vi,
                                           "panel_units": panels, "refusals": refusals})
    fields = _fields(
        hypothesis="Variational free energy (nats, information) and device energy (joules) are different dimensions; "
                   "a typed check refuses to combine them, and CIW records already keep them in disjoint fields.",
        mathematical_model="Quantities carry a dimension vector over {information, energy, time, count}; addition and "
                           "comparison require equal vectors; multiplication and division combine them.",
        input_data=["ciw.free_energy_math trace of the T122 declared problem", FIXTURE_NOTE],
        observation_model="Refusal messages of the typed algebra; field names of energy_records logs/analyses, "
                          "variational traces and the workbench energy view.",
        expected_invariant="nat + J, nat < J and J -> nat refused; same-dimension arithmetic exact; no field or panel "
                           "mixes the two dimensions.",
        experiment="Exercise the typed algebra; classify every field name by unit token; inspect experiment panels.",
        numerical_result=f"refusals {refusals}; energy fields {energy_fields}; information fields {vi_fields}; "
                         f"panel units {panels}.",
        uncertainty="Field classification is by naming convention (unit tokens); a field without a unit token is "
                    "not classified.",
        failure_modes_checked=["implicit nat/J addition", "cross-dimension comparison", "silent conversion",
                               "untyped number mixed with a quantity", "unit-dependent untyped sum"],
        unresolved_assumptions=["The Landauer relation (k_B T ln 2 per bit) is a physical bound on erasure, not an "
                                "estimate of what this computation dissipated"],
        recommended_next_task="T146: carry units in one canonical serialization across languages")
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}


# ----------------------------------------------------------------- T124
@task("T124", changed_files=FILES, regression_tests=(f"{TESTS}::test_raw_telemetry_retention_and_tampering",))
def retain_raw_telemetry(ctx):
    from ciw import energy_records
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked("T124", "Raw telemetry and device/runtime identity are retained in every fixture.")
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
    relabelled = telemetry.reseal(_mutated(base, lambda log: log.update(origin="physical_measurement")))
    mismatch = telemetry.reseal(_mutated(base, lambda log: log["sensor"].update(
        device_uuid="GPU-11111111-2222-3333-4444-555555555555")))
    relabelled_analysis = energy_records.analyze(relabelled)
    mismatch_reasons = energy_records.analyze(mismatch)["comparison"]["reasons"]
    generator = {"name": "examples/energy-accuracy synthetic fixtures", "origin": "synthetic_fixture", "seed": None}
    missing_time = sum(row["samples"] - row["timestamped_samples"] for row in inventory.values())
    missing_identity = sum(identity_fields - row["identity_fields_present"] for row in inventory.values())
    findings = [
        finding(f"Every fixture retains raw timestamped counter readings and {identity_fields} device/runtime "
                "identity fields", "provenance",
                {name: {"samples": row["samples"], "raw_counter_readings": row["raw_counter_readings"],
                        "batches": row["batches"]} for name, row in inventory.items()},
                {"generator": generator,
                 "checks": [_check("readings without monotonic and UTC brackets", missing_time, 0.0, kind="exact_arithmetic"),
                            _check("missing identity fields", missing_identity, 0.0, kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding(f"The energy-log validator refuses {len(telemetry.TAMPERING)} classes of tampering", "provenance",
                sorted(outcomes),
                {"generator": generator,
                 "checks": [_refusal(name, expected, outcomes[name]) for name, _, _, expected in telemetry.TAMPERING]}),
        finding("A log whose counter readings were doubled and then resealed passes validation", "provenance",
                {"gross_energy_j_original": energy_records.analyze(base)["measurement"]["gross_energy_j"],
                 "gross_energy_j_resealed": energy_records.analyze(resealed)["measurement"]["gross_energy_j"]},
                {"generator": generator,
                 "checks": [_check("validation refusals of the resealed log", 0.0, 0.0, kind="exact_arithmetic")]},
                tolerance={"abs": 1e-12, "rel": 0.0},
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
                counterexample={"statement": "The declared origin of a retained energy log authenticates a physical "
                                             "measurement",
                                "witness": {"origin": "physical_measurement", "run_id": base["run_id"]}}),
        finding("A resealed sensor/workload UUID mismatch validates but is withheld by the analysis", "provenance",
                mismatch_reasons,
                {"generator": generator,
                 "checks": [_check("workload_sensor_device_mismatch reported (1 = yes)",
                                   1.0 if "workload_sensor_device_mismatch" in mismatch_reasons else 0.0, 1.0, "ge",
                                   kind="exact_arithmetic")]}),
        _no_measurement("The fixtures' counter readings were produced by a physical GPU and NVML counter",
                        "the fixtures declare origin synthetic_fixture and carry placeholder library and executable "
                        "digests"),
    ]
    ctx.artifact_json("inventory.json", inventory)
    ctx.artifact_json("tampering.json", {"outcomes": outcomes, "resealed_doubled_digest": resealed["log_digest"],
                                         "relabelled_analysis_comparison": relabelled_analysis["comparison"],
                                         "uuid_mismatch_reasons": mismatch_reasons})
    ctx.artifact_json("baseline-measurement-samples.json", base["phases"][3]["samples"])
    fields = _fields(
        hypothesis="Every retained energy log keeps raw timestamped counter readings with device and runtime "
                   "identity, and the validator refuses edits to them unless the log is deliberately resealed.",
        mathematical_model="log_digest = sha256(canonical(log without digest)); structural profile of "
                           "energy_records.validate_log; analysis eligibility rules of energy_records.analyze.",
        input_data=[FIXTURE_NOTE],
        observation_model="Field inventory of each sealed fixture; validation outcome of each mutated copy.",
        expected_invariant="All readings carry monotonic and UTC brackets; identity fields present; each tampering "
                           "class refused with its specific message.",
        experiment=f"Validate the four fixtures; apply {len(telemetry.TAMPERING)} mutations (with or without "
                   "resealing); test resealed doubling, origin relabelling and UUID mismatch.",
        numerical_result=f"{sum(r['samples'] for r in inventory.values())} raw readings retained; refusals "
                         f"{outcomes}; resealed doubling accepted; relabelled fixture classified "
                         f"{relabelled_analysis['comparison']['classification']}.",
        uncertainty="Exact (structural checks).",
        failure_modes_checked=[name for name, *_ in telemetry.TAMPERING] + [
            "resealed modification", "origin relabelling", "sensor/workload UUID mismatch"],
        unresolved_assumptions=["Sealing is integrity, not authenticity: no signature binds a log to the device",
                                "Hardware provenance of every fixture is not established"],
        recommended_next_task="T125: replay the retained logs through a Session; later, signed capture on the RTX "
                              "2080 host (T116)")
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}


def _mutated(log, mutation):
    from copy import deepcopy
    changed = deepcopy(log)
    mutation(changed)
    return changed


# ----------------------------------------------------------------- T125
@task("T125", changed_files=FILES, regression_tests=(f"{TESTS}::test_session_replay_keeps_numerical_result_id",))
def replay_energy_reports(ctx):
    raws = telemetry.fixture_bytes()
    if raws is None:
        return _fixtures_blocked("T125", "Session replay of retained energy logs preserves numerical_result_id.")
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
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Each replay is a fresh analysis occurrence with new execution, result and bundle identities",
                "computational_pipeline", fresh,
                {"generator": generator,
                 "checks": [_check("missing fresh identities", fresh_deficit, 0.0, kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Replay receipts record a numerical match, no state admission, a non-independent method and no fresh "
                "hardware measurement", "provenance", sorted({row["verification_method"] for row in rows.values()}),
                {"generator": generator,
                 "checks": [_check("fixtures with a receipt or view outside policy", receipts_bad, 0.0,
                                   kind="exact_arithmetic")]}),
        finding("Saving and restoring the Session workspace reproduces the retained workbench exactly", "provenance",
                {"bundles": replay["bundle_count"], "restored_equal": replay["restored_equal"]},
                {"generator": generator,
                 "checks": [_check("restored workbench differs (1 = yes)", 0.0 if replay["restored_equal"] else 1.0, 0.0,
                                   kind="exact_arithmetic"),
                            _check("bundles retained minus 2 per fixture", replay["bundle_count"] - 2 * len(rows), 0.0,
                                   kind="exact_arithmetic")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Replaying a retained bundle whose numerical result was edited is refused", "provenance",
                replay["refusals"],
                {"generator": generator,
                 "checks": [_refusal("EnergyAccuracyWorkflow.replay_session on an edited bundle",
                                     "Retained energy analysis binding differs", replay["refusals"].get("edited_bundle"))]}),
        _no_measurement("Replayed energy values are physically valid measurements",
                        "replay recomputes the same retained synthetic readings; it acquires nothing"),
    ]
    ctx.artifact_json("session-replay.json", {"fixtures": rows, "refusals": replay["refusals"],
                                              "restored_equal": replay["restored_equal"],
                                              "bundle_count": replay["bundle_count"]})
    fields = _fields(
        hypothesis="Replaying a retained energy analysis through a ciw Session recomputes the same numerical result "
                   "(stable numerical_result_id) under fresh execution and result identities.",
        mathematical_model="numerical_result_id = digest({operation_id, data = energy_records.analyze(log)}); "
                           "execution and result identities include a fresh occurrence UUID.",
        input_data=[FIXTURE_NOTE],
        observation_model="Session protocol: source.add, operation.execute, bundle.replay, bundle.get, "
                          "experiment.inspect, workspace.save and Session.from_workspace.",
        expected_invariant="One numerical_result_id per fixture; four execution ids, two result ids and two bundle "
                           "digests per fixture; restored workspace equal; edited bundles refused.",
        experiment="Run every fixture through one Session, replay it, compare identities, save and restore the "
                   "workspace, and replay an edited copy of the baseline bundle.",
        numerical_result=f"distinct numerical ids {distinct_numerical}; fresh identities {fresh}; restored equal "
                         f"{replay['restored_equal']}; edited replay: {replay['refusals'].get('edited_bundle')}.",
        uncertainty="Exact; the numerical ids themselves may differ across platforms if float results differ in the "
                    "last bit, so only their stability within a run is asserted.",
        failure_modes_checked=["numerical id drift between original and replay", "reused execution identity",
                               "receipt claims independence or admission", "edited retained result",
                               "workspace restore mismatch"],
        unresolved_assumptions=["Replay determinism is not independent verification by another party",
                                "The retained readings are synthetic"],
        recommended_next_task="T146: canonical serialization so numerical ids agree across platforms and languages")
    return {"state": _state(findings, "completed"), "fields": fields, "findings": findings}
