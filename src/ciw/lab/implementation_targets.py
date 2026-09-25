"""Section 10 (T142-T154): Rust, Python, Julia, C++, GPU and FPGA implementation targets.

Scope: which ciw.lab kernels merit a Rust port (exact operation counts and
interpreter dispatch counts; timings retained only as artifacts), which
industrial interfaces are assigned to C/C++ (required, or preferred where a
pure-Python stack exists) behind pinned subprocess boundaries, a source scan
that evidence code stays in Python, a pinned Julia worker behind SCR's
dispatcher (``julia_worker``, ``implementation_targets_julia``: the damped
oscillator's acceptance set against CIW's closed form, run where the julia and
julia-depot roles are bound), one
canonical JSON encoding with a reference encoder independent of ``json``,
checked against CIW's Python encoders and a Rust implementation compiled at
run time, a CPU/GPU comparison harness exercised CPU-against-CPU and, where
an NVIDIA GPU answers the probe, against the gaussian_vi PTX kernel on the
common Gaussian VI workload of ``energy_gpu_workload``, deterministic reduction policies with
rigorous error bounds, a telemetry-only FPGA frame format with identity,
compatibility and rollback records, a seeded link simulation, and the refusal
boundary for actuator writes and control outputs, with source scans for
machine write paths and control-like outputs.

Non-claims: no FPGA or industrial library runs here, Julia runs only where a
provisioned runtime is bound and only on this platform, and the GPU
runs only on a host whose hardware:nvidia-gpu probe succeeds; all
bitstreams and link statistics are synthetic; Rust agreement is same-origin
(CIW-authored) evidence; no finding establishes physical performance, machine
safety, industrial readiness, production acceptance or actuator authority.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
import math
import sys
import time

import numpy as np

from . import implementation_targets_architecture as arch
from . import implementation_targets_authority as authority
from . import implementation_targets_fpga as fpga
from . import implementation_targets_kernels as kernels
from . import implementation_targets_serial as serial
from . import integrators, jacobi, julia_worker, svg
from .evidence import finding, holds as compare
from .registry import task
from .surfaces import Sphere, Torus

MODULE = "src/ciw/lab/implementation_targets.py"
KERNELS = "src/ciw/lab/implementation_targets_kernels.py"
SERIAL = "src/ciw/lab/implementation_targets_serial.py"
FPGA = "src/ciw/lab/implementation_targets_fpga.py"
AUTHORITY = "src/ciw/lab/implementation_targets_authority.py"
ARCH = "src/ciw/lab/implementation_targets_architecture.py"
DOC = "docs/lab/IMPLEMENTATION_TARGETS.md"
TESTS = "tests/test_lab_implementation_targets.py"
# Fixed evaluation instant for authorization checks; findings never read the wall clock.
NOW = "2026-09-23T00:00:00Z"
EXACT = {"abs": 0, "rel": 0}
# Per-finding uncertainty statements (AUTHORING rule 5).
EXACT_U = {"kind": "roundoff", "value": 0.0,
           "basis": "exact integer, byte, digest or refusal-code comparison; no rounding enters the value"}
DESIGN_U = {"kind": "reference_error", "value": 0.0,
            "basis": "declared design record; the value is a list of names or rules, not a measurement"}


def _roundoff(value, basis):
    return {"kind": "roundoff", "value": float(value), "basis": basis}


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = compare(observed, tolerance, comparison)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, action):
    """Run ``action`` and record the refusal code it raised ('none' when it did not refuse).

    An unexpected exception type is recorded as observed, so a crash is a
    failed check (a refuted finding), never a blocked task.
    """
    try:
        action()
        observed = "none"
    except (ValueError, PermissionError) as exc:
        observed = str(getattr(exc, "code", type(exc).__name__))
    except Exception as exc:  # recorded, not raised: the check fails and the report keeps every other finding
        observed = f"unexpected {type(exc).__name__}"
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _fields(hypothesis, model, inputs, observation, invariant, experiment, next_task, **extra):
    fields = {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
              "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
              "recommended_next_task": next_task, "failure_modes_checked": [], "unresolved_assumptions": []}
    fields.update(extra)
    return fields


def _state(findings, planned_complete: bool = True) -> str:
    """'completed' only when the planned parts ran and no computational finding is refuted.

    A refuted finding in a 'completed' report is refused by the report
    contract, which would turn the whole task blocked and drop every other
    finding; 'partial' keeps the refutation visible next to them.
    """
    from .evidence import COMPUTATIONAL_DOMAINS

    refuted = any(f["domain"] in COMPUTATIONAL_DOMAINS and f["evidence_status"] == "not_established"
                  and not f.get("expected_not_established") for f in findings)
    return "completed" if planned_complete and not refuted else "partial"


def _noop():
    return None


def _passed(checks) -> int:
    return sum(c["passed"] for c in checks)


def _source_digest(relative):
    from .runner import source_digest
    return source_digest(relative)


def _gpu_question() -> str:
    from .energy_gpu_workload import GPU_QUESTION
    return GPU_QUESTION


def _ciw_producer(implementation: str, source: str) -> dict:
    """A ciw producer identity for an independent check: the package version plus the producing module's digest."""
    from .. import __version__
    return {"implementation": implementation, "revision": f"ciw {__version__}", "source_sha256": _source_digest(source)}


def _rust_status(build, run):
    """('ran', result) when the probe ran; ('absent', reason) when rustc is unusable here; ('failed', reason).

    A missing or unusable toolchain is an honest non-claim; a probe that fails
    to build or run with a usable rustc is a refutation.
    """
    if build["available"]:
        try:
            return "ran", run()
        except Exception as exc:  # the probe itself is broken: a refutation, not a missing tool
            return "failed", f"Rust probe failed to run: {type(exc).__name__}"
    if build.get("usable"):
        return "failed", build["reason"]
    return "absent", build["reason"]


def _rust_not_run(claim, domain, status, reason):
    """Finding for a Rust claim the probe could not support: expected when rustc is absent, refuted when broken."""
    if status == "absent":
        return finding(claim, domain, None, {"notes": reason}, expected_not_established=True)
    return finding(claim, domain, {"failure": reason},
                   {"checks": [_check("Rust probe built and ran with a usable rustc (1 = failed)", 1.0)]},
                   uncertainty=EXACT_U, tolerance=EXACT)


# =================================================================== T142
REFERENCE_STEPS = 2000
REFERENCE_UPDATES = 2000
# Doubles read plus written per call of each port unit (state, stage vectors and matrices), for arithmetic intensity.
DOUBLES_MOVED = {"jacobi_transfer": 16, "geodesic_rhs": 8, "kalman_update": 54, "rk4_step": 72}


def kernel_profile() -> dict:
    """Exact operation counts, dispatch counts and agreement of the kernel restatements with their references."""
    torus = Torus(2.0, 1.0)
    f = jacobi.rhs(torus)
    rng = np.random.Generator(np.random.PCG64(142))
    S = kernels.Scalar

    def scal(v):
        return [S(x) for x in v]

    diffs = {"geodesic_rhs": 0.0, "jacobi_rhs": 0.0, "rk4_step": 0.0, "jacobi_transfer": 0.0, "kalman_update": 0.0}
    counts, count_sets = {}, {name: [] for name in diffs}

    def record(name, counted, reference):
        out, tally = counted
        diffs[name] = max(diffs[name], float(np.max(np.abs(kernels.values(out) - reference))))
        count_sets[name].append(tally)

    for trial in range(2):
        y = np.concatenate([rng.uniform(-2.0, 2.0, 2), rng.uniform(-1.0, 1.0, 2)])
        y8 = np.concatenate([y, rng.uniform(-1.0, 1.0, 4)])
        record("geodesic_rhs", kernels.count_ops(kernels.torus_geodesic_rhs, scal(y)), torus.geodesic_rhs(y))
        record("jacobi_rhs", kernels.count_ops(kernels.torus_jacobi_rhs, scal(y8)), f(y8))
        record("rk4_step", kernels.count_ops(kernels.rk4_step, kernels.torus_jacobi_rhs, scal(y8), S(0.01)),
               integrators.step_rk4(f, y8, 0.01))
        record("jacobi_transfer", kernels.count_ops(kernels.transfer, kernels.torus_jacobi_rhs, scal(y8), 1.0, 10),
               integrators.integrate_fixed(f, y8, 1.0, 10)[1][-1])
        # The Kalman update has no single-update core kernel (the core filters are vectorized across runs in
        # another section); the NumPy Joseph-form reference written in the kernels module stands in.
        x, P, z, H, R = kernels.kalman_case(seed=142 + trial)
        x_ref, P_ref = kernels.kalman_update(x, P, z, H, R)
        (x_s, P_s), tally = kernels.count_ops(kernels.kalman_update_scalar, scal(x), [scal(r) for r in P], scal(z),
                                              [scal(r) for r in H], [scal(r) for r in R])
        P_s = np.array([[v.v for v in row] for row in P_s])
        # The covariance is compared relative to its largest entry.
        record("kalman_update", (x_s, tally), x_ref)
        diffs["kalman_update"] = max(diffs["kalman_update"], float(np.max(np.abs(P_s - P_ref)) / np.max(np.abs(P_ref))))
    for name, sets in count_sets.items():
        counts[name] = sets[0]
    count_drift = sum(sets[0] != sets[1] for sets in count_sets.values())
    rk4_formula = 4 * counts["jacobi_rhs"]["flops"] + kernels.rk4_combination_ops(8)
    transfer_formula = 10 * counts["rk4_step"]["flops"] + 1

    # Dispatches: calls issued from ciw frames, minus the harness's own baseline (entry call kept).
    baseline = kernels.dispatch_count(_noop)
    y = np.array([0.4, 0.3, 0.5, -0.2])
    y8 = np.concatenate([y, [1.0, 0.0, 0.0, 1.0]])
    x, P, z, H, R = kernels.kalman_case()

    def measured(function, *args):
        return kernels.dispatch_count(function, *args) - baseline + 1

    transfer_d = {n: measured(integrators.integrate_fixed, f, y8, 1.0, n) for n in (10, 20, 30)}
    per_step = (transfer_d[20] - transfer_d[10]) // 10
    dispatches = {"geodesic_rhs": measured(torus.geodesic_rhs, y), "jacobi_rhs": measured(f, y8),
                  "rk4_step": measured(integrators.step_rk4, f, y8, 0.01), "jacobi_transfer_per_step": per_step,
                  "jacobi_transfer_fixed": transfer_d[10] - 10 * per_step,
                  "kalman_update": measured(kernels.kalman_update, x, P, z, H, R)}
    linearity = (transfer_d[30] - transfer_d[20]) - (transfer_d[20] - transfer_d[10])
    return {"counts": counts, "agreement": diffs, "count_drift": count_drift, "rk4_formula": rk4_formula,
            "transfer_formula": transfer_formula, "dispatches": dispatches, "transfer_dispatches": transfer_d,
            "dispatch_linearity": linearity, "baseline": baseline}


def kernel_ranking(profile: dict) -> list:
    """Rank port units by Python dispatches a port removes from one reference experiment.

    Reference experiment: a Jacobi transfer of 2000 RK4 steps (8000 right-hand
    sides) and 2000 Kalman updates. Porting a unit alone removes its own
    dispatches but keeps any callbacks into Python. Ties would be broken by
    arithmetic intensity (flops per byte moved, higher first), then by
    determinism need. The units are not disjoint: the fused transfer loop
    contains every geodesic_rhs call (jacobi.rhs calls surface.geodesic_rhs)
    and every RK4 step, so it outranks them by construction; its lead over
    geodesic_rhs is the Jacobi and RK4 glue outside the geodesic kernel.
    """
    d, c = profile["dispatches"], profile["counts"]
    transfer_total = REFERENCE_STEPS * d["jacobi_transfer_per_step"] + d["jacobi_transfer_fixed"]
    rows = [
        {"kernel": "jacobi_transfer (fused geodesic + Jacobi RK4 loop)", "unit": "jacobi_transfer",
         "calls_per_experiment": 1, "flops_per_call": REFERENCE_STEPS * c["rk4_step"]["flops"] + 1,
         "dispatches_per_call": transfer_total, "removable_dispatches": transfer_total - 1, "determinism_need": "high",
         "determinism_note": "retained trajectories; bitwise replay needs a fixed order over thousands of dependent steps"},
        {"kernel": "geodesic_rhs (generic embedded Christoffel contraction)", "unit": "geodesic_rhs",
         "calls_per_experiment": 4 * REFERENCE_STEPS, "flops_per_call": c["geodesic_rhs"]["flops"],
         "dispatches_per_call": d["geodesic_rhs"], "removable_dispatches": 4 * REFERENCE_STEPS * (d["geodesic_rhs"] - 1),
         "determinism_need": "high", "determinism_note": "einsum/inverse order must be fixed for bitwise replay"},
        {"kernel": "kalman_update (CIW Joseph-form reference, n=4, m=2; not a core kernel)", "unit": "kalman_update",
         "calls_per_experiment": REFERENCE_UPDATES, "flops_per_call": c["kalman_update"]["flops"],
         "dispatches_per_call": d["kalman_update"], "removable_dispatches": REFERENCE_UPDATES * (d["kalman_update"] - 1),
         "determinism_need": "medium",
         "determinism_note": "covariance symmetry and positive definiteness; small fixed-size reductions"},
        {"kernel": "rk4_step alone (right-hand side stays in Python)", "unit": "rk4_step",
         "calls_per_experiment": REFERENCE_STEPS, "flops_per_call": kernels.rk4_combination_ops(8),
         "dispatches_per_call": d["rk4_step"],
         "removable_dispatches": REFERENCE_STEPS * max(0, d["rk4_step"] - 1 - 4 * d["jacobi_rhs"]),
         "determinism_need": "high", "determinism_note": "stage combination order; negligible alone",
         "dispatch_note": "dispatches per call include the four right-hand-side callbacks, which a lone port keeps"},
    ]
    for row in rows:
        row["bytes_per_call"] = 8 * DOUBLES_MOVED[row["unit"]]
        row["arithmetic_intensity"] = row["flops_per_call"] / row["bytes_per_call"]
    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (-row["removable_dispatches"], -row["arithmetic_intensity"],
                               order[row["determinism_need"]]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _kernel_timings(profile) -> dict:
    torus = Torus(2.0, 1.0)
    f = jacobi.rhs(torus)
    y = np.array([0.4, 0.3, 0.5, -0.2])
    y8 = np.concatenate([y, [1.0, 0.0, 0.0, 1.0]])
    x, P, z, H, R = kernels.kalman_case()
    seconds = {"geodesic_rhs": kernels.time_per_call(torus.geodesic_rhs, y),
               "jacobi_rhs": kernels.time_per_call(f, y8),
               "rk4_step": kernels.time_per_call(integrators.step_rk4, f, y8, 0.01),
               "jacobi_transfer_200_steps": kernels.time_per_call(integrators.integrate_fixed, f, y8, 1.0, 200, budget=0.1),
               "kalman_update": kernels.time_per_call(kernels.kalman_update, x, P, z, H, R)}
    flops = {"geodesic_rhs": profile["counts"]["geodesic_rhs"]["flops"],
             "jacobi_rhs": profile["counts"]["jacobi_rhs"]["flops"], "rk4_step": profile["counts"]["rk4_step"]["flops"],
             "jacobi_transfer_200_steps": 200 * profile["counts"]["rk4_step"]["flops"] + 1,
             "kalman_update": profile["counts"]["kalman_update"]["flops"]}
    return {"seconds_per_call": seconds, "ns_per_flop": {k: 1e9 * seconds[k] / flops[k] for k in seconds},
            "note": "Wall-clock timings on the machine that ran this report; not reproducible, never used in findings."}


def rust_sphere_agreement(steps: int = 600, length: float = 3.0) -> dict:
    """Fused Rust RK4 loop on the unit sphere against ciw.lab.jacobi.transfer and the exact Jacobi field."""
    sphere = Sphere(1.0)
    u0, heading = (1.1, 0.2), 0.3
    start = time.perf_counter()
    core = jacobi.transfer(sphere, u0, heading, length, steps)
    python_seconds = time.perf_counter() - start
    state0 = jacobi.initial_state(sphere, u0, heading)
    rust_state, rust_ns = serial.rust_sphere_rk4(state0, length, steps, 1.0)
    exact = np.array([math.cos(length), -math.sin(length), math.sin(length), math.cos(length)])
    return {"max_abs_difference": float(np.max(np.abs(rust_state - core.states[-1]))),
            "rust_jacobi_error": float(np.max(np.abs(rust_state[4:] - exact))),
            "core_jacobi_error": float(np.max(np.abs(core.states[-1][4:] - exact))),
            "steps": steps, "length": length, "python_seconds": python_seconds, "rust_seconds": rust_ns * 1e-9}


RUST_SPHERE_CLAIM = "Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer on the unit sphere"


@task("T142", changed_files=(MODULE, KERNELS, SERIAL, DOC),
      regression_tests=(f"{TESTS}::test_t142_kernel_counts_and_ranking", f"{TESTS}::test_rust_fused_sphere_loop",
                        f"{TESTS}::test_next_steps_point_at_work_that_delivers"))
def rust_kernels(ctx):
    profile = ctx.memo("t142-profile", kernel_profile)
    ranking = kernel_ranking(profile)
    timings = _kernel_timings(profile)
    build = serial.rust_build()
    status, rust = _rust_status(build, rust_sphere_agreement)
    ctx.artifact_json("kernel-profile.json", {k: v for k, v in profile.items()})
    ctx.artifact_json("kernel-ranking.json", ranking)
    ctx.artifact_json("kernel-timings.json", {"timings": timings, "rust_fused_sphere": rust if status == "ran" else None,
                                              "rust_build": {k: v for k, v in build.items() if k != "binary"}})
    ctx.artifact_text("kernel-timings.svg", svg.line_plot(
        [(name, [profile["counts"][name]["flops"]], [timings["seconds_per_call"][name] * 1e6])
         for name in ("geodesic_rhs", "jacobi_rhs", "rk4_step", "kalman_update")],
        title="Per-call time against exact flops (this machine)", xlabel="flops per call", ylabel="microseconds per call",
        logx=True, logy=True), wall_clock_timing=True)
    max_diff = max(profile["agreement"].values())
    counts_finding = finding(
        "Exact floating-point operation counts of the scalar kernel restatements", "numerical",
        profile["counts"],
        {"generator": {"name": "PCG64 torus states and Kalman case", "seed": 142},
         "checks": [_check("scalar restatements against the core NumPy kernels (geodesic RHS, Jacobi RHS, RK4 step, "
                           "transfer) and the CIW NumPy Joseph-form Kalman reference (max abs difference)", max_diff,
                           1e-12, kind="cross_implementation"),
                    _check("RK4 step count minus 4*RHS - (13n+3), n=8", profile["counts"]["rk4_step"]["flops"]
                           - profile["rk4_formula"]),
                    _check("10-step transfer count minus 10*step - 1", profile["counts"]["jacobi_transfer"]["flops"]
                           - profile["transfer_formula"]),
                    _check("count differences between two random states (branch-free kernels)", profile["count_drift"])]},
        unit="operations per call", uncertainty=EXACT_U, tolerance=EXACT)
    dispatch_finding = finding(
        "Interpreter calls issued by ciw code per kernel call", "computational_pipeline", profile["dispatches"],
        {"checks": [_check("transfer dispatches are linear in steps: D(30)-D(20) - (D(20)-D(10))",
                           profile["dispatch_linearity"])]},
        unit="calls", uncertainty={"kind": "reference_error", "value": 0.25,
                                   "basis": "exact for one Python/NumPy version; relative drift allowed across versions"},
        tolerance={"abs": 0, "rel": 0.25})
    first, second = ranking[0]["removable_dispatches"], ranking[1]["removable_dispatches"]
    margin = (first - second) / first
    ranking_finding = finding(
        "Ranked Rust port recommendation", "computational_pipeline", [row["kernel"] for row in ranking],
        {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#kernel-ranking: rank by Python dispatches removed per "
                       "reference experiment (2000 RK4 steps, 2000 Kalman updates); ties broken by arithmetic "
                       "intensity, then determinism need. The fused loop contains every geodesic_rhs call and RK4 "
                       "step, so it ranks above them by construction; the margin measures only the Jacobi and RK4 "
                       "glue dispatches outside geodesic_rhs",
         "checks": [_check("rank-1 minus rank-2 removable dispatches, relative to rank 1 (the glue outside the "
                           "contained geodesic_rhs calls; order is not a tie)", margin, 0.05, "ge"),
                    _check("ties in removable dispatches (tie-breakers unused)",
                           len(ranking) - len({row["removable_dispatches"] for row in ranking}))]},
        uncertainty=_roundoff(0.0, f"order decided by exact dispatch counts; rank-1 margin {margin:.3f} exceeds the "
                                   "0.05 floor; the fused loop contains the geodesic RHS and RK4 step units, so its lead "
                                   "over them is structural"),
        tolerance=EXACT)
    if status == "ran":
        rust_finding = finding(
            RUST_SPHERE_CLAIM, "numerical", rust["max_abs_difference"],
            {"generator": {"name": "sphere geodesic u0=(1.1, 0.2), heading 0.3, L=3, 600 RK4 steps"},
             "checks": [_check("Rust against ciw.lab.jacobi.transfer final state", rust["max_abs_difference"], 1e-11,
                               kind="cross_implementation"),
                        _check("Rust Jacobi columns against cos(s), sin(s)", rust["rust_jacobi_error"], 1e-8,
                               kind="analytic")]},
            uncertainty=_roundoff(rust["max_abs_difference"], "largest componentwise difference after 600 steps; "
                                                               "different rounding of the same RK4 recurrence"),
            tolerance={"abs": 1e-11, "rel": 0})
    else:
        rust_finding = _rust_not_run(RUST_SPHERE_CLAIM, "numerical", status, rust)
    overhead = finding(
        "Python dispatch overhead, not arithmetic, dominates the run time of the geodesic/Jacobi kernels",
        "computational_pipeline", None,
        {"notes": "Run time is wall-clock and not reproducible, so it stays out of findings; kernel-timings.json "
                  "retains time per flop for each kernel on the machine that ran the report. The reproducible proxy "
                  "is interpreter calls per flop, from the dispatch and operation-count findings; no finding "
                  "separates dispatch time from arithmetic time."},
        expected_not_established=True)
    readiness = finding("Rust ports of the ranked kernels are ready for industrial deployment", "industrial_readiness",
                        None, {})
    top = ranking[0]
    fields = _fields(
        "Python dispatch overhead, not arithmetic, dominates the geodesic/Jacobi kernels (recorded as not established: "
        "only retained timings speak to it); ranked by Python dispatches a port removes, the fused transfer loop is the "
        "best Rust target (it contains every geodesic RHS call and RK4 step, so it ranks above them by construction); "
        "the Kalman update and a standalone RK4 step are poor targets.",
        "Scalar restatements on a counting number type give exact flop counts; RK4 on an n-vector adds 13n+3 flops to "
        "four right-hand sides; interpreter dispatches are calls issued from ciw frames (profile hook); arithmetic "
        "intensity = flops per byte of state, stage vectors and matrices read or written per call.",
        ["Torus R=2, r=1 (generic embedded path); two PCG64(142) states",
         "Joseph-form Kalman update n=4, m=2 written in this section as a reference (the core Kalman filters in other "
         "sections are vectorized across runs and were not profiled)",
         "Unit sphere transfer u0=(1.1, 0.2), heading 0.3, L=3, 600 steps for the Rust probe"],
        "Deterministic counts of arithmetic operations and interpreter calls; wall-clock timings are observed only for "
        "the artifact.",
        "Counts are state-independent and exact; RK4 and transfer counts decompose as predicted; dispatches grow "
        "linearly with steps; the rank-1 unit leads the unit it contains by a clear glue margin; a Rust port of the "
        "fused loop matches the core trajectory to rounding level.",
        "Count operations of scalar restatements, compare them with the core kernels, profile dispatches, rank port "
        "units by removable dispatches, compile the Rust probe and compare its fused sphere loop.",
        "Done in this task: the Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer (cross_implementation check). "
        "Remaining: whether a Rust port saves time or energy on the target machine, which wall-clock timings cannot "
        "establish as a finding; `python -m ciw.lab.energy_gpu_telemetry rapl-capture` now brackets a Rust port and "
        "the NumPy reference of the common Gaussian VI workload on a RAPL host, and the lab run it names reports "
        "their package energy per batch (docs/lab/ENERGY_GPU.md); a port of the ranked fused transfer loop itself "
        "would need its own bracket",
        numerical_result=(f"Flops per call: geodesic RHS {profile['counts']['geodesic_rhs']['flops']}, Jacobi RHS "
                          f"{profile['counts']['jacobi_rhs']['flops']}, RK4 step {profile['counts']['rk4_step']['flops']}, "
                          f"Kalman update {profile['counts']['kalman_update']['flops']}; dispatches per call "
                          f"{profile['dispatches']['geodesic_rhs']}, {profile['dispatches']['jacobi_rhs']}, "
                          f"{profile['dispatches']['rk4_step']}, {profile['dispatches']['kalman_update']} "
                          f"(interpreter calls per flop: geodesic RHS "
                          f"{profile['dispatches']['geodesic_rhs'] / profile['counts']['geodesic_rhs']['flops']:.2f}, "
                          f"Kalman update "
                          f"{profile['dispatches']['kalman_update'] / profile['counts']['kalman_update']['flops']:.3f}). "
                          f"Rank 1: {top['kernel']} ({top['removable_dispatches']} removable dispatches per reference "
                          f"experiment, margin {margin:.3f} over rank 2, which it contains)."
                          + (f" Rust fused sphere loop differs from the core by {rust['max_abs_difference']:.2e}."
                             if status == "ran" else f" Rust probe not run: {rust}.")),
        uncertainty=("Counts are exact for the scalar restatements, which use common subexpressions the NumPy core "
                     "does not (the core repeats trigonometry and forms the full metric), so they count the port "
                     "specification rather than the core's executed operations. Dispatch counts depend on the Python "
                     "and NumPy versions (regression tolerance 25%); operator applications on arrays are not calls "
                     "and are not counted. Timings are machine-specific and appear only in artifacts."),
        failure_modes_checked=["scalar restatement diverging from the core kernel", "branch-dependent counts",
                               "RK4 decomposition formula error (13n+3)", "nonlinear dispatch growth",
                               "ranking decided by a near tie", "Rust port disagreeing with the core or the exact "
                               "Jacobi field", "rustc absent (not established) versus probe broken (refuted)"],
        unresolved_assumptions=["Removable dispatches are a proxy for Python overhead; real speedups need a timed "
                                "port on the target machine", "Whether dispatch overhead dominates run time is not "
                                "established: timings are machine-specific artifacts, never findings",
                                "The ranked units are nested, not alternatives: the fused loop contains the geodesic "
                                "RHS and RK4 step, so the ranking says which enclosing unit to port, not which "
                                "disjoint kernel costs most", "The reference experiment sizes (2000 steps, 2000 "
                                "updates) are declared, not measured workloads",
                                "The Kalman row profiles a CIW reference restatement, not a core kernel",
                                "Only the sphere fused loop was ported, with closed-form sphere Christoffel symbols; "
                                "the generic embedded path is not, so the retained Python/Rust timing ratio overstates "
                                "what a generic port would gain"],
        provider_runtime_identity=_runtime_identity((MODULE, KERNELS, SERIAL), build))
    findings = [counts_finding, dispatch_finding, ranking_finding, overhead, rust_finding, readiness]
    return {"state": _state(findings, status == "ran"), "fields": fields, "findings": findings}


def _runtime_identity(changed, build=None):
    from .runner import builtin_identity
    identity = builtin_identity(changed)
    if build is not None:
        identity["rust_probe"] = dict(build.get("identity") or {}, available=build["available"])
    return identity


# =================================================================== T143
# Importable Python packages probed for the artifact: bindings to the C/C++ libraries (open3d, OCC, pysoem, pypylon,
# PySpin, vmbpy, pcl) and the two pure-Python OPC UA stacks (asyncua, opcua), which bind no C library.
PYTHON_PACKAGES = ("open3d", "OCC", "asyncua", "opcua", "pysoem", "pypylon", "PySpin", "vmbpy", "pcl")


@task("T143", changed_files=(MODULE, ARCH, DOC), regression_tests=(f"{TESTS}::test_t143_interface_inventory",))
def cpp_interfaces(ctx):
    entries = arch.validate_inventory(arch.INTERFACES)
    availability = {name: ctx.available(f"module:{name}") for name in PYTHON_PACKAGES}
    ctx.artifact_json("interface-inventory.json", {"interfaces": list(entries), "boundaries": sorted(arch.BOUNDARIES),
                                                   "python_packages_present_here": availability})
    base = dict(entries[0])
    necessity = {level: [entry["name"] for entry in entries if entry["native_necessity"] == level]
                 for level in ("required", "preferred")}
    fieldbus = next(dict(entry) for entry in entries if "fieldbus_role" in entry)
    mutations = ((base, "boundary", "in_process_binding", "in_process_binding"),
                 (base, "direction", "read_write", "write_capable_direction"),
                 (base, "write_path", "enabled", "write_path_enabled"),
                 (base, "identity", [], "incomplete_entry"),
                 (base, "identity", ["latest"], "unpinned_identity"),
                 (base, "identity", ["library version", "nightly build"], "unpinned_identity"),
                 (base, "identity", ["SDK version >=1.0"], "unpinned_identity"),
                 (base, "identity", ["2024.x"], "unpinned_identity"),
                 (base, "identity", ["HEAD"], "unpinned_identity"),
                 (base, "pure_python_alternatives", [], "incomplete_entry"),
                 (base, "native_necessity", "optional", "incomplete_entry"),
                 (fieldbus, "fieldbus_role", "master", "write_capable_direction"))
    checks = [_refusal(f"inventory entry '{entry['name']}' with {key}={value!r}", code,
                       lambda entry=entry, key=key, value=value: arch.validate_inventory([dict(entry, **{key: value})]))
              for entry, key, value, code in mutations]
    findings = [
        finding("Industrial interfaces assigned to C/C++ libraries behind a pinned subprocess boundary (required, or "
                "preferred over existing pure-Python stacks)", "computational_pipeline", necessity,
                {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#industrial-interfaces"}, uncertainty=DESIGN_U,
                tolerance=EXACT),
        finding("Inventory validator refuses in-process bindings, write-capable directions or bus roles, unpinned "
                "entries and unexplained native preferences", "computational_pipeline",
                sum(check["passed"] for check in checks), {"checks": checks},
                unit="refused mutations", uncertainty=EXACT_U, tolerance=EXACT),
        finding("The listed interfaces are qualified for plant integration", "industrial_readiness", None, {}),
        finding("Vendor camera SDK acquisition meets its timing on real cameras", "sensor_performance", None, {}),
    ]
    fields = _fields(
        "EtherCAT monitoring, vendor camera SDKs, PCL/Open3D and OpenCASCADE need C/C++ libraries; OPC UA does not "
        "(pure-Python stacks such as asyncua exist) but is assigned certified C/C++ stacks for certification and "
        "vendor support. Each interface can sit behind a pinned subprocess boundary that exchanges retained bytes, "
        "leaving evidence code in Python.",
        "Design inventory: interface -> (native libraries, necessity required or preferred with the pure-Python "
        "alternatives passed over, reason, boundary, direction, write path, fieldbus role, identity pins); "
        "validation rules: boundary = pinned_subprocess, direction in {read_only, geometry_exchange}, "
        "write path absent or disabled, fieldbus role passive_tap (a master originates output process data), identity "
        "pins concrete: refused are blank pins, the whole-pin labels current, main, master, trunk, head, dev, develop and x, "
        "the words latest, nightly, snapshot, stable, any, unknown, tbd, n/a and na anywhere, HEAD anywhere, the "
        "characters * ? < > = ~ ^ and N.x wildcards.",
        ["Declared inventory in ciw.lab.implementation_targets_architecture.INTERFACES",
         "Python package availability probe (artifact only): bindings to the C/C++ libraries and the pure-Python OPC "
         "UA stacks"],
        "No library was linked or executed; availability is probed with importlib only.",
        "Every entry validates; every mutated entry is refused with its specific code.",
        f"Validate the inventory and {len(checks)} mutated entries; probe Python packages for the artifact.",
        "Deferred research question: build the first provider executable behind a pinned subprocess boundary (none "
        "exists for any entry), starting with an OPC UA client, and check vendor SDK licensing and platform support "
        "and a passive TAP's inability to inject frames on real hardware, which this inventory assumes",
        numerical_result=(f"{len(entries)} interfaces inventoried ({len(necessity['required'])} requiring C/C++, "
                          f"{len(necessity['preferred'])} preferring it over pure-Python stacks); "
                          f"{findings[1]['value']}/{len(checks)} mutations refused."),
        uncertainty="Analytic design record; library capabilities are stated from vendor documentation, not tested.",
        failure_modes_checked=["in-process binding", "write-capable direction", "enabled write path",
                               "entry without identity pins",
                               "floating identity pins ('latest', 'nightly build', '>=1.0', '2024.x', 'HEAD')",
                               "EtherCAT master in place of a passive tap",
                               "native preference without the pure-Python alternatives it passes over",
                               "unknown native necessity"],
        unresolved_assumptions=["Vendor SDK licensing and platform support were not checked",
                                "The OPC UA choice of C/C++ is a preference (certification, vendor support); the "
                                "pure-Python stacks were not evaluated against a server here",
                                "A passive TAP is assumed to be electrically unable to inject frames; that is a "
                                "hardware property not verified here", "No provider executable exists yet for any entry"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T144
STRUCTURAL_RULES = ("evidence_closure_not_stdlib", "evidence_native_loading", "evidence_spawns_process",
                    "native_loading_outside_allowlist", "shell_invocation", "compiled_extension_in_package")


@task("T144", changed_files=(MODULE, ARCH, SERIAL, DOC), regression_tests=(f"{TESTS}::test_t144_architecture_scan",))
def python_orchestration(ctx):
    scan = ctx.memo("t144-scan", arch.scan_package)
    modules = scan["modules"]
    found = arch.violations(scan)
    structural = [v for v in found if v[0] in STRUCTURAL_RULES]
    heuristic = [v for v in found if v[0] == "spawn_without_runtime_identity"]
    evidence = arch.closure(modules, arch.EVIDENCE_MODULES)
    evidence_violations = [v for v in found if v[0].startswith("evidence_")]
    spawners = sorted(name for name, info in modules.items() if info["spawn_lines"])
    path_resolved = arch.path_resolved_spawners(scan)
    own = "ciw.lab.implementation_targets_serial"
    mention_only = sorted(name for name, info in modules.items() if info["mentions_subprocess"] and not info["spawn_lines"])
    witness = next((name for name in mention_only if not name.startswith("ciw.lab")), None)
    ctx.artifact_json("import-graph.json", {name: info["imports"] for name, info in modules.items()})
    ctx.artifact_json("architecture-scan.json", {
        "package_sha256": scan["package_sha256"], "files": scan["files"], "modules_scanned": len(modules),
        "evidence_closure": evidence, "violations": found,
        "spawning_modules": {n: modules[n]["spawn_lines"] for n in spawners},
        "path_resolved_spawners": {n: {"which_lines": modules[n]["which_lines"], "spawn_lines": modules[n]["spawn_lines"]}
                                   for n in path_resolved},
        "native_loading": {n: i["native"] for n, i in modules.items() if i["native"]},
        "mention_subprocess_without_spawn": mention_only, "unparsed_modules": scan["unparsed"],
        "compiled_extensions": scan["compiled_extensions"]})
    evidence_source = (arch.PACKAGE_ROOT / "lab" / "evidence.py").read_text(encoding="utf-8")
    report_source = (arch.PACKAGE_ROOT / "lab" / "report.py").read_text(encoding="utf-8")
    forged = (("ciw.lab.evidence", evidence_source + "\nimport ctypes\n", "evidence_native_loading"),
              ("ciw.lab.report", report_source + "\nimport numpy\n", "evidence_closure_not_stdlib"),
              ("ciw.lab.evidence", evidence_source + "\nimport subprocess\nsubprocess.run(['solver'])\n",
               "evidence_spawns_process"),
              # Package __init__ modules run whenever an evidence module is imported, so they are in the closure.
              ("ciw.lab", "import numpy\n", "evidence_closure_not_stdlib"),
              ("ciw.core", "import ctypes\n", "evidence_native_loading"),
              ("ciw", "import subprocess\nsubprocess.run(['solver'])\n", "evidence_spawns_process"),
              ("ciw.lab.forged_provider", "import subprocess\nsubprocess.run('solver --fast', shell=True)\n",
               "shell_invocation"),
              ("ciw.lab.forged_provider", "import asyncio\nasyncio.create_subprocess_shell('solver --fast')\n",
               "shell_invocation"),
              ("ciw.lab.forged_provider", "import subprocess\nsubprocess.run(['solver'])\n",
               "spawn_without_runtime_identity"),
              ("ciw.lab.forged_provider", "import subprocess\n# records the solver digest\nsubprocess.run(['solver'])\n",
               "spawn_without_runtime_identity"),
              ("ciw.lab.forged_provider", "from os import posix_spawn\nposix_spawn('/opt/solver', ['solver'], {})\n",
               "spawn_without_runtime_identity"))
    mutation_checks = []
    for name, text, rule in forged:
        rules = {v[0] for v in arch.violations(arch.mutated_scan(scan, name, text)) if v[1] == name}
        observed = rule if rule in rules else (",".join(sorted(rules)) or "none")
        mutation_checks.append({"reference_kind": "refusal", "reference": f"forged {name} ({rule})",
                                "expected_refusal": rule, "observed_refusal": observed, "passed": rule in rules})
    pin_claim = "Numerical providers are invoked only through pinned executables"
    findings = [
        finding("Evidence and identity closure is standard-library Python with no native loading or process spawns",
                "computational_pipeline", evidence,
                {"checks": [_check("evidence-closure rule violations", len(evidence_violations))]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Package-wide structural rules hold (no shell spawns, native loading only in declared hardware probes, "
                "no compiled extensions)", "computational_pipeline", len(structural),
                {"checks": [_check("structural rule violations across the scanned package", len(structural)),
                            _check("modules that failed to parse", len(scan["unparsed"]))]},
                unit="violations", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Every process-spawning module names an identity in its code (heuristic, not a pin)",
                "computational_pipeline", len(heuristic),
                {"checks": [_check("spawning modules without revision, source_tree, runtime_identity, sha256 or digest "
                                   "in identifiers or non-docstring strings", len(heuristic))]},
                unit="violations", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Scanner flags forged modules that cross the boundary", "computational_pipeline",
                sum(c["passed"] for c in mutation_checks), {"checks": mutation_checks}, unit="detected mutations",
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Some process spawns run a PATH-resolved executable without comparing it to a pinned identity",
                "computational_pipeline", {"witness": own},
                {"checks": [_check("modules that resolve an executable with shutil.which and spawn it",
                                   len(path_resolved), 1, "ge"),
                            _check(f"{own} is among them", float(own in path_resolved), 1, "ge")]},
                counterexample={"statement": pin_claim,
                                "witness": {"module": own, "executable": "rustc from shutil.which('rustc')",
                                            "recorded": "rustc -vV release and commit (provenance, compared with nothing)"}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding(pin_claim, "computational_pipeline", None,
                {"notes": "A source scan cannot show that a spawned binary equals an expected identity; see the "
                          "PATH-resolved counterexample"}, expected_not_established=True),
        finding("Text search for 'subprocess' finds modules that spawn no process", "computational_pipeline",
                {"witness": witness},
                {"checks": [_check("modules outside ciw.lab mentioning subprocess without a spawn call",
                                   sum(not n.startswith("ciw.lab") for n in mention_only), 1, "ge")]},
                counterexample={"statement": "A text search for 'subprocess' identifies the process-spawning modules",
                                "witness": {"module": witness, "reason": "mentions ciw.adapters.subprocess or quotes "
                                            "subprocess in text but has no spawn call"}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Keeping native code behind subprocess boundaries makes machine interfaces safe", "machine_safety",
                None, {}),
    ]
    fields = _fields(
        "The evidence and identity layer (ciw.lab.evidence, ciw.lab.report, ciw.core.identities, their ciw imports "
        "and the packages whose __init__ runs when they are imported) is pure standard-library Python; process "
        "spawns use argument vectors without a shell; native "
        "loading is confined to declared hardware probes. Whether spawned providers are pinned is not decidable "
        "from source and is tested only by a heuristic plus counterexample search.",
        "Directed import graph over parsed modules; transitive closure from the evidence roots, adding each visited "
        "module's ancestor packages (Python executes their __init__ first); structural rules "
        "{closure stdlib-only, no native/spawn in closure, native loading within allowlist, no shell (shell=True, "
        "os.system/popen, asyncio.create_subprocess_shell), no compiled extensions}; heuristic rule: a spawning "
        "module names an identity token in identifiers or non-docstring strings.",
        [f"{len(modules)} parsed modules of the installed ciw package (source text only, not imported); package "
         "digest in the provider/runtime identity"],
        "Static source analysis with ast; nothing from the scanned modules is imported or executed.",
        "Zero structural and heuristic violations on the real package; every forged mutation violates its intended "
        "rule; PATH-resolved spawners exist, so pinning is not established.",
        f"Scan the package, compute the evidence closure, evaluate the rules, replay {len(forged)} forged mutations "
        "and list spawners that resolve executables on PATH.",
        "Pin the Rust probe and other PATH-resolved tools against expected identities (a resolved rustc is recorded "
        "by version and binary digest, not checked against a pin)",
        numerical_result=(f"Evidence closure {evidence}; {len(structural)} structural and {len(heuristic)} heuristic "
                          f"violations over {len(modules)} modules; {len(spawners)} spawning modules, "
                          f"{len(path_resolved)} of them PATH-resolved; {findings[3]['value']}/{len(forged)} forged "
                          "mutations detected."),
        uncertainty=("Static scan: dynamic imports, exec, importlib and spawns through third-party libraries are not "
                     "seen; the identity-token rule shows that a name appears in code, not that the spawned binary "
                     "is pinned."),
        failure_modes_checked=["third-party import in evidence closure", "native loading", "process spawn in evidence",
                               "violation in a package __init__ that runs on import (ciw, ciw.lab, ciw.core)",
                               "shell=True", "asyncio shell spawn", "os.posix_spawn imported by name",
                               "spawn without identity", "identity token only in a comment", "PATH-resolved "
                               "executables", "compiled extension", "unparseable module"],
        unresolved_assumptions=["Modules added concurrently by other sections are scanned as found at run time "
                                "(identified by the package digest)",
                                "Hardware probes load native code in-process by declared exception: the energy probes "
                                "(ciw.energy_cuda, ciw.energy_nvml) load drivers, and ciw.lab.blas_probe opens the "
                                "OpenBLAS NumPy already loaded, through ctypes, only to read which kernel it runs"],
        provider_runtime_identity=dict(_runtime_identity((MODULE, ARCH)), scanned_package_sha256=scan["package_sha256"],
                                       scanned_files=scan["files"]))
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T145
JULIA = "src/ciw/lab/implementation_targets_julia.py"
JULIA_HOST = "src/ciw/lab/julia_worker.py"
JULIA_ENVIRONMENT = ("src/ciw/lab/julia/oscillator_worker.jl", "src/ciw/lab/julia/Project.toml",
                     "src/ciw/lab/julia/Manifest.toml", "src/ciw/lab/julia/julia-runtime.json")
OSCILLATOR = "src/ciw/adapters/oscillator.py"
JULIA_DOC = "docs/JULIA_SP1.md"
JULIA_PLAN = {
    "reference": "docs/JULIA_SP1.md (CIW -> SCR execution boundary)",
    "pinned_version": "Julia 1.10.12 LTS: official archive, its sha256 and Julia's checksum file "
                      "(src/ciw/lab/julia/julia-runtime.json)",
    "steps": [
        "Provision separately from execution: scripts/provision_julia.py downloads the official archive, checks it "
        "against the pin and Julia's checksum file, instantiates the committed Manifest into a depot and precompiles it",
        "Bind the executable and the depot explicitly: --provider julia=<julia executable> "
        "--provider julia-depot=<depot>",
        "Start the worker with --project=<packaged environment> --startup-file=no --threads=1, JULIA_DEPOT_PATH=<depot> "
        "and JULIA_LOAD_PATH=@ and @stdlib; it refuses to start on an environment that was not precompiled",
        "Compare the handshake field by field with the expected identity before any request",
        "Dispatch the oscillator fixtures through SCR's SpecificationDispatcher with the worker as its runner; retain "
        "the exact frames and recompute SCR's commitments from them",
        "Accept the oscillator operation only after the analytic-oracle fixtures pass on Windows and Linux",
    ],
    "identity_fields": list(julia_worker.HANDSHAKE_FIELDS),
}
NEXT_STEPS = {
    "T145": ("Run the worker acceptance set on Windows x86-64: provision the pinned julia-1.10.12-win64.zip with "
             "scripts/provision_julia.py in a Windows CI job, bind julia and julia-depot there, and compare every "
             "fixture value with this Linux run within the declared tolerances; widen the operation allowlist only "
             "through new fixtures"),
    "T145-unbound": ("Provision Julia 1.10.12 LTS and the worker environment (scripts/provision_julia.py), bind them "
                     "with --provider julia=<executable> --provider julia-depot=<depot> beside the SCR checkout and "
                     "rerun this task; then run the acceptance set on Windows x86-64"),
}


def sympy_torus_check(samples: int = 16) -> dict:
    """Christoffel symbols and curvature of the torus derived symbolically, compared with ciw.lab.surfaces."""
    import sympy as sp

    phi, theta = sp.symbols("phi theta", real=True)
    R, r = sp.symbols("R r", positive=True)
    X = sp.Matrix([(R + r * sp.cos(theta)) * sp.cos(phi), (R + r * sp.cos(theta)) * sp.sin(phi), r * sp.sin(theta)])
    coords = (phi, theta)
    J = [X.diff(c) for c in coords]
    g = sp.Matrix(2, 2, lambda i, j: sp.simplify(J[i].dot(J[j])))
    ginv = g.inv()
    gamma = [[[sp.simplify(sum(ginv[k, l] * (sp.diff(g[j, l], coords[i]) + sp.diff(g[i, l], coords[j])
                                             - sp.diff(g[i, j], coords[l])) for l in range(2)) / 2)
               for j in range(2)] for i in range(2)] for k in range(2)]
    E, G = g[0, 0], g[1, 1]
    W = sp.sqrt(E * G)
    K = sp.simplify(-(sp.diff(sp.diff(G, phi) / W, phi) + sp.diff(sp.diff(E, theta) / W, theta)) / (2 * W))
    values = {R: 2, r: 1}
    gamma_f = sp.lambdify((phi, theta), [[[entry.subs(values) for entry in row] for row in block] for block in gamma],
                          modules="math")
    K_f = sp.lambdify((phi, theta), K.subs(values), modules="math")
    torus = Torus(2.0, 1.0)
    rng = np.random.Generator(np.random.PCG64(145))
    worst = 0.0
    for _ in range(samples):
        u = rng.uniform(-math.pi, math.pi, 2)
        symbolic = np.array(gamma_f(*u), dtype=float)
        worst = max(worst, float(np.max(np.abs(symbolic - torus.christoffel(u)))),
                    abs(float(K_f(*u)) - torus.gaussian_curvature(u)))
    return {"max_abs_difference": worst, "samples": samples, "metric_off_diagonal": str(g[0, 1]),
            "christoffel": {f"Gamma^{k}_{i}{j}": str(gamma[k][i][j]) for k in range(2) for i in range(2)
                            for j in range(i, 2)}, "gaussian_curvature": str(K), "sympy": sp.__version__}


SYMBOLIC_CLAIM = "Symbolic torus Christoffel symbols and curvature agree with ciw.lab.surfaces.Torus"
JULIA_SCR_CLAIM = "Julia environment pinned and exercised through the CIW to SCR boundary"
JULIA_PIN_CLAIM = "Julia provider pin procedure"


def _symbolic_findings(ctx) -> tuple:
    """SymPy's derivation of the torus geometry against the core (the symbolic role), where SymPy imports."""
    if not ctx.available("module:sympy"):
        return None, []
    import sympy

    symbolic = sympy_torus_check()
    ctx.artifact_json("sympy-torus.json", symbolic)
    return symbolic, [finding(
        SYMBOLIC_CLAIM, "mathematical", symbolic["max_abs_difference"],
        {"independent_check": dict(_check("sympy-derived Gamma and K against the core at 16 seeded points",
                                          symbolic["max_abs_difference"], 1e-12, kind="analytic"),
                                   producer=_ciw_producer("ciw.lab.surfaces.Torus", "src/ciw/lab/surfaces.py"),
                                   checker={"implementation": "sympy", "revision": sympy.__version__})},
        uncertainty=_roundoff(symbolic["max_abs_difference"], "largest difference between float evaluations of "
                                                               "the symbolic expressions and the core"),
        tolerance={"abs": 1e-12, "rel": 0})]


def _scr_binding(ctx) -> dict:
    """Whether the bound SCR checkout may host the dispatch: at a CIW pin, clean (the check T097 shares)."""
    from subprocess import SubprocessError

    from . import exchange_provenance_bundles_providers as scr_providers

    if not ctx.available("provider:scr"):
        return {"accepted": False, "reason": "no SCR checkout is bound (--provider scr=<checkout>)", "identity": None}

    def read():
        try:
            return scr_providers.checkout_identity(ctx.providers["scr"]), None
        except (ValueError, OSError, SubprocessError) as exc:
            return None, type(exc).__name__
    identity, error = ctx.memo(("exchange-bundles:identity", str(ctx.providers["scr"])), read)
    if identity is None:
        return {"accepted": False, "reason": f"the bound SCR path is not a readable checkout ({error})",
                "identity": None}
    comparison = scr_providers.compare_with_pins("scr", identity, scr_providers.ciw_pins())
    summary = {"revision": identity["head"], "source_tree": identity["tree"], "clean": comparison["clean"],
               "matched_pins": comparison["matched"]}
    if not comparison["accepted"]:
        return {"accepted": False, "identity": summary,
                "reason": f"the bound SCR checkout (HEAD {identity['head']}) is not a clean checkout at a CIW pin"}
    return {"accepted": True, "reason": "", "identity": summary,
            "basis": {"provider": {"repository": scr_providers.REPOSITORIES["scr"], "revision": identity["head"],
                                   "source_tree": identity["tree"], "executed": True}}}


def _julia_unbound(ctx, symbolic, symbolic_findings, reason: str):
    """Today's findings when no Julia worker can run: the pin procedure as a derivation and the SCR claim open."""
    findings = [finding(JULIA_SCR_CLAIM, "computational_pipeline", None,
                        {"notes": f"The Julia worker, its handshake and the SCR dispatch path exist, but {reason}, so "
                                  "no Julia process ran and no Julia finding is established"},
                        expected_not_established=True),
                finding(JULIA_PIN_CLAIM, "provenance", JULIA_PLAN["identity_fields"],
                        {"derivation": "docs/JULIA_SP1.md#worker-lifecycle-and-environment"}, uncertainty=DESIGN_U,
                        tolerance=EXACT), *symbolic_findings]
    fields = _fields(
        "Julia can carry numerical, symbolic and exploratory work behind the pinned CIW to SCR boundary; until a "
        "pinned Julia runtime is bound, SymPy demonstrates the symbolic role on the geometry core.",
        "Pin procedure from docs/JULIA_SP1.md; symbolic Christoffel symbols Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il "
        "- d_l g_ij) and K = -(1/2W)[d_phi(G_phi/W) + d_theta(E_theta/W)], W = sqrt(EG), for the torus embedding.",
        ["Torus R=2, r=1", "16 PCG64(145) chart points", "docs/JULIA_SP1.md", "src/ciw/lab/julia/julia-runtime.json"],
        "Symbolic derivation evaluated at sample points; no Julia process ran.",
        "Symbolic and core Christoffel symbols and curvature agree to rounding; Julia claims stay unestablished.",
        "Probe the julia and julia-depot bindings; record the pin procedure; derive torus geometry with SymPy and "
        "compare with the core.",
        NEXT_STEPS["T145-unbound"],
        numerical_result=(f"Julia worker not run: {reason}; SymPy agreement max |difference| = "
                          f"{symbolic['max_abs_difference']:.2e}" if symbolic
                          else f"Julia worker not run: {reason}; SymPy absent"),
        uncertainty="SymPy stands in for the symbolic role only; it says nothing about Julia performance or pinning.",
        failure_modes_checked=["julia or julia-depot unbound (recorded, not substituted)",
                               "symbolic/core Christoffel disagreement", "symbolic/core curvature disagreement"],
        unresolved_assumptions=[f"No Julia worker ran: {reason}",
                                "Windows x86-64 execution of the worker is not run",
                                "Optimization and exploratory roles are not demonstrated"])
    return {"state": "partial", "fields": fields, "findings": findings}


@task("T145", changed_files=(MODULE, JULIA, JULIA_HOST, *JULIA_ENVIRONMENT, OSCILLATOR, DOC, JULIA_DOC),
      regression_tests=(f"{TESTS}::test_t145_julia_partial_with_plan",
                        f"{TESTS}::test_sympy_torus_geometry_matches_core",
                        f"{TESTS}::test_t145_julia_worker_acceptance_set",
                        f"{TESTS}::test_julia_host_refuses_invalid_requests_before_dispatch",
                        f"{TESTS}::test_julia_session_failures_end_the_session_without_a_result",
                        f"{TESTS}::test_julia_encodings_and_scr_commitments"))
def julia_role(ctx):
    from . import implementation_targets_julia as study

    ctx.artifact_json("julia-pin-procedure.json", JULIA_PLAN)
    symbolic, symbolic_findings = _symbolic_findings(ctx)
    bound = {role: ctx.available(f"provider:{role}") for role in ("julia", "julia-depot")}
    if not all(bound.values()):
        unbound = " and ".join(role for role, present in bound.items() if not present)
        return _julia_unbound(ctx, symbolic, symbolic_findings,
                              f"no {unbound} binding is present (--provider julia=<executable> "
                              "--provider julia-depot=<depot>)")
    runtime = julia_worker.JuliaRuntime(ctx.providers["julia"], ctx.providers["julia-depot"])
    scr = _scr_binding(ctx)
    result = study.run(runtime, study.plan(), ctx.providers["scr"] if scr["accepted"] else None)
    records = {record["label"]: record for record in result["records"]}
    first = records["start worker-1"]
    if first["outcome"] != "accepted":
        # A bound runtime the handshake refuses is a refused binding, not evidence about the claims.
        refusal = (f"JULIA_ENVIRONMENT_REFUSED: the bound Julia worker did not pass the handshake ({first['outcome']}: "
                   f"{', '.join((first.get('comparison') or {}).get('mismatches', [])) or first.get('failure_detail', '')})")
        return _julia_unbound(ctx, symbolic, symbolic_findings, refusal)
    analysis = study.analyse(result)
    ctx.artifact_json("julia-frames.json", study.retained_frames(records))
    exchanges = [{key: value for key, value in record.items()
                  if key not in ("request_frame", "response_frame", "handshake", "startup_s", "elapsed_s",
                                 "session_id")} | {"handshake": study.sanitized_handshake(record.get("handshake"))}
                 for record in result["records"]]
    ctx.artifact_json("julia-exchanges.json", {"path": result["path"], "exchanges": exchanges,
                                               "scr_recomputed": analysis["scr"]})
    ctx.artifact_json("julia-acceptance.json", {"thresholds": study.ACCEPTANCE, "configuration":
                                                study.DEFAULT_CONFIGURATION, "comparisons": analysis["comparisons"]})
    ctx.artifact_json("julia-timings.json", {
        "note": "wall-clock seconds on the recording host; never compared",
        "sessions": {record["session"]: record["startup_s"] for record in result["records"] if record["op"] == "start"},
        "requests": {record["label"]: record["elapsed_s"] for record in result["records"] if record["op"] == "request"},
        "session_ids": {name: entry["session_id"] for name, entry in result["sessions"].items()},
        "stderr_bytes": {name: entry["stderr_bytes"] for name, entry in result["sessions"].items()}})
    # Offline restore: the retained artifacts, read back from disk, decode and reproduce SCR's commitments.
    directory = ctx.output_dir / "artifacts" / ctx.task_id
    frames = study.restore_frames(json.loads((directory / "julia-frames.json").read_text(encoding="utf-8")))
    recorded = {entry["label"]: entry["scr"] for entry in json.loads(
        (directory / "julia-exchanges.json").read_text(encoding="utf-8"))["exchanges"]
        if entry.get("scr", {}).get("dispatch") == "measurement"}
    offline = study.offline_restore(frames, recorded)
    findings = study.acceptance_findings(result, analysis, offline, scr, JULIA_PLAN["identity_fields"])
    findings += symbolic_findings
    handshake = study.sanitized_handshake(first["handshake"])
    default, ladder = analysis["comparisons"].get("A1", {}), analysis["comparisons"]
    fields = _fields(
        "A pinned Julia 1.10.12 worker integrating the damped oscillator with OrdinaryDiffEq's Tsit5, dispatched "
        "through SCR's SpecificationDispatcher, agrees with CIW's closed form within declared componentwise "
        "thresholds, keeps its declared identity, refuses what it cannot run and never returns a result from a failed "
        "exchange; SymPy demonstrates the symbolic role.",
        "q' = v, v' = -2 gamma v - omega_0^2 q, E = 0.5 m (v^2 + omega_0^2 q^2); closed form q = exp(-gamma t) "
        "(q0 cos(w t) + b sin(w t)), w = sqrt(omega_0^2 - gamma^2), b = (v0 + gamma q0) / w; Tsit5 (order 5 with an "
        "order-4 error estimate, PI step control, free order-4 interpolation at the saved times); SCR commitment "
        "sha256(len|tag|count|(len|field)...); the torus symbolic derivation.",
        ["CIW default oscillator: omega_0 = 2 pi 0.8 rad/s, gamma = 0.15 1/s, m = 1 kg, q0 = 1 m, v0 = 0, 768 samples "
         "at 64 Hz over [0, 12)", "Mixed state: omega_0 = 3, gamma = 0.4, m = 2.5, q0 = -0.7, v0 = 2.5, 600 samples "
         "at 50 Hz", "Undamped: the default with gamma = 0", "abstol = reltol 1e-10 (ladder 1e-6 to 1e-12), dt 1e-3, "
         "dtmax 12, maxiters 1e6", "Julia 1.10.12 with the committed Manifest; the bound SCR checkout"],
        "Exact request and response frames are retained; outputs are decoded from them and compared sample by sample "
        "with the closed form; SCR's identities are recomputed from the frames; each handshake is compared field by "
        "field with the expected identity.",
        "|x - x_ref| <= abs + rel |x_ref| for q (1e-6 m, 1e-6), v (1e-5 m/s, 1e-6) and E (1e-5 J, 1e-6); undamped "
        "phase error <= 1e-6 rad and energy drift <= 1e-6; errors fall as tolerances tighten; repeated occurrences "
        "agree within 1e-12; recomputed SCR identities equal SCR's; no result from a refused, halted or failed "
        "exchange.",
        "Start a pinned worker, dispatch A (default), B (mixed), A, the undamped fixture and the tolerance ladder, a "
        "halting request, six requests the worker must refuse and A again, then an oversized frame; restart, A, a "
        "stalled frame (timeout); restart and crash; start with 2 threads where 1 is declared; six protocol-mock "
        "failures; read the retained frames back and recompute SCR's commitments.",
        NEXT_STEPS["T145"],
        numerical_result=(
            f"Worker handshake accepted ({handshake['julia_version']}, {handshake['machine']}); path: {result['path']}; "
            f"default fixture max |error| q {default['max_abs_error']['q']:.2e} m, v {default['max_abs_error']['v']:.2e} "
            f"m/s, E {default['max_abs_error']['energy']:.2e} J (largest threshold ratio "
            f"{default['max_threshold_ratio']:.1e}); accepted steps {ladder['reltol-1e-06']['naccept']} to "
            f"{ladder['reltol-1e-12']['naccept']} from reltol 1e-6 to 1e-12"
            + (f"; SymPy agreement max |difference| = {symbolic['max_abs_difference']:.2e}" if symbolic else "")),
        uncertainty=("Errors are the solver's global error against a closed form evaluated in binary64 (reference "
                     "rounding below 1e-13); they move by at most 0.8 % between FMA and non-FMA Julia code "
                     "generation, and step counts not at all. Timings are host measurements, retained only in "
                     "julia-timings.json."),
        failure_modes_checked=[
            "handshake mismatch (threads 2 against 1 declared)", "invalid numbers, grids and oversized frames before "
            "dispatch", "nonfinite, out-of-bound, unsorted, truncated and unknown-program requests at the worker",
            "a solve that halts (MaxIters)", "oversized frame, stalled request (timeout), crash", "truncated, "
            "malformed, oversized and mismatched responses (protocol mock)", "state leakage between occurrences",
            "SCR identities recomputed from retained bytes", "symbolic/core Christoffel and curvature disagreement"],
        unresolved_assumptions=[
            "Windows x86-64 execution of the worker is not run; only this Linux host's results exist",
            "The handshake is the worker's declaration: a process replaying it passes (the protocol mock does)",
            "SCR's dispatcher records every runner's output as simulation:deterministic_native_execution; the "
            "label is SCR's at the pin and does not name the Julia worker",
            "Admission of the dispatched measurements through SCR's run_experiment_step is not exercised",
            "Optimization and exploratory roles in Julia are not demonstrated; SymPy, not Julia, shows the symbolic "
            "role"],
        provider_runtime_identity=dict(
            _runtime_identity((MODULE, JULIA, JULIA_HOST, *JULIA_ENVIRONMENT, OSCILLATOR)),
            julia={"handshake": handshake, "accepted": True,
                   "compared_fields": first["comparison"]["compared"],
                   "packages_verified": first["comparison"]["packages_verified"],
                   "sysimage_sha256": first["comparison"]["sysimage_sha256"], "path": result["path"]},
            scr={"accepted": scr["accepted"], "reason": scr["reason"], "identity": scr["identity"]}))
    return {"state": "partial", "fields": fields, "findings": findings}


# =================================================================== T146
def serialization_study() -> dict:
    from ..core.identities import canonical_json
    from ..telemetry import canonical as telemetry_canonical

    accepted = serial.vectors()
    invalid = serial.invalid_vectors()
    corpus = serial.float_corpus()
    floats = corpus["vector"] + corpus["random"] + corpus["ties"]
    table = []
    telemetry_mismatch, ascii_mismatch, differs_from_spec, encoder_differs = [], [], [], []
    reencode_failures = 0
    for name, value in accepted:
        utf8, ascii_bytes = serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True)
        telemetry, identities = telemetry_canonical(value), canonical_json(value).encode("utf-8")
        if serial.canonical_bytes(json.loads(utf8)) != utf8:
            reencode_failures += 1
        if telemetry != utf8:
            telemetry_mismatch.append(name)
        if identities != ascii_bytes:
            ascii_mismatch.append(name)
        if identities != utf8:
            differs_from_spec.append(name)
        if identities != telemetry:
            encoder_differs.append(name)
        table.append({"name": name, "canonical": utf8.decode("utf-8"), "utf8_hex": utf8.hex(),
                      "sha256": hashlib.sha256(utf8).hexdigest(), "ascii_sha256": hashlib.sha256(ascii_bytes).hexdigest(),
                      "telemetry_sha256": hashlib.sha256(telemetry).hexdigest(),
                      "identities_sha256": hashlib.sha256(identities).hexdigest()})
    float_telemetry = sum(telemetry_canonical(x) != serial.canonical_bytes(x) for x in floats)
    float_identities = sum(canonical_json(x).encode("utf-8") != serial.canonical_bytes(x, ascii_only=True)
                           for x in floats)
    refusals = []
    for name, value, expected in invalid:
        row = {"name": name, "expected": expected}
        for label, encoder in (("specification", serial.canonical_bytes),
                               ("ciw.core.identities.canonical_json", canonical_json),
                               ("ciw.telemetry.canonical", telemetry_canonical)):
            try:
                encoder(value)
                row[label] = "accepted"
            except (ValueError, TypeError, RecursionError) as exc:
                row[label] = getattr(exc, "code", "refused")
        refusals.append(row)
    examples = sum(serial.canonical_bytes(value) != expected for value, expected in serial.SPEC_EXAMPLES)
    collision = {"int_key": canonical_json({1: "x"}), "str_key": canonical_json({"1": "x"})}
    reference = [serial.reference_form(x) for x in floats]
    cpython_mismatch = sum(serial.decimal_form(repr(x)) != form for x, form in zip(floats, reference))
    numpy_mismatch = sum(serial.decimal_form(serial.numpy_shortest(x)) != form for x, form in zip(floats, reference))
    roundtrip_failures = sum(float(serial.format_float(x)) != x
                             or math.copysign(1, float(serial.format_float(x))) != math.copysign(1, x) for x in floats)
    jcs_numbers = sorted({serial.format_float(x): serial.ecmascript_number(x) for x in corpus["vector"]
                          if serial.format_float(x) != serial.ecmascript_number(x)}.items())
    keys = list(dict(accepted)["key-order-code-points"])
    jcs_order, ciw_order = sorted(keys, key=lambda k: k.encode("utf-16-be")), sorted(keys)
    set_digest = serial.canonical_sha256([[row["name"], row["sha256"]] for row in table])
    return {"table": table, "refusals": refusals, "telemetry_mismatch": telemetry_mismatch,
            "ascii_mismatch": ascii_mismatch, "differs_from_spec": differs_from_spec, "encoder_differs": encoder_differs,
            "float_telemetry_mismatch": float_telemetry, "float_identities_mismatch": float_identities,
            "reencode_failures": reencode_failures, "spec_example_mismatches": examples, "collision": collision,
            "floats": floats, "corpus_sizes": {k: len(v) for k, v in corpus.items()}, "ties": corpus["ties"],
            "cpython_mismatch": cpython_mismatch, "numpy_mismatch": numpy_mismatch,
            "roundtrip_failures": roundtrip_failures, "jcs_numbers": jcs_numbers,
            "jcs_key_order": jcs_order, "ciw_key_order": ciw_order, "set_digest": set_digest}


TIE_WITNESS = 1e15 + 0.25


def rust_serialization(study) -> dict:
    """Rust probe on every vector and on every float of the corpus, plus Rust's own shortest formatting on ties."""
    accepted, invalid = serial.vectors(), serial.invalid_vectors()
    results = serial.rust_canonical([v for _, v in accepted] + [v for _, v, _ in invalid] + study["floats"])
    vector_results = results[:len(accepted)]
    refusal_results = results[len(accepted):len(accepted) + len(invalid)]
    float_results = results[len(accepted) + len(invalid):]
    byte_mismatch = [name for (name, value), got in zip(accepted, vector_results)
                     if got != (serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True))]
    float_mismatch = [x for x, got in zip(study["floats"], float_results)
                      if got != (serial.canonical_bytes(x), serial.canonical_bytes(x, ascii_only=True))]
    refusal_mismatch = [name for (name, _, expected), got in zip(invalid, refusal_results) if got != expected]
    ties = study["ties"]
    own = serial.rust_shortest(ties + [TIE_WITNESS])
    shortest_differs = sum(serial.decimal_form(text) != serial.reference_form(x) for text, x in zip(own, ties))
    return {"vectors": len(accepted) + len(invalid), "floats": len(study["floats"]), "decimal_ties": len(ties),
            "byte_mismatches": byte_mismatch, "float_mismatches": [repr(x) for x in float_mismatch],
            "refusal_mismatches": refusal_mismatch,
            "rust_refusals": {name: got for (name, _, _), got in zip(invalid, refusal_results)},
            "shortest_differs_on_ties": shortest_differs, "witness_rust_shortest": own[-1]}


RUST_CANONICAL_CLAIM = "Rust canonical JSON is byte-identical to the specification and refuses the same inputs"
RUST_TIES_CLAIM = "Rust's own shortest float formatting breaks exact decimal ties away from the specification"


@task("T146", changed_files=(MODULE, SERIAL, DOC),
      regression_tests=(f"{TESTS}::test_t146_python_canonicalizers_and_vectors", f"{TESTS}::test_t146_rust_byte_identity",
                        f"{TESTS}::test_t146_reference_float_rule",
                        f"{TESTS}::test_ciw_producers_of_independent_checks_carry_a_revision"))
def canonical_serialization(ctx):
    study = serialization_study()
    build = serial.rust_build()
    status, rust = _rust_status(build, lambda: rust_serialization(study))
    ctx.artifact_json("canonical-json-spec.json", dict(serial.SPEC, examples=[
        {"value_repr": repr(value), "bytes_hex": expected.hex()} for value, expected in serial.SPEC_EXAMPLES]))
    ctx.artifact_json("test-vectors.json", {"spec": serial.SPEC_ID, "vector_set_sha256": study["set_digest"],
                                            "vectors": study["table"], "invalid": study["refusals"]})
    ctx.artifact_text("ciw_targets.rs", serial.RUST_SOURCE)
    ctx.artifact_json("float-corpus.json", {"sizes": study["corpus_sizes"],
                                            "decimal_ties": [[repr(x), serial.format_float(x)] for x in study["ties"]]})
    ctx.artifact_json("cross-language.json", {"rust": rust if status == "ran" else None, "rust_status": status,
                                              "rust_build": {k: v for k, v in build.items() if k != "binary"},
                                              "jcs_number_differences": study["jcs_numbers"],
                                              "jcs_key_order": study["jcs_key_order"], "ciw_key_order": study["ciw_key_order"]})
    n_vectors, n_floats = len(study["table"]), len(study["floats"])
    spec_checks = [_check("hand-written specification examples the reference encoder does not reproduce",
                          study["spec_example_mismatches"], kind="analytic"),
                   _check("re-encoding a decoded vector changes its bytes", study["reencode_failures"])]
    spec_checks += [{"reference_kind": "refusal", "reference": f"specification on {row['name']}",
                     "expected_refusal": row["expected"], "observed_refusal": row["specification"],
                     "passed": row["specification"] == row["expected"]} for row in study["refusals"]]
    lax = {label: [row["name"] for row in study["refusals"] if row[label] == "accepted"]
           for label in ("ciw.core.identities.canonical_json", "ciw.telemetry.canonical")}
    rows = {row["name"]: row for row in study["table"]}
    witness_name = next((n for n in study["encoder_differs"] if n == "unicode-bmp"), None) or \
        (study["encoder_differs"][0] if study["encoder_differs"] else "unicode-bmp")
    witness = rows[witness_name]
    by_construction = ("CIW encoders call json.dumps; the reference encoder escapes from its own table and formats "
                       "floats by an exact digit search, so agreement is not true by construction")
    findings = [
        finding("Canonical JSON v1 test vectors (bytes and sha256)", "provenance",
                {"spec": serial.SPEC_ID, "vectors": n_vectors, "invalid_vectors": len(study["refusals"]),
                 "vector_set_sha256": study["set_digest"]},
                {"checks": spec_checks}, uncertainty=EXACT_U, tolerance=EXACT),
        finding("ciw.telemetry.canonical reproduces the specification bytes on every accepted vector and tested float",
                "computational_pipeline",
                {"vectors": n_vectors, "vector_mismatches": len(study["telemetry_mismatch"]), "floats": n_floats,
                 "float_mismatches": study["float_telemetry_mismatch"]},
                {"checks": [_check(f"ciw.telemetry.canonical against the reference encoder, vectors ({by_construction})",
                                   len(study["telemetry_mismatch"]), kind="cross_implementation"),
                            _check("ciw.telemetry.canonical against the reference encoder, single floats",
                                   study["float_telemetry_mismatch"], kind="cross_implementation")]},
                unit="mismatches", uncertainty=EXACT_U, tolerance=EXACT),
        finding("ciw.core.identities.canonical_json reproduces the ASCII-escaped variant, not the specification bytes",
                "computational_pipeline",
                {"ascii_variant_mismatches": len(study["ascii_mismatch"]),
                 "float_ascii_variant_mismatches": study["float_identities_mismatch"],
                 "differs_from_specification": study["differs_from_spec"]},
                {"checks": [_check("ciw.core.identities.canonical_json against the reference ASCII variant, vectors",
                                   len(study["ascii_mismatch"]), kind="cross_implementation"),
                            _check("ciw.core.identities.canonical_json against the reference ASCII variant, floats",
                                   study["float_identities_mismatch"], kind="cross_implementation"),
                            _check("vectors whose ciw.core.identities bytes differ from the specification bytes",
                                   len(study["differs_from_spec"]), 1, "ge")]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for non-ASCII text",
                "computational_pipeline", {"differing_vectors": study["encoder_differs"]},
                {"checks": [_check("vectors on which the two CIW encoders differ", len(study["encoder_differs"]), 1, "ge"),
                            _check("differing vectors that are printable ASCII only",
                                   sum(all(0x20 <= ord(ch) <= 0x7e for ch in rows[n]["canonical"])
                                       for n in study["encoder_differs"]))]},
                counterexample={"statement": "CIW already has one canonical JSON byte encoding",
                                "witness": {"vector": witness_name, "telemetry_sha256": witness["telemetry_sha256"],
                                            "identities_sha256": witness["identities_sha256"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("ciw.core.identities.canonical_json gives {1: 'x'} and {'1': 'x'} the same content identity",
                "computational_pipeline", hashlib.sha256(study["collision"]["int_key"].encode()).hexdigest(),
                {"checks": [_check("int-key and str-key canonical texts are equal",
                                   float(study["collision"]["int_key"] == study["collision"]["str_key"]), 1, "ge"),
                            _refusal("specification on {1: 'x'}", "non_string_key",
                                     lambda: serial.canonical_bytes({1: "x"}))]},
                counterexample={"statement": "Distinct Python values have distinct CIW content identities",
                                "witness": {"values": ["{1: 'x'}", "{'1': 'x'}"], "canonical": study["collision"]["str_key"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Python canonicalizers accept values the specification refuses", "computational_pipeline", lax,
                {"checks": [_check("refused-by-spec vectors accepted by a CIW encoder",
                                   sum(len(v) for v in lax.values()), 1, "ge")]},
                counterexample={"statement": "Existing CIW canonicalizers enforce the cross-language specification",
                                "witness": {"vector": "integer-2^53", "accepted_by": sorted(
                                    k for k, v in lax.items() if "integer-2^53" in v)}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("The exact shortest-digit rule agrees with CPython repr and NumPy Dragon4 on vector, random and "
                "decimal-tie binary64 values", "numerical",
                {"values": n_floats, "decimal_ties": len(study["ties"]), "cpython_mismatches": study["cpython_mismatch"],
                 "numpy_mismatches": study["numpy_mismatch"]},
                {"checks": [_check("round-trip failures of the specification text (value or sign)",
                                   study["roundtrip_failures"]),
                            _check("CPython repr digit/exponent mismatches against the exact rule",
                                   study["cpython_mismatch"]),
                            _check("constructed decimal ties in the corpus", len(study["ties"]), 1, "ge")],
                 "independent_check": dict(_check("NumPy digit/exponent mismatches against the exact rule",
                                                  study["numpy_mismatch"]),
                                           producer=_ciw_producer("ciw.lab.implementation_targets_serial."
                                                                  "shortest_digits", SERIAL),
                                           checker={"implementation": "numpy.format_float_scientific(unique=True)",
                                                    "revision": np.__version__})},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("CIW canonical JSON differs from RFC 8785 (JCS) numbers and key order", "computational_pipeline",
                {"number_differences": len(study["jcs_numbers"]),
                 "key_order_differs": study["jcs_key_order"] != study["ciw_key_order"]},
                {"derivation": "ECMA-262 Number::toString and UTF-16 code-unit key order (RFC 8785 section 3.2)",
                 "checks": [_check("vector floats formatted differently", len(study["jcs_numbers"]), 1, "ge"),
                            _check("key order differs (UTF-16 units against code points)",
                                   float(study["jcs_key_order"] != study["ciw_key_order"]), 1, "ge")]},
                counterexample={"statement": "CIW canonical JSON bytes equal RFC 8785 JCS bytes",
                                "witness": {"1.0": ["1.0", "1"], "1e+16": ["1e+16", "10000000000000000"],
                                            "keys": ["\uff21", "\U0001f600"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
    ]
    if status == "ran":
        findings.append(finding(
            RUST_CANONICAL_CLAIM, "computational_pipeline",
            {"vectors": rust["vectors"], "floats": rust["floats"], "decimal_ties": rust["decimal_ties"],
             "byte_mismatches": len(rust["byte_mismatches"]), "float_mismatches": len(rust["float_mismatches"]),
             "refusal_mismatches": len(rust["refusal_mismatches"])},
            {"checks": [_check("Rust byte mismatches on vectors (UTF-8 and ASCII forms)", len(rust["byte_mismatches"]),
                               kind="cross_implementation"),
                        _check("Rust byte mismatches on single floats, including decimal ties",
                               len(rust["float_mismatches"]), kind="cross_implementation"),
                        _check("Rust refusal-code mismatches", len(rust["refusal_mismatches"]),
                               kind="cross_implementation")]},
            uncertainty=EXACT_U, tolerance=EXACT))
        findings.append(finding(
            RUST_TIES_CLAIM, "computational_pipeline",
            {"decimal_ties": rust["decimal_ties"], "rust_shortest_differs": rust["shortest_differs_on_ties"]},
            {"checks": [_check("decimal ties on which Rust's format!(\"{:e}\") digits differ from the specification",
                               rust["shortest_differs_on_ties"], 1, "ge")]},
            counterexample={"statement": "Rust's shortest float formatting yields the CPython repr digits",
                            "witness": {"value": "1e15 + 0.25", "specification": serial.format_float(TIE_WITNESS),
                                        "rust_shortest": rust["witness_rust_shortest"]}},
            uncertainty=EXACT_U, tolerance=EXACT))
    else:
        findings.append(_rust_not_run(RUST_CANONICAL_CLAIM, "computational_pipeline", status, rust))
        findings.append(_rust_not_run(RUST_TIES_CLAIM, "computational_pipeline", status, rust))
    findings.append(finding("Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations",
                            "computational_pipeline", None, {}, expected_not_established=True))
    fields = _fields(
        "One canonical JSON encoding (sorted keys, no whitespace, UTF-8, shortest round-trip binary64 with decimal ties "
        "to even in CPython repr layout, safe integers, refusals) is reproducible byte for byte by a separately "
        "written implementation in another language (Rust), and CIW's existing Python encoders can be measured "
        "against it.",
        "Encoding E: JSON values -> bytes per ciw.canonical-json.v1; identity = sha256(E(v)). Floats: the fewest "
        "digits k for which a k-digit decimal round-trips (monotone in k, found by bisection with exact integers); "
        "the nearer of the two k-digit neighbours, even digit on a tie; fixed form iff -4 < decpt <= 16.",
        [f"{n_vectors} accepted and {len(study['refusals'])} refused vectors (binary64 limits, subnormals, -0.0, "
         "Unicode, controls, nested keys, depth 64/65, decimal ties)",
         f"{n_floats} binary64 values: every vector float, 2000 random bit patterns (PCG64(1460)) and "
         f"{len(study['ties'])} constructed decimal ties (PCG64(1461))", "5 hand-written specification examples"],
        "Bytes and sha256 digests of each encoder's output; refusal codes; digit strings.",
        "The reference encoder reproduces the hand-written examples; ciw.telemetry.canonical and the Rust probe "
        "agree with it on UTF-8 bytes; ciw.core.identities agrees with the ASCII variant only; refused inputs are "
        "refused by the reference and Rust; CPython and NumPy digits equal the exact rule.",
        "Encode every vector and float with the reference encoder, both CIW encoders and the Rust probe compiled at "
        "run time; compare digits with CPython repr and NumPy Dragon4; run Rust's own {:e} on decimal ties; contrast "
        "with JCS formatting.",
        "Deferred research question: migrate ciw.core.identities.canonical_json to ciw.canonical-json.v1 with a "
        "versioned hash change, so old and new identities coexist during the migration (every retained identity "
        "would change), and run a Julia or C++ encoder against the vectors",
        numerical_result=(f"{n_vectors} vectors, set digest {study['set_digest'][:16]}...; telemetry mismatches "
                          f"{len(study['telemetry_mismatch'])} (vectors) and {study['float_telemetry_mismatch']} "
                          f"(floats); identities differs from the specification on {len(study['differs_from_spec'])} "
                          f"vectors and from the ASCII variant on {len(study['ascii_mismatch'])}; the two CIW encoders "
                          f"differ on {len(study['encoder_differs'])} vectors; Rust "
                          + (f"byte mismatches {len(rust['byte_mismatches'])} (vectors) and "
                             f"{len(rust['float_mismatches'])}/{rust['floats']} (floats), refusal mismatches "
                             f"{len(rust['refusal_mismatches'])}, own {{:e}} differs on "
                             f"{rust['shortest_differs_on_ties']}/{rust['decimal_ties']} ties"
                             if status == "ran" else f"not run ({rust})")
                          + f"; CPython/NumPy digit mismatches {study['cpython_mismatch']}/{study['numpy_mismatch']} "
                            f"of {n_floats}."),
        uncertainty=("Exact byte comparison. The reference encoder, the Rust probe and the CIW encoders are all "
                     "CIW-authored, so their agreement is same-origin; digit agreement with NumPy is independent but "
                     "covers only the tested values (random bit patterns contain almost no decimal ties, hence the "
                     "constructed ones)."),
        failure_modes_checked=["NaN/Infinity", "unsafe integers", "non-string keys (identity collision)",
                               "lone surrogates", "nesting depth", "ensure_ascii divergence", "U+007F and U+2028",
                               "Unicode normalization", "-0.0", "subnormals and binary64 limits",
                               "exponent-switch boundaries", "decimal ties (Rust {:e} rounds them up)",
                               "reference encoder true by construction", "JCS divergence",
                               "rustc absent (not established) versus probe broken (refuted)"],
        unresolved_assumptions=["ciw.core.identities.canonical_json remains the ASCII variant because changing it "
                                "would change every retained identity", "Julia, C++ and GPU-host encoders were not run",
                                "JCS formatting was implemented here from ECMA-262, not from an external JCS library"],
        provider_runtime_identity=_runtime_identity((MODULE, SERIAL), build))
    return {"state": _state(findings, status == "ran"), "fields": fields, "findings": findings}


# =================================================================== T147
COMMON = "src/ciw/lab/energy_gpu_workload.py"
GPU_CLAIM = ("CPU and GPU outputs of the common Gaussian VI workload agree under T148's policy for fixed-order "
             "reductions (bitwise) on GPU hardware")
HARNESS_CLAIM = ("Under T148's fixed-order policy the harness accepts a second implementation of the common Gaussian VI "
                 "workload and flags builds that contract its multiply-adds")


def _common_workload_harness():
    """The common workload's float64 reference and the candidates the fixed-order (bitwise) policy must judge."""
    from . import energy_gpu_workload as common
    reference = common.run_numpy(np.float64, replicas=1)
    candidates = {"python-scalar": common.run_variant("kernel")}
    candidates.update({variant: common.run_variant(variant) for variant in common.VARIANTS if variant.startswith("fma-")})
    results = {name: kernels.compare_outputs(reference, candidate.reshape(reference.shape), FIXED_ORDER_POLICY)
               for name, candidate in candidates.items()}
    return {"iterations": common.plan_iterations(), "reference": reference[0].tolist(),
            "results": {name: {k: v for k, v in result.items() if k not in ("ratios", "violating")}
                        for name, result in results.items()}}


def _gpu_comparison_finding(ctx):
    """Compare the PTX kernel's outputs with the NumPy reference under the fixed-order policy, where a GPU answers."""
    from . import energy_gpu_workload as common
    gpu = common.gpu_run(ctx)
    if not gpu["ran"]:
        return finding(GPU_CLAIM, "numerical", None, {"notes": [gpu["reason"]]}, expected_not_established=True), gpu, None
    size = int(np.asarray(gpu["reference"]).size)
    if gpu.get("invalid"):
        # The kernel ran and the worker rejected its outputs: a failed comparison, never an expected gap.
        value = {"elements": size, "rejected_by_worker": gpu["invalid"], "policy": FIXED_ORDER_POLICY["mode"]}
        checks = [_check("GPU outputs rejected by the CUDA worker's own output validation (1 = yes)", 1.0,
                         kind="cross_implementation")]
    else:
        try:
            result = kernels.compare_outputs(gpu["reference"], gpu["outputs"], FIXED_ORDER_POLICY)
        except ValueError as exc:  # another shape or a nonfinite output: every value counts as a violation
            result = {"elements": size, "violations": size, "max_ulp": float(2 ** 62), "refusal": str(exc)}
        value = {"elements": result["elements"], "violations": result["violations"], "max_ulp": result["max_ulp"],
                 "policy": FIXED_ORDER_POLICY["mode"]}
        checks = [_check("compare_outputs violations under the fixed-order (bitwise) policy, every replica",
                         result["violations"], kind="cross_implementation")]
    checks.append(_check("GPU worker prepared-input digest differs from the reference's (1 = yes)",
                         0.0 if gpu["input_matches"] else 1.0))
    record = finding(
        GPU_CLAIM, "numerical", value,
        {"generator": {"name": "common Gaussian VI workload", "seed": None, "iterations": common.plan_iterations(),
                       "replicas": common.REPLICAS},
         "notes": ["the PTX kernel (ciw.energy_cuda) and the NumPy reference (ciw.lab.energy_gpu_workload) are both "
                   "ciw code: agreement is cross-implementation evidence, not independent verification"],
         "checks": checks},
        uncertainty=EXACT_U, tolerance=EXACT)
    identity = {key: gpu["identity"].get(key) for key in ("device_name", "device_uuid", "compute_capability",
                                                          "cuda_driver_version", "kernel_sha256")}
    return record, gpu, identity


@task("T147", changed_files=(MODULE, KERNELS, COMMON, DOC),
      regression_tests=(f"{TESTS}::test_t147_harness_detects_differences",
                        f"{TESTS}::test_t147_fault_study_is_judged_by_the_harness",
                        f"{TESTS}::test_t147_common_workload_comparison_follows_the_gpu_probe",
                        f"{TESTS}::test_t147_identity_digests_the_common_workload_sources"))
def cpu_gpu_comparison(ctx):
    from . import energy_gpu_workload as workload
    from .energy_gpu import GPU_PROTOCOL
    from .energy_gpu_workload import GPU_QUESTION
    study = ctx.memo("t147-dots", kernels.batched_dot_study)
    harness = ctx.memo("t147-common-workload", _common_workload_harness)
    gpu_finding, gpu, gpu_identity = _gpu_comparison_finding(ctx)
    out, bounds, exact = study["outputs"], study["bounds"], study["exact"]
    reference = out["f64-sequential"]
    f64_tolerance = bounds["f64-sequential"] + bounds["f64-blocked"]
    f32_tolerance = bounds["f64-sequential"] + bounds["f32-blocked"]
    f64_policy, f32_policy = {"mode": "bound", "tolerance": f64_tolerance}, {"mode": "bound", "tolerance": f32_tolerance}
    bitwise = kernels.compare_outputs(reference, out["f64-blocked"], {"mode": "bitwise"})
    reorder = kernels.compare_outputs(reference, out["f64-blocked"], f64_policy)
    pairwise = kernels.compare_outputs(reference, out["f64-pairwise"],
                                       {"mode": "bound", "tolerance": bounds["f64-sequential"] + bounds["f64-pairwise"]})
    f32_under_f64 = kernels.compare_outputs(reference, out["f32-blocked"], f64_policy)
    f32_under_f32 = kernels.compare_outputs(reference, out["f32-blocked"], f32_policy)
    against_exact = {name: float(np.max(np.abs(value - exact) / bounds[name])) for name, value in out.items()}
    self_bitwise = kernels.compare_outputs(reference, reference.copy(), {"mode": "bitwise"})
    # Fault model: one partial product a_ij x_j dropped from a candidate row, at every (row, column), judged by the
    # harness. The float32 products use the float32-rounded vector, so its entries are the column scale.
    products64 = study["A"] * study["x"]
    x32 = study["x"].astype(np.float32)
    products32 = (study["A"].astype(np.float32) * x32).astype(np.float64)
    power64 = kernels.dropped_product_study(reference, out["f64-blocked"], products64, f64_tolerance, study["x"])
    power32 = kernels.dropped_product_study(reference, out["f32-blocked"], products32, f32_tolerance,
                                            x32.astype(np.float64))
    ulp = kernels.ulp_distance(reference, out["f64-blocked"])
    summary = {name: {k: v for k, v in result.items() if k not in ("ratios", "violating")} for name, result in (
        ("bitwise f64 sequential vs blocked", bitwise), ("bitwise reference vs itself", self_bitwise),
        ("bound f64 sequential vs blocked", reorder), ("bound f64 sequential vs pairwise", pairwise),
        ("f32 blocked under f64 policy", f32_under_f64), ("f32 blocked under f32 policy", f32_under_f32))}
    ctx.artifact_json("comparison-summary.json", {"comparisons": summary, "against_exact_max_ratio": against_exact,
                                                  "dropped_product_study": {"float64 policy": power64,
                                                                            "float32 policy": power32},
                                                  "depths": study["depths"], "max_ulp_f64_reorder": float(np.max(ulp)),
                                                  "rows": study["rows"], "n": study["n"], "lane_width": study["width"],
                                                  "common_workload": dict(harness, policy=FIXED_ORDER_POLICY,
                                                                          gpu={"probed": gpu["probed"], "ran": gpu["ran"],
                                                                               "reason": gpu["reason"],
                                                                               "invalid": gpu["invalid"],
                                                                               "identity": gpu_identity})})
    series = []
    for name, result in (("f64 reorder / f64 bound", reorder), ("f32 / f64 bound", f32_under_f64),
                         ("f32 / f32 bound", f32_under_f32)):
        ratios = np.sort(result["ratios"])
        series.append((name, list(range(1, len(ratios) + 1)), [float(v) for v in ratios]))
    ctx.artifact_text("difference-over-bound.svg", svg.line_plot(
        series, title="CPU-vs-CPU differences relative to the tolerance policy", xlabel="row (sorted)",
        ylabel="|difference| / policy bound", logy=True, markers=False))
    missed, above32 = power32["largest_undetected"], power32["above_bound_witness"]
    guarantee_keys = ("faults", "fault_free_violations", "above_twice_tolerance", "undetected_above_twice_tolerance",
                      "undetected")
    guarantee_checks = []
    for label, power in (("float64", power64), ("float32", power32)):
        guarantee_checks += [
            _check(f"{label} policy: fault-free candidate rows violating the policy (the guarantee's premise)",
                   power["fault_free_violations"], kind="analytic"),
            _check(f"{label} policy: dropped products larger than twice the row tolerance that compare_outputs "
                   "leaves undetected", power["undetected_above_twice_tolerance"], kind="analytic"),
            _check(f"{label} policy: dropped products larger than twice the row tolerance (the guarantee is not "
                   "vacuous)", power["above_twice_tolerance"], 1, "ge")]
    common_results = harness["results"]
    fused = [name for name in common_results if name.startswith("fma-")]
    findings = [
        gpu_finding,
        finding(HARNESS_CLAIM, "numerical", {name: bool(row["violations"]) for name, row in common_results.items()},
                {"generator": {"name": "common Gaussian VI workload", "seed": None, "iterations": harness["iterations"],
                               "replicas": 1},
                 "notes": ["candidates: the scalar Python-float evaluation of the kernel's declared order, and the two "
                           "exact fused multiply-add contraction rules of ciw.lab.energy_gpu_workload.VARIANTS"],
                 "checks": [_check("elements the fixed-order policy flags for the scalar evaluation of the declared "
                                   "order", common_results["python-scalar"]["violations"], kind="cross_implementation")]
                           + [_check(f"elements the fixed-order policy flags for the {name} contraction",
                                     common_results[name]["violations"], 1, "ge") for name in fused]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Bitwise policy detects reduction-order differences between float64 CPU orders", "numerical",
                bitwise["violations"],
                {"generator": {"name": "PCG64 128x1024 uniform batched dot products", "seed": 147},
                 "checks": [_check("rows the bitwise policy flags (sequential vs 32-lane blocked)", bitwise["violations"],
                                   1, "ge"),
                            _check("rows the bitwise policy flags when the reference is compared with a copy of itself",
                                   self_bitwise["violations"])]},
                unit="rows of 128", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Float64 reduction-order differences lie within the analytic error-bound policy", "numerical",
                max(reorder["max_ratio"], pairwise["max_ratio"]),
                {"checks": [_check("max |difference|/bound, sequential vs blocked and pairwise",
                                   max(reorder["max_ratio"], pairwise["max_ratio"]), 1.0, "le", kind="analytic"),
                            _check("max |output - exact|/bound over all orders", max(against_exact.values()), 1.0, "le",
                                   kind="analytic")]},
                unit="fraction of the policy bound",
                uncertainty={"kind": "truncation_bound", "value": 1.0,
                             "basis": "worst-case bound gamma_k sum|a_j x_j| per output; the ratio is at most 1 by "
                                      "construction of the bound"},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Float32 results violate the float64 policy and satisfy the float32 policy", "numerical",
                {"violations_under_f64_policy": f32_under_f64["violations"], "max_ratio_under_f32_policy":
                 f32_under_f32["max_ratio"]},
                {"checks": [_check("rows violating the float64 policy", f32_under_f64["violations"], 1, "ge"),
                            _check("max |difference|/bound under the float32 policy", f32_under_f32["max_ratio"], 1.0,
                                   "le", kind="analytic")]},
                uncertainty={"kind": "truncation_bound", "value": 1.0,
                             "basis": "float32 bound with inputs rounded to float32 (depth + 2 roundings)"},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("The harness detects every dropped partial product larger than twice the row tolerance under both "
                "policies", "numerical",
                {"float64": {k: power64[k] for k in guarantee_keys}, "float32": {k: power32[k] for k in guarantee_keys}},
                {"checks": guarantee_checks + [
                    _check("float64 policy: undetected single dropped products (all 131072)", power64["undetected"])]},
                unit="faults", uncertainty=EXACT_U, tolerance=EXACT),
        finding("The float32 policy misses dropped partial products up to about its bound, as often as the product "
                "distribution predicts", "numerical",
                {"undetected": power32["undetected"], "faults": power32["faults"],
                 "expected_undetected": power32["expected_undetected"], "sigma": power32["sigma_undetected"],
                 "band_half_width": power32["fault_free_max_ratio"],
                 "detected_below_band": power32["detected_below_band"],
                 "undetected_above_band": power32["undetected_above_band"],
                 "undetected_above_bound": power32["undetected_above_bound"],
                 "max_undetected_ratio": power32["max_undetected_ratio"], "largest_undetected": missed,
                 "above_bound_witness": above32},
                {"checks": [_check("float32 policy: undetected single dropped products", power32["undetected"], 1, "ge"),
                            _check("float32 policy: undetected dropped products larger than the row bound",
                                   power32["undetected_above_bound"], 1, "ge"),
                            _check("float32 policy: faults on the wrong side of the band (1 +/- m) tol, m = largest "
                                   "fault-free |difference|/bound reported by the harness (detected below it or "
                                   "missed above it)",
                                   power32["detected_below_band"] + power32["undetected_above_band"], kind="analytic"),
                            _check("float32 policy: (undetected - sum_ij min(1, tol_i/|x_j|)) / sigma, the miss count "
                                   "predicted for a_ij uniform on [-1, 1)", power32["undetected_z"], 4.0,
                                   kind="analytic")]},
                counterexample={"statement": "The float32 tolerance policy detects every dropped partial product larger "
                                             "than its row bound",
                                "witness": {"largest_undetected": missed, "above_bound": above32}},
                uncertainty={"kind": "reference_error", "value": power32["sigma_undetected"],
                             "basis": "standard deviation of the predicted miss count (sum of Bernoulli variances over "
                                      "the faults); the observed counts are exact for the seed"},
                tolerance={"abs": 0, "rel": 1e-9}),
        finding("GPU/CPU agreement establishes industrial readiness", "industrial_readiness", None, {}),
    ]
    fields = _fields(
        "A comparison harness with an analytic tolerance policy separates legitimate reduction-order and precision "
        "differences from faults: once the fault-free candidate is within the policy, every fault larger than twice "
        "the tolerance is detected, detection switches at about the tolerance, and faults up to about the bound "
        "escape as often as the fault-size distribution predicts. On the common Gaussian VI workload, whose "
        "reductions have a fixed order, T148's policy prescribes a bitwise comparison: it accepts a second "
        "implementation of the declared order and flags a build that contracts multiply-adds, and on a GPU host "
        "it compares the gaussian_vi PTX kernel's outputs with the CPU reference.",
        "For a sum of terms each passing through k roundings, |computed - exact| <= gamma_k * sum|a_j x_j| with "
        "gamma_k = k u/(1 - k u), u = 2^-53 (float64) or 2^-24 (float32, inputs rounded). The policy tolerance for two "
        "outputs is the sum of their bounds, so the fault-free candidate satisfies |c - r| <= tol; bitwise mode "
        "compares bit patterns. A dropped term p gives |c - p - r| >= |p| - tol, so |p| > 2 tol is detected without "
        "knowing c - r; with m = max |c - r|/tol, detection switches inside (1 +/- m) tol. With a_ij uniform on "
        "[-1, 1), P(|a_ij x_j| <= tol_i) = min(1, tol_i/|x_j|), whose sum over faults predicts the miss count.",
        ["128x1024 uniform[-1,1) matrix and vector, PCG64(147)", "orders: sequential, pairwise tree, 32-lane blocked "
         "(sequential lanes then tree); precisions float64 and float32", "exact dot products by TwoProduct + math.fsum",
         "all 131072 single dropped products per policy, judged by compare_outputs",
         f"the common Gaussian VI workload (examples/energy-accuracy/problem.json, K = {harness['iterations']}): the "
         "float64 NumPy reference, a scalar evaluation of the same order, two exact fused multiply-add contractions, "
         "and on a GPU host the gaussian_vi PTX kernel's outputs for every replica"],
        "Elementwise absolute differences, bitwise equality and ratios to the policy bound, as reported by "
        "compare_outputs for the fault-free candidates and for every faulty candidate.",
        "Bitwise policy flags reordering and not a copy of the reference; bound policy accepts reordering; float32 "
        "fails the float64 policy and passes its own; no fault above twice the tolerance escapes; no fault is judged "
        "on the wrong side of the (1 +/- m) band; the float32 miss count lies within 4 sigma of its prediction.",
        "Compute the dot products in four orders/precisions, compare with the harness under each policy, drop each "
        "partial product in turn from the float64 and float32 candidates and judge every faulty candidate with the "
        "harness, then compare detection with the operational guarantee, the threshold band and the predicted miss "
        "count. Judge the common workload's candidates under the fixed-order policy, and when an NVIDIA GPU answers "
        "the probe run the PTX kernel and judge its outputs the same way.",
        "Judge the PTX kernel's outputs for every replica under the fixed-order policy on the RTX 2080 host (protocol "
        "in docs/lab/ENERGY_GPU.md): " + GPU_PROTOCOL + ". " + GPU_QUESTION,
        numerical_result=(f"{bitwise['violations']}/128 rows flagged by the bitwise policy between float64 orders (0 for "
                          f"a copy of the reference); max difference/bound "
                          f"{max(reorder['max_ratio'], pairwise['max_ratio']):.3g}; float32 violates the float64 policy "
                          f"on {f32_under_f64['violations']} rows and stays at {f32_under_f32['max_ratio']:.3g} of its "
                          f"own bound; faults above twice the tolerance left undetected: "
                          f"{power64['undetected_above_twice_tolerance']}/{power64['above_twice_tolerance']} (float64 "
                          f"policy), {power32['undetected_above_twice_tolerance']}/{power32['above_twice_tolerance']} "
                          f"(float32 policy); single dropped products undetected: {power64['undetected']}/"
                          f"{power64['faults']} (float64), {power32['undetected']}/{power32['faults']} (float32, "
                          f"predicted {power32['expected_undetected']:.1f} +/- {power32['sigma_undetected']:.1f}; largest "
                          f"missed |a_j x_j| {missed['magnitude'] if missed else 0:.3g} against row tolerance "
                          f"{missed['row_tolerance'] if missed else 0:.3g}; {power32['undetected_above_bound']} missed "
                          f"above the row bound{', ratio %.6g' % above32['ratio'] if above32 else ''}). Common workload "
                          f"under the fixed-order policy: flagged {sorted(n for n, r in common_results.items() if r['violations'])}"
                          f", accepted {sorted(n for n, r in common_results.items() if not r['violations'])}; "
                          + (f"GPU comparison not run: {gpu['reason']}." if not gpu["ran"] else
                             f"GPU comparison refuted: {gpu['invalid']}." if gpu["invalid"] else
                             f"GPU outputs violate it on {gpu_finding['value']['violations']} of "
                             f"{gpu_finding['value']['elements']} values.")),
        uncertainty="Bounds are worst-case (not probabilistic), so typical ratios are far below one; the twice-tolerance "
                    "guarantee needs only the policy's own premise (fault-free candidate within tolerance), while "
                    "faults up to about the bound escape at the rate the fault-size distribution predicts (4-sigma "
                    "check); on the common workload the bitwise policy is exact, and the emulated contractions are two "
                    "plausible compiler rules, not those of a specific nvcc version.",
        failure_modes_checked=["reduction reordering", "precision reduction (float32)",
                               "every single dropped partial product, judged by the harness",
                               "bitwise mode flagging identical outputs", "detection threshold off the bound",
                               "miss count away from its prediction", "shape mismatch and nonfinite outputs "
                               "(refused by the harness)", "fused multiply-add contraction of the common workload "
                               "(flagged by the fixed-order policy)", "GPU outputs differing from the reference (a "
                               "refuted finding; regression test with a simulated contracting device)",
                               "GPU outputs the CUDA worker rejects after the kernel ran (a refuted finding, not an "
                               "expected gap; regression test with a simulated worker)"],
        unresolved_assumptions=["The batched dot products have no GPU path and need none: the GPU comparison uses the "
                                "common Gaussian VI workload, whose PTX kernel exists; a device-wide reduction kernel "
                                "does not", GPU_QUESTION,
                                "GPU reductions may use FMA and tree shapes not modelled by the 32-lane order",
                                "Only single dropped products were injected; other fault classes (duplicated terms, "
                                "wrong operands) have their own detection limits"]
        + ([] if gpu["ran"] else [gpu["reason"]]),
        # The workload's defining modules (PTX text and preparation, planner, information system) are digested
        # beside this task's own sources, with the workload declaration they produce.
        provider_runtime_identity=dict(_runtime_identity((MODULE, KERNELS, COMMON) + workload.SOURCES),
                                       common_workload=workload.workload(),
                                       **({"gpu": gpu_identity} if gpu_identity else {})))
    return {"state": _state(findings, gpu["ran"]), "fields": fields, "findings": findings}


# =================================================================== T148
REDUCTION_POLICY = {
    "schema": "ciw.reduction-policy.v1",
    "identity_bearing_sums": "exact: integer accumulation at scale 2^1074 with one final rounding (bitwise "
                             "order-independent; cross-checked with math.fsum)",
    "fixed_layout_arrays": "pairwise: fixed binary tree split at n//2 over the stored order (bitwise reproducible "
                           "only for the same order and length); error <= gamma_{ceil(log2 n)} sum|x|",
    "streaming_accumulators": "neumaier: compensated, error <= u|S| + gamma_{n-1}^2 sum|x| = u|S| + O(n^2 u^2) sum|x| "
                              "(Ogita, Rump and Oishi 2005, Prop. 4.5); not order-invariant",
    "forbidden_for_identity": ["sequential float accumulation in unspecified order", "BLAS/numpy.sum (order is "
                               "implementation-defined)", "Kahan without the Neumaier branch (fails on large "
                               "cancelling terms)"],
}
# The compare_outputs policy REDUCTION_POLICY prescribes for a fixed-order reduction evaluated in the same order on both
# sides (fixed_layout_arrays: bitwise reproducible for the same order and length), as T147 and T121 apply it to the
# common Gaussian VI workload, whose reductions are fixed two-term sums and must not be contracted.
FIXED_ORDER_POLICY = {"mode": "bitwise", "rule": "REDUCTION_POLICY.fixed_layout_arrays, same order on both sides"}


def _reduction_scaling() -> dict:
    rng = np.random.Generator(np.random.PCG64(1481))
    sizes = [2 ** k for k in range(4, 13)]
    data = [float(v) for v in rng.uniform(-1.0, 1.0, sizes[-1])]
    out = {name: [] for name in kernels.REDUCTIONS}
    for n in sizes:
        xs = data[:n]
        exact = kernels.exact_fraction(xs)
        for name, reduce in kernels.REDUCTIONS.items():
            out[name].append(float(abs(Fraction(reduce(xs)) - exact)))
    return {"sizes": sizes, "abs_errors": out}


def _superseded_bound_witness(xs) -> dict:
    """Neumaier's error on the stored order against the earlier bound 2u|S| + 4 n u^2 sum|x|."""
    exact = kernels.exact_fraction(xs)
    abs_sum = float(kernels.exact_fraction([abs(x) for x in xs]))
    error = abs(Fraction(kernels.sum_neumaier(xs)) - exact)
    old = kernels.superseded_neumaier_bound(len(xs), abs_sum, float(exact))
    new = kernels.error_bound("neumaier", len(xs), abs_sum, float(exact))
    return {"n": len(xs), "error": float(error), "superseded_bound": old, "bound": new,
            "ratio_to_superseded": float(error / Fraction(old)), "ratio_to_bound": float(error / Fraction(new))}


@task("T148", changed_files=(MODULE, KERNELS, DOC),
      regression_tests=(f"{TESTS}::test_t148_reduction_policies",
                        f"{TESTS}::test_ciw_producers_of_independent_checks_carry_a_revision",
                        f"{TESTS}::test_next_steps_point_at_work_that_delivers"))
def reduction_policies(ctx):
    datasets = kernels.reduction_datasets()
    study = kernels.permutation_study(datasets)
    scaling = _reduction_scaling()
    old_bound = _superseded_bound_witness(datasets["absorbed-tiny-terms"])
    fsum_mismatch = sum(a != b for data in study.values()
                        for a, b in zip(data["algorithms"]["exact"]["results"], data["algorithms"]["fsum"]["results"]))
    exact_distinct = max(data["algorithms"]["exact"]["distinct_results"] for data in study.values())
    pairwise_distinct = study["uniform"]["algorithms"]["pairwise"]["distinct_results"]
    # The same fixed tree built a second way (NumPy level-wise additions) over every sampled order of the uniform
    # data: the result is a function of the order alone, not of the implementation that walks the tree.
    pairwise_orders = study["uniform"]["orders_data"]
    tree_mismatches = sum(kernels.sum_pairwise(order) != kernels.sum_pairwise_levels(order) for order in pairwise_orders)
    first_order_sum = study["uniform"]["algorithms"]["pairwise"]["first_order_result"]
    kahan_case = study["kahan-counterexample"]["algorithms"]
    worst = {alg: max(study[d]["algorithms"][alg]["max_error_over_bound"] for d in study)
             for alg in kernels.REDUCTIONS}
    table = {d: {alg: {k: v for k, v in study[d]["algorithms"][alg].items() if k != "results"}
                 for alg in kernels.REDUCTIONS} for d in study}
    ctx.artifact_json("permutation-study.json", {"datasets": {d: {"n": s["n"], "exact": s["exact"], "abs_sum": s["abs_sum"],
                                                                  "orders": s["orders"]} for d, s in study.items()},
                                                 "algorithms": table, "superseded_neumaier_bound": old_bound})
    ctx.artifact_json("reduction-policy.json", REDUCTION_POLICY)
    ctx.artifact_text("error-scaling.svg", svg.line_plot(
        [(name, scaling["sizes"], errors) for name, errors in scaling["abs_errors"].items() if any(errors)]
        + [("sequential bound, sum|x| ~ n/2", scaling["sizes"], [kernels.gamma(n - 1) * n * 0.5 for n in scaling["sizes"]])],
        title="Absolute summation error, uniform[-1,1)", xlabel="n", ylabel="|computed - exact|", logx=True, logy=True))
    distinct_table = {d: {alg: study[d]["algorithms"][alg]["distinct_results"] for alg in kernels.REDUCTIONS}
                      for d in study}
    rigorous = sorted(alg for alg in worst if alg != "kahan")
    findings = [
        finding("Correctly rounded exact accumulation is permutation-invariant", "numerical", exact_distinct,
                {"generator": {"name": "PCG64 datasets and 24 permutations", "seed": 148},
                 "checks": [_check("distinct results across permutations minus one", exact_distinct - 1)],
                 "independent_check": dict(_check("mismatches against math.fsum over every order and dataset",
                                                  fsum_mismatch),
                                           producer=_ciw_producer("ciw.lab.implementation_targets_kernels.sum_exact",
                                                                  KERNELS),
                                           checker={"implementation": "cpython.math.fsum",
                                                    "revision": sys.version.split()[0]})},
                unit="distinct results", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Fixed-order pairwise summation is reproducible for one order but not permutation-invariant", "numerical",
                {"distinct_results": pairwise_distinct, "orders": len(pairwise_orders), "tree_mismatches": tree_mismatches,
                 "first_order_sum_hex": first_order_sum.hex()},
                {"checks": [_check("orders on which recursive Python and NumPy level-wise implementations of the same "
                                   "tree differ in any bit (uniform, every sampled order)", tree_mismatches,
                                   kind="cross_implementation"),
                            _check("distinct pairwise results across permutations (uniform)", pairwise_distinct, 2, "ge")]},
                counterexample={"statement": "Pairwise summation is order-independent",
                                "witness": {"dataset": "uniform n=1024 PCG64(148)", "distinct_results": pairwise_distinct}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Kahan summation loses the sum [1, 1e100, 1, -1e100] that Neumaier summation keeps", "numerical",
                {"kahan": kahan_case["kahan"]["first_order_result"], "neumaier": kahan_case["neumaier"]["first_order_result"],
                 "exact": study["kahan-counterexample"]["exact"]},
                {"checks": [_check("Kahan error", abs(kahan_case["kahan"]["first_order_result"] - 2.0), 2.0, "ge"),
                            _check("Neumaier error", kahan_case["neumaier"]["first_order_result"] - 2.0)]},
                counterexample={"statement": "Kahan compensated summation is accurate whenever Neumaier's is",
                                "witness": {"input": [1.0, 1e100, 1.0, -1e100], "kahan": 0.0, "neumaier": 2.0}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Observed errors of the sequential, pairwise, Neumaier and exact sums lie within their rigorous bounds",
                "numerical", {alg: worst[alg] for alg in rigorous},
                {"checks": [_check(f"{alg}: max |error|/bound over datasets and permutations", worst[alg], 1.0, "le",
                                   kind="analytic") for alg in rigorous]},
                unit="fraction of the bound",
                uncertainty={"kind": "truncation_bound", "value": 1.0,
                             "basis": "rigorous worst-case bounds (Higham 2002 ch. 4; Ogita, Rump and Oishi 2005); "
                                      "errors are exact rationals"},
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Observed Kahan errors lie within 2u sum|x| plus the declared second-order allowance", "numerical",
                worst["kahan"],
                {"checks": [_check(f"kahan: max |error| / (2u + {kernels.KAHAN_SECOND_ORDER} n u^2) sum|x|",
                                   worst["kahan"], 1.0, "le")]},
                unit="fraction of the allowance",
                uncertainty={"kind": "truncation_bound", "value": 1.0,
                             "basis": "first-order bound 2u sum|x| (Higham eq. 4.8) plus a declared, unproven "
                                      "second-order constant"},
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("The bound 2u|S| + 4n u^2 sum|x| does not bound Neumaier summation", "numerical",
                {"n": old_bound["n"], "ratio_to_superseded_bound": old_bound["ratio_to_superseded"],
                 "ratio_to_rigorous_bound": old_bound["ratio_to_bound"]},
                {"checks": [_check("Neumaier error / (2u|S| + 4n u^2 sum|x|) on absorbed tiny terms",
                                   old_bound["ratio_to_superseded"], 1.0, "ge"),
                            _check("Neumaier error / (u|S| + gamma_{n-1}^2 sum|x|) on the same input",
                                   old_bound["ratio_to_bound"], 1.0, "le", kind="analytic")]},
                counterexample={"statement": "Neumaier summation error is at most 2u|S| + 4n u^2 sum|x|",
                                "witness": {"input": "[1] + 1000 x [0.7u(1 + 2^-20)] + [-1]", "n": old_bound["n"],
                                            "ratio": old_bound["ratio_to_superseded"]}},
                unit="error over bound",
                uncertainty=_roundoff(0.0, "exact rational error; ratios rounded once to binary64"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Distinct results under permutation for each algorithm and dataset", "numerical", distinct_table,
                {"generator": {"name": "PCG64 permutations", "seed": 1480, "orders": 25},
                 "checks": [_check("exact accumulation: distinct results minus one, worst dataset",
                                   max(t["exact"] for t in distinct_table.values()) - 1),
                            _check("sequential summation: distinct results on the cancelling dataset",
                                   distinct_table["cancelling"]["sequential"], 2, "ge")]},
                unit="distinct results", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Deterministic reduction policy record", "provenance", REDUCTION_POLICY,
                {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#reduction-policies (Higham 2002, ch. 4; Ogita, "
                               "Rump and Oishi 2005)"}, uncertainty=DESIGN_U, tolerance=EXACT),
    ]
    fields = _fields(
        "Only a correctly rounded (exact-accumulation) sum is bitwise independent of summation order; fixed-tree "
        "pairwise sums are reproducible only for a fixed order; compensated sums are accurate but not order-invariant, "
        "and plain Kahan fails on large cancelling terms.",
        "Rigorous bounds (u = 2^-53, S = sum|x|, s = exact sum): sequential gamma_{n-1} S; pairwise "
        "gamma_{ceil(log2 n)} S; Neumaier u|s| + gamma_{n-1}^2 S (its compensation terms are those of Sum2); exact "
        "rounding u|s|. Kahan: 2u S + O(n u^2) S, checked with a declared allowance 4 n u^2 S. Exact sums by integer "
        "accumulation at scale 2^1074.",
        ["uniform[-1,1) n=1024", "positive values 10^U(-8,8) n=1024", "cancelling +/-10^U(0,16) pairs plus noise n=1024",
         "[1, 1e100, 1, -1e100]", "absorbed tiny terms [1] + 1000 x [0.7u(1 + 2^-20)] + [-1]",
         "24 PCG64(1480) permutations of each"],
        "Exact rational errors of each floating-point result; count of distinct bit patterns across orders.",
        "Exact accumulation: one result per dataset and equal to math.fsum; the fixed pairwise tree gives the same "
        "bits in two implementations for every order but different bits across orders; every error within its "
        "bound; the earlier O(n u^2) Neumaier bound fails where the second-order term grows like n^2.",
        "Sum each permutation with five algorithms; compare with exact rationals and math.fsum; walk the pairwise "
        "tree a second way (NumPy level-wise additions) on every order; test the earlier Neumaier bound on absorbed "
        "tiny terms; retain the policy.",
        _gpu_question(),
        numerical_result=(f"Distinct results across 25 orders: exact {exact_distinct}, pairwise (uniform) "
                          f"{pairwise_distinct}; recursive/level-wise pairwise tree mismatches {tree_mismatches} of "
                          f"{len(pairwise_orders)} orders; exact vs math.fsum mismatches {fsum_mismatch}; Kahan on "
                          f"[1, 1e100, 1, -1e100] = {kahan_case['kahan']['first_order_result']}, Neumaier = "
                          f"{kahan_case['neumaier']['first_order_result']}; worst error/bound "
                          + ", ".join(f"{k} {v:.3g}" for k, v in sorted(worst.items()))
                          + f"; Neumaier error is {old_bound['ratio_to_superseded']:.3g} times the earlier bound "
                            f"2u|S| + 4n u^2 S at n = {old_bound['n']} and {old_bound['ratio_to_bound']:.3g} of the "
                            "rigorous one."),
        uncertainty="Exact arithmetic; permutation counts are samples (25 orders), so invariance of non-exact "
                    "algorithms is only refuted, never proven, and the exact algorithm's invariance is a theorem "
                    "sampled here. The Kahan allowance constant is declared, not proven.",
        failure_modes_checked=["order dependence", "catastrophic cancellation", "Kahan large-term failure",
                               "second-order growth of compensated error (absorbed tiny terms)", "bound violation",
                               "exact/fsum disagreement"],
        unresolved_assumptions=["The Kahan second-order constant (4) is a policy allowance without proof",
                                "Parallel (multi-thread) reductions are not exercised",
                                "Fixed-tree reproducibility was shown between two implementations on one platform; "
                                "on other platforms it rests on IEEE-754 round-to-nearest addition and on the "
                                "regression gate, which compares the retained first-order sum bits exactly"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T149
@task("T149", changed_files=(MODULE, FPGA, DOC),
      regression_tests=(f"{TESTS}::test_t149_telemetry_only_frames",
                        f"{TESTS}::test_ciw_producers_of_independent_checks_carry_a_revision"))
def fpga_telemetry(ctx):
    rng = np.random.Generator(np.random.PCG64(149))
    fpga.validate_interface(fpga.INTERFACE)
    roundtrip_failures = 0
    for k in range(400):
        channels = [int(v) for v in rng.integers(-2 ** 31, 2 ** 31, int(rng.integers(0, 17)))]
        sequence, stamp = int(rng.integers(0, 2 ** 32)), int(rng.integers(0, 2 ** 63))
        flags = int(rng.integers(0, 4))
        decoded = fpga.decode_frame(fpga.encode_frame(sequence, stamp, 7, channels, flags))
        roundtrip_failures += decoded != {"sequence": sequence, "timestamp_ns": stamp, "clock_id": 7, "flags": flags,
                                          "channels": channels}
    import zlib
    messages = [rng.integers(0, 256, int(rng.integers(0, 200)), dtype=np.uint8).tobytes() for _ in range(500)]
    crc_mismatch = sum(fpga.crc32(m) != zlib.crc32(m) for m in messages)
    frame = fpga.encode_frame(123456, 987654321, 7, [1, -2, 3, 2 ** 31 - 1])
    bits = len(frame) * 8

    def detected(data):
        try:
            fpga.decode_frame(data)
            return False
        except fpga.TelemetryRefusal:
            return True

    single = sum(detected(fpga._flip(frame, [p])) for p in range(bits))
    random_total = 5000
    random_detected = sum(detected(fpga._flip(frame, list(rng.choice(bits, int(rng.integers(2, 12)), replace=False))))
                          for _ in range(random_total))
    # Exact burst analysis: every 32-bit window of single-bit syndromes must be linearly independent.
    bursts = fpga.burst_rank_deficient_windows(frame)
    big_endian = frame[:-4] + fpga.crc32(frame[:-4]).to_bytes(4, "big")
    bursts_be = fpga.burst_rank_deficient_windows(big_endian, trailer_byteorder="big")
    # Witness taken inside the payload, so every header check passes and only the CRC could have refused it.
    bursts_msb = fpga.burst_rank_deficient_windows(frame, msb_first=True, witness_from=8 * fpga.HEADER.size)
    be_escapes = bursts_be["witness"] is not None and fpga.frame_check(
        fpga._flip(big_endian, bursts_be["witness"]), "big") == 0
    msb_escapes = bursts_msb["witness"] is not None and detected(
        fpga._flip(frame, bursts_msb["witness"], msb_first=True)) is False
    body = fpga.HEADER.pack(fpga.MAGIC, fpga.FRAME_VERSION, fpga.TELEMETRY, 0, 1, 0, 7, 2, 8)
    refusals = [
        _refusal("decode a forged command frame (type 0x80)", "command_path_refused",
                 lambda: fpga.decode_frame(fpga.forge_frame(0x80, 0))),
        _refusal("decode a forged register-write frame (type 0x81)", "command_path_refused",
                 lambda: fpga.decode_frame(fpga.forge_frame(0x81, 0))),
        _refusal("decode a telemetry frame with the write-request flag", "command_path_refused",
                 lambda: fpga.decode_frame(fpga.forge_frame(fpga.TELEMETRY, fpga.FLAG_WRITE_REQUEST))),
        _refusal("decode an undefined frame type 0x02", "unknown_frame_type",
                 lambda: fpga.decode_frame(fpga.forge_frame(0x02, 0))),
        _refusal("decode reserved flag bits", "reserved_flags", lambda: fpga.decode_frame(fpga.forge_frame(1, 0x0100))),
        _refusal("encode a write request", "command_path_refused",
                 lambda: fpga.encode_frame(1, 0, 7, [1], flags=fpga.FLAG_WRITE_REQUEST)),
        _refusal("interface spec with a host-to-device field", "command_path_refused",
                 lambda: fpga.validate_interface(dict(fpga.INTERFACE, host_to_device=[{"name": "setpoint"}]))),
        _refusal("interface spec with a write_enable field", "command_path_refused",
                 lambda: fpga.validate_interface(dict(fpga.INTERFACE, fields=fpga.INTERFACE["fields"] + [
                     {"name": "write_enable", "direction": "device_to_host"}]))),
        _refusal("truncated frame", "truncated_frame", lambda: fpga.decode_frame(frame[:20])),
        _refusal("valid CRC, payload_bytes not 4 x channel count", "length_mismatch",
                 lambda: fpga.decode_frame(fpga.forge_frame(fpga.TELEMETRY, 0, payload_bytes=12))),
        _refusal("valid CRC, header without payload (trailing bytes missing)", "length_mismatch",
                 lambda: fpga.decode_frame(body + fpga.TRAILER.pack(fpga.crc32(body)))),
        _refusal("valid CRC, magic 'XXXX'", "bad_magic",
                 lambda: fpga.decode_frame(fpga.forge_frame(fpga.TELEMETRY, 0, magic=b"XXXX"))),
        _refusal("valid CRC, frame version 2", "unsupported_version",
                 lambda: fpga.decode_frame(fpga.forge_frame(fpga.TELEMETRY, 0, version=2))),
    ]
    # Structural, not by name: after receiving good and corrupted frames the receiver may expose only accept() and
    # plain data. A forged subclass with an emit() method and a transport handle shows the check can fail.
    receiver = fpga.TelemetryReceiver(0, STALE_AFTER_NS)
    for data in (frame, fpga._flip(frame, [3]), fpga.encode_frame(123457, 987655321, 7, [4])):
        receiver.accept(data, 987_660_000)
    interface = fpga.receiver_interface(fpga.TelemetryReceiver, receiver)

    class _ForgedReceiver(fpga.TelemetryReceiver):
        def emit(self, data):
            return self.link.write(data)

    forged = _ForgedReceiver(0, STALE_AFTER_NS)
    forged.link = io.BytesIO()
    forged_interface = fpga.receiver_interface(_ForgedReceiver, forged)
    ctx.artifact_json("telemetry-interface.json", fpga.INTERFACE)
    ctx.artifact_json("frame-example.json", {"hex": frame.hex(), "decoded": fpga.decode_frame(frame),
                                             "header_bytes": fpga.HEADER.size, "crc_bytes": fpga.TRAILER.size,
                                             "crc_byte_order": "little"})
    ctx.artifact_json("error-detection.json", {
        "frame_bits": bits, "single_bit": [single, bits], "random_2_to_11_bit": [random_detected, random_total],
        "burst_windows_32": {"little_endian_lsb_first": bursts, "big_endian_trailer_lsb_first": bursts_be,
                             "little_endian_msb_first_numbering": bursts_msb}})
    findings = [
        finding("Telemetry frames round-trip every header field and channel bit-exactly", "computational_pipeline",
                roundtrip_failures,
                {"generator": {"name": "PCG64 random frames", "seed": 149, "frames": 400},
                 "checks": [_check("round-trip failures", roundtrip_failures)]}, unit="failures",
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("CIW table-driven CRC-32 agrees with zlib and the catalogue check value", "numerical",
                {"messages": len(messages), "mismatches": crc_mismatch, "check_value": f"{fpga.crc32(b'123456789'):#010x}"},
                {"checks": [_check("CRC-32(b'123456789') - 0xCBF43926", fpga.crc32(b"123456789") - 0xCBF43926,
                                   kind="analytic")],
                 "independent_check": dict(_check("mismatches against zlib.crc32 on random messages", crc_mismatch),
                                           producer=_ciw_producer("ciw.lab.implementation_targets_fpga.crc32", FPGA),
                                           checker={"implementation": "zlib.crc32", "revision": zlib.ZLIB_RUNTIME_VERSION})},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Every burst of at most 32 bits and every single-bit error in a frame is refused by the decoder",
                "numerical",
                {"single_bit": [single, bits], "burst_windows": bursts["windows"],
                 "rank_deficient_windows": len(bursts["deficient"]), "random_multi_bit": [random_detected, random_total]},
                {"checks": [_check("undetected single-bit errors", bits - single),
                            _check("32-bit windows (LSB-first bit order) whose syndromes are linearly dependent",
                                   len(bursts["deficient"]), kind="analytic"),
                            _check("undetected random 2-11 bit errors", random_total - random_detected)]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("A big-endian CRC trailer lets a 32-bit burst across the payload/CRC boundary escape", "numerical",
                {"rank_deficient_windows": len(bursts_be["deficient"]), "first_window": (bursts_be["deficient"] or [None])[0],
                 "witness_bits": bursts_be["witness"]},
                {"checks": [_check("rank-deficient 32-bit windows with the trailer stored big-endian",
                                   len(bursts_be["deficient"]), 1, "ge"),
                            _check("witness error pattern leaves the big-endian CRC check at zero", float(be_escapes),
                                   1, "ge")]},
                counterexample={"statement": "Appending CRC-32 in either byte order keeps the 32-bit burst guarantee",
                                "witness": {"frame": "encode_frame(123456, 987654321, 7, [1, -2, 3, 2**31 - 1])",
                                            "trailer": "big-endian", "flipped_bits": bursts_be["witness"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("The burst guarantee holds only in the LSB-first bit order of the reflected CRC", "numerical",
                {"rank_deficient_windows_msb_first": len(bursts_msb["deficient"]), "witness_bits": bursts_msb["witness"]},
                {"checks": [_check("rank-deficient 32-bit windows when bits are numbered MSB-first within bytes",
                                   len(bursts_msb["deficient"]), 1, "ge"),
                            _check("decoder accepts the MSB-first witness burst inside the payload",
                                   float(msb_escapes), 1, "ge")]},
                counterexample={"statement": "CRC-32 detects every 32-bit burst whatever order the link sends bits in",
                                "witness": {"numbering": "bit p = bit 7 - p % 8 of byte p // 8",
                                            "flipped_bits": bursts_msb["witness"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Decoder, encoder and interface validator refuse every command, write or malformed path",
                "computational_pipeline", sum(c["passed"] for c in refusals), {"checks": refusals}, unit="refusals",
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Host receiver exposes only accept() and plain data attributes, with no sending method or transport "
                "handle", "computational_pipeline", interface,
                {"checks": [_check("public receiver methods other than accept",
                                   len(set(interface["public_methods"]) - {"accept"})),
                            _check("receiver base classes other than object", len(set(interface["bases"]) - {"object"})),
                            _check("receiver attributes holding objects other than plain data, after receiving frames",
                                   len(interface["non_data_attributes"])),
                            _check("forged subclass: public methods other than accept (emit)",
                                   len(set(forged_interface["public_methods"]) - {"accept"}), 1, "ge"),
                            _check("forged subclass: attributes holding a transport handle (link)",
                                   len(forged_interface["non_data_attributes"]), 1, "ge")]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("The frame format works on real FPGA links", "physical", None, {}),
        finding("A telemetry-only interface guarantees the FPGA cannot actuate the machine", "machine_safety", None, {}),
    ]
    fields = _fields(
        "A read-only frame (header, sequence, timestamp, clock id, raw payload, CRC-32) with a single telemetry frame "
        "type and no host-to-device field lets the host refuse every corrupted frame and every command or write path "
        "before any payload is used, provided the CRC is appended in the order that keeps the frame one codeword.",
        "Frame = 28-byte big-endian header | 4*c bytes int32 payload | CRC-32/IEEE (reflected) appended little-endian. "
        "Bit p is bit p % 8 of byte p // 8 (LSB first, the reflected CRC's polynomial order). An error pattern e "
        "escapes iff its syndrome sum is zero; if the 32 single-bit syndromes of every 32-bit window are linearly "
        "independent over GF(2), no burst of length <= 32 escapes. Other patterns escape with probability about 2^-32.",
        ["400 PCG64(149) random frames", "500 random messages for CRC comparison", "one 48-byte frame for exhaustive "
         "single-bit, exact 32-bit-window rank analysis and 5000 random corruptions", "13 forged frames and specs"],
        "Decoder outcomes (accepted record or refusal code) and GF(2) ranks of syndrome windows; no device was attached.",
        "Round trip exact; CRC matches zlib and 0xCBF43926; all single-bit errors refused; every 32-bit window has "
        "full rank with the little-endian trailer (not with a big-endian trailer, nor in MSB-first numbering); "
        "command, write and malformed frames refused; the receiver's public interface is accept() and plain data, "
        "and a forged subclass with emit() and a transport handle is flagged.",
        "Encode/decode random frames, compare CRC implementations, flip every bit, compute window ranks for three "
        "layouts and extract escaping bursts, corrupt at random, forge command frames, headers and specs.",
        "Deferred research question: an HDL implementation of this frame format on an FPGA, driven over a physical "
        "transmit-only link, to test the LSB-first burst guarantee and the unidirectional transport this task "
        "assumes (the link simulation of this section is software only)",
        numerical_result=(f"Round-trip failures {roundtrip_failures}/400; CRC mismatches {crc_mismatch}/500; single-bit "
                          f"{single}/{bits} refused; rank-deficient 32-bit windows {len(bursts['deficient'])}/"
                          f"{bursts['windows']} (little-endian trailer), {len(bursts_be['deficient'])} (big-endian), "
                          f"{len(bursts_msb['deficient'])} (MSB-first numbering); random {random_detected}/{random_total} "
                          f"refused; {sum(c['passed'] for c in refusals)}/{len(refusals)} command/write/malformed "
                          "refusals."),
        uncertainty="Exact: exhaustive for single-bit errors and exact (rank) for bursts on one frame layout; random "
                    "corruptions escape CRC-32 with probability 2^-32 each.",
        failure_modes_checked=["command frame types", "write-request flag", "reserved flags", "unknown types",
                               "host-to-device spec fields", "truncation", "length mismatch", "bad magic",
                               "unsupported version", "bit errors", "bursts across the payload/CRC boundary",
                               "trailer byte order", "bit numbering on the link"],
        unresolved_assumptions=["No FPGA, HDL implementation or physical link exists; the device side is simulated",
                                "The link is assumed to send bits LSB first (as UART and Ethernet do); an MSB-first "
                                "link loses the burst guarantee", "A unidirectional physical transport (TX-only UART, "
                                "multicast UDP) is assumed but not demonstrated",
                                "CRC protects against noise, not against a malicious sender"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T150
# Placeholder toolchain installation (not an executable toolchain); its manifest digest is the installation digest.
PLACEHOLDER_TOOLCHAIN_FILES = {"bin/placeholder-synth": b"placeholder synthesis executable (not executed)\n",
                               "lib/placeholder-core.bin": b"placeholder toolchain library\n",
                               "data/placeholder-part.db": b"placeholder device database\n"}
PLACEHOLDER_TOOLCHAIN_VERSION = "0.0.1"


def _synthetic_build(version: str, seed: int) -> dict:
    constraints = {"pins.xdc": f"# placeholder constraints {version}\nset_property PACKAGE_PIN A1 [get_ports clk]\n".encode(),
                   "timing.xdc": f"create_clock -period 10.000 [get_ports clk] ;# {version}\n".encode()}
    sources = {"rtl/telemetry_tx.v": f"// placeholder source {version}\nmodule telemetry_tx(); endmodule\n".encode(),
               "rtl/crc32.v": b"// placeholder CRC-32 source\nmodule crc32(); endmodule\n"}
    bitstream = fpga.synthetic_bitstream(seed)
    record = fpga.bitstream_identity(bitstream, toolchain={"name": "placeholder-toolchain (not executed)",
                                                           "version": PLACEHOLDER_TOOLCHAIN_VERSION,
                                                           "installation_sha256": fpga.installation_digest(
                                                               PLACEHOLDER_TOOLCHAIN_FILES)},
                                     part="placeholder-part", constraints=constraints, source_files=sources,
                                     synthesis_options={"seed": seed, "version": version}, synthetic=True)
    return {"bitstream": bitstream, "constraints": constraints, "sources": sources, "record": record}


FLOATING_VERSIONS = ("latest", ">=2024.1", "2024.1.x", "1.0-latest", "2023.2_nightly", "2024.1-rc")


@task("T150", changed_files=(MODULE, FPGA, SERIAL, DOC), regression_tests=(f"{TESTS}::test_t150_bitstream_identity",))
def bitstream_identity(ctx):
    build = _synthetic_build("2.1.0", 150)
    record, bitstream = build["record"], build["bitstream"]
    fpga.validate_identity(record, bitstream=bitstream, constraints=build["constraints"], source_files=build["sources"],
                           toolchain_files=PLACEHOLDER_TOOLCHAIN_FILES, toolchain_version=PLACEHOLDER_TOOLCHAIN_VERSION)

    def edited(**changes):
        body = dict(record, **changes)
        body.pop("record_sha256")
        body["record_sha256"] = hashlib.sha256(serial.canonical_bytes(body)).hexdigest()
        return body

    def toolchain(**changes):
        return dict(record["toolchain"], **changes)

    flipped = bytearray(bitstream)
    flipped[len(flipped) // 2] ^= 0x01
    changed_constraints = dict(build["constraints"], **{"timing.xdc": b"create_clock -period 8.000 [get_ports clk]\n"})
    added_constraints = dict(build["constraints"], **{"extra.xdc": b"# added\n"})
    changed_sources = dict(build["sources"], **{"rtl/crc32.v": b"// edited\n"})
    changed_toolchain = dict(PLACEHOLDER_TOOLCHAIN_FILES, **{"lib/placeholder-core.bin": b"patched toolchain library\n"})
    added_toolchain = dict(PLACEHOLDER_TOOLCHAIN_FILES, **{"bin/placeholder-plugin": b"added executable\n"})
    missing = {k: v for k, v in record.items() if k != "toolchain"}
    no_installation = {k: v for k, v in record["toolchain"].items() if k != "installation_sha256"}
    mutations = [
        _refusal("one flipped bitstream bit", "bitstream_digest_mismatch",
                 lambda: fpga.validate_identity(record, bitstream=bytes(flipped))),
        _refusal("truncated bitstream", "bitstream_size_mismatch",
                 lambda: fpga.validate_identity(record, bitstream=bitstream[:-1])),
        _refusal("edited constraint file", "constraints_digest_mismatch",
                 lambda: fpga.validate_identity(record, constraints=changed_constraints)),
        _refusal("added constraint file", "constraints_digest_mismatch",
                 lambda: fpga.validate_identity(record, constraints=added_constraints)),
        _refusal("edited source file", "source_tree_mismatch",
                 lambda: fpga.validate_identity(record, source_files=changed_sources)),
        _refusal("toolchain installation with an edited library", "toolchain_installation_mismatch",
                 lambda: fpga.validate_identity(record, toolchain_files=changed_toolchain)),
        _refusal("toolchain installation with an added executable", "toolchain_installation_mismatch",
                 lambda: fpga.validate_identity(record, toolchain_files=added_toolchain)),
        _refusal("installed toolchain reporting version 0.0.2", "toolchain_version_mismatch",
                 lambda: fpga.validate_identity(record, toolchain_version="0.0.2")),
        *[_refusal(f"floating toolchain version {version!r}", "floating_toolchain_version",
                   lambda version=version: fpga.validate_identity(edited(toolchain=toolchain(version=version))))
          for version in FLOATING_VERSIONS],
        _refusal("toolchain version with a trailing newline '2024.1\\n'", "floating_toolchain_version",
                 lambda: fpga.validate_identity(edited(toolchain=toolchain(version="2024.1\n")))),
        _refusal("toolchain version as the number 2024.1, not a string", "floating_toolchain_version",
                 lambda: fpga.validate_identity(edited(toolchain=toolchain(version=2024.1)))),
        _refusal("toolchain without an installation digest", "unpinned_toolchain_installation",
                 lambda: fpga.validate_identity(edited(toolchain=no_installation))),
        _refusal("installation digest with a trailing newline", "unpinned_toolchain_installation",
                 lambda: fpga.validate_identity(edited(toolchain=toolchain(
                     installation_sha256=record["toolchain"]["installation_sha256"] + "\n")))),
        _refusal("part changed without recomputing the record digest", "record_digest_mismatch",
                 lambda: fpga.validate_identity(dict(record, part="other-part"))),
        _refusal("record without toolchain", "missing_field", lambda: fpga.validate_identity(missing)),
        _refusal("uppercase digest", "bad_digest_format",
                 lambda: fpga.validate_identity(dict(record, bitstream_sha256=record["bitstream_sha256"].upper()))),
        _refusal("source-tree digest 'TBD'", "bad_digest_format",
                 lambda: fpga.validate_identity(edited(source_tree={"kind": "manifest-sha256", "value": "TBD"}))),
        _refusal("source-tree digest with a trailing newline", "bad_digest_format",
                 lambda: fpga.validate_identity(edited(source_tree={"kind": "manifest-sha256",
                                                                    "value": record["source_tree"]["value"] + "\n"}))),
        _refusal("source tree given as a bare string, with source files supplied", "missing_field",
                 lambda: fpga.validate_identity(edited(source_tree="unknown"), source_files=build["sources"])),
        _refusal("synthetic flag 'no' instead of a boolean", "bad_field_type",
                 lambda: fpga.validate_identity(edited(synthetic="no"))),
        _refusal("negative bitstream size", "bad_field_type",
                 lambda: fpga.validate_identity(edited(bitstream_bytes=-1))),
        _refusal("pinned pre-release version '2024.1-rc2' (positive control)", "none",
                 lambda: fpga.validate_identity(edited(toolchain=toolchain(version="2024.1-rc2")))),
    ]
    ctx.artifact_json("bitstream-identity.json", record)
    ctx.artifact_json("identity-schema.json", {"schema": fpga.IDENTITY_SCHEMA, "fields": list(fpga.IDENTITY_FIELDS)
                                               + ["record_sha256"], "toolchain_version_pattern": fpga._PINNED_VERSION.pattern,
                                               "toolchain_fields": ["name", "version", "installation_sha256"],
                                               "canonical_encoding": serial.SPEC_ID})
    refused = sum(c["passed"] for c in mutations if c["expected_refusal"] != "none")
    controls = sum(c["passed"] for c in mutations if c["expected_refusal"] == "none")
    findings = [
        finding("Bitstream identity record binds bitstream, toolchain version and installation manifest, constraints "
                "and source tree", "provenance",
                {"record_sha256": record["record_sha256"], "refused_mutations": refused,
                 "accepted_positive_controls": controls},
                {"generator": {"name": "seeded synthetic placeholder bitstream", "seed": 150}, "checks": mutations},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("A real bitstream with this identity exists and is loaded on hardware", "physical", None, {}),
        finding("The bitstream is approved for production deployment", "production_acceptance", None, {}),
    ]
    decision = _refusal("deployment decision for a synthetic record", "synthetic_bitstream",
                        lambda: fpga.deployment_decision(record))
    findings.insert(1, finding("Deployment of a synthetic placeholder bitstream is refused", "computational_pipeline",
                               decision["observed_refusal"], {"checks": [decision]}, uncertainty=EXACT_U,
                               tolerance=EXACT))
    fields = _fields(
        "An identity record over canonical JSON can bind a bitstream's sha256 to its exact toolchain version and "
        "installation manifest digest, constraint files and source tree, so that a change to any of them is detected "
        "when the changed artifact (bitstream, installed toolchain files and reported version, constraint or source "
        "files) is checked against the record. The installation digest binds exactly the toolchain files supplied; "
        "which files of a vendor toolchain must be supplied is not specified here. Scope: the task records identity "
        "and toolchain, a build artifact rather than a measurement, so a synthetic placeholder bitstream exercises "
        "every field and refusal of the record; recording a real design's bitstream needs a vendor toolchain and is "
        "the deferred research question.",
        "record_sha256 = sha256(E(record without record_sha256)), E = ciw.canonical-json.v1; constraints_sha256 = "
        "sha256(E({file: sha256})); source tree = sha256(E({path: sha256})); installation_sha256 = "
        "sha256(E({toolchain file: sha256})); toolchain version must match "
        f"{fpga._PINNED_VERSION.pattern} and equal the version the installed toolchain reports; every digest field is "
        "64 lowercase hex digits, sizes are nonnegative integers and the synthetic flag is a boolean.",
        ["Seeded synthetic placeholder bitstream (prefix CIW-SYNTHETIC-BITSTREAM-NOT-A-CONFIGURATION)",
         "placeholder constraint and source files", "three placeholder toolchain installation files"],
        "Validator outcomes over the record and supplied artifacts; no FPGA toolchain ran.",
        f"The unmodified record validates against every artifact; each of {len(mutations) - 1} mutations is refused "
        "with its code and the pinned pre-release control is accepted; deployment is refused. Reproducibility of the "
        "record across runs is left to the regression gate, which compares record_sha256 exactly.",
        f"Build the record, validate it against the bitstream, constraint, source and toolchain files, apply "
        f"{len(mutations) - 1} mutations and one positive control, request a deployment decision.",
        "Deferred research question: toolchain determinism: build one design twice with a vendor toolchain whose "
        "installation manifest is recorded and compare the bitstream digests, and specify which toolchain files the "
        "installation digest must cover (no real bitstream, design or toolchain exists here)",
        numerical_result=f"record_sha256 {record['record_sha256'][:16]}...; {refused}/{len(mutations) - 1} mutations "
                         f"refused with their codes; {controls}/1 pinned pre-release accepted; deployment refused "
                         "(synthetic_bitstream).",
        uncertainty="Exact digests; the placeholder bitstream has no hardware meaning.",
        failure_modes_checked=["bitstream bit flip", "truncation", "constraint edit/addition", "source edit",
                               "toolchain installation file edited or added", "installed toolchain version differs",
                               "floating toolchain versions (latest, ranges, .x, nightly, unnumbered rc, trailing newline)",
                               "non-string toolchain version", "toolchain without installation digest",
                               "unrecomputed digest", "missing field",
                               "digest format (uppercase, TBD, trailing newline)", "malformed source tree",
                               "non-boolean synthetic flag",
                               "negative size"],
        unresolved_assumptions=["No real bitstream, toolchain log or device part exists here",
                                "Toolchain determinism (same inputs -> same bitstream) is not established",
                                "installation_sha256 binds the toolchain files supplied; which files of a vendor "
                                "toolchain must be supplied (executables, libraries, device databases) is not "
                                "specified here, and a change outside them is not detected",
                                "The installed toolchain's reported version is taken as given",
                                "Detection of unseen changes relies on sha256 collision resistance"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T151
SEEDS = {"1.2.0": 1512, "1.4.1": 1541, "2.0.0": 1520, "2.1.0": 1521}


def _registry():
    return {version: _synthetic_build(version, seed)["record"] for version, seed in SEEDS.items()}


@task("T151", changed_files=(MODULE, FPGA, DOC), regression_tests=(f"{TESTS}::test_t151_compatibility_and_rollback",))
def fpga_rollback(ctx):
    matrix = fpga.COMPATIBILITY
    registry = ctx.memo("t151-registry", _registry)
    combos = [(b, h, r) for b in matrix["bitstreams"] for h in matrix["hosts"] for r in matrix["boards"]]
    rule = {c for c in combos if fpga.compatible(matrix, *c)}
    built = fpga.compatible_set(matrix)
    disagreements = len(rule ^ built)
    # Counterexample search: rolling back to the immediately previous version.
    versions = sorted(matrix["bitstreams"], key=fpga.version_key)
    unsafe_previous = [{"host": h, "board": r, "from": versions[i], "to": versions[i - 1]}
                       for i in range(1, len(versions)) for h in matrix["hosts"] for r in matrix["boards"]
                       if fpga.compatible(matrix, versions[i], h, r) and not fpga.compatible(matrix, versions[i - 1], h, r)]
    good = fpga.rollback_record(matrix, registry, current="2.1.0", target="2.0.0", host="2.0", board="revC",
                                reason="Regression in telemetry timestamp field (synthetic scenario)")

    def attempt(**changes):
        record = dict(good, **changes)
        return lambda: fpga.validate_rollback(record, matrix, registry)

    witness = unsafe_previous[0] if unsafe_previous else None
    # Without an enumerated witness the refusal is still exercised on a fixed incompatible state; the
    # counterexample finding's own check then fails instead of the task crashing.
    probe = witness or {"host": "1.0", "board": "revC", "from": "2.1.0", "to": "1.2.0"}
    refusals = [
        _refusal("rollback to an older version in an incompatible state", "incompatible_target",
                 lambda: fpga.rollback_record(matrix, registry, current=probe["from"], target=probe["to"],
                                              host=probe["host"], board=probe["board"], reason="test")),
        _refusal("rollback to an unregistered bitstream", "unregistered_target",
                 attempt(to_bitstream="1.9.9", to_identity_sha256="0" * 64)),
        _refusal("rollback citing a different identity digest", "unregistered_target",
                 attempt(to_identity_sha256=registry["1.4.1"]["record_sha256"])),
        _refusal("rollback to the current bitstream", "no_op_rollback",
                 attempt(to_bitstream="2.1.0", to_identity_sha256=registry["2.1.0"]["record_sha256"])),
        _refusal("'rollback' to a newer bitstream", "not_a_rollback",
                 attempt(from_bitstream="2.0.0", from_identity_sha256=registry["2.0.0"]["record_sha256"],
                         to_bitstream="2.1.0", to_identity_sha256=registry["2.1.0"]["record_sha256"])),
        _refusal("rollback without a reason", "missing_reason", attempt(reason=" ")),
        _refusal("rollback checked against another matrix version", "matrix_mismatch",
                 attempt(compatibility_matrix_sha256="0" * 64)),
        _refusal("execute a validated rollback", "rollback_requires_machine_authority",
                 lambda: fpga.execute_rollback(good)),
    ]
    ctx.artifact_json("compatibility-matrix.json", dict(matrix, matrix_sha256=serial.canonical_sha256(matrix),
                                                        compatible=sorted(map(list, rule))))
    ctx.artifact_json("rollback-record.json", good)
    ctx.artifact_json("identity-registry.json", {v: r["record_sha256"] for v, r in registry.items()})
    findings = [
        finding("Compatibility rules and the set construction agree on every combination", "computational_pipeline",
                {"combinations": len(combos), "compatible": len(rule), "disagreements": disagreements},
                {"checks": [_check("rule/set disagreements over bitstream x host x board", disagreements,
                                   kind="cross_implementation")]}, uncertainty=EXACT_U, tolerance=EXACT),
        finding("Rollback validation refuses incompatible, unregistered, no-op, forward and unexplained rollbacks",
                "computational_pipeline", sum(c["passed"] for c in refusals), {"checks": refusals},
                unit="refusals", uncertainty=EXACT_U, tolerance=EXACT),
        finding("Rolling back to the previous bitstream version can be incompatible", "computational_pipeline",
                {"cases": len(unsafe_previous), "witness": witness},
                {"checks": [_check("(host, board) states where the previous version is incompatible", len(unsafe_previous),
                                   1, "ge")]},
                counterexample={"statement": "Rolling back to the immediately previous bitstream is always compatible",
                                "witness": witness},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("The rollback procedure is safe to execute on a production machine", "machine_safety", None, {}),
        finding("Rollback records are accepted for production change control", "production_acceptance", None, {}),
    ]
    fields = _fields(
        "A versioned compatibility matrix (bitstream frame format, supported boards, minimum host decoder) and a "
        "rollback record bound to retained identities let validation refuse every incompatible or unregistered "
        "rollback; 'previous version' is not a safe default.",
        "compatible(b, h, r) <=> format(b) in formats(h) and r in boards(b) and h >= min_host(b); a rollback is valid "
        "iff both identities are registered with matching digests, target < current, a reason is given, the matrix "
        "digest matches and compatible(target, host, board).",
        [f"{len(matrix['bitstreams'])} bitstreams x {len(matrix['hosts'])} host decoders x {len(matrix['boards'])} "
         "boards (declared matrix v3)", "synthetic identity records from T150's builder"],
        "Validator outcomes; nothing was flashed or executed.",
        "Rule and set forms agree; every forged rollback is refused with its code; execution is always refused.",
        "Enumerate the matrix two ways, search previous-version rollbacks for incompatibility, validate forged records.",
        "Deferred research question: qualify the compatibility matrix on real boards and decoder versions (it is "
        "illustrative here) and establish how board revision and host decoder version are reported trustworthily",
        numerical_result=(f"{len(rule)}/{len(combos)} combinations compatible, {disagreements} disagreements; "
                          f"{len(unsafe_previous)} previous-version rollbacks incompatible (witness {witness}); "
                          f"{sum(c['passed'] for c in refusals)}/{len(refusals)} refusals."),
        uncertainty="Exact enumeration over a declared, synthetic matrix.",
        failure_modes_checked=["incompatible frame format", "unsupported board", "unregistered target",
                               "digest substitution", "no-op", "forward 'rollback'", "missing reason",
                               "stale matrix", "execution without authority"],
        unresolved_assumptions=["The matrix is illustrative; real compatibility needs hardware qualification",
                                "Board revision detection and host decoder version reporting are assumed trustworthy"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T152
STALE_AFTER_NS = 4_000_000
CALIBRATION_DELIVERIES = 200
MODEL_RUN = {"seed": 1520, "frames": 200_000}


def _flagged(receiver, deliveries) -> list:
    """Per delivery, whether the receiver reported it stale (its stale list, matched by sequence and arrival)."""
    from collections import Counter

    reported = Counter(receiver.stale)
    out = []
    for item in deliveries:
        key = (item["sequence"], item["received_ns"])
        out.append(reported[key] > 0)
        reported[key] -= 1
    return out


def _run_receiver(offset_ns, deliveries):
    receiver = fpga.TelemetryReceiver(offset_ns, STALE_AFTER_NS)
    for item in deliveries:
        receiver.accept(item["frame"], item["received_ns"])
    return receiver


def link_study() -> dict:
    sim = fpga.simulate_link()
    deliveries, lost, model = sim["deliveries"], sim["lost"], sim["model"]
    exact = _run_receiver(sim["clock_offset_ns"], deliveries)
    first, last = exact.first, exact.highest
    span = (last - first) % fpga.SEQUENCE_MODULUS + 1

    def inside(seq):
        return (seq - first) % fpga.SEQUENCE_MODULUS < span

    true_lost = {s for s in lost if inside(s)}
    latency = np.array([d["latency_ns"] for d in deliveries], dtype=float)
    excess = np.array([d["excess_ns"] for d in deliveries], dtype=float)
    truth = latency > STALE_AFTER_NS
    flag = np.array(_flagged(exact, deliveries))
    # Integer latencies: L <= a  <=>  base + G < a + 1, so each event probability is a CDF difference at a + 1.
    thr = STALE_AFTER_NS + 1
    p_false = fpga.latency_cdf(thr, model) - fpga.latency_cdf(thr - excess, model)
    declared = {"stale": int(truth.sum()), "flagged": int(flag.sum()), "missed": int((truth & ~flag).sum()),
                "false_alarms": int((~truth & flag).sum()), "expected_false_alarms": float(p_false.sum()),
                "sigma": float(math.sqrt(np.sum(p_false * (1 - p_false))))}
    # Offset estimated from the minimum of (arrival - timestamp) over a calibration window; evaluated after it.
    window = deliveries[:CALIBRATION_DELIVERIES]
    estimate = min(d["received_ns"] - fpga.decode_frame(d["frame"])["timestamp_ns"] for d in window)
    bias = estimate - sim["clock_offset_ns"]
    estimated = _run_receiver(estimate, deliveries)
    flag_est = np.array(_flagged(estimated, deliveries))[CALIBRATION_DELIVERIES:]
    truth_eval, excess_eval = truth[CALIBRATION_DELIVERIES:], excess[CALIBRATION_DELIVERIES:]
    p_miss = np.clip(fpga.latency_cdf(thr + bias - excess_eval, model) - fpga.latency_cdf(thr, model), 0.0, 1.0)
    biased = {"bias_ns": int(bias), "max_excess_ns": int(excess.max()), "evaluated": int(len(flag_est)),
              "stale": int(truth_eval.sum()), "missed": int((truth_eval & ~flag_est).sum()),
              "expected_missed": float(p_miss.sum()), "sigma": float(math.sqrt(np.sum(p_miss * (1 - p_miss)))),
              "false_alarms": int((~truth_eval & flag_est).sum())}
    arrival = [d["sequence"] for d in deliveries]
    in_order = sorted(set(arrival), key=lambda s: (s - first) % fpga.SEQUENCE_MODULUS)
    wrap_gap = [s for s in sorted(true_lost, key=lambda s: (s - first) % fpga.SEQUENCE_MODULUS)
                if s >= fpga.SEQUENCE_MODULUS - 3 or s < 3]
    duplicates_sent = len(deliveries) - (sim["frames"] - len(lost))
    long_run = fpga.link_draws(MODEL_RUN["seed"], MODEL_RUN["frames"], model)
    moments = fpga.gilbert_elliott_moments(model)
    loss_rate = float(long_run["drop"].mean())
    loss_sigma = math.sqrt(moments["per_frame_variance"] / MODEL_RUN["frames"])
    latencies = long_run["latency"][:, 0].astype(float)
    # Mean of floor(base + G): E[base + G] - 1/2 to within the (negligible) nonuniformity of the fraction.
    mean_latency = model["base_latency_ns"] + model["jitter_shape"] * model["jitter_scale_ns"] - 0.5
    latency_sigma = math.sqrt(model["jitter_shape"]) * model["jitter_scale_ns"] / math.sqrt(len(latencies))
    return {"sim": sim, "exact": exact, "span": span, "true_lost": true_lost, "declared": declared, "biased": biased,
            "naive_arrival": fpga.naive_gap_count(arrival), "naive_in_order": fpga.naive_gap_count(in_order),
            "wrap_gap": wrap_gap, "duplicates_sent": duplicates_sent,
            "model_run": {"frames": MODEL_RUN["frames"], "seed": MODEL_RUN["seed"], "loss_rate": loss_rate,
                          "stationary_loss": moments["stationary_loss"], "loss_sigma": loss_sigma,
                          "loss_z": (loss_rate - moments["stationary_loss"]) / loss_sigma,
                          "mean_latency_ns": float(latencies.mean()), "expected_latency_ns": mean_latency,
                          "latency_z": (float(latencies.mean()) - mean_latency) / latency_sigma}}


@task("T152", changed_files=(MODULE, FPGA, DOC), regression_tests=(f"{TESTS}::test_t152_loss_latency_staleness",))
def link_simulation(ctx):
    study = link_study()
    sim, exact, true_lost = study["sim"], study["exact"], study["true_lost"]
    declared, biased, run = study["declared"], study["biased"], study["model_run"]
    model = sim["model"]
    ages = np.array(exact.ages, dtype=float)
    summary = {"frames": sim["frames"], "deliveries": len(sim["deliveries"]), "lost_total": len(sim["lost"]),
               "lost_between_first_and_last": len(true_lost), "detected_lost": len(exact.pending),
               "reordered": exact.reordered, "duplicates_detected": exact.duplicates,
               "duplicates_sent": study["duplicates_sent"], "refused_frames": exact.refused,
               "declared_offset": declared, "estimated_offset": biased,
               "naive_arrival_order": study["naive_arrival"], "naive_in_order": study["naive_in_order"],
               "wrap_gap": study["wrap_gap"], "model_run": run}
    ctx.artifact_json("link-simulation.json", {"model": model, "summary": summary, "stale_after_ns": STALE_AFTER_NS,
                                               "start_sequence": sim["start_sequence"]})
    order = np.argsort(ages)
    ctx.artifact_text("age-distribution.svg", svg.line_plot(
        [("frame age (declared offset)", [float(v) / 1e6 for v in ages[order]],
          [float(i + 1) / len(ages) for i in range(len(ages))]),
         ("stale threshold", [STALE_AFTER_NS / 1e6] * 2, [0.0, 1.0])],
        title="Empirical CDF of frame age", xlabel="age (ms)", ylabel="fraction of deliveries", markers=False))
    moments_tolerance = {"abs": 1e-6, "rel": 1e-9}
    false_z = (declared["false_alarms"] - declared["expected_false_alarms"]) / declared["sigma"]
    miss_z = (biased["missed"] - biased["expected_missed"]) / biased["sigma"]
    power = 4 * run["loss_sigma"]
    findings = [
        finding("Sequence-number gap detection recovers every lost frame across the 32-bit wrap", "computational_pipeline",
                {"lost": len(true_lost), "detected": len(exact.pending), "reordered": exact.reordered,
                 "duplicates": exact.duplicates},
                {"generator": {"name": "Gilbert-Elliott loss, gamma latency, duplicates, quantized timestamps, drift",
                               "seed": 152},
                 "checks": [_check("|detected set symmetric difference true lost set|", len(exact.pending ^ true_lost)),
                            _check("duplicates detected minus duplicates sent", exact.duplicates - study["duplicates_sent"]),
                            _check("forced loss at sequence 2^32-1 detected",
                                   0.0 if (2 ** 32 - 1) in exact.pending else 1.0),
                            _check("frames refused by the decoder", sum(exact.refused.values()))]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("With the declared clock offset no stale frame is missed and false alarms match the timestamp "
                "quantization and drift prediction", "numerical",
                {k: declared[k] for k in ("stale", "flagged", "missed", "false_alarms", "expected_false_alarms")},
                {"checks": [_check("missed stale frames (age = latency + excess, excess >= 0)", declared["missed"],
                                   kind="analytic"),
                            _check("(false alarms - expected) / (4 sigma), expected = sum of P(limit - excess < L <= "
                                   "limit)", false_z / 4, 1.0, kind="analytic")]},
                uncertainty={"kind": "monte_carlo_95ci", "value": 2 * declared["sigma"],
                             "basis": "false-alarm count is a sum of independent Bernoulli trials given each frame's "
                                      "quantization and drift excess"},
                tolerance=moments_tolerance),
        finding("An offset estimated from minimum delay misses stale frames at the rate its bias predicts", "numerical",
                {k: biased[k] for k in ("bias_ns", "evaluated", "stale", "missed", "expected_missed", "false_alarms")},
                {"checks": [_check("bias minus the largest excess (no false alarm possible when positive)",
                                   biased["bias_ns"] - biased["max_excess_ns"], 0.0, "signed_ge", kind="analytic"),
                            _check("false alarms after the calibration window", biased["false_alarms"]),
                            _check("(missed - expected) / (4 sigma), expected = sum of P(limit < L <= limit + bias - "
                                   "excess)", miss_z / 4, 1.0, kind="analytic")]},
                uncertainty={"kind": "monte_carlo_95ci", "value": 2 * biased["sigma"],
                             "basis": "missed-frame count is a sum of independent Bernoulli trials given the bias"},
                tolerance=moments_tolerance),
        finding("Loss rate and mean latency of a long run match the declared link model", "numerical",
                {"frames": run["frames"], "loss_rate": run["loss_rate"], "stationary_loss": run["stationary_loss"],
                 "mean_latency_ns": run["mean_latency_ns"]},
                {"generator": {"name": "link_draws", "seed": run["seed"], "frames": run["frames"]},
                 "checks": [_check("(loss rate - stationary loss) / (4 sigma), exact Gilbert-Elliott variance",
                                   run["loss_z"] / 4, 1.0, kind="analytic"),
                            _check("(mean latency - (base + shape*scale - 1/2)) / (4 sigma)", run["latency_z"] / 4, 1.0,
                                   kind="analytic")]},
                uncertainty={"kind": "monte_carlo_95ci", "value": 2 * run["loss_sigma"],
                             "basis": "loss rate over 200000 frames; sigma^2 = [p(1-p) + 2(l_b-l_g)^2 pi_g pi_b "
                                      "lambda/(1-lambda)]/N"},
                tolerance=moments_tolerance),
        finding("Differencing sequence numbers in arrival order miscounts losses under reordering",
                "computational_pipeline",
                {"arrival_order": study["naive_arrival"], "in_order": study["naive_in_order"], "true": len(true_lost)},
                {"checks": [_check("|arrival-order count - true lost|", abs(study["naive_arrival"] - len(true_lost)), 1,
                                   "ge"),
                            _check("reordered deliveries in the stream", exact.reordered, 1, "ge")]},
                counterexample={"statement": "Summing seq - previous - 1 in arrival order counts lost frames",
                                "witness": {"arrival_order": study["naive_arrival"], "in_order": study["naive_in_order"],
                                            "true": len(true_lost)}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Non-modular differencing misses the loss at the 32-bit wrap", "computational_pipeline",
                {"in_order": study["naive_in_order"], "true": len(true_lost), "wrap_gap": study["wrap_gap"]},
                {"checks": [_check("true lost minus the in-order non-modular count", len(true_lost)
                                   - study["naive_in_order"], 1, "ge"),
                            _check("losses at the wrap (2^32 - 3 .. 2)", len(study["wrap_gap"]), 1, "ge")]},
                counterexample={"statement": "Sequence differences without modular arithmetic count every loss in a "
                                             "sequence-ordered stream",
                                "witness": {"in_order": study["naive_in_order"], "true": len(true_lost),
                                            "missed_at_wrap": study["wrap_gap"]}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Simulated loss, latency and staleness represent the real FPGA telemetry link", "physical", None, {}),
    ]
    fields = _fields(
        "Sequence numbers with modulo-2^32 serial arithmetic detect every loss, duplicate and reordering; timestamps "
        "with a declared clock offset never miss a stale frame, and their false alarms, like the misses of an offset "
        "estimated from minimum delay, occur at the rates the quantization, drift and bias predict.",
        "Gilbert-Elliott loss (p_gb=0.005, p_bg=0.2, loss 0.01/0.5), latency L = 2 ms + Gamma(2, 0.5 ms), duplicates "
        "p=0.002, period 1 ms, sampling phase U(0, 0.25 ms), timestamps quantized to 50 us, device/host drift 20 ppm, "
        "start sequence 2^32-1500, forced loss at 2^32-1. Truly stale iff L > 4 ms; receiver age = L + excess, excess "
        "= quantization + drift >= 0. P(L <= x) = 1 - exp(-g/s)(1 + g/s), g = x - 2 ms, s = 0.5 ms. Loss-count "
        "variance per frame p(1-p) + 2(l_b-l_g)^2 pi_g pi_b lambda/(1-lambda).",
        ["4000 frames encoded with the T149 codec, PCG64(152) link draws and PCG64(153) payloads",
         "200000-frame loss/latency run, PCG64(1520)"],
        "Receiver outputs (pending gaps, reorders, duplicates, stale flags) compared with the simulation ground truth "
        "and with expected counts computed from the closed-form latency distribution.",
        "Detected losses = true losses between the first and last received sequence; zero missed stale frames with the "
        "declared offset; false alarms and estimated-offset misses within 4 sigma of their expectations; long-run loss "
        "rate and mean latency within 4 sigma of the model.",
        "Simulate the link, run receivers with the declared and an estimated offset, compare with ground truth and "
        "predicted rates, run naive detectors, and check a long run against the model moments.",
        "Deferred research question: negative clock drift and loss-dependent latency (bursty congestion) in the link "
        "simulation; both are assumed away here, and a negative drift would turn false staleness alarms into misses",
        numerical_result=(f"{len(true_lost)} losses detected exactly ({exact.reordered} reordered, {exact.duplicates} "
                          f"duplicates); declared offset: {declared['missed']} of {declared['stale']} stale frames "
                          f"missed, {declared['false_alarms']} false alarms (expected "
                          f"{declared['expected_false_alarms']:.1f} +/- {declared['sigma']:.1f}); estimated offset "
                          f"(bias {biased['bias_ns'] / 1e6:.3f} ms) misses {biased['missed']} of {biased['stale']} "
                          f"(expected {biased['expected_missed']:.1f} +/- {biased['sigma']:.1f}); naive counts "
                          f"{study['naive_arrival']} (arrival order) and {study['naive_in_order']} (in order) against "
                          f"{len(true_lost)}; long-run loss rate {run['loss_rate']:.5f} vs {run['stationary_loss']:.5f} "
                          f"(z = {run['loss_z']:.2f}), latency z = {run['latency_z']:.2f}."),
        uncertainty=(f"Counts are exact given the seed; rate checks use exact variances and a 4-sigma threshold. At "
                     f"200000 frames the loss check detects a stationary-loss error above about {power:.4f} (4 sigma) "
                     "with probability at least one half; a loss_bad change from 0.5 to 0.1 (a shift of 0.0098) would "
                     "be detected almost surely, but at the 4000 frames of the detection stream it would not (2.4 "
                     "sigma)."),
        failure_modes_checked=["32-bit sequence wrap", "reordering", "duplicates", "burst loss", "timestamp "
                               "quantization", "clock drift", "clock offset estimation bias", "naive gap arithmetic "
                               "in arrival order", "non-modular differencing at the wrap", "simulator/model mismatch"],
        unresolved_assumptions=["Gaps of 2^31 or more are unsupported by serial arithmetic",
                                "Losses before the first and after the last received frame are undetectable",
                                "Drift is constant and positive (device clock slow against the host); a negative "
                                "drift would turn false alarms into misses",
                                "Latency is independent of the loss state"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T153
EXTERNAL = authority.AuthorizationRecord(issuer="plant-safety-office (hypothetical)", subject="cell-7 spindle",
                                         channels=("spindle_speed",), valid_from="2026-01-01T00:00:00Z",
                                         valid_until="2027-01-01T00:00:00Z", purpose="test fixture",
                                         signature="unverifiable-placeholder")


WRITE_CHANNELS = ("spindle_speed", "feed_override", "axis_x_setpoint", "coolant", "laser_power")
# Forged modules that open a machine write path; the scan must flag each.
FORGED_WRITE_PATHS = (
    ("ciw.lab.forged_bus", "import serial\nserial.Serial('/dev/ttyUSB0').write(b'S1200')\n", "device", "serial port"),
    ("ciw.lab.forged_bus", "with open('COM3', 'wb') as port:\n    port.write(b'S1200')\n", "device",
     "Windows COM device opened as a file"),
    ("ciw.lab.forged_bus", "from pymodbus.client import ModbusTcpClient\n", "device", "Modbus fieldbus client"),
    ("ciw.lab.forged_bus", "import socket\nsocket.create_connection(('plc', 502))\n", "undeclared_network",
     "raw socket to a PLC"),
    ("ciw.lab.forged_bus", "import asyncio\nasyncio.open_connection('plc', 502)\n", "undeclared_network",
     "asyncio stream to a PLC"),
)


@task("T153", changed_files=(MODULE, AUTHORITY, ARCH, DOC),
      regression_tests=(f"{TESTS}::test_t153_writes_disabled_by_default",))
def actuator_writes(ctx):
    from dataclasses import replace

    policy = authority.ActuatorWritePolicy()
    # The default gate refuses before it reads the channel or value, so one attempt per declared channel documents
    # the refusal; random values would add nothing.
    default_writes = [_refusal(f"write to {channel} on the default policy", "writes_disabled_by_default",
                               lambda channel=channel: policy.check_write(channel, 1.0, now=NOW))
                      for channel in WRITE_CHANNELS]
    routes = [
        _refusal("enable without a record", "authorization_missing",
                 lambda: policy.enable(None, channel="spindle_speed", now=NOW)),
        _refusal("enable with a plain dict", "authorization_wrong_type",
                 lambda: policy.enable(dict(EXTERNAL.__dict__), channel="spindle_speed", now=NOW)),
        _refusal("enable with a lab-issued record", "self_issued_authority",
                 lambda: policy.enable(replace(EXTERNAL, issuer="ciw.lab"), channel="spindle_speed", now=NOW)),
        _refusal("enable with an expired record", "authorization_expired",
                 lambda: policy.enable(replace(EXTERNAL, valid_until="2026-06-01T00:00:00Z"), channel="spindle_speed",
                                       now=NOW)),
        _refusal("enable with an expiry written with a +02:00 offset, one hour before the instant", "authorization_expired",
                 lambda: policy.enable(replace(EXTERNAL, valid_until="2026-09-23T01:00:00+02:00"),
                                       channel="spindle_speed", now=NOW)),
        _refusal("enable with an expiry half a second after the instant (fractional seconds; passes the window and "
                 "stops at the trust anchor)", "no_trust_anchor",
                 lambda: policy.enable(replace(EXTERNAL, valid_until="2026-09-23T00:00:00.500Z"),
                                       channel="spindle_speed", now=NOW)),
        _refusal("enable with the expiry 'tomorrow'", "malformed_timestamp",
                 lambda: policy.enable(replace(EXTERNAL, valid_until="tomorrow"), channel="spindle_speed", now=NOW)),
        _refusal("enable with an expiry without a UTC offset", "malformed_timestamp",
                 lambda: policy.enable(replace(EXTERNAL, valid_until="2027-01-01T00:00:00"), channel="spindle_speed",
                                       now=NOW)),
        _refusal("enable for a channel outside scope", "out_of_scope",
                 lambda: policy.enable(EXTERNAL, channel="laser_power", now=NOW)),
        _refusal("enable with an unsigned record", "unsigned_authorization",
                 lambda: policy.enable(replace(EXTERNAL, signature=None), channel="spindle_speed", now=NOW)),
        _refusal("enable with a well-formed external record", "no_trust_anchor",
                 lambda: policy.enable(EXTERNAL, channel="spindle_speed", now=NOW)),
        _refusal("lab issues an authorization record", "lab_cannot_issue_authority",
                 lambda: authority.issue_authorization(channels=("spindle_speed",))),
    ]
    bypassed = authority.ActuatorWritePolicy()
    object.__setattr__(bypassed, "enabled", True)
    mutated = bypassed.enabled is True
    after_flag = _refusal("write after forcing enabled=True in memory", "writes_disabled_by_default",
                          lambda: bypassed.check_write("spindle_speed", 1.0, now=NOW))
    object.__setattr__(bypassed, "authorization", EXTERNAL)
    after_record = _refusal("write after also forcing an external record in memory", "no_trust_anchor",
                            lambda: bypassed.check_write("spindle_speed", 1.0, now=NOW))
    scan = ctx.memo("t144-scan", arch.scan_package)
    paths = arch.write_paths(scan)
    forged_checks = []
    for name, text, kind, label in FORGED_WRITE_PATHS:
        flagged = name in arch.write_paths(arch.mutated_scan(scan, name, text))[kind]
        forged_checks.append(_check(f"forged module with a {label} flagged as a {kind.replace('_', ' ')} path",
                                    float(flagged), 1, "ge"))
    ctx.artifact_json("write-policy.json", {"default_policy": {"enabled": policy.enabled, "authorization": None},
                                            "default_writes": [{k: c[k] for k in ("reference", "expected_refusal",
                                                                                  "observed_refusal")}
                                                               for c in default_writes],
                                            "enable_routes": [{k: c[k] for k in ("reference", "expected_refusal",
                                                                                 "observed_refusal")} for c in routes],
                                            "verification_order": ["authorization_missing", "authorization_wrong_type",
                                                                   "self_issued_authority", "malformed_timestamp",
                                                                   "authorization_expired", "out_of_scope",
                                                                   "unsigned_authorization", "no_trust_anchor"]})
    ctx.artifact_json("write-path-scan.json", {"package_sha256": scan["package_sha256"],
                                               "device_libraries": sorted(arch.DEVICE_LIBRARIES),
                                               "network_libraries": sorted(arch.NETWORK_LIBRARIES),
                                               "network_calls": sorted(arch.NETWORK_CALLS),
                                               "declared_network": arch.DECLARED_NETWORK, "paths": paths})
    findings = [
        finding("Default policy refuses writes on every declared channel", "computational_pipeline",
                sum(c["passed"] for c in default_writes), {"checks": default_writes}, unit="refused channels",
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Enabling writes is refused on every route, including a well-formed external record",
                "computational_pipeline", sum(c["passed"] for c in routes), {"checks": routes}, unit="refused routes",
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("A frozen in-process policy object can be mutated", "computational_pipeline",
                {"flag_mutated": mutated, "writes_still_refused": after_flag["passed"] and after_record["passed"]},
                {"checks": [_check("object.__setattr__ changed enabled to True", float(mutated), 1, "ge"),
                            after_flag, after_record]},
                counterexample={"statement": "An in-process Python flag is an actuator authority boundary",
                                "witness": {"mutation": "object.__setattr__(policy, 'enabled', True)",
                                            "outcome": "flag changed; the gate still refused because it re-verifies the "
                                                       "authorization on every write"}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("No ciw module imports a device, serial or fieldbus library or names a device node, and network "
                "libraries appear only in the declared workbench servers (source scan)", "computational_pipeline",
                {"device_paths": sorted(paths["device"]), "undeclared_network": sorted(paths["undeclared_network"]),
                 "declared_network": sorted(paths["declared_network"])},
                {"checks": [_check("modules importing a device, serial, fieldbus or instrument library or naming a "
                                   "device node", len(paths["device"])),
                            _check("modules outside the declared servers importing a network library or opening a "
                                   "connection", len(paths["undeclared_network"])),
                            _check("modules that failed to parse", len(scan["unparsed"])), *forged_checks]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("The lab holds actuator write authority", "actuator_authority", None, {}),
        finding("Disabled-by-default software writes make the machine safe", "machine_safety", None, {}),
    ]
    fields = _fields(
        "A deny-by-default write policy refuses every write; enabling needs an externally issued, signed, in-scope, "
        "unexpired authorization verified against a trust anchor the lab does not hold, so no route opens it here; "
        "and a source scan finds no device, serial or fieldbus write path in the package and network paths only in "
        "the declared workbench servers, so no other code path reaches a machine that the scan can see.",
        "Gate(channel) = refuse unless enabled and verify(authorization, channel, now) passes; verify checks type, "
        "issuer, timestamp form (ISO-8601 with a UTC offset, parsed to instants), validity window, scope, signature "
        "and trust anchor in that order; the lab build has no trust anchor. Write paths: a module is a device path if "
        "it imports a device, serial, fieldbus, industrial-protocol or instrument library or names a device node "
        "(/dev/..., \\\\.\\..., COMn), and a network path if it imports a network library or calls a connection "
        "or server constructor; network paths are allowed only in declared modules.",
        ["one write per declared channel (five)", "twelve enabling routes", "fixed instant " + NOW,
         f"{len(scan['modules'])} parsed modules of the installed ciw package (source text only; package digest in "
         "write-path-scan.json)", f"{len(FORGED_WRITE_PATHS)} forged write-path modules"],
        "Refusal codes raised by the policy; source-scan hits per module; no actuator exists.",
        "Every default write refused; each route stops at its expected check, including offset, fractional-second and "
        "malformed timestamps; mutation of the frozen object does not open the gate; no device path and no "
        "undeclared network path in the package; every forged write path flagged.",
        "Attempt a write per channel on the default policy, try each enabling route, bypass the frozen dataclass, then "
        "scan the package for device and network write paths and replay forged ones.",
        "Deferred research question: enforce that any future write transport is routed through ActuatorWritePolicy "
        "(the scan flags a new transport, routing is not enforced), with a real signature scheme for authorization "
        "records; enforcement itself belongs in hardware interlocks outside this package",
        numerical_result=(f"{_passed(default_writes)}/{len(default_writes)} default writes refused; "
                          f"{_passed(routes)}/{len(routes)} enabling routes refused with their expected codes; "
                          "in-memory mutation succeeded but writes stayed refused; device paths "
                          f"{len(paths['device'])}, undeclared network paths {len(paths['undeclared_network'])}, "
                          f"declared network modules {', '.join(sorted(paths['declared_network'])) or 'none'}; "
                          f"{sum(c['passed'] for c in forged_checks)}/{len(forged_checks)} forged write paths flagged."),
        uncertainty="Exhaustive over the declared routes; software refusals inside one process can be bypassed by "
                    "code with the same privileges (the counterexample shows the mutation itself succeeds). The scan "
                    "is static: dynamic imports, exec, third-party libraries and spawned providers that open devices "
                    "are not seen, and the library lists are denylists.",
        failure_modes_checked=["missing, wrong-type, self-issued, expired, out-of-scope, unsigned and unanchored "
                               "authorizations", "expiry with a non-Z UTC offset", "fractional-second expiry",
                               "malformed or offset-less timestamps", "lab issuing authority", "frozen-object mutation",
                               "device, serial or fieldbus library import", "device node opened as a file",
                               "undeclared socket or connection"],
        unresolved_assumptions=["Real enforcement belongs in hardware interlocks and a separate controller process",
                                "Authorization record format and signature scheme are placeholders",
                                "No module routes writes through ActuatorWritePolicy because the package has no "
                                "write path to route; a future transport would be flagged by the scan, but routing it "
                                "through the policy is not enforced",
                                "The declared servers (ciw.server, ciw.cli, ciw.lab.mcp_server) talk to the "
                                "workbench's own clients; that they cannot reach a machine is judged from the absence "
                                "of device libraries, not proven"])
    return {"state": _state(findings), "fields": fields, "findings": findings}


# =================================================================== T154
def proposal_study() -> dict:
    """Jacobi heading proposals: analytic check on the sphere, nonlinear residual order on the torus."""
    sphere, torus = Sphere(1.0), Torus(2.0, 1.0)
    length = 1.3
    proposal = authority.heading_correction(sphere, (1.1, 0.2), 0.3, length, 0.01, steps=400)
    analytic = -math.cos(length) / math.sin(length) * 0.01
    u0, heading, L, steps = np.array([0.4, 0.3]), 0.5, 2.0, 200
    base = integrators.integrate_fixed(torus.geodesic_rhs, np.concatenate([u0, torus.unit_tangent(u0, heading)]),
                                       L, steps)[1]
    rows = []
    for d in (0.02, 0.01, 0.005):
        value = authority.heading_correction(torus, u0, heading, L, d, steps).value
        residual = []
        for h in (0.0, value):
            y0 = jacobi.perturbed_start(torus, u0, heading, lateral=d, heading_change=h)
            states = integrators.integrate_fixed(torus.geodesic_rhs, y0, L, steps)[1]
            residual.append(float(jacobi.normal_separation(torus, base, states)[-1]))
        rows.append({"lateral": d, "heading_proposal": value, "uncorrected": residual[0], "corrected": residual[1]})
    lateral = [r["lateral"] for r in rows]
    corrected_order = integrators.observed_order(lateral, [abs(r["corrected"]) for r in rows])
    uncorrected_order = integrators.observed_order(lateral, [abs(r["uncorrected"]) for r in rows])
    return {"sphere_proposal": proposal.value, "sphere_analytic": analytic, "sphere_record": proposal.record(),
            "torus": rows, "corrected_order": corrected_order, "uncorrected_order": uncorrected_order}


@task("T154", changed_files=(MODULE, AUTHORITY, ARCH, DOC),
      regression_tests=(f"{TESTS}::test_t154_control_outputs_are_proposals",))
def control_proposals(ctx):
    from dataclasses import replace

    study = ctx.memo("t154-proposals", proposal_study)
    record = study["sphere_record"]
    proposal = authority.ControlProposal(**{k: record[k] for k in ("channel", "value", "unit", "produced_by",
                                                                  "inputs_sha256", "rationale")})
    sphere = Sphere(1.0)
    external = replace(EXTERNAL, channels=("heading_offset",))
    refusals = [
        _refusal("proposal at the conjugate point L = pi", "conjugate_point",
                 lambda: authority.heading_correction(sphere, (1.1, 0.2), 0.3, math.pi, 0.01, steps=400)),
        _refusal("construct a control output with status 'command'", "proposal_status_fixed",
                 lambda: replace(proposal, status="command")),
        _refusal("convert a proposal without authorization", "proposal_not_authorized",
                 lambda: authority.to_command(proposal, now=NOW)),
        _refusal("convert with a lab-issued authorization", "self_issued_authority",
                 lambda: authority.to_command(proposal, replace(external, issuer="ciw.lab.T154"), now=NOW)),
        _refusal("convert with a well-formed external authorization", "no_trust_anchor",
                 lambda: authority.to_command(proposal, external, now=NOW)),
        _refusal("convert a raw dict posing as a proposal", "not_a_proposal",
                 lambda: authority.to_command(dict(record), external, now=NOW)),
        _refusal("proposal with a nonfinite value", "nonfinite_proposal",
                 lambda: replace(proposal, value=float("nan"))),
    ]
    # In-process immutability is not a boundary: force the status and check that nothing downstream accepts it.
    tampered = authority.ControlProposal(**{k: record[k] for k in ("channel", "value", "unit", "produced_by",
                                                                   "inputs_sha256", "rationale")})
    object.__setattr__(tampered, "status", "command")
    forced = tampered.status == "command"
    tamper_checks = [
        _refusal("record() of a proposal whose status was forced to 'command'", "proposal_status_tampered",
                 tampered.record),
        _refusal("convert a status-forced proposal without authorization", "proposal_status_tampered",
                 lambda: authority.to_command(tampered, now=NOW)),
        _refusal("convert a status-forced proposal with an external authorization", "proposal_status_tampered",
                 lambda: authority.to_command(tampered, external, now=NOW)),
    ]
    # Scope: which control-like outputs of the workbench are proposals. The inventory is checked against a keyword
    # scan of the package, which must find no module outside it; a forged module shows the scan can.
    scan = ctx.memo("t144-scan", arch.scan_package)
    named = arch.control_term_modules(scan)
    inventoried = {module for entry in authority.CONTROL_OUTPUTS for module in entry["modules"]}
    outside = sorted(set(named) - inventoried)
    stale = sorted(inventoried - set(scan["modules"]))
    forged_named = arch.control_term_modules(arch.mutated_scan(
        scan, "ciw.lab.forged_controller", "def correct(axis):\n    return {'axis': axis, 'setpoint': 0.25}\n"))
    uncovered = [entry for entry in authority.CONTROL_OUTPUTS if not entry["proposal"]]
    ctx.artifact_json("proposal-record.json", record)
    ctx.artifact_json("proposal-study.json", {k: v for k, v in study.items() if k != "sphere_record"})
    ctx.artifact_json("control-outputs.json", {"inventory": list(authority.CONTROL_OUTPUTS),
                                               "control_terms": list(arch.CONTROL_TERMS),
                                               "modules_naming_control_terms": named,
                                               "package_sha256": scan["package_sha256"]})
    rows = study["torus"]
    ctx.artifact_text("proposal-residual.svg", svg.line_plot(
        [("uncorrected", [r["lateral"] for r in rows], [abs(r["uncorrected"]) for r in rows]),
         ("with heading proposal", [r["lateral"] for r in rows], [abs(r["corrected"]) for r in rows])],
        title="Normal separation at L = 2 on the torus", xlabel="lateral offset d", ylabel="|separation at L|",
        logx=True, logy=True))
    sphere_error = abs(study["sphere_proposal"] - study["sphere_analytic"]) / abs(study["sphere_analytic"])
    findings = [
        finding("Control outputs built by this section are constructed with status proposal and cannot be converted "
                "to a command here",
                "computational_pipeline", {"status": proposal.status, "refusals": sum(c["passed"] for c in refusals)},
                {"checks": refusals}, uncertainty=EXACT_U, tolerance=EXACT),
        finding("A frozen control proposal's status can be forced in memory, and the forced object is refused",
                "computational_pipeline", {"status_forced": forced, "refusals": sum(c["passed"] for c in tamper_checks)},
                {"checks": [_check("object.__setattr__ changed status to 'command'", float(forced), 1, "ge"),
                            *tamper_checks]},
                counterexample={"statement": "A frozen dataclass status field keeps every control output a proposal",
                                "witness": {"mutation": "object.__setattr__(proposal, 'status', 'command')",
                                            "outcome": "status changed; record() and to_command re-check it and refuse "
                                                       "(proposal_status_tampered)"}},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Jacobi heading proposal cancels a lateral offset to second order", "numerical",
                {"corrected_order": study["corrected_order"], "uncorrected_order": study["uncorrected_order"]},
                {"generator": {"name": "torus R=2 r=1, u0=(0.4, 0.3), heading 0.5, L=2, d in {0.02, 0.01, 0.005}"},
                 "checks": [_check("corrected residual order - 2", study["corrected_order"] - 2.0, 0.2,
                                   kind="self_convergence"),
                            _check("uncorrected residual order - 1", study["uncorrected_order"] - 1.0, 0.2,
                                   kind="self_convergence"),
                            _check("sphere proposal relative error against -cot(L) d", sphere_error, 1e-8,
                                   kind="analytic")]},
                unit="observed order",
                uncertainty={"kind": "reference_error",
                             "value": max(abs(study["corrected_order"] - 2.0), abs(study["uncorrected_order"] - 1.0)),
                             "basis": "departure of the least-squares orders (three offsets) from their asymptotic "
                                      "values, a pre-asymptotic spread; RK4 error at 200 steps is far below the "
                                      "residuals"},
                tolerance={"abs": 1e-3, "rel": 0}),
        finding("Control-like outputs named in the package are all inventoried (keyword scan)", "computational_pipeline",
                {"outputs": [entry["output"] for entry in authority.CONTROL_OUTPUTS],
                 "proposals": [entry["output"] for entry in authority.CONTROL_OUTPUTS if entry["proposal"]],
                 "modules_naming_control_terms": sorted(named)},
                {"checks": [_check("modules naming a control-like output (stop request, abort, setpoint, command "
                                   "conversion, gain change) outside the inventory", len(outside)),
                            _check("inventoried modules missing from the package", len(stale)),
                            _check("forged module returning a setpoint found outside the inventory",
                                   float("ciw.lab.forged_controller" in forged_named), 1, "ge")]},
                uncertainty=EXACT_U, tolerance=EXACT),
        finding("Every control-like output of the workbench is a ControlProposal", "computational_pipeline", None,
                {"notes": "Only the T154 heading correction is a ControlProposal. Not covered: "
                          + "; ".join(f"{entry['output']} ({entry['task']}: {entry['route']})" for entry in uncovered)},
                expected_not_established=True),
        finding("Heading proposals are authorized for execution as actuator commands", "actuator_authority", None, {}),
        finding("Applying the proposed heading corrections on a machine is safe", "machine_safety", None, {}),
    ]
    fields = _fields(
        "Control outputs can be computed and retained as proposals (frozen objects whose status is re-checked on "
        "every use), with content identities, while every conversion to a command is refused without separate "
        "authorization; the geometric proposal itself is sound to the order the Jacobi model predicts. This holds "
        "for the outputs built here; an inventory records which other control-like outputs of the workbench are "
        "not proposals.",
        "Normal Jacobi field j(L) = j_lat(L) d + j_head(L) h; proposal h = -j_lat(L) d / j_head(L), undefined at a "
        "conjugate point (j_head(L) = 0). On the unit sphere h = -cot(L) d. With h applied, the residual separation "
        "is O(d^2).",
        ["Unit sphere, u0=(1.1, 0.2), heading 0.3, L=1.3 and L=pi", "Torus R=2, r=1, u0=(0.4, 0.3), heading 0.5, "
         "L=2, 200 RK4 steps, d in {0.02, 0.01, 0.005}"],
        "Nonlinear geodesic integration of the offset start with and without the proposal; separation g(du, N) at L.",
        "Proposal matches -cot(L) d on the sphere; corrected residual order 2, uncorrected order 1; all conversions "
        "refused, including of an object whose status was forced in memory; every module naming a control-like "
        "output is in the inventory.",
        "Compute proposals, verify them by integration, then attempt construction as a command and conversion under "
        "each authorization route; check the control-output inventory against a keyword scan of the package.",
        "Deferred research question: a proposal-and-authorization record for stop requests to an independent safety "
        "function (the servo-axis abort declared in ciw.lab.lyapunov_research is not a ControlProposal), and a "
        "hardware-in-the-loop authority design review; production acceptance and actuation authority stay outside "
        "the system",
        numerical_result=(f"Sphere proposal relative error {sphere_error:.2e}; residual orders corrected "
                          f"{study['corrected_order']:.3f}, uncorrected {study['uncorrected_order']:.3f}; "
                          f"{sum(c['passed'] for c in refusals)}/{len(refusals)} refusals; a status forced in memory "
                          f"is refused by {sum(c['passed'] for c in tamper_checks)}/{len(tamper_checks)} paths; "
                          f"{len(authority.CONTROL_OUTPUTS)} control-like outputs inventoried, "
                          f"{len(authority.CONTROL_OUTPUTS) - len(uncovered)} of them proposals; modules naming control "
                          f"terms outside the inventory: {len(outside)}."),
        uncertainty="Orders from three offsets (least squares); RK4 error at 200 steps is far below the residuals.",
        failure_modes_checked=["conjugate point", "status forgery at construction", "status forced in memory "
                               "(object.__setattr__)", "missing, self-issued and unanchored authorization",
                               "dict posing as proposal", "nonfinite value"],
        unresolved_assumptions=["The proposal is a kinematic heading correction on an ideal surface, not a validated "
                                "controller", "Actuator dynamics, limits and latency are not modelled",
                                "Scope: only control outputs built in this section are ControlProposals; the T114 "
                                "servo-axis abort (a stop request to the bench's independent safety function, declared "
                                "in ciw.lab.lyapunov_research) is not, and the T149, T151 and T153 command paths are "
                                "refused rather than proposed",
                                "The inventory is checked against a keyword scan of code and non-docstring strings; "
                                "a control output named with other words is not found"])
    return {"state": _state(findings), "fields": fields, "findings": findings}
