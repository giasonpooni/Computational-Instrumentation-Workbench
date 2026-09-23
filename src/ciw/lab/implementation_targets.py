"""Section 10 (T142-T154): Rust, Python, Julia, C++, GPU and FPGA implementation targets.

Scope: which ciw.lab kernels merit a Rust port (exact operation counts and
interpreter dispatch counts; timings retained only as artifacts), which
industrial interfaces need C/C++ behind pinned subprocess boundaries, an
import-graph check that evidence code stays in Python, the Julia pin plan,
one canonical JSON encoding verified against CIW's Python encoders and a Rust
implementation compiled at run time, a CPU/GPU comparison harness exercised
CPU-against-CPU, deterministic reduction policies, a telemetry-only FPGA frame
format with identity, compatibility and rollback records, a seeded link
simulation, and the refusal boundary for actuator writes and control outputs.

Non-claims: no GPU, FPGA, Julia runtime or industrial library runs here; all
bitstreams and link statistics are synthetic; Rust agreement is same-origin
(CIW-authored) evidence; no finding establishes physical performance, machine
safety, industrial readiness, production acceptance or actuator authority.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
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
from . import integrators, jacobi, svg
from .evidence import finding
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


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, action):
    """Run ``action`` and record which refusal code it raised (None when it did not refuse)."""
    try:
        action()
        observed = None
    except (ValueError, PermissionError) as exc:
        observed = getattr(exc, "code", type(exc).__name__)
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _fields(hypothesis, model, inputs, observation, invariant, experiment, next_task, **extra):
    fields = {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
              "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
              "recommended_next_task": next_task, "failure_modes_checked": [], "unresolved_assumptions": []}
    fields.update(extra)
    return fields


def _noop():
    return None


def _source_digest(relative):
    from .runner import source_digest
    return source_digest(relative)


# =================================================================== T142
REFERENCE_STEPS = 2000
REFERENCE_UPDATES = 2000


def kernel_profile() -> dict:
    """Exact operation counts, dispatch counts and core agreement of the four kernel families."""
    torus = Torus(2.0, 1.0)
    f = jacobi.rhs(torus)
    rng = np.random.Generator(np.random.PCG64(142))
    S = kernels.Scalar

    def scal(v):
        return [S(x) for x in v]

    diffs = {"geodesic_rhs": 0.0, "jacobi_rhs": 0.0, "rk4_step": 0.0, "jacobi_transfer": 0.0, "kalman_update": 0.0}
    counts, count_sets = {}, {name: [] for name in diffs}
    for trial in range(2):
        y = np.concatenate([rng.uniform(-2.0, 2.0, 2), rng.uniform(-1.0, 1.0, 2)])
        y8 = np.concatenate([y, rng.uniform(-1.0, 1.0, 4)])
        out, c = kernels.count_ops(kernels.torus_geodesic_rhs, scal(y))
        diffs["geodesic_rhs"] = max(diffs["geodesic_rhs"], float(np.max(np.abs(kernels.values(out) - torus.geodesic_rhs(y)))))
        count_sets["geodesic_rhs"].append(c)
        out, c = kernels.count_ops(kernels.torus_jacobi_rhs, scal(y8))
        diffs["jacobi_rhs"] = max(diffs["jacobi_rhs"], float(np.max(np.abs(kernels.values(out) - f(y8)))))
        count_sets["jacobi_rhs"].append(c)
        out, c = kernels.count_ops(kernels.rk4_step, kernels.torus_jacobi_rhs, scal(y8), S(0.01))
        diffs["rk4_step"] = max(diffs["rk4_step"], float(np.max(np.abs(kernels.values(out) - integrators.step_rk4(f, y8, 0.01)))))
        count_sets["rk4_step"].append(c)
        out, c = kernels.count_ops(kernels.transfer, kernels.torus_jacobi_rhs, scal(y8), 1.0, 10)
        _, states = integrators.integrate_fixed(f, y8, 1.0, 10)
        diffs["jacobi_transfer"] = max(diffs["jacobi_transfer"], float(np.max(np.abs(kernels.values(out) - states[-1]))))
        count_sets["jacobi_transfer"].append(c)
        x, P, z, H, R = kernels.kalman_case(seed=142 + trial)
        x_ref, P_ref = kernels.kalman_update(x, P, z, H, R)
        (x_s, P_s), c = kernels.count_ops(kernels.kalman_update_scalar, scal(x), [scal(r) for r in P], scal(z),
                                          [scal(r) for r in H], [scal(r) for r in R])
        P_s = np.array([[v.v for v in row] for row in P_s])
        diffs["kalman_update"] = max(diffs["kalman_update"], float(np.max(np.abs(kernels.values(x_s) - x_ref))),
                                     float(np.max(np.abs(P_s - P_ref) / np.max(np.abs(P_ref)))))
        count_sets["kalman_update"].append(c)
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
                  "jacobi_transfer_fixed": transfer_d[10] - 10 * per_step, "kalman_update": measured(kernels.kalman_update, x, P, z, H, R)}
    linearity = (transfer_d[30] - transfer_d[20]) - (transfer_d[20] - transfer_d[10])
    return {"counts": counts, "agreement": diffs, "count_drift": count_drift, "rk4_formula": rk4_formula,
            "transfer_formula": transfer_formula, "dispatches": dispatches, "transfer_dispatches": transfer_d,
            "dispatch_linearity": linearity, "baseline": baseline}


def kernel_ranking(profile: dict) -> list:
    """Rank port units by Python dispatches a port removes from one reference experiment.

    Reference experiment: a Jacobi transfer of 2000 RK4 steps (8000 right-hand
    sides) and 2000 Kalman updates. Porting a unit alone removes its own
    dispatches but keeps any callbacks into Python.
    """
    d, c = profile["dispatches"], profile["counts"]
    transfer_total = REFERENCE_STEPS * d["jacobi_transfer_per_step"] + d["jacobi_transfer_fixed"]
    rows = [
        {"kernel": "jacobi_transfer (fused geodesic + Jacobi RK4 loop)", "calls_per_experiment": 1,
         "flops_per_call": REFERENCE_STEPS * c["rk4_step"]["flops"] + 1, "dispatches_per_call": transfer_total,
         "removable_dispatches": transfer_total - 1, "determinism_need": "high",
         "determinism_note": "retained trajectories; bitwise replay needs a fixed order over thousands of dependent steps"},
        {"kernel": "geodesic_rhs (generic embedded Christoffel contraction)", "calls_per_experiment": 4 * REFERENCE_STEPS,
         "flops_per_call": c["geodesic_rhs"]["flops"], "dispatches_per_call": d["geodesic_rhs"],
         "removable_dispatches": 4 * REFERENCE_STEPS * (d["geodesic_rhs"] - 1), "determinism_need": "high",
         "determinism_note": "einsum/inverse order must be fixed for bitwise replay"},
        {"kernel": "kalman_update (Joseph form, n=4, m=2)", "calls_per_experiment": REFERENCE_UPDATES,
         "flops_per_call": c["kalman_update"]["flops"], "dispatches_per_call": d["kalman_update"],
         "removable_dispatches": REFERENCE_UPDATES * (d["kalman_update"] - 1), "determinism_need": "medium",
         "determinism_note": "covariance symmetry and positive definiteness; small fixed-size reductions"},
        {"kernel": "rk4_step alone (right-hand side stays in Python)", "calls_per_experiment": REFERENCE_STEPS,
         "flops_per_call": kernels.rk4_combination_ops(8), "dispatches_per_call": d["rk4_step"],
         "removable_dispatches": REFERENCE_STEPS * max(0, d["rk4_step"] - 1 - 4 * d["jacobi_rhs"]),
         "determinism_need": "high", "determinism_note": "stage combination order; negligible alone"},
    ]
    for row in rows:
        row["flops_per_dispatch"] = row["flops_per_call"] / max(row["dispatches_per_call"], 1)
    order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda row: (-row["removable_dispatches"], order[row["determinism_need"]]))
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


@task("T142", changed_files=(MODULE, KERNELS, SERIAL, DOC),
      regression_tests=(f"{TESTS}::test_t142_kernel_counts_and_ranking", f"{TESTS}::test_rust_fused_sphere_loop"))
def rust_kernels(ctx):
    profile = ctx.memo("t142-profile", kernel_profile)
    ranking = kernel_ranking(profile)
    timings = _kernel_timings(profile)
    build = serial.rust_build()
    rust = rust_sphere_agreement() if build["available"] else None
    ctx.artifact_json("kernel-profile.json", {k: v for k, v in profile.items()})
    ctx.artifact_json("kernel-ranking.json", ranking)
    ctx.artifact_json("kernel-timings.json", {"timings": timings, "rust_fused_sphere": rust,
                                              "rust_build": {k: v for k, v in build.items() if k != "binary"}})
    ctx.artifact_text("kernel-timings.svg", svg.line_plot(
        [(name, [profile["counts"][name]["flops"]], [timings["seconds_per_call"][name] * 1e6])
         for name in ("geodesic_rhs", "jacobi_rhs", "rk4_step", "kalman_update")],
        title="Per-call time against exact flops (this machine)", xlabel="flops per call", ylabel="microseconds per call",
        logx=True, logy=True))
    max_diff = max(profile["agreement"].values())
    counts_finding = finding(
        "Exact floating-point operation counts of the scalar kernel restatements", "numerical",
        profile["counts"],
        {"generator": {"name": "PCG64 torus states and Kalman case", "seed": 142},
         "checks": [_check("scalar restatements against core NumPy kernels (max abs difference)", max_diff, 1e-12,
                           kind="cross_implementation"),
                    _check("RK4 step count minus 4*RHS - (13n+3), n=8", profile["counts"]["rk4_step"]["flops"]
                           - profile["rk4_formula"]),
                    _check("10-step transfer count minus 10*step - 1", profile["counts"]["jacobi_transfer"]["flops"]
                           - profile["transfer_formula"]),
                    _check("count differences between two random states (branch-free kernels)", profile["count_drift"])]},
        unit="operations per call", tolerance=EXACT)
    dispatch_finding = finding(
        "Interpreter calls issued by ciw code per kernel call", "computational_pipeline", profile["dispatches"],
        {"checks": [_check("transfer dispatches are linear in steps: D(30)-D(20) - (D(20)-D(10))",
                           profile["dispatch_linearity"])]},
        unit="calls", tolerance={"abs": 0, "rel": 0.25})
    ranking_finding = finding(
        "Ranked Rust port recommendation", "computational_pipeline", [row["kernel"] for row in ranking],
        {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#kernel-ranking: rank by Python dispatches removed per "
                       "reference experiment (2000 RK4 steps, 2000 Kalman updates), tie-break by determinism need"},
        tolerance=EXACT)
    if rust is not None:
        rust_finding = finding(
            "Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer on the unit sphere", "numerical",
            rust["max_abs_difference"],
            {"generator": {"name": "sphere geodesic u0=(1.1, 0.2), heading 0.3, L=3, 600 RK4 steps"},
             "checks": [_check("Rust against ciw.lab.jacobi.transfer final state", rust["max_abs_difference"], 1e-11,
                               kind="cross_implementation"),
                        _check("Rust Jacobi columns against cos(s), sin(s)", rust["rust_jacobi_error"], 1e-8,
                               kind="analytic")]},
            tolerance={"abs": 1e-11, "rel": 0})
    else:
        rust_finding = finding("Rust fused RK4 loop reproduces ciw.lab.jacobi.transfer on the unit sphere", "numerical",
                               None, {}, expected_not_established=True)
    readiness = finding("Rust ports of the ranked kernels are ready for industrial deployment", "industrial_readiness",
                        None, {})
    top = ranking[0]
    fields = _fields(
        "Python dispatch overhead, not arithmetic, dominates the geodesic/Jacobi kernels, so the fused transfer loop is "
        "the best Rust target; the Kalman update and a standalone RK4 step are poor targets.",
        "Scalar restatements on a counting number type give exact flop counts; RK4 on an n-vector adds 13n+3 flops to "
        "four right-hand sides; interpreter dispatches are calls issued from ciw frames (profile hook).",
        ["Torus R=2, r=1 (generic embedded path); two PCG64(142) states", "Joseph-form Kalman update n=4, m=2",
         "Unit sphere transfer u0=(1.1, 0.2), heading 0.3, L=3, 600 steps for the Rust probe"],
        "Deterministic counts of arithmetic operations and interpreter calls; wall-clock timings are observed only for "
        "the artifact.",
        "Counts are state-independent and exact; RK4 and transfer counts decompose as predicted; dispatches grow "
        "linearly with steps; a Rust port of the fused loop matches the core trajectory to rounding level.",
        "Count operations of scalar restatements, compare them with the core kernels, profile dispatches, rank port "
        "units by removable dispatches, compile the Rust probe and compare its fused sphere loop.",
        "T147 (compare a ported kernel against the CPU reference under the tolerance policy), then T148 for its reductions",
        numerical_result=(f"Flops per call: geodesic RHS {profile['counts']['geodesic_rhs']['flops']}, Jacobi RHS "
                          f"{profile['counts']['jacobi_rhs']['flops']}, RK4 step {profile['counts']['rk4_step']['flops']}, "
                          f"Kalman update {profile['counts']['kalman_update']['flops']}; dispatches per call "
                          f"{profile['dispatches']['geodesic_rhs']}, {profile['dispatches']['jacobi_rhs']}, "
                          f"{profile['dispatches']['rk4_step']}, {profile['dispatches']['kalman_update']}. Rank 1: "
                          f"{top['kernel']} ({top['removable_dispatches']} removable dispatches per reference experiment)."
                          + (f" Rust fused sphere loop differs from the core by {rust['max_abs_difference']:.2e}."
                             if rust else " Rust probe unavailable.")),
        uncertainty=("Counts are exact for the scalar restatements, which use common subexpressions the NumPy core "
                     "does not (the core repeats trigonometry and forms the full metric), so they count the port "
                     "specification rather than the core's executed operations. Dispatch counts depend on the Python "
                     "and NumPy versions (regression tolerance 25%); operator applications on arrays are not calls "
                     "and are not counted. Timings are machine-specific and appear only in artifacts."),
        failure_modes_checked=["scalar restatement diverging from the core kernel", "branch-dependent counts",
                               "RK4 decomposition formula error (13n+3)", "nonlinear dispatch growth",
                               "Rust port disagreeing with the core or the exact Jacobi field",
                               "rustc unavailable (finding recorded as not established)"],
        unresolved_assumptions=["Removable dispatches are a proxy for Python overhead; real speedups need a timed "
                                "port on the target machine", "The reference experiment sizes (2000 steps, 2000 "
                                "updates) are declared, not measured workloads",
                                "Only the sphere fused loop was ported, with closed-form sphere Christoffel symbols; "
                                "the generic embedded path is not, so the retained Python/Rust timing ratio overstates "
                                "what a generic port would gain"],
        provider_runtime_identity=_runtime_identity((MODULE, KERNELS, SERIAL), build))
    return {"state": "completed" if rust is not None else "partial", "fields": fields,
            "findings": [counts_finding, dispatch_finding, ranking_finding, rust_finding, readiness]}


def _runtime_identity(changed, build=None):
    from .runner import builtin_identity
    identity = builtin_identity(changed)
    if build is not None:
        identity["rust_probe"] = dict(build.get("identity") or {}, available=build["available"])
    return identity


# =================================================================== T143
PYTHON_BINDINGS = ("open3d", "OCC", "asyncua", "opcua", "pysoem", "pypylon", "PySpin", "vmbpy", "pcl")


@task("T143", changed_files=(MODULE, ARCH, DOC), regression_tests=(f"{TESTS}::test_t143_interface_inventory",))
def cpp_interfaces(ctx):
    entries = arch.validate_inventory(arch.INTERFACES)
    availability = {name: ctx.available(f"module:{name}") for name in PYTHON_BINDINGS}
    ctx.artifact_json("interface-inventory.json", {"interfaces": list(entries), "boundaries": sorted(arch.BOUNDARIES),
                                                   "python_bindings_present_here": availability})
    base = dict(entries[0])
    mutations = (("boundary", "in_process_binding", "in_process_binding"),
                 ("direction", "read_write", "write_capable_direction"),
                 ("write_path", "enabled", "write_path_enabled"),
                 ("identity", [], "incomplete_entry"))
    checks = [_refusal(f"inventory entry with {key}={value!r}", code,
                       lambda key=key, value=value: arch.validate_inventory([dict(base, **{key: value})]))
              for key, value, code in mutations]
    findings = [
        finding("Industrial interfaces that need C/C++ libraries behind a pinned subprocess boundary",
                "computational_pipeline", [entry["name"] for entry in entries],
                {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#industrial-interfaces"}, tolerance=EXACT),
        finding("Inventory validator refuses in-process bindings, write-capable directions and unpinned entries",
                "computational_pipeline", sum(check["passed"] for check in checks), {"checks": checks},
                unit="refused mutations", tolerance=EXACT),
        finding("The listed interfaces are qualified for plant integration", "industrial_readiness", None, {}),
        finding("Vendor camera SDK acquisition meets its timing on real cameras", "sensor_performance", None, {}),
    ]
    fields = _fields(
        "OPC UA, EtherCAT, vendor camera SDKs, PCL/Open3D and OpenCASCADE need C/C++ libraries, and each can sit "
        "behind a pinned subprocess boundary that exchanges retained bytes, leaving evidence code in Python.",
        "Design inventory: interface -> (native libraries, reason, boundary, direction, write path, identity pins); "
        "validation rules: boundary = pinned_subprocess, direction in {read_only, geometry_exchange}, write path absent "
        "or disabled.",
        ["Declared inventory in ciw.lab.implementation_targets_architecture.INTERFACES",
         "Python binding availability probe (artifact only)"],
        "No library was linked or executed; availability is probed with importlib only.",
        "Every entry validates; every mutated entry is refused with its specific code.",
        "Validate the inventory and four mutated entries; probe Python bindings for the artifact.",
        "T144 (verify the Python orchestration boundary in the package import graph)",
        numerical_result=f"{len(entries)} interfaces inventoried; {findings[1]['value']}/4 mutations refused.",
        uncertainty="Analytic design record; library capabilities are stated from vendor documentation, not tested.",
        failure_modes_checked=["in-process binding", "write-capable direction", "enabled write path",
                               "entry without identity pins"],
        unresolved_assumptions=["Vendor SDK licensing and platform support were not checked",
                                "EtherCAT monitoring without a write path may still need a master that owns the bus; "
                                "a passive tap is assumed", "No provider executable exists yet for any entry"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T144
@task("T144", changed_files=(MODULE, ARCH, DOC), regression_tests=(f"{TESTS}::test_t144_architecture_scan",))
def python_orchestration(ctx):
    scan = arch.scan_package()
    modules = scan["modules"]
    found = arch.violations(scan)
    evidence = arch.closure(modules, arch.EVIDENCE_MODULES)
    evidence_violations = [v for v in found if v[0].startswith("evidence_")]
    spawners = sorted(name for name, info in modules.items() if info["spawn_lines"])
    mention_only = sorted(name for name, info in modules.items() if info["mentions_subprocess"] and not info["spawn_lines"])
    witness = next((name for name in mention_only if not name.startswith("ciw.lab")), None)
    ctx.artifact_json("import-graph.json", {name: info["imports"] for name, info in modules.items()})
    ctx.artifact_json("architecture-scan.json", {
        "evidence_closure": evidence, "violations": found, "spawning_modules": {n: modules[n]["spawn_lines"] for n in spawners},
        "native_loading": {n: i["native"] for n, i in modules.items() if i["native"]},
        "mention_subprocess_without_spawn": mention_only, "unparsed_modules": scan["unparsed"],
        "compiled_extensions": scan["compiled_extensions"], "modules_scanned": len(modules)})
    evidence_source = (arch.PACKAGE_ROOT / "lab" / "evidence.py").read_text(encoding="utf-8")
    report_source = (arch.PACKAGE_ROOT / "lab" / "report.py").read_text(encoding="utf-8")
    forged = (("ciw.lab.evidence", evidence_source + "\nimport ctypes\n", "evidence_native_loading"),
              ("ciw.lab.report", report_source + "\nimport numpy\n", "evidence_closure_not_stdlib"),
              ("ciw.lab.evidence", evidence_source + "\nimport subprocess\nsubprocess.run(['solver'])\n",
               "evidence_spawns_process"),
              ("ciw.lab.forged_provider", "import subprocess\nsubprocess.run('solver --fast', shell=True)\n",
               "shell_invocation"),
              ("ciw.lab.forged_provider", "import subprocess\nsubprocess.run(['solver'])\n",
               "spawn_without_runtime_identity"))
    mutation_checks = []
    for name, text, rule in forged:
        rules = {v[0] for v in arch.violations(arch.mutated_scan(scan, name, text)) if v[1] == name}
        mutation_checks.append({"reference_kind": "refusal", "reference": f"forged {name} ({rule})",
                                "expected_refusal": rule, "observed_refusal": rule if rule in rules else ",".join(sorted(rules)) or None,
                                "passed": rule in rules})
    findings = [
        finding("Evidence and identity closure is standard-library Python with no native loading or process spawns",
                "computational_pipeline", evidence,
                {"checks": [_check("evidence-closure rule violations", len(evidence_violations))]}, tolerance=EXACT),
        finding("Package-wide boundary rules hold (no shell spawns, identity-recording spawners, native loading only in "
                "declared hardware probes, no compiled extensions)", "computational_pipeline", len(found),
                {"checks": [_check("architecture rule violations across the scanned package", len(found)),
                            _check("modules that failed to parse", len(scan["unparsed"]))]},
                unit="violations", tolerance=EXACT),
        finding("Scanner flags forged modules that cross the boundary", "computational_pipeline",
                sum(c["passed"] for c in mutation_checks), {"checks": mutation_checks}, unit="detected mutations",
                tolerance=EXACT),
        finding("Text search for 'subprocess' finds modules that spawn no process", "computational_pipeline",
                {"witness": witness},
                {"checks": [_check("modules outside ciw.lab mentioning subprocess without a spawn call",
                                   sum(not n.startswith("ciw.lab") for n in mention_only), 1, "ge")]},
                counterexample={"statement": "A text search for 'subprocess' identifies the process-spawning modules",
                                "witness": {"module": witness, "reason": "mentions ciw.adapters.subprocess or quotes "
                                            "subprocess in text but has no spawn call"}},
                tolerance=EXACT),
        finding("Keeping native code behind subprocess boundaries makes machine interfaces safe", "machine_safety",
                None, {}),
    ]
    fields = _fields(
        "The evidence and identity layer (ciw.lab.evidence, ciw.lab.report, ciw.core.identities and their ciw "
        "imports) is pure standard-library Python; numerical providers are reached through argument-vector process "
        "spawns from modules that record runtime identities; native loading is confined to declared hardware probes.",
        "Directed import graph over parsed modules; transitive closure from the evidence roots; rule set R = "
        "{closure stdlib-only, no native/spawn in closure, native loading within allowlist, no shell=True, spawners "
        "reference a runtime identity, no compiled extensions}.",
        [f"{len(modules)} parsed modules of the installed ciw package (source text only, not imported)"],
        "Static source analysis with ast; nothing from the scanned modules is imported or executed.",
        "Zero violations on the real package; every forged mutation violates its intended rule.",
        "Scan the package, compute the evidence closure, evaluate the rules, then replay five forged mutations.",
        "T146 (fix one canonical serialization for identities crossing the language boundary)",
        numerical_result=(f"Evidence closure {evidence}; {len(found)} violations over {len(modules)} modules; "
                          f"{len(spawners)} spawning modules; {findings[2]['value']}/5 forged mutations detected."),
        uncertainty=("Static scan: dynamic imports, exec, importlib and spawns through third-party libraries are not "
                     "seen; the identity-token rule is a heuristic that a module mentions revision, source_tree, "
                     "runtime_identity, sha256 or digest, not proof that the spawned binary is pinned."),
        failure_modes_checked=["third-party import in evidence closure", "native loading", "process spawn in evidence",
                               "shell=True", "spawn without identity", "compiled extension", "unparseable module"],
        unresolved_assumptions=["Modules added concurrently by other sections are scanned as found at run time",
                                "Hardware energy probes (ciw.energy_cuda, ciw.energy_nvml) load drivers in-process by "
                                "declared exception"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T145
JULIA_PLAN = {
    "reference": "docs/JULIA_SP1.md (CIW -> SCR execution boundary)",
    "candidate_version": "Julia 1.10.12 LTS (candidate only; not installed or accepted)",
    "steps": [
        "Install the candidate Julia release; record executable sha256, platform and architecture",
        "Create a dedicated project with OrdinaryDiffEqTsit5 and SciMLBase as direct dependencies",
        "Instantiate and precompile separately from execution; commit machine-generated Project.toml and Manifest.toml",
        "Run the worker with --project=<env> --startup-file=no --threads=1 and a fixed operation allowlist",
        "Handshake reports Julia version, project/manifest digests, package artifact identities, thread settings and "
        "worker source digest; the host compares them with the expected identity before accepting work",
        "Dispatch through SCR ExecutionSpecification (program, configuration, input_payload bytes); retain raw input "
        "and output bytes before decoding",
        "Accept the oscillator operation only after the analytic-oracle fixtures pass on Windows and Linux",
    ],
    "identity_fields": ["julia_version", "platform", "executable_sha256", "worker_source_sha256", "project_sha256",
                        "manifest_sha256", "package_artifacts", "threads", "numerical_preferences", "sysimage"],
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


@task("T145", changed_files=(MODULE, DOC), regression_tests=(f"{TESTS}::test_t145_julia_partial_with_plan",
                                                             f"{TESTS}::test_sympy_torus_geometry_matches_core"))
def julia_role(ctx):
    julia = ctx.available("tool:julia")
    ctx.artifact_json("julia-pin-procedure.json", dict(JULIA_PLAN, julia_on_path=julia))
    findings = [finding("Julia environment pinned and exercised through the CIW to SCR boundary",
                        "computational_pipeline", None, {}, expected_not_established=True),
                finding("Julia provider pin procedure", "provenance", JULIA_PLAN["identity_fields"],
                        {"derivation": "docs/JULIA_SP1.md#worker-lifecycle-and-environment"}, tolerance=EXACT)]
    symbolic = None
    if ctx.available("module:sympy"):
        symbolic = sympy_torus_check()
        ctx.artifact_json("sympy-torus.json", symbolic)
        import sympy
        findings.append(finding(
            "Symbolic torus Christoffel symbols and curvature agree with ciw.lab.surfaces.Torus", "mathematical",
            symbolic["max_abs_difference"],
            {"independent_check": dict(_check("sympy-derived Gamma and K against the core at 16 seeded points",
                                              symbolic["max_abs_difference"], 1e-12, kind="analytic"),
                                       producer={"implementation": "ciw.lab.surfaces.Torus",
                                                 "revision": _source_digest("src/ciw/lab/surfaces.py")},
                                       checker={"implementation": "sympy", "revision": sympy.__version__})},
            tolerance={"abs": 1e-12, "rel": 0}))
    fields = _fields(
        "Julia can carry symbolic, optimization and exploratory work behind the pinned CIW to SCR boundary; until a "
        "pinned Julia environment exists, SymPy demonstrates the symbolic role on the geometry core.",
        "Pin procedure from docs/JULIA_SP1.md; symbolic Christoffel symbols Gamma^k_ij = 1/2 g^kl (d_i g_jl + d_j g_il "
        "- d_l g_ij) and K = -(1/2W)[d_phi(G_phi/W) + d_theta(E_theta/W)], W = sqrt(EG), for the torus embedding.",
        ["Torus R=2, r=1", "16 PCG64(145) chart points", "docs/JULIA_SP1.md"],
        "Symbolic derivation evaluated at sample points; no Julia process ran.",
        "Symbolic and core Christoffel symbols and curvature agree to rounding; Julia claims stay unestablished.",
        "Probe for julia; record the pin procedure; derive torus geometry with SymPy and compare with the core.",
        "Provision Julia 1.10.12 LTS through SCR per docs/JULIA_SP1.md, then rerun T145 with the oscillator fixtures",
        numerical_result=(f"julia on PATH: {julia}; SymPy agreement max |difference| = "
                          f"{symbolic['max_abs_difference']:.2e}" if symbolic else f"julia on PATH: {julia}; SymPy absent"),
        uncertainty="SymPy stands in for the symbolic role only; it says nothing about Julia performance or pinning.",
        failure_modes_checked=["julia absent (recorded, not substituted)", "symbolic/core Christoffel disagreement",
                               "symbolic/core curvature disagreement"],
        unresolved_assumptions=["No Julia environment, manifest or worker exists; the pin procedure is unexecuted",
                                "Optimization and exploratory roles are not demonstrated"])
    return {"state": "partial", "fields": fields, "findings": findings}


# =================================================================== T146
def serialization_study() -> dict:
    from ..core.identities import canonical_json
    from ..telemetry import canonical as telemetry_canonical

    accepted = serial.vectors()
    invalid = serial.invalid_vectors()
    table, mismatches = [], {"ciw.telemetry.canonical": [], "ciw.core.identities.canonical_json": []}
    ascii_differs, reencode_failures = [], 0
    for name, value in accepted:
        utf8, ascii_bytes = serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True)
        if serial.canonical_bytes(json.loads(utf8)) != utf8:
            reencode_failures += 1
        if telemetry_canonical(value) != utf8:
            mismatches["ciw.telemetry.canonical"].append(name)
        if canonical_json(value).encode("utf-8") != ascii_bytes:
            mismatches["ciw.core.identities.canonical_json"].append(name)
        if utf8 != ascii_bytes:
            ascii_differs.append(name)
        table.append({"name": name, "canonical": utf8.decode("utf-8"), "utf8_hex": utf8.hex(),
                      "sha256": hashlib.sha256(utf8).hexdigest(), "ascii_sha256": hashlib.sha256(ascii_bytes).hexdigest()})
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
    collision = {"int_key": canonical_json({1: "x"}), "str_key": canonical_json({"1": "x"})}
    floats = [x for _, value in accepted for x in serial.floats_in(value)] + serial._random_floats(2000, 1460)
    digit_mismatch = sum(serial.decimal_form(repr(x)) != serial.decimal_form(serial.numpy_shortest(x)) for x in floats)
    roundtrip_failures = sum(float(repr(x)) != x or math.copysign(1, float(repr(x))) != math.copysign(1, x)
                             for x in floats)
    jcs_numbers = sorted({repr(x): serial.ecmascript_number(x) for x in floats
                          if repr(x) != serial.ecmascript_number(x)}.items())
    keys = list(dict(accepted)["key-order-code-points"])
    jcs_order, ciw_order = sorted(keys, key=lambda k: k.encode("utf-16-be")), sorted(keys)
    set_digest = serial.canonical_sha256([[row["name"], row["sha256"]] for row in table])
    return {"table": table, "refusals": refusals, "mismatches": mismatches, "ascii_differs": ascii_differs,
            "reencode_failures": reencode_failures, "collision": collision, "floats_checked": len(floats),
            "digit_mismatch": digit_mismatch, "roundtrip_failures": roundtrip_failures, "jcs_numbers": jcs_numbers,
            "jcs_key_order": jcs_order, "ciw_key_order": ciw_order, "set_digest": set_digest}


def rust_serialization(study) -> dict:
    accepted, invalid = serial.vectors(), serial.invalid_vectors()
    results = serial.rust_canonical([v for _, v in accepted] + [v for _, v, _ in invalid])
    byte_mismatch = [name for (name, value), got in zip(accepted, results)
                     if got != (serial.canonical_bytes(value), serial.canonical_bytes(value, ascii_only=True))]
    refusal_mismatch = [name for (name, _, expected), got in zip(invalid, results[len(accepted):]) if got != expected]
    return {"vectors": len(accepted) + len(invalid), "byte_mismatches": byte_mismatch,
            "refusal_mismatches": refusal_mismatch,
            "rust_refusals": {name: got for (name, _, _), got in zip(invalid, results[len(accepted):])}}


@task("T146", changed_files=(MODULE, SERIAL, DOC),
      regression_tests=(f"{TESTS}::test_t146_python_canonicalizers_and_vectors", f"{TESTS}::test_t146_rust_byte_identity"))
def canonical_serialization(ctx):
    study = serialization_study()
    build = serial.rust_build()
    rust = rust_serialization(study) if build["available"] else None
    ctx.artifact_json("canonical-json-spec.json", serial.SPEC)
    ctx.artifact_json("test-vectors.json", {"spec": serial.SPEC_ID, "vector_set_sha256": study["set_digest"],
                                            "vectors": study["table"], "invalid": study["refusals"]})
    ctx.artifact_text("ciw_targets.rs", serial.RUST_SOURCE)
    ctx.artifact_json("cross-language.json", {"rust": rust, "rust_build": {k: v for k, v in build.items() if k != "binary"},
                                              "jcs_number_differences": study["jcs_numbers"],
                                              "jcs_key_order": study["jcs_key_order"], "ciw_key_order": study["ciw_key_order"],
                                              "float_digit_values": study["floats_checked"]})
    spec_checks = [_check("re-encoding a decoded vector changes its bytes", study["reencode_failures"])]
    spec_checks += [{"reference_kind": "refusal", "reference": f"specification on {row['name']}",
                     "expected_refusal": row["expected"], "observed_refusal": row["specification"],
                     "passed": row["specification"] == row["expected"]} for row in study["refusals"]]
    lax = {label: [row["name"] for row in study["refusals"] if row[label] == "accepted"]
           for label in ("ciw.core.identities.canonical_json", "ciw.telemetry.canonical")}
    ascii_rows = {row["name"]: row for row in study["table"]}
    witness = ascii_rows["unicode-bmp"]
    findings = [
        finding("Canonical JSON v1 test vectors (bytes and sha256)", "provenance",
                {"spec": serial.SPEC_ID, "vectors": len(study["table"]), "invalid_vectors": len(study["refusals"]),
                 "vector_set_sha256": study["set_digest"]},
                {"checks": spec_checks}, tolerance=EXACT),
        finding("Existing Python canonicalizers reproduce the specification bytes on every accepted vector",
                "computational_pipeline", {k: len(v) for k, v in study["mismatches"].items()},
                {"checks": [_check("ciw.telemetry.canonical mismatches against the UTF-8 form",
                                   len(study["mismatches"]["ciw.telemetry.canonical"])),
                            _check("ciw.core.identities.canonical_json mismatches against the ASCII form",
                                   len(study["mismatches"]["ciw.core.identities.canonical_json"]))]},
                unit="mismatching vectors", tolerance=EXACT),
        finding("ciw.core.identities.canonical_json and ciw.telemetry.canonical produce different bytes for non-ASCII text",
                "computational_pipeline", {"differing_vectors": study["ascii_differs"]},
                {"checks": [_check("vectors whose ensure_ascii forms differ", len(study["ascii_differs"]), 1, "ge"),
                            _check("differing vectors that are printable ASCII only",
                                   sum(all(0x20 <= ord(ch) <= 0x7e for ch in ascii_rows[n]["canonical"])
                                       for n in study["ascii_differs"]))]},
                counterexample={"statement": "CIW already has one canonical JSON byte encoding",
                                "witness": {"vector": "unicode-bmp", "telemetry_sha256": witness["sha256"],
                                            "identities_sha256": witness["ascii_sha256"]}},
                tolerance=EXACT),
        finding("ciw.core.identities.canonical_json gives {1: 'x'} and {'1': 'x'} the same content identity",
                "computational_pipeline", hashlib.sha256(study["collision"]["int_key"].encode()).hexdigest(),
                {"checks": [_check("int-key and str-key canonical texts are equal",
                                   float(study["collision"]["int_key"] == study["collision"]["str_key"]), 1, "ge"),
                            _refusal("specification on {1: 'x'}", "non_string_key",
                                     lambda: serial.canonical_bytes({1: "x"}))]},
                counterexample={"statement": "Distinct Python values have distinct CIW content identities",
                                "witness": {"values": ["{1: 'x'}", "{'1': 'x'}"], "canonical": study["collision"]["str_key"]}},
                tolerance=EXACT),
        finding("Python canonicalizers accept values the specification refuses", "computational_pipeline", lax,
                {"checks": [_check("refused-by-spec vectors accepted by a CIW encoder",
                                   sum(len(v) for v in lax.values()), 1, "ge")]},
                counterexample={"statement": "Existing CIW canonicalizers enforce the cross-language specification",
                                "witness": {"vector": "integer-2^53", "accepted_by": sorted(
                                    k for k, v in lax.items() if "integer-2^53" in v)}},
                tolerance=EXACT),
        finding("CPython shortest float digits agree with NumPy Dragon4 on vector and random binary64 values",
                "numerical", {"values": study["floats_checked"], "mismatches": study["digit_mismatch"]},
                {"checks": [_check("repr round-trip failures (value or sign)", study["roundtrip_failures"])],
                 "independent_check": dict(_check("digit/exponent mismatches", study["digit_mismatch"]),
                                           producer={"implementation": "cpython.float_repr",
                                                     "revision": sys.version.split()[0]},
                                           checker={"implementation": "numpy.format_float_scientific(unique=True)",
                                                    "revision": np.__version__})},
                tolerance=EXACT),
        finding("CIW canonical JSON differs from RFC 8785 (JCS) numbers and key order", "computational_pipeline",
                {"number_differences": len(study["jcs_numbers"]),
                 "key_order_differs": study["jcs_key_order"] != study["ciw_key_order"]},
                {"derivation": "ECMA-262 Number::toString and UTF-16 code-unit key order (RFC 8785 section 3.2)",
                 "checks": [_check("vector floats formatted differently", len(study["jcs_numbers"]), 1, "ge"),
                            _check("key order differs (UTF-16 units against code points)",
                                   float(study["jcs_key_order"] != study["ciw_key_order"]), 1, "ge")]},
                counterexample={"statement": "CIW canonical JSON bytes equal RFC 8785 JCS bytes",
                                "witness": {"1.0": ["1.0", "1"], "1e+16": ["1e+16", "10000000000000000"],
                                            "keys": ["Ａ", "\U0001f600"]}},
                tolerance=EXACT),
    ]
    if rust is not None:
        findings.append(finding(
            "Rust canonical JSON is byte-identical to the specification and refuses the same inputs", "computational_pipeline",
            {"vectors": rust["vectors"], "byte_mismatches": len(rust["byte_mismatches"]),
             "refusal_mismatches": len(rust["refusal_mismatches"])},
            {"checks": [_check("Rust byte mismatches (UTF-8 and ASCII forms)", len(rust["byte_mismatches"]),
                               kind="cross_implementation"),
                        _check("Rust refusal-code mismatches", len(rust["refusal_mismatches"]),
                               kind="cross_implementation")]},
            tolerance=EXACT))
    else:
        findings.append(finding("Rust canonical JSON is byte-identical to the specification and refuses the same inputs",
                                "computational_pipeline", None, {}, expected_not_established=True))
    findings.append(finding("Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations",
                            "computational_pipeline", None, {}, expected_not_established=True))
    fields = _fields(
        "One canonical JSON encoding (sorted keys, no whitespace, UTF-8, shortest round-trip binary64 in CPython repr "
        "form, safe integers, refusals) is reproducible byte for byte by a separately written implementation in "
        "another language (Rust).",
        "Encoding E: JSON values -> bytes per ciw.canonical-json.v1; identity = sha256(E(v)). Floats: shortest digits d "
        "with decimal point position p, fixed form iff -4 < p <= 16.",
        [f"{len(study['table'])} accepted and {len(study['refusals'])} refused vectors (binary64 limits, subnormals, -0.0, "
         "Unicode, controls, nested keys, depth 64/65)", f"{study['floats_checked']} binary64 values for digit checks"],
        "Bytes and sha256 digests of each encoder's output; refusal codes.",
        "Spec, ciw.telemetry.canonical and the Rust probe agree on UTF-8 bytes; ciw.core.identities agrees with the "
        "ASCII variant; refused inputs are refused by the spec and Rust.",
        "Encode every vector with the spec reference, both CIW encoders and the Rust probe compiled at run time; "
        "compare float digits with NumPy Dragon4; contrast with JCS formatting.",
        "T150 (bind FPGA identity records to this encoding), then migrate ciw.core.identities with a versioned hash "
        "change",
        numerical_result=(f"{len(study['table'])} vectors, set digest {study['set_digest'][:16]}...; telemetry/identities "
                          f"mismatches {len(study['mismatches']['ciw.telemetry.canonical'])}/"
                          f"{len(study['mismatches']['ciw.core.identities.canonical_json'])}; ensure_ascii differs on "
                          f"{len(study['ascii_differs'])} vectors; Rust "
                          + (f"byte mismatches {len(rust['byte_mismatches'])}, refusal mismatches "
                             f"{len(rust['refusal_mismatches'])}" if rust else "unavailable")
                          + f"; float digit mismatches {study['digit_mismatch']}/{study['floats_checked']}."),
        uncertainty=("Exact byte comparison. The Rust probe and the spec reference were both written for CIW, so their "
                     "agreement is same-origin; float digit agreement between CPython and NumPy is independent but "
                     "covers only the tested values."),
        failure_modes_checked=["NaN/Infinity", "unsafe integers", "non-string keys (identity collision)",
                               "lone surrogates", "nesting depth", "ensure_ascii divergence", "U+007F and U+2028",
                               "Unicode normalization", "-0.0", "subnormals and binary64 limits",
                               "exponent-switch boundaries", "JCS divergence"],
        unresolved_assumptions=["ciw.core.identities.canonical_json remains the ASCII variant because changing it "
                                "would change every retained identity", "Julia, C++ and GPU-host encoders were not run",
                                "JCS formatting was implemented here from ECMA-262, not from an external JCS library"],
        provider_runtime_identity=_runtime_identity((MODULE, SERIAL), build))
    return {"state": "completed" if rust is not None else "partial", "fields": fields, "findings": findings}


# =================================================================== T147
@task("T147", changed_files=(MODULE, KERNELS, DOC), regression_tests=(f"{TESTS}::test_t147_harness_detects_differences",))
def cpu_gpu_comparison(ctx):
    study = ctx.memo("t147-dots", kernels.batched_dot_study)
    out, bounds, exact = study["outputs"], study["bounds"], study["exact"]
    reference = out["f64-sequential"]
    f64_policy = {"mode": "bound", "tolerance": bounds["f64-sequential"] + bounds["f64-blocked"]}
    f32_policy = {"mode": "bound", "tolerance": bounds["f64-sequential"] + bounds["f32-blocked"]}
    bitwise = kernels.compare_outputs(reference, out["f64-blocked"], {"mode": "bitwise"})
    reorder = kernels.compare_outputs(reference, out["f64-blocked"], f64_policy)
    pairwise = kernels.compare_outputs(reference, out["f64-pairwise"],
                                       {"mode": "bound", "tolerance": bounds["f64-sequential"] + bounds["f64-pairwise"]})
    f32_under_f64 = kernels.compare_outputs(reference, out["f32-blocked"], f64_policy)
    f32_under_f32 = kernels.compare_outputs(reference, out["f32-blocked"], f32_policy)
    against_exact = {name: float(np.max(np.abs(value - exact) / bounds[name])) for name, value in out.items()}
    faulty = out["f64-blocked"].copy()
    row, column = 17, 5
    faulty[row] -= study["A"][row, column] * study["x"][column]  # a dropped partial product (lost update)
    fault_f64 = kernels.compare_outputs(reference, faulty, f64_policy)
    fault_f32 = kernels.compare_outputs(reference, faulty, f32_policy)
    gpu = ctx.available("hardware:nvidia-gpu")
    ulp = kernels.ulp_distance(reference, out["f64-blocked"])
    summary = {name: {k: v for k, v in result.items() if k != "ratios"} for name, result in (
        ("bitwise f64 sequential vs blocked", bitwise), ("bound f64 sequential vs blocked", reorder),
        ("bound f64 sequential vs pairwise", pairwise), ("f32 blocked under f64 policy", f32_under_f64),
        ("f32 blocked under f32 policy", f32_under_f32), ("dropped product under f64 policy", fault_f64),
        ("dropped product under f32 policy", fault_f32))}
    ctx.artifact_json("comparison-summary.json", {"comparisons": summary, "against_exact_max_ratio": against_exact,
                                                  "depths": study["depths"], "max_ulp_f64_reorder": float(np.max(ulp)),
                                                  "gpu_present": gpu, "rows": study["rows"], "n": study["n"],
                                                  "lane_width": study["width"]})
    series = []
    for name, result in (("f64 reorder / f64 bound", reorder), ("f32 / f64 bound", f32_under_f64),
                         ("f32 / f32 bound", f32_under_f32)):
        ratios = np.sort(result["ratios"])
        series.append((name, list(range(1, len(ratios) + 1)), [float(v) for v in ratios]))
    ctx.artifact_text("difference-over-bound.svg", svg.line_plot(
        series, title="CPU-vs-CPU differences relative to the tolerance policy", xlabel="row (sorted)",
        ylabel="|difference| / policy bound", logy=True, markers=False))
    findings = [
        finding("CPU and GPU outputs agree under the tolerance policy on GPU hardware", "numerical", None, {},
                expected_not_established=True),
        finding("Bitwise policy detects reduction-order differences between float64 CPU orders", "numerical",
                bitwise["bitwise_differences"],
                {"generator": {"name": "PCG64 128x1024 uniform batched dot products", "seed": 147},
                 "checks": [_check("rows differing bitwise (sequential vs 32-lane blocked)", bitwise["bitwise_differences"],
                                   1, "ge")]},
                unit="rows of 128", tolerance=EXACT),
        finding("Float64 reduction-order differences lie within the analytic error-bound policy", "numerical",
                max(reorder["max_ratio"], pairwise["max_ratio"]),
                {"checks": [_check("max |difference|/bound, sequential vs blocked and pairwise",
                                   max(reorder["max_ratio"], pairwise["max_ratio"]), 1.0, "le", kind="analytic"),
                            _check("max |output - exact|/bound over all orders", max(against_exact.values()), 1.0, "le",
                                   kind="analytic")]},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Float32 results violate the float64 policy and satisfy the float32 policy", "numerical",
                {"violations_under_f64_policy": f32_under_f64["violations"], "max_ratio_under_f32_policy":
                 f32_under_f32["max_ratio"]},
                {"checks": [_check("rows violating the float64 policy", f32_under_f64["violations"], 1, "ge"),
                            _check("max |difference|/bound under the float32 policy", f32_under_f32["max_ratio"], 1.0,
                                   "le", kind="analytic")]},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("A dropped partial product is detected under both policies", "numerical",
                {"f64_policy_violations": fault_f64["violations"], "f32_policy_violations": fault_f32["violations"]},
                {"checks": [_check("violations under the float64 policy", fault_f64["violations"], 1, "ge"),
                            _check("violations under the float32 policy", fault_f32["violations"], 1, "ge")]},
                tolerance=EXACT),
        finding("GPU/CPU agreement establishes industrial readiness", "industrial_readiness", None, {}),
    ]
    fields = _fields(
        "A comparison harness with an analytic tolerance policy separates legitimate reduction-order and precision "
        "differences from faults; exercised CPU-against-CPU because no GPU is present.",
        "For a sum of terms each passing through k roundings, |computed - exact| <= gamma_k * sum|a_j x_j| with "
        "gamma_k = k u/(1 - k u), u = 2^-53 (float64) or 2^-24 (float32, inputs rounded). The policy tolerance for two "
        "outputs is the sum of their bounds; bitwise mode compares bit patterns.",
        ["128x1024 uniform[-1,1) matrix and vector, PCG64(147)", "orders: sequential, pairwise tree, 32-lane blocked "
         "(sequential lanes then tree); precisions float64 and float32", "exact dot products by TwoProduct + math.fsum"],
        "Elementwise absolute differences, bitwise equality and ratios to the policy bound.",
        "Bitwise policy flags reordering; bound policy accepts it; float32 fails the float64 policy and passes its "
        "own; a dropped product fails every policy.",
        "Compute the dot products in four orders/precisions, compare with the harness under each policy, inject a "
        "dropped-product fault.",
        "Run T147 on a CUDA host (hardware:nvidia-gpu) with the same data and policy",
        numerical_result=(f"{bitwise['bitwise_differences']}/128 rows differ bitwise between float64 orders; max "
                          f"difference/bound {max(reorder['max_ratio'], pairwise['max_ratio']):.3g}; float32 violates "
                          f"the float64 policy on {f32_under_f64['violations']} rows and stays at "
                          f"{f32_under_f32['max_ratio']:.3g} of its own bound; dropped product flagged on "
                          f"{fault_f64['violations']} row(s). GPU present: {gpu} (no GPU kernel was run)."),
        uncertainty="Bounds are worst-case (not probabilistic), so typical ratios are far below one; the harness has "
                    "not seen GPU fused multiply-add or atomics ordering.",
        failure_modes_checked=["reduction reordering", "precision reduction (float32)", "dropped partial product",
                               "shape mismatch and nonfinite outputs (refused by the harness)"],
        unresolved_assumptions=["GPU reductions may use FMA and tree shapes not modelled by the 32-lane order",
                                "No GPU hardware or driver was exercised"])
    return {"state": "partial", "fields": fields, "findings": findings}


# =================================================================== T148
REDUCTION_POLICY = {
    "schema": "ciw.reduction-policy.v1",
    "identity_bearing_sums": "exact: integer accumulation at scale 2^1074 with one final rounding (bitwise "
                             "order-independent; cross-checked with math.fsum)",
    "fixed_layout_arrays": "pairwise: fixed binary tree split at n//2 over the stored order (bitwise reproducible "
                           "only for the same order and length)",
    "streaming_accumulators": "neumaier: compensated, error <= 2u|S| + O(n u^2) sum|x|",
    "forbidden_for_identity": ["sequential float accumulation in unspecified order", "BLAS/numpy.sum (order is "
                               "implementation-defined)", "Kahan without the Neumaier branch (fails on large "
                               "cancelling terms)"],
}


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


@task("T148", changed_files=(MODULE, KERNELS, DOC), regression_tests=(f"{TESTS}::test_t148_reduction_policies",))
def reduction_policies(ctx):
    datasets = kernels.reduction_datasets()
    study = kernels.permutation_study(datasets)
    scaling = _reduction_scaling()
    fsum_mismatch = sum(a != b for data in study.values()
                        for a, b in zip(data["algorithms"]["exact"]["results"], data["algorithms"]["fsum"]["results"]))
    exact_distinct = max(data["algorithms"]["exact"]["distinct_results"] for data in study.values())
    pairwise_distinct = study["uniform"]["algorithms"]["pairwise"]["distinct_results"]
    replay = kernels.sum_pairwise(datasets["uniform"]) == kernels.sum_pairwise(list(datasets["uniform"]))
    kahan_case = study["kahan-counterexample"]["algorithms"]
    worst = {alg: max(study[d]["algorithms"][alg]["max_error_over_bound"] for d in study)
             for alg in kernels.REDUCTIONS}
    table = {d: {alg: {k: v for k, v in study[d]["algorithms"][alg].items() if k != "results"}
                 for alg in kernels.REDUCTIONS} for d in study}
    ctx.artifact_json("permutation-study.json", {"datasets": {d: {"n": s["n"], "exact": s["exact"], "abs_sum": s["abs_sum"],
                                                                  "orders": s["orders"]} for d, s in study.items()},
                                                 "algorithms": table})
    ctx.artifact_json("reduction-policy.json", REDUCTION_POLICY)
    ctx.artifact_text("error-scaling.svg", svg.line_plot(
        [(name, scaling["sizes"], errors) for name, errors in scaling["abs_errors"].items() if any(errors)]
        + [("sequential bound, sum|x| ~ n/2", scaling["sizes"], [kernels.gamma(n - 1) * n * 0.5 for n in scaling["sizes"]])],
        title="Absolute summation error, uniform[-1,1)", xlabel="n", ylabel="|computed - exact|", logx=True, logy=True))
    distinct_table = {d: {alg: study[d]["algorithms"][alg]["distinct_results"] for alg in kernels.REDUCTIONS}
                      for d in study}
    findings = [
        finding("Correctly rounded exact accumulation is permutation-invariant", "numerical", exact_distinct,
                {"generator": {"name": "PCG64 datasets and 24 permutations", "seed": 148},
                 "checks": [_check("distinct results across permutations minus one", exact_distinct - 1)],
                 "independent_check": dict(_check("mismatches against math.fsum over every order and dataset",
                                                  fsum_mismatch),
                                           producer={"implementation": "ciw.lab.implementation_targets_kernels.sum_exact"},
                                           checker={"implementation": "cpython.math.fsum",
                                                    "revision": sys.version.split()[0]})},
                unit="distinct results", tolerance=EXACT),
        finding("Fixed-order pairwise summation is reproducible for one order but not permutation-invariant", "numerical",
                pairwise_distinct,
                {"checks": [_check("replay of the same order differs", 0.0 if replay else 1.0),
                            _check("distinct pairwise results across permutations (uniform)", pairwise_distinct, 2, "ge")]},
                counterexample={"statement": "Pairwise summation is order-independent",
                                "witness": {"dataset": "uniform n=1024 PCG64(148)", "distinct_results": pairwise_distinct}},
                unit="distinct results", tolerance=EXACT),
        finding("Kahan summation loses the sum [1, 1e100, 1, -1e100] that Neumaier summation keeps", "numerical",
                {"kahan": kahan_case["kahan"]["first_order_result"], "neumaier": kahan_case["neumaier"]["first_order_result"],
                 "exact": study["kahan-counterexample"]["exact"]},
                {"checks": [_check("Kahan error", abs(kahan_case["kahan"]["first_order_result"] - 2.0), 2.0, "ge"),
                            _check("Neumaier error", kahan_case["neumaier"]["first_order_result"] - 2.0)]},
                counterexample={"statement": "Kahan compensated summation is accurate whenever Neumaier's is",
                                "witness": {"input": [1.0, 1e100, 1.0, -1e100], "kahan": 0.0, "neumaier": 2.0}},
                tolerance=EXACT),
        finding("Observed errors of every algorithm lie within their analytic bounds", "numerical", worst,
                {"checks": [_check(f"{alg}: max |error|/bound over datasets and permutations", ratio, 1.0, "le",
                                   kind="analytic") for alg, ratio in sorted(worst.items())]},
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        finding("Distinct results under permutation for each algorithm and dataset", "numerical", distinct_table,
                {"generator": {"name": "PCG64 permutations", "seed": 1480, "orders": 25}}, tolerance=EXACT),
        finding("Deterministic reduction policy record", "provenance", REDUCTION_POLICY,
                {"derivation": "docs/lab/IMPLEMENTATION_TARGETS.md#reduction-policies (Higham 2002, ch. 4)"},
                tolerance=EXACT),
    ]
    fields = _fields(
        "Only a correctly rounded (exact-accumulation) sum is bitwise independent of summation order; fixed-tree "
        "pairwise sums are reproducible only for a fixed order; compensated sums are accurate but not order-invariant, "
        "and plain Kahan fails on large cancelling terms.",
        "Bounds (u = 2^-53): sequential gamma_{n-1} sum|x|; pairwise gamma_{ceil(log2 n)} sum|x|; Kahan 2u sum|x| + "
        "O(n u^2) sum|x|; Neumaier 2u|S| + O(n u^2) sum|x| (second-order terms taken as 4 n u^2 sum|x|); exact "
        "rounding u|S|. Exact sums by integer accumulation at scale 2^1074.",
        ["uniform[-1,1) n=1024", "positive values 10^U(-8,8) n=1024", "cancelling +/-10^U(0,16) pairs plus noise n=1024",
         "[1, 1e100, 1, -1e100]", "24 PCG64(1480) permutations of each"],
        "Exact rational errors of each floating-point result; count of distinct bit patterns across orders.",
        "Exact accumulation: one result per dataset and equal to math.fsum; every error within its bound.",
        "Sum each permutation with five algorithms; compare with exact rationals and math.fsum; retain the policy.",
        "T147 (use the policy to set GPU comparison tolerances), then T155 (formal specification of the policy)",
        numerical_result=(f"Distinct results across 25 orders: exact {exact_distinct}, pairwise (uniform) "
                          f"{pairwise_distinct}; exact vs math.fsum mismatches {fsum_mismatch}; Kahan on "
                          f"[1, 1e100, 1, -1e100] = {kahan_case['kahan']['first_order_result']}, Neumaier = "
                          f"{kahan_case['neumaier']['first_order_result']}; worst error/bound "
                          + ", ".join(f"{k} {v:.3g}" for k, v in sorted(worst.items())) + "."),
        uncertainty="Exact arithmetic; permutation counts are samples (25 orders), so invariance of non-exact "
                    "algorithms is only refuted, never proven, and the exact algorithm's invariance is a theorem "
                    "sampled here.",
        failure_modes_checked=["order dependence", "catastrophic cancellation", "Kahan large-term failure",
                               "bound violation", "exact/fsum disagreement"],
        unresolved_assumptions=["The second-order constant in the compensated bounds is conservative, not tight",
                                "Parallel (multi-thread) reductions are not exercised"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T149
@task("T149", changed_files=(MODULE, FPGA, DOC), regression_tests=(f"{TESTS}::test_t149_telemetry_only_frames",))
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

    def corrupted(positions):
        data = bytearray(frame)
        for p in positions:
            data[p // 8] ^= 1 << (p % 8)
        return bytes(data)

    def detected(data):
        try:
            fpga.decode_frame(data)
            return False
        except fpga.TelemetryRefusal:
            return True

    single = sum(detected(corrupted([p])) for p in range(bits))
    burst_total = burst_detected = 0
    for length in range(2, 33):
        for start in range(0, bits - length + 1):
            interior = [start + i for i in range(1, length - 1) if rng.random() < 0.5]
            burst_total += 1
            burst_detected += detected(corrupted([start, *interior, start + length - 1]))
    random_total = 5000
    random_detected = sum(detected(corrupted(list(rng.choice(bits, int(rng.integers(2, 12)), replace=False))))
                          for _ in range(random_total))
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
    ]
    sending = sorted(name for name in dir(fpga.TelemetryReceiver) if not name.startswith("_")
                     and (fpga._FORBIDDEN_NAME.search(name) or name.startswith(("send", "transmit"))))
    ctx.artifact_json("telemetry-interface.json", fpga.INTERFACE)
    ctx.artifact_json("frame-example.json", {"hex": frame.hex(), "decoded": fpga.decode_frame(frame),
                                             "header_bytes": fpga.HEADER.size, "crc_bytes": fpga.TRAILER.size})
    ctx.artifact_json("error-detection.json", {"frame_bits": bits, "single_bit": [single, bits],
                                               "bursts_2_to_32": [burst_detected, burst_total],
                                               "random_2_to_11_bit": [random_detected, random_total]})
    findings = [
        finding("Telemetry frames round-trip every header field and channel bit-exactly", "computational_pipeline",
                roundtrip_failures,
                {"generator": {"name": "PCG64 random frames", "seed": 149, "frames": 400},
                 "checks": [_check("round-trip failures", roundtrip_failures)]}, unit="failures", tolerance=EXACT),
        finding("CIW table-driven CRC-32 agrees with zlib and the catalogue check value", "numerical",
                {"messages": len(messages), "mismatches": crc_mismatch, "check_value": f"{fpga.crc32(b'123456789'):#010x}"},
                {"checks": [_check("CRC-32(b'123456789') - 0xCBF43926", fpga.crc32(b"123456789") - 0xCBF43926,
                                   kind="analytic")],
                 "independent_check": dict(_check("mismatches against zlib.crc32 on random messages", crc_mismatch),
                                           producer={"implementation": "ciw.lab.implementation_targets_fpga.crc32"},
                                           checker={"implementation": "zlib.crc32", "revision": zlib.ZLIB_RUNTIME_VERSION})},
                tolerance=EXACT),
        finding("Every single-bit error and every 2-32 bit burst in a frame is refused by the decoder", "numerical",
                {"single_bit": [single, bits], "bursts": [burst_detected, burst_total],
                 "random_multi_bit": [random_detected, random_total]},
                {"checks": [_check("undetected single-bit errors", bits - single, kind="analytic"),
                            _check("undetected bursts of length <= 32", burst_total - burst_detected, kind="analytic"),
                            _check("undetected random 2-11 bit errors", random_total - random_detected)]},
                tolerance=EXACT),
        finding("Decoder, encoder and interface validator refuse every command or write path", "computational_pipeline",
                sum(c["passed"] for c in refusals), {"checks": refusals}, unit="refusals", tolerance=EXACT),
        finding("Host receiver exposes no sending or writing method", "computational_pipeline", sending,
                {"checks": [_check("public receiver attributes naming a send/write/command path", len(sending))]},
                tolerance=EXACT),
        finding("The frame format works on real FPGA links", "physical", None, {}),
        finding("A telemetry-only interface guarantees the FPGA cannot actuate the machine", "machine_safety", None, {}),
    ]
    fields = _fields(
        "A read-only frame (header, sequence, timestamp, clock id, raw payload, CRC-32) with a single telemetry frame "
        "type and no host-to-device field can be decoded safely, and every command or write path is refused.",
        "Frame = 28-byte big-endian header | 4*c bytes int32 payload | CRC-32/IEEE. CRC-32 detects all single-bit "
        "errors and all bursts of length <= 32 (degree-32 generator with nonzero constant term); other patterns "
        "escape with probability about 2^-32.",
        ["400 PCG64(149) random frames", "500 random messages for CRC comparison", "one 48-byte frame for exhaustive "
         "single-bit, burst and 5000 random corruptions"],
        "Decoder outcomes (accepted record or refusal code); no device was attached.",
        "Round trip exact; CRC matches zlib and 0xCBF43926; all single-bit and burst<=32 errors refused; command and "
        "write paths refused; the receiver has no sending method.",
        "Encode/decode random frames, compare CRC implementations, corrupt a frame exhaustively and at random, forge "
        "command frames and specs.",
        "T152 (drive the decoder with a lossy, jittered stream)",
        numerical_result=(f"Round-trip failures {roundtrip_failures}/400; CRC mismatches {crc_mismatch}/500; single-bit "
                          f"{single}/{bits}, bursts {burst_detected}/{burst_total}, random {random_detected}/"
                          f"{random_total} refused; {sum(c['passed'] for c in refusals)}/{len(refusals)} command/write "
                          "refusals."),
        uncertainty="Exhaustive for single-bit errors on one frame; bursts use one random interior per (length, start); "
                    "random corruptions escape CRC-32 with probability 2^-32 each.",
        failure_modes_checked=["command frame types", "write-request flag", "reserved flags", "unknown types",
                               "host-to-device spec fields", "truncation", "bit errors and bursts", "length mismatch"],
        unresolved_assumptions=["No FPGA, HDL implementation or physical link exists; the device side is simulated",
                                "A unidirectional physical transport (TX-only UART, multicast UDP) is assumed but not "
                                "demonstrated", "CRC protects against noise, not against a malicious sender"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T150
def _synthetic_build(version: str, seed: int) -> dict:
    constraints = {"pins.xdc": f"# placeholder constraints {version}\nset_property PACKAGE_PIN A1 [get_ports clk]\n".encode(),
                   "timing.xdc": f"create_clock -period 10.000 [get_ports clk] ;# {version}\n".encode()}
    sources = {"rtl/telemetry_tx.v": f"// placeholder source {version}\nmodule telemetry_tx(); endmodule\n".encode(),
               "rtl/crc32.v": b"// placeholder CRC-32 source\nmodule crc32(); endmodule\n"}
    bitstream = fpga.synthetic_bitstream(seed)
    record = fpga.bitstream_identity(bitstream, toolchain={"name": "placeholder-toolchain (not executed)",
                                                           "version": "0.0.1"},
                                     part="placeholder-part", constraints=constraints, source_files=sources,
                                     synthesis_options={"seed": seed, "version": version}, synthetic=True)
    return {"bitstream": bitstream, "constraints": constraints, "sources": sources, "record": record}


@task("T150", changed_files=(MODULE, FPGA, SERIAL, DOC), regression_tests=(f"{TESTS}::test_t150_bitstream_identity",))
def bitstream_identity(ctx):
    build = _synthetic_build("2.1.0", 150)
    record, bitstream = build["record"], build["bitstream"]
    fpga.validate_identity(record, bitstream=bitstream, constraints=build["constraints"], source_files=build["sources"])
    repeat = _synthetic_build("2.1.0", 150)["record"]

    def edited(**changes):
        body = dict(record, **changes)
        body.pop("record_sha256")
        body["record_sha256"] = hashlib.sha256(serial.canonical_bytes(body)).hexdigest()
        return body

    flipped = bytearray(bitstream)
    flipped[len(flipped) // 2] ^= 0x01
    changed_constraints = dict(build["constraints"], **{"timing.xdc": b"create_clock -period 8.000 [get_ports clk]\n"})
    added_constraints = dict(build["constraints"], **{"extra.xdc": b"# added\n"})
    changed_sources = dict(build["sources"], **{"rtl/crc32.v": b"// edited\n"})
    missing = {k: v for k, v in record.items() if k != "toolchain"}
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
        _refusal("floating toolchain version 'latest'", "floating_toolchain_version",
                 lambda: fpga.validate_identity(edited(toolchain={"name": "placeholder-toolchain", "version": "latest"}))),
        _refusal("toolchain version range '>=2024.1'", "floating_toolchain_version",
                 lambda: fpga.validate_identity(edited(toolchain={"name": "placeholder-toolchain", "version": ">=2024.1"}))),
        _refusal("part changed without recomputing the record digest", "record_digest_mismatch",
                 lambda: fpga.validate_identity(dict(record, part="other-part"))),
        _refusal("record without toolchain", "missing_field", lambda: fpga.validate_identity(missing)),
        _refusal("uppercase digest", "bad_digest_format",
                 lambda: fpga.validate_identity(dict(record, bitstream_sha256=record["bitstream_sha256"].upper()))),
    ]
    ctx.artifact_json("bitstream-identity.json", record)
    ctx.artifact_json("identity-schema.json", {"schema": fpga.IDENTITY_SCHEMA, "fields": list(fpga.IDENTITY_FIELDS)
                                               + ["record_sha256"], "toolchain_version_pattern": fpga._PINNED_VERSION.pattern,
                                               "canonical_encoding": serial.SPEC_ID})
    findings = [
        finding("Bitstream identity record binds bitstream, toolchain, constraints and source tree", "provenance",
                {"record_sha256": record["record_sha256"], "refused_mutations": sum(c["passed"] for c in mutations)},
                {"generator": {"name": "seeded synthetic placeholder bitstream", "seed": 150},
                 "checks": mutations + [_check("record digest differs on regeneration",
                                               0.0 if repeat == record else 1.0)]},
                tolerance=EXACT),
        finding("A real bitstream with this identity exists and is loaded on hardware", "physical", None, {}),
        finding("The bitstream is approved for production deployment", "production_acceptance", None, {}),
    ]
    decision = _refusal("deployment decision for a synthetic record", "synthetic_bitstream",
                        lambda: fpga.deployment_decision(record))
    findings.insert(1, finding("Deployment of a synthetic placeholder bitstream is refused", "computational_pipeline",
                               decision["observed_refusal"], {"checks": [decision]}, tolerance=EXACT))
    fields = _fields(
        "An identity record over canonical JSON can bind a bitstream's sha256 to its exact toolchain version, "
        "constraint files and source tree so that any change to any of them is detected.",
        "record_sha256 = sha256(E(record without record_sha256)), E = ciw.canonical-json.v1; constraints_sha256 = "
        "sha256(E({file: sha256})); source tree = sha256(E({path: sha256})); toolchain version must match "
        f"{fpga._PINNED_VERSION.pattern}.",
        ["Seeded synthetic placeholder bitstream (prefix CIW-SYNTHETIC-BITSTREAM-NOT-A-CONFIGURATION)",
         "placeholder constraint and source files"],
        "Validator outcomes over the record and supplied artifacts; no FPGA toolchain ran.",
        "The unmodified record validates and regenerates identically; each of ten mutations is refused with its code; "
        "deployment is refused.",
        "Build the record, regenerate it, apply ten mutations, request a deployment decision.",
        "T151 (compatibility and rollback records referencing these identities)",
        numerical_result=f"record_sha256 {record['record_sha256'][:16]}...; {sum(c['passed'] for c in mutations)}/10 "
                         "mutations refused; deployment refused (synthetic_bitstream).",
        uncertainty="Exact digests; the placeholder bitstream has no hardware meaning.",
        failure_modes_checked=["bitstream bit flip", "truncation", "constraint edit/addition", "source edit",
                               "floating toolchain version", "unrecomputed digest", "missing field", "digest format"],
        unresolved_assumptions=["No real bitstream, toolchain log or device part exists here",
                                "Toolchain determinism (same inputs -> same bitstream) is not established"])
    return {"state": "completed", "fields": fields, "findings": findings}


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
    refusals = [
        _refusal("rollback to the previous version in an enumerated incompatible state", "incompatible_target",
                 lambda: fpga.rollback_record(matrix, registry, current=witness["from"], target=witness["to"],
                                              host=witness["host"], board=witness["board"], reason="test")),
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
                                   kind="cross_implementation")]}, tolerance=EXACT),
        finding("Rollback validation refuses incompatible, unregistered, no-op, forward and unexplained rollbacks",
                "computational_pipeline", sum(c["passed"] for c in refusals), {"checks": refusals},
                unit="refusals", tolerance=EXACT),
        finding("Rolling back to the previous bitstream version can be incompatible", "computational_pipeline",
                {"cases": len(unsafe_previous), "witness": witness},
                {"checks": [_check("(host, board) states where the previous version is incompatible", len(unsafe_previous),
                                   1, "ge")]},
                counterexample={"statement": "Rolling back to the immediately previous bitstream is always compatible",
                                "witness": witness},
                tolerance=EXACT),
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
        "T152 (link behaviour after a rollback changes the frame format)",
        numerical_result=(f"{len(rule)}/{len(combos)} combinations compatible, {disagreements} disagreements; "
                          f"{len(unsafe_previous)} previous-version rollbacks incompatible (witness {witness}); "
                          f"{sum(c['passed'] for c in refusals)}/{len(refusals)} refusals."),
        uncertainty="Exact enumeration over a declared, synthetic matrix.",
        failure_modes_checked=["incompatible frame format", "unsupported board", "unregistered target",
                               "digest substitution", "no-op", "forward 'rollback'", "missing reason",
                               "stale matrix", "execution without authority"],
        unresolved_assumptions=["The matrix is illustrative; real compatibility needs hardware qualification",
                                "Board revision detection and host decoder version reporting are assumed trustworthy"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T152
STALE_AFTER_NS = 4_000_000


@task("T152", changed_files=(MODULE, FPGA, DOC), regression_tests=(f"{TESTS}::test_t152_loss_latency_staleness",))
def link_simulation(ctx):
    sim = fpga.simulate_link()
    deliveries, lost, model = sim["deliveries"], sim["lost"], sim["model"]
    exact = fpga.TelemetryReceiver(sim["clock_offset_ns"], STALE_AFTER_NS)
    for received, sequence, latency, frame in deliveries:
        exact.accept(frame, received)
    first, last = exact.first, exact.highest
    span = (last - first) % fpga.SEQUENCE_MODULUS + 1

    def inside(seq):
        return (seq - first) % fpga.SEQUENCE_MODULUS < span

    true_lost = {s for s in lost if inside(s)}
    true_stale = {(seq, rcv) for rcv, seq, lat, _ in deliveries if lat > STALE_AFTER_NS}
    detected_stale = set(exact.stale)
    # Offset estimated from the minimum delay over the first 200 deliveries (assumes zero minimum latency).
    estimate = min(rcv - fpga.decode_frame(fr)["timestamp_ns"] for rcv, _, _, fr in deliveries[:200])
    bias = estimate - sim["clock_offset_ns"]
    estimated = fpga.TelemetryReceiver(estimate, STALE_AFTER_NS)
    for received, sequence, latency, frame in deliveries:
        estimated.accept(frame, received)
    missed = true_stale - set(estimated.stale)
    predicted_missed = {(seq, rcv) for rcv, seq, lat, _ in deliveries if STALE_AFTER_NS < lat <= STALE_AFTER_NS + bias}
    false_alarms = set(estimated.stale) - true_stale
    naive = fpga.naive_gap_count([seq for _, seq, _, _ in deliveries])
    duplicates_sent = len(deliveries) - (sim["frames"] - len(lost))
    p_bad = model["p_good_to_bad"] / (model["p_good_to_bad"] + model["p_bad_to_good"])
    p_loss = (1 - p_bad) * model["loss_good"] + p_bad * model["loss_bad"]
    lam = 1 - model["p_good_to_bad"] - model["p_bad_to_good"]
    loss_rate = len(exact.pending) / span
    sigma = math.sqrt(p_loss * (1 - p_loss) / span * (1 + lam) / (1 - lam))
    ages = np.array(exact.ages, dtype=float)
    mean_latency = model["base_latency_ns"] + model["jitter_shape"] * model["jitter_scale_ns"]
    latency_sigma = math.sqrt(model["jitter_shape"]) * model["jitter_scale_ns"] / math.sqrt(len(ages))
    wrap_lost = [s for s in sorted(true_lost) if s >= fpga.SEQUENCE_MODULUS - 3 or s < 3]
    summary = {"frames": sim["frames"], "deliveries": len(deliveries), "lost_total": len(lost),
               "lost_between_first_and_last": len(true_lost), "detected_lost": len(exact.pending),
               "reordered": exact.reordered, "duplicates_detected": exact.duplicates, "duplicates_sent": duplicates_sent,
               "stale_true": len(true_stale), "stale_detected_exact_offset": len(detected_stale),
               "offset_bias_ns": bias, "stale_missed_with_estimate": len(missed),
               "predicted_missed": len(predicted_missed), "false_alarms_with_estimate": len(false_alarms),
               "naive_gap_count": naive, "loss_rate": loss_rate, "stationary_loss": p_loss,
               "mean_age_ns": float(ages.mean()), "wrap_losses": wrap_lost, "refused_frames": exact.refused}
    ctx.artifact_json("link-simulation.json", {"model": model, "summary": summary, "stale_after_ns": STALE_AFTER_NS,
                                               "start_sequence": sim["start_sequence"], "period_ns": sim["period_ns"]})
    order = np.argsort(ages)
    ctx.artifact_text("age-distribution.svg", svg.line_plot(
        [("frame age (exact offset)", [float(v) / 1e6 for v in ages[order]],
          [float(i + 1) / len(ages) for i in range(len(ages))]),
         ("stale threshold", [STALE_AFTER_NS / 1e6] * 2, [0.0, 1.0])],
        title="Empirical CDF of frame age", xlabel="age (ms)", ylabel="fraction of deliveries", markers=False))
    findings = [
        finding("Sequence-number gap detection recovers every lost frame across the 32-bit wrap", "computational_pipeline",
                {"lost": len(true_lost), "detected": len(exact.pending), "reordered": exact.reordered,
                 "duplicates": exact.duplicates},
                {"generator": {"name": "Gilbert-Elliott loss, gamma jitter, duplicates", "seed": 152},
                 "checks": [_check("|detected set symmetric difference true lost set|", len(exact.pending ^ true_lost)),
                            _check("duplicates detected minus duplicates sent", exact.duplicates - duplicates_sent),
                            _check("forced loss at sequence 2^32-1 detected",
                                   0.0 if (2 ** 32 - 1) in exact.pending else 1.0)]},
                tolerance=EXACT),
        finding("Staleness detection with the declared clock offset matches ground truth exactly",
                "computational_pipeline", {"stale": len(true_stale), "detected": len(detected_stale)},
                {"checks": [_check("|detected symmetric difference true stale|", len(detected_stale ^ true_stale))]},
                tolerance=EXACT),
        finding("An offset estimated from minimum delay misses exactly the stale frames within its bias",
                "numerical", {"bias_ns": bias, "missed": len(missed), "false_alarms": len(false_alarms)},
                {"checks": [_check("|missed symmetric difference predicted (threshold, threshold + bias]|",
                                   len(missed ^ predicted_missed), kind="analytic"),
                            _check("false alarms with a nonnegative bias", len(false_alarms), kind="analytic")]},
                tolerance=EXACT),
        finding("Loss rate and mean latency are consistent with the simulated model", "numerical",
                {"loss_rate": loss_rate, "mean_age_ns": float(ages.mean())},
                {"checks": [_check("|loss rate - stationary loss| / (4 sigma, burst-inflated)",
                                   abs(loss_rate - p_loss) / (4 * sigma), 1.0, "le"),
                            _check("|mean age - (base + shape*scale)| / (4 sigma)",
                                   abs(float(ages.mean()) - mean_latency) / (4 * latency_sigma), 1.0, "le")]},
                tolerance={"abs": 1e-9, "rel": 1e-9}),
        finding("A detector without modular sequence arithmetic miscounts losses", "computational_pipeline",
                {"naive_count": naive, "true_lost": len(true_lost)},
                {"checks": [_check("|naive count - true lost|", abs(naive - len(true_lost)), 1, "ge")]},
                counterexample={"statement": "Summing seq - previous - 1 in arrival order counts lost frames",
                                "witness": {"naive": naive, "true": len(true_lost), "wrap_losses": wrap_lost,
                                            "reordered": exact.reordered}},
                tolerance=EXACT),
        finding("Simulated loss, latency and staleness represent the real FPGA telemetry link", "physical", None, {}),
    ]
    fields = _fields(
        "Sequence numbers with modulo-2^32 serial arithmetic detect every loss, duplicate and reordering, and "
        "timestamps with a known clock offset detect every stale frame; an offset estimated from minimum delay "
        "misses exactly the stale frames within its bias.",
        "Gilbert-Elliott loss (p_gb=0.005, p_bg=0.2, loss 0.01/0.5), latency = 2 ms + Gamma(2, 0.5 ms), duplicates "
        "p=0.002, period 1 ms, start sequence 2^32-1500, forced loss at 2^32-1. Stationary loss = pi_g l_g + pi_b l_b; "
        "stale iff age = t_rx - (t_dev + offset) > 4 ms.",
        ["4000 frames encoded with the T149 codec", "PCG64(152)"],
        "Receiver outputs (pending gaps, reorders, duplicates, stale flags) compared with the simulation ground truth.",
        "Detected losses = true losses between the first and last received sequence; stale detection exact with the "
        "true offset; misses with an estimated offset = frames with latency in (4 ms, 4 ms + bias].",
        "Simulate the link, run two receivers (exact and estimated offset) and a naive detector.",
        "T153 (keep any response to telemetry gaps from reaching actuators)",
        numerical_result=(f"{len(true_lost)} losses detected exactly ({exact.reordered} reordered, {exact.duplicates} "
                          f"duplicates); stale {len(detected_stale)}/{len(true_stale)}; offset bias "
                          f"{bias / 1e6:.3f} ms misses {len(missed)} stale frames (predicted {len(predicted_missed)}); "
                          f"naive detector counts {naive}; loss rate {loss_rate:.4f} vs stationary {p_loss:.4f}."),
        uncertainty="Deterministic given the seed; the loss-rate check uses an approximate burst-inflated binomial "
                    "standard error.",
        failure_modes_checked=["32-bit sequence wrap", "reordering", "duplicates", "burst loss", "clock offset "
                               "estimation bias", "naive gap arithmetic"],
        unresolved_assumptions=["Gaps of 2^31 or more are unsupported by serial arithmetic",
                                "Losses before the first and after the last received frame are undetectable",
                                "Device and host clocks are assumed not to drift during the stream"])
    return {"state": "completed", "fields": fields, "findings": findings}


# =================================================================== T153
EXTERNAL = authority.AuthorizationRecord(issuer="plant-safety-office (hypothetical)", subject="cell-7 spindle",
                                         channels=("spindle_speed",), valid_from="2026-01-01T00:00:00Z",
                                         valid_until="2027-01-01T00:00:00Z", purpose="test fixture",
                                         signature="unverifiable-placeholder")


@task("T153", changed_files=(MODULE, AUTHORITY, DOC), regression_tests=(f"{TESTS}::test_t153_writes_disabled_by_default",))
def actuator_writes(ctx):
    from dataclasses import replace

    policy = authority.ActuatorWritePolicy()
    rng = np.random.Generator(np.random.PCG64(153))
    channels = ("spindle_speed", "feed_override", "axis_x_setpoint", "coolant", "laser_power")
    accepted, codes = 0, {}
    for _ in range(1000):
        channel = channels[int(rng.integers(len(channels)))]
        try:
            policy.check_write(channel, float(rng.normal()), now=NOW)
            accepted += 1
        except authority.AuthorityRefusal as refusal:
            codes[refusal.code] = codes.get(refusal.code, 0) + 1
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
    ctx.artifact_json("write-policy.json", {"default_policy": {"enabled": policy.enabled, "authorization": None},
                                            "random_attempts": 1000, "accepted": accepted, "refusal_codes": codes,
                                            "enable_routes": [{k: c[k] for k in ("reference", "expected_refusal",
                                                                                 "observed_refusal")} for c in routes],
                                            "verification_order": ["authorization_missing", "authorization_wrong_type",
                                                                   "self_issued_authority", "authorization_expired",
                                                                   "out_of_scope", "unsigned_authorization",
                                                                   "no_trust_anchor"]})
    findings = [
        finding("Default policy refuses every write attempt", "computational_pipeline",
                {"attempts": 1000, "accepted": accepted},
                {"generator": {"name": "PCG64 write attempts over five channels", "seed": 153},
                 "checks": [_check("accepted writes", accepted),
                            _refusal("default write", "writes_disabled_by_default",
                                     lambda: policy.check_write("spindle_speed", 0.0, now=NOW))]},
                tolerance=EXACT),
        finding("Enabling writes is refused on every route, including a well-formed external record",
                "computational_pipeline", sum(c["passed"] for c in routes), {"checks": routes}, unit="refused routes",
                tolerance=EXACT),
        finding("A frozen in-process policy object can be mutated", "computational_pipeline",
                {"flag_mutated": mutated, "writes_still_refused": after_flag["passed"] and after_record["passed"]},
                {"checks": [_check("object.__setattr__ changed enabled to True", float(mutated), 1, "ge"),
                            after_flag, after_record]},
                counterexample={"statement": "An in-process Python flag is an actuator authority boundary",
                                "witness": {"mutation": "object.__setattr__(policy, 'enabled', True)",
                                            "outcome": "flag changed; the gate still refused because it re-verifies the "
                                                       "authorization on every write"}},
                tolerance=EXACT),
        finding("The lab holds actuator write authority", "actuator_authority", None, {}),
        finding("Disabled-by-default software writes make the machine safe", "machine_safety", None, {}),
    ]
    fields = _fields(
        "A deny-by-default write policy refuses every write; enabling needs an externally issued, signed, in-scope, "
        "unexpired authorization verified against a trust anchor the lab does not hold, so no route opens it here.",
        "Gate(channel) = refuse unless enabled and verify(authorization, channel, now) passes; verify checks type, "
        "issuer, validity window, scope, signature, trust anchor in that order; the lab build has no trust anchor "
        "and no actuator transport.",
        ["1000 PCG64(153) write attempts on five channels", "eight enabling routes", "fixed instant " + NOW],
        "Refusal codes raised by the policy; no actuator exists.",
        "Zero accepted writes; each route stops at its expected check; mutation of the frozen object does not open "
        "the gate.",
        "Attempt random writes on the default policy, try each enabling route, then bypass the frozen dataclass.",
        "T154 (treat control outputs as proposals behind the same authorization)",
        numerical_result=f"{accepted}/1000 writes accepted; {sum(c['passed'] for c in routes)}/{len(routes)} enabling "
                         "routes refused with their expected codes; in-memory mutation succeeded but writes stayed refused.",
        uncertainty="Exhaustive over the declared routes; software refusals inside one process can be bypassed by "
                    "code with the same privileges (the counterexample shows the mutation itself succeeds).",
        failure_modes_checked=["missing, wrong-type, self-issued, expired, out-of-scope, unsigned and unanchored "
                               "authorizations", "lab issuing authority", "frozen-object mutation"],
        unresolved_assumptions=["Real enforcement belongs in hardware interlocks and a separate controller process",
                                "Authorization record format and signature scheme are placeholders"])
    return {"state": "completed", "fields": fields, "findings": findings}


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


@task("T154", changed_files=(MODULE, AUTHORITY, DOC), regression_tests=(f"{TESTS}::test_t154_control_outputs_are_proposals",))
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
    ctx.artifact_json("proposal-record.json", record)
    ctx.artifact_json("proposal-study.json", {k: v for k, v in study.items() if k != "sphere_record"})
    rows = study["torus"]
    ctx.artifact_text("proposal-residual.svg", svg.line_plot(
        [("uncorrected", [r["lateral"] for r in rows], [abs(r["uncorrected"]) for r in rows]),
         ("with heading proposal", [r["lateral"] for r in rows], [abs(r["corrected"]) for r in rows])],
        title="Normal separation at L = 2 on the torus", xlabel="lateral offset d", ylabel="|separation at L|",
        logx=True, logy=True))
    sphere_error = abs(study["sphere_proposal"] - study["sphere_analytic"]) / abs(study["sphere_analytic"])
    findings = [
        finding("Every control output carries status proposal and cannot be converted to a command here",
                "computational_pipeline", {"status": proposal.status, "refusals": sum(c["passed"] for c in refusals)},
                {"checks": refusals}, tolerance=EXACT),
        finding("Jacobi heading proposal cancels a lateral offset to second order", "numerical",
                {"corrected_order": study["corrected_order"], "uncorrected_order": study["uncorrected_order"]},
                {"generator": {"name": "torus R=2 r=1, u0=(0.4, 0.3), heading 0.5, L=2, d in {0.02, 0.01, 0.005}"},
                 "checks": [_check("corrected residual order - 2", study["corrected_order"] - 2.0, 0.2,
                                   kind="self_convergence"),
                            _check("uncorrected residual order - 1", study["uncorrected_order"] - 1.0, 0.2,
                                   kind="self_convergence"),
                            _check("sphere proposal relative error against -cot(L) d", sphere_error, 1e-8,
                                   kind="analytic")]},
                tolerance={"abs": 1e-3, "rel": 0}),
        finding("Heading proposals are authorized for execution as actuator commands", "actuator_authority", None, {}),
        finding("Applying the proposed heading corrections on a machine is safe", "machine_safety", None, {}),
    ]
    fields = _fields(
        "Control outputs can be computed and retained as immutable proposals, with content identities, while every "
        "conversion to a command is refused without separate authorization; the geometric proposal itself is sound "
        "to the order the Jacobi model predicts.",
        "Normal Jacobi field j(L) = j_lat(L) d + j_head(L) h; proposal h = -j_lat(L) d / j_head(L), undefined at a "
        "conjugate point (j_head(L) = 0). On the unit sphere h = -cot(L) d. With h applied, the residual separation "
        "is O(d^2).",
        ["Unit sphere, u0=(1.1, 0.2), heading 0.3, L=1.3 and L=pi", "Torus R=2, r=1, u0=(0.4, 0.3), heading 0.5, "
         "L=2, 200 RK4 steps, d in {0.02, 0.01, 0.005}"],
        "Nonlinear geodesic integration of the offset start with and without the proposal; separation g(du, N) at L.",
        "Proposal matches -cot(L) d on the sphere; corrected residual order 2, uncorrected order 1; all conversions "
        "refused.",
        "Compute proposals, verify them by integration, then attempt construction as a command and conversion under "
        "each authorization route.",
        "T141 (keep production acceptance outside the system) and a hardware-in-the-loop authority design review",
        numerical_result=(f"Sphere proposal relative error {sphere_error:.2e}; residual orders corrected "
                          f"{study['corrected_order']:.3f}, uncorrected {study['uncorrected_order']:.3f}; "
                          f"{sum(c['passed'] for c in refusals)}/{len(refusals)} refusals."),
        uncertainty="Orders from three offsets (least squares); RK4 error at 200 steps is far below the residuals.",
        failure_modes_checked=["conjugate point", "status forgery", "missing, self-issued and unanchored "
                               "authorization", "dict posing as proposal", "nonfinite value"],
        unresolved_assumptions=["The proposal is a kinematic heading correction on an ideal surface, not a validated "
                                "controller", "Actuator dynamics, limits and latency are not modelled"])
    return {"state": "completed", "fields": fields, "findings": findings}
