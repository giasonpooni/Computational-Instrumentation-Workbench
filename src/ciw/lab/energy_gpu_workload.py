"""The common workload of the energy, GPU and implementation experiments.

One computation runs on every implementation and device the lab compares
(T115, T117, T120, T121 and T147): the fixed-step Gaussian variational
iteration that the PTX kernel ``gaussian_vi`` of :mod:`ciw.energy_cuda`
executes on a GPU and that ``ciw energy record`` captures on the RTX 2080 host
(T116, T118, T119). Per replica, from the prepared inputs of
``ciw.energy_cuda._prepare`` (information precision P, information vector b,
initial mean m and precision Q, step sizes alpha and beta, and 1 - beta
rounded once on the host)::

    g_i  = ((P_i0 * m_0) + (P_i1 * m_1)) - b_i                  i = 0, 1
    m_i <- m_i - (alpha * g_i)
    Q_ij <- ((1 - beta) * Q_ij) + (beta * P_ij)
    after K iterations: det = (Q_00 * Q_11) - (Q_01 * Q_10)
    outputs (m_0, m_1, Q_11 / det, (-Q_01) / det, (-Q_10) / det, Q_00 / det)

Every operation is rounded to nearest in the declared precision with no fused
multiply-add, which is what the kernel's explicit ``.rn`` instructions
declare. The declared problem and solver settings are
``examples/energy-accuracy/problem.json`` (embedded here so an installed
package has them); K is the iteration count ``ciw energy record`` plans for
them (the first iteration whose KL to the exact posterior meets the declared
target), and a batch is 4096 identical replicas, the capture's default.

Implementations: a NumPy reference in float64 or float32 (the float32 run
rounds the float64 prepared inputs once and then stays in float32), an
emulation of fused multiply-add contraction with an exact rational FMA, an
embedded std-only Rust port compiled at run time, and the PTX kernel itself
when an NVIDIA GPU answers the probe.

Non-claims: nothing here measures energy, power or time. Agreement between
the NumPy reference, the Rust port and the PTX kernel is agreement of
implementations the workbench wrote (``cross_implementation``), never
independent verification. The PTX kernel exists only in binary64, and no
device-wide reduction kernel exists; both gaps are recorded as a deferred
research question, not hidden.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import functools
import hashlib
import json
import os
import re
import subprocess

import numpy as np

from . import energy_gpu_kernels

# examples/energy-accuracy/problem.json (tests/test_lab_energy_gpu.py checks the two are equal).
SPEC = {
    "schema": "ciw.energy-problem.v1",
    "problem": {"coordinate_system": "normalized_dimensionless", "prior_mean": [0.1, -0.2],
                "prior_covariance": [[1, 0.2], [0.2, 1]], "observation_matrix": [[1, 0.3], [-0.2, 1]],
                "observations": [2, -1], "noise_covariance": [[1, 0.1], [0.1, 0.7]]},
    "solver": {"initial_mean": [1, -1], "initial_covariance": [[1, 0.1], [0.1, 1]], "alpha": 0.2, "beta": 0.2,
               "max_iterations": 256, "gradient_tolerance": 1e-12, "precision_tolerance": 1e-12},
    "target_kl_nats": 1e-8,
}
REPLICAS = 4096
NAME = "ciw.lab.gaussian-vi-iteration.v1"
# The modules that define the common workload: the PTX text and the input preparation (ciw.energy_cuda), the
# planner that fixes K (ciw.energy_bench) and the information system and KL (ciw.free_energy_math). A task whose
# results depend on the workload digests them beside its own sources and records workload().
SOURCES = ("src/ciw/energy_cuda.py", "src/ciw/energy_bench.py", "src/ciw/free_energy_math.py")
INPUT_LAYOUT = ("P00", "P01", "P10", "P11", "b0", "b1", "m0", "m1", "Q00", "Q01", "Q10", "Q11", "alpha", "beta",
                "one_minus_beta")
OUTPUT_LAYOUT = ("mean_0", "mean_1", "covariance_00", "covariance_01", "covariance_10", "covariance_11")
PRECISIONS = ("float64", "float32")
# The fused multiply-add contractions a compiler may apply to the kernel's source order (nvcc does by default
# without explicit .rn instructions), and one reassociation of the dot products, emulated exactly in binary64.
VARIANTS = ("fma-first", "fma-second", "reassociated")
VARIANT_NOTES = {
    "fma-first": "a*b + c*d -> fma(a, b, c*d); m - alpha*g -> fma(-alpha, g, m); det -> fma(Q00, Q11, -(Q01*Q10))",
    "fma-second": "a*b + c*d -> fma(c, d, a*b); m - alpha*g -> fma(-alpha, g, m); det -> fma(-Q01, Q10, Q00*Q11)",
    "reassociated": "((P_i0 m_0) + (P_i1 m_1)) - b_i -> (P_i0 m_0) + ((P_i1 m_1) - b_i), no fused operations",
}
# The one deferred research question that owns the missing GPU code of the common workload (T117, T120, T121,
# T147 and T148 name it; the planner lists it once).
GPU_QUESTION = ("Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX "
                "kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its "
                "per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one "
                "atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host "
                "and captured with `ciw energy record` for energy; until then those claims stay not_established on "
                "every host")
NO_GPU_PROBE = ("no NVIDIA GPU answered the hardware:nvidia-gpu probe in this task; the PTX kernel of the common "
                "workload exists (ciw.energy_cuda gaussian_vi), so this comparison runs on a host where the probe "
                "succeeds (needs hardware:nvidia-gpu)")


def spec() -> dict:
    return deepcopy(SPEC)


@functools.lru_cache(maxsize=1)
def plan_iterations() -> int:
    """K: the iteration count ``ciw energy record`` plans for the declared problem (CPU planner, first KL <= target)."""
    from ciw import energy_bench
    return int(energy_bench.prepare_plan(spec())["iterations"])


def prepared_inputs(iterations: int | None = None, replicas: int = REPLICAS) -> np.ndarray:
    """The 15 binary64 kernel inputs, prepared exactly as the GPU worker prepares them (``INPUT_LAYOUT``)."""
    from ciw import energy_cuda
    k = plan_iterations() if iterations is None else iterations
    return energy_cuda._prepare(spec()["problem"], spec()["solver"], k, replicas, 0)


def kernel_sha256() -> str:
    from ciw import energy_cuda
    return hashlib.sha256(energy_cuda.PTX.encode("ascii")).hexdigest()


def workload(precision: str = "float64", implementation: str = "numpy") -> dict:
    """The declaration a capture or comparison names: kernel, problem, prepared inputs, K, replicas, precision."""
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    problem = spec()["problem"]
    return {"name": NAME, "kernel": "ciw.energy_cuda.PTX gaussian_vi", "kernel_sha256": kernel_sha256(),
            "problem_sha256": hashlib.sha256(json.dumps(problem, sort_keys=True, separators=(",", ":"),
                                                        allow_nan=False).encode()).hexdigest(),
            "prepared_input_sha256": hashlib.sha256(prepared_inputs().tobytes()).hexdigest(),
            "iterations": plan_iterations(), "replicas": REPLICAS, "precision": precision,
            "implementation": implementation}


# NumPy reference -------------------------------------------------------------
def _columns(values, dtype, replicas):
    """One array per input, ``replicas`` long, in ``dtype``: the float64 inputs rounded once to ``dtype``."""
    rounded = np.asarray(values, dtype=np.float64).astype(dtype)
    if rounded.shape != (len(INPUT_LAYOUT),):
        raise ValueError("The common workload takes 15 prepared inputs")
    return [np.full(replicas, value, dtype=dtype) for value in rounded]


def iteration(P00, P01, P10, P11, b0, b1, m0, m1, Q00, Q01, Q10, Q11, alpha, beta, omb):
    """One iteration in the kernel's operation order; returns the new (m0, m1, Q00, Q01, Q10, Q11)."""
    g0 = ((P00 * m0) + (P01 * m1)) - b0
    g1 = ((P10 * m0) + (P11 * m1)) - b1
    m0 = m0 - (alpha * g0)
    m1 = m1 - (alpha * g1)
    Q00 = (omb * Q00) + (beta * P00)
    Q01 = (omb * Q01) + (beta * P01)
    Q10 = (omb * Q10) + (beta * P10)
    Q11 = (omb * Q11) + (beta * P11)
    return m0, m1, Q00, Q01, Q10, Q11


def output(m0, m1, Q00, Q01, Q10, Q11):
    """The kernel's output block: the mean and the 2x2 inverse of the iterated precision."""
    det = (Q00 * Q11) - (Q01 * Q10)
    return m0, m1, Q11 / det, (-Q01) / det, (-Q10) / det, Q00 / det


def run_numpy(dtype=np.float64, iterations: int | None = None, replicas: int = REPLICAS, values=None) -> np.ndarray:
    """Outputs of every replica, shape (replicas, 6), computed entirely in ``dtype``."""
    dtype = np.dtype(dtype).type
    k = plan_iterations() if iterations is None else iterations
    columns = _columns(prepared_inputs() if values is None else values, dtype, replicas)
    fixed, state = columns[:6], columns[6:12]
    steps = columns[12:]
    for _ in range(k):
        state = list(iteration(*fixed, *state, *steps))
    result = np.stack(output(*state), axis=1)
    if result.dtype != np.dtype(dtype) or not np.all(np.isfinite(result)):
        raise FloatingPointError("The common workload left its declared precision or the finite domain")
    return result


def trace_numpy(dtype=np.float64, iterations: int = 256, values=None) -> np.ndarray:
    """Outputs of one replica after 0, 1, ..., ``iterations`` iterations, shape (iterations + 1, 6), in ``dtype``."""
    dtype = np.dtype(dtype).type
    columns = _columns(prepared_inputs() if values is None else values, dtype, 1)
    fixed, state, steps = columns[:6], columns[6:12], columns[12:]
    rows = []
    for index in range(iterations + 1):
        rows.append(np.concatenate(output(*state)))
        if index < iterations:
            state = list(iteration(*fixed, *state, *steps))
    return np.array(rows)


ARITHMETIC = (np.add, np.subtract, np.multiply, np.true_divide)


def counted_operations(dtype) -> dict:
    """Instrumented count of one iteration and of the output block per replica, on real ``dtype`` arrays.

    The inputs are counting views of ordinary ``dtype`` arrays: every ufunc
    call is counted per element (arithmetic, negation and anything else
    separately) and evaluated on the plain arrays, and every result dtype is
    recorded, so a constant promoted out of ``dtype`` shows up.
    """
    scalar = np.dtype(dtype).type
    counts = {"flops": 0, "negations": 0, "other_ufunc_calls": 0, "results_outside_dtype": 0}
    dtypes = set()

    class Counting(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            plain = [x.view(np.ndarray) if isinstance(x, Counting) else x for x in inputs]
            result = getattr(ufunc, method)(*plain, **kwargs)
            key = "flops" if ufunc in ARITHMETIC else "negations" if ufunc is np.negative else "other_ufunc_calls"
            counts[key] += int(np.size(result))
            dtypes.add(np.asarray(result).dtype.name)
            counts["results_outside_dtype"] += int(np.asarray(result).dtype != np.dtype(scalar))
            return result.view(Counting) if isinstance(result, np.ndarray) else result

    columns = [column.view(Counting) for column in _columns(prepared_inputs(), scalar, 1)]
    state = iteration(*columns[:6], *columns[6:12], *columns[12:])
    loop = dict(counts)
    for key in counts:
        counts[key] = 0
    output(*state)
    return {"iteration_flops": loop["flops"], "iteration_other": loop["negations"] + loop["other_ufunc_calls"],
            "output_flops": counts["flops"], "output_negations": counts["negations"],
            "output_other": counts["other_ufunc_calls"],
            "results_outside_dtype": loop["results_outside_dtype"] + counts["results_outside_dtype"],
            "result_dtypes": sorted(dtypes)}


PTX_ARITHMETIC = re.compile(r"^\s*(?:mul|add|sub|div)\.rn\.f64\b", re.MULTILINE)
PTX_NEGATION = re.compile(r"^\s*neg\.f64\b", re.MULTILINE)


def ptx_operation_counts() -> dict:
    """Arithmetic instructions of the gaussian_vi PTX kernel: loop body (per iteration) and output block."""
    from ciw import energy_cuda
    text = energy_cuda.PTX
    loop = text[text.index("LOOP:"):text.index("OUTPUT:")]
    block = text[text.index("OUTPUT:"):text.index("DONE:")]
    return {"iteration_flops": len(PTX_ARITHMETIC.findall(loop)), "output_flops": len(PTX_ARITHMETIC.findall(block)),
            "output_negations": len(PTX_NEGATION.findall(block)),
            "fused_instructions": len(re.findall(r"\b(?:fma|mad)\.[a-z0-9.]*f64\b", text))}


def flops_per_batch(iterations: int | None = None, replicas: int = REPLICAS) -> int:
    counts = ptx_operation_counts()
    k = plan_iterations() if iterations is None else iterations
    return replicas * (counts["iteration_flops"] * k + counts["output_flops"])


# Contracted and reassociated variants (binary64, exact FMA) -------------------
def fma(a: float, b: float, c: float) -> float:
    """a * b + c with one rounding to binary64 (exact rational arithmetic; correctly rounded int division)."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def run_variant(variant: str, iterations: int | None = None, values=None) -> np.ndarray:
    """One replica's six binary64 outputs with the kernel's reductions contracted or reassociated (``VARIANTS``)."""
    if variant not in ("kernel",) + VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    k = plan_iterations() if iterations is None else iterations
    P00, P01, P10, P11, b0, b1, m0, m1, Q00, Q01, Q10, Q11, alpha, beta, omb = (
        float(v) for v in (prepared_inputs() if values is None else values))
    pairs = ((P00, P01, b0), (P10, P11, b1))
    for _ in range(k):
        if variant == "fma-first":
            g = [fma(p0, m0, p1 * m1) - b for p0, p1, b in pairs]
            m0, m1 = fma(-alpha, g[0], m0), fma(-alpha, g[1], m1)
            Q00, Q01, Q10, Q11 = (fma(omb, q, beta * p) for q, p in ((Q00, P00), (Q01, P01), (Q10, P10), (Q11, P11)))
        elif variant == "fma-second":
            g = [fma(p1, m1, p0 * m0) - b for p0, p1, b in pairs]
            m0, m1 = fma(-alpha, g[0], m0), fma(-alpha, g[1], m1)
            Q00, Q01, Q10, Q11 = (fma(beta, p, omb * q) for q, p in ((Q00, P00), (Q01, P01), (Q10, P10), (Q11, P11)))
        else:
            g = ([p0 * m0 + (p1 * m1 - b) for p0, p1, b in pairs] if variant == "reassociated"
                 else [(p0 * m0 + p1 * m1) - b for p0, p1, b in pairs])
            m0, m1 = m0 - alpha * g[0], m1 - alpha * g[1]
            Q00, Q01, Q10, Q11 = (omb * q + beta * p for q, p in ((Q00, P00), (Q01, P01), (Q10, P10), (Q11, P11)))
    if variant == "fma-first":
        det = fma(Q00, Q11, -(Q01 * Q10))
    elif variant == "fma-second":
        det = fma(-Q01, Q10, Q00 * Q11)
    else:
        det = Q00 * Q11 - Q01 * Q10
    return np.array([m0, m1, Q11 / det, -Q01 / det, -Q10 / det, Q00 / det])


def ulp_distance(a, b, dtype=np.float64) -> int:
    """Largest distance in units in the last place between two arrays of ``dtype`` values, of any signs.

    Zero exactly when the arrays are bitwise equal: a sign flip, including
    +0.0 against -0.0, is a nonzero distance (``energy_gpu_kernels.ulp_distance``).
    """
    return energy_gpu_kernels.ulp_distance(a, b, dtype)


def differing_bits(a, b) -> int:
    """Number of float64 values whose bit patterns differ (``!=`` would equate +0.0 and -0.0)."""
    x = np.ascontiguousarray(a, dtype="<f8").view("<u8")
    y = np.ascontiguousarray(b, dtype="<f8").view("<u8")
    return int(np.sum(x != y))


# Accuracy ----------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _posterior():
    from ciw import free_energy_math
    reference = free_energy_math.gaussian_reference(spec()["problem"])
    return np.asarray(reference["mean"]), np.asarray(reference["covariance"])


def kl_nats(row) -> float:
    """KL(q || exact posterior) of one output row, by ciw.free_energy_math (observation-space reference)."""
    from ciw import free_energy_math
    row = np.asarray(row, dtype=np.float64)
    mean, covariance = _posterior()
    return free_energy_math.gaussian_kl(row[:2], row[2:].reshape(2, 2), mean, covariance)


def first_meeting(kls, target: float):
    """The first iteration whose KL meets ``target``, or None."""
    return next((index for index, value in enumerate(kls) if value <= target), None)


# Rust port -----------------------------------------------------------------
RUST_SCHEMA = "ciw.lab.rust-gaussian-vi.v1"
RUST_SOURCE = r'''// ciw.lab common workload: the Gaussian VI iteration of the gaussian_vi PTX kernel, std only.
// JSON over stdin/stdout. Every operation rounds to nearest in the declared precision; Rust never fuses a
// multiply and an add, matching the kernel's explicit .rn instructions.
use std::io::{self, Read, Write};

macro_rules! replica {
    ($name:ident, $t:ty) => {
        fn $name(v: &[f64; 15], iterations: u64, executed: &mut u64) -> [f64; 6] {
            let mut x = [0.0 as $t; 15];
            for i in 0..15 { x[i] = v[i] as $t; }
            let (p00, p01, p10, p11, b0, b1) = (x[0], x[1], x[2], x[3], x[4], x[5]);
            let (mut m0, mut m1) = (x[6], x[7]);
            let (mut q00, mut q01, mut q10, mut q11) = (x[8], x[9], x[10], x[11]);
            let (alpha, beta, omb) = (x[12], x[13], x[14]);
            for _ in 0..iterations {
                *executed += 1;
                let g0 = ((p00 * m0) + (p01 * m1)) - b0;
                let g1 = ((p10 * m0) + (p11 * m1)) - b1;
                m0 = m0 - (alpha * g0);
                m1 = m1 - (alpha * g1);
                q00 = (omb * q00) + (beta * p00);
                q01 = (omb * q01) + (beta * p01);
                q10 = (omb * q10) + (beta * p10);
                q11 = (omb * q11) + (beta * p11);
            }
            let det = (q00 * q11) - (q01 * q10);
            let out: [$t; 6] = [m0, m1, q11 / det, (-q01) / det, (-q10) / det, q00 / det];
            let mut wide = [0.0f64; 6];
            for i in 0..6 { wide[i] = out[i] as f64; }
            wide
        }
    };
}
replica!(replica_f64, f64);
replica!(replica_f32, f32);

fn numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_digit() || b == b'-' || b == b'.' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || b"+-.eE".contains(&bytes[i])) { i += 1; }
            let token = &text[start..i];
            values.push(token.parse::<f64>().map_err(|_| format!("invalid number {}", token))?);
        } else {
            i += 1;
        }
    }
    Ok(values)
}

fn field<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let quoted = format!("\"{}\"", key);
    let at = text.find(&quoted).ok_or(format!("missing field {}", key))?;
    let rest = &text[at + quoted.len()..];
    let colon = rest.find(':').ok_or(format!("malformed field {}", key))?;
    let rest = &rest[colon + 1..];
    if rest.trim_start().starts_with('[') {
        let open = rest.find('[').unwrap();
        let close = rest.find(']').ok_or(format!("unterminated array {}", key))?;
        Ok(&rest[open..close + 1])
    } else {
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Ok(rest[..end].trim())
    }
}

fn integer(text: &str, key: &str, low: f64, high: f64) -> Result<u64, String> {
    let value = *numbers(field(text, key)?)?.first().ok_or(format!("missing {}", key))?;
    if !(value >= low && value <= high && value.fract() == 0.0) {
        return Err(format!("{} must be an integer in [{}, {}]", key, low, high));
    }
    Ok(value as u64)
}

fn run(text: &str) -> Result<String, String> {
    let precision = field(text, "precision")?.trim_matches('"').to_string();
    let iterations = integer(text, "iterations", 0.0, 256.0)?;
    let replicas = integer(text, "replicas", 1.0, 65536.0)?;
    let repeats = integer(text, "repeats", 1.0, 1000.0)?;
    let flat = numbers(field(text, "values")?)?;
    if flat.len() != 15 { return Err("values must hold the 15 prepared inputs".into()); }
    if !flat.iter().all(|v| v.is_finite()) { return Err("values must be finite".into()); }
    let mut values = [0.0f64; 15];
    values.copy_from_slice(&flat);
    let kernel: fn(&[f64; 15], u64, &mut u64) -> [f64; 6] = match precision.as_str() {
        "float64" => replica_f64,
        "float32" => replica_f32,
        _ => return Err("precision must be float64 or float32".into()),
    };
    let mut executed: u64 = 0;
    let mut first = [0.0f64; 6];
    let mut identical = true;
    for repeat in 0..repeats {
        for index in 0..replicas {
            // black_box keeps every replica's work: the inputs are opaque to the optimizer.
            let out = kernel(std::hint::black_box(&values), std::hint::black_box(iterations), &mut executed);
            if repeat == 0 && index == 0 { first = out; }
            else if out.iter().zip(first.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) { identical = false; }
        }
    }
    if !first.iter().all(|v| v.is_finite()) { return Err("nonfinite output".into()); }
    let rows: Vec<String> = first.iter().map(|v| format!("{:?}", v)).collect();
    Ok(format!("{{\"schema\":\"ciw.lab.rust-gaussian-vi.v1\",\"precision\":\"{}\",\"iterations\":{},\"replicas\":{},\
\"repeats\":{},\"iterations_executed\":{},\"replicas_identical\":{},\"outputs\":[{}]}}",
               precision, iterations, replicas, repeats, executed, identical, rows.join(",")))
}

fn main() {
    let mut text = String::new();
    if io::stdin().read_to_string(&mut text).is_err() {
        eprintln!("unreadable input");
        std::process::exit(2);
    }
    match run(&text) {
        Ok(out) => { io::stdout().write_all(out.as_bytes()).unwrap(); }
        Err(message) => { eprintln!("{}", message); std::process::exit(2); }
    }
}
'''
RUST_IMPLEMENTATION = "ciw.lab.energy_gpu_workload.RUST_SOURCE"


def rust_source_sha256() -> str:
    return hashlib.sha256(RUST_SOURCE.encode("utf-8")).hexdigest()


def build_rust_port(directory) -> dict:
    """Compile the embedded Rust port with rustc; raises energy_gpu_kernels.NativeKernelUnavailable."""
    return energy_gpu_kernels.build_rust_program(directory, RUST_SOURCE, "gaussian_vi", RUST_IMPLEMENTATION)


def run_rust_port(executable, precision: str = "float64", iterations: int | None = None, replicas: int = REPLICAS,
                  repeats: int = 1, values=None, payload: str | None = None) -> dict:
    """Run the compiled port; floats cross as shortest round-trip text (float32 outputs widened exactly)."""
    k = plan_iterations() if iterations is None else iterations
    if payload is None:
        inputs = prepared_inputs() if values is None else values
        payload = json.dumps({"precision": precision, "iterations": int(k), "replicas": int(replicas),
                              "repeats": int(repeats), "values": [float(v) for v in inputs]}, allow_nan=False)
    try:
        done = subprocess.run([str(executable)], input=payload, capture_output=True, text=True, timeout=600,
                              env={**os.environ, "RUST_BACKTRACE": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise energy_gpu_kernels.NativeKernelUnavailable(f"Rust port could not run: {type(exc).__name__}") from exc
    if done.returncode != 0:
        raise energy_gpu_kernels.NativeKernelUnavailable("Rust port refused its input: " + done.stderr.strip()[:500])
    result = json.loads(done.stdout)
    if result.get("schema") != RUST_SCHEMA or len(result.get("outputs", [])) != len(OUTPUT_LAYOUT):
        raise energy_gpu_kernels.NativeKernelUnavailable("Rust port returned an unexpected document")
    dtype = np.float64 if result["precision"] == "float64" else np.float32
    result["outputs"] = np.asarray(result["outputs"], dtype=np.float64).astype(dtype)
    return result


# PTX kernel on the GPU ---------------------------------------------------------
def _reason(exc) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def run_gpu(iterations: int | None = None, replicas: int = REPLICAS, device_index: int = 0) -> dict:
    """The gaussian_vi PTX kernel on the common workload.

    Returns {"outputs", "identity"} when the kernel ran and the worker
    accepted its outputs; {"identity", "invalid": reason} when the kernel ran
    but ``CudaGaussianWorker.solve()`` rejected its outputs (nonfinite, out of
    bound, asymmetric or not positive definite covariance: a failed result,
    not a missing one); {"unavailable": reason} when no kernel result exists
    (driver, device, preparation or JIT failure while constructing the
    worker, or a launch, synchronization or copy failure reported by the
    driver). Call it only after ``ctx.available("hardware:nvidia-gpu")``
    succeeded in the calling task. A failure is never replaced by a CPU result.
    """
    from ciw import energy_cuda
    k = plan_iterations() if iterations is None else iterations
    try:
        worker = energy_cuda.CudaGaussianWorker(spec()["problem"], spec()["solver"], k, replicas=replicas,
                                                device_index=device_index)
    except (energy_cuda.CudaError, ValueError, OSError) as exc:
        return {"unavailable": "the PTX kernel did not run: " + _reason(exc)}
    outcome = None
    try:
        with worker:
            identity = worker.identity()
            try:
                outcome = {"outputs": np.asarray(worker.solve(), dtype=np.float64), "identity": identity}
            except ValueError as exc:  # raised by solve() after launch, sync and copy: the outputs failed its checks
                outcome = {"identity": identity, "invalid": "the worker rejected the GPU outputs: " + _reason(exc)}
    except (energy_cuda.CudaError, OSError) as exc:
        if outcome is None:
            return {"unavailable": "the PTX kernel did not complete: " + _reason(exc)}
        # A failure releasing the context after solve() returned does not change the kernel's result.
    return outcome


def gpu_run(ctx) -> dict:
    """Probe hardware:nvidia-gpu in the calling task; run the kernel once per lab run (shared through the memo).

    Returns {"probed": bool, "ran": bool, "reason": str | None, "invalid":
    str | None, "outputs", "identity", "reference"}; the reference is the
    float64 NumPy run on the same prepared inputs, and ``input_matches`` says
    whether the worker's prepared-input digest equals the reference's. When
    the kernel ran but the worker rejected its outputs, ``ran`` is True,
    ``outputs`` is None and ``invalid`` names the rejection: a comparison must
    record that as a failed check, never as a gap.
    """
    if not ctx.available("hardware:nvidia-gpu"):
        return {"probed": False, "ran": False, "reason": NO_GPU_PROBE, "invalid": None}
    result = ctx.memo("common-workload-gpu", run_gpu)
    if "unavailable" in result:
        return {"probed": True, "ran": False, "reason": result["unavailable"], "invalid": None}
    reference = ctx.memo("common-workload-numpy-float64", lambda: run_numpy(np.float64))
    digest = hashlib.sha256(prepared_inputs().tobytes()).hexdigest()
    return {"probed": True, "ran": True, "reason": None, "invalid": result.get("invalid"),
            "outputs": result.get("outputs"), "identity": result["identity"], "reference": reference,
            "input_matches": result["identity"].get("prepared_input_sha256") == digest}
