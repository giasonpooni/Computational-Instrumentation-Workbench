"""Numerical kernels for the energy and GPU experiments (T115-T123).

Scope: a closed-form unit-sphere geodesic right-hand side shared by a Python
RK4 (any NumPy float dtype) and an embedded std-only Rust RK4 compiled at run
time; CPU emulations of GPU-style reduction orders; and a small typed-quantity
algebra that refuses to combine information (nats) with energy (joules).

Non-claims: nothing here measures energy, power or time on any device. The
reduction functions emulate summation orders on the CPU; they do not show how
a particular GPU, driver or library orders its reductions. Agreement between
the Python and Rust kernels is agreement between two ciw implementations of
the same algorithm, not independent verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

# Geodesic workload ------------------------------------------------------
# Unit sphere, polar chart u = (theta, phi). A great circle leaving the
# equator at heading a from the meridian stays at polar distance >= a, so
# headings >= 0.7 rad keep every trajectory away from the chart singularities.
THETA0 = math.pi / 2
HEADINGS = (0.7, 0.85, 1.0, 1.15, 1.3, 1.45)
LENGTH = 3.0

# Per-step operation count of the closed-form RK4 below (see rk4_operation_count).
RHS_MUL, RHS_DIV, RHS_TRIG = 6, 1, 2


def initial_states(headings=HEADINGS) -> np.ndarray:
    """Rows (theta, phi, dtheta/ds, dphi/ds) of unit-speed geodesics on the equator."""
    return np.array([[THETA0, 0.0, math.cos(a), math.sin(a) / math.sin(THETA0)] for a in headings])


def sphere_rhs(y):
    """theta'' = sin cos phi'^2, phi'' = -2 cot(theta) theta' phi' (scalar math, float64)."""
    s, c = math.sin(y[0]), math.cos(y[0])
    return np.array([y[2], y[3], s * c * y[3] * y[3], -2.0 * (c / s) * y[2] * y[3]])


def sphere_rhs_batch(y, dtype):
    """Vectorized right-hand side on a (4, batch) array evaluated in ``dtype`` arithmetic."""
    s, c = np.sin(y[0]), np.cos(y[0])
    return np.stack([y[2], y[3], s * c * y[3] * y[3], dtype(-2.0) * (c / s) * y[2] * y[3]])


def rk4_batch(states, length, steps, dtype=np.float64) -> np.ndarray:
    """Fixed-step RK4 with every operation in ``dtype`` (NumPy 2 keeps float32 closed)."""
    dtype = np.dtype(dtype).type
    y = np.asarray(states, dtype=dtype).T.copy()
    h = dtype(length / steps)
    half, sixth, two = dtype(0.5) * h, h / dtype(6.0), dtype(2.0)
    for _ in range(steps):
        k1 = sphere_rhs_batch(y, dtype)
        k2 = sphere_rhs_batch(y + half * k1, dtype)
        k3 = sphere_rhs_batch(y + half * k2, dtype)
        k4 = sphere_rhs_batch(y + h * k3, dtype)
        y = y + sixth * (k1 + two * k2 + two * k3 + k4)
    if y.dtype != np.dtype(dtype) or not np.all(np.isfinite(y)):
        raise FloatingPointError("RK4 left its declared precision or the finite domain")
    return y.T


def embed(theta, phi) -> np.ndarray:
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])


def great_circle_endpoint(state, length) -> np.ndarray:
    """Exact endpoint X(L) = cos(L) X0 + sin(L) T0 on the unit sphere (float64)."""
    theta, phi, dtheta, dphi = (float(v) for v in state)
    x0 = embed(theta, phi)
    x_theta = np.array([math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)])
    x_phi = np.array([-math.sin(theta) * math.sin(phi), math.sin(theta) * math.cos(phi), 0.0])
    tangent = dtheta * x_theta + dphi * x_phi
    return math.cos(length) * x0 + math.sin(length) * tangent


def endpoint_errors(initial, final, length) -> np.ndarray:
    """Embedded distance between integrated and exact endpoints, evaluated in float64."""
    return np.array([float(np.linalg.norm(embed(float(f[0]), float(f[1])) - great_circle_endpoint(i, length)))
                     for i, f in zip(initial, final)])


def rk4_operation_count() -> dict:
    """Arithmetic per RK4 step of the closed-form kernel; identical for every precision."""
    rhs = 4 * (RHS_MUL + RHS_DIV)
    stages = 3 * 4 * 2            # y + c * k for three intermediate stages
    combine = 4 * (2 + 3) + 4 * 2  # k1 + 2 k2 + 2 k3 + k4, then y + (h/6) * sum
    return {"flops_per_step": rhs + stages + combine, "transcendentals_per_step": 4 * RHS_TRIG,
            "rhs_evaluations_per_step": 4, "state_values": 4}


# Rust kernel ------------------------------------------------------------
# The arithmetic mirrors ciw.lab.integrators.step_rk4 with sphere_rhs term by
# term: (0.5 h) k, ((k1 + 2 k2) + 2 k3) + k4 and (h / 6) * sum, no FMA.
RUST_SOURCE = r'''// ciw.lab energy_gpu T117: unit-sphere geodesic RK4, std only, JSON over stdin/stdout.
use std::io::{self, Read, Write};

fn rhs(y: &[f64; 4]) -> [f64; 4] {
    let s = y[0].sin();
    let c = y[0].cos();
    [y[2], y[3], s * c * y[3] * y[3], -2.0 * (c / s) * y[2] * y[3]]
}

fn step(y: &[f64; 4], h: f64) -> [f64; 4] {
    let hh = 0.5 * h;
    let k1 = rhs(y);
    let mut t = [0.0; 4];
    for i in 0..4 { t[i] = y[i] + hh * k1[i]; }
    let k2 = rhs(&t);
    for i in 0..4 { t[i] = y[i] + hh * k2[i]; }
    let k3 = rhs(&t);
    for i in 0..4 { t[i] = y[i] + h * k3[i]; }
    let k4 = rhs(&t);
    let h6 = h / 6.0;
    let mut out = [0.0; 4];
    for i in 0..4 { out[i] = y[i] + h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]); }
    out
}

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
        let mut depth = 0usize;
        for (offset, ch) in rest[open..].char_indices() {
            if ch == '[' { depth += 1; }
            if ch == ']' {
                depth -= 1;
                if depth == 0 { return Ok(&rest[open..open + offset + 1]); }
            }
        }
        Err(format!("unterminated array {}", key))
    } else {
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Ok(&rest[..end])
    }
}

fn run(text: &str) -> Result<String, String> {
    let length = *numbers(field(text, "length")?)?.first().ok_or("missing length")?;
    let steps = *numbers(field(text, "steps")?)?.first().ok_or("missing steps")?;
    let flat = numbers(field(text, "states")?)?;
    if !(length.is_finite() && length > 0.0) { return Err("length must be positive and finite".into()); }
    if !(steps >= 1.0 && steps <= 1.0e6 && steps.fract() == 0.0) { return Err("steps must be an integer in [1, 1e6]".into()); }
    if flat.is_empty() || flat.len() % 4 != 0 { return Err("states must be a list of 4-vectors".into()); }
    let n = steps as usize;
    let h = length / (n as f64);
    let mut rows = Vec::new();
    let mut evaluations: u64 = 0;
    for chunk in flat.chunks(4) {
        let mut y = [chunk[0], chunk[1], chunk[2], chunk[3]];
        for _ in 0..n {
            y = step(&y, h);
            evaluations += 4;
        }
        if !y.iter().all(|v| v.is_finite()) { return Err("nonfinite state".into()); }
        rows.push(format!("[{:?},{:?},{:?},{:?}]", y[0], y[1], y[2], y[3]));
    }
    Ok(format!("{{\"schema\":\"ciw.lab.rust-sphere-rk4.v1\",\"steps\":{},\"evaluations\":{},\"states\":[{}]}}",
               n, evaluations, rows.join(",")))
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
RUST_SCHEMA = "ciw.lab.rust-sphere-rk4.v1"


class NativeKernelUnavailable(RuntimeError):
    """The Rust toolchain is absent or the kernel could not be built or run."""


def rust_source_sha256() -> str:
    return hashlib.sha256(RUST_SOURCE.encode("utf-8")).hexdigest()


def build_rust_kernel(directory) -> dict:
    """Compile the embedded source with rustc into ``directory``; return its identity."""
    rustc = shutil.which("rustc")
    if rustc is None:
        raise NativeKernelUnavailable("rustc is not on PATH")
    directory = Path(directory)
    source = directory / "sphere_rk4.rs"
    source.write_text(RUST_SOURCE, encoding="utf-8", newline="\n")
    executable = directory / ("sphere_rk4.exe" if sys.platform == "win32" else "sphere_rk4")
    command = [rustc, "-O", "-C", "debuginfo=0", "--edition", "2021", "-o", str(executable), str(source)]
    try:
        built = subprocess.run(command, capture_output=True, text=True, timeout=120)
        version = subprocess.run([rustc, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeKernelUnavailable(f"rustc could not run: {exc}") from exc
    if built.returncode != 0 or not executable.is_file():
        raise NativeKernelUnavailable("rustc failed: " + built.stderr.strip()[:2000])
    return {"implementation": "ciw.lab.energy_gpu_kernels.RUST_SOURCE", "language": "rust", "rustc": version,
            "flags": command[1:6], "source_sha256": rust_source_sha256(),
            "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "executable": str(executable)}


def run_rust_kernel(executable, states, length, steps) -> dict:
    """Exchange JSON with the compiled kernel; floats cross as shortest round-trip text."""
    payload = json.dumps({"length": float(length), "steps": int(steps),
                          "states": [[float(v) for v in row] for row in states]}, allow_nan=False)
    try:
        done = subprocess.run([str(executable)], input=payload, capture_output=True, text=True, timeout=120,
                              env={**os.environ, "RUST_BACKTRACE": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeKernelUnavailable(f"Rust kernel could not run: {exc}") from exc
    if done.returncode != 0:
        raise NativeKernelUnavailable("Rust kernel refused its input: " + done.stderr.strip()[:500])
    result = json.loads(done.stdout)
    if result.get("schema") != RUST_SCHEMA or len(result.get("states", [])) != len(states):
        raise NativeKernelUnavailable("Rust kernel returned an unexpected document")
    return result


def ulp_distance(a, b) -> int:
    """Largest distance in units in the last place between two float64 arrays of equal sign."""
    a = np.ascontiguousarray(a, dtype="<f8").view("<i8")
    b = np.ascontiguousarray(b, dtype="<f8").view("<i8")
    return int(np.max(np.abs(a - b))) if a.size else 0


# Reduction orders -------------------------------------------------------
BLOCK = 256


def _pad(x, multiple):
    extra = (-len(x)) % multiple
    return np.concatenate([x, np.zeros(extra, dtype=x.dtype)]) if extra else x


def sum_sequential(x):
    """Left fold, one thread: ((x0 + x1) + x2) + ... (ufunc accumulate is strictly sequential)."""
    return np.add.accumulate(x)[-1]


def sum_tree_adjacent(x):
    """Pairwise tree pairing neighbours (x0 + x1), (x2 + x3), ... at every level."""
    while len(x) > 1:
        x = _pad(x, 2)
        x = x[0::2] + x[1::2]
    return x[0]


def sum_tree_strided(x):
    """Shared-memory style tree with sequential addressing: x[i] += x[i + n/2]."""
    while len(x) > 1:
        x = _pad(x, 2)
        half = len(x) // 2
        x = x[:half] + x[half:]
    return x[0]


def _block_partials(x, block=BLOCK):
    """Per-block strided tree (one CUDA block each), returning one partial per block."""
    blocks = _pad(x, block).reshape(-1, block)
    while blocks.shape[1] > 1:
        half = blocks.shape[1] // 2
        blocks = blocks[:, :half] + blocks[:, half:]
    return blocks[:, 0]


def sum_blocked(x, block=BLOCK):
    """Block partials, then a second strided-tree pass over the partials (two-kernel reduction)."""
    return sum_tree_strided(_block_partials(x, block))


def sum_block_sequential(x, block=BLOCK):
    """Each block sums sequentially, block totals are then folded in block order."""
    blocks = _pad(x, block).reshape(-1, block)
    return sum_sequential(np.add.accumulate(blocks, axis=1)[:, -1])


def sum_atomic(x, order, block=BLOCK):
    """Block partials accumulated by atomicAdd in a given completion order."""
    return sum_sequential(_block_partials(x, block)[order])


def sum_kahan(x):
    """Compensated summation in the array's precision."""
    kind = x.dtype.type
    total, compensation = kind(0), kind(0)
    for value in x:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def reduction_orders(x, atomic_orders) -> dict:
    """Sum ``x`` in every emulated order; values are Python floats of the array's precision."""
    results = {"sequential": sum_sequential(x), "tree-adjacent": sum_tree_adjacent(x),
               "tree-strided": sum_tree_strided(x), "blocked-two-pass": sum_blocked(x),
               "block-sequential": sum_block_sequential(x), "kahan": sum_kahan(x)}
    for index, order in enumerate(atomic_orders):
        results[f"atomic-{index}"] = sum_atomic(x, order)
    return {name: float(value) for name, value in results.items()}


def unit_roundoff(dtype) -> float:
    return float(np.finfo(dtype).eps) / 2.0


def summation_bound(x, dtype) -> float:
    """|computed - exact| <= gamma_{n-1} sum|x_i| for every summation order (Higham, Accuracy
    and Stability of Numerical Algorithms, 2nd ed., section 4.2); Kahan's bound is smaller."""
    n, u = len(x), unit_roundoff(dtype)
    gamma = (n - 1) * u / (1 - (n - 1) * u)
    return gamma * math.fsum(abs(float(v)) for v in x)


# Typed quantities -------------------------------------------------------
# Units map to (dimension, scale to the dimension's base unit). Information
# and energy are different dimensions; relating them needs a physical model
# (for example k_B T per nat at a declared temperature) that is not implied
# by a variational calculation.
UNITS = {"nat": ("information", 1.0), "bit": ("information", math.log(2.0)),
         "J": ("energy", 1.0), "mJ": ("energy", 1e-3), "kJ": ("energy", 1e3),
         "s": ("time", 1.0), "ms": ("time", 1e-3), "solve": ("count", 1.0), "1": ("dimensionless", 1.0)}


class QuantityRefusal(ValueError):
    """Two quantities of different dimensions were combined or compared."""


class Quantity:
    """A float with a dimension vector; addition and comparison require equal dimensions."""

    __slots__ = ("value", "dimension", "unit")

    def __init__(self, value, unit=None, *, dimension=None):
        if dimension is None:
            if unit not in UNITS:
                raise QuantityRefusal(f"Unsupported unit: {unit}")
            name, scale = UNITS[unit]
            dimension = {} if name == "dimensionless" else {name: 1}
            value = float(value) * scale
        if not math.isfinite(float(value)):
            raise QuantityRefusal("Quantities must be finite")
        self.value = float(value)
        self.dimension = {k: v for k, v in sorted(dimension.items()) if v}
        self.unit = unit or self.describe()

    def describe(self) -> str:
        top = [k for k, v in self.dimension.items() if v > 0]
        bottom = [k for k, v in self.dimension.items() if v < 0]
        return ("*".join(top) or "1") + ("/" + "/".join(bottom) if bottom else "")

    def _same(self, other, action):
        if not isinstance(other, Quantity):
            raise QuantityRefusal(f"Cannot {action} a quantity and an untyped number")
        if self.dimension != other.dimension:
            raise QuantityRefusal(f"Cannot {action} {self.describe()} and {other.describe()}")

    def __add__(self, other):
        self._same(other, "add")
        return Quantity(self.value + other.value, dimension=self.dimension)

    def __sub__(self, other):
        self._same(other, "subtract")
        return Quantity(self.value - other.value, dimension=self.dimension)

    def __lt__(self, other):
        self._same(other, "compare")
        return self.value < other.value

    def __le__(self, other):
        self._same(other, "compare")
        return self.value <= other.value

    def __mul__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityRefusal("Scale quantities by typed dimensionless quantities")
        merged = dict(self.dimension)
        for key, power in other.dimension.items():
            merged[key] = merged.get(key, 0) + power
        return Quantity(self.value * other.value, dimension=merged)

    def __truediv__(self, other):
        if not isinstance(other, Quantity):
            raise QuantityRefusal("Divide quantities by typed quantities")
        merged = dict(self.dimension)
        for key, power in other.dimension.items():
            merged[key] = merged.get(key, 0) - power
        return Quantity(self.value / other.value, dimension=merged)

    def to(self, unit) -> float:
        """Value in ``unit``; refused unless the unit has this quantity's dimension."""
        name, scale = UNITS.get(unit, (None, None))
        target = {} if name == "dimensionless" else {name: 1}
        if name is None or target != self.dimension:
            raise QuantityRefusal(f"Cannot express {self.describe()} in {unit}")
        return self.value / scale
