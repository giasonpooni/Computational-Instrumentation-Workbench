"""Ordinary adapter for the existing analytic damped oscillator.

Render geometry is a derived view. Every analysis in this module reads the full
precision channel arrays and validates its complete source recording first.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from .protocol import AdapterRefusal, InstrumentManifest


_UNITS = {"q": "m", "v": "m/s", "energy": "J"}
_FRAME = "oscillator-state"


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _array(value: Any, name: str, *, ndim: int = 1) -> np.ndarray:
    def contains_boolean(item: Any) -> bool:
        if isinstance(item, (list, tuple)):
            return any(contains_boolean(child) for child in item)
        return isinstance(item, (bool, np.bool_))

    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.ndim != ndim or contains_boolean(value):
            raise ValueError
        result = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a {ndim}-dimensional numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _finite_tree(value: Any, name: str = "run") -> None:
    """Reject hidden nonfinite values even outside the requested analysis channel."""
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, np.ndarray):
        _array(value, name, ndim=value.ndim)
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        _number(value, name)


def analytic_trajectory(omega_0: float, gamma: float, mass: float, q0: float, v0: float,
                        time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form underdamped state q, v and energy at the given times.

    The equation is q'' + 2 gamma q' + omega_0**2 q = 0 with the initial state
    at t = 0 and gamma < omega_0. This is the workbench's analytical reference
    for numerical integrators; the operation order is fixed so the demo
    recording and oracle comparisons stay bitwise reproducible on one host.
    """
    omega_d = math.sqrt(omega_0 * omega_0 - gamma * gamma)
    b = (v0 + gamma * q0) / omega_d
    cosine, sine = np.cos(omega_d * time), np.sin(omega_d * time)
    envelope = np.exp(-gamma * time)
    q = envelope * (q0 * cosine + b * sine)
    v = envelope * ((b * omega_d - gamma * q0) * cosine
                    + (-q0 * omega_d - gamma * b) * sine)
    energy = 0.5 * mass * (v * v + omega_0 * omega_0 * q * q)
    return q, v, energy


def make_demo_run() -> dict:
    """Return a deterministic analytic underdamped oscillator recording.

    The equation is q'' + 2 gamma q' + omega_0**2 q = 0. The
    mechanical energy is a Lyapunov function: E' = -2 m gamma v**2.
    The endpoint is excluded, giving exactly 768 samples over [0, 12).
    """
    duration, sample_rate = 12.0, 64.0
    omega_0, gamma, mass, q0, v0 = 2.0 * math.pi * 0.8, 0.15, 1.0, 1.0, 0.0
    time = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    q, v, energy = analytic_trajectory(omega_0, gamma, mass, q0, v0, time)

    # A sampled state-space energy surface, not a field measurement.
    q_axis = np.linspace(-1.1, 1.1, 25, dtype=np.float64)
    v_axis = np.linspace(-1.1 * omega_0, 1.1 * omega_0, 25, dtype=np.float64)
    vertices = [[float(qi), float(0.5 * mass * (vi * vi + omega_0**2 * qi * qi)),
                 float(vi)] for vi in v_axis for qi in q_axis]
    indices = []
    width = len(q_axis)
    for row in range(len(v_axis) - 1):
        for column in range(width - 1):
            a = row * width + column
            indices.extend([a, a + width, a + 1, a + 1, a + width, a + width + 1])

    channels = {
        name: {"unit": _UNITS[name], "values": values.tolist()}
        for name, values in (("q", q), ("v", v), ("energy", energy))
    }
    metadata = {
        "duration_s": duration,
        "sample_rate_hz": sample_rate,
        "sample_count": len(time),
        "coordinate_frame": _FRAME,
        "model": {
            "equation": "q'' + 2*gamma*q' + omega_0^2*q = 0",
            "mass_kg": mass,
            "omega_0_rad_s": omega_0,
            "gamma_s_inv": gamma,
            "initial_q_m": q0,
            "initial_v_m_s": v0,
            "energy_definition": "0.5*mass*(v^2 + omega_0^2*q^2)",
        },
        "provenance": {
            "source": "analytic model; synthetic evidence, not sensor acquisition",
            "generator": "ciw.instruments.make_demo_run",
            "generator_version": 1,
            "dtype": "float64",
            "time_reference": "seconds since run start",
            "sampling": "uniform; endpoint excluded",
        },
    }
    scientific_record = {"instrument": "analytic-damped-oscillator.v1",
                         "metadata": metadata, "time_s": time.tolist(), "channels": channels}
    digest = hashlib.sha256(json.dumps(scientific_record, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    run = {
        "run_id": "run-damped-oscillator-demo-v1",
        "evidence_id": f"sha256:{digest}",
        **scientific_record,
        "render": {
            "coordinate_frame": _FRAME,
            "axis_labels": ["q (m)", "energy (J)", "v (m/s)"],
            "trajectory": np.column_stack((q, energy, v)).tolist(),
            "sample_indices": list(range(len(time))),
            "surface": {"vertices": vertices, "indices": indices},
            "transform": {
                "origin": [0.0, 0.0, 0.0],
                "scale": [2.5, 0.15, 0.5],
                "note": "Visual display scaling only; source coordinates and units stay physical.",
            },
        },
    }
    validate_run(run)
    return run


def validate_run(run: dict) -> None:
    """Validate every source channel and the complete v1 record contract."""
    _mapping(run, "run")
    for name in ("run_id", "evidence_id", "instrument"):
        _string(run.get(name), name)
    _finite_tree(run)
    metadata = _mapping(run.get("metadata"), "metadata")
    duration = _number(metadata.get("duration_s"), "duration_s")
    sample_rate = _number(metadata.get("sample_rate_hz"), "sample_rate_hz")
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration_s and sample_rate_hz must be positive")
    count = _integer(metadata.get("sample_count"), "sample_count", 2)
    if metadata.get("coordinate_frame") != _FRAME:
        raise ValueError(f"coordinate_frame must be {_FRAME}")
    _mapping(metadata.get("model"), "metadata.model")
    _mapping(metadata.get("provenance"), "metadata.provenance")

    time = _array(run.get("time_s"), "time_s")
    if len(time) != count:
        raise ValueError("time_s length must match sample_count")
    spacing = 1.0 / sample_rate
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("sample_rate_hz does not define a finite positive spacing")
    if time[0] != 0.0 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must start at zero and be strictly increasing")
    if not np.allclose(np.diff(time), spacing, rtol=1e-8, atol=spacing * 1e-10):
        raise ValueError("time_s must be uniformly sampled at sample_rate_hz")
    if time[-1] >= duration or not math.isclose(duration, count / sample_rate,
                                               rel_tol=1e-8, abs_tol=spacing * 1e-10):
        raise ValueError("duration_s must equal sample_count / sample_rate_hz; endpoint is excluded")

    channels = _mapping(run.get("channels"), "channels")
    if set(channels) != set(_UNITS):
        raise ValueError("channels must contain exactly q, v, and energy")
    for name, unit in _UNITS.items():
        channel = _mapping(channels[name], f"channels.{name}")
        if channel.get("unit") != unit:
            raise ValueError(f"channel {name} unit must be {unit}")
        values = _array(channel.get("values"), f"channels.{name}.values")
        if len(values) != count:
            raise ValueError(f"channel {name} length must match sample_count")

    render = _mapping(run.get("render"), "render")
    if render.get("coordinate_frame") != _FRAME:
        raise ValueError("render coordinate_frame must match the source frame")
    if render.get("axis_labels") != ["q (m)", "energy (J)", "v (m/s)"]:
        raise ValueError("render axis_labels must identify q, energy, and v with units")
    trajectory = _array(render.get("trajectory"), "render.trajectory", ndim=2)
    if trajectory.shape[1] != 3 or len(trajectory) == 0:
        raise ValueError("render.trajectory must contain 3D points")
    sample_indices = render.get("sample_indices")
    if not isinstance(sample_indices, (list, tuple)) or len(sample_indices) != len(trajectory):
        raise ValueError("render.sample_indices must match the trajectory length")
    previous = -1
    for index in sample_indices:
        index = _integer(index, "render.sample_indices entry")
        if index >= count or index <= previous:
            raise ValueError("render.sample_indices must be increasing valid source indices")
        previous = index
    surface = _mapping(render.get("surface"), "render.surface")
    vertices = _array(surface.get("vertices"), "render.surface.vertices", ndim=2)
    if vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("render surface must contain at least three 3D vertices")
    mesh_indices = surface.get("indices")
    if not isinstance(mesh_indices, (list, tuple)) or not mesh_indices or len(mesh_indices) % 3:
        raise ValueError("render surface indices must contain complete triangles")
    for index in mesh_indices:
        if _integer(index, "render surface index") >= len(vertices):
            raise ValueError("render surface index is outside the vertices array")
    transform = _mapping(render.get("transform"), "render.transform")
    for name in ("origin", "scale"):
        vector = _array(transform.get(name), f"render.transform.{name}")
        if len(vector) != 3 or (name == "scale" and np.any(vector == 0)):
            raise ValueError(f"render.transform.{name} must have three valid components")
    _string(transform.get("note"), "render.transform.note")


def _selected(run: dict, channel: str, interval_s: Any) -> tuple[np.ndarray, str, float]:
    validate_run(run)
    if not isinstance(channel, str) or channel not in run["channels"]:
        raise ValueError("channel must be one of q, v, energy")
    if (not isinstance(interval_s, (list, tuple, np.ndarray))
            or (isinstance(interval_s, np.ndarray) and interval_s.ndim != 1)
            or len(interval_s) != 2):
        raise ValueError("interval_s must be [start, end]")
    start, end = (_number(value, "interval_s endpoint") for value in interval_s)
    if not 0 <= start < end <= run["metadata"]["duration_s"]:
        raise ValueError("interval_s must satisfy 0 <= start < end <= duration_s")
    time = np.asarray(run["time_s"], dtype=np.float64)
    first, last = np.searchsorted(time, [start, end], side="left")
    selected = np.asarray(run["channels"][channel]["values"], dtype=np.float64)[first:last]
    if len(selected) == 0:
        raise ValueError("interval_s contains no retained samples")
    return selected, run["channels"][channel]["unit"], float(run["metadata"]["sample_rate_hz"])


def compute_statistics(run: dict, channel: str, interval_s: list[float]) -> dict:
    """Compute statistics from all source samples in the half-open interval."""
    values, unit, _ = _selected(run, channel, interval_s)
    # Scale before reductions to avoid overflow for otherwise valid large inputs.
    scale = float(np.max(np.abs(values)))
    normalized = values / scale if scale else values
    return {
        "sample_count": len(values),
        "mean": float(np.mean(normalized) * scale),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(normalized * normalized)) * scale),
        "unit": unit,
    }


def compute_spectrum(run: dict, channel: str, interval_s: list[float]) -> dict:
    """Compute a one-sided, constant-detrended periodic-Hann periodogram.

    Density normalization is fs * sum(window**2). Interior positive-frequency
    bins are doubled; DC and the even-length Nyquist bin are not. Integral PSD
    equals the window-weighted detrended mean square, not the unwindowed RMS.
    """
    values, unit, sample_rate = _selected(run, channel, interval_s)
    count = len(values)
    if count < 4:
        raise ValueError("spectrum requires at least four retained samples")
    # np.hanning(n + 1)[:-1] is the periodic (DFT-even) Hann window.
    window = np.hanning(count + 1)[:-1]
    scale = float(np.max(np.abs(values)))
    normalized = values / scale if scale else values
    centered = normalized - np.mean(normalized)
    with np.errstate(over="ignore", invalid="ignore"):
        transform = np.fft.rfft(centered * window)
        psd = np.abs(transform) ** 2 / (sample_rate * np.sum(window * window))
        psd[1:-1 if count % 2 == 0 else None] *= 2.0
        # Multiplication in this order avoids needlessly overflowing scale**2.
        psd = (psd * scale) * scale
    if not np.all(np.isfinite(psd)):
        raise ValueError("spectrum exceeds finite float64 density range")
    frequencies = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    peak = float(frequencies[int(np.argmax(psd))]) if np.any(psd > 0.0) else None
    return {
        "sample_count": count, "method": "periodogram", "window": "hann",
        "detrend": "constant", "scaling": "density",
        "frequency_hz": frequencies.tolist(), "psd": psd.tolist(),
        "unit": f"({unit})^2/Hz", "peak_frequency_hz": peak,
        "sample_rate_hz": sample_rate,
    }


class OscillatorAdapter:
    """The legacy scientific implementation, now behind the shared adapter seam."""

    manifest = InstrumentManifest(
        instrument_id="analytic-damped-oscillator.v1",
        inputs=("analytic-oscillator-parameters.v1",),
        outputs=("run.v1", "statistics.v1", "spectrum.periodogram.v1"),
        units=dict(_UNITS), frames=(_FRAME,),
        sampling={"kind": "uniform", "time_unit": "s", "endpoint": "excluded"},
        normalization={"statistics": "population", "spectrum": "one-sided density; periodic Hann"},
        supported_operations=("statistics.v1", "spectrum.periodogram.v1"),
        determinism={"kind": "deterministic", "dtype": "float64"},
        tolerance_policy={"sampling_rtol": 1e-8, "sampling_atol_spacing_factor": 1e-10,
                          "replay": "numeric outputs compared separately from event identities"},
        calibration_requirements={"required": False, "reason": "synthetic analytic model"},
    )

    def validate_run(self, run: dict) -> None:
        if run.get("instrument") != self.manifest.instrument_id:
            raise ValueError("Run instrument does not match oscillator adapter")
        validate_run(run)

    def execute(self, operation_id: str, run: dict, parameters: dict) -> dict:
        self.validate_run(run)
        if parameters.keys() != {"channel", "interval_s"}:
            raise ValueError("Oscillator operations require channel and interval_s only")
        if operation_id == "statistics.v1":
            return compute_statistics(run, parameters["channel"], parameters["interval_s"])
        if operation_id == "spectrum.periodogram.v1":
            return compute_spectrum(run, parameters["channel"], parameters["interval_s"])
        raise AdapterRefusal("unsupported_operation", f"Oscillator does not support {operation_id}")
