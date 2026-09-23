"""Typed instrument observation modes, frame and clock bases, and retention without admission.

Scope: the record types and refusals used by the observation experiments
(T045, T055, T056, T058, T059). A mode declares the quantity it observes, its
unit, the frame kind and clock basis a record must carry, whether it sees
intrinsic or extrinsic surface geometry, declared noise-model parameters and
what it cannot observe. Validation refuses records without frame, clock or
calibration references and refuses substituting one mode for another. Frame
and clock mappings are applied only when declared, and are recorded on the
result. Retaining an observation never changes estimator state; only an
admitted observation, bound by content digest, may update it.

Non-claims: the noise parameters are declared placeholders, not measured
instrument characteristics. A validated record is well formed, not physically
true. Admission here is workbench bookkeeping; it confers no physical,
calibration or actuator authority.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math

import numpy as np

from ..core.identities import content_identity

CLOCK_BASES = ("acquisition", "arrival")
GEOMETRY_CLASSES = ("intrinsic", "extrinsic", "none")
RETENTION, NOT_ADMITTED, ADMITTED = "retained", "not_performed", "admitted"


class ObservationRefusal(ValueError):
    """A record or operation is refused; ``code`` is the stable refusal reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ObservationMode:
    name: str
    quantity: str
    unit: str
    components: int
    frame_kind: str
    clock_basis: str
    geometry: str
    requires_surface_model: bool
    noise_model: dict
    cannot_observe: tuple

    def describe(self) -> dict:
        return {"name": self.name, "quantity": self.quantity, "unit": self.unit, "components": self.components,
                "frame_kind": self.frame_kind, "clock_basis": self.clock_basis, "geometry": self.geometry,
                "requires_surface_model": self.requires_surface_model, "noise_model": deepcopy(self.noise_model),
                "cannot_observe": list(self.cannot_observe)}


_PIXEL_NOISE = {"model": "gaussian_pixel_noise_then_integer_rounding", "pixel_sigma_px": 0.25,
                "quantization_variance_px2": 1.0 / 12.0, "status": "declared_placeholder"}

MODES = {mode.name: mode for mode in (
    ObservationMode(
        "intrinsic_geodesic_distance", "arc length of a declared geodesic measured along the surface", "m", 1,
        "surface_chart", "acquisition", "intrinsic", False,
        {"model": "additive_gaussian", "sigma_m": 1e-4, "status": "declared_placeholder"},
        ("chord length or any other extrinsic embedding quantity",
         "which path joined the endpoints unless the geodesic is declared",
         "the pose of the part in any sensor frame")),
    ObservationMode(
        "camera_chord_distance", "straight-line distance between two triangulated markers", "m", 1,
        "camera_rig", "acquisition", "extrinsic", False, dict(_PIXEL_NOISE),
        ("intrinsic geodesic distance unless a surface model is declared",
         "the surface between the markers", "occluded or unmatched markers")),
    ObservationMode(
        "reconstructed_surface_distance", "geodesic length on a declared or reconstructed surface model", "m", 1,
        "reconstruction", "acquisition", "intrinsic", True,
        {"model": "input_noise_propagated_through_surface_model", "point_sigma_m": 2e-4,
         "model_form_error_m": "undeclared", "status": "declared_placeholder"},
        ("the true surface where it departs from the model", "curvature below the model resolution",
         "the physical path actually followed")),
    ObservationMode(
        "encoder_displacement", "actuator axis displacement", "m", 1, "axis", "acquisition", "none", False,
        {"model": "scale_bias_backlash", "scale": 0.0, "bias_m": 0.0, "backlash_m": 2e-5, "resolution_m": 1e-6,
         "status": "declared_placeholder"},
        ("tool or load position without a kinematic model", "surface geometry",
         "backlash engagement without the direction history")),
    ObservationMode(
        "tracker_measurement", "marker position", "m", 3, "tracker", "arrival", "extrinsic", False,
        {"model": "isotropic_gaussian_with_latency", "sigma_m": 5e-4, "latency_s": 0.008,
         "status": "declared_placeholder"},
        ("distance along the surface", "acquisition time unless a latency is declared",
         "orientation of the marked body")),
    ObservationMode(
        "image_residual", "reprojection residual", "px", 2, "image", "acquisition", "none", False,
        dict(_PIXEL_NOISE),
        ("depth along the viewing ray", "metric scale without calibration", "surface geometry")),
    ObservationMode(
        "imu_orientation", "body orientation as a rotation vector", "rad", 3, "body", "acquisition", "none", False,
        {"model": "gyro_bias_plus_angle_random_walk", "bias_rad_per_s": 1e-3, "arw_rad_per_sqrt_s": 1e-3,
         "status": "declared_placeholder"},
        ("absolute heading without an external reference", "position", "surface geometry")),
)}


@dataclass(frozen=True)
class Observation:
    """One observation record; ``mappings`` lists the declared frame/clock mappings applied."""

    mode: str
    value: tuple
    unit: str
    frame_id: str | None
    clock_id: str | None
    clock_basis: str | None
    epoch: str | None
    time_s: float | None
    calibration_ref: str | None
    sequence: int = 0
    raw_ref: str | None = None
    latency_s: float | None = None
    surface_model: dict | None = None
    derived_from: tuple = ()
    mappings: tuple = ()

    def record(self) -> dict:
        return {"mode": self.mode, "value": list(self.value), "unit": self.unit, "frame_id": self.frame_id,
                "clock_id": self.clock_id, "clock_basis": self.clock_basis, "epoch": self.epoch,
                "time_s": self.time_s, "calibration_ref": self.calibration_ref, "sequence": self.sequence,
                "raw_ref": self.raw_ref, "latency_s": self.latency_s, "surface_model": deepcopy(self.surface_model),
                "derived_from": list(self.derived_from), "mappings": list(self.mappings)}

    def digest(self) -> str:
        return content_identity(self.record())


def observation_from_record(record: dict) -> Observation:
    fields = dict(record)
    fields["value"] = tuple(float(v) for v in fields["value"])
    fields["derived_from"] = tuple(fields.get("derived_from", ()))
    fields["mappings"] = tuple(fields.get("mappings", ()))
    return Observation(**fields)


def observe(mode: str, value, **fields) -> Observation:
    """Build a record; the value is stored as a tuple of floats."""
    values = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    return Observation(mode=mode, value=tuple(float(v) for v in values), **fields)


def validate(observation: Observation) -> Observation:
    """Refuse a record that lacks its declared frame, clock, epoch, time or calibration basis."""
    mode = MODES.get(observation.mode)
    if mode is None:
        raise ObservationRefusal("unknown_mode", f"Unknown observation mode: {observation.mode!r}")
    if not observation.frame_id:
        raise ObservationRefusal("missing_frame", "Observation has no frame id")
    if not observation.clock_id:
        raise ObservationRefusal("missing_clock", "Observation has no clock id")
    if not observation.epoch:
        raise ObservationRefusal("missing_epoch", "Observation has no clock epoch")
    if not observation.calibration_ref:
        raise ObservationRefusal("missing_calibration",
                                 "Observation has no calibration reference; declare one or not_applied")
    if observation.clock_basis not in CLOCK_BASES:
        raise ObservationRefusal("missing_clock_basis", "Observation clock basis must be acquisition or arrival")
    if not observation.mappings:
        # Unmapped records must be in the frame kind and clock basis the mode is acquired in.
        if observation.frame_id.partition(":")[0] != mode.frame_kind:
            raise ObservationRefusal("frame_kind_mismatch",
                                     f"{mode.name} is acquired in a {mode.frame_kind} frame, not {observation.frame_id}")
        if observation.clock_basis != mode.clock_basis:
            raise ObservationRefusal("clock_basis_mismatch",
                                     f"{mode.name} is stamped on {mode.clock_basis}, not {observation.clock_basis}")
    if observation.unit != mode.unit:
        raise ObservationRefusal("unit_mismatch", f"{mode.name} is recorded in {mode.unit}, not {observation.unit}")
    if len(observation.value) != mode.components or not all(math.isfinite(v) for v in observation.value):
        raise ObservationRefusal("invalid_value", f"{mode.name} needs {mode.components} finite component(s)")
    if observation.time_s is None or not math.isfinite(observation.time_s):
        raise ObservationRefusal("missing_time", "Observation has no finite time")
    if mode.requires_surface_model and not observation.surface_model:
        raise ObservationRefusal("surface_model_required", f"{mode.name} requires a declared surface model")
    return observation


def require_mode(observation: Observation, expected: str) -> Observation:
    """Refuse using one mode where another is required; no implicit substitution exists."""
    if observation.mode != expected:
        raise ObservationRefusal("mode_substitution",
                                 f"A {observation.mode} observation cannot stand in for {expected}")
    return validate(observation)


# Chord to surface distance ------------------------------------------------

def helix_chord(s, radius, alpha):
    """Exact chord of a cylinder geodesic at angle ``alpha`` from the circumferential direction."""
    s = np.asarray(s, dtype=float)
    theta = s * math.cos(alpha) / radius
    return np.sqrt((2 * radius * np.sin(theta / 2)) ** 2 + (s * math.sin(alpha)) ** 2)


def arc_length_from_chord(chord: float, model: dict) -> float:
    """Invert chord -> geodesic arc length on a declared surface model."""
    kind = model.get("kind")
    if kind == "plane":
        return float(chord)
    if kind == "sphere":
        radius = float(model["radius"])
        if not 0 <= chord <= 2 * radius:
            raise ObservationRefusal("chord_outside_model", "Chord exceeds the declared sphere diameter")
        return 2 * radius * math.asin(chord / (2 * radius))
    if kind == "cylinder_geodesic":
        radius, alpha = float(model["radius"]), float(model["path_angle_rad"])
        if abs(math.cos(alpha)) < 1e-15:
            return float(chord)
        # The chord increases monotonically with s while s cos(alpha) / R <= pi.
        low, high = float(chord), math.pi * radius / abs(math.cos(alpha))
        if float(helix_chord(high, radius, alpha)) < chord:
            raise ObservationRefusal("chord_outside_model", "Chord exceeds the monotone range of the declared helix")
        for _ in range(200):
            mid = 0.5 * (low + high)
            if float(helix_chord(mid, radius, alpha)) < chord:
                low = mid
            else:
                high = mid
            if high - low <= 4e-16 * high:
                break
        return 0.5 * (low + high)
    raise ObservationRefusal("unsupported_surface_model", f"Unsupported surface model: {kind!r}")


def chord_to_surface_distance(observation: Observation, surface_model: dict | None = None) -> Observation:
    """A camera chord becomes a model-derived surface distance, never an intrinsic observation."""
    require_mode(observation, "camera_chord_distance")
    if not surface_model:
        raise ObservationRefusal("surface_model_required",
                                 "A camera chord is extrinsic; a surface distance requires a declared surface model")
    arc = arc_length_from_chord(observation.value[0], surface_model)
    return replace(observation, mode="reconstructed_surface_distance", value=(arc,),
                   frame_id="reconstruction:" + str(surface_model.get("name", surface_model["kind"])),
                   surface_model=deepcopy(surface_model), derived_from=(observation.digest(),),
                   mappings=observation.mappings + ("surface_model:" + content_identity(surface_model),))


# Frame and clock mappings ---------------------------------------------------

@dataclass(frozen=True)
class FrameMapping:
    """p_target = rotation @ p_source + translation, between frames of one kind."""

    source: str
    target: str
    rotation: tuple
    translation: tuple
    calibration_ref: str

    def record(self) -> dict:
        return {"kind": "frame", "source": self.source, "target": self.target,
                "rotation": [list(r) for r in self.rotation], "translation": list(self.translation),
                "calibration_ref": self.calibration_ref}

    def identity(self) -> str:
        return "frame:" + content_identity(self.record())

    def matrix(self):
        return np.array(self.rotation, dtype=float), np.array(self.translation, dtype=float)

    def inverse(self) -> "FrameMapping":
        rotation, translation = self.matrix()
        inverse_translation = -rotation.T @ translation
        return FrameMapping(self.target, self.source, tuple(tuple(float(v) for v in row) for row in rotation.T),
                            tuple(float(v) for v in inverse_translation), self.calibration_ref)


@dataclass(frozen=True)
class ClockMapping:
    """t_target = rate * t_source + offset_s, from one (clock, epoch, basis) to another."""

    source_clock: str
    source_epoch: str
    source_basis: str
    target_clock: str
    target_epoch: str
    target_basis: str
    rate: float
    offset_s: float
    reference: str

    def record(self) -> dict:
        return {"kind": "clock", "source": [self.source_clock, self.source_epoch, self.source_basis],
                "target": [self.target_clock, self.target_epoch, self.target_basis], "rate": self.rate,
                "offset_s": self.offset_s, "reference": self.reference}

    def identity(self) -> str:
        return "clock:" + content_identity(self.record())


def apply_frame(observation: Observation, mapping: FrameMapping) -> Observation:
    validate(observation)
    if observation.frame_id != mapping.source:
        raise ObservationRefusal("mapping_not_applicable",
                                 f"Frame mapping from {mapping.source} does not apply to {observation.frame_id}")
    if mapping.source.partition(":")[0] != mapping.target.partition(":")[0]:
        raise ObservationRefusal("frame_kind_mismatch", "A frame mapping must join frames of one kind")
    rotation, translation = mapping.matrix()
    if observation.mode == "tracker_measurement":
        value = tuple(float(v) for v in rotation @ np.array(observation.value) + translation)
    elif observation.mode in ("camera_chord_distance", "intrinsic_geodesic_distance", "reconstructed_surface_distance"):
        value = observation.value  # distances are invariant under a rigid frame change
    else:
        raise ObservationRefusal("mapping_not_applicable", f"No rigid frame mapping is defined for {observation.mode}")
    return replace(observation, value=value, frame_id=mapping.target,
                   mappings=observation.mappings + (mapping.identity(),))


def apply_clock(observation: Observation, mapping: ClockMapping) -> Observation:
    validate(observation)
    if observation.clock_id != mapping.source_clock:
        raise ObservationRefusal("mapping_not_applicable",
                                 f"Clock mapping from {mapping.source_clock} does not apply to {observation.clock_id}")
    if observation.epoch != mapping.source_epoch:
        raise ObservationRefusal("epoch_mismatch", "Clock mapping was declared for another epoch")
    if observation.clock_basis != mapping.source_basis:
        raise ObservationRefusal("clock_basis_mismatch", "Clock mapping was declared for another time basis")
    return replace(observation, time_s=mapping.rate * observation.time_s + mapping.offset_s,
                   clock_id=mapping.target_clock, epoch=mapping.target_epoch, clock_basis=mapping.target_basis,
                   mappings=observation.mappings + (mapping.identity(),))


def combine(first: Observation, second: Observation) -> dict:
    """Difference of two observations of one mode in one frame, clock, epoch and basis."""
    validate(first)
    validate(second)
    if first.mode != second.mode:
        raise ObservationRefusal("mode_substitution", "Observations of different modes cannot be combined")
    if first.frame_id != second.frame_id:
        raise ObservationRefusal("frame_mismatch",
                                 f"Frames {first.frame_id} and {second.frame_id} differ and no mapping was applied")
    if first.clock_id != second.clock_id:
        raise ObservationRefusal("clock_mismatch",
                                 f"Clocks {first.clock_id} and {second.clock_id} differ and no mapping was applied")
    if first.epoch != second.epoch:
        raise ObservationRefusal("epoch_mismatch", "Observations share a clock id but not its epoch")
    if first.clock_basis != second.clock_basis:
        raise ObservationRefusal("clock_basis_mismatch", "Acquisition and arrival times cannot be combined")
    return {"mode": first.mode, "frame_id": first.frame_id, "clock_id": first.clock_id, "epoch": first.epoch,
            "clock_basis": first.clock_basis,
            "difference": [a - b for a, b in zip(first.value, second.value)],
            "dt_s": first.time_s - second.time_s, "inputs": [first.digest(), second.digest()]}


def acquisition_age(observation: Observation, now_s: float) -> float:
    """Age since acquisition; arrival stamps need a declared latency."""
    validate(observation)
    time_s = observation.time_s
    if observation.clock_basis == "arrival":
        if observation.latency_s is None:
            raise ObservationRefusal("missing_latency",
                                     "An arrival time cannot establish acquisition age without a declared latency")
        time_s -= observation.latency_s
    age = now_s - time_s
    if age < 0:
        raise ObservationRefusal("future_observation", "Observation was acquired after the use time")
    return age


def admit_fresh(observation: Observation, now_s: float, limit_s: float) -> float:
    age = acquisition_age(observation, now_s)
    if age > limit_s:
        raise ObservationRefusal("stale_observation", f"Observation age {age:.6g} s exceeds the {limit_s:.6g} s limit")
    return age


# Retention without admission ------------------------------------------------

class StateStore:
    """Scalar estimator state that only admitted, digest-bound observations may update."""

    def __init__(self, mode: str, mean: float, variance: float):
        if mode not in MODES or MODES[mode].components != 1:
            raise ObservationRefusal("unknown_mode", "State store requires a scalar observation mode")
        self.mode = mode
        self._state = {"mode": mode, "mean": float(mean), "variance": float(variance), "updates": []}
        self._retained: dict = {}
        self._admissions: dict = {}

    @property
    def state(self) -> dict:
        return deepcopy(self._state)

    def digest(self) -> str:
        return content_identity(self._state)

    def retain(self, observation: Observation) -> dict:
        """Keep the evidence; the state is untouched and admission stays not_performed."""
        record = {"observation": observation.record(), "observation_digest": observation.digest(),
                  "retention": RETENTION, "state_admission": NOT_ADMITTED}
        self._retained[record["observation_digest"]] = deepcopy(record)
        return record

    def retained(self) -> list:
        return [deepcopy(self._retained[key]) for key in sorted(self._retained)]

    def admit(self, digest: str, decision: str) -> dict:
        if digest not in self._retained:
            raise ObservationRefusal("not_retained", "Only a retained observation can be admitted")
        validate(observation_from_record(self._retained[digest]["observation"]))
        admission = {"observation_digest": digest, "state_admission": ADMITTED, "decision": decision}
        self._admissions[digest] = admission
        return deepcopy(admission)

    def update(self, record: dict, variance: float) -> dict:
        """Scalar Kalman update from an admitted record; everything else is refused."""
        observation = observation_from_record(record["observation"])
        digest = observation.digest()
        if digest != record.get("observation_digest"):
            raise ObservationRefusal("admission_digest_mismatch", "Record content does not match its digest")
        if digest not in self._admissions:
            raise ObservationRefusal("not_admitted", "A retained observation was not admitted as state")
        require_mode(observation, self.mode)
        prior_mean, prior_variance = self._state["mean"], self._state["variance"]
        gain = prior_variance / (prior_variance + variance)
        self._state["mean"] = prior_mean + gain * (observation.value[0] - prior_mean)
        self._state["variance"] = (1 - gain) * prior_variance
        self._state["updates"].append(digest)
        return self.state


# Dropped observations ---------------------------------------------------------

def bernoulli_drops(n: int, probability: float, rng) -> np.ndarray:
    return rng.random(n) < probability


def gilbert_elliott_drops(n: int, good_to_bad: float, bad_to_good: float, rng, streams: int = 1) -> np.ndarray:
    """Two-state burst channel dropping every sample in the bad state; stationary start."""
    stationary_bad = good_to_bad / (good_to_bad + bad_to_good)
    uniforms = rng.random((streams, n))
    state = rng.random(streams) < stationary_bad
    out = np.empty((streams, n), dtype=bool)
    for k in range(n):
        out[:, k] = state
        state = np.where(state, uniforms[:, k] >= bad_to_good, uniforms[:, k] < good_to_bad)
    return out


def with_drops(observations: list, dropped) -> list:
    """Missing samples stay explicit ``None`` placeholders at their sequence position."""
    return [None if drop else item for item, drop in zip(observations, dropped)]


def zero_fill(stream: list, template: Observation) -> list:
    """The refused anti-pattern: a fabricated zero with no raw reference replaces each gap."""
    return [replace(template, value=(0.0,) * len(template.value), sequence=index, raw_ref=None)
            if item is None else item for index, item in enumerate(stream)]


def admit_stream(stream: list) -> dict:
    """Accept explicit gaps; refuse any value that has no raw acquisition reference."""
    missing, present = [], 0
    for index, item in enumerate(stream):
        if item is None:
            missing.append(index)
            continue
        validate(item)
        if item.sequence != index:
            raise ObservationRefusal("sequence_gap", f"Sample {index} carries sequence {item.sequence}")
        if not item.raw_ref:
            if all(v == 0.0 for v in item.value):
                raise ObservationRefusal("zero_filled_missing",
                                         f"Sample {index} is a zero with no raw reference: a filled gap")
            raise ObservationRefusal("unbacked_value", f"Sample {index} has no raw acquisition reference")
        present += 1
    return {"present": present, "missing": missing}


def detect_zero_fill(values, sigma: float, threshold: float = 8.0) -> list:
    """Flag exact zeros far from their neighbours when provenance was stripped."""
    values = list(values)
    flagged = []
    for index, value in enumerate(values):
        if value != 0.0:
            continue
        neighbours = [v for v in values[max(0, index - 3):index] + values[index + 1:index + 4] if v not in (None, 0.0)]
        if neighbours and abs(float(np.median(neighbours))) > threshold * sigma:
            flagged.append(index)
    return flagged
