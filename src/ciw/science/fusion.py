"""Asynchronous multi-stream position fusion with staged, refusable provenance.

Scope
-----
Five stages are kept distinct and never collapsed into one "measurement"::

    raw observation -> transformed observation -> filtered candidate
                    -> candidate physical state -> admitted operational state

Each stage is a finite-JSON record carrying ``schema``, ``stage``, ``parents``
(content identities of the records it was derived from) and its own content
``identity``; ``retain`` appends any of them to the evidence ledger as a
``ciw.science.fusion-state.v1`` entry that references its retained parents.

* A raw observation (``ciw.raw-observation.v1``) names its stream, sequence
  number, clock, device time, unit, frame, full (possibly correlated) noise
  covariance, observable and acquisition kind. A missing clock is refused.
* The transformed observation converts units (covariance by ``S C S``), maps a
  3-D point into the filter frame through ``FrameRegistry.transform_point`` at
  the acquisition time (the transform covariance is added), aligns the time
  stamp to the filter clock with ``FrameRegistry.align_time`` and inflates the
  noise for timing uncertainty: ``R_eff = R + (H v)(H v)^T sigma_t^2`` with the
  current velocity estimate ``v``. Before the filter has a velocity estimate the
  declared initial velocity spread is used: ``R_eff = R + sigma_v0^2 sigma_t^2 I``.
* The filtered candidate is a linear Kalman filter over a d-dimensional
  constant-velocity model (state ``[p, v]``) driven by continuous white
  acceleration of spectral density ``q``; ``F`` and ``Q`` are the exact
  discretisation. Observations are processed in aligned acquisition order
  (``ingest_batch`` sorts a batch; ``replay`` adds a hold-back reorder buffer),
  the covariance update is Joseph form, and innovations are gated by NIS
  against exact chi-square quantiles (``CHI2_QUANTILES``). Every refusal
  (malformed, duplicate, stale, out-of-sequence, gated outlier, frame or clock
  confusion) is retained as a record with its reason and, for outliers, NIS.
* A candidate physical state is the filter state predicted to a requested time,
  with covariance, unit, frame, clock, contributing observation identities,
  transform/clock validity record and a residual-monitor summary.
* ``admit`` applies an ``AdmissionPolicy`` and returns either an admitted state
  or an admission refusal that keeps the candidate a candidate and lists every
  failed check.

Honest limits
-------------
* Linear constant-velocity model only; position observables only (``H = [I 0]``).
  No orientation, IMU, nonlinear or multiple-model filtering and no smoothing.
* No retrodiction: an observation older than the filter time is refused as
  ``out_of_sequence``; late data can only be accommodated by a reorder buffer.
* Noise is treated as white and independent between observations. Calibration
  errors of a transform or clock mapping are really biases shared by every
  observation that used them; adding them per observation understates their
  correlation over time.
* Timing inflation is first order: it uses the velocity *estimate*, neglects
  ``P_vv sigma_t^2`` and any correlation between timing and state error. Without
  a frame registry the clock resolution is undeclared and contributes nothing.
* Windowed NIS bounds use the Wilson-Hilferty approximation
  ``chi2_p(k) ~= k (1 - 2/(9k) + z_p sqrt(2/(9k)))^3``; gating uses exact
  tabulated quantiles for 1..6 degrees of freedom at 0.99 and 0.999 only.
* NEES/NIS consistency on synthetic data shows the code matches its own model;
  it does not show a physical sensor obeys that model. Synthetic scenarios are
  labelled ``acquisition: "synthetic"``, and the default admission policy admits
  only physically acquired contributions.
* Admission is a recorded policy decision over diagnostics. It grants no
  actuation or execution authority.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from ._common import (Refusal, bounded_json, content_identity, covariance, finite, integer, mapping, plain,
                      require_keys, text, vector)
from .frames import FrameRegistry
from .units import conversion_factor, convert, convert_covariance, parse_unit, require_dimension
from .vocabulary import ACQUISITION_KINDS, OBSERVABLES

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import Ledger

RAW_SCHEMA = "ciw.raw-observation.v1"
TRANSFORMED_SCHEMA = "ciw.transformed-observation.v1"
FILTERED_SCHEMA = "ciw.filtered-candidate.v1"
CANDIDATE_SCHEMA = "ciw.candidate-state.v1"
ADMITTED_SCHEMA = "ciw.admitted-state.v1"
REFUSED_SCHEMA = "ciw.fusion-refusal.v1"
ADMISSION_REFUSED_SCHEMA = "ciw.admission-refusal.v1"
POLICY_SCHEMA = "ciw.admission-policy.v1"
SCENARIO_SCHEMA = "ciw.fusion-scenario.v1"
LEDGER_SCHEMA = "ciw.science.fusion-state.v1"

STAGES = ("raw_observation", "transformed_observation", "filtered_candidate", "candidate_state", "admitted_state")
REFUSAL_STAGES = ("refused", "admission_refused")
POSITION_OBSERVABLES = frozenset({"tracker_position"})

# Exact chi-square quantiles (inverse regularized lower incomplete gamma), 10 significant
# digits, indexed by degrees of freedom 1..6. Used for NIS gating.
CHI2_QUANTILES: dict[float, tuple[float, ...]] = {
    0.99: (6.634896601, 9.210340372, 11.34486673, 13.27670414, 15.08627247, 16.81189383),
    0.999: (10.82756617, 13.81551056, 16.26623620, 18.46682695, 20.51500565, 22.45774448),
}
MAX_MEASUREMENT_DIM = 6
MAX_INPUTS = 50_000
MAX_HISTORY = 4 * MAX_INPUTS
MAX_STREAMS = 256
MAX_WINDOW = 1024
MAX_LISTED_CONTRIBUTORS = 512
MAX_SEQUENCE = 2 ** 63 - 1


# ---------------------------------------------------------------------- statistics
def normal_quantile(probability: float) -> float:
    """Standard normal quantile by bisection on ``erfc`` (no third-party dependency)."""
    p = finite(probability, "probability", minimum=1e-12, maximum=1.0 - 1e-12)
    low, high = -10.0, 10.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if 0.5 * math.erfc(-middle / math.sqrt(2.0)) < p else (low, middle)
    return 0.5 * (low + high)


def chi2_quantile_wh(probability: float, dof: float) -> float:
    """Wilson-Hilferty approximation ``k (1 - 2/(9k) + z_p sqrt(2/(9k)))^3`` of a chi-square quantile."""
    k = finite(dof, "dof", minimum=0.0, exclusive_minimum=True)
    c = 2.0 / (9.0 * k)
    return k * max(0.0, 1.0 - c + normal_quantile(probability) * math.sqrt(c)) ** 3


def chi2_interval(dof: float, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided chi-square interval (Wilson-Hilferty) for ``dof`` degrees of freedom."""
    confidence = finite(confidence, "confidence", minimum=0.5, maximum=0.9999)
    return chi2_quantile_wh(0.5 * (1.0 - confidence), dof), chi2_quantile_wh(0.5 * (1.0 + confidence), dof)


# ---------------------------------------------------------------------- records
def record_identity(record: dict) -> str:
    try:
        return content_identity({key: value for key, value in mapping(record, "fusion record").items()
                                 if key != "identity"})
    except (TypeError, ValueError) as exc:
        raise Refusal("malformed_record", f"fusion record is not finite JSON: {exc}") from exc


def _seal(record: dict) -> dict:
    record = plain(record)
    record["identity"] = record_identity(record)
    return record


def verify_record(record: Any) -> dict:
    """Check a stage record's declared stage and content identity (never repaired)."""
    record = mapping(record, "fusion record")
    if record.get("stage") not in STAGES + REFUSAL_STAGES:
        raise Refusal("unknown_stage", f"Stage {record.get('stage')!r} is not a fusion stage")
    if record.get("identity") != record_identity(record):
        raise Refusal("identity_mismatch", "Fusion record content does not match its identity")
    return record


def _covariance(value: Any, name: str, size: int) -> np.ndarray:
    try:
        return covariance(value, name, size)
    except Refusal as exc:
        raise Refusal("malformed_covariance", f"{name}: {exc}", cause=exc.code, **exc.detail) from exc


# ---------------------------------------------------------------------- stage 1: raw observation
@dataclass(frozen=True)
class RawObservation:
    """A device report exactly as acquired; validated on construction, never repaired."""

    stream_id: str
    sequence: int
    clock: str | None
    device_time_s: float
    value: tuple[float, ...]
    unit: str
    frame: str
    covariance: tuple[tuple[float, ...], ...]
    observable: str
    acquisition: str

    def __post_init__(self) -> None:
        text(self.stream_id, "stream_id", 128)
        integer(self.sequence, "sequence", minimum=0, maximum=MAX_SEQUENCE)
        if self.clock is None:
            raise Refusal("clock_unspecified", "A raw observation must name the clock of its device_time_s")
        text(self.clock, "clock", 128)
        values = vector(self.value, "value")
        if not 1 <= len(values) <= MAX_MEASUREMENT_DIM:
            raise Refusal("malformed_record", f"value must have 1..{MAX_MEASUREMENT_DIM} components")
        cov = _covariance(self.covariance, "covariance", len(values))
        parse_unit(self.unit)
        text(self.frame, "frame", 128)
        if self.observable not in OBSERVABLES:
            raise Refusal("unknown_observable", f"Observable {self.observable!r} is not declared",
                          allowed=sorted(OBSERVABLES))
        if self.acquisition not in ACQUISITION_KINDS:
            raise Refusal("unknown_acquisition", f"Acquisition {self.acquisition!r} is not declared",
                          allowed=sorted(ACQUISITION_KINDS))
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "device_time_s", finite(self.device_time_s, "device_time_s"))
        object.__setattr__(self, "value", tuple(float(item) for item in values))
        object.__setattr__(self, "covariance", tuple(tuple(float(item) for item in row) for row in cov))

    @classmethod
    def from_json(cls, record: Any) -> "RawObservation":
        record = mapping(record, "raw observation")
        if record.get("clock") is None:
            raise Refusal("clock_unspecified", "A raw observation must name the clock of its device_time_s")
        require_keys(record, "raw observation", {"schema", "stream_id", "sequence", "clock", "device_time_s", "value",
                                                 "unit", "frame", "covariance", "observable", "acquisition"},
                     {"stage", "parents", "identity"})
        if record["schema"] != RAW_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {RAW_SCHEMA}")
        if record.get("stage", "raw_observation") != "raw_observation" or record.get("parents", []) != []:
            raise Refusal("malformed_record", "A raw observation has stage raw_observation and no parents")
        result = cls(*(record[key] for key in ("stream_id", "sequence", "clock", "device_time_s", "value", "unit",
                                              "frame", "covariance", "observable", "acquisition")))
        if "identity" in record and record["identity"] != result.identity:
            raise Refusal("identity_mismatch", "Raw observation content does not match its identity")
        return result

    def to_json(self) -> dict:
        return {"schema": RAW_SCHEMA, "stream_id": self.stream_id, "sequence": self.sequence, "clock": self.clock,
                "device_time_s": self.device_time_s, "value": list(self.value), "unit": self.unit,
                "frame": self.frame, "covariance": [list(row) for row in self.covariance],
                "observable": self.observable, "acquisition": self.acquisition}

    def record(self) -> dict:
        return _seal({"stage": "raw_observation", "parents": [], **self.to_json()})

    @property
    def identity(self) -> str:
        return self.record()["identity"]


# ---------------------------------------------------------------------- model and policy
@dataclass(frozen=True)
class ConstantVelocityModel:
    """``x = [p, v]`` in ``dimension`` axes with white acceleration of spectral density ``q``."""

    dimension: int = 3
    acceleration_psd: float = 1.0
    psd_unit: str = "m^2/s^3"
    observables: tuple[str, ...] = ("tracker_position",)

    def __post_init__(self) -> None:
        integer(self.dimension, "dimension", minimum=1, maximum=MAX_MEASUREMENT_DIM)
        finite(self.acceleration_psd, "acceleration_psd", minimum=0.0, exclusive_minimum=True)
        require_dimension(self.psd_unit, "m^2/s^3", "acceleration_psd unit")
        observables = tuple(sorted(set(self.observables))) if isinstance(self.observables, (list, tuple)) else ()
        if not observables:
            raise Refusal("malformed_record", "A measurement model must declare at least one observable")
        for item in observables:
            if item not in OBSERVABLES:
                raise Refusal("unknown_observable", f"Observable {item!r} is not declared")
            if item not in POSITION_OBSERVABLES:
                raise Refusal("unsupported_observable", f"This build only fuses position observables, not {item!r}",
                              supported=sorted(POSITION_OBSERVABLES))
        object.__setattr__(self, "observables", observables)

    def psd_in(self, unit: str) -> float:
        return self.acceleration_psd * conversion_factor(self.psd_unit, f"({unit})^2/s^3")

    @property
    def measurement_matrix(self) -> np.ndarray:
        return np.hstack([np.eye(self.dimension), np.zeros((self.dimension, self.dimension))])

    def discrete(self, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``F(dt)`` and ``Q(dt) = q [[dt^3/3, dt^2/2], [dt^2/2, dt]] (x) I``."""
        eye = np.eye(self.dimension)
        transition = np.kron(np.array([[1.0, dt], [0.0, 1.0]]), eye)
        noise = q * np.kron(np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]]), eye)
        return transition, noise

    def describe(self) -> dict:
        return {"kind": "constant_velocity", "dimension": self.dimension, "acceleration_psd": self.acceleration_psd,
                "psd_unit": self.psd_unit, "observables": list(self.observables), "discretisation": "exact"}


@dataclass(frozen=True)
class AdmissionPolicy:
    """Thresholds a candidate must satisfy to become operational state; every one is checked."""

    min_accepted_updates: int
    nis_window: int
    max_position_std: float
    std_unit: str
    max_refused_fraction: float
    refusal_window: int
    max_state_age_s: float
    nis_confidence: float = 0.95
    require_valid_provenance: bool = True
    allowed_acquisition: tuple[str, ...] = ("physical",)

    def __post_init__(self) -> None:
        integer(self.min_accepted_updates, "min_accepted_updates", minimum=0, maximum=MAX_INPUTS)
        integer(self.nis_window, "nis_window", minimum=1, maximum=MAX_WINDOW)
        integer(self.refusal_window, "refusal_window", minimum=1, maximum=MAX_WINDOW)
        finite(self.max_position_std, "max_position_std", minimum=0.0, exclusive_minimum=True)
        require_dimension(self.std_unit, "m", "std_unit")
        finite(self.max_refused_fraction, "max_refused_fraction", minimum=0.0, maximum=1.0)
        finite(self.max_state_age_s, "max_state_age_s", minimum=0.0)
        finite(self.nis_confidence, "nis_confidence", minimum=0.5, maximum=0.9999)
        if not isinstance(self.require_valid_provenance, bool):
            raise Refusal("malformed_record", "require_valid_provenance must be a boolean")
        kinds = self.allowed_acquisition
        kinds = tuple(sorted(set(kinds))) if isinstance(kinds, (list, tuple)) else ()
        if not kinds or not set(kinds) <= ACQUISITION_KINDS:
            raise Refusal("unknown_acquisition", "allowed_acquisition must be a nonempty subset of the vocabulary",
                          allowed=sorted(ACQUISITION_KINDS))
        object.__setattr__(self, "allowed_acquisition", kinds)

    @classmethod
    def from_json(cls, record: Any) -> "AdmissionPolicy":
        fields = {"min_accepted_updates", "nis_window", "max_position_std", "std_unit", "max_refused_fraction",
                  "refusal_window", "max_state_age_s"}
        record = require_keys(record, "admission policy", fields | {"schema"},
                              {"nis_confidence", "require_valid_provenance", "allowed_acquisition"})
        if record["schema"] != POLICY_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {POLICY_SCHEMA}")
        return cls(**{key: value for key, value in record.items() if key != "schema"})

    def to_json(self) -> dict:
        return {"schema": POLICY_SCHEMA, "min_accepted_updates": self.min_accepted_updates,
                "nis_window": self.nis_window, "nis_confidence": self.nis_confidence,
                "max_position_std": self.max_position_std, "std_unit": self.std_unit,
                "max_refused_fraction": self.max_refused_fraction, "refusal_window": self.refusal_window,
                "max_state_age_s": self.max_state_age_s, "require_valid_provenance": self.require_valid_provenance,
                "allowed_acquisition": list(self.allowed_acquisition)}


def _speed(value: Any, unit: str) -> float:
    if isinstance(value, dict):
        value = require_keys(value, "initial_velocity_std", {"value", "unit"})
        require_dimension(value["unit"], "m/s", "initial_velocity_std unit")
        value = finite(value["value"], "initial_velocity_std.value") * conversion_factor(value["unit"], f"({unit})/s")
    return finite(value, "initial_velocity_std", minimum=0.0, exclusive_minimum=True)


def _policy(value: Any) -> AdmissionPolicy:
    return value if isinstance(value, AdmissionPolicy) else AdmissionPolicy.from_json(value)


# ---------------------------------------------------------------------- engine
class FusionEngine:
    """Staged fusion of asynchronous position streams into one constant-velocity state.

    ``prior`` (``{"time", "mean", "covariance"}``) is in the filter unit, frame and clock.
    Without a prior the first accepted observation initializes position, and velocity
    gets ``initial_velocity_std``: a bare number is read in ``unit``/s, a
    ``{"value", "unit"}`` object is converted. ``history`` lists every record in creation
    order; ``records`` indexes them by identity.
    """

    def __init__(self, model: ConstantVelocityModel, frames: FrameRegistry | None, *, filter_frame: str,
                 filter_clock: str, unit: str, prior: dict | None = None,
                 initial_velocity_std: float | dict | None = None,
                 gate_probability: float = 0.999, max_age_s: float = 1.0, monitor_window: int = 64,
                 refusal_window: int = 128, admission_policy: AdmissionPolicy | dict | None = None):
        if not isinstance(model, ConstantVelocityModel):
            raise Refusal("malformed_record", "model must be a ConstantVelocityModel")
        if frames is not None and not isinstance(frames, FrameRegistry):
            raise Refusal("malformed_record", "frames must be a FrameRegistry or None")
        self.model, self.frames = model, frames
        self.filter_frame = text(filter_frame, "filter_frame", 128)
        self.filter_clock = text(filter_clock, "filter_clock", 128)
        self.unit = require_dimension(unit, "m", "filter unit (position observables)").expression
        if frames is not None:
            if filter_frame not in frames.frames:
                raise Refusal("unknown_frame", f"Filter frame {filter_frame!r} is not declared in the registry")
            if filter_clock not in frames.clocks:
                raise Refusal("unknown_clock", f"Filter clock {filter_clock!r} is not declared in the registry")
        if not isinstance(gate_probability, float) or gate_probability not in CHI2_QUANTILES:
            raise Refusal("unsupported_gate", "gate_probability must be a tabulated level",
                          allowed=sorted(CHI2_QUANTILES))
        self.gate_probability = float(gate_probability)
        self.max_age_s = finite(max_age_s, "max_age_s", minimum=0.0, exclusive_minimum=True)
        self.initial_velocity_std = None if initial_velocity_std is None else _speed(initial_velocity_std, self.unit)
        if prior is None and self.initial_velocity_std is None:
            raise Refusal("prior_unspecified", "Declare a prior state or an initial_velocity_std for first-fix "
                          "initialization; the filter never guesses its initial velocity uncertainty")
        self.q = model.psd_in(self.unit)
        self.admission_policy = None if admission_policy is None else _policy(admission_policy)
        d = model.dimension
        self.history: list[dict] = []
        self.records: dict[str, dict] = {}
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None
        self._state_id: str | None = None
        self._prior_id: str | None = None
        self._seen: dict[str, set[int]] = {}
        self._nis: deque = deque(maxlen=integer(monitor_window, "monitor_window", minimum=1, maximum=MAX_WINDOW))
        self._inputs: deque = deque(maxlen=integer(refusal_window, "refusal_window", minimum=1, maximum=MAX_WINDOW))
        self._received = 0
        self._accepted_updates = 0
        self._initializations = 0
        self._refused: Counter = Counter()
        self._contributors: list[str] = []
        self._acquisition: set[str] = set()
        self._mappings_used: set[str] = set()
        self._transforms_used: set[str] = set()
        self._validity_checked = 0
        if prior is not None:
            prior = require_keys(prior, "prior", {"time", "mean", "covariance"})
            mean = vector(prior["mean"], "prior.mean", 2 * d)
            cov = _covariance(prior["covariance"], "prior.covariance", 2 * d)
            start = finite(prior["time"], "prior.time")
            self._install(mean, cov, start, _seal({
                "schema": FILTERED_SCHEMA, "stage": "filtered_candidate", "parents": [], "claim_class": "estimated",
                "update": {"kind": "declared_prior"}, "time": {"value": start, "clock": self.filter_clock},
                "frame": self.filter_frame, "unit": self.unit, "velocity_unit": f"{self.unit}/s",
                "state": {"mean": mean, "covariance": cov}}))
            self._prior_id = self._state_id

    # ------------------------------------------------------------------ bookkeeping
    @property
    def time(self) -> float | None:
        """Filter time on the filter clock (time of the newest accepted update or prior)."""
        return self._t

    def _remember(self, record: dict) -> dict:
        if len(self.history) >= MAX_HISTORY:
            raise Refusal("oversized_input", "Fusion history reached its bound")
        self.history.append(record)
        self.records[record["identity"]] = record
        return record

    def _install(self, x: np.ndarray, P: np.ndarray, t: float, record: dict) -> None:
        self._x, self._P, self._t = x, P, t
        self._state_id = self._remember(record)["identity"]

    def configuration(self) -> dict:
        return plain({"model": self.model.describe(), "filter_frame": self.filter_frame,
                      "filter_clock": self.filter_clock, "unit": self.unit,
                      "acceleration_psd": {"value": self.q, "unit": f"({self.unit})^2/s^3"},
                      "gate": {"probability": self.gate_probability,
                               "quantiles": list(CHI2_QUANTILES[self.gate_probability]),
                               "quantile_basis": "exact_chi_square"},
                      "max_age_s": self.max_age_s, "initial_velocity_std": self.initial_velocity_std,
                      "registry_identity": None if self.frames is None else self.frames.identity()})

    def dropped(self) -> dict:
        """Per-stream sequence gaps among received sequence numbers (a leading gap is not detectable)."""
        result = {}
        for stream, seen in sorted(self._seen.items()):
            ordered = sorted(seen)
            ranges = [[a + 1, b - 1] for a, b in zip(ordered, ordered[1:]) if b > a + 1]
            result[stream] = {"received": len(ordered), "first": ordered[0], "last": ordered[-1],
                              "dropped": sum(b - a + 1 for a, b in ranges), "ranges": ranges}
        return result

    def lineage(self, identity: str) -> list[dict]:
        """The record and every ancestor held by this engine, nearest first."""
        pending, seen, result = [identity], set(), []
        while pending:
            current = pending.pop(0)
            if current in seen or current not in self.records:
                continue
            seen.add(current)
            result.append(self.records[current])
            pending.extend(self.records[current].get("parents", []))
        return result

    # ------------------------------------------------------------------ stage 2: time alignment and transform
    def aligned_time(self, raw: RawObservation | dict) -> tuple[float, float, list[str]]:
        """Acquisition time on the filter clock with its variance (s^2) and the clock mappings used."""
        obs = raw if isinstance(raw, RawObservation) else RawObservation.from_json(raw)
        if self.frames is None:
            if obs.clock != self.filter_clock:
                raise Refusal("clock_unmapped", f"Clock {obs.clock!r} differs from filter clock "
                              f"{self.filter_clock!r} and no frame registry declares a mapping")
            return obs.device_time_s, 0.0, []
        return self.frames.align_time(obs.device_time_s, obs.clock, self.filter_clock)

    def _timing_velocity(self) -> tuple[np.ndarray, str]:
        d = self.model.dimension
        if self._x is not None:
            return self._x[d:].copy(), "state_estimate"
        return np.zeros(d), "initial_velocity_std"

    def _transform(self, obs: RawObservation, raw_id: str, alignment: tuple[float, float, list[str]]) -> dict:
        t, variance, mappings = alignment
        m = len(obs.value)
        if m != self.model.dimension:
            raise Refusal("dimension_mismatch", f"Observation has {m} components; the model has {self.model.dimension}")
        require_dimension(obs.unit, self.unit, "observation unit")
        factor = conversion_factor(obs.unit, self.unit)
        z = np.atleast_1d(convert(list(obs.value), obs.unit, self.unit))
        R = convert_covariance(obs.covariance, [obs.unit] * m, [self.unit] * m)
        chain: list[dict] = []
        if obs.frame != self.filter_frame:
            if self.frames is None:
                raise Refusal("frame_unreachable", f"Observation frame {obs.frame!r} differs from filter frame "
                              f"{self.filter_frame!r} and no frame registry is declared",
                              source=obs.frame, target=self.filter_frame)
            if m != 3:
                raise Refusal("frame_transform_unsupported", "Frame transforms apply to 3-D points only")
            mapped = self.frames.transform_point(z, self.unit, obs.frame, self.filter_frame, obs.device_time_s,
                                                 obs.clock, point_covariance=R)
            z, R, chain = np.array(mapped["point"]), np.array(mapped["covariance"]), mapped["chain"]
        velocity, source = self._timing_velocity()
        if source == "state_estimate":
            timing = np.outer(velocity, velocity) * variance
        else:
            timing = np.eye(m) * self.initial_velocity_std ** 2 * variance
        velocity_record = {"value": velocity, "unit": f"{self.unit}/s", "source": source}
        if source != "state_estimate":
            velocity_record["std"] = self.initial_velocity_std
        return self._remember(_seal({
            "schema": TRANSFORMED_SCHEMA, "stage": "transformed_observation", "parents": [raw_id],
            "stream_id": obs.stream_id, "sequence": obs.sequence, "observable": obs.observable,
            "acquisition": obs.acquisition,
            "time": {"value": t, "clock": self.filter_clock, "variance_s2": variance, "source_clock": obs.clock,
                     "device_time_s": obs.device_time_s, "clock_mappings": mappings,
                     "basis": "frame_registry" if self.frames is not None else "same_clock_resolution_undeclared"},
            "unit": self.unit, "source_unit": obs.unit, "unit_factor": factor,
            "frame": self.filter_frame, "source_frame": obs.frame, "transform_chain": chain,
            "value": z, "covariance": R, "timing_covariance": timing, "effective_covariance": R + timing,
            "timing_velocity": velocity_record,
            "validity": {"checked_at": {"time": obs.device_time_s, "clock": obs.clock}, "valid": True,
                         "clock_mappings": mappings, "transforms": [link["transform_id"] for link in chain]},
        }))

    # ------------------------------------------------------------------ stage 3: filtering
    def _refuse(self, refused_at: str, exc: Refusal, parents: list[str], obs: RawObservation | None = None,
                **context: Any) -> dict:
        record = {"schema": REFUSED_SCHEMA, "stage": "refused", "refused_at": refused_at, "parents": parents,
                  "refusal": exc.to_dict(), **context}
        if obs is not None:
            record.update(stream_id=obs.stream_id, sequence=obs.sequence, acquisition=obs.acquisition)
        record = self._remember(_seal(record))
        self._refused[exc.code] += 1
        self._inputs.append({"outcome": "refused", "code": exc.code,
                             "stream_id": None if obs is None else obs.stream_id})
        return record

    def ingest(self, raw: RawObservation | dict, *, evaluated_at: float | None = None) -> dict:
        """Process one observation; returns its filtered candidate or its refusal record."""
        return self.ingest_batch([raw], evaluated_at=evaluated_at)[0]

    def ingest_batch(self, raws: Iterable[Any], *, evaluated_at: float | None = None) -> list[dict]:
        """Validate and align every observation, then filter them in aligned acquisition-time order.

        ``evaluated_at`` is the evaluation time on the filter clock used for staleness. When it is
        omitted, staleness is measured against the later of the filter time and the newest aligned
        time in the batch, and no future-timestamp check is possible. Results follow input order.
        """
        raws = list(raws) if isinstance(raws, (list, tuple)) else None
        if raws is None:
            raise Refusal("malformed_record", "ingest_batch takes a list of raw observations")
        if self._received + len(raws) > MAX_INPUTS:
            raise Refusal("oversized_input", f"Fusion engine accepts at most {MAX_INPUTS} observations")
        declared = evaluated_at is not None
        if declared:
            evaluated_at = finite(evaluated_at, "evaluated_at")
        results: list[dict | None] = [None] * len(raws)
        ready = []
        for index, item in enumerate(raws):
            self._received += 1
            outcome = self._receive(item)
            if isinstance(outcome, dict):
                results[index] = outcome
            else:
                ready.append((outcome[2][0], outcome[0].stream_id, outcome[0].sequence, index, outcome))
        if not declared:
            evaluated_at = max([item[0] for item in ready] + ([self._t] if self._t is not None else []), default=0.0)
        for _, _, _, index, (obs, raw_id, alignment) in sorted(ready, key=lambda item: item[:4]):
            results[index] = self._process(obs, raw_id, alignment, evaluated_at, declared)
        return results  # type: ignore[return-value]

    def _receive(self, item: Any) -> dict | tuple[RawObservation, str, tuple[float, float, list[str]]]:
        try:
            obs = item if isinstance(item, RawObservation) else RawObservation.from_json(item)
        except Refusal as exc:
            try:
                bounded_json(item, "input", 65536)
                context = {"input": plain(item)}
            except Refusal:
                context = {"input_excerpt": repr(item)[:1024]}
            return self._refuse("raw_observation", exc, [], **context)
        raw_id = self._remember(obs.record())["identity"]
        seen = self._seen.get(obs.stream_id)
        if seen is None:
            if len(self._seen) >= MAX_STREAMS:
                return self._refuse("transformed_observation", Refusal("oversized_input", "Too many streams"),
                                    [raw_id], obs)
            seen = self._seen[obs.stream_id] = set()
        if obs.sequence in seen:
            return self._refuse("transformed_observation", Refusal(
                "duplicate_observation", f"Stream {obs.stream_id!r} already delivered sequence {obs.sequence}"),
                [raw_id], obs)
        seen.add(obs.sequence)
        if obs.observable not in self.model.observables:
            return self._refuse("transformed_observation", Refusal(
                "observable_mismatch", f"The measurement model does not declare observable {obs.observable!r}",
                declared=list(self.model.observables)), [raw_id], obs)
        try:
            alignment = self.aligned_time(obs)
        except Refusal as exc:
            return self._refuse("transformed_observation", exc, [raw_id], obs)
        return obs, raw_id, alignment

    def _process(self, obs: RawObservation, raw_id: str, alignment: tuple[float, float, list[str]], now: float,
                 declared: bool) -> dict:
        try:
            transformed = self._transform(obs, raw_id, alignment)
        except Refusal as exc:
            return self._refuse("transformed_observation", exc, [raw_id], obs)
        t, variance = alignment[0], alignment[1]
        parents = [transformed["identity"]]
        timing = {"aligned_time": t, "evaluated_at": now, "filter_time": self._t,
                  "evaluated_at_basis": "declared" if declared else "newest_aligned_time"}
        if now - t > self.max_age_s:
            return self._refuse("filtered_candidate", Refusal(
                "stale_observation", f"Observation is {now - t:.6g} s old; the limit is {self.max_age_s} s",
                age_s=now - t, max_age_s=self.max_age_s), parents, obs, timing=timing)
        if declared and t - now > 6.0 * math.sqrt(variance) + 1e-9:
            return self._refuse("filtered_candidate", Refusal(
                "future_observation", "Aligned acquisition time lies after the evaluation time",
                lead_s=t - now), parents, obs, timing=timing)
        if self._t is not None and t < self._t:
            return self._refuse("filtered_candidate", Refusal(
                "out_of_sequence", "Observation precedes the filter time; retrodiction is not implemented",
                lag_s=self._t - t), parents, obs, timing=timing)
        try:
            record = self._filter(obs, transformed)
        except Refusal as exc:
            extra = {"nis": exc.detail["nis"]} if "nis" in exc.detail else {}
            return self._refuse("filtered_candidate", exc, parents + ([self._state_id] if self._state_id else []),
                                obs, timing=timing, **extra)
        self._inputs.append({"outcome": "accepted", "code": None, "stream_id": obs.stream_id})
        self._contributors.append(raw_id)
        self._acquisition.add(obs.acquisition)
        self._mappings_used.update(transformed["validity"]["clock_mappings"])
        self._transforms_used.update(transformed["validity"]["transforms"])
        self._validity_checked += bool(transformed["validity"]["valid"])
        return record

    def _filter(self, obs: RawObservation, transformed: dict) -> dict:
        d = self.model.dimension
        t = transformed["time"]["value"]
        z, R = np.array(transformed["value"]), np.array(transformed["effective_covariance"])
        common = {"schema": FILTERED_SCHEMA, "stage": "filtered_candidate", "claim_class": "estimated",
                  "stream_id": obs.stream_id, "sequence": obs.sequence,
                  "time": {"value": t, "clock": self.filter_clock}, "frame": self.filter_frame, "unit": self.unit,
                  "velocity_unit": f"{self.unit}/s"}
        if self._x is None:
            x = np.concatenate([z, np.zeros(d)])
            P = np.zeros((2 * d, 2 * d))
            P[:d, :d], P[d:, d:] = R, np.eye(d) * self.initial_velocity_std ** 2
            record = _seal({**common, "parents": [transformed["identity"]],
                            "update": {"kind": "initialization", "velocity_std": self.initial_velocity_std},
                            "state": {"mean": x, "covariance": P}})
            self._install(x, P, t, record)
            self._initializations += 1
            return record
        dt = t - self._t
        F, Q = self.model.discrete(dt, self.q)
        H = self.model.measurement_matrix
        x_pred, P_pred = F @ self._x, F @ self._P @ F.T + Q
        innovation = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            lower = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as exc:
            raise Refusal("innovation_not_positive_definite", "Innovation covariance is not positive definite") from exc
        whitened = np.linalg.solve(lower, innovation)
        nis, dof = float(whitened @ whitened), len(z)
        threshold = CHI2_QUANTILES[self.gate_probability][dof - 1]
        if nis > threshold:
            raise Refusal("outlier_gated", f"NIS {nis:.6g} exceeds the chi-square({dof}) "
                          f"{self.gate_probability} gate {threshold}", nis=nis, dof=dof, threshold=threshold,
                          gate_probability=self.gate_probability, innovation=innovation,
                          innovation_covariance=S)
        gain = np.linalg.solve(S, H @ P_pred).T
        x = x_pred + gain @ innovation
        joseph = np.eye(2 * d) - gain @ H
        P = joseph @ P_pred @ joseph.T + gain @ R @ gain.T
        P = 0.5 * (P + P.T)
        record = _seal({**common, "parents": [transformed["identity"], self._state_id],
                        "update": {"kind": "measurement", "prediction_interval_s": dt, "nis": nis, "dof": dof,
                                   "gate_threshold": threshold, "gate_probability": self.gate_probability,
                                   "innovation": innovation, "innovation_covariance": S,
                                   "covariance_update": "joseph"},
                        "state": {"mean": x, "covariance": P}})
        self._install(x, P, t, record)
        self._accepted_updates += 1
        self._nis.append({"nis": nis, "dof": dof, "time": t, "stream_id": obs.stream_id, "sequence": obs.sequence})
        return record

    # ------------------------------------------------------------------ stage 4: candidate physical state
    def _monitor(self) -> dict:
        recent = list(self._nis)
        window: dict[str, Any] = {"updates": len(recent)}
        if recent:
            total, dof = sum(item["nis"] for item in recent), sum(item["dof"] for item in recent)
            low, high = chi2_interval(dof, 0.95)
            window.update(average_nis=total / len(recent), dof=dof, bounds_95=[low / len(recent), high / len(recent)],
                          consistent=bool(low <= total <= high), approximation="wilson_hilferty")
        return {"accepted_updates": self._accepted_updates, "initializations": self._initializations,
                "received_inputs": self._received, "refused_inputs": sum(self._refused.values()),
                "refused_by_code": dict(sorted(self._refused.items())), "recent_nis": recent,
                "recent_inputs": list(self._inputs), "window": window, "dropped": self.dropped()}

    def candidate(self, at: float | None = None) -> dict:
        """The filter state predicted to ``at`` (filter clock); earlier times are refused."""
        if self._x is None:
            raise Refusal("filter_uninitialized", "No prior and no accepted observation yet")
        at = self._t if at is None else finite(at, "at")
        if at < self._t:
            raise Refusal("retrodiction_unsupported", "A candidate before the filter time would need smoothing",
                          filter_time=self._t, requested=at)
        d = self.model.dimension
        F, Q = self.model.discrete(at - self._t, self.q)
        x, P = F @ self._x, F @ self._P @ F.T + Q
        P = 0.5 * (P + P.T)
        variances = np.diag(P)
        if np.any(variances < 0):
            raise Refusal("covariance_degenerate", "Predicted covariance has a negative variance")
        listed = self._contributors[-MAX_LISTED_CONTRIBUTORS:]
        record = self._remember(_seal({
            "schema": CANDIDATE_SCHEMA, "stage": "candidate_state", "parents": [self._state_id],
            "claim_class": "estimated", "time": {"value": at, "clock": self.filter_clock},
            "last_update_time": self._t, "prediction_horizon_s": at - self._t,
            "frame": self.filter_frame, "unit": self.unit, "velocity_unit": f"{self.unit}/s",
            "state": {"position": x[:d], "velocity": x[d:], "mean": x}, "covariance": P,
            "position_std": np.sqrt(variances[:d]), "velocity_std": np.sqrt(variances[d:]),
            "acquisition": sorted(self._acquisition), "prior": self._prior_id,
            "contributing_observations": {"count": len(self._contributors),
                                          "digest": content_identity(self._contributors),
                                          "identities": listed,
                                          "truncated": len(listed) < len(self._contributors)},
            "provenance": {"all_checked_valid": self._validity_checked == len(self._contributors),
                           "checked_observations": self._validity_checked,
                           "checked_when": "at transform time, at each observation's acquisition time",
                           "clock_mappings": sorted(self._mappings_used), "transforms": sorted(self._transforms_used)},
            "residual_monitor": self._monitor(), "configuration": self.configuration()}))
        return record

    # ------------------------------------------------------------------ stage 5: admission
    def admit(self, candidate: dict, policy: AdmissionPolicy | dict | None = None,
              evaluated_at: float | None = None) -> dict:
        policy = self.admission_policy if policy is None else policy
        if policy is None:
            raise Refusal("policy_unspecified", "Admission needs an explicit policy")
        if evaluated_at is None:
            raise Refusal("evaluation_time_unspecified", "Admission needs the evaluation time on the filter clock")
        return self._remember(admit(candidate, policy, evaluated_at))


def admit(candidate: dict, policy: AdmissionPolicy | dict, evaluated_at: float) -> dict:
    """Admit a candidate as operational state, or refuse listing every failed check.

    A refusal keeps the candidate a candidate: its identity is referenced, never rewritten.
    """
    candidate = verify_record(candidate)
    if candidate["stage"] != "candidate_state":
        raise Refusal("not_a_candidate", "Only a candidate_state record can be admitted")
    policy = _policy(policy)
    evaluated_at = finite(evaluated_at, "evaluated_at")
    try:
        return _admit(candidate, policy, evaluated_at)
    except (KeyError, TypeError, IndexError) as exc:
        raise Refusal("malformed_record", f"candidate record lacks admission fields: {exc}") from exc


def _admit(candidate: dict, policy: AdmissionPolicy, evaluated_at: float) -> dict:
    monitor, checks = candidate["residual_monitor"], []

    def check(code: str, passed: bool, **detail: Any) -> None:
        checks.append({"check": code, "passed": bool(passed), **detail})

    accepted = monitor["accepted_updates"]
    check("insufficient_updates", accepted >= policy.min_accepted_updates, accepted=accepted,
          required=policy.min_accepted_updates)
    recent = monitor["recent_nis"][-policy.nis_window:]
    if len(recent) < policy.nis_window:
        check("nis_window_incomplete", False, available=len(recent), required=policy.nis_window)
    else:
        total, dof = sum(item["nis"] for item in recent), sum(item["dof"] for item in recent)
        low, high = chi2_interval(dof, policy.nis_confidence)
        check("nis_inconsistent", low <= total <= high, average_nis=total / len(recent), dof=dof,
              bounds=[low / len(recent), high / len(recent)], confidence=policy.nis_confidence,
              approximation="wilson_hilferty")
    limit = policy.max_position_std * conversion_factor(policy.std_unit, candidate["unit"])
    stds = candidate["position_std"]
    failing = [index for index, value in enumerate(stds) if value > limit]
    check("position_uncertainty_exceeded", not failing, axes=failing, position_std=stds, limit=limit,
          unit=candidate["unit"])
    inputs = monitor["recent_inputs"][-policy.refusal_window:]
    refused = sum(item["outcome"] == "refused" for item in inputs)
    fraction = refused / len(inputs) if inputs else 0.0
    check("refused_fraction_exceeded", fraction <= policy.max_refused_fraction, fraction=fraction,
          refused=refused, inputs=len(inputs), limit=policy.max_refused_fraction)
    age = evaluated_at - candidate["last_update_time"]
    check("state_too_old" if age >= 0 else "evaluated_before_state", 0 <= age <= policy.max_state_age_s,
          age_s=age, limit_s=policy.max_state_age_s)
    if policy.require_valid_provenance:
        provenance = candidate["provenance"]
        check("provenance_unverified", provenance["all_checked_valid"] is True
              and provenance["checked_observations"] == candidate["contributing_observations"]["count"],
              clock_mappings=provenance["clock_mappings"], transforms=provenance["transforms"])
    kinds = candidate["acquisition"]
    check("acquisition_not_admissible", bool(kinds) and set(kinds) <= set(policy.allowed_acquisition),
          acquisition=kinds, allowed=list(policy.allowed_acquisition))
    reasons = [{"code": item["check"], **{k: v for k, v in item.items() if k not in {"check", "passed"}}}
               for item in checks if not item["passed"]]
    common = {"parents": [candidate["identity"]], "candidate": candidate["identity"], "policy": policy.to_json(),
              "policy_identity": content_identity(policy.to_json()),
              "evaluated_at": {"value": evaluated_at, "clock": candidate["time"]["clock"]}, "checks": checks}
    if reasons:
        return _seal({"schema": ADMISSION_REFUSED_SCHEMA, "stage": "admission_refused",
                      "status": "retained_as_candidate", "reasons": reasons, **common})
    return _seal({"schema": ADMITTED_SCHEMA, "stage": "admitted_state", "status": "admitted",
                  "claim_class": "estimated", "time": candidate["time"], "frame": candidate["frame"],
                  "unit": candidate["unit"], "velocity_unit": candidate["velocity_unit"], "state": candidate["state"],
                  "covariance": candidate["covariance"], "position_std": candidate["position_std"],
                  "acquisition": kinds, "authority": "state_admission_only", **common})


# ---------------------------------------------------------------------- ledger
def retain(ledger: "Ledger", record: dict, refs: Iterable[str] = ()) -> dict:
    """Append a stage record as ``ciw.science.fusion-state.v1``, referencing retained parents."""
    record = verify_record(record)
    parents = set(record.get("parents") or [])
    automatic = [entry["entry_id"] for entry in ledger.entries("fusion_state")
                 if entry["body"].get("record_identity") in parents] if parents else []
    body = {"stage": record["stage"], "record_identity": record["identity"], "state": record}
    return ledger.append(LEDGER_SCHEMA, body, refs=sorted(set(refs) | set(automatic)))


# ---------------------------------------------------------------------- synthetic scenario
def _axis_noise(rng: np.random.Generator, dt: float, q: float, d: int) -> np.ndarray:
    block = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return (np.linalg.cholesky(block) @ rng.standard_normal((2, d))).reshape(-1)


def synthetic_scenario(seed: int = 0, *, duration_s: float = 6.0, acceleration_psd: float = 0.5,
                       dropout: float = 0.05, outliers: int = 2, burst: bool = True) -> dict:
    """Seeded two-tracker scenario (``acquisition: "synthetic"``) with ground truth.

    Tracker A reports world-frame positions in mm on ``clock_a`` (offset 12.5 s, rate
    1 + 25 ppm); tracker B reports camera-frame positions in m on ``clock_b`` (offset
    -3.2 s, rate 1 - 40 ppm). Declared calibrations equal the simulated truth; their
    small declared uncertainties are therefore conservative. Samples may be dropped
    (sequence gaps), a few carry 0.5 m outliers, and arrival latency differs per stream.
    """
    seed = integer(seed, "seed", minimum=0, maximum=2 ** 32 - 1)
    duration_s = finite(duration_s, "duration_s", minimum=0.5, maximum=600.0)
    q = finite(acceleration_psd, "acceleration_psd", minimum=0.0, exclusive_minimum=True)
    dropout = finite(dropout, "dropout", minimum=0.0, maximum=0.5)
    outliers = integer(outliers, "outliers", minimum=0, maximum=16)
    rng = np.random.default_rng(seed)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([2.0, -1.0, 0.8])
    calibration = lambda name: {"calibration_id": f"synthetic-{name}", "version": "1"}  # noqa: E731
    clocks = {"clock_a": (12.5, 1.0 + 25e-6), "clock_b": (-3.2, 1.0 - 40e-6)}
    registry = {
        "schema": "ciw.frame-registry.v1", "length_unit": "m",
        "description": "Synthetic two-tracker registry; not a physical calibration",
        "frames": [{"frame_id": "world", "kind": "world"}, {"frame_id": "camera_b", "kind": "camera"}],
        "clocks": [{"clock_id": "world_clock", "kind": "ptp", "resolution_s": 1e-6},
                   {"clock_id": "clock_a", "kind": "device", "resolution_s": 1e-4},
                   {"clock_id": "clock_b", "kind": "device", "resolution_s": 1e-4}],
        "clock_mappings": [{"mapping_id": f"{name}-to-world", "source": name, "target": "world_clock",
                            "offset_s": offset, "rate": rate, "covariance": [[1e-10, 0.0], [0.0, 1e-16]],
                            "valid_from": -1e6, "valid_until": 1e6, "calibration": calibration(f"{name}-sync")}
                           for name, (offset, rate) in clocks.items()],
        "transforms": [{"transform_id": "camera_b-to-world", "source": "camera_b", "target": "world",
                        "rotation": {"matrix": rotation.tolist()}, "translation": translation.tolist(), "unit": "m",
                        "covariance": np.diag([2.5e-9] * 3 + [1e-8] * 3).tolist(), "clock": "world_clock",
                        "estimated_at": 0.0, "valid_from": -1.0, "valid_until": duration_s + 10.0,
                        "calibration": calibration("extrinsic-b")}],
    }
    streams = {
        "tracker_a": {"rate": 20.0, "phase": 0.013, "clock": "clock_a", "frame": "world", "unit": "mm",
                      "covariance": np.array([[4.0, 1.5, 0.5], [1.5, 3.0, 0.4], [0.5, 0.4, 6.0]]),
                      "latency": (0.004, 0.004)},
        "tracker_b": {"rate": 15.0, "phase": 0.031, "clock": "clock_b", "frame": "camera_b", "unit": "m",
                      "covariance": np.array([[9e-6, 2e-6, 0.0], [2e-6, 9e-6, 0.0], [0.0, 0.0, 2.5e-5]]),
                      "latency": (0.03, 0.03)},
    }
    events = sorted((spec["phase"] + k / spec["rate"], name, k) for name, spec in streams.items()
                    for k in range(int((duration_s - spec["phase"]) * spec["rate"]) + 1))
    prior_mean = np.array([0.0, 0.0, 1.0, 0.8, -0.4, 0.1])
    prior_cov = np.diag([0.05 ** 2] * 3 + [0.2 ** 2] * 3)
    x = prior_mean + np.linalg.cholesky(prior_cov) @ rng.standard_normal(6)
    t_prev, truth, arrivals = 0.0, [], []
    dropped: dict[str, list[int]] = {name: [] for name in streams}
    emitted_outliers: list[list] = []
    counts = {name: sum(1 for event in events if event[1] == name) for name in streams}
    outlier_set = {(name, int(k)) for name in streams
                   for k in rng.choice(np.arange(5, max(6, counts[name])), size=min(outliers, max(0, counts[name] - 5)),
                                       replace=False)}
    for t, name, k in events:
        dt = t - t_prev
        if dt > 0:
            x = np.kron(np.array([[1.0, dt], [0.0, 1.0]]), np.eye(3)) @ x
            x = x + _axis_noise(rng, dt, q, 3)
            t_prev = t
        truth.append({"time": t, "stream_id": name, "sequence": k, "position": x[:3].tolist(),
                      "velocity": x[3:].tolist()})
        spec = streams[name]
        cov = spec["covariance"]
        noise = np.linalg.cholesky(cov) @ rng.standard_normal(3)
        lost = rng.random() < dropout or (burst and name == "tracker_b" and 20 <= k <= 22)
        if lost:
            dropped[name].append(k)
            continue
        if name == "tracker_a":
            value = x[:3] * 1000.0 + noise
        else:
            value = rotation.T @ (x[:3] - translation) + noise
        if (name, k) in outlier_set:
            emitted_outliers.append([name, k])
            direction = rng.standard_normal(3)
            value = value + direction / np.linalg.norm(direction) * (500.0 if spec["unit"] == "mm" else 0.5)
        offset, rate = clocks[spec["clock"]]
        device = round((t - offset) / rate / 1e-4) * 1e-4
        low, spread = spec["latency"]
        arrivals.append({"arrival_time": t + low + spread * rng.random(), "observation": {
            "schema": RAW_SCHEMA, "stream_id": name, "sequence": k, "clock": spec["clock"], "device_time_s": device,
            "value": value.tolist(), "unit": spec["unit"], "frame": spec["frame"], "covariance": cov.tolist(),
            "observable": "tracker_position", "acquisition": "synthetic"}})
    arrivals.sort(key=lambda item: item["arrival_time"])
    return plain({
        "schema": SCENARIO_SCHEMA, "acquisition": "synthetic", "seed": seed, "duration_s": duration_s,
        "description": "Seeded synthetic two-tracker scenario for tests and benchmarks; not physical evidence",
        "registry": registry, "filter": {"frame": "world", "clock": "world_clock", "unit": "m"},
        "model": {"dimension": 3, "acceleration_psd": q, "psd_unit": "m^2/s^3", "observables": ["tracker_position"]},
        "prior": {"time": 0.0, "mean": prior_mean, "covariance": prior_cov},
        "truth": truth, "arrivals": arrivals,
        "injected": {"dropped": dropped, "outliers": emitted_outliers,
                     "clock_offsets_s": {name: offset for name, (offset, _) in clocks.items()}},
    })


def scenario_engine(scenario: dict, **options: Any) -> FusionEngine:
    """A ``FusionEngine`` configured from a scenario's registry, model, filter and prior."""
    scenario = mapping(scenario, "scenario")
    if scenario.get("schema") != SCENARIO_SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {SCENARIO_SCHEMA}")
    model = scenario["model"]
    model = ConstantVelocityModel(model["dimension"], model["acceleration_psd"], model["psd_unit"],
                                  tuple(model["observables"]))
    settings = {"prior": scenario["prior"], **options}
    return FusionEngine(model, FrameRegistry.from_json(scenario["registry"]), filter_frame=scenario["filter"]["frame"],
                        filter_clock=scenario["filter"]["clock"], unit=scenario["filter"]["unit"], **settings)


def replay(engine: FusionEngine, scenario: dict, *, step_s: float = 0.05, hold_s: float = 0.1) -> list[dict]:
    """Feed scenario arrivals through a hold-back reorder buffer on the filter clock.

    At each tick ``T`` the observations that have arrived and whose aligned acquisition
    time is at most ``T - hold_s`` are ingested as one batch with ``evaluated_at = T``.
    Observations whose time cannot be aligned are ingested at once (and refused).
    """
    step_s = finite(step_s, "step_s", minimum=1e-4)
    hold_s = finite(hold_s, "hold_s", minimum=0.0)
    arrivals = scenario["arrivals"]
    if not arrivals:
        return []
    records, pending, index = [], [], 0
    tick = arrivals[0]["arrival_time"]
    while index < len(arrivals) or pending:
        tick += step_s
        while index < len(arrivals) and arrivals[index]["arrival_time"] <= tick:
            raw = arrivals[index]["observation"]
            try:
                aligned = engine.aligned_time(raw)[0]
            except Refusal:
                aligned = -math.inf
            pending.append((aligned, raw))
            index += 1
        cut = tick - hold_s if index < len(arrivals) else math.inf
        ready = [raw for aligned, raw in pending if aligned <= cut]
        pending = [(aligned, raw) for aligned, raw in pending if aligned > cut]
        if ready:
            records.extend(engine.ingest_batch(ready, evaluated_at=tick))
    return records


__all__ = [
    "RAW_SCHEMA", "TRANSFORMED_SCHEMA", "FILTERED_SCHEMA", "CANDIDATE_SCHEMA", "ADMITTED_SCHEMA", "REFUSED_SCHEMA",
    "ADMISSION_REFUSED_SCHEMA", "POLICY_SCHEMA", "SCENARIO_SCHEMA", "LEDGER_SCHEMA", "STAGES", "REFUSAL_STAGES",
    "CHI2_QUANTILES", "RawObservation", "ConstantVelocityModel", "AdmissionPolicy", "FusionEngine", "admit",
    "retain", "record_identity", "verify_record", "normal_quantile", "chi2_quantile_wh", "chi2_interval",
    "synthetic_scenario", "scenario_engine", "replay",
]
