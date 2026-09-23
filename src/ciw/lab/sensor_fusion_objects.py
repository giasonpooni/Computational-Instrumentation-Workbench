"""Typed observations, candidate states and admitted states for the synthetic fusion bench.

Scope: the API boundary used by the sensor-fusion experiments (T069, T071-T076).
An :class:`Observation` is a retained reading; a :class:`CandidateState` is what
a filter prediction or update proposes; an :class:`AdmittedState` exists only
after :meth:`FusionSession.admit` has evaluated declared consistency checks.
A :class:`FusionSession` defaults to read-only with the CIW authority
vocabulary (``sensor_fusion`` and ``state_admission`` ``not_performed``);
fusion must be enabled explicitly and is then labelled ``synthetic_only``.

Refusals are explicit: missing readings are never zero-filled, observations in
another frame or under an expired or revoked calibration are retained but not
fused, a lost track needs explicit two-point reacquisition, and nothing is
ever admitted automatically.

Non-claims: admission here is a software gate over synthetic data. It confers
no physical truth, calibration validity, safety or production authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .sensor_fusion_bench import H_POS, chi2_quantile, cv_model

DEFAULT_AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed",
                     "physical_truth": "not_established"}
SYNTHETIC_AUTHORITY = {"state_admission": "synthetic_only", "sensor_fusion": "synthetic_only",
                       "physical_truth": "not_established"}
GAP_STRATEGIES = ("predict_only",)


class FusionRefusal(ValueError):
    """A fusion, gap, reacquisition or admission request was refused; ``code`` names why."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise FusionRefusal("malformed_observation", f"{name} must be a nonempty string")
    return value


def _tick(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FusionRefusal("malformed_observation", f"{name} must be a nonnegative integer tick")
    return value


def _positive_definite(matrix) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        return False
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-15):
        return False
    return bool(np.linalg.eigvalsh(matrix).min() > 0)


def _rows(matrix) -> tuple:
    return tuple(tuple(float(v) for v in row) for row in np.asarray(matrix, dtype=float))


@dataclass(frozen=True)
class Observation:
    """A retained reading in a named frame under a named calibration."""

    sensor_id: str
    frame_id: str
    tick: int
    value: tuple
    covariance: tuple
    calibration_id: str

    def __post_init__(self):
        for name in ("sensor_id", "frame_id", "calibration_id"):
            _text(getattr(self, name), name)
        _tick(self.tick, "tick")
        try:
            value = tuple(float(v) for v in self.value)
        except (TypeError, ValueError):
            raise FusionRefusal("missing_reading", "A missing reading is absent, not a placeholder value") from None
        if not value or not all(math.isfinite(v) for v in value):
            raise FusionRefusal("nonfinite_observation",
                                "Observation values must be finite; a missing reading is absent, not NaN or zero")
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (len(value), len(value)) or not _positive_definite(covariance):
            raise FusionRefusal("covariance_not_positive_definite",
                                "Observation covariance must be symmetric positive definite and match the value")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "covariance", _rows(covariance))

    @property
    def digest(self) -> str:
        return _digest({"kind": "observation", **asdict(self)})


@dataclass(frozen=True)
class CalibrationRecord:
    """Calibration validity over the half-open tick interval [valid_from, valid_until)."""

    calibration_id: str
    sensor_id: str
    frame_id: str
    valid_from: int
    valid_until: int

    def covers(self, tick: int) -> bool:
        return self.valid_from <= tick < self.valid_until


@dataclass(frozen=True)
class FrameTransform:
    """Declared planar rigid transform from ``source`` to ``target`` coordinates."""

    source: str
    target: str
    rotation: tuple
    translation: tuple

    def apply(self, observation: Observation) -> Observation:
        if observation.frame_id != self.source:
            raise FusionRefusal("frame_mismatch", f"Transform expects frame {self.source}, "
                                                  f"observation is in {observation.frame_id}")
        rotation = np.asarray(self.rotation, dtype=float)
        value = rotation @ np.asarray(observation.value) + np.asarray(self.translation, dtype=float)
        covariance = rotation @ np.asarray(observation.covariance) @ rotation.T
        return Observation(observation.sensor_id, self.target, observation.tick, tuple(value),
                           _rows(0.5 * (covariance + covariance.T)), observation.calibration_id)


@dataclass(frozen=True)
class CandidateState:
    """A proposed state. It is never admitted by construction; ``digest`` seals its content."""

    frame_id: str
    tick: int
    mean: tuple
    covariance: tuple
    sources: tuple
    nis: tuple
    track_status: str
    calibration_ids: tuple
    authority: tuple
    session_id: str
    digest: str

    def content(self) -> dict:
        record = asdict(self)
        record.pop("digest")
        return {"kind": "candidate", **record}

    def content_digest(self) -> str:
        return _digest(self.content())


_GATE_TOKEN = object()


class AdmittedState:
    """A state that passed the admission gate; only :meth:`FusionSession.admit` can create one."""

    __slots__ = ("candidate_digest", "frame_id", "tick", "mean", "covariance", "checks", "declared", "authority")

    def __init__(self, candidate, checks, declared, authority, *, _token=None):
        if _token is not _GATE_TOKEN:
            raise FusionRefusal("admission_requires_gate", "An admitted state is created only by the admission gate")
        for name, value in (("candidate_digest", candidate.digest), ("frame_id", candidate.frame_id),
                            ("tick", candidate.tick), ("mean", candidate.mean), ("covariance", candidate.covariance),
                            ("checks", tuple(checks)), ("declared", tuple(sorted(declared.items()))),
                            ("authority", tuple(sorted(authority.items())))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise FusionRefusal("admitted_state_immutable", "An admitted state cannot be modified")


# Admission gate ----------------------------------------------------------------------
def _declared(session, candidate, declared):
    p, limit = declared.get("nis_probability"), declared.get("max_position_std")
    return (isinstance(declared.get("expected_frame_id"), str) and isinstance(p, float) and 0 < p < 1
            and isinstance(limit, float) and limit > 0)


def _position_std(candidate) -> float:
    covariance = np.asarray(candidate.covariance, dtype=float)
    return math.sqrt(float(np.linalg.eigvalsh(covariance[:2, :2]).max()))


ADMISSION_CHECKS = (
    ("declared", "admission_checks_not_declared", _declared),
    ("writable", "read_only_session", lambda s, c, d: not s.read_only),
    ("typed", "not_a_candidate", lambda s, c, d: type(c) is CandidateState),
    ("finite", "nonfinite_state", lambda s, c, d: bool(np.all(np.isfinite(np.asarray(c.mean, dtype=float))))),
    ("covariance", "covariance_not_positive_definite", lambda s, c, d: _positive_definite(c.covariance)),
    ("integrity", "digest_mismatch", lambda s, c, d: c.content_digest() == c.digest),
    ("provenance", "unknown_candidate", lambda s, c, d: c.digest in s._issued),
    ("frame", "frame_mismatch", lambda s, c, d: c.frame_id == d["expected_frame_id"] == s.frame_id),
    ("fresh", "stale_candidate", lambda s, c, d: c.tick == s.tick),
    ("track", "track_lost", lambda s, c, d: c.track_status == "tracking"),
    ("uncertainty", "uncertainty_exceeds_limit", lambda s, c, d: _position_std(c) <= d["max_position_std"]),
    ("innovation", "inconsistent_innovation",
     lambda s, c, d: all(value <= chi2_quantile(d["nis_probability"], dof) for value, dof in c.nis)),
    ("calibration", "calibration_revoked", lambda s, c, d: not set(c.calibration_ids) & s.revoked),
)


def admission_verdict(session, candidate, declared, checks=ADMISSION_CHECKS):
    """Evaluate checks in order; return (refusal code or None, names passed).

    A check that cannot be evaluated fails closed. Passing a reduced ``checks``
    tuple is how the mutation analysis probes whether each check is
    load-bearing; :meth:`FusionSession.admit` always uses the full tuple.
    """
    passed = []
    for name, code, check in checks:
        try:
            ok = bool(check(session, candidate, declared))
        except (AttributeError, TypeError, ValueError, KeyError, np.linalg.LinAlgError):
            ok = False
        if not ok:
            return code, passed
        passed.append(name)
    return None, passed


# Session -----------------------------------------------------------------------------
class FusionSession:
    """Planar constant-velocity fusion over 2-D position observations.

    Defaults to ``read_only=True``: observations are retained, but prediction,
    fusion, reacquisition and admission are refused and the authority record
    equals the CIW vocabulary. ``read_only=False`` enables synthetic fusion only.
    """

    def __init__(self, *, frame_id: str = "world", read_only: bool = True, dt: float = 0.1, q: float = 0.05,
                 track_radius: float | None = None, track_probability: float = 0.99,
                 session_id: str = "synthetic-session"):
        self.frame_id, self.read_only, self.session_id = frame_id, bool(read_only), session_id
        self.authority = dict(DEFAULT_AUTHORITY if self.read_only else SYNTHETIC_AUTHORITY)
        self.F, self.Q = cv_model(dt, q)
        self.dt, self.q = dt, q
        self.track_radius, self.track_probability = track_radius, track_probability
        self.log: list = []
        self.calibrations: dict = {}
        self.revoked: set = set()
        self.admitted: list = []
        self._issued: dict = {}
        self.x = self.P = None
        self.tick: int | None = None
        self.track_status = "uninitialized"

    # Records -----------------------------------------------------------------------
    def register_calibration(self, record: CalibrationRecord) -> None:
        self.calibrations[record.calibration_id] = record

    def revoke_calibration(self, calibration_id: str) -> None:
        self.revoked.add(calibration_id)

    def record(self, observation: Observation) -> dict:
        """Retain an observation without fusing it; allowed in read-only mode."""
        entry = {"digest": observation.digest, "observation": observation, "disposition": "recorded"}
        self.log.append(entry)
        return entry

    def _refuse(self, entry, code, message):
        if entry is not None:
            entry["disposition"] = f"refused:{code}"
        raise FusionRefusal(code, message)

    def _writable(self, entry=None):
        if self.read_only:
            self._refuse(entry, "read_only_session", "The session is read-only; sensor fusion is not performed")

    def _admissible_source(self, observation, entry):
        if observation.frame_id != self.frame_id:
            self._refuse(entry, "frame_mismatch", f"Observation frame {observation.frame_id} differs from session "
                                                  f"frame {self.frame_id}; transform it explicitly first")
        record = self.calibrations.get(observation.calibration_id)
        if record is None or record.sensor_id != observation.sensor_id:
            self._refuse(entry, "calibration_unknown", "Observation cites no registered calibration for its sensor")
        if observation.calibration_id in self.revoked:
            self._refuse(entry, "calibration_revoked", "Observation cites a revoked calibration")
        if not record.covers(observation.tick):
            self._refuse(entry, "calibration_expired", "Observation lies outside its calibration validity interval")
        if len(observation.value) != 2:
            self._refuse(entry, "unsupported_observation", "The session fuses two-dimensional positions only")

    # State ---------------------------------------------------------------------------
    def _issue(self, sources=(), nis=(), calibration_ids=()) -> CandidateState:
        fields = {"frame_id": self.frame_id, "tick": self.tick, "mean": tuple(float(v) for v in self.x),
                  "covariance": _rows(self.P), "sources": tuple(sources),
                  "nis": tuple((float(v), int(d)) for v, d in nis), "track_status": self.track_status,
                  "calibration_ids": tuple(calibration_ids), "authority": tuple(sorted(self.authority.items())),
                  "session_id": self.session_id}
        candidate = CandidateState(digest=_digest({"kind": "candidate", **fields}), **fields)
        self._issued[candidate.digest] = candidate
        return candidate

    def position_radius(self) -> float:
        """Semi-major axis of the ``track_probability`` position confidence ellipse."""
        eigen = float(np.linalg.eigvalsh(self.P[:2, :2]).max())
        return math.sqrt(eigen * chi2_quantile(self.track_probability, 2))

    def _advance(self, tick: int) -> None:
        # Tick-by-tick prediction keeps the arithmetic identical to the batch schedule.
        while self.tick < tick:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.tick += 1
            if self.track_radius is not None and self.position_radius() > self.track_radius:
                self.track_status = "lost"

    def initialize(self, mean, covariance, tick: int) -> CandidateState:
        self._writable()
        self.x, self.P = np.array(mean, dtype=float), np.array(covariance, dtype=float)
        self.tick, self.track_status = _tick(tick, "tick"), "tracking"
        return self._issue()

    def predict(self, tick: int) -> CandidateState:
        """Prediction-only step (the only supported response to a missing reading)."""
        self._writable()
        if self.x is None:
            raise FusionRefusal("not_initialized", "The session has no state to predict")
        if tick < self.tick:
            raise FusionRefusal("out_of_order", "Prediction cannot move backwards in time")
        self._advance(tick)
        return self._issue()

    def handle_gap(self, sensor_id: str, tick: int, strategy: str = "predict_only") -> CandidateState:
        """Missing reading at ``tick``: predict only; substitutes are refused, never fabricated."""
        if strategy == "zero_fill":
            raise FusionRefusal("zero_fill_refused", f"A missing {sensor_id} reading is not replaced by zeros")
        if strategy not in GAP_STRATEGIES:
            raise FusionRefusal("gap_strategy_refused", f"Gap strategy {strategy} would fabricate a reading")
        return self.predict(tick)

    def fuse(self, observation: Observation) -> CandidateState:
        """Retain the observation, then update the state or refuse with a coded reason."""
        entry = self.record(observation)
        self._writable(entry)
        self._admissible_source(observation, entry)
        if self.x is None:
            self._refuse(entry, "not_initialized", "The session has no state to update; reacquire explicitly")
        if observation.tick < self.tick:
            self._refuse(entry, "out_of_order", "Observation is older than the session state")
        self._advance(observation.tick)
        if self.track_status != "tracking":
            self._refuse(entry, "track_lost_requires_reacquisition",
                         "The track is lost; fusion resumes only after explicit reacquisition")
        z = np.asarray(observation.value)
        R = np.asarray(observation.covariance)
        S = H_POS @ self.P @ H_POS.T + R
        K = np.linalg.solve(S, H_POS @ self.P).T
        nu = z - H_POS @ self.x
        A = np.eye(4) - K @ H_POS
        self.x = self.x + K @ nu
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        entry["disposition"] = "fused"
        return self._issue((observation.digest,), ((float(nu @ np.linalg.solve(S, nu)), 2),),
                           (observation.calibration_id,))

    def reacquire(self, first: Observation, second: Observation) -> CandidateState:
        """Explicit two-point initialization from consecutive position readings.

        Position error covariance R, velocity error covariance 2R/dt^2 + q dt/3
        and cross covariance R/dt are exact for the constant-velocity truth.
        """
        entries = [self.record(first), self.record(second)]
        self._writable(entries[1])
        for observation, entry in zip((first, second), entries):
            self._admissible_source(observation, entry)
        if self.track_status == "tracking":
            self._refuse(entries[1], "reacquisition_not_needed", "The track is not lost")
        if second.tick != first.tick + 1 or second.sensor_id != first.sensor_id:
            self._refuse(entries[1], "reacquisition_needs_consecutive_readings",
                         "Two-point reacquisition needs consecutive readings from one sensor")
        R1, R2 = np.asarray(first.covariance), np.asarray(second.covariance)
        dt = self.dt
        velocity = (np.asarray(second.value) - np.asarray(first.value)) / dt
        self.x = np.concatenate([second.value, velocity])
        self.P = np.block([[R2, R2 / dt], [R2 / dt, (R1 + R2) / dt ** 2 + self.q * dt / 3 * np.eye(2)]])
        self.tick, self.track_status = second.tick, "tracking"
        for entry in entries:
            entry["disposition"] = "reacquired"
        return self._issue((first.digest, second.digest), (), (first.calibration_id, second.calibration_id))

    def admit(self, candidate, *, expected_frame_id=None, nis_probability=None,
              max_position_std=None) -> AdmittedState:
        """Admit a candidate only after every declared consistency check passes."""
        declared = {"expected_frame_id": expected_frame_id, "nis_probability": nis_probability,
                    "max_position_std": max_position_std}
        code, passed = admission_verdict(self, candidate, declared)
        if code is not None:
            raise FusionRefusal(code, f"Admission refused at check {len(passed) + 1}: {code}")
        admitted = AdmittedState(candidate, passed, declared, self.authority, _token=_GATE_TOKEN)
        self.admitted.append(admitted)
        return admitted
