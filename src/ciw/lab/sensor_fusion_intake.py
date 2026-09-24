"""Section-4 observation records into Section-5 fusion readings through declared mappings (the intake).

Scope: the pipeline observation -> declared mapping -> fusion -> candidate
state -> explicit admission, joining :mod:`ciw.lab.observation_modes` (typed
records, frame and clock bases, retention and admission, T045-T059) with
:mod:`ciw.lab.sensor_fusion_objects` (the :class:`FusionSession`, T060-T076).

An :class:`ObservationIntake` binds one fusion session to declared
:class:`IntakeChannel` records (which section-4 mode feeds which fusion sensor,
from which frame, through which projection, under which calibration), a
:class:`FusionClock` (fusion tick k is time k dt on a named clock and epoch in
the acquisition basis) and the declared section-4 frame and clock mappings.
A record reaches the session only if it was retained and admitted in an
:class:`~ciw.lab.observation_modes.ObservationLedger`, still matches its
digest, validates, has the geometry class of the session state, has a declared
channel and a declared (not ``not_applied``) calibration, reaches the channel
frame and the fusion clock through declared mappings only (a change of time
basis only by a latency mapping from arrival to acquisition on one clock and
epoch, checked against the record's latency), lies on the tick grid, cites a
registered, unrevoked, unexpired fusion calibration that does not declare the
latency a second time, and was not fused before, neither itself nor as a
re-sent copy with identical content. Every refusal is coded, logged and
leaves the session untouched; the session's own refusals (read-only,
out-of-order, gate) pass through unchanged. Each fused submission keeps its
own lineage entry back to its section-4 record, raw reference and mappings;
:meth:`ObservationIntake.trace` refuses a session that used any reading the
intake did not fuse. Nothing is admitted except by an explicit
:meth:`FusionSession.admit` call.

Acquisition-age staleness (T056) is not applied here: the session fuses each
reading at its own acquisition tick and refuses one older than its state
(``out_of_order``), and neither the intake nor the admission gate declares a
later use time against which an age could be measured.

Non-claims: the declared mappings, calibrations, latencies and noise
parameters are synthetic. The intake establishes bookkeeping and refusal
behaviour of this code, never the validity of a calibration, a clock
synchronization or a physical state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import numpy as np

from ..core.identities import content_identity
from . import observation_modes as om
from .sensor_fusion_bench import H_POS, batch_posterior_banded, cv_model, generator, simulate_truth
from .sensor_fusion_objects import CalibrationRecord, FusionRefusal, FusionSession, Observation

STATE_GEOMETRIES = ("intrinsic", "extrinsic")


class IntakeRefusal(FusionRefusal, om.ObservationRefusal):
    """An intake refusal; it is both a fusion and an observation refusal, with a stable ``code``."""

    def __init__(self, code: str, message: str):
        ValueError.__init__(self, message)
        self.code = code


def _name(value, what):
    if not isinstance(value, str) or not value.strip():
        raise IntakeRefusal("malformed_intake", f"{what} must be a nonempty string")
    return value


@dataclass(frozen=True)
class FusionClock:
    """Fusion tick k is time k * dt seconds on (clock_id, epoch) in the acquisition basis.

    A mapped time farther than ``tolerance_s`` from the tick grid is refused, never rounded silently.
    """

    clock_id: str
    epoch: str
    dt: float
    tolerance_s: float = 1e-9

    def __post_init__(self):
        _name(self.clock_id, "clock_id")
        _name(self.epoch, "epoch")
        if not (isinstance(self.dt, float) and math.isfinite(self.dt) and self.dt > 0):
            raise IntakeRefusal("malformed_intake", "dt must be a positive finite float")
        if not (isinstance(self.tolerance_s, float) and 0 <= self.tolerance_s < self.dt / 2):
            raise IntakeRefusal("malformed_intake", "tolerance_s must be a float in [0, dt / 2)")

    def record(self) -> dict:
        return {"clock_id": self.clock_id, "epoch": self.epoch, "basis": "acquisition", "dt": self.dt,
                "tolerance_s": self.tolerance_s}


@dataclass(frozen=True)
class IntakeChannel:
    """Declared binding of one section-4 mode to one fusion sensor.

    A record of ``mode`` citing ``calibration_ref``, once mapped into
    ``source_frame``, becomes the fusion reading ``projection @ value + offset``
    of ``sensor_id`` under the fusion calibration ``calibration_id`` (declared
    for ``source_frame``), with covariance ``projection C projection^T`` where C
    is the mode's declared noise covariance carried through the applied frame
    rotations.
    """

    mode: str
    sensor_id: str
    source_frame: str
    projection: tuple
    offset: tuple
    calibration_ref: str
    calibration_id: str

    def __post_init__(self):
        if self.mode not in om.MODES:
            raise IntakeRefusal("malformed_intake", f"Unknown observation mode: {self.mode!r}")
        for name in ("sensor_id", "source_frame", "calibration_ref", "calibration_id"):
            _name(getattr(self, name), name)
        if self.calibration_ref == "not_applied":
            raise IntakeRefusal("malformed_intake", "A channel must name the calibration its records cite")
        projection = np.asarray(self.projection, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        components = om.MODES[self.mode].components
        if (projection.shape != (2, components) or offset.shape != (2,) or not np.all(np.isfinite(projection))
                or not np.all(np.isfinite(offset)) or np.linalg.matrix_rank(projection) != 2):
            raise IntakeRefusal("malformed_intake", f"The projection must be a finite rank-2 map from the "
                                                    f"{components} component(s) of {self.mode} to a 2-D reading")
        object.__setattr__(self, "projection", tuple(tuple(float(v) for v in row) for row in projection))
        object.__setattr__(self, "offset", tuple(float(v) for v in offset))

    def record(self) -> dict:
        return {"mode": self.mode, "sensor_id": self.sensor_id, "source_frame": self.source_frame,
                "projection": [list(row) for row in self.projection], "offset": list(self.offset),
                "calibration_ref": self.calibration_ref, "calibration_id": self.calibration_id}

    def identity(self) -> str:
        return "channel:" + content_identity(self.record())


def declared_covariance(observation: om.Observation) -> np.ndarray:
    """The record's declared noise covariance: its variance components, else the mode's isotropic sigma."""
    components = len(observation.value)
    declared = om.declared_variance(observation)
    if declared is not None:
        if components != 1:
            raise IntakeRefusal("no_declared_covariance", "Variance components describe scalar records only")
        return np.array([[declared]])
    sigma = om.MODES[observation.mode].noise_model.get("sigma_m")
    if not isinstance(sigma, float) or not sigma > 0:
        raise IntakeRefusal("no_declared_covariance", f"{observation.mode} declares no metric noise covariance")
    return sigma ** 2 * np.eye(components)


class ObservationIntake:
    """Converts ledger-admitted section-4 records into readings of one :class:`FusionSession`.

    ``geometry`` is the geometry class of the session state: ``extrinsic`` for
    a position in an embedding frame, ``intrinsic`` for a position in a surface
    chart. A record of another geometry class is refused before any channel is
    consulted (``extrinsic_for_intrinsic`` when an extrinsic mode is offered to
    an intrinsic state). The intake holds no authority of its own: it fuses
    through the session, whose read-only default and authority apply.
    """

    def __init__(self, session: FusionSession, clock: FusionClock, channels=(), *, frame_mappings=(),
                 clock_mappings=(), geometry: str = "extrinsic"):
        if not isinstance(session, FusionSession) or not isinstance(clock, FusionClock):
            raise IntakeRefusal("malformed_intake", "An intake binds one FusionSession and one FusionClock")
        if geometry not in STATE_GEOMETRIES:
            raise IntakeRefusal("malformed_intake", f"State geometry must be one of {STATE_GEOMETRIES}")
        if clock.dt != session.dt:
            raise IntakeRefusal("malformed_intake", f"The fusion clock tick {clock.dt} s differs from the session "
                                                    f"step {session.dt} s")
        declared = {}
        for channel in channels:
            if not isinstance(channel, IntakeChannel) or channel.mode in declared:
                raise IntakeRefusal("malformed_intake", "Channels are IntakeChannel records, one per mode")
            if om.MODES[channel.mode].geometry != geometry:
                raise IntakeRefusal("channel_geometry_mismatch", f"{channel.mode} is "
                                                                 f"{om.MODES[channel.mode].geometry}; the session "
                                                                 f"state is {geometry}")
            declared[channel.mode] = channel
        if not all(isinstance(m, om.FrameMapping) for m in frame_mappings) or not all(
                isinstance(m, om.ClockMapping) for m in clock_mappings):
            raise IntakeRefusal("malformed_intake", "Mappings must be declared FrameMapping and ClockMapping records")
        self.session, self.clock, self.geometry = session, clock, geometry
        self.channels = MappingProxyType(declared)
        self.frame_mappings, self.clock_mappings = tuple(frame_mappings), tuple(clock_mappings)
        self.log: list = []
        # One lineage entry per fused submission, in fusion order (never keyed by reading content, so a second
        # record that maps to the same reading cannot overwrite the first one's lineage).
        self.lineage: list = []

    @property
    def authority(self):
        return self.session.authority

    # Conversion ------------------------------------------------------------------------
    def _map_frame(self, record, target):
        current, rotation, used = record, np.eye(len(record.value)), set()
        while current.frame_id != target:
            index = next((i for i, m in enumerate(self.frame_mappings)
                          if m.source == current.frame_id and i not in used), None)
            if index is None:
                raise IntakeRefusal("unmapped_frame", f"No declared frame mapping takes {current.frame_id} to "
                                                      f"{target}")
            used.add(index)
            mapping = self.frame_mappings[index]
            current = om.apply_frame(current, mapping)
            if current.mode == "tracker_measurement":
                rotation = mapping.matrix()[0] @ rotation
        return current, rotation

    def _map_clock(self, record):
        target = (self.clock.clock_id, self.clock.epoch, "acquisition")
        current, used = record, set()
        while (current.clock_id, current.epoch, current.clock_basis) != target:
            here = (current.clock_id, current.epoch, current.clock_basis)
            index = next((i for i, m in enumerate(self.clock_mappings) if i not in used and
                          (m.source_clock, m.source_epoch, m.source_basis) == here), None)
            if index is None:
                hint = (" (an arrival stamp needs a declared latency mapping on its own clock and epoch)"
                        if here[2] == "arrival" else "")
                raise IntakeRefusal("unmapped_clock", f"No declared clock mapping takes {here} to {target}{hint}")
            used.add(index)
            mapping = self.clock_mappings[index]
            if mapping.source_basis != mapping.target_basis:
                # A basis change is a latency mapping: arrival to acquisition on one clock and epoch, so that its
                # offset is the latency alone and can be checked against the record's. A mapping that also changes
                # clock or epoch would fold an unchecked latency into a synchronization offset.
                if (mapping.source_basis, mapping.target_basis) != ("arrival", "acquisition") or (
                        mapping.source_clock, mapping.source_epoch) != (mapping.target_clock, mapping.target_epoch):
                    raise IntakeRefusal("unmapped_clock", f"Clock mapping {mapping.reference!r} changes the time basis "
                                                          f"from {mapping.source_basis} to {mapping.target_basis} "
                                                          f"together with the clock or epoch; a basis change must be "
                                                          f"a latency mapping from arrival to acquisition on one "
                                                          f"clock and epoch, declared apart from any synchronization")
                if current.latency_s is not None and (mapping.rate, mapping.offset_s) != (1.0, -current.latency_s):
                    raise IntakeRefusal("latency_mismatch", f"The declared latency mapping (offset "
                                                            f"{mapping.offset_s} s) contradicts the record's latency "
                                                            f"{current.latency_s} s")
            current = om.apply_clock(current, mapping)
        return current

    def _tick(self, time_s: float) -> int:
        tick = round(time_s / self.clock.dt)
        if tick < 0:
            raise IntakeRefusal("before_epoch", "The mapped time precedes the fusion clock epoch")
        if abs(time_s - tick * self.clock.dt) > self.clock.tolerance_s:
            raise IntakeRefusal("off_tick_grid", f"Mapped time {time_s!r} s is {time_s - tick * self.clock.dt:.3g} s "
                                                 f"from the fusion tick grid")
        return tick

    def _calibration(self, channel, tick) -> CalibrationRecord:
        record = self.session.calibrations.get(channel.calibration_id)
        if record is None or record.sensor_id != channel.sensor_id:
            raise IntakeRefusal("calibration_unknown", f"No fusion calibration {channel.calibration_id} is registered "
                                                       f"for sensor {channel.sensor_id}")
        if channel.calibration_id in self.session.revoked:
            raise IntakeRefusal("calibration_revoked", f"Fusion calibration {channel.calibration_id} is revoked")
        if record.frame_id != channel.source_frame:
            raise IntakeRefusal("calibration_frame_mismatch", f"Calibration {record.calibration_id} is declared for "
                                                              f"{record.frame_id}, not {channel.source_frame}")
        if record.latency_ticks != 0:
            raise IntakeRefusal("double_latency", "The intake maps stamps to the acquisition basis; a fusion "
                                                  "calibration may not declare the latency again")
        if not record.covers(tick):
            raise IntakeRefusal("calibration_expired", f"Tick {tick} lies outside the validity of "
                                                       f"{record.calibration_id}")
        return record

    def _convert(self, ledger, digest):
        record = ledger.admitted(digest)
        om.validate(record)
        mode = om.MODES[record.mode]
        if mode.geometry != self.geometry:
            code = ("extrinsic_for_intrinsic" if (mode.geometry, self.geometry) == ("extrinsic", "intrinsic")
                    else "geometry_mismatch")
            raise IntakeRefusal(code, f"{record.mode} observes {mode.geometry} geometry; the session state is "
                                      f"{self.geometry}")
        channel = self.channels.get(record.mode)
        if channel is None:
            raise IntakeRefusal("no_channel", f"No intake channel is declared for {record.mode}")
        if record.calibration_ref == "not_applied":
            raise IntakeRefusal("uncalibrated_record", "An uncalibrated record cannot be fused")
        if record.calibration_ref != channel.calibration_ref:
            raise IntakeRefusal("calibration_not_declared", f"The channel is declared for {channel.calibration_ref}, "
                                                            f"the record cites {record.calibration_ref}")
        mapped, rotation = self._map_frame(record, channel.source_frame)
        mapped = self._map_clock(mapped)
        tick = self._tick(mapped.time_s)
        self._calibration(channel, tick)
        projection, offset = np.asarray(channel.projection), np.asarray(channel.offset)
        covariance = projection @ (rotation @ declared_covariance(record) @ rotation.T) @ projection.T
        reading = Observation(channel.sensor_id, self.session.frame_id, tick,
                              tuple(projection @ np.asarray(mapped.value) + offset), 0.5 * (covariance + covariance.T),
                              channel.calibration_id, channel.source_frame)
        link = {"observation_digest": digest, "fusion_digest": reading.digest, "mode": record.mode,
                "sensor_id": channel.sensor_id, "tick": tick, "channel": channel.identity(),
                "mappings": list(mapped.mappings), "raw_ref": record.raw_ref,
                "calibration_ref": record.calibration_ref, "calibration_id": channel.calibration_id,
                "state_admission": ledger.admission(digest)["state_admission"]}
        return reading, link

    def convert(self, ledger: om.ObservationLedger, digest: str) -> tuple[Observation, dict]:
        """The fusion reading and lineage entry for a ledger record, or a coded refusal (logged)."""
        entry = {"observation_digest": digest, "disposition": "submitted"}
        self.log.append(entry)
        try:
            reading, link = self._convert(ledger, digest)
        except (om.ObservationRefusal, FusionRefusal) as exc:
            entry["disposition"] = f"refused:{exc.code}"
            if isinstance(exc, FusionRefusal):  # an IntakeRefusal is one too
                raise
            raise IntakeRefusal(exc.code, str(exc)) from None
        entry.update(disposition="converted", fusion_digest=reading.digest)
        return reading, link

    def fuse(self, ledger: om.ObservationLedger, digest: str):
        """Convert, then fuse through the session; returns the session's new candidate state.

        Each measurement is counted once. A section-4 record this intake has
        already fused is refused before conversion (``already_fused``); a
        different record that converts to a reading identical to one already
        fused, such as a re-sent copy with a new sequence number and raw
        reference, is refused before the session sees it
        (``duplicate_reading``). A record the session refused may be offered
        again.
        """
        if any(link["observation_digest"] == digest for link in self.lineage):
            self.log.append({"observation_digest": digest, "disposition": "refused:already_fused"})
            raise IntakeRefusal("already_fused", "This section-4 record was already fused through this intake; a "
                                                 "measurement is counted once")
        reading, link = self.convert(ledger, digest)
        entry = self.log[-1]
        first = next((earlier for earlier in self.lineage if earlier["fusion_digest"] == reading.digest), None)
        if first is not None:
            entry["disposition"] = "refused:duplicate_reading"
            raise IntakeRefusal("duplicate_reading", f"The record converts to the reading already fused from "
                                                     f"{first['raw_ref']} at tick {first['tick']}; a re-sent copy is "
                                                     f"not a second measurement")
        try:
            candidate = self.session.fuse(reading)
        except FusionRefusal as exc:
            entry["disposition"] = f"refused:{exc.code}"
            raise
        entry["disposition"] = "fused"
        self.lineage.append(link)
        return candidate

    def trace(self) -> list:
        """Lineage of every reading the session used, in order; any reading used around the intake is refused.

        The session's fused and reacquired readings, in log order, must be
        exactly the readings this intake fused, one lineage entry each. A
        reading fused or reacquired directly on the session, including a
        repeat of one the intake fused, is a surplus and raises
        ``untraced_reading``.
        """
        used = [entry["digest"] for entry in self.session.log if entry["disposition"] in ("fused", "reacquired")]
        through = [link["fusion_digest"] for link in self.lineage]
        if used != through:
            surplus = sum((Counter(used) - Counter(through)).values())
            missing = sum((Counter(through) - Counter(used)).values())
            raise IntakeRefusal("untraced_reading", f"The session used {len(used)} reading(s) and the intake fused "
                                                    f"{len(through)}: {surplus} reading(s) did not come through the "
                                                    f"intake and {missing} lineage entries have no session reading")
        return [dict(link) for link in self.lineage]


# Demonstration: tracker records through the intake into a candidate and an explicit admission -----------------
DEMO_SEED = 75_2027
DEMO_TICKS = 12
DT, Q_SPECTRAL = 0.125, 0.05  # dyadic tick so that mapped stamps land on the grid exactly
PRIOR_MEAN = np.array([0.0, 0.0, 1.0, 0.5])  # the section-5 bench prior
PRIOR_COV = np.diag([0.25, 0.25, 0.04, 0.04])
LATENCY_S, SYNC_OFFSET_S, MARKER_HEIGHT_M = 0.0078125, -1.5, 0.625
QUARTER_TURN = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
TRANSLATION = (0.25, -0.5, 1.0)
ROOM_TO_CELL = om.FrameMapping("tracker:room", "tracker:cell", QUARTER_TURN, TRANSLATION,
                               "calibration:tracker-extrinsic-declared")
ARRIVAL_TO_ACQUISITION = om.ClockMapping("clock:tracker", "epoch:run-0", "arrival", "clock:tracker", "epoch:run-0",
                                         "acquisition", 1.0, -LATENCY_S, "declared tracker latency")
TRACKER_TO_FUSION = om.ClockMapping("clock:tracker", "epoch:run-0", "acquisition", "clock:fusion", "epoch:fusion-0",
                                    "acquisition", 1.0, SYNC_OFFSET_S, "declared synchronization")
FUSION_CLOCK = FusionClock("clock:fusion", "epoch:fusion-0", DT)
TRACKER_CALIBRATION = "calibration:tracker-declared-synthetic"
TRACKER_CHANNEL = IntakeChannel("tracker_measurement", "tracker", "tracker:cell", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                                (0.0, 0.0), TRACKER_CALIBRATION, "trk-cal")
DECLARED_ADMISSION = {"expected_frame_id": "world", "nis_probability": 0.999, "max_position_std": 0.01}
DECISION = "declared synthetic admission for the intake demonstration"


def tracker_records(seed: int = DEMO_SEED, ticks: int = DEMO_TICKS) -> tuple[np.ndarray, list]:
    """Planar truth and one room-frame tracker record per tick, stamped on arrival at the tracker clock.

    The marker's cell-frame position is (x, y, height); the room-frame value is
    the inverse of the declared room-to-cell mapping plus isotropic noise of
    the tracker mode's declared sigma.
    """
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    truth = simulate_truth(rng, F, Q, PRIOR_MEAN, PRIOR_COV, 1, ticks)[0]
    sigma = om.MODES["tracker_measurement"].noise_model["sigma_m"]
    noise = sigma * rng.standard_normal((ticks, 3))
    rotation, translation = ROOM_TO_CELL.matrix()
    records = []
    for k in range(1, ticks + 1):
        cell = np.array([truth[k, 0], truth[k, 1], MARKER_HEIGHT_M])
        room = rotation.T @ (cell - translation) + noise[k - 1]
        records.append(om.observe("tracker_measurement", room, unit="m", frame_id="tracker:room",
                                  clock_id="clock:tracker", clock_basis="arrival", epoch="epoch:run-0",
                                  time_s=k * DT - SYNC_OFFSET_S + LATENCY_S, latency_s=LATENCY_S,
                                  calibration_ref=TRACKER_CALIBRATION, sequence=k, raw_ref=f"raw:tracker:{k}"))
    return truth, records


def _state(session) -> str:
    """Digest of everything a refusal must leave untouched."""
    return content_identity({"x": [float(v) for v in session.x], "P": [[float(v) for v in row] for row in session.P],
                             "tick": session.tick, "track": session.track_status, "latest": session._latest,
                             "innovations": len(session._innovations), "admitted": len(session.admitted)})


def refusal_code(action) -> str:
    """The code of a fusion, intake or observation refusal raised by ``action``, or ``none``."""
    try:
        action()
    except (FusionRefusal, om.ObservationRefusal) as exc:
        return exc.code
    return "none"


def _hand_mapped(record) -> tuple:
    """The declared room-to-cell quarter turn and the channel projection written out by hand."""
    x, y, _ = record.value
    return (-y + TRANSLATION[0], x + TRANSLATION[1])


def demonstration(seed: int = DEMO_SEED, ticks: int = DEMO_TICKS) -> dict:
    """Run the pipeline end to end, then every intake refusal against the last tick, then admit explicitly."""
    truth, records = tracker_records(seed, ticks)
    session = FusionSession(read_only=False, frame_id="world", dt=DT, q=Q_SPECTRAL, session_id="intake-demonstration")
    session.register_calibration(CalibrationRecord("trk-cal", "tracker", "tracker:cell", 0, 1_000_000))
    ledger = om.ObservationLedger(read_only=False)
    mappings = {"frame_mappings": (ROOM_TO_CELL,), "clock_mappings": (ARRIVAL_TO_ACQUISITION, TRACKER_TO_FUSION)}
    intake = ObservationIntake(session, FUSION_CLOCK, (TRACKER_CHANNEL,), **mappings)
    digests = [ledger.retain(record)["observation_digest"] for record in records]
    admissions = [ledger.admit(digest, DECISION) for digest in digests[:-1]]
    session.initialize(PRIOR_MEAN, PRIOR_COV, 0)
    for digest in digests[:-1]:
        intake.fuse(ledger, digest)
    before = _state(session)
    last = records[-1]

    def retained(record):
        """Retain a variant and attempt its ledger admission; returns (digest, admission refusal code or none)."""
        digest = ledger.retain(record)["observation_digest"]
        return digest, refusal_code(lambda: ledger.admit(digest, DECISION))

    def variant(sequence, **changes):
        return replace(last, sequence=sequence, raw_ref=f"raw:tracker:variant-{sequence}", **changes)

    def other_intake(channel=TRACKER_CHANNEL, target=session, geometry="extrinsic", **declared):
        return ObservationIntake(target, FUSION_CLOCK, (channel,) if channel else (), geometry=geometry,
                                 **dict(mappings, **declared))

    chord = om.observe("camera_chord_distance", 0.1125, unit="m", frame_id="camera_rig:stereo-0",
                       clock_id="clock:daq", clock_basis="acquisition", epoch="epoch:run-0", time_s=1.0,
                       calibration_ref=TRACKER_CALIBRATION, sequence=950, raw_ref="raw:chord:950")
    encoder = om.observe("encoder_displacement", 0.0425, unit="m", frame_id="axis:x", clock_id="clock:daq",
                         clock_basis="acquisition", epoch="epoch:run-0", time_s=1.0,
                         calibration_ref=TRACKER_CALIBRATION, sequence=951, raw_ref="raw:encoder:951")
    for calibration_id, until, latency in (("trk-cal-short", ticks, 0), ("trk-cal-revoked", 1_000_000, 0),
                                           ("trk-cal-latency", 1_000_000, 1)):
        session.register_calibration(CalibrationRecord(calibration_id, "tracker", "tracker:cell", 0, until, latency))
    session.revoke_calibration("trk-cal-revoked")
    chart_session = FusionSession(read_only=False, frame_id="surface_chart:cylinder-r0.1", dt=DT, q=Q_SPECTRAL)
    read_only_session = FusionSession(dt=DT, q=Q_SPECTRAL)
    read_only_session.register_calibration(CalibrationRecord("trk-cal", "tracker", "tracker:cell", 0, 1_000_000))

    stranger = variant(900)
    cases = {}  # name: (intake, digest, ledger admission outcome)
    cases["not_retained"] = (intake, stranger.digest(), "not_attempted")
    cases["not_admitted"] = (intake, digests[-1], "not_attempted")
    cases["missing_calibration"] = (intake, *retained(variant(901, calibration_ref=None)))
    cases["uncalibrated_record"] = (intake, *retained(variant(902, calibration_ref="not_applied")))
    cases["calibration_not_declared"] = (intake, *retained(variant(903, calibration_ref="calibration:other")))
    cases["unmapped_frame"] = (intake, *retained(variant(904, frame_id="tracker:lab")))
    cases["unmapped_clock"] = (intake, *retained(variant(905, clock_id="clock:other")))
    cases["unmapped_epoch"] = (intake, *retained(variant(906, epoch="epoch:run-1")))
    cases["latency_mismatch"] = (intake, *retained(variant(907, latency_s=2 * LATENCY_S)))
    cases["off_tick_grid"] = (intake, *retained(variant(908, time_s=last.time_s + 0.015625)))
    genuine_copy = retained(variant(909))
    cases["arrival_without_latency_mapping"] = (other_intake(clock_mappings=(TRACKER_TO_FUSION,)), *genuine_copy)
    cases["no_channel"] = (intake, *retained(chord))
    cases["geometry_mismatch"] = (intake, *retained(encoder))
    cases["extrinsic_for_intrinsic"] = (other_intake(None, chart_session, "intrinsic"), *genuine_copy)
    cases["extrinsic_chord_for_intrinsic"] = (other_intake(None, chart_session, "intrinsic"), *retained(chord))
    cases["calibration_expired"] = (other_intake(replace(TRACKER_CHANNEL, calibration_id="trk-cal-short")),
                                    *genuine_copy)
    cases["calibration_revoked"] = (other_intake(replace(TRACKER_CHANNEL, calibration_id="trk-cal-revoked")),
                                    *genuine_copy)
    cases["double_latency"] = (other_intake(replace(TRACKER_CHANNEL, calibration_id="trk-cal-latency")),
                               *genuine_copy)
    cases["out_of_order"] = (intake, *retained(variant(910, time_s=last.time_s - 2 * DT)))
    cases["read_only_session"] = (other_intake(target=read_only_session), *genuine_copy)
    # One mapping that synchronizes an arrival stamp as if it were an acquisition time. For a record whose declared
    # latency is one tick it lands on the grid one tick late, so only the basis rule can catch it.
    sync_on_arrival = om.ClockMapping("clock:tracker", "epoch:run-0", "arrival", "clock:fusion", "epoch:fusion-0",
                                      "acquisition", 1.0, SYNC_OFFSET_S, "synchronization declared on arrival stamps")
    cases["combined_latency_and_synchronization"] = (
        other_intake(clock_mappings=(sync_on_arrival,)),
        *retained(variant(912, latency_s=DT, time_s=last.time_s - LATENCY_S + DT)))
    # Each measurement is fused once: the first record again, and a re-sent copy of it under a new sequence
    # number and raw reference, are both refused before the session sees them.
    cases["already_fused"] = (intake, digests[0], "none")
    cases["duplicate_reading"] = (intake, *retained(replace(records[0], sequence=911, raw_ref="raw:tracker:resent-1")))
    refusals = {name: {"admission": admission, "intake": refusal_code(lambda i=target, d=digest: i.fuse(ledger, d))}
                for name, (target, digest, admission) in cases.items()}
    unchanged = _state(session) == before and read_only_session.x is None and chart_session.log == []

    # The only difference between the refused tick-N record and the fused one is its ledger admission.
    admissions.append(ledger.admit(digests[-1], DECISION))
    candidate = intake.fuse(ledger, digests[-1])
    reading = session.log[-1]["observation"]
    hand = _hand_mapped(last)
    sigma = om.MODES["tracker_measurement"].noise_model["sigma_m"]
    F, Q = cv_model(DT, Q_SPECTRAL)
    means, covariance = batch_posterior_banded(F, Q, PRIOR_MEAN, PRIOR_COV, [(H_POS, sigma ** 2 * np.eye(2))] * ticks,
                                               [np.array(_hand_mapped(record)) for record in records])
    mean, cov = np.asarray(candidate.mean), np.asarray(candidate.covariance)
    auto_admitted = len(session.admitted)
    admitted = session.admit(candidate, **DECLARED_ADMISSION)
    trace = intake.trace()
    final_link = trace[-1]

    # A reading fused around the intake, here a repeat of one the intake fused, makes trace() refuse.
    side = FusionSession(read_only=False, frame_id="world", dt=DT, q=Q_SPECTRAL, session_id="intake-bypass")
    side.register_calibration(CalibrationRecord("trk-cal", "tracker", "tracker:cell", 0, 1_000_000))
    side.initialize(PRIOR_MEAN, PRIOR_COV, 0)
    side_intake = ObservationIntake(side, FUSION_CLOCK, (TRACKER_CHANNEL,), **mappings)
    side_intake.fuse(ledger, digests[0])
    traced_before = len(side_intake.trace())
    side.fuse(side.log[-1]["observation"])
    bypass = {"traced_before_bypass": traced_before,
              "fused_by_session_after_bypass": sum(e["disposition"] == "fused" for e in side.log),
              "trace_after_direct_fusion": refusal_code(side_intake.trace)}
    return {
        "seed": seed, "ticks": ticks, "dt": DT, "q": Q_SPECTRAL, "channel": TRACKER_CHANNEL.record(),
        "clock": FUSION_CLOCK.record(), "declared_sigma_m": sigma,
        "mappings": {"frame": ROOM_TO_CELL.record(), "arrival_to_acquisition": ARRIVAL_TO_ACQUISITION.record(),
                     "tracker_to_fusion": TRACKER_TO_FUSION.record()},
        "records": [record.record() for record in records],
        "ledger": {"retained": len(ledger.retained()), "admitted": len(admissions),
                   "admission_values": sorted({a["state_admission"] for a in admissions})},
        "final_reading": {"tick": reading.tick, "value": list(reading.value), "hand_mapped_value": list(hand),
                          "value_gap_m": max(abs(a - b) for a, b in zip(reading.value, hand)),
                          "covariance": [list(row) for row in reading.covariance],
                          "covariance_gap_m2": float(np.max(np.abs(np.asarray(reading.covariance)
                                                                   - sigma ** 2 * np.eye(2)))),
                          "origin_frame_id": reading.origin_frame_id, "calibration_id": reading.calibration_id},
        "reference": {"candidate_mean": list(candidate.mean), "batch_mean": [float(v) for v in means[-1]],
                      "relative_mean_gap": float(np.max(np.abs(means[-1] - mean)) / np.max(np.abs(means[-1]))),
                      "relative_covariance_gap": float(np.max(np.abs(covariance - cov)) / np.max(np.abs(covariance)))},
        "estimate_error_m": [float(v) for v in mean[:2] - truth[-1, :2]],
        "candidate_nis": [list(pair) for pair in candidate.nis],
        "refusals": refusals, "state_unchanged_by_refusals": unchanged,
        "auto_admitted_before_gate": auto_admitted, "admitted_after_gate": len(session.admitted),
        "declared_admission": dict(DECLARED_ADMISSION), "admitted_checks": list(admitted.checks),
        "admitted_candidate_digest_matches": admitted.candidate_digest == candidate.digest,
        "trace": {"readings": len(trace), "fused_by_session": sum(e["disposition"] == "fused" for e in session.log),
                  "all_ledger_admitted_with_raw_ref": all(
                      link["state_admission"] == om.SYNTHETIC_ONLY and link["raw_ref"]
                      and ledger.admission(link["observation_digest"]) is not None for link in trace),
                  "final_source_is_last_record": (final_link["observation_digest"] == digests[-1]
                                                  and final_link["fusion_digest"] == candidate.sources[0]),
                  "distinct_records": len({link["observation_digest"] for link in trace}),
                  "final_mappings": final_link["mappings"], "final_raw_ref": final_link["raw_ref"],
                  "bypass": bypass},
        "intake_dispositions": sorted({entry["disposition"] for entry in intake.log}),
        "authorities": {"ledger": dict(ledger.authority), "session": dict(session.authority),
                        "candidate": [list(pair) for pair in candidate.authority],
                        "admitted": [list(pair) for pair in admitted.authority], "intake": dict(intake.authority),
                        "lineage": trace, "ledger_admissions": admissions},
    }


# Vocabulary audit ---------------------------------------------------------------------------------------------
def _admission_values(value, found: list) -> list:
    """Every value written under a ``state_admission`` key (dicts, mappings and (key, value) pairs)."""
    if isinstance(value, (dict, MappingProxyType)):
        for key, item in value.items():
            if key == "state_admission":
                found.append(item)
            _admission_values(item, found)
    elif isinstance(value, (list, tuple)):
        if len(value) == 2 and value[0] == "state_admission":
            found.append(value[1])
        for item in value:
            _admission_values(item, found)
    return found


def admission_vocabulary_audit(demo: dict | None = None) -> dict:
    """Collect the state_admission values every lab estimator writes, read-only and writable.

    Estimators: the section-4 :class:`~ciw.lab.observation_modes.StateStore` and
    :class:`~ciw.lab.observation_modes.ObservationLedger`, the section-5
    :class:`FusionSession` with its candidate and admitted states, and the
    intake's lineage (taken from ``demo``, a :func:`demonstration` result).
    """
    demo = demonstration() if demo is None else demo
    record = om.observe("intrinsic_geodesic_distance", 0.12, unit="m", frame_id="surface_chart:cylinder-r0.1",
                        clock_id="clock:daq", clock_basis="acquisition", epoch="epoch:run-0", time_s=1.0,
                        calibration_ref="calibration:declared-synthetic", raw_ref="raw:audit:0")
    emitted = {}
    for name, read_only in (("StateStore (default)", True), ("StateStore (writable)", False)):
        store = om.StateStore("intrinsic_geodesic_distance", 0.0, 1.0, read_only=read_only)
        kept = store.retain(record)
        items = [dict(store.authority), kept]
        if not read_only:
            items += [store.admit(kept["observation_digest"], "audit"), store.update(kept, 0.5), store.retained()]
        emitted[name] = items
    for name, read_only in (("ObservationLedger (default)", True), ("ObservationLedger (writable)", False)):
        ledger = om.ObservationLedger(read_only=read_only)
        kept = ledger.retain(record)
        items = [dict(ledger.authority), kept]
        if not read_only:
            items += [ledger.admit(kept["observation_digest"], "audit"), ledger.retained()]
        emitted[name] = items
    emitted["FusionSession (default)"] = [dict(FusionSession().authority)]
    authorities = demo["authorities"]
    emitted["FusionSession (writable)"] = [authorities["session"], authorities["candidate"], authorities["admitted"]]
    emitted["ObservationIntake"] = [authorities["intake"], authorities["lineage"], authorities["ledger"],
                                    authorities["ledger_admissions"]]
    values = {name: sorted(set(_admission_values(items, []))) for name, items in emitted.items()}
    seen = sorted({value for found in values.values() for value in found})
    return {"vocabulary": list(om.ADMISSION_VOCABULARY), "estimators": values, "values_seen": seen,
            "outside_vocabulary": sorted(set(seen) - set(om.ADMISSION_VOCABULARY)),
            "estimators_without_values": sorted(name for name, found in values.items() if not found)}
