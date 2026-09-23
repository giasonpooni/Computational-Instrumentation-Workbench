"""Frame, clock and calibration registry with uncertainty-carrying transforms.

A transform maps coordinates expressed in ``source`` into ``target``:
``p_target = R p_source + t``. Its 6x6 covariance is over the left perturbation
``xi = [phi, rho]`` (rotation vector in rad, translation in the registry length
unit): ``T_true = Exp(xi) T``. Composition and inversion propagate covariance
to first order through the SE(3) adjoint; independent edges are assumed and the
chain retained so a reviewer can see which calibrations were combined.

Every transform carries its calibration reference, estimation time and a
half-open validity interval on a named clock. A query at a time outside that
interval is refused as stale rather than extrapolated. Timestamps without a
clock are refused, and simulation clocks never map to physical clocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ._common import Refusal, canonical_json, content_identity, covariance, finite, mapping, plain, require_keys, text, vector
from .units import conversion_factor, require_dimension

SCHEMA = "ciw.frame-registry.v1"
FRAME_KINDS = frozenset({"world", "machine", "tool", "camera", "image", "surface", "tangent", "sensor", "body",
                         "datum", "chart", "tracker", "imu"})
CLOCK_KINDS = frozenset({"utc", "tai", "gps", "ptp", "monotonic", "device", "simulation"})
ORTHONORMAL_TOLERANCE = 1e-9
MAX_ITEMS = 4096


def skew(vector3: np.ndarray) -> np.ndarray:
    x, y, z = vector3
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Logarithm of SO(3) as a rotation vector (for reporting and comparison)."""
    cosine = max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-12:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        diagonal = np.sqrt(np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0))
        axis = diagonal / np.linalg.norm(diagonal)
        index = int(np.argmax(diagonal))
        signs = np.sign(rotation[index] + rotation[:, index])
        signs[index] = 1.0
        return angle * axis * np.where(signs == 0, 1.0, signs)
    return angle / (2 * math.sin(angle)) * np.array([
        rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])


def adjoint(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, 3:] = rotation
    result[3:, :3] = skew(translation) @ rotation
    return result


def _rotation(value: Any, name: str) -> np.ndarray:
    value = mapping(value, name)
    if set(value) == {"quaternion_wxyz"}:
        q = vector(value["quaternion_wxyz"], name + ".quaternion_wxyz", 4)
        if abs(np.linalg.norm(q) - 1.0) > ORTHONORMAL_TOLERANCE:
            raise Refusal("malformed_rotation", f"{name} quaternion must have unit norm (it is never renormalized)")
        return rotation_from_quaternion(q)
    if set(value) == {"matrix"}:
        rotation = np.array([vector(row, name + ".matrix", 3) for row in value["matrix"]]) if isinstance(
            value["matrix"], list) and len(value["matrix"]) == 3 else None
        if rotation is None:
            raise Refusal("malformed_rotation", f"{name}.matrix must be 3x3")
        if (np.max(np.abs(rotation.T @ rotation - np.eye(3))) > ORTHONORMAL_TOLERANCE
                or np.linalg.det(rotation) <= 0):
            raise Refusal("malformed_rotation", f"{name}.matrix must be a proper orthonormal rotation")
        return rotation
    raise Refusal("malformed_rotation", f"{name} must declare exactly one of quaternion_wxyz or matrix")


@dataclass(frozen=True)
class Frame:
    frame_id: str
    kind: str
    description: str = ""


@dataclass(frozen=True)
class Clock:
    clock_id: str
    kind: str
    resolution_s: float


@dataclass(frozen=True)
class Calibration:
    calibration_id: str
    version: str
    digest: str | None = None

    def to_json(self) -> dict:
        record = {"calibration_id": self.calibration_id, "version": self.version}
        if self.digest is not None:
            record["digest"] = self.digest
        return record


@dataclass(frozen=True)
class ClockMapping:
    """``t_target = offset_s + rate * t_source`` with a 2x2 (offset, rate) covariance."""

    mapping_id: str
    source: str
    target: str
    offset_s: float
    rate: float
    covariance: np.ndarray
    valid_from: float
    valid_until: float
    calibration: Calibration


@dataclass(frozen=True)
class Transform:
    transform_id: str
    source: str
    target: str
    rotation: np.ndarray
    translation: np.ndarray
    covariance: np.ndarray
    clock: str
    estimated_at: float
    valid_from: float
    valid_until: float
    calibration: Calibration


@dataclass
class Resolved:
    """A composed transform with the chain, calibrations and validity it relied on."""

    source: str
    target: str
    rotation: np.ndarray
    translation: np.ndarray
    covariance: np.ndarray
    unit: str
    chain: list[dict] = field(default_factory=list)

    def apply(self, point: np.ndarray, point_covariance: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        mapped = self.rotation @ point + self.translation
        jacobian = np.hstack([-skew(mapped), np.eye(3)])
        result = jacobian @ self.covariance @ jacobian.T
        if point_covariance is not None:
            result = result + self.rotation @ point_covariance @ self.rotation.T
        return mapped, result

    def to_json(self) -> dict:
        return plain({"source": self.source, "target": self.target, "rotation_matrix": self.rotation,
                      "rotation_vector_rad": rotation_vector(self.rotation), "translation": self.translation,
                      "unit": self.unit, "covariance": self.covariance, "chain": self.chain})


def _calibration(value: Any, name: str) -> Calibration:
    value = require_keys(value, name, {"calibration_id", "version"}, {"digest"})
    digest = value.get("digest")
    if digest is not None and (not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71):
        raise Refusal("malformed_record", f"{name}.digest must be a sha256: content identity")
    return Calibration(text(value["calibration_id"], name + ".calibration_id", 256),
                       text(value["version"], name + ".version", 64), digest)


class FrameRegistry:
    """Named frames and clocks plus the calibrated transforms relating them."""

    def __init__(self, length_unit: str = "m"):
        require_dimension(length_unit, "m", "registry length unit")
        self.length_unit = length_unit
        self.frames: dict[str, Frame] = {}
        self.clocks: dict[str, Clock] = {}
        self.clock_mappings: dict[str, ClockMapping] = {}
        self.transforms: dict[str, Transform] = {}
        self._source: dict | None = None

    # ------------------------------------------------------------------ declaration
    def add_frame(self, frame_id: str, kind: str, description: str = "") -> Frame:
        text(frame_id, "frame_id", 128)
        if kind not in FRAME_KINDS:
            raise Refusal("unknown_frame_kind", f"Frame kind {kind!r} is not in the declared vocabulary",
                          allowed=sorted(FRAME_KINDS))
        if frame_id in self.frames:
            raise Refusal("duplicate_frame", f"Frame {frame_id!r} is already declared")
        self._bound()
        frame = Frame(frame_id, kind, description)
        self.frames[frame_id] = frame
        return frame

    def add_clock(self, clock_id: str, kind: str, resolution_s: float) -> Clock:
        text(clock_id, "clock_id", 128)
        if kind not in CLOCK_KINDS:
            raise Refusal("unknown_clock_kind", f"Clock kind {kind!r} is not declared", allowed=sorted(CLOCK_KINDS))
        if clock_id in self.clocks:
            raise Refusal("duplicate_clock", f"Clock {clock_id!r} is already declared")
        self._bound()
        clock = Clock(clock_id, kind, finite(resolution_s, "resolution_s", minimum=0.0, exclusive_minimum=True))
        self.clocks[clock_id] = clock
        return clock

    def add_clock_mapping(self, record: dict) -> ClockMapping:
        record = require_keys(record, "clock_mapping", {"mapping_id", "source", "target", "offset_s", "rate",
                                                        "covariance", "valid_from", "valid_until", "calibration"})
        source, target = self._clock(record["source"]), self._clock(record["target"])
        if (source.kind == "simulation") != (target.kind == "simulation"):
            raise Refusal("clock_domain_mismatch", "Simulation time cannot be mapped to acquisition or execution time")
        mapping_id = text(record["mapping_id"], "mapping_id", 128)
        if mapping_id in self.clock_mappings:
            raise Refusal("duplicate_mapping", f"Clock mapping {mapping_id!r} already declared")
        rate = finite(record["rate"], "rate", minimum=0.0, exclusive_minimum=True)
        valid_from, valid_until = self._interval(record, "clock_mapping")
        self._bound()
        result = ClockMapping(mapping_id, source.clock_id, target.clock_id,
                              finite(record["offset_s"], "offset_s"), rate,
                              covariance(record["covariance"], "clock_mapping.covariance", 2),
                              valid_from, valid_until, _calibration(record["calibration"], "clock_mapping.calibration"))
        self.clock_mappings[mapping_id] = result
        return result

    def add_transform(self, record: dict) -> Transform:
        record = require_keys(record, "transform", {"transform_id", "source", "target", "rotation", "translation",
                                                    "unit", "covariance", "clock", "estimated_at", "valid_from",
                                                    "valid_until", "calibration"})
        transform_id = text(record["transform_id"], "transform_id", 128)
        if transform_id in self.transforms:
            raise Refusal("duplicate_transform", f"Transform {transform_id!r} already declared")
        source, target = self._frame(record["source"]), self._frame(record["target"])
        if source.frame_id == target.frame_id:
            raise Refusal("malformed_transform", "A transform must relate two distinct frames")
        clock = self._clock(record["clock"])
        require_dimension(record["unit"], "m", "transform translation unit")
        factor = conversion_factor(record["unit"], self.length_unit)
        translation = vector(record["translation"], "transform.translation", 3) * factor
        cov = covariance(record["covariance"], "transform.covariance", 6)
        scale = np.array([1.0, 1.0, 1.0, factor, factor, factor])
        valid_from, valid_until = self._interval(record, "transform")
        estimated_at = finite(record["estimated_at"], "estimated_at")
        if estimated_at >= valid_until:
            raise Refusal("malformed_transform", "estimated_at must precede the end of validity")
        self._bound()
        result = Transform(transform_id, source.frame_id, target.frame_id, _rotation(record["rotation"], "transform.rotation"),
                           translation, scale[:, None] * cov * scale[None, :], clock.clock_id, estimated_at,
                           valid_from, valid_until, _calibration(record["calibration"], "transform.calibration"))
        self.transforms[transform_id] = result
        return result

    def _bound(self) -> None:
        if len(self.frames) + len(self.clocks) + len(self.clock_mappings) + len(self.transforms) >= MAX_ITEMS:
            raise Refusal("oversized_input", "Frame registry exceeds its declaration bound")

    @staticmethod
    def _interval(record: dict, name: str) -> tuple[float, float]:
        start = finite(record["valid_from"], name + ".valid_from")
        end = finite(record["valid_until"], name + ".valid_until")
        if not start < end:
            raise Refusal("malformed_interval", f"{name} validity must be a nonempty half-open interval")
        return start, end

    def _frame(self, frame_id: Any) -> Frame:
        if frame_id not in self.frames:
            raise Refusal("unknown_frame", f"Frame {frame_id!r} is not declared")
        return self.frames[frame_id]

    def _clock(self, clock_id: Any) -> Clock:
        if clock_id is None:
            raise Refusal("clock_unspecified", "A timestamp must name its clock")
        if clock_id not in self.clocks:
            raise Refusal("unknown_clock", f"Clock {clock_id!r} is not declared")
        return self.clocks[clock_id]

    # ------------------------------------------------------------------ clocks
    def align_time(self, time_s: float, source: str, target: str) -> tuple[float, float, list[str]]:
        """Map a timestamp between clocks, returning value, variance (s^2) and mappings used."""
        time_s = finite(time_s, "time_s")
        start, goal = self._clock(source), self._clock(target)
        variance = start.resolution_s ** 2 / 12.0
        if start.clock_id == goal.clock_id:
            return time_s, variance, []
        queue: deque[tuple[str, float, float, list[str]]] = deque([(start.clock_id, time_s, variance, [])])
        seen = {start.clock_id}
        refusals = []
        while queue:
            clock, value, var, used = queue.popleft()
            for item in self.clock_mappings.values():
                for forward in (True, False):
                    here, there = (item.source, item.target) if forward else (item.target, item.source)
                    if here != clock or there in seen:
                        continue
                    source_time = value if forward else (value - item.offset_s) / item.rate
                    if not item.valid_from <= source_time < item.valid_until:
                        refusals.append(item.mapping_id)
                        continue
                    if forward:
                        mapped = item.offset_s + item.rate * value
                        jacobian = np.array([1.0, value])
                        new_var = item.rate ** 2 * var + jacobian @ item.covariance @ jacobian
                    else:
                        mapped = source_time
                        jacobian = np.array([-1.0 / item.rate, -(value - item.offset_s) / item.rate ** 2])
                        new_var = var / item.rate ** 2 + jacobian @ item.covariance @ jacobian
                    new_var += self.clocks[there].resolution_s ** 2 / 12.0
                    if there == goal.clock_id:
                        return mapped, float(new_var), used + [item.mapping_id]
                    seen.add(there)
                    queue.append((there, mapped, float(new_var), used + [item.mapping_id]))
        if refusals:
            raise Refusal("clock_mapping_stale", f"No valid clock mapping from {source!r} to {target!r} at {time_s}",
                          expired_mappings=sorted(set(refusals)))
        raise Refusal("clock_unmapped", f"No declared mapping relates clock {source!r} to {target!r}")

    # ------------------------------------------------------------------ transforms
    def _valid_edges(self, at: float, clock: str) -> tuple[list[tuple[str, str, Transform, bool]], list[dict]]:
        edges, stale = [], []
        for item in self.transforms.values():
            try:
                local, _, _ = self.align_time(at, clock, item.clock)
            except Refusal as exc:
                stale.append({"transform_id": item.transform_id, "reason": exc.code})
                continue
            if not item.valid_from <= local < item.valid_until:
                stale.append({"transform_id": item.transform_id, "reason": "outside_validity",
                              "valid_from": item.valid_from, "valid_until": item.valid_until, "query": local})
                continue
            edges.append((item.source, item.target, item, True))
            edges.append((item.target, item.source, item, False))
        return edges, stale

    def resolve(self, source: str, target: str, at: float, clock: str, via: list[str] | None = None) -> Resolved:
        """Compose the unique shortest valid chain from ``source`` to ``target`` at a clock time."""
        self._frame(source), self._frame(target), self._clock(clock)
        finite(at, "at")
        if via:
            waypoints = [source, *via, target]
            result = self._identity(source)
            for first, second in zip(waypoints, waypoints[1:]):
                result = self._compose(self.resolve(first, second, at, clock), result)
            return result
        if source == target:
            return self._identity(source)
        edges, stale = self._valid_edges(at, clock)
        adjacency: dict[str, list[tuple[str, Transform, bool]]] = {}
        for here, there, item, forward in edges:
            adjacency.setdefault(here, []).append((there, item, forward))
        distance, count, parent = {source: 0}, {source: 1}, {}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for there, item, forward in adjacency.get(node, []):
                if there not in distance:
                    distance[there], count[there], parent[there] = distance[node] + 1, count[node], (node, item, forward)
                    queue.append(there)
                elif distance[there] == distance[node] + 1:
                    count[there] += count[node]
        if target not in distance:
            code = "transform_stale" if stale else "frame_unreachable"
            raise Refusal(code, f"No valid transform chain from {source!r} to {target!r} at {at} on {clock!r}",
                          stale=stale)
        if count[target] > 1:
            raise Refusal("frame_ambiguous", f"{count[target]} distinct shortest chains relate {source!r} and "
                          f"{target!r}; name the intended chain with via", paths=count[target])
        steps = []
        node = target
        while node != source:
            previous, item, forward = parent[node]
            steps.append((item, forward))
            node = previous
        result = self._identity(source)
        for item, forward in reversed(steps):
            result = self._compose(self._edge(item, forward), result)
        return result

    def _identity(self, frame: str) -> Resolved:
        return Resolved(frame, frame, np.eye(3), np.zeros(3), np.zeros((6, 6)), self.length_unit, [])

    def _edge(self, item: Transform, forward: bool) -> Resolved:
        link = {"transform_id": item.transform_id, "direction": "forward" if forward else "inverse",
                "calibration": item.calibration.to_json(), "clock": item.clock,
                "valid_from": item.valid_from, "valid_until": item.valid_until}
        if forward:
            return Resolved(item.source, item.target, item.rotation, item.translation, item.covariance,
                            self.length_unit, [link])
        rotation = item.rotation.T
        translation = -rotation @ item.translation
        ad = adjoint(rotation, translation)
        return Resolved(item.target, item.source, rotation, translation, ad @ item.covariance @ ad.T,
                        self.length_unit, [link])

    @staticmethod
    def _compose(outer: Resolved, inner: Resolved) -> Resolved:
        """``outer ∘ inner``: inner maps a→b, outer maps b→c."""
        if inner.target != outer.source:
            raise Refusal("frame_mismatch", f"Cannot compose {inner.source}->{inner.target} with {outer.source}->{outer.target}")
        ad = adjoint(outer.rotation, outer.translation)
        return Resolved(inner.source, outer.target, outer.rotation @ inner.rotation,
                        outer.rotation @ inner.translation + outer.translation,
                        outer.covariance + ad @ inner.covariance @ ad.T, outer.unit, inner.chain + outer.chain)

    def transform_point(self, point: Any, unit: str, source: str, target: str, at: float, clock: str,
                        point_covariance: Any = None, via: list[str] | None = None) -> dict:
        require_dimension(unit, "m", "point unit")
        factor = conversion_factor(unit, self.length_unit)
        p = vector(point, "point", 3) * factor
        cov = None if point_covariance is None else covariance(point_covariance, "point_covariance", 3) * factor ** 2
        resolved = self.resolve(source, target, at, clock, via)
        mapped, mapped_cov = resolved.apply(p, cov)
        back = conversion_factor(self.length_unit, unit)
        return plain({"point": mapped * back, "unit": unit, "frame": target, "covariance": mapped_cov * back ** 2,
                      "source_frame": source, "at": at, "clock": clock, "chain": resolved.chain})

    # ------------------------------------------------------------------ persistence
    @classmethod
    def from_json(cls, record: dict) -> "FrameRegistry":
        record = require_keys(record, "frame registry", {"schema", "length_unit", "frames", "clocks"},
                              {"clock_mappings", "transforms", "description"})
        if record["schema"] != SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {SCHEMA}")
        registry = cls(record["length_unit"])
        for item in _items(record["frames"], "frames"):
            require_keys(item, "frame", {"frame_id", "kind"}, {"description"})
            registry.add_frame(item["frame_id"], item["kind"], item.get("description", ""))
        for item in _items(record["clocks"], "clocks"):
            require_keys(item, "clock", {"clock_id", "kind", "resolution_s"})
            registry.add_clock(item["clock_id"], item["kind"], item["resolution_s"])
        for item in _items(record.get("clock_mappings", []), "clock_mappings"):
            registry.add_clock_mapping(item)
        for item in _items(record.get("transforms", []), "transforms"):
            registry.add_transform(item)
        registry._source = record
        return registry

    def to_json(self) -> dict:
        if self._source is not None:
            return self._source
        return plain({
            "schema": SCHEMA, "length_unit": self.length_unit,
            "frames": [{"frame_id": f.frame_id, "kind": f.kind, "description": f.description} for f in self.frames.values()],
            "clocks": [{"clock_id": c.clock_id, "kind": c.kind, "resolution_s": c.resolution_s} for c in self.clocks.values()],
            "clock_mappings": [{"mapping_id": m.mapping_id, "source": m.source, "target": m.target, "offset_s": m.offset_s,
                                "rate": m.rate, "covariance": m.covariance, "valid_from": m.valid_from,
                                "valid_until": m.valid_until, "calibration": m.calibration.to_json()}
                               for m in self.clock_mappings.values()],
            "transforms": [{"transform_id": t.transform_id, "source": t.source, "target": t.target,
                            "rotation": {"matrix": t.rotation}, "translation": t.translation, "unit": self.length_unit,
                            "covariance": t.covariance, "clock": t.clock, "estimated_at": t.estimated_at,
                            "valid_from": t.valid_from, "valid_until": t.valid_until,
                            "calibration": t.calibration.to_json()} for t in self.transforms.values()],
        })

    def identity(self) -> str:
        return content_identity(self.to_json())


def _items(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise Refusal("malformed_record", f"{name} must be an array")
    if len(value) > MAX_ITEMS:
        raise Refusal("oversized_input", f"{name} exceeds {MAX_ITEMS} entries")
    return value


__all__ = ["FrameRegistry", "Resolved", "Transform", "ClockMapping", "Calibration", "skew", "adjoint",
           "rotation_vector", "rotation_from_quaternion", "SCHEMA", "canonical_json"]
