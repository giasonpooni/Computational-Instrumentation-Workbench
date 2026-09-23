"""Observation-only hardware telemetry boundary for FPGA and device streams (CIWT v1).

Scope. The first hardware capability is observation: decoding CIWT packets,
timestamping them on a declared device clock (optionally aligned to another
clock through a ``FrameRegistry``), buffering with explicit drop accounting,
detecting loss, duplication, reordering, corruption and bitstream changes,
retaining the exact bytes in the ledger and replaying them to a digest.

CIWT v1 packet, little-endian::

    magic "CIWT" | version u8 = 1 | flags u8 | frame_id u16 | sequence u32 |
    timestamp_ticks u64 | bitstream_id 32 bytes (SHA-256 of the bitstream) |
    payload_len u16 | payload | crc32 u32 (zlib, over every preceding byte)

Payloads decode through a declared ``ciw.telemetry-layouts.v1`` registry into
``raw * scale`` in a parsed unit. ``sequence`` is one device-wide counter read
with serial-number arithmetic (RFC 1982), so u32 wraparound is not loss. Flag
bit 0x80 marks synthetic packets; such a stream is never labelled physical.

Limits. No control path is bound: ``request_control`` always refuses and this
module never opens, configures or writes to a device; callers hand in bytes.
CRC-32 detects accidental corruption, not tampering. The bitstream identity is
what packets *report*, not an attestation of what the FPGA runs. A late packet
and a device counter reset are told apart only by the half-range rule, u64 tick
counters are assumed not to wrap, and garbage containing ``CIWT`` may be counted
as a CRC error. A corrupt packet's own sequence is unknown, so it shows as loss.
Times are float64 seconds, so an aligned UTC epoch near 1.8e9 s resolves ~0.24 us.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any
import zlib

import numpy as np

from ._common import Refusal, canonical_json, content_identity, finite, integer, mapping, plain, require_keys, text
from .frames import FrameRegistry
from .ledger import IDENTITY, Ledger, blob_identity
from .units import parse_unit

MAGIC = b"CIWT"
VERSION = 1
HEADER = struct.Struct("<4sBBHIQ32sH")
TRAILER = struct.Struct("<I")
MIN_PACKET = HEADER.size + TRAILER.size
FLAG_SYNTHETIC = 0x80
FORMATS = frozenset("bBhHiIqQfd")
LAYOUTS_SCHEMA = "ciw.telemetry-layouts.v1"
CAPTURE_SCHEMA = "ciw.hardware-capture.v1"
SUMMARY_SCHEMA = "ciw.hardware-capture-summary.v1"
REPLAY_SCHEMA = "ciw.hardware-replay.v1"
CONTROL_SCHEMA = "ciw.hardware-control-refusal.v1"
CAPTURE_BODY_SCHEMA = "ciw.science.hardware-capture.v1"
MAX_INPUT_BYTES = 64 << 20
MAX_PACKETS = 1 << 20
MAX_WORK_FACTOR = 4  # CRC bytes examined per input byte before resynchronization is refused
MAX_FRAMES = 1024
MAX_FIELDS = 256
MAX_REPORTED = 256
SERIAL, HALF = 1 << 32, 1 << 31
SYNTHETIC_BITSTREAM = hashlib.sha256(b"ciw synthetic bitstream v1").digest()
CONTROL_PREREQUISITES = ("explicit_authority", "rollback_plan", "compatibility_check", "watchdog", "fresh_state",
                         "operator_approval", "emergency_disable")
# This build binds no device control path; request_control consults this and refuses.
CONTROL_PATHS: tuple[str, ...] = ()
INADMISSIBLE = ("bitstream_mismatch", "bitstream_changed", "synthetic_flag_mixed")
_FRAME_KEY = re.compile(r"0|[1-9][0-9]{0,4}")


# ---------------------------------------------------------------------- layouts
@dataclass(frozen=True)
class FieldLayout:
    name: str
    format: str
    unit: str
    scale: float


@dataclass(frozen=True)
class FrameLayout:
    frame_id: int
    name: str
    fields: tuple[FieldLayout, ...]
    codec: struct.Struct

    @property
    def size(self) -> int:
        return self.codec.size


class TelemetryLayouts:
    """A validated ``ciw.telemetry-layouts.v1`` registry: frame_id -> declared payload layout."""

    def __init__(self, frames: dict[int, FrameLayout], source: dict):
        self.frames = frames
        self._source = source

    @classmethod
    def from_json(cls, record: Any) -> "TelemetryLayouts":
        record = require_keys(record, "telemetry layouts", {"schema", "frames"}, {"description"})
        if record["schema"] != LAYOUTS_SCHEMA:
            raise Refusal("unsupported_schema", f"Expected {LAYOUTS_SCHEMA}")
        if "description" in record:
            text(record["description"], "description")
        declared = mapping(record["frames"], "frames")
        if not declared:
            raise Refusal("malformed_record", "frames must declare at least one frame layout")
        if len(declared) > MAX_FRAMES:
            raise Refusal("oversized_input", f"frames exceeds {MAX_FRAMES} layouts")
        frames: dict[int, FrameLayout] = {}
        for key, item in declared.items():
            if not isinstance(key, str) or _FRAME_KEY.fullmatch(key) is None or int(key) > 0xFFFF:
                raise Refusal("malformed_record", f"frame key {key!r} must be a decimal u16 without leading zeros")
            frame = _frame_layout(int(key), item)
            if any(other.name == frame.name for other in frames.values()):
                raise Refusal("duplicate_frame", f"Frame name {frame.name!r} is declared twice")
            frames[frame.frame_id] = frame
        return cls(frames, json.loads(canonical_json(plain(record))))

    def to_json(self) -> dict:
        return json.loads(canonical_json(self._source))

    def identity(self) -> str:
        return content_identity(self._source)

    def frame(self, frame_id: Any) -> FrameLayout:
        if frame_id not in self.frames:
            raise Refusal("unknown_frame", f"Frame id {frame_id!r} has no declared layout")
        return self.frames[frame_id]

    def pack(self, frame_id: int, raw: list | tuple) -> bytes:
        """Pack raw (unscaled) field values into a payload for ``frame_id``."""
        frame = self.frame(frame_id)
        if not isinstance(raw, (list, tuple)) or len(raw) != len(frame.fields):
            raise Refusal("malformed_record", f"Frame {frame.name!r} needs {len(frame.fields)} raw values")
        try:
            return frame.codec.pack(*raw)
        except struct.error as exc:
            raise Refusal("out_of_domain", f"Raw values do not fit frame {frame.name!r}: {exc}") from exc


def _frame_layout(frame_id: int, item: Any) -> FrameLayout:
    where = f"frames[{frame_id}]"
    item = require_keys(item, where, {"name", "fields"}, {"description"})
    declared = item["fields"]
    if not isinstance(declared, list) or not declared:
        raise Refusal("malformed_record", f"{where}.fields must be a nonempty array")
    if len(declared) > MAX_FIELDS:
        raise Refusal("oversized_input", f"{where}.fields exceeds {MAX_FIELDS} fields")
    fields: list[FieldLayout] = []
    for index, spec in enumerate(declared):
        name = f"{where}.fields[{index}]"
        spec = require_keys(spec, name, {"name", "format", "unit", "scale"}, {"description"})
        field_name = text(spec["name"], name + ".name", 128)
        if any(other.name == field_name for other in fields):
            raise Refusal("duplicate_field", f"{where} declares field {field_name!r} twice")
        code = spec["format"]
        if not isinstance(code, str) or code not in FORMATS:
            raise Refusal("unknown_format", f"{name}.format must be one struct code", allowed=sorted(FORMATS))
        parse_unit(spec["unit"])
        scale = finite(spec["scale"], name + ".scale")
        if scale == 0.0:
            raise Refusal("out_of_domain", f"{name}.scale must be nonzero")
        fields.append(FieldLayout(field_name, code, spec["unit"], scale))
    codec = struct.Struct("<" + "".join(spec.format for spec in fields))
    if codec.size > 0xFFFF:
        raise Refusal("oversized_input", f"{where} payload exceeds the u16 payload length")
    return FrameLayout(frame_id, text(item["name"], where + ".name", 128), tuple(fields), codec)


def validate_layouts(record: Any) -> TelemetryLayouts:
    """Validate a ``ciw.telemetry-layouts.v1`` record (units parsed, formats and scales checked)."""
    return TelemetryLayouts.from_json(record)


def _layouts(value: Any) -> TelemetryLayouts:
    return value if isinstance(value, TelemetryLayouts) else TelemetryLayouts.from_json(value)


# ---------------------------------------------------------------------- encoding
def _bitstream(value: Any, name: str) -> bytes:
    if isinstance(value, (bytes, bytearray)) and len(value) == 32:
        return bytes(value)
    if isinstance(value, str) and IDENTITY.fullmatch(value):
        return bytes.fromhex(value[7:])
    raise Refusal("malformed_identity", f"{name} must be 32 raw bytes or sha256:<64 lowercase hex>")


def bitstream_identity(image: bytes) -> str:
    """The identity a CIWT packet should carry for a bitstream image: its SHA-256."""
    if not isinstance(image, (bytes, bytearray)):
        raise Refusal("malformed_record", "A bitstream image must be bytes")
    return blob_identity(bytes(image))


def encode_packet(frame_id: int, sequence: int, ticks: int, bitstream: bytes | str, payload: bytes, *,
                  flags: int = 0, version: int = VERSION) -> bytes:
    """Encode one CIWT packet (for synthetic streams and tests; nothing is sent anywhere)."""
    if not isinstance(payload, (bytes, bytearray)) or len(payload) > 0xFFFF:
        raise Refusal("malformed_record", "payload must be at most 65535 bytes")
    head = HEADER.pack(MAGIC, integer(version, "version", minimum=0, maximum=0xFF),
                       integer(flags, "flags", minimum=0, maximum=0xFF),
                       integer(frame_id, "frame_id", minimum=0, maximum=0xFFFF),
                       integer(sequence, "sequence", minimum=0, maximum=SERIAL - 1),
                       integer(ticks, "ticks", minimum=0, maximum=(1 << 64) - 1),
                       _bitstream(bitstream, "bitstream"), len(payload)) + bytes(payload)
    return head + TRAILER.pack(zlib.crc32(head))


def _draw(rng: np.random.Generator, code: str) -> int | float:
    if code in "fd":
        return float(rng.normal(0.0, 1.0))
    bits = 8 * struct.calcsize(code)
    signed = code.islower()
    bound = min(1000, (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1)
    return int(rng.integers(-bound if signed else 0, bound + 1))


def synthetic_packets(layouts: Any, *, seed: int, count: int, frame_ids: list[int] | None = None,
                      start_sequence: int = 0, start_ticks: int = 0, ticks_per_packet: int = 1000,
                      bitstream: bytes | str = SYNTHETIC_BITSTREAM) -> list[bytes]:
    """Deterministic synthetic packets (seeded numpy); every packet carries ``FLAG_SYNTHETIC``."""
    table = _layouts(layouts)
    ids = sorted(table.frames) if frame_ids is None else [table.frame(item).frame_id for item in frame_ids]
    if not ids:
        raise Refusal("malformed_record", "frame_ids must name at least one frame")
    count = integer(count, "count", minimum=0, maximum=MAX_PACKETS)
    start_sequence = integer(start_sequence, "start_sequence", minimum=0, maximum=SERIAL - 1)
    start_ticks = integer(start_ticks, "start_ticks", minimum=0)
    ticks_per_packet = integer(ticks_per_packet, "ticks_per_packet", minimum=0)
    rng = np.random.default_rng(integer(seed, "seed", minimum=0))
    packets = []
    for index in range(count):
        frame = table.frames[ids[index % len(ids)]]
        payload = frame.codec.pack(*[_draw(rng, spec.format) for spec in frame.fields])
        packets.append(encode_packet(frame.frame_id, (start_sequence + index) % SERIAL,
                                     start_ticks + index * ticks_per_packet, bitstream, payload, flags=FLAG_SYNTHETIC))
    return packets


def synthetic_stream(layouts: Any, **options: Any) -> bytes:
    """``synthetic_packets`` concatenated into one stream."""
    return b"".join(synthetic_packets(layouts, **options))


# ---------------------------------------------------------------------- parsing
class _Serial:
    """Serial-number (RFC 1982) accounting over one u32 counter, in unwrapped positions."""

    def __init__(self) -> None:
        self.base: int | None = None
        self.high = 0
        self.seen: dict[int, tuple[int, int]] = {}
        self.gaps: list[tuple[int, int]] = []
        self.filled: list[int] = []
        self.duplicates = self.reordered = self.wraps = 0

    def wire(self, position: int) -> int:
        return (self.base + position) % SERIAL

    def observe(self, sequence: int, span: tuple[int, int], raw: bytes, note) -> str:
        if sequence in self.seen:
            first = self.seen[sequence]
            self.duplicates += 1
            note("duplicate", offset=span[0], sequence=sequence, first_offset=first[0],
                 identical=raw[first[0]:first[1]] == raw[span[0]:span[1]])
            return "duplicate"
        self.seen[sequence] = span
        if self.base is None:
            self.base = sequence
            return "new"
        highest = self.wire(self.high)
        delta = (sequence - highest) % SERIAL
        if delta < HALF:
            if delta > 1:
                self.gaps.append((self.high + 1, self.high + delta - 1))
                note("sequence_gap", offset=span[0], first=(highest + 1) % SERIAL, last=(sequence - 1) % SERIAL,
                     lost=delta - 1)
            self.wraps += sequence < highest
            self.high += delta
            return "new"
        position = self.high - (highest - sequence) % SERIAL
        if position > 0:
            self.filled.append(position)
        self.reordered += 1
        note("reordered", offset=span[0], sequence=sequence, highest=highest, fills_gap=position > 0)
        return "late"

    def missing(self) -> list[tuple[int, int]]:
        filled, result = sorted(self.filled), []
        for first, last in self.gaps:
            start = first
            for position in filled[bisect_left(filled, first):bisect_right(filled, last)]:
                if position > start:
                    result.append((start, position - 1))
                start = position + 1
            if start <= last:
                result.append((start, last))
        return result


@dataclass(frozen=True)
class Capture:
    """Decoded records, anomalies in arrival order, and a ledger-sized summary with the digest."""

    records: tuple[dict, ...]
    anomalies: tuple[dict, ...]
    summary: dict

    @property
    def digest(self) -> str:
        return self.summary["digest"]

    @property
    def admissible(self) -> bool:
        return self.summary["admissible"]

    def to_json(self) -> dict:
        return {"schema": CAPTURE_SCHEMA, "records": list(self.records), "anomalies": list(self.anomalies),
                "summary": self.summary}


def _input(data: Any) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise Refusal("malformed_record", "Telemetry input must be bytes")
    if len(data) > MAX_INPUT_BYTES:
        raise Refusal("oversized_input", f"Telemetry input exceeds {MAX_INPUT_BYTES} bytes", size=len(data))
    return bytes(data)


def parse_stream(data: bytes, layouts: Any, *, tick_rate_hz: float, device_clock: str,
                 expected_bitstream: bytes | str | None = None, frames: FrameRegistry | None = None,
                 target_clock: str | None = None, acquisition: str | None = None) -> Capture:
    """Decode a CIWT byte stream into records and anomalies; nothing is repaired or guessed.

    ``acquisition`` may declare "physical" or "synthetic"; undeclared streams are
    labelled "undeclared", and any synthetic-flagged packet labels the capture synthetic.
    """
    raw, table = _input(data), _layouts(layouts)
    rate = finite(tick_rate_hz, "tick_rate_hz", minimum=0.0, exclusive_minimum=True)
    clock = text(device_clock, "device_clock", 128)
    expected = None if expected_bitstream is None else _bitstream(expected_bitstream, "expected_bitstream")
    if acquisition is not None and acquisition not in ("physical", "synthetic"):
        raise Refusal("malformed_record", "acquisition must be 'physical', 'synthetic' or omitted")
    if (frames is None) != (target_clock is None):
        raise Refusal("malformed_record", "frames and target_clock must be given together")
    if frames is not None:
        if not isinstance(frames, FrameRegistry):
            raise Refusal("malformed_record", "frames must be a FrameRegistry")
        for name in (clock, target_clock):
            if name not in frames.clocks:
                raise Refusal("unknown_clock", f"Clock {name!r} is not declared in the frame registry")

    records: list[dict] = []
    anomalies: list[dict] = []
    accounting = {"packets": 0, "corrupt": 0, "skipped": 0, "truncated": 0}
    counts = {"valid": 0, "decoded": 0, "crc_errors": 0, "unknown_frame": 0, "length_mismatch": 0, "synthetic": 0}
    per_frame = {frame.name: 0 for frame in table.frames.values()}
    bitstreams: dict[str, int] = {}
    serial, view = _Serial(), memoryview(raw)
    current = last_ticks = None
    position, size, attempts, work = 0, len(raw), 0, 0

    def note(code: str, **detail: Any) -> None:
        anomalies.append({"code": code, **detail})

    def skip(start: int, search: int, code: str, bucket: str = "skipped", **detail: Any) -> int:
        found = raw.find(MAGIC, search)
        end = size if found < 0 else found
        accounting[bucket] += end - start
        note(code, offset=start, skipped=end - start, **detail)
        return end

    while position < size:
        if not raw.startswith(MAGIC, position):
            position = skip(position, position, "bad_magic")
            continue
        attempts += 1
        if attempts > MAX_PACKETS:
            raise Refusal("oversized_input", f"Telemetry input exceeds {MAX_PACKETS} packets")
        if size - position < MIN_PACKET:
            accounting["truncated"] += size - position
            note("truncated", offset=position, available=size - position)
            break
        _, version, flags, frame_id, sequence, ticks, bitstream, length = HEADER.unpack_from(raw, position)
        if version != VERSION:
            position = skip(position, position + 1, "unsupported_version", version=version)
            continue
        end = position + HEADER.size + length + TRAILER.size
        if end > size:
            if raw.find(MAGIC, position + 1) < 0:
                accounting["truncated"] += size - position
                note("truncated", offset=position, declared=end - position, available=size - position)
                break
            position = skip(position, position + 1, "length_overrun", declared=end - position)
            continue
        work += end - position
        if work > MAX_WORK_FACTOR * size:
            raise Refusal("oversized_input", "Resynchronization work exceeds its bound; the input is mostly corrupt")
        if zlib.crc32(view[position:end - TRAILER.size]) != TRAILER.unpack_from(raw, end - TRAILER.size)[0]:
            counts["crc_errors"] += 1
            if end == size or raw.startswith(MAGIC, end):
                accounting["corrupt"] += end - position
                note("crc_error", offset=position, skipped=end - position, resynchronized=False)
                position = end
            else:
                position = skip(position, position + 1, "crc_error", "corrupt", resynchronized=True)
            continue

        # A CRC-valid packet: account for it before deciding whether it decodes.
        start, position = position, end
        accounting["packets"] += end - start
        counts["valid"] += 1
        counts["synthetic"] += bool(flags & FLAG_SYNTHETIC)
        identity = "sha256:" + bitstream.hex()
        if current is not None and identity != current:
            note("bitstream_changed", offset=start, sequence=sequence, previous=current, current=identity)
        current = identity
        if expected is not None and bitstream != expected and identity not in bitstreams:
            note("bitstream_mismatch", offset=start, sequence=sequence, expected="sha256:" + expected.hex(),
                 found=identity)
        bitstreams[identity] = bitstreams.get(identity, 0) + 1
        status = serial.observe(sequence, (start, end), raw, note)
        if status == "duplicate":
            continue
        if status == "new":
            if last_ticks is not None and ticks < last_ticks:
                note("non_monotonic_timestamp", offset=start, sequence=sequence, ticks=ticks, previous_ticks=last_ticks)
            last_ticks = ticks
        layout = table.frames.get(frame_id)
        if layout is None:
            counts["unknown_frame"] += 1
            note("unknown_frame", offset=start, sequence=sequence, frame_id=frame_id)
            continue
        if length != layout.size:
            counts["length_mismatch"] += 1
            note("length_mismatch", offset=start, sequence=sequence, frame_id=frame_id, expected=layout.size,
                 declared=length)
            continue
        values = {}
        for spec, number in zip(layout.fields, layout.codec.unpack_from(raw, start + HEADER.size)):
            value = number * spec.scale if math.isfinite(number) else math.nan
            if not math.isfinite(value):
                note("non_finite_value", offset=start, sequence=sequence, field=spec.name)
            values[spec.name] = {"raw": number if math.isfinite(number) else None,
                                 "value": value if math.isfinite(value) else None, "unit": spec.unit}
        time_s = ticks / rate
        record = {"offset": start, "frame_id": frame_id, "frame": layout.name, "sequence": sequence, "flags": flags,
                  "ticks": ticks, "time_s": time_s, "clock": clock, "bitstream": identity, "late": status == "late",
                  "values": values}
        if frames is not None:
            try:
                aligned, variance, used = frames.align_time(time_s, clock, target_clock)
                record["aligned"] = {"clock": target_clock, "time_s": float(aligned), "variance_s2": float(variance),
                                     "mappings": list(used)}
            except Refusal as exc:
                note("clock_unaligned", offset=start, sequence=sequence, reason=exc.code)
                record["aligned"] = None
        records.append(record)
        counts["decoded"] += 1
        per_frame[layout.name] += 1

    synthetic = counts.pop("synthetic")
    if synthetic and acquisition == "physical":
        raise Refusal("acquisition_conflict", "Packets carry the synthetic flag; the stream cannot be declared physical",
                      synthetic_packets=synthetic)
    if 0 < synthetic < counts["valid"]:
        note("synthetic_flag_mixed", synthetic=synthetic, valid=counts["valid"])
    label = "synthetic" if synthetic or acquisition == "synthetic" else (acquisition or "undeclared")
    missing = serial.missing()
    reasons = [code for code in INADMISSIBLE if any(item["code"] == code for item in anomalies)]
    if not counts["valid"]:
        reasons.append("no_packets")
    codes: dict[str, int] = {}
    for item in anomalies:
        codes[item["code"]] = codes.get(item["code"], 0) + 1
    span = None
    if records:
        times = [record["time_s"] for record in records]
        span = {"clock": clock, "start_s": min(times), "end_s": max(times), "duration_s": max(times) - min(times),
                "first_ticks": records[0]["ticks"], "last_ticks": records[-1]["ticks"]}
        aligned = [record["aligned"]["time_s"] for record in records if record.get("aligned")]
        if aligned:
            span["aligned"] = {"clock": target_clock, "start_s": min(aligned), "end_s": max(aligned)}
    digest = content_identity({"schema": CAPTURE_SCHEMA, "records": records, "anomalies": anomalies})
    summary = {
        "schema": SUMMARY_SCHEMA, "acquisition": label, "admissible": not reasons, "inadmissible_reasons": reasons,
        "digest": digest, "input_identity": blob_identity(raw), "input_bytes": size, "bytes": accounting,
        "packets_valid": counts["valid"], "packets_decoded": counts["decoded"], "packets_per_frame": per_frame,
        "crc_errors": counts["crc_errors"], "unknown_frame": counts["unknown_frame"],
        "length_mismatch": counts["length_mismatch"], "lost": sum(last - first + 1 for first, last in missing),
        "gaps": [[serial.wire(first), serial.wire(last)] for first, last in missing[:MAX_REPORTED]],
        "gaps_total": len(missing), "duplicates": serial.duplicates, "reordered": serial.reordered,
        "sequence_wraps": serial.wraps, "anomalies": codes,
        "bitstreams": [{"bitstream": key, "packets": value} for key, value in list(bitstreams.items())[:MAX_REPORTED]],
        "bitstreams_total": len(bitstreams),
        "bitstream_expected": None if expected is None else "sha256:" + expected.hex(),
        "time_span": span,
        "parameters": {"tick_rate_hz": rate, "device_clock": clock, "target_clock": target_clock,
                       "expected_bitstream": None if expected is None else "sha256:" + expected.hex(),
                       "acquisition_declared": acquisition, "layouts_identity": table.identity(),
                       "frame_registry_identity": None if frames is None else frames.identity()},
    }
    return Capture(tuple(records), tuple(anomalies), summary)


# ---------------------------------------------------------------------- replay
def replay(raw_bytes: bytes, layouts: Any, *, retained_digest: str, **options: Any) -> dict:
    """Re-parse retained bytes and compare the capture digest with the retained one."""
    if not isinstance(retained_digest, str) or IDENTITY.fullmatch(retained_digest) is None:
        raise Refusal("malformed_identity", "retained_digest must be sha256:<64 lowercase hex>")
    capture = parse_stream(raw_bytes, layouts, **options)
    return {"schema": REPLAY_SCHEMA, "status": "reproduced" if capture.digest == retained_digest else "diverged",
            "retained_digest": retained_digest, "replayed_digest": capture.digest,
            "input_identity": capture.summary["input_identity"], "records": len(capture.records),
            "anomalies": len(capture.anomalies)}


def replay_retained(ledger: Ledger, entry_id: str, layouts: Any, *, frames: FrameRegistry | None = None,
                    record: bool = False) -> dict:
    """Replay a retained ``hardware-capture`` entry from its exact blob; optionally append a replay receipt."""
    entry = ledger.get(entry_id)
    if entry["body_schema"] != CAPTURE_BODY_SCHEMA:
        raise Refusal("not_a_capture", f"Entry {entry_id} is not a {CAPTURE_BODY_SCHEMA} entry")
    summary = mapping(entry["body"]["summary"], "capture summary")
    parameters = require_keys(summary.get("parameters"), "capture parameters",
                              {"tick_rate_hz", "device_clock", "target_clock", "expected_bitstream",
                               "acquisition_declared", "layouts_identity", "frame_registry_identity"})
    table = _layouts(layouts)
    if table.identity() != parameters["layouts_identity"]:
        raise Refusal("layouts_mismatch", "Replay needs the layouts the capture was decoded with",
                      retained=parameters["layouts_identity"], given=table.identity())
    registry = parameters["frame_registry_identity"]
    if registry is not None and (not isinstance(frames, FrameRegistry) or frames.identity() != registry):
        raise Refusal("frame_registry_mismatch", "Replay needs the frame registry the capture was aligned with",
                      retained=registry)
    result = replay(ledger.get_blob(entry["body"]["blob"]), table, retained_digest=summary["digest"],
                    tick_rate_hz=parameters["tick_rate_hz"], device_clock=parameters["device_clock"],
                    expected_bitstream=parameters["expected_bitstream"],
                    frames=frames if registry is not None else None, target_clock=parameters["target_clock"],
                    acquisition=parameters["acquisition_declared"])
    result["original"] = entry_id
    if record:
        receipt = ledger.append("ciw.science.replay-receipt.v1",
                                {"original": entry_id, "status": result["status"], "comparison": dict(result)},
                                refs=[entry_id])
        result["receipt"] = receipt["entry_id"]
    return result


# ---------------------------------------------------------------------- buffering and retention
class CaptureBuffer:
    """Bounded FIFO ring: overflow drops the oldest item and records the drop, never silently.

    Every pushed item gets an arrival index; drops are kept as merged
    ``[first, last]`` arrival-index ranges so ``pushed == drained + dropped + size``.
    """

    def __init__(self, capacity: int):
        self.capacity = integer(capacity, "capacity", minimum=1, maximum=MAX_PACKETS)
        self._items: deque[tuple[int, Any]] = deque()
        self._drops: list[list[int]] = []
        self.pushed = self.drained = self.dropped = self.high_water = 0

    def push(self, item: Any) -> Any:
        """Append an item; return the dropped oldest item when the buffer was full, else None."""
        dropped = None
        if len(self._items) == self.capacity:
            index, dropped = self._items.popleft()
            self.dropped += 1
            if self._drops and self._drops[-1][1] == index - 1:
                self._drops[-1][1] = index
            else:
                self._drops.append([index, index])
        self._items.append((self.pushed, item))
        self.pushed += 1
        self.high_water = max(self.high_water, len(self._items))
        return dropped

    def extend(self, items: Any) -> int:
        before = self.dropped
        for item in items:
            self.push(item)
        return self.dropped - before

    def drain(self, limit: int | None = None) -> list:
        count = len(self._items) if limit is None else min(integer(limit, "limit", minimum=0), len(self._items))
        self.drained += count
        return [self._items.popleft()[1] for _ in range(count)]

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict:
        return {"capacity": self.capacity, "size": len(self._items), "pushed": self.pushed, "drained": self.drained,
                "dropped": self.dropped, "high_water": self.high_water,
                "dropped_ranges": [list(item) for item in self._drops], "lossless": self.dropped == 0}


def retain_capture(ledger: Ledger, raw_bytes: bytes, capture: Capture) -> dict:
    """Retain the exact input bytes and append a ``hardware-capture`` entry with the capture summary."""
    if not isinstance(capture, Capture):
        raise Refusal("malformed_record", "capture must be a Capture from parse_stream")
    raw = _input(raw_bytes)
    if blob_identity(raw) != capture.summary["input_identity"]:
        raise Refusal("capture_input_mismatch", "The bytes are not the bytes this capture was parsed from",
                      expected=capture.summary["input_identity"], found=blob_identity(raw))
    identity = ledger.put_blob(raw)
    return ledger.append(CAPTURE_BODY_SCHEMA, {"blob": identity, "summary": capture.summary}, blobs=[identity])


# ---------------------------------------------------------------------- control (always refused)
def request_control(command: Any, context: Any) -> dict:
    """Refuse every control request, listing which prerequisites the context merely *claimed*.

    Claims are not verified, and even a context claiming all of them is refused
    with ``control_path_unbound`` because this build binds no control path.
    Nothing is written to any device.
    """
    claims = context if isinstance(context, dict) else {}
    prerequisites = [{"prerequisite": name, "claimed": bool(claims.get(name)), "verified": False}
                     for name in CONTROL_PREREQUISITES]
    unclaimed = [item["prerequisite"] for item in prerequisites if not item["claimed"]]
    try:
        command_identity = content_identity(plain(command)) if isinstance(command, dict) else None
    except (TypeError, ValueError):
        command_identity = None
    reasons = ["control_path_unbound"]
    if not isinstance(command, dict):
        reasons.append("malformed_command")
    if not isinstance(context, dict):
        reasons.append("malformed_context")
    if unclaimed:
        reasons.append("prerequisites_unclaimed")
    return {"schema": CONTROL_SCHEMA, "status": "refused", "code": "control_path_unbound",
            "message": "This build binds no hardware control path; observation is the only hardware capability",
            "reasons": reasons, "control_paths": list(CONTROL_PATHS), "device_writes": 0,
            "command_identity": command_identity, "prerequisites": prerequisites, "unclaimed": unclaimed,
            "all_claimed": not unclaimed,
            "unrecognized_context": sorted(str(key) for key in claims if key not in CONTROL_PREREQUISITES)}


__all__ = [
    "MAGIC", "VERSION", "HEADER", "FLAG_SYNTHETIC", "FORMATS", "LAYOUTS_SCHEMA", "CAPTURE_SCHEMA", "SUMMARY_SCHEMA",
    "CONTROL_PREREQUISITES", "SYNTHETIC_BITSTREAM", "FieldLayout", "FrameLayout", "TelemetryLayouts",
    "validate_layouts", "bitstream_identity", "encode_packet", "synthetic_packets", "synthetic_stream", "Capture",
    "parse_stream", "replay", "replay_retained", "CaptureBuffer", "retain_capture", "request_control",
]
