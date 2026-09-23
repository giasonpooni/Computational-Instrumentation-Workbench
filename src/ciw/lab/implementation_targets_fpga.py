"""FPGA telemetry-only frames, bitstream identity, compatibility and link simulation (T149-T152).

The frame format is read-only by construction: one frame type (telemetry),
no host-to-device field, and a decoder that refuses command, write and
unknown frame types before any payload is used. Bitstream identity records
bind a bitstream digest to its toolchain, constraints and source tree; the
bitstreams used here are seeded synthetic placeholders, never real
configurations. The link simulation is seeded and synthetic: it says how the
receiver's gap and staleness detectors behave on a declared loss/latency
model, not how any real FPGA link behaves.
"""
from __future__ import annotations

import hashlib
import re
import struct

import numpy as np

from .implementation_targets_serial import canonical_bytes, canonical_sha256

# --------------------------------------------------------------- CRC-32
_POLYNOMIAL = 0xEDB88320  # IEEE 802.3, reflected


def _table() -> tuple:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ _POLYNOMIAL if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_TABLE = _table()


def crc32(data: bytes) -> int:
    """Table-driven CRC-32 (init and final XOR 0xFFFFFFFF); check value 0xCBF43926 for b"123456789"."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ _TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


# ------------------------------------------------------ telemetry frame
MAGIC = b"CIWT"
FRAME_VERSION = 1
TELEMETRY = 0x01
HEADER = struct.Struct(">4sBBHIQIHH")  # magic, version, type, flags, sequence, timestamp_ns, clock_id, channels, payload bytes
TRAILER = struct.Struct(">I")
FLAG_OVERFLOW = 0x0001
FLAG_CLOCK_UNLOCKED = 0x0002
FLAG_WRITE_REQUEST = 0x8000  # never defined for use; its presence is a refused command path
DEFINED_FLAGS = FLAG_OVERFLOW | FLAG_CLOCK_UNLOCKED
COMMAND_TYPES = {0x80: "command", 0x81: "register_write", 0x82: "actuator_setpoint", 0x83: "bitstream_load"}
MAX_CHANNELS = 256
SEQUENCE_MODULUS = 2 ** 32

INTERFACE = {
    "schema": "ciw.fpga-telemetry-interface.v1",
    "direction": "device_to_host",
    "byte_order": "big-endian",
    "frame_types": {"0x01": "telemetry"},
    "host_to_device": [],
    "fields": [
        {"name": "magic", "offset": 0, "bytes": 4, "type": "ascii 'CIWT'", "direction": "device_to_host"},
        {"name": "version", "offset": 4, "bytes": 1, "type": "u8 = 1", "direction": "device_to_host"},
        {"name": "frame_type", "offset": 5, "bytes": 1, "type": "u8 = 0x01 telemetry", "direction": "device_to_host"},
        {"name": "flags", "offset": 6, "bytes": 2, "type": "u16; bit0 overflow, bit1 clock unlocked, others zero",
         "direction": "device_to_host"},
        {"name": "sequence", "offset": 8, "bytes": 4, "type": "u32, increments by one, wraps mod 2^32",
         "direction": "device_to_host"},
        {"name": "timestamp_ns", "offset": 12, "bytes": 8, "type": "u64 device clock nanoseconds",
         "direction": "device_to_host"},
        {"name": "clock_id", "offset": 20, "bytes": 4, "type": "u32 declared clock identity", "direction": "device_to_host"},
        {"name": "channel_count", "offset": 24, "bytes": 2, "type": "u16 <= 256", "direction": "device_to_host"},
        {"name": "payload_bytes", "offset": 26, "bytes": 2, "type": "u16 = 4 * channel_count", "direction": "device_to_host"},
        {"name": "payload", "offset": 28, "bytes": "4 * channel_count", "type": "i32 raw counts, calibration not applied",
         "direction": "device_to_host"},
        {"name": "crc32", "offset": "28 + payload_bytes", "bytes": 4, "type": "CRC-32/IEEE over all preceding bytes",
         "direction": "device_to_host"},
    ],
}
_FORBIDDEN_NAME = re.compile(r"(^|_)(command|cmd|write|actuator|actuate|setpoint|register|load|reset|control)(_|$)",
                             re.IGNORECASE)


class TelemetryRefusal(ValueError):
    """A frame or interface specification violates the telemetry-only contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_interface(spec: dict) -> dict:
    """Refuse any specification with a host-to-device path or a command-like field."""
    if spec.get("host_to_device"):
        raise TelemetryRefusal("command_path_refused", "A telemetry-only interface has no host-to-device fields")
    if spec.get("direction") != "device_to_host":
        raise TelemetryRefusal("command_path_refused", "A telemetry-only interface is device-to-host only")
    for field in spec.get("fields", []):
        if field.get("direction") != "device_to_host":
            raise TelemetryRefusal("command_path_refused", f"Field {field.get('name')} is not device-to-host")
        if _FORBIDDEN_NAME.search(str(field.get("name", ""))):
            raise TelemetryRefusal("command_path_refused", f"Field {field.get('name')} names a command or write path")
    if spec.get("frame_types") != {"0x01": "telemetry"}:
        raise TelemetryRefusal("unknown_frame_type", "A telemetry-only interface defines exactly one telemetry frame type")
    return spec


def _pack(frame_type: int, flags: int, sequence: int, timestamp_ns: int, clock_id: int, channels) -> bytes:
    channels = [int(c) for c in channels]
    body = HEADER.pack(MAGIC, FRAME_VERSION, frame_type, flags, sequence, timestamp_ns, clock_id, len(channels),
                       4 * len(channels)) + struct.pack(f">{len(channels)}i", *channels)
    return body + TRAILER.pack(crc32(body))


def encode_frame(sequence: int, timestamp_ns: int, clock_id: int, channels, flags: int = 0) -> bytes:
    """Device-side encoder (used by simulations); only telemetry frames can be produced."""
    if flags & FLAG_WRITE_REQUEST:
        raise TelemetryRefusal("command_path_refused", "Telemetry frames cannot carry a write request")
    if flags & ~DEFINED_FLAGS:
        raise TelemetryRefusal("reserved_flags", "Reserved telemetry flags must be zero")
    if not (0 <= sequence < SEQUENCE_MODULUS and 0 <= timestamp_ns < 2 ** 64 and 0 <= clock_id < 2 ** 32):
        raise TelemetryRefusal("field_out_of_range", "Telemetry header field is outside its declared range")
    if len(channels) > MAX_CHANNELS or any(not -2 ** 31 <= int(c) < 2 ** 31 for c in channels):
        raise TelemetryRefusal("field_out_of_range", "Telemetry payload is outside its declared range")
    return _pack(TELEMETRY, flags, sequence, timestamp_ns, clock_id, channels)


def forge_frame(frame_type: int, flags: int, sequence: int = 1, timestamp_ns: int = 0, clock_id: int = 7,
                channels=(1, 2)) -> bytes:
    """Negative-test generator: a well-formed frame with an arbitrary type and flags and a valid CRC."""
    return _pack(frame_type, flags, sequence, timestamp_ns, clock_id, channels)


def decode_frame(data: bytes) -> dict:
    """Host-side decoder. The CRC is checked first, then the telemetry-only contract."""
    if len(data) < HEADER.size + TRAILER.size:
        raise TelemetryRefusal("truncated_frame", "Frame is shorter than header plus CRC")
    if crc32(data[:-4]) != TRAILER.unpack(data[-4:])[0]:
        raise TelemetryRefusal("crc_mismatch", "Frame CRC does not match its bytes")
    magic, version, frame_type, flags, sequence, timestamp_ns, clock_id, count, size = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC:
        raise TelemetryRefusal("bad_magic", "Frame magic is not CIWT")
    if version != FRAME_VERSION:
        raise TelemetryRefusal("unsupported_version", f"Unsupported telemetry frame version: {version}")
    if frame_type in COMMAND_TYPES:
        raise TelemetryRefusal("command_path_refused", f"Refused {COMMAND_TYPES[frame_type]} frame: telemetry only")
    if frame_type != TELEMETRY:
        raise TelemetryRefusal("unknown_frame_type", f"Unknown frame type: {frame_type:#04x}")
    if flags & FLAG_WRITE_REQUEST:
        raise TelemetryRefusal("command_path_refused", "Refused frame with a write-request flag: telemetry only")
    if flags & ~DEFINED_FLAGS:
        raise TelemetryRefusal("reserved_flags", "Reserved telemetry flags must be zero")
    if size != 4 * count or count > MAX_CHANNELS or len(data) != HEADER.size + size + TRAILER.size:
        raise TelemetryRefusal("length_mismatch", "Frame length does not match its channel count")
    channels = list(struct.unpack(f">{count}i", data[HEADER.size:HEADER.size + size]))
    return {"sequence": sequence, "timestamp_ns": timestamp_ns, "clock_id": clock_id, "flags": flags,
            "channels": channels}


class TelemetryReceiver:
    """Host-side receiver: decodes frames, tracks sequence gaps and flags stale frames.

    It has no method that sends anything to the device. Sequence numbers are
    compared with serial-number arithmetic modulo 2^32 (a forward jump of
    2^31 or more is indistinguishable from a late frame and is not supported).
    """

    def __init__(self, clock_offset_ns: int, stale_after_ns: int):
        self.clock_offset_ns, self.stale_after_ns = int(clock_offset_ns), int(stale_after_ns)
        self.highest = None
        self.first = None
        self.pending: set = set()
        self.reordered = self.duplicates = self.accepted = 0
        self.refused: dict = {}
        self.stale: list = []
        self.ages: list = []

    def accept(self, frame: bytes, received_ns: int):
        try:
            record = decode_frame(frame)
        except TelemetryRefusal as refusal:
            self.refused[refusal.code] = self.refused.get(refusal.code, 0) + 1
            return None
        self.accepted += 1
        sequence = record["sequence"]
        if self.highest is None:
            self.highest = self.first = sequence
        else:
            delta = (sequence - self.highest + 2 ** 31) % SEQUENCE_MODULUS - 2 ** 31
            if delta > 0:
                for step in range(1, delta):
                    self.pending.add((self.highest + step) % SEQUENCE_MODULUS)
                self.highest = sequence
            elif sequence in self.pending:
                self.pending.discard(sequence)
                self.reordered += 1
            else:
                self.duplicates += 1
        age = received_ns - (record["timestamp_ns"] + self.clock_offset_ns)
        self.ages.append(age)
        if age > self.stale_after_ns:
            self.stale.append((sequence, received_ns))
        return record


def naive_gap_count(sequences) -> int:
    """Unsafe detector: sums positive (seq - previous - 1) in arrival order, without modular arithmetic."""
    lost, previous = 0, None
    for sequence in sequences:
        if previous is not None and sequence - previous - 1 > 0:
            lost += sequence - previous - 1
        previous = sequence
    return lost


def simulate_link(seed: int = 152, frames: int = 4000, period_ns: int = 1_000_000,
                  start_sequence: int = 2 ** 32 - 1500, clock_offset_ns: int = 3_700_000,
                  base_latency_ns: int = 2_000_000, jitter_shape: float = 2.0, jitter_scale_ns: float = 500_000.0,
                  p_good_to_bad: float = 0.005, p_bad_to_good: float = 0.2, loss_good: float = 0.01,
                  loss_bad: float = 0.5, duplicate_probability: float = 0.002, channels: int = 4,
                  forced_losses=(2 ** 32 - 1,)) -> dict:
    """Seeded Gilbert-Elliott loss, gamma jitter and duplicates on a stream of encoded frames."""
    rng = np.random.Generator(np.random.PCG64(seed))
    bad, lost, deliveries = False, set(), []
    payload = rng.integers(-2 ** 20, 2 ** 20, size=(frames, channels))
    for k in range(frames):
        bad = (rng.random() >= p_bad_to_good) if bad else (rng.random() < p_good_to_bad)
        sequence = (start_sequence + k) % SEQUENCE_MODULUS
        timestamp = k * period_ns
        drop = rng.random() < (loss_bad if bad else loss_good)
        copies = 1 + int(rng.random() < duplicate_probability)
        latencies = [int(base_latency_ns + rng.gamma(jitter_shape, jitter_scale_ns)) for _ in range(copies)]
        if drop or sequence in forced_losses:
            lost.add(sequence)
            continue
        frame = encode_frame(sequence, timestamp, 7, payload[k])
        for latency in latencies:
            deliveries.append((timestamp + clock_offset_ns + latency, sequence, latency, frame))
    deliveries.sort(key=lambda item: (item[0], item[1]))
    return {"deliveries": deliveries, "lost": lost, "frames": frames, "start_sequence": start_sequence,
            "clock_offset_ns": clock_offset_ns, "period_ns": period_ns,
            "model": {"p_good_to_bad": p_good_to_bad, "p_bad_to_good": p_bad_to_good, "loss_good": loss_good,
                      "loss_bad": loss_bad, "base_latency_ns": base_latency_ns, "jitter_shape": jitter_shape,
                      "jitter_scale_ns": jitter_scale_ns, "duplicate_probability": duplicate_probability,
                      "forced_losses": list(forced_losses)}}


# ------------------------------------------------------ bitstream identity
IDENTITY_SCHEMA = "ciw.fpga-bitstream-identity.v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PINNED_VERSION = re.compile(r"^\d+(\.\d+){1,3}([._-][A-Za-z0-9]+)?$")
IDENTITY_FIELDS = ("schema", "bitstream_sha256", "bitstream_bytes", "toolchain", "part", "constraints",
                   "constraints_sha256", "source_tree", "synthesis_options", "synthetic")


class IdentityRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _files_digest(files: dict) -> dict:
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    return {"files": manifest, "sha256": canonical_sha256(manifest)}


def synthetic_bitstream(seed: int, size: int = 32768) -> bytes:
    """Seeded placeholder bytes with an unmistakable prefix; not an FPGA configuration."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return b"CIW-SYNTHETIC-BITSTREAM-NOT-A-CONFIGURATION\0" + rng.integers(0, 256, size, dtype=np.uint8).tobytes()


def bitstream_identity(bitstream: bytes, *, toolchain: dict, part: str, constraints: dict, source_files: dict,
                       synthesis_options: dict, synthetic: bool) -> dict:
    constraint_digest = _files_digest(constraints)
    record = {"schema": IDENTITY_SCHEMA, "bitstream_sha256": hashlib.sha256(bitstream).hexdigest(),
              "bitstream_bytes": len(bitstream), "toolchain": dict(toolchain), "part": part,
              "constraints": constraint_digest["files"], "constraints_sha256": constraint_digest["sha256"],
              "source_tree": {"kind": "manifest-sha256", "value": _files_digest(source_files)["sha256"]},
              "synthesis_options": dict(synthesis_options), "synthetic": bool(synthetic)}
    record["record_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return validate_identity(record)


def validate_identity(record: dict, *, bitstream: bytes | None = None, constraints: dict | None = None,
                      source_files: dict | None = None) -> dict:
    """Refuse incomplete, floating, unbound or tampered identity records."""
    missing = [name for name in IDENTITY_FIELDS + ("record_sha256",) if name not in record]
    if missing:
        raise IdentityRefusal("missing_field", f"Bitstream identity is missing fields: {missing}")
    if record["schema"] != IDENTITY_SCHEMA:
        raise IdentityRefusal("wrong_schema", f"Expected {IDENTITY_SCHEMA}")
    for name in ("bitstream_sha256", "constraints_sha256", "record_sha256"):
        if not isinstance(record[name], str) or not _HEX64.match(record[name]):
            raise IdentityRefusal("bad_digest_format", f"{name} must be 64 lowercase hex digits")
    toolchain = record["toolchain"]
    if not isinstance(toolchain, dict) or not toolchain.get("name"):
        raise IdentityRefusal("missing_field", "Toolchain name is required")
    if not _PINNED_VERSION.match(str(toolchain.get("version", ""))):
        raise IdentityRefusal("floating_toolchain_version", "Toolchain version must be an exact pinned version")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != record["record_sha256"]:
        raise IdentityRefusal("record_digest_mismatch", "Identity record content does not match record_sha256")
    if canonical_sha256(record["constraints"]) != record["constraints_sha256"]:
        raise IdentityRefusal("constraints_digest_mismatch", "Constraint manifest does not match constraints_sha256")
    if bitstream is not None:
        if len(bitstream) != record["bitstream_bytes"]:
            raise IdentityRefusal("bitstream_size_mismatch", "Bitstream length differs from its identity record")
        if hashlib.sha256(bitstream).hexdigest() != record["bitstream_sha256"]:
            raise IdentityRefusal("bitstream_digest_mismatch", "Bitstream bytes differ from their identity record")
    if constraints is not None and _files_digest(constraints)["sha256"] != record["constraints_sha256"]:
        raise IdentityRefusal("constraints_digest_mismatch", "Constraint files differ from their identity record")
    if source_files is not None and _files_digest(source_files)["sha256"] != record["source_tree"]["value"]:
        raise IdentityRefusal("source_tree_mismatch", "Source files differ from the recorded source tree")
    return record


def deployment_decision(record: dict) -> None:
    """Deployment is outside the lab: synthetic records and every real record are refused here."""
    validate_identity(record)
    if record["synthetic"]:
        raise IdentityRefusal("synthetic_bitstream", "A synthetic placeholder bitstream is never deployable")
    raise IdentityRefusal("deployment_authority_absent", "Bitstream deployment requires authority the lab does not hold")


# ------------------------------------------------- compatibility and rollback
COMPATIBILITY = {
    "schema": "ciw.fpga-compatibility.v1",
    "matrix_version": 3,
    "bitstreams": {
        "1.2.0": {"frame_format": 1, "boards": ["revA", "revB"], "min_host": "1.0"},
        "1.4.1": {"frame_format": 1, "boards": ["revA", "revB", "revC"], "min_host": "1.0"},
        "2.0.0": {"frame_format": 2, "boards": ["revB", "revC"], "min_host": "1.5"},
        "2.1.0": {"frame_format": 2, "boards": ["revC"], "min_host": "2.0"},
    },
    "hosts": {"1.0": {"frame_formats": [1]}, "1.5": {"frame_formats": [1, 2]}, "2.0": {"frame_formats": [2]}},
    "boards": ["revA", "revB", "revC"],
}
ROLLBACK_SCHEMA = "ciw.fpga-rollback.v1"


def version_key(version: str) -> tuple:
    return tuple(int(part) for part in version.split("."))


def compatible(matrix: dict, bitstream: str, host: str, board: str) -> bool:
    """Rule form: the host decodes the frame format, supports the board and meets the minimum host version."""
    entry, decoder = matrix["bitstreams"][bitstream], matrix["hosts"][host]
    return (entry["frame_format"] in decoder["frame_formats"] and board in entry["boards"]
            and version_key(host) >= version_key(entry["min_host"]))


def compatible_set(matrix: dict) -> set:
    """Set form of the same relation, built by filtering in a different order."""
    by_format = {}
    for host, decoder in matrix["hosts"].items():
        for fmt in decoder["frame_formats"]:
            by_format.setdefault(fmt, set()).add(host)
    out = set()
    for board in matrix["boards"]:
        for bitstream, entry in matrix["bitstreams"].items():
            if board not in entry["boards"]:
                continue
            hosts = {h for h in by_format.get(entry["frame_format"], set())
                     if version_key(h) >= version_key(entry["min_host"])}
            out |= {(bitstream, host, board) for host in hosts}
    return out


class RollbackRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def rollback_record(matrix: dict, registry: dict, *, current: str, target: str, host: str, board: str,
                    reason: str) -> dict:
    record = {"schema": ROLLBACK_SCHEMA, "from_bitstream": current, "to_bitstream": target, "host": host,
              "board": board, "reason": reason, "compatibility_matrix_sha256": canonical_sha256(matrix),
              "from_identity_sha256": registry.get(current, {}).get("record_sha256"),
              "to_identity_sha256": registry.get(target, {}).get("record_sha256"), "authorization": None}
    return validate_rollback(record, matrix, registry)


def validate_rollback(record: dict, matrix: dict, registry: dict) -> dict:
    """Refuse rollbacks that are unregistered, incompatible, not backwards or unexplained."""
    if record.get("schema") != ROLLBACK_SCHEMA:
        raise RollbackRefusal("wrong_schema", f"Expected {ROLLBACK_SCHEMA}")
    if record.get("compatibility_matrix_sha256") != canonical_sha256(matrix):
        raise RollbackRefusal("matrix_mismatch", "Rollback was checked against a different compatibility matrix")
    if not str(record.get("reason", "")).strip():
        raise RollbackRefusal("missing_reason", "A rollback record states its reason")
    current, target = record.get("from_bitstream"), record.get("to_bitstream")
    for name, digest in ((current, record.get("from_identity_sha256")), (target, record.get("to_identity_sha256"))):
        if name not in registry or name not in matrix["bitstreams"] or registry[name]["record_sha256"] != digest:
            raise RollbackRefusal("unregistered_target", f"Bitstream {name} has no matching retained identity")
    if target == current:
        raise RollbackRefusal("no_op_rollback", "Rollback target equals the current bitstream")
    if version_key(target) > version_key(current):
        raise RollbackRefusal("not_a_rollback", "Target is newer than the current bitstream")
    if record.get("host") not in matrix["hosts"] or record.get("board") not in matrix["boards"]:
        raise RollbackRefusal("unknown_host_or_board", "Host decoder or board revision is not in the matrix")
    if not compatible(matrix, target, record["host"], record["board"]):
        raise RollbackRefusal("incompatible_target", "Rollback target is incompatible with the host decoder or board")
    return record


def execute_rollback(record: dict) -> None:
    raise RollbackRefusal("rollback_requires_machine_authority",
                          "Executing a rollback on hardware requires authority the lab does not hold")
