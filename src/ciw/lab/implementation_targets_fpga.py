"""FPGA telemetry-only frames, bitstream identity, compatibility and link simulation (T149-T152).

The frame format is read-only by construction: one frame type (telemetry),
no host-to-device field, and a decoder that refuses command, write and
unknown frame types before any payload is used. Bitstream identity records
bind a bitstream digest to its toolchain, constraints and source tree; the
bitstreams used here are seeded synthetic placeholders, never real
configurations. The link simulation is seeded and synthetic (Gilbert-Elliott loss, gamma
latency, duplicates, quantized device timestamps and clock drift): it says
how the receiver's gap and staleness detectors behave on a declared model,
not how any real FPGA link behaves.
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
# The reflected CRC-32 register is appended least significant byte first, so the frame read LSB-first per byte
# is one codeword in polynomial order and every burst of at most 32 bits is detected. A big-endian trailer
# breaks that order at the payload/CRC boundary (see burst_rank_deficient_windows).
TRAILER = struct.Struct("<I")
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
    "byte_order": "header and payload big-endian; CRC-32 little-endian (least significant byte first)",
    "bit_order": "bit p of the frame is bit p % 8 (least significant first) of byte p // 8",
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
        {"name": "crc32", "offset": "28 + payload_bytes", "bytes": 4,
         "type": "CRC-32/IEEE (reflected) over all preceding bytes, little-endian", "direction": "device_to_host"},
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


def _pack(frame_type: int, flags: int, sequence: int, timestamp_ns: int, clock_id: int, channels, *,
          magic: bytes = MAGIC, version: int = FRAME_VERSION, channel_count: int | None = None,
          payload_bytes: int | None = None) -> bytes:
    channels = [int(c) for c in channels]
    count = len(channels) if channel_count is None else channel_count
    size = 4 * len(channels) if payload_bytes is None else payload_bytes
    body = HEADER.pack(magic, version, frame_type, flags, sequence, timestamp_ns, clock_id, count,
                       size) + struct.pack(f">{len(channels)}i", *channels)
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
                channels=(1, 2), **header) -> bytes:
    """Negative-test generator: a frame with arbitrary type, flags or header fields and a valid CRC.

    ``header`` may override ``magic``, ``version``, ``channel_count`` and ``payload_bytes``.
    """
    return _pack(frame_type, flags, sequence, timestamp_ns, clock_id, channels, **header)


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


def frame_check(data: bytes, trailer_byteorder: str = "little") -> int:
    """CRC residual of a frame: zero when its trailer matches its bytes (affine in the frame bits)."""
    return crc32(data[:-4]) ^ int.from_bytes(data[-4:], trailer_byteorder)


def _flip(data: bytes, positions, msb_first: bool = False) -> bytes:
    out = bytearray(data)
    for p in positions:
        out[p // 8] ^= 1 << (7 - p % 8 if msb_first else p % 8)
    return bytes(out)


def _dependency(vectors) -> int | None:
    """Bit mask of a nonempty subset of GF(2) vectors summing to zero, or None if they are independent."""
    basis = {}  # leading bit -> (vector, combination mask)
    for index, vector in enumerate(vectors):
        mask = 1 << index
        while vector:
            lead = vector.bit_length() - 1
            if lead not in basis:
                basis[lead] = (vector, mask)
                break
            vector ^= basis[lead][0]
            mask ^= basis[lead][1]
        else:
            return mask
    return None


def burst_rank_deficient_windows(frame: bytes, width: int = 32, trailer_byteorder: str = "little",
                                 msb_first: bool = False, witness_from: int = 0) -> dict:
    """Exact burst analysis: windows of ``width`` consecutive bits whose single-bit syndromes are dependent.

    An error pattern confined to a window escapes the CRC iff its syndromes
    sum to zero. If every window of 32 bits has independent syndromes, no
    burst of length <= 32 escapes. Returns the deficient window starts and,
    for the first one starting at or after ``witness_from``, an undetected
    error pattern (bit positions).
    """
    bits = len(frame) * 8
    syndromes = [frame_check(_flip(frame, [p], msb_first), trailer_byteorder) for p in range(bits)]
    deficient, witness = [], None
    for start in range(bits - width + 1):
        mask = _dependency(syndromes[start:start + width])
        if mask is not None:
            deficient.append(start)
            if witness is None and start >= witness_from:
                witness = [start + i for i in range(width) if mask >> i & 1]
    return {"bits": bits, "windows": bits - width + 1, "deficient": deficient, "witness": witness}


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
        self.received: set = set()  # unbounded here; a deployed receiver would keep a sliding bitmap
        self.pending: set = set()
        self.reordered = self.duplicates = self.before_start = self.accepted = 0
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
        if sequence in self.received:
            self.duplicates += 1
        elif self.highest is None:
            self.received.add(sequence)
            self.highest = self.first = sequence
        else:
            self.received.add(sequence)
            delta = (sequence - self.highest + 2 ** 31) % SEQUENCE_MODULUS - 2 ** 31
            if delta > 0:
                for step in range(1, delta):
                    self.pending.add((self.highest + step) % SEQUENCE_MODULUS)
                self.highest = sequence
            elif sequence in self.pending:
                self.pending.discard(sequence)
                self.reordered += 1
            else:
                self.before_start += 1  # older than the first frame received
        age = received_ns - (record["timestamp_ns"] + self.clock_offset_ns)
        self.ages.append(age)
        if age > self.stale_after_ns:
            self.stale.append((sequence, received_ns))
        return record


def naive_gap_count(sequences) -> int:
    """Unsafe detector: sums positive (seq - previous - 1) in the given order, without modular arithmetic."""
    lost, previous = 0, None
    for sequence in sequences:
        if previous is not None and sequence - previous - 1 > 0:
            lost += sequence - previous - 1
        previous = sequence
    return lost


LINK_MODEL = {
    "period_ns": 1_000_000, "clock_offset_ns": 3_700_000, "base_latency_ns": 2_000_000, "jitter_shape": 2.0,
    "jitter_scale_ns": 500_000.0, "p_good_to_bad": 0.005, "p_bad_to_good": 0.2, "loss_good": 0.01, "loss_bad": 0.5,
    "duplicate_probability": 0.002, "sample_jitter_ns": 250_000, "tick_ns": 50_000, "drift_ppm": 20.0,
}


def link_draws(seed: int, frames: int, model: dict = LINK_MODEL) -> dict:
    """Per-frame Gilbert-Elliott state, loss, duplicate, latency and sampling-phase draws.

    Everything is drawn in bulk; only the two-state chain is iterated. The
    chain starts in the good state and switches after each draw of u_state.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    u_state, u_drop, u_duplicate = rng.random(frames), rng.random(frames), rng.random(frames)
    latency = np.floor(model["base_latency_ns"] + rng.gamma(model["jitter_shape"], model["jitter_scale_ns"],
                                                            size=(frames, 2))).astype(np.int64)
    phase = np.floor(rng.random(frames) * model["sample_jitter_ns"]).astype(np.int64)
    bad, state = [], False
    for u in u_state.tolist():
        state = (u >= model["p_bad_to_good"]) if state else (u < model["p_good_to_bad"])
        bad.append(state)
    bad = np.array(bad, dtype=bool)
    drop = u_drop < np.where(bad, model["loss_bad"], model["loss_good"])
    return {"bad": bad, "drop": drop, "duplicate": u_duplicate < model["duplicate_probability"],
            "latency": latency, "phase": phase}


def gilbert_elliott_moments(model: dict) -> dict:
    """Stationary loss probability and the exact asymptotic variance of the loss count per frame.

    X_k ~ Bernoulli(l_{S_k}) given the chain state; Cov(X_0, X_h) = (l_b - l_g)^2 pi_g pi_b lambda^h with
    lambda = 1 - p_gb - p_bg, so N Var(mean) -> p(1 - p) + 2 (l_b - l_g)^2 pi_g pi_b lambda / (1 - lambda).
    """
    p_gb, p_bg = model["p_good_to_bad"], model["p_bad_to_good"]
    pi_b = p_gb / (p_gb + p_bg)
    pi_g = 1 - pi_b
    p = pi_g * model["loss_good"] + pi_b * model["loss_bad"]
    lam = 1 - p_gb - p_bg
    variance = p * (1 - p) + 2 * (model["loss_bad"] - model["loss_good"]) ** 2 * pi_g * pi_b * lam / (1 - lam)
    return {"stationary_loss": p, "per_frame_variance": variance, "pi_bad": pi_b, "lambda": lam}


def latency_cdf(x_ns, model: dict = LINK_MODEL):
    """P(latency <= x) for latency = base + Gamma(2, scale): 1 - exp(-g/s)(1 + g/s), g = x - base >= 0."""
    if model["jitter_shape"] != 2.0:
        raise ValueError("The closed-form latency distribution assumes gamma shape 2")
    g = np.maximum(np.asarray(x_ns, dtype=float) - model["base_latency_ns"], 0.0) / model["jitter_scale_ns"]
    return 1.0 - np.exp(-g) * (1.0 + g)


def simulate_link(seed: int = 152, frames: int = 4000, start_sequence: int = 2 ** 32 - 1500,
                  model: dict = LINK_MODEL, channels: int = 4, forced_losses=(2 ** 32 - 1,)) -> dict:
    """Seeded stream of encoded frames with loss, duplicates, latency, quantized timestamps and clock drift.

    Device sample time t_k = k period + phase_k; timestamp = floor(t_k / tick) tick; host receive time =
    t_k (1 + drift) + offset + latency. The receiver's age with the declared offset is latency + excess,
    excess = (t_k - timestamp) + drift t_k >= 0; a frame is truly stale iff its latency exceeds the limit.
    """
    draws = link_draws(seed, frames, model)
    payload = np.random.Generator(np.random.PCG64(seed + 1)).integers(-2 ** 20, 2 ** 20, size=(frames, channels))
    lost, deliveries = set(), []
    for k in range(frames):
        sequence = (start_sequence + k) % SEQUENCE_MODULUS
        if bool(draws["drop"][k]) or sequence in forced_losses:
            lost.add(sequence)
            continue
        true_ns = k * model["period_ns"] + int(draws["phase"][k])
        stamp = true_ns // model["tick_ns"] * model["tick_ns"]
        drift = round(true_ns * model["drift_ppm"] * 1e-6)
        frame = encode_frame(sequence, stamp, 7, payload[k])
        for copy in range(1 + int(draws["duplicate"][k])):
            latency = int(draws["latency"][k, copy])
            deliveries.append({"received_ns": true_ns + drift + model["clock_offset_ns"] + latency,
                               "sequence": sequence, "latency_ns": latency, "excess_ns": true_ns - stamp + drift,
                               "frame": frame})
    deliveries.sort(key=lambda item: (item["received_ns"], item["sequence"]))
    return {"deliveries": deliveries, "lost": lost, "frames": frames, "start_sequence": start_sequence,
            "clock_offset_ns": model["clock_offset_ns"], "model": dict(model, forced_losses=list(forced_losses))}


# ------------------------------------------------------ bitstream identity
IDENTITY_SCHEMA = "ciw.fpga-bitstream-identity.v1"
# Patterns end in \Z (not $, which also matches before a final newline) and are applied with fullmatch.
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
# Exact release numbers, optionally with a numbered pre-release; wildcards, ranges and moving labels are refused.
_PINNED_VERSION = re.compile(r"[0-9]+(\.[0-9]+){1,3}([-_.](rc|beta|alpha)[0-9]+)?\Z")
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
    """Refuse incomplete, malformed, floating, unbound or tampered identity records."""
    if not isinstance(record, dict):
        raise IdentityRefusal("missing_field", "Bitstream identity must be an object")
    missing = [name for name in IDENTITY_FIELDS + ("record_sha256",) if name not in record]
    if missing:
        raise IdentityRefusal("missing_field", f"Bitstream identity is missing fields: {missing}")
    if record["schema"] != IDENTITY_SCHEMA:
        raise IdentityRefusal("wrong_schema", f"Expected {IDENTITY_SCHEMA}")
    for name in ("bitstream_sha256", "constraints_sha256", "record_sha256"):
        if not isinstance(record[name], str) or not _HEX64.fullmatch(record[name]):
            raise IdentityRefusal("bad_digest_format", f"{name} must be 64 lowercase hex digits")
    tree = record["source_tree"]
    if not isinstance(tree, dict) or tree.get("kind") != "manifest-sha256" or set(tree) != {"kind", "value"}:
        raise IdentityRefusal("missing_field", "source_tree must be {kind: manifest-sha256, value: <sha256>}")
    constraint_files = record["constraints"]
    if not isinstance(constraint_files, dict) or not constraint_files:
        raise IdentityRefusal("missing_field", "constraints must map each constraint file to its sha256")
    for digest in [tree["value"], *constraint_files.values()]:
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise IdentityRefusal("bad_digest_format", "Source-tree and constraint digests must be 64 lowercase hex digits")
    if type(record["bitstream_bytes"]) is not int or record["bitstream_bytes"] < 0:
        raise IdentityRefusal("bad_field_type", "bitstream_bytes must be a nonnegative integer")
    if type(record["synthetic"]) is not bool:
        raise IdentityRefusal("bad_field_type", "synthetic must be a boolean")
    if not isinstance(record["part"], str) or not record["part"].strip():
        raise IdentityRefusal("missing_field", "Device part is required")
    if not isinstance(record["synthesis_options"], dict):
        raise IdentityRefusal("bad_field_type", "synthesis_options must be an object")
    toolchain = record["toolchain"]
    if not isinstance(toolchain, dict) or not isinstance(toolchain.get("name"), str) or not toolchain["name"].strip():
        raise IdentityRefusal("missing_field", "Toolchain name is required")
    version = toolchain.get("version")
    if not isinstance(version, str) or not _PINNED_VERSION.fullmatch(version):
        raise IdentityRefusal("floating_toolchain_version", "Toolchain version must be an exact pinned version string")
    # A version string names a release; the installation digest pins the executables actually used.
    installation = toolchain.get("installation_sha256")
    if not isinstance(installation, str) or not _HEX64.fullmatch(installation):
        raise IdentityRefusal("unpinned_toolchain_installation",
                              "Toolchain needs installation_sha256 over its executables and libraries")
    try:
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    except ValueError as exc:
        raise IdentityRefusal("bad_field_type", f"Identity record has no canonical encoding: {exc}") from None
    if digest != record["record_sha256"]:
        raise IdentityRefusal("record_digest_mismatch", "Identity record content does not match record_sha256")
    if canonical_sha256(constraint_files) != record["constraints_sha256"]:
        raise IdentityRefusal("constraints_digest_mismatch", "Constraint manifest does not match constraints_sha256")
    if bitstream is not None:
        if len(bitstream) != record["bitstream_bytes"]:
            raise IdentityRefusal("bitstream_size_mismatch", "Bitstream length differs from its identity record")
        if hashlib.sha256(bitstream).hexdigest() != record["bitstream_sha256"]:
            raise IdentityRefusal("bitstream_digest_mismatch", "Bitstream bytes differ from their identity record")
    if constraints is not None and _files_digest(constraints)["sha256"] != record["constraints_sha256"]:
        raise IdentityRefusal("constraints_digest_mismatch", "Constraint files differ from their identity record")
    if source_files is not None and _files_digest(source_files)["sha256"] != tree["value"]:
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
