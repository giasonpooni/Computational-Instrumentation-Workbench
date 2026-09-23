"""``CIWB`` v1: a bounded canonical binary encoding for worker bytes.

Every integer and binary64 value has fixed little-endian width; strings are
UTF-8 with explicit lengths; map keys are strictly increasing by UTF-8 bytes;
arrays carry explicit dimensions and row-major binary64 data. There is no
locale-dependent text formatting, so values round-trip exactly and equal
values have equal bytes. Decoding refuses noncanonical input rather than
normalizing it. The JSON views of decoded values are derived representations.

    header  b"CIWB" u16 version=1
    value   0x00 null | 0x01 false | 0x02 true | 0x03 i64 | 0x04 f64
            0x05 u32 n, n bytes UTF-8 | 0x06 u32 n, n raw bytes
            0x07 u32 n, n values (list) | 0x08 u32 n, n (u32 k, key, value) (map)
            0x09 u8 d, d x u32 dims, prod(dims) f64 (row-major array)
"""
from __future__ import annotations

import math
import struct

import numpy as np

MAGIC = b"CIWB"
VERSION = 1
MAX_DEPTH = 32
MAX_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 1 << 22


class CodecError(ValueError):
    pass


class F64Array:
    """A binary64 array value; encoded with dimensions, never as a list of numbers."""

    __slots__ = ("array",)

    def __init__(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 1 or array.ndim > 4:
            raise CodecError("Binary64 arrays have 1..4 dimensions")
        self.array = np.ascontiguousarray(array)

    def __eq__(self, other):
        return (isinstance(other, F64Array) and self.array.shape == other.array.shape
                and self.array.tobytes() == other.array.tobytes())

    def __repr__(self):
        return f"F64Array(shape={self.array.shape})"


def encode(value, *, allow_nonfinite: bool = False) -> bytes:
    out = bytearray(MAGIC + struct.pack("<H", VERSION))
    _encode(value, out, 0, allow_nonfinite)
    if len(out) > MAX_BYTES:
        raise CodecError("Encoded value exceeds its byte bound")
    return bytes(out)


def _encode(value, out, depth, allow_nonfinite):
    if depth > MAX_DEPTH:
        raise CodecError("Value exceeds its nesting bound")
    if value is None:
        out += b"\x00"
    elif value is False:
        out += b"\x01"
    elif value is True:
        out += b"\x02"
    elif type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            raise CodecError("Integer outside i64")
        out += b"\x03" + struct.pack("<q", value)
    elif type(value) is float:
        if not allow_nonfinite and not math.isfinite(value):
            raise CodecError("Nonfinite binary64 value")
        out += b"\x04" + struct.pack("<d", value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        out += b"\x05" + struct.pack("<I", len(raw)) + raw
    elif isinstance(value, (bytes, bytearray)):
        out += b"\x06" + struct.pack("<I", len(value)) + bytes(value)
    elif isinstance(value, (list, tuple)):
        out += b"\x07" + struct.pack("<I", len(value))
        for item in value:
            _encode(item, out, depth + 1, allow_nonfinite)
    elif isinstance(value, dict):
        keys = []
        for key in value:
            if not isinstance(key, str):
                raise CodecError("Map keys must be strings")
            keys.append((key.encode("utf-8"), key))
        keys.sort()
        out += b"\x08" + struct.pack("<I", len(keys))
        for raw, key in keys:
            out += struct.pack("<I", len(raw)) + raw
            _encode(value[key], out, depth + 1, allow_nonfinite)
    elif isinstance(value, F64Array):
        array = value.array
        if not allow_nonfinite and not np.all(np.isfinite(array)):
            raise CodecError("Nonfinite binary64 array value")
        out += b"\x09" + struct.pack("<B", array.ndim) + b"".join(struct.pack("<I", n) for n in array.shape)
        out += array.astype("<f8", copy=False).tobytes(order="C")
    else:
        raise CodecError(f"Unsupported value type: {type(value).__name__}")


def decode(raw: bytes, *, allow_nonfinite: bool = False):
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_BYTES:
        raise CodecError("Encoded value must be bounded bytes")
    raw = bytes(raw)
    if raw[:4] != MAGIC or len(raw) < 6 or struct.unpack_from("<H", raw, 4)[0] != VERSION:
        raise CodecError("Not a CIWB v1 value")
    value, position = _decode(raw, 6, 0, allow_nonfinite)
    if position != len(raw):
        raise CodecError("Trailing bytes after CIWB value")
    return value


def _take(raw, position, count):
    end = position + count
    if count < 0 or end > len(raw):
        raise CodecError("Truncated CIWB value")
    return raw[position:end], end


def _decode(raw, position, depth, allow_nonfinite):
    if depth > MAX_DEPTH:
        raise CodecError("Value exceeds its nesting bound")
    tag, position = _take(raw, position, 1)
    tag = tag[0]
    if tag == 0:
        return None, position
    if tag in (1, 2):
        return tag == 2, position
    if tag == 3:
        chunk, position = _take(raw, position, 8)
        return struct.unpack("<q", chunk)[0], position
    if tag == 4:
        chunk, position = _take(raw, position, 8)
        value = struct.unpack("<d", chunk)[0]
        if not allow_nonfinite and not math.isfinite(value):
            raise CodecError("Nonfinite binary64 value")
        return value, position
    if tag in (5, 6):
        chunk, position = _take(raw, position, 4)
        data, position = _take(raw, position, struct.unpack("<I", chunk)[0])
        if tag == 6:
            return data, position
        try:
            return data.decode("utf-8"), position
        except UnicodeDecodeError as exc:
            raise CodecError("Invalid UTF-8 string") from exc
    if tag in (7, 8):
        chunk, position = _take(raw, position, 4)
        count = struct.unpack("<I", chunk)[0]
        if count > MAX_ITEMS:
            raise CodecError("Container exceeds its item bound")
        if tag == 7:
            items = []
            for _ in range(count):
                item, position = _decode(raw, position, depth + 1, allow_nonfinite)
                items.append(item)
            return items, position
        result, previous = {}, None
        for _ in range(count):
            chunk, position = _take(raw, position, 4)
            key_raw, position = _take(raw, position, struct.unpack("<I", chunk)[0])
            if previous is not None and key_raw <= previous:
                raise CodecError("Map keys must be strictly increasing (canonical order)")
            previous = key_raw
            try:
                key = key_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CodecError("Invalid UTF-8 key") from exc
            result[key], position = _decode(raw, position, depth + 1, allow_nonfinite)
        return result, position
    if tag == 9:
        chunk, position = _take(raw, position, 1)
        ndim = chunk[0]
        if not 1 <= ndim <= 4:
            raise CodecError("Binary64 arrays have 1..4 dimensions")
        chunk, position = _take(raw, position, 4 * ndim)
        shape = struct.unpack(f"<{ndim}I", chunk)
        size = 1
        for n in shape:
            size *= n
        if size > MAX_ITEMS:
            raise CodecError("Array exceeds its item bound")
        data, position = _take(raw, position, 8 * size)
        array = np.frombuffer(data, dtype="<f8").reshape(shape).astype(np.float64)
        if not allow_nonfinite and not np.all(np.isfinite(array)):
            raise CodecError("Nonfinite binary64 array value")
        return F64Array(array), position
    raise CodecError(f"Unknown CIWB tag {tag}")


def json_view(value):
    """A JSON-compatible derived view; arrays become nested lists."""
    if isinstance(value, F64Array):
        return value.array.tolist()
    if isinstance(value, (bytes, bytearray)):
        return {"bytes_hex": bytes(value).hex()}
    if isinstance(value, dict):
        return {key: json_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_view(item) for item in value]
    return value
