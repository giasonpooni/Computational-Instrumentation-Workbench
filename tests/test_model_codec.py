"""CIWB canonical bytes and SCR-identical commitments."""
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

from ciw.model import codec
from ciw.model.commitments import (COMPUTATION_TAG, Specification, canonical, commit_hex, computation_identity,
                                   output_identity)


def test_round_trip_is_exact_and_canonical():
    value = {"b": [1, -2, 3.5, None, True, False, "π"], "a": codec.F64Array([[0.1, 1e-300], [-0.0, 2.0]]),
             "raw": b"\x00\xff", "n": -(1 << 63)}
    raw = codec.encode(value)
    assert codec.encode(codec.decode(raw)) == raw
    decoded = codec.decode(raw)
    assert decoded["a"] == value["a"] and decoded["a"].array.tobytes() == value["a"].array.tobytes()
    assert list(decoded) == ["a", "b", "n", "raw"]
    assert decoded["b"][4] is True and decoded["b"][0] == 1 and type(decoded["b"][0]) is int
    assert math.copysign(1.0, decoded["a"].array[1, 0]) == -1.0  # signed zero survives
    assert codec.encode({"x": 1, "y": 2}) == codec.encode({"y": 2, "x": 1})


def test_booleans_and_integers_remain_distinct():
    assert codec.encode(True) != codec.encode(1)
    assert codec.encode(1) != codec.encode(1.0)


@pytest.mark.parametrize("raw, message", [
    (b"CIWB\x01\x00" + b"\x08" + struct.pack("<I", 2) + struct.pack("<I", 1) + b"b\x00" + struct.pack("<I", 1) + b"a\x00",
     "strictly increasing"),
    (codec.encode(1) + b"\x00", "Trailing"),
    (codec.encode("abc")[:-1], "Truncated"),
    (b"CIWB\x02\x00\x00", "Not a CIWB"),
    (b"CIWB\x01\x00\x04" + struct.pack("<d", float("nan")), "Nonfinite"),
    (b"CIWB\x01\x00\x0a", "Unknown CIWB tag"),
])
def test_noncanonical_or_malformed_bytes_are_refused(raw, message):
    with pytest.raises(codec.CodecError, match=message):
        codec.decode(raw)


def test_encoder_refuses_nonfinite_and_unsupported_values():
    with pytest.raises(codec.CodecError):
        codec.encode(float("inf"))
    with pytest.raises(codec.CodecError):
        codec.encode(codec.F64Array([1.0, np.nan]))
    with pytest.raises(codec.CodecError):
        codec.encode({1: "x"})
    with pytest.raises(codec.CodecError):
        codec.encode(object())


def test_commitment_vectors_are_fixed():
    spec = Specification(b"program", b"configuration", b"input")
    assert canonical("t", [b"ab"]) == struct.pack("<Q", 1) + b"t" + struct.pack("<Q", 1) + struct.pack("<Q", 2) + b"ab"
    assert spec.identity() == "899119154f523d1dc54b3bf58cfa43a4454a1162ccefeff1e87490e25510e600"
    out = output_identity(b"output")
    assert computation_identity(spec.program_identity(), spec.input_identity(), out, 0) == commit_hex(
        COMPUTATION_TAG, [bytes.fromhex(spec.program_identity()), bytes.fromhex(spec.input_identity()),
                          bytes.fromhex(out), struct.pack("<I", 0)])


SCR = os.environ.get("CIW_SCR_REPO")


@pytest.mark.skipif(not SCR, reason="CIW_SCR_REPO names the pinned SCR checkout")
def test_commitments_agree_byte_for_byte_with_scr():
    sys.path.insert(0, str(Path(SCR).resolve()))
    try:
        from execution import commitments as scr
        from execution.specification import ExecutionSpecification
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(7)
    for size in (0, 1, 31, 1024):
        program, configuration, payload = (rng.bytes(size) for _ in range(3))
        ours = Specification(program, configuration, payload)
        theirs = ExecutionSpecification(program, configuration, payload)
        assert ours.identity() == theirs.identity()
        assert ours.program_identity() == theirs.program_identity()
        assert ours.input_identity() == theirs.input_identity()
        assert output_identity(payload) == scr.commit_hex(scr.OUTPUT_TAG, [payload])
        assert computation_identity(ours.program_identity(), ours.input_identity(), output_identity(payload), 5) == \
            scr.commit_hex(scr.COMPUTATION_TAG, [bytes.fromhex(ours.program_identity()), bytes.fromhex(ours.input_identity()),
                                                 bytes.fromhex(output_identity(payload)), scr.canonical_u32(5)])
