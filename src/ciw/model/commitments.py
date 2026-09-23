"""SCR execution commitments, byte-for-byte ``execution.commitments``.

CIW recomputes every identity a worker echoes. The encoding is SCR's
``canonical(tag, fields)``: ``len(tag) u64 LE | tag | count u64 LE | (len u64
LE | bytes)*`` hashed with SHA-256. ``tests/test_model_worker.py`` checks these
functions against the pinned SCR checkout when it is supplied, so this is the
same function in the same seam rather than a second identity system.

``SpecificationIdentity`` binds program, configuration and input bytes.
``ComputationIdentity`` binds program, input, output and exit code only; solver
settings are bound through the retained specification link, never implied by
the computation identity alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

PROGRAM_TAG = "scout.execution.program.v1"
INPUT_TAG = "scout.execution.input.v1"
OUTPUT_TAG = "scout.execution.output.v1"
COMPUTATION_TAG = "scout.execution.computation.v1"
SPECIFICATION_TAG = "scout.execution.specification.v1"


def canonical(tag: str, fields) -> bytes:
    tag_bytes = tag.encode("utf-8")
    out = struct.pack("<Q", len(tag_bytes)) + tag_bytes + struct.pack("<Q", len(fields))
    for field in fields:
        out += struct.pack("<Q", len(field)) + field
    return out


def commit_hex(tag: str, fields) -> str:
    return hashlib.sha256(canonical(tag, fields)).hexdigest()


def canonical_u32(value: int) -> bytes:
    return struct.pack("<I", value)


@dataclass(frozen=True)
class Specification:
    """The SCR ``ExecutionSpecification`` triple: a request, not an event."""

    program: bytes
    configuration: bytes
    input_payload: bytes

    def identity(self) -> str:
        return commit_hex(SPECIFICATION_TAG, [self.program, self.configuration, self.input_payload])

    def program_identity(self) -> str:
        return commit_hex(PROGRAM_TAG, [self.program])

    def input_identity(self) -> str:
        return commit_hex(INPUT_TAG, [self.input_payload])


def output_identity(output: bytes) -> str:
    return commit_hex(OUTPUT_TAG, [output])


def computation_identity(program_identity: str, input_identity: str, output_id: str, exit_code: int) -> str:
    return commit_hex(COMPUTATION_TAG, [bytes.fromhex(program_identity), bytes.fromhex(input_identity),
                                        bytes.fromhex(output_id), canonical_u32(exit_code)])
