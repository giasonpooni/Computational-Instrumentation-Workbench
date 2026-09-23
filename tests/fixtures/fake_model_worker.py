"""Protocol double for Julia worker failure tests. It performs no numerical work.

Usage: python fake_model_worker.py <mode> [state-file] <profile>
Protocol doubles exercise lifecycle and refusal paths only; they never satisfy
a numerical gate.
"""
import hashlib
import os
import struct
import sys

from ciw.model import codec
from ciw.model.commitments import Specification, computation_identity, output_identity

mode, profile = sys.argv[1], sys.argv[-1]
state = sys.argv[2] if len(sys.argv) == 4 else None
out = sys.stdout.buffer


def frame(value):
    payload = codec.encode(value)
    out.write(b"CIWF" + struct.pack("<I", len(payload)) + payload)
    out.flush()


def read_frame():
    header = sys.stdin.buffer.read(8)
    if len(header) < 8:
        return None
    size = struct.unpack("<I", header[4:])[0]
    return codec.decode(sys.stdin.buffer.read(size))


identity = {"protocol": "ciw.julia-model-worker.v1", "profile": profile, "fake": True}
digest = hashlib.sha256(codec.encode(identity)).hexdigest()
if mode == "garbage":
    out.write(b"not a frame at all")
    out.flush()
    sys.exit(0)
frame({"type": "hello", "identity": identity, "context": {"double": True},
       "runtime_digest": "0" * 64 if mode == "bad_digest" else digest})
if mode == "fail_first" and state and not os.path.exists(state):
    open(state, "w").close()
    mode = "crash"
elif mode == "fail_first":
    mode = "ok"
while True:
    message = read_frame()
    if message is None or message["type"] == "shutdown":
        sys.exit(0)
    if mode == "hang":
        sys.stdin.buffer.read()
        sys.exit(0)
    if mode == "crash":
        sys.exit(3)
    if mode == "truncate":
        out.write(b"CIWF" + struct.pack("<I", 100) + b"0123456789")
        out.flush()
        sys.exit(0)
    spec = Specification(message["program"], message["configuration"], message["input"])
    response = {"type": "result", "occurrence": message["occurrence"], "specification_identity": spec.identity(),
                "program_identity": spec.program_identity(), "input_identity": spec.input_identity(),
                "status": "completed", "exit_code": 0, "detail": None, "elapsed_ns": 1,
                "output": None, "output_identity": None, "computation_identity": None}
    output = codec.encode({"input_bytes": len(message["input"])})
    if mode in ("ok", "wrong_occurrence", "wrong_output_identity", "wrong_input_identity"):
        oid = output_identity(output)
        response.update(output=output, output_identity=oid,
                        computation_identity=computation_identity(spec.program_identity(), spec.input_identity(), oid, 0))
    if mode == "wrong_occurrence":
        response["occurrence"] += 1
    if mode == "wrong_output_identity":
        response["output_identity"] = "f" * 64
    if mode == "wrong_input_identity":
        response["input_identity"] = "e" * 64
    if mode == "unrunnable":
        response.update(status="unrunnable", detail="program is not registered in this worker runtime")
    if mode == "halted":
        response.update(status="halted", exit_code=3, detail="solver retcode MaxIters")
    if mode == "halted_with_output":
        response.update(status="halted", exit_code=3, output=output, output_identity=output_identity(output))
    frame(response)
