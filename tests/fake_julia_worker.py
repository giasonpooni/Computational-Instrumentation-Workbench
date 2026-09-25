"""Protocol test double for the Julia oscillator worker; it never runs Julia.

The host launches it through a patched command. Modes exercise the failure
paths a real worker could take: a wrong environment identity, malformed hello,
crash, truncated or oversized frames, timeouts, occurrence mismatches and
nonfinite output. Its numerical replies copy the analytic reference and are
explicitly labelled doubles by the tests that use them.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import sys
import time


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_frame(payload):
    sys.stdout.buffer.write(struct.pack("<I", len(payload)) + payload)
    sys.stdout.buffer.flush()


def read_exact(count):
    data = sys.stdin.buffer.read(count)
    return data if len(data) == count else None


def hello(root, pin, mode):
    value = {
        "schema": "ciw.julia-worker-hello.v1", "protocol_version": 1, "operations": list(pin["operations"]),
        "julia_version": "1.9.0" if mode == "hello-wrong-version" else pin["julia_version"],
        "platform": {"machine": "fake-test-double", "arch": "x86_64", "kernel": "Test", "word_size": 64},
        "threads": 1,
        "options": {"startup_file": "disabled", "fast_math": "default", "check_bounds": 0, "opt_level": 2},
        "worker_source_sha256": digest(root / pin["worker"]["path"]),
        "project_sha256": digest(root / pin["project"]["path"]),
        "manifest_sha256": digest(root / pin["manifest"]["path"]),
        "sysimage": {"name": "fake-sys.so", "sha256": "0" * 64},
        "packages": pin["packages"],
        "solver": {"algorithm": "Tsit5", "package": "OrdinaryDiffEqTsit5",
                   "version": pin["packages"]["OrdinaryDiffEqTsit5"]["version"], "arithmetic": "binary64"},
    }
    if mode == "hello-tampered-worker":
        value["worker_source_sha256"] = "1" * 64
    return json.dumps(value, sort_keys=True).encode()


def parse_request(frame):
    offset = 0
    magic, version, number = struct.unpack_from("<4sIQ", frame, offset)
    offset += 16
    length, = struct.unpack_from("<I", frame, offset)
    offset += 4
    operation = frame[offset:offset + length].decode()
    offset += length
    length, = struct.unpack_from("<I", frame, offset)
    offset += 4
    configuration = frame[offset:offset + length]
    offset += length
    length, = struct.unpack_from("<I", frame, offset)
    offset += 4
    payload = frame[offset:offset + length]
    return number, operation, configuration, payload


def solve(payload, mode):
    """Copy the analytic reference in place of an ODE solve (test double)."""
    omega_0, gamma, mass, q0, v0 = struct.unpack_from("<ddddd", payload, 8)
    count, = struct.unpack_from("<I", payload, 48)
    times = list(struct.unpack_from("<" + "d" * count, payload, 52))
    omega_d = math.sqrt(omega_0 * omega_0 - gamma * gamma)
    b = (v0 + gamma * q0) / omega_d
    q, v, energy = [], [], []
    for t in times:
        c, s, e = math.cos(omega_d * t), math.sin(omega_d * t), math.exp(-gamma * t)
        qi = e * (q0 * c + b * s)
        vi = e * ((b * omega_d - gamma * q0) * c + (-q0 * omega_d - gamma * b) * s)
        q.append(qi)
        v.append(vi)
        energy.append(0.5 * mass * (vi * vi + omega_0 * omega_0 * qi * qi))
    if mode == "nonfinite":
        q[3] = float("nan")
    if mode == "bad-energy":
        energy[3] += 1.0
    if mode == "wrong-coverage":
        times[-1] += 0.001
    code = b"Success"
    body = struct.pack("<I", 0) + struct.pack("<I", len(code)) + code + struct.pack("<I", count)
    for values in (times, q, v, energy):
        body += struct.pack("<" + "d" * count, *values)
    body += struct.pack("<QQQ", 600, 0, 3600)
    return body


def main():
    mode = sys.argv[-1]
    root = Path(os.environ["CIW_FAKE_JULIA_ROOT"])
    pin = json.loads(Path(os.environ["CIW_FAKE_JULIA_PIN"]).read_text())
    if mode == "hello-bad-json":
        write_frame(b"{not json")
        return 0
    if mode == "hello-oversized":
        sys.stdout.buffer.write(struct.pack("<I", 5 << 20))
        sys.stdout.buffer.flush()
        return 0
    if mode == "hello-silent":
        time.sleep(30)
        return 0
    write_frame(hello(root, pin, mode))
    if mode == "crash-after-hello":
        return 3
    served = 0
    while True:
        header = read_exact(4)
        if header is None:
            return 0
        count, = struct.unpack("<I", header)
        frame = read_exact(count)
        if frame is None:
            return 3
        served += 1
        number, operation, configuration, payload = parse_request(frame)
        if mode == "timeout":
            time.sleep(30)
            return 0
        if mode == "crash-mid-request":
            os._exit(9)
        if mode == "truncated-response":
            sys.stdout.buffer.write(struct.pack("<I", 1000) + b"OSCO" + b"\x00" * 20)
            sys.stdout.buffer.flush()
            return 0
        if mode == "oversized-response":
            sys.stdout.buffer.write(struct.pack("<I", 5 << 20))
            sys.stdout.buffer.flush()
            return 0
        if mode == "refuse":
            message = b"double refused"
            body = struct.pack("<I", 4) + struct.pack("<I", len(message)) + message
        else:
            body = solve(payload, mode)
        bound = number + 1 if mode == "wrong-number-header" else number
        write_frame(struct.pack("<4sIQ", b"OSCO", 1, bound) + body)
        if mode == "junk-metadata":
            write_frame(b"\xff\xfe")
        else:
            metadata_number = number + 1 if mode == "wrong-number-metadata" else number
            write_frame(json.dumps({"schema": "ciw.julia-worker-occurrence.v1", "request_number": metadata_number,
                                    "worker_occurrence": number, "elapsed_ns": 1000}).encode())
        if mode == "crash-after-response":
            return 3


if __name__ == "__main__":
    sys.exit(main())
