#!/usr/bin/env python3
"""Import Godot and check its live protocol against an isolated CIW service.

Run with the same Python environment used for CIW, for example:
    python scripts/check_godot.py --godot godot

Port 8765 must be unused. This runner never attaches to or stops an existing
service. Its temporary service and result files are removed on success/failure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8765
READY_MESSAGE = f"Computational Instrumentation Workbench: ws://{HOST}:{PORT}"
SMOKE_PASS = "PASS: full run, response correlation, cross-client broadcast"
GODOT_ERROR = re.compile(r"^\s*(?:SCRIPT ERROR|ERROR)\s*:|\b(?:Parse|Compile) Error\s*:", re.IGNORECASE | re.MULTILINE)
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class CheckError(Exception):
    """A failed prerequisite or verification step."""


def show_output(label: str, output: str | bytes | None) -> None:
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    output = output.strip()
    if not output:
        return
    lines = output.splitlines()
    clipped = "\n".join(lines[-80:])[-12_000:]
    print(f"--- {label} ---", flush=True)
    if clipped != output:
        print("[Earlier diagnostics omitted; showing the final output.]", flush=True)
    print(clipped, flush=True)


def require_unused_port() -> None:
    # Binding checks ownership without initiating a connection to an existing
    # interactive service. Do not enable address reuse for this preflight.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((HOST, PORT))
        except OSError as exc:
            raise CheckError(
                f"Cannot reserve {HOST}:{PORT}; it may already be in use. "
                "Stop your existing CIW service before running this check. "
                "No existing service was contacted or stopped."
            ) from exc


def wait_for_service(process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CheckError(f"Temporary CIW service exited with code {process.returncode}")
        # CIW prints this only after its websocket listener has bound. Requiring
        # our own child's fresh log avoids attaching to another service if it
        # acquired the port between preflight and child startup.
        if READY_MESSAGE in log_path.read_text(encoding="utf-8", errors="replace"):
            if process.poll() is None:
                return
        time.sleep(0.1)
    raise CheckError("Temporary CIW service did not start listening within 15 seconds")


def stop_service(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_godot(executable: str, label: str, arguments: list[str], timeout: int) -> str:
    print(f"Checking {label}...", flush=True)
    try:
        completed = subprocess.run(
            [executable, "--headless", "--path", str(ROOT / "godot"), *arguments],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False, creationflags=HIDDEN,
        )
    except subprocess.TimeoutExpired as exc:
        show_output(f"{label} stdout", exc.stdout)
        show_output(f"{label} stderr", exc.stderr)
        raise CheckError(f"{label} exceeded {timeout} seconds") from exc
    output = completed.stdout + "\n" + completed.stderr
    show_output(f"{label} stdout", completed.stdout)
    show_output(f"{label} stderr", completed.stderr)
    if completed.returncode != 0:
        raise CheckError(f"{label} exited with code {completed.returncode}")
    if GODOT_ERROR.search(output):
        raise CheckError(f"{label} reported a Godot or GDScript error despite exit code zero")
    return output


GENERALITY_PASS = "PASS: the viewport builds channels"


def check(executable: str) -> None:
    require_unused_port()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="ciw-godot-check-") as temporary:
        directory = Path(temporary)
        log_path = directory / "service.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "ciw", "serve", "--port", str(PORT),
                 "--output-dir", str(directory / "results")],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=HIDDEN,
            )
            failed = True
            try:
                wait_for_service(process, log_path)
                print(f"Temporary CIW service ready on {HOST}:{PORT} (pid {process.pid}).", flush=True)
                run_godot(executable, "Godot import", ["--editor", "--import", "--quit"], timeout=60)
                if process.poll() is not None:
                    raise CheckError("Temporary CIW service stopped during Godot import")
                output = run_godot(
                    executable, "live protocol smoke", ["--script", "res://tests/protocol_smoke.gd"], timeout=30,
                )
                if not any(line.startswith(SMOKE_PASS) for line in output.splitlines()):
                    raise CheckError("Live protocol smoke exited without its PASS sentinel")
                if process.poll() is not None:
                    raise CheckError("Temporary CIW service stopped during the live protocol check")
                generality = run_godot(
                    executable, "channel generality",
                    ["--script", "res://tests/channel_generality.gd"], timeout=60,
                )
                if not any(line.startswith(GENERALITY_PASS) for line in generality.splitlines()):
                    raise CheckError("Channel generality check exited without its PASS sentinel")
                failed = False
            finally:
                stop_service(process)
                if failed:
                    show_output("temporary service diagnostics", log_path.read_text(encoding="utf-8", errors="replace"))
    print("PASS: Godot import, live protocol and channel generality checks; temporary service stopped.",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--godot", required=True, help="Godot executable path or command available on PATH")
    args = parser.parse_args(argv)
    executable = shutil.which(args.godot)
    if executable is None:
        print(f"FAIL: Godot executable not found: {args.godot}", file=sys.stderr)
        return 2
    try:
        check(executable)
    except KeyboardInterrupt:
        print("FAIL: Check interrupted; temporary service stopped.", file=sys.stderr)
        return 130
    except (CheckError, OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
