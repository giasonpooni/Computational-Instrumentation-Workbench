"""Build and verify an isolated Compose deployment, including persisted replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

from ciw.cli import request_remote

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker", help="Docker CLI path or command")
    args = parser.parse_args()
    docker = shutil.which(args.docker)
    if not docker:
        raise SystemExit("Docker CLI is required for the container check")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    project = "ciw-check-" + uuid.uuid4().hex[:12]
    environment = {**os.environ, "CIW_PORT": str(port)}
    compose = [docker, "compose", "--project-name", project, "--file", str(ROOT / "compose.yaml")]
    url = f"ws://127.0.0.1:{port}"

    def run(arguments: list[str], timeout: int = 60) -> str:
        completed = subprocess.run(arguments, cwd=ROOT, env=environment, text=True,
                                   capture_output=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"Command failed: {' '.join(arguments)}\n{completed.stdout[-8000:]}\n{completed.stderr[-8000:]}")
        return completed.stdout

    def request(kind: str, payload: dict | None = None) -> dict:
        response = asyncio.run(request_remote(url, kind, payload or {}))
        if response["type"] != "response":
            raise RuntimeError(f"{kind}: {response}")
        return response["payload"]

    def wait_ready(previous_session: str | None = None) -> dict:
        deadline = time.monotonic() + 60
        last_error = "No response"
        while time.monotonic() < deadline:
            try:
                snapshot = request("session.get")
                if snapshot["session_id"] != previous_session:
                    return snapshot
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"Container did not become ready: {last_error}")

    started = False
    try:
        run([docker, "info"], timeout=20)
        run(compose + ["config", "--quiet"])
        print(f"Building isolated Compose project {project}...", flush=True)
        started = True
        run(compose + ["up", "--build", "--detach", "--wait", "--wait-timeout", "90"], timeout=600)
        initial = wait_ready()
        container = run(compose + ["ps", "--quiet", "backend"]).strip()
        detail = json.loads(run([docker, "inspect", container]))[0]
        assert detail["Config"]["User"] not in {"", "root", "0", "0:0"}, "Container must run as non-root"
        assert detail["HostConfig"]["ReadonlyRootfs"], "Container root filesystem must be read-only"
        bindings = detail["HostConfig"]["PortBindings"]["8765/tcp"]
        assert all(binding["HostIp"] == "127.0.0.1" for binding in bindings), "Publish only on loopback"
        selected = request("selection.update", {"expected_revision": initial["selection"]["revision"],
                                                "cursor_s": 3.0, "interval_s": [2.0, 8.0]})
        result = request("analysis.spectrum")
        print("Restarting container to verify graceful persistence...", flush=True)
        # Deliberately do not send workspace.save: SIGTERM must drain and save.
        run(compose + ["restart", "--timeout", "20", "backend"], timeout=40)
        restored = wait_ready(initial["session_id"])
        assert restored["selection"] == selected
        assert any(row["result_id"] == result["result_id"] for row in restored["results"])
        assert request("result.get", {"result_id": result["result_id"]}) == result
        print("PASS: non-root loopback container, read-only root, health, shutdown persistence and exact replay", flush=True)
    except Exception:
        if started:
            diagnostics = subprocess.run(compose + ["logs", "--no-color", "--tail", "80"], cwd=ROOT,
                                         env=environment, text=True, capture_output=True, timeout=20)
            print(diagnostics.stdout[-12000:])
        raise
    finally:
        if started:
            # Project name is generated above; these are exclusively this check's resources.
            run(compose + ["down", "--volumes", "--remove-orphans"], timeout=40)


if __name__ == "__main__":
    main()
