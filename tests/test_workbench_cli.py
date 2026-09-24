"""Terminal verbs over the shared workbench drive a live session as a separate process."""

import asyncio
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from ciw.cli import parser, request_remote
from ciw.core.identities import evidence_id
from ciw.instruments import make_demo_run
from ciw.server import run_server
from ciw.session import Session
from test_calibration_status import status_run

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "uncertainty-validation" / "consistent.json"
OPERATION = "ciw.uncertainty-validation.v1"


def _environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def _available_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def test_parser_exposes_source_operation_and_bundle_verbs():
    add = parser().parse_args(["source", "add", "--kind", "uncertainty-validation", "--file", str(FIXTURE)])
    assert (add.command, add.source_command, add.kind, add.label, add.timeout) == ("source", "add", "uncertainty-validation", None, 15)
    execute = parser().parse_args(["operation", "execute", OPERATION, "--source", "source:x", "--timeout", "60"])
    assert (execute.operation, execute.source, execute.upstream, execute.configuration_file, execute.timeout) == (OPERATION, "source:x", None, None, 60)
    replay = parser().parse_args(["bundle", "replay", "sha256:abc", "--url", "ws://127.0.0.1:9000"])
    assert (replay.bundle_command, replay.bundle, replay.url) == ("replay", "sha256:abc", "ws://127.0.0.1:9000")
    assert parser().parse_args(["bundle", "list"]).bundle_command == "list"
    with pytest.raises(SystemExit):
        parser().parse_args(["operation", "execute", OPERATION])


def test_terminal_retains_executes_inspects_and_replays_against_a_live_session(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        stopped = asyncio.Event()
        port = _available_port()
        url = f"ws://127.0.0.1:{port}"
        task = asyncio.create_task(run_server(session, port, stop_event=stopped))
        try:
            for _ in range(100):
                try:
                    await request_remote(url, "session.get", {})
                    break
                except OSError:
                    await asyncio.sleep(0.02)
            else:
                pytest.fail("Server did not start")

            async def ciw(*arguments):
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "ciw", *arguments, "--url", url, env=_environment(),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 120)
                return process.returncode, (json.loads(stdout) if stdout else None), stderr.decode()

            code, added, stderr = await ciw("source", "add", "--kind", "uncertainty-validation", "--file", str(FIXTURE))
            assert code == 0, stderr
            assert added["type"] == "response" and added["payload"]["label"] == FIXTURE.name
            assert added["payload"]["byte_count"] == FIXTURE.stat().st_size
            source_id = added["payload"]["source_id"]
            code, listed, _ = await ciw("source", "list")
            assert code == 0 and [item["source_id"] for item in listed["payload"]["sources"]] == [source_id]
            code, operations, _ = await ciw("operation", "list")
            assert any(item["operation_id"] == OPERATION and item["available"] for item in operations["payload"]["operations"])
            code, executed, stderr = await ciw("operation", "execute", OPERATION, "--source", source_id, "--timeout", "120")
            assert code == 0, stderr
            bundle_id = executed["payload"]["bundle_id"]
            code, bundle, _ = await ciw("bundle", "get", bundle_id)
            assert code == 0 and bundle["payload"]["steps"][0]["result"]["data"]["verdict"]["status"] == "consistent"
            code, view, _ = await ciw("bundle", "inspect", bundle_id)
            assert code == 0 and view["payload"]["object_context"]["object_kind"] == "uncertainty_consistency"
            code, replay, stderr = await ciw("bundle", "replay", bundle_id, "--timeout", "120")
            assert code == 0, stderr
            assert replay["payload"]["replay_receipt"]["numerical_match"] is True
            code, bundles, _ = await ciw("bundle", "list")
            assert code == 0 and len(bundles["payload"]["bundles"]) == 2
            # A refusal is printed as the error envelope and reported through the exit status.
            code, refused, _ = await ciw("source", "add", "--kind", "no-such-kind", "--file", str(FIXTURE))
            assert code == 2 and refused["type"] == "error"
            code, refused, _ = await ciw("operation", "execute", OPERATION, "--source", source_id,
                                         "--upstream", "sha256:" + "0" * 64)
            assert code == 2 and refused["type"] == "error"
            assert len(session.workbench.list_bundles()) == 2
        finally:
            stopped.set()
            await asyncio.wait_for(task, 10)
    asyncio.run(exercise())


def test_send_reports_calibration_time_refusals_through_the_exit_status(tmp_path):
    """The serving-time calibration status needs an aware instant; a refusal exits 2 with the error envelope."""
    async def exercise():
        run = make_demo_run()
        run["metadata"].update(status_run()["metadata"])
        run["evidence_id"] = evidence_id(run)
        session = Session(run, tmp_path)
        stopped = asyncio.Event()
        port = _available_port()
        url = f"ws://127.0.0.1:{port}"
        task = asyncio.create_task(run_server(session, port, stop_event=stopped))
        try:
            for _ in range(100):
                try:
                    await request_remote(url, "session.get", {})
                    break
                except OSError:
                    await asyncio.sleep(0.02)
            else:
                pytest.fail("Server did not start")

            async def send(payload):
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "ciw", "send", "session.get", "--payload", payload, "--url", url,
                    env=_environment(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 120)
                return process.returncode, (json.loads(stdout) if stdout else None), stderr.decode()

            code, expired, stderr = await send('{"evaluated_at": "2026-09-20T12:00:00Z"}')
            assert code == 0, stderr
            assert expired["type"] == "response"
            assert expired["payload"]["calibration"][0]["serving"]["expired"] is True
            code, refused, _ = await send('{"evaluated_at": "2026-01-01T00:00:00"}')
            assert code == 2 and refused["type"] == "error" and refused["payload"]["code"] == "invalid_payload"
            assert "timezone-aware" in refused["payload"]["message"]
            code, refused, _ = await send('{"evaluated_at": null}')
            assert code == 2 and refused["type"] == "error"
            code, nothing, stderr = await send("[]")
            assert code == 2 and nothing is None and "payload must be a JSON object" in stderr
            assert session.results == {}
        finally:
            stopped.set()
            await asyncio.wait_for(task, 10)
    asyncio.run(exercise())
