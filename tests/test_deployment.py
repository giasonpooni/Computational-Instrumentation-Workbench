"""Deployment entry points, bounded health, and graceful workspace persistence."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import sys
from unittest.mock import AsyncMock, patch

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from ciw.cli import health_remote, load_server_session, main, parser, request_remote
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer, run_server
from ciw.session import Session, read_json


def available_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def command_environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return environment


def arguments(output_dir, *extra):
    return parser().parse_args(["serve", "--output-dir", str(output_dir), *extra])


def test_bind_defaults_to_loopback_and_requires_explicit_container_selection(tmp_path):
    assert arguments(tmp_path).bind == "127.0.0.1"
    assert arguments(tmp_path, "--bind", "0.0.0.0").bind == "0.0.0.0"
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--bind", "other-host")


@pytest.mark.parametrize("other", ["--recording", "--workspace"])
def test_resume_is_mutually_exclusive_with_explicit_source(tmp_path, other):
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--resume", other, "saved.json")


def test_resume_creates_demo_only_when_workspace_is_absent(tmp_path):
    session = load_server_session(arguments(tmp_path, "--resume"))
    assert session.run["instrument"] == "analytic-damped-oscillator.v1"
    assert session.results == {}


def test_resume_reopens_exact_evidence_and_results_without_computation(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    selected = session.handle({"protocol_version": 1, "request_id": "selection",
                               "type": "selection.update", "payload": {
                                   "expected_revision": 0, "channel": "v", "cursor_s": 0.5}})
    assert selected["type"] == "response"
    result = session.handle({"protocol_version": 1, "request_id": "stats",
                             "type": "analysis.stats", "payload": {}})["payload"]
    session.save_workspace(tmp_path / "workspace.json")
    with patch("ciw.cli.make_demo_run", side_effect=AssertionError("no new evidence")), \
            patch("ciw.session.compute_statistics", side_effect=AssertionError("no rerun")), \
            patch("ciw.session.compute_spectrum", side_effect=AssertionError("no rerun")):
        resumed = load_server_session(arguments(tmp_path, "--resume"))
    assert resumed.run == session.run
    assert resumed.selection == session.selection
    assert resumed.results == {result["result_id"]: result}


def test_existing_bad_workspace_fails_instead_of_replacing_it_with_demo(tmp_path):
    workspace = tmp_path / "workspace.json"
    workspace.write_text('{"workspace_version": 9000}', encoding="utf-8")
    with patch("ciw.cli.make_demo_run", side_effect=AssertionError("must not fall back")):
        assert main(["serve", "--resume", "--output-dir", str(tmp_path)]) == 2
    assert read_json(workspace) == {"workspace_version": 9000}
    assert list(tmp_path.iterdir()) == [workspace]


def test_health_cli_succeeds_for_a_real_live_session(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        bridge = WorkbenchServer(session)
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ciw", "health", "--url", f"ws://127.0.0.1:{port}",
                env=command_environment(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, stderr.decode()
            assert json.loads(stdout) == {"status": "healthy", "session_id": session.session_id,
                                          "run_id": session.run["run_id"]}
            assert session.results == {}
    asyncio.run(exercise())


def test_health_cli_has_a_bounded_failure_when_server_does_not_reply():
    async def silent(websocket):
        await websocket.wait_closed()

    async def exercise():
        async with serve(silent, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ciw", "health", "--url", f"ws://127.0.0.1:{port}",
                env=command_environment(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), 8)
                assert process.returncode == 2
                assert b"ciw:" in stderr
                assert not stdout
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
    asyncio.run(exercise())


def test_health_cli_rejects_non_websocket_urls():
    assert main(["health", "--url", "file:///not-a-service"]) == 2


@pytest.mark.parametrize("case", ["error", "version", "session", "selection", "bounds", "results"])
def test_health_rejects_invalid_session_responses(tmp_path, case):
    session = Session(make_demo_run(), tmp_path)
    response = {"protocol_version": 1, "request_id": "health", "type": "response",
                "payload": deepcopy(session.snapshot())}
    if case == "error":
        response["type"] = "error"
    elif case == "version":
        response["protocol_version"] = True
    elif case == "session":
        response["payload"]["session_id"] = None
    elif case == "selection":
        response["payload"]["selection"]["run_id"] = "different"
    elif case == "bounds":
        response["payload"]["selection"]["cursor_s"] = float("nan")
    else:
        response["payload"]["results"] = "not-an-index"
    with patch("ciw.cli.request_remote", new=AsyncMock(return_value=response)):
        with pytest.raises(ValueError):
            asyncio.run(health_remote("ws://127.0.0.1:8765"))


def test_graceful_event_shutdown_saves_selection_and_exact_result(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        stopped = asyncio.Event()
        port = available_port()
        url = f"ws://127.0.0.1:{port}"
        task = asyncio.create_task(run_server(session, port, stop_event=stopped))
        try:
            for attempt in range(100):
                try:
                    await request_remote(url, "session.get", {})
                    break
                except OSError:
                    await asyncio.sleep(0.02)
            else:
                pytest.fail("Server did not start")
            selection = (await request_remote(url, "selection.update", {
                "expected_revision": 0, "channel": "v", "cursor_s": 0.25,
                "interval_s": [0, 6]}))["payload"]
            result = (await request_remote(url, "analysis.spectrum", {}))["payload"]
        finally:
            stopped.set()
            await asyncio.wait_for(task, 10)
        workspace = read_json(tmp_path / "workspace.json")
        assert workspace["selection"] == selection
        assert workspace["results"] == [result]
        with pytest.raises(OSError):
            await request_remote(url, "session.get", {})
    asyncio.run(exercise())


def test_oversized_frames_close_only_the_offending_client(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        stopped = asyncio.Event()
        port = available_port()
        url = f"ws://127.0.0.1:{port}"
        task = asyncio.create_task(run_server(session, port, stop_event=stopped))
        try:
            for attempt in range(100):
                try:
                    await request_remote(url, "session.get", {})
                    break
                except OSError:
                    await asyncio.sleep(0.02)
            else:
                pytest.fail("Server did not start")
            async with connect(url, max_size=None, proxy=None) as offender, connect(url, proxy=None) as observer:
                assert json.loads(await offender.recv())["type"] == "session.snapshot"
                assert json.loads(await observer.recv())["type"] == "session.snapshot"
                # Above the websockets default but within the served 8 MiB limit:
                # the frame is read and answered as an invalid request.
                await offender.send("x" * (1024 * 1024 + 1))
                reply = json.loads(await asyncio.wait_for(offender.recv(), 5))
                assert reply["type"] == "error" and reply["payload"]["code"] == "invalid_request"
                # Above the served limit: only the offending connection is closed as too big.
                with pytest.raises(ConnectionClosedError) as closed:
                    await offender.send("x" * (8 * 1024 * 1024 + 1))
                    await asyncio.wait_for(offender.recv(), 5)
                assert closed.value.rcvd is not None and closed.value.rcvd.code == 1009
                await observer.send(json.dumps({"protocol_version": 1, "request_id": "after-oversize",
                                                "type": "session.get", "payload": {}}))
                reply = json.loads(await asyncio.wait_for(observer.recv(), 5))
                assert reply["type"] == "response" and reply["request_id"] == "after-oversize"
        finally:
            stopped.set()
            await asyncio.wait_for(task, 10)
    asyncio.run(exercise())


def test_shutdown_save_failure_propagates_and_is_logged(tmp_path, caplog):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        stopped = asyncio.Event()
        stopped.set()
        with patch.object(session, "save_workspace", side_effect=OSError("disk unavailable")):
            with pytest.raises(OSError, match="disk unavailable"):
                await run_server(session, port=0, stop_event=stopped)
    asyncio.run(exercise())
    assert "Unable to save workspace during shutdown" in caplog.text


@pytest.mark.skipif(os.name == "nt", reason="Windows terminate() does not deliver POSIX process signals")
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_posix_signal_shutdown_and_restart_preserve_exact_saved_json(tmp_path, stop_signal):
    async def exercise():
        port = available_port()
        url = f"ws://127.0.0.1:{port}"

        async def start():
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ciw", "serve", "--resume", "--output-dir", str(tmp_path),
                "--port", str(port), env=command_environment(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                banner = await asyncio.wait_for(process.stdout.readline(), 15)
                assert f"ws://127.0.0.1:{port}" in banner.decode()
                return process
            except BaseException:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise

        async def stop(process):
            process.send_signal(stop_signal)
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), 15)
                assert process.returncode == 0, stderr.decode()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        first = await start()
        try:
            selected = (await request_remote(url, "selection.update", {
                "expected_revision": 0, "channel": "energy", "cursor_s": 1,
                "interval_s": [1, 4]}))["payload"]
            result = (await request_remote(url, "analysis.stats", {}))["payload"]
        finally:
            await stop(first)
        original = read_json(tmp_path / "workspace.json")
        assert original["selection"] == selected
        assert original["results"] == [result]
        second = await start()
        try:
            snapshot = (await request_remote(url, "session.get", {}))["payload"]
            assert snapshot["selection"] == selected
            assert len(snapshot["results"]) == 1
            reopened = (await request_remote(url, "result.get", {"result_id": result["result_id"]}))["payload"]
            assert reopened == result
        finally:
            await stop(second)
        saved_again = read_json(tmp_path / "workspace.json")
        assert saved_again["run"] == original["run"]
        assert saved_again["selection"] == original["selection"]
        assert saved_again["results"] == original["results"]
    asyncio.run(exercise())
