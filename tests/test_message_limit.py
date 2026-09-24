"""The transport admits requests up to 8 MiB (a base64 source plus its envelope) and closes on anything larger."""
import asyncio
import json
import socket

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from ciw.instruments import make_demo_run
from ciw.server import run_server
from ciw.session import Session

LIMIT = 8 * 1024 * 1024


def _port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _exchange(tmp_path, size):
    session = Session(make_demo_run(), tmp_path)
    port, stop = _port(), asyncio.Event()
    server = asyncio.create_task(run_server(session, port, stop_event=stop))
    try:
        for _ in range(100):
            try:
                async with connect(f"ws://127.0.0.1:{port}", max_size=None, proxy=None) as client:
                    await client.recv()  # snapshot
                    request = {"protocol_version": 1, "request_id": "big", "type": "session.get", "payload": {}}
                    body = json.dumps(request)
                    padded = body[:-1] + ', "padding": "' + "x" * max(0, size - len(body) - 16) + '"}'
                    await client.send(padded)
                    try:
                        return "reply", json.loads(await asyncio.wait_for(client.recv(), 10))["type"]
                    except ConnectionClosed as closed:
                        return "closed", closed.rcvd.code if closed.rcvd else None
            except OSError:
                await asyncio.sleep(0.05)
        raise AssertionError("server did not start")
    finally:
        stop.set()
        await server


def test_a_request_just_under_the_limit_is_answered(tmp_path):
    outcome, detail = asyncio.run(_exchange(tmp_path, LIMIT - 1024))
    assert outcome == "reply" and detail in {"response", "error"}


def test_a_request_over_the_limit_closes_the_connection_as_too_big(tmp_path):
    outcome, code = asyncio.run(_exchange(tmp_path, LIMIT + 1024))
    assert (outcome, code) == ("closed", 1009)
