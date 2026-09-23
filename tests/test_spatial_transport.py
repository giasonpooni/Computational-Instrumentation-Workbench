"""Real sockets enforce geographic read-only scope without exposing the session."""
import asyncio
import base64
import json
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer, spatial_origins
from ciw.session import Session

ORIGIN = "http://127.0.0.1:5173"
EXAMPLES = Path(__file__).parents[1] / "examples"


def request(kind, payload=None):
    return json.dumps({"protocol_version": 1, "request_id": kind, "type": kind, "payload": payload or {}})


def source_payload(kind, path):
    return {"kind": kind, "label": kind, "bytes_b64": base64.b64encode(path.read_bytes()).decode()}


def test_native_mutation_spatial_observation_and_restore(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        hidden = session.workbench.add_source(source_payload("calibrated-observable", EXAMPLES / "calibrated-observable/source.json"))
        bridge = WorkbenchServer(session, spatial_view_origins=[ORIGIN])
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url) as native, connect(url + "/spatial", origin=ORIGIN) as spatial:
                assert json.loads(await native.recv())["type"] == "session.snapshot"
                hello = json.loads(await spatial.recv())
                assert hello["type"] == "spatial.ready"
                assert hello["payload"]["read_only"] is True
                assert "workbench" not in hello["payload"]
                for kind in ("source.add", "operation.execute", "selection.update", "session.get", "source.get", "workspace.save"):
                    await spatial.send(request(kind))
                    assert json.loads(await spatial.recv())["payload"]["code"] == "read_only_view"
                await spatial.send(request("spatial.inspect", {"source_id": hidden["source_id"]}))
                assert json.loads(await spatial.recv())["type"] == "error"
                payload = source_payload("geographic-context", EXAMPLES / "workbench/geographic-context.json")
                await native.send(request("source.add", payload))
                added = json.loads(await native.recv())["payload"]
                assert json.loads(await native.recv())["type"] == "workbench.changed"
                assert json.loads(await spatial.recv()) == {"protocol_version": 1, "request_id": None, "type": "workbench.changed", "payload": {"session_id": session.session_id}}
                await native.send(request("selection.update", {"expected_revision": 0, "cursor_s": 0.5}))
                assert json.loads(await native.recv())["type"] == "response"
                assert json.loads(await native.recv())["type"] == "selection.changed"
                await spatial.send(request("spatial.list"))
                listing = json.loads(await spatial.recv())
                assert listing["type"] == "response"  # no leaked selection event
                assert [s["source_id"] for s in listing["payload"]["sources"]] == [added["source_id"]]
                await spatial.send(request("spatial.inspect", {"source_id": added["source_id"]}))
                packet = json.loads(await spatial.recv())["payload"]
                assert packet["source"]["bytes_b64"] == payload["bytes_b64"]
                assert packet["authority"]["read_only"] is True
                assert session.workbench.fusion_contexts() == []
                retained = session.workbench.serialize()
                restored = type(session.workbench).restore(retained)
                assert restored.inspect_spatial({"source_id": added["source_id"]}) == packet
                assert {item["operation_id"] for item in restored.describe_operations() if item["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1"}
    asyncio.run(exercise())


@pytest.mark.parametrize("path,origin", [("/", ORIGIN), ("/spatial", "http://unlisted.invalid"), ("/other", None)])
def test_origin_cannot_bypass_endpoint_scope(tmp_path, path, origin):
    async def exercise():
        bridge = WorkbenchServer(Session(make_demo_run(), tmp_path), spatial_view_origins=[ORIGIN])
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1]) + path
            async with connect(url, origin=origin) as client:
                with pytest.raises(ConnectionClosed) as closed:
                    await client.recv()
                assert closed.value.rcvd.code == 1008
    asyncio.run(exercise())


def test_no_origin_does_not_make_spatial_endpoint_writable(tmp_path):
    async def exercise():
        bridge = WorkbenchServer(Session(make_demo_run(), tmp_path))
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            async with connect("ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1]) + "/spatial") as client:
                await client.recv()
                await client.send(request("source.add"))
                assert json.loads(await client.recv())["payload"]["code"] == "read_only_view"
    asyncio.run(exercise())


@pytest.mark.parametrize("origin", ["*", "null", "https://*.example.com", "http://localhost/", "http://a:b@localhost", "ws://localhost", "http://localhost:99999", "http://localhost?x=1"])
def test_browser_origin_must_be_an_exact_http_origin(origin):
    with pytest.raises(ValueError):
        spatial_origins([origin])
