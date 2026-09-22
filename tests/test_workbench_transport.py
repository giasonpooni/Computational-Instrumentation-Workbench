"""Shared source visibility survives the submitting client's disconnection."""
import asyncio
import base64
import json
from pathlib import Path
import threading

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw.cli import parser
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session


def test_observer_sees_committed_source_after_submitter_disconnects(tmp_path):
    async def exercise():
        session = Session(make_demo_run(), tmp_path)
        started, release = threading.Event(), threading.Event()
        retain = session.workbench.add_source

        def delayed(payload):
            started.set()
            if not release.wait(5):
                raise AssertionError("test did not release retention")
            return retain(payload)

        # Delay real source admission; no substitute scientific result.
        session.workbench.add_source = delayed
        bridge = WorkbenchServer(session)
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as observer:
                await observer.recv()
                sender = await connect(f"ws://127.0.0.1:{port}")
                await sender.recv()
                source = (Path(__file__).resolve().parents[1]
                          / "examples/calibrated-observable/source.json").read_bytes()
                try:
                    await sender.send(json.dumps({"protocol_version": 1, "request_id": "retain",
                        "type": "source.add", "payload": {"kind": "calibrated-observable",
                        "label": "shared source", "bytes_b64": base64.b64encode(source).decode()}}))
                    assert await asyncio.to_thread(started.wait, 3)
                    await sender.close()
                finally:
                    release.set()
                    await sender.close()
                event = json.loads(await asyncio.wait_for(observer.recv(), 5))
                assert event["type"] == "workbench.changed"
                assert event["payload"]["session_id"] == session.session_id
                await observer.send(json.dumps({"protocol_version": 1, "request_id": "read",
                                                "type": "session.get", "payload": {}}))
                refreshed = json.loads(await asyncio.wait_for(observer.recv(), 5))
                assert refreshed["request_id"] == "read"
                assert len(refreshed["payload"]["workbench"]["sources"]) == 1
                assert refreshed["payload"]["workbench"] == session.workbench.snapshot()
                # Invalid operation data must remain a protocol refusal, even
                # when mutation notification checks inspect operation IDs.
                await observer.send(json.dumps({"protocol_version": 1, "request_id": "bad",
                    "type": "operation.execute", "payload": {"operation_id": [], "parameters": {}}}))
                refused = json.loads(await asyncio.wait_for(observer.recv(), 5))
                assert refused["type"] == "error"
                await observer.send(json.dumps({"protocol_version": 1, "request_id": "alive",
                                                "type": "source.list", "payload": {}}))
                alive = json.loads(await asyncio.wait_for(observer.recv(), 5))
                assert alive["request_id"] == "alive"
                assert len(alive["payload"]["sources"]) == 1
    asyncio.run(exercise())


def test_workbench_startup_binding_and_request_deadline():
    args = parser().parse_args(["serve", "--identified-stack-root", "trusted-providers"])
    assert args.identified_stack_root == Path("trusted-providers")
    assert args.calibrated_stack_root is None
    assert parser().parse_args(["send", "operation.execute", "--timeout", "300"]).timeout == 300
