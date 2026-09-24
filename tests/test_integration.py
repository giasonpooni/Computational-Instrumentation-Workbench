"""Real loopback websocket clients exercise client independence and recovery."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError

from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session


def request(kind, payload=None, request_id="test-request", version=1):
    return {
        "protocol_version": version,
        "request_id": request_id,
        "type": kind,
        "payload": {} if payload is None else payload,
    }


class WorkbenchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.directory.name)
        self.run = make_demo_run()
        self.session = Session(self.run, self.output_dir)
        self.service = WorkbenchServer(self.session)
        self.server = await serve(self.service.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.uri = f"ws://127.0.0.1:{port}"
        self.clients = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        self.server.close()
        await self.server.wait_closed()
        self.directory.cleanup()

    async def receive(self, client):
        return json.loads(await asyncio.wait_for(client.recv(), timeout=5))

    async def new_client(self):
        client = await connect(self.uri, max_size=8 * 1024 * 1024)
        self.clients.append(client)
        greeting = await self.receive(client)
        self.assertEqual(greeting["type"], "session.snapshot")
        self.assertEqual(greeting["protocol_version"], 1)
        self.assertIsNone(greeting["request_id"])
        return client, greeting["payload"]

    async def call(self, client, kind, payload=None, request_id="test-request", version=1):
        await client.send(json.dumps(request(kind, payload, request_id, version)))
        response = await self.receive(client)
        self.assertEqual(response["request_id"], request_id)
        return response

    async def run_cli(self, *args):
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "ciw", *map(str, args),
            cwd=root, env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        self.assertEqual(process.returncode, 0, stderr.decode())
        return json.loads(stdout)

    async def update(self, sender, observer, payload, request_id):
        await sender.send(json.dumps(request("selection.update", payload, request_id)))
        sender_messages = [await self.receive(sender), await self.receive(sender)]
        responses = [message for message in sender_messages if message["type"] == "response"]
        events = [message for message in sender_messages if message["type"] == "selection.changed"]
        self.assertEqual(len(responses), 1, sender_messages)
        self.assertEqual(len(events), 1, sender_messages)
        self.assertEqual(responses[0]["request_id"], request_id)
        self.assertIsNone(events[0]["request_id"])
        peer_event = await self.receive(observer)
        self.assertEqual(peer_event["type"], "selection.changed")
        self.assertEqual(events[0]["payload"], responses[0]["payload"])
        self.assertEqual(peer_event["payload"], responses[0]["payload"])
        return responses[0]["payload"]

    async def test_two_clients_share_selection_and_reconnect_after_disconnect(self):
        first, initial = await self.new_client()
        second, second_initial = await self.new_client()
        self.assertEqual(initial, second_initial)
        duration = self.run["metadata"]["duration_s"]
        interval = [duration / 4, duration / 2]
        selected = await self.update(first, second, {"expected_revision": 0, "interval_s": interval, "channel": "v"}, "interval")
        moved = await self.update(second, first, {"expected_revision": 1, "cursor_s": 3 * duration / 4}, "cursor")
        self.assertEqual(moved["interval_s"], selected["interval_s"])
        self.assertEqual(moved["channel"], selected["channel"])
        self.assertEqual(moved["revision"], 2)
        conflict = await self.call(first, "selection.update", {"expected_revision": 0, "cursor_s": 0}, "stale")
        self.assertEqual(conflict["type"], "error")
        self.assertEqual(conflict["payload"]["code"], "revision_conflict")
        await second.close()
        replacement, refreshed = await self.new_client()
        self.assertEqual(refreshed["selection"], moved)
        self.assertEqual(refreshed["session_id"], initial["session_id"])
        await replacement.close()
        surviving = await self.call(first, "session.get")
        self.assertEqual(surviving["type"], "response")
        self.assertEqual(surviving["payload"]["selection"], moved)

    async def test_headless_and_websocket_analysis_return_identical_numbers(self):
        headless_dir = self.output_dir / "headless"
        headless_response = await self.run_cli("analyze", "stats", "--output-dir", headless_dir)
        self.assertEqual(headless_response["type"], "response")
        headless = headless_response["payload"]
        client, _ = await self.new_client()
        response = await self.call(client, "analysis.stats")
        self.assertEqual(response["type"], "response")
        remote = response["payload"]
        announced = await self.receive(client)
        self.assertEqual(announced["type"], "result.created")
        self.assertEqual(announced["payload"]["result_id"], remote["result_id"])
        self.assertEqual(remote["data"], headless["data"])
        self.assertEqual(remote["evidence_id"], headless["evidence_id"])
        self.assertNotEqual(remote["execution_id"], headless["execution_id"])
        self.assertNotEqual(remote["result_id"], headless["result_id"])
        records = [json.loads(path.read_text(encoding="utf-8")) for path in self.output_dir.rglob("*.json")]
        self.assertIn(headless, records)
        self.assertIn(remote, records)
        restored = await self.call(client, "result.get", {"result_id": remote["result_id"]}, "restore")
        self.assertEqual(restored["payload"], remote)
        workspace = headless_dir / "workspace.json"
        self.assertTrue(workspace.is_file())
        inspected = await self.run_cli("inspect", workspace)
        self.assertEqual(inspected["results"], [headless])
        cli_response = await self.run_cli("send", "session.get", "--url", self.uri)
        self.assertEqual(cli_response["type"], "response")
        self.assertEqual(cli_response["payload"], self.session.snapshot())

    async def test_new_results_are_announced_to_every_client_as_summaries(self):
        producer, _ = await self.new_client()
        observer, _ = await self.new_client()
        response = await self.call(producer, "analysis.spectrum")
        self.assertEqual(response["type"], "response")
        result = response["payload"]
        summary = {key: result[key] for key in ("result_id", "operation_id", "execution_id", "channel", "interval_s",
                                                "created_at", "verification_status", "selection_revision")}
        for client in (producer, observer):
            event = await self.receive(client)
            self.assertEqual(event["type"], "result.created")
            self.assertIsNone(event["request_id"])
            self.assertEqual(event["payload"], summary)
            self.assertNotIn("data", event["payload"])
        # A refused generic operation creates no result and announces nothing, so the
        # observer's next message is its own response.
        refused = await self.call(producer, "operation.execute", {"operation_id": "missing.v1", "parameters": {}}, "missing")
        self.assertEqual(refused["payload"]["status"], "refused")
        listing = await self.call(observer, "result.list", request_id="listing")
        self.assertIn(result["result_id"], [item["result_id"] for item in listing["payload"]["results"]])
        completed = await self.call(producer, "operation.execute", {"operation_id": "statistics.v1", "parameters": {}}, "generic")
        self.assertEqual(completed["payload"]["status"], "completed")
        event = await self.receive(observer)
        self.assertEqual(event["type"], "result.created")
        self.assertEqual(event["payload"]["result_id"], completed["payload"]["result"]["result_id"])

    async def test_bad_messages_do_not_terminate_connection_or_change_selection(self):
        client, original = await self.new_client()
        for raw in ["{", "[]", "null", b"binary-not-v1", '{"protocol_version":1,"request_id":"nan","type":"selection.update","payload":{"expected_revision":0,"cursor_s":NaN}}']:
            with self.subTest(raw=raw):
                await client.send(raw)
                response = await self.receive(client)
                self.assertEqual(response["type"], "error", response)
                self.assertTrue(response["payload"]["code"])
        for kind, payload, version in [
            ("unknown.operation", {}, 1),
            ("session.get", {}, 999),
            ("selection.update", {"expected_revision": 0, "cursor_s": -1}, 1),
        ]:
            response = await self.call(client, kind, payload, version=version)
            self.assertEqual(response["type"], "error", response)
        after = await self.call(client, "session.get", request_id="still-alive")
        self.assertEqual(after["type"], "response")
        self.assertEqual(after["payload"], original)

    async def test_concurrent_edits_accept_one_revision_and_notify_both_clients(self):
        first, _ = await self.new_client()
        second, _ = await self.new_client()
        await asyncio.gather(
            first.send(json.dumps(request("selection.update", {"expected_revision": 0, "cursor_s": 0.5}, "first"))),
            second.send(json.dumps(request("selection.update", {"expected_revision": 0, "cursor_s": 1.0}, "second"))),
        )
        first_messages = [await self.receive(first), await self.receive(first)]
        second_messages = [await self.receive(second), await self.receive(second)]
        all_messages = first_messages + second_messages
        responses = [message for message in all_messages if message["type"] == "response"]
        errors = [message for message in all_messages if message["type"] == "error"]
        self.assertEqual(len(responses), 1, all_messages)
        self.assertEqual(len(errors), 1, all_messages)
        self.assertEqual(errors[0]["payload"]["code"], "revision_conflict")
        for messages in (first_messages, second_messages):
            events = [message for message in messages if message["type"] == "selection.changed"]
            self.assertEqual(len(events), 1, messages)
            self.assertEqual(events[0]["payload"], self.session.selection)
            self.assertEqual(events[0]["payload"]["revision"], 1)

    async def test_reopened_workspace_results_are_discoverable_by_fresh_clients(self):
        original = self.session.handle(request("analysis.spectrum"))["payload"]
        workspace = self.output_dir / "workspace.json"
        self.session.save_workspace(workspace)
        restored = Session.from_workspace(workspace, self.output_dir / "reopened")
        bridge = WorkbenchServer(restored)
        async with serve(bridge.handler, "127.0.0.1", 0) as server:
            self.uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            client, snapshot = await self.new_client()
            self.assertEqual(len(snapshot["results"]), 1)
            self.assertEqual(snapshot["results"][0]["result_id"], original["result_id"])
            self.assertNotIn("data", snapshot["results"][0])
            listing = await self.call(client, "result.list")
            self.assertEqual(listing["type"], "response")
            self.assertEqual(listing["payload"]["results"], snapshot["results"])
            result = await self.call(client, "result.get", {"result_id": snapshot["results"][0]["result_id"]})
            self.assertEqual(result["payload"], original)

    async def test_committed_selection_reaches_peers_if_sender_disconnects_before_reply(self):
        class DisconnectingSender:
            def __init__(self):
                from websockets.datastructures import Headers
                from websockets.http11 import Request
                self.request = Request(path="/", headers=Headers())
                self.sent = 0

            async def send(self, message):
                self.sent += 1
                if self.sent > 1:
                    raise ConnectionClosedError(None, None)

            async def close(self, **kwargs):
                pass

            def __aiter__(self):
                async def requests():
                    yield json.dumps(request("selection.update", {"expected_revision": 0, "cursor_s": 0.5}))
                return requests()

        class ObservingPeer:
            def __init__(self):
                self.messages = []

            async def send(self, message):
                self.messages.append(json.loads(message))

            async def close(self, **kwargs):
                pass

        peer = ObservingPeer()
        self.service.clients.add(peer)
        await self.service.handler(DisconnectingSender())
        self.assertEqual(self.session.selection["revision"], 1)
        self.assertEqual(len(peer.messages), 1)
        self.assertEqual(peer.messages[0]["type"], "selection.changed")
        self.assertEqual(peer.messages[0]["payload"], self.session.selection)


if __name__ == "__main__":
    unittest.main()
