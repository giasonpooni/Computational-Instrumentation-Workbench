"""Run the declared heat proof through a live CIW session; optionally replay it."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import uuid

from websockets.asyncio.client import connect


async def run(url, replay=False):
    async with connect(url, max_size=33_554_432, proxy=None, open_timeout=5, close_timeout=2) as socket:
        async def call(kind, payload=None):
            request_id = uuid.uuid4().hex
            async with asyncio.timeout(2400):
                await socket.send(json.dumps({"protocol_version": 1, "request_id": request_id,
                    "type": kind, "payload": payload or {}}, allow_nan=False))
                async for raw in socket:
                    response = json.loads(raw)
                    if response.get("request_id") == request_id:
                        if response["type"] != "response":
                            raise RuntimeError(response["payload"])
                        return response["payload"]
            raise RuntimeError("Workbench disconnected before the requested response")

        raw = (Path(__file__).parent / "source.json").read_bytes()
        source = await call("source.add", {"kind": "proved-heat", "label": "Synthetic integer heat proof",
            "bytes_b64": base64.b64encode(raw).decode("ascii")})
        original = await call("operation.execute", {"operation_id": "ciw.proved-heat.v1",
            "parameters": {"source_id": source["source_id"]}})
        result = {"original": original, "view": await call("experiment.inspect", {"bundle_id": original["bundle_id"]})}
        if replay:
            result["replay"] = await call("bundle.replay", {"bundle_id": original["bundle_id"]})
        result["workspace"] = await call("workspace.save")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--replay", action="store_true", help="Perform another native execution and real proof")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url, args.replay)), indent=2, allow_nan=False))
