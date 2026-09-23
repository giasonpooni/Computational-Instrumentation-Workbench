"""Run the retained synthetic free-energy comparison in a shared workbench."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import uuid

from websockets.asyncio.client import connect

CASES = ("baseline", "correlated-noise", "ignored-correlation", "sensor-bias", "wrong-curvature", "unstable-step")


async def run(url, case="baseline"):
    if case not in (*CASES,"all"):
        raise ValueError("Unknown demonstration case")
    async with connect(url, max_size=16_777_216, proxy=None, open_timeout=5, close_timeout=2) as socket:
        async def call(kind, payload=None):
            request_id = uuid.uuid4().hex
            async with asyncio.timeout(180):
                await socket.send(json.dumps({"protocol_version": 1, "request_id": request_id,
                    "type": kind, "payload": payload or {}}, allow_nan=False))
                async for raw in socket:
                    response = json.loads(raw)
                    if response.get("request_id") == request_id:
                        if response["type"] != "response":
                            raise RuntimeError(response["payload"])
                        return response["payload"]
            raise RuntimeError("Workbench disconnected before returning the requested response")

        results = {}
        for name in (CASES if case == "all" else (case,)):
            kind, filename = "variational-free-energy", name + ".json"
            raw = (Path(__file__).parent / filename).read_bytes()
            source = await call("source.add", {"kind": kind, "label": "Synthetic free-energy demonstration: " + name,
                "bytes_b64": base64.b64encode(raw).decode("ascii")})
            original = await call("operation.execute", {"operation_id": "ciw." + kind + ".v1",
                "parameters": {"source_id": source["source_id"]}})
            replay = await call("bundle.replay", {"bundle_id": original["bundle_id"]})
            results[name] = {"original": original["bundle_id"], "replay": replay["bundle"]["bundle_id"],
                "view": await call("experiment.inspect", {"bundle_id": original["bundle_id"]})}
        return {"experiments": results, "workspace": await call("workspace.save")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--case", choices=(*CASES,"all"), default="baseline")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url,args.case)), indent=2, allow_nan=False))
