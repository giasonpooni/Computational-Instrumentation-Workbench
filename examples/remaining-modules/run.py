"""Execute native measurement, geometry and identified stability on one bench."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import runpy
import uuid

from websockets.asyncio.client import connect


KINDS = ("measurement-chain", "geometric-circle", "identified-stability")


async def _operate(call):
    examples = Path(__file__).resolve().parents[1]

    async def execute(kind, upstream=None, declaration=None):
        raw = ((examples / kind / "source.json").read_bytes() if declaration is None else
               json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        source = await call("source.add", {"kind": kind, "label": "Public synthetic " + kind,
            "bytes_b64": base64.b64encode(raw).decode("ascii")})
        parameters = {"source_id": source["source_id"]}
        if upstream is not None:
            parameters["upstream_bundle_id"] = upstream
        return await call("operation.execute", {"operation_id": "ciw." + kind + ".v1",
            "parameters": parameters})

    # Stability explicitly selects the identified model already retained on
    # this session. Its calibrated GSIE state is never replaced by geometry
    # or a measurement-chain candidate.
    calibrated = await execute("calibrated-observable")
    identified = await execute("identified-design", calibrated["bundle_id"])
    upstream = await call("bundle.get", {"bundle_id": identified["bundle_id"]})
    stability_source = runpy.run_path(str(examples / "identified-stability/make_source.py"))["make_source"](upstream)
    bundles, replays, views = {}, {}, {}
    for kind in KINDS:
        summary = await execute(kind, identified["bundle_id"] if kind == "identified-stability" else None,
                                stability_source if kind == "identified-stability" else None)
        bundles[kind] = summary["bundle_id"]
        replay = await call("bundle.replay", {"bundle_id": summary["bundle_id"]})
        replays[kind] = replay["bundle"]["bundle_id"]
        views[kind] = await call("experiment.inspect", {"bundle_id": summary["bundle_id"]})
    return {"calibrated_bundle_id": calibrated["bundle_id"],
        "identified_bundle_id": identified["bundle_id"], "bundle_ids": bundles,
        "replay_bundle_ids": replays, "views": views,
        "catalog": await call("bundle.list"), "fusion_contexts": await call("fusion.list"),
        "workspace": await call("workspace.save")}


async def run(url):
    async with connect(url, max_size=16_777_216, proxy=None, open_timeout=5, close_timeout=2) as socket:
        async def call(kind, payload=None):
            request_id = uuid.uuid4().hex
            async with asyncio.timeout(600):
                await socket.send(json.dumps({"protocol_version": 1, "request_id": request_id,
                    "type": kind, "payload": payload or {}}, allow_nan=False))
                async for raw in socket:
                    response = json.loads(raw)
                    if response.get("request_id") == request_id:
                        if response["type"] != "response":
                            raise RuntimeError(response["payload"])
                        return response["payload"]
            raise RuntimeError("Workbench disconnected without the requested response")
        return await _operate(call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url)), indent=2, allow_nan=False))
