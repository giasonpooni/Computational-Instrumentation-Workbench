"""Acquire three declared windows and monitor their retained innovations on one server."""
import argparse
import asyncio
import base64
from copy import deepcopy
import json
from pathlib import Path
import runpy
import uuid

from websockets.asyncio.client import connect

from ciw.residual_monitor import DEFAULT_CONFIGURATION
from ciw.telemetry import canonical


async def _operate(call):
    examples = Path(__file__).resolve().parents[1]
    helpers = runpy.run_path(str(examples / "acquired-window/make_source.py"))
    templates = runpy.run_path(str(examples / "acquired-stream/make_sequence.py"))["templates"]()

    async def execute(kind, declaration, upstream=None):
        source = await call("source.add", {"kind": kind, "label": "Synthetic " + declaration["experiment_id"],
            "bytes_b64": base64.b64encode(canonical(declaration)).decode()})
        parameters = {"source_id": source["source_id"]}
        if upstream is not None:
            parameters["upstream_bundle_id"] = upstream
        summary = await call("operation.execute", {"operation_id": "ciw." + kind + ".v1", "parameters": parameters})
        return await call("bundle.get", {"bundle_id": summary["bundle_id"]})

    acquired = await execute("acquired-dataset", helpers["build_acquisition"](templates))
    windows = []
    for index, template in enumerate(templates):
        mapping = helpers["build_mapping"](acquired, template, index)
        windows.append(await execute("acquired-calibrated-window", mapping, acquired["bundle_digest"]))
    monitor = await execute("residual-monitor", {
        "schema": "ciw.residual-monitor-source.v1",
        "experiment_id": "experiment:synthetic-acquired-stream-monitor",
        "window_bundle_ids": [window["bundle_digest"] for window in windows],
        "configuration": deepcopy(DEFAULT_CONFIGURATION),
    })
    replay = await call("bundle.replay", {"bundle_id": monitor["bundle_digest"]})
    saved = await call("workspace.save")
    view = await call("experiment.inspect", {"bundle_id": monitor["bundle_digest"]})
    return {"acquisition_bundle_id": acquired["bundle_digest"],
        "window_bundle_ids": [window["bundle_digest"] for window in windows],
        "monitor_bundle_id": monitor["bundle_digest"], "monitor_replay": replay,
        "workspace": saved, "monitor_view": view}


async def run(url):
    # Keep one connection open: catalog updates and results arrive on the same
    # session, and each request remains paired with its own occurrence identity.
    async with connect(url, max_size=8_388_608, proxy=None, open_timeout=5, close_timeout=2) as socket:
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
