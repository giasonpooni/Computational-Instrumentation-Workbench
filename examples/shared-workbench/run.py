"""Operate the existing shared CIW server through its public session protocol."""
import argparse
import asyncio
import base64
import json
from pathlib import Path

from ciw.cli import request_remote


async def run(url, with_design, with_telemetry=False, with_calibrated_window=False):
    examples = Path(__file__).resolve().parents[1]

    async def call(kind, payload=None):
        response = await request_remote(url, kind, payload or {}, timeout_s=300)
        if response["type"] != "response":
            raise RuntimeError(response["payload"])
        return response["payload"]

    async def source(kind, folder):
        return await call("source.add", {
            "kind": kind, "label": "Synthetic " + kind,
            "bytes_b64": base64.b64encode((examples / folder / "source.json").read_bytes()).decode(),
        })

    retained = await source("calibrated-observable", "calibrated-observable")
    process = await call("operation.execute", {
        "operation_id": "ciw.calibrated-observable.v1",
        "parameters": {"source_id": retained["source_id"]},
    })
    if with_design:
        declared = await source("identified-design", "identified-design")
        await call("operation.execute", {
            "operation_id": "ciw.identified-design.v1", "parameters": {
                "source_id": declared["source_id"], "upstream_bundle_id": process["bundle_id"],
            },
        })
    if with_telemetry:
        acquired = await source("telemetry", "telemetry")
        await call("operation.execute", {"operation_id": "ciw.telemetry.v1", "parameters": {
            "source_id": acquired["source_id"],
            "configuration": json.loads((examples / "telemetry/configuration.json").read_text()),
        }})
    if with_calibrated_window:
        acquired = await source("calibrated-window", "calibrated-window")
        await call("operation.execute", {"operation_id": "ciw.calibrated-window.v1",
            "parameters": {"source_id": acquired["source_id"]}})
    await call("workspace.save")
    print(json.dumps(await call("experiment.inspect", {"view": "fusion"}), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--with-design", action="store_true")
    parser.add_argument("--with-telemetry", action="store_true", help="Also execute the retained PPDA/STFE scalar window in this session")
    parser.add_argument("--with-calibrated-window", action="store_true", help="Also execute TBRT/MCUR/STFE/GSIE in this session")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.with_design, args.with_telemetry, args.with_calibrated_window))
