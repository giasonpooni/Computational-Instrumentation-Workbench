"""Submit both native workloads to the same existing CIW session."""
import argparse
import asyncio
import base64
import json
from pathlib import Path

from ciw.cli import request_remote


async def main(url, replay):
    root = Path(__file__).resolve().parent
    async def call(kind, payload):
        response = await request_remote(url, kind, payload, timeout_s=120)
        if response["type"] != "response":
            raise RuntimeError(json.dumps(response))
        return response["payload"]
    for kind in ("schematic-assessment", "numerical-heat"):
        source = await call("source.add", {"kind": kind, "label": kind,
            "bytes_b64": base64.b64encode((root / (kind + ".json")).read_bytes()).decode()})
        bundle = await call("operation.execute", {"operation_id": "ciw." + kind + ".v1", "parameters": {"source_id": source["source_id"]}})
        print(json.dumps(bundle, indent=2))
        if replay:
            print(json.dumps(await call("bundle.replay", {"bundle_id": bundle["bundle_id"]}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.replay))
