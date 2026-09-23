"""Retain and inspect an energy log through the shared workbench protocol.

The bundled baseline/reset/missing/under-target logs are synthetic fixtures.
Passing a captured log.json retains that operator record; this client never
starts GPU work, authenticates hardware evidence or changes device settings.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import uuid

from websockets.asyncio.client import connect

DEFAULT_LOG = Path(__file__).with_name("baseline.json")


async def run(url, log_path=DEFAULT_LOG, output_dir=None):
    path = Path(log_path)
    raw = path.read_bytes()
    destination = None if output_dir is None else Path(output_dir)
    if destination is not None:
        destination.mkdir(parents=True,exist_ok=False)
    async with connect(url,max_size=16_777_216,proxy=None,open_timeout=5,close_timeout=2) as socket:
        async def call(kind,payload=None):
            request_id = uuid.uuid4().hex
            async with asyncio.timeout(180):
                await socket.send(json.dumps({"protocol_version":1,"request_id":request_id,
                    "type":kind,"payload":payload or {}},allow_nan=False))
                async for message in socket:
                    response = json.loads(message)
                    if response.get("request_id") == request_id:
                        if response["type"] != "response":
                            raise RuntimeError(response["payload"])
                        return response["payload"]
            raise RuntimeError("Workbench disconnected before returning the requested response")

        source = await call("source.add",{"kind":"energy-accuracy","label":"Retained energy/accuracy: "+path.stem,
            "bytes_b64":base64.b64encode(raw).decode("ascii")})
        original = await call("operation.execute",{"operation_id":"ciw.energy-accuracy.v1",
            "parameters":{"source_id":source["source_id"]}})
        replay = await call("bundle.replay",{"bundle_id":original["bundle_id"]})
        view = await call("experiment.inspect",{"bundle_id":original["bundle_id"]})
        workspace = await call("workspace.save")
        report = {"source_id":source["source_id"],"original":original["bundle_id"],
            "replay":replay["bundle"]["bundle_id"],"view":view,"workspace":workspace}
        if destination is not None:
            with (destination/"report.json").open("x",encoding="utf-8") as output:
                json.dump(report,output,indent=2,allow_nan=False)
                output.write("\n")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="ws://127.0.0.1:8765")
    parser.add_argument("--log",type=Path,default=DEFAULT_LOG,
                        help="A captured log.json or an explicitly synthetic example log")
    parser.add_argument("--output-dir",type=Path,
                        help="New directory in which to retain the shared-client report.json")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url,args.log,args.output_dir)),indent=2,allow_nan=False))
