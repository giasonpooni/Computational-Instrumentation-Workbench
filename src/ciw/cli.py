"""Headless calculations and terminal access to the same live session as Godot."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from websockets.asyncio.client import connect

from .instruments import make_demo_run, validate_run
from .server import run_server
from .session import Session, _reject_constant, read_json, write_json


def print_json(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def load_run(path: Path | None) -> dict:
    run = read_json(path) if path else make_demo_run()
    validate_run(run)
    return run


async def request_remote(url: str, kind: str, payload: dict) -> dict:
    request_id = uuid.uuid4().hex
    async with connect(url, max_size=8_388_608, open_timeout=5, close_timeout=2,
                       proxy=None) as socket:
        await socket.send(json.dumps({"protocol_version": 1, "request_id": request_id,
                                      "type": kind, "payload": payload}, allow_nan=False))
        async with asyncio.timeout(15):
            async for raw in socket:
                response = json.loads(raw, parse_constant=_reject_constant)
                if response.get("request_id") == request_id:
                    return response
    raise RuntimeError("Service disconnected without returning a response")


async def watch_remote(url: str) -> None:
    async with connect(url, max_size=8_388_608, open_timeout=5, close_timeout=2,
                       proxy=None) as socket:
        async for raw in socket:
            event = json.loads(raw, parse_constant=_reject_constant)
            print(json.dumps(event, allow_nan=False), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ciw", description="Computational Instrumentation Workbench")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Save deterministic synthetic oscillator evidence")
    demo.add_argument("--output", type=Path, default=Path("recordings/demo.json"))
    analyze = commands.add_parser("analyze", help="Run headless analysis and save a reopenable workspace")
    analyze.add_argument("operation", choices=["stats", "spectrum"])
    analyze.add_argument("--recording", type=Path)
    analyze.add_argument("--channel", default="q")
    analyze.add_argument("--start", type=float, default=0)
    analyze.add_argument("--end", type=float)
    analyze.add_argument("--output-dir", type=Path, default=Path("results"))
    server = commands.add_parser("serve", help="Start the shared loopback-only instrument session")
    source = server.add_mutually_exclusive_group()
    source.add_argument("--recording", type=Path)
    source.add_argument("--workspace", type=Path)
    server.add_argument("--output-dir", type=Path, default=Path("results"))
    server.add_argument("--port", type=int, default=8765)
    send = commands.add_parser("send", help="Send a structured request to a running session")
    send.add_argument("type", help="For example session.get, analysis.spectrum, selection.update")
    send.add_argument("--payload", default="{}", help="JSON object (or use --payload-file)")
    send.add_argument("--payload-file", type=Path)
    send.add_argument("--url", default="ws://127.0.0.1:8765")
    watch = commands.add_parser("watch", help="Print live selection events as JSON lines")
    watch.add_argument("--url", default="ws://127.0.0.1:8765")
    inspect = commands.add_parser("inspect", help="Inspect a saved result/workspace without executing it")
    inspect.add_argument("path", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            run = make_demo_run()
            write_json(args.output, run)
            print_json({"recording_file": str(args.output), "run_id": run["run_id"],
                        "evidence_id": run["evidence_id"], "sample_count": len(run["time_s"])})
        elif args.command == "analyze":
            run = load_run(args.recording)
            session = Session(run, args.output_dir)
            interval = [args.start, args.end if args.end is not None else run["metadata"]["duration_s"]]
            response = session.handle({"protocol_version": 1, "request_id": uuid.uuid4().hex,
                                       "type": "analysis." + args.operation,
                                       "payload": {"channel": args.channel, "interval_s": interval}})
            print_json(response)
            if response["type"] == "error":
                return 2
            session.save_workspace(args.output_dir / "workspace.json")
        elif args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            session = (Session.from_workspace(args.workspace, args.output_dir) if args.workspace
                       else Session(load_run(args.recording), args.output_dir))
            asyncio.run(run_server(session, args.port))
        elif args.command == "send":
            payload = (read_json(args.payload_file) if args.payload_file else
                       json.loads(args.payload, parse_constant=_reject_constant))
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            response = asyncio.run(request_remote(args.url, args.type, payload))
            print_json(response)
            return 0 if response["type"] == "response" else 2
        elif args.command == "watch":
            asyncio.run(watch_remote(args.url))
        elif args.command == "inspect":
            print_json(read_json(args.path))
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"ciw: {exc}", file=sys.stderr)
        return 2
