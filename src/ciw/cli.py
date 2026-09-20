"""Headless calculations and terminal access to the same live session as Godot."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import uuid
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .instruments import make_demo_run, validate_run
from .server import run_server
from .session import Session, _reject_constant, read_json, write_json


def print_json(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def load_run(path: Path | None) -> dict:
    run = read_json(path) if path else make_demo_run()
    validate_run(run)
    return run


async def request_remote(url: str, kind: str, payload: dict, *, timeout_s: float = 15) -> dict:
    request_id = uuid.uuid4().hex
    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=8_388_608, open_timeout=min(5, timeout_s), close_timeout=1,
                           proxy=None) as socket:
            await socket.send(json.dumps({"protocol_version": 1, "request_id": request_id,
                                          "type": kind, "payload": payload}, allow_nan=False))
            async for raw in socket:
                if not isinstance(raw, str):
                    raise ValueError("Protocol v1 requires a text JSON response")
                response = json.loads(raw, parse_constant=_reject_constant)
                if not isinstance(response, dict):
                    raise ValueError("Service returned a non-object response")
                if response.get("request_id") == request_id:
                    return response
    raise RuntimeError("Service disconnected without returning a response")


async def health_remote(url: str) -> dict:
    """Probe session.get with a three-second request budget and bounded close."""
    try:
        response = await request_remote(url, "session.get", {}, timeout_s=3)
    except TimeoutError as exc:
        raise RuntimeError("Health probe timed out waiting for session.get") from exc
    if (type(response.get("protocol_version")) is not int or response["protocol_version"] != 1
            or response.get("type") != "response"):
        raise ValueError("Health probe did not receive a protocol v1 session response")
    payload = response.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("session_id"), str) or not payload["session_id"]:
        raise ValueError("Health probe received an invalid session snapshot")
    run, selection = payload.get("run"), payload.get("selection")
    if not isinstance(run, dict) or not isinstance(selection, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Health probe received an incomplete session snapshot")
    if not all(isinstance(run.get(key), str) and run[key] for key in ("run_id", "evidence_id", "instrument")):
        raise ValueError("Health probe received invalid recording identity")
    metadata, channels = run.get("metadata"), run.get("channels")
    if not isinstance(metadata, dict) or not isinstance(channels, dict) or not channels:
        raise ValueError("Health probe received invalid recording metadata")
    if (selection.get("run_id") != run["run_id"]
            or not isinstance(selection.get("channel"), str) or selection["channel"] not in channels
            or not isinstance(metadata.get("coordinate_frame"), str)
            or selection.get("coordinate_frame") != metadata["coordinate_frame"]
            or type(selection.get("revision")) is not int or selection["revision"] < 0):
        raise ValueError("Health probe received an invalid shared selection")
    try:
        duration, rate, cursor = metadata["duration_s"], metadata["sample_rate_hz"], selection["cursor_s"]
        interval = selection["interval_s"]
        count = metadata["sample_count"]
        if not isinstance(interval, list) or len(interval) != 2 or type(count) is not int or count < 2:
            raise ValueError
        numbers = (duration, rate, cursor, *interval)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
            raise ValueError
        if not (duration > 0 and rate > 0 and 0 <= cursor <= duration
                and 0 <= interval[0] < interval[1] <= duration):
            raise ValueError
        if (not math.isclose(duration, count / rate, rel_tol=1e-8)
                or cursor > (count - 1) / rate):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Health probe received invalid recording bounds") from exc
    return {"status": "healthy", "session_id": payload["session_id"], "run_id": run["run_id"]}


def load_server_session(args: argparse.Namespace) -> Session:
    if args.workspace:
        return Session.from_workspace(args.workspace, args.output_dir)
    if args.resume:
        workspace = args.output_dir / "workspace.json"
        try:
            workspace.stat()
        except FileNotFoundError:
            pass
        else:
            # Corrupt or unreadable saved state is an error, never a reason to
            # silently replace the workspace with a freshly generated demo.
            return Session.from_workspace(workspace, args.output_dir)
    return Session(load_run(args.recording), args.output_dir)


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
    server = commands.add_parser("serve", help="Start the shared local instrument session")
    source = server.add_mutually_exclusive_group()
    source.add_argument("--recording", type=Path)
    source.add_argument("--workspace", type=Path)
    source.add_argument("--resume", action="store_true", help="Reopen output-dir/workspace.json if present")
    server.add_argument("--output-dir", type=Path, default=Path("results"))
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--bind", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1",
                        help="Use 0.0.0.0 explicitly inside a container; default stays loopback")
    health = commands.add_parser("health", help="Check a live session with a bounded read-only request")
    health.add_argument("--url", default="ws://127.0.0.1:8765")
    send = commands.add_parser("send", help="Send a structured request to a running session")
    send.add_argument("type", help="For example session.get, analysis.spectrum, selection.update")
    send.add_argument("--payload", default="{}", help="JSON object (or use --payload-file)")
    send.add_argument("--payload-file", type=Path)
    send.add_argument("--url", default="ws://127.0.0.1:8765")
    watch = commands.add_parser("watch", help="Print live selection events as JSON lines")
    watch.add_argument("--url", default="ws://127.0.0.1:8765")
    inspect = commands.add_parser("inspect", help="Inspect a saved result/workspace without executing it")
    inspect.add_argument("path", type=Path)
    plsr = commands.add_parser("plsr", help="Import, evaluate, inspect and replay pinned Lyapunov artifacts")
    actions = plsr.add_subparsers(dest="plsr_command", required=True)
    import_model = actions.add_parser("import", help="Validate and retain a sealed model artifact")
    import_model.add_argument("model", type=Path)
    import_model.add_argument("--output", type=Path, required=True)
    evaluate = actions.add_parser("evaluate", help="Evaluate explicit JSON sample inputs and save evidence")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--sample", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    inspect_plsr = actions.add_parser("inspect", help="Validate a saved PLSR run without reevaluating it")
    inspect_plsr.add_argument("path", type=Path)
    replay = actions.add_parser("replay", help="Reevaluate saved inputs; retain new evidence and comparison")
    replay.add_argument("path", type=Path)
    replay.add_argument("--output-dir", type=Path, required=True)
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
            session = load_server_session(args)
            asyncio.run(run_server(session, args.port, args.bind))
        elif args.command == "health":
            print_json(asyncio.run(health_remote(args.url)))
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
        elif args.command == "plsr":
            # The optional engine is loaded only through this terminal boundary.
            from .plsr import evaluate_run, import_model, inspect_run, replay_run
            if args.plsr_command == "import":
                result = import_model(args.model, args.output)
            elif args.plsr_command == "evaluate":
                result = evaluate_run(args.model, args.sample, args.output_dir)
            elif args.plsr_command == "inspect":
                result = inspect_run(args.path)
            else:
                result = replay_run(args.path, args.output_dir)
            print_json(result)
            if args.plsr_command == "replay" and not result["bundle"]["replay_of"]["record_digest_matches"]:
                return 3
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, WebSocketException) as exc:
        print(f"ciw: {exc}", file=sys.stderr)
        return 2
