"""Headless calculations and terminal access to the same live session as Godot."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .adapters.protocol import AdapterRefusal
from .instruments import make_demo_run, validate_run
from .server import run_server
from .session import Session, _reject_constant, read_json, write_json


def print_json(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _write_new_proof_report(path: Path, report: dict) -> None:
    """Publish a complete report without replacing a concurrent writer's file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ciw-proof-report-", dir=path.parent) as directory:
        staged = write_json(Path(directory) / "report.json", report)
        # Same-filesystem hard-link creation is atomic and fails if the target
        # exists. The staging link is removed on either success or refusal.
        os.link(staged, path)


def _print_matrix(name: str, matrix: list, order: list | None = None, unit: object = None) -> None:
    """Render every covariance entry; never substitute marginal sigmas."""
    print(f"Covariance: {name}")
    if order:
        print("  Axis order: " + ", ".join(map(str, order)))
    if unit:
        if isinstance(unit, list):
            print("  Axis units: " + json.dumps(unit) + "; entry (i, j) uses unit[i] * unit[j]")
        else:
            print("  Units: " + (json.dumps(unit) if isinstance(unit, dict) else str(unit)))
    for row in matrix:
        print("  [" + ", ".join(format(value, ".12g") for value in row) + "]")


def print_investigation(summary: dict) -> None:
    """Present retained states without interpreting or calculating domain values."""
    print(f"Investigation: {summary['status']}")
    if summary.get("workspace_file"):
        print(f"Workspace: {summary['workspace_file']}")
    if summary.get("run_id"):
        print(f"Run: {summary['run_id']}")
    rows = [["State", "Quantity", "Value", "Unit"]]
    for row in summary.get("terminal_rows", []):
        value = row.get("value")
        rendered = (format(value, ".12g") if type(value) in (int, float)
                    else json.dumps(value, allow_nan=False, ensure_ascii=False))
        rows.append([str(row.get("state", "")), str(row.get("quantity", "")),
                     rendered, str(row.get("unit", ""))])
    if len(rows) > 1:
        widths = [max(len(row[index]) for row in rows) for index in range(4)]
        print()
        for row in rows:
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
    for status in summary.get("calibration", []):
        acquisition, serving = status["acquisition"], status["serving"]
        print(f"Calibration: {status['source']} / {status['calibration_id']}")
        print(f"  Acquisition applicable: {str(acquisition['applicable_at_acquisition']).lower()}"
              f" at {acquisition['observed_at']} ({acquisition['provenance']})")
        print(f"  Current applicable: {str(serving['current_applicability']).lower()}; "
              f"expired: {str(serving['expired']).lower()}; "
              f"not yet valid: {str(serving['not_yet_valid']).lower()}")
        print(f"  Evaluated at: {serving['evaluated_at']}; valid interval: "
              f"[{status['valid_from']}, {status['valid_until']})")
    covariances = summary.get("covariances", [])
    for covariance in covariances:
        _print_matrix(covariance.get("name", covariance.get("covariance_id", "retained covariance")),
                      covariance.get("matrix", covariance.get("covariance", [])),
                      covariance.get("quantity_ids", covariance.get("order", covariance.get("state_order", covariance.get("parameter_order")))),
                      covariance.get("units", covariance.get("unit")))
        for field in ("frame", "method", "covariance_id", "result_id"):
            if field in covariance:
                print(f"  {field}: {covariance[field]}")
    if not covariances:
        for result in summary.get("results", []):
            data = result.get("data", {})
            estimate = data.get("estimate", {})
            if isinstance(estimate, dict) and isinstance(estimate.get("covariance"), list):
                sources = data.get("calibrated_observation", {}).get("source_ids")
                _print_matrix("estimated state / " + result["result_id"], estimate["covariance"],
                              sources, str(estimate.get("unit", "")) + "²")
    for refusal in summary.get("refusals", []):
        detail = refusal.get("refusal", refusal)
        print(f"Refusal: {detail.get('code', 'refused')}: {detail.get('message', '')}")
    if summary.get("note"):
        print(summary["note"])
    print("Use --json to inspect full identities, covariance and provenance.")


def load_run(path: Path | None) -> dict:
    run = read_json(path) if path else make_demo_run()
    validate_run(run)
    return run


async def request_remote(url: str, kind: str, payload: dict, *, timeout_s: float = 15) -> dict:
    request_id = uuid.uuid4().hex
    async with asyncio.timeout(timeout_s):
        async with connect(url, max_size=33_554_432, open_timeout=min(5, timeout_s), close_timeout=1,
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
        duration, cursor = metadata["duration_s"], selection["cursor_s"]
        interval = selection["interval_s"]
        count = metadata["sample_count"]
        oscillator = run["instrument"] == "analytic-damped-oscillator.v1"
        if (not isinstance(interval, list) or len(interval) != 2
                or type(count) is not int or count < (2 if oscillator else 1)):
            raise ValueError
        numbers = (duration, cursor, *interval)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
            raise ValueError
        if not (duration > 0 and 0 <= cursor <= duration
                and 0 <= interval[0] < interval[1] <= duration):
            raise ValueError
        rate = metadata.get("sample_rate_hz")
        if oscillator or rate is not None:
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise ValueError
        if oscillator:
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
    analyze.add_argument("--channel", help="Defaults to the recording's first channel")
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
    server.add_argument("--fsrt-repo", type=Path,
                        help="Explicit trusted checkout for pinned FSRT operations")
    server.add_argument("--jspt-repo", type=Path,
                        help="Explicit trusted checkout for pinned JSPT covariance operations")
    server.add_argument("--gte-repo", type=Path,
                        help="Explicit trusted checkout for pinned GTE circle projection")
    stack = server.add_mutually_exclusive_group()
    stack.add_argument("--calibrated-stack-root", type=Path,
                       help="Bind eight role-named pinned checkouts to the shared process workbench")
    stack.add_argument("--identified-stack-root", type=Path,
                       help="Bind eleven pinned checkouts for shared process fusion and observation design")
    server.add_argument("--esm-binding", type=Path,
                        help="Trusted local ESM executable, replay, policy and optional candidate-store configuration")
    server.add_argument("--esm-telemetry-binding", type=Path,
                        help="Separate trusted ESM configuration at the scalar telemetry provider pins")
    server.add_argument("--telemetry-stack-root", type=Path,
                        help="Bind role-named ppda/stfe/gsie/set/cbsr checkouts alongside the process stack")
    server.add_argument("--calibrated-window-stack-root", type=Path,
                        help="Bind tbrt/mcur/stfe/gsie/set for shared calibrated window estimation")
    server.add_argument("--schematic-repo", type=Path, help="Bind pinned SRA declared schematic assessment")
    server.add_argument("--schematic-companions-root", type=Path, help="Bind pinned sra/jspt/plsr directories for selected schematic companion calls")
    server.add_argument("--construction-repo", type=Path, help="Bind pinned CSE quantity conditioning and ledger replay")
    server.add_argument("--acquisition-repo", type=Path, help="Bind pinned PPDA and its scout gitlink for retained dataset acquisition")
    server.add_argument("--exchange-set-repo", type=Path,
                        help="Bind the exact State Estimation Evaluation Testbed checkout for typed exchange adaptation")
    server.add_argument("--acquired-stream-stack-root", type=Path,
                        help="Bind pinned ppda/tbrt/mcur/stfe/gsie/set/oit/fdir for acquired calibrated windows and residual monitoring")
    server.add_argument("--measurement-chain-stack-root", type=Path,
                        help="Bind pinned rci/fsrt/jspt directories for native measurement-chain investigations")
    server.add_argument("--geometry-repo", type=Path,
                        help="Bind pinned GTE for retained circle inspection in the shared workbench")
    server.add_argument("--stability-repo", type=Path,
                        help="Bind pinned PLSR for explicitly selected identified model/state assessment")
    server.add_argument("--flat-torus-repo", type=Path,
                        help="Bind pinned flat-torus geodesic reference provider")
    server.add_argument("--curved-surface-repo", type=Path,
                        help="Bind pinned constant-curvature Jacobi transfer provider")
    server.add_argument("--spatial-view-origin", action="append", default=[], help="Exact browser http(s) origin allowed on the read-only /spatial endpoint; repeat to allow more")
    server.add_argument("--computation-repo", type=Path, help="Bind pinned SCR numerical execution")
    server.add_argument("--computation-engine", type=Path, help="Host-built SCR execution-cli (required with --computation-repo)")
    server.add_argument("--free-energy-stack-root", type=Path, help="Bind exact csg/gsie/plsr checkouts for the synthetic variational inference demonstration")
    server.add_argument("--covariance-geometry-repo", type=Path, help="Bind pinned SPD covariance geometry provider")
    server.add_argument("--intrinsic-surface-repo", type=Path, help="Bind pinned mesh edge-path baseline provider")
    server.add_argument("--translation-surface-repo", type=Path, help="Bind pinned square-tiled translation-flow provider")
    server.add_argument("--bind-role", action="append", default=[], metavar="KIND:ROLE=PATH",
                        help="Bind one provider role of any declared kind to an explicit host path; repeat per role. "
                             "The kind's declared roles and pins are enforced")
    server.add_argument("--sp1-prover", type=Path, help="Explicit SCR sp1-host binary; requires computation bindings and --sp1-heat-guest")
    server.add_argument("--sp1-heat-guest", type=Path, help="Exact registered SP1 heat guest ELF; requires --sp1-prover")
    server.add_argument("--python", dest="python_executable", type=Path,
                        help="Python for FSRT/JSPT/GTE adapters; workbench stacks use this running interpreter")
    health = commands.add_parser("health", help="Check a live session with a bounded read-only request")
    health.add_argument("--url", default="ws://127.0.0.1:8765")
    send = commands.add_parser("send", help="Send a structured request to a running session")
    send.add_argument("type", help="For example session.get, analysis.spectrum, selection.update")
    send.add_argument("--payload", default="{}", help="JSON object (or use --payload-file)")
    send.add_argument("--payload-file", type=Path)
    send.add_argument("--url", default="ws://127.0.0.1:8765")
    send.add_argument("--timeout", type=float, default=15,
                      help="Request deadline in seconds; allow longer for pinned provider workflows")
    watch = commands.add_parser("watch", help="Print shared selection and workbench change events as JSON lines")
    watch.add_argument("--url", default="ws://127.0.0.1:8765")
    inspect = commands.add_parser("inspect", help="Inspect a saved result/workspace without executing it")
    inspect.add_argument("path", type=Path)
    proof = commands.add_parser("proof", help="Freshly verify a retained registered SCR heat proof")
    proof_actions = proof.add_subparsers(dest="proof_command", required=True)
    proof_verify = proof_actions.add_parser("verify", help="Run the full registered-ELF verifier without producing a new proof")
    proof_verify.add_argument("path", type=Path, help="Retained ciw.proved-heat-session.v1 JSON bundle")
    proof_verify.add_argument("--computation-repo", type=Path, required=True)
    proof_verify.add_argument("--computation-engine", type=Path, required=True)
    proof_verify.add_argument("--sp1-prover", type=Path, required=True)
    proof_verify.add_argument("--sp1-heat-guest", type=Path, required=True)
    proof_verify.add_argument("--output", type=Path, required=True, help="New verification report file")
    exchange = commands.add_parser("exchange", help="Inspect external exchange artifacts without admission")
    exchange_actions = exchange.add_subparsers(dest="exchange_command", required=True)
    exchange_inspect = exchange_actions.add_parser("inspect", help="Read-only pinned contract conformance")
    exchange_inspect.add_argument("paths", type=Path, nargs="+")
    exchange_inspect.add_argument("--validator-repo", type=Path, required=True,
                                  help="Explicit checkout/export of the pinned testbed validator")
    telemetry = commands.add_parser("telemetry", help="Retained scalar telemetry operation script and numerical replay")
    telemetry_actions = telemetry.add_subparsers(dest="telemetry_command", required=True)
    telemetry_create = telemetry_actions.add_parser("create", help="Execute PPDA, STFE, GSIE and SET with a fresh replay")
    telemetry_create.add_argument("--source", type=Path, required=True)
    telemetry_create.add_argument("--configuration", type=Path, required=True)
    telemetry_inspect = telemetry_actions.add_parser("inspect", help="Inspect content bindings without running engines")
    telemetry_inspect.add_argument("path", type=Path)
    telemetry_replay = telemetry_actions.add_parser("replay", help="Recompute retained source and compare numerical outputs")
    telemetry_replay.add_argument("path", type=Path)
    for action in (telemetry_create, telemetry_replay):
        for role in ("ppda", "stfe", "gsie", "set"):
            action.add_argument("--" + role + "-repo", type=Path, required=True)
        action.add_argument("--cbsr-repo", type=Path)
        action.add_argument("--output-dir", type=Path, required=True)
    calibrated = commands.add_parser("calibrated-observable", help="Execute and replay the calibrated two-channel process experiment")
    calibrated_actions = calibrated.add_subparsers(dest="calibrated_command", required=True)
    calibrated_create = calibrated_actions.add_parser("create")
    calibrated_create.add_argument("--source", type=Path, required=True)
    calibrated_inspect = calibrated_actions.add_parser("inspect")
    calibrated_inspect.add_argument("path", type=Path)
    calibrated_replay = calibrated_actions.add_parser("replay")
    calibrated_replay.add_argument("path", type=Path)
    for action in (calibrated_create, calibrated_replay):
        for role in ("fsrt", "tbrt", "mcur", "oit", "gsie", "cbsr", "fdir", "set"):
            action.add_argument("--" + role + "-repo", type=Path, required=True)
        action.add_argument("--output-dir", type=Path, required=True)
    identified = commands.add_parser("identified-design", help="Identify dynamics and rank the next budgeted observation")
    identified_actions = identified.add_subparsers(dest="identified_command", required=True)
    identified_create = identified_actions.add_parser("create")
    identified_create.add_argument("--source", type=Path, required=True)
    identified_create.add_argument("--upstream", type=Path, required=True, help="Retained calibrated-observable session")
    identified_inspect = identified_actions.add_parser("inspect")
    identified_inspect.add_argument("path", type=Path)
    identified_replay = identified_actions.add_parser("replay")
    identified_replay.add_argument("path", type=Path)
    for action in (identified_create, identified_replay):
        for role in ("fsrt", "tbrt", "mcur", "oit", "gsie", "cbsr", "fdir", "set", "sidt", "edspt", "ywir"):
            action.add_argument("--" + role + "-repo", type=Path, required=True)
        action.add_argument("--output-dir", type=Path, required=True)
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
    investigation = commands.add_parser(
        "investigation", help="Calibrate RCI observations, evaluate FSRT and retain a shared workspace")
    investigation_actions = investigation.add_subparsers(dest="investigation_command", required=True)
    create = investigation_actions.add_parser("create", help="Create an RCI/FSRT investigation from explicit inputs")
    create.add_argument("--inputs", type=Path, required=True)
    inspect_investigation = investigation_actions.add_parser(
        "inspect", help="Inspect a saved investigation offline without executing its adapters")
    inspect_investigation.add_argument("path", type=Path)
    inspect_investigation.add_argument("--evaluated-at",
                                      help="Aware ISO timestamp for read-only calibration status")
    replay_investigation = investigation_actions.add_parser(
        "replay", help="Replay retained inputs with fresh execution and result identities")
    replay_investigation.add_argument("path", type=Path)
    for action in (create, replay_investigation):
        action.add_argument("--rci-repo", type=Path, required=True,
                            help="Explicit trusted RCI checkout at the pinned revision")
        action.add_argument("--fsrt-repo", type=Path, required=True,
                            help="Explicit trusted FSRT checkout at the pinned revision")
        action.add_argument("--output-dir", type=Path, required=True)
        action.add_argument("--python", dest="python_executable", type=Path,
                            help="Python for external adapters; defaults to this interpreter")
    for action in (create, inspect_investigation, replay_investigation):
        action.add_argument("--json", action="store_true", help="Print the complete machine-readable summary")
    covariance = commands.add_parser("covariance", help="Run pinned JSPT covariance operations on a saved investigation")
    covariance_replay = commands.add_parser("covariance-replay", help="Replay retained JSPT covariance inputs")
    covariance.add_argument("--parameters", type=Path, required=True,
                            help="JSON scientific operation parameters, never executable code")
    for action in (covariance, covariance_replay):
        action.add_argument("path", type=Path)
        action.add_argument("--jspt-repo", type=Path, required=True)
        action.add_argument("--output-dir", type=Path, required=True)
        action.add_argument("--adapter-python", "--python", dest="python_executable", type=Path)
        action.add_argument("--json", action="store_true")
    geodesic = commands.add_parser(
        "geodesic", help="Project declared circle telemetry and retain a shared investigation")
    geodesic_actions = geodesic.add_subparsers(dest="geodesic_command", required=True)
    geodesic_create = geodesic_actions.add_parser("create", help="Retain exact request bytes and execute GTE")
    geodesic_create.add_argument("--inputs", type=Path, required=True)
    geodesic_inspect = geodesic_actions.add_parser("inspect", help="Inspect retained GTE evidence without executing code")
    geodesic_inspect.add_argument("path", type=Path)
    geodesic_replay = geodesic_actions.add_parser("replay", help="Replay retained GTE inputs with fresh result identities")
    geodesic_replay.add_argument("path", type=Path)
    for action in (geodesic_create, geodesic_replay):
        action.add_argument("--gte-repo", type=Path, required=True)
        action.add_argument("--output-dir", type=Path, required=True)
        action.add_argument("--python", dest="python_executable", type=Path)
    for action in (geodesic_create, geodesic_inspect, geodesic_replay):
        action.add_argument("--json", action="store_true", help="Print covariance, identities and complete provenance")
    from .model.cli import add_parser as add_model_parser
    add_model_parser(commands)
    energy = commands.add_parser("energy", help="Capture GPU energy or replay retained energy/accuracy logs")
    energy_actions = energy.add_subparsers(dest="energy_command", required=True)
    energy_probe = energy_actions.add_parser("probe", help="Read an actual NVML counter without running a workload")
    energy_probe.add_argument("--gpu-index", type=int, default=0)
    energy_record = energy_actions.add_parser("record", help="Run a bounded Gaussian GPU experiment into a new directory")
    energy_record.add_argument("--problem", type=Path, required=True)
    energy_record.add_argument("--output-dir", type=Path, required=True)
    energy_record.add_argument("--duration", type=float, default=3)
    energy_record.add_argument("--replicas", type=int, default=4096)
    energy_record.add_argument("--iterations", type=int)
    energy_record.add_argument("--gpu-index", type=int, default=0)
    energy_record.add_argument("--max-batches", type=int, default=2048)
    energy_record.add_argument("--warmup-batches", type=int, default=2)
    energy_record.add_argument("--idle-duration", type=float, default=1)
    energy_replay = energy_actions.add_parser("replay", help="Recompute analysis from raw logs; never acquire new measurements")
    energy_replay.add_argument("path", type=Path)
    energy_replay.add_argument("--output", type=Path)
    return root


def parse_bindings(values):
    """Group ``KIND:ROLE=PATH`` operator bindings by kind; each role is bound once."""
    grouped = {}
    for value in values:
        kind, separator, rest = value.partition(":")
        role, equals, path = rest.partition("=")
        if not separator or not equals or not kind or not role or not path.strip():
            raise ValueError(f"--bind-role expects KIND:ROLE=PATH, not {value!r}")
        roles = grouped.setdefault(kind, {})
        if role in roles:
            raise ValueError(f"--bind-role names {kind}:{role} more than once")
        roles[role] = Path(path)
    return grouped


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "model":
            from .model.cli import run as run_model
            return run_model(args, print_json)
        if args.command == "energy":
            if args.energy_command == "probe":
                from .energy_bench import probe
                print_json(probe(args.gpu_index))
            elif args.energy_command == "record":
                from .energy_bench import capture
                log, report = capture(read_json(args.problem), args.output_dir,
                    minimum_duration_s=args.duration, replicas=args.replicas, iterations=args.iterations,
                    device_index=args.gpu_index, max_batches=args.max_batches,
                    warmup_batches=args.warmup_batches, idle_duration_s=args.idle_duration)
                print_json({"log_file": str(args.output_dir / "log.json"), "log_digest": log["log_digest"],
                            "measurement": report["measurement"], "comparison": report["comparison"]})
            else:
                from .energy_records import analyze, MAX_BYTES
                from .adapters.subprocess import _json
                with args.path.open("rb") as stream:
                    raw = stream.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("Energy source exceeds its exact-byte size bound")
                report = analyze(_json(raw))
                if args.output:
                    _write_new_proof_report(args.output, report)
                print_json(report)
        elif args.command == "demo":
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
                                       "payload": {"channel": (session.selection["channel"] if args.channel is None
                                                               else args.channel),
                                                   "interval_s": interval}})
            print_json(response)
            if response["type"] == "error":
                return 2
            session.save_workspace(args.output_dir / "workspace.json")
        elif args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            session = load_server_session(args)
            if args.fsrt_repo is not None:
                from .investigation import bind_fsrt
                bind_fsrt(session, args.fsrt_repo, python_executable=args.python_executable)
            if args.jspt_repo is not None:
                from .covariance_workflow import bind_jspt
                bind_jspt(session, args.jspt_repo, python_executable=args.python_executable)
            if args.gte_repo is not None:
                from .geodesic import bind_gte
                bind_gte(session, args.gte_repo, python_executable=args.python_executable)
            if args.calibrated_stack_root is not None or args.identified_stack_root is not None:
                from .calibrated_observable import ROLES
                stack_root = args.calibrated_stack_root or args.identified_stack_root
                session.workbench.bind_workflow("calibrated-observable", {
                    role: stack_root / role for role in ROLES})
                if args.identified_stack_root is not None:
                    from .identified_design import ROLES as DESIGN_ROLES
                    session.workbench.bind_workflow("identified-design", {
                        role: stack_root / role for role in DESIGN_ROLES})
            if args.esm_binding is not None:
                session.workbench.bind_candidate_adapter(read_json(args.esm_binding))
            if args.telemetry_stack_root is not None:
                from .telemetry import ROLES as TELEMETRY_ROLES
                session.workbench.bind_workflow("telemetry", {role: args.telemetry_stack_root / role for role in TELEMETRY_ROLES})
            if args.calibrated_window_stack_root is not None:
                from .calibrated_window import ROLES as WINDOW_ROLES
                session.workbench.bind_workflow("calibrated-window", {role: args.calibrated_window_stack_root / role for role in WINDOW_ROLES})
            if args.schematic_repo is not None:
                session.workbench.bind_workflow("schematic-assessment", {"sra": args.schematic_repo})
            if args.schematic_companions_root is not None:
                root = args.schematic_companions_root
                session.workbench.bind_workflow("schematic-assessment", {"sra": root / "sra"})
                session.workbench.bind_workflow("schematic-companions", {role: root / role for role in ("sra", "jspt", "plsr")})
            if args.construction_repo is not None:
                session.workbench.bind_workflow("bim-quantity", {"cse": args.construction_repo})
            if args.acquisition_repo is not None:
                session.workbench.bind_workflow("acquired-dataset", {"ppda": args.acquisition_repo})
            if args.exchange_set_repo is not None:
                session.workbench.bind_workflow("instrument-exchange", {"set": args.exchange_set_repo})
            if args.acquired_stream_stack_root is not None:
                root = args.acquired_stream_stack_root
                from .calibrated_window import ROLES as WINDOW_ROLES
                session.workbench.bind_workflow("acquired-dataset", {"ppda": root / "ppda"})
                for kind in ("calibrated-window", "acquired-calibrated-window"):
                    session.workbench.bind_workflow(kind, {role: root / role for role in WINDOW_ROLES})
                session.workbench.bind_workflow("residual-monitor", {role: root / role for role in ("oit", "fdir")})
            if args.measurement_chain_stack_root is not None:
                session.workbench.bind_workflow("measurement-chain", {
                    role: args.measurement_chain_stack_root / role for role in ("rci", "fsrt", "jspt")})
            if args.geometry_repo is not None:
                session.workbench.bind_workflow("geometric-circle", {"gte": args.geometry_repo})
            if args.stability_repo is not None:
                session.workbench.bind_workflow("identified-stability", {"plsr": args.stability_repo})
            if args.flat_torus_repo is not None:
                session.workbench.bind_workflow("flat-torus-reference", {"ftr": args.flat_torus_repo})
            if args.curved_surface_repo is not None:
                session.workbench.bind_workflow("curved-path-transfer", {"csg": args.curved_surface_repo})
            if args.free_energy_stack_root is not None:
                session.workbench.bind_workflow("variational-free-energy", {role: args.free_energy_stack_root / role for role in ("csg", "gsie", "plsr")})
            if args.covariance_geometry_repo is not None:
                session.workbench.bind_workflow("covariance-geometry", {"cggt": args.covariance_geometry_repo})
            if args.intrinsic_surface_repo is not None:
                session.workbench.bind_workflow("mesh-path", {"isgt": args.intrinsic_surface_repo})
            if args.translation_surface_repo is not None:
                session.workbench.bind_workflow("translation-flow", {"tsde": args.translation_surface_repo})
            if (args.computation_repo is None) != (args.computation_engine is None):
                raise ValueError("--computation-repo and --computation-engine must be supplied together")
            if args.computation_repo is not None:
                session.workbench.bind_workflow("numerical-heat", {"scr": args.computation_repo, "engine": args.computation_engine})
            if (args.sp1_prover is None) != (args.sp1_heat_guest is None):
                raise ValueError("--sp1-prover and --sp1-heat-guest must be supplied together")
            if args.sp1_prover is not None:
                if args.computation_repo is None:
                    raise ValueError("SP1 requires --computation-repo and --computation-engine")
                session.workbench.bind_workflow("proved-heat", {"scr": args.computation_repo,
                    "engine": args.computation_engine, "prover": args.sp1_prover, "guest": args.sp1_heat_guest})
            for kind, repositories in parse_bindings(args.bind_role).items():
                if session.workbench._bindings.get(kind):
                    raise ValueError(f"--bind-role {kind} repeats a binding another option already made")
                session.workbench.bind_workflow(kind, repositories)
            if args.esm_telemetry_binding is not None:
                configuration = read_json(args.esm_telemetry_binding)
                if "ppda" not in configuration.get("runtime", {}).get("repositories", {}):
                    raise ValueError("--esm-telemetry-binding must declare the telemetry provider set")
                session.workbench.bind_candidate_adapter(configuration)
            asyncio.run(run_server(session, args.port, args.bind, spatial_view_origins=args.spatial_view_origin))
        elif args.command == "health":
            print_json(asyncio.run(health_remote(args.url)))
        elif args.command == "send":
            payload = (read_json(args.payload_file) if args.payload_file else
                       json.loads(args.payload, parse_constant=_reject_constant))
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
                raise ValueError("timeout must be finite, positive and at most 3600 seconds")
            response = asyncio.run(request_remote(args.url, args.type, payload, timeout_s=args.timeout))
            print_json(response)
            return 0 if response["type"] == "response" else 2
        elif args.command == "watch":
            asyncio.run(watch_remote(args.url))
        elif args.command == "inspect":
            print_json(read_json(args.path))
        elif args.command == "proof":
            from .adapters.subprocess import _json
            from .exchange import _read
            from .proved_heat import ProvedHeatWorkflow, MAX_BYTES
            if args.output.exists():
                raise ValueError("Use a new verification report path to preserve previous occurrences")
            report = ProvedHeatWorkflow().verify_session(_json(_read(args.path, MAX_BYTES)), {
                "scr": args.computation_repo, "engine": args.computation_engine,
                "prover": args.sp1_prover, "guest": args.sp1_heat_guest})
            _write_new_proof_report(args.output, report)
            print_json({"verification_file": str(args.output), "verification": report})
        elif args.command == "exchange":
            from .exchange import inspect_exchange
            print_json(inspect_exchange(args.paths, validator_repo=args.validator_repo))
        elif args.command == "telemetry":
            from .telemetry import create_session, inspect_session, replay_session, read_session, save_session
            if args.telemetry_command == "inspect":
                print_json(inspect_session(read_session(args.path)))
            else:
                repositories = {role: getattr(args, role + "_repo") for role in ("ppda", "stfe", "gsie", "set")}
                if args.cbsr_repo is not None:
                    repositories["cbsr"] = args.cbsr_repo
                if args.telemetry_command == "create":
                    from .exchange import _read
                    from .telemetry import MAX_BYTES
                    bundle = create_session(_read(args.source, MAX_BYTES), read_json(args.configuration), repositories)
                    path = save_session(bundle, args.output_dir)
                    print_json({"session_file": str(path), "inspection": inspect_session(bundle)})
                else:
                    result = replay_session(read_session(args.path), repositories)
                    path = save_session(result["session"], args.output_dir)
                    print_json({"session_file": str(path), "replay_receipt": result["replay_receipt"]})
        elif args.command == "calibrated-observable":
            from .calibrated_observable import (ROLES, MAX_BYTES, create_session, inspect_session,
                                               replay_session, read_session, save_session)
            if args.calibrated_command == "inspect":
                print_json(inspect_session(read_session(args.path)))
            else:
                repositories = {role: getattr(args, role + "_repo") for role in ROLES}
                if args.calibrated_command == "create":
                    from .exchange import _read
                    bundle = create_session(_read(args.source, MAX_BYTES), repositories)
                    path = save_session(bundle, args.output_dir)
                    print_json({"session_file": str(path), "inspection": inspect_session(bundle)})
                else:
                    result = replay_session(read_session(args.path), repositories)
                    path = save_session(result["session"], args.output_dir)
                    print_json({"session_file": str(path), "replay_receipt": result["replay_receipt"]})
        elif args.command == "identified-design":
            from .identified_design import (ROLES, MAX_BYTES, create_session, inspect_session,
                                            replay_session, read_session, save_session)
            if args.identified_command == "inspect":
                print_json(inspect_session(read_session(args.path)))
            else:
                repositories = {role: getattr(args, role + "_repo") for role in ROLES}
                if args.identified_command == "create":
                    from .exchange import _read
                    bundle = create_session(_read(args.source, MAX_BYTES), read_session(args.upstream), repositories)
                    path = save_session(bundle, args.output_dir)
                    print_json({"session_file": str(path), "inspection": inspect_session(bundle)})
                else:
                    result = replay_session(read_session(args.path), repositories)
                    path = save_session(result["session"], args.output_dir)
                    print_json({"session_file": str(path), "replay_receipt": result["replay_receipt"]})
        elif args.command == "investigation":
            from .investigation import create_investigation, inspect_investigation, replay_investigation
            if args.investigation_command == "inspect":
                result = inspect_investigation(args.path, evaluated_at=args.evaluated_at)
            else:
                arguments = {"rci_repo": args.rci_repo, "fsrt_repo": args.fsrt_repo,
                             "output_dir": args.output_dir, "python_executable": args.python_executable}
                if args.investigation_command == "create":
                    inputs = read_json(args.inputs)
                    if not isinstance(inputs, dict):
                        raise ValueError("investigation inputs must be a JSON object")
                    result = create_investigation(inputs, **arguments)
                else:
                    result = replay_investigation(args.path, **arguments)
            if args.json:
                print_json(result)
            else:
                print_investigation(result)
        elif args.command in {"covariance", "covariance-replay"}:
            from .covariance_workflow import execute_covariance, replay_covariance
            arguments = {"jspt_repo": args.jspt_repo, "output_dir": args.output_dir,
                         "python_executable": args.python_executable}
            if args.command == "covariance":
                parameters = read_json(args.parameters)
                if not isinstance(parameters, dict):
                    raise ValueError("covariance parameters must be a JSON object")
                result = execute_covariance(args.path, parameters=parameters, **arguments)
            else:
                result = replay_covariance(args.path, **arguments)
            if args.json:
                print_json(result)
            else:
                print_investigation(result)
        elif args.command == "geodesic":
            from .geodesic import create_investigation, inspect_investigation, replay_investigation
            if args.geodesic_command == "inspect":
                result = inspect_investigation(args.path)
            else:
                arguments = {"gte_repo": args.gte_repo, "output_dir": args.output_dir,
                             "python_executable": args.python_executable}
                if args.geodesic_command == "create":
                    result = create_investigation(args.inputs, **arguments)
                else:
                    result = replay_investigation(args.path, **arguments)
            if args.json:
                print_json(result)
            else:
                print_investigation(result)
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
        return 130 if args.command == "energy" and args.energy_command == "record" else 0
    except AdapterRefusal as exc:
        print(f"ciw: {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, TimeoutError, WebSocketException) as exc:
        print(f"ciw: {exc}", file=sys.stderr)
        return 2
