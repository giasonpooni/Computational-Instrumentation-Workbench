"""``ciw science``: terminal access to specifications, ledgers, replay, reports and the bench.

Every command prints JSON (or the rendered report) and exits 0 on success,
2 on a declared refusal and 1 on an unexpected error. Commands that only read
never write; commands that write take an explicit ledger or output path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ._common import Refusal, plain


def _load(path: Path) -> tuple[Any, bytes]:
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as exc:
        raise Refusal("malformed_record", f"{path} is not JSON: {exc}") from exc


def _print(value: Any) -> None:
    sys.stdout.write(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _keys(values: list[str] | None) -> dict[str, bytes]:
    keys = {}
    for item in values or []:
        key_id, _, hexadecimal = item.partition("=")
        keys[key_id] = bytes.fromhex(hexadecimal)
    return keys


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ciw science", description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)

    spec = commands.add_parser("spec", help="Validate, compile or render an experiment specification")
    spec.add_argument("action", choices=("validate", "compile", "latex"))
    spec.add_argument("path", type=Path)

    run = commands.add_parser("run", help="Compile and execute specifications into a ledger")
    run.add_argument("paths", type=Path, nargs="+")
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--frames", type=Path, help="Frame registry to retain and bind to the runs")

    replay = commands.add_parser("replay", help="Re-execute a retained experiment and record replay receipts")
    replay.add_argument("experiment_id")
    replay.add_argument("--ledger", type=Path, required=True)

    status = commands.add_parser("status", help="Derive the five status statements for an experiment")
    status.add_argument("experiment_id")
    status.add_argument("--ledger", type=Path, required=True)

    ledger = commands.add_parser("ledger", help="Verify, list or adversarially probe a ledger")
    ledger.add_argument("action", choices=("verify", "show", "probe", "audit"))
    ledger.add_argument("path", type=Path)
    ledger.add_argument("--kind")
    ledger.add_argument("--limit", type=int, default=20)

    report = commands.add_parser("report", help="Render a reproducible report from a ledger")
    report.add_argument("--ledger", type=Path, required=True)
    report.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    report.add_argument("--output", type=Path)
    report.add_argument("--experiment", action="append")
    report.add_argument("--retain", action="store_true", help="Retain the rendered report in the ledger")

    design = commands.add_parser("design", help="Rank candidate experiments by expected information gain")
    design.add_argument("path", type=Path)
    design.add_argument("--ledger", type=Path, help="Retain the recommendation")

    frames = commands.add_parser("frames", help="Resolve a transform chain or map a point between frames")
    frames.add_argument("registry", type=Path)
    frames.add_argument("source")
    frames.add_argument("target")
    frames.add_argument("--at", type=float, required=True)
    frames.add_argument("--clock", required=True)
    frames.add_argument("--point", type=float, nargs=3)
    frames.add_argument("--unit", default="m")

    units = commands.add_parser("units", help="Convert a value between commensurable units")
    units.add_argument("value", type=float)
    units.add_argument("source")
    units.add_argument("target")

    commands.add_parser("solvers", help="List registered solvers with equations, capabilities and failure modes")
    backends = commands.add_parser("backends", help="Describe execution backends")
    backends.add_argument("--probe", action="store_true", help="Run toolchain --version and open NVML")

    usecase = commands.add_parser("usecase", help="Compile an industrial use case into requirements")
    usecase.add_argument("path", type=Path)

    physical = commands.add_parser("physical", help="Evaluate a physical protocol against measurements")
    physical.add_argument("protocol", type=Path)
    physical.add_argument("measurements", type=Path)
    physical.add_argument("predictions", type=Path)

    fusion = commands.add_parser("fusion", help="Run a fusion scenario to candidate and admission records")
    fusion.add_argument("scenario", type=Path)
    fusion.add_argument("--policy", type=Path, required=True)

    hardware = commands.add_parser("hardware", help="Parse a CIWT telemetry capture (observation only)")
    hardware.add_argument("capture", type=Path)
    hardware.add_argument("--layouts", type=Path, required=True)
    hardware.add_argument("--tick-rate-hz", type=float, required=True)
    hardware.add_argument("--device-clock", required=True)
    hardware.add_argument("--records", action="store_true", help="Print decoded records, not only the summary")

    bundle = commands.add_parser("bundle", help="Export, inspect or import signed offline evidence bundles")
    bundle.add_argument("action", choices=("export", "inspect", "import"))
    bundle.add_argument("path", type=Path, help="Bundle file")
    bundle.add_argument("--ledger", type=Path, help="Source ledger (export) or destination (import)")
    bundle.add_argument("--signing-key-hex", help="32-byte Ed25519 secret in hex (export)")
    bundle.add_argument("--key-id")
    bundle.add_argument("--trusted", action="append", help="key_id=public_key_hex (inspect/import)")
    bundle.add_argument("--allow-unsigned", action="store_true")

    bench = commands.add_parser("bench", help="Run the synthetic bench into a new output directory")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--examples", type=Path)
    bench.add_argument("--quick", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return _dispatch(args)
    except Refusal as exc:
        _print({"refused": exc.to_dict()})
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    from .ledger import Ledger
    if args.command == "spec":
        from .experiment import compile_spec, latex, spec_identity, validate_spec
        spec, _ = _load(args.path)
        if args.action == "validate":
            validate_spec(spec)
            _print({"valid": True, "spec_identity": spec_identity(spec)})
        elif args.action == "compile":
            plan = compile_spec(spec)
            _print({key: value for key, value in plan.items() if key != "jobs"} | {"job_ids": [job["job_id"] for job in plan["jobs"]]})
        else:
            sys.stdout.write(latex(spec))
    elif args.command == "run":
        from .frames import FrameRegistry
        from .runner import execute
        ledger = Ledger.open_or_create(args.ledger)
        frames_entry = None
        if args.frames:
            record, raw = _load(args.frames)
            registry = FrameRegistry.from_json(record)
            blob = ledger.put_blob(raw)
            frames_entry = ledger.append("ciw.science.frame-registry.v1", {
                "registry": record, "registry_identity": registry.identity(), "source_blob": blob}, blobs=[blob])["entry_id"]
        results = []
        for path in args.paths:
            spec, raw = _load(path)
            results.append(execute(spec, ledger, source_bytes=raw, frame_registry_entry=frames_entry))
        _print(results)
    elif args.command == "replay":
        from .runner import replay
        _print(replay(Ledger.open(args.ledger), args.experiment_id))
    elif args.command == "status":
        from .claims import status_statements
        _print(status_statements(Ledger.open(args.ledger), args.experiment_id))
    elif args.command == "ledger":
        if args.action == "verify":
            report = Ledger.open(args.path, strict=False).verify()
            _print(report.to_json())
            return 0 if report.ok else 2
        if args.action == "probe":
            from .agents import adversarial_ledger_probe
            results = adversarial_ledger_probe(args.path)
            _print(results)
            return 0 if all(item["detected"] for item in results) else 2
        ledger = Ledger.open(args.path)
        if args.action == "audit":
            from .agents import provenance_audit
            findings = provenance_audit(ledger)
            _print(findings)
            return 0 if not findings else 2
        entries = list(ledger.entries(args.kind))[-max(1, args.limit):]
        _print([{"sequence": e["sequence"], "kind": e["kind"], "entry_id": e["entry_id"], "recorded_at": e["recorded_at"],
                 "refs": len(e["refs"]), "blobs": len(e["blobs"]),
                 "summary": {k: v for k, v in e["body"].items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 200}}
                for e in entries])
    elif args.command == "report":
        from .report import build, latex, markdown, retain
        ledger = Ledger.open(args.ledger)
        model = build(ledger, args.experiment)
        text = markdown(model) if args.format == "markdown" else latex(model)
        if args.output:
            args.output.write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        if args.retain:
            entry = retain(ledger, text, args.format)
            sys.stderr.write(f"retained report {entry['entry_id']}\n")
    elif args.command == "design":
        from . import design
        problem, _ = _load(args.path)
        ranking = design.rank(problem)
        if args.ledger:
            design.retain(Ledger.open_or_create(args.ledger), problem, ranking)
        _print(ranking)
    elif args.command == "frames":
        from .frames import FrameRegistry
        record, _ = _load(args.registry)
        registry = FrameRegistry.from_json(record)
        if args.point:
            _print(registry.transform_point(args.point, args.unit, args.source, args.target, args.at, args.clock))
        else:
            _print(registry.resolve(args.source, args.target, args.at, args.clock).to_json())
    elif args.command == "units":
        from .units import convert, parse_unit
        _print({"value": convert(args.value, args.source, args.target), "unit": args.target,
                "source": parse_unit(args.source).describe(), "target": parse_unit(args.target).describe()})
    elif args.command == "solvers":
        from .solvers import default_registry
        _print(default_registry().describe())
    elif args.command == "backends":
        from .backends import detect
        _print([backend.describe() for backend in detect(args.probe)])
    elif args.command == "usecase":
        from .solvers import default_registry
        from .usecase import compile_use_case
        record, _ = _load(args.path)
        available = {name for item in default_registry().describe() for name, ok in item["capabilities"].items() if ok}
        _print(compile_use_case(record, available))
    elif args.command == "physical":
        from .physical import evaluate
        _print(evaluate(_load(args.protocol)[0], _load(args.measurements)[0], _load(args.predictions)[0]))
    elif args.command == "fusion":
        from .solvers import run_fusion
        result = run_fusion({"scenario": _load(args.scenario)[0], "policy": _load(args.policy)[0]})
        _print({"stages": result["stages"], "admission": {key: result["admission"].get(key) for key in
                                                         ("stage", "status", "reasons", "identity")},
                "position_std": result["candidate"]["position_std"], "unit": result["candidate"]["unit"]})
    elif args.command == "hardware":
        from .hardware import parse_stream
        capture = parse_stream(args.capture.read_bytes(), _load(args.layouts)[0], tick_rate_hz=args.tick_rate_hz,
                               device_clock=args.device_clock)
        _print(capture.to_json() if args.records else capture.summary)
    elif args.command == "bundle":
        from .bundle import export_bundle, import_bundle, inspect_bundle
        if args.action == "export":
            if args.ledger is None:
                raise Refusal("malformed_request", "--ledger is required for export")
            key = bytes.fromhex(args.signing_key_hex) if args.signing_key_hex else None
            _print(export_bundle(Ledger.open(args.ledger), args.path, signing_key=key, key_id=args.key_id))
        elif args.action == "inspect":
            report = inspect_bundle(args.path, trusted_keys=_keys(args.trusted))
            _print(report)
            return 0 if report.get("ok") else 2
        else:
            if args.ledger is None:
                raise Refusal("malformed_request", "--ledger is required for import")
            ledger = import_bundle(args.path, args.ledger, trusted_keys=_keys(args.trusted),
                                   require_signature=not args.allow_unsigned)
            _print({"imported": str(args.ledger), "entries": len(ledger), "head": ledger.head()})
    elif args.command == "bench":
        from .bench import run_bench
        summary = run_bench(args.output, examples=args.examples, quick=args.quick)
        _print(summary)
        return 0 if summary["ledger"]["verified"] else 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
