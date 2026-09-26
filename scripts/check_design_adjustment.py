"""Solve a bounded JuMP study or check/inspect an exact design certificate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ciw import design_adjustment as design, jump_design  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "inspect", "solve", "inspect-study"):
        command = sub.add_parser(name)
        command.add_argument("path", type=Path)
        command.add_argument("--details", action="store_true")
        if name in {"check", "solve"}:
            command.add_argument("--output", type=Path, required=True)
        if name == "solve":
            command.add_argument("--julia", type=Path, required=True)
            command.add_argument("--depot", type=Path)
            command.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        if args.action in {"check", "solve"} and args.output.exists():
            raise ValueError("Output already exists")
        if args.action == "solve":
            study = jump_design.run_study(design.load_problem(args.path), args.julia,
                                          depot=args.depot, timeout=args.timeout)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            jump_design.save_study(args.output, study)
            report = study["check"]
            print("Julia invoked; exact check applies to the explicitly projected rational candidate.")
            print("Study content identity:", study["study_id"])
        elif args.action == "inspect-study":
            study = jump_design.load_study(args.path)
            report = study["check"]
            print("Retained Julia study; no solver or certificate check rerun.")
            print("Study content identity:", study["study_id"])
        elif args.action == "check":
            report = design.check_candidate(design.load_candidate(args.path))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            design.save_check(args.output, report)
            print("Explicit exact local certificate check completed; no SP1 proof produced.")
        else:
            report = design.load_check(args.path)
        print(design.render_check(report, details=args.details))
        return 1 if args.action in {"check", "solve"} and report["status"] == "rejected" else 0
    except (OSError, ValueError) as exc:
        print("Design adjustment refused:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
