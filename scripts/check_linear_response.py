"""Create or inspect a bounded linear-response preview without a provider."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ciw.linear_response import (load_preview, load_request, preview,
                                 render_preview, save_preview)  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create", help="Explicitly evaluate an allowlisted analytic profile")
    create.add_argument("source", type=Path, help="Data-only response request JSON")
    create.add_argument("--output", type=Path, required=True, help="New preview path; existing files are refused")
    create.add_argument("--details", action="store_true", help="Expand the same calculation with equations and bindings")
    inspect = actions.add_parser("inspect", help="Check and display saved content without model or provider calls")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            record = preview(load_request(args.source))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_preview(args.output, record)
        else:
            record = load_preview(args.path)
        print(render_preview(record, details=args.details))
        print("Preview content identity:", record["preview_id"])
        return 0
    except (OSError, ValueError) as exc:
        print("Linear response refused:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
