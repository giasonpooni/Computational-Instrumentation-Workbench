"""Regenerate the retained lab evidence in ``lab/`` from a clean-room run.

Runs ``scripts/check_lab.py --no-compare`` (isolated wheel, pinned providers,
lab tests, whole queue), then replaces the retained reports, artifacts, queue
state and report book with the fresh ones and renders ``lab/index.html``.
Elapsed times, JUnit records and gate records stay with the run output: they
are machine-specific and not retained evidence. Review ``git diff lab`` and
``ciw lab verify`` output before committing a refresh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ("reports", "artifacts", "queue-state.json", "REPORTS.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=os.environ.get("CIW_LAB_STACK_ROOT"))
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--from-run", type=Path, help="Use an existing clean-room output instead of running one")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ciw-lab-refresh-", dir=args.temporary_root) as directory:
        run = args.from_run
        if run is None:
            run = Path(directory) / "run"
            command = [sys.executable, str(ROOT / "scripts" / "check_lab.py"), "--no-compare", "--output-dir", str(run)]
            if args.stack_root:
                command += ["--stack-root", str(args.stack_root)]
            if args.temporary_root:
                command += ["--temporary-root", str(args.temporary_root)]
            subprocess.run(command, check=True)
        gate = run / "gate.json"
        if not gate.is_file() or json.loads(gate.read_text(encoding="utf-8")).get("schema") != "ciw.lab-clean-room-gate.v1":
            raise SystemExit(f"Not a clean-room output (no ciw.lab-clean-room-gate.v1 gate.json): {run}")
        missing = [name for name in RETAINED if not (run / name).exists()]
        if missing:
            raise SystemExit(f"Clean-room output is incomplete: {missing}")
        target = ROOT / "lab"
        target.mkdir(exist_ok=True)
        for name in RETAINED:
            destination = target / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            source = run / name
            (shutil.copytree if source.is_dir() else shutil.copy2)(source, destination)
    subprocess.run([sys.executable, "-m", "ciw", "lab", "dashboard", "--retained", str(target),
                    "--output", str(target / "index.html")], check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    print(f"Refreshed {target}; review git diff before committing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
