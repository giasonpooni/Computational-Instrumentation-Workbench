"""Run the lab queue from an isolated wheel with pinned providers and verify retained reports.

Provisions the exact CSG and FTR revisions pinned by ``ciw.geodesic_reference``,
installs the pinned PLSR runtime through the ``plsr`` extra on Python 3.12+,
and delegates to ``scripts/reproduce_lab.py``: wheel build, clean virtual
environment, lab tests with a JUnit record, the whole queue, and a
tolerance-aware comparison with ``lab/``. Hardware-dependent tasks remain
blocked; no physical measurement is acquired.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from provider_checkouts import validate_checkout

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {"csg": "Curved-Surface-Geodesic-Sensitivity-Runtime", "ftr": "Flat-Torus-Geodesic-Reference"}


def call(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], check=True, timeout=3600, **kwargs)


def pins() -> dict:
    tree = ast.parse((ROOT / "src" / "ciw" / "geodesic_reference.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PINS" for t in node.targets):
            return {entry["role"]: entry["revision"] for entry in ast.literal_eval(node.value).values()}
    raise SystemExit("PINS declaration not found in geodesic_reference.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lab-gate")
    parser.add_argument("--stack-root", type=Path, default=os.environ.get("CIW_LAB_STACK_ROOT"),
                        help="Existing clean checkouts named csg and ftr; cloned when absent")
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    revisions = pins()
    with tempfile.TemporaryDirectory(prefix="ciw-lab-providers-", dir=args.temporary_root) as directory:
        providers = {}
        for role, repository in REPOSITORIES.items():
            path = (Path(args.stack_root) / role) if args.stack_root else Path(directory) / role
            if not args.stack_root:
                call(["git", "-c", "core.autocrlf=false", "clone", "--quiet",
                      f"https://github.com/giasonpooni/{repository}.git", path])
                call(["git", "-C", path, "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", revisions[role]])
            providers[role] = validate_checkout(path, revisions[role])
        command = [sys.executable, ROOT / "scripts" / "reproduce_lab.py", "--output-dir", args.output_dir]
        if args.temporary_root:
            command += ["--temporary-root", args.temporary_root]
        if args.no_compare:
            command.append("--no-compare")
        for role, path in providers.items():
            command += ["--provider", f"{role}={path}"]
        if sys.version_info >= (3, 12):
            command += ["--extras", "dev,lab,plsr", "--provider", "plsr-python=@venv"]
        call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
