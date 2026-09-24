"""Run the lab queue from an isolated wheel with pinned providers and verify retained reports.

Provisions the exact CSG, FTR and SCR revisions pinned by ``ciw.geodesic_reference``
and ``ciw.declared_workload``, and the SET, PPDA and SCR revisions of
``.github/workflows/exchange.yml`` for the exchange roundtrip (T097),
installs the pinned PLSR runtime through the ``plsr`` extra on Python 3.12+,
where the clean-room interpreter also runs FTR,
and delegates to ``scripts/reproduce_lab.py``: wheel build, clean virtual
environment, lab tests with a JUnit record (provider-gated tests against the
bound providers), the whole queue, and a tolerance-aware comparison with
``lab/``. The comparison needs Python 3.12+, like the retained run; older
interpreters may only run with ``--no-compare``. Hardware-dependent tasks
remain blocked; no physical measurement is acquired. Retained operator
hardware runs (``lab/hardware/<run-id>/``, made on their capture host) are
verified for integrity only, in either mode: reports validate and hold no host
path, artifacts match their digests, every hardware-measured finding cites raw
bytes its task retained after a hardware probe succeeded in that task itself,
and ``capture.json`` agrees with the run. A run whose physical findings rest
on a probe of the capture host's hardware cannot be recomputed here or in CI.
``--blas-core CORE`` runs the clean room on that OpenBLAS kernel (for example
Haswell or Sandybridge), to verify the retained run on the kernels other hosts
would pick; the gate record names the kernel the clean room ran.
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
REPOSITORIES = {"csg": "Curved-Surface-Geodesic-Sensitivity-Runtime", "ftr": "Flat-Torus-Geodesic-Reference",
                "scr": "Scientific-Computation-Runtime", "set": "State-Estimation-Evaluation-Testbed",
                "ppda": "Provenance-Preserving-Data-Acquisition", "scr-exchange": "Scientific-Computation-Runtime"}


def call(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], check=True, timeout=3600, **kwargs)


def pins() -> dict:
    """Provider revisions exactly as pinned by the CIW workflows that own them."""
    revisions = {}
    for module in ("geodesic_reference.py", "declared_workload.py"):
        tree = ast.parse((ROOT / "src" / "ciw" / module).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PINS" for t in node.targets):
                revisions.update({entry["role"]: entry["revision"] for entry in ast.literal_eval(node.value).values()})
    # The exchange roundtrip roles (T097) use the pins of .github/workflows/exchange.yml, which the lab section
    # mirrors in EXCHANGE_WORKFLOW_PINS and its tests keep in step.
    tree = ast.parse((ROOT / "src" / "ciw" / "lab" / "exchange_provenance_bundles_providers.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EXCHANGE_WORKFLOW_PINS" for t in node.targets):
            revisions.update(ast.literal_eval(node.value))
    missing = set(REPOSITORIES) - set(revisions)
    if missing:
        raise SystemExit(f"Provider pins not found: {sorted(missing)}")
    return revisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lab-gate")
    parser.add_argument("--stack-root", type=Path, default=os.environ.get("CIW_LAB_STACK_ROOT"),
                        help="Existing clean checkouts named csg, ftr, scr, set, ppda and scr-exchange; cloned when absent")
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--blas-core", metavar="CORE",
                        help="OpenBLAS kernel for the clean room (e.g. Haswell or Sandybridge); see reproduce_lab.py")
    args = parser.parse_args()
    if args.temporary_root:
        args.temporary_root = args.temporary_root.resolve()
    if not args.no_compare and sys.version_info < (3, 12):
        # Without the PLSR and FTR interpreters those tasks end partial, unlike the retained run.
        raise SystemExit("Comparing with the retained run needs Python 3.12+, which hosts the PLSR and FTR "
                         f"providers (this is {sys.version.split()[0]}); run under Python 3.12 or pass --no-compare")
    if not args.no_compare:
        # Refuse before provisioning providers: reproduce_lab.py would refuse after.
        reports = ROOT / "lab" / "reports"
        if not reports.is_dir() or not any(reports.glob("T*.json")):
            raise SystemExit(f"No retained reports to compare with in {reports}; retain a reviewed run first "
                             "(scripts/refresh_lab.py) or pass --no-compare")
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
        command = [sys.executable, ROOT / "scripts" / "reproduce_lab.py", "--output-dir", args.output_dir,
                   "--extras", "dev,lab,mcp"]
        if args.temporary_root:
            command += ["--temporary-root", args.temporary_root]
        if args.no_compare:
            command.append("--no-compare")
        if args.blas_core:
            command += ["--blas-core", args.blas_core]
        for role, path in providers.items():
            command += ["--provider", f"{role}={path}"]
        if sys.version_info >= (3, 12):
            # PLSR and FTR require Python 3.12; the clean-room interpreter hosts both.
            command += ["--extras", "dev,lab,mcp,plsr", "--provider", "plsr-python=@venv", "--provider", "ftr-python=@venv"]
        call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
