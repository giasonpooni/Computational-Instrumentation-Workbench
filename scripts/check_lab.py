"""Run the lab queue from an isolated wheel with pinned providers and verify retained reports.

Provisions the exact CSG, FTR and SCR revisions pinned by ``ciw.geodesic_reference``
and ``ciw.declared_workload``, the SET, PPDA and SCR revisions of
``.github/workflows/exchange.yml`` for the exchange roundtrip (T097), and the
telemetry stack of ``src/ciw/telemetry-runtimes.json`` (PPDA, STFE, GSIE, SET and
CBSR, bound together as ``telemetry-stack`` for T077's telemetry session),
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
The latest retained run of the SP1 proved-heat gate (``lab/proved-heat/<run-id>/``)
is bound as ``proved-heat-record`` from this checkout, since the clean room has
no copy of ``lab/``: T099 validates it and rebuilds its engine with the pinned
Rust toolchain when that toolchain is installed. Every retained record is
checked for integrity by ``ciw lab verify`` (or ``ciw lab proved-heat verify``
with ``--no-compare``). Likewise the latest retained second-platform figure
record (``lab/figure-platforms/<record-id>/``, a CI run of
``scripts/check_figures.py`` on Windows) is bound as ``figure-platform-record``
for T158, and every one is checked by ``ciw lab verify`` (or ``ciw lab
figure-platform verify`` with ``--no-compare``).

``--blas-core CORE`` runs the clean room on that OpenBLAS kernel (for example
Haswell or Sandybridge), to verify the retained run on the kernels other hosts
would pick; the gate record names the kernel the clean room ran.

``--julia`` and ``--julia-depot`` (or ``CIW_LAB_JULIA_EXECUTABLE`` and
``CIW_LAB_JULIA_DEPOT``) bind the Julia 1.10.12 runtime and depot that
``scripts/provision_julia.py`` provisioned as the ``julia`` and ``julia-depot``
roles, for T145's worker; like the rustup toolchain T099 uses, the gate does
not install Julia itself (CI's lab job provisions it first).
"""
from __future__ import annotations

import argparse
import ast
import json
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
# The telemetry workflow's stack (src/ciw/telemetry-runtimes.json), bound to T077 as one directory
# ``telemetry-stack`` holding a checkout per role under its repository name, as scripts/check_telemetry.py lays
# them out; ciw.lab.exchange_provenance_common.TELEMETRY_REPOSITORIES reads the same names.
TELEMETRY_STACK = "telemetry-stack"
TELEMETRY_REPOSITORIES = {"ppda": "Provenance-Preserving-Data-Acquisition",
                          "stfe": "Streaming-Telemetry-Feature-Extraction",
                          "gsie": "Geometric-State-Inference-Engine",
                          "set": "State-Estimation-Evaluation-Testbed",
                          "cbsr": "Constraint-Based-State-Reconciliation"}


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


def telemetry_pins() -> dict:
    """The telemetry stack's revisions by role, exactly as src/ciw/telemetry-runtimes.json pins them."""
    manifest = json.loads((ROOT / "src" / "ciw" / "telemetry-runtimes.json").read_text(encoding="utf-8"))
    missing = set(TELEMETRY_REPOSITORIES) - set(manifest)
    if missing:
        raise SystemExit(f"Telemetry pins not found: {sorted(missing)}")
    return {role: manifest[role]["revision"] for role in TELEMETRY_REPOSITORIES}


def provision(path: Path, repository: str, revision: str, clone: bool) -> Path:
    """Clone ``repository`` at ``revision`` into ``path`` when asked, then require the exact clean pin there."""
    if clone:
        call(["git", "-c", "core.autocrlf=false", "clone", "--quiet", f"https://github.com/giasonpooni/{repository}.git",
              path])
        call(["git", "-C", path, "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", revision])
    return validate_checkout(path, revision)


def proved_heat_record(root: Path = ROOT / "lab" / "proved-heat") -> Path | None:
    """The retained proved-heat gate record T099 reads: the latest under ``root`` by its run.json date, then run id.

    ``ciw lab verify`` checks every record; an unreadable run.json sorts first rather than hiding the others.
    """
    records = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not path.is_dir():
            continue
        try:
            date = json.loads((path / "run.json").read_text(encoding="utf-8")).get("date")
        except (OSError, ValueError, AttributeError):
            date = None
        records.append((date if isinstance(date, str) else "", path.name, path))
    return max(records)[2] if records else None


def figure_platform_record(root: Path = ROOT / "lab" / "figure-platforms") -> Path | None:
    """The retained second-platform figure record T158 reads: the latest under ``root`` by its record.json date, then
    its CI run id and run attempt (a record without one is the first attempt), then its identity.

    ``ciw lab verify`` checks every record; an unreadable record.json sorts first rather than hiding the others.
    """
    records = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not path.is_dir():
            continue
        try:
            record = json.loads((path / "record.json").read_text(encoding="utf-8"))
            source = record.get("source", {})
            date, run, attempt = record.get("date"), source.get("run_id"), source.get("run_attempt")
        except (OSError, ValueError, AttributeError):
            date, run, attempt = None, None, None
        records.append((date if isinstance(date, str) else "", run if type(run) is int else -1,
                        attempt if type(attempt) is int else 1, path.name, path))
    return max(records)[4] if records else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lab-gate")
    parser.add_argument("--stack-root", type=Path, default=os.environ.get("CIW_LAB_STACK_ROOT"),
                        help="Existing clean checkouts named csg, ftr, scr, set, ppda and scr-exchange, and a "
                             "telemetry-stack directory holding the telemetry checkouts under their repository "
                             "names; cloned when absent")
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--blas-core", metavar="CORE",
                        help="OpenBLAS kernel for the clean room (e.g. Haswell or Sandybridge); see reproduce_lab.py")
    parser.add_argument("--julia", type=Path, default=os.environ.get("CIW_LAB_JULIA_EXECUTABLE"),
                        help="Julia 1.10.12 executable provisioned by scripts/provision_julia.py (T145's worker)")
    parser.add_argument("--julia-depot", type=Path, default=os.environ.get("CIW_LAB_JULIA_DEPOT"),
                        help="The depot scripts/provision_julia.py instantiated the worker environment in")
    args = parser.parse_args()
    if (args.julia is None) != (args.julia_depot is None):
        raise SystemExit("Bind Julia with both --julia and --julia-depot (scripts/provision_julia.py prints them)")
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
    revisions, telemetry = pins(), telemetry_pins()
    with tempfile.TemporaryDirectory(prefix="ciw-lab-providers-", dir=args.temporary_root) as directory:
        root = Path(args.stack_root) if args.stack_root else Path(directory)
        providers = {role: provision(root / role, repository, revisions[role], not args.stack_root)
                     for role, repository in REPOSITORIES.items()}
        stack = root / TELEMETRY_STACK
        for role, repository in TELEMETRY_REPOSITORIES.items():
            provision(stack / repository, repository, telemetry[role], not args.stack_root)
        providers[TELEMETRY_STACK] = stack.resolve()
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
        record = proved_heat_record()
        if record is not None:
            command += ["--provider", f"proved-heat-record={record}"]
        if args.julia is not None:
            # T145 runs its Julia worker with this executable and depot (provisioned, never resolved, here).
            command += ["--provider", f"julia={Path(os.path.abspath(args.julia))}",
                        "--provider", f"julia-depot={Path(os.path.abspath(args.julia_depot))}"]
        record = figure_platform_record()
        if record is not None:
            command += ["--provider", f"figure-platform-record={record}"]
        if sys.version_info >= (3, 12):
            # PLSR and FTR require Python 3.12; the clean-room interpreter hosts both.
            command += ["--extras", "dev,lab,mcp,plsr", "--provider", "plsr-python=@venv", "--provider", "ftr-python=@venv"]
        call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
