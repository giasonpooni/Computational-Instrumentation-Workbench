"""Regenerate the retained lab evidence in ``lab/`` from a clean-room run.

Runs ``scripts/check_lab.py --no-compare`` (isolated wheel, pinned providers,
lab tests, whole queue) under Python 3.12+, or takes such a run with
``--from-run`` (it must have run under Python 3.12+ with CSG, FTR, SCR, the
exchange SET, PPDA and SCR checkouts, the telemetry stack and the PLSR/FTR
interpreter bound, none of them refused and T077's telemetry session run, as
in the CI comparison),
then replaces the retained reports, artifacts, queue state, report book and
dashboard (rendered inside the clean room by the installed wheel) with the
fresh ones.
Elapsed times, JUnit records and gate records stay with the run output: they
are machine-specific and not retained evidence. Retained operator hardware
runs (``lab/hardware/``), retained SP1 proved-heat gate records
(``lab/proved-heat/``) and ``lab/README.md`` are never touched: a clean-room
run cannot reproduce a hardware run or a proved-heat gate run, which are
retained with ``ciw lab hardware retain`` and ``ciw lab proved-heat retain``.
When ``lab/proved-heat/`` holds a record, the run must have bound one as
``proved-heat-record`` (``scripts/check_lab.py`` does, as in the CI
comparison) and T099 must have found the CI-pinned rustup toolchain, which
CI's lab gate installs. Review ``git diff lab`` and ``ciw lab verify`` output before
committing a refresh.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ("reports", "artifacts", "queue-state.json", "REPORTS.md", "index.html")
# Entries of lab/ a refresh never replaces: operator hardware runs, proved-heat gate records and the README.
PRESERVED = ("hardware", "proved-heat", "README.md")
assert not set(RETAINED) & set(PRESERVED)
# The bindings scripts/check_lab.py makes on Python 3.12+; CI compares with a run that had all of them.
REQUIRED_PROVIDERS = ("csg", "ftr", "scr", "set", "ppda", "scr-exchange", "plsr-python", "ftr-python",
                      "telemetry-stack")
# Refusal codes of those providers (CSG_TREE_MISMATCH, FTR_INTERPRETER_UNBOUND, PLSR_UNAVAILABLE, ...): a
# report carrying one did not run its bound provider, whatever the binding was named.
PROVIDER_REFUSAL = re.compile(r"\b(?:CSG|FTR|PLSR)_[A-Z]+(?:_[A-Z]+)*\b")


def pinned_toolchain_probed(run: Path) -> bool:
    """Whether T099's report in ``run`` found the CI-pinned rustup toolchain (every ``tool:cargo+<toolchain>`` probe
    succeeded, and there is one). Without it T099 is partial here and completed in CI, whose lab gate installs it."""
    try:
        report = json.loads((run / "reports" / "T099.json").read_text(encoding="utf-8"))
        probes = report["provider_runtime_identity"]["requirement_probes"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    pinned = [found for name, found in probes.items() if name.startswith("tool:cargo+")] if isinstance(
        probes, dict) else []
    return bool(pinned) and all(found is True for found in pinned)


def telemetry_session_ran(run: Path) -> bool:
    """Whether T077's report in ``run`` ran its telemetry session on the bound stack: every checkout of the
    ``telemetry-stack`` binding at its pin and the executed runtimes recorded (a bound stack that did not run
    leaves T077 as a run without the binding would)."""
    try:
        identity = json.loads((run / "reports" / "T077.json").read_text(encoding="utf-8"))["provider_runtime_identity"]
        states = [record["state"] for record in identity["telemetry-stack"].values()]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return bool(states) and all(state == "ready" for state in states) and bool(identity.get("executed_runtimes"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, default=os.environ.get("CIW_LAB_STACK_ROOT"))
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--from-run", type=Path, help="Use an existing clean-room output instead of running one")
    args = parser.parse_args()
    if args.temporary_root:
        args.temporary_root = args.temporary_root.resolve()
    if args.from_run is None and sys.version_info < (3, 12):
        raise SystemExit("Retained evidence is generated under Python 3.12+, which hosts the PLSR and FTR "
                         f"providers (this is {sys.version.split()[0]})")
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
        record = json.loads(gate.read_text(encoding="utf-8")) if gate.is_file() else {}
        if record.get("schema") != "ciw.lab-clean-room-gate.v1":
            raise SystemExit(f"Not a clean-room output (no ciw.lab-clean-room-gate.v1 gate.json): {run}")
        python = str(record.get("python") or "unrecorded")
        if tuple(int(part) for part in re.findall(r"\d+", python)[:2]) < (3, 12):
            raise SystemExit(f"The clean-room run used Python {python}; retain a run of scripts/check_lab.py "
                             "under Python 3.12+")
        bound = {binding.partition("=")[0] for binding in record.get("providers") or []}
        # T099 reads the retained proved-heat gate record; CI binds it whenever lab/proved-heat/ holds one.
        records = ROOT / "lab" / "proved-heat"
        required = REQUIRED_PROVIDERS + (("proved-heat-record",) if records.is_dir() and any(
            path.is_dir() for path in records.iterdir()) else ())
        unbound = [role for role in required if role not in bound]
        if unbound:
            raise SystemExit(f"The clean-room run bound no {', '.join(unbound)}; retain a run of "
                             "scripts/check_lab.py under Python 3.12+")
        missing = [name for name in RETAINED if not (run / name).exists()]
        if missing:
            raise SystemExit(f"Clean-room output is incomplete: {missing}")
        refused = sorted(path.stem for path in (run / "reports").glob("T*.json")
                         if PROVIDER_REFUSAL.search(path.read_text(encoding="utf-8")))
        if refused:
            raise SystemExit(f"A bound provider refused to run in {', '.join(refused)}; retain a run of "
                             "scripts/check_lab.py whose providers ran")
        if not telemetry_session_ran(run):
            raise SystemExit("T077 did not run its telemetry session on the bound telemetry stack (a checkout off "
                             "its pin, or the session failed; its report says which); retain a run of "
                             "scripts/check_lab.py whose telemetry stack ran")
        if "proved-heat-record" in required and not pinned_toolchain_probed(run):
            raise SystemExit("T099 did not find the CI-pinned rustup toolchain in the clean-room run (its "
                             "tool:cargo+<toolchain> probe), so it could not rebuild the gate's engine as CI's lab "
                             "gate does; install it (rustup toolchain install <toolchain> --profile minimal, as "
                             ".github/workflows/lab.yml does) and retain a new run")
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
    print(f"Refreshed {target}; review git diff before committing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
