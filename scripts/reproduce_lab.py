"""One-command clean-room reproduction of the computational-experimentalist queue.

Builds a wheel from this checkout (with pip build isolation), installs it with
the lab extras into a new virtual environment outside the checkout, runs the
lab tests with a JUnit record, runs every queue task with the given provider
bindings, and compares the fresh reports with the retained ones in ``lab/``.
Each binding also reaches the lab tests as the ``CIW_LAB_*`` variable their
provider-gated tests read. The retained run binds CSG, FTR, SCR, the exchange
SET, PPDA and SCR checkouts and the Python 3.12 PLSR/FTR interpreter; ``scripts/check_lab.py`` provisions exactly
that, and a comparison without those bindings fails. Nothing here acquires
physical measurements; hardware-dependent tasks are reported as blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
# Provider bindings as the provider-gated lab tests read them, so those tests run
# against the providers bound to the queue instead of skipping.
TEST_VARIABLES = {"csg": "CIW_LAB_CSG_REPO", "ftr": "CIW_LAB_FTR_REPO", "scr": "CIW_LAB_SCR_REPO",
                  "set": "CIW_LAB_SET_REPO", "ppda": "CIW_LAB_PPDA_REPO",
                  "scr-exchange": "CIW_LAB_SCR_EXCHANGE_REPO", "scr-engine": "CIW_LAB_SCR_ENGINE",
                  "ftr-python": "CIW_LAB_FTR_PYTHON", "plsr-python": "CIW_LAB_PLSR_PYTHON"}
# Operator hardware captures the energy tasks read; the gate acquires and analyzes none.
OPERATOR_CAPTURES = ("CIW_LAB_RAPL_LOG", "CIW_LAB_ENERGY_LOG", "CIW_LAB_NVIDIA_SMI_CSV", "CIW_LAB_NVIDIA_SMI_UTC_OFFSET")
# The clean room reproduces the packaged queue: interpreter paths, pytest options, queue
# extensions, provider-test bindings and operator captures of the calling shell never reach it.
INHERITED_EXCLUDED = ("PYTHONPATH", "PYTEST_ADDOPTS", "CIW_LAB_EXTENSIONS", "CIW_LAB_MODULES",
                      *TEST_VARIABLES.values(), *OPERATOR_CAPTURES)


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def retained_problem(retained: Path) -> str | None:
    """Why ``retained`` cannot be compared with: comparing no reports would verify nothing."""
    reports = Path(retained) / "reports"
    if not reports.is_dir() or not any(reports.glob("T*.json")):
        return (f"No retained reports to compare with in {reports}; retain a reviewed run first "
                "(scripts/refresh_lab.py) or pass --no-compare")
    return None


def bindings(providers, python: Path) -> list:
    """``ROLE=PATH`` bindings with absolute paths (the clean room runs elsewhere); '@venv' is ``python``."""
    resolved = []
    for binding in providers:
        role, _, path = binding.partition("=")
        if path == "@venv":
            path = str(python)
        elif path:  # an empty path is left for `ciw lab run` to refuse
            # Symlinks are kept: a virtual environment's python links to a base interpreter without its packages.
            path = os.path.abspath(path)
        resolved.append((role, path))
    return resolved


def clean_room_environment(work: Path, providers=()) -> dict:
    environment = {key: value for key, value in os.environ.items() if key not in INHERITED_EXCLUDED}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CIW_LAB_REPOSITORY_ROOT"] = str(work)
    environment.update({TEST_VARIABLES[role]: path for role, path in providers if role in TEST_VARIABLES})
    # Single-threaded BLAS keeps reduction order, and so retained values, stable.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "lab")
    parser.add_argument("--retained", type=Path, default=ROOT / "lab",
                        help="Retained reports to compare with; pass --no-compare to skip")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--provider", action="append", default=[], metavar="ROLE=PATH",
                        help="Provider binding; PATH '@venv' names the clean-room interpreter")
    parser.add_argument("--extras", default="dev,lab",
                        help="Wheel extras to install, e.g. dev,lab,plsr on Python 3.12+")
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    problem = None if args.no_compare else retained_problem(args.retained)
    if problem:
        raise SystemExit(problem)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ciw-lab-clean-room-", dir=args.temporary_root) as directory:
        work = Path(directory).resolve()  # relative on Python 3.11; subprocesses run inside it
        build = work / "build"
        shutil.copytree(ROOT / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, build / name)
        # Build isolation supplies the declared build backend (setuptools>=77); the host needs only pip.
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", str(build), "--wheel-dir", str(work / "dist")])
        wheel = next((work / "dist").glob("*.whl"))
        wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
        venv.EnvBuilder(with_pip=True).create(work / "venv")
        python = work / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        run([python, "-m", "pip", "install", "--quiet", f"{wheel}[{args.extras}]"])
        tests = work / "tests"
        tests.mkdir()
        for path in sorted((ROOT / "tests").glob("test_lab_*.py")):
            shutil.copy2(path, tests / path.name)
        if (ROOT / "tests" / "fixtures" / "lab").is_dir():
            shutil.copytree(ROOT / "tests" / "fixtures" / "lab", tests / "fixtures" / "lab")
        # Tasks that read example inputs find these copies, never the checkout.
        shutil.copytree(ROOT / "examples", work / "examples")
        for name in ("docs",):
            shutil.copytree(ROOT / name, work / name)
        providers = bindings(args.provider, python)
        environment = clean_room_environment(work, providers)
        located = subprocess.run([python, "-c", "import ciw, sys; print(ciw.__file__)"], check=True,
                                 capture_output=True, text=True, cwd=work, env=environment).stdout.strip()
        if Path(located).resolve().is_relative_to(ROOT):
            raise SystemExit("The clean-room interpreter imported ciw from the checkout")
        junit = output / "tests.xml"
        run([python, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(work),
             "--junitxml", str(junit), str(tests)], cwd=work, env=environment)
        environment["CIW_LAB_CLEAN_ROOM"] = json.dumps({"wheel_sha256": wheel_sha256, "wheel_path": str(wheel),
                                                        "python": sys.version.split()[0]})
        command = [python, "-m", "ciw", "lab", "run", "--all", "--output-dir", str(output), "--junit", str(junit)]
        for role, path in providers:
            command += ["--provider", f"{role}={path}"]
        run(command, cwd=work, env=environment)
        # The dashboard is rendered by the installed wheel, so refreshing lab/ needs no CIW dependencies on the host.
        run([python, "-m", "ciw", "lab", "dashboard", "--retained", str(output), "--output", str(output / "index.html")],
            cwd=work, env=environment)
        if not args.no_compare:
            run([python, "-m", "ciw", "lab", "verify", "--retained", str(args.retained.resolve()), "--fresh", str(output)],
                cwd=work, env=environment)
    # The record names the bindings the queue and tests received, and the clean-room interpreter's version.
    (output / "gate.json").write_text(json.dumps({"schema": "ciw.lab-clean-room-gate.v1", "wheel_sha256": wheel_sha256,
                                                   "compared_with": None if args.no_compare else str(args.retained),
                                                   "providers": [f"{role}={path}" for role, path in providers],
                                                   "python": sys.version.split()[0],
                                                   "physical_validation": "not_established"}, indent=2) + "\n")
    print(f"PASS: clean-room lab queue from wheel {wheel_sha256[:16]}; reports in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
