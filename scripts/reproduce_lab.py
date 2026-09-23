"""One-command clean-room reproduction of the computational-experimentalist queue.

Builds a wheel from this checkout, installs it with the lab extras into a new
virtual environment outside the checkout, runs the lab tests with a JUnit
record, runs every queue task with optional pinned providers, and compares the
fresh reports with the retained ones in ``lab/``. Nothing here acquires
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


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


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
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ciw-lab-clean-room-", dir=args.temporary_root) as directory:
        work = Path(directory)
        build = work / "build"
        shutil.copytree(ROOT / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, build / name)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build),
             "--wheel-dir", str(work / "dist")])
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
        environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTEST_ADDOPTS")}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CIW_LAB_REPOSITORY_ROOT"] = str(work)
        # Single-threaded BLAS keeps reduction order, and so retained values, stable.
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[variable] = "1"
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
        for binding in args.provider:
            role, _, path = binding.partition("=")
            command += ["--provider", f"{role}={python}" if path == "@venv" else binding]
        run(command, cwd=work, env=environment)
        if not args.no_compare:
            run([python, "-m", "ciw", "lab", "verify", "--retained", str(args.retained.resolve()), "--fresh", str(output)],
                cwd=work, env=environment)
    (output / "gate.json").write_text(json.dumps({"schema": "ciw.lab-clean-room-gate.v1", "wheel_sha256": wheel_sha256,
                                                   "compared_with": None if args.no_compare else str(args.retained),
                                                   "providers": args.provider,
                                                   "physical_validation": "not_established"}, indent=2) + "\n")
    print(f"PASS: clean-room lab queue from wheel {wheel_sha256[:16]}; reports in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
