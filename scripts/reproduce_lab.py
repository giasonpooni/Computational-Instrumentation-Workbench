"""One-command clean-room reproduction of the computational-experimentalist queue.

Builds a wheel from this checkout (with pip build isolation), installs it with
the lab extras into a new virtual environment outside the checkout, runs the
lab tests with a JUnit record, runs every queue task with the given provider
bindings, and compares the fresh reports with the retained ones in ``lab/``.
Retained operator hardware runs (``lab/hardware/<run-id>/``) are not
recomputed here (one whose physical findings rest on a probe of the capture
host's hardware cannot be); ``ciw lab verify`` (or ``ciw lab hardware verify``
with ``--no-compare``) checks them for integrity only, and ``gate.json`` names
them. Retained SP1 proved-heat gate records (``lab/proved-heat/<run-id>/``) are
likewise checked for integrity only (``ciw lab proved-heat verify`` with
``--no-compare``); T099 reads the one bound as ``proved-heat-record``. So are
retained second-platform figure records (``lab/figure-platforms/<record-id>/``,
``ciw lab figure-platform verify``); T158 reads the one bound as
``figure-platform-record``.
Each binding also reaches the lab tests as the ``CIW_LAB_*`` variable their
provider-gated tests read. The retained run binds CSG, FTR, SCR, the exchange
SET, PPDA and SCR checkouts, the telemetry stack and the Python 3.12 PLSR/FTR interpreter; ``scripts/check_lab.py``
provisions exactly that, and a comparison without those bindings fails. ``--blas-core`` runs the
clean room on another OpenBLAS kernel (``OPENBLAS_CORETYPE``), so the retained
evidence can be verified on the kernels other hosts would pick; ``gate.json``
records the kernel the clean room's NumPy ran. Nothing here acquires
physical measurements; hardware-dependent tasks are reported as blocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]
# Provider bindings as the provider-gated lab tests read them, so those tests run
# against the providers bound to the queue instead of skipping.
TEST_VARIABLES = {"csg": "CIW_LAB_CSG_REPO", "ftr": "CIW_LAB_FTR_REPO", "scr": "CIW_LAB_SCR_REPO",
                  "set": "CIW_LAB_SET_REPO", "ppda": "CIW_LAB_PPDA_REPO",
                  "scr-exchange": "CIW_LAB_SCR_EXCHANGE_REPO", "scr-engine": "CIW_LAB_SCR_ENGINE",
                  "ftr-python": "CIW_LAB_FTR_PYTHON", "plsr-python": "CIW_LAB_PLSR_PYTHON",
                  "proved-heat-record": "CIW_LAB_PROVED_HEAT_RECORD", "telemetry-stack": "CIW_LAB_TELEMETRY_STACK",
                  "julia": "CIW_LAB_JULIA_EXECUTABLE", "julia-depot": "CIW_LAB_JULIA_DEPOT",
                  "figure-platform-record": "CIW_LAB_FIGURE_PLATFORM_RECORD"}
# Operator hardware captures the energy tasks read; the gate acquires and analyzes none.
OPERATOR_CAPTURES = ("CIW_LAB_RAPL_LOG", "CIW_LAB_ENERGY_LOG", "CIW_LAB_NVIDIA_SMI_CSV", "CIW_LAB_NVIDIA_SMI_UTC_OFFSET")
# The clean room reproduces the packaged queue: interpreter paths, pytest options, queue
# extensions, an OpenBLAS kernel choice (--blas-core makes it), provider-test bindings and
# operator captures of the calling shell never reach it.
INHERITED_EXCLUDED = ("PYTHONPATH", "PYTEST_ADDOPTS", "CIW_LAB_EXTENSIONS", "CIW_LAB_MODULES", "OPENBLAS_CORETYPE",
                      *TEST_VARIABLES.values(), *OPERATOR_CAPTURES)
# Asked of the clean-room interpreter: the installed wheel's own reader, the one T094's platform fingerprint uses.
BLAS_CORE_QUERY = "from ciw.lab.blas_probe import openblas_core; print(openblas_core() or '')"


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


def pytest_ini() -> str:
    """The checkout's pytest markers (``lab_task``) as a ``pytest.ini``: the clean room copies no pyproject.toml."""
    markers = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    return "[pytest]\nmarkers =\n" + "".join(f"    {marker}\n" for marker in markers["markers"])


def clean_room_environment(work: Path, providers=(), blas_core=None) -> dict:
    environment = {key: value for key, value in os.environ.items() if key not in INHERITED_EXCLUDED}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CIW_LAB_REPOSITORY_ROOT"] = str(work)
    environment.update({TEST_VARIABLES[role]: path for role, path in providers if role in TEST_VARIABLES})
    # Single-threaded BLAS keeps reduction order, and so retained values, stable.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    if blas_core:
        # OpenBLAS then runs these kernels instead of the ones it picks for the CPU; blas_core_problem checks it did.
        environment["OPENBLAS_CORETYPE"] = blas_core
    return environment


def clean_room_blas_core(python, work: Path, environment: dict) -> str | None:
    """The OpenBLAS kernel the clean-room NumPy runs; None where its BLAS names none."""
    answer = subprocess.run([python, "-c", BLAS_CORE_QUERY], check=True, capture_output=True, text=True,
                            cwd=work, env=environment).stdout.strip()
    return answer if re.fullmatch(r"\w+", answer) else None


def blas_core_problem(requested, reported) -> str | None:
    """Why a clean room asked to run ``requested`` kernels cannot: OpenBLAS silently ignores a name it does not know."""
    if requested and (reported or "").lower() != requested.lower():
        return (f"The clean-room NumPy runs the OpenBLAS kernel {reported or '(none reported)'}, not the requested "
                f"{requested}; pass a kernel name as OpenBLAS reports it (for example Haswell or Sandybridge) on a "
                "NumPy that bundles OpenBLAS")
    return None


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
    parser.add_argument("--blas-core", metavar="CORE",
                        help="OpenBLAS kernel for the clean room (OPENBLAS_CORETYPE, e.g. Haswell or Sandybridge); "
                             "the run is refused unless the clean-room NumPy then reports it")
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
        (work / "pytest.ini").write_text(pytest_ini(), encoding="utf-8")  # the rootdir's configuration
        # Tasks that read example inputs find these copies, never the checkout.
        shutil.copytree(ROOT / "examples", work / "examples")
        for name in ("docs",):
            shutil.copytree(ROOT / name, work / name)
        providers = bindings(args.provider, python)
        environment = clean_room_environment(work, providers, args.blas_core)
        located = subprocess.run([python, "-c", "import ciw, sys; print(ciw.__file__)"], check=True,
                                 capture_output=True, text=True, cwd=work, env=environment).stdout.strip()
        if Path(located).resolve().is_relative_to(ROOT):
            raise SystemExit("The clean-room interpreter imported ciw from the checkout")
        blas_core = clean_room_blas_core(python, work, environment)
        # Named before any step can fail, so a failed gate's log still says which kernel the clean room ran.
        print(f"Clean room runs OpenBLAS kernel {blas_core or '(none reported)'} "
              f"(requested: {args.blas_core or 'none'})", flush=True)
        problem = blas_core_problem(args.blas_core, blas_core)
        if problem:
            raise SystemExit(problem)
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
        # Hardware runs were made on their capture host; the clean room checks their integrity and recomputes nothing.
        hardware_runs = sorted(path.name for path in (args.retained / "hardware").iterdir()
                               if path.is_dir()) if (args.retained / "hardware").is_dir() else []
        # Proved-heat gate records were made on their recording host; they too are checked for integrity only.
        proved_heat_records = sorted(path.name for path in (args.retained / "proved-heat").iterdir()
                                     if path.is_dir()) if (args.retained / "proved-heat").is_dir() else []
        # Second-platform figure records were made by a CI run on another platform; checked for integrity only.
        figure_platform_records = sorted(path.name for path in (args.retained / "figure-platforms").iterdir()
                                         if path.is_dir()) if (args.retained / "figure-platforms").is_dir() else []
        if not args.no_compare:
            run([python, "-m", "ciw", "lab", "verify", "--retained", str(args.retained.resolve()), "--fresh", str(output)],
                cwd=work, env=environment)
        else:
            if hardware_runs:
                run([python, "-m", "ciw", "lab", "hardware", "verify", "--retained", str(args.retained.resolve())],
                    cwd=work, env=environment)
            if proved_heat_records:
                run([python, "-m", "ciw", "lab", "proved-heat", "verify", "--retained", str(args.retained.resolve())],
                    cwd=work, env=environment)
            if figure_platform_records:
                run([python, "-m", "ciw", "lab", "figure-platform", "verify", "--retained",
                     str(args.retained.resolve())], cwd=work, env=environment)
    # The record names the bindings the queue and tests received, the clean-room interpreter's version and the
    # OpenBLAS kernel its NumPy ran (None where NumPy's BLAS names none), with the kernel requested, if any.
    (output / "gate.json").write_text(json.dumps({"schema": "ciw.lab-clean-room-gate.v1", "wheel_sha256": wheel_sha256,
                                                   "compared_with": None if args.no_compare else str(args.retained),
                                                   "providers": [f"{role}={path}" for role, path in providers],
                                                   "python": sys.version.split()[0],
                                                   "openblas_core": blas_core, "openblas_coretype": args.blas_core,
                                                   "hardware_runs_verified_for_integrity": hardware_runs,
                                                   "proved_heat_records_verified_for_integrity": proved_heat_records,
                                                   "figure_platform_records_verified_for_integrity":
                                                       figure_platform_records,
                                                   "physical_validation": "not_established"}, indent=2) + "\n")
    print(f"PASS: clean-room lab queue from wheel {wheel_sha256[:16]} on OpenBLAS kernel "
          f"{blas_core or '(none reported)'}; reports in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
