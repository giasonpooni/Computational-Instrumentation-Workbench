"""Installed-wheel gate for the language-neutral model core and the pinned Julia worker.

Provisioning is separate from execution: the gate verifies or downloads the
pinned Julia 1.10.12 distribution, instantiates the worker's committed
Manifest.toml into a depot, then runs the model tests from an installed CIW
wheel with the genuine worker and the pinned SCR checkout. Skipped tests fail
the gate. Protocol doubles cover failure paths only; the numerical acceptance
tests run against the real worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

TESTS = ("test_model_spec.py", "test_model_transform.py", "test_model_compose.py", "test_model_latex.py",
         "test_model_codec.py", "test_model_worker_protocol.py", "test_model_julia.py",
         "test_model_scr_boundary.py", "test_model_cli.py")
LATEX_TEST = "tests/test_model_latex.py::test_generated_documents_compile"


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 1800), **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def platform_key():
    return f"{platform.system()}-{platform.machine().replace('AMD64', 'x86_64')}"


def resolve_julia(pin, supplied, temporary):
    key = platform_key()
    if supplied is None:
        distribution = pin["distribution"].get(key)
        if distribution is None:
            raise RuntimeError(f"No pinned Julia distribution for {key}; supply --julia")
        archive = temporary / "julia.tar.gz"
        urllib.request.urlretrieve(distribution["url"], archive)
        if sha256(archive) != distribution["tarball_sha256"]:
            raise RuntimeError("Downloaded Julia archive differs from its pinned checksum")
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary / "julia-dist", filter="data")
        supplied = next((temporary / "julia-dist").glob("julia-*/bin/julia"))
    julia = Path(supplied).resolve(strict=True)
    pins = pin["platforms"].get(key)
    if pins is None:
        raise RuntimeError(f"Julia runtime digests for {key} are not pinned yet: {pin['unverified_platforms'].get(key)}")
    if sha256(julia) != pins["julia_executable_sha256"]:
        raise RuntimeError("The Julia executable differs from the pinned runtime")
    version = call([str(julia), "--startup-file=no", "-e", "print(VERSION)"], capture_output=True, text=True).stdout
    if version.strip() != pin["julia_version"]:
        raise RuntimeError(f"Julia {version.strip()} is not the pinned {pin['julia_version']}")
    return julia


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", type=Path, help="Existing pinned Julia executable; otherwise download and verify")
    parser.add_argument("--julia-depot", type=Path, help="Depot to instantiate into (default: a fresh temporary depot)")
    parser.add_argument("--scr-repo", type=Path, help="Clean exact SCR checkout; otherwise clone the pin")
    parser.add_argument("--without-latex-compile", action="store_true",
                        help="Deselect the pdflatex compile check (recorded as not run)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/model-core-gate"))
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise RuntimeError("The model core requires Python 3.11 or newer")
    if not args.without_latex_compile and shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex is required to check generated LaTeX; pass --without-latex-compile to record it as not run")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    # The provider descriptor is the worker's pin definition, including its SCR boundary.
    descriptor = json.loads((root / "src/ciw/pipelines/providers/julia-model-worker.json").read_text(encoding="utf-8"))
    pin = descriptor["pin"]
    scr_pin, = (item["pin"] for item in descriptor["boundary"] if item["role"] == "scr")
    scr_revision = scr_pin["revision"]
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    timings = {}
    with tempfile.TemporaryDirectory(prefix="ciw-model-") as directory:
        temporary = Path(directory)
        julia = resolve_julia(pin, args.julia, temporary)
        depot = (args.julia_depot or temporary / "depot").resolve()
        depot.mkdir(parents=True, exist_ok=True)
        if args.scr_repo is not None:
            scr = args.scr_repo.resolve(strict=True)
        else:
            scr = temporary / "scr"
            call(["git", "clone", "--quiet", "--no-checkout", scr_pin["repository"] + ".git", str(scr)])
            call(["git", "-C", str(scr), "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", scr_revision])
        from provider_checkouts import validate_checkout
        validate_checkout(scr, scr_revision)

        # Snapshot sources, tests and examples together before the long
        # provisioning step, so the wheel and its tests describe one revision.
        work = temporary / "check"
        (work / "tests" / "fixtures").mkdir(parents=True)
        for name in TESTS:
            shutil.copyfile(root / "tests" / name, work / "tests" / name)
        shutil.copyfile(root / "tests/fixtures/fake_model_worker.py", work / "tests/fixtures/fake_model_worker.py")
        shutil.copytree(root / "examples/model-core", work / "examples/model-core")
        build = temporary / "build"
        shutil.copytree(root / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copyfile(root / name, build / name)
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build), "-w", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "env"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", "--quiet", str(wheel), "pytest==9.0.2"])
        worker_root = Path(call([str(python), "-I", "-c", "from ciw.model.worker import worker_root; print(worker_root())"],
                                capture_output=True, text=True).stdout.strip())
        if not worker_root.resolve().is_relative_to(environment.resolve()):
            raise AssertionError("The gate must use the worker shipped in the installed wheel")
        committed = {name: sha256(worker_root / name) for name in ("Project.toml", "Manifest.toml")}

        julia_env = {key: value for key, value in os.environ.items() if not key.startswith("JULIA_")}
        julia_env.update(JULIA_DEPOT_PATH=str(depot), JULIA_LOAD_PATH="@" + os.pathsep + "@stdlib")
        started = time.perf_counter()
        call([str(julia), "--startup-file=no", "--threads=1", f"--project={worker_root}", "-e",
              "using Pkg; Pkg.instantiate(); Pkg.precompile(strict=true)"], env=julia_env, timeout=3600)
        timings["instantiate_and_precompile_s"] = time.perf_counter() - started
        if {name: sha256(worker_root / name) for name in committed} != committed:
            raise AssertionError("Instantiation changed the committed Project.toml or Manifest.toml")

        env = {key: value for key, value in os.environ.items() if not key.startswith("JULIA_")}
        env.update(CIW_JULIA=str(julia), CIW_JULIA_DEPOT=str(depot), CIW_SCR_REPO=str(scr))
        env.pop("PYTHONPATH", None)
        report = destination / "tests.xml"
        selection = ["--deselect", LATEX_TEST] if args.without_latex_compile else []
        started = time.perf_counter()
        call([str(python), "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(report),
              *selection, *["tests/" + name for name in TESTS]], cwd=work, env=env, timeout=3600)
        timings["tests_s"] = time.perf_counter() - started
        suites = list(ET.parse(report).getroot().iter("testsuite"))
        if any(int(suite.get("skipped", 0)) for suite in suites):
            raise AssertionError("The model-core gate cannot pass skipped tests")
        validate_checkout(scr, scr_revision)
        summary = {
            "gate": "model-core", "installed_wheel": wheel.name, "platform": platform_key(),
            "julia": {"version": pin["julia_version"], "executable_sha256": sha256(julia)},
            "worker": committed, "scr_revision": scr_revision,
            "tests": {"count": sum(int(s.get("tests", 0)) for s in suites),
                      "failures": sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites),
                      "skipped": 0, "latex_compile": "not_run" if args.without_latex_compile else "passed"},
            "timings_s": timings,
        }
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("PASS: installed model core, pinned Julia worker (simulate, linearize, measurement selection, "
          "ModelingToolkit), SCR dispatch, offline restore and fresh replay")


if __name__ == "__main__":
    main()
