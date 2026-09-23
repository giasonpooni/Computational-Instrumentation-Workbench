"""Run all three real geometry providers through an isolated installed CIW wheel."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from provider_checkouts import validate_checkout

REPOSITORIES = {"cggt": "Covariance-Geometry-and-Geodesic-Testbed",
    "isgt": "Intrinsic-Surface-Geodesics-Testbed", "tsde": "Translation-Surface-Dynamics-Explorer"}
TESTS = ("test_geometry_research.py", "test_geometry_research_session.py",
         "test_geometry_covariance_contract.py", "test_geometry_mesh_contract.py",
         "test_geometry_translation_contract.py")


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def pins(path):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PINS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing exact provider pins")


def exact_source(path, pin):
    path = validate_checkout(path, pin["revision"])
    tree = call(["git", "--no-replace-objects", "-C", str(path), "rev-parse", "HEAD^{tree}"],
        capture_output=True, text=True).stdout.strip()
    if tree != pin["source_tree"]:
        raise ValueError("Provider tree differs from its approved pin")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing exact clean cggt/isgt/tsde checkouts; no fetching")
    parser.add_argument("--output-dir", type=Path, default=Path("results/geometry-research"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    declarations = pins(root / "src/ciw/geometry_research.py")
    by_role = {pin["role"]:pin for pin in declarations.values()}
    if set(by_role) != set(REPOSITORIES):
        raise ValueError("Require all three geometry providers")
    destination = args.output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Gate output must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ciw-geometry-") as directory:
        temporary, providers = Path(directory), {}
        for role, repository in REPOSITORIES.items():
            path = (args.stack_root / role).resolve() if args.stack_root else temporary / role
            if not args.stack_root:
                call(["git", "-c", "core.autocrlf=false", "clone", "--quiet", "--no-checkout",
                    "https://github.com/giasonpooni/" + repository + ".git", str(path)])
                call(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", by_role[role]["revision"]])
            providers[role] = exact_source(path, by_role[role])
        build = temporary / "build"
        shutil.copytree(root / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            if (root / name).is_file():
                shutil.copyfile(root / name, build / name)
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build), "-w", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for name in TESTS:
            shutil.copyfile(root / "tests" / name, work / "tests" / name)
        shutil.copytree(root / "examples/geometry-research", work / "examples/geometry-research")
        env = {**os.environ, **{"CIW_" + role.upper() + "_REPO":str(path) for role,path in providers.items()},
               "CIW_GEOMETRY_FIXTURE_DIR": str(destination), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        env.pop("PYTEST_ADDOPTS", None)
        imported = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=env, text=True, capture_output=True)
        if not Path(imported.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Gate must use the installed wheel")
        report = destination / "tests.xml"
        call([str(python), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(report),
              *["tests/" + name for name in TESTS]], cwd=work, env=env, timeout=900)
        cases = list(ET.parse(report).getroot().iter("testcase"))
        if not cases or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
            raise AssertionError("Native geometry gate requires completed tests with no skips or failures")
        for role,path in providers.items():
            exact_source(path, by_role[role])
        (destination / "gate.json").write_text(json.dumps({"schema":"ciw.geometry-provider-gate.v1", "status":"passed",
            "tests_passed":len(cases), "providers":by_role, "installed_wheel":wheel.name,
            "checks":"real_native_results_shared_session_offline_restore_and_fresh_replay",
            "independent_algorithm_certification":"not_established", "physical_calibration":"not_established"}, indent=2) + "\n", encoding="utf-8")
    print("PASS: three native mathematical providers, shared session, offline restore and fresh replay")


if __name__ == "__main__":
    main()
