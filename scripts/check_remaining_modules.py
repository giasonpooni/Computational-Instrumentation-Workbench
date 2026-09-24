"""Check three remaining native instruments together from an isolated CIW wheel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provider_checkouts import descriptor_pins  # noqa: E402


IDENTIFIED_REPOSITORIES = {
    "fsrt": "Fluid-State-Reconstruction-Testbed", "tbrt": "Time-Base-Reconciliation-Runtime",
    "mcur": "Metrological-Calibration-Uncertainty-Runtime", "oit": "Observability-Identifiability-Testbed",
    "gsie": "Geometric-State-Inference-Engine", "cbsr": "Constraint-Based-State-Reconciliation",
    "fdir": "Fault-Detection-Isolation-Runtime", "set": "State-Estimation-Evaluation-Testbed",
    "sidt": "System-Identification-Dynamics-Testbed", "edspt": "Experiment-Design-Sensor-Placement-Testbed",
    "ywir": "Yield-Weighted-Inference-Runtime",
}
MEASUREMENT_REPOSITORIES = {
    "rci": "Retrofitted-Computational-Instrumentation", "fsrt": "Fluid-State-Reconstruction-Testbed",
    "jspt": "Jacobian-Sensitivity-Propagation-Testbed",
}
# The live common bench retains the actual identified upstream consumed by
# the following PLSR unit checks; no stale external fixture is substituted.
TEST_FILES = ("test_remaining_modules_gate.py", "test_measurement_chain.py",
              "test_geometric_circle.py", "test_identified_stability.py")


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def exact_source(repository, revision):
    actual = call(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"Require exact provider revision {revision}: {repository}")
    dirty = call(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
                 capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError(f"Require clean provider sources: {repository}")
    return call(["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-chain-stack-root", type=Path, help="Exact role-named RCI/FSRT/JSPT checkouts")
    parser.add_argument("--identified-design-stack-root", type=Path, help="Separate exact eleven-provider identified stack")
    parser.add_argument("--geometry-repo", type=Path, help="Exact clean GTE checkout")
    parser.add_argument("--stability-repo", type=Path, help="Exact clean PLSR checkout")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ciw-remaining-module-fixtures"))
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("The identified upstream YWIR provider requires Python 3.12 or newer")
    root = Path(__file__).resolve().parents[1]
    package = root / "src/ciw"
    process = descriptor_pins("calibrated-observable", root)
    design = descriptor_pins("identified-design", root)
    if any(design.get(role) != pin for role, pin in process.items()) or set(design) != set(IDENTIFIED_REPOSITORIES):
        raise ValueError("The identified upstream requires its complete unchanged eleven-provider pin set")
    identified_pins = {role: pin["revision"] for role, pin in design.items()}
    measurement = descriptor_pins("measurement-chain", root)
    if set(measurement) != set(MEASUREMENT_REPOSITORIES):
        raise ValueError("The measurement chain requires exactly RCI, FSRT and JSPT")
    measurement_pins = {role: pin["revision"] for role, pin in measurement.items()}
    # The pipeline descriptor is the GTE pin definition the module executes.
    geometry = json.loads((package / "pipelines/descriptors/geometric-circle.json").read_text())["steps"][0]["pin"]["revision"]
    stability = json.loads((package / "pipelines/providers/plsr.json").read_text())["pin"]["commit"]
    with tempfile.TemporaryDirectory(prefix="ciw-remaining-modules-gate-") as directory:
        temporary = Path(directory)
        checked = []

        def checkout(destination, repository, revision, existing):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if existing is not None:
                existing = existing.resolve(strict=True)
                tree = exact_source(existing, revision)
                destination.symlink_to(existing, target_is_directory=True)
            else:
                call(["git", "clone", "--quiet", "--no-checkout",
                      "https://github.com/giasonpooni/" + repository + ".git", str(destination)])
                call(["git", "-C", str(destination), "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", revision])
                tree = exact_source(destination, revision)
            checked.append((destination, revision, tree))

        identified = temporary / "identified"
        for role, repository in IDENTIFIED_REPOSITORIES.items():
            checkout(identified / role, repository, identified_pins[role],
                     args.identified_design_stack_root / role if args.identified_design_stack_root else None)
        measurement_root = temporary / "measurement"
        for role, repository in MEASUREMENT_REPOSITORIES.items():
            checkout(measurement_root / role, repository, measurement_pins[role],
                     args.measurement_chain_stack_root / role if args.measurement_chain_stack_root else None)
        gte, plsr = temporary / "gte", temporary / "plsr"
        checkout(gte, "Geometric-Telemetry-Engine", geometry, args.geometry_repo)
        checkout(plsr, "Parameterized-Lyapunov-Stability-Runtime", stability, args.stability_repo)
        # Build from a fresh source snapshot so old setuptools build products
        # cannot contaminate the wheel or collide with another local gate.
        build_source = temporary / "build-source"
        shutil.copytree(root / "src", build_source / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            if (root / name).is_file():
                shutil.copyfile(root / name, build_source / name)
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build_source), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel) + "[bench-models]", "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for filename in TEST_FILES:
            shutil.copyfile(root / "tests" / filename, work / "tests" / filename)
        shutil.copytree(root / "examples", work / "examples")
        env = {**os.environ, "CIW_REMAINING_MEASUREMENT_STACK_ROOT": str(measurement_root),
               "CIW_REMAINING_IDENTIFIED_STACK_ROOT": str(identified), "CIW_REMAINING_GTE_REPO": str(gte),
               "CIW_REMAINING_PLSR_REPO": str(plsr), "CIW_REMAINING_FIXTURE_DIR": str(args.output_dir.resolve()),
               "CIW_MEASUREMENT_CHAIN_STACK_ROOT": str(measurement_root), "CIW_GTE_REPO": str(gte),
               "CIW_IDENTIFIED_DESIGN_STACK_ROOT": str(identified), "CIW_IDENTIFIED_STABILITY_REPO": str(plsr),
               "CIW_IDENTIFIED_STABILITY_UPSTREAM": str(args.output_dir.resolve() / "identified-stability/upstream.json")}
        env.pop("PYTHONPATH", None)
        location = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work,
                        env=env, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Gate must import its isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(python), "-I", "-m", "pytest", "-q", *["tests/" + name for name in TEST_FILES], "--junitxml", str(report)],
             cwd=work, env=env, timeout=1800)
        if any(int(s.attrib.get("skipped", 0)) for s in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Remaining-modules gate cannot skip native provider tests")
        for repository, revision, tree in checked:
            if exact_source(repository, revision) != tree:
                raise AssertionError("Provider source tree changed during installed execution")
    print("PASS: installed native measurement, geometry and stability; shared live catalog, inspection, restore and replay")


if __name__ == "__main__":
    main()
