"""Check installed acquisition, calibrated windows and retained residual monitoring."""
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

REPOSITORIES = {
    "ppda": "Provenance-Preserving-Data-Acquisition",
    "tbrt": "Time-Base-Reconciliation-Runtime",
    "mcur": "Metrological-Calibration-Uncertainty-Runtime",
    "stfe": "Streaming-Telemetry-Feature-Extraction",
    "gsie": "Geometric-State-Inference-Engine",
    "set": "State-Estimation-Evaluation-Testbed",
    "oit": "Observability-Identifiability-Testbed",
    "fdir": "Fault-Detection-Isolation-Runtime",
}
TEST_FILES = ("test_acquired_window.py", "test_residual_monitor.py", "test_residual_view.py", "test_acquired_stream_gate.py")


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def exact_revision(repository, revision):
    actual = call(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"Require exact provider revision {revision}: {repository}")


def pins(root):
    """Read literal pin data without importing source-tree CIW code."""
    package = root / "src/ciw"
    windows = json.loads((package / "calibrated-window-runtimes.json").read_text())
    process = json.loads((package / "calibrated-observable-runtimes.json").read_text())
    constants = {}
    for statement in ast.parse((package / "adapters/ppda_acquisition.py").read_text()).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in {"PPDA_REVISION", "VENDOR_REVISION", "VENDOR_PATH"}:
                constants[target.id] = ast.literal_eval(statement.value)
    revisions = {role: windows[role]["revision"] for role in ("tbrt", "mcur", "stfe", "gsie", "set")}
    revisions.update({role: process[role]["revision"] for role in ("oit", "fdir")})
    revisions["ppda"] = constants["PPDA_REVISION"]
    if set(revisions) != set(REPOSITORIES):
        raise ValueError("Acquired stream requires its complete eight-provider pin set")
    return revisions, constants["VENDOR_PATH"], constants["VENDOR_REVISION"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing exact role-named provider checkouts")
    for role in REPOSITORIES:
        parser.add_argument("--" + role + "-repo", type=Path, help="Existing exact checkout; overrides --stack-root")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ciw-acquired-stream-fixtures"),
                        help="Retain native source, original, replay and refusal fixtures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revisions, vendor_path, vendor_revision = pins(root)
    with tempfile.TemporaryDirectory(prefix="ciw-acquired-stream-gate-") as directory:
        temporary = Path(directory)
        providers = temporary / "providers"
        providers.mkdir()
        for role, repository in REPOSITORIES.items():
            explicit = getattr(args, role + "_repo")
            existing = explicit or (args.stack_root / role if args.stack_root else None)
            destination = providers / role
            if existing is not None:
                existing = existing.resolve(strict=True)
                exact_revision(existing, revisions[role])
                destination.symlink_to(existing, target_is_directory=True)
            else:
                call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/giasonpooni/" + repository + ".git", str(destination)])
                call(["git", "-C", str(destination), "checkout", "--quiet", "--detach", revisions[role]])
                if role == "ppda":
                    call(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive", "--", vendor_path])
                exact_revision(destination, revisions[role])
            if role == "ppda":
                exact_revision(destination / vendor_path, vendor_revision)

        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for filename in TEST_FILES:
            shutil.copyfile(root / "tests" / filename, work / "tests" / filename)
        shutil.copyfile(root / "tests" / "native_operations.py", work / "tests" / "native_operations.py")
        shutil.copytree(root / "examples", work / "examples")
        env = {**os.environ, "CIW_ACQUIRED_STREAM_STACK_ROOT": str(providers),
               "CIW_ACQUIRED_STREAM_FIXTURE_DIR": str(args.output_dir.resolve())}
        env.pop("PYTHONPATH", None)
        location = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=env, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Gate must import its isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(python), "-I", "-m", "pytest", "-q", "tests", "--junitxml", str(report)], cwd=work, env=env, timeout=1800)
        if any(int(s.attrib.get("skipped", 0)) for s in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Acquired stream gate cannot skip native provider tests")
        for role, revision in revisions.items():
            exact_revision(providers / role, revision)
        exact_revision(providers / "ppda" / vendor_path, vendor_revision)
    print("PASS: installed acquisition lineage, calibrated windows, retained residual monitoring and replay")


if __name__ == "__main__":
    main()
