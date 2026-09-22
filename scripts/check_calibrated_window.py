"""Installed-wheel gate for shared calibrated window execution and replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

REPOSITORIES = {"tbrt": "Time-Base-Reconciliation-Runtime", "mcur": "Metrological-Calibration-Uncertainty-Runtime",
                "stfe": "Streaming-Telemetry-Feature-Extraction", "gsie": "Geometric-State-Inference-Engine",
                "set": "State-Estimation-Evaluation-Testbed"}


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing exact role checkouts; pins are still enforced")
    parser.add_argument("--output-dir", type=Path, help="Retain actual original/replay bundles for ICRH")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "src/ciw/calibrated-window-runtimes.json").read_text())
    if set(pins) != set(REPOSITORIES): raise ValueError("Required five-provider pin manifest mismatch")
    with tempfile.TemporaryDirectory(prefix="ciw-window-gate-") as directory:
        temporary = Path(directory)
        providers = args.stack_root.resolve() if args.stack_root else temporary / "providers"
        if not args.stack_root:
            providers.mkdir()
            for role, repository in REPOSITORIES.items():
                path = providers / role
                call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/giasonpooni/" + repository + ".git", str(path)])
                call(["git", "-C", str(path), "checkout", "--quiet", "--detach", pins[role]["revision"]])
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        venv = temporary / "environment"
        call([sys.executable, "-m", "venv", str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        shutil.copyfile(root / "tests/test_calibrated_window.py", work / "tests/test_calibrated_window.py")
        shutil.copytree(root / "examples/calibrated-window", work / "examples/calibrated-window")
        environment = {**os.environ, "CIW_CALIBRATED_WINDOW_STACK_ROOT": str(providers)}
        environment.pop("PYTHONPATH", None)
        if args.output_dir: environment["CIW_WINDOW_FIXTURE_DIR"] = str(args.output_dir.resolve())
        location = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=environment, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(venv.resolve()):
            raise AssertionError("Gate must import the isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(python), "-I", "-m", "pytest", "-q", "tests", "--junitxml", str(report)], cwd=work, env=environment, timeout=1200)
        if any(int(suite.attrib.get("skipped", 0)) for suite in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Calibrated window gate cannot skip real integration cases")
    print("PASS: installed shared calibrated window, full joint covariance, native refusal, restore and fresh replay")


if __name__ == "__main__": main()
