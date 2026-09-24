"""Exercise all identified-design pins from an installed wheel outside the checkout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provider_checkouts import descriptor_pins  # noqa: E402


REPOSITORIES = {
    "fsrt": "Fluid-State-Reconstruction-Testbed",
    "tbrt": "Time-Base-Reconciliation-Runtime",
    "mcur": "Metrological-Calibration-Uncertainty-Runtime",
    "oit": "Observability-Identifiability-Testbed",
    "gsie": "Geometric-State-Inference-Engine",
    "cbsr": "Constraint-Based-State-Reconciliation",
    "fdir": "Fault-Detection-Isolation-Runtime",
    "set": "State-Estimation-Evaluation-Testbed",
    "sidt": "System-Identification-Dynamics-Testbed",
    "edspt": "Experiment-Design-Sensor-Placement-Testbed",
    "ywir": "Yield-Weighted-Inference-Runtime",
}


def call(arguments, **kwargs):
    return subprocess.run(arguments, check=True, timeout=kwargs.pop("timeout", 180), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path,
                        help="Use existing clean role-named checkouts at the exact manifest pins")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("The YWIR provider requires Python 3.12 or newer")
    root = Path(__file__).resolve().parents[1]
    inherited = descriptor_pins("calibrated-observable", root)
    pins = descriptor_pins("identified-design", root)
    if any(pins.get(role) != pin for role, pin in inherited.items()):
        raise ValueError("Identified-design pins must extend, never replace, calibrated provider pins")
    if set(pins) != set(REPOSITORIES):
        raise ValueError("Identified-design manifest must bind all eleven providers")
    with tempfile.TemporaryDirectory(prefix="ciw-identified-design-") as directory:
        temporary = Path(directory)
        providers = args.stack_root.resolve() if args.stack_root else temporary / "providers"
        if args.stack_root is None:
            providers.mkdir()
            for role, repository in REPOSITORIES.items():
                path = providers / role
                call(["git", "clone", "--no-checkout", "--filter=blob:none",
                      "https://github.com/giasonpooni/" + repository + ".git", str(path)])
                call(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout",
                      "--detach", pins[role]["revision"]])
        for role, pin in pins.items():
            path = providers / role
            revision = call(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
            if revision != pin["revision"]:
                raise ValueError(f"{role}: checkout is not at the exact source pin")
            dirty = call(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True).stdout.strip()
            if dirty:
                raise ValueError(f"{role}: provider checkout must be clean")

        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
              str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment_path = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment_path)])
        interpreter = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(interpreter), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])

        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for name in ("test_identified_design.py", "test_workbench_session.py", "test_workbench_transport.py"):
            shutil.copyfile(root / "tests" / name, work / "tests" / name)
        shutil.copytree(root / "examples", work / "examples")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["CIW_IDENTIFIED_DESIGN_STACK_ROOT"] = str(providers)
        environment["CIW_CALIBRATED_STACK_ROOT"] = str(providers)
        location = call([str(interpreter), "-I", "-c", "import ciw; print(ciw.__file__)"],
                        cwd=work, env=environment, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment_path.resolve()):
            raise AssertionError("Integration must import the isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(interpreter), "-I", "-m", "pytest", "-q", "tests/test_identified_design.py",
              "tests/test_workbench_session.py",
              "tests/test_workbench_transport.py",
              "--junitxml", str(report)], cwd=work, env=environment, timeout=1800)
        suites = ET.parse(report).getroot().iter("testsuite")
        if any(int(suite.attrib.get("skipped", 0)) for suite in suites):
            raise AssertionError("Pinned installed integration cannot pass with skipped tests")
    print("PASS: installed wheel, shared workbench, eleven source pins, advisory ranking and fresh replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
