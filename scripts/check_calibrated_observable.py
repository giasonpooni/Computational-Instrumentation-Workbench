"""Clone the reviewed eight-provider pins and exercise the process experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORIES = {
    "fsrt": "Fluid-State-Reconstruction-Testbed",
    "tbrt": "Time-Base-Reconciliation-Runtime",
    "mcur": "Metrological-Calibration-Uncertainty-Runtime",
    "oit": "Observability-Identifiability-Testbed",
    "gsie": "Geometric-State-Inference-Engine",
    "cbsr": "Constraint-Based-State-Reconciliation",
    "fdir": "Fault-Detection-Isolation-Runtime",
    "set": "State-Estimation-Evaluation-Testbed",
}


def main():
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "src/ciw/calibrated-observable-runtimes.json").read_text())
    if set(pins) != set(REPOSITORIES):
        raise ValueError("Calibrated runtime manifest must bind all eight providers")
    with tempfile.TemporaryDirectory(prefix="ciw-calibrated-providers-") as directory:
        for role, repository in REPOSITORIES.items():
            path = Path(directory) / role
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            "https://github.com/giasonpooni/" + repository + ".git", str(path)],
                           check=True, timeout=120)
            subprocess.run(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout",
                            "--detach", pins[role]["revision"]], check=True, timeout=120)
        environment = {**os.environ, "CIW_CALIBRATED_STACK_ROOT": directory,
                       "PYTHONPATH": str(root / "src")}
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_calibrated_observable.py"],
                               cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
