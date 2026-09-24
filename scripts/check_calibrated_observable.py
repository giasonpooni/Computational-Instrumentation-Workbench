"""Clone the reviewed eight-provider pins and exercise the process experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from .provider_checkouts import descriptor_pins, validate_checkout
else:
    from provider_checkouts import descriptor_pins, validate_checkout

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path,
                        help="Existing exact checkouts, named by manifest role")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    pins = descriptor_pins("calibrated-observable", root)
    if set(pins) != set(REPOSITORIES):
        raise ValueError("Calibrated runtime manifest must bind all eight providers")
    with tempfile.TemporaryDirectory(prefix="ciw-calibrated-providers-") as directory:
        stack_root = args.stack_root.resolve() if args.stack_root else Path(directory)
        for role, repository in REPOSITORIES.items():
            path = stack_root / role
            if not args.stack_root:
                subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", "--filter=blob:none",
                                "https://github.com/giasonpooni/" + repository + ".git", str(path)],
                               check=True, timeout=120)
                subprocess.run(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout",
                                "--detach", pins[role]["revision"]], check=True, timeout=120)
            validate_checkout(path, pins[role]["revision"])
        environment = {**os.environ, "CIW_CALIBRATED_STACK_ROOT": str(stack_root),
                       "PYTHONPATH": str(root / "src")}
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_calibrated_observable.py"],
                               cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
