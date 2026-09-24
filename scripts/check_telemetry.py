"""Exercise only reviewed manifest pins; no branch-following provider installs."""
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
    "ppda": "Provenance-Preserving-Data-Acquisition",
    "stfe": "Streaming-Telemetry-Feature-Extraction",
    "gsie": "Geometric-State-Inference-Engine",
    "set": "State-Estimation-Evaluation-Testbed",
    "cbsr": "Constraint-Based-State-Reconciliation",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path,
                        help="Existing exact checkouts, using full repository directory names")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    pins = descriptor_pins("telemetry", root)
    with tempfile.TemporaryDirectory(prefix="ciw-telemetry-providers-") as directory:
        stack_root = args.stack_root.resolve() if args.stack_root else Path(directory)
        for role, repository in REPOSITORIES.items():
            path = stack_root / repository
            if not args.stack_root:
                subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", "--filter=blob:none",
                                "https://github.com/giasonpooni/" + repository + ".git", str(path)],
                               check=True, timeout=120)
                subprocess.run(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout",
                                "--detach", pins[role]["revision"]], check=True, timeout=120)
            validate_checkout(path, pins[role]["revision"])
        environment = {**os.environ, "CIW_TELEMETRY_STACK_ROOT": str(stack_root),
                       "PYTHONPATH": str(root / "src")}
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_telemetry.py"],
                               cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
