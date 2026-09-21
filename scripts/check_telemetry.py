"""Exercise only reviewed manifest pins; no branch-following provider installs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORIES = {
    "ppda": "Provenance-Preserving-Data-Acquisition",
    "stfe": "Streaming-Telemetry-Feature-Extraction",
    "gsie": "Geometric-State-Inference-Engine",
    "set": "State-Estimation-Evaluation-Testbed",
    "cbsr": "Constraint-Based-State-Reconciliation",
}


def main():
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "src/ciw/telemetry-runtimes.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ciw-telemetry-providers-") as directory:
        for role, repository in REPOSITORIES.items():
            path = Path(directory) / repository
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            "https://github.com/giasonpooni/" + repository + ".git", str(path)],
                           check=True, timeout=120)
            subprocess.run(["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout",
                            "--detach", pins[role]["revision"]], check=True, timeout=120)
        environment = {**os.environ, "CIW_TELEMETRY_STACK_ROOT": directory,
                       "PYTHONPATH": str(root / "src")}
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/test_telemetry.py"],
                               cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
