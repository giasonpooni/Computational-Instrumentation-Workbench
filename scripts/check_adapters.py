"""Clone declared source pins and exercise retained scientific investigations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    pins = json.loads((ROOT / "src/ciw/adapter-runtimes.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ciw-domain-pins-") as directory:
        environment = dict(os.environ)
        for name, spec in pins.items():
            target = Path(directory) / name
            subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout",
                            spec["repository"] + ".git", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "config", "core.autocrlf", "false"], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["revision"]], check=True)
            environment["CIW_" + name.upper() + "_REPO"] = str(target)
            if spec.get("historical"):
                previous = spec["historical"][0]
                historic = Path(directory) / (name + "-legacy")
                subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", str(target), str(historic)], check=True)
                subprocess.run(["git", "-C", str(historic), "config", "core.autocrlf", "false"], check=True)
                subprocess.run(["git", "-C", str(historic), "checkout", "--detach", previous["revision"]], check=True)
                environment["CIW_" + name.upper() + "_LEGACY_REPO"] = str(historic)
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_investigation.py",
                        "tests/test_adapter_cli.py", "tests/test_covariance_integration.py", "tests/test_geodesic.py"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
