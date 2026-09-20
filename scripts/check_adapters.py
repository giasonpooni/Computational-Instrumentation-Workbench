"""Clone declared public source pins and exercise the calibrated investigation gate."""
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
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_investigation.py",
                        "tests/test_adapter_cli.py"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
