"""Clone declared source pins and exercise retained scientific investigations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if __package__:
    from .provider_checkouts import validate_checkout
else:
    from provider_checkouts import validate_checkout


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing exact current checkouts, named by manifest role")
    parser.add_argument("--historical-stack-root", type=Path,
                        help="Existing historical checkouts, named by role; otherwise use <role>-legacy under --stack-root")
    args = parser.parse_args(argv)
    if args.historical_stack_root and not args.stack_root:
        parser.error("--historical-stack-root requires --stack-root")
    # Terminal adapter pins are defined by their provider descriptors.
    pins = {}
    for path in sorted((ROOT / "src/ciw/pipelines/providers").glob("*.json")):
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        if descriptor["invocation"] == "pinned_subprocess" and descriptor["surface"] == "terminal":
            pins[descriptor["role"]] = descriptor["pin"]
    with tempfile.TemporaryDirectory(prefix="ciw-domain-pins-") as directory:
        environment = dict(os.environ)
        for name, spec in pins.items():
            target = (args.stack_root if args.stack_root else Path(directory)) / name
            if not args.stack_root:
                subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout",
                                spec["repository"] + ".git", str(target)], check=True)
                subprocess.run(["git", "-C", str(target), "config", "core.autocrlf", "false"], check=True)
                subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["revision"]], check=True)
            target = validate_checkout(target, spec["revision"])
            environment["CIW_" + name.upper() + "_REPO"] = str(target)
            if spec.get("historical"):
                previous = spec["historical"][0]
                historic = (args.historical_stack_root / name if args.historical_stack_root else
                            (args.stack_root if args.stack_root else Path(directory)) / (name + "-legacy"))
                if not args.stack_root:
                    subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--no-checkout", str(target), str(historic)], check=True)
                    subprocess.run(["git", "-C", str(historic), "config", "core.autocrlf", "false"], check=True)
                    subprocess.run(["git", "-C", str(historic), "checkout", "--detach", previous["revision"]], check=True)
                historic = validate_checkout(historic, previous["revision"])
                environment["CIW_" + name.upper() + "_LEGACY_REPO"] = str(historic)
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_investigation.py",
                        "tests/test_adapter_cli.py", "tests/test_covariance_integration.py", "tests/test_geodesic.py"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
