"""Installed-wheel CIW → pinned ESM → calibrated replay, with no skipped tests."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def call(arguments, **kwargs):
    return subprocess.run(arguments, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", type=Path, help="Existing built ESM; artifact/helper hashes remain enforced")
    parser.add_argument("--fixture-root", type=Path, help="Prepared bundle.json/runtime.json; fresh replay remains mandatory")
    args = parser.parse_args()
    if bool(args.esm_root) != bool(args.fixture_root):
        parser.error("Existing fixture and ESM roots must be supplied together")
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / "src/ciw/esm-runtime.json").read_text())
    with tempfile.TemporaryDirectory(prefix="ciw-candidate-gate-") as directory:
        temporary = Path(directory)
        if args.esm_root:
            esm, fixture = args.esm_root.resolve(), args.fixture_root.resolve()
        else:
            esm = temporary / "esm"
            call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/" + pin["repository"] + ".git", str(esm)])
            call(["git", "-C", str(esm), "checkout", "--quiet", "--detach", pin["revision"]])
            call(["npm", "ci", "--ignore-scripts"], cwd=esm)
            call(["npm", "run", "instrument:workbench:build"], cwd=esm)
            fixture = temporary / "fixture"
            call([sys.executable, "-B", str(esm / "scripts/check_calibrated_workbench.py"), "--output-dir", str(fixture)], timeout=1200)
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment_path = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment_path)])
        interpreter = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(interpreter), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        work.mkdir()
        shutil.copyfile(root / "tests/test_workbench_candidates.py", work / "test_workbench_candidates.py")
        environment = {**os.environ, "CIW_ESM_ROOT": str(esm), "CIW_ESM_BUNDLE_FILE": str(fixture / "bundle.json"),
                       "CIW_ESM_RUNTIME_FILE": str(fixture / "runtime.json")}
        environment.pop("PYTHONPATH", None)
        location = call([str(interpreter), "-I", "-c", "import ciw; print(ciw.__file__)"],
                        cwd=work, env=environment, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment_path.resolve()):
            raise AssertionError("Gate must import the isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(interpreter), "-I", "-m", "pytest", "-q", "test_workbench_candidates.py", "--junitxml", str(report)],
             cwd=work, env=environment, timeout=1200)
        if any(int(suite.attrib.get("skipped", 0)) for suite in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Candidate gate cannot pass with skipped integration tests")
    print("PASS: installed workbench, native state/diagnostics, real ESM replay and candidate-only retention")


if __name__ == "__main__":
    main()
