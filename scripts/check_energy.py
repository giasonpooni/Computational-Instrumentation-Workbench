"""Verify the installed energy bench; physical GPU checks require --hardware."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def run(command, **kwargs):
    return subprocess.run(command, check=True, timeout=600, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware", action="store_true", help="Require actual CUDA/NVML tests; never skip failures")
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ciw-energy-", dir=args.temporary_root) as directory:
        temp = Path(directory).resolve()
        build = temp / "build"
        shutil.copytree(root / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            if (root / name).is_file():
                shutil.copyfile(root / name, build / name)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build), "-w", str(temp / "wheels")])
        wheel, = (temp / "wheels").glob("*.whl")
        run([sys.executable, "-m", "venv", str(temp / "env")])
        python = temp / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temp / "installed-check"
        (work / "tests").mkdir(parents=True)
        tests = sorted((root / "tests").glob("test_energy_*.py"))
        if len(tests) < 5:
            raise AssertionError("Incomplete energy test suite")
        for path in tests:
            shutil.copyfile(path, work / "tests" / path.name)
        shutil.copytree(root / "examples/energy-accuracy", work / "examples/energy-accuracy")
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "CIW_TEST_CUDA", "CIW_ENERGY_GPU"):
            env.pop(name, None)
        if args.hardware:
            env.update(CIW_TEST_CUDA="1", CIW_ENERGY_GPU="1")
        imported = run([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=env, capture_output=True, text=True)
        if not Path(imported.stdout.strip()).resolve().is_relative_to((temp / "env").resolve()):
            raise AssertionError("Tests must import the installed wheel")
        report = destination / "tests.xml"
        run([str(python), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--basetemp", str(temp / "pytest"), "--junitxml", str(report), *["tests/" + p.name for p in tests]], cwd=work, env=env)
        cases = list(ET.parse(report).getroot().iter("testcase"))
        skips = [c for c in cases if c.find("skipped") is not None]
        if not cases or any(c.find(t) is not None for c in cases for t in ("error", "failure")):
            raise AssertionError("Energy checks failed")
        if args.hardware and skips:
            raise AssertionError("The physical hardware gate permits no skipped tests")
        if any("actual_" not in c.get("name", "") for c in skips):
            raise AssertionError("Only explicitly optional actual-hardware tests may skip on CPU CI")
        (destination / "gate.json").write_text(json.dumps({"schema": "ciw.energy-installed-gate.v1", "status": "passed",
            "tests_passed": len(cases) - len(skips), "hardware_tests_skipped": len(skips), "hardware_required": args.hardware,
            "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
            "scope": "hardware_contract_and_offline_analysis;synthetic_CI_is_not_physical_measurement"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
