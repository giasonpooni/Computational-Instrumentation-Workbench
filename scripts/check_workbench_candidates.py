"""Installed CIW → pinned ESM, calibrated and telemetry replay, with no skips."""
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

TELEMETRY_REPOSITORIES = {
    "ppda": "Provenance-Preserving-Data-Acquisition", "stfe": "Streaming-Telemetry-Feature-Extraction",
    "gsie": "Geometric-State-Inference-Engine", "set": "State-Estimation-Evaluation-Testbed",
    "cbsr": "Constraint-Based-State-Reconciliation",
}


def call(arguments, **kwargs):
    # Pinned runtimes check tracked bytes. Inherit this override into helper
    # scripts too, so Windows checkout settings cannot rewrite their sources.
    environment = dict(kwargs.pop("env", os.environ))
    index = int(environment.get("GIT_CONFIG_COUNT", "0"))
    environment.update({"GIT_CONFIG_COUNT": str(index + 1),
                        f"GIT_CONFIG_KEY_{index}": "core.autocrlf",
                        f"GIT_CONFIG_VALUE_{index}": "false"})
    kwargs["env"] = environment
    return subprocess.run(arguments, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-root", type=Path, help="Existing built ESM; artifact/helper hashes remain enforced")
    parser.add_argument("--fixture-root", type=Path, help="Prepared bundle.json/runtime.json; fresh replay remains mandatory")
    parser.add_argument("--telemetry-stack-root", type=Path, help="Existing exact role-named scalar telemetry checkouts")
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
            npm = shutil.which("npm")
            if npm is None:
                raise RuntimeError("npm must be available to build the pinned ESM provider")
            esm = temporary / "esm"
            call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/" + pin["repository"] + ".git", str(esm)])
            call(["git", "-C", str(esm), "checkout", "--quiet", "--detach", pin["revision"]])
            call([npm, "ci", "--ignore-scripts"], cwd=esm)
            call([npm, "run", "instrument:workbench:build"], cwd=esm)
            fixture = temporary / "fixture"
            call([sys.executable, "-B", str(esm / "scripts/check_calibrated_workbench.py"), "--output-dir", str(fixture)], timeout=1200)
        providers = args.telemetry_stack_root.resolve() if args.telemetry_stack_root else temporary / "telemetry-providers"
        if not args.telemetry_stack_root:
            providers.mkdir()
            pins = json.loads((root / "src/ciw/telemetry-runtimes.json").read_text())
            for role, repository in TELEMETRY_REPOSITORIES.items():
                path = providers / role
                call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/giasonpooni/" + repository + ".git", str(path)])
                call(["git", "-C", str(path), "checkout", "--quiet", "--detach", pins[role]["revision"]])
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment_path = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment_path)])
        interpreter = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(interpreter), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for name in ("test_workbench_candidates.py", "test_workbench_telemetry.py"):
            shutil.copyfile(root / "tests" / name, work / "tests" / name)
        shutil.copyfile(root / "tests" / "native_operations.py", work / "tests" / "native_operations.py")
        shutil.copytree(root / "examples", work / "examples")
        replay_runtime = json.loads((fixture / "runtime.json").read_text())
        environment = {**os.environ, "CIW_ESM_ROOT": str(esm), "CIW_ESM_BUNDLE_FILE": str(fixture / "bundle.json"),
                       "CIW_ESM_RUNTIME_FILE": str(fixture / "runtime.json"), "CIW_SHARED_TELEMETRY_STACK_ROOT": str(providers),
                       "CIW_ESM_REPLAY_CIW": replay_runtime["repositories"]["ciw"]["path"]}
        environment.pop("PYTHONPATH", None)
        location = call([str(interpreter), "-I", "-c", "import ciw; print(ciw.__file__)"],
                        cwd=work, env=environment, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment_path.resolve()):
            raise AssertionError("Gate must import the isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(interpreter), "-I", "-m", "pytest", "-q", "tests", "--junitxml", str(report)],
             cwd=work, env=environment, timeout=1200)
        if any(int(suite.attrib.get("skipped", 0)) for suite in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Candidate gate cannot pass with skipped integration tests")
    print("PASS: installed workbench, native acquisition/features/state/diagnostics, both ESM replay lanes and candidate retention")


if __name__ == "__main__":
    main()
