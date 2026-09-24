"""Install CIW outside its source tree and exercise real SRA and SCR providers."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PROVIDERS = {
    "sra": ("Schematics-Retrieval-Agent", "a6e79585950bb6860e5edce5ebd2cce39ea481f2"),
    "scr": ("Scientific-Computation-Runtime", "a59aba283b0304faeeb3e5d305087e7709e171ca"),
}


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing role-named exact checkouts; pins remain enforced")
    parser.add_argument("--engine", type=Path, help="Existing trusted SCR execution-cli; otherwise build the pinned Rust source")
    parser.add_argument("--output-dir", type=Path, help="Retain fresh native original/replay artifacts for ICRH")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="ciw-declared-gate-") as directory:
        temporary = Path(directory)
        providers = args.stack_root.resolve() if args.stack_root else temporary / "providers"
        if not args.stack_root:
            providers.mkdir()
            for role, (repository, revision) in PROVIDERS.items():
                destination = providers / role
                call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/giasonpooni/" + repository + ".git", str(destination)])
                call(["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision])
        engine = args.engine.resolve() if args.engine else temporary / "rust-build/release/execution-cli"
        if not args.engine:
            call(["cargo", "build", "--release", "--locked", "--offline", "--manifest-path", str(providers / "scr/crates/Cargo.toml"),
                  "--target-dir", str(temporary / "rust-build"), "-p", "execution-cli"])
        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        shutil.copyfile(root / "tests/test_declared_workloads.py", work / "tests/test_declared_workloads.py")
        shutil.copytree(root / "examples/declared-workloads", work / "examples/declared-workloads")
        env = {**os.environ, "CIW_DECLARED_STACK_ROOT": str(providers), "CIW_SCR_ENGINE": str(engine)}
        env.pop("PYTHONPATH", None)
        if args.output_dir: env["CIW_DECLARED_FIXTURE_DIR"] = str(args.output_dir.resolve())
        location = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=env, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Gate must import the isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(python), "-I", "-m", "pytest", "-q", "tests", "--junitxml", str(report)], cwd=work, env=env, timeout=600)
        if any(int(s.attrib.get("skipped", 0)) for s in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Native declared workload gate cannot skip provider tests")
    print("PASS: installed shared SRA/SCR execution, typed views, refusal, restore and fresh replay")


if __name__ == "__main__": main()
