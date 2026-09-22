"""Exercise installed CIW against exact native companion, CSE and PPDA revisions."""
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
    "jspt": ("Jacobian-Sensitivity-Propagation-Testbed", "7399ab03087b27683620b4c57f97b2ac14546c7f"),
    "plsr": ("Parameterized-Lyapunov-Stability-Runtime", "9d0e7b4a1162e038150a71c63d986945d78135d4"),
    "cse": ("Construction-State-Estimator-for-BIM", "4b74abda40bba3277de69bf61e9e09283ae2d5b3"),
    "ppda": ("Provenance-Preserving-Data-Acquisition", "477d6cb454423a27543b16961d3b169709c40c31"),
}
VENDOR_PATH = "vendor/scout-retrieval-agent"
VENDOR_REVISION = "5e146d5924675cd7b6e1d1ed44fb39f5da012610"
TEST_FILES = (
    "test_integrated_modules_gate.py",
    "test_schematic_companions.py",
    "test_bim_quantity.py",
    "test_acquired_dataset.py",
    "test_spatial_view.py",
    "test_spatial_transport.py",
)


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def exact_revision(repository, revision):
    actual = call(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"Require exact provider revision {revision}: {repository}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", type=Path, help="Existing exact checkouts named sra/jspt/plsr/cse/ppda")
    for role in PROVIDERS:
        parser.add_argument("--" + role + "-repo", type=Path, help=f"Existing exact {role.upper()} checkout; overrides --stack-root")
    parser.add_argument("--output-dir", type=Path, help="Retain fresh original/replay, held/refused and geographic-view fixtures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="ciw-integrated-gate-") as directory:
        temporary = Path(directory)
        providers = temporary / "providers"
        providers.mkdir()
        for role, (repository, revision) in PROVIDERS.items():
            explicit = getattr(args, role + "_repo")
            existing = explicit or (args.stack_root / role if args.stack_root else None)
            destination = providers / role
            if existing is not None:
                existing = existing.resolve(strict=True)
                exact_revision(existing, revision)
                destination.symlink_to(existing, target_is_directory=True)
            else:
                call(["git", "clone", "--quiet", "--no-checkout", "https://github.com/giasonpooni/" + repository + ".git", str(destination)])
                call(["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision])
                if role == "ppda":
                    call(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive", "--", VENDOR_PATH])
                exact_revision(destination, revision)
            if role == "ppda":
                exact_revision(destination / VENDOR_PATH, VENDOR_REVISION)

        wheels = temporary / "wheels"
        call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(root), "--wheel-dir", str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary / "environment"
        call([sys.executable, "-m", "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
        work = temporary / "installed-check"
        (work / "tests").mkdir(parents=True)
        for filename in TEST_FILES:
            shutil.copyfile(root / "tests" / filename, work / "tests" / filename)
        shutil.copytree(root / "examples", work / "examples")
        env = {**os.environ, "CIW_INTEGRATED_STACK_ROOT": str(providers),
               "CIW_SCHEMATIC_STACK_ROOT": str(providers), "CIW_CSE_REPO": str(providers / "cse"),
               "CIW_PPDA_REPO": str(providers / "ppda")}
        env.pop("PYTHONPATH", None)
        if args.output_dir:
            env["CIW_INTEGRATED_FIXTURE_DIR"] = str(args.output_dir.resolve())
        location = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"], cwd=work, env=env, capture_output=True, text=True)
        if not Path(location.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Gate must import its isolated installed wheel")
        report = temporary / "tests.xml"
        call([str(python), "-I", "-m", "pytest", "-q", "tests", "--junitxml", str(report)], cwd=work, env=env, timeout=900)
        if any(int(s.attrib.get("skipped", 0)) for s in ET.parse(report).getroot().iter("testsuite")):
            raise AssertionError("Integrated native gate cannot skip provider tests")
        # Adapters also verify the full source trees before and after each call.
        for role, (_, revision) in PROVIDERS.items():
            exact_revision(providers / role, revision)
        exact_revision(providers / "ppda" / VENDOR_PATH, VENDOR_REVISION)
    print("PASS: installed companion execution, BIM conditioning, acquisition, geographic view and replay")


if __name__ == "__main__":
    main()
