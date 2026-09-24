"""Require real SP1 proving, fresh replay and verification from an installed CIW wheel.

Providers and executables are explicit host bindings. This gate does not clone,
repair, rebind or build them; scripts/provision_proved_heat.py provisions their
exact sources and verifies the committed guest build recipe first.
"""
from __future__ import annotations

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

from provider_checkouts import descriptor_pin, extra_pin, validate_checkout

# SCR is the proved-heat pipeline's pinned provider; SP1 is the guest recipe's
# pinned source, declared in ci/gates.json.
_SCR, _SP1 = descriptor_pin("proved-heat", "scr"), extra_pin("sp1")
SCR_REVISION, SCR_TREE = _SCR["revision"], _SCR["source_tree"]
SP1_REVISION, SP1_TREE = _SP1["revision"], _SP1["source_tree"]
GUEST_SHA256 = "a14e3750da7e221d31842bd6cf983fcc8c0f530b2811537e2a9a9fe803dacf82"
TESTS = ("test_proved_heat.py", "test_proved_heat_session.py")
REQUIRED_NATIVE_TESTS = {
    "test_native_proved_heat_shared_session",
    "test_native_tampered_proof_fails_fresh_verification",
    "test_native_retained_proof_can_be_reverified_without_reexecution",
}


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 300), **kwargs)


def source_identity(path, revision, expected_tree):
    path = validate_checkout(path, revision)
    tree = call(["git", "--no-replace-objects", "-C", str(path), "rev-parse", "HEAD^{tree}"],
                capture_output=True, text=True).stdout.strip()
    if tree != expected_tree:
        raise ValueError("Provider source tree differs from its approved pin")
    return path


def file_identity(path, *, executable=False):
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"Required native artifact is unavailable or not executable: {path}")
    with path.open("rb") as stream:
        checksum = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return path, {"sha256": checksum.hexdigest(), "byte_count": path.stat().st_size}


def check_tests(path):
    """A missing, skipped, failed or renamed native gate cannot silently pass."""
    report = ET.parse(path).getroot()
    cases = list(report.iter("testcase"))
    if not cases or any(case.find(tag) is not None for case in cases
                        for tag in ("skipped", "failure", "error")):
        raise AssertionError("Real SP1 gate requires completed tests with no skips, failures or errors")
    if not REQUIRED_NATIVE_TESTS.issubset({case.get("name") for case in cases}):
        raise AssertionError("Real SP1 proof/replay integration test did not run")
    return len(cases)


def retained_measurements(directory):
    """Project recorded measurements without executing or estimating anything."""
    result = {}
    for name in ("original", "replay"):
        bundle = json.loads((directory / (name + ".json")).read_bytes())
        data = bundle["steps"][0]["result"]["data"]
        timings = data["timings"]
        result[name] = {
            "proof_byte_count": data["proof"]["byte_count"],
            **{key: timings[key] for key in ("native_seconds", "prove_and_verify_seconds", "reverify_seconds")},
            "memory": timings["memory"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scr-repo", required=True, type=Path, help="Clean SCR checkout at the exact approved pin")
    parser.add_argument("--sp1-repo", required=True, type=Path, help="Clean SP1 checkout at the recipe's exact pin")
    parser.add_argument("--engine", required=True, type=Path, help="Trusted execution-cli built from the SCR crates workspace")
    parser.add_argument("--prover", required=True, type=Path, help="Trusted sp1-host built from the SCR zk workspace")
    parser.add_argument("--guest", required=True, type=Path, help="Registered heat ELF, rebuilt with the committed recipe")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty directory for proof artifacts and gate report")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be new or empty; stale proof artifacts cannot satisfy the gate")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema": "ciw.proved-heat-gate.v1", "status": "failed",
              "execution": "not_completed", "physical_truth": "not_established"}
    try:
        scr = source_identity(args.scr_repo, SCR_REVISION, SCR_TREE)
        sp1 = source_identity(args.sp1_repo, SP1_REVISION, SP1_TREE)
        bindings, identities = {}, {}
        for role, value in (("engine", args.engine), ("prover", args.prover), ("guest", args.guest)):
            path, identity = file_identity(value, executable=role != "guest")
            if path.is_relative_to(scr) or path.is_relative_to(sp1):
                raise ValueError("Native artifacts must be outside the immutable provider checkouts")
            bindings[role], identities[role] = path, identity
        if identities["guest"]["sha256"] != GUEST_SHA256:
            raise ValueError("Guest ELF does not match the committed heat registry")
        report.update({"scr": {"revision": SCR_REVISION, "tree": SCR_TREE},
                       "sp1": {"revision": SP1_REVISION, "tree": SP1_TREE}, "native_artifacts": identities,
                       "host_executable_binding": "operator_asserted_not_attested"})
        with tempfile.TemporaryDirectory(prefix="ciw-proved-heat-gate-") as directory:
            temporary = Path(directory)
            build = temporary / "build"
            shutil.copytree(root / "src", build / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                if (root / name).is_file():
                    shutil.copyfile(root / name, build / name)
            wheels = temporary / "wheels"
            call([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", str(build), "-w", str(wheels)])
            wheel, = wheels.glob("*.whl")
            environment = temporary / "environment"
            call([sys.executable, "-m", "venv", str(environment)])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            call([str(python), "-m", "pip", "install", str(wheel), "pytest==9.0.2"])
            work = temporary / "installed-check"
            (work / "tests").mkdir(parents=True)
            for name in TESTS:
                shutil.copyfile(root / "tests" / name, work / "tests" / name)
            shutil.copytree(root / "examples/proved-heat", work / "examples/proved-heat")
            env = {**os.environ, "CIW_SCR_PROOF_REPO": str(scr),
                   "CIW_SCR_PROOF_ENGINE": str(bindings["engine"]), "CIW_SP1_PROVER": str(bindings["prover"]),
                   "CIW_SP1_HEAT_GUEST": str(bindings["guest"]),
                   "CIW_PROVED_HEAT_FIXTURE_DIR": str(destination), "PYTHONDONTWRITEBYTECODE": "1"}
            env.pop("PYTHONPATH", None)
            env.pop("PYTEST_ADDOPTS", None)
            imported = call([str(python), "-I", "-c", "import ciw; print(ciw.__file__)"],
                            cwd=work, env=env, capture_output=True, text=True)
            if not Path(imported.stdout.strip()).resolve().is_relative_to(environment.resolve()):
                raise AssertionError("Gate must execute the isolated installed wheel")
            xml = destination / "tests.xml"
            call([str(python), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml),
                  *["tests/" + name for name in TESTS]], cwd=work, env=env, timeout=7400)
            report["tests_passed"] = check_tests(xml)
            report["installed_wheel"] = {"filename": wheel.name, **file_identity(wheel)[1]}
        source_identity(scr, SCR_REVISION, SCR_TREE)
        source_identity(sp1, SP1_REVISION, SP1_TREE)
        for role, path in bindings.items():
            if file_identity(path, executable=role != "guest")[1] != identities[role]:
                raise AssertionError("Native artifact changed during the gate")
        report["measurements"] = retained_measurements(destination)
        report.update(status="passed", execution="real_sp1_proof_fresh_replay_and_verification")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (destination / "gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: installed workbench, real SP1 proof, fresh replay, verification and tamper rejection")
    for name, measurements in report["measurements"].items():
        print(f"MEASUREMENTS {name}: " + json.dumps(measurements, sort_keys=True))


if __name__ == "__main__":
    main()
