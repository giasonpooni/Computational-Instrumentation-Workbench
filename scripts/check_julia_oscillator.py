#!/usr/bin/env python3
"""Require the real pinned Julia oscillator gate from an installed CIW wheel.

The Julia executable is an explicit host binding. This gate does not download
or install Julia or its packages; the accompanying workflow provisions Julia
1.10.12 and instantiates the packaged environment first. A skipped, failed or
renamed native test cannot pass the gate, and no measurement is reused across
runs: every number in the report comes from this invocation.
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

TESTS = ("test_julia_oscillator.py", "test_julia_oscillator_session.py", "fake_julia_worker.py")
REQUIRED_NATIVE_TESTS = {
    "test_native_runtime_identity_is_the_pinned_environment",
    "test_native_default_fixture_matches_the_analytic_reference_at_every_sample",
    "test_native_mixed_initial_state_keeps_state_order_and_signs",
    "test_native_undamped_limit_conserves_energy_within_tolerance",
    "test_native_tolerance_profiles_record_error_and_work_without_proportionality_claims",
    "test_native_worker_isolation_a_b_a_and_restart",
    "test_native_refusals_before_and_at_the_worker",
    "test_native_timeout_and_crash_end_the_session_and_a_fresh_one_serves",
    "test_native_wrong_environment_is_refused_explicitly",
    "test_native_offline_restore_then_explicit_replay",
    "test_native_recording_projection_drives_the_legacy_viewport_session",
    "test_native_terminal_run_inspect_replay_recording",
}
FIXTURES = ("default", "mixed-initial-state", "undamped", "loose-tolerance", "tight-tolerance", "stepped-output")


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout", 600), **kwargs)


def file_identity(path, *, executable=False):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"Required artifact is unavailable or not executable: {path}")
    with path.open("rb") as stream:
        checksum = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return path, {"sha256": checksum.hexdigest(), "byte_count": path.stat().st_size}


def check_tests(path):
    report = ET.parse(path).getroot()
    cases = list(report.iter("testcase"))
    if not cases or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise AssertionError("Julia gate requires completed tests with no skips, failures or errors")
    if not REQUIRED_NATIVE_TESTS.issubset({case.get("name") for case in cases}):
        raise AssertionError("A required real Julia integration test did not run")
    return len(cases)


def retained_measurements(directory):
    """Project measured oracle errors and solver work without recomputing them."""
    result = {}
    for name in FIXTURES:
        bundle = json.loads((directory / (name + ".json")).read_bytes())
        decoded = bundle["steps"][0]["result"]["data"]["native"]["decoded"]
        measured = bundle["verification"]["measured"]
        result[name] = {"oracle_outcome": bundle["verification"]["outcome"],
                        "normalized_max_error": {k: measured[k]["normalized_max_error"] for k in ("q", "v", "energy")},
                        "max_abs_error": {k: measured[k]["max_abs_error"] for k in ("q", "v", "energy")},
                        "energy_conservation": measured["numerical_energy_conservation"],
                        "solver": decoded["solver"], "elapsed_ns": bundle["steps"][0]["result"]["data"]["occurrence"]["worker_metadata"]["elapsed_ns"]}
    replay = json.loads((directory / "default-replay.json").read_bytes())["replay_receipts"][0]["verification"]
    result["default-replay"] = {"byte_identical": replay["byte_identical"], "agreement": replay["agreement"]}
    result["runtime"] = {k: json.loads((directory / "default.json").read_bytes())["runtimes"]["julia"][k]
                         for k in ("julia_version", "platform", "packages", "solver", "sysimage")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", required=True, type=Path, help="Julia 1.10.12 executable with the packaged environment instantiated")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty directory for retained fixtures and the gate report")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be new or empty; stale fixtures cannot satisfy the gate")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"schema": "ciw.julia-oscillator-gate.v1", "status": "failed", "execution": "not_completed",
              "physical_truth": "not_established", "origin": "simulation"}
    try:
        julia, identity = file_identity(args.julia, executable=True)
        report["julia"] = {"path": str(julia), **identity}
        with tempfile.TemporaryDirectory(prefix="ciw-julia-gate-") as directory:
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
            shutil.copytree(root / "examples/julia-oscillator", work / "examples/julia-oscillator")
            shutil.copyfile(root / "scripts/pin_julia_runtime.py", work / "scripts_pin_julia_runtime.py")
            (work / "scripts").mkdir()
            shutil.copyfile(root / "scripts/pin_julia_runtime.py", work / "scripts/pin_julia_runtime.py")
            env = {**os.environ, "CIW_JULIA_EXECUTABLE": str(julia), "CIW_JULIA_FIXTURE_DIR": str(destination / "fixtures"),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            env.pop("PYTHONPATH", None)
            env.pop("PYTEST_ADDOPTS", None)
            imported = call([str(python), "-I", "-c", "import ciw, ciw.julia_worker; print(ciw.__file__); print(ciw.julia_worker.environment_root())"],
                            cwd=work, env=env, capture_output=True, text=True)
            location, packaged = imported.stdout.strip().splitlines()
            if not Path(location).resolve().is_relative_to(environment.resolve()) or not Path(packaged).resolve().is_relative_to(environment.resolve()):
                raise AssertionError("Gate must execute the isolated installed wheel and its packaged Julia environment")
            xml = destination / "tests.xml"
            call([str(python), "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml),
                  *["tests/" + name for name in TESTS if name.startswith("test_")]], cwd=work, env=env, timeout=3600)
            report["tests_passed"] = check_tests(xml)
            report["installed_wheel"] = {"filename": wheel.name, **file_identity(wheel)[1]}
        if file_identity(julia, executable=True)[1] != identity:
            raise AssertionError("Julia executable changed during the gate")
        report["measurements"] = retained_measurements(destination / "fixtures")
        report.update(status="passed", execution="real_julia_worker_oracle_comparison_fresh_replay_and_refusals")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (destination / "gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: installed workbench, real Julia integration, oracle comparison, worker isolation, refusals and fresh replay")
    for name, measurements in report["measurements"].items():
        print(f"MEASUREMENTS {name}: " + json.dumps(measurements, sort_keys=True))


if __name__ == "__main__":
    main()
