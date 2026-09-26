"""Explicit source-checkout JuMP study with retained bytes and exact checking.

Runtime paths are supplied by the operator, never obtained from saved studies.
This exploratory caller is not a registered SCR provider or proof operation.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import tomllib
import uuid

from . import design_adjustment as design
from . import exact_response as exact
from . import linear_response as linear
from .telemetry import byte_digest, canonical, digest

SCHEMA = "ciw.jump-design-study.v1"
_FILES = ("Project.toml", "Manifest.toml", "worker.jl")
_SOLVER_FIELDS = {"schema", "termination_status", "primal_status", "delta", "objective_value",
                  "solve_seconds", "julia_version", "jump_version", "highs_version"}
AUTHORITY = {"kind": "unregistered_julia_design_study",
             "retained_trust": "historical_producer_report_requires_explicit_rerun",
             "registered_operation_id": None, "execution_id": None, "result_id": None,
             "verification_id": None, "proof_id": None, "sp1_verification": "not_performed",
             "physical_validation": "not_established", "hardware_actuation": "not_performed"}


def _input(problem):
    lines = ['schema = "ciw.jump-scalar-design-input.v1"']
    for name in ("x", "target", "regularization", "lower", "upper"):
        q = problem[name]
        lines.extend([f"[{name}]", f"numerator = {q['numerator']}", f"denominator = {q['denominator']}"])
    return ("\n".join(lines)+"\n").encode("ascii")


def _artifact(raw):
    return {"sha256": byte_digest(raw), "base64": base64.b64encode(raw).decode("ascii")}


def _raw(value, limit):
    exact._keys(value, {"sha256", "base64"})
    if type(value["base64"]) is not str or len(value["base64"]) > 4*((limit+2)//3):
        raise ValueError("Retained Julia artifact exceeds its byte budget")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed retained Julia artifact") from exc
    if len(raw) > limit or canonical(value) != canonical(_artifact(raw)):
        raise ValueError("Retained Julia artifact binding differs")
    return raw


def _solver(raw):
    if not 1 <= len(raw) <= 16_384:
        raise ValueError("Julia response exceeds its byte budget")
    try:
        result = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Malformed Julia TOML response") from exc
    exact._keys(result, _SOLVER_FIELDS)
    if (result["schema"] != "ciw.jump-scalar-design-output.v1"
            or result["termination_status"] != "OPTIMAL" or result["primal_status"] != "FEASIBLE_POINT"):
        raise ValueError("JuMP did not report an optimal feasible numerical solution")
    for name in ("delta", "objective_value", "solve_seconds"):
        v = result[name]
        if type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e12:
            raise ValueError("Invalid numerical solver report")
    if result["solve_seconds"] < 0 or abs(result["delta"]) > 100:
        raise ValueError("Invalid solver timing or candidate")
    for name in ("julia_version", "jump_version", "highs_version"):
        if type(result[name]) is not str or not re.fullmatch(r"\d+\.\d+\.\d+(?:[+.-][A-Za-z0-9.-]+)?", result[name]):
            raise ValueError("Invalid solver version declaration")
    return result


def conversion(problem, solver_delta):
    """Preserve the exact binary64 value and explicitly project to exact bounds."""
    if type(solver_delta) not in (int, float) or not math.isfinite(solver_delta) or abs(solver_delta) > 100:
        raise ValueError("Invalid solver delta")
    original = Fraction.from_float(float(solver_delta))
    lo, hi = design._ratio(problem["lower"]), design._ratio(problem["upper"])
    projected = min(max(original, lo), hi)
    return {"method": "exact_binary64_ratio_then_projection_to_exact_box",
            "binary64_hex": float(solver_delta).hex(), "raw_delta": exact._pair(original),
            "checked_delta": exact._pair(projected), "adjustment": exact._pair(projected-original),
            "claim_target": "projected_exact_candidate_not_raw_solver_output"}


def _runtime(artifacts, solver, executable_digest):
    if type(executable_digest) is not str or not re.fullmatch(r"sha256:[a-f0-9]{64}", executable_digest):
        raise ValueError("Invalid Julia executable digest")
    manifest = tomllib.loads(_raw(artifacts["Manifest.toml"], 96_000).decode("utf-8"))
    try:
        expected = {"julia_version": manifest["julia_version"],
                    "jump_version": manifest["deps"]["JuMP"][0]["version"],
                    "highs_version": manifest["deps"]["HiGHS"][0]["version"]}
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("JuMP manifest is unresolved") from exc
    if any(solver[k] != v for k, v in expected.items()):
        raise ValueError("Julia provider versions differ from the retained manifest")
    return {**expected, "julia_executable_sha256": executable_digest,
            "files": {name: artifacts[name]["sha256"] for name in _FILES},
            "source_to_binary_attestation": "not_established"}


def run_study(problem, julia_executable, *, depot=None, timeout=120):
    problem = design.validate_problem(problem)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("Study timeout must be in (0,600] seconds")
    project = Path(__file__).resolve().parents[2]/"runtimes"/"jump-design"
    artifacts = {name: _artifact((project/name).read_bytes()) for name in _FILES}
    for name in _FILES:
        _raw(artifacts[name], 96_000)
    executable = Path(julia_executable).resolve(strict=True)
    executable_digest = "sha256:"+sha256(executable.read_bytes()).hexdigest()
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "1"
    env["JULIA_LOAD_PATH"] = os.pathsep.join(("@", "@stdlib"))
    env["JULIA_PKG_OFFLINE"] = "true"
    if depot is not None:
        env["JULIA_DEPOT_PATH"] = str(Path(depot).resolve(strict=True))
    raw_input = _input(problem)
    started = time.perf_counter()
    scratch = project.parents[1]/".ciw-jump-runs"
    scratch.mkdir(exist_ok=True)
    try:
        # Execute exactly the source/environment bytes retained above, even if
        # another contributor edits the working checkout during the solve.
        with tempfile.TemporaryDirectory(prefix="study-", dir=scratch) as temp:
            snapshot = Path(temp).resolve()
            if snapshot.parent != scratch.resolve():
                raise ValueError("Julia snapshot escaped its scratch directory")
            for name in _FILES:
                (snapshot/name).write_bytes(_raw(artifacts[name], 96_000))
            completed = subprocess.run([str(executable), "--startup-file=no", "--history-file=no",
                                        "--project="+str(snapshot), str(snapshot/"worker.jl")],
                                       input=raw_input, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Julia design worker timed out; no successful study retained") from exc
    process_seconds = time.perf_counter()-started
    if "sha256:"+sha256(executable.read_bytes()).hexdigest() != executable_digest:
        raise ValueError("Julia executable changed during the study")
    if len(completed.stdout) > 16_384 or len(completed.stderr) > 131_072:
        raise ValueError("Julia worker output exceeds its retained byte budget")
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise ValueError(f"Julia design worker failed ({completed.returncode}): {diagnostic}")
    solver = _solver(completed.stdout)
    runtime = _runtime(artifacts, solver, executable_digest)
    prepared = time.perf_counter()
    converted = conversion(problem, solver["delta"])
    candidate = design.make_candidate(problem, converted["checked_delta"])
    preparation_seconds = time.perf_counter()-prepared
    checked = time.perf_counter()
    report = design.check_candidate(candidate)
    check_seconds = time.perf_counter()-checked
    study = {"schema": SCHEMA, "problem": problem, "problem_digest": digest(problem),
             "producer_invocation_id": str(uuid.uuid4()), "artifacts": artifacts, "runtime": runtime,
             "worker_input": _artifact(raw_input), "worker_stdout": _artifact(completed.stdout),
             "worker_stderr": _artifact(completed.stderr), "solver": solver, "conversion": converted,
             "check": report, "timing": {"process_seconds": process_seconds,
                                         "solver_seconds": solver["solve_seconds"],
                                         "certificate_preparation_seconds": preparation_seconds,
                                         "exact_check_seconds": check_seconds,
                                         "proving_seconds": None, "verification_seconds": None},
             "authority": deepcopy(AUTHORITY)}
    study["study_id"] = digest(study)
    return inspect_study(study)


def inspect_study(study):
    """Read retained data only: no Julia invocation, solver or exact recheck."""
    linear._bounded(study)
    exact._keys(study, {"schema", "problem", "problem_digest", "producer_invocation_id", "artifacts",
                       "runtime", "worker_input", "worker_stdout", "worker_stderr", "solver", "conversion",
                       "check", "timing", "authority", "study_id"})
    if study["schema"] != SCHEMA or study["study_id"] != digest({k: v for k, v in study.items() if k != "study_id"}):
        raise ValueError("Julia study content identity differs")
    problem = design.validate_problem(study["problem"])
    if study["problem_digest"] != digest(problem) or _raw(study["worker_input"], 8192) != _input(problem):
        raise ValueError("Julia study source binding differs")
    if type(study["producer_invocation_id"]) is not str:
        raise ValueError("Invalid producer invocation identity")
    try:
        if str(uuid.UUID(study["producer_invocation_id"])) != study["producer_invocation_id"]:
            raise ValueError("Invalid producer invocation identity")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid producer invocation identity") from exc
    exact._keys(study["artifacts"], _FILES)
    for name in _FILES:
        _raw(study["artifacts"][name], 96_000)
    solver = _solver(_raw(study["worker_stdout"], 16_384))
    _raw(study["worker_stderr"], 131_072)
    exact._keys(study["runtime"], {"julia_version", "jump_version", "highs_version", "julia_executable_sha256",
                                 "files", "source_to_binary_attestation"})
    runtime = _runtime(study["artifacts"], solver, study["runtime"]["julia_executable_sha256"])
    converted = conversion(problem, solver["delta"])
    report = design.inspect_check(study["check"])
    if (canonical(study["solver"]) != canonical(solver) or canonical(study["runtime"]) != canonical(runtime)
            or canonical(study["conversion"]) != canonical(converted)
            or canonical(report["candidate"]["problem"]) != canonical(problem)
            or canonical(report["candidate"]["delta"]) != canonical(converted["checked_delta"])
            or canonical(study["authority"]) != canonical(AUTHORITY)):
        raise ValueError("Julia study candidate, runtime or authority binding differs")
    timing = study["timing"]
    exact._keys(timing, {"process_seconds", "solver_seconds", "certificate_preparation_seconds",
                         "exact_check_seconds", "proving_seconds", "verification_seconds"})
    for name in ("process_seconds", "solver_seconds", "certificate_preparation_seconds", "exact_check_seconds"):
        if type(timing[name]) not in (int, float) or not math.isfinite(timing[name]) or not 0 <= timing[name] <= 1e6:
            raise ValueError("Invalid retained study timing")
    if (timing["solver_seconds"] != solver["solve_seconds"] or timing["proving_seconds"] is not None
            or timing["verification_seconds"] is not None):
        raise ValueError("Invalid retained proof/solver timing claim")
    return deepcopy(study)


def save_study(path, study):
    raw = canonical(inspect_study(study))
    with Path(path).open("xb") as handle:
        handle.write(raw)


def load_study(path):
    raw, value = linear._load_json(path)
    if raw != canonical(value):
        raise ValueError("Julia study bytes must be canonical")
    return inspect_study(value)
