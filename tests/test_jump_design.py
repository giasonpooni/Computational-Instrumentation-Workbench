"""Host conversion and offline retention checks using explicitly synthetic records.

These fixtures do not assert that a Julia executable or physical experiment ran.
Native worker validation is a separate gate.
"""
import base64
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace

import pytest

from ciw import design_adjustment as da
from ciw import jump_design as jd
from ciw.telemetry import byte_digest, canonical, digest


def ratio(numerator, denominator=1):
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def problem():
    return da.make_problem(ratio(3), ratio(10), ratio(1), ratio(-1, 10), ratio(1, 10), ratio(0))


def artifact(raw):
    return {"sha256": byte_digest(raw), "base64": base64.b64encode(raw).decode("ascii")}


def solver_bytes(**updates):
    values = {"schema": "ciw.jump-scalar-design-output.v1", "termination_status": "OPTIMAL",
              "primal_status": "FEASIBLE_POINT", "delta": 0.1, "objective_value": 0.17,
              "solve_seconds": 0.001, "julia_version": "1.10.10",
              "jump_version": "1.29.0", "highs_version": "1.21.0"}
    values.update(updates)
    return ("\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items()) + "\n").encode()


def synthetic_study():
    """Only a retained binding fixture; no executable is launched or attested."""
    p = problem()
    raw_input = ['schema = "ciw.jump-scalar-design-input.v1"']
    for name in ("x", "target", "regularization", "lower", "upper"):
        raw_input.extend([f"[{name}]", f"numerator = {p[name]['numerator']}",
                          f"denominator = {p[name]['denominator']}"])
    files = {
        "Project.toml": artifact(b"# Synthetic test fixture; not a resolved project\n[deps]\n"),
        "Manifest.toml": artifact(b'julia_version = "1.10.10"\n'
                                  b'[[deps.JuMP]]\nversion = "1.29.0"\n'
                                  b'[[deps.HiGHS]]\nversion = "1.21.0"\n'),
        "worker.jl": artifact(b"# Synthetic test fixture; no worker execution\n"),
    }
    solver = jd._solver(solver_bytes())
    conversion = {
        "method": "exact_binary64_ratio_then_projection_to_exact_box",
        "binary64_hex": "0x1.999999999999ap-4",
        "raw_delta": ratio(3602879701896397, 36028797018963968),
        "checked_delta": ratio(1, 10), "adjustment": ratio(-1, 180143985094819840),
        "claim_target": "projected_exact_candidate_not_raw_solver_output",
    }
    study = {
        "schema": "ciw.jump-design-study.v1", "problem": p, "problem_digest": digest(p),
        "producer_invocation_id": "12345678-1234-4234-8234-123456789abc", "artifacts": files,
        "runtime": {"julia_version": "1.10.10", "jump_version": "1.29.0", "highs_version": "1.21.0",
                    "julia_executable_sha256": byte_digest(b"synthetic test executable identity"),
                    "files": {name: value["sha256"] for name, value in files.items()},
                    "source_to_binary_attestation": "not_established"},
        "worker_input": artifact(("\n".join(raw_input) + "\n").encode()),
        "worker_stdout": artifact(solver_bytes()), "worker_stderr": artifact(b"synthetic diagnostic\n"),
        "solver": solver, "conversion": conversion,
        "check": da.check_candidate(da.make_candidate(p, ratio(1, 10))),
        "timing": {"process_seconds": 0.25, "solver_seconds": 0.001,
                   "certificate_preparation_seconds": 0.003, "exact_check_seconds": 0.004,
                   "proving_seconds": None, "verification_seconds": None},
        "authority": deepcopy(jd.AUTHORITY),
    }
    return reseal(study)


def reseal(study):
    study["study_id"] = digest({key: value for key, value in study.items() if key != "study_id"})
    return study


def test_binary64_boundary_adjustment_is_exact_visible_and_has_narrow_claim():
    converted = jd.conversion(problem(), 0.1)
    assert converted["raw_delta"] == ratio(3602879701896397, 36028797018963968)
    assert converted["checked_delta"] == ratio(1, 10)
    assert converted["adjustment"] == ratio(-1, 180143985094819840)
    assert converted["binary64_hex"] == "0x1.999999999999ap-4"
    assert converted["claim_target"] == "projected_exact_candidate_not_raw_solver_output"


@pytest.mark.parametrize("raw", [0.0, -0.0, 0.0625, -0.0625])
def test_interior_binary64_value_is_retained_without_decimal_rounding(raw):
    converted = jd.conversion(problem(), raw)
    assert converted["raw_delta"] == ratio(Fraction.from_float(raw))
    assert converted["checked_delta"] == ratio(Fraction.from_float(raw))
    assert converted["adjustment"] == ratio(0)
    assert converted["binary64_hex"] == raw.hex()


@pytest.mark.parametrize("raw, checked, adjustment", [
    (0.125, Fraction(1, 10), Fraction(-1, 40)),
    (-0.125, Fraction(-1, 10), Fraction(1, 40)),
    (100.0, Fraction(1, 10), Fraction(-999, 10)),
])
def test_off_box_output_is_projected_explicitly_not_relabelled_feasible(raw, checked, adjustment):
    converted = jd.conversion(problem(), raw)
    assert converted["raw_delta"] == ratio(Fraction.from_float(raw))
    assert converted["checked_delta"] == ratio(checked)
    assert converted["adjustment"] == ratio(adjustment)
    assert converted["claim_target"] == "projected_exact_candidate_not_raw_solver_output"


@pytest.mark.parametrize("raw", [True, None, "0.1", float("nan"), float("inf"), -101, 101,
                                  float.fromhex("0x0.0000000000001p-1022")])
def test_conversion_refuses_invalid_values_and_rational_limb_overflow(raw):
    with pytest.raises(ValueError):
        jd.conversion(problem(), raw)


@pytest.mark.parametrize("updates", [
    {"schema": "wrong"}, {"termination_status": "TIME_LIMIT"}, {"primal_status": "NO_SOLUTION"},
    {"delta": True}, {"delta": "0.1"}, {"delta": 101}, {"objective_value": 1e13},
    {"solve_seconds": -1}, {"julia_version": "unknown"}, {"jump_version": False},
    {"highs_version": "1.21"}, {"extra": "ignored?"},
])
def test_solver_response_refuses_nonconforming_status_fields_and_versions(updates):
    with pytest.raises(ValueError):
        jd._solver(solver_bytes(**updates))


@pytest.mark.parametrize("raw", [
    b"", b"[", b"\xff", b'delta = nan\n',
    solver_bytes().replace(b"delta = 0.1\n", b"delta = nan\n"),
    solver_bytes().replace(b"delta = 0.1\n", b"delta = inf\n"),
    solver_bytes().replace(b"delta = 0.1\n", b""),
    solver_bytes() + b'delta = 0.1\n', b" " * 16385,
], ids=["empty", "broken-toml", "utf8", "incomplete", "nan", "infinity", "missing", "duplicate", "oversized"])
def test_malformed_solver_bytes_are_refused(raw):
    with pytest.raises(ValueError):
        jd._solver(raw)


def test_synthetic_retained_study_is_inspected_offline_without_fresh_check(tmp_path, monkeypatch):
    study = synthetic_study()
    destination = tmp_path / "study.json"
    jd.save_study(destination, study)

    def forbidden(*args, **kwargs):
        raise AssertionError("Retained inspection invoked numerical execution/checking")

    monkeypatch.setattr(jd, "run_study", forbidden)
    monkeypatch.setattr(jd.subprocess, "run", forbidden)
    monkeypatch.setattr(da, "make_candidate", forbidden)
    monkeypatch.setattr(da, "check_candidate", forbidden)
    assert jd.inspect_study(study) == study
    assert jd.load_study(destination) == study
    assert destination.read_bytes() == canonical(study)
    with pytest.raises((ValueError, FileExistsError)):
        jd.save_study(destination, study)


@pytest.mark.parametrize("mutate", [
    lambda s: s["conversion"].update(adjustment=ratio(0)),
    lambda s: s["conversion"].update(raw_delta=ratio(1, 10)),
    lambda s: s["conversion"].update(claim_target="raw_solver_output_certified"),
    lambda s: s["worker_stdout"].update(sha256="sha256:" + "0" * 64),
    lambda s: s.update(worker_stdout=artifact(solver_bytes(delta=0.0625))),
    lambda s: s.update(worker_input=artifact(b"not the bound problem")),
    lambda s: s["runtime"].update(jump_version="1.29.1"),
    lambda s: s["runtime"].update(julia_executable_sha256="unknown"),
    lambda s: s["runtime"].update(source_to_binary_attestation="verified"),
    lambda s: s["runtime"]["files"].update(**{"worker.jl": "sha256:" + "0" * 64}),
    lambda s: s["authority"].update(verification_id="invented"),
    lambda s: s["authority"].update(hardware_actuation="authorized"),
    lambda s: s.update(producer_invocation_id="not-an-id"),
    lambda s: s["timing"].update(proving_seconds=0.0),
    lambda s: s["timing"].update(verification_seconds=0.0),
    lambda s: s["timing"].update(solver_seconds=0.002),
    lambda s: s["timing"].update(exact_check_seconds=True),
    lambda s: s["solver"].update(delta=0.125),
])
def test_resealed_study_tampering_cannot_break_its_internal_bindings(mutate):
    study = synthetic_study()
    mutate(study)
    with pytest.raises(ValueError):
        jd.inspect_study(reseal(study))


def test_study_rejects_otherwise_valid_embedded_candidate_with_different_delta():
    study = synthetic_study()
    study["check"] = da.check_candidate(da.make_candidate(study["problem"], ratio(0)))
    with pytest.raises(ValueError):
        jd.inspect_study(reseal(study))


def test_manifest_versions_must_bind_reported_runtime():
    study = synthetic_study()
    study["artifacts"]["Manifest.toml"] = artifact(
        b'julia_version = "1.10.10"\n[[deps.JuMP]]\nversion = "1.30.0"\n'
        b'[[deps.HiGHS]]\nversion = "1.21.0"\n')
    with pytest.raises(ValueError):
        jd.inspect_study(reseal(study))


def test_retained_study_has_no_registered_execution_or_proof_claim():
    study = synthetic_study()
    assert jd.inspect_study(study)["check"]["status"] == "accepted"
    for key in ("registered_operation_id", "execution_id", "result_id", "verification_id", "proof_id"):
        assert study["authority"][key] is None
    assert study["timing"]["proving_seconds"] is None
    assert study["timing"]["verification_seconds"] is None
    assert study["runtime"]["source_to_binary_attestation"] == "not_established"


def test_study_loader_requires_canonical_bytes(tmp_path):
    destination = tmp_path / "study.json"
    destination.write_text(json.dumps(synthetic_study(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        jd.load_study(destination)


def checkout_solver_bytes():
    """A mocked response matching the checkout, not evidence it executed."""
    project = Path(jd.__file__).resolve().parents[2] / "runtimes" / "jump-design"
    manifest = tomllib.loads((project / "Manifest.toml").read_text(encoding="utf-8"))
    return solver_bytes(julia_version=manifest["julia_version"],
                        jump_version=manifest["deps"]["JuMP"][0]["version"],
                        highs_version=manifest["deps"]["HiGHS"][0]["version"])


def test_synthetic_run_executes_exact_snapshot_and_preserves_original_bytes(tmp_path, monkeypatch):
    project = Path(jd.__file__).resolve().parents[2] / "runtimes" / "jump-design"
    originals = {name: (project / name).read_bytes() for name in ("Project.toml", "Manifest.toml", "worker.jl")}
    executable = tmp_path / "synthetic-julia.exe"
    executable.write_bytes(b"Not an executable: subprocess is mocked")
    depot = tmp_path / "depot"
    depot.mkdir()
    observed = {}

    def synthetic_run(command, **kwargs):
        snapshot = Path(command[-1]).parent
        observed["snapshot"] = snapshot
        assert snapshot != project
        assert command[0] == str(executable.resolve())
        assert command[1:3] == ["--startup-file=no", "--history-file=no"]
        assert command[-2] == "--project=" + str(snapshot)
        assert Path(command[-1]).name == "worker.jl"
        assert kwargs["env"]["JULIA_PKG_OFFLINE"] == "true"
        assert kwargs["env"]["JULIA_NUM_THREADS"] == "1"
        assert kwargs["env"]["JULIA_LOAD_PATH"] == __import__("os").pathsep.join(("@", "@stdlib"))
        assert kwargs["env"]["JULIA_DEPOT_PATH"] == str(depot.resolve())
        assert kwargs["timeout"] == 30
        assert kwargs["capture_output"] is True
        assert kwargs["input"] == jd._input(problem())
        for name, raw in originals.items():
            assert (snapshot / name).read_bytes() == raw
        return SimpleNamespace(returncode=0, stdout=checkout_solver_bytes(), stderr=b"mocked invocation\n")

    monkeypatch.setattr(jd.subprocess, "run", synthetic_run)
    study = jd.run_study(problem(), executable, depot=depot, timeout=30)
    assert not observed["snapshot"].exists()
    assert study["check"]["status"] == "accepted"
    assert study["runtime"]["julia_executable_sha256"] == byte_digest(executable.read_bytes())
    for name, raw in originals.items():
        assert study["artifacts"][name] == artifact(raw)
        assert (project / name).read_bytes() == raw


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "oversized", "changed-executable"])
def test_synthetic_run_failure_does_not_emit_successful_study(tmp_path, monkeypatch, failure):
    executable = tmp_path / "synthetic-julia.exe"
    executable.write_bytes(b"Mock executable bytes; never launched")

    def failed_run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "changed-executable":
            executable.write_bytes(b"Changed during mocked invocation")
        return SimpleNamespace(returncode=1 if failure == "nonzero" else 0,
                               stdout=b" " * 16385 if failure == "oversized" else checkout_solver_bytes(),
                               stderr=b"mock worker refusal")

    monkeypatch.setattr(jd.subprocess, "run", failed_run)
    with pytest.raises(ValueError):
        jd.run_study(problem(), executable, timeout=1)


@pytest.mark.parametrize("timeout", [0, -1, 601, True, float("nan"), float("inf")])
def test_invalid_timeout_is_refused_before_any_process(tmp_path, monkeypatch, timeout):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid timeout reached process execution")

    monkeypatch.setattr(jd.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        jd.run_study(problem(), tmp_path / "does-not-exist", timeout=timeout)
