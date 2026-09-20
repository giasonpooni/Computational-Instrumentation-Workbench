"""Exercise the pinned PLSR terminal boundary and retained scientific evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


L = pytest.importorskip("lyapunov", reason="install the optional plsr extra")

from ciw import plsr


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "plsr"
PIN = "19ea6967060166ba09db6cd4563bd87bd6b3d196"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def cli(*args, exit_code=0):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "ciw", *map(str, args)], cwd=ROOT,
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == exit_code, completed.stderr
    if exit_code == 2:
        assert not completed.stdout
        assert completed.stderr
        return completed
    assert not completed.stderr
    return json.loads(completed.stdout)


@pytest.fixture
def inputs(tmp_path):
    model = write(tmp_path / "model.json", load(EXAMPLES / "continuous-affine.json"))
    sample = write(tmp_path / "sample.json", {
        "sample_schema": "plsr-sample-v1", "x": [0.1, 0.2],
        "theta": [0.5], "theta_dot": [0.1],
    })
    return model, sample


def saved_run(tmp_path, inputs):
    model, sample = inputs
    return plsr.evaluate_run(model, sample, tmp_path / "runs")


def test_terminal_import_evaluate_inspect_and_replay_are_self_contained(tmp_path, inputs):
    model, sample = inputs
    source_bytes = model.read_bytes(), sample.read_bytes()
    imported_path = tmp_path / "imported.json"
    imported = cli("plsr", "import", model, "--output", imported_path)
    declared = load(imported_path)
    assert declared == load(model)
    assert imported["model_artifact_digest"] == declared["artifact_digest"]

    evaluated = cli("plsr", "evaluate", "--model", imported_path, "--sample", sample,
                    "--output-dir", tmp_path / "runs")
    path = Path(evaluated["saved_file"])
    original_bytes = path.read_bytes()
    bundle = load(path)
    assert bundle["bundle_schema"] == "ciw-plsr-run-v1"
    assert bundle["model"] == declared
    assert bundle["sample"] == load(sample)
    assert bundle["record"]["model_artifact_digest"] == declared["artifact_digest"]
    assert L.verify_record(bundle["record"])
    assert bundle["record"]["code"] == L.CERTIFIED_WITH_MARGIN
    assert bundle["runtime"]["commit"] == PIN
    assert bundle["verification_id"] is None
    assert bundle["verification_status"] == "not_verified"
    assert bundle["replay_of"] is None
    assert bundle["record"]["proof_status"] == "NOT_CHECKED"
    assert bundle["record"]["may_authorize"] is False
    assert (model.read_bytes(), sample.read_bytes()) == source_bytes

    # Neither inspection nor replay should need the original input paths.
    imported_path.unlink()
    model.unlink()
    sample.unlink()
    inspected = cli("plsr", "inspect", path)
    assert inspected["bundle"] == bundle
    replayed = cli("plsr", "replay", path, "--output-dir", tmp_path / "replayed")
    replay = load(replayed["saved_file"])
    assert replay["evidence_id"] == bundle["evidence_id"]
    assert replay["operation_id"] == bundle["operation_id"]
    assert replay["execution_id"] != bundle["execution_id"]
    assert replay["result_id"] != bundle["result_id"]
    assert replay["record"] == bundle["record"]
    assert replay["replay_of"]["source_result_id"] == bundle["result_id"]
    assert replay["replay_of"]["source_bundle_digest"] == bundle["bundle_digest"]
    assert replay["replay_of"]["source_record_digest"] == bundle["record"]["record_digest"]
    assert replay["replay_of"]["record_digest_matches"] is True
    assert replay["verification_status"] == "not_verified"
    assert path.read_bytes() == original_bytes


def test_import_refuses_to_reseal_or_overwrite_a_declaration(tmp_path, inputs):
    model, _ = inputs
    target = tmp_path / "imported.json"
    plsr.import_model(model, target)
    original = target.read_bytes()
    plsr.import_model(model, target)
    assert target.read_bytes() == original
    changed = load(model)
    changed["model_version"] = "2"
    write(model, L.seal_model_artifact(changed))
    with pytest.raises((ValueError, FileExistsError)):
        plsr.import_model(model, target)
    assert target.read_bytes() == original
    changed = load(model)
    changed["policy"]["required_margin"] = 0.0
    write(model, changed)
    rejected_target = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="digest"):
        plsr.import_model(model, rejected_target)
    assert not rejected_target.exists()


@pytest.mark.parametrize("updates,removed", [
    ({"x": [0.1]}, None),
    ({"x": [True, 0.1]}, None),
    ({"x": ["0.1", 0.2]}, None),
    ({"x": [2**53 + 1, 0.2]}, None),
    ({"theta": None}, None),
    ({"theta": []}, None),
    ({"theta_dot": None}, None),
    ({"theta_dot": [0, 0]}, None),
    ({"sample_schema": "plsr-sample-v999"}, None),
    ({"invented": 1}, None),
    ({}, "theta_dot"),
])
def test_bad_samples_are_errors_before_any_evidence_is_written(tmp_path, inputs, updates, removed):
    model, sample = inputs
    document = load(sample)
    document.update(updates)
    if removed:
        del document[removed]
    write(sample, document)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        plsr.evaluate_run(model, sample, output)
    assert not output.exists()


@pytest.mark.parametrize("bad_number", ["NaN", "Infinity", "1e400", "1e-400"])
def test_nonfinite_or_underflowing_json_is_rejected(tmp_path, inputs, bad_number):
    model, sample = inputs
    sample.write_text(
        '{"sample_schema":"plsr-sample-v1","x":[' + bad_number
        + ',0.2],"theta":[0.5],"theta_dot":[0.1]}', encoding="utf-8",
    )
    output = tmp_path / "not-created"
    cli("plsr", "evaluate", "--model", model, "--sample", sample,
        "--output-dir", output, exit_code=2)
    assert not output.exists()


def test_duplicate_sample_keys_are_not_silently_overwritten(tmp_path, inputs):
    model, sample = inputs
    sample.write_text(
        '{"sample_schema":"plsr-sample-v1","x":[0.1,0.2],'
        '"theta":[0.5],"theta_dot":[0.1],"theta_dot":[0]}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="(?i)duplicate"):
        plsr.evaluate_run(model, sample, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("change,expected_code", [
    ({"theta": [2.0]}, L.OUTSIDE_PARAMETER_BOX),
    ({"theta_dot": [0.3]}, L.OUTSIDE_PARAMETER_BOX),
    ({"x": [3.0, 0.0]}, L.OUTSIDE_LEVEL_SET),
])
def test_domain_refusals_are_saved_evaluations_with_successful_exit(tmp_path, inputs, change, expected_code):
    model, sample = inputs
    sample_data = load(sample)
    sample_data.update(change)
    write(sample, sample_data)
    result = cli("plsr", "evaluate", "--model", model, "--sample", sample,
                 "--output-dir", tmp_path / "runs")
    bundle = load(result["saved_file"])
    assert bundle["sample"] == sample_data
    assert bundle["record"]["code"] == expected_code
    assert bundle["record"]["presentation_category"] == "outside_declared_domain"
    assert bundle["record"]["operationally_acceptable"] is False
    assert L.verify_record(bundle["record"])


def test_artifact_policy_margin_is_applied_without_a_caller_override(tmp_path, inputs):
    model, sample = inputs
    declared = load(model)
    declared["policy"]["required_margin"] = 10.0
    write(model, L.seal_model_artifact(declared))
    result = plsr.evaluate_run(model, sample, tmp_path / "runs")
    record = result["bundle"]["record"]
    assert record["code"] == L.MARGIN_LOW
    assert record["required_margin"] == 10.0
    assert record["level"] == declared["policy"]["level"]
    assert record["inequality_certified"] is True
    assert record["meets_required_margin"] is False
    assert record["operationally_acceptable"] is False


@pytest.mark.parametrize("matrix,code,category", [
    ([[-1e-17, 1.0], [-1.0, -1e-17]], L.NUMERICAL_INCONCLUSIVE, "numerical_refusal"),
    ([[1.0, 0.0], [0.0, 1.0]], L.NOT_CERTIFIED, "certificate_violation"),
])
def test_numerical_refusal_is_distinct_from_a_certificate_violation(tmp_path, matrix, code, category):
    declared = load(EXAMPLES / "discrete-linear.json")
    declared["time"] = {"convention": "continuous", "sample_period_s": None}
    declared["plant"]["A"] = matrix
    model = write(tmp_path / "model.json", L.seal_model_artifact(declared))
    sample = write(tmp_path / "sample.json", {
        "sample_schema": "plsr-sample-v1", "x": [1.0, 0.0],
        "theta": None, "theta_dot": None,
    })
    result = cli("plsr", "evaluate", "--model", model, "--sample", sample,
                 "--output-dir", tmp_path / "runs")
    assert result["bundle"]["record"]["code"] == code
    assert result["bundle"]["record"]["presentation_category"] == category
    assert result["bundle"]["record"]["inequality_certified"] is False
    assert result["bundle"]["record"]["operationally_acceptable"] is False
    assert L.verify_record(result["bundle"]["record"])


def test_discrete_linear_model_requires_null_parameters_and_rates(tmp_path):
    model = EXAMPLES / "discrete-linear.json"
    document = {"sample_schema": "plsr-sample-v1", "x": [0.25, -0.5],
                "theta": None, "theta_dot": None}
    sample = write(tmp_path / "sample.json", document)
    result = plsr.evaluate_run(model, sample, tmp_path / "runs")
    assert result["bundle"]["record"]["code"] == L.CERTIFIED_WITH_MARGIN
    assert result["bundle"]["model"]["time"]["sample_period_s"] == 0.01
    for key in ("theta", "theta_dot"):
        write(sample, {**document, key: [0.0]})
        with pytest.raises(ValueError):
            plsr.evaluate_run(model, sample, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_inspection_checks_saved_evidence_without_calling_the_numerical_engine(tmp_path, inputs, monkeypatch):
    result = saved_run(tmp_path, inputs)
    def unexpected_verdict(*args, **kwargs):
        raise AssertionError("inspection must not evaluate a model")
    monkeypatch.setattr(L.ModelArtifact, "verdict", unexpected_verdict)
    inspected = plsr.inspect_run(Path(result["saved_file"]))
    assert inspected["bundle"] == result["bundle"]


@pytest.mark.parametrize("path,value", [
    (("model", "policy", "required_margin"), 0.0),
    (("sample", "x", 0), 1.0),
    (("record", "code"), L.NOT_CERTIFIED),
    (("runtime", "commit"), "0" * 40),
    (("verification_status",), "verified"),
    (("verification_id",), "unearned-verification"),
    (("bundle_digest",), "0" * 64),
])
def test_tampered_bundle_is_rejected_before_replay_writes(tmp_path, inputs, path, value):
    result = saved_run(tmp_path, inputs)
    bundle = load(result["saved_file"])
    parent = bundle
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    source = write(tmp_path / "tampered.json", bundle)
    with pytest.raises(ValueError):
        plsr.inspect_run(source)
    with pytest.raises(ValueError):
        plsr.replay_run(source, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("path,value", [
    (("record", "model_artifact_digest"), "0" * 64),
    (("record", "sample", "x", 0), 1.0),
    (("record", "required_margin"), 0.0),
    (("record", "level"), None),
    (("record", "may_authorize"), True),
    (("record", "proof_status"), "VERIFIED"),
    (("record", "presentation_category"), "numerical_refusal"),
    (("runtime", "commit"), "0" * 40),
    (("evidence_id",), "sha256:" + "0" * 64),
    (("verification_status",), "verified"),
])
def test_rehashing_cannot_detach_bindings_or_lift_claim_restrictions(tmp_path, inputs, path, value):
    result = saved_run(tmp_path, inputs)
    bundle = load(result["saved_file"])
    parent = bundle
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    # Digests are integrity checks, not signatures. Even freshly recomputed
    # checksums must not allow contradictory sample/policy/claim bindings.
    bundle["record"]["record_digest"] = L.record_digest(bundle["record"])
    assert L.verify_record(bundle["record"])
    bundle["bundle_digest"] = plsr._bundle_digest(bundle)
    source = write(tmp_path / "rehashed.json", bundle)
    with pytest.raises(ValueError):
        plsr.inspect_run(source)
    with pytest.raises(ValueError):
        plsr.replay_run(source, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_replay_mismatch_retains_new_evidence_and_returns_distinct_exit_code(tmp_path, inputs):
    result = saved_run(tmp_path, inputs)
    bundle = load(result["saved_file"])
    # A well-formed receipt can be consistently resealed by anyone; inspection
    # checks integrity and bindings, while replay checks the recorded outcome.
    bundle["record"]["details"] = "An independently supplied recorded explanation."
    bundle["record"]["record_digest"] = L.record_digest(bundle["record"])
    bundle["bundle_digest"] = plsr._bundle_digest(bundle)
    source = write(tmp_path / "different-receipt.json", bundle)
    original_bytes = source.read_bytes()
    assert plsr.inspect_run(source)["bundle"] == bundle
    replayed = cli("plsr", "replay", source, "--output-dir", tmp_path / "replayed", exit_code=3)
    replay = load(replayed["saved_file"])
    assert replay["record"]["code"] == bundle["record"]["code"]
    assert replay["record"]["record_digest"] != bundle["record"]["record_digest"]
    assert replay["replay_of"]["record_digest_matches"] is False
    assert replay["verification_id"] is None
    assert replay["verification_status"] == "not_verified"
    assert replay["execution_id"] != bundle["execution_id"]
    assert source.read_bytes() == original_bytes


def test_discrete_affine_model_uses_explicit_parameters_and_null_rate(tmp_path, inputs):
    model, sample = inputs
    declared = load(model)
    declared["time"] = {"convention": "discrete", "sample_period_s": 0.01}
    declared["plant"]["A0"] = [[0.8, 0.0], [0.0, 0.7]]
    declared["plant"]["terms"] = [[[0.05, 0.0], [0.0, 0.025]]]
    declared["plant"]["rate_box"] = None
    declared["certificate"] = {"kind": "quadratic", "P": [[1.0, 0.0], [0.0, 1.0]]}
    write(model, L.seal_model_artifact(declared))
    data = load(sample)
    data["theta_dot"] = None
    write(sample, data)
    result = plsr.evaluate_run(model, sample, tmp_path / "runs")
    record = result["bundle"]["record"]
    assert record["code"] == L.CERTIFIED_WITH_MARGIN
    assert record["sample"]["theta"] == [0.5]
    assert record["sample"]["theta_dot"] is None
    assert record["diagnostics"]["P_rate"] is None
    assert plsr.inspect_run(Path(result["saved_file"])) == result
    write(sample, {**data, "theta_dot": [0.0]})
    with pytest.raises(ValueError, match="theta_dot"):
        plsr.evaluate_run(model, sample, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_matrix_overflow_is_retained_as_a_sampleless_numerical_refusal(tmp_path, inputs):
    model, sample = inputs
    declared = load(model)
    declared["plant"]["A0"] = [[-1e308, 0.0], [0.0, -1e308]]
    declared["certificate"] = {"kind": "quadratic", "P": [[1.0, 0.0], [0.0, 1.0]]}
    write(model, L.seal_model_artifact(declared))
    result = plsr.evaluate_run(model, sample, tmp_path / "runs")
    record = result["bundle"]["record"]
    assert record["code"] == L.NUMERICAL_OVERFLOW
    assert record["presentation_category"] == "numerical_refusal"
    assert record["diagnostics"] is None
    assert record["inequality_certified"] is False
    assert record["meets_required_margin"] is False
    assert record["operationally_acceptable"] is False
    assert L.verify_record(record)
    assert plsr.inspect_run(Path(result["saved_file"])) == result


@pytest.mark.parametrize("magnitude,code,overflow", [
    (1e200, L.OUTSIDE_LEVEL_SET, True),
    (1e-200, L.CERTIFIED_WITH_MARGIN, False),
])
def test_extreme_state_retains_scaled_diagnostics_without_changing_the_verdict(
    tmp_path, inputs, magnitude, code, overflow,
):
    model, sample = inputs
    data = load(sample)
    data["x"] = [magnitude, -magnitude]
    write(sample, data)
    result = plsr.evaluate_run(model, sample, tmp_path / "runs")
    record = result["bundle"]["record"]
    diagnostics = record["diagnostics"]
    assert record["code"] == code
    assert diagnostics["value_out_of_range"] is True
    assert diagnostics["scaled_value"] > 0
    assert diagnostics["scaled_decrease"] < 0
    for quantity in ("value", "decrease"):
        assert diagnostics[quantity] == (None if overflow else 0.0)
        assert diagnostics[quantity + "_overflow"] is overflow
        assert diagnostics[quantity + "_underflow"] is not overflow
        assert diagnostics[quantity + "_mantissa_exponent"][0] != 0
    assert L.verify_record(record)
    json.dumps(result["bundle"], allow_nan=False)
    assert plsr.inspect_run(Path(result["saved_file"])) == result
