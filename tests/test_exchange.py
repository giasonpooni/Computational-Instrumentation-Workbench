"""Exchange records remain external claims, not native CIW scientific state."""

from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ciw import exchange
from ciw.cli import main


ROOT = Path(__file__).resolve().parents[1]


def observation():
    return json.loads((ROOT / "examples/exchange/observation.json").read_text())


def seal(artifact, field):
    payload = {key: value for key, value in artifact.items() if key != field}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode()
    artifact[field] = "sha256:" + sha256(artifact["schema"].encode() + b"\0" + canonical).hexdigest()
    return artifact


@pytest.fixture
def validator_repo():
    path = os.environ.get("CIW_SET_REPO")
    if not path:
        pytest.skip("Set CIW_SET_REPO to the pinned State Estimation Evaluation Testbed")
    return Path(path)


@pytest.mark.parametrize("data, message", [
    (b'[]', "unsupported"),
    (b'{"schema":"bogus"}', "unsupported"),
    (b'{"schema":[]}', "unsupported"),
    (b'{"schema":{}}', "unsupported"),
    (b'{"schema":true}', "unsupported"),
    (b'{"schema":null}', "unsupported"),
    (b'{"schema":"a","schema":"b"}', "duplicate"),
    (b'{"value":NaN}', "nonfinite"),
    (b'{"value":Infinity}', "nonfinite"),
    (b'{"value":1e999}', "overflows"),
    (b'\xff', "UTF-8"),
    (b'[' * 2000 + b']' * 2000, "bounded"),
])
def test_bad_json_rejected_before_validator_binding(tmp_path, data, message, monkeypatch):
    source = tmp_path / "input.json"
    source.write_bytes(data)
    def forbidden(*args):
        pytest.fail("malformed input must not bind an external validator")
    monkeypatch.setattr(exchange, "_validator", forbidden)
    with pytest.raises(ValueError, match=message):
        exchange.inspect_exchange([source], validator_repo=tmp_path)
    assert source.read_bytes() == data


def test_input_budgets_and_regular_file_refusal(tmp_path):
    with pytest.raises(ValueError, match="between"):
        exchange.inspect_exchange([], validator_repo=tmp_path)
    with pytest.raises(ValueError, match="between"):
        exchange.inspect_exchange([tmp_path] * 33, validator_repo=tmp_path)
    with pytest.raises(ValueError, match="regular"):
        exchange.inspect_exchange([tmp_path], validator_repo=tmp_path)
    source = tmp_path / "large.json"
    source.write_bytes(b" " * (exchange.MAX_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        exchange.inspect_exchange([source], validator_repo=tmp_path)


def test_dimension_is_bounded_before_numerical_work(tmp_path):
    data = observation()
    data["components"] *= 40
    source = tmp_path / "large.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="1 to 64"):
        exchange.inspect_exchange([source], validator_repo=tmp_path)


def test_combined_input_budget_is_enforced_before_validator_loading(tmp_path, monkeypatch):
    source = tmp_path / "observation.json"
    raw = json.dumps(observation()).encode()
    source.write_bytes(raw)
    monkeypatch.setattr(exchange, "MAX_TOTAL_BYTES", len(raw) + 1)
    with pytest.raises(ValueError, match="combined"):
        exchange.inspect_exchange([source, source], validator_repo=tmp_path)


def test_changed_validator_source_is_not_executed(tmp_path):
    directory = tmp_path / "state_estimation_testbed"
    directory.mkdir()
    (directory / "contracts.py").write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="source pin"):
        exchange._validator(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO race check")
def test_replaced_regular_path_cannot_hang_on_a_fifo(tmp_path, monkeypatch):
    regular = tmp_path / "regular"
    regular.write_text("{}")
    metadata = regular.stat()
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr(Path, "stat", lambda *args, **kwargs: metadata)
    with pytest.raises(ValueError, match="regular"):
        exchange._read(fifo, 1024)


def test_ci_gate_pins_the_same_checker_and_exact_producer_revisions():
    import re
    descriptor = json.loads((ROOT / "src/ciw/pipelines/descriptors/instrument-exchange.json").read_text())
    manifest = descriptor["steps"][0]["pin"]
    registry = json.loads((ROOT / "ci/gates.json").read_text())
    gate, = (entry for entry in registry["gates"] if entry["gate"] == "exchange")
    # The gate checks out the checker the pipeline executes and exact producer revisions.
    assert gate["kinds"] == ["instrument-exchange"]
    from ciw import exchange_adapter
    assert exchange_adapter.PIN == manifest and re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
    producers = [registry["extra_pins"][name]["revision"] for name in gate["extra_pins"]]
    assert len(producers) == 2 and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in producers)


def test_cli_failure_is_explicit_without_output_or_workspace(tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text("[]")
    assert main(["exchange", "inspect", str(source), "--validator-repo", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported" in captured.err
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("field,schema", [
    ("result_id", exchange.RESULT_SCHEMA),
    ("verification_id", exchange.VERIFICATION_SCHEMA),
])
def test_exchange_content_identity_cannot_be_reused_for_other_content(field, schema):
    record = seal({"schema": schema, "example": "é"}, field)
    assert exchange._identity(record, field) == "content_recomputed_not_authenticated"
    record["example"] = "different"
    with pytest.raises(ValueError, match="content"):
        exchange._identity(record, field)


def test_orchestration_does_not_mutate_or_promote_valid_claims(tmp_path, monkeypatch):
    # This unit seam tests CIW orchestration only; the real validator is exercised
    # by the opt-in conformance tests and cross-repository producer roundtrip.
    checker = SimpleNamespace(validate_observation_batch=lambda value: None)
    monkeypatch.setattr(exchange, "_validator", lambda path: (checker, {"test": True}))
    source = tmp_path / "observation.json"
    data = observation()
    raw = json.dumps(data, indent=4).encode()
    source.write_bytes(raw)
    report = exchange.inspect_exchange([source], validator_repo=tmp_path)
    assert report["artifacts"][0]["artifact"] == data
    assert report["artifacts"][0]["source_bytes_sha256"] == sha256(raw).hexdigest()
    assert report["authority"]["may_authorize"] is False
    assert report["authority"]["native_workspace_import"] == "not_performed"
    assert report["links"][0]["status"] == "unresolved_external_reference"
    assert data["covariance"]["numerical_status"] == "unchecked"
    assert source.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [source]
    with pytest.raises(ValueError, match="duplicate"):
        exchange.inspect_exchange([source, source], validator_repo=tmp_path)


@pytest.mark.parametrize("status,matrix,rank", [
    ("reported", [[0.04, 0.01], [0.01, 0.09]], 2),
    ("reported", [[0, 0], [0, 0]], 0),
    ("reported", [[1, 1], [1, 1]], 1),
    ("unknown", None, None),
    ("not_applicable", None, None),
])
def test_actual_validator_preserves_full_covariance_or_explicit_absence(
    validator_repo, tmp_path, status, matrix, rank,
):
    data = observation()
    data["covariance"].update(status=status, matrix=matrix)
    source = tmp_path / "observation.json"
    source.write_text(json.dumps(data))
    before = source.read_bytes()
    report = exchange.inspect_exchange([source], validator_repo=validator_repo)
    result = report["artifacts"][0]
    assert result["covariance_validation"]["effective_rank"] == rank
    assert result["artifact"] == data
    assert source.read_bytes() == before


@pytest.mark.parametrize("mutate", [
    lambda item: item["covariance"].update(matrix=[[1, 2], [2, 1]]),
    lambda item: item["covariance"].update(matrix=[[1, 0], [1, 1]]),
    lambda item: item["covariance"].update(status="unknown"),
    lambda item: item["covariance"].update(variables=["north", "east"]),
    lambda item: item["covariance"].update(units=["s", "m"]),
    lambda item: item.update(observed_at="2026-09-20"),
])
def test_actual_validator_refuses_inconsistent_scientific_claims(validator_repo, tmp_path, mutate):
    data = observation()
    mutate(data)
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        exchange.inspect_exchange([source], validator_repo=validator_repo)


def test_actual_validator_does_not_execute_package_initializers(validator_repo, tmp_path):
    import shutil
    path = tmp_path / "state_estimation_testbed"
    path.mkdir()
    shutil.copyfile(validator_repo / "state_estimation_testbed/contracts.py", path / "contracts.py")
    (path / "__init__.py").write_text("raise AssertionError('not approved')")
    report = exchange.inspect_exchange([ROOT / "examples/exchange/observation.json"], validator_repo=tmp_path)
    assert report["status"] == "conformant"
