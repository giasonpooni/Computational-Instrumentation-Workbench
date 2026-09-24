"""Assembly/lineage tests; native provider science remains in its own repos."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from ciw import measurement_chain as module
from ciw.measurement_chain_view import project
from ciw.telemetry import canonical, digest


EXAMPLE = Path(__file__).parents[1] / "examples" / "measurement-chain" / "source.json"


@pytest.fixture(scope="module")
def declared():
    return json.loads(EXAMPLE.read_text())


@pytest.fixture(scope="module")
def repos():
    root = os.environ.get("CIW_MEASUREMENT_CHAIN_STACK_ROOT")
    if not root:
        pytest.skip("Bind CIW_MEASUREMENT_CHAIN_STACK_ROOT for the native assembly gate")
    return {role: Path(root) / role for role in module.ROLES}


@pytest.fixture(scope="module")
def original(repos):
    return module.create_session(EXAMPLE.read_bytes(), repos)


@pytest.fixture(scope="module")
def replayed(original, repos):
    return module.replay_session(original, repos)


def test_source_retains_exact_original_public_raw_bytes(declared):
    original = json.loads((EXAMPLE.parents[1] / "adapters" / "two-reservoir-covariance.json").read_text())
    assert module._source(EXAMPLE.read_bytes())["investigation"] == original
    assert declared["configuration"]["gsie_fusion"] == "not_performed"


@pytest.mark.parametrize("change", ["independence", "record_count", "mapping_dimension", "unknown_source", "configuration"])
def test_source_refuses_implicit_or_unsupported_mapping(declared, change):
    source = deepcopy(declared)
    if change == "independence":
        source["investigation"]["cross_assembly_independent"] = "unknown"
    elif change == "record_count":
        source["investigation"]["sensors"][0]["request"]["inputs"]["records"] *= 2
    elif change == "mapping_dimension":
        source["covariance_map"]["jacobian"][0].append(0.0)
    elif change == "unknown_source":
        source["covariance_map"]["source_artifact"] = "estimated_diagonal"
    else:
        source["configuration"]["gsie_fusion"] = "performed"
    with pytest.raises(ValueError):
        module._source(canonical(source))


def test_native_lineage_and_shared_catalog_preserve_occurrences(original):
    module._validate(original)
    steps = module.catalog_steps(original)
    assert [s["operation_id"] for s in steps] == [module.FSRT_OPERATION, module.JSPT_OPERATION]
    assert steps[0]["result_id"] in steps[1]["input_refs"]
    assert all(s["result"]["verification_status"] == "not_verified" for s in steps)
    claims = module.identity_claims(original)
    assert all(s["execution_id"] in claims and s["result_id"] in claims for s in steps)
    assert len(module.native_occurrences(original)) == 4


def test_fresh_replay_has_stable_science_and_new_native_occurrences(original, replayed):
    fresh = replayed["session"]
    assert original["steps"][0]["numerical_result"] == fresh["steps"][0]["numerical_result"]
    assert not module.native_occurrences(original) & module.native_occurrences(fresh)
    assert replayed["replay_receipt"]["numerical_match"] is True
    assert fresh["source"] == original["source"]
    module._validate(fresh)


def test_inspection_cannot_bind_or_call_a_runtime(original, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Inspection attempted provider execution")
    monkeypatch.setattr(module, "_runtime", refuse)
    monkeypatch.setattr(module, "create_investigation", refuse)
    module._validate(original)


def test_native_copy_cannot_masquerade_as_fresh_reproduction(original):
    altered = deepcopy(original)
    new = altered["verification"]["reproduction"]
    old = altered["steps"][0]
    new["result"]["data"] = deepcopy(old["result"]["data"])
    new["result"]["result_id"] = digest({k: v for k, v in new["result"].items() if k != "result_id"})
    new["result_id"] = new["result"]["result_id"]
    new["result_sha256"] = digest(new["result"])
    from ciw.declared_workload import _verification
    altered["verification"] = _verification(altered, new)
    with pytest.raises(ValueError, match="fresh native"):
        module._validate(altered)


def test_native_map_cannot_select_another_result_even_when_outer_resealed(original):
    altered = deepcopy(original)
    workspace = altered["steps"][0]["result"]["data"]["native_workspace"]
    workspace["results"][1]["parameters"]["source_result_id"] = "result-" + "f" * 32
    with pytest.raises(ValueError):
        module._check_data(module._source(EXAMPLE.read_bytes()), altered["steps"][0]["result"]["data"])


def test_view_uses_native_covariances_without_gsie_state(original, declared):
    record = {"native": original, "kind": "measurement-chain", "bundle_id": original["bundle_digest"]}
    source = {"source_id": "source:measurement", "evidence_id": original["source"]["evidence"][0]["artifact_ref"], "label": "Declared measurement chain"}
    view = project(record, source, declared, 1)
    assert view["fusion_context"] is None
    assert view["object_context"]["object_kind"] == "measurement-chain-testbed"
    panel = next(p for p in view["panels"] if p["panel_id"] == "posterior")
    assert panel["covariance"] == original["steps"][0]["result"]["data"]["native_workspace"]["results"][0]["data"]["unprojected_estimate"]["covariance"]
    panel["covariance"][0][0] = -999
    module._validate(original)


def test_expired_calibration_refuses_native_execution(declared, repos):
    altered = deepcopy(declared)
    altered["investigation"]["sensors"][0]["request"]["inputs"]["calibration"]["valid_until"] = "2026-01-02T00:00:00Z"
    with pytest.raises(module.AdapterRefusal) as error:
        module.create_session(canonical(altered), repos)
    assert error.value.code == "calibration_unavailable"


def test_large_balance_disagreement_preserves_native_hold(declared, repos):
    altered = deepcopy(declared)
    altered["investigation"]["model"]["total_mass_kg"] = 150.0
    held = module.create_session(canonical(altered), repos)
    fsrt = held["steps"][0]["result"]["data"]["native_workspace"]["results"][0]["data"]
    assert fsrt["diagnostics"]["reconciliation_status"] == "model_inconsistent"
    for field in ("values", "covariance", "unit"):
        assert fsrt["estimate"][field] == fsrt["unprojected_estimate"][field]
