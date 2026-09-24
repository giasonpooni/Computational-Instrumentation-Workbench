"""Native GTE execution, identity, refusal and read-only catalog view gates."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.declared_workload import _verification
from ciw.geometric_circle import POLICY, _source, request, request_bytes, workflow
from ciw.geometric_circle_view import project
from ciw.core.canonical import bundle_digest, canonical, digest

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples/geometric-circle"


@pytest.fixture(scope="module")
def repositories():
    repo = os.environ.get("CIW_GTE_REPO")
    if not repo:
        pytest.skip("Set CIW_GTE_REPO for the pinned native GTE catalog gate")
    return {"gte": Path(repo).resolve()}


@pytest.fixture(scope="module")
def sessions(repositories):
    before = set(sys.modules)
    raw = (EXAMPLES / "source.json").read_bytes()
    original = workflow.create_session(raw, repositories)
    replay = workflow.replay_session(original, repositories)
    held = workflow.create_session((EXAMPLES / "held.json").read_bytes(), repositories)
    return {"raw": raw, "original": original, "replay": replay, "held": held,
            "imported": set(sys.modules) - before}


def _seal(bundle):
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = digest(result)
        step["numerical_result"] = {"operation_id": workflow.operation, "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])


def test_source_retains_exact_native_bytes_and_bounded_scope():
    source = _source((EXAMPLES / "source.json").read_bytes())
    assert request_bytes(source) == (ROOT / "examples/adapters/circle.json").read_bytes()
    assert source["configuration"] == POLICY
    source["configuration"]["frame_authority"] = "surveyed"
    with pytest.raises(ValueError, match="explicit policy"):
        _source(canonical(source))


@pytest.mark.parametrize("payload", [b"[]", b'{"schema":"a","schema":"b"}', b"null"])
def test_invalid_source_refuses_before_provider_binding(payload):
    with pytest.raises(ValueError):
        workflow.create_session(payload, {"gte": "absent"})


def test_sample_budget_refuses_before_native_binding():
    source = _source((EXAMPLES / "source.json").read_bytes())
    declared = request(source)
    declared["observations"]["time_s"] = list(range(17))
    declared["observations"]["points_m"] = [[1.0, 0.0]] * 17
    source["request_bytes_b64"] = base64.b64encode(canonical(declared)).decode()
    with pytest.raises(ValueError, match="1..16"):
        workflow.create_session(canonical(source), {"gte": "absent"})


def test_native_covariance_projection_and_original_bytes(sessions):
    bundle = sessions["original"]
    assert workflow._validate(bundle) == sessions["raw"]
    source = _source(sessions["raw"])
    native = bundle["steps"][0]["result"]["data"]
    assert native["observed_points_m"] == request(source)["observations"]["points_m"]
    assert native["uncertainty"]["input_joint_covariance"] == request(source)["observations"]["covariance"]["matrix"]
    np.testing.assert_allclose(native["projected_points_m"], [[1, 0], [0, 1]], atol=1e-14)
    # For these two axis-aligned samples the independent tangent map is exact.
    tangent_map = np.array([[0, 1 / 1.01, 0, 0], [0, 0, -1 / .99, 0]])
    expected = tangent_map @ np.array(native["uncertainty"]["input_joint_covariance"]) @ tangent_map.T
    np.testing.assert_allclose(native["uncertainty"]["tangent_joint_covariance"], expected, atol=1e-16)
    assert expected[0, 1] != 0
    assert native["reconciliation"]["status"] == "eligible"
    assert bundle["verification"]["independent"] is False
    assert not any(name == "geodesic_telemetry" or name.startswith("geodesic_telemetry.") for name in sessions["imported"])


def test_replay_retains_evidence_and_fresh_occurrences(sessions):
    original, replay = sessions["original"], sessions["replay"]
    fresh = replay["session"]
    assert workflow._validate(fresh) == sessions["raw"]
    assert fresh["source"] == original["source"]
    occurrences = [item["execution_id"] for bundle in (original, fresh)
                   for item in (bundle["steps"][0], bundle["verification"]["reproduction"])]
    assert len(occurrences) == len(set(occurrences)) == 4
    assert fresh["steps"][0]["numerical_result"] == original["steps"][0]["numerical_result"]
    assert replay["replay_receipt"]["source_bundle_digest"] == original["bundle_digest"]
    assert replay["replay_receipt"]["verification"]["subject_ref"] == original["bundle_digest"]


def test_held_candidate_keeps_original_diagnostics_and_full_covariance(sessions):
    held = sessions["held"]["steps"][0]["result"]["data"]
    original = sessions["original"]["steps"][0]["result"]["data"]
    assert held["reconciliation"]["status"] == "held"
    assert held["reconciled_points_m"] is None
    for key in ("observed_points_m", "projected_points_m", "uncertainty", "diagnostics"):
        assert held[key] == original[key]
    assert held["diagnostics"]["radial_residual_before_m"] != [0, 0]


@pytest.mark.parametrize(("case", "code"), [
    ("stale", "constraint_not_valid"), ("singular", "numeric_geometry"),
    ("covariance", "input_covariance"), ("unknown-geometry", "unsupported_uncertainty"),
    ("frame-mismatch", "input_frame"),
])
def test_native_refusals_never_become_candidate_results(repositories, case, code):
    with pytest.raises(AdapterRefusal) as refused:
        workflow.create_session((EXAMPLES / (case + ".json")).read_bytes(), repositories)
    assert refused.value.code == code


def test_native_roundoff_tolerant_input_is_not_silently_repaired(repositories):
    source = _source((EXAMPLES / "source.json").read_bytes())
    declared = request(source)
    declared["observations"]["covariance"]["matrix"][0][3] += 1e-16
    source["request_bytes_b64"] = base64.b64encode(canonical(declared)).decode()
    with pytest.raises(ValueError, match="exactly symmetric"):
        workflow.create_session(canonical(source), repositories)


def test_source_tree_is_fixed_even_when_all_commitments_are_resealed(sessions):
    bundle = deepcopy(sessions["original"])
    bundle["runtimes"]["gte"]["source_tree"] = "f" * 40
    _seal(bundle)
    with pytest.raises(ValueError, match="source tree"):
        workflow._validate(bundle)


@pytest.mark.parametrize("fault", ["covariance", "time", "held", "scope", "raw"])
def test_resealed_payload_must_preserve_source_and_semantics(sessions, fault):
    bundle = deepcopy(sessions["original"])
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        data = step["result"]["data"]
        if fault == "covariance": data["uncertainty"]["tangent_joint_covariance"] = [[1, 2], [2, 1]]
        elif fault == "time": data["time_s"][1] = True
        elif fault == "held": data["reconciliation"]["status"] = "held"
        elif fault == "scope": data["uncertainty"]["limitations"] = []
        else: data["observed_points_m"][0][0] = 99
    _seal(bundle)
    with pytest.raises(ValueError):
        workflow._validate(bundle)


def test_read_only_view_preserves_native_bases_and_is_detached(sessions, monkeypatch):
    original = sessions["held"]
    snapshot = canonical(original)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection tried to bind a native provider")
    monkeypatch.setattr(workflow, "_adapters", forbidden)
    declaration = _source(workflow._validate(original))
    record = {"native": original, "kind": workflow.kind, "bundle_id": original["bundle_digest"], "upstream_bundle_id": None}
    source = {"source_id": "catalog-source", "evidence_id": original["source"]["evidence"][0]["artifact_ref"], "label": "Synthetic geometry"}
    view = project(record, source, declaration, 2)
    assert view["fusion_context"] is None
    assert view["object_context"]["uncertainty"] == original["steps"][0]["result"]["data"]["uncertainty"]
    assert not any(panel["panel_id"] == "reconciled_points_m" for panel in view["panels"])
    assert len(view["raw_observations"]) == 2
    view["object_context"]["uncertainty"]["tangent_bases"][0][0] = 99
    assert canonical(original) == snapshot
