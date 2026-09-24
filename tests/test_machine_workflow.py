"""Shared CIW lifecycle coverage for the provider-free machine manifest."""

import base64
from copy import deepcopy

import pytest

from ciw import machine_manifest as manifest
from ciw import machine_workflow
from ciw.instruments import make_demo_run
from ciw.session import Session

from test_machine_manifest import _fixture


def _source():
    evidence, candidate, report = _fixture()
    return {
        "schema": machine_workflow.SOURCE_SCHEMA,
        "experiment_id": "machine:workflow-fixture",
        "configuration": deepcopy(machine_workflow.CONFIGURATION),
        "evidence_bundle": evidence,
        "candidate_manifest": candidate,
        "challenge_report": report,
        "request": {"counts": 300, "covariance": None},
    }


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind,
                               "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _session_with_source(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = manifest.canonical(_source())
    descriptor = _call(session, "source.add", {"kind": "machine-manifest", "label": "machine fixture",
                                                "bytes_b64": base64.b64encode(raw).decode()})
    return session, descriptor


def test_machine_operation_save_reopen_and_replay_preserve_identities(tmp_path):
    session, source = _session_with_source(tmp_path / "original")
    operation = next(item for item in _call(session, "operation.list", {})["operations"]
                     if item["operation_id"] == "ciw.encoder-position.v1")
    assert operation["available"] is True
    completed = _call(session, "operation.execute", {"operation_id": operation["operation_id"],
                                                      "parameters": {"source_id": source["source_id"]}})
    original = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    step = original["steps"][0]
    assert step["operation_id"] == "ciw.encoder-position.v1"
    assert step["result"]["data"]["position"]["position"] == pytest.approx(1.001)
    assert original["verification"]["authority"]["state_admission"] == "not_performed"
    assert original["runtimes"]["machine"]["execution_scope"] == "independent_python_reference_only"
    listed = _call(session, "result.list", {})["results"]
    assert any(item["result_id"] == step["result_id"] for item in listed)
    assert _call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["physical_validation"] == "not_performed"
    assert view["panels"][0]["panel_id"] == "position"

    saved = session.save_workspace(tmp_path / "saved.json")
    original_create = machine_workflow.MachineManifestWorkflow.create_session
    machine_workflow.MachineManifestWorkflow.create_session = lambda *args, **kwargs: pytest.fail("restore executed a provider")
    try:
        reopened = Session.from_workspace(saved, tmp_path / "reopened")
    finally:
        machine_workflow.MachineManifestWorkflow.create_session = original_create
    assert _call(reopened, "bundle.get", {"bundle_id": completed["bundle_id"]}) == original

    replay = _call(reopened, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    fresh = _call(reopened, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    fresh_step = fresh["steps"][0]
    assert fresh["bundle_digest"] != original["bundle_digest"]
    assert fresh_step["execution_id"] != step["execution_id"]
    assert fresh_step["result_id"] != step["result_id"]
    assert fresh_step["numerical_result_id"] == step["numerical_result_id"]
    assert replay["replay_receipt"]["admission"] == "not_performed"
    assert replay["replay_receipt"]["numerical_match"] is True


def test_machine_replay_refuses_changed_reference_identity(tmp_path, monkeypatch):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.encoder-position.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    old_identity = machine_workflow.runtime_identity

    def changed_identity():
        value = deepcopy(old_identity())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(machine_workflow, "runtime_identity", changed_identity)
    response = session.handle({"protocol_version": 1, "request_id": "replay",
                               "type": "bundle.replay", "payload": {"bundle_id": completed["bundle_id"]}})
    assert response["type"] == "error"
    assert "runtime identity" in response["payload"]["message"]


def test_machine_source_tampering_refuses_before_retention(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = manifest.canonical(_source())
    encoded = base64.b64encode(raw).decode()
    source = _call(session, "source.add", {"kind": "machine-manifest", "label": "machine fixture",
                                             "bytes_b64": encoded})
    assert source["source_id"]
    retained = session.workbench.serialize()
    retained["sources"][0]["bytes_b64"] = base64.b64encode(raw + b" ").decode()
    with pytest.raises(ValueError):
        session.workbench.restore(retained)


def test_machine_source_near_its_byte_budget_still_executes_and_replays(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = manifest.canonical(_source())
    padded = raw[:-1] + b" " * (900 * 1024 - len(raw)) + raw[-1:]
    assert len(raw) < len(padded) <= machine_workflow.SOURCE_LIMIT
    descriptor = _call(session, "source.add", {"kind": "machine-manifest", "label": "padded machine fixture",
                                                "bytes_b64": base64.b64encode(padded).decode()})
    completed = _call(session, "operation.execute", {"operation_id": "ciw.encoder-position.v1",
                                                      "parameters": {"source_id": descriptor["source_id"]}})
    bundle = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    assert base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"]) == padded
    assert bundle["steps"][0]["result"]["data"]["position"]["position"] == pytest.approx(1.001)
    replay = _call(session, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    assert replay["replay_receipt"]["numerical_match"] is True
