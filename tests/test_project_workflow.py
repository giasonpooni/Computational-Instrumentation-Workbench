"""Shared CIW lifecycle coverage for the provider-free project graph operation."""

import base64
from copy import deepcopy
import json

import pytest

from ciw import project_model as project
from ciw import project_workflow
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import canonical


def _project():
    value = project.create("project:workflow-fixture", "Workflow fixture",
                           {"place": {"value": None, "evidence_refs": []}})
    value = project.put(value, {"object_id": "evidence:manual", "kind": "evidence", "label": "Manual", "content": {
        "status": "documented", "source_digest": "sha256:" + "1" * 64,
        "locator": "fixture://manual", "subject_id": "machine:fixture"}})
    value = project.put(value, {"object_id": "signal:position", "kind": "signal", "label": "Position", "content": {
        "quantity": "position", "unit": "m", "frame": "frame:carriage",
        "time_basis": "clock:utc", "semantics": "observed"}})
    value = project.put(value, {"object_id": "computation:position", "kind": "computation", "label": "Position model",
                                "content": {"operation": "ciw.encoder-position.v1", "parameters": {"mode": "declared"}}})
    objects = {item["object_id"]: item for item in project.inspect(value)["objects"]}
    value = project.put(value, {"object_id": "result:position", "kind": "result", "label": "Position estimate", "content": {
        "value": 1.0, "input_revisions": {
            "signal:position": objects["signal:position"]["revision"],
            "computation:position": objects["computation:position"]["revision"]},
        "unit": "m", "frame": "frame:carriage", "time_basis": "clock:utc", "semantics": "estimated"}})
    value = project.connect(value, {"edge_id": "edge:model-result", "relation": "computation",
                                    "from": "computation:position", "to": "result:position", "resolution": "resolved"})
    value = project.connect(value, {"edge_id": "edge:evidence-result", "relation": "evidence",
                                    "from": "evidence:manual", "to": "result:position", "resolution": "resolved"})
    return value


def _source(value=None):
    value = _project() if value is None else value
    return {"schema": project_workflow.SOURCE_SCHEMA, "experiment_id": "project:workflow-fixture",
            "configuration": deepcopy(project_workflow.CONFIGURATION), "project": value,
            "request": {"expected_revision": value["revision"]}}


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind,
                               "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _refused(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "refused-" + kind,
                               "type": kind, "payload": payload})
    assert response["type"] == "error", response
    return response["payload"]["message"]


def _add(session, source):
    raw = canonical(source)
    return _call(session, "source.add", {"kind": "project-graph", "label": "project fixture",
                                         "bytes_b64": base64.b64encode(raw).decode()})


def test_project_operation_save_reopen_and_replay_preserve_identities(tmp_path):
    session = Session(make_demo_run(), tmp_path / "original")
    source = _add(session, _source())
    operation = next(item for item in _call(session, "operation.list", {})["operations"]
                     if item["operation_id"] == "ciw.project-graph.v1")
    assert operation["available"] is True
    assert operation["role"] == "project_graph_inspection"
    completed = _call(session, "operation.execute", {"operation_id": operation["operation_id"],
                                                      "parameters": {"source_id": source["source_id"]}})
    original = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    step = original["steps"][0]
    data = step["result"]["data"]
    assert step["operation_id"] == "ciw.project-graph.v1"
    assert step["runtime_ref"] == "project"
    assert data["summary"]["status"] == "declared"
    assert data["summary"]["object_counts"]["signal"] == 1
    assert data["summary"]["result_statuses"] == {"current_for_declared_inputs": 1, "needs_reevaluation": 0}
    assert data["summary"]["edge_counts"] == {"physical": 0, "computation": 1, "evidence": 1}
    assert data["inspection"] == project.inspect(_project())
    assert data["inspection"]["authority"]["execution"] == "not_performed"
    assert data["authority"]["declared_computation_execution"] == "not_performed"
    assert original["verification"]["authority"]["state_admission"] == "not_performed"
    assert original["runtimes"]["project"]["execution_scope"] == "independent_python_reference_only"
    listed = _call(session, "result.list", {})["results"]
    assert any(item["result_id"] == step["result_id"] for item in listed)
    assert _call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    executions = _call(session, "execution.list", {})["executions"]
    assert any(item["execution_id"] == step["execution_id"] and item["runtime_ref"] == "project"
               for item in executions)
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["object_kind"] == "project_graph_declaration"
    assert view["object_context"]["declared_computation_execution"] == "not_performed"
    assert view["object_context"]["status"] == "declared"
    assert view["panels"][0]["panel_id"] == "object-counts"
    assert view["panels"][0]["labels"] == sorted(project.KINDS)
    assert view["authority"]["read_only"] is True

    saved = session.save_workspace(tmp_path / "saved.json")
    original_create = project_workflow.ProjectGraphWorkflow.create_session
    project_workflow.ProjectGraphWorkflow.create_session = lambda *args, **kwargs: pytest.fail("restore executed a provider")
    try:
        reopened = Session.from_workspace(saved, tmp_path / "reopened")
    finally:
        project_workflow.ProjectGraphWorkflow.create_session = original_create
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
    assert replay["replay_receipt"]["verification"]["subject_ref"] == original["bundle_digest"]


def test_project_stale_result_is_reported_without_executing_anything(tmp_path):
    value = _project()
    signal = next(item for item in project.inspect(value)["objects"] if item["object_id"] == "signal:position")
    changed = {"object_id": "signal:position", "kind": "signal", "label": "Position", "content": {
        "quantity": "position", "unit": "m", "frame": "frame:inspection",
        "time_basis": "clock:utc", "semantics": "observed"}}
    value = project.put(value, changed, expected_revision=signal["revision"])
    value = project.connect(value, {"edge_id": "edge:unresolved", "relation": "physical",
                                    "from": "component:missing", "to": "signal:position", "resolution": "unresolved"})
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source(value))
    completed = _call(session, "operation.execute", {"operation_id": "ciw.project-graph.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    data = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})["steps"][0]["result"]["data"]
    assert data["summary"]["status"] == "draft"
    assert data["summary"]["result_statuses"] == {"current_for_declared_inputs": 0, "needs_reevaluation": 1}
    assert data["summary"]["unresolved_physical_edges"] == 1
    assert data["inspection"]["unresolved_physical_edges"][0]["missing_refs"] == ["component:missing"]
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["status"] == "draft"
    assert view["object_context"]["result_statuses"]["needs_reevaluation"] == 1


def test_project_source_refuses_stale_revision_and_broken_history_before_retention(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    stale = _source()
    stale["request"]["expected_revision"] = "sha256:" + "0" * 64
    assert "requested revision" in _refused(session, "source.add", {
        "kind": "project-graph", "label": "stale", "bytes_b64": base64.b64encode(canonical(stale)).decode()})
    broken = _source()
    broken["project"]["history"][1]["payload"]["label"] = "Edited after sealing"
    assert "digest mismatch" in _refused(session, "source.add", {
        "kind": "project-graph", "label": "broken", "bytes_b64": base64.b64encode(canonical(broken)).decode()})
    executing = _source()
    executing["configuration"]["declared_computation_execution"] = "performed"
    assert "authority policy" in _refused(session, "source.add", {
        "kind": "project-graph", "label": "executing", "bytes_b64": base64.b64encode(canonical(executing)).decode()})
    assert _call(session, "source.list", {})["sources"] == []


def test_project_replay_refuses_changed_reference_identity(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source())
    completed = _call(session, "operation.execute", {"operation_id": "ciw.project-graph.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    old_identity = project_workflow.runtime_identity

    def changed_identity():
        value = deepcopy(old_identity())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(project_workflow, "runtime_identity", changed_identity)
    assert "runtime identity" in _refused(session, "bundle.replay", {"bundle_id": completed["bundle_id"]})


def test_project_retained_records_refuse_tampering_before_any_restore(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source())
    completed = _call(session, "operation.execute", {"operation_id": "ciw.project-graph.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    retained = session.workbench.serialize()
    assert retained["bundles"][0]["bundle_id"] == completed["bundle_id"]
    tampered_source = deepcopy(retained)
    raw = base64.b64decode(retained["sources"][0]["bytes_b64"])
    tampered_source["sources"][0]["bytes_b64"] = base64.b64encode(raw + b" ").decode()
    with pytest.raises(ValueError):
        session.workbench.restore(tampered_source)
    tampered_result = deepcopy(retained)
    data = tampered_result["bundles"][0]["native"]["steps"][0]["result"]["data"]
    assert data["summary"]["result_statuses"]["needs_reevaluation"] == 0
    data["summary"]["result_statuses"]["needs_reevaluation"] = 1
    data["inspection"]["objects"][-1]["result_status"] = "needs_reevaluation"
    with pytest.raises(ValueError):
        session.workbench.restore(tampered_result)
    assert json.loads(json.dumps(retained)) == retained
    session.workbench.restore(retained)


def test_project_reproduction_occurrences_are_claimed_identities(tmp_path):
    from ciw import workbench as catalog
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source())
    completed = _call(session, "operation.execute", {"operation_id": "ciw.project-graph.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    record = session.workbench._bundles[completed["bundle_id"]]
    claims = catalog._claims(record)
    reproduction = record["native"]["verification"]["reproduction"]
    assert claims[reproduction["execution_id"]][0] == "execution"
    assert claims[reproduction["result_id"]][0] == "result"
    assert claims[reproduction["numerical_result_id"]][0] == "numerical_result"
    assert reproduction["execution_id"] != record["native"]["steps"][0]["execution_id"]
