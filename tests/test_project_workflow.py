"""Project-graph compilation through the shared operation, execution and result envelopes."""

import base64
from copy import deepcopy

import pytest

from ciw import project_model as project
from ciw import project_workflow
from ciw.instruments import make_demo_run
from ciw.session import Session

from test_project_model import _computation, _evidence, _signal
from test_thermal_workflow import _request as _thermal_request


def _project():
    value = project.create("project:compile", "Compile fixture", {"place": {"value": None, "evidence_refs": []}})
    value = project.put(value, _evidence())
    value = project.put(value, _signal())
    value = project.put(value, _computation())
    value = project.put(value, {"object_id": "model:thermal", "kind": "thermal-model", "label": "Thermal model",
                                "content": _thermal_request()["model"]})
    objects = {item["object_id"]: item for item in project.inspect(value)["objects"]}
    value = project.put(value, {"object_id": "result:position", "kind": "result", "label": "Position", "content": {
        "value": 1.0, "input_revisions": {
            "signal:position": objects["signal:position"]["revision"],
            "computation:position": objects["computation:position"]["revision"]},
        "unit": "m", "frame": "frame:carriage", "time_basis": "clock:utc", "semantics": "estimated"}})
    value = project.connect(value, {"edge_id": "edge:signal-model", "relation": "computation",
                                    "from": "signal:position", "to": "computation:position", "resolution": "resolved"})
    value = project.connect(value, {"edge_id": "edge:model-result", "relation": "computation",
                                    "from": "computation:position", "to": "result:position", "resolution": "resolved"})
    value = project.connect(value, {"edge_id": "edge:evidence", "relation": "evidence",
                                    "from": "evidence:manual", "to": "result:position", "resolution": "resolved"})
    return value


def _source(value=None, targets=None):
    value = value or _project()
    return {"schema": project_workflow.SOURCE_SCHEMA, "experiment_id": "project:compile-fixture",
            "configuration": deepcopy(project_workflow.CONFIGURATION), "project": value,
            "request": {"project_revision": value["revision"], "targets": targets}}


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind, "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _add(session, source):
    raw = project_workflow.canonical(source)
    return _call(session, "source.add", {"kind": "project-graph", "label": "project fixture",
                                         "bytes_b64": base64.b64encode(raw).decode()})


def test_compiled_plan_orders_computations_and_reports_typed_model_checks():
    plan = project_workflow.compile_plan(_source())
    assert plan["order"].index("signal:position") < plan["order"].index("computation:position") < plan["order"].index("result:position")
    step = next(item for item in plan["steps"] if item["object_id"] == "computation:position")
    assert step["declared_operation"] == "ciw.encoder-position.v1" and step["dispatch"] == "not_performed"
    result = next(item for item in plan["steps"] if item["object_id"] == "result:position")
    assert result["result_status"] == "current_for_declared_inputs" and result["drifted_inputs"] == []
    assert plan["needs_reevaluation"] == [] and plan["unevidenced_results"] == []
    assert plan["model_checks"] == [{"object_id": "model:thermal", "revision": plan["steps"][plan["order"].index("model:thermal")]["revision"],
                                     "contract": "ciw.thermal-observer-request.v1#model", "status": "contract_valid", "reason": None}]
    assert plan["authority"]["execution"] == "not_performed"
    assert project_workflow.compile_plan(_source()) == plan


def test_drifted_input_and_invalid_model_are_findings_not_refusals():
    value = _project()
    signal_revision = next(item for item in project.inspect(value)["objects"] if item["object_id"] == "signal:position")["revision"]
    changed = _signal()
    changed["content"]["frame"] = "frame:inspection"
    value = project.put(value, changed, expected_revision=signal_revision)
    broken = {"object_id": "model:thermal", "kind": "thermal-model", "label": "Thermal model",
              "content": {**_thermal_request()["model"], "state_units": ["degC", "degC"]}}
    value = project.put(value, broken)
    plan = project_workflow.compile_plan(project_workflow.validate_source(project_workflow.canonical(_source(value))))
    assert plan["needs_reevaluation"] == ["result:position"]
    result = next(item for item in plan["steps"] if item["object_id"] == "result:position")
    assert result["drifted_inputs"] == ["signal:position"]
    assert plan["model_checks"][0]["status"] == "contract_invalid"


def test_targets_restrict_the_plan_to_declared_ancestors():
    plan = project_workflow.compile_plan(_source(targets=["computation:position"]))
    assert plan["order"] == ["signal:position", "computation:position"]
    with pytest.raises(ValueError, match="not a declared project object"):
        project_workflow.validate_source(project_workflow.canonical(_source(targets=["result:missing"])))


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["request"].update(project_revision="sha256:" + "0" * 64), "retained project revision"),
    (lambda s: s["configuration"].update(execution="performed"), "authority policy"),
    (lambda s: s["project"]["history"][1]["payload"].update(label="Edited"), "digest mismatch"),
])
def test_source_refusals_retain_nothing(tmp_path, mutate, message):
    source = _source()
    mutate(source)
    with pytest.raises(ValueError, match=message):
        project_workflow.validate_source(project_workflow.canonical(source))
    session = Session(make_demo_run(), tmp_path)
    raw = project_workflow.canonical(source)
    response = session.handle({"protocol_version": 1, "request_id": "refuse", "type": "source.add",
                               "payload": {"kind": "project-graph", "label": "bad", "bytes_b64": base64.b64encode(raw).decode()}})
    assert response["type"] == "error"
    assert _call(session, "source.list", {})["sources"] == []


def test_project_operation_save_reopen_and_replay_preserve_identities(tmp_path):
    session = Session(make_demo_run(), tmp_path / "original")
    source = _add(session, _source())
    operation = next(item for item in _call(session, "operation.list", {})["operations"]
                     if item["operation_id"] == project_workflow.OPERATION)
    assert operation["available"] is True and operation["role"] == "project_graph_compiler"
    completed = _call(session, "operation.execute", {"operation_id": project_workflow.OPERATION,
                                                      "parameters": {"source_id": source["source_id"]}})
    original = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    step = original["steps"][0]
    assert step["operation_id"] == project_workflow.OPERATION
    assert step["result"]["data"]["schema"] == project_workflow.DATA_SCHEMA
    assert original["verification"]["independent"] is False
    assert original["runtimes"]["project"]["execution_scope"] == "independent_python_reference_only"
    assert _call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["object_kind"] == "project_graph_plan"
    assert view["object_context"]["execution"] == "not_performed"

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
    assert fresh["steps"][0]["execution_id"] != step["execution_id"]
    assert fresh["steps"][0]["result_id"] != step["result_id"]
    assert fresh["steps"][0]["numerical_result_id"] == step["numerical_result_id"]
    assert replay["replay_receipt"]["admission"] == "not_performed"
    resaved = reopened.save_workspace(tmp_path / "resaved.json")
    assert len(_call(Session.from_workspace(resaved, tmp_path / "again"), "bundle.list", {})["bundles"]) == 2


def test_project_replay_refuses_a_changed_reference_identity(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source())
    completed = _call(session, "operation.execute", {"operation_id": project_workflow.OPERATION,
                                                      "parameters": {"source_id": source["source_id"]}})
    original = project_workflow.ProjectGraphWorkflow.runtime_identity

    def changed(self):
        value = original(self)
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(project_workflow.ProjectGraphWorkflow, "runtime_identity", changed)
    response = session.handle({"protocol_version": 1, "request_id": "replay", "type": "bundle.replay",
                               "payload": {"bundle_id": completed["bundle_id"]}})
    assert response["type"] == "error" and "runtime identity" in response["payload"]["message"]


def test_retained_project_result_tampering_is_refused_on_restore(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = _add(session, _source())
    _call(session, "operation.execute", {"operation_id": project_workflow.OPERATION,
                                         "parameters": {"source_id": source["source_id"]}})
    retained = session.workbench.serialize()
    retained["bundles"][0]["native"]["steps"][0]["result"]["data"]["order"].reverse()
    with pytest.raises(ValueError):
        session.workbench.restore(retained)
