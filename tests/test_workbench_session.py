"""Shared-session retention, scientific handoff and replay across real providers."""

import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from ciw import calibrated_observable as calibrated
from ciw import identified_design as design
from ciw.instruments import make_demo_run
from ciw.session import Session, read_json


ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_SOURCE = (ROOT / "examples/calibrated-observable/source.json").read_bytes()
DESIGN_SOURCE = (ROOT / "examples/identified-design/source.json").read_bytes()
CALIBRATED_OPERATION = "ciw.calibrated-observable.v1"
DESIGN_OPERATION = "ciw.identified-design.v1"


def request(session, kind, payload=None):
    return session.handle({
        "protocol_version": 1, "request_id": "shared-workbench-test",
        "type": kind, "payload": {} if payload is None else payload,
    })


def response(session, kind, payload=None):
    reply = request(session, kind, payload)
    assert reply["type"] == "response", reply
    return reply["payload"]


def source_payload(raw=CALIBRATED_SOURCE, kind="calibrated-observable"):
    return {"kind": kind, "label": "Two-channel retained process experiment",
            "bytes_b64": base64.b64encode(raw).decode("ascii")}


def add_source(session, raw=CALIBRATED_SOURCE, kind="calibrated-observable"):
    return response(session, "source.add", source_payload(raw, kind))


def execute(session, operation_id, source_id, **parameters):
    return response(session, "operation.execute", {
        "operation_id": operation_id,
        "parameters": {"source_id": source_id, **parameters},
    })


def native_data(bundle, role):
    step, = [step for step in bundle["steps"] if step["runtime_ref"] == role]
    return step["result"].get("data", step["result"])


def deny_scientific_execution(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Inspection/restore must not execute or bind a provider")

    for module in (calibrated, design):
        for name in ("create_session", "replay_session", "_adapters"):
            monkeypatch.setattr(module, name, denied)


@pytest.fixture(scope="module")
def calibrated_repositories():
    root = os.environ.get("CIW_CALIBRATED_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_CALIBRATED_STACK_ROOT to the eight pinned role checkouts")
    return {role: Path(root) / role for role in calibrated.ROLES}


@pytest.fixture(scope="module")
def design_repositories():
    root = os.environ.get("CIW_IDENTIFIED_DESIGN_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_IDENTIFIED_DESIGN_STACK_ROOT to the eleven pinned role checkouts")
    return {role: Path(root) / role for role in design.ROLES}


@pytest.fixture(scope="module")
def retained_process(tmp_path_factory, calibrated_repositories):
    directory = tmp_path_factory.mktemp("shared-process")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("calibrated-observable", calibrated_repositories)
    source = add_source(session)
    summary = execute(session, CALIBRATED_OPERATION, source["source_id"])
    bundle = response(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    legacy = response(session, "analysis.stats")
    path = session.save_workspace(directory / "workspace.json")
    return {"source": source, "summary": summary, "bundle": bundle,
            "legacy": legacy, "workspace": read_json(path), "path": path}


def test_source_bytes_and_device_timestamps_survive_shared_session(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    # Whitespace is evidence too: retention must not normalize input JSON.
    raw = b"\n " + CALIBRATED_SOURCE + b"\n\n"
    descriptor = add_source(session, raw)
    retained = response(session, "source.get", {"source_id": descriptor["source_id"]})
    assert base64.b64decode(retained["bytes_b64"], validate=True) == raw
    assert descriptor["byte_count"] == len(raw)
    original = json.loads(raw)
    assert [channel["observation"]["device_time"] for channel in original["channels"]] == [1005, 2004]
    assert [channel["observation"]["raw_value"] for channel in original["channels"]] == [51000, 45000]
    assert response(session, "source.list")["sources"] == [descriptor]
    assert response(session, "bundle.list")["bundles"] == []
    assert response(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []


@pytest.mark.parametrize("payload", [{}, {"bundle_id": []}, {"bundle_id": "missing"},
    {"bundle_id": "missing", "recompute": True}])
def test_experiment_inspection_refuses_unknown_or_mutating_request(tmp_path, payload):
    session = Session(make_demo_run(), tmp_path)
    before = session.workbench.serialize()
    assert request(session, "experiment.inspect", payload)["type"] == "error"
    assert session.workbench.serialize() == before


def test_process_experiment_projection_uses_retained_residuals(retained_process, tmp_path, monkeypatch):
    deny_scientific_execution(monkeypatch)
    session = Session.from_workspace(retained_process["path"], tmp_path)
    bundle = retained_process["bundle"]
    before = session.workbench.serialize()
    view = response(session, "experiment.inspect", {"bundle_id": bundle["bundle_digest"]})
    panels = {p["panel_id"]: p for p in view["panels"]}
    assert panels["measurements"]["values"] == [52, 46]
    assert panels["state"]["values"] == native_data(bundle, "gsie")["mean"]
    assert panels["innovation"]["values"] == [2, -4]
    assert panels["innovation"]["covariance"] == [[1.25, .25], [.25, 1.25]]
    assert panels["posterior-residual"]["covariance"] is None
    assert panels["reconciled"]["values"] == [50.75, 49.25]
    assert panels["reconciled"]["covariance"][0][1] < 0
    assert view["fusion_context"]["fault_assessment"]["isolability"] == native_data(bundle, "fdir")["isolability"]["status"]
    assert view["verification"] == bundle["verification"]
    assert session.workbench.serialize() == before
    # Projection behavior only: a held native candidate must never be plotted
    # as an accepted reconciled state, even if a candidate vector is present.
    from ciw.experiment_view import project
    record = deepcopy(before["bundles"][0])
    next(s for s in record["native"]["steps"] if s["runtime_ref"] == "cbsr")["result"]["data"]["status"] = "held"
    source = session.workbench.get_source(view["source_id"])
    held = project(record, source, json.loads(base64.b64decode(source["bytes_b64"])), view["fusion_context"], before["revision"])
    assert "reconciled" not in [p["panel_id"] for p in held["panels"]]


@pytest.mark.parametrize("mutate", [
    lambda payload: payload.update(kind="unknown-provider"),
    lambda payload: payload.update(kind="identified-design"),
    lambda payload: payload.update(bytes_b64="not valid base64!"),
    lambda payload: payload.update(bytes_b64=base64.b64encode(b'{"schema":1,"schema":2}').decode()),
    lambda payload: payload.update(repository="/client-selected/code"),
])
def test_invalid_source_never_enters_registry(tmp_path, mutate):
    session = Session(make_demo_run(), tmp_path)
    payload = source_payload()
    mutate(payload)
    before = session.snapshot()["workbench"]
    reply = request(session, "source.add", payload)
    assert reply["type"] == "error", reply
    assert session.snapshot()["workbench"] == before


def test_unbound_execution_cannot_promote_retained_source(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = add_source(session)
    reply = request(session, "operation.execute", {
        "operation_id": CALIBRATED_OPERATION,
        "parameters": {"source_id": source["source_id"]},
    })
    # Unbound is a retained refused execution with no result, never a bundle.
    assert reply["type"] == "response" and reply["payload"]["status"] == "refused", reply
    refused = reply["payload"]["execution"]
    assert refused["refusal"]["code"] == "operation_unavailable" and reply["payload"]["result"] is None
    assert refused in response(session, "execution.list")["executions"]
    assert response(session, "bundle.list")["bundles"] == []
    assert response(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []
    assert response(session, "source.list")["sources"] == [source]


def test_source_discovery_and_saved_workspace_do_not_alias_internal_state(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = add_source(session)
    returned = response(session, "source.get", {"source_id": source["source_id"]})
    original = deepcopy(returned)
    returned["bytes_b64"] = ""
    returned["label"] = "client mutation"
    listed = response(session, "source.list")
    listed["sources"][0]["label"] = "list mutation"
    snapshot = session.snapshot()
    snapshot["workbench"]["sources"].clear()
    assert response(session, "source.get", {"source_id": source["source_id"]}) == original
    assert response(session, "source.list")["sources"] == [source]
    saved = read_json(session.save_workspace(tmp_path / "saved.json"))
    assert saved["workspace_version"] == 3
    assert saved["workbench"]["sources"] == [original]


@pytest.mark.parametrize("field,value", [
    ("bytes_b64", base64.b64encode(CALIBRATED_SOURCE + b" ").decode()),
    ("kind", "unknown-provider"),
    ("evidence_id", "substituted-evidence"),
    ("byte_count", 1),
])
def test_corrupt_source_restore_refuses_before_any_write(tmp_path, monkeypatch, field, value):
    session = Session(make_demo_run(), tmp_path / "original")
    add_source(session)
    path = session.save_workspace(tmp_path / "saved.json")
    workspace = read_json(path)
    workspace["workbench"]["sources"][0][field] = value
    path.write_text(json.dumps(workspace), encoding="utf-8")
    deny_scientific_execution(monkeypatch)
    writes = []
    monkeypatch.setattr("ciw.session.write_json", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError):
        Session.from_workspace(path, tmp_path / "must-not-exist")
    assert writes == []
    assert not (tmp_path / "must-not-exist").exists()


def test_operations_expose_native_workflows_alongside_existing_instruments(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    operations = response(session, "operation.list")["operations"]
    ids = [operation["operation_id"] for operation in operations]
    assert {"statistics.v1", "spectrum.periodogram.v1",
            CALIBRATED_OPERATION, DESIGN_OPERATION} <= set(ids)
    assert len(ids) == len(set(ids))


def test_nested_parameter_cannot_override_requested_workflow(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    source = add_source(session)

    def denied(*args, **kwargs):
        raise AssertionError("Unexpected parameter must be rejected before workflow dispatch")

    monkeypatch.setattr(session.workbench, "execute", denied)
    reply = request(session, "operation.execute", {
        "operation_id": CALIBRATED_OPERATION,
        "parameters": {"source_id": source["source_id"], "operation_id": DESIGN_OPERATION},
    })
    assert reply["type"] == "error", reply
    assert reply["payload"]["code"] == "invalid_payload"


def test_unknown_calibration_crosscovariance_cannot_publish_partial_fusion(
        calibrated_repositories, tmp_path):
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("calibrated-observable", calibrated_repositories)
    experiment = json.loads(CALIBRATED_SOURCE)
    experiment["calibrated_covariance"]["cross_covariance_policy"] = "unknown"
    source = add_source(session, calibrated.canonical(experiment))
    reply = request(session, "operation.execute", {
        "operation_id": CALIBRATED_OPERATION,
        "parameters": {"source_id": source["source_id"]},
    })
    # The provider refusal is retained as a refused execution with its code; no bundle or fusion is published.
    assert reply["type"] == "response" and reply["payload"]["status"] == "refused", reply
    assert reply["payload"]["result"] is None
    assert reply["payload"]["execution"]["refusal"]["code"] == "CALIBRATED_MCUR_REFUSED"
    assert response(session, "source.list")["sources"] == [source]
    assert response(session, "bundle.list")["bundles"] == []
    assert response(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []
    assert response(session, "result.list")["results"] == []


def test_one_session_discovers_exact_native_results_and_legacy_analysis(retained_process, tmp_path):
    session = Session.from_workspace(retained_process["path"], tmp_path)
    bundle = retained_process["bundle"]
    summary = retained_process["summary"]
    assert summary["source_id"] == retained_process["source"]["source_id"]
    assert summary["operation_id"] == CALIBRATED_OPERATION
    assert summary["verification_id"] == bundle["verification"]["verification_id"]
    assert summary["retained_verification_outcome"] == "passed"
    assert summary["validation"] == "content_consistent"
    listed_ids = {item["result_id"] for item in response(session, "result.list")["results"]}
    assert set(summary["result_ids"]) <= listed_ids
    assert retained_process["legacy"]["result_id"] in listed_ids
    for step in bundle["steps"]:
        assert response(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    executions = response(session, "execution.list")["executions"]
    assert set(summary["execution_ids"]) <= {item["execution_id"] for item in executions}
    assert native_data(bundle, "gsie")["mean"] == pytest.approx([607 / 12, 589 / 12])
    assert native_data(bundle, "fdir")["detection"]["innovation_covariance"] == native_data(bundle, "gsie")["innovation_covariance"]


def test_fusion_context_exposes_retained_gsie_authority_and_gate_results(retained_process, tmp_path):
    session = Session.from_workspace(retained_process["path"], tmp_path)
    context, = response(session, "experiment.inspect", {"view": "fusion"})["contexts"]
    bundle = retained_process["bundle"]
    state = native_data(bundle, "gsie")
    assert context["owner"] == "gsie"
    assert context["state_kind"] == "posterior"
    assert context["source_id"] == retained_process["source"]["source_id"]
    assert context["bundle_id"] == retained_process["summary"]["bundle_id"]
    for key in ("mean", "covariance", "frame_id", "units", "state_id"):
        assert context[key] == state[key]
    assert context["event_time"] == state["time"] == 5
    assert context["channel_ids"] == ["tank-1.mass", "tank-2.mass"]
    assert context["observability"]["status"] == "observable"
    assert context["observability"]["rank"] == 2
    assert context["observability"]["condition_number"] == pytest.approx(1)
    assert context["reconciliation"]["status"] == "accepted"
    assert context["fault_assessment"]["detection"] == "statistical_anomaly"
    assert context["fault_assessment"]["isolability"] == "isolated"
    assert context["retained_verification_outcome"] == "passed"
    assert context["state_admission"] == "not_performed"
    # CBSR's constrained posterior is distinct from the GSIE state owner.
    assert context["mean"] != native_data(bundle, "cbsr")["reconciled"]["estimate"]
    context["mean"][0] = -999
    context["covariance"][0][0] = -999
    fresh, = response(session, "experiment.inspect", {"view": "fusion"})["contexts"]
    assert fresh["mean"] == state["mean"]
    assert fresh["covariance"] == state["covariance"]


def test_retained_workspace_reopens_without_code_binding_or_provider_execution(
        retained_process, tmp_path, monkeypatch):
    deny_scientific_execution(monkeypatch)
    session = Session.from_workspace(retained_process["path"], tmp_path)
    assert response(session, "bundle.get", {
        "bundle_id": retained_process["summary"]["bundle_id"],
    }) == retained_process["bundle"]
    operations = response(session, "operation.list")["operations"]
    native_operations = [operation for operation in operations
                         if operation["operation_id"] in {CALIBRATED_OPERATION, DESIGN_OPERATION}]
    assert all(operation["available"] is False for operation in native_operations)
    path = session.save_workspace(tmp_path / "resaved.json")
    assert read_json(path)["workbench"] == retained_process["workspace"]["workbench"]
    reply = request(session, "bundle.replay", {
        "bundle_id": retained_process["summary"]["bundle_id"],
    })
    # Reopening binds no provider, so replay is a refused replay naming its subject.
    assert reply["type"] == "response" and reply["payload"]["status"] == "refused", reply
    assert reply["payload"]["execution"]["subject_bundle_id"] == retained_process["summary"]["bundle_id"]
    assert len(response(session, "bundle.list")["bundles"]) == 1


def test_native_bundle_and_result_retrieval_are_detached(retained_process, tmp_path):
    session = Session.from_workspace(retained_process["path"], tmp_path)
    bundle_id = retained_process["summary"]["bundle_id"]
    bundle = response(session, "bundle.get", {"bundle_id": bundle_id})
    step = next(step for step in bundle["steps"] if step["runtime_ref"] == "gsie")
    result = response(session, "result.get", {"result_id": step["result_id"]})
    result["data"]["mean"][0] = -999
    step["result"]["data"]["mean"][0] = -777
    bundle["source"]["evidence"].clear()
    assert response(session, "bundle.get", {"bundle_id": bundle_id}) == retained_process["bundle"]
    assert response(session, "result.get", {"result_id": step["result_id"]})["data"]["mean"] == pytest.approx([607 / 12, 589 / 12])


@pytest.mark.parametrize("mutate", [
    lambda record: record.update(source_id="absent-source"),
    lambda record: record.update(kind="identified-design"),
    lambda record: record["native"]["steps"][4]["result"]["data"]["mean"].__setitem__(0, 999),
])
def test_corrupt_native_bundle_restore_refuses_before_any_write(
        retained_process, tmp_path, monkeypatch, mutate):
    workspace = deepcopy(retained_process["workspace"])
    mutate(workspace["workbench"]["bundles"][0])
    path = tmp_path / "corrupted.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    deny_scientific_execution(monkeypatch)
    writes = []
    monkeypatch.setattr("ciw.session.write_json", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError):
        Session.from_workspace(path, tmp_path / "must-not-exist")
    assert writes == []
    assert not (tmp_path / "must-not-exist").exists()


def test_legacy_and_native_execution_ids_cannot_alias_on_restore(
        retained_process, tmp_path, monkeypatch):
    workspace = deepcopy(retained_process["workspace"])
    workspace["results"][0]["execution_id"] = workspace["workbench"]["bundles"][0]["native"]["steps"][0]["execution_id"]
    path = tmp_path / "cross-registry-alias.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    deny_scientific_execution(monkeypatch)
    writes = []
    monkeypatch.setattr("ciw.session.write_json", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError):
        Session.from_workspace(path, tmp_path / "must-not-exist")
    assert writes == []
    assert not (tmp_path / "must-not-exist").exists()


def test_explicit_replay_retains_original_and_adds_fresh_occurrences(
        retained_process, calibrated_repositories, tmp_path):
    session = Session.from_workspace(retained_process["path"], tmp_path)
    session.workbench.bind_workflow("calibrated-observable", calibrated_repositories)
    original = retained_process["bundle"]
    original_id = retained_process["summary"]["bundle_id"]
    replay = response(session, "bundle.replay", {"bundle_id": original_id})
    summary = replay["bundle"]
    fresh = response(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    assert summary["bundle_id"] != original_id
    assert summary["source_id"] == retained_process["source"]["source_id"]
    assert fresh["session_id"] != original["session_id"]
    assert fresh["verification"]["verification_id"] != original["verification"]["verification_id"]
    assert replay["replay_receipt"]["numerical_match"] is True
    assert replay["replay_receipt"]["admission"] == "not_performed"
    for before, after in zip(original["steps"], fresh["steps"]):
        assert before["numerical_result_id"] == after["numerical_result_id"]
        assert before["execution_id"] != after["execution_id"]
        assert before["result_id"] != after["result_id"]
    assert response(session, "bundle.get", {"bundle_id": original_id}) == original
    assert len(response(session, "bundle.list")["bundles"]) == 2
    # Replaying the same evidence must never act like a second independent sensor.
    contexts = response(session, "experiment.inspect", {"view": "fusion"})["contexts"]
    assert len(contexts) == 2
    assert len({context["context_id"] for context in contexts}) == 2
    assert {context["source_id"] for context in contexts} == {summary["source_id"]}
    for context in contexts:
        assert context["mean"] == native_data(original, "gsie")["mean"]
        assert context["covariance"] == native_data(original, "gsie")["covariance"]
        assert context["state_admission"] == "not_performed"


def test_design_consumes_retained_upstream_bundle_without_an_export_handoff(
        retained_process, design_repositories, tmp_path):
    session = Session.from_workspace(retained_process["path"], tmp_path)
    session.workbench.bind_workflow("identified-design", design_repositories)
    declaration = add_source(session, DESIGN_SOURCE, "identified-design")
    upstream_id = retained_process["summary"]["bundle_id"]
    summary = execute(session, DESIGN_OPERATION, declaration["source_id"], upstream_bundle_id=upstream_id)
    bundle = response(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    assert summary["upstream_bundle_id"] == upstream_id
    assert summary["source_id"] == declaration["source_id"]
    assert bundle["upstream"] == retained_process["bundle"]
    assert native_data(bundle, "gsie")["predecessor_state_id"] == native_data(bundle["upstream"], "gsie")["state_id"]
    assert native_data(bundle, "edspt")["selected_candidate_id"] == "sensor:tank-2:precise"
    assert bundle["decision"]["state_admission"] == "not_performed"
    assert bundle["decision"]["acquisition"] == "not_performed"
    assert bundle["decision"]["uncertainty_scope"] == "conditional_on_identified_point_model"
    contexts = response(session, "experiment.inspect", {"view": "fusion"})["contexts"]
    assert len(contexts) == 2
    posterior, = [context for context in contexts if context["state_kind"] == "posterior"]
    prediction, = [context for context in contexts if context["state_kind"] == "conditional_prediction"]
    assert prediction["upstream_bundle_id"] == posterior["bundle_id"]
    assert prediction["predecessor_state_id"] == posterior["state_id"]
    assert prediction["mean"] == native_data(bundle, "gsie")["mean"]
    assert prediction["covariance"] == native_data(bundle, "gsie")["covariance"]
    assert prediction["parameter_covariance_status"] == "unknown"
    assert prediction["uncertainty_scope"] == "conditional_on_identified_point_model"
    assert prediction["state_admission"] == "not_performed"
    view = response(session, "experiment.inspect", {"bundle_id": summary["bundle_id"]})
    assert view["panels"][0]["context"]["uncertainty_scope"] == "conditional_on_identified_point_model"
    assert view["fusion_context"]["parameter_covariance_status"] == "unknown"
    assert view["upstream_bundle_id"] == posterior["bundle_id"]
    for step in bundle["steps"] + bundle["upstream_replay"]["session"]["steps"]:
        assert response(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    saved = session.save_workspace(tmp_path / "with-design.json")
    reopened = Session.from_workspace(saved, tmp_path / "reopened")
    assert response(reopened, "bundle.get", {"bundle_id": summary["bundle_id"]}) == bundle


@pytest.mark.parametrize("parameters", [
    {},
    {"upstream_bundle_id": "missing-bundle"},
    {"upstream_path": "/client-chosen/upstream.json"},
    {"repositories": {"gsie": "/client-chosen/code"}},
])
def test_design_cannot_select_unretained_upstream_or_client_code(tmp_path, parameters):
    session = Session(make_demo_run(), tmp_path)
    source = add_source(session, DESIGN_SOURCE, "identified-design")
    reply = request(session, "operation.execute", {
        "operation_id": DESIGN_OPERATION,
        "parameters": {"source_id": source["source_id"], **parameters},
    })
    # A malformed or client-chosen upstream is rejected before any execution exists.
    assert reply["type"] == "error", reply
    assert response(session, "execution.list")["executions"] == []
    assert response(session, "bundle.list")["bundles"] == []
    assert response(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []


def test_bound_design_rejects_unknown_upstream_before_calling_provider(
        design_repositories, tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("identified-design", design_repositories)
    declaration = add_source(session, DESIGN_SOURCE, "identified-design")
    deny_scientific_execution(monkeypatch)
    reply = request(session, "operation.execute", {
        "operation_id": DESIGN_OPERATION,
        "parameters": {"source_id": declaration["source_id"], "upstream_bundle_id": "absent-bundle"},
    })
    # Unbound is a retained refused execution with no result, never a bundle.
    assert reply["type"] == "response" and reply["payload"]["status"] == "refused", reply
    refused = reply["payload"]["execution"]
    assert refused["refusal"]["code"] == "operation_unavailable" and reply["payload"]["result"] is None
    assert refused in response(session, "execution.list")["executions"]
    assert response(session, "bundle.list")["bundles"] == []
