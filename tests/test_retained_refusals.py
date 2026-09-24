"""Unbound, unsupported or failed workbench operations are retained as refused executions."""
from copy import deepcopy
import json

import pytest

from ciw import workbench as wb
from ciw.core.canonical import digest
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.workbench import REFUSAL_SCHEMA, Workbench

from test_thermal_workflow import _call, _session_with_source

CALIBRATED = "ciw.calibrated-observable.v1"


def _calibrated_source(session):
    from test_workbench_session import add_source
    return add_source(session)


def _refuse(session, source):
    reply = session.handle({"protocol_version": 1, "request_id": "r", "type": "operation.execute",
                            "payload": {"operation_id": CALIBRATED, "parameters": {"source_id": source["source_id"]}}})
    assert reply["type"] == "response" and reply["payload"]["status"] == "refused", reply
    return reply["payload"]["execution"]


def test_refused_execution_is_an_identity_and_a_reason_never_a_result(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = _calibrated_source(session)
    refused = _refuse(session, source)
    assert refused["schema"] == REFUSAL_SCHEMA and refused["status"] == "refused" and refused["result_id"] is None
    assert refused["source_id"] == source["source_id"] and refused["evidence_id"] == source["evidence_id"]
    assert refused["refusal"]["code"] == "operation_unavailable" and refused["action"] == "execute"
    assert refused["record_digest"] == digest({k: v for k, v in refused.items() if k != "record_digest"})
    assert refused in _call(session, "execution.list", {})["executions"]
    assert _call(session, "bundle.list", {})["bundles"] == [] and _call(session, "result.list", {})["results"] == []
    project = _call(session, "session.get", {})["workbench"]["project"]
    assert not [node for node in project["nodes"] if node["kind"] == "workflow_result"]
    # A second refusal is a second occurrence, never merged with the first.
    again = _refuse(session, source)
    assert again["execution_id"] != refused["execution_id"]
    assert session.workbench.serialize()["schema"] == "ciw.retained-workbench.v3"


def test_refusals_survive_save_and_reopen_without_binding_a_provider(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path / "original")
    refused = _refuse(session, _calibrated_source(session))
    path = session.save_workspace(tmp_path / "workspace.json")
    monkeypatch.setattr(Workbench, "bind_workflow", lambda *_: pytest.fail("Reopen bound a provider"))
    restored = Session.from_workspace(path, tmp_path / "restored")
    assert refused in _call(restored, "execution.list", {})["executions"]
    assert restored.workbench.serialize() == session.workbench.serialize()


def test_refused_replay_names_its_subject_and_keeps_the_original(tmp_path, monkeypatch):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                     "parameters": {"source_id": source["source_id"]}})
    from ciw import thermal_workflow
    original = thermal_workflow.runtime_identity

    def changed():
        value = deepcopy(original())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(thermal_workflow, "runtime_identity", changed)
    reply = _call(session, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    refused = reply["execution"]
    assert reply["status"] == "refused" and refused["action"] == "replay"
    assert refused["subject_bundle_id"] == completed["bundle_id"]
    assert [bundle["bundle_id"] for bundle in session.workbench.list_bundles()] == [completed["bundle_id"]]
    Workbench.restore(session.workbench.serialize())


@pytest.mark.parametrize("tamper", ["digest", "source", "subject", "code", "status", "result", "revision", "duplicate", "empty"])
def test_restore_refuses_tampered_refusal_records(tmp_path, tamper):
    session = Session(make_demo_run(), tmp_path)
    source = _calibrated_source(session)
    _refuse(session, source)
    saved = session.workbench.serialize()
    record = saved["refusals"][0]

    def reseal():
        record["record_digest"] = digest({k: v for k, v in record.items() if k != "record_digest"})

    if tamper == "digest":
        record["refusal"]["message"] = "rewritten"
    elif tamper == "source":
        record["source_id"] = "source:missing"
        reseal()
    elif tamper == "subject":
        record["subject_bundle_id"] = "sha256:" + "0" * 64
        reseal()
    elif tamper == "code":
        record["refusal"]["code"] = ""
        reseal()
    elif tamper == "status":
        record["status"] = "completed"
        reseal()
    elif tamper == "result":
        record["result_id"] = "result-" + "0" * 32
        reseal()
    elif tamper == "revision":
        saved["revision"] += 1
    elif tamper == "duplicate":
        saved["refusals"].append(deepcopy(record))
        saved["revision"] += 1
    else:
        saved["refusals"] = []
        saved["revision"] -= 1
    with pytest.raises(ValueError):
        Workbench.restore(saved)


def test_capacity_is_rejected_not_retained(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    source = _calibrated_source(session)
    monkeypatch.setattr(wb, "MAX_REFUSALS", 1)
    _refuse(session, source)
    reply = session.handle({"protocol_version": 1, "request_id": "r", "type": "operation.execute",
                            "payload": {"operation_id": CALIBRATED, "parameters": {"source_id": source["source_id"]}}})
    assert reply["type"] == "error" and reply["payload"]["code"] == "workbench_capacity"
    assert len(session.workbench.refused_executions()) == 1


def test_malformed_requests_are_rejected_before_an_execution_exists(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    source = _calibrated_source(session)
    for parameters in ({"source_id": "source:missing"}, {"source_id": source["source_id"], "upstream_bundle_id": "x"}):
        reply = session.handle({"protocol_version": 1, "request_id": "r", "type": "operation.execute",
                                "payload": {"operation_id": CALIBRATED, "parameters": parameters}})
        assert reply["type"] == "error"
    assert session.workbench.refused_executions() == []
    assert json.loads(json.dumps(session.workbench.serialize()))["schema"] == "ciw.retained-workbench.v1"


def test_restore_refuses_lineage_the_runtime_cannot_produce(tmp_path):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                     "parameters": {"source_id": source["source_id"]}})
    calibrated = _calibrated_source(session)
    _refuse(session, calibrated)
    saved = session.workbench.serialize()
    record = saved["refusals"][0]
    record["upstream_bundle_ids"] = [completed["bundle_id"], completed["bundle_id"]]
    record["record_digest"] = digest({k: v for k, v in record.items() if k != "record_digest"})
    with pytest.raises(ValueError, match="lineage|upstream"):
        Workbench.restore(saved)


def test_restore_refuses_a_refusal_reusing_a_reproduction_identity(tmp_path):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                     "parameters": {"source_id": source["source_id"]}})
    _refuse(session, _calibrated_source(session))
    saved = session.workbench.serialize()
    bundle, = saved["bundles"]
    reproduction = bundle["native"]["verification"]["reproduction"]["execution_id"]
    assert reproduction not in {step["execution_id"] for step in bundle["native"]["steps"]}
    record = saved["refusals"][0]
    record["execution_id"] = reproduction
    record["record_digest"] = digest({k: v for k, v in record.items() if k != "record_digest"})
    with pytest.raises(ValueError, match="Duplicate refused-execution identity"):
        Workbench.restore(saved)
    assert completed["bundle_id"] == bundle["bundle_id"]


def test_refusal_capacity_protects_other_runs_reservations(tmp_path, monkeypatch):
    session = Session(make_demo_run(), tmp_path)
    source = _calibrated_source(session)
    bench = session.workbench
    monkeypatch.setattr(bench, "_reserved_bytes", wb.MAX_BYTES)
    reply = session.handle({"protocol_version": 1, "request_id": "r", "type": "operation.execute",
                            "payload": {"operation_id": CALIBRATED, "parameters": {"source_id": source["source_id"]}}})
    assert reply["type"] == "error" and reply["payload"]["code"] == "workbench_capacity"
    assert bench.refused_executions() == []


def test_an_unreadable_bound_executable_is_an_unavailable_runtime(tmp_path):
    from ciw.adapters.protocol import AdapterRefusal
    from ciw.declared_workload import _read_bound
    with pytest.raises(AdapterRefusal) as caught:
        _read_bound(tmp_path / "missing-engine", 1024)
    assert caught.value.code == "RUNTIME_UNAVAILABLE" and str(tmp_path) not in str(caught.value)
