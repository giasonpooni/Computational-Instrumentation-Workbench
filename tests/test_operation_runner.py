"""Execution identity, effective selection and fail-closed offline payload gates."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest

from ciw.instruments import make_demo_run
from ciw.operations.registry import Operation, OperationRegistry, default_registry
from ciw.operations.runner import seal
from ciw.session import Session, read_json, write_json


def request(session, operation="statistics.v1", parameters=None):
    return session.handle({"protocol_version": 1, "request_id": "operation-test",
                           "type": "operation.execute", "payload": {
                               "operation_id": operation, "parameters": parameters or {}}})


def test_generic_analysis_captures_effective_selection_and_preserves_global_selection(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    before = deepcopy(session.selection)
    reply = request(session, parameters={"channel": "v", "interval_s": [1., 2.]})
    assert reply["type"] == "response"
    execution, result = reply["payload"]["execution"], reply["payload"]["result"]
    assert reply["payload"]["status"] == "completed"
    for row in (execution, result):
        assert row["channel"] == "v" and row["interval_s"] == [1., 2.]
        assert row["parameters"] == {"channel": "v", "interval_s": [1., 2.]}
    assert result["data"]["sample_count"] == 64 and result["data"]["unit"] == "m/s"
    assert session.selection == before
    path = session.save_workspace(tmp_path / "workspace.json")
    with patch("ciw.operations.runner.execute", side_effect=AssertionError("must not execute")):
        restored = Session.from_workspace(path, tmp_path / "restored")
    assert restored.results == session.results and restored.executions == session.executions
    second = request(restored)["payload"]
    assert second["status"] == "completed"
    assert second["result"]["result_id"] != result["result_id"]
    assert second["execution"]["execution_id"] != execution["execution_id"]


def test_unknown_operation_retains_refusal_without_a_result(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    response = request(session, "missing.v1")["payload"]
    assert response["status"] == "refused" and response["result"] is None
    assert response["execution"]["refusal"]["code"] == "operation_unavailable"
    assert session.results == {} and len(session.executions) == 1
    path = session.save_workspace(tmp_path / "workspace.json")
    assert Session.from_workspace(path, tmp_path / "restored").executions == session.executions


def test_provider_cannot_publish_without_an_offline_payload_schema(tmp_path):
    operations = OperationRegistry()
    operations.register(Operation("unvalidated.v1", "backend", lambda run, params: {"value": 1},
                                  lambda: {"version": "declared"}))
    session = Session(make_demo_run(), tmp_path, operations=operations)
    response = request(session, "unvalidated.v1")["payload"]
    assert response["status"] == "refused" and not session.results
    assert "schema" in response["execution"]["refusal"]["message"]


@pytest.mark.parametrize("mutation", ["data", "parameters", "execution_interval", "refused_link", "duplicate", "role"])
def test_resealed_semantic_corruption_is_rejected_before_any_write(tmp_path, mutation):
    session = Session(make_demo_run(), tmp_path / "initial")
    request(session)
    path = session.save_workspace(tmp_path / "workspace.json")
    workspace = read_json(path)
    result, execution = workspace["results"][0], workspace["executions"][0]
    if mutation == "data":
        result["data"]["rms"] = -1
    elif mutation == "parameters":
        result["parameters"]["channel"] = "v"
        execution["parameters"]["channel"] = "v"
    elif mutation == "execution_interval":
        execution["interval_s"] = [1, 2]
    elif mutation == "refused_link":
        execution.update(status="refused", result_id=None,
                         refusal={"code": "refused", "message": "No result"})
    elif mutation == "duplicate":
        workspace["executions"].append(deepcopy(execution))
    elif mutation == "role":
        result["role"] = "verification"
    seal(result)
    seal(execution)
    write_json(path, workspace)
    with patch("ciw.session.write_json") as writer:
        with pytest.raises(ValueError):
            Session.from_workspace(path, tmp_path / "rejected")
        writer.assert_not_called()


def test_duplicate_json_keys_are_rejected_at_workspace_boundary(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text('{"workspace_version":1,"workspace_version":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        Session.from_workspace(path)


def waiting_operations(entered, release):
    scientific = default_registry()

    def wait_then_calculate(run, parameters):
        entered.set()
        assert release.wait(timeout=10), "test provider was never released"
        return scientific.get("statistics.v1").execute(run, parameters)

    operations = OperationRegistry()
    operations.register(Operation("statistics.v1", "analysis", wait_then_calculate,
                                  lambda: {"provider": "waiting-test", "version": "1"}))
    operations.register(scientific.get("spectrum.periodogram.v1"))
    return operations


def test_waiting_provider_leaves_session_and_selection_available(tmp_path):
    entered, release = Event(), Event()
    session = Session(make_demo_run(), tmp_path, operations=waiting_operations(entered, release))
    parameters = {"channel": "q", "interval_s": [1.0, 2.0]}
    with ThreadPoolExecutor(max_workers=3) as workers:
        calculating = workers.submit(request, session, parameters=parameters)
        try:
            assert entered.wait(timeout=3)
            # Both requests must finish before the numerical provider resumes.
            snapshot = workers.submit(session.handle, {
                "protocol_version": 1, "request_id": "concurrent-read",
                "type": "session.get", "payload": {},
            }).result(timeout=2)
            assert snapshot["payload"]["selection"]["revision"] == 0
            update = workers.submit(session.handle, {
                "protocol_version": 1, "request_id": "concurrent-selection", "type": "selection.update",
                "payload": {"expected_revision": 0, "channel": "v", "interval_s": [2.0, 3.0]},
            }).result(timeout=2)
            assert update["type"] == "response" and update["payload"]["revision"] == 1
            # Mutating the original caller-owned payload must not retarget it.
            parameters["interval_s"][0] = 0.5
            assert not calculating.done()
        finally:
            release.set()
        completed = calculating.result(timeout=3)["payload"]
    assert completed["status"] == "completed"
    for record in (completed["execution"], completed["result"]):
        assert record["selection_revision"] == 0
        assert record["channel"] == "q" and record["interval_s"] == [1.0, 2.0]
        assert record["parameters"] == {"channel": "q", "interval_s": [1.0, 2.0]}
    assert session.selection["revision"] == 1 and session.selection["interval_s"] == [2.0, 3.0]
    saved = session.save_workspace(tmp_path / "workspace.json")
    reopened = Session.from_workspace(saved, tmp_path / "reopened")
    assert reopened.results == session.results and reopened.selection == session.selection


@pytest.mark.parametrize("capacity", ["results", "executions"])
def test_pending_operation_reserves_the_last_generic_capacity_slot(tmp_path, capacity):
    entered, release = Event(), Event()
    session = Session(make_demo_run(), tmp_path, operations=waiting_operations(entered, release))
    # Only cardinality matters to admission; avoid generating 1023 calculations.
    getattr(session, capacity).update({f"retained-{index}": {} for index in range(1023)})
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(request, session)
        try:
            assert entered.wait(timeout=3)
            second = workers.submit(request, session).result(timeout=2)
            assert second["type"] == "error" and second["payload"]["code"] == "capacity_exceeded"
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=3)["payload"]["status"] == "completed"
    assert len(getattr(session, capacity)) == 1024
    assert len(session.results) <= 1024 and len(session.executions) <= 1024


def test_generic_publication_rechecks_capacity_after_legacy_analysis(tmp_path):
    entered, release = Event(), Event()
    session = Session(make_demo_run(), tmp_path, operations=waiting_operations(entered, release))
    session.results.update({f"retained-{index}": {} for index in range(1023)})
    with ThreadPoolExecutor(max_workers=2) as workers:
        calculating = workers.submit(request, session)
        try:
            assert entered.wait(timeout=3)
            legacy = workers.submit(session.handle, {
                "protocol_version": 1, "request_id": "legacy-fill", "type": "analysis.spectrum", "payload": {},
            }).result(timeout=2)
            assert legacy["type"] == "response"
        finally:
            release.set()
        declined = calculating.result(timeout=3)
    assert declined["type"] == "error" and declined["payload"]["code"] == "capacity_exceeded"
    assert len(session.results) == 1024 and not session.executions
    assert not list(tmp_path.glob("execution-*.json"))


def test_failed_publication_releases_pending_reservation(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    session.results.update({f"retained-{index}": {} for index in range(1023)})
    with patch("ciw.session.write_json", side_effect=OSError("unavailable storage")):
        declined = request(session)
    assert declined["type"] == "error" and declined["payload"]["code"] == "storage_error"
    assert len(session.results) == 1023 and not session.executions
    assert request(session)["payload"]["status"] == "completed"
    assert len(session.results) == 1024 and len(session.executions) == 1
