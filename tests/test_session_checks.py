"""Every request, payload and saved-workspace check of the session refuses what it names.

An AST mutation probe dropped each ``if ...: raise`` of ``session`` in turn;
58 of 79 survived because the suite asserted only that an error came back,
never which one. Each case here expects the error code and message of the
check it exercises, so a dropped check either accepts the input or answers
with a different error.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import session as session_module
from ciw.instruments import make_demo_run
from ciw.operations.runner import seal
from ciw.session import Session, read_json

ROOT = Path(__file__).resolve().parents[1]
HEX = "0123456789abcdef0123456789abcdef"


def request(kind, payload=None, **overrides):
    envelope = {"protocol_version": 1, "request_id": "check", "type": kind, "payload": {} if payload is None else payload}
    envelope.update(overrides)
    return envelope


def error(session, envelope):
    response = session.handle(envelope)
    assert response["type"] == "error", response
    return response["payload"]["code"], response["payload"]["message"]


@pytest.fixture
def session(tmp_path):
    return Session(make_demo_run(), tmp_path)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    directory = tmp_path_factory.mktemp("session-checks")
    session = Session(make_demo_run(), directory)
    for envelope in (request("analysis.stats"),
                     request("operation.execute", {"operation_id": "statistics.v1", "parameters": {}}),
                     request("operation.execute", {"operation_id": "spectrum.periodogram.v1", "parameters": {}}),
                     request("operation.execute", {"operation_id": "ciw.stats.v1", "parameters": {}})):
        response = session.handle(envelope)
        assert response["type"] == "response", response
    return read_json(session.save_workspace(directory / "workspace.json"))


def reopen(tmp_path, workspace):
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    return Session.from_workspace(path, tmp_path / "reopened")


def legacy(workspace):
    return next(item for item in workspace["results"] if item.get("schema") != "ciw.operation-result.v1")


def operation_result(workspace, operation_id="statistics.v1"):
    return next(item for item in workspace["results"]
                if item.get("schema") == "ciw.operation-result.v1" and item["operation_id"] == operation_id)


def execution_of(workspace, status):
    return next(item for item in workspace["executions"] if item["status"] == status)


def resealed(record, **changes):
    record.update(changes)
    return seal(record)


ENVELOPES = {
    "request not an object": ("text", "invalid_request", "Request must be an object"),
    "request without an id": ({"protocol_version": 1, "type": "session.get", "payload": {}}, "invalid_request", "request_id must be a nonempty string"),
    "protocol version as text": (request("session.get", protocol_version="1"), "unsupported_version", "Supported protocol_version is 1"),
    "protocol version two": (request("session.get", protocol_version=2), "unsupported_version", "Supported protocol_version is 1"),
    "extra envelope key": ({**request("session.get"), "extra": 1}, "invalid_request", "Use protocol_version, request_id, type and payload only"),
    "type not text": (request(5), "invalid_request", "type must be a string and payload an object"),
    "payload not an object": (request("session.get", payload=[]), "invalid_request", "type must be a string and payload an object"),
}

PAYLOADS = {
    "missing required field": ("selection.update", {}, "invalid_payload", "Missing payload fields: expected_revision"),
    "evaluated_at not text": ("session.get", {"evaluated_at": 5}, "invalid_payload", "evaluated_at must be a timezone-aware ISO timestamp"),
    "expected revision as text": ("selection.update", {"expected_revision": "0", "channel": "q"}, "invalid_payload", "expected_revision must be a nonnegative integer"),
    "expected revision negative": ("selection.update", {"expected_revision": -1, "channel": "q"}, "invalid_payload", "expected_revision must be a nonnegative integer"),
    "nothing to update": ("selection.update", {"expected_revision": 0}, "invalid_payload", "Provide a channel, interval_s or cursor_s"),
    "interval not a pair": ("selection.update", {"expected_revision": 0, "interval_s": "ab"}, "invalid_payload", r"interval_s must be \[start, end\]"),
    "interval with one bound": ("selection.update", {"expected_revision": 0, "interval_s": [1.0]}, "invalid_payload", r"interval_s must be \[start, end\]"),
    "interval between two samples": ("selection.update", {"expected_revision": 0, "interval_s": [0.005, 0.01]}, "invalid_payload", "contains no retained samples"),
    "channel not text": ("selection.update", {"expected_revision": 0, "channel": ["q"]}, "invalid_payload", "Unknown channel"),
    "unversioned operation": ("operation.execute", {"operation_id": "statistics", "parameters": {}}, "invalid_payload", "operation_id must be versioned"),
    "operation parameters not an object": ("operation.execute", {"operation_id": "statistics.v1", "parameters": "x"}, "invalid_payload", "operation_id must be versioned"),
    "result id not text": ("result.get", {"result_id": ["x"]}, "not_found", "Result not found"),
}


@pytest.mark.parametrize("envelope, code, message", ENVELOPES.values(), ids=list(ENVELOPES))
def test_a_malformed_envelope_is_named_by_its_own_check(session, envelope, code, message):
    actual_code, actual_message = error(session, envelope)
    assert actual_code == code and message in actual_message, (actual_code, actual_message)


@pytest.mark.parametrize("kind, payload, code, message", PAYLOADS.values(), ids=list(PAYLOADS))
def test_a_malformed_payload_is_named_by_its_own_check(session, kind, payload, code, message):
    import re
    actual_code, actual_message = error(session, request(kind, payload))
    assert actual_code == code and re.search(message, actual_message), (actual_code, actual_message)


def test_the_legacy_result_budget_is_enforced(session):
    session.results = {f"result-{index:032x}": {} for index in range(1024)}
    assert error(session, request("analysis.stats")) == (
        "capacity_exceeded", "Save the workspace and start a new session after 1024 results")


def test_the_operation_budget_is_enforced_before_reserving(session, monkeypatch):
    calls = []
    monkeypatch.setattr(session_module, "execute_operation", lambda *args, **kwargs: calls.append(args))
    message = ("capacity_exceeded", "Save and start a new session after 1024 operations")
    session.results = {f"result-{index:032x}": {} for index in range(1024)}
    assert error(session, request("operation.execute", {"operation_id": "statistics.v1", "parameters": {}})) == message
    session.results = {}
    session.executions = {f"execution-{index:032x}": {} for index in range(1024)}
    assert error(session, request("operation.execute", {"operation_id": "statistics.v1", "parameters": {}})) == message
    assert calls == []  # a full session never starts the provider


@pytest.mark.parametrize("crowd", ["executions", "results"])
def test_the_operation_budget_is_rechecked_after_a_provider_returns(session, monkeypatch, crowd):
    original = session_module.execute_operation

    def crowded(*args, **kwargs):
        execution, result = original(*args, **kwargs)
        # Another publication filled the session while the provider ran.
        getattr(session, crowd).update({f"{crowd[:-1]}-{index:032x}": {} for index in range(1024)})
        return execution, result

    monkeypatch.setattr(session_module, "execute_operation", crowded)
    assert error(session, request("operation.execute", {"operation_id": "statistics.v1", "parameters": {}})) == (
        "capacity_exceeded", "Save and start a new session after 1024 operations")


def test_a_duplicate_json_key_is_refused_when_a_file_is_read(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key: a"):
        read_json(path)


def whole(mutate):
    return mutate


def selection(key, value):
    def apply(workspace):
        workspace["selection"][key] = value
    return apply


def legacy_field(key, value):
    def apply(workspace):
        legacy(workspace)[key] = value
    return apply


def operation_field(key, value):
    def apply(workspace):
        resealed(operation_result(workspace), **{key: value})
    return apply


def too_many_results(workspace):
    workspace["results"] = [deepcopy(legacy(workspace)) for _ in range(1025)]


def too_many_executions(workspace):
    workspace["executions"] = [deepcopy(execution_of(workspace, "completed")) for _ in range(1025)]


def duplicate_execution(workspace):
    workspace["executions"].append(deepcopy(execution_of(workspace, "completed")))


def refused_execution_reusing_a_result_occurrence(workspace):
    completed = operation_result(workspace)["execution_id"]
    workspace["executions"] = [item for item in workspace["executions"] if item["execution_id"] != completed]
    resealed(execution_of(workspace, "refused"), execution_id=completed)


def operation_result_without_its_execution(workspace):
    completed = operation_result(workspace)["execution_id"]
    workspace["executions"] = [item for item in workspace["executions"] if item["execution_id"] != completed]


def collision_with_a_retained_workflow(workspace):
    catalog = json.loads((ROOT / "tests" / "fixtures" / "retained" / "workbench.json").read_text(encoding="utf-8"))
    native = catalog["bundles"][0]["native"]["steps"][0]["execution_id"]
    assert native.startswith("execution-") and len(native) == len("execution-") + 32
    workspace["workspace_version"] = 3
    workspace["workbench"] = catalog
    legacy(workspace)["execution_id"] = native


WORKSPACES = {
    "workspace not an object": (lambda workspace: workspace.clear(), "Unsupported workspace format"),
    "version as text": (lambda workspace: workspace.__setitem__("workspace_version", "2"), "Unsupported workspace format"),
    "version four": (lambda workspace: workspace.__setitem__("workspace_version", 4), "Unsupported workspace format"),
    "workbench before version three": (lambda workspace: workspace.__setitem__("workbench", {}), "Retained workbench sources require workspace version 3"),
    "selection not an object": (lambda workspace: workspace.__setitem__("selection", "x"), "Invalid saved selection"),
    "selection missing a key": (lambda workspace: workspace["selection"].pop("cursor_s"), "Invalid saved selection"),
    "selection of another recording": (selection("run_id", "other"), "does not refer to the recording"),
    "selection in another frame": (selection("coordinate_frame", "other"), "does not refer to the recording"),
    "negative selection revision": (selection("revision", -1), "Invalid saved selection revision"),
    "results not a list": (lambda workspace: workspace.__setitem__("results", "x"), "Invalid saved results"),
    "too many results": (too_many_results, "Invalid saved results"),
    "executions not a list": (lambda workspace: workspace.__setitem__("executions", "x"), "Invalid saved executions"),
    "too many executions": (too_many_executions, "Invalid saved executions"),
    "executions in a version one workspace": (lambda workspace: workspace.__setitem__("workspace_version", 1), "Executions require workspace version 2"),
    "duplicate execution": (duplicate_execution, "Duplicate execution identity"),
    "refused execution reusing a result occurrence": (refused_execution_reusing_a_result_occurrence, "Execution identity collision"),
    "operation result without its execution": (operation_result_without_its_execution, "missing its completed execution"),
    "collision with a retained workflow": (collision_with_a_retained_workflow, "Identity collision between recording operations and retained workflows"),
    "result not an object": (lambda workspace: workspace["results"].__setitem__(0, "x"), "Saved result is missing identity"),
    "result identity not text": (legacy_field("result_id", 5), "Invalid saved result identity"),
    "result identity with another prefix of the same length": (legacy_field("result_id", "resulx-" + HEX), "Invalid saved result identity"),
    "result identity too short": (legacy_field("result_id", "result-abc"), "Invalid saved result identity"),
    "result identity in upper case": (legacy_field("result_id", "result-" + HEX.upper()), "Invalid saved result identity"),
    "created_at not text": (legacy_field("created_at", 5), "created_at must be a timezone-aware ISO timestamp"),
    "created_at without a timezone": (legacy_field("created_at", "2026-09-19T12:00:00"), "must include a timezone"),
    "selection revision as a float": (legacy_field("selection_revision", 0.0), "selection_revision is outside"),
    "operation result runtime not an object": (operation_field("runtime", "x"), "Invalid saved operation result"),
    "operation result parameters not an object": (operation_field("parameters", "x"), "Invalid saved operation result"),
    "operation result data not an object": (operation_field("data", "x"), "Invalid saved operation result"),
    "operation result with an unknown role": (operation_field("role", "other"), "Invalid saved operation result"),
    "operation result with an unversioned operation": (operation_field("operation_id", "statistics"), "Invalid saved operation result"),
    "operation parameters contradicting the selection": (operation_field("parameters", {"channel": "p", "interval_s": [0.0, 12.0]}), "contradict its captured selection"),
}


def test_a_workspace_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported workspace format"):
        Session.from_workspace(path, tmp_path / "reopened")


def test_the_template_workspace_reopens(template, tmp_path):
    assert template["workspace_version"] == 2
    assert {item["status"] for item in template["executions"]} == {"completed", "refused"}
    reopen(tmp_path, deepcopy(template))


@pytest.mark.parametrize("mutate, message", WORKSPACES.values(), ids=list(WORKSPACES))
def test_one_change_to_a_saved_workspace_is_refused_by_its_own_check(template, tmp_path, mutate, message):
    workspace = deepcopy(template)
    mutate(workspace)
    with pytest.raises(ValueError, match=message):
        reopen(tmp_path, workspace)
    assert not (tmp_path / "reopened").exists()
