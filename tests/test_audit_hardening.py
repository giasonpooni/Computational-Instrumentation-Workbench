"""Probes from the system audit that became permanent tests."""

import base64
import logging
from pathlib import Path


from ciw import uncertainty_validation as validation
from ciw.instruments import make_demo_run
from ciw.session import Session

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "uncertainty-validation" / "consistent.json"
OPERATION = "ciw.uncertainty-validation.v1"


def _call(session, kind, payload, request_id="audit"):
    return session.handle({"protocol_version": 1, "request_id": request_id, "type": kind, "payload": payload})


def test_a_defect_inside_one_request_answers_an_internal_error_envelope_and_leaves_the_session_usable(tmp_path, monkeypatch, caplog):
    session = Session(make_demo_run(), tmp_path)
    source = _call(session, "source.add", {"kind": "uncertainty-validation", "label": "u",
                                           "bytes_b64": base64.b64encode(SOURCE.read_bytes()).decode()})["payload"]
    def boom(self, raw, repositories):
        raise RuntimeError("provider blew up unexpectedly")
    monkeypatch.setattr(validation.UncertaintyValidationWorkflow, "create_session", boom)
    with caplog.at_level(logging.ERROR, logger="ciw.session"):
        response = _call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source["source_id"]}}, "req-7")
    assert response["type"] == "error" and response["request_id"] == "req-7"
    assert response["payload"]["code"] == "internal_error"
    assert "blew up" not in response["payload"]["message"], "internal details stay in the log"
    assert any("req-7" in record.getMessage() and record.exc_info for record in caplog.records)
    assert session.workbench.pending_operations == 0 and session.workbench.list_bundles() == []
    monkeypatch.undo()
    completed = _call(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source["source_id"]}})
    assert completed["type"] == "response" and len(session.workbench.list_bundles()) == 1
    listed = _call(session, "execution.list", {})["payload"]["executions"]
    assert {entry["status"] for entry in listed} == {"completed"}


def test_the_operation_catalogue_page_indexes_every_operation():
    from ciw.workbench import OPERATIONS
    catalogue = (ROOT / "docs" / "INSTRUMENTS.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    for kind, operation in OPERATIONS.items():
        assert f"`{operation}`" in catalogue and f"`{kind}`" in catalogue, operation
    for code in ("invalid_request", "invalid_payload", "unsupported_version", "internal_error", "storage_error", "read_only_view"):
        assert f"`{code}`" in protocol, code
