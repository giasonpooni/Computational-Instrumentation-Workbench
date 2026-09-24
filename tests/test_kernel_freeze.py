"""The kernel surface is frozen: verbs, envelopes, workspace formats and kinds."""
from pathlib import Path
import re

from ciw import kernel
from ciw.core.covariance import COVARIANCE_SCHEMA
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw import workbench

SRC = Path(__file__).resolve().parents[1] / "src" / "ciw"


def dispatched_verbs():
    text = (SRC / "session.py").read_text()
    body = text[text.index("def _dispatch"):]
    verbs = set(re.findall(r'kind == "([a-z_.]+)"', body))
    for group in re.findall(r"kind in \{([^}]*)\}", body):
        verbs |= set(re.findall(r'"([a-z_.]+)"', group))
    return verbs


def test_dispatcher_serves_exactly_the_declared_verbs():
    assert dispatched_verbs() == kernel.VERBS
    assert not kernel.KERNEL_VERBS & kernel.LEGACY_VERBS and not kernel.VERBS & set(kernel.REMOVED_VERBS)


def test_undeclared_verbs_are_refused_and_declared_verbs_are_routed(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    refused = session.handle({"protocol_version": 1, "request_id": "r", "type": "pipeline.run", "payload": {}})
    assert refused["type"] == "error" and refused["payload"]["code"] == "unknown_command"
    for verb in sorted(kernel.VERBS):
        response = session.handle({"protocol_version": 1, "request_id": "r", "type": verb, "payload": {}})
        assert response["type"] == "response" or response["payload"]["code"] != "unknown_command", verb


def test_workbench_kinds_are_frozen():
    assert set(workbench.OPERATIONS) == kernel.FROZEN_KINDS
    assert workbench.SOURCE_SCHEMA in kernel.ENVELOPES and COVARIANCE_SCHEMA in kernel.ENVELOPES
    runner = (SRC / "operations" / "runner.py").read_text()
    assert '"ciw.execution.v1"' in runner and '"ciw.operation-result.v1"' in runner


def test_no_workspace_format_beyond_the_final_one():
    text = (SRC / "session.py").read_text()
    written = {int(value) for value in re.findall(r'"workspace_version"\] = (\d+)', text)}
    written |= {int(value) for value in re.findall(r'"workspace_version": (\d+)', text)}
    assert written <= set(kernel.WORKSPACE_FORMATS) and max(kernel.WORKSPACE_FORMATS) == kernel.FINAL_WORKSPACE_FORMAT
    assert "not in (1, 2, 3)" in text


def test_kernel_description_is_machine_readable():
    description = kernel.describe()
    assert description["kernel_version"] == "ciw.kernel.v2"
    assert "projection" not in description["verbs"] and set(description["removed_verbs"]) == set(kernel.REMOVED_VERBS)
    assert "experiment.inspect" in description["verbs"]["kernel"]
    assert description["distinct_identities"] == ["evidence", "operation", "execution", "result", "verification"]


def test_removed_projection_verbs_name_their_inspection_view(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    for verb, view in kernel.REMOVED_VERBS.items():
        response = session.handle({"protocol_version": 1, "request_id": "r", "type": verb, "payload": {}})
        assert response["type"] == "error" and response["payload"]["code"] == "unknown_command"
        assert "experiment.inspect" in response["payload"]["message"] and view["view"] in response["payload"]["message"]
        assert view["view"] in kernel.INSPECTION_VIEWS


def test_every_inspection_view_is_served(tmp_path):
    session = Session(make_demo_run(), tmp_path)

    def inspect(payload):
        return session.handle({"protocol_version": 1, "request_id": "r", "type": "experiment.inspect", "payload": payload})

    assert inspect({"view": "instruments"})["payload"] == {"instruments": []}
    assert inspect({"view": "fusion"})["payload"] == {"contexts": []}
    assert inspect({"view": "spatial"})["payload"] == {"sources": []}
    assert inspect({"view": "candidates"})["payload"] == {"candidates": []}
    for payload in ({"view": "experiment", "bundle_id": "missing"}, {"bundle_id": "missing"},
                    {"view": "instrument", "bundle_id": "missing", "instrument": "gsie"},
                    {"view": "spatial", "source_id": "missing"}, {"view": "candidate", "candidate_id": "missing"},
                    {"view": "fusion", "bundle_id": "x"}, {"view": "unknown"}):
        response = inspect(payload)
        assert response["type"] == "error" and response["payload"]["code"] == "invalid_payload", payload
    assert set(kernel.INSPECTION_VIEWS) == {"experiment", "instrument", "instruments", "fusion", "spatial", "candidate", "candidates"}
