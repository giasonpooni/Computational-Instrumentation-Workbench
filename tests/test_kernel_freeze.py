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
    assert not kernel.KERNEL_VERBS & kernel.PROJECTION_VERBS and not kernel.KERNEL_VERBS & kernel.LEGACY_VERBS


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
    assert description["kernel_version"] == "ciw.kernel.v1"
    assert "experiment.inspect" in description["verbs"]["kernel"]
    assert description["distinct_identities"] == ["evidence", "operation", "execution", "result", "verification"]
