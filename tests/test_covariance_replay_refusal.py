"""A retained admission refusal cannot select today's runtime through replay."""
from copy import deepcopy

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.covariance_workflow import JSPT_OPERATION, replay_covariance
from ciw.instruments import make_demo_run
from ciw.operations.registry import Operation, default_registry
from ciw.session import Session


def execute(session):
    return session.handle({"protocol_version": 1, "request_id": "unbound-covariance",
                           "type": "operation.execute", "payload": {
                               "operation_id": JSPT_OPERATION, "parameters": {}}})["payload"]


@pytest.mark.parametrize("earlier_bound_attempt", [False, True])
def test_unbound_latest_attempt_refuses_replay_before_runtime_resolution_or_writes(
    tmp_path, monkeypatch, earlier_bound_attempt,
):
    session = Session(make_demo_run(), tmp_path / "source")
    if earlier_bound_attempt:
        def refuse(run, parameters):
            raise AdapterRefusal("numerical_refusal", "Earlier bound computation was refused")

        session.operations.register(Operation(JSPT_OPERATION, "backend", refuse,
                                               lambda: {"provider": "earlier-runtime"}))
        old = execute(session)
        assert old["execution"]["runtime"] == {"provider": "earlier-runtime"}
        # A new unbound session must not borrow an earlier attempt's identity.
        session.operations = default_registry()
    latest = execute(session)
    assert latest["status"] == "refused" and latest["result"] is None
    assert latest["execution"]["runtime"] is None
    assert latest["execution"]["refusal"]["code"] == "operation_unavailable"
    workspace = session.save_workspace(session.output_dir / "workspace.json")
    source_bytes = {path: path.read_bytes() for path in session.output_dir.iterdir()}
    executions = deepcopy(session.executions)

    def no_runtime(*args, **kwargs):
        raise AssertionError("An unbound attempt must not resolve any runtime pin")

    monkeypatch.setattr("ciw.covariance_workflow._runtime", no_runtime)
    destination = tmp_path / "must-not-write"
    with pytest.raises(AdapterRefusal, match="no bound numerical runtime") as error:
        replay_covariance(workspace, tmp_path / "not-a-checkout", destination)
    assert error.value.code == "replay_unavailable"
    assert not destination.exists()
    assert session.executions == executions
    assert {path: path.read_bytes() for path in session.output_dir.iterdir()} == source_bytes
