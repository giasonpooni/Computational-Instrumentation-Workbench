"""An exchange bundle shares the catalog with every other kind, and its retained verdict stays pinned.

The pinned SET checker is replaced by a declared test double that returns the
report shape ``exchange.inspect_exchange`` produces for conformant artifacts.
Nothing here is conformance evidence; it exercises catalog and retention rules.
"""

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ciw import exchange
from ciw import exchange_adapter as adapter
from ciw import workbench as catalog
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import digest

from test_machine_workflow import _source as machine_source
from test_project_workflow import _source as project_source

ROOT = Path(__file__).resolve().parents[1]


def _exchange_source():
    artifact = json.loads((ROOT / "examples/exchange/observation.json").read_text(encoding="utf-8"))
    return {"schema": adapter.SOURCE_SCHEMA,
            "producer": {"name": "synthetic-producer", "revision": "fixture-1",
                          "operation_ids": ["bridge.instrumentation.observation_batch_v1"]},
            "artifacts": [artifact]}


def _fake_report(source):
    artifacts = []
    for artifact in source["artifacts"]:
        field = exchange._SCHEMAS[artifact["schema"]][0]
        artifacts.append({"path": "fixture", "source_bytes_sha256": "0" * 64, "source_bytes_count": 1,
                          "identity_field": field, "identity_status": "caller_declared_reference",
                          "artifact": deepcopy(artifact), "conformance": "passed", "covariance_validation": None})
    return {"schema": "ciw.exchange-inspection.v1", "adapter_version": exchange.ADAPTER_VERSION,
            "status": "conformant", "validator": deepcopy(adapter.PIN), "artifacts": artifacts, "links": [],
            "authority": {"may_authorize": False, "native_workspace_import": "not_performed",
                          "source_admission": "not_assessed", "execution_behavior": "not_assessed",
                          "verification_independence": "not_established", "physical_validation": "not_established"},
            "limits": {}}


@pytest.fixture
def doubled_checker(monkeypatch):
    identity = {"repository": adapter.PIN["repository"], "revision": adapter.PIN["revision"],
                "source_tree": "fixture-tree", "module": adapter.PIN["path"],
                "source_sha256": adapter.PIN["sha256"], "execution_scope": "standalone_checked_source_only"}
    monkeypatch.setattr(adapter, "_runtime", lambda repo: dict(identity))
    monkeypatch.setattr(adapter, "_inspect", lambda source, repo: _fake_report(source))


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind, "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _add(session, kind, value):
    raw = adapter.canonical(value)
    return _call(session, "source.add", {"kind": kind, "label": kind + " fixture",
                                         "bytes_b64": base64.b64encode(raw).decode()})["source_id"]


def _execute(session, operation, source_id):
    return _call(session, "operation.execute", {"operation_id": operation, "parameters": {"source_id": source_id}})


def test_declared_kinds_retain_beside_an_exchange_bundle_and_reopen(tmp_path, doubled_checker):
    session = Session(make_demo_run(), tmp_path / "original")
    session.workbench.bind_workflow("instrument-exchange", {"set": tmp_path})
    exchange_bundle = _execute(session, adapter.OPERATION, _add(session, "instrument-exchange", _exchange_source()))
    # Before the fix every later declared retention crashed on the exchange record.
    machine = _execute(session, "ciw.encoder-position.v1", _add(session, "machine-manifest", machine_source()))
    project = _execute(session, "ciw.project-graph.v1", _add(session, "project-graph", project_source()))
    listed = {item["bundle_id"] for item in _call(session, "bundle.list", {})["bundles"]}
    assert listed == {exchange_bundle["bundle_id"], machine["bundle_id"], project["bundle_id"]}
    replay = _call(session, "bundle.replay", {"bundle_id": exchange_bundle["bundle_id"]})
    fresh = _call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    assert fresh["replay_of"] == exchange_bundle["bundle_id"]
    assert fresh["steps"][0]["execution_id"] != _call(session, "bundle.get", {"bundle_id": exchange_bundle["bundle_id"]})["steps"][0]["execution_id"]
    saved = session.save_workspace(tmp_path / "saved.json")
    reopened = Session.from_workspace(saved, tmp_path / "reopened")
    assert {item["bundle_id"] for item in _call(reopened, "bundle.list", {})["bundles"]} == listed | {replay["bundle"]["bundle_id"]}
    executions = [item["execution_id"] for item in _call(reopened, "execution.list", {})["executions"]]
    assert len(executions) == len(set(executions))


def test_exchange_bundle_is_not_counted_as_a_reproduced_kind():
    assert "instrument-exchange" in catalog.DECLARED_KINDS
    assert "instrument-exchange" not in catalog.REPRODUCED_KINDS


def _reseal(native):
    step = native["steps"][0]
    step["result_sha256"] = digest(step["result"])
    step["numerical_result_id"] = digest(step["numerical_result"])
    step["result"]["numerical_result_id"] = step["numerical_result_id"]
    step["result_sha256"] = digest(step["result"])
    native["bundle_digest"] = adapter._bundle_digest(native)
    verification = native["verification"]
    verification["subject_ref"] = native["bundle_digest"]
    verification["verification_id"] = adapter._identity(verification, "verification_id")
    return native


def _retained_exchange(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("instrument-exchange", {"set": tmp_path})
    completed = _execute(session, adapter.OPERATION, _add(session, "instrument-exchange", _exchange_source()))
    return session.workbench.get_bundle(completed["bundle_id"])


def test_resealing_preserves_an_untouched_exchange_bundle(tmp_path, doubled_checker):
    native = _retained_exchange(tmp_path)
    resealed = _reseal(deepcopy(native))
    assert resealed == native
    adapter._validate(resealed)


@pytest.mark.parametrize("tamper,message", [
    (lambda n: n["steps"][0]["result"]["authority"].__setitem__("may_authorize", True), "fixed scope"),
    (lambda n: n["steps"][0]["result"].__setitem__("claim_scope", "physical_validation_established"), "fixed scope"),
    (lambda n: n["steps"][0]["result"].__setitem__("physical_validation", "established"), "fixed scope"),
    (lambda n: n["steps"][0]["result"]["validator"].__setitem__("revision", "0" * 40), "validator"),
    (lambda n: n["authority"].__setitem__("may_authorize", True), "authority"),
    (lambda n: n["verification"].__setitem__("outcome", "failed"), "verification scope"),
    (lambda n: n["verification"].__setitem__("verifier_ref", "independent-laboratory"), "verification scope"),
    (lambda n: n["steps"][0]["result"].__setitem__("producer_artifact_ids", ["batch:forged"]), "producer identities"),
    (lambda n: n.__setitem__("session_id", "session-" + "0" * 32), "unexpected shape"),
])
def test_resealed_exchange_bundle_cannot_promote_its_verdict(tmp_path, doubled_checker, tamper, message):
    native = _retained_exchange(tmp_path)
    tamper(native)
    with pytest.raises(ValueError, match=message):
        adapter._validate(_reseal(native))


def test_resealed_exchange_execution_identity_shape_is_refused(tmp_path, doubled_checker):
    native = _retained_exchange(tmp_path)
    step = native["steps"][0]
    forged = "execution-" + "0" * 32
    step["execution_id"] = forged
    step["result"]["execution_id"] = forged
    step["result"]["execution_ref"] = forged
    with pytest.raises(ValueError, match="unexpected shape"):
        adapter._validate(_reseal(native))
