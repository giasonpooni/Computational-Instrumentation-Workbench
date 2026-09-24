"""One shared state/diagnostic/evidence boundary with actual pinned ESM replay."""
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw.candidate_evidence import CandidateAdapter
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session, read_json
from ciw.workbench import Workbench, _digest


def request(session, kind, payload=None):
    return session.handle({"protocol_version": 1, "request_id": "candidate-test", "type": kind, "payload": payload or {}})


def response(session, kind, payload=None):
    reply = request(session, kind, payload)
    assert reply["type"] == "response", reply
    return reply["payload"]


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    required = [os.environ.get(key) for key in ("CIW_ESM_BUNDLE_FILE", "CIW_ESM_RUNTIME_FILE", "CIW_ESM_ROOT")]
    if not all(required):
        pytest.skip("set CIW_ESM_BUNDLE_FILE, CIW_ESM_RUNTIME_FILE and CIW_ESM_ROOT for actual ESM integration")
    bundle_path, runtime_path, esm_path = map(Path, required)
    bundle = read_json(bundle_path)
    runtime = read_json(runtime_path)
    root = tmp_path_factory.mktemp("candidate-workbench")
    session = Session(make_demo_run(), root / "workspace")
    raw = base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"])
    source = response(session, "source.add", {"kind": "calibrated-observable", "label": "Synthetic process",
                                             "bytes_b64": base64.b64encode(raw).decode()})
    session.workbench._retain("calibrated-observable", session.workbench.get_source(source["source_id"]), None, bundle)
    registration = {"registrationId": "synthetic-raw-policy", "sourceId": "synthetic-raw", "displayName": "Synthetic fixture only",
        "sourceClass": "OPERATOR_DECLARATION", "licenseId": "synthetic-test-only", "policyVersion": "1",
        "effectiveFrom": "2026-09-01T00:00:00Z", "allowedOperations": ["RETRIEVE", "DERIVE", "INGEST"],
        "allowedAudiences": ["INTERNAL"], "permittedPurposes": ["SYNTHETIC_TEST"], "retention": {"mode": "INDEFINITE"}}
    node = str(Path(shutil.which("node")).resolve())
    configuration = {"node": node, "node_sha256": sha256(Path(node).read_bytes()).hexdigest(),
        "artifact": str(esm_path / ".stamp/workbench-candidate.mjs"), "runtime": runtime,
        "store_root": str(root / "evidence-store"),
        "capture_registration": {**registration, "registrationId": "synthetic-derived-policy", "sourceId": "synthetic-derived"},
        "review_context": {"requestId": "synthetic-workbench", "authority": "role:test-reviewer", "purpose": "SYNTHETIC_TEST",
            "retractions": [], "sources": [{"registration": registration,
                "evidence": [{"artifactRef": e["artifact_ref"], "digest": e["sha256"]} for e in bundle["source"]["evidence"]]}]}}
    session.workbench.bind_candidate_adapter(configuration)
    at = (max(datetime.now(timezone.utc), datetime.fromisoformat(bundle["created_at"].replace("Z", "+00:00"))) + timedelta(seconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    path = session.save_workspace(root / "before.json")
    return {"session": session, "bundle": bundle, "configuration": configuration, "at": at, "before": path, "root": root}


def test_candidate_operations_discoverable_but_unbound(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    operations = response(session, "operation.list")["operations"]
    esm = [item for item in operations if item["operation_id"].startswith("esm.")]
    assert len(esm) == 2 and all(not item["available"] for item in esm)
    assert response(session, "experiment.inspect", {"view": "instruments"}) == {"instruments": []}
    assert response(session, "experiment.inspect", {"view": "candidates"}) == {"candidates": []}


@pytest.mark.parametrize("field", ["artifact", "store_root", "runtime", "review_context", "capture_registration", "operation_id"])
def test_client_cannot_supply_paths_policies_or_override_action(tmp_path, field):
    session = Session(make_demo_run(), tmp_path)
    reply = request(session, "operation.execute", {"operation_id": "esm.inspect-candidate.v1",
        "parameters": {"bundle_id": "untrusted", "inspected_at": "2026-09-22T00:00:00Z", field: "untrusted"}})
    assert reply["type"] == "error"
    assert session.workbench.pending_operations == 0


def test_native_instruments_share_one_explicit_bundle_without_recomputing(fixture, monkeypatch):
    session, bundle = fixture["session"], fixture["bundle"]
    def denied(*args, **kwargs):
        raise AssertionError("Inspection must not execute a provider")
    monkeypatch.setattr("ciw.candidate_evidence._bounded_process", denied)
    listed = response(session, "experiment.inspect", {"view": "instruments"})["instruments"]
    assert {item["instrument"] for item in listed} == {"fsrt", "tbrt", "mcur", "oit", "gsie", "cbsr", "fdir"}
    for role in ("fsrt", "tbrt", "mcur", "oit", "gsie", "cbsr", "fdir"):
        view = response(session, "experiment.inspect", {"view": "instrument", "bundle_id": bundle["bundle_digest"], "instrument": role})
        step, = [s for s in bundle["steps"] if s["runtime_ref"] == role]
        assert view["step"] == step
        assert view["fusion_context"]["covariance"] == bundle["steps"][4]["result"]["data"]["covariance"]
        assert view["linked_results"]["gsie"] == bundle["steps"][4]["result_id"]
        assert view["state_admission"] == "not_performed"
        view["step"].clear()
        assert response(session, "bundle.get", {"bundle_id": bundle["bundle_digest"]}) == bundle


@pytest.fixture(scope="module")
def actions(fixture):
    session, bundle = fixture["session"], fixture["bundle"]
    async def inspect_socket():
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", max_size=8 * 1024 * 1024) as client:
                await client.recv()
                await client.send(json.dumps({"protocol_version": 1, "request_id": "fresh-inspect",
                    "type": "operation.execute", "payload": {"operation_id": "esm.inspect-candidate.v1",
                    "parameters": {"bundle_id": bundle["bundle_digest"], "inspected_at": fixture["at"]}}}))
                reply = json.loads(await asyncio.wait_for(client.recv(), 360))
                assert reply["type"] == "response", reply
                assert json.loads(await asyncio.wait_for(client.recv(), 5))["type"] == "workbench.changed"
                return reply["payload"]
    inspection = asyncio.run(inspect_socket())
    assert inspection["native_response"]["state"] == "ELIGIBLE_FOR_CANDIDATE_REVIEW", inspection["native_response"]["reasons"]
    assert not Path(fixture["configuration"]["store_root"]).exists()
    captured = response(session, "operation.execute", {"operation_id": "esm.capture-candidate.v1",
        "parameters": {"bundle_id": bundle["bundle_digest"], "evidence_id": "artifact:synthetic-shared-candidate",
                       "workflow_id": "synthetic-retention", "retained_at": fixture["at"]}})
    assert captured["native_response"]["state"] == "CANDIDATE_EVIDENCE_RETAINED", captured
    return {"inspection": inspection, "capture": captured}


def test_real_session_capture_binds_exact_evidence_and_no_canonical_state(fixture, actions):
    from ciw.core.canonical import canonical
    capture = actions["capture"]["native_response"]
    files = [path for path in Path(fixture["configuration"]["store_root"]).rglob("*") if path.is_file()]
    assert len(files) == 1
    raw = files[0].read_bytes()
    envelope = json.loads(raw)
    assert capture["capture"]["evidence"]["contentDigest"] == "sha256:" + sha256(raw).hexdigest()
    assert base64.b64decode(envelope["bundleBytesBase64"]) == canonical(fixture["bundle"])
    assert envelope["state"] == "UNADMITTED"
    assert envelope["canonicalAdmission"] == "REFUSED"
    assert envelope["canonicalStateMutated"] is envelope["releaseActivated"] is False
    assert actions["inspection"]["candidate_id"] != actions["capture"]["candidate_id"]
    assert len(response(fixture["session"], "experiment.inspect", {"view": "candidates"})["candidates"]) == 2
    executions = response(fixture["session"], "execution.list")["executions"]
    esm = [entry for entry in executions if entry["runtime_ref"] == "esm"]
    assert {entry["candidate_id"] for entry in esm} == {entry["candidate_id"] for entry in actions.values()}
    assert all("result_id" not in entry for entry in esm)  # evidence receipt is not a numerical result


def test_restore_retains_receipts_but_never_executable_bindings_or_current_eligibility(fixture, actions, tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Restore may not execute or bind ESM")
    monkeypatch.setattr("ciw.candidate_evidence._bounded_process", denied)
    session = fixture["session"]
    path = session.save_workspace(tmp_path / "workspace.json")
    saved = read_json(path)
    assert saved["workbench"]["schema"] == "ciw.retained-workbench.v2"
    text = path.read_text()
    for forbidden in (fixture["configuration"]["node"], fixture["configuration"]["artifact"], fixture["configuration"]["store_root"]):
        assert forbidden not in text
    restored = Session.from_workspace(path, tmp_path / "restored")
    for record in actions.values():
        actual = response(restored, "experiment.inspect", {"view": "candidate", "candidate_id": record["candidate_id"]})
        assert actual == record
        assert actual["eligibility"] == "historical_receipt_only"
    assert all(not entry["available"] for entry in response(restored, "operation.list")["operations"] if entry["operation_id"].startswith("esm."))
    reply = request(restored, "operation.execute", {"operation_id": "esm.inspect-candidate.v1",
        "parameters": {"bundle_id": fixture["bundle"]["bundle_digest"], "inspected_at": fixture["at"]}})
    assert reply["type"] == "error"
    assert reply["payload"]["code"] == "operation_unavailable"


@pytest.mark.parametrize("defect", ["bytes", "bundle", "admission", "fault", "duplicate"])
def test_corrupt_saved_candidate_refused_even_after_outer_redigest(fixture, actions, defect):
    saved = fixture["session"].workbench.serialize()
    record = saved["candidates"][0]
    if defect == "bytes":
        record["response_bytes_b64"] = "not base64!"
    elif defect == "bundle":
        record["parameters"]["bundle_id"] = "missing"
    elif defect == "duplicate":
        saved["candidates"].append(deepcopy(record)); saved["revision"] += 1
    else:
        native = json.loads(base64.b64decode(record["response_bytes_b64"]))
        if defect == "admission": native["canonicalStateMutated"] = True
        if defect == "fault": native["candidate"]["processAssessment"]["isolatedFault"] = "invented-sensor"
        record["response_bytes_b64"] = base64.b64encode(json.dumps(native).encode()).decode()
    record["candidate_id"] = "candidate:" + _digest({k: v for k, v in record.items() if k != "candidate_id"})
    with pytest.raises(ValueError):
        Workbench.restore(saved)


def test_withdrawal_stops_capture_and_preserves_existing_evidence(fixture, actions, tmp_path):
    configuration = deepcopy(fixture["configuration"])
    configuration["review_context"]["retractions"] = [{"retractionId": "withdraw-gsie", "targetKind": "NUMERICAL_RESULT",
        "targetId": fixture["bundle"]["steps"][4]["result"]["data"]["state_id"], "knownAt": fixture["at"],
        "authority": "role:reviewer", "reason": "Synthetic withdrawal"}]
    configuration["store_root"] = str(tmp_path / "not-created")
    session = Session.from_workspace(fixture["before"], tmp_path / "withdrawn")
    session.workbench.bind_candidate_adapter(configuration)
    refused = response(session, "operation.execute", {"operation_id": "esm.capture-candidate.v1",
        "parameters": {"bundle_id": fixture["bundle"]["bundle_digest"], "evidence_id": "artifact:withdrawn",
                       "workflow_id": "withdrawn", "retained_at": fixture["at"]}})
    assert refused["native_response"]["state"] == "REFUSED"
    assert "BOUND_DEPENDENCY_RETRACTED" in refused["native_response"]["reasons"]
    assert not Path(configuration["store_root"]).exists()


@pytest.mark.parametrize("defect", ["artifact", "helper", "node", "runtime-pin", "role", "store-without-policy"])
def test_operator_binding_rejects_code_drift_or_incomplete_contract(fixture, tmp_path, defect):
    configuration = deepcopy(fixture["configuration"])
    if defect in {"artifact", "helper", "node"}:
        target = tmp_path / "substituted"
        target.write_text("untrusted bytes")
        if defect == "helper": configuration["runtime"]["helperPath"] = str(target)
        else: configuration[defect] = str(target)
    if defect == "runtime-pin": configuration["runtime"]["repositories"]["gsie"]["revision"] = "0" * 40
    if defect == "role": configuration["runtime"]["repositories"].pop("fdir")
    if defect == "store-without-policy": configuration.pop("capture_registration")
    from ciw.adapters.protocol import AdapterRefusal
    with pytest.raises((ValueError, AdapterRefusal)):
        CandidateAdapter(configuration)


def test_node_environment_cannot_inject_unpinned_code(tmp_path, monkeypatch):
    from ciw.adapters.subprocess import _bounded_process
    monkeypatch.setenv("NODE_OPTIONS", "--require=/not/allowed.js")
    monkeypatch.setenv("NODE_PATH", "/not/allowed")
    status, output = _bounded_process([shutil.which("node"), "-e", "process.stdout.write(JSON.stringify([process.env.NODE_OPTIONS??null, process.env.NODE_PATH??null]))"],
                                      cwd=tmp_path, timeout=30, limit=1024)
    assert status == 0 and json.loads(output) == [None, None]
