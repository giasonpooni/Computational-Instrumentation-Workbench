"""Actual PPDA/STFE acquisition projection and feature chain in one shared Session."""
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

from ciw import telemetry
from ciw.cli import parser
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session, read_json
from ciw.workbench import Workbench, _claims, _validate_links

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples/telemetry/source.json").read_bytes()
CONFIG = read_json(ROOT / "examples/telemetry/configuration.json")


def request(session, kind, payload=None):
    return session.handle({"protocol_version": 1, "request_id": "telemetry-session", "type": kind, "payload": payload or {}})


def response(session, kind, payload=None):
    result = request(session, kind, payload)
    assert result["type"] == "response", result
    return result["payload"]


def add_source(session, raw=SOURCE, label="Synthetic retained telemetry"):
    return response(session, "source.add", {"kind": "telemetry", "label": label, "bytes_b64": base64.b64encode(raw).decode()})


def execute(session, source_id, config=CONFIG):
    return response(session, "operation.execute", {"operation_id": "ciw.telemetry.v1",
        "parameters": {"source_id": source_id, "configuration": config}})


@pytest.fixture(scope="module")
def repositories():
    root = os.environ.get("CIW_SHARED_TELEMETRY_STACK_ROOT")
    if not root:
        pytest.skip("set CIW_SHARED_TELEMETRY_STACK_ROOT to exact role-named telemetry checkouts")
    return {role: Path(root) / role for role in telemetry.ROLES}


@pytest.fixture(scope="module")
def retained(repositories, tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-telemetry")
    session = Session(make_demo_run(), root / "session")
    session.workbench.bind_workflow("telemetry", repositories)
    raw = b"\n " + SOURCE + b"\n"  # Exact bytes, not just parsed sample values.
    source = add_source(session, raw)
    summary = execute(session, source["source_id"])
    bundle = response(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    path = session.save_workspace(root / "original.json")
    return {"session": session, "source": source, "bundle": bundle, "path": path, "root": root, "raw": raw}


def test_source_retention_and_startup_bindings_extend_existing_interfaces(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    descriptor = add_source(session)
    assert base64.b64decode(response(session, "source.get", {"source_id": descriptor["source_id"]})["bytes_b64"]) == SOURCE
    operations = response(session, "operation.list")["operations"]
    assert {"ciw.telemetry.v1", "ciw.calibrated-observable.v1", "ciw.identified-design.v1"} <= {x["operation_id"] for x in operations}
    entry, = [x for x in operations if x["operation_id"] == "ciw.telemetry.v1"]
    assert entry["available"] is False
    args = parser().parse_args(["serve", "--telemetry-stack-root", "/trusted/telemetry", "--identified-stack-root", "/trusted/process",
                               "--esm-telemetry-binding", "/trusted/esm-telemetry.json"])
    assert args.telemetry_stack_root == Path("/trusted/telemetry")
    assert args.identified_stack_root == Path("/trusted/process")


@pytest.mark.parametrize("field,value", [("crosscov_policy", "unknown"), ("covariance", None), ("mappings", {}), ("schema", "other")])
def test_unbound_or_undeclared_source_cannot_enter_shared_pipeline(tmp_path, field, value):
    session = Session(make_demo_run(), tmp_path)
    source = json.loads(SOURCE); source[field] = value
    result = request(session, "source.add", {"kind": "telemetry", "label": "invalid", "bytes_b64": base64.b64encode(telemetry.canonical(source)).decode()})
    assert result["type"] == "error"
    assert response(session, "source.list")["sources"] == []


def test_features_and_state_are_native_and_full_temporal_covariance_survives(retained):
    session, bundle = retained["session"], retained["bundle"]
    assert base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"]) == retained["raw"]
    batch, feature, estimate = [s["result"] for s in bundle["steps"]]
    assert batch["covariance"]["matrix"] == [[1.0, .25], [.25, 1.0]]
    assert feature["result_artifact"]["components"][0]["value"] == 3.0
    assert feature["result_artifact"]["covariance"]["matrix"] == [[.625]]  # 1/n² · 1ᵀR1
    assert estimate["result_artifact"]["components"][0]["value"] == pytest.approx(24 / 13)
    assert estimate["result_artifact"]["covariance"]["matrix"][0][0] == pytest.approx(5 / 13)
    for step in bundle["steps"]:
        assert response(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
        view = response(session, "instrument.inspect", {"bundle_id": bundle["bundle_digest"], "instrument": step["runtime_ref"]})
        assert view["step"] == step
        assert view["linked_results"]["ppda"] == batch["batch_id"]
    context, = response(session, "fusion.list")["contexts"]
    assert context["state_kind"] == "window_feature_posterior"
    assert context["observability"] == {"status": "unresolved", "reason": "not_evaluated_by_telemetry_profile"}
    assert context["calibration_validity"] == "not_assessed"
    assert context["fault_assessment"]["status"] == "not_run"
    assert context["state_admission"] == "not_performed"
    assert response(session, "bundle.get", {"bundle_id": bundle["bundle_digest"]}) == bundle


def test_real_socket_replay_retains_stable_batch_and_all_fresh_executions(retained, repositories, tmp_path):
    session = Session.from_workspace(retained["path"], tmp_path)
    session.workbench.bind_workflow("telemetry", repositories)  # Has CBSR bound; original did not request it.
    async def replay():
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            async with connect(f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}") as client:
                await client.recv()
                await client.send(json.dumps({"protocol_version": 1, "request_id": "replay", "type": "bundle.replay",
                                             "payload": {"bundle_id": retained["bundle"]["bundle_digest"]}}))
                reply = json.loads(await asyncio.wait_for(client.recv(), 180))
                assert reply["type"] == "response", reply
                assert json.loads(await asyncio.wait_for(client.recv(), 5))["type"] == "workbench.changed"
                return reply["payload"]
    replayed = asyncio.run(replay())
    new = response(session, "bundle.get", {"bundle_id": replayed["bundle"]["bundle_id"]})
    old = retained["bundle"]
    assert new["steps"][0]["result_id"] == old["steps"][0]["result_id"]
    assert new["steps"][0]["result"] == old["steps"][0]["result"]
    for before, after in zip(old["steps"], new["steps"]):
        assert before["execution_id"] != after["execution_id"]
        assert before["numerical_result_id"] == after["numerical_result_id"]
    executions = response(session, "execution.list")["executions"]
    assert len(executions) == 6
    assert len({e["execution_id"] for e in executions}) == 6
    assert len(response(session, "result.list")["results"]) == 5  # One observation batch, two feature/state occurrences.
    assert len(response(session, "fusion.list")["contexts"]) == 2
    records = session.workbench.serialize()["bundles"]
    changed = deepcopy(records[-1])
    changed["native"]["configuration"]["gsie"]["observation_model"]["model_id"] = "different-declaration-same-numerics"
    with pytest.raises(ValueError, match="configuration"):
        _validate_links(changed, {record["bundle_id"]: record for record in records})
    # Saving/reopening must not lose the second PPDA execution to batch deduplication.
    restored = Session.from_workspace(session.save_workspace(tmp_path / "replayed.json"), tmp_path / "restored")
    assert response(restored, "execution.list") == response(session, "execution.list")
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1", "ciw.project-graph.v1"}


def test_cbsr_consumes_retained_estimate_and_reuses_raw_observations(retained, repositories, tmp_path):
    session = Session.from_workspace(retained["path"], tmp_path)
    session.workbench.bind_workflow("telemetry", repositories)
    config = deepcopy(CONFIG)
    config["cbsr"] = {"state_labels": ["synthetic-state"], "state_units": ["m"], "frame_ref": "stfe-feature:frame:synthetic",
        "constraints": {"constraint_id": "synthetic:exact-target", "coefficients": [[1.0]], "rhs": [2.0],
                        "row_units": ["m"], "coefficient_policy": "declared_exact"},
        "crosscov_policy": "declared", "max_normalized_residual": 4.0}
    summary = execute(session, retained["source"]["source_id"], config)
    bundle = response(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    estimate, reconciled = bundle["steps"][2:]
    assert reconciled["input_refs"] == [estimate["result_id"]]
    assert reconciled["request"]["source_result_id"] == estimate["result_id"]
    assert reconciled["result"]["status"] == "accepted"
    assert bundle["steps"][0]["result_id"] == retained["bundle"]["steps"][0]["result_id"]
    assert response(session, "instrument.inspect", {"bundle_id": summary["bundle_id"], "instrument": "cbsr"})["step"] == reconciled
    view = response(session, "experiment.inspect", {"bundle_id": summary["bundle_id"]})
    panels = {p["panel_id"]: p for p in view["panels"]}
    assert panels["measurements"]["covariance"] == [[1, .25], [.25, 1]]
    assert panels["feature"]["values"] == [3]
    assert panels["state"]["values"] == [pytest.approx(24/13)]
    assert view["fusion_context"]["observability"]["status"] == "unresolved"
    # Existing batch identity cannot be reused for different observation bytes.
    changed = deepcopy(session.workbench.serialize()["bundles"][-1])
    changed["native"]["steps"][0]["result"]["components"][0]["value"] = 999
    with pytest.raises(ValueError, match="collision"):
        session.workbench._check_claims(_claims(changed))


@pytest.mark.parametrize("defect", ["missing-sample", "late", "prior-crosscov", "implicit-calibration"])
def test_bad_window_or_undeclared_transform_publishes_no_partial_state(repositories, tmp_path, defect):
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("telemetry", repositories)
    config = deepcopy(CONFIG)
    if defect == "missing-sample": config["window"]["sample_period"] = .5
    if defect == "late": config["window"]["received_by"] = 0.0
    if defect == "prior-crosscov": config["gsie"]["prior_measurement_crosscov_policy"] = "unknown"
    if defect == "implicit-calibration": config["calibration"] = {"transform": "nonlinear", "order": "after_mean"}
    source = add_source(session)
    result = request(session, "operation.execute", {"operation_id": "ciw.telemetry.v1",
        "parameters": {"source_id": source["source_id"], "configuration": config}})
    assert result["type"] == "error", result
    assert response(session, "bundle.list")["bundles"] == []
    assert response(session, "result.list")["results"] == []
    assert response(session, "source.list")["sources"] == [source]
    assert session.workbench.pending_operations == 0


def test_restore_and_inspection_do_not_execute_and_refuse_config_tamper(retained, tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Read-only restore must not execute or bind a runtime")
    monkeypatch.setattr(telemetry, "_adapters", denied)
    monkeypatch.setattr(telemetry, "_invoke", denied)
    restored = Session.from_workspace(retained["path"], tmp_path)
    assert response(restored, "bundle.get", {"bundle_id": retained["bundle"]["bundle_digest"]}) == retained["bundle"]
    changed = restored.workbench.serialize()
    changed["bundles"][0]["native"]["configuration"]["window"]["end"] = 99
    with pytest.raises(ValueError): Workbench.restore(changed)
    assert request(restored, "instrument.inspect", {"bundle_id": retained["bundle"]["bundle_digest"], "instrument": "fdir"})["type"] == "error"


@pytest.fixture(scope="module")
def esm_configuration(retained, repositories):
    esm = os.environ.get("CIW_ESM_ROOT")
    replay_ciw = os.environ.get("CIW_ESM_REPLAY_CIW")
    if not esm or not replay_ciw:
        pytest.skip("set CIW_ESM_ROOT and CIW_ESM_REPLAY_CIW for actual ESM replay/capture")
    import sys
    node = str(Path(shutil.which("node")).resolve())
    pins = read_json(Path(telemetry.__file__).with_name("telemetry-runtimes.json"))
    pin = read_json(Path(telemetry.__file__).with_name("esm-runtime.json"))
    registration = {"registrationId": "synthetic-telemetry-raw-policy", "sourceId": "synthetic-telemetry-raw",
        "displayName": "Synthetic telemetry fixture only", "sourceClass": "OPERATOR_DECLARATION", "licenseId": "synthetic-test-only",
        "policyVersion": "1", "effectiveFrom": "2026-09-01T00:00:00Z", "allowedOperations": ["RETRIEVE", "DERIVE", "INGEST"],
        "allowedAudiences": ["INTERNAL"], "permittedPurposes": ["SYNTHETIC_TEST"], "retention": {"mode": "INDEFINITE"}}
    return {"node": node, "node_sha256": sha256(Path(node).read_bytes()).hexdigest(),
        "artifact": str(Path(esm) / ".stamp/workbench-candidate.mjs"),
        "runtime": {"python": sys.executable, "pythonSha256": sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "helperPath": str(Path(esm) / "scripts/instrument-replay-verify.py"),
            "repositories": {"ciw": {"path": str(Path(replay_ciw).resolve()), "revision": pin["replay_ciw_revision"]},
                **{role: {"path": str(path.resolve()), "revision": pins[role]["revision"]} for role, path in repositories.items()}}},
        "review_context": {"requestId": "synthetic-telemetry-review", "authority": "role:test-reviewer", "purpose": "SYNTHETIC_TEST",
            "sources": [{"registration": registration, "evidence": [{"artifactRef": e["artifact_ref"], "digest": e["sha256"]}
                         for e in retained["bundle"]["source"]["evidence"]]}], "retractions": []},
        "capture_registration": {**registration, "registrationId": "synthetic-telemetry-derived-policy", "sourceId": "synthetic-telemetry-derived"},
        "store_root": str(retained["root"] / "esm-store")}


def test_telemetry_candidate_replays_and_captures_through_same_session(retained, esm_configuration, tmp_path):
    session = Session.from_workspace(retained["path"], tmp_path)
    session.workbench.bind_candidate_adapter(esm_configuration)
    at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    bundle_id = retained["bundle"]["bundle_digest"]
    inspected = response(session, "operation.execute", {"operation_id": "esm.inspect-candidate.v1",
        "parameters": {"bundle_id": bundle_id, "inspected_at": at}})
    assert inspected["native_response"]["state"] == "ELIGIBLE_FOR_CANDIDATE_REVIEW", inspected["native_response"]["reasons"]
    assert "processAssessment" not in inspected["native_response"]["candidate"]
    assert not Path(esm_configuration["store_root"]).exists()
    capture = response(session, "operation.execute", {"operation_id": "esm.capture-candidate.v1",
        "parameters": {"bundle_id": bundle_id, "evidence_id": "artifact:telemetry-candidate",
                       "workflow_id": "synthetic-telemetry-retain", "retained_at": at}})
    assert capture["native_response"]["state"] == "CANDIDATE_EVIDENCE_RETAINED", capture["native_response"]["reasons"]
    files = [p for p in Path(esm_configuration["store_root"]).rglob("*") if p.is_file()]
    assert len(files) == 1
    envelope = read_json(files[0])
    assert base64.b64decode(envelope["bundleBytesBase64"]) == telemetry.canonical(retained["bundle"])
    assert envelope["state"] == "UNADMITTED" and envelope["canonicalAdmission"] == "REFUSED"
    assert envelope["canonicalStateMutated"] is False and envelope["releaseActivated"] is False
    assert len(response(session, "execution.list")["executions"]) == 5
    path = session.save_workspace(tmp_path / "saved.json")
    restored = Session.from_workspace(path, tmp_path / "restored")
    assert response(restored, "candidate.get", {"candidate_id": capture["candidate_id"]}) == capture
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1", "ciw.project-graph.v1"}
    # Restore cannot silently treat legacy telemetry as a calibrated process.
    saved = read_json(path)["workbench"]
    changed = saved["candidates"][0]
    native = json.loads(base64.b64decode(changed["response_bytes_b64"]))
    native["candidate"]["processAssessment"] = {"observabilityStatus": "observable"}
    changed["response_bytes_b64"] = base64.b64encode(json.dumps(native).encode()).decode()
    changed["candidate_id"] = "candidate:" + telemetry.digest({k: v for k, v in changed.items() if k != "candidate_id"})
    with pytest.raises(ValueError, match="inherit"):
        Workbench.restore(saved)


def test_provider_families_can_coexist_without_overwriting_esm_bindings(retained, esm_configuration, tmp_path):
    from ciw.candidate_evidence import CandidateAdapter
    session = Session.from_workspace(retained["path"], tmp_path)
    session.workbench.bind_candidate_adapter(esm_configuration)
    assert session.workbench.describe_operations()[-1]["available_bundle_kinds"] == ["telemetry"]
    invalid = deepcopy(esm_configuration)
    invalid["runtime"]["repositories"]["fsrt"] = invalid["runtime"]["repositories"]["ppda"]
    with pytest.raises(ValueError): CandidateAdapter(invalid)
    assert session.workbench.describe_operations()[-1]["available_bundle_kinds"] == ["telemetry"]
    runtime_path, bundle_path = os.environ.get("CIW_ESM_RUNTIME_FILE"), os.environ.get("CIW_ESM_BUNDLE_FILE")
    if not runtime_path or not bundle_path:
        pytest.skip("set CIW_ESM_RUNTIME_FILE and CIW_ESM_BUNDLE_FILE for both native lanes together")
    calibrated = read_json(bundle_path)
    raw = base64.b64decode(calibrated["source"]["evidence"][0]["bytes_b64"])
    source = response(session, "source.add", {"kind": "calibrated-observable", "label": "Synthetic two-channel experiment",
                                             "bytes_b64": base64.b64encode(raw).decode()})
    session.workbench._retain("calibrated-observable", session.workbench.get_source(source["source_id"]), None, calibrated)
    configuration = deepcopy(esm_configuration)
    configuration["runtime"] = read_json(runtime_path)
    configuration["review_context"]["sources"][0]["evidence"] = [
        {"artifactRef": e["artifact_ref"], "digest": e["sha256"]} for e in calibrated["source"]["evidence"]]
    session.workbench.bind_candidate_adapter(configuration)
    assert session.workbench.describe_operations()[-1]["available_bundle_kinds"] == ["calibrated-observable", "telemetry"]
    assert len(response(session, "fusion.list")["contexts"]) == 2
    assert {c["state_kind"] for c in response(session, "fusion.list")["contexts"]} == {"posterior", "window_feature_posterior"}
    design_source = response(session, "source.add", {"kind": "identified-design", "label": "Synthetic next observation",
        "bytes_b64": base64.b64encode((ROOT / "examples/identified-design/source.json").read_bytes()).decode()})
    refused = request(session, "operation.execute", {"operation_id": "ciw.identified-design.v1", "parameters": {
        "source_id": design_source["source_id"], "upstream_bundle_id": retained["bundle"]["bundle_digest"]}})
    assert refused["type"] == "error"  # Ungated window state cannot satisfy the calibrated prior contract.
    restored = Session.from_workspace(session.save_workspace(tmp_path / "both.json"), tmp_path / "both-restored")
    assert len(response(restored, "bundle.list")["bundles"]) == 2
    assert response(restored, "fusion.list") == response(session, "fusion.list")
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1", "ciw.project-graph.v1"}
