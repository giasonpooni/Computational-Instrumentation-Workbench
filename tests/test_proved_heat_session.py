"""Shared proof operation, retained-only inspection, and real Linux proof gate."""
import asyncio
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw.adapters.protocol import AdapterRefusal
from ciw.cli import main, parser, request_remote
from ciw.instruments import make_demo_run
from ciw.proved_heat import ProvedHeatWorkflow, TRUST_SCOPE, _verification
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.core.canonical import canonical, digest, byte_digest, bundle_digest

ROOT = Path(__file__).resolve().parents[1]


def call(session, kind, payload=None, *, error=False):
    response = session.handle({"protocol_version": 1, "request_id": "proof-gate", "type": kind,
                               "payload": payload or {}})
    assert response["type"] == ("error" if error else "response"), response
    return response["payload"]


def add_source(session):
    raw = (ROOT / "examples/proved-heat/source.json").read_bytes()
    return call(session, "source.add", {"kind": "proved-heat", "label": "Declared integer heat proof",
        "bytes_b64": base64.b64encode(raw).decode()})


def test_proof_operation_requires_explicit_host_binding(tmp_path):
    args = parser().parse_args(["serve", "--computation-repo", "scr", "--computation-engine", "engine",
                               "--sp1-prover", "prover", "--sp1-heat-guest", "guest"])
    assert args.sp1_prover == Path("prover") and args.sp1_heat_guest == Path("guest")
    session = Session(make_demo_run(), tmp_path)
    source = add_source(session)
    operation = next(o for o in call(session, "operation.list")["operations"] if o["operation_id"] == "ciw.proved-heat.v1")
    assert operation["available"] is False
    assert operation["role"] == "proved_numerical_execution"
    call(session, "operation.execute", {"operation_id": operation["operation_id"],
         "parameters": {"source_id": source["source_id"]}}, error=True)
    assert call(session, "bundle.list")["bundles"] == []
    assert call(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []


@pytest.mark.parametrize("destination_state", ["new", "already_exists", "appears_during_verification"])
def test_proof_report_publication_preserves_other_occurrences(tmp_path, monkeypatch, destination_state):
    # A verifier double isolates report publication; it authenticates no proof.
    from test_proved_heat import fake_bundle
    bundle = fake_bundle()
    retained = tmp_path / "retained.json"
    retained.write_bytes(canonical(bundle))
    destination = tmp_path / "reports" / "verification.json"
    existing = b'{"verification_occurrence":"another-writer"}\n'
    report = {"verification_occurrence": "publication-test-double"}
    calls = []

    if destination_state == "already_exists":
        destination.parent.mkdir()
        destination.write_bytes(existing)

    def verify(self, value, bindings):
        assert value == bundle
        calls.append(bindings)
        if destination_state == "appears_during_verification":
            destination.parent.mkdir()
            destination.write_bytes(existing)
        return report

    monkeypatch.setattr(ProvedHeatWorkflow, "verify_session", verify)
    code = main(["proof", "verify", str(retained), "--computation-repo", "scr",
                 "--computation-engine", "engine", "--sp1-prover", "prover",
                 "--sp1-heat-guest", "guest", "--output", str(destination)])
    if destination_state == "new":
        assert code == 0
        assert json.loads(destination.read_text(encoding="utf-8")) == report
    else:
        assert code == 2
        assert destination.read_bytes() == existing
    assert len(calls) == (0 if destination_state == "already_exists" else 1)
    assert not list(destination.parent.glob(".ciw-proof-report-*"))


def test_historical_proof_session_restore_and_replay_bindings(tmp_path, monkeypatch):
    from test_proved_heat import fake_runtime, fake_data
    monkeypatch.setattr(ProvedHeatWorkflow, "_adapters", lambda *args: (None, fake_runtime(), {}))
    monkeypatch.setattr(ProvedHeatWorkflow, "_invoke", lambda self, source, *args: fake_data(source))
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("proved-heat", {role: tmp_path / role for role in ProvedHeatWorkflow.ROLES})
    source = add_source(session)
    original = call(session, "operation.execute", {"operation_id": "ciw.proved-heat.v1",
        "parameters": {"source_id": source["source_id"]}})
    replay = call(session, "bundle.replay", {"bundle_id": original["bundle_id"]})
    assert replay["bundle"]["bundle_id"] != original["bundle_id"]
    saved = session.save_workspace(tmp_path / "synthetic-structural-fixture.json")
    monkeypatch.setattr(ProvedHeatWorkflow, "_invoke", lambda *args: pytest.fail("Inspection called provider"))
    restored = Session.from_workspace(saved, tmp_path / "restored")
    for bundle in call(restored, "bundle.list")["bundles"]:
        assert bundle["cryptographic_verification"] == "not_performed_by_inspection"
        view = call(restored, "experiment.inspect", {"bundle_id": bundle["bundle_id"]})
        assert view["panels"][1]["values"] == [0, 65, 92, 65, 0]
        assert view["object_context"]["verification_trust_scope"] == TRUST_SCOPE
    assert call(restored, "experiment.inspect", {"view": "fusion"})["contexts"] == []
    call(restored, "bundle.replay", {"bundle_id": original["bundle_id"]}, error=True)


def test_terminal_receives_supported_large_proof_envelope():
    # Transport-only payload; this test does not create or authenticate a proof.
    encoded = base64.b64encode(b"synthetic transport data " * 320000).decode()

    async def handler(socket):
        request = json.loads(await socket.recv())
        await socket.send(json.dumps({"protocol_version": 1, "type": "response",
            "request_id": request["request_id"], "payload": {"proof_bytes_b64": encoded}}))

    async def exercise():
        async with serve(handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            result = await request_remote(url, "bundle.get", {"bundle_id": "transport-only"})
            assert result["payload"]["proof_bytes_b64"] == encoded
    asyncio.run(exercise())


def test_proving_refusal_does_not_block_other_clients_or_publish_result(tmp_path, monkeypatch):
    # Delay a failing provider boundary only; this produces no substitute proof.
    started, release = threading.Event(), threading.Event()

    def refuse(self, raw, bindings):
        started.set()
        if not release.wait(5):
            raise AssertionError("test failed to release the provider")
        raise AdapterRefusal("PROVED_HEAT_REFUSED", "deliberate provider refusal")

    monkeypatch.setattr(ProvedHeatWorkflow, "_adapters", lambda *args: None)
    monkeypatch.setattr(ProvedHeatWorkflow, "create_session", refuse)
    session = Session(make_demo_run(), tmp_path)
    session.workbench.bind_workflow("proved-heat", {role: tmp_path / role for role in ProvedHeatWorkflow.ROLES})
    source = add_source(session)

    async def exercise():
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, proxy=None) as submitter, connect(url, proxy=None) as observer:
                await submitter.recv()
                await observer.recv()
                try:
                    await submitter.send(json.dumps({"protocol_version": 1, "request_id": "prove",
                        "type": "operation.execute", "payload": {"operation_id": "ciw.proved-heat.v1",
                        "parameters": {"source_id": source["source_id"]}}}))
                    assert await asyncio.to_thread(started.wait, 2)
                    await observer.send(json.dumps({"protocol_version": 1, "request_id": "inspect",
                        "type": "session.get", "payload": {}}))
                    response = json.loads(await asyncio.wait_for(observer.recv(), 2))
                    assert response["request_id"] == "inspect" and response["type"] == "response"
                    assert session.workbench.pending_operations == 1
                    assert call(session, "bundle.list")["bundles"] == []
                finally:
                    release.set()
                refused = json.loads(await asyncio.wait_for(submitter.recv(), 5))
                assert refused["type"] == "error" and refused["payload"]["code"] == "PROVED_HEAT_REFUSED"
                assert session.workbench.pending_operations == 0
                assert call(session, "bundle.list")["bundles"] == []
    asyncio.run(exercise())


@pytest.fixture(scope="module")
def native_retained(tmp_path_factory):
    names = {"scr": "CIW_SCR_PROOF_REPO", "engine": "CIW_SCR_PROOF_ENGINE",
             "prover": "CIW_SP1_PROVER", "guest": "CIW_SP1_HEAT_GUEST"}
    bindings = {role: os.environ.get(name) for role, name in names.items()}
    if not all(bindings.values()):
        pytest.skip("Real SP1 gate requires SCR, native engine, CPU host and registered heat ELF")
    session = Session(make_demo_run(), tmp_path_factory.mktemp("native-proof"))
    session.workbench.bind_workflow("proved-heat", bindings)
    source = add_source(session)
    summary = call(session, "operation.execute", {"operation_id": "ciw.proved-heat.v1",
                   "parameters": {"source_id": source["source_id"]}})
    original = call(session, "bundle.get", {"bundle_id": summary["bundle_id"]})
    replayed = call(session, "bundle.replay", {"bundle_id": summary["bundle_id"]})
    fresh = call(session, "bundle.get", {"bundle_id": replayed["bundle"]["bundle_id"]})
    destination = os.environ.get("CIW_PROVED_HEAT_FIXTURE_DIR")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / "source.json").write_bytes(base64.b64decode(original["source"]["evidence"][0]["bytes_b64"]))
        for name, bundle in (("original", original), ("replay", fresh)):
            (path / (name + ".json")).write_bytes(canonical(bundle))
        session.save_workspace(path / "workspace.json")
    return session, original, fresh, bindings


def test_native_proved_heat_shared_session(native_retained, tmp_path, monkeypatch):
    session, original, fresh, _ = native_retained
    for bundle in (original, fresh):
        data = bundle["steps"][0]["result"]["data"]
        assert data["native"]["values"] == [0, 65, 92, 65, 0]
        assert data["verifier"]["outcome"] == "verified"
        assert data["verifier"]["backend"] == "sp1-cpu v6.1.0"
        memory = data["timings"]["memory"]
        assert memory["scope"] == "max_waited_child_peak_rss" and memory["unit"] == "byte"
        if os.sys.platform == "linux":
            assert memory["status"] == "measured" and memory["bytes"] > 0
        assert bundle["verification"]["trust_scope"] == TRUST_SCOPE
    assert original["steps"][0]["numerical_result_id"] == fresh["steps"][0]["numerical_result_id"]
    assert original["steps"][0]["execution_id"] != fresh["steps"][0]["execution_id"]
    view = call(session, "experiment.inspect", {"bundle_id": original["bundle_digest"]})
    assert view["fusion_context"] is None
    assert view["panels"][1]["values"] == [0, 65, 92, 65, 0]
    assert view["authority"]["cryptographic_verification"] == "not_performed_by_inspection"
    assert "bytes_b64" not in view["object_context"]["proof"]
    assert call(session, "experiment.inspect", {"view": "fusion"})["contexts"] == []
    saved = session.save_workspace(tmp_path / "proof-workspace.json")

    def never(*args, **kwargs):
        raise AssertionError("Offline restore/inspection must not execute or reverify")

    monkeypatch.setattr(ProvedHeatWorkflow, "_invoke", never)
    restored = Session.from_workspace(saved, tmp_path / "restored")
    assert call(restored, "bundle.get", {"bundle_id": original["bundle_digest"]}) == original
    assert call(restored, "experiment.inspect", {"bundle_id": original["bundle_digest"]}) == view
    assert not any(o["available"] for o in restored.workbench.describe_operations() if o.get("source_kind") == "proved-heat")


def test_native_tampered_proof_fails_fresh_verification(native_retained):
    # Recompute all JSON commitments to ensure rejection comes from SP1, not a stale seal.
    from ciw.declared_workload import _commit
    _, original, _, bindings = native_retained
    tampered = deepcopy(original)
    step = tampered["steps"][0]
    data = step["result"]["data"]
    proof = data["proof"]
    raw = bytearray(base64.b64decode(proof["bytes_b64"]))
    raw[-1] ^= 1
    raw = bytes(raw)
    proof["bytes_b64"] = base64.b64encode(raw).decode()
    proof["sha256"] = byte_digest(raw)
    proof["identity"] = _commit("proof", [b"sp1-cpu", b"v6.1.0", raw])
    data["verifier"]["proof_identity"] = proof["identity"]
    step["result"]["result_id"] = digest({k: v for k, v in step["result"].items() if k != "result_id"})
    step["result_id"] = step["result"]["result_id"]
    step["result_sha256"] = digest(step["result"])
    tampered["bundle_digest"] = bundle_digest(tampered)
    tampered["verification"] = _verification(tampered, original["verification"]["verification_operation_id"])
    workflow = ProvedHeatWorkflow()
    workflow._validate(tampered)  # Structural consistency deliberately grants no cryptographic authority.
    with pytest.raises(AdapterRefusal, match="verification failed"):
        workflow.verify_session(tampered, bindings)


def test_native_retained_proof_can_be_reverified_without_reexecution(native_retained, monkeypatch, tmp_path):
    from ciw.cli import main
    _, original, _, bindings = native_retained
    monkeypatch.setattr(ProvedHeatWorkflow, "_step", lambda *args: pytest.fail("Reverification must not execute/prove"))
    bundle_path, report_path = tmp_path / "bundle.json", tmp_path / "verified.json"
    bundle_path.write_bytes(canonical(original))
    command = ["proof", "verify", str(bundle_path), "--computation-repo", bindings["scr"],
        "--computation-engine", bindings["engine"], "--sp1-prover", bindings["prover"],
        "--sp1-heat-guest", bindings["guest"], "--output", str(report_path)]
    assert main(command) == 0
    report = json.loads(report_path.read_bytes())
    assert main(command) == 2  # Previous verification occurrence cannot be overwritten.
    assert report["subject_ref"] == original["bundle_digest"]
    assert report["proof_identity"] == original["verification"]["proof_identity"]
    assert report["verifier_runtime_digest"] == digest(report["verifier_runtimes"])
    if os.sys.platform == "linux":
        assert report["memory"]["status"] == "measured" and report["memory"]["bytes"] > 0
    assert report["verification_operation_id"] != original["verification"]["verification_operation_id"]
    destination = os.environ.get("CIW_PROVED_HEAT_FIXTURE_DIR")
    if destination:
        (Path(destination) / "reverification.json").write_bytes(canonical(report))
