"""Exercise both reference providers through one live retained session."""
import asyncio
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw.cli import parser
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"flat-torus-reference": "ftr", "curved-path-transfer": "csg"}


def call(session, kind, payload=None, *, error=False):
    response = session.handle({"protocol_version": 1, "request_id": "reference-gate",
        "type": kind, "payload": payload or {}})
    assert response["type"] == ("error" if error else "response"), response
    return response["payload"]


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    paths = {role: os.environ.get("CIW_" + role.upper() + "_REPO") for role in KINDS.values()}
    if not all(paths.values()):
        pytest.skip("Set CIW_FTR_REPO and CIW_CSG_REPO to exact native checkouts")
    session = Session(make_demo_run(), tmp_path_factory.mktemp("reference-session"))
    for kind, role in KINDS.items():
        session.workbench.bind_workflow(kind, {role: paths[role]})

    async def live():
        client = runpy.run_path(str(ROOT / "examples/geodesic-reference/run.py"))["run"]
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, max_size=16_777_216, proxy=None) as observer:
                assert json.loads(await observer.recv())["type"] == "session.snapshot"
                result = await client(url)
                events = [json.loads(await asyncio.wait_for(observer.recv(), 5)) for _ in range(6)]
                assert all(event["type"] == "workbench.changed" for event in events)
                assert all(event["payload"]["session_id"] == session.session_id for event in events)
                return result

    result = asyncio.run(live())
    destination = os.environ.get("CIW_GEODESIC_FIXTURE_DIR")
    if destination:
        for kind, refs in result["references"].items():
            folder = Path(destination) / kind
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("original", "replay"):
                native = call(session, "bundle.get", {"bundle_id": refs[name]})
                (folder / (name + ".json")).write_bytes(canonical(native))
            (folder / "source.json").write_bytes(base64.b64decode(native["source"]["evidence"][0]["bytes_b64"]))
        session.save_workspace(Path(destination) / "workspace.json")
    return session, result


def test_reference_startup_and_unbound_operations(tmp_path):
    args = parser().parse_args(["serve", "--flat-torus-repo", "ftr", "--curved-surface-repo", "csg"])
    assert args.flat_torus_repo == Path("ftr") and args.curved_surface_repo == Path("csg")
    session = Session(make_demo_run(), tmp_path)
    operations = {o["source_kind"]: o for o in session.workbench.describe_operations() if "source_kind" in o}
    for kind in KINDS:
        assert operations[kind]["available"] is False
        assert operations[kind]["role"] in {"geometric_reference", "geometric_sensitivity"}


def test_native_references_share_history_without_becoming_fusion_state(retained):
    session, result = retained
    executions = {e["execution_id"]: e for e in call(session, "execution.list")["executions"]}
    assert len(call(session, "bundle.list")["bundles"]) == 4
    assert call(session, "fusion.list")["contexts"] == []
    for kind, refs in result["references"].items():
        original = call(session, "bundle.get", {"bundle_id": refs["original"]})
        replay = call(session, "bundle.get", {"bundle_id": refs["replay"]})
        assert original["source"] == replay["source"]
        assert original["bundle_digest"] != replay["bundle_digest"]
        assert original["steps"][0]["numerical_result"] == replay["steps"][0]["numerical_result"]
        assert original["steps"][0]["execution_id"] != replay["steps"][0]["execution_id"]
        step = original["steps"][0]
        assert executions[step["execution_id"]]["bundle_id"] == refs["original"]
        assert call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
        view = refs["view"]
        assert view["kind"] == kind and view["fusion_context"] is None
        assert view["authority"]["read_only"] is True
        assert view["authority"]["state_admission"] == "not_performed"
        native = call(session, "instrument.inspect", {"bundle_id": refs["original"], "instrument": KINDS[kind]})
        assert native["step"] == step
        assert view["panels"]
    assert session.workbench.pending_operations == 0


def test_reference_restore_needs_no_provider_and_replay_refuses_without_binding(retained, tmp_path, monkeypatch):
    from ciw.geodesic_reference import GeodesicReferenceWorkflow
    session, result = retained
    workspace = session.save_workspace(tmp_path / "references.json")

    def no_provider(*args, **kwargs):
        raise AssertionError("Inspection must not bind or execute providers")

    monkeypatch.setattr(GeodesicReferenceWorkflow, "_adapters", no_provider)
    restored = Session.from_workspace(workspace, tmp_path / "restored")
    assert restored.workbench.serialize() == session.workbench.serialize()
    for kind, refs in result["references"].items():
        before = deepcopy(restored.workbench.serialize())
        assert call(restored, "experiment.inspect", {"bundle_id": refs["original"]})["kind"] == kind
        call(restored, "bundle.replay", {"bundle_id": refs["original"]}, error=True)
        assert restored.workbench.serialize() == before
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1"}
