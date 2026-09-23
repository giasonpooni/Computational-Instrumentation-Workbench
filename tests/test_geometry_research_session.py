"""Three real mathematical providers in one live, retained workbench session."""
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
from ciw.geometry_research import GeometryResearchWorkflow, PINS
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]


def call(session, kind, payload=None, *, error=False):
    answer = session.handle({"protocol_version":1,"request_id":"geometry-gate","type":kind,"payload":payload or {}})
    assert answer["type"] == ("error" if error else "response"), answer
    return answer["payload"]


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    paths = {pin["role"]:os.environ.get("CIW_" + pin["role"].upper() + "_REPO") for pin in PINS.values()}
    if not all(paths.values()):
        pytest.skip("Native geometry session requires exact CGGT, ISGT and TSDE checkouts")
    session = Session(make_demo_run(), tmp_path_factory.mktemp("geometry-session"))
    for kind,pin in PINS.items():
        session.workbench.bind_workflow(kind, {pin["role"]:paths[pin["role"]]})
    async def live():
        client = runpy.run_path(str(ROOT / "examples/geometry-research/run.py"))["run"]
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, max_size=32*1024*1024, proxy=None) as observer:
                assert json.loads(await observer.recv())["type"] == "session.snapshot"
                result = await client(url)
                events = [json.loads(await asyncio.wait_for(observer.recv(), 5)) for _ in range(9)]
                assert all(e["type"] == "workbench.changed" for e in events)
                return result
    result = asyncio.run(live())
    destination = os.environ.get("CIW_GEOMETRY_FIXTURE_DIR")
    if destination:
        for kind,refs in result["references"].items():
            folder = Path(destination) / kind
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("original", "replay"):
                bundle = call(session, "bundle.get", {"bundle_id":refs[name]})
                (folder / (name + ".json")).write_bytes(canonical(bundle))
            (folder / "source.json").write_bytes(base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"]))
        session.save_workspace(Path(destination) / "workspace.json")
    return session,result


def test_geometry_startup_and_unbound_operations(tmp_path):
    args = parser().parse_args(["serve", "--covariance-geometry-repo", "cggt", "--intrinsic-surface-repo", "isgt", "--translation-surface-repo", "tsde"])
    assert args.covariance_geometry_repo == Path("cggt")
    assert args.intrinsic_surface_repo == Path("isgt")
    assert args.translation_surface_repo == Path("tsde")
    operations = {row["source_kind"]:row for row in Session(make_demo_run(),tmp_path).workbench.describe_operations() if "source_kind" in row}
    for kind in PINS:
        assert operations[kind]["available"] is False
        assert operations[kind]["role"] in {"covariance_geometry", "mesh_path_baseline", "translation_dynamics"}


def test_three_native_providers_share_catalog_and_remain_outside_fusion(retained):
    session,result = retained
    assert len(call(session,"bundle.list")["bundles"]) == 6
    assert call(session,"fusion.list")["contexts"] == []
    for kind,refs in result["references"].items():
        original = call(session,"bundle.get",{"bundle_id":refs["original"]})
        replay = call(session,"bundle.get",{"bundle_id":refs["replay"]})
        old,fresh = original["steps"][0],replay["steps"][0]
        assert old["execution_id"] != fresh["execution_id"]
        assert old["numerical_result_id"] == fresh["numerical_result_id"]
        assert refs["view"]["fusion_context"] is None
        assert refs["view"]["authority"]["state_admission"] == "not_performed"
        assert refs["view"]["panels"]
        assert call(session,"instrument.inspect",{"bundle_id":refs["original"],"instrument":PINS[kind]["role"]})["step"] == old
        assert call(session,"result.get",{"result_id":old["result_id"]}) == old["result"]
    assert session.workbench.pending_operations == 0


def test_geometry_offline_restore_never_binds_provider(retained,tmp_path,monkeypatch):
    session,result = retained
    saved = session.save_workspace(tmp_path / "workspace.json")
    monkeypatch.setattr(GeometryResearchWorkflow,"_adapters",lambda *_: pytest.fail("Offline inspection executed a provider"))
    restored = Session.from_workspace(saved,tmp_path / "restored")
    assert restored.workbench.serialize() == session.workbench.serialize()
    for kind,refs in result["references"].items():
        before = deepcopy(restored.workbench.serialize())
        assert call(restored,"experiment.inspect",{"bundle_id":refs["original"]}) == call(session,"experiment.inspect",{"bundle_id":refs["original"]})
        call(restored,"bundle.replay",{"bundle_id":refs["original"]},error=True)
        assert restored.workbench.serialize() == before
