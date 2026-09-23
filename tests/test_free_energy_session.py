"""Live client, shared catalog, read-only views and offline workspace restore."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw.cli import parser
from ciw.free_energy_workflow import FreeEnergyWorkflow
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from test_free_energy_workflow import native_experiments

ROOT = Path(__file__).resolve().parents[1]


def call(session,kind,payload=None,*,error=False):
    answer = session.handle({"protocol_version":1,"request_id":"free-energy-gate","type":kind,"payload":payload or {}})
    assert answer["type"] == ("error" if error else "response"),answer
    return answer["payload"]


@pytest.fixture(scope="module")
def retained(native_experiments,tmp_path_factory):
    session = Session(make_demo_run(),tmp_path_factory.mktemp("free-energy-session"))
    session.workbench.bind_workflow("variational-free-energy",native_experiments["bindings"])
    async def live():
        client = runpy.run_path(str(ROOT / "examples/variational-free-energy/run.py"))["run"]
        async with serve(WorkbenchServer(session).handler,"127.0.0.1",0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url,max_size=32*1024*1024,proxy=None) as observer:
                assert json.loads(await observer.recv())["type"] == "session.snapshot"
                result = await client(url)
                events = [json.loads(await asyncio.wait_for(observer.recv(),5)) for _ in range(3)]
                assert all(e["type"] == "workbench.changed" for e in events)
                return result
    result = asyncio.run(live())
    destination = os.environ.get("CIW_FREE_ENERGY_FIXTURE_DIR")
    if destination:
        session.save_workspace(Path(destination) / "workspace.json")
    return session,result


def test_free_energy_startup_is_explicit_and_unbound(tmp_path):
    args = parser().parse_args(["serve","--free-energy-stack-root","providers"])
    assert args.free_energy_stack_root == Path("providers")
    operation = next(row for row in Session(make_demo_run(),tmp_path).workbench.describe_operations() if row.get("source_kind") == "variational-free-energy")
    assert operation["available"] is False
    assert operation["role"] == "variational_inference"


def test_shared_session_exposes_three_native_stages_and_diagnostics(retained):
    session,result = retained
    refs = result["experiments"]["baseline"]
    assert len(call(session,"bundle.list")["bundles"]) == 2
    assert call(session,"fusion.list")["contexts"] == []
    original = call(session,"bundle.get",{"bundle_id":refs["original"]})
    instruments = [entry for entry in session.workbench.instrument_views() if entry["bundle_id"] == refs["original"]]
    assert {entry["instrument"] for entry in instruments} == {"csg","gsie","plsr"}
    assert len(instruments) == 3
    for stage in original["steps"][0]["result"]["data"]["stages"]:
        assert call(session,"instrument.inspect",{"bundle_id":refs["original"],"instrument":stage["runtime_ref"]})["step"] == stage
        assert call(session,"result.get",{"result_id":stage["result_id"]}) == stage["result"]
    view = refs["view"]
    assert view["fusion_context"] is None
    assert view["authority"]["state_admission"] == "not_performed"
    assert len(view["panels"]) == 7
    assert len(view["graph"]["nodes"]) == 3
    assert view["object_context"]["optimizer_status"] == "converged"
    assert session.workbench.pending_operations == 0


def test_saved_workspace_reopens_without_runtime_or_numerical_execution(retained,tmp_path,monkeypatch):
    session,result = retained
    saved = session.save_workspace(tmp_path / "workspace.json")
    monkeypatch.setattr(FreeEnergyWorkflow,"_adapters",lambda *_: pytest.fail("Offline restore bound a native provider"))
    restored = Session.from_workspace(saved,tmp_path / "restored")
    assert restored.workbench.serialize() == session.workbench.serialize()
    refs = result["experiments"]["baseline"]
    for key in ("original","replay"):
        before = deepcopy(restored.workbench.serialize())
        assert call(restored,"experiment.inspect",{"bundle_id":refs[key]}) == call(session,"experiment.inspect",{"bundle_id":refs[key]})
        call(restored,"bundle.replay",{"bundle_id":refs[key]},error=True)
        assert restored.workbench.serialize() == before
