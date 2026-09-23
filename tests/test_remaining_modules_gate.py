"""Remaining native modules share one live catalog and retain their authority."""
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

from ciw import calibrated_observable, identified_design, measurement_chain, geometric_circle, identified_stability
from ciw.cli import parser, request_remote
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.telemetry import canonical


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("measurement-chain", "geometric-circle", "identified-stability")


def call(session, kind, payload=None, *, error=False):
    response = session.handle({"protocol_version": 1, "request_id": "remaining-modules-gate",
                               "type": kind, "payload": payload or {}})
    assert response["type"] == ("error" if error else "response"), response
    return response["payload"]


def output(kind, name, value):
    directory = os.environ.get("CIW_REMAINING_FIXTURE_DIR")
    if directory:
        target = Path(directory) / kind
        target.mkdir(parents=True, exist_ok=True)
        (target / (name + ".json")).write_bytes(value if isinstance(value, bytes) else canonical(value))


def source_bytes(bundle):
    return base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"], validate=True)


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    variables = ("CIW_REMAINING_MEASUREMENT_STACK_ROOT", "CIW_REMAINING_IDENTIFIED_STACK_ROOT",
                 "CIW_REMAINING_GTE_REPO", "CIW_REMAINING_PLSR_REPO")
    if not all(os.environ.get(name) for name in variables):
        pytest.skip("set CIW_REMAINING_* exact native provider paths for the installed common bench gate")
    measurement, identified, geometry, stability = (Path(os.environ[name]) for name in variables)
    directory = tmp_path_factory.mktemp("remaining-modules")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("calibrated-observable", {role: identified / role for role in calibrated_observable.ROLES})
    session.workbench.bind_workflow("identified-design", {role: identified / role for role in identified_design.ROLES})
    session.workbench.bind_workflow("measurement-chain", {role: measurement / role for role in measurement_chain.ROLES})
    session.workbench.bind_workflow("geometric-circle", {"gte": geometry})
    session.workbench.bind_workflow("identified-stability", {"plsr": stability})

    async def exercise_client():
        bridge = WorkbenchServer(session)
        client = runpy.run_path(str(ROOT / "examples/remaining-modules/run.py"))["run"]
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, max_size=16_777_216, proxy=None) as observer:
                snapshot = json.loads(await observer.recv())
                assert snapshot["type"] == "session.snapshot"
                result = await client(url)
                events = [json.loads(await asyncio.wait_for(observer.recv(), 5)) for _ in range(13)]
                assert all(event["type"] == "workbench.changed" for event in events)
                assert all(event["payload"]["session_id"] == session.session_id for event in events)
                remote = await request_remote(url, "bundle.list", {}, timeout_s=30)
                assert remote["type"] == "response"
                assert len(remote["payload"]["bundles"]) == 8
                return result

    result = asyncio.run(exercise_client())
    originals = {kind: call(session, "bundle.get", {"bundle_id": result["bundle_ids"][kind]}) for kind in KINDS}
    replays = {kind: call(session, "bundle.get", {"bundle_id": result["replay_bundle_ids"][kind]}) for kind in KINDS}
    for kind in KINDS:
        output(kind, "source", source_bytes(originals[kind]))
        output(kind, "original", originals[kind])
        output(kind, "replay", replays[kind])
    upstream = call(session, "bundle.get", {"bundle_id": result["identified_bundle_id"]})
    output("identified-stability", "upstream", upstream)
    output("identified-stability", "upstream-source", source_bytes(upstream))
    return {"session": session, "result": result, "originals": originals, "replays": replays,
            "upstream": upstream, "directory": directory}


def test_remaining_startup_requires_explicit_native_bindings():
    args = parser().parse_args(["serve", "--measurement-chain-stack-root", "/trusted/measurement",
                               "--geometry-repo", "/trusted/gte", "--stability-repo", "/trusted/plsr"])
    assert args.measurement_chain_stack_root == Path("/trusted/measurement")
    assert args.geometry_repo == Path("/trusted/gte")
    assert args.stability_repo == Path("/trusted/plsr")


def test_one_catalog_resolves_all_retained_occurrences_and_native_views(retained):
    session = retained["session"]
    catalog = {row["bundle_id"]: row for row in call(session, "bundle.list")["bundles"]}
    executions = {row["execution_id"]: row for row in call(session, "execution.list")["executions"]}
    for kind, bundle in retained["originals"].items():
        assert catalog[bundle["bundle_digest"]]["kind"] == kind
        view = retained["result"]["views"][kind]
        assert view["kind"] == kind
        assert view["fusion_context"] is None
        assert isinstance(view["object_context"]["object_kind"], str)
        assert view["authority"]["read_only"] is True
        assert view["authority"]["state_admission"] == "not_performed"
        for step in bundle["steps"]:
            assert call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
            occurrence = executions[step["execution_id"]]
            assert occurrence["bundle_id"] == bundle["bundle_digest"]
            assert occurrence["source_id"] == catalog[bundle["bundle_digest"]]["source_id"]
    contexts = call(session, "fusion.list")["contexts"]
    assert {context["bundle_id"] for context in contexts} == {
        retained["result"]["calibrated_bundle_id"], retained["result"]["identified_bundle_id"]}
    assert session.workbench.pending_operations == 0


def test_replay_preserves_sources_and_numerics_with_fresh_occurrences(retained):
    for kind, original in retained["originals"].items():
        fresh = retained["replays"][kind]
        assert original["bundle_digest"] != fresh["bundle_digest"]
        assert original["session_id"] != fresh["session_id"]
        assert source_bytes(original) == source_bytes(fresh)
        assert original["configuration"] == fresh["configuration"]
        for before, after in zip(original["steps"], fresh["steps"], strict=True):
            assert before["numerical_result_id"] == after["numerical_result_id"]
            assert before["execution_id"] != after["execution_id"]
            assert before["result_id"] != after["result_id"]
        assert fresh["replay_receipts"][0]["numerical_match"] is True


def test_native_held_outcomes_retain_their_original_meaning(retained):
    session = retained["session"]
    measurement = json.loads(source_bytes(retained["originals"]["measurement-chain"]))
    measurement["investigation"]["model"]["total_mass_kg"] = 150.0
    geometry = json.loads((ROOT / "examples/geometric-circle/held.json").read_bytes())
    stability = runpy.run_path(str(ROOT / "examples/identified-stability/make_source.py"))["make_source"](
        retained["upstream"], level=0)
    held = {}
    for kind, declaration in zip(KINDS, (measurement, geometry, stability), strict=True):
        raw = canonical(declaration)
        source = call(session, "source.add", {"kind": kind, "label": "Public synthetic held " + kind,
            "bytes_b64": base64.b64encode(raw).decode("ascii")})
        parameters = {"source_id": source["source_id"]}
        if kind == "identified-stability":
            parameters["upstream_bundle_id"] = retained["upstream"]["bundle_digest"]
        result = call(session, "operation.execute", {"operation_id": "ciw." + kind + ".v1", "parameters": parameters})
        held[kind] = call(session, "bundle.get", {"bundle_id": result["bundle_id"]})
        view = call(session, "experiment.inspect", {"bundle_id": result["bundle_id"]})
        assert view["fusion_context"] is None
        output(kind, "held-source", raw)
        output(kind, "held", held[kind])
    measurement_native = held["measurement-chain"]["steps"][0]["result"]["data"]["native_workspace"]
    measurement_data = measurement_native["results"][0]["data"]
    assert measurement_data["diagnostics"]["reconciliation_status"] == "model_inconsistent"
    for field in ("values", "covariance", "unit"):
        assert measurement_data["estimate"][field] == measurement_data["unprojected_estimate"][field]
    assert held["geometric-circle"]["steps"][0]["result"]["data"]["reconciliation"]["status"] == "held"
    assert held["identified-stability"]["steps"][0]["result"]["data"]["record"]["code"] == "OUTSIDE_LEVEL_SET"
    retained["held"] = held


def test_native_refusals_cannot_leave_partial_bundles(retained):
    session = retained["session"]
    measurement = json.loads(source_bytes(retained["originals"]["measurement-chain"]))
    measurement["investigation"]["sensors"][0]["request"]["inputs"]["calibration"]["valid_until"] = "2026-01-02T00:00:00Z"
    geometry = json.loads((ROOT / "examples/geometric-circle/stale.json").read_bytes())
    stability = json.loads(source_bytes(retained["originals"]["identified-stability"]))
    stability["equilibrium"]["frame_id"] = "frame:foreign"
    for kind, declaration in zip(KINDS, (measurement, geometry, stability), strict=True):
        raw = canonical(declaration)
        before = deepcopy(call(session, "bundle.list"))
        source = call(session, "source.add", {"kind": kind, "label": "Public synthetic refusal " + kind,
            "bytes_b64": base64.b64encode(raw).decode("ascii")})
        parameters = {"source_id": source["source_id"]}
        if kind == "identified-stability":
            parameters["upstream_bundle_id"] = retained["upstream"]["bundle_digest"]
        error = call(session, "operation.execute", {"operation_id": "ciw." + kind + ".v1",
                     "parameters": parameters}, error=True)
        assert call(session, "bundle.list") == before
        output(kind, "native-refusals", [{"case": "expired_calibration_or_geometry_or_foreign_frame",
            "stage": "operation.execute", "source_bytes_b64": base64.b64encode(raw).decode("ascii"),
            "error": error, "execution": "refused", "scientific_result": "not_retained"}])
    assert session.workbench.pending_operations == 0


def test_source_refusals_do_not_create_scientific_results(retained):
    session = retained["session"]
    for kind, original in retained["originals"].items():
        declaration = json.loads(source_bytes(original))
        declaration["configuration"]["state_admission"] = "automatic"
        raw = canonical(declaration)
        before = deepcopy(call(session, "bundle.list"))
        error = call(session, "source.add", {"kind": kind, "label": "Refuse unsupported authority",
            "bytes_b64": base64.b64encode(raw).decode("ascii")}, error=True)
        assert call(session, "bundle.list") == before
        output(kind, "source-refusals", [{"case": "undeclared_state_admission", "stage": "source.add",
            "source_bytes_b64": base64.b64encode(raw).decode("ascii"), "error": error,
            "execution": "not_performed"}])
    assert session.workbench.pending_operations == 0


def test_saved_common_bench_restores_without_host_binding_or_execution(retained, tmp_path):
    original = retained["session"]
    workspace = original.save_workspace(tmp_path / "workspace.json")
    restored = Session.from_workspace(workspace, tmp_path / "restored")
    assert restored.workbench.serialize() == original.workbench.serialize()
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1"}
    for kind, bundle in [*retained["originals"].items(), *retained.get("held", {}).items()]:
        assert call(restored, "bundle.get", {"bundle_id": bundle["bundle_digest"]}) == bundle
        view = call(restored, "experiment.inspect", {"bundle_id": bundle["bundle_digest"]})
        assert view["kind"] == kind and view["fusion_context"] is None
        call(restored, "bundle.replay", {"bundle_id": bundle["bundle_digest"]}, error=True)
    directory = os.environ.get("CIW_REMAINING_FIXTURE_DIR")
    if directory:
        original.save_workspace(Path(directory) / "workspace.json")
    assert restored.workbench.pending_operations == 0
