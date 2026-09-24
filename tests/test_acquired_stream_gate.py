"""Native acquisition to calibration to monitoring through one live shared bench."""
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

from ciw import acquired_window, calibrated_window, residual_monitor
from ciw.cli import parser, request_remote
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]


def call(session, kind, payload=None, *, error=False):
    response = session.handle({"protocol_version": 1, "request_id": "acquired-stream-gate", "type": kind, "payload": payload or {}})
    assert response["type"] == ("error" if error else "response"), response
    return response["payload"]


def output(kind, name, value):
    directory = os.environ.get("CIW_ACQUIRED_STREAM_FIXTURE_DIR")
    if directory:
        target = Path(directory) / kind
        target.mkdir(parents=True, exist_ok=True)
        (target / (name + ".json")).write_bytes(value if isinstance(value, bytes) else canonical(value))


def source_bytes(bundle):
    return base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"], validate=True)


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    stack = os.environ.get("CIW_ACQUIRED_STREAM_STACK_ROOT")
    if not stack:
        pytest.skip("set CIW_ACQUIRED_STREAM_STACK_ROOT for exact native acquisition and monitoring")
    providers = Path(stack)
    directory = tmp_path_factory.mktemp("acquired-stream")
    session = Session(make_demo_run(), directory)
    session.workbench.bind_workflow("acquired-dataset", {"ppda": providers / "ppda"})
    session.workbench.bind_workflow("acquired-calibrated-window", {role: providers / role for role in calibrated_window.ROLES})
    session.workbench.bind_workflow("residual-monitor", {role: providers / role for role in residual_monitor.ROLES})

    async def exercise_client():
        bridge = WorkbenchServer(session)
        client = runpy.run_path(str(ROOT / "examples/acquired-stream/run.py"))["run"]
        async with serve(bridge.handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, max_size=8_388_608, proxy=None) as observer:
                snapshot = json.loads(await observer.recv())
                assert snapshot["type"] == "session.snapshot"
                result = await client(url)
                # Five retained sources, five executions and the monitor replay
                # produce committed catalog invalidations on the same socket.
                events = [json.loads(await asyncio.wait_for(observer.recv(), 5)) for _ in range(11)]
                assert all(event["type"] == "workbench.changed" for event in events)
                assert all(event["payload"]["session_id"] == session.session_id for event in events)
                remote = await request_remote(url, "bundle.list", {}, timeout_s=30)
                assert remote["type"] == "response"
                assert len(remote["payload"]["bundles"]) == 6
                return result

    result = asyncio.run(exercise_client())
    acquisition = call(session, "bundle.get", {"bundle_id": result["acquisition_bundle_id"]})
    windows = [call(session, "bundle.get", {"bundle_id": identity}) for identity in result["window_bundle_ids"]]
    monitor = call(session, "bundle.get", {"bundle_id": result["monitor_bundle_id"]})
    monitor_replay = call(session, "bundle.get", {"bundle_id": result["monitor_replay"]["bundle"]["bundle_id"]})
    replays = {}
    for kind, bundle in (("acquired-dataset", acquisition), ("acquired-calibrated-window", windows[0])):
        replay = call(session, "bundle.replay", {"bundle_id": bundle["bundle_digest"]})
        fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
        replays[kind] = fresh
        output(kind, "source", source_bytes(bundle))
        output(kind, "original", bundle)
        output(kind, "replay", fresh)
    for index, bundle in enumerate(windows, 1):
        output("acquired-calibrated-window", "source-" + str(index), source_bytes(bundle))
        output("acquired-calibrated-window", "original-" + str(index), bundle)
    output("residual-monitor", "source", source_bytes(monitor))
    output("residual-monitor", "original", monitor)
    output("residual-monitor", "replay", monitor_replay)

    held_source = json.loads(source_bytes(monitor))
    held_source["experiment_id"] += "-undeclared-conditioning-limit"
    held_source["configuration"]["observability"]["condition_limit"] = None
    held_raw = canonical(held_source)
    source = call(session, "source.add", {"kind": "residual-monitor", "label": "Declared unresolved conditioning gate",
        "bytes_b64": base64.b64encode(held_raw).decode()})
    held_result = call(session, "operation.execute", {"operation_id": "ciw.residual-monitor.v1", "parameters": {"source_id": source["source_id"]}})
    held = call(session, "bundle.get", {"bundle_id": held_result["bundle_id"]})
    output("residual-monitor", "held-source", held_raw)
    output("residual-monitor", "held", held)
    workspace = session.save_workspace(directory / "workspace.json")
    fixture_directory = os.environ.get("CIW_ACQUIRED_STREAM_FIXTURE_DIR")
    if fixture_directory:
        session.save_workspace(Path(fixture_directory) / "workspace.json")
    return {"session": session, "acquisition": acquisition, "windows": windows, "monitor": monitor,
            "monitor_replay": monitor_replay, "replays": replays, "held": held, "workspace": workspace}


def test_acquired_stream_startup_and_explicit_synthetic_reference():
    args = parser().parse_args(["serve", "--acquired-stream-stack-root", "/trusted/stream"])
    assert args.acquired_stream_stack_root == Path("/trusted/stream")
    templates = runpy.run_path(str(ROOT / "examples/acquired-stream/make_sequence.py"))["templates"]()
    assert [source["configuration"]["gsie"]["target_time"] for source in templates] == [2, 4, 6]
    assert all(source["configuration"]["gsie"]["prior"] == templates[0]["configuration"]["gsie"]["prior"] for source in templates)
    assert [sum(sample["indicated_value"] * 2 + 1 for sample in source["samples"]) / 2 for source in templates] == [7, 7.5, 17]


def test_one_catalog_retains_acquisition_each_window_and_monitor(retained):
    session = retained["session"]
    catalog = {row["bundle_id"]: row for row in call(session, "bundle.list")["bundles"]}
    executions = {row["execution_id"]: row for row in call(session, "execution.list")["executions"]}
    assert len(catalog) == 9
    acquisition = retained["acquisition"]
    for index, window in enumerate(retained["windows"]):
        assert acquired_window._validate(window) == source_bytes(window)
        assert window["upstream_acquisition"] == acquisition
        assert window["acquisition_binding"]["bundle_id"] == acquisition["bundle_digest"]
        assert {selected["snapshot_index"] for selected in window["acquisition_binding"]["selected_records"]} == {index}
        assert window["steps"] == window["child_window"]["steps"]
        for step in window["steps"]:
            assert call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
            occurrence = executions[step["execution_id"]]
            assert occurrence["bundle_id"] == window["bundle_digest"]
            assert occurrence["native_bundle_id"] == window["child_window"]["bundle_digest"]
            assert occurrence["source_id"] == catalog[window["bundle_digest"]]["source_id"]
            assert occurrence["source_id"] != catalog[acquisition["bundle_digest"]]["source_id"]
            assert occurrence["native_source_evidence_id"] == window["child_window"]["source"]["evidence"][0]["artifact_ref"]
    monitor = retained["monitor"]
    assert set(monitor["upstream_windows"]) == {window["bundle_digest"] for window in retained["windows"]}
    assert all(step["runtime_ref"] in {"oit", "fdir"} for step in monitor["steps"])
    assert all(original == monitor["upstream_windows"][original["bundle_digest"]] for original in retained["windows"])
    context_ids = {context["bundle_id"] for context in session.workbench.fusion_contexts()}
    assert context_ids == {window["bundle_digest"] for window in retained["windows"]} | {
        retained["replays"]["acquired-calibrated-window"]["bundle_digest"]}
    assert session.workbench.pending_operations == 0


def test_monitor_consumes_declared_gsie_covariance_and_holds_unsupported_claims(retained):
    rows = retained["monitor"]["steps"][0]["result"]["data"]["rows"]
    assert [row["detection"]["status"] for row in rows] == ["nominal", "nominal", "statistical_anomaly"]
    assert rows[-1]["cusum"]["status"] == "statistical_anomaly"
    assert rows[-1]["isolability"]["status"] == "ambiguous"
    for row, window in zip(rows, retained["windows"], strict=True):
        diagnostics = window["steps"][3]["result"]["result_artifact"]["diagnostics"]
        assert row["detection"]["raw_residual"] == diagnostics["innovation"]
        assert row["detection"]["innovation_covariance"] == diagnostics["innovation_covariance"]
        assert row["interpretation"]["physical_drift"] == "not_established"
        assert row["interpretation"]["unique_sensor_fault"] is None
        assert row["interpretation"]["alarm_authority"] == "held_unknown_temporal_dependence"
    held_rows = retained["held"]["steps"][0]["result"]["data"]["rows"]
    assert all(row["observability"]["status"] == "unresolved" for row in held_rows)
    assert all(row["interpretation"]["observability_gate"] == "held" for row in held_rows)
    assert not any(row["interpretation"]["diagnostic_drift_candidate"] for row in held_rows)
    assert all(row["cusum"] is None and row["monitor_state_before"] == row["monitor_state_after"] for row in held_rows)


def test_replay_retains_selected_evidence_but_has_fresh_execution_occurrences(retained):
    pairs = ((retained["acquisition"], retained["replays"]["acquired-dataset"]),
             (retained["windows"][0], retained["replays"]["acquired-calibrated-window"]),
             (retained["monitor"], retained["monitor_replay"]))
    for original, fresh in pairs:
        assert original["bundle_digest"] != fresh["bundle_digest"]
        assert source_bytes(original) == source_bytes(fresh)
        assert original["configuration"] == fresh["configuration"]
        for before, after in zip(original["steps"], fresh["steps"], strict=True):
            assert before["numerical_result_id"] == after["numerical_result_id"]
            assert before["execution_id"] != after["execution_id"]
            assert before["result_id"] != after["result_id"]
    assert retained["monitor"]["upstream_windows"] == retained["monitor_replay"]["upstream_windows"]
    assert retained["windows"][0]["upstream_acquisition"] == retained["replays"]["acquired-calibrated-window"]["upstream_acquisition"]


def test_unknown_covariance_and_missing_clock_mapping_cannot_create_results(retained):
    session = retained["session"]
    original = json.loads(source_bytes(retained["windows"][0]))
    faults = []
    for name in ("unknown_cross_covariance", "missing_clock_map", "foreign_observation"):
        declaration = deepcopy(original)
        if name == "unknown_cross_covariance":
            declaration["declaration"]["joint_covariance"]["cross_covariance_policy"] = "unknown"
        elif name == "missing_clock_map":
            declaration["declaration"]["clock_model"] = None
        else:
            declaration["selections"][0]["observation_id"] = "0" * 64
        raw = canonical(declaration)
        before = deepcopy(call(session, "bundle.list"))
        source = call(session, "source.add", {"kind": "acquired-calibrated-window", "label": name,
            "bytes_b64": base64.b64encode(raw).decode()}, error=name == "missing_clock_map")
        if name == "missing_clock_map":
            error, stage = source, "source.add"
        else:
            error = call(session, "operation.execute", {"operation_id": "ciw.acquired-calibrated-window.v1",
                "parameters": {"source_id": source["source_id"], "upstream_bundle_id": retained["acquisition"]["bundle_digest"]}}, error=True)
            stage = "operation.execute"
        assert call(session, "bundle.list") == before
        faults.append({"case": name, "source_bytes_b64": base64.b64encode(raw).decode(), "stage": stage,
                       "error": error, "execution": "not_performed"})
    output("acquired-calibrated-window", "source-refusals", faults)
    assert session.workbench.pending_operations == 0


def test_saved_stream_restores_without_host_bindings_or_native_reexecution(retained, tmp_path):
    original = retained["session"]
    workspace = original.save_workspace(tmp_path / "workspace.json")
    restored = Session.from_workspace(workspace, tmp_path / "restored")
    assert restored.workbench.serialize() == original.workbench.serialize()
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1", "ciw.encoder-position.v1", "ciw.thermal-observer.v1", "ciw.project-graph.v1", "ciw.uncertainty-validation.v1"}
    for bundle in [retained["acquisition"], *retained["windows"], retained["monitor"], retained["held"]]:
        assert call(restored, "bundle.get", {"bundle_id": bundle["bundle_digest"]}) == bundle
        call(restored, "bundle.replay", {"bundle_id": bundle["bundle_digest"]}, error=True)
    assert restored.workbench.pending_operations == 0
