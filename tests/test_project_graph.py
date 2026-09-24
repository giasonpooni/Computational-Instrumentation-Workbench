"""The retained workbench read as the project graph spine."""
import base64
from copy import deepcopy
import json

import pytest

from ciw import project_graph, project_model, thermal_workflow
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.workbench import OPERATIONS

from test_thermal_workflow import _call, _session_with_source


def _node(view, prefix):
    return [node for node in view["nodes"] if node["node_id"].startswith(prefix)]


def test_executed_bundle_becomes_evidence_computation_and_result_nodes(tmp_path):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    view = _call(session, "session.get", {})["workbench"]["project"]
    assert view["schema"] == "ciw.project-graph-view.v1" and view["needs_reevaluation"] == []
    (evidence,), (computation,), (result,) = _node(view, "source:"), _node(view, "computation:"), _node(view, "result:")
    assert evidence["kind"] == "evidence" and computation["operation"] == "ciw.thermal-observer.v1"
    assert result["status"] == "current_for_declared_inputs" and result["current_occurrences"] == [completed["bundle_id"]]
    assert {(edge["relation"], edge["from"], edge["to"]) for edge in view["edges"]} == {
        ("computation", computation["node_id"], result["node_id"]), ("evidence", evidence["node_id"], result["node_id"])}
    assert view["authority"]["execution"] == "not_performed"


def test_replay_reevaluates_the_same_node_and_reopen_reads_the_same_graph(tmp_path):
    session, source = _session_with_source(tmp_path / "original")
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    before = session.workbench.project_view()
    saved = session.save_workspace(tmp_path / "saved.json")
    original_create = thermal_workflow.ThermalWorkflow.create_session
    thermal_workflow.ThermalWorkflow.create_session = lambda *a, **k: pytest.fail("reopen executed a provider")
    try:
        reopened = Session.from_workspace(saved, tmp_path / "reopened")
    finally:
        thermal_workflow.ThermalWorkflow.create_session = original_create
    assert reopened.workbench.project_view() == before  # reopen is reading the graph
    replay = _call(reopened, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    after = reopened.workbench.project_view()
    (result,) = _node(after, "result:")
    assert [item["bundle_id"] for item in result["occurrences"]] == [completed["bundle_id"], replay["bundle"]["bundle_id"]]
    assert result["occurrences"][1]["replay_of"] == completed["bundle_id"]
    # Identical numerical results keep the node revision: execution history does not revise it.
    assert result["revision"] == _node(before, "result:")[0]["revision"]
    assert result["status"] == "current_for_declared_inputs" and len(result["current_occurrences"]) == 2


def test_pin_drift_marks_results_for_reevaluation_without_host_paths(tmp_path):
    session, source = _session_with_source(tmp_path)
    _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                         "parameters": {"source_id": source["source_id"]}})
    recorded = _node(session.workbench.project_view(), "computation:")[0]["pins"]
    same = session.workbench.project_view(current_pins={"thermal-observer": recorded})
    assert same["needs_reevaluation"] == []
    drifted = session.workbench.project_view(current_pins={"thermal-observer": {"thermal": "sha256:" + "0" * 64}})
    assert [node["node_id"] for node in _node(drifted, "result:")] == drifted["needs_reevaluation"]
    runtime = {"revision": "a" * 40, "repository_root": "/home/alice/provider", "python_executable": "C:\\py\\python.exe",
               "nested": {"tool": "/opt/x", "version": "1"}}
    moved = dict(runtime, repository_root="/srv/other/provider", python_executable="/usr/bin/python3")
    assert project_graph.pin_projection(runtime) == project_graph.pin_projection(moved) == {"nested": {"version": "1"}, "revision": "a" * 40}


def _synthetic(numerical_b="sha256:" + "b" * 64):
    def bundle(bundle_id, kind, source_id, upstream=None, numerical="sha256:" + "a" * 64, replay_of=None, revision="1" * 40):
        native = {"runtimes": {"role": {"revision": revision, "repository_root": "/tmp/x"}},
                  "steps": [{"numerical_result_id": numerical}]}
        if replay_of:
            native["replay_receipts"] = [{"source_bundle_digest": replay_of}]
        return {"kind": kind, "bundle_id": bundle_id, "source_id": source_id, "upstream_bundle_id": upstream, "native": native}
    sources = [{"source_id": f"source:sha256:{c * 64}", "evidence_id": f"sha256:{c * 64}", "label": c} for c in "cd"]
    a = bundle("sha256:" + "1" * 64, "calibrated-observable", sources[0]["source_id"])
    b = bundle("sha256:" + "2" * 64, "identified-design", sources[1]["source_id"], upstream=a["bundle_id"])
    return {"sources": sources, "bundles": [a, b]}, bundle


def test_upstream_correction_invalidates_dependents_but_reproduction_does_not():
    state, bundle = _synthetic()
    upstream = {"sha256:" + "2" * 64: ["sha256:" + "1" * 64]}
    base = project_graph.view(state, OPERATIONS, upstream)
    assert base["needs_reevaluation"] == []
    reproduced = deepcopy(state)
    reproduced["bundles"].append(bundle("sha256:" + "3" * 64, "calibrated-observable", state["sources"][0]["source_id"],
                                        replay_of="sha256:" + "1" * 64))
    assert project_graph.view(reproduced, OPERATIONS, upstream)["needs_reevaluation"] == []
    corrected = deepcopy(state)
    corrected["bundles"].append(bundle("sha256:" + "3" * 64, "calibrated-observable", state["sources"][0]["source_id"],
                                       numerical="sha256:" + "f" * 64, replay_of="sha256:" + "1" * 64))
    stale = project_graph.view(corrected, OPERATIONS, upstream)["needs_reevaluation"]
    design = next(n for n in project_graph.view(corrected, OPERATIONS, upstream)["nodes"] if n.get("source_kind") == "identified-design")
    assert stale == [design["node_id"]]
    rebound = deepcopy(state)
    rebound["bundles"].append(bundle("sha256:" + "4" * 64, "calibrated-observable", state["sources"][0]["source_id"], revision="2" * 40))
    stale = project_graph.view(rebound, OPERATIONS, upstream)["needs_reevaluation"]
    assert len(stale) == 2  # the first calibrated result and its dependent predate the new pin


def test_graph_is_a_valid_append_only_project_and_scales_to_workspace_bounds():
    state, bundle = _synthetic()
    sources = [{"source_id": f"source:sha256:{i:064x}", "evidence_id": f"sha256:{i:064x}", "label": str(i)} for i in range(64)]
    bundles = [bundle(f"sha256:{i + 1000:064x}", "thermal-observer", sources[i % 64]["source_id"],
                      numerical=f"sha256:{i + 5000:064x}") for i in range(128)]
    project, occurrences = project_graph.build({"sources": sources, "bundles": bundles}, OPERATIONS, {})
    assert project_model.validate(json.loads(json.dumps(project))) == project
    assert len(occurrences["bundle_nodes"]) == 128
