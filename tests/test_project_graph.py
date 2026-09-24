"""The retained workbench read as the project graph spine."""
from copy import deepcopy
import json

import pytest

from ciw import project_graph, project_model, thermal_workflow
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


def test_a_retained_execution_refreshes_the_pins_a_provider_was_bound_with(tmp_path):
    session, source = _session_with_source(tmp_path)
    # The provider changed after binding; the next execution retains what it found.
    session.workbench._bound_pins["thermal-observer"] = {"thermal": "sha256:" + "0" * 64}
    _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                         "parameters": {"source_id": source["source_id"]}})
    view = session.workbench.project_view()
    assert session.workbench.current_pins()["thermal-observer"] == _node(view, "computation:")[0]["pins"]
    assert view["needs_reevaluation"] == []


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


def _result(node_id, kind, status="current_for_declared_inputs"):
    return {"node_id": node_id, "kind": "workflow_result", "source_kind": kind, "status": status}


def _edge(origin, target):
    return {"relation": "computation", "from": origin, "to": target}


CATALOG = {"loop": {"title": "Loop", "question": "Does the retained loop close?",
                    "default_pipeline": ["ciw.a.v1", "ciw.b.v1"],
                    "pipelines": ["ciw.a.v1", "ciw.b.v1", "ciw.c.v1", "ciw.d.v1", "ciw.e.v1"]}}
OPS = {"a": "ciw.a.v1", "b": "ciw.b.v1", "c": "ciw.c.v1", "d": "ciw.d.v1", "e": "ciw.e.v1", "x": "ciw.x.v1"}


def test_an_investigation_without_retained_results_is_not_started():
    (progress,) = project_graph.investigations([_result("result:x", "x")], [], OPS, CATALOG)
    assert progress["state"] == "not_started" and progress["chains"] == []
    assert progress["missing_default_stages"] == ["ciw.a.v1", "ciw.b.v1"]


def test_connected_current_results_answer_the_default_pipeline_and_form_chains():
    nodes = [_result("result:a", "a"), _result("result:b", "b"), _result("result:c", "c"),
             _result("result:d", "d"), _result("result:e", "e"), {"node_id": "computation:a", "kind": "computation"}]
    edges = [_edge("computation:a", "result:a"), _edge("result:a", "result:b"),
             _edge("result:c", "result:d"), _edge("result:d", "result:e")]
    (progress,) = project_graph.investigations(nodes, edges, OPS, CATALOG)
    assert progress["state"] == "default_pipeline_current" and progress["missing_default_stages"] == []
    assert [chain["nodes"] for chain in progress["chains"]] == [["result:a", "result:b"],
                                                               ["result:c", "result:d", "result:e"]]
    assert all(chain["status"] == "current_for_declared_inputs" for chain in progress["chains"])
    assert progress["chains"][1]["pipelines"] == ["ciw.c.v1", "ciw.d.v1", "ciw.e.v1"]
    assert progress["chains"][1]["sequence"] == ["ciw.c.v1", "ciw.d.v1", "ciw.e.v1"]


def test_a_stale_upstream_makes_its_chain_and_its_default_stage_need_reevaluation():
    nodes = [_result("result:a", "a"), _result("result:b", "b", "needs_reevaluation")]
    (progress,) = project_graph.investigations(nodes, [_edge("result:a", "result:b")], OPS, CATALOG)
    assert progress["state"] == "incomplete" and progress["missing_default_stages"] == ["ciw.b.v1"]
    (chain,) = progress["chains"]
    assert chain["status"] == "needs_reevaluation"
    stages = {stage["pipeline_id"]: stage for stage in progress["stages"]}
    assert stages["ciw.b.v1"]["results"] == ["result:b"] and stages["ciw.b.v1"]["current"] == []


def test_a_fan_in_is_one_chain_whose_links_keep_its_branches():
    from ciw.cli import chain_text
    nodes = [_result("result:c1", "c"), _result("result:c2", "c"), _result("result:d1", "d"),
             _result("result:d2", "d"), _result("result:e", "e")]
    edges = [_edge("result:c1", "result:d1"), _edge("result:c2", "result:d2"),
             _edge("result:d1", "result:e"), _edge("result:d2", "result:e")]
    (progress,) = project_graph.investigations(nodes, edges, OPS, CATALOG)
    (chain,) = progress["chains"]
    assert chain["result"] == "result:e" and len(chain["nodes"]) == 5
    assert chain["links"] == [["result:c1", "result:d1"], ["result:c2", "result:d2"],
                              ["result:d1", "result:e"], ["result:d2", "result:e"]]
    # Never printed as a line that claims a d result feeds a c result.
    assert chain_text(chain) == "ciw.c.v1 -> ciw.d.v1; ciw.c.v1 -> ciw.d.v1; ciw.d.v1 -> ciw.e.v1; ciw.d.v1 -> ciw.e.v1"


def test_a_current_branch_is_its_own_chain_beside_a_stale_sibling():
    from ciw.cli import chain_text
    nodes = [_result("result:c", "c"), _result("result:d1", "d"), _result("result:e1", "e", "needs_reevaluation"),
             _result("result:d2", "d"), _result("result:e2", "e")]
    edges = [_edge("result:c", "result:d1"), _edge("result:d1", "result:e1"),
             _edge("result:c", "result:d2"), _edge("result:d2", "result:e2")]
    (progress,) = project_graph.investigations(nodes, edges, OPS, CATALOG)
    stale, current = progress["chains"]
    assert (stale["result"], stale["status"]) == ("result:e1", "needs_reevaluation")
    assert (current["result"], current["status"]) == ("result:e2", "current_for_declared_inputs")
    assert current["nodes"] == ["result:c", "result:d2", "result:e2"]
    assert chain_text(current) == "ciw.c.v1 -> ciw.d.v1 -> ciw.e.v1"


def test_the_graph_view_reports_a_retained_member_result_under_its_investigation(tmp_path):
    from test_machine_workflow import _session_with_source as machine_session
    session, source = machine_session(tmp_path)
    _call(session, "operation.execute", {"operation_id": "ciw.encoder-position.v1",
                                         "parameters": {"source_id": source["source_id"]}})
    view = _call(session, "session.get", {})["workbench"]["project"]
    progress = {item["investigation_id"]: item for item in view["investigations"]}
    cycle = progress["manufacturing-cycle"]
    stage = next(item for item in cycle["stages"] if item["pipeline_id"] == "ciw.encoder-position.v1")
    assert len(stage["current"]) == 1 and cycle["state"] == "incomplete" and cycle["chains"] == []
    assert "ciw.acquired-dataset.v1" in cycle["missing_default_stages"]
    assert progress["geometry-bim"]["state"] == "not_started"


def test_a_changed_reference_code_identity_marks_retained_results_for_reevaluation(tmp_path, monkeypatch):
    from ciw import machine_workflow
    from test_machine_workflow import _session_with_source as machine_session
    session, source = machine_session(tmp_path)
    _call(session, "operation.execute", {"operation_id": "ciw.encoder-position.v1",
                                         "parameters": {"source_id": source["source_id"]}})
    before = session.workbench.project_view()
    assert before["needs_reevaluation"] == []
    original = machine_workflow.runtime_identity

    def changed():
        value = deepcopy(original())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(machine_workflow, "runtime_identity", changed)
    after = session.workbench.project_view()
    (result,) = _node(after, "result:")
    assert result["node_id"] in after["needs_reevaluation"] and result["status"] == "needs_reevaluation"
    cycle = next(item for item in after["investigations"] if item["investigation_id"] == "manufacturing-cycle")
    stage = next(item for item in cycle["stages"] if item["pipeline_id"] == "ciw.encoder-position.v1")
    assert stage["results"] == [result["node_id"]] and stage["current"] == []
    # Replay reproduces only under the retained pins; executing the source
    # again under the current ones retains a current result beside it.
    replay = _call(session, "bundle.replay", {"bundle_id": result["current_occurrences"][0]})
    assert replay["status"] == "refused"
    _call(session, "operation.execute", {"operation_id": "ciw.encoder-position.v1",
                                         "parameters": {"source_id": source["source_id"]}})
    rerun = session.workbench.project_view()
    assert rerun["needs_reevaluation"] == [result["node_id"]]
    cycle = next(item for item in rerun["investigations"] if item["investigation_id"] == "manufacturing-cycle")
    stage = next(item for item in cycle["stages"] if item["pipeline_id"] == "ciw.encoder-position.v1")
    assert len(stage["results"]) == 2 and len(stage["current"]) == 1
    # Restoring the identity restores currency: nothing was rewritten.
    monkeypatch.setattr(machine_workflow, "runtime_identity", original)
    assert session.workbench.project_view()["needs_reevaluation"] == [node["node_id"] for node in _node(rerun, "result:")
                                                                      if node["node_id"] != result["node_id"]]
