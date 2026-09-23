"""Shared-session boundaries and real pinned SRA/Rust SCR execution."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from ciw.declared_workload import DeclaredWorkflow, _verification
from ciw.instruments import make_demo_run
from ciw.session import Session
from ciw.telemetry import canonical, _bundle_digest
from ciw.workbench import Workbench

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("schematic-assessment", "numerical-heat")


def source(kind):
    return json.loads((ROOT / "examples/declared-workloads" / (kind + ".json")).read_bytes())


def call(session, kind, payload=None, *, error=False):
    result = session.handle({"protocol_version": 1, "request_id": "declared-test", "type": kind, "payload": payload or {}})
    assert result["type"] == ("error" if error else "response"), result
    return result["payload"]


@pytest.mark.parametrize("kind", KINDS)
def test_source_and_operation_available_without_execution(kind, tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = b"\n" + canonical(source(kind)) + b"\n "
    retained = call(session, "source.add", {"kind": kind, "label": kind, "bytes_b64": base64.b64encode(raw).decode()})
    assert base64.b64decode(call(session, "source.get", {"source_id": retained["source_id"]})["bytes_b64"]) == raw
    operation = next(o for o in call(session, "operation.list")["operations"] if o["operation_id"] == "ciw." + kind + ".v1")
    assert not operation["available"] and operation["role"] != "state_estimator"
    call(session, "operation.execute", {"operation_id": operation["operation_id"], "parameters": {"source_id": retained["source_id"]}}, error=True)
    assert session.workbench.serialize()["bundles"] == []


@pytest.mark.parametrize("change", [
    {"steps": True}, {"steps": -1}, {"steps": 1025}, {"initial_values": [0, 1]},
    {"initial_values": [0, 1.0, 0]}, {"initial_values": [0, 2**40 + 1, 0]},
    {"initial_values": [0] * 257}, {"engine": "/artifact/selected/executable"},
    {"configuration": {"unit": "K"}}, {"configuration": {"covariance_status": "unknown"}},
])
def test_numerical_source_refuses_undeclared_semantics(change):
    with pytest.raises(ValueError):
        DeclaredWorkflow("numerical-heat")._source(canonical(source("numerical-heat") | change))


@pytest.mark.parametrize("fault", ["dangling", "wrong-type", "duplicate", "unknown-kind", "extra-field", "unknown-query"])
def test_schematic_source_refuses_undeclared_topology(fault):
    value = source("schematic-assessment")
    graph = value["schematic"]
    if fault == "dangling": graph["edges"][0]["src"] = "not-declared"
    if fault == "wrong-type": graph["edges"][0]["src"] = "observer"
    if fault == "duplicate": graph["nodes"].append(deepcopy(graph["nodes"][0]))
    if fault == "unknown-kind": graph["nodes"][0]["kind"] = "inferred_state"
    if fault == "extra-field": graph["nodes"][0]["silently_ignored"] = True
    if fault == "unknown-query": value["queries"] = ["not-declared"]
    with pytest.raises(ValueError): DeclaredWorkflow("schematic-assessment")._source(canonical(value))


@pytest.fixture(scope="module")
def bindings():
    root, engine = os.environ.get("CIW_DECLARED_STACK_ROOT"), os.environ.get("CIW_SCR_ENGINE")
    if not root or not engine:
        pytest.skip("set CIW_DECLARED_STACK_ROOT and CIW_SCR_ENGINE for native SRA/SCR integration")
    return {"schematic-assessment": {"sra": Path(root) / "sra"},
            "numerical-heat": {"scr": Path(root) / "scr", "engine": Path(engine)}}


@pytest.fixture(scope="module")
def retained(bindings, tmp_path_factory):
    directory = tmp_path_factory.mktemp("declared-workbench")
    session = Session(make_demo_run(), directory)
    bundles = {}
    for kind in KINDS:
        session.workbench.bind_workflow(kind, bindings[kind])
        raw = b"\n" + canonical(source(kind)) + b"\n "
        declared = call(session, "source.add", {"kind": kind, "label": kind, "bytes_b64": base64.b64encode(raw).decode()})
        result = call(session, "operation.execute", {"operation_id": "ciw." + kind + ".v1", "parameters": {"source_id": declared["source_id"]}})
        original = call(session, "bundle.get", {"bundle_id": result["bundle_id"]})
        replay = call(session, "bundle.replay", {"bundle_id": original["bundle_digest"]})
        fresh = call(session, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
        bundles[kind] = original, fresh
        output = os.environ.get("CIW_DECLARED_FIXTURE_DIR")
        if output:
            target = Path(output) / kind
            target.mkdir(parents=True, exist_ok=True)
            for name, value in (("original", original), ("replay", fresh)):
                (target / (name + ".json")).write_bytes(canonical(value))
    return session, bundles, session.save_workspace(directory / "workspace.json")


def test_shared_catalog_native_results_and_restore(retained, monkeypatch):
    session, bundles, path = retained
    assert len(call(session, "bundle.list")["bundles"]) == 4
    assert session.workbench.fusion_contexts() == []
    assert {row["instrument"] for row in session.workbench.instrument_views()} == {"sra", "scr"}
    before = session.workbench.serialize()
    def forbidden(*args, **kwargs): raise AssertionError("Inspection must not execute a provider")
    monkeypatch.setattr(DeclaredWorkflow, "_adapters", forbidden)
    for kind, (original, fresh) in bundles.items():
        view = call(session, "experiment.inspect", {"bundle_id": original["bundle_digest"]})
        assert view["fusion_context"] is None and view["object_context"]["sensor_fusion"] == "not_performed"
        assert view["raw_observations"] == [] and view["raw_declaration"]["schema"] == "ciw." + kind + "-source.v1"
        assert original["steps"][0]["numerical_result_id"] == fresh["steps"][0]["numerical_result_id"]
        assert original["steps"][0]["result_id"] != fresh["steps"][0]["result_id"]
        assert original["steps"][0]["execution_id"] != fresh["steps"][0]["execution_id"]
        native = call(session, "result.get", {"result_id": original["steps"][0]["result_id"]})
        assert native == original["steps"][0]["result"]
        if kind == "numerical-heat":
            assert view["panels"][1]["values"] == [0, 16, 24, 16, 0]
            assert all(p["covariance"] is None for p in view["panels"])
        else:
            assert view["schematic"]["nodes"] == source(kind)["schematic"]["nodes"]
            assert view["panels"] == []
        view["object_context"]["sensor_fusion"] = "forged"
    assert session.workbench.serialize() == before
    restored = Session.from_workspace(path, path.parent / "restored")
    assert restored.workbench.serialize() == before
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == {"ciw.energy-accuracy.v1"}
    call(restored, "bundle.replay", {"bundle_id": bundles[KINDS[0]][0]["bundle_digest"]}, error=True)
    assert restored.workbench.pending_operations == 0


def test_native_schematic_unknown_and_stale_certificates(bindings):
    value = source("schematic-assessment")
    graph = value["schematic"]
    next(n for n in graph["nodes"] if n["id"] == "f")["attrs"]["class"] = "unknown"
    graph["nodes"].append({"id": "old", "kind": "certificate", "attrs": {"owner": "jspt", "result": "SAMPLED", "A": [[-1.0]]}})
    graph["edges"].append({"kind": "linearizes", "src": "old", "dst": "f", "attrs": {}})
    result = DeclaredWorkflow(KINDS[0]).create_session(canonical(value), bindings[KINDS[0]])["steps"][0]["result"]["data"]
    assert all(d["status"] == "NOT_ELIGIBLE" for d in result["decisions"])
    old = next(n for n in result["schematic"]["nodes"] if n["id"] == "old")
    assert old["attrs"]["A"] == [[-1.0]] and old["attrs"]["currentness"] == "stale"
    assert old["attrs"]["historical_result"] == "SAMPLED" and old["attrs"]["result"] == "NOT_ELIGIBLE"


@pytest.mark.parametrize("values,steps,expected", [([0, -3, 0], 1, [0, -2, 0]), ([0, 3, 0], 1, [0, 2, 0]), ([7, -3, 5], 0, [7, -3, 5]), ([7, -3, 5], 1, [7, 1, 5])])
def test_native_integer_rounding_and_fixed_boundaries(bindings, values, steps, expected):
    value = source("numerical-heat") | {"initial_values": values, "steps": steps}
    result = DeclaredWorkflow("numerical-heat").create_session(canonical(value), bindings["numerical-heat"])
    assert result["steps"][0]["result"]["data"]["values"] == expected


def test_engine_drift_refuses_replay_without_running_it(retained, bindings, tmp_path):
    original = retained[1]["numerical-heat"][0]
    engine = tmp_path / "changed-engine"
    engine.write_bytes(Path(bindings["numerical-heat"]["engine"]).read_bytes() + b"changed")
    with pytest.raises(ValueError, match="runtime differs"):
        DeclaredWorkflow("numerical-heat").replay_session(original, bindings["numerical-heat"] | {"engine": engine})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", ["independence", "reuse", "topology-or-output", "runtime", "verification-subject"])
def test_resealed_retained_faults_are_refused(retained, kind, fault):
    native = deepcopy(retained[1][kind][0])
    if fault == "independence": native["verification"]["independent"] = True
    if fault == "reuse": native["verification"]["reproduction"] = deepcopy(native["steps"][0])
    if fault == "verification-subject": native["verification"]["subject_ref"] = "sha256:" + "0" * 64
    if fault == "runtime": next(iter(native["runtimes"].values()))["revision"] = "0" * 40
    if fault == "topology-or-output":
        data = native["steps"][0]["result"]["data"]
        if kind == "numerical-heat": data["values"][1] += 1
        else: data["schematic"]["edges"].pop()
    native["bundle_digest"] = _bundle_digest(native)
    with pytest.raises(ValueError): DeclaredWorkflow(kind)._validate(native)


def test_non_observation_workloads_cannot_enter_esm_candidate_capture(retained):
    session, bundles, _ = retained
    for original, _ in bundles.values():
        with pytest.raises(ValueError, match="calibrated or telemetry"):
            session.workbench.execute_candidate("esm.inspect-candidate.v1", {"bundle_id": original["bundle_digest"], "inspected_at": "2026-09-22T00:00:00Z"})


@pytest.mark.parametrize("kind", KINDS)
def test_new_bundle_cannot_reuse_old_execution_or_reproduction(retained, kind):
    saved = retained[0].workbench.serialize()
    original = next(r for r in saved["bundles"] if r["kind"] == kind)
    duplicate = deepcopy(original)
    native = duplicate["native"]
    native["session_id"] = "session-" + "0" * 32
    native["bundle_digest"] = _bundle_digest(native)
    native["verification"] = _verification(native, native["verification"]["reproduction"])
    duplicate["bundle_id"] = native["bundle_digest"]
    saved["bundles"].append(duplicate)
    saved["revision"] += 1
    with pytest.raises(ValueError, match="distinct execution"):
        Workbench.restore(saved)
