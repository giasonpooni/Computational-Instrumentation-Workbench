"""Analytical anchors and refusal/binding checks for real mathematical providers."""
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ciw.adapters.protocol import AdapterRefusal
from ciw.geometry_research import GeometryResearchWorkflow, KINDS, PINS, _check_data
from ciw.core.canonical import canonical, digest

ROOT = Path(__file__).resolve().parents[1]


def source(kind):
    return json.loads((ROOT / "examples/geometry-research" / (kind + ".json")).read_bytes())


@pytest.fixture(scope="module", params=sorted(KINDS))
def native(request):
    kind = request.param
    path = os.environ.get("CIW_" + PINS[kind]["role"].upper() + "_REPO")
    if not path:
        pytest.skip("Native geometry gate requires exact CGGT, ISGT and TSDE checkouts")
    workflow = GeometryResearchWorkflow(kind)
    raw = (ROOT / "examples/geometry-research" / (kind + ".json")).read_bytes()
    bindings = {workflow.role:path}
    original = workflow.create_session(raw, bindings)
    replay = workflow.replay_session(original, bindings)["session"]
    return kind, workflow, bindings, raw, original, replay


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_source_requires_explicit_policy_and_no_dynamic_provider(kind):
    workflow = GeometryResearchWorkflow(kind)
    value = source(kind)
    value["configuration"]["state_admission"] = "approved"
    with pytest.raises(ValueError):
        workflow._source(canonical(value))
    value = source(kind)
    value["provider_path"] = "/tmp/arbitrary-provider"
    with pytest.raises(ValueError):
        workflow._source(canonical(value))
    with pytest.raises(ValueError):
        workflow.create_session(canonical(source(kind)), {})


@pytest.mark.parametrize("kind,field,value", [
    ("covariance-geometry", "parameters", [0, True, 1]),
    ("covariance-geometry", "parameters", [0, .5, .5, 1]),
    ("covariance-geometry", "covariance_a", [[True,0],[0,1]]),
    ("mesh-path", "source_vertex", True),
    ("mesh-path", "target_vertex", 999),
    ("translation-flow", "max_events", 257),
    ("translation-flow", "max_events", True),
    ("translation-flow", "direction", ["2/4", "1"]),
    ("translation-flow", "duration", "NaN"),
    ("translation-flow", "gluing", {"right":[0,0],"up":[0,1]}),
])
def test_invalid_source_is_refused_before_provider(kind, field, value, monkeypatch):
    request = source(kind)
    request["request"][field] = value
    workflow = GeometryResearchWorkflow(kind)
    monkeypatch.setattr(workflow, "_adapters", lambda *_: pytest.fail("Invalid source reached runtime"))
    with pytest.raises(ValueError):
        workflow.create_session(canonical(request), {})


def test_native_evidence_replay_and_provider_isolation(native):
    kind, workflow, _, raw, original, replay = native
    assert workflow._validate(original) == raw
    assert workflow._validate(replay) == raw
    assert base64.b64decode(original["source"]["evidence"][0]["bytes_b64"]) == raw
    assert original["source"] == replay["source"]
    assert original["session_id"] != replay["session_id"]
    old, fresh = original["steps"][0], replay["steps"][0]
    assert old["execution_id"] != fresh["execution_id"]
    assert old["result_id"] != fresh["result_id"]
    assert old["numerical_result_id"] == fresh["numerical_result_id"]
    assert original["verification"]["independent"] is False
    assert original["verification"]["reproduction"]["execution_id"] != old["execution_id"]
    assert PINS[kind]["module"] not in sys.modules
    assert original["runtimes"][workflow.role]["source_tree"] == PINS[kind]["source_tree"]


def test_native_results_against_independent_analytic_anchors(native):
    kind, _, _, _, original, _ = native
    data = original["steps"][0]["result"]["data"]
    if kind == "covariance-geometry":
        assert data["distance"] == pytest.approx(math.sqrt(2) * math.log(4), rel=1e-12)
        midpoint = next(s for s in data["samples"] if s["parameter"] == .5)
        assert midpoint["covariance"][0] == pytest.approx([2,0])
        assert midpoint["covariance"][1] == pytest.approx([0,8])
    elif kind == "mesh-path":
        assert data["solution"]["target_distance"] == pytest.approx(2)
        assert data["bounds"]["target_lower_bound"] == pytest.approx(math.sqrt(2))
        assert data["bounds"]["target_gap"] == pytest.approx(2-math.sqrt(2))
        assert data["claim_scope"] == "edge_constrained_upper_bound_on_declared_piecewise_flat_mesh"
    else:
        assert data["status"] == "completed"
        assert data["final_state"]["tile"] == 1
        assert data["final_state"]["position"] == ["1/4", "5/6"]
        assert len(data["events"]) == 4
        assert data["elapsed"] == "3" and data["remaining"] == "0"


def test_native_seals_and_request_binding_are_checked_offline(native):
    kind, workflow, _, raw, original, _ = native
    data = deepcopy(original["steps"][0]["result"]["data"])
    data["request"]["schema"] = "another-request"
    data["request_digest"] = digest(data["request"])
    data["artifact_digest"] = digest({k:v for k,v in data.items() if k != "artifact_digest"})
    with pytest.raises(ValueError):
        _check_data(kind, workflow._source(raw), data)
    changed = deepcopy(original)
    changed["runtimes"][workflow.role]["source_tree"] = "f" * 40
    from ciw.telemetry import _bundle_digest
    changed["bundle_digest"] = _bundle_digest(changed)
    with pytest.raises(ValueError):
        workflow._validate(changed)


def test_native_provider_refuses_semantically_invalid_request(native):
    kind, workflow, bindings, _, _, _ = native
    value = source(kind)
    if kind == "covariance-geometry":
        value["request"]["covariance_a"] = [[0,0],[0,1]]
    elif kind == "mesh-path":
        mesh = value["request"]["mesh"]
        mesh["vertices"][2] = [2,0,0]
        mesh["mesh_digest"] = digest({k:v for k,v in mesh.items() if k != "mesh_digest"})
    else:
        value["request"]["gluing"] = {"right":[0,1,2],"up":[0,1,2]}
    with pytest.raises((AdapterRefusal, ValueError)):
        workflow.create_session(canonical(value), bindings)


def test_wrong_and_dirty_provider_never_execute(native, tmp_path):
    _, workflow, bindings, raw, _, _ = native
    checkout = tmp_path / "provider"
    subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "--quiet", "--no-hardlinks",
        str(bindings[workflow.role]), str(checkout)], check=True, capture_output=True)
    (checkout / "unexpected.py").write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    with pytest.raises((AdapterRefusal, ValueError)):
        workflow.create_session(raw, {workflow.role:checkout})
    with pytest.raises((AdapterRefusal, ValueError)):
        workflow.create_session(raw, {workflow.role:tmp_path / "absent"})


def test_translation_vertex_stop_is_retained_as_partial(native):
    kind, workflow, bindings, _, _, _ = native
    if kind != "translation-flow":
        return
    value = source(kind)
    value["request"].update(start={"tile":0,"position":["1/2","1/2"]}, direction=["1","1"], duration="1")
    bundle = workflow.create_session(canonical(value), bindings)
    data = bundle["steps"][0]["result"]["data"]
    assert data["status"] == "stopped_at_vertex"
    assert data["elapsed"] == "1/2" and data["remaining"] == "1/2"
    assert data["final_state"]["pending_edges"]
