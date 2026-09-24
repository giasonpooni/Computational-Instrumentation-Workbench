"""Project graph history is typed, content-addressed and provider-free."""

import json
from copy import deepcopy

import pytest

from ciw import project_model as project


def _evidence(object_id="evidence:manual"):
    return {"object_id": object_id, "kind": "evidence", "label": "Manual", "content": {
        "status": "documented", "source_digest": "sha256:" + "1" * 64,
        "locator": "fixture://manual", "subject_id": "machine:fixture",
    }}


def _signal(object_id="signal:position"):
    return {"object_id": object_id, "kind": "signal", "label": "Position", "content": {
        "quantity": "position", "unit": "m", "frame": "frame:carriage",
        "time_basis": "clock:utc", "semantics": "observed",
    }}


def _computation():
    return {"object_id": "computation:position", "kind": "computation", "label": "Position model",
            "content": {"operation": "ciw.encoder-position.v1", "parameters": {"mode": "declared"}}}


def test_project_roundtrip_graph_and_result_invalidation():
    value = project.create("project:fixture", "Fixture", {"place": {"value": None, "evidence_refs": []}})
    value = project.put(value, _evidence())
    value = project.put(value, _signal())
    value = project.put(value, _computation())
    objects = {item["object_id"]: item for item in project.inspect(value)["objects"]}
    result = {"object_id": "result:position", "kind": "result", "label": "Position estimate", "content": {
        "value": 1.0, "input_revisions": {
            "signal:position": objects["signal:position"]["revision"],
            "computation:position": objects["computation:position"]["revision"],
        },
        "unit": "m", "frame": "frame:carriage", "time_basis": "clock:utc", "semantics": "estimated",
        "claim_scope": "declared_kinematic_model",
    }}
    value = project.put(value, result)
    value = project.connect(value, {"edge_id": "edge:model-result", "relation": "computation",
                                    "from": "computation:position", "to": "result:position", "resolution": "resolved"})
    value = project.connect(value, {"edge_id": "edge:evidence-result", "relation": "evidence",
                                    "from": "evidence:manual", "to": "result:position", "resolution": "resolved"})
    inspected = project.inspect(value)
    assert inspected["status"] == "declared"
    assert next(item for item in inspected["objects"] if item["object_id"] == "result:position")["result_status"] == "current_for_declared_inputs"
    reopened = json.loads(json.dumps(value))
    assert project.inspect(project.validate(reopened)) == inspected

    signal_revision = next(item for item in inspected["objects"] if item["object_id"] == "signal:position")["revision"]
    changed = deepcopy(_signal())
    changed["content"]["frame"] = "frame:inspection"
    value = project.put(value, changed, expected_revision=signal_revision)
    stale = next(item for item in project.inspect(value)["objects"] if item["object_id"] == "result:position")
    assert stale["result_status"] == "needs_reevaluation"


def test_project_keeps_unresolved_physical_edges_explicit_and_rejects_cycles():
    value = project.put(project.create("project:edges", "Edges"), _signal("signal:known"))
    value = project.connect(value, {"edge_id": "edge:unresolved", "relation": "physical",
                                    "from": "component:missing", "to": "signal:known", "resolution": "unresolved"})
    assert project.inspect(value)["status"] == "draft"
    with pytest.raises(ValueError, match="acyclic"):
        project.connect(value, {"edge_id": "edge:cycle", "relation": "computation",
                                "from": "signal:known", "to": "signal:known", "resolution": "resolved"})
    with pytest.raises(ValueError, match="Missing references"):
        project.connect(value, {"edge_id": "edge:bad-evidence", "relation": "evidence",
                                "from": "evidence:missing", "to": "signal:known", "resolution": "resolved"})
