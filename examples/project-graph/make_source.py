"""Author a small versioned project graph and print its exact workbench source.

The graph declares one evidence object, one observed signal, one computation
and one estimated result whose input revisions are pinned to the current signal
and computation revisions. Nothing here executes a computation or fetches
evidence; the operation only replays this history and reports its consistency.
"""
import json

from ciw import project_model as project
from ciw import project_workflow


def make_project():
    value = project.create("project:example-carriage", "Carriage position example",
                           {"place": {"value": None, "evidence_refs": []}})
    value = project.put(value, {"object_id": "evidence:manual", "kind": "evidence", "label": "Encoder manual",
                                "content": {"status": "documented", "source_digest": "sha256:" + "1" * 64,
                                            "locator": "fixture://encoder-manual", "subject_id": "machine:carriage"}})
    value = project.put(value, {"object_id": "signal:position", "kind": "signal", "label": "Carriage position",
                                "content": {"quantity": "position", "unit": "m", "frame": "frame:carriage",
                                            "time_basis": "clock:utc", "semantics": "observed"}})
    value = project.put(value, {"object_id": "computation:position", "kind": "computation", "label": "Position model",
                                "content": {"operation": "ciw.encoder-position.v1", "parameters": {"mode": "declared"}}})
    objects = {item["object_id"]: item for item in project.inspect(value)["objects"]}
    value = project.put(value, {"object_id": "result:position", "kind": "result", "label": "Position estimate",
                                "content": {"value": 1.0, "unit": "m", "frame": "frame:carriage",
                                            "time_basis": "clock:utc", "semantics": "estimated",
                                            "input_revisions": {
                                                "signal:position": objects["signal:position"]["revision"],
                                                "computation:position": objects["computation:position"]["revision"]}}})
    value = project.connect(value, {"edge_id": "edge:model-result", "relation": "computation",
                                    "from": "computation:position", "to": "result:position", "resolution": "resolved"})
    value = project.connect(value, {"edge_id": "edge:evidence-result", "relation": "evidence",
                                    "from": "evidence:manual", "to": "result:position", "resolution": "resolved"})
    return value


def source():
    value = make_project()
    return {"schema": project_workflow.SOURCE_SCHEMA, "experiment_id": "project:example-carriage",
            "configuration": dict(project_workflow.CONFIGURATION), "project": value,
            "request": {"expected_revision": value["revision"]}}


if __name__ == "__main__":
    print(json.dumps(source(), indent=2, sort_keys=True))
