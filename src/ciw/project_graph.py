"""The retained workbench as the ``ciw.project.v1`` spine.

Retained sources become evidence nodes. Each source kind's operation at its
recorded provider pins becomes one computation node, revised when the pins
change. Each retained bundle becomes a ``workflow_result`` node whose revision
binds only replay-invariant numerical result identities and the revisions of
every computational ancestor. Consequences:

* Reopen reads the graph: it is rebuilt from retained records; no provider runs.
* Replay re-evaluates a node: a replay bundle is a new occurrence of the same
  node. Identical numerical results keep its revision; different ones revise
  it, and every dependent result becomes ``needs_reevaluation``.
* A pin change (the provider revision or runtime a kind is now bound to)
  revises the computation node, so results computed under the old pins become
  ``needs_reevaluation``.

Execution occurrences (bundle, execution and result identities) stay in the
workbench records and appear here only as the occurrence map; evidence,
operation, execution, result and verification identities remain distinct.
Host paths are binding details, not pins, and never enter the graph.
"""
from __future__ import annotations

from copy import deepcopy
import re

from . import project_model
from .telemetry import digest

GRAPH_SCHEMA = "ciw.project-graph-view.v1"
PROJECT_ID = "project:workspace"
_HOST_PATH = re.compile(r"(/|[A-Za-z]:[\\/]|\\\\)")


def _host_path(value) -> bool:
    return isinstance(value, str) and _HOST_PATH.match(value) is not None


def pin_projection(runtime):
    """The runtime identity without host paths: what must match to stay current."""
    if isinstance(runtime, dict):
        return {key: pin_projection(value) for key, value in sorted(runtime.items()) if not _host_path(value)}
    if isinstance(runtime, list):
        return [pin_projection(value) for value in runtime if not _host_path(value)]
    return runtime


def pins_of(native: dict) -> dict:
    return {role: digest(pin_projection(runtime)) for role, runtime in sorted(native["runtimes"].items())}


def _short(identity: str) -> str:
    return identity.rsplit(":", 1)[-1][:16]


def build(state: dict, operations: dict, upstream: dict, *, current_pins: dict | None = None,
          label: str = "Retained workbench") -> tuple[dict, dict]:
    """Project a serialized workbench into a validated project and occurrence map.

    ``operations`` maps source kind to operation identity, ``upstream`` maps a
    bundle identity to the ordered upstream bundle identities it consumed and
    ``current_pins`` optionally maps a source kind to its currently bound pins.
    """
    project = project_model.create(PROJECT_ID, label)
    events, revisions, ancestors, edges = [], {}, {}, set()
    source_nodes, bundle_nodes, occurrences = {}, {}, {}

    def put(spec):
        revision = digest(spec)
        if revisions.get(spec["object_id"]) != revision:
            events.append(("put", spec))
            revisions[spec["object_id"]] = revision

    def connect(relation, origin, target):
        if (relation, origin, target) not in edges:
            edges.add((relation, origin, target))
            events.append(("connect", {"edge_id": f"edge:{len(edges)}", "relation": relation,
                                       "from": origin, "to": target, "resolution": "resolved"}))

    for source in state["sources"]:
        node = f"source:{_short(source['source_id'])}"
        source_nodes[source["source_id"]] = node
        put({"object_id": node, "kind": "evidence", "label": source["label"][:512],
             "content": {"status": "documented", "source_digest": source["evidence_id"],
                         "locator": f"ciw-source:{source['source_id']}"}})

    for record in state["bundles"]:
        kind, native = record["kind"], record["native"]
        operation = operations[kind]
        computation = f"computation:{kind}"
        put({"object_id": computation, "kind": "computation", "label": f"{operation} at declared pins",
             "content": {"operation": operation, "parameters": {"source_kind": kind, "pins": pins_of(native)}}})
        receipts = native.get("replay_receipts") or []
        origin = receipts[0]["source_bundle_digest"] if receipts else None
        node = bundle_nodes.get(origin) or f"result:{_short(record['bundle_id'])}"
        bundle_nodes[record["bundle_id"]] = node
        parents = [bundle_nodes[item] for item in upstream.get(record["bundle_id"], []) if item in bundle_nodes]
        lineage = {computation} | set(parents)
        for parent in parents:
            lineage |= ancestors.get(parent, set())
        ancestors[node] = lineage
        source_node = source_nodes[record["source_id"]]
        inputs = {ref: revisions[ref] for ref in sorted(lineage | {source_node})}
        put({"object_id": node, "kind": "workflow_result", "label": f"{kind} result",
             "content": {"operation": operation, "source_kind": kind,
                         "numerical_result_ids": [step["numerical_result_id"] for step in native["steps"]],
                         "input_revisions": inputs}})
        occurrences.setdefault(node, []).append({"bundle_id": record["bundle_id"], "revision": revisions[node],
                                                 "replay_of": origin})
        connect("computation", computation, node)
        for parent in parents:
            connect("computation", parent, node)
        connect("evidence", source_node, node)

    for kind, pins in sorted((current_pins or {}).items()):
        computation = f"computation:{kind}"
        if computation in revisions:
            put({"object_id": computation, "kind": "computation", "label": f"{operations[kind]} at declared pins",
                 "content": {"operation": operations[kind], "parameters": {"source_kind": kind, "pins": dict(pins)}}})
    if events:
        project = project_model.extend(project, events)
    return project, {"bundle_nodes": bundle_nodes, "occurrences": occurrences}


def view(state: dict, operations: dict, upstream: dict, *, current_pins: dict | None = None) -> dict:
    """A compact, read-only inspection of the graph spine."""
    project, occurrence_map = build(state, operations, upstream, current_pins=current_pins)
    inspected = project_model.inspect(project)
    nodes = []
    for record in inspected["objects"]:
        entry = {"node_id": record["object_id"], "kind": record["kind"], "revision": record["revision"],
                 "status": record["result_status"]}
        if record["kind"] == "workflow_result":
            entry["source_kind"] = record["content"]["source_kind"]
            entry["occurrences"] = deepcopy(occurrence_map["occurrences"][record["object_id"]])
            entry["current_occurrences"] = [item["bundle_id"] for item in entry["occurrences"]
                                            if item["revision"] == record["revision"]]
        if record["kind"] == "computation":
            entry["operation"] = record["content"]["operation"]
            entry["pins"] = deepcopy(record["content"]["parameters"]["pins"])
        nodes.append(entry)
    return {"schema": GRAPH_SCHEMA, "project_revision": project["revision"], "history_length": inspected["history_length"],
            "nodes": nodes, "edges": [{key: edge[key] for key in ("relation", "from", "to")} for edge in inspected["edges"]],
            "needs_reevaluation": [node["node_id"] for node in nodes if node["status"] == "needs_reevaluation"],
            "authority": inspected["authority"]}
