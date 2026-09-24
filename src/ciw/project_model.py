"""Pure, versioned project authoring; graph declarations never execute code.

History is replayed on every validation. Content digests identify declarations,
not authors or truth. Physical cycles are meaningful; computational cycles are
not. Retained results become needs_reevaluation when their pinned inputs drift.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .telemetry import canonical, digest

SCHEMA = "ciw.project.v1"
MAX_BYTES = 16 * 1024 * 1024
MAX_HISTORY = 4096
MAX_OBJECTS = 1024
MAX_EDGES = 2048
KINDS = {"component", "signal", "computation", "result", "evidence", "thermal-model",
         "evidence_bundle", "candidate_manifest", "challenge_report", "workflow_result"}
RESULT_KINDS = {"result", "workflow_result"}
SEMANTICS = {"observed", "estimated", "command"}
CONTEXT_FIELDS = {"place", "period", "purpose", "units", "resolution", "uncertainty", "use_policy"}
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _identifier(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("Require a bounded stable identifier")
    return value


def _text(value, maximum=512):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError("Require bounded nonempty text")


def _hash(value):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise ValueError("Require a canonical SHA256 digest")


def _json(value, depth=0):
    if depth > 16:
        raise ValueError("Project JSON nesting exceeds bound")
    if value is None or type(value) in (bool, int, float, str):
        if type(value) in (int, float) and (abs(value) > 1e100 or not math.isfinite(value)):
            raise ValueError("Require bounded finite project numbers")
        if type(value) is str and len(value) > 16384:
            raise ValueError("Project text exceeds bound")
        return
    if type(value) is list and len(value) <= 1024:
        for child in value:
            _json(child, depth + 1)
        return
    if type(value) is dict and len(value) <= 256 and all(type(key) is str and len(key) <= 128 for key in value):
        for child in value.values():
            _json(child, depth + 1)
        return
    raise ValueError("Require bounded JSON project content")


def _context(value):
    if value is None:
        return
    if type(value) is not dict or not set(value) <= CONTEXT_FIELDS:
        raise ValueError("Unknown project context field")
    for field in value.values():
        if type(field) is not dict or set(field) != {"value", "evidence_refs"}:
            raise ValueError("Context values require explicit evidence references")
        _json(field["value"])
        refs = field["evidence_refs"]
        if type(refs) is not list or len(refs) > 32 or len(set(refs)) != len(refs):
            raise ValueError("Invalid context evidence references")
        for ref in refs:
            _identifier(ref)
        if field["value"] is not None and not refs:
            raise ValueError("Known context values require evidence; use null for unknown")


def _descriptor(content):
    for key in ("unit", "frame", "time_basis"):
        _text(content.get(key), 128)
    if content.get("semantics") not in SEMANTICS:
        raise ValueError("Signal semantics must be observed, estimated or command")


def _object(spec):
    if type(spec) is not dict or set(spec) != {"object_id", "kind", "label", "content"}:
        raise ValueError("Object requires object_id, kind, label and content")
    _identifier(spec["object_id"])
    _text(spec["label"])
    kind, content = spec["kind"], spec["content"]
    if kind not in KINDS or type(content) is not dict:
        raise ValueError("Unsupported project object kind or content")
    _json(content)
    if kind == "signal":
        if set(content) != {"quantity", "unit", "frame", "time_basis", "semantics"}:
            raise ValueError("Signal requires quantity, unit, frame, time basis and semantics")
        _text(content["quantity"], 128)
        _descriptor(content)
    elif kind == "computation":
        if set(content) != {"operation", "parameters"} or type(content["parameters"]) is not dict:
            raise ValueError("Computation requires a declared operation and parameters")
        _identifier(content["operation"])
    elif kind == "result":
        required = {"value", "input_revisions", "unit", "frame", "time_basis", "semantics"}
        if not required <= set(content) or not set(content) <= required | {"claim_scope"}:
            raise ValueError("Result must retain its value, input revisions and signal descriptor")
        _descriptor(content)
        refs = content["input_revisions"]
        if type(refs) is not dict or not refs or len(refs) > 64:
            raise ValueError("Result requires bounded pinned input revisions")
        for ref, revision in refs.items():
            _identifier(ref)
            _hash(revision)
    elif kind == "workflow_result":
        # A retained workflow bundle as a computation node. Content holds only
        # replay-invariant numerical identities; execution occurrences stay in
        # the workbench records, never in this revision.
        if set(content) != {"operation", "source_kind", "numerical_result_ids", "input_revisions"}:
            raise ValueError("Workflow result requires operation, source kind, numerical results and input revisions")
        _identifier(content["operation"])
        _identifier(content["source_kind"])
        ids = content["numerical_result_ids"]
        if type(ids) is not list or not ids or len(ids) > 64:
            raise ValueError("Workflow result requires bounded numerical result identities")
        for value in ids:
            _text(value, 160)
        refs = content["input_revisions"]
        if type(refs) is not dict or not refs or len(refs) > 64:
            raise ValueError("Workflow result requires bounded pinned input revisions")
        for ref, revision in refs.items():
            _identifier(ref)
            _hash(revision)
    elif kind == "evidence":
        required = {"status", "source_digest", "locator"}
        if not required <= set(content) or not set(content) <= required | {"claims", "subject_id"}:
            raise ValueError("Evidence requires status, source digest and locator")
        if content["status"] not in {"documented", "observed", "validated", "unresolved"}:
            raise ValueError("Unknown evidence status")
        _hash(content["source_digest"])
        _text(content["locator"], 2048)
        if "subject_id" in content:
            _identifier(content["subject_id"])
    elif kind in {"evidence_bundle", "candidate_manifest", "challenge_report"}:
        from . import machine_manifest
        machine_manifest.validate(content)
        if content["role"] != kind:
            raise ValueError("Project object kind differs from artifact role")
    result = deepcopy(spec)
    result["revision"] = digest(spec)
    return result


def _edge(spec, objects):
    if type(spec) is not dict or set(spec) != {"edge_id", "relation", "from", "to", "resolution"}:
        raise ValueError("Edge requires ID, relation, endpoints and explicit resolution")
    for key in ("edge_id", "from", "to"):
        _identifier(spec[key])
    if spec["relation"] not in {"physical", "computation", "evidence"}:
        raise ValueError("Unknown edge relation")
    missing = [ref for ref in (spec["from"], spec["to"]) if ref not in objects]
    if spec["resolution"] not in {"resolved", "unresolved"}:
        raise ValueError("Unknown reference resolution")
    if missing and (spec["relation"] != "physical" or spec["resolution"] != "unresolved"):
        raise ValueError("Missing references are allowed only on explicitly unresolved physical edges")
    if spec["resolution"] == "unresolved" and spec["relation"] != "physical":
        raise ValueError("Computational and evidence references must resolve")
    if spec["relation"] == "evidence" and (
            spec["from"] not in objects or
            objects[spec["from"]]["kind"] not in {"evidence", "evidence_bundle", "challenge_report"}):
        raise ValueError("Evidence edge must originate at an evidence object")
    return deepcopy(spec)


def _dag(edges):
    adjacency = {}
    for edge in edges.values():
        if edge["relation"] == "computation":
            adjacency.setdefault(edge["from"], []).append(edge["to"])
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            raise ValueError("Computation graph must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)
    for node in adjacency:
        visit(node)


def _reaches(adjacency, start, target):
    if start == target:
        return True
    pending, seen = [start], {start}
    while pending:
        for child in adjacency.get(pending.pop(), ()):
            if child == target:
                return True
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return False


def _replay(project):
    if type(project) is not dict or set(project) != {"schema", "project_id", "history", "revision"} or project["schema"] != SCHEMA:
        raise ValueError("Require a ciw.project.v1 artifact")
    _identifier(project["project_id"])
    _hash(project["revision"])
    history = project["history"]
    if type(history) is not list or not 1 <= len(history) <= MAX_HISTORY:
        raise ValueError("Project history exceeds bound")
    if len(canonical(project)) > MAX_BYTES:
        raise ValueError("Project exceeds byte bound")
    objects, edges, seen_revisions, adjacency = {}, {}, {}, {}
    previous, label, context = None, None, None
    for index, event in enumerate(history):
        if type(event) is not dict or set(event) != {"sequence", "previous_revision", "operation", "payload", "revision"}:
            raise ValueError("Malformed project history event")
        if type(event["sequence"]) is not int or event["sequence"] != index or event["previous_revision"] != previous:
            raise ValueError("Broken project history chain")
        if event["revision"] != digest({key: value for key, value in event.items() if key != "revision"}):
            raise ValueError("Project history digest mismatch")
        payload, operation = event["payload"], event["operation"]
        if index == 0:
            if operation != "create" or type(payload) is not dict or set(payload) != {"project_id", "label", "context"} or payload["project_id"] != project["project_id"]:
                raise ValueError("Project must begin with its bound creation event")
            label, context = payload["label"], payload["context"]
            _text(label)
            _context(context)
        elif operation == "put":
            record = _object(payload)
            object_id = record["object_id"]
            if object_id in objects and objects[object_id]["kind"] != record["kind"]:
                raise ValueError("Stable object identifiers cannot change kind")
            if record["kind"] in RESULT_KINDS:
                for ref, revision in record["content"]["input_revisions"].items():
                    if revision not in seen_revisions.get(ref, set()):
                        raise ValueError("Result refers to an unknown historical input revision")
            objects[object_id] = record
            seen_revisions.setdefault(object_id, set()).add(record["revision"])
            if len(objects) > MAX_OBJECTS:
                raise ValueError("Project object count exceeds bound")
        elif operation == "connect":
            edge = _edge(payload, objects)
            if edge["edge_id"] in edges:
                raise ValueError("Edge identifiers must remain unique in project history")
            edges[edge["edge_id"]] = edge
            if len(edges) > MAX_EDGES:
                raise ValueError("Project edge count exceeds bound")
            if edge["relation"] == "computation":
                # Incremental acyclicity: the new edge closes a cycle exactly
                # when its target already reaches its source.
                if _reaches(adjacency, edge["to"], edge["from"]):
                    raise ValueError("Computation graph must be acyclic")
                adjacency.setdefault(edge["from"], []).append(edge["to"])
        elif operation == "context":
            _context(payload)
            context = payload
        else:
            raise ValueError("Unknown project history operation")
        previous = event["revision"]
    if previous != project["revision"]:
        raise ValueError("Project revision does not identify its final history event")
    return objects, edges, label, context


def validate(project):
    """Validate locally without fetching evidence or executing declarations."""
    try:
        _replay(project)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed project artifact") from exc
    return deepcopy(project)


def _append(project, operation, payload):
    value = deepcopy(project)
    event = {"sequence": len(value["history"]), "previous_revision": value["revision"],
             "operation": operation, "payload": deepcopy(payload)}
    event["revision"] = digest(event)
    value["history"].append(event)
    value["revision"] = event["revision"]
    return validate(value)


def extend(project, events):
    """Append several ``(operation, payload)`` events and validate once.

    Equivalent to successive put/connect/context calls, which each replay the
    whole history; used when a graph is projected from retained records.
    """
    value = deepcopy(project)
    for operation, payload in events:
        if operation not in {"put", "connect", "context"}:
            raise ValueError("Unknown project history operation")
        event = {"sequence": len(value["history"]), "previous_revision": value["revision"],
                 "operation": operation, "payload": deepcopy(payload)}
        event["revision"] = digest(event)
        value["history"].append(event)
        value["revision"] = event["revision"]
    return validate(value)


def create(project_id, label, context=None):
    _identifier(project_id)
    event = {"sequence": 0, "previous_revision": None, "operation": "create",
             "payload": {"project_id": project_id, "label": label, "context": deepcopy(context)}}
    event["revision"] = digest(event)
    return validate({"schema": SCHEMA, "project_id": project_id, "history": [event], "revision": event["revision"]})


def put(project, object_spec, expected_revision=None):
    """Return a new revision; existing history and caller inputs are untouched."""
    objects, _, _, _ = _replay(project)
    record = _object(object_spec)
    current = objects.get(record["object_id"])
    if expected_revision is not None and (current is None or current["revision"] != expected_revision):
        raise ValueError("Object changed since the expected revision")
    return _append(project, "put", object_spec)


def connect(project, edge_spec):
    objects, _, _, _ = _replay(project)
    _edge(edge_spec, objects)
    return _append(project, "connect", edge_spec)


def set_context(project, context):
    validate(project)
    _context(context)
    return _append(project, "context", context)


def inspect(project):
    objects, edges, label, context = _replay(project)
    parents = {}
    for edge in edges.values():
        if edge["relation"] == "computation":
            parents.setdefault(edge["to"], set()).add(edge["from"])
    def ancestors(node):
        found = set()
        pending = list(parents.get(node, ()))
        while pending:
            item = pending.pop()
            if item not in found:
                found.add(item)
                pending.extend(parents.get(item, ()))
        return found
    statuses = {}
    def result_status(object_id, visiting=None):
        if object_id in statuses:
            return statuses[object_id]
        record = objects[object_id]
        if record["kind"] not in RESULT_KINDS:
            return "declared"
        visiting = set() if visiting is None else set(visiting)
        if object_id in visiting:
            return "needs_reevaluation"
        visiting.add(object_id)
        bindings = record["content"]["input_revisions"]
        reasons = [ref for ref, revision in bindings.items() if objects[ref]["revision"] != revision or
                   (objects[ref]["kind"] in RESULT_KINDS and result_status(ref, visiting) == "needs_reevaluation")]
        missing_pins = ancestors(object_id) - set(bindings)
        status = "needs_reevaluation" if reasons or missing_pins else "current_for_declared_inputs"
        statuses[object_id] = status
        return status
    displayed = []
    for record in objects.values():
        displayed.append({**deepcopy(record), "result_status": result_status(record["object_id"])})
    gaps = [{"edge_id": edge["edge_id"], "status": "unresolved", "missing_refs":
             [ref for ref in (edge["from"], edge["to"]) if ref not in objects]}
            for edge in edges.values() if edge["resolution"] == "unresolved"]
    context_status = {}
    for key, field in (context or {}).items():
        context_status[key] = "unknown" if field["value"] is None else ("evidence_bound" if all(
            ref in objects and objects[ref]["kind"] in {"evidence", "evidence_bundle", "challenge_report"}
            for ref in field["evidence_refs"]) else "unresolved_evidence")
    return {"schema": "ciw.project-inspection.v1", "project_id": project["project_id"], "label": label,
            "revision": project["revision"], "objects": displayed, "edges": deepcopy(list(edges.values())),
            "history_length": len(project["history"]), "context": deepcopy(context), "context_status": context_status,
            "unresolved_physical_edges": gaps, "status": "draft" if gaps or "unresolved_evidence" in context_status.values() else "declared",
            "authority": {"execution": "not_performed", "physical_validation": "not_performed", "state_admission": "not_performed"}}
