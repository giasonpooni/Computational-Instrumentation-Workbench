"""Project-graph compiler seam in the shared operation, execution and result envelopes.

The operation replays a retained ``ciw.project.v1`` history and compiles it into
a deterministic plan: the topological order of declared computations, the
inputs each retained result pins, the results that need reevaluation and the
typed model checks CIW can perform without execution. Declared operations are
reported, never dispatched; stale results, drafts and unresolved evidence are
findings rather than refusals.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
from pathlib import Path

from . import machine_manifest
from . import project_model
from . import thermal_contract
from .native_workflow import NativeReferenceWorkflow, parse_source, text
from .telemetry import canonical, digest, _keys

KIND = "project-graph"
SCHEMA = "ciw.project-graph-session.v1"
SOURCE_SCHEMA = "ciw.project-graph-source.v1"
DATA_SCHEMA = "ciw.project-graph-plan.v1"
RESULT_SCHEMA = "ciw.project-graph-workbench-result.v1"
VERIFY_SCHEMA = "ciw.project-graph-verification.v1"
OPERATION = "ciw.project-graph.v1"
ROLE = "project"
CONFIGURATION = {
    "profile": "project_graph_compile",
    "execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}
AUTHORITY = {
    "execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}
CLAIM_SCOPE = "deterministic_compilation_of_declared_project_history_without_execution"
SOURCE_LIMIT = 512 * 1024


def _request(value, project):
    _keys(value, {"project_revision", "targets"})
    if value["project_revision"] != project["revision"]:
        raise ValueError("Compile request must name the retained project revision")
    targets = value["targets"]
    if targets is not None:
        if type(targets) is not list or not 1 <= len(targets) <= 64 or len(set(targets)) != len(targets):
            raise ValueError("Compile targets must be null or a bounded unique list")
        for target in targets:
            project_model._identifier(target)
    return deepcopy(value)


def validate_source(raw):
    source = parse_source(raw, SOURCE_LIMIT, "Project graph")
    _keys(source, {"schema", "experiment_id", "configuration", "project", "request"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(CONFIGURATION):
        raise ValueError("Unsupported project graph source or authority policy")
    text(source["experiment_id"], 128)
    project = project_model.validate(source["project"])
    objects, _, _, _ = project_model._replay(project)
    request = _request(source["request"], project)
    for target in request["targets"] or ():
        if target not in objects:
            raise ValueError("Compile target is not a declared project object")
    return deepcopy(source)


def _topological(nodes, edges):
    """Kahn order with lexical tie-break so equal histories compile equally."""
    children, indegree = {node: [] for node in nodes}, {node: 0 for node in nodes}
    for edge in edges:
        children[edge["from"]].append(edge["to"])
        indegree[edge["to"]] += 1
    ready = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    return order


def _model_checks(objects):
    checks = []
    for object_id in sorted(objects):
        record = objects[object_id]
        if record["kind"] == "thermal-model":
            try:
                thermal_contract.validate_model(record["content"])
                status, reason = "contract_valid", None
            except (ValueError, TypeError, KeyError) as exc:
                status, reason = "contract_invalid", str(exc)[:256]
            checks.append({"object_id": object_id, "revision": record["revision"],
                           "contract": "ciw.thermal-observer-request.v1#model", "status": status, "reason": reason})
    by_role = {role: [objects[i] for i in sorted(objects) if objects[i]["kind"] == role]
               for role in ("evidence_bundle", "candidate_manifest", "challenge_report")}
    for candidate in by_role["candidate_manifest"]:
        content = candidate["content"]
        evidence = next((item for item in by_role["evidence_bundle"]
                         if item["content"]["artifact_digest"] == content["evidence_bundle_digest"]), None)
        report = next((item for item in by_role["challenge_report"]
                       if item["content"]["candidate_digest"] == content["artifact_digest"]), None)
        entry = {"object_id": candidate["object_id"], "revision": candidate["revision"],
                 "contract": machine_manifest.CANDIDATE_SCHEMA, "compiled_manifest_digest": None}
        if evidence is None or report is None:
            entry.update(status="unlinked", reason="Candidate lacks a declared evidence bundle or challenge report")
        else:
            try:
                compiled = machine_manifest.compile(content, evidence["content"], report["content"])
                entry.update(status="compiled", reason=None, compiled_manifest_digest=compiled["artifact_digest"])
            except (ValueError, TypeError, KeyError) as exc:
                entry.update(status="refused_by_manifest_compiler", reason=str(exc)[:256])
        checks.append(entry)
    return checks


def compile_plan(source):
    """Deterministic plan data; no clocks, UUIDs or live registries."""
    project, request = source["project"], source["request"]
    objects, edges, _, _ = project_model._replay(project)
    inspection = project_model.inspect(project)
    status = {item["object_id"]: item["result_status"] for item in inspection["objects"]}
    computation = [edge for edge in edges.values() if edge["relation"] == "computation"]
    parents = {}
    for edge in computation:
        parents.setdefault(edge["to"], set()).add(edge["from"])
    selected = set(objects)
    if request["targets"] is not None:
        selected, pending = set(), list(request["targets"])
        while pending:
            node = pending.pop()
            if node not in selected:
                selected.add(node)
                pending.extend(parents.get(node, ()))
    order = _topological(sorted(selected), [e for e in computation if e["from"] in selected and e["to"] in selected])
    evidence = {}
    for edge in edges.values():
        if edge["relation"] == "evidence" and edge["to"] in objects:
            evidence.setdefault(edge["to"], []).append(edge["from"])
    steps = []
    for object_id in order:
        record = objects[object_id]
        step = {"object_id": object_id, "kind": record["kind"], "revision": record["revision"],
                "parents": sorted(parents.get(object_id, set()) & selected),
                "evidence_refs": sorted(evidence.get(object_id, ())), "result_status": status[object_id]}
        if record["kind"] == "computation":
            step["declared_operation"] = record["content"]["operation"]
            step["parameters_digest"] = digest(record["content"]["parameters"])
            step["dispatch"] = "not_performed"
        if record["kind"] == "result":
            pins = record["content"]["input_revisions"]
            step["input_revisions"] = deepcopy(pins)
            step["drifted_inputs"] = sorted(ref for ref, revision in pins.items()
                                            if objects[ref]["revision"] != revision)
            step["unpinned_ancestors"] = sorted(_ancestors(object_id, parents) - set(pins))
        steps.append(step)
    return {
        "schema": DATA_SCHEMA,
        "operation_id": OPERATION,
        "project_id": project["project_id"],
        "project_revision": project["revision"],
        "history_length": len(project["history"]),
        "targets": deepcopy(request["targets"]),
        "project_status": inspection["status"],
        "order": order,
        "steps": steps,
        "needs_reevaluation": sorted(item for item in selected if status[item] == "needs_reevaluation"),
        "unresolved_physical_edges": deepcopy(inspection["unresolved_physical_edges"]),
        "context_status": deepcopy(inspection["context_status"]),
        "unevidenced_results": sorted(item for item in selected
                                      if objects[item]["kind"] == "result" and not evidence.get(item)),
        "model_checks": _model_checks({key: objects[key] for key in selected}),
        "inspection_digest": digest(inspection),
        "claim_scope": CLAIM_SCOPE,
        "authority": deepcopy(AUTHORITY),
    }


def _ancestors(node, parents):
    found, pending = set(), list(parents.get(node, ()))
    while pending:
        item = pending.pop()
        if item not in found:
            found.add(item)
            pending.extend(parents.get(item, ()))
    return found


class ProjectGraphWorkflow(NativeReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    operation = OPERATION
    role = ROLE
    SOURCE_SCHEMA = SOURCE_SCHEMA
    RESULT_SCHEMA = RESULT_SCHEMA
    VERIFY_SCHEMA = VERIFY_SCHEMA
    PROFILE = "ciw.project-graph.python-reference.v1"
    MAX_BYTES = 4 * 1024 * 1024
    SOURCE_LIMIT = SOURCE_LIMIT
    AUTHORITY = AUTHORITY
    FILES = (Path(project_model.__file__), Path(machine_manifest.__file__),
             Path(thermal_contract.__file__), Path(__file__))
    LABEL = "project graph compiler"

    def validate_source(self, raw):
        return validate_source(raw)

    def evaluate(self, source):
        return compile_plan(source)
