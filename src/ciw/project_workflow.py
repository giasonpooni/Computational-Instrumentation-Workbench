"""Provider-free CIW adapter for the versioned project graph.

The project graph is an authoring contract: typed components, signals,
computations, results and evidence with separate physical, computation and
evidence edges in an append-only, content-addressed history.  This operation
retains one exact project artifact, replays its history through the
independent Python reference and records the resulting inspection as a native
result with its own operation, execution and result identities.  It never
executes a declared computation, fetches evidence, admits state or authorizes
an action.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import platform
import re

from . import project_model as project
from . import reference_workflow as base
from .adapters.subprocess import _json
from .telemetry import canonical, _keys

KIND = "project-graph"
SCHEMA = "ciw.project-graph-session.v1"
SOURCE_SCHEMA = "ciw.project-graph-source.v1"
DATA_SCHEMA = "ciw.project-graph-workbench-data.v1"
RESULT_SCHEMA = "ciw.project-graph-workbench-result.v1"
VERIFY_SCHEMA = "ciw.project-graph-verification.v1"
OPERATION = "ciw.project-graph.v1"
PROFILE = "ciw.project-graph.python-reference.v1"
ROLE = "project"
ROLES = set()
# A retained bundle holds the source bytes, the inspection, its numerical copy
# and one reproduction; the source limit keeps the whole bundle inside MAX_BYTES.
SOURCE_LIMIT = 512 * 1024
MAX_BYTES = 4 * 1024 * 1024
CONFIGURATION = {
    "profile": "versioned_project_graph",
    "activation": "read_only",
    "declared_computation_execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
}
AUTHORITY = {
    "declared_computation_execution": "not_performed",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
}
CLAIM_SCOPE = "declared_graph_consistency_and_result_staleness_under_retained_history"
_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _text(value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


@lru_cache(maxsize=1)
def _algorithm_identity():
    return base.algorithm_identity(PROFILE, [Path(project.__file__), Path(base.__file__), Path(__file__)],
                                   python_version=platform.python_version())


def runtime_identity():
    return {
        "schema": base.RUNTIME_SCHEMA,
        "role": ROLE,
        "profile": PROFILE,
        "algorithm": _algorithm_identity(),
        "execution_scope": base.EXECUTION_SCOPE,
        "declared_computation_execution": "not_performed",
        "physical_validation": "not_performed",
        "state_admission": "not_performed",
    }


def _request(value, artifact):
    _keys(value, {"expected_revision"})
    if not isinstance(value["expected_revision"], str) or not _REVISION.fullmatch(value["expected_revision"]):
        raise ValueError("Requested project revision must be a canonical SHA256 digest")
    if value["expected_revision"] != artifact["revision"]:
        raise ValueError("Project revision differs from the requested revision; reselect the project")
    return deepcopy(value)


def validate_source(raw):
    """Validate exact source bytes and replay the project history without executing it."""
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Project graph source requires bounded exact JSON bytes")
    source = _json(raw)
    _keys(source, {"schema", "experiment_id", "configuration", "project", "request"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(CONFIGURATION):
        raise ValueError("Unsupported project graph source or authority policy")
    _text(source["experiment_id"], 128)
    artifact = project.validate(source["project"])
    _request(source["request"], artifact)
    if len(canonical(source)) > SOURCE_LIMIT:
        raise ValueError("Project graph source exceeds the byte budget")
    return deepcopy(source)


def _native_data(source):
    inspection = project.inspect(source["project"])
    objects = inspection["objects"]
    counts = {kind: sum(1 for item in objects if item["kind"] == kind) for kind in sorted(project.KINDS)}
    edges = {relation: sum(1 for edge in inspection["edges"] if edge["relation"] == relation)
             for relation in ("physical", "computation", "evidence")}
    statuses = {status: sum(1 for item in objects if item["kind"] == "result" and item["result_status"] == status)
                for status in ("current_for_declared_inputs", "needs_reevaluation")}
    return {
        "schema": DATA_SCHEMA,
        "operation_id": OPERATION,
        "project_id": inspection["project_id"],
        "project_revision": inspection["revision"],
        "history_length": inspection["history_length"],
        "request": deepcopy(source["request"]),
        "inspection": inspection,
        "summary": {
            "status": inspection["status"],
            "object_counts": counts,
            "edge_counts": edges,
            "result_statuses": statuses,
            "unresolved_physical_edges": len(inspection["unresolved_physical_edges"]),
            "context_status": deepcopy(inspection["context_status"]),
        },
        "claim_scope": CLAIM_SCOPE,
        "authority": deepcopy(AUTHORITY),
    }


class ProjectGraphWorkflow(base.ReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    SOURCE_SCHEMA = SOURCE_SCHEMA
    result_schema = RESULT_SCHEMA
    verify_schema = VERIFY_SCHEMA
    operation = OPERATION
    role = ROLE
    label = "project graph"
    ROLES = ROLES
    MAX_BYTES = MAX_BYTES
    AUTHORITY = AUTHORITY

    def _source(self, raw):
        return validate_source(raw)

    def _native_data(self, source):
        return _native_data(source)

    def _runtime_identity(self):
        return runtime_identity()

    def _configuration(self, source):
        return deepcopy(CONFIGURATION)


def _verification(bundle, reproduced):
    return ProjectGraphWorkflow()._verification(bundle, reproduced)
