"""Offline verification of a saved workspace and its retained workbench.

``verify`` reopens a file exactly as ``ciw serve --workspace`` would, without a
provider, a process or a write outside a temporary directory, and reports what
the reopen established: whether every retained source and bundle validates,
and for each bundle whether this host could replay it. A provider-free
reference compares its current runtime identity with the retained one and
names the fields that differ, such as ``algorithm.kernel_probe`` on a host
whose linear-algebra kernels round differently; a provider-backed kind needs
a repository binding that a verification never supplies. Nothing is executed
beyond the deterministic reference checks a reopen already performs, and no
state is admitted.
"""
from __future__ import annotations

from pathlib import Path
import platform
import tempfile

import numpy as np

from . import reference_workflow as base
from .session import Session, read_json
from .workbench import OPERATIONS, Workbench, _validate_record, _workflow

SCHEMA = "ciw.workspace-verification.v1"
RETAINED_SCHEMAS = {"ciw.retained-workbench.v1", "ciw.retained-workbench.v2"}
MATCHES = "runtime_identity_matches"
DIFFERS = "runtime_identity_differs"
PROVIDER = "requires_provider_binding"


def host_identity():
    return {"python_version": platform.python_version(), "numpy_version": np.__version__,
            "kernel_probe": base.numerical_kernel_probe()}


def _form(value):
    if isinstance(value, dict) and type(value.get("workspace_version")) is int:
        return "workspace"
    if isinstance(value, dict) and value.get("schema") in RETAINED_SCHEMAS:
        return "retained-workbench"
    raise ValueError("Not a saved workspace or a retained workbench file")


def _replay_assessment(kind, native, available):
    workflow = _workflow(kind)
    if kind not in available or not hasattr(workflow, "_runtime_identity"):
        return {"replay_here": PROVIDER, "runtime_roles": sorted(native["runtimes"]), "differences": None}
    current = workflow._runtime_projection(workflow._runtime_identity())
    retained = workflow._runtime_projection(native["runtimes"][workflow.role])
    differences = base.identity_differences(current, retained)
    return {"replay_here": MATCHES if not differences else DIFFERS, "runtime_roles": [workflow.role],
            "differences": differences}


def _bundle_reports(workbench):
    available = {kind for kind, operation in OPERATIONS.items()
                 if any(row["available"] and row["operation_id"] == operation for row in workbench.describe_operations())}
    reports = []
    for summary in workbench.list_bundles():
        native = workbench.get_bundle(summary["bundle_id"])
        reports.append({
            "kind": summary["kind"], "bundle_id": summary["bundle_id"], "operation_id": summary["operation_id"],
            "source_id": summary["source_id"], "upstream_bundle_id": summary["upstream_bundle_id"],
            "execution_ids": summary["execution_ids"], "result_ids": summary["result_ids"],
            "numerical_result_ids": [step["numerical_result_id"] for step in native["steps"] if "numerical_result_id" in step],
            "retained_verification_outcome": summary["retained_verification_outcome"],
            "replay_receipts": len(native.get("replay_receipts", [])),
            "validation": "valid",
            **_replay_assessment(summary["kind"], native, available),
        })
    return reports


def _diagnose(retained):
    """Per-bundle validation after a refused reopen, so the report names what failed."""
    sources = {source.get("source_id"): source for source in retained.get("sources", []) if isinstance(source, dict)}
    reports = []
    for record in retained.get("bundles", []):
        entry = {"kind": record.get("kind") if isinstance(record, dict) else None,
                 "bundle_id": record.get("bundle_id") if isinstance(record, dict) else None}
        try:
            _validate_record(record, sources)
            entry["validation"] = "valid"
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            entry.update(validation="invalid", refusal=str(exc) or type(exc).__name__)
        reports.append(entry)
    return reports


def verify(path):
    """Reopen ``path`` offline and report the retained catalog's state on this host."""
    path = Path(path)
    value = read_json(path)
    form = _form(value)
    report = {"schema": SCHEMA, "path": str(path), "form": form,
              "workspace_version": value.get("workspace_version") if form == "workspace" else None,
              "host": host_identity(), "providers_executed": False, "state_admission": "not_performed",
              "verification": "content_consistent_without_execution"}
    retained = value.get("workbench", {"schema": "ciw.retained-workbench.v1", "revision": 0, "sources": [], "bundles": []}) \
        if form == "workspace" else value
    if isinstance(retained, dict):
        report["retained"] = {"schema": retained.get("schema"), "revision": retained.get("revision"),
                              "sources": len(retained.get("sources", []) or []),
                              "bundles": len(retained.get("bundles", []) or []),
                              "candidates": len(retained.get("candidates", []) or [])}
    try:
        if form == "workspace":
            with tempfile.TemporaryDirectory() as scratch:
                workbench = Session.from_workspace(path, Path(scratch) / "reopened").workbench
        else:
            workbench = Workbench.restore(retained)
    except ValueError as exc:
        report.update(status="invalid", refusal=str(exc),
                      bundles=_diagnose(retained) if isinstance(retained, dict) else [])
        return report
    bundles = _bundle_reports(workbench)
    report.update(status="valid", refusal=None, bundles=bundles,
                  replayable_here=sum(1 for entry in bundles if entry["replay_here"] == MATCHES))
    return report
