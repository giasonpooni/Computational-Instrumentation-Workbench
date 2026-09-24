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
from .numerical_backend import linear_algebra_backend
from .session import Session, read_json
from .workbench import Workbench, _validate_record, _workflow

SCHEMA = "ciw.workspace-verification.v1"
RETAINED_SCHEMAS = {"ciw.retained-workbench.v1", "ciw.retained-workbench.v2"}
MATCHES = "runtime_identity_matches"
DIFFERS = "runtime_identity_differs"
PROVIDER = "requires_provider_binding"


def host_identity():
    return {"python_version": platform.python_version(), "numpy_version": np.__version__,
            "kernel_probe": base.numerical_kernel_probe(), "linear_algebra": linear_algebra_backend()}


def _form(value):
    if isinstance(value, dict) and type(value.get("workspace_version")) is int:
        return "workspace"
    if isinstance(value, dict) and value.get("schema") in RETAINED_SCHEMAS:
        return "retained-workbench"
    raise ValueError("Not a saved workspace or a retained workbench file")


def _replay_assessment(kind, runtimes, available):
    workflow = _workflow(kind)
    if kind not in available or not hasattr(workflow, "_runtime_identity"):
        return {"replay_here": PROVIDER, "runtime_roles": sorted(runtimes), "differences": None}
    current = workflow._runtime_projection(workflow._runtime_identity())
    retained = workflow._runtime_projection(runtimes[workflow.role])
    differences = base.identity_differences(current, retained)
    return {"replay_here": MATCHES if not differences else DIFFERS, "runtime_roles": [workflow.role],
            "differences": differences}


def bundle_reports(workbench):
    """Per-bundle identities and replayability on this host for a restored or live workbench."""
    available = {row["source_kind"] for row in workbench.describe_operations()
                 if row.get("available") and "source_kind" in row}
    facts = {fact["bundle_id"]: fact for fact in workbench.bundle_facts()}
    reports = []
    for summary in workbench.list_bundles():
        fact = facts[summary["bundle_id"]]
        reports.append({
            "kind": summary["kind"], "bundle_id": summary["bundle_id"], "operation_id": summary["operation_id"],
            "source_id": summary["source_id"], "upstream_bundle_id": summary["upstream_bundle_id"],
            "execution_ids": summary["execution_ids"], "result_ids": summary["result_ids"],
            "numerical_result_ids": fact["numerical_result_ids"],
            "retained_verification_outcome": summary["retained_verification_outcome"],
            "replay_receipts": fact["replay_receipts"],
            "validation": "valid",
            **_replay_assessment(summary["kind"], fact["runtimes"], available),
        })
    return reports


def _listed(value, key):
    items = value.get(key) if isinstance(value, dict) else None
    return items if isinstance(items, list) else []


def _counts(retained):
    if not isinstance(retained, dict):
        return None
    return {"schema": retained.get("schema"), "revision": retained.get("revision"),
            "sources": len(_listed(retained, "sources")), "bundles": len(_listed(retained, "bundles")),
            "candidates": len(_listed(retained, "candidates"))}


def _diagnose(retained):
    """Per-bundle validation after a refused reopen, so the report names what failed."""
    sources = {source.get("source_id"): source for source in _listed(retained, "sources") if isinstance(source, dict)}
    reports = []
    for record in _listed(retained, "bundles"):
        entry = {"kind": record.get("kind") if isinstance(record, dict) else None,
                 "bundle_id": record.get("bundle_id") if isinstance(record, dict) else None}
        try:
            _validate_record(record, sources)
            entry["validation"] = "valid"
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
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
    empty = {"schema": "ciw.retained-workbench.v1", "revision": 0, "sources": [], "bundles": []}
    retained = value.get("workbench", empty) if form == "workspace" else value
    report["retained"] = _counts(retained)
    try:
        if form == "workspace":
            with tempfile.TemporaryDirectory() as scratch:
                workbench = Session.from_workspace(path, Path(scratch) / "reopened").workbench
        else:
            workbench = Workbench.restore(retained)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        # A file whose purpose is to be diagnosed always yields a report.
        report.update(status="invalid", refusal=str(exc) or type(exc).__name__, bundles=_diagnose(retained))
        return report
    bundles = bundle_reports(workbench)
    report.update(status="valid", refusal=None, bundles=bundles,
                  replayable_here=sum(1 for entry in bundles if entry["replay_here"] == MATCHES))
    return report
