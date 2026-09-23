"""Choose the next experiments from retained queue state.

The queue is persistent: its definition is fixed and its state is whatever the
retained reports say. The planner ranks what to do next without running
anything and without reading anything but retained reports, the registered
implementations and requirement probes:

1. ``ready``: implemented tasks never reported, or reported as deferred.
2. ``unblocked``: blocked tasks whose hard requirements are now available.
3. ``refinement``: partial tasks, carrying their own recommended next step.
4. ``follow_up``: completed tasks' recommended next tasks (research questions).

Hardware-blocked tasks stay listed as blocked until the hardware is bound; the
planner never proposes treating a synthetic result as a measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .registry import load_implementations, load_queue
from .runner import Context, load_reports

PRIORITY = ("ready", "unblocked", "refinement", "follow_up")


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def next_tasks(retained=None, providers=None, limit=10) -> dict:
    """Rank candidate next experiments; ``retained`` is a directory of lab reports."""
    queue = load_queue()
    implementations, errors = load_implementations()
    reports = {r["task_id"]: r for r in load_reports(retained)} if retained and Path(retained, "reports").is_dir() else {}
    with tempfile.TemporaryDirectory(prefix="ciw-lab-plan-") as directory:
        probe = Context(Path(directory), providers)
        rows = []
        for item in queue["tasks"]:
            report, implementation = reports.get(item["id"]), implementations.get(item["id"])
            state = report["state"] if report else "not_run"
            base = {"task_id": item["id"], "title": item["title"], "section": item["section_key"], "state": state}
            if implementation is None:
                continue
            missing = [need for need in implementation.requires if not probe.available(need)]
            if state in ("not_run", "deferred") and not missing:
                rows.append(dict(base, kind="ready", reason="Implemented and not yet reported"))
            elif state == "blocked" and implementation.requires and not missing:
                rows.append(dict(base, kind="unblocked",
                                 reason="Requirements now available: " + ", ".join(implementation.requires)))
            elif state in ("not_run", "deferred", "blocked"):
                rows.append(dict(base, kind="blocked", reason="Still unavailable: " + ", ".join(missing)
                                 if missing else _text(report["experiment"])[:200]))
            elif state == "partial":
                rows.append(dict(base, kind="refinement", reason=_text(report["recommended_next_task"])[:300]))
            elif state == "completed":
                rows.append(dict(base, kind="follow_up", reason=_text(report["recommended_next_task"])[:300]))
    order = {kind: index for index, kind in enumerate(PRIORITY)}
    ranked = sorted((r for r in rows if r["kind"] in order), key=lambda r: (order[r["kind"]], r["task_id"]))
    return {"schema": "ciw.lab-next.v1", "retained": str(retained) if retained else None,
            "next": ranked[:limit], "still_blocked": [r for r in rows if r["kind"] == "blocked"],
            "unimplemented": sorted(t["id"] for t in queue["tasks"] if t["id"] not in implementations),
            "section_import_errors": errors}
