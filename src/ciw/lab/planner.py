"""Choose the next experiments from retained queue state.

The queue is persistent: its definition is fixed and its state is whatever the
retained reports say. The planner ranks what to do next without running
anything and without reading anything but retained reports, the registered
implementations and requirement probes:

1. ``ready``: implemented tasks never reported, or reported as deferred.
2. ``unblocked``: blocked tasks whose hard requirements are now available.
3. ``retry``: blocked tasks that declare no requirement, only when a re-run
   could end differently: the block was an unexpected exception (possibly
   transient), or the built-in runtime identity the report recorded (task
   sources and library versions) differs from the current one. Otherwise
   such a task stays blocked, since re-running it unchanged reproduces the
   same report. Either way the reason is the report's unresolved assumptions.
4. ``refinement``: partial tasks, carrying their own recommended next step.
5. ``follow_up``: completed tasks' recommended next tasks (research questions).

Hardware-blocked tasks stay listed as blocked until the hardware is bound; the
planner never proposes treating a synthetic result as a measurement. Reports do
not record which providers were bound, so a task that needs one must declare it
in ``requires`` to be proposed once the provider is bound.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .registry import load_implementations, load_queue
from .runner import Context, builtin_identity, load_reports

PRIORITY = ("ready", "unblocked", "retry", "refinement", "follow_up")


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def _cause(report) -> str:
    """A blocked report's own reason: its unresolved assumptions, else its experiment."""
    assumptions = report.get("unresolved_assumptions")
    return ("; ".join(map(_text, assumptions)) if isinstance(assumptions, list) and assumptions
            else _text(report["experiment"]))


def _could_differ(report, implementation) -> str | None:
    """Why re-running a task blocked without a declared requirement could end differently, or None."""
    assumptions = report.get("unresolved_assumptions")
    if isinstance(assumptions, list) and any(isinstance(a, str) and a.startswith("Blocked by unexpected")
                                             for a in assumptions):
        return "Blocked by an unexpected exception, which may be transient"
    recorded = report.get("provider_runtime_identity")
    if (isinstance(recorded, dict) and recorded.get("implementation") == "ciw.lab"
            and recorded != builtin_identity(implementation.changed_files)):
        return "Its sources or runtime changed since the blocked report"
    return None


def _reports(directories) -> dict:
    """Reports merged across directories; the first directory holding a task's report wins."""
    reports = {}
    for directory in reversed(directories):
        if Path(directory, "reports").is_dir():
            reports.update({r["task_id"]: r for r in load_reports(directory)})
    return reports


def next_tasks(retained=None, providers=None, limit=10) -> dict:
    """Rank candidate next experiments.

    ``retained`` is a directory of lab reports, or a list of directories in
    precedence order (for example a work directory, then the retained reports).
    """
    queue = load_queue()
    implementations, errors = load_implementations()
    single = retained is None or isinstance(retained, (str, Path))
    directories = ([] if retained is None else [retained]) if single else list(retained)
    reports = _reports(directories)
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
            elif state == "blocked" and not implementation.requires and (why := _could_differ(report, implementation)):
                rows.append(dict(base, kind="retry", reason=(why + "; re-run to recheck: " + _cause(report))[:300]))
            elif state in ("not_run", "deferred", "blocked"):
                rows.append(dict(base, kind="blocked", reason="Still unavailable: " + ", ".join(missing)
                                 if missing else _cause(report)[:200]))
            elif state == "partial":
                rows.append(dict(base, kind="refinement", reason=_text(report["recommended_next_task"])[:300]))
            elif state == "completed":
                rows.append(dict(base, kind="follow_up", reason=_text(report["recommended_next_task"])[:300]))
    order = {kind: index for index, kind in enumerate(PRIORITY)}
    ranked = sorted((r for r in rows if r["kind"] in order), key=lambda r: (order[r["kind"]], r["task_id"]))
    shown = (str(retained) if retained else None) if single else [str(d) for d in directories]
    return {"schema": "ciw.lab-next.v1", "retained": shown,
            "next": ranked[:limit], "still_blocked": [r for r in rows if r["kind"] == "blocked"],
            "unimplemented": sorted(t["id"] for t in queue["tasks"] if t["id"] not in implementations),
            "section_import_errors": errors}
