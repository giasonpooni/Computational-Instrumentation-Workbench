"""MCP adapter for the lab queue (``ciw_lab_mcp``), served over stdio.

An assistant connected through MCP can list tasks, read reports, plan the next
experiments, run tasks into this server's own run directory, verify that run
against retained reports and label an existing workspace. It cannot state a
label: no tool accepts a finding, a label, a report or a physical result, and
every report returned has been revalidated, so evidence status can only come
from ``ciw.lab.evidence``. Running a task never acquires hardware data.

Requires the optional ``mcp`` extra. The numerical core does not import this
module or depend on MCP.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import functools
import json
from pathlib import Path
import re
import sys

from .bridge import classify_workspace
from .evidence import BOUNDARY, LABELS
from .planner import next_tasks
from .registry import load_queue
from .report import render_markdown, validate_report
from .runner import compare, load_reports, run_queue

SERVER_NAME = "ciw_lab_mcp"
MAX_TASKS_PER_RUN = 20
INSTRUCTIONS = (
    "CIW computational-experimentalist queue. Evidence labels (analytic, synthetic, numerically_verified, "
    "provider_backed, hardware_measured, independently_verified, not_established) are assigned by the server "
    "from each finding's recorded basis; no tool accepts or changes a label. Physical, calibration, safety and "
    "authority claims are not_established unless acquired hardware evidence exists, and running tasks never "
    "acquires any. Use ciw_lab_plan_next to choose work, ciw_lab_run_tasks to execute it, ciw_lab_get_report "
    "to read the nineteen-question report and ciw_lab_verify_run to compare with retained reports.")
TASK_ID = re.compile(r"T[0-9]{3}")


def _format(value, response_format, markdown):
    if response_format not in ("markdown", "json"):
        raise ValueError("response_format must be 'markdown' or 'json'")
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) if response_format == "json" else markdown


def _report(task_id, directories):
    if not TASK_ID.fullmatch(task_id):
        raise ValueError(f"Task identities look like T001; got {task_id!r}. Use ciw_lab_list_tasks to find them.")
    for directory in directories:
        path = Path(directory) / "reports" / f"{task_id}.json"
        if path.is_file():
            return validate_report(json.loads(path.read_text(encoding="utf-8"))), str(directory)
    raise ValueError(f"No report for {task_id} in the run or retained directories; run it with ciw_lab_run_tasks.")


def build_server(retained, workdir, providers=None):
    """Create the MCP server; ``workdir`` receives runs, ``retained`` is read only."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    def guarded(function):
        """Report anticipated refusals to the client; other exceptions remain crashes."""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except (ValueError, OSError) as exc:
                raise ToolError(str(exc)) from exc
        return wrapper

    retained = Path(retained) if retained else None
    workdir = Path(workdir)
    providers = dict(providers or {})
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    def sources():
        return [d for d in (workdir, retained) if d is not None and (Path(d) / "reports").is_dir()]

    @server.tool(name="ciw_lab_list_tasks", annotations=read_only)
    @guarded
    def list_tasks(section: str | None = None, state: str | None = None, offset: int = 0, limit: int = 50,
                   response_format: str = "markdown") -> str:
        """List queue tasks with their state and primary evidence label.

        State comes from this server's run directory first, then the retained
        reports. Filter by section key (e.g. 'geodesic-jacobi') or state
        ('completed', 'partial', 'blocked', 'deferred', 'not_run'). Paginate with
        offset/limit (limit 1-200). response_format is 'markdown' or 'json'.
        """
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("Use 0 <= offset and 1 <= limit <= 200")
        known = {}
        for directory in reversed(sources()):
            known.update({r["task_id"]: r for r in load_reports(directory)})
        rows = []
        for item in load_queue()["tasks"]:
            report = known.get(item["id"])
            row = {"task_id": item["id"], "section": item["section_key"], "title": item["title"],
                   "state": report["state"] if report else "not_run",
                   "evidence_status": report["evidence_status"]["primary"] if report else "not_established"}
            if (section is None or row["section"] == section) and (state is None or row["state"] == state):
                rows.append(row)
        page = rows[offset:offset + limit]
        result = {"total": len(rows), "count": len(page), "offset": offset, "items": page,
                  "has_more": offset + limit < len(rows), "next_offset": offset + limit if offset + limit < len(rows) else None}
        lines = [f"{len(rows)} tasks; showing {offset + 1}-{offset + len(page)}", ""] + [
            f"- {r['task_id']} [{r['state']}, {r['evidence_status']}] {r['title']}" for r in page]
        return _format(result, response_format, "\n".join(lines))

    @server.tool(name="ciw_lab_get_report", annotations=read_only)
    @guarded
    def get_report(task_id: str, response_format: str = "markdown") -> str:
        """Return one task's revalidated nineteen-question report (run directory first, then retained)."""
        report, directory = _report(task_id, sources())
        return _format(dict(report, source_directory=directory), response_format,
                       render_markdown(report) + f"\n_Source: {directory}_\n")

    @server.tool(name="ciw_lab_plan_next", annotations=read_only)
    @guarded
    def plan_next(limit: int = 10, response_format: str = "markdown") -> str:
        """Rank the next experiments: ready, newly unblocked, partial, then follow-up tasks. Runs nothing."""
        plan = next_tasks(sources()[0] if sources() else None, providers, max(1, min(limit, 50)))
        lines = [f"- {r['task_id']} ({r['kind']}): {r['title']} — {r['reason']}" for r in plan["next"]]
        lines += ["", f"Still blocked: {len(plan['still_blocked'])}; unimplemented: {len(plan['unimplemented'])}"]
        return _format(plan, response_format, "\n".join(lines))

    @server.tool(name="ciw_lab_run_tasks", annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    @guarded
    def run_tasks(task_ids: list[str]) -> str:
        """Execute up to 20 queue tasks into this server's run directory and return their states and labels.

        Replaces earlier reports of the same tasks in the run directory only;
        retained reports are never modified. Physical data is never acquired.
        """
        if not task_ids or len(task_ids) > MAX_TASKS_PER_RUN or not all(TASK_ID.fullmatch(t) for t in task_ids):
            raise ValueError(f"Pass 1-{MAX_TASKS_PER_RUN} task identities such as ['T003', 'T005']")
        with redirect_stdout(sys.stderr):  # stdout carries the MCP protocol
            summary = run_queue(workdir, task_ids, providers)
        reports = {r["task_id"]: r for r in load_reports(workdir)}
        lines = [f"Ran {summary['tasks']} tasks into {workdir}: {summary['states']}", ""]
        for task_id in sorted(task_ids):
            report = reports[task_id]
            lines.append(f"- {task_id} {report['state']} · {report['evidence_status']['primary']} · "
                         f"physical validation {report['physical_validation_status']['status']}")
        return "\n".join(lines)

    @server.tool(name="ciw_lab_verify_run", annotations=read_only)
    @guarded
    def verify_run() -> str:
        """Compare this server's run directory with the retained reports (states, labels, tolerances)."""
        if retained is None or not (retained / "reports").is_dir():
            raise ValueError("The server was started without a retained report directory")
        return json.dumps(compare(retained, workdir), indent=1, sort_keys=True)

    @server.tool(name="ciw_lab_classify_workspace", annotations=read_only)
    @guarded
    def classify(workspace_path: str) -> str:
        """Label every result retained in a saved CIW workspace file; binds no provider and edits nothing."""
        path = Path(workspace_path)
        if not path.is_file():
            raise ValueError(f"No workspace file at {path}")
        return json.dumps(classify_workspace(path), indent=1, sort_keys=True)

    @server.tool(name="ciw_lab_explain_labels", annotations=read_only)
    @guarded
    def explain_labels() -> str:
        """Explain the seven evidence labels and what computation can and cannot establish."""
        from . import evidence
        rows = "\n".join(f"| {left} | {right} |" for left, right in BOUNDARY)
        return (evidence.__doc__ + "\n\n| Computation may establish | It cannot establish alone |\n| --- | --- |\n"
                + rows + "\n\nLabels: " + ", ".join(LABELS))

    return server


def serve(retained, workdir, providers=None) -> None:
    """Run the adapter over stdio; nothing else may write to stdout."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    build_server(retained, workdir, providers).run("stdio")
