"""MCP adapter for the lab queue (``ciw_lab_mcp``), served over stdio.

An assistant connected through MCP can list tasks, read reports, plan the next
experiments, run tasks into this server's own run directory, verify that run
against retained reports and label an existing workspace. It cannot state a
label: no tool accepts a finding, a label, a report or a physical result, and
every report returned has been revalidated, so evidence status can only come
from ``ciw.lab.evidence``. Running a task never acquires hardware data.

The run directory must be separate from the retained directory, so retained
reports are never rewritten: neither may contain the other, compared both as
resolved paths and as file identities (so a second mount point or a case
variant is caught), and no entry of one may be the same file as an entry of
the other (symlinked ``reports``/``artifacts`` directories, hard-linked copies).
The file check is repeated before every run. Tools run in worker threads;
those that read or write the run directory are serialized, within a process by
a lock and across every server process on the same directory by an OS lock on
``.ciw-lab-mcp.lock`` in it, because a run replaces reports and artifact
directories in place.

Requires the optional ``mcp`` extra. The numerical core does not import this
module or depend on MCP.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import errno
import functools
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading

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
STATES = ("completed", "partial", "blocked", "deferred", "not_run")
LOCK_FILE = ".ciw-lab-mcp.lock"


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


def _identity(path) -> tuple | None:
    """(st_dev, st_ino) of a path with links followed, or None when it cannot be read."""
    try:
        status = os.stat(path)
    except (OSError, ValueError):
        return None
    return (status.st_dev, status.st_ino) if status.st_ino else None


def _ancestry(path: Path) -> set:
    return {identity for identity in map(_identity, (path, *path.parents)) if identity}


def _entries(root: Path) -> dict:
    """Identities of ``root`` and every entry beneath it (links followed), each with one of its paths."""
    found = {}
    for directory, directories, files in os.walk(root):
        for path in (directory, *(os.path.join(directory, name) for name in directories + files)):
            identity = _identity(path)
            if identity is not None:
                found.setdefault(identity, path)
    return found


def _separate(retained, workdir) -> None:
    """Refuse a run directory that is, contains, lies inside or shares a file with the retained directory.

    Resolved paths miss a second mount point, a case variant on a
    case-insensitive file system, symlinked subdirectories and hard links, so
    file identities are compared as well.
    """
    if retained is None:
        return
    kept, work = Path(retained).resolve(), Path(workdir).resolve()
    refusal = f"The work directory {workdir} must be separate from the retained directory {retained}: "
    if (kept == work or kept.is_relative_to(work) or work.is_relative_to(kept)
            or _identity(kept) in _ancestry(work) or _identity(work) in _ancestry(kept)):
        raise ValueError(refusal + "runs replace reports and delete artifact directories in the work directory")
    kept_entries = _entries(kept)
    for identity, path in _entries(work).items():
        if identity in kept_entries:
            raise ValueError(refusal + f"{path} is the same file as {kept_entries[identity]}, "
                             "and a run would rewrite or delete it through the link")


@contextmanager
def workdir_lock(workdir):
    """Hold an exclusive OS lock on ``workdir`` so every server process using it takes turns."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / LOCK_FILE, "a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:  # LK_LOCK gives up after ten seconds; a run may take longer
                    if exc.errno != errno.EDEADLOCK:
                        raise
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # released when the handle closes
            yield


def _task_of(problem: str) -> str | None:
    head = problem.split(":", 1)[0]
    return head if TASK_ID.fullmatch(head) else None


def _compare_tasks(retained, workdir, tasks) -> dict:
    """compare() restricted to ``tasks``: passed to compare when it accepts them, and problems filtered."""
    if "tasks" in inspect.signature(compare).parameters:
        result = compare(retained, workdir, tasks=list(tasks))
    else:
        result = compare(retained, workdir)
    wanted = set(tasks)
    kept = [p for p in result["problems"] if _task_of(p) is None or _task_of(p) in wanted]
    dropped = len(result["problems"]) - len(kept)
    return dict(result, problems=kept, passed=not kept and (result["passed"] or dropped > 0))


def build_server(retained, workdir, providers=None):
    """Create the MCP server; ``workdir`` receives runs, ``retained`` is read only."""
    _separate(retained, workdir)
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

    lock = threading.Lock()

    def exclusive(function):
        """One workdir tool at a time, across threads and server processes: runs rewrite reports
        non-atomically and delete artifact directories."""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with lock, workdir_lock(workdir):
                return function(*args, **kwargs)
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
    @exclusive
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
        queue = load_queue()
        sections = sorted({item["section_key"] for item in queue["tasks"]})
        if state is not None and state not in STATES:
            raise ValueError(f"Unknown state {state!r}; use one of: {', '.join(STATES)}")
        if section is not None and section not in sections:
            raise ValueError(f"Unknown section {section!r}; use one of: {', '.join(sections)}")
        known = {}
        for directory in reversed(sources()):
            known.update({r["task_id"]: r for r in load_reports(directory)})
        rows = []
        for item in queue["tasks"]:
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
    @exclusive
    def get_report(task_id: str, response_format: str = "markdown") -> str:
        """Return one task's revalidated nineteen-question report (run directory first, then retained)."""
        report, directory = _report(task_id, sources())
        return _format(dict(report, source_directory=directory), response_format,
                       render_markdown(report) + f"\n_Source: {directory}_\n")

    @server.tool(name="ciw_lab_plan_next", annotations=read_only)
    @guarded
    @exclusive
    def plan_next(limit: int = 10, response_format: str = "markdown") -> str:
        """Rank the next experiments: ready, newly unblocked, retry, partial (all listed), follow-up tasks, then
        open research questions. Runs nothing.

        Reports come from this server's run directory first, then the retained reports and their retained
        hardware runs; a row ranked from a hardware run names it. Pointers to completed tasks are listed as
        stale, never proposed. The json format lists every follow-up under 'follow_ups' and every research
        question under 'research', whatever the limit.
        """
        plan = next_tasks(sources() or None, providers, max(1, min(limit, 50)))

        def line(row):
            marker = f" [hardware run {row['hardware_run']['run_id']}]" if row.get("hardware_run") else ""
            target = f" -> {row['points_to']}" if row.get("points_to") else ""
            return f"- {row['task_id']} ({row['kind']}{target}){marker}: {row['title']} — {row['reason']}"
        lines = [line(r) for r in plan["next"]]
        lines += [line(r) for r in plan["follow_ups"] if r not in plan["next"]]  # never hidden by the limit
        lines += ["", f"Still blocked: {len(plan['still_blocked'])}; follow-ups: {len(plan['follow_ups'])}; "
                      f"research questions: {len(plan['research'])}; "
                      f"stale pointers: {len(plan['stale_pointers'])}; unimplemented: {len(plan['unimplemented'])}"]
        if plan["hardware_run_problems"]:
            lines.append(f"Hardware runs ignored for integrity problems: {len(plan['hardware_run_problems'])}")
        return _format(plan, response_format, "\n".join(lines))

    # Destructive: a run deletes the tasks' artifact directories and replaces their
    # reports and run-log.json in the run directory (never the retained one).
    @server.tool(name="ciw_lab_run_tasks", annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
    @guarded
    @exclusive
    def run_tasks(task_ids: list[str]) -> str:
        """Execute up to 20 queue tasks into this server's run directory and return their states and labels.

        Deletes and replaces earlier reports, artifacts and the run log of the
        same tasks in the run directory only; retained reports are never
        modified. Physical data is never acquired.
        """
        if not task_ids or len(task_ids) > MAX_TASKS_PER_RUN or not all(TASK_ID.fullmatch(t) for t in task_ids):
            raise ValueError(f"Pass 1-{MAX_TASKS_PER_RUN} task identities such as ['T003', 'T005']")
        _separate(retained, workdir)  # links into the retained directory may have appeared since startup
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
    @exclusive
    def verify_run() -> str:
        """Compare the tasks in this server's run directory with their retained reports (states, labels, tolerances).

        Retained tasks that were not run here are listed under not_regenerated
        and do not fail the comparison.
        """
        if retained is None or not (retained / "reports").is_dir():
            raise ValueError("The server was started without a retained report directory")
        kept = {r["task_id"] for r in load_reports(retained)}
        if not kept:
            raise ValueError(f"The retained directory {retained} holds no reports; there is nothing to compare against")
        ran = sorted(r["task_id"] for r in load_reports(workdir)) if (workdir / "reports").is_dir() else []
        if not ran:
            raise ValueError("The run directory has no reports yet; run tasks with ciw_lab_run_tasks first")
        result = _compare_tasks(retained, workdir, ran)
        result.update(compared=len(kept & set(ran)), tasks=ran, not_regenerated=sorted(kept - set(ran)))
        return json.dumps(result, indent=1, sort_keys=True)

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
    _separate(retained, workdir)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    build_server(retained, workdir, providers).run("stdio")
