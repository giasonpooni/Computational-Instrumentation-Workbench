"""Choose the next experiments from retained queue state.

The queue is persistent: its definition is fixed and its state is whatever the
retained reports say. The planner ranks what to do next without running
anything and without reading anything but retained reports (with the retained
hardware runs under each directory's ``hardware/``), the registered
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
   Every refinement is listed whatever the limit.
5. ``follow_up``: a completed task's recommended next step that points at a
   queue task which has not completed (``points_to``). Every follow-up is
   listed under ``follow_ups`` whatever the limit.
6. ``research``: open research questions: the part of a completed task's
   recommended next step that points at no queue task, and every deferred
   research question in any report's unresolved assumptions.

A pointer to a task that already completed is never proposed: it is listed
under ``stale_pointers``, and text it carries beyond the pointer (a later
sentence, a separate item, or a variant of the completed experiment) stays a
research question. A completed pointer whose step waits on a physical
execution ("before executing MFG-SCAN-01", "once a real study exists") is a
research question that revisits the pointed task, not a stale pointer.
Identical questions are listed once, with the other tasks under ``also_from``.

Hardware-blocked tasks stay listed as blocked until the hardware is bound or a
retained hardware run holds them. A task blocked in the main run whose latest
valid hardware run completed or partly completed it is ranked from that run's
report instead, marked with ``hardware_run``; the main state is kept and
nothing is relabelled. The planner never proposes treating a synthetic result
as a measurement. Reports do not record which providers were bound, so a task
that needs one must declare it in ``requires`` to be proposed once the
provider is bound.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile

from .registry import load_implementations, load_queue
from .runner import Context, builtin_identity, latest_hardware, load_reports

PRIORITY = ("ready", "unblocked", "retry", "refinement", "follow_up", "research")
SCHEMA = "ciw.lab-next.v2"
REASON_LIMIT = 300
# A completed pointer's trailing modifier ("with resolvability-aware step selection") is kept as a
# research variant only with at least this many words; shorter ones ("for its reductions") are pointers.
MIN_VARIANT_WORDS = 4

# A clause starts with a task pointer, optionally after a connective.
POINTER = re.compile(r"(?:(?i:then|and|also)\s+)?(T[0-9]{3})\b\s*")
# Before a later pointer, ',', 'and' or 'then' starts a new clause.
JOIN = re.compile(r"(?:\s*,\s*(?:(?i:and|then)\s+)?|\s+(?i:and|then)\s+)(?=T[0-9]{3}\b)")
CONNECTIVE = re.compile(r"^(?:(?i:then|and|also)\b[\s,]*)+")
DEFERRED = re.compile(r"deferred research question", re.IGNORECASE)
# A sentence ends at '.', '!' or '?' followed by whitespace and an upper-case letter (outside brackets and backticks),
# unless the word before it is a single letter or an abbreviation such as "e.g." or "cf.".
SENTENCE_END = re.compile(r"[.!?]\s+(?=[A-Z])")
ABBREVIATION = re.compile(r"(?:^|[\s(])(?:[A-Za-z]|e\.g|i\.e|cf|etc|vs|approx|Fig|Eq|Ref|Sec|No)\.$")
# A step that waits on a physical execution: kept as research revisiting the completed task it points at.
PHYSICAL_CONDITION = re.compile(r"\b(?:before|after|once|when|until)\b[^.;]*?\b(?:execut\w*|measur\w*|real|physical"
                                r"|hardware|captur\w*|acquir\w*|scann\w*|instrument\w*)\b", re.IGNORECASE)
DEFERRED_PREFIX = re.compile(r"^\s*deferred research question\s*[:\-–—]\s*", re.IGNORECASE)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def clip(value, limit: int = REASON_LIMIT) -> str:
    """Whitespace-normalized text cut at a word boundary with an ellipsis when longer than ``limit``."""
    text = " ".join(_text(value).split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:(") + "…"


def _key(text: str) -> str:
    """Deduplication key: case, spacing, trailing punctuation and a 'Deferred research question:' prefix ignored."""
    return " ".join(DEFERRED_PREFIX.sub("", text).casefold().split()).rstrip(" .;:")


def _closing(text: str, start: int) -> int | None:
    """Index of the parenthesis closing ``text[start]``, or None."""
    depth = 0
    for index in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return index
    return None


def clauses(text: str) -> list:
    """Top-level clauses of a next step: split at ';' and before a later task pointer joined by ',', 'and' or 'then'.

    Parentheses, brackets and backticks are opaque, so a pointer inside them
    never splits a clause.
    """
    parts, start, depth, code, index = [], 0, 0, False, 0
    while index < len(text):
        char = text[index]
        if char == "`":
            code = not code
        elif not code and char in "([{":
            depth += 1
        elif not code and char in ")]}":
            depth = max(0, depth - 1)
        elif not code and depth == 0:
            if char == ";":
                parts.append(text[start:index])
                start = index + 1
            elif (joined := JOIN.match(text, index)) and joined.end() > index:
                parts.append(text[start:index])
                start = index = joined.end()
                continue
        index += 1
    parts.append(text[start:])
    return [part.strip(" \t\n,.;") for part in parts if part.strip(" \t\n,.;")]


def sentences(text: str) -> list:
    """Sentences of ``text``; brackets, parentheses and backticks are opaque, abbreviations never end one."""
    parts, start, depth, code = [], 0, 0, False
    for index, char in enumerate(text):
        if char == "`":
            code = not code
        elif not code and char in "([{":
            depth += 1
        elif not code and char in ")]}":
            depth = max(0, depth - 1)
        elif (not code and depth == 0 and (end := SENTENCE_END.match(text, index))
              and not ABBREVIATION.search(text[start:index + 1])):
            parts.append(text[start:index + 1])
            start = end.end()
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _completed_pointer(task: str, clause: str, rest: str, stale: list) -> list:
    """Questions kept from one sentence pointing at completed ``task``; the pointer itself goes to ``stale``."""
    clause = CONNECTIVE.sub("", clause).strip().rstrip(" ,.;")
    if PHYSICAL_CONDITION.search(rest):
        return [("question", clause, task)]
    if rest.startswith(":"):
        rest = ""  # the colon form describes the pointed task
    elif rest.startswith("("):
        end = _closing(rest, 0)
        rest = rest[end + 1:] if end is not None else ""
    rest = rest.strip(" ,.;")
    separate = CONNECTIVE.match(rest) is not None
    remainder = CONNECTIVE.sub("", rest).strip()
    if remainder and separate:
        head = clause[:-len(rest)].rstrip(" ,") if clause.endswith(rest) else clause
        stale.append({"points_to": task, "text": clip(head)})
        return [("question", remainder, None)]
    if len(remainder.split()) >= MIN_VARIANT_WORDS:
        return [("question", clause, task)]
    stale.append({"points_to": task, "text": clip(clause)})
    return []


def next_step_items(text: str, own: str, completed: set) -> tuple[list, list]:
    """Split a recommended next step into kept items and stale pointers.

    Kept items are ``("pointer", task, clause)`` for a task that has not
    completed and ``("question", text, revisits)`` for text naming no queue
    task, where ``revisits`` is the completed task whose experiment the
    question varies (or None). The first sentence of a clause pointing at a
    completed task (``T002: ...``, ``T012 (...)``) becomes ``{"points_to",
    "text"}`` in the stale list; every later sentence stays a question, and so
    does text after its parenthetical when it is a separate item (``T012 (...)
    and a fold-scaling study``) or a variant of at least
    :data:`MIN_VARIANT_WORDS` words (``T005 (...) with resolvability-aware step
    selection``). A first sentence that waits on a physical execution
    (:data:`PHYSICAL_CONDITION`) is a question revisiting the pointed task.
    """
    kept, stale = [], []
    for clause in clauses(text):
        pointer = POINTER.match(clause)
        if pointer is None or pointer.group(1) == own:
            kept.append(("question", CONNECTIVE.sub("", clause).strip(), None))
            continue
        task = pointer.group(1)
        if task not in completed:
            kept.append(("pointer", task, CONNECTIVE.sub("", clause).strip()))
            continue
        # The pointer describes the pointed task in its own sentence; a later sentence is the task's own question.
        head, *later = sentences(clause)
        kept += _completed_pointer(task, head, head[pointer.end():], stale)
        for sentence in later:  # a later sentence may itself open with a pointer
            more, dropped = next_step_items(sentence, own, completed)
            kept, stale = kept + more, stale + dropped
    return kept, stale


def deferred_questions(report) -> list:
    """Deferred research questions a report records among its unresolved assumptions."""
    assumptions = report.get("unresolved_assumptions")
    return [item for item in assumptions if isinstance(item, str) and DEFERRED.search(item)] \
        if isinstance(assumptions, list) else []


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


def _marker(entry) -> dict:
    return {key: entry[key] for key in ("run_id", "date", "state", "evidence_status", "physical_validation_status",
                                        "hardware_measured")}


def _report_rows(base, report, completed, stale) -> list:
    """Refinement, follow-up and research rows from one partial or completed report."""
    kept, dropped = next_step_items(_text(report["recommended_next_task"]), report["task_id"], completed)
    stale += [dict(item, task_id=report["task_id"]) for item in dropped]
    rows = []
    if report["state"] == "partial":
        text = "; ".join(item[2] if item[0] == "pointer" else item[1] for item in kept) or _cause(report)
        rows.append(dict(base, kind="refinement", reason=clip(text), _key=_key(text)))
        return rows
    questions = [item for item in kept if item[0] == "question"]
    for _, task, clause in (item for item in kept if item[0] == "pointer"):
        rows.append(dict(base, kind="follow_up", points_to=task, reason=clip(clause), _key=_key(clause)))
    if questions:
        text = "; ".join(DEFERRED_PREFIX.sub("", item[1]) for item in questions)
        row = dict(base, kind="research", origin="recommended_next_task", reason=clip(text), _key=_key(text))
        revisits = sorted({item[2] for item in questions if item[2]})
        if revisits:
            row["revisits"] = revisits
        rows.append(row)
    return rows


def _deduplicated(rows) -> list:
    """One follow-up per pointed task and one research row per question; the others are named in ``also_from``."""
    claimed = {row["_key"] for row in rows if row["kind"] == "refinement"}
    first, kept = {}, []
    for row in rows:
        key = row.pop("_key", None)
        if row["kind"] == "follow_up":
            key = ("follow_up", row["points_to"])
        elif row["kind"] == "research":
            if key in claimed:
                continue
            key = ("research", key)
        else:
            kept.append(row)
            continue
        if key in first:
            if row["task_id"] != first[key]["task_id"] and row["task_id"] not in first[key].setdefault("also_from", []):
                first[key]["also_from"].append(row["task_id"])
            continue
        first[key] = row
        kept.append(row)
    for row in kept:
        if row.get("also_from") == []:
            del row["also_from"]
    return kept


def next_tasks(retained=None, providers=None, limit=10) -> dict:
    """Rank candidate next experiments.

    ``retained`` is a directory of lab reports, or a list of directories in
    precedence order (for example a work directory, then the retained
    reports); valid hardware runs under each directory's ``hardware/`` are
    read too. ``next`` holds the first ``limit`` ranked rows and every
    refinement; ``follow_ups`` and ``research`` list every follow-up and
    research row whatever the limit.
    """
    queue = load_queue()
    implementations, errors = load_implementations()
    single = retained is None or isinstance(retained, (str, Path))
    directories = ([] if retained is None else [retained]) if single else list(retained)
    reports = _reports(directories)
    hardware_problems: list = []
    hardware = latest_hardware(directories, hardware_problems)
    # A task blocked (or never reported) in the main run is ranked from its latest hardware run when that run
    # completed or partly completed it.
    driven = {task_id: entry for task_id, entry in hardware.items()
              if entry["state"] in ("completed", "partial")
              and (task_id not in reports or reports[task_id]["state"] in ("blocked", "deferred"))}
    effective = dict(reports, **{task_id: entry["report"] for task_id, entry in driven.items()})
    completed = {task_id for task_id, report in effective.items() if report["state"] == "completed"}
    rows, stale = [], []
    with tempfile.TemporaryDirectory(prefix="ciw-lab-plan-") as directory:
        probe = Context(Path(directory), providers)
        for item in queue["tasks"]:
            report, implementation = reports.get(item["id"]), implementations.get(item["id"])
            state = report["state"] if report else "not_run"
            base = {"task_id": item["id"], "title": item["title"], "section": item["section_key"], "state": state}
            if item["id"] in hardware:
                base["hardware_run"] = _marker(hardware[item["id"]])
            if implementation is None:
                continue
            if item["id"] in driven:
                rows += _report_rows(base, driven[item["id"]]["report"], completed, stale)
                continue
            missing = [need for need in implementation.requires if not probe.available(need)]
            if state in ("not_run", "deferred") and not missing:
                rows.append(dict(base, kind="ready", reason="Implemented and not yet reported"))
            elif state == "blocked" and implementation.requires and not missing:
                rows.append(dict(base, kind="unblocked",
                                 reason="Requirements now available: " + ", ".join(implementation.requires)))
            elif state == "blocked" and not implementation.requires and (why := _could_differ(report, implementation)):
                rows.append(dict(base, kind="retry", reason=clip(why + "; re-run to recheck: " + _cause(report))))
            elif state in ("not_run", "deferred", "blocked"):
                rows.append(dict(base, kind="blocked", reason=clip("Still unavailable: " + ", ".join(missing)
                                                                   if missing else _cause(report))))
            elif state in ("partial", "completed"):
                rows += _report_rows(base, report, completed, stale)
        for item in queue["tasks"]:
            report = effective.get(item["id"])
            if report is None or item["id"] not in implementations:
                continue
            base = {"task_id": item["id"], "title": item["title"], "section": item["section_key"],
                    "state": reports[item["id"]]["state"] if item["id"] in reports else "not_run"}
            if item["id"] in hardware:
                base["hardware_run"] = _marker(hardware[item["id"]])
            for question in deferred_questions(report):
                text = DEFERRED_PREFIX.sub("", question)
                rows.append(dict(base, kind="research", origin="unresolved_assumptions", reason=clip(text),
                                 _key=_key(question)))
    rows = _deduplicated(rows)
    order = {kind: index for index, kind in enumerate(PRIORITY)}
    ranked = sorted((r for r in rows if r["kind"] in order), key=lambda r: (order[r["kind"]], r["task_id"]))
    shown = ranked[:limit] + [r for r in ranked[limit:] if r["kind"] == "refinement"]
    directories_shown = (str(retained) if retained else None) if single else [str(d) for d in directories]
    return {"schema": SCHEMA, "retained": directories_shown,
            "next": shown, "still_blocked": [r for r in rows if r["kind"] == "blocked"],
            "follow_ups": [r for r in ranked if r["kind"] == "follow_up"],
            "research": [r for r in ranked if r["kind"] == "research"], "stale_pointers": stale,
            "hardware_runs": sorted({entry["run_id"] for entry in hardware.values()}),
            "hardware_run_problems": hardware_problems,
            "unimplemented": sorted(t["id"] for t in queue["tasks"] if t["id"] not in implementations),
            "section_import_errors": errors}
