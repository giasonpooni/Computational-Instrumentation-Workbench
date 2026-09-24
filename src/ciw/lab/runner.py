"""Run queue tasks, retain their reports and artifacts, and verify regenerations.

Every task in the queue gets a report on every full run. Unimplemented tasks
and tasks whose hard requirements are unavailable are reported as deferred or
blocked with the reason; an unexpected exception is retained as ``blocked``
with its message instead of being hidden. Nothing here acquires physical data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from .. import __version__
from .evidence import PHYSICAL_DOMAINS, EvidenceRefusal, validate_finding
from .registry import SECTION_MODULES, base_section_modules, load_implementations, load_queue
from .report import FIELDS, FIELD_NAMES, build_report, render_markdown, validate_report

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Retained evidence is committed; one artifact larger than this is refused so
# tables stay summaries (sample, aggregate or truncate long trajectories).
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


def dumps(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def repository_root() -> Path | None:
    """Directory holding the repository's ``examples/`` and ``tests/`` for tasks that read them.

    ``CIW_LAB_REPOSITORY_ROOT`` wins (the clean-room reproduction points it at
    copies); otherwise a source checkout around the package is used. Returns
    None in an installed package without either, and such tasks report blocked.
    """
    declared = os.environ.get("CIW_LAB_REPOSITORY_ROOT")
    if declared:
        return Path(declared)
    candidate = PACKAGE_ROOT.parents[1]
    return candidate if (candidate / "examples").is_dir() else None


def repository_path(*parts: str) -> Path | None:
    """A path under :func:`repository_root`, or None when no repository files are reachable."""
    root = repository_root()
    return None if root is None else root.joinpath(*parts)


def source_digest(relative: str) -> str | None:
    """SHA-256 of a package source file with normalized line endings."""
    if not relative.startswith("src/ciw/"):
        return None
    path = PACKAGE_ROOT / relative[len("src/ciw/"):]
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


OPTIONAL_MODULES = ("scipy", "sympy", "mpmath")


def builtin_identity(changed_files) -> dict:
    """The workbench runtime: sources, interpreter and the optional reference modules that import here.

    An installed optional module that fails on import is recorded under
    ``optional_module_errors`` (it is not present), and one without a string
    ``__version__`` as ``unknown (...)``; identifying the runtime never aborts a run.
    """
    sources = {name: source_digest(name) for name in changed_files if source_digest(name)}
    identity = {"implementation": "ciw.lab", "ciw_version": __version__,
                "python": platform.python_version(), "numpy": np.__version__, "sources": sources}
    errors = {}
    for optional in OPTIONAL_MODULES:
        try:
            if importlib.util.find_spec(optional) is None:
                continue
            module = importlib.import_module(optional)
        except Exception as exc:  # recorded, never raised
            errors[optional] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            identity[optional] = str(module.__version__)
        except Exception as exc:
            identity[optional] = f"unknown ({type(exc).__name__}: {exc})"
    if errors:
        identity["optional_module_errors"] = errors
    return identity


# Runtime caches a provider run may create inside its checkout; they never shadow tracked sources.
RUNTIME_CACHES = (".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


def git_identity(path: Path) -> dict:
    """HEAD, tree and cleanliness of a provider checkout; refuses non-repositories.

    Untracked files make the checkout dirty, ignored ones included: an
    untracked module can shadow pinned tracked code on import whatever
    ``.gitignore``, ``.git/info/exclude`` or ``core.excludesFile`` say. So does
    a tracked file flagged skip-worktree or assume-unchanged, whose bytes
    ``git status`` never compares (``ls-files -v`` tags it ``S`` or in lower case).
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    caches = [f":(top,exclude,glob)**/{name}/**" for name in RUNTIME_CACHES]
    status = git("status", "--porcelain", "--untracked-files=all", "--ignored", "--", ":(top)", *caches)
    overlooked = [line for line in git("ls-files", "-v", "--", ":(top)").splitlines()
                  if line[:1] == "S" or line[:1].islower()]
    return {"path": str(path), "revision": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(status or overlooked)}


class Context:
    """Per-run services: artifact retention, shared memo and requirement probes."""

    def __init__(self, output_dir: Path, providers: dict | None = None):
        self.output_dir = Path(output_dir)
        self.providers = {role: Path(path) for role, path in (providers or {}).items()}
        self._memo: dict = {}
        self.task_id: str | None = None
        self.artifacts: list = []
        self.hardware: set = set()
        self.probes: dict = {}
        self._recording: list = []  # the probes of each memo computation in progress

    def begin(self, task_id: str) -> None:
        self.task_id, self.artifacts, self.hardware, self.probes = task_id, [], set(), {}

    def memo(self, key, compute):
        """Share one deterministic computation between tasks in a run.

        Every task that uses the value records the requirement probes its
        computation made, so a report does not depend on which tasks ran
        before it. A replayed hardware probe is not the task's own: a physical
        finding still needs a probe that succeeded in its task.
        """
        if key not in self._memo:
            made: dict = {}
            self._recording.append(made)
            try:
                value = compute()
            finally:
                self._recording.pop()
            self._memo[key] = (value, made)
        value, made = self._memo[key]
        for requirement, present in made.items():
            self._probed(requirement, present)
        return value

    def _probed(self, requirement: str, present: bool) -> None:
        self.probes[requirement] = present
        for made in self._recording:
            made[requirement] = present

    def available(self, requirement: str) -> bool:
        """Whether a requirement is met here; the task's report records each outcome (the latest per requirement)."""
        kind, _, name = requirement.partition(":")
        if kind == "module":
            present = importlib.util.find_spec(name) is not None
        elif kind == "provider":
            present = name in self.providers and self.providers[name].exists()
        elif kind == "tool":
            present = shutil.which(name) is not None
        elif kind == "hardware":
            present = bool(_probe_hardware(name))
            if present:
                self.hardware.add(name)  # a physical finding needs a probe that succeeded in its task
        else:
            raise ValueError(f"Unsupported lab requirement: {requirement}")
        self._probed(requirement, present)
        return present

    def _write(self, name: str, data: bytes) -> str:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Artifact names are single file names")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"Artifact {name} exceeds {MAX_ARTIFACT_BYTES} bytes; retain a summary instead")
        directory = self.output_dir / "artifacts" / self.task_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
        relative = f"artifacts/{self.task_id}/{name}"
        # A rewrite replaces the earlier entry: one entry per path, matching the bytes on disk.
        self.artifacts[:] = [a for a in self.artifacts if a["path"] != relative]
        self.artifacts.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return relative

    def artifact_json(self, name: str, value) -> str:
        return self._write(name, dumps(value).encode("utf-8"))

    def artifact_text(self, name: str, text: str) -> str:
        return self._write(name, text.encode("utf-8"))


# Linux powercap sysfs directory holding the intel-rapl energy counters.
POWERCAP = Path("/sys/class/powercap")


def _probe_hardware(name: str) -> bool:
    if name == "nvidia-gpu":
        if shutil.which("nvidia-smi") is None:
            return False
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "GPU" in result.stdout
    if name == "rapl":
        # The counters often exist but are readable only by root: a counter answers only if it reads as a number.
        for path in POWERCAP.glob("intel-rapl:*/energy_uj"):
            try:
                int(path.read_text().strip())
                return True
            except (OSError, ValueError):
                continue
        return False
    return False


# Outcomes of one node (parametrized cases, repeated records) fold by severity:
# any failure wins over a pass, and a pass over a skip, whatever the JUnit order.
SEVERITY = {"skipped": 0, "passed": 1, "failed": 2}


def _fold(outcome: str | None, value: str) -> str:
    return value if outcome is None else max(outcome, value, key=SEVERITY.__getitem__)


def _node_id(classname: str, name: str, root: Path | None) -> str:
    """pytest node id of a JUnit test case.

    The module is the longest dotted prefix of ``classname`` that is a ``.py``
    file under ``root``; the remaining components are test classes. Without the
    sources, classes start at the first later component with pytest's ``Test``
    prefix.
    """
    parts = classname.split(".") if classname else []
    split = None
    if root is not None:
        split = next((i for i in range(len(parts), 0, -1)
                      if root.joinpath(*parts[:i - 1], parts[i - 1] + ".py").is_file()), None)
    if split is None:
        split = next((i for i in range(1, len(parts)) if parts[i].startswith("Test")), len(parts))
    return "::".join(["/".join(parts[:split]) + ".py", *parts[split:], name])


def read_junit(path: Path | None, root: Path | None = None) -> dict:
    """Map pytest node ids to passed, skipped or failed.

    ``root`` is the pytest root directory the class names are relative to
    (default: :func:`repository_root`).
    """
    if path is None:
        return {}
    root = repository_root() if root is None else Path(root)
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = _node_id(case.get("classname", ""), case.get("name", ""), root)
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes[node] = _fold(outcomes.get(node), outcome)
    return outcomes


def _test_fields(implementation, findings, junit):
    passed, skipped, failed = [], [], []
    for record in findings:
        for check in record["basis"].get("checks") or []:
            (passed if check.get("passed") else failed).append(f"check: {record['claim']} vs {check['reference']}")
        independent = record["basis"].get("independent_check")
        if independent:
            (passed if independent.get("passed") else failed).append(
                f"independent: {record['claim']} vs {independent['checker']['implementation']}")
    for node in implementation.regression_tests if implementation else ():
        outcome = None
        for key, value in junit.items():
            # The registered node itself or its parametrized cases; never a suffix of another path.
            if key == node or key.startswith(node + "["):
                outcome = _fold(outcome, value)
        if outcome == "passed":
            passed.append(f"pytest: {node}")
        elif outcome == "failed":
            failed.append(f"pytest: {node}")
        else:
            skipped.append(f"pytest: {node} ({'skipped' if outcome == 'skipped' else 'not run in this invocation'})")
    return passed, skipped, failed


def _default_fields(task, reason):
    return {
        "hypothesis": "Not formulated: the task has not been executed.",
        "mathematical_model": "Not stated.",
        "input_data": [],
        "observation_model": "No observation was made.",
        "expected_invariant": "Not stated.",
        "experiment": reason,
        "numerical_result": "none",
        "uncertainty": "not quantified",
        "failure_modes_checked": [],
        "unresolved_assumptions": [reason],
        "recommended_next_task": f"Implement {task['id']}: {task['title']}",
    }


NOT_STATED = "Not stated by the implementation."


def _executed_defaults(task):
    """Answers an executed task's implementation omitted: said to be unstated, never 'not executed'."""
    return {name: NOT_STATED if isinstance(value, str) else [NOT_STATED]
            for name, value in _default_fields(task, "").items()}


def _gate_physical(findings, ctx):
    """Refuse physical labels unless hardware answered a probe and the raw bytes were retained."""
    retained = {artifact["sha256"] for artifact in ctx.artifacts}
    for record in findings:
        if record["domain"] in PHYSICAL_DOMAINS and record["evidence_status"] != "not_established":
            digest = record["basis"]["acquisition"]["raw_sha256"]
            if not ctx.hardware:
                raise EvidenceRefusal(f"Hardware evidence refused for '{record['claim']}': no hardware probe succeeded in this task")
            if digest not in retained:
                raise EvidenceRefusal(f"Hardware evidence refused for '{record['claim']}': raw bytes {digest[:12]} are not a retained artifact")


def _blocked(task, reason, failure):
    fields = _default_fields(task, reason)
    fields["failure_modes_checked"] = [failure]
    return "blocked", fields, []


def _with_probes(identity, ctx, state, requires):
    """The runtime identity plus the requirement probes the task queried and their outcomes (none: unchanged).

    A blocked task that declares no requirement keeps its identity unchanged:
    the planner retries it only when that identity differs from the current
    built-in one, which recorded probes always would.
    """
    if not ctx.probes or not isinstance(identity, dict) or (state == "blocked" and not requires):
        return identity
    return dict(identity, requirement_probes=dict(sorted(ctx.probes.items())))


def _unavailable(requires, ctx) -> list:
    missing = []
    for need in requires:
        try:
            if not ctx.available(need):
                missing.append(need)
        except Exception as exc:  # an unprobeable requirement cannot be met; the task reports blocked
            missing.append(f"{need} ({type(exc).__name__}: {exc})")
    return missing


def run_task(task, implementation, ctx: Context, junit: dict, import_error: str | None = None) -> dict:
    ctx.begin(task["id"])
    findings, state, executed = [], "deferred", False
    if implementation is None:
        reason = f"Deferred: {import_error or 'no implementation is registered for this task'}."
        fields = _default_fields(task, reason)
    else:
        missing = _unavailable(implementation.requires, ctx)
        if missing:
            reason = "Blocked: unavailable requirement(s) " + ", ".join(missing) + "."
            fields = _default_fields(task, reason)
            fields["unresolved_assumptions"] = [reason]
            state = "blocked"
            plan = getattr(implementation.run, "plan", None)
            if plan:
                fields.update({k: v for k, v in plan.items() if k in FIELD_NAMES})
                fields["experiment"] = reason + " Planned: " + str(plan.get("experiment", ""))
                # A blocked task may record the claims it cannot establish, and nothing else.
                try:
                    findings = [validate_finding(record) for record in plan.get("findings", [])]
                    if any(record["evidence_status"] != "not_established" for record in findings):
                        raise EvidenceRefusal("Planned findings of a blocked task must be not_established")
                except EvidenceRefusal as exc:
                    state, fields, findings = _blocked(task, reason + f" Plan findings refused: {exc}", str(exc))
        else:
            try:
                executed = True
                outcome = implementation.run(ctx)
                state = outcome.get("state", "completed")
                fields = _executed_defaults(task)
                fields.update(outcome.get("fields", {}))
                findings = outcome.get("findings", [])
                for record in findings:
                    validate_finding(record)
                _gate_physical(findings, ctx)
            except Exception as exc:  # retained as a blocked report, never hidden
                reason = f"Blocked by unexpected {type(exc).__name__}: {exc}"
                state, fields, findings = _blocked(task, reason,
                                                   traceback.format_exception_only(type(exc), exc)[-1].strip())
    changed = list(implementation.changed_files) if implementation else []
    requires = implementation.requires if implementation else ()
    fields.setdefault("changed_files", changed)
    if not fields.get("changed_files"):
        fields["changed_files"] = changed
    fields["generated_artifacts"] = deepcopy(ctx.artifacts)
    if not fields.get("provider_runtime_identity"):
        fields["provider_runtime_identity"] = builtin_identity(changed)
    fields["provider_runtime_identity"] = _with_probes(fields["provider_runtime_identity"], ctx, state, requires)
    # Checks count as tests only when the implementation ran; a blocked plan's checks never executed.
    passed, skipped, failed = _test_fields(implementation, findings if executed else [], junit)
    fields["tests_passed"], fields["tests_skipped"] = passed, skipped
    if failed and state == "completed":
        state = "partial"
    try:
        report = build_report(task, state, {k: v for k, v in fields.items() if k in FIELD_NAMES
                                            and k not in ("evidence_status", "physical_validation_status")},
                              findings, extra={"tests_failed": failed} if failed else None)
        return validate_report(report)
    except (EvidenceRefusal, ValueError, TypeError) as exc:
        # A report the contract refuses is retained as blocked so the queue continues.
        state, fields, _ = _blocked(task, f"Blocked: report refused by the evidence contract: {exc}", str(exc))
        fields.update(changed_files=changed, generated_artifacts=deepcopy(ctx.artifacts),
                      provider_runtime_identity=_with_probes(builtin_identity(changed), ctx, state, requires),
                      tests_passed=[], tests_skipped=[])
        return validate_report(build_report(task, state, {k: v for k, v in fields.items() if k in FIELD_NAMES}, []))


def _section_import_error(section, errors) -> str | None:
    """Import errors that explain a section's unimplemented tasks.

    A packaged section owns exactly its :data:`SECTION_MODULES`; an extension
    section is implemented by the configured modules, whose errors apply to
    every extension task left without an implementation.
    """
    if section.get("extension"):
        names = [name for name in errors if name not in SECTION_MODULES]
    else:
        names = base_section_modules(section["key"])
    return "; ".join(f"{name}: {errors[name]}" for name in sorted(names) if name in errors) or None


def run_queue(output_dir, task_ids=None, providers=None, junit_path=None, budget_seconds=None) -> dict:
    """Run selected tasks (default: all) in queue order and retain their reports.

    Elapsed times are written to ``run-log.json`` beside the reports, never into
    them: timing is not reproducible and is not a finding.
    """
    output_dir = Path(output_dir)
    queue = load_queue()
    implementations, errors = load_implementations()
    selected = set(task_ids) if task_ids else None
    unknown = (selected or set()) - {t["id"] for t in queue["tasks"]}
    if unknown:
        raise ValueError(f"Unknown lab task identities: {sorted(unknown)}")
    junit = read_junit(Path(junit_path)) if junit_path else {}
    ctx = Context(output_dir, providers)
    reports, timings = [], []
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    import_errors = {s["section"]: _section_import_error(s, errors) for s in queue["sections"]}
    for item in queue["tasks"]:
        if selected is not None and item["id"] not in selected:
            continue
        artifact_dir = output_dir / "artifacts" / item["id"]
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        error = import_errors[item["section"]]
        started = time.perf_counter()
        report = run_task(item, implementations.get(item["id"]), ctx, junit, error)
        timings.append({"task_id": item["id"], "state": report["state"],
                        "seconds": round(time.perf_counter() - started, 3)})
        (output_dir / "reports" / f"{item['id']}.json").write_text(dumps(report), encoding="utf-8")
        reports.append(report)
    if selected is None:
        write_index(output_dir, queue)
    total = round(sum(t["seconds"] for t in timings), 3)
    over = [t for t in timings if budget_seconds is not None and t["seconds"] > budget_seconds]
    (output_dir / "run-log.json").write_text(dumps({
        "schema": "ciw.lab-run-log.v1", "note": "Elapsed wall-clock seconds; not reproducible and not findings",
        "total_seconds": total, "budget_seconds": budget_seconds, "tasks": timings}), encoding="utf-8")
    summary = {"output_dir": str(output_dir), "tasks": len(reports),
               "states": {s: sum(r["state"] == s for r in reports) for s in ("completed", "partial", "deferred", "blocked")},
               "labels": _label_totals(reports), "total_seconds": total,
               "slowest": sorted(timings, key=lambda t: -t["seconds"])[:5]}
    if budget_seconds is not None:
        summary["over_budget"] = over
    return summary


def _label_totals(reports):
    totals = {}
    for report in reports:
        for label, count in report["evidence_status"]["counts"].items():
            totals[label] = totals.get(label, 0) + count
    return totals


def load_reports(directory) -> list:
    directory = Path(directory) / "reports"
    reports = []
    for path in sorted(directory.glob("T*.json")):
        reports.append(validate_report(json.loads(path.read_text(encoding="utf-8"))))
    return reports


def write_index(output_dir, queue=None) -> None:
    """Write the queue state and the full human-readable report book."""
    output_dir = Path(output_dir)
    queue = queue or load_queue()
    reports = {r["task_id"]: r for r in load_reports(output_dir)}
    state = [{"id": t["id"], "section": t["section_key"], "title": t["title"],
              "state": reports[t["id"]]["state"] if t["id"] in reports else "not_run",
              "evidence_status": reports[t["id"]]["evidence_status"]["primary"] if t["id"] in reports else "not_established",
              "report_id": reports[t["id"]]["report_id"] if t["id"] in reports else None}
             for t in queue["tasks"]]
    (output_dir / "queue-state.json").write_text(dumps({"schema": "ciw.lab-queue-state.v1", "tasks": state}),
                                                 encoding="utf-8")
    lines = ["# Computational experimentalist queue — retained reports", "",
             "Generated by `ciw lab run --all`. Every finding carries one evidence label assigned by",
             "`ciw.lab.evidence`; no report states physical validation without acquired hardware evidence.", "",
             "| Task | Title | State | Primary evidence |", "| --- | --- | --- | --- |"]
    for row in state:
        lines.append(f"| [{row['id']}](#{row['id'].lower()}) | {row['title']} | `{row['state']}` | `{row['evidence_status']}` |")
    for section in queue["sections"]:
        lines += ["", f"## {section['section']}. {section['name']}", ""]
        for t in queue["tasks"]:
            if t["section"] == section["section"] and t["id"] in reports:
                lines += [f'<a id="{t["id"].lower()}"></a>', "", render_markdown(reports[t["id"]])]
    (output_dir / "REPORTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _close(retained, fresh, tolerance):
    if isinstance(retained, bool) or isinstance(fresh, bool):
        return type(retained) is type(fresh) and retained == fresh  # True is not the number 1
    if retained is None or fresh is None or isinstance(retained, str):
        return retained == fresh
    if isinstance(retained, (int, float)) and isinstance(fresh, (int, float)):
        if not (math.isfinite(retained) and math.isfinite(fresh)):
            return retained == fresh
        return abs(retained - fresh) <= tolerance.get("abs", 0.0) + tolerance.get("rel", 0.0) * abs(retained)
    if isinstance(retained, list) and isinstance(fresh, list):
        return len(retained) == len(fresh) and all(_close(a, b, tolerance) for a, b in zip(retained, fresh))
    if isinstance(retained, dict) and isinstance(fresh, dict):
        return retained.keys() == fresh.keys() and all(_close(retained[k], fresh[k], tolerance) for k in retained)
    return retained == fresh


def _optional_difference(old, new) -> str:
    """Optional reference modules present in one run's identity but not the other's."""
    def present(report):
        identity = report.get("provider_runtime_identity") or {}
        return {name for name in OPTIONAL_MODULES if name in identity}
    before, after = present(old), present(new)
    return ", ".join([f"-{m}" for m in sorted(before - after)] + [f"+{m}" for m in sorted(after - before)])


def _probes(report) -> dict:
    identity = report.get("provider_runtime_identity")
    probes = identity.get("requirement_probes") if isinstance(identity, dict) else None
    return probes if isinstance(probes, dict) else {}


def _probe_difference(old, new) -> str:
    """Requirement probes (module:, tool:, provider:, hardware:) whose recorded outcome differs between runs."""
    def outcome(value):
        return {True: "available", False: "unavailable", None: "not recorded"}.get(value, repr(value))
    before, after = _probes(old), _probes(new)
    return ", ".join(f"{name} {outcome(before.get(name))} -> {outcome(after.get(name))}"
                     for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name))


def _environment_note(old, new) -> str:
    """Environment differences between two runs that can explain a changed state or label."""
    notes = [f"{kind} differ: {text}" for kind, text in (("optional modules", _optional_difference(old, new)),
                                                        ("requirement probes", _probe_difference(old, new))) if text]
    return f" ({'; '.join(notes)})" if notes else ""


PROSE_FIELDS = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                "experiment", "numerical_result", "uncertainty", "failure_modes_checked",
                "unresolved_assumptions", "recommended_next_task", "physical_validation_status")
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _skeleton(value) -> str:
    """Report prose with numbers masked: wording must match, last digits may differ by platform."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return NUMBER.sub("#", text)


def _witness(counterexample):
    """A counterexample's witness values, compared within the finding's tolerance; its wording by skeleton."""
    return counterexample.get("witness") if isinstance(counterexample, dict) else None


def artifact_problems(directory, tasks=None) -> list:
    """Retained artifacts must still hash to the digests their reports record (optionally for ``tasks`` only)."""
    problems = []
    for report in load_reports(directory):
        if tasks is not None and report["task_id"] not in tasks:
            continue
        for artifact in report["generated_artifacts"]:
            path = Path(directory) / artifact["path"]
            if not path.is_file():
                problems.append(f"{report['task_id']}: artifact {artifact['path']} missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                problems.append(f"{report['task_id']}: artifact {artifact['path']} differs from its recorded digest")
    return problems


def compare(retained_dir, fresh_dir, tasks=None) -> dict:
    """Regression gate: same states, labels, claims, units and wording; values within tolerance.

    Every task retained or regenerated is compared, or only ``tasks`` (the
    identities of a partial run), each of which must then be in both
    directories. A missing or empty retained directory, or an empty
    ``tasks``, fails: comparing nothing verifies nothing. A changed state or
    label names the optional modules and requirement probes whose recorded
    outcomes differ between the two runs.
    """
    selected = None if tasks is None else set(tasks)
    retained_reports = load_reports(retained_dir)
    retained = {r["task_id"]: r for r in retained_reports if selected is None or r["task_id"] in selected}
    fresh = {r["task_id"]: r for r in load_reports(fresh_dir) if selected is None or r["task_id"] in selected}
    problems = []
    if not retained_reports:
        problems.append(f"no retained reports in {Path(retained_dir) / 'reports'}: nothing to verify against")
    if selected is not None and not selected:
        problems.append("no tasks selected: nothing to verify")
    problems += artifact_problems(retained_dir, selected) + artifact_problems(fresh_dir, selected)
    expected = set(retained) | set(fresh) if selected is None else selected
    # New evidence must be reviewed and retained, not slip in through a gate run.
    problems += [f"{task_id}: not retained" for task_id in sorted(expected - set(retained))]
    problems += [f"{task_id}: not regenerated" for task_id in sorted(expected - set(fresh))]
    for task_id, old in retained.items():
        new = fresh.get(task_id)
        if new is None:
            continue
        if old["state"] != new["state"]:
            problems.append(f"{task_id}: state {old['state']} -> {new['state']}" + _environment_note(old, new))
        for name in PROSE_FIELDS:
            if _skeleton(old[name]) != _skeleton(new[name]):
                problems.append(f"{task_id}: '{name}' wording differs")
        old_claims = [f["claim"] for f in old["findings"]]
        new_claims = [f["claim"] for f in new["findings"]]
        if old_claims != new_claims:
            changed = sorted(set(old_claims) ^ set(new_claims)) or ["order"]
            problems.append(f"{task_id}: finding claims differ: {changed[:4]}")
        if len(old["findings"]) != len(new["findings"]):
            problems.append(f"{task_id}: finding count {len(old['findings'])} -> {len(new['findings'])}")
        new_findings = {f["claim"]: f for f in new["findings"]}
        for record in old["findings"]:
            claim, other = record["claim"], new_findings.get(record["claim"])
            if other is None:
                continue
            validate_finding(other)
            if record["evidence_status"] != other["evidence_status"]:
                problems.append(f"{task_id}: '{claim}' label {record['evidence_status']} -> {other['evidence_status']}"
                                + _environment_note(old, new))
            if record.get("unit") != other.get("unit") or record["domain"] != other["domain"]:
                problems.append(f"{task_id}: '{claim}' unit or domain differs")
            if record.get("expected_not_established") != other.get("expected_not_established"):
                problems.append(f"{task_id}: '{claim}' expected_not_established differs")
            tolerance = record.get("regression_tolerance", {"abs": 0.0, "rel": 1e-9})
            if not _close(record["value"], other["value"], tolerance):
                problems.append(f"{task_id}: '{claim}' value outside regression tolerance")
            old_counter, new_counter = record.get("counterexample"), other.get("counterexample")
            if _skeleton(old_counter) != _skeleton(new_counter) or not _close(
                    _witness(old_counter), _witness(new_counter), tolerance):
                problems.append(f"{task_id}: '{claim}' counterexample differs")
    return {"compared": len(retained), "problems": problems, "passed": not problems}


REPORT_QUESTIONS = [label for _, label in FIELDS]


def schema_errors(report) -> list:
    """Structural errors against task-report.schema.json (requires jsonschema)."""
    from importlib import resources
    import jsonschema

    schema = json.loads(resources.files("ciw.lab").joinpath("task-report.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                  for error in validator.iter_errors(report))
