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
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from .. import __version__
from .evidence import validate_finding
from .registry import load_implementations, load_queue
from .report import FIELDS, FIELD_NAMES, build_report, render_markdown, validate_report

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Retained evidence is committed; one artifact larger than this is refused so
# tables stay summaries (sample, aggregate or truncate long trajectories).
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


def dumps(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def source_digest(relative: str) -> str | None:
    """SHA-256 of a package source file with normalized line endings."""
    if not relative.startswith("src/ciw/"):
        return None
    path = PACKAGE_ROOT / relative[len("src/ciw/"):]
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def builtin_identity(changed_files) -> dict:
    sources = {name: source_digest(name) for name in changed_files if source_digest(name)}
    identity = {"implementation": "ciw.lab", "ciw_version": __version__,
                "python": platform.python_version(), "numpy": np.__version__, "sources": sources}
    for optional in ("scipy", "sympy", "mpmath"):
        if importlib.util.find_spec(optional) is not None:
            identity[optional] = __import__(optional).__version__
    return identity


def git_identity(path: Path) -> dict:
    """HEAD, tree and cleanliness of a provider checkout; refuses non-repositories."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    return {"path": str(path), "revision": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


class Context:
    """Per-run services: artifact retention, shared memo and requirement probes."""

    def __init__(self, output_dir: Path, providers: dict | None = None):
        self.output_dir = Path(output_dir)
        self.providers = {role: Path(path) for role, path in (providers or {}).items()}
        self._memo: dict = {}
        self.task_id: str | None = None
        self.artifacts: list = []

    def begin(self, task_id: str) -> None:
        self.task_id, self.artifacts = task_id, []

    def memo(self, key, compute):
        """Share one deterministic computation between tasks in a run."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def available(self, requirement: str) -> bool:
        kind, _, name = requirement.partition(":")
        if kind == "module":
            return importlib.util.find_spec(name) is not None
        if kind == "provider":
            return name in self.providers and self.providers[name].exists()
        if kind == "tool":
            return shutil.which(name) is not None
        if kind == "hardware":
            return _probe_hardware(name)
        raise ValueError(f"Unsupported lab requirement: {requirement}")

    def _write(self, name: str, data: bytes) -> str:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Artifact names are single file names")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"Artifact {name} exceeds {MAX_ARTIFACT_BYTES} bytes; retain a summary instead")
        directory = self.output_dir / "artifacts" / self.task_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
        relative = f"artifacts/{self.task_id}/{name}"
        self.artifacts.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return relative

    def artifact_json(self, name: str, value) -> str:
        return self._write(name, dumps(value).encode("utf-8"))

    def artifact_text(self, name: str, text: str) -> str:
        return self._write(name, text.encode("utf-8"))


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
        return any(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"))
    return False


def read_junit(path: Path | None) -> dict:
    """Map pytest node ids to passed, skipped or failed."""
    if path is None:
        return {}
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        module = classname.replace(".", "/") + ".py"
        node = f"{module}::{name}"
        if case.find("skipped") is not None:
            outcomes[node] = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcomes[node] = "failed"
        else:
            outcomes[node] = "passed"
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
            if key == node or key.endswith(node) or (key.split("[")[0] == node):
                outcome = value if outcome in (None, "passed") else outcome
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


def run_task(task, implementation, ctx: Context, junit: dict, import_error: str | None = None) -> dict:
    ctx.begin(task["id"])
    findings, state = [], "deferred"
    if implementation is None:
        reason = f"Deferred: {import_error or 'no implementation is registered for this task'}."
        fields = _default_fields(task, reason)
    else:
        missing = [need for need in implementation.requires if not ctx.available(need)]
        if missing:
            reason = "Blocked: unavailable requirement(s) " + ", ".join(missing) + "."
            fields = _default_fields(task, reason)
            fields["unresolved_assumptions"] = [reason]
            state = "blocked"
            plan = getattr(implementation.run, "plan", None)
            if plan:
                fields.update({k: v for k, v in plan.items() if k in FIELD_NAMES})
                fields["experiment"] = reason + " Planned: " + str(plan.get("experiment", ""))
        else:
            try:
                outcome = implementation.run(ctx)
                state = outcome.get("state", "completed")
                fields = _default_fields(task, "")
                fields.update(outcome.get("fields", {}))
                findings = outcome.get("findings", [])
            except Exception as exc:  # retained as a blocked report, never hidden
                reason = f"Blocked by unexpected {type(exc).__name__}: {exc}"
                fields = _default_fields(task, reason)
                fields["failure_modes_checked"] = [traceback.format_exception_only(type(exc), exc)[-1].strip()]
                state, findings = "blocked", []
    changed = list(implementation.changed_files) if implementation else []
    fields.setdefault("changed_files", changed)
    if not fields.get("changed_files"):
        fields["changed_files"] = changed
    fields["generated_artifacts"] = deepcopy(ctx.artifacts)
    if not fields.get("provider_runtime_identity"):
        fields["provider_runtime_identity"] = builtin_identity(changed)
    passed, skipped, failed = _test_fields(implementation, findings, junit)
    fields["tests_passed"], fields["tests_skipped"] = passed, skipped
    if failed and state == "completed":
        state = "partial"
    report = build_report(task, state, {k: v for k, v in fields.items() if k in FIELD_NAMES
                                        and k not in ("evidence_status", "physical_validation_status")},
                          findings, extra={"tests_failed": failed} if failed else None)
    return validate_report(report)


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
    section_modules = {s["section"]: s["key"] for s in queue["sections"]}
    for item in queue["tasks"]:
        if selected is not None and item["id"] not in selected:
            continue
        artifact_dir = output_dir / "artifacts" / item["id"]
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        prefix = section_modules[item["section"]].replace("-", "_")
        error = "; ".join(f"{name}: {message}" for name, message in sorted(errors.items())
                          if name.startswith(prefix)) or None
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
    if isinstance(retained, bool) or isinstance(fresh, bool) or retained is None or fresh is None or isinstance(retained, str):
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


def compare(retained_dir, fresh_dir) -> dict:
    """Regression gate: same states, same labels, values within declared tolerance."""
    retained = {r["task_id"]: r for r in load_reports(retained_dir)}
    fresh = {r["task_id"]: r for r in load_reports(fresh_dir)}
    problems = []
    if retained:
        # New evidence must be reviewed and retained, not slip in through a gate run.
        problems += [f"{task_id}: not retained" for task_id in sorted(set(fresh) - set(retained))]
    for task_id, old in retained.items():
        new = fresh.get(task_id)
        if new is None:
            problems.append(f"{task_id}: not regenerated")
            continue
        if old["state"] != new["state"]:
            problems.append(f"{task_id}: state {old['state']} -> {new['state']}")
        old_findings = {f["claim"]: f for f in old["findings"]}
        new_findings = {f["claim"]: f for f in new["findings"]}
        if old_findings.keys() != new_findings.keys():
            changed = sorted(set(old_findings) ^ set(new_findings))
            problems.append(f"{task_id}: finding claims differ: {changed[:4]}")
        if len(old["findings"]) != len(new["findings"]):
            problems.append(f"{task_id}: finding count {len(old['findings'])} -> {len(new['findings'])}")
        for claim, record in old_findings.items():
            other = new_findings.get(claim)
            if other is None:
                continue
            validate_finding(other)
            if record["evidence_status"] != other["evidence_status"]:
                problems.append(f"{task_id}: '{claim}' label {record['evidence_status']} -> {other['evidence_status']}")
            tolerance = record.get("regression_tolerance", {"abs": 0.0, "rel": 1e-9})
            if not _close(record["value"], other["value"], tolerance):
                problems.append(f"{task_id}: '{claim}' value outside regression tolerance")
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
