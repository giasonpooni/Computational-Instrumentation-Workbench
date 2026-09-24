"""Task reports for the computational-experimentalist queue.

A report answers the same nineteen questions for every task, including tasks
that could not be executed here. Reports never state physical validation
unless a physical-domain finding cites acquired hardware evidence, and their
evidence status is recomputed from the retained findings.
"""
from __future__ import annotations

from copy import deepcopy

from ..core.identities import canonical_json, content_identity
from .evidence import (AUTHORITY_DOMAINS, LABELS, PHYSICAL_DOMAINS, EvidenceRefusal, physical_status,
                       primary_label, summarize, validate_finding)

REPORT_SCHEMA = "ciw.lab-task-report.v1"

# The required report questions, in order, with their display labels.
FIELDS = (
    ("task_id", "Task ID"),
    ("hypothesis", "Hypothesis"),
    ("mathematical_model", "Mathematical model"),
    ("input_data", "Input data"),
    ("observation_model", "Observation model"),
    ("expected_invariant", "Expected invariant"),
    ("experiment", "Experiment or test"),
    ("changed_files", "Changed files"),
    ("generated_artifacts", "Generated artifacts"),
    ("numerical_result", "Numerical result"),
    ("uncertainty", "Uncertainty"),
    ("evidence_status", "Evidence status"),
    ("provider_runtime_identity", "Provider/runtime identity"),
    ("failure_modes_checked", "Failure modes checked"),
    ("tests_passed", "Tests passed"),
    ("tests_skipped", "Tests skipped"),
    ("physical_validation_status", "Physical validation status"),
    ("unresolved_assumptions", "Unresolved assumptions"),
    ("recommended_next_task", "Recommended next task"),
)
FIELD_NAMES = tuple(name for name, _ in FIELDS)

# Task states. ``completed`` means the planned computation ran and its checks
# passed; it never means physically validated.
STATES = ("completed", "partial", "deferred", "blocked")


def evidence_status(findings) -> dict:
    """Label counts plus the order-independent primary label, recomputed from the findings."""
    return {"primary": primary_label(findings), "counts": summarize(findings)}


PHYSICAL_STATEMENTS = {
    "not_established": "Physical validation requires acquired hardware evidence; none was acquired for this task.",
    "hardware_measured": "Physical-domain findings cite acquired hardware evidence.",
}


def physical_validation(findings) -> dict:
    """Derived physical validation status with its fixed statement."""
    status = physical_status([f for f in findings if f["domain"] in PHYSICAL_DOMAINS])
    return {"status": status, "statement": PHYSICAL_STATEMENTS[status]}


def validate_report(report: dict) -> dict:
    """Refuse missing questions, relabelled findings or unsupported physical status."""
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise EvidenceRefusal(f"Expected {REPORT_SCHEMA}")
    missing = [name for name in FIELD_NAMES if name not in report]
    if missing:
        raise EvidenceRefusal(f"Report is missing required fields: {missing}")
    if report.get("state") not in STATES:
        raise EvidenceRefusal(f"Report state must be one of {STATES}")
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise EvidenceRefusal("Report findings must be a list")
    for record in findings:
        validate_finding(record)
    if report["evidence_status"] != evidence_status(findings):
        raise EvidenceRefusal("Report evidence status differs from its findings")
    expected_physical = physical_validation(findings)
    if report["physical_validation_status"] != expected_physical:
        raise EvidenceRefusal(f"Physical validation status must be {expected_physical['status']} with its derived statement")
    claims = [f["claim"] for f in findings]
    if len(claims) != len(set(claims)):
        raise EvidenceRefusal("Finding claims must be unique within a report")
    if report["state"] in ("blocked", "deferred") and any(f["evidence_status"] != "not_established" for f in findings):
        raise EvidenceRefusal("A blocked or deferred task cannot carry established findings")
    if report["state"] == "completed" and not findings:
        raise EvidenceRefusal("A completed task must retain at least one finding")
    if report["state"] == "completed" and report.get("tests_failed"):
        raise EvidenceRefusal("A completed task cannot record failed tests; a failure makes it partial")
    if report["state"] == "completed" and any(f["evidence_status"] == "not_established"
                                              and f["domain"] not in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                                              and not f.get("expected_not_established") for f in findings):
        raise EvidenceRefusal("A completed task cannot contain an unexpectedly unestablished computational finding")
    for label in report["evidence_status"]["counts"]:
        if label not in LABELS:
            raise EvidenceRefusal(f"Unknown evidence status: {label}")
    if report.get("report_id") != report_identity(report):
        raise EvidenceRefusal("Report identity does not match its content")
    return report


def report_identity(report: dict) -> str:
    """Content identity over the scientific report, excluding the identity itself."""
    return content_identity({key: value for key, value in report.items() if key != "report_id"})


def build_report(task: dict, state: str, fields: dict, findings: list, extra: dict | None = None) -> dict:
    """Assemble a report; evidence and physical status are always derived."""
    unknown = set(fields) - set(FIELD_NAMES)
    if unknown:
        raise EvidenceRefusal(f"Unknown report fields: {sorted(unknown)}")
    if "evidence_status" in fields or "physical_validation_status" in fields:
        raise EvidenceRefusal("Evidence and physical validation status are derived, not supplied")
    for record in findings:
        validate_finding(record)
    report = {"schema": REPORT_SCHEMA, "task_id": task["id"], "number": task["number"],
              "section": task["section_key"], "title": task["title"], "state": state}
    for name in FIELD_NAMES:
        if name == "task_id":
            continue
        report[name] = deepcopy(fields.get(name, []))
    report["evidence_status"] = evidence_status(findings)
    report["physical_validation_status"] = physical_validation(findings)
    report["findings"] = deepcopy(findings)
    report.update(deepcopy(extra or {}))
    canonical_json(report)
    report["report_id"] = report_identity(report)
    return report


def render_markdown(report: dict) -> str:
    """Render the nineteen questions in their required order."""
    lines = [f"### {report['task_id']} — {report['title']}", "", f"State: `{report['state']}`", ""]
    for name, label in FIELDS:
        value = report[name]
        if name == "evidence_status":
            counts = ", ".join(f"{k}: {v}" for k, v in value["counts"].items() if v)
            value = f"`{value['primary']}` (findings — {counts or 'none'})"
        elif name == "physical_validation_status":
            value = f"`{value['status']}` — {value['statement']}"
        lines.append(f"- **{label}:** {_inline(value)}")
    if report["findings"]:
        lines += ["", "| Finding | Value | Evidence status |", "| --- | --- | --- |"]
        for record in report["findings"]:
            lines.append(f"| {record['claim']} | {_inline(record['value'], limit=80)}"
                         f"{' ' + record['unit'] if record.get('unit') else ''} | `{record['evidence_status']}` |")
    return "\n".join(lines) + "\n"


def _inline(value, limit=600):
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        text = "; ".join(value) if value else "none"
    elif isinstance(value, float):
        text = format(value, ".6g")
    else:
        text = canonical_json(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 1] + "…"
