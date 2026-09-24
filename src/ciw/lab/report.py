"""Task reports for the computational-experimentalist queue.

A report answers the same nineteen questions for every task, including tasks
that could not be executed here. Reports never state physical validation
unless every physical-domain finding rests on acquired hardware evidence, and
they say how many do; their evidence status is recomputed from the retained
findings. Every rendered finding shows its declared basis beside its label:
the basis components (the finding's ``origin`` key) with the generator,
provider or device each names, so a check on synthetic inputs and a check on a
provider's output read differently although both are ``numerically_verified``.
The basis components are not the implementation origin that
``independently_verified`` compares.
"""
from __future__ import annotations

from copy import deepcopy

from ..core.identities import canonical_json, content_identity
from .evidence import (AUTHORITY_DOMAINS, LABELS, ORIGINS, PHYSICAL_DOMAINS, EvidenceRefusal, describe_basis,
                       describe_origin, finding_origin, physical_status, primary_label, summarize, validate_finding)

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

# Keys of a generated-artifact entry declaring an SVG figure whose bytes are not
# reproducible byte for byte; present only as ``true``, at most one per figure.
# ``wall_clock_timing`` (``ctx.artifact_text(..., wall_clock_timing=True)``): the
# figure plots wall-clock timings; figure re-executions compare it for presence
# and structure. ``rounding_level`` (``rounding_level=True``): it plots values
# at binary64 rounding level (errors and residuals near machine epsilon), whose
# last bits, and so the figure's coordinates and axis range, follow the BLAS
# kernel and platform; it records its plotted values with their rounding bounds
# (``svg.line_plot(..., rounding=...)``), and re-executions compare those
# values within the bounds. Neither is compared byte for byte (T158,
# scripts/check_figures.py).
WALL_CLOCK_TIMING = "wall_clock_timing"
ROUNDING_LEVEL = "rounding_level"
FIGURE_DECLARATIONS = (WALL_CLOCK_TIMING, ROUNDING_LEVEL)


def _figure_declaration_problem(artifacts) -> str | None:
    """Why the figure declarations of a generated-artifact list are malformed, or None."""
    for artifact in artifacts if isinstance(artifacts, list) else []:
        if not isinstance(artifact, dict):
            continue
        for key in FIGURE_DECLARATIONS:
            if key in artifact and (artifact[key] is not True or not str(artifact.get("path")).endswith(".svg")):
                return f"Artifact {artifact.get('path')!r}: {key} is declared only as true, on an SVG figure"
        if all(key in artifact for key in FIGURE_DECLARATIONS):
            return (f"Artifact {artifact.get('path')!r}: a figure is declared as {WALL_CLOCK_TIMING} or "
                    f"{ROUNDING_LEVEL}, not both")
    return None


def evidence_status(findings) -> dict:
    """Label counts plus the order-independent primary label, recomputed from the findings."""
    return {"primary": primary_label(findings), "counts": summarize(findings)}


NO_HARDWARE_STATEMENT = "Physical validation requires acquired hardware evidence; none was acquired for this task."


def physical_validation(findings) -> dict:
    """Derived physical validation status with its derived statement.

    The status is ``hardware_measured`` only when every physical-domain finding
    is; the statement counts the physical-domain findings that rest on acquired
    hardware evidence, so a partly measured task neither claims validation nor
    denies the evidence it acquired.
    """
    physical = [f for f in findings if f["domain"] in PHYSICAL_DOMAINS]
    status = physical_status(physical)
    measured = sum(physical_status([record]) == "hardware_measured" for record in physical)
    if not measured:
        return {"status": status, "statement": NO_HARDWARE_STATEMENT}
    statement = (f"{measured} of {len(physical)} physical-domain findings {'rests' if measured == 1 else 'rest'} "
                 "on acquired hardware evidence")
    rest = "." if measured == len(physical) else "; the rest are not_established."
    return {"status": status, "statement": statement + rest}


def validate_report(report: dict) -> dict:
    """Refuse missing questions, relabelled findings or unsupported physical status."""
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise EvidenceRefusal(f"Expected {REPORT_SCHEMA}")
    missing = [name for name in FIELD_NAMES if name not in report]
    if missing:
        raise EvidenceRefusal(f"Report is missing required fields: {missing}")
    if report.get("state") not in STATES:
        raise EvidenceRefusal(f"Report state must be one of {STATES}")
    declaration = _figure_declaration_problem(report["generated_artifacts"])
    if declaration:
        raise EvidenceRefusal(declaration)
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise EvidenceRefusal("Report findings must be a list")
    for record in findings:
        validate_finding(record)
    if report["evidence_status"] != evidence_status(findings):
        raise EvidenceRefusal("Report evidence status differs from its findings")
    expected_physical = physical_validation(findings)
    if report["physical_validation_status"] != expected_physical:
        raise EvidenceRefusal(f"Physical validation status must be {expected_physical['status']} with its derived "
                              f"statement: {expected_physical['statement']}")
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
        if "origin" not in record:
            # Only reports retained before origins were recorded lack them; new ones never do.
            raise EvidenceRefusal(f"Finding lacks its derived origin; build it with ciw.lab.evidence.finding: "
                                  f"{record.get('claim')!r}")
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


# The finding table of a rendered report. The label keeps its own column and the
# declared basis sits in the next one (``finding_row``). Other renderers and
# audits use these constants and ``finding_row`` rather than restating them.
FINDINGS_HEADER = "| Finding | Value | Evidence status | Basis |"
FINDINGS_RULE = "| --- | --- | --- | --- |"


def origin_summary(findings) -> str:
    """Findings per declared basis component, in :data:`ORIGINS` order: 'reference checks: 5, synthetic inputs: 2'."""
    counts = {}
    for record in findings:
        for item in finding_origin(record) or ["none"]:
            counts[item] = counts.get(item, 0) + 1
    words = [(describe_origin([item]) if item != "none" else describe_origin([]), counts[item])
             for item in (*ORIGINS, "none") if counts.get(item)]
    return ", ".join(f"{word}: {count}" for word, count in words) or "none"


def finding_row(record: dict) -> str:
    """One Markdown row: claim, value with unit, label in backticks, then the declared basis in words."""
    # Claims, units and declared identities are escaped like every other cell: a raw pipe would shift a column.
    unit = f" {_inline(record['unit'])}" if record.get("unit") else ""
    return (f"| {_inline(record['claim'])} | {_inline(record['value'], limit=80)}{unit}"
            f" | `{record['evidence_status']}` | {_inline(describe_basis(record))} |")


def render_markdown(report: dict) -> str:
    """Render the nineteen questions in their required order, then the findings with label and declared basis."""
    lines = [f"### {report['task_id']} — {report['title']}", "", f"State: `{report['state']}`", ""]
    for name, label in FIELDS:
        value = report[name]
        if name == "evidence_status":
            counts = ", ".join(f"{k}: {v}" for k, v in value["counts"].items() if v)
            value = (f"`{value['primary']}` (findings — {counts or 'none'}; declared basis — "
                     f"{origin_summary(report['findings'])})")
        elif name == "physical_validation_status":
            value = f"`{value['status']}` — {value['statement']}"
        lines.append(f"- **{label}:** {_inline(value)}")
    if report["findings"]:
        lines += ["", FINDINGS_HEADER, FINDINGS_RULE]
        lines += [finding_row(record) for record in report["findings"]]
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
