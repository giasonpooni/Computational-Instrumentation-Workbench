"""Research and portfolio tasks T155-T168: aggregate retained lab evidence.

These tasks run after every other queue task in the same run and read the
reports retained in the output directory for the tasks that precede them in
queue order (this section's earlier tasks included; a report of a later task
is never read). T158 also re-executes a declared set of figure tasks in a
scratch directory. They generate specifications, catalogues, drafts and
ledgers from those records; their tables never state a number that is not in
the retained report they cite, and they never upgrade a label.

Every aggregate is backed by a check through a second path that can fail for
a wrong aggregate: a raw-text recount of the retained report files, a parse
of the written table against the source findings, or a reference oracle
written from the specification. Re-reading an artifact just written is never
used as a check. Drafted papers are drafts: review, submission and acceptance
are not claims the workbench can make.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

from .. import __version__
from .evidence import (AUTHORITY_DOMAINS, COMPARISONS, COMPUTATIONAL_DOMAINS, COMPUTATIONAL_ORDER, DOMAINS,
                       INDEPENDENT_ORIGINS, LABELS, MAX_THRESHOLD, PHYSICAL_DOMAINS, EvidenceRefusal, finding,
                       supported_label, holds as compare)
from .registry import load_implementations, load_queue, task
from .report import validate_report

MODULE = "src/ciw/lab/research_portfolio.py"
TESTS = "tests/test_lab_research_portfolio.py"
SPECIFICATIONS = "docs/lab/SPECIFICATIONS.md"
FIRST_OWN = 155
ZERO = {"abs": 0, "rel": 0}


def _exact(basis: str) -> dict:
    return {"kind": "exact", "value": 0, "basis": basis}


COUNT = _exact("integer count over retained report files; no rounding, sampling or tolerance")


# ------------------------------------------------------------ reading reports
def _reports_before(ctx, number: int) -> list:
    """Validated reports retained on disk for the tasks that precede task ``number`` in queue order."""
    reports = []
    for path in sorted((ctx.output_dir / "reports").glob("T*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(report.get("number"), int) and report["number"] < number:
            reports.append(validate_report(report))
    return reports


def _raw_reports(ctx, number: int) -> list:
    """(task id, file text) of the same report files, for second-path recounts that never parse JSON."""
    texts = []
    for path in sorted((ctx.output_dir / "reports").glob("T*.json")):
        if path.stem[1:].isdigit() and int(path.stem[1:]) < number:
            texts.append((path.stem, path.read_text(encoding="utf-8")))
    return texts


# Raw-text patterns over the retained layout (``runner.dumps``: one-space indent, sorted keys):
# report keys sit at indent 1, finding keys at indent 3, and "domain" is followed by "evidence_status".
RAW_STATE = re.compile(r'^ "state": "([a-z_]+)"', re.M)
RAW_LABEL = re.compile(r'^   "evidence_status": "([a-z_]+)"', re.M)
RAW_DOMAIN_LABEL = re.compile(r'^   "domain": "([a-z_]+)",\n   "evidence_status": "([a-z_]+)"', re.M)
RAW_COUNTEREXAMPLE = re.compile(r'^   "counterexample": \{', re.M)
RAW_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _raw_findings(text: str) -> list:
    """Finding blocks of one report file, located by indentation alone."""
    start = re.search(r'^ "findings": \[$', text, re.M)
    if not start:
        return []
    end = re.compile(r'^ \],?$', re.M).search(text, start.end())
    body = text[start.end(): end.start() if end else len(text)]
    return re.split(r'^  \},?$', body, flags=re.M)[:-1]


def _raw_numeric(block: str) -> bool:
    """Whether a raw finding block's value (its last key) holds a number outside string literals."""
    match = re.search(r'^   "value": ', block, re.M)
    return bool(match) and re.search(r"\d", RAW_STRING.sub('""', block[match.end():])) is not None


def _raw_assumption_count(text: str) -> int:
    """Items of a report file's unresolved_assumptions answer, counted from the text."""
    match = re.search(r'^ "unresolved_assumptions": (.*)$', text, re.M)
    if not match or match.group(1).startswith("[]"):
        return 0
    if match.group(1) != "[":
        return 1
    end = re.compile(r'^ \],?$', re.M).search(text, match.end())
    return len(re.findall(r'^  [^\s\]}]', text[match.end(): end.start() if end else len(text)], re.M))


def _numbers(value) -> list:
    """Every number in a JSON value (booleans are not numbers; object keys are text)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [value]
    items = value.values() if isinstance(value, dict) else value
    return [number for item in items for number in _numbers(item)]


# ------------------------------------------------------------ prose and tables
def _flat(text) -> str:
    return str(text).replace("\r", " ").replace("\n", " ")


def _words(text, limit: int) -> str:
    """Text cut at a word boundary (never inside a number), marked with an ellipsis."""
    text = _flat(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def _scalar(value) -> str:
    if isinstance(value, str):
        return json.dumps(_words(value, 60), ensure_ascii=False)
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return ("[…]" if value else "[]") if isinstance(value, list) else ("{…}" if value else "{}")


def _leaves(value, path=""):
    """(path, leaf) pairs of a nested value: scalar entries of an object before its containers, keys sorted."""
    if isinstance(value, dict) and value:
        for key in sorted(value, key=lambda k: (isinstance(value[k], (list, dict)), k)):
            yield from _leaves(value[key], f"{path}.{key}" if path else str(key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def _headline(value, entries: int = 4) -> str:
    """A value for prose tables that never cuts inside a number.

    Scalars appear whole in shortest round-trip form. A flat list shows its
    first entries; any other list or object shows its first leaves with their
    paths (``"euler.order": 1.02``). ``…(+N)`` marks the N entries or leaves
    left out. Every number shown is a number of the value.
    """
    if isinstance(value, list) and not any(isinstance(item, (list, dict)) for item in value):
        shown = [_scalar(item) for item in value[:entries]]
        more = f" …(+{len(value) - entries})" if len(value) > entries else ""
        return "[" + ", ".join(shown) + more + "]"
    if isinstance(value, (list, dict)) and value:
        leaves = list(_leaves(value))
        shown = [f"{json.dumps(path, ensure_ascii=False)}: {_scalar(leaf)}" for path, leaf in leaves[:entries]]
        more = f" …(+{len(leaves) - entries})" if len(leaves) > entries else ""
        return "{" + ", ".join(shown) + more + "}"
    return _scalar(value)


def _stated_numbers(cell: str) -> list:
    """Numbers a headline cell states: string literals and ``…(+N)`` markers removed."""
    text = re.sub(r"…\(\+\d+\)", "", RAW_STRING.sub('""', cell))
    return [float(token) for token in NUMBER.findall(text)]


def _headline_problem(cell: str, value) -> bool:
    """A value cell differs from its source value's headline or states a number the value does not hold."""
    held = {float(number) for number in _numbers(value)}
    return cell != _headline(value) or any(number not in held for number in _stated_numbers(cell))


def _cell(value) -> str:
    """A Markdown table cell: backslashes and pipes escaped, line breaks flattened."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return _flat(text).replace("\\", "\\\\").replace("|", "\\|")


def _row(*cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _cells(line: str):
    """Cells of a table row written by :func:`_row`, escapes undone; None for a malformed row."""
    text = line.strip()
    if len(text) < 2 or not (text.startswith("|") and text.endswith("|")):
        return None
    cells, current, escaped = [], [], False
    for char in text[1:]:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    return None if escaped or "".join(current).strip() else cells


def _table_rows(text: str, header: str) -> list:
    """Parsed body rows of the Markdown table whose header row starts with ``header``."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if start is None:
        return []
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append(_cells(line))
    return rows


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": compare(observed, tolerance, comparison)}


def _count(claim, domain, value, checks, unit, basis=None):
    """A count finding with its checks, exact uncertainty and zero regression tolerance."""
    return finding(claim, domain, value, {"checks": checks}, unit=unit,
                   uncertainty=_exact(basis) if basis else COUNT, tolerance=ZERO)


def _refuted(findings) -> bool:
    return any(f["domain"] in COMPUTATIONAL_DOMAINS and f["evidence_status"] == "not_established"
               and not f.get("expected_not_established") for f in findings)


def _state(findings, complete=True) -> str:
    """``completed`` only when every planned part ran and no computational finding is refuted."""
    return "completed" if complete and not _refuted(findings) else "partial"


def _no_prior(fields):
    fields = dict(fields)
    fields["unresolved_assumptions"] = list(fields.get("unresolved_assumptions", [])) + [
        "No earlier queue reports were retained in this output directory; run the full queue first."]
    return {"state": "blocked", "fields": fields, "findings": []}


READS_RECORDS = "No physical observation; reads retained computational records only."


def _fields(hypothesis, model, inputs, invariant, experiment, failure_modes, next_task, *,
            observation=READS_RECORDS, assumptions=()):
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": list(inputs),
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "failure_modes_checked": list(failure_modes), "recommended_next_task": next_task,
            "unresolved_assumptions": list(assumptions)}


def _earlier(number: int) -> str:
    return f"Retained reports of T001-T{number - 1:03d} in the same output directory"


# --------------------------------------------------------------- T155
# Rule 5 of docs/lab/SPECIFICATIONS.md, restated here rather than imported, so
# that a change to ciw.lab.evidence that departs from the written rules shows
# up as disagreement with the oracle.
SPEC_PHYSICAL = frozenset({"physical", "calibration", "sensor_performance"})
SPEC_AUTHORITY = frozenset({"machine_safety", "industrial_readiness", "customer_demand", "actuator_authority",
                            "production_acceptance"})
SPEC_FAMILIES = frozenset({"ciw", "scipy", "sympy", "mpmath", "numpy", "cpython", "zlib", "git",
                           "curved-surface-geodesic-sensitivity-runtime", "flat-torus-geodesic-reference",
                           "parameterized-lyapunov-stability-runtime", "scientific-computation-runtime"})
RULE_BRANCHES = ("1 authority domain", "2 failed check", "3 no acquisition", "3 acquisition with I",
                 "3 acquisition", "4 acquisition refused", "4 I", "4 C", "4 P", "4 G", "4 D", "4 no basis",
                 "5 cross_implementation is not I", "5 unknown family", "5 ciw inside another family",
                 "5 same origin")


def _family(identifier: str):
    text = unicodedata.normalize("NFKC", identifier).strip()
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9_-]*", text) if text.isascii() else None
    return match.group(0).casefold() if match else None


def _reference_label(basis: dict, domain: str) -> tuple:
    """Rules 1-5 of docs/lab/SPECIFICATIONS.md, written without ciw.lab.evidence: (label, rule branch)."""
    if domain in SPEC_AUTHORITY:
        return "not_established", "1 authority domain"
    checks = basis.get("checks") or []
    failed = any(abs(check["observed"]) > check["tolerance"] for check in checks)
    independent = basis.get("independent_check")
    if independent is not None:
        if independent["reference_kind"] == "cross_implementation":
            return "refused", "5 cross_implementation is not I"
        names = [independent[role]["implementation"] for role in ("producer", "checker")]
        families = [_family(name) for name in names]
        if any(family not in SPEC_FAMILIES for family in families):
            return "refused", "5 unknown family"
        if any(family != "ciw" and "ciw" in re.split(r"[^a-z0-9]+", name.casefold())
               for name, family in zip(names, families)):
            return "refused", "5 ciw inside another family"
        if families[0] == families[1]:
            return "refused", "5 same origin"
        failed = failed or abs(independent["observed"]) > independent["tolerance"]
    if failed:
        return "not_established", "2 failed check"
    acquired = basis.get("acquisition") is not None
    if domain in SPEC_PHYSICAL:
        if not acquired:
            return "not_established", "3 no acquisition"
        return ("independently_verified", "3 acquisition with I") if independent else ("hardware_measured", "3 acquisition")
    if acquired:
        return "refused", "4 acquisition refused"
    if independent is not None:
        return "independently_verified", "4 I"
    if checks:
        return "numerically_verified", "4 C"
    provider = basis.get("provider")
    if provider is not None and provider.get("executed") is True:
        return "provider_backed", "4 P"
    if basis.get("generator") is not None:
        return "synthetic", "4 G"
    if basis.get("derivation") is not None:
        return "analytic", "4 D"
    return "not_established", "4 no basis"


def _basis_grammar() -> dict:
    passing = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
    failing = dict(passing, observed=2.0, passed=False)

    def independent(producer, checker, **change):
        return dict(passing, producer={"implementation": producer, "revision": "r"},
                    checker={"implementation": checker, "revision": "r"}, **change)

    provider = {"repository": "p", "revision": "r", "source_tree": "t", "executed": True}
    return {
        "derivation": (None, "docs/lab/SPECIFICATIONS.md"),
        "generator": (None, {"name": "g", "seed": 1}),
        "checks": (None, [passing], [failing], [passing, failing]),
        "provider": (None, provider, dict(provider, executed=False)),
        "independent_check": (None,
                              independent("ciw.lab", "scipy.integrate.solve_ivp"),
                              independent("scientific-computation-runtime", "ciw.lab"),
                              independent("ciw.lab", "mpmath", observed=2.0, passed=False),
                              independent("ciw.lab.jacobi", "ciw.core"),
                              independent("ciw.lab", "scipy", reference_kind="cross_implementation"),
                              independent("ciw.lab", "fortranlib"),
                              independent("numpy.ciw_bridge", "scipy")),
        "acquisition": (None, {"device": "d", "raw_sha256": "0" * 64, "acquired_at": "t", "calibration": "not_applied"}),
    }


def label_invariants() -> dict:
    """Compare the label function with the reference oracle on every basis of a finite grammar in every domain."""
    options = _basis_grammar()
    violations, cases, labels, branches = [], 0, Counter(), Counter()
    for values in product(*(range(len(choices)) for choices in options.values())):
        basis = {key: options[key][index] for key, index in zip(options, values) if options[key][index] is not None}
        signature = ",".join(f"{key[:3]}{index}" for key, index in zip(options, values))
        for domain in sorted(DOMAINS):
            cases += 1
            try:
                label = supported_label(basis, domain)
            except EvidenceRefusal:
                label = "refused"
            expected, branch = _reference_label(basis, domain)
            labels[label] += 1
            branches[branch] += 1
            if label != expected:
                violations.append(f"{domain} [{signature}]: {label}, rule {branch} gives {expected}")
    return {"cases": cases, "grammar": {key: len(choices) for key, choices in options.items()},
            "violation_count": len(violations), "violations": violations[:50],
            "label_counts": dict(sorted(labels.items())), "rule_branches": dict(sorted(branches.items())),
            "unexercised_branches": [branch for branch in RULE_BRANCHES if not branches[branch]]}


FAMILY_INDEX = "Specifications of record by family"


def _task_ids(text: str, title: str = "") -> set:
    """Computational task identities named in ``text``, plus ranges written in a page ``title``."""
    ids = {int(n) for n in re.findall(r"\bT(\d{3})\b", text)}
    for first, last in re.findall(r"T(\d{3})\s*[–-]\s*T(\d{3})", title):
        ids |= set(range(int(first), int(last) + 1))
    return {f"T{n:03d}" for n in ids if 0 < n < FIRST_OWN}


def specification_documents():
    """Specification units and the tasks each names, or None when docs/lab is not reachable.

    Units are the sections of docs/lab/SPECIFICATIONS.md (its family index
    aside) and the family pages that index lists as specifications of record.
    A section that names no computational task (the aggregates of this
    section) is not a unit of computational coverage.
    """
    from .runner import repository_path
    page = repository_path(*SPECIFICATIONS.split("/"))
    if page is None or not page.is_file():
        return None
    text = page.read_text(encoding="utf-8")
    units, missing, pages = {}, [], []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = section.partition("\n")
        if title.strip() == FAMILY_INDEX:
            pages = sorted(set(re.findall(r"\]\(([A-Z_]+\.md)\)", body)))
        elif _task_ids(body):
            units[f"SPECIFICATIONS.md: {title.strip()}"] = _task_ids(body)
    for name in pages:
        path = page.parent / name
        if not path.is_file():
            missing.append(name)
            continue
        family = path.read_text(encoding="utf-8")
        units[name] = _task_ids(family, family.partition("\n")[0])
    return {"units": units, "missing_pages": missing}


@task("T155", changed_files=(MODULE, SPECIFICATIONS),
      regression_tests=(f"{TESTS}::test_label_function_matches_the_reference_oracle_exhaustively",
                        f"{TESTS}::test_formal_specifications_cover_the_queue"))
def formal_specifications(ctx):
    reports = _reports_before(ctx, 155)
    invariants = label_invariants()
    grammar = " x ".join(str(n) for n in invariants["grammar"].values())
    fields = _fields(
        "The label function agrees with rules 1-5 of the specification on every basis of a finite grammar, every "
        "computational queue task is named by a specification document, and every specification document is "
        "exercised by retained established findings.",
        "Label function L(basis, domain) from ciw.lab.evidence against a reference oracle restating rules 1-5 of "
        "docs/lab/SPECIFICATIONS.md; specification units = its sections plus the family pages it lists as "
        "specifications of record.",
        [_earlier(155), "docs/lab/SPECIFICATIONS.md and the family pages it lists",
         "Packaged queue definition (tasks T001-T154)"],
        "L equals the oracle on every case (refusals included), every oracle rule branch is exercised, every "
        "computational task is named, and established-finding counts agree with a raw-text recount.",
        f"Enumerate {grammar} bases x {len(DOMAINS)} domains = {invariants['cases']} cases (failing, reversed, "
        "same-origin, cross-implementation, unknown-family and embedded-ciw independent checks included) and compare "
        "L with the oracle; map each specification unit to the tasks it names and count established findings.",
        ["label function departing from a written rule (precedence, origin rule, physical gate, authority)",
         "grammar that leaves a rule branch unexercised", "computational task named by no specification",
         "family page listed but absent", "established-finding traversal disagreeing with the report text"],
        "Machine-check the label rules (for example in Lean or with a SAT encoding over the basis grammar).",
        assumptions=[
            "Only the families on SPECIFICATIONS.md are stated in formal notation; the family pages it lists specify "
            "the other families as method descriptions (models, references and decision rules), not in one uniform "
            "notation.",
            "T155 does not recompute retained values from the specification formulas; each task's own reference "
            "checks compare its results with its model.",
            "The oracle restates rules 1-5 by hand: agreement shows that the implementation and the written rules "
            "coincide on the grammar, not that the rules are the right ones or that they hold off the grammar."])
    ctx.artifact_json("label-invariants.json", invariants)
    findings = [finding(
        "The evidence-label function agrees with the reference oracle for rules 1-5 on the exhaustive basis grammar",
        "mathematical", invariants["violation_count"],
        {"derivation": "docs/lab/SPECIFICATIONS.md#evidence-labels",
         "checks": [_check("cases where L differs from the rules 1-5 oracle", invariants["violation_count"]),
                    _check("oracle rule branches not exercised by the grammar", len(invariants["unexercised_branches"]))]},
        unit="cases", uncertainty=_exact("exhaustive enumeration of a finite grammar"), tolerance=ZERO)]
    documents = specification_documents()
    queue_ids = [t["id"] for t in load_queue()["tasks"] if t["number"] < FIRST_OWN]
    complete = documents is not None and bool(reports)
    if documents is None:
        fields["unresolved_assumptions"].append(
            "docs/lab is not reachable from this installation (set CIW_LAB_REPOSITORY_ROOT to a checkout); "
            "specification coverage was not evaluated.")
        findings += [finding("Every computational queue task is named by a specification document", "provenance",
                             None, {}, expected_not_established=True),
                     finding("Specification documents are exercised by retained established findings",
                             "computational_pipeline", None, {}, expected_not_established=True)]
        fields["numerical_result"] = (f"{invariants['cases']} label cases, {invariants['violation_count']} "
                                      "disagreements with the oracle; specification coverage not evaluated.")
        fields["uncertainty"] = "Exact enumeration."
        return {"state": _state(findings, complete), "fields": fields, "findings": findings}
    units = documents["units"]
    named = set().union(*units.values()) if units else set()
    unnamed = [tid for tid in queue_ids if tid not in named]
    findings.append(_count(
        "Every computational queue task is named by a specification document", "provenance", len(unnamed),
        [_check("computational queue tasks named by no specification document", len(unnamed)),
         _check("family pages listed in the index but absent", len(documents["missing_pages"]))], "tasks"))
    established = {r["task_id"]: sum(f["evidence_status"] != "not_established" for f in r["findings"]) for r in reports}
    recount = {tid: sum(label != "not_established" for label in RAW_LABEL.findall(text))
               for tid, text in _raw_reports(ctx, 155)}
    disagreements = sorted(tid for tid in set(established) | set(recount) if established.get(tid) != recount.get(tid))
    coverage = {unit: {tid: established.get(tid) for tid in sorted(ids)} for unit, ids in units.items()}
    uncovered = [unit for unit, row in coverage.items() if not any(row.values())]
    ctx.artifact_json("specification-coverage.json", {"units": coverage, "unnamed_tasks": unnamed,
                                                      "missing_pages": documents["missing_pages"],
                                                      "uncovered_units": uncovered,
                                                      "recount_disagreements": disagreements})
    if reports:
        findings.append(_count(
            "Specification documents are exercised by retained established findings", "computational_pipeline",
            len(coverage) - len(uncovered),
            [_check("specification units whose named tasks retain no established finding", len(uncovered)),
             _check("tasks whose established-finding count differs from a raw-text recount", len(disagreements))],
            "units"))
    else:
        fields["unresolved_assumptions"].append("Coverage could not be evaluated: no earlier reports were retained.")
        findings.append(finding("Specification documents are exercised by retained established findings",
                                "computational_pipeline", None, {}, expected_not_established=True))
    if unnamed:
        fields["unresolved_assumptions"].append("Computational tasks named by no specification: " + ", ".join(unnamed))
    if uncovered:
        fields["unresolved_assumptions"].append("Specification units without established findings: " + "; ".join(uncovered))
    fields["numerical_result"] = (
        f"{invariants['cases']} label cases, {invariants['violation_count']} disagreements with the rules 1-5 oracle, "
        f"{len(invariants['unexercised_branches'])} unexercised rule branches; {len(units)} specification units name "
        f"{len(named & set(queue_ids))}/{len(queue_ids)} computational tasks; "
        f"{len(coverage) - len(uncovered)}/{len(coverage)} units have established findings.")
    fields["uncertainty"] = "Exact enumeration and exact counts; coverage depends on which tasks ran in this output directory."
    return {"state": _state(findings, complete), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T156
# Textbook results with a reference and the pattern that names them in report
# text (hypothesis, model, experiment, invariant, inputs, failure modes, claims).
TEXTBOOK = (
    ("Geodesic equation and Christoffel symbols", "do Carmo (1976), Differential Geometry of Curves and Surfaces, ch. 4",
     r"Christoffel|(?i:geodesic equation)"),
    ("Jacobi equation j'' + K j = 0; conjugate and focal points", "do Carmo (1992), Riemannian Geometry, ch. 5",
     r"\bJacobi\b|(?i:conjugate point|focal)"),
    ("Clairaut relation on surfaces of revolution", "do Carmo (1976), section 4-4", r"Clairaut"),
    ("Gauss-Bonnet theorem; cone angles", "do Carmo (1976), section 4-5; Troyanov (1986)",
     r"Gauss[-–]Bonnet|(?i:cone angle|cone point)"),
    ("Chord-arc expansion c = s - kappa^2 s^3/24", "Taylor expansion of a space curve (do Carmo 1976, ch. 1)",
     r"(?i:\bchord)"),
    ("Explicit Runge-Kutta methods (Euler, midpoint, RK4, Dormand-Prince 5(4))",
     "Hairer, Norsett and Wanner (1993), Solving ODEs I", r"\bRK4\b|Runge|Dormand|(?i:explicit euler|\bmidpoint\b)"),
    ("Implicit and A-stable integrators; stiffness", "Hairer and Wanner (1996), Solving ODEs II",
     r"(?i:implicit midpoint|gauss[-–]legendre|a-stable|\bstiff)"),
    ("Richardson extrapolation and observed order of convergence", "Richardson (1911); Roache (1998)",
     r"Richardson|(?i:observed order|log-log)"),
    ("Kalman filter, extended Kalman filter and Rauch-Tung-Striebel smoother",
     "Kalman (1960); Rauch, Tung and Striebel (1965)", r"Kalman|\bEKF\b|Rauch|(?i:smoother|innovation)"),
    ("NEES/NIS chi-square consistency, gating and Mahalanobis distance", "Bar-Shalom, Li and Kirubarajan (2001)",
     r"\bNEES\b|\bNIS\b|Mahalanobis|(?i:chi-square)|χ²"),
    ("Brioschi formula; Theorema Egregium; fundamental forms", "Brioschi (1852); Gauss (1827)",
     r"Brioschi|Egregium|(?i:fundamental form)"),
    ("Lattice reduction and the SL(2,Z) action", "Lagrange (1773) and Gauss (1801) reduction of binary quadratic forms",
     r"(?i:\blattice)|SL\(2"),
    ("Gage R&R by the ANOVA method", "AIAG, Measurement Systems Analysis, 4th ed. (2010)", r"\bANOVA\b|\bGage\b"),
    ("Lyapunov equation, Hurwitz stability and Sylvester equations", "Lyapunov (1892); Khalil (2002), Nonlinear Systems",
     r"Lyapunov|Hurwitz|Sylvester"),
    ("Bartels-Stewart and Schur-form solvers", "Bartels and Stewart (1972)", r"Bartels|\bSchur\b"),
    ("Compensated and pairwise summation", "Kahan (1965); Neumaier (1974); Higham (2002), ch. 4",
     r"Kahan|Neumaier|(?i:compensated summation|pairwise)"),
    ("Fast marching and Dijkstra shortest paths", "Sethian (1996); Dijkstra (1959)", r"Dijkstra|(?i:fast marching)"),
    ("Monte Carlo estimation and binomial confidence intervals", "Metropolis and Ulam (1949); Wilson (1927)",
     r"Monte Carlo|\bWilson\b"),
    ("Matrix factorizations and least squares (Cholesky, QR, SVD, eigenvalues)", "Golub and Van Loan (2013)",
     r"Cholesky|\bQR\b|\bSVD\b|(?i:least[- ]squares|eigenvalue)"),
    ("Floating-point arithmetic, rounding and exact rational reference arithmetic",
     "IEEE 754-2019; Goldberg (1991); Higham (2002)", r"binary64|float64|IEEE|\bULP\b|(?i:rounding|exact rational)"),
    ("Finite differences and step-size selection", "Nocedal and Wright (2006), ch. 8", r"(?i:finite[- ]difference)"),
    ("Automatic differentiation with dual numbers", "Griewank and Walther (2008)",
     r"(?i:automatic differentiation|dual number)"),
    ("Pinhole camera model and lens distortion", "Hartley and Zisserman (2004); Brown (1966)",
     r"(?i:pinhole|distortion)"),
    ("Rigid registration (Kabsch)", "Kabsch (1976)", r"Kabsch|Procrustes"),
    ("Interpolation and quadrature (Hermite, trapezoid, Gauss)", "Burden and Faires (2010)",
     r"Hermite|(?i:quadrature|trapezoid|interpolation)"),
    ("Measurement uncertainty (GUM) and tolerance stacking", "JCGM 100:2008 (GUM)",
     r"\bGUM\b|Type A|(?i:measurement uncertainty|tolerance stack)"),
    ("Cryptographic digests and canonical serialization", "FIPS 180-4 (SHA-256); RFC 8785 (JSON canonicalization)",
     r"SHA-256|\bCRC\b|zlib|(?i:canonical json)"),
    ("Root finding by bisection and Newton iteration", "Burden and Faires (2010), ch. 2", r"Newton|(?i:bisection)"),
    ("Roofline model and operation counts", "Williams, Waterman and Patterson (2009)", r"(?i:roofline|\bflops?\b)"),
    ("Sampling and aliasing", "Shannon (1949)", r"(?i:aliasing|nyquist)"),
)
# Curated descriptions of workbench contributions; every other package file a
# report names is listed as an implementation artifact without a description.
CONTRIBUTIONS = (
    ("Evidence-label validator with non-upgrade rule and origin-based independence", "src/ciw/lab/evidence.py"),
    ("Nineteen-question task report with derived evidence and physical status", "src/ciw/lab/report.py"),
    ("Queue runner that reports blocked, deferred and failed tasks instead of hiding them", "src/ciw/lab/runner.py"),
    ("Chart-level surface interface with exact embedding derivatives and pullback charts", "src/ciw/lab/surfaces.py"),
    ("Joint geodesic/Jacobi transfer integration without renormalization", "src/ciw/lab/jacobi.py"),
    ("Identity matrix and mutation experiments over retained CIW workspaces", "src/ciw/lab/exchange_provenance.py"),
    ("Counterexample library for shortest-versus-safest routes", "src/ciw/lab/flat_torus_topology.py"),
    ("Focus-margin route ranking for manufacturing paths", "src/ciw/lab/manufacturing.py"),
    ("Deterministic SVG figures", "src/ciw/lab/svg.py"),
    ("Retained-evidence aggregation with second-path checks", MODULE),
)
REPORT_TEXT_FIELDS = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                      "experiment", "failure_modes_checked")


def _report_text(report) -> str:
    parts = [report[name] if isinstance(report[name], str) else json.dumps(report[name], ensure_ascii=False)
             for name in REPORT_TEXT_FIELDS]
    return "\n".join(parts + [f["claim"] for f in report["findings"]])


def _package_file(path: str) -> bool:
    return path.startswith("src/ciw/") and (Path(__file__).resolve().parents[1] / path[len("src/ciw/"):]).is_file()


@task("T156", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_textbook_and_contribution_ledger",))
def textbook_versus_contribution(ctx):
    reports = _reports_before(ctx, 156)
    fields = _fields(
        "Every result used by the lab can be attributed either to established literature or to a named "
        "implementation artifact in this repository.",
        "Per retained task: textbook results whose names its report text uses (pattern per ledger entry) and the "
        "package source files its report names as changed files; a task with neither is unattributed.",
        [_earlier(156) + " (report text and changed files)",
         "The textbook ledger and curated contribution descriptions in src/ciw/lab/research_portfolio.py",
         "Package source files named by the reports"],
        "Every retained task is attributed; every ledger textbook entry is used by a retained task; every curated "
        "contribution names a present source file.",
        "Match each ledger entry's pattern against each retained report's text, collect the package files each "
        "report names, and write the per-task ledger as JSON and Markdown.",
        ["task with no textbook match and no present source file", "ledger entry that no retained task uses",
         "curated contribution naming a missing file", "report identity mismatch (validate_report)"],
        "Record per-finding provenance tags (textbook result or contribution) so attribution is declared by each "
        "task rather than inferred from its wording.",
        assumptions=["Novelty of contributions relative to published literature is not established by this ledger.",
                     "Textbook attribution matches names in report text; a method a report uses without naming it "
                     "is not attributed, and a name used in another sense can be attributed wrongly."])
    missing = [path for _, path in CONTRIBUTIONS if not _package_file(path)]
    contributions = _count("Curated implementation contributions each name a present source file", "provenance",
                           len(missing), [_check("curated contribution files missing from the package", len(missing))],
                           "files")
    novelty = finding("Contributions are novel relative to the literature", "mathematical", None, {},
                      expected_not_established=True)
    if not reports:
        # The curated half is checked; attribution needs the reports and stays unestablished.
        fields["unresolved_assumptions"].insert(0, "No earlier queue reports were retained in this output directory; "
                                                   "task attribution was not evaluated (run the full queue first).")
        fields["numerical_result"] = (f"No retained reports; {len(CONTRIBUTIONS)} curated contributions, "
                                      f"{len(missing)} missing a source file.")
        fields["uncertainty"] = "Exact counts."
        findings = [finding("Every retained task is attributed to a textbook result or a present implementation "
                            "source file", "provenance", None, {}, expected_not_established=True),
                    finding("Every textbook result in the ledger is named by a retained task", "provenance", None, {},
                            expected_not_established=True), contributions, novelty]
        return {"state": "partial", "fields": fields, "findings": findings}
    patterns = [(result, reference, re.compile(pattern)) for result, reference, pattern in TEXTBOOK]
    tasks, textbook_use = {}, {result: [] for result, _, _ in TEXTBOOK}
    for report in reports:
        text = _report_text(report)
        matched = [result for result, _, pattern in patterns if pattern.search(text)]
        sources = sorted(path for path in report["changed_files"] if _package_file(path))
        tasks[report["task_id"]] = {"textbook": matched, "sources": sources}
        for result in matched:
            textbook_use[result].append(report["task_id"])
    unattributed = [tid for tid, row in tasks.items() if not row["textbook"] and not row["sources"]]
    unused = [result for result, users in textbook_use.items() if not users]
    described = dict((path, text) for text, path in CONTRIBUTIONS)
    files = sorted({path for row in tasks.values() for path in row["sources"]} | set(described))
    ledger = {"textbook": [{"result": result, "reference": reference, "tasks": textbook_use[result]}
                           for result, reference, _ in TEXTBOOK],
              "contributions": [{"file": path, "description": described.get(path), "present": _package_file(path),
                                 "tasks": [tid for tid, row in tasks.items() if path in row["sources"]]}
                                for path in files],
              "tasks": tasks, "unattributed": unattributed, "unused_textbook": unused}
    ctx.artifact_json("attribution-ledger.json", ledger)
    lines = ["# Textbook results and implementation contributions", "",
             "Generated from retained lab reports. Textbook attribution matches names in report text.", "",
             _row("Textbook result", "Reference", "Tasks"), _row("---", "---", "---")]
    lines += [_row(_cell(e["result"]), _cell(e["reference"]), _cell(", ".join(e["tasks"]) or "none"))
              for e in ledger["textbook"]]
    lines += ["", _row("Implementation file", "Contribution", "Tasks"), _row("---", "---", "---")]
    lines += [_row(f"`{_cell(c['file'])}`", _cell(c["description"] or "(implementation artifact)"),
                   _cell(", ".join(c["tasks"]) or "none")) for c in ledger["contributions"]]
    ctx.artifact_text("ATTRIBUTION.md", "\n".join(lines) + "\n")
    with_textbook = sum(bool(row["textbook"]) for row in tasks.values())
    fields["numerical_result"] = (
        f"{len(tasks)} reports: {with_textbook} use at least one of {len(TEXTBOOK)} textbook results, "
        f"{len(tasks) - with_textbook} rest on implementation sources only, {len(unattributed)} unattributed; "
        f"{len(unused)} ledger textbook results unused; {len(files)} implementation files, {len(missing)} curated "
        "ones missing.")
    fields["uncertainty"] = "Exact counts of name matches; references identify standard sources, not priority."
    if unattributed:
        fields["unresolved_assumptions"].append("Unattributed tasks: " + ", ".join(unattributed))
    if unused:
        fields["unresolved_assumptions"].append("Ledger textbook results no retained task names: " + "; ".join(unused))
    findings = [
        _count("Every retained task is attributed to a textbook result or a present implementation source file",
               "provenance", len(unattributed),
               [_check("retained tasks with no textbook match and no present package source file", len(unattributed))],
               "tasks"),
        _count("Every textbook result in the ledger is named by a retained task", "provenance", len(unused),
               [_check("ledger textbook results no retained report names", len(unused))], "results"),
        contributions, novelty,
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T157
@task("T157", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_counterexample_catalogue",))
def counterexample_catalogue(ctx):
    reports = _reports_before(ctx, 157)
    fields = _fields(
        "Every counterexample recorded by a queue task can be collected with its witness and evidence status.",
        "Catalogue = all retained findings carrying a counterexample statement, keyed by task and claim.",
        [_earlier(157) + " (findings with a counterexample)"],
        "The catalogue holds exactly the retained findings that carry a counterexample, with their retained labels.",
        "Collect counterexample findings from the retained reports, recount counterexample keys in the raw report "
        "text, and write JSON and Markdown catalogues.",
        ["counterexample findings dropped by the traversal (raw-text recount)", "report identity mismatch (validate_report)"],
        "Turn each catalogued counterexample into a named regression fixture (T168).",
        assumptions=["Whether each catalogued counterexample is genuine is not re-judged here; each rests on its "
                     "source finding's checks.",
                     "Counterexamples recorded by T158 and later tasks run after this catalogue and are not in it."])
    if not reports:
        return _no_prior(fields)
    entries = [{"task_id": r["task_id"], "claim": f["claim"], "statement": f["counterexample"]["statement"],
                "witness": f["counterexample"].get("witness"), "evidence_status": f["evidence_status"],
                "report_id": r["report_id"]}
               for r in reports for f in r["findings"] if f.get("counterexample")]
    ctx.artifact_json("counterexamples.json", entries)
    scanned = sum(len(RAW_COUNTEREXAMPLE.findall(text)) for _, text in _raw_reports(ctx, 157))
    lines = ["# Counterexample catalogue", "",
             "Generated from retained lab reports. Each entry refutes the quoted general statement.", ""]
    for entry in entries:
        lines += [f"## {entry['task_id']}: {_flat(entry['statement'])}", "", f"- Finding: {_flat(entry['claim'])}",
                  f"- Evidence status: `{entry['evidence_status']}`", f"- Witness: `{_headline(entry['witness'])}`", ""]
    ctx.artifact_text("COUNTEREXAMPLES.md", "\n".join(lines))
    labels = Counter(e["evidence_status"] for e in entries)
    fields["numerical_result"] = (f"{len(entries)} counterexamples from {len({e['task_id'] for e in entries})} tasks; "
                                  f"labels {dict(sorted(labels.items()))}.")
    fields["uncertainty"] = "Exact count; each counterexample carries its own uncertainty in its source report."
    findings = [_count("Counterexample catalogue lists every retained counterexample finding", "computational_pipeline",
                       len(entries), [_check("catalogue size minus the raw-text count of counterexample keys",
                                             len(entries) - scanned)], "counterexamples")]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T158
# Figure tasks re-executed on every run, chosen to fit the section's time
# budget (about 10 s on one core). Every figure task that retains wall-clock
# timings is re-executed as well; the others are listed as not compared.
REGENERATED = ("T013", "T020", "T023", "T030", "T031", "T033", "T035", "T037", "T048", "T049", "T050", "T052",
               "T053", "T054", "T055", "T056", "T057", "T060", "T061", "T062", "T063", "T065", "T066", "T069",
               "T070", "T071", "T072", "T073", "T074", "T112", "T120", "T121", "T122", "T127", "T129", "T132",
               "T142", "T147", "T148", "T152", "T154")
WALL_CLOCK = re.compile(r"wall-clock|elapsed", re.I)
FIGURE_HYPOTHESIS = "Retained SVG figures are byte-for-byte reproducible when their tasks are re-executed."


def _well_formed(data) -> bool:
    if data is None:
        return False
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return False
    return root.tag.rsplit("}", 1)[-1] == "svg"


def _figure_index(ctx, reports) -> list:
    figures = []
    for report in reports:
        for artifact in report["generated_artifacts"]:
            if artifact["path"].endswith(".svg"):
                path = ctx.output_dir / artifact["path"]
                data = path.read_bytes() if path.is_file() else None
                figures.append({"task_id": report["task_id"], "path": artifact["path"], "sha256": artifact["sha256"],
                                "digest_matches": data is not None and hashlib.sha256(data).hexdigest() == artifact["sha256"],
                                "well_formed": _well_formed(data)})
    return figures


def _retains_wall_clock(ctx, report) -> bool:
    """Whether a task retains a JSON artifact that describes wall-clock or elapsed timings."""
    for artifact in report["generated_artifacts"]:
        path = ctx.output_dir / artifact["path"]
        if artifact["path"].endswith(".json") and path.is_file() and WALL_CLOCK.search(
                path.read_text(encoding="utf-8", errors="replace")):
            return True
    return False


@task("T158", changed_files=(MODULE, "src/ciw/lab/svg.py"),
      regression_tests=(f"{TESTS}::test_figures_are_reproducible",
                        f"{TESTS}::test_a_changed_figure_is_a_mismatch_and_a_timing_figure_a_counterexample"))
def reproducible_figures(ctx):
    from .runner import Context, run_task
    reports = _reports_before(ctx, 158)
    fields = _fields(
        FIGURE_HYPOTHESIS,
        "Figure bytes are a function of the task's data: ciw.lab.svg uses no clock or randomness, so a regenerated "
        "figure can differ only if its data does. A figure plotting wall-clock timings is expected to differ.",
        [_earlier(158) + " and their SVG artifacts",
         "Retained JSON artifacts of figure tasks (to recognize wall-clock timing records)",
         "Task implementations re-executed in a scratch directory with this run's provider bindings"],
        "A re-executed task that ends in its retained state writes every retained figure with the same SHA-256; "
        "only figures plotting wall-clock timings may differ.",
        f"Hash and parse every retained SVG; re-execute the {len(REGENERATED)} declared inexpensive figure tasks plus "
        "every figure task that retains wall-clock timings in a scratch directory, and compare each regenerated "
        "figure's SHA-256 (and, for a difference, its bytes) with the retained one.",
        ["figure bytes changed by nondeterministic data (timings, unseeded randomness, dictionary order)",
         "re-executed task ending in another state (for example an unbound provider): its figures are not comparable",
         "retained figure edited after its report (digest mismatch)", "malformed SVG"],
        "Have figure tasks declare timing figures in their reports, and re-execute every figure task on a second "
        "platform (Windows CI) outside the section's time budget.")
    if not reports:
        return _no_prior(fields)
    figures = _figure_index(ctx, reports)
    retained = {r["task_id"]: r for r in reports}
    by_task: dict = {}
    for figure in figures:
        by_task.setdefault(figure["task_id"], []).append(figure)
    timing = sorted(tid for tid in by_task if _retains_wall_clock(ctx, retained[tid]))
    selected = sorted((set(REGENERATED) & set(by_task)) | set(timing))
    implementations, _ = load_implementations()
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    outcomes, not_comparable = [], {}
    with tempfile.TemporaryDirectory(prefix="ciw-lab-figures-") as directory:
        scratch = Context(Path(directory), ctx.providers)
        for task_id in selected:
            rerun = run_task(queue[task_id], implementations.get(task_id), scratch, {})
            if rerun["state"] != retained[task_id]["state"]:
                not_comparable[task_id] = f"state {retained[task_id]['state']} -> {rerun['state']}"
                continue
            fresh = {a["path"]: a["sha256"] for a in rerun["generated_artifacts"]}
            for figure in by_task[task_id]:
                if figure["path"] not in fresh:
                    outcome = "not regenerated"
                elif fresh[figure["path"]] == figure["sha256"]:
                    outcome = "identical"
                else:
                    # Confirm a digest difference on the bytes themselves.
                    original = ctx.output_dir / figure["path"]
                    same = original.is_file() and (Path(directory) / figure["path"]).read_bytes() == original.read_bytes()
                    outcome = "identical" if same else "differs"
                outcomes.append({"task_id": task_id, "path": figure["path"], "timing": task_id in timing,
                                 "outcome": outcome})
    deterministic = [o for o in outcomes if not o["timing"]]
    # A figure its re-executed task no longer writes is a mismatch, timing figure or not.
    mismatched = [o["path"] for o in outcomes if o["outcome"] == "not regenerated"
                  or (not o["timing"] and o["outcome"] != "identical")]
    timing_outcomes = [o for o in outcomes if o["timing"]]
    timing_differs = [o["path"] for o in timing_outcomes if o["outcome"] == "differs"]
    compared = {o["task_id"] for o in outcomes}
    uncompared = [tid for tid in sorted(by_task) if tid not in compared and tid not in not_comparable]
    digest_problems = sum(not f["digest_matches"] for f in figures)
    malformed = sum(not f["well_formed"] for f in figures)
    ctx.artifact_json("figure-index.json", {"figures": figures, "wall_clock_tasks": timing, "regenerated": outcomes,
                                            "not_comparable": not_comparable, "not_reexecuted": uncompared})
    findings = []
    if deterministic:
        findings.append(_count("Re-executed figure tasks without wall-clock timings regenerate byte-identical figures",
                               "computational_pipeline", len(mismatched),
                               [_check("regenerated timing-free figures whose SHA-256 differs from the retained one",
                                       len(mismatched))], "figures"))
    else:
        findings.append(finding("Re-executed figure tasks without wall-clock timings regenerate byte-identical figures",
                                "computational_pipeline", None, {}, expected_not_established=True))
    if timing_differs:
        findings.append(finding(
            "Re-executed figures that plot wall-clock timings differ from their retained bytes",
            "computational_pipeline", len(timing_differs),
            {"checks": [_check("timing figures whose regenerated bytes differ from the retained bytes",
                               len(timing_differs), 1, "ge")]},
            unit="figures", uncertainty=COUNT, tolerance=ZERO,
            counterexample={"statement": FIGURE_HYPOTHESIS, "witness": {"figures": timing_differs}}))
    elif timing_outcomes and all(o["outcome"] == "identical" for o in timing_outcomes):
        findings.append(_count("Re-executed figures that plot wall-clock timings reproduced byte for byte in this run",
                               "computational_pipeline", 0,
                               [_check("timing figures whose regenerated bytes differ", 0)], "figures"))
    findings += [
        _count("Retained figures hash to the digests their reports record", "provenance", digest_problems,
               [_check("retained SVG files missing or differing from their recorded SHA-256", digest_problems)], "figures"),
        _count("Retained figures are well-formed SVG documents", "computational_pipeline", malformed,
               [_check("retained figures that do not parse as XML with an svg root", malformed)], "figures"),
    ]
    compared_figures = len(outcomes)
    fields["numerical_result"] = (
        f"{len(figures)} retained figures from {len(by_task)} tasks; {compared_figures} regenerated and compared from "
        f"{len(compared)} tasks: {len(mismatched)} timing-free mismatches, {len(timing_differs)} of "
        f"{len(timing_outcomes)} wall-clock timing figures differ; {len(uncompared)} tasks not re-executed, "
        f"{len(not_comparable)} not comparable; {digest_problems} digest mismatches, {malformed} malformed.")
    fields["uncertainty"] = "Byte comparison on this platform only; other platforms are not compared."
    assumptions = ["Wall-clock timing figures are recognized by a retained JSON artifact of their task that "
                   "mentions wall-clock or elapsed time; a timing figure without such a note counts as a mismatch.",
                   "Byte identity is established on one platform; Windows and other BLAS builds are not compared."]
    if uncompared:
        assumptions.insert(0, f"{len(uncompared)} figure tasks ({sum(len(by_task[t]) for t in uncompared)} figures) "
                              "were not re-executed within the section's time budget: " + ", ".join(uncompared))
    if not_comparable:
        assumptions.insert(0, "Re-executed tasks ending in another state here, figures not compared: "
                           + "; ".join(f"{tid} ({why})" for tid, why in sorted(not_comparable.items())))
    fields["unresolved_assumptions"] = assumptions
    complete = not uncompared and not not_comparable and bool(outcomes)
    return {"state": _state(findings, complete), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T159
UNCERTAINTY_HEADER = "| Task | Claim | Value | Unit | Uncertainty kind | Uncertainty value | Uncertainty basis | Evidence |"


def _uncertainty_parts(declared) -> tuple:
    """(kind, value, basis) of a declared per-finding uncertainty."""
    if declared is None:
        return "none declared", None, ""
    if isinstance(declared, dict):
        return str(declared.get("kind", "(unstated)")), declared.get("value"), _flat(declared.get("basis", ""))
    if isinstance(declared, str):
        return "(text)", None, _flat(declared)
    return "(value)", declared, ""


def _uncertainty_table_problems(text: str, rows: list) -> list:
    """Rows of the written table that do not restate their source finding and uncertainty exactly."""
    parsed = _table_rows(text, UNCERTAINTY_HEADER)
    problems = [] if len(parsed) == len(rows) else [f"{len(parsed)} table rows for {len(rows)} findings"]
    for index, (cells, row) in enumerate(zip(parsed, rows)):
        kind, value, basis = _uncertainty_parts(row["uncertainty"])
        if cells is None or len(cells) != 8:
            problems.append(f"row {index + 1}: malformed")
        elif (cells[0] != row["task_id"] or cells[1] != _flat(row["claim"]) or _headline_problem(cells[2], row["value"])
              or cells[3] != _flat(row["unit"] or "") or cells[4] != kind or _headline_problem(cells[5], value)
              or cells[6] != basis or cells[7] != f"`{row['evidence_status']}`"):
            problems.append(f"row {index + 1}: {row['task_id']} {row['claim'][:40]}")
    return problems


@task("T159", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_uncertainty_table_restates_every_numerical_finding",))
def uncertainty_budgets(ctx):
    reports = _reports_before(ctx, 159)
    fields = _fields(
        "Every numerical finding either declares an uncertainty or is identified as lacking one.",
        "Table rows = every finding whose value contains a number: (task, claim, value, unit, uncertainty kind, "
        "value and basis, evidence status). A listing of per-finding components, not a combined budget.",
        [_earlier(159) + " (this section's T155-T158 included)"],
        "Every numerical finding appears once with its declared uncertainty restated exactly; the numerical and "
        "undeclared counts agree with a raw-text recount of the report files.",
        "Collect every finding whose value contains a number, write the table as JSON and Markdown, parse the "
        "Markdown back against the source findings, and recount numerical and undeclared findings from the raw text.",
        ["uncertainty or value cells cut inside a number", "pipes in claims breaking table columns",
         "non-scalar numerical findings left out", "traversal disagreeing with the raw report text"],
        "Require structured uncertainty components (instrument, geometry, solver) on every numerical finding, so a "
        "budget can combine components of one quantity.")
    if not reports:
        return _no_prior(fields)
    rows, total = [], 0
    for report in reports:
        for record in report["findings"]:
            total += 1
            if _numbers(record["value"]):
                rows.append({"task_id": report["task_id"], "claim": record["claim"], "value": record["value"],
                             "unit": record.get("unit"), "uncertainty": record.get("uncertainty"),
                             "evidence_status": record["evidence_status"]})
    lacking = [row for row in rows if row["uncertainty"] is None]
    ctx.artifact_json("uncertainty-budget.json", rows)
    lines = ["# Per-finding uncertainty listing", "",
             "Every retained finding whose value contains a number, with its declared uncertainty. Components are "
             "listed, not combined: the findings of one task measure different quantities in different units.", "",
             UNCERTAINTY_HEADER, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        kind, value, basis = _uncertainty_parts(row["uncertainty"])
        lines.append(_row(row["task_id"], _cell(row["claim"]), _cell(_headline(row["value"])), _cell(row["unit"] or ""),
                          _cell(kind), _cell(_headline(value)), _cell(basis), f"`{row['evidence_status']}`"))
    kinds = Counter(_uncertainty_parts(row["uncertainty"])[0] for row in rows)
    lines += ["", _row("Uncertainty kind", "Findings"), _row("---", "---")] + [
        _row(_cell(kind), str(count)) for kind, count in sorted(kinds.items())]
    text = "\n".join(lines) + "\n"
    ctx.artifact_text("uncertainty-budget.md", text)
    problems = _uncertainty_table_problems(text, rows)
    raw_numeric = raw_lacking = 0
    for _, raw in _raw_reports(ctx, 159):
        for block in _raw_findings(raw):
            if _raw_numeric(block):
                raw_numeric += 1
                raw_lacking += re.search(r'^   "uncertainty": null', block, re.M) is not None
    by_task = Counter(row["task_id"] for row in lacking)
    fields["numerical_result"] = (
        f"{total} findings, {len(rows)} numerical (value contains a number); {len(rows) - len(lacking)} declare a "
        f"per-finding uncertainty, {len(lacking)} do not; kinds {dict(sorted(kinds.items()))}.")
    fields["uncertainty"] = "Exact counts; declared values are restated in full, and undeclared uncertainty is reported, not imputed."
    fields["unresolved_assumptions"] = [
        "Components are listed per finding and not combined: the findings of one task measure different quantities "
        "in different units, so a root-sum-square across them has no meaning, and no finding declares separate "
        "components of one quantity.",
        "Findings of T160-T168 run after this table; the section's regression test asserts that each numerical one "
        "declares an uncertainty."]
    if lacking:
        fields["unresolved_assumptions"].insert(0, f"{len(lacking)} numerical findings declare no per-finding "
                                                   "uncertainty: " + ", ".join(f"{tid} ({n})" for tid, n in sorted(by_task.items())))
    findings = [
        _count("Uncertainty table restates every numerical finding with its declared uncertainty",
               "computational_pipeline", len(rows),
               [_check("table rows that do not parse back to their source finding and uncertainty", len(problems)),
                _check("numerical findings minus a raw-text recount of numerical finding values",
                       len(rows) - raw_numeric)], "findings"),
        _count("Numerical findings that declare no per-finding uncertainty", "computational_pipeline", len(lacking),
               [_check("undeclared numerical findings minus a raw-text recount", len(lacking) - raw_lacking)], "findings"),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T160-T162
PAPERS = {
    "T160": ("instrument-methods", "Evidence-labelled computational instruments for curved-surface observation",
             ("observation", "sensor-fusion", "manufacturing", "energy-gpu")),
    "T161": ("geometry-methods", "Geodesic, Jacobi and route-sensitivity experiments with explicit validity domains",
             ("geodesic-jacobi", "flat-torus-topology", "surfaces-discrete")),
    "T162": ("evidence-provenance-note", "A non-upgrading evidence-label discipline for computational experiments",
             ("exchange-provenance", "implementation-targets", "lyapunov")),
}
RESULTS_HEADER = "| Task | Finding | Value | Unit | Evidence | Report |"


def _article(phrase: str) -> str:
    return ("An " if phrase[:1].lower() in "aeiou" else "A ") + phrase


def _specification_section(title: str):
    """Body of one section of docs/lab/SPECIFICATIONS.md, or None when docs/lab is not reachable."""
    from .runner import repository_path
    page = repository_path(*SPECIFICATIONS.split("/"))
    if page is None or not page.is_file():
        return None
    for section in re.split(r"^## ", page.read_text(encoding="utf-8"), flags=re.M)[1:]:
        heading, _, body = section.partition("\n")
        if heading.strip() == title:
            return body.strip()
    return None


def _spec_rules(body: str) -> list:
    """The numbered rules of a specification section, each as one flattened line."""
    rules, current = [], None
    for line in body.splitlines():
        if re.match(r"\d+\. ", line):
            current = [line.strip()]
            rules.append(current)
        elif current is not None and line.startswith("   "):
            current.append(line.strip())
        else:
            current = None
    return [" ".join(parts) for parts in rules]


def _label_discipline() -> tuple:
    """Methods text for the label discipline: the specification's rules and the validator's constants."""
    order = " < ".join(f"`{label}`" for label in COMPUTATIONAL_ORDER[1:])
    lines = ["### Evidence-label discipline", "",
             f"Labels: {', '.join(f'`{label}`' for label in LABELS)}.",
             f"Report headline: the weakest established computational label in the order {order}; `not_established` "
             "when a computational finding is refuted or none is established.",
             f"Computational domains: {', '.join(sorted(COMPUTATIONAL_DOMAINS))}. Physical domains (need a hardware "
             f"acquisition): {', '.join(sorted(PHYSICAL_DOMAINS))}. Authority domains (never established): "
             f"{', '.join(sorted(AUTHORITY_DOMAINS))}.",
             f"Independent origins besides `ciw`: {', '.join(sorted(INDEPENDENT_ORIGINS))}.",
             f"Check comparisons: {', '.join(COMPARISONS)}; thresholds bounded by {MAX_THRESHOLD:g}.", ""]
    body = _specification_section("Evidence labels")
    rules = _spec_rules(body) if body else []
    if rules:
        lines += ["The label function `ciw.lab.evidence.supported_label` and the report rules, as specified in "
                  "docs/lab/SPECIFICATIONS.md:", ""] + rules + [""]
    return lines, rules


def _draft_problems(text: str, cited: list) -> list:
    """Results rows of a written draft that do not restate their source finding (task, claim, value, unit, label, report)."""
    parsed = _table_rows(text, RESULTS_HEADER)
    problems = [] if len(parsed) == len(cited) else [f"{len(parsed)} results rows for {len(cited)} findings"]
    for index, (cells, (report, record)) in enumerate(zip(parsed, cited)):
        if cells is None or len(cells) != 6:
            problems.append(f"row {index + 1}: malformed")
        elif (cells[0] != report["task_id"] or cells[1] != _flat(record["claim"])
              or _headline_problem(cells[2], record["value"]) or cells[3] != _flat(record.get("unit") or "")
              or cells[4] != f"`{record['evidence_status']}`" or cells[5] != f"`{report['report_id'][:19]}`"):
            problems.append(f"row {index + 1}: {report['task_id']} {record['claim'][:40]}")
    return problems


def _paper(task_id, ctx):
    slug, title, sections = PAPERS[task_id]
    reports = [r for r in _reports_before(ctx, int(task_id[1:])) if r["section"] in sections]
    fields = _fields(
        f"{_article(slug.replace('-', ' '))} draft can be assembled entirely from retained findings, with every "
        "number traceable to a report identity and every limitation drawn from not_established findings.",
        "Draft = generated Markdown: abstract, methods (task hypotheses and models"
        + ("; the evidence-label rules of the specification" if task_id == "T162" else "")
        + "), results (one row per finding), limitations (not_established findings and unfinished tasks).",
        [f"Retained reports of sections {', '.join(sections)}"]
        + (["docs/lab/SPECIFICATIONS.md (evidence-label rules)", "ciw.lab.evidence constants"] if task_id == "T162" else []),
        "Every results row restates its source finding's task, claim, value headline, unit, label and report "
        "identity, and its value cell states no number that the source finding's value does not hold.",
        f"Generate the draft from retained reports of sections {', '.join(sections)} and parse its results table back "
        "against the source findings.",
        ["claims or units containing '|' breaking table rows", "labels dropped from rows",
         "values cut inside a number", "rows out of order or missing"],
        "Human authorship pass, related-work section and external review; physical experiments from the "
        "manufacturing protocols.",
        assumptions=["Prose beyond the generated structure, related work and peer review are outstanding."])
    if not reports:
        return _no_prior(fields)
    measured = [r for r in reports if r["physical_validation_status"]["status"] != "not_established"]
    hardware = sum(f["evidence_status"] == "hardware_measured" for r in reports for f in r["findings"])
    physical = ("physical validation is not established for any of them" if not measured else
                f"physical validation is established for {len(measured)} of them ({', '.join(r['task_id'] for r in measured)})")
    lines = [f"# {title}", "",
             f"*Generated draft from retained CIW lab reports. Not peer reviewed. Contains {hardware or 'no'} "
             f"hardware-measured finding{'' if hardware == 1 else 's'}.*", "",
             "## Abstract", "", f"This draft summarizes {len(reports)} queued computational tasks. Every result below "
             f"carries the evidence label assigned by `ciw.lab.evidence`; {physical}.", "", "## Methods", ""]
    rules = []
    if task_id == "T162":
        discipline, rules = _label_discipline()
        lines += discipline
    for report in reports:
        model = report["mathematical_model"]
        lines += [f"### {report['task_id']} — {report['title']}", "", f"*Hypothesis.* {_flat(report['hypothesis'])}", "",
                  f"*Model.* {_flat(model if isinstance(model, str) else json.dumps(model, ensure_ascii=False))}", ""]
    lines += ["## Results", "", RESULTS_HEADER, "| --- | --- | --- | --- | --- | --- |"]
    cited = [(report, record) for report in reports for record in report["findings"]]
    for report, record in cited:
        lines.append(_row(report["task_id"], _cell(record["claim"]), _cell(_headline(record["value"])),
                          _cell(record.get("unit") or ""), f"`{record['evidence_status']}`",
                          f"`{report['report_id'][:19]}`"))
    lines += ["", "## Limitations", ""]
    for report in reports:
        for record in report["findings"]:
            if record["evidence_status"] == "not_established":
                lines.append(f"- {report['task_id']}: {_flat(record['claim'])} — not established.")
        if report["state"] in ("blocked", "deferred", "partial"):
            lines.append(f"- {report['task_id']} is {report['state']}: {_words(report['experiment'], 200)}")
    text = "\n".join(lines) + "\n"
    ctx.artifact_text(f"{slug}-draft.md", text)
    problems = _draft_problems(text, cited)
    fields["numerical_result"] = f"Draft cites {len(cited)} findings from {len(reports)} reports; {len(problems)} results rows differ from their source."
    fields["uncertainty"] = "Numbers carry the uncertainty stated in their source findings."
    findings = [_count("Draft results table restates every retained finding of its sections with its retained label",
                       "provenance", len(cited),
                       [_check("results rows that do not parse back to their source finding", len(problems))], "findings")]
    if task_id == "T162":
        # The stated rules are backed by T155, which compares the label function with an oracle of those rules.
        oracle = next((f for r in _reports_before(ctx, 156) if r["task_id"] == "T155" for f in r["findings"]
                       if f["claim"].startswith("The evidence-label function agrees with the reference oracle")), None)
        claim = "Draft methods state the evidence-label rules that T155 found the label function to follow"
        if rules and oracle is not None and isinstance(oracle["value"], int):
            findings.append(_count(claim, "provenance", len(rules),
                                   [_check("T155 cases where the label function departs from the stated rules",
                                           oracle["value"])], "rules"))
        else:
            fields["unresolved_assumptions"].append(
                "The methods state the validator's constants only, or T155's oracle comparison is not retained here; "
                "the stated rules are not backed by T155 in this run.")
            findings.append(finding(claim, "provenance", None, {}, expected_not_established=True))
    findings.append(finding("Draft has passed external peer review", "provenance", None, {}, expected_not_established=True))
    return {"state": "partial", "fields": fields, "findings": findings}


for _task_id in PAPERS:
    task(_task_id, changed_files=(MODULE,),
         regression_tests=(f"{TESTS}::test_paper_drafts_trace_to_reports",))(lambda ctx, _t=_task_id: _paper(_t, ctx))


# --------------------------------------------------------------- T163
DEMONSTRATION = ("T003", "T005", "T010", "T032", "T047", "T062", "T066", "T084", "T121", "T137")
SHOWN_LABEL = re.compile(r" → `([a-z_]+)`$", re.M)


def _panel_findings(report) -> list:
    """Headline findings of a panel: the first four, the first of every other label, and every not_established one."""
    shown, labels = [], set()
    for index, record in enumerate(report["findings"]):
        label = record["evidence_status"]
        if index < 4 or label not in labels or label == "not_established":
            shown.append(record)
            labels.add(label)
    return shown


@task("T163", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_portfolio_shows_every_label_in_use",))
def portfolio_demonstration(ctx):
    reports = _reports_before(ctx, 163)
    fields = _fields(
        "A short tour of retained experiments can show every evidence label in use and what each result does not prove.",
        "Curated panels plus, for every label in use that they do not show, the first retained task holding it; each "
        "panel shows its headline findings and every not_established finding.",
        [_earlier(163)],
        "Every label used by a retained finding appears on the page, and every not_established finding of a panel "
        "is shown.",
        "Assemble a Markdown portfolio page from the selected reports and their retained figures, then read the "
        "labels and not_established claims back from the page.",
        ["label in use missing from the page", "not_established findings hidden by a finding limit",
         "curated task not retained"],
        "Present the portfolio page with the physical flat-plate/cylinder experiments once measured.",
        assumptions=["The panel selection is curated to demonstrate labels, not to rank results.",
                     "Figures are linked, not checked for legibility by this task."])
    by_id = {r["task_id"]: r for r in reports}
    in_use = sorted({f["evidence_status"] for r in reports for f in r["findings"]}, key=LABELS.index)
    chosen = [tid for tid in DEMONSTRATION if tid in by_id]
    if not chosen:
        return _no_prior(fields)
    added = []
    for label in in_use:
        if not any(f["evidence_status"] == label for tid in chosen for f in by_id[tid]["findings"]):
            extra = next((r["task_id"] for r in reports if any(f["evidence_status"] == label for f in r["findings"])), None)
            if extra and extra not in chosen:
                chosen.append(extra)
                added.append(f"{extra} ({label})")
    lines = ["# Computational experimentalist portfolio", "",
             "Each panel: the hypothesis, the headline findings with their evidence labels, every finding that is "
             "not established, the figure, and the physical validation status.", ""]
    for tid in chosen:
        report = by_id[tid]
        lines += [f"## {tid} — {report['title']}", "", _flat(report["hypothesis"]) if isinstance(report["hypothesis"], str) else "", ""]
        for record in _panel_findings(report):
            lines.append(f"- {_flat(record['claim'])}: `{_headline(record['value'])}` → `{record['evidence_status']}`")
        for artifact in report["generated_artifacts"]:
            if artifact["path"].endswith(".svg"):
                lines.append(f"\n![{tid}](../../{artifact['path']})")
                break
        lines.append(f"\n*Physical validation:* `{report['physical_validation_status']['status']}`\n")
    text = "\n".join(lines) + "\n"
    ctx.artifact_text("PORTFOLIO.md", text)
    shown = set(SHOWN_LABEL.findall(text))
    missing = [label for label in in_use if label not in shown]
    hidden = [f"{tid}: {f['claim']}" for tid in chosen for f in by_id[tid]["findings"]
              if f["evidence_status"] == "not_established" and f"- {_flat(f['claim'])}: " not in text]
    fields["numerical_result"] = (f"{len(chosen)} panels ({len([t for t in DEMONSTRATION if t in by_id])} of "
                                  f"{len(DEMONSTRATION)} curated retained, added {', '.join(added) or 'none'}); labels in "
                                  f"use: {', '.join(in_use)}; labels shown: {', '.join(sorted(shown, key=LABELS.index))}.")
    fields["uncertainty"] = "Exact reading of the rendered page; see each source report for its uncertainty."
    if missing:
        fields["unresolved_assumptions"].append("Labels in use but not shown: " + ", ".join(missing))
    findings = [_count("Portfolio page shows every evidence label in use and every not_established finding of its panels",
                       "computational_pipeline", len(chosen),
                       [_check("labels in use absent from the rendered page", len(missing)),
                        _check("not_established findings of the panels absent from the rendered page", len(hidden))],
                       "panels")]
    complete = all(tid in by_id for tid in DEMONSTRATION)
    return {"state": _state(findings, complete), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T164
def _installed_from(wheel: bytes, package_dir: Path) -> bool:
    """Whether the imported package holds exactly the wheel's files, byte for byte (bytecode caches aside).

    The marker names a wheel and its digest; only this ties that file to the
    code that ran. Anything that is not a wheel with a ``RECORD`` fails.
    """
    root = package_dir.parent
    try:
        with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
            names = archive.namelist()
            members = {name: archive.read(name) for name in names
                       if name.startswith(f"{package_dir.name}/") and not name.endswith("/")}
        installed = {path.relative_to(root).as_posix(): path.read_bytes() for path in package_dir.rglob("*")
                     if path.is_file() and "__pycache__" not in path.relative_to(package_dir).parts}
    except Exception:  # an unreadable archive or installation ties nothing to the wheel
        return False
    return any(name.endswith(".dist-info/RECORD") for name in names) and bool(members) and installed == members


CLEAN_ROOM_CLAIM = "The imported ciw package is the named wheel's files inside an isolated virtual environment"


@task("T164", changed_files=(MODULE, "scripts/reproduce_lab.py"),
      regression_tests=(f"{TESTS}::test_clean_room_marker_is_recognized",
                        f"{TESTS}::test_clean_room_needs_the_installed_package_to_be_the_named_wheel",
                        f"{TESTS}::test_clean_room_prose_does_not_carry_the_wheel_digest"))
def clean_room_reproduction(ctx):
    fields = _fields(
        "When the queue runs inside scripts/reproduce_lab.py, the imported ciw package is exactly the named wheel's "
        "files, installed in an isolated virtual environment.",
        "scripts/reproduce_lab.py: wheel -> venv -> pytest (JUnit) -> ciw lab run --all (this task) -> ciw lab "
        "verify (unless --no-compare). This task observes the step it runs in: package files against the wheel's "
        "members, interpreter prefixes and the package location.",
        ["CIW_LAB_CLEAN_ROOM marker exported by scripts/reproduce_lab.py (wheel path and digest)",
         "Files of the imported ciw package", "sys.prefix and sys.base_prefix of this interpreter"],
        "The wheel's bytes match the marker's digest, the installed package files equal the wheel's members byte "
        "for byte (bytecode caches aside), and the interpreter is a virtual environment containing the package.",
        "Read the marker, hash the named wheel, compare the imported package file by file with the wheel's members, "
        "and check the interpreter prefixes; retain the observed conditions without the wheel digest.",
        ["marker absent", "marker naming a file that is not the wheel (digest or RECORD mismatch)",
         "installed files differing from the wheel's members", "interpreter not isolated",
         "package outside the environment"],
        "Run scripts/check_lab.py under Python 3.12 on Windows and on a second Linux host and retain both gate "
        "records.",
        observation="Observes the imported package files, the interpreter prefixes and the named wheel's bytes; no "
                    "physical observation.",
        assumptions=["The comparison of this run with the retained reports (ciw lab verify) runs after the queue, "
                     "outside this report; the retained baseline is produced with --no-compare, so no comparison "
                     "backs it.",
                     "The lab tests' JUnit outcomes are recorded in each task's report, not by this task."])
    marker = os.environ.get("CIW_LAB_CLEAN_ROOM")
    if not marker:
        ctx.artifact_json("clean-room-observation.json", {"marker": "absent"})
        fields["numerical_result"] = "Queue not running inside the clean-room command."
        fields["uncertainty"] = "not applicable"
        fields["unresolved_assumptions"].insert(0, "This run was not a clean-room reproduction.")
        return {"state": "partial", "fields": fields, "findings": [
            finding(CLEAN_ROOM_CLAIM, "computational_pipeline", False, {}, expected_not_established=True)]}
    record = json.loads(marker)
    wheel = str(record.get("wheel_sha256", ""))
    import ciw as package
    import sys
    wheel_path = Path(str(record.get("wheel_path", "")))
    data = wheel_path.read_bytes() if wheel_path.is_file() else b""
    package_dir = Path(package.__file__).resolve().parent
    observed = {
        "wheel_bytes_match": wheel_path.is_file() and hashlib.sha256(data).hexdigest() == wheel,
        # The digest only names a file; the imported code must be that wheel's.
        "installed_from_wheel": _installed_from(data, package_dir),
        "isolated_interpreter": sys.prefix != sys.base_prefix,
        "package_inside_environment": package_dir.is_relative_to(Path(sys.prefix).resolve()),
    }
    failures = [name for name, ok in observed.items() if not ok]
    # The wheel digest changes with every build (file times, toolchain); compared prose and artifacts omit it.
    ctx.artifact_json("clean-room-observation.json", {"marker": "present", "conditions": observed,
                                                      "python": platform.python_version(), "ciw_version": __version__})
    fields["numerical_result"] = f"Clean-room evidence for the built wheel: {observed}."
    fields["uncertainty"] = "Exact byte and path comparisons; no numerical quantity."
    # The interpreter is observed, never taken from the marker.
    fields["provider_runtime_identity"] = {"implementation": "ciw.lab", "wheel_sha256": wheel,
                                           "python": platform.python_version(), "ciw_version": __version__}
    verified = not failures
    return {"state": "completed" if verified else "partial", "fields": fields, "findings": [
        finding(CLEAN_ROOM_CLAIM, "computational_pipeline", verified,
                {"checks": [_check("clean-room conditions failing (wheel digest, package files equal to the wheel's, "
                                   "isolated venv, installed package)",
                                   len(failures))]} if verified else {},
                expected_not_established=not verified)]}


# --------------------------------------------------------------- T165
RUNTIME_ALIASES = {"curved-surface-geodesic-sensitivity-runtime": "csg", "flat-torus-geodesic-reference": "ftr",
                   "parameterized-lyapunov-stability-runtime": "plsr", "scientific-computation-runtime": "scr",
                   "rust_probe": "rust"}
REVISION_KEYS = ("revision", "head", "commit")
TREE_KEYS = ("source_tree", "tree", "source_digest", "runtime_digest", "engine_sha256", "binary_sha256")


def runtime_identities(identity, path=()) -> list:
    """External runtimes named anywhere in a provider/runtime identity: revision- or tree-bearing entries."""
    found = []
    if isinstance(identity, dict):
        revision = next((identity[k] for k in REVISION_KEYS if isinstance(identity.get(k), str) and identity[k]), None)
        tree = next((identity[k] for k in TREE_KEYS if isinstance(identity.get(k), str) and identity[k]), None)
        if revision or tree:
            raw = identity.get("repository") or identity.get("implementation") or (path[-1] if path else "runtime")
            name = str(raw).rstrip("/").rsplit("/", 1)[-1].casefold()
            found.append({"runtime": RUNTIME_ALIASES.get(name, name), "revision": revision, "tree": tree})
        for key in sorted(identity):
            if key not in ("sources", "requirement_probes"):
                found += runtime_identities(identity[key], path + (key,))
    elif isinstance(identity, list):
        for item in identity:
            found += runtime_identities(item, path)
    return found


def _release(reports, queue) -> dict:
    states = Counter(r["state"] for r in reports)
    labels = Counter(f["evidence_status"] for r in reports for f in r["findings"])
    inventory: dict = {}
    for report in reports:
        for entry in runtime_identities(report["provider_runtime_identity"]):
            key = (entry["runtime"], entry["revision"] or "", entry["tree"] or "")
            inventory.setdefault(key, set()).add(report["task_id"])
    reported = {r["task_id"]: r for r in reports}
    # Reproducible content only: states, headline labels, claims and their labels. Values (compared within
    # tolerance) and artifact bytes (timing figures differ between runs) stay out of the digest.
    content = [[r["task_id"], r["state"], r["evidence_status"]["primary"],
                [[f["claim"], f["evidence_status"]] for f in r["findings"]]] for r in reports]
    return {"schema": "ciw.lab-release-report.v2", "ciw_version": __version__,
            "queue": {"tasks": len(queue["tasks"]),
                      "sections": [{"key": s["key"], "name": s["name"],
                                    "tasks": sum(t["section"] == s["section"] for t in queue["tasks"]),
                                    "reported": sum(t["section"] == s["section"] and t["id"] in reported
                                                    for t in queue["tasks"])} for s in queue["sections"]],
                      "not_reported": [t["id"] for t in queue["tasks"] if t["id"] not in reported]},
            "reports": len(reports), "states": dict(sorted(states.items())),
            "labels": {label: labels.get(label, 0) for label in LABELS},
            "runtimes": [{"runtime": name, "revision": revision or None, "tree": tree or None, "tasks": sorted(tasks)}
                         for (name, revision, tree), tasks in sorted(inventory.items())],
            "release_digest": "sha256:" + hashlib.sha256(json.dumps(content, separators=(",", ":"), ensure_ascii=False)
                                                         .encode("utf-8")).hexdigest(),
            "release_digest_covers": "task states, headline labels, finding claims and finding labels",
            "physical_validation": "not_established" if all(r["physical_validation_status"]["status"] == "not_established"
                                                           for r in reports) else "mixed"}


@task("T165", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_release_report_inventories_nested_runtimes",))
def release_report(ctx):
    reports = _reports_before(ctx, 165)
    fields = _fields(
        "A release report can be generated entirely from machine-readable queue metadata and retained reports.",
        "Totals over states and labels; queue coverage per section; runtime inventory = revision- or tree-bearing "
        "entries anywhere in each provider/runtime identity; release digest = SHA-256 over (task, state, headline "
        "label, claims with labels).",
        [_earlier(165), "Packaged queue definition (src/ciw/lab/queue.json)"],
        "State and label totals agree with a raw-text recount of the report files; every report whose provider probe "
        "succeeded contributes a runtime identity; the digest depends only on reproducible content.",
        "Aggregate queue coverage, states, labels, runtimes and report content into JSON and Markdown and recount "
        "states and labels from the raw report text.",
        ["runtime identities nested below the top level missed", "totals disagreeing with the report text",
         "digest over run-specific bytes (timing artifacts, wheel digests)"],
        "Sign the release digest with a project key once key custody is defined.",
        assumptions=["The release digest is unsigned.",
                     "The digest covers task states, headline labels, claims and their labels, not finding values or "
                     "artifact bytes; values are compared within tolerance by ciw lab verify.",
                     "T165-T168 run after this report and are listed as not reported."])
    if not reports:
        return _no_prior(fields)
    queue = load_queue()
    release = _release(reports, queue)
    ctx.artifact_json("release-report.json", release)
    raw = _raw_reports(ctx, 165)
    raw_states = Counter(state for _, text in raw for state in RAW_STATE.findall(text))
    raw_labels = Counter(label for _, text in raw for label in RAW_LABEL.findall(text))
    state_gap = sum(abs(release["states"].get(s, 0) - raw_states.get(s, 0)) for s in set(release["states"]) | set(raw_states))
    label_gap = sum(abs(release["labels"].get(l, 0) - raw_labels.get(l, 0)) for l in set(release["labels"]) | set(raw_labels))
    unidentified = []
    for report in reports:
        identity = report["provider_runtime_identity"] if isinstance(report["provider_runtime_identity"], dict) else {}
        probes = identity.get("requirement_probes") if isinstance(identity.get("requirement_probes"), dict) else {}
        if any(k.startswith("provider:") and v is True for k, v in probes.items()) and not runtime_identities(identity):
            unidentified.append(report["task_id"])
    lines = ["# Lab release report", "", f"- CIW version: {__version__}",
             f"- Queue: {release['queue']['tasks']} tasks; {len(reports)} reported; not reported: "
             f"{', '.join(release['queue']['not_reported']) or 'none'}",
             f"- Release digest: `{release['release_digest']}` (unsigned; covers {release['release_digest_covers']})",
             f"- Physical validation: `{release['physical_validation']}`", "",
             _row("Section", "Tasks", "Reported"), _row("---", "---", "---")] + [
             _row(_cell(s["name"]), str(s["tasks"]), str(s["reported"])) for s in release["queue"]["sections"]] + [
             "", _row("State", "Tasks"), _row("---", "---")] + [_row(k, str(v)) for k, v in release["states"].items()] + [
             "", _row("Evidence label", "Findings"), _row("---", "---")] + [
             _row(f"`{k}`", str(v)) for k, v in release["labels"].items()] + [
             "", _row("Runtime", "Revision", "Tree or digest", "Tasks"), _row("---", "---", "---", "---")] + [
             _row(r["runtime"], f"`{r['revision'] or '-'}`", f"`{r['tree'] or '-'}`", ", ".join(r["tasks"]))
             for r in release["runtimes"]]
    ctx.artifact_text("RELEASE.md", "\n".join(lines) + "\n")
    names = sorted({r["runtime"] for r in release["runtimes"]})
    fields["numerical_result"] = (f"{len(reports)} of {release['queue']['tasks']} queue tasks reported; states "
                                  f"{release['states']}; labels {dict((k, v) for k, v in release['labels'].items() if v)}; "
                                  f"runtimes {', '.join(names) or 'none'}.")
    fields["uncertainty"] = "Exact counts."
    findings = [
        _count("Release state and label totals match a raw-text recount of the retained report files", "provenance",
               len(reports), [_check("absolute state-count differences from the raw-text recount", state_gap),
                              _check("absolute label-count differences from the raw-text recount", label_gap)], "reports"),
        _count("Every report whose provider probe succeeded contributes a runtime identity to the release inventory",
               "provenance", len(names),
               [_check("reports with a successful provider probe and no runtime identity", len(unidentified))], "runtimes"),
        finding("The release digest is signed by a project key", "provenance", None, {}, expected_not_established=True),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T166
def _ledger_problems(text: str, rows: list) -> list:
    """Ledger lines that do not restate their assumption and source tasks."""
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    problems = [] if len(lines) == len(rows) else [f"{len(lines)} ledger lines for {len(rows)} assumptions"]
    for index, (line, row) in enumerate(zip(lines, rows)):
        assumption, _, tasks = line[2:].rpartition(" (")
        if assumption != _flat(row["assumption"]) or tasks != ", ".join(row["tasks"]) + ")":
            problems.append(f"line {index + 1}")
    return problems


@task("T166", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_unresolved_assumption_ledger",))
def unresolved_assumptions(ctx):
    reports = _reports_before(ctx, 166)
    fields = _fields(
        "Every unresolved assumption stated by any task can be listed in one ledger with its source task.",
        "Ledger = union over reports of unresolved_assumptions, deduplicated by exact wording, with task references.",
        [_earlier(166) + " (this section's T155-T165 included)"],
        "Every stated assumption appears once per exact wording with every task stating it; citations agree with a "
        "raw-text recount of the report files.",
        "Collect, deduplicate and count assumptions; write JSON and Markdown ledgers; parse the Markdown back and "
        "recount assumption items from the raw report text.",
        ["assumptions dropped by the traversal (raw-text recount)", "ledger lines not restating their assumption",
         "reports stating no assumption"],
        "Attach each assumption to the experiment that would resolve it and track closure.")
    if not reports:
        return _no_prior(fields)
    ledger: dict = {}
    for report in reports:
        items = report["unresolved_assumptions"]
        for item in (items if isinstance(items, list) else [items]):
            ledger.setdefault(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False), []).append(report["task_id"])
    rows = [{"assumption": k, "tasks": v} for k, v in sorted(ledger.items())]
    ctx.artifact_json("unresolved-assumptions.json", rows)
    text = "# Unresolved assumptions\n\n" + "\n".join(f"- {_flat(row['assumption'])} ({', '.join(row['tasks'])})"
                                                      for row in rows) + "\n"
    ctx.artifact_text("UNRESOLVED_ASSUMPTIONS.md", text)
    problems = _ledger_problems(text, rows)
    citations = sum(len(row["tasks"]) for row in rows)
    raw_citations = sum(_raw_assumption_count(raw) for _, raw in _raw_reports(ctx, 166))
    silent = [r["task_id"] for r in reports if not r["unresolved_assumptions"]]
    fields["numerical_result"] = (f"{len(rows)} distinct unresolved assumptions ({citations} statements) from "
                                  f"{len(reports)} reports; {len(silent)} reports state none.")
    fields["uncertainty"] = "Exact counts; completeness depends on what each task states."
    fields["unresolved_assumptions"] = (
        ([f"Reports stating no unresolved assumption: {', '.join(silent)}"] if silent else [])
        + ["T167 and T168 run after this ledger; their assumptions are in their own reports.",
           "Assumptions are deduplicated by exact wording; one assumption worded differently is listed twice."])
    findings = [_count("Unresolved-assumption ledger lists every assumption stated by the retained reports",
                       "provenance", len(rows),
                       [_check("ledger lines that do not restate their assumption and tasks", len(problems)),
                        _check("assumption statements minus a raw-text recount of the report files",
                               citations - raw_citations)], "assumptions")]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T167
@task("T167", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_unmeasured_ledger",))
def unmeasured(ctx):
    reports = _reports_before(ctx, 167)
    fields = _fields(
        "Everything that remains physically unmeasured or unrun is enumerable from the retained reports.",
        "Unmeasured = physical/authority-domain findings not established; blocked, deferred and partial tasks with "
        "their unrun parts; not_established findings of tasks whose hardware probe failed.",
        [_earlier(167)],
        "The ledger's counts of not-established physical/authority claims and of unfinished tasks agree with a "
        "raw-text recount of the report files.",
        "Collect those findings, task states and hardware probe outcomes into JSON and Markdown and recount them "
        "from the raw report text.",
        ["claims dropped by the traversal (raw-text recount)", "partial tasks omitted",
         "hardware claims filed under a computational domain"],
        "Execute the manufacturing measurement protocols (T126-T128) on hardware and retain raw data.",
        assumptions=["T168 runs after this ledger.",
                     "Hardware needs are inferred from failed hardware probes; a claim that needs hardware in a task "
                     "that probed none is listed only when its domain is physical or authority."])
    if not reports:
        return _no_prior(fields)
    claims = [{"task_id": r["task_id"], "claim": f["claim"], "domain": f["domain"]}
              for r in reports for f in r["findings"]
              if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and f["evidence_status"] == "not_established"]
    unfinished = [{"task_id": r["task_id"], "state": r["state"], "reason": _words(r["experiment"], 300),
                   "unrun": [_flat(a) for a in (r["unresolved_assumptions"] if isinstance(r["unresolved_assumptions"], list)
                                                 else [r["unresolved_assumptions"]])]}
                  for r in reports if r["state"] in ("blocked", "deferred", "partial")]
    hardware = []
    for r in reports:
        identity = r["provider_runtime_identity"] if isinstance(r["provider_runtime_identity"], dict) else {}
        failed = sorted(k for k, v in (identity.get("requirement_probes") or {}).items()
                        if k.startswith("hardware:") and v is False)
        if failed:
            hardware += [{"task_id": r["task_id"], "probes": failed, "claim": f["claim"], "domain": f["domain"]}
                         for f in r["findings"] if f["evidence_status"] == "not_established"]
    measured = [r["task_id"] for r in reports for f in r["findings"] if f["evidence_status"] == "hardware_measured"]
    document = {"not_established_claims": claims, "unfinished_tasks": unfinished,
                "hardware_unavailable_findings": hardware, "hardware_measured_findings": measured}
    ctx.artifact_json("unmeasured.json", document)
    lines = ["# What remains unmeasured", "", f"Hardware-measured findings in this run: {len(measured)}.", "",
             "## Physical and authority claims not established", ""] + [
        f"- {c['task_id']} [{c['domain']}]: {_flat(c['claim'])}" for c in claims] + [
        "", "## Findings of tasks whose hardware probe failed", ""] + [
        f"- {h['task_id']} [{h['domain']}; {', '.join(h['probes'])} unavailable]: {_flat(h['claim'])}" for h in hardware] + [
        "", "## Blocked, deferred and partial tasks", ""]
    for item in unfinished:
        lines.append(f"- {item['task_id']} ({item['state']}): {item['reason']}")
        lines += [f"  - Unrun or unresolved: {part}" for part in item["unrun"]]
    ctx.artifact_text("UNMEASURED.md", "\n".join(lines) + "\n")
    raw = _raw_reports(ctx, 167)
    raw_claims = sum(domain in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and label == "not_established"
                     for _, text in raw for domain, label in RAW_DOMAIN_LABEL.findall(text))
    raw_unfinished = sum(state in ("blocked", "deferred", "partial") for _, text in raw for state in RAW_STATE.findall(text))
    states = Counter(item["state"] for item in unfinished)
    fields["numerical_result"] = (f"{len(claims)} physical/authority claims not established; {len(unfinished)} unfinished "
                                  f"tasks {dict(sorted(states.items()))}; {len(hardware)} not_established findings in "
                                  f"tasks whose hardware probe failed; {len(measured)} hardware-measured findings.")
    fields["uncertainty"] = "Exact counts."
    findings = [
        _count("Unmeasured ledger lists every not-established physical or authority claim", "provenance", len(claims),
               [_check("listed claims minus a raw-text recount of not-established physical/authority findings",
                       len(claims) - raw_claims)], "claims"),
        _count("Unmeasured ledger lists every blocked, deferred or partial task", "provenance", len(unfinished),
               [_check("listed tasks minus a raw-text recount of blocked, deferred and partial states",
                       len(unfinished) - raw_unfinished)], "tasks"),
        finding("Physical validity of the lab's computational results", "physical", None, {}),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T168
def _index_source(relative: str, text: str, index: dict) -> None:
    """Add the test functions of one test module: node id -> (name, source of the test and the module names it uses)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)

    def segment(node) -> str:
        # Whole source lines of a node (ast.get_source_segment re-splits the file on every call).
        return "".join(lines[node.lineno - 1: node.end_lineno])

    top = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Name):
                    top[target.id] = node

    def closure(node) -> str:
        # The test, its decorators, and the module-level helpers, fixtures and constants it names (two levels).
        segments, seen, frontier = [], set(), [node]
        for _ in range(3):
            following = []
            for current in frontier:
                segments.append(segment(current))
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    segments += [segment(d) for d in current.decorator_list]
                    names = {argument.arg for argument in current.args.args}
                else:
                    names = set()
                names |= {n.id for n in ast.walk(current) if isinstance(n, ast.Name)}
                for name in sorted(names):
                    if name in top and name not in seen:
                        seen.add(name)
                        following.append(top[name])
            frontier = following
        return "\n".join(segments)

    def collect(prefix, body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                index[f"{prefix}::{node.name}"] = (node.name, closure(node))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                collect(f"{prefix}::{node.name}", node.body)

    collect(relative, tree.body)


def _test_index(tests_dir: Path) -> dict:
    index: dict = {}
    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            _index_source(path.relative_to(tests_dir.parent).as_posix(), path.read_text(encoding="utf-8"), index)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue  # pytest cannot collect it either; its node ids stay dangling
    return index


def _test_names(tests_dir: Path) -> set:
    """pytest node ids of the test functions and test-class methods (sync or async) under ``tests_dir``."""
    return set(_test_index(tests_dir))


LABEL_WORDS = re.compile(r"evidence_status|" + "|".join(LABELS))


def _tie(task_id: str, function: str, node: str, index: dict) -> tuple:
    """(names the task, asserts a label) for one registered node, from source text.

    A node names its task through the task id in its node id, test name or
    source (helpers, fixtures and constants it uses included) or through the
    task function's name; it asserts a label when that source mentions
    ``evidence_status`` or a label.
    """
    entry = index.get(node.split("[")[0])
    if entry is None:
        return False, False
    name, source = entry
    names = (task_id in node or task_id.lower() in name or task_id in source
             or (function not in ("", "<lambda>") and re.search(rf"\b{re.escape(function)}\b", source) is not None))
    return names, LABEL_WORDS.search(source) is not None


# Probe cases for the tie analysis: (node id, expected (names task, asserts label)).
TIE_PROBE = '''
SECTION = ("T901",)
def _label(report):
    return report["evidence_status"]["primary"]
def test_literal(run):
    assert run("T901")["findings"][0]["evidence_status"] == "numerically_verified"
def test_constant(run):
    for task_id in SECTION:
        assert _label(run(task_id)) == "analytic"
def test_t901_named(run):
    assert _label(run.last)
def test_parametrized(run, task_id):
    assert run(task_id)["state"] == "completed" and _label(run(task_id))
def test_unrelated():
    assert 1 + 1 == 2
def test_values_only(run):
    assert run("T901")["findings"][0]["value"] == 1
'''
TIE_EXPECTED = {"test_literal": (True, True), "test_constant": (True, True), "test_t901_named": (True, True),
                "test_parametrized[T901]": (True, True), "test_unrelated": (False, False),
                "test_values_only": (True, False)}


def _tie_probe_errors() -> list:
    index: dict = {}
    _index_source("tests/test_probe.py", TIE_PROBE, index)
    return [node for node, expected in TIE_EXPECTED.items()
            if _tie("T901", "", f"tests/test_probe.py::{node}", index) != expected]


def _junit_outcomes(report) -> dict:
    """Outcome of each registered pytest node as the report recorded it from the JUnit record."""
    outcomes = {}
    for entry in report["tests_passed"]:
        if entry.startswith("pytest: "):
            outcomes[entry[8:]] = "passed"
    for entry in report.get("tests_failed", []):
        if entry.startswith("pytest: "):
            outcomes[entry[8:]] = "failed"
    for entry in report["tests_skipped"]:
        if entry.startswith("pytest: "):
            node, _, reason = entry[8:].partition(" (")
            outcomes[node] = "skipped" if reason.startswith("skipped") else "not run"
    return outcomes


@task("T168", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_regression_coverage_is_checked",))
def permanent_regression_tests(ctx):
    from .runner import repository_path
    reports = _reports_before(ctx, 168)
    fields = _fields(
        "Every completed or partial experiment is backed by at least one registered pytest regression test that "
        "runs the task and asserts its evidence labels, and those tests pass.",
        "Per task: registered node ids resolved against test functions (AST); a task is tied when one of its nodes "
        "names the task (id in node id, test name or source, or the task function) and asserts an evidence label; "
        "JUnit outcomes per node as recorded in the reports.",
        [_earlier(168) + " (this section's T155-T167 included)", "Registered regression node ids of T001-T168",
         "tests/ of the repository (CIW_LAB_REPOSITORY_ROOT in the clean room)"],
        "No completed or partial task lacks a registered test; every registered node resolves; no registered node "
        "failed in the JUnit record.",
        "Resolve every registered node id against the test functions under tests/, analyze each node's source for its "
        "task and a label assertion (the analysis is checked on probe cases first), and fold the JUnit outcomes the "
        "reports recorded.",
        ["registered node ids that do not resolve", "tests registered for a task they never run",
         "tests asserting values only", "registered tests failing or not run"],
        "Mark each regression test with the task identities it guards (a pytest marker), so the tie between tests "
        "and tasks is declared rather than inferred from source text.",
        assumptions=["The tie analysis reads source text: a task id computed at run time (for example from a range) "
                     "is not seen, and mentioning a label is taken as asserting it.",
                     "The tolerance-aware comparison with retained reports (ciw lab verify, run by scripts/check_lab.py "
                     "in CI) is outside this report."])
    tests_dir = repository_path("tests")  # never the current directory: it may hold another project's tests
    if not reports:
        return _no_prior(fields)
    if tests_dir is None or not tests_dir.is_dir():
        fields["numerical_result"] = "Registered regression node ids were not resolved."
        fields["uncertainty"] = "not applicable"
        fields["unresolved_assumptions"].insert(0, "No repository tests/ directory is available to this installation "
                                                   "(set CIW_LAB_REPOSITORY_ROOT to a checkout).")
        return {"state": "partial", "fields": fields, "findings": [
            finding("Registered regression node ids that do not resolve to a test function", "computational_pipeline",
                    None, {}, expected_not_established=True)]}
    index = _test_index(tests_dir)
    implementations, _ = load_implementations()
    states = {r["task_id"]: r["state"] for r in reports}
    outcomes = {r["task_id"]: _junit_outcomes(r) for r in reports}
    rows = []
    for task_id in sorted(set(states) | ({"T168"} & set(implementations))):
        implementation = implementations.get(task_id)
        nodes = list(implementation.regression_tests) if implementation else []
        function = getattr(getattr(implementation, "run", None), "__name__", "")
        ties = {node: _tie(task_id, function, node, index) for node in nodes}
        rows.append({"task_id": task_id, "state": states.get(task_id, "running"), "regression_tests": nodes,
                     "missing": [n for n in nodes if n.split("[")[0] not in index],
                     "tied": [n for n, (named, labelled) in ties.items() if named and labelled],
                     "junit": {n: outcomes.get(task_id, {}).get(n, "not recorded") for n in nodes}})
    uncovered = [r["task_id"] for r in rows if r["state"] in ("completed", "partial") and not r["regression_tests"]]
    dangling = [f"{r['task_id']}: {n}" for r in rows for n in r["missing"]]
    untied = [r["task_id"] for r in rows if r["regression_tests"] and not r["tied"]]
    probe_errors = _tie_probe_errors()
    recorded = Counter(outcome for r in rows for outcome in r["junit"].values())
    failed = [f"{r['task_id']}: {n}" for r in rows for n, o in r["junit"].items() if o == "failed"]
    ctx.artifact_json("regression-coverage.json", rows)
    findings = [
        _count("Completed or partial tasks lacking a regression test", "computational_pipeline", len(uncovered),
               [_check("completed or partial tasks with no registered regression test", len(uncovered))], "tasks"),
        _count("Registered regression node ids that do not resolve to a test function", "computational_pipeline",
               len(dangling), [_check("dangling node ids", len(dangling))], "node ids"),
        _count("Tasks without a registered test that both names the task and asserts an evidence label",
               "computational_pipeline", len(untied),
               [_check("tie-analysis probe cases classified wrongly", len(probe_errors))], "tasks",
               basis="integer count from a static source analysis; see the unresolved assumptions for what it cannot see"),
    ]
    junit_supplied = recorded["passed"] + recorded["failed"] + recorded["skipped"] > 0
    if junit_supplied:
        findings.append(_count("Registered regression tests failing in the JUnit record of this run",
                               "computational_pipeline", len(failed),
                               [_check("registered nodes recorded as failed", len(failed))], "node ids"))
    else:
        findings.append(finding("Registered regression tests failing in the JUnit record of this run",
                                "computational_pipeline", None, {}, expected_not_established=True))
        fields["unresolved_assumptions"].insert(0, "No JUnit record was supplied to this run (ciw lab run --junit); "
                                                   "registered tests' outcomes are unknown.")
    if dangling:
        fields["unresolved_assumptions"].insert(0, "Dangling regression node ids: " + "; ".join(dangling[:20]))
    if untied:
        fields["unresolved_assumptions"].insert(0, "Tasks without a registered test that names the task and asserts a "
                                                   "label: " + ", ".join(untied))
    not_run = recorded["skipped"] + recorded["not run"]
    fields["numerical_result"] = (
        f"{len(rows)} tasks, {sum(len(r['regression_tests']) for r in rows)} registered nodes; {len(uncovered)} "
        f"completed/partial tasks without regression tests; {len(dangling)} dangling node ids; {len(untied)} tasks "
        f"without a tied test; JUnit: {recorded['passed']} passed, {recorded['failed']} failed, {not_run} skipped or "
        "not run.")
    fields["uncertainty"] = "Exact counts; ties are inferred statically from source text."
    complete = junit_supplied and not untied and not not_run
    return {"state": _state(findings, complete), "fields": fields, "findings": findings}
