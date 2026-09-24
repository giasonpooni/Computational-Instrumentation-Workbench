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
are not claims the workbench can make, and the prose the lab cannot write
(the argument in the authors' words, the discussion against the literature,
conclusions) stays marked outstanding.

Pages meant for readers outside the lab (T160-T163) define every label, state
that ``independently_verified`` is agreement between implementations of
different origin and not verification by another party, list the boundary of
``ciw.lab.evidence.BOUNDARY``, and show each finding's basis beside its label.
No task here reads retained hardware runs: they are aggregated outside the
queue by ``ciw lab unmeasured``.
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
                       INDEPENDENT_ORIGINS, LABELS, MAX_THRESHOLD, PHYSICAL_DOMAINS, BOUNDARY, EvidenceRefusal,
                       ORIGINS, describe_basis, finding, finding_origin, supported_label, holds as compare)
from ..core.identities import content_identity
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


# A finding's basis and its keys (indent 4) in the retained layout; the raw basis components they declare.
RAW_BASIS = re.compile(r'^   "basis": (\{\}|\{\n.*?\n   \})', re.M | re.S)
RAW_BASIS_KEY = re.compile(r'^    "([a-z_]+)": (.*)$', re.M)
RAW_ORIGIN_LINE = re.compile(r'^   "origin": (\[\]|\[\n(?:    "[a-z_]+",?\n)*   \])', re.M)
RAW_KEY_COMPONENT = {"acquisition": "acquisition", "derivation": "derivation", "independent_check": "independent_check",
                     "generator": "synthetic_inputs"}


def _raw_block_components(block: str) -> list:
    """Basis components of one raw finding block: its ``origin`` list, or, for a finding retained before basis
    components were recorded (no ``origin`` key; rule 11 of the specification), the components its raw basis
    keys declare (a nonempty check list, a generator, an executed provider, a derivation, an acquisition, an
    independent check)."""
    stated = RAW_ORIGIN_LINE.search(block)
    if stated:
        return sorted(re.findall(r'"([a-z_]+)"', stated.group(1)))
    basis = RAW_BASIS.search(block)
    if not basis:
        return []
    body, found = basis.group(1), []
    for key, rest in RAW_BASIS_KEY.findall(body):
        if key in RAW_KEY_COMPONENT and not rest.startswith("null"):
            found.append(RAW_KEY_COMPONENT[key])
        elif key == "checks" and rest == "[":
            found.append("reference_checks")
        elif key == "provider" and rest == "{":
            provider = re.search(r'^    "provider": \{\n(.*?)\n    \}', body, re.M | re.S)
            if provider and re.search(r'^     "executed": true,?$', provider.group(1), re.M):
                found.append("provider")
    return sorted(found)


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


PROBE_ACQUISITION = {"device": "d", "raw_sha256": "0" * 64, "acquired_at": "t", "calibration": "not_applied"}
PROBE_CHECK = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
# Rule 10 of docs/lab/SPECIFICATIONS.md on hand-classified claims: (claim, asserts an authority outcome). The
# first is the acceptance statement T141 filed in a computational domain (its computational-domain loophole).
RULE10_PROBES = (
    ("Coupon lot accepted for production", True),
    ("Coupon lot rejected for production", True),
    ("Coupon lot accepted for production; it does not need rework", True),
    ("The lot that never failed is accepted for production", True),
    ("The press is safe to operate", True),
    ("The monitor authorizes actuation of the axis", True),
    ("The workbench is ready for industrial deployment", True),
    ("The mesh is certified for production metrology", True),
    ("There is customer demand for curvature-aware path checking", True),
    ("The lab API cannot mark a lot accepted for production", False),
    ("The simulation records production acceptance as not performed", False),
    ("The monitor does not authorize actuation", False),
    ("Accepted steps keep the local error below the tolerance", False),
    ("A safe step size for RK4 on this surface", False),
)
# A paraphrase of customer demand that no outcome phrase names: rule 10 says the screen does not catch every
# paraphrase, and T155 records whether this one passes it.
RULE10_PARAPHRASE = "Manufacturers need curvature-aware placement checks"
# The claim states what the probe covers: its hand-classified statements, not every paraphrase (the paraphrase
# result is recorded among T155's unresolved assumptions).
RULE10_AUTHORITY_PROBES = sum(asserts for _, asserts in RULE10_PROBES)
RULE10_CLAIM = (f"The authority-wording screen refuses each of the {RULE10_AUTHORITY_PROBES} hand-classified authority "
                f"statements of its probe (T141's acceptance statement first) in the {len(set(DOMAINS) - SPEC_AUTHORITY)} "
                "computational and physical domains, leaves them not_established in the authority domains, and gives "
                f"the {len(RULE10_PROBES) - RULE10_AUTHORITY_PROBES} declined or ordinary statements their rules 1-5 "
                "labels (rule 10)")


def _probe_basis(domain: str) -> dict:
    """A basis that would establish the claim if the domain allowed it: checks, plus an acquisition where legal."""
    if domain in SPEC_PHYSICAL | SPEC_AUTHORITY:
        return {"acquisition": PROBE_ACQUISITION, "checks": [PROBE_CHECK]}
    return {"checks": [PROBE_CHECK]}


def rule10_probe() -> dict:
    """Compare ``evidence.finding`` with a hand-written rule 10 oracle on every probe claim in every domain.

    The oracle refuses a claim that asserts an authority outcome in a computational or physical domain and
    otherwise gives the rules 1-5 label of :func:`_reference_label`.
    """
    violations, cases, outcomes = [], 0, Counter()
    for claim, asserts in RULE10_PROBES:
        for domain in sorted(DOMAINS):
            cases += 1
            basis = _probe_basis(domain)
            try:
                label = finding(claim, domain, None, basis)["evidence_status"]
            except EvidenceRefusal:
                label = "refused"
            expected = ("refused" if asserts and domain not in SPEC_AUTHORITY
                        else _reference_label(basis, domain)[0])
            outcomes[label] += 1
            if label != expected:
                violations.append(f"{domain}: {claim!r} -> {label}, rule 10 gives {expected}")
    try:
        finding(RULE10_PARAPHRASE, "computational_pipeline", None, {"checks": [PROBE_CHECK]})
        paraphrase = "passes the screen"
    except EvidenceRefusal:
        paraphrase = "refused"
    return {"cases": cases, "claims": len(RULE10_PROBES), "violation_count": len(violations),
            "violations": violations[:50], "outcomes": dict(sorted(outcomes.items())),
            "paraphrase": {"claim": RULE10_PARAPHRASE, "outcome": paraphrase}}


# What the evidence-label section of the specification must state beyond rules 1-5: the authority-wording
# screen with T141's loophole, basis components, and workspace classification with T100's fabricated bundle.
RULE_ANCHORS = ("screen_authority_claim", "T141", "basis_origin", "T100")
RULES_CLAIM = ("The specification states the evidence rules as a numbered list that includes the authority-wording "
               "screen (T141's loophole), basis components and workspace classification (T100)")


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
    section) is not a unit of computational coverage. ``rules`` holds the
    numbered rules of the evidence-label section, one flattened line each.
    """
    from .runner import repository_path
    page = repository_path(*SPECIFICATIONS.split("/"))
    if page is None or not page.is_file():
        return None
    text = page.read_text(encoding="utf-8")
    units, missing, pages, rules = {}, [], [], []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = section.partition("\n")
        if title.strip() == FAMILY_INDEX:
            pages = sorted(set(re.findall(r"\]\(([A-Z_]+\.md)\)", body)))
        elif _task_ids(body):
            units[f"SPECIFICATIONS.md: {title.strip()}"] = _task_ids(body)
        if title.strip() == "Evidence labels":
            rules = _spec_rules(body)
    for name in pages:
        path = page.parent / name
        if not path.is_file():
            missing.append(name)
            continue
        family = path.read_text(encoding="utf-8")
        units[name] = _task_ids(family, family.partition("\n")[0])
    return {"units": units, "missing_pages": missing, "rules": rules}


def _rule_numbers_problem(rules: list) -> int:
    """1 when the stated rules are not numbered 1, 2, ... in order, else 0."""
    numbers = [int(re.match(r"(\d+)\. ", rule).group(1)) for rule in rules]
    return int(numbers != list(range(1, len(numbers) + 1)))


def _counterexample_statements(reports, ids) -> list:
    """Counterexample statements the retained findings of tasks ``ids`` refute, in report order."""
    return [{"task_id": r["task_id"], "statement": f["counterexample"]["statement"], "claim": f["claim"],
             "evidence_status": f["evidence_status"]}
            for r in reports if r["task_id"] in ids for f in r["findings"] if f.get("counterexample")]


@task("T155", changed_files=(MODULE, SPECIFICATIONS),
      regression_tests=(f"{TESTS}::test_label_function_matches_the_reference_oracle_exhaustively",
                        f"{TESTS}::test_formal_specifications_cover_the_queue",
                        f"{TESTS}::test_rule_10_screen_closes_the_computational_domain_loophole",
                        f"{TESTS}::test_specification_units_list_the_counterexamples_of_their_tasks",
                        f"{TESTS}::test_next_steps_name_open_work_rather_than_work_done_elsewhere"))
def formal_specifications(ctx):
    reports = _reports_before(ctx, 155)
    invariants = label_invariants()
    screen = rule10_probe()
    grammar = " x ".join(str(n) for n in invariants["grammar"].values())
    fields = _fields(
        "The label function agrees with rules 1-5 of the specification on every basis of a finite grammar, the "
        "authority-wording screen closes T141's computational-domain loophole as rule 10 states, the specification "
        "states its evidence rules including that boundary, every computational queue task is named by a "
        "specification document, and every specification document is exercised by retained established findings.",
        "Label function L(basis, domain) from ciw.lab.evidence against a reference oracle restating rules 1-5 of "
        "docs/lab/SPECIFICATIONS.md; evidence.finding against a rule 10 oracle (refuse an asserted authority outcome "
        "in a computational or physical domain, else the rules 1-5 label) on hand-classified claims; specification "
        "units = its sections plus the family pages it lists as specifications of record, each with the "
        "counterexample statements that its named tasks' retained findings refute.",
        [_earlier(155), "docs/lab/SPECIFICATIONS.md and the family pages it lists",
         "Packaged queue definition (tasks T001-T154)",
         f"{len(RULE10_PROBES)} hand-classified authority and ordinary claims, T141's acceptance statement first"],
        "L equals the oracle on every case (refusals included) and every oracle rule branch is exercised; "
        "evidence.finding equals the rule 10 oracle on every probe case; the evidence-label section numbers its "
        "rules from 1 and names the screen, T141, basis components and T100; every computational task is named; "
        "established-finding counts agree with a raw-text recount.",
        f"Enumerate {grammar} bases x {len(DOMAINS)} domains = {invariants['cases']} cases (failing, reversed, "
        "same-origin, cross-implementation, unknown-family and embedded-ciw independent checks included) and compare "
        f"L with the oracle; file each of {len(RULE10_PROBES)} probe claims in each of the {len(DOMAINS)} domains "
        f"({screen['cases']} cases) with a basis that would establish it where the domain allows; read the "
        "evidence-label rules; map each specification unit to the tasks it names, count their established findings "
        "and collect their counterexample statements.",
        ["label function departing from a written rule (precedence, origin rule, physical gate, authority)",
         "grammar that leaves a rule branch unexercised",
         "authority statement filed in a computational or physical domain and labelled by its checks (T141)",
         "specification that omits the authority boundary or numbers its rules out of order",
         "computational task named by no specification", "family page listed but absent",
         "established-finding traversal disagreeing with the report text",
         "unit counted as covered while its tasks refute a general statement it may restate"],
        "Prove the label rules off the finite grammar, for example with property-based tests over unbounded bases or "
        "in Lean (the grammar itself is already enumerated exhaustively here), and have a reviewer confirm that no "
        "specification unit restates a counterexample statement listed for it in specification-coverage.json.",
        assumptions=[
            "Only the families on SPECIFICATIONS.md are stated in formal notation; the family pages it lists specify "
            "the other families as method descriptions (models, references and decision rules), not in one uniform "
            "notation.",
            "T155 does not recompute retained values from the specification formulas; each task's own reference "
            "checks compare its results with its model.",
            "The oracle restates rules 1-5 by hand: agreement shows that the implementation and the written rules "
            "coincide on the grammar, not that the rules are the right ones or that they hold off the grammar.",
            "The rule 10 probe covers hand-classified claims only; rules 6-9, 11 and 12 are enforced by the validator, "
            "the report builder, the runner and the workspace classifier and tested in tests/test_lab_core.py and "
            "tests/test_lab_bridge.py, not by T155.",
            "An established finding exercises a unit; it does not show that the unit is right. Whether a unit "
            "restates a general statement that a counterexample of its tasks refutes is a review question: the "
            "statements differ in notation from the units' formulas, so text matching cannot decide it."])
    if screen["paraphrase"]["outcome"] == "passes the screen":
        fields["unresolved_assumptions"].append(
            f"The screen matches phrases, not paraphrases: {RULE10_PARAPHRASE!r} passes it in a computational domain "
            "(rule10-probe.json), so assigning a free-text claim to a domain remains a review question.")
    ctx.artifact_json("label-invariants.json", invariants)
    ctx.artifact_json("rule10-probe.json", screen)
    findings = [finding(
        "The evidence-label function agrees with the reference oracle for rules 1-5 on the exhaustive basis grammar",
        "mathematical", invariants["violation_count"],
        {"derivation": "docs/lab/SPECIFICATIONS.md#evidence-labels",
         "checks": [_check("cases where L differs from the rules 1-5 oracle", invariants["violation_count"]),
                    _check("oracle rule branches not exercised by the grammar", len(invariants["unexercised_branches"]))]},
        unit="cases", uncertainty=_exact("exhaustive enumeration of a finite grammar"), tolerance=ZERO),
        finding(RULE10_CLAIM, "computational_pipeline",
                {"cases": screen["cases"], "violations": screen["violation_count"],
                 "refused": screen["outcomes"].get("refused", 0)},
                {"derivation": "docs/lab/SPECIFICATIONS.md#evidence-labels (rules 1 and 10)",
                 "checks": [_check("probe cases where evidence.finding departs from the rule 10 oracle",
                                   screen["violation_count"])]},
                unit="cases", uncertainty=_exact("every probe claim in every domain; no sampling"), tolerance=ZERO)]
    documents = specification_documents()
    queue_ids = [t["id"] for t in load_queue()["tasks"] if t["number"] < FIRST_OWN]
    complete = documents is not None and bool(reports)
    if documents is None:
        fields["unresolved_assumptions"].append(
            "docs/lab is not reachable from this installation (set CIW_LAB_REPOSITORY_ROOT to a checkout); "
            "specification coverage was not evaluated.")
        findings += [finding(RULES_CLAIM, "provenance", None, {}, expected_not_established=True),
                     finding("Every computational queue task is named by a specification document", "provenance",
                             None, {}, expected_not_established=True),
                     finding("Specification documents are exercised by retained established findings",
                             "computational_pipeline", None, {}, expected_not_established=True)]
        fields["numerical_result"] = (f"{invariants['cases']} label cases, {invariants['violation_count']} "
                                      f"disagreements with the oracle; {screen['cases']} rule 10 probe cases, "
                                      f"{screen['violation_count']} disagreements; specification coverage not evaluated.")
        fields["uncertainty"] = "Exact enumeration."
        return {"state": _state(findings, complete), "fields": fields, "findings": findings}
    rules = documents["rules"]
    body = " ".join(rules)
    anchors = [anchor for anchor in RULE_ANCHORS if not re.search(rf"\b{re.escape(anchor)}\b", body)]
    findings.append(_count(RULES_CLAIM, "provenance", len(rules),
                           [_check("evidence rules not numbered 1, 2, ... in order", _rule_numbers_problem(rules)),
                            _check("rules 1-5 absent from the section", max(0, 5 - len(rules))),
                            _check("screen, T141, basis-component and T100 anchors the rules do not name",
                                   len(anchors))], "rules"))
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
    refuted = {unit: _counterexample_statements(reports, ids) for unit, ids in units.items()}
    uncovered = [unit for unit, row in coverage.items() if not any(row.values())]
    ctx.artifact_json("specification-coverage.json", {
        "units": {unit: {"established_findings": coverage[unit], "counterexample_statements": refuted[unit]}
                  for unit in coverage},
        "note": ("counterexample_statements are the general statements that retained findings of a unit's tasks "
                 "refute; the unit must not restate them without the qualification their witnesses show, and "
                 "whether it does is a review question"),
        "unnamed_tasks": unnamed, "missing_pages": documents["missing_pages"], "uncovered_units": uncovered,
        "recount_disagreements": disagreements})
    with_counterexamples = [unit for unit, rows in refuted.items() if rows]
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
    findings.append(finding(
        "No specification unit restates a general statement that a retained counterexample of its named tasks refutes",
        "provenance", None,
        {"notes": "Statements are listed per unit in specification-coverage.json for review; their notation differs "
                  "from the units' formulas, so no text comparison decides whether a unit restates one."},
        expected_not_established=True))
    if unnamed:
        fields["unresolved_assumptions"].append("Computational tasks named by no specification: " + ", ".join(unnamed))
    if uncovered:
        fields["unresolved_assumptions"].append("Specification units without established findings: " + "; ".join(uncovered))
    if anchors:
        fields["unresolved_assumptions"].append("The evidence-label rules do not name: " + ", ".join(anchors))
    statements = sum(len(rows) for rows in refuted.values())
    fields["numerical_result"] = (
        f"{invariants['cases']} label cases, {invariants['violation_count']} disagreements with the rules 1-5 oracle, "
        f"{len(invariants['unexercised_branches'])} unexercised rule branches; {screen['cases']} rule 10 probe cases, "
        f"{screen['violation_count']} disagreements ({screen['outcomes'].get('refused', 0)} refused); "
        f"{len(rules)} evidence rules stated; {len(units)} specification units name "
        f"{len(named & set(queue_ids))}/{len(queue_ids)} computational tasks; "
        f"{len(coverage) - len(uncovered)}/{len(coverage)} units have established findings; "
        f"{len(with_counterexamples)} units name tasks that refute {statements} general statements by counterexample, "
        "listed per unit for review.")
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
    ("Chord-arc expansion c = s - kappa^2 s^3/24 to leading order (with start-point curvature an s^4 term "
     "-kappa kappa' s^4/24 follows; 2 sin(kappa s/2)/kappa holds for a plane circle, not a helix)",
     "Taylor expansion of a space curve (do Carmo 1976, ch. 1)", r"(?i:\bchord)"),
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


def _textbook_results(report) -> list:
    """(result, reference) of the textbook ledger entries whose pattern the report's text matches, in ledger order."""
    text = _report_text(report)
    return [(result, reference) for result, reference, pattern in TEXTBOOK if re.search(pattern, text)]


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
    tasks, textbook_use = {}, {result: [] for result, _, _ in TEXTBOOK}
    for report in reports:
        matched = [result for result, _ in _textbook_results(report)]
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
@task("T157", changed_files=(MODULE,), regression_tests=(
    f"{TESTS}::test_counterexample_catalogue", f"{TESTS}::test_next_steps_name_open_work_rather_than_work_done_elsewhere"))
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
        "Counterexamples are compared with a fresh run by ciw lab verify in CI (scripts/check_lab.py); optional: name "
        "one pytest witness per counterexample in each section's tests, so a counterexample is also guarded outside "
        "the clean-room comparison.",
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
UNMEASURED_KIND = "none (no basis: nothing measured)"
BUDGET_CLAIM = re.compile(r"(?i)uncertainty budget")
RAW_UNCERTAINTY_NULL = re.compile(r'^   "uncertainty": null', re.M)


def _uncertainty_parts(declared, measured=True) -> tuple:
    """(kind, value, basis) of a declared per-finding uncertainty.

    A finding whose basis declares no component measured nothing (it records an unestablished claim), so its
    missing uncertainty is shown as such rather than as an undeclared one.
    """
    if declared is None:
        return ("none declared" if measured else UNMEASURED_KIND), None, ""
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
        kind, value, basis = _uncertainty_parts(row["uncertainty"], row.get("measured", True))
        if cells is None or len(cells) != 8:
            problems.append(f"row {index + 1}: malformed")
        elif (cells[0] != row["task_id"] or cells[1] != _flat(row["claim"]) or _headline_problem(cells[2], row["value"])
              or cells[3] != _flat(row["unit"] or "") or cells[4] != kind or _headline_problem(cells[5], value)
              or cells[6] != basis or cells[7] != f"`{row['evidence_status']}`"):
            problems.append(f"row {index + 1}: {row['task_id']} {row['claim'][:40]}")
    return problems


LACKING_CLAIM = "Measured numerical findings that declare no per-finding uncertainty"


@task("T159", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_uncertainty_table_restates_every_numerical_finding",
                        f"{TESTS}::test_uncertainty_listing_separates_unmeasured_records",
                        f"{TESTS}::test_uncertainty_listing_names_the_per_quantity_budgets_it_holds",
                        f"{TESTS}::test_raw_recounts_derive_the_components_of_findings_retained_without_origin"))
def uncertainty_budgets(ctx):
    reports = _reports_before(ctx, 159)
    fields = _fields(
        "Every numerical finding either declares an uncertainty or is identified as lacking one.",
        "Table rows = every finding whose value contains a number: (task, claim, value, unit, uncertainty kind, "
        "value and basis, evidence status). A listing of per-finding components, not a combined budget. A finding "
        "whose basis declares no component (evidence.basis_origin is empty) measured nothing: it records an "
        "unestablished claim, and its value is listed with the kind 'none (no basis: nothing measured)'.",
        [_earlier(159) + " (this section's T155-T158 included)"],
        "Every numerical finding appears once with its declared uncertainty restated exactly; the numerical, "
        "undeclared and unmeasured counts agree with a raw-text recount of the report files.",
        "Collect every finding whose value contains a number, write the table as JSON and Markdown, parse the "
        "Markdown back against the source findings, and recount numerical, undeclared and unmeasured findings from "
        "the raw text.",
        ["uncertainty or value cells cut inside a number", "pipes in claims breaking table columns",
         "non-scalar numerical findings left out", "traversal disagreeing with the raw report text",
         "a record with no basis counted as a measurement without uncertainty"],
        "")
    if not reports:
        fields["recommended_next_task"] = "Run the full queue first, then this listing."
        return _no_prior(fields)
    rows, total = [], 0
    for report in reports:
        for record in report["findings"]:
            total += 1
            if _numbers(record["value"]):
                rows.append({"task_id": report["task_id"], "claim": record["claim"], "value": record["value"],
                             "unit": record.get("unit"), "uncertainty": record.get("uncertainty"),
                             "evidence_status": record["evidence_status"],
                             "measured": bool(finding_origin(record))})
    lacking = [row for row in rows if row["uncertainty"] is None and row["measured"]]
    unmeasured = [row for row in rows if row["uncertainty"] is None and not row["measured"]]
    budgets = sorted({row["task_id"] for row in rows if BUDGET_CLAIM.search(row["claim"])})
    ctx.artifact_json("uncertainty-budget.json", rows)
    lines = ["# Per-finding uncertainty listing", "",
             "Every retained finding whose value contains a number, with its declared uncertainty. Components are "
             "listed, not combined: the findings of one task measure different quantities in different units. A "
             f"finding whose basis declares no component measured nothing; its kind reads `{UNMEASURED_KIND}`.", "",
             UNCERTAINTY_HEADER, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        kind, value, basis = _uncertainty_parts(row["uncertainty"], row["measured"])
        lines.append(_row(row["task_id"], _cell(row["claim"]), _cell(_headline(row["value"])), _cell(row["unit"] or ""),
                          _cell(kind), _cell(_headline(value)), _cell(basis), f"`{row['evidence_status']}`"))
    kinds = Counter(_uncertainty_parts(row["uncertainty"], row["measured"])[0] for row in rows)
    lines += ["", _row("Uncertainty kind", "Findings"), _row("---", "---")] + [
        _row(_cell(kind), str(count)) for kind, count in sorted(kinds.items())]
    text = "\n".join(lines) + "\n"
    ctx.artifact_text("uncertainty-budget.md", text)
    problems = _uncertainty_table_problems(text, rows)
    raw_numeric = raw_lacking = raw_unmeasured = 0
    for _, raw in _raw_reports(ctx, 159):
        for block in _raw_findings(raw):
            if _raw_numeric(block):
                raw_numeric += 1
                if RAW_UNCERTAINTY_NULL.search(block):
                    if not _raw_block_components(block):
                        raw_unmeasured += 1
                    else:
                        raw_lacking += 1
    by_task = Counter(row["task_id"] for row in lacking)
    fields["numerical_result"] = (
        f"{total} findings, {len(rows)} numerical (value contains a number); {len(rows) - len(lacking) - len(unmeasured)} "
        f"declare a per-finding uncertainty, {len(lacking)} measured ones do not, and {len(unmeasured)} record an "
        f"unestablished claim with no basis (nothing measured); kinds {dict(sorted(kinds.items()))}.")
    fields["uncertainty"] = "Exact counts; declared values are restated in full, and undeclared uncertainty is reported, not imputed."
    fields["unresolved_assumptions"] = [
        "Components are listed per finding and not combined across findings: the findings of one task measure "
        "different quantities in different units, so a root-sum-square across them has no meaning. "
        + (f"Budgets of one quantity's components are declared inside the findings of {', '.join(budgets)} (an "
           "uncertainty budget per quantity) and are listed as their values declare them."
           if budgets else "No retained finding declares a budget of components of one quantity."),
        "A numerical finding whose basis declares no component is a record of an unestablished claim, not a "
        "measurement; it is listed with its value but not counted as a measured finding without uncertainty.",
        "Findings of T160-T168 run after this table; the section's regression test asserts that each numerical one "
        "declares an uncertainty."]
    if lacking:
        fields["unresolved_assumptions"].insert(0, f"{len(lacking)} measured numerical findings declare no per-finding "
                                                   "uncertainty: " + ", ".join(f"{tid} ({n})" for tid, n in sorted(by_task.items())))
        fields["recommended_next_task"] = ("Declare a per-finding uncertainty on the measured numerical findings listed "
                                           "without one: " + ", ".join(f"{tid} ({n})" for tid, n in sorted(by_task.items())) + ".")
    else:
        fields["recommended_next_task"] = (
            "None open in this listing: every measured numerical finding declares an uncertainty. Deferred research "
            "question: combine per-quantity components into budgets where one predicted quantity has several "
            "sources, in the form " + (", ".join(budgets) if budgets else "a per-quantity budget") + " uses.")
    if unmeasured:
        fields["unresolved_assumptions"].append(
            f"{len(unmeasured)} numerical findings record an unestablished claim with no basis: "
            + "; ".join(f"{row['task_id']}: {_words(row['claim'], 80)}" for row in unmeasured))
    findings = [
        _count("Uncertainty table restates every numerical finding with its declared uncertainty",
               "computational_pipeline", len(rows),
               [_check("table rows that do not parse back to their source finding and uncertainty", len(problems)),
                _check("numerical findings minus a raw-text recount of numerical finding values",
                       len(rows) - raw_numeric)], "findings"),
        _count(LACKING_CLAIM, "computational_pipeline", len(lacking),
               [_check("undeclared measured numerical findings minus a raw-text recount", len(lacking) - raw_lacking),
                _check("numerical records with no basis minus a raw-text recount", len(unmeasured) - raw_unmeasured)],
               "findings"),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- label qualifications (T160-T163)
# The labels with the meanings of the label table of docs/LAB.md, and what produces each: (label, meaning,
# produced by). Outward-facing pages state them, so that a reader never takes a label in its ordinary-language sense.
LABEL_MEANINGS = (
    ("analytic", "Derived in closed form from declared assumptions", "A cited derivation only"),
    ("synthetic", "Computed from declared generated inputs", "A generator without a passing reference check"),
    ("numerically_verified", "A stated numerical condition passed",
     "Passing analytic, high-precision, invariant, self-convergence, exact or refusal checks"),
    ("provider_backed", "Returned by a pinned external runtime", "Executed provider with repository, revision and tree"),
    ("hardware_measured", "Acquired from an identified physical device",
     "Acquisition record with raw digest, time and calibration reference, after a hardware probe succeeded in the "
     "same task"),
    ("independently_verified", "Agreement between implementations of different origin, not verification by another party",
     "e.g. ciw against scipy, sympy, mpmath or a pinned provider"),
    ("not_established", "Not supported by the basis",
     "Any failed check, any physical claim without acquisition, every claim filed in an authority domain"),
)
LABEL_HEADER = "| Label | Meaning | Produced by |"
BOUNDARY_HEADER = "| A computational experiment may establish | It cannot establish alone |"
INDEPENDENCE = ("`independently_verified` means independent implementation agreement; independent verification by "
                "another party is outside what the queue can establish.")
BASIS_NOTE = ("Each finding's basis is shown beside its label: the basis components it declares and the identity each "
              "declares (generator and seed, provider repository@revision, acquisition device), as "
              "`ciw.lab.evidence.describe_basis` renders them. A passing check outranks provenance in the label rules, so "
              "two findings with one label can rest on different bases; identities are as declared, not authenticated.")


def _label_lines() -> list:
    """The label definitions and the independence qualification, as Markdown."""
    return ([LABEL_HEADER, "| --- | --- | --- |"]
            + [_row(f"`{label}`", _cell(meaning), _cell(source)) for label, meaning, source in LABEL_MEANINGS]
            + ["", INDEPENDENCE, "", BASIS_NOTE, ""])


def _boundary_lines() -> list:
    """The computational boundary of ``ciw.lab.evidence.BOUNDARY``, as a Markdown table."""
    return [BOUNDARY_HEADER, "| --- | --- |"] + [_row(_cell(may), _cell(cannot)) for may, cannot in BOUNDARY] + [""]


def _qualification_problems(text: str) -> list:
    """Label definitions, the independence qualification or boundary rows missing from, or altered on, a page."""
    problems = []
    if _table_rows(text, LABEL_HEADER) != [[f"`{label}`", meaning, source] for label, meaning, source in LABEL_MEANINGS]:
        problems.append("label table differs from the label definitions")
    boundary = _table_rows(text, BOUNDARY_HEADER)
    problems += [f"boundary row missing: {may} / {cannot}" for may, cannot in BOUNDARY if [may, cannot] not in boundary]
    if len(boundary) != len(BOUNDARY):
        problems.append(f"{len(boundary)} boundary rows for {len(BOUNDARY)}")
    if INDEPENDENCE not in text:
        problems.append("independence qualification absent")
    return problems


def _section(text: str, heading: str) -> str:
    """The body of the level-2 section ``## heading`` of a Markdown page (empty when absent)."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1) if match else ""


def _statements(value) -> list:
    """An answer that may be a string or a list, as a list of flattened strings."""
    items = value if isinstance(value, list) else [value]
    return [_flat(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)) for item in items if item]


# --------------------------------------------------------------- T160-T162
PAPERS = {
    "T160": ("instrument-methods", "Evidence-labelled computational instruments for curved-surface observation",
             ("observation", "sensor-fusion", "manufacturing", "energy-gpu")),
    "T161": ("geometry-methods", "Geodesic, Jacobi and route-sensitivity experiments with explicit validity domains",
             ("geodesic-jacobi", "flat-torus-topology", "surfaces-discrete")),
    "T162": ("evidence-provenance-note", "A non-upgrading evidence-label discipline for computational experiments",
             ("exchange-provenance", "implementation-targets", "lyapunov")),
}
RESULTS_HEADER = "| Task | Finding | Value | Unit | Evidence | Basis | Report |"
SELECTED_HEADER = "| Task | Selected finding | Value | Unit | Evidence | Basis | Report |"
BOUNDARY_FINDINGS_HEADER = "| Task | Boundary finding | Value | Unit | Evidence | Basis | Report |"
# Tasks whose findings test the label discipline's boundary: visibly distinct labels and workspace
# classification (T100), and production acceptance with the computational-domain loophole (T141).
BOUNDARY_TASKS = ("T100", "T141")
# Assumptions that qualify the independence of a task's references (not statistical independence of noise).
INDEPENDENCE_LIMITS = re.compile(
    r"ciw-authored|ciw-written|written in ciw|same[- ]origin|same-specification"
    r"|shares? (?:its|the|ciw's) (?:origin|right-hand side)"
    r"|independen\w* (?:of ciw|covers|in the integrator|implementation|verification|reproduction|solver)"
    r"|(?:sympy|scipy|mpmath)'s independence|geometry independence|not an? independent|stands in"
    r"|integrator may be independent|independent \w+ comparison did not run", re.I)
# At least these are carried for a task with independently_verified rows: every assumption that mentions
# independence, same origin or ciw-written code, except statements about statistical independence of noise,
# errors or events ("Noise is independent per vertex", "Drops are independent of the signal value").
INDEPENDENCE_WORDS = re.compile(r"independen|same[- ]origin|written in ciw|ciw-written|ciw-authored", re.I)
STATISTICAL_INDEPENDENCE = re.compile(
    r"\b(?:noise|errors?|residuals?|drops|outliers|latency|state|sources|readings|samples)\b(?:\s+[\w-]+){0,3}\s+"
    r"(?:is|are)\s+(?:[\w-]+\s+and\s+)?independent\b|\bindependent (?:between|across|per)\b|per-component independent",
    re.I)


def _qualifies_independence(item: str, verified: bool) -> bool:
    """Whether an assumption limits the independence of a task's references: selected by phrase, and for a task
    with independently_verified rows every mention of independence or origin that is not statistical."""
    return bool(INDEPENDENCE_LIMITS.search(item)) or (
        verified and bool(INDEPENDENCE_WORDS.search(item)) and not STATISTICAL_INDEPENDENCE.search(item))
OUTSTANDING = ("Missing from this draft, which the lab cannot write: the thesis in the authors' words (the generated "
               "thesis restates counts and counterexamples of the retained findings only), a related-work discussion "
               "(the reference list is the T156 textbook ledger's name matches), an interpretation of the "
               "counterexamples against the literature, conclusions, figure selection and captions, and external "
               "peer review. The draft stays partial until they exist.")
UNFINISHED = ("blocked", "deferred", "partial")
QUALIFIED_CLAIM = ("Draft limitations state what each unfinished task is missing, and its methods carry every assumption "
                   "that mentions independence, same origin or ciw-written code in a task with independently_verified "
                   "rows and every other assumption research_portfolio.INDEPENDENCE_LIMITS selects")


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
        lines += ["The label function `ciw.lab.evidence.supported_label`, the authority-wording screen, basis "
                  "components and the report rules, as specified in docs/lab/SPECIFICATIONS.md:", ""] + rules + [""]
    return lines, rules


def _finding_row(report, record) -> str:
    """One draft table row: task, claim, value headline, unit, label, basis in words and report identity."""
    return _row(report["task_id"], _cell(record["claim"]), _cell(_headline(record["value"])),
                _cell(record.get("unit") or ""), f"`{record['evidence_status']}`", _cell(describe_basis(record)),
                f"`{report['report_id'][:19]}`")


def _draft_problems(text: str, cited: list, header: str = RESULTS_HEADER) -> list:
    """Rows of a written draft table that do not restate their source finding (task, claim, value, unit, label,
    basis, report)."""
    parsed = _table_rows(text, header)
    problems = [] if len(parsed) == len(cited) else [f"{len(parsed)} rows for {len(cited)} findings"]
    for index, (cells, (report, record)) in enumerate(zip(parsed, cited)):
        if cells is None or len(cells) != 7:
            problems.append(f"row {index + 1}: malformed")
        elif (cells[0] != report["task_id"] or cells[1] != _flat(record["claim"])
              or _headline_problem(cells[2], record["value"]) or cells[3] != _flat(record.get("unit") or "")
              or cells[4] != f"`{record['evidence_status']}`" or cells[5] != _flat(describe_basis(record))
              or cells[6] != f"`{report['report_id'][:19]}`"):
            problems.append(f"row {index + 1}: {report['task_id']} {record['claim'][:40]}")
    return problems


def _selected(report):
    """A task's selected finding: its first established finding, else its first finding (None without findings)."""
    established = [f for f in report["findings"] if f["evidence_status"] != "not_established"]
    return (established or report["findings"] or [None])[0]


def _figure(report):
    """The first retained SVG figure of a report, or None."""
    return next((a["path"] for a in report["generated_artifacts"] if a["path"].endswith(".svg")), None)


def _thesis(reports, cited, counterexamples, open_claims, physical) -> str:
    """The draft's thesis, stated from counts and counterexamples of the retained findings only."""
    labels = Counter(f["evidence_status"] for _, f in cited)
    established = sum(n for label, n in labels.items() if label != "not_established")
    summary = ", ".join(f"{labels[label]} {label}" for label in LABELS if labels[label] and label != "not_established")
    examples = "; ".join(f"“{_words(f['counterexample']['statement'], 110)}” ({r['task_id']})"
                         for r, f in counterexamples[:3])
    hardware = labels["hardware_measured"]
    return (f"Across {len(reports)} tasks, {established} of {len(cited)} findings are established"
            + (f" ({summary})" if summary else "") + f"; {len(counterexamples)} findings refute a general statement by "
            "counterexample" + (f", for example {examples}" if examples else "") + f"; {len(open_claims)} physical, "
            f"calibration, sensor or authority claims remain not established, and {physical}. "
            + ("The results are computational: none of them is a physical measurement." if not hardware else
               f"{hardware} of the findings are hardware-measured; every other result is computational."))


def _next_steps(report) -> list:
    """A report's stated next steps (an answer the implementation left unstated is not a step)."""
    from .runner import NOT_STATED
    return [item for item in _statements(report["recommended_next_task"]) if item != NOT_STATED]


def _open_physical(report) -> int:
    """Not-established physical, calibration or sensor claims of a report."""
    return sum(f["domain"] in PHYSICAL_DOMAINS and f["evidence_status"] == "not_established" for f in report["findings"])


def _draft_next_step(unfinished, open_physical, authority: int) -> str:
    """The draft's next step, derived from its own limitations: authorship; the unfinished tasks' own next steps
    (tasks sharing one step named together); the completed tasks whose physical claims stay open, named with a
    pointer to their own next steps under Outstanding work; and the authority claims no evidence establishes.

    Nothing here says that a step closes a limitation: a task's next step is its own statement, and it may
    address another question than the open claim.
    """
    grouped: dict = {}
    for report in unfinished:
        steps = _next_steps(report)
        if steps:  # steps that read the same once cut are named once, with every task that states them
            grouped.setdefault(_words(steps[0], 160), []).append(report["task_id"])
    closing = "; ".join(f"{', '.join(ids)}: {step}" for step, ids in grouped.items())
    claims = sum(_open_physical(r) for r in open_physical)
    return ("Write the parts the lab cannot write (the thesis in the authors' words, the related-work discussion, the "
            "interpretation of the counterexamples and the conclusions) and submit the draft for external review"
            + (f"; then the unfinished tasks' own next steps ({closing})" if closing else "")
            + (f"; {claims} physical, calibration or sensor claim{' of completed tasks' if claims == 1 else 's of completed tasks'} "
               f"({', '.join(r['task_id'] for r in open_physical)}) {'stays' if claims == 1 else 'stay'} not "
               "established, and each task's own next step is listed under Outstanding work" if open_physical else "")
            + (f"; {authority} authority-domain claim{' stays' if authority == 1 else 's stay'} not established on "
               "any evidence" if authority else "") + ".")


def _paper(task_id, ctx):
    slug, title, sections = PAPERS[task_id]
    earlier = _reports_before(ctx, int(task_id[1:]))
    reports = [r for r in earlier if r["section"] in sections]
    discipline = task_id == "T162"
    fields = _fields(
        f"{_article(slug.replace('-', ' '))} draft with a thesis, introduction, methods, results, discussion, "
        "limitations and outstanding work can be generated from retained findings alone: every number traceable to "
        "a report identity, every label defined and shown with its basis, and every limitation drawn from "
        "not_established findings and the unfinished tasks' own statements.",
        "Draft = generated Markdown: a thesis restating counts and counterexamples of the retained findings; an "
        "introduction listing the tasks' questions; methods (label definitions, the independence qualification"
        + ("; the specification's evidence rules and the T100 and T141 boundary findings" if discipline else "")
        + ", task hypotheses and models, the assumptions that limit independence); results (one selected finding "
        "and figure per task); a discussion listing each counterexample with its witness; limitations (the "
        "computational boundary, not_established findings, what each unfinished task is missing); outstanding work; "
        "references from the T156 textbook ledger; an appendix with every finding, its label and its basis.",
        [f"Retained reports of sections {', '.join(sections)}", "Textbook ledger of T156 (research_portfolio.TEXTBOOK)",
         "Label definitions (docs/LAB.md) and ciw.lab.evidence.BOUNDARY"]
        + (["docs/lab/SPECIFICATIONS.md (evidence-label rules)", "ciw.lab.evidence constants",
            "Retained reports of T100 and T141 (boundary findings) and T155 (rule checks)"] if discipline else []),
        "Every appendix, selected-finding" + (" and boundary-finding" if discipline else "") + " row restates its "
        "source finding's task, claim, value headline, unit, label, basis and report identity, and its value cell "
        "states no number the source value does not hold; the page states every label definition, the independence "
        "qualification and every boundary row; every unresolved assumption of an unfinished task and every "
        "assumption limiting independence appears in full.",
        f"Generate the draft from retained reports of sections {', '.join(sections)}; parse its finding tables back "
        "against the source findings and its label and boundary tables against their definitions; look up each "
        "required assumption in its section.",
        ["claims or units containing '|' breaking table rows", "labels or bases dropped from rows",
         "values cut inside a number", "rows out of order or missing",
         "partial task described by what it did instead of what is missing",
         "independently_verified shown without its definition or the assumptions that limit it",
         "boundary rows or label definitions missing",
         "a next step that says the unfinished tasks' steps close every limitation (completed tasks keep open "
         "physical claims, and authority claims never close)"],
        "", assumptions=[OUTSTANDING,
                         "For a task with independently_verified rows every assumption that mentions independence, "
                         "same origin or ciw-written code is carried into the Qualifications, statistical independence "
                         "of noise and events excepted (research_portfolio.STATISTICAL_INDEPENDENCE); for other tasks "
                         "they are selected by phrase (research_portfolio.INDEPENDENCE_LIMITS), so one worded otherwise "
                         "is not carried.",
                         "The selected finding of a task is its first established one, not a judgment of importance."])
    if not reports:
        fields["recommended_next_task"] = "Run the full queue first, then generate the draft."
        return _no_prior(fields)
    cited = [(report, record) for report in reports for record in report["findings"]]
    counterexamples = [(r, f) for r, f in cited if f.get("counterexample")]
    open_claims = [(r, f) for r, f in cited
                   if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and f["evidence_status"] == "not_established"]
    unfinished = [r for r in reports if r["state"] in UNFINISHED]
    open_physical = [r for r in reports if r["state"] not in UNFINISHED and _open_physical(r)]
    authority = sum(f["domain"] in AUTHORITY_DOMAINS for _, f in open_claims)
    measured = [r for r in reports if r["physical_validation_status"]["status"] != "not_established"]
    hardware = sum(f["evidence_status"] == "hardware_measured" for _, f in cited)
    physical = ("physical validation is not established for any of them" if not measured else
                f"physical validation is established for {len(measured)} of them ({', '.join(r['task_id'] for r in measured)})")
    fields["recommended_next_task"] = _draft_next_step(unfinished, open_physical, authority)
    lines = [f"# {title}", "",
             f"*Generated draft from retained CIW lab reports. Not peer reviewed. Contains {hardware or 'no'} "
             f"hardware-measured finding{'' if hardware == 1 else 's'}.*", "",
             "## Abstract", "", "*Thesis (generated from the retained findings).* "
             + _thesis(reports, cited, counterexamples, open_claims, physical), "",
             "The argument in the authors' words, its discussion against the literature and its conclusions are "
             "outstanding (see Outstanding work).", "",
             "## Introduction", "",
             f"This draft reports {len(reports)} queued computational experiments from the sections "
             f"{', '.join(sections)}. Each task states a hypothesis and a mathematical prediction, runs a synthetic or "
             "provider-backed protocol, compares the result with an independent reference where one exists, and "
             "records every result as a finding whose evidence label is computed from its declared basis. "
             "The questions:", ""]
    lines += [f"- {r['task_id']} — {_flat(r['title'])}" for r in reports]
    lines += ["", "The textbook results these tasks use are listed under References; a related-work discussion is "
              "outstanding.", "", "## Methods", "", "### Evidence labels", ""] + _label_lines()
    rules, boundary = [], []
    if discipline:
        rule_lines, rules = _label_discipline()
        lines += rule_lines
        boundary = [(r, f) for r in earlier if r["task_id"] in BOUNDARY_TASKS for f in r["findings"]]
        titles = "; ".join(f"{r['task_id']} ({_flat(r['title'])}; {r['state']}, {len(r['findings'])} findings)"
                           for r in earlier if r["task_id"] in BOUNDARY_TASKS)
        lines += ["### Boundary findings", "",
                  f"Retained findings of the tasks that test the discipline's boundary: {titles or 'none retained'}.", ""]
        if boundary:
            lines += [BOUNDARY_FINDINGS_HEADER, "| --- | --- | --- | --- | --- | --- | --- |"]
            lines += [_finding_row(r, f) for r, f in boundary] + [""]
    lines += ["### Tasks", ""]
    for report in reports:
        model = report["mathematical_model"]
        lines += [f"#### {report['task_id']} — {_flat(report['title'])}", "", f"*Hypothesis.* {_flat(report['hypothesis'])}",
                  "", f"*Model.* {_flat(model if isinstance(model, str) else json.dumps(model, ensure_ascii=False))}", ""]
    verified = {r["task_id"] for r, f in cited if f["evidence_status"] == "independently_verified"}
    qualifications = [(r["task_id"], item) for r in reports for item in _statements(r["unresolved_assumptions"])
                      if _qualifies_independence(item, r["task_id"] in verified)]
    lines += ["### Qualifications", "",
              "Assumptions the tasks state that limit the independence of their references or of their "
              "`independently_verified` rows (for a task with such rows, every assumption that mentions independence, "
              "same origin or ciw-written code, statistical independence of noise and events excepted):", ""]
    for tid in sorted({tid for tid, _ in qualifications}):
        lines += [f"- {tid}:"] + [f"  - {item}" for other, item in qualifications if other == tid]
    if not qualifications:
        lines.append("- None stated by these tasks.")
    selected = [(r, _selected(r)) for r in reports if _selected(r) is not None]
    lines += ["", "## Results", "", "### Selected findings", "",
              "One finding per task: its first established finding, or its first finding when none is established. "
              "Every finding is in the Appendix.", "", SELECTED_HEADER, "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [_finding_row(r, f) for r, f in selected]
    figures = [(r, _figure(r)) for r in reports if _figure(r)]
    lines += ["", "### Figures", ""]
    lines += [f"![{r['task_id']} — {_flat(r['title'])}](../../{path})" for r, path in figures] or [
        "No retained figures."]
    lines += ["", "## Discussion", "", "### Counterexamples", "",
              "Each finding below refutes a general statement; the statement must not be restated without the "
              "qualification its witness shows.", ""]
    lines += [f"- {r['task_id']} refutes “{_flat(f['counterexample']['statement'])}”: {_flat(f['claim'])} "
              f"(`{f['evidence_status']}`; {describe_basis(f)}). Witness: `{_headline(f['counterexample'].get('witness'))}`."
              for r, f in counterexamples] or ["- No retained finding of these sections records a counterexample."]
    lines += ["", "Their interpretation against the literature is outstanding.", "",
              "## Limitations", "", "What a computational experiment may establish, and what it cannot establish alone:", ""]
    lines += _boundary_lines()
    lines += ["### Claims not established", ""]
    lines += [f"- {r['task_id']} [{f['domain']}]: {_flat(f['claim'])} — not established."
              for r, f in cited if f["evidence_status"] == "not_established"] or ["- None."]
    lines += ["", "### Unfinished tasks", "", "What each blocked, deferred or partial task is missing, in its own "
              "words (its unresolved assumptions):", ""]
    for report in unfinished:
        missing = _statements(report["unresolved_assumptions"])
        lines.append(f"- {report['task_id']} is {report['state']}; missing or unresolved:")
        lines += [f"  - {item}" for item in missing] or ["  - (its report states no unresolved assumption)"]
    if not unfinished:
        lines.append("- None.")
    lines += ["", "## Outstanding work", "", f"- {OUTSTANDING}"]
    lines += [f"- {r['task_id']}: {item}" for r in unfinished for item in _next_steps(r)[:1]]
    lines += [f"- {r['task_id']} ({_open_physical(r)} physical claim{'' if _open_physical(r) == 1 else 's'} not "
              f"established; its own next step): {item}" for r in open_physical for item in _next_steps(r)[:1]]
    used = {r["task_id"]: _textbook_results(r) for r in reports}
    references = [(result, reference, [tid for tid, matched in used.items() if (result, reference) in matched])
                  for result, reference, _ in TEXTBOOK]
    lines += ["", "## References", "", "Textbook results named in these tasks' reports (T156 ledger):", ""]
    lines += [f"- {reference}: {result} ({', '.join(tasks)})" for result, reference, tasks in references if tasks] or [
        "- No ledger textbook result is named by these tasks."]
    lines += ["", "## Appendix: every finding", "", RESULTS_HEADER, "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [_finding_row(r, f) for r, f in cited]
    text = "\n".join(lines) + "\n"
    ctx.artifact_text(f"{slug}-draft.md", text)
    problems = _draft_problems(text, cited)
    selected_problems = _draft_problems(text, selected, SELECTED_HEADER)
    boundary_problems = _draft_problems(text, boundary, BOUNDARY_FINDINGS_HEADER) if discipline and boundary else []
    qualification_problems = _qualification_problems(text)
    limitations, methods = _section(text, "Limitations"), _section(text, "Methods")
    required = ([("Limitations", item) for r in unfinished for item in _statements(r["unresolved_assumptions"])]
                + [("Methods", item) for _, item in qualifications])
    absent = [item for where, item in required
              if f"  - {item}" not in (limitations if where == "Limitations" else methods)]
    fields["numerical_result"] = (
        f"Draft cites {len(cited)} findings from {len(reports)} reports ({len(selected)} selected, {len(figures)} "
        f"figures, {len(counterexamples)} counterexamples discussed, {len(unfinished)} unfinished tasks, "
        f"{len(qualifications)} independence qualifications); {len(problems) + len(selected_problems) + len(boundary_problems)} "
        f"table rows differ from their source; {len(qualification_problems)} label or boundary statements missing; "
        f"{len(absent)} required assumptions absent.")
    fields["uncertainty"] = "Numbers carry the uncertainty stated in their source findings."
    checks = [_check("appendix rows that do not parse back to their source finding", len(problems)),
              _check("selected-finding rows that do not parse back to their source finding", len(selected_problems))]
    if discipline and boundary:
        checks.append(_check("boundary-finding rows that do not parse back to their source finding", len(boundary_problems)))
    findings = [
        _count("Draft finding tables restate every retained finding of their sections with its retained label and basis",
               "provenance", len(cited), checks, "findings"),
        _count("Draft defines every evidence label, states that independently_verified is implementation agreement "
               "and not verification by another party, and lists every boundary row", "provenance", len(BOUNDARY),
               [_check("label definitions, independence qualification or boundary rows missing or altered",
                       len(qualification_problems))], "boundary rows"),
        _count(QUALIFIED_CLAIM, "provenance", len(required),
               [_check("required assumption statements absent from their section", len(absent))], "statements"),
    ]
    if discipline:
        findings.append(_rules_finding(earlier, rules, fields))
        fields["unresolved_assumptions"].append(
            "Rules 6-9, 11 and 12 are stated from the specification; they are enforced by the validator, the report "
            "builder, the runner and the workspace classifier and tested in tests/test_lab_core.py and "
            "tests/test_lab_bridge.py, not backed by a T155 finding.")
    findings.append(finding("Draft has passed external peer review", "provenance", None, {}, expected_not_established=True))
    return {"state": "partial", "fields": fields, "findings": findings}


RULES_BACKED_CLAIM = ("Draft methods state the specification's evidence rules, of which T155 found the label function "
                      "to follow rules 1-5 and the authority screen to follow rule 10 on its probe claims")


def _rules_finding(earlier, rules, fields):
    """T162's rules are backed by T155's oracle (rules 1-5) and rule 10 probe, when both are retained."""
    t155 = {f["claim"]: f for r in earlier if r["task_id"] == "T155" for f in r["findings"]}
    oracle = next((f for claim, f in t155.items()
                   if claim.startswith("The evidence-label function agrees with the reference oracle")), None)
    screen = t155.get(RULE10_CLAIM)
    violations = screen["value"].get("violations") if screen and isinstance(screen["value"], dict) else None
    if rules and oracle is not None and isinstance(oracle["value"], int) and isinstance(violations, int):
        return _count(RULES_BACKED_CLAIM, "provenance", len(rules),
                      [_check("T155 cases where the label function departs from rules 1-5", oracle["value"]),
                       _check("T155 probe cases where evidence.finding departs from rule 10", violations),
                       _check("stated rules not numbered 1, 2, ... in order", _rule_numbers_problem(rules))], "rules")
    fields["unresolved_assumptions"].append(
        "The methods state the validator's constants only, or T155's oracle comparison and rule 10 probe are not "
        "retained here; the stated rules are not backed by T155 in this run.")
    return finding(RULES_BACKED_CLAIM, "provenance", None, {}, expected_not_established=True)


PAPER_TESTS = {"T161": (f"{TESTS}::test_drafts_carry_every_qualification_of_independently_verified_rows",),
               "T162": (f"{TESTS}::test_draft_next_step_names_completed_tasks_with_open_physical_claims",)}
for _task_id in PAPERS:
    task(_task_id, changed_files=(MODULE,),
         regression_tests=(f"{TESTS}::test_paper_drafts_trace_to_reports",
                           f"{TESTS}::test_paper_drafts_have_a_structure_and_qualify_their_labels")
         + PAPER_TESTS.get(_task_id, ()))(
        lambda ctx, _t=_task_id: _paper(_t, ctx))


# --------------------------------------------------------------- T163
DEMONSTRATION = ("T003", "T005", "T010", "T032", "T047", "T062", "T066", "T084", "T121", "T137")
SHOWN_LABEL = re.compile(r" → `([a-z_]+)`; basis: [^\n]+$", re.M)
CUSTOMER_DEMAND = "- Customer demand is not established:"
PORTFOLIO_QUALIFIED = ("Portfolio page shows each finding's basis beside its label, defines the labels with the "
                       "independence qualification, lists every boundary row and states customer demand")


def _panel_findings(report) -> list:
    """Headline findings of a panel: the first four, the first of every other label, and every not_established one."""
    shown, labels = [], set()
    for index, record in enumerate(report["findings"]):
        label = record["evidence_status"]
        if index < 4 or label not in labels or label == "not_established":
            shown.append(record)
            labels.add(label)
    return shown


def _panel_line(record) -> str:
    return (f"- {_flat(record['claim'])}: `{_headline(record['value'])}` → `{record['evidence_status']}`; "
            f"basis: {_flat(describe_basis(record))}")


def _customer_demand_line(reports) -> str:
    """The portfolio's customer-demand limitation, from the retained findings filed in that domain."""
    demand = [(r["task_id"], f) for r in reports for f in r["findings"] if f["domain"] == "customer_demand"]
    if demand:
        tasks = ", ".join(sorted({tid for tid, _ in demand}))
        return (f"{CUSTOMER_DEMAND} {len(demand)} retained finding{'' if len(demand) == 1 else 's'} filed in the "
                f"customer_demand domain ({tasks}), each `not_established`: a plausible use case is not demand.")
    return (f"{CUSTOMER_DEMAND} no retained finding is filed in the customer_demand domain, and a plausible use case "
            "is not demand (see the boundary above).")


@task("T163", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_portfolio_shows_every_label_in_use",
                        f"{TESTS}::test_portfolio_qualifies_labels_and_states_customer_demand"))
def portfolio_demonstration(ctx):
    reports = _reports_before(ctx, 163)
    fields = _fields(
        "A short tour of retained experiments can show every evidence label in use, with its basis and its "
        "definition, and what each result does not prove.",
        "Curated panels plus, for every label in use that they do not show, the first retained task holding it; each "
        "panel shows its headline findings with their bases and every not_established finding; the page defines the "
        "labels, qualifies independently_verified, and lists the computational boundary and customer demand as "
        "limitations.",
        [_earlier(163), "Label definitions (docs/LAB.md) and ciw.lab.evidence.BOUNDARY"],
        "Every label used by a retained finding appears on the page with the finding's basis, every not_established "
        "finding of a panel is shown, and the label definitions, the independence qualification, every boundary row "
        "and the customer-demand limitation are stated.",
        "Assemble a Markdown portfolio page from the selected reports and their retained figures, then read the "
        "labels, bases, not_established claims, definitions and boundary rows back from the page.",
        ["label in use missing from the page", "not_established findings hidden by a finding limit",
         "curated task not retained", "label shown without its basis",
         "independently_verified read as verification by another party", "customer demand omitted"],
        "Add panels for a retained hardware run's findings beside the clean-room ones once one exists; hardware "
        "runs are aggregated outside the queue by `ciw lab unmeasured`, and this page reads only the clean-room "
        "reports.",
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
             "Each panel: the hypothesis, the headline findings with their evidence labels and bases, every finding "
             "that is not established, the figure, and the physical validation status.", "",
             "## How to read the labels", ""] + _label_lines()
    for tid in chosen:
        report = by_id[tid]
        lines += [f"## {tid} — {report['title']}", "", _flat(report["hypothesis"]) if isinstance(report["hypothesis"], str) else "", ""]
        lines += [_panel_line(record) for record in _panel_findings(report)]
        for artifact in report["generated_artifacts"]:
            if artifact["path"].endswith(".svg"):
                lines.append(f"\n![{tid}](../../{artifact['path']})")
                break
        lines.append(f"\n*Physical validation:* `{report['physical_validation_status']['status']}`\n")
    open_domains = Counter(f["domain"] for r in reports for f in r["findings"]
                           if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and f["evidence_status"] == "not_established")
    lines += ["## Limitations", "", "What a computational experiment may establish, and what it cannot establish alone:",
              ""] + _boundary_lines() + [
        _customer_demand_line(reports),
        f"- Physical, calibration, sensor and authority claims not established in the retained reports: "
        + (", ".join(f"{domain} {count}" for domain, count in sorted(open_domains.items())) or "none") + ".",
        f"- Hardware-measured findings in the retained reports: "
        f"{sum(f['evidence_status'] == 'hardware_measured' for r in reports for f in r['findings'])}.", ""]
    text = "\n".join(lines) + "\n"
    ctx.artifact_text("PORTFOLIO.md", text)
    shown = set(SHOWN_LABEL.findall(text))
    missing = [label for label in in_use if label not in shown]
    hidden = [f"{tid}: {f['claim']}" for tid in chosen for f in by_id[tid]["findings"]
              if f["evidence_status"] == "not_established" and f"- {_flat(f['claim'])}: " not in text]
    unqualified = [record["claim"] for tid in chosen for record in _panel_findings(by_id[tid])
                   if _panel_line(record) not in text]
    qualification_problems = _qualification_problems(text)
    demand_absent = int(_customer_demand_line(reports) not in _section(text, "Limitations"))
    fields["numerical_result"] = (f"{len(chosen)} panels ({len([t for t in DEMONSTRATION if t in by_id])} of "
                                  f"{len(DEMONSTRATION)} curated retained, added {', '.join(added) or 'none'}); labels in "
                                  f"use: {', '.join(in_use)}; labels shown: {', '.join(sorted(shown, key=LABELS.index))}; "
                                  f"{len(unqualified)} shown findings without their basis; "
                                  f"{len(qualification_problems)} label or boundary statements missing.")
    fields["uncertainty"] = "Exact reading of the rendered page; see each source report for its uncertainty."
    if missing:
        fields["unresolved_assumptions"].append("Labels in use but not shown: " + ", ".join(missing))
    findings = [_count("Portfolio page shows every evidence label in use and every not_established finding of its panels",
                       "computational_pipeline", len(chosen),
                       [_check("labels in use absent from the rendered page", len(missing)),
                        _check("not_established findings of the panels absent from the rendered page", len(hidden))],
                       "panels"),
                _count(PORTFOLIO_QUALIFIED, "computational_pipeline", len(BOUNDARY),
                       [_check("shown findings whose line lacks their basis", len(unqualified)),
                        _check("label definitions, independence qualification or boundary rows missing or altered",
                               len(qualification_problems)),
                        _check("customer-demand limitation absent", demand_absent)], "boundary rows")]
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
                   "rust_probe": "rust", "rust_port": "rust"}  # Rust builds: one runtime, one identity per binary
REVISION_KEYS = ("revision", "head", "commit")
TREE_KEYS = ("source_tree", "tree", "source_digest", "runtime_digest", "engine_sha256", "binary_sha256")
RELEASE_SCHEMA = "ciw.lab-release-report.v3"
RELEASE_ENCODING = ("sha256 over the UTF-8 bytes of ciw.core.identities.canonical_json (sorted keys, no whitespace, "
                    "ASCII escapes, NaN refused; the encoding of report identities) of the list "
                    "[[task, state, headline label, [[claim, label], ...]], ...] in queue order")


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


def runtime_inventory(reports) -> list:
    """One row per runtime: every identity recorded for it, with the tasks that recorded each.

    A commit determines its tree, so an entry that records a revision without a tree is merged into the entry
    that records the same runtime and revision with exactly one tree (T097 records heads, T098 heads and trees
    of the same checkouts). Distinct trees or digests of one runtime (two Rust builds) stay distinct identities.
    """
    seen: dict = {}
    for report in reports:
        for entry in runtime_identities(report["provider_runtime_identity"]):
            seen.setdefault((entry["runtime"], entry["revision"] or "", entry["tree"] or ""), set()).add(report["task_id"])
    merged: dict = {}
    for (name, revision, tree), tasks in sorted(seen.items()):
        trees = [t for (n, r, t) in seen if n == name and r == revision and t] if revision and not tree else []
        key = (name, revision, trees[0]) if len(trees) == 1 else (name, revision, tree)
        merged.setdefault(key, set()).update(tasks)
    rows: dict = {}
    for (name, revision, tree), tasks in sorted(merged.items()):
        rows.setdefault(name, []).append({"revision": revision or None, "tree": tree or None, "tasks": sorted(tasks)})
    return [{"runtime": name, "identities": identities,
             "tasks": sorted({tid for identity in identities for tid in identity["tasks"]})}
            for name, identities in sorted(rows.items())]


def _inventory_gaps(reports, inventory) -> list:
    """Recorded runtime identities the inventory does not hold (same runtime and revision, and the same tree or none)."""
    held = {(row["runtime"], i["revision"] or "", i["tree"] or ""): set(i["tasks"])
            for row in inventory for i in row["identities"]}
    gaps = []
    for report in reports:
        for entry in runtime_identities(report["provider_runtime_identity"]):
            name, revision, tree = entry["runtime"], entry["revision"] or "", entry["tree"] or ""
            if not any(n == name and r == revision and (t == tree or not tree) and report["task_id"] in tasks
                       for (n, r, t), tasks in held.items()):
                gaps.append(f"{report['task_id']}: {name}@{revision or '-'}")
    return gaps


def release_record(reports, queue, scope=None) -> dict:
    """The release record of ``reports`` against ``queue``: coverage, states, labels, basis components, runtimes, digest.

    ``scope`` names the tasks the record covers (T165 passes ``"T001-T164"``, the reports before it); a caller
    outside the queue may pass all retained reports of a run to describe the whole run.
    """
    states = Counter(r["state"] for r in reports)
    labels = Counter(f["evidence_status"] for r in reports for f in r["findings"])
    components = Counter(item for r in reports for f in r["findings"] for item in (finding_origin(f) or ["none"]))
    reported = {r["task_id"]: r for r in reports}
    # Reproducible content only: states, headline labels, claims and their labels. Values (compared within
    # tolerance) and artifact bytes (timing figures differ between runs) stay out of the digest.
    content = [[r["task_id"], r["state"], r["evidence_status"]["primary"],
                [[f["claim"], f["evidence_status"]] for f in r["findings"]]] for r in reports]
    return {"schema": RELEASE_SCHEMA, "ciw_version": __version__,
            "scope": scope or (f"{reports[0]['task_id']}-{reports[-1]['task_id']}" if reports else "none"),
            "queue": {"tasks": len(queue["tasks"]),
                      "sections": [{"key": s["key"], "name": s["name"],
                                    "tasks": sum(t["section"] == s["section"] for t in queue["tasks"]),
                                    "reported": sum(t["section"] == s["section"] and t["id"] in reported
                                                    for t in queue["tasks"])} for s in queue["sections"]],
                      "not_reported": [t["id"] for t in queue["tasks"] if t["id"] not in reported]},
            "reports": len(reports), "states": dict(sorted(states.items())),
            "labels": {label: labels.get(label, 0) for label in LABELS},
            "basis_components": {item: components.get(item, 0) for item in ORIGINS + ("none",)},
            "runtimes": runtime_inventory(reports),
            "release_digest": content_identity(content),
            "release_digest_covers": "task states, headline labels, finding claims and finding labels",
            "release_digest_encoding": RELEASE_ENCODING,
            "physical_validation": "not_established" if all(r["physical_validation_status"]["status"] == "not_established"
                                                           for r in reports) else "mixed"}


def _raw_components(text: str) -> Counter:
    """Basis components of a report file's findings, counted from the raw text (``none`` for a finding that
    declares none); a finding without an ``origin`` key is counted from its raw basis keys."""
    counts = Counter()
    for block in _raw_findings(text):
        counts.update(_raw_block_components(block) or ["none"])
    return counts


@task("T165", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_release_report_inventories_nested_runtimes",
                        f"{TESTS}::test_release_report_lists_each_runtime_once_and_states_its_scope",
                        f"{TESTS}::test_raw_recounts_derive_the_components_of_findings_retained_without_origin"))
def release_report(ctx):
    reports = _reports_before(ctx, 165)
    scope = f"T001-T{165 - 1:03d}"  # the reports this task reads: every task before it in queue order
    fields = _fields(
        f"A release report of the tasks before T165 ({scope}) can be generated entirely from machine-readable queue "
        "metadata and their retained reports.",
        f"Scope {scope}: T165 reads only earlier reports, so T165-T168 are not in it; queue-state.json and the "
        "dashboard cover the whole run. Totals over states, labels and basis components; queue coverage per section; "
        "runtime inventory = revision- or tree-bearing entries anywhere in each provider/runtime identity, one row per "
        "runtime with each identity it recorded (a revision recorded without its tree is merged with the entry that "
        f"records that revision's tree); release digest = {RELEASE_ENCODING}.",
        [_earlier(165), "Packaged queue definition (src/ciw/lab/queue.json)"],
        "State, label and basis-component totals agree with a raw-text recount of the report files; every report "
        "whose provider probe succeeded contributes a runtime identity, and every identity recorded is held by the "
        "inventory; the digest depends only on reproducible content.",
        f"Aggregate queue coverage, states, labels, basis components, runtimes and report content of {scope} into JSON "
        "and Markdown and recount states, labels and basis components from the raw report text.",
        ["runtime identities nested below the top level missed", "one runtime listed once per task that recorded it",
         "totals disagreeing with the report text", "digest over run-specific bytes (timing artifacts, wheel digests)",
         "an undeclared digest encoding", "the report read as covering the whole run"],
        "Sign the release digest with a project key once key custody is defined (a cross-cutting open item of "
        "T166 and T167, owned by no queue task); a whole-run release record, including T165-T168, is computed outside "
        "the queue with research_portfolio.release_record over all retained reports.",
        assumptions=["The release digest is unsigned.",
                     "The digest covers task states, headline labels, claims and their labels, not finding values or "
                     "artifact bytes; values are compared within tolerance by ciw lab verify.",
                     f"The report covers {scope}: T165-T168 run after it and are listed as not reported, so its totals "
                     "differ from queue-state.json and the dashboard of the same run."])
    if not reports:
        return _no_prior(fields)
    queue = load_queue()
    release = release_record(reports, queue, scope)
    ctx.artifact_json("release-report.json", release)
    raw = _raw_reports(ctx, 165)
    raw_states = Counter(state for _, text in raw for state in RAW_STATE.findall(text))
    raw_labels = Counter(label for _, text in raw for label in RAW_LABEL.findall(text))
    raw_components = sum((_raw_components(text) for _, text in raw), Counter())
    state_gap = sum(abs(release["states"].get(s, 0) - raw_states.get(s, 0)) for s in set(release["states"]) | set(raw_states))
    label_gap = sum(abs(release["labels"].get(l, 0) - raw_labels.get(l, 0)) for l in set(release["labels"]) | set(raw_labels))
    component_gap = sum(abs(release["basis_components"].get(c, 0) - raw_components.get(c, 0))
                        for c in set(release["basis_components"]) | set(raw_components))
    unidentified = []
    for report in reports:
        identity = report["provider_runtime_identity"] if isinstance(report["provider_runtime_identity"], dict) else {}
        probes = identity.get("requirement_probes") if isinstance(identity.get("requirement_probes"), dict) else {}
        if any(k.startswith("provider:") and v is True for k, v in probes.items()) and not runtime_identities(identity):
            unidentified.append(report["task_id"])
    gaps = _inventory_gaps(reports, release["runtimes"])
    lines = [f"# Lab release report of {scope}", "",
             f"Scope: the reports of {scope}, retained before this task ran. T165 reads only earlier reports, so "
             "T165-T168 are not in it; queue-state.json and the dashboard cover the whole run.", "",
             f"- CIW version: {__version__}",
             f"- Queue: {release['queue']['tasks']} tasks; {len(reports)} reported; not reported: "
             f"{', '.join(release['queue']['not_reported']) or 'none'}",
             f"- Release digest: `{release['release_digest']}` (unsigned; covers {release['release_digest_covers']}; "
             f"{RELEASE_ENCODING})",
             f"- Physical validation: `{release['physical_validation']}`", "",
             _row("Section", "Tasks", "Reported"), _row("---", "---", "---")] + [
             _row(_cell(s["name"]), str(s["tasks"]), str(s["reported"])) for s in release["queue"]["sections"]] + [
             "", _row("State", "Tasks"), _row("---", "---")] + [_row(k, str(v)) for k, v in release["states"].items()] + [
             "", _row("Evidence label", "Findings"), _row("---", "---")] + [
             _row(f"`{k}`", str(v)) for k, v in release["labels"].items()] + [
             "", "A passing check outranks provenance in the label rules, so the basis components each finding declares "
             "are counted beside the labels (a finding counts once per component it declares):", "",
             _row("Basis component", "Findings"), _row("---", "---")] + [
             _row(f"`{k}`", str(v)) for k, v in release["basis_components"].items()] + [
             "", _row("Runtime", "Revision", "Tree or digest", "Tasks"), _row("---", "---", "---", "---")] + [
             _row(r["runtime"], "<br>".join(f"`{i['revision'] or '-'}`" for i in r["identities"]),
                  "<br>".join(f"`{i['tree'] or '-'}`" for i in r["identities"]),
                  "<br>".join(", ".join(i["tasks"]) for i in r["identities"]))
             for r in release["runtimes"]]
    ctx.artifact_text("RELEASE.md", "\n".join(lines) + "\n")
    names = [r["runtime"] for r in release["runtimes"]]
    fields["numerical_result"] = (f"Release report of {scope}: {len(reports)} of {release['queue']['tasks']} queue tasks "
                                  f"reported; states {release['states']}; labels "
                                  f"{dict((k, v) for k, v in release['labels'].items() if v)}; basis components "
                                  f"{dict((k, v) for k, v in release['basis_components'].items() if v)}; "
                                  f"{len(names)} runtimes ({', '.join(names) or 'none'}) with "
                                  f"{sum(len(r['identities']) for r in release['runtimes'])} identities.")
    fields["uncertainty"] = "Exact counts."
    findings = [
        _count("Release state and label totals match a raw-text recount of the retained report files", "provenance",
               len(reports), [_check("absolute state-count differences from the raw-text recount", state_gap),
                              _check("absolute label-count differences from the raw-text recount", label_gap),
                              _check("absolute basis-component count differences from the raw-text recount",
                                     component_gap)], "reports"),
        _count("Every report whose provider probe succeeded contributes a runtime identity to the release inventory",
               "provenance", len(names),
               [_check("reports with a successful provider probe and no runtime identity", len(unidentified)),
                _check("recorded runtime identities the inventory does not hold", len(gaps)),
                _check("runtimes listed on more than one inventory row", len(names) - len(set(names)))], "runtimes"),
        finding("The release digest is signed by a project key", "provenance", None, {}, expected_not_established=True),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- open items (T166, T167)
# Cross-cutting open items that no queue task closes: (key, item, what would close it, pattern over a report's
# own statements: its unresolved assumptions, its next step and its not_established claims).
OPEN_ITEMS = (
    ("key-custody", "Key custody and signatures",
     "a signing key held outside the workspace and signatures over workspace records, receipts, operator captures "
     "and the release digest (a queue extension; content identities and seals are unkeyed hashes)",
     re.compile(r"\bkeyed\b|signature|signed (?:capture|by|with)|key custody|trust anchor|sign (?:the|receipts)"
                r"|host-held key|authenticat", re.I)),
    ("telemetry-provisioning", "Telemetry provider provisioning",
     "provider checkouts bound to the telemetry and declared-workload workflows (the pins of "
     "src/ciw/telemetry-runtimes.json), provisioned for the clean-room run like the other providers",
     re.compile(r"telemetry (?:workflow|stack|or calibrated)|declared-workload or telemetry|workflows \(telemetry", re.I)),
    ("cross-platform", "Cross-platform reproduction",
     "a non-gating clean-room run on Windows and on another BLAS build or architecture, compared with the retained "
     "reports by ciw lab verify, whose differences answer the platform assumptions",
     re.compile(r"cross-platform|platform-sensitive|other platforms?|(?-i:\bWindows\b)|second (?:Linux )?host"
                r"|other BLAS|BLAS builds?|LAPACK/BLAS|one platform", re.I)),
)


def _own_statements(report) -> list:
    """(kind, text) of a report's own statements about what is open: assumptions, next step, unestablished claims."""
    return ([("assumption", item) for item in _statements(report["unresolved_assumptions"])]
            + [("next step", item) for item in _statements(report["recommended_next_task"])]
            + [("finding", _flat(f["claim"])) for f in report["findings"] if f["evidence_status"] == "not_established"])


def open_items(reports) -> list:
    """Each cross-cutting open item with the statements of the retained reports that raise it."""
    items = []
    for key, name, closes, pattern in OPEN_ITEMS:
        statements = [{"task_id": r["task_id"], "kind": kind, "text": text, "phrase": match.group(0)}
                      for r in reports for kind, text in _own_statements(r) if (match := pattern.search(text))]
        items.append({"key": key, "item": name, "closes_it": closes, "owning_task": None,
                      "tasks": sorted({s["task_id"] for s in statements}), "statements": statements})
    return items


def _open_item_lines(items) -> list:
    lines = ["## Cross-cutting open items", "",
             "No queue task closes these; each needs a queue extension. Statements are grouped by phrases in each "
             "report's own assumptions, next step and unestablished claims (research_portfolio.OPEN_ITEMS).", ""]
    for item in items:
        lines += [f"### {item['item']}", "", f"Closes it: {item['closes_it']}.", ""]
        lines += [f"- {s['task_id']} ({s['kind']}): {s['text']}" for s in item["statements"]] or [
            "- Raised by no retained report."]
        lines.append("")
    return lines


def _open_item_problems(ctx, number, items) -> list:
    """Grouped statements whose matched phrase is absent from their task's raw report file (a second path)."""
    raw = dict(_raw_reports(ctx, number))
    return [f"{s['task_id']}: {s['phrase']}" for item in items for s in item["statements"]
            if s["phrase"] not in raw.get(s["task_id"], "")]


# --------------------------------------------------------------- T166
def _ledger_problems(text: str, rows: list) -> list:
    """Ledger lines that do not restate their assumption and source tasks."""
    lines = [line for line in _section(text, "Ledger").splitlines() if line.startswith("- ")]
    problems = [] if len(lines) == len(rows) else [f"{len(lines)} ledger lines for {len(rows)} assumptions"]
    for index, (line, row) in enumerate(zip(lines, rows)):
        assumption, _, tasks = line[2:].rpartition(" (")
        if assumption != _flat(row["assumption"]) or tasks != ", ".join(row["tasks"]) + ")":
            problems.append(f"line {index + 1}")
    return problems


OPEN_ITEMS_CLAIM = "Cross-cutting open items are listed with every retained statement that raises them"


@task("T166", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_unresolved_assumption_ledger", f"{TESTS}::test_ledgers_list_cross_cutting_open_items"))
def unresolved_assumptions(ctx):
    reports = _reports_before(ctx, 166)
    fields = _fields(
        "Every unresolved assumption stated by any task can be listed in one ledger with its source task, and the "
        "open items that cut across tasks can be grouped from the tasks' own statements.",
        "Ledger = union over reports of unresolved_assumptions, deduplicated by exact wording, with task references. "
        "Open items = key custody and signatures, telemetry provider provisioning and cross-platform reproduction, "
        "each with the assumptions, next steps and unestablished claims whose wording raises it.",
        [_earlier(166) + " (this section's T155-T165 included)"],
        "Every stated assumption appears once per exact wording with every task stating it; citations agree with a "
        "raw-text recount of the report files; every statement grouped under an open item holds its matched phrase "
        "in its task's raw report file.",
        "Collect, deduplicate and count assumptions; group the open items; write JSON and Markdown ledgers; parse the "
        "Markdown ledger back and recount assumption items and open-item phrases from the raw report text.",
        ["assumptions dropped by the traversal (raw-text recount)", "ledger lines not restating their assumption",
         "reports stating no assumption", "cross-cutting open items missing from the ledger"],
        "Attach each assumption to the experiment that would resolve it and track closure, and open a queue "
        "extension for each of the three open items this ledger groups, which no queue task closes.")
    if not reports:
        return _no_prior(fields)
    items = open_items(reports)
    ledger: dict = {}
    for report in reports:
        for item in _statements(report["unresolved_assumptions"]):
            ledger.setdefault(item, []).append(report["task_id"])
    raised = {s["text"]: [] for item in items for s in item["statements"] if s["kind"] == "assumption"}
    for item in items:
        for s in item["statements"]:
            if s["kind"] == "assumption" and item["key"] not in raised[s["text"]]:
                raised[s["text"]].append(item["key"])
    rows = [{"assumption": k, "tasks": v, "open_items": raised.get(k, [])} for k, v in sorted(ledger.items())]
    ctx.artifact_json("unresolved-assumptions.json", rows)
    ctx.artifact_json("open-items.json", items)
    text = ("# Unresolved assumptions\n\n## Ledger\n\n"
            + "\n".join(f"- {row['assumption']} ({', '.join(row['tasks'])})" for row in rows) + "\n\n"
            + "\n".join(_open_item_lines(items)) + "\n")
    ctx.artifact_text("UNRESOLVED_ASSUMPTIONS.md", text)
    problems = _ledger_problems(text, rows)
    item_problems = _open_item_problems(ctx, 166, items)
    citations = sum(len(row["tasks"]) for row in rows)
    raw_citations = sum(_raw_assumption_count(raw) for _, raw in _raw_reports(ctx, 166))
    silent = [r["task_id"] for r in reports if not r["unresolved_assumptions"]]
    raised_items = [item["item"] for item in items if item["statements"]]
    fields["numerical_result"] = (f"{len(rows)} distinct unresolved assumptions ({citations} statements) from "
                                  f"{len(reports)} reports; {len(silent)} reports state none; open items: "
                                  + "; ".join(f"{item['item']} {len(item['statements'])} statements from "
                                              f"{len(item['tasks'])} tasks" for item in items) + ".")
    fields["uncertainty"] = "Exact counts; completeness depends on what each task states."
    fields["unresolved_assumptions"] = (
        ([f"Reports stating no unresolved assumption: {', '.join(silent)}"] if silent else [])
        + ["T167 and T168 run after this ledger; their assumptions are in their own reports.",
           "Assumptions are deduplicated by exact wording; one assumption worded differently is listed twice.",
           "Open items are grouped by phrases (research_portfolio.OPEN_ITEMS); a statement that raises one in other "
           "words is not grouped, and that no queue task closes them is read from the queue's titles, not checked."])
    findings = [_count("Unresolved-assumption ledger lists every assumption stated by the retained reports",
                       "provenance", len(rows),
                       [_check("ledger lines that do not restate their assumption and tasks", len(problems)),
                        _check("assumption statements minus a raw-text recount of the report files",
                               citations - raw_citations)], "assumptions"),
                _count(OPEN_ITEMS_CLAIM, "provenance", len(raised_items),
                       [_check("grouped statements whose phrase is absent from their task's raw report file",
                               len(item_problems))], "items")]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T167
# Acquisition routes: a hardware probe of the runner (the only way a physical label is established), the words
# by which a claim names its device, and the go/no-go step on the capture host.
ACQUISITION_ROUTES = {
    "nvidia-gpu": (re.compile(r"\bGPU\b|NVML|nvidia|RTX|CUDA", re.I),
                   "`ciw energy probe --gpu-index 0` must report status ok"),
    "rapl": (re.compile(r"RAPL|CPU package|package energy", re.I),
             "an intel-rapl energy_uj counter under /sys/class/powercap must read as a number (the runner's "
             "hardware:rapl probe); `python -m ciw.lab.energy_gpu_telemetry rapl-capture` then records the capture"),
}
# A finding's own statement (claim or basis notes) that the code it needs does not exist, or that it needs a
# reference instrument no acquisition route provides.
IMPLEMENTATION_MISSING = re.compile(
    r"\bno (?:[\w/-]+ ){0,3}(?:implementation|port|kernel path|capture path|reader|execution path)\b|not implemented"
    r"|implementation missing|does not (?:exist|ingest)|not ingested|(?:cannot|none can) run on any host"
    r"|was not written|no [\w/-]+ (?:environment|worker)",
    re.I)
REFERENCE_MISSING = re.compile(r"no external (?:[\w-]+ ){0,2}(?:meter|reference|instrument)"
                               r"|independent reference instrument", re.I)
HARDWARE_CLAIM = re.compile(r"\bGPU\b|NVML|RAPL|CUDA|RTX|FPGA|\bhardware\b"
                            r"|real (?:sensor|camera|device|machine|scanner|part|instrument)s?\b", re.I)
# A task's own statement that the code a claim names was not run or written ("Julia, C++ and GPU-host encoders
# were not run"); it applies to a claim that names one of the statement's subjects.
UNRUN_CODE = re.compile(r"\b(?:were|was) not (?:run|written)\b|\bnot written\b|\bnever (?:run|written)\b", re.I)
SUBJECT_NAME = re.compile(r"[A-Z][\w.#+-]*\+*|\b\w+\+\+")
ORDINARY_CAPITALS = frozenset({"A", "An", "The", "This", "These", "That", "Those", "No", "Its", "Their", "Only",
                               "Every", "Each", "All", "Some"})
# A claim about retained synthetic fixtures themselves: their origin is fixed, and no acquisition changes it.
FIXTURE_CLAIM = re.compile(r"\bfixtures?'?(?=\s|$)", re.I)
SYNTHETIC_ORIGIN = re.compile(r"synthetic[_ ]fixture|fixtures? (?:are|is) synthetic", re.I)
# A physical quantity stated as a result: such a claim in a computational domain is misfiled, whereas a
# computational comparison that names a device (bitwise agreement on the GPU) runs on that device's host.
PHYSICAL_QUANTITY = re.compile(r"\benerg(?:y|ies)\b|\bpower\b|temperature|\bclock\b|\bduration\b|\bjoules?\b"
                               r"|\bwatts?\b|utili[sz]ation|\blatency\b|wall[- ]clock|run time", re.I)
NEEDS = {"implementation": "Needs implementation first",
         "reference_instrument": "Needs a reference instrument no acquisition route provides",
         "fixture_origin": "Claims about retained synthetic fixtures (no acquisition changes their origin)",
         "computational_domain": "Physical quantities filed under a computational domain",
         "no_probe": "No instrument probe for it in its task"}
EXECUTION = "execution:"  # prefix of a computational comparison that runs only on a route's host


def _need_heading(need: str) -> str:
    if need.startswith(EXECUTION + "hardware:"):
        return f"Runs on the {need.split(':')[-1]} host (computational comparison; no acquisition needed)"
    return NEEDS.get(need, f"Acquisition on {need}")


BOUNDARY_DOMAINS = {"Physical truth": "physical", "Calibration validity": "calibration",
                    "Real sensor performance": "sensor_performance", "Machine safety": "machine_safety",
                    "Industrial readiness": "industrial_readiness", "Actual customer demand": "customer_demand",
                    "Safe actuator authority": "actuator_authority"}


def _notes(record) -> str:
    notes = record["basis"].get("notes") if isinstance(record.get("basis"), dict) else None
    return " ".join(_statements(notes)) if notes else ""


def _failed_probes(report) -> list:
    identity = report["provider_runtime_identity"] if isinstance(report["provider_runtime_identity"], dict) else {}
    probes = identity.get("requirement_probes") if isinstance(identity.get("requirement_probes"), dict) else {}
    return sorted(k for k, v in probes.items() if v is False and k.startswith(("hardware:", "tool:")))


def _route_tasks(reports) -> dict:
    """Tasks on each acquisition route: a failed probe of it, or a named capture variable of one of its roles."""
    from .runner import CAPTURE_INSTRUMENTS, OPERATOR_CAPTURE_VARIABLES
    routes = {}
    for probe in ACQUISITION_ROUTES:
        variables = [OPERATOR_CAPTURE_VARIABLES[role] for role, instrument in sorted(CAPTURE_INSTRUMENTS.items())
                     if instrument == probe and role in OPERATOR_CAPTURE_VARIABLES]
        routes[probe] = [r["task_id"] for r in reports
                         if f"hardware:{probe}" in _failed_probes(r)
                         or any(v in json.dumps(r, ensure_ascii=False) for v in variables)]
    return routes


def _unrun_subjects(report, claim: str) -> list:
    """The task's statements that code the claim names was not run or written (its unresolved assumptions)."""
    found = []
    for item in _statements(report.get("unresolved_assumptions")):
        match = UNRUN_CODE.search(item)
        if not match:
            continue
        names = [n for n in SUBJECT_NAME.findall(item[:match.start()]) if n not in ORDINARY_CAPITALS]
        if any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", claim) for name in names):
            found.append(item)
    return found


def classify_unmeasured(report, record, routes) -> str:
    """What an open finding needs, from the task's and the finding's own statements.

    ``implementation`` when the claim or its basis notes say the code does not exist, or the task says that
    code the claim names was not run or written; ``reference_instrument`` when they name a missing external
    reference; ``fixture_origin`` for a claim about retained synthetic fixtures (their origin is fixed);
    ``computational_domain`` for a physical quantity stated in a computational domain (no acquisition can
    support it there); ``execution:hardware:<probe>`` for another computational claim in a task on that
    acquisition route whose notes name the route's requirement or whose claim names its device (a comparison
    that runs on the route's host and needs no acquisition); ``hardware:<probe>`` for a physical claim when the
    task is on that acquisition route and the claim or notes name its device; else ``no_probe``.
    """
    notes = _notes(record)
    text = f"{record['claim']} {notes}"
    if IMPLEMENTATION_MISSING.search(text) or _unrun_subjects(report, record["claim"]):
        return "implementation"
    if REFERENCE_MISSING.search(text):
        return "reference_instrument"
    if FIXTURE_CLAIM.search(record["claim"]) and SYNTHETIC_ORIGIN.search(text):
        return "fixture_origin"
    on_route = [probe for probe in ACQUISITION_ROUTES if report["task_id"] in routes.get(probe, [])]
    computational = record["domain"] in COMPUTATIONAL_DOMAINS
    if computational and PHYSICAL_QUANTITY.search(record["claim"]):
        return "computational_domain"
    for probe in on_route:
        if computational and (f"needs hardware:{probe}" in notes or ACQUISITION_ROUTES[probe][0].search(record["claim"])):
            return f"{EXECUTION}hardware:{probe}"
        if not computational and ACQUISITION_ROUTES[probe][0].search(text):
            return f"hardware:{probe}"
    return "no_probe"


def _unmeasured_rows(reports, routes) -> list:
    """Open physical claims, and open computational claims that name hardware or (in a task whose requirement
    probe failed) say their code does not exist, each with what it needs."""
    rows = []
    for report in reports:
        failed = _failed_probes(report)
        for record in report["findings"]:
            if record["evidence_status"] != "not_established" or record["domain"] in AUTHORITY_DOMAINS:
                continue
            physical = record["domain"] in PHYSICAL_DOMAINS
            missing = failed and IMPLEMENTATION_MISSING.search(f"{record['claim']} {_notes(record)}")
            if physical or HARDWARE_CLAIM.search(record["claim"]) or missing:
                rows.append({"task_id": report["task_id"], "domain": record["domain"], "claim": record["claim"],
                             "needs": classify_unmeasured(report, record, routes), "failed_probes": failed,
                             "notes": _notes(record), "task_statements": _unrun_subjects(report, record["claim"])})
    return rows


def _next_acquisitions(rows, routes) -> list:
    """Acquisition routes ranked by the open physical claims they address, then by the computational comparisons
    that run only on their host (a route with neither is not ready)."""
    ranked = []
    for probe, (device, go) in ACQUISITION_ROUTES.items():
        claims = [row for row in rows if row["needs"] == f"hardware:{probe}"]
        runs = [row for row in rows if row["needs"] == f"{EXECUTION}hardware:{probe}"]
        if claims or runs:
            hosts = sorted(set(re.findall(r"RTX \d{3,4}", " ".join(row["claim"] for row in claims + runs))))
            ranked.append({"probe": f"hardware:{probe}", "go_no_go": go,
                           "tasks": sorted({r["task_id"] for r in claims + runs}), "route_tasks": routes[probe],
                           "claims": len(claims), "claim_tasks": sorted({r["task_id"] for r in claims}),
                           "comparisons": len(runs), "comparison_tasks": sorted({r["task_id"] for r in runs}),
                           "hosts": hosts})
    return sorted(ranked, key=lambda route: (-route["claims"], -route["comparisons"], route["probe"]))


def _acquisition_step(route) -> str:
    host = f"the {route['hosts'][0]} host" if route["hosts"] else "the capture host"
    counts = [f"{route['claims']} open physical claim{'' if route['claims'] == 1 else 's'} name this route's device"
              ] if route["claims"] else []
    if route["comparisons"]:
        counts.append(f"{route['comparisons']} computational comparison{'' if route['comparisons'] == 1 else 's'} "
                      f"({', '.join(route['comparison_tasks'])}) run only on this host")
    return (f"On {host}: {route['go_no_go']}; then run {', '.join(route['tasks'])} there into a fresh output "
            f"directory following their protocols ({'; '.join(counts)}) and retain the run with "
            "`ciw lab hardware retain`")


# Hand-labelled probe records for the classifier, checked before the ledger is trusted (like T168's tie probe):
# (task, failed probes, unresolved assumptions, claim, domain, basis notes, expected need; None = not listed).
UNMEASURED_PROBE = (
    ("T116", ("hardware:nvidia-gpu",), (), "GPU-domain gross energy per measured batch", "physical",
     "no NVIDIA GPU or NVML in this environment", "hardware:nvidia-gpu"),
    ("T116", ("hardware:nvidia-gpu",), (),
     "The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution", "sensor_performance",
     "NVML declares no accuracy or resolution for this counter and no external power meter was compared",
     "reference_instrument"),
    ("T118", ("hardware:nvidia-gpu",), (), "RTX 2080 kernel-only duration of the Gaussian VI kernel", "physical",
     "log.json brackets launch, synchronization and copy; kernel spans need an Nsight Systems report, which this "
     "section does not ingest", "implementation"),
    ("T117", ("hardware:nvidia-gpu", "tool:julia"), (),
     "The gaussian_vi PTX kernel on the GPU reproduces the NumPy reference bitwise on every replica", "numerical",
     "no NVIDIA GPU answered the hardware:nvidia-gpu probe in this task; the PTX kernel of the common workload exists "
     "(ciw.energy_cuda gaussian_vi), so this comparison runs on a host where the probe succeeds (needs "
     "hardware:nvidia-gpu)", "execution:hardware:nvidia-gpu"),
    ("T117", ("hardware:nvidia-gpu", "tool:julia"), (),
     "A Julia implementation of the common workload agrees with the NumPy reference", "numerical",
     "implementation missing: no Julia port of the common workload exists, so none can run on any host",
     "implementation"),
    ("T120", ("hardware:nvidia-gpu",), (), "GPU energy per batch of the float32 kernel is below the float64 kernel's",
     "numerical", "", "computational_domain"),
    ("T147", ("hardware:nvidia-gpu",), (), "GPU/CPU agreement establishes industrial readiness", "industrial_readiness",
     "", None),
    ("T124", (), (), "The fixtures' counter readings were produced by a physical GPU and NVML counter", "physical",
     "the fixtures declare origin synthetic_fixture and carry placeholder library and executable digests",
     "fixture_origin"),
    ("T124", (), (), "The bound operator log's counter readings come from an NVML device present on this analyzing host",
     "physical", "no operator NVML log was bound (--capture energy-log=PATH or CIW_LAB_ENERGY_LOG)",
     "hardware:nvidia-gpu"),
    ("T119", (), (), "Physical GPU energy per accepted numerical result", "physical",
     "no operator NVML log was supplied (--capture energy-log=PATH or CIW_LAB_ENERGY_LOG); the repository fixtures "
     "are synthetic, so no physical energy per accepted result was measured", "hardware:nvidia-gpu"),
    ("T146", (), ("Julia, C++ and GPU-host encoders were not run",),
     "Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations", "computational_pipeline", "",
     "implementation"),
    ("T115", (), (), "Gross CPU package energy per geodesic trajectory", "physical",
     "no rapl-log capture was bound (--capture rapl-log=PATH or CIW_LAB_RAPL_LOG)", "hardware:rapl"),
    ("T005", (), (), "Nearby real trajectories on a physical curved surface separate according to this Jacobi law",
     "physical", "", "no_probe"),
    ("T149", (), (), "The frame format keeps its LSB-first burst order on an FPGA over a physical link",
     "computational_pipeline", "", "no_probe"),
)


def _unmeasured_probe_errors() -> list:
    """Probe records whose need the classifier gets wrong (a listed claim with another need, or a listing error)."""
    reports: dict = {}
    for task_id, failed, assumptions, claim, domain, notes, _ in UNMEASURED_PROBE:
        report = reports.setdefault(task_id, {
            "task_id": task_id, "findings": [], "unresolved_assumptions": list(assumptions),
            "provider_runtime_identity": {"requirement_probes": {probe: False for probe in failed}}})
        report["findings"].append(finding(claim, domain, None, {"notes": [notes]} if notes else {},
                                          expected_not_established=domain in COMPUTATIONAL_DOMAINS))
    probe_reports = list(reports.values())
    routes = _route_tasks(probe_reports)
    needs = {(row["task_id"], row["claim"]): row["needs"] for row in _unmeasured_rows(probe_reports, routes)}
    return [f"{task_id}: {claim[:60]} -> {needs.get((task_id, claim))}, expected {expected}"
            for task_id, _, _, claim, _, _, expected in UNMEASURED_PROBE if needs.get((task_id, claim)) != expected]


@task("T167", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_unmeasured_ledger", f"{TESTS}::test_unmeasured_ledger_separates_code_from_hardware",
                        f"{TESTS}::test_unmeasured_classifier_is_checked_on_probe_records",
                        f"{TESTS}::test_ledgers_list_cross_cutting_open_items"))
def unmeasured(ctx):
    reports = _reports_before(ctx, 167)
    fields = _fields(
        "Everything that remains physically unmeasured or unrun is enumerable from the retained reports, with what "
        "each open claim needs: an acquisition on a named route, a run on a named route's host, code that does not "
        "exist yet, a reference instrument, or an instrument probe.",
        "Unmeasured = physical/authority-domain findings not established; blocked, deferred and partial tasks with "
        "what they leave unresolved; open physical claims and open computational claims that name hardware, each "
        "classified from its own claim and basis notes and its task's statements (implementation missing, including "
        "code its task says was not run or written; reference instrument missing; a claim about synthetic fixtures, "
        "whose origin no acquisition changes; a computational comparison that runs only on the host of its task's "
        "acquisition route; a physical quantity filed under a computational domain; hardware:<probe> when its task "
        "is on that acquisition route and the claim names the route's device, else no instrument probe), the "
        "classifier checked on hand-labelled probe records first; acquisition routes ranked by the physical claims "
        "they address, then by the comparisons that run only on their host; the computational boundary per domain; "
        "cross-cutting open items.",
        [_earlier(167), "Acquisition routes: the runner's hardware probes and capture roles "
                        "(runner.CAPTURE_INSTRUMENTS, runner.OPERATOR_CAPTURE_VARIABLES)", "ciw.lab.evidence.BOUNDARY"],
        "The ledger's counts of not-established physical/authority claims and of unfinished tasks agree with a "
        "raw-text recount of the report files; every open physical claim is classified exactly once, and the count "
        "classified agrees with a raw-text recount; the classifier gives every hand-labelled probe record its "
        "label; every open-item statement holds its phrase in its task's raw report file.",
        "Classify the hand-labelled probe records; collect the findings, task states and probe outcomes; classify "
        "each open claim; rank the acquisition routes; write JSON and Markdown; recount claims, states and "
        "open-item phrases from the raw report text.",
        ["claims dropped by the traversal (raw-text recount)", "partial tasks omitted",
         "hardware claims filed under a computational domain",
         "missing code attributed to missing hardware (a claim listed under a probe its own notes say cannot help)",
         "a computational comparison that needs only a GPU host filed as a misfiled hardware claim and left out of "
         "the host's run", "claims about synthetic fixtures counted as acquirable",
         "authority claims attributed to hardware", "customer demand omitted",
         "a classifier that loses every acquisition route (hand-labelled probe records)"],
        "")
    if not reports:
        fields["recommended_next_task"] = "Run the full queue first, then this ledger."
        return _no_prior(fields)
    scope = f"T001-T{167 - 1:03d}"
    claims = [{"task_id": r["task_id"], "claim": f["claim"], "domain": f["domain"]}
              for r in reports for f in r["findings"]
              if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and f["evidence_status"] == "not_established"]
    unfinished = [{"task_id": r["task_id"], "state": r["state"], "reason": _words(r["experiment"], 300),
                   "unrun": _statements(r["unresolved_assumptions"])}
                  for r in reports if r["state"] in UNFINISHED]
    probe_errors = _unmeasured_probe_errors()
    routes = _route_tasks(reports)
    rows = _unmeasured_rows(reports, routes)
    ranked = _next_acquisitions(rows, routes)
    items = open_items(reports)
    measured = [r["task_id"] for r in reports for f in r["findings"] if f["evidence_status"] == "hardware_measured"]
    boundary = []
    for may, cannot in BOUNDARY:
        domain = BOUNDARY_DOMAINS.get(cannot)
        boundary.append({"may_establish": may, "cannot_establish_alone": cannot, "domain": domain,
                         "not_established": None if domain is None else sum(c["domain"] == domain for c in claims)})
    for domain in sorted((PHYSICAL_DOMAINS | AUTHORITY_DOMAINS) - set(BOUNDARY_DOMAINS.values())):
        boundary.append({"may_establish": None, "cannot_establish_alone": domain.replace("_", " ").capitalize(),
                         "domain": domain, "not_established": sum(c["domain"] == domain for c in claims)})
    document = {"scope": scope, "not_established_claims": claims, "unfinished_tasks": unfinished,
                "open_claims_by_need": rows, "next_acquisitions": ranked, "acquisition_route_tasks": routes,
                "boundary": boundary, "open_items": items, "hardware_measured_findings": measured,
                "hardware_runs": "aggregated outside the queue by `ciw lab unmeasured`; never read here"}
    ctx.artifact_json("unmeasured.json", document)
    lines = ["# What remains unmeasured", "",
             f"Scope: the clean-room reports of {scope} in this output directory. Hardware runs retained under "
             "lab/hardware/ are aggregated outside the queue by `ciw lab unmeasured --retained lab`; this ledger never "
             "reads them, so its counts cover this run only.", "",
             f"Hardware-measured findings in this run: {len(measured)}.", "", "## Next acquisitions", ""]
    lines += [f"{index}. {route['probe']} — {_acquisition_step(route)}." for index, route in enumerate(ranked, 1)] or [
        "No acquisition route of the runner addresses an open physical claim or a host-only comparison in these "
        "reports."]
    lines += ["", "Open physical claims with no instrument probe for them in their task (below) need a probe of their "
              "instrument, or a signed-capture trust anchor, before any acquisition can establish them.", "",
              "## Open claims by what they need", ""]
    order = ([f"hardware:{probe}" for probe in ACQUISITION_ROUTES]
             + [f"{EXECUTION}hardware:{probe}" for probe in ACQUISITION_ROUTES] + list(NEEDS))
    for need in order:
        group = [row for row in rows if row["needs"] == need]
        if not group:
            continue
        lines += [f"### {_need_heading(need)}", ""]
        if need.startswith(EXECUTION):
            lines += ["Computational comparisons whose code exists and that run only where the route's probe "
                      "succeeds; a run on that host (listed under Next acquisitions) decides them, and they need no "
                      "acquisition record:", ""]
        if need == "computational_domain":
            lines += ["Domain assignment is the author's choice (T141; rule 10 screens authority wording only), so a "
                      "claim that states a physical quantity can sit in a computational domain, where no acquisition "
                      "can support it; it needs refiling in a physical domain:", ""]
        for row in group:
            extra = f"; failed probes {', '.join(row['failed_probes'])}" if row["failed_probes"] else ""
            said = row["notes"] or "; ".join(f"its task: {item}" for item in row.get("task_statements", []))
            note = f" — {said}" if said and need in ("implementation", "reference_instrument", "fixture_origin") else ""
            lines.append(f"- {row['task_id']} [{row['domain']}{extra}]: {_flat(row['claim'])}{note}")
        lines.append("")
    lines += ["## Physical and authority claims not established", ""]
    lines += [f"- {c['task_id']} [{c['domain']}]: {_flat(c['claim'])}" for c in claims]
    lines += ["", "## Computational boundary", "",
              "What a computational experiment may establish, what it cannot establish alone, and the not-established "
              "claims filed in the domain that records it:", "",
              _row("May establish", "Cannot establish alone", "Domain", "Not established"), _row("---", "---", "---", "---")]
    lines += [_row(_cell(b["may_establish"] or "-"), _cell(b["cannot_establish_alone"]), b["domain"] or "(outside the queue)",
                   "-" if b["not_established"] is None else str(b["not_established"])) for b in boundary]
    lines += ["", "Customer demand: " + (
        f"{next(b['not_established'] for b in boundary if b['domain'] == 'customer_demand')} retained claims filed in "
        "the customer_demand domain, none established; a plausible use case is not demand."), ""]
    lines += _open_item_lines(items)
    lines += ["## Blocked, deferred and partial tasks", ""]
    for item in unfinished:
        lines.append(f"- {item['task_id']} ({item['state']}): {item['reason']}")
        lines += [f"  - Unrun or unresolved: {part}" for part in item["unrun"]]
    ctx.artifact_text("UNMEASURED.md", "\n".join(lines) + "\n")
    raw = _raw_reports(ctx, 167)
    raw_claims = sum(domain in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and label == "not_established"
                     for _, text in raw for domain, label in RAW_DOMAIN_LABEL.findall(text))
    raw_unfinished = sum(state in UNFINISHED for _, text in raw for state in RAW_STATE.findall(text))
    raw_physical = sum(domain in PHYSICAL_DOMAINS and label == "not_established"
                       for _, text in raw for domain, label in RAW_DOMAIN_LABEL.findall(text))
    classified_physical = sum(row["domain"] in PHYSICAL_DOMAINS for row in rows)
    item_problems = _open_item_problems(ctx, 167, items)
    states = Counter(item["state"] for item in unfinished)
    needs = Counter(row["needs"] for row in rows)
    fields["recommended_next_task"] = (
        (_acquisition_step(ranked[0]) + (f"; next, {ranked[1]['probe']}: {ranked[1]['go_no_go']}" if len(ranked) > 1 else "")
         + ". Claims needing implementation first, and those with no instrument probe, need code before hardware.")
        if ranked else
        "No acquisition route addresses an open physical claim or a host-only comparison here: implement the "
        "instrument probes, readers and kernels the open claims need (see 'Open claims by what they need') before "
        "any acquisition.")
    fields["numerical_result"] = (
        f"Scope {scope}: {len(claims)} physical/authority claims not established; {len(unfinished)} unfinished tasks "
        f"{dict(sorted(states.items()))}; open claims by need {dict(sorted(needs.items()))}; ready acquisitions "
        + (", ".join(f"{route['probe']} ({route['claims']} physical claims: {', '.join(route['claim_tasks']) or '-'}; "
                     f"{route['comparisons']} host-only comparisons: {', '.join(route['comparison_tasks']) or '-'})"
                     for route in ranked) or "none")
        + f"; classifier probe records wrong {len(probe_errors)} of {len(UNMEASURED_PROBE)}; "
        f"{len(measured)} hardware-measured findings; open items "
        + ", ".join(f"{item['key']} {len(item['tasks'])} tasks" for item in items) + ".")
    fields["uncertainty"] = "Exact counts."
    fields["unresolved_assumptions"] = [
        "T168 runs after this ledger.",
        "Hardware runs retained under lab/hardware/ are aggregated outside the queue by `ciw lab unmeasured`; this "
        "ledger reads only the clean-room reports of its output directory, so its hardware-measured count covers "
        "this run only.",
        "What an open claim needs is read from its own claim and basis notes, its task's statements that code was "
        "not run or written, and its task's probes (phrases in research_portfolio.IMPLEMENTATION_MISSING, "
        "UNRUN_CODE, REFERENCE_MISSING, SYNTHETIC_ORIGIN, PHYSICAL_QUANTITY and ACQUISITION_ROUTES); a claim whose "
        "statements do not say so is classified by its domain and route alone. The hand-labelled probe records "
        "(research_portfolio.UNMEASURED_PROBE) test the classifier on the wordings the queue uses, not on every "
        "wording.",
        "A route ranks by the open physical claims that name its device in tasks on it, then by the computational "
        "comparisons that run only on its host; whether a physical claim changes label depends on the capture "
        "passing the acquisition gate on the capture host, and a comparison's label on its own checks there."]
    if probe_errors:
        fields["unresolved_assumptions"].insert(0, "Classifier probe records given another need: "
                                                + "; ".join(probe_errors))
    findings = [
        _count("Unmeasured ledger lists every not-established physical or authority claim", "provenance", len(claims),
               [_check("listed claims minus a raw-text recount of not-established physical/authority findings",
                       len(claims) - raw_claims)], "claims"),
        _count("Unmeasured ledger lists every blocked, deferred or partial task", "provenance", len(unfinished),
               [_check("listed tasks minus a raw-text recount of blocked, deferred and partial states",
                       len(unfinished) - raw_unfinished)], "tasks"),
        _count("Unmeasured ledger classifies every open physical claim by what it needs", "provenance",
               classified_physical,
               [_check("open physical claims classified minus a raw-text recount of not-established physical findings",
                       classified_physical - raw_physical),
                _check("hand-labelled probe records the classifier gives another need (or lists wrongly)",
                       len(probe_errors))], "claims"),
        _count(OPEN_ITEMS_CLAIM, "provenance", sum(bool(item["statements"]) for item in items),
               [_check("grouped statements whose phrase is absent from their task's raw report file",
                       len(item_problems))], "items"),
        finding("Physical validity of the lab's computational results", "physical", None, {}),
    ]
    return {"state": _state(findings), "fields": fields, "findings": findings}


# --------------------------------------------------------------- T168
MARKER = "lab_task"  # @pytest.mark.lab_task("T001", ...) declares the tasks a registered regression test guards


def _is_marker(decorator) -> bool:
    """Whether a decorator (or a ``marks=`` entry) is a ``pytest.mark.lab_task(...)`` or ``mark.lab_task(...)`` call."""
    target = decorator.func if isinstance(decorator, ast.Call) else None
    return (isinstance(target, ast.Attribute) and target.attr == MARKER
            and (isinstance(target.value, ast.Attribute) and target.value.attr == "mark"
                 or isinstance(target.value, ast.Name) and target.value.id == "mark"))


def _declared_tasks(decorators) -> frozenset:
    """Task ids that ``lab_task`` marker decorators (``pytest.mark.lab_task`` or ``mark.lab_task``) name as literals."""
    return frozenset(a.value for decorator in decorators if _is_marker(decorator) for a in decorator.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str))


def _marked_cases(decorators) -> list:
    """(case id, marks) of each ``pytest.param(..., marks=...)`` literal in a test's one parametrize decorator.

    The case id is the param's literal ``id``, else its string values joined
    by "-", as pytest forms it; a param with other values is skipped.
    """
    parametrize = [d for d in decorators if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                   and d.func.attr == "parametrize"]
    if len(parametrize) != 1 or len(parametrize[0].args) < 2 or any(k.arg == "ids" for k in parametrize[0].keywords):
        return []  # stacked parametrizations and ids= form other case ids: those cases carry the test's markers only
    values = parametrize[0].args[1]
    cases = []
    for value in values.elts if isinstance(values, (ast.List, ast.Tuple)) else []:
        if not (isinstance(value, ast.Call) and getattr(value.func, "attr", getattr(value.func, "id", "")) == "param"):
            continue
        keywords = {k.arg: k.value for k in value.keywords}
        parts = [keywords["id"]] if "id" in keywords else value.args
        if "marks" in keywords and parts and all(isinstance(p, ast.Constant) and isinstance(p.value, str)
                                                 for p in parts):
            marks = keywords["marks"]
            cases.append(("-".join(p.value for p in parts),
                          marks.elts if isinstance(marks, (ast.List, ast.Tuple)) else [marks]))
    return cases


def _index_source(relative: str, text: str, index: dict) -> None:
    """Add the test functions of one test module, and the parametrized cases whose marks are literal: node id ->
    (source of the test and the module names it uses, its ``lab_task`` markers left out; task ids they declare)."""
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
                    segments += [segment(d) for d in current.decorator_list if not _is_marker(d)]
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

    def collect(prefix, body, inherited):
        # A marker on a test class applies to its methods, and a pytest.param's marks to its case, as pytest does.
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                source, declared = closure(node), inherited | _declared_tasks(node.decorator_list)
                index[f"{prefix}::{node.name}"] = (source, declared)
                for case, marks in _marked_cases(node.decorator_list):
                    index[f"{prefix}::{node.name}[{case}]"] = (source, declared | _declared_tasks(marks))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                collect(f"{prefix}::{node.name}", node.body, inherited | _declared_tasks(node.decorator_list))

    collect(relative, tree.body, frozenset())


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
    return {node for node in _test_index(tests_dir) if "[" not in node}


LABEL_WORDS = re.compile(r"evidence_status|" + "|".join(LABELS))


def _tie(task_id: str, node: str, index: dict) -> tuple:
    """(declares the task, asserts a label) for one registered node.

    A node declares its task when a ``lab_task`` marker on its test function,
    an enclosing test class or its ``pytest.param`` case names the task id as
    a literal string; it asserts a label when its source (helpers, fixtures
    and constants it uses included) mentions ``evidence_status`` or a label.
    """
    entry = index.get(node, index.get(node.split("[")[0]))
    if entry is None:
        return False, False
    source, declared = entry
    return task_id in declared, LABEL_WORDS.search(source) is not None


def _names(task_id: str, function: str, node: str, index: dict) -> bool:
    """Whether a registered node names its task: the task id in its node id, test name or source (its ``lab_task``
    markers left out), or the task function's name in that source. Advisory: naming a task is not running it."""
    entry = index.get(node, index.get(node.split("[")[0]))
    if entry is None:
        return False
    source = entry[0]
    return (task_id in node or task_id.lower() in node.rsplit("::", 1)[-1] or task_id in source
            or (function not in ("", "<lambda>") and re.search(rf"\b{re.escape(function)}\b", source) is not None))


def _stray_declarations(index: dict, registered: dict) -> list:
    """``node: task`` for each ``lab_task`` declaration naming a task that does not register what it marks.

    A case's own marks may name the tasks that register the case or its
    test; a test's (or its class's) marker applies to every case, so it may
    name only the tasks that register the test or every registered case.
    """
    tasks = {}
    for task_id, nodes in registered.items():
        for node in nodes:
            tasks.setdefault(node, set()).add(task_id)
    stray = []
    for node, (_, declared) in sorted(index.items()):
        test = node.split("[")[0]
        if node != test:
            declared, allowed = declared - index[test][1], tasks.get(node, set()) | tasks.get(test, set())
        else:
            cases = [owners for case, owners in tasks.items() if case.startswith(node + "[")]
            allowed = tasks.get(node, set()) | (set.intersection(*cases) if cases else set())
        stray += [f"{node}: {task_id}" for task_id in sorted(declared - allowed)]
    return stray


# Probe cases for the tie analysis, every node registered for T901 (and one case for T904): (node id, expected
# (declares T901, asserts label)), whether nodes name a task, and the declarations naming a task that does not
# register what they mark.
TIE_PROBE = '''
import pytest
from pytest import mark
SECTION = ("T901",)
def _label(report):
    return report["evidence_status"]["primary"]
@pytest.mark.lab_task("T901")
def test_literal(run):
    assert run("T901")["findings"][0]["evidence_status"] == "numerically_verified"
@pytest.mark.lab_task("T900", "T901")
def test_constant(run):
    for task_id in SECTION:
        assert _label(run(task_id)) == "analytic"
def test_t901_named(run):
    assert _label(run.last)
@pytest.mark.parametrize("task_id", SECTION)
@mark.lab_task("T901")
def test_parametrized(run, task_id):
    assert run(task_id)["state"] == "completed" and _label(run(task_id))
def test_unrelated():
    assert 1 + 1 == 2
@pytest.mark.lab_task("T901")
def test_values_only(run):
    assert run("T901")["findings"][0]["value"] == 1
@pytest.mark.lab_task("T902")
@pytest.mark.skipif(False, reason="T901")
def test_other_task(run):
    assert _label(run("T901"))
@pytest.mark.lab_task(*SECTION)
def test_computed_ids(run):
    assert _label(run("T901"))
@pytest.mark.lab_task("T901")
class TestGroup:
    @pytest.mark.lab_task("T902")
    def test_method(self, run):
        assert _label(run("T901"))
@pytest.mark.parametrize("task_id", [pytest.param("T901", marks=pytest.mark.lab_task("T901")),
                                     pytest.param("T903", marks=[mark.lab_task("T903")])])
def test_case_marks(run, task_id):
    assert _label(run(task_id))
@pytest.mark.lab_task("T901", "T904")
@pytest.mark.parametrize("task_id", ["T901", "T904"])
def test_shared_marker(run, task_id):
    assert _label(run(task_id))
'''
TIE_EXPECTED = {"test_literal": (True, True), "test_constant": (True, True), "test_t901_named": (False, True),
                "test_parametrized[T901]": (True, True), "test_unrelated": (False, False),
                "test_values_only": (True, False), "test_other_task": (False, True),
                "test_computed_ids": (False, True), "TestGroup::test_method": (True, True),
                "test_case_marks[T901]": (True, True), "test_shared_marker[T901]": (True, True)}
# A marker is not read as naming its task: test_other_task and TestGroup::test_method run T901 only.
NAMES_EXPECTED = {("T901", "test_literal"): True, ("T901", "test_constant"): True, ("T901", "test_t901_named"): True,
                  ("T901", "test_parametrized[T901]"): True, ("T901", "test_unrelated"): False,
                  ("T902", "test_other_task"): False, ("T902", "TestGroup::test_method"): False}
STRAY_EXPECTED = ["tests/test_probe.py::TestGroup::test_method: T902",
                  "tests/test_probe.py::test_case_marks[T903]: T903", "tests/test_probe.py::test_constant: T900",
                  "tests/test_probe.py::test_other_task: T902", "tests/test_probe.py::test_shared_marker: T901",
                  "tests/test_probe.py::test_shared_marker: T904"]


def _tie_probe_errors() -> list:
    index: dict = {}
    _index_source("tests/test_probe.py", TIE_PROBE, index)
    nodes = {node: f"tests/test_probe.py::{node}" for node in TIE_EXPECTED}
    errors = [node for node, expected in TIE_EXPECTED.items() if _tie("T901", nodes[node], index) != expected]
    errors += [f"{node} names {task_id}" for (task_id, node), expected in NAMES_EXPECTED.items()
               if _names(task_id, "", nodes[node], index) != expected]
    registered = {"T901": list(nodes.values()), "T904": ["tests/test_probe.py::test_shared_marker[T904]"]}
    if _stray_declarations(index, registered) != STRAY_EXPECTED:
        errors.append("stray declarations")
    return errors


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


OBSERVED_LABELS_STEP = ("record the evidence labels each lab_task-marked test asserts when it runs (for example as "
                        "JUnit properties written by a fixture that wraps report access), so a label assertion is "
                        "observed in the record rather than inferred from the test's source text")
STATIC = "integer count from a static source analysis; see the unresolved assumptions for what it cannot see"


@task("T168", changed_files=(MODULE,),
      regression_tests=(f"{TESTS}::test_regression_coverage_is_checked",
                        f"{TESTS}::test_regression_outcomes_add_up_and_name_the_own_node"))
def permanent_regression_tests(ctx):
    from .runner import repository_path
    reports = _reports_before(ctx, 168)
    fields = _fields(
        "Every completed or partial experiment is backed by at least one registered pytest regression test that "
        "declares the task it guards and asserts its evidence labels, and those tests pass.",
        "Per task: registered node ids resolved against test functions (AST); a registration is declared when a "
        "lab_task marker on the test (its class, or its pytest.param case) names the task; a task is tied when one "
        "of its declared nodes asserts an evidence label (source text); marker declarations naming a task that does "
        "not register what they mark are counted; tied tasks whose tied tests never name the task are counted as "
        "advisory; JUnit outcomes per node as recorded in the reports.",
        [_earlier(168) + " (this section's T155-T167 included)", "Registered regression node ids of T001-T168",
         "tests/ of the repository (CIW_LAB_REPOSITORY_ROOT in the clean room)"],
        "No completed or partial task lacks a registered test; every registered node resolves and its lab_task "
        "marker names every task that registers it, and no marker names another task; every task has a declared "
        "test that asserts an evidence label; no registered node failed in the JUnit record; every task-to-node "
        "registration has a recorded outcome except T168's own, which its own run's JUnit record cannot hold.",
        "Resolve every registered node id against the test functions under tests/, read each test's lab_task "
        "markers from its decorators and its source for a label assertion (the analysis is checked on probe cases "
        "first), and fold the JUnit outcomes the reports recorded.",
        ["registered node ids that do not resolve", "registrations the test's lab_task marker does not declare",
         "lab_task markers naming a task that does not register the test", "tests asserting values only",
         "tests registered for a task they never run (advisory: no tied test names the task)",
         "registered tests failing or not run", "registrations without a recorded outcome dropped from the totals"],
        OBSERVED_LABELS_STEP[0].upper() + OBSERVED_LABELS_STEP[1:] + ".",
        assumptions=["Declarations are read from literal task ids in lab_task decorators of the test or its class, "
                     "and in the marks of the pytest.param literals of its one parametrize decorator for that case: "
                     "a module-level pytestmark or a task id computed at run time is not read, and such a "
                     "registration counts as undeclared.",
                     "A label assertion is inferred from source text: mentioning a label is taken as asserting it. "
                     "Whether a tied test runs its task is not established: a marker declares the tie, and the "
                     "advisory count only checks that some tied test names the task or its function.",
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
    rows, values_only, unnamed = [], [], []
    for task_id in sorted(set(states) | ({"T168"} & set(implementations))):
        implementation = implementations.get(task_id)
        nodes = list(implementation.regression_tests) if implementation else []
        function = getattr(getattr(implementation, "run", None), "__name__", "")
        ties = {node: _tie(task_id, node, index) for node in nodes}
        missing = [n for n in nodes if n.split("[")[0] not in index]
        tied = [n for n, (declared, labelled) in ties.items() if declared and labelled]
        rows.append({"task_id": task_id, "state": states.get(task_id, "running"), "regression_tests": nodes,
                     "missing": missing,
                     "undeclared": [n for n, (declared, _) in ties.items() if not declared and n not in missing],
                     "tied": tied, "tied_naming_the_task": [n for n in tied if _names(task_id, function, n, index)],
                     "junit": {n: outcomes.get(task_id, {}).get(n, "not recorded") for n in nodes}})
        if len(missing) < len(nodes) and not any(labelled for _, labelled in ties.values()):
            values_only.append(task_id)
        if tied and not rows[-1]["tied_naming_the_task"]:
            unnamed.append(task_id)
    uncovered = [r["task_id"] for r in rows if r["state"] in ("completed", "partial") and not r["regression_tests"]]
    dangling = [f"{r['task_id']}: {n}" for r in rows for n in r["missing"]]
    undeclared = [f"{r['task_id']}: {n}" for r in rows for n in r["undeclared"]]
    stray = _stray_declarations(index, {t: i.regression_tests for t, i in implementations.items()})
    untied = [r["task_id"] for r in rows if r["regression_tests"] and not r["tied"]]
    probe_errors = _tie_probe_errors()
    probe = _check("tie-analysis probe cases classified wrongly", len(probe_errors))
    recorded = Counter(outcome for r in rows for outcome in r["junit"].values())
    failed = [f"{r['task_id']}: {n}" for r in rows for n, o in r["junit"].items() if o == "failed"]
    ctx.artifact_json("regression-coverage.json", rows)
    findings = [
        _count("Completed or partial tasks lacking a regression test", "computational_pipeline", len(uncovered),
               [_check("completed or partial tasks with no registered regression test", len(uncovered))], "tasks"),
        _count("Registered regression node ids that do not resolve to a test function", "computational_pipeline",
               len(dangling), [_check("dangling node ids", len(dangling))], "node ids"),
        _count("Tasks without a registered test that both declares the task with a lab_task marker and asserts an "
               "evidence label", "computational_pipeline", len(untied), [probe], "tasks", basis=STATIC),
        _count("Task-to-node registrations whose test's lab_task marker does not name the task",
               "computational_pipeline", len(undeclared), [probe], "registrations", basis=STATIC),
        _count("Declarations in lab_task markers naming a task that does not register the test",
               "computational_pipeline", len(stray), [probe], "declarations", basis=STATIC),
        _count("Tied tasks none of whose tied tests names the task or its function (advisory, not a completion "
               "condition)", "computational_pipeline", len(unnamed), [probe], "tasks", basis=STATIC),
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
    if stray:
        fields["unresolved_assumptions"].insert(0, "Declarations in lab_task markers naming a task that does not "
                                                   "register the test: " + "; ".join(stray[:20]))
    if undeclared:
        fields["unresolved_assumptions"].insert(0, "Registrations whose test's lab_task marker does not name the "
                                                   "task: " + "; ".join(undeclared[:20]))
    if untied:
        fields["unresolved_assumptions"].insert(0, "Tasks without a registered test that declares the task and "
                                                   "asserts a label: " + ", ".join(untied))
    if unnamed:
        fields["unresolved_assumptions"].insert(0, "Advisory: tied tasks none of whose tied tests names the task or "
                                                   "its function, so a declared test may not run it: "
                                                   + ", ".join(unnamed))
    if failed:
        fields["unresolved_assumptions"].insert(0, "Registered regression tests failing in the JUnit record: "
                                                   + "; ".join(failed[:20]))
    if uncovered:
        fields["unresolved_assumptions"].insert(0, "Completed or partial tasks without a registered regression "
                                                   "test: " + ", ".join(uncovered))
    not_run = recorded["skipped"] + recorded["not run"]
    # T168's own nodes run in the pytest session before the queue, but the JUnit record is read per task when its
    # report is built, and this report is the one being built: its outcome cannot be recorded here.
    own = sum(outcome == "not recorded" for r in rows if r["task_id"] == "T168" for outcome in r["junit"].values())
    unrecorded = recorded["not recorded"] - own
    registrations = sum(len(r["regression_tests"]) for r in rows)
    distinct = len({n for r in rows for n in r["regression_tests"]})
    declared = registrations - len(dangling) - len(undeclared)
    fields["numerical_result"] = (
        f"{len(rows)} tasks, {registrations} task-to-node registrations ({distinct} distinct node ids), {declared} "
        f"declared by a lab_task marker; {len(undeclared)} registrations whose marker does not name the task; "
        f"{len(stray)} marker declarations naming a task that does not register the test; {len(uncovered)} "
        f"completed/partial tasks without regression tests; {len(dangling)} dangling node ids; {len(untied)} tasks "
        f"without a tied test; {len(unnamed)} tied tasks whose tied tests never name the task (advisory); JUnit: "
        f"{recorded['passed']} passed, {recorded['failed']} failed, {not_run} skipped or not run, "
        f"{recorded['not recorded']} not recorded ({own} of them T168's own node{'' if own == 1 else 's'}, which its "
        f"own run's record cannot hold).")
    fields["uncertainty"] = ("Exact counts; declarations are read from lab_task decorators, label assertions and "
                             "(advisory) task naming inferred from source text.")
    if unrecorded:
        fields["unresolved_assumptions"].insert(0, f"{unrecorded} task-to-node registrations of other tasks have no "
                                                   "outcome recorded in their reports (a node registered after its "
                                                   "report was written, or a report that records none).")
    # What keeps the task partial, each as forward work; the next step after all of it is OBSERVED_LABELS_STEP.
    gaps = []
    if dangling:
        gaps.append(f"register node ids that resolve to test functions ({len(dangling)} dangling)")
    if undeclared or stray:
        gaps.append(f"make each registered test's lab_task marker name exactly the tasks that register it "
                    f"({len(undeclared)} registrations undeclared, {len(stray)} declarations of another task)")
    if values_only:
        nodes = [n for r in rows if r["task_id"] in values_only for n in r["regression_tests"]]
        gaps.append(f"assert evidence labels in {', '.join(nodes)} (the registered tests of {', '.join(values_only)}, "
                    "which assert values only)")
    if uncovered:  # the task list stays in parentheses, so the planner reads no pointer to a completed task
        gaps.append(f"register a regression test for each completed or partial task without one "
                    f"({', '.join(uncovered)})")
    if failed:
        gaps.append(f"make the failing registered tests pass ({'; '.join(failed[:20])})")
    skipped = sorted({n for r in rows for n, o in r["junit"].items() if o in ("skipped", "not run")})
    if skipped:
        gaps.append(f"make {', '.join(skipped)} run where the JUnit record is written ({not_run} registrations "
                    "skipped or not run there)")
    if not junit_supplied:
        gaps.append("run the lab tests with a JUnit record and pass it to the queue run (ciw lab run --junit)")
    elif unrecorded:
        gaps.append(f"re-run the tasks whose {unrecorded} registrations have no recorded outcome with the JUnit record")
    if gaps:
        step = "; ".join(gaps) + f"; then {OBSERVED_LABELS_STEP}."
        fields["recommended_next_task"] = step[0].upper() + step[1:]
    complete = junit_supplied and not untied and not undeclared and not stray and not not_run and not unrecorded
    return {"state": _state(findings, complete), "fields": fields, "findings": findings}
