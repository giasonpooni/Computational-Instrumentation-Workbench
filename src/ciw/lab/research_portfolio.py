"""Research and portfolio tasks T155-T168: aggregate retained lab evidence.

These tasks run after every other queue task in the same run and read only
the reports already retained in the output directory. They generate
specifications, catalogues, drafts and ledgers from those records; they never
restate a number that is not in a retained finding and never upgrade a label.
Drafted papers are drafts: review, submission and acceptance are not claims
the workbench can make.
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
import tempfile
import zipfile

from .. import __version__
from .evidence import (AUTHORITY_DOMAINS, COMPUTATIONAL_DOMAINS, DOMAINS, LABELS, PHYSICAL_DOMAINS,
                       EvidenceRefusal, finding, supported_label, holds as compare)
from .registry import load_implementations, load_queue, task
from .report import validate_report

MODULE = "src/ciw/lab/research_portfolio.py"
TESTS = "tests/test_lab_research_portfolio.py"
FIRST_OWN = 155
REPO_ROOT = Path(__file__).resolve().parents[3]


def _prior_reports(ctx):
    """Validated reports of tasks before this section, as retained on disk."""
    directory = ctx.output_dir / "reports"
    reports = []
    for path in sorted(directory.glob("T*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("number", FIRST_OWN) < FIRST_OWN:
            reports.append(validate_report(report))
    return reports


def _recomputed(ctx, name, value, build):
    """Retain an aggregate and check it reproduces from a fresh read of the on-disk reports.

    ``build`` recomputes the aggregate from reports re-read from disk. The check
    establishes reproducibility of the retained artifact, not the correctness
    of the aggregation code; independent cross-checks are separate findings.
    """
    path = ctx.artifact_json(name, value)
    retained = json.loads((ctx.output_dir / path).read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(build(_prior_reports(ctx))))
    same = retained == fresh
    return path, {"reference_kind": "exact_arithmetic",
                  "reference": f"{name} recomputed from a fresh read of the retained reports",
                  "observed": 0.0 if same else 1.0, "tolerance": 0.0, "passed": same}


def _brief(value, limit=60):
    """A value for prose tables that never cuts inside a number: full scalars, else a summary."""
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(text) <= limit:
        return text
    if isinstance(value, str):
        return json.dumps(value[: max(8, limit - 12)] + "…")
    size = len(value) if isinstance(value, (list, dict)) else 0
    kind = "list" if isinstance(value, list) else "object"
    return f"({kind} of {size} entries; see the source report)"


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = compare(observed, tolerance, comparison)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _no_prior(fields):
    fields = dict(fields)
    fields["unresolved_assumptions"] = list(fields.get("unresolved_assumptions", [])) + [
        "No earlier queue reports were retained in this output directory; run the full queue first."]
    return {"state": "blocked", "fields": fields, "findings": []}


def _common(hypothesis, model, experiment, next_task, **extra):
    fields = {"hypothesis": hypothesis, "mathematical_model": model,
              "input_data": ["Retained reports T001-T154 in the same output directory"],
              "observation_model": "No physical observation; reads retained computational records only.",
              "expected_invariant": "Aggregates are exact functions of retained reports and never change a label.",
              "experiment": experiment, "recommended_next_task": next_task,
              "failure_modes_checked": ["missing reports", "report identity mismatch (validate_report)",
                                        "aggregate differs from its re-read bytes"],
              "unresolved_assumptions": []}
    fields.update(extra)
    return fields


# --------------------------------------------------------------- T155
SPEC_SECTIONS = {
    "Evidence labels": ("T100", "T155"),
    "Geodesic equation and references": ("T001", "T002", "T003", "T004"),
    "Jacobi equation, transfer matrix and Wronskian": ("T005", "T006", "T007", "T008", "T009"),
    "First-order validity and focal counterexamples": ("T010", "T017"),
    "Chord versus geodesic distance": ("T046", "T047"),
    "Filter consistency (NEES/NIS)": ("T062", "T066", "T074"),
    "Flat torus lattices and winding": ("T019", "T020", "T026"),
    "Deterministic serialization and reduction": ("T146", "T148"),
}


def label_invariants() -> dict:
    """Exhaustively check the label function over a finite basis grammar."""
    passing = {"reference_kind": "analytic", "reference": "r", "observed": 0.0, "tolerance": 1.0, "passed": True}
    failing = dict(passing, observed=2.0, passed=False)
    acquisition = {"device": "d", "raw_sha256": "0" * 64, "acquired_at": "t", "calibration": "not_applied"}
    provider = {"repository": "p", "revision": "r", "source_tree": "t", "executed": True}
    independent_ok = dict(passing, producer={"implementation": "ciw.lab"}, checker={"implementation": "scipy"})
    options = {"derivation": (None, "doc"), "generator": (None, {"name": "g"}), "checks": (None, [passing], [failing]),
               "provider": (None, provider), "independent_check": (None, independent_ok),
               "acquisition": (None, acquisition)}
    violations, cases, labels = [], 0, Counter()
    for values in product(*options.values()):
        basis = {key: value for key, value in zip(options, values) if value is not None}
        for domain in sorted(DOMAINS):
            cases += 1
            try:
                label = supported_label(basis, domain)
            except EvidenceRefusal:
                label = "refused"
            labels[label] += 1
            failed = basis.get("checks") == [failing]
            if domain in AUTHORITY_DOMAINS and label not in ("not_established", "refused"):
                violations.append(f"authority {domain}: {label}")
            if domain in PHYSICAL_DOMAINS and label not in ("not_established", "hardware_measured",
                                                            "independently_verified", "refused"):
                violations.append(f"physical {domain}: {label}")
            if domain in PHYSICAL_DOMAINS and "acquisition" not in basis and label != "not_established":
                violations.append(f"physical without acquisition: {label}")
            if failed and label not in ("not_established", "refused"):
                violations.append(f"failed check yields {label}")
            if domain in COMPUTATIONAL_DOMAINS and label == "hardware_measured":
                violations.append("computational claim labelled hardware_measured")
    return {"cases": cases, "violations": violations, "label_counts": dict(sorted(labels.items()))}


@task("T155", changed_files=(MODULE, "docs/lab/SPECIFICATIONS.md"), regression_tests=(f"{TESTS}::test_label_invariants_hold_exhaustively",))
def formal_specifications(ctx):
    prior = {r["task_id"]: r for r in _prior_reports(ctx)}
    fields = _common(
        "The label function satisfies its stated invariants on every basis in a finite grammar, and every "
        "specification section is exercised by retained tasks with established computational findings.",
        "Specification text in docs/lab/SPECIFICATIONS.md; label function L(basis, domain) from ciw.lab.evidence.",
        "Enumerate 2*2*3*2*2*2 bases x 14 domains; check authority, physical, failed-check and hardware invariants; "
        "map each specification section to its tasks and count established findings.",
        "Extend the grammar with malformed bases (T168 regression) and add a machine-checked proof of the invariants.")
    invariants = label_invariants()

    def build(reports):
        known = {r["task_id"]: r for r in reports}
        return {section: {tid: sum(f["evidence_status"] != "not_established" for f in known[tid]["findings"])
                          if tid in known else None for tid in ids if tid != "T155"}
                for section, ids in SPEC_SECTIONS.items()}
    coverage = build(list(prior.values()))
    ctx.artifact_json("label-invariants.json", invariants)
    _, retained = _recomputed(ctx, "specification-coverage.json", coverage, build)
    uncovered = [s for s, ids in coverage.items() if not any(v for v in ids.values())]
    fields["numerical_result"] = (f"{invariants['cases']} label cases, {len(invariants['violations'])} invariant "
                                  f"violations; {len(coverage) - len(uncovered)}/{len(coverage)} specification "
                                  "sections have established findings.")
    fields["uncertainty"] = "Exact enumeration; coverage depends on which tasks ran in this output directory."
    if not prior:
        fields["unresolved_assumptions"].append("Coverage could not be evaluated: no prior reports retained.")
    if uncovered:
        fields["unresolved_assumptions"].append("Sections without established findings: " + ", ".join(uncovered))
    findings = [
        finding("Evidence-label invariants hold on the exhaustive basis grammar", "mathematical",
                len(invariants["violations"]),
                {"derivation": "docs/lab/SPECIFICATIONS.md#evidence-labels",
                 "checks": [_check("authority/physical/failed-check/hardware invariants", len(invariants["violations"]))]},
                tolerance={"abs": 0, "rel": 0}),
        finding("Specification coverage reproduces from retained reports", "computational_pipeline",
                len(coverage) - len(uncovered),
                {"checks": [retained, _check("sections with established findings", len(coverage) - len(uncovered),
                                             1 if prior else 0, "ge")]},
                unit="sections", tolerance={"abs": 0, "rel": 0}),
    ]
    return {"state": "completed" if prior and not uncovered else "partial", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T156
TEXTBOOK = (
    ("Geodesic equation and Christoffel symbols", "do Carmo, Differential Geometry of Curves and Surfaces, ch. 4"),
    ("Jacobi equation j'' + K j = 0, conjugate and focal points", "do Carmo, Riemannian Geometry, ch. 5"),
    ("Clairaut relation on surfaces of revolution", "do Carmo, Differential Geometry of Curves and Surfaces, 4-4"),
    ("Gauss-Bonnet and cone angles", "Gauss-Bonnet theorem; flat surfaces with cone points (Troyanov)"),
    ("Chord-arc expansion c = s - kappa^2 s^3/24", "Taylor expansion of a space curve"),
    ("Classical RK4, explicit Euler/midpoint, Dormand-Prince 5(4)", "Hairer, Norsett and Wanner, Solving ODEs I"),
    ("Richardson extrapolation", "Richardson (1911); Hairer et al."),
    ("Kalman filter and Rauch-Tung-Striebel smoother", "Kalman (1960); Rauch, Tung and Striebel (1965)"),
    ("NEES/NIS chi-square consistency and gating", "Bar-Shalom, Li and Kirubarajan (2001)"),
    ("Brioschi formula for Gaussian curvature", "Brioschi (1852); Gauss's Theorema Egregium"),
    ("Lattice reduction and SL(2,Z) action", "Lagrange/Gauss reduction of binary quadratic forms"),
    ("Gage R&R ANOVA method", "AIAG Measurement Systems Analysis manual"),
    ("Lyapunov equation and quadratic stability", "Lyapunov (1892); Khalil, Nonlinear Systems"),
)
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
)


@task("T156", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_textbook_and_contribution_ledger",))
def textbook_versus_contribution(ctx):
    fields = _common(
        "Every result used by the lab can be attributed either to established literature or to a named "
        "implementation artifact in this repository.",
        "A two-column ledger: textbook results with references; implementation contributions with their files.",
        "Build the ledger and verify each contribution names an existing source file in the package.",
        "Add per-finding provenance tags so the ledger is derived from findings rather than curated.")
    missing = [path for _, path in CONTRIBUTIONS
               if not (Path(__file__).resolve().parents[1] / path[len("src/ciw/"):]).is_file()]
    def build(_reports):
        return {"textbook": [{"result": r, "reference": ref} for r, ref in TEXTBOOK],
                "contributions": [{"contribution": c, "file": p, "present": p not in missing} for c, p in CONTRIBUTIONS]}
    ledger = build(None)
    _, retained = _recomputed(ctx, "attribution-ledger.json", ledger, build)
    fields["numerical_result"] = f"{len(TEXTBOOK)} textbook results; {len(CONTRIBUTIONS)} contributions, {len(missing)} without a source file."
    fields["uncertainty"] = "Curated attribution; references identify standard sources, not exhaustive priority."
    fields["unresolved_assumptions"] = ["Novelty of contributions relative to published literature is not established by this ledger."]
    findings = [
        finding("Implementation contributions each name a present source file", "provenance", len(missing),
                {"checks": [retained, _check("contribution files missing", len(missing))]}, tolerance={"abs": 0, "rel": 0}),
        finding("Contributions are novel relative to the literature", "mathematical", None, {},
                expected_not_established=True),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T157
@task("T157", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def counterexample_catalogue(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "Every counterexample recorded by a queue task can be collected with its witness and evidence status.",
        "Catalogue = all findings carrying a counterexample statement, keyed by task and claim.",
        "Scan retained reports for counterexample findings; write JSON and Markdown catalogues.",
        "Turn each catalogued counterexample into a named regression fixture (T168).")
    if not prior:
        return _no_prior(fields)
    def build(reports):
        return [{"task_id": r["task_id"], "claim": f["claim"], "statement": f["counterexample"]["statement"],
                 "witness": f["counterexample"].get("witness"), "evidence_status": f["evidence_status"],
                 "report_id": r["report_id"]}
                for r in reports for f in r["findings"] if f.get("counterexample")]
    entries = build(prior)
    _, retained = _recomputed(ctx, "counterexamples.json", entries, build)
    # Independent path: count counterexample keys in the raw report text, no traversal.
    scanned = sum(path.read_text(encoding="utf-8").count('"counterexample": {')
                  for path in sorted((ctx.output_dir / "reports").glob("T*.json"))
                  if int(path.stem[1:]) < FIRST_OWN)
    lines = ["# Counterexample catalogue", "", "Generated from retained lab reports. Each entry refutes the quoted general statement.", ""]
    for entry in entries:
        lines += [f"## {entry['task_id']}: {entry['statement']}", "", f"- Finding: {entry['claim']}",
                  f"- Evidence status: `{entry['evidence_status']}`",
                  f"- Witness: `{_brief(entry['witness'], 400)}`", ""]
    ctx.artifact_text("COUNTEREXAMPLES.md", "\n".join(lines))
    labels = Counter(e["evidence_status"] for e in entries)
    fields["numerical_result"] = f"{len(entries)} counterexamples from {len({e['task_id'] for e in entries})} tasks; labels {dict(labels)}."
    fields["uncertainty"] = "Exact aggregation; each counterexample carries its own uncertainty in its source report."
    findings = [finding("Counterexample catalogue reproduces from retained reports", "computational_pipeline",
                        len(entries), {"checks": [retained, _check("catalogue size minus raw-text key count",
                                                                   len(entries) - scanned)]},
                        unit="counterexamples", tolerance={"abs": 0, "rel": 0})]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T158
def _figure_index(ctx, reports):
    figures = [dict(a, task_id=r["task_id"]) for r in reports for a in r["generated_artifacts"] if a["path"].endswith(".svg")]
    for figure in figures:
        text = (ctx.output_dir / figure["path"]).read_text(encoding="utf-8")
        figure["well_formed"] = text.startswith("<svg") and text.rstrip().endswith("</svg>")
    return figures


@task("T158", changed_files=(MODULE, "src/ciw/lab/svg.py"), regression_tests=(f"{TESTS}::test_figures_are_reproducible",))
def reproducible_figures(ctx):
    from .runner import Context, run_task
    prior = _prior_reports(ctx)
    fields = _common(
        "Retained SVG figures are byte-for-byte reproducible when their tasks are re-executed.",
        "Figure bytes are a deterministic function of task inputs (ciw.lab.svg has no clock or randomness).",
        "Index every retained SVG; re-execute up to two figure-producing tasks in a scratch directory and "
        "compare SHA-256 of every regenerated figure with the retained one.",
        "Re-execute every figure-producing task on a second platform (Windows CI) and compare hashes.")
    if not prior:
        return _no_prior(fields)
    figures = _figure_index(ctx, prior)
    producing = sorted({f["task_id"] for f in figures})[:2]
    implementations, _ = load_implementations()
    queue = {t["id"]: t for t in load_queue()["tasks"]}
    mismatches, compared = [], 0
    with tempfile.TemporaryDirectory(prefix="ciw-lab-figures-") as directory:
        scratch = Context(Path(directory), ctx.providers)
        for task_id in producing:
            rerun = run_task(queue[task_id], implementations.get(task_id), scratch, {})
            fresh = {a["path"]: a["sha256"] for a in rerun["generated_artifacts"]}
            for figure in (f for f in figures if f["task_id"] == task_id):
                compared += 1
                if fresh.get(figure["path"]) != figure["sha256"]:
                    mismatches.append(figure["path"])
    _, retained = _recomputed(ctx, "figure-index.json", figures, lambda reports: _figure_index(ctx, reports))
    fields["numerical_result"] = (f"{len(figures)} retained figures from {len({f['task_id'] for f in figures})} tasks; "
                                  f"{compared} regenerated from {producing}, {len(mismatches)} byte mismatches.")
    fields["uncertainty"] = "Regeneration covers the named tasks on this platform only."
    fields["unresolved_assumptions"] = ["Figures of tasks not re-executed are assumed reproducible from the same code path."]
    findings = [
        finding("Regenerated figures are byte-identical to retained figures", "computational_pipeline", len(mismatches),
                {"checks": [_check("regenerated SVG SHA-256 mismatches", len(mismatches)),
                            _check("figures compared", compared, 1, "ge")]}, tolerance={"abs": 0, "rel": 0}),
        finding("Retained figures are well-formed SVG documents", "computational_pipeline",
                sum(not f["well_formed"] for f in figures),
                {"checks": [retained, _check("malformed figures", sum(not f["well_formed"] for f in figures))]},
                tolerance={"abs": 0, "rel": 0}),
    ]
    return {"state": "completed" if compared else "partial", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T159
@task("T159", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def uncertainty_budgets(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "Every numerical finding either declares an uncertainty or is identified as lacking one.",
        "Budget rows = (task, claim, value, unit, declared uncertainty, evidence status).",
        "Collect every finding with a numerical value; tabulate declared uncertainties and list those without.",
        "Require structured uncertainty components (instrument, geometry, solver) on every numerical finding.")
    if not prior:
        return _no_prior(fields)
    def build(reports):
        rows = []
        for report in reports:
            task_level = report["uncertainty"] if isinstance(report["uncertainty"], str) else json.dumps(report["uncertainty"])
            for record in report["findings"]:
                if isinstance(record["value"], (int, float)) and not isinstance(record["value"], bool):
                    declared = record.get("uncertainty")
                    rows.append({"task_id": report["task_id"], "claim": record["claim"], "value": record["value"],
                                 "unit": record.get("unit"), "uncertainty": declared,
                                 "source": "finding" if declared is not None else "task_statement_only",
                                 "task_uncertainty": task_level, "evidence_status": record["evidence_status"]})
        return rows
    rows = build(prior)
    declared = sum(row["source"] == "finding" for row in rows)
    _, retained = _recomputed(ctx, "uncertainty-budget.json", rows, build)
    lines = ["| Task | Claim | Value | Unit | Uncertainty | Source | Evidence |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        shown = row["uncertainty"] if row["source"] == "finding" else row["task_uncertainty"]
        lines.append(f"| {row['task_id']} | {row['claim']} | {row['value']:.6g} | {row['unit'] or ''} | "
                     f"{json.dumps(shown)[:120]} | {row['source']} | `{row['evidence_status']}` |")
    ctx.artifact_text("uncertainty-budget.md", "\n".join(lines) + "\n")
    fields["numerical_result"] = (f"{len(rows)} scalar numerical findings; {declared} declare a per-finding uncertainty, "
                                  f"{len(rows) - declared} rely on their task's uncertainty statement only.")
    fields["uncertainty"] = "Exact aggregation of declared values; undeclared per-finding uncertainty is reported, not imputed."
    if rows and declared < len(rows):
        fields["unresolved_assumptions"] = [f"{len(rows) - declared} scalar findings carry no per-finding uncertainty."]
    findings = [finding("Uncertainty budget reproduces from retained reports (findings with declared uncertainty)", "computational_pipeline",
                        declared, {"checks": [retained]}, unit="findings", tolerance={"abs": 0, "rel": 0})]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T160-T162
PAPERS = {
    "T160": ("instrument-methods", "Evidence-labelled computational instruments for curved-surface observation",
             ("observation", "sensor-fusion", "manufacturing", "energy-gpu")),
    "T161": ("geometry-methods", "Geodesic, Jacobi and route-sensitivity experiments with explicit validity domains",
             ("geodesic-jacobi", "flat-torus-topology", "surfaces-discrete")),
    "T162": ("evidence-provenance-note", "A non-upgrading evidence-label discipline for computational experiments",
             ("exchange-provenance", "implementation-targets", "lyapunov")),
}


def _paper(task_id, ctx):
    slug, title, sections = PAPERS[task_id]
    prior = [r for r in _prior_reports(ctx) if r["section"] in sections]
    fields = _common(
        f"A {slug.replace('-', ' ')} draft can be assembled entirely from retained findings, with every number "
        "traceable to a report identity and every limitation drawn from not_established findings.",
        "Draft = generated Markdown: abstract, methods (task hypotheses and models), results (findings tables), "
        "limitations (not_established findings and blocked tasks).",
        f"Generate the draft from retained reports of sections {', '.join(sections)}.",
        "Human authorship pass, related-work section and external review; physical experiments from the manufacturing protocols.")
    if not prior:
        return _no_prior(fields)
    lines = [f"# {title}", "", "*Generated draft from retained CIW lab reports. Not peer reviewed. Contains no physical measurement.*", "",
             "## Abstract", "", f"This draft summarizes {len(prior)} queued computational tasks. Every result below carries the "
             "evidence label assigned by `ciw.lab.evidence`; physical validation is not established for any of them.", "",
             "## Methods", ""]
    for report in prior:
        lines += [f"### {report['task_id']} — {report['title']}", "", f"*Hypothesis.* {report['hypothesis']}", "",
                  f"*Model.* {report['mathematical_model'] if isinstance(report['mathematical_model'], str) else json.dumps(report['mathematical_model'])}", ""]
    lines += ["## Results", "", "| Task | Finding | Value | Evidence | Report |", "| --- | --- | --- | --- | --- |"]
    cited = []
    for report in prior:
        for record in report["findings"]:
            cited.append((report["task_id"], record["claim"], record["evidence_status"]))
            lines.append(f"| {report['task_id']} | {record['claim']} | {_brief(record['value'])} {record.get('unit') or ''} | "
                         f"`{record['evidence_status']}` | `{report['report_id'][:19]}` |")
    lines += ["", "## Limitations", ""]
    for report in prior:
        for record in report["findings"]:
            if record["evidence_status"] == "not_established":
                lines.append(f"- {report['task_id']}: {record['claim']} — not established.")
        if report["state"] in ("blocked", "deferred", "partial"):
            lines.append(f"- {report['task_id']} is {report['state']}: {report['experiment'][:200]}")
    ctx.artifact_text(f"{slug}-draft.md", "\n".join(lines) + "\n")
    fields["numerical_result"] = f"Draft cites {len(cited)} findings from {len(prior)} reports."
    retained_pairs = {(r["task_id"], f["claim"], f["evidence_status"]) for r in _prior_reports(ctx) for f in r["findings"]}
    unmatched = [pair for pair in cited if pair not in retained_pairs]
    fields["uncertainty"] = "Numbers carry the uncertainty stated in their source findings."
    fields["unresolved_assumptions"] = ["Prose beyond the generated structure, related work and peer review are outstanding."]
    findings = [
        finding("Draft cites only retained findings with their retained labels", "provenance", len(cited),
                {"checks": [_check("cited (task, claim, label) triples absent from the retained reports", len(unmatched))]},
                unit="findings", tolerance={"abs": 0, "rel": 0}),
        finding("Draft has passed external peer review", "provenance", None, {}, expected_not_established=True),
    ]
    return {"state": "partial", "fields": fields, "findings": findings}


for _task_id in PAPERS:
    task(_task_id, changed_files=(MODULE,),
         regression_tests=(f"{TESTS}::test_paper_drafts_trace_to_reports",))(lambda ctx, _t=_task_id: _paper(_t, ctx))


# --------------------------------------------------------------- T163
DEMONSTRATION = ("T003", "T005", "T010", "T032", "T047", "T062", "T066", "T084", "T121", "T137")


@task("T163", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def portfolio_demonstration(ctx):
    prior = {r["task_id"]: r for r in _prior_reports(ctx)}
    fields = _common(
        "A short tour of retained experiments can show every evidence label in use and what each result does not prove.",
        "Curated selection of task reports with their figures and not_established findings.",
        "Assemble a Markdown portfolio page from the selected reports and their retained figures.",
        "Present the portfolio page with the physical flat-plate/cylinder experiments once measured.")
    chosen = [prior[t] for t in DEMONSTRATION if t in prior]
    if not chosen:
        return _no_prior(fields)
    lines = ["# Computational experimentalist portfolio", "",
             "Each panel: the hypothesis, the headline findings with their evidence labels, the figure, and what is not established.", ""]
    for report in chosen:
        lines += [f"## {report['task_id']} — {report['title']}", "", report["hypothesis"] if isinstance(report["hypothesis"], str) else "", ""]
        for record in report["findings"][:4]:
            lines.append(f"- {record['claim']}: `{_brief(record['value'], 80)}` → `{record['evidence_status']}`")
        for artifact in report["generated_artifacts"]:
            if artifact["path"].endswith(".svg"):
                lines.append(f"\n![{report['task_id']}](../../{artifact['path']})")
                break
        lines.append(f"\n*Physical validation:* `{report['physical_validation_status']['status']}`\n")
    ctx.artifact_text("PORTFOLIO.md", "\n".join(lines) + "\n")
    used = sorted({f["evidence_status"] for r in chosen for f in r["findings"]})
    fields["numerical_result"] = f"{len(chosen)} of {len(DEMONSTRATION)} selected reports retained; labels shown: {', '.join(used)}."
    fields["uncertainty"] = "Curated selection; see each source report."
    findings = [finding("Portfolio panels assembled from retained reports", "computational_pipeline", len(chosen),
                        {"checks": [_check("panels", len(chosen), 1, "ge")]}, unit="panels", tolerance={"abs": 0, "rel": 0})]
    return {"state": "completed" if len(chosen) == len(DEMONSTRATION) else "partial", "fields": fields, "findings": findings}


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


@task("T164", changed_files=(MODULE, "scripts/reproduce_lab.py"),
      regression_tests=(f"{TESTS}::test_clean_room_marker_is_recognized",
                        f"{TESTS}::test_clean_room_needs_the_installed_package_to_be_the_named_wheel"))
def clean_room_reproduction(ctx):
    fields = _common(
        "One command builds an isolated wheel, installs it into a fresh virtual environment, runs the lab tests "
        "and the whole queue, and matches the retained reports within their regression tolerances.",
        "scripts/reproduce_lab.py: wheel -> venv -> pytest (JUnit) -> ciw lab run --all -> ciw lab verify.",
        "When the queue runs inside that command, it exports CIW_LAB_CLEAN_ROOM with the wheel digest; this task "
        "checks that the imported package is that wheel's files. Outside the command the task is partial.",
        "Run scripts/reproduce_lab.py on Windows and on a second Linux host and retain both gate records.")
    marker = os.environ.get("CIW_LAB_CLEAN_ROOM")
    if not marker:
        fields["numerical_result"] = "Queue not running inside the clean-room command."
        fields["uncertainty"] = "not applicable"
        fields["unresolved_assumptions"] = ["This run was not a clean-room reproduction."]
        return {"state": "partial", "fields": fields, "findings": [
            finding("This queue run executed inside the clean-room reproduction", "computational_pipeline", False, {},
                    expected_not_established=True)]}
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
    fields["numerical_result"] = f"Clean-room evidence for wheel {wheel[:16]}: {observed}."
    fields["uncertainty"] = "Comparison with retained reports is performed by the command after this run."
    # The interpreter is observed, never taken from the marker.
    fields["provider_runtime_identity"] = {"implementation": "ciw.lab", "wheel_sha256": wheel,
                                           "python": platform.python_version(), "ciw_version": __version__}
    verified = not failures
    return {"state": "completed" if verified else "partial", "fields": fields, "findings": [
        finding("This queue run executed inside the clean-room reproduction", "computational_pipeline", verified,
                {"checks": [_check("clean-room conditions failing (wheel digest, package files equal to the wheel's, "
                                   "isolated venv, installed package)",
                                   len(failures))]} if verified else {},
                expected_not_established=not verified)]}


# --------------------------------------------------------------- T165
def _release(reports):
    states = Counter(r["state"] for r in reports)
    labels = Counter(f["evidence_status"] for r in reports for f in r["findings"])
    providers = sorted({json.dumps(r["provider_runtime_identity"].get("provider") or r["provider_runtime_identity"].get("repository"))
                        for r in reports if isinstance(r["provider_runtime_identity"], dict)
                        and (r["provider_runtime_identity"].get("provider") or r["provider_runtime_identity"].get("repository"))})
    identities = [[r["task_id"], r["report_id"]] for r in reports]
    return {"schema": "ciw.lab-release-report.v1", "ciw_version": __version__, "tasks": len(reports),
            "states": dict(sorted(states.items())), "labels": {label: labels.get(label, 0) for label in LABELS},
            "providers": providers,
            "release_digest": "sha256:" + hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
            "physical_validation": "not_established" if all(r["physical_validation_status"]["status"] == "not_established"
                                                           for r in reports) else "mixed"}


@task("T165", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def release_report(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "A release report can be generated entirely from machine-readable queue metadata and retained reports.",
        "Release digest = SHA-256 over sorted (task_id, report_id); totals over states and labels.",
        "Aggregate states, labels, providers and report identities into JSON and Markdown.",
        "Sign the release digest with a project key once key custody is defined.")
    if not prior:
        return _no_prior(fields)
    release = _release(prior)
    states, labels = Counter(release["states"]), Counter(release["labels"])
    _, retained = _recomputed(ctx, "release-report.json", release, _release)
    lines = ["# Lab release report", "", f"- CIW version: {__version__}", f"- Tasks reported: {len(prior)}",
             f"- Release digest: `{release['release_digest']}`", f"- Physical validation: `{release['physical_validation']}`", "",
             "| State | Tasks |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in release["states"].items()] + [
             "", "| Evidence label | Findings |", "| --- | --- |"] + [f"| `{k}` | {v} |" for k, v in release["labels"].items()]
    ctx.artifact_text("RELEASE.md", "\n".join(lines) + "\n")
    fields["numerical_result"] = f"{len(prior)} reports; states {dict(states)}; labels {dict(labels)}."
    fields["uncertainty"] = "Exact aggregation."
    findings = [finding("Release report reproduces from retained reports", "provenance", len(prior),
                        {"checks": [retained]}, unit="reports", tolerance={"abs": 0, "rel": 0})]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T166
@task("T166", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def unresolved_assumptions(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "Every unresolved assumption stated by any task can be listed in one ledger with its source task.",
        "Ledger = union over reports of unresolved_assumptions, deduplicated with task references.",
        "Collect, deduplicate and count assumptions; write JSON and Markdown ledgers.",
        "Attach each assumption to the experiment that would resolve it and track closure.")
    if not prior:
        return _no_prior(fields)
    def build(reports):
        ledger = {}
        for report in reports:
            items = report["unresolved_assumptions"]
            for item in (items if isinstance(items, list) else [items]):
                ledger.setdefault(str(item), []).append(report["task_id"])
        return [{"assumption": k, "tasks": v} for k, v in sorted(ledger.items())]
    rows = build(prior)
    _, retained = _recomputed(ctx, "unresolved-assumptions.json", rows, build)
    ctx.artifact_text("UNRESOLVED_ASSUMPTIONS.md", "# Unresolved assumptions\n\n" + "\n".join(
        f"- {row['assumption']} ({', '.join(row['tasks'])})" for row in rows) + "\n")
    silent = [r["task_id"] for r in prior if not r["unresolved_assumptions"]]
    fields["numerical_result"] = f"{len(rows)} distinct unresolved assumptions from {len(prior)} reports; {len(silent)} reports state none."
    fields["uncertainty"] = "Exact aggregation; completeness depends on each task's honesty."
    fields["unresolved_assumptions"] = ([f"Reports stating no unresolved assumption: {', '.join(silent)}"] if silent else [])
    findings = [finding("Unresolved-assumption ledger reproduces from retained reports", "provenance", len(rows),
                        {"checks": [retained]}, unit="assumptions", tolerance={"abs": 0, "rel": 0})]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T167
@task("T167", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_catalogue_tasks_aggregate_retained_reports",))
def unmeasured(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "Everything that remains physically unmeasured is enumerable from the retained reports.",
        "Unmeasured = physical/authority-domain findings not established, plus blocked or deferred tasks.",
        "Collect those findings and task states into JSON and Markdown.",
        "Execute the manufacturing measurement protocols (T126-T128) on hardware and retain raw data.")
    if not prior:
        return _no_prior(fields)
    def build(reports):
        return {"not_established_claims": [{"task_id": r["task_id"], "claim": f["claim"], "domain": f["domain"]}
                                           for r in reports for f in r["findings"]
                                           if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS
                                           and f["evidence_status"] == "not_established"],
                "blocked_or_deferred_tasks": [{"task_id": r["task_id"], "state": r["state"],
                                               "reason": str(r["experiment"])[:300]}
                                              for r in reports if r["state"] in ("blocked", "deferred")],
                "hardware_measured_findings": [r["task_id"] for r in reports for f in r["findings"]
                                               if f["evidence_status"] == "hardware_measured"]}
    document = build(prior)
    claims, stalled, measured = (document["not_established_claims"], document["blocked_or_deferred_tasks"],
                                 document["hardware_measured_findings"])
    _, retained = _recomputed(ctx, "unmeasured.json", document, build)
    lines = ["# What remains unmeasured", "", f"Hardware-measured findings in this run: {len(measured)}.", "",
             "## Physical and authority claims not established", ""] + [
        f"- {c['task_id']} [{c['domain']}]: {c['claim']}" for c in claims] + [
        "", "## Blocked or deferred tasks", ""] + [f"- {s['task_id']} ({s['state']}): {s['reason']}" for s in stalled]
    ctx.artifact_text("UNMEASURED.md", "\n".join(lines) + "\n")
    fields["numerical_result"] = f"{len(claims)} physical/authority claims not established; {len(stalled)} blocked or deferred tasks; {len(measured)} hardware-measured findings."
    fields["uncertainty"] = "Exact aggregation."
    findings = [
        finding("Unmeasured ledger reproduces from retained reports (physical and authority claims)", "provenance", len(claims),
                {"checks": [retained]}, unit="claims", tolerance={"abs": 0, "rel": 0}),
        finding("Physical validity of the lab's computational results", "physical", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# --------------------------------------------------------------- T168
def _test_names(tests_dir: Path) -> set:
    """pytest node ids of the test functions and test-class methods (sync or async) under ``tests_dir``."""
    def collect(prefix, body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                names.add(f"{prefix}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                collect(f"{prefix}::{node.name}", node.body)

    names = set()
    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue  # pytest cannot collect it either; its node ids stay dangling
        collect(path.relative_to(tests_dir.parent).as_posix(), tree.body)
    return names


@task("T168", changed_files=(MODULE,), regression_tests=(f"{TESTS}::test_regression_coverage_is_checked",))
def permanent_regression_tests(ctx):
    prior = _prior_reports(ctx)
    fields = _common(
        "Every completed or partial experiment is backed by at least one existing pytest regression test, and the "
        "whole queue is guarded by a tolerance-aware comparison with retained reports.",
        "Coverage = tasks whose registered regression_tests all resolve to test functions in tests/.",
        "Resolve every registered node id against test function definitions; count passed/failed outcomes from the "
        "JUnit record when supplied.",
        "Run scripts/check_lab.py in CI so ciw lab verify guards every retained finding.")
    from .runner import repository_path
    tests_dir = repository_path("tests")  # never the current directory: it may hold another project's tests
    if not prior:
        return _no_prior(fields)
    if tests_dir is None or not tests_dir.is_dir():
        fields["numerical_result"] = "Registered regression node ids were not resolved."
        fields["uncertainty"] = "not applicable"
        fields["unresolved_assumptions"] = ["No repository tests/ directory is available to this installation "
                                            "(set CIW_LAB_REPOSITORY_ROOT to a checkout)."]
        return {"state": "partial", "fields": fields, "findings": [
            finding("Registered regression node ids that do not resolve to a test function", "computational_pipeline",
                    None, {}, expected_not_established=True)]}
    known = _test_names(tests_dir)
    implementations, _ = load_implementations()

    def build(reports):
        rows = []
        for report in reports:
            implementation = implementations.get(report["task_id"])
            nodes = list(implementation.regression_tests) if implementation else []
            rows.append({"task_id": report["task_id"], "state": report["state"], "regression_tests": nodes,
                         "missing": [n for n in nodes if n.split("[")[0] not in known],
                         "passed": [t for t in report["tests_passed"] if t.startswith("pytest:")],
                         "failed": report.get("tests_failed", [])})
        return rows
    rows = build(prior)
    uncovered = [r["task_id"] for r in rows if r["state"] in ("completed", "partial") and not r["regression_tests"]]
    dangling = [f"{r['task_id']}: {n}" for r in rows for n in r["missing"]]
    _, retained = _recomputed(ctx, "regression-coverage.json", rows, build)
    failed = sum(len(r["failed"]) for r in rows)
    fields["numerical_result"] = (f"{len(rows)} reports; {len(uncovered)} completed/partial tasks without regression tests; "
                                  f"{len(dangling)} dangling node ids; {failed} failed test records.")
    fields["uncertainty"] = "Static resolution of node ids; pass/fail only where a JUnit record was supplied."
    if dangling:
        fields["unresolved_assumptions"] = ["Dangling regression node ids: " + "; ".join(dangling[:20])]
    findings = [
        finding("Completed or partial tasks lacking a regression test", "computational_pipeline", len(uncovered),
                {"checks": [retained, _check("tasks without regression tests", len(uncovered))]},
                tolerance={"abs": 0, "rel": 0}),
        finding("Registered regression node ids that do not resolve to a test function", "computational_pipeline", len(dangling),
                {"checks": [_check("dangling node ids", len(dangling))]}, tolerance={"abs": 0, "rel": 0}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}

