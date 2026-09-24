import json
import math
from pathlib import Path
import platform

import pytest

from ciw.lab import research_portfolio, runner
from ciw.lab.evidence import DOMAINS, EvidenceRefusal, finding
from ciw.lab.registry import Implementation, load_implementations, load_queue
from ciw.lab.report import build_report, validate_report

SECTION = ("T155", "T156", "T157", "T158", "T159", "T160", "T161", "T162", "T163", "T164", "T165", "T166",
           "T167", "T168")
CHECK = {"reference_kind": "analytic", "reference": "closed form", "observed": 0.0, "tolerance": 1e-9, "passed": True}
PIPE_CLAIM = "Unit-speed drift max|g(v,v) - 1| stays below the bound"
TINY = 1.1122324405657753e-10
ROUNDOFF = 1.7763568394002505e-15
ORACLE_CLAIM = ("The evidence-label function agrees with the reference oracle for rules 1-5 on the exhaustive basis "
                "grammar")
DRAFT_TABLES = "Draft finding tables restate every retained finding of their sections with its retained label and basis"
# Twelve numbered evidence rules naming the anchors T155 requires, as the specification's section states them.
RULES = [f"{n}. Rule {n}." for n in range(1, 10)] + [
    "10. Authority wording: `evidence.screen_authority_claim` closes the loophole T141 found.",
    "11. Basis components: `evidence.basis_origin`.", "12. Workspace classification: T100's fabricated bundle."]


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _retain(directory, task_id, state, findings, extra=None, **fields):
    answers = {"hypothesis": f"h {task_id}", "experiment": "e", "unresolved_assumptions": [f"assumption of {task_id}"],
               "changed_files": ["src/ciw/lab/geodesic_jacobi.py"], **fields}
    built = build_report(_queue()[task_id], state, answers, findings, extra=extra)
    (directory / "reports").mkdir(parents=True, exist_ok=True)
    (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(built), encoding="utf-8")
    return built


@pytest.fixture
def retained(tmp_path):
    """Five retained reports: a counterexample, a pipe claim, an analytic, a provider-backed and a hardware-blocked task."""
    counter = finding("Separation is not monotone past the focus", "numerical", -0.5,
                      {"generator": {"name": "g"}, "checks": [CHECK]},
                      uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "binary64"}, tolerance={"abs": 1e-9, "rel": 0},
                      counterexample={"statement": "separation grows with length", "witness": {"s": 4.0}})
    drift = finding(PIPE_CLAIM, "numerical", {"drift": TINY, "orders": [4.16, 3.962, 4.0, 4.01, 3.99]},
                    {"checks": [CHECK]}, unit="m|s", tolerance={"abs": 0, "rel": 1e-6},
                    uncertainty={"kind": "roundoff", "value": ROUNDOFF, "basis": "binary64 | eps"})
    orders = finding("Atlas integration keeps fourth-order convergence", "numerical", [4.16, 3.962], {"checks": [CHECK]},
                     tolerance={"abs": 0.1, "rel": 0})
    _retain(tmp_path, "T010", "completed", [counter, drift, orders],
            mathematical_model="Jacobi equation integrated with RK4")
    _retain(tmp_path, "T021", "completed",
            [finding("Heading sensitivity equals path length on a flat torus", "mathematical", 1.0,
                     {"derivation": "K = 0"}, uncertainty={"kind": "exact", "value": 0, "basis": "closed form"},
                     tolerance={"abs": 0, "rel": 0})])
    provider = {"repository": "giasonpooni/Scientific-Computation-Runtime", "revision": "a" * 40,
                "source_tree": "b" * 40, "executed": True}
    _retain(tmp_path, "T098", "completed",
            [finding("Bundle digests agree with the pinned runtime", "provenance", "sha256:x", {"provider": provider})],
            provider_runtime_identity={"ciw": {"implementation": "ciw.lab"},
                                       "scr": {"head": "a" * 40, "tree": "b" * 40, "engine_sha256": "c" * 64},
                                       "requirement_probes": {"provider:scr": True}})
    _retain(tmp_path, "T116", "blocked", [finding("GPU energy per batch", "physical", None, {})],
            experiment="Blocked: unavailable requirement(s) hardware:nvidia-gpu.",
            provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {"hardware:nvidia-gpu": False}})
    _retain(tmp_path, "T147", "partial",
            [finding("CPU and GPU outputs agree under the tolerance policy on GPU hardware", "numerical", None, {},
                     expected_not_established=True),
             finding("Kernel is ready for industrial deployment", "industrial_readiness", None, {})],
            provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {"hardware:nvidia-gpu": False}},
            unresolved_assumptions=["The GPU half did not run: no GPU was probed."],
            recommended_next_task="Write a GPU kernel of the batched dot products, then rerun on a CUDA host.")
    return tmp_path


def _run(task_id, directory, junit=None, keep=False):
    implementations, _ = load_implementations()
    report = validate_report(runner.run_task(_queue()[task_id], implementations[task_id], runner.Context(directory),
                                             junit or {}))
    if keep:  # retained like a queue run, for the section tasks that follow
        (directory / "reports").mkdir(parents=True, exist_ok=True)
        (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(report), encoding="utf-8")
    return report


def _labels(report):
    return {f["claim"]: f["evidence_status"] for f in report["findings"]}


def _finding(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


# --------------------------------------------------------------- T155
@pytest.mark.lab_task("T155")
def test_label_function_matches_the_reference_oracle_exhaustively():
    result = research_portfolio.label_invariants()
    assert result["grammar"] == {"derivation": 2, "generator": 2, "checks": 4, "provider": 3, "independent_check": 8,
                                 "acquisition": 2}
    assert result["cases"] == 2 * 2 * 4 * 3 * 8 * 2 * len(DOMAINS)
    assert result["violations"] == [] and result["violation_count"] == 0
    # Every rule branch of the oracle is exercised, refusals and precedence included.
    assert result["unexercised_branches"] == []
    assert set(result["rule_branches"]) == set(research_portfolio.RULE_BRANCHES)
    counts = result["label_counts"]
    assert all(counts[label] for label in ("refused", "hardware_measured", "provider_backed", "synthetic", "analytic",
                                           "independently_verified", "numerically_verified", "not_established"))


@pytest.mark.parametrize("mutation", ["swap_precedence", "ignore_independent", "accept_same_origin", "physical_without_acquisition"])
def test_label_oracle_catches_a_label_function_that_departs_from_the_rules(monkeypatch, mutation):
    original = research_portfolio.supported_label

    def mutated(basis, domain):
        if mutation == "swap_precedence":
            label = original(basis, domain)
            return {"numerically_verified": "synthetic", "analytic": "independently_verified"}.get(label, label)
        if mutation == "ignore_independent":
            return original({k: v for k, v in basis.items() if k != "independent_check"}, domain)
        if mutation == "accept_same_origin":
            try:
                return original(basis, domain)
            except EvidenceRefusal:
                return "independently_verified"
        if domain in research_portfolio.SPEC_PHYSICAL and basis.get("checks") and "acquisition" not in basis:
            return "numerically_verified"
        return original(basis, domain)

    monkeypatch.setattr(research_portfolio, "supported_label", mutated)
    assert research_portfolio.label_invariants()["violation_count"] > 0


@pytest.mark.lab_task("T155")
def test_formal_specifications_cover_the_queue(retained, tmp_path):
    report = _run("T155", retained)
    labels = _labels(report)
    assert labels[ORACLE_CLAIM] == "numerically_verified"
    assert labels["Every computational queue task is named by a specification document"] == "numerically_verified"
    # The specification of record states rules 1-12, the authority screen with T141's loophole among them.
    assert labels[research_portfolio.RULES_CLAIM] == "numerically_verified"
    assert _finding(report, research_portfolio.RULES_CLAIM)["value"] >= 12
    assert _finding(report, "Every computational queue task")["value"] == 0
    # Five retained reports cannot exercise every specification unit: the coverage claim is refuted, not vacuous.
    assert labels["Specification documents are exercised by retained established findings"] == "not_established"
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert "14 domains" not in report["experiment"] and f"x {len(DOMAINS)} domains" in report["experiment"]
    # Without earlier reports the coverage finding is honestly unestablished, never passed by a zero threshold.
    empty = _run("T155", tmp_path / "empty")
    coverage = _finding(empty, "Specification documents are exercised")
    assert coverage["expected_not_established"] is True and coverage["value"] is None and coverage["basis"] == {}
    assert empty["state"] == "partial" and empty["evidence_status"]["primary"] == "numerically_verified"


def test_formal_specifications_complete_when_every_unit_is_exercised(retained, monkeypatch):
    queue = load_queue()
    monkeypatch.setattr(research_portfolio, "load_queue",
                        lambda: dict(queue, tasks=[t for t in queue["tasks"] if t["id"] in ("T010", "T021")]))
    monkeypatch.setattr(research_portfolio, "specification_documents",
                        lambda: {"units": {"SPECIFICATIONS.md: A": {"T010"}, "B.md": {"T021"}}, "missing_pages": [],
                                 "rules": RULES})
    report = _run("T155", retained)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"
    # Whether a unit restates a refuted statement is a review question, recorded honestly as not established.
    restated = _finding(report, "No specification unit restates")
    assert restated["evidence_status"] == "not_established" and restated["expected_not_established"] is True
    assert {label for claim, label in _labels(report).items() if claim != restated["claim"]} == {"numerically_verified"}
    assert _finding(report, "Specification documents are exercised")["value"] == 2


@pytest.mark.lab_task("T155")
def test_rule_10_screen_closes_the_computational_domain_loophole(retained, monkeypatch):
    probe = research_portfolio.rule10_probe()
    assert probe["violation_count"] == 0 and probe["claims"] == len(research_portfolio.RULE10_PROBES)
    assert probe["cases"] == len(research_portfolio.RULE10_PROBES) * len(DOMAINS)
    # T141's acceptance statement is refused in every computational and physical domain and stays not_established
    # in the authority domains; its declined and ordinary counterparts keep their labels.
    asserted = sum(asserts for _, asserts in research_portfolio.RULE10_PROBES)
    assert probe["outcomes"]["refused"] == asserted * 7
    assert research_portfolio.RULE10_PROBES[0] == ("Coupon lot accepted for production", True)
    report = _run("T155", retained)
    screen = _finding(report, research_portfolio.RULE10_CLAIM)
    assert screen["evidence_status"] == "numerically_verified" and screen["value"]["violations"] == 0
    assert (retained / "artifacts" / "T155" / "rule10-probe.json").is_file()
    # The paraphrase limit is recorded, not hidden, and the claim states the probe it rests on, not every wording.
    assert probe["paraphrase"]["outcome"] == "passes the screen"
    assert any(research_portfolio.RULE10_PARAPHRASE in a for a in report["unresolved_assumptions"])
    assert research_portfolio.RULE10_CLAIM.startswith(
        f"The authority-wording screen refuses each of the {asserted} hand-classified authority statements of its "
        f"probe (T141's acceptance statement first) in the 7 computational and physical domains")
    assert f"the {len(research_portfolio.RULE10_PROBES) - asserted} declined or ordinary" in research_portfolio.RULE10_CLAIM
    assert "other authority outcomes" not in research_portfolio.RULE10_CLAIM
    # Without the screen (the loophole T141 found) the probe refutes rule 10.
    from ciw.lab import evidence
    monkeypatch.setattr(evidence, "screen_authority_claim", lambda claim, domain: None)
    loophole = _run("T155", retained)
    assert _labels(loophole)[research_portfolio.RULE10_CLAIM] == "not_established"
    assert loophole["evidence_status"]["primary"] == "not_established"


@pytest.mark.lab_task("T155", "T157")
def test_next_steps_name_open_work_rather_than_work_done_elsewhere(retained):
    # T155 already enumerates the finite grammar, so a SAT encoding over it adds nothing: the open part is
    # off-grammar bases (property-based tests or Lean).
    step = _run("T155", retained)["recommended_next_task"]
    assert "SAT" not in step and "Lean" in step and "property-based" in step and "off the finite grammar" in step
    # Counterexamples are compared by ciw lab verify in CI; T168 creates no regression fixtures.
    step = _run("T157", retained)["recommended_next_task"]
    assert "ciw lab verify" in step and "(T168)" not in step and "regression fixture" not in step


@pytest.mark.lab_task("T155")
def test_specification_units_list_the_counterexamples_of_their_tasks(retained, monkeypatch):
    units = {"SPECIFICATIONS.md: A": {"T010"}, "B.md": {"T021"}}
    monkeypatch.setattr(research_portfolio, "specification_documents",
                        lambda: {"units": units, "missing_pages": [], "rules": RULES})
    report = _run("T155", retained)
    coverage = json.loads((retained / "artifacts" / "T155" / "specification-coverage.json").read_text(encoding="utf-8"))
    unit = coverage["units"]["SPECIFICATIONS.md: A"]
    assert unit["established_findings"] == {"T010": 3}
    assert unit["counterexample_statements"] == [{
        "task_id": "T010", "statement": "separation grows with length",
        "claim": "Separation is not monotone past the focus", "evidence_status": "numerically_verified"}]
    assert coverage["units"]["B.md"]["counterexample_statements"] == []
    assert "review question" in coverage["note"]
    # Coverage counts exercised units; whether a unit restates a refuted statement is not claimed.
    restated = _finding(report, "No specification unit restates")
    assert restated["evidence_status"] == "not_established" and restated["value"] is None
    assert "1 units name tasks that refute 1 general statements" in report["numerical_result"]
    # A specification whose rules omit the authority screen, or number them out of order, refutes the rules claim.
    for rules in (RULES[:9], RULES[:9] + RULES[10:], [RULES[1], RULES[0]] + RULES[2:]):
        monkeypatch.setattr(research_portfolio, "specification_documents",
                            lambda rules=rules: {"units": units, "missing_pages": [], "rules": rules})
        assert _labels(_run("T155", retained))[research_portfolio.RULES_CLAIM] == "not_established", rules


def test_specification_chord_paragraph_is_not_restated_as_refuted():
    """T046's counterexamples: the s^4 term with start-point curvature, and the circle formula only for tau = 0."""
    section = research_portfolio._specification_section("Chord versus geodesic distance")
    if section is None:
        pytest.skip("docs/lab is not reachable from this installation")
    flat = " ".join(section.split())
    # The s^4 term with start-point curvature and its rate, and the circle formula qualified by zero torsion.
    assert "s⁴" in flat and "κ₀′" in flat and "leading order" in flat
    assert "τ = 0" in flat and "720" in flat
    ledger = {result for result, _, _ in research_portfolio.TEXTBOOK}
    assert any(result.startswith("Chord-arc expansion") and "leading order" in result for result in ledger)


# --------------------------------------------------------------- T156
@pytest.mark.lab_task("T156")
def test_textbook_and_contribution_ledger(retained, monkeypatch):
    report = _run("T156", retained)
    labels = _labels(report)
    assert labels["Contributions are novel relative to the literature"] == "not_established"
    assert labels["Every retained task is attributed to a textbook result or a present implementation source file"] \
        == "numerically_verified"
    # Most ledger entries name results these five tasks never use: that claim is refuted here.
    assert labels["Every textbook result in the ledger is named by a retained task"] == "not_established"
    assert report["state"] == "partial"
    ledger = json.loads((retained / "artifacts" / "T156" / "attribution-ledger.json").read_text(encoding="utf-8"))
    rows = {row["result"]: row["tasks"] for row in ledger["textbook"]}
    assert "T010" in rows["Jacobi equation j'' + K j = 0; conjugate and focal points"]
    # Methods retained tasks name (compensated summation, fast marching, Bartels-Stewart, Monte Carlo) are in the ledger.
    names = " ".join(result + reference for result, reference, _ in research_portfolio.TEXTBOOK)
    assert all(name in names for name in ("Kahan", "Dijkstra", "Bartels", "Monte Carlo", "Cholesky"))
    # A task with no textbook name and no package source is unattributed and refutes the hypothesis.
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], changed_files=[])
    small = (("Jacobi equation", "do Carmo", r"\bJacobi\b"), ("Flat torus sensitivity", "folklore", r"(?i:flat torus)"))
    monkeypatch.setattr(research_portfolio, "TEXTBOOK", small)
    report = _run("T156", retained)
    assert _finding(report, "Every retained task is attributed")["value"] == 1
    assert _labels(report)["Every retained task is attributed to a textbook result or a present implementation source file"] \
        == "not_established"
    (retained / "reports" / "T023.json").unlink()
    report = _run("T156", retained)
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"


# --------------------------------------------------------------- T157
@pytest.mark.lab_task("T157")
def test_counterexample_catalogue(retained):
    catalogue = _run("T157", retained)
    record = catalogue["findings"][0]
    assert record["value"] == 1 and record["evidence_status"] == "numerically_verified"
    assert catalogue["state"] == "completed" and catalogue["unresolved_assumptions"]
    entries = json.loads((retained / "artifacts" / "T157" / "counterexamples.json").read_text(encoding="utf-8"))
    assert entries[0]["statement"] == "separation grows with length" and entries[0]["task_id"] == "T010"
    # The raw-text recount is a second path: a counterexample key the traversal does not see refutes the catalogue.
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})],
            input_data=[{"counterexample": {"statement": "outside any finding"}}])
    refuted = _run("T157", retained)
    assert refuted["findings"][0]["evidence_status"] == "not_established" and refuted["state"] == "partial"


# --------------------------------------------------------------- T158
def _figure_task(task_id, timing=False, varying=False):
    from ciw.lab import svg
    calls = iter(range(1, 100))

    def figure(ctx):
        scale = next(calls) if varying else 1
        if timing:
            ctx.artifact_json("timings.json", {"note": "Wall-clock timings of this run; not reproducible.", "seconds": scale})
        ctx.artifact_text("plot.svg", svg.line_plot([("a", [1, 2, 3], [1, 4, 9 * scale])], title="t", xlabel="x",
                                                    ylabel="y"))
        return {"fields": {}, "findings": [finding("f", "numerical", 1.0, {"generator": {"name": "g"}, "checks": [CHECK]},
                                                   uncertainty={"kind": "exact", "value": 0, "basis": "b"},
                                                   tolerance={"abs": 0, "rel": 0})]}
    return Implementation(task_id, figure)


def _retain_figures(directory, monkeypatch, fakes):
    for task_id, fake in fakes.items():
        saved = runner.run_task(_queue()[task_id], fake, runner.Context(directory), {})
        (directory / "reports").mkdir(exist_ok=True)
        (directory / "reports" / f"{task_id}.json").write_text(runner.dumps(saved))
    real, errors = load_implementations()
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({**real, **fakes}, errors))
    monkeypatch.setattr(research_portfolio, "REGENERATED", tuple(fakes))


@pytest.mark.lab_task("T158")
def test_figures_are_reproducible(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010")})
    report = _run("T158", tmp_path)
    values = {f["claim"]: f["value"] for f in report["findings"]}
    assert values["Re-executed figure tasks without wall-clock timings regenerate byte-identical figures"] == 0
    assert set(_labels(report).values()) == {"numerically_verified"}
    assert report["state"] == "completed" and report["evidence_status"]["primary"] == "numerically_verified"


@pytest.mark.lab_task("T158")
def test_a_changed_figure_is_a_mismatch_and_a_timing_figure_a_counterexample(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010", varying=True),
                                            "T013": _figure_task("T013", timing=True, varying=True)})
    report = _run("T158", tmp_path)
    labels = _labels(report)
    # A timing-free figure that changes refutes byte reproducibility.
    assert labels["Re-executed figure tasks without wall-clock timings regenerate byte-identical figures"] == "not_established"
    assert _finding(report, "Re-executed figure tasks without")["value"] == 1
    # A timing figure that changes is a counterexample to the hypothesis, established by its check.
    counter = _finding(report, "Re-executed figures that plot wall-clock timings differ")
    assert counter["evidence_status"] == "numerically_verified" and counter["value"] == 1
    assert counter["counterexample"]["witness"] == {"figures": ["artifacts/T013/plot.svg"]}
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    # An edited retained figure fails its digest.
    (tmp_path / "artifacts" / "T010" / "plot.svg").write_text("<svg></svg>")
    edited = _run("T158", tmp_path)
    assert _labels(edited)["Retained figures hash to the digests their reports record"] == "not_established"


def test_figures_not_reexecuted_leave_the_task_partial(tmp_path, monkeypatch):
    _retain_figures(tmp_path, monkeypatch, {"T010": _figure_task("T010")})
    monkeypatch.setattr(research_portfolio, "REGENERATED", ())
    report = _run("T158", tmp_path)
    assert report["state"] == "partial"
    assert _finding(report, "Re-executed figure tasks without")["expected_not_established"] is True
    assert any("not re-executed" in a and "T010" in a for a in report["unresolved_assumptions"])


# --------------------------------------------------------------- T159
@pytest.mark.lab_task("T159")
def test_uncertainty_table_restates_every_numerical_finding(retained):
    report = _run("T159", retained)
    assert set(_labels(report).values()) == {"numerically_verified"} and report["state"] == "completed"
    # Scalar, object and list values all count; the list without an uncertainty is identified.
    assert _finding(report, "Uncertainty table restates")["value"] == 4
    assert _finding(report, research_portfolio.LACKING_CLAIM)["value"] == 1
    text = (retained / "artifacts" / "T159" / "uncertainty-budget.md").read_text(encoding="utf-8")
    rows = research_portfolio._table_rows(text, research_portfolio.UNCERTAINTY_HEADER)
    assert all(len(cells) == 8 for cells in rows)
    drift = next(cells for cells in rows if cells[1] == PIPE_CLAIM)
    assert float(drift[5]) == ROUNDOFF and drift[5] == repr(ROUNDOFF) and drift[6] == "binary64 | eps"
    assert str(TINY) in drift[2] and drift[3] == "m|s"
    rows_json = json.loads((retained / "artifacts" / "T159" / "uncertainty-budget.json").read_text(encoding="utf-8"))
    assert research_portfolio._uncertainty_table_problems(text, rows_json) == []
    # A cell cut inside a number is caught by the parse-back.
    cut = text.replace(repr(ROUNDOFF), repr(ROUNDOFF)[:9])
    assert research_portfolio._uncertainty_table_problems(cut, rows_json)


@pytest.mark.lab_task("T159")
def test_uncertainty_listing_separates_unmeasured_records(retained):
    # A physical claim recorded as not established with a count and no basis measured nothing (T139's "0 records").
    _retain(retained, "T139", "partial",
            [finding("A real measurement with raw bytes has been retained", "physical", 0, {}),
             finding("Retention digests agree", "provenance", 3, {"checks": [CHECK]},
                     uncertainty={"kind": "exact", "value": 0, "basis": "count"}, tolerance={"abs": 0, "rel": 0})])
    report = _run("T159", retained)
    assert set(_labels(report).values()) == {"numerically_verified"} and report["state"] == "completed"
    # Only the measured finding without an uncertainty is counted; the record is listed with its own kind.
    assert _finding(report, research_portfolio.LACKING_CLAIM)["value"] == 1
    assert _finding(report, "Uncertainty table restates")["value"] == 6
    rows = json.loads((retained / "artifacts" / "T159" / "uncertainty-budget.json").read_text(encoding="utf-8"))
    record = next(row for row in rows if row["task_id"] == "T139" and row["uncertainty"] is None)
    assert record["measured"] is False
    text = (retained / "artifacts" / "T159" / "uncertainty-budget.md").read_text(encoding="utf-8")
    cells = next(c for c in research_portfolio._table_rows(text, research_portfolio.UNCERTAINTY_HEADER) if c[0] == "T139"
                 and c[1].startswith("A real measurement"))
    assert cells[4] == research_portfolio.UNMEASURED_KIND
    assert any("1 numerical findings record an unestablished claim with no basis" in a
               for a in report["unresolved_assumptions"])
    # The split is recounted from the raw text: a traversal that counts the record as measured is caught.
    assert research_portfolio._uncertainty_table_problems(text, rows) == []
    assert research_portfolio._uncertainty_table_problems(text, [dict(row, measured=True) for row in rows])


@pytest.mark.lab_task("T159")
def test_uncertainty_listing_names_the_per_quantity_budgets_it_holds(tmp_path):
    exact = {"kind": "exact", "value": 0, "basis": "closed form"}
    _retain(tmp_path, "T021", "completed", [finding("Heading sensitivity equals path length on a flat torus",
                                                    "mathematical", 1.0, {"derivation": "K = 0"}, uncertainty=exact)])
    report = _run("T159", tmp_path)
    assert "No retained finding declares a budget of components of one quantity." in report["unresolved_assumptions"][0]
    assert "a per-quantity budget uses" in report["recommended_next_task"]
    # T140's budget holds separate components of one predicted quantity: the listing names it, and its next step
    # is the deferred budget question in that form, not a request for components on every finding.
    budget = finding("Uncertainty budget per predicted quantity and its limiting term", "numerical",
                     {"marker gap": {"instrument": 1e-3, "geometry": 2e-3, "execution": 5e-4, "solver": 1e-9}},
                     {"checks": [CHECK]}, uncertainty={"kind": "reference_error", "value": 2e-3, "basis": "limiting term"},
                     tolerance={"abs": 0, "rel": 1e-6})
    _retain(tmp_path, "T140", "completed", [budget])
    report = _run("T159", tmp_path)
    assert set(_labels(report).values()) == {"numerically_verified"} and report["state"] == "completed"
    assumptions = " ".join(report["unresolved_assumptions"])
    assert "Budgets of one quantity's components are declared inside the findings of T140" in assumptions
    assert "declares separate components" not in assumptions and "No retained finding declares a budget" not in assumptions
    assert report["recommended_next_task"].startswith("None open in this listing: every measured numerical finding")
    assert report["recommended_next_task"].endswith("Deferred research question: combine per-quantity components into "
                                                    "budgets where one predicted quantity has several sources, in the "
                                                    "form T140 uses.")


@pytest.mark.lab_task("T159", "T165")
def test_raw_recounts_derive_the_components_of_findings_retained_without_origin(retained):
    """Rule 11: a finding retained before basis components were recorded has no origin key; readers derive it."""
    from ciw.lab.evidence import finding_origin
    from ciw.lab.report import report_identity
    _retain(retained, "T139", "partial", [finding("A real measurement with raw bytes has been retained", "physical", 0, {})])
    for path in (retained / "reports").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        for record in report["findings"]:
            record.pop("origin")
        report["report_id"] = report_identity(report)
        path.write_text(runner.dumps(report), encoding="utf-8")
        text = path.read_text(encoding="utf-8")
        assert '"origin":' not in text
        blocks = research_portfolio._raw_findings(text)
        assert [research_portfolio._raw_block_components(b) for b in blocks] == [
            finding_origin(record) for record in report["findings"]]
    uncertainty = _run("T159", retained)
    assert set(_labels(uncertainty).values()) == {"numerically_verified"} and uncertainty["state"] == "completed"
    assert _finding(uncertainty, research_portfolio.LACKING_CLAIM)["value"] == 1
    release = _run("T165", retained)
    assert _labels(release)["Release state and label totals match a raw-text recount of the retained report files"] \
        == "numerically_verified"
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    assert record["basis_components"]["provider"] == 1 and record["basis_components"]["none"] == 4


def test_headline_never_cuts_a_number():
    value = {"b": [TINY, 2.5e300, -3.0, 4, 5], "a": 1.7763568394002505e-15, "c": "text 12", "d": True, "e": 7}
    cell = research_portfolio._headline(value)
    assert cell == '{"a": 1.7763568394002505e-15, "c": "text 12", "d": true, "e": 7 …(+5)}'
    assert research_portfolio._stated_numbers(cell) == [1.7763568394002505e-15, 7.0]
    assert not research_portfolio._headline_problem(cell, value)
    assert research_portfolio._headline_problem(cell.replace("e-15", "e-1"), value)
    assert research_portfolio._headline([TINY, 1, 2, 3, 4, 5]) == f"[{TINY!r}, 1, 2, 3 …(+2)]"
    # Nested values show their leading numbers with paths, never an empty placeholder.
    nested = {"euler": {"order": 1.02, "errors": [0.5, 0.25]}, "rk4": {"order": 3.99}}
    assert research_portfolio._headline(nested) == \
        '{"euler.order": 1.02, "euler.errors[0]": 0.5, "euler.errors[1]": 0.25, "rk4.order": 3.99}'
    assert research_portfolio._headline([{"x": 1}, {"x": 2e-300}]) == '{"[0].x": 1, "[1].x": 2e-300}'
    assert research_portfolio._headline({}) == "{}" and research_portfolio._headline([]) == "[]"


# --------------------------------------------------------------- T160-T162
@pytest.mark.lab_task("T160", "T161", "T162")
@pytest.mark.parametrize("task_id", ["T160", "T161", "T162"])
def test_paper_drafts_trace_to_reports(retained, task_id):
    if task_id == "T162":
        _run("T155", retained, keep=True)
    report = _run(task_id, retained)
    assert report["state"] == "partial"
    labels = _labels(report)
    assert labels["Draft has passed external peer review"] == "not_established"
    assert labels[DRAFT_TABLES] == "numerically_verified"
    slug = research_portfolio.PAPERS[task_id][0]
    text = (retained / "artifacts" / task_id / f"{slug}-draft.md").read_text(encoding="utf-8")
    assert "Not peer reviewed" in text and report["hypothesis"].startswith(("An ", "A "))
    assert "physical validation is not established for any of them" in text
    for cells in research_portfolio._table_rows(text, research_portfolio.RESULTS_HEADER):
        assert cells is not None and len(cells) == 7 and cells[4].startswith("`") and cells[5]
    if task_id == "T161":
        assert "max\\|g(v,v) - 1\\|" in text and PIPE_CLAIM in [c[1] for c in research_portfolio._table_rows(
            text, research_portfolio.RESULTS_HEADER)]
    if task_id == "T162":
        assert "weakest" in text and "supported_label" in text and "raw_sha256" in text
        assert labels[research_portfolio.RULES_BACKED_CLAIM] == "numerically_verified"


SECTIONS = ("Abstract", "Introduction", "Methods", "Results", "Discussion", "Limitations", "Outstanding work",
            "References", "Appendix: every finding")


@pytest.mark.lab_task("T160", "T161", "T162")
@pytest.mark.parametrize("task_id", ["T160", "T161", "T162"])
def test_paper_drafts_have_a_structure_and_qualify_their_labels(retained, task_id):
    # T010 cites a sympy reference written in ciw: its qualification must reach the draft's methods.
    qualification = "Both sympy references are curve models written in ciw.lab.observation_chord"
    counter = finding("Separation is not monotone past the focus", "numerical", -0.5,
                      {"generator": {"name": "g", "seed": 7}, "checks": [CHECK]},
                      uncertainty={"kind": "roundoff", "value": 1e-12, "basis": "binary64"}, tolerance={"abs": 1e-9, "rel": 0},
                      counterexample={"statement": "separation grows with length", "witness": {"s": 4.0}})
    _retain(retained, "T010", "completed", [counter], unresolved_assumptions=[qualification, "Noise is independent"],
            mathematical_model="Jacobi equation integrated with RK4")
    _retain(retained, "T141", "completed", [finding("Production acceptance of the coupon", "production_acceptance",
                                                    None, {})])
    if task_id == "T162":
        _run("T155", retained, keep=True)
    report = _run(task_id, retained)
    slug = research_portfolio.PAPERS[task_id][0]
    text = (retained / "artifacts" / task_id / f"{slug}-draft.md").read_text(encoding="utf-8")
    headings = [line[3:] for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)
    assert "*Thesis (generated from the retained findings).* Across" in text
    # independently_verified is defined, and every boundary row is stated.
    assert research_portfolio.INDEPENDENCE in text and research_portfolio._qualification_problems(text) == []
    assert research_portfolio._qualification_problems(text.replace("| A plausible use case | Actual customer demand |\n", ""))
    labels = _labels(report)
    assert labels[DRAFT_TABLES] == "numerically_verified"
    for claim in ("Draft defines every evidence label", "Draft limitations state what each unfinished task"):
        assert _finding(report, claim)["evidence_status"] == "numerically_verified", claim
    assert report["state"] == "partial" and any("conclusions" in a for a in report["unresolved_assumptions"])
    limitations = research_portfolio._section(text, "Limitations")
    methods = research_portfolio._section(text, "Methods")
    if task_id == "T160":
        # T116 is blocked: its line states what is missing (its assumption), not what it planned to do.
        assert "- T116 is blocked; missing or unresolved:\n  - assumption of T116" in limitations
        assert "Blocked: unavailable requirement(s)" not in limitations
    if task_id == "T161":
        assert f"  - {qualification}" in methods and "Noise is independent" not in methods
        assert "T010 refutes “separation grows with length”" in text and "synthetic inputs (g, seed 7)" in text
        assert "do Carmo (1992), Riemannian Geometry, ch. 5" in research_portfolio._section(text, "References")
    if task_id == "T162":
        assert "  - The GPU half did not run: no GPU was probed." in limitations
        # Rules 10-12 of the specification and the boundary findings of T141 are in the methods.
        assert "10. Authority wording" in methods and "T141 (" in methods
        rows = research_portfolio._table_rows(text, research_portfolio.BOUNDARY_FINDINGS_HEADER)
        assert [cells[1] for cells in rows] == ["Production acceptance of the coupon"]
        assert labels[research_portfolio.RULES_BACKED_CLAIM] == "numerically_verified"
        # The next step comes from the draft's own limitations, not from the manufacturing protocols.
        assert "T147:" in report["recommended_next_task"] and "manufacturing" not in report["recommended_next_task"]


SAME_ORIGIN = "Dual-number agreement is same-origin (ciw) evidence."
NO_SYMPY = ("Without sympy the variable-curvature references integrate the ciw equations (scipy) or reuse ciw RK4: the "
            "integrator may be independent, the equations are not")


@pytest.mark.lab_task("T161")
def test_drafts_carry_every_qualification_of_independently_verified_rows(retained):
    # T034's independently_verified row is qualified in hyphenated wording, and by a sentence no phrase names;
    # its statement about noise is statistical independence, not a qualification.
    iv = finding("Dual-number curvature agrees with sympy.diffgeom", "numerical", 0.0,
                 {"independent_check": {**CHECK, "producer": {"implementation": "ciw.lab.surfaces", "revision": "r"},
                                        "checker": {"implementation": "sympy.diffgeom", "revision": "1.13"}}},
                 uncertainty={"kind": "roundoff", "value": 1e-15, "basis": "binary64"}, tolerance={"abs": 1e-12, "rel": 0})
    unphrased = "The curvature of three surfaces is assembled independently of sympy.diffgeom from its derivatives."
    _retain(retained, "T034", "completed", [iv], unresolved_assumptions=[
        SAME_ORIGIN, unphrased, "Noise is isotropic and independent per vertex."])
    # T002 has no independently_verified row here (sympy absent): its no-sympy statement is selected by phrase.
    _retain(retained, "T002", "completed", [finding("Geodesic reference agrees", "numerical", 0.0, {"checks": [CHECK]})],
            unresolved_assumptions=[NO_SYMPY, "Step sizes are independent of the surface."])
    report = _run("T161", retained)
    text = (retained / "artifacts" / "T161" / "geometry-methods-draft.md").read_text(encoding="utf-8")
    methods = research_portfolio._section(text, "Methods")
    for item in (SAME_ORIGIN, unphrased, NO_SYMPY):
        assert f"  - {item}" in methods, item
    assert "Noise is isotropic" not in methods and "Step sizes are independent" not in methods
    qualified = _finding(report, research_portfolio.QUALIFIED_CLAIM)
    assert qualified["evidence_status"] == "numerically_verified"
    assert "in a task with independently_verified rows" in research_portfolio.QUALIFIED_CLAIM
    assert research_portfolio.INDEPENDENCE_LIMITS.search("x is same-origin evidence")
    assert research_portfolio.INDEPENDENCE_LIMITS.search("they come from ciw-written assembly of sympy derivatives")


@pytest.mark.lab_task("T162")
def test_draft_next_step_names_completed_tasks_with_open_physical_claims(retained):
    # T113 completed, but its bench claims stay open: the next step names it and does not say the unfinished
    # tasks' steps close the draft's limitations.
    bench = finding("Encoder residuals of the servo-axis bench fall inside the interval", "physical", None, {})
    _retain(retained, "T113", "completed", [bench, finding("g", "numerical", 1, {"checks": [CHECK]})],
            recommended_next_task="Deferred research question (hardware-gated): bind the residual adapter to acquired "
                                  "encoder data.")
    _run("T155", retained, keep=True)
    report = _run("T162", retained)
    step = report["recommended_next_task"]
    assert "limitations close" not in step and "T147: Write a GPU kernel" in step
    assert "1 physical, calibration or sensor claim of completed tasks (T113) stays not established" in step
    assert "authority-domain claim" in step
    text = (retained / "artifacts" / "T162" / "evidence-provenance-note-draft.md").read_text(encoding="utf-8")
    outstanding = research_portfolio._section(text, "Outstanding work")
    assert ("- T113 (1 physical claim not established; its own next step): Deferred research question "
            "(hardware-gated): bind the residual adapter") in outstanding


def test_label_definitions_match_the_lab_guide():
    from ciw.lab.evidence import LABELS
    assert [label for label, _, _ in research_portfolio.LABEL_MEANINGS] == list(LABELS)
    guide = runner.repository_path("docs", "LAB.md")
    if guide is None or not guide.is_file():
        pytest.skip("docs/LAB.md is not reachable from this installation")
    rows = research_portfolio._table_rows(guide.read_text(encoding="utf-8"), "| Label | Meaning | Produced by |")
    assert {cells[0]: cells[1] for cells in rows} == {f"`{label}`": meaning
                                                       for label, meaning, _ in research_portfolio.LABEL_MEANINGS}


def test_a_corrupted_draft_row_fails_the_citation_check(retained):
    _run("T161", retained)
    text = (retained / "artifacts" / "T161" / "geometry-methods-draft.md").read_text(encoding="utf-8")
    reports = [r for r in runner.load_reports(retained) if r["section"] in research_portfolio.PAPERS["T161"][2]]
    cited = [(r, f) for r in reports for f in r["findings"]]
    assert research_portfolio._draft_problems(text, cited) == []
    unescaped = text.replace("max\\|g(v,v) - 1\\|", "max|g(v,v) - 1|")
    appendix = text.index(research_portfolio.RESULTS_HEADER)
    relabelled = text[:appendix] + text[appendix:].replace("`numerically_verified`", "`independently_verified`", 1)
    rebased = text[:appendix] + text[appendix:].replace("reference checks, synthetic inputs (g)", "reference checks", 1)
    dropped = "\n".join(line for line in text.splitlines() if "Heading sensitivity" not in line)
    for corrupted in (unescaped, relabelled, rebased, dropped):
        assert research_portfolio._draft_problems(corrupted, cited)


# --------------------------------------------------------------- T163
@pytest.mark.lab_task("T163")
def test_portfolio_shows_every_label_in_use(retained):
    report = _run("T163", retained)
    record = report["findings"][0]
    assert record["evidence_status"] == "numerically_verified" and report["evidence_status"]["primary"] == "numerically_verified"
    # Only T010 of the curated panels is retained: panels are added for analytic, provider_backed and not_established.
    assert record["value"] == 4 and report["state"] == "partial"
    text = (retained / "artifacts" / "T163" / "PORTFOLIO.md").read_text(encoding="utf-8")
    shown = set(research_portfolio.SHOWN_LABEL.findall(text))
    assert shown == {"analytic", "numerically_verified", "provider_backed", "not_established"}
    assert "- GPU energy per batch: " in text


@pytest.mark.lab_task("T163")
def test_portfolio_qualifies_labels_and_states_customer_demand(retained):
    report = _run("T163", retained)
    assert _labels(report)[research_portfolio.PORTFOLIO_QUALIFIED] == "numerically_verified"
    text = (retained / "artifacts" / "T163" / "PORTFOLIO.md").read_text(encoding="utf-8")
    # Each shown finding carries its basis; the provider-backed one names its pinned provider.
    assert "→ `provider_backed`; basis: pinned provider run (giasonpooni/Scientific-Computation-Runtime@aaaaaaaaaaaa)" in text
    assert "→ `numerically_verified`; basis: reference checks, synthetic inputs (g)" in text
    assert research_portfolio.INDEPENDENCE in text and research_portfolio._qualification_problems(text) == []
    limitations = research_portfolio._section(text, "Limitations")
    assert "- Customer demand is not established: no retained finding is filed in the customer_demand domain" in limitations
    # A retained customer-demand claim is counted, and it stays not_established.
    _retain(retained, "T132", "completed", [finding("Manufacturers need curvature-aware placement checks",
                                                    "customer_demand", None, {})])
    report = _run("T163", retained)
    text = (retained / "artifacts" / "T163" / "PORTFOLIO.md").read_text(encoding="utf-8")
    assert "1 retained finding filed in the customer_demand domain (T132), each `not_established`" in text
    assert _labels(report)[research_portfolio.PORTFOLIO_QUALIFIED] == "numerically_verified"
    assert research_portfolio._qualification_problems(text.replace(research_portfolio.INDEPENDENCE, ""))


# --------------------------------------------------------------- T164
@pytest.mark.lab_task("T164")
def test_clean_room_marker_is_recognized(tmp_path, monkeypatch):
    monkeypatch.delenv("CIW_LAB_CLEAN_ROOM", raising=False)
    report = _run("T164", tmp_path)
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    assert any("--no-compare" in a for a in report["unresolved_assumptions"])
    assert "Retained reports" not in json.dumps(report["input_data"])
    monkeypatch.setenv("CIW_LAB_CLEAN_ROOM", json.dumps({"wheel_sha256": "a" * 64, "python": "2.7.18"}))
    report = _run("T164", tmp_path)
    # An asserted marker without a matching wheel and isolated install is not evidence.
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    # The interpreter is observed, never copied from the marker.
    assert report["provider_runtime_identity"]["python"] == platform.python_version()


def _wheel(path, package_dir, drop=(), extra=None, record=True):
    """A wheel-shaped zip of ``package_dir`` (bytecode caches aside), optionally altered."""
    import zipfile
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(package_dir.rglob("*")):
            name = file.relative_to(package_dir.parent).as_posix()
            if file.is_file() and "__pycache__" not in file.parts and name not in drop:
                archive.writestr(name, file.read_bytes())
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
        if record:
            archive.writestr("ciw-0.dist-info/RECORD", "")
    return path


def _clean_room(monkeypatch, wheel_path):
    import hashlib
    import sys
    import ciw
    # An isolated interpreter whose prefix holds the imported package, as in the clean room.
    monkeypatch.setattr(sys, "prefix", str(Path(ciw.__file__).resolve().parents[1]))
    monkeypatch.setattr(sys, "base_prefix", "/nonexistent-base-interpreter")
    monkeypatch.setenv("CIW_LAB_CLEAN_ROOM", json.dumps({
        "wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(), "wheel_path": str(wheel_path)}))


@pytest.mark.lab_task("T164")
def test_clean_room_needs_the_installed_package_to_be_the_named_wheel(tmp_path, monkeypatch):
    import ciw
    package = Path(ciw.__file__).resolve().parent
    # A forged marker names any file with its own digest; nothing ties it to the installed code.
    notes = tmp_path / "notes.txt"
    notes.write_text("not a wheel at all\n")
    _clean_room(monkeypatch, notes)
    report = _run("T164", tmp_path / "forged")
    assert report["state"] == "partial" and report["findings"][0]["evidence_status"] == "not_established"
    assert "'installed_from_wheel': False" in report["numerical_result"]
    # The wheel the installed package came from: every file present with the same bytes.
    _clean_room(monkeypatch, _wheel(tmp_path / "ciw-0-py3-none-any.whl", package))
    report = _run("T164", tmp_path / "genuine")
    assert report["state"] == "completed" and report["findings"][0]["evidence_status"] == "numerically_verified"
    observation = json.loads((tmp_path / "genuine" / "artifacts" / "T164" / "clean-room-observation.json").read_text())
    assert observation["conditions"]["installed_from_wheel"] is True and "wheel_sha256" not in json.dumps(observation)


@pytest.mark.lab_task("T164")
def test_clean_room_prose_does_not_carry_the_wheel_digest(tmp_path, monkeypatch):
    import hashlib
    import shutil
    import zipfile
    import ciw
    package = Path(ciw.__file__).resolve().parent
    first = _wheel(tmp_path / "ciw-0-py3-none-any.whl", package)
    # Another build of the same files: equal members, different bytes and digest (as file times do).
    second = tmp_path / "rebuilt" / first.name
    second.parent.mkdir()
    shutil.copy2(first, second)
    with zipfile.ZipFile(second, "a") as archive:
        archive.comment = b"second build"
    assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(second.read_bytes()).digest()
    for wheel, run in ((first, "run1"), (second, "run2")):
        _clean_room(monkeypatch, wheel)
        report = _run("T164", tmp_path / run)
        assert report["state"] == "completed" and _labels(report) == {research_portfolio.CLEAN_ROOM_CLAIM: "numerically_verified"}
        assert report["provider_runtime_identity"]["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
        (tmp_path / run / "reports").mkdir(parents=True)
        (tmp_path / run / "reports" / "T164.json").write_text(runner.dumps(report), encoding="utf-8")
    # Compared prose is identical across builds; the digest stays in the runtime identity.
    assert runner.compare(tmp_path / "run1", tmp_path / "run2")["problems"] == []


def test_installed_package_is_compared_file_by_file_with_the_wheel(tmp_path):
    package = tmp_path / "site" / "ciw"
    (package / "lab" / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    (package / "lab" / "task.py").write_bytes(b"def run():\n    pass\n")
    (package / "lab" / "__pycache__" / "task.cpython-311.pyc").write_bytes(b"bytecode")
    matches = research_portfolio._installed_from
    assert matches(_wheel(tmp_path / "same.whl", package).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "no-record.whl", package, record=False).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "missing.whl", package, drop=("ciw/lab/task.py",)).read_bytes(), package)
    assert not matches(_wheel(tmp_path / "extra.whl", package, extra={"ciw/lab/more.py": b""}).read_bytes(), package)
    assert not matches(b"not a zip", package)
    wheel = _wheel(tmp_path / "before.whl", package).read_bytes()
    (package / "lab" / "task.py").write_bytes(b"def run():\n    return 999\n")  # edited after installation
    assert not matches(wheel, package)


# --------------------------------------------------------------- T165
@pytest.mark.lab_task("T165")
def test_release_report_inventories_nested_runtimes(retained, tmp_path):
    release = _run("T165", retained)
    assert release["state"] == "completed"
    labels = _labels(release)
    assert labels["Release state and label totals match a raw-text recount of the retained report files"] == "numerically_verified"
    assert labels["Every report whose provider probe succeeded contributes a runtime identity to the release inventory"] \
        == "numerically_verified"
    assert labels["The release digest is signed by a project key"] == "not_established"
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    # The SCR identity sits below the top level of T098's runtime identity.
    assert {r["runtime"] for r in record["runtimes"]} == {"scr"} and record["runtimes"][0]["tasks"] == ["T098"]
    assert record["physical_validation"] == "not_established" and record["reports"] == 5
    assert record["queue"]["tasks"] == 168 and "T165" in record["queue"]["not_reported"]
    # Artifact bytes (timing figures) stay out of the digest: a rerun with other artifacts gives the same digest.
    other = tmp_path / "other"
    for path in (retained / "reports").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["generated_artifacts"] = [{"path": f"artifacts/{report['task_id']}/t.svg", "sha256": "d" * 64, "bytes": 1}]
        report["report_id"] = __import__("ciw.lab.report", fromlist=["report_identity"]).report_identity(report)
        (other / "reports").mkdir(parents=True, exist_ok=True)
        (other / "reports" / path.name).write_text(runner.dumps(report), encoding="utf-8")
    _run("T165", other)
    again = json.loads((other / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    assert again["release_digest"] == record["release_digest"]
    # A report that ran a provider without recording its identity refutes the inventory.
    _retain(retained, "T099", "completed", [finding("g", "numerical", 1, {"checks": [CHECK]},
                                                    uncertainty={"kind": "exact", "value": 0, "basis": "b"})],
            provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {"provider:scr": True}})
    refuted = _run("T165", retained)
    assert refuted["state"] == "partial" and refuted["evidence_status"]["primary"] == "not_established"


@pytest.mark.lab_task("T165")
def test_release_report_lists_each_runtime_once_and_states_its_scope(retained):
    from ciw.core.identities import content_identity
    # T097 records the heads of the exchange checkouts, T098 heads and trees of the same checkouts; two Rust
    # builds have different binary digests.
    head, tree = "5" * 40, "6" * 40
    check = finding("g", "numerical", 1, {"checks": [CHECK]}, uncertainty={"kind": "exact", "value": 0, "basis": "b"})
    _retain(retained, "T097", "completed", [check], provider_runtime_identity={
        "set": {"head": head, "state": "ready"}, "rust": {"implementation": "rust_probe", "binary_sha256": "7" * 64}})
    _retain(retained, "T099", "completed", [check], provider_runtime_identity={
        "set": {"head": head, "tree": tree}, "rust": {"implementation": "rust_probe", "binary_sha256": "8" * 64}})
    release = _run("T165", retained)
    assert set(_labels(release).values()) == {"numerically_verified", "not_established"}
    assert _labels(release)["Every report whose provider probe succeeded contributes a runtime identity to the release "
                            "inventory"] == "numerically_verified"
    record = json.loads((retained / "artifacts" / "T165" / "release-report.json").read_text(encoding="utf-8"))
    runtimes = {r["runtime"]: r for r in record["runtimes"]}
    assert [r["runtime"] for r in record["runtimes"]] == sorted(runtimes)
    assert runtimes["set"]["identities"] == [{"revision": head, "tree": tree, "tasks": ["T097", "T099"]}]
    assert [i["tree"] for i in runtimes["rust"]["identities"]] == ["7" * 64, "8" * 64]
    # The scope is stated wherever the report describes itself, and the digest's encoding is declared.
    assert record["scope"] == "T001-T164" and record["schema"] == "ciw.lab-release-report.v3"
    text = (retained / "artifacts" / "T165" / "RELEASE.md").read_text(encoding="utf-8")
    assert text.startswith("# Lab release report of T001-T164") and "queue-state.json and the dashboard" in text
    assert "T001-T164" in release["hypothesis"] and release["numerical_result"].startswith("Release report of T001-T164")
    reports = [r for r in runner.load_reports(retained) if r["number"] < 165]
    content = [[r["task_id"], r["state"], r["evidence_status"]["primary"],
                [[f["claim"], f["evidence_status"]] for f in r["findings"]]] for r in reports]
    assert record["release_digest"] == content_identity(content) and "canonical_json" in record["release_digest_encoding"]
    assert record["basis_components"]["reference_checks"] == sum(
        bool(f["basis"].get("checks")) for r in reports for f in r["findings"])
    # An inventory that drops a recorded identity is refuted by the second path.
    original = research_portfolio.runtime_inventory
    try:
        research_portfolio.runtime_inventory = lambda reports: [r for r in original(reports) if r["runtime"] != "rust"]
        dropped = _run("T165", retained)
    finally:
        research_portfolio.runtime_inventory = original
    assert _labels(dropped)["Every report whose provider probe succeeded contributes a runtime identity to the release "
                            "inventory"] == "not_established"


# --------------------------------------------------------------- T166
@pytest.mark.lab_task("T166")
def test_unresolved_assumption_ledger(retained):
    _run("T155", retained, keep=True)  # this section's earlier reports are in the ledger too
    _retain(retained, "T023", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], unresolved_assumptions=[])
    ledger = _run("T166", retained)
    assert ledger["findings"][0]["evidence_status"] == "numerically_verified" and ledger["state"] == "completed"
    rows = json.loads((retained / "artifacts" / "T166" / "unresolved-assumptions.json").read_text(encoding="utf-8"))
    assert {task for row in rows for task in row["tasks"]} == {"T010", "T021", "T098", "T116", "T147", "T155"}
    assert ledger["findings"][0]["value"] == len(rows)
    assert "Reports stating no unresolved assumption: T023" in ledger["unresolved_assumptions"]
    text = (retained / "artifacts" / "T166" / "UNRESOLVED_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert research_portfolio._ledger_problems(text, rows) == []
    assert research_portfolio._ledger_problems(text.replace("(T010)", "(T011)"), rows)


# --------------------------------------------------------------- T167
@pytest.mark.lab_task("T167")
def test_unmeasured_ledger(retained):
    unmeasured = _run("T167", retained)
    document = json.loads((retained / "artifacts" / "T167" / "unmeasured.json").read_text(encoding="utf-8"))
    assert [c["task_id"] for c in document["not_established_claims"]] == ["T116", "T147"]
    assert {t["task_id"]: t["state"] for t in document["unfinished_tasks"]} == {"T116": "blocked", "T147": "partial"}
    # A computational GPU/CPU comparison in a task on the GPU route runs on the GPU host (the boundary lists
    # GPU/CPU agreement as computational), and its task is in that host's run with the physical claim's task.
    needs = {row["claim"]: row["needs"] for row in document["open_claims_by_need"]}
    assert needs["CPU and GPU outputs agree under the tolerance policy on GPU hardware"] == "execution:hardware:nvidia-gpu"
    assert needs["GPU energy per batch"] == "hardware:nvidia-gpu"
    route = document["next_acquisitions"][0]
    assert route["tasks"] == ["T116", "T147"] and route["claims"] == 1 and route["comparison_tasks"] == ["T147"]
    # The authority claim of a task whose probe failed is never attributed to hardware.
    assert "Kernel is ready for industrial deployment" not in needs
    text = (retained / "artifacts" / "T167" / "UNMEASURED.md").read_text(encoding="utf-8")
    assert "The GPU half did not run" in text and "`ciw lab unmeasured --retained lab`" in text
    labels = _labels(unmeasured)
    assert labels["Unmeasured ledger lists every not-established physical or authority claim"] == "numerically_verified"
    assert labels["Unmeasured ledger lists every blocked, deferred or partial task"] == "numerically_verified"
    assert labels["Unmeasured ledger classifies every open physical claim by what it needs"] == "numerically_verified"
    assert labels["Physical validity of the lab's computational results"] == "not_established"
    assert unmeasured["physical_validation_status"]["status"] == "not_established" and unmeasured["state"] == "completed"


@pytest.mark.lab_task("T166", "T167")
def test_ledgers_list_cross_cutting_open_items(retained):
    check = finding("g", "numerical", 1, {"checks": [CHECK]}, uncertainty={"kind": "exact", "value": 0, "basis": "b"})
    _retain(retained, "T081", "completed", [check], unresolved_assumptions=[
        "Cross-platform stability of float-valued numerical identities is not established."],
        recommended_next_task="CIW change: keyed signatures over workspace records.")
    _retain(retained, "T090", "completed",
            [check, finding("Retained workspace records are authenticated", "provenance", None, {},
                            expected_not_established=True)],
            unresolved_assumptions=["No provider checkout was bound to a declared-workload or telemetry workflow."])
    for task_id in ("T166", "T167"):
        report = _run(task_id, retained, keep=task_id == "T166")
        assert _labels(report)[research_portfolio.OPEN_ITEMS_CLAIM] == "numerically_verified", task_id
        assert _finding(report, research_portfolio.OPEN_ITEMS_CLAIM)["value"] == 3, task_id
    items = {item["key"]: item for item in json.loads(
        (retained / "artifacts" / "T166" / "open-items.json").read_text(encoding="utf-8"))}
    assert items["cross-platform"]["tasks"] == ["T081"] and items["telemetry-provisioning"]["tasks"] == ["T090"]
    assert {(s["task_id"], s["kind"]) for s in items["key-custody"]["statements"]} == {
        ("T081", "next step"), ("T090", "finding")}
    assert all(item["owning_task"] is None for item in items.values())
    # The ledger keeps parsing back, and each assumption carries the open items it raises.
    rows = json.loads((retained / "artifacts" / "T166" / "unresolved-assumptions.json").read_text(encoding="utf-8"))
    text = (retained / "artifacts" / "T166" / "UNRESOLVED_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert research_portfolio._ledger_problems(text, rows) == []
    assert {row["assumption"]: row["open_items"] for row in rows}[
        "Cross-platform stability of float-valued numerical identities is not established."] == ["cross-platform"]
    for heading in ("### Key custody and signatures", "### Telemetry provider provisioning", "### Cross-platform reproduction"):
        assert heading in text
        assert heading in (retained / "artifacts" / "T167" / "UNMEASURED.md").read_text(encoding="utf-8")
    # A statement whose phrase is not in its task's report file is caught by the raw-text second path.
    forged = [dict(items["cross-platform"], statements=[dict(items["cross-platform"]["statements"][0], task_id="T021")])]
    assert research_portfolio._open_item_problems(runner.Context(retained), 166, forged) == ["T021: Cross-platform"]


PTX_CLAIM = ("The gaussian_vi PTX kernel on the GPU reproduces the NumPy reference bitwise on every replica of the "
             "common Gaussian VI workload")


@pytest.mark.lab_task("T167")
def test_unmeasured_ledger_separates_code_from_hardware(tmp_path):
    from ciw.lab.energy_gpu_workload import NO_GPU_PROBE
    from ciw.lab.runner import CAPTURE_INSTRUMENTS
    assert set(CAPTURE_INSTRUMENTS.values()) <= set(research_portfolio.ACQUISITION_ROUTES)
    gpu = {"hardware:nvidia-gpu": False}
    _retain(tmp_path, "T115", "partial", [
        finding("Gross CPU package energy per geodesic trajectory", "physical", None,
                {"notes": ["no RAPL capture was supplied (set CIW_LAB_RAPL_LOG)"]})])
    _retain(tmp_path, "T116", "blocked", [
        finding("GPU-domain gross energy per measured batch", "physical", None, {"notes": ["no NVIDIA GPU or NVML"]}),
        finding("The NVML total-energy counter of the RTX 2080 has a characterized accuracy", "sensor_performance",
                None, {"notes": ["no external power meter was compared"]})],
        provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": gpu})
    _retain(tmp_path, "T117", "partial", [
        finding("A GPU implementation agrees with the Python kernel", "numerical", None,
                {"notes": ["no GPU implementation of this kernel was written, so none can run on any host"]},
                expected_not_established=True),
        finding("The Rust kernel uses less energy per trajectory than the Python kernel on real hardware", "physical",
                None, {"notes": ["no energy counter was read"]}),
        # The PTX kernel exists: this comparison needs only a host whose GPU probe succeeds.
        finding(PTX_CLAIM, "numerical", None, {"notes": [NO_GPU_PROBE]}, expected_not_established=True),
        # A physical quantity stated in a computational domain is misfiled; no acquisition supports it there.
        finding("GPU energy per batch is below the CPU package energy per batch", "numerical", None, {},
                expected_not_established=True)],
        provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": {**gpu, "tool:julia": False}})
    _retain(tmp_path, "T124", "partial", [
        finding("The fixtures' counter readings were produced by a physical GPU and NVML counter", "physical", None,
                {"notes": ["the fixtures declare origin synthetic_fixture and carry placeholder digests"]}),
        finding("The bound operator log's counter readings come from an NVML device on this host", "physical", None,
                {"notes": ["no operator NVML log was bound (--capture energy-log=PATH or CIW_LAB_ENERGY_LOG)"]})])
    # The encoders the claim names were never written: its task says so in its assumptions, not in notes.
    _retain(tmp_path, "T146", "completed", [
        finding("Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations",
                "computational_pipeline", None, {}, expected_not_established=True)],
        unresolved_assumptions=["Julia, C++ and GPU-host encoders were not run"])
    _retain(tmp_path, "T147", "partial", [
        finding("GPU/CPU agreement establishes industrial readiness", "industrial_readiness", None, {})],
        provider_runtime_identity={"implementation": "ciw.lab", "requirement_probes": gpu})
    report = _run("T167", tmp_path)
    document = json.loads((tmp_path / "artifacts" / "T167" / "unmeasured.json").read_text(encoding="utf-8"))
    needs = {row["claim"]: row["needs"] for row in document["open_claims_by_need"]}
    assert needs == {
        "Gross CPU package energy per geodesic trajectory": "hardware:rapl",
        "GPU-domain gross energy per measured batch": "hardware:nvidia-gpu",
        "The NVML total-energy counter of the RTX 2080 has a characterized accuracy": "reference_instrument",
        # Missing code is not missing hardware: a GPU host would not unblock these.
        "A GPU implementation agrees with the Python kernel": "implementation",
        "The Rust kernel uses less energy per trajectory than the Python kernel on real hardware": "no_probe",
        PTX_CLAIM: "execution:hardware:nvidia-gpu",
        "GPU energy per batch is below the CPU package energy per batch": "computational_domain",
        "The fixtures' counter readings were produced by a physical GPU and NVML counter": "fixture_origin",
        "The bound operator log's counter readings come from an NVML device on this host": "hardware:nvidia-gpu",
        "Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations": "implementation"}
    assert [route["probe"] for route in document["next_acquisitions"]] == ["hardware:nvidia-gpu", "hardware:rapl"]
    # The GPU host's run includes the task whose comparison runs only there; the fixture claim is not counted.
    gpu_route = document["next_acquisitions"][0]
    assert gpu_route["tasks"] == ["T116", "T117", "T124"] and gpu_route["claims"] == 2
    assert gpu_route["comparisons"] == 1 and gpu_route["comparison_tasks"] == ["T117"]
    assert report["recommended_next_task"].startswith("On the capture host: `ciw energy probe --gpu-index 0`")
    assert "then run T116, T117, T124 there" in report["recommended_next_task"]
    text = (tmp_path / "artifacts" / "T167" / "UNMEASURED.md").read_text(encoding="utf-8")
    heading = "### Runs on the nvidia-gpu host (computational comparison; no acquisition needed)"
    assert f"{heading}\n\nComputational comparisons whose code exists" in text
    misfiled = text.split("### Physical quantities filed under a computational domain")[1].split("###")[0]
    assert "GPU energy per batch is below" in misfiled and "PTX" not in misfiled and "Byte-identical" not in misfiled
    assert "### Claims about retained synthetic fixtures (no acquisition changes their origin)" in text
    assert document["hardware_runs"].startswith("aggregated outside the queue by `ciw lab unmeasured`")
    boundary = {row["domain"]: row["not_established"] for row in document["boundary"]}
    assert boundary["customer_demand"] == 0 and boundary["industrial_readiness"] == 1 and None in boundary
    assert set(boundary) - {None} == set(research_portfolio.PHYSICAL_DOMAINS | research_portfolio.AUTHORITY_DOMAINS)
    labels = _labels(report)
    assert labels["Unmeasured ledger classifies every open physical claim by what it needs"] == "numerically_verified"
    assert report["state"] == "completed" and labels["Physical validity of the lab's computational results"] == "not_established"


@pytest.mark.lab_task("T167")
def test_unmeasured_classifier_is_checked_on_probe_records(retained, monkeypatch):
    # Every hand-labelled record gets its label, and each need the ledger uses is exercised by one.
    assert research_portfolio._unmeasured_probe_errors() == []
    expected = {need for *_, need in research_portfolio.UNMEASURED_PROBE}
    assert expected == {None, "execution:hardware:nvidia-gpu", "hardware:nvidia-gpu", "hardware:rapl"} | set(
        research_portfolio.NEEDS)
    claim = "Unmeasured ledger classifies every open physical claim by what it needs"
    assert _labels(_run("T167", retained))[claim] == "numerically_verified"
    # A classifier that loses every acquisition route (or reads code as hardware) is caught by the probe records.
    for broken in (lambda report, record, routes: "no_probe",
                   lambda report, record, routes: "hardware:nvidia-gpu"):
        monkeypatch.setattr(research_portfolio, "classify_unmeasured", broken)
        assert research_portfolio._unmeasured_probe_errors()
        report = _run("T167", retained)
        assert _labels(report)[claim] == "not_established"
        assert report["evidence_status"]["primary"] == "not_established"
        assert any(a.startswith("Classifier probe records given another need:") for a in report["unresolved_assumptions"])


def test_aggregates_block_without_prior_reports(tmp_path):
    for task_id in ("T157", "T158", "T159", "T160", "T161", "T162", "T163", "T165", "T166", "T167"):
        report = _run(task_id, tmp_path)
        assert report["state"] == "blocked" and report["findings"] == [], task_id
    # T156 checks its curated half without reports; attribution stays unestablished and the task partial.
    ledger = _run("T156", tmp_path)
    assert ledger["state"] == "partial"
    assert [f["evidence_status"] for f in ledger["findings"]] == [
        "not_established", "not_established", "numerically_verified", "not_established"]
    assert all(f.get("expected_not_established") for f in ledger["findings"] if f["evidence_status"] == "not_established")


# --------------------------------------------------------------- T168
TIED_TESTS = '''
import pytest

@pytest.mark.lab_task("T010")
def test_t010_runs_and_labels(lab):
    assert lab("T010")["findings"][0]["evidence_status"] == "numerically_verified"

@pytest.mark.lab_task("T116")
def test_values_only(lab):
    assert lab("T116")["findings"][0]["value"] is None

def test_unrelated():
    assert 1 + 1 == 2

@pytest.mark.lab_task("T168")
def test_t168_labels(lab):
    assert lab("T168")["findings"][0]["evidence_status"] == "numerically_verified"
'''
OWN = "tests/test_fake.py::test_t168_labels"


def _fake_repository(root, monkeypatch):
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_fake.py").write_text(TIED_TESTS, encoding="utf-8")
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(root))


@pytest.mark.lab_task("T168")
def test_regression_coverage_is_checked(retained, tmp_path, monkeypatch):
    _fake_repository(tmp_path / "repo", monkeypatch)
    tied, values = "tests/test_fake.py::test_t010_runs_and_labels", "tests/test_fake.py::test_values_only"

    def registry(t010, t116=(values,)):
        # Only the fixture's tasks, so real section registrations cannot leak in.
        fakes = {"T010": Implementation("T010", None, regression_tests=t010),
                 "T116": Implementation("T116", None, regression_tests=t116),
                 "T168": Implementation("T168", None, regression_tests=(OWN,))}
        monkeypatch.setattr(research_portfolio, "load_implementations", lambda: (fakes, {}))

    registry((tied,))
    report = _run("T168", retained)
    labels = _labels(report)
    rows = json.loads((retained / "artifacts" / "T168" / "regression-coverage.json").read_text(encoding="utf-8"))
    assert {row["task_id"]: row["tied"] for row in rows}["T010"] == [tied]
    # Every registration is declared by its test's marker, and no marker names a task that does not register it.
    assert "3 task-to-node registrations (3 distinct node ids), 3 declared by a lab_task marker" \
        in report["numerical_result"]
    assert _finding(report, "Task-to-node registrations whose")["value"] == 0
    assert _finding(report, "Declarations in lab_task markers")["value"] == 0
    # T116's test asserts values only, and T021, T098 and T147 register no test at all.
    assert _finding(report, "Tasks without a registered test that both")["value"] == 1
    assert labels["Tasks without a registered test that both declares the task with a lab_task marker and asserts "
                  "an evidence label"] == "numerically_verified"
    assert _finding(report, "Completed or partial tasks lacking")["value"] == 3
    assert labels["Completed or partial tasks lacking a regression test"] == "not_established"
    # No JUnit outcomes were recorded in these reports: the pass/fail finding is honestly unestablished.
    assert _finding(report, "Registered regression tests failing")["expected_not_established"] is True
    assert report["state"] == "partial"
    # The next step names the untied task's test that asserts values only.
    assert report["recommended_next_task"].startswith(f"Assert evidence labels in {values} (the registered tests of T116")
    # A registered test that fails in the recorded JUnit outcomes refutes the task.
    _retain(retained, "T010", "partial", [finding("f", "numerical", 1, {"checks": [CHECK]})],
            tests_passed=[], extra={"tests_failed": [f"pytest: {tied}"]})
    failing = _run("T168", retained)
    assert _labels(failing)["Registered regression tests failing in the JUnit record of this run"] == "not_established"
    # A registration whose test's marker names another task is undeclared; a marker naming a task that does not
    # register the test is a stray declaration. Either keeps the task partial, and the next step says so first.
    registry((tied,), t116=(tied,))
    report = _run("T168", retained)
    assert _finding(report, "Task-to-node registrations whose")["value"] == 1
    assert _finding(report, "Declarations in lab_task markers")["value"] == 1
    assert "Registrations whose test's lab_task marker does not name the task: T116: " + tied \
        in report["unresolved_assumptions"]
    assert ("Declarations in lab_task markers naming a task that does not register the test: "
            "tests/test_fake.py::test_values_only: T116") in report["unresolved_assumptions"]
    assert report["state"] == "partial"
    assert report["recommended_next_task"].startswith("Make each registered test's lab_task marker name exactly the "
                                                      "tasks that register it (1 registrations undeclared, 1 ")
    registry(("tests/test_fake.py::test_does_not_exist",))
    report = _run("T168", retained)
    assert report["state"] == "partial" and _finding(report, "Registered regression node ids")["value"] == 1
    assert report["recommended_next_task"].startswith("Register node ids that resolve to test functions (1 dangling)")


@pytest.mark.lab_task("T168")
def test_regression_outcomes_add_up_and_name_the_own_node(tmp_path, monkeypatch):
    _fake_repository(tmp_path / "repo", monkeypatch)
    tied, values = "tests/test_fake.py::test_t010_runs_and_labels", "tests/test_fake.py::test_values_only"
    # T116 (no report in this run) registers the test its marker names, so no declaration is stray.
    fakes = {"T010": Implementation("T010", None, regression_tests=(tied,)),
             "T021": Implementation("T021", None, regression_tests=(tied,)),
             "T116": Implementation("T116", None, regression_tests=(values,)),
             "T168": Implementation("T168", None, regression_tests=(OWN,))}
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: (fakes, {}))
    run = tmp_path / "run"
    _retain(run, "T010", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], tests_passed=[f"pytest: {tied}"])
    report = _run("T168", run)
    # T168's own node cannot be in its own run's record: it is named, and it does not block completion.
    assert "JUnit: 1 passed, 0 failed, 0 skipped or not run, 1 not recorded (1 of them T168's own node" \
        in report["numerical_result"]
    assert "2 task-to-node registrations (2 distinct node ids)" in report["numerical_result"]
    assert report["state"] == "completed" and set(_labels(report).values()) == {"numerically_verified"}
    # A registration of another task without an outcome is counted and blocks completion.
    _retain(run, "T021", "completed", [finding("f", "numerical", 1, {"checks": [CHECK]})], tests_passed=[])
    report = _run("T168", run)
    assert "3 task-to-node registrations (2 distinct node ids)" in report["numerical_result"]
    assert "2 not recorded (1 of them T168's own node" in report["numerical_result"]
    # T021 shares T010's test, whose marker does not name T021.
    assert "2 declared by a lab_task marker; 1 registrations whose marker does not name the task" \
        in report["numerical_result"]
    assert report["state"] == "partial"
    assert any(a.startswith("1 task-to-node registrations of other tasks have no outcome")
               for a in report["unresolved_assumptions"])


def test_regression_tie_analysis_is_checked_on_probe_cases(monkeypatch):
    assert research_portfolio._tie_probe_errors() == []
    index = {}
    research_portfolio._index_source("tests/test_fake.py", TIED_TESTS, index)
    assert research_portfolio._tie("T010", "tests/test_fake.py::test_t010_runs_and_labels", index) == (True, True)
    assert research_portfolio._tie("T010", "tests/test_fake.py::test_unrelated", index) == (False, False)
    assert research_portfolio._tie("T116", "tests/test_fake.py::test_values_only", index) == (True, False)
    # A task id in a test's name or source no longer ties it: only a marker declares.
    assert research_portfolio._tie("T168", "tests/test_fake.py::test_t168_labels", index) == (True, True)
    assert research_portfolio._tie("T116", "tests/test_fake.py::test_t010_runs_and_labels", index) == (False, True)
    # A probe that reads markers wrongly is caught before the counts are trusted.
    monkeypatch.setattr(research_portfolio, "_declared_tasks", lambda decorators: frozenset({"T901"}))
    assert research_portfolio._tie_probe_errors()


def test_every_section_registration_is_tied_to_its_task():
    implementations, _ = load_implementations()
    index = research_portfolio._test_index(Path(__file__).resolve().parent)
    for task_id in SECTION:
        assert any(research_portfolio._tie(task_id, node, index) == (True, True)
                   for node in implementations[task_id].regression_tests), task_id


def test_every_registered_regression_test_declares_exactly_its_tasks():
    """Each registered node's lab_task marker names every task that registers it, and no other task."""
    implementations, _ = load_implementations()
    registered = {task_id: implementation.regression_tests for task_id, implementation in implementations.items()}
    tests = Path(__file__).resolve().parent
    files = {node.split("::")[0] for nodes in registered.values() for node in nodes}
    if not all((tests.parent / path).is_file() for path in files):
        pytest.skip("the registered test modules are not beside this one")
    index = research_portfolio._test_index(tests)
    undeclared = [f"{task_id}: {node}" for task_id, nodes in registered.items() for node in nodes
                  if not research_portfolio._tie(task_id, node, index)[0]]
    assert undeclared == []
    assert research_portfolio._stray_declarations(index, registered) == []


def test_the_lab_task_marker_is_registered(pytestconfig):
    # pyproject.toml registers it here; in the clean room, the pytest.ini that scripts/reproduce_lab.py writes.
    assert any(line.startswith("lab_task(") for line in pytestconfig.getini("markers"))


def test_regression_node_ids_resolve_class_methods_and_async_tests(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_demo.py").write_text(
        "def test_plain():\n    pass\n\n"
        "async def test_async():\n    pass\n\n"
        "class TestGroup:\n    def test_method(self):\n        pass\n\n"
        "    class TestNested:\n        async def test_inner(self):\n            pass\n\n"
        "def helper():\n    pass\n", encoding="utf-8")
    assert research_portfolio._test_names(tests) == {
        "tests/test_demo.py::test_plain", "tests/test_demo.py::test_async",
        "tests/test_demo.py::TestGroup::test_method", "tests/test_demo.py::TestGroup::TestNested::test_inner"}


def test_regression_coverage_never_reads_the_current_directory(retained, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tests").mkdir(parents=True)
    (elsewhere / "tests" / "test_unrelated.py").write_text("def test_other():\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("CIW_LAB_REPOSITORY_ROOT", str(tmp_path / "installed-without-tests"))
    monkeypatch.setattr(research_portfolio, "load_implementations", lambda: ({"T010": Implementation(
        "T010", None, regression_tests=("tests/test_unrelated.py::test_other",))}, {}))
    report = _run("T168", retained)
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == "not_established"
    assert [f["evidence_status"] for f in report["findings"]] == ["not_established"]


# --------------------------------------------------------------- whole section
def test_every_numerical_finding_declares_uncertainty_and_tolerance(retained, monkeypatch):
    monkeypatch.delenv("CIW_LAB_CLEAN_ROOM", raising=False)
    for task_id in SECTION:
        report = _run(task_id, retained, keep=True)
        assert report["unresolved_assumptions"], task_id
        for record in report["findings"]:
            if not research_portfolio._numbers(record["value"]):
                continue
            uncertainty = record["uncertainty"]
            assert isinstance(uncertainty, dict) and set(uncertainty) == {"kind", "value", "basis"}, (task_id, record["claim"])
            assert math.isfinite(uncertainty["value"]) and uncertainty["basis"], (task_id, record["claim"])
            assert record["regression_tolerance"] == {"abs": 0, "rel": 0}, (task_id, record["claim"])
            assert record["basis"].get("checks"), (task_id, record["claim"])
