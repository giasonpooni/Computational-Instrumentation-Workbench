"""Independent exact primal/dual checks for bounded local scalar design."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import runpy

import pytest

from ciw import design_adjustment as da
from ciw.telemetry import canonical, digest


def ratio(numerator, denominator=1):
    q = Fraction(numerator, denominator)
    return {"numerator": q.numerator, "denominator": q.denominator}


def fraction(encoded):
    return Fraction(encoded["numerator"], encoded["denominator"])


def problem(x=3, target=Fraction(19, 2), regularization=1,
            lower=Fraction(-1, 10), upper=Fraction(1, 10), epsilon=0):
    return da.make_problem(*(ratio(v) for v in
                             (x, target, regularization, lower, upper, epsilon)))


def literal_interior():
    """An exact reference independent of the candidate producer."""
    return {
        "problem": problem(), "delta": ratio(3, 37),
        "lower_multiplier": ratio(0), "upper_multiplier": ratio(0),
        "predicted_output": ratio(351, 37),
        "nonlinear_output": ratio(12996, 1369),
        "objective": ratio(1, 148), "lower_bound": ratio(1, 148), "gap": ratio(0),
    }


def reseal(report):
    report["report_id"] = digest({key: value for key, value in report.items()
                                  if key != "report_id"})
    return report


def test_handwritten_interior_optimum_is_exact_without_candidate_producer(monkeypatch):
    proposal = literal_interior()

    def forbidden(*args, **kwargs):
        raise AssertionError("Checking cannot trust the candidate producer")

    monkeypatch.setattr(da, "make_candidate", forbidden)
    report = da.check_candidate(proposal)
    assert report["schema"] == "ciw.design-adjustment-check.v1"
    assert report["checks"] == {
        "evaluation": True, "feasibility": True, "dual_feasibility": True,
        "bounded_suboptimality": True, "exact_optimality": True,
    }
    assert report["status"] == "accepted"
    assert report["candidate"] == proposal
    assert report["bindings"]["candidate_digest"] == digest(proposal)
    assert report["bindings"]["specification_digest"] == digest(da.specification())
    assert report["report_id"] == digest({key: value for key, value in report.items()
                                            if key != "report_id"})


def test_candidate_producer_matches_independent_interior_reference():
    assert da.make_candidate(problem(), ratio(3, 37)) == literal_interior()


@pytest.mark.parametrize("target, delta, lower_dual, upper_dual, predicted, nonlinear", [
    (10, Fraction(1, 10), 0, Fraction(23, 5), Fraction(48, 5), Fraction(961, 100)),
    (8, Fraction(-1, 10), Fraction(23, 5), 0, Fraction(42, 5), Fraction(841, 100)),
])
def test_active_boundary_goldens_include_correct_multiplier_sign(
        target, delta, lower_dual, upper_dual, predicted, nonlinear):
    proposal = {
        "problem": problem(target=target), "delta": ratio(delta),
        "lower_multiplier": ratio(lower_dual), "upper_multiplier": ratio(upper_dual),
        "predicted_output": ratio(predicted), "nonlinear_output": ratio(nonlinear),
        "objective": ratio(17, 100), "lower_bound": ratio(17, 100), "gap": ratio(0),
    }
    assert da.make_candidate(proposal["problem"], ratio(delta)) == proposal
    report = da.check_candidate(proposal)
    assert all(report["checks"].values())
    assert report["status"] == "accepted"


def test_zero_slope_retains_nonzero_model_mismatch():
    proposal = da.make_candidate(problem(x=0, target=1), ratio(0))
    assert proposal["predicted_output"] == ratio(0)
    assert proposal["nonlinear_output"] == ratio(0)
    assert proposal["objective"] == ratio(1)
    assert proposal["lower_bound"] == ratio(1)
    assert proposal["gap"] == ratio(0)
    assert da.check_candidate(proposal)["checks"]["exact_optimality"] is True


def test_zero_slope_nonzero_step_can_be_approximately_acceptable():
    proposal = da.make_candidate(problem(x=0, target=1, epsilon=Fraction(1, 100)), ratio(1, 10),
                                 ratio(0), ratio(0))
    assert proposal["predicted_output"] == ratio(0)
    assert proposal["nonlinear_output"] == ratio(1, 100)
    assert proposal["objective"] == ratio(101, 100)
    assert proposal["lower_bound"] == ratio(1)
    assert proposal["gap"] == ratio(1, 100)
    report = da.check_candidate(proposal)
    assert report["checks"]["bounded_suboptimality"] is True
    assert report["checks"]["exact_optimality"] is False
    assert report["status"] == "accepted"


@pytest.mark.parametrize("target, expected_lower, expected_upper", [
    (10, 0, 12), (8, 12, 0), (9, 0, 0),
])
def test_degenerate_interval_still_supports_dual_stationarity(target, expected_lower, expected_upper):
    proposal = da.make_candidate(problem(target=target, lower=0, upper=0), ratio(0))
    assert proposal["lower_multiplier"] == ratio(expected_lower)
    assert proposal["upper_multiplier"] == ratio(expected_upper)
    assert proposal["objective"] == ratio((9 - target) ** 2)
    assert proposal["gap"] == ratio(0)
    assert da.check_candidate(proposal)["checks"]["exact_optimality"] is True


@pytest.mark.parametrize("epsilon, accepted", [
    (Fraction(1, 20000), True), (Fraction(1, 23125), True), (Fraction(1, 25000), False), (0, False),
])
def test_near_optimum_uses_exact_gap_instead_of_solver_success(epsilon, accepted):
    proposal = da.make_candidate(problem(epsilon=epsilon), ratio(2, 25), ratio(0), ratio(0))
    assert proposal["objective"] == ratio(17, 2500)
    assert proposal["lower_bound"] == ratio(1, 148)
    assert proposal["gap"] == ratio(1, 23125)
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is True
    assert report["checks"]["feasibility"] is True
    assert report["checks"]["dual_feasibility"] is True
    assert report["checks"]["bounded_suboptimality"] is accepted
    assert report["checks"]["exact_optimality"] is False
    assert report["status"] == ("accepted" if accepted else "rejected")


@pytest.mark.parametrize("x, target, regularization, lower, upper, delta, ml, mu", [
    (Fraction(2, 3), Fraction(5, 7), Fraction(3, 11), Fraction(-2, 5), Fraction(1, 2),
     Fraction(1, 7), Fraction(1, 13), Fraction(2, 17)),
    (-2, 3, 2, -1, 1, Fraction(-1, 4), 0, Fraction(7, 3)),
])
def test_primal_and_dual_values_match_independent_fraction_formulas(
        x, target, regularization, lower, upper, delta, ml, mu):
    x, target, regularization, lower, upper, delta, ml, mu = map(
        Fraction, (x, target, regularization, lower, upper, delta, ml, mu))
    p = problem(x, target, regularization, lower, upper, epsilon=1)
    proposal = da.make_candidate(p, ratio(delta), ratio(ml), ratio(mu))
    j = 2 * x
    h = j * j + regularization
    b = j * (x * x - target)
    c = (x * x - target) ** 2
    primal = (x * x + j * delta - target) ** 2 + regularization * delta * delta
    dual = c + ml * lower - mu * upper - (2 * b - ml + mu) ** 2 / (4 * h)
    assert proposal["predicted_output"] == ratio(x * x + j * delta)
    assert proposal["nonlinear_output"] == ratio((x + delta) ** 2)
    assert proposal["objective"] == ratio(primal)
    assert proposal["lower_bound"] == ratio(dual)
    assert proposal["gap"] == ratio(primal - dual)


def test_primal_infeasibility_cannot_be_accepted_with_generous_epsilon():
    proposal = da.make_candidate(problem(epsilon=1), ratio(1, 5), ratio(0), ratio(0))
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is True
    assert report["checks"]["feasibility"] is False
    assert report["checks"]["dual_feasibility"] is True
    assert report["checks"]["bounded_suboptimality"] is False
    assert report["checks"]["exact_optimality"] is False
    assert report["status"] == "rejected"


def test_negative_dual_is_well_formed_but_not_a_valid_lower_bound_witness():
    proposal = da.make_candidate(problem(epsilon=1), ratio(3, 37), ratio(-1), ratio(0))
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is True
    assert report["checks"]["feasibility"] is True
    assert report["checks"]["dual_feasibility"] is False
    assert report["checks"]["bounded_suboptimality"] is False
    assert report["status"] == "rejected"


def test_noncomplementary_duals_cannot_claim_exact_optimality():
    proposal = da.make_candidate(problem(epsilon=1), ratio(3, 37), ratio(1), ratio(1))
    assert proposal["gap"] == ratio(1, 5)
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is True
    assert report["checks"]["bounded_suboptimality"] is True
    assert report["checks"]["exact_optimality"] is False


@pytest.mark.parametrize("field", ["predicted_output", "nonlinear_output", "objective", "lower_bound", "gap"])
def test_false_claimed_evaluation_is_rejected(field):
    proposal = literal_interior()
    proposal[field] = ratio(fraction(proposal[field]) + 1)
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is False
    assert report["checks"]["bounded_suboptimality"] is False
    assert report["checks"]["exact_optimality"] is False
    assert report["status"] == "rejected"


def test_consistently_forged_objective_and_gap_do_not_bypass_evaluation():
    proposal = literal_interior()
    proposal["objective"] = ratio(0)
    proposal["gap"] = ratio(-1, 148)
    report = da.check_candidate(proposal)
    assert report["checks"]["evaluation"] is False
    assert report["checks"]["bounded_suboptimality"] is False
    assert report["status"] == "rejected"


@pytest.mark.parametrize("updates", [
    {"x": 11}, {"x": -11}, {"target": 101}, {"target": -101},
    {"regularization": 0}, {"regularization": Fraction(1, 1000001)}, {"regularization": 101},
    {"lower": -2}, {"upper": 2}, {"lower": 1, "upper": 0},
    {"epsilon": -1}, {"epsilon": 2},
])
def test_problem_domain_violations_are_refused(updates):
    with pytest.raises(ValueError):
        problem(**updates)


@pytest.mark.parametrize("kwargs", [
    {"x": -10, "target": -100, "regularization": Fraction(1, 1000000), "lower": -1, "upper": 1},
    {"x": 10, "target": 100, "regularization": 100, "lower": 0, "upper": 0, "epsilon": 1},
])
def test_closed_problem_domain_endpoints_are_supported(kwargs):
    p = problem(**kwargs)
    proposal = da.make_candidate(p, ratio(0))
    assert da.check_candidate(proposal)["checks"]["evaluation"] is True


@pytest.mark.parametrize("bad", [
    3, 3.0, True, None, [3, 1],
    {"numerator": 2, "denominator": 2}, {"numerator": 0, "denominator": 2},
    {"numerator": 3, "denominator": 0}, {"numerator": 3, "denominator": -1},
    {"numerator": True, "denominator": 1}, {"numerator": 3.0, "denominator": 1},
    {"numerator": float("nan"), "denominator": 1},
    {"numerator": 1, "denominator": 1, "extra": 1},
    {"numerator": 1000001, "denominator": 1000000},
])
def test_problem_ratio_encoding_is_strict_and_bounded(bad):
    with pytest.raises(ValueError):
        da.make_problem(bad, ratio(1), ratio(1), ratio(-1), ratio(1), ratio(0))


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(expression="print('unexpected')"),
    lambda p: p.pop("regularization"),
    lambda p: p.update(specification_digest="sha256:" + "0" * 64),
])
def test_unknown_problem_fields_and_specification_are_refused(mutate):
    p = problem()
    mutate(p)
    with pytest.raises(ValueError):
        da.make_candidate(p, ratio(0))


@pytest.mark.parametrize("delta, ml, mu", [
    (ratio(101), ratio(0), ratio(0)),
    (ratio(0), ratio(1000001), ratio(0)),
    (ratio(0), ratio(0), ratio(-1000001)),
    ({"numerator": 1, "denominator": 2 ** 255}, ratio(0), ratio(0)),
])
def test_candidate_delta_and_dual_resource_bounds(delta, ml, mu):
    with pytest.raises(ValueError):
        da.make_candidate(problem(), delta, ml, mu)


@pytest.mark.parametrize("ml, mu", [(ratio(0), None), (None, ratio(0))])
def test_only_one_explicit_multiplier_is_refused(ml, mu):
    with pytest.raises(ValueError):
        da.make_candidate(problem(), ratio(0), ml, mu)


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("gap"),
    lambda c: c.update(solver_status="optimal"),
    lambda c: c.update(objective={"numerator": 2 ** 255, "denominator": 1}),
    lambda c: c.update(gap={"numerator": False, "denominator": 1}),
])
def test_candidate_shape_and_output_encoding_are_strict(mutate):
    proposal = literal_interior()
    mutate(proposal)
    with pytest.raises(ValueError):
        da.check_candidate(proposal)


def test_specification_problem_and_report_are_independent_mutable_copies():
    p = problem()
    proposal = da.make_candidate(p, ratio(3, 37))
    p["x"]["numerator"] = 0
    assert proposal["problem"]["x"] == ratio(3)
    report = da.check_candidate(proposal)
    proposal["delta"]["numerator"] = 0
    assert report["candidate"]["delta"] == ratio(3, 37)
    a = da.specification()
    a["extra"] = "mutation"
    assert "extra" not in da.specification()


def test_reopen_is_retained_inspection_without_candidate_generation_or_checking(tmp_path, monkeypatch):
    report = da.check_candidate(literal_interior())
    destination = tmp_path / "design.json"
    da.save_check(destination, report)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved inspection invoked numerical candidate checking")

    monkeypatch.setattr(da, "make_candidate", forbidden)
    monkeypatch.setattr(da, "check_candidate", forbidden)
    assert da.load_check(destination) == report
    assert da.inspect_check(report) == report
    assert "retained" in da.render_check(report).lower()
    assert isinstance(da.render_check(report, details=True), str)
    assert destination.read_bytes() == canonical(report)
    with pytest.raises((FileExistsError, ValueError)):
        da.save_check(destination, report)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(schema="ciw.design-adjustment-check.v99"),
    lambda r: r.update(status="rejected"),
    lambda r: r["checks"].update(exact_optimality=1),
    lambda r: r["checks"].pop("dual_feasibility"),
    lambda r: r["checks"].update(physical=True),
    lambda r: r["authority"].update(proof_id="invented"),
    lambda r: r["bindings"].update(candidate_digest="sha256:" + "0" * 64),
    lambda r: r["bindings"].update(statement_digest="sha256:" + "0" * 64),
    lambda r: r["specification"].update(extra="unexpected"),
])
def test_retained_structural_and_authority_tampering_is_refused(mutate):
    report = da.check_candidate(literal_interior())
    mutate(report)
    with pytest.raises(ValueError):
        da.inspect_check(reseal(report))


@pytest.mark.parametrize("claim", ["evaluation", "feasibility", "dual_feasibility", "bounded_suboptimality"])
def test_retained_success_requires_its_declared_logical_prerequisites(claim):
    report = da.check_candidate(literal_interior())
    report["checks"][claim] = False
    if claim == "bounded_suboptimality":
        report["status"] = "rejected"
    with pytest.raises(ValueError):
        da.inspect_check(reseal(report))


def test_approximate_acceptance_survives_save_and_reopen_without_becoming_exact(tmp_path):
    proposal = da.make_candidate(problem(epsilon=Fraction(1, 20000)), ratio(2, 25))
    report = da.check_candidate(proposal)
    destination = tmp_path / "approximate.json"
    da.save_check(destination, report)
    reopened = da.load_check(destination)
    assert reopened["status"] == "accepted"
    assert reopened["checks"]["bounded_suboptimality"] is True
    assert reopened["checks"]["exact_optimality"] is False
    assert reopened["authority"]["nonlinear_optimality"] == "not_established"


def test_reference_report_grants_no_execution_proof_or_physical_authority():
    report = da.check_candidate(literal_interior())
    assert not {"execution_id", "result_id", "verification_id", "proof_id"}.intersection(report)
    for name in ("execution_id", "result_id", "verification_id", "proof_id"):
        assert report["authority"][name] is None
    for name in ("formal_verification", "sp1_verification", "state_admission", "hardware_actuation"):
        assert report["authority"][name] == "not_performed"
    assert report["authority"]["physical_validation"] == "not_established"


def test_coherent_forged_checks_are_historical_claims_not_authentication(tmp_path, monkeypatch):
    proposal = literal_interior()
    proposal["objective"] = ratio(0)
    report = da.check_candidate(proposal)
    assert report["status"] == "rejected"
    report["checks"] = {name: True for name in report["checks"]}
    report["status"] = "accepted"
    report = reseal(report)
    destination = tmp_path / "rewritten.json"
    destination.write_bytes(canonical(report))
    fresh_checker = da.check_candidate

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not silently redo checking")

    monkeypatch.setattr(da, "check_candidate", forbidden)
    assert da.load_check(destination) == report
    assert fresh_checker(report["candidate"])["status"] == "rejected"


def test_retained_save_requires_canonical_unique_json(tmp_path):
    report = da.check_candidate(literal_interior())
    destination = tmp_path / "design.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        da.load_check(destination)
    destination.write_bytes(canonical(report)[:-1] + b',"status":"accepted"}')
    with pytest.raises(ValueError):
        da.load_check(destination)


@pytest.mark.parametrize("payload", [b'{"x":0,"x":1}', b'{"x":NaN}', b'null', b'[]', b'\xff',
                                     b" " * (64 * 1024 + 1)],
                         ids=["duplicate", "nan", "null", "array", "invalid-utf8", "oversized"])
def test_data_only_candidate_loader_rejects_invalid_or_oversized_inputs(tmp_path, payload):
    source = tmp_path / "candidate.json"
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        da.load_candidate(source)


def test_cli_status_codes_preserve_rejected_reports_and_refuse_overwrite(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts" / "check_design_adjustment.py"))["main"]
    source = tmp_path / "candidate.json"
    output = tmp_path / "report.json"
    source.write_bytes(canonical(literal_interior()))
    assert main(["check", str(source), "--output", str(output)]) == 0
    assert main(["inspect", str(output)]) == 0
    assert main(["inspect", str(output), "--details"]) == 0
    assert main(["check", str(source), "--output", str(output)]) == 2
    wrong = literal_interior()
    wrong["objective"] = ratio(0)
    source.write_bytes(canonical(wrong))
    rejected_path = tmp_path / "rejected.json"
    assert main(["check", str(source), "--output", str(rejected_path)]) == 1
    assert da.load_check(rejected_path)["status"] == "rejected"
    assert main(["inspect", str(rejected_path)]) == 0
    source.write_bytes(b'{"expression":"exit()"}')
    malformed_path = tmp_path / "malformed.json"
    assert main(["check", str(source), "--output", str(malformed_path)]) == 2
    assert not malformed_path.exists()
    capsys.readouterr()
