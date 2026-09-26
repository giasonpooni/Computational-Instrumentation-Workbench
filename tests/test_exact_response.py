"""Independent exact-arithmetic and retained-only checks for scalar response."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import runpy

import pytest

from ciw import exact_response as er
from ciw.telemetry import canonical, digest


def rational(numerator, denominator=1):
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def value(encoded):
    return Fraction(encoded["numerator"], encoded["denominator"])


def candidate(x=3, delta=Fraction(1, 10), radius=Fraction(1, 10)):
    return er.make_candidate(rational(x), rational(delta), rational(radius))


def reseal(report):
    report["report_id"] = digest({key: item for key, item in report.items()
                                  if key != "report_id"})
    return report


def handwritten_candidate():
    return {
        "specification_digest": digest(er.specification()),
        "x": rational(3), "delta": rational(1, 10), "radius": rational(1, 10),
        "baseline_output": rational(9), "local_multiplier": rational(6),
        "predicted_output": rational(48, 5), "model_output": rational(961, 100),
        "residual": rational(1, 100), "error_bound": rational(1, 100),
    }


def test_checker_accepts_handwritten_golden_without_using_candidate_producer(monkeypatch):
    proposal = handwritten_candidate()

    def forbidden(*args, **kwargs):
        raise AssertionError("The checker must not trust the proposal routine")

    monkeypatch.setattr(er, "make_candidate", forbidden)
    report = er.check_candidate(proposal)
    assert report["checks"] == {"arithmetic": True, "meaning": True, "approximation": True}
    assert report["status"] == "accepted"
    statement = {"specification_digest": digest(er.specification()),
                 "candidate_digest": digest(proposal),
                 "claims": ["arithmetic", "meaning", "approximation"]}
    assert report["bindings"]["statement_digest"] == digest(statement)
    assert report["report_id"] == digest({key: item for key, item in report.items()
                                            if key != "report_id"})


@pytest.mark.parametrize("delta, predicted, model", [
    (Fraction(1, 10), Fraction(48, 5), Fraction(961, 100)),
    (Fraction(-1, 10), Fraction(42, 5), Fraction(841, 100)),
])
def test_exact_golden_values_with_positive_and_negative_variation(delta, predicted, model):
    proposal = candidate(delta=delta)
    assert proposal["x"] == rational(3)
    assert proposal["delta"] == rational(delta)
    assert proposal["radius"] == rational(1, 10)
    assert proposal["baseline_output"] == rational(9)
    assert proposal["local_multiplier"] == rational(6)
    assert proposal["predicted_output"] == rational(predicted)
    assert proposal["model_output"] == rational(model)
    assert proposal["residual"] == rational(1, 100)
    assert proposal["error_bound"] == rational(1, 100)
    report = er.check_candidate(proposal)
    assert report["schema"] == "ciw.exact-response-check.v1"
    assert report["checks"] == {"arithmetic": True, "meaning": True, "approximation": True}
    assert report["status"] == "accepted"
    assert report["candidate"] == proposal
    assert report["bindings"]["candidate_digest"] == digest(proposal)
    assert report["bindings"]["specification_digest"] == proposal["specification_digest"]


@pytest.mark.parametrize("x, delta, radius", [
    (Fraction(2, 3), Fraction(-1, 7), Fraction(2, 7)),
    (Fraction(-4, 9), Fraction(5, 13), Fraction(1, 2)),
    (Fraction(999999, 1000000), Fraction(1, 1000000), Fraction(1, 1000000)),
])
def test_nondecimal_denominators_are_exact_and_reduced(x, delta, radius):
    proposal = candidate(x, delta, radius)
    expected = {
        "baseline_output": x * x,
        "local_multiplier": 2 * x,
        "predicted_output": x * x + 2 * x * delta,
        "model_output": (x + delta) ** 2,
        "residual": delta * delta,
        "error_bound": radius * radius,
    }
    for key, wanted in expected.items():
        assert proposal[key] == rational(wanted)
        assert value(proposal[key]) == wanted
    assert er.check_candidate(proposal)["status"] == "accepted"


def test_zero_local_multiplier_does_not_mean_zero_change():
    proposal = candidate(0, Fraction(1, 10), Fraction(1, 10))
    assert proposal["local_multiplier"] == rational(0)
    assert proposal["predicted_output"] == rational(0)
    assert proposal["model_output"] == rational(1, 100)
    assert proposal["residual"] == rational(1, 100)
    assert er.check_candidate(proposal)["status"] == "accepted"


@pytest.mark.parametrize("x, delta, radius", [(3, 0, 0), (100, 0, 0), (-100, 0, 0),
                                                     (99, 1, 1), (-99, -1, 1), (0, 100, 100)])
def test_zero_and_exact_domain_edges_are_admissible(x, delta, radius):
    assert er.check_candidate(candidate(x, delta, radius))["status"] == "accepted"


def test_self_consistent_arithmetic_with_wrong_multiplier_fails_meaning():
    proposal = candidate()
    proposal["local_multiplier"] = rational(7)
    proposal["predicted_output"] = rational(97, 10)
    proposal["residual"] = rational(-9, 100)
    report = er.check_candidate(proposal)
    assert report["checks"]["arithmetic"] is True
    assert report["checks"]["meaning"] is False
    assert report["checks"]["approximation"] is False
    assert report["status"] == "rejected"


def test_wrong_prediction_fails_only_arithmetic_when_other_fields_are_correct():
    proposal = candidate()
    proposal["predicted_output"] = rational(0)
    report = er.check_candidate(proposal)
    assert report["checks"] == {"arithmetic": False, "meaning": True, "approximation": True}
    assert report["status"] == "rejected"


@pytest.mark.parametrize("bound", [rational(0), rational(1, 1000), rational(-1), rational(1)])
def test_too_tight_negative_or_nonmatching_bound_is_rejected(bound):
    proposal = candidate()
    proposal["error_bound"] = bound
    report = er.check_candidate(proposal)
    assert report["checks"] == {"arithmetic": True, "meaning": True, "approximation": False}
    assert report["status"] == "rejected"


def test_variation_outside_declared_radius_is_reported_without_faking_a_domain_error():
    proposal = candidate(3, Fraction(1, 5), Fraction(1, 10))
    report = er.check_candidate(proposal)
    assert report["checks"] == {"arithmetic": True, "meaning": True, "approximation": False}
    assert report["status"] == "rejected"


def test_proposal_and_report_do_not_alias_supplied_mutable_objects():
    x = rational(3)
    proposal = er.make_candidate(x, rational(1, 10), rational(1, 10))
    original = deepcopy(proposal)
    x["numerator"] = 8
    assert proposal == original
    report = er.check_candidate(proposal)
    proposal["x"]["numerator"] = 4
    assert report["candidate"] == original
    a, b = er.specification(), er.specification()
    a["unexpected"] = "mutation"
    assert "unexpected" not in b


@pytest.mark.parametrize("bad", [
    None, 3, 3.0, "3", True, [3, 1],
    {"numerator": 3}, {"denominator": 1},
    {"numerator": 3, "denominator": 1, "extra": 0},
    {"numerator": True, "denominator": 1},
    {"numerator": 3, "denominator": False},
    {"numerator": 3.0, "denominator": 1},
    {"numerator": float("nan"), "denominator": 1},
    {"numerator": float("inf"), "denominator": 1},
    {"numerator": 3, "denominator": 0},
    {"numerator": -3, "denominator": -1},
    {"numerator": 2, "denominator": 2},
    {"numerator": 0, "denominator": 2},
    {"numerator": 1, "denominator": 1000001},
    {"numerator": 1000001, "denominator": 1000000},
])
def test_malformed_or_noncanonical_rational_input_is_refused(bad):
    with pytest.raises(ValueError):
        er.make_candidate(bad, rational(0), rational(0))


@pytest.mark.parametrize("x, delta, radius", [
    (101, 0, 0), (-101, 0, 0), (0, 101, 0), (0, 0, 101), (0, 0, -1),
    (100, 0, Fraction(1, 10)), (-100, 0, Fraction(1, 10)),
    (100, Fraction(1, 10), 0), (-100, Fraction(-1, 10), 0),
])
def test_source_and_neighbourhood_domain_violations_are_refused(x, delta, radius):
    with pytest.raises(ValueError):
        candidate(x, delta, radius)


@pytest.mark.parametrize("change", [
    lambda item: item.pop("model_output"),
    lambda item: item.update(expression="__import__('os').getcwd()"),
    lambda item: item.update(execution_id="invented"),
    lambda item: item.update(x=rational(101)),
    lambda item: item.update(radius=rational(-1)),
    lambda item: item.update(baseline_output={"numerator": 2 ** 255, "denominator": 1}),
    lambda item: item.update(baseline_output={"numerator": 1, "denominator": 2 ** 255}),
    lambda item: item.update(residual={"numerator": False, "denominator": 1}),
])
def test_candidate_schema_domain_and_output_limbs_are_bounded(change):
    proposal = candidate()
    change(proposal)
    with pytest.raises(ValueError):
        er.check_candidate(proposal)


def test_unknown_specification_digest_is_not_silently_rebound():
    proposal = candidate()
    proposal["specification_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        er.check_candidate(proposal)


def test_output_limb_limit_is_inclusive_and_does_not_make_false_values_true():
    proposal = candidate()
    proposal["baseline_output"] = rational(2 ** 255 - 1)
    report = er.check_candidate(proposal)
    assert report["status"] == "rejected"


def test_reopen_and_render_do_not_recompute_mathematical_claims(tmp_path, monkeypatch):
    report = er.check_candidate(candidate())
    destination = tmp_path / "check.json"
    er.save_check(destination, report)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection invoked fresh mathematical checking")

    monkeypatch.setattr(er, "check_candidate", forbidden)
    monkeypatch.setattr(er, "make_candidate", forbidden)
    loaded = er.load_check(destination)
    assert loaded == report
    er.inspect_check(loaded)
    assert isinstance(er.render_check(loaded), str)
    assert isinstance(er.render_check(loaded, details=True), str)
    assert destination.read_bytes() == canonical(report)


def test_save_does_not_overwrite_retained_record(tmp_path):
    report = er.check_candidate(candidate())
    destination = tmp_path / "check.json"
    er.save_check(destination, report)
    with pytest.raises((FileExistsError, ValueError)):
        er.save_check(destination, report)
    assert er.load_check(destination) == report


def test_unsealed_and_resealed_structural_tampering_is_refused():
    original = er.check_candidate(candidate())
    changed = deepcopy(original)
    changed["status"] = "rejected"
    with pytest.raises(ValueError):
        er.inspect_check(changed)
    with pytest.raises(ValueError):
        er.inspect_check(reseal(changed))
    changed = deepcopy(original)
    changed["checks"]["meaning"] = 1
    with pytest.raises(ValueError):
        er.inspect_check(reseal(changed))
    changed = deepcopy(original)
    changed["authority"]["hardware_actuation"] = "authorized"
    with pytest.raises(ValueError):
        er.inspect_check(reseal(changed))
    changed = deepcopy(original)
    changed["bindings"]["candidate_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        er.inspect_check(reseal(changed))


def test_coherent_forged_outcome_is_retained_only_and_fresh_check_still_rejects(tmp_path, monkeypatch):
    proposal = handwritten_candidate()
    proposal["local_multiplier"] = rational(7)
    report = er.check_candidate(proposal)
    assert report["status"] == "rejected"
    report["checks"] = {"arithmetic": True, "meaning": True, "approximation": True}
    report["status"] = "accepted"
    report = reseal(report)
    destination = tmp_path / "coherently-rewritten.json"
    destination.write_bytes(canonical(report))
    fresh_checker = er.check_candidate

    def forbidden(*args, **kwargs):
        raise AssertionError("Reopening cannot substitute fresh checking for inspection")

    monkeypatch.setattr(er, "check_candidate", forbidden)
    retained = er.load_check(destination)
    assert retained == report
    text = er.render_check(retained)
    assert "retained" in text.lower()
    assert "not freshly rechecked" in text.lower()
    assert fresh_checker(retained["candidate"])["status"] == "rejected"


@pytest.mark.parametrize("mutate", [
    lambda report: report.update(schema="ciw.exact-response-check.v2"),
    lambda report: report.update(execution_id="invented"),
    lambda report: report["checks"].update(physical=True),
    lambda report: report["checks"].pop("meaning"),
    lambda report: report["checks"].update(meaning=1),
    lambda report: report["checks"].update(meaning=None),
    lambda report: report["specification"].update(domain={}),
    lambda report: report["specification"].update(input_limb_limit=1000000.0),
    lambda report: report["bindings"].update(statement_digest="sha256:" + "0" * 64),
    lambda report: report["authority"].update(verification_id="sha256:" + "0" * 64),
    lambda report: report["authority"].update(proof_id="sha256:" + "0" * 64),
])
def test_retained_schema_and_fixed_claim_boundaries_are_strict(mutate):
    report = er.check_candidate(handwritten_candidate())
    mutate(report)
    with pytest.raises(ValueError):
        er.inspect_check(reseal(report))


def test_reference_check_emits_no_execution_result_or_proof_occurrence():
    report = er.check_candidate(handwritten_candidate())
    assert not {"execution_id", "result_id", "verification_id", "proof_id"}.intersection(report)
    for name in ("execution_id", "result_id", "verification_id", "proof_id"):
        assert report["authority"][name] is None
    assert report["authority"]["formal_verification"] == "not_performed"
    assert report["authority"]["sp1_verification"] == "not_performed"
    assert report["authority"]["state_admission"] == "not_performed"
    assert report["authority"]["hardware_actuation"] == "not_performed"


def test_exact_check_has_no_tolerance_that_can_hide_small_error():
    proposal = handwritten_candidate()
    proposal["local_multiplier"] = rational(6000000000000001, 1000000000000000)
    report = er.check_candidate(proposal)
    assert report["checks"]["meaning"] is False
    assert report["status"] == "rejected"


def test_retained_report_requires_canonical_bytes_and_unique_keys(tmp_path):
    report = er.check_candidate(handwritten_candidate())
    source = tmp_path / "check.json"
    source.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        er.load_check(source)
    source.write_bytes(canonical(report)[:-1] + b',"status":"accepted"}')
    with pytest.raises(ValueError):
        er.load_check(source)


@pytest.mark.parametrize("payload", [
    b'{"x":0,"x":1}',
    b'{"x":NaN}',
    b'{"x":Infinity}',
    b'{"x":1} trailing',
    b'\xff',
    b'[]',
    b'null',
])
def test_candidate_loader_refuses_ambiguous_or_malformed_json(tmp_path, payload):
    source = tmp_path / "candidate.json"
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        er.load_candidate(source)


def test_bounded_candidate_loader_and_deep_nesting(tmp_path):
    source = tmp_path / "candidate.json"
    source.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError):
        er.load_candidate(source)
    source.write_bytes(b"[" * 1000 + b"0" + b"]" * 1000)
    with pytest.raises(ValueError):
        er.load_candidate(source)


def test_candidate_file_has_no_execution_side_effects(tmp_path):
    proposal = candidate()
    source = tmp_path / "candidate.json"
    source.write_text(json.dumps(proposal), encoding="utf-8")
    assert er.load_candidate(source) == proposal
    proposal["program"] = "open('should-not-exist', 'w').write('bad')"
    source.write_text(json.dumps(proposal), encoding="utf-8")
    with pytest.raises(ValueError):
        er.load_candidate(source)
    assert not (tmp_path / "should-not-exist").exists()


def test_cli_create_inspect_rejected_and_malformed_statuses(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts" / "check_linear_response.py"))["main"]
    source = tmp_path / "candidate.json"
    output = tmp_path / "check.json"
    source.write_bytes(canonical(candidate()))
    assert main(["check-exact", str(source), "--output", str(output)]) == 0
    assert main(["inspect-exact", str(output)]) == 0
    assert main(["inspect-exact", str(output), "--details"]) == 0
    assert main(["check-exact", str(source), "--output", str(output)]) == 2
    rejected = candidate()
    rejected["local_multiplier"] = rational(7)
    source.write_bytes(canonical(rejected))
    rejected_output = tmp_path / "rejected.json"
    assert main(["check-exact", str(source), "--output", str(rejected_output)]) == 1
    assert er.load_check(rejected_output)["status"] == "rejected"
    assert main(["inspect-exact", str(rejected_output)]) == 0
    source.write_text('{"x":null}', encoding="utf-8")
    malformed_output = tmp_path / "malformed.json"
    assert main(["check-exact", str(source), "--output", str(malformed_output)]) == 2
    assert not malformed_output.exists()
    capsys.readouterr()


def test_published_exact_examples_are_data_only_and_have_declared_outcomes():
    directory = Path(__file__).resolve().parents[1] / "examples" / "linear-response"
    for filename, expected in [("exact-square.json", "accepted"),
                               ("exact-square-wrong-multiplier.json", "rejected")]:
        proposal = er.load_candidate(directory / filename)
        assert er.check_candidate(proposal)["status"] == expected
