"""Independent arithmetic and authority checks for educational response views."""
from copy import deepcopy
from pathlib import Path
import runpy
import sys

import pytest

from ciw import julia_oscillator as jo
from ciw import linear_response as lr
from ciw.telemetry import canonical, digest


def oscillator_source():
    return {
        "schema": jo.SOURCE_SCHEMA,
        "experiment_id": "linear-response-oscillator-v1",
        "operation_id": jo.OPERATION,
        "model": {"omega_0_rad_s": 2.0, "gamma_s_inv": 0.1, "mass_kg": 2.0},
        "initial_state": {"q0_m": 1.0, "v0_m_s": -0.25},
        "time_s": [0.0, 0.1, 0.2],
        "solver": {"abstol": 1e-10, "reltol": 1e-10, "maxiters": 100000},
        "claim_scope": "simulated_numerical_trajectory_against_independent_analytic_oracle",
    }


def view(profile, base, delta, source=None):
    return lr.preview(lr.make_request(profile, base, delta, source=source))


def assert_matrix(actual, expected):
    assert len(actual) == len(expected)
    for row, wanted in zip(actual, expected):
        assert row == pytest.approx(wanted, rel=1e-12, abs=1e-12)


def reseal(record):
    """An editable content digest is no substitute for structural validation."""
    record["preview_id"] = digest({key: value for key, value in record.items()
                                   if key != "preview_id"})
    return record


def test_scalar_derivative_and_finite_change_ratio_are_distinct():
    record = view("scalar-square.v1", [3.0], [0.1])
    result = record["calculation"]
    assert result["baseline_output"] == [9.0]
    assert_matrix(result["jacobian"], [[6.0]])
    assert_matrix(result["contributions"], [[0.6]])
    assert result["predicted_delta"] == pytest.approx([0.6])
    assert result["predicted_output"] == pytest.approx([9.6])
    assert result["model_output"] == pytest.approx([9.61])
    assert result["residual"] == pytest.approx([0.01])
    assert result["finite_change_ratio"] == pytest.approx([6.1])
    assert record["profile"]["scope"] == "local_first_order"

    halved = view("scalar-square.v1", [3.0], [0.05])["calculation"]
    assert halved["residual"] == pytest.approx([0.0025])
    assert halved["finite_change_ratio"] == pytest.approx([6.05])
    zero = view("scalar-square.v1", [3.0], [0.0])["calculation"]
    assert zero["finite_change_ratio"] is None
    assert_matrix(zero["jacobian"], [[6.0]])


def test_affine_offset_is_retained_and_change_relation_is_exact():
    record = view("affine.v1", [3.0, 4.0], [0.1, -0.2])
    result = record["calculation"]
    assert result["baseline_output"] == [15.0, 7.0]
    assert_matrix(result["jacobian"], [[2.0, 1.0], [-1.0, 3.0]])
    assert_matrix(result["contributions"], [[0.2, -0.2], [-0.1, -0.6]])
    assert result["predicted_delta"] == pytest.approx([0.0, -0.7])
    assert result["predicted_output"] == pytest.approx([15.0, 6.3])
    assert result["model_output"] == pytest.approx([15.0, 6.3])
    assert result["residual"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert result["finite_change_ratio"] is None
    assert record["profile"]["scope"] == "exact_affine_change"


def test_nonlinear_contributions_and_known_remainders():
    record = view("nonlinear-vector.v1", [2.0, 3.0], [0.1, -0.2])
    result = record["calculation"]
    assert result["baseline_output"] == [7.0, 6.0]
    assert_matrix(result["jacobian"], [[4.0, 1.0], [3.0, 2.0]])
    assert_matrix(result["contributions"], [[0.4, -0.2], [0.3, -0.4]])
    assert result["predicted_delta"] == pytest.approx([0.2, -0.1])
    assert result["predicted_output"] == pytest.approx([7.2, 5.9])
    assert result["model_output"] == pytest.approx([7.21, 5.88])
    assert result["residual"] == pytest.approx([0.01, -0.02])
    assert record["profile"]["scope"] == "local_first_order"

    smaller = view("nonlinear-vector.v1", [2.0, 3.0], [0.05, -0.1])
    assert smaller["calculation"]["residual"] == pytest.approx([0.0025, -0.005])
    larger = view("nonlinear-vector.v1", [2.0, 3.0], [1.0, -2.0])
    assert larger["calculation"]["delta"] == [1.0, -2.0]
    assert larger["calculation"]["residual"] == pytest.approx([1.0, -2.0])


def test_zero_first_order_response_does_not_erase_higher_order_change():
    result = view("nonlinear-vector.v1", [0.0, 0.0], [0.1, 0.0])["calculation"]
    assert_matrix(result["jacobian"], [[0.0, 1.0], [0.0, 0.0]])
    assert result["predicted_delta"] == [0.0, 0.0]
    assert result["actual_delta"] == pytest.approx([0.01, 0.0])
    assert result["residual"] == pytest.approx([0.01, 0.0])


def test_chain_rule_uses_outer_derivative_at_inner_output_in_correct_order():
    result = view("composed-vector.v1", [2.0, 3.0], [0.1, -0.2])["calculation"]
    assert result["baseline_output"] == [49.0, 6.0]
    assert_matrix(result["jacobian"], [[56.0, 14.0], [3.0, 2.0]])
    assert result["predicted_delta"] == pytest.approx([2.8, -0.1])
    assert result["predicted_output"] == pytest.approx([51.8, 5.9])
    assert result["model_output"] == pytest.approx([51.9841, 5.88])
    assert result["residual"] == pytest.approx([0.1841, -0.02])


def test_rectangular_response_helper_and_row_accounting():
    result = lr.response_arithmetic([10.0, -2.0], [[1.0, 2.0, -1.0], [0.0, 3.0, 4.0]],
                                    [0.5, -1.0, 2.0], [6.5, 3.0])
    assert_matrix(result["contributions"], [[0.5, -2.0, -2.0], [0.0, -3.0, 8.0]])
    assert result["predicted_delta"] == pytest.approx([-3.5, 5.0])
    assert result["predicted_output"] == pytest.approx([6.5, 3.0])
    for contributions, change in zip(result["contributions"], result["predicted_delta"]):
        assert sum(contributions) == pytest.approx(change, abs=1e-12)


def test_singular_response_helper_does_not_assume_an_inverse():
    result = lr.response_arithmetic([5.0, 10.0], [[1.0, 2.0], [2.0, 4.0]],
                                    [2.0, -1.0], [5.0, 10.0])
    assert result["predicted_delta"] == [0.0, 0.0]
    assert result["predicted_output"] == [5.0, 10.0]
    assert result["residual"] == [0.0, 0.0]


def test_coordinate_permutation_requires_matching_matrix_columns():
    original = lr.response_arithmetic([7.0, 6.0], [[4.0, 1.0], [3.0, 2.0]],
                                      [0.1, -0.2], [7.21, 5.88])
    reordered = lr.response_arithmetic([7.0, 6.0], [[1.0, 4.0], [2.0, 3.0]],
                                       [-0.2, 0.1], [7.21, 5.88])
    relabel_only = lr.response_arithmetic([7.0, 6.0], [[4.0, 1.0], [3.0, 2.0]],
                                         [-0.2, 0.1], [7.21, 5.88])
    assert reordered["predicted_delta"] == pytest.approx(original["predicted_delta"])
    assert reordered["predicted_output"] == pytest.approx(original["predicted_output"])
    assert relabel_only["predicted_delta"] != pytest.approx(original["predicted_delta"])


def test_absolute_value_at_zero_refuses_derivative_claim():
    with pytest.raises(ValueError):
        view("absolute.v1", [0.0], [0.1])
    positive = view("absolute.v1", [2.0], [0.1])["calculation"]
    negative = view("absolute.v1", [-2.0], [0.1])["calculation"]
    assert_matrix(positive["jacobian"], [[1.0]])
    assert_matrix(negative["jacobian"], [[-1.0]])


def test_oscillator_rate_is_state_to_rate_with_fixed_parameters():
    source = oscillator_source()
    before = deepcopy(source)
    record = view("oscillator-rate.v1", [1.0, -0.25], [0.1, 0.2], source)
    result = record["calculation"]
    assert result["baseline_output"] == pytest.approx([-0.25, -3.95])
    assert_matrix(result["jacobian"], [[0.0, 1.0], [-4.0, -0.2]])
    assert result["predicted_delta"] == pytest.approx([0.2, -0.44])
    assert result["model_output"] == pytest.approx([-0.05, -4.39])
    assert result["residual"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert record["profile"]["scope"] == "exact_affine_change"
    assert record["profile"]["input_units"] == ["m", "m/s"]
    assert record["profile"]["output_units"] == ["m/s", "m/s^2"]
    assert record["profile"]["map_class"] == "linear_state_to_rate"
    assert record["binding"]["integration"]["performed"] is False
    assert record["binding"]["integration"]["trajectory_execution_ref"] is None
    assert record["binding"]["fixed_parameters"] == source["model"]
    assert record["binding"]["source_digest"] == digest(source)
    assert record["binding"]["changed_source_digest"] != digest(source)
    assert source == before


def test_oscillator_energy_gradient_has_nonunit_mass_and_quadratic_remainder():
    record = view("oscillator-energy.v1", [1.0, -0.25], [0.1, 0.2], oscillator_source())
    result = record["calculation"]
    assert result["baseline_output"] == [4.0625]
    assert_matrix(result["jacobian"], [[8.0, -0.5]])
    assert_matrix(result["contributions"], [[0.8, -0.1]])
    assert result["predicted_delta"] == pytest.approx([0.7])
    assert result["actual_delta"] == pytest.approx([0.78])
    assert result["residual"] == pytest.approx([0.08])
    assert result["finite_change_ratio"] is None
    assert record["profile"]["output_units"] == ["J"]
    assert record["profile"]["scope"] == "local_first_order"


def test_undamped_initial_state_change_need_not_preserve_energy():
    source = oscillator_source()
    source["model"]["gamma_s_inv"] = 0.0
    result = view("oscillator-energy.v1", [1.0, -0.25], [0.1, 0.2], source)["calculation"]
    assert result["actual_delta"] == pytest.approx([0.78])


def test_energy_projection_matches_existing_model_at_initial_state():
    source = oscillator_source()
    record = view("oscillator-energy.v1", [1.0, -0.25], [0.1, 0.2], source)
    # Existing oracle is a compatibility check; golden constants above remain
    # the independently specified expected values for this new calculation.
    reference = jo.analytic_oracle(source)
    assert record["calculation"]["baseline_output"] == pytest.approx([reference["energy_j"][0]])
    changed = deepcopy(source)
    changed["initial_state"] = {"q0_m": 1.1, "v0_m_s": -0.05}
    assert record["calculation"]["model_output"] == pytest.approx([jo.analytic_oracle(changed)["energy_j"][0]])


@pytest.mark.parametrize("base,delta", [([0.0, 0.0], [0.1, 0.2]),
                                       ([1.0, -0.25], [10.0, 0.0]),
                                       ([1.0, -0.25], [0.0, 101.0])])
def test_oscillator_requires_valid_initial_anchor_and_changed_source(base, delta):
    with pytest.raises(ValueError):
        view("oscillator-rate.v1", base, delta, oscillator_source())


@pytest.mark.parametrize("profile,base,delta", [
    ("unknown.v1", [1.0], [0.1]),
    ("scalar-square.v1", [], []),
    ("scalar-square.v1", [1.0, 2.0], [0.1, 0.2]),
    ("nonlinear-vector.v1", [1.0, 2.0], [0.1]),
    ("scalar-square.v1", [True], [0.1]),
    ("scalar-square.v1", ["3"], [0.1]),
    ("scalar-square.v1", [1e300], [0.1]),
    ("scalar-square.v1", (3.0,), [0.1]),
    ("scalar-square.v1", [3.0], (0.1,)),
    ("scalar-square.v1", {"x": 3.0}, [0.1]),
    ("scalar-square.v1", [[3.0]], [0.1]),
])
def test_bounded_allowlisted_data_only_requests(profile, base, delta):
    with pytest.raises(ValueError):
        view(profile, base, delta)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_and_perturbation_refused(bad):
    with pytest.raises(ValueError):
        view("scalar-square.v1", [bad], [0.1])
    with pytest.raises(ValueError):
        view("scalar-square.v1", [3.0], [bad])
    with pytest.raises(ValueError):
        lr.response_arithmetic([1.0], [[bad]], [0.1], [2.0])


@pytest.mark.parametrize("baseline,jacobian,delta,model", [
    ([1.0, 2.0], [[1.0]], [0.1], [1.1, 2.1]),
    ([1.0], [[1.0, 2.0]], [0.1], [1.1]),
    ([1.0, 2.0], [[1.0, 2.0], [3.0]], [0.1, 0.2], [1.1, 2.1]),
    ([1.0], [[1.0]], [0.1], [1.1, 2.1]),
    ([], [], [], []),
])
def test_response_helper_refuses_inconsistent_shapes(baseline, jacobian, delta, model):
    with pytest.raises(ValueError):
        lr.response_arithmetic(baseline, jacobian, delta, model)


@pytest.mark.parametrize("field,value", [
    ("input_order", ["x2", "x1"]),
    ("input_units", ["m", "m"]),
    ("frame", "unbound-other-frame"),
    ("anchor", "arbitrary-trajectory-sample"),
    ("executable", "untrusted.py"),
])
def test_request_metadata_cannot_relabel_an_input(field, value):
    request = lr.make_request("nonlinear-vector.v1", [2.0, 3.0], [0.1, -0.2])
    request[field] = value
    with pytest.raises(ValueError):
        lr.preview(request)


def test_new_preview_does_not_assign_execution_result_or_verification_identity():
    record = view("scalar-square.v1", [3.0], [0.1])
    assert record["schema"] == "ciw.linear-response-preview.v1"
    assert record["preview_id"].startswith("sha256:")
    assert record["request_digest"] == digest(record["request"])
    assert record["profile"]["derivative_method"] == "analytic_code_owned"
    assert record["authority"] == {
        "kind": "analytic_educational_preview", "execution_id": None, "result_id": None,
        "verification_id": None, "provider_execution": "not_performed",
        "physical_validation": "not_established", "measurement_uncertainty": "not_declared",
        "state_admission": "not_performed", "hardware_actuation": "not_performed",
    }


def test_make_request_copies_input_source_and_does_not_accept_supplied_derivatives():
    source = oscillator_source()
    base, delta = [1.0, -0.25], [0.1, 0.2]
    request = lr.make_request("oscillator-energy.v1", base, delta, source=source)
    source["model"]["mass_kg"] = 10.0
    base[0] = 9.0
    delta[0] = 0.9
    record = lr.preview(request)
    assert record["calculation"]["baseline_output"] == [4.0625]
    assert record["calculation"]["delta"] == [0.1, 0.2]
    request["jacobian"] = [[8.0, -0.5]]
    with pytest.raises(ValueError):
        lr.preview(request)


def test_arithmetic_helper_does_not_promote_a_caller_declared_matrix():
    result = lr.response_arithmetic([9.0], [[123.0]], [0.1], [9.61])
    assert result["predicted_delta"] == pytest.approx([12.3])
    assert "derivative_method" not in result
    assert "authority" not in result
    assert "verification_id" not in result


@pytest.mark.parametrize("profile,base,delta,source", [
    ("nonlinear-vector.v1", [2.0, 3.0], [0.1, -0.2], None),
    ("oscillator-energy.v1", [1.0, -0.25], [0.1, 0.2], oscillator_source()),
])
def test_save_reopen_inspection_is_offline_and_does_not_recompute(
        tmp_path, monkeypatch, profile, base, delta, source):
    record = view(profile, base, delta, source)
    path = tmp_path / "response.json"
    lr.save_preview(path, record)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not evaluate a model or provider")

    monkeypatch.setattr(lr, "_evaluate", forbidden)
    monkeypatch.setattr(lr, "_jacobian", forbidden)
    monkeypatch.setattr(jo, "analytic_oracle", forbidden)
    monkeypatch.setattr(jo, "_invoke", forbidden)
    reopened = lr.load_preview(path)
    assert reopened == record
    inspected = lr.inspect_preview(reopened)
    assert inspected == record
    assert inspected is not reopened
    inspected["calculation"]["base_point"][0] += 1.0
    assert reopened == record
    assert "model" in lr.render_preview(reopened).lower()
    assert "analytic" in lr.render_preview(reopened, details=True).lower()


@pytest.mark.parametrize("path,value", [
    (("authority", "execution_id"), "invented-execution"),
    (("authority", "physical_validation"), "verified"),
    (("profile", "scope"), "exact_affine_change"),
    (("binding", "source_digest"), "sha256:" + "0" * 64),
    (("binding", "changed_source_digest"), "sha256:" + "0" * 64),
    (("binding", "fixed_parameters"), {"omega_0_rad_s": 2.0, "gamma_s_inv": 0.1, "mass_kg": 1.0}),
    (("profile", "derivative_method"), "provider_verified"),
])
def test_resealed_authority_scope_and_source_binding_changes_are_refused(path, value):
    record = view("oscillator-energy.v1", [1.0, -0.25], [0.1, 0.2], oscillator_source())
    record[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        lr.inspect_preview(reseal(record))


def test_resealed_arithmetic_inconsistency_and_missing_binding_are_refused():
    record = view("scalar-square.v1", [3.0], [0.1])
    changed = deepcopy(record)
    changed["calculation"]["predicted_output"][0] = 42.0
    with pytest.raises(ValueError):
        lr.inspect_preview(reseal(changed))
    del record["binding"]
    with pytest.raises(ValueError):
        lr.inspect_preview(reseal(record))


@pytest.mark.parametrize("field,value", [("gamma_s_inv", False), ("integration_performed", 0)])
def test_resealed_binding_rejects_boolean_number_aliases(field, value):
    source = oscillator_source()
    source["model"]["gamma_s_inv"] = 0.0
    record = view("oscillator-rate.v1", [1.0, -0.25], [0.1, 0.2], source)
    if field == "integration_performed":
        record["binding"]["integration"]["performed"] = value
    else:
        record["binding"]["fixed_parameters"][field] = value
    with pytest.raises(ValueError):
        lr.inspect_preview(reseal(record))


def test_reopen_refuses_unsealed_tampering(tmp_path):
    record = view("scalar-square.v1", [3.0], [0.1])
    record["calculation"]["model_output"][0] = 123.0
    path = tmp_path / "changed.json"
    path.write_bytes(canonical(record))
    with pytest.raises(ValueError):
        lr.load_preview(path)


def test_saved_preview_cannot_be_overwritten(tmp_path):
    original = view("scalar-square.v1", [3.0], [0.1])
    path = tmp_path / "response.json"
    lr.save_preview(path, original)
    with pytest.raises(FileExistsError):
        lr.save_preview(path, view("scalar-square.v1", [3.0], [0.05]))
    assert lr.load_preview(path) == original


def test_reopen_refuses_duplicate_json_fields(tmp_path):
    record = view("scalar-square.v1", [3.0], [0.1])
    path = tmp_path / "duplicate.json"
    raw = canonical(record)
    path.write_bytes(b'{"schema":"ciw.linear-response-preview.v1",' + raw[1:])
    with pytest.raises(ValueError):
        lr.load_preview(path)


@pytest.fixture
def response_cli(monkeypatch):
    # Loading the thin caller must not evaluate a profile or start its CLI.
    monkeypatch.setattr(sys, "path", list(sys.path))
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_linear_response.py"
    return runpy.run_path(str(script), run_name="linear_response_cli_test")["main"]


@pytest.mark.parametrize("name", ["scalar", "scalar-half", "affine", "nonlinear",
                                  "nonlinear-origin", "chain", "oscillator-rate", "oscillator-energy"])
def test_runnable_examples_create_valid_previews(response_cli, tmp_path, capsys, name):
    request = Path(__file__).resolve().parents[1] / "examples" / "linear-response" / (name + ".json")
    output = tmp_path / (name + "-preview.json")
    assert response_cli(["create", str(request), "--output", str(output)]) == 0
    record = lr.load_preview(output)
    text = capsys.readouterr().out
    assert "model-evaluated" in text
    assert record["preview_id"] in text
    assert record["authority"]["provider_execution"] == "not_performed"


def test_terminal_create_inspect_and_overwrite_refusal(response_cli, tmp_path, capsys, monkeypatch):
    request = tmp_path / "request.json"
    request.write_bytes(canonical(lr.make_request("scalar-square.v1", [3.0], [0.1])))
    output = tmp_path / "preview.json"
    assert response_cli(["create", str(request), "--output", str(output)]) == 0
    first_output = capsys.readouterr().out
    before = output.read_bytes()
    assert response_cli(["create", str(request), "--output", str(output)]) == 2
    assert "refused" in capsys.readouterr().err.lower()
    assert output.read_bytes() == before

    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal inspection must be evaluation-free")

    monkeypatch.setattr(lr, "_evaluate", forbidden)
    monkeypatch.setattr(lr, "_jacobian", forbidden)
    monkeypatch.setattr(jo, "_invoke", forbidden)
    assert response_cli(["inspect", str(output)]) == 0
    assert capsys.readouterr().out == first_output


@pytest.mark.parametrize("invalid_kind", ["duplicate", "malformed", "oversized"])
def test_terminal_rejects_invalid_data_without_saving(response_cli, tmp_path, capsys, invalid_kind):
    if invalid_kind == "duplicate":
        raw = canonical(lr.make_request("scalar-square.v1", [3.0], [0.1]))
        raw = b'{"profile":"scalar-square.v1",' + raw[1:]
    elif invalid_kind == "oversized":
        raw = b" " * (lr.MAX_BYTES + 1)
    else:
        raw = b'{"profile":'
    request = tmp_path / "invalid.json"
    request.write_bytes(raw)
    output = tmp_path / "preview.json"
    assert response_cli(["create", str(request), "--output", str(output)]) == 2
    assert "refused" in capsys.readouterr().err.lower()
    assert not output.exists()
