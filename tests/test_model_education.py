import copy

import pytest

from ciw.julia_oscillator import OPERATION, analytic_oracle
from ciw.model_education import (PREVIEW_SCHEMA, SCHEMA, SWEEP_SCHEMA,
                                  oscillator_model_card, parse_overrides,
                                  preview_oscillator, sensitivity_oscillator)


def source():
    return {
        "schema": "ciw.julia-oscillator-source.v1",
        "experiment_id": "education-oscillator-v1",
        "operation_id": OPERATION,
        "model": {"omega_0_rad_s": 2.0, "gamma_s_inv": 0.1, "mass_kg": 1.0},
        "initial_state": {"q0_m": 1.0, "v0_m_s": 0.0},
        "time_s": [0.0, 0.1, 0.2, 0.3],
        "solver": {"abstol": 1e-10, "reltol": 1e-10, "maxiters": 100000},
        "claim_scope": "simulated_numerical_trajectory_against_independent_analytic_oracle",
    }


def result_data(value=None):
    value = source() if value is None else value
    output = analytic_oracle(value)
    return {"operation_id": OPERATION, "output": output}


def test_model_card_exposes_equations_assumptions_and_provenance():
    provenance = {"source_id": "source-1", "evidence_id": "evidence-1",
                  "result_id": "result-1", "execution_id": "execution-1"}
    card = oscillator_model_card(source(), result_data(), provenance)
    assert card["schema"] == SCHEMA
    assert card["model_card_id"].startswith("sha256:")
    assert [equation["id"] for equation in card["equations"]] == [
        "state_position", "state_velocity", "mechanical_energy"]
    assert card["derived"]["regime"] == "underdamped"
    assert card["derived"]["energy_behavior"] == "decreasing_in_this_trajectory"
    assert card["authority"]["physical_validation"] == "not_established"
    assert card["provenance"] == provenance
    assert card["augmentation"]["requires_new_retained_execution"] is True


def test_preview_is_bounded_hypothetical_and_changes_the_model():
    preview = preview_oscillator(source(), {"model.gamma_s_inv": 0.0})
    assert preview["schema"] == PREVIEW_SCHEMA
    assert preview["authority"]["execution_id"] == "not_assigned"
    assert preview["authority"]["result_id"] == "not_assigned"
    assert preview["authority"]["retention"] == "not_performed"
    assert preview["output"]["energy_j"][-1] == pytest.approx(preview["output"]["energy_j"][0])
    assert preview["comparison"]["q_m"]["max_abs_delta"] > 0
    assert preview["comparison"]["energy"]["preview_final_j"] == pytest.approx(
        preview["comparison"]["energy"]["preview_initial_j"])
    assert preview["comparison"]["energy"]["baseline_final_j"] < preview["comparison"]["energy"]["baseline_initial_j"]
    assert preview["base_source_digest"].startswith("sha256:")


@pytest.mark.parametrize("item", ["model.omega_0_rad_s", "model.gamma_s_inv", "initial_state.q0_m"])
def test_preview_override_parser_accepts_declared_paths(item):
    assert parse_overrides([item + "=1.5"])[item] == pytest.approx(1.5)


def test_preview_refuses_unknown_or_out_of_domain_controls():
    with pytest.raises(ValueError, match="Unsupported preview parameter"):
        parse_overrides(["model.damping_mode=free"])
    with pytest.raises(ValueError, match="outside its declared finite bounds"):
        preview_oscillator(source(), {"model.omega_0_rad_s": 30.0})
    changed = copy.deepcopy(source())
    changed["model"]["gamma_s_inv"] = 0.0
    assert preview_oscillator(source(), {})["base_source_digest"] != preview_oscillator(changed, {})["base_source_digest"]


def test_sensitivity_sweep_is_bounded_and_hypothetical():
    sweep = sensitivity_oscillator(source(), "model.gamma_s_inv", [0.0, 0.1, 0.2])
    assert sweep["schema"] == SWEEP_SCHEMA
    assert sweep["swept_path"] == "model.gamma_s_inv"
    assert len(sweep["cases"]) == 3
    assert sweep["authority"]["execution_id"] == "not_assigned"
    assert sweep["authority"]["result_id"] == "not_assigned"
    assert sweep["cases"][0]["comparison"]["energy"]["preview_final_j"] == pytest.approx(
        sweep["cases"][0]["comparison"]["energy"]["preview_initial_j"])
    assert sweep["sweep_id"].startswith("sha256:")


def test_sensitivity_sweep_refuses_ambiguous_or_unbounded_requests():
    with pytest.raises(ValueError, match="between two and nine"):
        sensitivity_oscillator(source(), "model.gamma_s_inv", [0.0])
    with pytest.raises(ValueError, match="distinct"):
        sensitivity_oscillator(source(), "model.gamma_s_inv", [0.0, 0.0])
    with pytest.raises(ValueError, match="Unsupported sensitivity parameter"):
        sensitivity_oscillator(source(), "model.unknown", [0.0, 0.1])

