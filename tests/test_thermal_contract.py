"""Offline thermal contract checks use an independent Python analytical reference."""

from copy import deepcopy
import json

import numpy as np
import pytest

from ciw import thermal_contract as contract
from ciw import thermal_reference as reference


def _request():
    return {
        "schema": contract.REQUEST_SCHEMA, "operation_id": contract.OPERATION_ID,
        "model": {"state_order": contract.STATE_ORDER, "state_units": ["K", "K"],
                   "input_order": contract.INPUT_ORDER, "input_units": ["W", "K"],
                   "sensor_order": contract.SENSOR_ORDER, "sensor_units": ["K", "K"],
                   "capacities_j_per_k": [100.0, 200.0], "conductances_w_per_k": [10.0, 5.0],
                   "observation_matrix": [[1, 0], [0, 1]]},
        "sample_interval_s": 1.0,
        "prior": {"mean": [300.0, 300.0], "covariance": [[4.0, 0.2], [0.2, 9.0]]},
        "process_noise_covariance": [[0.1, 0.0], [0.0, 0.2]],
        "observation_noise_covariance": [[0.5, 0.0], [0.0, 0.75]],
        "inputs": [[100.0, 290.0], [80.0, 290.0], [60.0, 290.0]],
        "observations": [[None, None], [None, None], [None, None]],
        "selection": {"costs": [1.0, 2.0], "budget": 2.0, "minimum_sensors": 1,
                       "tie_tolerance_nats": 1e-10, "tie_policy": "lowest_mask_within_tolerance"},
    }


def _source():
    request = _request()
    matrices = reference.model_matrices(request["model"], request["sample_interval_s"])
    ad, bd = np.asarray(matrices["Ad"]), np.asarray(matrices["Bd"])
    state = np.asarray([300.0, 300.0])
    process_noise = [[0.1, 0.0], [0.0, 0.1], [0.0, 0.1]]
    measurement_noise = [[0.5, -0.25], [0.25, 0.0], [-0.25, 0.5]]
    states, observations = [], []
    for inputs, process, noise in zip(request["inputs"], process_noise, measurement_noise):
        state = ad @ state + bd @ np.asarray(inputs) + np.asarray(process)
        observed = state + np.asarray(noise)
        states.append(state.tolist())
        observations.append(observed.tolist())
    request["observations"] = observations
    held_input = [50.0, 290.0]
    held_process = [0.0, 0.0]
    held_noise = [0.0, 0.0]
    held_state = ad @ state + bd @ np.asarray(held_input) + np.asarray(held_process)
    held_observations = held_state + np.asarray(held_noise)
    return {"schema": contract.SOURCE_SCHEMA, "experiment_id": "thermal:fixture",
            "configuration": deepcopy(contract.POLICY), "request": request,
            "evaluation": {"origin": "synthetic_fixture", "generator_model": deepcopy(request["model"]),
                "initial_state": [300.0, 300.0], "process_noise": process_noise,
                "measurement_noise": measurement_noise, "states": states,
                "held_out": {"input": held_input, "process_noise": held_process, "state": held_state.tolist(),
                              "measurement_noise": held_noise, "observations": held_observations.tolist()},
                "noise_law": "operator_declared_not_authenticated"}}


def _native(source):
    request = source["request"]
    result = reference.reference(request)
    result["schema"] = contract.RESULT_SCHEMA
    result["request"] = deepcopy(request)
    result["claim_scope"] = contract.CLAIM_SCOPE
    result["model"].update({"state_order": contract.STATE_ORDER, "input_order": contract.INPUT_ORDER,
                             "sensor_order": contract.SENSOR_ORDER,
                             "symbolic": {"equations": ["core equation", "shell equation"],
                                          "latex": ["core equation", "shell equation"],
                                          "native_state_order": contract.STATE_ORDER,
                                          "state_permutation": [1, 2], "rendering": "Symbolics/Latexify"}})
    result["selection"]["solver"] = {"name": "HiGHS", "termination_status": "OPTIMAL",
        "primal_status": "FEASIBLE_POINT", "objective_value": result["selection"]["objective_nats"],
        "objective_bound": result["selection"]["objective_nats"], "relative_gap": 0.0,
        "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9,
        "mip_feasibility_tolerance": 1e-9, "mip_relative_gap_tolerance": 0.0}
    return result


def test_thermal_source_reference_result_and_replay_roundtrip():
    source = _source()
    checked = contract.validate_source(contract.canonical(source))
    native = _native(checked)
    contract.validate_result(checked["request"], native)
    data = {"native": native, "evaluation": reference.evaluate(checked, native["observer"])}
    contract.validate_data(checked, data)
    reopened = json.loads(contract.canonical({"native": native, "evaluation": data["evaluation"]}))
    contract.validate_data(checked, reopened)
    assert data["evaluation"]["physical_validation"] == "not_established"


def test_thermal_contract_rejects_changed_request_binding():
    source = _source()
    native = _native(source)
    changed = deepcopy(source["request"])
    changed["sample_interval_s"] = 2.0
    with pytest.raises(ValueError, match="request or authority binding"):
        contract.validate_result(changed, native)
