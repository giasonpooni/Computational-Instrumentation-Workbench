"""Author the two-capacity thermal observer example and print its exact source.

The request declares a dissipative core/shell RC model, a prior, noise
covariances, three constant-input steps and a sensor selection budget. The
evaluation block is a synthetic trajectory generated from the same model with
declared process and measurement perturbations, plus one held-out step, so the
retained result can report held-out diagnostics. Nothing here touches hardware
and the noise law is declared by the operator, not authenticated.

The committed ``source.json`` is the retained input; regenerating it on another
platform may change the last digit of a few floating-point values because the
discretized matrices come from an eigendecomposition.
"""
from copy import deepcopy
import json

import numpy as np

from ciw import thermal_contract as contract
from ciw import thermal_reference as reference


def request():
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


def source():
    value = request()
    matrices = reference.model_matrices(value["model"], value["sample_interval_s"])
    ad, bd = np.asarray(matrices["Ad"]), np.asarray(matrices["Bd"])
    state = np.asarray([300.0, 300.0])
    process_noise = [[0.1, 0.0], [0.0, 0.1], [0.0, 0.1]]
    measurement_noise = [[0.5, -0.25], [0.25, 0.0], [-0.25, 0.5]]
    states, observations = [], []
    for inputs, process, noise in zip(value["inputs"], process_noise, measurement_noise):
        state = ad @ state + bd @ np.asarray(inputs) + np.asarray(process)
        states.append(state.tolist())
        observations.append((state + np.asarray(noise)).tolist())
    value["observations"] = observations
    held_input = [50.0, 290.0]
    held_state = ad @ state + bd @ np.asarray(held_input)
    return {"schema": contract.SOURCE_SCHEMA, "experiment_id": "thermal:example-two-capacity",
            "configuration": deepcopy(contract.POLICY), "request": value,
            "evaluation": {"origin": "synthetic_fixture", "generator_model": deepcopy(value["model"]),
                           "initial_state": [300.0, 300.0], "process_noise": process_noise,
                           "measurement_noise": measurement_noise, "states": states,
                           "held_out": {"input": held_input, "process_noise": [0.0, 0.0],
                                        "state": held_state.tolist(), "measurement_noise": [0.0, 0.0],
                                        "observations": held_state.tolist()},
                           "noise_law": "operator_declared_not_authenticated"}}


if __name__ == "__main__":
    print(json.dumps(source(), indent=1, sort_keys=True))
