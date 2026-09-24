"""Provider-free thermal observer workflow with native CIW identities.

The Python reference is the first executable operation for the thermal contract.
It is deliberately separate from the Julia worker: the retained result records
which implementation produced them, while the contract can later validate a
Julia occurrence against the same independent reference.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import reference_workflow as base
from . import thermal_contract as contract
from . import thermal_reference as reference

KIND = "thermal-observer"
SCHEMA = "ciw.thermal-observer-session.v1"
OPERATION = "ciw.thermal-observer.v1"
SOURCE_SCHEMA = contract.SOURCE_SCHEMA
RESULT_SCHEMA = "ciw.thermal-observer-workbench-result.v1"
VERIFY_SCHEMA = "ciw.thermal-observer-verification.v1"
PROFILE = "ciw.thermal-observer.python-reference.v1"
ROLE = "thermal"
ROLES = set()
# A retained bundle carries the base64 source plus the result, its numerical
# copy, the reproduction and that copy, so the budget is a multiple of the
# contract's result limit, as for the other references.
MAX_BYTES = 4 * contract.RESULT_LIMIT
AUTHORITY = {
    "physical_validation": "not_established",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}
# Provenance the Python reference always records; a retained result claiming
# the Julia worker's rendering or solver is not this reference's result.
SYMBOLIC_RENDERING = "python-reference"
SOLVER_NAME = "python-reference-enumeration"


@lru_cache(maxsize=1)
def _algorithm_identity():
    return base.algorithm_identity(PROFILE, [Path(contract.__file__), Path(reference.__file__),
                                             Path(base.__file__), Path(__file__)],
                                   numpy_version=np.__version__)


def runtime_identity():
    return {
        "schema": base.RUNTIME_SCHEMA,
        "role": ROLE,
        "profile": PROFILE,
        "algorithm": _algorithm_identity(),
        "execution_scope": base.EXECUTION_SCOPE,
        "physical_validation": "not_established",
    }


def _native_result(source):
    request = source["request"]
    expected = reference.reference(request)
    model = deepcopy(expected["model"])
    model.update({
        "state_order": contract.STATE_ORDER,
        "input_order": contract.INPUT_ORDER,
        "sensor_order": contract.SENSOR_ORDER,
        "symbolic": {
            "equations": [
                "d(core_temperature)/dt = (heat_power - g_cs*(core_temperature - shell_temperature))/C_core",
                "d(shell_temperature)/dt = (g_cs*(core_temperature - shell_temperature) - g_sa*(shell_temperature - ambient_temperature))/C_shell",
            ],
            "latex": [
                "\\dot{T}_{core}=(P-g_{cs}(T_{core}-T_{shell}))/C_{core}",
                "\\dot{T}_{shell}=(g_{cs}(T_{core}-T_{shell})-g_{sa}(T_{shell}-T_a))/C_{shell}",
            ],
            "native_state_order": contract.STATE_ORDER,
            "state_permutation": [1, 2],
            "rendering": SYMBOLIC_RENDERING,
        },
    })
    selection = deepcopy(expected["selection"])
    solver = {
        "name": SOLVER_NAME,
        "termination_status": "INFEASIBLE" if selection["status"] == "infeasible" else "OPTIMAL",
        "primal_status": "NO_SOLUTION" if selection["status"] == "infeasible" else "FEASIBLE_POINT",
        "objective_value": selection["objective_nats"],
        "objective_bound": selection["objective_nats"],
        "relative_gap": None if selection["status"] == "infeasible" else 0.0,
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "mip_feasibility_tolerance": 1e-9,
        "mip_relative_gap_tolerance": 0.0,
    }
    selection["solver"] = solver
    result = {
        "schema": contract.RESULT_SCHEMA,
        "request": deepcopy(request),
        "claim_scope": contract.CLAIM_SCOPE,
        "model": model,
        "observer": expected["observer"],
        "selection": selection,
    }
    contract.validate_result(request, result)
    return result


class ThermalWorkflow(base.ReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    SOURCE_SCHEMA = SOURCE_SCHEMA
    result_schema = RESULT_SCHEMA
    verify_schema = VERIFY_SCHEMA
    operation = OPERATION
    role = ROLE
    label = "thermal"
    ROLES = ROLES
    MAX_BYTES = MAX_BYTES
    AUTHORITY = AUTHORITY

    def _source(self, raw):
        return contract.validate_source(raw)

    def _native_data(self, source):
        return _native_result(source)

    def _runtime_identity(self):
        return runtime_identity()

    def _configuration(self, source):
        return deepcopy(source["configuration"])

    def _check_data(self, result, source, expected=None):
        # The contract tolerates platform rounding, so the retained numbers are
        # checked against the independent reference rather than bit for bit;
        # provenance, however, must be this reference's own.
        contract.validate_result(source["request"], result["data"])
        if (result["data"]["model"]["symbolic"]["rendering"] != SYMBOLIC_RENDERING or
                result["data"]["selection"]["solver"]["name"] != SOLVER_NAME):
            raise ValueError("Thermal native result provenance differs from the Python reference")


def _verification(bundle, reproduced):
    return ThermalWorkflow()._verification(bundle, reproduced)
