"""Provider-free thermal observer workflow with native CIW identities.

The Python reference is the first executable operation for the thermal contract.
It is deliberately separate from the Julia worker: the retained result records
which implementation produced them, while the contract can later validate a
Julia occurrence against the same independent reference.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from . import thermal_contract as contract
from . import thermal_reference as reference
from .core.canonical import exact_keys
from .pipelines import runner as _runner
from .pipelines.runner import PipelineRunner, RecordProfile, seal_step

KIND = "thermal-observer"
SCHEMA = "ciw.thermal-observer-session.v1"
OPERATION = "ciw.thermal-observer.v1"
SOURCE_SCHEMA = contract.SOURCE_SCHEMA
RESULT_SCHEMA = "ciw.thermal-observer-workbench-result.v1"
VERIFY_SCHEMA = "ciw.thermal-observer-verification.v1"
ROLE = "thermal"
ROLES = set()
MAX_BYTES = contract.RESULT_LIMIT
AUTHORITY = {
    "physical_validation": "not_established",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}


def _algorithm_identity():
    files = [Path(contract.__file__), Path(reference.__file__), Path(__file__), Path(_runner.__file__)]
    content = b"\0".join(
        path.name.encode("utf-8") + b"\0" + path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        for path in files
    )
    return {
        "profile": "ciw.thermal-observer.python-reference.v1",
        "code_sha256": sha256(content).hexdigest(),
        "source_normalization": "utf8_lf",
        "numpy_version": np.__version__,
    }


def runtime_identity():
    return {
        "schema": "ciw.python-reference-runtime.v1",
        "role": ROLE,
        "profile": "ciw.thermal-observer.python-reference.v1",
        "algorithm": _algorithm_identity(),
        "execution_scope": "independent_python_reference_only",
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
            "rendering": "python-reference",
        },
    })
    selection = deepcopy(expected["selection"])
    solver = {
        "name": "python-reference-enumeration",
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


class ThermalWorkflow(PipelineRunner):
    """The Python thermal reference sealed, reproduced and replayed by the shared runner."""

    MAX_BYTES = MAX_BYTES
    LABEL = "Thermal"
    PROFILE = RecordProfile(RESULT_SCHEMA, VERIFY_SCHEMA, "same_python_reference_fresh_occurrence_reproduction", AUTHORITY)

    def __init__(self):
        super().__init__(KIND, {"role": ROLE})
        self.ROLES = frozenset(ROLES)

    def parse_source(self, raw):
        return contract.validate_source(raw)

    def step_request(self, source):
        return source["request"]

    def check_data(self, source, data):
        contract.validate_result(source["request"], data)

    def _adapters(self, repositories, expected=None):
        if not isinstance(repositories, dict) or repositories:
            raise ValueError("The provider-free thermal reference accepts no repository bindings")
        runtime = runtime_identity()
        if expected is not None:
            retained = expected.get(ROLE, expected) if isinstance(expected, dict) else expected
            if runtime != retained:
                raise ValueError("Thermal reference runtime identity differs from the retained execution")
        return None, runtime, None

    def _step(self, source, evidence_id, bound):
        data = _native_result(source)
        self.check_data(source, data)
        return seal_step(ROLE, OPERATION, self.step_request(source), [evidence_id], data, profile=self.PROFILE)

    def _check_runtimes(self, runtimes):
        exact_keys(runtimes, {ROLE})
        runtime = runtimes[ROLE]
        exact_keys(runtime, {"schema", "role", "profile", "algorithm", "execution_scope", "physical_validation"})
        if (runtime["schema"] != "ciw.python-reference-runtime.v1" or runtime["role"] != ROLE or
                runtime["profile"] != "ciw.thermal-observer.python-reference.v1" or
                runtime["execution_scope"] != "independent_python_reference_only" or
                runtime["physical_validation"] != "not_established"):
            raise ValueError("Unapproved thermal reference runtime")
        algorithm = runtime["algorithm"]
        exact_keys(algorithm, {"profile", "code_sha256", "source_normalization", "numpy_version"})
        if (algorithm["profile"] != "ciw.thermal-observer.python-reference.v1" or
                algorithm["source_normalization"] != "utf8_lf" or
                not re.fullmatch(r"[a-f0-9]{64}", algorithm["code_sha256"]) or
                not isinstance(algorithm["numpy_version"], str)):
            raise ValueError("Malformed thermal algorithm identity")
