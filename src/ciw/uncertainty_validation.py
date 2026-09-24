"""Provider-free finite-sample validation of declared estimate uncertainty.

A retained source holds ordered samples of a declared reference value, an
estimate and the estimate's covariance, optionally with innovations and their
covariances.  The operation reports the normalized estimation error squared,
the normalized innovation squared, per-component interval coverage and bias,
each against a two-sided band at the declared confidence, and names which
statistics fall outside their bands.  It separates a covariance that is too
small from one that is too large instead of merging both into one verdict.

The reference values are declared, not measured: a synthetic fixture or a
held-out reference the operator supplies.  Their own uncertainty is not
modelled.  Statistical authority requires declared independence between
samples; with unknown dependence the same numbers are reported as diagnostics
only.  Nothing here validates a physical model or admits state.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from . import consistency_math as consistency
from . import reference_workflow as base
from .core.covariance import _validate_matrix
from .telemetry import canonical, _keys

KIND = "uncertainty-validation"
SCHEMA = "ciw.uncertainty-validation-session.v1"
SOURCE_SCHEMA = "ciw.uncertainty-validation-source.v1"
DATA_SCHEMA = "ciw.uncertainty-validation-workbench-data.v1"
RESULT_SCHEMA = "ciw.uncertainty-validation-workbench-result.v1"
VERIFY_SCHEMA = "ciw.uncertainty-validation-verification.v1"
OPERATION = "ciw.uncertainty-validation.v1"
PROFILE = "ciw.uncertainty-validation.python-reference.v1"
ROLE = "consistency"
ROLES = set()
SOURCE_LIMIT = 2 * 1024 * 1024
MAX_BYTES = 8 * SOURCE_LIMIT
MAX_SAMPLES = 4096
MAX_DIMENSION = 16
MAX_CONDITION = 1e12
CONFIGURATION = {
    "profile": "finite_sample_consistency",
    "activation": "read_only",
    "reference_values": "declared_not_measured",
    "physical_validation": "not_established",
    "state_admission": "not_performed",
}
AUTHORITY = {
    "physical_validation": "not_established",
    "state_admission": "not_performed",
    "reference_uncertainty": "not_modelled_reference_treated_as_exact",
}
CLAIM_SCOPE = "finite_sample_consistency_of_declared_covariances_against_declared_reference_values"
TRUTH_ORIGINS = {"synthetic_fixture", "declared_reference"}
DEPENDENCE = {"declared_independent", "unknown"}
STATISTICAL_SCOPE = {"declared_independent": "finite_sample_consistency_under_declared_independence",
                     "unknown": "diagnostic_only_cross_sample_dependence_unknown"}
REL_TOL = base.REL_TOL
ABS_TOL = base.ABS_TOL


_text = base._text


@lru_cache(maxsize=1)
def _algorithm_identity():
    return base.algorithm_identity(PROFILE, [Path(consistency.__file__), Path(base.__file__), Path(__file__)],
                                   numpy_version=np.__version__)


def runtime_identity():
    return {
        "schema": base.RUNTIME_SCHEMA,
        "role": ROLE,
        "profile": PROFILE,
        "algorithm": _algorithm_identity(),
        "execution_scope": base.EXECUTION_SCOPE,
        "physical_validation": "not_established",
        "state_admission": "not_performed",
    }


def _names(value, name, limit=MAX_DIMENSION):
    if type(value) is not list or not 1 <= len(value) <= limit or len(set(value)) != len(value):
        raise ValueError(f"{name} must list between 1 and {limit} distinct names")
    for item in value:
        _text(item, 128)
    return list(value)


def _vector(value, size, name):
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must hold exactly {size} ordered numbers")
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(f"{name} must hold finite numbers")
    return [float(item) for item in value]


def _definite_covariance(value, size, name):
    """Admit a covariance the workbench accepts that is also invertible for a normalized square."""
    _validate_matrix(value, size)
    matrix = np.asarray(value, dtype=np.float64)
    if np.any(np.diag(matrix) <= 0):
        raise ValueError(f"{name} must have strictly positive variances for a normalized square")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 0 or eigenvalues.max() / eigenvalues.min() > MAX_CONDITION:
        raise ValueError(f"{name} must be positive definite with a bounded condition number")
    return [[float(item) for item in row] for row in value]


def _samples(value, name, vector_keys, size):
    if type(value) is not list or not 2 <= len(value) <= MAX_SAMPLES:
        raise ValueError(f"{name} must hold between 2 and {MAX_SAMPLES} entries")
    retained = []
    previous = None
    for index, entry in enumerate(value):
        _keys(entry, {"time", "covariance", *vector_keys})
        if type(entry["time"]) not in (int, float) or not math.isfinite(entry["time"]):
            raise ValueError(f"{name}[{index}].time must be a finite number")
        if previous is not None and entry["time"] <= previous:
            raise ValueError(f"{name} times must strictly increase")
        previous = entry["time"]
        item = {"time": float(entry["time"]), "covariance": _definite_covariance(entry["covariance"], size,
                                                                                    f"{name}[{index}].covariance")}
        for key in vector_keys:
            item[key] = _vector(entry[key], size, f"{name}[{index}].{key}")
        retained.append(item)
    return retained


def validate_source(raw):
    """Validate exact source bytes; no statistic is computed here."""
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Uncertainty validation source requires bounded exact JSON bytes")
    source = base.parse_json(raw, "Uncertainty validation source")
    _keys(source, {"schema", "experiment_id", "configuration", "request"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(CONFIGURATION):
        raise ValueError("Unsupported uncertainty validation source or authority policy")
    _text(source["experiment_id"], 128)
    request = source["request"]
    _keys(request, {"confidence", "state_names", "state_units", "frame", "time_basis", "reference_origin",
                    "cross_sample_dependence", "samples"},
          {"innovations", "measurement_names", "measurement_units"})
    confidence = request["confidence"]
    if type(confidence) is not float or not 0.5 <= confidence <= 0.999:
        raise ValueError("confidence must be a probability between 0.5 and 0.999")
    names = _names(request["state_names"], "state_names")
    if type(request["state_units"]) is not list or len(request["state_units"]) != len(names):
        raise ValueError("state_units must match state_names")
    for unit in request["state_units"]:
        _text(unit, 64)
    _text(request["frame"], 128)
    _text(request["time_basis"], 128)
    if request["reference_origin"] not in TRUTH_ORIGINS:
        raise ValueError("reference_origin must declare a synthetic fixture or a declared reference")
    if request["cross_sample_dependence"] not in DEPENDENCE:
        raise ValueError("cross_sample_dependence must be declared_independent or unknown")
    _samples(request["samples"], "samples", ("reference", "estimate"), len(names))
    has_innovations = "innovations" in request
    if has_innovations != ("measurement_names" in request) or has_innovations != ("measurement_units" in request):
        raise ValueError("innovations require measurement_names and measurement_units together")
    if has_innovations:
        measurements = _names(request["measurement_names"], "measurement_names")
        if type(request["measurement_units"]) is not list or len(request["measurement_units"]) != len(measurements):
            raise ValueError("measurement_units must match measurement_names")
        for unit in request["measurement_units"]:
            _text(unit, 64)
        _samples(request["innovations"], "innovations", ("innovation",), len(measurements))
    if len(canonical(source)) > SOURCE_LIMIT:
        raise ValueError("Uncertainty validation source exceeds the byte budget")
    return deepcopy(source)


def _native_data(source):
    request = source["request"]
    confidence = request["confidence"]
    names, units = request["state_names"], request["state_units"]
    errors = [[estimate - reference for estimate, reference in zip(sample["estimate"], sample["reference"])]
              for sample in request["samples"]]
    covariances = [sample["covariance"] for sample in request["samples"]]
    nees = consistency.mean_square_status(consistency.normalized_squares(errors, covariances), len(names), confidence)
    coverage = consistency.coverage_status(errors, covariances, confidence)
    bias = consistency.bias_status(errors, covariances, confidence)
    for component, name, unit in zip(coverage["components"], names, units):
        component.update(name=name, unit=unit)
    for component, name, unit in zip(bias["components"], names, units):
        component.update(name=name, unit=unit)
    nis = None
    if "innovations" in request:
        innovations = [entry["innovation"] for entry in request["innovations"]]
        nis = consistency.mean_square_status(
            consistency.normalized_squares(innovations, [entry["covariance"] for entry in request["innovations"]]),
            len(request["measurement_names"]), confidence)
    failures = []
    if nees["status"] != consistency.CONSISTENT:
        failures.append("nees:" + nees["status"])
    if nis is not None and nis["status"] != consistency.CONSISTENT:
        failures.append("nis:" + nis["status"])
    failures.extend("coverage:" + item["name"] + ":" + item["status"]
                    for item in coverage["components"] if item["status"] != consistency.CONSISTENT)
    failures.extend("bias:" + item["name"] + ":" + item["status"]
                    for item in bias["components"] if item["status"] != consistency.UNBIASED)
    dependence = request["cross_sample_dependence"]
    return {
        "schema": DATA_SCHEMA,
        "operation_id": OPERATION,
        "confidence": confidence,
        "sample_count": len(request["samples"]),
        "state_dimension": len(names),
        "state_names": list(names),
        "state_units": list(units),
        "frame": request["frame"],
        "time_basis": request["time_basis"],
        "reference_origin": request["reference_origin"],
        "cross_sample_dependence": dependence,
        "statistical_scope": STATISTICAL_SCOPE[dependence],
        "nees": nees,
        "nis": nis,
        "coverage": coverage,
        "bias": bias,
        "verdict": {"status": "consistent" if not failures else "inconsistent", "failures": failures,
                    "authority": "statistical" if dependence == "declared_independent" else "diagnostic_only"},
        "claim_scope": CLAIM_SCOPE,
        "authority": deepcopy(AUTHORITY),
    }


class UncertaintyValidationWorkflow(base.ReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    SOURCE_SCHEMA = SOURCE_SCHEMA
    result_schema = RESULT_SCHEMA
    verify_schema = VERIFY_SCHEMA
    operation = OPERATION
    role = ROLE
    label = "uncertainty validation"
    ROLES = ROLES
    MAX_BYTES = MAX_BYTES
    AUTHORITY = AUTHORITY

    def _source(self, raw):
        return validate_source(raw)

    def _native_data(self, source):
        return _native_data(source)

    def _runtime_identity(self):
        return runtime_identity()

    def _configuration(self, source):
        return deepcopy(CONFIGURATION)

    def _check_data(self, result, source, expected=None):
        # Statuses, names and structure must match exactly; the statistics are
        # compared with a binary64 tolerance because linear solves may round
        # differently across platforms without changing any conclusion.
        base.close_data(result["data"], _native_data(source) if expected is None else expected)


def _verification(bundle, reproduced):
    return UncertaintyValidationWorkflow()._verification(bundle, reproduced)
