"""Offline contract checks for retained affine-invariant covariance geometry.

These checks validate bounded declarations, content identities and consistency
of reported numerical evidence. They neither import a provider nor solve an
eigenproblem, authenticate execution, establish SPD eigenpairs, or confer state
estimation, calibration or physical authority. Fresh reproduction is separate.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .core.canonical import canonical, digest

REQUEST_SCHEMA = "covariance-geometry-request-v1"
RESULT_SCHEMA = "covariance-geometry-result-v1"
INPUT_LIMIT, RESULT_LIMIT = 65536, 524288
EPS = 2.220446049250313e-16
AUTHORITY = {"state_estimation": "not_performed", "calibration": "not_established",
             "physical_accuracy": "not_established", "admission": "not_performed"}
SETTINGS = {"symmetry_tolerance": (1e-14, 1e-8), "eigenvalue_floor": (1e-12, 1e-2),
            "condition_limit": (1.0, 1e6), "relative_tolerance": (1e-12, 1e-6),
            "absolute_tolerance": (1e-12, 1e-6)}
SPECTRAL_KEYS = {"eigenvalues", "minimum_eigenvalue", "maximum_eigenvalue", "eigenvalue_floor",
                 "eigenvalue_margin", "condition_number", "log_determinant", "matrix_frobenius",
                 "decomposition_residual", "decomposition_budget"}


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Covariance contract requires exactly the declared fields")


def _bounded_json(value, limit):
    try:
        encoded = canonical(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Require finite bounded covariance JSON") from exc
    if len(encoded) > limit:
        raise ValueError("Covariance JSON exceeds the profile byte budget")
    return encoded


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("Covariance quantities must be finite JSON numbers, not Booleans")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Covariance quantity exceeds binary64") from exc
    if not math.isfinite(result) or (type(value) is int and value != result):
        raise ValueError("Covariance quantity must be exactly representable as finite binary64")
    return result


def _nonnegative(value):
    number = _number(value)
    if number < 0:
        raise ValueError("Covariance distance, residual and norm must be nonnegative")
    return number


def _text(value):
    if type(value) is not str or not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError("Require bounded nonempty covariance metadata without control characters")


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Covariance identity or fixed policy differs")


def _derived(actual, expected):
    # Budgets derive from retained binary64 scalars by the same basic arithmetic.
    if _number(actual) != expected:
        raise ValueError("Covariance reported value differs from its declared derivation")


def _close(actual, expected, *, scale=None, residual=0.0):
    actual = _number(actual)
    roundoff_scale = max(abs(actual), abs(expected), 0.0 if scale is None else abs(scale))
    tolerance = 64 * EPS * roundoff_scale + 8 * math.ulp(expected) + residual
    if abs(actual - expected) > tolerance:
        raise ValueError("Covariance retained numerical evidence is internally inconsistent")


def _norm(matrix):
    return math.hypot(*(value for row in matrix for value in row))


def _matrix(value, size, *, input_matrix=False):
    if type(value) is not list or len(value) != size:
        raise ValueError("Covariance matrix row count differs from its coordinates")
    rows = []
    for row in value:
        if type(row) is not list or len(row) != size:
            raise ValueError("Covariance matrix must be square")
        rows.append([_number(number) for number in row])
    bound = 1e6 if input_matrix else 1e8
    if any(abs(number) > bound for row in rows for number in row):
        raise ValueError("Covariance matrix entries exceed the bounded profile")
    if any(rows[i][j] != rows[j][i] for i in range(size) for j in range(size)):
        raise ValueError("Declared covariance matrices require exact symmetry")
    return rows


def validate_request(request):
    """Check the bounded declaration; numerical SPD admissibility is provider work."""
    _bounded_json(request, INPUT_LIMIT)
    _keys(request, {"schema", "experiment_id", "metric", "coordinates", "frame", "covariance_a",
                    "covariance_b", "parameters", "settings"})
    if request["schema"] != REQUEST_SCHEMA or request["metric"] != "affine_invariant":
        raise ValueError("Unsupported covariance request schema or metric")
    _text(request["experiment_id"])
    _text(request["frame"])
    coordinates = request["coordinates"]
    if type(coordinates) is not list or not 1 <= len(coordinates) <= 8:
        raise ValueError("Declare 1..8 ordered covariance coordinates")
    names = []
    for coordinate in coordinates:
        _keys(coordinate, {"name", "unit"})
        _text(coordinate["name"])
        _text(coordinate["unit"])
        names.append(coordinate["name"])
    if len(set(names)) != len(names):
        raise ValueError("Covariance coordinate names must be distinct")
    _keys(request["settings"], SETTINGS)
    for name, (lower, upper) in SETTINGS.items():
        if not lower <= _number(request["settings"][name]) <= upper:
            raise ValueError("Covariance setting exceeds its declared profile")
    parameters = request["parameters"]
    if type(parameters) is not list or not 2 <= len(parameters) <= 33:
        raise ValueError("Require 2..33 geodesic parameters")
    values = [_number(value) for value in parameters]
    if values[0] != 0 or values[-1] != 1 or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("Geodesic parameters must strictly increase from zero to one")
    for name in ("covariance_a", "covariance_b"):
        _matrix(request[name], len(coordinates), input_matrix=True)
    return deepcopy(request)


def _spectrum(record, size, settings, *, matrix=None, relative=False):
    _keys(record, SPECTRAL_KEYS)
    values = record["eigenvalues"]
    if type(values) is not list or len(values) != size:
        raise ValueError("Spectral evidence must cover every declared coordinate")
    values = [_number(value) for value in values]
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("Spectral evidence requires ordered eigenvalues")
    floor = 8 * size * EPS * max(abs(values[0]), abs(values[-1])) if relative else settings["eigenvalue_floor"]
    if values[0] <= floor:
        raise ValueError("Reported spectrum violates its strict positive eigenvalue margin")
    _derived(record["minimum_eigenvalue"], values[0])
    _derived(record["maximum_eigenvalue"], values[-1])
    _derived(record["eigenvalue_floor"], floor)
    _derived(record["eigenvalue_margin"], values[0] - floor)
    condition = values[-1] / values[0]
    _derived(record["condition_number"], condition)
    if condition > settings["condition_limit"]:
        raise ValueError("Reported spectrum exceeds the declared conditioning limit")
    norm = _nonnegative(record["matrix_frobenius"])
    if not 0 < norm <= (1e20 if relative else 1e8):
        raise ValueError("Spectral matrix scale exceeds the bounded profile")
    residual = _nonnegative(record["decomposition_residual"])
    budget = settings["absolute_tolerance"] + settings["relative_tolerance"] * norm
    _derived(record["decomposition_budget"], budget)
    if residual > budget:
        raise ValueError("Reported decomposition residual exceeds its declared budget")
    logs = [math.log(value) for value in values]
    _close(record["log_determinant"], math.fsum(logs), scale=math.fsum(abs(value) for value in logs))
    _close(norm, math.hypot(*values), residual=residual)
    if matrix is not None:
        _close(norm, _norm(matrix))
        _close(math.fsum(matrix[i][i] for i in range(size)), math.fsum(values),
               scale=norm, residual=math.sqrt(size) * residual)
    return record


def _check_invariant(record, name, scale, settings, *, residual=None):
    _keys(record, {"name", "residual", "budget", "passed"})
    if record["name"] != name or record["passed"] is not True:
        raise ValueError("Covariance invariants are missing, reordered or unsuccessful")
    budget = settings["absolute_tolerance"] + settings["relative_tolerance"] * abs(scale)
    _derived(record["budget"], budget)
    observed = _nonnegative(record["residual"])
    if observed > budget:
        raise ValueError("Covariance invariant exceeds its declared budget")
    if residual is not None:
        _close(observed, residual)


def validate_result(request, data):
    """Validate retained scope and diagnostics without executing a numerical solver."""
    request = validate_request(request)
    _bounded_json(data, RESULT_LIMIT)
    _keys(data, {"schema", "claim_scope", "operation", "request", "request_digest", "metric", "dimension",
                 "distance", "distance_unit", "samples", "input_evidence", "evidence", "runtime", "authority", "artifact_digest"})
    if (data["schema"] != RESULT_SCHEMA or data["claim_scope"] != "declared-spd-affine-invariant-geometry" or
            data["operation"] != "affine-invariant-geodesic-v1" or data["metric"] != "affine_invariant" or
            data["distance_unit"] != "1"):
        raise ValueError("Unsupported covariance mathematical claim or operation")
    _same(data["request"], request)
    if data["request_digest"] != digest(request) or data["artifact_digest"] != digest({k:v for k,v in data.items() if k != "artifact_digest"}):
        raise ValueError("Covariance request or artifact digest mismatch")
    _same(data["authority"], AUTHORITY)
    _same(data["runtime"], {"numpy_version": "2.4.3", "arithmetic": "float64", "eigensolver": "numpy.linalg.eigh"})
    size, settings = len(request["coordinates"]), request["settings"]
    if type(data["dimension"]) is not int or data["dimension"] != size:
        raise ValueError("Covariance result dimension differs from its declaration")
    _keys(data["input_evidence"], {"covariance_a", "covariance_b", "relative_covariance"})
    spectra = data["input_evidence"]
    for name in ("covariance_a", "covariance_b"):
        _spectrum(spectra[name], size, settings, matrix=_matrix(request[name], size))
    relative = _spectrum(spectra["relative_covariance"], size, settings, relative=True)
    distance = _nonnegative(data["distance"])
    _close(distance, math.hypot(*(math.log(v) for v in relative["eigenvalues"])))
    samples = data["samples"]
    if type(samples) is not list or len(samples) != len(request["parameters"]):
        raise ValueError("Covariance samples must cover exactly the declared parameters")
    expected_checks = [("distance_symmetry", distance, None)]
    expected_stages = ["relative_covariance", "reverse_distance"]
    symmetric_norms = {"relative_covariance": relative["matrix_frobenius"]}
    for index, (parameter, sample) in enumerate(zip(request["parameters"], samples)):
        _keys(sample, {"parameter", "covariance", "spectrum", "distance_from_a", "distance_to_b"})
        _number(sample["parameter"])
        _same(sample["parameter"], parameter)
        matrix = _matrix(sample["covariance"], size)
        spectral = _spectrum(sample["spectrum"], size, settings, matrix=matrix)
        from_a, to_b = _nonnegative(sample["distance_from_a"]), _nonnegative(sample["distance_to_b"])
        logdet = (1 - parameter) * spectra["covariance_a"]["log_determinant"] + parameter * spectra["covariance_b"]["log_determinant"]
        expected_checks.extend([
            ("constant_speed_from_a:" + str(index), distance, abs(from_a - parameter * distance)),
            ("constant_speed_to_b:" + str(index), distance, abs(to_b - (1 - parameter) * distance)),
            ("log_determinant_affinity:" + str(index), logdet, abs(spectral["log_determinant"] - logdet))])
        if index in (0, len(samples) - 1):
            name = "covariance_a" if index == 0 else "covariance_b"
            endpoint = request[name]
            residual = _norm([[matrix[i][j] - endpoint[i][j] for j in range(size)] for i in range(size)])
            expected_checks.append(("endpoint:" + str(index), spectra[name]["matrix_frobenius"], residual))
        stage = "sample:" + str(index)
        symmetric_norms[stage] = spectral["matrix_frobenius"]
        expected_stages.extend([stage, "distance_from_a:" + str(index), "distance_to_b:" + str(index)])
    evidence = data["evidence"]
    _keys(evidence, {"invariants", "roundoff_symmetry", "input_projection", "regularization", "intermediate_symmetry"})
    if (evidence["input_projection"] != "not_performed" or evidence["regularization"] != "not_performed" or
            evidence["intermediate_symmetry"] != "recorded_transpose_averaging"):
        raise ValueError("Covariance records cannot silently repair or reinterpret input")
    checks = evidence["invariants"]
    if type(checks) is not list or len(checks) != len(expected_checks):
        raise ValueError("Require every declared covariance invariant exactly once")
    for record, (name, scale, residual) in zip(checks, expected_checks):
        _check_invariant(record, name, scale, settings, residual=residual)
    records = evidence["roundoff_symmetry"]
    if type(records) is not list or len(records) != len(expected_stages):
        raise ValueError("Require every declared intermediate roundoff record")
    for record, stage in zip(records, expected_stages):
        _keys(record, {"stage", "matrix_frobenius", "max_asymmetry", "allowed_asymmetry", "averaging_adjustment_frobenius"})
        if record["stage"] != stage:
            raise ValueError("Roundoff evidence stages are missing or reordered")
        norm = _nonnegative(record["matrix_frobenius"])
        if not 0 < norm <= 1e20:
            raise ValueError("Roundoff matrix scale exceeds the bounded profile")
        allowed = settings["symmetry_tolerance"] * max(1.0, norm)
        _derived(record["allowed_asymmetry"], allowed)
        asymmetry = _nonnegative(record["max_asymmetry"])
        adjustment = _nonnegative(record["averaging_adjustment_frobenius"])
        if asymmetry > allowed or adjustment > size * asymmetry or adjustment < asymmetry / 2:
            raise ValueError("Roundoff correction exceeds the declared asymmetry evidence")
        if stage in symmetric_norms:
            _close(norm, symmetric_norms[stage], residual=adjustment)
    return deepcopy(data)
