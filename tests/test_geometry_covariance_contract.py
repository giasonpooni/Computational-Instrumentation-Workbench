"""Offline declarations and resealed-tamper checks; no native provider needed."""
from copy import deepcopy
import builtins
import json
import subprocess

import pytest

from ciw.geometry_covariance_contract import validate_request, validate_result
from ciw.core.canonical import digest


def result():
    return deepcopy(REFERENCE)


def request():
    return deepcopy(REFERENCE["request"])


def reseal(value):
    value["artifact_digest"] = digest({k:v for k,v in value.items() if k != "artifact_digest"})
    return value


def test_actual_retained_reference_is_valid_without_provider_or_solver(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("covariance_geometry"):
            pytest.fail("Offline contract imported a native provider")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline contract executed a process or eigensolver")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", forbidden)
    import numpy as np
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    value, retained = request(), result()
    assert validate_request(value) == value
    accepted = validate_result(value, retained)
    assert accepted == retained
    assert accepted is not retained and accepted["request"] is not retained["request"]
    assert retained == REFERENCE


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("metric", "log_euclidean"), ("experiment_id", ""),
    ("frame", "invalid\nframe"), ("frame", "x" * 129), ("parameters", [0, True, 1]),
    ("parameters", [0, 0.5, 0.5, 1]), ("parameters", [0.1, 1]),
    ("parameters", [0, 0.5]), ("parameters", [i / 33 for i in range(34)]),
    ("coordinates", []), ("coordinates", [{"name":"x", "unit":"1"}] * 2),
    ("coordinates", [{"name":"x", "unit":""}, {"name":"y", "unit":"1"}]),
    ("covariance_a", [[1, 1e-16], [0, 4]]), ("covariance_a", [[True,0],[0,4]]),
    ("covariance_a", [[float("inf"),0],[0,4]]), ("covariance_a", [[1000001,0],[0,4]]),
    ("covariance_a", [[2**53+1,0],[0,4]]), ("covariance_a", [[10**400,0],[0,4]]),
    ("covariance_b", [[1, 0]]),
])
def test_bounded_request_refusal(field, value):
    declared = request()
    declared[field] = value
    with pytest.raises(ValueError):
        validate_request(declared)


@pytest.mark.parametrize("name,value", [
    ("eigenvalue_floor", 0), ("eigenvalue_floor", 1), ("condition_limit", 1e7),
    ("condition_limit", True), ("symmetry_tolerance", 1e-6),
    ("relative_tolerance", 1e-10 - 1e-9), ("absolute_tolerance", float("nan")),
])
def test_request_settings_require_fixed_shape_and_bounds(name, value):
    declared = request()
    declared["settings"][name] = value
    with pytest.raises(ValueError):
        validate_request(declared)


@pytest.mark.parametrize("location", [(), ("settings",), ("coordinates", 0)])
def test_unknown_request_fields_are_refused(location):
    declared = request()
    target = declared
    for key in location:
        target = target[key]
    target["unknown"] = "unsupported"
    with pytest.raises(ValueError):
        validate_request(declared)


@pytest.mark.parametrize("field,value", [
    ("claim_scope", "physically_calibrated_covariance"), ("metric", "euclidean"),
    ("operation", "custom"), ("dimension", True), ("dimension", 3),
    ("distance_unit", "m"), ("distance", -1), ("distance", True),
    ("authority", {"state_estimation":"approved"}),
    ("runtime", {"numpy_version":"2.4.2", "arithmetic":"float64", "eigensolver":"numpy.linalg.eigh"}),
])
def test_resealed_scope_runtime_or_dimension_change_is_refused(field, value):
    retained = result()
    retained[field] = value
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_request_binding_preserves_numeric_json_encoding():
    retained = result()
    retained["request"]["covariance_a"][0][0] = 1.0
    retained["request_digest"] = digest(retained["request"])
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


@pytest.mark.parametrize("mutation", [
    lambda values: values.pop(),
    lambda values: values.append(deepcopy(values[0])),
    lambda values: values.reverse(),
    lambda values: values.clear(),
])
@pytest.mark.parametrize("field", ["invariants", "roundoff_symmetry"])
def test_complete_ordered_diagnostic_coverage_is_required(field, mutation):
    retained = result()
    mutation(retained["evidence"][field])
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


@pytest.mark.parametrize("field,value", [
    ("eigenvalues", [1, 1]), ("minimum_eigenvalue", 0), ("maximum_eigenvalue", 17),
    ("eigenvalue_floor", 0), ("eigenvalue_margin", 1), ("condition_number", 1),
    ("log_determinant", 0), ("decomposition_residual", -1),
    ("decomposition_budget", 1000), ("matrix_frobenius", 1000),
])
def test_resealed_spectral_evidence_is_checked(field, value):
    retained = result()
    retained["input_evidence"]["covariance_b"][field] = value
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_relative_spectral_floor_is_derived_from_binary64_margin():
    retained = result()
    spectrum = retained["input_evidence"]["relative_covariance"]
    spectrum["eigenvalue_floor"] = retained["request"]["settings"]["eigenvalue_floor"]
    spectrum["eigenvalue_margin"] = spectrum["minimum_eigenvalue"] - spectrum["eigenvalue_floor"]
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


@pytest.mark.parametrize("section", ["invariants", "roundoff_symmetry"])
def test_reported_budget_cannot_be_inflated_even_with_passing_residual(section):
    retained = result()
    diagnostic = retained["evidence"][section][0]
    field = "budget" if section == "invariants" else "allowed_asymmetry"
    diagnostic[field] *= 2
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_zeroed_invariant_residual_cannot_hide_changed_sample():
    retained = result()
    retained["samples"][1]["distance_from_a"] += 0.1
    for invariant in retained["evidence"]["invariants"]:
        invariant["residual"] = 0
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_spectral_norm_is_checked_against_retained_matrix():
    retained = result()
    retained["samples"][1]["covariance"][0][0] += 0.1
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


@pytest.mark.parametrize("location", [
    (), ("input_evidence", "relative_covariance"), ("samples", 1),
    ("samples", 1, "spectrum"), ("evidence",), ("evidence", "roundoff_symmetry", 0),
])
def test_unknown_result_fields_are_refused(location):
    retained = result()
    target = retained
    for key in location:
        target = target[key]
    target["unknown"] = "unsupported"
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_missing_recorded_matrix_scale_is_refused():
    retained = result()
    retained["evidence"]["roundoff_symmetry"][1].pop("matrix_frobenius")
    with pytest.raises(ValueError):
        validate_result(request(), reseal(retained))


def test_byte_budgets_apply_before_retaining_oversized_records():
    declared = request()
    declared["unexpected"] = "x" * 65536
    with pytest.raises(ValueError):
        validate_request(declared)
    retained = result()
    retained["unexpected"] = "x" * 524288
    with pytest.raises(ValueError):
        validate_result(request(), retained)


# Actual provider output, captured from CGGT commit 720677ff8b54b0052ce962ed35111defcd07d9e2
# under NumPy 2.4.3 on Windows/Python 3.12. Analytical diagonal reference, no runtime
# provider import or fallback; the native integration gate exercises fresh runs.
REFERENCE = json.loads(r'''
{
  "artifact_digest": "sha256:a566ea3f8ba32df75449d287a40b99de91d1ca08d25221a8a27499d453ed3b34",
  "authority": {
    "admission": "not_performed",
    "calibration": "not_established",
    "physical_accuracy": "not_established",
    "state_estimation": "not_performed"
  },
  "claim_scope": "declared-spd-affine-invariant-geometry",
  "dimension": 2,
  "distance": 1.9605162869370942,
  "distance_unit": "1",
  "evidence": {
    "input_projection": "not_performed",
    "intermediate_symmetry": "recorded_transpose_averaging",
    "invariants": [
      {
        "budget": 2.0605162869370946e-09,
        "name": "distance_symmetry",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_from_a:0",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_to_b:0",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 1.4862943611198907e-09,
        "name": "log_determinant_affinity:0",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 4.223105625617661e-09,
        "name": "endpoint:0",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_from_a:1",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_to_b:1",
        "passed": true,
        "residual": 3.3306690738754696e-16
      },
      {
        "budget": 2.8725887222397813e-09,
        "name": "log_determinant_affinity:1",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_from_a:2",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 2.0605162869370946e-09,
        "name": "constant_speed_to_b:2",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 4.2588830833596716e-09,
        "name": "log_determinant_affinity:2",
        "passed": true,
        "residual": 0.0
      },
      {
        "budget": 1.6592422502470644e-08,
        "name": "endpoint:2",
        "passed": true,
        "residual": 0.0
      }
    ],
    "regularization": "not_performed",
    "roundoff_symmetry": [
      {
        "allowed_asymmetry": 5.656854249492381e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 5.656854249492381,
        "max_asymmetry": 0.0,
        "stage": "relative_covariance"
      },
      {
        "allowed_asymmetry": 1e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 0.3535533905932738,
        "max_asymmetry": 0.0,
        "stage": "reverse_distance"
      },
      {
        "allowed_asymmetry": 4.1231056256176605e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 4.123105625617661,
        "max_asymmetry": 0.0,
        "stage": "sample:0"
      },
      {
        "allowed_asymmetry": 1.4142135623730952e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 1.4142135623730951,
        "max_asymmetry": 0.0,
        "stage": "distance_from_a:0"
      },
      {
        "allowed_asymmetry": 5.656854249492381e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 5.656854249492381,
        "max_asymmetry": 0.0,
        "stage": "distance_to_b:0"
      },
      {
        "allowed_asymmetry": 8.246211251235321e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 8.246211251235321,
        "max_asymmetry": 0.0,
        "stage": "sample:1"
      },
      {
        "allowed_asymmetry": 2.8284271247461903e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 2.8284271247461903,
        "max_asymmetry": 0.0,
        "stage": "distance_from_a:1"
      },
      {
        "allowed_asymmetry": 2.8284271247461907e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 2.8284271247461907,
        "max_asymmetry": 0.0,
        "stage": "distance_to_b:1"
      },
      {
        "allowed_asymmetry": 1.6492422502470642e-11,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 16.492422502470642,
        "max_asymmetry": 0.0,
        "stage": "sample:2"
      },
      {
        "allowed_asymmetry": 5.656854249492381e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 5.656854249492381,
        "max_asymmetry": 0.0,
        "stage": "distance_from_a:2"
      },
      {
        "allowed_asymmetry": 1.4142135623730952e-12,
        "averaging_adjustment_frobenius": 0.0,
        "matrix_frobenius": 1.4142135623730951,
        "max_asymmetry": 0.0,
        "stage": "distance_to_b:2"
      }
    ]
  },
  "input_evidence": {
    "covariance_a": {
      "condition_number": 4.0,
      "decomposition_budget": 4.223105625617661e-09,
      "decomposition_residual": 0.0,
      "eigenvalue_floor": 1e-08,
      "eigenvalue_margin": 0.99999999,
      "eigenvalues": [
        1.0,
        4.0
      ],
      "log_determinant": 1.3862943611198906,
      "matrix_frobenius": 4.123105625617661,
      "maximum_eigenvalue": 4.0,
      "minimum_eigenvalue": 1.0
    },
    "covariance_b": {
      "condition_number": 4.0,
      "decomposition_budget": 1.6592422502470644e-08,
      "decomposition_residual": 0.0,
      "eigenvalue_floor": 1e-08,
      "eigenvalue_margin": 3.99999999,
      "eigenvalues": [
        4.0,
        16.0
      ],
      "log_determinant": 4.1588830833596715,
      "matrix_frobenius": 16.492422502470642,
      "maximum_eigenvalue": 16.0,
      "minimum_eigenvalue": 4.0
    },
    "relative_covariance": {
      "condition_number": 1.0,
      "decomposition_budget": 5.756854249492381e-09,
      "decomposition_residual": 0.0,
      "eigenvalue_floor": 1.4210854715202004e-14,
      "eigenvalue_margin": 3.999999999999986,
      "eigenvalues": [
        4.0,
        4.0
      ],
      "log_determinant": 2.772588722239781,
      "matrix_frobenius": 5.656854249492381,
      "maximum_eigenvalue": 4.0,
      "minimum_eigenvalue": 4.0
    }
  },
  "metric": "affine_invariant",
  "operation": "affine-invariant-geodesic-v1",
  "request": {
    "coordinates": [
      {
        "name": "x",
        "unit": "1"
      },
      {
        "name": "y",
        "unit": "1"
      }
    ],
    "covariance_a": [
      [
        1,
        0
      ],
      [
        0,
        4
      ]
    ],
    "covariance_b": [
      [
        4,
        0
      ],
      [
        0,
        16
      ]
    ],
    "experiment_id": "offline-covariance-reference",
    "frame": "declared-dimensionless-coordinate-basis",
    "metric": "affine_invariant",
    "parameters": [
      0,
      0.5,
      1
    ],
    "schema": "covariance-geometry-request-v1",
    "settings": {
      "absolute_tolerance": 1e-10,
      "condition_limit": 1000000,
      "eigenvalue_floor": 1e-08,
      "relative_tolerance": 1e-09,
      "symmetry_tolerance": 1e-12
    }
  },
  "request_digest": "sha256:0c15d6bacf5434a6ae8dd28cdb27138a4cbf5d5bd8ac86fdb90e4ae60d0b9edf",
  "runtime": {
    "arithmetic": "float64",
    "eigensolver": "numpy.linalg.eigh",
    "numpy_version": "2.4.3"
  },
  "samples": [
    {
      "covariance": [
        [
          1.0,
          0.0
        ],
        [
          0.0,
          4.0
        ]
      ],
      "distance_from_a": 0.0,
      "distance_to_b": 1.9605162869370942,
      "parameter": 0,
      "spectrum": {
        "condition_number": 4.0,
        "decomposition_budget": 4.223105625617661e-09,
        "decomposition_residual": 0.0,
        "eigenvalue_floor": 1e-08,
        "eigenvalue_margin": 0.99999999,
        "eigenvalues": [
          1.0,
          4.0
        ],
        "log_determinant": 1.3862943611198906,
        "matrix_frobenius": 4.123105625617661,
        "maximum_eigenvalue": 4.0,
        "minimum_eigenvalue": 1.0
      }
    },
    {
      "covariance": [
        [
          2.0,
          0.0
        ],
        [
          0.0,
          8.0
        ]
      ],
      "distance_from_a": 0.9802581434685471,
      "distance_to_b": 0.9802581434685474,
      "parameter": 0.5,
      "spectrum": {
        "condition_number": 4.0,
        "decomposition_budget": 8.346211251235321e-09,
        "decomposition_residual": 0.0,
        "eigenvalue_floor": 1e-08,
        "eigenvalue_margin": 1.99999999,
        "eigenvalues": [
          2.0,
          8.0
        ],
        "log_determinant": 2.772588722239781,
        "matrix_frobenius": 8.246211251235321,
        "maximum_eigenvalue": 8.0,
        "minimum_eigenvalue": 2.0
      }
    },
    {
      "covariance": [
        [
          4.0,
          0.0
        ],
        [
          0.0,
          16.0
        ]
      ],
      "distance_from_a": 1.9605162869370942,
      "distance_to_b": 0.0,
      "parameter": 1,
      "spectrum": {
        "condition_number": 4.0,
        "decomposition_budget": 1.6592422502470644e-08,
        "decomposition_residual": 0.0,
        "eigenvalue_floor": 1e-08,
        "eigenvalue_margin": 3.99999999,
        "eigenvalues": [
          4.0,
          16.0
        ],
        "log_determinant": 4.1588830833596715,
        "matrix_frobenius": 16.492422502470642,
        "maximum_eigenvalue": 16.0,
        "minimum_eigenvalue": 4.0
      }
    }
  ],
  "schema": "covariance-geometry-result-v1"
}
''')
