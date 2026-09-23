"""Unit rescaling must transform model, covariance, bounds, tolerances and grid together."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw.model.providers import finite_difference_linearization
from ciw.model.simulation import simulate_reference, uniform_request
from ciw.model.spec import SpecificationError, validate_spec
from ciw.model.transform import covariance_si, rescale, rescale_request

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
MM = {"q": "mm", "v": "mm/s", "y": "mm"}


def oscillator():
    return validate_spec(json.loads((EXAMPLES / "damped-oscillator.json").read_text()))


def with_process_noise():
    spec = deepcopy(oscillator().spec)
    spec["uncertainty"]["process_noise"] = {"order": ["q", "v"], "matrix": [[1e-10, 0.0], [0.0, 4e-6]]}
    spec["uncertainty"]["assumptions"] = [a for a in spec["uncertainty"]["assumptions"] if a != "no_process_noise"] + ["white_process_noise"]
    return validate_spec(spec)


def mixed_request(model):
    return uniform_request(model, initial_state=[0.3, -0.4], inputs={"F": 0.5}, abstol=[1e-12, 1e-12],
                           state_atol=[1e-8, 1e-8])


def test_metres_to_millimetres_transforms_model_and_covariance_consistently():
    original = oscillator()
    mm = rescale(original, MM)
    assert mm.spec["states"][0]["unit"] == "mm" and mm.spec["dynamics"] == original.spec["dynamics"]
    assert np.allclose(mm.spec["uncertainty"]["initial_state"]["matrix"], [[1.0, 0.0], [0.0, 1.0]], rtol=1e-12)
    assert np.allclose(mm.spec["uncertainty"]["measurement_noise"]["matrix"], [[0.01]], rtol=1e-12)
    assert mm.spec["validity_domain"]["bounds"]["q"] == pytest.approx([-10000.0, 10000.0])
    for block in ("initial_state", "measurement_noise", "parameters"):
        assert np.allclose(covariance_si(mm, block)["matrix"], covariance_si(original, block)["matrix"], rtol=1e-12, atol=0)
    assert mm.spec["provenance"]["derived_from"]["spec_digest"] == original.digest
    assert mm.spec["provenance"]["derived_from"]["factors"]["q"] == pytest.approx(1000.0)

    request = mixed_request(original)
    transformed = rescale_request(original, mm, request)
    assert transformed["solver"]["abstol"] == pytest.approx([1e-9, 1e-9])
    assert transformed["acceptance"]["state_atol"] == pytest.approx([1e-5, 1e-5])
    a, b = simulate_reference(original, request), simulate_reference(mm, transformed)
    assert np.allclose(np.asarray(b["states"]) / 1000.0, a["states"], rtol=1e-12, atol=1e-15)
    assert np.allclose(np.asarray(b["observations"]["y"]) / 1000.0, a["observations"]["y"], rtol=1e-12, atol=1e-15)
    assert np.allclose(b["derived"]["E"], a["derived"]["E"], rtol=1e-12)  # energy stays in J


def test_relabelling_units_without_converting_values_is_a_different_model():
    original = oscillator()
    relabelled = deepcopy(original.spec)
    for entry in relabelled["states"]:
        entry["unit"] = {"q": "mm", "v": "mm/s"}[entry["symbol"]]
    relabelled["observations"][0]["unit"] = "mm"
    naive = validate_spec(relabelled)  # syntactically valid, semantically different
    request = mixed_request(original)
    naive_request = deepcopy(request)
    naive_request["spec_digest"] = naive.digest
    a, b = simulate_reference(original, request), simulate_reference(naive, naive_request)
    # Same numbers now mean millimetres: the trajectories disagree after conversion.
    assert not np.allclose(np.asarray(b["states"]) / 1000.0, a["states"], atol=1e-6)
    assert covariance_si(naive, "initial_state") != covariance_si(original, "initial_state")


def test_time_rescale_transforms_grid_rates_and_process_noise_density():
    original = with_process_noise()
    ms = rescale(original, {"t": "ms"})
    assert ms.spec["parameters"][0]["value"] == pytest.approx(original.spec["parameters"][0]["value"])  # rad/s unchanged
    assert ms.spec["validity_domain"]["independent_variable"] == pytest.approx([0.0, 12000.0])
    # Q has state^2 / time units: per millisecond it is 1000x smaller.
    assert ms.spec["uncertainty"]["process_noise"]["matrix"][1][1] == pytest.approx(4e-9)
    assert np.allclose(covariance_si(ms, "process_noise")["matrix"], covariance_si(original, "process_noise")["matrix"])
    request = mixed_request(original)
    transformed = rescale_request(original, ms, request)
    assert transformed["sample_times"][1] == pytest.approx(15.625)
    a, b = simulate_reference(original, request), simulate_reference(ms, transformed)
    assert np.allclose(b["states"], a["states"], rtol=1e-12, atol=1e-14)


def test_linearization_transforms_by_similarity_under_rescale():
    original = oscillator()
    mm_ms = rescale(original, {"q": "mm", "v": "mm/s", "y": "mm", "t": "ms"})
    request = {"state": [0.2, -0.1], "inputs": {"F": 0.3}, "time": 0.0,
               "reference": {"fd_relative_step": 1e-6}}
    scaled = {"state": [200.0, -100.0], "inputs": {"F": 0.3}, "time": 0.0,
              "reference": {"fd_relative_step": 1e-6}}
    a = finite_difference_linearization(original, request)
    b = finite_difference_linearization(mm_ms, scaled)
    K = np.diag([1000.0, 1000.0])
    # x' = K x and tau' = 1000 tau, so A' = K A K^-1 / 1000 and B' = K B / 1000.
    assert np.allclose(b["A"], K @ a["A"] @ np.linalg.inv(K) / 1000.0, rtol=1e-6, atol=1e-12)
    assert np.allclose(b["B"], K @ a["B"] / 1000.0, rtol=1e-6)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(b["A"])), np.sort_complex(np.linalg.eigvals(a["A"])) / 1000.0,
                       rtol=1e-6)


def test_rescale_refuses_dimension_changes_and_unknown_symbols():
    original = oscillator()
    with pytest.raises(SpecificationError, match="dimensions differ"):
        rescale(original, {"q": "s"})
    with pytest.raises(SpecificationError, match="undeclared"):
        rescale(original, {"x": "m"})
    with pytest.raises(SpecificationError, match="Affine"):
        rescale(original, {"q": "degC"})
    unrelated = validate_spec(json.loads((EXAMPLES / "pd-controller.json").read_text()))
    with pytest.raises(SpecificationError, match="not a unit rescale"):
        rescale_request(original, unrelated, mixed_request(original))
