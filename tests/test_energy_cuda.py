"""CPU-only contract checks and opt-in real CUDA scientific parity checks.

CIW_TEST_CUDA=1 requires actual hardware; driver errors then fail, never skip.
Ordinary CPU CI does not pretend a mock kernel is a physical GPU execution.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import os

import numpy as np
import pytest

from ciw import energy_cuda as cuda, free_energy_math as mathematics


def problem(correlated=True):
    return {"coordinate_system": "normalized_dimensionless", "prior_mean": [.1, -.2],
        "prior_covariance": [[1, .2], [.2, 1]] if correlated else [[1, 0], [0, 1]],
        "observation_matrix": [[1, .3], [-.2, 1]] if correlated else [[1, 0], [0, 1]],
        "observations": [2, -1], "noise_covariance": [[1, .1], [.1, .7]] if correlated else [[1, 0], [0, .7]]}


def settings():
    return {"initial_mean": [1, -1], "initial_covariance": [[1, .1], [.1, 1]],
        "alpha": .2, "beta": .2, "max_iterations": 256,
        "gradient_tolerance": 1e-12, "precision_tolerance": 1e-12}


def fixed_cpu(value, solver, iterations):
    information = mathematics.information_system(value)
    precision, eta = np.asarray(information["precision"]), np.asarray(information["information_vector"])
    mean = np.asarray(solver["initial_mean"], dtype=float)
    q = np.linalg.solve(solver["initial_covariance"], np.eye(2))
    q = (q + q.T) / 2
    for _ in range(iterations):
        mean = mean - solver["alpha"] * (precision @ mean - eta)
        q = (1 - solver["beta"]) * q + solver["beta"] * precision
    return np.r_[mean, np.linalg.solve(q, np.eye(2)).ravel()]


@pytest.mark.parametrize("field,value", [
    ("iterations", True), ("iterations", -1), ("iterations", 257), ("iterations", 1.0),
    ("replicas", 0), ("replicas", 65537), ("replicas", True),
    ("device_index", -1), ("device_index", 64), ("device_index", False),
])
def test_resource_bounds_are_checked_before_loading_driver(monkeypatch, field, value):
    monkeypatch.setattr(cuda, "_Driver", lambda: pytest.fail("Invalid request loaded the CUDA driver"))
    arguments = {"iterations": 10, "replicas": 1, "device_index": 0, field: value}
    with pytest.raises(ValueError):
        cuda.CudaGaussianWorker(problem(), settings(), **arguments)


@pytest.mark.parametrize("key,value", [
    ("alpha", 0), ("alpha", True), ("alpha", float("nan")), ("alpha", 1e7),
    ("beta", 0), ("beta", 1), ("beta", False),
    ("initial_mean", [0, float("inf")]), ("initial_mean", [0]),
    ("initial_covariance", [[1, 2], [2, 1]]), ("initial_covariance", [[1, .2], [0, 1]]),
    ("gradient_tolerance", 0), ("max_iterations", 0), ("max_iterations", 2),
    ("unknown", "not supported"),
])
def test_solver_domain_is_checked_before_driver_loading(monkeypatch, key, value):
    monkeypatch.setattr(cuda, "_Driver", lambda: pytest.fail("Invalid solver loaded the CUDA driver"))
    solver = settings()
    solver[key] = value
    with pytest.raises(ValueError):
        cuda.CudaGaussianWorker(problem(), solver, 10)


@pytest.mark.parametrize("change", [
    lambda p: p.update(coordinate_system="physical"),
    lambda p: p.update(observation_matrix=[[1, 0]]),
    lambda p: p.update(noise_covariance=[[1, 2], [2, 1]]),
    lambda p: p.update(observations=[True, 1]),
])
def test_normalized_problem_domain_is_checked_before_driver_loading(monkeypatch, change):
    monkeypatch.setattr(cuda, "_Driver", lambda: pytest.fail("Invalid problem loaded the CUDA driver"))
    value = problem()
    change(value)
    with pytest.raises(ValueError):
        cuda.CudaGaussianWorker(value, settings(), 10)


def test_preparation_retains_information_and_initial_state_without_reference_solve(monkeypatch):
    monkeypatch.setattr(mathematics, "gaussian_reference", lambda *_: pytest.fail("Preparation used exact posterior"))
    monkeypatch.setattr(mathematics, "variational_fit", lambda *_: pytest.fail("Preparation used CPU iteration"))
    value, solver = problem(), settings()
    saved = deepcopy((value, solver))
    packed = cuda._prepare(value, solver, 10, 4096, 0)
    information = mathematics.information_system(value)
    assert packed.shape == (15,) and packed.dtype == np.float64 and packed.flags.c_contiguous
    np.testing.assert_array_equal(packed[:4].reshape(2, 2), information["precision"])
    np.testing.assert_array_equal(packed[4:6], information["information_vector"])
    np.testing.assert_array_equal(packed[6:8], solver["initial_mean"])
    np.testing.assert_allclose(packed[8:12].reshape(2, 2) @ solver["initial_covariance"], np.eye(2), atol=1e-15)
    assert packed[12:].tolist() == [.2, .2, .8]
    assert (value, solver) == saved


requires_cuda = pytest.mark.skipif(os.environ.get("CIW_TEST_CUDA") != "1", reason="Set CIW_TEST_CUDA=1 to require actual CUDA execution")


@requires_cuda
@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("iterations,replicas", [(0, 1), (1, 129), (8, 257), (64, 4096), (256, 65536)])
def test_actual_gpu_fixed_iteration_parity_and_identical_replica_bytes(correlated, iterations, replicas):
    value, solver = problem(correlated), settings()
    expected = fixed_cpu(value, solver, iterations)
    with cuda.CudaGaussianWorker(value, solver, iterations, replicas=replicas) as worker:
        output = worker.solve()
        assert output.shape == (replicas, 6) and output.dtype == np.float64 and output.flags.c_contiguous
        np.testing.assert_allclose(output, np.tile(expected, (replicas, 1)), rtol=2e-13, atol=2e-14)
        assert output.tobytes() == output[0].tobytes() * replicas
        assert worker.solve().tobytes() == output.tobytes()  # A fresh launch restarts the initial state.
        identity = worker.identity()
        assert identity["execution_device"] == "gpu"
        assert identity["kernel_sha256"] == sha256(cuda.PTX.encode("ascii")).hexdigest()
        assert identity["iterations"] == iterations and identity["replicas"] == replicas
        assert identity["output_layout"] == cuda.OUTPUT_LAYOUT
        assert identity["device_uuid"].startswith("GPU-") and len(identity["device_uuid"]) == 40
        identity["solver_settings"]["alpha"] = 999
        assert worker.identity()["solver_settings"]["alpha"] == .2
    worker.close()
    with pytest.raises(cuda.CudaError, match="closed"):
        worker.solve()


@requires_cuda
def test_actual_gpu_one_update_has_analytic_mean_and_nonfinal_covariance():
    value = problem(False)
    value.update(prior_mean=[0, 0], noise_covariance=[[1, 0], [0, 1]])
    solver = settings()
    solver["initial_covariance"] = [[1, 0], [0, 1]]
    with cuda.CudaGaussianWorker(value, solver, 1, replicas=3) as worker:
        output = worker.solve()
    np.testing.assert_allclose(output, [[1, -.8, 5 / 6, 0, 0, 5 / 6]] * 3, rtol=0, atol=2e-16)
    reference = mathematics.gaussian_reference(value)
    assert not np.allclose(output[0, 2:].reshape(2, 2), reference["covariance"])


@requires_cuda
def test_actual_gpu_agrees_with_existing_variational_trace_and_independent_reference():
    value, solver = problem(), settings()
    fit = mathematics.variational_fit(value, **solver)
    reference = mathematics.gaussian_reference(value)
    with cuda.CudaGaussianWorker(value, solver, fit["iterations"], replicas=4096) as worker:
        output = worker.solve()[0]
    np.testing.assert_allclose(output[:2], fit["mean"], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(output[2:].reshape(2, 2), fit["covariance"], rtol=2e-13, atol=2e-14)
    assert mathematics.gaussian_kl(output[:2].tolist(), output[2:].reshape(2, 2).tolist(),
                                  reference["mean"], reference["covariance"]) < 1e-20


@requires_cuda
def test_actual_gpu_context_ownership_allows_repeated_workers_and_thread_migration():
    with cuda.CudaGaussianWorker(problem(), settings(), 8, replicas=129) as first:
        expected = first.solve().tobytes()
        with cuda.CudaGaussianWorker(problem(False), settings(), 1, replicas=7) as second:
            assert second.solve().shape == (7, 6)
            assert first.solve().tobytes() == expected
            with ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(lambda _: first.solve().tobytes(), range(4)))
                assert outputs == [expected] * 4
        assert first.solve().tobytes() == expected


@requires_cuda
def test_actual_gpu_divergent_nonfinite_output_is_refused_without_cpu_fallback():
    solver = settings()
    solver["alpha"] = 1e6
    with cuda.CudaGaussianWorker(problem(), solver, 256, replicas=129) as worker:
        with pytest.raises(ValueError, match="nonfinite or out-of-bound"):
            worker.solve()
