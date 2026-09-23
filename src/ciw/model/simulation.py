"""Simulation requests and the independent Python reference integrator.

A ``ciw.model-simulation-request.v1`` states every setting that affects a
numerical result: the exact sample times, the half-open interval, initial
state in state order and declared units, constant input values, the Julia
solver configuration (Tsit5 with explicit tolerances) and the reference
method used by this module (classical fixed-substep RK4). Solver tolerances
are declared per state in the state's own unit, so a consistent unit rescale
transforms them with the model; they are never a measurement uncertainty.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from . import expression
from .spec import Model, SpecificationError, _inside, _keys, lower

REQUEST_SCHEMA = "ciw.model-simulation-request.v1"
REFERENCE_SCHEMA = "ciw.model-reference-trajectory.v1"
MAX_SAMPLES = 4096
MAX_SUBSTEPS = 4096
SAVE_POLICY = "interpolated_saveat"
INITIAL_DT = "solver_automatic"


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SpecificationError(f"{name} must be a finite JSON number, not a boolean")
    return float(value)


def validate_request(model: Model, request) -> dict:
    """Validate a request against its model; returns the request unchanged."""
    _keys(request, {"schema", "spec_digest", "initial_state", "inputs", "sample_times", "interval",
                    "solver", "reference", "acceptance"}, name="simulation request")
    if request["schema"] != REQUEST_SCHEMA:
        raise SpecificationError("Unsupported simulation request schema")
    if request["spec_digest"] != model.digest:
        raise SpecificationError("Simulation request is bound to a different model specification")
    states, spec = model.state_order, model.spec
    initial = request["initial_state"]
    if not isinstance(initial, list) or len(initial) != len(states):
        raise SpecificationError(f"initial_state must list {len(states)} values in state order {states}")
    bounds = spec["validity_domain"]["bounds"]
    for symbol, value in zip(states, initial):
        _finite(value, f"initial_state.{symbol}")
        if symbol in bounds and not _inside(value, bounds[symbol]):
            raise SpecificationError(f"Initial {symbol} lies outside the declared validity domain")
    inputs = request["inputs"]
    declared_inputs = [item["symbol"] for item in spec["inputs"]]
    if not isinstance(inputs, dict) or set(inputs) != set(declared_inputs):
        raise SpecificationError(f"inputs must give a constant value for exactly {declared_inputs}")
    for symbol, value in inputs.items():
        _finite(value, f"inputs.{symbol}")
        if symbol in bounds and not _inside(value, bounds[symbol]):
            raise SpecificationError(f"Input {symbol} lies outside the declared validity domain")
    times, interval = request["sample_times"], request["interval"]
    if not isinstance(times, list) or not 2 <= len(times) <= MAX_SAMPLES:
        raise SpecificationError(f"sample_times must contain 2..{MAX_SAMPLES} values")
    for index, value in enumerate(times):
        _finite(value, f"sample_times[{index}]")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise SpecificationError("sample_times must be strictly increasing")
    if not isinstance(interval, list) or len(interval) != 2:
        raise SpecificationError("interval must be [start, end)")
    start, end = (_finite(value, "interval endpoint") for value in interval)
    if times[0] != start or not times[-1] < end:
        raise SpecificationError("The first sample is the interval start and the half-open end is excluded")
    domain = spec["validity_domain"]["independent_variable"]
    if domain is not None and not (_inside(start, domain) and _inside(end, domain)):
        raise SpecificationError("The requested interval lies outside the declared validity domain")
    solver = request["solver"]
    _keys(solver, {"algorithm", "reltol", "abstol", "maxiters", "dtmax", "initial_dt", "save_policy"}, name="solver")
    if solver["algorithm"] != "Tsit5" or solver["save_policy"] != SAVE_POLICY or solver["initial_dt"] != INITIAL_DT:
        raise SpecificationError(f"solver must declare Tsit5, {SAVE_POLICY} and {INITIAL_DT}")
    reltol = _finite(solver["reltol"], "solver.reltol")
    if not 1e-14 <= reltol <= 1e-2:
        raise SpecificationError("solver.reltol must lie in [1e-14, 1e-2]")
    abstol = solver["abstol"]
    if not isinstance(abstol, list) or len(abstol) != len(states):
        raise SpecificationError("solver.abstol must give one tolerance per state, in state units")
    for symbol, value in zip(states, abstol):
        if _finite(value, f"solver.abstol.{symbol}") <= 0.0:
            raise SpecificationError("solver.abstol entries must be positive")
    if type(solver["maxiters"]) is not int or not 1 <= solver["maxiters"] <= 10_000_000:
        raise SpecificationError("solver.maxiters must be an integer in 1..10000000")
    if solver["dtmax"] is not None and _finite(solver["dtmax"], "solver.dtmax") <= 0.0:
        raise SpecificationError("solver.dtmax must be positive or null (unbounded)")
    reference = request["reference"]
    _keys(reference, {"method", "substeps"}, name="reference")
    if reference["method"] != "rk4" or type(reference["substeps"]) is not int or not 1 <= reference["substeps"] <= MAX_SUBSTEPS:
        raise SpecificationError(f"reference must be rk4 with 1..{MAX_SUBSTEPS} substeps")
    acceptance = request["acceptance"]
    _keys(acceptance, {"state_atol", "state_rtol"}, name="acceptance")
    if (not isinstance(acceptance["state_atol"], list) or len(acceptance["state_atol"]) != len(states)
            or any(_finite(value, "acceptance.state_atol") < 0 for value in acceptance["state_atol"])):
        raise SpecificationError("acceptance.state_atol must give one nonnegative threshold per state, in state units")
    if not 0 <= _finite(acceptance["state_rtol"], "acceptance.state_rtol") < 1:
        raise SpecificationError("acceptance.state_rtol must lie in [0, 1)")
    return request


def uniform_request(model: Model, *, initial_state, inputs=None, start=0.0, rate=64.0, count=768,
                    reltol=1e-10, abstol=None, substeps=64, maxiters=1_000_000, dtmax=None,
                    state_atol=None, state_rtol=1e-6) -> dict:
    """Build a request on the uniform half-open grid ``start + k / rate``."""
    times = [start + index / rate for index in range(count)]
    request = {
        "schema": REQUEST_SCHEMA, "spec_digest": model.digest,
        "initial_state": [float(value) for value in initial_state],
        "inputs": {key: float(value) for key, value in (inputs or {}).items()},
        "sample_times": times, "interval": [start, start + count / rate],
        "solver": {"algorithm": "Tsit5", "reltol": reltol,
                   "abstol": list(abstol) if abstol is not None else [1e-12] * len(model.state_order),
                   "maxiters": maxiters, "dtmax": dtmax, "initial_dt": INITIAL_DT, "save_policy": SAVE_POLICY},
        "reference": {"method": "rk4", "substeps": substeps},
        # Declared before execution: componentwise |julia - rk4| <= atol + rtol*|rk4|.
        "acceptance": {"state_atol": list(state_atol) if state_atol is not None else [1e-8] * len(model.state_order),
                       "state_rtol": state_rtol},
    }
    return validate_request(model, request)


class CompiledModel:
    """Evaluate a lowered model in declared units (shared by every Python path)."""

    def __init__(self, model: Model):
        self.model, self.lowered = model, lower(model)
        lowered = self.lowered
        self.t_symbol, self.t_scale = lowered["independent"]["symbol"], lowered["independent"]["scale"]
        self.state_symbols = [item["symbol"] for item in lowered["states"]]
        self.state_scales = np.array([item["scale"] for item in lowered["states"]], dtype=np.float64)
        self.parameters = {item["symbol"]: np.float64(item["value"]) * item["scale"] for item in lowered["parameters"]}
        self.input_scales = {item["symbol"]: item["scale"] for item in lowered["inputs"]}

    def environment(self, t, x, inputs, parameters=None):
        values = dict(self.parameters if parameters is None else parameters)
        values[self.t_symbol] = t * self.t_scale
        for index, symbol in enumerate(self.state_symbols):
            values[symbol] = x[index] * self.state_scales[index]
        for symbol, value in inputs.items():
            values[symbol] = np.float64(value) * self.input_scales[symbol]
        for item in self.lowered["derived"]:
            values[item["symbol"]] = expression.evaluate(item["expression"], values)
        return values

    def rhs(self, t, x, inputs, parameters=None):
        values = self.environment(t, x, inputs, parameters)
        return np.array([expression.evaluate(item, values) for item in self.lowered["dynamics"]],
                        dtype=np.float64) * self.t_scale / self.state_scales

    def outputs(self, t, x, inputs, parameters=None):
        values = self.environment(t, x, inputs, parameters)
        derived = {item["symbol"]: values[item["symbol"]] / item["scale"] for item in self.lowered["derived"]}
        observed = {item["symbol"]: expression.evaluate(item["expression"], values) / item["scale"]
                    for item in self.lowered["observations"]}
        return derived, observed


def simulate_reference(model: Model, request: dict) -> dict:
    """Integrate with classical RK4, ``substeps`` equal steps per sample interval.

    This is an independent reference for the Julia Tsit5 provider; it is not
    adaptive and claims no error bound beyond the recorded comparison.
    """
    validate_request(model, request)
    compiled = CompiledModel(model)
    times = np.asarray(request["sample_times"], dtype=np.float64)
    inputs = {key: float(value) for key, value in request["inputs"].items()}
    substeps = request["reference"]["substeps"]
    x = np.asarray(request["initial_state"], dtype=np.float64)
    states = np.empty((len(times), len(x)), dtype=np.float64)
    states[0] = x
    evaluations = 0
    with np.errstate(all="ignore"):
        for index in range(1, len(times)):
            t, h = times[index - 1], (times[index] - times[index - 1]) / substeps
            for step in range(substeps):
                tau = t + step * h
                k1 = compiled.rhs(tau, x, inputs)
                k2 = compiled.rhs(tau + h / 2, x + h / 2 * k1, inputs)
                k3 = compiled.rhs(tau + h / 2, x + h / 2 * k2, inputs)
                k4 = compiled.rhs(tau + h, x + h * k3, inputs)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                evaluations += 4
            if not np.all(np.isfinite(x)):
                return {"schema": REFERENCE_SCHEMA, "status": "halted", "spec_digest": model.digest,
                        "detail": f"nonfinite state before sample {index}", "rhs_evaluations": evaluations}
            states[index] = x
        derived = {item["symbol"]: [] for item in compiled.lowered["derived"]}
        observed = {item["symbol"]: [] for item in compiled.lowered["observations"]}
        for t, row in zip(times, states):
            d, o = compiled.outputs(t, row, inputs)
            for key in derived:
                derived[key].append(float(d[key]))
            for key in observed:
                observed[key].append(float(o[key]))
    return {
        "schema": REFERENCE_SCHEMA, "status": "completed", "spec_digest": model.digest,
        "method": "rk4", "substeps": substeps, "rhs_evaluations": evaluations,
        "sample_times": times.tolist(), "state_order": list(compiled.state_symbols),
        "states": states.tolist(), "derived": derived, "observations": observed,
        "units": {symbol: model.entries[symbol]["unit"] for symbol in
                  [*compiled.state_symbols, *derived, *observed]},
    }


def request_copy(request: dict) -> dict:
    return deepcopy(request)
