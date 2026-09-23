"""Consistent multiplicative unit rescaling of a model and everything bound to it.

Changing a declared unit changes numbers, not meaning. For each rescaled
symbol ``value_new = k * value_old`` with ``k = scale_old / scale_new``. The
transformation is applied together to parameter values, covariance entries
(``P'_ij = k_i k_j P_ij``, with process-noise densities also divided by the
independent-variable factor), validity bounds and, for a request, initial
states, inputs, solver tolerances and the sample grid. Equations are
unit-aware SI expressions and remain unchanged. A partial transformation,
such as relabelling ``m`` as ``mm`` without converting values, is a different
model and is detectable by the invariance checks in the tests.
"""
from __future__ import annotations

from copy import deepcopy

from .simulation import validate_request
from .spec import Model, SpecificationError, validate_spec
from .units import parse_unit


def factors(model: Model, units: dict) -> dict[str, float]:
    if not isinstance(units, dict) or not units:
        raise SpecificationError("A rescale requires at least one symbol -> unit mapping")
    result = {}
    for symbol, text in units.items():
        if symbol not in model.units:
            raise SpecificationError(f"Cannot rescale undeclared symbol: {symbol}")
        try:
            new = parse_unit(text)
            result[symbol] = model.units[symbol].factor_to(new)
        except ValueError as exc:
            raise SpecificationError(f"{symbol}: {exc}") from exc
    return result


def _matrix(block, k, divisor=1.0):
    if block is None:
        return None
    order = block["order"]
    scale = [k.get(symbol, 1.0) for symbol in order]
    return {"order": list(order), "matrix": [[(value * scale[i]) * scale[j] / divisor
                                              for j, value in enumerate(row)] for i, row in enumerate(block["matrix"])]}


def rescale(model: Model, units: dict, *, note: str | None = None) -> Model:
    """Return a new validated model expressing ``units`` for the named symbols."""
    k = factors(model, units)
    spec = deepcopy(model.spec)
    entries = [spec["independent_variable"], *spec["states"], *spec["inputs"], *spec["parameters"],
               *spec["derived"], *spec["observations"]]
    for entry in entries:
        symbol = entry["symbol"]
        if symbol in units:
            entry["unit"] = units[symbol]
            if "value" in entry:
                entry["value"] = entry["value"] * k[symbol]
    uncertainty = spec["uncertainty"]
    kt = k.get(model.independent, 1.0)
    for name in ("parameters", "initial_state", "measurement_noise"):
        uncertainty[name] = _matrix(uncertainty[name], k)
    # Continuous white-noise density: Cov(dW) = Q dt, so Q has state^2 / independent units.
    uncertainty["process_noise"] = _matrix(uncertainty["process_noise"], k, kt)
    domain = spec["validity_domain"]
    if domain["independent_variable"] is not None:
        domain["independent_variable"] = [None if value is None else value * kt for value in domain["independent_variable"]]
    for symbol, bounds in domain["bounds"].items():
        factor = k.get(symbol, 1.0)
        domain["bounds"][symbol] = [None if value is None else value * factor for value in bounds]
    spec["provenance"] = {
        "kind": "unit_rescale",
        "derived_from": {"spec_digest": model.digest,
                         "units": {symbol: [model.entries[symbol]["unit"], text] for symbol, text in sorted(units.items())},
                         "factors": {symbol: k[symbol] for symbol in sorted(k)}},
        "notes": [note] if note else [],
    }
    return validate_spec(spec)


def rescale_request(original: Model, rescaled: Model, request: dict) -> dict:
    """Transform a request consistently with ``rescale``; tolerances move with their states."""
    validate_request(original, request)
    derivation = rescaled.spec["provenance"]
    if derivation["kind"] != "unit_rescale" or derivation["derived_from"]["spec_digest"] != original.digest:
        raise SpecificationError("The target model is not a unit rescale of the request's model")
    k = derivation["derived_from"]["factors"]
    kt = k.get(original.independent, 1.0)
    states = original.state_order
    result = deepcopy(request)
    result["spec_digest"] = rescaled.digest
    result["initial_state"] = [value * k.get(symbol, 1.0) for symbol, value in zip(states, request["initial_state"])]
    result["inputs"] = {symbol: value * k.get(symbol, 1.0) for symbol, value in request["inputs"].items()}
    result["sample_times"] = [value * kt for value in request["sample_times"]]
    result["interval"] = [value * kt for value in request["interval"]]
    solver = result["solver"]
    solver["abstol"] = [value * k.get(symbol, 1.0) for symbol, value in zip(states, request["solver"]["abstol"])]
    if solver["dtmax"] is not None:
        solver["dtmax"] = solver["dtmax"] * kt
    result["acceptance"]["state_atol"] = [value * k.get(symbol, 1.0) for symbol, value
                                          in zip(states, request["acceptance"]["state_atol"])]
    return validate_request(rescaled, result)


def to_si(model: Model, symbol: str, value):
    """Convert a declared value to coherent SI for cross-unit comparison."""
    return value * model.units[symbol].scale


def covariance_si(model: Model, block: str) -> dict | None:
    """Return a covariance block in coherent SI units (for invariance comparisons)."""
    source = model.spec["uncertainty"][block]
    if source is None:
        return None
    scales = {symbol: model.units[symbol].scale for symbol in source["order"]}
    divisor = model.units[model.independent].scale if block == "process_noise" else 1.0
    return {"order": list(source["order"]),
            "matrix": [[value * scales[a] * scales[b] / divisor for b, value in zip(source["order"], row)]
                       for a, row in zip(source["order"], source["matrix"])]}
