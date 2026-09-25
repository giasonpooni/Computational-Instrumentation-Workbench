"""Read-only state-trajectory projection over retained values.

A projection copies retained samples into one declared independent axis, an
ordered list of state axes and optional derived quantities, each with units,
and binds them to the result, execution and verification identities they came
from. It is a display contract: it is not a state estimate, it carries no
covariance, and it never becomes an input to computation. Each projection has
exactly one origin so simulated, reference, observed and estimated trajectories
are never merged into one object.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = "ciw.state-trajectory-projection.v1"
ORIGINS = ("simulation", "reference")
MAX_SAMPLES = 65536


def _axis(value, name):
    if not isinstance(value, dict) or not {"name", "unit", "values"} <= value.keys():
        raise ValueError(f"{name} must declare name, unit and values")
    if not isinstance(value["name"], str) or not value["name"] or not isinstance(value["unit"], str) or not value["unit"]:
        raise ValueError(f"{name} needs a nonempty name and unit")
    values = value["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_SAMPLES:
        raise ValueError(f"{name} values must be a bounded nonempty list")
    for item in values:
        if isinstance(item, bool) or type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(f"{name} values must be finite numbers")
    return deepcopy(value)


def projection(*, origin, label, independent_axis, state_axes, derived_quantities, coordinate_basis,
               provenance, verification_ref=None, selected_index=None):
    """Build a validated projection; every axis has the independent axis's length."""
    if origin not in ORIGINS:
        raise ValueError("Projection origin must be one of " + ", ".join(ORIGINS))
    if not isinstance(label, str) or not label:
        raise ValueError("Projection label must be a nonempty string")
    independent = _axis(independent_axis, "independent_axis")
    count = len(independent["values"])
    previous = -math.inf
    for value in independent["values"]:
        if value <= previous:
            raise ValueError("The independent axis must be strictly increasing")
        previous = value
    if not isinstance(state_axes, list) or not state_axes:
        raise ValueError("At least one state axis is required")
    axes = [_axis(axis, "state_axes[" + str(index) + "]") for index, axis in enumerate(state_axes)]
    derived = []
    for index, quantity in enumerate(derived_quantities or []):
        item = _axis(quantity, "derived_quantities[" + str(index) + "]")
        if not isinstance(quantity.get("definition"), str) or not quantity["definition"]:
            raise ValueError("Derived quantities must state their definition")
        derived.append(item)
    for axis in axes + derived:
        if len(axis["values"]) != count:
            raise ValueError("Every projected axis must have the independent axis's sample count")
    if not isinstance(coordinate_basis, dict) or not isinstance(provenance, dict) or not provenance:
        raise ValueError("Projection needs a coordinate basis and provenance references")
    for key in ("result_id", "execution_id"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise ValueError("Projection provenance must reference the retained result and execution")
    if selected_index is not None and (type(selected_index) is not int or not 0 <= selected_index < count):
        raise ValueError("selected_index must address a retained sample")
    return {"schema": SCHEMA, "origin": origin, "label": label, "sample_count": count,
            "independent_axis": independent, "state_axes": axes, "derived_quantities": derived,
            "coordinate_basis": deepcopy(coordinate_basis), "selected_index": selected_index,
            "provenance": deepcopy(provenance), "verification_ref": verification_ref,
            "authority": {"read_only": True, "state_estimate": False, "covariance": "not_carried",
                          "computation_input": "never"}}
