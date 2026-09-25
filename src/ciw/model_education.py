"""Educational projections and bounded what-if previews for declared models.

These helpers are deliberately outside the retained operation path.  A model
card explains a retained result; a preview explores a changed declaration
without creating an execution or result identity.  A preview must be run
through the registered operation again before it can be retained as evidence.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .julia_oscillator import analytic_oracle, validate_source
from .telemetry import canonical, digest


SCHEMA = "ciw.educational-model-card.v1"
PREVIEW_SCHEMA = "ciw.educational-model-preview.v1"
SWEEP_SCHEMA = "ciw.educational-model-sweep.v1"
_OVERRIDE_FIELDS = {
    "model.omega_0_rad_s": ("model", "omega_0_rad_s"),
    "model.gamma_s_inv": ("model", "gamma_s_inv"),
    "model.mass_kg": ("model", "mass_kg"),
    "initial_state.q0_m": ("initial_state", "q0_m"),
    "initial_state.v0_m_s": ("initial_state", "v0_m_s"),
}


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_overrides(items):
    """Parse explicit ``path=value`` preview controls from the CLI."""
    if not isinstance(items, (list, tuple)) or len(items) > 5:
        raise ValueError("A preview accepts at most five parameter overrides")
    result = {}
    for item in items:
        if not isinstance(item, str) or item.count("=") != 1:
            raise ValueError("Preview overrides must use path=value")
        path, raw = item.split("=", 1)
        if path not in _OVERRIDE_FIELDS:
            raise ValueError(f"Unsupported preview parameter: {path}")
        if path in result:
            raise ValueError(f"Preview parameter repeated: {path}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Preview value for {path} must be numeric") from exc
        result[path] = _finite(value, path)
    return result


def _regime(omega, gamma):
    if gamma == 0.0:
        return "undamped"
    if gamma < omega:
        return "underdamped"
    if math.isclose(gamma, omega, rel_tol=1e-12, abs_tol=1e-12):
        return "critically_damped"
    return "overdamped"


def _energy_behavior(output, gamma):
    initial = float(output["energy_j"][0])
    final = float(output["energy_j"][-1])
    tolerance = 1e-10 * max(1.0, abs(initial))
    if gamma == 0.0 and abs(final - initial) <= tolerance:
        return "conserved_within_reference_tolerance"
    if final < initial - tolerance:
        return "decreasing_in_this_trajectory"
    if final > initial + tolerance:
        return "increased_in_this_trajectory"
    return "approximately_unchanged_in_this_trajectory"


def oscillator_model_card(source, result_data, provenance):
    """Describe a retained oscillator result for teaching and inspection."""
    source = validate_source(canonical(source))
    if not isinstance(result_data, dict) or result_data.get("operation_id") != "ciw.julia-oscillator.v1":
        raise ValueError("Educational card requires a retained Julia oscillator result")
    output = result_data.get("output")
    if not isinstance(output, dict) or len(output.get("time_s", [])) < 2:
        raise ValueError("Educational card requires a trajectory with at least two samples")
    if not isinstance(provenance, dict) or not all(isinstance(provenance.get(k), str) and provenance[k]
                                                    for k in ("source_id", "evidence_id", "result_id", "execution_id")):
        raise ValueError("Educational card requires source, evidence, result and execution provenance")
    model = source["model"]
    initial = source["initial_state"]
    omega = float(model["omega_0_rad_s"])
    gamma = float(model["gamma_s_inv"])
    mass = float(model["mass_kg"])
    card = {
        "schema": SCHEMA,
        "model_family": "linear_damped_harmonic_oscillator",
        "operation_id": "ciw.julia-oscillator.v1",
        "experiment_id": source["experiment_id"],
        "equations": [
            {"id": "state_position", "latex": r"\dot q = v", "meaning": "position derivative equals velocity"},
            {"id": "state_velocity", "latex": r"\dot v = -2\gamma v - \omega_0^2 q", "meaning": "linear restoring and damping dynamics"},
            {"id": "mechanical_energy", "latex": r"E = \frac{1}{2}m\left(v^2 + \omega_0^2 q^2\right)", "meaning": "declared oscillator energy"},
        ],
        "symbols": [
            {"symbol": "q", "meaning": "position", "unit": "m", "value": initial["q0_m"]},
            {"symbol": "v", "meaning": "velocity", "unit": "m/s", "value": initial["v0_m_s"]},
            {"symbol": "\\omega_0", "meaning": "undamped angular frequency", "unit": "rad/s", "value": omega},
            {"symbol": "\\gamma", "meaning": "damping rate", "unit": "1/s", "value": gamma},
            {"symbol": "m", "meaning": "mass", "unit": "kg", "value": mass},
            {"symbol": "E", "meaning": "declared mechanical energy", "unit": "J", "value": output["energy_j"][0]},
        ],
        "parameters": deepcopy(model),
        "initial_state": deepcopy(initial),
        "solver": deepcopy(source["solver"]),
        "derived": {
            "damping_ratio": gamma / omega,
            "regime": _regime(omega, gamma),
            "initial_energy_j": float(output["energy_j"][0]),
            "final_energy_j": float(output["energy_j"][-1]),
            "energy_behavior": _energy_behavior(output, gamma),
        },
        "assumptions": [
            "scalar linear time-invariant second-order model",
            "SI quantities with radians treated as dimensionless in arithmetic",
            "initial state is declared exactly for this simulation request",
            "no measurement noise, sensor model or physical calibration is inferred",
        ],
        "lessons": [
            "Compare gamma with omega_0 to classify the damping regime.",
            "Check the energy trace against the damping assumption rather than looking only at position.",
            "Change one parameter at a time and compare the resulting trajectory with the retained baseline.",
            "A close numerical oracle match establishes the simulated calculation, not the physical model.",
        ],
        "augmentation": {
            "kind": "bounded_parameter_preview",
            "allowed_paths": sorted(_OVERRIDE_FIELDS),
            "requires_new_retained_execution": True,
            "suggested_experiments": [
                "set model.gamma_s_inv=0 and compare energy conservation",
                "increase model.mass_kg while holding the initial state fixed",
                "compare weakly and strongly underdamped responses within the declared bound",
            ],
        },
        "provenance": deepcopy(provenance),
        "authority": {
            "kind": "educational_projection",
            "physical_validation": "not_established",
            "measurement_uncertainty": "not_declared",
            "state_admission": "not_performed",
            "hardware_actuation": "not_performed",
        },
    }
    card["model_card_id"] = digest(card)
    return card


def _trajectory_comparison(baseline, preview):
    """Summarize the bounded preview against its baseline trajectory."""
    comparison = {}
    for key in ("q_m", "v_m_s", "energy_j"):
        baseline_values = [float(value) for value in baseline[key]]
        preview_values = [float(value) for value in preview[key]]
        if len(baseline_values) != len(preview_values):
            raise ValueError("Preview trajectory grids differ")
        deltas = [new - old for old, new in zip(baseline_values, preview_values)]
        comparison[key] = {
            "max_abs_delta": max(abs(delta) for delta in deltas),
            "final_delta": deltas[-1],
        }
    comparison["energy"] = {
        "baseline_initial_j": baseline["energy_j"][0],
        "baseline_final_j": baseline["energy_j"][-1],
        "preview_initial_j": preview["energy_j"][0],
        "preview_final_j": preview["energy_j"][-1],
    }
    return comparison


def preview_oscillator(source, overrides):
    """Evaluate a bounded analytic what-if without creating CIW identities."""
    source = validate_source(canonical(source))
    if not isinstance(overrides, dict):
        raise ValueError("Preview overrides must be an object")
    parsed = parse_overrides([f"{key}={value}" for key, value in overrides.items()])
    augmented = deepcopy(source)
    for path, value in parsed.items():
        section, key = _OVERRIDE_FIELDS[path]
        augmented[section][key] = value
    augmented = validate_source(canonical(augmented))
    baseline_output = analytic_oracle(source)
    output = analytic_oracle(augmented)
    preview = {
        "schema": PREVIEW_SCHEMA,
        "model_family": "linear_damped_harmonic_oscillator",
        "base_source_digest": digest(source),
        "augmented_source": augmented,
        "overrides": parsed,
        "output": output,
        "comparison": _trajectory_comparison(baseline_output, output),
        "authority": {
            "kind": "hypothetical_offline_preview",
            "retention": "not_performed",
            "execution_id": "not_assigned",
            "result_id": "not_assigned",
            "physical_validation": "not_established",
            "state_admission": "not_performed",
        },
        "next_step": "Submit the augmented source to ciw.julia-oscillator.v1 to retain a new execution and result.",
    }
    preview["preview_id"] = digest(preview)
    return preview


def sensitivity_oscillator(source, path, values):
    """Evaluate a bounded one-parameter sensitivity sweep offline.

    Each case is an ordinary hypothetical preview. The sweep groups those
    cases for teaching and comparison, but never creates operation,
    execution, or result identities.
    """
    source = validate_source(canonical(source))
    if not isinstance(path, str) or path not in _OVERRIDE_FIELDS:
        raise ValueError(f"Unsupported sensitivity parameter: {path}")
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 9:
        raise ValueError("A sensitivity sweep requires between two and nine values")
    values = [_finite(value, path) for value in values]
    if len(set(values)) != len(values):
        raise ValueError("Sensitivity sweep values must be distinct")
    cases = []
    for value in values:
        preview = preview_oscillator(source, {path: value})
        cases.append({
            "value": value,
            "preview_id": preview["preview_id"],
            "output": preview["output"],
            "comparison": preview["comparison"],
        })
    sweep = {
        "schema": SWEEP_SCHEMA,
        "model_family": "linear_damped_harmonic_oscillator",
        "base_source_digest": digest(source),
        "swept_path": path,
        "values": values,
        "cases": cases,
        "authority": {
            "kind": "hypothetical_offline_sensitivity",
            "retention": "not_performed",
            "execution_id": "not_assigned",
            "result_id": "not_assigned",
            "physical_validation": "not_established",
            "state_admission": "not_performed",
        },
        "next_step": "Submit selected augmented sources to ciw.julia-oscillator.v1 to retain new executions and results.",
    }
    sweep["sweep_id"] = digest(sweep)
    return sweep

