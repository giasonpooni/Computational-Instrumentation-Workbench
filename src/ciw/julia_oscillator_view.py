"""Read-only projection of one retained Julia oscillator occurrence.

Panels copy retained scalars; the state-trajectory projections copy retained
samples. Nothing here re-solves the model, recomputes energy, re-derives the
oracle comparison or executes a provider.
"""
from __future__ import annotations

from copy import deepcopy

from .experiment_view import SCHEMA, _panel
from .julia_oscillator import UNITS, state_trajectory_view


def project(record, source, declaration, revision):
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    decoded = data["native"]["decoded"]
    verification = native["verification"]
    measured = verification["measured"]
    model = declaration["model"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    declared = {"source_id": source["source_id"], "evidence_id": source["evidence_id"]}
    context = {"object_kind": "numerical_state_trajectory", "owner": step["runtime_ref"], "origin": "simulation",
               "configuration": native["configuration"], "model": deepcopy(model), "grid": deepcopy(declaration["grid"]),
               "covariance_status": "not_applicable", "sensor_fusion": "not_performed", "state_admission": "not_performed",
               "physical_validation": "not_established", "measurement_uncertainty": "not_inferred",
               "specification_identity": data["native"]["specification_identity"],
               "computation_identity": data["native"]["computation_identity"],
               "worker_session_id": data["occurrence"]["worker_session_id"], "engine_occurrence": data["occurrence"]["engine_occurrence"],
               "solver_return_code": decoded["return_code"], "solver_statistics": deepcopy(decoded["solver"]),
               "oracle_outcome": verification["outcome"], "oracle_thresholds": deepcopy(verification["thresholds"]),
               "summary": "Julia Tsit5 numerical integration compared with the analytic oscillator oracle; oracle outcome " + verification["outcome"]}
    panels = [
        _panel("initial-state", "Declared initial state", ["q0", "v0"], [model["initial_q_m"], model["initial_v_m_s"]],
               ["m", "m/s"], None, declared, time_s=0.0, state_order=["q", "v"]),
        _panel("final-state", "Retained final sample", ["q", "v", "energy"],
               [decoded["q"][-1], decoded["v"][-1], decoded["energy"][-1]], ["m", "m/s", "J"], None, provenance,
               time_s=decoded["time_s"][-1], sample_index=len(decoded["time_s"]) - 1),
        _panel("oracle-error", "Normalized maximum error against the analytic oracle", list(UNITS),
               [measured[name]["normalized_max_error"] for name in UNITS], ["1"] * len(UNITS), None, provenance,
               max_abs_error={name: measured[name]["max_abs_error"] for name in UNITS},
               reference_scale={name: measured[name]["reference_scale"] for name in UNITS},
               at_sample_index={name: measured[name]["at_sample_index"] for name in UNITS},
               thresholds=deepcopy(verification["thresholds"]), outcome=verification["outcome"],
               verification_id=verification["verification_id"]),
        _panel("solver-work", "Solver work", ["accepted_steps", "rejected_steps", "function_evaluations"],
               [decoded["solver"][name] for name in ("accepted_steps", "rejected_steps", "function_evaluations")],
               ["count"] * 3, None, provenance, algorithm=native["configuration"]["solver"]["algorithm"],
               abstol=native["configuration"]["solver"]["abstol"], reltol=native["configuration"]["solver"]["reltol"],
               output_policy=native["configuration"]["solver"]["output_policy"]),
    ]
    conservation = measured["numerical_energy_conservation"]
    if conservation["status"] == "evaluated":
        panels.append(_panel("energy-conservation", "Numerical energy drift (undamped)", ["relative_drift"],
                             [conservation["relative_drift"]], ["1"], None, provenance,
                             initial_energy_j=conservation["initial_energy_j"], max_abs_change_j=conservation["max_abs_change_j"]))
    trajectories = state_trajectory_view(native)
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": record["upstream_bundle_id"],
        "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "state_trajectories": trajectories, "schematic": None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k: step[k] for k in ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration,
        "verification": {k: v for k, v in verification.items() if k != "oracle"} | {"oracle": {k: v for k, v in verification["oracle"].items() if k not in UNITS}},
        "runtimes": native["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed",
                      "origin": "simulation", "physical_validation": "not_established"}})
