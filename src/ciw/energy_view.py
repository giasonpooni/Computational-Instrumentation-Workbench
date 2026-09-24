"""Display retained energy analysis without sampling hardware or rerunning it."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    bundle = record["native"]
    step, = bundle["steps"]
    data = step["result"]["data"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    basis = {"origin": data["origin"], "hardware_provenance": data["hardware_provenance"],
             "measurement_scope": data["measurement_scope"], "measurement_run_id": declaration["run_id"],
             "log_digest": data["log_digest"], "replay_scope": bundle["verification"]["method"]}
    phases = data["phases"]
    energy = [phase for phase in phases if phase["gross_energy_j"] is not None]
    measurement = data["measurement"]
    batches = phases[3]["batches"]
    ratio = measurement["amortized_domain_energy_j_per_qualified_solve"]
    panels = [
        _panel("gross-energy", "Gross GPU-domain counter energy by phase", [p["name"] for p in energy],
            [p["gross_energy_j"] for p in energy], ["J"] * len(energy), None, provenance, **basis,
            idle_subtracted=False, unavailable_phases={p["name"]: p["reasons"] for p in phases if p["gross_energy_j"] is None},
            energy_status_by_phase={p["name"]: p["energy_status"] for p in phases}),
        _panel("elapsed-time", "Host elapsed time by phase", [p["name"] for p in phases],
            [p["elapsed_s"] * 1000 for p in phases], ["ms"] * len(phases), None, provenance, **basis,
            clock=declaration["clock"], counter_bracket_elapsed_ms=[None if p["bracket_elapsed_s"] is None else p["bracket_elapsed_s"] * 1000 for p in phases],
            measurement_boundary=declaration["runtime"]["workload"]["measurement_boundary"]),
        _panel("accuracy", "Gaussian KL for each retained measurement batch", [str(p["batch_index"]) for p in batches],
            [p["kl_nats"] for p in batches], ["nat"] * len(batches), None, provenance, **basis,
            target_kl_nats=measurement["target_kl_nats"], target_met=measurement["target_met"],
            accuracy_basis=data["comparison"]["accuracy_basis"], qualified_solves=measurement["qualified_solves"],
            total_solves=measurement["total_solves"], first_attainment="not_measured"),
        _panel("energy-per-solve", "Amortized gross GPU energy per qualified solve", [] if ratio is None else ["qualified solve"],
            [] if ratio is None else [ratio], [] if ratio is None else ["J/solve"], None, provenance, **basis,
            available=ratio is not None, comparison=data["comparison"], uncertainty="sensor_resolution_and_accuracy_not_characterized"),
    ]
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": None, "replay_source_bundle_ids": [r["source_bundle_digest"] for r in bundle.get("replay_receipts", [])],
        "experiment_id": declaration["run_id"], "fusion_context": None,
        "object_context": {"object_kind": record["kind"],
            "summary": "Retained GPU energy and Gaussian accuracy: " + ("synthetic fixture" if data["origin"] == "synthetic_fixture" else "physical measurement log"),
            **basis, "measurement": measurement, "comparison": data["comparison"], "sensor": declaration["sensor"],
            "capture_runtime": declaration["runtime"], "measurement_plan": declaration["plan"],
            "analysis_scope": "offline_retained_log_only", "fresh_hardware_measurement": False},
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k: step[k] for k in ("operation_id", "execution_id", "result_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "verification": bundle["verification"], "runtimes": bundle["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection",
                      "physical_measurement": "not_performed_by_inspection", "state_admission": "not_performed"}})
