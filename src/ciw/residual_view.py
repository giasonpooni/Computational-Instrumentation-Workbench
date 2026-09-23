"""Read-only display of retained native residual and observability diagnostics.

Each window keeps its original covariance and lineage. The sequence has no
declared joint covariance, so plotting it must not manufacture a diagonal
covariance or a probability for the declared CUSUM recurrence.
"""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    """Copy a validated monitor occurrence; never call a scientific provider."""
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    rows, scope, policy = data["rows"], data["scope"], data["policy"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    basis = {"epoch": scope["epoch"], "clock_frame": scope["receipt_clock"],
             "frame": scope["measurement_frame"],
             "event_time_basis": "declared_window_target_time",
             "time_unit": "s", "time_policy": "nominal_grid_with_retained_joint_time_uncertainty",
             "temporal_covariance_policy": policy["temporal_covariance_policy"],
             "cross_covariance_policy": policy["cross_covariance_policy"],
             "false_alarm_probability": policy["false_alarm_probability"],
             "prior_feedback": policy["prior_feedback"]}
    panels = []

    def add(name, title, values, unit, *, selected_rows=None, **context):
        selected_rows = rows if selected_rows is None else selected_rows
        bindings = [row["binding"] for row in selected_rows]
        panels.append(_panel(name, title, [binding["target_at"] for binding in bindings],
            values, [unit] * len(selected_rows), None,
            {**provenance, "window_bindings": bindings}, **basis,
            event_times=[binding["target_time"] for binding in bindings],
            sample_event_times=[binding["mapped_event_times"] for binding in bindings],
            device_times=[binding["device_times"] for binding in bindings],
            window_intervals=[binding["window_interval"] for binding in bindings],
            observability_statuses=[row["observability"]["status"] for row in selected_rows],
            interpretations=[row["interpretation"] for row in selected_rows], **context))

    add("innovation", "Retained GSIE prior innovations", [row["detection"]["raw_residual"][0] for row in rows],
        scope["measurement_units"][0],
        covariance_status="individual_window_covariance_only",
        per_window_covariance=[row["detection"]["innovation_covariance"] for row in rows],
        variable_order=scope["measurement_variables"])
    add("normalized-residual", "FDIR marginal normalized innovations",
        [row["detection"]["marginal_normalized_residual"][0] for row in rows], "1",
        covariance_status="joint_covariance_not_declared")
    add("nis", "FDIR normalized innovation squared", [row["detection"]["nis"] for row in rows], "1",
        thresholds=[row["detection"]["threshold"] for row in rows],
        statuses=[row["detection"]["status"] for row in rows])
    monitored_rows = [row for row in rows if row["cusum"] is not None]
    for side in ("positive", "negative") if monitored_rows else ():
        add("cusum-" + side, "FDIR " + side + " CUSUM",
            [row["cusum"]["observed_state"][side] for row in monitored_rows], "1", selected_rows=monitored_rows,
            threshold=policy["cusum"]["threshold"], drift=policy["cusum"]["drift"],
            direction=policy["cusum"]["direction"], reset_on_alarm=policy["cusum"]["reset_on_alarm"],
            accumulator_phase="observed_before_optional_reset",
            statuses=[row["cusum"]["status"] for row in monitored_rows],
            alarm_sides=[row["cusum"]["alarm_sides"] for row in monitored_rows])

    observability = ", ".join(dict.fromkeys(row["observability"]["status"] for row in rows))
    isolability = ", ".join(dict.fromkeys(row["isolability"]["status"] for row in rows))
    context = {"object_kind": "residual_sequence", "owner": "fdir", "companions": ["oit"],
               "summary": f"{len(rows)} retained windows · fixed reference prior · observability: {observability} · "
                          f"isolation: {isolability} · temporal covariance: {policy['temporal_covariance_policy']}",
               "scope": scope, "policy": policy, "overlaps": data["overlaps"],
               "rows": rows, "final_state": data["final_state"],
               "covariance_status": "individual_window_covariance_only",
               "upstream_bundle_ids": declaration["window_bundle_ids"],
               "sensor_fusion": "not_performed", "state_admission": "not_performed"}
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision,
        "bundle_id": record["bundle_id"], "kind": record["kind"], "label": source["label"],
        "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": record["upstream_bundle_id"],
        "upstream_bundle_ids": declaration["window_bundle_ids"],
        "replay_source_bundle_ids": [receipt["source_bundle_digest"] for receipt in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{key: step[key] for key in
                   ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration,
        "verification": native["verification"], "runtimes": native["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection",
                      "state_admission": "not_performed"}})
