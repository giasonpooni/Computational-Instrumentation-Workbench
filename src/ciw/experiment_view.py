"""Read-only display projection of one retained experiment occurrence.

No provider is called and no result is resealed. Panels copy native values and
full covariance; marginal standard deviations are presentation aids only.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = "ciw.experiment-view.v1"


def _panel(name, title, labels, values, units, covariance, provenance, **context):
    return {"panel_id": name, "title": title, "labels": labels, "values": values,
            "units": units, "covariance": covariance,
            "marginal_standard_deviation": None if covariance is None else
                [math.sqrt(covariance[i][i]) for i in range(len(values))],
            "provenance": provenance, "context": context,
            "display_policy": "points_only; marginal_1sigma_not_joint_confidence; no_interpolation"}


def project(record, source, declaration, context, revision):
    """Called with validated retained records under the catalog lock."""
    native = record["native"]
    steps = {s["runtime_ref"]: s for s in native["steps"]}
    panels = []

    def provenance(role):
        s = steps[role]
        return {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                "result_id": s["result_id"], "execution_id": s["execution_id"]}

    def add(role, name, title, labels, values, units, covariance, **basis):
        panels.append(_panel(name, title, labels, values, units, covariance,
                             provenance(role), **basis))

    state_basis = {k: context[k] for k in
                   ("frame_id", "event_time", "epoch", "clock_frame", "clock_frames", "state_kind") if k in context}
    state_basis["uncertainty_scope"] = context.get("uncertainty_scope", "native_declared_model")
    if "epoch_utc" in declaration:
        state_basis["epoch"] = declaration["epoch_utc"]
    add("gsie", "state", "GSIE state", context["state_names"], context["mean"],
        context["units"], context["covariance"], **state_basis)

    if record["kind"] in {"telemetry", "calibrated-window", "acquired-calibrated-window"}:
        calibrated = record["kind"] in {"calibrated-window", "acquired-calibrated-window"}
        feature = steps["stfe"]["result"]
        numerical = feature["numerical_result"]
        config = numerical["config"]
        basis = {"clock_frame": config["clock_basis"], "epoch": declaration["epoch"],
                 "frame_id": config["frame"], "time_unit": "s",
                 "event_times": numerical["sample_event_times"],
                 "time_policy": context.get("time_policy", "declared_identity_clock")}
        add("mcur" if calibrated else "ppda", "measurements",
            "Calibrated samples" if calibrated else "Retained samples",
            [str(t) for t in numerical["sample_event_times"]], numerical["sample_values"],
            [config["value_unit"]] * len(numerical["sample_values"]), numerical["input_covariance"], **basis)
        components = feature["result_artifact"]["components"]
        add("stfe", "feature", "STFE window mean", [c["name"] for c in components],
            [c["value"] for c in components], [c["unit"] for c in components],
            feature["result_artifact"]["covariance"]["matrix"],
            window=native.get("child_window", native)["configuration"]["window"], clock_frame=config["clock_basis"],
            epoch=declaration["epoch"], frame_id=feature["result_artifact"]["covariance"]["frame"]["id"])
        diagnostics = steps["gsie"]["result"]["result_artifact"]["diagnostics"]
        add("gsie", "innovation", "GSIE prior innovation", diagnostics["measurement_variables"],
            diagnostics["innovation"], diagnostics["measurement_units"], diagnostics["innovation_covariance"],
            frame_id=diagnostics["measurement_frame"]["id"],
            normalized_innovation_squared=diagnostics["normalized_innovation_squared"])
        # Posterior residual has no declared covariance here. Do not attach S.
        add("gsie", "posterior-residual", "GSIE posterior residual", diagnostics["measurement_variables"],
            diagnostics["posterior_residual"], diagnostics["measurement_units"], None,
            covariance_status="not_supplied", frame_id=diagnostics["measurement_frame"]["id"])
        if calibrated:
            samples = declaration["samples"]
            n = len(samples)
            joint = declaration["joint_covariance"]["matrix"]
            indicated_cov = [row[n + 2:2 * n + 2] for row in joint[n + 2:2 * n + 2]]
            panels.insert(0, _panel("indications", "Device indications", [str(s["device_time"]) for s in samples],
                [s["indicated_value"] for s in samples], [s["unit"] for s in samples], indicated_cov,
                {"source_id": source["source_id"], "evidence_id": source["evidence_id"]},
                clock_frame=declaration["device_clock"], time_unit=declaration["device_clock"]["unit"],
                event_times=[s["device_time"] for s in samples], frame_id=declaration["frame"]["id"]))
            measurements = next(p for p in panels if p["panel_id"] == "measurements")
            measurements["context"]["joint_time_value_covariance"] = context["joint_time_value_covariance"]
            clock = steps["tbrt"]["result"]["data"]
            add("tbrt", "aligned-time", "TBRT aligned event times", [s["observation_id"] for s in samples],
                [s["event_time"] for s in clock["samples"]], ["s"] * n, clock["temporal_covariance"],
                clock_frame=declaration["receipt_clock"], epoch=declaration["epoch"],
                device_times=[s["device_time"] for s in samples],
                time_policy=context["time_policy"])
    elif record["kind"] == "calibrated-observable":
        measurements = steps["mcur"]["result"]["data"]
        add("mcur", "measurements", "Calibrated measurements", context["channel_ids"],
            measurements["values"], measurements["units"], measurements["covariance"]["matrix"],
            frame_id=context["frame_id"], event_time=context["event_time"],
            epoch=declaration["epoch_utc"], clock_frames=context["clock_frames"])
        gsie = steps["gsie"]["result"]["data"]
        add("gsie", "innovation", "GSIE prior innovation", context["channel_ids"],
            gsie["innovation"], measurements["units"], gsie["innovation_covariance"],
            frame_id=context["frame_id"], normalized_innovation_squared=gsie["nis"])
        add("gsie", "posterior-residual", "GSIE posterior residual", context["channel_ids"],
            gsie["residual"], measurements["units"], None, covariance_status="not_supplied")

    if "cbsr" in steps:
        result = steps["cbsr"]["result"]
        result = result.get("data", result)
        reconciled = result.get("reconciled")
        # Held/refused candidates stay in assessment/native records, never in
        # a panel labeled as a reconciled state.
        if result["status"] == "accepted" and reconciled is not None:
            add("cbsr", "reconciled", "CBSR reconciled candidate", context["state_names"],
                reconciled["estimate"], context["units"], reconciled["covariance"],
                **state_basis, disposition=result["status"])
        if result.get("residual_pre") is not None:
            rows = result["request"]["constraints"]["row_units"]
            add("cbsr", "constraint-residual", "CBSR constraint residual before reconciliation",
                ["constraint " + str(i + 1) for i in range(len(rows))], result["residual_pre"], rows,
                result["residual_covariance_pre"], constraint_id=result["constraint_id"])

    nodes = [{"role": s["runtime_ref"], "operation_id": s["operation_id"],
              "result_id": s["result_id"], "execution_id": s["execution_id"],
              "numerical_result_id": s.get("numerical_result_id"), "input_refs": s["input_refs"]}
             for s in native["steps"]]
    # The graph is the recorded input relation, not a dependency inferred from
    # display order. Upstream references remain explicit external nodes.
    view = {"schema": SCHEMA, "catalog_revision": revision,
            "bundle_id": record["bundle_id"], "kind": record["kind"], "label": source["label"],
            "source_id": source["source_id"], "evidence_id": source["evidence_id"],
            "upstream_bundle_id": record["upstream_bundle_id"],
            "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
            "experiment_id": declaration.get("experiment_id"),
            "fusion_context": context, "panels": panels, "graph": {"nodes": nodes},
            "raw_observations": declaration.get("samples", declaration.get("channels", [])),
            "verification": native["verification"], "runtimes": native["runtimes"],
            "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection",
                          "state_admission": "not_performed"}}
    if record["kind"] == "acquired-calibrated-window":
        view["acquisition_binding"] = native["acquisition_binding"]
        view["native_window_bundle_id"] = native["child_window"]["bundle_digest"]
        view["native_window_verification"] = native["child_window"]["verification"]
        view["derived_window_declaration"] = declaration
    return deepcopy(view)
