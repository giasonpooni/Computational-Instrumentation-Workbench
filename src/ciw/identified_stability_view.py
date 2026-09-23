"""Read-only projection of an already retained PLSR computational verdict."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    binding, verdict = data["binding"], data["record"]
    diagnostics = verdict["diagnostics"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    state_provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                        "result_id": binding["state_result_id"], "execution_id": binding["state_execution_id"],
                        "upstream_bundle_id": binding["upstream_bundle_id"]}
    panels = [_panel("retained-state", "Selected GSIE prediction", binding["state_names"], binding["mean"],
        binding["state_units"], binding["covariance"], state_provenance,
        frame_id=binding["frame_id"], clock_frame=binding["clock_frame"], event_time=binding["time"],
        uncertainty_scope=binding["uncertainty_scope"], covariance_usage="context_only_not_a_certificate_bound")]
    if diagnostics is not None:
        panels.append(_panel("quadratic-values", "PLSR declared quadratic sample", ["V", "Delta V"],
            [diagnostics["value"], diagnostics["decrease"]], ["1", "1"], None, provenance,
            time_convention="discrete", sample_period_s=binding["sample_interval"], status=verdict["code"],
            covariance_status="not_propagated", level=verdict["level"]))
        panels.append(_panel("certificate-margin", "PLSR margin and numerical resolution",
            ["margin", "required margin", "resolution"],
            [diagnostics["margin"], verdict["required_margin"], diagnostics["resolution"]],
            [declaration["certificate_unit"]] * 3, None, provenance,
            coordinate_metric=data["policy"]["coordinate_metric"], status=verdict["code"]))
    context = {"object_kind": "identified_model_stability", "owner": "plsr",
               "summary": "Native PLSR: " + verdict["code"] + " · conditional discrete point model · state covariance retained as context",
               "status": verdict["code"], "native_record": verdict, "binding": binding, "policy": data["policy"],
               "equilibrium": declaration["equilibrium"], "certificate_unit": declaration["certificate_unit"],
               "covariance_status": "retained_context_not_propagated_through_certificate",
               "physical_stability": "not_established", "sensor_fusion": "not_performed", "state_admission": "not_performed"}
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": record["upstream_bundle_id"],
        "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": "plsr", **{k: step[k] for k in
                   ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration, "verification": native["verification"],
        "runtimes": native["runtimes"], "authority": {"read_only": True,
            "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
