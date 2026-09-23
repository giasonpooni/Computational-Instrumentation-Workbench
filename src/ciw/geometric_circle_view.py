"""Read-only projection of native GTE geometry, preserving covariance bases."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel
from .geometric_circle import request


def project(record, source, declaration, revision):
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    declared = request(declaration)
    observations = declared["observations"]
    ids = observations["observation_ids"]
    labels = [identifier + "." + axis for identifier in ids for axis in ("x", "y")]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    context = {"object_kind": "geometric_reconciliation", "owner": "gte",
               "summary": "Native circle candidate: " + data["reconciliation"]["status"],
               "configuration": native["configuration"], "coordinate_frame": data["coordinate_frame"],
               "frame_authority": "declared_not_surveyed", "constraint": data["constraint"],
               "reconciliation": data["reconciliation"], "time_origin": data["time_origin"],
               "time_s": data["time_s"], "geodesic_steps": data["geodesic_steps"],
               "uncertainty": data["uncertainty"], "self_checks": data["self_checks"],
               "covariance_status": "native_first_order_full_joint_not_posterior",
               "sensor_fusion": "not_performed", "state_admission": "not_performed"}
    panels = []
    for key, label, covariance in (
        ("observed_points_m", "Retained observed coordinates", data["uncertainty"]["input_joint_covariance"]),
        ("projected_points_m", "Projected candidate coordinates", data["uncertainty"]["ambient_joint_covariance"]),
        ("reconciled_points_m", "Geometrically eligible coordinates", data["uncertainty"]["ambient_joint_covariance"]),
    ):
        if data[key] is not None:
            panels.append(_panel(key, label, labels, [v for point in data[key] for v in point],
                                 ["m"] * len(labels), covariance, provenance,
                                 coordinate_frame=data["coordinate_frame"], ordering="sample-major:x,y",
                                 interpretation="declared_geometric_policy_only"))
    for key, label in (("radial_residual_before_m", "Original radial residual"),
                       ("correction_norm_m", "Candidate correction magnitude")):
        panels.append(_panel(key, label, ids, data["diagnostics"][key], ["m"] * len(ids), None, provenance,
                             covariance_status="not_propagated_for_this_diagnostic"))
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"],
        "evidence_id": source["evidence_id"], "upstream_bundle_id": record["upstream_bundle_id"],
        "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": "gte", **{k: step[k] for k in
            ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [{"observation_id": identifier, "time_s": observations["time_s"][index],
                              "point_m": observations["points_m"][index]} for index, identifier in enumerate(ids)],
        "raw_declaration": declaration, "verification": native["verification"], "runtimes": native["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
