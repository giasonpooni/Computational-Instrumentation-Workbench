"""Read-only panels for mathematical profiles; no geometry provider is executed."""
from copy import deepcopy
from fractions import Fraction

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    bundle = record["native"]
    step, = bundle["steps"]
    data, request = step["result"]["data"], declaration["request"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "result_id": step["result_id"], "execution_id": step["execution_id"]}
    context = {"object_kind": record["kind"], "owner": step["runtime_ref"],
        "configuration": declaration["configuration"], "claim_scope": data["claim_scope"],
        "native_artifact_digest": data["artifact_digest"], "sensor_fusion": "not_performed",
        "state_admission": "not_performed", "physical_calibration": "not_established"}
    panels = []
    if record["kind"] == "covariance-geometry":
        coordinates = request["coordinates"]
        context.update(summary="Declared SPD matrices under the affine-invariant metric", frame=request["frame"],
            coordinates=coordinates, input_evidence=data["input_evidence"], invariants=data["evidence"],
            covariance_interpretation="declared_geometry_not_an_estimated_state")
        panels.append(_panel("metric-distance", "Affine-invariant matrix distance", ["distance"], [data["distance"]],
            ["1"], None, provenance, metric=data["metric"], uncertainty="not_estimated"))
        for index, sample in enumerate(data["samples"]):
            labels = [a["name"] + "," + b["name"] for a in coordinates for b in coordinates]
            units = [a["unit"] + "*" + b["unit"] for a in coordinates for b in coordinates]
            values = [v for row in sample["covariance"] for v in row]
            panels.append(_panel("matrix-" + str(index), "Covariance matrix at t=" + str(sample["parameter"]),
                labels, values, units, None, provenance, parameter=sample["parameter"],
                matrix=sample["covariance"], spectrum=sample["spectrum"],
                interpretation="matrix_entries; uncertainty_of_entries_not_estimated"))
    elif record["kind"] == "mesh-path":
        solution, mesh = data["solution"], request["mesh"]
        context.update(summary="Edge-constrained mesh path baseline", mesh_quality=data["mesh_quality"],
            mesh=mesh, bounds=data["bounds"], path=solution["target_path"], reachable=solution["reachable"],
            tolerance_budget=data["tolerance_budget"],
            continuous_surface_shortest_path="not_computed",
            unreachable_vertices=[i for i,v in enumerate(solution["vertex_distances"]) if v is None])
        selected = [(i,v) for i,v in enumerate(solution["vertex_distances"]) if v is not None]
        panels.append(_panel("vertex-distances", "Shortest distances along mesh edges", ["vertex-" + str(i) for i,v in selected],
            [v for i,v in selected], [mesh["units"]] * len(selected), None, provenance,
            metric="edge_length_graph", uncertainty="not_estimated"))
        if solution["reachable"]:
            panels.append(_panel("target-bounds", "Target path and Euclidean lower bound",
                ["edge_path", "euclidean_lower_bound"], [solution["target_distance"], data["bounds"]["target_lower_bound"]],
                [mesh["units"]] * 2, None, provenance, roundoff_certification="not_established"))
    else:
        context.update(summary="Exact rational flow on declared square-tiled surface", status=data["status"],
            gluing_validation=data["gluing_validation"], gluing=request["gluing"],
            arithmetic=data["arithmetic"], invariants=data["invariants"],
            exact_segments=data["segments"], exact_events=data["events"], final_state=data["final_state"],
            elapsed=data["elapsed"], remaining=data["remaining"], display_arithmetic="binary64_projection_of_exact_rationals")
        panels.append(_panel("duration", "Trajectory parameter accounting", ["elapsed", "remaining"],
            [float(Fraction(data["elapsed"])), float(Fraction(data["remaining"]))], ["flow_parameter"] * 2,
            None, provenance, status=data["status"], observation_time="not_applicable"))
        panels.append(_panel("crossings", "Directed edge crossings", ["events"], [len(data["events"])],
            ["count"], None, provenance, status=data["status"], vertex_continuation="not_inferred"))
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": None, "replay_source_bundle_ids": [r["source_bundle_digest"] for r in bundle.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k:step[k] for k in
            ("operation_id", "execution_id", "result_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration, "verification": bundle["verification"], "runtimes": bundle["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
