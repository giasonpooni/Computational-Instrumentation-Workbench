"""Read-only native reference projections; arclength never becomes event time."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    bundle = record["native"]
    step, = bundle["steps"]
    data = step["result"]["data"]
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    context = {"object_kind": record["kind"], "owner": step["runtime_ref"], "configuration": declaration["configuration"],
               "sensor_fusion": "not_performed", "state_admission": "not_performed", "physical_geometry": "not_established"}
    panels = []
    if record["kind"] == "flat-torus-reference":
        reference, trajectory = data["reference"], data["trajectory"]
        geometry = reference["geometry"]
        context.update(summary="Native area-one flat quotient reference", geometry=geometry,
                       geometry_digest=reference["geometry_digest"], limitations=reference["limitations"],
                       quotient_crossings=trajectory["crossings"], covariance_status="not_applicable")
        panels.append(_panel("loop-length", "Flat quotient loop length", ["loop_length"], [geometry["length"]],
            ["normalized_length"], None, provenance, winding=geometry["winding"], closed=geometry["closed"],
            covariance_status="not_applicable", geometry_basis="area_one_flat_quotient"))
        for field, title in (("cover_points", "Lifted path coordinates"), ("parallelogram_points", "Wrapped quotient coordinates")):
            points = trajectory[field]
            labels = [f"sample-{i}.{axis}" for i in range(len(points)) for axis in ("x", "y")]
            values = [point[axis] for point in points for axis in ("re", "im")]
            panels.append(_panel(field, title, labels, values, ["normalized_length"] * len(values), None, provenance,
                path_parameter=trajectory["times"], coordinate_representation=field, covariance_status="not_applicable",
                topology="identified_flat_parallelogram", physical_embedding="not_established"))
    else:
        transfer = data["record"]
        length_unit = transfer["units"]["length"]
        grid = transfer["arclength"]
        context.update(summary="Native constant-curvature Jacobi transfer", frame_id=transfer["frame"],
            arclength=grid, units=transfer["units"], curvature=transfer["gaussian_curvature"], validity=transfer["validity"],
            convergence=transfer["resolution"]["convergence"], calibration=transfer["calibration"],
            covariance_scope="conditional_marginal_pose_covariance_at_each_arclength",
            propagated_covariance=data["propagated_covariance"], geometry=None)
        panels.append(_panel("starting-pose", "Declared initial perturbation and covariance", ["lateral", "heading"],
            declaration["initial_perturbation"], [length_unit, "radian"], transfer["covariance"]["matrix"], provenance,
            frame_id=transfer["frame"], arclength=grid[0], covariance_basis=transfer["covariance"]["basis"],
            interpretation="assumed_reference_distribution"))
        panels.append(_panel("endpoint-pose", "Native propagated endpoint perturbation", ["lateral", "heading"],
            [data["separation"][-1], data["heading_change"][-1]], [length_unit, "radian"], data["propagated_covariance"][-1], provenance,
            frame_id=transfer["frame"], arclength=grid[-1], covariance_basis=transfer["covariance"]["basis"],
            validity=transfer["validity"], interpretation="first_order_reference_not_observed_state"))
        labels = ["s=" + str(s) for s in grid]
        for name, title, values, unit in (
            ("separation", "Native transverse separation", data["separation"], length_unit),
            ("heading-change", "Native heading change", data["heading_change"], "radian"),
            ("transfer-determinant", "Native transfer determinant", data["determinant"], "1"),
        ):
            panels.append(_panel(name, title, labels, values, [unit] * len(grid), None, provenance,
                frame_id=transfer["frame"], arclength=grid, axis_quantity="arclength",
                covariance_status="joint_cross_arclength_covariance_not_supplied"))
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": None, "replay_source_bundle_ids": [r["source_bundle_digest"] for r in bundle.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k: step[k] for k in
            ("operation_id", "execution_id", "result_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration, "verification": bundle["verification"], "runtimes": bundle["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
