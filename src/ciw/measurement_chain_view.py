"""Read-only native measurement-chain projection for the shared desktop."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel
from .measurement_chain import catalog_steps


def project(record, source, declaration, revision):
    bundle = record["native"]
    outer = bundle["steps"][0]
    workspace = outer["result"]["data"]["native_workspace"]
    run = workspace["run"]
    sensors = run["metadata"]["rci_source"]["sensors"]
    fsrt, jspt = workspace["results"]
    common = {"source_id": source["source_id"], "evidence_id": source["evidence_id"]}
    panels = []
    for sensor in sensors:
        native = sensor["measurement"]
        raw = native["records"][0]
        panels.append(_panel("raw-" + sensor["name"], sensor["name"] + " raw indication", [sensor["name"]],
            [raw["raw"]["value"]], [raw["raw"]["unit"]], sensor["request"]["inputs"]["raw_covariance"], common,
            observed_at=raw["observed_at"], calibration_state="not_applied", raw_record_bytes=sensor["request"]["inputs"]["records"][0]["raw_record_b64"]))
        panels.append(_panel(
                "calibrated-" + sensor["name"], sensor["name"] + " calibrated measurement", [sensor["name"]],
                [raw["calibrated"]["value"]], [raw["calibrated"]["unit"]], native["uncertainty"]["output_covariance"], common,
                covariance_artifact=sensor["covariance"], observed_at=raw["observed_at"], calibration=native["calibration"], uncertainty=native["uncertainty"]))
    for name in ("prior", "posterior", "reconciled", "innovation"):
        artifact = fsrt["data"]["covariance_artifacts"][name]
        title = {"prior": "FSRT declared prior", "posterior": "FSRT unprojected estimate",
                 "reconciled": "FSRT retained estimate", "innovation": "FSRT prior innovation"}[name]
        panels.append(_panel(name, title, artifact["quantity_ids"], artifact["reference_values"], artifact["units"], artifact["matrix"],
            {**common, "result_id": fsrt["result_id"], "execution_id": fsrt["execution_id"]}, covariance_artifact=artifact,
            reconciliation_status=fsrt["data"]["diagnostics"]["reconciliation_status"], native_verification_status=fsrt["verification_status"]))
    artifact = jspt["data"]["output_covariance"]
    panels.append(_panel("propagated", "JSPT declared output reference and covariance", artifact["quantity_ids"], artifact["reference_values"],
        artifact["units"], artifact["matrix"], {**common, "result_id": jspt["result_id"], "execution_id": jspt["execution_id"]},
        covariance_artifact=artifact, jacobian=jspt["data"]["jacobian"], jacobian_source="caller_declared",
        reference_scope="caller_declared_not_a_new_measurement", native_verification_status=jspt["verification_status"]))
    nodes = [{"role": step["runtime_ref"], "operation_id": step["operation_id"], "execution_id": step["execution_id"],
              "result_id": step["result_id"], "input_refs": step["input_refs"]} for step in catalog_steps(bundle)]
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"], "kind": record["kind"],
        "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"], "upstream_bundle_id": None,
        "replay_source_bundle_ids": [r["source_bundle_digest"] for r in bundle.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None,
        "object_context": {"object_kind": "measurement-chain-testbed", "summary": "RCI calibration, FSRT snapshot and JSPT declared covariance map",
            "scope": declaration["configuration"], "diagnostics": fsrt["data"]["diagnostics"], "native_run_id": run["run_id"],
            "native_evidence_id": run["evidence_id"], "native_result_ids": [r["result_id"] for r in workspace["results"]]},
        "panels": panels, "graph": {"nodes": nodes}, "raw_observations": declaration["investigation"]["sensors"],
        "verification": bundle["verification"], "runtimes": bundle["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
