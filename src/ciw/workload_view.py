"""Read-only native workbench object views; no implicit observation conversion."""
from copy import deepcopy
from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    schematic = record["kind"] == "schematic-assessment"
    context = {"object_kind": "declared_schematic" if schematic else "integer_numerical_field",
               "owner": step["runtime_ref"], "configuration": native["configuration"],
               "covariance_status": "not_applicable", "sensor_fusion": "not_performed",
               "state_admission": "not_performed"}
    panels = []
    if schematic:
        context.update(decisions=data["decisions"], neighborhoods=data["neighborhoods"], next_step=data["next_step"])
    else:
        labels = ["cell " + str(i) for i in range(len(data["values"]))]
        context.update(steps=declaration["steps"], specification_identity=data["specification_identity"],
                       computation_identity=data["computation_identity"])
        provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"]}
        panels.append(_panel("initial", "Declared integer field", labels, declaration["initial_values"],
                             ["1"] * len(labels), None, provenance, **context))
        panels.append(_panel("final", "SCR final integer field", labels, data["values"], ["1"] * len(labels), None,
                             {**provenance, "result_id": step["result_id"], "execution_id": step["execution_id"]}, **context))
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": None, "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": data["schematic"] if schematic else None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k: step[k] for k in ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration,
        "verification": native["verification"], "runtimes": native["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed"}})
