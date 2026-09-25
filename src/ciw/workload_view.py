"""Read-only native workbench object views; no implicit observation conversion."""
from copy import deepcopy
from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    native = record["native"]
    step, = native["steps"]
    data = step["result"]["data"]
    kind = record["kind"]
    schematic = kind in {"schematic-assessment", "schematic-companions"}
    object_kinds = {"schematic-assessment": "declared_schematic", "schematic-companions": "local_model_analysis",
                    "numerical-heat": "integer_numerical_field", "proved-heat": "proved_integer_numerical_field", "bim-quantity": "construction_quantity",
                    "acquired-dataset": "acquired_evidence", "thermal-observer": "thermal_observer_reference",
                    "machine-manifest": "machine_manifest_reference", "julia-oscillator": "julia_tsit5_trajectory"}
    context = {"object_kind": object_kinds[kind],
               "owner": step["runtime_ref"], "configuration": native["configuration"],
               "covariance_status": "not_applicable", "sensor_fusion": "not_performed",
               "state_admission": "not_performed"}
    panels = []
    provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"],
                  "result_id": step["result_id"], "execution_id": step["execution_id"]}
    if kind == "schematic-assessment":
        context.update(decisions=data["decisions"], neighborhoods=data["neighborhoods"], next_step=data["next_step"])
    elif kind == "schematic-companions":
        context.update(summary="Selected native Jacobian, local structure, covariance and linear Lyapunov sample",
                       events=data["events"], scope=data["scope"], function_id=data["function_id"],
                       covariance_status="declared_input_only_first_order",
                       decisions_before=data["decisions_before"], decisions_after=data["decisions_after"])
    elif kind == "bim-quantity":
        context.update(summary="Native quantity conditioning: " + data["status"], status=data["status"],
                       reason=data["reason"], geometry_authority=data["geometry_authority"],
                       ledger=data["ledger"], ledger_replay=data["ledger_replay"], invariants=data["invariants"],
                       covariance_status="native_declared_quantity_covariance")
        for name in ("prior", "posterior"):
            state = data[name]
            if state is not None:
                labels = [q["quantity"] + " / " + q["global_id"] for q in state["quantities"]]
                panels.append(_panel(name, "CSE " + name + " quantities", labels, state["mean"],
                                     [q["unit"] for q in state["quantities"]], state["covariance"], provenance,
                                     model_frame=declaration["model_frame"], geometry_authority=data["geometry_authority"],
                                     quantities=state["quantities"], status=data["status"]))
    elif kind == "acquired-dataset":
        counts = {key: len(value) for key, value in data["evidence"].items()}
        context.update(summary="Native incremental acquisition with retained checkpoints",
                       evidence_counts=counts, runs=data["runs"], checkpoint=data["checkpoint"],
                       restored_pool_fingerprint=data["restored_pool_fingerprint"], scope=data["scope"])
        panels.append(_panel("evidence-counts", "PPDA retained evidence", list(counts), list(counts.values()),
                             ["count"] * len(counts), None, provenance, temporal_order="not_inferred"))
    elif kind in {"numerical-heat", "proved-heat"}:
        if kind == "proved-heat":
            context.update(proof={k: v for k, v in data["proof"].items() if k != "bytes_b64"},
                           verifier=data["verifier"], timings=data["timings"],
                           verification_trust_scope=native["verification"]["trust_scope"],
                           cryptographic_verification="not_performed_by_inspection",
                           proof_claim="Registered integer heat guest: exact input/output commitments and exit code",
                           proof_policy="required_before_result")
            data = data["native"]
        labels = ["cell " + str(i) for i in range(len(data["values"]))]
        context.update(steps=declaration["steps"], specification_identity=data["specification_identity"],
                       computation_identity=data["computation_identity"], summary="Native integer execution; dimensionless values")
        if kind == "proved-heat":
            context["summary"] = "Retained SP1 heat proof and verifier report; inspection does not reverify"
        provenance = {"source_id": source["source_id"], "evidence_id": source["evidence_id"]}
        panels.append(_panel("initial", "Declared integer field", labels, declaration["initial_values"],
                             ["1"] * len(labels), None, provenance, **context))
        panels.append(_panel("final", "SCR final integer field", labels, data["values"], ["1"] * len(labels), None,
                             {**provenance, "result_id": step["result_id"], "execution_id": step["execution_id"]}, **context))
    elif kind == "thermal-observer":
        observer = data["observer"]
        selection = data["selection"]
        context.update(
            summary="Provider-free linear thermal observer reference",
            physical_validation="not_established",
            uncertainty_scope="conditional_on_declared_linear_model_and_noise",
            selection_status=selection["status"],
            selected_sensor_mask=selection["selected_mask"],
            information_gain_nats=selection["objective_nats"],
            model=deepcopy(data["model"]),
        )
        panels.append(_panel("state", "Thermal posterior state", data["model"]["state_order"],
                             observer["final_mean"], ["K", "K"],
                             observer["final_covariance"],
                             {**provenance, "result_id": step["result_id"], "execution_id": step["execution_id"]},
                             uncertainty_scope=context["uncertainty_scope"]))
    elif kind == "machine-manifest":
        position = data["position"]
        context.update(
            summary="Provider-free read-only encoder position evaluation",
            machine_id=data["machine_id"],
            compiled_manifest_digest=data["compiled_manifest_digest"],
            frame=position["frame"],
            time_basis=position["time_basis"],
            uncertainty_scope="first_order_joint_covariance_declared_prior_or_explicit_evaluation",
            physical_validation="not_performed",
            hardware_actuation="not_performed",
            manifest=deepcopy(data["manifest"]),
        )
        panels.append(_panel("position", "Estimated encoder position", ["position"],
                             [position["position"]], [position["unit"]],
                             [[position["variance"]]],
                             {**provenance, "result_id": step["result_id"], "execution_id": step["execution_id"]},
                             frame=position["frame"], time_basis=position["time_basis"],
                             uncertainty_scope=context["uncertainty_scope"], claim_scope=data["claim_scope"]))
    elif kind == "julia-oscillator":
        context.update(summary="Julia OrdinaryDiffEqTsit5 trajectory against analytic oracle",
                       solver=deepcopy(data["output"]["solver"]),
                       oracle_comparison=deepcopy(data["oracle_comparison"]),
                       claim_scope=data["authority"]["claim_scope"],
                       physical_validation=data["authority"]["physical_validation"],
                       measurement_uncertainty=data["authority"]["measurement_uncertainty"])
        panels.append(_panel("trajectory", "Julia oscillator trajectory", ["q", "v", "energy"],
                             [data["output"]["q_m"][0], data["output"]["v_m_s"][0], data["output"]["energy_j"][0]],
                             ["m", "m/s", "J"], None,
                             {**provenance, "result_id": step["result_id"], "execution_id": step["execution_id"]},
                             claim_scope=data["authority"]["claim_scope"]))
    return deepcopy({"schema": SCHEMA, "catalog_revision": revision, "bundle_id": record["bundle_id"],
        "kind": record["kind"], "label": source["label"], "source_id": source["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": record["upstream_bundle_id"], "replay_source_bundle_ids": [r["source_bundle_digest"] for r in native.get("replay_receipts", [])],
        "experiment_id": declaration["experiment_id"], "fusion_context": None, "object_context": context,
        "panels": panels, "schematic": data["schematic"] if schematic else None,
        "graph": {"nodes": [{"role": step["runtime_ref"], **{k: step[k] for k in ("operation_id", "result_id", "execution_id", "numerical_result_id", "input_refs")}}]},
        "raw_observations": [], "raw_declaration": declaration,
        "verification": native["verification"], "runtimes": native["runtimes"],
        "authority": {"read_only": True, "numerical_replay": "not_performed_by_inspection", "state_admission": "not_performed",
                      **({"cryptographic_verification": "not_performed_by_inspection"} if kind == "proved-heat" else {})}})
