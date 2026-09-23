"""Read-only views of retained variational inference and model diagnostics."""
from copy import deepcopy

from .experiment_view import SCHEMA, _panel


def project(record, source, declaration, revision):
    bundle = record["native"]
    step, = bundle["steps"]
    data = step["result"]["data"]
    fit, physical = data["fit"], data["physical_posterior"]
    sample = declaration["samples"][declaration["representative_index"]]
    names, units = declaration["coordinates"]["names"], declaration["coordinates"]["units"]
    provenance = {"source_id":source["source_id"], "evidence_id":source["evidence_id"],
                  "result_id":step["result_id"], "execution_id":step["execution_id"]}
    panels = []
    def add(key, title, labels, values, dimensions, covariance=None, **context):
        panels.append(_panel(key,title,labels,values,dimensions,covariance,provenance,**context))
    add("posterior","Variational posterior in declared physical coordinates",names,physical["mean"],units,physical["covariance"],
        uncertainty_scope=physical["uncertainty_scope"], optimizer_status=fit["status"], physical_validation="not_established")
    scale = data["normalization"]["latent_scales"]
    reference = fit["reference"]
    add("reference","Exact Gaussian reference for the assumed model",names,
        [scale[i]*reference["mean"][i] for i in range(2)],units,
        [[scale[i]*scale[j]*reference["covariance"][i][j] for j in range(2)] for i in range(2)],
        method=reference["method"], native_gsie_agreement=data["gsie_agreement"])
    add("truth","Retained simulated truth",names,sample["truth"],units,
        basis="synthetic_reference_not_hardware_measurement")
    trace = fit["trace"]
    add("variational-gap","Gaussian KL gap during optimization",[str(row["iteration"]) for row in trace],
        [row["kl_to_reference"] for row in trace],["1"]*len(trace),
        abscissa="algorithm_iteration_not_physical_time", status=fit["status"],
        free_energy=[row["free_energy"] for row in trace], log_evidence=reference["log_evidence"],
        interpretation="a_small_gap_tests_inference_under_the_assumed_model")
    prediction = data["held_out"]
    sensors = declaration["sensors"]
    obs_scale = data["normalization"]["observation_scales"]
    bias = declaration["assumed_model"]["heldout_bias"]
    add("heldout","Held-out noisy-observation prediction",[s["id"] for s in sensors],
        [prediction["mean"][i]*obs_scale[i]+bias[i] for i in range(2)], [s["unit"] for s in sensors],
        [[prediction["covariance"][i][j]*obs_scale[i]*obs_scale[j] for j in range(2)] for i in range(2)],
        retained_observations=sample["heldout"], prediction_diagnostics=prediction,
        heldout_used_for_inference=False, optimizer_status=fit["status"], physical_validation="not_established")
    metrics = data["ensemble"]["metrics"]
    add("coverage","Exact-reference coverage on the retained synthetic ensemble",
        ["latent_lateral", "latent_heading", "latent_joint", "heldout_lateral", "heldout_heading", "heldout_joint"],
        metrics["latent_marginal_coverage"]+[metrics["latent_joint_coverage"]]+metrics["heldout_marginal_coverage"]+[metrics["heldout_joint_coverage"]],
        ["1"]*6, nominal_probability=metrics["coverage"], replicates=metrics["replicates"],
        basis=data["ensemble"]["basis"], generator_relationship=data["ensemble"]["generator_relationship"],
        latent_marginal_wilson_95=metrics["latent_marginal_wilson_95"], latent_joint_wilson_95=metrics["latent_joint_wilson_95"],
        heldout_marginal_wilson_95=metrics["heldout_marginal_wilson_95"], heldout_joint_wilson_95=metrics["heldout_joint_wilson_95"])
    assessment = data["stages"][2]["result"]["data"]["matrix_assessment"]
    margin = assessment["matrix_margin"]
    if margin is not None:
        add("iteration-margin","PLSR fixed-iteration matrix margin",["matrix_margin"],[margin],["1"],
            assessment=assessment, spectral_radius=fit["stability"]["spectral_radius"],
            mean_error_dynamics=fit["stability"]["iteration_matrix"])
    return deepcopy({"schema":SCHEMA, "catalog_revision":revision, "bundle_id":record["bundle_id"],
        "kind":record["kind"], "label":source["label"], "source_id":source["source_id"], "evidence_id":source["evidence_id"],
        "upstream_bundle_id":None, "replay_source_bundle_ids":[r["source_bundle_digest"] for r in bundle.get("replay_receipts",[])],
        "experiment_id":declaration["experiment_id"], "fusion_context":None,
        "object_context":{"object_kind":record["kind"], "summary":"Variational Free-Energy Sensor Fusion on Curved Surfaces",
            "configuration":declaration["configuration"], "optimizer_status":fit["status"],
            "normalization":data["normalization"], "objective_units":data["objective_units"],
            "ensemble":data["ensemble"], "assumed_geometry":declaration["geometry"],
            "generator":declaration["generator"], "native_gsie_agreement":data["gsie_agreement"],
            "physical_validation":"not_established", "plant_stability":"not_established"},
        "panels":panels, "schematic":None,
        "graph":{"nodes":[{"role":s["runtime_ref"], **{k:s[k] for k in ("operation_id","execution_id","result_id","numerical_result_id","input_refs")}}
                          for s in data["stages"]]},
        "raw_observations":[], "raw_declaration":declaration, "verification":bundle["verification"], "runtimes":bundle["runtimes"],
        "authority":{"read_only":True,"numerical_replay":"not_performed_by_inspection","state_admission":"not_performed"}})
