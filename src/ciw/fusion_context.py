"""Fusion contexts of the pinned-set estimation kinds.

A fusion context is the retained GSIE state a pinned-set bundle exposes to
inspection, with its evidence, frames, policies and verification binding. It
is projected from retained records only; nothing here executes a provider or
admits state. Declared kinds expose no fusion context.
"""
from __future__ import annotations

import base64
from copy import deepcopy

from .adapters.subprocess import _json


def _digest(value):
    from .workbench import _digest as digest
    return digest(value)


def _workflow(kind):
    from .workbench import _workflow as workflow
    return workflow(kind)


def project(record, sources):
    """The fusion context a retained pinned-set bundle exposes; inspection only, never admission."""
    native = record["native"]
    if record["kind"] in {"telemetry", "calibrated-window", "acquired-calibrated-window"}:
        return _telemetry_context(record, sources)
    calibrated = record["kind"] == "calibrated-observable"
    step = native["steps"][4 if calibrated else 2]
    state = step["result"]["data"]
    configuration = native["configuration"]
    source = sources[record["source_id"]]
    common = {"context_id": "context:" + _digest({"bundle_id": record["bundle_id"],
                                                   "result_id": step["result_id"]}),
        "bundle_id": record["bundle_id"], "source_id": record["source_id"],
        "evidence_id": source["evidence_id"], "upstream_bundle_id": record["upstream_bundle_id"],
        "state_kind": "posterior" if calibrated else "conditional_prediction",
        "owner": "gsie", "state_id": state["state_id"], "result_id": step["result_id"],
        "execution_id": step["execution_id"], "event_time": state["time"],
        "frame_id": state["frame_id"], "units": state["units"],
        "mean": state["mean"], "covariance": state["covariance"],
        "validation": "content_consistent", "numerical_replay": "not_performed_by_inspection",
        "state_admission": "not_performed",
        "verification_id": native["verification"]["verification_id"],
        "retained_verification_outcome": native["verification"].get("outcome")}
    if calibrated:
        experiment = _json(base64.b64decode(source["bytes_b64"], validate=True))
        observable = native["steps"][3]["result"]["data"]
        reconciled = native["steps"][5]["result"]["data"]
        faults = native["steps"][6]["result"]["data"]
        common.update(state_names=configuration["observability"]["state_names"],
            model_id=configuration["gsie"]["observation_model"]["model_id"],
            dynamics_model_id=configuration["gsie"]["dynamics"]["model_id"],
            clock_frames=[c["clock_model"]["reference_frame"] for c in experiment["channels"]],
            channel_ids=[c["channel_id"] for c in experiment["channels"]],
            claim_scope=experiment["claim_scope"],
            observability={k: observable[k] for k in ("status", "rank", "condition_number", "condition_limit")},
            reconciliation={"result_id": native["steps"][5]["result_id"],
                            "status": reconciled["status"], "reason": reconciled["reason"]},
            fault_assessment={"result_id": native["steps"][6]["result_id"],
                "detection": faults["detection"]["status"], "isolability": faults["isolability"]["status"],
                "isolated_fault": faults["isolability"].get("isolated_fault"),
                "cross_covariance_policy": faults["isolability"]["cross_covariance_policy"]})
    else:
        previous = native["upstream"]
        previous_faults = previous["steps"][6]["result"]["data"]
        common.update(state_names=configuration["identification"]["state_names"],
            model_id=state["model_result_id"],
            declared_model_id=configuration["identification"]["model_id"],
            clock_frame=configuration["identification"]["clock_frame"],
            predecessor_state_id=state["predecessor_state_id"],
            uncertainty_scope=state["uncertainty_scope"],
            parameter_covariance_status=state["parameter_covariance_status"],
            upstream_assessment={"bundle_id": record["upstream_bundle_id"],
                "reconciliation_status": previous["steps"][5]["result"]["data"]["status"],
                "detection_status": previous_faults["detection"]["status"],
                "isolability_status": previous_faults["isolability"]["status"]},
            decision=native["decision"])
    return common


def _telemetry_context(record, sources):
    native = record["native"]
    window = record["kind"] in {"calibrated-window", "acquired-calibrated-window"}
    step = next(s for s in native["steps"] if s["runtime_ref"] == "gsie")
    state = step["result"]["result_artifact"]
    source = sources[record["source_id"]]
    declaration = _json(base64.b64decode(source["bytes_b64"], validate=True))
    mapped = getattr(_workflow(record["kind"]), "mapped_source", None)
    if mapped is not None:
        declaration = mapped(native)
    configuration = native.get("child_window", native)["configuration"]
    cbsr = next((s for s in native["steps"] if s["runtime_ref"] == "cbsr"), None)
    context = {"context_id": "context:" + _digest({"bundle_id": record["bundle_id"], "result_id": step["result_id"]}),
        "bundle_id": record["bundle_id"], "source_id": record["source_id"], "evidence_id": source["evidence_id"],
        "upstream_bundle_id": record["upstream_bundle_id"], "owner": "gsie", "state_kind": "window_feature_posterior",
        "state_id": state["state_id"], "result_id": step["result_id"], "execution_id": step["execution_id"],
        "event_time": state["observation_binding"]["elapsed_seconds"], "epoch": declaration["epoch"],
        "mean": [item["value"] for item in state["components"]], "covariance": state["covariance"]["matrix"],
        "state_names": [item["name"] for item in state["components"]], "units": [item["unit"] for item in state["components"]],
        "frame_id": state["covariance"]["frame"]["id"], "clock_frame": declaration["receipt_clock"]["clock_id"] if window else declaration["clock_basis"],
        "channel_ids": [declaration["channel_id"]], "observation_batch_id": None if window else native["steps"][0]["result_id"],
        "feature_result_id": next(s["result_id"] for s in native["steps"] if s["runtime_ref"] == "stfe"), "window": deepcopy(configuration["window"]),
        "feature_observation_semantics": configuration["gsie"]["feature_observation_semantics"],
        "model_id": configuration["gsie"]["observation_model"]["model_id"],
        "cross_covariance_policy": declaration["joint_covariance"]["cross_covariance_policy"] if window else declaration["crosscov_policy"],
        "prior_measurement_crosscov_policy": configuration["gsie"]["prior_measurement_crosscov_policy"],
        "observability": {"status": "unresolved", "reason": "not_evaluated_by_telemetry_profile"}, "calibration_validity": "not_assessed",
        "reconciliation": {"result_id": cbsr["result_id"], "status": cbsr["result"]["status"]} if cbsr else {"status": "not_run"},
        "fault_assessment": {"status": "not_run"}, "state_admission": "not_performed",
        "validation": "content_consistent", "numerical_replay": "not_performed_by_inspection",
        "verification_id": native["verification"]["verification_id"],
        "retained_verification_outcome": native["verification"].get("outcome")}
    if window:
        context.update(calibration_validity="checked_at_nominal_mapped_event_times",
            clock_result_id=native["steps"][0]["result_id"], calibration_result_id=native["steps"][1]["result_id"],
            time_policy=configuration["composition"]["time_policy"],
            joint_time_value_covariance=deepcopy(native["steps"][1]["result"]["data"]["joint_time_value_covariance"]),
            uncertainty_scope="first_order_conditional_on_declared_nominal_grid",
            calibration_feature_compatibility=deepcopy(native["steps"][1]["result"]["data"]["compatibility"]),
            observability={"status": "unresolved", "reason": "not_evaluated_by_calibrated_window_profile"})
    if record["kind"] == "acquired-calibrated-window":
        context.update(acquisition_binding=deepcopy(native["acquisition_binding"]),
                       native_window_bundle_id=native["child_window"]["bundle_digest"],
                       numerical_verification_id=native["child_window"]["verification"]["verification_id"],
                       reference_prior_policy="explicit_per_window_no_posterior_feedback",
                       inter_window_covariance="not_declared_by_this_window")
    return context
