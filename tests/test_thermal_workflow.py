"""Shared CIW lifecycle coverage for the provider-free thermal operation."""

from copy import deepcopy
import json

import numpy as np
import pytest

from ciw import thermal_contract as contract
from ciw import thermal_reference as reference
from ciw import thermal_workflow
from ciw.instruments import make_demo_run
from ciw.session import Session


def _request():
    return {
        "schema": contract.REQUEST_SCHEMA, "operation_id": contract.OPERATION_ID,
        "model": {"state_order": contract.STATE_ORDER, "state_units": ["K", "K"],
                   "input_order": contract.INPUT_ORDER, "input_units": ["W", "K"],
                   "sensor_order": contract.SENSOR_ORDER, "sensor_units": ["K", "K"],
                   "capacities_j_per_k": [100.0, 200.0], "conductances_w_per_k": [10.0, 5.0],
                   "observation_matrix": [[1, 0], [0, 1]]},
        "sample_interval_s": 1.0,
        "prior": {"mean": [300.0, 300.0], "covariance": [[4.0, 0.2], [0.2, 9.0]]},
        "process_noise_covariance": [[0.1, 0.0], [0.0, 0.2]],
        "observation_noise_covariance": [[0.5, 0.0], [0.0, 0.75]],
        "inputs": [[100.0, 290.0], [80.0, 290.0], [60.0, 290.0]],
        "observations": [[None, None], [None, None], [None, None]],
        "selection": {"costs": [1.0, 2.0], "budget": 2.0, "minimum_sensors": 1,
                       "tie_tolerance_nats": 1e-10, "tie_policy": "lowest_mask_within_tolerance"},
    }


def _source():
    request = _request()
    matrices = reference.model_matrices(request["model"], request["sample_interval_s"])
    ad, bd = np.asarray(matrices["Ad"]), np.asarray(matrices["Bd"])
    state = np.asarray([300.0, 300.0])
    process_noise = [[0.1, 0.0], [0.0, 0.1], [0.0, 0.1]]
    measurement_noise = [[0.5, -0.25], [0.25, 0.0], [-0.25, 0.5]]
    states, observations = [], []
    for inputs, process, noise in zip(request["inputs"], process_noise, measurement_noise):
        state = ad @ state + bd @ np.asarray(inputs) + np.asarray(process)
        observed = state + np.asarray(noise)
        states.append(state.tolist())
        observations.append(observed.tolist())
    request["observations"] = observations
    held_input = [50.0, 290.0]
    held_state = ad @ state + bd @ np.asarray(held_input)
    held_observations = held_state.copy()
    return {"schema": contract.SOURCE_SCHEMA, "experiment_id": "thermal:workflow-fixture",
            "configuration": deepcopy(contract.POLICY), "request": request,
            "evaluation": {"origin": "synthetic_fixture", "generator_model": deepcopy(request["model"]),
                "initial_state": [300.0, 300.0], "process_noise": process_noise,
                "measurement_noise": measurement_noise, "states": states,
                "held_out": {"input": held_input, "process_noise": [0.0, 0.0],
                              "state": held_state.tolist(), "measurement_noise": [0.0, 0.0],
                              "observations": held_observations.tolist()},
                "noise_law": "operator_declared_not_authenticated"}}


def _call(session, kind, payload):
    response = session.handle({"protocol_version": 1, "request_id": "test-" + kind,
                               "type": kind, "payload": payload})
    assert response["type"] == "response", response
    return response["payload"]


def _session_with_source(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = contract.canonical(_source())
    descriptor = _call(session, "source.add", {"kind": "thermal-observer", "label": "thermal fixture",
                                                "bytes_b64": __import__("base64").b64encode(raw).decode()})
    return session, descriptor


def test_thermal_operation_save_reopen_and_replay_preserve_identities(tmp_path):
    session, source = _session_with_source(tmp_path / "original")
    operation = next(item for item in _call(session, "operation.list", {})
                     ["operations"] if item["operation_id"] == "ciw.thermal-observer.v1")
    assert operation["available"] is True
    completed = _call(session, "operation.execute", {"operation_id": operation["operation_id"],
                                                      "parameters": {"source_id": source["source_id"]}})
    original = _call(session, "bundle.get", {"bundle_id": completed["bundle_id"]})
    step = original["steps"][0]
    assert step["operation_id"] == "ciw.thermal-observer.v1"
    assert original["verification"]["authority"]["state_admission"] == "not_performed"
    assert original["runtimes"]["thermal"]["execution_scope"] == "independent_python_reference_only"
    assert step["result"]["data"]["claim_scope"] == contract.CLAIM_SCOPE
    listed = _call(session, "result.list", {})["results"]
    assert any(item["result_id"] == step["result_id"] for item in listed)
    assert _call(session, "result.get", {"result_id": step["result_id"]}) == step["result"]
    view = _call(session, "experiment.inspect", {"bundle_id": completed["bundle_id"]})
    assert view["object_context"]["physical_validation"] == "not_established"
    assert view["panels"][0]["panel_id"] == "state"

    saved = session.save_workspace(tmp_path / "saved.json")
    original_create = thermal_workflow.ThermalWorkflow.create_session
    thermal_workflow.ThermalWorkflow.create_session = lambda *args, **kwargs: pytest.fail("restore executed a provider")
    try:
        reopened = Session.from_workspace(saved, tmp_path / "reopened")
    finally:
        thermal_workflow.ThermalWorkflow.create_session = original_create
    restored = _call(reopened, "bundle.get", {"bundle_id": completed["bundle_id"]})
    assert restored == original

    replay = _call(reopened, "bundle.replay", {"bundle_id": completed["bundle_id"]})
    fresh = _call(reopened, "bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
    fresh_step = fresh["steps"][0]
    assert fresh["bundle_digest"] != original["bundle_digest"]
    assert fresh_step["execution_id"] != step["execution_id"]
    assert fresh_step["result_id"] != step["result_id"]
    assert fresh_step["numerical_result_id"] == step["numerical_result_id"]
    assert replay["replay_receipt"]["admission"] == "not_performed"
    assert replay["replay_receipt"]["numerical_match"] is True


def test_thermal_replay_refuses_a_changed_reference_identity(tmp_path, monkeypatch):
    session, source = _session_with_source(tmp_path)
    completed = _call(session, "operation.execute", {"operation_id": "ciw.thermal-observer.v1",
                                                      "parameters": {"source_id": source["source_id"]}})
    old_identity = thermal_workflow.runtime_identity

    def changed_identity():
        value = deepcopy(old_identity())
        value["algorithm"]["code_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(thermal_workflow, "runtime_identity", changed_identity)
    response = session.handle({"protocol_version": 1, "request_id": "replay",
                               "type": "bundle.replay", "payload": {"bundle_id": completed["bundle_id"]}})
    assert response["type"] == "error"
    assert "runtime identity" in response["payload"]["message"]


def test_thermal_source_tampering_refuses_before_retention(tmp_path):
    session = Session(make_demo_run(), tmp_path)
    raw = contract.canonical(_source())
    encoded = __import__("base64").b64encode(raw).decode()
    source = _call(session, "source.add", {"kind": "thermal-observer", "label": "thermal fixture",
                                             "bytes_b64": encoded})
    assert source["source_id"]
    retained = session.workbench.serialize()
    retained["sources"][0]["bytes_b64"] = __import__("base64").b64encode(raw + b" ").decode()
    with pytest.raises(ValueError):
        session.workbench.restore(retained)


def _reseal_thermal(bundle):
    from ciw.telemetry import _bundle_digest, digest
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({key: value for key, value in result.items() if key != "result_id"})
        step["result_id"] = result["result_id"]
        step["result_sha256"] = digest(result)
        step["numerical_result"] = {"operation_id": thermal_workflow.OPERATION, "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = thermal_workflow._verification(bundle, bundle["verification"]["reproduction"])
    return bundle


def _retained_thermal(tmp_path):
    import base64
    session = Session(make_demo_run(), tmp_path)
    raw = contract.canonical(_source())
    added = session.handle({"protocol_version": 1, "request_id": "add", "type": "source.add", "payload": {
        "kind": "thermal-observer", "label": "thermal fixture", "bytes_b64": base64.b64encode(raw).decode()}})
    assert added["type"] == "response", added
    completed = session.handle({"protocol_version": 1, "request_id": "run", "type": "operation.execute", "payload": {
        "operation_id": "ciw.thermal-observer.v1", "parameters": {"source_id": added["payload"]["source_id"]}}})
    assert completed["type"] == "response", completed
    return session.workbench.get_bundle(completed["payload"]["bundle_id"])


def test_resealing_preserves_an_untouched_thermal_bundle(tmp_path):
    bundle = _retained_thermal(tmp_path)
    resealed = _reseal_thermal(deepcopy(bundle))
    assert resealed == bundle
    thermal_workflow.ThermalWorkflow()._validate(resealed)


@pytest.mark.parametrize("path,value", [
    (("model", "symbolic", "rendering"), "Symbolics/Latexify"),
    (("selection", "solver", "name"), "HiGHS"),
])
def test_resealed_thermal_bundle_cannot_claim_julia_provenance(tmp_path, path, value):
    bundle = _retained_thermal(tmp_path)
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        target = step["result"]["data"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        contract.validate_result(step["request"], step["result"]["data"])  # the contract alone accepts it
    with pytest.raises(ValueError, match="provenance differs from the Python reference"):
        thermal_workflow.ThermalWorkflow()._validate(_reseal_thermal(bundle))
