"""Real CSE conditioning, declared applicability holds and native ledger rollback."""
import base64
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import struct

import pytest

from ciw.bim_quantity import workflow, _check_data
from ciw.telemetry import canonical, byte_digest, digest, _bundle_digest

ROOT = Path(__file__).resolve().parents[1]


def source():
    return runpy.run_path(str(ROOT / "examples/bim-quantity/make_source.py"))["source"]()


def observation_change(declaration, **changes):
    value = deepcopy(declaration)
    observation = json.loads(base64.b64decode(value["observation_bytes_b64"]))
    observation.update(changes)
    value["observation_bytes_b64"] = base64.b64encode(canonical(observation)).decode()
    return value


@pytest.fixture(scope="module")
def repositories():
    path = os.environ.get("CIW_CSE_REPO")
    if not path:
        pytest.skip("set CIW_CSE_REPO for pinned native CSE integration")
    return {"cse": Path(path)}


@pytest.fixture(scope="module")
def retained(repositories):
    raw = b"\n" + canonical(source()) + b"\n "
    original = workflow.create_session(raw, repositories)
    replay = workflow.replay_session(original, repositories)["session"]
    return original, replay


@pytest.mark.parametrize("changes", [
    {"value": True}, {"value": float("inf")}, {"variance": 0}, {"variance": -1},
    {"variance": 1e-30}, {"variance": True}, {"cross_covariance_policy": "assume_independent"},
    {"calibration_assumed": True}, {"ifc_sha256": "unbound"}, {"frame": ""},
])
def test_malformed_observation_cannot_enter_catalog(changes):
    with pytest.raises(ValueError):
        workflow._source(canonical(observation_change(source(), **changes)))


@pytest.mark.parametrize("field,value", [
    ("engine", "/artifact/selected/provider"), ("ifc_bytes_b64", "not-base64"),
    ("model_frame", ""), ("configuration", {}), ("target", {"global_id": "ambiguous"}),
])
def test_malformed_source_cannot_select_execution(field, value):
    with pytest.raises(ValueError):
        workflow._source(canonical(source() | {field: value}))


def test_native_scalar_conditioning_matches_independent_closed_form(retained):
    original, replay = retained
    data = original["steps"][0]["result"]["data"]
    prior, posterior = data["prior"], data["posterior"]
    assert data["status"] == "accepted" and data["reason"] == "conditioned"
    row = next(i for i, q in enumerate(prior["quantities"]) if q["quantity"] == "ClearHeight")
    assert prior["mean"][row] == 3.0
    variance, noise = prior["covariance"][row][row], 0.000025
    expected_mean = 3.0 + variance / (variance + noise) * (2.99 - 3.0)
    expected_variance = variance * noise / (variance + noise)
    assert posterior["mean"][row] == pytest.approx(expected_mean, abs=1e-14)
    assert posterior["covariance"][row][row] == pytest.approx(expected_variance, abs=1e-18)
    volume = next(i for i, q in enumerate(posterior["quantities"]) if q["quantity"] == "Volume")
    assert posterior["mean"][volume] == pytest.approx(5 * 4 * expected_mean)
    assert data["ledger_replay"]["accepted"] == 1 and data["ledger_replay"]["rejected"] == 0
    assert data["ledger"]["events"][1]["operation"]["op"] == "observe_quantity"
    assert data["geometry_authority"] == "QUANTITY_ONLY"
    assert original["verification"]["independent"] is False
    assert original["steps"][0]["numerical_result_id"] == replay["steps"][0]["numerical_result_id"]
    assert original["steps"][0]["result_id"] != replay["steps"][0]["result_id"]
    assert original["steps"][0]["execution_id"] != replay["steps"][0]["execution_id"]
    assert base64.b64decode(original["source"]["evidence"][0]["bytes_b64"]).startswith(b"\n")
    assert base64.b64decode(source()["ifc_bytes_b64"]) == (ROOT / "examples/bim-quantity/room.ifc").read_bytes()


@pytest.mark.parametrize("changes,reason", [
    ({"cross_covariance_policy": "unknown"}, "unknown_cross_covariance"),
    ({"frame": None}, "frame_unresolved"),
    ({"frame": "another-survey-frame"}, "frame_mismatch"),
    ({"ifc_sha256": "sha256:" + "0" * 64}, "ifc_binding_mismatch"),
    ({"quantity": "Volume"}, "target_binding_mismatch"),
    ({"unit": "mm"}, "unit_mismatch"),
])
def test_native_holds_inapplicable_measurement_without_conditioning(repositories, changes, reason):
    bundle = workflow.create_session(canonical(observation_change(source(), **changes)), repositories)
    data = bundle["steps"][0]["result"]["data"]
    assert data["status"] == "held" and data["reason"] == reason
    assert data["prior"] == data["posterior"]
    assert len(data["ledger"]["events"]) == 1
    assert data["ledger_replay"]["accepted"] == data["ledger_replay"]["rejected"] == 0


def test_native_refuses_negative_height_and_replays_rejected_event(repositories):
    bundle = workflow.create_session(canonical(observation_change(source(), value=-1.0, variance=1e-8)), repositories)
    data = bundle["steps"][0]["result"]["data"]
    assert data["status"] == "refused" and data["reason"] == "invariant_rejection"
    assert data["prior"] == data["posterior"]
    event = data["ledger"]["events"][1]
    assert event["kind"] == "rejection" and not event["verification"]["passed"]
    assert data["invariants"]["passed"]
    assert data["ledger_replay"]["accepted"] == 0 and data["ledger_replay"]["rejected"] == 1


@pytest.mark.parametrize("mode,reason", [("derived", "derived_quantity_unsupported"), ("missing", "target_missing"), ("units", "source_units_assumed")])
def test_native_quantity_and_unit_authority_remains_bounded(repositories, mode, reason):
    value = source()
    if mode == "units":
        raw = base64.b64decode(value["ifc_bytes_b64"])
        raw = raw.replace(b",$,#3);", b",$,$);").replace(b"#2=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);\n", b"")
        value["ifc_bytes_b64"] = base64.b64encode(raw).decode()
        value = observation_change(value, ifc_sha256=byte_digest(raw))
    else:
        target = {"ifc_class": "IfcSpace", "global_id": "CIWSPACE00000000000300", "quantity": "Volume" if mode == "derived" else "Missing"}
        value["target"] = target
        value = observation_change(value, **target)
    bundle = workflow.create_session(canonical(value), repositories)
    data = bundle["steps"][0]["result"]["data"]
    assert data["status"] == "held" and data["reason"] == reason
    assert data["prior"] == data["posterior"]


def test_native_parse_refusal_is_retained_without_invented_world(repositories):
    value = source()
    raw = b"unparseable IFC source\n"
    value["ifc_bytes_b64"] = base64.b64encode(raw).decode()
    value = observation_change(value, ifc_sha256=byte_digest(raw))
    data = workflow.create_session(canonical(value), repositories)["steps"][0]["result"]["data"]
    assert data["status"] == "refused" and data["reason"] == "native_refusal"
    assert data["prior"] is None and data["posterior"] is None and data["ledger"] is None


@pytest.mark.parametrize("fault", ["posterior", "raw-belief", "operation", "ledger", "geometry", "observation", "report"])
def test_retained_native_content_tampering_is_refused(retained, fault):
    bundle = deepcopy(retained[0])
    declaration = workflow._source(base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"]))
    data = bundle["steps"][0]["result"]["data"]
    if fault == "posterior": data["posterior"]["mean"][0] += 0.1
    if fault == "raw-belief": data["prior"]["belief_digest"] = "0" * 64
    if fault == "operation": data["ledger"]["events"][1]["operation"]["measurements"][0]["value"] = 2.5
    if fault == "ledger": data["ledger"]["events"].reverse()
    if fault == "geometry": data["geometry_authority"] = "SWEPT_SOLID"
    if fault == "observation": data["observation_sha256"] = "sha256:" + "0" * 64
    if fault == "report": data["invariants"]["passed"] = False
    with pytest.raises(ValueError):
        _check_data(declaration, data)


def test_replay_runtime_pin_drift_is_refused(retained):
    bundle = deepcopy(retained[0])
    bundle["runtimes"]["cse"]["revision"] = "0" * 40
    bundle["bundle_digest"] = _bundle_digest(bundle)
    with pytest.raises(ValueError, match="runtime pin"):
        workflow._validate(bundle)


def test_shared_inspection_and_restore_do_not_execute_or_admit_state(retained, monkeypatch):
    from ciw.workbench import Workbench
    bench = Workbench()
    original, replay = retained
    descriptor = bench.add_source({"kind": "bim-quantity", "label": "room quantities",
                                  "bytes_b64": original["source"]["evidence"][0]["bytes_b64"]})
    retained_source = bench.get_source(descriptor["source_id"])
    for bundle in (original, replay):
        bench._retain("bim-quantity", retained_source, None, bundle)
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only operations must not bind or execute CSE")
    monkeypatch.setattr(workflow, "_adapters", forbidden)
    before = bench.serialize()
    view = bench.inspect_experiment({"bundle_id": original["bundle_digest"]})
    assert view["fusion_context"] is None
    assert view["object_context"]["object_kind"] == "construction_quantity"
    assert view["object_context"]["state_admission"] == "not_performed"
    assert view["panels"][1]["values"][0] == pytest.approx(2.992)
    assert bench.fusion_contexts() == []
    assert Workbench.restore(before).serialize() == before
    assert not next(row for row in Workbench.restore(before).describe_operations()
                    if row["operation_id"] == "ciw.bim-quantity.v1")["available"]
    view["object_context"]["status"] = "forged"
    assert bench.serialize() == before


@pytest.mark.parametrize("coordinates", [(0, 1), (3, 4)], ids=["raw-block", "derived-block"])
def test_fully_resealed_held_indefinite_covariance_is_refused(repositories, coordinates, monkeypatch):
    from ciw import bim_quantity
    from ciw.declared_workload import _verification

    bundle = workflow.create_session(canonical(observation_change(source(), cross_covariance_policy="unknown")), repositories)
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        data = step["result"]["data"]
        i, j = coordinates
        module = data["ledger"]["events"][0]["operation"]["module_digest"]
        for state in (data["prior"], data["posterior"]):
            state["covariance"][i][j] = state["covariance"][j][i] = 1.0
            packed = lambda values: b"".join(struct.pack("<d", float(v)) for v in values)
            rows = [k for k, q in enumerate(state["quantities"]) if q["role"] == "raw"]
            state["belief_digest"] = sha256(packed(state["mean"][k] for k in rows) +
                packed(state["covariance"][k][l] for k in rows for l in rows)).hexdigest()
            state["world_digest"] = sha256(module.encode() + packed(state["mean"]) +
                packed(v for row in state["covariance"] for v in row)).hexdigest()
        genesis = data["ledger"]["events"][0]
        genesis["operation"]["belief_digest"] = data["prior"]["belief_digest"]
        genesis["prior_world_digest"] = genesis["result_world_digest"] = data["prior"]["world_digest"]
        genesis["event_hash"] = sha256(canonical({k: v for k, v in genesis.items() if k != "event_hash"})).hexdigest()
        data["ledger"]["integrity"]["head"] = genesis["event_hash"]
        data["ledger_replay"].update(head=genesis["event_hash"], world_digest=data["prior"]["world_digest"])
        result = step["result"]
        result["result_id"] = digest({k: v for k, v in result.items() if k != "result_id"})
        step.update(result_id=result["result_id"], result_sha256=digest(result),
                    numerical_result={"operation_id": workflow.operation, "data": data})
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = _verification(bundle, bundle["verification"]["reproduction"])

    # Prove this fault is fully resealed and would satisfy every previous
    # identity, unchanged-state and invariant-report consistency check.
    with monkeypatch.context() as context:
        context.setattr(bim_quantity, "_validate_matrix", lambda *args: None)
        workflow._validate(bundle)
    with pytest.raises(ValueError, match="positive semidefinite"):
        workflow._validate(bundle)
