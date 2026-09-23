"""Retained-log analysis, scoped replay, offline restore and display contracts."""
import asyncio
import base64
from copy import deepcopy
import json
import subprocess

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from ciw import energy_records, energy_workflow, free_energy_math
from ciw.energy_workflow import EnergyAccuracyWorkflow, OPERATION, KIND
from ciw.instruments import make_demo_run
from ciw.server import WorkbenchServer
from ciw.session import Session
from ciw.telemetry import canonical, digest, byte_digest, _bundle_digest
from ciw.workbench import Workbench
from test_energy_records import make_log, reseal
from native_operations import PROVIDER_FREE_OPERATIONS


def source_payload(raw):
    return {"kind": KIND, "label": "Synthetic energy-log analysis fixture",
            "bytes_b64": base64.b64encode(raw).decode("ascii")}


@pytest.fixture(scope="module")
def retained():
    raw = canonical(make_log())
    workflow = EnergyAccuracyWorkflow()
    bundle = workflow.create_session(raw, {})
    return raw, bundle


def reseal_bundle(bundle):
    """Recompute envelopes so mutation tests exercise semantics, not stale hashes."""
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        result = step["result"]
        result["result_id"] = digest({k:v for k,v in result.items() if k != "result_id"})
        step["result_id"], step["result_sha256"] = result["result_id"], digest(result)
        step["request_sha256"] = digest(step["request"])
        step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(result["data"])}
        step["numerical_result_id"] = digest(step["numerical_result"])
    bundle["bundle_digest"] = _bundle_digest(bundle)
    verification = bundle["verification"]
    verification["subject_ref"] = bundle["bundle_digest"]
    verification["runtime_digest"] = digest(bundle["runtimes"])
    verification["verification_id"] = byte_digest(verification["schema"].encode() + b"\0" +
                                                  canonical({k:v for k,v in verification.items() if k != "verification_id"}))
    return bundle


def test_builtin_is_available_without_hardware_or_repository_bindings():
    workbench = Workbench()
    assert {row["operation_id"] for row in workbench.describe_operations() if row["available"]} == PROVIDER_FREE_OPERATIONS
    operation = next(row for row in workbench.describe_operations() if row["operation_id"] == OPERATION)
    assert operation["role"] == "offline_energy_accuracy_analysis"
    workbench.bind_workflow(KIND, {})
    with pytest.raises(ValueError):
        workbench.bind_workflow(KIND, {"energy": "not-a-provider"})


def test_exact_log_bytes_retained_once_and_not_embedded_in_request(retained):
    raw, bundle = retained
    assert EnergyAccuracyWorkflow()._validate(bundle) == raw
    assert bundle["source"]["experiment_id"] == json.loads(raw)["run_id"]
    assert bundle["configuration"]["physical_measurement"] == "not_performed"
    for step in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        assert set(step["request"]) == {"schema", "evidence_id", "log_digest", "measurement_run_id", "analysis_profile"}
        assert step["request"]["evidence_id"] == byte_digest(raw)
        assert step["result"]["authority"]["physical_measurement"] == "not_performed_by_analysis"
    assert canonical(bundle).count(base64.b64encode(raw)) == 1
    assert bundle["verification"]["method"] == "fresh_analysis_of_same_retained_measurement"
    assert bundle["verification"]["independent"] is False


def test_serialization_changes_evidence_identity_without_inventing_new_measurement(retained):
    raw, original = retained
    pretty = json.dumps(json.loads(raw), indent=2).encode()
    other = EnergyAccuracyWorkflow().create_session(pretty, {})
    assert original["source"]["evidence"][0]["artifact_ref"] != other["source"]["evidence"][0]["artifact_ref"]
    assert original["source"]["experiment_id"] == other["source"]["experiment_id"]
    assert original["steps"][0]["numerical_result_id"] == other["steps"][0]["numerical_result_id"]


def test_replay_creates_analysis_occurrences_and_preserves_original_measurement(retained):
    raw, original = retained
    workflow = EnergyAccuracyWorkflow()
    replayed = workflow.replay_session(original, {})
    replay = replayed["session"]
    assert workflow._validate(replay) == raw
    assert replay["source"] == original["source"]
    assert replay["runtimes"] == original["runtimes"]
    assert replay["bundle_digest"] != original["bundle_digest"]
    assert replay["steps"][0]["numerical_result_id"] == original["steps"][0]["numerical_result_id"]
    occurrences = {b["steps"][0]["execution_id"] for b in (original, replay)}
    occurrences |= {b["verification"]["reproduction"]["execution_id"] for b in (original, replay)}
    assert len(occurrences) == 4
    assert replayed["replay_receipt"]["verification"]["method"] == energy_workflow.METHOD


@pytest.mark.parametrize("change", [
    lambda step: step["request"].update(log_digest="sha256:"+"0"*64),
    lambda step: step["request"].update(raw_log={"pretended": "duplication"}),
    lambda step: step["result"]["authority"].update(physical_measurement="performed"),
    lambda step: step["result"]["data"]["measurement"].update(gross_energy_j=99),
    lambda step: step["result"]["data"]["measurement"].update(amortized_domain_energy_j_per_qualified_solve=99),
    lambda step: step["result"]["data"]["comparison"].update(eligible=True),
    lambda step: step["result"]["data"].update(origin="physical_measurement"),
    lambda step: step["result"]["data"]["reference"]["mean"].__setitem__(0, 99),
])
def test_resealed_result_and_request_tampering_is_refused(retained, change):
    _, bundle = retained
    changed = deepcopy(bundle)
    for step in (changed["steps"][0], changed["verification"]["reproduction"]):
        change(step)
    reseal_bundle(changed)
    with pytest.raises(ValueError):
        EnergyAccuracyWorkflow()._validate(changed)


@pytest.mark.parametrize("change", [
    lambda b: b["verification"].update(method="fresh_physical_measurement"),
    lambda b: b["verification"].update(independent=True),
    lambda b: b["verification"].update(reproduction=deepcopy(b["steps"][0])),
    lambda b: b["configuration"].update(physical_measurement="performed"),
    lambda b: b["runtimes"]["energy"].update(execution_scope="gpu_measurement"),
    lambda b: b["runtimes"]["energy"].update(device_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef"),
])
def test_resealed_authority_runtime_and_occurrence_tampering_is_refused(retained, change):
    changed = deepcopy(retained[1])
    change(changed)
    reseal_bundle(changed)
    with pytest.raises(ValueError):
        EnergyAccuracyWorkflow()._validate(changed)


def test_historical_runtime_can_be_inspected_but_drift_refuses_fresh_analysis(retained):
    changed = deepcopy(retained[1])
    changed["runtimes"]["energy"]["code_sha256"] = "a"*64
    reseal_bundle(changed)
    workflow = EnergyAccuracyWorkflow()
    assert workflow._validate(changed) == retained[0]
    with pytest.raises(ValueError):
        workflow.replay_session(changed, {})


def test_bad_or_oversized_log_is_refused_before_local_analysis(monkeypatch):
    monkeypatch.setattr(energy_records, "analyze", lambda *_: pytest.fail("Invalid source reached analysis"))
    workflow = EnergyAccuracyWorkflow()
    for raw in (b"", b"{}", b" " * (energy_workflow.SOURCE_LIMIT + 1)):
        with pytest.raises(ValueError):
            workflow.create_session(raw, {})
    log = make_log()
    log["phases"][3]["samples"][-1]["energy_mj"] = "999999"
    with pytest.raises(ValueError):
        workflow.create_session(canonical(log), {})


def test_reanalysis_restore_and_projection_never_execute_hardware_or_variational_solver(retained, monkeypatch):
    from ciw import energy_cuda, energy_nvml
    def refuse(*_args, **_kwargs):
        pytest.fail("Offline energy analysis executed hardware, a process or variational iterations")
    monkeypatch.setattr(energy_cuda, "CudaGaussianWorker", refuse)
    monkeypatch.setattr(energy_nvml, "NVMLEnergyCounter", refuse)
    monkeypatch.setattr(free_energy_math, "variational_fit", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    w = Workbench()
    source = w.add_source(source_payload(retained[0]))
    original = w.execute({"operation_id": OPERATION, "source_id": source["source_id"]})
    replay = w.replay({"bundle_id": original["bundle_id"]})["bundle"]
    restored = Workbench.restore(w.serialize())
    assert restored.serialize() == w.serialize()
    before = restored.serialize()
    # Projection copies already retained data; even pure analysis is unnecessary.
    monkeypatch.setattr(energy_records, "analyze", refuse)
    for bundle in (original, replay):
        view = restored.inspect_experiment({"bundle_id": bundle["bundle_id"]})
        assert view["object_context"]["fresh_hardware_measurement"] is False
        assert "synthetic fixture" in view["object_context"]["summary"]
        assert view["fusion_context"] is None
        assert len(view["panels"]) == 4
        assert [p["units"][0] for p in view["panels"]] == ["J", "ms", "nat", "J/solve"]
        assert all(p["covariance"] is None for p in view["panels"])
    assert restored.serialize() == before
    assert restored.fusion_contexts() == []
    assert {r["instrument"] for r in restored.instrument_views()} == {"energy"}


def test_missing_counter_evidence_is_not_drawn_as_zero():
    log = make_log()
    log["phases"][3]["samples"] = []
    w = Workbench()
    source = w.add_source(source_payload(canonical(reseal(log))))
    result = w.execute({"operation_id": OPERATION, "source_id": source["source_id"]})
    view = w.inspect_experiment({"bundle_id": result["bundle_id"]})
    panels = {p["panel_id"]: p for p in view["panels"]}
    assert "measurement" not in panels["gross-energy"]["labels"]
    assert "missing_endpoint_brackets" in panels["gross-energy"]["context"]["unavailable_phases"]["measurement"]
    assert panels["energy-per-solve"]["values"] == []
    assert panels["energy-per-solve"]["context"]["available"] is False


def test_same_measurement_occurrence_cannot_be_rebound_to_a_different_log(retained):
    w = Workbench()
    source = w.add_source(source_payload(retained[0]))
    w.execute({"operation_id": OPERATION, "source_id": source["source_id"]})
    log = json.loads(retained[0])
    log["phases"][3]["samples"][-1]["energy_mj"] = "9999"
    other = w.add_source(source_payload(canonical(reseal(log))))
    with pytest.raises(ValueError, match="Identity collision"):
        w.execute({"operation_id": OPERATION, "source_id": other["source_id"]})
    assert w.pending_operations == 0
    assert len(w.serialize()["bundles"]) == 1


def test_shared_session_live_transport_save_restore_and_reanalysis(retained, tmp_path):
    session = Session(make_demo_run(), tmp_path / "session")
    async def exercise():
        async with serve(WorkbenchServer(session).handler, "127.0.0.1", 0) as listener:
            url = "ws://127.0.0.1:" + str(listener.sockets[0].getsockname()[1])
            async with connect(url, max_size=16*1024*1024, proxy=None) as socket:
                assert json.loads(await socket.recv())["type"] == "session.snapshot"
                async def call(kind, payload):
                    await socket.send(json.dumps({"protocol_version": 1, "request_id": kind, "type": kind, "payload": payload}))
                    while True:
                        answer = json.loads(await asyncio.wait_for(socket.recv(), 5))
                        if answer.get("request_id") == kind:
                            assert answer["type"] == "response", answer
                            return answer["payload"]
                source = await call("source.add", source_payload(retained[0]))
                result = await call("operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source["source_id"]}})
                replay = await call("bundle.replay", {"bundle_id": result["bundle_id"]})
                view = await call("experiment.inspect", {"bundle_id": replay["bundle"]["bundle_id"]})
                assert view["object_context"]["replay_scope"] == energy_workflow.METHOD
                return result
    result = asyncio.run(exercise())
    saved = session.save_workspace(tmp_path / "workspace.json")
    restored = Session.from_workspace(saved, tmp_path / "restored")
    assert restored.workbench.serialize() == session.workbench.serialize()
    assert {o["operation_id"] for o in restored.workbench.describe_operations() if o["available"]} == PROVIDER_FREE_OPERATIONS
    restored.workbench.replay({"bundle_id": result["bundle_id"]})
    assert len(restored.workbench.serialize()["bundles"]) == 3
    assert restored.workbench.pending_operations == 0
