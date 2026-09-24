"""Whole native experiments: correct inference, incorrect models, and replay."""
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path

import pytest

from ciw import free_energy_math as mathematics, free_energy_native as native
from ciw.free_energy_profile import problems
from ciw.free_energy_workflow import FreeEnergyWorkflow, native_occurrences
from ciw.telemetry import canonical

ROOT = Path(__file__).resolve().parents[1]
CASES = ("baseline", "correlated-noise", "ignored-correlation", "sensor-bias", "wrong-curvature", "unstable-step")


@pytest.fixture(scope="session")
def native_experiments():
    return _native_experiments()


@lru_cache(maxsize=1)
def _native_experiments():
    root = os.environ.get("CIW_FREE_ENERGY_STACK_ROOT")
    if not root:
        pytest.skip("Native free-energy gate requires pinned CSG, GSIE and PLSR")
    bindings = {role:Path(root) / role for role in ("csg", "gsie", "plsr")}
    workflow = FreeEnergyWorkflow()
    cases = {}
    for name in CASES:
        raw = (ROOT / "examples/variational-free-energy" / (name + ".json")).read_bytes()
        cases[name] = {"raw":raw, "source":json.loads(raw), "bundle":workflow.create_session(raw, bindings)}
    replay = workflow.replay_session(cases["baseline"]["bundle"], bindings)["session"]
    destination = os.environ.get("CIW_FREE_ENERGY_FIXTURE_DIR")
    if destination:
        for name, case in cases.items():
            folder = Path(destination) / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "source.json").write_bytes(case["raw"])
            (folder / "original.json").write_bytes(canonical(case["bundle"]))
        (Path(destination) / "baseline" / "replay.json").write_bytes(canonical(replay))
    return {"workflow":workflow,"bindings":bindings,"cases":cases,"replay":replay}


def data(experiments, name):
    return experiments["cases"][name]["bundle"]["steps"][0]["result"]["data"]


def test_all_cases_expose_inference_and_model_adequacy_separately(native_experiments):
    for name in CASES[:-1]:
        result = data(native_experiments,name)
        assert result["fit"]["status"] == "converged"
        assert result["objective_units"]["kl_gap"] < 1e-12
        assert result["gsie_agreement"]["passed"] is True
        assert result["stages"][2]["result"]["data"]["matrix_assessment"]["numerically_negative_definite"] is True
    baseline = data(native_experiments,"baseline")["ensemble"]["metrics"]
    assert baseline["latent_joint_count"] == 122
    assert baseline["heldout_joint_count"] == 121
    for name in ("sensor-bias","wrong-curvature"):
        failed = data(native_experiments,name)["ensemble"]["metrics"]
        assert failed["heldout_joint_coverage"] < .4
        assert failed["latent_joint_coverage"] < .4
    unstable = data(native_experiments,"unstable-step")
    assert unstable["fit"]["status"] == "iteration_limit"
    assert unstable["fit"]["stability"]["spectral_radius"] > 1
    assert unstable["objective_units"]["kl_gap"] > 1
    assert unstable["stages"][2]["result"]["data"]["matrix_assessment"]["numerically_negative_definite"] is False
    assert unstable["ensemble"]["basis"] == "exact_reference_posterior_on_retained_ensemble"


def test_paired_controls_preserve_samples_and_correlations(native_experiments):
    cases = native_experiments["cases"]
    for name in ("wrong-curvature","unstable-step"):
        assert cases[name]["source"]["samples"] == cases["baseline"]["source"]["samples"]
    assert cases["correlated-noise"]["source"]["samples"] == cases["ignored-correlation"]["source"]["samples"]
    assert data(native_experiments,"correlated-noise")["problem"]["noise_covariance"][0][1] != 0
    assert data(native_experiments,"ignored-correlation")["problem"]["noise_covariance"][0][1] == 0


def test_reproduction_and_replay_have_fresh_occurrences(native_experiments):
    workflow = native_experiments["workflow"]
    original = native_experiments["cases"]["baseline"]["bundle"]
    replay = native_experiments["replay"]
    assert workflow._validate(original) == workflow._validate(replay)
    assert original["bundle_digest"] != replay["bundle_digest"]
    assert original["steps"][0]["numerical_result_id"] == replay["steps"][0]["numerical_result_id"]
    assert len(native_occurrences(original)) == len(native_occurrences(replay)) == 6
    assert native_occurrences(original).isdisjoint(native_occurrences(replay))
    assert original["verification"]["independent"] is False
    assert replay["replay_receipts"][0]["admission"] == "not_performed"


def test_truth_and_heldout_do_not_enter_training_problem(native_experiments):
    source = deepcopy(native_experiments["cases"]["baseline"]["source"])
    phi = data(native_experiments,"baseline")["stages"][0]["result"]["data"]["Phi"]
    before = problems(source,phi)[0]
    for sample in source["samples"]:
        sample["truth"] = [99.,99.]
        sample["heldout"] = [99.,99.]
    assert problems(source,phi)[0] == before


def test_offline_validation_does_not_execute_solver_or_provider(native_experiments,monkeypatch):
    def refuse(*args,**kwargs):
        pytest.fail("Offline retained-record validation executed an estimator")
    for name in ("bind","invoke","_dispatch"):
        monkeypatch.setattr(native,name,refuse)
    for name in ("variational_fit","gaussian_reference","evaluate_ensemble"):
        monkeypatch.setattr(mathematics,name,refuse)
    for case in native_experiments["cases"].values():
        assert native_experiments["workflow"]._validate(case["bundle"]) == case["raw"]


@pytest.mark.parametrize("mutation", ["authority","truth","noise","policy","unknown"])
def test_invalid_source_is_refused_before_binding(mutation,monkeypatch):
    source = json.loads((ROOT / "examples/variational-free-energy/baseline.json").read_bytes())
    if mutation == "authority": source["configuration"]["state_admission"] = "approved"
    elif mutation == "truth": source["samples"][0]["truth"][0] += .1
    elif mutation == "noise": source["assumed_model"]["training_noise_covariance"] = [[1,2],[2,1]]
    elif mutation == "policy": source["solver"]["beta"] = 1
    else: source["provider_path"] = "arbitrary"
    workflow = FreeEnergyWorkflow()
    monkeypatch.setattr(workflow,"_adapters",lambda *_: pytest.fail("Invalid source reached provider binding"))
    with pytest.raises(ValueError): workflow.create_session(canonical(source),{})


def test_missing_provider_binding_is_refused():
    raw = (ROOT / "examples/variational-free-energy/baseline.json").read_bytes()
    with pytest.raises(ValueError): FreeEnergyWorkflow().create_session(raw,{})


@pytest.mark.parametrize("direction", ["original_stage_reused_as_reproduction_aggregate", "original_aggregate_reused_as_reproduction_stage"])
@pytest.mark.parametrize("stage_index", range(3))
def test_reproduction_cannot_reuse_execution_identity_across_aggregate_and_native_stages(native_experiments,direction,stage_index):
    from ciw.declared_workload import _verification
    from ciw.telemetry import digest

    case = native_experiments["cases"]["baseline"]
    bundle = deepcopy(case["bundle"])
    original, reproduction = bundle["steps"][0], bundle["verification"]["reproduction"]
    stages = reproduction["result"]["data"]["stages"]
    if direction == "original_stage_reused_as_reproduction_aggregate":
        reproduction["execution_id"] = original["result"]["data"]["stages"][stage_index]["execution_id"]
    else:
        stages[stage_index]["execution_id"] = original["execution_id"]

    def reseal(step, inputs):
        step["input_refs"] = deepcopy(inputs)
        result = step["result"]
        result["input_refs"] = deepcopy(inputs)
        result["execution_ref"] = step["execution_id"]
        result["result_id"] = digest({key:value for key,value in result.items() if key != "result_id"})
        step["result_id"], step["result_sha256"] = result["result_id"], digest(result)

    evidence = bundle["source"]["evidence"][0]["artifact_ref"]
    inputs = [evidence]
    for stage in stages:
        reseal(stage, inputs)
        inputs += [stage["result_id"]]
    reseal(reproduction, [evidence])
    bundle["verification"] = _verification(bundle, reproduction)

    workflow = native_experiments["workflow"]
    # The envelopes and numerical evidence remain valid. Only cross-occurrence
    # freshness is violated, which must be refused by the standalone workflow.
    workflow._validate_step(reproduction, case["source"], evidence)
    with pytest.raises(ValueError, match="fresh aggregate and stage occurrences"):
        workflow._validate(bundle)
