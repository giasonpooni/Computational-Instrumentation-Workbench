"""Each identity check of the shared reference lifecycle is guarded by a test that reseals everything except the tampered field.

A mutation probe removed these checks one at a time and no test failed; the
checks were redundant with nothing. Every case here fails if its check is
removed again.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from ciw import energy_workflow, machine_workflow
from ciw.telemetry import _bundle_digest, digest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (machine_workflow.MachineManifestWorkflow(), ROOT / "examples" / "machine-manifest" / "source.json"),
    (energy_workflow.EnergyAccuracyWorkflow(), ROOT / "examples" / "energy-accuracy" / "baseline.json"),
]


def _reseal(workflow, bundle, subject=None):
    """Recompute the bundle digest and the verification artifact for whatever the bundle now holds."""
    bundle["bundle_digest"] = _bundle_digest(bundle) if subject is None else subject
    bundle["verification"] = workflow._artifact(bundle["bundle_digest"], bundle["runtimes"],
                                                bundle["verification"]["reproduction"])
    return bundle


def _reseal_step(step):
    result = step["result"]
    result["result_id"] = digest({key: value for key, value in result.items() if key != "result_id"})
    step["result_id"], step["result_sha256"] = result["result_id"], digest(result)
    return step


@pytest.fixture(params=CASES, ids=["machine", "energy"])
def retained(request):
    workflow, path = request.param
    raw = path.read_bytes()
    return workflow, raw, workflow.create_session(raw, {})


def test_the_resealed_bundle_is_still_accepted(retained):
    workflow, raw, bundle = retained
    assert workflow._validate(_reseal(workflow, deepcopy(bundle))) == raw


def test_input_refs_must_name_the_retained_evidence(retained):
    workflow, _, bundle = retained
    changed = deepcopy(bundle)
    for step in (changed["steps"][0], changed["verification"]["reproduction"]):
        step["input_refs"] = ["sha256:" + "0" * 64]
        step["result"]["input_refs"] = ["sha256:" + "0" * 64]
        _reseal_step(step)
    with pytest.raises(ValueError, match="occurrence binding differs|Malformed"):
        workflow._validate(_reseal(workflow, changed))


def test_request_digest_must_commit_to_the_request(retained):
    workflow, _, bundle = retained
    changed = deepcopy(bundle)
    changed["steps"][0]["request_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="occurrence binding differs"):
        workflow._validate(_reseal(workflow, changed))


def test_numerical_copy_must_equal_the_result_data(retained):
    workflow, _, bundle = retained
    diverged = deepcopy(bundle)
    step = diverged["steps"][0]
    step["numerical_result"] = {"operation_id": step["operation_id"], "data": {"replaced": True}}
    step["numerical_result_id"] = digest(step["numerical_result"])
    with pytest.raises(ValueError, match="numerical result identity differs"):
        workflow._validate(_reseal(workflow, diverged))
    mislabelled = deepcopy(bundle)
    mislabelled["steps"][0]["numerical_result_id"] = "sha256:" + "2" * 64
    with pytest.raises(ValueError, match="numerical result identity differs"):
        workflow._validate(_reseal(workflow, mislabelled))


def test_bundle_digest_must_be_the_digest_of_the_bundle(retained):
    workflow, _, bundle = retained
    changed = _reseal(workflow, deepcopy(bundle), subject="sha256:" + "3" * 64)
    assert changed["verification"]["subject_ref"] == changed["bundle_digest"], "the artifact agrees with the false digest"
    with pytest.raises(ValueError, match="identity or size differs"):
        workflow._validate(changed)


def test_session_identity_must_keep_its_shape(retained):
    workflow, _, bundle = retained
    changed = deepcopy(bundle)
    changed["session_id"] = "session-not-hexadecimal"
    with pytest.raises(ValueError, match="identity or size differs"):
        workflow._validate(_reseal(workflow, changed))


def test_code_digest_must_be_a_hex_digest(retained):
    workflow, _, bundle = retained
    changed = deepcopy(bundle)
    runtime = changed["runtimes"][workflow.role]
    runtime.get("algorithm", runtime)["code_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="algorithm identity|implementation identity"):
        workflow._validate(_reseal(workflow, changed))


def test_a_receipt_may_not_name_its_own_bundle_as_the_source(retained):
    workflow, _, bundle = retained
    replayed = workflow.replay_session(bundle, {})["session"]
    receipt = deepcopy(replayed["replay_receipts"][0])
    receipt["source_bundle_digest"] = replayed["bundle_digest"]
    receipt["verification"] = workflow._artifact(receipt["source_bundle_digest"], replayed["runtimes"], replayed["steps"][0])
    receipt["replay_id"] = digest({key: value for key, value in receipt.items() if key != "replay_id"})
    self_referencing = deepcopy(replayed)
    self_referencing["replay_receipts"] = [receipt]
    with pytest.raises(ValueError, match="replay receipt"):
        workflow._validate(self_referencing)
    assert workflow._validate(replayed)
