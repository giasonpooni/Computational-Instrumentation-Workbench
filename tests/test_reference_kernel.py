"""The numerical kernel probe makes a host's linear-algebra rounding part of the runtime identity."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw import energy_workflow, machine_workflow, project_workflow, thermal_workflow
from ciw import reference_workflow as base
from ciw import uncertainty_validation as validation
from ciw.telemetry import _bundle_digest, canonical

ROOT = Path(__file__).resolve().parents[1]


def _reseal(workflow, bundle):
    """Recommit a bundle whose runtimes changed, so the check exercises semantics rather than stale digests."""
    bundle["bundle_digest"] = _bundle_digest(bundle)
    bundle["verification"] = workflow._verification(bundle, bundle["verification"]["reproduction"])
    return bundle


def _retained(workflow, raw):
    bundle = workflow.create_session(raw, {})
    runtime = bundle["runtimes"][workflow.role]
    return bundle, runtime.get("algorithm", runtime)


def test_probe_is_a_stable_digest_of_the_kernels_arithmetic(monkeypatch):
    probe = base.numerical_kernel_probe()
    assert base.CODE_DIGEST.fullmatch(probe)
    assert base._kernel_probe() == probe
    solve = np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve", lambda a, b: solve(a, b) * (1 + 2 ** -52))
    assert base._kernel_probe() != probe
    monkeypatch.undo()
    assert base._kernel_probe() == probe


def test_numpy_backed_references_record_the_probe_and_the_pure_python_one_does_not():
    probe = base.numerical_kernel_probe()
    for module in (thermal_workflow, machine_workflow, validation):
        assert module.runtime_identity()["algorithm"]["kernel_probe"] == probe
    assert energy_workflow.analysis_identity()["kernel_probe"] == probe
    assert "kernel_probe" not in project_workflow.runtime_identity()["algorithm"]


@pytest.mark.parametrize("workflow,path", [
    (machine_workflow.MachineManifestWorkflow(), ROOT / "examples" / "machine-manifest" / "source.json"),
    (thermal_workflow.ThermalWorkflow(), ROOT / "examples" / "thermal-observer" / "source.json"),
    (energy_workflow.EnergyAccuracyWorkflow(), ROOT / "examples" / "energy-accuracy" / "baseline.json"),
])
def test_a_retained_identity_with_another_or_no_probe_reopens_but_replay_names_the_probe(workflow, path):
    raw = path.read_bytes()
    bundle, algorithm = _retained(workflow, raw)
    assert workflow.replay_session(bundle, {})["replay_receipt"]["numerical_match"] is True
    other_host = deepcopy(bundle)
    runtime = other_host["runtimes"][workflow.role]
    runtime.get("algorithm", runtime)["kernel_probe"] = "f" * 64
    assert workflow._validate(_reseal(workflow, other_host)) == raw
    with pytest.raises(ValueError, match="runtime identity differs from the retained execution: .*kernel_probe"):
        workflow.replay_session(other_host, {})
    before_probes = deepcopy(bundle)
    runtime = before_probes["runtimes"][workflow.role]
    del runtime.get("algorithm", runtime)["kernel_probe"]
    assert workflow._validate(_reseal(workflow, before_probes)) == raw
    with pytest.raises(ValueError, match="kernel_probe"):
        workflow.replay_session(before_probes, {})
    malformed = deepcopy(bundle)
    runtime = malformed["runtimes"][workflow.role]
    runtime.get("algorithm", runtime)["kernel_probe"] = "not-a-digest"
    with pytest.raises(ValueError):
        workflow._validate(_reseal(workflow, malformed))


def test_refusal_names_every_differing_identity_field():
    current = {"schema": "s", "algorithm": {"code_sha256": "a", "kernel_probe": "p", "numpy_version": "2"}, "role": "x"}
    retained = {"schema": "s", "algorithm": {"code_sha256": "b", "numpy_version": "2"}, "role": "x"}
    assert base.identity_differences(current, retained) == ["algorithm.code_sha256", "algorithm.kernel_probe"]
    assert base.identity_differences(current, current) == []
    assert base.identity_differences("a", "b") == ["identity"]
    assert json.loads(canonical(current)) == current
