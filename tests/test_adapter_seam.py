"""The domain-neutral seam preserves oscillator science and refuses implicit execution."""

from copy import deepcopy
import hashlib
import json
import math

import pytest

from ciw.adapters.oscillator import OscillatorAdapter
from ciw.adapters.protocol import AdapterRefusal, InstrumentManifest
from ciw.adapters.registry import AdapterRegistry, RecordAdapter, default_registry
from ciw.core.identities import (
    digest, evidence_id, new_identity, validate_evidence_identity, validate_identity,
)
from ciw.core.records import validate_run_structure
from ciw.instruments import inspect_sample, make_demo_run, run_metadata, validate_run


def event_run():
    manifest = InstrumentManifest(
        instrument_id="example.measurement-adapter", role="measurement_adapter",
        inputs=("measurement-record.v1",), units={"load": "kg", "missing": "kg"},
        frames=("rig-A",), sampling={"kind": "event"},
        supported_operations=("example.calibrate.v1",),
        calibration_requirements={"required": True, "parameter_covariance": "retained"},
    )
    run = {
        "run_schema": "run.v1", "run_id": "run-event-A", "evidence_id": "pending",
        "instrument": manifest.instrument_id,
        "metadata": {"duration_s": 1.0, "sample_count": 1, "sample_rate_hz": None,
                     "coordinate_frame": "rig-A", "provenance": {"source": "retained observation"},
                     "manifest": manifest.to_dict()},
        "time_s": [0.0], "channels": {"load": {"unit": "kg", "values": [2.0]},
                                      "missing": {"unit": "kg", "values": [None]}},
        "render": {},
    }
    run["evidence_id"] = evidence_id(run)
    return run


def test_oscillator_is_ordinary_adapter_with_unchanged_evidence_and_outputs():
    run = make_demo_run()
    # libm/NumPy may differ in the last float bits between operating systems.
    # Bind the retained values exactly; scientific equivalence is checked by
    # the analytic/tolerance tests in test_instruments, not a platform hash.
    scientific = {key: run[key] for key in ("instrument", "metadata", "time_s", "channels")}
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert run["evidence_id"] == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    baseline_digest = digest(run)
    assert make_demo_run() == run
    assert "manifest" not in run["metadata"]  # No retroactive hash or provenance rewrite.
    registry = AdapterRegistry()
    registry.register(OscillatorAdapter())
    assert registry.get(run["instrument"]).manifest.role == "instrument"
    result = registry.execute("statistics.v1", run, {"channel": "q", "interval_s": [0.0, 12.0]})
    assert result["sample_count"] == 768
    assert result["unit"] == "m"
    assert digest(run) == baseline_digest


def test_event_record_can_be_reopened_without_engine_or_fictitious_sampling():
    run = event_run()
    validate_run(run)
    validate_evidence_identity(run)
    adapter = default_registry().resolve(run)
    assert isinstance(adapter, RecordAdapter)
    sample = inspect_sample(run, 0.0)
    assert sample["values"] == {"load": 2.0, "missing": None}
    with pytest.raises(AdapterRefusal, match="not bound") as error:
        default_registry().execute("example.calibrate.v1", run, {})
    assert error.value.code == "adapter_unavailable"


def test_manifest_round_trip_is_detached_strict_and_data_only():
    manifest = InstrumentManifest.from_dict(event_run()["metadata"]["manifest"])
    snapshot = manifest.to_dict()
    snapshot["sampling"]["kind"] = "modified"
    assert manifest.sampling["kind"] == "event"
    bad = manifest.to_dict()
    bad["python_module"] = "do.not.import"
    with pytest.raises(ValueError, match="unknown fields"):
        InstrumentManifest.from_dict(bad)
    bad = manifest.to_dict()
    bad["tolerance_policy"] = {"rtol": math.inf}
    with pytest.raises(ValueError, match="finite"):
        InstrumentManifest.from_dict(bad)


@pytest.mark.parametrize("mutate", [
    lambda r: r["channels"]["load"]["values"].__setitem__(0, True),
    lambda r: r["channels"]["load"]["values"].__setitem__(0, math.nan),
    lambda r: r["channels"]["missing"]["values"].append(None),
    lambda r: r["metadata"].__setitem__("sample_count", True),
    lambda r: r["metadata"].__setitem__("sample_rate_hz", 0),
    lambda r: r["time_s"].__setitem__(0, 1.0),
    lambda r: r["metadata"]["provenance"].__setitem__("hidden", math.inf),
])
def test_structural_validation_rejects_bad_record_anywhere(mutate):
    run = event_run()
    mutate(run)
    with pytest.raises(ValueError):
        validate_run_structure(run)


def test_adapter_declaration_binds_instrument_frame_and_units():
    for key, value in (("instrument", "different"),):
        run = event_run()
        run[key] = value
        with pytest.raises(ValueError, match="instrument"):
            validate_run(run)
    run = event_run()
    run["metadata"]["coordinate_frame"] = "other-rig"
    with pytest.raises(ValueError, match="frame"):
        validate_run(run)
    run = event_run()
    run["channels"]["load"]["unit"] = "lb"
    with pytest.raises(ValueError, match="unit"):
        validate_run(run)
    run = event_run()
    del run["metadata"]["manifest"]["units"]["load"]
    with pytest.raises(ValueError, match="no declared"):
        validate_run(run)


def test_channel_descriptions_retain_observation_roles_without_copying_samples():
    run = event_run()
    run["channels"]["load"].update({"kind": "calibrated_observation", "source": {"channel": "raw"}})
    description = run_metadata(run)
    assert description["channels"]["load"] == {
        "unit": "kg", "kind": "calibrated_observation", "source": {"channel": "raw"}}
    description["channels"]["load"]["source"]["channel"] = "changed"
    assert run["channels"]["load"]["source"]["channel"] == "raw"


def test_embedded_manifest_cannot_override_registered_domain_checks():
    run = make_demo_run()
    manifest = OscillatorAdapter.manifest.to_dict()
    manifest["calibration_requirements"] = {"required": True}
    run["metadata"]["manifest"] = manifest
    with pytest.raises(ValueError, match="registered adapter contract"):
        validate_run(run)


def test_unknown_run_requires_explicit_manifest_and_duplicate_adapter_fails():
    run = event_run()
    del run["metadata"]["manifest"]
    with pytest.raises(AdapterRefusal, match="embedded"):
        validate_run(run)
    registry = AdapterRegistry()
    registry.register(OscillatorAdapter())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(OscillatorAdapter())


def test_evidence_is_content_bound_while_execution_and_verification_are_distinct():
    run = event_run()
    identity = evidence_id(run)
    run["render"] = {"coordinate_frame": "rig-A", "display": "alternate"}
    assert evidence_id(run) == identity
    run["metadata"]["provenance"]["calibration"] = {"covariance": [[0.02, 0.001], [0.001, 0.003]]}
    assert evidence_id(run) != identity
    with pytest.raises(ValueError, match="integrity"):
        validate_evidence_identity(run)
    ids = [new_identity(kind) for kind in ("execution", "result", "verification", "execution")]
    assert len(set(ids)) == 4
    for kind, value in zip(("execution", "result", "verification", "execution"), ids):
        assert validate_identity(value, kind) == value
    with pytest.raises(ValueError):
        validate_identity(ids[0], "verification")
    with pytest.raises(ValueError):
        new_identity("evidence")


def test_registry_dispatch_does_not_allow_engine_to_mutate_retained_inputs():
    class MutatingAdapter(RecordAdapter):
        def execute(self, operation_id, run, parameters):
            run["channels"]["load"]["values"][0] = 99.0
            parameters["nested"]["value"] = 4
            return {"ok": True}

    run = event_run()
    before = deepcopy(run)
    parameters = {"nested": {"value": 1}}
    registry = AdapterRegistry()
    registry.register(MutatingAdapter(InstrumentManifest.from_dict(run["metadata"]["manifest"])))
    assert registry.execute("example.calibrate.v1", run, parameters) == {"ok": True}
    assert run == before
    assert parameters == {"nested": {"value": 1}}
