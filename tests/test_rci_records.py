"""Contradictory resealed v2 provenance must fail without executing RCI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pytest

from ciw.adapters.rci_records import validate_v2_provenance


def native_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reseal(batch):
    batch["calibration_digest"] = native_digest(batch["calibration"])
    batch["uncertainty_digest"] = native_digest(batch["uncertainty"])
    for record in batch["records"]:
        record["calibration_digest"] = batch["calibration_digest"]
        record["uncertainty_digest"] = batch["uncertainty_digest"]
        record.pop("derived_evidence_digest", None)
        record["derived_evidence_digest"] = native_digest(record)


@pytest.fixture
def retained():
    path = Path(__file__).parents[1] / "examples" / "adapters" / "two-reservoir-covariance.json"
    request = json.loads(path.read_text())["sensors"][0]["request"]
    binding = deepcopy(request["inputs"]["calibration"])
    basis = binding["covariance_basis"]
    # This is a provenance-only fixture: no propagated numeric result is invented.
    # Its source declarations come from the runnable retained example.
    calibration = tomllib.loads(request["inputs"]["assembly_toml"])["calibration"]
    blocks = [*basis["parameter_components"], basis["raw"], basis["residual"]]
    summary = {
        "scope": "declared_components_only", "completeness_claimed": False,
        "excluded_component_ids": ["other-systematics"], "represented_component_ids": [],
        "evidence_ids": sorted({item for block in [*blocks, basis["independence"]] for item in block["evidence_ids"]}),
        "dependency_ids": sorted({item for block in blocks for item in block["dependency_ids"]}),
        "shared_source_ids": sorted({item for block in blocks for item in block["shared_source_ids"]}),
    }
    units = [calibration["output_unit"] + "/" + calibration["raw_unit"], calibration["raw_unit"]]
    uncertainty = {
        "covariance_basis": deepcopy(basis), "covariance_basis_digest": native_digest(basis),
        "covariance_coverage": summary, "dependency_ids": summary["dependency_ids"],
        "shared_source_ids": summary["shared_source_ids"],
        "parameterization": {"name": "scale_zero_raw", "order": ["scale", "zero_raw"],
                             "values": calibration["theta"], "units": units},
        "parameter_order": ["scale", "zero_raw"], "parameter_units": units,
        "parameter_covariance": deepcopy(binding["parameter_covariance"]),
        "raw_covariance": deepcopy(request["inputs"]["raw_covariance"]),
        "raw_parameter_independent": True, "residual_correlation": "independent",
        "parameter_correlation_across_records": "shared_profile", "traceability": "none_claimed",
    }
    observed_at = request["inputs"]["records"][0]["observed_at"]
    record = {
        "schema": "measurement-record.v2", "kind": "calibrated_observation",
        "covariance_basis_digest": native_digest(basis), "covariance_row": 0,
        "observed_at": observed_at, "observation_time_basis": "caller_declared_acquisition_time",
        "claim_scope": "declared_calibration_only",
        "acquisition_applicability": {"observed_at": observed_at, "applicable": True,
                                      "valid_from": binding["valid_from"], "valid_until": binding["valid_until"],
                                      "basis": "caller_declared_acquisition_time"},
    }
    batch = {"schema": "measurement-record-batch.v2", "calibration": binding, "uncertainty": uncertainty,
             "covariance_basis_digest": native_digest(basis), "records": [record]}
    reseal(batch)
    return batch, request


def test_offline_provenance_check_never_imports_provider_and_never_mutates(retained):
    batch, request = retained
    before = deepcopy(retained)
    modules = set(sys.modules)
    validate_v2_provenance(batch, request)
    assert retained == before
    assert not any(name == "instrument_chain" or name.startswith("instrument_chain.")
                   for name in set(sys.modules) - modules)


@pytest.mark.parametrize("location", ["batch", "uncertainty", "record"])
def test_resealed_wrong_native_basis_commitment_refuses(retained, location):
    batch, request = retained
    target = {"batch": batch, "uncertainty": batch["uncertainty"], "record": batch["records"][0]}[location]
    target["covariance_basis_digest"] = "0" * 64
    reseal(batch)
    with pytest.raises(ValueError, match="basis.*(digest|commitment)"):
        validate_v2_provenance(batch, request)


@pytest.mark.parametrize("field,value", [
    ("scope", "complete_uncertainty"), ("completeness_claimed", True),
    ("completeness_claimed", 0), ("excluded_component_ids", []),
    ("represented_component_ids", ["reference"]), ("dependency_ids", []),
    ("evidence_ids", ["invented-certificate"]), ("shared_source_ids", []),
])
def test_resealed_coverage_summary_cannot_promote_or_drop_sources(retained, field, value):
    batch, request = retained
    batch["uncertainty"]["covariance_coverage"][field] = value
    reseal(batch)
    with pytest.raises(ValueError, match="coverage"):
        validate_v2_provenance(batch, request)


@pytest.mark.parametrize("field,value", [
    ("name", "affine_gain_offset"), ("order", ["zero_raw", "scale"]),
    ("values", [0.01, 5.0]), ("values", [0.01, False]),
    ("units", ["kg", "count"]),
])
def test_resealed_parameterization_stays_bound_to_native_assembly(retained, field, value):
    batch, request = retained
    batch["uncertainty"]["parameterization"][field] = value
    reseal(batch)
    with pytest.raises(ValueError, match="parameterization|JSON number"):
        validate_v2_provenance(batch, request)


@pytest.mark.parametrize("field,value", [
    ("traceability", "certified"), ("raw_parameter_independent", False),
    ("parameter_correlation_across_records", "independent"),
    ("parameter_covariance", [[1.0, 0.0], [0.0, 1.0]]),
    ("raw_covariance", [[99.0]]), ("dependency_ids", []),
])
def test_resealed_uncertainty_cannot_change_its_source_or_scope(retained, field, value):
    batch, request = retained
    batch["uncertainty"][field] = value
    reseal(batch)
    with pytest.raises(ValueError):
        validate_v2_provenance(batch, request)


@pytest.mark.parametrize("field,value", [
    ("applicable", 1), ("applicable", False),
    ("valid_until", "2027-02-01T00:00:00Z"),
    ("observed_at", "2026-01-16T12:00:00Z"),
])
def test_resealed_historical_applicability_cannot_be_rewritten(retained, field, value):
    batch, request = retained
    batch["records"][0]["acquisition_applicability"][field] = value
    reseal(batch)
    with pytest.raises(ValueError, match="applicability"):
        validate_v2_provenance(batch, request)


def test_legacy_aware_timestamp_forms_are_retained_without_normalization(retained):
    batch, request = retained
    for binding in (batch["calibration"], request["inputs"]["calibration"]):
        binding["valid_from"] = "2026-01-01T00:00+00:00"
    record = batch["records"][0]
    record["observed_at"] = request["inputs"]["records"][0]["observed_at"] = "2026-01-15 12:00:00+00:00"
    record["acquisition_applicability"].update(observed_at=record["observed_at"], valid_from=batch["calibration"]["valid_from"])
    reseal(batch)
    original = deepcopy(retained)
    validate_v2_provenance(batch, request)
    assert retained == original


def test_unicode_basis_uses_native_utf8_identity(retained):
    batch, request = retained
    for basis in (request["inputs"]["calibration"]["covariance_basis"],
                  batch["calibration"]["covariance_basis"], batch["uncertainty"]["covariance_basis"]):
        basis["parameter_components"][0]["reason"] += " — étalonnage"
    basis = batch["calibration"]["covariance_basis"]
    native = native_digest(basis)
    for target in (batch, batch["uncertainty"], batch["records"][0]):
        target["covariance_basis_digest"] = native
    reseal(batch)
    validate_v2_provenance(batch, request)
    ascii_digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert native != ascii_digest
