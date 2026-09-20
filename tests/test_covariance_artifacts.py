"""Covariance retains ordered physical coordinates and shared uncertainty as evidence."""

from copy import deepcopy
import hashlib
import json
import math

import pytest

from ciw.core.covariance import (
    covariance_identity, create_covariance_artifact, validate_covariance_artifact,
)


def declarations():
    return {
        "quantity_ids": ["calibration.scale", "calibration.zero_raw"],
        "units": ["kg/count", "count"], "frame": "load-cell-calibration",
        "reference_values": [0.01, 0.0], "matrix": [[1e-10, 1e-7], [1e-7, 0.04]],
        "method": "declared_parameter_covariance",
        "basis": {"kind": "parameter", "id": "scale-zero-raw.v1"},
        "provenance": {"provider": "org.notationsystems.rci",
                       "source_evidence_ids": ["sha256:" + "a" * 64],
                       "source_covariance_ids": [],
                       "metadata": {"calibration_id": "declared-calibration-1", "traceability": "none_claimed"}},
        "assumptions": ["Parameter covariance is shared by all records using this calibration"],
    }


def artifact():
    return create_covariance_artifact(**declarations())


def reseal(value):
    value["covariance_id"] = covariance_identity(value)
    return value


def test_constructor_preserves_mixed_units_cross_covariance_and_detaches_source():
    source = declarations()
    retained = create_covariance_artifact(**source)
    assert retained["matrix"] == [[1e-10, 1e-7], [1e-7, 0.04]]
    assert retained["units"] == ["kg/count", "count"]
    assert retained["quantity_ids"] == ["calibration.scale", "calibration.zero_raw"]
    source["matrix"][0][1] = 0.0
    source["provenance"]["metadata"]["calibration_id"] = "changed"
    assert retained["matrix"][0][1] == 1e-7
    assert retained["provenance"]["metadata"]["calibration_id"] == "declared-calibration-1"
    before = deepcopy(retained)
    validate_covariance_artifact(retained)
    assert retained == before


def test_content_identity_uses_existing_ascii_canonical_encoder_and_no_event_identity():
    fields = declarations()
    fields["provenance"]["metadata"]["description"] = "Étalonnage simulé"
    retained = create_covariance_artifact(**fields)
    payload = {key: value for key, value in retained.items() if key != "covariance_id"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    assert retained["covariance_id"] == "sha256:" + expected
    assert not {"execution_id", "result_id", "verification_id"} & retained.keys()
    assert retained == create_covariance_artifact(**fields)
    retained["covariance_id"] = "sha256:" + "f" * 64
    assert covariance_identity(retained) == "sha256:" + expected
    with pytest.raises(ValueError, match="identity"):
        validate_covariance_artifact(retained)


def test_shared_sample_covariance_roundtrips_without_becoming_independent_sigmas():
    parent = artifact()
    fields = declarations()
    fields.update(quantity_ids=["sample:0.mass", "sample:1.mass"], units=["kg", "kg"],
                  reference_values=[70.0, 30.0], frame="two-samples-one-load-cell",
                  matrix=[[0.0049, 0.0021], [0.0021, 0.0009]],
                  method="first_order_calibration_transport",
                  basis={"kind": "calibrated_observation", "id": "shared-calibration-batch"})
    fields["provenance"]["source_covariance_ids"] = [parent["covariance_id"]]
    retained = create_covariance_artifact(**fields)
    reopened = json.loads(json.dumps(retained))
    validate_covariance_artifact(reopened)
    assert reopened == retained
    assert reopened["matrix"][0][1] == 0.0021
    assert reopened["provenance"]["source_covariance_ids"] == [parent["covariance_id"]]


@pytest.mark.parametrize("matrix", [
    [[0.0, 0.0], [0.0, 0.0]],
    [[1.0, -1.0], [-1.0, 1.0]],
    [[0.0, 0.0], [0.0, 2.0]],
    [[1e-300, 0.4], [0.4, 1e300]],
    [[5e-324, 0.0], [0.0, 1e308]],
])
def test_positive_semidefinite_including_singular_and_extreme_mixed_scales(matrix):
    fields = declarations()
    fields["matrix"] = matrix
    retained = create_covariance_artifact(**fields)
    assert retained["matrix"] == matrix


@pytest.mark.parametrize("matrix", [
    [[-1e-300, 0.0], [0.0, 1e300]],
    [[1e-300, 1.1], [1.1, 1e300]],
    [[1e-20, 5e-21], [0.0, 1e-20]],
    [[0.0, 1e-300], [1e-300, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]],
    [[1.0, math.inf], [math.inf, 1.0]],
    [[1.0, math.nan], [math.nan, 1.0]],
    [[True, 0.0], [0.0, 1.0]],
    [[1.0, None], [None, 1.0]],
    [[1.0]],
    [[1.0, 0.0], [0.0]],
])
def test_invalid_matrix_refuses_without_repair(matrix):
    fields = declarations()
    fields["matrix"] = matrix
    with pytest.raises(ValueError):
        create_covariance_artifact(**fields)


def test_indefiniteness_in_small_variance_block_cannot_hide_behind_large_variance():
    fields = declarations()
    fields.update(quantity_ids=["large", "small1", "small2"], units=["m", "m", "m"],
                  reference_values=[0.0, 0.0, 0.0],
                  matrix=[[1e20, 0.0, 0.0], [0.0, 1e-20, 2e-20], [0.0, 2e-20, 1e-20]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        create_covariance_artifact(**fields)


def test_float_tolerance_acceptance_does_not_symmetrize_or_clip_retained_entries():
    fields = declarations()
    fields["matrix"] = [[1.0, 1.0 + 1e-13], [1.0, 1.0]]
    retained = create_covariance_artifact(**fields)
    assert retained["matrix"] == fields["matrix"]
    assert retained["matrix"][0][1] != retained["matrix"][1][0]


@pytest.mark.parametrize(("key", "value"), [
    ("quantity_ids", ["duplicate", "duplicate"]),
    ("quantity_ids", []),
    ("units", ["kg"]),
    ("units", ["kg", ""]),
    ("reference_values", [1.0, None]),
    ("reference_values", [1.0, True]),
    ("reference_values", [1.0, math.inf]),
    ("basis", {"kind": "unknown", "id": "basis"}),
    ("basis", {"kind": "parameter"}),
    ("frame", ""),
    ("method", ""),
    ("assumptions", [""]),
])
def test_invalid_axis_and_domain_declarations_refuse(key, value):
    fields = declarations()
    fields[key] = value
    with pytest.raises(ValueError):
        create_covariance_artifact(**fields)


def test_external_binding_rejects_reordered_quantities_mixed_unit_mismatch_and_frame():
    retained = artifact()
    expected = declarations()
    validate_covariance_artifact(retained, expected_quantity_ids=expected["quantity_ids"],
                                 expected_units=expected["units"], expected_frame=expected["frame"])
    reordered = deepcopy(retained)
    reordered["quantity_ids"].reverse()
    reseal(reordered)
    with pytest.raises(ValueError, match="quantity order"):
        validate_covariance_artifact(reordered, expected_quantity_ids=expected["quantity_ids"])
    with pytest.raises(ValueError, match="quantity units"):
        validate_covariance_artifact(retained, expected_units=["kg", "count"])
    with pytest.raises(ValueError, match="coordinate frame"):
        validate_covariance_artifact(retained, expected_frame="other-load-cell")


@pytest.mark.parametrize("mutate", [
    lambda a: a["matrix"][0].__setitem__(0, 2e-10),
    lambda a: a["units"].__setitem__(0, "g/count"),
    lambda a: a["reference_values"].__setitem__(0, 0.02),
    lambda a: a["provenance"]["metadata"].__setitem__("calibration_id", "different"),
    lambda a: a["assumptions"].append("New independence assumption"),
])
def test_all_retained_scientific_declarations_are_content_bound(mutate):
    retained = artifact()
    mutate(retained)
    with pytest.raises(ValueError, match="identity"):
        validate_covariance_artifact(retained)


def test_provenance_and_artifact_schema_refuse_untyped_or_hidden_content():
    for key, value in (("source_evidence_ids", ["execution-" + "0" * 32]),
                       ("source_evidence_ids", ["sha256:" + "a" * 64] * 2),
                       ("source_covariance_ids", ["a" * 64]),
                       ("source_covariance_ids", ["sha256:" + "a" * 64] * 2),
                       ("metadata", {"hidden": math.nan}), ("metadata", {1: "untyped"})):
        fields = declarations()
        fields["provenance"][key] = value
        with pytest.raises(ValueError):
            create_covariance_artifact(**fields)
    retained = artifact()
    retained["result_id"] = "result-" + "0" * 32
    reseal(retained)
    with pytest.raises(ValueError, match="exactly"):
        validate_covariance_artifact(retained)
