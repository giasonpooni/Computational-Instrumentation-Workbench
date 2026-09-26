import copy

import pytest

from ciw.state_transformation import SCHEMA, SEALED_SCHEMA, seal_contract, validate_contract, validate_sealed


def contract():
    return {
        "schema": SCHEMA,
        "transformation_id": "telemetry.filter.v1",
        "input_state_space": {
            "name": "raw_telemetry",
            "coordinates": [
                {"name": "temperature", "unit": "K", "role": "measured"},
                {"name": "temperature_variance", "unit": "K^2", "role": "uncertainty"},
            ],
            "frame": "sensor:ambient-1",
            "clock": "utc:source-clock",
        },
        "transformation": {
            "name": "bounded_low_pass",
            "kind": "deterministic_filter",
            "parameters": {"cutoff_hz": 2.0, "order": 2},
        },
        "constraints": [{"id": "finite", "statement": "all values are finite", "check": "finite_json"}],
        "invariants": [
            {"id": "sample-order", "statement": "sample ordering is preserved", "check": "monotone_timestamps"},
            {"id": "lineage", "statement": "source lineage remains attached", "check": "source_ref_present"},
        ],
        "output_state_space": {
            "name": "filtered_telemetry",
            "coordinates": [
                {"name": "temperature", "unit": "K", "role": "estimated"},
                {"name": "temperature_variance", "unit": "K^2", "role": "uncertainty"},
            ],
            "frame": "sensor:ambient-1",
            "clock": "utc:source-clock",
        },
        "evidence": [{"ref": "source-1", "kind": "input_record", "role": "raw_measurement"}],
        "authority": {
            "execution": "not_performed",
            "verification": "not_performed",
            "state_admission": "not_performed",
            "hardware_actuation": "not_performed",
        },
    }


def test_contract_validates_and_seals_without_execution_claims():
    validated = validate_contract(contract())
    assert validated["transformation_id"] == "telemetry.filter.v1"
    sealed = seal_contract(validated)
    assert sealed["schema"] == SEALED_SCHEMA
    assert validate_sealed(sealed)["authority"]["execution"] == "not_performed"


def test_sealed_contract_refuses_tampering():
    sealed = seal_contract(contract())
    changed = copy.deepcopy(sealed)
    changed["contract"]["output_state_space"]["frame"] = "sensor:other"
    with pytest.raises(ValueError, match="identity differs"):
        validate_sealed(changed)


def test_contract_refuses_duplicate_coordinates_or_authority_escalation():
    duplicate = copy.deepcopy(contract())
    duplicate["output_state_space"]["coordinates"].append(
        {"name": "temperature", "unit": "K", "role": "estimated"})
    with pytest.raises(ValueError, match="names must be distinct"):
        validate_contract(duplicate)
    unauthorized = copy.deepcopy(contract())
    unauthorized["authority"]["execution"] = "performed"
    with pytest.raises(ValueError, match="authority"):
        validate_contract(unauthorized)

