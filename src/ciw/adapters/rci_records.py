"""Offline RCI v2 provenance relations; no calibration engine is executed.

This validator checks retained declarations against each other and their native
content commitments. It neither recomputes a calibration/covariance nor proves
the documentary references true, complete, independent, or traceable.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import tomllib


def _require(condition, message):
    if not condition:
        raise ValueError("Invalid RCI v2 provenance: " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _object(value, fields, name):
    _require(isinstance(value, dict) and set(value) == set(fields), name + " fields differ")


def _text(value):
    _require(isinstance(value, str) and bool(value.strip()) and value == value.strip(),
             "identifiers and reasons must be nonempty trimmed strings")


def _ids(value, *, required=False):
    _require(isinstance(value, list) and (bool(value) or not required), "identifier list is missing")
    for item in value:
        _text(item)
    _require(len(value) == len(set(value)), "duplicate provenance identifier")
    return set(value)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "expected a finite JSON number")


def _matrix(value):
    _require(isinstance(value, list) and len(value) == 2, "parameter covariance must be 2 by 2")
    for row in value:
        _require(isinstance(row, list) and len(row) == 2, "parameter covariance must be 2 by 2")
        for item in row:
            _number(item)


def _basis_summary(basis):
    _object(basis, {"schema", "parameter_components", "raw", "residual", "independence"}, "basis")
    _require(basis["schema"] == "rci-covariance-basis.v1", "unknown covariance basis schema")
    components = basis["parameter_components"]
    _require(isinstance(components, list) and bool(components), "parameter components are missing")
    evidence, dependencies, shared = set(), set(), set()
    excluded, represented, additive, by_id, kinds = [], [], [], {}, set()
    provenance_fields = {"reason", "evidence_ids", "dependency_ids", "shared_source_ids"}

    def provenance(block):
        _text(block["reason"])
        evidence.update(_ids(block["evidence_ids"], required=True))
        deps = _ids(block["dependency_ids"])
        sources = _ids(block["shared_source_ids"])
        dependencies.update(deps)
        shared.update(sources)
        return deps | sources

    for component in components:
        _object(component, provenance_fields | {"component_id", "kind", "status", "covariance", "represented_by"}, "component")
        identifier = component["component_id"]
        _text(identifier)
        _require(identifier not in by_id, "duplicate component identity")
        by_id[identifier] = component
        _require(isinstance(component["kind"], str) and component["kind"] in {
            "fitting", "reference_standard", "shared_systematic"}, "unknown parameter component kind")
        kinds.add(component["kind"])
        sources = provenance(component)
        if component["status"] == "included":
            _require(component["represented_by"] is None and bool(sources), "included component has ambiguous source")
            _matrix(component["covariance"])
            additive.append(sources)
        elif component["status"] == "excluded":
            _require(component["covariance"] is None, "excluded component cannot have additive covariance")
            if component["represented_by"] is None:
                excluded.append(identifier)
            else:
                _text(component["represented_by"])
                represented.append(identifier)
        else:
            raise ValueError("Invalid RCI v2 provenance: unknown component status")
    _require(kinds == {"fitting", "reference_standard", "shared_systematic"} and bool(additive),
             "all parameter kinds and at least one included component must be declared")
    for identifier in represented:
        component = by_id[identifier]
        target = by_id.get(component["represented_by"])
        sources = set(component["dependency_ids"]) | set(component["shared_source_ids"])
        _require(target is not None and target["status"] == "included", "represented component has no included target")
        target_sources = set(target["dependency_ids"]) | set(target["shared_source_ids"])
        _require(bool(sources) and sources <= target_sources, "represented source differs from its target")
    for name in ("raw", "residual"):
        block = basis[name]
        _object(block, provenance_fields | ({"scope"} if name == "residual" else set()), name)
        if name == "residual":
            _require(block["scope"] == "additional_independent_output_residual", "residual scope double-counts uncertainty")
        sources = provenance(block)
        _require(bool(sources), name + " requires source identities")
        additive.append(sources)
    for index, sources in enumerate(additive):
        _require(not any(sources & previous for previous in additive[:index]), "additive uncertainty sources overlap")
    for identifier in excluded:
        component = by_id[identifier]
        sources = set(component["dependency_ids"]) | set(component["shared_source_ids"])
        _require(not any(sources & included for included in additive), "excluded source needs explicit representation")
    independence = basis["independence"]
    _object(independence, {"parameter_components", "raw_parameter", "residual_other", "reason", "evidence_ids"}, "independence")
    _require(all(independence[key] is True for key in ("parameter_components", "raw_parameter", "residual_other")),
             "independence must be explicitly declared")
    _text(independence["reason"])
    evidence.update(_ids(independence["evidence_ids"], required=True))
    return {"scope": "declared_components_only", "completeness_claimed": False,
            "excluded_component_ids": excluded, "represented_component_ids": represented,
            "evidence_ids": sorted(evidence), "dependency_ids": sorted(dependencies),
            "shared_source_ids": sorted(shared)}


def _timestamp(value):
    _require(isinstance(value, str), "acquisition time must be a timestamp")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(instant.utcoffset() is not None, "acquisition time requires a timezone")
    return instant


def validate_v2_provenance(batch, request):
    """Validate v2-only semantic bindings after the common raw/digest checks."""
    try:
        _validate_v2(batch, request)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid RCI v2 provenance structure") from exc


def _validate_v2(batch, request):
    _require(request["operation_id"] == "rci.calibrate.v2" and
             batch["schema"] == "measurement-record-batch.v2", "version mismatch")
    binding, uncertainty = batch["calibration"], batch["uncertainty"]
    _require(binding == request["inputs"]["calibration"] and binding["schema"] == "rci-calibration-binding.v2",
             "calibration binding mismatch")
    basis = binding["covariance_basis"]
    _require(uncertainty["covariance_basis"] == basis, "uncertainty basis differs from calibration")
    basis_digest = _digest(basis)
    _require(batch["covariance_basis_digest"] == basis_digest == uncertainty["covariance_basis_digest"],
             "native covariance basis digest mismatch")
    summary = _basis_summary(basis)
    coverage = uncertainty["covariance_coverage"]
    _require(coverage == summary and coverage.get("completeness_claimed") is False,
             "covariance coverage contradicts declared components")
    for name in ("dependency_ids", "shared_source_ids"):
        _require(uncertainty[name] == summary[name], "uncertainty dependency summary mismatch")

    # Parse only the retained declaration; no RCI module or numerical transform.
    calibration = tomllib.loads(request["inputs"]["assembly_toml"])["calibration"]
    theta = calibration["theta"]
    _require(isinstance(theta, list) and len(theta) == 2, "native parameter values must have two entries")
    for value in theta:
        _number(value)
    units = [calibration["output_unit"].strip() + "/" + calibration["raw_unit"].strip(), calibration["raw_unit"].strip()]
    parameterization = uncertainty["parameterization"]
    expected = {"name": "scale_zero_raw", "order": ["scale", "zero_raw"], "values": theta, "units": units}
    _require(parameterization == expected, "native scale/zero parameterization mismatch")
    for value in parameterization["values"]:
        _number(value)
    _require(uncertainty["parameter_order"] == binding["parameter_order"] == ["scale", "zero_raw"]
             and uncertainty["parameter_units"] == units, "parameter order or units mismatch")
    _matrix(binding["parameter_covariance"])
    _matrix(uncertainty["parameter_covariance"])
    _require(uncertainty["parameter_covariance"] == binding["parameter_covariance"], "parameter covariance source mismatch")
    _require(uncertainty["raw_covariance"] == request["inputs"]["raw_covariance"], "raw covariance source mismatch")
    _require(uncertainty["raw_parameter_independent"] is True and binding["raw_parameter_independent"] is True
             and uncertainty["residual_correlation"] == binding["residual_correlation"] == "independent"
             and uncertainty["parameter_correlation_across_records"] == "shared_profile"
             and uncertainty["traceability"] == "none_claimed", "uncertainty scope or independence was promoted")

    start, end = _timestamp(binding["valid_from"]), _timestamp(binding["valid_until"])
    _require(start < end, "invalid calibration validity interval")
    records, requested = batch["records"], request["inputs"]["records"]
    _require(isinstance(records, list) and bool(records) and len(records) == len(requested), "record count mismatch")
    for index, (record, source) in enumerate(zip(records, requested)):
        _require(record["schema"] == "measurement-record.v2" and record["covariance_basis_digest"] == basis_digest,
                 "record basis commitment mismatch")
        _require(record["observed_at"] == source["observed_at"] and type(record["covariance_row"]) is int
                 and record["covariance_row"] == index, "observation ordering mismatch")
        expected = {"observed_at": record["observed_at"], "applicable": True,
                    "valid_from": binding["valid_from"], "valid_until": binding["valid_until"],
                    "basis": "caller_declared_acquisition_time"}
        applicability = record["acquisition_applicability"]
        _require(applicability == expected and applicability.get("applicable") is True
                 and start <= _timestamp(record["observed_at"]) < end, "historical acquisition applicability mismatch")
        _require(record["observation_time_basis"] == "caller_declared_acquisition_time"
                 and record["kind"] == "calibrated_observation"
                 and record["claim_scope"] == "declared_calibration_only", "measurement scope was promoted")
