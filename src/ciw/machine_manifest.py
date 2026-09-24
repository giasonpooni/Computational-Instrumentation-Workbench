"""Evidence-bound, read-only encoder/gearbox/leadscrew manifest compilation.

The three authoring roles retain proposals and challenges. Deterministic local
validation accepts only a bounded declared model. It authenticates neither the
documents nor the physical machine, and never commands a device or loads code.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import re

import numpy as np

from .core.canonical import canonical, digest

EVIDENCE_SCHEMA = "ciw.machine-evidence.v1"
CANDIDATE_SCHEMA = "ciw.machine-candidate.v1"
CHALLENGE_SCHEMA = "ciw.machine-challenge.v1"
COMPILED_SCHEMA = "ciw.compiled-machine.v1"
OPERATION = "ciw.encoder-position.v1"
VALIDATOR = "ciw.machine-manifest-validator.v1"
COORDINATES = ["N", "N0", "x0", "L", "C", "g"]
CLAIMS = {"C", "g", "L", "s", "N0", "x0", "count_basis", "firmware", "homing", "frame", "time_basis"}
STATUSES = {"documented", "observed", "validated", "unresolved"}
MAX_BYTES = 1024 * 1024
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CLAIM_SCOPE = "declared_evidence_consistency_and_algebra_not_independent_physical_validation"


def _keys(value, expected, what):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Malformed " + what)


def _id(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("Invalid stable identifier")


def _hash(value):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise ValueError("Invalid SHA256 commitment")


def _text(value, limit=512):
    if type(value) is not str or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


def _number(value, maximum=1e24):
    if type(value) not in (int, float) or abs(value) > maximum or not math.isfinite(value):
        raise ValueError("Require a bounded finite number")
    return float(value)


def _refs(value, allow_empty=False):
    if type(value) is not list or len(value) > 32 or (not value and not allow_empty):
        raise ValueError("Require bounded evidence references")
    for ref in value:
        _id(ref)
    if len(set(value)) != len(value):
        raise ValueError("Repeated evidence reference")


def seal(unsigned):
    """Content commitment only; call validate or compile for contract checks."""
    if type(unsigned) is not dict or "artifact_digest" in unsigned:
        raise ValueError("Require an unsigned artifact mapping")
    value = deepcopy(unsigned)
    if len(canonical(value)) > MAX_BYTES - 100:
        raise ValueError("Machine artifact exceeds byte bound")
    value["artifact_digest"] = digest(value)
    return value


def _commitment(value):
    if type(value) is not dict or "artifact_digest" not in value or len(canonical(value)) > MAX_BYTES:
        raise ValueError("Invalid or oversized machine artifact")
    _hash(value["artifact_digest"])
    if value["artifact_digest"] != digest({key: item for key, item in value.items() if key != "artifact_digest"}):
        raise ValueError("Machine artifact digest mismatch")


def _evidence(value):
    _keys(value, {"schema", "role", "machine_id", "origin", "sources", "evidence", "artifact_digest"}, "evidence bundle")
    if value["schema"] != EVIDENCE_SCHEMA or value["role"] != "evidence_bundle" or value["origin"] not in {"synthetic_fixture", "operator_record"}:
        raise ValueError("Invalid evidence role or origin")
    _id(value["machine_id"])
    if type(value["sources"]) is not list or not 1 <= len(value["sources"]) <= 32:
        raise ValueError("Evidence bundle requires bounded retained sources")
    sources, source_hashes = {}, set()
    for source in value["sources"]:
        _keys(source, {"source_id", "machine_id", "kind", "locator", "content_utf8", "source_digest"}, "retained source")
        _id(source["source_id"])
        _id(source["machine_id"])
        _text(source["locator"], 2048)
        _text(source["content_utf8"], 16384)
        if source["kind"] not in {"documentation", "observation", "validation", "firmware_configuration"}:
            raise ValueError("Unknown retained source kind")
        actual = "sha256:" + sha256(source["content_utf8"].encode("utf-8")).hexdigest()
        if source["source_digest"] != actual:
            raise ValueError("Retained source bytes do not match source digest")
        if source["source_id"] in sources or actual in source_hashes:
            raise ValueError("Reused source identifier or duplicate provenance bytes")
        sources[source["source_id"]] = source
        source_hashes.add(actual)
    if type(value["evidence"]) is not list or not 1 <= len(value["evidence"]) <= 64:
        raise ValueError("Evidence bundle requires bounded claim records")
    seen = set()
    for item in value["evidence"]:
        _keys(item, {"evidence_id", "machine_id", "source_id", "source_digest", "status", "claims"}, "evidence item")
        for key in ("evidence_id", "machine_id", "source_id"):
            _id(item[key])
        _hash(item["source_digest"])
        if item["evidence_id"] in seen:
            raise ValueError("Reused evidence identifier")
        seen.add(item["evidence_id"])
        if item["status"] not in STATUSES or type(item["claims"]) is not dict or not item["claims"] or not set(item["claims"]) <= CLAIMS | {"uncertainty"}:
            raise ValueError("Invalid evidence claims or status")
        for claim in item["claims"].values():
            _hash(claim)
    return sources, {item["evidence_id"]: item for item in value["evidence"]}


def _covariance(value):
    if type(value) is not list or len(value) != 6 or any(type(row) is not list or len(row) != 6 for row in value):
        raise ValueError("Joint covariance must be a 6 by 6 matrix")
    matrix = np.asarray([[_number(item) for item in row] for row in value], dtype=np.float64)
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) < 0):
        raise ValueError("Joint covariance must be symmetric with nonnegative diagonal")
    positive = np.diag(matrix) > 0
    if np.any(matrix[~positive, :] != 0):
        raise ValueError("Zero-variance coordinates must have zero covariances")
    normalized = matrix[np.ix_(positive, positive)]
    if normalized.size:
        scale = np.sqrt(np.diag(normalized))
        normalized = normalized / scale[:, None] / scale[None, :]
        # A fixed correlation-scale rounding bound, invariant to mm/m units.
        if np.linalg.eigvalsh(normalized)[0] < -64 * np.finfo(float).eps * len(normalized):
            raise ValueError("Joint covariance is not positive semidefinite")
    return matrix


def _uncertainty(value, length_unit):
    _keys(value, {"method", "coordinates", "units", "covariance", "basis", "distribution", "confidence_level", "status", "evidence_refs"}, "uncertainty declaration")
    if value["method"] != "first_order_joint_covariance" or value["coordinates"] != COORDINATES:
        raise ValueError("Unsupported uncertainty propagation contract")
    units = ["count", "count", length_unit, length_unit + "/rev_output", "count/rev_motor", "rev_motor/rev_output"]
    if value["units"] != units:
        raise ValueError("Joint covariance coordinate units differ from manifest")
    if value["basis"] != "declared_prior_covariance" or value["distribution"] != "unspecified" or value["confidence_level"] is not None:
        raise ValueError("This profile supports declared prior covariance, not probability coverage claims")
    if value["status"] not in STATUSES:
        raise ValueError("Unknown uncertainty evidence status")
    _refs(value["evidence_refs"], allow_empty=value["status"] == "unresolved")
    if value["covariance"] is None and value["status"] == "unresolved":
        return
    _covariance(value["covariance"])


def _candidate(value):
    _keys(value, {"schema", "role", "machine_id", "status", "claims", "uncertainty", "evidence_bundle_digest", "artifact_digest"}, "candidate manifest")
    if value["schema"] != CANDIDATE_SCHEMA or value["role"] != "candidate_manifest" or value["status"] != "candidate":
        raise ValueError("Manifest must remain a candidate until deterministic compilation")
    _id(value["machine_id"])
    _hash(value["evidence_bundle_digest"])
    _keys(value["claims"], CLAIMS, "candidate claims")
    for key, claim in value["claims"].items():
        _keys(claim, {"value", "unit", "status", "evidence_refs"}, "candidate claim")
        if claim["status"] not in STATUSES:
            raise ValueError("Unknown claim status")
        _refs(claim["evidence_refs"], allow_empty=claim["status"] == "unresolved")
        if claim["value"] is None and claim["status"] == "unresolved":
            continue
        item = claim["value"]
        if key == "C":
            if type(item) is not int or not 1 <= item <= 10**9:
                raise ValueError("Decoded counts per motor revolution must be a positive integer")
        elif key == "g":
            if not 1e-6 <= _number(item) <= 1e6:
                raise ValueError("Motor/output gear ratio exceeds the bounded model")
        elif key == "L":
            if not 1e-9 <= _number(item) <= 1e6:
                raise ValueError("Lead must be positive and bounded")
        elif key == "s":
            if type(item) is not int or item not in (-1, 1):
                raise ValueError("Orientation must be exactly +1 or -1")
        elif key == "N0":
            if type(item) is not int or abs(item) > 2**53 - 1:
                raise ValueError("Reference count must be an exactly retained integer")
        elif key == "x0":
            _number(item, 1e9)
        elif key == "firmware":
            _keys(item, {"id", "configuration_digest"}, "firmware declaration")
            _text(item["id"], 128)
            _hash(item["configuration_digest"])
        elif key == "homing":
            _keys(item, {"method", "reference_established"}, "homing declaration")
            _text(item["method"], 128)
            if type(item["reference_established"]) is not bool:
                raise ValueError("Homing reference status must be boolean")
        else:
            _text(item, 128)
    length_unit = value["claims"]["x0"]["unit"]
    if length_unit not in {"m", "mm"}:
        raise ValueError("Only explicit m or mm position units are supported")
    units = {"C": "count/rev_motor", "g": "rev_motor/rev_output", "L": length_unit + "/rev_output", "s": "1", "N0": "count", "x0": length_unit}
    for key, claim in value["claims"].items():
        if claim["unit"] != units.get(key):
            raise ValueError("Claim has incompatible units: " + key)
    _uncertainty(value["uncertainty"], length_unit)


def _report(value):
    _keys(value, {"schema", "role", "machine_id", "validator", "candidate_digest", "evidence_bundle_digest", "status", "findings", "claim_scope", "artifact_digest"}, "challenge report")
    if value["schema"] != CHALLENGE_SCHEMA or value["role"] != "challenge_report" or value["validator"] != VALIDATOR or value["claim_scope"] != _CLAIM_SCOPE:
        raise ValueError("Unsupported deterministic challenge contract")
    _id(value["machine_id"])
    _hash(value["candidate_digest"])
    _hash(value["evidence_bundle_digest"])
    if value["status"] not in {"validated", "unresolved"} or type(value["findings"]) is not list or len(value["findings"]) > 128:
        raise ValueError("Invalid challenge report status or findings")
    for finding in value["findings"]:
        _keys(finding, {"code", "subject", "status", "evidence_refs"}, "challenge finding")
        _id(finding["code"])
        _text(finding["subject"], 128)
        if finding["status"] != "unresolved":
            raise ValueError("Challenge findings must remain unresolved until rerun")
        _refs(finding["evidence_refs"], allow_empty=True)
    if (value["status"] == "validated") != (not value["findings"]):
        raise ValueError("Challenge status disagrees with its unresolved findings")


def validate(value):
    """Validate retained structure and commitments; never fetch linked sources."""
    try:
        _commitment(value)
        schema = value.get("schema")
        if schema == EVIDENCE_SCHEMA:
            _evidence(value)
        elif schema == CANDIDATE_SCHEMA:
            _candidate(value)
        elif schema == CHALLENGE_SCHEMA:
            _report(value)
        elif schema == COMPILED_SCHEMA:
            _keys(value, {"schema", "role", "machine_id", "status", "operation", "formula", "claim_scope", "inputs", "artifact_digest"}, "compiled manifest")
            _keys(value["inputs"], {"candidate_manifest", "evidence_bundle", "challenge_report"}, "compiled inputs")
            expected = compile(value["inputs"]["candidate_manifest"], value["inputs"]["evidence_bundle"], value["inputs"]["challenge_report"])
            if canonical(value) != canonical(expected):
                raise ValueError("Compiled manifest differs from deterministic compilation")
        else:
            raise ValueError("Unsupported machine artifact schema")
    except (KeyError, TypeError, OverflowError, RecursionError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed machine artifact") from exc
    return deepcopy(value)


def _claim_value(claim):
    return {"value": claim["value"], "unit": claim["unit"]}


def _uncertainty_value(declaration):
    return {key: value for key, value in declaration.items() if key not in {"status", "evidence_refs"}}


def challenge(candidate, evidence_bundle):
    validate(candidate)
    validate(evidence_bundle)
    if candidate["schema"] != CANDIDATE_SCHEMA or evidence_bundle["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("Challenge requires a candidate and evidence bundle")
    sources, evidence = _evidence(evidence_bundle)
    findings = []
    def gap(code, subject, refs=()):
        findings.append({"code": code, "subject": subject, "status": "unresolved", "evidence_refs": list(refs)})
    if candidate["machine_id"] != evidence_bundle["machine_id"]:
        gap("machine_identity_mismatch", "machine_id")
    if candidate["evidence_bundle_digest"] != evidence_bundle["artifact_digest"]:
        gap("evidence_revision_changed", "evidence_bundle_digest")
    claims = candidate["claims"]
    for key, claim in list(claims.items()) + [("uncertainty", candidate["uncertainty"])]:
        refs = claim["evidence_refs"]
        if claim["status"] == "unresolved":
            gap("claim_unresolved", key, refs)
        expected = digest(_uncertainty_value(claim) if key == "uncertainty" else _claim_value(claim))
        for ref in refs:
            item = evidence.get(ref)
            if item is None:
                gap("missing_evidence_reference", key, [ref])
                continue
            source = sources.get(item["source_id"])
            if source is None:
                gap("missing_retained_source", key, [ref])
                continue
            if item["machine_id"] != candidate["machine_id"] or source["machine_id"] != candidate["machine_id"]:
                gap("reused_provenance_other_machine", key, [ref])
            if item["source_digest"] != source["source_digest"]:
                gap("source_revision_changed", key, [ref])
            if item["status"] == "unresolved" or item["status"] != claim["status"]:
                gap("evidence_status_mismatch", key, [ref])
            if item["claims"].get(key) != expected:
                gap("claim_not_bound_to_evidence", key, [ref])
            if key == "firmware" and (source["kind"] != "firmware_configuration" or
                    claim["value"] is None or claim["value"]["configuration_digest"] != source["source_digest"]):
                gap("firmware_configuration_unbound", key, [ref])
    if claims["count_basis"]["value"] != "firmware_decoded_counts":
        gap("ambiguous_encoder_count_basis", "count_basis", claims["count_basis"]["evidence_refs"])
    if claims["homing"]["value"] is None or claims["homing"]["value"]["reference_established"] is not True:
        gap("homing_reference_not_established", "homing", claims["homing"]["evidence_refs"])
    return seal({"schema": CHALLENGE_SCHEMA, "role": "challenge_report", "machine_id": candidate["machine_id"],
        "validator": VALIDATOR, "candidate_digest": candidate["artifact_digest"],
        "evidence_bundle_digest": evidence_bundle["artifact_digest"], "status": "unresolved" if findings else "validated",
        "findings": findings, "claim_scope": _CLAIM_SCOPE})


def compile(candidate, evidence_bundle, challenge_report=None):
    """Rerun validation; a previous challenge cannot bless changed source bytes."""
    report = challenge(candidate, evidence_bundle)
    if challenge_report is not None:
        validate(challenge_report)
        if canonical(report) != canonical(challenge_report):
            raise ValueError("Challenge report is stale or differs from deterministic validation; rerun it")
    if report["status"] != "validated":
        raise ValueError("Unresolved machine manifest: " + ", ".join(sorted({item["code"] for item in report["findings"]})))
    return seal({"schema": COMPILED_SCHEMA, "role": "compiled_manifest", "machine_id": candidate["machine_id"],
        "status": "accepted_read_only", "operation": OPERATION, "formula": "x=x0+s*(N-N0)*L/(C*g)",
        "claim_scope": _CLAIM_SCOPE, "inputs": {"candidate_manifest": deepcopy(candidate),
        "evidence_bundle": deepcopy(evidence_bundle), "challenge_report": report}})


def evaluate(compiled, counts, covariance=None):
    """Evaluate the fixed binder and its first-order *joint* covariance.

    The optional covariance is a new explicit observation input, not a changed
    acceptance threshold. All six coordinates and their cross terms survive in
    the result. No confidence interval or coverage probability is inferred.
    """
    validate(compiled)
    if compiled["schema"] != COMPILED_SCHEMA:
        raise ValueError("Evaluation requires a compiled read-only manifest")
    if type(counts) is not int or abs(counts) > 2**53 - 1:
        raise ValueError("Decoded count must be an exactly retained bounded integer")
    candidate = compiled["inputs"]["candidate_manifest"]
    claims = candidate["claims"]
    N0, x0, L, C, g, s = [claims[key]["value"] for key in ("N0", "x0", "L", "C", "g", "s")]
    offset = counts - N0
    gain = s * L / (C * g)
    position = x0 + offset * gain
    jacobian = np.asarray([gain, -gain, 1., s * offset / (C * g), -gain * offset / C, -gain * offset / g])
    raw_covariance = candidate["uncertainty"]["covariance"] if covariance is None else covariance
    matrix = _covariance(raw_covariance)
    terms = jacobian[:, None] * matrix * jacobian[None, :]
    raw_variance = math.fsum(float(item) for item in terms.ravel())
    rounding_bound = 64 * np.finfo(float).eps * math.fsum(abs(float(item)) for item in terms.ravel())
    if raw_variance < -rounding_bound or not math.isfinite(position) or not math.isfinite(raw_variance):
        raise ValueError("Propagation produced invalid finite variance or position")
    variance = max(0., raw_variance)
    unit = claims["x0"]["unit"]
    scale = 1. if unit == "m" else .001
    return seal({"schema": "ciw.encoder-position-result.v1", "manifest_digest": compiled["artifact_digest"],
        "operation": OPERATION, "counts": counts, "frame": claims["frame"]["value"],
        "time_basis": claims["time_basis"]["value"], "semantics": "estimated", "position": position,
        "unit": unit, "position_m": position * scale, "variance": variance, "variance_unit": unit + "^2",
        "variance_m2": variance * scale**2, "standard_uncertainty": math.sqrt(variance),
        "uncertainty": {"method": "first_order_joint_covariance", "coordinates": list(COORDINATES),
            "units": deepcopy(candidate["uncertainty"]["units"]), "covariance": deepcopy(raw_covariance),
            "jacobian": jacobian.tolist(), "variance_terms": terms.tolist(), "raw_variance": raw_variance,
            "roundoff_bound": rounding_bound, "roundoff_clamped": raw_variance < 0,
            "basis": "declared_prior_covariance" if covariance is None else "explicit_evaluation_covariance",
            "distribution": "unspecified", "confidence_level": None},
        "claim_scope": "position_and_local_linearized_uncertainty_under_declared_kinematic_model",
        "physical_validation": "not_performed", "state_admission": "not_performed"})


def inspect(value):
    validate(value)
    return {"schema": "ciw.machine-inspection.v1", "artifact_schema": value["schema"], "role": value["role"],
        "machine_id": value["machine_id"], "artifact_digest": value["artifact_digest"],
        "status": value.get("status", "retained_evidence"), "findings": deepcopy(value.get("findings", [])),
        "claim_scope": _CLAIM_SCOPE, "activation": "read_only", "execution": "not_performed",
        "physical_validation": "not_performed", "state_admission": "not_performed"}
