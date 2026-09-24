"""Provider-free CIW adapter for the evidence-bound machine manifest.

The machine-manifest contract is deliberately narrower than a commissioning
system.  It accepts a retained evidence bundle, a candidate manifest and its
deterministic challenge report, then evaluates one encoder position under the
declared kinematic model.  The operation never retrieves a document, opens a
device, loads firmware, admits state or authorizes actuation.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from . import machine_manifest as manifest
from . import reference_workflow as base
from .telemetry import canonical, _keys

KIND = "machine-manifest"
SCHEMA = "ciw.machine-manifest-session.v1"
SOURCE_SCHEMA = "ciw.machine-manifest-source.v1"
DATA_SCHEMA = "ciw.encoder-position-workbench-data.v1"
RESULT_SCHEMA = "ciw.encoder-position-workbench-result.v1"
VERIFY_SCHEMA = "ciw.machine-manifest-verification.v1"
OPERATION = manifest.OPERATION
PROFILE = "ciw.machine-manifest.python-reference.v1"
ROLE = "machine"
ROLES = set()
# A retained bundle embeds the source as base64 plus the inspection, its
# numerical copy and one reproduction, so the bundle budget exceeds the source
# budget by the same factor the project graph workflow uses.
SOURCE_LIMIT = manifest.MAX_BYTES
MAX_BYTES = 8 * SOURCE_LIMIT
CONFIGURATION = {
    "profile": "encoder_gearbox_leadscrew",
    "activation": "read_only",
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}
AUTHORITY = {
    "physical_validation": "not_performed",
    "state_admission": "not_performed",
    "hardware_actuation": "not_performed",
}
CLAIM_SCOPE = "position_and_local_linearized_uncertainty_under_declared_kinematic_model"


_text = base._text


@lru_cache(maxsize=1)
def _algorithm_identity():
    return base.algorithm_identity(PROFILE, [Path(manifest.__file__), Path(base.__file__), Path(__file__)],
                                   numpy_version=np.__version__)


def runtime_identity():
    return {
        "schema": base.RUNTIME_SCHEMA,
        "role": ROLE,
        "profile": PROFILE,
        "algorithm": _algorithm_identity(),
        "execution_scope": base.EXECUTION_SCOPE,
        "physical_validation": "not_performed",
        "state_admission": "not_performed",
        "hardware_actuation": "not_performed",
    }


def _request(value):
    _keys(value, {"counts", "covariance"})
    if type(value["counts"]) is not int or abs(value["counts"]) > 2**53 - 1:
        raise ValueError("Decoded count must be an exactly retained bounded integer")
    if value["covariance"] is not None:
        manifest._covariance(value["covariance"])
    return deepcopy(value)


def validate_source(raw):
    """Validate exact source bytes and all deterministic upstream artifacts."""
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Machine manifest source requires bounded exact JSON bytes")
    source = base.parse_json(raw, "Machine manifest source")
    _keys(source, {"schema", "experiment_id", "configuration", "evidence_bundle",
                   "candidate_manifest", "challenge_report", "request"})
    if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(CONFIGURATION):
        raise ValueError("Unsupported machine manifest source or authority policy")
    _text(source["experiment_id"], 128)
    evidence = source["evidence_bundle"]
    candidate = source["candidate_manifest"]
    report = source["challenge_report"]
    manifest.validate(evidence)
    manifest.validate(candidate)
    manifest.validate(report)
    if (evidence["schema"] != manifest.EVIDENCE_SCHEMA or
            candidate["schema"] != manifest.CANDIDATE_SCHEMA or
            report["schema"] != manifest.CHALLENGE_SCHEMA):
        raise ValueError("Machine source must retain evidence, candidate and challenge artifacts")
    if (candidate["machine_id"] != evidence["machine_id"] or
            report["machine_id"] != evidence["machine_id"] or
            candidate["evidence_bundle_digest"] != evidence["artifact_digest"] or
            report["candidate_digest"] != candidate["artifact_digest"] or
            report["evidence_bundle_digest"] != evidence["artifact_digest"]):
        raise ValueError("Machine source artifact identities are not linked")
    _compiled(source)
    _request(source["request"])
    if len(canonical(source)) > SOURCE_LIMIT:
        raise ValueError("Machine manifest source exceeds the byte budget")
    return deepcopy(source)


@lru_cache(maxsize=8)
def _compile_exact(candidate, evidence, report):
    """Compile once per exact artifact triple; the compiler is pure and revalidates everything.

    Keyed by canonical bytes so the cache is safe under the session's worker threads.
    """
    return manifest.compile(json.loads(candidate), json.loads(evidence), json.loads(report))


def _compiled(source):
    return deepcopy(_compile_exact(canonical(source["candidate_manifest"]), canonical(source["evidence_bundle"]),
                                   canonical(source["challenge_report"])))


def _native_data(source):
    compiled = _compiled(source)
    request = source["request"]
    position = manifest.evaluate(compiled, request["counts"], request["covariance"])
    return {
        "schema": DATA_SCHEMA,
        "operation_id": OPERATION,
        "machine_id": compiled["machine_id"],
        "compiled_manifest_digest": compiled["artifact_digest"],
        "manifest": manifest.inspect(compiled),
        "request": deepcopy(request),
        "position": position,
        "claim_scope": CLAIM_SCOPE,
        "authority": deepcopy(AUTHORITY),
    }


class MachineManifestWorkflow(base.ReferenceWorkflow):
    kind = KIND
    schema = SCHEMA
    SOURCE_SCHEMA = SOURCE_SCHEMA
    result_schema = RESULT_SCHEMA
    verify_schema = VERIFY_SCHEMA
    operation = OPERATION
    role = ROLE
    label = "machine"
    ROLES = ROLES
    MAX_BYTES = MAX_BYTES
    AUTHORITY = AUTHORITY

    def _source(self, raw):
        return validate_source(raw)

    def _native_data(self, source):
        return _native_data(source)

    def _runtime_identity(self):
        return runtime_identity()

    def _configuration(self, source):
        return deepcopy(CONFIGURATION)


def _verification(bundle, reproduced):
    return MachineManifestWorkflow()._verification(bundle, reproduced)
