"""Provider-free CIW adapter for the evidence-bound machine manifest.

The machine-manifest contract is deliberately narrower than a commissioning
system.  It accepts a retained evidence bundle, a candidate manifest and its
deterministic challenge report, then evaluates one encoder position under the
declared kinematic model.  The operation never retrieves a document, opens a
device, loads firmware, admits state or authorizes actuation.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from . import machine_manifest as manifest
from .adapters.subprocess import _json
from .core.canonical import canonical, exact_keys
from .pipelines import runner as _runner
from .pipelines.runner import PipelineRunner, RecordProfile, seal_step

KIND = "machine-manifest"
SCHEMA = "ciw.machine-manifest-session.v1"
SOURCE_SCHEMA = "ciw.machine-manifest-source.v1"
DATA_SCHEMA = "ciw.encoder-position-workbench-data.v1"
RESULT_SCHEMA = "ciw.encoder-position-workbench-result.v1"
VERIFY_SCHEMA = "ciw.machine-manifest-verification.v1"
OPERATION = manifest.OPERATION
ROLE = "machine"
ROLES = set()
MAX_BYTES = manifest.MAX_BYTES
SOURCE_LIMIT = manifest.MAX_BYTES
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


def _text(value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


def _algorithm_identity():
    files = [Path(manifest.__file__), Path(__file__), Path(_runner.__file__)]
    content = b"\0".join(
        path.name.encode("utf-8") + b"\0" +
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        for path in files
    )
    return {
        "profile": "ciw.machine-manifest.python-reference.v1",
        "code_sha256": sha256(content).hexdigest(),
        "source_normalization": "utf8_lf",
        "numpy_version": np.__version__,
    }


def runtime_identity():
    return {
        "schema": "ciw.python-reference-runtime.v1",
        "role": ROLE,
        "profile": "ciw.machine-manifest.python-reference.v1",
        "algorithm": _algorithm_identity(),
        "execution_scope": "independent_python_reference_only",
        "physical_validation": "not_performed",
        "state_admission": "not_performed",
        "hardware_actuation": "not_performed",
    }


def _request(value):
    exact_keys(value, {"counts", "covariance"})
    if type(value["counts"]) is not int or abs(value["counts"]) > 2**53 - 1:
        raise ValueError("Decoded count must be an exactly retained bounded integer")
    if value["covariance"] is not None:
        manifest._covariance(value["covariance"])
    return deepcopy(value)


def validate_source(raw):
    """Validate exact source bytes and all deterministic upstream artifacts."""
    if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
        raise ValueError("Machine manifest source requires bounded exact JSON bytes")
    source = _json(raw)
    exact_keys(source, {"schema", "experiment_id", "configuration", "evidence_bundle",
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
    manifest.compile(candidate, evidence, report)
    _request(source["request"])
    if len(canonical(source)) > SOURCE_LIMIT:
        raise ValueError("Machine manifest source exceeds the byte budget")
    return deepcopy(source)


def _compiled(source):
    return manifest.compile(source["candidate_manifest"], source["evidence_bundle"], source["challenge_report"])


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


class MachineManifestWorkflow(PipelineRunner):
    """The Python machine-manifest reference sealed, reproduced and replayed by the shared runner."""

    MAX_BYTES = MAX_BYTES
    LABEL = "Machine"
    PROFILE = RecordProfile(RESULT_SCHEMA, VERIFY_SCHEMA, "same_python_reference_fresh_occurrence_reproduction", AUTHORITY)

    def __init__(self):
        super().__init__(KIND, {"role": ROLE})
        self.ROLES = frozenset(ROLES)
        self.operation = OPERATION

    def parse_source(self, raw):
        return validate_source(raw)

    def step_request(self, source):
        return source["request"]

    def check_data(self, source, data):
        """The retained result must equal the deterministic evaluation of the retained source."""
        if canonical(data) != canonical(_native_data(source)):
            raise ValueError("Machine native result differs from deterministic evaluation")

    def _adapters(self, repositories, expected=None):
        if not isinstance(repositories, dict) or repositories:
            raise ValueError("The provider-free machine reference accepts no repository bindings")
        runtime = runtime_identity()
        if expected is not None:
            retained = expected.get(ROLE, expected) if isinstance(expected, dict) else expected
            if runtime != retained:
                raise ValueError("Machine reference runtime identity differs from the retained execution")
        return None, runtime, None

    def _step(self, source, evidence_id, bound):
        data = _native_data(source)
        return seal_step(ROLE, self.operation, self.step_request(source), [evidence_id], data, profile=self.PROFILE)

    def _check_runtimes(self, runtimes):
        exact_keys(runtimes, {ROLE})
        runtime = runtimes[ROLE]
        expected = runtime_identity()
        if runtime == expected:
            return
        # Retained runtime identity remains strict in shape and code
        # commitment; replay compares the complete current identity.
        exact_keys(runtime, set(expected))
        if any(runtime[key] != expected[key] for key in ("schema", "role", "profile", "execution_scope",
                                                          "physical_validation", "state_admission", "hardware_actuation")):
            raise ValueError("Unapproved machine reference runtime")
        algorithm = runtime["algorithm"]
        exact_keys(algorithm, {"profile", "code_sha256", "source_normalization", "numpy_version"})
        if (algorithm["profile"] != expected["algorithm"]["profile"] or
                algorithm["source_normalization"] != "utf8_lf" or
                not re.fullmatch(r"[a-f0-9]{64}", algorithm["code_sha256"]) or
                not isinstance(algorithm["numpy_version"], str)):
            raise ValueError("Malformed machine reference algorithm identity")
