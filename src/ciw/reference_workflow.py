"""Shared lifecycle for provider-free Python reference workflows.

A reference workflow retains one exact source, executes an independent Python
reference in this process and records the outcome with separate operation,
execution, result and numerical-result identities.  Reopening a workspace
validates the retained bundle against the deterministic reference check the
workflow declares without executing a provider; replay records a fresh
occurrence whose numerical identity must match the original.

The thermal, machine-manifest, project-graph, uncertainty-validation and
energy-accuracy workflows subclass ``ReferenceWorkflow``.  Each supplies its
constants and four hooks: how to validate exact source bytes, how to compute
the native data, which runtime identity it publishes, and which configuration a
bundle retains.  Three further hooks have defaults: the experiment identity a
bundle records, the request a step retains, and the shape check on a retained
runtime identity.  Everything identity-critical lives here once.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import re
import uuid

from .exchange import _identity
from .telemetry import _bundle_digest, _now, byte_digest, canonical, digest, _keys

RUNTIME_SCHEMA = "ciw.python-reference-runtime.v1"
EXECUTION_SCOPE = "independent_python_reference_only"
METHOD = "same_python_reference_fresh_occurrence_reproduction"
EXECUTION_ID = re.compile(r"execution-[a-f0-9]{32}")
SESSION_ID = re.compile(r"session-[a-f0-9]{32}")
CODE_DIGEST = re.compile(r"[a-f0-9]{64}")
BUNDLE_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
FIXED_ALGORITHM_KEYS = {"profile", "code_sha256", "source_normalization"}
STEP_KEYS = {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
             "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"}
RESULT_KEYS = {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"}
VERIFICATION_KEYS = {"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                     "reproduction", "authority", "verification_id"}
RECEIPT_KEYS = {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification",
                "admission", "replay_id"}
BUNDLE_KEYS = {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps",
               "bundle_digest", "verification"}


def _text(value, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Require bounded nonempty text")


def algorithm_identity(profile, files, **versions):
    """Content identity of the reference's own source files, normalized to LF."""
    content = b"\0".join(
        path.name.encode("utf-8") + b"\0" +
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        for path in files
    )
    return {"profile": profile, "code_sha256": sha256(content).hexdigest(),
            "source_normalization": "utf8_lf", **versions}


class ReferenceWorkflow:
    """Subclasses set the class constants and implement the hooks below."""

    kind = None
    schema = None
    SOURCE_SCHEMA = None
    result_schema = None
    verify_schema = None
    operation = None
    role = None
    label = "reference"
    method = METHOD
    ROLES = set()
    MAX_BYTES = 0
    AUTHORITY = {}

    # ------------------------------------------------------------------ hooks
    def _source(self, raw):
        """Validate exact source bytes and return the parsed source."""
        raise NotImplementedError

    def _native_data(self, source):
        """Compute the deterministic native data for a validated source."""
        raise NotImplementedError

    def _runtime_identity(self):
        """The runtime identity this reference publishes, resolved at call time."""
        raise NotImplementedError

    def _configuration(self, source):
        """The configuration a bundle retains for this source."""
        raise NotImplementedError

    def _check_data(self, result, source):
        """Refuse a retained result that the reference does not reproduce exactly."""
        if canonical(result["data"]) != canonical(self._native_data(source)):
            raise ValueError(f"{self._label} native result differs from the deterministic reference")

    def _experiment_id(self, source):
        """The experiment identity a bundle records for a validated source."""
        return source["experiment_id"]

    def _request(self, source, evidence_id):
        """The request a step retains; by default the source's own request."""
        return deepcopy(source["request"])

    # -------------------------------------------------------------- helpers
    @property
    def _label(self):
        return self.label[:1].upper() + self.label[1:]

    @property
    def replay_schema(self):
        return "ciw." + self.kind + "-replay.v1"

    @staticmethod
    def _runtime_projection(value):
        return deepcopy(value)

    def _adapters(self, repositories, expected=None):
        if not isinstance(repositories, dict) or repositories:
            raise ValueError(f"The provider-free {self.label} reference accepts no repository bindings")
        runtime = self._runtime_identity()
        if expected is not None:
            expected_runtime = expected.get(self.role, expected) if isinstance(expected, dict) else expected
            if self._runtime_projection(runtime) != self._runtime_projection(expected_runtime):
                raise ValueError(f"{self._label} reference runtime identity differs from the retained execution")
        return None, runtime

    def _bundle_source(self, source, raw):
        evidence = byte_digest(raw)
        return {"experiment_id": self._experiment_id(source), "experiment_digest": digest(source),
                "evidence": [{"artifact_ref": evidence, "sha256": evidence,
                              "bytes_b64": base64.b64encode(raw).decode()}]}

    # ---------------------------------------------------------- occurrences
    def _step(self, source, evidence_id, execution_id=None):
        occurrence = execution_id or "execution-" + uuid.uuid4().hex
        data = self._native_data(source)
        result = {
            "schema": self.result_schema,
            "operation_id": self.operation,
            "execution_ref": occurrence,
            "input_refs": [evidence_id],
            "data": data,
            "authority": deepcopy(self.AUTHORITY),
        }
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": deepcopy(data)}
        request = self._request(source, evidence_id)
        return {
            "runtime_ref": self.role,
            "operation_id": self.operation,
            "execution_id": occurrence,
            "input_refs": [evidence_id],
            "request": request,
            "request_sha256": digest(request),
            "result": result,
            "result_sha256": digest(result),
            "result_id": result["result_id"],
            "numerical_result": numerical,
            "numerical_result_id": digest(numerical),
        }

    def _validate_step(self, step, source, evidence_id):
        _keys(step, STEP_KEYS)
        request = self._request(source, evidence_id)
        if (step["runtime_ref"] != self.role or step["operation_id"] != self.operation or
                not isinstance(step["execution_id"], str) or not EXECUTION_ID.fullmatch(step["execution_id"]) or
                step["input_refs"] != [evidence_id] or
                canonical(step["request"]) != canonical(request) or
                step["request_sha256"] != digest(request)):
            raise ValueError(f"{self._label} step request or occurrence binding differs")
        result = step["result"]
        _keys(result, RESULT_KEYS)
        if (result["schema"] != self.result_schema or result["operation_id"] != self.operation or
                result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence_id] or
                result["authority"] != self.AUTHORITY or
                result["result_id"] != digest({key: value for key, value in result.items() if key != "result_id"})):
            raise ValueError(f"{self._label} native result envelope differs")
        self._check_data(result, source)
        numerical = {"operation_id": self.operation, "data": result["data"]}
        if step["numerical_result"] != numerical or step["numerical_result_id"] != digest(numerical):
            raise ValueError(f"{self._label} numerical result identity differs")
        if step["result_id"] != result["result_id"] or step["result_sha256"] != digest(result):
            raise ValueError(f"{self._label} step result commitment differs")

    def _artifact(self, subject_ref, runtimes, reproduced):
        """The verification artifact binding a reproduction to a subject bundle and its runtimes."""
        value = {
            "schema": self.verify_schema,
            "subject_ref": subject_ref,
            "outcome": "passed",
            "independent": False,
            "method": self.method,
            "runtime_digest": digest(runtimes),
            "reproduction": deepcopy(reproduced),
            "authority": deepcopy(self.AUTHORITY),
        }
        value["verification_id"] = byte_digest(self.verify_schema.encode() + b"\0" + canonical(value))
        return value

    def _verification(self, bundle, reproduced):
        if canonical(bundle["steps"][0]["numerical_result"]) != canonical(reproduced["numerical_result"]):
            raise ValueError(f"{self._label} replay numerical result differs")
        return self._artifact(bundle["bundle_digest"], bundle["runtimes"], reproduced)

    def _check_verification(self, bundle, verification, source, evidence):
        _keys(verification, VERIFICATION_KEYS)
        self._validate_step(verification["reproduction"], source, evidence)
        old, new = bundle["steps"][0], verification["reproduction"]
        # Recomputing the artifact pins its schema, subject, outcome, method,
        # runtime digest and authority at once.
        if (old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or
                verification != self._verification(bundle, new)):
            raise ValueError(f"{self._label} verification must bind a fresh occurrence")
        _identity(verification, "verification_id")

    def _check_runtime(self, runtime):
        """A retained identity keeps the current shape and fixed fields; replay compares it whole."""
        expected = self._runtime_identity()
        _keys(runtime, set(expected))
        if any(runtime[key] != expected[key] for key in expected if key != "algorithm"):
            raise ValueError(f"Unapproved {self.label} reference runtime")
        algorithm, reference = runtime["algorithm"], expected["algorithm"]
        _keys(algorithm, set(reference))
        if (algorithm["profile"] != reference["profile"] or algorithm["source_normalization"] != "utf8_lf" or
                not isinstance(algorithm["code_sha256"], str) or not CODE_DIGEST.fullmatch(algorithm["code_sha256"]) or
                any(not isinstance(algorithm[key], str) or not 1 <= len(algorithm[key]) <= 64
                    for key in reference if key not in FIXED_ALGORITHM_KEYS)):
            raise ValueError(f"Malformed {self.label} reference algorithm identity")

    # ------------------------------------------------------------ lifecycle
    def _validate(self, bundle):
        """Validate a retained bundle; only the deterministic reference check runs."""
        try:
            _keys(bundle, BUNDLE_KEYS, {"replay_receipts"})
            if (len(canonical(bundle)) > self.MAX_BYTES or bundle["schema"] != self.schema or
                    bundle["bundle_digest"] != _bundle_digest(bundle) or
                    not isinstance(bundle["session_id"], str) or not SESSION_ID.fullmatch(bundle["session_id"])):
                raise ValueError(f"{self._label} bundle identity or size differs")
            _text(bundle["created_at"])
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"], validate=True)
            source = self._source(raw)
            if (bundle["source"] != self._bundle_source(source, raw) or
                    canonical(bundle["configuration"]) != canonical(self._configuration(source))):
                raise ValueError(f"{self._label} source or configuration binding differs")
            _keys(bundle["runtimes"], {self.role})
            self._check_runtime(bundle["runtimes"][self.role])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle, bundle["verification"], source, evidence["artifact_ref"])
            receipts = bundle.get("replay_receipts", [])
            if not isinstance(receipts, list) or len(receipts) > 1:
                raise ValueError(f"At most one {self.label} replay receipt belongs to an occurrence")
            for receipt in receipts:
                _keys(receipt, RECEIPT_KEYS)
                if (receipt["schema"] != self.replay_schema or
                        receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        not isinstance(receipt["source_bundle_digest"], str) or
                        not BUNDLE_DIGEST.fullmatch(receipt["source_bundle_digest"]) or
                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or
                        receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                        receipt["replay_id"] != digest({key: value for key, value in receipt.items() if key != "replay_id"})):
                    raise ValueError(f"Invalid {self.label} replay receipt")
                # A receipt's verification reproduces exactly this bundle's
                # occurrence against the source bundle and the shared runtimes;
                # recomputing the artifact pins every field and its identity.
                if receipt["verification"] != self._artifact(receipt["source_bundle_digest"], bundle["runtimes"], step):
                    raise ValueError(f"Invalid {self.label} replay verification scope")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError, UnicodeError) as exc:
            raise ValueError(f"Malformed retained {self.label} session") from exc

    def _execute(self, raw, runtime):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = self._step(source, evidence)
        bundle = {
            "schema": self.schema,
            "session_id": "session-" + uuid.uuid4().hex,
            "created_at": _now(),
            "source": self._bundle_source(source, raw),
            "configuration": self._configuration(source),
            "runtimes": {self.role: deepcopy(runtime)},
            "steps": [step],
        }
        bundle["bundle_digest"] = _bundle_digest(bundle)
        bundle["verification"] = self._verification(bundle, self._step(source, evidence))
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories)[1])

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        runtime = self._adapters(repositories, bundle["runtimes"])[1]
        fresh = self._execute(raw, runtime)
        receipt = {
            "schema": self.replay_schema,
            "source_bundle_digest": bundle["bundle_digest"],
            "replayed_bundle_digest": fresh["bundle_digest"],
            "numerical_match": True,
            "verification": self._verification(bundle, fresh["steps"][0]),
            "admission": "not_performed",
        }
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        self._validate(fresh)
        return {"session": fresh, "replay_receipt": receipt}
