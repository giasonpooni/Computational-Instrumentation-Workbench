"""Shared runner for declared pipelines.

A declared pipeline runs one pinned provider over one retained source:

    parse_source -> bind -> invoke -> check_data -> seal -> reproduce -> verify

The runner owns every record that carries an identity: the step and its
result, the session bundle, the same-runtime reproduction that verifies it and
the replay receipt. A pipeline supplies named domain hooks and nothing else:

``parse_source(raw)``
    Admit the retained source bytes or raise ``ValueError``.
``invoke(source, bound)``
    Call the pinned provider and return its native data; a provider refusal
    raises ``AdapterRefusal``.
``check_data(source, data)``
    Accept or refuse the provider's native output for this source.
``check_runtime(runtime)``
    Pin checks beyond revision, module, source root and source tree.
``make_adapter(repository, retained)``
    The subprocess adapter for the provider checkout (default: pinned).
``bind_extra(repositories, adapter, runtime)``
    Host bindings beyond the provider checkout, such as an executable.

A pipeline that composes several providers inside one step seals each
companion call with ``StageChain`` and re-checks the sequence with
``check_chain``; it overrides ``_step``/``_validate_step`` and, when its
runtimes are not one subprocess checkout, ``_check_runtimes``. The bundle,
verification and receipt records stay the runner's.

A pipeline whose verification is not a same-runtime reproduction (a proof
checked by a registered verifier) overrides ``_verify``,
``_check_verification`` and ``_check_receipts``; the source, bundle and step
envelopes stay the runner's.

A class that overrides only these hooks is a ``generic_runner`` pipeline;
``pipelines.check`` enforces that. Records are the ones the hand-written
workflows produced, so retained workspaces reopen and replay unchanged.
Repository and executable paths are operator bindings, never read from a
retained record, and the host fields of a runtime never count as its pin.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import re
import uuid

from ..adapters.subprocess import PinnedSubprocessAdapter
from ..core.canonical import bundle_digest, byte_digest, canonical, digest, exact_keys, utc_now
from ..exchange import _identity

MAX_BYTES = 4 * 1024 * 1024
RESULT_SCHEMA = "ciw.declared-workload-result.v1"
VERIFY_SCHEMA = "ciw.declared-workload-verification.v1"
VERIFY_METHOD = "same_runtime_fresh_occurrence_reproduction"
AUTHORITY = {"state_admission": "not_performed", "sensor_fusion": "not_performed", "physical_truth": "not_established"}


class RecordProfile:
    """The result schema, verification schema and method, and authority a pipeline's records carry."""

    def __init__(self, result_schema: str, verify_schema: str, verify_method: str, authority: dict):
        self.result_schema, self.verify_schema, self.verify_method = result_schema, verify_schema, verify_method
        self.authority = dict(authority)


DECLARED = RecordProfile(RESULT_SCHEMA, VERIFY_SCHEMA, VERIFY_METHOD, AUTHORITY)
STEP_FIELDS = frozenset({"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                         "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
RESULT_FIELDS = frozenset({"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
VERIFICATION_FIELDS = frozenset({"schema", "subject_ref", "outcome", "independent", "method", "runtime_digest",
                                 "reproduction", "authority", "verification_id"})
RECEIPT_FIELDS = frozenset({"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match",
                            "verification", "admission", "replay_id"})
BUNDLE_FIELDS = frozenset({"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps",
                           "bundle_digest", "verification"})
RUNTIME_FIELDS = frozenset({"schema", "adapter_version", "repository_root", "revision", "source_tree", "module",
                            "source_root", "python_executable", "python_sha256", "python_version", "dependencies"})
HOST_FIELDS = frozenset({"repository_root", "python_executable"})
HOOKS = frozenset({"__init__", "parse_source", "invoke", "check_data", "check_runtime", "make_adapter", "bind_extra",
                   "step_request", "experiment_id", "configuration"})
_EXECUTION = re.compile(r"execution-[a-f0-9]{32}")
_SESSION = re.compile(r"session-[a-f0-9]{32}")
_SHA256 = re.compile(r"sha256:[a-f0-9]{64}")
_MALFORMED = (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError)


def text(value) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("Require a bounded nonempty identifier")


def same(actual, expected, message: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError(message)


NESTED_RUNTIMES = ("vendor",)


def host_projection(runtime):
    """A runtime identity without its host bindings.

    Host fields are dropped from the runtime and from the nested provider
    runtimes it names (``vendor``); every other value, including
    ``dependencies`` and ``engine``, is compared exactly.
    """
    value = {key: item for key, item in runtime.items() if key not in HOST_FIELDS}
    for key in NESTED_RUNTIMES:
        if isinstance(value.get(key), dict):
            value[key] = host_projection(value[key])
    return value


def seal_step(role: str, operation: str, source: dict, input_refs: list, data, numerical=None,
              profile: RecordProfile = DECLARED) -> dict:
    """One execution occurrence and its sealed result over the provider's native data.

    ``numerical`` replaces the default ``{operation_id, data}`` projection that
    replay compares, for a step whose data embeds sealed companion stages.
    """
    occurrence = "execution-" + uuid.uuid4().hex
    result = {"schema": profile.result_schema, "operation_id": operation, "execution_ref": occurrence,
              "input_refs": list(input_refs), "data": deepcopy(data), "authority": deepcopy(profile.authority)}
    result["result_id"] = digest(result)
    numerical = {"operation_id": operation, "data": deepcopy(data)} if numerical is None else deepcopy(numerical)
    return {"runtime_ref": role, "operation_id": operation, "execution_id": occurrence,
            "input_refs": list(input_refs), "request": deepcopy(source), "request_sha256": digest(source),
            "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
            "numerical_result": numerical, "numerical_result_id": digest(numerical)}


def check_step(step, *, role: str, operation: str, source: dict, input_refs: list, check_data, label: str,
               numerical=None, profile: RecordProfile = DECLARED, request=None) -> None:
    """Refuse a step unless every identity in it binds this source, operation and native data.

    ``check_data`` may be ``None`` when the caller checks the data itself.
    ``numerical`` is the expected projection, or a function of the checked data.
    ``request`` is what the step retains as its request when that is narrower
    than the source; ``check_data`` always receives the whole source.
    """
    request = source if request is None else request
    exact_keys(step, STEP_FIELDS)
    if (step["runtime_ref"] != role or step["operation_id"] != operation or step["input_refs"] != input_refs or
            not isinstance(step["execution_id"], str) or not _EXECUTION.fullmatch(step["execution_id"])):
        raise ValueError(f"{label} operation, evidence or execution occurrence mismatch")
    same(step["request"], request, f"{label} request differs from the retained source")
    result = step["result"]
    exact_keys(result, RESULT_FIELDS)
    if check_data is not None:
        check_data(source, result["data"])
    same(result["authority"], profile.authority, f"{label} result cannot confer state or physical authority")
    if (result["schema"] != profile.result_schema or result["operation_id"] != operation or
            result["execution_ref"] != step["execution_id"] or result["input_refs"] != input_refs or
            result["result_id"] != step["result_id"] or
            result["result_id"] != digest({key: value for key, value in result.items() if key != "result_id"})):
        raise ValueError(f"{label} result binding mismatch")
    if numerical is None:
        numerical = {"operation_id": operation, "data": result["data"]}
    elif callable(numerical):
        numerical = numerical(result["data"])
    same(step["numerical_result"], numerical, f"{label} numerical projection mismatch")
    for key, content in (("request_sha256", request), ("result_sha256", result),
                         ("numerical_result_id", step["numerical_result"])):
        if step[key] != digest(content):
            raise ValueError(f"{label} step content binding mismatch")


class StageChain:
    """Companion provider calls inside one step, sealed in order.

    Each stage is its own execution occurrence and result. Its ``input_refs``
    record cumulative order, not data lineage: the evidence and then every
    earlier stage's result identity.
    """

    def __init__(self, evidence_id: str):
        self.evidence_id, self.stages = evidence_id, []

    def seal(self, role: str, operation: str, request, data) -> dict:
        stage = seal_step(role, operation, request, chain_refs(self.evidence_id, self.stages), data)
        self.stages.append(stage)
        return stage


def chain_refs(evidence_id: str, stages: list) -> list:
    return [evidence_id] + [stage["result_id"] for stage in stages]


def check_chain(stages, operations: dict, evidence_id: str, *, seen: set, label: str) -> None:
    """Refuse a stage sequence unless it is exactly ``operations`` in order, chained and freshly occurring.

    ``seen`` holds the enclosing step's occurrence and receives every stage's.
    """
    if type(stages) is not list or len(stages) != len(operations):
        raise ValueError(f"Require all {len(operations)} {label} stages")
    for index, ((role, operation), stage) in enumerate(zip(operations.items(), stages)):
        check_step(stage, role=role, operation=operation, source=stage["request"],
                   input_refs=chain_refs(evidence_id, stages[:index]), check_data=None, label=f"{label} {role} stage")
        if stage["execution_id"] in seen:
            raise ValueError(f"{label} stages require distinct execution occurrences")
        seen.add(stage["execution_id"])


def chain_catalog(bundle: dict) -> list:
    """The primary step's sealed stages, for the workbench execution/result catalog."""
    return deepcopy(bundle["steps"][0]["result"]["data"]["stages"])


def chain_occurrences(bundle: dict) -> set:
    """Stage occurrences of the primary step and its reproduction."""
    return {stage["execution_id"] for outer in (bundle["steps"][0], bundle["verification"]["reproduction"])
            for stage in outer["result"]["data"]["stages"]}


def chain_claims(bundle: dict) -> dict:
    """Every stage identity with the one content it may name; a reused identity must name the same content."""
    claims = {}
    for outer in (bundle["steps"][0], bundle["verification"]["reproduction"]):
        for stage in outer["result"]["data"]["stages"]:
            for identity, role, body in ((stage["execution_id"], "execution", {"step": stage}),
                                         (stage["result_id"], "result", {"result": stage["result"]}),
                                         (stage["numerical_result_id"], "numerical_result", stage["numerical_result"]),
                                         (stage["operation_id"], "operation", stage["operation_id"])):
                if identity in claims:
                    same(claims[identity], (role, body), "Stage identity names different content")
                claims[identity] = (role, deepcopy(body))
    return claims


def verification(bundle: dict, reproduced: dict, profile: RecordProfile = DECLARED) -> dict:
    """Same-runtime reproduction of a bundle's primary step; never independent verification."""
    if canonical(bundle["steps"][0]["numerical_result"]) != canonical(reproduced["numerical_result"]):
        raise ValueError("Native workload replay mismatch")
    value = {"schema": profile.verify_schema, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
             "independent": False, "method": profile.verify_method, "runtime_digest": digest(bundle["runtimes"]),
             "reproduction": deepcopy(reproduced), "authority": deepcopy(profile.authority)}
    value["verification_id"] = byte_digest(profile.verify_schema.encode() + b"\0" + canonical(value))
    return value


def check_receipts(bundle: dict, kind: str, profile: RecordProfile = DECLARED) -> None:
    """A replayed bundle retains at most one receipt naming the earlier occurrence it reproduced.

    The receipt's runtime digest covers the earlier occurrence's host paths;
    ``Workbench._validate_links`` checks it against retained history.
    """
    receipts = bundle.get("replay_receipts", [])
    if not isinstance(receipts, list) or len(receipts) > 1:
        raise ValueError("A session occurrence retains at most one replay receipt")
    for receipt in receipts:
        exact_keys(receipt, RECEIPT_FIELDS)
        earlier = receipt["source_bundle_digest"]
        if (receipt["schema"] != "ciw." + kind + "-replay.v1" or not isinstance(earlier, str) or
                not _SHA256.fullmatch(earlier) or earlier == bundle["bundle_digest"] or
                receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                receipt["numerical_match"] is not True or receipt["admission"] != "not_performed"):
            raise ValueError("Invalid replay receipt binding or authority")
        proof = receipt["verification"]
        exact_keys(proof, VERIFICATION_FIELDS)
        if (proof["schema"] != profile.verify_schema or proof["subject_ref"] != earlier or proof["outcome"] != "passed" or
                proof["independent"] is not False or proof["method"] != profile.verify_method or
                not isinstance(proof["runtime_digest"], str) or not _SHA256.fullmatch(proof["runtime_digest"])):
            raise ValueError("Invalid replay verification scope")
        same(proof["authority"], profile.authority, "Replay cannot confer authority")
        same(proof["reproduction"], bundle["steps"][0], "Replay must bind this exact fresh step")
        _identity(proof, "verification_id")
        if receipt["replay_id"] != digest({key: value for key, value in receipt.items() if key != "replay_id"}):
            raise ValueError("Replay receipt content identity mismatch")


class PipelineRunner:
    """One pinned provider step over one retained source, sealed, reproduced and replayable."""

    MAX_BYTES = MAX_BYTES
    LABEL = "Native"
    PROFILE = DECLARED
    EXTRA_ROLES = frozenset()
    RUNTIME_EXTRA = frozenset()

    def __init__(self, kind: str, pin: dict):
        self.kind, self.pin = kind, dict(pin)
        self.role = self.pin["role"]
        self.ROLES = frozenset({self.role}) | self.EXTRA_ROLES
        self.SOURCE_SCHEMA = "ciw." + kind + "-source.v1"
        self.schema = "ciw." + kind + "-session.v1"
        self.operation = "ciw." + kind + ".v1"

    # Workbench catalog hooks ---------------------------------------------
    # Native occurrences inside a step beyond the step itself. The workbench
    # lists ``catalog_steps``, merges ``identity_claims`` into its collision
    # check and refuses two bundles of one kind sharing ``native_occurrences``.

    FRESH_OCCURRENCE_MESSAGE = "Bundles must retain fresh native execution occurrences"

    def catalog_steps(self, bundle):
        return []

    def identity_claims(self, bundle):
        return {}

    def native_occurrences(self, bundle):
        return None

    # Domain hooks -------------------------------------------------------

    def parse_source(self, raw: bytes) -> dict:
        raise NotImplementedError

    def invoke(self, source: dict, bound: tuple):
        raise NotImplementedError

    def check_data(self, source: dict, data) -> None:
        raise NotImplementedError

    def check_runtime(self, runtime: dict) -> None:
        return None

    def step_request(self, source: dict, evidence_id: str):
        """What a step retains as its request: the whole source unless the pipeline narrows it."""
        return source

    def experiment_id(self, source: dict) -> str:
        """The experiment a bundle names for this source."""
        return source["experiment_id"]

    def configuration(self, source: dict):
        """The configuration a bundle retains for this source."""
        return source["configuration"]

    def make_adapter(self, repository, retained: dict):
        return PinnedSubprocessAdapter(
            repository, self.pin["revision"], self.pin["module"], source_root=self.pin["source_root"],
            expected_python_sha256=retained.get("python_sha256"), expected_python_version=retained.get("python_version"),
            expected_dependencies=retained.get("dependencies"))

    def bind_extra(self, repositories: dict, adapter, runtime: dict):
        return None

    # Runner -------------------------------------------------------------

    def _source(self, raw):
        return self.parse_source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        return host_projection(runtime)

    def _input_refs(self, source, evidence_id):
        return [evidence_id]

    def _adapters(self, repositories, expected=None):
        if set(repositories) != set(self.ROLES):
            raise ValueError("Bind exactly the declared provider roles: " + ", ".join(sorted(self.ROLES)))
        retained = expected[self.role] if expected else {}
        adapter = self.make_adapter(repositories[self.role], retained)
        runtime = adapter.runtime_identity()
        extra = self.bind_extra(repositories, adapter, runtime)
        self._check_pin(runtime)
        if retained and self._runtime_projection(runtime) != self._runtime_projection(retained):
            raise ValueError(f"{self.LABEL} runtime differs from the retained execution")
        return adapter, runtime, extra

    def _check_pin(self, runtime):
        if (runtime["schema"] != "ciw.subprocess-runtime.v1" or
                any(runtime[key] != self.pin[key] for key in ("revision", "module", "source_root"))):
            raise ValueError(f"Unapproved {self.LABEL} runtime pin")
        if (not isinstance(runtime["source_tree"], str) or not re.fullmatch("[a-f0-9]{40}", runtime["source_tree"]) or
                runtime["source_tree"] != self.pin.get("source_tree", runtime["source_tree"])):
            raise ValueError(f"{self.LABEL} source tree differs from the approved provider pin")
        self.check_runtime(runtime)

    def run_provider(self, bound, script, arguments, payload):
        """Run a provider script, then re-verify its identity before its outcome is read.

        A provider whose checkout, interpreter or dependencies changed while it
        ran is refused as a pin mismatch whatever its exit code or output.
        """
        adapter, runtime, _ = bound
        code, raw = adapter._run(script, arguments, payload)
        self._unchanged(adapter, runtime, "during")
        return code, raw

    def _unchanged(self, adapter, runtime, when):
        current = self._runtime_projection(adapter.runtime_identity())
        retained = self._runtime_projection(runtime)
        if any(key not in retained or canonical(retained[key]) != canonical(value) for key, value in current.items()):
            raise ValueError(f"{self.LABEL} provider changed {when} execution")

    def _step(self, source, evidence_id, bound):
        adapter, runtime, _ = bound
        self._unchanged(adapter, runtime, "before")
        data = self.invoke(source, bound)
        self._unchanged(adapter, runtime, "during")
        self.check_data(source, data)
        return seal_step(self.role, self.operation, self.step_request(source, evidence_id), self._input_refs(source, evidence_id), data,
                         profile=self.PROFILE)

    def _validate_step(self, step, source, evidence_id):
        check_step(step, role=self.role, operation=self.operation, source=source, request=self.step_request(source, evidence_id),
                   profile=self.PROFILE,
                   input_refs=self._input_refs(source, evidence_id), check_data=self.check_data, label=self.LABEL)

    def _check_verification(self, bundle, proof, source, evidence):
        exact_keys(proof, VERIFICATION_FIELDS)
        self._validate_step(proof["reproduction"], source, evidence)
        old, new = bundle["steps"][0], proof["reproduction"]
        if old["execution_id"] == new["execution_id"] or old["result_id"] == new["result_id"] or proof != verification(bundle, new, self.PROFILE):
            raise ValueError("Verification must bind fresh native reproduction and limited authority")
        _identity(proof, "verification_id")

    def _check_envelope(self, bundle):
        """The bundle, source evidence and configuration bindings; returns ``(raw, source, evidence_ref)``."""
        exact_keys(bundle, BUNDLE_FIELDS, {"replay_receipts"})
        if (len(canonical(bundle)) > self.MAX_BYTES or bundle["schema"] != self.schema or
                bundle["bundle_digest"] != bundle_digest(bundle)):
            raise ValueError(f"{self.LABEL} session bundle binding mismatch")
        if not isinstance(bundle["session_id"], str) or not _SESSION.fullmatch(bundle["session_id"]):
            raise ValueError(f"Invalid {self.LABEL} session occurrence")
        text(bundle["created_at"])
        evidence, = bundle["source"]["evidence"]
        raw = base64.b64decode(evidence["bytes_b64"], validate=True)
        source = self._source(raw)
        if (evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw),
                         "bytes_b64": base64.b64encode(raw).decode()} or
                bundle["source"] != {"experiment_id": self.experiment_id(source), "experiment_digest": digest(source),
                                     "evidence": [evidence]} or
                canonical(bundle["configuration"]) != canonical(self.configuration(source))):
            raise ValueError(f"{self.LABEL} source/configuration binding mismatch")
        return raw, source, evidence["artifact_ref"]

    def _check_receipts(self, bundle):
        check_receipts(bundle, self.kind, self.PROFILE)

    def _verify(self, bundle, source, evidence, bound):
        return verification(bundle, self._step(source, evidence, bound), self.PROFILE)

    def _validate(self, bundle):
        try:
            raw, source, evidence = self._check_envelope(bundle)
            self._check_runtimes(bundle["runtimes"])
            step, = bundle["steps"]
            self._validate_step(step, source, evidence)
            self._check_verification(bundle, bundle["verification"], source, evidence)
            self._check_receipts(bundle)
            return raw
        except _MALFORMED as exc:
            raise ValueError(f"Malformed {self.LABEL} session") from exc

    def _check_runtimes(self, runtimes):
        if set(runtimes) != {self.role}:
            raise ValueError(f"Unexpected {self.LABEL} runtime")
        runtime = runtimes[self.role]
        exact_keys(runtime, RUNTIME_FIELDS | self.RUNTIME_EXTRA)
        if not re.fullmatch("[a-f0-9]{64}", runtime["python_sha256"]) or not isinstance(runtime["dependencies"], dict):
            raise ValueError(f"Unapproved {self.LABEL} runtime identity")
        for key in ("adapter_version", "repository_root", "python_executable", "python_version"):
            text(runtime[key])
        self._check_pin(runtime)

    def _execute(self, raw, bound):
        source = self._source(raw)
        evidence = byte_digest(raw)
        step = self._step(source, evidence, bound)
        bundle = {"schema": self.schema, "session_id": "session-" + uuid.uuid4().hex, "created_at": utc_now(),
                  "source": {"experiment_id": self.experiment_id(source), "experiment_digest": digest(source),
                             "evidence": [{"artifact_ref": evidence, "sha256": evidence,
                                           "bytes_b64": base64.b64encode(raw).decode()}]},
                  "configuration": deepcopy(self.configuration(source)), "runtimes": {self.role: bound[1]}, "steps": [step]}
        bundle["bundle_digest"] = bundle_digest(bundle)
        bundle["verification"] = self._verify(bundle, source, evidence, bound)
        self._validate(bundle)
        return bundle

    def create_session(self, raw, repositories):
        self._source(raw)
        return self._execute(raw, self._adapters(repositories))

    def replay_session(self, bundle, repositories):
        raw = self._validate(bundle)
        fresh = self._execute(raw, self._adapters(repositories, bundle["runtimes"]))
        receipt = {"schema": "ciw." + self.kind + "-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
                   "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                   "verification": verification(bundle, fresh["steps"][0], self.PROFILE), "admission": "not_performed"}
        receipt["replay_id"] = digest(receipt)
        fresh["replay_receipts"] = [receipt]
        return {"session": fresh, "replay_receipt": receipt}
