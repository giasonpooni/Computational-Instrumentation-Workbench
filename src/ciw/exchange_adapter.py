"""Typed instrument-exchange producer to native CIW adapter.

The adapter consumes a retained, typed producer envelope and projects the
validated records into the shared workbench.  It never treats a producer ID as
an operation, execution, or native result identity.  Replays run the pinned
SET contract checker again and receive fresh native occurrence identities;
the numerical projection identity remains stable when the checker result is
byte-for-byte equivalent.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile
import uuid

from . import exchange
from .telemetry import canonical, digest

SOURCE_SCHEMA = "ciw.instrument-exchange-source.v1"
SESSION_SCHEMA = "ciw.instrument-exchange-session.v1"
RESULT_SCHEMA = "ciw.instrument-exchange-native-result.v1"
NUMERICAL_SCHEMA = "ciw.instrument-exchange-numerical.v1"
REPLAY_SCHEMA = "ciw.instrument-exchange-replay.v1"
OPERATION = "ciw.instrument-exchange.v1"
ROLES = {"set"}
MAX_BYTES = 4 * 1024 * 1024
MAX_ARTIFACTS = 32
PIN = json.loads((Path(__file__).with_name("exchange-runtime.json")).read_text(encoding="utf-8"))


def _keys(value, required, optional=()):
    keys = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or not set(required) <= keys <= set(required) | set(optional):
        raise ValueError("Malformed instrument-exchange object")


def _text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be bounded nonempty text")


def _id(value, name):
    _text(value, name, 256)


def _json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON number: {value}")))


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _source(raw):
    """Validate the structural envelope without executing the checker."""
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_BYTES:
        raise ValueError("Exchange source exceeds its byte budget")
    value = _json(raw)
    _keys(value, {"schema", "artifacts", "producer"})
    if value["schema"] != SOURCE_SCHEMA:
        raise ValueError("Unsupported typed exchange source schema")
    if not isinstance(value["producer"], dict):
        raise ValueError("Producer declaration must be an object")
    _keys(value["producer"], {"name", "revision", "operation_ids"})
    if set(value["producer"]) - {"name", "revision", "operation_ids"}:
        raise ValueError("Unknown producer declaration field")
    _text(value["producer"]["name"], "producer.name", 128)
    _text(value["producer"]["revision"], "producer.revision", 128)
    if (not isinstance(value["producer"]["operation_ids"], list) or
            not 1 <= len(value["producer"]["operation_ids"]) <= 32):
        raise ValueError("producer.operation_ids must be a bounded list")
    for operation in value["producer"]["operation_ids"]:
        _text(operation, "producer operation", 128)
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= MAX_ARTIFACTS:
        raise ValueError("Exchange source requires 1..32 typed artifacts")
    identities = set()
    supported = set(exchange._SCHEMAS)
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("schema") not in supported:
            raise ValueError("Exchange source contains an unsupported artifact schema")
        schema = artifact["schema"]
        field = exchange._SCHEMAS[schema][0]
        identity = artifact.get(field)
        _id(identity, field)
        if identity in identities:
            raise ValueError("Exchange producer supplied duplicate artifact identity")
        identities.add(identity)
    return deepcopy(value)


def _source_bytes(value):
    return canonical(value)


def _git(repo: Path, *args):
    try:
        completed = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("SET provider must be a readable git checkout") from exc
    return completed.stdout.decode("ascii", "strict").strip()


def _runtime(repo: Path):
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise ValueError("SET provider checkout does not exist")
    revision = _git(repo, "rev-parse", "HEAD")
    if revision != PIN["revision"]:
        raise ValueError("SET provider revision differs from the exchange pin")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    validator_path = repo / PIN["path"]
    if not validator_path.is_file():
        raise ValueError("Pinned SET validator source is missing")
    source = validator_path.read_bytes()
    source_digest = sha256(source.replace(b"\r\n", b"\n")).hexdigest()
    if source_digest != PIN["sha256"]:
        raise ValueError("Pinned SET validator source bytes differ")
    return {
        "repository": PIN["repository"], "revision": revision, "source_tree": tree,
        "module": PIN["path"], "source_sha256": source_digest,
        "execution_scope": "standalone_checked_source_only",
    }


def _adapters(repositories, expected=None):
    if not isinstance(repositories, dict) or set(repositories) != ROLES:
        raise ValueError("Bind exactly the pinned SET validator checkout")
    identity = _runtime(Path(repositories["set"]))
    if expected is not None:
        if not isinstance(expected, dict) or any(identity.get(key) != expected.get(key)
                                                for key in identity):
            raise ValueError("SET provider identity differs from the retained binding")
    return {"set": identity}


def _inspect(source, repo):
    artifacts = source["artifacts"]
    with tempfile.TemporaryDirectory(prefix="ciw-exchange-") as directory:
        paths = []
        for index, artifact in enumerate(artifacts):
            path = Path(directory) / f"artifact-{index}.json"
            path.write_bytes(canonical(artifact))
            paths.append(path)
        report = exchange.inspect_exchange(paths, validator_repo=Path(repo))
    if report.get("status") != "conformant":
        raise ValueError("Pinned SET checker did not establish exchange conformance")
    return report


def _native_verification(subject_ref, report):
    value = {
        "schema": exchange.VERIFICATION_SCHEMA,
        "subject_ref": subject_ref,
        "verifier_ref": "ciw-exchange-adapter.v1",
        "outcome": "passed",
        "checks": [{"name": "pinned_set_conformance", "outcome": "passed",
                    "basis": "typed producer records accepted by the pinned SET contract"}],
        "limitations": ["content conformance is not physical validation",
                        "source admission, execution behavior and verifier independence remain unestablished"],
        "authority": {"may_authorize": False, "state_admission": "not_performed"},
    }
    value["verification_id"] = _identity(value, "verification_id")
    return value


def _identity(value, field):
    payload = {key: item for key, item in value.items() if key != field}
    return "sha256:" + sha256(value["schema"].encode("utf-8") + b"\x00" + canonical(payload)).hexdigest()


def _numerical(report):
    return {
        "schema": NUMERICAL_SCHEMA,
        "artifacts": [deepcopy(item["artifact"]) for item in report["artifacts"]],
        "links": deepcopy(report["links"]),
        "validator": deepcopy(report["validator"]),
        "covariance_validation": [deepcopy(item["covariance_validation"]) for item in report["artifacts"]],
        "authority": deepcopy(report["authority"]),
    }


def _bundle_digest(value):
    body = {key: item for key, item in value.items() if key not in {"verification", "replay_receipts", "bundle_digest"}}
    return digest(body)


def _make_native(raw, source, report, runtime, *, replay_of=None):
    source_id = "sha256:" + sha256(raw).hexdigest()
    session_id = "session:" + uuid.uuid4().hex
    execution_id = "execution:" + uuid.uuid4().hex
    numerical = _numerical(report)
    numerical_id = digest(numerical)
    result_id = "result:" + uuid.uuid4().hex
    result = {
        "schema": RESULT_SCHEMA, "operation_id": OPERATION,
        "execution_ref": execution_id, "execution_id": execution_id,
        "result_id": result_id, "numerical_result_id": numerical_id,
        "producer_artifact_ids": [item["artifact"][exchange._SCHEMAS[item["artifact"]["schema"]][0]]
                                   for item in report["artifacts"]],
        "artifacts": deepcopy(numerical["artifacts"]), "links": deepcopy(numerical["links"]),
        "covariance_validation": deepcopy(numerical["covariance_validation"]),
        "validator": deepcopy(report["validator"]), "authority": deepcopy(report["authority"]),
        "claim_scope": "typed_exchange_content_conformance_only",
        "physical_validation": "not_established", "state_admission": "not_performed",
    }
    step = {
        "runtime_ref": "exchange", "operation_id": OPERATION,
        "execution_id": execution_id, "input_refs": [source_id],
        "request": deepcopy(source), "request_sha256": digest(source),
        "result": result, "result_sha256": digest(result), "result_id": result_id,
        "numerical_result": numerical, "numerical_result_id": numerical_id,
    }
    native = {
        "schema": SESSION_SCHEMA, "session_id": session_id,
        "source": {"schema": SOURCE_SCHEMA, "evidence": [{"artifact_ref": source_id,
                                                               "bytes_b64": base64.b64encode(raw).decode("ascii")}],
                    "source_kind": "typed_exchange_producer"},
        "configuration": {"operation": OPERATION, "validator_revision": runtime["revision"],
                          "validator_source_sha256": runtime["source_sha256"],
                          "authority": "read_only_content_conformance"},
        "runtimes": {"set": runtime}, "steps": [step],
        "verification": None, "replay_receipts": [],
        "authority": {"may_authorize": False, "state_admission": "not_performed",
                       "physical_validation": "not_established"},
    }
    if replay_of is not None:
        native["replay_of"] = replay_of
    native["bundle_digest"] = _bundle_digest(native)
    native["verification"] = _native_verification(native["bundle_digest"], report)
    _validate(native)
    return native


def _validate(native):
    """Validate a retained native bundle without executing the provider."""
    _keys(native, {"schema", "session_id", "source", "configuration", "runtimes", "steps", "verification",
                   "replay_receipts", "authority", "bundle_digest"}, {"replay_of"})
    if native["schema"] != SESSION_SCHEMA:
        raise ValueError("Unsupported native exchange session schema")
    _text(native["session_id"], "session_id")
    _keys(native["source"], {"schema", "evidence", "source_kind"})
    if native["source"]["schema"] != SOURCE_SCHEMA or native["source"]["source_kind"] != "typed_exchange_producer":
        raise ValueError("Native exchange source declaration differs")
    _keys(native["configuration"], {"operation", "validator_revision", "validator_source_sha256", "authority"})
    if (native["configuration"]["operation"] != OPERATION or
            native["configuration"]["validator_revision"] != PIN["revision"] or
            native["configuration"]["validator_source_sha256"] != PIN["sha256"] or
            native["configuration"]["authority"] != "read_only_content_conformance"):
        raise ValueError("Native exchange configuration differs from the pin")
    evidence = native["source"]["evidence"]
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError("Native exchange session must retain exactly one source envelope")
    _keys(evidence[0], {"artifact_ref", "bytes_b64"})
    try:
        raw = base64.b64decode(evidence[0]["bytes_b64"], validate=True)
    except Exception as exc:
        raise ValueError("Native exchange source bytes are not canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != evidence[0]["bytes_b64"]:
        raise ValueError("Native exchange source bytes are not canonical base64")
    source_id = "sha256:" + sha256(raw).hexdigest()
    if evidence[0]["artifact_ref"] != source_id:
        raise ValueError("Native exchange source identity does not bind exact bytes")
    source = _source(raw)
    runtimes = native["runtimes"]
    if not isinstance(runtimes, dict) or set(runtimes) != ROLES:
        raise ValueError("Native exchange session must retain the SET runtime identity")
    runtime = runtimes["set"]
    _keys(runtime, {"repository", "revision", "source_tree", "module", "source_sha256", "execution_scope"})
    if runtime["repository"] != PIN["repository"] or runtime["revision"] != PIN["revision"] or runtime["module"] != PIN["path"] or runtime["source_sha256"] != PIN["sha256"]:
        raise ValueError("Retained SET runtime identity differs from the pin")
    steps = native["steps"]
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("Native exchange session must contain one adapter occurrence")
    step = steps[0]
    _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                 "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
    if step["runtime_ref"] != "exchange" or step["operation_id"] != OPERATION or step["input_refs"] != [source_id]:
        raise ValueError("Native exchange occurrence binding differs")
    _text(step["execution_id"], "execution_id")
    if step["request_sha256"] != digest(source) or canonical(step["request"]) != canonical(source):
        raise ValueError("Native exchange request is not the retained source")
    result = step["result"]
    _keys(result, {"schema", "operation_id", "execution_ref", "execution_id", "result_id", "numerical_result_id",
                   "producer_artifact_ids", "artifacts", "links", "covariance_validation", "validator", "authority",
                   "claim_scope", "physical_validation", "state_admission"})
    if result["schema"] != RESULT_SCHEMA or result["operation_id"] != OPERATION or result["execution_id"] != step["execution_id"] or result["execution_ref"] != step["execution_id"] or result["result_id"] != step["result_id"]:
        raise ValueError("Native result does not bind its execution occurrence")
    if step["result_sha256"] != digest(result) or step["numerical_result_id"] != result["numerical_result_id"]:
        raise ValueError("Native result commitment differs")
    numerical = step["numerical_result"]
    if digest(numerical) != step["numerical_result_id"] or numerical["schema"] != NUMERICAL_SCHEMA:
        raise ValueError("Native numerical result commitment differs")
    if canonical(result["artifacts"]) != canonical(numerical["artifacts"]) or canonical(result["links"]) != canonical(numerical["links"]):
        raise ValueError("Native result projection differs from its numerical content")
    source_artifacts = source["artifacts"]
    if canonical(result["artifacts"]) != canonical(source_artifacts):
        raise ValueError("Native result does not retain typed producer artifacts")
    _keys(native["verification"], {"schema", "subject_ref", "verifier_ref", "outcome", "checks", "limitations", "authority", "verification_id"})
    if native["verification"]["schema"] != exchange.VERIFICATION_SCHEMA or native["verification"]["subject_ref"] != native["bundle_digest"]:
        raise ValueError("Native verification does not bind the bundle")
    exchange._identity(native["verification"], "verification_id")
    if native["bundle_digest"] != _bundle_digest(native):
        raise ValueError("Native bundle digest differs")
    receipts = native["replay_receipts"]
    if not isinstance(receipts, list) or len(receipts) > 1:
        raise ValueError("Native exchange retains at most one replay receipt")
    for receipt in receipts:
        _keys(receipt, {"schema", "source_bundle_digest", "replayed_bundle_digest", "numerical_match", "verification", "admission", "replay_id"})
        if receipt["schema"] != REPLAY_SCHEMA or receipt["source_bundle_digest"] == receipt["replayed_bundle_digest"] or receipt["replayed_bundle_digest"] != native["bundle_digest"] or receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or receipt["replay_id"] != digest({k: v for k, v in receipt.items() if k != "replay_id"}):
            raise ValueError("Native exchange replay receipt differs")
        if receipt["verification"]["schema"] != exchange.VERIFICATION_SCHEMA or receipt["verification"]["subject_ref"] != receipt["source_bundle_digest"]:
            raise ValueError("Replay verification does not bind the original bundle")
        exchange._identity(receipt["verification"], "verification_id")
    return raw


def create_session(raw, repositories):
    source = _source(raw)
    runtime = _runtime(Path(repositories["set"]))
    report = _inspect(source, repositories["set"])
    return _make_native(raw, source, report, runtime)


def replay_session(native, repositories):
    raw = _validate(native)
    source = _source(raw)
    runtime = _runtime(Path(repositories["set"]))
    report = _inspect(source, repositories["set"])
    fresh = _make_native(raw, source, report, runtime, replay_of=native["bundle_digest"])
    old_numerical = native["steps"][0]["numerical_result_id"]
    numerical_match = fresh["steps"][0]["numerical_result_id"] == old_numerical
    if not numerical_match:
        raise ValueError("Pinned exchange replay changed the numerical projection")
    verification = _native_verification(native["bundle_digest"], report)
    receipt = {"schema": REPLAY_SCHEMA, "source_bundle_digest": native["bundle_digest"],
               "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
               "verification": verification, "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    _validate(fresh)
    return {"session": fresh, "replay_receipt": receipt}
