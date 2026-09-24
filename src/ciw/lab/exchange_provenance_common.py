"""CIW session fixtures and workspace forgery harness for tasks T077-T090.

Scope: drives the real CIW integrity layer (``ciw.session``, the operation
runner, ``ciw.workbench``, the energy-accuracy workflow, and the pure exchange
and candidate-evidence validators) through its request protocol and saved
workspaces, offline. No GPU, network or hardware is used; the energy-accuracy
input is the bundled synthetic fixture log. Only T077's telemetry fixture runs
provider code: the telemetry workflow's pinned PPDA, STFE, GSIE and SET
checkouts of an operator-bound ``telemetry-stack``, each validated against
``ciw/telemetry-runtimes.json`` first, on the bundled synthetic telemetry
source.

A forgery here edits a saved workspace (records, or a retained source log whose
own log_digest is resealed) and recomputes only unkeyed SHA-256 digests with
CIW's public canonicalization, which any holder of the file can do. A mutation
accepted on reopen is a surviving mutant: the retained records are
content-consistent, not authenticated. The ESM validator runs on a synthetic
telemetry-shaped record that no telemetry workflow validated. Nothing here
changes CIW code, and nothing here says whether a forged record would mislead
a particular reader.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Callable

FIXTURE_LOGS = ("baseline", "reset", "missing", "under-target")
KIND = "energy-accuracy"
OPERATION = "ciw.energy-accuracy.v1"
STATS = {"operation_id": "statistics.v1", "parameters": {"channel": "v", "interval_s": [1.0, 2.0]}}

# Fixed forged values keep mutation outcomes and witnesses deterministic.
FORGED_DIGEST = "sha256:" + hashlib.sha256(b"ciw-lab forged bundle").hexdigest()
FORGED_CODE = hashlib.sha256(b"ciw-lab forged analysis code").hexdigest()
FORGED_SESSION = "session-" + hashlib.sha256(b"ciw-lab forged session").hexdigest()[:32]
FORGED_TIME = "2001-01-01T00:00:00+00:00"
FORGED_RUNTIME = {"provider": "lab.forged-provider", "version": "99"}
FRESH_PATTERNS = {"execution": r"execution-[0-9a-f]{32}", "result": r"result-[0-9a-f]{32}",
                  "session": r"session-[0-9a-f]{32}", "sha256": r"sha256:[0-9a-f]{64}"}
# Fixed forged constants are deterministic and stay readable in witnesses.
CONSTANTS = {FORGED_DIGEST: "sha256:<forged-constant>", FORGED_SESSION: "session:<forged-constant>"}

# Cross-cutting deferred research questions, recorded verbatim among the unresolved assumptions of every report
# that depends on them, so the planner lists each once with the tasks that raise it.
KEY_CUSTODY_QUESTION = (
    "Deferred research question (key custody and signatures): which key signs CIW workspace records, replay "
    "receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a "
    "verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained "
    "workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over "
    "the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run "
    "against the signed records to check that every surviving forgery is refused.")
ESM_CANDIDATE_QUESTION = (
    "Deferred research question (ESM candidate binding): bind the ESM runtime pinned in src/ciw/esm-runtime.json "
    "(giasonpooni/Evidence-and-State-Management at 7c0642ebbcd95c301470e4be556342ab73c819cb, its built "
    "workbench-candidate artifact and replay helper at their pinned SHA-256, a node binary, and a CIW checkout at "
    "the replay revision e0af0472d9731eff603c13568325f15d14d2de5b) to the telemetry session that T077 runs on the "
    "telemetry-stack binding, and observe the ESM candidate_id and candidate execution identities, now read from "
    "code, and the ESM validator rows on a telemetry-validated bundle instead of the synthetic telemetry-shaped "
    "record. scripts/check_lab.py provisions none of these (scripts/check_workbench_candidates.py exercises them "
    "outside the queue), and an ESM action runs ESM's replay helper against that CIW checkout and the provider "
    "checkouts, for up to 330 s in CIW's candidate adapter.")

# The operator binding T077 reads (``--provider telemetry-stack=<dir>``): one directory holding the checkouts
# pinned in ciw/telemetry-runtimes.json under their repository names, as scripts/check_telemetry.py and
# scripts/check_lab.py lay them out.
TELEMETRY_ROLE = "telemetry-stack"
TELEMETRY_REPOSITORIES = {"ppda": "Provenance-Preserving-Data-Acquisition",
                          "stfe": "Streaming-Telemetry-Feature-Extraction",
                          "gsie": "Geometric-State-Inference-Engine",
                          "set": "State-Estimation-Evaluation-Testbed",
                          "cbsr": "Constraint-Based-State-Reconciliation"}
TELEMETRY_OWNER = "giasonpooni"


def energy_dir() -> Path | None:
    """The bundled synthetic energy logs, resolved like every lab repository input."""
    from .runner import repository_path
    return repository_path("examples", "energy-accuracy")


def fixture_available() -> bool:
    directory = energy_dir()
    return directory is not None and all((directory / f"{name}.json").is_file() for name in FIXTURE_LOGS)


def fixture_bytes(name: str) -> bytes:
    directory = energy_dir()
    if directory is None:
        raise FileNotFoundError("No repository examples are reachable for the energy-accuracy fixture logs")
    return (directory / f"{name}.json").read_bytes()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def request(session, kind: str, payload: dict) -> dict:
    """One protocol request; a refusal here is a fixture failure, not an outcome."""
    response = session.handle({"protocol_version": 1, "request_id": "lab-" + kind, "type": kind,
                               "payload": payload})
    if response["type"] != "response":
        raise RuntimeError(f"Fixture request {kind} was refused: {response['payload']}")
    return response["payload"]


def attempt(session, kind: str, payload: dict) -> dict:
    """One protocol request whose refusal is an observed outcome."""
    response = session.handle({"protocol_version": 1, "request_id": "lab-" + kind, "type": kind,
                               "payload": payload})
    if response["type"] == "response":
        return {"outcome": "accepted", "payload": response["payload"]}
    return {"outcome": "refused", "code": response["payload"].get("code"),
            "message": response["payload"].get("message")}


def energy_source(session, raw: bytes, label: str) -> dict:
    return request(session, "source.add", {"kind": KIND, "label": label, "bytes_b64": b64(raw)})


def energy_execute(session, source_id: str) -> dict:
    return request(session, "operation.execute", {"operation_id": OPERATION, "parameters": {"source_id": source_id}})


def save(session) -> dict:
    path = Path(request(session, "workspace.save", {})["workspace_file"])
    raw = path.read_bytes()
    return {"path": path, "bytes": raw, "workspace": json.loads(raw.decode("utf-8"))}


def build_session_fixture(root: Path) -> dict:
    """Oscillator operations, energy bundles, replays, save, reopen and a separate session (same process and code)."""
    from ..instruments import make_demo_run
    from ..session import Session

    run = make_demo_run()
    session = Session(run, root / "session-a")
    # Session writes in text mode, so newlines follow the platform; compare with LF normalized.
    recording = (root / "session-a" / session.recording_file).read_bytes().replace(b"\r\n", b"\n")
    first = request(session, "operation.execute", deepcopy(STATS))
    second = request(session, "operation.execute", deepcopy(STATS))
    legacy = request(session, "analysis.stats", deepcopy(STATS["parameters"]))
    baseline, other = fixture_bytes("baseline"), fixture_bytes("reset")
    source = energy_source(session, baseline, "baseline")
    other_source = energy_source(session, other, "reset")
    b0 = energy_execute(session, source["source_id"])
    b0b = energy_execute(session, source["source_id"])
    bother = energy_execute(session, other_source["source_id"])
    replay = request(session, "bundle.replay", {"bundle_id": b0["bundle_id"]})
    saved = save(session)
    # Reopen offline, replay again from the reopened catalog, and replay a replay.
    reopened = Session.from_workspace(saved["path"], root / "session-b")
    replay_reopened = request(reopened, "bundle.replay", {"bundle_id": b0["bundle_id"]})
    replay_of_replay = request(reopened, "bundle.replay", {"bundle_id": replay["bundle"]["bundle_id"]})
    saved_reopened = save(reopened)
    # A separate session in the same process retains the same bytes under the
    # same label and under a second label; only the same-label source is executed.
    separate = Session(run, root / "session-c")
    source_c = energy_source(separate, baseline, "baseline")
    relabelled = energy_source(separate, baseline, "baseline (relabelled copy)")
    b4 = energy_execute(separate, source_c["source_id"])
    bundles = {"B0": b0["bundle_id"], "B0b": b0b["bundle_id"], "Bother": bother["bundle_id"],
               "B1": replay["bundle"]["bundle_id"], "B2": replay_reopened["bundle"]["bundle_id"],
               "B3": replay_of_replay["bundle"]["bundle_id"]}
    natives = {role: reopened.workbench.get_bundle(identity) for role, identity in bundles.items()}
    natives["B4"] = separate.workbench.get_bundle(b4["bundle_id"])
    return {
        "run": deepcopy(run), "session_ids": [session.session_id, reopened.session_id, separate.session_id],
        "oscillator": {"E1": first["execution"], "R1": first["result"], "E2": second["execution"],
                       "R2": second["result"], "RL": legacy},
        "runtime_declared": session.operations.get("statistics.v1").runtime_identity(),
        "sources": {"S": source, "Sother": other_source, "SC": source_c, "Srelabelled": relabelled},
        "raw": {"baseline": baseline, "reset": other},
        "natives": natives,
        "receipts": {"B1": replay["replay_receipt"], "B2": replay_reopened["replay_receipt"],
                     "B3": replay_of_replay["replay_receipt"]},
        "workspace": saved["workspace"], "workspace_bytes": saved["bytes"],
        "workspace_reopened": saved_reopened["workspace"],
        "reopened_run_evidence_id": reopened.run["evidence_id"],
        "reopened_recording_file": reopened.recording_file,
        "recording": {"file": session.recording_file, "sha256": hashlib.sha256(recording).hexdigest(),
                      "is_reserialization": recording == (json.dumps(run, indent=2) + "\n").encode("utf-8")},
    }


def role_labels(fixture: dict) -> dict:
    """Map fresh (run-specific) identities to stable role names for retained witnesses."""
    labels = {}
    for role, record in fixture["oscillator"].items():
        if role.startswith("E"):
            labels[record["execution_id"]] = f"execution:{role}"
        else:
            labels[record["result_id"]] = f"result:{role}"
            labels.setdefault(record["execution_id"], f"execution:{role}")
    for role, native in fixture["natives"].items():
        labels[native["bundle_digest"]] = f"bundle:{role}"
        labels[native["session_id"]] = f"session:{role}"
        labels[native["steps"][0]["execution_id"]] = f"execution:{role}.step"
        labels[native["steps"][0]["result_id"]] = f"result:{role}.step"
        reproduction = native["verification"]["reproduction"]
        labels[reproduction["execution_id"]] = f"execution:{role}.reproduction"
        labels[reproduction["result_id"]] = f"result:{role}.reproduction"
        labels[native["verification"]["verification_id"]] = f"verification:{role}"
        for receipt in native.get("replay_receipts", []):
            labels[receipt["replay_id"]] = f"replay:{role}"
            labels[receipt["verification"]["verification_id"]] = f"verification:{role}.receipt"
    for index, identity in enumerate(fixture["session_ids"]):
        labels[identity] = f"protocol-session:{index}"
    return labels


def relabel(value, labels: dict):
    """Replace run-specific identities and timestamps so retained witnesses are deterministic."""
    if isinstance(value, dict):
        return {key: relabel(item, labels) for key, item in value.items()}
    if isinstance(value, list):
        return [relabel(item, labels) for item in value]
    if isinstance(value, str):
        if value in labels:
            return labels[value]
        if value in CONSTANTS:
            return CONSTANTS[value]
        for kind, pattern in FRESH_PATTERNS.items():
            if re.fullmatch(pattern, value):
                return f"{kind}:<unlabelled>"
    return value


# ------------------------------------------------------------------ forging
def verification_id(value: dict) -> str:
    from ..declared_workload import VERIFY_SCHEMA
    from ..telemetry import byte_digest, canonical
    body = {key: item for key, item in value.items() if key != "verification_id"}
    return byte_digest(VERIFY_SCHEMA.encode() + b"\0" + canonical(body))


def build_verification(bundle: dict, reproduction: dict, **overrides) -> dict:
    """A forger's verification record: CIW's layout, with no numerical-agreement guard."""
    from ..declared_workload import VERIFY_SCHEMA
    from ..energy_workflow import AUTHORITY, METHOD
    from ..telemetry import digest
    value = {"schema": VERIFY_SCHEMA, "subject_ref": bundle["bundle_digest"], "outcome": "passed",
             "independent": False, "method": METHOD, "runtime_digest": digest(bundle["runtimes"]),
             "reproduction": deepcopy(reproduction), "authority": deepcopy(AUTHORITY)}
    value.update(deepcopy(overrides))
    value["verification_id"] = verification_id(value)
    return value


def restep(step: dict) -> None:
    """Recompute every unkeyed digest inside one energy analysis step."""
    from ..telemetry import digest
    result = step["result"]
    result["execution_ref"] = step["execution_id"]
    result["result_id"] = digest({key: item for key, item in result.items() if key != "result_id"})
    step["result_id"], step["result_sha256"] = result["result_id"], digest(result)
    step["numerical_result"] = {"operation_id": step["operation_id"], "data": deepcopy(result["data"])}
    step["numerical_result_id"] = digest(step["numerical_result"])
    step["request_sha256"] = digest(step["request"])


def reseal_receipt(receipt: dict) -> None:
    from ..telemetry import digest
    receipt["verification"]["verification_id"] = verification_id(receipt["verification"])
    receipt["replay_id"] = digest({key: item for key, item in receipt.items() if key != "replay_id"})


def forge_receipt(source: dict, replayed: dict) -> dict:
    """A receipt claiming that ``replayed`` replays ``source``, built from the two bundles alone.

    Every field is copied from or recomputed over retained records (CIW's
    layout, the replayed bundle's own step as the reproduction); no replay
    event is needed to write it.
    """
    receipt = {"schema": f"ciw.{KIND}-replay.v1", "source_bundle_digest": source["bundle_digest"],
               "replayed_bundle_digest": replayed["bundle_digest"], "numerical_match": True,
               "verification": build_verification(source, replayed["steps"][0]), "admission": "not_performed"}
    reseal_receipt(receipt)
    return receipt


def reforge(workspace: dict, *, keep_verification=False) -> None:
    """Recompute every derived energy-bundle digest in catalog order, as a forger would.

    Bundle digests, verifications, receipts, catalog bundle identities and the
    catalog's replay-receipt seal are regenerated; renamed bundles are followed
    into later replay receipts.
    """
    from ..telemetry import _bundle_digest
    renamed, natives = {}, {}
    for record in workspace["workbench"]["bundles"]:
        if record["kind"] != KIND:
            continue
        native = record["native"]
        for step in native["steps"] + [native["verification"]["reproduction"]]:
            restep(step)
        native["bundle_digest"] = _bundle_digest(native)
        if keep_verification:
            native["verification"]["subject_ref"] = native["bundle_digest"]
            native["verification"]["verification_id"] = verification_id(native["verification"])
        else:
            native["verification"] = build_verification(native, native["verification"]["reproduction"])
        for receipt in native.get("replay_receipts", []):
            source = renamed.get(receipt["source_bundle_digest"], receipt["source_bundle_digest"])
            receipt["source_bundle_digest"] = source
            receipt["replayed_bundle_digest"] = native["bundle_digest"]
            original = natives.get(source)
            receipt["verification"] = (build_verification(original, native["steps"][0]) if original is not None
                                       else dict(receipt["verification"], subject_ref=source))
            reseal_receipt(receipt)
        renamed[record["bundle_id"]] = native["bundle_digest"]
        record["bundle_id"] = native["bundle_digest"]
        natives[native["bundle_digest"]] = native
    reseal_catalog(workspace)


def reforge_source(workspace: dict, source_id: str, edit: Callable) -> dict:
    """Rewrite one retained source log and every record derived from it, keeping occurrence ids and times.

    The log's unkeyed log_digest is resealed, the workbench source record is
    rebuilt, each bundle of that source is re-analysed with CIW's own
    ``analyze`` and all downstream digests are recomputed with :func:`reforge`.
    """
    from ..energy_records import analyze
    from ..energy_workflow import _request
    from ..telemetry import digest
    from ..workbench import _source
    sources = workspace["workbench"]["sources"]
    index = next(position for position, source in enumerate(sources) if source["source_id"] == source_id)
    old = sources[index]
    log = edited_log(base64.b64decode(old["bytes_b64"]), edit, reseal=True)
    new = _source({"kind": old["kind"], "label": old["label"], "bytes_b64": b64(serialize_log(log))})
    sources[index] = new
    evidence, data = new["evidence_id"], analyze(log)
    for record in workspace["workbench"]["bundles"]:
        if record["source_id"] != source_id:
            continue
        record["source_id"] = new["source_id"]
        native = record["native"]
        native["source"] = {"experiment_id": log["run_id"], "experiment_digest": digest(log),
                            "evidence": [{"artifact_ref": evidence, "sha256": evidence, "bytes_b64": new["bytes_b64"]}]}
        for step in (native["steps"][0], native["verification"]["reproduction"]):
            step["input_refs"], step["result"]["input_refs"] = [evidence], [evidence]
            step["request"] = _request(log, evidence)
            step["result"]["data"] = deepcopy(data)
    reforge(workspace)
    return new


def reseal_oscillator(*records: dict) -> None:
    """Recompute the unkeyed digests inside each record: its numerical_result_id, where it carries one, then its seal."""
    from ..operations.runner import numerical_result_id, seal
    for record in records:
        if "numerical_result_id" in record:
            record["numerical_result_id"] = numerical_result_id(record.get("operation_id"), record.get("data"))
        seal(record)


def reseal_catalog(workspace: dict) -> None:
    """Recompute the workbench's catalog-level replay-receipt seal over the (possibly forged) bundles.

    The seal is an unkeyed SHA-256 over CIW's public canonical JSON of the
    (bundle_id, replay_id) pairs, so any holder of the file recomputes it; a
    forgery that recomputes every downstream digest recomputes this one too.
    """
    from ..workbench import receipt_seal
    catalog = workspace["workbench"]
    catalog["replay_receipt_seal"] = receipt_seal(catalog["bundles"])


def reopen(workspace: dict, root: Path):
    """Write a (possibly forged) workspace and reopen it offline; refusal is an outcome."""
    from ..session import Session
    directory = Path(tempfile.mkdtemp(prefix="reopen-", dir=root))
    path = directory / "workspace.json"
    path.write_text(json.dumps(workspace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        session = Session.from_workspace(path, directory / "out")
    except Exception as exc:  # every refusal type is retained as the observed outcome
        return {"outcome": "refused", "error": type(exc).__name__, "message": str(exc)}, None
    return {"outcome": "accepted", "error": None, "message": None}, session


# ------------------------------------------------------------------ workspace views
class View:
    """Named handles into a deep-copied saved workspace."""

    def __init__(self, workspace: dict, fixture: dict):
        self.workspace = workspace
        self._fixture = fixture
        executions = {record["execution_id"]: record for record in workspace.get("executions", [])}
        results = {record["result_id"]: record for record in workspace["results"]}
        oscillator = fixture["oscillator"]
        self.E1, self.E2 = executions[oscillator["E1"]["execution_id"]], executions[oscillator["E2"]["execution_id"]]
        self.R1, self.R2 = results[oscillator["R1"]["result_id"]], results[oscillator["R2"]["result_id"]]
        self.RL = results[oscillator["RL"]["result_id"]]
        by_id = {record["bundle_id"]: record for record in workspace["workbench"]["bundles"]}
        self.records = {role: by_id[fixture["natives"][role]["bundle_digest"]]
                        for role in ("B0", "B0b", "Bother", "B1")}

    def native(self, role: str) -> dict:
        return self.records[role]["native"]

    def receipt(self) -> dict:
        return self.native("B1")["replay_receipts"][0]


@dataclass(frozen=True)
class Mutant:
    """One workspace forgery.

    ``pinned`` is ``accepted`` for a predicted survivor, otherwise the exact
    refusal message recorded from an observed run. Only the kill/survive
    outcome is a prediction; the message is a regression pin.
    """
    name: str
    task: str
    target: str
    recompute: str
    description: str
    pinned: str
    apply: Callable
    witness: Callable | None = None


def _outcome(pinned: str, accepted: bool, observed: str, error: str | None) -> dict:
    """Kill/survive prediction and message pin, reported separately."""
    predicted = "accepted" if pinned.startswith("accepted") else "killed"
    outcome = "accepted" if accepted else "killed"
    return {"predicted_outcome": predicted, "pinned_message": pinned, "observed_outcome": outcome,
            "observed": observed, "error": error, "killed": not accepted,
            "outcome_matches_prediction": outcome == predicted, "message_matches_pin": observed == pinned}


def run_mutant(mutant: Mutant, fixture: dict, root: Path, labels: dict) -> dict:
    workspace = deepcopy(fixture["workspace"])
    mutant.apply(View(workspace, fixture))
    observed, session = reopen(workspace, root)
    accepted = observed["outcome"] == "accepted"
    row = {"name": mutant.name, "task": mutant.task, "kind": "workspace", "target": mutant.target,
           "recompute": mutant.recompute, "description": mutant.description,
           **_outcome(mutant.pinned, accepted, "accepted" if accepted else observed["message"], observed["error"])}
    if session is not None and mutant.witness is not None:
        row["post_reopen"] = relabel(mutant.witness(session, fixture), labels)
    return row


def validator_row(name, task, target, description, pinned, call, recompute="none", by_design=None) -> dict:
    """Outcome of a pure offline validator on a synthetic record (``recompute`` as for workspace mutants).

    ``by_design`` names the documented behaviour when the validator does not
    claim to refuse the edit, so an acceptance is not a surviving mutant.
    """
    try:
        value = call()
    except Exception as exc:  # the refusal message is the observation
        observed, error = str(exc), type(exc).__name__
    else:
        observed, error = "accepted" if value is None else f"accepted:{value}", None
    row = {"name": name, "task": task, "kind": "validator", "target": target, "recompute": recompute,
           "description": description, **_outcome(pinned, error is None, observed, error)}
    if by_design is not None:
        row["accepted_by_design"] = by_design
    return row


def telemetry_shaped_bundle(tag: str) -> dict:
    """A synthetic record with the telemetry-session fields the ESM response validator reads.

    Workbench._validate_candidate submits only calibrated-observable or
    telemetry bundles; building a validated telemetry session needs provider
    checkouts, so this record carries only the schema, a digest and three step
    occurrences (ppda, stfe, gsie; no cbsr, so reconciliation is not_run).
    """
    from ..telemetry import _bundle_digest

    def occurrence(role):
        return "execution-" + hashlib.sha256(f"ciw-lab {tag} {role}".encode()).hexdigest()[:32]
    body = {"schema": "ciw.telemetry-session.v1",
            "session_id": "session-" + hashlib.sha256(f"ciw-lab {tag}".encode()).hexdigest()[:32],
            "configuration": {}, "steps": [{"runtime_ref": role, "execution_id": occurrence(role)}
                                           for role in ("ppda", "stfe", "gsie")]}
    body["bundle_digest"] = _bundle_digest(body)
    return body


def esm_case(native: dict) -> dict:
    """A minimal ESM candidate inspection that the pure validator accepts for ``native``.

    Only :func:`ciw.candidate_evidence.validate_response` runs; no ESM process,
    adapter binding or candidate store is involved. ``raw`` is CIW's canonical
    serialization of the bundle, as Workbench._validate_candidate passes it.
    """
    from ..telemetry import canonical
    raw = canonical(native)
    parameters = {"bundle_id": native["bundle_digest"], "inspected_at": "2026-01-01T00:00:00Z"}
    policy = {"review_context": {"requestId": "lab-request-1", "authority": "lab-operator-review"}}
    response = {"schema": "payload.instrument-candidate-inspection.v1", "inspectedAt": parameters["inspected_at"],
                "requestId": "lab-request-1", "authority": "lab-operator-review",
                "state": "ELIGIBLE_FOR_CANDIDATE_REVIEW", "bundleBytesDigest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "canonicalAdmission": "REFUSED", "canonicalStateMutated": False, "evidenceRetained": False,
                "releaseActivated": False, "sourceTruthClaimed": False, "independentlyVerified": False,
                "retractionHistoryComplete": False,
                "candidate": {"state": "UNADMITTED", "bundleDigest": native["bundle_digest"],
                              "executionIds": sorted(step["execution_id"] for step in native["steps"]),
                              "verification": {"outcome": "passed", "independent": False},
                              "reconciliation": {"status": "not_run"}}}
    return {"response": response, "parameters": parameters, "policy": policy, "raw": raw, "bundle": native}


def check_esm(case: dict, edit=None) -> None:
    """Run the pure ESM response validator on an edited copy of ``case``."""
    from ..candidate_evidence import validate_response
    case = deepcopy(case)
    if edit is not None:
        edit(case)
    return validate_response(case["response"], "inspect", case["bundle"], case["raw"], case["parameters"],
                             case["policy"])


def reseal_exchange_identity(artifact: dict, field: str) -> None:
    """Recompute an exchange content identity (schema, NUL, canonical JSON without ``field``) as any holder can."""
    payload = json.dumps({key: value for key, value in artifact.items() if key != field}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    artifact[field] = "sha256:" + hashlib.sha256(artifact["schema"].encode("utf-8") + b"\x00" + payload).hexdigest()


def exchange_artifact(schema: str, body: dict, field: str) -> dict:
    """A synthetic exchange artifact whose content identity is recomputed locally."""
    artifact = {"schema": schema, **deepcopy(body)}
    reseal_exchange_identity(artifact, field)
    return artifact


# ------------------------------------------------------------------ telemetry stack (T077)
# A fixed revision that is not the GSIE pin, so the forged-runtime witness is deterministic.
FORGED_REVISION = hashlib.sha1(b"ciw-lab forged provider revision").hexdigest()
# Runtime identity fields telemetry._runtime does not compare on replay: a forged adapter version and an injected key.
FORGED_ADAPTER_VERSION = "ciw-lab-forged-adapter-v9"
INJECTED_RUNTIME_KEY = ("audited_by", "ciw-lab forged independent audit")


def telemetry_manifest() -> dict:
    """The telemetry pins CIW declares (package data, as ciw.telemetry reads it)."""
    from importlib import resources
    return json.loads(resources.files("ciw").joinpath("telemetry-runtimes.json").read_text(encoding="utf-8"))


def telemetry_checkouts(stack) -> dict:
    """Each checkout of a bound telemetry stack against its ``ciw/telemetry-runtimes.json`` pin, without host paths.

    ``state`` is ``ready`` (clean and at the pin), ``refused`` (at the pin, but its working tree is not the pinned
    tree: modified, untracked or dirty), ``off_pin`` (another revision) or ``unreadable`` (not a Git repository
    root); ``reason`` says why for every state but ready. The recomputed tree is the lab's own hash of the working
    bytes, a second reading of the tree Git reports.
    """
    import subprocess
    from .exchange_provenance_bundles_providers import checkout_identity, ciw_pins, compare_with_pins
    pins, manifest, states = ciw_pins(), telemetry_manifest(), {}
    for role, name in TELEMETRY_REPOSITORIES.items():
        declared, pin = f"ciw/telemetry-runtimes.json[{role}]", manifest[role]["revision"]
        record = {"repository": f"{TELEMETRY_OWNER}/{name}", "pin": pin}
        try:
            identity = checkout_identity(Path(stack) / name)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            states[role] = dict(record, state="unreadable", reason=f"{name} in the bound telemetry stack is not a "
                                                                   f"readable Git repository root ({type(exc).__name__})")
            continue
        comparison = compare_with_pins(role, identity, pins)
        record.update(head=identity["head"], tree=identity["tree"], recomputed_tree=identity["recomputed_tree"],
                      matched=comparison["matched"], clean=comparison["clean"])
        if declared not in comparison["matched"]:
            record.update(state="off_pin", reason=f"{name} is at {identity['head']}, not the {declared} pin {pin}")
        elif not comparison["accepted"]:
            record.update(state="refused", reason=f"{name} is at the {declared} pin but its working tree is not the "
                                                  "pinned tree (modified, untracked or dirty files)")
        else:
            record.update(state="ready", reason=None)
        states[role] = record
    return states


def build_telemetry_fixture(root: Path, stack) -> dict:
    """A real telemetry session in the shared workbench on a bound, validated telemetry stack.

    Binds every stack checkout (Workbench.bind_workflow checks each pin again),
    retains the bundled synthetic telemetry source, executes ``ciw.telemetry.v1``
    with the bundled configuration (no reconciliation, so CBSR is bound but does
    not execute), saves, replays, saves again, classifies the saved workspace with
    ``ciw.lab.bridge`` and reopens it offline. Then two forgeries of the first
    save, each with every unkeyed digest over it recomputed (bundle digest, SET
    verification identity, catalog identity and replay-receipt seal), are
    reopened: a replaced GSIE runtime revision, which a replay with the stack
    bound then meets, and a replaced GSIE adapter_version with an injected key,
    which meets only the runtime identity comparison that replay makes before
    executing (``telemetry._runtime`` with the retained identity as expected; a
    full replay would add about 48 more interpreter probes). Runtime identities
    are returned as CIW retains them, host paths included; callers keep those
    out of reports.
    """
    import sys
    from ..instruments import make_demo_run
    from ..session import Session
    from ..telemetry import _bundle_digest, _runtime, replay_session
    from .bridge import classify_workspace
    from .runner import repository_path
    examples = repository_path("examples", "telemetry")
    if examples is None:
        raise FileNotFoundError("No repository examples are reachable for the telemetry source and configuration")
    source = (examples / "source.json").read_bytes()
    configuration = json.loads((examples / "configuration.json").read_text(encoding="utf-8"))
    bindings = {role: Path(stack) / name for role, name in TELEMETRY_REPOSITORIES.items()}
    session = Session(make_demo_run(), root / "telemetry-a")
    session.workbench.bind_workflow("telemetry", bindings)
    added = request(session, "source.add", {"kind": "telemetry", "label": "telemetry example",
                                            "bytes_b64": b64(source)})
    executed = request(session, "operation.execute", {"operation_id": "ciw.telemetry.v1", "parameters": {
        "source_id": added["source_id"], "configuration": configuration}})
    first = save(session)
    replayed = request(session, "bundle.replay", {"bundle_id": executed["bundle_id"]})
    saved = save(session)
    classification = classify_workspace(saved["path"])
    reopened = Session.from_workspace(saved["path"], root / "telemetry-b")
    available = next(entry["available"] for entry in reopened.workbench.describe_operations()
                     if entry["operation_id"] == "ciw.telemetry.v1")
    again = save(reopened)

    def forge(edit):
        """Reopen the first save with the GSIE runtime identity edited and every unkeyed digest recomputed."""
        forged = deepcopy(first["workspace"])
        record = forged["workbench"]["bundles"][0]
        native = record["native"]
        edit(native["runtimes"]["gsie"])
        native["bundle_digest"] = record["bundle_id"] = _bundle_digest(native)
        native["verification"]["subject_ref"] = native["bundle_digest"]
        reseal_exchange_identity(native["verification"], "verification_id")
        reseal_catalog(forged)
        outcome, forged_session = reopen(forged, root)
        retained = forged_session.workbench.get_bundle(record["bundle_id"]) if forged_session is not None else None
        return outcome, retained

    def attempt_replay(call):
        try:
            call()
        except ValueError as exc:  # AdapterRefusal is a ValueError too
            return {"outcome": "refused", "error": type(exc).__name__, "message": str(exc)}
        return {"outcome": "accepted", "error": None, "message": None}

    def forged_fields(runtime):
        runtime["adapter_version"] = FORGED_ADAPTER_VERSION
        runtime[INJECTED_RUNTIME_KEY[0]] = INJECTED_RUNTIME_KEY[1]

    outcome, retained = forge(lambda runtime: runtime.update(revision=FORGED_REVISION))
    # The workflow replay Workbench.replay runs, on the reopened retained bundle with the stack bound: it builds the
    # pinned adapters and compares each with the retained identity before executing anything.
    forged_replay = None if retained is None else attempt_replay(
        lambda: replay_session(retained, {role: bindings[role] for role in retained["runtimes"]}))
    fields_outcome, retained = forge(forged_fields)
    # That comparison for the edited role alone: the adapter replay_session builds for GSIE, with the retained
    # identity as expected (the other roles are unedited and compared as in the honest replay).
    fields_comparison = None if retained is None else attempt_replay(
        lambda: _runtime("gsie", bindings, retained["runtimes"]["gsie"]))
    interpreter, original = Path(sys.executable), session.workbench.get_bundle(executed["bundle_id"])
    return {"bound_roles": sorted(bindings), "configuration_roles": sorted(original["runtimes"]), "original": original,
            "replay": session.workbench.get_bundle(replayed["bundle"]["bundle_id"]),
            "receipt": replayed["replay_receipt"], "saved": saved["workspace"], "again": again["workspace"],
            "telemetry_available_after_reopen": available, "classification": classification["items"],
            "forged": {"reopen": outcome, "replay": forged_replay},
            "forged_fields": {"reopen": fields_outcome, "replay_comparison": fields_comparison},
            "bound_paths": {role: str(path.resolve()) for role, path in bindings.items()},
            "interpreter": str(interpreter.absolute()),
            "interpreter_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest()}


# ------------------------------------------------------------------ source-log edits
def _reverse_keys(value):
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def serialize_log(log: dict) -> bytes:
    """The fixture's own layout (two-space indent, trailing newline), so edits change only the edited tokens."""
    return (json.dumps(log, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def edited_log(raw: bytes, edit: Callable, *, reseal: bool) -> dict:
    """Apply ``edit`` to a parsed log; with ``reseal`` recompute its unkeyed log_digest as any holder can."""
    from ..energy_records import seal
    log = json.loads(raw.decode("utf-8"))
    unsigned = {key: value for key, value in log.items() if key != "log_digest"}
    edit(unsigned)
    return seal(unsigned) if reseal else {**unsigned, "log_digest": log["log_digest"]}


def integer_initial_covariance(log: dict) -> None:
    """Write the unit diagonal of both copies of initial_covariance as JSON integers (1.0 -> 1), consistently."""
    for matrix in (log["runtime"]["workload"]["solver_settings"]["initial_covariance"],
                   log["plan"]["solver"]["initial_covariance"]):
        for row in matrix:
            for index, value in enumerate(row):
                if type(value) is float and value == 1.0:
                    row[index] = 1


def rename_sensor(log: dict) -> None:
    """A metadata-only edit: the sensor name is retained but never analysed."""
    log["sensor"]["name"] = "renamed fixture device"


def measurement_edit(log: dict) -> None:
    """Lower the first measurement-phase counter sample by 50 mJ (it stays monotone)."""
    sample = log["phases"][3]["samples"][0]
    sample["energy_mj"] = str(int(sample["energy_mj"]) - 50)


def byte_variants(raw: bytes) -> dict:
    """Byte strings whose parsed JSON is canonically equal to ``raw`` (whitespace, order, spelling)."""
    parsed = json.loads(raw.decode("utf-8"))

    def dump(value, **options):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, **options).encode("utf-8")
    return {"original": raw, "crlf": raw.replace(b"\n", b"\r\n"),
            "minified": dump(parsed, separators=(",", ":")), "tab-indented": dump(parsed, indent="\t"),
            "sorted-keys": dump(parsed, indent=4, sort_keys=True),
            "reversed-keys": dump(_reverse_keys(parsed), indent=2),
            "trailing-whitespace": raw + b"\n\n \t\n",
            "float-spelling": raw.replace(b"1e-09", b"0.000000001")}


def resealed_variants(raw: bytes) -> dict:
    """Content edits whose unkeyed log_digest is recomputed: accepted as new, distinct evidence."""
    return {"metadata-renamed": serialize_log(edited_log(raw, rename_sensor, reseal=True)),
            "int-for-float-consistent": serialize_log(edited_log(raw, integer_initial_covariance, reseal=True))}


def refused_variants(raw: bytes) -> dict:
    """Re-encodings that change canonical content without resealing, or are not plain JSON."""
    return {"int-for-float-partial": raw.replace(b"1.0,", b"1,", 1),
            "int-for-float-consistent-unsealed": serialize_log(edited_log(raw, integer_initial_covariance,
                                                                          reseal=False)),
            "byte-order-mark": b"\xef\xbb\xbf" + raw}


def canonical_content(raw: bytes) -> bytes:
    """CIW's own canonical JSON of the parsed bytes (type-aware: 1 and 1.0 differ); a BOM is stripped first."""
    from ..telemetry import canonical
    return canonical(json.loads(raw.decode("utf-8-sig")))


def _retain_and_reopen(root: Path, tag: str, inputs: list) -> dict:
    """Retain and execute each (name, label, kind, bytes) in one fresh workbench, save and reopen it."""
    from ..instruments import make_demo_run
    from ..session import Session

    session = Session(make_demo_run(), root / f"{tag}-a")
    rows = []
    for name, label, kind, raw in inputs:
        source = energy_source(session, raw, label)
        bundle = energy_execute(session, source["source_id"])
        rows.append({"name": name, "label": label, "kind": kind, "raw": raw, "source": source,
                     "bundle_id": bundle["bundle_id"]})
    for row in rows:
        row["live"] = request(session, "source.get", {"source_id": row["source"]["source_id"]})
    saved = save(session)
    reopened = Session.from_workspace(saved["path"], root / f"{tag}-b")
    for row in rows:
        row["restored"] = request(reopened, "source.get", {"source_id": row["source"]["source_id"]})
        row["native"] = reopened.workbench.get_bundle(row["bundle_id"])
    return {"session": session, "rows": rows, "workspace_bytes": len(saved["bytes"])}


def build_variant_fixture(root: Path) -> dict:
    """Retain every fixture log and the byte variants of baseline in one workbench, each resealed variant in its own.

    The eight byte variants share one label, so any difference in their source
    identities comes from their bytes. A resealed content variant keeps the
    log's run_id, and CIW refuses to analyse two logs with one run_id and
    different log digests in one workbench, so each resealed variant gets a
    workbench of its own; that refusal is recorded separately.
    """
    from ..instruments import make_demo_run
    from ..session import Session
    from ..telemetry import canonical

    baseline = fixture_bytes("baseline")
    inputs = [(name, name, "fixture_log", fixture_bytes(name)) for name in FIXTURE_LOGS]
    inputs += [(name, "baseline/byte-variant", "byte_variant", raw) for name, raw in byte_variants(baseline).items()]
    main = _retain_and_reopen(root, "variants", inputs)
    groups = [main] + [_retain_and_reopen(root, f"resealed-{name}", [(name, f"baseline/resealed-{name}",
                                                                        "resealed_variant", raw)])
                       for name, raw in resealed_variants(baseline).items()]
    baseline_content = canonical_content(baseline)
    reference = main["rows"][0]["native"]["steps"][0]["result"]["data"]
    records = []
    for row in (row for group in groups for row in group["rows"]):
        raw, source, native = row["raw"], row["source"], row["native"]
        evidence = native["source"]["evidence"][0]
        step = native["steps"][0]
        data = step["result"]["data"]
        records.append({
            "name": row["name"], "label": row["label"], "kind": row["kind"], "byte_count": len(raw),
            "input_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "evidence_id": source["evidence_id"], "source_id": source["source_id"],
            "declared_byte_count": source["byte_count"],
            "live_bytes_equal": base64.b64decode(row["live"]["bytes_b64"]) == raw,
            "reopened_bytes_equal": base64.b64decode(row["restored"]["bytes_b64"]) == raw,
            "bundle_bytes_equal": base64.b64decode(evidence["bytes_b64"]) == raw,
            "artifact_ref": evidence["artifact_ref"], "experiment_id": native["source"]["experiment_id"],
            "experiment_digest": native["source"]["experiment_digest"],
            "log_digest": data["log_digest"], "numerical_result_id": step["numerical_result_id"],
            "data_keys_differing_from_baseline": sorted(key for key in set(data) | set(reference)
                                                        if canonical(data.get(key)) != canonical(reference.get(key))),
            "canonical_equal_to_baseline": canonical_content(raw) == baseline_content,
            "python_equal_to_baseline": json.loads(raw.decode("utf-8")) == json.loads(baseline.decode("utf-8")),
        })
    # Refused submissions go to the main live session after its save; none may retain a source.
    session = main["session"]
    before = len(request(session, "source.list", {})["sources"])
    refusals = {}
    for name, raw in refused_variants(baseline).items():
        refusals[name] = dict(attempt(session, "source.add", {"kind": KIND, "label": f"baseline/{name}",
                                                              "bytes_b64": b64(raw)}),
                              python_equal=json.loads(raw.decode("utf-8-sig")) == json.loads(baseline.decode("utf-8")),
                              canonical_equal=canonical_content(raw) == baseline_content)
    encoded = b64(baseline)
    transports = {"missing-padding": b64(b"A").rstrip("="), "line-wrapped": encoded[:76] + "\n" + encoded[76:],
                  "noncanonical-trailing-bits": "QR=="}
    for name, text in transports.items():
        refusals["base64/" + name] = attempt(session, "source.add", {"kind": KIND, "label": "transport", "bytes_b64": text})
    after = len(request(session, "source.list", {})["sources"])
    # One workbench, one run_id, two log digests: the resealed metadata variant cannot be analysed beside baseline.
    shared = Session(make_demo_run(), root / "shared-run-id")
    energy_execute(shared, energy_source(shared, baseline, "baseline")["source_id"])
    renamed = energy_source(shared, resealed_variants(baseline)["metadata-renamed"], "baseline/resealed-metadata-renamed")
    collision = attempt(shared, "operation.execute", {"operation_id": OPERATION,
                                                      "parameters": {"source_id": renamed["source_id"]}})
    origins = sorted({json.loads(fixture_bytes(name).decode("utf-8"))["origin"] for name in FIXTURE_LOGS})
    return {"records": records, "refusals": refusals, "workspace_bytes": main["workspace_bytes"],
            "sources_before_refusals": before, "sources_after_refusals": after, "fixture_origins": origins,
            "shared_run_id_execution": {key: collision.get(key) for key in ("outcome", "code", "message")}}
