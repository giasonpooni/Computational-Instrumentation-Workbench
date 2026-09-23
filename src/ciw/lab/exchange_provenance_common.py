"""Offline CIW session fixture and workspace forgery harness for tasks T077-T090.

Scope: drives the real CIW integrity layer (``ciw.session``, the operation
runner, ``ciw.workbench``, the energy-accuracy workflow, and the pure exchange
and candidate-evidence validators) through its request protocol and saved
workspaces, entirely offline. No provider checkout, GPU, network or hardware
is used; the energy-accuracy input is the bundled synthetic fixture log.

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


def reforge(workspace: dict, *, keep_verification=False) -> None:
    """Recompute every derived energy-bundle digest in catalog order, as a forger would.

    Bundle digests, verifications, receipts and catalog bundle identities are
    regenerated; renamed bundles are followed into later replay receipts.
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
    from ..operations.runner import seal
    for record in records:
        seal(record)


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


def exchange_artifact(schema: str, body: dict, field: str) -> dict:
    """A synthetic exchange artifact whose content identity is recomputed locally."""
    artifact = {"schema": schema, **deepcopy(body)}
    payload = json.dumps({key: value for key, value in artifact.items() if key != field}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    artifact[field] = "sha256:" + hashlib.sha256(schema.encode("utf-8") + b"\x00" + payload).hexdigest()
    return artifact


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
