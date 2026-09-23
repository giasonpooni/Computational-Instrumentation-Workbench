"""Offline CIW session fixture and workspace forgery harness for tasks T077-T090.

Scope: drives the real CIW integrity layer (``ciw.session``, the operation
runner, ``ciw.workbench``, the energy-accuracy workflow, and the pure exchange
and candidate-evidence validators) through its request protocol and saved
workspaces, entirely offline. No provider checkout, GPU, network or hardware
is used; the energy-accuracy input is the bundled synthetic fixture log.

A forgery here edits a saved workspace and recomputes only unkeyed SHA-256
digests with CIW's public canonicalization, which any holder of the file can
do. A mutation accepted on reopen is a surviving mutant: the retained records
are content-consistent, not authenticated. Nothing here changes CIW code, and
nothing here says whether a forged record would mislead a particular reader.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
ENERGY_DIR = REPO_ROOT / "examples" / "energy-accuracy"
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


def fixture_available() -> bool:
    return all((ENERGY_DIR / f"{name}.json").is_file() for name in FIXTURE_LOGS)


def fixture_bytes(name: str) -> bytes:
    return (ENERGY_DIR / f"{name}.json").read_bytes()


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
    """Oscillator operations, energy bundles, replays, save, reopen and an independent session."""
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
    # An independent session analysing the same bytes, plus a relabelled copy.
    independent = Session(run, root / "session-c")
    source_c = energy_source(independent, baseline, "baseline")
    relabelled = energy_source(independent, baseline, "baseline (relabelled copy)")
    b4 = energy_execute(independent, source_c["source_id"])
    bundles = {"B0": b0["bundle_id"], "B0b": b0b["bundle_id"], "Bother": bother["bundle_id"],
               "B1": replay["bundle"]["bundle_id"], "B2": replay_reopened["bundle"]["bundle_id"],
               "B3": replay_of_replay["bundle"]["bundle_id"]}
    natives = {role: reopened.workbench.get_bundle(identity) for role, identity in bundles.items()}
    natives["B4"] = independent.workbench.get_bundle(b4["bundle_id"])
    return {
        "run": deepcopy(run), "session_ids": [session.session_id, reopened.session_id, independent.session_id],
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
    name: str
    task: str
    target: str
    recompute: str
    description: str
    predicted: str
    apply: Callable
    witness: Callable | None = None


def run_mutant(mutant: Mutant, fixture: dict, root: Path, labels: dict) -> dict:
    workspace = deepcopy(fixture["workspace"])
    mutant.apply(View(workspace, fixture))
    observed, session = reopen(workspace, root)
    row = {"name": mutant.name, "task": mutant.task, "target": mutant.target, "recompute": mutant.recompute,
           "description": mutant.description, "predicted": mutant.predicted,
           "observed": observed["outcome"] if observed["outcome"] == "accepted" else observed["message"],
           "error": observed["error"]}
    row["killed"] = observed["outcome"] == "refused"
    row["matches_prediction"] = row["observed"] == mutant.predicted
    if session is not None and mutant.witness is not None:
        row["post_reopen"] = relabel(mutant.witness(session, fixture), labels)
    return row


def validator_row(name, task, target, description, predicted, call) -> dict:
    """Outcome of a pure offline validator on a synthetic, locally recomputed record."""
    try:
        value = call()
    except Exception as exc:  # the refusal message is the observation
        observed, error = str(exc), type(exc).__name__
    else:
        observed, error = "accepted" if value is None else f"accepted:{value}", None
    return {"name": name, "task": task, "target": target, "recompute": "local", "description": description,
            "predicted": predicted, "observed": observed, "error": error,
            "killed": error is not None, "matches_prediction": observed == predicted}


def esm_case(native: dict) -> dict:
    """A minimal ESM candidate inspection that the pure validator accepts for ``native``.

    Only :func:`ciw.candidate_evidence.validate_response` runs; no ESM process,
    adapter binding or candidate store is involved.
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


def _reverse_keys(value):
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


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


def refused_variants(raw: bytes) -> dict:
    """Numerically equal re-encodings that are not canonically equal, or not plain JSON."""
    return {"int-for-float": raw.replace(b"1.0,", b"1,", 1), "byte-order-mark": b"\xef\xbb\xbf" + raw}


def canonical_content(raw: bytes) -> str:
    """Type-aware canonical text of parsed JSON (1 and 1.0 differ); a BOM is stripped first."""
    return json.dumps(json.loads(raw.decode("utf-8-sig")), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def build_variant_fixture(root: Path) -> dict:
    """Retain every fixture log and every baseline byte variant, execute, save and reopen."""
    from ..instruments import make_demo_run
    from ..session import Session

    session = Session(make_demo_run(), root / "variants-a")
    baseline = fixture_bytes("baseline")
    inputs = [(name, "fixture_log", fixture_bytes(name)) for name in FIXTURE_LOGS]
    inputs += [(f"baseline/{name}", "byte_variant", raw) for name, raw in byte_variants(baseline).items()]
    rows = []
    for label, kind, raw in inputs:
        source = energy_source(session, raw, label)
        bundle = energy_execute(session, source["source_id"])
        rows.append({"label": label, "kind": kind, "raw": raw, "source": source, "bundle_id": bundle["bundle_id"]})
    live = {row["label"]: request(session, "source.get", {"source_id": row["source"]["source_id"]}) for row in rows}
    saved = save(session)
    reopened = Session.from_workspace(saved["path"], root / "variants-b")
    baseline_content = canonical_content(baseline)
    records = []
    for row in rows:
        raw, source = row["raw"], row["source"]
        restored = request(reopened, "source.get", {"source_id": source["source_id"]})
        native = reopened.workbench.get_bundle(row["bundle_id"])
        evidence = native["source"]["evidence"][0]
        step = native["steps"][0]
        records.append({
            "label": row["label"], "kind": row["kind"], "byte_count": len(raw),
            "input_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "evidence_id": source["evidence_id"], "source_id": source["source_id"],
            "declared_byte_count": source["byte_count"],
            "live_bytes_equal": base64.b64decode(live[row["label"]]["bytes_b64"]) == raw,
            "reopened_bytes_equal": base64.b64decode(restored["bytes_b64"]) == raw,
            "bundle_bytes_equal": base64.b64decode(evidence["bytes_b64"]) == raw,
            "artifact_ref": evidence["artifact_ref"], "experiment_digest": native["source"]["experiment_digest"],
            "log_digest": step["result"]["data"]["log_digest"], "numerical_result_id": step["numerical_result_id"],
            "canonical_equal_to_baseline": canonical_content(raw) == baseline_content,
            "python_equal_to_baseline": json.loads(raw.decode("utf-8")) == json.loads(baseline.decode("utf-8")),
        })
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
    return {"records": records, "refusals": refusals, "workspace_bytes": len(saved["bytes"])}
