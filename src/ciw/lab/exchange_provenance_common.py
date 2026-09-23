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
                  "session": r"session-[0-9a-f]{32}"}


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
        for kind, pattern in FRESH_PATTERNS.items():
            if re.fullmatch(pattern, value):
                return f"{kind}:<fresh>"
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
