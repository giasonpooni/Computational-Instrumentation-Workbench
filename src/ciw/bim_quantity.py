"""Native CSE scalar conditioning over exact retained IFC and observation bytes.

Only an explicitly independent measurement of a declared raw metre quantity is
conditioned. Frame labels are supplied applicability declarations, not surveyed
frame authority. No coordinate transform, clearance acceptance, or state release
is implied. CSE owns compilation, Gaussian conditioning, invariants and replay.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import math
import re
import struct
import uuid

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _json
from .core.covariance import _validate_matrix
from .declared_workload import DeclaredWorkflow, AUTHORITY, RESULT_SCHEMA, _text
from .telemetry import canonical, digest, byte_digest, _keys

PIN = {"role": "cse", "revision": "4b74abda40bba3277de69bf61e9e09283ae2d5b3",
       "source_root": ".", "module": "gat.session"}
POLICY = {"observation_model": "native_raw_quantity", "geometry_authority": "QUANTITY_ONLY",
          "cross_covariance": "explicit_independent_or_hold", "frame_transform": "not_performed",
          "state_admission": "not_performed"}
REASONS = {"conditioned", "unknown_cross_covariance", "frame_unresolved", "frame_mismatch",
           "ifc_binding_mismatch", "target_binding_mismatch", "unit_mismatch", "target_missing",
           "derived_quantity_unsupported", "source_units_assumed", "initial_invariants_failed",
           "invariant_rejection", "native_refusal"}


def _bytes(value, limit):
    if not isinstance(value, str):
        raise ValueError("Require canonical base64 evidence")
    raw = base64.b64decode(value, validate=True)
    if not 1 <= len(raw) <= limit or base64.b64encode(raw).decode() != value:
        raise ValueError("Evidence exceeds budget or is not canonical base64")
    return raw


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12 or (positive and not 1e-20 <= value <= 1e12):
        raise ValueError("Require bounded finite quantity/positive variance")


def _target(value):
    _keys(value, {"ifc_class", "global_id", "quantity"})
    for field in value.values():
        _text(field)


def _observation(source):
    observation = _json(_bytes(source["observation_bytes_b64"], 4096))
    _keys(observation, {"schema", "value", "variance", "unit", "frame", "ifc_sha256",
                        "ifc_class", "global_id", "quantity", "cross_covariance_policy"})
    if observation["schema"] != "ciw.bim-scalar-observation.v1":
        raise ValueError("Require a typed scalar observation")
    _number(observation["value"])
    _number(observation["variance"], positive=True)
    _target({key: observation[key] for key in ("ifc_class", "global_id", "quantity")})
    _text(observation["unit"])
    if observation["frame"] is not None:
        _text(observation["frame"])
    if not isinstance(observation["ifc_sha256"], str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", observation["ifc_sha256"]):
        raise ValueError("Observation must declare exact IFC byte identity")
    if observation["cross_covariance_policy"] not in ("independent", "unknown"):
        raise ValueError("Declare independent measurement noise or unknown cross covariance")
    return observation


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > 128 * 1024:
        raise ValueError("BIM source exceeds byte budget")
    source = _json(raw)
    canonical(source)
    _keys(source, {"schema", "experiment_id", "configuration", "ifc_bytes_b64",
                   "observation_bytes_b64", "target", "model_frame"})
    if source["schema"] != "ciw.bim-quantity-source.v1" or source["configuration"] != POLICY:
        raise ValueError("Require bounded BIM quantity source and explicit policy")
    _text(source["experiment_id"])
    _target(source["target"])
    if source["model_frame"] is not None:
        _text(source["model_frame"])
    text = _bytes(source["ifc_bytes_b64"], 64 * 1024).decode("utf-8")
    if len(re.findall(r"#\d+\s*=", text)) > 256:
        raise ValueError("IFC record budget is 256")
    _observation(source)
    return source


_BOOTSTRAP = r'''
import base64, hashlib, json, math, sys
sys.path.insert(0, sys.argv[1])
from gat.session import GatSession
from gat.ids import EntityId, VarId
from gat.ir.core import Role
from gat.engine.transform import ObserveQuantity
from gat.ledger import replay_ledger, verification_payload
from gat.errors import GatError
from gat.adapters.ifc.units import length_unit_context
if sys.byteorder != 'little': raise ValueError('CSE v1 state-byte contract requires little endian')
source = json.loads(sys.stdin.buffer.read())
ifc = base64.b64decode(source['ifc_bytes_b64'], validate=True)
observation_bytes = base64.b64decode(source['observation_bytes_b64'], validate=True)
observation = json.loads(observation_bytes)
sha = lambda raw: 'sha256:' + hashlib.sha256(raw).hexdigest()
target = source['target']
data = {'status': 'refused', 'reason': 'native_refusal', 'ifc_sha256': sha(ifc),
        'observation_sha256': sha(observation_bytes), 'target': target,
        'model_frame': source['model_frame'], 'geometry_authority': 'QUANTITY_ONLY',
        'unit_context': None, 'prior': None, 'posterior': None, 'ledger': None,
        'ledger_replay': None, 'invariants': None, 'native_error': None}
def state(world):
    return {'world_digest': world.digest(), 'belief_digest': world.belief.digest(),
            'quantities': [{'ifc_class': v.entity.ifc_class, 'global_id': v.entity.global_id,
                            'quantity': v.quantity, 'unit': world.module.slot(v).unit.value,
                            'role': world.module.slot(v).role.value} for v in world.full.index.vars],
            'mean': world.full.mu.tolist(), 'covariance': world.full.sigma.tolist()}
try:
    session = GatSession.from_text(ifc.decode('utf-8'), source=sha(ifc))
    initial = session.world
    if initial.binding.n_full > 64:
        raise ValueError('BIM quantity budget is 64 full variables')
    units = length_unit_context(session.source_file)
    data['unit_context'] = {'scale_to_metres': units.scale_to_metres, 'kind': units.kind,
                            'name': units.name, 'prefix': units.prefix, 'source_step_id': units.source_step_id,
                            'assumed': units.assumed}
    data['prior'] = state(initial)
    var = VarId(EntityId(target['ifc_class'], target['global_id']), target['quantity'])
    reason = None
    if observation['cross_covariance_policy'] == 'unknown': reason = 'unknown_cross_covariance'
    elif source['model_frame'] is None or observation['frame'] is None: reason = 'frame_unresolved'
    elif observation['frame'] != source['model_frame']: reason = 'frame_mismatch'
    elif observation['ifc_sha256'] != sha(ifc): reason = 'ifc_binding_mismatch'
    elif any(observation[k] != target[k] for k in target): reason = 'target_binding_mismatch'
    elif var not in initial.full.index: reason = 'target_missing'
    elif initial.module.slot(var).role != Role.RAW: reason = 'derived_quantity_unsupported'
    elif initial.module.slot(var).unit.value != 'm' or observation['unit'] != 'm': reason = 'unit_mismatch'
    elif units.assumed: reason = 'source_units_assumed'
    elif not session.initial_report.passed: reason = 'initial_invariants_failed'
    if reason:
        data.update(status='held', reason=reason)
    else:
        transformation = ObserveQuantity.single(var, observation['value'], math.sqrt(observation['variance']))
        result = session.run(transformation, provenance={
            'ifc_sha256': sha(ifc), 'observation_sha256': sha(observation_bytes),
            'model_frame': source['model_frame'], 'frame_authority': 'declared_not_surveyed',
            'cross_covariance_policy': observation['cross_covariance_policy']}, strict=False)
        data.update(status='accepted' if result.committed else 'refused',
                    reason='conditioned' if result.committed else 'invariant_rejection')
    data['posterior'] = state(session.world)
    data['ledger'] = session.ledger.to_dict()
    replay = replay_ledger(initial, session.ledger)
    if replay.world.digest() != session.world.digest(): raise ValueError('Native CSE ledger replay differed')
    data['ledger_replay'] = {'world_digest': replay.world.digest(), 'accepted': replay.accepted,
                             'rejected': replay.rejected, 'non_state': replay.non_state,
                             'events_replayed': replay.events_replayed, 'head': replay.head}
    data['invariants'] = verification_payload(session.verify())
except GatError as error:
    data.update(status='refused', reason='native_refusal', native_error=type(error).__name__)
print(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
'''


def _state(value):
    _keys(value, {"world_digest", "belief_digest", "quantities", "mean", "covariance"})
    for key in ("world_digest", "belief_digest"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]):
            raise ValueError("Invalid native world/belief identity")
    quantities = value["quantities"]
    if not isinstance(quantities, list) or not 1 <= len(quantities) <= 64:
        raise ValueError("BIM quantity state exceeds budget")
    identities = set()
    for quantity in quantities:
        _keys(quantity, {"ifc_class", "global_id", "quantity", "unit", "role"})
        _target({k: quantity[k] for k in ("ifc_class", "global_id", "quantity")})
        _text(quantity["unit"])
        if quantity["role"] not in ("raw", "derived"):
            raise ValueError("Unknown native quantity role")
        identity = tuple(quantity[k] for k in ("ifc_class", "global_id", "quantity"))
        if identity in identities:
            raise ValueError("Duplicate native quantity")
        identities.add(identity)
    count = len(quantities)
    if not isinstance(value["mean"], list) or len(value["mean"]) != count or not isinstance(value["covariance"], list) or len(value["covariance"]) != count:
        raise ValueError("Native quantity dimensions differ")
    for item in value["mean"]:
        _number(item)
    for row in value["covariance"]:
        if not isinstance(row, list) or len(row) != count:
            raise ValueError("Native covariance dimensions differ")
        for item in row:
            _number(item)
    for i in range(count):
        if value["covariance"][i][i] < 0 or any(value["covariance"][i][j] != value["covariance"][j][i] for j in range(count)):
            raise ValueError("Native covariance must be symmetric with nonnegative marginals")
    # Reuse the workbench covariance domain: dimensionless correlation
    # eigenvalues >= -1e-10, with exactly zero rows for zero variances. Native
    # derived quantities make the full view singular; do not require full rank,
    # add jitter, clip eigenvalues, or replace the retained covariance entries.
    _validate_matrix(value["covariance"], count)
    raw = [i for i, quantity in enumerate(quantities) if quantity["role"] == "raw"]
    if not raw:
        raise ValueError("Native BIM state requires at least one raw quantity")
    _validate_matrix([[value["covariance"][i][j] for j in raw] for i in raw], len(raw))


def _report(report):
    _keys(report, {"passed", "results"})
    if type(report["passed"]) is not bool or not isinstance(report["results"], list) or not 1 <= len(report["results"]) <= 512:
        raise ValueError("Invalid native invariant report")
    for row in report["results"]:
        _keys(row, {"invariant_id", "status", "subject", "residual", "detail"})
        if row["status"] not in ("PASS", "WARN", "FAIL"):
            raise ValueError("Unknown native invariant status")
        for key in ("invariant_id", "subject"):
            _text(row[key])
        if not isinstance(row["detail"], str) or len(row["detail"]) > 4096:
            raise ValueError("Unbounded native invariant detail")
        _number(row["residual"])
    if report["passed"] != all(row["status"] != "FAIL" for row in report["results"]):
        raise ValueError("Native invariant verdict contradicts retained checks")


def _state_digests(state, module_digest):
    # These are the owner's documented binary commitments, not an estimator.
    packed = lambda values: b"".join(struct.pack("<d", float(v)) for v in values)
    full = packed(state["mean"]) + packed(v for row in state["covariance"] for v in row)
    rows = [i for i, q in enumerate(state["quantities"]) if q["role"] == "raw"]
    raw = packed(state["mean"][i] for i in rows) + packed(state["covariance"][i][j] for i in rows for j in rows)
    if (state["world_digest"] != sha256(module_digest.encode() + full).hexdigest() or
            state["belief_digest"] != sha256(raw).hexdigest()):
        raise ValueError("Native state bytes differ from world/belief commitments")


def _check_data(source, data):
    _keys(data, {"status", "reason", "ifc_sha256", "observation_sha256", "target", "model_frame",
                 "geometry_authority", "unit_context", "prior", "posterior", "ledger", "ledger_replay", "invariants", "native_error"})
    if (data["status"] not in ("accepted", "held", "refused") or data["reason"] not in REASONS or
            data["target"] != source["target"] or data["model_frame"] != source["model_frame"] or
            data["geometry_authority"] != "QUANTITY_ONLY" or
            data["ifc_sha256"] != byte_digest(_bytes(source["ifc_bytes_b64"], 64 * 1024)) or
            data["observation_sha256"] != byte_digest(_bytes(source["observation_bytes_b64"], 4096))):
        raise ValueError("Native BIM result must retain declared evidence, target and authority")
    if data["reason"] == "native_refusal":
        if data["status"] != "refused" or not isinstance(data["native_error"], str) or any(data[k] is not None for k in ("posterior", "ledger", "ledger_replay", "invariants")):
            raise ValueError("Native refusal cannot claim a completed transition")
        if data["prior"] is not None:
            _state(data["prior"])
        return
    if data["native_error"] is not None:
        raise ValueError("Completed native assessment cannot contain unbound error")
    _state(data["prior"])
    _state(data["posterior"])
    if data["prior"]["quantities"] != data["posterior"]["quantities"]:
        raise ValueError("Native conditioning changed quantity identity")
    unit_context = data["unit_context"]
    _keys(unit_context, {"scale_to_metres", "kind", "name", "prefix", "source_step_id", "assumed"})
    _number(unit_context["scale_to_metres"], positive=True)
    if type(unit_context["assumed"]) is not bool:
        raise ValueError("Invalid source unit authority")
    observation = _observation(source)
    prior = data["prior"]
    target = next((q for q in prior["quantities"] if all(q[k] == source["target"][k] for k in source["target"])), None)
    reason = None
    if observation["cross_covariance_policy"] == "unknown": reason = "unknown_cross_covariance"
    elif source["model_frame"] is None or observation["frame"] is None: reason = "frame_unresolved"
    elif observation["frame"] != source["model_frame"]: reason = "frame_mismatch"
    elif observation["ifc_sha256"] != data["ifc_sha256"]: reason = "ifc_binding_mismatch"
    elif any(observation[k] != source["target"][k] for k in source["target"]): reason = "target_binding_mismatch"
    elif target is None: reason = "target_missing"
    elif target["role"] != "raw": reason = "derived_quantity_unsupported"
    elif target["unit"] != "m" or observation["unit"] != "m": reason = "unit_mismatch"
    elif unit_context["assumed"]: reason = "source_units_assumed"
    elif not data["ledger"]["events"][0]["verification"]["passed"]: reason = "initial_invariants_failed"
    if reason is not None and (data["status"] != "held" or data["reason"] != reason):
        raise ValueError("Inapplicable measurement must remain held")
    if reason is None and data["status"] not in ("accepted", "refused"):
        raise ValueError("Held outcome lacks declared incompatibility")
    expected_reason = "conditioned" if data["status"] == "accepted" else "invariant_rejection" if data["status"] == "refused" else reason
    if data["reason"] != expected_reason:
        raise ValueError("Native BIM outcome reason mismatch")
    ledger = data["ledger"]
    _keys(ledger, {"format", "schema_version", "runtime_contract", "events", "integrity"})
    if ledger["format"] != "gat-execution-ledger" or ledger["schema_version"] != 1 or ledger["runtime_contract"] != "gat-world-v1":
        raise ValueError("Require native CSE execution ledger")
    events = ledger["events"]
    count = 1 if data["status"] == "held" else 2
    if not isinstance(events, list) or len(events) != count:
        raise ValueError("Unexpected native ledger topology")
    previous = "0" * 64
    for seq, event in enumerate(events):
        _keys(event, {"seq", "kind", "operation", "provenance", "prior_world_digest", "result_world_digest",
                      "verification", "verification_digest", "error_type", "error_message", "error_digest", "previous_hash", "event_hash"})
        if event["seq"] != seq or event["previous_hash"] != previous:
            raise ValueError("Broken native ledger chain")
        _report(event["verification"])
        if event["verification_digest"] != sha256(canonical(event["verification"])).hexdigest():
            raise ValueError("Native ledger invariant commitment differs")
        if event["event_hash"] != sha256(canonical({k: v for k, v in event.items() if k != "event_hash"})).hexdigest():
            raise ValueError("Native ledger event content mismatch")
        previous = event["event_hash"]
    if ledger["integrity"] != {"algorithm": "sha256", "head": previous} or events[0]["result_world_digest"] != prior["world_digest"] or events[-1]["result_world_digest"] != data["posterior"]["world_digest"]:
        raise ValueError("Native ledger world binding mismatch")
    genesis = events[0]
    _keys(genesis["operation"], {"belief_digest", "module_digest", "configuration_digest", "runtime_contract"})
    if (genesis["kind"] != "genesis" or genesis["prior_world_digest"] != prior["world_digest"] or
            genesis["operation"]["belief_digest"] != prior["belief_digest"] or
            genesis["operation"]["runtime_contract"] != "gat-world-v1" or genesis["provenance"] != {} or
            any(genesis[k] is not None for k in ("error_type", "error_message", "error_digest"))):
        raise ValueError("Invalid native genesis binding")
    for key in ("module_digest", "configuration_digest"):
        if not re.fullmatch(r"[a-f0-9]{64}", genesis["operation"][key]):
            raise ValueError("Invalid native model/configuration commitment")
    _state_digests(prior, genesis["operation"]["module_digest"])
    _state_digests(data["posterior"], genesis["operation"]["module_digest"])
    if data["status"] != "accepted" and canonical(data["prior"]) != canonical(data["posterior"]):
        raise ValueError("Held/refused conditioning must preserve the exact prior state")
    replay = data["ledger_replay"]
    _keys(replay, {"world_digest", "accepted", "rejected", "non_state", "events_replayed", "head"})
    expected = {"world_digest": data["posterior"]["world_digest"], "accepted": int(data["status"] == "accepted"),
                "rejected": int(data["status"] == "refused"), "non_state": 0,
                "events_replayed": count - 1, "head": previous}
    if replay != expected:
        raise ValueError("Native ledger replay outcome mismatch")
    _report(data["invariants"])
    expected_invariants = events[-1]["verification"] if data["status"] == "accepted" else genesis["verification"]
    if canonical(data["invariants"]) != canonical(expected_invariants):
        raise ValueError("Reported invariants differ from the retained final world")
    if count == 2:
        event = events[1]
        expected_operation = {"op": "observe_quantity", "measurements": [{"var": {
            "entity": {"ifc_class": source["target"]["ifc_class"], "global_id": source["target"]["global_id"]},
            "quantity": source["target"]["quantity"]}, "value": float(observation["value"]),
            "noise_sigma": math.sqrt(observation["variance"])}]}
        if (canonical(event["operation"]) != canonical(expected_operation) or
                event["kind"] != ("transition" if data["status"] == "accepted" else "rejection") or
                event["prior_world_digest"] != prior["world_digest"] or event["provenance"] != {
                    "ifc_sha256": data["ifc_sha256"], "observation_sha256": data["observation_sha256"],
                    "model_frame": source["model_frame"], "frame_authority": "declared_not_surveyed",
                    "cross_covariance_policy": "independent"}):
            raise ValueError("Native ledger does not execute the declared scalar observation")
        if data["status"] == "accepted":
            if not event["verification"]["passed"] or any(event[k] is not None for k in ("error_type", "error_message", "error_digest")):
                raise ValueError("Accepted transition must pass native invariants")
        else:
            if event["verification"]["passed"] or event["error_type"] != "VerificationError":
                raise ValueError("Refused transition must retain native invariant failure")
            _text(event["error_message"])
            expected_error = sha256((event["error_type"] + "\0" + event["error_message"]).encode()).hexdigest()
            if event["error_digest"] != expected_error:
                raise ValueError("Native rejection error binding differs")


class BimQuantityWorkflow(DeclaredWorkflow):
    def __init__(self):
        self.kind, self.pin, self.role = "bim-quantity", deepcopy(PIN), "cse"
        self.ROLES = {"cse"}
        self.SOURCE_SCHEMA = "ciw.bim-quantity-source.v1"
        self.schema = "ciw.bim-quantity-session.v1"
        self.operation = "ciw.bim-quantity.v1"

    def _source(self, raw):
        return _source(raw)

    def _step(self, source, evidence_id, bound):
        adapter, runtime, _ = bound
        if self._runtime_projection(adapter.runtime_identity()) != self._runtime_projection(runtime):
            raise ValueError("CSE provider identity changed")
        code, raw = adapter._run(_BOOTSTRAP, [str(adapter.source_root)], canonical(source))
        adapter.runtime_identity()
        if code:
            raise AdapterRefusal("BIM_QUANTITY_REFUSED", "Pinned CSE refused the bounded IFC workload")
        data = _json(raw)
        _check_data(source, data)
        occurrence = "execution-" + uuid.uuid4().hex
        result = {"schema": RESULT_SCHEMA, "operation_id": self.operation, "execution_ref": occurrence,
                  "input_refs": [evidence_id], "data": data, "authority": AUTHORITY}
        result["result_id"] = digest(result)
        numerical = {"operation_id": self.operation, "data": data}
        return {"runtime_ref": self.role, "operation_id": self.operation, "execution_id": occurrence,
                "input_refs": [evidence_id], "request": source, "request_sha256": digest(source),
                "result": result, "result_sha256": digest(result), "result_id": result["result_id"],
                "numerical_result": numerical, "numerical_result_id": digest(numerical)}

    def _validate_step(self, step, source, evidence_id):
        _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256", "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
        if step["runtime_ref"] != self.role or step["operation_id"] != self.operation or step["input_refs"] != [evidence_id] or canonical(step["request"]) != canonical(source):
            raise ValueError("BIM request/operation/evidence binding mismatch")
        if not isinstance(step["execution_id"], str) or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]):
            raise ValueError("Invalid native execution occurrence")
        result = step["result"]
        _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
        _check_data(source, result["data"])
        if (result["schema"] != RESULT_SCHEMA or result["authority"] != AUTHORITY or result["operation_id"] != self.operation or
                result["execution_ref"] != step["execution_id"] or result["input_refs"] != [evidence_id] or
                result["result_id"] != digest({k: v for k, v in result.items() if k != "result_id"}) or result["result_id"] != step["result_id"] or
                canonical(step["numerical_result"]) != canonical({"operation_id": self.operation, "data": result["data"]})):
            raise ValueError("BIM result binding or authority mismatch")
        for key, content in (("request_sha256", source), ("result_sha256", result), ("numerical_result_id", step["numerical_result"])):
            if step[key] != digest(content):
                raise ValueError("BIM step content mismatch")


workflow = BimQuantityWorkflow()
