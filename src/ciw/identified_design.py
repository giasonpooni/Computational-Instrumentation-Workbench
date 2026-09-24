"""Identified dynamics to budgeted observation advice over retained telemetry.

Provider code is fixed here and source-pinned. Artifacts carry data, never code
or repository bindings. Prediction and design are conditional on an identified
point model; unknown parameter uncertainty is never silently set to zero.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import math
from pathlib import Path
import re
import uuid

from . import calibrated_observable as calibrated
from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .exchange import _identity, _read
from .session import write_json
from .core.canonical import canonical, digest, byte_digest, bundle_digest, utc_now
from .pipelines import pin_map

SCHEMA = "ciw.identified-design-session.v1"
SOURCE_SCHEMA = "ciw.identified-design-input.v1"
RESULT_SCHEMA = "notation.instrument.result-artifact.v1"
MAX_BYTES = 8 * 1024 * 1024
# How retained verification is produced; the descriptor must declare the same.
VERIFICATION_METHOD = "pinned_set_replay_verification"
OPERATIONS = (
    ("sidt", "sidt.declared-lti-identification.v1"),
    ("oit", "ciw.identified-candidate-observability.v1"),
    ("gsie", "ciw.identified-state-prediction.v1"),
    ("edspt", "edspt.budgeted-next-observation.v1"),
    ("ywir", "ywir.observation-design-token-admission.v1"),
)
ROLES = calibrated.ROLES | {"sidt", "edspt", "ywir"}
SCOPE = "conditional_on_identified_point_model"

_BOOTSTRAP = r'''
import dataclasses, enum, json, math, sys
import numpy as np
role, root = sys.argv[1:3]
sys.path.insert(0, root)
request = json.loads(sys.stdin.buffer.read())
d = request['inputs']
def native(v):
    if dataclasses.is_dataclass(v): return native(dataclasses.asdict(v))
    if isinstance(v, dict): return {k:native(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [native(x) for x in v]
    if isinstance(v, np.ndarray): return native(v.tolist())
    if isinstance(v, enum.Enum): return v.value
    if isinstance(v, float) and not math.isfinite(v): return None
    return v
if role == 'sidt':
    from sidt import identify_declared
    result = identify_declared(d['declaration'], execution_ref=d['execution_ref'])
elif role == 'oit':
    from oit import lti_observability
    rows = []
    for item in d['candidates']:
        assessed = lti_observability(transition=d['transition'], observation=item['observation_matrix'],
            horizon=d['horizon'], state_names=d['state_names'], state_scales=d['state_scales'],
            condition_limit=d['condition_limit'])
        rows.append({'candidate_id':item['candidate_id'], 'status':assessed.status,
            'rank':assessed.rank, 'condition_number':native(assessed.condition_number),
            'condition_limit':assessed.condition_limit, 'assessment':native(assessed),
            'observation_matrix':item['observation_matrix']})
    result = {'model_result_id':d['model_result_id'], 'transition':d['transition'],
              'horizon':d['horizon'], 'candidates':rows,
              'eligible_candidate_ids':[r['candidate_id'] for r in rows if r['status']=='observable']}
elif role == 'gsie':
    from geometric_state_inference import StatePrior, LinearDynamics, predict
    predicted = predict(StatePrior(**d['prior']), LinearDynamics(**d['dynamics']), d['target_time'])
    result = {k:native(getattr(predicted,k)) for k in ('time','mean','covariance','frame_id','units',
        'state_id','dynamics_model_id','predecessor_state_id','operation_ref')}
    result.update(replay_snapshot=predicted.replay_snapshot, uncertainty_scope=d['uncertainty_scope'],
        parameter_covariance_status='unknown', model_result_id=d['model_result_id'])
elif role == 'edspt':
    from edspt import ParameterCoordinate, Candidate, ModelPrior, BudgetedCandidate, rank_budgeted_candidates
    coordinates = tuple(ParameterCoordinate(**c) for c in d['coordinates'])
    scales = np.array([c.scale for c in coordinates])
    covariance = np.array(d['prior_covariance']) / scales[:,None] / scales[None,:]
    prior = ModelPrior(model_result_id=d['model_result_id'], coordinates=coordinates, covariance=covariance)
    candidates = [BudgetedCandidate(candidate=Candidate(c['candidate_id'],coordinates,
        np.array(c['observation_matrix'])*scales[None,:],c['noise_covariance']),cost=c['cost'],
        cost_unit=c['cost_unit'], model_result_id=d['model_result_id'],
        prior_cross_covariance_policy=c['prior_cross_covariance_policy']) for c in d['candidates']]
    result = rank_budgeted_candidates(candidates,prior=prior,available_budget=d['budget'],
        budget_unit=d['cost_unit'],criterion=d['criterion']).to_dict()
elif role == 'ywir':
    from ywir import evaluate_token_admission
    result = evaluate_token_admission(d)
elif role == 'set':
    from state_estimation_testbed.contracts import validate_result_artifact, validate_verification_artifact
    for artifact in d['results']: validate_result_artifact(artifact)
    validate_verification_artifact(d['verification'])
    result = {'status':'conformant', 'scope':'existing_exchange_envelopes_only'}
else: raise ValueError('Unsupported fixed design adapter')
print(json.dumps(native(result),allow_nan=False,ensure_ascii=False))
'''


def _pins():
    pins = pin_map("identified-design")
    if set(pins) != calibrated.ROLES | {"sidt", "edspt", "ywir"}:
        raise ValueError("Unexpected identified design runtime pins")
    return pins


def _adapters(repositories, expected=None):
    if set(repositories) != ROLES:
        raise ValueError("Bind exactly the eleven identified-design repositories")
    adapters = {}
    for role, pin in _pins().items():
        retained = expected[role] if expected else None
        adapter = PinnedSubprocessAdapter(repositories[role], pin["revision"], pin["module"],
            source_root=pin["source_root"],
            expected_python_sha256=retained["python_sha256"] if retained else None,
            expected_python_version=retained["python_version"] if retained else None,
            expected_dependencies=retained["dependencies"] if retained else None)
        identity = adapter.runtime_identity()
        if retained and any(identity[k] != retained[k] for k in (
            "schema", "adapter_version", "revision", "source_tree", "module", "source_root",
            "python_sha256", "python_version", "dependencies")):
            raise ValueError("Identified-design runtime identity mismatch")
        adapters[role] = adapter
    return adapters


def _invoke(role, adapters, request):
    raw = canonical(request)
    if len(raw) > MAX_BYTES:
        raise ValueError("Design operation exceeds byte budget")
    adapter = adapters[role]
    adapter.runtime_identity()
    code, output = adapter._run(_BOOTSTRAP, [role, str(adapter.source_root)], raw)
    if code:
        raise AdapterRefusal("DESIGN_" + role.upper() + "_REFUSED", "Pinned " + role + " operation refused declared inputs")
    adapter.runtime_identity()
    result = _json(output)
    if not isinstance(result, dict):
        raise ValueError("Design provider must return an object")
    return result


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing identified-design fields")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty declared identifier is required")


def _refs(value):
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        raise ValueError("Unique nonempty evidence references are required")
    for item in value:
        _text(item)


def _declared_refs(source):
    refs = list(source["prediction"]["process_covariance_evidence_refs"])
    for c in source["design"]["candidates"]:
        refs.extend(c["evidence_refs"])
    for split in ("training","holdout"):
        if split in source["identification"]:
            rows = source["identification"][split]
            refs.extend(rows["evidence_refs"])
            refs.extend(rows["sample_refs"])
    conditioning = source["identification"].get("conditioning_reference")
    if conditioning is not None:
        refs.append(conditioning)
    return set(refs)


def _source(raw, upstream):
    try:
        return _source_inner(raw, upstream)
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed identified-design input") from exc


def _source_inner(raw, upstream):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("Design input must be bounded exact bytes")
    s = _json(raw)
    _keys(s, {"schema", "experiment_id", "claim_scope", "identification", "prediction", "design", "token_admission"})
    if s["schema"] != SOURCE_SCHEMA:
        raise ValueError("Unsupported identified-design input")
    _text(s["experiment_id"])
    _text(s["claim_scope"])
    calibrated.inspect_session(upstream)
    state = upstream["steps"][4]["result"]["data"]
    names = upstream["configuration"]["observability"]["state_names"]
    ident, prediction, design = s["identification"], s["prediction"], s["design"]
    if not isinstance(ident["input_names"],list) or len(ident["input_names"]) > 16:
        raise ValueError("The identified model supports at most sixteen declared inputs")
    if (ident["state_names"] != names or ident["state_units"] != state["units"] or
        ident["state_frame"] != state["frame_id"]):
        raise ValueError("Identification state coordinates differ from retained GSIE state")
    experiment = calibrated._source(calibrated._validate(upstream))
    if any(c["clock_model"]["reference_frame"]["clock_id"] != ident["clock_frame"] for c in experiment["channels"]):
        raise ValueError("Identification and telemetry must declare the same reference clock")
    for split in ("training", "holdout"):
        if split in ident:
            times = ident[split]["sample_times"]
            if not isinstance(times, list) or not 2 <= len(times) <= 1025:
                raise ValueError("Identification requires two to 1025 ordered sample times")
            if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
                raise ValueError("Identification sample times must be finite numbers")
            if times[-1] > state["time"]:
                raise ValueError("Identification cannot use samples after the retained prior time")
            for field,count,width in (("states",len(times),len(names)),("inputs",len(times)-1,len(ident["input_names"]))):
                rows = ident[split][field]
                if (not isinstance(rows,list) or len(rows)!=count or
                    any(not isinstance(row,list) or len(row)!=width for row in rows)):
                    raise ValueError("Identification sample rows differ from declared coordinate dimensions")
    _keys(prediction, {"prior_source", "uncertainty_scope", "process_covariance", "process_covariance_evidence_refs",
                       "prior_process_crosscov_policy", "next_input"})
    if prediction["prior_source"] != "retained_gsie" or prediction["uncertainty_scope"] != SCOPE:
        raise ValueError("Declare the retained GSIE prior and conditional point-model uncertainty")
    if prediction["prior_process_crosscov_policy"] != "declared_zero":
        raise ValueError("Unknown prior-process cross-covariance blocks prediction")
    _refs(prediction["process_covariance_evidence_refs"])
    if (not isinstance(prediction["next_input"], list) or len(prediction["next_input"]) != len(ident["input_names"]) or
        any(type(x) not in (int, float) or x != 0 for x in prediction["next_input"])):
        raise ValueError("This next-state operation requires an explicit zero input")
    _keys(design, {"criterion", "budget", "cost_unit", "state_scales", "horizon", "condition_limit", "candidates"})
    if design["criterion"] not in {"a_opt", "d_opt"}:
        raise ValueError("Unsupported information criterion")
    if type(design["horizon"]) is not int or not 1 <= design["horizon"] <= 64:
        raise ValueError("The finite observability horizon must be an integer from one to 64")
    limit = design["condition_limit"]
    if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit < 1):
        raise ValueError("Observability condition limit must be at least one or explicit unknown")
    if type(design["budget"]) not in (int, float) or not math.isfinite(design["budget"]) or design["budget"] < 0:
        raise ValueError("Observation budget must be finite and nonnegative")
    _text(design["cost_unit"])
    if design["cost_unit"] == "inference_token":
        raise ValueError("Observation cost budget must be separate from inference tokens")
    if (not isinstance(design["state_scales"], list) or len(design["state_scales"]) != len(names) or
        any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in design["state_scales"])):
        raise ValueError("Ordered positive state scales are required")
    candidates = design["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise ValueError("Declare one to 32 finite observation candidates")
    ids = []
    for c in candidates:
        _keys(c, {"candidate_id", "observation_matrix", "noise_covariance", "measurement_units", "cost", "cost_unit",
                  "prior_cross_covariance_policy", "evidence_refs"})
        _text(c["candidate_id"])
        ids.append(c["candidate_id"])
        _refs(c["evidence_refs"])
        if c["prior_cross_covariance_policy"] != "declared_zero":
            raise ValueError("Unknown prior-candidate cross-covariance blocks information addition")
        if c["cost_unit"] != design["cost_unit"]:
            raise ValueError("Candidate observation cost unit differs from budget")
        if type(c["cost"]) not in (int, float) or not math.isfinite(c["cost"]) or c["cost"] < 0:
            raise ValueError("Candidate observation cost must be finite and nonnegative")
        for field,width in (("observation_matrix",len(names)), ("noise_covariance",len(c["observation_matrix"]))):
            matrix = c[field]
            if (not isinstance(matrix,list) or not 1 <= len(matrix) <= 32 or
                any(not isinstance(row,list) or len(row)!=width or any(
                    type(x) not in (int,float) or not math.isfinite(x) or float(x)!=x for x in row) for row in matrix)):
                raise ValueError("Candidate matrices must be bounded finite numeric rows with exact representability")
        if len(c["noise_covariance"]) != len(c["observation_matrix"]):
            raise ValueError("Candidate noise covariance must match observation rows")
        if not isinstance(c["measurement_units"], list) or len(c["measurement_units"]) != len(c["observation_matrix"]):
            raise ValueError("Candidate measurement unit order differs from observation rows")
        for unit in c["measurement_units"]:
            _text(unit)
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate identifiers must be unique")
    _keys(s["token_admission"], {"budget_unit", "department", "token_budget", "requested_tokens", "yield_claim", "eta_hat", "similarity_to_store"})
    if s["token_admission"]["budget_unit"] != "inference_token":
        raise ValueError("YWIR owns only the separate inference-token budget")
    if _declared_refs(s) & ({op for _,op in OPERATIONS} | {op for _,op in calibrated.OPERATIONS} | {ident["model_id"]}):
        raise ValueError("Declared evidence identities must differ from operations and model identity")
    canonical(s)
    return s


def _data(step):
    return step["result"]["data"]


def _numerical(role, data):
    return deepcopy(data["numerical_result"] if role == "sidt" else data)


def _request(index, source, upstream, steps, execution, evidence):
    role, operation = OPERATIONS[index]
    ident, prediction, design = source["identification"], source["prediction"], source["design"]
    prior_step = upstream["steps"][4]
    if role == "sidt":
        inputs = {"declaration": ident, "execution_ref": execution}
        refs = [evidence, prior_step["result_id"]]
    else:
        identified = _data(steps[0])["numerical_result"]
        if identified["status"] != "identified":
            raise AdapterRefusal("DESIGN_MODEL_" + identified["status"].upper(), "Identification did not produce an informative model")
        model = identified["candidate"]
        model_ref = steps[0]["numerical_result_id"]
        if role == "oit":
            inputs = {"model_result_id": model_ref, "transition": model["A"], "candidates": design["candidates"],
                "horizon": design["horizon"], "state_names": ident["state_names"], "state_scales": design["state_scales"],
                "condition_limit": design["condition_limit"]}
            refs = [evidence, steps[0]["result_id"]]
        elif role == "gsie":
            state = prior_step["result"]["data"]
            dt = ident["sample_interval"]
            time = state["time"] + dt
            if Fraction(time) != Fraction(state["time"]) + Fraction(dt):
                raise ValueError("Prediction time would lose declared sample interval precision")
            inputs = {"prior": {k:state[k] for k in ("time", "mean", "covariance", "frame_id", "units", "state_id")},
                "dynamics": {"matrix": model["A"], "process_covariance": prediction["process_covariance"], "model_id": model_ref},
                "target_time": time, "model_result_id": model_ref, "uncertainty_scope": SCOPE}
            refs = [evidence, prior_step["result_id"], steps[0]["result_id"], steps[1]["result_id"]]
        elif role == "edspt":
            eligible = _data(steps[1])["eligible_candidate_ids"]
            if not eligible:
                raise AdapterRefusal("DESIGN_NO_OBSERVABLE_CANDIDATE", "No candidate passed declared finite-horizon observability")
            inputs = {"model_result_id": model_ref, "prior_state_id": _data(steps[2])["state_id"],
                "prior_covariance": _data(steps[2])["covariance"], "coordinates": [
                    {"name":name,"unit":unit,"scale":scale} for name,unit,scale in
                    zip(ident["state_names"],ident["state_units"],design["state_scales"])],
                "candidates": [c for c in design["candidates"] if c["candidate_id"] in eligible],
                "budget":design["budget"], "cost_unit":design["cost_unit"], "criterion":design["criterion"]}
            refs = [evidence, *[s["result_id"] for s in steps]]
        else:
            selected = _data(steps[3])["selected_candidate_id"]
            if selected is None:
                raise AdapterRefusal("DESIGN_NO_AFFORDABLE_CANDIDATE", "No informative observation fits the declared observation budget")
            inputs = {"schema":"ywir.observation-design-token-request.v1", **source["token_admission"],
                      "selection_content_id":steps[3]["numerical_result_id"], "selected_candidate_id":selected}
            refs = [evidence, steps[3]["result_id"]]
    return {"schema":"ciw.adapter-request.v1", "operation_id":operation, "inputs":deepcopy(inputs)}, refs


def _seal_artifact(artifact, field):
    artifact[field] = "sha256:" + sha256(artifact["schema"].encode() + b"\0" + canonical(artifact)).hexdigest()
    return artifact


def _result(role, operation, execution, refs, data, source, created):
    ident = source["identification"]
    names, units = ident["state_names"], ident["state_units"]
    covariance_status, matrix = "not_applicable", None
    if role == "gsie":
        components = [{"name":n,"unit":u,"value":v} for n,u,v in zip(names,units,data["mean"])]
        covariance_status, matrix = "propagated", data["covariance"]
    elif role == "sidt":
        numerical = data["numerical_result"]
        rank = (numerical.get("diagnostics") or {}).get("rank", 0)
        components = [{"name":"regressor_rank","unit":"1","value":rank}]
        covariance_status = "unknown"
    elif role == "oit":
        components = [{"name":c["candidate_id"]+":observability_rank","unit":"1","value":c["rank"]} for c in data["candidates"]]
    elif role == "edspt":
        components = [{"name":"affordable_ranked_candidates","unit":"1","value":len(data["ranked_candidate_ids"])}]
    else:
        components = [{"name":"advisory_token_cap","unit":"inference_token","value":data["advisory_token_cap"]}]
    artifact = {"schema":RESULT_SCHEMA,"operation_id":operation,"execution_ref":execution,
        "created_at":created,"input_refs":refs,"model_refs":[ident["model_id"]],"calibration_refs":[],
        "components":components,"covariance":{"status":covariance_status,"matrix":matrix,
            "variables":[c["name"] for c in components],"units":[c["unit"] for c in components],
            "frame":{"id":ident["state_frame"] if role=="gsie" else "design:diagnostic-coordinates",
                     "semantics":"arbitrary_model_space"},"source_refs":refs,"calibration_refs":[],
            "method":SCOPE if role=="gsie" else "diagnostic_or_advisory_output"},
        "applicability":SCOPE+"; advisory only; no acquisition or state admission", "data":data,"verification_refs":[]}
    return _seal_artifact(artifact,"result_id")


def _execute(raw, upstream, upstream_replay, adapters, template=None):
    source = _source(raw, upstream)
    evidence = byte_digest(raw)
    created = template["created_at"] if template else utc_now()
    steps = []
    for index,(role,operation) in enumerate(OPERATIONS):
        execution = template["steps"][index]["execution_id"] if template else "execution-"+uuid.uuid4().hex
        request, refs = _request(index,source,upstream,steps,execution,evidence)
        data = _invoke(role,adapters,request)
        result = _result(role,operation,execution,refs,data,source,created)
        numerical = {"operation_id":operation,"data":_numerical(role,data)}
        steps.append({"runtime_ref":role,"operation_id":operation,"execution_id":execution,"input_refs":refs,
            "request":request,"request_sha256":digest(request),"result":result,"result_id":result["result_id"],
            "result_sha256":digest(result),"numerical_result":numerical,"numerical_result_id":digest(numerical)})
    bundle = {"schema":SCHEMA,"session_id":"session-"+uuid.uuid4().hex,"created_at":created,
        "source":{"experiment_id":source["experiment_id"],"configuration_digest":digest(source),
                  "evidence":[{"artifact_ref":evidence,"sha256":evidence,"bytes_b64":base64.b64encode(raw).decode("ascii")}]},
        "configuration":deepcopy(source),"upstream":deepcopy(upstream),"upstream_replay":deepcopy(upstream_replay),
        "runtimes":{role:a.runtime_identity() for role,a in adapters.items()},"steps":steps,
        "decision":{"selected_candidate_id":_data(steps[3])["selected_candidate_id"],
                    "token_admitted":_data(steps[4])["admitted"],"authority":"advisory_only",
                    "uncertainty_scope":SCOPE,"acquisition":"not_performed","state_admission":"not_performed"}}
    bundle["bundle_digest"] = bundle_digest(bundle)
    return bundle


def validate_upstream(bundle, upstream):
    """A design embeds its calibrated upstream; the selected retained bundle must be exactly it."""
    if canonical(bundle["upstream"]) != canonical(upstream):
        raise ValueError("Design upstream must exactly match a retained calibrated bundle")


def _validate_upstream_pair(original, replay):
    calibrated.inspect_session(original)
    fresh, receipt = replay["session"], replay["replay_receipt"]
    calibrated.inspect_session(fresh)
    if (receipt["source_bundle_digest"] != original["bundle_digest"] or
        receipt["replayed_bundle_digest"] != fresh["bundle_digest"] or receipt["numerical_match"] is not True or
        receipt["replay_id"] != digest({k:v for k,v in receipt.items() if k!="replay_id"}) or
        original["session_id"] == fresh["session_id"] or canonical(original["source"]) != canonical(fresh["source"])):
        raise ValueError("Upstream replay pair binding mismatch")
    for old,new in zip(original["steps"],fresh["steps"]):
        if (old["numerical_result_id"] != new["numerical_result_id"] or old["execution_id"] == new["execution_id"] or
            old["result_id"] == new["result_id"]):
            raise ValueError("Upstream replay requires stable numerics and fresh occurrences")
    for subject,verification in ((original,receipt["verification"]),(fresh,fresh["verification"])):
        _identity(verification,"verification_id")
        if verification["subject_ref"] != subject["bundle_digest"] or verification["outcome"] != "passed":
            raise ValueError("Upstream verification subject/outcome mismatch")


def _validate_verification(value,subject):
    _keys(value,{"schema","subject_ref","verifier_ref","created_at","outcome","independent","external_verifier_ref",
                 "checks","limitations","verification_id"})
    _identity(value,"verification_id")
    if (value["schema"]!="notation.instrument.verification-artifact.v1" or value["subject_ref"]!=subject or
        value["verifier_ref"]!="ciw:identified-design-pinned-replay.v1" or value["outcome"]!="passed" or
        value["independent"] is not False or value["external_verifier_ref"] is not None):
        raise ValueError("Design verification authority/subject mismatch")
    if [c["name"] for c in value["checks"]] != ["exact_pinned_recomputation","upstream_calibrated_replay"]:
        raise ValueError("Unexpected design verification checks")
    for check in value["checks"]:
        _keys(check,{"name","outcome","basis"})
        if check["outcome"]!="passed": raise ValueError("Design verification outcome contradicts checks")
        _text(check["basis"])
    _refs(value["limitations"])


def _validate(bundle):
    try:
        required = {"schema","session_id","created_at","source","configuration","upstream","upstream_replay",
                    "runtimes","steps","decision","bundle_digest"}
        if not required <= set(bundle) <= required | {"verification","replay_receipts"}:
            raise ValueError("Unexpected identified-design session fields")
        if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or bundle["bundle_digest"] != bundle_digest(bundle):
            raise ValueError("Identified-design session content binding mismatch")
        evidence, = bundle["source"]["evidence"]
        raw = base64.b64decode(evidence["bytes_b64"],validate=True)
        source = _source(raw,bundle["upstream"])
        if (evidence["artifact_ref"] != byte_digest(raw) or evidence["sha256"] != byte_digest(raw) or
            canonical(bundle["configuration"]) != canonical(source) or bundle["source"]["configuration_digest"] != digest(source) or
            bundle["source"]["experiment_id"] != source["experiment_id"]):
            raise ValueError("Design evidence/configuration binding mismatch")
        _validate_upstream_pair(bundle["upstream"],bundle["upstream_replay"])
        steps = bundle["steps"]
        if [(s["runtime_ref"],s["operation_id"]) for s in steps] != list(OPERATIONS) or set(bundle["runtimes"]) != ROLES:
            raise ValueError("Design operation order/runtime mismatch")
        for role,pin in _pins().items():
            runtime = bundle["runtimes"][role]
            if runtime["schema"] != "ciw.subprocess-runtime.v1" or any(runtime[k] != pin[k] for k in ("revision","module","source_root")):
                raise ValueError("Design runtime pin mismatch")
            if not re.fullmatch(r"[0-9a-f]{40}",runtime["source_tree"]) or not re.fullmatch(r"[0-9a-f]{64}",runtime["python_sha256"]):
                raise ValueError("Design runtime identity incomplete")
        if not re.fullmatch(r"session-[0-9a-f]{32}", bundle["session_id"]):
            raise ValueError("Invalid design session identity")
        identity_list = [bundle["session_id"],evidence["artifact_ref"],*[op for _,op in OPERATIONS]]
        if len(set(identity_list)) != len(identity_list):
            raise ValueError("Evidence, session and operation identities must differ")
        identities = set(identity_list)
        declared_refs = _declared_refs(source)
        if bundle["session_id"] in declared_refs:
            raise ValueError("Session identity must differ from declared evidence")
        identities.update(declared_refs)
        upstream_ids = set()
        for previous in (bundle["upstream"],bundle["upstream_replay"]["session"]):
            upstream_ids.add(previous["session_id"])
            if "verification" in previous:
                upstream_ids.add(previous["verification"]["verification_id"])
            for old in previous["steps"]:
                upstream_ids.update((old["execution_id"],old["result_id"]))
        if bundle["session_id"] in upstream_ids:
            raise ValueError("Design session identity must differ from retained upstream occurrences")
        identities.update(upstream_ids)
        for index,step in enumerate(steps):
            role,operation = OPERATIONS[index]
            request, refs = _request(index,source,bundle["upstream"],steps[:index],step["execution_id"],evidence["artifact_ref"])
            if canonical(step["request"]) != canonical(request) or canonical(step["input_refs"]) != canonical(refs):
                raise ValueError("Design request differs from retained operation graph")
            result = step["result"]
            expected = _result(role,operation,step["execution_id"],refs,result["data"],source,bundle["created_at"])
            numerical = {"operation_id":operation,"data":_numerical(role,result["data"])}
            if canonical(result) != canonical(expected) or step["result_id"] != result["result_id"] or canonical(step["numerical_result"]) != canonical(numerical):
                raise ValueError("Design result/numerical binding mismatch")
            for key,value in (("request_sha256",request),("result_sha256",result),("numerical_result_id",numerical)):
                if step[key] != digest(value): raise ValueError("Design content identity mismatch")
            for occurrence in (step["execution_id"],step["result_id"]):
                if occurrence in identities: raise ValueError("Evidence, execution and result identities must differ")
                identities.add(occurrence)
            if not re.fullmatch(r"execution-[0-9a-f]{32}",step["execution_id"]):
                raise ValueError("Invalid design execution identity")
        decision = {"selected_candidate_id":_data(steps[3])["selected_candidate_id"],"token_admitted":_data(steps[4])["admitted"],
            "authority":"advisory_only","uncertainty_scope":SCOPE,"acquisition":"not_performed","state_admission":"not_performed"}
        if canonical(bundle["decision"]) != canonical(decision):
            raise ValueError("Decision differs from retained selection/token advice")
        from .identified_semantics import validate_outputs
        validate_outputs(bundle)
        if "verification" in bundle:
            _validate_verification(bundle["verification"],bundle["bundle_digest"])
            if bundle["verification"]["verification_id"] in identities:
                raise ValueError("Design verification identity must differ from other roles")
        if "replay_receipts" in bundle:
            receipt, = bundle["replay_receipts"]
            _keys(receipt,{"schema","source_bundle_digest","replayed_bundle_digest","numerical_match","admission","verification","replay_id"})
            if (receipt["schema"]!="ciw.identified-design-replay.v1" or receipt["numerical_match"] is not True or
                receipt["admission"]!="not_performed" or receipt["replayed_bundle_digest"]!=bundle["bundle_digest"] or
                receipt["source_bundle_digest"]==bundle["bundle_digest"] or
                receipt["replay_id"]!=digest({k:v for k,v in receipt.items() if k!="replay_id"})):
                raise ValueError("Design replay receipt binding mismatch")
            _validate_verification(receipt["verification"],receipt["source_bundle_digest"])
        return raw
    except (KeyError,TypeError,IndexError,OverflowError,RecursionError) as exc:
        raise ValueError("Malformed identified-design session") from exc


def inspect_session(bundle):
    _validate(bundle)
    return {"schema":"ciw.identified-design-inspection.v1","session_id":bundle["session_id"],
        "status":"content_consistent","numerical_replay":"not_performed","decision":deepcopy(bundle["decision"]),
        "operation_ids":[op for _,op in OPERATIONS]}


def _compare(original,reproduced):
    if any(any(a[k] != b[k] for k in ("request_sha256","result_sha256","numerical_result_id"))
           for a,b in zip(original["steps"],reproduced["steps"])):
        raise ValueError("Retained design result differs from exact pinned recomputation")


def _verify(bundle,adapters):
    artifact = {"schema":"notation.instrument.verification-artifact.v1","subject_ref":bundle["bundle_digest"],
        "verifier_ref":"ciw:identified-design-pinned-replay.v1","created_at":utc_now(),"outcome":"passed",
        "independent":False,"external_verifier_ref":None,
        "checks":[{"name":"exact_pinned_recomputation","outcome":"passed","basis":"all five retained operation outputs matched fresh pinned execution"},
                  {"name":"upstream_calibrated_replay","outcome":"passed","basis":"upstream original replayed before use"}],
        "limitations":["conditional identified point model; parameter covariance unknown", "same implementation replay", "advisory; no acquisition, reservation, or state admission"]}
    _seal_artifact(artifact,"verification_id")
    _invoke("set",adapters,{"inputs":{"results":[s["result"] for s in bundle["steps"]],"verification":artifact}})
    return artifact


def _replay_upstream(upstream,repositories):
    replay = calibrated.replay_session(upstream,{r:repositories[r] for r in calibrated.ROLES})
    return {"session":replay["session"],"replay_receipt":replay["replay_receipt"]}


def create_session(source_bytes,upstream_bundle,repositories):
    _source(source_bytes,upstream_bundle)
    adapters = _adapters(repositories)
    checked = _replay_upstream(upstream_bundle,repositories)
    bundle = _execute(source_bytes,upstream_bundle,checked,adapters)
    _validate(bundle)
    reproduced = _execute(source_bytes,upstream_bundle,checked,adapters,bundle)
    _compare(bundle,reproduced)
    bundle["verification"] = _verify(bundle,adapters)
    return bundle


def replay_session(bundle,repositories):
    raw = _validate(bundle)
    adapters = _adapters(repositories,bundle["runtimes"])
    checked = _replay_upstream(bundle["upstream"],repositories)
    reproduced = _execute(raw,bundle["upstream"],checked,adapters,bundle)
    _compare(bundle,reproduced)
    fresh = _execute(raw,bundle["upstream"],checked,adapters)
    if any(a["numerical_result_id"] != b["numerical_result_id"] for a,b in zip(bundle["steps"],fresh["steps"])):
        raise ValueError("Fresh design execution changed the numerical result")
    fresh["verification"] = _verify(fresh,adapters)
    receipt = {"schema":"ciw.identified-design-replay.v1","source_bundle_digest":bundle["bundle_digest"],
        "replayed_bundle_digest":fresh["bundle_digest"],"numerical_match":True,"admission":"not_performed",
        "verification":_verify(bundle,adapters)}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    return {"session":fresh,"replay_receipt":receipt}


def read_session(path):
    return _json(_read(Path(path),MAX_BYTES))


def save_session(bundle,output_dir):
    _validate(bundle)
    path = Path(output_dir)/"identified-design-session.json"
    if path.exists(): raise ValueError("Refusing to overwrite an existing design session")
    return write_json(path,bundle)
