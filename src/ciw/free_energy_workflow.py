"""CSG, GSIE and PLSR composed around a retained Gaussian VI experiment."""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
import uuid

import numpy as np

from . import free_energy_math as mathematics
from . import free_energy_native as native
from .free_energy_profile import KIND, SOURCE_SCHEMA, POLICY, validate_source, csg_request, problems
from .declared_workload import AUTHORITY, DeclaredWorkflow, RESULT_SCHEMA, MAX_BYTES
from .exchange import _identity
from .telemetry import canonical, digest, byte_digest, _bundle_digest, _keys

DATA_SCHEMA = "ciw.variational-free-energy-result.v1"
OPERATION = "ciw.variational-free-energy.v1"
NATIVE_OPERATIONS = {"csg":"ciw.free-energy-jacobi.v1", "gsie":"ciw.free-energy-gaussian-update.v1", "plsr":"ciw.free-energy-iteration-assessment.v1"}
CODE_PROFILE = "ciw.gaussian-variational-inference.v1"


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Free-energy record binding differs")


def algorithm_identity():
    files = ["free_energy_math.py", "free_energy_native.py", "free_energy_profile.py", "free_energy_workflow.py", "plsr_engine.py"]
    content = b"\0".join(name.encode() + b"\0" + Path(__file__).with_name(name).read_text(encoding="utf-8").encode("utf-8") for name in files)
    return {"profile": CODE_PROFILE, "code_sha256": sha256(content).hexdigest(), "source_normalization":"utf8_lf", "numpy_version":np.__version__}


def make_step(role, operation, request, data, inputs):
    occurrence = "execution-" + uuid.uuid4().hex
    result = {"schema":RESULT_SCHEMA, "operation_id":operation, "execution_ref":occurrence,
              "input_refs":deepcopy(inputs), "data":deepcopy(data), "authority":deepcopy(AUTHORITY)}
    result["result_id"] = digest(result)
    numerical = {"operation_id":operation, "data":deepcopy(data)}
    return {"runtime_ref":role, "operation_id":operation, "execution_id":occurrence, "input_refs":deepcopy(inputs),
            "request":deepcopy(request), "request_sha256":digest(request), "result":result,
            "result_sha256":digest(result), "result_id":result["result_id"],
            "numerical_result":numerical, "numerical_result_id":digest(numerical)}


def numerical_projection(data):
    return {"operation_id":OPERATION, "data":{**{k:deepcopy(v) for k,v in data.items() if k != "stages"},
        "native_stages":[{"operation_id":step["operation_id"], "request":deepcopy(step["request"]), "data":deepcopy(step["result"]["data"])} for step in data["stages"]]}}


def _compute(source, evidence, adapters):
    stages = []
    def invoke(role, request):
        data = native.invoke(role, adapters, request)
        inputs = [evidence] + [step["result_id"] for step in stages]
        stages.append(make_step(role, NATIVE_OPERATIONS[role], request, data, inputs))
        return data
    csg = invoke("csg", csg_request(source))
    problem, held, normalization = problems(source, csg["Phi"])
    # Validate the normalized numerical domain before native fusion/stability.
    reference = mathematics.gaussian_reference(problem)
    mathematics.predict_held_out(held, reference["mean"], reference["covariance"])
    gsie_request = {"G":problem["observation_matrix"], "y":problem["observations"], "Sigma":problem["noise_covariance"],
        "prior":{"mean":problem["prior_mean"], "covariance":problem["prior_covariance"]}, "evidence_id":evidence}
    gsie = invoke("gsie", gsie_request)
    fit = mathematics.variational_fit(problem, **source["solver"])
    precision = fit["information_system"]["precision"]
    error = (np.asarray(source["solver"]["initial_mean"])-reference["mean"]).tolist()
    invoke("plsr", {"Lambda":precision, "alpha":source["solver"]["alpha"], "error":error,
                    "evidence_id":evidence, "required_margin":0.0})
    mean_error = float(np.max(np.abs(np.asarray(gsie["estimate"]["mean"])-reference["mean"])))
    covariance_error = float(np.max(np.abs(np.asarray(gsie["estimate"]["covariance"])-reference["covariance"])))
    agreement = {"mean_max_abs_error":mean_error, "covariance_max_abs_error":covariance_error, "tolerance":1e-9,
                 "passed":max(mean_error,covariance_error) <= 1e-9}
    if not agreement["passed"]:
        raise ValueError("Native Gaussian fusion disagrees with the independent reference")
    d = np.asarray(normalization["latent_scales"])
    t = np.asarray(normalization["observation_scales"])
    model, samples = source["assumed_model"], source["samples"]
    metrics = mathematics.evaluate_ensemble(problem, [np.asarray(s["truth"])/d for s in samples],
        [(np.asarray(s["training"])-model["training_bias"])/t for s in samples], held,
        [(np.asarray(s["heldout"])-model["heldout_bias"])/t for s in samples])
    final = fit["trace"][-1]
    jacobian = normalization["log_observation_jacobian"]
    return {"schema":DATA_SCHEMA, "claim_scope":"synthetic_static_linear_gaussian_inference",
        "normalization":normalization, "problem":problem, "held_out_problem":held, "fit":fit,
        "gsie_agreement":agreement,
        "physical_posterior":{"mean":(d*np.asarray(fit["mean"])).tolist(),
            "covariance":(np.asarray(fit["covariance"])*np.outer(d,d)).tolist(), "units":source["coordinates"]["units"],
            "frame":source["coordinates"]["frame"], "uncertainty_scope":"conditional_on_declared_model"},
        "held_out":mathematics.predict_held_out(held, fit["mean"], fit["covariance"]),
        "ensemble":{"basis":"exact_reference_posterior_on_retained_ensemble",
            "generator_relationship":"operator_declared_iid_prior_predictive_not_authenticated", "metrics":metrics},
        "objective_units":{"free_energy_normalized":final["free_energy"], "free_energy_physical":final["free_energy"]+jacobian,
            "log_evidence_normalized":reference["log_evidence"], "log_evidence_physical":reference["log_evidence"]-jacobian,
            "kl_gap":final["kl_to_reference"]}, "stages":stages}


def _validate_step_envelope(step, role, operation, request, inputs, *, numerical=None):
    _keys(step, {"runtime_ref", "operation_id", "execution_id", "input_refs", "request", "request_sha256",
                "result", "result_sha256", "result_id", "numerical_result", "numerical_result_id"})
    if step["runtime_ref"] != role or step["operation_id"] != operation or not re.fullmatch(r"execution-[a-f0-9]{32}", step["execution_id"]):
        raise ValueError("Free-energy stage occurrence or operation differs")
    _same(step["request"], request)
    _same(step["input_refs"], inputs)
    result = step["result"]
    _keys(result, {"schema", "operation_id", "execution_ref", "input_refs", "data", "authority", "result_id"})
    _same(result, {"schema":RESULT_SCHEMA, "operation_id":operation, "execution_ref":step["execution_id"],
        "input_refs":inputs, "data":result["data"], "authority":AUTHORITY,
        "result_id":digest({k:v for k,v in result.items() if k != "result_id"})})
    _same(step["numerical_result"], numerical or {"operation_id":operation, "data":result["data"]})
    if step["result_id"] != result["result_id"]:
        raise ValueError("Stage result identity differs")
    for key,value in (("request_sha256",request),("result_sha256",result),("numerical_result_id",step["numerical_result"])):
        if step[key] != digest(value):
            raise ValueError("Stage commitment differs")


def catalog_steps(bundle):
    return deepcopy(bundle["steps"][0]["result"]["data"]["stages"])


def native_occurrences(bundle):
    return {s["execution_id"] for outer in (bundle["steps"][0],bundle["verification"]["reproduction"])
            for s in outer["result"]["data"]["stages"]}


def identity_claims(bundle):
    claims = {}
    for outer in (bundle["steps"][0],bundle["verification"]["reproduction"]):
        for s in outer["result"]["data"]["stages"]:
            for identity, role, body in ((s["execution_id"],"execution",{"step":s}),
                    (s["result_id"],"result",{"result":s["result"]}),
                    (s["numerical_result_id"],"numerical_result",s["numerical_result"]),
                    (s["operation_id"],"operation",s["operation_id"])):
                if identity in claims:
                    _same(claims[identity], (role,body))
                claims[identity] = (role,deepcopy(body))
    return claims


class FreeEnergyWorkflow(DeclaredWorkflow):
    def __init__(self):
        self.kind, self.role, self.ROLES = KIND, "csg", native.ROLES
        self.SOURCE_SCHEMA, self.schema, self.operation = SOURCE_SCHEMA, "ciw.variational-free-energy-session.v1", OPERATION
        self.pin = native.PINS["csg"]

    def _source(self, raw):
        return validate_source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        primary = DeclaredWorkflow._runtime_projection(runtime)
        primary["companions"] = {r:DeclaredWorkflow._runtime_projection(v) for r,v in runtime["companions"].items()}
        return primary

    def _adapters(self, repositories, expected=None):
        old = expected["csg"] if expected else None
        old_native = None if old is None else {"csg":{k:v for k,v in old.items() if k not in {"companions","workbench_algorithm"}}, **old["companions"]}
        adapters, runtimes = native.bind(repositories, old_native)
        runtime = {**runtimes["csg"], "companions":{r:runtimes[r] for r in ("gsie","plsr")}, "workbench_algorithm":algorithm_identity()}
        if old:
            _same(self._runtime_projection(runtime), self._runtime_projection(old))
        return adapters, runtime, None

    def _step(self, source, evidence_id, bound):
        adapters, runtime, _ = bound
        before = {r:native.runtime_identity(r,a) for r,a in adapters.items()}
        expected = {"csg":{k:v for k,v in runtime.items() if k not in {"companions","workbench_algorithm"}}, **runtime["companions"]}
        _same(before, expected)
        _same(algorithm_identity(), runtime["workbench_algorithm"])
        data = _compute(source, evidence_id, adapters)
        _same(before, {r:native.runtime_identity(r,a) for r,a in adapters.items()})
        _same(algorithm_identity(), runtime["workbench_algorithm"])
        step = make_step(self.role, OPERATION, source, data, [evidence_id])
        step["numerical_result"] = numerical_projection(data)
        step["numerical_result_id"] = digest(step["numerical_result"])
        self._validate_step(step, source, evidence_id)
        return step

    def _validate_step(self, step, source, evidence_id):
        from .free_energy_contract import validate_data
        data = step["result"]["data"]
        _validate_step_envelope(step, self.role, OPERATION, source, [evidence_id], numerical=numerical_projection(data))
        if type(data["stages"]) is not list or len(data["stages"]) != 3:
            raise ValueError("Require all three native stages")
        seen, inputs = {step["execution_id"]}, [evidence_id]
        for role, s in zip(("csg","gsie","plsr"), data["stages"]):
            _validate_step_envelope(s, role, NATIVE_OPERATIONS[role], s["request"], inputs)
            if s["execution_id"] in seen:
                raise ValueError("Native stages require distinct execution occurrences")
            seen.add(s["execution_id"])
            inputs = inputs + [s["result_id"]]
        validate_data(source, data, evidence_id)

    def _check_verification(self, bundle, verification, source, evidence):
        super()._check_verification(bundle, verification, source, evidence)
        original, reproduced = bundle["steps"][0], verification["reproduction"]
        old = {original["execution_id"]} | {s["execution_id"] for s in original["result"]["data"]["stages"]}
        new = {reproduced["execution_id"]} | {s["execution_id"] for s in reproduced["result"]["data"]["stages"]}
        if old & new:
            raise ValueError("Native reproduction must create fresh aggregate and stage occurrences")

    def _validate(self, bundle):
        try:
            _keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps", "bundle_digest", "verification"}, {"replay_receipts"})
            if len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != self.schema or bundle["bundle_digest"] != _bundle_digest(bundle) or not re.fullmatch(r"session-[a-f0-9]{32}",bundle["session_id"]):
                raise ValueError("Free-energy bundle identity or size differs")
            if type(bundle["created_at"]) is not str or not bundle["created_at"]:
                raise ValueError("Missing execution time metadata")
            evidence, = bundle["source"]["evidence"]
            raw = base64.b64decode(evidence["bytes_b64"],validate=True)
            source = self._source(raw)
            _same(evidence,{"artifact_ref":byte_digest(raw),"sha256":byte_digest(raw),"bytes_b64":base64.b64encode(raw).decode()})
            _same(bundle["source"],{"experiment_id":source["experiment_id"],"experiment_digest":digest(source),"evidence":[evidence]})
            _same(bundle["configuration"],POLICY)
            _keys(bundle["runtimes"],{"csg"})
            primary = bundle["runtimes"]["csg"]
            native.check_runtime("csg",{k:v for k,v in primary.items() if k not in {"companions","workbench_algorithm"}})
            _keys(primary["companions"],{"gsie","plsr"})
            for role,runtime in primary["companions"].items():
                native.check_runtime(role,runtime)
            algorithm = primary["workbench_algorithm"]
            _keys(algorithm,{"profile","code_sha256","source_normalization","numpy_version"})
            if algorithm["profile"] != CODE_PROFILE or not re.fullmatch(r"[a-f0-9]{64}",algorithm["code_sha256"]) or algorithm["source_normalization"] != "utf8_lf" or algorithm["numpy_version"] != "2.4.3":
                raise ValueError("Unsupported workbench numerical algorithm identity")
            step, = bundle["steps"]
            self._validate_step(step, source, evidence["artifact_ref"])
            self._check_verification(bundle,bundle["verification"],source,evidence["artifact_ref"])
            receipts = bundle.get("replay_receipts",[])
            if type(receipts) is not list or len(receipts) > 1:
                raise ValueError("At most one replay receipt belongs to an occurrence")
            for receipt in receipts:
                _keys(receipt,{"schema","source_bundle_digest","replayed_bundle_digest","numerical_match","verification","admission","replay_id"})
                if (receipt["schema"] != "ciw."+KIND+"-replay.v1" or receipt["replayed_bundle_digest"] != bundle["bundle_digest"] or
                        receipt["source_bundle_digest"] == bundle["bundle_digest"] or not re.fullmatch(r"sha256:[a-f0-9]{64}",receipt["source_bundle_digest"]) or
                        receipt["numerical_match"] is not True or receipt["admission"] != "not_performed" or
                        receipt["replay_id"] != digest({k:v for k,v in receipt.items() if k != "replay_id"})):
                    raise ValueError("Invalid free-energy replay receipt")
                verification = receipt["verification"]
                _keys(verification,{"schema","subject_ref","outcome","independent","method","runtime_digest","reproduction","authority","verification_id"})
                _same(verification["reproduction"],step)
                _same(verification["authority"],AUTHORITY)
                if (verification["schema"] != "ciw.declared-workload-verification.v1" or verification["subject_ref"] != receipt["source_bundle_digest"] or
                        verification["outcome"] != "passed" or verification["independent"] is not False or
                        verification["method"] != "same_runtime_fresh_occurrence_reproduction" or not re.fullmatch(r"sha256:[a-f0-9]{64}",verification["runtime_digest"])):
                    raise ValueError("Invalid free-energy replay verification scope")
                _identity(verification,"verification_id")
            return raw
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("Malformed retained free-energy session") from exc
