"""CSG, GSIE and PLSR composed around a retained Gaussian VI experiment."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from . import free_energy_math as mathematics
from . import free_energy_native as native
from .free_energy_profile import KIND, validate_source, csg_request, problems
from .core.canonical import canonical, exact_keys
from .pipelines.runner import (PipelineRunner, StageChain, chain_catalog, chain_claims, chain_occurrences, check_chain,
                               check_step, seal_step)

DATA_SCHEMA = "ciw.variational-free-energy-result.v1"
OPERATION = "ciw.variational-free-energy.v1"
NATIVE_OPERATIONS = {"csg":"ciw.free-energy-jacobi.v1", "gsie":"ciw.free-energy-gaussian-update.v1", "plsr":"ciw.free-energy-iteration-assessment.v1"}
CODE_PROFILE = "ciw.gaussian-variational-inference.v1"


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Free-energy record binding differs")


def algorithm_identity():
    files = ["free_energy_math.py", "free_energy_native.py", "free_energy_profile.py", "free_energy_workflow.py", "plsr_engine.py",
             "pipelines/runner.py"]
    content = b"\0".join(name.encode() + b"\0" + Path(__file__).parent.joinpath(name).read_text(encoding="utf-8").encode("utf-8") for name in files)
    return {"profile": CODE_PROFILE, "code_sha256": sha256(content).hexdigest(), "source_normalization":"utf8_lf", "numpy_version":np.__version__}


def numerical_projection(data):
    return {"operation_id":OPERATION, "data":{**{k:deepcopy(v) for k,v in data.items() if k != "stages"},
        "native_stages":[{"operation_id":step["operation_id"], "request":deepcopy(step["request"]), "data":deepcopy(step["result"]["data"])} for step in data["stages"]]}}


def _compute(source, evidence, adapters):
    chain = StageChain(evidence)
    def invoke(role, request):
        data = native.invoke(role, adapters, request)
        chain.seal(role, NATIVE_OPERATIONS[role], request, data)
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
            "kl_gap":final["kl_to_reference"]}, "stages":chain.stages}


# Module names kept for callers; the chained-stage versions live in the runner.
catalog_steps, native_occurrences, identity_claims = chain_catalog, chain_occurrences, chain_claims


class FreeEnergyWorkflow(PipelineRunner):
    """CSG with GSIE and PLSR companions: one aggregate step over three chained native stages."""

    LABEL = "Free-energy"
    FRESH_OCCURRENCE_MESSAGE = "Free-energy experiments require fresh native stage occurrences"

    def catalog_steps(self, bundle):
        return chain_catalog(bundle)

    def identity_claims(self, bundle):
        return chain_claims(bundle)

    def native_occurrences(self, bundle):
        return chain_occurrences(bundle)

    def __init__(self):
        super().__init__(KIND, {**native.PINS["csg"], "role": "csg"})
        self.ROLES = frozenset(native.ROLES)

    def parse_source(self, raw):
        return validate_source(raw)

    @staticmethod
    def _runtime_projection(runtime):
        primary = PipelineRunner._runtime_projection(runtime)
        primary["companions"] = {r:PipelineRunner._runtime_projection(v) for r,v in runtime["companions"].items()}
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
        step = seal_step(self.role, OPERATION, source, [evidence_id], data, numerical_projection(data))
        self._validate_step(step, source, evidence_id)
        return step

    def _validate_step(self, step, source, evidence_id):
        from .free_energy_contract import validate_data
        data = step["result"]["data"]
        check_step(step, role=self.role, operation=OPERATION, source=source, input_refs=[evidence_id], check_data=None,
                   label=self.LABEL, numerical=numerical_projection(data))
        check_chain(data["stages"], NATIVE_OPERATIONS, evidence_id, seen={step["execution_id"]}, label="Native")
        validate_data(source, data, evidence_id)

    def _check_verification(self, bundle, verification, source, evidence):
        super()._check_verification(bundle, verification, source, evidence)
        original, reproduced = bundle["steps"][0], verification["reproduction"]
        old = {original["execution_id"]} | {s["execution_id"] for s in original["result"]["data"]["stages"]}
        new = {reproduced["execution_id"]} | {s["execution_id"] for s in reproduced["result"]["data"]["stages"]}
        if old & new:
            raise ValueError("Native reproduction must create fresh aggregate and stage occurrences")

    def _check_runtimes(self, runtimes):
        exact_keys(runtimes,{"csg"})
        primary = runtimes["csg"]
        native.check_runtime("csg",{k:v for k,v in primary.items() if k not in {"companions","workbench_algorithm"}})
        exact_keys(primary["companions"],{"gsie","plsr"})
        for role,runtime in primary["companions"].items():
            native.check_runtime(role,runtime)
        algorithm = primary["workbench_algorithm"]
        exact_keys(algorithm,{"profile","code_sha256","source_normalization","numpy_version"})
        if algorithm["profile"] != CODE_PROFILE or not re.fullmatch(r"[a-f0-9]{64}",algorithm["code_sha256"]) or algorithm["source_normalization"] != "utf8_lf" or algorithm["numpy_version"] != "2.4.3":
            raise ValueError("Unsupported workbench numerical algorithm identity")
