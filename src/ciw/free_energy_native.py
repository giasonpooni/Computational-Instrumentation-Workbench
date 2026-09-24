"""Fixed, source-pinned native stages for the declared free-energy experiment.

Providers are imported only in fresh subprocesses. This bridge authors the
fixed quadratic error model; PLSR evaluates it without synthesizing a model.
Retained native results are deterministic content, not execution occurrences.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import PinnedSubprocessAdapter, _json
from .core.canonical import canonical, digest

PINS = {
    "csg": {"revision":"bbc535af29c30997e56fd120320c570830676462",
            "source_tree":"181b6eb73288d001f45c39bb149b1a80a431f34b",
            "module":"geodesic_testbed.jacobi", "source_root":"src"},
    "gsie": {"revision":"5241eee6dab434533bdf0cf0e824bc43b4a79831",
             "source_tree":"375c031c07592d5bcb1d224780f18ceb886df4a1",
             "module":"geometric_state_inference.contracts", "source_root":"src"},
    "plsr": {"revision":"19ea6967060166ba09db6cd4563bd87bd6b3d196",
             "source_tree":"e261315f46851d99053e305ef2c03d4160f02a92",
             "module":"lyapunov.model_artifact", "source_root":"src"},
}
ROLES = frozenset(PINS)
MAX_REQUEST_BYTES = 256 * 1024
FRAME = "ciw.free-energy.normalized-initial-transverse-state.v1"
MEASUREMENT_FRAME = "ciw.free-energy.normalized-observation-vector.v1"
CSG_COVARIANCE_NOTE = "Synthetic identity covariance for the transfer reference; not the fused prior."


def _keys(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Unexpected native-stage request fields")


def _number(value, limit=1e12):
    if type(value) not in (int, float):
        raise ValueError("Require finite binary64 numbers, not coerced values")
    result = float(value)
    if not math.isfinite(result) or Fraction(value) != Fraction(result) or abs(result) > limit:
        raise ValueError("Native-stage number exceeds its binary64 budget")
    return result


def _vector(value, count, limit=1e12):
    if type(value) is not list or len(value) != count:
        raise ValueError("Native-stage vector shape differs")
    return [_number(item, limit) for item in value]


def _matrix(value, rows, columns, limit=1e12):
    if type(value) is not list or len(value) != rows:
        raise ValueError("Native-stage matrix shape differs")
    return np.asarray([_vector(row, columns, limit) for row in value])


def _spd(value, size, limit=1e12):
    matrix = _matrix(value, size, size, limit)
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Native-stage covariance/precision must be exactly symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Native-stage covariance/precision must be positive definite")


def validate_request(role, request):
    """Check transport bounds; the enclosing workflow owns scientific policy."""
    try:
        if role not in ROLES or len(canonical(request)) > MAX_REQUEST_BYTES:
            raise ValueError("Unsupported or oversized native stage")
        if role == "csg":
            _keys(request, {"arclength", "gaussian_curvature", "units", "relative_tolerance"})
            grid = request["arclength"]
            if type(grid) is not list or not 2 <= len(grid) <= 128:
                raise ValueError("CSG stage requires 2..128 arclength samples")
            _vector(grid,len(grid),8)
            if grid[0] != 0 or any(a >= b for a,b in zip(grid,grid[1:])):
                raise ValueError("CSG grid must begin at zero and increase")
            _number(request["gaussian_curvature"],1)
            _keys(request["units"], {"length", "angle"})
            if request["units"]["length"] not in {"m", "mm", "normalized_length"} or request["units"]["angle"] != "radian":
                raise ValueError("CSG stage requires declared length units and radians")
            if not 0 < _number(request["relative_tolerance"]) <= .01:
                raise ValueError("Invalid CSG heading-linearization tolerance")
        elif role == "gsie":
            _keys(request, {"G", "y", "Sigma", "prior", "evidence_id"})
            count = len(request["y"])
            if not 1 <= count <= 32:
                raise ValueError("GSIE stage requires 1..32 jointly modeled observations")
            _vector(request["y"],count)
            _matrix(request["G"],count,2)
            _spd(request["Sigma"],count)
            _keys(request["prior"], {"mean", "covariance"})
            _vector(request["prior"]["mean"],2)
            _spd(request["prior"]["covariance"],2)
        else:
            _keys(request, {"Lambda", "alpha", "error", "evidence_id", "required_margin"})
            # Derived information precision and reference error can exceed the
            # original observations. These caps still keep A.T P A and the
            # sampled quadratic below 1e100, far inside finite binary64 range.
            _spd(request["Lambda"],2,1e20)
            _vector(request["error"],2,1e12)
            if not 0 < _number(request["alpha"],1e6):
                raise ValueError("PLSR stage requires a positive iteration step")
            if _number(request["required_margin"]) < 0:
                raise ValueError("PLSR required margin cannot be negative")
        if role != "csg" and (type(request["evidence_id"]) is not str or not re.fullmatch(r"sha256:[a-f0-9]{64}",request["evidence_id"])):
            raise ValueError("Native stage requires retained evidence identity")
        return deepcopy(request)
    except (TypeError, KeyError, IndexError, OverflowError, RecursionError, UnicodeError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed bounded native-stage request") from exc


def csg_source(request):
    """Reference wrapper usable by the existing CSG retained-record validator."""
    request = validate_request("csg",request)
    from .geodesic_reference import POLICIES
    return {"schema":"ciw.curved-path-transfer-source.v1", "experiment_id":"free-energy-transfer-reference",
        "configuration":deepcopy(POLICIES["curved-path-transfer"]), **request,
        "initial_perturbation":[0.0,0.0], "starting_covariance":{"matrix":[[1.0,0.0],[0.0,1.0]],
            "basis":"assumed", "frame":"transverse-to-gamma, parallel-transported",
            "coverage_factor":None, "note":CSG_COVARIANCE_NOTE}}


def _projection(runtime):
    return {k:v for k,v in runtime.items() if k not in {"repository_root", "python_executable"}}


def check_runtime(role, runtime):
    """Validate the pins and declared execution environment without execution."""
    if role not in ROLES:
        raise ValueError("Unsupported native provider")
    fields = {"schema", "adapter_version", "repository_root", "revision", "source_tree", "module", "source_root",
              "python_executable", "python_sha256", "python_version", "dependencies"}
    _keys(runtime, fields | ({"model_schema_dependency"} if role == "plsr" else set()))
    if (runtime["schema"] != "ciw.subprocess-runtime.v1" or runtime["adapter_version"] != "ciw-pinned-subprocess-v1"
            or any(runtime[key] != value for key,value in PINS[role].items())):
        raise ValueError("Native provider differs from its exact source pin")
    for field in ("repository_root", "python_executable"):
        if type(runtime[field]) is not str or not runtime[field].strip():
            raise ValueError("Native runtime location metadata is missing")
    if type(runtime["python_sha256"]) is not str or not re.fullmatch(r"[a-f0-9]{64}",runtime["python_sha256"]):
        raise ValueError("Invalid native Python binary identity")
    version = runtime["python_version"]
    if type(version) is not str or not re.fullmatch(r"\d+\.\d+\.\d+",version):
        raise ValueError("Invalid native Python version")
    if tuple(map(int,version.split(".")[:2])) < ((3,12) if role == "plsr" else (3,11)):
        raise ValueError("Native provider Python version is unsupported")
    _keys(runtime["dependencies"], {"numpy", "scipy"})
    if runtime["dependencies"]["numpy"] != "2.4.3":
        raise ValueError("Free-energy providers require NumPy 2.4.3")
    scipy = runtime["dependencies"]["scipy"]
    if scipy is not None and (type(scipy) is not str or not scipy):
        raise ValueError("Invalid retained SciPy version")
    if role == "plsr" and runtime["model_schema_dependency"] != {"name":"jsonschema", "version":"4.26.0"}:
        raise ValueError("PLSR model schema loader requires jsonschema 4.26.0")


def runtime_identity(role, adapter):
    identity = adapter.runtime_identity()
    if role == "plsr":
        code, raw = adapter._run("import importlib.metadata,json;print(json.dumps(importlib.metadata.version('jsonschema')))", [])
        if code:
            raise AdapterRefusal("FREE_ENERGY_SCHEMA_UNAVAILABLE", "Native PLSR model loading requires jsonschema 4.26.0")
        identity["model_schema_dependency"] = {"name":"jsonschema", "version":_json(raw)}
    check_runtime(role,identity)
    return identity


def bind(repositories, expected=None):
    """Bind the exact three host-selected checkouts; never take paths from JSON."""
    if type(repositories) is not dict or set(repositories) != ROLES or expected is not None and set(expected) != ROLES:
        raise ValueError("Bind exactly CSG, GSIE and PLSR")
    adapters, runtimes = {}, {}
    for role in sorted(ROLES):
        pin = PINS[role]
        prior = expected[role] if expected is not None else {}
        adapter = PinnedSubprocessAdapter(repositories[role],pin["revision"],pin["module"],source_root=pin["source_root"],
            expected_python_sha256=prior.get("python_sha256"), expected_python_version=prior.get("python_version"),
            expected_dependencies=prior.get("dependencies"))
        identity = runtime_identity(role,adapter)
        if prior and canonical(_projection(identity)) != canonical(_projection(prior)):
            raise ValueError("Native runtime differs from the retained execution")
        adapters[role],runtimes[role] = adapter,identity
    return adapters,runtimes


_BOOTSTRAP = r'''
import json,pathlib,sys
role, root, bridge_root = sys.argv[1:4]
sys.path.insert(0,bridge_root)
sys.path.insert(0,root)
from ciw.free_energy_native import _dispatch
request = json.loads(sys.stdin.buffer.read())
print(json.dumps(_dispatch(role,pathlib.Path(root),request),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))
'''


def invoke(role, adapters, request):
    request = validate_request(role,request)
    adapter = adapters[role]
    before = runtime_identity(role,adapter)
    code, raw = adapter._run(_BOOTSTRAP,[role,str(adapter.source_root),str(Path(__file__).resolve().parent.parent)],canonical(request))
    after = runtime_identity(role,adapter)
    if canonical(_projection(before)) != canonical(_projection(after)):
        raise ValueError("Native provider changed during execution")
    if code:
        raise AdapterRefusal("FREE_ENERGY_NATIVE_REFUSED", "Pinned " + role + " refused the bounded native stage")
    response = _json(raw)
    if (type(response) is not dict or response.get("schema") != "ciw.free-energy-"+role+"-native.v1" or
            canonical(response.get("request")) != canonical(request)):
        raise ValueError("Native-stage response does not bind its request")
    validate_response(role,request,response)
    return response


def _same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Native-stage retained binding differs")


def _domain_digest(domain, value):
    return "sha256:" + sha256(domain.encode()+b"\0"+canonical(value)).hexdigest()


def _close(actual, expected):
    actual, expected = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    if (actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)) or
            not np.allclose(actual,expected,rtol=1e-10,atol=scale*1e-12)):
        raise ValueError("Native-stage numerical evidence contradicts its declared inputs")


def validate_response(role, request, response):
    """Offline binding and numerical consistency checks, without provider imports.

    Passing checks authenticates neither an execution nor a physical model.
    The aggregate workflow adds its own conditioning and acceptance policies.
    """
    try:
        request = validate_request(role,request)
        canonical(response)
        common = {"schema", "request"}
        fields = {"csg":{"record","Phi","determinant"},
            "gsie":{"operation","estimate","replay_snapshot","numerical_result_id","noise_policy","observability"},
            "plsr":{"model_artifact","sample","record","matrix_assessment"}}
        _keys(response,common|fields[role])
        if response["schema"] != "ciw.free-energy-"+role+"-native.v1":
            raise ValueError("Unsupported native-stage response schema")
        _same(response["request"],request)
        if role == "csg":
            _validate_csg(request,response)
        elif role == "gsie":
            _validate_gsie(request,response)
        else:
            from .identified_stability import _record
            authored = plsr_model_document(request)
            authored["artifact_digest"] = sha256(json.dumps(authored,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            _same(response["model_artifact"],authored)
            sample = {"sample_schema":"plsr-sample-v1","x":request["error"],"theta":None,"theta_dot":None}
            _same(response["sample"],sample)
            _record({"model_artifact":authored},{"mean":request["error"]},response["record"])
            _same(response["matrix_assessment"],_matrix_assessment(response["record"]))
    except (TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError, UnicodeError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed native-stage retained response") from exc


def _validate_csg(request,response):
    from .geodesic_reference import _check_curved
    count = len(request["arclength"])
    phi = response["Phi"]
    if type(phi) is not list or len(phi) != count:
        raise ValueError("Native transfer count differs from the declared grid")
    matrices = np.asarray([_matrix(matrix,2,2) for matrix in phi])
    record = response["record"]
    for i,matrix in enumerate(phi):
        _same(matrix,[[record["a"][i],record["b"][i]],[record["a_rate"][i],record["b_rate"][i]]])
    determinant = _vector(response["determinant"],count)
    _close(determinant,matrices[:,0,0]*matrices[:,1,1]-matrices[:,1,0]*matrices[:,0,1])
    # These three convenience arrays are derived checker inputs, not newly
    # retained provider evidence or a substitute for the received record.
    _check_curved(csg_source(request),{"schema":"ciw.curved-path-transfer-native.v1","record":record,
        "propagated_covariance":(matrices@matrices.transpose(0,2,1)).tolist(),
        "separation":[0.0]*count,"heading_change":[0.0]*count,"determinant":determinant})


def _validate_gsie(request,response):
    count = len(request["y"])
    g = np.asarray(request["G"],dtype=float)
    y, prior_mean = np.asarray(request["y"],dtype=float),np.asarray(request["prior"]["mean"],dtype=float)
    sigma, prior_covariance = np.asarray(request["Sigma"],dtype=float),np.asarray(request["prior"]["covariance"],dtype=float)
    prior_id = "ciw-free-energy-prior:"+digest(request["prior"])
    observation_id = "ciw-free-energy-observation:"+digest(request)
    model_id = "ciw-free-energy-observation-map:"+digest(request["G"])
    operation = "geometric-state-inference.update.v1"
    _same({k:response[k] for k in ("operation","noise_policy","observability")},
          {"operation":operation,"noise_policy":"measurement_independent_of_prior","observability":"not_assessed"})
    replay = {"schema":"geometric-state-inference.replay.v1","operation_ref":operation,
        "prior":{"time":0.0,"mean":prior_mean.tolist(),"covariance":prior_covariance.tolist(),"frame_id":FRAME,
            "units":["1","1"],"state_id":prior_id,"dynamics_model_id":None,"predecessor_state_id":None,"operation_ref":None,"replay":None},
        "observation":{"time":0.0,"values":y.tolist(),"covariance":sigma.tolist(),"frame_id":MEASUREMENT_FRAME,
            "units":["1"]*count,"observation_id":observation_id,"evidence_refs":[request["evidence_id"]]},
        "model":{"matrix":g.tolist(),"model_id":model_id,"measurement_geometry":"euclidean.v1",
            "measurement_units":["1"]*count,"measurement_frame_id":MEASUREMENT_FRAME},
        "state_geometry":"euclidean.v1","noise_policy":"measurement_independent_of_prior"}
    _same(response["replay_snapshot"],replay)
    estimate = response["estimate"]
    _keys(estimate,{"time","mean","covariance","frame_id","units","innovation","innovation_covariance","residual","nis",
        "observation_id","evidence_refs","observation_model_id","dynamics_model_id","prior_state_id","state_id","replay_json","observability_assessment_id"})
    expected = {"time":0.0,"frame_id":FRAME,"units":["1","1"],"observation_id":observation_id,
        "evidence_refs":[request["evidence_id"]],"observation_model_id":model_id,"dynamics_model_id":None,
        "prior_state_id":prior_id,"replay_json":canonical(replay).decode(),"observability_assessment_id":None}
    _same({k:estimate[k] for k in expected},expected)
    mean = np.asarray(_vector(estimate["mean"],2,float("inf")))
    covariance = _matrix(estimate["covariance"],2,2,float("inf"))
    innovation = np.asarray(_vector(estimate["innovation"],count,float("inf")))
    retained_innovation_covariance = _matrix(estimate["innovation_covariance"],count,count,float("inf"))
    residual = np.asarray(_vector(estimate["residual"],count,float("inf")))
    nis = _number(estimate["nis"],float("inf"))
    for matrix in (covariance,retained_innovation_covariance):
        if not np.array_equal(matrix,matrix.T) or np.linalg.eigvalsh(matrix)[0] <= 0:
            raise ValueError("Native Gaussian covariance must remain symmetric positive definite")
    if nis < 0:
        raise ValueError("Native normalized innovation cannot be negative")
    expected_innovation = y-g@prior_mean
    s = g@prior_covariance@g.T+sigma
    gain = np.linalg.solve(s,g@prior_covariance).T
    remainder = np.eye(2)-gain@g
    _close(innovation,expected_innovation)
    _close(retained_innovation_covariance,s)
    _close(mean,prior_mean+gain@expected_innovation)
    _close(covariance,remainder@prior_covariance@remainder.T+gain@sigma@gain.T)
    _close(residual,y-g@mean)
    _close(nis,float(expected_innovation@np.linalg.solve(s,expected_innovation)))
    if estimate["state_id"] != _domain_digest("geometric-state-inference.state-transition.v1",
            {"configuration":replay,"mean":estimate["mean"],"covariance":estimate["covariance"]}):
        raise ValueError("Native Gaussian state identity differs")
    numerical = {key:estimate[key] for key in ("time","mean","covariance","frame_id","units","innovation","innovation_covariance","residual","nis")}
    if response["numerical_result_id"] != _domain_digest("geometric-state-inference.numerical-result.v1",numerical):
        raise ValueError("Native Gaussian numerical identity differs")


def _matrix_assessment(record):
    diagnostic = record["diagnostics"]
    resolved = diagnostic is not None and diagnostic["min_P"] > 0 and diagnostic["margin"] > diagnostic["resolution"]
    return {"scope":"fixed_declared_normalized_linear_error_iteration", "basis":"native_full_symmetric_decrease_matrix_spectrum",
        "max_decrease_eigenvalue":None if diagnostic is None else diagnostic["max_decrease"],
        "min_certificate_eigenvalue":None if diagnostic is None else diagnostic["min_P"],
        "matrix_margin":None if diagnostic is None else diagnostic["margin"],
        "numerical_resolution":None if diagnostic is None else diagnostic["resolution"],
        "numerically_negative_definite":bool(resolved),
        "state_scope":"all_nonzero_errors_for_this_fixed_A_and_P_if_resolvably_negative_definite",
        "iteration_time":"sample_period_s_1_is_bookkeeping_not_physical_time",
        "physical_stability":"not_established", "interval_certification":"not_performed", "proof_status":"NOT_CHECKED"}


def _native(value):
    import dataclasses
    if dataclasses.is_dataclass(value):
        return _native(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key:_native(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_native(item) for item in value]
    if isinstance(value,np.ndarray):
        return _native(value.tolist())
    if isinstance(value,np.generic):
        return _native(value.item())
    return value


def plsr_model_document(request):
    """Author the fixed model from declared precision, without sealing a result."""
    request = validate_request("plsr",request)
    precision = np.asarray(request["Lambda"],dtype=float)
    transition = np.eye(2) - request["alpha"] * precision
    certificate = precision / 2
    commitment = digest(request).removeprefix("sha256:")
    return {
        "artifact_schema":"model-artifact-v1", "model_id":"ciw-free-energy-mean-error", "model_version":"1",
        "state":{"definition":"Normalized mean error e=mean-reference for a fixed posterior. One discrete step is one algorithm iteration; sample_period_s=1 is bookkeeping, not a physical clock.",
                 "coordinates":[{"name":"normalized_lateral_error","unit":"1"},{"name":"normalized_heading_error","unit":"1"}]},
        "time":{"convention":"discrete", "sample_period_s":1.0}, "plant":{"kind":"linear","A":transition.tolist()},
        "certificate":{"kind":"quadratic","P":certificate.tolist()},
        "policy":{"required_margin":float(request["required_margin"]), "level":None,
            "margin_derivation":{"method":"declared-computational-margin", "description":"Caller margin for the fixed normalized iteration; no allowance for physical uncertainty.",
                "evidence_digest":request["evidence_id"].removeprefix("sha256:"),
                "quantity":"negative-largest-eigenvalue-of-decrease-matrix"},
            "numerical_policy":"float64-decrease-v1", "runtime_status_schema":"runtime-status-v1", "claim_codes_schema":"claim-codes-v1"},
        "estimator":{"identity":"ciw-fixed-gaussian-free-energy-reference", "version":"1", "configuration_digest":commitment,
            "state_compatibility":"Dimensionless ordered lateral and heading mean errors relative to the fixed declared Gaussian posterior."},
        "provenance":{"producer":"ciw-free-energy-bridge", "producer_version":"1", "model_data_digest":commitment,
                      "construction_report_digest":commitment},
        "claim_scope":"computational-integrity-only", "may_authorize":False,
    }


def _dispatch(role, root, request):
    """Child-process entry point; do not call inside the workbench process."""
    import importlib
    request = validate_request(role,request)
    package = importlib.import_module({"csg":"geodesic_testbed", "gsie":"geometric_state_inference", "plsr":"lyapunov"}[role])
    if not Path(package.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError("Native package escaped its bound checkout")
    result = {"schema":"ciw.free-energy-"+role+"-native.v1", "request":request}
    if role == "csg":
        from geodesic_testbed.jacobi import integrate_jacobi
        from geodesic_testbed.boundary import Units,StartingCovariance
        reference = csg_source(request)
        units = Units(**request["units"])
        record = integrate_jacobi(request["arclength"],request["gaussian_curvature"]).as_transfer_record(
            units=units, covariance=StartingCovariance(units=units,**reference["starting_covariance"]),
            relative_tolerance=request["relative_tolerance"], observation_mode="intrinsic-surface-distance")
        result.update(record=record.to_dict(include_samples=True),Phi=record.transfer_map().matrices(),determinant=record.determinant)
    elif role == "gsie":
        from geometric_state_inference import StatePrior,Observation,LinearObservation,update
        prior_id = "ciw-free-energy-prior:" + digest(request["prior"])
        prior = StatePrior(time=0.0,frame_id=FRAME,units=("1","1"),state_id=prior_id,**request["prior"])
        units = tuple("1" for _ in request["y"])
        observation = Observation(time=0.0,values=request["y"],covariance=request["Sigma"],frame_id=MEASUREMENT_FRAME,
            units=units,observation_id="ciw-free-energy-observation:"+digest(request),evidence_refs=(request["evidence_id"],))
        model = LinearObservation(matrix=request["G"],model_id="ciw-free-energy-observation-map:"+digest(request["G"]),
            measurement_units=units,measurement_frame_id=MEASUREMENT_FRAME)
        estimate = update(prior,observation,model)
        result.update(operation="geometric-state-inference.update.v1",estimate=_native(estimate),
            replay_snapshot=estimate.replay_snapshot,numerical_result_id=estimate.numerical_result_id,
            noise_policy="measurement_independent_of_prior",observability="not_assessed")
    else:
        import lyapunov
        from . import plsr_engine as bridge
        model = lyapunov.model_artifact_from_dict(lyapunov.seal_model_artifact(plsr_model_document(request)))
        sample = {"sample_schema":"plsr-sample-v1", "x":request["error"], "theta":None, "theta_dot":None}
        verdict = model.verdict(sample["x"],theta=None,theta_dot=None)
        body = {"adapter_schema":bridge.EVALUATION_SCHEMA,"model_artifact_schema":"model-artifact-v1",
            "model_artifact_digest":model.artifact_digest,"sample":sample,"runtime_status_schema":lyapunov.RUNTIME_STATUS_VERSION,
            "code":verdict.code,"presentation_category":bridge._CATEGORIES[verdict.code],
            "inequality_certified":bool(verdict.inequality_certified),"meets_required_margin":bool(verdict.meets_required_margin),
            "operationally_acceptable":bool(verdict.operationally_acceptable),"required_margin":float(verdict.required_margin),
            "level":model.level,"details":verdict.details,"proof_status":"NOT_CHECKED","diagnostics":bridge._diagnostics(verdict)}
        record = lyapunov.as_companion_record(kind="ciw-plsr-evaluation",body=body)
        result.update(model_artifact=model.to_dict(),sample=sample,record=record,matrix_assessment=_matrix_assessment(record))
    return _native(result)
