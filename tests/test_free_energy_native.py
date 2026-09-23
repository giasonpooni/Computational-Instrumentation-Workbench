"""Contract tests plus optional real, source-pinned native-stage checks."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ciw.free_energy_native import (PINS, ROLES, bind, check_runtime, csg_source, invoke,
                                    plsr_model_document, validate_request, validate_response)
from ciw.telemetry import canonical


def requests():
    evidence = "sha256:" + "1" * 64
    return {
        "csg":{"arclength":[0,.25,.5,.75,1],"gaussian_curvature":0,
               "units":{"length":"m","angle":"radian"},"relative_tolerance":1e-6},
        "gsie":{"G":[[1,0],[0,1]],"y":[1,-1],"Sigma":[[.5,.2],[.2,.75]],
                "prior":{"mean":[0,0],"covariance":[[2,.5],[.5,1]]},"evidence_id":evidence},
        "plsr":{"Lambda":[[1,0],[0,4]],"alpha":.1,"error":[1,0],"evidence_id":evidence,"required_margin":0},
    }


def test_valid_requests_are_retained_without_coercion_or_aliasing():
    for role,value in requests().items():
        saved = deepcopy(value)
        validated = validate_request(role,value)
        assert validated == saved and validated is not value
        assert json.dumps(validated,sort_keys=True) == json.dumps(value,sort_keys=True)
    source = csg_source(requests()["csg"])
    assert source["initial_perturbation"] == [0.0,0.0]
    assert source["starting_covariance"]["basis"] == "assumed"
    assert "not the fused prior" in source["starting_covariance"]["note"]


@pytest.mark.parametrize("role,change", [
    ("csg",lambda r:r.update(arclength=[0,0,1])),
    ("csg",lambda r:r.update(gaussian_curvature=True)),
    ("csg",lambda r:r["units"].update(angle="degree")),
    ("csg",lambda r:r.update(relative_tolerance=0)),
    ("gsie",lambda r:r.update(G=[[1,0]])),
    ("gsie",lambda r:r.update(y=[True,1])),
    ("gsie",lambda r:r.update(Sigma=[[1,.5],[0,1]])),
    ("gsie",lambda r:r.update(Sigma=[[1,2],[2,1]])),
    ("gsie",lambda r:r["prior"].update(covariance=[[0,0],[0,1]])),
    ("gsie",lambda r:r.update(evidence_id="unretained")),
    ("plsr",lambda r:r.update(error=[0])),
    ("plsr",lambda r:r.update(alpha=False)),
    ("plsr",lambda r:r.update(required_margin=-1)),
    ("plsr",lambda r:r.update(Lambda=[[1,2],[2,1]])),
    ("plsr",lambda r:r.update(Lambda=[[1e21,0],[0,1e21]])),
    ("plsr",lambda r:r.update(error=[1e13,0])),
    ("plsr",lambda r:r.update(error=[10**400,1])),
])
def test_bad_stage_inputs_are_refused_before_execution(role,change):
    value = requests()[role]
    change(value)
    with pytest.raises(ValueError):
        validate_request(role,value)


def test_model_authorship_binds_precision_step_and_iteration_semantics():
    request = requests()["plsr"]
    model = plsr_model_document(request)
    assert model["plant"]["A"] == [[.9,0.0],[0.0,.6]]
    assert model["certificate"]["P"] == [[.5,0.0],[0.0,2.0]]
    assert model["policy"]["level"] is None
    assert model["may_authorize"] is False
    assert "not a physical clock" in model["state"]["definition"]
    assert "artifact_digest" not in model  # The native loader seals the authored model.


def test_derived_precision_and_error_transport_preserves_finite_quadratics():
    request = requests()["plsr"]
    request.update(Lambda=[[1e20,0],[0,1e20]],alpha=1e6,error=[1e12,-1e12])
    model = plsr_model_document(request)
    a, p, error = (np.asarray(value,dtype=float) for value in
                   (model["plant"]["A"],model["certificate"]["P"],request["error"]))
    decrease = a.T@p@a-p
    assert np.all(np.isfinite(decrease))
    assert np.isfinite(error@decrease@error)
    assert np.max(np.abs(decrease)) < 1e100


def test_runtime_contract_requires_pins_and_schema_dependency():
    for role in ROLES:
        runtime = {"schema":"ciw.subprocess-runtime.v1","adapter_version":"ciw-pinned-subprocess-v1",
            "repository_root":"test-only-reference","python_executable":"test-only-reference",
            "python_sha256":"a"*64,"python_version":"3.12.14","dependencies":{"numpy":"2.4.3","scipy":None},
            **PINS[role]}
        if role == "plsr":
            runtime["model_schema_dependency"] = {"name":"jsonschema","version":"4.26.0"}
        check_runtime(role,runtime)
        changed = deepcopy(runtime)
        changed["source_tree"] = "f"*40
        with pytest.raises(ValueError):
            check_runtime(role,changed)
        if role == "plsr":
            runtime["model_schema_dependency"]["version"] = "4.25.0"
            with pytest.raises(ValueError):
                check_runtime(role,runtime)
    with pytest.raises(ValueError):
        bind({})


def test_bridge_import_does_not_import_optional_native_packages():
    import ciw
    code = "import sys;sys.path.insert(0,sys.argv[1]);import ciw.free_energy_native;assert not {'lyapunov','geometric_state_inference','geodesic_testbed'} & sys.modules.keys()"
    subprocess.run([sys.executable,"-I","-c",code,str(Path(ciw.__file__).resolve().parent.parent)],check=True)


@pytest.fixture(scope="module")
def bound(tmp_path_factory):
    stack = os.environ.get("CIW_FREE_ENERGY_STACK_ROOT")
    declared = ({role:str(Path(stack)/role) for role in ROLES} if stack else
                {role:os.environ.get("CIW_"+role.upper()+"_REPO") for role in ROLES})
    if not all(declared.values()):
        pytest.skip("Real native stages require CIW_FREE_ENERGY_STACK_ROOT or all three per-provider paths")
    # Clone to isolate provider bytes and ownership without altering host checkouts.
    directory = tmp_path_factory.mktemp("free-energy-native")
    paths = {}
    for role,path in declared.items():
        target = directory / role
        trusted = Path(path).resolve()
        subprocess.run(["git","-c","safe.directory="+trusted.as_posix(),
                        "-c","safe.directory="+(trusted/".git").as_posix(),
                        "-c","core.autocrlf=false","clone","--quiet","--no-hardlinks",path,str(target)],
                       check=True,capture_output=True)
        subprocess.run(["git","-C",str(target),"checkout","--quiet","--detach",PINS[role]["revision"]],
                       check=True,capture_output=True)
        paths[role] = target
    return (*bind(paths), paths)


def test_native_csg_transfer_matches_flat_analytic_reference(bound):
    adapters,_,_ = bound
    request = requests()["csg"]
    response = invoke("csg",adapters,request)
    assert response == invoke("csg",adapters,request)
    for s,phi in zip(request["arclength"],response["Phi"]):
        np.testing.assert_allclose(phi,[[1,s],[0,1]],rtol=0,atol=0)
    assert response["record"]["covariance"]["basis"] == "assumed"
    assert response["record"]["calibration"]["bound"] is False


def test_native_gsie_full_covariance_matches_information_form(bound):
    adapters,_,_ = bound
    request = requests()["gsie"]
    response = invoke("gsie",adapters,request)
    assert response == invoke("gsie",adapters,request)
    precision = np.linalg.inv(request["prior"]["covariance"]) + np.linalg.inv(request["Sigma"])
    expected_covariance = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision,np.linalg.solve(request["Sigma"],request["y"]))
    np.testing.assert_allclose(response["estimate"]["mean"],expected_mean,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(response["estimate"]["covariance"],expected_covariance,rtol=1e-12,atol=1e-14)
    assert response["replay_snapshot"]["observation"]["covariance"] == request["Sigma"]
    assert response["observability"] == "not_assessed"
    assert response["noise_policy"] == "measurement_independent_of_prior"


def test_native_plsr_checks_entire_fixed_matrix_not_only_sample_direction(bound):
    adapters,_,_ = bound
    request = requests()["plsr"]
    stable = invoke("plsr",adapters,request)
    assert stable == invoke("plsr",adapters,request)
    assert stable["record"]["code"] == "CERTIFIED_WITH_MARGIN"
    assert stable["matrix_assessment"]["numerically_negative_definite"] is True
    assert stable["matrix_assessment"]["matrix_margin"] == pytest.approx(.095)
    assert stable["record"]["proof_status"] == "NOT_CHECKED"
    zero = deepcopy(request)
    zero["error"] = [0,0]
    equilibrium = invoke("plsr",adapters,zero)
    assert equilibrium["record"]["diagnostics"]["value"] == 0
    assert equilibrium["record"]["diagnostics"]["decrease"] == 0
    assert equilibrium["matrix_assessment"]["numerically_negative_definite"] is True
    request["alpha"] = .6  # Stable in sampled first axis, unstable in second axis.
    unstable = invoke("plsr",adapters,request)
    assert unstable["record"]["diagnostics"]["scaled_decrease"] < 0
    assert unstable["record"]["code"] == "DECREASE_NOT_DEFINITE"
    assert unstable["matrix_assessment"]["numerically_negative_definite"] is False


def test_native_plsr_accepts_largest_derived_transport_budget(bound):
    adapters,_,_ = bound
    request = requests()["plsr"]
    request.update(Lambda=[[1e20,0],[0,1e20]],alpha=1e6,error=[1e12,-1e12])
    response = invoke("plsr",adapters,request)
    assert response["record"]["code"] == "NOT_CERTIFIED"
    assert response["matrix_assessment"]["numerically_negative_definite"] is False
    assert response["record"]["diagnostics"]["nonfinite_fields"] == []


def test_native_replay_bindings_and_no_parent_provider_import(bound):
    _,runtimes,paths = bound
    _,again = bind(paths,runtimes)
    assert again == runtimes
    altered = deepcopy(runtimes)
    altered["gsie"]["python_sha256"] = "f"*64
    with pytest.raises((ValueError,RuntimeError)):
        bind(paths,altered)
    assert not {"lyapunov","geometric_state_inference","geodesic_testbed"} & sys.modules.keys()


@pytest.fixture(scope="module")
def responses(bound):
    return {role:invoke(role,bound[0],request) for role,request in requests().items()}


def _reseal_gsie(response):
    estimate = response["estimate"]
    def identity(domain,value):
        return "sha256:"+sha256(domain.encode()+b"\0"+canonical(value)).hexdigest()
    estimate["state_id"] = identity("geometric-state-inference.state-transition.v1",
        {"configuration":response["replay_snapshot"],"mean":estimate["mean"],"covariance":estimate["covariance"]})
    response["numerical_result_id"] = identity("geometric-state-inference.numerical-result.v1",
        {key:estimate[key] for key in ("time","mean","covariance","frame_id","units","innovation","innovation_covariance","residual","nis")})


@pytest.mark.parametrize("role,change", [
    ("csg",lambda r:r["Phi"][1][0].__setitem__(1,.9)),
    ("csg",lambda r:r["record"]["units"].update(angle="degree")),
    ("csg",lambda r:r["determinant"].__setitem__(1,.5)),
    ("gsie",lambda r:r["estimate"]["mean"].__setitem__(0,9.0)),
    ("gsie",lambda r:r["estimate"]["covariance"][0].__setitem__(0,9.0)),
    ("gsie",lambda r:r["estimate"].update(nis=True)),
    ("gsie",lambda r:r["estimate"].update(replay_json="{}")),
    ("gsie",lambda r:r.update(observability="observable")),
    ("plsr",lambda r:r["model_artifact"]["plant"]["A"][0].__setitem__(0,.2)),
    ("plsr",lambda r:r["matrix_assessment"].update(matrix_margin=99)),
    ("plsr",lambda r:r["record"].update(proof_status="VERIFIED")),
])
def test_offline_native_response_checks_resealed_or_altered_evidence(responses,role,change,monkeypatch):
    response = deepcopy(responses[role])
    change(response)
    if role == "gsie":
        _reseal_gsie(response)
    if role == "plsr":
        model = response["model_artifact"]
        model["artifact_digest"] = sha256(json.dumps({k:v for k,v in model.items() if k != "artifact_digest"},
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    monkeypatch.setattr(subprocess,"Popen",lambda *_args,**_kwargs:pytest.fail("Offline validation ran a provider"))
    with pytest.raises(ValueError):
        validate_response(role,requests()[role],response)
    assert not {"lyapunov","geometric_state_inference","geodesic_testbed"} & sys.modules.keys()
