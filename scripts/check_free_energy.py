"""Run pinned CSG, GSIE and PLSR free-energy experiments from an isolated wheel."""
import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from provider_checkouts import validate_checkout

REPOSITORIES = {"csg":"Curved-Surface-Geodesic-Sensitivity-Runtime",
    "gsie":"Geometric-State-Inference-Engine", "plsr":"Parameterized-Lyapunov-Stability-Runtime"}
REQUIRED_TESTS = {
    "test_native_csg_transfer_matches_flat_analytic_reference",
    "test_native_gsie_full_covariance_matches_information_form",
    "test_native_plsr_checks_entire_fixed_matrix_not_only_sample_direction",
    "test_all_cases_expose_inference_and_model_adequacy_separately",
    "test_reproduction_and_replay_have_fresh_occurrences",
    "test_shared_session_exposes_three_native_stages_and_diagnostics",
    "test_saved_workspace_reopens_without_runtime_or_numerical_execution",
}


def call(command, **kwargs):
    return subprocess.run(command, check=True, timeout=kwargs.pop("timeout",300), **kwargs)


def pins(path):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "PINS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing exact free-energy provider pins")


def exact_source(path,pin):
    path = validate_checkout(path,pin["revision"])
    tree = call(["git","--no-replace-objects","-C",str(path),"rev-parse","HEAD^{tree}"],
                capture_output=True,text=True).stdout.strip()
    if tree != pin["source_tree"]:
        raise ValueError("Free-energy provider tree differs from its approved pin")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    configured_stack = os.environ.get("CIW_FREE_ENERGY_STACK_ROOT")
    parser.add_argument("--stack-root",type=Path,default=Path(configured_stack) if configured_stack else None,
                        help="Existing exact clean csg/gsie/plsr checkouts; no fetching or modification")
    parser.add_argument("--output-dir",type=Path,default=Path("results/free-energy"))
    parser.add_argument("--temporary-root",type=Path,help="Optional parent for isolated build and test directories")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    declarations = pins(root/"src/ciw/free_energy_native.py")
    if set(declarations) != set(REPOSITORIES):
        raise ValueError("Require all three pinned free-energy providers")
    tests = sorted((root/"tests").glob("test_free_energy_*.py"))
    if not tests:
        raise ValueError("The installed free-energy test suite is missing")
    destination = args.output_dir.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Gate output must be new or empty")
    destination.mkdir(parents=True,exist_ok=True)
    if args.temporary_root is not None:
        args.temporary_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ciw-free-energy-",dir=args.temporary_root) as directory:
        temporary,providers = Path(directory).resolve(),{}
        stack = args.stack_root.resolve() if args.stack_root else temporary/"providers"
        if not args.stack_root:
            stack.mkdir()
        for role,repository in REPOSITORIES.items():
            path = stack/role
            if not args.stack_root:
                call(["git","-c","core.autocrlf=false","clone","--quiet","--no-checkout",
                      "https://github.com/giasonpooni/"+repository+".git",str(path)])
                call(["git","-C",str(path),"-c","core.autocrlf=false","checkout","--quiet","--detach",declarations[role]["revision"]])
            providers[role] = exact_source(path,declarations[role])
        build = temporary/"build"
        shutil.copytree(root/"src",build/"src",ignore=shutil.ignore_patterns("__pycache__","*.egg-info"))
        for name in ("pyproject.toml","README.md","LICENSE"):
            if (root/name).is_file():
                shutil.copyfile(root/name,build/name)
        wheels = temporary/"wheels"
        call([sys.executable,"-m","pip","wheel","--no-deps","--no-build-isolation",str(build),"-w",str(wheels)])
        wheel, = wheels.glob("*.whl")
        environment = temporary/"environment"
        call([sys.executable,"-m","venv",str(environment)])
        python = environment/("Scripts/python.exe" if os.name == "nt" else "bin/python")
        call([str(python),"-m","pip","install",str(wheel)+"[bench-models]","pytest==9.0.2"])
        work = temporary/"installed-check"
        (work/"tests").mkdir(parents=True)
        for path in tests:
            shutil.copyfile(path,work/"tests"/path.name)
        shutil.copytree(root/"examples/variational-free-energy",work/"examples/variational-free-energy")
        env = {**os.environ,**{"CIW_"+role.upper()+"_REPO":str(path) for role,path in providers.items()},
               "CIW_FREE_ENERGY_STACK_ROOT":str(stack),"CIW_FREE_ENERGY_FIXTURE_DIR":str(destination),
               "PYTHONDONTWRITEBYTECODE":"1"}
        env.pop("PYTHONPATH",None)
        env.pop("PYTEST_ADDOPTS",None)
        imported = call([str(python),"-I","-c","import ciw; print(ciw.__file__)"],
                        cwd=work,env=env,text=True,capture_output=True)
        if not Path(imported.stdout.strip()).resolve().is_relative_to(environment.resolve()):
            raise AssertionError("Free-energy gate must execute the installed wheel")
        report = destination/"tests.xml"
        call([str(python),"-I","-B","-m","pytest","-q","-p","no:cacheprovider",
              "--basetemp",str(temporary/"pytest"),"--junitxml",str(report),
              *["tests/"+path.name for path in tests]],cwd=work,env=env,timeout=1200)
        cases = list(ET.parse(report).getroot().iter("testcase"))
        if (not cases or any(case.find(tag) is not None for case in cases for tag in ("skipped","failure","error"))
                or not REQUIRED_TESTS <= {case.get("name") for case in cases}):
            raise AssertionError("Native free-energy gate requires all native stages and no skips or failures")
        for role,path in providers.items():
            exact_source(path,declarations[role])
        (destination/"gate.json").write_text(json.dumps({"schema":"ciw.free-energy-provider-gate.v1","status":"passed",
            "tests_passed":len(cases),"providers":declarations,"installed_wheel":wheel.name,
            "installed_wheel_sha256":sha256(wheel.read_bytes()).hexdigest(),
            "test_modules":[path.name for path in tests],
            "checks":"native_jacobi_gaussian_fusion_and_fixed_error_iteration_with_offline_restore_and_fresh_replay",
            "claim_scope":"synthetic_static_linear_gaussian_inference","physical_calibration":"not_established",
            "proof_status":"NOT_CHECKED"},indent=2)+"\n",encoding="utf-8")
    print("PASS: installed free-energy bench with pinned CSG, GSIE and PLSR; no skipped tests")


if __name__ == "__main__":
    main()
