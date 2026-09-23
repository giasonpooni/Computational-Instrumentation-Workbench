"""Subprocess bridge from the lab (Python 3.11+) to the pinned PLSR runtime (Python 3.12+).

The lab never imports ``lyapunov``. A bound interpreter (provider role
``plsr-python``) runs a fixed bootstrap that first verifies the installed
runtime exactly as ``ciw.plsr_engine`` does -- package version equal to the
pin and every packaged source file's SHA-256 (CRLF normalised) equal to
``src/ciw/plsr-runtime.json`` -- and refuses otherwise. Only then does it
build declared plants, certificates and samples from JSON numbers and return
the runtime's own outputs. Source hashes detect installation drift; they do
not authenticate an untrusted interpreter.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import resources
import json
import subprocess

PLSR_ROLE = "plsr-python"
PACKAGE = "parameterized-lyapunov-stability-runtime"
IMPLEMENTATION = "Parameterized-Lyapunov-Stability-Runtime"

_BOOTSTRAP = r'''
import hashlib, importlib, json, platform, sys, warnings
from importlib import metadata, resources


def inventory(root, prefix=""):
    result = {}
    for child in root.iterdir():
        if child.name == "__pycache__":
            continue
        name = prefix + child.name
        if child.is_dir():
            result.update(inventory(child, name + "/"))
        elif name.endswith((".py", ".json")) or child.name == "py.typed":
            result[name] = hashlib.sha256(child.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return result


def emit(payload):
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.flush()


def refuse(code, message):
    emit({"refusal": {"code": code, "message": message}})
    raise SystemExit(0)


request = json.loads(sys.stdin.read())
manifest = request["manifest"]
if sys.version_info < (3, 12):
    refuse("PLSR_PYTHON_UNSUPPORTED", "PLSR requires Python 3.12 or newer; the bound interpreter is "
           + platform.python_version())
try:
    installed = metadata.version(request["package"])
    L = importlib.import_module("lyapunov")
    files = inventory(resources.files("lyapunov"))
except Exception as exc:
    refuse("PLSR_UNAVAILABLE", type(exc).__name__ + ": " + str(exc))
if installed != manifest["package_version"] or getattr(L, "__version__", None) != manifest["package_version"]:
    refuse("PLSR_VERSION_MISMATCH", "installed " + str(installed) + " / module " + str(getattr(L, "__version__", None))
           + " differs from pin " + manifest["package_version"])
if files != manifest["files"]:
    differing = sorted(set(files) ^ set(manifest["files"]) | {key for key in files.keys() & manifest["files"].keys()
                                                              if files[key] != manifest["files"][key]})
    refuse("PLSR_SOURCE_MISMATCH", "installed source differs from commit " + manifest["commit"] + ": "
           + ", ".join(differing))

import numpy as np


def array(value):
    return np.array(value, dtype=float)


def plant(spec):
    if "A0" in spec:
        return L.affine_box_plant(array(spec["A0"]), [array(t) for t in spec["terms"]], spec["theta_min"],
                                  spec["theta_max"], name=spec.get("name", "A(theta)"),
                                  time=spec.get("time", "continuous"), rate_min=spec.get("rate_min"),
                                  rate_max=spec.get("rate_max"))
    return L.LinearPlant(name=spec.get("name", "A"), A=array(spec["A"]), time=spec.get("time", "continuous"))


def certificate(spec):
    if "P0" in spec:
        return L.affine_quadratic(array(spec["P0"]), [array(t) for t in spec["terms"]])
    return L.quadratic(array(spec["P"]))


def sample(s, full):
    out = {name: float(getattr(s, name)) for name in ("value", "decrease", "scaled_value", "scaled_decrease",
                                                      "min_P", "max_decrease", "state_scale", "resolution")}
    out["state_scale_exponent"] = int(s.state_scale_exponent)
    out["value_out_of_range"] = bool(s.value_out_of_range)
    if full:
        out["decrease_matrix"] = s.decrease_matrix.tolist()
        out["A"] = s.A.tolist()
        out["P"] = s.P.tolist()
    return out


def run(case):
    op = case["op"]
    if op == "verdict":
        v = L.verdict(plant(case["plant"]), certificate(case["certificate"]), case["x"], theta=case.get("theta"),
                      theta_dot=case.get("theta_dot"), level=case.get("level"),
                      required_margin=case.get("required_margin", 0.0))
        return {"code": v.code, "details": v.details, "resolution": float(v.resolution),
                "required_margin": float(v.required_margin), "inequality_certified": bool(v.inequality_certified),
                "meets_required_margin": bool(v.meets_required_margin),
                "operationally_acceptable": bool(v.operationally_acceptable), "margin": float(v.margin),
                "margin_ratio": float(v.margin_ratio),
                "sample": None if v.sample is None else sample(v.sample, case.get("full", False))}
    if op == "evaluate":
        s = L.evaluate(plant(case["plant"]), certificate(case["certificate"]), case["x"], theta=case.get("theta"),
                       theta_dot=case.get("theta_dot"))
        return {"sample": sample(s, True)}
    if op == "resolution":
        return {"value": float(L.decrease_resolution(array(case["A"]), array(case["P"]), time=case["time"]))}
    if op == "decrease_matrix":
        return {"matrix": L.decrease_matrix(array(case["A"]), array(case["P"]), time=case["time"]).tolist()}
    if op == "solve":
        q = None if case.get("Q") is None else array(case["Q"])
        return {"P": L.solve_lyapunov(array(case["A"]), q, time=case["time"]).P.tolist()}
    if op == "check_vertices":
        r = L.check_vertices(plant(case["plant"]), certificate(case["certificate"]))
        return {"passed": bool(r.passed), "residual": float(r.residual), "claim": r.claim, "details": r.details}
    if op == "quadratic":
        return {"P": L.quadratic(array(case["P"])).P.tolist()}
    if op == "require_status":
        return {"accepted": L.require_status(case["code"])}
    if op == "host_verdict":
        return {"accepted": L.Verdict(code=case["code"], sample=None, details="host status").code}
    if op == "constants":
        return {"DECREASE_RESOLUTION_FACTOR": float(L.DECREASE_RESOLUTION_FACTOR),
                "NUMERICAL_POLICY_VERSION": L.NUMERICAL_POLICY_VERSION, "MAX_KRONECKER_DIM": int(L.MAX_KRONECKER_DIM),
                "MIN_DECREASE_MARGIN": float(L.MIN_DECREASE_MARGIN), "RUNTIME_STATUS_VERSION": L.RUNTIME_STATUS_VERSION,
                "RUNTIME_STATUSES": sorted(L.RUNTIME_STATUSES), "HOST_OWNED_STATUSES": sorted(L.HOST_OWNED_STATUSES)}
    raise ValueError("unknown bridge operation " + repr(op))


results = []
for case in request["cases"]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = {"ok": True, **run(case)}
        except Exception as exc:
            result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    result["id"] = case["id"]
    result["warnings"] = sorted({w.category.__name__ for w in caught})
    results.append(result)
emit({"identity": {"python_version": platform.python_version(), "numpy_version": np.__version__,
                   "package_version": installed, "files_verified": len(files)}, "results": results})
'''


class ProviderRefusal(ValueError):
    """The bound PLSR interpreter is unusable or does not match the pin."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def manifest() -> dict:
    return json.loads(resources.files("ciw").joinpath("plsr-runtime.json").read_text(encoding="utf-8"))


def source_digest(pin: dict) -> str:
    """The digest ciw.plsr_engine.runtime_identity reports for the pinned file inventory."""
    canonical = json.dumps(pin["files"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return sha256(canonical.encode("ascii")).hexdigest()


def run_plsr(python, cases: list, *, pin: dict | None = None, timeout: float = 300.0) -> dict:
    """Evaluate declared cases with the pinned runtime in a subprocess; refuse any identity mismatch."""
    pin = manifest() if pin is None else pin
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Bridge case identities must be unique")
    payload = json.dumps({"manifest": pin, "package": PACKAGE, "cases": cases})
    try:
        result = subprocess.run([str(python), "-I", "-c", _BOOTSTRAP], input=payload, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderRefusal("PLSR_EXECUTION_FAILED", f"Provider subprocess did not complete: {exc}")
    if result.returncode != 0:
        raise ProviderRefusal("PLSR_EXECUTION_FAILED", result.stderr.strip()[-600:] or "provider exited nonzero")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProviderRefusal("PLSR_EXECUTION_FAILED", f"Provider output is not JSON: {exc}")
    if "refusal" in data:
        raise ProviderRefusal(data["refusal"]["code"], data["refusal"]["message"])
    results = {item["id"]: item for item in data.get("results", [])}
    if list(results) != ids:
        raise ProviderRefusal("PLSR_EXECUTION_FAILED", "Provider returned a different set of case results")
    identity = {"role": PLSR_ROLE, "implementation": IMPLEMENTATION, "repository": pin["repository"],
                "commit": pin["commit"], "package_version": pin["package_version"],
                "source_digest": source_digest(pin), "files_pinned": len(pin["files"]),
                "entry": "lyapunov.verdict / lyapunov.solve_lyapunov / lyapunov.decrease_resolution",
                "verification": "package version and per-file sha256 checked in the subprocess before use",
                **data["identity"]}
    return {"identity": identity, "results": results}


def provider_basis(identity: dict) -> dict:
    """basis.provider for findings returned by the executed, verified runtime."""
    return {"repository": identity["repository"], "revision": identity["commit"],
            "runtime_digest": identity["source_digest"], "executed": True, "role": PLSR_ROLE,
            "package_version": identity["package_version"]}


def producer(identity: dict) -> dict:
    return {"implementation": f"{IMPLEMENTATION}@{identity['commit'][:12]}", "revision": identity["commit"]}
