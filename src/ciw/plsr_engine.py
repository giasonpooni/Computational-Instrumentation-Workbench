"""Public-API bridge to the source-pinned, optional PLSR instrument.

Source hashes detect installation drift; they do not authenticate an untrusted
Python installation. The runtime and its dependencies must be trusted locally.
No PLSR import occurs until an instrument operation is requested.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
from importlib import import_module, metadata, resources
import json
import math
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lyapunov import ModelArtifact

ADAPTER_VERSION = "ciw-plsr-adapter-v1"
RUNTIME_COMMIT = "19ea6967060166ba09db6cd4563bd87bd6b3d196"
RUNTIME_REPOSITORY = "https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime"
EVALUATION_SCHEMA = "ciw-plsr-evaluation-v1"
SAMPLE_SCHEMA = "plsr-sample-v1"
_INSTALL_HINT = 'Install the pinned instrument with Python 3.12+ and pip install ".[plsr]".'
_PACKAGE_NAME = "parameterized-lyapunov-stability-runtime"
_CATEGORIES = {
    "CERTIFIED_WITH_MARGIN": "computationally_acceptable",
    "MARGIN_LOW": "margin_shortfall",
    "NOT_CERTIFIED": "certificate_violation",
    "DECREASE_NOT_DEFINITE": "decrease_not_definite",
    "CERTIFICATE_NOT_POSITIVE": "certificate_invalid",
    "NUMERICAL_INCONCLUSIVE": "numerical_refusal",
    "NUMERICAL_OVERFLOW": "numerical_refusal",
    "OUTSIDE_PARAMETER_BOX": "outside_declared_domain",
    "OUTSIDE_LEVEL_SET": "outside_declared_domain",
}
_SAMPLELESS = {"OUTSIDE_PARAMETER_BOX", "NUMERICAL_OVERFLOW"}
_SCALARS = {
    "value", "decrease", "scaled_value", "scaled_decrease", "state_scale",
    "min_P", "max_decrease", "resolution", "margin", "margin_ratio",
}
_DIAGNOSTIC_KEYS = _SCALARS | {
    "nonfinite_fields", "value_overflow", "decrease_overflow", "value_underflow",
    "decrease_underflow", "value_out_of_range", "state_scale_exponent",
    "value_mantissa_exponent", "decrease_mantissa_exponent", "A", "P",
    "decrease_matrix", "P_rate",
}
_RECORD_KEYS = {
    "record_schema", "record_kind", "claim_scope", "may_authorize",
    "confirmed_out_of_development", "maturity", "record_digest", "adapter_schema",
    "model_artifact_schema", "model_artifact_digest", "sample", "runtime_status_schema",
    "code", "presentation_category", "inequality_certified", "meets_required_margin",
    "operationally_acceptable", "required_margin", "level", "details", "proof_status",
    "diagnostics",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _manifest() -> dict[str, Any]:
    """The pinned PLSR package, defined by its provider descriptor."""
    from .pipelines import provider_descriptor
    return provider_descriptor("plsr")["pin"]


def _source_files(root: Any, prefix: str = "") -> dict[str, str]:
    """Inventory package source and schema resources, ignoring bytecode caches."""
    result = {}
    for child in root.iterdir():
        if child.name == "__pycache__":
            continue
        name = prefix + child.name
        if child.is_dir():
            result.update(_source_files(child, name + "/"))
        elif name.endswith((".py", ".json")) or child.name == "py.typed":
            # Git checkouts may use CRLF; wheel construction preserves those
            # bytes. All tracked package resources are text, including py.typed.
            data = child.read_bytes().replace(b"\r\n", b"\n")
            result[name] = sha256(data).hexdigest()
    return result


@lru_cache(maxsize=1)
def _runtime() -> Any:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"PLSR requires Python 3.12 or newer. {_INSTALL_HINT}")
    try:
        manifest = _manifest()
        installed_version = metadata.version(_PACKAGE_NAME)
        runtime = import_module("lyapunov")
        actual = _source_files(resources.files("lyapunov"))
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError(f"The optional PLSR instrument is unavailable. {_INSTALL_HINT}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read the pinned PLSR source. {_INSTALL_HINT}") from exc
    if installed_version != manifest["package_version"] or getattr(
        runtime, "__version__", None
    ) != manifest["package_version"]:
        raise RuntimeError(f"PLSR package version does not match the source pin. {_INSTALL_HINT}")
    if actual != manifest["files"]:
        differing = sorted(set(actual) ^ set(manifest["files"]) | {
            key for key in actual.keys() & manifest["files"].keys()
            if actual[key] != manifest["files"][key]
        })
        raise RuntimeError(
            f"PLSR installed source differs from commit {manifest['commit']}: "
            f"{', '.join(differing)}. {_INSTALL_HINT}"
        )
    return runtime


def runtime_identity() -> dict[str, Any]:
    """Return the verified source pin and execution environment, separately from evidence."""
    _runtime()
    manifest = _manifest()
    return {
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "package_version": manifest["package_version"],
        "source_digest": sha256(_canonical(manifest["files"]).encode("ascii")).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": metadata.version("numpy"),
        "jsonschema_version": metadata.version("jsonschema"),
        "adapter_version": ADAPTER_VERSION,
    }


def load_model(path: str | Path) -> ModelArtifact:
    """Load a complete declared model; never form or reseal a received artifact."""
    return _runtime().load_model_artifact(path)


def model_from_dict(mapping: Mapping[str, Any]) -> ModelArtifact:
    return _runtime().model_artifact_from_dict(mapping)


def _number(value: Any, label: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite JSON number, not a boolean or string")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite in float64")
    if type(value) is int and int(float(value)) != value:
        raise ValueError(f"{label} must be exactly representable in float64")


def _vector(value: Any, count: int, label: str) -> list[int | float]:
    if type(value) is not list or len(value) != count:
        raise ValueError(f"{label} must be an explicit JSON array of {count} numbers")
    for index, item in enumerate(value):
        _number(item, f"{label}[{index}]")
    return list(value)


def validate_sample(model: ModelArtifact, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Check explicit inputs without rejecting out-of-box runtime observations."""
    runtime = _runtime()
    if not isinstance(model, runtime.ModelArtifact):
        raise ValueError("model must be a validated PLSR ModelArtifact")
    if not isinstance(sample, Mapping) or set(sample) != {"sample_schema", "x", "theta", "theta_dot"}:
        raise ValueError("sample requires exactly sample_schema, x, theta, and theta_dot")
    if sample["sample_schema"] != SAMPLE_SCHEMA:
        raise ValueError(f"sample_schema must be {SAMPLE_SCHEMA!r}")
    declaration = model.to_dict()
    plant = declaration["plant"]
    state = _vector(sample["x"], len(declaration["state"]["coordinates"]), "x")
    theta = rate = None
    if plant["kind"] == "linear":
        if sample["theta"] is not None:
            raise ValueError("theta must be null for a linear plant")
    else:
        count = len(plant["parameters"])
        theta = _vector(sample["theta"], count, "theta")
        if declaration["time"]["convention"] == "continuous":
            rate = _vector(sample["theta_dot"], count, "theta_dot")
    if rate is None and sample["theta_dot"] is not None:
        raise ValueError("theta_dot must be null for linear or discrete plants")
    return {"sample_schema": SAMPLE_SCHEMA, "x": state, "theta": theta, "theta_dot": rate}


def _diagnostics(verdict: Any) -> dict[str, Any] | None:
    sample = verdict.sample
    if sample is None:
        return None
    scalars = {key: float(getattr(sample, key)) for key in _SCALARS - {"margin", "margin_ratio"}}
    scalars.update(margin=float(verdict.margin), margin_ratio=float(verdict.margin_ratio))
    result = {key: value if math.isfinite(value) else None for key, value in scalars.items()}
    result.update({
        "nonfinite_fields": sorted(key for key, value in scalars.items() if not math.isfinite(value)),
        "value_overflow": not math.isfinite(sample.value),
        "decrease_overflow": not math.isfinite(sample.decrease),
        "value_underflow": sample.value == 0.0 and sample.scaled_value != 0.0,
        "decrease_underflow": sample.decrease == 0.0 and sample.scaled_decrease != 0.0,
        "value_out_of_range": bool(sample.value_out_of_range),
        "state_scale_exponent": int(sample.state_scale_exponent),
        "value_mantissa_exponent": list(sample.value_mantissa_exponent()),
        "decrease_mantissa_exponent": list(sample.decrease_mantissa_exponent()),
        "A": sample.A.tolist(), "P": sample.P.tolist(),
        "decrease_matrix": sample.decrease_matrix.tolist(),
        "P_rate": None if sample.P_rate is None else sample.P_rate.tolist(),
    })
    return result


def evaluate(model: ModelArtifact, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate once through PLSR and preserve its complete status semantics."""
    runtime = _runtime()
    data = validate_sample(model, sample)
    verdict = model.verdict(data["x"], theta=data["theta"], theta_dot=data["theta_dot"])
    declaration = model.to_dict()
    body = {
        "adapter_schema": EVALUATION_SCHEMA,
        "model_artifact_schema": declaration["artifact_schema"],
        "model_artifact_digest": model.artifact_digest,
        "sample": data,
        "runtime_status_schema": runtime.RUNTIME_STATUS_VERSION,
        "code": verdict.code,
        "presentation_category": _CATEGORIES[verdict.code],
        "inequality_certified": bool(verdict.inequality_certified),
        "meets_required_margin": bool(verdict.meets_required_margin),
        "operationally_acceptable": bool(verdict.operationally_acceptable),
        "required_margin": float(verdict.required_margin),
        "level": model.level,
        "details": verdict.details,
        "proof_status": "NOT_CHECKED",
        "diagnostics": _diagnostics(verdict),
    }
    # Reject any accidental non-JSON or nonfinite payload before sealing.
    _canonical(body)
    return runtime.as_companion_record(kind="ciw-plsr-evaluation", body=body)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid PLSR record: {message}")


def _matrix(value: Any, dim: int, label: str) -> None:
    _require(type(value) is list and len(value) == dim, f"{label} matrix shape")
    for row in value:
        _vector(row, dim, label)


def _validate_diagnostics(model: ModelArtifact, data: dict[str, Any], record: Mapping[str, Any]) -> None:
    diagnostics = record["diagnostics"]
    code = record["code"]
    flags = ("inequality_certified", "meets_required_margin", "operationally_acceptable")
    if code in _SAMPLELESS:
        _require(diagnostics is None, "sampleless status must have null diagnostics")
        _require(all(record[key] is False for key in flags), "sampleless status has positive flags")
        return
    _require(type(diagnostics) is dict and set(diagnostics) == _DIAGNOSTIC_KEYS,
             "unexpected diagnostic fields")
    d = diagnostics
    for key in _SCALARS:
        if d[key] is not None:
            _number(d[key], f"diagnostics.{key}")
    _require(d["nonfinite_fields"] == sorted(key for key in _SCALARS if d[key] is None),
             "nonfinite_fields must identify precisely the null scalar diagnostics")
    for key in ("scaled_value", "scaled_decrease", "state_scale"):
        _require(d[key] is not None, f"{key} must be finite")
    for key in ("value_overflow", "decrease_overflow", "value_underflow", "decrease_underflow", "value_out_of_range"):
        _require(type(d[key]) is bool, f"{key} must be boolean")
    for name in ("value", "decrease"):
        _require(d[f"{name}_overflow"] == (d[name] is None), f"{name} overflow flag")
        _require(d[f"{name}_underflow"] == (d[name] == 0 and d[f"scaled_{name}"] != 0),
                 f"{name} underflow flag")
    _require(d["value_out_of_range"] == any(d[key] for key in (
        "value_overflow", "decrease_overflow", "value_underflow", "decrease_underflow"
    )), "value_out_of_range flag")
    exponent = d["state_scale_exponent"]
    _require(type(exponent) is int and -1074 <= exponent <= 1023, "state scale exponent")
    _require(d["state_scale"] == math.ldexp(1.0, exponent), "state scale value")
    peak = max(abs(float(value)) for value in data["x"])
    _require(exponent == (math.frexp(peak)[1] - 1 if peak else 0), "scale does not bind the supplied state")
    for name in ("value", "decrease"):
        pair = d[f"{name}_mantissa_exponent"]
        _require(type(pair) is list and len(pair) == 2 and type(pair[1]) is int,
                 f"{name} mantissa/exponent representation")
        _number(pair[0], f"{name} mantissa")
        mantissa, shift = math.frexp(d[f"scaled_{name}"])
        _require(pair == [mantissa, shift + 2 * exponent], f"{name} mantissa/exponent value")
        # Match the runtime's reported reconstruction, including overflow of
        # scale squared and a zero produced by underflow. Classification uses
        # the retained scaled value, so neither case changes the runtime code.
        reconstructed = (d["state_scale"] * d["state_scale"]) * d[f"scaled_{name}"]
        expected = reconstructed if math.isfinite(reconstructed) else None
        _require(d[name] == expected, f"{name} differs from the scaled representation")
    declaration = model.to_dict()
    dim = len(declaration["state"]["coordinates"])
    for key in ("A", "P", "decrease_matrix"):
        _matrix(d[key], dim, key)
    if declaration["time"]["convention"] == "discrete":
        _require(d["P_rate"] is None, "discrete P_rate must be null")
    else:
        _matrix(d["P_rate"], dim, "P_rate")
    resolution = math.inf if d["resolution"] is None else d["resolution"]
    _require(resolution >= 0, "negative resolution")
    # A null margin/eigenvalue cannot support a positive assertion. The pinned
    # engine's ordinary finite path preserves these exact algebraic relations.
    margin = d["margin"]
    if margin is not None and d["max_decrease"] is not None:
        _require(margin == -d["max_decrease"], "margin differs from -max_decrease")
    else:
        _require(margin is None and d["max_decrease"] is None, "margin/eigenvalue null mismatch")
    finite_margin = margin if margin is not None else -math.inf
    if margin is not None:
        ratio = margin / resolution if resolution > 0 else (math.inf if margin > 0 else 0.0)
        expected_ratio = ratio if math.isfinite(ratio) else None
        _require(d["margin_ratio"] == expected_ratio, "margin_ratio contradicts margin and resolution")
    positive_p = d["min_P"] is not None and d["min_P"] > 0
    _require(record["inequality_certified"] == (positive_p and finite_margin > resolution),
             "inequality flag contradicts diagnostics")
    _require(record["meets_required_margin"] == (finite_margin > max(model.required_margin, resolution)),
             "margin flag contradicts diagnostics")
    if code == "CERTIFIED_WITH_MARGIN":
        _require(record["inequality_certified"] and record["meets_required_margin"], "certified status flags")
    elif code == "MARGIN_LOW":
        _require(record["inequality_certified"] and not record["meets_required_margin"], "low-margin status flags")
    elif code in {"NUMERICAL_INCONCLUSIVE", "NOT_CERTIFIED", "DECREASE_NOT_DEFINITE", "CERTIFICATE_NOT_POSITIVE"}:
        _require(record["inequality_certified"] is False, "noncertifying status has certified inequality")
    if code == "NUMERICAL_INCONCLUSIVE" and d["max_decrease"] is not None:
        _require(abs(d["max_decrease"]) <= resolution, "inconclusive eigenvalue exceeds resolution")
    if code == "DECREASE_NOT_DEFINITE":
        _require(d["max_decrease"] is not None and d["max_decrease"] > resolution,
                 "indefinite decrease must exceed resolution")
    if code == "NOT_CERTIFIED":
        _require(d["scaled_decrease"] > 0, "sample violation requires positive scalar decrease")
    if code == "CERTIFICATE_NOT_POSITIVE":
        _require(not positive_p or d["scaled_value"] < 0, "positive certificate labelled nonpositive")
    if code == "OUTSIDE_LEVEL_SET":
        _require(model.level is not None, "outside-level status with no declared level")
    if model.level is not None and code != "CERTIFICATE_NOT_POSITIVE":
        squared = d["state_scale"] * d["state_scale"]
        if model.level < 0:
            outside_level = True
        elif model.level == 0 or not math.isfinite(squared):
            outside_level = d["scaled_value"] > 0
        elif squared == 0:
            outside_level = False
        else:
            outside_level = d["scaled_value"] > model.level / squared
        _require((code == "OUTSIDE_LEVEL_SET") == outside_level,
                 "level-set status contradicts declared level and scaled value")


def validate_record(model: ModelArtifact, sample: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Verify structure, bindings, restrictions and integrity without reevaluating.

    This does not authenticate an issuer or prove the recorded arithmetic.
    Replay is a separate operation that invokes the pinned engine again.
    """
    runtime = _runtime()
    data = validate_sample(model, sample)
    _require(isinstance(record, Mapping) and set(record) == _RECORD_KEYS, "unexpected record fields")
    try:
        _canonical(record)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid PLSR record: only finite JSON data is accepted") from exc
    declaration = model.to_dict()
    required = {
        "record_schema": runtime.COMPANION_RECORD_VERSION,
        "record_kind": "ciw-plsr-evaluation",
        "claim_scope": "computational-integrity-only",
        "adapter_schema": EVALUATION_SCHEMA,
        "model_artifact_schema": declaration["artifact_schema"],
        "model_artifact_digest": model.artifact_digest,
        "runtime_status_schema": runtime.RUNTIME_STATUS_VERSION,
        "proof_status": "NOT_CHECKED",
    }
    for key, expected in required.items():
        _require(record[key] == expected, f"{key} does not match this instrument and model")
    _require(record["may_authorize"] is False and record["confirmed_out_of_development"] is False,
             "development and authorization restrictions")
    _require(record["maturity"] == dict(runtime.MATURITY), "maturity restrictions")
    _require(_canonical(record["sample"]) == _canonical(data), "sample binding differs")
    _number(record["required_margin"], "required_margin")
    _require(record["required_margin"] == model.required_margin, "required margin differs")
    if record["level"] is not None:
        _number(record["level"], "level")
    _require(record["level"] == model.level, "declared level differs")
    _require(type(record["details"]) is str and bool(record["details"]), "details must be a nonempty string")
    code = record["code"]
    _require(type(code) is str and code in _CATEGORIES, "unknown or host-owned runtime code")
    _require(record["presentation_category"] == _CATEGORIES[code], "presentation category differs from code")
    for key in ("inequality_certified", "meets_required_margin", "operationally_acceptable"):
        _require(type(record[key]) is bool, f"{key} must be boolean")
    _require(record["operationally_acceptable"] == (code == "CERTIFIED_WITH_MARGIN"), "operational flag differs from code")
    _validate_diagnostics(model, data, record)
    plant = declaration["plant"]
    outside = False
    if plant["kind"] == "affine":
        for field, box in (("theta", plant["parameter_box"]), ("theta_dot", plant["rate_box"])):
            if box is not None:
                outside |= any(value < low or value > high for value, low, high in
                               zip(data[field], box["lower"], box["upper"], strict=True))
    _require((code == "OUTSIDE_PARAMETER_BOX") == outside, "parameter-domain status differs from input")
    _require(type(record["record_digest"]) is str and runtime.verify_record(record), "record digest does not match")
