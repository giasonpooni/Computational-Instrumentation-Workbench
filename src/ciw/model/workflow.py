"""Retained model runs: exact bytes, execution provenance, derived views and replay.

``ciw.model-run.v1`` links the four products of an experiment: scientific
outputs (the decoded view of committed output bytes), numerical diagnostics
(solver statistics and the independent reference comparison), physical
measurements (explicitly not acquired for these simulation-only operations)
and execution provenance (SCR identities, worker session and runtime).

Creating a run executes the bound Julia worker. Inspection needs no Julia: it
recomputes every SCR identity from the retained bytes and checks that the
derived view equals the decoded output. Replay is an explicit fresh
occurrence with new execution and result identities; it requires a worker
whose runtime digest matches the retained program bytes.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import uuid

from ..core.identities import content_identity
from . import codec
from .commitments import Specification, computation_identity, output_identity
from .latex import BINDING_SCHEMA
from .providers import committed_bytes, reference_and_compare, validate_operation_request
from .spec import SpecificationError, validate_spec
from .worker import DESCRIPTORS, OPERATION_PROFILE, JuliaWorker, WorkerError, program_bytes

RUN_SCHEMA = "ciw.model-run.v1"
MAX_BYTES = 96 * 1024 * 1024
OPERATIONS = tuple(DESCRIPTORS)
NOT_ACQUIRED = {"status": "not_acquired",
                "reason": "simulation-only model operation; no sensor observation was made"}
_B64 = lambda raw: None if raw is None else base64.b64encode(raw).decode("ascii")  # noqa: E731


def _unb64(value, name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid base64") from exc


def _seal(bundle: dict) -> dict:
    bundle["record_digest"] = content_identity({k: v for k, v in bundle.items() if k != "record_digest"})
    return bundle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_run(operation: str, spec: dict, request: dict, worker: JuliaWorker, *,
                replay_of: dict | None = None, timeout: float | None = None) -> dict:
    """Validate, commit bytes, execute through the worker and retain everything."""
    if operation not in OPERATIONS:
        raise SpecificationError(f"Unsupported model operation: {operation}")
    model = validate_spec(spec)
    validate_operation_request(operation, model, request)
    configuration, payload = committed_bytes(operation, model, request)
    execution = {"execution_id": "execution-" + uuid.uuid4().hex, "status": "refused", "refusal": None,
                 "origin": "simulated", "created_at": _now(), "worker": None}
    scr = {"configuration_b64": _B64(configuration), "input_b64": _B64(payload), "program_b64": None,
           "specification_identity": None, "program_identity": None, "input_identity": None,
           "output_b64": None, "output_identity": None, "computation_identity": None,
           "exit_code": None, "detail": None}
    result, diagnostics, runtime = None, None, None
    try:
        worker.start()
        runtime = worker.runtime_record()
        outcome = worker.execute(operation, configuration, payload,
                                 **({"timeout": timeout} if timeout is not None else {}))
    except WorkerError as exc:
        execution["status"] = "refused" if exc.code in {"execution_refused", "unsupported_operation",
                                                         "runtime_mismatch", "runtime_unavailable",
                                                         "oversized_request"} else "failed"
        execution["refusal"] = {"code": exc.code, "message": str(exc)[:4000]}
        if runtime is None and worker.runtime_digest is not None:
            runtime = worker.runtime_record()
        if runtime is not None and runtime.get("runtime_digest"):
            spec_triple = Specification(program_bytes(operation, runtime["runtime_digest"]), configuration, payload)
            scr.update(program_b64=_B64(spec_triple.program), specification_identity=spec_triple.identity(),
                       program_identity=spec_triple.program_identity(), input_identity=spec_triple.input_identity())
    else:
        scr.update(program_b64=_B64(outcome.specification.program),
                   specification_identity=outcome.specification_identity,
                   program_identity=outcome.program_identity, input_identity=outcome.input_identity,
                   output_b64=_B64(outcome.output), output_identity=outcome.output_identity,
                   computation_identity=outcome.computation_identity, exit_code=outcome.exit_code,
                   detail=outcome.detail)
        execution["worker"] = {"session_id": outcome.session_id, "occurrence": outcome.occurrence,
                               "elapsed_ns": outcome.elapsed_ns, "wall_time_s": outcome.wall_time_s,
                               "startup_s": worker.startup_s}
        execution["status"] = outcome.status
        if outcome.status == "completed":
            view = codec.json_view(codec.decode(outcome.output))
            diagnostics = reference_and_compare(operation, model, request, view)
            result = {"result_id": "result-" + uuid.uuid4().hex, "view": view,
                      "representation": "json view derived from committed CIWB output bytes"}
    bundle = {
        "schema": RUN_SCHEMA, "run_id": "model-run-" + uuid.uuid4().hex, "created_at": _now(),
        "operation": operation, "specification": deepcopy(spec), "spec_digest": model.digest,
        "request": deepcopy(request), "request_digest": content_identity(request),
        "execution": execution, "scr": scr, "runtime": runtime, "result": result,
        "diagnostics": diagnostics, "physical_measurements": dict(NOT_ACQUIRED),
        "verification": {"status": "not_verified",
                         "note": "Reference comparison and identity checks are not a verification claim"},
        "replay_of": replay_of,
    }
    return _seal(bundle)


def inspect_run(bundle: dict) -> dict:
    """Offline inspection: recompute identities and derived views; execute nothing."""
    if not isinstance(bundle, dict) or bundle.get("schema") != RUN_SCHEMA:
        raise ValueError("Not a ciw.model-run.v1 record")
    if bundle.get("record_digest") != content_identity({k: v for k, v in bundle.items() if k != "record_digest"}):
        raise ValueError("Model run record integrity mismatch")
    if not re.fullmatch(r"model-run-[0-9a-f]{32}", str(bundle.get("run_id"))):
        raise ValueError("Invalid model run identity")
    operation = bundle["operation"]
    if operation not in OPERATIONS:
        raise ValueError("Unsupported retained operation")
    model = validate_spec(bundle["specification"])
    if model.digest != bundle["spec_digest"]:
        raise ValueError("Retained specification digest mismatch")
    validate_operation_request(operation, model, bundle["request"])
    if content_identity(bundle["request"]) != bundle["request_digest"]:
        raise ValueError("Retained request digest mismatch")
    configuration, payload = committed_bytes(operation, model, bundle["request"])
    scr, execution = bundle["scr"], bundle["execution"]
    if _unb64(scr["configuration_b64"], "configuration") != configuration or _unb64(scr["input_b64"], "input") != payload:
        raise ValueError("Retained configuration or input bytes do not re-derive from the specification and request")
    status = execution["status"]
    if status not in {"completed", "halted", "refused", "failed"}:
        raise ValueError("Unsupported execution status")
    if not re.fullmatch(r"execution-[0-9a-f]{32}", str(execution.get("execution_id"))):
        raise ValueError("Invalid execution identity")
    checks = {"record_digest": True, "configuration_and_input_rederived": True}
    program = _unb64(scr["program_b64"], "program")
    if program is not None:
        runtime = bundle["runtime"]
        if runtime is None or program != program_bytes(operation, runtime["runtime_digest"]):
            raise ValueError("Retained program bytes do not name the retained runtime")
        triple = Specification(program, configuration, payload)
        for key, value in (("specification_identity", triple.identity()), ("program_identity", triple.program_identity()),
                           ("input_identity", triple.input_identity())):
            if scr[key] != value:
                raise ValueError(f"Retained {key} does not match its bytes")
        checks["scr_identities"] = True
    elif status in {"completed", "halted"}:
        raise ValueError("An executed run must retain its program bytes")
    output = _unb64(scr["output_b64"], "output")
    if status == "completed":
        if output is None or scr["output_identity"] != output_identity(output):
            raise ValueError("Completed run output identity mismatch")
        if scr["computation_identity"] != computation_identity(scr["program_identity"], scr["input_identity"],
                                                               scr["output_identity"], scr["exit_code"]):
            raise ValueError("Completed run computation identity mismatch")
        result = bundle["result"]
        if (result is None or not re.fullmatch(r"result-[0-9a-f]{32}", str(result.get("result_id")))
                or result["view"] != codec.json_view(codec.decode(output))):
            raise ValueError("Derived result view does not match the committed output bytes")
        checks["output_identity"] = checks["derived_view"] = True
    else:
        if output is not None or scr["output_identity"] is not None or scr["computation_identity"] is not None \
                or bundle["result"] is not None:
            raise ValueError("A run without completion carries no output, computation identity or result")
    if bundle["physical_measurements"] != NOT_ACQUIRED:
        raise ValueError("Simulation-only runs must mark physical measurements as not acquired")
    diagnostics = bundle["diagnostics"]
    return {"run_id": bundle["run_id"], "operation": operation, "status": status,
            "spec_digest": bundle["spec_digest"], "model_id": model.spec["model_id"],
            "execution_id": execution["execution_id"],
            "result_id": bundle["result"]["result_id"] if bundle["result"] else None,
            "specification_identity": scr["specification_identity"],
            "computation_identity": scr["computation_identity"],
            "runtime_digest": bundle["runtime"]["runtime_digest"] if bundle["runtime"] else None,
            "worker": execution["worker"], "refusal": execution["refusal"], "detail": scr["detail"],
            "reference_passed": diagnostics.get("passed") if diagnostics else None,
            "physical_measurements": bundle["physical_measurements"]["status"],
            "replay_of": bundle["replay_of"], "offline_checks": checks}


def replay_run(bundle: dict, worker: JuliaWorker) -> dict:
    """Re-execute the retained request as a new occurrence; the original stays retained."""
    original = inspect_run(bundle)
    if bundle["runtime"] is None:
        raise ValueError("The original attempt was not bound to a runtime; nothing can be replayed")
    worker.start()
    if worker.runtime_digest != bundle["runtime"]["runtime_digest"]:
        raise WorkerError("runtime_mismatch", "Replay requires the retained runtime digest; "
                                              "a different environment is a different program")
    replay = execute_run(bundle["operation"], bundle["specification"], bundle["request"], worker,
                         replay_of={"run_id": bundle["run_id"], "execution_id": original["execution_id"],
                                    "result_id": original["result_id"],
                                    "output_identity": bundle["scr"]["output_identity"]})
    comparison = {"specification_identity_equal": replay["scr"]["specification_identity"] == bundle["scr"]["specification_identity"],
                  "output_bytes_identical": (replay["scr"]["output_identity"] is not None and
                                             replay["scr"]["output_identity"] == bundle["scr"]["output_identity"]),
                  "fresh_execution_identity": replay["execution"]["execution_id"] != original["execution_id"]}
    if bundle["result"] and replay["result"]:
        comparison["max_abs_difference"] = _max_difference(bundle["result"]["view"], replay["result"]["view"])
    replay["replay_of"]["comparison"] = comparison
    return _seal(replay)


def _max_difference(a, b) -> float | None:
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        values = [_max_difference(a[k], b[k]) for k in a]
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        values = [_max_difference(x, y) for x, y in zip(a, b)]
    elif type(a) in (int, float) and type(b) in (int, float):
        return abs(float(a) - float(b))
    else:
        return None if a != b else 0.0
    if any(value is None for value in values):
        return None
    return max(values, default=0.0)


def binding_for_run(bundle: dict) -> dict:
    """Link symbols to a completed run: declared parameters and requested initial states."""
    inspect_run(bundle)
    if bundle["execution"]["status"] != "completed":
        raise ValueError("Only a completed run can bind a LaTeX view")
    model = validate_spec(bundle["specification"])
    uncertainty = model.spec["uncertainty"]
    evidence = [bundle["spec_digest"], bundle["request_digest"]]

    def sigma(block, symbol):
        if block is None or symbol not in block["order"]:
            return None
        index = block["order"].index(symbol)
        return math.sqrt(block["matrix"][index][index])

    estimates = {}
    for entry in model.spec["parameters"]:
        estimates[entry["symbol"]] = {"value": entry["value"], "unit": entry["unit"],
                                      "standard_uncertainty": sigma(uncertainty["parameters"], entry["symbol"]),
                                      "source": "declared_in_specification", "evidence_ids": [bundle["spec_digest"]]}
    request = bundle["request"]
    initial = request.get("initial_state") if "initial_state" in request else request.get("state")
    source = "requested_initial_condition" if "initial_state" in request else "requested_operating_point"
    for symbol, value in zip(model.state_order, initial or []):
        estimates[symbol] = {"value": value, "unit": model.entries[symbol]["unit"],
                             "standard_uncertainty": sigma(uncertainty["initial_state"], symbol),
                             "source": source, "evidence_ids": evidence}
    return {"schema": BINDING_SCHEMA, "spec_digest": bundle["spec_digest"],
            "result_id": bundle["result"]["result_id"], "execution_id": bundle["execution"]["execution_id"],
            "estimates": estimates}


def save_run(bundle: dict, output_dir: Path) -> Path:
    inspect_run(bundle)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{bundle['run_id']}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def read_run(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Model run record exceeds its byte bound")

    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(raw, parse_constant=reject)


__all__ = ["OPERATION_PROFILE", "execute_run", "inspect_run", "replay_run", "binding_for_run", "save_run", "read_run"]
