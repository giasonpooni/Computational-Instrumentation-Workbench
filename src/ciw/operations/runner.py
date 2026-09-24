"""Execution envelopes preserve refusal separately from immutable result records."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone

from ..adapters.protocol import AdapterRefusal
from ..core.records import finite_tree
from .registry import OperationRegistry, valid_operation_id
from .schemas import validate_payload, validate_role


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def seal(record: dict) -> dict:
    record["record_digest"] = digest({key: value for key, value in record.items() if key != "record_digest"})
    return record


def check_seal(record: dict) -> None:
    if not isinstance(record, dict) or record.get("record_digest") != digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    ):
        raise ValueError("Operation record integrity mismatch")


def numerical_result_id(operation_id: str, data: dict) -> str:
    """Content identity of an operation's numbers, as the native workflows derive theirs.

    ``sha256`` over CIW's canonical JSON of ``{operation_id, data}``: equal data
    from two executions share it, and it excludes every occurrence field
    (execution, result, time, selection). It is unkeyed, like the record seal.
    """
    from ..telemetry import digest as content_digest  # telemetry imports session, which imports this module
    return content_digest({"operation_id": operation_id, "data": data})


def check_numerical_result_id(result: dict) -> None:
    """A result that carries a numerical identity must carry the one its data give (older results carry none)."""
    if "numerical_result_id" in result and result["numerical_result_id"] != numerical_result_id(
            result.get("operation_id"), result.get("data")):
        raise ValueError("Saved operation numerical_result_id differs from its data")


def execute(registry: OperationRegistry, run: dict, selection: dict, recording_file: str,
            operation_id: str, parameters: dict) -> tuple[dict, dict | None]:
    parameters = copy.deepcopy(parameters)
    finite_tree(parameters, "operation parameters")
    execution = {
        "schema": "ciw.execution.v1", "execution_id": "execution-" + uuid.uuid4().hex,
        "operation_id": operation_id, "evidence_id": run["evidence_id"],
        "run_id": run["run_id"], "selection_revision": selection["revision"],
        "channel": selection["channel"], "interval_s": copy.deepcopy(selection["interval_s"]),
        "parameters": copy.deepcopy(parameters), "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": None, "status": "refused", "result_id": None,
    }
    try:
        operation = registry.get(operation_id)
        validate_role(operation_id, operation.role)
        if operation.role == "analysis":
            parameters.setdefault("channel", selection["channel"])
            parameters.setdefault("interval_s", copy.deepcopy(selection["interval_s"]))
        for key in ("channel", "interval_s"):
            if key in parameters and parameters[key] != selection[key]:
                raise AdapterRefusal("selection_mismatch", "Operation parameters contradict the captured selection")
        execution["parameters"] = copy.deepcopy(parameters)
        runtime = copy.deepcopy(operation.runtime_identity())
        if not isinstance(runtime, dict) or not runtime:
            raise AdapterRefusal("invalid_runtime_identity", "An operation must identify its numerical runtime")
        finite_tree(runtime, "runtime identity")
        execution["runtime"] = runtime
        # A provider receives detached scientific evidence, never live mutable state.
        # Cached provider objects remain provider-owned. Detach the returned
        # payload before validating or retaining it as immutable evidence.
        data = copy.deepcopy(operation.execute(copy.deepcopy(run), copy.deepcopy(parameters)))
        if not isinstance(data, dict):
            raise AdapterRefusal("invalid_adapter_output", "Operation data must be an object")
        validate_payload(operation_id, data, run, parameters, selection)
        json.dumps(data, allow_nan=False)
    except AdapterRefusal as exc:
        execution["refusal"] = exc.to_dict()
        return seal(execution), None
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        execution["refusal"] = {"code": "invalid_operation", "message": str(exc)}
        return seal(execution), None
    result = {
        "schema": "ciw.operation-result.v1", "result_id": "result-" + uuid.uuid4().hex,
        "execution_id": execution["execution_id"], "operation_id": operation_id,
        "evidence_id": run["evidence_id"], "run_id": run["run_id"],
        "selection_revision": selection["revision"], "channel": selection["channel"],
        "interval_s": copy.deepcopy(selection["interval_s"]), "created_at": execution["created_at"],
        "recording_file": recording_file, "verification_id": None,
        "verification_status": "not_verified", "role": operation.role,
        "runtime": execution["runtime"], "parameters": copy.deepcopy(parameters), "data": data,
        "numerical_result_id": numerical_result_id(operation_id, data),
    }
    execution.update(status="completed", result_id=result["result_id"])
    return seal(execution), seal(result)


def validate_execution(execution: dict, run: dict, revision: int, results: dict) -> None:
    check_seal(execution)
    if execution.get("schema") != "ciw.execution.v1":
        raise ValueError("Unsupported execution schema")
    if not re.fullmatch(r"execution-[0-9a-f]{32}", str(execution.get("execution_id"))):
        raise ValueError("Invalid execution identity")
    if execution.get("run_id") != run["run_id"] or execution.get("evidence_id") != run["evidence_id"]:
        raise ValueError("Execution source binding mismatch")
    if (type(execution.get("selection_revision")) is not int
            or not 0 <= execution["selection_revision"] <= revision):
        raise ValueError("Invalid execution selection revision")
    if not isinstance(execution.get("parameters"), dict):
        raise ValueError("Invalid execution parameters")
    finite_tree(execution, "execution")
    operation_id = execution.get("operation_id")
    if not valid_operation_id(operation_id):
        raise ValueError("Invalid execution operation identity")
    channel, interval = execution.get("channel"), execution.get("interval_s")
    if not isinstance(channel, str) or channel not in run["channels"]:
        raise ValueError("Invalid execution channel")
    if (not isinstance(interval, list) or len(interval) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
            or not 0 <= interval[0] < interval[1] <= run["metadata"]["duration_s"]
            or not any(interval[0] <= value < interval[1] for value in run["time_s"])):
        raise ValueError("Invalid execution interval")
    for key in ("channel", "interval_s"):
        if key in execution["parameters"] and execution["parameters"][key] != execution[key]:
            raise ValueError("Execution parameter/selection mismatch")
    runtime = execution.get("runtime")
    if runtime is not None and (not isinstance(runtime, dict) or not runtime):
        raise ValueError("Invalid execution runtime identity")
    try:
        timestamp = datetime.fromisoformat(execution["created_at"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid execution timestamp") from exc
    if execution.get("status") == "completed":
        if runtime is None:
            raise ValueError("Completed execution has no runtime identity")
        result = results.get(execution.get("result_id"))
        if result is None or result.get("schema") != "ciw.operation-result.v1":
            raise ValueError("Completed execution has no matching operation result")
        for key in ("execution_id", "operation_id", "runtime", "parameters", "created_at", "selection_revision", "channel", "interval_s"):
            if result.get(key) != execution.get(key):
                raise ValueError(f"Execution/result {key} binding mismatch")
        if "refusal" in execution:
            raise ValueError("Completed execution cannot carry refusal")
    elif execution.get("status") == "refused":
        refusal = execution.get("refusal")
        if (execution.get("result_id") is not None or not isinstance(refusal, dict)
                or not {"code", "message"} <= refusal.keys() <= {"code", "message", "reason_code"}
                or not all(isinstance(value, str) and value.strip() for value in refusal.values())):
            raise ValueError("Refused execution must have a reason and no result")
    else:
        raise ValueError("Unsupported execution status")
