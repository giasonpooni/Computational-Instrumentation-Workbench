"""Explicit retained PPDA records into the native calibrated-window workflow.

Acquisition is not repeated. Exact snapshot bytes, selected evidence identities,
the mapping declaration and the unchanged numerical child remain inspectable.
This operation does not connect a previous posterior to a new window prior.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import re
import uuid

from . import acquired_dataset as acquisition
from . import calibrated_window as window
from .adapters.subprocess import _json
from .exchange import _identity
from .core.canonical import canonical, digest, byte_digest, bundle_digest, exact_keys

MAX_BYTES = 4 * 1024 * 1024
SOURCE_LIMIT = 262144
SOURCE_SCHEMA = "ciw.acquired-calibrated-window-source.v1"
ROW_SCHEMA = "ciw.acquired-window-sample.v1"
SCHEMA = "ciw.acquired-calibrated-window-session.v1"
VERIFICATION_SCHEMA = "ciw.acquired-window-verification.v1"
OPERATION = "ciw.acquired-calibrated-window.v1"
ROLES = window.ROLES
POLICY = {
    "selection": "explicit_retained_ppda_observation_record_and_snapshot",
    "mapping": "typed_scalar_row_without_value_or_timestamp_overrides",
    "acquisition_reexecution": "not_performed",
    "event_time_order": "declared_clock_map_only",
    "window_prior": "explicit_independent_reference_not_previous_posterior",
    "inter_window_cross_covariance": "not_inferred",
    "state_admission": "not_performed",
}
CHECKS = ["exact_selected_acquisition_binding", "exact_snapshot_row_lineage",
          "deterministic_typed_mapping", "unchanged_calibrated_window_child",
          "child_SET_receipt_content_binding"]
DECLARATION_FIELDS = {"experiment_id", "epoch", "channel_id", "frame", "frame_mapping",
                      "device_clock", "receipt_clock", "clock_model", "calibration_profile",
                      "joint_covariance", "configuration"}
SAMPLE_FIELDS = {"sensor_id", "quantity_id", "unit", "indicated_value", "raw_value",
                 "device_time", "received_at"}
ROW_FIELDS = {"schema", "sequence", "window_id", "channel_id", "epoch", "frame", "device_clock",
              "receipt_clock", "clock_mapping_ref", "calibration_ref", "cross_covariance_policy",
              "uncertainty_evidence_ids", "sample"}


def _source(raw):
    if not isinstance(raw, bytes) or len(raw) > SOURCE_LIMIT:
        raise ValueError("Acquired window mapping source exceeds 256 KiB")
    value = _json(raw)
    canonical(value)
    exact_keys(value, {"schema", "experiment_id", "selections", "declaration", "configuration"})
    if value["schema"] != SOURCE_SCHEMA or value["configuration"] != POLICY:
        raise ValueError("Require the explicit retained-record mapping policy")
    window._text(value["experiment_id"])
    exact_keys(value["declaration"], DECLARATION_FIELDS)
    if value["declaration"]["experiment_id"] != value["experiment_id"]:
        raise ValueError("Mapping and window experiment identities differ")
    exact_keys(value["declaration"]["clock_model"], {"model_id", "source_frame", "reference_frame", "device_origin", "reference_origin",
          "skew", "offset", "valid_device_interval", "synchronization_evidence_ids"})
    exact_keys(value["declaration"]["calibration_profile"], {"profile_id", "artifact_id", "sensor_id", "quantity_id", "input_unit", "output_unit",
          "gain", "offset", "coefficient_covariance", "valid_from", "valid_until", "reference_ids", "input_range"})
    exact_keys(value["declaration"]["joint_covariance"], {"order", "matrix", "cross_covariance_policy", "evidence_ids"})
    selections = value["selections"]
    if not isinstance(selections, list) or not 1 <= len(selections) <= 16:
        raise ValueError("Select 1..16 explicit scalar observations")
    for selection in selections:
        exact_keys(selection, {"observation_id", "record_id", "document_id", "snapshot_index", "row_index"})
        for key in ("observation_id", "record_id", "document_id"):
            if not isinstance(selection[key], str) or not re.fullmatch(r"[a-f0-9]{64}", selection[key]):
                raise ValueError("Selections require native PPDA content identities")
        for key, maximum in (("snapshot_index", 7), ("row_index", 63)):
            if type(selection[key]) is not int or not 0 <= selection[key] <= maximum:
                raise ValueError("Snapshot and row indices must be bounded nonnegative integers")
    for key in ("observation_id", "record_id", "document_id"):
        if len({s[key] for s in selections}) != len(selections):
            raise ValueError("Duplicate selected acquisition evidence")
    return value


def _derive(source, upstream):
    raw = acquisition._validate(upstream)
    retained = acquisition._source(raw)
    step, = upstream["steps"]
    evidence = step["result"]["data"]["evidence"]
    observations = {item["id"]: item for item in evidence["observations"]}
    records = {item["id"]: item for item in evidence["records"]}
    documents = {item["id"]: item for item in evidence["documents"]}
    declaration = source["declaration"]
    samples, selected = [], []
    for selector in source["selections"]:
        try:
            observation = observations[selector["observation_id"]]
            record = records[selector["record_id"]]
            document = documents[selector["document_id"]]
            snapshot = retained["snapshots"][selector["snapshot_index"]]
            snapshot_raw = base64.b64decode(snapshot["bytes_b64"], validate=True)
            row = _json(snapshot_raw)[selector["row_index"]]
        except (KeyError, IndexError) as exc:
            raise ValueError("Selected acquisition evidence or snapshot row is absent") from exc
        if (observation["record_ids"] != [record["id"]] or record["document_id"] != document["id"] or
                canonical(observation["content"]) != canonical(row) or
                document["retrieved_at"] != snapshot["requested_at"]):
            raise ValueError("Selected observation/record/document/first-snapshot lineage differs")
        exact_keys(row, ROW_FIELDS)
        exact_keys(row["sample"], SAMPLE_FIELDS)
        if row["schema"] != ROW_SCHEMA or row["window_id"] != source["experiment_id"]:
            raise ValueError("Selected record is not the declared typed window sample")
        expected = {"channel_id": declaration["channel_id"], "epoch": declaration["epoch"],
                    "frame": declaration["frame"], "device_clock": declaration["device_clock"],
                    "receipt_clock": declaration["receipt_clock"],
                    "clock_mapping_ref": declaration["clock_model"]["model_id"],
                    "calibration_ref": declaration["calibration_profile"]["artifact_id"],
                    "cross_covariance_policy": declaration["joint_covariance"]["cross_covariance_policy"],
                    "uncertainty_evidence_ids": declaration["joint_covariance"]["evidence_ids"]}
        if any(canonical(row[key]) != canonical(value) for key, value in expected.items()):
            raise ValueError("Acquired channel/frame/clock/calibration/uncertainty declaration differs")
        sample = dict(deepcopy(row["sample"]), observation_id="ppda-observation:" + observation["id"],
                      artifact_id="ppda-record:" + record["id"])
        samples.append(sample)
        selected.append(dict(deepcopy(selector), sequence=row["sequence"],
                             snapshot_sha256=byte_digest(snapshot_raw),
                             source_id=document["source_id"],
                             mapped_observation_id=sample["observation_id"],
                             mapped_artifact_id=sample["artifact_id"]))
    mapped = dict(deepcopy(declaration), schema=window.SOURCE_SCHEMA, samples=samples)
    derived_raw = canonical(mapped)
    window._source(derived_raw)
    binding = {"bundle_id": upstream["bundle_digest"], "result_id": step["result_id"],
               "numerical_result_id": step["numerical_result_id"], "evidence_id": byte_digest(raw),
               "selected_records": selected}
    return derived_raw, binding


def validate_upstream(bundle, upstream):
    if canonical(bundle["upstream_acquisition"]) != canonical(upstream):
        raise ValueError("Acquisition upstream differs from the selected retained bundle")
    raw = base64.b64decode(bundle["source"]["evidence"][0]["bytes_b64"], validate=True)
    derived, binding = _derive(_source(raw), upstream)
    if canonical(binding) != canonical(bundle["acquisition_binding"]):
        raise ValueError("Acquisition result/evidence selection binding differs")
    return window._source(derived)


def _verification(bundle, child=None, child_verification=None):
    child = child or bundle["child_window"]
    child_verification = child_verification or child["verification"]
    value = {"schema": VERIFICATION_SCHEMA, "subject_ref": bundle["bundle_digest"],
             "outcome": "passed", "independent": False, "checks": deepcopy(CHECKS),
             "scope": "mapping_integrity_and_referenced_child_numerical_verification",
             "acquisition_binding_digest": digest(bundle["acquisition_binding"]),
             "mapping_source_id": bundle["source"]["evidence"][0]["artifact_ref"],
             "derived_source_id": child["source"]["evidence"][0]["artifact_ref"],
             "child_bundle_id": child["bundle_digest"],
             "child_verification_id": child_verification["verification_id"],
             "admission": "not_performed"}
    value["verification_id"] = "sha256:" + sha256(VERIFICATION_SCHEMA.encode() + b"\0" + canonical(value)).hexdigest()
    return value


def _check_child_receipt(child, receipt=None):
    receipt = receipt if receipt is not None else child.get("verification")
    if not isinstance(receipt, dict):
        raise ValueError("Require a retained child SET verification receipt")
    _identity(receipt, "verification_id")
    if (receipt.get("schema") != "notation.instrument.verification-artifact.v1" or
            receipt.get("verifier_ref") != "set:replay-binding.v1" or
            receipt.get("subject_ref") != child["bundle_digest"] or receipt.get("outcome") != "passed" or receipt.get("independent") is not False):
        raise ValueError("Child SET receipt must bind the exact numerical child")
    expected = {"bundle_digest": child["bundle_digest"],
                "evidence_digests": {e["artifact_ref"]: e["sha256"] for e in child["source"]["evidence"]},
                "operation_pins": [{"operation_id": s["operation_id"], "runtime_ref": s["runtime_ref"],
                                    "runtime_digest": digest(child["runtimes"][s["runtime_ref"]]),
                                    "request_digest": s["request_sha256"]} for s in child["steps"]],
                "numerical_result_ids": {s["execution_id"]: s["numerical_result_id"] for s in child["steps"]},
                "configuration_digest": digest(child["configuration"]),
                "execution_ids": sorted(s["execution_id"] for s in child["steps"])}
    if receipt.get("binding") != expected or not isinstance(receipt.get("checks"), list) or len(receipt["checks"]) != 2 or [
            (c.get("name"), c.get("outcome")) for c in receipt["checks"] if isinstance(c, dict)] != [
            ("retained-content-and-exchange-binding", "passed"), ("numerical-replay", "passed")]:
        raise ValueError("Child SET receipt numerical/runtime/evidence binding differs")


def _validate(bundle):
    try:
        exact_keys(bundle, {"schema", "session_id", "created_at", "source", "configuration", "runtimes", "steps",
                       "upstream_acquisition", "acquisition_binding", "child_window", "bundle_digest", "verification"},
              {"replay_receipts"})
        if (len(canonical(bundle)) > MAX_BYTES or bundle["schema"] != SCHEMA or
                bundle["bundle_digest"] != bundle_digest(bundle) or
                not isinstance(bundle["session_id"], str) or not re.fullmatch(r"session-[a-f0-9]{32}", bundle["session_id"])):
            raise ValueError("Acquired window bundle identity mismatch")
        evidence, = bundle["source"]["evidence"]
        raw = base64.b64decode(evidence["bytes_b64"], validate=True)
        source = _source(raw)
        if evidence != {"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}:
            raise ValueError("Exact mapping source bytes differ")
        if bundle["source"] != {"experiment_id": source["experiment_id"], "experiment_digest": digest(source), "evidence": [evidence]} or bundle["configuration"] != POLICY:
            raise ValueError("Acquired window source/configuration binding differs")
        derived, binding = _derive(source, bundle["upstream_acquisition"])
        if canonical(binding) != canonical(bundle["acquisition_binding"]):
            raise ValueError("Acquisition result/evidence selection binding differs")
        child = bundle["child_window"]
        if window._validate(child) != derived:
            raise ValueError("Native child source differs from the derived acquisition mapping")
        _check_child_receipt(child)
        if (canonical(bundle["steps"]) != canonical(child["steps"]) or
                canonical(bundle["runtimes"]) != canonical(child["runtimes"]) or
                bundle["created_at"] != child["created_at"] or bundle["session_id"] == child["session_id"]):
            raise ValueError("Outer native steps/runtime/time must retain the exact child")
        _identity(bundle["verification"], "verification_id")
        if canonical(bundle["verification"]) != canonical(_verification(bundle)):
            raise ValueError("Outer verification must bind mapping and exact child SET receipt")
        return raw
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed acquired calibrated window session") from exc


def mapped_source(bundle):
    """Read-only, validated numerical declaration for shared instrument views."""
    _validate(bundle)
    return window._source(base64.b64decode(bundle["child_window"]["source"]["evidence"][0]["bytes_b64"], validate=True))


def _adapters(repositories, expected=None):
    return window._adapters(repositories, expected)


def _wrap(raw, upstream, child):
    source = _source(raw)
    derived, binding = _derive(source, upstream)
    if window._validate(child) != derived:
        raise ValueError("Native child does not consume the mapped acquisition bytes")
    bundle = {"schema": SCHEMA, "session_id": "session-" + uuid.uuid4().hex,
              "created_at": child["created_at"],
              "source": {"experiment_id": source["experiment_id"], "experiment_digest": digest(source),
                         "evidence": [{"artifact_ref": byte_digest(raw), "sha256": byte_digest(raw), "bytes_b64": base64.b64encode(raw).decode()}]},
              "configuration": deepcopy(POLICY), "runtimes": deepcopy(child["runtimes"]),
              "steps": deepcopy(child["steps"]), "upstream_acquisition": deepcopy(upstream),
              "acquisition_binding": binding, "child_window": deepcopy(child)}
    bundle["bundle_digest"] = bundle_digest(bundle)
    bundle["verification"] = _verification(bundle)
    _validate(bundle)
    return bundle


def create_session(raw, upstream, repositories):
    source = _source(raw)
    derived, _ = _derive(source, upstream)
    child = window.create_session(derived, repositories)
    return _wrap(raw, upstream, child)


def validate_replay(original, fresh, receipt):
    if _validate(original) != _validate(fresh):
        raise ValueError("Replay changed the mapping source bytes")
    if canonical(original["upstream_acquisition"]) != canonical(fresh["upstream_acquisition"]) or original["acquisition_binding"] != fresh["acquisition_binding"]:
        raise ValueError("Replay must keep the exact selected acquisition occurrence")
    old_child, new_child = original["child_window"], fresh["child_window"]
    child_receipt, = new_child.get("replay_receipts", [])
    expected_child = {"schema": "ciw.calibrated-window-replay.v1", "source_bundle_digest": old_child["bundle_digest"],
                      "replayed_bundle_digest": new_child["bundle_digest"], "numerical_match": True,
                      "verification": child_receipt["verification"], "admission": "not_performed"}
    expected_child["replay_id"] = digest(expected_child)
    if canonical(child_receipt) != canonical(expected_child):
        raise ValueError("Child replay receipt does not bind the original and fresh numerical windows")
    child_proof = child_receipt["verification"]
    _check_child_receipt(old_child, child_proof)
    if original["session_id"] == fresh["session_id"] or old_child["session_id"] == new_child["session_id"]:
        raise ValueError("Replay needs fresh outer and child sessions")
    old_ids = {v for step in original["steps"] for v in (step["execution_id"], step["result_id"])}
    fresh_ids = {v for step in fresh["steps"] for v in (step["execution_id"], step["result_id"])}
    if old_ids & fresh_ids or [s["numerical_result_id"] for s in original["steps"]] != [s["numerical_result_id"] for s in fresh["steps"]]:
        raise ValueError("Replay must retain numerical results with fresh native occurrences")
    expected = {"schema": "ciw.acquired-calibrated-window-replay.v1", "source_bundle_digest": original["bundle_digest"],
                "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
                "verification": _verification(original, new_child, child_proof), "admission": "not_performed"}
    expected["replay_id"] = digest(expected)
    if canonical(receipt) != canonical(expected) or canonical(fresh.get("replay_receipts")) != canonical([receipt]):
        raise ValueError("Acquired window replay receipt differs from selected child replay")


def replay_session(bundle, repositories):
    raw = _validate(bundle)
    child_replay = window.replay_session(bundle["child_window"], repositories)
    fresh = _wrap(raw, bundle["upstream_acquisition"], child_replay["session"])
    receipt = {"schema": "ciw.acquired-calibrated-window-replay.v1", "source_bundle_digest": bundle["bundle_digest"],
               "replayed_bundle_digest": fresh["bundle_digest"], "numerical_match": True,
               "verification": _verification(bundle, fresh["child_window"], child_replay["replay_receipt"]["verification"]),
               "admission": "not_performed"}
    receipt["replay_id"] = digest(receipt)
    fresh["replay_receipts"] = [receipt]
    validate_replay(bundle, fresh, receipt)
    return {"session": fresh, "replay_receipt": receipt}
