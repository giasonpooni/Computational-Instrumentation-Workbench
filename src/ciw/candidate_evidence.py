"""Operator-bound ESM bridge: fresh replay and candidate retention, never admission.

Only the prebuilt, hash-pinned ESM entry runs here. Client and saved workspace
values cannot select executable paths, source rights, retractions or stores.
ESM owns policy decisions, verification and its existing evidence object store.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .adapters.protocol import AdapterRefusal
from .adapters.subprocess import _bounded_process, _json
from .pipelines import pin_map

OPERATIONS = {"esm.inspect-candidate.v1": "inspect", "esm.capture-candidate.v1": "capture"}


def provider_binding():
    """The ESM candidate operations this module runs, for ``pipelines.check_providers``."""
    return {"esm": {"pin": {}, "operations": {"default": list(OPERATIONS)}}}
MAX_RESPONSE = 4 * 1024 * 1024


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() <= set(required) | set(optional):
        raise ValueError("Unexpected or missing candidate fields")


def _path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("ESM operator bindings require absolute paths")
    return Path(value).resolve()


def validate_response(response, action, bundle, raw, parameters, policy):
    """Check boundary bindings; do not reinterpret ESM's native record digest."""
    if not isinstance(response, dict) or response.get("canonicalAdmission") != "REFUSED":
        raise ValueError("ESM may retain candidate evidence only")
    for key in ("canonicalStateMutated", "releaseActivated", "sourceTruthClaimed"):
        if response.get(key) is not False:
            raise ValueError("ESM response exceeds candidate-only scope")
    if action == "capture":
        if response.get("state") not in {"CANDIDATE_EVIDENCE_RETAINED", "REFUSED"}:
            raise ValueError("Invalid ESM capture state")
        if response["state"] == "CANDIDATE_EVIDENCE_RETAINED":
            if not isinstance(response.get("capture"), dict) or not isinstance(response.get("inspection"), dict):
                raise ValueError("Retention requires a native capture receipt and inspection")
            if response["inspection"].get("state") != "ELIGIBLE_FOR_CANDIDATE_REVIEW":
                raise ValueError("Retention requires fresh eligible inspection")
            capture = response["capture"]
            _keys(capture, {"evidence", "receipt"})
            evidence, receipt = capture["evidence"], capture["receipt"]
            if (not isinstance(evidence, dict) or not isinstance(receipt, dict) or
                    evidence.get("evidenceId") != parameters["evidence_id"] or
                    receipt.get("evidenceId") != parameters["evidence_id"] or
                    receipt.get("receiptId") != parameters["workflow_id"] + ":receipt" or
                    evidence.get("capturedAt") != parameters["retained_at"] or
                    receipt.get("storedAt") != parameters["retained_at"] or
                    evidence.get("sourceId") != policy["capture_registration"].get("sourceId") or
                    evidence.get("sourceTruthClaimed") is not False or
                    not isinstance(evidence.get("contentDigest"), str) or
                    evidence.get("contentDigest") != receipt.get("contentDigest") or
                    evidence.get("storageKey") != receipt.get("storageKey")):
                raise ValueError("ESM capture receipt differs from the explicit request")
        elif response.get("capture") is not None:
            raise ValueError("Refused retention cannot have a successful receipt")
        inspection = response.get("inspection")
    else:
        inspection = response
    if inspection is None and action == "capture" and response["state"] == "REFUSED":
        return
    if not isinstance(inspection, dict):
        raise ValueError("Malformed native ESM inspection")
    context = policy["review_context"]
    if (inspection.get("schema") != "payload.instrument-candidate-inspection.v1" or
            inspection.get("inspectedAt") != parameters["inspected_at" if action == "inspect" else "retained_at"] or
            inspection.get("requestId") != context.get("requestId") or inspection.get("authority") != context.get("authority") or
            inspection.get("state") not in {"ELIGIBLE_FOR_CANDIDATE_REVIEW", "REFUSED"} or
            inspection.get("bundleBytesDigest") != "sha256:" + sha256(raw).hexdigest() or
            inspection.get("canonicalAdmission") != "REFUSED" or
            any(inspection.get(key) is not False for key in
                ("canonicalStateMutated", "evidenceRetained", "releaseActivated", "sourceTruthClaimed", "independentlyVerified", "retractionHistoryComplete"))):
        raise ValueError("Native ESM inspection binding or scope mismatch")
    candidate = inspection.get("candidate")
    if inspection["state"] == "ELIGIBLE_FOR_CANDIDATE_REVIEW" and not isinstance(candidate, dict):
        raise ValueError("Eligible inspection requires a candidate")
    if candidate is not None:
        steps = {step["runtime_ref"]: step for step in bundle["steps"]}
        calibrated = bundle["schema"] == "ciw.calibrated-observable-session.v1"
        reconciliation = (steps["cbsr"]["result"]["data"] if calibrated else steps["cbsr"]["result"]) if "cbsr" in steps else {"status": "not_run"}
        if (candidate.get("state") != "UNADMITTED" or candidate.get("bundleDigest") != bundle["bundle_digest"] or
                candidate.get("executionIds") != sorted(step["execution_id"] for step in bundle["steps"]) or
                candidate.get("verification", {}).get("outcome") != "passed" or
                candidate.get("verification", {}).get("independent") is not False or
                candidate.get("reconciliation", {}).get("status") != reconciliation["status"]):
            raise ValueError("ESM candidate does not bind the selected native bundle")
        if not calibrated:
            if "processAssessment" in candidate:
                raise ValueError("Telemetry cannot inherit calibrated observability or fault claims")
            return
        assessment = candidate.get("processAssessment", {})
        faults = steps["fdir"]["result"]["data"]
        expected = {"stateResultId": steps["gsie"]["result_id"], "stateId": steps["gsie"]["result"]["data"]["state_id"],
            "observabilityResultId": steps["oit"]["result_id"], "observabilityStatus": steps["oit"]["result"]["data"]["status"],
            "reconciliationResultId": steps["cbsr"]["result_id"], "faultResultId": steps["fdir"]["result_id"],
            "residualBasis": faults["residual_basis"], "detectionStatus": faults["detection"]["status"],
            "isolabilityStatus": faults["isolability"]["status"], "crossCovariancePolicy": faults["isolability"]["cross_covariance_policy"],
            "isolatedFault": faults["isolability"]["isolated_fault"]}
        if assessment != expected:
            raise ValueError("ESM changed native state/reconciliation/fault bindings")


class CandidateAdapter:
    def __init__(self, configuration):
        from .core.canonical import canonical
        self._configuration = deepcopy(configuration)
        _keys(configuration, {"node", "node_sha256", "artifact", "runtime", "review_context"},
              {"store_root", "capture_registration"})
        if ("store_root" in configuration) != ("capture_registration" in configuration):
            raise ValueError("Candidate capture requires both a store and separate derived-source registration")
        self.node = _path(configuration["node"])
        self.artifact = _path(configuration["artifact"])
        from .pipelines import provider_descriptor
        self.pin = provider_descriptor("esm")["pin"]
        runtime = configuration["runtime"]
        _keys(runtime, {"python", "pythonSha256", "helperPath", "repositories"})
        _path(runtime["python"])
        self.helper = _path(runtime["helperPath"])
        if not isinstance(runtime["repositories"], dict):
            raise ValueError("Candidate runtime map must be explicit")
        self.kind = "telemetry" if "ppda" in runtime["repositories"] else "calibrated-observable"
        expected = {role: value["revision"] for role, value in pin_map(self.kind).items()}
        if self.kind == "telemetry" and "cbsr" not in runtime["repositories"]:
            expected.pop("cbsr")
        expected["ciw"] = self.pin["replay_ciw_revision"]
        if not isinstance(runtime["repositories"], dict) or set(runtime["repositories"]) != set(expected):
            raise ValueError("Candidate replay requires exactly its native provider set plus CIW")
        for role, revision in expected.items():
            entry = runtime["repositories"][role]
            _keys(entry, {"path", "revision"})
            _path(entry["path"])
            if entry["revision"] != revision:
                raise ValueError("Candidate replay source pin mismatch")
        if self.capture_available:
            _path(configuration["store_root"])
        if len(canonical(configuration)) > 1024 * 1024:
            raise ValueError("Operator binding exceeds the byte budget")
        self._check()

    @property
    def capture_available(self):
        return "store_root" in self._configuration

    def _check(self):
        if (_hash(self.node) != self._configuration["node_sha256"] or
                _hash(self.artifact) != self.pin["artifact_sha256"] or
                _hash(self.helper) != self.pin["helper_sha256"]):
            raise AdapterRefusal("ESM_PIN_MISMATCH", "Bound ESM executable or helper changed")

    def execute(self, action, parameters, bundle):
        from .core.canonical import canonical
        if bundle["schema"] != "ciw." + self.kind + "-session.v1":
            raise ValueError("ESM binding cannot execute a different native workflow")
        self._check()
        raw = canonical(bundle)
        payload = {"action": action, "bundleBase64": base64.b64encode(raw).decode("ascii"),
                   "context": deepcopy(self._configuration["review_context"]),
                   "runtime": deepcopy(self._configuration["runtime"])}
        roles = set(bundle["runtimes"]) | {"ciw"}
        if not roles <= payload["runtime"]["repositories"].keys():
            raise AdapterRefusal("operation_unavailable", "ESM binding lacks a provider required by the retained bundle")
        payload["runtime"]["repositories"] = {role: payload["runtime"]["repositories"][role] for role in sorted(roles)}
        policy = {"review_context": deepcopy(payload["context"])}
        if action == "capture":
            if not self.capture_available:
                raise AdapterRefusal("operation_unavailable", "No operator-selected candidate retention store")
            registration = deepcopy(self._configuration["capture_registration"])
            payload.update(storeRoot=self._configuration["store_root"], captureRequest={
                "evidenceId": parameters["evidence_id"], "workflowId": parameters["workflow_id"],
                "retainedAt": parameters["retained_at"], "sourceRegistration": registration})
            policy["capture_registration"] = registration
        else:
            payload["inspectedAt"] = parameters["inspected_at"]
        # A timeout, failed readback or drift after capture may leave evidence
        # bytes in ESM. Never turn an uncertain physical outcome into success.
        status, output = _bounded_process([str(self.node), str(self.artifact)], cwd=self.artifact.parent,
                                          timeout=330, limit=MAX_RESPONSE, stdin=canonical(payload))
        self._check()
        if status != 0:
            raise AdapterRefusal("ESM_ACTION_FAILED", "No successful candidate receipt; a capture may have left evidence bytes")
        response = _json(output)
        validate_response(response, action, bundle, raw, parameters, policy)
        return {"response_bytes_b64": base64.b64encode(output).decode("ascii"),
                "operator_policy": policy, "adapter_identity": {**deepcopy(self.pin),
                    "node_sha256": self._configuration["node_sha256"]}}
