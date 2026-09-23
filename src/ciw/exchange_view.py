"""Read-only shared-session projection for a retained instrument exchange."""
from __future__ import annotations

from copy import deepcopy


def project(record, source, declaration, revision):
    native = record["native"]
    step = native["steps"][0]
    result = step["result"]
    return {
        "schema": "ciw.instrument-exchange-view.v1",
        "bundle_id": record["bundle_id"],
        "source_id": record["source_id"],
        "operation_id": step["operation_id"],
        "execution_id": step["execution_id"],
        "result_id": step["result_id"],
        "numerical_result_id": step["numerical_result_id"],
        "producer_artifact_ids": deepcopy(result["producer_artifact_ids"]),
        "artifacts": deepcopy(result["artifacts"]),
        "links": deepcopy(result["links"]),
        "covariance_validation": deepcopy(result["covariance_validation"]),
        "validator": deepcopy(result["validator"]),
        "authority": deepcopy(native["authority"]),
        "verification": deepcopy(native["verification"]),
        "replay": "available_with_pinned_set_binding",
        "workbench_revision": revision,
    }
