"""Author a synthetic certificate declaration for an exact retained design.

This helper authors new evidence. The workflow itself only verifies received
artifact digests, and never seals or synthesizes a caller's certificate.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from ciw import identified_stability as stability


def make_source(upstream, *, certificate=None, margin=0.1, level=None):
    binding = stability._binding(upstream)
    count = len(binding["mean"])
    p = deepcopy(certificate) if certificate is not None else [[float(i == j) for j in range(count)] for i in range(count)]
    model = {
        "artifact_schema": "model-artifact-v1", "model_id": "synthetic:retained-discrete-stability",
        "model_version": "1",
        "state": {"definition": "Retained synthetic GSIE prediction in declared coordinates; model zero equilibrium only.",
                  "coordinates": [{"name": n, "unit": u} for n, u in zip(binding["state_names"], binding["state_units"])]},
        "time": {"convention": "discrete", "sample_period_s": binding["sample_interval"]},
        "plant": {"kind": "linear", "A": deepcopy(binding["A"])},
        "certificate": {"kind": "quadratic", "P": p},
        "policy": {"required_margin": margin, "level": level,
                   "margin_derivation": {"method": "synthetic-demonstration-threshold", "description": "Declared fixture margin in native common-unit coordinates; no physical uncertainty allowance.",
                                         "evidence_digest": None, "quantity": "negative-largest-eigenvalue-of-decrease-matrix"},
                   "numerical_policy": "float64-decrease-v1", "runtime_status_schema": "runtime-status-v1", "claim_codes_schema": "claim-codes-v1"},
        "estimator": {"identity": "gsie:retained-conditional-prediction", "version": "1",
                      "configuration_digest": binding["prediction_request_digest"].removeprefix("sha256:"),
                      "state_compatibility": "Exact retained GSIE prediction; no coordinate or time conversion."},
        "provenance": {"producer": "ciw-synthetic-stability-example", "producer_version": "1",
                       "model_data_digest": binding["model_numerical_result_id"].removeprefix("sha256:"),
                       "construction_report_digest": stability.digest({"fixture": "caller-supplied-certificate", "P": p}).removeprefix("sha256:")},
        "claim_scope": "computational-integrity-only", "may_authorize": False,
    }
    model["artifact_digest"] = stability._native_digest(model, {"artifact_digest"})
    unit = binding["state_units"][0]
    source = {"schema": stability.SOURCE_SCHEMA, "experiment_id": "synthetic:identified-stability",
              "configuration": deepcopy(stability.POLICY),
              "selection": {k: binding[k] for k in stability.SELECTION_FIELDS},
              "equilibrium": {"coordinates": binding["state_names"], "units": binding["state_units"],
                              "frame_id": binding["frame_id"], "value": [0.0] * count},
              "certificate_unit": "1/(" + unit + "*" + unit + ")", "model_artifact": model}
    stability._source(stability.canonical(source))
    return source


if __name__ == "__main__":
    upstream = json.loads(Path(sys.argv[1]).read_bytes())
    Path(sys.argv[2]).write_text(json.dumps(make_source(upstream), indent=2) + "\n")
