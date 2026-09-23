"""Build explicitly synthetic acquisition/mapping declarations for native demos.

The complete calibrated-window templates are caller supplied. Generated receipt
times and acquisition request times are fixture declarations, not measured data.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

from ciw import acquired_dataset as acquisition
from ciw import acquired_window as bridge
from ciw import calibrated_window as window


def build_acquisition(window_templates):
    if not 1 <= len(window_templates) <= 8:
        raise ValueError("Declare 1..8 synthetic windows")
    if len({w["experiment_id"] for w in window_templates}) != len(window_templates):
        raise ValueError("Synthetic window experiment identities must differ")
    rows, snapshots = [], []
    requested = datetime.fromisoformat(window_templates[0]["epoch"].replace("Z", "+00:00"))
    for index, template in enumerate(window_templates):
        window._source(window.canonical(template))
        for sample in template["samples"]:
            rows.append({"schema": bridge.ROW_SCHEMA, "sequence": len(rows) + 1,
                         "window_id": template["experiment_id"], "channel_id": template["channel_id"],
                         "epoch": template["epoch"], "frame": deepcopy(template["frame"]),
                         "device_clock": deepcopy(template["device_clock"]), "receipt_clock": deepcopy(template["receipt_clock"]),
                         "clock_mapping_ref": template["clock_model"]["model_id"],
                         "calibration_ref": template["calibration_profile"]["artifact_id"],
                         "cross_covariance_policy": template["joint_covariance"]["cross_covariance_policy"],
                         "uncertainty_evidence_ids": deepcopy(template["joint_covariance"]["evidence_ids"]),
                         "sample": {k: deepcopy(v) for k, v in sample.items() if k not in {"observation_id", "artifact_id"}}})
        requested += timedelta(seconds=100 + index)
        raw = json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        snapshots.append({"requested_at": requested.isoformat().replace("+00:00", "Z"),
                          "bytes_b64": base64.b64encode(raw).decode()})
    result = {"schema": acquisition.SOURCE_SCHEMA, "experiment_id": "synthetic-acquired-window-sequence",
              "source": {"source_id": "synthetic-calibrated-stream", "name": "Synthetic declared scalar window snapshots", "domain": "offline_fixture"},
              "plan_id": "synthetic-calibrated-window-acquisition", "snapshots": snapshots,
              "configuration": deepcopy(acquisition.POLICY)}
    acquisition._source(acquisition.canonical(result))
    return result


def build_mapping(upstream, template, window_index):
    """Select native PPDA identities; never recover values from the template."""
    retained = acquisition._source(acquisition._validate(upstream))
    snapshot = retained["snapshots"][window_index]
    rows = json.loads(base64.b64decode(snapshot["bytes_b64"], validate=True))
    evidence = upstream["steps"][0]["result"]["data"]["evidence"]
    records = {record["id"]: record for record in evidence["records"]}
    by_sequence = {obs["content"]["sequence"]: obs for obs in evidence["observations"]}
    selections = []
    for index, row in enumerate(rows):
        if row["window_id"] != template["experiment_id"]:
            continue
        obs = by_sequence[row["sequence"]]
        record = records[obs["record_ids"][0]]
        selections.append({"observation_id": obs["id"], "record_id": record["id"], "document_id": record["document_id"],
                           "snapshot_index": window_index, "row_index": index})
    declaration = {k: deepcopy(v) for k, v in template.items() if k not in {"schema", "samples"}}
    identities = ["ppda-observation:" + selection["observation_id"] for selection in selections]
    declaration["joint_covariance"]["order"] = (["device_time:" + name for name in identities] + ["clock_skew", "clock_offset"] +
                                                ["indicated_value:" + name for name in identities] + ["calibration_gain", "calibration_offset"])
    result = {"schema": bridge.SOURCE_SCHEMA, "experiment_id": template["experiment_id"], "selections": selections,
              "declaration": declaration, "configuration": deepcopy(bridge.POLICY)}
    bridge._derive(bridge._source(bridge.canonical(result)), upstream)
    return result


def default_template():
    return json.loads((Path(__file__).parents[1] / "calibrated-window/source.json").read_bytes())
