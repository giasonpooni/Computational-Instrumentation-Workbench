"""Explicit synthetic fixed-reference experiment; each window has the same prior.

The third window deliberately changes the input. These are constructed fixture
values, not a physical sensor record or an inferred drift cause. No posterior
from an earlier window becomes a later prior.
"""
from copy import deepcopy
import json
from pathlib import Path


def templates():
    base = json.loads((Path(__file__).resolve().parents[1] / "calibrated-window/source.json").read_bytes())
    base["clock_model"]["valid_device_interval"] = [10, 22]
    base["calibration_profile"]["coefficient_covariance"] = [[0.0001, 0], [0, 0.0004]]
    matrix = [[0.0 for _ in range(8)] for _ in range(8)]
    matrix[4][4] = matrix[5][5] = 0.04
    matrix[6][6], matrix[7][7] = 0.0001, 0.0004
    base["joint_covariance"]["matrix"] = matrix
    base["joint_covariance"]["evidence_ids"] = ["evidence:synthetic-stream-uncertainty-declaration"]
    prior = base["configuration"]["gsie"]["prior"]
    prior["mean"], prior["covariance"] = [7.0], [[0.01]]
    prior["state_id"] = "synthetic:fixed-calibrated-reference-prior"
    values = []
    for index, indication in enumerate((3.0, 3.25, 8.0)):
        source = deepcopy(base)
        source["experiment_id"] = f"experiment:synthetic-acquired-window-{index + 1}"
        start, end = 2 * index, 2 * index + 2
        source["configuration"]["window"].update(start=start, end=end, received_by=end, decision_time=end)
        source["configuration"]["gsie"]["target_time"] = end
        ids = []
        for position, sample in enumerate(source["samples"]):
            sample["observation_id"] = f"sample:synthetic-stream-{index + 1}-{position}"
            sample["artifact_id"] = f"evidence:synthetic-stream-{index + 1}-{position}"
            sample["raw_value"] = sample["indicated_value"] = indication
            sample["device_time"] = 10 + 2 * (start + position)
            sample["received_at"] = start + position
            ids.append(sample["observation_id"])
        source["joint_covariance"]["order"] = [
            *("device_time:" + identity for identity in ids), "clock_skew", "clock_offset",
            *("indicated_value:" + identity for identity in ids), "calibration_gain", "calibration_offset",
        ]
        values.append(source)
    return values


if __name__ == "__main__":
    print(json.dumps(templates(), indent=2, allow_nan=False))
