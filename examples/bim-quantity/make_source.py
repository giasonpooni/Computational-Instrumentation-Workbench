"""Build exact-byte BIM source evidence from the adjacent declared IFC fixture."""
import base64
import json
from pathlib import Path

from ciw.bim_quantity import POLICY
from ciw.core.canonical import byte_digest, canonical


def source():
    raw = Path(__file__).with_name("room.ifc").read_bytes()
    target = {"ifc_class": "IfcBuildingStorey", "global_id": "CIWSTOREY00000000000015", "quantity": "ClearHeight"}
    observation = {"schema": "ciw.bim-scalar-observation.v1", "value": 2.99, "variance": 0.000025,
                   "unit": "m", "frame": "declared-room-survey-frame", "ifc_sha256": byte_digest(raw),
                   **target, "cross_covariance_policy": "independent"}
    return {"schema": "ciw.bim-quantity-source.v1", "experiment_id": "declared-room-quantity",
            "configuration": POLICY, "target": target, "model_frame": observation["frame"],
            "ifc_bytes_b64": base64.b64encode(raw).decode(),
            "observation_bytes_b64": base64.b64encode(canonical(observation)).decode()}


if __name__ == "__main__":
    print(json.dumps(source(), indent=2, sort_keys=True))
