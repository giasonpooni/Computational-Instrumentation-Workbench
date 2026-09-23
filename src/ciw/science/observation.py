"""Observation models: what an instrument actually measures, kept distinct.

An instrument can appear to contradict a model while measuring a different
geometric quantity: a camera reconstructs straight chords between fiducials,
whereas a geodesic model predicts intrinsic surface distance. Each observation
therefore declares an ``observable`` from ``vocabulary.OBSERVABLES``, and a
comparison between an observation and a prediction is refused unless both
declare the same observable, unit dimension and frame.

Bound predictors in this build: ``intrinsic_distance``, ``camera_chord``,
``image_residual`` (pinhole projection), ``tracker_position`` (frame transform)
and ``encoder_displacement`` (projection on a declared axis). ``imu_orientation``,
``filtered_state`` and ``reconstructed_geometry`` are declared observables whose
predictors are not bound; asking for them is refused, not approximated.

``explain`` evaluates every bound geometric observable for the same pair of
points and reports which ones are statistically consistent with a measured
value; it never re-labels the measurement.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._common import Refusal, covariance, finite, plain, require_keys, text, vector
from .frames import FrameRegistry
from .geometry import Surface, chord_distance, closed_form_distance, log_map
from .units import convert, require_dimension
from .vocabulary import ACQUISITION_KINDS, OBSERVABLES

BOUND = frozenset({"intrinsic_distance", "camera_chord", "image_residual", "tracker_position", "encoder_displacement"})
CONSISTENCY_SIGMA = 3.0


def _observable(value: Any) -> str:
    if value not in OBSERVABLES:
        raise Refusal("unknown_observable", f"Observable {value!r} is not declared", allowed=sorted(OBSERVABLES))
    if value not in BOUND:
        raise Refusal("observable_model_unbound", f"No predictor is bound for {value!r} in this build")
    return value


def intrinsic_distance(surface: Surface, u: Any, w: Any, *, winding: int | None = None, steps: int = 400) -> float:
    """Geodesic distance (closed form where declared, otherwise log-map shooting)."""
    u, w = np.asarray(u, dtype=float), np.asarray(w, dtype=float)
    if winding is None:
        reference = closed_form_distance(surface, u, w)
        if reference is not None:
            return reference
    target = w + (np.array([2 * math.pi * winding, 0.0]) if winding else 0.0)
    return float(log_map(surface, u, target, steps=steps)["length"])


def predict(observable: str, surface: Surface | None = None, *, points: tuple[Any, Any] | None = None,
            camera: dict | None = None, point_camera: Any = None, frames: FrameRegistry | None = None,
            transform: dict | None = None, axis: Any = None, winding: int | None = None, unit: str = "m") -> dict:
    """Predict an observable, returning value, unit and the model assumptions used."""
    observable = _observable(observable)
    if observable in {"intrinsic_distance", "camera_chord"}:
        if surface is None or points is None:
            raise Refusal("malformed_prediction", f"{observable} needs a surface and a pair of chart points")
        value = (intrinsic_distance(surface, *points, winding=winding) if observable == "intrinsic_distance"
                 else chord_distance(surface, *points))
        return {"observable": observable, "value": value, "unit": surface.length_unit,
                "assumptions": [f"declared {surface.kind} surface", "chart-0 coordinates",
                                "minimal geodesic" if observable == "intrinsic_distance" else "straight segment in R^3"]}
    if observable == "image_residual":
        return _project(camera, point_camera)
    if observable == "tracker_position":
        if frames is None or transform is None:
            raise Refusal("malformed_prediction", "tracker_position needs a frame registry and a transform request")
        require_keys(transform, "transform", {"point", "unit", "source", "target", "at", "clock"}, {"point_covariance"})
        mapped = frames.transform_point(transform["point"], transform["unit"], transform["source"], transform["target"],
                                        transform["at"], transform["clock"], transform.get("point_covariance"))
        return {"observable": observable, "value": mapped["point"], "unit": transform["unit"],
                "frame": transform["target"], "covariance": mapped["covariance"], "chain": mapped["chain"],
                "assumptions": ["rigid calibrated transform chain", "first-order covariance propagation"]}
    direction = vector(axis, "axis direction", 3)
    norm = np.linalg.norm(direction)
    if norm == 0 or points is None:
        raise Refusal("malformed_prediction", "encoder_displacement needs a nonzero axis and two 3-D points")
    a, b = vector(points[0], "start", 3), vector(points[1], "end", 3)
    require_dimension(unit, "m", "encoder point unit")
    return {"observable": observable, "value": float((b - a) @ direction / norm), "unit": unit,
            "assumptions": ["rigid axis", "encoder reports displacement along the declared axis"]}


def _project(camera: dict | None, point: Any) -> dict:
    if camera is None:
        raise Refusal("malformed_prediction", "image projection needs camera intrinsics")
    require_keys(camera, "camera", {"fx", "fy", "cx", "cy", "unit"}, {"frame"})
    require_dimension(camera["unit"], "m", "camera point unit")
    p = vector(point, "camera-frame point", 3)
    if p[2] <= 0:
        raise Refusal("behind_camera", "A point with nonpositive depth has no pinhole projection")
    fx, fy = finite(camera["fx"], "fx", minimum=0, exclusive_minimum=True), finite(camera["fy"], "fy", minimum=0,
                                                                                     exclusive_minimum=True)
    pixel = np.array([fx * p[0] / p[2] + finite(camera["cx"], "cx"), fy * p[1] / p[2] + finite(camera["cy"], "cy")])
    jacobian = np.array([[fx / p[2], 0.0, -fx * p[0] / p[2] ** 2], [0.0, fy / p[2], -fy * p[1] / p[2] ** 2]])
    return {"observable": "image_residual", "value": pixel, "unit": "px", "jacobian_px_per_unit": jacobian,
            "frame": camera.get("frame", "camera"), "assumptions": ["undistorted pinhole camera", "calibrated intrinsics"]}


def observation(observable: str, value: Any, unit: str, std: float, *, acquisition: str, frame: str | None = None,
                clock: str | None = None, time_s: float | None = None, instrument: str | None = None,
                cov: Any = None) -> dict:
    """A typed observation record; ``acquisition`` distinguishes physical from synthetic."""
    if observable not in OBSERVABLES:
        raise Refusal("unknown_observable", f"Observable {observable!r} is not declared")
    if acquisition not in ACQUISITION_KINDS:
        raise Refusal("unknown_acquisition", f"Acquisition {acquisition!r} is not declared")
    if time_s is not None and clock is None:
        raise Refusal("clock_unspecified", "An observation timestamp must name its clock")
    record = {"observable": observable, "value": plain(np.asarray(value, dtype=float)), "unit": text(unit, "unit", 64),
              "std": finite(std, "std", minimum=0.0), "acquisition": acquisition, "frame": frame, "clock": clock,
              "time_s": time_s, "instrument": instrument}
    if cov is not None:
        record["covariance"] = plain(covariance(cov, "observation covariance"))
    return record


def compare(measured: dict, predicted: dict, *, predicted_std: float = 0.0) -> dict:
    """Residual of a measurement against a prediction of the *same* observable."""
    if measured.get("observable") != predicted.get("observable"):
        raise Refusal("observable_mismatch", f"Measured {measured.get('observable')!r} cannot be compared with predicted "
                      f"{predicted.get('observable')!r}; they are different quantities",
                      measured=measured.get("observable"), predicted=predicted.get("observable"))
    if measured.get("frame") and predicted.get("frame") and measured["frame"] != predicted["frame"]:
        raise Refusal("frame_mismatch", f"Measured in {measured['frame']!r}, predicted in {predicted['frame']!r}")
    value = np.asarray(convert(predicted["value"], predicted["unit"], measured["unit"]), dtype=float)
    residual = np.asarray(measured["value"], dtype=float) - value
    scale = abs(float(convert(1.0, predicted["unit"], measured["unit"])))
    sigma = math.hypot(float(measured.get("std", 0.0)), predicted_std * scale)
    normalized = None if sigma == 0 else float(np.max(np.abs(residual)) / sigma)
    return plain({"observable": measured["observable"], "unit": measured["unit"], "residual": residual,
                  "combined_std": sigma, "normalized_residual": normalized,
                  "consistent": None if normalized is None else normalized <= CONSISTENCY_SIGMA,
                  "acquisition": measured.get("acquisition"),
                  "physical_evidence": measured.get("acquisition") == "physical"})


def explain(surface: Surface, points: tuple[Any, Any], measured_value: float, unit: str, std: float) -> dict:
    """Which bound geometric observables are consistent with a measured distance?"""
    std = finite(std, "std", minimum=0.0, exclusive_minimum=True)
    rows = []
    for observable in ("intrinsic_distance", "camera_chord"):
        try:
            predicted = predict(observable, surface, points=points)
        except Refusal as exc:
            rows.append({"observable": observable, "available": False, "reason": exc.code})
            continue
        value = float(convert(predicted["value"], predicted["unit"], unit))
        z = (measured_value - value) / std
        rows.append({"observable": observable, "available": True, "predicted": value, "residual": measured_value - value,
                     "z": z, "consistent": abs(z) <= CONSISTENCY_SIGMA})
    consistent = [row["observable"] for row in rows if row.get("consistent")]
    separation = None
    available = [row for row in rows if row.get("available")]
    if len(available) == 2:
        separation = abs(available[0]["predicted"] - available[1]["predicted"]) / std
    return plain({"measured": measured_value, "unit": unit, "std": std, "candidates": rows,
                  "consistent_with": consistent, "observable_separation_sigma": separation,
                  "reading": ("indistinguishable at this separation" if separation is not None and separation < 2 * CONSISTENCY_SIGMA
                              else ("consistent with " + ", ".join(consistent)) if consistent
                              else "inconsistent with every bound observable")})
