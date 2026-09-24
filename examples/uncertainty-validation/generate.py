"""Regenerate the four uncertainty-validation fixtures, retaining every draw.

One constant-velocity trajectory supplies the declared reference values. The
estimate errors are one seeded standard-normal draw shaped by the true
covariance, and a scalar position innovation shares the same draw. The four
cases declare different covariances over identical errors:

- consistent: the declared covariance is the true one;
- covariance-too-small: the declared covariance is a quarter of the truth;
- covariance-too-large: the declared covariance is four times the truth;
- unknown-dependence: the consistent case with cross-sample dependence left unknown.

No case is selected for its outcome; the seed and sample count are fixed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ciw.uncertainty_validation import CONFIGURATION, SOURCE_SCHEMA, validate_source
from ciw.telemetry import canonical

SEED = 20260924
SAMPLES = 64
TRUE_COVARIANCE = np.array([[0.010, 0.0015], [0.0015, 0.0025]])
OBSERVATION = np.array([[1.0, 0.0]])
MEASUREMENT_NOISE = np.array([[0.004]])
CASES = {"consistent": (1.0, "declared_independent"), "covariance-too-small": (0.25, "declared_independent"),
         "covariance-too-large": (4.0, "declared_independent"), "unknown-dependence": (1.0, "unknown")}


def build_sources():
    generator = np.random.Generator(np.random.PCG64(SEED))
    standard = generator.standard_normal((SAMPLES, 2))
    measurement_standard = generator.standard_normal((SAMPLES, 1))
    cholesky = np.linalg.cholesky(TRUE_COVARIANCE)
    errors = standard @ cholesky.T
    times = [float(index) for index in range(SAMPLES)]
    references = [[1.0 + 0.5 * time, 0.5] for time in times]
    sources = {}
    for name, (scale, dependence) in CASES.items():
        declared = (scale * TRUE_COVARIANCE).tolist()
        innovation_covariance = (scale * OBSERVATION @ TRUE_COVARIANCE @ OBSERVATION.T + MEASUREMENT_NOISE).tolist()
        true_innovation_covariance = OBSERVATION @ TRUE_COVARIANCE @ OBSERVATION.T + MEASUREMENT_NOISE
        samples = [{"time": time, "reference": reference,
                    "estimate": [reference[0] + errors[index, 0], reference[1] + errors[index, 1]],
                    "covariance": declared}
                   for index, (time, reference) in enumerate(zip(times, references))]
        innovations = [{"time": time,
                        "innovation": [float(np.sqrt(true_innovation_covariance[0, 0]) * measurement_standard[index, 0])],
                        "covariance": innovation_covariance}
                       for index, time in enumerate(times)]
        sources[name] = {
            "schema": SOURCE_SCHEMA, "experiment_id": "uncertainty-validation:" + name,
            "configuration": dict(CONFIGURATION),
            "request": {"confidence": 0.95, "state_names": ["position", "velocity"], "state_units": ["m", "m/s"],
                        "frame": "frame:track", "time_basis": "clock:fixture", "reference_origin": "synthetic_fixture",
                        "cross_sample_dependence": dependence, "samples": samples,
                        "measurement_names": ["position_measurement"], "measurement_units": ["m"],
                        "innovations": innovations},
        }
        validate_source(canonical(sources[name]))
    return sources


if __name__ == "__main__":
    directory = Path(__file__).parent
    for name, source in build_sources().items():
        (directory / (name + ".json")).write_text(json.dumps(source, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", name + ".json")
