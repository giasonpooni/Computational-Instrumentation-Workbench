"""Backwards-compatible recording facade over the generic scientific adapter seam.

Domain calculations live in adapters; sample selection and record descriptions
operate on the common CIW contract, including explicit missing observations.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .adapters.oscillator import make_demo_run
from .adapters.registry import default_registry
from .core.records import number


def validate_run(run: dict) -> None:
    """Validate generic structure, then delegate domain checks to its adapter."""
    default_registry().validate_run(run)


def run_metadata(run: dict) -> dict:
    """Return a detached description without scientific or render arrays."""
    validate_run(run)
    return {
        "run_id": run["run_id"], "evidence_id": run["evidence_id"],
        "instrument": run["instrument"], "metadata": deepcopy(run["metadata"]),
        "channels": {name: {key: deepcopy(value) for key, value in channel.items() if key != "values"}
                     for name, channel in run["channels"].items()},
    }


def inspect_sample(run: dict, time_s: float) -> dict:
    """Resolve a cursor to the nearest retained sample, taking the earlier tie."""
    validate_run(run)
    target = number(time_s, "time_s")
    if not 0 <= target <= run["metadata"]["duration_s"]:
        raise ValueError("time_s must be within [0, duration_s]")
    time = np.asarray(run["time_s"], dtype=np.float64)
    right = int(np.searchsorted(time, target, side="left"))
    if right == 0:
        index = 0
    elif right == len(time):
        index = len(time) - 1
    else:
        midpoint = time[right - 1] + 0.5 * (time[right] - time[right - 1])
        index = right - 1 if target <= midpoint else right
    return {
        "run_id": run["run_id"], "evidence_id": run["evidence_id"],
        "sample_index": index, "time_s": float(time[index]),
        "values": {name: None if channel["values"][index] is None else float(channel["values"][index])
                   for name, channel in run["channels"].items()},
        "units": {name: channel["unit"] for name, channel in run["channels"].items()},
    }


def compute_statistics(run: dict, channel: str, interval_s: list[float]) -> dict:
    """Dispatch the historical statistics API to the selected scientific adapter."""
    return default_registry().execute("statistics.v1", run,
                                      {"channel": channel, "interval_s": interval_s})


def compute_spectrum(run: dict, channel: str, interval_s: list[float]) -> dict:
    """Dispatch the historical spectrum API to the selected scientific adapter."""
    return default_registry().execute("spectrum.periodogram.v1", run,
                                      {"channel": channel, "interval_s": interval_s})
