"""Read-only schemas for oscillator operation payloads; never rerun numerics."""
from __future__ import annotations

import math

from ..core.records import number as _number


def validate_payload(operation: str, data: dict, run: dict, parameters: dict, selection: dict) -> None:
    channel = selection["channel"]
    start, end = selection["interval_s"]
    # Counting retained timestamps checks the stored shape, without evaluating
    # statistics, applying a window, or executing a Fourier transform.
    expected_count = sum(start <= time < end for time in run["time_s"])
    if not isinstance(data, dict):
        raise ValueError("Saved result data must be an object")
    count = data.get("sample_count")
    if type(count) is not int or count != expected_count:
        raise ValueError("Saved result sample_count does not match its source interval")
    unit = run["channels"][channel]["unit"]
    if operation == "statistics.v1":
        if data.keys() != {"sample_count", "mean", "minimum", "maximum", "rms", "unit"}:
            raise ValueError("Invalid saved statistics data fields")
        if data["unit"] != unit:
            raise ValueError("Saved statistics unit does not match its source channel")
        mean, minimum, maximum, rms = (_number(data[key], f"Saved statistics {key}")
                                      for key in ("mean", "minimum", "maximum", "rms"))
        if minimum > maximum or rms < 0:
            raise ValueError("Saved statistics have invalid bounds or negative RMS")
        if ((mean < minimum and not math.isclose(mean, minimum, rel_tol=1e-12))
                or (mean > maximum and not math.isclose(mean, maximum, rel_tol=1e-12))):
            raise ValueError("Saved statistics mean is outside its bounds")
    elif operation == "spectrum.periodogram.v1":
        fields = {"sample_count", "method", "window", "detrend", "scaling", "frequency_hz",
                  "psd", "unit", "peak_frequency_hz", "sample_rate_hz"}
        if data.keys() != fields or count < 4:
            raise ValueError("Invalid saved spectrum data fields or sample count")
        for name, expected in (("method", "periodogram"), ("window", "hann"),
                               ("detrend", "constant"), ("scaling", "density")):
            if data[name] != expected:
                raise ValueError(f"Unsupported saved spectrum {name}")
        sample_rate = _number(data["sample_rate_hz"], "Saved spectrum sample_rate_hz")
        if sample_rate != run["metadata"]["sample_rate_hz"] or data["unit"] != f"({unit})^2/Hz":
            raise ValueError("Saved spectrum sample rate or density unit does not match its source")
        frequencies, psd = data["frequency_hz"], data["psd"]
        size = count // 2 + 1
        if not isinstance(frequencies, list) or not isinstance(psd, list) or len(frequencies) != size or len(psd) != size:
            raise ValueError("Saved spectrum arrays do not match the one-sided transform shape")
        for index, (frequency, density) in enumerate(zip(frequencies, psd)):
            frequency = _number(frequency, "Saved spectrum frequency_hz")
            density = _number(density, "Saved spectrum psd")
            if density < 0:
                raise ValueError("Saved spectrum density must be nonnegative")
            expected_frequency = index * (sample_rate / count)
            if not math.isclose(frequency, expected_frequency, rel_tol=1e-12, abs_tol=sample_rate * 1e-14):
                raise ValueError("Saved spectrum frequency grid does not match its sample rate and count")
        peak = data["peak_frequency_hz"]
        maximum = max(psd)
        if maximum == 0:
            if peak is not None:
                raise ValueError("A zero saved spectrum must have a null peak")
        elif _number(peak, "Saved spectrum peak_frequency_hz") != frequencies[psd.index(maximum)]:
            raise ValueError("Saved spectrum peak does not match its stored density array")
    else:
        raise ValueError("Unsupported saved result operation_id")
