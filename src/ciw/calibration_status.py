"""Read-only calibration applicability derived for one explicit serving time.

Serving metadata never enters an evidence/result digest. RCI owns immutable
acquisition applicability; CIW reports whether that profile is currently in its
declared time interval without reclassifying a historical observation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re


_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z")


def _timestamp(value: object, name: str, *, strict: bool = False) -> datetime:
    if not isinstance(value, str) or (strict and not _TIMESTAMP.fullmatch(value)):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp with seconds")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid timezone-aware ISO timestamp") from exc
    if result.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def calibration_status(run: dict, evaluated_at: str | None = None) -> list[dict]:
    """Describe acquisition and current time validity without changing ``run``.

    V1 records have no persisted applicability artifact: their acquisition
    comparison is explicitly labelled a legacy derivation. Neither version's
    declared calibration interval establishes traceability or authenticates the
    profile. Expiry does not invalidate an observation acquired while applicable.
    """
    now = (datetime.now(timezone.utc) if evaluated_at is None
           else _timestamp(evaluated_at, "evaluated_at", strict=True))
    serving_time = now.astimezone(timezone.utc).isoformat()
    source = run.get("metadata", {}).get("rci_source")
    if source is None:
        return []
    try:
        sensors = source["sensors"]
        if not isinstance(sensors, list):
            raise ValueError("Calibration sensors must be a list")
        statuses = []
        for sensor in sensors:
            measurement = sensor["measurement"]
            profile = measurement["calibration"]
            records = measurement["records"]
            if not isinstance(records, list) or len(records) != 1:
                raise ValueError("Calibration serving status requires one observation per sensor")
            record = records[0]
            start = _timestamp(profile["valid_from"], "calibration valid_from")
            end = _timestamp(profile["valid_until"], "calibration valid_until")
            acquired = _timestamp(record["observed_at"], "observation observed_at")
            if end <= start:
                raise ValueError("Calibration valid_until must be after valid_from")
            applicable = start <= acquired < end
            version = record.get("schema")
            if version == "measurement-record.v2":
                persisted = record["acquisition_applicability"]
                expected = {"observed_at": record["observed_at"], "applicable": True,
                            "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
                            "basis": "caller_declared_acquisition_time"}
                if (not isinstance(persisted, dict) or set(persisted) != set(expected)
                        or persisted != expected or persisted.get("applicable") is not True or not applicable):
                    raise ValueError("Persisted acquisition applicability contradicts its observation/profile")
                provenance = "persisted.v2"
                basis = persisted["basis"]
            elif version == "measurement-record.v1":
                if "acquisition_applicability" in record:
                    raise ValueError("Legacy v1 cannot claim a persisted v2 applicability artifact")
                provenance = "derived.legacy-v1"
                basis = "legacy_declared_acquisition_time"
            else:
                raise ValueError("Unknown calibration measurement record schema")
            if not all(isinstance(value, str) and value for value in (sensor["name"], profile["calibration_id"])):
                raise ValueError("Calibration source and profile identities must be nonempty strings")
            statuses.append({
                "source": sensor["name"], "calibration_id": profile["calibration_id"],
                "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
                "acquisition": {"observed_at": record["observed_at"],
                                "applicable_at_acquisition": applicable,
                                "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
                                "basis": basis, "provenance": provenance},
                "serving": {"evaluated_at": serving_time, "expired": now >= end,
                            "not_yet_valid": now < start, "current_applicability": start <= now < end},
            })
        return statuses
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid retained calibration source structure") from exc
