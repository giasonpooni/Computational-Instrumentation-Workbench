"""Compare two retained batches without evaluating anything again.

Both sides are read from what was saved: the batch index, and the
``ciw-plsr-run-v1`` bundle each entry names. Nothing here loads a model into
the numerical engine or asks for a verdict, so a comparison says what the
retained evidence says and cannot quietly substitute a fresh answer.

Every compared sample resolves to the bundle it came from. A bundle written
under the installed source pin is validated in full, through the same check
``inspect`` uses. A bundle written under a different pin is still resolved
structurally -- its own digest and the companion record digest are recomputed
from the saved JSON with the stdlib alone, which is the published consumer
contract -- and the report says which resolution each side received.

Differences are exact unless a tolerance policy names the field, the same rule
and the same comparison the reference corpus uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from . import plsr
from . import plsr_batch
from . import plsr_corpus
from . import plsr_engine as engine

COMPARISON_SCHEMA = "ciw-plsr-comparison-v1"

#: Fields the companion record excludes from its own digest. Restated here so
#: a saved record can be resolved without importing the runtime, exactly as an
#: outside consumer would. `test_plsr_compare` checks this against the pinned
#: runtime's own definition.
_VOLATILE_RECORD_FIELDS = frozenset(
    {"generated_at", "record_digest", "cargo_prove_available", "guest_manifest"})

_VERDICT_FIELDS = ("code", "presentation_category", "inequality_certified",
                   "meets_required_margin", "operationally_acceptable",
                   "required_margin", "level", "details")


class ComparisonError(ValueError):
    """A batch that cannot be read, or a comparison that has nothing to compare."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(f"invalid PLSR comparison: {message}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def record_digest(record: Mapping[str, Any]) -> str:
    """Recompute a companion record digest with the stdlib alone."""
    stable = {key: value for key, value in record.items()
              if key not in _VOLATILE_RECORD_FIELDS}
    return sha256(json.dumps(stable, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def bundle_digest(bundle: Mapping[str, Any]) -> str:
    """Recompute a saved run's own digest with the stdlib alone."""
    body = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    return sha256(_canonical(body).encode("ascii")).hexdigest()


def _resolve_bundle(path: Path, entry: Mapping[str, Any], supported: Mapping[str, Any]
                    ) -> dict[str, Any]:
    """Read one retained bundle and say how far it could be resolved."""
    resolved: dict[str, Any] = {"saved_file": str(path), "resolution": "unresolved",
                                "problems": [], "bundle": None}
    try:
        bundle = plsr._read(path)
    except (OSError, ValueError) as exc:
        resolved["problems"].append(f"unreadable: {exc}")
        return resolved
    if not isinstance(bundle, Mapping) or bundle.get("bundle_schema") != plsr.BUNDLE_SCHEMA:
        resolved["problems"].append(f"not a {plsr.BUNDLE_SCHEMA} document")
        return resolved
    resolved["bundle"] = dict(bundle)
    resolved["resolution"] = "structural"
    if bundle.get("bundle_digest") != bundle_digest(bundle):
        resolved["problems"].append("bundle_digest does not match the saved bundle")
    record = bundle.get("record")
    if not isinstance(record, Mapping) or record.get("record_digest") != record_digest(record):
        resolved["problems"].append("record_digest does not match the saved record")
    for field in ("result_id", "evidence_id"):
        if entry[field] is not None and bundle.get(field) != entry[field]:
            resolved["problems"].append(f"{field} differs from the batch index")
    if isinstance(record, Mapping) and entry["record_digest"] is not None \
            and record.get("record_digest") != entry["record_digest"]:
        resolved["problems"].append("record_digest differs from the batch index")
    runtime = bundle.get("runtime")
    if isinstance(runtime, Mapping) and all(
            runtime.get(key) == supported[key] for key in plsr._PIN_FIELDS):
        try:
            plsr._validate_bundle(dict(bundle))
        except (OSError, ValueError) as exc:
            resolved["problems"].append(f"failed full validation: {exc}")
        else:
            resolved["resolution"] = "validated"
    if resolved["problems"]:
        resolved["resolution"] = "unresolved"
    return resolved


def _side(directory: str | Path, supported: Mapping[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    index = plsr_batch.read_index(directory)
    entries = {entry["sample_id"]: entry for entry in index["entries"]}
    bundles: dict[str, dict[str, Any]] = {}
    for sample_id, entry in entries.items():
        if entry["saved_file"] is None:
            continue
        bundles[sample_id] = _resolve_bundle(
            directory / plsr_batch.RUNS_NAME / entry["saved_file"], entry, supported)
    return {"output_dir": str(directory), "index": index, "entries": entries, "bundles": bundles}


def _header(side: Mapping[str, Any]) -> dict[str, Any]:
    index = side["index"]
    return {key: index[key] for key in
            ("batch_id", "batch_digest", "index_digest", "requested", "status",
             "outcomes", "codes", "operationally_acceptable", "runtime",
             "started_at", "finished_at")} | {"output_dir": side["output_dir"]}


def _runtime_differences(baseline: Mapping[str, Any], candidate: Mapping[str, Any]
                         ) -> list[dict[str, Any]]:
    keys = sorted(set(baseline) | set(candidate))
    return [{"field": key, "baseline": baseline.get(key), "candidate": candidate.get(key),
             "pin": key in plsr._PIN_FIELDS}
            for key in keys if baseline.get(key) != candidate.get(key)]


def _differences(baseline: Any, candidate: Any, path: str,
                 fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    plsr_corpus._compare(baseline, candidate, path, fields, found)
    # The shared comparison names the two sides declared/observed; here they
    # are the baseline and the candidate.
    return [{"field": item["field"], "reason": item["reason"], "baseline": item["declared"],
             "candidate": item["observed"], "tolerance": item["tolerance"]} for item in found]


def _transition(before: Any, after: Any) -> str | None:
    return None if before == after else f"{before}->{after}"


def _compare_sample(sample_id: str, baseline: Mapping[str, Any], candidate: Mapping[str, Any],
                    fields: Mapping[str, Any]) -> dict[str, Any]:
    before, after = baseline["entries"][sample_id], candidate["entries"][sample_id]
    left = baseline["bundles"].get(sample_id)
    right = candidate["bundles"].get(sample_id)
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "baseline": {"outcome": before["outcome"], "code": before["code"],
                     "result_id": before["result_id"],
                     "record_digest": before["record_digest"],
                     "saved_file": None if left is None else left["saved_file"],
                     "resolution": "absent" if left is None else left["resolution"],
                     "resolution_problems": [] if left is None else left["problems"]},
        "candidate": {"outcome": after["outcome"], "code": after["code"],
                      "result_id": after["result_id"],
                      "record_digest": after["record_digest"],
                      "saved_file": None if right is None else right["saved_file"],
                      "resolution": "absent" if right is None else right["resolution"],
                      "resolution_problems": [] if right is None else right["problems"]},
        "outcome_transition": _transition(before["outcome"], after["outcome"]),
        "code_transition": _transition(before["code"], after["code"]),
        "record_digest_matches": None,
        "verdict_differences": [], "diagnostic_differences": [],
        "model_differences": [], "sample_differences": [], "runtime_differences": [],
        "model_artifact_digest_changed": before["model_artifact_digest"] != after["model_artifact_digest"],
        "comparable": False, "changed": False,
    }
    if left is None or right is None or left["bundle"] is None or right["bundle"] is None:
        result["changed"] = bool(result["outcome_transition"] or result["code_transition"]
                                 or result["model_artifact_digest_changed"])
        return result
    result["comparable"] = True
    first, second = left["bundle"], right["bundle"]
    result["record_digest_matches"] = (
        first["record"]["record_digest"] == second["record"]["record_digest"])
    result["verdict_differences"] = _differences(
        {key: first["record"][key] for key in _VERDICT_FIELDS},
        {key: second["record"][key] for key in _VERDICT_FIELDS}, "", fields)
    result["diagnostic_differences"] = _differences(
        first["record"]["diagnostics"], second["record"]["diagnostics"], "diagnostics", fields)
    if first["model"]["artifact_digest"] != second["model"]["artifact_digest"]:
        result["model_differences"] = _differences(first["model"], second["model"], "model", fields)
    result["sample_differences"] = _differences(first["sample"], second["sample"], "sample", fields)
    result["runtime_differences"] = _runtime_differences(first["runtime"], second["runtime"])
    result["changed"] = bool(
        result["outcome_transition"] or result["code_transition"]
        or result["verdict_differences"] or result["diagnostic_differences"]
        or result["model_differences"] or result["sample_differences"]
        or result["runtime_differences"] or result["model_artifact_digest_changed"])
    return result


def _touches(differences: list[dict[str, Any]], prefix: str) -> bool:
    return any(item["field"] == prefix or item["field"].startswith(prefix + ".")
               for item in differences)


def _summary(samples: list[dict[str, Any]]) -> dict[str, int]:
    def count(predicate) -> int:
        return sum(1 for sample in samples if predicate(sample))

    return {
        "compared": len(samples),
        "identical": count(lambda s: not s["changed"]),
        "changed": count(lambda s: s["changed"]),
        "unresolved": count(lambda s: "unresolved" in
                            (s["baseline"]["resolution"], s["candidate"]["resolution"])),
        "outcome_changed": count(lambda s: s["outcome_transition"]),
        "code_changed": count(lambda s: s["code_transition"]),
        "verdict_changed": count(lambda s: s["verdict_differences"]),
        "diagnostics_changed": count(lambda s: s["diagnostic_differences"]),
        "record_digest_changed": count(lambda s: s["record_digest_matches"] is False),
        "model_changed": count(lambda s: s["model_artifact_digest_changed"] or s["model_differences"]),
        "plant_changed": count(lambda s: _touches(s["model_differences"], "model.plant")),
        "certificate_changed": count(lambda s: _touches(s["model_differences"], "model.certificate")),
        "required_margin_changed": count(
            lambda s: _touches(s["model_differences"], "model.policy.required_margin")),
        "level_changed": count(lambda s: _touches(s["model_differences"], "model.policy.level")),
        "sample_changed": count(lambda s: s["sample_differences"]),
        "runtime_changed": count(lambda s: s["runtime_differences"]),
    }


def _tally(samples: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        transition = sample[key]
        if transition is not None:
            counts[transition] = counts.get(transition, 0) + 1
    return dict(sorted(counts.items()))


def load_tolerance_policy(path: str | Path) -> dict[str, Any]:
    """Read a tolerance policy, or take the one a reference corpus declares."""
    document = plsr._read(Path(path))
    _require(isinstance(document, Mapping), "a tolerance policy must be a JSON object")
    policy = document.get("tolerance_policy", document)
    plsr_corpus._tolerance_fields(policy)
    return dict(policy)


def compare_batches(baseline_dir: str | Path, candidate_dir: str | Path, *,
                    tolerance_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare two saved batches, resolving each result to its retained bundle."""
    fields = {} if tolerance_policy is None else plsr_corpus._tolerance_fields(tolerance_policy)
    supported = engine.runtime_identity()
    baseline, candidate = _side(baseline_dir, supported), _side(candidate_dir, supported)
    common = [sample_id for sample_id in baseline["entries"] if sample_id in candidate["entries"]]
    samples = [_compare_sample(sample_id, baseline, candidate, fields) for sample_id in common]
    only_baseline = sorted(set(baseline["entries"]) - set(candidate["entries"]))
    only_candidate = sorted(set(candidate["entries"]) - set(baseline["entries"]))
    pin = _runtime_differences(baseline["index"]["runtime"], candidate["index"]["runtime"])
    reasons: list[str] = []
    if not common:
        reasons.append("the two batches share no sample_id, so there is nothing to compare")
    if baseline["index"]["batch_digest"] != candidate["index"]["batch_digest"]:
        reasons.append("the batches were run from different declared collections")
    if only_baseline or only_candidate:
        reasons.append(
            f"{len(only_baseline)} sample(s) only in the baseline and "
            f"{len(only_candidate)} only in the candidate")
    if any(item["pin"] for item in pin):
        reasons.append("the batches were run against different runtime source pins")
    unresolved = [sample["sample_id"] for sample in samples
                  if "unresolved" in (sample["baseline"]["resolution"],
                                      sample["candidate"]["resolution"])]
    if unresolved:
        reasons.append(f"{len(unresolved)} sample(s) did not resolve to their retained evidence")
    summary = _summary(samples)
    report = {
        "comparison_schema": COMPARISON_SCHEMA,
        "baseline": _header(baseline),
        "candidate": _header(candidate),
        "tolerance_policy": None if tolerance_policy is None else dict(tolerance_policy),
        "compatibility": {
            "same_collection": baseline["index"]["batch_digest"] == candidate["index"]["batch_digest"],
            "same_runtime_pin": not any(item["pin"] for item in pin),
            "runtime_differences": pin,
            "only_in_baseline": only_baseline,
            "only_in_candidate": only_candidate,
            "unresolved_samples": unresolved,
            "incompatible": bool(reasons),
            "reasons": reasons,
        },
        "outcome_transitions": _tally(samples, "outcome_transition"),
        "code_transitions": _tally(samples, "code_transition"),
        "summary": summary,
        "samples": samples,
        "status": "incompatible" if reasons else ("changed" if summary["changed"] else "identical"),
    }
    return report
