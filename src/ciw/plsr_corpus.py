"""A declared reference corpus and the contract check that reproduces it.

The corpus stores complete models, explicit samples, the expected output of
each case, the runtime identity the expectations were recorded against, and
the numerical tolerance policy under which they are compared. Checking never
writes an expectation: recording is a separate operation that shows what it
would change and refuses to replace a differing corpus without being told to.

Status codes, booleans, strings and integers compare exactly. A float
compares exactly unless the tolerance policy names its field, so a tolerance
is opt-in per field and visible in the corpus rather than implied by the
comparison code.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import plsr
from . import plsr_engine as engine

CORPUS_SCHEMA = "ciw-plsr-corpus-v1"
REPORT_SCHEMA = "ciw-plsr-corpus-report-v1"

#: Runtime identity fields a corpus binds by default. The remaining fields
#: (interpreter and dependency versions) are retained and reported, because a
#: corpus that bound them could not be checked on a second supported platform.
DEFAULT_RUNTIME_BINDING = ("repository", "commit", "package_version",
                           "source_digest", "adapter_version")

_HEADER_FIELDS = {"corpus_schema", "corpus_id", "corpus_version", "description",
                  "runtime", "runtime_binding", "tolerance_policy", "cases", "corpus_digest"}
_CASE_FIELDS = {"case_id", "purpose", "description", "model_file",
                "model_artifact_digest", "sample", "expect"}
_POLICY_FIELDS = {"policy_id", "description", "digest_enforcement", "fields"}
_TOLERANCE_FIELDS = {"relative", "absolute"}
_PURPOSES = {"certification", "margin_shortfall", "violation", "domain_refusal",
             "numerical_boundary", "contract_refusal"}
_EXPECT_EVALUATED = {"outcome", "code", "presentation_category", "inequality_certified",
                     "meets_required_margin", "operationally_acceptable", "required_margin",
                     "level", "record_digest", "diagnostics"}
_EXPECT_REFUSED = {"outcome", "reason_contains"}
_DIGEST_ENFORCEMENT = {"reported", "enforced"}


class CorpusError(ValueError):
    """An invalid corpus document, or one that does not bind this runtime."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def corpus_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the corpus with its own digest field removed."""
    body = {key: value for key, value in document.items() if key != "corpus_digest"}
    return sha256(_canonical(body).encode("ascii")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(f"invalid PLSR corpus: {message}")


def _text(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label} must be a nonempty string")
    return value


def _tolerance_fields(policy: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    _require(isinstance(policy, Mapping) and set(policy) == _POLICY_FIELDS,
             "tolerance_policy fields")
    _text(policy["policy_id"], "policy_id")
    _text(policy["description"], "tolerance policy description")
    _require(policy["digest_enforcement"] in _DIGEST_ENFORCEMENT,
             "digest_enforcement must be 'reported' or 'enforced'")
    fields = policy["fields"]
    _require(isinstance(fields, Mapping), "tolerance_policy.fields must be an object")
    result: dict[str, dict[str, float]] = {}
    for name, entry in fields.items():
        _require(isinstance(entry, Mapping) and set(entry) == _TOLERANCE_FIELDS,
                 f"tolerance entry {name!r} needs exactly relative and absolute")
        for key in _TOLERANCE_FIELDS:
            value = entry[key]
            _require(type(value) is float and math.isfinite(value) and value >= 0.0,
                     f"tolerance {name}.{key} must be a finite non-negative float")
        result[name] = {"relative": float(entry["relative"]),
                        "absolute": float(entry["absolute"])}
    return result


def _validate_sample_shape(sample: Any) -> None:
    _require(isinstance(sample, Mapping) and set(sample) ==
             {"sample_schema", "x", "theta", "theta_dot"},
             "a case sample needs exactly sample_schema, x, theta and theta_dot")


def _validate_expect(expect: Any, case_id: str) -> None:
    _require(isinstance(expect, Mapping) and "outcome" in expect, f"{case_id}: expect object")
    outcome = expect["outcome"]
    if outcome == "refused":
        _require(set(expect) == _EXPECT_REFUSED, f"{case_id}: refused expectation fields")
        _text(expect["reason_contains"], f"{case_id}: reason_contains")
        return
    _require(outcome == "evaluated", f"{case_id}: outcome must be 'evaluated' or 'refused'")
    _require(set(expect) == _EXPECT_EVALUATED, f"{case_id}: evaluated expectation fields")
    _text(expect["code"], f"{case_id}: code")
    _text(expect["presentation_category"], f"{case_id}: presentation_category")
    _text(expect["record_digest"], f"{case_id}: record_digest")
    for key in ("inequality_certified", "meets_required_margin", "operationally_acceptable"):
        _require(type(expect[key]) is bool, f"{case_id}: {key} must be boolean")


def load_corpus(path: str | Path, *, sealed: bool = True) -> dict[str, Any]:
    """Read and structurally validate a corpus document.

    With ``sealed`` the recorded runtime identity and corpus digest must be
    present and consistent. A recording plan is loaded without them.
    """
    document = plsr._read(Path(path))
    _require(isinstance(document, Mapping), "a corpus must be a JSON object")
    missing = _HEADER_FIELDS - set(document)
    optional = {"runtime", "corpus_digest"} if not sealed else set()
    _require(not (missing - optional) and not (set(document) - _HEADER_FIELDS),
             "corpus header fields")
    _require(document["corpus_schema"] == CORPUS_SCHEMA,
             f"corpus_schema must be {CORPUS_SCHEMA!r}")
    _text(document["corpus_id"], "corpus_id")
    _text(document["corpus_version"], "corpus_version")
    _text(document["description"], "corpus description")
    binding = document["runtime_binding"]
    _require(type(binding) is list and binding and all(type(item) is str for item in binding)
             and len(set(binding)) == len(binding), "runtime_binding must be distinct strings")
    _require(set(binding) <= set(engine.runtime_identity()),
             "runtime_binding names a field the runtime identity does not have")
    _tolerance_fields(document["tolerance_policy"])
    cases = document["cases"]
    _require(type(cases) is list and bool(cases), "a corpus needs at least one case")
    seen: set[str] = set()
    for case in cases:
        _require(isinstance(case, Mapping), "each case must be an object")
        _require(set(case) == _CASE_FIELDS
                 or (not sealed and set(case) == _CASE_FIELDS - {"expect"}),
                 "case fields")
        case_id = _text(case.get("case_id"), "case_id")
        _require(case_id not in seen, f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        _require(case["purpose"] in _PURPOSES,
                 f"{case_id}: purpose must be one of {sorted(_PURPOSES)}")
        _text(case["description"], f"{case_id}: description")
        name = _text(case["model_file"], f"{case_id}: model_file")
        _require("\\" not in name and not Path(name).is_absolute()
                 and ".." not in Path(name).parts,
                 f"{case_id}: model_file must be a relative path beside the corpus")
        _require(type(case["model_artifact_digest"]) is str
                 and len(case["model_artifact_digest"]) == 64,
                 f"{case_id}: model_artifact_digest")
        _validate_sample_shape(case["sample"])
        if sealed or case.get("expect") is not None:
            _validate_expect(case["expect"], case_id)
    if sealed:
        runtime = document["runtime"]
        _require(isinstance(runtime, Mapping)
                 and set(runtime) == set(engine.runtime_identity()),
                 "recorded runtime identity fields")
        _require(type(document["corpus_digest"]) is str
                 and document["corpus_digest"] == corpus_digest(document),
                 "corpus_digest does not match the declared corpus")
    return dict(document)


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _exact_float(declared: float, observed: float) -> bool:
    if math.isnan(declared) or math.isnan(observed):
        return False
    # -0.0 == 0.0 in float64; a declared expectation distinguishes them.
    return declared == observed and math.copysign(1.0, declared) == math.copysign(1.0, observed)


def _compare(declared: Any, observed: Any, path: str, fields: Mapping[str, Any],
             differences: list[dict[str, Any]]) -> None:
    def note(message: str, tolerance: Any = None) -> None:
        differences.append({"field": path or "$", "reason": message,
                            "declared": declared if _kind(declared) not in ("object", "array")
                            else _kind(declared),
                            "observed": observed if _kind(observed) not in ("object", "array")
                            else _kind(observed),
                            "tolerance": tolerance})

    if isinstance(declared, Mapping):
        if not isinstance(observed, Mapping):
            note(f"expected an object, observed {_kind(observed)}")
            return
        extra = sorted(set(observed) - set(declared))
        absent = sorted(set(declared) - set(observed))
        if extra or absent:
            differences.append({"field": path or "$", "reason": "object keys differ",
                                "declared": absent, "observed": extra, "tolerance": None})
        for key in sorted(set(declared) & set(observed)):
            _compare(declared[key], observed[key],
                     f"{path}.{key}" if path else key, fields, differences)
        return
    if type(declared) is list:
        if type(observed) is not list:
            note(f"expected an array, observed {_kind(observed)}")
            return
        if len(declared) != len(observed):
            differences.append({"field": path or "$", "reason": "array length differs",
                                "declared": len(declared), "observed": len(observed),
                                "tolerance": None})
            return
        # Array indices are deliberately absent from the path, so a tolerance
        # named for a matrix field applies to every entry of that matrix.
        for item, other in zip(declared, observed, strict=True):
            _compare(item, other, path, fields, differences)
        return
    tolerance = fields.get(path)
    if _kind(declared) != _kind(observed):
        note(f"expected {_kind(declared)}, observed {_kind(observed)}", tolerance)
        return
    if type(declared) is float:
        if tolerance is None:
            if not _exact_float(declared, observed):
                note("float differs and no tolerance is declared for this field")
            return
        allowed = max(tolerance["absolute"], tolerance["relative"] * abs(declared))
        if not (math.isfinite(declared) and math.isfinite(observed)
                and abs(observed - declared) <= allowed):
            note(f"float differs by more than the declared tolerance {allowed:.3e}", tolerance)
        return
    if tolerance is not None:
        note(f"tolerance policy names a {_kind(declared)} field; only floats take a tolerance",
             tolerance)
        return
    if declared != observed:
        note("value differs")


def unused_tolerances(document: Mapping[str, Any]) -> list[str]:
    """Tolerance fields no declared expectation contains, as a policy hygiene report."""
    fields = _tolerance_fields(document["tolerance_policy"])
    reached: set[str] = set()

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else key)
        elif type(value) is list:
            for item in value:
                walk(item, path)
        elif type(value) is float:
            reached.add(path)

    for case in document["cases"]:
        expect = case.get("expect")
        if isinstance(expect, Mapping) and expect.get("outcome") == "evaluated":
            walk(dict(expect), "")
    return sorted(set(fields) - reached)


def _runtime_report(document: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    declared = document.get("runtime")
    if declared is None:
        return {"runtime_status": "unrecorded", "runtime_differences": []}
    binding = set(document["runtime_binding"])
    differences = [{"field": key, "declared": declared[key], "observed": observed[key],
                    "binding": key in binding}
                   for key in sorted(observed) if declared.get(key) != observed[key]]
    stale = any(item["binding"] for item in differences)
    return {"runtime_status": "stale" if stale else "supported",
            "runtime_differences": differences}


def _observe(model: Any, sample: Mapping[str, Any], output_dir: Path | None
             ) -> tuple[dict[str, Any], str | None]:
    if output_dir is None:
        return engine.evaluate(model, sample), None
    saved = plsr.evaluate_sample(model, sample, output_dir)
    return saved["bundle"]["record"], saved["saved_file"]


def _expectation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "outcome": "evaluated",
        "code": record["code"],
        "presentation_category": record["presentation_category"],
        "inequality_certified": record["inequality_certified"],
        "meets_required_margin": record["meets_required_margin"],
        "operationally_acceptable": record["operationally_acceptable"],
        "required_margin": record["required_margin"],
        "level": record["level"],
        "record_digest": record["record_digest"],
        "diagnostics": record["diagnostics"],
    }


def _run_case(case: Mapping[str, Any], root: Path, fields: Mapping[str, Any],
              output_dir: Path | None, enforce_digest: bool) -> dict[str, Any]:
    """Reproduce one declared case and compare it with its declared expectation."""
    result: dict[str, Any] = {
        "case_id": case["case_id"], "purpose": case["purpose"],
        "expected_outcome": case["expect"]["outcome"], "observed_outcome": None,
        "code": None, "record_digest": None, "record_digest_matches": None,
        "saved_file": None, "differences": [], "outcome": "mismatched",
        "observed_expectation": None, "refusal_reason": None,
    }
    try:
        model = engine.load_model(root / case["model_file"])
    except (OSError, ValueError) as exc:
        result["observed_outcome"] = "unreadable_model"
        result["outcome"] = "errored"
        result["refusal_reason"] = str(exc)
        return result
    if model.artifact_digest != case["model_artifact_digest"]:
        result["observed_outcome"] = "model_digest_mismatch"
        result["outcome"] = "errored"
        result["differences"] = [{"field": "model_artifact_digest", "reason": "value differs",
                                  "declared": case["model_artifact_digest"],
                                  "observed": model.artifact_digest, "tolerance": None}]
        return result
    try:
        record, saved = _observe(model, case["sample"], output_dir)
    except (OSError, ValueError) as exc:
        result["observed_outcome"] = "refused"
        result["refusal_reason"] = str(exc)
        if case["expect"]["outcome"] != "refused":
            result["differences"] = [{"field": "outcome", "reason": "the adapter refused this case",
                                      "declared": "evaluated", "observed": str(exc),
                                      "tolerance": None}]
            return result
        needle = case["expect"]["reason_contains"]
        if needle not in str(exc):
            result["differences"] = [{"field": "reason_contains",
                                      "reason": "refusal reason does not contain the declared text",
                                      "declared": needle, "observed": str(exc), "tolerance": None}]
            return result
        result["outcome"] = "reproduced"
        return result
    result["observed_outcome"] = "evaluated"
    result["code"] = record["code"]
    result["record_digest"] = record["record_digest"]
    result["saved_file"] = saved
    observed = _expectation(record)
    result["observed_expectation"] = observed
    if case["expect"]["outcome"] != "evaluated":
        result["differences"] = [{"field": "outcome",
                                  "reason": "the case was expected to be refused",
                                  "declared": "refused", "observed": record["code"],
                                  "tolerance": None}]
        return result
    declared = dict(case["expect"])
    result["record_digest_matches"] = declared["record_digest"] == record["record_digest"]
    if not enforce_digest:
        declared.pop("record_digest")
        observed = {key: value for key, value in observed.items() if key != "record_digest"}
    differences: list[dict[str, Any]] = []
    _compare(declared, observed, "", fields, differences)
    result["differences"] = differences
    result["outcome"] = "reproduced" if not differences else "mismatched"
    return result


def check_corpus(path: str | Path, *, output_dir: str | Path | None = None,
                 case_ids: list[str] | None = None) -> dict[str, Any]:
    """Reproduce a sealed corpus. This never writes an expectation."""
    path = Path(path)
    document = load_corpus(path, sealed=True)
    observed_runtime = engine.runtime_identity()
    report: dict[str, Any] = {
        "report_schema": REPORT_SCHEMA,
        "corpus_file": str(path),
        "corpus_id": document["corpus_id"],
        "corpus_version": document["corpus_version"],
        "corpus_digest": document["corpus_digest"],
        "declared_runtime": document["runtime"],
        "observed_runtime": observed_runtime,
        "runtime_binding": list(document["runtime_binding"]),
        "tolerance_policy": document["tolerance_policy"],
        "unused_tolerances": unused_tolerances(document),
        "digest_enforcement": document["tolerance_policy"]["digest_enforcement"],
        "selected_cases": None if case_ids is None else sorted(set(case_ids)),
        "case_count": len(document["cases"]),
        "cases": [],
        "outcome_counts": {"reproduced": 0, "mismatched": 0, "errored": 0, "skipped": 0},
        "status": "stale",
    }
    report.update(_runtime_report(document, observed_runtime))
    if report["runtime_status"] == "stale":
        # A stale pin makes every expectation unreproducible by construction.
        # Running the cases anyway would report mismatches with the wrong cause.
        report["outcome_counts"]["skipped"] = len(document["cases"])
        return report
    if case_ids is not None:
        known = {case["case_id"] for case in document["cases"]}
        unknown = sorted(set(case_ids) - known)
        _require(not unknown, f"no such case: {', '.join(unknown)}")
    fields = _tolerance_fields(document["tolerance_policy"])
    enforce = document["tolerance_policy"]["digest_enforcement"] == "enforced"
    directory = None if output_dir is None else Path(output_dir)
    for case in document["cases"]:
        if case_ids is not None and case["case_id"] not in case_ids:
            report["outcome_counts"]["skipped"] += 1
            continue
        outcome = _run_case(case, path.parent, fields, directory, enforce)
        report["cases"].append(outcome)
        report["outcome_counts"][outcome["outcome"]] += 1
    counts = report["outcome_counts"]
    report["status"] = "reproduced" if not (counts["mismatched"] or counts["errored"]) else "mismatched"
    return report


def build_corpus(document: Mapping[str, Any], root: Path,
                 output_dir: Path | None = None) -> dict[str, Any]:
    """Re-observe every case and return a sealed corpus recording the outcome."""
    observed_runtime = engine.runtime_identity()
    cases = []
    for case in document["cases"]:
        model = engine.load_model(root / case["model_file"])
        entry = {key: case[key] for key in
                 ("case_id", "purpose", "description", "model_file")}
        entry["model_artifact_digest"] = model.artifact_digest
        entry["sample"] = case["sample"]
        declared = case.get("expect") or {}
        try:
            record, _ = _observe(model, case["sample"], output_dir)
        except (OSError, ValueError) as exc:
            needle = declared.get("reason_contains")
            if declared.get("outcome") == "refused" and needle and needle in str(exc):
                entry["expect"] = {"outcome": "refused", "reason_contains": needle}
            else:
                entry["expect"] = {"outcome": "refused", "reason_contains": str(exc)}
        else:
            entry["expect"] = _expectation(record)
        cases.append(entry)
    sealed = {
        "corpus_schema": CORPUS_SCHEMA,
        "corpus_id": document["corpus_id"],
        "corpus_version": document["corpus_version"],
        "description": document["description"],
        "runtime": observed_runtime,
        "runtime_binding": list(document["runtime_binding"]),
        "tolerance_policy": document["tolerance_policy"],
        "cases": cases,
    }
    sealed["corpus_digest"] = corpus_digest(sealed)
    return sealed


def _changes(previous: Mapping[str, Any] | None, sealed: Mapping[str, Any],
             fields: Mapping[str, Any]) -> list[dict[str, Any]]:
    if previous is None:
        return [{"case_id": case["case_id"], "change": "added", "differences": []}
                for case in sealed["cases"]]
    before = {case["case_id"]: case for case in previous["cases"]}
    after = {case["case_id"]: case for case in sealed["cases"]}
    changes = []
    for case_id in sorted(set(before) | set(after)):
        if case_id not in before:
            changes.append({"case_id": case_id, "change": "added", "differences": []})
            continue
        if case_id not in after:
            changes.append({"case_id": case_id, "change": "removed", "differences": []})
            continue
        differences: list[dict[str, Any]] = []
        _compare(before[case_id], after[case_id], "", fields, differences)
        if differences:
            changes.append({"case_id": case_id, "change": "changed", "differences": differences})
    for key in ("runtime", "tolerance_policy", "runtime_binding", "description",
                "corpus_id", "corpus_version"):
        differences = []
        _compare(previous.get(key), sealed[key], key, fields, differences)
        if differences:
            changes.append({"case_id": None, "change": f"{key} changed",
                            "differences": differences})
    return changes


def record_corpus(path: str | Path, output: str | Path, *, accept_changes: bool = False,
                  output_dir: str | Path | None = None) -> dict[str, Any]:
    """Re-record expectations, reporting every change before it is written.

    Writing is refused when the destination already holds a different corpus,
    unless the caller accepts the changes explicitly. Expectations are never
    refreshed by ``check``.
    """
    path, output = Path(path), Path(output)
    document = load_corpus(path, sealed=False)
    fields = _tolerance_fields(document["tolerance_policy"])
    sealed = build_corpus(document, path.parent,
                          None if output_dir is None else Path(output_dir))
    existing = None
    if output.exists():
        existing = load_corpus(output, sealed=True)
    elif document.get("corpus_digest") is not None:
        existing = document
    changes = _changes(existing, sealed, fields)
    report = {
        "report_schema": "ciw-plsr-corpus-record-v1",
        "corpus_file": str(path), "output_file": str(output),
        "corpus_digest": sealed["corpus_digest"],
        "previous_corpus_digest": None if existing is None else existing["corpus_digest"],
        "case_count": len(sealed["cases"]), "change_count": len(changes),
        "changes": changes, "accept_changes": bool(accept_changes), "written": False,
    }
    if existing is not None and not changes:
        report["written"] = True  # identical content; the file already holds it
        plsr._write_immutable(output, sealed)
        return report
    if changes and existing is not None and not accept_changes:
        report["written"] = False
        return report
    if accept_changes and output.exists():
        _replace(output, sealed)
    else:
        plsr._write_immutable(output, sealed)
    report["written"] = True
    return report


def _replace(path: Path, value: Any) -> Path:
    """Replace an accepted corpus atomically, having reported what changes."""
    content = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".plsr-", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path
