"""Bounded, resumable evaluation of a declared collection of samples.

Every requested sample is accounted for. Each evaluated sample keeps its own
ordinary ``ciw-plsr-run-v1`` bundle, and a batch index records what became of
all of them, including the ones that were never reached.

Durability is a journal, not the index. One newline-terminated JSON entry is
appended and flushed to disk as each sample finishes, so an interrupted run
loses at most the sample that was in flight. The index is derived from the
journal, and resuming replays the journal rather than the evaluations it
records: retained evidence is never recomputed or rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any

from . import plsr
from . import plsr_engine as engine

BATCH_SCHEMA = "ciw-plsr-batch-v1"
INDEX_SCHEMA = "ciw-plsr-batch-index-v1"
HEADER_SCHEMA = "ciw-plsr-batch-header-v1"
ENTRY_SCHEMA = "ciw-plsr-batch-entry-v1"

JOURNAL_NAME = "journal.jsonl"
INDEX_NAME = "index.json"
RUNS_NAME = "runs"

#: Accounting outcomes, derived from the adapter's presentation categories so
#: there is one vocabulary rather than two. This is bookkeeping about what
#: happened to a request; it is NOT a verdict. ``completed`` means a verdict
#: was retained and the certificate inequality resolvably held at that sample,
#: which includes MARGIN_LOW -- a sample that is not operationally acceptable.
#: The index counts runtime codes and operationally acceptable samples
#: separately, and every entry keeps the raw code beside its outcome.
OUTCOME_OF_CATEGORY = {
    "computationally_acceptable": "completed",
    "margin_shortfall": "completed",
    "certificate_violation": "violated",
    "decrease_not_definite": "violated",
    "certificate_invalid": "violated",
    "numerical_refusal": "refused",
    "outside_declared_domain": "refused",
}
OUTCOMES = ("completed", "refused", "violated", "errored", "unfinished")

_PLAN_FIELDS = {"batch_schema", "batch_id", "description", "model_file", "samples"}
_ITEM_FIELDS = {"sample_id", "sample"}
_ENTRY_FIELDS = {"entry_schema", "sample_id", "outcome", "code", "presentation_category",
                 "inequality_certified", "meets_required_margin", "operationally_acceptable",
                 "model_file", "model_artifact_digest", "evidence_id", "result_id",
                 "record_digest", "saved_file", "reason", "timings_s", "completed_at"}
_TIMING_KEYS = ("engine", "adapter", "evidence", "total")


class BatchError(ValueError):
    """An invalid batch plan, or a saved batch that does not admit this run."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BatchError(f"invalid PLSR batch: {message}")


def _text(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{label} must be a nonempty string")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def batch_digest(plan: Mapping[str, Any]) -> str:
    """SHA-256 of the declared collection, binding a run to the plan it ran."""
    return sha256(_canonical(dict(plan)).encode("ascii")).hexdigest()


def _relative(name: Any, label: str) -> str:
    name = _text(name, label)
    _require("\\" not in name and not Path(name).is_absolute() and ".." not in Path(name).parts,
             f"{label} must be a relative path beside the plan")
    return name


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate a declared collection of samples."""
    plan = plsr._read(Path(path))
    _require(isinstance(plan, Mapping) and set(plan) == _PLAN_FIELDS, "plan fields")
    _require(plan["batch_schema"] == BATCH_SCHEMA, f"batch_schema must be {BATCH_SCHEMA!r}")
    _text(plan["batch_id"], "batch_id")
    _text(plan["description"], "description")
    _relative(plan["model_file"], "model_file")
    samples = plan["samples"]
    _require(type(samples) is list and bool(samples), "a batch needs at least one sample")
    seen: set[str] = set()
    for item in samples:
        _require(isinstance(item, Mapping)
                 and set(item) in (_ITEM_FIELDS, _ITEM_FIELDS | {"model_file"}),
                 "each sample needs sample_id and sample, and may name a model_file")
        sample_id = _text(item["sample_id"], "sample_id")
        _require(sample_id not in seen, f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        _require(isinstance(item["sample"], Mapping), f"{sample_id}: sample must be an object")
        if "model_file" in item:
            _relative(item["model_file"], f"{sample_id}: model_file")
    return dict(plan)


def _model_file(plan: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    return item.get("model_file", plan["model_file"])


def _header(plan: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {"entry_schema": HEADER_SCHEMA, "batch_id": plan["batch_id"],
            "batch_digest": batch_digest(plan), "requested": len(plan["samples"]),
            "runtime": dict(runtime)}


def _read_journal(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    """Return the header, the complete entries, and any partial trailing bytes.

    A run killed mid-write can leave an unterminated final line. Those bytes
    are reported so the caller can discard exactly them; a line that parses but
    does not match the entry contract is corruption, not a torn write.
    """
    if not path.exists():
        return None, [], 0
    data = path.read_bytes()
    complete, _, partial = data.rpartition(b"\n")
    if not _:
        return None, [], len(data)
    lines = [line for line in complete.split(b"\n") if line]
    header: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line.decode("utf-8"), parse_constant=plsr._reject_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise BatchError(f"invalid PLSR batch: journal line {number} is not JSON: {exc}")
        _require(isinstance(value, Mapping), f"journal line {number} is not an object")
        if number == 1:
            _require(value.get("entry_schema") == HEADER_SCHEMA,
                     "the journal must begin with its batch header")
            header = dict(value)
            continue
        _require(set(value) == _ENTRY_FIELDS and value["entry_schema"] == ENTRY_SCHEMA,
                 f"journal line {number} is not a {ENTRY_SCHEMA} entry")
        entries.append(dict(value))
    return header, entries, len(partial)


def _append(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete journal line durably before the next sample starts."""
    line = (_canonical(value) + "\n").encode("utf-8")
    with open(path, "ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _write_index(path: Path, index: Mapping[str, Any]) -> None:
    """Replace the derived index atomically.

    The index is a derived summary, so it is replaced rather than published
    immutably: a resumed run rewrites it. The journal and the run bundles are
    the retained evidence, and neither is ever replaced.
    """
    content = (json.dumps(index, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": None, "min": None, "max": None}
    return {"count": len(values), "total": sum(values), "mean": sum(values) / len(values),
            "min": min(values), "max": max(values)}


def _index(plan: Mapping[str, Any], plan_path: Path, output_dir: Path,
           runtime: Mapping[str, Any], entries: Mapping[str, dict[str, Any]],
           *, started_at: str, finished_at: str, wall_s: float,
           discarded_partial_bytes: int, evaluated_now: int, limit: int | None,
           resumed: bool, interrupted: bool) -> dict[str, Any]:
    rows = []
    counts = dict.fromkeys(OUTCOMES, 0)
    codes: dict[str, int] = {}
    acceptable = 0
    for item in plan["samples"]:
        entry = entries.get(item["sample_id"])
        if entry is None:
            entry = {"entry_schema": ENTRY_SCHEMA, "sample_id": item["sample_id"],
                     "outcome": "unfinished", "code": None, "presentation_category": None,
                     "inequality_certified": None, "meets_required_margin": None,
                     "operationally_acceptable": None,
                     "model_file": _model_file(plan, item), "model_artifact_digest": None,
                     "evidence_id": None, "result_id": None, "record_digest": None,
                     "saved_file": None, "reason": None, "timings_s": None,
                     "completed_at": None}
        rows.append(entry)
        counts[entry["outcome"]] += 1
        if entry["code"] is not None:
            codes[entry["code"]] = codes.get(entry["code"], 0) + 1
        if entry["operationally_acceptable"]:
            acceptable += 1
    timings = {key: _statistics([row["timings_s"][key] for row in rows
                                 if row["timings_s"] is not None])
               for key in _TIMING_KEYS}
    timings["wall"] = wall_s
    index = {
        "index_schema": INDEX_SCHEMA,
        "batch_id": plan["batch_id"],
        "batch_digest": batch_digest(plan),
        "description": plan["description"],
        "plan_file": str(plan_path),
        "output_dir": str(output_dir),
        "runtime": dict(runtime),
        "started_at": started_at,
        "finished_at": finished_at,
        "resumed": resumed,
        "interrupted": interrupted,
        "limit": limit,
        "evaluated_this_run": evaluated_now,
        "discarded_partial_bytes": discarded_partial_bytes,
        "requested": len(plan["samples"]),
        "status": "complete" if counts["unfinished"] == 0 else "incomplete",
        "outcomes": counts,
        "codes": dict(sorted(codes.items())),
        # Counted apart from the outcomes on purpose: `completed` includes
        # MARGIN_LOW, where the inequality held and the declared margin did not.
        "operationally_acceptable": acceptable,
        "timings_s": timings,
        "entries": rows,
    }
    index["index_digest"] = sha256(_canonical(index).encode("ascii")).hexdigest()
    return index


def _entry(sample_id: str, model_file: str, outcome: str, *, record: Mapping[str, Any] | None,
           bundle: Mapping[str, Any] | None, saved_file: str | None, reason: str | None,
           timings: Mapping[str, float] | None, stamp: str) -> dict[str, Any]:
    return {
        "entry_schema": ENTRY_SCHEMA, "sample_id": sample_id, "outcome": outcome,
        "code": None if record is None else record["code"],
        "presentation_category": None if record is None else record["presentation_category"],
        "inequality_certified": None if record is None else record["inequality_certified"],
        "meets_required_margin": None if record is None else record["meets_required_margin"],
        "operationally_acceptable": None if record is None else record["operationally_acceptable"],
        "model_file": model_file,
        "model_artifact_digest": None if record is None else record["model_artifact_digest"],
        "evidence_id": None if bundle is None else bundle["evidence_id"],
        "result_id": None if bundle is None else bundle["result_id"],
        "record_digest": None if record is None else record["record_digest"],
        "saved_file": saved_file, "reason": reason,
        "timings_s": None if timings is None else {key: float(timings[key]) for key in _TIMING_KEYS},
        "completed_at": stamp,
    }


def run_batch(plan_path: str | Path, output_dir: str | Path, *, resume: bool = False,
              limit: int | None = None) -> dict[str, Any]:
    """Evaluate a declared collection, retaining one bundle per result.

    Without ``resume`` an output directory that already holds a journal is
    refused, so a second run never appends to unrelated work. With it, the
    saved journal must have been written for this plan and this runtime pin.
    """
    plan_path, output_dir = Path(plan_path), Path(output_dir)
    _require(limit is None or (type(limit) is int and limit > 0), "limit must be a positive integer")
    plan = load_plan(plan_path)
    runtime = engine.runtime_identity()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RUNS_NAME).mkdir(parents=True, exist_ok=True)
    journal = output_dir / JOURNAL_NAME
    header, recorded, partial = _read_journal(journal)
    if header is None and partial:
        raise BatchError("invalid PLSR batch: the saved journal has no complete header")
    if header is not None:
        _require(resume, f"{journal} already holds a batch; pass resume to continue it")
        expected = _header(plan, runtime)
        _require(header["batch_digest"] == expected["batch_digest"],
                 "the saved journal was written for a different declared collection")
        _require(header["runtime"] == expected["runtime"],
                 "the saved journal was written against a different runtime identity")
        if partial:
            # Exactly the bytes of the line that was in flight when the run
            # stopped. Every complete entry before them is kept.
            with open(journal, "r+b") as stream:
                stream.truncate(journal.stat().st_size - partial)
                stream.flush()
                os.fsync(stream.fileno())
    else:
        _append(journal, _header(plan, runtime))
    entries = {entry["sample_id"]: entry for entry in recorded}
    models: dict[str, Any] = {}
    evaluated = 0
    interrupted = False
    try:
        for item in plan["samples"]:
            if item["sample_id"] in entries:
                continue
            if limit is not None and evaluated >= limit:
                break
            entry = _evaluate_item(plan, item, plan_path.parent, output_dir / RUNS_NAME, models)
            _append(journal, entry)
            entries[entry["sample_id"]] = entry
            evaluated += 1
    except KeyboardInterrupt:
        # The journal already holds every sample that finished. Summarise what
        # was reached rather than losing the accounting with the process.
        interrupted = True
    index = _index(plan, plan_path, output_dir, runtime, entries,
                   started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                   wall_s=time.perf_counter() - started, discarded_partial_bytes=partial,
                   evaluated_now=evaluated, limit=limit, resumed=header is not None,
                   interrupted=interrupted)
    _write_index(output_dir / INDEX_NAME, index)
    return index


def _evaluate_item(plan: Mapping[str, Any], item: Mapping[str, Any], root: Path,
                   runs: Path, models: dict[str, Any]) -> dict[str, Any]:
    sample_id = item["sample_id"]
    model_file = _model_file(plan, item)
    stamp = datetime.now(timezone.utc).isoformat()
    timings: dict[str, float] = {}
    try:
        if model_file not in models:
            models[model_file] = engine.load_model(root / model_file)
        saved = plsr.evaluate_sample(models[model_file], dict(item["sample"]), runs, timings)
    except (OSError, ValueError) as exc:
        return _entry(sample_id, model_file, "errored", record=None, bundle=None,
                      saved_file=None, reason=str(exc), timings=None, stamp=stamp)
    record = saved["bundle"]["record"]
    outcome = OUTCOME_OF_CATEGORY[record["presentation_category"]]
    return _entry(sample_id, model_file, outcome, record=record, bundle=saved["bundle"],
                  saved_file=str(Path(saved["saved_file"]).name), reason=None,
                  timings=timings, stamp=stamp)


def read_index(output_dir: str | Path) -> dict[str, Any]:
    """Read a saved batch index without evaluating anything."""
    path = Path(output_dir) / INDEX_NAME
    index = plsr._read(path)
    _require(isinstance(index, Mapping) and index.get("index_schema") == INDEX_SCHEMA,
             f"{path} is not a {INDEX_SCHEMA} document")
    body = {key: value for key, value in index.items() if key != "index_digest"}
    _require(index["index_digest"] == sha256(_canonical(body).encode("ascii")).hexdigest(),
             "index_digest does not match the saved index")
    return dict(index)
