"""Second-platform figure records: CI runs of ``scripts/check_figures.py`` retained for T158.

Scope: ``.github/workflows/figures.yml`` re-executes every retained figure
task on another platform (``windows-latest``) with ``scripts/check_figures.py``
and uploads its ``figure-check.json`` (``ciw.lab-figure-check.v1``), its
``figure-check.md``, the fresh SVGs and the fresh reports as one artifact zip.
:func:`retain_record` retains that comparison from the downloaded zip under
``<retained>/figure-platforms/<record-id>/``: ``figure-check.json`` and
``figure-check.md`` as the run wrote them, with the CRLF line endings a
Windows run writes turned into the LF the repository stores
(``.gitattributes``), the fresh SVG of every figure it reports as a mismatch
(``fresh/<figure path>``, and no other SVG) byte for byte, since it must hash
to its recorded digest (``.gitattributes`` exempts ``fresh/`` from line-ending
normalization), ``record.json``
(``ciw.lab-figure-platform-record.v1``: the CI provenance, as repository,
workflow, run id and attempt, head commit and the artifact's id, name and
digest, the SHA-256 and line endings of both files as the zip held them, and
the platform block copied from ``figure-check.json``) and ``manifest.json``
(``ciw.lab-figure-platform-manifest.v1``: the SHA-256 and size of every other
file). A zip whose SHA-256 differs from the declared artifact digest is
refused. :func:`inspect_record` recomputes the manifest and the zip's bytes of
both files, checks both schemas and every figure entry, recounts the summary
from the figure list and checks that ``record.json`` carries
``figure-check.json``'s platform. T158 reads a record through the ``figure-platform-record`` provider
role, which ``scripts/check_lab.py`` binds from the repository's latest record;
``ciw lab verify`` and ``ciw lab figure-platform verify`` check every retained
record for integrity.

Non-claims: nothing here re-executes a figure task; a record is the CI run's
comparison as that run recorded it. The artifact digest is checked against the
zip only when the record is retained (the zip, with the fresh reports, is not
kept), and the provenance is as declared by whoever downloaded the artifact:
GitHub's artifact attestation is not checked. The manifest digests are
unkeyed, so a fabricated record that recomputes every digest passes.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import re
import shutil
import zipfile

from .runner import _host_paths, check_name, dumps

DIRECTORY = "figure-platforms"
RECORD_SCHEMA = "ciw.lab-figure-platform-record.v1"
MANIFEST_SCHEMA = "ciw.lab-figure-platform-manifest.v1"
# Written by scripts/check_figures.py (its RECORD_SCHEMA).
CHECK_SCHEMA = "ciw.lab-figure-check.v1"
CHECK_FILE, SUMMARY_FILE = "figure-check.json", "figure-check.md"
RECORD_FILE, MANIFEST_FILE = "record.json", "manifest.json"
# Fresh SVGs of mismatched figures: uploaded under run/<figure path>, retained under fresh/<figure path>.
UPLOADED, FRESH = "run", "fresh"
# The outcome scripts/check_figures.py gives a figure the re-executed task writes but the retained report lacks.
NOT_RETAINED = "not retained"
# Outcomes of ciw.lab.research_portfolio.compare_figure (and NOT_RETAINED) by declaration.
OUTCOMES = {
    "undeclared": ("identical", "differs", "not regenerated", NOT_RETAINED),
    "wall_clock_timing": ("identical", "same structure", "structure differs", "not regenerated", NOT_RETAINED),
    "rounding_level": ("identical", "within rounding bounds", "values differ", "structure differs",
                       "not regenerated", NOT_RETAINED),
}
DECLARATIONS = {"undeclared": "an undeclared", "wall_clock_timing": "a wall-clock timing",
                "rounding_level": "a rounding-level"}
CHECK_KEYS = {"schema", "note", "retained", "platform", "providers", "summary", "figures", "not_reexecuted",
              "not_comparable"}
FIGURE_KEYS = {"task_id", "path", "wall_clock_timing", "rounding_level", "outcome", "retained_sha256",
               "fresh_sha256"}
# Kernel-independent identities of a declared figure's retained copy (scripts/check_figures.py records them): its
# series and points, and a rounding-level figure's recorded values with their rounding bounds.
IDENTITY_KEYS = {"retained_structure", "retained_values"}
# The source digests of the code that regenerated a figure there: those its task's retained report records
# (scripts/check_figures.py re-executes the task only when the installed sources have them); T158 compares them with
# its own run's report of the task.
SOURCES_KEY = "task_sources"
SUMMARY_KEYS = ("figure_tasks", "figures", "compared_tasks", "compared_figures", "identical",
                "declared_timing_same_structure", "declared_timing_identical",
                "declared_rounding_level_within_bounds", "declared_rounding_level_identical", "mismatched",
                "not_reexecuted_tasks", "not_reexecuted_figures", "not_comparable_tasks", "not_comparable_figures")
RECORD_KEYS = {"schema", "record_id", "date", "source", "artifact_files", "platform", "note", "limitations"}
SOURCE_KEYS = {"repository", "workflow", "run_id", "run_attempt", "head_sha", "artifact"}
ARTIFACT_KEYS = {"id", "name", "digest"}
# The two text files as the artifact zip held them: their digest and line endings (a Windows run writes CRLF).
TEXT_FILES = (CHECK_FILE, SUMMARY_FILE)
LINE_ENDINGS = {"lf": b"\n", "crlf": b"\r\n"}
CATEGORIES = ("integrity", "record", "figure_check")
LIMITATIONS = [
    "The comparison is the CI run's, as figure-check.json records it; nothing here re-executes a figure task.",
    "The artifact digest was compared with the downloaded zip when the record was retained; the zip is not kept, "
    "and the provenance is as declared by whoever retained it (GitHub's artifact attestation is not checked).",
    "Figures of tasks the CI run did not re-execute (a provider it could not bind, or sources that differed from "
    "the retained run's) are listed with the run's reason and never counted as compared.",
    "Digests are unkeyed: they detect accidental edits, not who produced the record.",
]
NOTE = ("Retained second-platform figure records are verified for integrity only: every retained file matches the "
        "manifest and no unrecorded file is present, figure-check.json and figure-check.md give back the digests "
        "the artifact zip held them with, record.json and figure-check.json follow their schemas and name the same "
        "platform, every figure entry is consistent with its declaration and digests, the summary counts agree with "
        "the figure list, and a fresh SVG is retained exactly for each mismatched figure. Nothing is re-executed; "
        "the digests are unkeyed.")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TASK = re.compile(r"T\d{3}")
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+")
_WORKFLOW = re.compile(r"\.github/workflows/[A-Za-z0-9._-]+\.ya?ml")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_FIGURE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.svg")
_FRESH_PATH = re.compile(FRESH + r"/artifacts/T\d{3}/[A-Za-z0-9][A-Za-z0-9._-]*\.svg")
# platform.platform() spells macOS where platform.system() says Darwin.
_SYSTEMS = {"macos": "Darwin", "darwin": "Darwin", "windows": "Windows", "linux": "Linux"}


def host_system() -> str:
    """The operating system this process runs on, as ``platform.system()`` names it."""
    return platform.system()


def record_system(block) -> str | None:
    """The operating system a figure-check platform block names: its ``system`` (``platform.system()``, recorded by
    scripts/check_figures.py) or else the first word of its ``platform`` string (``Windows-2025Server-...``)."""
    if not isinstance(block, dict):
        return None
    name = block.get("system")
    if not isinstance(name, str) or not name:
        text = block.get("platform")
        name = text.split("-", 1)[0] if isinstance(text, str) and text else None
    return _SYSTEMS.get(name.casefold(), name) if name else None


def platform_identity(block) -> dict:
    """The platform identity T158 reports: OS, platform string, machine, Python, NumPy and OpenBLAS kernel."""
    blas = block.get("blas") if isinstance(block.get("blas"), dict) else {}
    return {"os": record_system(block), "platform": block.get("platform"), "machine": block.get("machine"),
            "python": block.get("python"), "numpy": block.get("numpy"), "openblas_core": blas.get("openblas_core")}


def second_platform_problem(summary, host: str | None = None) -> str | None:
    """Why a valid record is no second platform for a run on ``host`` (default: this process's OS), or None."""
    host = host or host_system()
    system = summary["system"]
    if (system or "").casefold() == host.casefold():
        return (f"the record was made on {system}, the operating system of this run, so it is no second platform")
    return None


def _entry(data: bytes) -> dict:
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def _fresh_path(path: str) -> str:
    return f"{FRESH}/{path}"


def _declaration(figure) -> str:
    if figure.get("wall_clock_timing") is True:
        return "wall_clock_timing"
    return "rounding_level" if figure.get("rounding_level") is True else "undeclared"


def mismatch_outcomes() -> tuple:
    """Outcomes that refute byte reproducibility: compare_figure's mismatches and :data:`NOT_RETAINED`."""
    from .research_portfolio import MISMATCHES
    return (*MISMATCHES, NOT_RETAINED)


def _values_problem(values) -> bool:
    return not isinstance(values, list) or not all(
        isinstance(series, dict) and set(series) == {"name", "x", "y", "bound"} and isinstance(series["name"], str)
        and all(isinstance(series[key], list) and len(series[key]) == len(series["x"])
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in series[key]) for key in ("x", "y", "bound")) for series in values)


def _figure_problems(index: int, figure) -> list:
    """What is inconsistent in one figure entry of figure-check.json."""
    where = f"{CHECK_FILE} figures[{index}]"
    if not isinstance(figure, dict):
        return [f"{where} is not an object"]
    problems = []
    missing, unknown = FIGURE_KEYS - set(figure), set(figure) - FIGURE_KEYS - IDENTITY_KEYS - {SOURCES_KEY}
    if missing or unknown:
        problems.append(f"{where} fields differ from {CHECK_SCHEMA}: missing {sorted(missing)}, unknown "
                        f"{sorted(unknown)}")
        return problems
    task, path = figure["task_id"], figure["path"]
    if not isinstance(task, str) or not _TASK.fullmatch(task):
        problems.append(f"{where} task_id {task!r} is not a task identity")
    parts = path.split("/") if isinstance(path, str) else []
    if len(parts) != 3 or parts[:2] != ["artifacts", task] or not _FIGURE_NAME.fullmatch(parts[2]):
        problems.append(f"{where} path {path!r} is not an SVG artifact of its task")
    timing, rounding = figure["wall_clock_timing"], figure["rounding_level"]
    if not isinstance(timing, bool) or not isinstance(rounding, bool) or (timing and rounding):
        problems.append(f"{where} declares {timing!r} wall-clock timing and {rounding!r} rounding level")
        return problems
    outcome, declaration = figure["outcome"], _declaration(figure)
    if outcome not in OUTCOMES[declaration]:
        problems.append(f"{where} outcome {outcome!r} is not an outcome of {DECLARATIONS[declaration]} figure")
    retained, fresh = figure["retained_sha256"], figure["fresh_sha256"]
    for key, digest, absent in (("retained_sha256", retained, NOT_RETAINED),
                                ("fresh_sha256", fresh, "not regenerated")):
        if (digest is None) != (outcome == absent) or (digest is not None and not (
                isinstance(digest, str) and _HEX64.fullmatch(digest))):
            problems.append(f"{where} {key} is {digest!r} for outcome {outcome!r}")
    if isinstance(retained, str) and isinstance(fresh, str) and (outcome == "identical") != (retained == fresh):
        problems.append(f"{where} outcome {outcome!r} disagrees with its retained and fresh digests")
    structure = figure.get("retained_structure")
    if "retained_structure" in figure and (declaration == "undeclared" or not (
            isinstance(structure, dict) and set(structure) == {"series", "points"}
            and all(type(structure[key]) is int and structure[key] >= 0 for key in structure))):
        problems.append(f"{where} retained_structure is not the series and points of a declared figure")
    if "retained_values" in figure and (not rounding or _values_problem(figure["retained_values"])):
        problems.append(f"{where} retained_values are not the recorded values of a rounding-level figure")
    sources = figure.get(SOURCES_KEY)
    if SOURCES_KEY in figure and not (isinstance(sources, dict) and sources and all(
            isinstance(name, str) and name.startswith("src/ciw/") and isinstance(digest, str)
            and _HEX64.fullmatch(digest) for name, digest in sources.items())):
        problems.append(f"{where} {SOURCES_KEY} are not the source digests of its task's retained report")
    return problems


def recount(check: dict, figure_count) -> dict:
    """The summary counts that follow from figure-check.json's figure list and task listings.

    ``figure_count(task_ids)`` counts the retained figures of those tasks where the retained reports are at hand
    (scripts/check_figures.py); :func:`inspect_record` has only the record, so it passes None and the figure
    totals are checked for consistency instead.
    """
    figures = check["figures"]
    mismatches = mismatch_outcomes()

    def count(declaration, outcome):
        return sum(_declaration(f) == declaration and f["outcome"] == outcome for f in figures)

    counts = {"compared_tasks": len({f["task_id"] for f in figures}),
              "compared_figures": sum(f["outcome"] != NOT_RETAINED for f in figures),
              "identical": count("undeclared", "identical"),
              "declared_timing_same_structure": count("wall_clock_timing", "same structure"),
              "declared_timing_identical": count("wall_clock_timing", "identical"),
              "declared_rounding_level_within_bounds": count("rounding_level", "within rounding bounds"),
              "declared_rounding_level_identical": count("rounding_level", "identical"),
              "mismatched": sum(f["outcome"] in mismatches for f in figures),
              "not_reexecuted_tasks": len(check["not_reexecuted"]),
              "not_comparable_tasks": len(check["not_comparable"])}
    if figure_count is not None:
        counts.update(figure_tasks=counts["compared_tasks"] + counts["not_reexecuted_tasks"]
                      + counts["not_comparable_tasks"],
                      figures=figure_count({f["task_id"] for f in figures}) + figure_count(check["not_reexecuted"])
                      + figure_count(check["not_comparable"]),
                      not_reexecuted_figures=figure_count(check["not_reexecuted"]),
                      not_comparable_figures=figure_count(check["not_comparable"]))
    return counts


def _check_figure_check(check, problem) -> list:
    """Schema, figure entries and summary of figure-check.json; the mismatched figures' paths and fresh digests."""
    if not isinstance(check, dict) or check.get("schema") != CHECK_SCHEMA:
        problem("figure_check", f"{CHECK_FILE} is not {CHECK_SCHEMA}")
        return []
    if set(check) != CHECK_KEYS:
        problem("figure_check", f"{CHECK_FILE} fields differ from {CHECK_SCHEMA}: {sorted(set(check) ^ CHECK_KEYS)}")
        return []
    block = check["platform"]
    if not isinstance(block, dict) or not all(isinstance(block.get(key), str) and block[key]
                                              for key in ("platform", "python", "numpy")) \
            or not isinstance(block.get("blas"), dict) or record_system(block) is None:
        problem("figure_check", f"{CHECK_FILE} does not name its platform, Python, NumPy and BLAS")
    elif not isinstance(block.get("ciw"), dict) or not _HEX64.fullmatch(str(block["ciw"].get("package_digest"))):
        problem("figure_check", f"{CHECK_FILE} does not name the digest of the CIW package the run executed")
    if not isinstance(check["providers"], list) or not all(isinstance(role, str) for role in check["providers"]):
        problem("figure_check", f"{CHECK_FILE} providers is not a list of roles")
    listings = {}
    for key in ("not_reexecuted", "not_comparable"):
        listing = check[key]
        if not isinstance(listing, dict) or not all(isinstance(task, str) and _TASK.fullmatch(task)
                                                    and isinstance(reason, str) and reason.strip()
                                                    for task, reason in listing.items()):
            problem("figure_check", f"{CHECK_FILE} {key} does not map task identities to reasons")
            listing = {}
        listings[key] = set(listing)
    figures = check["figures"]
    if not isinstance(figures, list):
        problem("figure_check", f"{CHECK_FILE} figures is not a list")
        return []
    found = [text for index, figure in enumerate(figures) for text in _figure_problems(index, figure)]
    for text in found:
        problem("figure_check", text)
    if found:
        return []
    paths = [f["path"] for f in figures]
    for path in sorted({p for p in paths if paths.count(p) > 1}):
        problem("figure_check", f"{CHECK_FILE} lists {path} more than once")
    compared = {f["task_id"] for f in figures}
    for first, second in ((compared, listings["not_reexecuted"]), (compared, listings["not_comparable"]),
                          (listings["not_reexecuted"], listings["not_comparable"])):
        for task in sorted(first & second):
            problem("figure_check", f"{CHECK_FILE} lists {task} as compared and left out, or left out twice")
    summary = check["summary"]
    if not isinstance(summary, dict) or set(summary) != set(SUMMARY_KEYS) or not all(
            type(value) is int and value >= 0 for value in summary.values()):
        problem("figure_check", f"{CHECK_FILE} summary does not hold its {len(SUMMARY_KEYS)} counts")
        return []
    for key, value in recount(check, None).items():
        if summary[key] != value:
            problem("figure_check", f"{CHECK_FILE} summary {key} is {summary[key]}; its figure list gives {value}")
    tasks = summary["compared_tasks"] + summary["not_reexecuted_tasks"] + summary["not_comparable_tasks"]
    if summary["figure_tasks"] != tasks:
        problem("figure_check", f"{CHECK_FILE} summary counts {summary['figure_tasks']} figure tasks, not the {tasks} "
                                "compared, not re-executed and not comparable")
    total = summary["compared_figures"] + summary["not_reexecuted_figures"] + summary["not_comparable_figures"]
    if summary["figures"] != total:
        problem("figure_check", f"{CHECK_FILE} summary counts {summary['figures']} figures, not the {total} compared, "
                                "not re-executed and not comparable")
    mismatches = mismatch_outcomes()
    return [(f["path"], f["fresh_sha256"]) for f in figures if f["outcome"] in mismatches and f["fresh_sha256"]]


def _check_source(source, problem) -> None:
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        problem("record", f"{RECORD_FILE} source fields differ from {RECORD_SCHEMA}")
        return
    if not isinstance(source["repository"], str) or not _REPOSITORY.fullmatch(source["repository"]):
        problem("record", f"{RECORD_FILE} source repository is not OWNER/NAME")
    if not isinstance(source["workflow"], str) or not _WORKFLOW.fullmatch(source["workflow"]):
        problem("record", f"{RECORD_FILE} source workflow is not a path under .github/workflows")
    for key in ("run_id", "run_attempt"):
        value = source[key]
        if not (type(value) is int and value > 0) and not (key == "run_attempt" and value is None):
            problem("record", f"{RECORD_FILE} source {key} is not a positive integer")
    if not isinstance(source["head_sha"], str) or not _HEX40.fullmatch(source["head_sha"]):
        problem("record", f"{RECORD_FILE} source head_sha is not a 40-digit commit")
    artifact = source["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
        problem("record", f"{RECORD_FILE} source artifact fields differ from {RECORD_SCHEMA}")
        return
    if not (type(artifact["id"]) is int and artifact["id"] > 0):
        problem("record", f"{RECORD_FILE} source artifact id is not a positive integer")
    if not isinstance(artifact["name"], str) or not _ARTIFACT_NAME.fullmatch(artifact["name"]):
        problem("record", f"{RECORD_FILE} source artifact name is not an artifact name")
    digest = artifact["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:") or not _HEX64.fullmatch(digest[7:]):
        problem("record", f"{RECORD_FILE} source artifact digest is not sha256:<64 hex digits>")


def _check_record(record, record_id: str, check, problem) -> None:
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        problem("record", f"{RECORD_FILE} is not {RECORD_SCHEMA}")
        return
    if set(record) != RECORD_KEYS:
        problem("record", f"{RECORD_FILE} fields differ from {RECORD_SCHEMA}: {sorted(set(record) ^ RECORD_KEYS)}")
        return
    if record["record_id"] != record_id:
        problem("record", f"{RECORD_FILE} names record {record['record_id']!r}, retained as {record_id!r}")
    if not isinstance(record["date"], str) or not _DATE.fullmatch(record["date"]):
        problem("record", f"{RECORD_FILE} date is not YYYY-MM-DD")
    _check_source(record["source"], problem)
    files = record["artifact_files"]
    if not isinstance(files, dict) or set(files) != set(TEXT_FILES) or not all(
            isinstance(entry, dict) and set(entry) == {"sha256", "bytes", "line_endings"}
            and isinstance(entry["sha256"], str) and _HEX64.fullmatch(entry["sha256"])
            and type(entry["bytes"]) is int and entry["bytes"] >= 0 and entry["line_endings"] in LINE_ENDINGS
            for entry in files.values()):
        problem("record", f"{RECORD_FILE} artifact_files does not give the digest, size and line endings of "
                          f"{' and '.join(TEXT_FILES)} in the artifact")
    if isinstance(check, dict) and record["platform"] != check.get("platform"):
        problem("record", f"{RECORD_FILE} platform differs from the platform {CHECK_FILE} records")
    if not isinstance(record["note"], str):
        problem("record", f"{RECORD_FILE} note is not a sentence")
    if not isinstance(record["limitations"], list) or not all(isinstance(text, str) for text in record["limitations"]):
        problem("record", f"{RECORD_FILE} limitations is not a list of sentences")
    for text in _host_paths(record, RECORD_FILE):
        problem("record", text)


def stored_text(raw: bytes, name: str) -> tuple[bytes, str]:
    """A text file of the artifact as the record stores it, LF line endings, and the line endings it had there."""
    text = raw.replace(b"\r\n", b"\n")
    if b"\r" in text:
        raise ValueError(f"{name} holds a carriage return outside a CRLF line ending")
    if text == raw:
        return text, "lf"
    if text.replace(b"\n", b"\r\n") != raw:
        raise ValueError(f"{name} mixes LF and CRLF line endings")
    return text, "crlf"


def _check_text_files(record, contents: dict, problem) -> None:
    """Each retained text file, given back its recorded line endings, must be the bytes the artifact zip held."""
    files = record.get("artifact_files") if isinstance(record, dict) else None
    for name in TEXT_FILES:
        entry = files.get(name) if isinstance(files, dict) else None
        data = contents.get(name)
        if data is None or not isinstance(entry, dict) or entry.get("line_endings") not in LINE_ENDINGS:
            continue
        if b"\r" in data:
            problem("integrity", f"{name} holds a carriage return; the record stores it with LF line endings")
        elif _entry(data.replace(b"\n", LINE_ENDINGS[entry["line_endings"]])) != {
                "sha256": entry.get("sha256"), "bytes": entry.get("bytes")}:
            problem("integrity", f"{name} does not give back the bytes {RECORD_FILE} records for it in the artifact")


def _files(directory: Path, problem) -> tuple[dict, bytes | None, set]:
    """The record's files by retained path, checked against manifest.json; the manifest bytes; the listed paths."""
    try:
        raw = (directory / MANIFEST_FILE).read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        problem("integrity", f"{MANIFEST_FILE} unreadable: {type(exc).__name__}")
        return {}, None, set()
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        problem("integrity", f"{MANIFEST_FILE} is not {MANIFEST_SCHEMA}")
        return {}, raw, set()
    if manifest.get("record_id") != directory.name:
        problem("integrity", f"{MANIFEST_FILE} names record {manifest.get('record_id')!r}, retained as "
                             f"{directory.name!r}")
    listed = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    contents = {}
    for path, entry in sorted(listed.items()):
        if not (path in (RECORD_FILE, CHECK_FILE, SUMMARY_FILE) or _FRESH_PATH.fullmatch(path)) \
                or not isinstance(entry, dict):
            problem("integrity", f"{MANIFEST_FILE} lists {path!r}, which is not a file of a figure-platform record")
            continue
        location = directory / path
        if location.is_symlink() or not location.is_file():
            problem("integrity", f"{path} is missing or is a link, not retained bytes")
            continue
        data = location.read_bytes()
        if {"sha256": entry.get("sha256"), "bytes": entry.get("bytes")} != _entry(data):
            problem("integrity", f"{path} differs from its manifest digest")
            continue
        contents[path] = data
    for path in (RECORD_FILE, CHECK_FILE, SUMMARY_FILE):
        if path not in listed:
            problem("integrity", f"{path} is missing from {MANIFEST_FILE}")
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            problem("integrity", f"{relative} is a link")
        elif path.is_file() and relative != MANIFEST_FILE and relative not in listed:
            problem("integrity", f"{relative} is not recorded in {MANIFEST_FILE}")
    return contents, raw, set(listed)


def _parsed(contents: dict, name: str, problem, category: str):
    if name not in contents:
        return None
    try:
        return json.loads(contents[name])
    except ValueError as exc:
        problem(category, f"{name} is not JSON ({exc})")
        return None


def inspect_record(directory) -> dict:
    """Integrity of one retained second-platform figure record, with every problem by category.

    Returns ``{"record_id", "problems", "categories", "files", "summary"}``:
    ``categories`` maps each of :data:`CATEGORIES` to its problems, and
    ``summary`` (the provenance, platform and comparison T158 reads) is None
    unless the record has no problem. Nothing is executed.
    """
    directory = Path(directory)
    found = {category: [] for category in CATEGORIES}

    def problem(category, text):
        found[category].append(text)

    try:
        check_name(directory.name, "figure-platform record identity")
    except ValueError as exc:
        problem("record", str(exc))
    if not directory.is_dir():
        problem("integrity", "the record directory does not exist")
        contents, manifest, listed = {}, None, set()
    else:
        contents, manifest, listed = _files(directory, problem)
    check = _parsed(contents, CHECK_FILE, problem, "figure_check")
    record = _parsed(contents, RECORD_FILE, problem, "record")
    mismatched = []
    if CHECK_FILE in contents:
        mismatched = _check_figure_check(check, problem)
        for text in _host_paths(check, CHECK_FILE) if isinstance(check, dict) else []:
            problem("figure_check", text)
    if RECORD_FILE in contents:
        _check_record(record, directory.name, check, problem)
        _check_text_files(record, contents, problem)
    fresh = {path[len(FRESH) + 1:]: data for path, data in contents.items() if path.startswith(FRESH + "/")}
    expected = dict(mismatched)
    # A fresh SVG the manifest lists but that failed its digest is already reported as differing or missing.
    for path in sorted(set(expected) - set(fresh)):
        if _fresh_path(path) not in listed:
            problem("integrity", f"the fresh SVG of mismatched figure {path} is not retained")
    for path, data in sorted(fresh.items()):
        if path not in expected:
            problem("integrity", f"{_fresh_path(path)} is retained although {path} did not mismatch")
        elif sha256(data).hexdigest() != expected[path]:
            problem("integrity", f"{_fresh_path(path)} is not the fresh figure {CHECK_FILE} records")
    problems = [f"{category}: {text}" for category in CATEGORIES for text in found[category]]
    summary = None
    if not problems:
        summary = {"record_id": directory.name, "date": record["date"], "source": record["source"],
                   "platform": check["platform"], "system": record_system(check["platform"]),
                   "identity": platform_identity(check["platform"]), "providers": check["providers"],
                   "counts": check["summary"], "figures": check["figures"],
                   "not_reexecuted": check["not_reexecuted"], "not_comparable": check["not_comparable"],
                   "fresh_figures": sorted(fresh), "manifest_sha256": sha256(manifest).hexdigest()}
    return {"record_id": directory.name, "problems": problems, "categories": found, "files": len(contents),
            "summary": summary}


def record_directories(retained) -> list:
    """Every record directory under ``<retained>/figure-platforms`` (its README is not a record)."""
    root = Path(retained) / DIRECTORY
    return [path for path in sorted(root.iterdir()) if path.is_dir()] if root.is_dir() else []


def verify_records(retained) -> dict:
    """Integrity of every record under ``<retained>/figure-platforms``; nothing is re-executed."""
    records = []
    for directory in record_directories(retained):
        inspected = inspect_record(directory)
        summary = inspected["summary"] or {}
        records.append({"record_id": inspected["record_id"], "files": inspected["files"], "date": summary.get("date"),
                        "platform": summary.get("platform", {}).get("platform"), "problems": inspected["problems"]})
    problems = [f"{DIRECTORY}/{record['record_id']}: {problem}" for record in records for problem in record["problems"]]
    return {"verified": len(records), "records": records, "problems": problems, "passed": not problems, "note": NOTE}


def retain_record(zip_path, retained, *, repository: str, workflow: str, run_id: int, head_sha: str,
                  artifact_id: int, artifact_name: str, artifact_digest: str, run_attempt: int | None = None,
                  record_id: str | None = None, date: str | None = None) -> dict:
    """Retain the figure check in a downloaded CI artifact zip as ``<retained>/figure-platforms/<record_id>``.

    The zip must hash to ``artifact_digest`` (``sha256:<hex>``, as GitHub reports it) and hold
    ``figure-check.json`` and ``figure-check.md`` at its root. Only those two files are retained, with LF line
    endings (``record.json`` records their digest and line endings in the zip), and the fresh SVG
    (``run/<figure path>`` in the zip) of each figure the check reports as a mismatch; the fresh reports and the
    other SVGs are left out. ``record_id`` defaults to ``<os>-<run id>`` (``-<attempt>`` after a first attempt)
    and ``date`` to the date of ``figure-check.json`` in the zip. The copy is inspected with
    :func:`inspect_record` before this returns and removed when it has any problem.
    """
    zip_path = Path(zip_path)
    data = zip_path.read_bytes()
    digest = str(artifact_digest)
    if not digest.startswith("sha256:") or not _HEX64.fullmatch(digest[7:]):
        raise ValueError(f"The artifact digest is sha256:<64 hex digits>, not {artifact_digest!r}")
    if sha256(data).hexdigest() != digest[7:]:
        raise ValueError(f"Refusing the artifact: {zip_path.name} hashes to sha256:{sha256(data).hexdigest()}, not the "
                         f"declared artifact digest {digest}")
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{zip_path.name} is not a zip archive") from exc
    with archive:
        names = set(archive.namelist())
        missing = [name for name in (CHECK_FILE, SUMMARY_FILE) if name not in names]
        if missing:
            raise ValueError(f"The artifact holds no {', '.join(missing)} at its root; it is not a figure-check upload")
        files, artifact_files = {}, {}
        for name in TEXT_FILES:
            raw = archive.read(name)
            files[name], endings = stored_text(raw, name)
            artifact_files[name] = {**_entry(raw), "line_endings": endings}
        stamp = archive.getinfo(CHECK_FILE).date_time
        try:
            check = json.loads(files[CHECK_FILE])
        except ValueError as exc:
            raise ValueError(f"{CHECK_FILE} in the artifact is not JSON") from exc
        found = {category: [] for category in CATEGORIES}
        mismatched = _check_figure_check(check, lambda category, text: found[category].append(text))
        if any(found.values()):
            raise ValueError(f"Refusing the artifact's {CHECK_FILE}: " + "; ".join(sum(found.values(), [])[:5]))
        if not check["summary"]["compared_figures"]:
            raise ValueError(f"Refusing the artifact: its {CHECK_FILE} compared no figure")
        paths = _host_paths(check, CHECK_FILE)
        if paths:
            raise ValueError(f"Refusing the artifact's {CHECK_FILE}: " + "; ".join(paths[:5]))
        for path, fresh in mismatched:
            member = f"{UPLOADED}/{path}"
            if member not in names:
                raise ValueError(f"The artifact holds no fresh SVG {member} of mismatched figure {path}")
            content = archive.read(member)
            if sha256(content).hexdigest() != fresh:
                raise ValueError(f"The artifact's {member} is not the fresh figure {CHECK_FILE} records")
            files[_fresh_path(path)] = content
    system = record_system(check["platform"])
    record_id = check_name(record_id or f"{system.casefold()}-{run_id}"
                           + (f"-{run_attempt}" if run_attempt not in (None, 1) else ""),
                           "figure-platform record identity")
    date = date or f"{stamp[0]:04d}-{stamp[1]:02d}-{stamp[2]:02d}"
    record = {"schema": RECORD_SCHEMA, "record_id": record_id, "date": date,
              "source": {"repository": repository, "workflow": workflow, "run_id": run_id, "run_attempt": run_attempt,
                         "head_sha": head_sha,
                         "artifact": {"id": artifact_id, "name": artifact_name, "digest": digest}},
              "artifact_files": artifact_files,
              "platform": check["platform"], "note": NOTE, "limitations": LIMITATIONS}
    found = {category: [] for category in CATEGORIES}
    _check_record(record, record_id, check, lambda category, text: found[category].append(text))
    if any(found.values()):
        raise ValueError("Refusing the provenance: " + "; ".join(sum(found.values(), [])[:5]))
    destination = Path(retained) / DIRECTORY / record_id
    if destination.exists():
        raise ValueError(f"Figure-platform record {record_id} is already retained at {destination}")
    destination.mkdir(parents=True)
    try:
        files[RECORD_FILE] = dumps(record).encode("utf-8")
        for path, content in files.items():
            (destination / path).parent.mkdir(parents=True, exist_ok=True)
            (destination / path).write_bytes(content)
        manifest = {"schema": MANIFEST_SCHEMA, "record_id": record_id,
                    "files": {path: _entry(content) for path, content in sorted(files.items())},
                    "note": "SHA-256 and size of every file of this record except this manifest"}
        (destination / MANIFEST_FILE).write_bytes(dumps(manifest).encode("utf-8"))
        inspected = inspect_record(destination)
        if inspected["problems"]:
            raise ValueError("The retained copy fails verification: " + "; ".join(inspected["problems"][:5]))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    summary = check["summary"]
    return {"record_id": record_id, "retained": str(destination), "date": date,
            "platform": check["platform"]["platform"],
            "files": len(files), "compared_figures": summary["compared_figures"], "mismatched": summary["mismatched"],
            "not_reexecuted_tasks": summary["not_reexecuted_tasks"], "note": NOTE}
