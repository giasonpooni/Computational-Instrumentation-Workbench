"""Run queue tasks, retain their reports and artifacts, and verify regenerations.

Every task in the queue gets a report on every full run. Unimplemented tasks
and tasks whose hard requirements are unavailable are reported as deferred or
blocked with the reason; an unexpected exception is retained as ``blocked``
with its message instead of being hidden. Nothing here acquires physical data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from .. import __version__
from .evidence import AUTHORITY_DOMAINS, PHYSICAL_DOMAINS, EvidenceRefusal, origin_difference, validate_finding
from .registry import SECTION_MODULES, base_section_modules, load_implementations, load_queue
from .report import FIELDS, FIELD_NAMES, WALL_CLOCK_TIMING, build_report, render_markdown, validate_report

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Retained evidence is committed; one artifact larger than this is refused so
# tables stay summaries (sample, aggregate or truncate long trajectories).
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


def dumps(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def repository_root() -> Path | None:
    """Directory holding the repository's ``examples/`` and ``tests/`` for tasks that read them.

    ``CIW_LAB_REPOSITORY_ROOT`` wins (the clean-room reproduction points it at
    copies); otherwise a source checkout around the package is used. Returns
    None in an installed package without either, and such tasks report blocked.
    """
    declared = os.environ.get("CIW_LAB_REPOSITORY_ROOT")
    if declared:
        return Path(declared)
    candidate = PACKAGE_ROOT.parents[1]
    return candidate if (candidate / "examples").is_dir() else None


def repository_path(*parts: str) -> Path | None:
    """A path under :func:`repository_root`, or None when no repository files are reachable."""
    root = repository_root()
    return None if root is None else root.joinpath(*parts)


def source_digest(relative: str) -> str | None:
    """SHA-256 of a package source file with normalized line endings."""
    if not relative.startswith("src/ciw/"):
        return None
    path = PACKAGE_ROOT / relative[len("src/ciw/"):]
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


OPTIONAL_MODULES = ("scipy", "sympy", "mpmath", "pygeodesic", "potpourri3d")
# Optional modules without a ``__version__``: the metadata of their installed distribution names the version.
DISTRIBUTION_VERSIONED = ("potpourri3d",)


def builtin_identity(changed_files) -> dict:
    """The workbench runtime: sources, interpreter and the optional reference modules that import here.

    An installed optional module that fails on import is recorded under
    ``optional_module_errors`` (it is not present), and one whose version
    cannot be read (``__version__``, or the distribution metadata for those in
    ``DISTRIBUTION_VERSIONED``) as ``unknown (...)``; identifying the runtime
    never aborts a run.
    """
    sources = {name: source_digest(name) for name in changed_files if source_digest(name)}
    identity = {"implementation": "ciw.lab", "ciw_version": __version__,
                "python": platform.python_version(), "numpy": np.__version__, "sources": sources}
    errors = {}
    for optional in OPTIONAL_MODULES:
        try:
            if importlib.util.find_spec(optional) is None:
                continue
            module = importlib.import_module(optional)
        except Exception as exc:  # recorded, never raised
            errors[optional] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            identity[optional] = str(importlib.metadata.version(optional) if optional in DISTRIBUTION_VERSIONED
                                     else module.__version__)
        except Exception as exc:
            identity[optional] = f"unknown ({type(exc).__name__}: {exc})"
    if errors:
        identity["optional_module_errors"] = errors
    return identity


# Runtime caches a provider run may create inside its checkout; they never shadow tracked sources.
RUNTIME_CACHES = (".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


def git_identity(path: Path) -> dict:
    """HEAD, tree and cleanliness of a provider checkout; refuses non-repositories.

    Untracked files make the checkout dirty, ignored ones included: an
    untracked module can shadow pinned tracked code on import whatever
    ``.gitignore``, ``.git/info/exclude`` or ``core.excludesFile`` say. So does
    a tracked file flagged skip-worktree or assume-unchanged, whose bytes
    ``git status`` never compares (``ls-files -v`` tags it ``S`` or in lower case).
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    caches = [f":(top,exclude,glob)**/{name}/**" for name in RUNTIME_CACHES]
    status = git("status", "--porcelain", "--untracked-files=all", "--ignored", "--", ":(top)", *caches)
    overlooked = [line for line in git("ls-files", "-v", "--", ":(top)").splitlines()
                  if line[:1] == "S" or line[:1].islower()]
    return {"path": str(path), "revision": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(status or overlooked)}


# Capture roles and retained hardware run identities: lower-case kebab-case names.
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
MAX_NAME = 64


def check_name(value, what: str) -> str:
    """A capture role or hardware run identity: lower-case kebab-case, at most 64 characters."""
    if not isinstance(value, str) or len(value) > MAX_NAME or not NAME.fullmatch(value):
        raise ValueError(f"A {what} is lower-case kebab-case of at most {MAX_NAME} characters "
                         f"(for example rtx2080-2026-10-01), not {value!r}")
    return value


def _readable(path) -> bool:
    """Whether a bound capture is a regular file this process can read."""
    try:
        if path is None or not Path(path).is_file():
            return False
        with open(path, "rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


class Context:
    """Per-run services: artifact retention, shared memo, requirement probes and operator captures.

    ``captures`` binds operator capture roles to files (``ciw lab run --capture
    ROLE=PATH``): raw bytes an operator acquired from an instrument. The lab
    retains the bytes it reads and does not authenticate them, so a capture
    never supports a physical label by itself (see :func:`_gate_physical`).
    """

    def __init__(self, output_dir: Path, providers: dict | None = None, captures: dict | None = None):
        self.output_dir = Path(output_dir)
        self.providers = {role: Path(path) for role, path in (providers or {}).items()}
        self.captures = {check_name(role, "capture role"): Path(path) for role, path in (captures or {}).items()}
        self._memo: dict = {}
        self.task_id: str | None = None
        self.artifacts: list = []
        self.hardware: set = set()
        self.captured: dict = {}          # role -> sha256 of the bytes ctx.capture retained in this task
        self.probes: dict = {}
        self._recording: list = []  # the probes of each memo computation in progress

    def begin(self, task_id: str) -> None:
        self.task_id, self.artifacts, self.hardware, self.probes = task_id, [], set(), {}
        self.captured = {}

    def memo(self, key, compute):
        """Share one deterministic computation between tasks in a run.

        Every task that uses the value records the requirement probes its
        computation made, so a report does not depend on which tasks ran
        before it. A replayed hardware probe is not the task's own: a physical
        finding still needs a probe that succeeded in its task.
        """
        if key not in self._memo:
            made: dict = {}
            self._recording.append(made)
            try:
                value = compute()
            finally:
                self._recording.pop()
            self._memo[key] = (value, made)
        value, made = self._memo[key]
        for requirement, present in made.items():
            self._probed(requirement, present)
        return value

    def _probed(self, requirement: str, present: bool) -> None:
        self.probes[requirement] = present
        for made in self._recording:
            made[requirement] = present

    def available(self, requirement: str) -> bool:
        """Whether a requirement is met here; the task's report records each outcome (the latest per requirement)."""
        kind, _, name = requirement.partition(":")
        if kind == "module":
            present = importlib.util.find_spec(name) is not None
        elif kind == "provider":
            present = name in self.providers and self.providers[name].exists()
        elif kind == "tool":
            present = shutil.which(name) is not None
        elif kind == "hardware":
            present = bool(_probe_hardware(name))
            if present:
                self.hardware.add(name)  # a physical finding needs a probe that succeeded in its task
        elif kind == "capture":
            present = _readable(self.captures.get(name))
        else:
            raise ValueError(f"Unsupported lab requirement: {requirement}")
        self._probed(requirement, present)
        return present

    def capture(self, role: str) -> bytes:
        """The bytes bound to capture ``role``, retained as the task's artifact ``capture-<role><suffix>``.

        Probes ``capture:<role>`` in this task and raises ValueError when no
        readable file is bound. The lab retains the bytes and does not
        authenticate them: findings computed from them are computational, and
        a physical finding citing them passes the physical gate only with a
        probe of the role's instrument on this host (:data:`CAPTURE_INSTRUMENTS`)
        that succeeded in this task. A role without such a probe (a CMM, a
        photogrammetry rig) keeps its physical claims ``not_established``.
        """
        if not self.available(f"capture:{role}"):
            raise ValueError(f"No readable capture is bound to role {role}; bind one with --capture {role}=PATH")
        path = self.captures[role]
        data = path.read_bytes()
        suffix = path.suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,10}", path.suffix.lower()) else ""
        self._write(f"capture-{role}{suffix}", data)
        self.captured[role] = hashlib.sha256(data).hexdigest()
        return data

    def _write(self, name: str, data: bytes, wall_clock_timing: bool = False) -> str:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("Artifact names are single file names")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"Artifact {name} exceeds {MAX_ARTIFACT_BYTES} bytes; retain a summary instead")
        if wall_clock_timing and not name.endswith(".svg"):
            raise ValueError(f"Only SVG figures are declared as wall-clock timing figures, not {name}")
        directory = self.output_dir / "artifacts" / self.task_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
        relative = f"artifacts/{self.task_id}/{name}"
        # A rewrite replaces the earlier entry: one entry per path, matching the bytes on disk.
        self.artifacts[:] = [a for a in self.artifacts if a["path"] != relative]
        entry = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if wall_clock_timing:
            entry[WALL_CLOCK_TIMING] = True
        self.artifacts.append(entry)
        return relative

    def artifact_json(self, name: str, value) -> str:
        return self._write(name, dumps(value).encode("utf-8"))

    def artifact_text(self, name: str, text: str, *, wall_clock_timing: bool = False) -> str:
        """Retain a text artifact; ``wall_clock_timing=True`` declares an SVG figure whose bytes depend on wall-clock
        timing, recorded in the report's artifact list and compared for presence and structure only on re-execution."""
        return self._write(name, text.encode("utf-8"), wall_clock_timing)


# Linux powercap sysfs directory holding the intel-rapl energy counters.
POWERCAP = Path("/sys/class/powercap")


def _probe_hardware(name: str) -> bool:
    if name == "nvidia-gpu":
        if shutil.which("nvidia-smi") is None:
            return False
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "GPU" in result.stdout
    if name == "rapl":
        # The counters often exist but are readable only by root: a counter answers only if it reads as a number.
        for path in POWERCAP.glob("intel-rapl:*/energy_uj"):
            try:
                int(path.read_text().strip())
                return True
            except (OSError, ValueError):
                continue
        return False
    return False


# Outcomes of one node (parametrized cases, repeated records) fold by severity:
# any failure wins over a pass, and a pass over a skip, whatever the JUnit order.
SEVERITY = {"skipped": 0, "passed": 1, "failed": 2}


def _fold(outcome: str | None, value: str) -> str:
    return value if outcome is None else max(outcome, value, key=SEVERITY.__getitem__)


def _node_id(classname: str, name: str, root: Path | None) -> str:
    """pytest node id of a JUnit test case.

    The module is the longest dotted prefix of ``classname`` that is a ``.py``
    file under ``root``; the remaining components are test classes. Without the
    sources, classes start at the first later component with pytest's ``Test``
    prefix.
    """
    parts = classname.split(".") if classname else []
    split = None
    if root is not None:
        split = next((i for i in range(len(parts), 0, -1)
                      if root.joinpath(*parts[:i - 1], parts[i - 1] + ".py").is_file()), None)
    if split is None:
        split = next((i for i in range(1, len(parts)) if parts[i].startswith("Test")), len(parts))
    return "::".join(["/".join(parts[:split]) + ".py", *parts[split:], name])


def read_junit(path: Path | None, root: Path | None = None) -> dict:
    """Map pytest node ids to passed, skipped or failed.

    ``root`` is the pytest root directory the class names are relative to
    (default: :func:`repository_root`).
    """
    if path is None:
        return {}
    root = repository_root() if root is None else Path(root)
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = _node_id(case.get("classname", ""), case.get("name", ""), root)
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "failed"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes[node] = _fold(outcomes.get(node), outcome)
    return outcomes


def _test_fields(implementation, findings, junit):
    passed, skipped, failed = [], [], []
    for record in findings:
        for check in record["basis"].get("checks") or []:
            (passed if check.get("passed") else failed).append(f"check: {record['claim']} vs {check['reference']}")
        independent = record["basis"].get("independent_check")
        if independent:
            (passed if independent.get("passed") else failed).append(
                f"independent: {record['claim']} vs {independent['checker']['implementation']}")
    for node in implementation.regression_tests if implementation else ():
        outcome = None
        for key, value in junit.items():
            # The registered node itself or its parametrized cases; never a suffix of another path.
            if key == node or key.startswith(node + "["):
                outcome = _fold(outcome, value)
        if outcome == "passed":
            passed.append(f"pytest: {node}")
        elif outcome == "failed":
            failed.append(f"pytest: {node}")
        else:
            skipped.append(f"pytest: {node} ({'skipped' if outcome == 'skipped' else 'not run in this invocation'})")
    return passed, skipped, failed


def _default_fields(task, reason):
    return {
        "hypothesis": "Not formulated: the task has not been executed.",
        "mathematical_model": "Not stated.",
        "input_data": [],
        "observation_model": "No observation was made.",
        "expected_invariant": "Not stated.",
        "experiment": reason,
        "numerical_result": "none",
        "uncertainty": "not quantified",
        "failure_modes_checked": [],
        "unresolved_assumptions": [reason],
        "recommended_next_task": f"Implement {task['id']}: {task['title']}",
    }


NOT_STATED = "Not stated by the implementation."


def _executed_defaults(task):
    """Answers an executed task's implementation omitted: said to be unstated, never 'not executed'."""
    return {name: NOT_STATED if isinstance(value, str) else [NOT_STATED]
            for name, value in _default_fields(task, "").items()}


def _capture_refusal(role: str, probed) -> str | None:
    """Why bytes of operator capture ``role`` cannot back a physical label given this task's own hardware probes."""
    instrument = CAPTURE_INSTRUMENTS.get(role)
    if instrument is None:
        return (f"its raw bytes are the operator capture {role}, and no probe of its instrument exists on this host "
                "(the lab retains operator captures and does not authenticate them)")
    if instrument not in probed:
        return (f"its raw bytes are the operator capture {role}, and no probe of its instrument succeeded in this "
                f"task (needs hardware:{instrument})")
    return None


def _gate_physical(findings, ctx):
    """Refuse physical labels unless hardware answered a probe in this task and the raw bytes were retained.

    A probe replayed from another task's memo does not count. Bytes an operator
    capture bound (:meth:`Context.capture`) count only with a probe of that
    capture role's instrument that succeeded in this task; a capture by itself
    is retained and unauthenticated, never hardware evidence.
    """
    retained = {artifact["sha256"] for artifact in ctx.artifacts}
    for record in findings:
        if record["domain"] in PHYSICAL_DOMAINS and record["evidence_status"] != "not_established":
            digest = record["basis"]["acquisition"]["raw_sha256"]
            if not ctx.hardware:
                raise EvidenceRefusal(f"Hardware evidence refused for '{record['claim']}': no hardware probe succeeded "
                                      "in this task")
            for role in sorted(role for role, captured in ctx.captured.items() if captured == digest):
                why = _capture_refusal(role, ctx.hardware)
                if why:
                    raise EvidenceRefusal(f"Hardware evidence refused for '{record['claim']}': {why}")
            if digest not in retained:
                raise EvidenceRefusal(f"Hardware evidence refused for '{record['claim']}': raw bytes {digest[:12]} are not a retained artifact")


def _blocked(task, reason, failure):
    fields = _default_fields(task, reason)
    fields["failure_modes_checked"] = [failure]
    return "blocked", fields, []


def _with_probes(identity, ctx, state, requires, measured=False):
    """The runtime identity plus the requirement probes the task queried and their outcomes (none: unchanged).

    A blocked task that declares no requirement keeps its identity unchanged:
    the planner retries it only when that identity differs from the current
    built-in one, which recorded probes always would. A task with a
    hardware-measured finding (``measured``) also records the hardware probes
    that succeeded in the task itself, which the physical gate relied on;
    other reports leave them out, so they stay independent of which task
    first computed a shared memo.
    """
    if not ctx.probes or not isinstance(identity, dict) or (state == "blocked" and not requires):
        return identity
    extra = {"requirement_probes": dict(sorted(ctx.probes.items()))}
    if measured and ctx.hardware:  # never a probe replayed from another task's memo
        extra["hardware_probed_in_task"] = sorted(ctx.hardware)
    if ctx.captured:  # the operator captures this task retained, by role and digest (never a host path)
        extra["operator_captures"] = {role: ctx.captured[role] for role in sorted(ctx.captured)}
    return dict(identity, **extra)


def _unavailable(requires, ctx) -> list:
    missing = []
    for need in requires:
        try:
            if not ctx.available(need):
                missing.append(need)
        except Exception as exc:  # an unprobeable requirement cannot be met; the task reports blocked
            missing.append(f"{need} ({type(exc).__name__}: {exc})")
    return missing


def run_task(task, implementation, ctx: Context, junit: dict, import_error: str | None = None) -> dict:
    ctx.begin(task["id"])
    findings, state, executed = [], "deferred", False
    if implementation is None:
        reason = f"Deferred: {import_error or 'no implementation is registered for this task'}."
        fields = _default_fields(task, reason)
    else:
        missing = _unavailable(implementation.requires, ctx)
        if missing:
            reason = "Blocked: unavailable requirement(s) " + ", ".join(missing) + "."
            fields = _default_fields(task, reason)
            fields["unresolved_assumptions"] = [reason]
            state = "blocked"
            plan = getattr(implementation.run, "plan", None)
            if plan:
                fields.update({k: v for k, v in plan.items() if k in FIELD_NAMES})
                fields["experiment"] = reason + " Planned: " + str(plan.get("experiment", ""))
                # A blocked task may record the claims it cannot establish, and nothing else.
                try:
                    findings = [validate_finding(record) for record in plan.get("findings", [])]
                    if any(record["evidence_status"] != "not_established" for record in findings):
                        raise EvidenceRefusal("Planned findings of a blocked task must be not_established")
                except EvidenceRefusal as exc:
                    state, fields, findings = _blocked(task, reason + f" Plan findings refused: {exc}", str(exc))
        else:
            try:
                executed = True
                outcome = implementation.run(ctx)
                state = outcome.get("state", "completed")
                fields = _executed_defaults(task)
                fields.update(outcome.get("fields", {}))
                findings = outcome.get("findings", [])
                for record in findings:
                    validate_finding(record)
                _gate_physical(findings, ctx)
            except Exception as exc:  # retained as a blocked report, never hidden
                reason = f"Blocked by unexpected {type(exc).__name__}: {exc}"
                state, fields, findings = _blocked(task, reason,
                                                   traceback.format_exception_only(type(exc), exc)[-1].strip())
    changed = list(implementation.changed_files) if implementation else []
    requires = implementation.requires if implementation else ()
    fields.setdefault("changed_files", changed)
    if not fields.get("changed_files"):
        fields["changed_files"] = changed
    fields["generated_artifacts"] = deepcopy(ctx.artifacts)
    if not fields.get("provider_runtime_identity"):
        fields["provider_runtime_identity"] = builtin_identity(changed)
    measured = any(f["domain"] in PHYSICAL_DOMAINS and f["evidence_status"] != "not_established" for f in findings)
    fields["provider_runtime_identity"] = _with_probes(fields["provider_runtime_identity"], ctx, state, requires,
                                                       measured)
    # Checks count as tests only when the implementation ran; a blocked plan's checks never executed.
    passed, skipped, failed = _test_fields(implementation, findings if executed else [], junit)
    fields["tests_passed"], fields["tests_skipped"] = passed, skipped
    if failed and state == "completed":
        state = "partial"
    try:
        report = build_report(task, state, {k: v for k, v in fields.items() if k in FIELD_NAMES
                                            and k not in ("evidence_status", "physical_validation_status")},
                              findings, extra={"tests_failed": failed} if failed else None)
        return validate_report(report)
    except (EvidenceRefusal, ValueError, TypeError) as exc:
        # A report the contract refuses is retained as blocked so the queue continues.
        state, fields, _ = _blocked(task, f"Blocked: report refused by the evidence contract: {exc}", str(exc))
        fields.update(changed_files=changed, generated_artifacts=deepcopy(ctx.artifacts),
                      provider_runtime_identity=_with_probes(builtin_identity(changed), ctx, state, requires),
                      tests_passed=[], tests_skipped=[])
        return validate_report(build_report(task, state, {k: v for k, v in fields.items() if k in FIELD_NAMES}, []))


def _section_import_error(section, errors) -> str | None:
    """Import errors that explain a section's unimplemented tasks.

    A packaged section owns exactly its :data:`SECTION_MODULES`; an extension
    section is implemented by the configured modules, whose errors apply to
    every extension task left without an implementation.
    """
    if section.get("extension"):
        names = [name for name in errors if name not in SECTION_MODULES]
    else:
        names = base_section_modules(section["key"])
    return "; ".join(f"{name}: {errors[name]}" for name in sorted(names) if name in errors) or None


# Operator capture variables the energy tasks read (T115, T116, T118, T119), as capture roles: a
# `--capture ROLE=PATH` binding of one of these roles also sets its variable for the run, and a variable
# set by the operator is recorded as that role's binding.
OPERATOR_CAPTURE_VARIABLES = {"energy-log": "CIW_LAB_ENERGY_LOG", "nvidia-smi-csv": "CIW_LAB_NVIDIA_SMI_CSV",
                              "rapl-log": "CIW_LAB_RAPL_LOG"}
# The instrument whose hardware probe must succeed in a task before bytes of a capture role can back a physical
# label there (the task still binds the capture to the probed device's identity, as the energy tasks do with NVML
# and RAPL). A role without an entry has no instrument probe: its physical claims stay not_established until an
# instrument probe or a signed-capture trust anchor exists.
CAPTURE_INSTRUMENTS = {"energy-log": "nvidia-gpu", "nvidia-smi-csv": "nvidia-gpu", "rapl-log": "rapl"}
# Operator settings that qualify a capture; recorded by value (they hold no path).
OPERATOR_SETTINGS = ("CIW_LAB_NVIDIA_SMI_UTC_OFFSET",)
RUN_RECORD = "run-record.json"
RUN_RECORD_SCHEMA = "ciw.lab-run-record.v1"


def operator_captures(captures=None) -> dict:
    """Capture roles of a run: explicit bindings plus the operator capture variables that are set.

    A role bound both ways must name the same file; otherwise the binding is refused.
    """
    bound = {check_name(role, "capture role"): Path(path) for role, path in (captures or {}).items()}
    for role, variable in OPERATOR_CAPTURE_VARIABLES.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            continue
        if role in bound and os.path.abspath(bound[role]) != os.path.abspath(value):
            raise ValueError(f"Capture role {role} is bound to one file and {variable} names another; bind one")
        bound.setdefault(role, Path(value))
    return bound


class _OperatorVariables:
    """Set the operator capture variable of each bound role for the run, restoring the caller's afterwards."""

    def __init__(self, captures):
        self.values = {OPERATOR_CAPTURE_VARIABLES[role]: os.path.abspath(path)
                       for role, path in captures.items() if role in OPERATOR_CAPTURE_VARIABLES}
        self.saved: dict = {}

    def __enter__(self):
        for variable, value in self.values.items():
            self.saved[variable] = os.environ.get(variable)
            os.environ[variable] = value
        return self

    def __exit__(self, *exc):
        for variable, value in self.saved.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value


def _file_identity(path: Path) -> dict:
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def binding_identity(path) -> dict:
    """A provider binding as digests only: a Git checkout's revision and tree, or a file's sha256 (never its path)."""
    path = Path(path)
    try:
        if path.is_dir():
            try:
                identity = git_identity(path)
            except (OSError, subprocess.CalledProcessError):
                return {"kind": "directory", "digest": None}
            return {"kind": "git_checkout", "revision": identity["revision"], "source_tree": identity["source_tree"],
                    "dirty": identity["dirty"]}
        if path.is_file():
            return {"kind": "file", **_file_identity(path)}
    except OSError as exc:
        return {"kind": "unreadable", "error": type(exc).__name__}
    return {"kind": "missing"}


def package_digest() -> str:
    """SHA-256 over the imported ``ciw`` package files (relative path and normalized bytes), caches excluded."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        content = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        digest.update(f"{path.relative_to(PACKAGE_ROOT).as_posix()}\0{content}\n".encode("utf-8"))
    return digest.hexdigest()


def _ciw_identity() -> dict:
    identity = {"version": __version__, "package_digest": package_digest()}
    try:  # the clean-room marker names the wheel this package was installed from
        wheel = json.loads(os.environ.get("CIW_LAB_CLEAN_ROOM") or "{}").get("wheel_sha256")
    except (ValueError, AttributeError):
        wheel = None
    if isinstance(wheel, str) and re.fullmatch(r"[0-9a-f]{64}", wheel):
        identity["wheel_sha256"] = wheel
    return identity


def run_record(started_utc: str, task_ids, providers, captures) -> dict:
    """The bindings of one run as role names and digests (``run-record.json``); it holds no host path."""
    capture_rows = {}
    for role, path in sorted(captures.items()):
        row = _file_identity(path) if _readable(path) else {"readable": False}
        row["variable"] = OPERATOR_CAPTURE_VARIABLES.get(role)
        capture_rows[role] = row
    return {"schema": RUN_RECORD_SCHEMA,
            "note": "Bindings of this run as role names and digests; no host path is recorded",
            "started_utc": started_utc, "ciw": _ciw_identity(), "python": platform.python_version(),
            "tasks": list(task_ids),
            "providers": {role: binding_identity(path) for role, path in sorted((providers or {}).items())},
            "captures": capture_rows,
            "settings": {name: os.environ[name] for name in OPERATOR_SETTINGS if os.environ.get(name)}}


def run_queue(output_dir, task_ids=None, providers=None, junit_path=None, budget_seconds=None, captures=None) -> dict:
    """Run selected tasks (default: all) in queue order and retain their reports.

    Elapsed times are written to ``run-log.json`` beside the reports, never into
    them: timing is not reproducible and is not a finding. ``captures`` binds
    operator capture roles to files (see :class:`Context`); ``run-record.json``
    records the run's provider and capture bindings as digests.
    """
    output_dir = Path(output_dir)
    queue = load_queue()
    implementations, errors = load_implementations()
    selected = set(task_ids) if task_ids else None
    unknown = (selected or set()) - {t["id"] for t in queue["tasks"]}
    if unknown:
        raise ValueError(f"Unknown lab task identities: {sorted(unknown)}")
    captures = operator_captures(captures)
    junit = read_junit(Path(junit_path)) if junit_path else {}
    ctx = Context(output_dir, providers, captures)
    reports, timings = [], []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record = run_record(started_utc, [t["id"] for t in queue["tasks"] if selected is None or t["id"] in selected],
                        providers, captures)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    import_errors = {s["section"]: _section_import_error(s, errors) for s in queue["sections"]}
    with _OperatorVariables(captures):
        for item in queue["tasks"]:
            if selected is not None and item["id"] not in selected:
                continue
            artifact_dir = output_dir / "artifacts" / item["id"]
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            error = import_errors[item["section"]]
            started = time.perf_counter()
            report = run_task(item, implementations.get(item["id"]), ctx, junit, error)
            timings.append({"task_id": item["id"], "state": report["state"],
                            "seconds": round(time.perf_counter() - started, 3)})
            (output_dir / "reports" / f"{item['id']}.json").write_text(dumps(report), encoding="utf-8")
            reports.append(report)
    (output_dir / RUN_RECORD).write_text(dumps(record), encoding="utf-8")
    if selected is None:
        write_index(output_dir, queue)
    total = round(sum(t["seconds"] for t in timings), 3)
    over = [t for t in timings if budget_seconds is not None and t["seconds"] > budget_seconds]
    (output_dir / "run-log.json").write_text(dumps({
        "schema": "ciw.lab-run-log.v1", "note": "Elapsed wall-clock seconds; not reproducible and not findings",
        "total_seconds": total, "budget_seconds": budget_seconds, "tasks": timings}), encoding="utf-8")
    summary = {"output_dir": str(output_dir), "tasks": len(reports),
               "states": {s: sum(r["state"] == s for r in reports) for s in ("completed", "partial", "deferred", "blocked")},
               "labels": _label_totals(reports), "total_seconds": total,
               "slowest": sorted(timings, key=lambda t: -t["seconds"])[:5]}
    if budget_seconds is not None:
        summary["over_budget"] = over
    return summary


def _label_totals(reports):
    totals = {}
    for report in reports:
        for label, count in report["evidence_status"]["counts"].items():
            totals[label] = totals.get(label, 0) + count
    return totals


def load_reports(directory) -> list:
    directory = Path(directory) / "reports"
    reports = []
    for path in sorted(directory.glob("T*.json")):
        reports.append(validate_report(json.loads(path.read_text(encoding="utf-8"))))
    return reports


def write_index(output_dir, queue=None) -> None:
    """Write the queue state and the full human-readable report book."""
    output_dir = Path(output_dir)
    queue = queue or load_queue()
    reports = {r["task_id"]: r for r in load_reports(output_dir)}
    state = [{"id": t["id"], "section": t["section_key"], "title": t["title"],
              "state": reports[t["id"]]["state"] if t["id"] in reports else "not_run",
              "evidence_status": reports[t["id"]]["evidence_status"]["primary"] if t["id"] in reports else "not_established",
              "report_id": reports[t["id"]]["report_id"] if t["id"] in reports else None}
             for t in queue["tasks"]]
    (output_dir / "queue-state.json").write_text(dumps({"schema": "ciw.lab-queue-state.v1", "tasks": state}),
                                                 encoding="utf-8")
    lines = ["# Computational experimentalist queue — retained reports", "",
             "Generated by `ciw lab run --all`. Every finding carries one evidence label assigned by",
             "`ciw.lab.evidence`; no report states physical validation without acquired hardware evidence.", "",
             "| Task | Title | State | Primary evidence |", "| --- | --- | --- | --- |"]
    for row in state:
        lines.append(f"| [{row['id']}](#{row['id'].lower()}) | {row['title']} | `{row['state']}` | `{row['evidence_status']}` |")
    for section in queue["sections"]:
        lines += ["", f"## {section['section']}. {section['name']}", ""]
        for t in queue["tasks"]:
            if t["section"] == section["section"] and t["id"] in reports:
                lines += [f'<a id="{t["id"].lower()}"></a>', "", render_markdown(reports[t["id"]])]
    (output_dir / "REPORTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _close(retained, fresh, tolerance):
    if isinstance(retained, bool) or isinstance(fresh, bool):
        return type(retained) is type(fresh) and retained == fresh  # True is not the number 1
    if retained is None or fresh is None or isinstance(retained, str):
        return retained == fresh
    if isinstance(retained, (int, float)) and isinstance(fresh, (int, float)):
        if not (math.isfinite(retained) and math.isfinite(fresh)):
            return retained == fresh
        return abs(retained - fresh) <= tolerance.get("abs", 0.0) + tolerance.get("rel", 0.0) * abs(retained)
    if isinstance(retained, list) and isinstance(fresh, list):
        return len(retained) == len(fresh) and all(_close(a, b, tolerance) for a, b in zip(retained, fresh))
    if isinstance(retained, dict) and isinstance(fresh, dict):
        return retained.keys() == fresh.keys() and all(_close(retained[k], fresh[k], tolerance) for k in retained)
    return retained == fresh


def _optional_difference(old, new) -> str:
    """Optional reference modules present in one run's identity but not the other's."""
    def present(report):
        identity = report.get("provider_runtime_identity") or {}
        return {name for name in OPTIONAL_MODULES if name in identity}
    before, after = present(old), present(new)
    return ", ".join([f"-{m}" for m in sorted(before - after)] + [f"+{m}" for m in sorted(after - before)])


def _probes(report) -> dict:
    identity = report.get("provider_runtime_identity")
    probes = identity.get("requirement_probes") if isinstance(identity, dict) else None
    return probes if isinstance(probes, dict) else {}


def _probe_difference(old, new) -> str:
    """Requirement probes (module:, tool:, provider:, hardware:) whose recorded outcome differs between runs."""
    def outcome(value):
        return {True: "available", False: "unavailable", None: "not recorded"}.get(value, repr(value))
    before, after = _probes(old), _probes(new)
    return ", ".join(f"{name} {outcome(before.get(name))} -> {outcome(after.get(name))}"
                     for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name))


def _environment_note(old, new) -> str:
    """Environment differences between two runs that can explain a changed state or label."""
    notes = [f"{kind} differ: {text}" for kind, text in (("optional modules", _optional_difference(old, new)),
                                                        ("requirement probes", _probe_difference(old, new))) if text]
    return f" ({'; '.join(notes)})" if notes else ""


PROSE_FIELDS = ("hypothesis", "mathematical_model", "input_data", "observation_model", "expected_invariant",
                "experiment", "numerical_result", "uncertainty", "failure_modes_checked",
                "unresolved_assumptions", "recommended_next_task", "physical_validation_status")
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _skeleton(value) -> str:
    """Report prose with numbers masked: wording must match, last digits may differ by platform."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return NUMBER.sub("#", text)


def _witness(counterexample):
    """A counterexample's witness values, compared within the finding's tolerance; its wording by skeleton."""
    return counterexample.get("witness") if isinstance(counterexample, dict) else None


def artifact_problems(directory, tasks=None) -> list:
    """Retained artifacts must still hash to the digests their reports record (optionally for ``tasks`` only)."""
    problems = []
    for report in load_reports(directory):
        if tasks is not None and report["task_id"] not in tasks:
            continue
        for artifact in report["generated_artifacts"]:
            path = Path(directory) / artifact["path"]
            if not path.is_file():
                problems.append(f"{report['task_id']}: artifact {artifact['path']} missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                problems.append(f"{report['task_id']}: artifact {artifact['path']} differs from its recorded digest")
    return problems


def compare(retained_dir, fresh_dir, tasks=None) -> dict:
    """Regression gate: same states, labels, claims, units and wording; values within tolerance.

    Every task retained or regenerated is compared, or only ``tasks`` (the
    identities of a partial run), each of which must then be in both
    directories. A missing or empty retained directory, or an empty
    ``tasks``, fails: comparing nothing verifies nothing. A changed state or
    label names the optional modules and requirement probes whose recorded
    outcomes differ between the two runs.
    """
    selected = None if tasks is None else set(tasks)
    retained_reports = load_reports(retained_dir)
    retained = {r["task_id"]: r for r in retained_reports if selected is None or r["task_id"] in selected}
    fresh = {r["task_id"]: r for r in load_reports(fresh_dir) if selected is None or r["task_id"] in selected}
    problems = []
    if not retained_reports:
        problems.append(f"no retained reports in {Path(retained_dir) / 'reports'}: nothing to verify against")
    if selected is not None and not selected:
        problems.append("no tasks selected: nothing to verify")
    problems += artifact_problems(retained_dir, selected) + artifact_problems(fresh_dir, selected)
    expected = set(retained) | set(fresh) if selected is None else selected
    # New evidence must be reviewed and retained, not slip in through a gate run.
    problems += [f"{task_id}: not retained" for task_id in sorted(expected - set(retained))]
    problems += [f"{task_id}: not regenerated" for task_id in sorted(expected - set(fresh))]
    for task_id, old in retained.items():
        new = fresh.get(task_id)
        if new is None:
            continue
        if old["state"] != new["state"]:
            problems.append(f"{task_id}: state {old['state']} -> {new['state']}" + _environment_note(old, new))
        for name in PROSE_FIELDS:
            if _skeleton(old[name]) != _skeleton(new[name]):
                problems.append(f"{task_id}: '{name}' wording differs")
        old_claims = [f["claim"] for f in old["findings"]]
        new_claims = [f["claim"] for f in new["findings"]]
        if old_claims != new_claims:
            changed = sorted(set(old_claims) ^ set(new_claims)) or ["order"]
            problems.append(f"{task_id}: finding claims differ: {changed[:4]}")
        if len(old["findings"]) != len(new["findings"]):
            problems.append(f"{task_id}: finding count {len(old['findings'])} -> {len(new['findings'])}")
        new_findings = {f["claim"]: f for f in new["findings"]}
        for record in old["findings"]:
            claim, other = record["claim"], new_findings.get(record["claim"])
            if other is None:
                continue
            validate_finding(other)
            if record["evidence_status"] != other["evidence_status"]:
                problems.append(f"{task_id}: '{claim}' label {record['evidence_status']} -> {other['evidence_status']}"
                                + _environment_note(old, new))
            elif change := origin_difference(record, other):
                # With the label unchanged, a different basis origin is its own regression; a label change
                # already reports the basis change behind it, with its environment note.
                problems.append(f"{task_id}: '{claim}' {change}")
            if record.get("unit") != other.get("unit") or record["domain"] != other["domain"]:
                problems.append(f"{task_id}: '{claim}' unit or domain differs")
            if record.get("expected_not_established") != other.get("expected_not_established"):
                problems.append(f"{task_id}: '{claim}' expected_not_established differs")
            tolerance = record.get("regression_tolerance", {"abs": 0.0, "rel": 1e-9})
            if not _close(record["value"], other["value"], tolerance):
                problems.append(f"{task_id}: '{claim}' value outside regression tolerance")
            old_counter, new_counter = record.get("counterexample"), other.get("counterexample")
            if _skeleton(old_counter) != _skeleton(new_counter) or not _close(
                    _witness(old_counter), _witness(new_counter), tolerance):
                problems.append(f"{task_id}: '{claim}' counterexample differs")
    return {"compared": len(retained), "problems": problems, "passed": not problems}


# Retained operator hardware runs ------------------------------------------------
# A run made on a capture host, in which a hardware probe succeeded or an
# operator capture was retained, is retained under <retained>/hardware/<run-id>/
# with capture.json. It is verified for integrity only.
HARDWARE_DIRECTORY = "hardware"
HARDWARE_RUN_SCHEMA = "ciw.lab-hardware-run.v1"
HARDWARE_ENTRIES = {"reports", "artifacts", "capture.json", "run-log.json"}
HARDWARE_NOTE = ("Retained hardware runs are verified for integrity only: their reports validate and hold no host "
                 "path, their artifacts match the recorded digests, every hardware-measured finding cites raw bytes "
                 "its task retained after a hardware probe succeeded in that task itself (and, for an operator "
                 "capture, a probe of that capture's instrument), and capture.json agrees with the run. A run whose "
                 "physical findings rest on a probe of the capture host's hardware cannot be recomputed in CI or on "
                 "any other host. Operator captures are retained and not authenticated: they are bound to no device "
                 "and never support a hardware-measured finding by themselves.")
ARTIFACT_PATH = re.compile(r"artifacts/(T[0-9]{3})/([^/\\]+)")
# In capture.json and the declared host: an absolute POSIX, home-relative, UNC or drive path at the start of a word
# ("RTX 2080 / Linux" is not one).
HOST_PATH = re.compile(r"(?:^|\s)(?:/[^\s/]|~[\w/]|\\\\|[A-Za-z]:[\\/])")
# In report text, where "~1e-6", "0.2 /mm" and "3/2" are quantities: an absolute POSIX path of at least two
# components, a home-relative path, a UNC path or a drive path, after a space, quote, bracket, '=' or ':'.
REPORT_HOST_PATH = re.compile(r"(?:^|[\s'\"`(\[{=:,])(?:/[^\s/'\"`]+/|~(?:[A-Za-z_][\w.-]*)?/|\\\\[^\s\\]|[A-Za-z]:[\\/])")


def _host_paths(value, where="capture.json", pattern=HOST_PATH) -> list:
    """Strings that look like absolute host paths anywhere inside ``value``."""
    if isinstance(value, dict):
        return [problem for key, item in value.items() for problem in _host_paths(item, f"{where}.{key}", pattern)]
    if isinstance(value, list):
        return [problem for index, item in enumerate(value)
                for problem in _host_paths(item, f"{where}[{index}]", pattern)]
    return [f"{where} holds a host path"] if isinstance(value, str) and pattern.search(value) else []


def _report_host_paths(reports: dict) -> list:
    """Host paths quoted anywhere in a run's reports (an exception message naming an operator's file, say)."""
    return [problem for task_id, report in sorted(reports.items())
            for problem in _host_paths(report, f"reports/{task_id}.json", REPORT_HOST_PATH)]


def _identity(report) -> dict:
    identity = report.get("provider_runtime_identity")
    return identity if isinstance(identity, dict) else {}


def _own_hardware(report) -> set:
    """Hardware whose probe the report records as having succeeded in the task itself (never a memo replay).

    Recorded only in reports with a hardware-measured finding, the ones the physical gate relied on it for.
    """
    names = _identity(report).get("hardware_probed_in_task")
    probes = _probes(report)
    return {name for name in names if isinstance(name, str) and probes.get(f"hardware:{name}") is True} \
        if isinstance(names, list) else set()


def _operator_captures(report) -> dict:
    captures = _identity(report).get("operator_captures")
    return captures if isinstance(captures, dict) else {}


def _hardware_probed(report) -> bool:
    """Whether a hardware probe succeeded while the report's task ran (made there or replayed from a memo)."""
    return any(value is True and name.startswith("hardware:") for name, value in _probes(report).items())


def _measured(report) -> list:
    """Physical-domain findings resting on acquired hardware evidence (hardware_measured or independently checked)."""
    return [f for f in report["findings"] if f["domain"] in PHYSICAL_DOMAINS and f["evidence_status"] != "not_established"]


def _physical_problems(report) -> list:
    """The physical gate, checked on a retained report (:func:`_gate_physical`).

    A hardware-measured finding must cite raw bytes its own task retained,
    after a hardware probe that succeeded in the task itself; bytes of an
    operator capture also need a probe of that capture's instrument.
    """
    problems = []
    digests = {artifact["sha256"] for artifact in report["generated_artifacts"]}
    probed, captures = _own_hardware(report), _operator_captures(report)
    for record in _measured(report):
        digest = record["basis"]["acquisition"]["raw_sha256"]
        where = f"{report['task_id']}: '{record['claim']}'"
        if digest not in digests:
            problems.append(f"{where} raw bytes {digest[:12]} are not a retained artifact of {report['task_id']}")
        if not probed:
            problems.append(f"{where} is hardware evidence but the report records no hardware probe succeeded in the "
                            "task itself (hardware_probed_in_task)")
            continue
        for role in sorted(role for role, captured in captures.items() if captured == digest):
            why = _capture_refusal(str(role), probed)
            if why:
                problems.append(f"{where} cites the operator capture {role}: {why}")
    return problems


def _run_artifact_problems(directory: Path, reports: dict) -> list:
    """Every recorded artifact present with its digest, inside its task's directory; no unrecorded file."""
    problems, recorded = [], set()
    for task_id, report in sorted(reports.items()):
        for artifact in report["generated_artifacts"]:
            match = ARTIFACT_PATH.fullmatch(str(artifact.get("path", "")))
            if not match or match.group(1) != task_id:
                problems.append(f"{task_id}: artifact path {artifact.get('path')!r} is outside artifacts/{task_id}/")
                continue
            recorded.add(artifact["path"])
            path = directory / artifact["path"]
            if path.is_symlink():
                problems.append(f"{task_id}: artifact {artifact['path']} is a link, not retained bytes")
            elif not path.is_file():
                problems.append(f"{task_id}: artifact {artifact['path']} missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                problems.append(f"{task_id}: artifact {artifact['path']} differs from its recorded digest")
    root = directory / "artifacts"
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(directory).as_posix()
            if (path.is_file() or path.is_symlink()) and relative not in recorded:
                problems.append(f"{relative} is not recorded by any report")
    return problems


def _load_run_reports(directory: Path) -> tuple[dict, list]:
    reports, problems = {}, []
    folder = directory / "reports"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            report = validate_report(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError) as exc:  # EvidenceRefusal is a ValueError
            problems.append(f"reports/{path.name} refused: {exc}")
            continue
        if path.name != f"{report['task_id']}.json":
            problems.append(f"reports/{path.name} holds {report['task_id']}")
            continue
        reports[report["task_id"]] = report
    if not reports:
        problems.append("no valid reports")
    return reports, problems


def _run_summary(reports: dict) -> dict:
    return {"tasks": sorted(reports),
            "states": {s: sum(r["state"] == s for r in reports.values()) for s in ("completed", "partial", "deferred", "blocked")},
            "labels": _label_totals(reports.values()),
            "hardware_measured": sum(len(_measured(r)) for r in reports.values())}


CAPTURE_KEYS = {"schema", "run_id", "host", "date", "ciw", "python", "tasks", "reports", "providers", "captures",
                "settings", "hardware_measured", "note"}


def _capture_problems(capture, run_id: str, reports: dict, summary: dict) -> list:
    """capture.json against the run it describes."""
    if not isinstance(capture, dict) or capture.get("schema") != HARDWARE_RUN_SCHEMA:
        return [f"capture.json is not {HARDWARE_RUN_SCHEMA}"]
    problems = []
    if set(capture) != CAPTURE_KEYS:
        problems.append(f"capture.json fields differ from {HARDWARE_RUN_SCHEMA}: {sorted(set(capture) ^ CAPTURE_KEYS)}")
    if capture.get("run_id") != run_id:
        problems.append(f"capture.json names run {capture.get('run_id')!r}, retained as {run_id!r}")
    if not isinstance(capture.get("host"), str) or not capture["host"].strip():
        problems.append("capture.json declares no host")
    if not isinstance(capture.get("date"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", capture["date"]):
        problems.append("capture.json date is not YYYY-MM-DD")
    if capture.get("tasks") != summary["tasks"]:
        problems.append(f"capture.json tasks {capture.get('tasks')} differ from the retained reports {summary['tasks']}")
    if capture.get("reports") != {task_id: report["report_id"] for task_id, report in sorted(reports.items())}:
        problems.append("capture.json report identities differ from the retained reports")
    if capture.get("hardware_measured") != summary["hardware_measured"]:
        problems.append(f"capture.json counts {capture.get('hardware_measured')} hardware-measured findings; the "
                        f"reports hold {summary['hardware_measured']}")
    ciw = capture.get("ciw") if isinstance(capture.get("ciw"), dict) else {}
    if not isinstance(ciw.get("version"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(ciw.get("package_digest"))):
        problems.append("capture.json lacks the CIW version and package digest")
    for task_id, report in sorted(reports.items()):
        identity = report.get("provider_runtime_identity")
        if (isinstance(identity, dict) and identity.get("implementation") == "ciw.lab"
                and identity.get("ciw_version") != ciw.get("version")):
            problems.append(f"{task_id} ran CIW {identity.get('ciw_version')}; capture.json names {ciw.get('version')}")
    artifacts = {a["path"]: a["sha256"] for r in reports.values() for a in r["generated_artifacts"]}
    captures = capture.get("captures")
    for role, entry in sorted(captures.items()) if isinstance(captures, dict) else []:
        if not NAME.fullmatch(str(role)) or not isinstance(entry, dict):
            problems.append(f"capture.json capture {role!r} is malformed")
            continue
        for path in entry.get("artifacts") or []:
            if artifacts.get(path) != entry.get("sha256"):
                problems.append(f"capture.json capture {role} names {path}, which is not retained with its digest")
    if not isinstance(captures, dict) or not isinstance(capture.get("providers"), dict):
        problems.append("capture.json providers and captures must be objects keyed by role")
    return problems + _host_paths(capture)


def _inspect_hardware_run(directory: Path) -> tuple[dict, dict | None]:
    """Integrity summary of one retained hardware run, and the run itself when it has no problem."""
    directory = Path(directory)
    problems = []
    try:
        check_name(directory.name, "hardware run identity")
    except ValueError as exc:
        problems.append(str(exc))
    extra = sorted(entry.name for entry in directory.iterdir() if entry.name not in HARDWARE_ENTRIES)
    if extra:
        problems.append(f"unexpected entries {extra}")
    reports, refused = _load_run_reports(directory)
    problems += refused
    summary = _run_summary(reports)
    problems += _run_artifact_problems(directory, reports)
    for report in reports.values():
        problems += _physical_problems(report)
    problems += _report_host_paths(reports)
    try:
        capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        capture = None
        problems.append(f"capture.json unreadable: {type(exc).__name__}: {exc}")
    if capture is not None:
        problems += _capture_problems(capture, directory.name, reports, summary)
    summary = {"run_id": directory.name, **summary, "problems": problems}
    if isinstance(capture, dict):
        summary.update(date=capture.get("date"), host=capture.get("host"))
    if problems:
        return summary, None
    return summary, {"run_id": directory.name, "date": capture["date"], "host": capture["host"],
                     "directory": directory, "capture": capture, "reports": reports,
                     "hardware_measured": summary["hardware_measured"]}


def _directories(retained) -> list:
    if retained is None:
        return []
    return [Path(retained)] if isinstance(retained, (str, Path)) else [Path(d) for d in retained]


def _run_directories(retained) -> list:
    found = []
    for base in _directories(retained):
        root = base / HARDWARE_DIRECTORY
        if root.is_dir():
            found += [path for path in sorted(root.iterdir()) if path.is_dir()]
    return found


def verify_hardware_runs(retained) -> dict:
    """Integrity of every run under ``<retained>/hardware`` (a directory or a list); nothing is recomputed."""
    runs = [_inspect_hardware_run(directory)[0] for directory in _run_directories(retained)]
    problems = [f"hardware/{run['run_id']}: {problem}" for run in runs for problem in run["problems"]]
    return {"verified": len(runs), "runs": runs, "problems": problems, "passed": not problems, "note": HARDWARE_NOTE}


def hardware_runs(retained, problems: list | None = None) -> list:
    """Retained hardware runs that pass :func:`verify_hardware_runs`, oldest first (by date, then run identity).

    Each run is ``{"run_id", "date", "host", "directory", "capture", "reports",
    "hardware_measured"}``: ``reports`` maps task identities to validated
    reports and ``hardware_measured`` counts the physical-domain findings that
    rest on acquired hardware evidence. A run with any integrity problem is
    left out; its problems are appended to ``problems`` when a list is given.
    """
    runs = []
    for directory in _run_directories(retained):
        summary, run = _inspect_hardware_run(directory)
        if run is None:
            if problems is not None:
                problems += [f"hardware/{summary['run_id']}: {problem}" for problem in summary["problems"]]
            continue
        runs.append(run)
    return sorted(runs, key=lambda run: (run["date"], run["run_id"]))


def latest_hardware(retained, problems: list | None = None) -> dict:
    """Per task, the latest valid retained hardware run holding it: run identity, date, host, state and labels.

    Each value is ``{"run_id", "date", "host", "state", "evidence_status",
    "physical_validation_status", "counts", "hardware_measured", "report"}``;
    labels are the hardware run's own, never merged into the main run's.
    """
    latest = {}
    for run in hardware_runs(retained, problems):
        for task_id, report in run["reports"].items():
            latest[task_id] = {"run_id": run["run_id"], "date": run["date"], "host": run["host"],
                               "state": report["state"], "evidence_status": report["evidence_status"]["primary"],
                               "physical_validation_status": report["physical_validation_status"]["status"],
                               "counts": {k: v for k, v in report["evidence_status"]["counts"].items() if v},
                               "hardware_measured": len(_measured(report)), "report": report}
    return latest


def hardware_note(entry) -> str:
    """One line naming a task's latest hardware run, its state and labels; empty without one."""
    if not entry:
        return ""
    count = entry["hardware_measured"]
    return (f"hardware run {entry['run_id']} ({entry['date']}): {entry['state']}, "
            f"{entry['evidence_status']}, {count} hardware-measured finding{'' if count == 1 else 's'}")


def queue_view(retained, queue=None) -> list:
    """Queue rows with the main run's state and label, plus the latest hardware run of each task that has one.

    Rows are ``{"id", "section", "title", "state", "evidence_status",
    "report_id", "hardware_run"}``; ``hardware_run`` is None or a
    :func:`latest_hardware` entry without its report. The main state and label
    are never replaced by a hardware run's.
    """
    queue = queue or load_queue()
    reports = {r["task_id"]: r for r in load_reports(retained)} if retained and Path(retained, "reports").is_dir() else {}
    hardware = latest_hardware(retained) if retained else {}
    rows = []
    for item in queue["tasks"]:
        report, entry = reports.get(item["id"]), hardware.get(item["id"])
        rows.append({"id": item["id"], "section": item["section_key"], "title": item["title"],
                     "state": report["state"] if report else "not_run",
                     "evidence_status": report["evidence_status"]["primary"] if report else "not_established",
                     "report_id": report["report_id"] if report else None,
                     "hardware_run": {k: v for k, v in entry.items() if k != "report"} if entry else None})
    return rows


UNMEASURED_SCHEMA = "ciw.lab-unmeasured.v1"
UNMEASURED_NOTE = ("The main run is the retained clean-room run; its hardware-measured count and T167's ledger cover "
                   "its reports only. Each valid retained hardware run's counts are its own, listed per run and per "
                   "task, and never added to the main run's count or to T167's. " + HARDWARE_NOTE)


def _open_claims(report) -> list:
    """Physical and authority claims a report records as not established."""
    return [f["claim"] for f in report["findings"]
            if f["domain"] in PHYSICAL_DOMAINS | AUTHORITY_DOMAINS and f["evidence_status"] == "not_established"]


def _ledger_count(directory: Path, report) -> int | None:
    """Hardware-measured findings T167 counted, from its retained ``unmeasured.json`` when its digest matches."""
    for artifact in report["generated_artifacts"]:
        if artifact["path"] != "artifacts/T167/unmeasured.json":
            continue
        try:
            data = (directory / artifact["path"]).read_bytes()
            measured = json.loads(data).get("hardware_measured_findings")
        except (OSError, ValueError, AttributeError):
            return None
        if hashlib.sha256(data).hexdigest() == artifact["sha256"] and isinstance(measured, list):
            return len(measured)
    return None


def unmeasured_view(retained) -> dict:
    """What remains unmeasured, hardware-aware: the main run beside each valid retained hardware run.

    ``retained`` is a directory or a list of directories in precedence order
    (the first holding a task's report wins); hardware runs are read from each
    directory's ``hardware/``. For every task with an open physical or
    authority claim in the main run, or held by a hardware run, a row gives
    the main run's state, hardware-measured count and open claims, and each
    hardware run's own state, labels and measured claims. The main run's count
    and T167's are reported as they are; hardware runs are never merged into
    them. Nothing runs inside the queue, so the clean-room reproduction never
    reads ``hardware/``.
    """
    directories = _directories(retained)
    reports, sources = {}, {}
    for directory in reversed(directories):
        if Path(directory, "reports").is_dir():
            for report in load_reports(directory):
                reports[report["task_id"]], sources[report["task_id"]] = report, Path(directory)
    problems: list = []
    runs = hardware_runs(directories, problems)

    def main(task_id):
        report = reports.get(task_id)
        if report is None:
            return {"state": "not_run", "hardware_measured": 0, "not_established": []}
        return {"state": report["state"], "hardware_measured": len(_measured(report)),
                "not_established": _open_claims(report)}

    rows = {task_id: {"task_id": task_id, "main": main(task_id)} for task_id, report in reports.items()
            if _open_claims(report) or _measured(report)}
    for run in runs:
        for task_id, report in sorted(run["reports"].items()):
            row = rows.setdefault(task_id, {"task_id": task_id, "main": main(task_id)})
            row.setdefault("hardware_runs", []).append({
                "run_id": run["run_id"], "date": run["date"], "host": run["host"], "state": report["state"],
                "evidence_status": report["evidence_status"]["primary"],
                "physical_validation_status": report["physical_validation_status"]["status"],
                "hardware_measured": len(_measured(report)), "measured_claims": [f["claim"] for f in _measured(report)]})
    ledger = reports.get("T167")
    return {"schema": UNMEASURED_SCHEMA, "retained": [str(d) for d in directories],
            "main_run": {"hardware_measured": sum(len(_measured(r)) for r in reports.values()),
                         "not_established_claims": sum(len(_open_claims(r)) for r in reports.values()),
                         "t167": None if ledger is None else {
                             "report_id": ledger["report_id"], "state": ledger["state"],
                             "hardware_measured_findings": _ledger_count(sources["T167"], ledger)}},
            "hardware_runs": [{"run_id": run["run_id"], "date": run["date"], "host": run["host"],
                               "tasks": sorted(run["reports"]), "hardware_measured": run["hardware_measured"]}
                              for run in runs],
            "tasks": [rows[task_id] for task_id in sorted(rows)],
            "hardware_run_problems": problems, "note": UNMEASURED_NOTE}


def _declared_host(host) -> str:
    text = " ".join(str(host or "").split())
    if not text or len(text) > 300:
        raise ValueError("Declare the capture host in one line of at most 300 characters (--host)")
    if HOST_PATH.search(text):
        raise ValueError("The host description must not hold a host path")
    return text


def retain_hardware_run(run_dir, retained, run_id, host) -> dict:
    """Validate a ``ciw lab run`` output made on the capture host and copy it to ``<retained>/hardware/<run_id>``.

    The run must hold the reports of exactly the tasks its ``run-record.json``
    names, all valid and free of host paths, with artifacts matching their
    digests, at least one task in which a hardware probe succeeded or an
    operator capture was retained, and every hardware-measured finding passing
    the physical gate again (:func:`_physical_problems`). ``capture.json`` records the
    declared host, the run date, the CIW version and package digest, the tasks
    and report identities, and the provider and capture bindings as role
    names with digests, never host paths. The copy is verified again before
    this returns; a failed verification removes it.
    """
    run_dir, run_id = Path(run_dir), check_name(run_id, "hardware run identity")
    host = _declared_host(host)
    destination = Path(retained) / HARDWARE_DIRECTORY / run_id
    if destination.exists():
        raise ValueError(f"Hardware run {run_id} is already retained at {destination}")
    try:
        record = json.loads((run_dir / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Not a lab run of this CIW version (no readable {RUN_RECORD}): {run_dir}") from exc
    if not isinstance(record, dict) or record.get("schema") != RUN_RECORD_SCHEMA:
        raise ValueError(f"{run_dir / RUN_RECORD} is not {RUN_RECORD_SCHEMA}")
    reports, problems = _load_run_reports(run_dir)
    if problems:
        raise ValueError("Refusing the run: " + "; ".join(problems[:5]))
    if sorted(reports) != sorted(record.get("tasks") or []):
        raise ValueError(f"The run's reports {sorted(reports)} are not those of its recorded run {record.get('tasks')}; "
                         "retain the output directory of one `ciw lab run`")
    problems = _run_artifact_problems(run_dir, reports)
    for report in reports.values():
        problems += _physical_problems(report)
    problems += _report_host_paths(reports)
    if problems:
        raise ValueError("Refusing the run: " + "; ".join(problems[:5]))
    if not any(_hardware_probed(report) or _operator_captures(report) for report in reports.values()):
        raise ValueError("No hardware probe succeeded and no operator capture was retained in any task of this run; "
                         "it is not a hardware run")
    summary = _run_summary(reports)
    artifacts = {}
    for report in reports.values():
        for artifact in report["generated_artifacts"]:
            artifacts.setdefault(artifact["sha256"], []).append(artifact["path"])
    captures = {role: dict(entry, artifacts=sorted(artifacts.get(entry.get("sha256"), [])))
                for role, entry in sorted((record.get("captures") or {}).items())}
    capture = {"schema": HARDWARE_RUN_SCHEMA, "run_id": run_id, "host": host,
               "date": str(record.get("started_utc", ""))[:10], "ciw": record.get("ciw"),
               "python": record.get("python"), "tasks": summary["tasks"],
               "reports": {task_id: report["report_id"] for task_id, report in sorted(reports.items())},
               "providers": record.get("providers") or {}, "captures": captures,
               "settings": record.get("settings") or {}, "hardware_measured": summary["hardware_measured"],
               "note": HARDWARE_NOTE}
    paths = _host_paths(capture)
    if paths:
        raise ValueError("Refusing the run: " + "; ".join(paths[:5]))
    destination.mkdir(parents=True)
    try:
        (destination / "reports").mkdir()
        for task_id in summary["tasks"]:
            shutil.copy2(run_dir / "reports" / f"{task_id}.json", destination / "reports" / f"{task_id}.json")
            if reports[task_id]["generated_artifacts"]:
                shutil.copytree(run_dir / "artifacts" / task_id, destination / "artifacts" / task_id)
        if (run_dir / "run-log.json").is_file():
            shutil.copy2(run_dir / "run-log.json", destination / "run-log.json")
        (destination / "capture.json").write_text(dumps(capture), encoding="utf-8")
        verified, _ = _inspect_hardware_run(destination)
        if verified["problems"]:
            raise ValueError("The retained copy fails verification: " + "; ".join(verified["problems"][:5]))
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {"run_id": run_id, "retained": str(destination), "date": capture["date"], "host": host,
            "tasks": summary["tasks"], "states": summary["states"], "hardware_measured": summary["hardware_measured"],
            "note": HARDWARE_NOTE}


def verify_retained(retained, fresh, tasks=None) -> dict:
    """:func:`compare` plus the integrity of every retained hardware run (``hardware_runs`` in the result)."""
    result = compare(retained, fresh, tasks)
    hardware = verify_hardware_runs(retained)
    result["hardware_runs"] = {key: hardware[key] for key in ("verified", "runs", "note")}
    result["problems"] = result["problems"] + hardware["problems"]
    result["passed"] = not result["problems"]
    return result


REPORT_QUESTIONS = [label for _, label in FIELDS]


def schema_errors(report) -> list:
    """Structural errors against task-report.schema.json (requires jsonschema)."""
    from importlib import resources
    import jsonschema

    schema = json.loads(resources.files("ciw.lab").joinpath("task-report.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return sorted(f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                  for error in validator.iter_errors(report))
