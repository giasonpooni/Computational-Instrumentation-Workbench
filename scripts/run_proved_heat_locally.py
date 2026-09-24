"""Run the SP1 proved-heat gate on this host by replaying ``.github/workflows/proved-heat.yml``'s own steps.

The workflow provisions a GitHub runner and then runs the gate. This script
reads the workflow file and runs its gate steps (``run`` scripts), unchanged and
in order, from the repository root, against checkouts and toolchains the
operator provisioned. It writes ``results/proved-heat/`` exactly as the
workflow does, plus ``local-run.json`` (``ciw.proved-heat-local-run.v1``: which
steps ran and for how long, the host, toolchain versions and source identities,
as role names and versions, never host paths) and one log per step under
``local-logs/``.

The provisioning steps (checkout, setup-python, the apt and rustup installs,
the SCR and SP1 clones, the Succinct compiler archive, the build cache and the
artifact upload) are not run. Their results are checked first, and the script
refuses when any is missing: Linux; Python 3.12+ (``--python``) with
setuptools 77+ and wheel; git, clang, protoc, pkg-config with OpenSSL, cargo
and rustup; the rustup toolchain the workflow installs and the linked
``succinct`` toolchain; the compiler archive (``--compiler-archive``) with the
SHA-256 the workflow pins; clean SCR and SP1 checkouts at the revisions the
workflow clones and the trees ``scripts/check_proved_heat.py`` pins (use a
fresh SP1 clone for every run: the SP1 build writes generated files into it);
PyYAML to read the workflow; and no earlier ``results/proved-heat``. The
workflow's own resource step then checks memory and disk. A workflow step this
script does not know, or a condition it does not understand, is refused rather
than guessed; the build cache is always a miss, so the build step always runs.
The first failing step ends the run, and nothing is retried or repaired.

Retain a passing run with
``ciw lab proved-heat retain results/proved-heat --retained lab --run-id <id> --host "<host>"``
(docs/LAB.md, Proved-heat gate records).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/proved-heat.yml"
DRIVER = "scripts/run_proved_heat_locally.py"
OUTPUT = Path("results") / "proved-heat"
LOCAL_RUN_SCHEMA = "ciw.proved-heat-local-run.v1"
CACHE_MISS = "steps.native-build-cache.outputs.cache-hit != 'true'"
# Workflow steps that provision the runner: not run here, their result is checked instead.
PROVISIONING = {
    "actions/checkout@v4": "the gate runs from this CIW checkout",
    "actions/setup-python@v5": "the interpreter given with --python is `python` for every step",
    "Install native build dependencies": "system packages, setuptools, wheel and the rustup toolchain are the "
                                         "operator's; the tools and toolchains the gate needs were checked",
    "Check out exact SCR and upstream SP1 sources": "the operator's clean SCR and SP1 checkouts were checked at the "
                                                    "pinned revisions and trees and linked into the stack",
    "Restore exact pinned host builds and Cargo downloads": "no build cache: a miss, so the build step ran",
    "Install the recipe's exact Succinct compiler archive": "the operator linked the archive as the succinct "
                                                            "toolchain; its SHA-256 was checked against the pin",
    "Save completed pinned hosts before running proof tests": "no build cache",
    "actions/upload-artifact@v4": "the outputs stay in results/proved-heat",
}
# The workflow's gate steps, run unchanged.
GATE_STEPS = ("Check Linux resource budget", "Rebuild and verify the committed heat ELF recipe",
              "Build native execution and CPU proving hosts",
              "Check build source bytes and provision clean SP1 reference",
              "Exercise actual proofs through an isolated installed CIW wheel")


def _name(step: dict) -> str:
    return str(step.get("name") or step.get("uses") or "")


def plan(workflow: dict) -> tuple[list, list]:
    """(gate steps to run, provisioning steps not run with the reason); refuses a step or condition it does not know."""
    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict) or len(jobs) != 1:
        raise SystemExit(f"{WORKFLOW} no longer has exactly one job; update {DRIVER}")
    job_name, job = next(iter(jobs.items()))
    run, skipped = [], []
    for step in job.get("steps") or []:
        name = _name(step)
        condition = step.get("if")
        if name in PROVISIONING:
            skipped.append({"name": name, "status": "not_run", "reason": PROVISIONING[name]})
        elif name in GATE_STEPS and "run" in step:
            if condition not in (None, CACHE_MISS):
                raise SystemExit(f"{WORKFLOW} step {name!r} has a condition {DRIVER} does not understand: {condition}")
            run.append(step)
        else:
            raise SystemExit(f"{WORKFLOW} has a step {DRIVER} does not know: {name!r}; update {DRIVER}")
    missing = set(GATE_STEPS) - {_name(step) for step in run}
    if missing:
        raise SystemExit(f"{WORKFLOW} no longer runs {sorted(missing)}; update {DRIVER}")
    return run, skipped


def _step_text(workflow: dict, name: str) -> str:
    job = next(iter(workflow["jobs"].values()))
    return next(str(step.get("run", "")) for step in job["steps"] if _name(step) == name)


def workflow_pins(workflow: dict) -> dict:
    """The toolchain, compiler archive digest and SCR/SP1 revisions the workflow provisions, read from its steps."""
    toolchain = re.search(r"rustup toolchain install (\S+) --profile minimal",
                          _step_text(workflow, "Install native build dependencies"))
    archive = re.search(r'echo "([0-9a-f]{64})  \$archive"',
                        _step_text(workflow, "Install the recipe's exact Succinct compiler archive"))
    revisions = re.findall(r"checkout --detach ([0-9a-f]{40})",
                           _step_text(workflow, "Check out exact SCR and upstream SP1 sources"))
    if not toolchain or not archive or len(revisions) != 2:
        raise SystemExit(f"Cannot read the toolchain, archive digest and source revisions from {WORKFLOW}; "
                         f"update {DRIVER}")
    return {"toolchain": toolchain.group(1), "archive_sha256": archive.group(1),
            "scr_revision": revisions[0], "sp1_revision": revisions[1]}


def _output(command, environment=None) -> str | None:
    try:
        completed = subprocess.run([str(part) for part in command], capture_output=True, text=True, timeout=120,
                                   env=environment)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prerequisite_problems(args, pins: dict) -> list:
    """Every missing prerequisite of the gate steps, as sentences; empty when the host is provisioned."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_proved_heat import SCR_REVISION, SCR_TREE, SP1_REVISION, SP1_TREE, source_identity
    problems = []
    if sys.platform != "linux":
        problems.append("the proved-heat gate runs on Linux only")
    version = _output([args.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
    if version is None or tuple(map(int, version.split("."))) < (3, 12):
        problems.append(f"--python must be Python 3.12 or newer (it is {version or 'not runnable'})")
    else:
        setuptools = _output([args.python, "-c", "import setuptools, wheel; print(setuptools.__version__)"])
        if setuptools is None or int(setuptools.split(".")[0]) < 77:
            problems.append("--python needs setuptools 77 or newer and wheel (the workflow installs them)")
    for tool in ("bash", "git", "clang", "protoc", "pkg-config", "cargo", "rustup"):
        if shutil.which(tool) is None:
            problems.append(f"{tool} is not on PATH")
    if shutil.which("pkg-config") and _output(["pkg-config", "--exists", "openssl"]) is None:
        problems.append("pkg-config finds no OpenSSL (libssl-dev)")
    no_install = {**os.environ, "RUSTUP_AUTO_INSTALL": "0"}
    if shutil.which("cargo") and _output(["cargo", f"+{pins['toolchain']}", "--version"], no_install) is None:
        problems.append(f"the rustup toolchain {pins['toolchain']} is not installed "
                        f"(rustup toolchain install {pins['toolchain']} --profile minimal)")
    if shutil.which("rustc") and _output(["rustc", "+succinct", "-vV"], no_install) is None:
        problems.append("no linked succinct toolchain answers rustc +succinct -vV")
    archive = Path(args.compiler_archive)
    if not archive.is_file():
        problems.append("--compiler-archive is not a file")
    elif _file_sha256(archive) != pins["archive_sha256"]:
        problems.append("--compiler-archive does not have the SHA-256 the workflow pins")
    if (pins["scr_revision"], pins["sp1_revision"]) != (SCR_REVISION, SP1_REVISION):
        problems.append(f"{WORKFLOW} and scripts/check_proved_heat.py pin different SCR or SP1 revisions")
    for role, path, revision, tree in (("SCR", args.scr, SCR_REVISION, SCR_TREE), ("SP1", args.sp1, SP1_REVISION, SP1_TREE)):
        try:
            source_identity(path, revision, tree)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            problems.append(f"the {role} checkout is not clean at {revision} with tree {tree}: {exc}")
    if (ROOT / OUTPUT).exists():
        problems.append(f"{OUTPUT} already exists; move it away so that stale outputs cannot satisfy the gate")
    return problems


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def host_facts() -> dict:
    facts = {"system": platform.system(), "kernel": platform.release(), "machine": platform.machine(),
             "cpus": os.cpu_count()}
    try:
        release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        facts["os"] = release.get("PRETTY_NAME", "").strip('"') or None
    except OSError:
        facts["os"] = None
    try:
        memory = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
        facts["memory_total_bytes"] = int(memory.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        facts["memory_total_bytes"] = None
    return facts


def toolchains(args, pins: dict) -> dict:
    no_install = {**os.environ, "RUSTUP_AUTO_INSTALL": "0"}
    succinct = _output(["rustc", "+succinct", "-vV"], no_install) or ""
    release = re.search(r"^release: (.+)$", succinct, re.MULTILINE)
    commit = re.search(r"^commit-hash: (.+)$", succinct, re.MULTILINE)
    return {"python": _output([args.python, "-c", "import platform; print(platform.python_version())"]),
            "rustc": _output(["rustc", f"+{pins['toolchain']}", "--version"], no_install),
            "cargo": _output(["cargo", f"+{pins['toolchain']}", "--version"], no_install),
            "rustc_succinct": (f"rustc {release.group(1)} (commit-hash {commit.group(1) if commit else 'unknown'})"
                               if release else None),
            "default_rustc": _output(["rustc", "--version"]), "rustup": _first_line(_output(["rustup", "--version"])),
            "protoc": _output(["protoc", "--version"]), "clang": _first_line(_output(["clang", "--version"])),
            "git": _output(["git", "--version"])}


def sources(args) -> dict:
    def identity(path):
        return {"revision": _output(["git", "-C", path, "rev-parse", "HEAD"]),
                "tree": _output(["git", "-C", path, "rev-parse", "HEAD^{tree}"])}
    changes = _output(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"])
    return {"scr": {**identity(args.scr), "provisioning": "operator checkout linked into the stack"},
            "sp1": {**identity(args.sp1), "provisioning": "operator checkout linked into the stack"},
            "ciw": {"commit": _output(["git", "-C", ROOT, "rev-parse", "HEAD"]),
                    "uncommitted_changes": None if changes is None else bool(changes)}}


def stack(args, work: Path) -> None:
    """RUNNER_TEMP as the workflow lays it out, with the operator's checkouts and target directories linked in."""
    root = work / "proved-heat-stack"
    (root / "notationsystems").mkdir(parents=True)
    (root / "scr").symlink_to(Path(args.scr).resolve(), target_is_directory=True)
    (root / "notationsystems" / "SP1-zero-knowledge-virtual-machine").symlink_to(Path(args.sp1).resolve(),
                                                                               target_is_directory=True)
    if args.target_root:
        for name in ("native-target", "proof-target"):
            target = Path(args.target_root).resolve() / name
            target.mkdir(parents=True, exist_ok=True)
            (root / name).symlink_to(target, target_is_directory=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scr", required=True, type=Path, help="Clean SCR checkout at the workflow's revision")
    parser.add_argument("--sp1", required=True, type=Path, help="Fresh, clean SP1 clone at the workflow's revision")
    parser.add_argument("--compiler-archive", required=True, type=Path,
                        help="The Succinct compiler archive the workflow downloads (checked against its SHA-256)")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python 3.12+ with setuptools 77+ and wheel, run as `python` by every step")
    parser.add_argument("--target-root", type=Path,
                        help="Directory holding native-target/ and proof-target/ to reuse across runs (the "
                             "workflow's cache); fresh ones in the work directory otherwise")
    parser.add_argument("--work-root", type=Path, help="New or empty directory used as RUNNER_TEMP")
    args = parser.parse_args(argv)
    try:
        import yaml
    except ImportError:
        raise SystemExit(f"Reading {WORKFLOW} needs PyYAML (python -m pip install pyyaml)") from None
    text = (ROOT / WORKFLOW).read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    steps, skipped = plan(workflow)
    pins = workflow_pins(workflow)
    problems = prerequisite_problems(args, pins)
    if problems:
        raise SystemExit("Refusing to run the proved-heat gate:\n- " + "\n- ".join(problems))
    if args.work_root:
        work = args.work_root.resolve()
        if work.exists() and any(work.iterdir()):
            raise SystemExit(f"--work-root must be new or empty: {work}")
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="ciw-proved-heat-runner-"))
    stack(args, work)
    shim = work / "bin"
    shim.mkdir()
    # A wrapper rather than a link keeps a virtual environment's interpreter inside its environment.
    (shim / "python").write_text(f'#!/bin/sh\nexec "{Path(args.python).absolute()}" "$@"\n', encoding="utf-8")
    (shim / "python").chmod(0o755)
    job = next(iter(workflow["jobs"].values()))
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTEST_ADDOPTS")}
    environment.update({key: str(value) for key, value in (job.get("env") or {}).items()})
    if os.environ.get("CARGO_BUILD_JOBS"):
        environment["CARGO_BUILD_JOBS"] = os.environ["CARGO_BUILD_JOBS"]
    environment.update(RUNNER_TEMP=str(work), PATH=f"{shim}{os.pathsep}{environment.get('PATH', '')}")
    record = {"schema": LOCAL_RUN_SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "record_origin": "written_by_driver",
              "procedure": {"kind": "local_workflow_replay", "workflow": WORKFLOW,
                            "workflow_sha256": sha256(text.encode("utf-8")).hexdigest(), "job": next(iter(workflow["jobs"])),
                            "driver": DRIVER, "driver_sha256": _file_sha256(Path(__file__)),
                            "cargo_build_jobs": environment.get("CARGO_BUILD_JOBS"), "outcome": "failed"},
              "steps": [], "host_facts": host_facts(), "toolchains": toolchains(args, pins), "sources": sources(args),
              "compiler_archive": {"sha256": pins["archive_sha256"],
                                   "checked": "SHA-256 of --compiler-archive equals the workflow's pin"}}
    logs = work / "local-logs"
    logs.mkdir()
    deadline = time.monotonic() + 60 * int(job.get("timeout-minutes") or 180)
    ran = {_name(step): step for step in steps}
    order = [_name(step) for step in job["steps"]]
    failed = None
    for number, name in enumerate(order, 1):
        if name not in ran:
            record["steps"].append(next(row for row in skipped if row["name"] == name))
            continue
        if failed is not None:
            record["steps"].append({"name": name, "status": "not_run", "reason": "an earlier step failed"})
            continue
        log = logs / f"{number:02d}-{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')[:60]}.log"
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            try:
                code = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", ran[name]["run"]],
                                      cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                      timeout=max(1.0, deadline - time.monotonic())).returncode
            except subprocess.TimeoutExpired:
                code = None
        seconds = round(time.monotonic() - started, 1)
        record["steps"].append({"name": name, "status": "ran", "seconds": seconds, "exit_code": code})
        print(f"    exit {code} after {seconds:.0f} s (log {log.name})", flush=True)
        if code != 0:
            failed = name
    if failed is None:
        record["procedure"]["outcome"] = "passed"
    output = ROOT / OUTPUT
    if output.is_dir():
        shutil.copytree(logs, output / "local-logs")
        (output / "local-run.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if failed is not None:
        print(f"FAIL: {failed}; logs in {output / 'local-logs' if output.is_dir() else logs}", file=sys.stderr)
        return 1
    print(f"PASS: every gate step of {WORKFLOW} passed; outputs in {OUTPUT}. Retain them with\n"
          f"  ciw lab proved-heat retain {OUTPUT} --retained lab --run-id <id> --host \"<host>\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
