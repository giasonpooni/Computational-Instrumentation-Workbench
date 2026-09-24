"""Provider checkout identities, locked builds and pinned integrations for T097-T099.

Scope: read-only identity of operator-bound provider checkouts (HEAD, HEAD
tree, a digest of every tracked file's working bytes, the digest of every
recognised lockfile), a second reading of the same digests from Git's HEAD
objects, an independent recomputation of the Git tree id from the working
bytes, comparison with every provider pin CIW declares, the identity of bound
provider interpreters (version, executable digest and, for PLSR, the installed
runtime's version and source digests), the SCR ``execution-cli`` locked offline
build (with the default or a named rustup toolchain), execution of the SCR heat kernel through its own Python API and through
CIW's numerical-heat workflow, and the SET/PPDA exchange integrations when their
exact checkouts are bound.

Non-claims: a matching HEAD and tree show that the bound bytes are the pinned
source; they do not authenticate the upstream repository, the Rust toolchain or
a built binary beyond the inputs actually executed. Engine and interpreter
digests depend on the toolchain and host build and are retained as provenance,
not as regression values. Nothing here clones, checks out, repairs or writes
into a provider checkout.
"""
from __future__ import annotations

from hashlib import new as new_hash, sha256
from importlib import resources
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORIES = {
    "csg": "giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime",
    "ftr": "giasonpooni/Flat-Torus-Geodesic-Reference",
    "scr": "giasonpooni/Scientific-Computation-Runtime",
    "scr-exchange": "giasonpooni/Scientific-Computation-Runtime",
    "sra": "giasonpooni/Schematics-Retrieval-Agent",
    "plsr": "giasonpooni/Parameterized-Lyapunov-Stability-Runtime",
    "set": "giasonpooni/State-Estimation-Evaluation-Testbed",
    "ppda": "giasonpooni/Provenance-Preserving-Data-Acquisition",
}
# Pins that live only in .github/workflows/exchange.yml (not in the installed
# package). tests/test_lab_exchange_provenance_bundles.py keeps them in step.
EXCHANGE_WORKFLOW_PINS = {
    "set": "542e672be512bf43b61253f2b2a43cd967cb3062",
    "ppda": "a29845e13e55de30b24ae752b896058041d653e6",
    "scr-exchange": "5f0409743e0098a0691a88302a9b3dcdcbcf25fd",
}
# Everything .github/workflows/proved-heat.yml provisions before its SP1 build;
# T099 records these and never attempts the build.
SP1_REQUIREMENTS = {
    "source": ".github/workflows/proved-heat.yml",
    "command": ("cargo +1.94.0 build --release --locked --manifest-path <scr>/zk/Cargo.toml "
                "--target-dir <proof-target> -p sp1-adapter --bin sp1-host"),
    "sp1_checkout": {"repository": "https://github.com/succinctlabs/sp1.git",
                     "revision": "b38b61209e45e969289e70d5cf79dc763460bc41",
                     "tree": "7deca3aced8d8eb84dfcede98285a192c862ea4c",
                     "path": "<scr>/../notationsystems/SP1-zero-knowledge-virtual-machine"},
    "succinct_compiler_archive": {
        "url": ("https://github.com/succinctlabs/rust/releases/download/succinct-1.94.0-64bit/"
                "rust-toolchain-x86_64-unknown-linux-gnu.tar.gz"),
        "sha256": "12c94435d41bfe4e20131bbcce40b35abd32270ad792befc653af4e3fabc192f",
        "install": "tar -xzf <archive> -C <dir>; rustup toolchain link succinct <dir>; rustc +succinct -vV"},
    "guest_recipe": {"path": "zk/recipes/sp1-heat.recipe",
                     "identity": "6e5d1687bcc55243d712553a2b7768b6c587a76418bb48a7a2c44224470d423d",
                     "verify": "execution.build.verify_build(recipe, <guest sha256>, repo_root=<scr>)"},
    "guest_sha256": "a14e3750da7e221d31842bd6cf983fcc8c0f530b2811537e2a9a9fe803dacf82",
    "network": ("crates.io and git dependencies (the zk workspace is not --offline), github.com clones of SCR and "
                "SP1, and the github.com release download of the Succinct compiler archive"),
    "system_packages": ["build-essential", "clang", "libclang-dev", "libssl-dev", "pkg-config", "protobuf-compiler"],
    "toolchain": "rustup toolchain 1.94.0 (minimal profile) plus the linked succinct toolchain",
    "resources": {"available_memory_bytes_at_least": 7 * 1024 ** 3, "free_disk_bytes_at_least": 20 * 1024 ** 3},
    "gate": "scripts/check_proved_heat.py via .github/workflows/proved-heat.yml",
}
# The rustup toolchain .github/workflows/proved-heat.yml builds execution-cli and sp1-host with (and lab.yml
# installs, so that T099 rebuilds execution-cli with it); tests keep both workflows in step.
NATIVE_TOOLCHAIN = "1.94.0"
_CACHE_DIRS = (".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
# Dependency lockfiles recognised by name; requirements*.txt counts only when
# every requirement in it is pinned (see is_lockfile).
LOCKFILE_NAMES = frozenset({"Cargo.lock", "uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock", "package-lock.json",
                            "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "Gemfile.lock", "go.sum",
                            "composer.lock", "flake.lock"})
_REQUIREMENTS = re.compile(r"requirements[\w.-]*\.txt")
PLSR_DISTRIBUTION = "parameterized-lyapunov-stability-runtime"


def is_lockfile(name: str, data: bytes) -> bool:
    """A recognised lockfile, or a requirements*.txt whose every requirement is pinned (``==``/``===``/``@``)."""
    leaf = name.rsplit("/", 1)[-1]
    if leaf in LOCKFILE_NAMES:
        return True
    if not _REQUIREMENTS.fullmatch(leaf):
        return False
    requirements = []
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.split(" #", 1)[0].strip()
        if line and not line.startswith(("#", "-")):
            requirements.append(line.rstrip("\\").strip())
    return bool(requirements) and all("==" in line or " @ " in line for line in requirements)


def ciw_pins() -> dict:
    """Every provider revision CIW declares, by role, with where it is declared."""
    from .. import declared_workload, geodesic_reference, proved_heat
    pins: dict = {}

    def add(role, revision, declared_in, **extra):
        pins.setdefault(role, []).append({"revision": revision, "declared_in": declared_in, **extra})

    for kind, pin in sorted(geodesic_reference.PINS.items()):
        add(pin["role"], pin["revision"], f"ciw.geodesic_reference.PINS[{kind}]", source_tree=pin["source_tree"],
            module=pin["module"], source_root=pin["source_root"])
    for kind, pin in sorted(declared_workload.PINS.items()):
        add(pin["role"], pin["revision"], f"ciw.declared_workload.PINS[{kind}]", module=pin["module"],
            source_root=pin["source_root"])
    add("scr", proved_heat.PIN["revision"], "ciw.proved_heat.PIN", source_tree=proved_heat.PIN["source_tree"],
        module=proved_heat.PIN["module"], source_root=proved_heat.PIN["source_root"])
    package = resources.files("ciw")
    plsr = json.loads(package.joinpath("plsr-runtime.json").read_text(encoding="utf-8"))
    add("plsr", plsr["commit"], "ciw/plsr-runtime.json")
    exchange = json.loads(package.joinpath("exchange-runtime.json").read_text(encoding="utf-8"))
    add("set", exchange["revision"], "ciw/exchange-runtime.json", validator_path=exchange["path"],
        validator_sha256=exchange["sha256"])
    for manifest in ("telemetry-runtimes.json", "calibrated-observable-runtimes.json", "calibrated-window-runtimes.json"):
        for role, entry in sorted(json.loads(package.joinpath(manifest).read_text(encoding="utf-8")).items()):
            add(role, entry["revision"], f"ciw/{manifest}[{role}]", module=entry["module"],
                source_root=entry["source_root"])
    for role, revision in sorted(EXCHANGE_WORKFLOW_PINS.items()):
        add(role, revision, ".github/workflows/exchange.yml")
    return pins


def _git(path: Path, *arguments: str, stdin: bytes | None = None) -> bytes:
    # No optional index refresh or fsmonitor: inspection never writes to an
    # operator-owned checkout.
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "core.autocrlf=false",
                           "-C", str(path), *arguments], check=True, capture_output=True, timeout=120,
                          env=environment, input=stdin).stdout


def _tree_id(entries: dict, algorithm: str) -> bytes:
    """Git tree object id from {name: (mode, raw id | subtree dict)}; trees sort as name + '/'."""
    body = b""
    for name in sorted(entries, key=lambda item: item + ("/" if isinstance(entries[item][1], dict) else "")):
        mode, value = entries[name]
        raw = _tree_id(value, algorithm) if isinstance(value, dict) else value
        body += mode.encode("ascii") + b" " + name.encode("utf-8") + b"\0" + raw
    return new_hash(algorithm, b"tree " + str(len(body)).encode("ascii") + b"\0" + body).digest()


def checkout_identity(path) -> dict:
    """Read-only identity of a provider checkout from its working bytes.

    The recomputed tree uses the working bytes and working executable bits (on
    POSIX), so it equals ``git rev-parse HEAD^{tree}`` only when every tracked
    file is present, unmodified and of the tracked type.
    """
    path = Path(path).resolve()
    top = Path(os.fsdecode(_git(path, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top != path:
        raise ValueError(f"Provider path must be a repository root: {path}")
    head = _git(path, "rev-parse", "HEAD").decode().strip()
    tree = _git(path, "rev-parse", "HEAD^{tree}").decode().strip()
    algorithm = _git(path, "rev-parse", "--show-object-format").decode().strip()
    if algorithm not in ("sha1", "sha256"):
        raise ValueError(f"Unsupported Git object format: {algorithm}")
    root: dict = {}
    manifest, mismatched, lockfiles, gitlinks = [], [], {}, 0
    for entry in _git(path, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.decode("ascii").split()
        name = raw_name.decode("utf-8")
        node = root
        *folders, leaf = name.split("/")
        for folder in folders:
            node = node.setdefault(folder, ("40000", {}))[1]
        if kind == "commit":
            gitlinks += 1
            node[leaf] = (mode, bytes.fromhex(expected))
            manifest.append(f"{mode} commit {expected} {name}")
            continue
        item = path / name
        try:
            if item.is_symlink():
                data, working_mode = os.fsencode(os.readlink(item)), "120000"
            else:
                data = item.read_bytes()
                working_mode = (("100755" if item.stat().st_mode & 0o111 else "100644")
                                if os.name == "posix" else mode)
        except OSError:
            data, working_mode = b"", "missing"
        blob = new_hash(algorithm, b"blob " + str(len(data)).encode("ascii") + b"\0" + data)
        if blob.hexdigest() != expected or working_mode != mode:
            mismatched.append(name)
        node[leaf] = (working_mode if working_mode != "missing" else mode, blob.digest())
        digest = sha256(data).hexdigest()
        manifest.append(f"{mode} blob {digest} {name}")
        if is_lockfile(name, data):
            lockfiles[name] = digest
    exclusions = [f":(exclude,glob)**/{name}/**" for name in _CACHE_DIRS]
    untracked = [item for item in _git(path, "ls-files", "--others", "-z", "--", ".", *exclusions).split(b"\0") if item]
    dirty = bool(_git(path, "status", "--porcelain=v1", "--untracked-files=no"))
    return {"path": str(path), "head": head, "tree": tree, "object_format": algorithm,
            "recomputed_tree": _tree_id(root, algorithm).hex(), "tracked_files": len(manifest),
            "gitlinks": gitlinks, "tracked_sha256": sha256("\n".join(manifest).encode("utf-8")).hexdigest(),
            "mismatched_files": sorted(mismatched), "untracked_outside_caches": len(untracked), "dirty": dirty,
            "lockfiles": dict(sorted(lockfiles.items()))}


def head_object_digests(path) -> dict:
    """The tracked-source and lockfile digests of :func:`checkout_identity`, read from Git's HEAD objects.

    A second reader of the same quantities: blob bytes come from ``git cat-file
    --batch`` (the object database), not from the working tree, so for a clean
    checkout both readings agree and a working-tree reading error does not.
    """
    path = Path(path).resolve()
    listing, blobs = [], []
    for entry in _git(path, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.decode("ascii").split()
        listing.append((mode, kind, expected, raw_name.decode("utf-8")))
        if kind == "blob":
            blobs.append(expected)
    output = _git(path, "cat-file", "--batch", stdin=b"".join(oid.encode("ascii") + b"\n" for oid in blobs))
    contents, offset = {}, 0
    for oid in blobs:
        end = output.index(b"\n", offset)
        header = output[offset:end].decode("ascii").split()
        if header[0] != oid or header[1] != "blob":
            raise ValueError(f"Unexpected git cat-file record for {oid}")
        size = int(header[2])
        contents[oid] = output[end + 1:end + 1 + size]
        offset = end + 1 + size + 1
    manifest, lockfiles = [], {}
    for mode, kind, expected, name in listing:
        if kind == "commit":
            manifest.append(f"{mode} commit {expected} {name}")
            continue
        data = contents[expected]
        digest = sha256(data).hexdigest()
        manifest.append(f"{mode} blob {digest} {name}")
        if is_lockfile(name, data):
            lockfiles[name] = digest
    return {"tracked_files": len(manifest), "tracked_sha256": sha256("\n".join(manifest).encode("utf-8")).hexdigest(),
            "lockfiles": dict(sorted(lockfiles.items()))}


_INTERPRETER_PROBE = r'''
import hashlib, json, platform, sys
from importlib import metadata, resources
out = {"python_version": platform.python_version(), "implementation": platform.python_implementation()}
try:
    distribution = metadata.distribution(sys.argv[1])
except metadata.PackageNotFoundError:
    distribution = None
if distribution is not None:
    record = {"version": distribution.version}
    try:
        direct = json.loads(distribution.read_text("direct_url.json") or "null") or {}
    except ValueError:
        direct = {}
    record["install_commit"] = (direct.get("vcs_info") or {}).get("commit_id")
    record["install_kind"] = next((kind for kind in ("vcs_info", "dir_info", "archive_info") if kind in direct), None)
    files = {}

    def walk(node, prefix=""):
        for child in node.iterdir():
            if child.name == "__pycache__":
                continue
            name = prefix + child.name
            if child.is_dir():
                walk(child, name + "/")
            elif name.endswith((".py", ".json")) or child.name == "py.typed":
                # CRLF-normalised, as ciw.plsr_engine and ciw.lab.lyapunov_provider hash them.
                files[name] = hashlib.sha256(child.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    try:
        walk(resources.files("lyapunov"))
        record["files"] = files
    except Exception as exc:
        record["error"] = type(exc).__name__
    out["plsr"] = record
print(json.dumps(out, sort_keys=True))
'''


def interpreter_identity(executable) -> dict:
    """Version and executable SHA-256 of a bound interpreter, plus the installed PLSR runtime if present.

    The executable digest follows symlinks (a virtual environment's entry point
    hashes as its base interpreter). Nothing is installed or imported into this
    process; the probe runs in isolated mode.
    """
    executable = Path(executable)
    data = executable.read_bytes()
    with tempfile.TemporaryDirectory(prefix="ciw-lab-interpreter-") as directory:
        completed = subprocess.run([str(executable), "-I", "-B", "-c", _INTERPRETER_PROBE, PLSR_DISTRIBUTION],
                                   capture_output=True, text=True, timeout=120, cwd=directory)
    if completed.returncode:
        raise ValueError("Interpreter probe failed: " + (completed.stderr.strip().splitlines() or ["no output"])[-1])
    return {"sha256": sha256(data).hexdigest(), "byte_count": len(data), "entry_is_symlink": executable.is_symlink(),
            **json.loads(completed.stdout)}


def plsr_manifest() -> dict:
    return json.loads(resources.files("ciw").joinpath("plsr-runtime.json").read_text(encoding="utf-8"))


def status_entries(path) -> int:
    """Entries of ``git status`` including ignored and untracked files outside runtime caches.

    A second reader of cleanliness, independent of :func:`checkout_identity`'s
    byte recomputation (used to corroborate a refusal).
    """
    exclusions = [f":(exclude,glob)**/{name}/**" for name in _CACHE_DIRS]
    output = _git(Path(path).resolve(), "status", "--porcelain=v1", "-z", "--ignored", "--untracked-files=all",
                  "--", ".", *exclusions)
    return sum(1 for entry in output.split(b"\0") if entry)


def compare_with_pins(role: str, identity: dict, pins: dict) -> dict:
    """Which CIW pins the checkout satisfies; a revision match with a tree mismatch is refused."""
    declared = pins.get(role, [])
    matched = [pin["declared_in"] for pin in declared if pin["revision"] == identity["head"]]
    tree_refusals = [pin["declared_in"] for pin in declared if pin["revision"] == identity["head"]
                     and pin.get("source_tree") and pin["source_tree"] != identity["tree"]]
    unmatched = sorted({pin["declared_in"] for pin in declared} - set(matched))
    clean = not identity["dirty"] and not identity["mismatched_files"] and not identity["untracked_outside_caches"]
    return {"role": role, "matched": matched, "unmatched": unmatched, "tree_refusals": tree_refusals,
            "accepted": bool(matched) and not tree_refusals and clean, "clean": clean}


# ------------------------------------------------------------ SCR engine build

def engine_name() -> str:
    return "execution-cli.exe" if os.name == "nt" else "execution-cli"


def _toolchain_environment(toolchain: str | None) -> dict | None:
    """With a named rustup toolchain, rustup must not install it when it is missing (the call fails instead)."""
    return {**os.environ, "RUSTUP_AUTO_INSTALL": "0"} if toolchain else None


def _version(tool: str, toolchain: str | None = None) -> str | None:
    command = [tool, f"+{toolchain}", "--version"] if toolchain else [tool, "--version"]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60,
                              env=_toolchain_environment(toolchain)).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_engine(scr, builds: int = 2, toolchain: str | None = None) -> dict:
    """``cargo build --release --locked --offline`` of SCR ``execution-cli`` into fresh target dirs.

    Returns the command, return codes, Cargo.lock digests before and after, the
    toolchain versions, the binary bytes of the first build and every build's
    digest. Target directories are temporary; the checkout is never a target.
    ``toolchain`` builds with that rustup toolchain (``cargo +TOOLCHAIN``)
    instead of the default one; a missing toolchain fails the build.
    """
    scr = Path(scr).resolve()
    manifest = scr / "crates" / "Cargo.toml"
    lock = scr / "crates" / "Cargo.lock"
    selector = [f"+{toolchain}"] if toolchain else []
    flags = ["build", "--release", "--locked", "--offline"]
    command = ["cargo", *selector, *flags, "--manifest-path", str(manifest), "-p", "execution-cli"]
    record = {"command": " ".join(["cargo", *selector, *flags, "--manifest-path", "<scr>/crates/Cargo.toml",
                                   "-p", "execution-cli"]),
              "cargo_target_dir": "fresh temporary directory per build", "cargo": _version("cargo", toolchain),
              "rustc": _version("rustc", toolchain), "cargo_lock_sha256_before": sha256(lock.read_bytes()).hexdigest(),
              "builds": [], "binary": None}
    if toolchain:
        record["toolchain"] = toolchain
    for _ in range(builds):
        with tempfile.TemporaryDirectory(prefix="ciw-lab-scr-target-") as target:
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=900,
                                           env={**(_toolchain_environment(toolchain) or os.environ),
                                                "CARGO_TARGET_DIR": target})
                returncode, log = completed.returncode, completed.stderr.strip().splitlines()[-3:]
            except subprocess.TimeoutExpired:
                returncode, log = None, ["cargo build exceeded 900 s and was stopped"]
            except OSError as exc:
                returncode, log = None, [f"cargo could not start ({type(exc).__name__})"]
            binary = Path(target) / "release" / engine_name()
            data = binary.read_bytes() if returncode == 0 and binary.is_file() else None
            record["builds"].append({"returncode": returncode,
                                     "binary_sha256": None if data is None else sha256(data).hexdigest(),
                                     "byte_count": None if data is None else len(data),
                                     "log_tail": [line.replace(str(scr), "<scr>") for line in log]})
            if data is not None and record["binary"] is None:
                record["binary"] = data
    record["cargo_lock_sha256_after"] = sha256(lock.read_bytes()).hexdigest()
    return record


def materialize(data: bytes, directory) -> Path:
    """Write an engine binary into a private directory and mark it executable."""
    path = Path(directory) / engine_name()
    path.write_bytes(data)
    path.chmod(0o700)
    return path


_RUN_KERNEL = r'''
import json, struct, sys
from pathlib import Path
root, engine = sys.argv[1:3]
sys.path.insert(0, root)
from execution.specification import ExecutionSpecification, HEAT_DIFFUSION_DESCRIPTOR, encode_heat_input
from execution.engine import run_specification
import hashlib
cases = json.loads(sys.stdin.read())
out = []
for steps, values in cases:
    spec = ExecutionSpecification(HEAT_DIFFUSION_DESCRIPTOR, b'', encode_heat_input(steps, values))
    result = run_specification(spec, cli_path=Path(engine))
    out.append({"status": result.status, "exit_code": result.exit_code,
                "values": None if result.output is None else list(struct.unpack('<' + 'q' * len(values), result.output))})
print(json.dumps({"descriptor_sha256": hashlib.sha256(HEAT_DIFFUSION_DESCRIPTOR).hexdigest(), "cases": out}))
'''


def run_heat_kernel(scr, engine, cases) -> dict:
    """Execute the SCR heat kernel through SCR's own Python API in an isolated interpreter."""
    with tempfile.TemporaryDirectory(prefix="ciw-lab-scr-api-") as directory:
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", _RUN_KERNEL, str(Path(scr).resolve()), str(engine)],
                                   input=json.dumps(cases), capture_output=True, text=True, timeout=120, cwd=directory)
    if completed.returncode:
        raise ValueError("SCR heat kernel API call failed: " + (completed.stderr.strip().splitlines() or ["no output"])[-1])
    return json.loads(completed.stdout)


def scr_workbench_integration(scr, engine, directory, sources) -> dict:
    """CIW's shared numerical-heat path: bind, execute, replay, save, reopen unbound, refuse replay."""
    from ..instruments import make_demo_run
    from ..session import Session
    from .exchange_provenance_bundles_fixtures import Client, execution_guard, source_payload
    directory = Path(directory)
    session = Session(make_demo_run(), directory / "bound")
    session.workbench.bind_workflow("numerical-heat", {"scr": Path(scr), "engine": Path(engine)})
    client = Client(session)
    runs = []
    for label, raw in sources:
        source = client.ok("source.add", source_payload("numerical-heat", raw, label))
        original = client.ok("operation.execute", {"operation_id": "ciw.numerical-heat.v1",
                                                    "parameters": {"source_id": source["source_id"]}})
        replay = client.ok("bundle.replay", {"bundle_id": original["bundle_id"]})
        first = client.ok("bundle.get", {"bundle_id": original["bundle_id"]})
        second = client.ok("bundle.get", {"bundle_id": replay["bundle"]["bundle_id"]})
        declared = json.loads(raw)
        runs.append({"label": label, "initial_values": declared["initial_values"], "steps": declared["steps"],
                     "values": first["steps"][0]["result"]["data"]["values"],
                     "replay_values": second["steps"][0]["result"]["data"]["values"],
                     "numerical_identity_equal": first["steps"][0]["numerical_result_id"]
                     == second["steps"][0]["numerical_result_id"],
                     "fresh_occurrences": len({first["steps"][0]["execution_id"], second["steps"][0]["execution_id"],
                                               first["verification"]["reproduction"]["execution_id"],
                                               second["verification"]["reproduction"]["execution_id"]}),
                     "replay_numerical_match": replay["replay_receipt"]["numerical_match"],
                     "verification_independent": first["verification"]["independent"],
                     "runtime": first["runtimes"]["scr"], "bundle_id": original["bundle_id"]})
    path = session.save_workspace(directory / "workspace.json")
    with execution_guard() as guard:
        reopened = Session.from_workspace(path, directory / "reopened")
    refused = Client(reopened).call("bundle.replay", {"bundle_id": runs[0]["bundle_id"]})
    return {"runs": runs, "reopen_attempts": list(guard["attempts"]),
            "reopened_bindings": sorted(k for k, v in reopened.workbench._bindings.items() if v),
            "reopened_available": sorted(o["operation_id"] for o in reopened.workbench.describe_operations()
                                         if o["available"]),
            "unbound_replay": refused["payload"] if refused["type"] == "error" else {"code": "not_refused"},
            "workspace_mentions_engine_path": str(Path(engine)) in path.read_text(encoding="utf-8")}


# --------------------------------------------------------- SET / PPDA exchange

EXCHANGE_OBSERVATION = {
    "schema": "notation.instrument.observation-batch.v1", "batch_id": "example:synthetic-position-1",
    "observed_at": "2026-09-20T12:00:00Z", "received_at": "2026-09-20T12:00:01Z", "clock_basis": "synthetic UTC fixture",
    "components": [{"name": "east", "value": 1.2, "unit": "m"}, {"name": "north", "value": -0.4, "unit": "m"}],
    "covariance": {"status": "reported", "variables": ["east", "north"], "units": ["m", "m"],
                   "matrix": [[0.04, 0.01], [0.01, 0.09]],
                   "frame": {"id": "frame:synthetic-plane", "semantics": "intrinsic_physical", "basis": ["east", "north"]},
                   "method": "declared synthetic fixture", "source_refs": ["example:fixture"], "calibration_refs": [],
                   "numerical_status": "unchecked"},
    "source_artifact_refs": ["example:fixture"], "admission_ref": "example:unresolved-declaration",
    "admission_status": "reference_only", "calibration_refs": [],
}


def set_exchange_inspection(set_repo, directory) -> dict:
    """Validate the synthetic observation with the pinned SET contracts module (executed in-process)."""
    from .. import exchange
    path = Path(directory) / "observation.json"
    path.write_text(json.dumps(EXCHANGE_OBSERVATION, indent=2), encoding="utf-8")
    before = path.read_bytes()
    report = exchange.inspect_exchange([path], validator_repo=Path(set_repo))
    unchanged = path.read_bytes() == before
    artifact = report["artifacts"][0]
    changed = json.loads(before)
    changed["covariance"]["matrix"] = [[1, 2], [2, 1]]
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        exchange.inspect_exchange([path], validator_repo=Path(set_repo))
        indefinite = "not_refused"
    except ValueError as exc:
        # The full message: SET raises the same exception class for other contract defects.
        indefinite = str(exc)
    return {"status": report["status"], "validator": report["validator"],
            "effective_rank": artifact["covariance_validation"]["effective_rank"],
            "identity_status": artifact["identity_status"], "authority": report["authority"],
            "unchanged_input": unchanged, "indefinite_covariance": indefinite}


_PRODUCE = r'''
import json, os
from bridge import instrumentation as acquisition
from execution import instrumentation as runtime
from execution.commitments import COMPUTATION_TAG, OUTPUT_TAG, canonical_u32, commit_hex
from execution.engine import ExecutionResult
from execution.specification import ExecutionSpecification
common = dict(components=[dict(name="east", value=1.2, unit="m"), dict(name="north", value=-0.4, unit="m")],
    covariance_status="reported", covariance=[[0.04, 0.01], [0.01, 0.09]],
    frame=dict(id="frame:synthetic-enu", semantics="tangent", basis=["east", "north"], evaluation_point=[-79.5, 43.7, 100.0]),
    covariance_method="synthetic declaration", covariance_source_refs=["example:source"], calibration_refs=[])
observation = acquisition.observation_batch_v1(batch_id="example:synthetic-batch", observed_at="2026-09-20T12:00:00Z",
    received_at="2026-09-20T12:00:01Z", clock_basis="synthetic UTC fixture", source_artifact_refs=["example:source"],
    admission_ref="example:unresolved-admission", **common)
spec = ExecutionSpecification(program=b"synthetic-fixture", configuration=b"", input_payload=json.dumps(observation, sort_keys=True).encode())
output = b"synthetic-output-not-an-engine-run"
output_id = commit_hex(OUTPUT_TAG, [output])
computation_id = commit_hex(COMPUTATION_TAG, [bytes.fromhex(spec.program_identity()), bytes.fromhex(spec.input_identity()),
    bytes.fromhex(output_id), canonical_u32(0)])
execution = ExecutionResult(specification=spec, specification_identity=spec.identity(), program_identity=spec.program_identity(),
    input_identity=spec.input_identity(), engine_occurrence=0, status="completed", exit_code=0, output=output,
    output_identity=output_id, computation_identity=computation_id, detail=None)
result = runtime.result_artifact_v1(execution, input_refs=[observation["batch_id"]], applicability="synthetic interoperability fixture only",
    created_at="2026-09-20T12:00:02Z", **common)
verification = runtime.verification_artifact_v1(subject_ref=result["result_id"], verifier_ref="example:fixture-checker",
    created_at="2026-09-20T12:00:03Z", checks=[dict(name="fixture assertion", outcome="failed", basis="intentional failed-check fixture")],
    limitations=["synthetic check; not independent verification"])
print(json.dumps([observation, result, verification], allow_nan=False))
'''


def exchange_roundtrip(ppda, scr_exchange, set_repo, directory) -> dict:
    """PPDA and SCR producers -> pinned SET validator (the tests/test_exchange_integration.py path)."""
    from .. import exchange
    directory = Path(directory)
    environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME",)}
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(ppda).resolve()), str(Path(scr_exchange).resolve())])
    completed = subprocess.run([sys.executable, "-B", "-c", _PRODUCE], cwd=directory, env=environment, text=True,
                               capture_output=True, timeout=60)
    if completed.returncode:
        raise ValueError("Exchange producers failed: " + (completed.stderr.strip().splitlines() or ["no output"])[-1])
    artifacts = json.loads(completed.stdout)
    paths = []
    for index, artifact in enumerate(artifacts):
        path = directory / f"artifact-{index}.json"
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        paths.append(path)
    report = exchange.inspect_exchange(paths, validator_repo=Path(set_repo))
    artifacts[1]["components"][0]["value"] += 1
    paths[1].write_text(json.dumps(artifacts[1]), encoding="utf-8")
    try:
        exchange.inspect_exchange(paths, validator_repo=Path(set_repo))
        refusal = "not_refused"
    except ValueError as exc:
        refusal = str(exc)
    return {"status": report["status"], "links": sorted(item["status"] for item in report["links"]),
            "verification_outcome": report["artifacts"][2]["artifact"]["outcome"],
            "authority": report["authority"], "changed_result_refusal": refusal}


def which(tool: str) -> str | None:
    return shutil.which(tool)
