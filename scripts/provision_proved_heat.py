"""Provision the proved-heat gate: exact SCR and SP1 sources, the verified guest and native hosts.

Run after the host-build cache is restored and before it is saved. The guest
recipe rebuild and every source check run on every call, cache hit or not; the
native hosts are built only when the restored cache does not already hold
them. No proof, verdict or acceptance receipt is ever cached.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_proved_heat import SCR_REVISION, SCR_TREE, SP1_REVISION, SP1_TREE, source_identity  # noqa: E402
from provider_checkouts import extra_pin, validate_checkout  # noqa: E402

RUST = "1.94.0"
SUCCINCT_URL = "https://github.com/succinctlabs/rust/releases/download/succinct-1.94.0-64bit/rust-toolchain-x86_64-unknown-linux-gnu.tar.gz"
SUCCINCT_SHA256 = "12c94435d41bfe4e20131bbcce40b35abd32270ad792befc653af4e3fabc192f"
RECIPE_IDENTITY = "6e5d1687bcc55243d712553a2b7768b6c587a76418bb48a7a2c44224470d423d"
GUEST_SHA256 = "a14e3750da7e221d31842bd6cf983fcc8c0f530b2811537e2a9a9fe803dacf82"
SCR_REPOSITORY = "https://github.com/giasonpooni/Scientific-Computation-Runtime.git"
SP1_REPOSITORY = "https://github.com/" + extra_pin("sp1")["repository"] + ".git"


def call(*command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def git(path, *args):
    return subprocess.run(["git", "--no-replace-objects", "-c", "core.autocrlf=false", "-C", str(path), *args],
                          check=True, capture_output=True).stdout


def resources(stack: Path, output: Path) -> None:
    memory = {key: int(value.split()[0]) * 1024 for key, value in
              (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    stack.mkdir(parents=True, exist_ok=True)
    info = {"available_memory_bytes": memory["MemAvailable"], "free_disk_bytes": shutil.disk_usage(stack).free}
    (output / "resources.json").write_text(json.dumps(info, indent=2) + "\n")
    if info["available_memory_bytes"] < 7 * 1024**3 or info["free_disk_bytes"] < 20 * 1024**3:
        raise SystemExit("Real SP1 gate needs at least 7 GiB available RAM and 20 GiB free disk; no mock fallback")


def checkout(repository: str, revision: str, destination: Path) -> None:
    if not destination.exists():
        call("git", "-c", "core.autocrlf=false", "clone", "--no-checkout", repository, destination)
        call("git", "-C", destination, "-c", "core.autocrlf=false", "checkout", "--detach", revision)


def succinct_toolchain(stack: Path) -> None:
    archive = stack / "succinct-1.94.0-64bit.tar.gz"
    toolchain = stack / "succinct-1.94.0-64bit"
    call("curl", "--fail", "--location", "--retry", "3", "--output", archive, SUCCINCT_URL)
    call("sha256sum", "--check", "--strict", input=f"{SUCCINCT_SHA256}  {archive}\n".encode())
    shutil.rmtree(toolchain, ignore_errors=True)
    toolchain.mkdir()
    call("tar", "-xzf", archive, "-C", toolchain)
    call("rustup", "toolchain", "link", "succinct", toolchain)
    call("rustc", "+succinct", "-vV")


def guest(stack: Path, output: Path) -> None:
    scr = validate_checkout(stack / "scr", SCR_REVISION)
    validate_checkout(stack / "notationsystems/SP1-zero-knowledge-virtual-machine", SP1_REVISION)
    sys.path.insert(0, str(scr))
    from execution.build import load_recipe, verify_build
    recipe = load_recipe(scr / "zk/recipes/sp1-heat.recipe")
    if recipe.identity() != RECIPE_IDENTITY:
        raise SystemExit("Committed heat recipe identity changed")
    artifact = verify_build(recipe, GUEST_SHA256, repo_root=scr)
    destination = stack / "artifacts"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir()
    shutil.copyfile(artifact.elf_path, destination / "sp1-heat.elf")
    report = {**dataclasses.asdict(artifact), "toolchain_identity": recipe.toolchain_id,
              "compiler_archive_sha256": SUCCINCT_SHA256, "sp1_revision": recipe.fork_commit,
              "guest_build": "verified_against_committed_registry"}
    (output / "build.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def hosts(stack: Path) -> None:
    engine = stack / "native-target/release/execution-cli"
    prover = stack / "proof-target/release/sp1-host"
    if engine.is_file() and prover.is_file():
        print("Native hosts restored from the pin-keyed cache", flush=True)
        return
    call("cargo", f"+{RUST}", "build", "--release", "--locked", "--offline", "--manifest-path", stack / "scr/crates/Cargo.toml",
         "--target-dir", stack / "native-target", "-p", "execution-cli")
    call("cargo", f"+{RUST}", "build", "--release", "--locked", "--manifest-path", stack / "scr/zk/Cargo.toml",
         "--target-dir", stack / "proof-target", "-p", "sp1-adapter", "--bin", "sp1-host")


def source_checks(stack: Path, output: Path) -> None:
    source_identity(stack / "scr", SCR_REVISION, SCR_TREE)
    build = stack / "notationsystems/SP1-zero-knowledge-virtual-machine"
    reference = stack / "sp1-reference"
    if git(build, "rev-parse", "HEAD").decode().strip() != SP1_REVISION or git(build, "rev-parse", "HEAD^{tree}").decode().strip() != SP1_TREE:
        raise SystemExit("SP1 build checkout no longer has the pinned commit and tree")
    git(build, "diff", "--exit-code", "HEAD", "--")
    # SP1's nested build may write generated files inside its source tree.
    # Clone committed Git objects, never copy that working tree or clean it.
    shutil.rmtree(reference, ignore_errors=True)
    call("git", "-c", "core.autocrlf=false", "clone", "--no-checkout", "--no-hardlinks", build, reference)
    git(reference, "checkout", "--detach", SP1_REVISION)
    source_identity(reference, SP1_REVISION, SP1_TREE)
    for entry in git(reference, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, _ = metadata.decode("ascii").split()
        if kind == "commit" and mode == "160000":
            continue
        name = os.fsdecode(raw_name)
        actual, expected = build / name, reference / name
        if actual.is_symlink() != expected.is_symlink():
            raise SystemExit("SP1 tracked file type changed during build: " + name)
        raw = os.fsencode(os.readlink(actual)) if actual.is_symlink() else actual.read_bytes()
        pinned = os.fsencode(os.readlink(expected)) if expected.is_symlink() else expected.read_bytes()
        if raw != pinned or (mode != "120000" and bool(actual.stat().st_mode & 0o111) != (mode == "100755")):
            raise SystemExit("SP1 tracked source changed during build: " + name)
    generated = [os.fsdecode(name) for name in git(build, "ls-files", "--others", "-z").split(b"\0") if name]
    report = {"sp1_revision": SP1_REVISION, "sp1_tree": SP1_TREE, "tracked_build_source_bytes": "unchanged",
              "runtime_reference": "separate_clean_checkout", "generated_build_files_excluded_from_reference": generated}
    (output / "source-checks.json").write_text(json.dumps(report, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True, type=Path, help="Gate stack directory (host builds are cached by pin)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Evidence directory for resource, build and source reports")
    args = parser.parse_args(argv)
    stack, output = args.stack.expanduser().resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resources(stack, output)
    call(sys.executable, "-m", "pip", "install", "setuptools>=77", "wheel")
    call("rustup", "toolchain", "install", RUST, "--profile", "minimal")
    checkout(SCR_REPOSITORY, SCR_REVISION, stack / "scr")
    checkout(SP1_REPOSITORY, SP1_REVISION, stack / "notationsystems/SP1-zero-knowledge-virtual-machine")
    succinct_toolchain(stack)
    guest(stack, output)
    hosts(stack)
    source_checks(stack, output)
    print("PASS: exact SCR and SP1 sources, verified heat guest and pinned native hosts")


if __name__ == "__main__":
    main()
