"""Provision the pinned Julia runtime and T145's worker environment, separately from any lab run.

Reads the pin in ``src/ciw/lab/julia/julia-runtime.json`` (Julia 1.10.12 LTS), downloads the
official archive for this platform (or takes ``--archive``), requires its SHA-256 to equal the
pin and the entry in Julia's published checksum file, and extracts it under ``--prefix``. Then
instantiates the committed ``src/ciw/lab/julia/Manifest.toml`` into ``--depot`` from the Julia
package server (Pkg checks every package's git tree hash) and precompiles it for this CPU,
refusing a run that changes the committed Project.toml or Manifest.toml. Finally it starts the
worker once and requires its handshake to match the expected identity. It prints the identities
and the bindings a lab run takes:

    --provider julia=<prefix>/julia-1.10.12/bin/julia --provider julia-depot=<depot>

The worker never resolves, downloads or compiles packages while serving: it refuses to start on
a depot that was not precompiled here. ``--resolve`` re-resolves the manifest from Project.toml
(a maintainer step; review and commit the machine-generated result).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "src" / "ciw" / "lab" / "julia"
PIN = json.loads((PROJECT / "julia-runtime.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def host_platform() -> str:
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise SystemExit(f"No pinned Julia archive for machine {platform.machine()}")
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    if sys.platform == "win32":
        return "windows-x86_64"
    raise SystemExit(f"No pinned Julia archive for platform {sys.platform}")


def published_checksum(name: str) -> str:
    """The archive's SHA-256 in Julia's published checksum file."""
    with urllib.request.urlopen(PIN["checksums_url"], timeout=120) as response:
        lines = response.read().decode("ascii").splitlines()
    entries = {parts[1]: parts[0] for parts in (line.split() for line in lines) if len(parts) == 2}
    if name not in entries:
        raise SystemExit(f"{name} is not listed in {PIN['checksums_url']}")
    return entries[name]


def fetch(url: str, target: Path) -> None:
    print(f"+ download {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as response, open(target, "wb") as stream:
        shutil.copyfileobj(response, stream)


def extract(archive: Path, prefix: Path) -> None:
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as bundle:
            bundle.extractall(prefix, filter="data" if hasattr(tarfile, "data_filter") else None)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(prefix)
    else:
        raise SystemExit(f"Unsupported archive type: {archive.name}")


def julia_environment(depot: Path) -> dict:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("JULIA_")}
    environment.update({"JULIA_DEPOT_PATH": str(depot), "JULIA_LOAD_PATH": os.pathsep.join(["@", "@stdlib"]),
                        "JULIA_PKG_SERVER": PIN["package_server"]})
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", type=Path, required=True, help="Directory the Julia archive is extracted into")
    parser.add_argument("--depot", type=Path, required=True, help="Julia depot the worker environment is instantiated in")
    parser.add_argument("--archive", type=Path, help="An already downloaded official archive (checked the same way)")
    parser.add_argument("--resolve", action="store_true",
                        help="Re-resolve Manifest.toml from Project.toml (maintainers only; commit the result)")
    args = parser.parse_args()
    name = host_platform()
    entry = PIN["archives"][name]
    archive_name = entry["url"].rsplit("/", 1)[1]
    published = published_checksum(archive_name)
    if published != entry["sha256"]:
        raise SystemExit(f"The pinned sha256 of {archive_name} differs from Julia's checksum file ({published})")
    prefix, depot = args.prefix.resolve(), args.depot.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    depot.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else prefix / archive_name
    if not archive.is_file():
        fetch(entry["url"], archive)
    digest = sha256(archive)
    if digest != entry["sha256"]:
        raise SystemExit(f"{archive.name} has sha256 {digest}, not the pinned {entry['sha256']}")
    root = prefix / entry["root"]
    executable = root / entry["executable"]
    if not executable.is_file():
        extract(archive, prefix)
    version = subprocess.run([str(executable), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != f"julia version {PIN['version']}":
        raise SystemExit(f"{executable} reports {version!r}, not Julia {PIN['version']}")
    committed = {path.name: sha256(path) for path in (PROJECT / "Project.toml", PROJECT / "Manifest.toml")}
    steps = "Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()" if args.resolve else "Pkg.instantiate(); Pkg.precompile()"
    command = [str(executable), "--startup-file=no", "--history-file=no", f"--project={PROJECT}", "-e",
               f"import Pkg; {steps}"]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=julia_environment(depot), timeout=3600)
    after = {path.name: sha256(path) for path in (PROJECT / "Project.toml", PROJECT / "Manifest.toml")}
    if after != committed and not args.resolve:
        raise SystemExit("Instantiating changed the committed Project.toml or Manifest.toml; the environment is not "
                         "the committed one (git diff src/ciw/lab/julia)")
    sys.path.insert(0, str(ROOT / "src"))
    from ciw.lab import julia_worker

    session = julia_worker.WorkerSession.for_runtime(julia_worker.JuliaRuntime(executable, depot),
                                                     cwd=tempfile.gettempdir())
    try:
        session.start()
    except julia_worker.WorkerFailure as failure:
        raise SystemExit(f"The provisioned worker was refused at its handshake: {failure}") from failure
    finally:
        session.close()
    print(json.dumps({"julia_version": PIN["version"], "platform": name, "archive": archive_name,
                      "archive_sha256": digest, "executable_sha256": sha256(executable),
                      "project_sha256": after["Project.toml"], "manifest_sha256": after["Manifest.toml"],
                      "handshake_accepted": session.comparison["accepted"],
                      "packages_verified": session.comparison["packages_verified"],
                      "sysimage_sha256": session.comparison["sysimage_sha256"],
                      "bindings": [f"julia={executable}", f"julia-depot={depot}"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
