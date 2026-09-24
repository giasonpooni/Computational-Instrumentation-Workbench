"""Read-only provisioning checks shared by source-pinned integration gates."""
from __future__ import annotations

from hashlib import new as new_hash
import os
from pathlib import Path
import re
import subprocess


_CACHE_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _git(path: Path, *arguments: str) -> bytes:
    # Avoid even Git's optional index refresh when inspecting operator-owned
    # checkouts. Command-scoped configuration never changes their config files.
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "core.autocrlf=false",
             "-C", str(path), *arguments], check=True, capture_output=True, timeout=60,
            env=environment,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify provider checkout: {path}") from exc


def validate_checkout(path: Path, revision: str) -> Path:
    """Require the exact existing pin and tracked bytes, without checking out.

    Gitlinks remain separately owned boundaries: an uninitialized submodule is
    permitted, while Git reports initialized submodule drift as dirty. The
    operation-specific runtime still enforces its executable source allowlist.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Provider checkout is unavailable: {path}")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Provider revision must be a full lowercase commit ID")
    if Path(os.fsdecode(_git(path, "rev-parse", "--show-toplevel")).strip()).resolve() != path:
        raise ValueError(f"Provider path must be a repository root: {path}")
    if _git(path, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError(f"Provider checkout has the wrong pin; require {revision}: {path}")
    if _git(path, "status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none"):
        raise ValueError(f"Provider checkout is dirty: {path}")
    exclusions = [f":(exclude,glob)**/{name}/**" for name in sorted(_CACHE_DIRS)]
    if _git(path, "ls-files", "--others", "-z", "--", ".", *exclusions):
        raise ValueError(f"Provider checkout contains untracked files outside runtime caches: {path}")
    object_format = _git(path, "rev-parse", "--show-object-format").decode().strip()
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"Unsupported Git object format: {object_format}")
    for entry in _git(path, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.decode("ascii").split()
        if kind == "commit" and mode == "160000":
            continue
        name = os.fsdecode(raw_name)
        item = path / name
        try:
            if kind != "blob" or (mode != "120000" and item.is_symlink()):
                raise ValueError(f"Provider tracked file changed type: {item}")
            data = os.fsencode(os.readlink(item)) if item.is_symlink() else item.read_bytes()
            if mode != "120000" and os.name == "posix" and bool(item.stat().st_mode & 0o111) != (mode == "100755"):
                raise ValueError(f"Provider tracked executable mode differs from its pin: {item}")
        except OSError as exc:
            raise ValueError(f"Cannot read pinned provider file: {item}") from exc
        actual = new_hash(object_format, b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError(f"Provider tracked bytes differ from their pin: {item}")
    return path


_ROOT = Path(__file__).resolve().parents[1]


def descriptor_pin(kind: str, role: str) -> dict:
    """The exact pin a pipeline descriptor declares for one provider role (the pin definition)."""
    import json
    descriptor = json.loads((_ROOT / "src/ciw/pipelines/descriptors" / f"{kind}.json").read_text(encoding="utf-8"))
    pins = [step["pin"] for step in descriptor["steps"] if step["role"] == role and "pin" in step]
    if len(pins) != 1:
        raise ValueError(f"{kind} declares no single pinned {role} step")
    return dict(pins[0])


def extra_pin(name: str) -> dict:
    """A pin the CI gate registry declares beyond the descriptors (checkers, producers, vendored sources)."""
    import json
    registry = json.loads((_ROOT / "ci/gates.json").read_text(encoding="utf-8"))
    return dict(registry["extra_pins"][name])


def clone_at(repository: str, revision: str, destination: Path) -> Path:
    """Clone ``owner/name`` and detach at the exact revision; the tracked bytes are then validated."""
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", f"https://github.com/{repository}.git", str(destination)],
                   check=True, timeout=600)
    subprocess.run(["git", "-C", str(destination), "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", revision],
                   check=True, timeout=600)
    return validate_checkout(destination, revision)


def provider_pin(name: str) -> dict:
    """The pin a provider descriptor declares (terminal instruments, ESM, the Julia worker)."""
    import json
    return dict(json.loads((_ROOT / "src/ciw/pipelines/providers" / f"{name}.json").read_text(encoding="utf-8"))["pin"])


def descriptor_pins(kind: str, root: Path | None = None) -> dict:
    """Every pinned provider step of a pipeline descriptor, by role (the pin definition)."""
    import json
    root = Path(root) if root is not None else _ROOT
    descriptor = json.loads((root / "src/ciw/pipelines/descriptors" / f"{kind}.json").read_text(encoding="utf-8"))
    return {step["role"]: dict(step["pin"]) for step in descriptor["steps"] if "pin" in step}
