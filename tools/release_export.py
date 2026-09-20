#!/usr/bin/env python3
"""Export only explicitly approved blobs from one Git commit; never Git history.

This is a mechanical publication boundary, not a copyright or secrets audit.
The approval references must come from completed human review, not this tool.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


PUBLIC_CLASSES = {"source", "test", "documentation", "synthetic_fixture", "license_notice"}
INTERNAL = re.compile(r"(^|/)(\.git|\.claude|\.codex|\.env(?:\..*)?|AGENTS\.md|CLAUDE\.md|[^/]*handoff[^/]*|[^/]*transcript[^/]*|customer_profiles|production_profiles|private)(/|$)", re.I)
SESSION_URL = re.compile(rb"https?://(?:chatgpt\.com/(?:c|share|codex/tasks)/|claude\.ai/(?:chat|share|code)/|[^/\s]+/sessions/)[^\s]+", re.I)


class Refusal(ValueError):
    pass


def git(repo: Path, *args: str) -> bytes:
    # Local replacement refs must never reinterpret a reviewed commit or blob.
    result = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise Refusal("Git operation failed: " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def checked_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise Refusal("A release path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", ".."} for p in value.split("/")) or str(path) != value:
        raise Refusal("Noncanonical release path: " + value)
    if INTERNAL.search(value) or value.split("/", 1)[0].casefold() == "release-manifest.json":
        raise Refusal("Internal or reserved release path: " + value)
    return value


def scope_sha256(spec: dict) -> str:
    """Identify exactly what a human reviewed; this grants no approval.

    Canonical UTF-8 JSON: sorted object keys, compact separators, ASCII escape
    encoding, and no nonfinite values. Array order remains significant.
    """
    if not isinstance(spec, dict):
        raise Refusal("The export specification must be an object")
    keys = ("publication", "source_commit", "license", "files")
    if any(key not in spec for key in keys):
        raise Refusal("Publication, source_commit, license, and files are required for the review scope")
    try:
        payload = json.dumps({key: spec[key] for key in keys}, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Refusal("Review scope must be finite JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def prepare(repo: Path, spec: dict) -> tuple[dict, list[tuple[str, bytes, int]]]:
    if not isinstance(spec, dict):
        raise Refusal("The export specification must be an object")
    if spec.get("schema") != "publication-export.v1":
        raise Refusal("Expected publication-export.v1")
    commit = spec.get("source_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refusal("An exact 40-character source commit is required")
    if git(repo, "rev-parse", "--verify", commit + "^{commit}").decode().strip() != commit:
        raise Refusal("Source must be an exact commit, not a tag or moving branch")
    if spec.get("publication") != "public-release":
        raise Refusal("This exporter accepts reviewed public releases only")
    if not isinstance(spec.get("license"), str) or spec["license"] not in {"MPL-2.0", "Apache-2.0", "AGPL-3.0-only", "AGPL-3.0-or-later"}:
        raise Refusal("No approved public source license declared")
    if spec["license"].startswith("AGPL") and spec.get("network_reciprocity_review") != "approved":
        raise Refusal("AGPL requires an explicit service-specific review")
    reviews = spec.get("reviews", {})
    if not isinstance(reviews, dict):
        raise Refusal("Reviews must be an object containing completed human review records")
    expected_scope = scope_sha256(spec)
    for key in ("copyright_and_license", "privacy", "technical_validation"):
        review = reviews.get(key, {})
        if not isinstance(review, dict):
            raise Refusal("Review record must be an object: " + key)
        reference = review.get("reference")
        if (review.get("status") != "approved" or review.get("source_commit") != commit
                or review.get("scope_sha256") != expected_scope
                or not isinstance(reference, str) or not reference.strip()):
            raise Refusal("Missing commit-and-scope-bound completed review: " + key)
    files = spec.get("files")
    if not isinstance(files, list) or not files:
        raise Refusal("An explicit nonempty file allowlist is required")
    tree = {}
    for entry in git(repo, "ls-tree", "-rz", commit).split(b"\x00"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, kind, sha = metadata.decode().split()
            tree[name.decode()] = (mode, kind, sha)
    blobs = []
    manifest_files = []
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            raise Refusal("Each allowlist item must be an object")
        path = checked_path(item.get("path"))
        if path in seen:
            raise Refusal("Duplicate release path: " + path)
        seen.add(path)
        if item.get("classification") not in PUBLIC_CLASSES:
            raise Refusal("No public classification for " + path)
        license_id = item.get("license")
        if not isinstance(license_id, str) or not license_id.strip():
            raise Refusal("Every file requires its own license expression: " + path)
        if path not in tree:
            raise Refusal("Allowlisted path absent at source commit: " + path)
        mode, kind, sha = tree[path]
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise Refusal("Symlinks and submodules require separate reviewed exports: " + path)
        data = git(repo, "cat-file", "blob", sha)
        digest = hashlib.sha256(data).hexdigest()
        if item.get("sha256") != digest:
            raise Refusal("Approved content digest absent or mismatched: " + path)
        if SESSION_URL.search(data):
            raise Refusal("Conversation/session URL in release content: " + path)
        blobs.append((path, data, 0o755 if mode == "100755" else 0o644))
        manifest_files.append({"path": path, "sha256": digest, "size_bytes": len(data),
                               "license": license_id, "classification": item["classification"]})
    if not any(item["classification"] == "license_notice" for item in files):
        raise Refusal("No license notice included in release")
    manifest = {"schema": "public-release-manifest.v1", "source_commit": commit,
                "license": spec["license"], "scope_sha256": expected_scope,
                "files": sorted(manifest_files, key=lambda x: x["path"])}
    return manifest, sorted(blobs)


def export(repo: Path, spec: dict, output: Path) -> dict:
    manifest, blobs = prepare(repo, spec)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    blobs.append(("release-manifest.json", payload, 0o644))
    # Exclusive creation prevents overwriting a previously reviewed export.
    with output.open("xb") as destination:
        with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path, data, mode in sorted(blobs):
                    info = tarfile.TarInfo(path)
                    info.size, info.mode, info.mtime = len(data), mode, 0
                    archive.addfile(info, io.BytesIO(data))
    return {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": len(blobs), "source_commit": spec["source_commit"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scope-digest", action="store_true",
                        help="Print the canonical review-scope digest only; does not approve or export")
    args = parser.parse_args()
    if args.scope_digest and args.output:
        parser.error("--scope-digest cannot be combined with --output")
    if not args.scope_digest and args.repo is None:
        parser.error("--repo is required to validate or export")
    try:
        spec = json.loads(args.spec.read_text())
        if args.scope_digest:
            result = {"scope_sha256": scope_sha256(spec)}
        else:
            result = export(args.repo, spec, args.output) if args.output else prepare(args.repo, spec)[0]
    except (Refusal, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, "REFUSED: " + str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
