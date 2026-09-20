import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tools.release_export import Refusal, export, prepare, scope_sha256


class ExportBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.repo / "LICENSE").write_text("Synthetic test license notice\n")
        (self.repo / "public.py").write_text("x = 1\n")
        (self.repo / "AGENTS.md").write_text("Private instructions\n")
        (self.repo / "linked.py").symlink_to("public.py")
        self.git("add", ".")
        self.git("commit", "-qm", "Fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.spec = {"schema": "publication-export.v1", "publication": "public-release",
                     "source_commit": self.commit, "license": "MPL-2.0",
                     "reviews": {key: {"status": "approved", "source_commit": self.commit,
                                       "reference": "Synthetic test review only"}
                                 for key in ("copyright_and_license", "privacy", "technical_validation")},
                     "files": [self.file_spec("LICENSE", "license_notice"), self.file_spec("public.py", "source")]}
        self.bind_scope(self.spec)

    def bind_scope(self, spec):
        for review in spec["reviews"].values():
            review["scope_sha256"] = scope_sha256(spec)

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args]).decode()

    def file_spec(self, path, classification):
        return {"path": path, "classification": classification, "license": "MPL-2.0",
                "sha256": hashlib.sha256((self.repo / path).read_bytes()).hexdigest()}

    def test_exact_commit_only_and_deterministic_archive(self):
        (self.repo / "public.py").write_text("private uncommitted modification\n")
        (self.repo / "customer.txt").write_text("untracked private data\n")
        a, b = self.root / "a.tgz", self.root / "b.tgz"
        export(self.repo, self.spec, a)
        export(self.repo, self.spec, b)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        with tarfile.open(a) as archive:
            self.assertEqual(set(archive.getnames()), {"LICENSE", "public.py", "release-manifest.json"})
            self.assertEqual(archive.extractfile("public.py").read(), b"x = 1\n")
            manifest = json.load(archive.extractfile("release-manifest.json"))
            self.assertNotIn("reviews", manifest)
            self.assertEqual(manifest["scope_sha256"], scope_sha256(self.spec))

    def test_missing_or_stale_review_refuses(self):
        for key in self.spec["reviews"]:
            spec = copy.deepcopy(self.spec)
            spec["reviews"][key]["source_commit"] = "0" * 40
            with self.subTest(key=key), self.assertRaises(Refusal):
                prepare(self.repo, spec)

    def test_unsafe_paths_and_modes_refuse(self):
        for name in ("../public.py", "/public.py", "AGENTS.md", "linked.py", "a//b", "release-manifest.json",
                     "release-manifest.json/child", "Release-Manifest.json"):
            spec = copy.deepcopy(self.spec)
            spec["files"][1]["path"] = name
            self.bind_scope(spec)
            with self.subTest(name=name), self.assertRaises(Refusal):
                prepare(self.repo, spec)

    def test_digest_or_classification_mismatch_refuses(self):
        for key, value in (("sha256", "0" * 64), ("classification", "customer_profile")):
            spec = copy.deepcopy(self.spec)
            spec["files"][1][key] = value
            self.bind_scope(spec)
            with self.subTest(key=key), self.assertRaises(Refusal):
                prepare(self.repo, spec)

    def test_session_links_refuse(self):
        (self.repo / "public.py").write_text("https://claude.ai/chat/private-session\n")
        self.git("add", "public.py")
        self.git("commit", "-qm", "Forbidden link fixture")
        commit = self.git("rev-parse", "HEAD").strip()
        self.spec["source_commit"] = commit
        for review in self.spec["reviews"].values():
            review["source_commit"] = commit
        self.spec["files"][1] = self.file_spec("public.py", "source")
        self.bind_scope(self.spec)
        with self.assertRaises(Refusal):
            prepare(self.repo, self.spec)

    def test_existing_export_is_not_overwritten(self):
        target = self.root / "existing.tgz"
        target.write_bytes(b"previous export")
        with self.assertRaises(FileExistsError):
            export(self.repo, self.spec, target)
        self.assertEqual(target.read_bytes(), b"previous export")

    def test_agpl_requires_selective_review(self):
        self.spec["license"] = "AGPL-3.0-only"
        self.bind_scope(self.spec)
        with self.assertRaises(Refusal):
            prepare(self.repo, self.spec)

    def test_nonstring_or_blank_review_references_refuse(self):
        for value in (None, False, 0, {}, [], "", "  \n"):
            spec = copy.deepcopy(self.spec)
            spec["reviews"]["privacy"]["reference"] = value
            with self.subTest(value=value), self.assertRaisesRegex(Refusal, "completed review"):
                prepare(self.repo, spec)

    def test_review_cannot_be_reused_for_another_allowlist_or_license(self):
        mutations = [
            lambda s: s.update(license="Apache-2.0"),
            lambda s: s["files"].pop(),
            lambda s: s["files"][1].update(license="Apache-2.0"),
            lambda s: s["files"][1].update(classification="documentation"),
        ]
        for change in mutations:
            spec = copy.deepcopy(self.spec)
            change(spec)
            with self.subTest(scope=scope_sha256(spec)), self.assertRaisesRegex(Refusal, "scope-bound"):
                prepare(self.repo, spec)

    def test_git_replace_cannot_substitute_an_unreviewed_tree(self):
        (self.repo / "public.py").write_text("replacement-only data\n")
        self.git("add", "public.py")
        self.git("commit", "-qm", "Unreviewed replacement tree")
        replacement = self.git("rev-parse", "HEAD").strip()
        self.git("replace", self.commit, replacement)
        manifest, blobs = prepare(self.repo, self.spec)
        self.assertEqual(dict((path, data) for path, data, _ in blobs)["public.py"], b"x = 1\n")
        self.assertEqual(manifest["source_commit"], self.commit)
        # Even a declaration carrying the substituted bytes cannot mislabel
        # them as content from the original reviewed commit.
        self.spec["files"][1] = self.file_spec("public.py", "source")
        self.bind_scope(self.spec)
        with self.assertRaisesRegex(Refusal, "digest absent or mismatched"):
            prepare(self.repo, self.spec)

    def test_submodule_tree_entry_is_not_exportable(self):
        self.git("update-index", "--add", "--cacheinfo", "160000", self.commit, "external-module")
        self.git("commit", "-qm", "Synthetic gitlink without downloading a submodule")
        self.spec["source_commit"] = self.git("rev-parse", "HEAD").strip()
        for review in self.spec["reviews"].values():
            review["source_commit"] = self.spec["source_commit"]
        self.spec["files"][1].update(path="external-module")
        self.bind_scope(self.spec)
        with self.assertRaisesRegex(Refusal, "Symlinks and submodules"):
            prepare(self.repo, self.spec)

    def test_scope_digest_command_does_not_create_approval_or_export(self):
        spec = copy.deepcopy(self.spec)
        del spec["reviews"]
        path = self.root / "scope.json"
        path.write_text(json.dumps(spec))
        before = set(self.root.iterdir())
        process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "release_export.py"),
                                  "--spec", str(path), "--scope-digest"],
                                 text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout), {"scope_sha256": scope_sha256(spec)})
        self.assertEqual(json.loads(path.read_text()), spec)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_invalid_review_container_refuses_without_traceback(self):
        for reviews in (None, [], {"copyright_and_license": None}):
            spec = copy.deepcopy(self.spec)
            spec["reviews"] = reviews
            with self.subTest(reviews=reviews), self.assertRaises(Refusal):
                prepare(self.repo, spec)


if __name__ == "__main__":
    unittest.main()
