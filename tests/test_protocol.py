"""Acceptance checks for the authoritative, transport-independent session."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ciw.instruments import make_demo_run
from ciw.session import Session


def request(kind, payload=None, request_id="test-request", version=1):
    return {
        "protocol_version": version,
        "request_id": request_id,
        "type": kind,
        "payload": {} if payload is None else payload,
    }


class SessionContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output_dir = Path(self.directory.name)
        self.run = make_demo_run()
        self.session = Session(self.run, self.output_dir)

    def success(self, kind, payload=None):
        response = self.session.handle(request(kind, payload))
        self.assertEqual(response["protocol_version"], 1)
        self.assertEqual(response["request_id"], "test-request")
        self.assertEqual(response["type"], "response", response)
        return response["payload"]

    def test_initial_snapshot_is_small_and_scientific_record_is_separate(self):
        snapshot = self.success("session.get")
        self.assertEqual(snapshot, self.session.snapshot())
        self.assertTrue(snapshot["session_id"])
        self.assertEqual(snapshot["run"]["run_id"], self.run["run_id"])
        self.assertNotIn("time_s", snapshot["run"])
        self.assertNotIn("values", snapshot["run"]["channels"]["q"])
        self.assertEqual(snapshot["selection"]["revision"], 0)
        self.assertEqual(snapshot["selection"]["cursor_s"], 0)
        self.assertEqual(snapshot["selection"]["interval_s"], [0, self.run["metadata"]["duration_s"]])
        self.assertEqual(self.success("run.get"), self.run)

    def test_cursor_and_interval_have_independent_semantics(self):
        duration = self.run["metadata"]["duration_s"]
        interval = [duration / 4, duration / 2]
        first = self.success("selection.update", {"expected_revision": 0, "interval_s": interval, "channel": "v"})
        result = self.success("analysis.stats")
        files_before = sorted(self.output_dir.rglob("*.json"))
        second = self.success("selection.update", {"expected_revision": 1, "cursor_s": 3 * duration / 4})
        self.assertEqual(first["cursor_s"], 0)
        self.assertEqual(second["interval_s"], interval)
        self.assertEqual(second["channel"], "v")
        self.assertEqual(second["revision"], 2)
        self.assertEqual(sorted(self.output_dir.rglob("*.json")), files_before)
        self.assertEqual(self.success("result.get", {"result_id": result["result_id"]}), result)
        self.assertEqual(result["selection_revision"], 1)
        self.assertEqual(result["interval_s"], interval)

    def test_stale_revision_is_rejected_without_mutation(self):
        self.success("selection.update", {"expected_revision": 0, "cursor_s": 0.1})
        before = copy.deepcopy(self.session.snapshot())
        response = self.session.handle(request("selection.update", {"expected_revision": 0, "cursor_s": 0.2}))
        self.assertEqual(response["type"], "error")
        self.assertEqual(response["payload"]["code"], "revision_conflict")
        self.assertEqual(self.session.snapshot(), before)

    def test_invalid_payloads_are_atomic_and_session_remains_usable(self):
        duration = self.run["metadata"]["duration_s"]
        invalid_payloads = [
            {"expected_revision": 0, "cursor_s": -0.1},
            {"expected_revision": 0, "cursor_s": duration + 1},
            {"expected_revision": 0, "cursor_s": float("nan")},
            {"expected_revision": 0, "cursor_s": float("inf")},
            {"expected_revision": 0, "cursor_s": 10**400},
            {"expected_revision": 0, "cursor_s": True},
            {"expected_revision": 0, "interval_s": [1, 1]},
            {"expected_revision": 0, "interval_s": [2, 1]},
            {"expected_revision": 0, "interval_s": [0, float("inf")]},
            {"expected_revision": 0, "interval_s": [0, 10**400]},
            {"expected_revision": 0, "interval_s": [0]},
            {"expected_revision": 0, "interval_s": "0,1"},
            {"expected_revision": 0, "channel": "nonexistent"},
            {"expected_revision": 0, "cursor_s": 0.1, "channel": "nonexistent"},
            {"expected_revision": -1, "cursor_s": 0.1},
            {"expected_revision": True, "cursor_s": 0.1},
            {"cursor_s": 0.1},
        ]
        before = copy.deepcopy(self.session.snapshot())
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.session.handle(request("selection.update", payload))
                self.assertEqual(response["type"], "error", response)
                self.assertTrue(response["payload"]["code"])
                self.assertEqual(self.session.snapshot(), before)
                self.assertEqual(self.success("session.get"), before)

    def test_bad_envelopes_and_unknown_operations_return_errors(self):
        malformed = [
            [], None, "request", {},
            request("session.get", version=2),
            request("session.get", version=True),
            request("session.get", version="1"),
            request("session.get", payload=[]),
            request("does.not.exist"),
        ]
        for message in malformed:
            with self.subTest(message=message):
                response = self.session.handle(message)
                self.assertEqual(response["type"], "error", response)
                self.assertEqual(response["protocol_version"], 1)
                self.assertTrue(response["payload"]["code"])
        self.success("session.get")

    def test_analysis_results_capture_provenance_and_are_persisted(self):
        for operation, operation_id in [("analysis.stats", "statistics.v1"), ("analysis.spectrum", "spectrum.periodogram.v1")]:
            with self.subTest(operation=operation):
                result = self.success(operation)
                self.assertEqual(result["evidence_id"], self.run["evidence_id"])
                self.assertEqual(result["run_id"], self.run["run_id"])
                self.assertEqual(result["operation_id"], operation_id)
                self.assertEqual(result["verification_status"], "not_verified")
                self.assertIsNone(result["verification_id"])
                self.assertEqual(result["selection_revision"], 0)
                self.assertEqual(result["channel"], "q")
                self.assertEqual(len({result["evidence_id"], result["execution_id"], result["result_id"]}), 3)
                self.assertTrue(result["created_at"])
                records = [json.loads(path.read_text(encoding="utf-8")) for path in self.output_dir.rglob("*.json")]
                self.assertIn(result, records)
                self.assertEqual(self.success("result.get", {"result_id": result["result_id"]}), result)

    def test_explicit_analysis_interval_does_not_mutate_shared_selection(self):
        duration = self.run["metadata"]["duration_s"]
        before = copy.deepcopy(self.session.snapshot()["selection"])
        result = self.success("analysis.stats", {"channel": "energy", "interval_s": [0, duration / 2]})
        self.assertEqual(result["channel"], "energy")
        self.assertEqual(result["interval_s"], [0, duration / 2])
        self.assertEqual(self.session.snapshot()["selection"], before)

    def test_invalid_analyses_do_not_create_results(self):
        before = list(self.output_dir.rglob("*.json"))
        for kind, payload in [
            ("analysis.stats", {"channel": "invalid"}),
            ("analysis.stats", {"interval_s": [1, 1]}),
            ("analysis.spectrum", {"interval_s": [0, float("nan")]}),
            ("sample.get", {"time_s": float("inf")}),
            ("result.get", {"result_id": "../../outside"}),
        ]:
            with self.subTest(kind=kind, payload=payload):
                response = self.session.handle(request(kind, payload))
                self.assertEqual(response["type"], "error", response)
        self.assertEqual(list(self.output_dir.rglob("*.json")), before)

    def test_saved_workspace_reopens_results_without_running_instruments(self):
        self.success("selection.update", {"expected_revision": 0, "cursor_s": 0.5, "channel": "v"})
        stats = self.success("analysis.stats")
        spectrum = self.success("analysis.spectrum")
        selected = copy.deepcopy(self.session.snapshot()["selection"])
        workspace = self.output_dir / "saved" / "workspace.json"
        self.session.save_workspace(workspace)
        self.assertTrue(workspace.is_file())
        with patch("ciw.session.compute_statistics", side_effect=AssertionError("workspace restore recalculated statistics")), patch("ciw.session.compute_spectrum", side_effect=AssertionError("workspace restore recalculated spectrum")):
            restored = Session.from_workspace(workspace, output_dir=self.output_dir / "reopened")
            self.assertEqual(restored.snapshot()["selection"], selected)
            self.assertEqual(restored.handle(request("run.get"))["payload"], self.run)
            summaries = restored.snapshot()["results"]
            listed = restored.handle(request("result.list"))
            self.assertEqual(listed["type"], "response")
            self.assertEqual(listed["payload"]["results"], summaries)
            self.assertEqual({item["result_id"] for item in summaries}, {stats["result_id"], spectrum["result_id"]})
            for summary in summaries:
                self.assertNotIn("data", summary)
            for original in (stats, spectrum):
                response = restored.handle(request("result.get", {"result_id": original["result_id"]}))
                self.assertEqual(response["type"], "response")
                self.assertEqual(response["payload"], original)


if __name__ == "__main__":
    unittest.main()
