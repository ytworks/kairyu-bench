"""Protect failure/unknown preservation in the one-off N30 numerical audit."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kairyu_bench.wsp.manifest import load_manifest


class N30EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/fresh-n30-complete-audit.py"
        spec = importlib.util.spec_from_file_location("n30_audit", path)
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)
        cls.items = load_manifest().select(30, "dev")[:2]

    def fixture(self):
        rows, responses = [], []
        for item in self.items:
            rows.append({"id": item.id, "source": item.source,
                         "area": self.audit.SOURCE_BY_NAME[item.source].area,
                         "kind": item.kind, "tier": item.tier, "max_tokens": item.max_tokens,
                         "samples": item.samples, "score": 0.0, "sample_scores": [0.0], "error": None})
            responses.append({"id": item.id, "outcome": {"samples": []}, "error": None})
        return {x.id: x for x in self.items}, rows, responses

    def test_real_zero_is_scored_and_null_error_is_unknown(self):
        expected, rows, responses = self.fixture()
        rows[1].update(score=None, sample_scores=[], error="API failed")
        responses[1].update(outcome=None, error="API failed")
        result, issues = self.audit.item_evidence(expected, rows, responses)
        self.assertEqual(result[0]["score"], 0.0)
        self.assertEqual(result[0]["scoring_status"], "scored")
        self.assertIsNone(result[1]["score"])
        self.assertEqual(result[1]["error"], "API failed")
        self.assertEqual(result[1]["scoring_status"], "unknown_or_anomalous")
        self.assertEqual(len(issues["item_issues"]), 1)

    def test_missing_duplicate_and_unexpected_records_are_all_retained(self):
        expected, rows, responses = self.fixture()
        rows = [rows[0], dict(rows[0], score=1.0), dict(rows[1], id="unexpected:item")]
        result, issues = self.audit.item_evidence(expected, rows, responses[:1])
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0]["score"])
        self.assertEqual([x["score"] for x in result[0]["actual_item_records"]], [0.0, 1.0])
        self.assertIn("missing_item_record", result[1]["audit_issues"])
        self.assertIn("missing_response_record", result[1]["audit_issues"])
        self.assertEqual(issues["unexpected_items"][0]["id"], "unexpected:item")

    def test_nonfinite_evidence_is_serializable_without_zero_imputation(self):
        expected, rows, responses = self.fixture()
        rows[0].update(score=float("nan"), sample_scores=[float("inf")])
        result, issues = self.audit.item_evidence(expected, rows, responses)
        self.assertIsNone(result[0]["score"])
        self.assertIn("invalid_sample_scores", result[0]["audit_issues"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.json"
            self.audit.save(path, {"items": result, "issues": issues})
            saved = json.loads(path.read_text())
        self.assertEqual(saved["items"][0]["raw_score"], {"invalid_nonfinite_float": "nan"})
        self.assertIsNone(saved["items"][0]["score"])


    def test_versioned_owner_plan_ignores_invalid_old_original(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "launch-plan-approved.json").write_text("{}" + chr(92) + "n")
            actual = study / "launch-plan-approved-v2.json"
            actual.write_text(json.dumps({"approved_for_execution": True}))
            (study / "owner-start.json").write_text(json.dumps({
                "plan_sha256": hashlib.sha256(actual.read_bytes()).hexdigest()}))
            path, plan = self.audit.audit_launch_plan(study, actual)
            self.assertEqual(path, actual)
            self.assertTrue(plan["approved_for_execution"])

    def test_explicit_plan_requires_matching_owner_and_legacy_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            actual = study / "launch-plan-approved.json"
            actual.write_text("{}")
            self.assertEqual(self.audit.audit_launch_plan(study), (actual, {}))
            with self.assertRaises(FileNotFoundError):
                self.audit.audit_launch_plan(study, actual)
            (study / "owner-start.json").write_text(json.dumps({"plan_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "executed owner"):
                self.audit.audit_launch_plan(study, actual)
            with self.assertRaisesRegex(ValueError, "executed owner"):
                self.audit.audit_launch_plan(study)


if __name__ == "__main__":
    unittest.main()
