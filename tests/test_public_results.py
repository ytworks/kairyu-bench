from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from pathlib import Path

from kairyu_bench.public_results import (
    best_deepswe,
    load_public_results,
    references_for,
    validate_catalog,
)


class PublicResultsTest(unittest.TestCase):
    def test_snapshot_has_all_models_and_configurations_without_model_alias_guessing(
        self,
    ):
        catalog = load_public_results()
        rows = [r for r in catalog["results"] if r["benchmark"] == "deepswe"]
        self.assertEqual(len(rows), 70)
        self.assertEqual(len(best_deepswe(catalog)), 28)
        self.assertEqual(len(catalog["table_models"]), 19)
        opus = next(
            r
            for r in rows
            if r["model"] == "claude-opus-4-8" and r["reasoning_effort"] == "max"
        )
        self.assertEqual(
            (opus["passed_attempts"], opus["scored_attempts"], opus["task_count"]),
            (253, 429, 111),
        )
        self.assertNotIn("DeepSeek-V4-Flash-0731", catalog["deepswe_display_mapping"])

    def test_missing_score_remains_null_and_no_source_no_score(self):
        catalog = load_public_results()
        missing = next(r for r in catalog["results"] if r["score_percent"] is None)
        self.assertEqual(missing["status"], "not-reported-in-reviewed-sources")
        broken = copy.deepcopy(catalog)
        published = next(r for r in broken["results"] if r["score_percent"] is not None)
        published["source_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "dated source"):
            validate_catalog(broken)

    def test_public_values_are_dated_noncomparable_references(self):
        rows = references_for(["deepswe", "gpqa-diamond"])
        self.assertTrue(rows)
        self.assertTrue(
            all(r["comparable"] is False and r["source"]["retrieved_at"] for r in rows)
        )
        self.assertTrue(
            all(r["benchmark"] in {"deepswe", "gpqa-diamond"} for r in rows)
        )

    def test_generated_document_and_catalog_match_official_snapshot(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/update_model_comparison.py", "--check"],
            cwd=root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
