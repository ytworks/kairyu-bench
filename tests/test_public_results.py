from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from pathlib import Path

from kairyu_bench.public_results import (
    best_deepswe,
    comparison_table,
    deepswe_table,
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
        self.assertEqual(len(catalog["table_models"]), 37)
        self.assertEqual(
            set(catalog["deepswe_display_mapping"].values()),
            {r["model"] for r in rows},
        )
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
        self.assertNotIn("Gemini 3.1 Pro", catalog["deepswe_display_mapping"])

    def test_main_table_keeps_preview_separate_and_shows_best_effort(self):
        table = comparison_table(load_public_results())
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in table.splitlines()
        ]
        header = rows[0]
        self.assertTrue(all(len(row) == len(header) for row in rows))
        deepswe = next(row for row in rows if row[0] == "DeepSWE v1.1")
        for model, expected in {
            "GPT-6 Astra": "[74.12 (xhigh)][S23]",
            "Grok 4.6": "[67.48 (medium)][S23]",
            "Gemini 3.1 Pro": "—",
            "Gemini 3.1 Pro Preview": "[11.73 (high)][S23]",
            "DeepSeek-V4-Flash-0731": "—",
            "DeepSeek-V4-Flash": "[53.32 (max)][S23]",
            "Kimi K2.7 Code": "[30.53 (unspecified)][S23]",
        }.items():
            self.assertIn(model, header)
            self.assertEqual(deepswe[header.index(model)], expected)
        gpqa = next(row for row in rows if row[0] == "GPQA Diamond")
        self.assertEqual(gpqa[header.index("GPT-6 Astra")], "[96.0][S21]")
        self.assertEqual(gpqa[header.index("Gemini 3.8 Flash")], "—")

    def test_detail_table_preserves_every_effort_and_actual_denominators(self):
        table = deepswe_table(load_public_results())
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in table.splitlines()[2:]
        ]
        self.assertEqual(len(rows), 70)
        self.assertEqual(len({(row[0], row[1]) for row in rows}), 70)
        self.assertIn(
            [
                "`claude-opus-4-8`", "max", "[58.97][S23]", "253/429",
                "111/113", "79.28", "88/111", "57.21–60.74",
            ],
            rows,
        )
        self.assertIn(
            [
                "`gpt-6-astra`", "low", "[67.04][S23]", "303/452",
                "113/113", "79.65", "90/113", "65.73–68.34",
            ],
            rows,
        )

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
