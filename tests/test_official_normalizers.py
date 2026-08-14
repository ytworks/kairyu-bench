from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.official import normalize_official, unsupported_result


def _context(name: str, *, limit: int | None = 2) -> dict[str, object]:
    scoring = {
        "swe-bench-verified": "swebench-official-resolved-percent",
        "swe-bench-pro": "swebench-official-resolved-percent",
        "terminal-bench": "harbor-official-task-reward",
        "livecodebench": "livecodebench-official-pass-at-1",
        "livecodebench-pro": "livecodebench-pro-lightcpverifier-pass-at-1",
        "hle": "hle-official-llm-judge-self",
        "charxiv-reasoning": "charxiv-official-reasoning-judge-self",
        "scicode": "scicode-official-inspect-tests",
        "tau-bench-banking": "tau2-official-average-reward-self-simulated",
    }[name]
    context = {
        "schema_version": 1,
        "run_id": "run-1",
        "benchmark": name,
        "endpoint_fingerprint": "sha256:0123456789abcdef",
        "model_id": "chat-capable",
        "limit": limit,
        "source": {
            "repository": "https://example.test/upstream",
            "revision": "source-pin",
        },
        "dataset": {"id": "owner/data", "revision": "dataset-pin"},
        "scoring": {
            "method": scoring,
            "unit": "percent",
            "self_judged": name in {"hle", "charxiv-reasoning"},
            "self_simulated": name == "tau-bench-banking",
        },
        "run_dir": "/work/results/run-1",
        "result_path": f"/work/results/run-1/normalized/{name}.json",
    }
    if name == "tau-bench-banking":
        context["conditions"] = {"embedding_model_id": "embed-small"}
    return context


class OfficialNormalizerTest(unittest.TestCase):
    def _write_json(self, root: Path, name: str, value: object) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_swebench_uses_official_resolved_count_and_selected_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "report.json",
                {
                    "total_instances": 2,
                    "submitted_instances": 2,
                    "completed_instances": 2,
                    "resolved_instances": 1,
                    "submitted_ids": ["django__django-1", "django__django-2"],
                    "resolved_ids": ["django__django-1"],
                },
            )

            result = normalize_official(_context("swe-bench-verified"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)
        self.assertEqual(
            result.data["selection"]["problem_ids"],
            ["django__django-1", "django__django-2"],
        )

    def test_harbor_reads_trial_rewards_without_counting_job_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_json(root, "result.json", {"stats": {"n_trials": 2}})
            self._write_json(
                root,
                "task-a/result.json",
                {"task_name": "task-a", "verifier_result": {"rewards": {"reward": 1}}},
            )
            self._write_json(
                root,
                "task-b/result.json",
                {"task_name": "task-b", "verifier_result": {"rewards": {"reward": 0}}},
            )

            result = normalize_official(_context("terminal-bench"), root)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)
        self.assertEqual(result.data["counts"], {"requested": 2, "evaluated": 2})

    def test_hle_takes_official_metrics_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "hle.jsonl"
            lines = [
                {"id": "hle-1", "is_correct": True},
                {"id": "hle-2", "is_correct": False},
                {
                    "_type": "metrics_summary",
                    "metrics": {
                        "accuracy": 50.0,
                        "accuracy_success_only": 50.0,
                        "calibration_error": 0.2,
                        "evaluated_questions": 2,
                        "failed_questions": 0,
                        "total_questions": 2,
                    },
                },
            ]
            raw.write_text("".join(json.dumps(row) + "\n" for row in lines))

            result = normalize_official(_context("hle"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)
        self.assertTrue(result.data["scoring"]["self_judged"])

    def test_livecodebench_reads_pass_at_one_and_problem_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "summary.json",
                {
                    "metrics": [{"pass@1": 0.25}],
                    "records": [
                        {"question_id": "lcb-1"},
                        {"question_id": "lcb-2"},
                    ],
                },
            )

            result = normalize_official(_context("livecodebench"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 25.0)

    def test_livecodebench_pro_scores_only_official_accepted_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "benchmark_result.json",
                [
                    {"problem_id": "2000A", "judge_result": "Accepted"},
                    {"problem_id": "2000B", "judge_result": "Wrong Answer"},
                ],
            )

            result = normalize_official(_context("livecodebench-pro"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)

    def test_charxiv_uses_official_overall_score_and_valid_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "reasoning_summary.json",
                {
                    "scores": {"1001": 4, "1002": 2},
                    "stats": {"Overall Score": 60.0, "N_valid": 2, "N_invalid": 0},
                },
            )

            result = normalize_official(_context("charxiv-reasoning"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 60.0)

    def test_scicode_uses_inspect_official_main_and_subproblem_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "inspect-summary.json",
                {
                    "problem_ids": ["13", "62"],
                    "metrics": {
                        "Problem Correctness": 0.5,
                        "sub_problem_correctness": 0.75,
                    },
                },
            )

            result = normalize_official(_context("scicode"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)
        self.assertEqual(result.data["score"]["metrics"]["subproblem_percent"], 75.0)

    def test_tau_uses_reward_info_and_marks_self_simulated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "results.json",
                {
                    "simulations": [
                        {"task_id": "bank-1", "reward_info": {"reward": 1.0}},
                        {"task_id": "bank-2", "reward_info": {"reward": 0.0}},
                    ]
                },
            )

            result = normalize_official(_context("tau-bench-banking"), raw)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 50.0)
        self.assertTrue(result.data["scoring"]["self_simulated"])
        self.assertEqual(
            result.data["conditions"],
            {"embedding_model_id": "embed-small"},
        )

    def test_missing_official_rows_are_partial_not_a_fabricated_complete_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_json(
                Path(directory),
                "benchmark_result.json",
                [
                    {"problem_id": "2000A", "judge_result": "Accepted"},
                    {"problem_id": "2000B", "judge_result": "Judging"},
                ],
            )

            result = normalize_official(_context("livecodebench-pro"), raw)

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["counts"], {"requested": 2, "evaluated": 1})

    def test_unsupported_result_never_contains_a_score(self) -> None:
        result = unsupported_result(
            _context("livecodebench-pro"), "official gated testcases are unavailable"
        )

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.data["score"]["primary"])
        self.assertIn("gated", result.data["error"])


if __name__ == "__main__":
    unittest.main()
