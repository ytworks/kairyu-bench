from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.results import BenchmarkResult, ResultValidationError


def completed_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "20260813T120000Z-a1b2c3d4",
        "benchmark": "gpqa-diamond",
        "status": "completed",
        "endpoint": {"fingerprint": "sha256:0123456789abcdef"},
        "model_id": "chat-capable",
        "source": {
            "repository": "https://github.com/idavidrein/gpqa",
            "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
            "dataset": "Idavidrein/gpqa",
            "dataset_revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        },
        "selection": {"requested_limit": 2, "problem_ids": ["q-1", "q-2"]},
        "counts": {"requested": 2, "evaluated": 2},
        "score": {
            "primary": 50.0,
            "unit": "percent",
            "metrics": {"accuracy": 50.0},
        },
        "scoring": {
            "method": "exact-choice-match",
            "self_judged": False,
            "self_simulated": False,
        },
        "artifacts": {"raw": ["raw/gpqa.jsonl"], "logs": ["logs/gpqa.log"]},
        "timestamps": {
            "started_at": "2026-08-13T12:00:00Z",
            "finished_at": "2026-08-13T12:01:00Z",
        },
        "error": None,
    }


class BenchmarkResultContractTest(unittest.TestCase):
    def test_completed_result_round_trips_to_stable_json(self) -> None:
        result = BenchmarkResult.from_dict(completed_payload())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpqa-diamond.json"
            result.write(path)
            decoded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(decoded, completed_payload())

    def test_completed_result_cannot_hide_missing_evaluations(self) -> None:
        payload = completed_payload()
        payload["counts"] = {"requested": 2, "evaluated": 1}

        with self.assertRaisesRegex(ResultValidationError, "all selected problems"):
            BenchmarkResult.from_dict(payload)

    def test_completed_result_requires_a_primary_score(self) -> None:
        payload = completed_payload()
        payload["score"] = {"primary": None, "unit": "percent", "metrics": {}}

        with self.assertRaisesRegex(ResultValidationError, "primary score"):
            BenchmarkResult.from_dict(payload)

    def test_percent_score_must_be_in_closed_zero_to_one_hundred_range(self) -> None:
        payload = completed_payload()
        payload["score"] = {
            "primary": 101.0,
            "unit": "percent",
            "metrics": {"accuracy": 101.0},
        }

        with self.assertRaisesRegex(ResultValidationError, "between 0 and 100"):
            BenchmarkResult.from_dict(payload)

    def test_partial_result_preserves_score_and_error_without_claiming_completion(self) -> None:
        payload = completed_payload()
        payload["status"] = "partial"
        payload["counts"] = {"requested": 2, "evaluated": 1}
        payload["score"] = {
            "primary": 100.0,
            "unit": "percent",
            "metrics": {"accuracy": 100.0},
        }
        payload["error"] = "second problem timed out"

        result = BenchmarkResult.from_dict(payload)

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.data["counts"]["evaluated"], 1)
        self.assertEqual(result.data["error"], "second problem timed out")

    def test_embedding_condition_requires_a_non_empty_model_id(self) -> None:
        payload = completed_payload()
        payload["conditions"] = {"embedding_model_id": ""}

        with self.assertRaisesRegex(
            ResultValidationError,
            "conditions.embedding_model_id",
        ):
            BenchmarkResult.from_dict(payload)

    def test_conditions_must_be_an_object(self) -> None:
        payload = completed_payload()
        payload["conditions"] = ["embed-small"]

        with self.assertRaisesRegex(ResultValidationError, "conditions must be an object"):
            BenchmarkResult.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
