from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.reporting import compare_runs, write_score_report
from kairyu_bench.results import BenchmarkResult


def _result(
    run_id: str,
    benchmark: str,
    score: float,
    *,
    problem_ids: list[str] | None = None,
    source_revision: str = "source-pin",
) -> BenchmarkResult:
    ids = problem_ids or ["one", "two"]
    return BenchmarkResult.from_dict(
        {
            "schema_version": 1,
            "run_id": run_id,
            "benchmark": benchmark,
            "status": "completed",
            "endpoint": {"fingerprint": "sha256:0123456789abcdef"},
            "model_id": f"model-{run_id}",
            "source": {
                "repository": "https://example.test/official",
                "revision": source_revision,
                "dataset": "owner/data",
                "dataset_revision": "dataset-pin",
            },
            "selection": {"requested_limit": 2, "problem_ids": ids},
            "counts": {"requested": len(ids), "evaluated": len(ids)},
            "score": {"primary": score, "unit": "percent", "metrics": {}},
            "scoring": {
                "method": "official-method",
                "self_judged": False,
                "self_simulated": False,
            },
            "artifacts": {"raw": ["raw/output.json"], "logs": ["logs/run.log"]},
            "timestamps": {
                "started_at": "2026-08-13T00:00:00Z",
                "finished_at": "2026-08-13T00:01:00Z",
            },
            "error": None,
        }
    )


def _run(root: Path, run_id: str, results: list[BenchmarkResult]) -> Path:
    run = root / run_id
    normalized = run / "normalized"
    normalized.mkdir(parents=True)
    for result in results:
        result.write(normalized / f"{result.benchmark}.json")
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "endpoint": {"fingerprint": "sha256:0123456789abcdef"},
                "model_id": f"model-{run_id}",
                "benchmarks": [result.benchmark for result in results],
                "limit": 2,
                "started_at": "2026-08-13T00:00:00Z",
                "finished_at": "2026-08-13T00:01:00Z",
                "statuses": {result.benchmark: result.status for result in results},
            }
        ),
        encoding="utf-8",
    )
    return run


class ReportingTest(unittest.TestCase):
    def test_score_report_writes_json_and_readable_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _run(
                Path(directory),
                "run-a",
                [_result("run-a", "gpqa-diamond", 75.0), _result("run-a", "hle", 25.0)],
            )

            report = write_score_report(run)

            decoded = json.loads((run / "report.json").read_text())
            markdown = (run / "report.md").read_text()
        self.assertEqual(report["macro_average_percent"], 50.0)
        self.assertEqual(decoded["counts"], {"completed": 2, "selected": 2})
        self.assertIn("| gpqa-diamond | completed | 75.00 |", markdown)
        self.assertIn("Macro average: 50.00%", markdown)

    def test_compare_only_calculates_delta_for_exactly_compatible_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(
                root,
                "run-a",
                [
                    _result("run-a", "gpqa-diamond", 50.0),
                    _result("run-a", "hle", 25.0),
                ],
            )
            candidate = _run(
                root,
                "run-b",
                [
                    _result("run-b", "gpqa-diamond", 75.0),
                    _result("run-b", "hle", 30.0, problem_ids=["different"]),
                ],
            )

            comparison = compare_runs(baseline, candidate)

        rows = {row["benchmark"]: row for row in comparison["rows"]}
        self.assertTrue(rows["gpqa-diamond"]["compatible"])
        self.assertEqual(rows["gpqa-diamond"]["delta"], 25.0)
        self.assertFalse(rows["hle"]["compatible"])
        self.assertIsNone(rows["hle"]["delta"])
        self.assertIn("problem IDs differ", rows["hle"]["reason"])

    def test_compare_rejects_source_revision_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _run(root, "run-a", [_result("run-a", "mrcr-v2", 40.0)])
            candidate = _run(
                root,
                "run-b",
                [_result("run-b", "mrcr-v2", 60.0, source_revision="new-source")],
            )

            comparison = compare_runs(baseline, candidate)

        row = comparison["rows"][0]
        self.assertFalse(row["compatible"])
        self.assertIsNone(row["delta"])
        self.assertIn("source lock differs", row["reason"])


if __name__ == "__main__":
    unittest.main()
