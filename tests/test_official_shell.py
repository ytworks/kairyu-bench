from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kairyu_bench.harness_data import export_selected_rows
from kairyu_bench.inspect_summary import summarize_inspect_log
from kairyu_bench.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


class OfficialShellSupportTest(unittest.TestCase):
    def test_checkout_source_resolves_and_reuses_exact_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            subprocess.run(["git", "init", "-q", str(upstream)], check=True)
            subprocess.run(
                ["git", "-C", str(upstream), "config", "user.email", "test@example.test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(upstream), "config", "user.name", "Test"], check=True
            )
            (upstream / "version.txt").write_text("pinned\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(upstream), "add", "version.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(upstream), "commit", "-q", "-m", "pinned"], check=True
            )
            revision = subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
            ).strip()
            cache = root / "cache"
            command = (
                '. "scripts/lib/official.sh"; '
                f'KAIRYU_BENCH_CACHE_DIR="{cache}"; '
                f'checkout_source demo "{upstream}" "{revision}"'
            )

            first = subprocess.check_output(["sh", "-c", command], cwd=ROOT, text=True).strip()
            second = subprocess.check_output(["sh", "-c", command], cwd=ROOT, text=True).strip()

            self.assertEqual(first, second)
            self.assertEqual((Path(first) / "version.txt").read_text(), "pinned\n")
            actual = subprocess.check_output(
                ["git", "-C", first, "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(actual, revision)

    def test_context_get_reads_only_the_adapter_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context_path = Path(directory) / "context.json"
            context_path.write_text(
                json.dumps({"benchmark": "hle", "dataset": {"revision": "abc"}}),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["KAIRYU_BENCH_CONTEXT"] = str(context_path)

            output = subprocess.check_output(
                ["python3", "-m", "kairyu_bench.adapter_context", "dataset.revision"],
                cwd=ROOT,
                env={**environment, "PYTHONPATH": str(ROOT / "src")},
                text=True,
            )

        self.assertEqual(output.strip(), "abc")

    def test_every_manifest_adapter_has_executable_runner_and_normalizer(self) -> None:
        for name, entry in load_manifest().items():
            with self.subTest(benchmark=name):
                runner = ROOT / entry["adapter"]
                self.assertTrue(runner.is_file(), runner)
                self.assertTrue(os.access(runner, os.X_OK), runner)
                self.assertTrue((runner.parent / "normalize.py").is_file())

    def test_dataset_export_pins_revision_and_writes_canonical_prefix(self) -> None:
        context = {
            "limit": 2,
            "dataset": {"id": "owner/data", "revision": "deadbeef"},
        }
        rows = [
            {"instance_id": "one", "value": 1},
            {"instance_id": "two", "value": 2},
            {"instance_id": "three", "value": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selected.jsonl"
            ids = Path(directory) / "ids.txt"
            with mock.patch("kairyu_bench.harness_data._load_dataset", return_value=rows) as load:
                export_selected_rows(context, "test", "instance_id", output, ids)

            load.assert_called_once_with("owner/data", "deadbeef", "test", None)
            self.assertEqual(ids.read_text().splitlines(), ["one", "two"])
            self.assertEqual(
                [json.loads(line)["instance_id"] for line in output.read_text().splitlines()],
                ["one", "two"],
            )

    def test_inspect_summary_preserves_scicode_official_metrics(self) -> None:
        inspect_log = {
            "status": "success",
            "results": {
                "scores": [
                    {
                        "name": "scicode_scorer",
                        "metrics": {
                            "Problem Correctness": {"value": 0.5},
                            "sub_problem_correctness": {"value": 0.75},
                        },
                    }
                ]
            },
            "samples": [{"id": "13"}, {"id": "62"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scicode.json"
            path.write_text(json.dumps(inspect_log), encoding="utf-8")

            summary = summarize_inspect_log(path)

        self.assertEqual(summary["problem_ids"], ["13", "62"])
        self.assertEqual(summary["metrics"]["Problem Correctness"], 0.5)
        self.assertEqual(summary["metrics"]["sub_problem_correctness"], 0.75)


if __name__ == "__main__":
    unittest.main()
