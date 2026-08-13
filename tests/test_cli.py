from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliContractTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "kairyu_bench", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_list_prints_the_twelve_public_adapter_names(self) -> None:
        result = self.run_cli("list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "swe-bench-pro",
                "swe-bench-verified",
                "terminal-bench",
                "livecodebench",
                "livecodebench-pro",
                "hle",
                "charxiv-reasoning",
                "gpqa-diamond",
                "scicode",
                "tau-bench-banking",
                "long-context-reasoning",
                "mrcr-v2",
            ],
        )

    def test_run_requires_only_the_endpoint_positional_argument(self) -> None:
        result = self.run_cli("run", "https://example.test/v1", "--dry-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("endpoint: https://example.test/v1", result.stdout)
        self.assertIn("benchmarks: 12", result.stdout)
        self.assertNotIn("model", result.stderr.lower())

    def test_dry_run_validates_only_and_positive_limit_without_api_access(self) -> None:
        result = self.run_cli(
            "run",
            "https://example.test",
            "--only",
            "mrcr-v2,gpqa-diamond",
            "--limit",
            "3",
            "--dry-run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("selected: gpqa-diamond,mrcr-v2", result.stdout)
        self.assertIn("limit: 3", result.stdout)

        invalid = self.run_cli(
            "run", "https://example.test", "--limit", "0", "--dry-run"
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("positive integer", invalid.stderr)

    def test_unknown_benchmark_fails_before_contacting_endpoint(self) -> None:
        result = self.run_cli(
            "run",
            "https://example.test",
            "--only",
            "not-a-benchmark",
            "--dry-run",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown benchmark", result.stderr)


if __name__ == "__main__":
    unittest.main()
