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
        self.assertNotIn("model", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
