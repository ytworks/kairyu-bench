from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest import mock

from kairyu_bench.manifest import load_manifest
from scripts.shims.livecodebench_pro import (
    configured_retries,
    configured_workers,
    run_problem_pool,
)


ROOT = Path(__file__).resolve().parents[1]


class LiveCodeBenchProHarnessTest(unittest.TestCase):
    def test_worker_count_defaults_to_one_and_accepts_four(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_workers(), 1)
        with mock.patch.dict(
            os.environ,
            {"KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS": "4"},
            clear=True,
        ):
            self.assertEqual(configured_workers(), 4)

    def test_worker_count_rejects_invalid_values(self) -> None:
        for value in ("0", "17", "four"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS": value},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "integer from 1 to 16"):
                    configured_workers()

    def test_generation_retries_default_to_three(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_retries(), 3)

    def test_generation_retries_reject_invalid_values(self) -> None:
        for value in ("-1", "11", "many"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES": value},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "integer from 0 to 10"):
                    configured_retries()

    def test_problem_pool_refills_a_free_slot(self) -> None:
        events: list[str] = []

        def solve_problem(index: int, name: str) -> tuple[int, str]:
            events.append(f"start {name}")
            time.sleep(0.2 if name == "slow" else 0.02)
            events.append(f"end {name}")
            return index, name

        completed = list(
            run_problem_pool(["fast", "slow", "refill"], 2, solve_problem)
        )

        self.assertEqual(
            {problem for _, problem in completed},
            {"fast", "slow", "refill"},
        )
        self.assertLess(events.index("start refill"), events.index("end slow"))

    def test_verifier_image_pins_node_compatible_npm(self) -> None:
        entry = load_manifest()["livecodebench-pro"]
        verifier_revision = next(
            source["revision"]
            for source in entry["secondary_sources"]
            if source["id"].endswith("/LightCPVerifier")
        )
        dockerfile = (
            ROOT / "scripts/harnesses/livecodebench-pro.Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn(verifier_revision, dockerfile)
        self.assertIn("https://deb.nodesource.com/setup_20.x", dockerfile)
        self.assertIn("RUN npm install -g npm@10.8.2", dockerfile)
        self.assertNotIn("RUN npm install -g npm@latest", dockerfile)

    def test_harness_uses_the_revision_scoped_verifier_image(self) -> None:
        harness = (ROOT / "scripts/harnesses/livecodebench-pro.sh").read_text(
            encoding="utf-8"
        )
        shim = (ROOT / "scripts/shims/livecodebench_pro.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'verifier_image="lightcpverifier:${verifier_revision}-kairyu"',
            harness,
        )
        self.assertIn('export KAIRYU_LIGHTCPVERIFIER_IMAGE="$verifier_image"', harness)
        self.assertIn(
            'KAIRYU_LIGHTCPVERIFIER_HOST:-container', harness
        )
        self.assertIn('os.environ.get("KAIRYU_LIGHTCPVERIFIER_IMAGE")', shim)
        self.assertIn("LightCPVerifierJudge.IMAGE_NAME = verifier_image", shim)
        self.assertIn("ThreadPoolExecutor(max_workers=workers)", shim)
        self.assertIn("timeout=1200", shim)
        self.assertIn("Retrying solution", shim)
        self.assertIn('problem.judge_result = "Judge Failed"', shim)
        self.assertIn("write_results(output, completed_problems, BenchmarkResult)", shim)
        self.assertIn("class RoutedLightCPVerifierJudge", shim)
        self.assertIn('verifier_host == "container"', shim)
        self.assertIn("verifier_container_base_url(self.container)", shim)
        self.assertIn('self.base_url = f"http://{verifier_host}:{port}"', shim)


if __name__ == "__main__":
    unittest.main()
