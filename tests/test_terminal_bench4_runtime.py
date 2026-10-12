"""Checks against the pinned Harbor runtime (installed only in its CI job)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from typing import Any

from kairyu_bench.manifest import load_manifest
from kairyu_bench.terminal_bench4 import (
    job_config_failures,
    load_lock,
    resolve_terminal_bench4_conditions,
)
from kairyu_bench.terminal_bench4_accounting import Outcome, classify_trial
from kairyu_bench.terminal_bench4_cli import prepare

try:
    metadata.version("harbor")
    HAS_HARBOR = True
except metadata.PackageNotFoundError:
    HAS_HARBOR = False

NETWORK = os.environ.get("KAIRYU_BENCH_NETWORK_TESTS") == "1"
FIXTURE = Path(__file__).parent / "fixtures/terminal-bench-4/gpu-request"
INFO = {"ServerVersion": "test", "NCPU": 32, "MemTotal": 1, "Runtimes": {"runc": {}}}


def _docker_without_gpu_runtime() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return info.returncode == 0 and "nvidia" not in info.stdout


def _prepared(raw: Path, limit: int, attempts: str) -> dict[str, Any]:
    entry = load_manifest()["terminal-bench-4"]
    context = {
        "run_id": "runtime",
        "benchmark": "terminal-bench-4",
        "endpoint": "https://example.test/v1",
        "model_id": "kairyu-auto",
        "agent": "claude-code",
        "limit": limit,
        "source": entry["source"],
        "dataset": entry["dataset"],
        "conditions": resolve_terminal_bench4_conditions(
            entry, {"KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS": attempts}, "claude-code"
        ),
    }
    prepare(context, raw, INFO, lambda: True, metadata.version("harbor"))
    return json.loads((raw / "job-config.json").read_text())


@unittest.skipUnless(HAS_HARBOR, "the pinned Harbor runtime is not installed")
class PinnedHarborTest(unittest.TestCase):
    def test_installed_harbor_is_the_locked_release(self) -> None:
        self.assertEqual(metadata.version("harbor"), load_lock()["harbor_version"])

    def test_harbor_cli_accepts_the_generated_job_config_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            expected = _prepared(raw, 2, "5")
            completed = subprocess.run(
                [
                    "harbor", "run", "--config", str(raw / "job-config.json"),
                    "--plugin", "kairyu_bench.terminal_bench4_plugin:KairyuTB4Plugin",
                    "--plugin-kwarg", f"raw_dir={raw}", "--yes", "--quiet", "--print-config",
                ],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "HARBOR_TELEMETRY": "0"},
            )
        resolved = json.loads(completed.stdout)
        self.assertEqual(job_config_failures(resolved), [])
        self.assertEqual(resolved["datasets"], expected["datasets"])
        self.assertEqual(resolved["agents"][0]["kwargs"], {"version": "2.1.231"})
        self.assertEqual(resolved["n_attempts"], 5)

    @unittest.skipUnless(NETWORK, "set KAIRYU_BENCH_NETWORK_TESTS=1 to query the registry")
    def test_registry_digests_match_the_lock(self) -> None:
        from harbor.registry.client.package import PackageDatasetClient

        lock = load_lock()

        async def fetch() -> dict[str, str]:
            meta = await PackageDatasetClient().get_dataset_metadata(
                f"{lock['dataset']}@{lock['dataset_ref']}"
            )
            return {f"{task.org}/{task.name}": task.ref for task in meta.task_ids}

        digests = asyncio.run(fetch())
        self.assertEqual(digests, {task: value["digest"] for task, value in lock["tasks"].items()})

    @unittest.skipUnless(NETWORK, "set KAIRYU_BENCH_NETWORK_TESTS=1 to download task metadata")
    def test_plugin_attaches_to_a_real_job_and_writes_the_plan(self) -> None:
        from harbor.cli.job_plugins import attach_job_plugin
        from harbor.job import Job
        from harbor.models.job.config import JobConfig

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            config = JobConfig.model_validate(_prepared(raw, 2, "5"))

            async def attach() -> None:
                job = await Job.create(config)
                plugin = await attach_job_plugin(
                    job,
                    "kairyu_bench.terminal_bench4_plugin:KairyuTB4Plugin",
                    kwargs={"raw_dir": str(raw)},
                )
                plugin.heartbeat.cancel()

            asyncio.run(attach())
            plan = json.loads((raw / "trial-plan.json").read_text())
        self.assertEqual(len(plan["slots"]), 10)
        self.assertEqual(
            sorted({slot["attempt_index"] for slot in plan["slots"]}), [0, 1, 2, 3, 4]
        )

    @unittest.skipUnless(_docker_without_gpu_runtime(), "needs Docker without an NVIDIA runtime")
    def test_gpu_request_without_a_runtime_is_classified_gpu_unavailable(self) -> None:
        from harbor.job import Job
        from harbor.models.job.config import JobConfig

        with tempfile.TemporaryDirectory() as directory:
            config = JobConfig.model_validate(
                {
                    "job_name": "gpu",
                    "jobs_dir": directory,
                    "quiet": True,
                    "agents": [{"name": "oracle"}],
                    "environment": {"type": "docker", "delete": True},
                    "tasks": [{"path": str(FIXTURE)}],
                }
            )

            async def run() -> None:
                job = await Job.create(config)
                await job.run()

            asyncio.run(run())
            results = list(Path(directory).glob("gpu/*/result.json"))
            self.assertEqual(len(results), 1)
            payload = json.loads(results[0].read_text())
        self.assertEqual(
            classify_trial(payload, gpu_task=True, gpu_available=False),
            Outcome(0, "gpu-unavailable"),
            payload.get("exception_info"),
        )


if __name__ == "__main__":
    unittest.main()
