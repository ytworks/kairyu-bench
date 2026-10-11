from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from kairyu_bench.terminal_bench4 import load_lock
from kairyu_bench.terminal_bench4_plugin import (
    KairyuTB4Plugin,
    progress_line,
    verify_job_start,
)

TASK = "terminal-bench/batched-eval-parity"
OTHER = "terminal-bench/lake-temp-glm"


class FakeModel:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return json.loads(json.dumps(self.data))


class FakeJob:
    def __init__(self, config: dict[str, Any], trials: list[dict[str, Any]]) -> None:
        self.config = FakeModel(config)
        self._trial_configs = [FakeModel(trial) for trial in trials]
        self.hooks: dict[str, list[Any]] = {}

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("on_"):
            raise AttributeError(name)

        def register(callback: Any) -> "FakeJob":
            self.hooks.setdefault(name, []).append(callback)
            return self

        return register


def trial_config(task: str, name: str, **agent: Any) -> dict[str, Any]:
    return {
        "task": {"name": task, "ref": load_lock()["tasks"][task]["digest"]},
        "trial_name": name,
        "timeout_multiplier": 1.0,
        "agent": {
            "name": "claude-code",
            "model_name": "kairyu-auto",
            "kwargs": {"version": "2.1.231"},
            **agent,
        },
        "environment": {"type": "docker"},
        "verifier": {},
    }


def selected(tasks: list[str], attempts: int) -> dict[str, Any]:
    return {
        "problem_ids": tasks,
        "attempts": attempts,
        "agent": "claude-code",
        "model_id": "kairyu-auto",
        "model_name": "kairyu-auto",
        "agent_kwargs": {"version": "2.1.231"},
        "job_name": "job",
    }


def job_config(max_retries: int = 3) -> dict[str, Any]:
    return {
        "job_name": "job",
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": max_retries, "exclude_exceptions": ["AgentTimeoutError"]},
        "agents": [{"name": "claude-code", "model_name": "kairyu-auto"}],
        "environment": {"type": "docker"},
    }


def event(
    root: Path, task: str, trial: str, kind: str | None = None, phase: str = "end"
) -> SimpleNamespace:
    info = None if kind is None else SimpleNamespace(exception_type=kind)
    return SimpleNamespace(
        trial_name=trial,
        task_name=task,
        event=SimpleNamespace(value=phase),
        result=SimpleNamespace(exception_info=info),
        config=SimpleNamespace(trials_dir=root / "jobs" / "job"),
    )


class VerifyJobStartTest(unittest.TestCase):
    def test_accepts_the_pinned_plan_and_rejects_drift(self) -> None:
        lock = load_lock()
        trials = [trial_config(TASK, "a__1"), trial_config(TASK, "a__2")]
        verify_job_start(job_config(), trials, selected([TASK], 2), lock)

        drifted = {
            "count": ([trials[0]], job_config()),
            "digest": (
                [trials[0], {**trials[1], "task": {"name": TASK, "ref": "sha256:" + "0" * 64}}],
                job_config(),
            ),
            "model": ([trials[0], trial_config(TASK, "a__2", model_name="x")], job_config()),
            "override": ([trials[0], {**trials[1], "agent_timeout_multiplier": 2.0}], job_config()),
            "job override": (trials, {**job_config(), "timeout_multiplier": 2.0}),
            "task": ([trials[0], trial_config(OTHER, "b__1")], job_config()),
        }
        for name, (configs, config) in drifted.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_job_start(config, configs, selected([TASK], 2), lock)


class PluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.docker_log = self.root / "docker.log"
        docker = self.bin / "docker"
        docker.write_text(
            "#!/bin/sh\n"
            f'echo "$*" >> "{self.docker_log}"\n'
            'case "$*" in *fail-rm*) [ "$2" = rm ] && exit 1;; esac\n'
            "exit 0\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        self.path = mock.patch.dict(os.environ, {"PATH": f"{self.bin}:{os.environ['PATH']}"})
        self.path.start()

    def tearDown(self) -> None:
        self.path.stop()
        self.directory.cleanup()

    def start(self, tasks: list[str], attempts: int, clean: bool = False) -> tuple[KairyuTB4Plugin, FakeJob]:
        (self.root / "selected.json").write_text(json.dumps(selected(tasks, attempts)))
        trials = [
            trial_config(task, f"{task.split('/')[-1]}__{attempt}")
            for attempt in range(attempts)
            for task in tasks
        ]
        job = FakeJob(job_config(), trials)
        plugin = KairyuTB4Plugin(raw_dir=str(self.root))
        plugin.env = {"KAIRYU_BENCH_CLEAN_TASK_IMAGES": "1" if clean else "0"}

        async def run() -> None:
            await plugin.on_job_start(job)
            assert plugin.heartbeat is not None
            plugin.heartbeat.cancel()

        asyncio.run(run())
        return plugin, job

    def test_job_start_writes_the_plan_and_registers_every_hook(self) -> None:
        _, job = self.start([TASK], 5)

        plan = json.loads((self.root / "trial-plan.json").read_text())
        self.assertEqual([slot["attempt_index"] for slot in plan["slots"]], [0, 1, 2, 3, 4])
        self.assertEqual(
            sorted(job.hooks),
            [
                "on_agent_started",
                "on_environment_started",
                "on_trial_cancelled",
                "on_trial_ended",
                "on_trial_started",
                "on_verification_started",
            ],
        )

    def test_retried_attempts_are_preserved_before_harbor_deletes_them(self) -> None:
        plugin, _ = self.start([TASK], 1)
        trial_dir = self.root / "jobs" / "job" / "batched-eval-parity__0"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.log").write_text("first attempt")

        asyncio.run(plugin._ended(event(self.root, TASK, "batched-eval-parity__0", "RuntimeError")))

        self.assertEqual(
            (self.root / "retries" / "batched-eval-parity__0" / "1" / "trial.log").read_text(),
            "first attempt",
        )
        self.assertEqual(plugin.final_counts, {})

    def test_retry_copies_keep_agent_symlinks_as_links(self) -> None:
        plugin, _ = self.start([TASK], 1)
        trial_dir = self.root / "jobs" / "job" / "batched-eval-parity__0"
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "agent" / "latest").symlink_to("/logs/agent/sessions/missing")

        asyncio.run(plugin._ended(event(self.root, TASK, "batched-eval-parity__0", "RuntimeError")))

        copied = self.root / "retries" / "batched-eval-parity__0" / "1" / "agent" / "latest"
        self.assertTrue(copied.is_symlink())
        self.assertEqual(os.readlink(copied), "/logs/agent/sessions/missing")
        self.assertFalse((self.root / "plugin-errors").exists())

    def test_cancelled_trials_are_neither_retried_nor_final(self) -> None:
        plugin, _ = self.start([TASK], 1, clean=True)

        asyncio.run(plugin._ended(event(self.root, TASK, "batched-eval-parity__0", "CancelledError")))

        self.assertFalse((self.root / "retries").exists())
        self.assertEqual(plugin.final_counts, {})
        self.assertNotIn("image rm", self.docker_log.read_text() if self.docker_log.exists() else "")

    def test_images_are_removed_after_the_last_attempt_and_failures_recorded(self) -> None:
        plugin, _ = self.start([TASK], 5, clean=True)
        images = load_lock()["tasks"][TASK]["images"]
        plugin.lock = {
            **plugin.lock,
            "tasks": {
                **plugin.lock["tasks"],
                TASK: {**plugin.lock["tasks"][TASK], "images": [images[0], "fail-rm:latest"]},
            },
        }

        for attempt in range(4):
            asyncio.run(plugin._ended(event(self.root, TASK, f"batched-eval-parity__{attempt}")))
        self.assertNotIn("image rm", self.docker_log.read_text())

        asyncio.run(plugin._ended(event(self.root, TASK, "batched-eval-parity__4", "AgentTimeoutError")))

        log = self.docker_log.read_text()
        self.assertIn("image rm harborframework/terminal-bench@sha256:", log)
        self.assertIn("image rm fail-rm:latest", log)
        errors = list((self.root / "cleanup-errors").glob("*.json"))
        self.assertEqual(len(errors), 1)
        self.assertEqual(json.loads(errors[0].read_text())["image"], "fail-rm:latest")
        progress = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(progress["docker"], "ok")
        self.assertEqual(progress["api"], "unconfigured")

    def test_hooks_record_errors_instead_of_raising_into_harbor(self) -> None:
        plugin, _ = self.start([TASK], 1)

        asyncio.run(plugin._ended(SimpleNamespace(trial_name="x", task_name=TASK)))

        errors = list((self.root / "plugin-errors").glob("*-end.json"))
        self.assertEqual(len(errors), 1)
        self.assertEqual(json.loads(errors[0].read_text())["type"], "AttributeError")

    def test_progress_line_reports_the_monitoring_fields(self) -> None:
        state = {
            "scored_attempts": 3,
            "requested_attempts": 330,
            "passed_attempts": 1,
            "gpu_unavailable_attempts": 1,
            "errors": [{}],
            "evaluated_tasks": 0,
            "problem_ids": ["a"] * 66,
            "active": {"t": "agent-start"},
            "elapsed_seconds": 60,
            "api": "ok",
            "docker": "ok",
            "root_free_bytes": 1,
            "cache_free_bytes": 2,
        }

        self.assertEqual(
            progress_line(state),
            "Terminal-Bench 4.0 progress: scored=3/330 passed=1 gpu_unavailable=1 "
            "unscored=1 tasks=0/66 active=1 elapsed=60s api=ok docker=ok "
            "root_free=1 cache_free=2",
        )


if __name__ == "__main__":
    unittest.main()
