"""Harbor job plugin for Terminal-Bench 4.0: plan, retry evidence, cleanup, progress.

Loaded by the pinned `harbor run --plugin`. Harbor is never imported here; the
plugin only reads the job and hook-event objects it receives. Trial hooks run
inside Harbor's TaskGroup, so every hook catches its own errors and records
them as plugin errors (which block a completed run) instead of raising.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from kairyu_bench.artifacts import read_object, write_json
from kairyu_bench.terminal_bench4 import (
    LABEL,
    build_trial_plan,
    is_final_attempt,
    job_config_failures,
    load_lock,
    removable_images,
    removal_reference,
    trial_config_failures,
)
from kairyu_bench.terminal_bench4_accounting import summarize

HEARTBEAT_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 15


def verify_job_start(
    job_config: Mapping[str, Any],
    trial_configs: list[Mapping[str, Any]],
    selected: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    """Fail closed before any trial starts if Harbor resolved anything unpinned."""
    failures = job_config_failures(job_config)
    if failures:
        raise ValueError(f"{LABEL} job config breaks leaderboard rules: {failures[0]}")
    tasks = selected["problem_ids"]
    if len(trial_configs) != len(tasks) * selected["attempts"]:
        raise ValueError(f"{LABEL} Harbor planned an unexpected number of trials")
    for config in trial_configs:
        task = (config.get("task") or {}).get("name")
        agent = config.get("agent") or {}
        if task not in tasks:
            raise ValueError(f"{LABEL} Harbor planned an unselected task: {task}")
        if config["task"].get("ref") != lock["tasks"][task]["digest"]:
            raise ValueError(f"{LABEL} Harbor resolved an unpinned digest for {task}")
        if agent.get("name") != selected["agent"] or agent.get("model_name") != selected["model_name"]:
            raise ValueError(f"{LABEL} Harbor planned another agent or model")
        failures = trial_config_failures(config)
        if failures:
            raise ValueError(f"{LABEL} trial config breaks leaderboard rules: {failures[0]}")
    if {config["task"]["name"] for config in trial_configs} != set(tasks):
        raise ValueError(f"{LABEL} Harbor did not plan every selected task")


def progress_line(state: Mapping[str, Any]) -> str:
    return (
        f"{LABEL} progress: scored={state['scored_attempts']}/{state['requested_attempts']} "
        f"passed={state['passed_attempts']} gpu_unavailable={state['gpu_unavailable_attempts']} "
        f"unscored={len(state['errors'])} tasks={state['evaluated_tasks']}/{len(state['problem_ids'])} "
        f"active={len(state['active'])} elapsed={state['elapsed_seconds']}s "
        f"api={state['api']} docker={state['docker']} "
        f"root_free={state['root_free_bytes']} cache_free={state['cache_free_bytes']}"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def _api_health(env: Mapping[str, str]) -> str:
    base = env.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        return "unconfigured"
    request = Request(base + "/models")
    key = env.get("OPENAI_API_KEY", "")
    if key and key != "not-required":
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urlopen(request, timeout=PROBE_TIMEOUT_SECONDS) as response:
            return "ok" if response.status == 200 else f"http-{response.status}"
    except Exception as error:  # noqa: BLE001 - reported, never raised into Harbor
        return type(error).__name__


async def _docker(*args: str) -> int:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        return await asyncio.wait_for(process.wait(), timeout=PROBE_TIMEOUT_SECONDS * 4)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 124


class KairyuTB4Plugin:
    def __init__(self, raw_dir: str, **_: Any) -> None:
        self.raw_dir = Path(raw_dir)
        self.env: Mapping[str, str] = os.environ
        self.lock: dict[str, Any] = {}
        self.selected: dict[str, Any] = {}
        self.retry: dict[str, Any] = {}
        self.active: dict[str, str] = {}
        self.ends: dict[str, int] = {}
        self.final_counts: dict[str, int] = {}
        self.finished_tasks: set[str] = set()
        self.removed_images: set[str] = set()
        self.plugin_errors = 0
        self.started = time.monotonic()
        self.heartbeat: asyncio.Task[None] | None = None

    async def on_job_start(self, job: Any) -> None:
        self.lock = load_lock()
        self.selected = read_object(self.raw_dir / "selected.json", LABEL)
        job_config = job.config.model_dump(mode="json")
        trial_configs = [config.model_dump(mode="json") for config in job._trial_configs]
        verify_job_start(job_config, trial_configs, self.selected, self.lock)
        self.retry = dict(job_config["retry"])
        write_json(
            self.raw_dir / "trial-plan.json",
            build_trial_plan(trial_configs, job_config["job_name"]),
        )
        job.on_trial_started(self._phase)
        job.on_environment_started(self._phase)
        job.on_agent_started(self._phase)
        job.on_verification_started(self._phase)
        job.on_trial_ended(self._ended)
        job.on_trial_cancelled(self._cancelled)
        self.started = time.monotonic()
        self.heartbeat = asyncio.create_task(self._beat())
        print(f"{LABEL} plan: trials={len(trial_configs)} retry={self.retry.get('max_retries')}", flush=True)

    async def on_job_end(self, job_result: Any) -> None:
        if self.heartbeat is not None:
            self.heartbeat.cancel()
            await asyncio.gather(self.heartbeat, return_exceptions=True)
        await self.snapshot()

    async def _phase(self, event: Any) -> None:
        try:
            self.active[event.trial_name] = event.event.value
            print(f"{LABEL} {event.event.value}: task={event.task_name} trial={event.trial_name}", flush=True)
        except Exception as error:  # noqa: BLE001
            self._record_error("phase", error)

    async def _cancelled(self, event: Any) -> None:
        try:
            self.active.pop(event.trial_name, None)
            print(f"{LABEL} cancelled: task={event.task_name} trial={event.trial_name}", flush=True)
        except Exception as error:  # noqa: BLE001
            self._record_error("cancel", error)

    async def _ended(self, event: Any) -> None:
        try:
            await self._handle_end(event)
        except Exception as error:  # noqa: BLE001
            self._record_error("end", error)

    async def _handle_end(self, event: Any) -> None:
        trial, task = event.trial_name, event.task_name
        self.ends[trial] = self.ends.get(trial, 0) + 1
        info = event.result.exception_info
        kind = info.exception_type if info is not None else None
        self.active.pop(trial, None)
        if kind == "CancelledError":
            # A stopped job: Harbor keeps the directory and never retries it.
            print(f"{LABEL} interrupted: task={task} trial={trial}", flush=True)
            return
        if not is_final_attempt(kind, self.ends[trial], self.retry):
            # Harbor deletes this directory before retrying; keep the evidence.
            # Agent-written logs may hold absolute symlinks; copy them as links.
            source = Path(event.config.trials_dir) / trial
            destination = self.raw_dir / "retries" / trial / str(self.ends[trial])
            await asyncio.to_thread(
                shutil.copytree, source, destination, symlinks=True, dirs_exist_ok=True
            )
            print(f"{LABEL} retry: task={task} trial={trial} type={kind} attempt={self.ends[trial]}", flush=True)
            return
        self.final_counts[task] = self.final_counts.get(task, 0) + 1
        print(f"{LABEL} trial finished: task={task} trial={trial} error={kind}", flush=True)
        if self.final_counts[task] == self.selected["attempts"]:
            self.finished_tasks.add(task)
            await self.cleanup_images()
        await self.snapshot()

    async def cleanup_images(self) -> None:
        if self.env.get("KAIRYU_BENCH_CLEAN_TASK_IMAGES") != "1":
            return
        selected = self.selected["problem_ids"]
        for image in removable_images(self.lock, selected, self.finished_tasks):
            if image in self.removed_images:
                continue
            self.removed_images.add(image)
            reference = removal_reference(image)
            if await _docker("image", "inspect", reference) != 0:
                continue
            if await _docker("image", "rm", reference) != 0:
                write_json(
                    self.raw_dir / "cleanup-errors" / f"{_safe_name(reference)}.json",
                    {"type": "TaskImageRemovalError", "image": reference},
                )
                print(f"{LABEL} cleanup failed: {reference}", flush=True)

    async def snapshot(self) -> None:
        try:
            summary = summarize(self.raw_dir)
            api = await asyncio.to_thread(_api_health, self.env)
            docker = "ok" if await _docker("info", "--format", "{{.ServerVersion}}") == 0 else "error"
            state = {
                **summary,
                "active": dict(self.active),
                "elapsed_seconds": round(time.monotonic() - self.started),
                "api": api,
                "docker": docker,
                "root_free_bytes": shutil.disk_usage("/").free,
                "cache_free_bytes": shutil.disk_usage(
                    self.env.get("KAIRYU_BENCH_CACHE_DIR", str(self.raw_dir))
                ).free,
            }
            write_json(self.raw_dir / "progress.json", state)
            write_json(self.raw_dir / "summary.json", summary)
            print(progress_line(state), flush=True)
        except Exception as error:  # noqa: BLE001
            # Harbor writes result.json non-atomically, so a snapshot may race a
            # trial end. Finalization re-reads every result and fails closed.
            print(f"{LABEL} progress unavailable: {type(error).__name__}: {error}", flush=True)

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            await self.snapshot()

    def _record_error(self, stage: str, error: BaseException) -> None:
        self.plugin_errors += 1
        try:
            write_json(
                self.raw_dir / "plugin-errors" / f"{self.plugin_errors:04d}-{stage}.json",
                {
                    "stage": stage,
                    "type": type(error).__name__,
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                },
            )
        finally:
            print(f"{LABEL} plugin error: stage={stage} type={type(error).__name__}", flush=True)
