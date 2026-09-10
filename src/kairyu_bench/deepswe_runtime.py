"""Thin integration with the pinned Pier API (installed only for DeepSWE)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.environments.docker.docker import DockerEnvironment
from pier.job import Job
from pier.models.agent.network import NetworkAllowlist
from pier.models.job.config import JobConfig
from pier.models.task.task import Task

from kairyu_bench.deepswe import (
    read_object,
    select_deepswe_tasks,
    sha256_file,
    summarize_deepswe,
    write_json,
    write_trial_plan,
)

TRANSIENT_AGENT_ERRORS = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
}


class TransientAgentError(RuntimeError):
    """An exhausted mini agent API retry, eligible for a bounded Pier retry."""


def _model_id(value: str) -> str:
    # Upstream interpolates --model into a shell command without quoting it.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:-]*", value):
        raise ValueError("DeepSWE model ID contains unsupported shell characters")
    return value


class KairyuMiniSweAgent(MiniSweAgent):
    # Use the packaged, pinned cost map rather than refreshing from main.
    def install_spec(self):
        spec = super().install_spec()
        for step in spec.steps:
            if 'url = "' + self._LITELLM_MODEL_COST_MAP_URL + '"' in step.run:
                start = step.run.index("\"$python_bin\" <<'PY'")
                end = step.run.index("\nPY\n", start) + len("\nPY\n")
                step.run = step.run[:start] + step.run[end:]
        return spec

    def network_allowlist(self) -> NetworkAllowlist:
        endpoint = (
            self._get_env("OPENAI_BASE_URL") or self._get_env("OPENAI_API_BASE") or ""
        )
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "DeepSWE requires an HTTP(S) API endpoint without embedded credentials"
            )
        return NetworkAllowlist(domains=[parsed.hostname])

    def populate_context_post_run(self, context):
        super().populate_context_post_run(context)
        path = self.logs_dir / "mini-swe-agent.trajectory.json"
        status = "MissingAgentTrajectory"
        if path.is_file():
            status = (
                read_object(path).get("info", {}).get("exit_status")
                or "MissingAgentExitStatus"
            )
        context.metadata = {**(context.metadata or {}), "mini_exit_status": status}
        context.cost_usd = None  # No valid pricing is available for a local API.

    async def run(self, instruction, environment, context):
        await super().run(instruction, environment, context)
        self.populate_context_post_run(context)
        status = context.metadata["mini_exit_status"]
        if status in TRANSIENT_AGENT_ERRORS:
            raise TransientAgentError(status)


def add_host_gateway(compose: dict[str, Any], address: str) -> dict[str, Any]:
    ipaddress.IPv4Address(address)
    result = copy.deepcopy(compose)
    result["services"]["pier-egress-proxy"]["extra_hosts"] = [
        f"host.docker.internal:{address}"
    ]
    return result


def restrict_proxy_port(script: str, endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for name, original in (("Safe_ports", "80 443"), ("SSL_ports", "443")):
        old = f"acl {name} port {original}"
        if script.count(old) != 1:
            raise ValueError("Pinned Pier proxy template changed")
        script = script.replace(old, f"acl {name} port {port}")
    return script


class KairyuDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, gateway_address=None, **kwargs):
        self.gateway_address = gateway_address
        self._stopping = False
        super().__init__(*args, **kwargs)
        write_json(
            self.trial_paths.trial_dir / f"managed-images-{self.session_id}.json",
            {
                "names": [
                    self._env_vars.main_image_name,
                    self.task_env_config.docker_image,
                ],
            },
        )

    def _prepare_egress_proxy_compose(self):
        super()._prepare_egress_proxy_compose()
        if self._egress_proxy_compose_path:
            path = self._egress_proxy_compose_path
            if self.gateway_address:
                write_json(
                    path, add_host_gateway(read_object(path), self.gateway_address)
                )
            script = self.trial_paths.trial_dir / "egress-proxy/start-squid.sh"
            script.write_text(
                restrict_proxy_port(script.read_text(), os.environ["OPENAI_BASE_URL"])
            )

    async def _run_docker_compose_command(self, command, check=True, timeout_sec=None):
        # Separate verification retains the agent image, but its temporary proxy
        # has no consumers after agent shutdown. Remove that untagged Compose image.
        if self._stopping and command == ["down"]:
            command = ["down", "--rmi", "local"]
        try:
            return await super()._run_docker_compose_command(
                command, check, timeout_sec
            )
        except Exception as error:
            if command[0] == "down":
                write_json(
                    self.trial_paths.trial_dir / "cleanup-error.json",
                    {"type": type(error).__name__},
                )
                print(f"DeepSWE cleanup failed: {self.environment_name}", flush=True)
            raise

    async def stop(self, delete):
        self._stopping = True
        try:
            await super().stop(delete)
        finally:
            self._stopping = False

    async def start(self, force_build):
        await super().start(force_build)
        names = [self._env_vars.main_image_name, self.task_env_config.docker_image]
        images = []
        for name in dict.fromkeys(name for name in names if name):
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                name,
                "--format",
                "{{json .}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await proc.communicate()
            if proc.returncode == 0:
                info = json.loads(output)
                images.append(
                    {
                        "name": name,
                        "id": info["Id"],
                        "repo_digests": info.get("RepoDigests", []),
                        "task_image": name == self.task_env_config.docker_image,
                    }
                )
        suffix = "agent" if self.agent_install_spec else "verifier"
        write_json(self.trial_paths.trial_dir / f"images-{suffix}.json", images)


def build_job_config(
    context, raw_dir, task_root, task_ids, repeat, gateway_address=None
) -> JobConfig:
    conditions = context["conditions"]
    kwargs = {
        "version": context["generator"]["version"],
        "model_class": "litellm",
        "cost_limit": 0,
        "extra_python_packages": read_object(
            Path(__file__).parent / "data/deepswe-agent-lock.json"
        )["packages"],
    }
    if conditions["reasoning_effort"] != "server_default":
        kwargs["reasoning_effort"] = conditions["reasoning_effort"]
    return JobConfig.model_validate(
        {
            "job_name": f"repeat-{repeat}",
            "jobs_dir": raw_dir / "jobs",
            "n_attempts": 1,
            "n_concurrent_trials": conditions["workers"],
            "quiet": True,
            "timeout_multiplier": conditions["timeout_multiplier"],
            "agent_setup_timeout_multiplier": conditions[
                "agent_setup_timeout_multiplier"
            ],
            "retry": {
                "max_retries": conditions["max_retries"],
                "include_exceptions": [
                    "TransientAgentError",
                    "ConnectionError",
                    "TimeoutError",
                    "RuntimeError",
                    "EnvironmentBuildTimeoutError",
                    "AgentSetupTimeoutError",
                ],
            },
            "agents": [
                {
                    "import_path": "kairyu_bench.deepswe_runtime:KairyuMiniSweAgent",
                    "model_name": "openai/" + _model_id(context["model_id"]),
                    "kwargs": kwargs,
                }
            ],
            "environment": {
                "import_path": "kairyu_bench.deepswe_runtime:KairyuDockerEnvironment",
                "delete": True,
                "kwargs": {"gateway_address": gateway_address},
            },
            "tasks": [{"path": task_root / task} for task in task_ids],
        }
    )


def redact_artifacts(raw_dir: Path) -> None:
    # Retain trajectories and official outputs, with credential literals removed.
    values = [os.environ.get(key, "") for key in ("OPENAI_API_KEY", "HF_TOKEN")]
    secrets = [value.encode() for value in values if value and value != "not-required"]
    for path in raw_dir.rglob("*"):
        if path.is_file() and not path.is_symlink() and secrets:
            content = path.read_bytes()
            redacted = content
            for secret in secrets:
                redacted = redacted.replace(secret, b"[REDACTED]")
            if redacted != content:
                path.write_bytes(redacted)


def validate_tasks(task_root: Path, ids: list[str]) -> dict[str, str]:
    names = {}
    images = set()
    for task_id in ids:
        task = Task(task_root / task_id)
        config = task.config
        if task.name != "datacurve/" + task_id:
            raise ValueError("DeepSWE task.toml name differs from its directory ID")
        if (
            config.agent.network_mode != "no-network"
            or config.verifier.network_mode != "no-network"
            or config.verifier.environment_mode != "separate"
        ):
            raise ValueError("DeepSWE task isolation differs from the locked dataset")
        image = config.environment.docker_image
        if not image or image in images:
            raise ValueError(
                "DeepSWE requires a distinct task image per problem for bounded cleanup"
            )
        images.add(image)
        names[task.name] = task_id
    return names


async def cleanup_images(trial_dir: Path) -> None:
    if os.environ.get("KAIRYU_BENCH_CLEAN_TASK_IMAGES") != "1":
        return
    names: set[str] = set()
    for path in trial_dir.rglob("managed-images-*.json"):
        names.update(name for name in read_object(path)["names"] if name)
    for name in sorted(names):
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await inspect.wait() != 0:
            continue
        remove = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "rm",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await remove.wait() != 0:
            write_json(
                trial_dir / "cleanup-error.json",
                {"type": "TaskImageRemovalError", "image": name},
            )
            print(f"DeepSWE cleanup failed: {name}", flush=True)


async def run(context: dict[str, Any], raw_dir: Path, task_root: Path) -> None:
    all_ids = select_deepswe_tasks(task_root, None)
    if len(all_ids) != context["dataset"]["tasks"]:
        raise ValueError("DeepSWE source does not contain the locked 113 tasks")
    names = validate_tasks(task_root, all_ids)
    ids = select_deepswe_tasks(task_root, context.get("limit"))
    attempts = context["conditions"]["attempts_per_task"]
    write_json(
        raw_dir / "selected.json",
        {
            "problem_ids": ids,
            "model_id": context["model_id"],
            "attempts": attempts,
            "agent_version": context["generator"]["version"],
            "dataset": context["dataset"],
        },
    )
    (raw_dir / "instance-ids.txt").write_text("".join(task + "\n" for task in ids))
    write_trial_plan(raw_dir, ids, attempts)
    packages = {
        name: metadata.version(name)
        for name in ("datacurve-pier", "mini-swe-agent", "litellm", "openai")
    }
    write_json(
        raw_dir / "runtime.json",
        {
            "packages": packages,
            "source": context["source"],
            "dataset": context["dataset"],
            "agent_lock_sha256": sha256_file(
                Path(__file__).parent / "data/deepswe-agent-lock.json"
            ),
            "conditions": context["conditions"],
        },
    )
    gateway = None
    if urlsplit(os.environ["OPENAI_BASE_URL"]).hostname == "host.docker.internal":
        gateway = socket.gethostbyname("host.docker.internal")
    active: dict[str, str] = {}
    started = time.monotonic()

    def snapshot():
        summary = summarize_deepswe(raw_dir)
        state = {
            **summary,
            "active": active.copy(),
            "elapsed_seconds": round(time.monotonic() - started),
            "root_free_bytes": shutil.disk_usage("/").free,
            "cache_free_bytes": shutil.disk_usage(
                os.environ.get("KAIRYU_BENCH_CACHE_DIR", raw_dir)
            ).free,
        }
        write_json(raw_dir / "progress.json", state)
        print(
            f"DeepSWE progress: scored={summary['scored_attempts']}/{summary['requested_attempts']} "
            f"passed={summary['passed_attempts']} tasks={summary['evaluated_tasks']}/{len(ids)} "
            f"elapsed={state['elapsed_seconds']}s active={','.join(active)} "
            f"root_free={state['root_free_bytes']} cache_free={state['cache_free_bytes']}",
            flush=True,
        )

    for repeat in range(attempts):
        config = build_job_config(context, raw_dir, task_root, ids, repeat, gateway)
        job = await Job.create(config)

        async def progress(event):
            active[event.task_name] = event.event.value
            print(
                f"DeepSWE {event.event.value}: repeat={repeat + 1}/{attempts} task={event.task_name}",
                flush=True,
            )

        async def ended(event):
            if (
                event.task_name not in names
                or names[event.task_name] not in ids
                or event.result is None
            ):
                raise ValueError("DeepSWE received an unknown or empty trial result")
            task_id = names[event.task_name]
            trial_dir = event.config.trials_dir / event.config.trial_name
            await cleanup_images(trial_dir)
            destination = (
                raw_dir / "trials" / f"repeat-{repeat}" / task_id / "result.json"
            )
            write_json(destination, event.result.model_dump(mode="json"))
            images = []
            for path in trial_dir.rglob("images-*.json"):
                images.extend(json.loads(path.read_text()))
            write_json(destination.parent / "images.json", {"images": images})
            # Pier deletes failed trial directories before retrying. Preserve
            # their trajectories, verifier logs and cleanup failures first.
            if event.result.exception_info:
                retry_dir = (
                    raw_dir
                    / "retries"
                    / f"repeat-{repeat}"
                    / task_id
                    / str(event.result.id)
                )
                shutil.copytree(trial_dir, retry_dir, dirs_exist_ok=True)
            active.pop(event.task_name, None)
            summary = summarize_deepswe(raw_dir)
            write_json(raw_dir / "summary.json", summary)
            snapshot()
            print(
                f"DeepSWE completed: trials={summary['scored_attempts']}/{summary['requested_attempts']} "
                f"passed={summary['passed_attempts']} task={event.task_name} repeat={repeat + 1}",
                flush=True,
            )
            if event.result.exception_info:
                print(
                    f"DeepSWE error/retry candidate: {event.result.exception_info.exception_type} task={event.task_name}",
                    flush=True,
                )

        job.on_trial_started(progress).on_agent_started(
            progress
        ).on_verification_started(progress).on_trial_ended(ended)
        running = asyncio.create_task(job.run())
        try:
            while not running.done():
                await asyncio.wait({running}, timeout=60)
                if not running.done():
                    snapshot()
            await running
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
        redact_artifacts(raw_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_root", type=Path)
    parser.add_argument("raw_dir", type=Path)
    args = parser.parse_args()
    context = read_object(Path(os.environ["KAIRYU_BENCH_CONTEXT"]))
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    # Pier handles cancellation through its trial finalizers.
    async def execute():
        task = asyncio.current_task()
        if task is not None:
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        await run(context, args.raw_dir, args.task_root)

    code = 0
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
        asyncio.run(execute())
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
        write_json(args.raw_dir / "runtime-error.json", {"type": type(error).__name__})
        print(f"DeepSWE runtime failed: {type(error).__name__}", flush=True)
        code = 1
    finally:
        redact_artifacts(args.raw_dir)
        if (args.raw_dir / "selected.json").exists():
            write_json(args.raw_dir / "summary.json", summarize_deepswe(args.raw_dir))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
