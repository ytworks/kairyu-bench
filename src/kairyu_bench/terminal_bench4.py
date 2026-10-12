"""Terminal-Bench 4.0 settings, Harbor job config and the pinned leaderboard rules.

Standard library only: Harbor is imported by the job plugin and the prepare
step, never here, so accounting stays testable without the pinned runtime.
The metric and static-check functions are copied from
harbor-framework/terminal-bench@209679e34327a78ce8e9bf5300b2f317992863c9
(leaderboard/src/leaderboard/core/metrics.py and ci/static_analysis.py).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kairyu_bench.artifacts import env_integer, read_object

BENCHMARK = "terminal-bench-4"
LABEL = "Terminal-Bench 4.0"
ENV_PREFIX = "KAIRYU_BENCH_TERMINAL_BENCH_4_"
ATTEMPT_CHOICES = (1, 5)
LOCK_PATH = Path(__file__).parent / "data/terminal-bench-4.json"
CODE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("terminal_bench4_accounting.py"),
    Path(__file__).with_name("terminal_bench4_plugin.py"),
    LOCK_PATH,
)
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
GPU_POLICY = "run-all-gpu-unavailable-scored-zero-v1"
ERROR_POLICY = "tb4-model-scored-zero-infrastructure-unscored-v1"
COST_POLICY = "unavailable-for-local-api"

PASS_AT_KS = (2, 3, 4, 5)
CI95_Z = 1.96
TIMEOUT_MULTIPLIERS = (
    "agent_timeout_multiplier",
    "verifier_timeout_multiplier",
    "agent_setup_timeout_multiplier",
    "environment_build_timeout_multiplier",
)
AGENT_OVERRIDES = ("override_timeout_sec", "override_setup_timeout_sec", "max_timeout_sec")
VERIFIER_OVERRIDES = ("override_timeout_sec", "max_timeout_sec")
ENVIRONMENT_OVERRIDES = (
    "override_cpus",
    "override_gpus",
    "override_memory_mb",
    "override_storage_mb",
)


def load_lock() -> dict[str, Any]:
    lock = read_object(LOCK_PATH, LABEL)
    tasks = lock.get("tasks")
    if (
        lock.get("schema_version") != 1
        or not isinstance(tasks, dict)
        or len(tasks) != 66
        or not DIGEST.fullmatch(str(lock.get("dataset_ref")))
        or any(not DIGEST.fullmatch(str(task.get("digest"))) for task in tasks.values())
    ):
        raise ValueError("Terminal-Bench 4.0 dataset lock is invalid")
    return lock


def agent_lock(lock: Mapping[str, Any], agent: str) -> dict[str, Any]:
    try:
        return dict(lock["agents"][agent])
    except KeyError as error:
        raise ValueError(f"unsupported Terminal-Bench 4.0 agent: {agent}") from error


@dataclass(frozen=True)
class TerminalBench4Settings:
    workers: int = 4
    attempts: int = 1
    retries: int = 3
    reasoning_effort: str | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], agent: str
    ) -> "TerminalBench4Settings":
        workers = env_integer(env, ENV_PREFIX + "WORKERS", 4, 1, 16)
        attempts = env_integer(env, ENV_PREFIX + "ATTEMPTS", 1, 1, 5)
        if attempts not in ATTEMPT_CHOICES:
            raise ValueError(f"{ENV_PREFIX}ATTEMPTS must be 1 or 5")
        retries = env_integer(env, ENV_PREFIX + "RETRIES", 3, 0, 16)
        effort = env.get(ENV_PREFIX + "REASONING_EFFORT") or None
        allowed = agent_lock(load_lock(), agent)["reasoning_efforts"]
        if effort is not None and effort not in allowed:
            raise ValueError(
                f"{ENV_PREFIX}REASONING_EFFORT for {agent} must be one of "
                + ", ".join(allowed)
            )
        return cls(workers, attempts, retries, effort)


def code_sha256() -> str:
    digest = hashlib.sha256()
    for path in CODE_PATHS:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def resolve_terminal_bench4_conditions(
    entry: Mapping[str, Any], env: Mapping[str, str], agent: str
) -> dict[str, Any]:
    settings = TerminalBench4Settings.from_env(env, agent)
    lock = load_lock()
    pinned = agent_lock(lock, agent)
    if entry["dataset"]["revision"] != lock["dataset_ref"]:
        raise ValueError("Terminal-Bench 4.0 manifest digest differs from its lock")
    return {
        "benchmark_version": lock["benchmark_version"],
        "dataset_ref": lock["dataset_ref"],
        "harbor_version": lock["harbor_version"],
        "harbor_revision": entry["source"]["revision"],
        "code_sha256": code_sha256(),
        "agent": agent,
        "agent_kwargs": pinned["kwargs"],
        "agent_env": pinned["env"],
        "model_prefix": pinned["model_prefix"],
        "reasoning_effort": settings.reasoning_effort or "server_default",
        "attempts_per_task": settings.attempts,
        "workers": settings.workers,
        "max_retries": settings.retries,
        "retry_exclude_exceptions": pinned["retry_exclude_exceptions"],
        "timeout_multiplier": 1.0,
        **{name: None for name in TIMEOUT_MULTIPLIERS},
        "resource_overrides": None,
        "environment": "docker",
        "gpu_policy": GPU_POLICY,
        "error_policy": ERROR_POLICY,
        "cost_policy": COST_POLICY,
    }


def select_tasks(lock: Mapping[str, Any], limit: object) -> list[str]:
    tasks = sorted(lock["tasks"])
    if limit is None:
        return tasks
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Terminal-Bench 4.0 limit must be a positive integer")
    return tasks[:limit]


def model_name(conditions: Mapping[str, Any], model_id: str) -> str:
    return str(conditions["model_prefix"]) + model_id


def build_job_config(
    context: Mapping[str, Any], raw_dir: Path, overlay: Path, task_cache: Path
) -> dict[str, Any]:
    conditions = context["conditions"]
    agent = context["agent"]
    if conditions["agent"] != agent:
        raise ValueError("Terminal-Bench 4.0 conditions were resolved for another agent")
    kwargs = dict(conditions["agent_kwargs"])
    if conditions["reasoning_effort"] != "server_default":
        kwargs["reasoning_effort"] = conditions["reasoning_effort"]
    if agent == "terminus-2":
        kwargs["api_base"] = context["endpoint"]
    agent_config: dict[str, Any] = {
        "name": agent,
        "model_name": model_name(conditions, context["model_id"]),
        "kwargs": kwargs,
    }
    if conditions["agent_env"]:
        agent_config["env"] = dict(conditions["agent_env"])
    dataset: dict[str, Any] = {
        "name": context["dataset"]["id"],
        "ref": conditions["dataset_ref"],
        "download_dir": str(task_cache),
    }
    if context.get("limit") is not None:
        dataset["task_names"] = select_tasks(load_lock(), context["limit"])
    config = {
        "job_name": f"{context['run_id']}-{BENCHMARK}",
        "jobs_dir": str(raw_dir / "jobs"),
        "n_attempts": conditions["attempts_per_task"],
        "n_concurrent_trials": conditions["workers"],
        "quiet": True,
        "retry": {
            "max_retries": conditions["max_retries"],
            "exclude_exceptions": list(conditions["retry_exclude_exceptions"]),
        },
        "environment": {
            "type": "docker",
            "delete": True,
            "extra_docker_compose": [str(overlay)],
        },
        "agents": [agent_config],
        "datasets": [dataset],
    }
    failures = job_config_failures(config)
    if failures:
        raise ValueError("Terminal-Bench 4.0 job config breaks leaderboard rules: " + failures[0])
    return config


def is_success(reward: object) -> bool:
    """A trial succeeds iff reward > 0 (errored trials have none -> failure)."""
    return bool(reward) and float(reward) > 0  # type: ignore[arg-type]


def compute_metrics(by_task: Mapping[str, list[Any]]) -> tuple[float, float]:
    """Return (accuracy, standard_error) as percentages, as the leaderboard does."""
    total = sum(len(rewards) for rewards in by_task.values())
    successful = sum(
        1 for rewards in by_task.values() for reward in rewards if is_success(reward)
    )
    accuracy = 100.0 * successful / total if total else 0.0
    n = len(by_task)
    variance = 0.0
    for rewards in by_task.values():
        k = len(rewards)
        if k < 2:
            continue
        p = sum(1 for reward in rewards if is_success(reward)) / k
        variance += p * (1.0 - p) / (k - 1)
    variance = variance / (n * n) if n else 0.0
    return accuracy, 100.0 * variance**0.5


def compute_pass_at_k(
    by_task: Mapping[str, list[Any]], ks: tuple[int, ...] = PASS_AT_KS
) -> dict[str, float]:
    """The unbiased per-task estimator 1 - C(n-c, k)/C(n, k), averaged over tasks."""
    out: dict[str, float] = {}
    for k in ks:
        values = []
        for rewards in by_task.values():
            n = len(rewards)
            if n < k:
                continue
            c = sum(1 for reward in rewards if is_success(reward))
            if n - c < k:
                values.append(1.0)
                continue
            miss = 1.0
            for i in range(k):
                miss *= (n - c - i) / (n - i)
            values.append(1.0 - miss)
        out[f"pass_at_{k}"] = round(sum(values) / len(values), 4) if values else 0.0
    return out


def _section_overrides(
    section: Mapping[str, Any], keys: tuple[str, ...], prefix: str = ""
) -> list[str]:
    return [f"{prefix}{key}={section.get(key)}" for key in keys if section.get(key) is not None]


def _timeout_multiplier_ok(value: object) -> bool:
    return value is None or value == 1.0


def trial_config_failures(config: Mapping[str, Any]) -> list[str]:
    """Per-trial checks from the leaderboard static analysis."""
    failures: list[str] = []
    if not _timeout_multiplier_ok(config.get("timeout_multiplier")):
        failures.append(f"timeout_multiplier={config.get('timeout_multiplier')}")
    failures += _section_overrides(config, TIMEOUT_MULTIPLIERS)
    failures += _section_overrides(config.get("agent") or {}, AGENT_OVERRIDES, "agent.")
    failures += _section_overrides(
        config.get("verifier") or {}, VERIFIER_OVERRIDES, "verifier."
    )
    failures += _section_overrides(
        config.get("environment") or {}, ENVIRONMENT_OVERRIDES, "environment."
    )
    return failures


def job_config_failures(config: Mapping[str, Any]) -> list[str]:
    """Job-level checks: the same keys, with every configured agent inspected."""
    failures = trial_config_failures({**config, "agent": None})
    for agent in config.get("agents") or []:
        failures += _section_overrides(agent, AGENT_OVERRIDES, "agent.")
    return failures


def build_trial_plan(
    trial_configs: list[Mapping[str, Any]], job_name: str
) -> dict[str, Any]:
    """Slots in Harbor's own order; attempt indices count each task's trials."""
    seen: dict[str, int] = {}
    slots = []
    for config in trial_configs:
        task = config["task"]["name"]
        trial_name = config["trial_name"]
        slots.append(
            {
                "task": task,
                "attempt_index": seen.get(task, 0),
                "trial_name": trial_name,
                "result_path": f"jobs/{job_name}/{trial_name}/result.json",
            }
        )
        seen[task] = seen.get(task, 0) + 1
    names = [slot["trial_name"] for slot in slots]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names
    ):
        raise ValueError("Terminal-Bench 4.0 trial names are empty, unsafe or duplicated")
    return {"schema_version": 1, "job_name": job_name, "slots": slots}


def is_final_attempt(
    exception_type: str | None, attempts_so_far: int, retry: Mapping[str, Any]
) -> bool:
    """Mirror Harbor's TrialQueue retry decision for one finished attempt."""
    if exception_type is None:
        return True
    exclude = retry.get("exclude_exceptions") or ()
    include = retry.get("include_exceptions") or ()
    if exclude and exception_type in exclude:
        return True
    if include and exception_type not in include:
        return True
    return attempts_so_far > int(retry.get("max_retries", 0))


def removable_images(
    lock: Mapping[str, Any], selected: list[str], finished_tasks: set[str]
) -> list[str]:
    """Images whose every selected referencing task has finished all attempts."""
    users: dict[str, set[str]] = {}
    for task in selected:
        for image in lock["tasks"][task]["images"]:
            users.setdefault(image, set()).add(task)
    return sorted(image for image, tasks in users.items() if tasks <= finished_tasks)


def removal_reference(image: str) -> str:
    """Docker removes digest-pinned images by repository@digest, not tag@digest."""
    name, at, digest = image.partition("@")
    if not at:
        return image
    repository = name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name
    return f"{repository}@{digest}"
