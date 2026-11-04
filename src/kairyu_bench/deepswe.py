"""DeepSWE selection and accounting, independent of the Pier environment."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kairyu_bench import artifacts
from kairyu_bench.artifacts import env_integer, sha256_file, write_json


MODEL_FAILURES = {"AgentTimeoutError", "ContextWindowExceededError"}
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
MINI_SWE_AGENT = "mini-swe-agent"
DEEPSEEK_HARNESS = "deepseek-harness"
DATA = Path(__file__).parent / "data"
DSH_LOCK = DATA / "deepswe-deepseek-harness-lock.json"
DSH_PATCH = DATA / "deepswe-deepseek-harness.patch.json"
DSH_MAX_TOKENS = 49152
# Agent identity recorded by Pier, the metadata key holding its exit status, and
# the exit statuses whose committed patch the verifier grades.
AGENTS = {
    MINI_SWE_AGENT: (
        "openai",
        "mini_exit_status",
        {"Submitted", "LimitsExceeded", "RepeatedFormatError"},
    ),
    DEEPSEEK_HARNESS: (
        "deepseek-official",
        "dsh_exit_status",
        {"completed", "max-tokens"},
    ),
}


def read_object(path: Path) -> dict[str, Any]:
    return artifacts.read_object(path, "DeepSWE")


def _integer(
    env: Mapping[str, str], suffix: str, default: int, low: int, high: int
) -> int:
    return env_integer(env, "KAIRYU_BENCH_DEEPSWE_" + suffix, default, low, high)


def dsh_compaction_valid(context_window: int, max_tokens: int) -> bool:
    """Mirror dsh compaction-basic: automatic compaction needs retain < threshold."""
    threshold = math.floor(min(0.8 * context_window, context_window - max_tokens - 65536))
    retain = math.floor(0.16 * (context_window - max_tokens))
    return 0 < threshold and retain < threshold


@dataclass(frozen=True)
class DeepSWESettings:
    workers: int = 4
    attempts: int = 1
    retries: int = 3
    reasoning_effort: str | None = None
    api_timeout_seconds: int | None = None
    tasks: tuple[str, ...] | None = None
    agent: str = MINI_SWE_AGENT
    context_window: int | None = None
    max_tokens: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DeepSWESettings":
        workers = _integer(env, "WORKERS", 4, 1, 16)
        attempts = _integer(env, "ATTEMPTS", 1, 1, 4)
        if attempts not in {1, 4}:
            raise ValueError("KAIRYU_BENCH_DEEPSWE_ATTEMPTS must be 1 or 4")
        retries = _integer(env, "RETRIES", 3, 0, 16)
        agent = env.get("KAIRYU_BENCH_DEEPSWE_AGENT") or MINI_SWE_AGENT
        if agent not in AGENTS:
            raise ValueError(
                "KAIRYU_BENCH_DEEPSWE_AGENT must be one of: " + ", ".join(AGENTS)
            )
        effort = env.get("KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT") or None
        efforts = (
            set(read_object(DSH_LOCK)["reasoning_efforts"])
            if agent == DEEPSEEK_HARNESS
            else REASONING_EFFORTS
        )
        if effort is not None and effort not in efforts:
            raise ValueError(
                f"KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT for {agent} must be one of: "
                + ", ".join(sorted(efforts))
            )
        dsh_only = ("CONTEXT_WINDOW", "MAX_TOKENS")
        context_window = max_tokens = None
        if agent == DEEPSEEK_HARNESS:
            if "KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS" in env:
                raise ValueError(
                    "KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS applies only to mini-swe-agent"
                )
            if not env.get("KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW"):
                raise ValueError(
                    "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW is required for deepseek-harness"
                )
            context_window = _integer(env, "CONTEXT_WINDOW", 0, 1, 100_000_000)
            max_tokens = _integer(env, "MAX_TOKENS", DSH_MAX_TOKENS, 1, 10_000_000)
            if not dsh_compaction_valid(context_window, max_tokens):
                raise ValueError(
                    "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW is too small for "
                    "KAIRYU_BENCH_DEEPSWE_MAX_TOKENS; dsh would disable compaction"
                )
        else:
            for suffix in dsh_only:
                if "KAIRYU_BENCH_DEEPSWE_" + suffix in env:
                    raise ValueError(
                        f"KAIRYU_BENCH_DEEPSWE_{suffix} applies only to deepseek-harness"
                    )
        api_timeout = None
        if "KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS" in env:
            api_timeout = _integer(env, "API_TIMEOUT_SECONDS", 600, 1, 10800)
        tasks = None
        if env.get("KAIRYU_BENCH_DEEPSWE_TASKS"):
            tasks = tuple(
                _task_ids(
                    [task.strip() for task in env["KAIRYU_BENCH_DEEPSWE_TASKS"].split(",")]
                )
            )
        return cls(
            workers,
            attempts,
            retries,
            effort,
            api_timeout,
            tasks,
            agent,
            context_window,
            max_tokens,
        )


def resolve_deepswe_conditions(
    entry: dict[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    settings = DeepSWESettings.from_env(env)
    if settings.agent == DEEPSEEK_HARNESS:
        conditions = _dsh_conditions(entry, settings)
    else:
        conditions = _mini_conditions(entry, settings)
    if settings.tasks is not None:
        conditions["selected_tasks"] = list(settings.tasks)
    return conditions


def _mini_conditions(entry: dict[str, Any], settings: DeepSWESettings) -> dict[str, Any]:
    conditions = {
        "benchmark_version": entry["dataset"]["version"],
        "agent_revision": entry["generator"]["revision"],
        "agent_version": entry["generator"]["version"],
        "adapter_sha256": sha256_file(Path(__file__).with_name("deepswe_runtime.py")),
        "agent_lock_sha256": sha256_file(
            Path(__file__).parent / "data/deepswe-agent-lock.json"
        ),
        "model_class": "litellm",
        "transport": "chat_completions",
        "reasoning_effort": settings.reasoning_effort or "server_default",
        "attempts_per_task": settings.attempts,
        "workers": settings.workers,
        "max_retries": settings.retries,
        "agent_setup_timeout_multiplier": 4,
        "timeout_multiplier": 1,
        "error_policy": "deepswe-v1.1-provider-verifier-network-excluded-v1",
        "cost_policy": "unavailable-for-local-api",
    }
    if settings.api_timeout_seconds is not None:
        conditions["api_timeout_seconds"] = settings.api_timeout_seconds
    return conditions


def _dsh_conditions(entry: dict[str, Any], settings: DeepSWESettings) -> dict[str, Any]:
    lock = read_object(DSH_LOCK)
    return {
        "benchmark_version": entry["dataset"]["version"],
        "agent": DEEPSEEK_HARNESS,
        "agent_repository": lock["repository"],
        "agent_revision": lock["revision"],
        "agent_version": lock["version"],
        "agent_profile": lock["profile"],
        "agent_provider": lock["provider"],
        "adapter_sha256": sha256_file(Path(__file__).with_name("deepswe_runtime.py")),
        "agent_adapter_sha256": sha256_file(Path(__file__).with_name("deepswe_dsh.py")),
        "agent_runner_sha256": sha256_file(
            Path(__file__).with_name("deepswe_dsh_runner.py")
        ),
        "agent_patch_sha256": sha256_file(DSH_PATCH),
        "agent_lock_sha256": sha256_file(DSH_LOCK),
        "disabled_profile_rows": [
            row["id"] for row in json.loads(DSH_PATCH.read_text(encoding="utf-8"))
        ],
        "uv_version": lock["uv"],
        "transport": "anthropic_messages",
        "reasoning_effort": settings.reasoning_effort or "agent_default",
        "context_window": settings.context_window,
        "max_tokens": settings.max_tokens,
        "attempts_per_task": settings.attempts,
        "workers": settings.workers,
        "max_retries": settings.retries,
        "agent_setup_timeout_multiplier": 4,
        "timeout_multiplier": 1,
        "error_policy": "deepswe-v1.1-provider-verifier-network-excluded-v1",
        "cost_policy": "unavailable-for-local-api",
    }


def _task_ids(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("DeepSWE selection must contain task IDs")
    if any(
        not isinstance(item, str) or not TASK_ID.fullmatch(item) or ".." in item
        for item in value
    ):
        raise ValueError("DeepSWE task ID is invalid")
    if len(value) != len(set(value)):
        raise ValueError("DeepSWE selection contains duplicate task IDs")
    return value


def select_deepswe_tasks(
    task_root: Path, limit: int | None, tasks: list[str] | None = None
) -> list[str]:
    ids = _task_ids(
        sorted(
            task.name
            for task in task_root.iterdir()
            if task.is_dir()
            and not task.is_symlink()
            and (task / "task.toml").is_file()
            and (task / "instruction.md").is_file()
        )
    )
    if tasks is not None:
        unknown = sorted(set(_task_ids(list(tasks))) - set(ids))
        if unknown:
            raise ValueError(f"DeepSWE tasks are not in the dataset: {unknown}")
        ids = [task for task in ids if task in set(tasks)]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("DeepSWE limit must be a positive integer")
        ids = ids[:limit]
    return ids


def write_trial_plan(raw_dir: Path, task_ids: list[str], attempts: int) -> Path:
    _task_ids(task_ids)
    if isinstance(attempts, bool) or attempts not in {1, 4}:
        raise ValueError("DeepSWE attempts must be 1 or 4")
    path = raw_dir / "trial-plan.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "slots": [
                {
                    "task_id": task,
                    "repeat_index": repeat,
                    "result_path": f"trials/repeat-{repeat}/{task}/result.json",
                }
                for repeat in range(attempts)
                for task in task_ids
            ],
        },
    )
    return path


def _trial_reward(
    payload: dict[str, Any], agent: str = MINI_SWE_AGENT
) -> tuple[int | None, str | None]:
    exception = payload.get("exception_info")
    if exception:
        if not isinstance(exception, dict) or not isinstance(
            exception.get("exception_type"), str
        ):
            raise ValueError("DeepSWE exception_info is invalid")
        kind = exception["exception_type"]
        return (0, None) if kind in MODEL_FAILURES else (None, kind)
    metadata = (payload.get("agent_result") or {}).get("metadata") or {}
    _, status_key, graded = AGENTS[agent]
    exit_status = metadata.get(status_key)
    if exit_status in MODEL_FAILURES or exit_status == "TimeExceeded":
        return 0, None
    if agent == DEEPSEEK_HARNESS and not exit_status:
        return None, "MissingAgentExitStatus"
    if exit_status and exit_status not in graded:
        return None, exit_status
    verifier = payload.get("verifier_result")
    if not isinstance(verifier, dict) or not isinstance(verifier.get("rewards"), dict):
        return None, "MissingVerifierResult"
    reward = verifier["rewards"].get("reward")
    if reward is None:
        return None, "MissingBinaryReward"
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or reward not in {0, 1}
    ):
        raise ValueError("DeepSWE verifier reward must be numeric binary 0 or 1")
    return int(reward), None


def _validate_trial(
    payload: dict[str, Any], selected: dict[str, Any], task: str
) -> None:
    # Pier records task.toml's namespaced name, not the directory name.
    if payload.get("task_name") != "datacurve/" + task:
        raise ValueError("DeepSWE trial task identity differs from the selection")
    agent = selected.get("agent", MINI_SWE_AGENT)
    info = payload.get("agent_info")
    if not isinstance(info, dict) or info.get("name") != agent:
        raise ValueError(f"DeepSWE trial agent differs from {agent}")
    if info.get("version") != selected["agent_version"]:
        raise ValueError("DeepSWE trial agent version differs from the source lock")
    model = info.get("model_info")
    if (
        not isinstance(model, dict)
        or model.get("name") != selected["model_id"]
        or model.get("provider") != AGENTS[agent][0]
    ):
        raise ValueError("DeepSWE trial model differs from the discovered API model")


def summarize_deepswe(raw_dir: Path) -> dict[str, Any]:
    selected = read_object(raw_dir / "selected.json")
    ids = _task_ids(selected.get("problem_ids"))
    attempts = selected.get("attempts")
    if isinstance(attempts, bool) or attempts not in {1, 4}:
        raise ValueError("DeepSWE selected attempts must be 1 or 4")
    for field in ("model_id", "agent_version"):
        if not isinstance(selected.get(field), str) or not selected[field]:
            raise ValueError(f"DeepSWE selection is missing {field}")
    agent = selected.get("agent", MINI_SWE_AGENT)
    if agent not in AGENTS:
        raise ValueError("DeepSWE selection names an unsupported agent")
    expected: list[dict[str, Any]] = [
        {
            "task_id": task,
            "repeat_index": repeat,
            "result_path": f"trials/repeat-{repeat}/{task}/result.json",
        }
        for repeat in range(attempts)
        for task in ids
    ]
    plan = read_object(raw_dir / "trial-plan.json")
    if plan.get("schema_version") != 1 or plan.get("slots") != expected:
        raise ValueError("DeepSWE trial plan differs from selected tasks and attempts")
    expected_paths = {slot["result_path"] for slot in expected}
    actual_paths = {
        str(path.relative_to(raw_dir))
        for path in (raw_dir / "trials").rglob("result.json")
    }
    if actual_paths - expected_paths:
        raise ValueError("DeepSWE contains unexpected trial results")
    outcomes: dict[tuple[str, int], int] = {}
    errors: list[dict[str, Any]] = []
    trial_ids: set[str] = set()
    for slot in expected:
        task, repeat = slot["task_id"], slot["repeat_index"]
        path = raw_dir / slot["result_path"]
        if not path.is_file():
            errors.append(
                {"task_id": task, "repeat_index": repeat, "type": "MissingTrial"}
            )
            continue
        if not path.resolve().is_relative_to(raw_dir.resolve()):
            raise ValueError("DeepSWE trial path escapes its result directory")
        payload = read_object(path)
        _validate_trial(payload, selected, task)
        trial_id = payload.get("id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in trial_ids:
            raise ValueError("DeepSWE trial ID is empty or duplicated")
        trial_ids.add(trial_id)
        error: str | None
        if not payload.get("finished_at"):
            reward, error = None, "UnfinishedTrial"
        else:
            reward, error = _trial_reward(payload, agent)
        if reward is None:
            errors.append({"task_id": task, "repeat_index": repeat, "type": error})
        else:
            outcomes[(task, repeat)] = reward
    scored = len(outcomes)
    passed = sum(outcomes.values())
    evaluated = sum(
        all((task, repeat) in outcomes for repeat in range(attempts)) for task in ids
    )
    runtime_errors = sorted(
        str(p.relative_to(raw_dir)) for p in raw_dir.rglob("cleanup-error.json")
    )
    if (raw_dir / "runtime-error.json").exists():
        runtime_errors.append("runtime-error.json")
    complete = evaluated == len(ids) and not runtime_errors
    runtime_provenance = None
    runtime_path = raw_dir / "runtime.json"
    if runtime_path.is_file():
        runtime = read_object(runtime_path)
        image_records: list[dict[str, Any]] = []
        image_slots = 0
        for slot in expected:
            images_path = (raw_dir / slot["result_path"]).with_name("images.json")
            if images_path.is_file():
                records = [
                    image
                    for image in read_object(images_path)["images"]
                    if image.get("task_image")
                ]
                image_slots += bool(records)
                image_records.extend(records)
        if image_slots == len(expected):
            runtime_provenance = {
                "packages": runtime["packages"],
                "agent_lock_sha256": runtime["agent_lock_sha256"],
                "task_images": sorted(
                    {json.dumps(record, sort_keys=True) for record in image_records}
                ),
            }
    repeat_scores = []
    for repeat in range(attempts):
        rewards = [
            outcomes[(task, repeat)] for task in ids if (task, repeat) in outcomes
        ]
        repeat_scores.append(100.0 * sum(rewards) / len(rewards) if rewards else None)
    ci_half = None
    if complete and attempts == 4:
        ci_half = (
            1.96
            * statistics.stdev(
                float(score) for score in repeat_scores if score is not None
            )
            / math.sqrt(attempts)
        )
    return {
        "schema_version": 1,
        "problem_ids": ids,
        "agent": agent,
        "model_id": selected["model_id"],
        "attempts_per_task": attempts,
        "evaluated_tasks": evaluated,
        "requested_attempts": len(ids) * attempts,
        "scored_attempts": scored,
        "passed_attempts": passed,
        "excluded_attempts": len(errors),
        "errors": errors,
        "runtime_errors": runtime_errors,
        "runtime_provenance": runtime_provenance,
        "pass_at_1_percent": 100.0 * passed / scored if scored else None,
        "pass_at_4_percent": 100.0
        * sum(
            any(outcomes[(task, repeat)] for repeat in range(attempts)) for task in ids
        )
        / len(ids)
        if complete and attempts == 4
        else None,
        "repeat_scores": repeat_scores,
        "ci_half_percent": ci_half,
        "ci_method": "95% run-to-run: 1.96 * sample_std(repeat_scores) / sqrt(4)"
        if ci_half is not None
        else None,
        "status": "completed" if complete else "partial" if scored else "failed",
        "cost_usd": None,
    }
