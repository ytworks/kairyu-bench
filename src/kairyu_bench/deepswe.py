"""DeepSWE selection and accounting, independent of the Pier environment."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MODEL_FAILURES = {"AgentTimeoutError", "ContextWindowExceededError"}
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read DeepSWE artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"DeepSWE artifact must be an object: {path}")
    return value


def _integer(
    env: Mapping[str, str], suffix: str, default: int, low: int, high: int
) -> int:
    key = "KAIRYU_BENCH_DEEPSWE_" + suffix
    text = env.get(key, str(default))
    if not re.fullmatch(r"[0-9]+", text) or not low <= int(text) <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return int(text)


@dataclass(frozen=True)
class DeepSWESettings:
    workers: int = 4
    attempts: int = 1
    retries: int = 3
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DeepSWESettings":
        workers = _integer(env, "WORKERS", 4, 1, 16)
        attempts = _integer(env, "ATTEMPTS", 1, 1, 4)
        if attempts not in {1, 4}:
            raise ValueError("KAIRYU_BENCH_DEEPSWE_ATTEMPTS must be 1 or 4")
        retries = _integer(env, "RETRIES", 3, 0, 16)
        effort = env.get("KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT") or None
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError("KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT is invalid")
        return cls(workers, attempts, retries, effort)


def resolve_deepswe_conditions(
    entry: dict[str, Any], env: Mapping[str, str]
) -> dict[str, Any]:
    settings = DeepSWESettings.from_env(env)
    return {
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


def select_deepswe_tasks(task_root: Path, limit: int | None) -> list[str]:
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


def _trial_reward(payload: dict[str, Any]) -> tuple[int | None, str | None]:
    exception = payload.get("exception_info")
    if exception:
        if not isinstance(exception, dict) or not isinstance(
            exception.get("exception_type"), str
        ):
            raise ValueError("DeepSWE exception_info is invalid")
        kind = exception["exception_type"]
        return (0, None) if kind in MODEL_FAILURES else (None, kind)
    metadata = (payload.get("agent_result") or {}).get("metadata") or {}
    exit_status = metadata.get("mini_exit_status")
    if exit_status in MODEL_FAILURES or exit_status == "TimeExceeded":
        return 0, None
    if exit_status and exit_status not in {
        "Submitted",
        "LimitsExceeded",
        "RepeatedFormatError",
    }:
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
    info = payload.get("agent_info")
    if not isinstance(info, dict) or info.get("name") != "mini-swe-agent":
        raise ValueError("DeepSWE trial agent differs from mini-swe-agent")
    if info.get("version") != selected["agent_version"]:
        raise ValueError("DeepSWE trial agent version differs from the source lock")
    model = info.get("model_info")
    if (
        not isinstance(model, dict)
        or model.get("name") != selected["model_id"]
        or model.get("provider") != "openai"
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
            reward, error = _trial_reward(payload)
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
        "agent": "mini-swe-agent",
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
