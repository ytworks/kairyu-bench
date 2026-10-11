"""Terminal-Bench 4.0 trial classification, identity checks and run summary."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kairyu_bench.artifacts import read_object
from kairyu_bench.terminal_bench4 import (
    ATTEMPT_CHOICES,
    LABEL,
    compute_metrics,
    compute_pass_at_k,
    is_success,
    load_lock,
    trial_config_failures,
)

# The leaderboard scores these trials 0 and so does this runner: the model
# itself ran out of time or context, overflowed its output, or refused.
MODEL_FAILURES = frozenset(
    {
        "AgentTimeoutError",
        "ContextWindowExceededError",
        "OutputTokenExceededError",
        "AgentSafetyRefusalError",
    }
)
# A failed trial with one of these is an API outage, not a model result.
INFRASTRUCTURE_API_ERRORS = frozenset(
    {
        "ApiRateLimitError",
        "ApiUsageLimitError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiConnectionClosedError",
        "ApiResponseStalledError",
        "UnknownApiError",
        "ApiProviderResourceNotFoundError",
        "AgentAuthenticationError",
        "ModelNotFoundError",
        "NetworkConnectionError",
    }
)
GPU_UNAVAILABLE = "gpu-unavailable"
# dockerd's message when a Compose NVIDIA device request has no GPU runtime.
GPU_DRIVER_ERROR = re.compile(
    r'could not select device driver "[^"]*" with capabilities: \[\[gpu\]\]'
)


@dataclass(frozen=True)
class Outcome:
    success: int | None
    reason: str | None = None


def _reward(payload: Mapping[str, Any]) -> float | None:
    rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
    if not isinstance(rewards, dict):
        raise ValueError(f"{LABEL} verifier rewards must be an object")
    reward = rewards.get("reward")
    if reward is None:
        return None
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise ValueError(f"{LABEL} verifier reward must be a finite number")
    return float(reward)


def _exception(payload: Mapping[str, Any]) -> tuple[str | None, str]:
    info = payload.get("exception_info")
    if not info:
        return None, ""
    if not isinstance(info, dict) or not isinstance(info.get("exception_type"), str):
        raise ValueError(f"{LABEL} exception_info is invalid")
    return info["exception_type"], str(info.get("exception_message") or "")


def is_gpu_unavailable(
    payload: Mapping[str, Any], *, gpu_task: bool, gpu_available: bool
) -> bool:
    """A GPU task whose environment never started because no GPU runtime exists."""
    kind, message = _exception(payload)
    return (
        gpu_task
        and not gpu_available
        and kind == "RuntimeError"
        and bool(GPU_DRIVER_ERROR.search(message))
        and not payload.get("agent_setup")
        and not payload.get("agent_execution")
        and not payload.get("verifier_result")
    )


def classify_trial(
    payload: Mapping[str, Any], *, gpu_task: bool, gpu_available: bool
) -> Outcome:
    if not payload.get("finished_at"):
        return Outcome(None, "UnfinishedTrial")
    kind, _ = _exception(payload)
    reward = _reward(payload)
    if reward is not None:
        success = int(is_success(reward))
        if kind in INFRASTRUCTURE_API_ERRORS and not success:
            return Outcome(None, kind)
        return Outcome(success, kind)
    if is_gpu_unavailable(payload, gpu_task=gpu_task, gpu_available=gpu_available):
        return Outcome(0, GPU_UNAVAILABLE)
    if kind in MODEL_FAILURES:
        return Outcome(0, kind)
    return Outcome(None, kind or "MissingVerifierReward")


def validate_trial(
    payload: Mapping[str, Any],
    slot: Mapping[str, Any],
    selected: Mapping[str, Any],
    digests: Mapping[str, str],
) -> None:
    task = slot["task"]
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{LABEL} trial is missing its config")
    if payload.get("task_name") != task or (config.get("task") or {}).get("name") != task:
        raise ValueError(f"{LABEL} trial task differs from the trial plan")
    if payload.get("trial_name") != slot["trial_name"]:
        raise ValueError(f"{LABEL} trial name differs from the trial plan")
    if (config.get("task") or {}).get("ref") != digests[task]:
        raise ValueError(f"{LABEL} trial ran a task digest other than the pinned one")
    agent = config.get("agent") or {}
    info = payload.get("agent_info") or {}
    if agent.get("name") != selected["agent"] or info.get("name") != selected["agent"]:
        raise ValueError(f"{LABEL} trial agent differs from the selected agent")
    if agent.get("model_name") != selected["model_name"]:
        raise ValueError(f"{LABEL} trial model differs from the discovered API model")
    for key, value in selected["agent_kwargs"].items():
        if (agent.get("kwargs") or {}).get(key) != value:
            raise ValueError(f"{LABEL} trial agent {key} differs from the pin")
    failures = trial_config_failures(config)
    if failures:
        raise ValueError(f"{LABEL} trial breaks leaderboard rules: {failures[0]}")


def _plan_slots(raw_dir: Path, selected: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan_path = raw_dir / "trial-plan.json"
    if not plan_path.is_file():
        return []
    plan = read_object(plan_path, LABEL)
    slots = plan.get("slots")
    expected = sorted(
        (task, attempt)
        for task in selected["problem_ids"]
        for attempt in range(selected["attempts"])
    )
    if (
        plan.get("schema_version") != 1
        or plan.get("job_name") != selected["job_name"]
        or not isinstance(slots, list)
        or sorted((slot.get("task"), slot.get("attempt_index")) for slot in slots)
        != expected
    ):
        raise ValueError(f"{LABEL} trial plan differs from selected tasks and attempts")
    for slot in slots:
        expected_path = f"jobs/{plan['job_name']}/{slot['trial_name']}/result.json"
        if slot.get("result_path") != expected_path or "/" in slot["trial_name"]:
            raise ValueError(f"{LABEL} trial plan result path is invalid")
    return slots


def _check_unexpected_results(raw_dir: Path, slots: list[dict[str, Any]]) -> None:
    planned = {slot["result_path"] for slot in slots}
    actual = {
        str(path.relative_to(raw_dir))
        for path in (raw_dir / "jobs").glob("*/*/result.json")
    }
    if actual - planned:
        raise ValueError(f"{LABEL} contains trial results outside the trial plan")


def _runtime_errors(raw_dir: Path) -> list[str]:
    errors = sorted(
        str(path.relative_to(raw_dir))
        for directory in ("cleanup-errors", "plugin-errors")
        for path in (raw_dir / directory).glob("*.json")
    )
    if (raw_dir / "runtime-error.json").exists():
        errors.append("runtime-error.json")
    return errors


def _selected(raw_dir: Path) -> dict[str, Any]:
    selected = read_object(raw_dir / "selected.json", LABEL)
    tasks = selected.get("problem_ids")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(tasks) != len(set(tasks))
        or selected.get("attempts") not in ATTEMPT_CHOICES
    ):
        raise ValueError(f"{LABEL} selection is invalid")
    for field in ("agent", "model_id", "model_name", "job_name"):
        if not isinstance(selected.get(field), str) or not selected[field]:
            raise ValueError(f"{LABEL} selection is missing {field}")
    if not isinstance(selected.get("agent_kwargs"), dict):
        raise ValueError(f"{LABEL} selection is missing agent_kwargs")
    return selected


def _score(by_task: dict[str, list[Any]], attempts: int) -> dict[str, Any]:
    if not by_task or not any(by_task.values()):
        return {"accuracy_percent": None, "ci95_half_width_percent": None, "pass_at_k": None}
    accuracy, stderr = compute_metrics(by_task)
    repeated = attempts > 1
    return {
        "accuracy_percent": accuracy,
        "ci95_half_width_percent": 1.96 * stderr if repeated else None,
        "pass_at_k": compute_pass_at_k(by_task) if repeated else None,
    }


def summarize(raw_dir: Path) -> dict[str, Any]:
    lock = load_lock()
    selected = _selected(raw_dir)
    tasks: list[str] = selected["problem_ids"]
    if set(tasks) - set(lock["tasks"]):
        raise ValueError(f"{LABEL} selection contains tasks outside the pinned dataset")
    digests = {task: lock["tasks"][task]["digest"] for task in tasks}
    gpu_tasks = set(lock["gpu_tasks"])
    runtime_path = raw_dir / "runtime.json"
    gpu_available = bool(
        runtime_path.is_file() and read_object(runtime_path, LABEL).get("gpu_available")
    )
    slots = _plan_slots(raw_dir, selected)
    _check_unexpected_results(raw_dir, slots)
    scored: dict[str, list[int]] = {task: [] for task in tasks}
    planned: dict[str, list[int]] = {task: [] for task in tasks}
    errors: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    if not slots:
        errors.append({"type": "MissingTrialPlan"})
    for slot in slots:
        outcome = _slot_outcome(raw_dir, slot, selected, digests, gpu_tasks, gpu_available)
        if outcome.reason:
            reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
        if outcome.success is None:
            errors.append(
                {"task": slot["task"], "attempt_index": slot["attempt_index"], "type": outcome.reason}
            )
            planned[slot["task"]].append(0)
            continue
        scored[slot["task"]].append(outcome.success)
        planned[slot["task"]].append(outcome.success)
    summary = _summary(raw_dir, selected, scored, planned, errors, reasons, gpu_tasks)
    return {**summary, "gpu_available": gpu_available}


def _slot_outcome(
    raw_dir: Path,
    slot: Mapping[str, Any],
    selected: Mapping[str, Any],
    digests: Mapping[str, str],
    gpu_tasks: set[str],
    gpu_available: bool,
) -> Outcome:
    path = raw_dir / slot["result_path"]
    if not path.is_file():
        return Outcome(None, "MissingTrial")
    if not path.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError(f"{LABEL} trial path escapes its result directory")
    payload = read_object(path, LABEL)
    validate_trial(payload, slot, selected, digests)
    return classify_trial(
        payload, gpu_task=slot["task"] in gpu_tasks, gpu_available=gpu_available
    )


def _summary(
    raw_dir: Path,
    selected: Mapping[str, Any],
    scored: dict[str, list[int]],
    planned: dict[str, list[int]],
    errors: list[dict[str, Any]],
    reasons: dict[str, int],
    gpu_tasks: set[str],
) -> dict[str, Any]:
    tasks: list[str] = selected["problem_ids"]
    attempts: int = selected["attempts"]
    scored_count = sum(len(values) for values in scored.values())
    passed = sum(sum(values) for values in scored.values())
    evaluated = sum(len(scored[task]) == attempts for task in tasks)
    runtime_errors = _runtime_errors(raw_dir)
    complete = evaluated == len(tasks) and not errors and not runtime_errors
    non_gpu = {task: values for task, values in scored.items() if task not in gpu_tasks}
    gpu_unavailable = reasons.get("gpu-unavailable", 0)
    notes = ["local Docker environment, not the maintainers' Modal runs"]
    if gpu_unavailable:
        notes.append(f"{gpu_unavailable} GPU trial slots scored 0 without a GPU")
    return {
        "schema_version": 1,
        "problem_ids": tasks,
        "agent": selected["agent"],
        "model_id": selected["model_id"],
        "attempts_per_task": attempts,
        "evaluated_tasks": evaluated,
        "requested_attempts": len(tasks) * attempts,
        "scored_attempts": scored_count,
        "passed_attempts": passed,
        "gpu_unavailable_attempts": gpu_unavailable,
        "excluded_attempts": len(tasks) * attempts - scored_count,
        "outcome_reasons": dict(sorted(reasons.items())),
        "errors": errors,
        "runtime_errors": runtime_errors,
        **_score({task: values for task, values in scored.items() if values}, attempts),
        "leaderboard_equivalent": _score(
            {task: values for task, values in planned.items() if values}, attempts
        ),
        "non_gpu_accuracy_percent": _score(
            {task: values for task, values in non_gpu.items() if values}, attempts
        )["accuracy_percent"],
        "published_leaderboard_comparable": False,
        "comparability_notes": notes,
        "status": "completed" if complete else "partial" if scored_count else "failed",
        "cost_usd": None,
    }
