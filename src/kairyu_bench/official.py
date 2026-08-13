from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from kairyu_bench.results import BenchmarkResult


class OfficialNormalizationError(ValueError):
    """An official harness artifact cannot be normalized without guessing."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialNormalizationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OfficialNormalizationError(f"{field} must be finite")
    return result


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OfficialNormalizationError(f"{field} must be a non-negative integer")
    return value


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OfficialNormalizationError(f"{field} must be an object")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OfficialNormalizationError(f"{field} must be an array")
    return value


def _ids(values: Iterable[object], field: str) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise OfficialNormalizationError(f"{field} contains an invalid ID")
        item_id = str(value)
        if not item_id or item_id in result:
            raise OfficialNormalizationError(f"{field} contains an empty or duplicate ID")
        result.append(item_id)
    if not result:
        raise OfficialNormalizationError(f"{field} contains no problem IDs")
    return result


@dataclass(frozen=True)
class OfficialObservation:
    problem_ids: list[str]
    evaluated: int
    primary: float
    metrics: dict[str, Any]
    error: str | None = None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OfficialNormalizationError(f"cannot read official JSON {path}: {error}") from error


def _swebench(path: Path) -> OfficialObservation:
    report = _object(_load_json(path), "SWE-bench report")
    problem_ids = _ids(_array(report.get("submitted_ids"), "submitted_ids"), "submitted_ids")
    total = _integer(report.get("total_instances"), "total_instances")
    completed = _integer(report.get("completed_instances"), "completed_instances")
    resolved = _integer(report.get("resolved_instances"), "resolved_instances")
    if total != len(problem_ids):
        raise OfficialNormalizationError(
            "SWE-bench total_instances differs from submitted_ids; explicit --instance_ids is required"
        )
    if resolved > completed or completed > total:
        raise OfficialNormalizationError("SWE-bench report counts are inconsistent")
    primary = 100.0 * resolved / total
    return OfficialObservation(
        problem_ids,
        completed,
        primary,
        {
            "resolved_percent": primary,
            "resolved_instances": resolved,
            "completed_instances": completed,
            "total_instances": total,
        },
        None if completed == total else "official harness did not complete every selected instance",
    )


def _harbor(path: Path) -> OfficialObservation:
    candidates = sorted(path.rglob("result.json")) if path.is_dir() else [path]
    rows: list[tuple[str, float | None]] = []
    for candidate in candidates:
        payload = _load_json(candidate)
        if not isinstance(payload, dict) or "task_name" not in payload:
            continue
        task_id = str(payload["task_name"])
        verifier = payload.get("verifier_result")
        reward: float | None = None
        if isinstance(verifier, dict):
            rewards = verifier.get("rewards")
            if isinstance(rewards, dict):
                raw_reward = rewards.get("reward")
                if raw_reward is None and len(rewards) == 1:
                    raw_reward = next(iter(rewards.values()))
                if raw_reward is not None:
                    reward = _number(raw_reward, f"{candidate}.verifier_result.rewards")
        rows.append((task_id, reward))
    problem_ids = _ids((task_id for task_id, _ in rows), "Harbor task_name")
    rewards = [reward for _, reward in rows if reward is not None]
    if not rewards:
        raise OfficialNormalizationError("Harbor produced no verifier rewards")
    primary = 100.0 * sum(rewards) / len(problem_ids)
    return OfficialObservation(
        problem_ids,
        len(rewards),
        primary,
        {"mean_reward_percent": primary, "rewarded_tasks": len(rewards)},
        None if len(rewards) == len(problem_ids) else "one or more Harbor trials have no verifier reward",
    )


def _hle(path: Path) -> OfficialObservation:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = _object(json.loads(line), f"HLE line {line_number}")
            if row.get("_type") == "metrics_summary":
                summary = row
            else:
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise OfficialNormalizationError(f"cannot read HLE JSONL: {error}") from error
    if summary is None:
        raise OfficialNormalizationError("HLE metrics_summary line is missing")
    problem_ids = _ids(
        (row.get("id", row.get("question_id")) for row in rows), "HLE result IDs"
    )
    metrics = _object(summary.get("metrics"), "HLE metrics")
    total = _integer(metrics.get("total_questions"), "HLE total_questions")
    successful = _integer(metrics.get("evaluated_questions"), "HLE evaluated_questions")
    failed = _integer(metrics.get("failed_questions"), "HLE failed_questions")
    if total != len(problem_ids) or successful + failed != total:
        raise OfficialNormalizationError("HLE summary counts differ from result rows")
    accuracy = _number(metrics.get("accuracy"), "HLE accuracy")
    return OfficialObservation(
        problem_ids,
        total,
        accuracy,
        {
            "accuracy": accuracy,
            "accuracy_success_only": metrics.get("accuracy_success_only"),
            "calibration_error": metrics.get("calibration_error"),
            "successful_questions": successful,
            "failed_questions": failed,
        },
        None,
    )


def _livecodebench(path: Path) -> OfficialObservation:
    payload = _object(_load_json(path), "LiveCodeBench summary")
    metrics_rows = _array(payload.get("metrics"), "LiveCodeBench metrics")
    if not metrics_rows:
        raise OfficialNormalizationError("LiveCodeBench metrics are empty")
    metrics = _object(metrics_rows[0], "LiveCodeBench metrics[0]")
    primary = 100.0 * _number(metrics.get("pass@1"), "LiveCodeBench pass@1")
    records = _array(payload.get("records"), "LiveCodeBench records")
    problem_ids = _ids(
        (
            _object(record, "LiveCodeBench record").get(
                "question_id", _object(record, "LiveCodeBench record").get("id")
            )
            for record in records
        ),
        "LiveCodeBench record IDs",
    )
    return OfficialObservation(
        problem_ids,
        len(problem_ids),
        primary,
        {"pass_at_1": primary},
    )


def _livecodebench_pro(path: Path) -> OfficialObservation:
    rows = _array(_load_json(path), "LiveCodeBench Pro results")
    objects = [_object(row, "LiveCodeBench Pro result") for row in rows]
    problem_ids = _ids((row.get("problem_id") for row in objects), "problem_id")
    final = [row for row in objects if row.get("judge_result") not in {None, "Judging"}]
    accepted = sum(row.get("judge_result") == "Accepted" for row in final)
    primary = 100.0 * accepted / len(problem_ids)
    return OfficialObservation(
        problem_ids,
        len(final),
        primary,
        {
            "pass_at_1": primary,
            "accepted": accepted,
            "final_judgements": len(final),
        },
        None if len(final) == len(problem_ids) else "official checker did not return a final result for every problem",
    )


def _charxiv(path: Path) -> OfficialObservation:
    payload = _object(_load_json(path), "CharXiv summary")
    scores = _object(payload.get("scores"), "CharXiv scores")
    problem_ids = _ids(scores.keys(), "CharXiv figure IDs")
    stats = _object(payload.get("stats"), "CharXiv stats")
    valid = _integer(stats.get("N_valid"), "CharXiv N_valid")
    invalid = _integer(stats.get("N_invalid"), "CharXiv N_invalid")
    if valid != len(problem_ids) or invalid > valid:
        raise OfficialNormalizationError("CharXiv valid/invalid counts differ from score rows")
    primary = _number(stats.get("Overall Score"), "CharXiv Overall Score")
    return OfficialObservation(
        problem_ids,
        valid - invalid,
        primary,
        {"overall_score": primary, "valid": valid, "invalid": invalid},
        None if invalid == 0 else "one or more CharXiv judge results were invalid",
    )


def _scicode(path: Path) -> OfficialObservation:
    payload = _object(_load_json(path), "SciCode Inspect summary")
    problem_ids = _ids(_array(payload.get("problem_ids"), "problem_ids"), "problem_ids")
    metrics = _object(payload.get("metrics"), "SciCode metrics")
    main = _number(metrics.get("Problem Correctness"), "SciCode Problem Correctness")
    sub = _number(
        metrics.get("sub_problem_correctness"), "SciCode sub_problem_correctness"
    )
    primary = main * 100.0
    return OfficialObservation(
        problem_ids,
        len(problem_ids),
        primary,
        {"main_problem_percent": primary, "subproblem_percent": sub * 100.0},
    )


def _tau(path: Path) -> OfficialObservation:
    payload = _load_json(path)
    if isinstance(payload, dict):
        rows = _array(payload.get("simulations"), "tau2 simulations")
    else:
        rows = _array(payload, "tau2 results")
    objects = [_object(row, "tau2 simulation") for row in rows]
    problem_ids = _ids(
        (
            (
                f"{row.get('task_id')}#trial-{row.get('trial')}"
                if row.get("trial") is not None
                else row.get("task_id")
            )
            for row in objects
        ),
        "tau2 task_id/trial",
    )
    rewards: list[float] = []
    for row in objects:
        reward_info = row.get("reward_info")
        if isinstance(reward_info, dict) and reward_info.get("reward") is not None:
            rewards.append(_number(reward_info["reward"], "tau2 reward_info.reward"))
        elif row.get("reward") is not None:
            rewards.append(_number(row["reward"], "tau2 reward"))
    if not rewards:
        raise OfficialNormalizationError("tau2 produced no rewards")
    primary = 100.0 * sum(rewards) / len(problem_ids)
    return OfficialObservation(
        problem_ids,
        len(rewards),
        primary,
        {"average_reward_percent": primary, "rewarded_simulations": len(rewards)},
        None if len(rewards) == len(problem_ids) else "one or more tau2 simulations have no reward",
    )


NORMALIZERS: dict[str, Callable[[Path], OfficialObservation]] = {
    "swe-bench-pro": _swebench,
    "swe-bench-verified": _swebench,
    "terminal-bench": _harbor,
    "livecodebench": _livecodebench,
    "livecodebench-pro": _livecodebench_pro,
    "hle": _hle,
    "charxiv-reasoning": _charxiv,
    "scicode": _scicode,
    "tau-bench-banking": _tau,
}


def _artifact_path(context: dict[str, Any], raw_path: Path) -> str:
    try:
        return str(raw_path.resolve().relative_to(Path(context["run_dir"]).resolve()))
    except (KeyError, ValueError):
        return f"raw/{context['benchmark']}/{raw_path.name}"


def _result(
    context: dict[str, Any],
    *,
    status: str,
    problem_ids: list[str],
    evaluated: int,
    primary: float | None,
    metrics: dict[str, Any],
    raw: list[str],
    error: str | None,
) -> BenchmarkResult:
    scoring = context["scoring"]
    return BenchmarkResult.from_dict(
        {
            "schema_version": 1,
            "run_id": context["run_id"],
            "benchmark": context["benchmark"],
            "status": status,
            "endpoint": {"fingerprint": context["endpoint_fingerprint"]},
            "model_id": context["model_id"],
            "source": {
                "repository": context["source"]["repository"],
                "revision": context["source"]["revision"],
                "dataset": context["dataset"]["id"],
                "dataset_revision": context["dataset"]["revision"],
            },
            "selection": {
                "requested_limit": context.get("limit"),
                "problem_ids": problem_ids,
            },
            "counts": {"requested": len(problem_ids), "evaluated": evaluated},
            "score": {
                "primary": primary,
                "unit": scoring["unit"],
                "metrics": metrics,
            },
            "scoring": {
                "method": scoring["method"],
                "self_judged": bool(scoring.get("self_judged", False)),
                "self_simulated": bool(scoring.get("self_simulated", False)),
            },
            "artifacts": {
                "raw": raw,
                "logs": [f"logs/{context['benchmark']}.log"],
            },
            "timestamps": {"started_at": _utc_now(), "finished_at": _utc_now()},
            "error": error,
        }
    )


def normalize_official(context: dict[str, Any], raw_path: Path) -> BenchmarkResult:
    name = context["benchmark"]
    try:
        normalizer = NORMALIZERS[name]
    except KeyError as error:
        raise OfficialNormalizationError(f"no official normalizer for {name}") from error
    observation = normalizer(raw_path)
    if not 0 <= observation.primary <= 100:
        raise OfficialNormalizationError("official primary score is outside 0..100")
    if observation.evaluated > len(observation.problem_ids):
        raise OfficialNormalizationError("official evaluated count exceeds selected problems")
    status = (
        "completed"
        if observation.evaluated == len(observation.problem_ids) and observation.error is None
        else "partial"
    )
    return _result(
        context,
        status=status,
        problem_ids=observation.problem_ids,
        evaluated=observation.evaluated,
        primary=observation.primary,
        metrics=observation.metrics,
        raw=[_artifact_path(context, raw_path)],
        error=observation.error,
    )


def unsupported_result(context: dict[str, Any], message: str) -> BenchmarkResult:
    if not message:
        raise ValueError("unsupported reason must not be empty")
    return _result(
        context,
        status="unsupported",
        problem_ids=[],
        evaluated=0,
        primary=None,
        metrics={},
        raw=[],
        error=message,
    )


def _environment() -> tuple[dict[str, Any], Path]:
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    result_path = os.environ.get("KAIRYU_BENCH_RESULT_PATH")
    if not context_path or not result_path:
        raise OfficialNormalizationError("adapter environment is incomplete")
    try:
        context = _object(json.loads(Path(context_path).read_text(encoding="utf-8")), "context")
    except (OSError, json.JSONDecodeError) as error:
        raise OfficialNormalizationError(f"cannot read adapter context: {error}") from error
    return context, Path(result_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.official")
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize")
    normalize.add_argument("raw_path", type=Path)
    unsupported = commands.add_parser("unsupported")
    unsupported.add_argument("reason")
    args = parser.parse_args(argv)
    try:
        context, result_path = _environment()
        if args.command == "normalize":
            result = normalize_official(context, args.raw_path)
        else:
            result = unsupported_result(context, args.reason)
        result.write(result_path)
    except (OfficialNormalizationError, OSError, ValueError) as error:
        print(f"kairyu-bench normalizer: {error}", file=sys.stderr)
        return 2
    return 0 if result.status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
