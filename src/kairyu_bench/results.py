from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairyu_bench.benchmarks import BENCHMARK_NAMES


STATUSES = {"completed", "partial", "skipped", "failed", "unsupported"}


class ResultValidationError(ValueError):
    """A normalized benchmark result violates the public result contract."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{field} must be an object")
    return value


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResultValidationError(f"{field} must be a non-empty string")
    return value


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultValidationError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class BenchmarkResult:
    data: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BenchmarkResult":
        if not isinstance(payload, dict):
            raise ResultValidationError("result must be an object")
        if payload.get("schema_version") != 1:
            raise ResultValidationError("schema_version must be 1")
        _non_empty_string(payload.get("run_id"), "run_id")
        benchmark = _non_empty_string(payload.get("benchmark"), "benchmark")
        if benchmark not in BENCHMARK_NAMES:
            raise ResultValidationError(f"unknown benchmark: {benchmark}")
        status = _non_empty_string(payload.get("status"), "status")
        if status not in STATUSES:
            raise ResultValidationError(f"unknown status: {status}")
        _non_empty_string(payload.get("model_id"), "model_id")
        if payload.get("agent") is not None:
            _non_empty_string(payload.get("agent"), "agent")

        endpoint = _mapping(payload.get("endpoint"), "endpoint")
        _non_empty_string(endpoint.get("fingerprint"), "endpoint.fingerprint")

        source = _mapping(payload.get("source"), "source")
        for field in ("repository", "revision", "dataset", "dataset_revision"):
            _non_empty_string(source.get(field), f"source.{field}")

        selection = _mapping(payload.get("selection"), "selection")
        requested_limit = selection.get("requested_limit")
        if requested_limit is not None:
            if (
                isinstance(requested_limit, bool)
                or not isinstance(requested_limit, int)
                or requested_limit <= 0
            ):
                raise ResultValidationError(
                    "selection.requested_limit must be null or a positive integer"
                )
        problem_ids = selection.get("problem_ids")
        if not isinstance(problem_ids, list) or not all(
            isinstance(problem_id, str) and problem_id for problem_id in problem_ids
        ):
            raise ResultValidationError(
                "selection.problem_ids must be a list of non-empty strings"
            )
        if len(problem_ids) != len(set(problem_ids)):
            raise ResultValidationError("selection.problem_ids contains a duplicate")

        counts = _mapping(payload.get("counts"), "counts")
        requested = _non_negative_integer(counts.get("requested"), "counts.requested")
        evaluated = _non_negative_integer(counts.get("evaluated"), "counts.evaluated")
        if requested != len(problem_ids):
            raise ResultValidationError(
                "counts.requested must equal the number of selected problem IDs"
            )
        if evaluated > requested:
            raise ResultValidationError(
                "counts.evaluated cannot exceed counts.requested"
            )

        score = _mapping(payload.get("score"), "score")
        primary = score.get("primary")
        if primary is not None and (
            isinstance(primary, bool) or not isinstance(primary, (int, float))
        ):
            raise ResultValidationError("score.primary must be null or numeric")
        unit = _non_empty_string(score.get("unit"), "score.unit")
        _mapping(score.get("metrics"), "score.metrics")
        if unit == "percent" and primary is not None and not 0 <= primary <= 100:
            raise ResultValidationError(
                "percent primary score must be between 0 and 100"
            )
        if unit == "proportion" and primary is not None and not 0 <= primary <= 1:
            raise ResultValidationError(
                "proportion primary score must be between 0 and 1"
            )

        scoring = _mapping(payload.get("scoring"), "scoring")
        _non_empty_string(scoring.get("method"), "scoring.method")
        for field in ("self_judged", "self_simulated"):
            if not isinstance(scoring.get(field), bool):
                raise ResultValidationError(f"scoring.{field} must be a boolean")

        if "conditions" in payload:
            conditions = _mapping(payload.get("conditions"), "conditions")
            embedding_model_id = conditions.get("embedding_model_id")
            if embedding_model_id is not None:
                _non_empty_string(
                    embedding_model_id,
                    "conditions.embedding_model_id",
                )

        artifacts = _mapping(payload.get("artifacts"), "artifacts")
        for field in ("raw", "logs"):
            paths = artifacts.get(field)
            if not isinstance(paths, list) or not all(
                isinstance(path, str) and path for path in paths
            ):
                raise ResultValidationError(
                    f"artifacts.{field} must be a list of non-empty paths"
                )

        timestamps = _mapping(payload.get("timestamps"), "timestamps")
        _non_empty_string(timestamps.get("started_at"), "timestamps.started_at")
        _non_empty_string(timestamps.get("finished_at"), "timestamps.finished_at")
        error = payload.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            raise ResultValidationError("error must be null or a non-empty string")

        if status == "completed":
            if primary is None:
                raise ResultValidationError("completed result requires a primary score")
            if evaluated != requested:
                raise ResultValidationError(
                    "completed result must evaluate all selected problems"
                )
            if error is not None:
                raise ResultValidationError("completed result cannot contain an error")

        stable_data = json.loads(json.dumps(payload, allow_nan=False))
        return cls(stable_data)

    @property
    def benchmark(self) -> str:
        return self.data["benchmark"]

    @property
    def status(self) -> str:
        return self.data["status"]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
