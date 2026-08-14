from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from kairyu_bench.results import BenchmarkResult, ResultValidationError
from kairyu_bench.reporting import write_score_report
from kairyu_bench.target import Endpoint


class ModelDiscovery(Protocol):
    def discover_chat_model(self) -> str:
        ...

    def discover_embedding_model(self) -> str:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def endpoint_fingerprint(endpoint: Endpoint) -> str:
    digest = hashlib.sha256(endpoint.base_url.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


@dataclass(frozen=True)
class RunConfig:
    endpoint: Endpoint
    selected: tuple[str, ...]
    limit: int | None
    results_root: Path
    run_id: str
    app_root: Path
    adapter_timeout: float | None = None


@dataclass(frozen=True)
class RunOutcome:
    run_dir: Path
    model_id: str
    embedding_model_id: str | None
    exit_code: int
    statuses: dict[str, str]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _failure_result(
    *,
    config: RunConfig,
    model_id: str,
    name: str,
    entry: dict[str, Any],
    started_at: str,
    error: str,
    conditions: dict[str, Any],
) -> BenchmarkResult:
    scoring = entry["scoring"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "benchmark": name,
        "status": "failed",
        "endpoint": {"fingerprint": endpoint_fingerprint(config.endpoint)},
        "model_id": model_id,
        "source": {
            "repository": entry["source"]["repository"],
            "revision": entry["source"]["revision"],
            "dataset": entry["dataset"]["id"],
            "dataset_revision": entry["dataset"]["revision"],
        },
        "selection": {
            "requested_limit": config.limit,
            "problem_ids": [],
        },
        "counts": {"requested": 0, "evaluated": 0},
        "score": {
            "primary": None,
            "unit": scoring["unit"],
            "metrics": {},
        },
        "scoring": {
            "method": scoring["method"],
            "self_judged": bool(scoring.get("self_judged", False)),
            "self_simulated": bool(scoring.get("self_simulated", False)),
        },
        "artifacts": {"raw": [], "logs": [f"logs/{name}.log"]},
        "timestamps": {"started_at": started_at, "finished_at": _utc_now()},
        "error": error,
    }
    if conditions:
        payload["conditions"] = conditions
    return BenchmarkResult.from_dict(payload)


def _validate_identity(
    result: BenchmarkResult,
    config: RunConfig,
    model_id: str,
    name: str,
    entry: dict[str, Any],
    conditions: dict[str, Any],
) -> None:
    data = result.data
    expected = {
        "run_id": config.run_id,
        "benchmark": name,
        "model_id": model_id,
    }
    for field, value in expected.items():
        if data[field] != value:
            raise ResultValidationError(
                f"adapter result {field} is {data[field]!r}, expected {value!r}"
            )
    if data["endpoint"]["fingerprint"] != endpoint_fingerprint(config.endpoint):
        raise ResultValidationError("adapter result endpoint fingerprint changed")
    expected_source = {
        "repository": entry["source"]["repository"],
        "revision": entry["source"]["revision"],
        "dataset": entry["dataset"]["id"],
        "dataset_revision": entry["dataset"]["revision"],
    }
    if data["source"] != expected_source:
        raise ResultValidationError("adapter result source lock differs from manifest")
    scoring = entry["scoring"]
    expected_scoring = {
        "method": scoring["method"],
        "self_judged": bool(scoring.get("self_judged", False)),
        "self_simulated": bool(scoring.get("self_simulated", False)),
    }
    if data["scoring"] != expected_scoring:
        raise ResultValidationError("adapter result scoring method differs from manifest")
    if data.get("conditions", {}) != conditions:
        raise ResultValidationError("adapter result benchmark conditions changed")


def run_benchmarks(
    config: RunConfig,
    client: ModelDiscovery,
    manifest: dict[str, dict[str, Any]],
) -> RunOutcome:
    run_dir = config.results_root / config.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {config.run_id}")

    model_id = client.discover_chat_model()
    requires_embeddings = any(
        "embeddings" in manifest[name].get("requirements", [])
        for name in config.selected
    )
    embedding_model_id = (
        client.discover_embedding_model() if requires_embeddings else None
    )
    run_dir.mkdir(parents=True)
    for child in ("context", "logs", "normalized", "raw"):
        (run_dir / child).mkdir()

    started_at = _utc_now()
    fingerprint = endpoint_fingerprint(config.endpoint)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "status": "running",
        "endpoint": {"fingerprint": fingerprint},
        "model_id": model_id,
        "embedding_model_id": embedding_model_id,
        "benchmarks": list(config.selected),
        "limit": config.limit,
        "started_at": started_at,
        "finished_at": None,
        "statuses": {},
    }
    _write_json(run_dir / "run.json", metadata)

    statuses: dict[str, str] = {}
    for name in config.selected:
        entry = manifest[name]
        adapter_started_at = _utc_now()
        result_path = run_dir / "normalized" / f"{name}.json"
        log_path = run_dir / "logs" / f"{name}.log"
        context_path = run_dir / "context" / f"{name}.json"
        conditions = (
            {"embedding_model_id": embedding_model_id}
            if "embeddings" in entry.get("requirements", [])
            else {}
        )
        context = {
            "schema_version": 1,
            "run_id": config.run_id,
            "benchmark": name,
            "endpoint": config.endpoint.base_url,
            "endpoint_fingerprint": fingerprint,
            "model_id": model_id,
            "limit": config.limit,
            "source": entry["source"],
            "generator": entry.get("generator"),
            "dataset": entry["dataset"],
            "secondary_sources": entry.get("secondary_sources", []),
            "scoring": entry["scoring"],
            "run_dir": str(run_dir),
            "result_path": str(result_path),
        }
        if conditions:
            context["conditions"] = conditions
        _write_json(context_path, context)
        adapter_path = config.app_root / entry["adapter"]
        error: str | None = None
        return_code: int | None = None
        if not adapter_path.is_file():
            error = f"adapter executable not found: {entry['adapter']}"
        elif not os.access(adapter_path, os.X_OK):
            error = f"adapter is not executable: {entry['adapter']}"
        else:
            env = os.environ.copy()
            api_key = env.get("KAIRYU_API_KEY", "")
            env.pop("KAIRYU_EMBEDDING_MODEL", None)
            env.update(
                {
                    "KAIRYU_BENCH_CONTEXT": str(context_path),
                    "KAIRYU_BENCH_RESULT_PATH": str(result_path),
                    "KAIRYU_BENCH_RUN_DIR": str(run_dir),
                    "KAIRYU_BENCH_CACHE_DIR": env.get(
                        "KAIRYU_BENCH_CACHE_DIR", "/work/cache"
                    ),
                    "KAIRYU_ENDPOINT": config.endpoint.base_url,
                    "KAIRYU_MODEL": model_id,
                    "KAIRYU_BENCH_LIMIT": "" if config.limit is None else str(config.limit),
                    "OPENAI_BASE_URL": config.endpoint.base_url,
                    "OPENAI_API_BASE": config.endpoint.base_url,
                    "OPENAI_API_KEY": api_key or "not-required",
                }
            )
            if conditions and embedding_model_id is not None:
                env["KAIRYU_EMBEDDING_MODEL"] = embedding_model_id
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        [str(adapter_path)],
                        cwd=config.app_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=config.adapter_timeout,
                        check=False,
                    )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                error = "adapter timed out"
            except OSError as exception:
                error = f"adapter could not start: {exception}"

        result: BenchmarkResult | None = None
        if error is None and result_path.is_file():
            try:
                decoded = json.loads(result_path.read_text(encoding="utf-8"))
                result = BenchmarkResult.from_dict(decoded)
                _validate_identity(
                    result,
                    config,
                    model_id,
                    name,
                    entry,
                    conditions,
                )
                if return_code and result.status == "completed":
                    raise ResultValidationError(
                        f"adapter exited {return_code} but claimed completion"
                    )
            except (json.JSONDecodeError, ResultValidationError) as exception:
                invalid_path = run_dir / "raw" / f"invalid-{name}.json"
                invalid_path.write_text(
                    result_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                error = f"invalid adapter result: {exception}"
                result = None
        elif error is None:
            error = f"adapter exited {return_code} without writing a result"

        if result is None:
            result = _failure_result(
                config=config,
                model_id=model_id,
                name=name,
                entry=entry,
                started_at=adapter_started_at,
                error=error or f"adapter exited {return_code}",
                conditions=conditions,
            )
            result.write(result_path)
        statuses[name] = result.status
        metadata["statuses"] = dict(statuses)
        _write_json(run_dir / "run.json", metadata)

    all_completed = all(status == "completed" for status in statuses.values())
    metadata["status"] = "completed" if all_completed else "incomplete"
    metadata["finished_at"] = _utc_now()
    _write_json(run_dir / "run.json", metadata)
    write_score_report(run_dir)
    return RunOutcome(
        run_dir=run_dir,
        model_id=model_id,
        embedding_model_id=embedding_model_id,
        exit_code=0 if all_completed else 3,
        statuses=statuses,
    )
