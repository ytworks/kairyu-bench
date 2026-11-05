"""One timed evaluation of a tier and split against a served model."""

from __future__ import annotations

import functools
import json
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kairyu_bench.artifacts import write_json
from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.client import ChatClient, LogprobClient
from kairyu_bench.wsp.manifest import Item, Manifest
from kairyu_bench.wsp.report import build_report, report_markdown
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, adapters


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunSettings:
    run_dir: Path
    tier: int
    split: str
    chat: ChatClient
    logprob: LogprobClient | None
    loader: Loader
    chat_concurrency: int
    logprob_concurrency: int
    agent_concurrency: int
    label: str | None
    endpoint_fingerprint: str
    observe_config: Path | None = None


class _Recorder:
    """Appends one JSON line per finished item and counts what remains."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.stream = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.lock:
            self.stream.write(line + "\n")
            self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def allocate_workers(counts: dict[str, int], total: int, limits: dict[str, int]) -> list[dict[str, int]]:
    """Waves of ``source -> workers`` so at most ``total`` agent tasks run at once.

    Sources run together when each can have a worker; the total is split by
    task count (largest remainder, at least one each). Otherwise they run
    one after another. No source exceeds its harness's own worker setting
    or its task count.
    """

    if total < 1:
        raise ValueError("agent concurrency must be at least 1")
    sources = sorted(source for source, count in counts.items() if count > 0)
    if not sources:
        return []
    if total < len(sources):
        return [{source: max(1, min(total, limits[source], counts[source]))} for source in sources]
    tasks = sum(counts[source] for source in sources)
    shares = {source: max(1, total * counts[source] // tasks) for source in sources}
    remainders = sorted(sources, key=lambda source: (-(total * counts[source] % tasks), source))
    for source in remainders[: max(0, total - sum(shares.values()))]:
        shares[source] += 1
    return [{source: max(1, min(shares[source], limits[source], counts[source])) for source in sources}]


def _finish_later(finish: Any, item: Item, done: Future[Outcome]) -> None:
    finish(item, done)


def _ordered(items: list[Item]) -> list[Item]:
    # Longest request first so the slowest do not start last (an item's
    # samples run at once, so its length is that of one request).
    return sorted(items, key=lambda item: (-(item.expected_tokens or item.max_tokens), -item.samples, item.id))


def execute(manifest: Manifest, items: list[Item], settings: RunSettings) -> dict[str, Any]:
    registry = adapters()
    sources = sorted({item.source for item in items})
    missing = [source for source in sources if source not in registry]
    if missing:
        raise RuntimeError(f"no adapter for {missing}")
    if settings.logprob is None and any(item.kind == "logprob" for item in items):
        raise RuntimeError("logprob items need --logprob-endpoint")

    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "kind": "wsp",
        "status": "running",
        "tier": settings.tier,
        "split": settings.split,
        "label": settings.label,
        "model_id": settings.chat.model,
        "endpoint": {"fingerprint": settings.endpoint_fingerprint},
        "manifest_sha256": manifest.sha256,
        "items": len(items),
        "started_at": _utc_now(),
    }
    write_json(run_dir / "run.json", run_meta)

    started = time.monotonic()
    rows: dict[str, dict[str, Row]] = {source: registry[source].load(settings.loader) for source in sources}
    absent = [item.id for item in items if item.id not in rows[item.source]]
    if absent:
        raise RuntimeError(f"items absent from their pinned datasets: {absent[:5]}")
    loaded_s = time.monotonic() - started

    observer = None
    if settings.observe_config is not None:
        from kairyu_bench.wsp.observe.tracing import RequestRecorder

        observer = RequestRecorder(settings.observe_config, run_id=run_dir.name)
        run_meta["observation"] = {"capture_id": observer.capture_id, "output_dir": str(observer.output), "run_id": run_dir.name}
        write_json(run_dir / "run.json", run_meta)

    context = Context(
        chat=settings.chat, logprob=settings.logprob, loader=settings.loader, run_dir=run_dir,
        observe_config=settings.observe_config, run_id=run_dir.name,
    )
    outcomes: dict[str, Outcome] = {}
    errors: dict[str, str] = {}
    remaining: defaultdict[str, int] = defaultdict(int)
    for item in items:
        remaining[item.source] += 1
    finished_at: dict[str, float] = {}
    lock = threading.Lock()
    scored: dict[str, dict[str, Any]] = {}
    responses = _Recorder(run_dir / "responses.jsonl")
    score_futures: list[Future[None]] = []

    def score_source(source: str) -> None:
        adapter: Adapter = registry[source]
        entries = [
            (item, rows[source][item.id], outcomes[item.id])
            for item in items
            if item.source == source and item.id in outcomes
        ]
        try:
            results = adapter.score(entries, settings.loader) if entries else []
        except Exception:  # noqa: BLE001 - recorded per source, never a fallback score
            message = traceback.format_exc(limit=5)
            with lock:
                for item, _, _ in entries:
                    errors[item.id] = f"scoring failed: {message}"
            return
        with lock:
            for (item, _, _), result in zip(entries, results, strict=True):
                scored[item.id] = {
                    "score": result.score,
                    "sample_scores": list(result.samples),
                    "detail": result.detail,
                }

    scoring_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wsp-score")

    def finish(item: Item, future: Future[Outcome]) -> None:
        try:
            outcome = future.result()
            error = None
        except Exception as caught:  # noqa: BLE001 - an unscored item, reported as such
            outcome = None
            error = f"{type(caught).__name__}: {caught}"
        responses.write({"id": item.id, "outcome": outcome, "error": error})
        with lock:
            if outcome is None:
                errors[item.id] = error or "unknown error"
            else:
                outcomes[item.id] = outcome
            remaining[item.source] -= 1
            done = remaining[item.source] == 0
            if done:
                finished_at[item.source] = time.monotonic() - started
        if done:
            score_futures.append(scoring_pool.submit(score_source, item.source))

    def run_one(item: Item) -> Outcome:
        if observer is not None:
            with observer.item(item.id):
                return registry[item.source].execute(context, item, rows[item.source][item.id])
        return registry[item.source].execute(context, item, rows[item.source][item.id])

    def run_batch(source: str, batch_items: list[Item], workers: int) -> None:
        batch = registry[source].batch
        assert batch is not None
        try:
            found = batch(context, batch_items, rows[source], workers)
            failure = None
        except Exception as caught:  # noqa: BLE001 - every item of the batch is unscored
            found, failure = {}, f"{type(caught).__name__}: {caught}"
        for item in batch_items:
            done: Future[Outcome] = Future()
            if item.id in found:
                done.set_result(found[item.id])
            else:
                done.set_exception(RuntimeError(failure or "no official result for this task"))
            finish(item, done)

    pools = {
        "agent": ThreadPoolExecutor(settings.agent_concurrency, thread_name_prefix="wsp-agent"),
        "chat": ThreadPoolExecutor(settings.chat_concurrency, thread_name_prefix="wsp-chat"),
        "logprob": ThreadPoolExecutor(settings.logprob_concurrency, thread_name_prefix="wsp-logprob"),
    }
    try:
        batched = sorted({item.source for item in items if registry[item.source].batch is not None})
        waves = allocate_workers(
            {source: sum(1 for item in items if item.source == source) for source in batched},
            settings.agent_concurrency,
            {source: registry[source].worker_limit() for source in batched},  # type: ignore[misc]
        )

        def run_waves() -> None:
            for wave in waves:
                with ThreadPoolExecutor(len(wave), thread_name_prefix="wsp-batch") as batches:
                    for source, workers in wave.items():
                        batches.submit(
                            run_batch, source, [item for item in items if item.source == source], workers
                        )

        pools["agent"].submit(run_waves)
        for kind in ("agent", "chat", "logprob"):
            for item in _ordered([entry for entry in items if entry.kind == kind and entry.source not in batched]):
                submitted = pools[kind].submit(run_one, item)
                submitted.add_done_callback(functools.partial(_finish_later, finish, item))
        # Done-callbacks run in the worker threads, so after the shutdowns
        # every source's scoring has been submitted.
        for pool in pools.values():
            pool.shutdown(wait=True)
        generation_s = time.monotonic() - started
        for scoring in score_futures:
            scoring.result()
        scoring_pool.shutdown(wait=True)
    finally:
        responses.close()
    wall_s = time.monotonic() - started

    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as stream:
        for item in items:
            record = {
                "id": item.id,
                "source": item.source,
                "area": SOURCE_BY_NAME[item.source].area,
                "kind": item.kind,
                "tier": item.tier,
                "max_tokens": item.max_tokens,
                "samples": item.samples,
                "outcome": outcomes.get(item.id),
                "error": errors.get(item.id),
                **scored.get(item.id, {"score": None, "sample_scores": [], "detail": {}}),
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    timing = {
        "load_seconds": round(loaded_s, 1),
        "generation_seconds": round(generation_s, 1),
        "wall_seconds": round(wall_s, 1),
        "source_finished_seconds": {key: round(value, 1) for key, value in sorted(finished_at.items())},
        "target_minutes": manifest.tier_minutes[settings.tier],
        "chat_api_calls": settings.chat.calls,
        "logprob_api_calls": settings.logprob.calls if settings.logprob else 0,
    }
    unscored = sorted(set(errors))
    run_meta.update(
        {
            "status": "completed" if not unscored else "partial",
            "finished_at": _utc_now(),
            "timing": timing,
            "unscored": unscored,
        }
    )
    write_json(run_dir / "run.json", run_meta)
    report = build_report(run_dir)
    write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return run_meta
