"""Capture configuration and bounded, benchmark-linked report preparation."""

from __future__ import annotations

import gzip
import json
import math
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from kairyu_bench.artifacts import read_object, write_json

_REQUEST = re.compile(r"observe-([0-9a-f]{32})-([0-9a-f]{32})")


def initialize_capture(
    checkpoint: Path, output_dir: Path, *, model: str,
    tp_size: int = 2, replicas: int = 4, mtp_tokens: int = 4,
    mode: str = "live",
) -> dict[str, Any]:
    from kairyu_bench.wsp.weights.checkpoint import Checkpoint

    if tp_size < 1 or replicas < 1 or mtp_tokens < 0 or mode not in {"live", "replay", "teacher_forced"}:
        raise ValueError("invalid topology or observation mode")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("capture output directory must be new or empty")
    identity = Checkpoint(checkpoint).identity()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1, "capture_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "observation_mode": mode, "data_kind": "recorded", "output_dir": str(output_dir),
        "checkpoint": identity,
        "runtime": {
            "adapter": "vllm-0.33", "model": model, "tp_size": tp_size,
            "replicas": replicas, "mtp_tokens": mtp_tokens,
        },
        "limits": {
            "gpu_buffer_mib": 256, "storage_bytes": 2 * 1024**3,
            "prefill_positions": 32, "decode_positions": 32,
        },
    }
    write_json(output_dir / "capture.json", document)
    return document


def json_lines(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"invalid JSON record at {path.name}:{number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"expected object at {path.name}:{number}")
                yield record
    except (EOFError, gzip.BadGzipFile) as error:
        raise ValueError(f"incomplete or invalid compressed capture: {path.name}; wait for writer drain") from error



def observation_records(path: Path, sites: dict[str, dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Expand compressed columnar batches lazily, retaining old JSONL compatibility."""
    fields = ["request_id", "token_position", "phase", "speculative_index", "accepted"]
    for batch in json_lines(path):
        if "record_type" not in batch:
            yield batch
            continue
        if batch["record_type"] != "activation_batch" or batch.get("row_fields") != fields:
            raise ValueError("unsupported activation batch layout")
        ids, names = batch.get("site_ids"), batch.get("metric_names")
        rows, values = batch.get("rows"), batch.get("values")
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids) or any(x not in sites for x in ids)
                or not isinstance(names, list) or not names or any(not isinstance(x, str) for x in names)
                or len(set(names)) != len(names) or not isinstance(rows, list)
                or not isinstance(values, list) or len(values) != len(ids)):
            raise ValueError("invalid activation batch axes")
        for row in rows:
            if (not isinstance(row, list) or len(row) != len(fields)
                    or not isinstance(row[0], str) or type(row[1]) is not int
                    or row[2] not in {"prefill", "decode"}
                    or (row[3] is not None and type(row[3]) is not int)
                    or (row[4] is not None and type(row[4]) is not bool)):
                raise ValueError("invalid activation batch row")
        common = {name: batch.get(name) for name in (
            "schema_version", "capture_id", "observation_mode", "replica", "rank", "step", "sample",
        )}
        for site_id, site_values in zip(ids, values, strict=True):
            site = sites[site_id]
            if not isinstance(site_values, list) or len(site_values) != len(rows):
                raise ValueError("activation batch values differ from row layout")
            for row, metrics in zip(rows, site_values, strict=True):
                if (not isinstance(metrics, list) or len(metrics) != len(names)
                        or any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
                               for value in metrics)):
                    raise ValueError("invalid activation batch metric vector")
                metrics_map = dict(zip(names, metrics, strict=True))
                if site.get("detail_kind") == "channel":
                    metrics_map["value"] = metrics_map.get("mean")
                yield {
                    **common, **dict(zip(fields, row, strict=True)),
                    **{name: site.get(name) for name in (
                        "site_id", "layer", "role", "detail_kind", "parent_site_id", "channel_index",
                        "head_local_index", "head_global_index", "head_dim", "activation_axis",
                    )},
                    "token_kind": "target", "metrics": metrics_map,
                }


def _task(item: dict[str, Any]) -> dict[str, Any]:
    from kairyu_bench.wsp.catalog import SOURCE_BY_NAME

    source = item.get("source", "unknown")
    spec = SOURCE_BY_NAME.get(source)
    scored = item.get("score")
    if isinstance(scored, dict):
        scored = scored.get("score")
    if scored is None:
        scored = (item.get("scored") or {}).get("score")
    outcome = item.get("outcome") or {}
    samples = outcome.get("samples") if isinstance(outcome, dict) else None
    # Only use one authoritative sample's duration/steps. Summing parallel
    # samples would silently turn elapsed time into aggregate work.
    sample = samples[0] if isinstance(samples, list) and len(samples) == 1 and isinstance(samples[0], dict) else {}
    return {
        "id": item["id"], "source": source, "area": spec.area if spec else "unknown",
        "score": scored, "status": "unscored" if scored is None else "scored",
        "seconds": item.get("seconds", sample.get("seconds")),
        "steps": item.get("steps", sample.get("steps")),
        "observations": [],
    }


def merge_capture(
    directory: Path, *, results: Path | None = None, token_bin_size: int = 64,
    task_filter: str | None = None,
) -> dict[str, Any]:
    if token_bin_size < 1:
        raise ValueError("token_bin_size must be positive")
    config = read_object(directory / "capture.json", "capture")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported capture schema")
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("requests-*.jsonl")):
        for request in json_lines(path):
            if request.get("capture_id") != config["capture_id"]:
                raise ValueError("request belongs to another capture")
            previous = traces.setdefault(request["trace_id"], request)
            if previous != request:
                raise ValueError("one trace ID maps to multiple requests")
    tasks: dict[str, dict[str, Any]] = {}
    run_meta: dict[str, Any] = {}
    if results is not None:
        run_meta = read_object(results / "run.json", "benchmark run")
        if (run_meta.get("observation") or {}).get("capture_id") != config["capture_id"]:
            raise ValueError("benchmark run does not reference this capture ID")
        for item in json_lines(results / "items.jsonl"):
            if item["id"] in tasks:
                raise ValueError("duplicate benchmark item ID")
            tasks[item["id"]] = _task(item)
    def verify_capture(record: dict[str, Any]) -> None:
        if record.get("capture_id") != config["capture_id"]:
            raise ValueError("worker record belongs to another capture")
        if record.get("observation_mode", config["observation_mode"]) != config["observation_mode"]:
            raise ValueError("worker observation mode differs from capture")

    acceptance: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(directory.glob("workers/*/acceptance.jsonl")):
        for record in json_lines(path):
            verify_capture(record)
            scheduled = record.get("scheduled_draft_tokens")
            accepted_count = record.get("accepted_draft_tokens")
            if (type(scheduled) is not int or type(accepted_count) is not int
                    or not 0 <= accepted_count <= scheduled):
                raise ValueError("invalid speculative acceptance counts")
            acceptance_key = (str(record.get("replica")), record.get("step"), record.get("request_id"))
            prior = acceptance.setdefault(acceptance_key, record)
            if prior != record:
                raise ValueError("conflicting speculative acceptance records")
    sites: dict[tuple[str, str], dict[str, Any]] = {}
    statuses = []
    worker_ids: set[tuple[str, int]] = set()
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    count = 0
    unlinked = 0
    excluded_other_runs = 0
    requested_run = (run_meta.get("observation") or {}).get("run_id")
    if results is not None and requested_run is None and any(t.get("run_id") is not None for t in traces.values()):
        raise ValueError("benchmark run lacks run_id required by this capture")
    for worker in sorted((directory / "workers").glob("*")):
        status_path = worker / "status.json"
        if status_path.is_file():
            status = read_object(status_path, "worker status")
            verify_capture(status)
            worker_id = (str(status["replica"]), int(status["rank"]))
            if worker_id in worker_ids or not 0 <= worker_id[1] < int(config["runtime"]["tp_size"]):
                raise ValueError("duplicate worker or invalid TP rank")
            worker_ids.add(worker_id)
            statuses.append(status)
        worker_sites: dict[str, dict[str, Any]] = {}
        sites_path = worker / "sites.json"
        if sites_path.is_file():
            document = json.loads(sites_path.read_text())
            if not isinstance(document, dict):
                raise ValueError("site manifest must be an object")
            verify_capture(document)
            if document.get("checkpoint") != config["checkpoint"] or document.get("runtime") != config["runtime"]:
                raise ValueError("worker checkpoint/runtime differs from capture")
            rows = document.get("sites", [])
            for site in rows:
                if site["site_id"] in worker_sites:
                    raise ValueError("duplicate site in worker manifest")
                worker_sites[site["site_id"]] = site
                site_key = (str(site["site_id"]), str(site.get("tp_layout", "unknown")))
                previous_site = sites.setdefault(site_key, site)
                if previous_site != site:
                    raise ValueError("conflicting site definitions")
        paths = [path for path in (worker / "observations.jsonl", worker / "observations.jsonl.gz") if path.is_file()]
        if len(paths) > 1:
            raise ValueError("worker has both legacy and compact observations")
        if not paths:
            continue
        for observation in observation_records(paths[0], worker_sites):
            verify_capture(observation)
            if status_path.is_file() and (str(observation["replica"]), int(observation["rank"])) != worker_id:
                raise ValueError("observation worker differs from its status")
            count += 1
            request_id = str(observation["request_id"])
            matched = _REQUEST.search(request_id)
            trace = traces.get(matched.group(1)) if matched else None
            if requested_run is not None and trace is not None and trace.get("run_id") != requested_run:
                excluded_other_runs += 1
                continue
            if trace is None:
                unlinked += 1
            item_id = str(trace["item_id"]) if trace else f"unattributed/{request_id}"
            if task_filter is not None and item_id != task_filter:
                continue
            tasks.setdefault(item_id, _task({"id": item_id, "source": item_id.split(":", 1)[0]}))
            position = int(observation["token_position"])
            if position < 0:
                raise ValueError("negative token position")
            accepted = observation.get("accepted")
            acceptance_record = acceptance.get((
                str(observation.get("replica")), observation.get("step"), request_id,
            ))
            speculative_index = observation.get("speculative_index")
            if speculative_index is not None:
                if type(speculative_index) is not int or speculative_index < 0:
                    raise ValueError("invalid speculative token index")
                if acceptance_record is not None and speculative_index >= acceptance_record["scheduled_draft_tokens"]:
                    raise ValueError("speculative index outside scheduled draft prefix")
            if accepted is None and acceptance_record is not None and speculative_index is not None:
                accepted = int(speculative_index) < int(acceptance_record["accepted_draft_tokens"])
            key = (
                item_id, request_id, observation["site_id"], observation.get("phase"),
                observation.get("token_kind"), accepted, observation.get("replica"),
                observation.get("rank"), position // token_bin_size,
            )
            if key not in cells:
                cells[key] = {
                    **{name: observation.get(name) for name in (
                        "site_id", "layer", "role", "phase", "token_kind", "replica", "rank",
                        "detail_kind", "parent_site_id", "channel_index", "head_local_index",
                        "head_global_index", "head_dim", "activation_axis",
                    )},
                    "request_id": request_id,
                    "call_id": trace.get("call_id") if trace else None,
                    "request_scope": trace.get("request_scope", "request") if trace else None,
                    "trial_id": trace.get("trial_id") if trace else None,
                    "token_position": position // token_bin_size * token_bin_size,
                    "token_end": position, "accepted": accepted, "sample_count": 0,
                    "metrics": {}, "_sum": defaultdict(float), "metric_sample_counts": defaultdict(int),
                }
            cell = cells[key]
            n = int(observation.get("sample_count", 1))
            if n < 1:
                raise ValueError("sample_count must be positive")
            metrics = observation["metrics"]
            for name, value in metrics.items():
                if value is None:
                    continue
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("nonfinite serialized activation metric")
                cell["metric_sample_counts"][name] += n
                if name == "max_abs":
                    cell["metrics"][name] = max(cell["metrics"].get(name, 0.0), value)
                elif name == "nonfinite_count":
                    cell["metrics"][name] = cell["metrics"].get(name, 0) + int(value)
                elif name == "rms":
                    cell["_sum"][name] += value * value * n
                else:
                    cell["_sum"][name] += value * n
            cell["sample_count"] += n
            cell["token_end"] = max(cell["token_end"], position)
    for key, cell in cells.items():
        for name, value in cell.pop("_sum").items():
            average = value / cell["metric_sample_counts"][name]
            cell["metrics"][name] = math.sqrt(average) if name == "rms" else average
        tasks[key[0]]["observations"].append(cell)
    if task_filter is not None:
        tasks = {key: value for key, value in tasks.items() if key == task_filter}
    expected = int(config["runtime"]["tp_size"]) * int(config["runtime"]["replicas"])
    quota_exhausted = [path.name for path in directory.glob("*.quota-exhausted")]
    complete = (
        not quota_exhausted
        and len(statuses) == expected
        and len({replica for replica, rank in worker_ids}) == int(config["runtime"]["replicas"])
        and all(status.get("closed") for status in statuses)
        and not any(status.get("errors") or status.get("dropped_records") for status in statuses)
        and unlinked == 0
    )
    run = {
        "id": f"{config['capture_id']}/{requested_run}" if requested_run else config["capture_id"], "label": run_meta.get("label") or config["runtime"].get("model"),
        "data_kind": config.get("data_kind", "unknown"), "observation_mode": config["observation_mode"],
        "checkpoint": config["checkpoint"], "runtime": config["runtime"],
        "input_kind": config.get("input_kind", "unspecified"),
        "capture": {
            "status": "complete" if complete else "partial_or_running",
            "limits": config["limits"], "token_bin_size": token_bin_size,
            "counts": {"records": count, "unlinked_records": unlinked,
                       "workers": len(statuses), "expected_workers": expected,
                       "excluded_other_run_records": excluded_other_runs},
            "workers": statuses, "quota_exhausted": quota_exhausted,
        },
        "sites": list(sites.values()), "tasks": list(tasks.values()),
    }
    return {"schema_version": 1, "data_kind": config.get("data_kind", "unknown"), "runs": [run]}


def compare_reports(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if base.get("schema_version") != 1 or candidate.get("schema_version") != 1:
        raise ValueError("unsupported report schema")
    runs = [*base["runs"], *candidate["runs"]]
    identifiers = [run["id"] for run in runs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("comparison contains duplicate run IDs")
    return {"schema_version": 1, "runs": runs}
