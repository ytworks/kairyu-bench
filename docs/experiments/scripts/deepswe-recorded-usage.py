#!/usr/bin/env python3
"""Read official DeepSWE scores and distinct-trial recorded usage, without regrading.

Canonical results determine scores. Retry archives add usage, never extra scores.
By default no trajectories are read. Opt-in agent API metadata reads only a bounded
first info object; messages, tool arguments and environment values are never output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from kairyu_bench.deepswe import _validate_trial, read_object, summarize_deepswe

METRICS = {
    "agent_steps": "n_agent_steps",
    "input_tokens": "n_input_tokens",
    "output_tokens": "n_output_tokens",
}


def numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def elapsed(start: Any, finish: Any) -> float | None:
    if not start or not finish:
        return None
    try:
        value = (
            dt.datetime.fromisoformat(finish.replace("Z", "+00:00"))
            - dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except (ValueError, TypeError, AttributeError):
        return None
    return value if numeric(value) else None


def metric(values: list[Any], expected: int) -> dict[str, Any]:
    known = [v for v in values if numeric(v)]
    return {
        "total": sum(known) if len(known) == expected else None,
        "sum_recorded": sum(known),
        "measured_trials": len(known),
        "missing_trials": expected - len(known),
        "expected_trials": expected,
    }



MAX_INFO_PREFIX_BYTES = 2 * 1024 * 1024


def file_state(path: Path) -> tuple[int, ...]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def agent_api_prefix(path: Path, raw: Path) -> dict[str, Any]:
    """Stop at the first info object's closing brace; never scan messages."""
    import hashlib

    if not path.resolve().is_relative_to(raw.resolve()):
        raise ValueError("trajectory path escapes this run")
    result: dict[str, Any] = {
        "trajectory": str(path.relative_to(raw)), "api_calls": None,
        "status": "missing_trajectory", "prefix_bytes": 0,
    }
    if not path.exists():
        return result
    before = file_state(path)
    prefix = bytearray()

    class Unavailable(Exception):
        pass

    try:
        with path.open("rb") as stream:
            def take() -> bytes:
                if len(prefix) >= MAX_INFO_PREFIX_BYTES:
                    raise Unavailable("prefix_limit")
                char = stream.read(1)
                if not char:
                    raise Unavailable("truncated_info")
                prefix.extend(char)
                return char

            def nonspace() -> bytes:
                char = take()
                while char in b" \t\r\n":
                    char = take()
                return char

            if nonspace() != b"{" or nonspace() != b'"':
                raise Unavailable("unsupported_layout")
            key = bytearray(b'"')
            escaped = False
            while True:
                char = take()
                key.extend(char)
                if char == b'"' and not escaped:
                    break
                escaped = char == b"\\" and not escaped
            if json.loads(key) != "info":
                raise Unavailable("info_not_first")
            if nonspace() != b":" or nonspace() != b"{":
                raise Unavailable("info_not_object")
            start = len(prefix) - 1
            depth, quoted, escaped = 1, False, False
            while depth:
                char = take()
                if quoted:
                    if char == b'"' and not escaped:
                        quoted = False
                    escaped = char == b"\\" and not escaped
                elif char == b'"':
                    quoted, escaped = True, False
                elif char == b"{":
                    depth += 1
                elif char == b"}":
                    depth -= 1
            info = json.loads(prefix[start:])
            stats = info.get("model_stats")
            value = stats.get("api_calls") if isinstance(stats, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise Unavailable("missing_or_invalid_api_calls")
            result.update(api_calls=value, status="recorded")
    except Unavailable as error:
        result["status"] = str(error)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result["status"] = "malformed_info"
    if file_state(path) != before:
        raise ValueError("trajectory changed during metadata read")
    result.update(
        prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        source_stat=dict(zip(("device", "inode", "size", "mtime_ns", "ctime_ns"), before)),
    )
    return result


def add_agent_api_metadata(
    raw: Path, selected: dict[str, Any], records: dict[str, Any], expected: int,
) -> dict[str, Any]:
    """Join job metadata to existing trial UUIDs; jobs never add scores or trials."""
    values: dict[str, int] = {}
    observations: dict[str, list[dict[str, Any]]] = {key: [] for key in records}
    paths = {raw / path for record in records.values() for path in record["paths"]}
    jobs = set((raw / "jobs").glob("repeat-*/*/result.json"))
    for path in sorted(paths | jobs):
        if not path.resolve().is_relative_to(raw.resolve()):
            raise ValueError("agent metadata result path escapes this run")
        before = file_state(path)
        payload = read_object(path)
        identity = payload.get("id")
        if identity not in records:
            continue
        recorded = records[identity]["measurement"]
        task, repeat = recorded["task_id"], recorded["repeat_index"]
        _validate_trial(payload, selected, task)
        if not isinstance(recorded["task_checksum"], str) or not recorded["task_checksum"]:
            raise ValueError("agent metadata join requires a task checksum")
        if payload.get("task_checksum") != recorded["task_checksum"]:
            raise ValueError("same trial UUID has conflicting task checksum")
        if path in jobs and path.relative_to(raw / "jobs").parts[0] != f"repeat-{repeat}":
            raise ValueError("same trial UUID has conflicting repeat")
        agent = payload.get("agent_result") or {}
        execution = payload.get("agent_execution") or {}
        usage = {key: agent.get(field) if numeric(agent.get(field)) else None
                 for key, field in METRICS.items()}
        if (usage != recorded["usage"]
                or execution.get("started_at") != recorded["agent_started_at"]
                or execution.get("finished_at") != recorded["agent_finished_at"]):
            raise ValueError("same trial UUID has conflicting identity/usage/time")
        trajectory = path.parent / "agent" / "mini-swe-agent.trajectory.json"
        if os.path.lexists(trajectory):
            observation = agent_api_prefix(trajectory, raw)
            observation["sibling_result"] = str(path.relative_to(raw))
            observations[identity].append(observation)
            value = observation["api_calls"]
            if value is not None:
                if identity in values and values[identity] != value:
                    raise ValueError("same trial UUID has conflicting agent API counters")
                values[identity] = value
        if file_state(path) != before:
            raise ValueError("sibling result changed during metadata read")
    for identity, record in records.items():
        record["agent_api_metadata"] = {
            "api_calls": values.get(identity), "observations": observations[identity],
            "status": "recorded" if identity in values else "unavailable",
        }
    return metric(list(values.values()), expected)

def collect(run_dir: Path, *, agent_api_metadata: bool = False) -> dict[str, Any]:
    raw = run_dir / "raw" / "deepswe"
    # Existing official validator and score semantics remain authoritative.
    official = summarize_deepswe(raw)
    selected = read_object(raw / "selected.json")
    slots = read_object(raw / "trial-plan.json")["slots"]
    allowed = {(s["repeat_index"], s["task_id"]) for s in slots}
    records: dict[str, Any] = {}
    canonical_ids: set[str] = set()

    def add(task: str, repeat: int, path: Path, archived: bool) -> None:
        if not path.resolve().is_relative_to(raw.resolve()):
            raise ValueError("trial path escapes this run")
        payload = read_object(path)
        _validate_trial(payload, selected, task)
        identity = payload.get("id")
        if not isinstance(identity, str) or str(uuid.UUID(identity)) != identity:
            raise ValueError("usage trial has no canonical UUID")
        if archived and path.parent.name != identity:
            raise ValueError("archive directory/UUID mismatch")
        agent = payload.get("agent_result") or {}
        execution = payload.get("agent_execution") or {}
        usage = {
            key: agent.get(field) if numeric(agent.get(field)) else None
            for key, field in METRICS.items()
        }
        measurement = {
            "task_id": task,
            "repeat_index": repeat,
            "task_checksum": payload.get("task_checksum"),
            "usage": usage,
            "agent_started_at": execution.get("started_at"),
            "agent_finished_at": execution.get("finished_at"),
            "agent_seconds": elapsed(
                execution.get("started_at"), execution.get("finished_at")
            ),
        }
        if identity in records:
            if records[identity]["measurement"] != measurement:
                raise ValueError("same trial UUID has conflicting identity/usage/time")
            records[identity]["paths"].append(str(path.relative_to(raw)))
        else:
            records[identity] = {
                "trial_id": identity,
                "measurement": measurement,
                "paths": [str(path.relative_to(raw))],
            }
        if not archived:
            canonical_ids.add(identity)

    for slot in slots:
        path = raw / slot["result_path"]
        if path.is_file():
            add(slot["task_id"], slot["repeat_index"], path, False)
    # Deliberately exclude jobs/ copies. Archive directory names identify slot+UUID.
    for path in sorted((raw / "retries").rglob("result.json")):
        parts = path.relative_to(raw / "retries").parts
        if len(parts) != 4:
            raise ValueError("unexpected retry result path")
        repeat_name, task, _, _ = parts
        matching = [
            (repeat, name)
            for repeat, name in allowed
            if name == task and repeat_name == f"repeat-{repeat}"
        ]
        if len(matching) != 1:
            raise ValueError("archive task/repeat differs from trial plan")
        repeat, task = matching[0]
        add(task, repeat, path, True)
    archived_only = set(records) - canonical_ids
    expected = len(slots) + len(archived_only)
    for identity, record in records.items():
        record["canonical"] = identity in canonical_ids
    metadata = read_object(run_dir / "run.json")
    report = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "official_summary": official,
        "wall_seconds": elapsed(
            metadata.get("started_at"), metadata.get("finished_at")
        ),
        "usage_scope": "canonical plus archived distinct trial UUIDs; missing canonical slots included in coverage",
        "recorded_trial_count": len(records),
        "canonical_recorded_trial_count": len(canonical_ids),
        "archived_only_trial_count": len(archived_only),
        "usage": {
            key: metric(
                [r["measurement"]["usage"][key] for r in records.values()], expected
            )
            for key in METRICS
        },
        "agent_seconds": metric(
            [r["measurement"]["agent_seconds"] for r in records.values()], expected
        ),
        "recorded_trials": [records[key] for key in sorted(records)],
        "limits": [
            "Scores and their scored denominator come only from the official canonical summarizer; unscored trials are not invented zeros.",
            "n_agent_steps is reported as agent_steps, not asserted to equal HTTP requests or tool calls.",
            "Concurrent agent durations sum across trials; wall_seconds already includes retries and is not their sum.",
            "Missing recorded usage remains null; sum_recorded is only the measured portion.",
        ],
    }

    if agent_api_metadata:
        report["usage"]["agent_api_calls"] = add_agent_api_metadata(raw, selected, records, expected)
        report["agent_api_metadata"] = {
            "field": "trajectory.info.model_stats.api_calls",
            "scope": "Agent-recorded model counter, not independent HTTP transport/server telemetry.",
            "join_keys": ["task_id", "repeat_index", "task_checksum", "trial UUID"],
            "prefix_limit_bytes": MAX_INFO_PREFIX_BYTES,
            "read_scope": "Only first top-level info object; stop before messages.",
            "steps_used_to_infer_api_calls": False,
        }
        report["limits"].append(
            "Opt-in agent API counters are metadata only; missing/invalid prefixes remain null, "
            "and distinct trial UUIDs are counted once. They do not measure independent HTTP traffic."
        )
    return report


def write_fresh(path: Path, value: dict[str, Any], run_dir: Path) -> None:
    if path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("output must be outside input run directory")
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-api-metadata", action="store_true",
                        help="Read bounded first-info trajectory counters, never messages")
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    report = collect(args.run, agent_api_metadata=args.agent_api_metadata)
    write_fresh(args.output, report, args.run)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "distinct_trials": report["recorded_trial_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
