#!/usr/bin/env python3
"""Read official DeepSWE scores and distinct-trial recorded usage, without regrading.

Canonical results determine scores. Retry archives add usage, never extra scores.
No trajectories, prompt/answer text, tool arguments, or environment values are read.
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


def collect(run_dir: Path) -> dict[str, Any]:
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
    return {
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
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    report = collect(args.run)
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
