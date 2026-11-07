"""Cross-process limits for observation records; no inference data is retained."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any


def allocations(total: int, workers: int, replicas: int) -> dict[str, int]:
    if type(total) is not int or total < 32 or workers < 1 or replicas < 1:
        raise ValueError("invalid capture storage budget")
    return {
        "worker_events": (total * 7 // 8) // workers,
        "worker_metadata": (total // 32) // workers,
        "global_metadata": total // 32,
        "client_events": total // 32,
        "server_events": (total // 32) // replicas,
    }


def append_bounded_jsonl(path: Path, record: dict[str, Any], max_bytes: int) -> bool:
    """Append one complete record under a file lock shared by threads/processes.

    False means the quota was exhausted. The zero-byte marker makes a truncated
    capture visible to merge even if no final process callback is run.
    """
    line = (json.dumps(record, allow_nan=False, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode()
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            stream.seek(0, 2)
            size = stream.tell()
            if size + len(line) > max_bytes:
                path.with_suffix(".quota-exhausted").touch(exist_ok=True)
                return False
            try:
                stream.write(line)
                stream.flush()
            except OSError:
                stream.truncate(size)
                raise
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    return True
