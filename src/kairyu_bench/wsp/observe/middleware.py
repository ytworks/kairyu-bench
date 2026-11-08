"""vLLM --middleware adapter linking W3C traces to engine request IDs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.observe.budget import allocations, append_bounded_jsonl

_TRACE = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}\Z")


class CaptureMiddleware:
    def __init__(self, app: Any):
        self.app = app
        config_path = os.environ.get("KAIRYU_BENCH_CAPTURE_CONFIG")
        if not config_path:
            raise ValueError("KAIRYU_BENCH_CAPTURE_CONFIG is required for CaptureMiddleware")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config.get("schema_version") != 1:
            raise ValueError("unsupported capture schema")
        directory = Path(config["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        self.capture_id = config["capture_id"]
        self.replica = os.environ.get("KAIRYU_BENCH_CAPTURE_REPLICA", os.uname().nodename)
        suffix = hashlib.sha256(self.replica.encode()).hexdigest()[:16]
        self.path = directory / f"server-requests-{suffix}.jsonl"
        self.quota = allocations(config["limits"]["storage_bytes"], 1,
                                 config["runtime"]["replicas"])["server_events"]
        self.write_errors = 0
        self.lock = threading.Lock()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") not in {"/v1/chat/completions", "/v1/completions"}:
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers", []))
        incoming = {key.lower(): value for key, value in headers}
        match = _TRACE.fullmatch(incoming.get(b"traceparent", b"").decode("ascii", "replace"))
        trace_id = match.group(1) if match and int(match.group(1), 16) else None
        request_id = f"observe-{trace_id or 'unattributed'}-{uuid.uuid4().hex}"
        scope = dict(scope)
        scope["headers"] = [(key, value) for key, value in headers if key.lower() != b"x-request-id"]
        scope["headers"].append((b"x-request-id", request_id.encode("ascii")))
        started = time.monotonic()
        status = None

        async def record_status(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, record_status)
        finally:
            record = {
                "schema_version": 1, "capture_id": self.capture_id,
                "request_id": request_id, "trace_id": trace_id,
                "replica": self.replica, "status": status,
                "seconds": time.monotonic() - started,
            }
            with self.lock:
                try:
                    if not append_bounded_jsonl(self.path, record, self.quota):
                        self.write_errors += 1
                except OSError:
                    self.write_errors += 1
