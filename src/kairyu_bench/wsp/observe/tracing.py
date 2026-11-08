"""Request provenance for optional activation capture, without storing prompts."""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from kairyu_bench.wsp.observe.budget import allocations, append_bounded_jsonl

_scope: contextvars.ContextVar[tuple["RequestRecorder", str] | None] = contextvars.ContextVar(
    "wsp_observation_scope", default=None,
)


class RequestRecorder:
    def __init__(self, config: Path, run_id: str | None = None):
        document = json.loads(config.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("unsupported capture schema")
        self.output = Path(document["output_dir"])
        if not self.output.is_absolute() or not self.output.is_dir():
            raise ValueError("capture output_dir must be an existing absolute directory")
        if self.output.resolve() != config.resolve().parent:
            raise ValueError("capture config must reside inside output_dir")
        self.capture_id = str(document["capture_id"])
        self.run_id = run_id
        self.path = self.output / "requests-client.jsonl"
        self.quota = allocations(document["limits"]["storage_bytes"], 1, 1)["client_events"]
        self.lock = threading.Lock()
        self.write_errors = 0

    def write(self, record: dict[str, Any]) -> None:
        with self.lock:
            try:
                if not append_bounded_jsonl(self.path, record, self.quota):
                    self.write_errors += 1
            except OSError:
                self.write_errors += 1

    def register_trial(
        self, item_id: str, trial_id: str, attempt_index: int | None = 0,
    ) -> dict[str, str]:
        """Bind all HTTP requests in an official agent trial to one benchmark item.

        Trial tracing does not pretend to observe individual agent HTTP attempts.
        The serving middleware still assigns each backend call a unique request ID.
        """
        trace_id, call_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.write({
            "schema_version": 1, "capture_id": self.capture_id, "run_id": self.run_id,
            "item_id": item_id, "trial_id": trial_id, "request_scope": "trial",
            "trace_id": trace_id, "call_id": call_id, "attempt_index": attempt_index,
            "started_at_unix": time.time(),
        })
        return {"traceparent": f"00-{trace_id}-{call_id[:16]}-01"}

    @contextmanager
    def item(self, item_id: str) -> Iterator[None]:
        token = _scope.set((self, item_id))
        try:
            yield
        finally:
            _scope.reset(token)


class RequestTrace:
    def __init__(self, recorder: RequestRecorder, item_id: str, payload: dict[str, Any], url: str,
                 call_id: str, attempt_index: int):
        self.recorder = recorder
        self.started = time.monotonic()
        self.trace_id = uuid.uuid4().hex
        self.call_id = call_id
        self.headers = {"traceparent": f"00-{self.trace_id}-{self.call_id[:16]}-01"}
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "capture_id": recorder.capture_id,
            "run_id": recorder.run_id,
            "request_scope": "request",
            "item_id": item_id,
            "trace_id": self.trace_id,
            "call_id": self.call_id,
            "attempt_index": attempt_index,
            "operation": url.rsplit("/", 1)[-1],
            "input_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest(),
            "seed": payload.get("seed"),
            "started_at_unix": time.time(),
        }

    def finish(self, *, response: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.record.update(seconds=time.monotonic() - self.started, error=error)
        if response is not None:
            self.record["response_id"] = response.get("id")
            self.record["usage"] = response.get("usage")
        self.recorder.write(self.record)


def begin_request(url: str, payload: dict[str, Any], *, call_id: str, attempt_index: int) -> RequestTrace | None:
    scope = _scope.get()
    if scope is None or url.endswith("/tokenize"):
        return None
    recorder, item_id = scope
    return RequestTrace(recorder, item_id, payload, url, call_id, attempt_index)
