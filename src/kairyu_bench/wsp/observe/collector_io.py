"""Bounded CPU serialization queue. No torch/vLLM import or model execution."""
from __future__ import annotations

import gzip
import json
import queue
import threading
import time
from typing import Any

import numpy as np

from kairyu_bench.wsp.observe.collector import METRICS


class RecordQueue:
    def __init__(self, writer: Any, sites: list[dict[str, Any]], *,
                 max_batches: int = 8, max_bytes: int = 64 * 1024**2):
        self.writer, self.sites = writer, sites
        self.max_batches, self.max_bytes = max_batches, max_bytes
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max_batches)
        self.lock = threading.Lock()
        self.accepting = True
        self.pending_bytes = 0
        self.highwater_bytes = 0
        self.highwater_batches = 0
        self.dropped_batches = 0
        self.enqueued_batches = 0
        self.completed_batches = 0
        self.last_enqueued_at_unix: float | None = None
        self.last_completed_at_unix: float | None = None
        self.serialize_seconds = 0.0
        self.compress_seconds = 0.0
        self.writer_cpu_seconds = 0.0
        self.drain_seconds = 0.0
        self.close_timeout = False
        self.clean_exit = False
        self.thread = threading.Thread(target=self._run, name="kairyu-observe-writer", daemon=True)
        self.thread.start()

    def submit(self, pending: Any, accepted: dict[str, int]) -> bool:
        logical = len(pending.rows) * len(self.sites)
        if not logical:
            return True
        size = pending.values.nbytes + len(pending.rows) * 512 + len(accepted) * 256
        with self.lock:
            if (not self.accepting or self.writer.quota_exhausted
                    or self.pending_bytes + size > self.max_bytes
                    or self.enqueued_batches - self.completed_batches >= self.max_batches
                    or not self.thread.is_alive()):
                self.dropped_batches += 1
                self.writer.drop(logical)
                return False
            self.pending_bytes += size
            self.highwater_bytes = max(self.highwater_bytes, self.pending_bytes)
            self.enqueued_batches += 1
            self.queue.put_nowait((pending, accepted, size))
            self.last_enqueued_at_unix = time.time()
            self.highwater_batches = max(self.highwater_batches,
                                         self.enqueued_batches - self.completed_batches)
            return True

    def _encode(self, pending: Any, accepted: dict[str, int]) -> bytes:
        rows = []
        for row in pending.rows:
            accept: bool | None = True
            if row.speculative_index is not None:
                accept = (row.speculative_index < accepted[row.request_id]
                          if row.request_id in accepted else None)
            rows.append([row.request_id, row.token_position, row.phase,
                         row.speculative_index, accept])
        values: Any = pending.values
        if not np.isfinite(values).all():
            finite = np.isfinite(values)
            values = values.astype(object)
            values[~finite] = None
        data = {
            "schema_version": 1, "record_type": "activation_batch",
            "capture_id": self.writer.config.capture_id, "observation_mode": "live",
            "replica": self.writer.replica, "rank": self.writer.rank, "step": pending.step,
            "site_ids": [site["site_id"] for site in self.sites],
            "metric_names": list(METRICS),
            "row_fields": ["request_id", "token_position", "phase", "speculative_index", "accepted"],
            "rows": rows, "values": values.tolist(),
            "sample": {"policy": "first_execution_of_first_unique_positions",
                       "limits": {"prefill": self.writer.config.prefill_positions,
                                  "decode": self.writer.config.decode_positions}},
        }
        try:
            import msgspec
        except ImportError:
            return (json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n").encode()
        return msgspec.json.encode(data) + b"\n"

    def _write(self, pending: Any, accepted: dict[str, int]) -> None:
        logical = len(pending.rows) * len(self.sites)
        if self.writer.quota_exhausted:
            self.writer.drop(logical)
            return
        started = time.perf_counter()
        line = self._encode(pending, accepted)
        encoded_at = time.perf_counter()
        compressed = gzip.compress(line, compresslevel=1, mtime=0)
        compressed_at = time.perf_counter()
        self.serialize_seconds += encoded_at - started
        self.compress_seconds += compressed_at - encoded_at
        self.writer.write_encoded("observations", compressed, logical)
        if accepted:
            self.writer.write("acceptance", [
                {
                    "schema_version": 1, "capture_id": self.writer.config.capture_id,
                    "replica": self.writer.replica, "rank": self.writer.rank,
                    "step": pending.step, "request_id": req,
                    "scheduled_draft_tokens": pending.drafts[req],
                    "accepted_draft_tokens": count,
                    "semantics": "rejection_sampler_prefix_not_stop_filtered",
                }
                for req, count in accepted.items()
            ])

    def _publish_status(self, *, closed: bool = False) -> None:
        with self.lock:
            snapshot = {
                "max_batches": self.max_batches, "max_bytes": self.max_bytes,
                "depth": self.enqueued_batches - self.completed_batches, "pending_bytes": self.pending_bytes,
                "highwater_batches": self.highwater_batches,
                "highwater_bytes": self.highwater_bytes, "dropped_batches": self.dropped_batches,
                "enqueued_batches": self.enqueued_batches, "completed_batches": self.completed_batches,
                "last_enqueued_at_unix": self.last_enqueued_at_unix,
                "last_completed_at_unix": self.last_completed_at_unix,
                "serialize_seconds": self.serialize_seconds,
                "compress_seconds": self.compress_seconds, "writer_cpu_seconds": self.writer_cpu_seconds,
                "drain_seconds": self.drain_seconds, "close_timeout": self.close_timeout,
            }
        self.writer.extra_status["writer_queue"] = snapshot
        self.writer.flush(closed=closed)

    def _run(self) -> None:
        last_flush = time.monotonic()
        clean_exit = False
        try:
            while True:
                try:
                    item = self.queue.get(timeout=1.0)
                except queue.Empty:
                    self._publish_status()
                    last_flush = time.monotonic()
                    continue
                if item is None:
                    self.queue.task_done()
                    clean_exit = True
                    self.clean_exit = True
                    break
                pending, accepted, size = item
                started_cpu = time.thread_time()
                try:
                    self._write(pending, accepted)
                except Exception as error:
                    self.writer.error(f"background_{type(error).__name__}")
                    self.writer.drop(len(pending.rows) * len(self.sites))
                finally:
                    self.writer_cpu_seconds += time.thread_time() - started_cpu
                    with self.lock:
                        self.pending_bytes -= size
                        self.completed_batches += 1
                        self.last_completed_at_unix = time.time()
                    self.queue.task_done()
                if time.monotonic() - last_flush >= 1.0:
                    self._publish_status()
                    last_flush = time.monotonic()
        finally:
            self._publish_status(closed=clean_exit)

    def close(self, timeout: float = 30.0) -> bool:
        started = time.perf_counter()
        with self.lock:
            if not self.accepting:
                return not self.thread.is_alive()
            self.accepting = False
        try:
            self.queue.put(None, timeout=timeout)
        except queue.Full:
            self.close_timeout = True
            return False
        self.thread.join(timeout=max(0, timeout - (time.perf_counter() - started)))
        self.drain_seconds = time.perf_counter() - started
        if self.thread.is_alive():
            self.close_timeout = True
            # Never mark a live background writer closed. Its last status stays
            # partial and the process shutdown may discard queued records.
            return False
        self._publish_status(closed=self.clean_exit)
        return self.clean_exit
