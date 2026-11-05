from __future__ import annotations

import asyncio
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from kairyu_bench.wsp.client import ApiError, ChatClient, _post
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.observe.core import merge_capture
from kairyu_bench.wsp.observe.middleware import CaptureMiddleware
from kairyu_bench.wsp.observe.tracing import RequestRecorder, begin_request
from kairyu_bench.wsp.run import RunSettings, execute
from kairyu_bench.wsp.sources import Context, Loader, run_chat


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def lines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in records))


class ObservationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root/"capture"
        self.capture.mkdir()
        self.config = {
            "schema_version": 1, "capture_id": "a"*32, "observation_mode": "live",
            "output_dir": str(self.capture), "checkpoint": {"sha256": "b"*64},
            "runtime": {"model": "synthetic", "adapter": "vllm-0.33", "tp_size": 2, "replicas": 1, "mtp_tokens": 4},
            "limits": {"storage_bytes": 2*1024**3, "gpu_buffer_mib": 256,
                       "prefill_positions": 32, "decode_positions": 32}}
        self.config_path = self.capture/"capture.json"
        write(self.config_path, self.config)
        self.item = "aime-2026:fixture"
        self.trace_id = "c"*32
        self.request_id = "observe-"+self.trace_id+"-"+"d"*32

    def traces(self) -> list[dict]:
        return [json.loads(line) for path in self.capture.glob("requests-*.jsonl")
                for line in path.read_text().splitlines() if line]

    @staticmethod
    def response() -> io.BytesIO:
        return io.BytesIO(json.dumps({"id": "response", "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                        "choices": [{"message": {"content": "synthetic answer", "reasoning_content": "hidden fixture"},
                                     "finish_reason": "stop"}]}).encode())

    def test_retries_have_one_call_distinct_attempts_and_no_prompt_or_secrets(self) -> None:
        recorder = RequestRecorder(self.config_path)
        sent = []
        def transport(request, **kwargs):
            sent.append(request)
            if len(sent) == 1:
                raise URLError("synthetic connection failure")
            return self.response()
        with patch("kairyu_bench.wsp.client.urlopen", side_effect=transport), patch("kairyu_bench.wsp.client.time.sleep"):
            with recorder.item(self.item):
                result = _post("http://unit.test/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "private prompt fixture"}], "seed": 7},
                               api_key="synthetic credential", timeout=1)
        self.assertEqual(result["id"], "response")
        records = self.traces()
        self.assertEqual(len(records), 2)
        self.assertEqual(len({r["call_id"] for r in records}), 1)
        self.assertEqual([r["attempt_index"] for r in records], [0, 1])
        self.assertEqual(len({r["trace_id"] for r in records}), 2)
        self.assertEqual({r["item_id"] for r in records}, {self.item})
        text = json.dumps(records)
        for secret in ("private prompt fixture", "synthetic credential", "hidden fixture", "synthetic answer"):
            self.assertNotIn(secret, text)
        self.assertIsNone(begin_request("http://unit.test/v1/completions", {}, call_id="e"*32, attempt_index=0))
        self.assertTrue(all(r.get_header("Traceparent") for r in sent))

    def test_invalid_response_is_traced_and_recording_error_does_not_change_response(self) -> None:
        recorder = RequestRecorder(self.config_path)
        with patch("kairyu_bench.wsp.client.urlopen", return_value=io.BytesIO(b"[]")):
            with recorder.item(self.item):
                with self.assertRaises(ApiError):
                    _post("http://unit.test/v1/completions", {}, api_key=None, timeout=1)
        self.assertEqual(len(self.traces()), 1)
        self.assertIsNotNone(self.traces()[0]["error"])
        with patch("kairyu_bench.wsp.client.urlopen", side_effect=lambda *a, **k: self.response()):
            with patch.object(Path, "open", side_effect=OSError("synthetic full disk")):
                with recorder.item(self.item):
                    result = _post("http://unit.test/v1/completions", {}, api_key=None, timeout=1)
        self.assertEqual(result["id"], "response")
        self.assertEqual(recorder.write_errors, 1)

    def test_scope_propagates_to_parallel_samples_and_resets_after_exception(self) -> None:
        recorder = RequestRecorder(self.config_path)
        context = Context(chat=ChatClient("http://unit.test/v1", "synthetic", concurrency=4), logprob=None)
        item = Item(self.item, "aime-2026", "dev", 10, 8, 4, "", 0)
        with patch("kairyu_bench.wsp.client.urlopen", side_effect=lambda *a, **k: self.response()):
            with recorder.item(item.id):
                result = run_chat(context, item, [{"role": "user", "content": "synthetic"}])
                self.assertIsNone(begin_request("http://unit.test/v1/tokenize", {}, call_id="e"*32, attempt_index=0))
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual({r["item_id"] for r in self.traces()}, {item.id})
        self.assertEqual(len({r["call_id"] for r in self.traces()}), 4)
        with self.assertRaises(RuntimeError):
            with recorder.item("failing"):
                raise RuntimeError("synthetic")
        self.assertIsNone(begin_request("http://unit.test/v1/completions", {}, call_id="e"*32, attempt_index=0))

    def test_executor_item_scope_joins_scored_artifacts_without_model_calls(self) -> None:
        item = Item(self.item, "aime-2026", "dev", 10, 8, 2, "", 0)
        adapter = SimpleNamespace(
            load=lambda loader: {item.id: {}},
            execute=lambda ctx, target, row: run_chat(ctx, target, [{"role": "user", "content": "synthetic"}]),
            score=lambda entries, loader: [SimpleNamespace(score=1.0, samples=(1.0,), detail={}) for _ in entries],
            batch=None)
        run_dir = self.root/"synthetic-results"
        settings = RunSettings(run_dir, 10, "dev", ChatClient("http://unit.test/v1", "synthetic"),
                               None, Loader(), 2, 1, 1, "fixture", "fixture", self.config_path)
        with patch("kairyu_bench.wsp.run.adapters", return_value={"aime-2026": adapter}), \
             patch("kairyu_bench.wsp.client.urlopen", side_effect=lambda *a, **k: self.response()), \
             patch("kairyu_bench.wsp.run.build_report", return_value={}), \
             patch("kairyu_bench.wsp.run.report_markdown", return_value="synthetic"):
            meta = execute(SimpleNamespace(sha256="manifest", tier_minutes={10: 10}), [item], settings)
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["observation"]["capture_id"], self.config["capture_id"])
        self.assertEqual({r["item_id"] for r in self.traces()}, {self.item})
        merged = merge_capture(self.capture, results=run_dir)
        self.assertEqual(merged["runs"][0]["tasks"][0]["score"], 1.0)
        # Official agent durations are stored inside one outcome sample.
        artifact = json.loads((run_dir/"items.jsonl").read_text().splitlines()[0])
        artifact["outcome"] = {"samples": [{"seconds": 12.5, "steps": 3}]}
        lines(run_dir/"items.jsonl", [artifact])
        task = merge_capture(self.capture, results=run_dir)["runs"][0]["tasks"][0]
        self.assertEqual((task["seconds"], task["steps"]), (12.5, 3))

    def worker(self, rank: int, observations: list[dict], acceptance: list[dict] | None = None) -> Path:
        directory = self.capture/"workers"/f"replica-tp{rank}"
        site = {"site_id": "layer.0.residual", "layer": 0, "role": "residual",
                "weight_names": ["model.layer.0.weight"], "tp_layout": "rank_local"}
        common = {"schema_version": 1, "capture_id": self.config["capture_id"], "replica": "replica", "rank": rank}
        write(directory/"status.json", {**common, "closed": True, "errors": {}, "dropped_records": 0})
        write(directory/"sites.json", {**common, "sites": [site], "checkpoint": self.config["checkpoint"],
                                        "runtime": self.config["runtime"]})
        lines(directory/"observations.jsonl", observations)
        lines(directory/"acceptance.jsonl", acceptance or [])
        return directory

    def observation(self, rank: int, position: int, rms: float | None, *, speculative: int | None = None) -> dict:
        return {"schema_version": 1, "capture_id": self.config["capture_id"],
                "request_id": self.request_id, "replica": "replica", "rank": rank,
                "step": 4, "site_id": "layer.0.residual", "layer": 0, "role": "residual",
                "phase": "decode", "token_kind": "target", "token_position": position,
                "speculative_index": speculative, "accepted": None if speculative is not None else True,
                "metrics": {"rms": rms, "mean": rms, "max_abs": rms, "nonfinite_count": 0}}

    def linked(self) -> None:
        lines(self.capture/"requests-fixture.jsonl", [{"schema_version": 1, "capture_id": self.config["capture_id"],
              "trace_id": self.trace_id, "call_id": "e"*32, "attempt_index": 0,
              "item_id": self.item, "input_sha256": "f"*64, "error": None}])

    def test_merge_acceptance_boundary_and_metric_missingness(self) -> None:
        self.linked()
        accept = {"schema_version": 1, "capture_id": self.config["capture_id"], "replica": "replica",
                  "rank": 0, "step": 4, "request_id": self.request_id,
                  "scheduled_draft_tokens": 1, "accepted_draft_tokens": 0}
        self.worker(0, [self.observation(0, 2, None), self.observation(0, 3, 2)], [accept])
        self.worker(1, [self.observation(1, 4, 8, speculative=0)])
        merged = merge_capture(self.capture)
        run = merged["runs"][0]
        self.assertEqual(run["capture"]["status"], "complete")
        rows = run["tasks"][0]["observations"]
        ordinary = next(r for r in rows if r["rank"] == 0)
        rejected = next(r for r in rows if r["rank"] == 1)
        self.assertEqual(ordinary["metrics"]["rms"], 2)
        self.assertEqual(ordinary["metric_sample_counts"]["rms"], 1)
        self.assertFalse(rejected["accepted"])
        acceptance_path = self.capture/"workers"/"replica-tp0"/"acceptance.jsonl"
        lines(acceptance_path, [{**accept, "accepted_draft_tokens": 2}])
        with self.assertRaisesRegex(ValueError, "invalid speculative acceptance counts"):
            merge_capture(self.capture)

    def test_compact_gzip_members_merge_like_legacy_and_reject_misaligned_values(self) -> None:
        self.linked()
        worker = self.worker(0, [])
        self.worker(1, [])
        (worker/"observations.jsonl").unlink()
        batch = {
            "schema_version": 1, "record_type": "activation_batch",
            "capture_id": self.config["capture_id"], "observation_mode": "live",
            "replica": "replica", "rank": 0, "step": 1,
            "site_ids": ["layer.0.residual"], "metric_names": ["rms", "mean", "nonfinite_count"],
            "row_fields": ["request_id", "token_position", "phase", "speculative_index", "accepted"],
            "rows": [[self.request_id, 2, "decode", None, True]],
            "values": [[[2, 2, 0]]],
        }
        path = worker/"observations.jsonl.gz"
        with path.open("wb") as stream:
            stream.write(gzip.compress((json.dumps(batch)+"\n").encode()))
            later = {**batch, "step": 2, "rows": [[self.request_id, 3, "decode", None, True]],
                     "values": [[[4, 4, 0]]]}
            stream.write(gzip.compress((json.dumps(later)+"\n").encode()))
        result = merge_capture(self.capture)["runs"][0]
        cell = result["tasks"][0]["observations"][0]
        self.assertAlmostEqual(cell["metrics"]["rms"], 10**0.5)
        self.assertEqual(cell["sample_count"], 2)
        valid_bytes = path.read_bytes()
        path.write_bytes(valid_bytes[:-4])
        with self.assertRaisesRegex(ValueError, "wait for writer drain"):
            merge_capture(self.capture)
        bad = {**batch, "values": [[]]}
        path.write_bytes(gzip.compress((json.dumps(bad)+"\n").encode()))
        with self.assertRaisesRegex(ValueError, "row layout"):
            merge_capture(self.capture)

    def test_request_budget_is_shared_across_processes_and_marks_incomplete(self) -> None:
        path = self.capture/"requests-budget.jsonl"
        script = (
            "from pathlib import Path; "
            "from kairyu_bench.wsp.observe.budget import append_bounded_jsonl; "
            "import sys; "
            "[append_bounded_jsonl(Path(sys.argv[1]), {'event': 'calibration'}, 2048) for _ in range(100)]"
        )
        processes = [subprocess.Popen([sys.executable, "-c", script, str(path)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(4)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertLessEqual(path.stat().st_size, 2048)
        self.assertGreater(path.stat().st_size, 2000)
        self.assertTrue(all(json.loads(line) == {"event": "calibration"} for line in path.read_text().splitlines()))
        self.assertTrue(path.with_suffix(".quota-exhausted").is_file())

    def test_trial_trace_and_shared_capture_keep_runs_separate(self) -> None:
        first = RequestRecorder(self.config_path, run_id="run-a")
        second = RequestRecorder(self.config_path, run_id="run-b")
        one = first.register_trial(self.item, "trial-a", attempt_index=None)
        two = second.register_trial(self.item, "trial-b", attempt_index=0)
        records = self.traces()
        self.assertEqual({r["request_scope"] for r in records}, {"trial"})
        self.assertEqual({r["run_id"] for r in records}, {"run-a", "run-b"})
        observations = []
        for headers, value in ((one, 2), (two, 99)):
            record = self.observation(0, 1, value)
            record["request_id"] = "observe-" + headers["traceparent"].split("-")[1] + "-" + "d"*32
            observations.append(record)
        self.worker(0, observations)
        self.worker(1, [])
        run_dir = self.root/"run-a"
        write(run_dir/"run.json", {"observation": {"capture_id": self.config["capture_id"], "run_id": "run-a"}})
        lines(run_dir/"items.jsonl", [{"id": self.item, "source": "aime-2026", "score": 1}])
        merged = merge_capture(self.capture, results=run_dir)["runs"][0]
        self.assertEqual(merged["capture"]["counts"]["excluded_other_run_records"], 1)
        self.assertEqual(merged["tasks"][0]["observations"][0]["metrics"]["rms"], 2)
        self.assertEqual(merged["tasks"][0]["observations"][0]["trial_id"], "trial-a")
        self.assertEqual(merged["tasks"][0]["observations"][0]["request_scope"], "trial")

    def test_merge_rejects_wrong_capture_and_duplicate_worker_identity(self) -> None:
        self.linked()
        first = self.worker(0, [self.observation(0, 2, 1)])
        second = self.worker(1, [self.observation(1, 2, 1)])
        original = json.loads((second/"status.json").read_text())
        write(second/"status.json", {**original, "capture_id": "wrong"})
        with self.assertRaises(ValueError):
            merge_capture(self.capture)
        write(second/"status.json", {**original, "rank": 0})
        with self.assertRaises(ValueError):
            merge_capture(self.capture)
        write(second/"status.json", original)
        bad = self.observation(0, 3, 1)
        bad["capture_id"] = "wrong"
        lines(first/"observations.jsonl", [bad])
        with self.assertRaises(ValueError):
            merge_capture(self.capture)

    def test_middleware_preserves_body_and_forwards_trace_without_secret_logging(self) -> None:
        observed, sent = [], []
        async def app(scope, receive, send):
            observed.append(scope)
            body = await receive()
            self.assertEqual(body["body"], b"private synthetic body")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"synthetic response"})
        async def receive():
            return {"type": "http.request", "body": b"private synthetic body"}
        async def send(message):
            sent.append(message)
        env = {"KAIRYU_BENCH_CAPTURE_CONFIG": str(self.config_path), "KAIRYU_BENCH_CAPTURE_REPLICA": "replica"}
        with patch.dict(os.environ, env):
            middleware = CaptureMiddleware(app)
        scope = {"type": "http", "path": "/v1/chat/completions", "headers": [
            (b"traceparent", ("00-"+self.trace_id+"-"+"e"*16+"-01").encode()),
            (b"x-request-id", b"old"), (b"authorization", b"synthetic credential")]}
        asyncio.run(middleware(scope, receive, send))
        headers = dict(observed[0]["headers"])
        self.assertTrue(headers[b"x-request-id"].decode().startswith("observe-"+self.trace_id+"-"))
        self.assertEqual(sent[-1]["body"], b"synthetic response")
        record = json.loads(middleware.path.read_text())
        self.assertEqual(record["status"], 200)
        self.assertEqual(record["trace_id"], self.trace_id)
        self.assertNotIn("private synthetic body", middleware.path.read_text())
        self.assertNotIn("synthetic credential", middleware.path.read_text())


if __name__ == "__main__":
    unittest.main()
