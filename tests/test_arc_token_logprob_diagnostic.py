from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kairyu_bench.wsp.client import LogprobClient
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import ADAPTERS, Context, Loader
from kairyu_bench.wsp.sources import logprob

_SPEC = importlib.util.spec_from_file_location(
    "arc_token_diagnostic", Path(__file__).resolve().parents[1] / "scripts/experiments/arc-token-logprob-diagnostic.py")
assert _SPEC is not None and _SPEC.loader is not None
DIAG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DIAG)


class TokenDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.saved_bos = dict(logprob._bos)
        self.addCleanup(lambda: (logprob._bos.clear(), logprob._bos.update(self.saved_bos)))
        self.row = {"question": "Synthetic fixture?", "choices": {
            "text": ["one", "two words", "three", "four"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B"}

    def run_official(self, *, parallel=False):
        logprob._bos.clear()
        requests, responses = [], []
        overlap = threading.Barrier(10, timeout=5) if parallel else None

        def transport(url, payload, *, api_key, timeout):
            self.assertIsNone(api_key)
            requests.append((url, copy.deepcopy(payload)))
            if url.endswith("/tokenize"):
                if overlap is not None and payload["prompt"].endswith(" one"):
                    overlap.wait()  # All ten distinct item scopes overlap before their head requests.
                response = {"tokens": [1] if payload["prompt"] == "<bos>"
                            else [ord(char) + 10 for char in payload["prompt"]]}
            else:
                response = {"choices": [{"prompt_logprobs": [None] + [
                    {str(token): {"logprob": -(token % 7 + index) / 10, "rank": 2}}
                    for index, token in enumerate(payload["prompt"][1:], 1)]}]}
            responses.append((response, copy.deepcopy(response)))
            return response

        context = Context(chat=None, logprob=LogprobClient(("http://fixture/v1",), "synthetic"))
        row_before = copy.deepcopy(self.row)
        adapter = ADAPTERS["arc-challenge"]
        self.assertIs(adapter.score, logprob._score_choice)
        def run_item(index_and_id):
            index, item_id = index_and_id
            row = copy.deepcopy(self.row)
            if parallel:
                row["question"] = f"Synthetic fixture {index}?" + " x" * index
            item = Item(item_id, "arc-challenge", "dev", 10, 1, 1, "", 0)
            outcome = adapter.execute(context, item, row)
            return outcome, adapter.score([(item, row, outcome)], Loader())

        with patch("kairyu_bench.wsp.client._post", side_effect=transport):
            if parallel:
                with ThreadPoolExecutor(max_workers=32) as pool:
                    results = list(pool.map(run_item, enumerate(DIAG.TARGETS)))
            else:
                results = list(map(run_item, enumerate(DIAG.TARGETS)))
        outcomes, scores = zip(*results, strict=True)
        self.assertEqual(self.row, row_before)
        for response, before in responses:
            self.assertEqual(response, before)
        return requests, outcomes, scores

    def test_exact_http_payloads_counts_scores_and_token_sum_with_real_adapter(self):
        baseline = self.run_official()
        original = ADAPTERS["arc-challenge"]
        recorder = DIAG.Recorder()
        with DIAG.recording_adapter(recorder):
            self.assertIs(ADAPTERS["arc-challenge"].score, original.score)
            observed = self.run_official()
        self.assertIs(ADAPTERS["arc-challenge"], original)
        self.assertEqual(observed, baseline)
        self.assertEqual(len(observed[0]), 121)  # 81 tokenize + 40 completion, no extra API call.
        self.assertEqual(recorder.client_calls, {"tokenize": 81, "prompt_logprobs": 40})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "tokens.json"
            summary = recorder.flush(output, 0)
            self.assertTrue(summary["capture_complete"])
            self.assertLessEqual(output.stat().st_size, DIAG.MAX_BYTES)
            saved = json.loads(output.read_text())
        self.assertEqual(len(saved["records"]), 40)
        for record in saved["records"]:
            self.assertEqual(record["continuation_start"], 1 + record["tokenization_lengths"]["head"])
            self.assertEqual(len(record["continuation_logprobs"]),
                             record["continuation_stop"] - record["continuation_start"])
            # Python 3.12 sum() can compensate rounding; the official adapter
            # explicitly accumulates left-to-right from 0.0 instead.
            total = 0.0
            for value in record["continuation_logprobs"]:
                total += value
            self.assertEqual(total, record["official_sum"])
            self.assertTrue(record["official_sum_matches"])
            self.assertEqual(len(record["text_sha256"]["head"]), 64)
            self.assertNotIn("gold", record)
            self.assertNotIn("prompt", record)

    def test_overlapping_item_scopes_keep_distinct_boundaries_and_payloads(self):
        baseline = self.run_official(parallel=True)
        recorder = DIAG.Recorder()
        with DIAG.recording_adapter(recorder):
            observed = self.run_official(parallel=True)
        self.assertEqual(observed[1:], baseline[1:])
        def wire(requests):
            return sorted(json.dumps(x, sort_keys=True) for x in requests)
        self.assertEqual(wire(observed[0]), wire(baseline[0]))
        self.assertEqual(recorder.client_calls, {"tokenize": 81, "prompt_logprobs": 40})
        self.assertEqual(len(recorder.records), 40)
        for raw in recorder.records:
            record = json.loads(raw)
            index = DIAG.TARGETS.index(record["item_id"])
            question = f"Synthetic fixture {index}?" + " x" * index
            prefix = f"Question: {question}\nAnswer:"
            self.assertEqual(record["text_sha256"]["head"], hashlib.sha256(prefix.encode()).hexdigest())
            self.assertEqual(record["continuation_start"], 1 + len(prefix))
            self.assertEqual(record["token_ids"][1:record["continuation_start"]],
                             [ord(char) + 10 for char in prefix])
            self.assertTrue(record["official_sum_matches"])

    def test_quota_exhaustion_preserves_requests_outputs_and_reports_incomplete(self):
        baseline = self.run_official()
        recorder = DIAG.Recorder()
        with patch.object(DIAG, "RECORD_BYTES", 1), DIAG.recording_adapter(recorder):
            self.assertEqual(self.run_official(), baseline)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "missing.json"
            summary = recorder.flush(output, 0)
            self.assertFalse(summary["capture_complete"])
            self.assertEqual(summary["recorded_calls"], 40)
            self.assertEqual(summary["capture_errors"], {"capture_record_quota": 40})
            failures = json.loads(output.read_text())["records"]
            self.assertEqual({x["item_id"] for x in failures}, set(DIAG.TARGETS))
            self.assertTrue(all(x["capture_error"]["category"] == "record_quota" for x in failures))
            self.assertTrue(all(x["official_sum_matches"] is False for x in failures))
            self.assertLess(output.stat().st_size, DIAG.MAX_BYTES)

    def test_proxy_returns_identical_objects_and_original_exceptions(self):
        class Delegate:
            tokens = [11, 12, 13]
            positions = [None, {12: (-1., 1)}, {13: (-2., 1)}]
            failure = None

            def tokenize(self, text, *, add_special_tokens):
                return self.tokens

            def prompt_logprobs(self, tokens, *, top_k):
                if self.failure:
                    raise self.failure
                return self.positions

        delegate = Delegate()
        recorder = DIAG.Recorder()
        proxy = DIAG.RecordingClient(delegate, recorder, DIAG.TARGETS[0])
        self.assertIs(proxy.tokenize("whole", add_special_tokens=False), delegate.tokens)
        self.assertIs(proxy.tokenize("head", add_special_tokens=False), delegate.tokens)
        self.assertIs(proxy.prompt_logprobs(delegate.tokens, top_k=1), delegate.positions)
        delegate.failure = RuntimeError("synthetic-secret-not-for-evidence")
        with self.assertRaises(RuntimeError) as caught:
            proxy.prompt_logprobs(delegate.tokens, top_k=1)
        self.assertIs(caught.exception, delegate.failure)
        proxy.finish(None)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failure.json"
            recorder.flush(output, 1, "RuntimeError")
            self.assertNotIn("synthetic-secret", output.read_text())


if __name__ == "__main__":
    unittest.main()
