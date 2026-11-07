import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/experiments/serving-counters.py"
SPEC = importlib.util.spec_from_file_location("serving_counters", PATH)
counters = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counters)

MODEL = "gemma-4-31b"


def body(value="12", engine="0", reason="stop"):
    return (f'vllm:generation_tokens_total{{model_name="{MODEL}",engine="{engine}"}} {value}\n'
            f'vllm:num_requests_running{{engine="{engine}",model_name="{MODEL}"}} 1\n'
            f'vllm:request_success_total{{finished_reason="{reason}",model_name="{MODEL}"}} 3\n'
            'process_start_time_seconds 100\n').encode()


def snapshot(raw=None, when="2026-11-06T00:00:00+00:00"):
    return {"schema_version": 1, "kind": "serving_counter_snapshot", "captured_utc": when,
            "source": counters.identity("http://127.0.0.1:8000/metrics", MODEL, "abc"),
            "series": counters.parse_metrics(body() if raw is None else raw, MODEL)}


class ServingCounterTests(unittest.TestCase):
    def test_labels_reordered_decimal_delta_and_gauge(self):
        before = snapshot(body("9007199254740993"))
        after = snapshot(body("9007199254740995").replace(
            b'model_name="gemma-4-31b",engine="0"', b'engine="0", model_name="gemma-4-31b"'),
            "2026-11-06T00:00:02+00:00")
        report = counters.compare(before, after)
        self.assertTrue(report["comparable"])
        rows = {r["metric"]: r for r in report["series"]}
        self.assertEqual(rows["vllm:generation_tokens_total"]["counter_delta"], "2")
        self.assertIsNone(rows["vllm:num_requests_running"]["counter_delta"])
        self.assertIsNone(rows["process_start_time_seconds"]["counter_delta"])
        before, after = snapshot(body("1")), snapshot(body("1e30"))
        before["source"]["untrusted_extra"] = "not persisted"
        report = counters.compare(before, after)
        self.assertNotIn("untrusted_extra", report["source"])
        delta = next(r["counter_delta"] for r in report["series"]
                     if r["metric"] == "vllm:generation_tokens_total")
        self.assertEqual(delta, "9" * 30)

    def test_resets_and_identity(self):
        before = snapshot()
        for after in [snapshot(body("11")), snapshot(body().replace(b" 100\n", b" 101\n"))]:
            with self.assertRaises(ValueError):
                counters.compare(before, after)
        after = snapshot()
        after["source"]["container_id"] = "different"
        with self.assertRaises(ValueError):
            counters.compare(before, after)

    def test_missing_series_unknown_not_zero(self):
        before = snapshot()
        after = snapshot()
        after["series"] = after["series"][:-1]
        report = counters.compare(before, after)
        self.assertFalse(report["comparable"])
        self.assertTrue(report["missing_after"])
        self.assertTrue(all(r["counter_delta"] is None for r in report["series"]))

    def test_mixed_model_and_arbitrary_labels_refused(self):
        for raw in [body().replace(MODEL.encode(), b"other"),
                    body() + b'vllm:prompt_tokens_total{model_name="other"} 3\n',
                    body().replace(b'engine="0"', b'request_id="secret"')]:
            with self.assertRaises(ValueError):
                counters.parse_metrics(raw, MODEL)

    def test_invalid_selected_values_body_and_duplicates(self):
        for raw in [b'{"error":"private"}', b"\xff", b"", body("NaN"), body("+Inf"),
                    body("-1"), body("1e999"), body("1e-1025"), body("1" * 129), body("1.2.3"), body() + body(), body() + b"x" * counters.MAX_BYTES,
                    body().replace(b'engine="0"', b'engine="0",engine="1"'),
                    body().replace(b'engine="0"', b'engine="\\q"')]:
            with self.assertRaises(ValueError):
                counters.parse_metrics(raw, MODEL)

    def test_escape_grammar_and_unknown_metrics_not_persisted(self):
        self.assertEqual(counters.labels(r'a="quote\"slash\\", b="line\n"'),
                         {"a": 'quote"slash\\', "b": "line\n"})
        rows = counters.parse_metrics(body("1e3") + b'other_metric{secret="not retained"} 999\n', MODEL)
        self.assertNotIn("secret", json.dumps(rows))

    def test_url_and_preview_no_io(self):
        for endpoint in ["http://a:b@host/metrics", "http://host/metrics?q=secret",
                         "http://host/#x", "file:///x", "http://host/\nmetrics"]:
            with self.assertRaises(ValueError):
                counters.endpoint_url(endpoint)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "out.json"
            with patch.object(counters, "capture", side_effect=AssertionError("no network")):
                result = counters.main(["snapshot", "--endpoint", "http://host:8000",
                                        "--model", MODEL, "--container-id", "abc", "--output", str(target)])
            self.assertEqual(result, 0)
            self.assertFalse(target.exists())

    def test_observed_vllm_033_numeric_schema(self):
        # Sanitized historical metrics only: no requests, model content or credentials.
        raw = b'vllm:num_requests_running{engine="0",model_name="gemma-4-31b"} 1.0\nvllm:num_requests_waiting{engine="0",model_name="gemma-4-31b"} 0.0\nvllm:prompt_tokens_total{engine="0",model_name="gemma-4-31b"} 101724.0\nvllm:generation_tokens_total{engine="0",model_name="gemma-4-31b"} 436259.0\nvllm:request_success_total{engine="0",finished_reason="stop",model_name="gemma-4-31b"} 142.0\nvllm:request_success_total{engine="0",finished_reason="length",model_name="gemma-4-31b"} 1.0\nvllm:request_success_total{engine="0",finished_reason="abort",model_name="gemma-4-31b"} 0.0\nvllm:request_success_total{engine="0",finished_reason="error",model_name="gemma-4-31b"} 0.0\nvllm:request_success_total{engine="0",finished_reason="repetition",model_name="gemma-4-31b"} 0.0\n'
        rows = counters.parse_metrics(raw, MODEL)
        self.assertEqual(len(rows), 9)
        reasons = {r["labels"]["finished_reason"]: r["value"] for r in rows
                   if r["metric"] == "vllm:request_success_total"}
        self.assertEqual(reasons, {"stop": "142.0", "length": "1.0", "abort": "0.0",
                                   "error": "0.0", "repetition": "0.0"})
        report = counters.compare(snapshot(raw), snapshot(raw))
        self.assertTrue(report["comparable"])
        self.assertFalse(report["process_start_available"])

    def test_bounded_fetch_and_fresh_output(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                self.size = size
                return body()

        response = Response()
        with patch.object(counters, "build_opener") as opener:
            opener.return_value.open.return_value = response
            report = counters.capture(counters.identity("http://host", MODEL, "abc"), 10)
            self.assertEqual(response.size, counters.MAX_BYTES + 1)
            self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 10)
        self.assertNotIn("body", report)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.json"
            counters.write_fresh(path, report)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                counters.write_fresh(path, report)
            with self.assertRaises(ValueError):
                counters.write_fresh(Path(temp) / "input", report, (Path(temp) / "input",))


if __name__ == "__main__":
    unittest.main()
