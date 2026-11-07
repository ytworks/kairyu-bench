from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
spec = importlib.util.spec_from_file_location(
    "science_generate_fixture", DIRECTORY / "science_generate.py"
)
GENERATOR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GENERATOR)


class ScienceGenerationTest(unittest.TestCase):
    def fixture(self, directory, split="train"):
        path = Path(directory) / "cases.json"
        row = {
            "id": "fixture",
            "split": split,
            "prompt": "Write solve(x).",
            "correct": "reference-not-sent",
            "foil": "foil-not-sent",
        }
        data = (
            {"cases": [row], "split": split}
            if split == "transfer"
            else {"purpose": [[row]]}
        )
        path.write_text(json.dumps(data))
        return path

    def arguments(self, cases, output, split="train"):
        return [
            "--cases",
            str(cases),
            "--split",
            split,
            "--endpoint",
            "http://127.0.0.1:9999/v1",
            "--model",
            "fixture-model",
            "--checkpoint-sha256",
            "a" * 64,
            "--output",
            str(output),
        ]

    def test_preview_no_network_and_both_case_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:

            def forbidden(*_args, **_kwargs):
                self.fail("preview made a request")

            for split in ("train", "transfer"):
                cases = self.fixture(directory, split)
                output = Path(directory) / split
                report = GENERATOR.main(
                    self.arguments(cases, output, split), opener=forbidden
                )
                self.assertEqual((report["status"], report["cases"]), ("preview", 1))
                self.assertFalse((output / (split + "-responses.json")).exists())
            with self.assertRaises(FileExistsError):
                GENERATOR.main(self.arguments(cases, output, split), opener=forbidden)

    def test_only_prompt_sent_public_answer_saved_and_server_sampling_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = self.fixture(directory)
            before = cases.read_bytes()
            output = Path(directory) / "run"
            requests = []

            def opener(request, timeout):
                body = json.loads(request.data)
                requests.append(body)
                self.assertEqual(timeout, 240)
                return io.BytesIO(
                    json.dumps(
                        {
                            "id": "request1",
                            "choices": [
                                {
                                    "message": {
                                        "content": "def solve(x): return x",
                                        "reasoning_content": "NEVER-SAVE-THIS",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 8, "completion_tokens": 12},
                        }
                    ).encode()
                )

            receipt = GENERATOR.main(
                self.arguments(cases, output) + ["--run"], opener=opener
            )
            self.assertEqual((receipt["received"], receipt["errors"]), (1, 0))
            self.assertEqual(requests[0], {
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "Write solve(x)."}],
                "max_tokens": 8192,
                "seed": 0,
                "chat_template_kwargs": {"enable_thinking": True},
            })
            self.assertNotIn("reference-not-sent", json.dumps(requests))
            self.assertNotIn("foil-not-sent", json.dumps(requests))
            text = (output / "train-responses.json").read_text()
            self.assertNotIn("NEVER-SAVE-THIS", text)
            self.assertEqual(
                json.loads(text)["responses"][0]["usage"]["output_tokens"], 12
            )
            self.assertEqual(cases.read_bytes(), before)

    def test_server_default_omits_only_override_and_records_unknown_thinking(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = self.fixture(directory)
            output = Path(directory) / "server-default"
            requests = []

            def opener(request, timeout):
                requests.append(json.loads(request.data))
                return io.BytesIO(json.dumps({
                    "choices": [{"message": {"content": "7"}, "finish_reason": "stop"}]
                }).encode())

            result = GENERATOR.main(
                self.arguments(cases, output)
                + ["--thinking-control", "server-default", "--run"],
                opener=opener,
            )
            self.assertEqual(requests, [{
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "Write solve(x)."}],
                "max_tokens": 8192,
                "seed": 0,
            }])
            generation = json.loads(
                (output / "train-responses.json").read_text()
            )["generation"]
            self.assertEqual(generation["thinking_control"], "server-default")
            self.assertEqual(generation["frame"], "live-server-default-public-answer")
            self.assertIsNone(generation["enable_thinking"])
            self.assertEqual(result["thinking_control"], "server-default")
            self.assertEqual((result["received"], result["errors"]), (1, 0))

    def test_http_400_is_retained_without_body_answer_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = self.fixture(directory)
            output = Path(directory) / "http-error"
            calls = []

            def opener(*_args, **_kwargs):
                calls.append(1)
                raise urllib.error.HTTPError(
                    "http://private-url-sentinel", 400, "private-message-sentinel",
                    {"private-header-sentinel": "private-value-sentinel"},
                    io.BytesIO(b"private-body-sentinel"),
                )

            result = GENERATOR.main(
                self.arguments(cases, output) + ["--run"], opener=opener
            )
            text = (output / "train-responses.json").read_text()
            self.assertNotIn("sentinel", text)
            self.assertEqual(json.loads(text)["responses"], [{
                "id": "fixture", "error_type": "HTTPError", "http_status": 400,
            }])
            self.assertEqual((len(calls), result["received"], result["errors"]), (1, 1, 1))

    def test_error_body_never_persisted_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = self.fixture(directory)
            output = Path(directory) / "error"
            calls = []

            def opener(*_args, **_kwargs):
                calls.append(1)
                raise RuntimeError("PRIVATE-ERROR-BODY")

            result = GENERATOR.main(
                self.arguments(cases, output) + ["--run"], opener=opener
            )
            self.assertEqual((len(calls), result["errors"]), (1, 1))
            text = (output / "train-responses.json").read_text()
            self.assertNotIn("PRIVATE-ERROR-BODY", text)
            self.assertEqual(
                json.loads(text)["responses"][0]["error_type"], "RuntimeError"
            )


if __name__ == "__main__":
    unittest.main()
