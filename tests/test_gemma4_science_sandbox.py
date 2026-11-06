from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "science_sandbox_fixture", DIRECTORY / "science_sandbox.py"
)
SANDBOX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SANDBOX)
SCIENCE = SANDBOX.science_cases


class ScienceSandboxTest(unittest.TestCase):
    def inputs(self, directory):
        path = Path(directory)
        cases = path / "train.json"
        cases.write_text(
            json.dumps(
                SCIENCE.training_payload(
                    [c for c in SCIENCE.catalog() if c.split == "train"]
                )
            )
        )
        responses = path / "responses.json"
        response = {
            "schema_version": 1,
            "split": "train",
            "cases_sha256": SANDBOX.sha(cases.read_bytes()),
            "generation": {"reasoning_saved": False, "frame": "fixture"},
            "responses": [
                {
                    "id": "science-weighted-moments",
                    "public_answer": "def solve(x, w):\n    return x",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ],
        }
        responses.write_text(json.dumps(response))
        return cases, responses

    def test_preview_preserves_inputs_and_does_not_construct_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            cases, responses = self.inputs(directory)
            original = responses.read_bytes()
            output = Path(directory) / "preview"
            args = [
                "--cases",
                str(cases),
                "--responses",
                str(responses),
                "--output",
                str(output),
                "--docker-socket",
                "/nonexistent/docker.sock",
                "--daemon-id",
                "expected",
                "--image",
                "sha256:" + "a" * 64,
            ]

            def forbidden(*_args, **_kwargs):
                self.fail("preview attempted Docker")

            report = SANDBOX.main(args, docker_factory=forbidden)
            self.assertEqual(report["status"], "preview")
            self.assertEqual(responses.read_bytes(), original)
            self.assertTrue((output / "report.json").is_file())
            with self.assertRaises(FileExistsError):
                SANDBOX.main(args, docker_factory=forbidden)

    def test_no_reference_payload_and_host_numeric_contracts(self):
        case = SCIENCE.catalog()[0]
        payload, expected = SANDBOX.prepare(case, case.code(), seed=4, trials=4)
        self.assertEqual(set(payload), {"code", "parameters", "trials"})
        self.assertNotIn("expected", json.dumps(payload))
        rows = [
            {
                "shape": list(v.shape),
                "dtype": "float64",
                "data": v.tolist(),
                "unchanged": True,
            }
            for v in expected
        ]
        self.assertTrue(SANDBOX.compare({"trials": rows}, expected)["passed"])
        rows[0]["data"][0] += 1
        rows[1]["shape"] = [99]
        rows[2]["dtype"] = "float32"
        rows[3]["unchanged"] = False
        result = SANDBOX.compare({"trials": rows}, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(
            [r["kind"] for r in result["failures"]],
            ["numeric", "output_shape", "output_dtype", "input_mutation"],
        )
        with self.assertRaises(ValueError):
            SANDBOX.compare({"trials": []}, expected)

    def test_strict_public_answer_extraction_and_reasoning_rejection(self):
        code = "import numpy as np\ndef solve(x):\n    return x\n"
        fence = chr(96) * 3
        self.assertEqual(SANDBOX.extract(fence + "python\n" + code + fence), code)
        with self.assertRaises((SyntaxError, ValueError)):
            SANDBOX.extract("Here is code:\n" + fence + "python\n" + code + fence)
        with tempfile.TemporaryDirectory() as directory:
            cases, responses = self.inputs(directory)
            data = json.loads(responses.read_text())
            data["responses"][0]["reasoning_content"] = "must not be read as an answer"
            responses.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "no reasoning"):
                SANDBOX.read_inputs(cases, responses)

    def test_daemon_guard_and_bounded_own_container_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "docker.sock"
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(path))
            self.addCleanup(listener.close)
            calls, state = [], {"daemon": "wrong", "name": None, "nonce": None}

            def command(argv, **kwargs):
                args = argv[3:]
                calls.append(args)
                if args[0] == "info":
                    return 0, state["daemon"].encode()
                if args[:2] == ["image", "inspect"]:
                    return 0, ("sha256:" + "a" * 64).encode()
                if args[0] == "run":
                    state["name"] = args[args.index("--name") + 1]
                    state["nonce"] = args[args.index("--label") + 1].split("=")[1]
                    self.assertNotIn("--gpus", args)
                    self.assertNotIn("--privileged", args)
                    self.assertNotIn("--volume", args)
                    for key, value in (
                        ("--network", "none"),
                        ("--user", "1003:1003"),
                        ("--pull", "never"),
                        ("--cap-drop", "ALL"),
                    ):
                        self.assertEqual(args[args.index(key) + 1], value)
                    self.assertIn("--read-only", args)
                    self.assertEqual(args[args.index("--entrypoint") + 1], "python3")
                    self.assertEqual(kwargs["seconds"], 2)
                    raise TimeoutError("fixture")
                if args[:2] == ["container", "ls"]:
                    return 0, b"bbbbbbbbbbbb"
                if args[:2] == ["container", "inspect"]:
                    return 0, json.dumps("b" * 64).encode() + b" " + json.dumps(
                        state["nonce"]
                    ).encode()
                if args[:3] == ["container", "rm", "--force"]:
                    self.assertEqual(args[3], "b" * 64)
                    return 0, b""
                self.fail("unexpected operation")

            docker = SANDBOX.Docker(
                path,
                "expected",
                "sha256:" + "a" * 64,
                seconds=2,
                memory_mib=512,
                cpus=1,
                command=command,
            )
            with self.assertRaisesRegex(Exception, "unexpected Docker daemon"):
                docker.run({})
            self.assertFalse(any(c[0] == "run" for c in calls))
            state["daemon"] = "expected"
            with self.assertRaises(TimeoutError):
                docker.run({})
            self.assertEqual(sum(c[:2] == ["container", "rm"] for c in calls), 1)
            state["nonce"] = "c" * 32
            with self.assertRaisesRegex(Exception, "ownership differs"):
                docker.cleanup(state["name"], "d" * 32)
            self.assertEqual(sum(c[:2] == ["container", "rm"] for c in calls), 1)

    def test_process_output_and_time_limits_on_trusted_fixtures(self):
        with self.assertRaisesRegex(ValueError, "output limit"):
            SANDBOX.bounded_command(
                [sys.executable, "-c", "print('x'*100000)"], max_output=100, seconds=5
            )
        with self.assertRaises(TimeoutError):
            SANDBOX.bounded_command(
                [sys.executable, "-c", "import time; time.sleep(10)"], seconds=0.1
            )
        code, output = SANDBOX.bounded_command([sys.executable, "-c", "print('ok')"])
        self.assertEqual((code, output.strip()), (0, b"ok"))


if __name__ == "__main__":
    unittest.main()
