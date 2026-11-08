from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WeightCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/"model.safetensors"
        header = json.dumps({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
        self.model.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, 2, 3, 4))
        self.original = self.model.read_bytes()
        self.bin = self.root/"bin"
        self.bin.mkdir()
        self.marker = self.root/"docker-called"
        docker = self.bin/"docker"
        docker.write_text("#!/bin/sh\n: > '" + str(self.marker).replace("'", "'\\''") + "'\nexit 91\n")
        docker.chmod(0o755)
        self.env = {**os.environ, "KAIRYU_BENCH_PYTHON": sys.executable,
                    "PATH": str(self.bin)+os.pathsep+os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1"}

    def cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(ROOT/"kairyu-bench"), "wsp", *arguments], cwd=self.root,
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_normal_wrapper_runs_all_static_operations_without_docker(self) -> None:
        commands = [
            ["inspect", str(self.model)],
            ["diff", str(self.model), str(self.model)],
            ["spectrum", str(self.model), "--tensor", "w"],
            ["align", str(self.model), "--left", "w", "--right", "w"],
        ]
        for command in commands:
            with self.subTest(operation=command[0]):
                output = self.root/(command[0]+".json")
                result = self.cli("weights", *command, "--output", str(output))
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(output.read_text())
                self.assertEqual(data["schema_version"], 1)
                self.assertFalse(self.marker.exists())
        self.assertEqual(self.model.read_bytes(), self.original)

    def test_benchmark_wrapper_mounts_external_capture_at_identical_path(self) -> None:
        app = self.root / "app"
        app.mkdir()
        wrapper = app / "kairyu-bench"
        wrapper.write_bytes((ROOT / "kairyu-bench").read_bytes())
        wrapper.chmod(0o755)
        capture = self.root / "external capture"
        capture.mkdir()
        config = capture / "capture.json"
        config.write_text("{}")
        arguments_log = self.root / "docker-arguments.jsonl"
        docker = self.bin / "docker"
        docker.write_text(
            "#!" + sys.executable + "\nimport json, os, sys\n"
            "with open(os.environ['DOCKER_ARGUMENTS_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        docker.chmod(0o755)
        socket_path = self.root / "docker.sock"
        with socket.socket(socket.AF_UNIX) as fake_socket:
            fake_socket.bind(str(socket_path))
            for flags in (["--observe-config", str(config)], ["--observe-config=" + str(config)]):
                with self.subTest(flags=flags):
                    arguments_log.unlink(missing_ok=True)
                    result = subprocess.run(
                        [str(wrapper), "wsp", "run", "http://example.invalid/v1", *flags],
                        cwd=self.root,
                        env={**self.env, "KAIRYU_BENCH_DOCKER_SOCKET": str(socket_path),
                             "KAIRYU_BENCH_CACHE_DIR": str(self.root / "cache"),
                             "DOCKER_ARGUMENTS_LOG": str(arguments_log)},
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in arguments_log.read_text().splitlines()]
                    self.assertEqual([call[0] for call in calls], ["build", "run"])
                    command = calls[1]
                    mounts = [command[i + 1] for i, argument in enumerate(command) if argument == "-v"]
                    self.assertIn(f"{capture.resolve()}:{capture.resolve()}", mounts)
                    self.assertEqual(command[-len(flags):], flags)

    def test_static_output_refuses_input_and_hardlink_alias(self) -> None:
        alias = self.root/"alias.json"
        os.link(self.model, alias)
        symbolic = self.root / "symbolic.json"
        symbolic.symlink_to(self.model)
        for output in (self.model, alias, symbolic):
            with self.subTest(output=output.name):
                result = self.cli("weights", "inspect", str(self.model), "--output", str(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("overwrite checkpoint input", result.stderr)
                self.assertEqual(self.model.read_bytes(), self.original)
                self.assertEqual(alias.read_bytes(), self.original)

    def test_static_output_cannot_create_model_metadata_or_replace_unreferenced_weights(self) -> None:
        extra = self.root / "unreferenced.safetensors"
        extra.write_bytes(self.original)
        for output in (self.root / "config.json", self.root / "model.safetensors.index.json", extra):
            with self.subTest(output=output.name):
                original = output.read_bytes() if output.exists() else None
                result = self.cli("weights", "inspect", str(self.model), "--output", str(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("alter checkpoint files", result.stderr)
                if original is None:
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(output.read_bytes(), original)
        self.assertEqual(self.model.read_bytes(), self.original)

    def test_partial_diff_is_an_explicit_cli_choice(self) -> None:
        candidate = self.root/"candidate.safetensors"
        header = json.dumps({"other": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        candidate.write_bytes(struct.pack("<Q", len(header))+header+struct.pack("<f", 1))
        result = self.cli("weights", "diff", str(self.model), str(candidate))
        self.assertNotEqual(result.returncode, 0)
        result = self.cli("weights", "diff", str(self.model), str(candidate), "--allow-partial", "--no-hash")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertFalse(data["coverage_complete"])
        self.assertEqual(data["status"], "no_comparable_tensors")
        self.assertIsNone(data["global"]["delta"]["l2"])

    def test_observe_output_cannot_replace_capture_or_report_inputs(self) -> None:
        capture = self.root/"capture"
        initialized = self.cli("observe", "init", str(self.model), "--output-dir", str(capture),
                               "--model", "synthetic", "--tp-size", "1", "--replicas", "1")
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        config = capture/"capture.json"
        original = config.read_bytes()
        attempted = self.cli("observe", "merge", str(capture), "--output", str(config))
        self.assertNotEqual(attempted.returncode, 0)
        self.assertEqual(config.read_bytes(), original)
        report = self.root/"report.json"
        merged = self.cli("observe", "merge", str(capture), "--output", str(report))
        self.assertEqual(merged.returncode, 0, merged.stderr)
        original_report = report.read_bytes()
        alias = self.root/"report-alias.html"
        os.link(report, alias)
        attempted = self.cli("observe", "view", str(report), "--output", str(alias))
        self.assertNotEqual(attempted.returncode, 0)
        self.assertEqual(report.read_bytes(), original_report)
        self.assertEqual(alias.read_bytes(), original_report)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
