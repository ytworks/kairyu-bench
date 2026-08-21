from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kairyu_bench.harness_data import export_selected_rows
from kairyu_bench.inspect_summary import summarize_inspect_log
from kairyu_bench.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


class OfficialShellSupportTest(unittest.TestCase):
    def test_swebench_pro_uses_its_official_harness_and_image_layout(self) -> None:
        harness = (ROOT / "scripts/harnesses/swebench.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("environment.cwd=/app", harness)
        self.assertIn("swe_bench_pro_eval.py", harness)
        self.assertIn("--use_local_docker", harness)
        self.assertIn('--expected-ids "$item_ids"', harness)
        self.assertIn("SWE-bench Pro completed $index/$total", harness)
        self.assertIn('docker image rm -f "$image"', harness)
        self.assertIn('environment.run_args=["--rm","--entrypoint",""]', harness)
        self.assertIn("ln -s /app /testbed", harness)
        self.assertIn("KAIRYU_BENCH_SWEBENCH_PRO_WORKERS", harness)
        self.assertIn('--output "$item_generation"', harness)
        self.assertIn("aggregate-items", harness)

    def test_terminal_bench_selects_each_supported_harbor_agent(self) -> None:
        entry = load_manifest()["terminal-bench"]
        revision = entry["source"]["revision"]
        cases = {
            "terminus-2": (
                "openai/chat-capable",
                "api_base=https://example.test/v1",
                [],
            ),
            "claude-code": (
                "chat-capable",
                None,
                [
                    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1",
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1",
                    "CLAUDE_CODE_ATTRIBUTION_HEADER=0",
                ],
            ),
            "codex": ("chat-capable", None, []),
        }
        for agent, (
            expected_model,
            expected_kwarg,
            expected_agent_env,
        ) in cases.items():
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache = root / "cache"
                run_dir = root / "run"
                result_path = run_dir / "normalized" / "terminal-bench.json"
                capture_path = root / "harbor.json"
                context_path = root / "context.json"
                context_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_id": "run-1",
                            "benchmark": "terminal-bench",
                            "endpoint_fingerprint": "sha256:test",
                            "model_id": "chat-capable",
                            "agent": agent,
                            "limit": 1,
                            "source": entry["source"],
                            "dataset": entry["dataset"],
                            "scoring": entry["scoring"],
                            "run_dir": str(run_dir),
                            "result_path": str(result_path),
                        }
                    ),
                    encoding="utf-8",
                )

                source = cache / "sources" / f"harbor-{revision}"
                source.mkdir(parents=True)
                (source / ".kairyu-bench-revision").write_text(
                    revision + "\n", encoding="utf-8"
                )
                environment = cache / "venvs" / f"v2-harbor-{revision}"
                (environment / "bin").mkdir(parents=True)
                (environment / ".kairyu-bench-ready").write_text(
                    f"{revision}\n{environment.resolve()}\n", encoding="utf-8"
                )
                harbor = environment / "bin" / "harbor"
                harbor.write_text(
                    """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
agent = args[args.index("--agent") + 1]
capture = {
    "args": args,
    "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
    "anthropic_base_url": os.environ.get("ANTHROPIC_BASE_URL"),
}
Path(os.environ["HARBOR_ARGS_PATH"]).write_text(json.dumps(capture))
jobs = Path(args[args.index("--jobs-dir") + 1])
trial = jobs / "task-a"
trial.mkdir(parents=True)
(trial / "config.json").write_text("{}")
(trial / "result.json").write_text(json.dumps({
    "task_name": "task-a",
    "agent_info": {"name": agent, "version": "1.0"},
    "verifier_result": {"rewards": {"reward": 1}},
}))
""",
                    encoding="utf-8",
                )
                harbor.chmod(0o755)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                docker = fake_bin / "docker"
                docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                docker.chmod(0o755)

                process_environment = {
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "PYTHONPATH": str(ROOT / "src"),
                    "KAIRYU_BENCH_CONTEXT": str(context_path),
                    "KAIRYU_BENCH_RESULT_PATH": str(result_path),
                    "KAIRYU_BENCH_RUN_DIR": str(run_dir),
                    "KAIRYU_BENCH_CACHE_DIR": str(cache),
                    "KAIRYU_ENDPOINT": "https://example.test/v1",
                    "KAIRYU_MODEL": "chat-capable",
                    "KAIRYU_HARBOR_AGENT": agent,
                    "OPENAI_BASE_URL": "https://example.test/v1",
                    "OPENAI_API_KEY": "super-secret",
                    "ANTHROPIC_BASE_URL": "https://example.test",
                    "ANTHROPIC_API_KEY": "super-secret",
                    "HARBOR_ARGS_PATH": str(capture_path),
                }
                subprocess.run(
                    [str(ROOT / "scripts/harnesses/terminal-bench.sh")],
                    cwd=ROOT,
                    env=process_environment,
                    check=True,
                )

                capture = json.loads(capture_path.read_text(encoding="utf-8"))
                arguments = capture["args"]
                self.assertEqual(arguments[arguments.index("--agent") + 1], agent)
                self.assertEqual(
                    arguments[arguments.index("--model") + 1], expected_model
                )
                if expected_kwarg is None:
                    self.assertNotIn("--agent-kwarg", arguments)
                else:
                    self.assertEqual(
                        arguments[arguments.index("--agent-kwarg") + 1],
                        expected_kwarg,
                    )
                actual_agent_env = [
                    arguments[index + 1]
                    for index, argument in enumerate(arguments)
                    if argument == "--agent-env"
                ]
                self.assertEqual(actual_agent_env, expected_agent_env)
                overlay = Path(
                    arguments[arguments.index("--extra-docker-compose") + 1]
                )
                self.assertEqual(
                    overlay, ROOT / "scripts/harnesses/harbor-host-gateway.yaml"
                )
                self.assertIn(
                    "host.docker.internal:host-gateway",
                    overlay.read_text(encoding="utf-8"),
                )
                self.assertNotIn("super-secret", arguments)
                self.assertEqual(capture["openai_base_url"], "https://example.test/v1")
                self.assertEqual(capture["anthropic_base_url"], "https://example.test")
                self.assertEqual(
                    json.loads(result_path.read_text(encoding="utf-8"))["agent"], agent
                )

    def test_terminal_bench_rejects_codex_model_ids_with_a_slash(self) -> None:
        entry = load_manifest()["terminal-bench"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            result_path = run_dir / "normalized/terminal-bench.json"
            context_path = root / "context.json"
            context_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run-slash-model",
                        "benchmark": "terminal-bench",
                        "endpoint_fingerprint": "sha256:test",
                        "model_id": "organization/chat-capable",
                        "agent": "codex",
                        "limit": 1,
                        "source": entry["source"],
                        "dataset": entry["dataset"],
                        "scoring": entry["scoring"],
                        "run_dir": str(run_dir),
                        "result_path": str(result_path),
                    }
                ),
                encoding="utf-8",
            )
            process = subprocess.run(
                [str(ROOT / "scripts/harnesses/terminal-bench.sh")],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "src"),
                    "KAIRYU_BENCH_CONTEXT": str(context_path),
                    "KAIRYU_BENCH_RESULT_PATH": str(result_path),
                    "KAIRYU_BENCH_RUN_DIR": str(run_dir),
                    "KAIRYU_BENCH_CACHE_DIR": str(root / "cache"),
                    "KAIRYU_ENDPOINT": "https://example.test/v1",
                    "KAIRYU_MODEL": "organization/chat-capable",
                    "KAIRYU_HARBOR_AGENT": "codex",
                },
                text=True,
                capture_output=True,
                check=False,
            )

            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(process.returncode, 3)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["agent"], "codex")
        self.assertIn("cannot preserve model IDs containing '/'", result["error"])

    def test_checkout_source_resolves_and_reuses_exact_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            subprocess.run(["git", "init", "-q", str(upstream)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(upstream),
                    "config",
                    "user.email",
                    "test@example.test",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(upstream), "config", "user.name", "Test"], check=True
            )
            (upstream / "version.txt").write_text("pinned\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(upstream), "add", "version.txt"], check=True
            )
            subprocess.run(
                ["git", "-C", str(upstream), "commit", "-q", "-m", "pinned"], check=True
            )
            revision = subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
            ).strip()
            cache = root / "cache"
            command = (
                '. "scripts/lib/official.sh"; '
                f'KAIRYU_BENCH_CACHE_DIR="{cache}"; '
                f'checkout_source demo "{upstream}" "{revision}"'
            )

            first = subprocess.check_output(
                ["sh", "-c", command], cwd=ROOT, text=True
            ).strip()
            second = subprocess.check_output(
                ["sh", "-c", command], cwd=ROOT, text=True
            ).strip()

            self.assertEqual(first, second)
            self.assertEqual((Path(first) / "version.txt").read_text(), "pinned\n")
            actual = subprocess.check_output(
                ["git", "-C", first, "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(actual, revision)

    def test_ensure_venv_preserves_console_scripts_after_publish_reuse_and_relocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            relocated_cache = root / "relocated-cache"
            executable_dir = root / "bin"
            executable_dir.mkdir()
            (executable_dir / "python").symlink_to(sys.executable)
            environment = os.environ.copy()
            environment["PATH"] = f"{executable_dir}{os.pathsep}{environment['PATH']}"
            command = (
                '. "scripts/lib/official.sh"; '
                f'KAIRYU_BENCH_CACHE_DIR="{cache}"; '
                'first=$(ensure_venv demo revision --help); '
                '"$first/bin/pip" --version; '
                'second=$(ensure_venv demo revision --help); '
                'test "$first" = "$second"; '
                '"$second/bin/pip" --version; '
                f'mv "{cache}" "{relocated_cache}"; '
                f'KAIRYU_BENCH_CACHE_DIR="{relocated_cache}"; '
                'third=$(ensure_venv demo revision --help); '
                'test "$second" != "$third"; '
                '"$third/bin/pip" --version'
            )

            completed = subprocess.run(
                ["sh", "-c", command],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        version_lines = completed.stdout.splitlines()
        self.assertEqual(len(version_lines), 3, completed.stdout)
        self.assertTrue(all(line.startswith("pip ") for line in version_lines))

    def test_context_get_reads_only_the_adapter_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context_path = Path(directory) / "context.json"
            context_path.write_text(
                json.dumps({"benchmark": "hle", "dataset": {"revision": "abc"}}),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["KAIRYU_BENCH_CONTEXT"] = str(context_path)

            output = subprocess.check_output(
                ["python3", "-m", "kairyu_bench.adapter_context", "dataset.revision"],
                cwd=ROOT,
                env={**environment, "PYTHONPATH": str(ROOT / "src")},
                text=True,
            )

        self.assertEqual(output.strip(), "abc")

    def test_every_manifest_adapter_has_executable_runner_and_normalizer(self) -> None:
        for name, entry in load_manifest().items():
            with self.subTest(benchmark=name):
                runner = ROOT / entry["adapter"]
                self.assertTrue(runner.is_file(), runner)
                self.assertTrue(os.access(runner, os.X_OK), runner)
                self.assertTrue((runner.parent / "normalize.py").is_file())

    def test_dataset_export_pins_revision_and_writes_canonical_prefix(self) -> None:
        context = {
            "limit": 2,
            "dataset": {"id": "owner/data", "revision": "deadbeef"},
        }
        rows = [
            {"instance_id": "one", "value": 1},
            {"instance_id": "two", "value": 2},
            {"instance_id": "three", "value": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selected.jsonl"
            ids = Path(directory) / "ids.txt"
            with mock.patch(
                "kairyu_bench.harness_data._load_dataset", return_value=rows
            ) as load:
                export_selected_rows(context, "test", "instance_id", output, ids)

            load.assert_called_once_with("owner/data", "deadbeef", "test", None)
            self.assertEqual(ids.read_text().splitlines(), ["one", "two"])
            self.assertEqual(
                [
                    json.loads(line)["instance_id"]
                    for line in output.read_text().splitlines()
                ],
                ["one", "two"],
            )

    def test_inspect_summary_preserves_scicode_official_metrics(self) -> None:
        inspect_log = {
            "status": "success",
            "results": {
                "scores": [
                    {
                        "name": "scicode_scorer",
                        "metrics": {
                            "Problem Correctness": {"value": 0.5},
                            "sub_problem_correctness": {"value": 0.75},
                        },
                    }
                ]
            },
            "samples": [{"id": "13"}, {"id": "62"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scicode.json"
            path.write_text(json.dumps(inspect_log), encoding="utf-8")

            summary = summarize_inspect_log(path)

        self.assertEqual(summary["problem_ids"], ["13", "62"])
        self.assertEqual(summary["metrics"]["Problem Correctness"], 0.5)
        self.assertEqual(summary["metrics"]["sub_problem_correctness"], 0.75)


if __name__ == "__main__":
    unittest.main()
