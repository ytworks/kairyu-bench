from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kairyu_bench.manifest import load_manifest
from kairyu_bench.terminal_bench4 import resolve_terminal_bench4_conditions

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/harnesses/terminal-bench-4.sh"

FAKE_HARBOR = """#!{python}
import json, os, sys
from pathlib import Path
from kairyu_bench.terminal_bench4 import build_trial_plan, load_lock

args = sys.argv[1:]
if args == ["--version"]:
    print("0.24.0")
    raise SystemExit(0)
capture = {{
    "args": args,
    "has_hf_token": "HF_TOKEN" in os.environ,
    "has_kairyu_key": "KAIRYU_API_KEY" in os.environ,
    "telemetry": os.environ.get("HARBOR_TELEMETRY"),
}}
Path(os.environ["HARBOR_CAPTURE"]).write_text(json.dumps(capture))
config = json.loads(Path(args[args.index("--config") + 1]).read_text())
raw = Path(args[args.index("--plugin-kwarg") + 1].split("=", 1)[1])
selected = json.loads((raw / "selected.json").read_text())
lock = load_lock()
trials = [
    {{"task": {{"name": task, "ref": lock["tasks"][task]["digest"]}},
      "trial_name": task.split("/")[-1] + "__" + str(attempt)}}
    for attempt in range(config["n_attempts"]) for task in selected["problem_ids"]
]
plan = build_trial_plan(trials, config["job_name"])
(raw / "trial-plan.json").write_text(json.dumps(plan))
agent = config["agents"][0]
for trial, slot in zip(trials, plan["slots"]):
    path = raw / slot["result_path"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({{
        "id": slot["trial_name"], "task_name": slot["task"], "trial_name": slot["trial_name"],
        "config": {{**trial, "timeout_multiplier": 1.0, "agent": agent, "environment": {{}}, "verifier": {{}}}},
        "agent_info": {{"name": agent["name"], "version": "2.1.231"}},
        "verifier_result": {{"rewards": {{"reward": 1.0}}}},
        "finished_at": "2026-10-11T00:00:00Z",
    }}))
    (path.parent / "agent.log").write_text("key=" + os.environ.get("ANTHROPIC_API_KEY", ""))
raise SystemExit(int(os.environ.get("FAKE_HARBOR_EXIT", "0")))
"""


class TerminalBench4HarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.run_dir = self.root / "run"
        self.result_path = self.run_dir / "normalized" / "terminal-bench-4.json"
        self.capture = self.root / "harbor.json"
        entry = load_manifest()["terminal-bench-4"]
        context: dict[str, Any] = {
            "schema_version": 1,
            "run_id": "run-1",
            "benchmark": "terminal-bench-4",
            "endpoint": "https://example.test/v1",
            "endpoint_fingerprint": "sha256:test",
            "model_id": "kairyu-auto",
            "agent": "claude-code",
            "limit": 1,
            "source": entry["source"],
            "dataset": entry["dataset"],
            "scoring": entry["scoring"],
            "run_dir": str(self.run_dir),
            "result_path": str(self.result_path),
            "conditions": resolve_terminal_bench4_conditions(entry, {}, "claude-code"),
        }
        self.context_path = self.root / "context.json"
        self.context_path.write_text(json.dumps(context))
        self.install_fake_environment(entry["source"]["revision"])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def install_fake_environment(self, revision: str) -> None:
        lock = ROOT / "scripts/locks/terminal-bench-4-harbor.txt"
        key = revision + "-" + hashlib.sha256(lock.read_bytes()).hexdigest()
        environment = self.root / "cache" / "venvs" / f"v2-terminal-bench-4-{key}"
        (environment / "bin").mkdir(parents=True)
        (environment / ".kairyu-bench-ready").write_text(f"{key}\n{environment.resolve()}\n")
        harbor = environment / "bin" / "harbor"
        harbor.write_text(FAKE_HARBOR.format(python=sys.executable))
        python = environment / "bin" / "python"
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        self.bin = self.root / "bin"
        self.bin.mkdir()
        docker = self.bin / "docker"
        docker.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = info ]; then echo "{\\"NCPU\\": ${FAKE_NCPU:-32}, \\"Runtimes\\": {\\"runc\\": {}}}"; fi\n'
        )
        python_bin = self.bin / "python"
        python_bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        for path in (harbor, python, docker, python_bin):
            path.chmod(0o755)

    def run_harness(self, **extra: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "PYTHONPATH": str(ROOT / "src"),
            "KAIRYU_BENCH_CONTEXT": str(self.context_path),
            "KAIRYU_BENCH_RESULT_PATH": str(self.result_path),
            "KAIRYU_BENCH_RUN_DIR": str(self.run_dir),
            "KAIRYU_BENCH_CACHE_DIR": str(self.root / "cache"),
            "HARBOR_CAPTURE": str(self.capture),
            "HF_TOKEN": "hf-secret",
            "KAIRYU_API_KEY": "k-secret",
            "ANTHROPIC_API_KEY": "k-secret",
            "OPENAI_API_KEY": "k-secret",
            **extra,
        }
        return subprocess.run(
            [str(HARNESS)], env=env, text=True, capture_output=True, check=False
        )

    def result(self) -> dict[str, Any]:
        return json.loads(self.result_path.read_text())

    def test_runs_the_official_cli_with_the_plugin_and_normalizes(self) -> None:
        completed = self.run_harness()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = self.result()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["score"]["primary"], 100.0)
        self.assertEqual(result["agent"], "claude-code")
        self.assertEqual(result["conditions"]["attempts_per_task"], 1)
        capture = json.loads(self.capture.read_text())
        raw = self.run_dir / "raw" / "terminal-bench-4"
        self.assertEqual(
            capture["args"],
            [
                "run",
                "--config",
                str(raw / "job-config.json"),
                "--plugin",
                "kairyu_bench.terminal_bench4_plugin:KairyuTB4Plugin",
                "--plugin-kwarg",
                f"raw_dir={raw}",
                "--yes",
                "--quiet",
            ],
        )
        self.assertFalse(capture["has_hf_token"])
        self.assertFalse(capture["has_kairyu_key"])
        self.assertEqual(capture["telemetry"], "0")
        logs = list(raw.glob("jobs/*/*/agent.log"))
        self.assertEqual([path.read_text() for path in logs], ["key=[REDACTED]"])
        self.assertTrue((raw / "summary.json").is_file())

    def test_preserves_partial_results_when_harbor_fails(self) -> None:
        completed = self.run_harness(FAKE_HARBOR_EXIT="1")

        self.assertEqual(completed.returncode, 3, completed.stderr)
        result = self.result()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["score"]["metrics"]["runtime_errors"], ["runtime-error.json"])
        self.assertIn("preserving partial official results", completed.stderr)

    def test_reports_an_unsupported_daemon(self) -> None:
        completed = self.run_harness(FAKE_NCPU="1")

        self.assertEqual(completed.returncode, 3, completed.stderr)
        result = self.result()
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("CPUs; found 1", result["error"])
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
