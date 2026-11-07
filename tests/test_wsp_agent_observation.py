from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from kairyu_bench.manifest import load_manifest
from kairyu_bench.runner import RunConfig, run_benchmarks
from kairyu_bench.target import Endpoint
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Context
from kairyu_bench.wsp.sources.agents import SELECTION_ENV, _batch

ROOT = Path(__file__).resolve().parents[1]


class AgentObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.capture = self.directory / "capture"
        self.capture.mkdir()
        self.config = self.capture / "capture.json"
        self.config.write_text(json.dumps({
            "schema_version": 1, "capture_id": "synthetic-agents",
            "output_dir": str(self.capture),
            "limits": {"storage_bytes": 2 * 1024**3},
        }))

    def records(self) -> list[dict]:
        return [json.loads(line) for path in self.capture.glob("requests-*.jsonl")
                for line in path.read_text().splitlines()]

    def test_mini_cli_registers_only_trial_metadata_and_valid_extra_headers(self) -> None:
        output = subprocess.check_output([
            sys.executable, "-m", "kairyu_bench.wsp.sources.agents",
            "--config", str(self.config), "--run-id", "run-one",
            "--item-id", "swe-bench-pro:instance-a", "--trial-id", "trial-1",
        ], text=True)
        headers = json.loads(output)
        self.assertRegex(headers["traceparent"], r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$")
        record, = self.records()
        self.assertEqual(record["request_scope"], "trial")
        self.assertEqual(record["item_id"], "swe-bench-pro:instance-a")
        self.assertEqual(record["run_id"], "run-one")
        self.assertEqual(record["attempt_index"], 0)
        self.assertNotIn("input_sha256", record)
        self.assertNotIn("messages", record)

    def test_harbor_setup_uses_official_task_uuid_and_preserves_native_kwargs(self) -> None:
        class NativeTerminus:
            def __init__(native, logs_dir, *, llm_call_kwargs, **kwargs):
                native.logs_dir = logs_dir
                # Mirrors the pinned public constructor's shallow outer copy.
                native.native_kwargs = dict(llm_call_kwargs)
                native.context_id = None
                native.setup_calls = 0

            async def setup(native, environment):
                native.setup_calls += 1

        module_name = "kairyu_bench.wsp.sources.terminal_bench_observe"
        fake = types.ModuleType("harbor.agents.terminus_2.terminus_2")
        fake.Terminus2 = NativeTerminus
        names = ["harbor", "harbor.agents", "harbor.agents.terminus_2"]
        modules = {name: types.ModuleType(name) for name in names}
        modules[fake.__name__] = fake
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            "KAIRYU_BENCH_WSP_OBSERVE_CONFIG": str(self.config),
            "KAIRYU_BENCH_WSP_OBSERVE_RUN_ID": "run-one",
            "KAIRYU_BENCH_TERMINAL_BENCH_TASKS": "task-a",
        }):
            sys.modules.pop(module_name, None)
            observed = importlib.import_module(module_name)
            self.addCleanup(sys.modules.pop, module_name, None)
            trial_dir = self.directory / "task-a__trial"
            logs = trial_dir / "agent"
            logs.mkdir(parents=True)
            (trial_dir / "config.json").write_text(json.dumps({
                "trial_name": trial_dir.name, "task": {"name": "terminal-bench/task-a"},
            }))
            kwargs = {"timeout": 1200, "extra_headers": {"x-native": "preserved"}}
            first = observed.ObservedTerminus2(logs, llm_call_kwargs=kwargs)
            first.context_id = "official-uuid-1"
            asyncio.run(first.setup(object()))
            second = observed.ObservedTerminus2(logs, llm_call_kwargs=kwargs)
            second.context_id = "official-uuid-retry"
            asyncio.run(second.setup(object()))
            self.assertEqual(first.setup_calls, 1)
            self.assertEqual(first.native_kwargs["timeout"], 1200)
            self.assertEqual(first.native_kwargs["extra_headers"]["x-native"], "preserved")
            self.assertNotIn("traceparent", kwargs["extra_headers"])
            self.assertNotEqual(first.observation_headers["traceparent"], second.observation_headers["traceparent"])
            rows = self.records()
            self.assertEqual([r["trial_id"] for r in rows], ["official-uuid-1", "official-uuid-retry"])
            self.assertTrue(all(r["attempt_index"] is None for r in rows))
            self.assertTrue(all(r["item_id"] == "terminal-bench:task-a" for r in rows))
            (trial_dir / "config.json").write_text(json.dumps({
                "trial_name": trial_dir.name, "task": {"name": "unselected-task"},
            }))
            third = observed.ObservedTerminus2(logs)
            third.context_id = "bad-trial"
            with self.assertRaisesRegex(ValueError, "unselected"):
                asyncio.run(third.setup(object()))
            self.assertEqual(third.setup_calls, 0)
            self.assertEqual(len(self.records()), 2)

    def test_parallel_agent_batches_use_private_environments(self) -> None:
        seen = {}
        chat = types.SimpleNamespace(base_url="http://example.invalid/v1", api_key="fixture")
        context = Context(chat=chat, logprob=None, run_dir=self.directory / "wsp",
                          observe_config=self.config, run_id="outer-run")
        before = dict(os.environ)

        def mocked_run(config, *_):
            seen[config.selected[0]] = dict(config.adapter_env)
            return types.SimpleNamespace(run_dir=self.directory)

        def invoke(source):
            item = Item(f"{source}:task-a", source, "dev", 10, 16, 1, "fixture", 1)
            return _batch(source)(context, [item], {item.id: {"name": "task-a"}}, 2)

        with patch("kairyu_bench.runner.run_benchmarks", mocked_run), \
             patch("kairyu_bench.wsp.sources.agents._terminal_bench_outcomes", return_value={}), \
             patch("kairyu_bench.wsp.sources.agents._swe_bench_pro_outcomes", return_value={}):
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(invoke, ("terminal-bench", "swe-bench-pro")))
        self.assertEqual(dict(os.environ), before)
        for source, environment in seen.items():
            self.assertEqual(environment[SELECTION_ENV[source]], "task-a")
            self.assertEqual(environment["KAIRYU_BENCH_WSP_OBSERVE_CONFIG"], str(self.config))
            self.assertEqual(environment["KAIRYU_BENCH_WSP_OBSERVE_RUN_ID"], "outer-run")
        self.assertNotIn("KAIRYU_BENCH_TERMINAL_BENCH_TASKS", seen["swe-bench-pro"])

    def test_runner_passes_adapter_env_without_mutation_or_reserved_override(self) -> None:
        adapter = self.directory / "adapter.sh"
        adapter.write_text("#!/bin/sh\nexit 2\n")
        adapter.chmod(0o755)
        entry = {**load_manifest()["terminal-bench"], "adapter": adapter.name}
        env = {"KAIRYU_BENCH_WSP_OBSERVE_CONFIG": str(self.config), "KAIRYU_MODEL": "must-not-win"}
        before = dict(os.environ)
        config = RunConfig(Endpoint.parse("http://example.invalid/v1"), ("terminal-bench",), 1,
                           self.directory / "results", "run-one", self.directory, adapter_env=env)
        client = types.SimpleNamespace(discover_chat_model=lambda: "actual-model")
        with patch("kairyu_bench.runner.subprocess.run", return_value=types.SimpleNamespace(returncode=2)) as run:
            run_benchmarks(config, client, {"terminal-bench": entry})
        passed = run.call_args.kwargs["env"]
        self.assertEqual(passed["KAIRYU_BENCH_WSP_OBSERVE_CONFIG"], str(self.config))
        self.assertEqual(passed["KAIRYU_MODEL"], "actual-model")
        self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
