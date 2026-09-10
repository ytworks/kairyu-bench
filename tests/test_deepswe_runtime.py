"""Run with the locked Pier environment; no Docker or paid API is needed."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kairyu_bench.manifest import load_manifest
from kairyu_bench.deepswe import resolve_deepswe_conditions


@unittest.skipUnless(
    importlib.util.find_spec("pier"), "requires the locked DeepSWE runtime"
)
class DeepSWERuntimeTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("kairyu_bench.deepswe_runtime"))
        return importlib.import_module("kairyu_bench.deepswe_runtime")

    def test_job_preserves_model_name_and_has_no_embedded_credentials(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        context = {
            "run_id": "test",
            "model_id": "org/model",
            "generator": entry["generator"],
            "conditions": resolve_deepswe_conditions(entry, {}),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "OPENAI_BASE_URL": "http://host.docker.internal:8003/v1",
                    "OPENAI_API_KEY": "fake-secret",
                },
            ),
        ):
            config = module.build_job_config(
                context, Path(directory), Path(directory), ["task-a"], 0
            )
        self.assertEqual(config.n_concurrent_trials, 4)
        self.assertEqual(config.n_attempts, 1)
        self.assertEqual(config.agents[0].model_name, "openai/org/model")
        self.assertEqual(config.agents[0].kwargs["model_class"], "litellm")
        self.assertEqual(config.agents[0].kwargs["version"], "2.4.6")
        self.assertNotIn("fake-secret", config.model_dump_json())
        self.assertNotIn("AgentTimeoutError", config.retry.include_exceptions)

    def test_installed_agent_allows_only_target_api_and_reports_actual_identity(self):
        module = self.module()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "OPENAI_BASE_URL": "http://host.docker.internal:8003/v1",
                },
            ),
        ):
            agent = module.KairyuMiniSweAgent(
                logs_dir=Path(directory),
                model_name="openai/org/model",
                version="2.4.6",
                model_class="litellm",
            )
            self.assertEqual(
                agent.network_allowlist().domains, ["host.docker.internal"]
            )
            self.assertEqual(agent.to_agent_info().model_info.name, "org/model")
            self.assertEqual(agent.to_agent_info().model_info.provider, "openai")
            self.assertEqual(agent.to_agent_info().name, "mini-swe-agent")

    def test_gateway_overlay_preserves_isolated_task_network(self):
        module = self.module()
        compose = {
            "services": {
                "main": {"networks": ["isolated"]},
                "pier-egress-proxy": {"networks": ["isolated", "egress"]},
            },
            "networks": {"isolated": {"internal": True}, "egress": {}},
        }
        result = module.add_host_gateway(compose, "172.17.0.1")
        self.assertEqual(result["networks"]["isolated"], {"internal": True})
        self.assertEqual(result["services"]["main"], {"networks": ["isolated"]})
        self.assertEqual(
            result["services"]["pier-egress-proxy"]["extra_hosts"],
            ["host.docker.internal:172.17.0.1"],
        )

    def test_unsafe_model_id_is_rejected_before_shell_execution(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        context = {
            "run_id": "test",
            "model_id": "model; touch /tmp/unwanted",
            "generator": entry["generator"],
            "conditions": resolve_deepswe_conditions(entry, {}),
        }
        with self.assertRaisesRegex(ValueError, "model ID"):
            module.build_job_config(context, Path("/tmp"), Path("/tmp"), ["task-a"], 0)

    def test_agent_exit_error_survives_official_trajectory_conversion(self):
        module = self.module()
        from pier.models.agent.context import AgentContext

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = module.KairyuMiniSweAgent(
                logs_dir=root, model_name="openai/org/model", version="2.4.6"
            )
            (root / "mini-swe-agent.trajectory.json").write_text(
                json.dumps(
                    {
                        "info": {
                            "exit_status": "ContextWindowExceededError",
                            "mini_version": "2.4.6",
                        },
                        "messages": [],
                    }
                )
            )
            context = AgentContext()
            agent.populate_context_post_run(context)
        self.assertEqual(
            context.metadata["mini_exit_status"], "ContextWindowExceededError"
        )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    importlib.util.find_spec("pier"), "requires the locked DeepSWE runtime"
)
class DeepSWETransportTest(unittest.TestCase):
    def test_proxy_permits_only_the_configured_api_port(self):
        from pier.environments.agent_setup import squid_bootstrap_command
        from kairyu_bench.deepswe_runtime import restrict_proxy_port

        script = restrict_proxy_port(
            squid_bootstrap_command(), "http://host.docker.internal:8003/v1"
        )
        self.assertIn("acl Safe_ports port 8003\n", script)
        self.assertIn("http_access allow authenticated allowed_domains", script)
        self.assertIn("http_access deny all", script)
        self.assertNotIn("acl Safe_ports port 80 443", script)

    def test_agent_installer_keeps_pinned_litellm_cost_map(self):
        from kairyu_bench.deepswe_runtime import KairyuMiniSweAgent

        with tempfile.TemporaryDirectory() as directory:
            agent = KairyuMiniSweAgent(
                logs_dir=Path(directory), model_name="openai/org/model", version="2.4.6"
            )
            text = "\n".join(step.run for step in agent.install_spec().steps)
        self.assertIn("uv tool install mini-swe-agent==2.4.6", text)
        self.assertNotIn("raw.githubusercontent.com/BerriAI/litellm/main", text)

    def test_real_litellm_sends_chat_completions_with_unmodified_model_id(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from minisweagent.models.litellm_model import LitellmModel

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body, self.headers.get("Authorization")))
                response = json.dumps(
                    {
                        "id": "fake",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "org/model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "bash",
                                                "arguments": '{"command":"true"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 2,
                            "total_tokens": 7,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                model = LitellmModel(
                    model_name="openai/org/model",
                    cost_tracking="ignore_errors",
                    model_kwargs={
                        "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                        "api_key": "fake-secret",
                    },
                )
                result = model.query([{"role": "user", "content": "run true"}])
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(len(requests), 1)
        path, body, auth = requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "org/model")
        self.assertEqual(body["tools"][0]["function"]["name"], "bash")
        self.assertEqual(auth, "Bearer fake-secret")
        self.assertEqual(result["extra"]["actions"][0]["command"], "true")

    def test_artifacts_redact_credentials(self):
        from kairyu_bench.deepswe_runtime import redact_artifacts

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENAI_API_KEY": "fake-secret"}),
        ):
            root = Path(directory)
            path = root / "trajectory.json"
            path.write_text('{"response": "fake-secret"}')
            redact_artifacts(root)
            self.assertNotIn("fake-secret", path.read_text())


@unittest.skipUnless(
    importlib.util.find_spec("pier"), "requires the locked DeepSWE runtime"
)
class DeepSWEJobTest(unittest.TestCase):
    def test_actual_pier_queue_refills_retries_and_preserves_four_repeats(self):
        import asyncio
        from collections import Counter
        from datetime import datetime, timezone
        from pier.models.trial.result import TrialResult
        from pier.trial.hooks import TrialEvent
        from pier.trial.trial import Trial
        from kairyu_bench.deepswe import read_object
        from kairyu_bench.deepswe_runtime import run

        entry = load_manifest()["deepswe"]
        env = {
            "KAIRYU_BENCH_DEEPSWE_ATTEMPTS": "4",
            "KAIRYU_BENCH_DEEPSWE_WORKERS": "2",
        }
        context = {
            "run_id": "fake-job",
            "model_id": "org/model",
            "generator": entry["generator"],
            "source": entry["source"],
            "dataset": dict(entry["dataset"], tasks=3),
            "conditions": resolve_deepswe_conditions(entry, env),
        }
        active = 0
        peak = 0
        calls = Counter()
        events = []

        async def fake_execution(trial):
            nonlocal active, peak
            task = trial._task.name
            calls[task] += 1
            active += 1
            peak = max(peak, active)
            events.append(("start", task))
            await trial._invoke_hooks(TrialEvent.START)
            await asyncio.sleep(0.03 if task.endswith("task-a") else 0.001)
            fail = task.endswith("task-b") and calls[task] == 1
            now = datetime.now(timezone.utc)
            payload = {
                "task_name": task,
                "trial_name": trial.config.trial_name,
                "trial_uri": trial.trial_dir.as_uri(),
                "task_id": trial.config.task.get_task_id(),
                "task_checksum": trial._task.checksum,
                "config": trial.config,
                "agent_info": trial._agent.to_agent_info(),
                "started_at": now,
                "finished_at": now,
                "agent_result": {"metadata": {"mini_exit_status": "Submitted"}},
                "verifier_result": {"rewards": {"reward": 1}},
            }
            if fail:
                payload["exception_info"] = {
                    "exception_type": "TransientAgentError",
                    "exception_message": "fake",
                    "exception_traceback": "",
                    "occurred_at": now,
                }
            trial._result = TrialResult.model_validate(payload)
            trial._trial_paths.result_path.write_text(trial._result.model_dump_json())
            (trial.trial_dir / "retry-evidence.txt").write_text("retained")
            events.append(("end", task))
            active -= 1
            await trial._invoke_hooks(TrialEvent.END)
            return trial._result

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "OPENAI_BASE_URL": "http://example.test:8003/v1",
                    "OPENAI_API_KEY": "fake-secret",
                    "KAIRYU_BENCH_CLEAN_TASK_IMAGES": "0",
                },
            ),
            patch.object(Trial, "run", fake_execution),
        ):
            root = Path(directory)
            task_root = root / "tasks"
            for task in ("task-a", "task-b", "task-c"):
                path = task_root / task
                (path / "environment").mkdir(parents=True)
                (path / "environment/Dockerfile").write_text("FROM scratch\n")
                (path / "instruction.md").write_text(
                    "A synthetic orchestration fixture, not a scored benchmark task."
                )
                (path / "task.toml").write_text(f"""schema_version = "1.3"
[task]
name = "datacurve/{task}"
[agent]
network_mode = "no-network"
[verifier]
network_mode = "no-network"
environment_mode = "separate"
[environment]
docker_image = "example.test/{task}:fixed"
""")
            raw = root / "raw"
            raw.mkdir()
            asyncio.run(run(context, raw, task_root))
            summary = read_object(raw / "summary.json")
            retries = list((raw / "retries").rglob("retry-evidence.txt"))
            self.assertEqual(len(retries), 1)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["scored_attempts"], 12)
            self.assertEqual(summary["passed_attempts"], 12)
            self.assertEqual(summary["evaluated_tasks"], 3)
            self.assertEqual(summary["pass_at_4_percent"], 100)
            self.assertEqual(len(list((raw / "jobs").glob("repeat-*/result.json"))), 4)
        self.assertEqual(peak, 2)
        self.assertEqual(sum(calls.values()), 13)
        # A retry keeps its problem's slot during backoff. Verify immediate
        # refill on the second (healthy) repetition, while task-a still runs.
        second_start = [
            i
            for i, event in enumerate(events)
            if event == ("start", "datacurve/task-a")
        ][1]
        second_events = events[second_start:]
        self.assertLess(
            second_events.index(("start", "datacurve/task-c")),
            second_events.index(("end", "datacurve/task-a")),
        )
