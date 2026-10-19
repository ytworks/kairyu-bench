"""DeepSeek Harness agent for DeepSWE.

The runner tests need nothing. The Pier tests need the locked DeepSWE runtime.
The wire test needs a separate venv with the locked dsh wheels, named by
``KAIRYU_BENCH_DSH_PYTHON``; it runs the real runtime against a fake proxy.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import shlex
import socket
import socketserver
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from kairyu_bench import deepswe_dsh_runner as runner
from kairyu_bench.deepswe import resolve_deepswe_conditions
from kairyu_bench.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src/kairyu_bench/deepswe_dsh_runner.py"
PATCH = ROOT / "src/kairyu_bench/data/deepswe-deepseek-harness.patch.json"
PROXY_USER, PROXY_TOKEN = "agent", "proxy-secret"
DSH_ENV = {
    "KAIRYU_BENCH_DEEPSWE_AGENT": "deepseek-harness",
    "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144",
}
API_ENV = {
    "OPENAI_BASE_URL": "http://host.docker.internal:8003/v1",
    "OPENAI_API_KEY": "fake-secret",
}


class RunnerClassificationTest(unittest.TestCase):
    def test_finished_turns_are_graded_and_failures_keep_their_code(self):
        self.assertEqual(runner.classify("completed", None), "completed")
        self.assertEqual(runner.classify("max-tokens", None), "max-tokens")
        self.assertEqual(
            runner.classify("error", {"code": "CONTEXT_WINDOW_EXCEEDED"}),
            "ContextWindowExceededError",
        )
        self.assertEqual(runner.classify("error", {"code": "SERVER"}), "provider:SERVER")
        self.assertEqual(runner.classify("error", None), "provider:UNKNOWN")
        self.assertEqual(runner.classify("aborted", None), "aborted")
        self.assertEqual(runner.classify(None, None), "MissingTurnEnd")

    def test_summary_sums_usage_and_keeps_only_error_routing_fields(self):
        events = [
            {"type": "assistant/message", "data": {"usage": {"inputTokens": 7, "outputTokens": 3}}},
            {"type": "tool/call", "data": {}},
            {
                "type": "assistant/message",
                "data": {"usage": {"inputTokens": 11, "outputTokens": 2, "cacheReadTokens": 5}},
            },
            {"type": "compaction/summary", "data": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
            {
                "type": "turn/end",
                "data": {
                    "reason": {
                        "kind": "error",
                        "error": {"code": "SERVER", "status": 503, "message": "busy", "requestId": "r"},
                    }
                },
            },
        ]
        summary = runner.summarize(events)
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual(
            summary["usage"],
            {"input_tokens": 19, "output_tokens": 6, "cache_read_tokens": 5},
        )
        self.assertEqual(summary["finish_reason"], "error")
        self.assertEqual(summary["error"], {"code": "SERVER", "status": 503, "message": "busy"})


def dsh_context(env: dict[str, str] | None = None) -> dict:
    entry = load_manifest()["deepswe"]
    return {
        "run_id": "test",
        "model_id": "org/model",
        "generator": entry["generator"],
        "source": entry["source"],
        "dataset": entry["dataset"],
        "conditions": resolve_deepswe_conditions(entry, {**DSH_ENV, **(env or {})}),
    }


@unittest.skipUnless(
    importlib.util.find_spec("pier"), "requires the locked DeepSWE runtime"
)
class DeepSeekHarnessAgentTest(unittest.TestCase):
    def agent(self, root: Path, env: dict[str, str] | None = None):
        from kairyu_bench.deepswe_dsh import KairyuDeepSeekHarnessAgent
        from kairyu_bench.deepswe_runtime import build_job_config

        config = build_job_config(dsh_context(env), root, root, ["task-a"], 0)
        agent = config.agents[0]
        return config, KairyuDeepSeekHarnessAgent(
            logs_dir=root, model_name=agent.model_name, **agent.kwargs
        )

    def test_job_selects_the_locked_agent_without_credentials_or_new_timeouts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, API_ENV):
            config, agent = self.agent(Path(directory), {"KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT": "max"})
        self.assertEqual(
            config.agents[0].import_path,
            "kairyu_bench.deepswe_dsh:KairyuDeepSeekHarnessAgent",
        )
        self.assertEqual(config.agents[0].model_name, "deepseek-official/org/model")
        self.assertEqual(
            config.agents[0].kwargs,
            {"context_window": 262144, "max_tokens": 49152, "reasoning_effort": "max"},
        )
        self.assertNotIn("fake-secret", config.model_dump_json())
        self.assertEqual(config.timeout_multiplier, 1)
        self.assertIsNone(config.agent_timeout_multiplier)
        self.assertIsNone(config.agents[0].override_timeout_sec)
        self.assertIn("TransientAgentError", config.retry.include_exceptions)
        info = agent.to_agent_info()
        self.assertEqual(info.name, "deepseek-harness")
        self.assertEqual(info.version, "0.2.1rc2")
        self.assertEqual(info.model_info.name, "org/model")
        self.assertEqual(info.model_info.provider, "deepseek-official")
        with tempfile.TemporaryDirectory() as directory:
            self.assertNotIn("reasoning_effort", self.agent(Path(directory))[0].agents[0].kwargs)

    def test_installer_uses_hash_locked_wheels_and_absolute_paths(self):
        from pier.agents.installed.antigravity_sdk import _UV_VERSION
        from kairyu_bench.deepswe import DSH_LOCK

        lock = json.loads(DSH_LOCK.read_text())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, API_ENV):
            spec = self.agent(Path(directory))[1].install_spec()
        root, verify = spec.steps
        self.assertEqual(lock["uv"], _UV_VERSION)
        self.assertEqual((root.user, verify.user), ("root", "agent"))
        self.assertEqual(spec.agent_name, "deepseek-harness")
        self.assertEqual(spec.version, "0.2.1rc2")
        self.assertIn("sha256sum -c -", root.run)
        self.assertIn("--require-hashes --no-deps --only-binary :all:", root.run)
        self.assertIn("UV_PYTHON_INSTALL_DIR=/installed-agent/python", root.run)
        self.assertIn("uv venv --python 3.12 /installed-agent/venv", root.run)
        for requirement in lock["requirements"]:
            self.assertIn(requirement, root.run)
        for executable in lock["runtime_executables"]:
            self.assertIn(
                f"/installed-agent/venv/lib/python3.12/site-packages/{executable}", root.run
            )
        self.assertNotIn(" cd ", root.run)
        self.assertIn("/installed-agent/venv/bin/dsh --version", verify.run)

    def test_allowlist_names_only_the_api_host(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, API_ENV):
            agent = self.agent(Path(directory))[1]
            self.assertEqual(agent.network_allowlist().domains, ["host.docker.internal"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENAI_BASE_URL": "http://user:pass@example.test/v1"}),
        ):
            with self.assertRaisesRegex(ValueError, "embedded credentials"):
                self.agent(Path(directory))[1].network_allowlist()

    def test_setup_uploads_runner_and_patch_after_the_base_setup(self):
        environment = AsyncMock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, API_ENV):
            agent = self.agent(Path(directory))[1]
            with patch.object(agent, "exec_as_root", new_callable=AsyncMock) as execute:
                from pier.agents.installed.base import BaseInstalledAgent

                with patch.object(BaseInstalledAgent, "setup", new_callable=AsyncMock):
                    asyncio.run(agent.setup(environment))
        targets = [call.args[1] for call in environment.upload_file.call_args_list]
        sources = [Path(call.args[0]).name for call in environment.upload_file.call_args_list]
        self.assertEqual(
            targets,
            ["/installed-agent/kairyu_dsh_runner.py", "/installed-agent/kairyu-dsh-minimal.patch.json"],
        )
        self.assertEqual(
            sources, ["deepswe_dsh_runner.py", "deepswe-deepseek-harness.patch.json"]
        )
        self.assertIn("chmod a+r", execute.call_args.args[1])

    def test_run_passes_settings_by_environment_and_maps_exit_statuses(self):
        from pier.models.agent.context import AgentContext
        from kairyu_bench.deepswe_runtime import TransientAgentError

        for status, transient in (
            ("completed", False),
            ("provider:SERVER", True),
            ("provider:INVALID_REQUEST", False),
            (None, False),
        ):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, API_ENV),
            ):
                root = Path(directory)
                agent = self.agent(root, {"KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT": "low"})[1]

                async def execute(*_args, **_kwargs):
                    if status is not None:
                        (root / "dsh-result.json").write_text(json.dumps({
                            "status": status, "model_calls": 3,
                            "usage": {"input_tokens": 20, "output_tokens": 4, "cache_read_tokens": 2},
                        }))

                context = AgentContext()
                with patch.object(agent, "exec_as_agent", side_effect=execute) as call:
                    if transient:
                        with self.assertRaises(TransientAgentError):
                            asyncio.run(agent.run("Fix it; don't stop.", object(), context))
                    else:
                        asyncio.run(agent.run("Fix it; don't stop.", object(), context))
                command = call.call_args.kwargs["command"]
                env = call.call_args.kwargs["env"]
                self.assertEqual(
                    context.metadata["dsh_exit_status"], status or "MissingAgentResult"
                )
                self.assertIsNone(context.cost_usd)
                if status is not None:
                    self.assertEqual(
                        (context.n_input_tokens, context.n_output_tokens, context.n_agent_steps),
                        (20, 4, 3),
                    )
                self.assertNotIn("fake-secret", command)
                words = shlex.split(command.split(" 2>&1")[0])
                self.assertEqual(
                    words[:4],
                    ["/installed-agent/venv/bin/python", "/installed-agent/kairyu_dsh_runner.py",
                     "--instruction", "Fix it; don't stop."],
                )
                self.assertIn("--patch", words)
                self.assertEqual(
                    {key: env[key] for key in (
                        "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "DSH_CONTEXT_WINDOW",
                        "KAIRYU_DSH_MODEL", "KAIRYU_DSH_MAX_TOKENS", "KAIRYU_DSH_REASONING_EFFORT",
                    )},
                    {
                        "DEEPSEEK_BASE_URL": "http://host.docker.internal:8003/v1",
                        "DEEPSEEK_API_KEY": "fake-secret",
                        "DSH_CONTEXT_WINDOW": "262144",
                        "KAIRYU_DSH_MODEL": "org/model",
                        "KAIRYU_DSH_MAX_TOKENS": "49152",
                        "KAIRYU_DSH_REASONING_EFFORT": "low",
                    },
                )

    def test_actual_pier_queue_records_the_dsh_agent_and_its_lock(self):
        from datetime import datetime, timezone
        from pier.models.trial.result import TrialResult
        from pier.trial.hooks import TrialEvent
        from pier.trial.trial import Trial
        from kairyu_bench.deepswe import read_object
        from kairyu_bench.deepswe_runtime import run

        context = dsh_context({"KAIRYU_BENCH_DEEPSWE_WORKERS": "2"})
        context["dataset"] = dict(context["dataset"], tasks=2)

        async def fake_execution(trial):
            await trial._invoke_hooks(TrialEvent.START)
            now = datetime.now(timezone.utc)
            trial._result = TrialResult.model_validate({
                "task_name": trial._task.name,
                "trial_name": trial.config.trial_name,
                "trial_uri": trial.trial_dir.as_uri(),
                "task_id": trial.config.task.get_task_id(),
                "task_checksum": trial._task.checksum,
                "config": trial.config,
                "agent_info": trial._agent.to_agent_info(),
                "started_at": now,
                "finished_at": now,
                "agent_result": {"metadata": {"dsh_exit_status": "completed"}},
                "verifier_result": {"rewards": {"reward": int(trial._task.name.endswith("task-a"))}},
            })
            trial._trial_paths.result_path.write_text(trial._result.model_dump_json())
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
            for task in ("task-a", "task-b"):
                path = task_root / task
                (path / "environment").mkdir(parents=True)
                (path / "environment/Dockerfile").write_text("FROM scratch\n")
                (path / "instruction.md").write_text("A synthetic orchestration fixture.")
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
            selected = read_object(raw / "selected.json")
            runtime = read_object(raw / "runtime.json")
        self.assertEqual(selected["agent"], "deepseek-harness")
        self.assertEqual(selected["agent_version"], "0.2.1rc2")
        self.assertEqual(summary["agent"], "deepseek-harness")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual((summary["scored_attempts"], summary["passed_attempts"]), (2, 1))
        self.assertEqual(runtime["packages"]["deepseek-harness-sdk"], "0.2.1rc2")
        self.assertEqual(runtime["packages"]["deepseek-harness-runtime-bin"], "0.2.1rc2")
        self.assertNotIn("mini-swe-agent", runtime["packages"])
        self.assertEqual(runtime["agent_lock_sha256"], context["conditions"]["agent_lock_sha256"])


def _sse(events: list[dict]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


def _message(index: int, model: str) -> bytes:
    start = {
        "type": "message_start",
        "message": {
            "id": f"msg_{index}", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    }
    if index == 0:
        command = (
            "echo wire > done.txt && git add done.txt && "
            "git -c user.name=t -c user.email=t@example.test commit -qm wire"
        )
        block = {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps({"command": command})}
        stop = "tool_use"
    else:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": "Committed."}
        stop = "end_turn"
    return _sse(
        [
            start,
            {"type": "content_block_start", "index": 0, "content_block": block},
            {"type": "content_block_delta", "index": 0, "delta": delta},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    )


class FakeAnthropicProxy(socketserver.ThreadingTCPServer):
    """An authenticating forward proxy that answers Messages requests itself."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.lock = threading.Lock()
        self.tunnels: list[dict] = []
        self.requests: list[dict] = []


class _ProxyHandler(socketserver.StreamRequestHandler):
    server: FakeAnthropicProxy

    def _head(self) -> tuple[str, dict[str, str]] | None:
        line = self.rfile.readline().decode("latin-1").rstrip("\r\n")
        if not line:
            return None
        headers = {}
        while True:
            header = self.rfile.readline().decode("latin-1").rstrip("\r\n")
            if not header:
                return line, headers
            name, _, value = header.partition(":")
            headers[name.strip().lower()] = value.strip()

    def _body(self, headers: dict[str, str]) -> bytes:
        if headers.get("transfer-encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0], 16)
                if size == 0:
                    self.rfile.readline()
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(headers.get("content-length", "0")))

    def handle(self) -> None:
        head = self._head()
        if head is None:
            return
        line, headers = head
        method, target, _ = line.split(" ", 2)
        expected = "Basic " + base64.b64encode(f"{PROXY_USER}:{PROXY_TOKEN}".encode()).decode()
        authorized = headers.get("proxy-authorization") == expected
        if method == "CONNECT":
            with self.server.lock:
                self.server.tunnels.append({"target": target, "authorized": authorized})
            if not authorized:
                self.wfile.write(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            head = self._head()
            if head is None:
                return
            line, headers = head
            method, target, _ = line.split(" ", 2)
        elif not authorized:
            self.wfile.write(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
            return
        body = self._body(headers)
        with self.server.lock:
            index = sum(r["path"].endswith("/messages") for r in self.server.requests)
            self.server.requests.append(
                {
                    "method": method,
                    "path": target,
                    "authorized": authorized,
                    "headers": headers,
                    "body": json.loads(body) if body else None,
                }
            )
        if not target.endswith("/v1/messages"):
            self.wfile.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            return
        payload = _message(min(index, 1), json.loads(body)["model"])
        self.wfile.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\nConnection: close\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        self.wfile.flush()


@unittest.skipUnless(
    os.environ.get("KAIRYU_BENCH_DSH_PYTHON"), "requires the locked dsh runtime venv"
)
class RealRuntimeWireTest(unittest.TestCase):
    def run_task(self, patch: Path) -> tuple[dict, str, list[dict]]:
        proxy = FakeAnthropicProxy()
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, logs, runtime_cwd = root / "app", root / "logs", root / "runtime-cwd"
            for path in (workspace, logs, runtime_cwd):
                path.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            proxy_url = f"http://{PROXY_USER}:{PROXY_TOKEN}@127.0.0.1:{proxy.server_address[1]}"
            env = {
                "PATH": os.environ["PATH"],
                "HOME": str(root),
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": "localhost,127.0.0.1",
                "no_proxy": "localhost,127.0.0.1",
                "DEEPSEEK_BASE_URL": "http://host.docker.internal:8003/v1",
                "DEEPSEEK_API_KEY": "fake-api-key",
                "DSH_CONTEXT_WINDOW": "131072",
                "DSH_TELEMETRY_MODE": "DISABLED",
                "KAIRYU_DSH_MODEL": "org/model",
                "KAIRYU_DSH_MAX_TOKENS": "4096",
                "KAIRYU_DSH_REASONING_EFFORT": "max",
            }
            completed = subprocess.run(
                [
                    os.environ["KAIRYU_BENCH_DSH_PYTHON"], str(RUNNER),
                    "--instruction", "Create done.txt and commit it.",
                    "--logs-dir", str(logs),
                    "--runtime-cwd", str(runtime_cwd),
                    "--patch", str(patch),
                ],
                cwd=workspace, env=env, capture_output=True, text=True, timeout=300,
            )
            output = completed.stdout + completed.stderr
            self.assertEqual(completed.returncode, 0, output)
            result = json.loads((logs / "dsh-result.json").read_text())
            self.assertEqual(result["status"], "completed", output)
            self.assertTrue(list((logs / "dsh-home").rglob("*.jsonl")))
            self.assertTrue((logs / runner.EVENTS_FILE).read_text())
            log = subprocess.run(
                ["git", "-C", str(workspace), "log", "--format=%s"],
                capture_output=True, text=True, check=True,
            ).stdout
        return result, log.strip(), proxy.requests

    def test_minimal_profile_sends_plain_messages_through_the_authenticated_proxy(self):
        # The task container cannot resolve the API host; only the proxy can.
        with self.assertRaises(OSError):
            socket.getaddrinfo("host.docker.internal", 8003)
        result, log, requests = self.run_task(PATCH)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(log, "wire")
        self.assertEqual(len(requests), 2)
        for request in requests:
            body = request["body"]
            self.assertTrue(request["authorized"])
            self.assertEqual(request["method"], "POST")
            self.assertEqual(
                request["path"].removeprefix("http://host.docker.internal:8003"),
                "/v1/messages",
            )
            self.assertEqual(request["headers"]["host"], "host.docker.internal:8003")
            self.assertEqual(request["headers"]["x-api-key"], "fake-api-key")
            self.assertEqual(body["model"], "org/model")
            self.assertEqual(body["max_tokens"], 4096)
            self.assertTrue(body["stream"])
            self.assertEqual(body["output_config"], {"effort": "max"})
            self.assertNotIn("dsh_session_log", body)
            self.assertNotIn("dsh_plugin_packages", body)
            self.assertEqual(
                [tool["name"] for tool in body["tools"]],
                ["bash", "read_image", "working_directory"],
            )

    def test_patch_is_what_removes_the_deepseek_only_fields(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as empty:
            empty.write("[]")
            empty.flush()
            _, _, requests = self.run_task(Path(empty.name))
        self.assertTrue(
            any("dsh_plugin_packages" in request["body"] for request in requests)
        )


if __name__ == "__main__":
    unittest.main()
