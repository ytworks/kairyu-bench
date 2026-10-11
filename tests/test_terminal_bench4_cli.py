from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from kairyu_bench.manifest import load_manifest
from kairyu_bench.terminal_bench4 import load_lock, resolve_terminal_bench4_conditions
from kairyu_bench.terminal_bench4_cli import (
    Unsupported,
    check_daemon,
    finalize,
    gateway_address,
    main,
    prepare,
)

INFO = {"ServerVersion": "27.5.1", "NCPU": 32, "MemTotal": 1, "Runtimes": {"runc": {}}}


def context(limit: int | None = 1, attempts: str = "1") -> dict[str, Any]:
    entry = load_manifest()["terminal-bench-4"]
    return {
        "run_id": "run-1",
        "benchmark": "terminal-bench-4",
        "endpoint": "http://host.docker.internal:8003/v1",
        "model_id": "kairyu-auto",
        "agent": "claude-code",
        "limit": limit,
        "source": entry["source"],
        "dataset": entry["dataset"],
        "scoring": entry["scoring"],
        "conditions": resolve_terminal_bench4_conditions(
            entry, {"KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS": attempts}, "claude-code"
        ),
    }


class GatewayTest(unittest.TestCase):
    def test_resolves_the_runner_host_alias_for_nested_containers(self) -> None:
        self.assertEqual(
            gateway_address("http://host.docker.internal:8003/v1", lambda _: "172.17.0.1"),
            "172.17.0.1",
        )
        self.assertEqual(
            gateway_address("https://api.example.test/v1", lambda _: "unused"), "host-gateway"
        )
        with self.assertRaises(ValueError):
            gateway_address("http://host.docker.internal:8003/v1", lambda _: "not-an-ip")


class DaemonCheckTest(unittest.TestCase):
    def test_requires_enough_cpus_for_the_largest_selected_task(self) -> None:
        lock = load_lock()
        big = max(lock["tasks"], key=lambda task: lock["tasks"][task]["cpus"])

        with self.assertRaisesRegex(Unsupported, "at least 16 CPUs; found 8"):
            check_daemon({**INFO, "NCPU": 8}, lock, [big], lambda: True)

    def test_offline_verifiers_require_egress_control(self) -> None:
        lock = load_lock()
        offline = lock["offline_verifier_tasks"][:1]

        with self.assertRaisesRegex(Unsupported, "nftables egress control"):
            check_daemon(INFO, lock, offline, lambda: False)
        self.assertTrue(check_daemon(INFO, lock, offline, lambda: True)["egress_control_supported"])

    def test_detects_the_nvidia_runtime(self) -> None:
        lock = load_lock()
        task = sorted(lock["tasks"])[:1]

        self.assertFalse(check_daemon(INFO, lock, task, lambda: True)["gpu_available"])
        with_gpu = {**INFO, "Runtimes": {"nvidia": {}, "runc": {}}}
        self.assertTrue(check_daemon(with_gpu, lock, task, lambda: True)["gpu_available"])


class PrepareFinalizeTest(unittest.TestCase):
    def test_prepare_writes_the_selection_config_overlay_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)

            runtime = prepare(
                context(), raw, INFO, lambda: True, "0.24.0", lambda _: "172.17.0.1",
                cache_dir=Path("/cache"),
            )

            selected = json.loads((raw / "selected.json").read_text())
            config = json.loads((raw / "job-config.json").read_text())
            overlay = json.loads((raw / "host-gateway.json").read_text())
            instance_ids = (raw / "instance-ids.txt").read_text().splitlines()

        first = sorted(load_lock()["tasks"])[0]
        self.assertEqual(selected["problem_ids"], [first])
        self.assertEqual(selected["model_name"], "kairyu-auto")
        self.assertEqual(selected["job_name"], "run-1-terminal-bench-4")
        self.assertEqual(instance_ids, [first])
        self.assertEqual(config["datasets"][0]["task_names"], [first])
        self.assertEqual(config["datasets"][0]["download_dir"], "/cache/harbor-tasks")
        self.assertEqual(
            overlay, {"services": {"main": {"extra_hosts": ["host.docker.internal:172.17.0.1"]}}}
        )
        self.assertFalse(runtime["gpu_available"])
        self.assertEqual(runtime["harbor_version"], "0.24.0")

    def test_prepare_rejects_another_harbor_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "needs Harbor 0.24.0"):
                prepare(context(), Path(directory), INFO, lambda: True, "0.22.0")

    def test_finalize_redacts_secrets_and_writes_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            prepare(context(), raw, INFO, lambda: True, "0.24.0", lambda _: "172.17.0.1")
            (raw / "jobs").mkdir()
            (raw / "jobs" / "log.txt").write_text("Authorization: Bearer secret-key")

            finalize(raw, {"ANTHROPIC_API_KEY": "secret-key"})

            self.assertEqual((raw / "jobs" / "log.txt").read_text(), "Authorization: Bearer [REDACTED]")
            summary = json.loads((raw / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["errors"], [{"type": "MissingTrialPlan"}])

    def test_main_reports_unsupported_daemons_with_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            context_path = Path(directory) / "context.json"
            context_path.write_text(json.dumps(context(limit=None)))
            with mock.patch.dict(os.environ, {"KAIRYU_BENCH_CONTEXT": str(context_path)}), mock.patch(
                "kairyu_bench.terminal_bench4_cli.docker_info", return_value={**INFO, "NCPU": 4}
            ):
                code = main(["prepare", str(raw), "--harbor-version", "0.24.0"])

            self.assertEqual(code, 10)
            self.assertIn("at least 16 CPUs", (raw / "unsupported.txt").read_text())

    def test_main_records_runtime_errors_for_foreign_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            context_path = Path(directory) / "context.json"
            context_path.write_text(json.dumps({**context(), "benchmark": "deepswe"}))
            with mock.patch.dict(os.environ, {"KAIRYU_BENCH_CONTEXT": str(context_path)}):
                code = main(["prepare", str(raw)])

            self.assertEqual(code, 2)
            self.assertEqual(
                json.loads((raw / "runtime-error.json").read_text()),
                {"type": "ValueError", "stage": "prepare"},
            )


if __name__ == "__main__":
    unittest.main()
