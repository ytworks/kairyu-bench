from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kairyu_bench.manifest import load_manifest
from kairyu_bench.runner import RunConfig, run_benchmarks
from kairyu_bench.target import Endpoint


FAKE_ADAPTER = r"""#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path

context = json.loads(Path(os.environ["KAIRYU_BENCH_CONTEXT"]).read_text())
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
payload = {
    "schema_version": 1,
    "run_id": context["run_id"],
    "benchmark": context["benchmark"],
    "status": "completed",
    "endpoint": {"fingerprint": context["endpoint_fingerprint"]},
    "model_id": context["model_id"],
    "source": {
        "repository": context["source"]["repository"],
        "revision": context["source"]["revision"],
        "dataset": context["dataset"]["id"],
        "dataset_revision": context["dataset"]["revision"],
    },
    "selection": {"requested_limit": context["limit"], "problem_ids": ["q-1"]},
    "counts": {"requested": 1, "evaluated": 1},
    "score": {"primary": 100.0, "unit": "percent", "metrics": {"accuracy": 100.0}},
    "scoring": {"method": context["scoring"]["method"], "self_judged": False, "self_simulated": False},
    "artifacts": {"raw": [], "logs": ["logs/gpqa-diamond.log"]},
    "timestamps": {"started_at": now, "finished_at": now},
    "error": None,
}
Path(os.environ["KAIRYU_BENCH_RESULT_PATH"]).write_text(json.dumps(payload))
"""


CAPTURE_AGENT_ADAPTER = r"""#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path

context = json.loads(Path(os.environ["KAIRYU_BENCH_CONTEXT"]).read_text())
capture = {
    name: os.environ.get(name)
    for name in (
        "KAIRYU_HARBOR_AGENT",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
    )
}
Path(os.environ["KAIRYU_TEST_CAPTURE"]).write_text(json.dumps(capture))
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
payload = {
    "schema_version": 1,
    "run_id": context["run_id"],
    "benchmark": context["benchmark"],
    "status": "completed",
    "endpoint": {"fingerprint": context["endpoint_fingerprint"]},
    "model_id": context["model_id"],
    "source": {
        "repository": context["source"]["repository"],
        "revision": context["source"]["revision"],
        "dataset": context["dataset"]["id"],
        "dataset_revision": context["dataset"]["revision"],
    },
    "selection": {"requested_limit": context["limit"], "problem_ids": ["task-1"]},
    "counts": {"requested": 1, "evaluated": 1},
    "score": {"primary": 100.0, "unit": "percent", "metrics": {}},
    "scoring": {"method": context["scoring"]["method"], "self_judged": False, "self_simulated": False},
    "artifacts": {"raw": [], "logs": ["logs/terminal-bench.log"]},
    "timestamps": {"started_at": now, "finished_at": now},
    "error": None,
}
if os.environ.get("KAIRYU_TEST_OMIT_AGENT") != "1":
    payload["agent"] = os.environ.get("KAIRYU_TEST_RESULT_AGENT", context["agent"])
Path(os.environ["KAIRYU_BENCH_RESULT_PATH"]).write_text(json.dumps(payload))
"""


class _DiscoveredClient:
    def discover_chat_model(self) -> str:
        return "chat-capable"


class BenchmarkRunnerTest(unittest.TestCase):
    def test_run_discovers_model_invokes_selected_adapter_and_validates_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapters/gpqa-diamond/run.sh"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
            adapter.chmod(0o755)
            results = root / "results"
            config = RunConfig(
                endpoint=Endpoint.parse("https://secret-host.example/v1"),
                selected=("gpqa-diamond",),
                limit=1,
                results_root=results,
                run_id="run-success",
                app_root=root,
            )

            outcome = run_benchmarks(config, _DiscoveredClient(), load_manifest())

            self.assertEqual(outcome.exit_code, 0)
            self.assertEqual(outcome.model_id, "chat-capable")
            normalized = json.loads(
                (results / "run-success/normalized/gpqa-diamond.json").read_text()
            )
            self.assertEqual(normalized["score"]["primary"], 100.0)
            metadata_text = (results / "run-success/run.json").read_text()
            self.assertNotIn("secret-host.example", metadata_text)
            metadata = json.loads(metadata_text)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["benchmarks"], ["gpqa-diamond"])

    def test_missing_adapter_is_a_failed_result_and_nonzero_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("gpqa-diamond",),
                limit=None,
                results_root=root / "results",
                run_id="run-failed",
                app_root=root,
            )

            outcome = run_benchmarks(config, _DiscoveredClient(), load_manifest())

            self.assertEqual(outcome.exit_code, 3)
            failed = json.loads(
                (root / "results/run-failed/normalized/gpqa-diamond.json").read_text()
            )
            self.assertEqual(failed["status"], "failed")
            self.assertIn("adapter executable not found", failed["error"])
            metadata = json.loads((root / "results/run-failed/run.json").read_text())
            self.assertEqual(metadata["status"], "incomplete")

    def test_existing_run_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "results/run-existing"
            existing.mkdir(parents=True)
            marker = existing / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("gpqa-diamond",),
                limit=None,
                results_root=root / "results",
                run_id="run-existing",
                app_root=root,
            )

            with self.assertRaisesRegex(FileExistsError, "run-existing"):
                run_benchmarks(config, _DiscoveredClient(), load_manifest())

            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_run_passes_selected_harbor_agent_and_protocol_endpoints_to_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapters/terminal-bench/run.sh"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(CAPTURE_AGENT_ADAPTER, encoding="utf-8")
            adapter.chmod(0o755)
            capture = root / "agent-environment.json"
            config = RunConfig(
                endpoint=Endpoint.parse("https://secret-host.example/prefix/v1"),
                selected=("terminal-bench",),
                limit=1,
                results_root=root / "results",
                run_id="run-claude",
                app_root=root,
                harbor_agent="claude-code",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "KAIRYU_API_KEY": "secret-key",
                    "KAIRYU_TEST_CAPTURE": str(capture),
                },
            ):
                outcome = run_benchmarks(config, _DiscoveredClient(), load_manifest())

            self.assertEqual(outcome.exit_code, 0)
            context = json.loads(
                (root / "results/run-claude/context/terminal-bench.json").read_text()
            )
            self.assertEqual(context["agent"], "claude-code")
            normalized = json.loads(
                (root / "results/run-claude/normalized/terminal-bench.json").read_text()
            )
            self.assertEqual(normalized["agent"], "claude-code")
            metadata = json.loads(
                (root / "results/run-claude/run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["harbor_agent"], "claude-code")
            self.assertEqual(
                json.loads(capture.read_text()),
                {
                    "KAIRYU_HARBOR_AGENT": "claude-code",
                    "OPENAI_BASE_URL": "https://secret-host.example/prefix/v1",
                    "OPENAI_API_KEY": "secret-key",
                    "ANTHROPIC_BASE_URL": "https://secret-host.example/prefix",
                    "ANTHROPIC_API_KEY": "secret-key",
                },
            )

    def test_run_rejects_an_adapter_result_from_a_different_harbor_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapters/terminal-bench/run.sh"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(CAPTURE_AGENT_ADAPTER, encoding="utf-8")
            adapter.chmod(0o755)
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("terminal-bench",),
                limit=1,
                results_root=root / "results",
                run_id="run-agent-mismatch",
                app_root=root,
                harbor_agent="claude-code",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "KAIRYU_TEST_CAPTURE": str(root / "capture.json"),
                    "KAIRYU_TEST_RESULT_AGENT": "codex",
                },
            ):
                outcome = run_benchmarks(config, _DiscoveredClient(), load_manifest())

            failed = json.loads(
                (
                    root / "results/run-agent-mismatch/normalized/terminal-bench.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(outcome.exit_code, 3)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["agent"], "claude-code")
            self.assertIn("adapter result agent", failed["error"])

    def test_run_rejects_a_harbor_result_that_omits_agent_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapters/terminal-bench/run.sh"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(CAPTURE_AGENT_ADAPTER, encoding="utf-8")
            adapter.chmod(0o755)
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("terminal-bench",),
                limit=1,
                results_root=root / "results",
                run_id="run-agent-missing",
                app_root=root,
                harbor_agent="claude-code",
            )

            with mock.patch.dict(
                os.environ,
                {
                    "KAIRYU_TEST_CAPTURE": str(root / "capture.json"),
                    "KAIRYU_TEST_OMIT_AGENT": "1",
                },
            ):
                outcome = run_benchmarks(config, _DiscoveredClient(), load_manifest())

            failed = json.loads(
                (
                    root / "results/run-agent-missing/normalized/terminal-bench.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(outcome.exit_code, 3)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["agent"], "claude-code")
            self.assertIn("adapter result agent", failed["error"])


if __name__ == "__main__":
    unittest.main()
