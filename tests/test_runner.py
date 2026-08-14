from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.manifest import load_manifest
from kairyu_bench.runner import RunConfig, run_benchmarks
from kairyu_bench.target import Endpoint, PreflightError


FAKE_ADAPTER = r'''#!/usr/bin/env python3
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
    "scoring": {
        "method": context["scoring"]["method"],
        "self_judged": bool(context["scoring"].get("self_judged", False)),
        "self_simulated": bool(context["scoring"].get("self_simulated", False)),
    },
    "artifacts": {"raw": [], "logs": [f"logs/{context['benchmark']}.log"]},
    "timestamps": {"started_at": now, "finished_at": now},
    "error": None,
}
if "conditions" in context:
    payload["conditions"] = context["conditions"]
    expected = context["conditions"]["embedding_model_id"]
    if os.environ.get("KAIRYU_EMBEDDING_MODEL") != expected:
        raise RuntimeError("embedding model environment differs from context")
Path(os.environ["KAIRYU_BENCH_RESULT_PATH"]).write_text(json.dumps(payload))
'''


class _DiscoveredClient:
    def __init__(self, embedding_model: str = "embed-small") -> None:
        self.embedding_model = embedding_model
        self.chat_discoveries = 0
        self.embedding_discoveries = 0

    def discover_chat_model(self) -> str:
        self.chat_discoveries += 1
        return "chat-capable"

    def discover_embedding_model(self) -> str:
        self.embedding_discoveries += 1
        return self.embedding_model


class BenchmarkRunnerTest(unittest.TestCase):
    def test_run_discovers_model_invokes_selected_adapter_and_validates_result(self) -> None:
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

            client = _DiscoveredClient()
            outcome = run_benchmarks(config, client, load_manifest())

            self.assertEqual(outcome.exit_code, 0)
            self.assertEqual(outcome.model_id, "chat-capable")
            self.assertIsNone(outcome.embedding_model_id)
            self.assertEqual(client.chat_discoveries, 1)
            self.assertEqual(client.embedding_discoveries, 0)
            normalized = json.loads(
                (results / "run-success/normalized/gpqa-diamond.json").read_text()
            )
            self.assertEqual(normalized["score"]["primary"], 100.0)
            metadata_text = (results / "run-success/run.json").read_text()
            self.assertNotIn("secret-host.example", metadata_text)
            metadata = json.loads(metadata_text)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["benchmarks"], ["gpqa-diamond"])
            self.assertIsNone(metadata["embedding_model_id"])

    def test_tau_discovers_and_propagates_embedding_model_as_a_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapters/tau-bench-banking/run.sh"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(FAKE_ADAPTER, encoding="utf-8")
            adapter.chmod(0o755)
            results = root / "results"
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("tau-bench-banking",),
                limit=1,
                results_root=results,
                run_id="run-tau",
                app_root=root,
            )
            client = _DiscoveredClient("embed-small")

            outcome = run_benchmarks(config, client, load_manifest())

            metadata = json.loads((results / "run-tau/run.json").read_text())
            context = json.loads(
                (results / "run-tau/context/tau-bench-banking.json").read_text()
            )
            normalized = json.loads(
                (results / "run-tau/normalized/tau-bench-banking.json").read_text()
            )

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.embedding_model_id, "embed-small")
        self.assertEqual(client.embedding_discoveries, 1)
        self.assertEqual(metadata["embedding_model_id"], "embed-small")
        self.assertEqual(
            context["conditions"],
            {"embedding_model_id": "embed-small"},
        )
        self.assertEqual(normalized["conditions"], context["conditions"])

    def test_tau_embedding_preflight_fails_before_creating_run_directory(self) -> None:
        class MissingEmbeddingClient(_DiscoveredClient):
            def discover_embedding_model(self) -> str:
                self.embedding_discoveries += 1
                raise PreflightError("no advertised model accepted embedding requests")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = RunConfig(
                endpoint=Endpoint.parse("https://example.test/v1"),
                selected=("tau-bench-banking",),
                limit=1,
                results_root=root / "results",
                run_id="run-no-embedding",
                app_root=root,
            )
            client = MissingEmbeddingClient()

            with self.assertRaisesRegex(PreflightError, "embedding requests"):
                run_benchmarks(config, client, load_manifest())

            self.assertFalse((root / "results/run-no-embedding").exists())

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


if __name__ == "__main__":
    unittest.main()
