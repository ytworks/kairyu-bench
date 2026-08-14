from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SHIM_PATH = ROOT / "scripts/shims/tau_bench_banking.py"
SPEC = importlib.util.spec_from_file_location("tau_bench_banking_shim", SHIM_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load tau-bench-banking shim")
SHIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHIM)


class TauBenchBankingShimTest(unittest.TestCase):
    def test_harness_passes_the_discovered_model_through_the_shim(self) -> None:
        harness = (ROOT / "scripts/harnesses/tau-bench-banking.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("conditions.embedding_model_id", harness)
        self.assertIn("KAIRYU_EMBEDDING_MODEL", harness)
        self.assertIn("scripts/shims/tau_bench_banking.py", harness)
        self.assertNotIn('"$environment/bin/tau2"', harness)

    def test_main_reads_environment_and_preserves_official_cli_arguments(self) -> None:
        seen_arguments: list[str] = []
        tau2_package = types.ModuleType("tau2")
        tau2_package.__path__ = []
        tau2_cli = types.ModuleType("tau2.cli")

        def tau2_main() -> None:
            seen_arguments.extend(sys.argv[1:])

        tau2_cli.main = tau2_main
        arguments = [
            "tau_bench_banking.py",
            "run",
            "--domain",
            "banking_knowledge",
            "--retrieval-config",
            "alltools",
            "--num-trials",
            "4",
        ]

        with mock.patch.dict(os.environ, {"KAIRYU_EMBEDDING_MODEL": "embed-small"}), \
            mock.patch.dict(
                sys.modules,
                {"tau2": tau2_package, "tau2.cli": tau2_cli},
            ), \
            mock.patch.object(sys, "argv", arguments), \
            mock.patch.object(SHIM, "configure_embedding_model") as configure:
            result = SHIM.main()

        self.assertEqual(result, 0)
        configure.assert_called_once_with("embed-small")
        self.assertEqual(seen_arguments, arguments[1:])

    def test_configures_per_query_retrieval_and_cache_warmup(self) -> None:
        dense = types.SimpleNamespace(
            embedder_type="openai",
            embedder_model="text-embedding-3-large",
        )
        retrieval = types.SimpleNamespace(
            RETRIEVAL_VARIANTS={
                "alltools": types.SimpleNamespace(kb_search_dense=dense),
            }
        )

        def original_selector(
            names: list[str],
            kwargs: dict[str, object] | None = None,
        ) -> list[tuple[str, dict[str, object]]]:
            if any(name in {"alltools", "AllTools"} for name in names):
                return [("openai", {"model": "text-embedding-3-large"})]
            return [("openrouter", {"model": "qwen3-embedding-8b"})]

        cache = types.SimpleNamespace(
            get_unique_embedder_configs_for_retrieval_configs=original_selector,
        )

        SHIM.configure_embedding_model(
            "embed-small",
            retrieval_module=retrieval,
            embeddings_cache_module=cache,
        )

        self.assertEqual(dense.embedder_model, "embed-small")
        self.assertEqual(
            cache.get_unique_embedder_configs_for_retrieval_configs(["alltools"]),
            [("openai", {"model": "embed-small"})],
        )
        self.assertEqual(
            cache.get_unique_embedder_configs_for_retrieval_configs(
                ["alltools"],
                {"top_k": 10},
            ),
            [("openai", {"model": "embed-small"})],
        )

    def test_preserves_unrelated_retrieval_configurations(self) -> None:
        dense = types.SimpleNamespace(
            embedder_type="openai",
            embedder_model="text-embedding-3-large",
        )
        retrieval = types.SimpleNamespace(
            RETRIEVAL_VARIANTS={
                "alltools": types.SimpleNamespace(kb_search_dense=dense),
            }
        )

        def original_selector(
            names: list[str],
            kwargs: dict[str, object] | None = None,
        ) -> list[tuple[str, dict[str, object]]]:
            return [("openrouter", {"model": "qwen3-embedding-8b"})]

        cache = types.SimpleNamespace(
            get_unique_embedder_configs_for_retrieval_configs=original_selector,
        )
        SHIM.configure_embedding_model(
            "embed-small",
            retrieval_module=retrieval,
            embeddings_cache_module=cache,
        )

        self.assertEqual(
            cache.get_unique_embedder_configs_for_retrieval_configs(["alltools-qwen"]),
            [("openrouter", {"model": "qwen3-embedding-8b"})],
        )

    def test_fails_closed_if_the_pinned_tau_layout_changes(self) -> None:
        dense = types.SimpleNamespace(
            embedder_type="openai",
            embedder_model="different-default",
        )
        retrieval = types.SimpleNamespace(
            RETRIEVAL_VARIANTS={
                "alltools": types.SimpleNamespace(kb_search_dense=dense),
            }
        )
        cache = types.SimpleNamespace(
            get_unique_embedder_configs_for_retrieval_configs=lambda names, kwargs=None: []
        )

        with self.assertRaisesRegex(RuntimeError, "no longer matches the shim"):
            SHIM.configure_embedding_model(
                "embed-small",
                retrieval_module=retrieval,
                embeddings_cache_module=cache,
            )


if __name__ == "__main__":
    unittest.main()
