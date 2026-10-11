from __future__ import annotations

import unittest

from kairyu_bench.benchmarks import BENCHMARK_NAMES
from kairyu_bench.manifest import ManifestError, load_manifest, select_benchmarks


class OfficialSourceManifestTest(unittest.TestCase):
    def test_manifest_locks_one_official_source_for_every_public_adapter(self) -> None:
        manifest = load_manifest()

        self.assertEqual(tuple(manifest), BENCHMARK_NAMES)
        for name, entry in manifest.items():
            with self.subTest(name=name):
                self.assertEqual(entry["name"], name)
                self.assertTrue(entry["source"]["repository"].startswith("https://"))
                self.assertRegex(entry["source"]["revision"], r"^[0-9a-f]{40}$")
                self.assertTrue(entry["dataset"]["id"])
                self.assertTrue(entry["dataset"]["revision"])
                self.assertEqual(entry["adapter"], f"adapters/{name}/run.sh")

    def test_only_preserves_public_order_not_argument_order(self) -> None:
        selected = select_benchmarks("mrcr-v2,gpqa-diamond")

        self.assertEqual(selected, ["gpqa-diamond", "mrcr-v2"])

    def test_only_rejects_unknown_or_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ManifestError, "unknown benchmark"):
            select_benchmarks("gpqa-diamond,not-real")
        with self.assertRaisesRegex(ManifestError, "duplicate benchmark"):
            select_benchmarks("hle,hle")

    def test_gated_dataset_requirements_match_their_adapters(self) -> None:
        manifest = load_manifest()

        self.assertEqual(manifest["livecodebench"]["requirements"], [])
        self.assertEqual(
            manifest["gpqa-diamond"]["requirements"],
            ["hf_token", "gated-dataset-acceptance"],
        )
        self.assertIn("embeddings", manifest["tau-bench-banking"]["requirements"])

    def test_terminal_bench_4_pins_the_leaderboard_dataset_digest(self) -> None:
        entry = load_manifest()["terminal-bench-4"]

        self.assertEqual(
            entry["source"],
            {
                "repository": "https://github.com/harbor-framework/harbor",
                "revision": "b53b8134e1241686dca7759af188f987ecc48e8b",
                "version": "0.24.0",
            },
        )
        self.assertEqual(
            entry["dataset"],
            {
                "id": "terminal-bench/terminal-bench",
                "revision": "sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732",
                "version": "4.0.0",
                "tasks": 66,
            },
        )
        self.assertEqual(
            [source["revision"] for source in entry["secondary_sources"]],
            [
                "452bf305c6daa62fc59061d22133a7cbc7c1572e",
                "209679e34327a78ce8e9bf5300b2f317992863c9",
            ],
        )
        self.assertEqual(
            entry["scoring"],
            {"method": "terminal-bench-4.0-leaderboard-accuracy", "unit": "percent"},
        )


if __name__ == "__main__":
    unittest.main()
