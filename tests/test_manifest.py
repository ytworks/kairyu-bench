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


if __name__ == "__main__":
    unittest.main()
