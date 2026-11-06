"""The CPU profile compares real wrapper/kernel paths without source writes."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
import sys
import unittest

import numpy as np
import test_gemma4_module_lora as fixtures


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("c++"), "Linux C++/OpenMP recipe")
class MergeProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ModuleRecipeTests.setUpClass()
        cls.profile = importlib.import_module("profile_merge")

    @classmethod
    def tearDownClass(cls):
        fixtures.ModuleRecipeTests.tearDownClass()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU Torch byte-view contract")
    def test_staged_byte_equality_preserves_nan_payloads_and_signed_zero(self):
        import torch
        original = np.tile(np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x7FC1, 0xFFC2], np.uint16), (3, 1))
        bits = original[:, ::-1].copy()
        before = original.copy(), bits.copy()
        report = self.profile.device_transfer_profile(original, bits, "cpu", torch)
        self.assertTrue(report["staged_device_int16_equal"])
        self.assertEqual(report["target_cpu_sha256"], report["final_independent_cpu_oracle_sha256"])
        self.assertTrue(report["scratch_only"])
        np.testing.assert_array_equal(original, before[0])
        np.testing.assert_array_equal(bits, before[1])

    def test_signed_multiblock_paths_and_fresh_readonly_profile(self):
        profile = self.profile
        source = np.tile(np.array([1., -0., -1., .125], np.float32), (257, 1))
        bits = (source.view(np.uint32) >> 16).astype(np.uint16)
        delta = np.tile(np.array([.125, -.125, 1., -2.], np.float32), (257, 1))
        for coefficient in (-.25, .25):
            expected = None
            for path in ("module", "wrapper", "kernel"):
                target, report = profile.tiled(bits, delta, coefficient, path)
                if expected is None:
                    expected = target.copy(), report
                np.testing.assert_array_equal(target, expected[0])
                self.assertEqual(report["target_sha256"], expected[1]["target_sha256"])
                self.assertEqual(report["changed_elements"], expected[1]["changed_elements"])
                self.assertAlmostEqual(report["delta_l2"], expected[1]["delta_l2"], places=12)
        np.testing.assert_array_equal(bits, (source.view(np.uint32) >> 16).astype(np.uint16))
        fixture = fixtures.ModuleRecipeTests("test_uniform_quarter_is_old_scalar_bit_identical_and_verified")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        output = fixture.f.work / "profile.json"
        argv = ["--checkpoint", str(fixture.f.base), "--spec", str(fixture.spec_path),
                "--output", str(output), "--maximum-seconds", "10"]
        before = fixture.f.recipe.tree(fixture.f.base)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(profile.main(argv), 0)
            self.assertFalse(output.exists())
            self.assertEqual(profile.main([*argv, "--run"]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(result["source_stable"])
        self.assertTrue(all(row["all_paths_byte_count_stat_agreement"] for row in result["results"]))
        self.assertEqual(before, fixture.f.recipe.tree(fixture.f.base))
        with self.assertRaisesRegex(ValueError, "fresh"):
            profile.main([*argv, "--run"])
        self.assertEqual(result, json.loads(output.read_text()))
        bad = [*argv]
        bad[bad.index("--output") + 1] = str(fixture.f.base / "analysis.json")
        with self.assertRaisesRegex(ValueError, "outside"):
            profile.main(bad)
        self.assertFalse((fixture.f.base / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()
