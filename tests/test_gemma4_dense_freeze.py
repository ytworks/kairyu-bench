"""Tiny CPU contract checks for the dense verify-to-freeze bridge."""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import test_gemma4_lora_recipes as fixtures


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("c++"), "Linux C++ recipe")
class DenseFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate modules that capture a temporary native library at import.
        names = ("bf16_lora_ops", "dense_lora_recipe", "verify_dense_checkpoint")
        prior = {name: sys.modules[name] for name in names if name in sys.modules}
        def restore_modules():
            for name in names:
                sys.modules.pop(name, None)
            sys.modules.update(prior)
        cls.addClassCleanup(restore_modules)
        for name in names:
            sys.modules.pop(name, None)
        fixtures.LoraRecipeTests.setUpClass()
        cls.recipe = importlib.import_module("dense_lora_recipe")
        cls.bridge = importlib.import_module("verify_dense_checkpoint")

    @classmethod
    def tearDownClass(cls):
        fixtures.LoraRecipeTests.tearDownClass()

    def setUp(self):
        self.f = fixtures.LoraRecipeTests("test_create_verify_and_source_ownership")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        f.recipe = self.recipe
        with contextlib.redirect_stdout(io.StringIO()):
            self.record = f.recipe.create(f.base, f.dest, f.spec_path, f.manifest)
        self.output = f.work / "freeze.json"
        self.args = SimpleNamespace(
            source=f.base, candidate=f.dest, manifest=f.manifest, spec=f.spec_path,
            manifest_sha256=f.recipe.digest(f.manifest), spec_sha256=f.recipe.digest(f.spec_path),
            kernel=f.recipe.LIBRARY, kernel_sha256=f.recipe.digest(f.recipe.LIBRARY),
            output=self.output, apply=False)

    def test_preview_then_one_real_verify_and_redeploy_identity(self):
        b, f = self.bridge, self.f
        with mock.patch.object(self.recipe, "verify", wraps=self.recipe.verify) as verify:
            self.assertEqual(b.verify(self.args)["status"], "preview")
            verify.assert_not_called()
            self.assertFalse(self.output.exists())
            source_before = f.recipe.tree(f.base)
            self.args.apply = True
            result = b.verify(self.args)
            verify.assert_called_once_with(f.manifest)
        self.assertEqual(result["all_checkpoint_files"], self.record["candidate_files"])
        self.assertEqual(result["control_tree_sha256"], self.record["candidate_tree_sha256"])
        self.assertEqual(f.recipe.tree(f.base), source_before)
        self.assertFalse(result["ram_comparison_performed"])
        path = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/redeploy-replica.py"
        spec = importlib.util.spec_from_file_location("dense_freeze_consumer", path)
        consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consumer)
        proof = consumer.identity(SimpleNamespace(candidate=f.dest, freeze=self.output))
        self.assertEqual(proof["tree_sha256"], result["control_tree_sha256"])

    def test_real_payload_or_untouched_corruption_never_publishes(self):
        self.args.apply = True
        f = self.f
        for target, offset in ((f.dest/self.record["edits"][0]["shard"], self.record["edits"][0]["offset"]),
                               (f.dest/"tokenizer_config.json", 0)):
            with self.subTest(file=target.name):
                original = target.read_bytes()
                with target.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(b"!!")
                with self.assertRaises(ValueError):
                    self.bridge.verify(self.args)
                self.assertFalse(self.output.exists())
                target.write_bytes(original)

    def test_after_verify_mutation_output_overlap_and_pins_reject(self):
        self.args.apply = True
        f = self.f
        for mutation in ("candidate", "spec"):
            with self.subTest(mutation=mutation):
                path = f.dest/"tokenizer_config.json" if mutation == "candidate" else f.spec_path
                original = path.read_bytes()
                real = self.recipe.verify
                def changed(manifest):
                    result = real(manifest)
                    path.write_bytes(original + b" ")
                    return result
                with mock.patch.object(self.recipe, "verify", changed), self.assertRaisesRegex(ValueError, "changed"):
                    self.bridge.verify(self.args)
                self.assertFalse(self.output.exists())
                path.write_bytes(original)
        self.args.output = f.dest/"freeze.json"
        with mock.patch.object(self.recipe, "verify") as verify, self.assertRaisesRegex(ValueError, "outside"):
            self.bridge.verify(self.args)
        verify.assert_not_called()
        self.args.output = self.output
        self.args.kernel_sha256 = "0"*64
        with self.assertRaisesRegex(ValueError, "kernel pins"):
            self.bridge.verify(self.args)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
