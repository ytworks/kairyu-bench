"""Tiny CPU contract checks for the dense verify-to-freeze bridge."""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

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
        self.assertTrue(result["verification"]["all_410_matrices_cpu_regenerated"])
        self.assertEqual(result["verified_scope"]["regenerated_projection_count"], 410)
        path = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/redeploy-replica.py"
        spec = importlib.util.spec_from_file_location("dense_freeze_consumer", path)
        consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consumer)
        proof = consumer.identity(SimpleNamespace(candidate=f.dest, freeze=self.output))
        self.assertEqual(proof["tree_sha256"], result["control_tree_sha256"])

    def test_mlp_scope_f32_roundtrip_freeze_and_untouched_signed_zero(self):
        f, recipe, bridge = self.f, self.recipe, self.bridge
        factors = f._f32_factors()
        _, mapping, _, _ = recipe.materialization_recipe(f.base, f.spec)
        by_name = {row["tensor"]: row for row in mapping}
        attention = next(name for name in by_name if ".self_attn." in name)
        native = "model.language_model.norm.weight"
        index = json.loads((f.base / "model.safetensors.index.json").read_text())["weight_map"]

        def location(name):
            shard = index[name]
            header, offset = recipe.header(f.base / shard)
            lo, hi = header[name]["data_offsets"]
            return shard, offset + lo, hi - lo

        def raw(root, name):
            shard, offset, size = location(name)
            with (root / shard).open("rb") as stream:
                stream.seek(offset)
                return stream.read(size)

        # A zero scope must preserve the original negative-zero word without +0.
        for name in (attention, native):
            shard, offset, _ = location(name)
            with (f.base / shard).open("r+b") as stream:
                stream.seek(offset)
                stream.write(bytes((0, 128)))
        f.spec["base_tree_sha256"] = recipe.tree(f.base)[1]
        f.spec["structured_scope"] = "mlp_triplets_all_layers"
        f.spec["module_coefficients"] = {
            name: f.spec["coefficient"] if ".mlp." in name else 0.0 for name in by_name
        }
        f.spec_path.write_text(json.dumps(f.spec))
        source_before = recipe.tree(f.base)
        untouched_before = {name: raw(f.base, name)
                            for name in index if name not in by_name or ".self_attn." in name}
        candidate, manifest = f.work / "mlp-candidate", f.work / "mlp-manifest.json"
        with contextlib.redirect_stdout(io.StringIO()):
            record = recipe.create(f.base, candidate, f.spec_path, manifest)
        self.assertEqual(len(record["edits"]), 180)
        for name, original in untouched_before.items():
            self.assertEqual(raw(candidate, name), original)
        for row in record["edits"]:
            a, b = (factors[row[key]] for key in ("lora_A", "lora_B"))
            # Independent one-term product oracle, then the specified F32 operations.
            product = np.array([[float(b[i, 0]) * float(a[0, j])
                                 for j in range(a.shape[1])] for i in range(b.shape[0])], np.float32)
            delta = np.multiply(product, np.float32(2), dtype=np.float32)
            original = (np.frombuffer(raw(f.base, row["tensor"]), dtype="<u2")
                        .astype(np.uint32) << 16).view(np.float32).reshape(delta.shape)
            bits = np.add(original, delta, dtype=np.float32).view(np.uint32)
            expected = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
            self.assertEqual(raw(candidate, row["tensor"]), expected.tobytes())
        self.args.source, self.args.candidate, self.args.manifest = f.base, candidate, manifest
        self.args.manifest_sha256 = recipe.digest(manifest)
        self.args.spec_sha256 = recipe.digest(f.spec_path)
        self.args.output, self.args.apply = f.work / "mlp-freeze.json", True
        with mock.patch.object(recipe, "verify", wraps=recipe.verify) as verify:
            result = bridge.verify(self.args)
            verify.assert_called_once_with(manifest)
        self.assertFalse(result["verification"]["all_410_matrices_cpu_regenerated"])
        self.assertEqual(result["verified_scope"], {
            "structured_scope": "mlp_triplets_all_layers", "validated_projection_count": 410,
            "regenerated_projection_count": 180, "zero_coefficient_projection_count": 230,
            "zero_scope_and_other_spans_byte_verified": True})
        self.assertEqual(recipe.tree(f.base), source_before)
        # A bit-only modification outside the scope must fail the same verifier.
        shard, offset, _ = location(attention)
        with (candidate / shard).open("r+b") as stream:
            stream.seek(offset)
            stream.write(bytes((0, 0)))
        self.args.output = f.work / "corrupted-mlp-freeze.json"
        with self.assertRaises(ValueError):
            bridge.verify(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(recipe.tree(f.base), source_before)

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
