"""Small persistent joint fixtures: exact bytes, exclusive publication and cleanup."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import shutil
import sys
import unittest
from unittest import mock

import test_gemma4_module_lora as fixtures
from kairyu_bench.wsp.weights import editor


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("c++"), "Linux C++ recipe")
class JointRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These modules bind the temporary native library at import time. Do not
        # reload or leave them in another recipe test class's module cache.
        names = ("bf16_lora_ops", "dense_lora_recipe", "module_lora_recipe", "joint_module_recipe")
        prior = {name: sys.modules[name] for name in names if name in sys.modules}
        def restore_modules():
            for name in names:
                sys.modules.pop(name, None)
            sys.modules.update(prior)
        cls.addClassCleanup(restore_modules)
        for name in names:
            sys.modules.pop(name, None)
        fixtures.ModuleRecipeTests.setUpClass()
        cls.joint = importlib.import_module("joint_module_recipe")

    @classmethod
    def tearDownClass(cls):
        fixtures.ModuleRecipeTests.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.ModuleRecipeTests("test_signed_zero_and_sparse_scope_actual_bytes")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture.f
        self.f = f
        self.report = f.work / "joint-report.json"
        self.report.write_text(json.dumps({"source_checkpoint": str(f.base), "disjoint_weight_tensors": True}))
        common = {"companion_required": True, "joint_report_path": str(self.report),
                  "joint_report_sha256": f.recipe.digest(self.report)}
        self.fixture.spec.update(common)
        self.fixture.save()
        self.native = {"id": "native", **common, "edits": [
            {"tensor": "model.language_model.norm.weight", "indices": [0, 1, 6],
             "set_values": [1.5, -1.5, -0.0]}]}
        self.native_path = f.work / "native.json"
        self.native_path.write_text(json.dumps(self.native))

    def create(self):
        return self.joint.create(self.f.base, self.f.dest, self.fixture.spec_path,
                                 self.native_path, self.f.manifest, min_free_gib=0)

    def test_exact_joint_bytes_metrics_and_default_preview(self):
        f, m, j = self.f, self.fixture.module, self.joint
        source_tree = f.recipe.tree(f.base)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            j.main(["create", str(f.base), str(f.dest), "--spec", str(self.fixture.spec_path),
                    "--native-spec", str(self.native_path), "--manifest", str(f.manifest)])
        self.assertTrue(json.loads(stream.getvalue())["dry_run"])
        self.assertFalse(f.dest.exists())
        record = self.create()
        self.assertEqual(record["schema_version"], 4)
        self.assertEqual(record["kind"], j.KIND)
        self.assertTrue(record["tensor_disjoint"])
        self.assertEqual(f.recipe.tree(f.base), source_tree)
        self.assertEqual(f.recipe.tree(f.dest), (record["candidate_files"], record["candidate_tree_sha256"]))
        native = [r for r in record["edits"] if r["entry_kind"] == "native_sparse_bytes"]
        module = [r for r in record["edits"] if r["entry_kind"] == "lora_matrix"]
        self.assertEqual(len(native), 1)
        self.assertEqual(len(module), 410)
        n = native[0]
        self.assertEqual(n["changed_count"], 3)
        self.assertEqual(editor.read_at(f.dest/n["shard"], n["offset"], n["size"]), bytes.fromhex(n["after_hex"]))
        stored = editor.decode(bytes.fromhex(n["after_hex"]), "BF16")
        self.assertEqual([stored[i] for i in [0, 1, 6]], [1.5, -1.5, -0.0])
        self.assertLess(math.copysign(1, stored[6]), 0)  # preserve exact requested negative zero
        prepared = m.prepared(f.base, self.fixture.spec_path)
        expected = m.entries(f.base, f.dest, prepared[3], prepared[4], prepared[5], prepared[6])
        f.recipe.metrics(module, expected)
        self.assertEqual(record["changed_elements"], sum(r["changed_count"] for r in record["edits"]))
        self.assertEqual(record["delta_l2"], math.hypot(*(r["delta_l2"] for r in record["edits"])))
        for row in record["files"]:
            self.assertFalse((f.dest/row["path"]).samefile(f.base/row["path"]))
            self.assertEqual((f.dest/row["path"]).stat().st_nlink, 1)
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.create()
        self.assertEqual(f.recipe.tree(f.dest)[1], record["candidate_tree_sha256"])

    def test_invalid_exact_values_companion_and_overlap_fail_before_copy(self):
        f = self.f
        self.native["edits"][0]["set_values"][0] = 1.1
        self.native_path.write_text(json.dumps(self.native))
        with self.assertRaisesRegex(ValueError, "exact BF16"):
            self.create()
        self.native["edits"][0]["set_values"][0] = 1.5
        self.native["joint_report_sha256"] = "0" * 64
        self.native_path.write_text(json.dumps(self.native))
        with self.assertRaisesRegex(ValueError, "binding"):
            self.create()
        self.native["joint_report_sha256"] = f.recipe.digest(self.report)
        self.native["edits"][0]["tensor"] = self.fixture.mapping[0]["tensor"]
        self.native_path.write_text(json.dumps(self.native))
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(f.dest.exists())
        self.assertFalse(f.manifest.exists())

    def test_partial_native_write_input_change_and_publication_boundaries(self):
        f, j = self.f, self.joint
        source = f.recipe.tree(f.base)
        original_write = j.write_native
        def partial(*args):
            original_write(*args)
            raise OSError("after native write")
        with mock.patch.object(j, "write_native", partial), self.assertRaises(OSError):
            self.create()
        self.assertFalse(f.dest.exists())
        self.assertFalse(f.manifest.exists())
        self.assertEqual(f.recipe.tree(f.base), source)
        def changed(*args):
            original_write(*args)
            self.native_path.write_text(self.native_path.read_text() + " ")
        with mock.patch.object(j, "write_native", changed), self.assertRaisesRegex(ValueError, "changed during"):
            self.create()
        self.assertFalse(f.dest.exists())
        self.native_path.write_text(json.dumps(self.native))
        def collision(*args):
            original_write(*args)
            f.manifest.write_text("foreign")
        with mock.patch.object(j, "write_native", collision), self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(f.manifest.read_text(), "foreign")
        self.assertFalse(f.dest.exists())
        f.manifest.unlink()
        original_save = j.dense.save
        def committed_error(*args):
            original_save(*args)
            raise OSError("after own manifest commit")
        with mock.patch.object(j.dense, "save", committed_error), self.assertRaises(OSError):
            self.create()
        self.assertTrue(f.dest.exists())
        self.assertEqual(f.recipe.tree(f.base), source)
        manifest = json.loads(f.manifest.read_text())
        self.assertEqual(f.recipe.tree(f.dest)[1], manifest["candidate_tree_sha256"])


if __name__ == "__main__":
    unittest.main()
