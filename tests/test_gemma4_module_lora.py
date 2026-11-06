"""Signed module recipes share old bytes; continuous gate solves stay unquantized."""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import numpy as np
import test_gemma4_lora_recipes as fixtures

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"
sys.path.insert(0, str(SCRIPTS))
import solve_module_directions as solver  # noqa: E402


class ModuleSolverTests(unittest.TestCase):
    def test_signed_constrained_solution_and_scope(self):
        g = np.array([[1., -1., 100.], [0., -1., 100.]])
        lower = np.array([.1, .05])
        coef, report = solver.solve_coefficients(g, lower, np.ones(3), [True, True, False])
        self.assertEqual(report["status"], "continuous_proposal")
        self.assertGreater(coef[0], 0)
        self.assertLess(coef[1], 0)
        self.assertEqual(coef[2], 0)
        self.assertTrue(np.all(g @ coef >= lower))
        self.assertTrue(np.array_equal(coef, coef.astype(np.float32).astype(np.float64)))
        self.assertFalse(report["actual_bf16_metrics"])
        # A BF16-rounded gate would not retain this exact coefficient.
        bits = coef.astype(np.float32).view(np.uint32)
        self.assertTrue(np.any((bits & 65535) != 0))

    def test_box_redistribution_and_norm_rejection_without_clipping(self):
        coef, report = solver.solve_coefficients([[1., 1.]], [.4], [.1, 1.], [True, True])
        self.assertEqual(report["status"], "continuous_proposal")
        self.assertTrue(report["box_constraints_in_solver"])
        self.assertLessEqual(coef.max(), .25)
        self.assertGreater(coef[1], .15)
        self.assertGreaterEqual(coef.sum(), .4)
        coef, report = solver.solve_coefficients(np.eye(2), [.2, .2], [4., 4.], [True, True])
        self.assertEqual(report["status"], "outside_trust_bounds")
        self.assertGreater(report["continuous_delta_l2"], 1.)
        self.assertTrue(np.all(coef >= [.2, .2]))


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("c++"), "Linux C++/OpenMP recipe")
class ModuleRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LoraRecipeTests.setUpClass()
        # A previous test class may have unloaded its temporary native library path.
        importlib.reload(importlib.import_module("bf16_lora_ops"))
        importlib.reload(importlib.import_module("dense_lora_recipe"))
        cls.module = importlib.import_module("module_lora_recipe")

    @classmethod
    def tearDownClass(cls):
        fixtures.LoraRecipeTests.tearDownClass()

    def setUp(self):
        f = fixtures.LoraRecipeTests("test_create_verify_and_source_ownership")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        _, mapping, _, _ = f.recipe.validated_recipe(f.base, f.spec)
        self.mapping = mapping
        costs = [{"base_tensor": e["tensor"], "adapter_A": e["lora_A"], "adapter_B": e["lora_B"],
                  "shape": e["shape"], "rank": 16, "lora_scale": 2., "unrounded_delta_l2": .03125} for e in mapping]
        self.cost = f.work / "costs.json"
        self.cost.write_text(json.dumps(costs))
        self.spec = dict(f.spec, schema_version=3, kind=self.module.KIND,
                         module_coefficients={e["tensor"]: .25 for e in mapping},
                         unit_costs_path=str(self.cost), unit_costs_sha256=f.recipe.digest(self.cost))
        self.spec.pop("coefficient")
        self.spec_path = f.work / "module-spec.json"
        self.save()

    def save(self):
        self.spec_path.write_text(json.dumps(self.spec))

    def test_uniform_quarter_is_old_scalar_bit_identical_and_verified(self):
        f, m = self.f, self.module
        f.spec["coefficient"] = .25
        f.spec_path.write_text(json.dumps(f.spec))
        old = f.recipe.create(f.base, f.dest, f.spec_path, f.manifest)
        target, manifest = f.work / "module-candidate", f.work / "module-manifest.json"
        new = m.create(f.base, target, self.spec_path, manifest)
        m.verify(manifest)
        self.assertEqual(f.recipe.tree(f.dest), f.recipe.tree(target))
        self.assertEqual(old["changed_elements"], new["changed_elements"])
        self.assertEqual(old["delta_l2"], new["delta_l2"])
        self.assertTrue(new["actual_bf16_metrics"])
        self.assertAlmostEqual(new["recipe"]["continuous_delta_l2"], math.sqrt(410) / 128)
        self.assertEqual(f.recipe.tree(f.base)[1], f.spec["base_tree_sha256"])
        entry = new["edits"][0]
        alias = f.work / "external-alias"
        os.link(target / entry["shard"], alias)
        with self.assertRaisesRegex(ValueError, "private"):
            m.verify(manifest)

    def test_signed_zero_and_sparse_scope_actual_bytes(self):
        m, f = self.module, self.f
        original = np.array([0x8000, 0, 0x3f80, 0xbf80], dtype=np.uint16)
        d = np.array([1., -1., .03125, -.03125], dtype=np.float32)
        self.assertTrue(np.array_equal(m.merge_block(original, d, 0)[0], original))
        for coefficient in [-.25, .25]:
            value = np.add((original.astype(np.uint32) << 16).view(np.float32),
                           np.multiply(d, np.float32(coefficient), dtype=np.float32), dtype=np.float32)
            bits = value.view(np.uint32)
            oracle = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            self.assertTrue(np.array_equal(m.merge_block(original, d, coefficient)[0], oracle))
        names = list(self.spec["module_coefficients"])
        self.spec["module_coefficients"] = dict.fromkeys(names, 0.)
        self.spec["module_coefficients"].update({names[0]: .25, names[1]: -.25})
        self.save()
        record = m.create(f.base, f.dest, self.spec_path, f.manifest)
        m.verify(f.manifest)
        self.assertEqual(len(record["edits"]), 2)
        self.assertEqual(record["changed_elements"], 6)
        self.assertEqual(f.recipe.tree(f.base)[1], f.spec["base_tree_sha256"])

    def test_invalid_maps_and_cost_bounds_fail_before_copy(self):
        m, f = self.module, self.f
        name = next(iter(self.spec["module_coefficients"]))
        for value in [True, .3, .1, float("nan")]:
            with self.subTest(value=value):
                bad = dict(self.spec, module_coefficients=dict(self.spec["module_coefficients"], **{name: value}))
                with self.assertRaises(ValueError):
                    m.validated_recipe(f.base, bad)
        bad = dict(self.spec, module_coefficients={name: .25})
        with self.assertRaisesRegex(ValueError, "410"):
            m.validated_recipe(f.base, bad)
        rows = json.loads(self.cost.read_text())
        for row in rows:
            row["unrounded_delta_l2"] = 1.
        self.cost.write_text(json.dumps(rows))
        self.spec["unit_costs_sha256"] = f.recipe.digest(self.cost)
        with self.assertRaisesRegex(ValueError, "L2"):
            m.validated_recipe(f.base, self.spec)
        self.assertFalse(f.dest.exists())

    def test_exclusive_publication_and_source_change_cleanup(self):
        m, f = self.module, self.f
        original_entries = m.entries
        def foreign_manifest(*args, **kwargs):
            result = original_entries(*args, **kwargs)
            f.manifest.write_text("foreign")
            return result
        with mock.patch.object(m, "entries", foreign_manifest):
            with self.assertRaises(FileExistsError):
                m.create(f.base, f.dest, self.spec_path, f.manifest)
        self.assertEqual(f.manifest.read_text(), "foreign")
        self.assertFalse(f.dest.exists())
        f.manifest.unlink()
        def changed_spec(*args, **kwargs):
            result = original_entries(*args, **kwargs)
            self.spec_path.write_text(self.spec_path.read_text() + " ")
            return result
        with mock.patch.object(m, "entries", changed_spec):
            with self.assertRaisesRegex(ValueError, "changed during"):
                m.create(f.base, f.dest, self.spec_path, f.manifest)
        self.assertFalse(f.dest.exists())
        self.assertFalse(f.manifest.exists())
        self.assertEqual(f.recipe.tree(f.base)[1], f.spec["base_tree_sha256"])


if __name__ == "__main__":
    unittest.main()
