from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_gemma4_purpose_recipes import COMMON, load, torch

COSTS = load("compare_costs")


class PurposeCostTests(unittest.TestCase):
    def test_role_relative_floor_protects_zero_and_is_explicit(self):
        names = ["model.language_model.layers.0.norm.weight", "model.language_model.layers.1.norm.weight"]
        base = np.array([0., 0., 1., 2.])
        cost, floors = COSTS.diagonal_cost(base, names, [0, 2],
            {"type": "role_relative", "floor_ratio": .1, "absolute_floor": 1e-6})
        np.testing.assert_allclose(cost, [20., 20., 1., .5])
        self.assertEqual(len(floors), 1)
        self.assertEqual(next(iter(floors.values()))["floor"], .05)
        larger, _ = COSTS.diagonal_cost(base, names, [0, 2],
            {"type": "role_relative", "floor_ratio": .2, "absolute_floor": 1e-6})
        np.testing.assert_allclose(larger, [10., 10., 1., .5])
        with self.assertRaises(ValueError):
            COSTS.diagonal_cost(base, names, [0, 2],
                {"type": "role_relative", "floor_ratio": 0., "absolute_floor": 1e-6})

    def test_preview_rejects_unmatched_constraints_before_loading_derivatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"profiles": [
                {"id": "a", "group_mean_lower": {"code": .02}, "fold_lower": {"code": -.08}},
                {"id": "b", "group_mean_lower": {"code": .03}, "fold_lower": {"code": -.08}}]}))
            with self.assertRaisesRegex(ValueError, "identical"):
                COSTS.main(["--legacy-chat", str(root / "chat"), "--basis", str(root / "basis.npz"),
                            "--basis-provenance", str(root / "provenance.json"),
                            "--plan", str(plan), "--output", str(root / "output")])
            self.assertFalse((root / "output").exists())

    def test_output_name_collisions_rejected_before_any_run_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            output = root / "output"
            for identifiers in (["summary"], ["input-evidence"], ["progress"],
                                ["failure"], ["a", "a-report"]):
                with self.subTest(identifiers=identifiers):
                    plan.write_text(json.dumps({
                        "diagnostic_relative_floor": 1e-6,
                        "profiles": [{"id": identifier, "group_mean_lower": {"code": .02},
                                      "fold_lower": {"code": -.08}} for identifier in identifiers]}))
                    before = plan.read_bytes()
                    with self.assertRaisesRegex(ValueError, "output filenames"):
                        COSTS.main(["--legacy-chat", str(root / "chat"),
                                    "--basis", str(root / "basis.npz"),
                                    "--basis-provenance", str(root / "provenance.json"),
                                    "--plan", str(plan), "--output", str(output), "--run"])
                    self.assertFalse(output.exists())
                    self.assertEqual(plan.read_bytes(), before)

    @unittest.skipIf(torch is None, "optional saved-tensor CPU fixture")
    def test_legacy_chat_hash_frame_and_tensor_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, basisdir = root / "chat", root / "basis"
            source.mkdir()
            basisdir.mkdir()
            name = "model.language_model.norm.weight"
            COMMON.save(source / "cases.json", {"code": [[{"id": "x"}]]})
            cases_sha = COMMON.sha(source / "cases.json")
            method = "official-thinking-on-public-answer-v1"
            COMMON.save(source / "scan-coverage.json", {
                "method_id": method, "raw_gradients_reused": False, "kinds": {name: "native"}})
            COMMON.save(source / "observations.json", {
                "method_id": method, "cases_sha256": cases_sha, "base_tree_sha256": "a" * 64})
            for path in (source / "frame-evidence.json", basisdir / "frame-evidence.json"):
                COMMON.save(path, {"frame": "fixture"})
            np.savez(basisdir / "basis.npz", base=np.array([1., 2.], dtype=np.float32),
                     tensor_names=np.array([name]), tensor_offsets=np.array([0]))
            COMMON.save(basisdir / "provenance.json", {
                "source_sha256": {"cases.json": cases_sha}, "expected_base_tree_sha256": "a" * 64})
            torch.save({name: torch.tensor([.5, 1.])}, source / "code-0.pt")
            args = argparse.Namespace(legacy_chat=source, basis=basisdir / "basis.npz",
                                      basis_provenance=basisdir / "provenance.json")
            base, gradient, names, offsets, groups, hashes, evidence = COSTS.load_inputs(args)
            np.testing.assert_array_equal(base, [1., 2.])
            np.testing.assert_array_equal(gradient, [[.5, 1.]])
            self.assertEqual(groups, [("code", 0)])
            self.assertFalse(evidence["raw_G_mixed"])
            self.assertIn("not reread", evidence["basis_trust_boundary"])
            self.assertIn(str(source / "code-0.pt"), hashes)
            COMMON.save(basisdir / "frame-evidence.json", {"frame": "different"})
            with self.assertRaisesRegex(ValueError, "frames"):
                COSTS.load_inputs(args)


if __name__ == "__main__":
    unittest.main()
