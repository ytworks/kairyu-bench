from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora/profile_donor_directions.py"
spec = importlib.util.spec_from_file_location("profile_donor_directions", SCRIPT)
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class DonorProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.collection = self.root / "collection"
        self.collection.mkdir()
        entries = [{"tensor": f"tensor{i}", "coordinate": i, "layer": i, "family": family,
                    "unit_gate_weight_l2": cost} for i, (family, cost) in enumerate([("q", 2.), ("down", 1.), ("o", 0.)])]
        self.coverage = {"entries": entries, "direction_coordinates": 3}
        folds = []
        for i, g in enumerate(([2, -1, 0], [4, -2, 0], [-2, -1, 0], [-4, -2, 0])):
            file = f"g{i}.npy"
            np.save(self.collection / file, np.array(g, np.float64), allow_pickle=False)
            folds.append({"purpose": "first" if i < 2 else "second", "fold": i % 2, "file": file})
        self.summary = {"status": "completed", "source_values_unchanged": True, "mode": "chat-public-answer",
                        "coverage": self.coverage, "folds": folds,
                        "zero_alpha_checks": {"every_microbatch_compared": True, "max_absolute_margin_difference": 0, "tolerance": 1e-5}}
        (self.collection / "summary.json").write_text(json.dumps(self.summary))
        (self.collection / "coverage.json").write_text(json.dumps(self.coverage))
        (self.collection / "provenance.json").write_text(json.dumps({"mode": "chat-public-answer"}))
        np.save(self.collection / "unit-gate-weight-l2.npy", np.array([2., 1., 0.]), allow_pickle=False)

    def test_profile_preserves_fold_units_costs_and_zero_semantics(self):
        output = self.root / "profile.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(profile.main(["--collection", str(self.collection), "--output", str(output)]), 0)
        result = json.loads(output.read_text())
        self.assertEqual([f["uniform_all_gate_derivative"] for f in result["folds"]], [1, 2, -3, -6])
        self.assertEqual(result["all_fold_signs"], {"all_positive": 0, "all_negative": 1, "mixed_sign": 1, "all_zero": 1, "zero_or_one_sided": 0})
        self.assertEqual(result["purposes"]["first"]["all_positive"], 1)
        self.assertAlmostEqual(result["weight_cost_normalized_gradient_cosine"][0][1], 1)
        self.assertAlmostEqual(result["weight_cost_normalized_gradient_cosine"][0][2], 0)
        self.assertIsNone(result["folds"][0]["groups"]["family:o"]["uniform_derivative_per_unit_weight_l2"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            profile.main(["--collection", str(self.collection), "--output", str(output)])

    def test_inconsistent_mode_cost_and_zero_cost_derivative_are_rejected(self):
        path = self.collection / "provenance.json"
        path.write_text(json.dumps({"mode": "raw-next-token"}))
        with self.assertRaisesRegex(ValueError, "mode-matched"):
            profile.analyze(self.collection)
        path.write_text(json.dumps({"mode": "chat-public-answer"}))
        np.save(self.collection / "unit-gate-weight-l2.npy", np.array([3., 1., 0.]), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "costs differ"):
            profile.analyze(self.collection)
        np.save(self.collection / "unit-gate-weight-l2.npy", np.array([2., 1., 0.]), allow_pickle=False)
        np.save(self.collection / "g0.npy", np.array([2., -1., 1.]), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "zero-cost"):
            profile.analyze(self.collection)

    def test_correct_objective_requires_explicit_expectation_and_matching_provenance(self):
        self.summary["objective"] = "correct-logprob"
        self.summary["zero_alpha_checks"]["max_absolute_objective_difference"] = 0.
        self.summary["zero_alpha_checks"].pop("max_absolute_margin_difference")
        (self.collection / "summary.json").write_text(json.dumps(self.summary))
        (self.collection / "provenance.json").write_text(json.dumps({"mode": "chat-public-answer", "objective": "correct-logprob"}))
        with self.assertRaisesRegex(ValueError, "objective"):
            profile.analyze(self.collection)  # The old margin solver uses this default.
        result, _ = profile.analyze(self.collection, expected_objective="correct-logprob")
        self.assertEqual(result["objective"], "correct-logprob")
        (self.collection / "provenance.json").write_text(json.dumps({"mode": "chat-public-answer", "objective": "margin"}))
        with self.assertRaisesRegex(ValueError, "objective"):
            profile.analyze(self.collection, expected_objective="correct-logprob")
