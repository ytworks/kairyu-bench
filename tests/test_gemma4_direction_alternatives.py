"""Saved-gradient alternative probes report extrapolation and real BF16 failures."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"))
import probe_direction_alternatives as tool  # noqa: E402


class DirectionAlternativeTests(unittest.TestCase):
    def test_uniform_direction_keeps_cost_tradeoff_and_probability_ceiling(self):
        result = tool.uniform_donor([[1.], [-2.], [0.]], [[.5], [1.], [-3.]], [10.],
                                    [("code", 0), ("tools", 0), ("general", 0)], [-.001, -1., -2.])
        self.assertFalse(result["radius_one_applied"])
        self.assertFalse(result["actual_bf16_rounding_measured"])
        first, _, last = result["rows"]
        self.assertAlmostEqual(first["continuous_weight_l2"], float(np.float32(.05)) * 10)
        self.assertEqual(last["continuous_weight_l2"], 2.5)  # Not silently clipped to the old radius.
        self.assertEqual(first["groups"]["primary"]["correct_ll_positive"], 1)
        self.assertEqual(first["groups"]["related"]["correct_ll_negative"], 1)
        self.assertEqual(first["linear_logprob_ceiling_exceed_folds"], [0])
        self.assertEqual(first["margin_prediction"][2], -3 * float(np.float32(.05)))

    def test_one_native_residual_correction_checks_actual_rounding(self):
        base, frozen = np.array([1.]), np.array([1.], dtype=np.float32)
        values, passed = tool.native_residual_once(base, frozen, [[1.]], [.004])
        self.assertEqual(values[0], np.float32(1.0078125))
        self.assertTrue(passed["eligible_for_native_finite"])
        self.assertEqual(passed["solver_calls"], 1)
        self.assertFalse(passed["donor_coefficients_used"])
        self.assertAlmostEqual(passed["actual_native_l2"], .0078125)
        _, rejected = tool.native_residual_once(base, frozen, [[1.]], [.001])
        self.assertFalse(rejected["eligible_for_native_finite"])
        self.assertAlmostEqual(rejected["maximum_shortfall"], .001)
        self.assertEqual(rejected["solver_calls"], 1)
        self.assertFalse(rejected["actual_model_response_measured"])


if __name__ == "__main__":
    unittest.main()
