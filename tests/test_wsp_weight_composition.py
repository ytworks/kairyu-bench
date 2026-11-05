from __future__ import annotations

import unittest

import numpy as np

from kairyu_bench.wsp.weights.composition import compose_bf16_sets


class CompositionTests(unittest.TestCase):
    def test_overlap_agreement_and_cancellation(self) -> None:
        base = [1, 1, 1, 1]
        a = [1.0078125, 1.0078125, 1.015625, 1]
        b = [1.0078125, 0.9921875, 0.9921875, 1.0078125]
        merged, report = compose_bf16_sets(base, [a, b])
        np.testing.assert_array_equal(merged, [1.015625, 1, 1.0078125, 1.0078125])
        pair = report["pairwise"][0]
        self.assertEqual(pair["overlap_elements"], 3)
        self.assertEqual(pair["same_sign_elements"], 1)
        self.assertEqual(pair["opposite_sign_elements"], 2)
        self.assertEqual(pair["exact_cancellation_elements"], 1)
        self.assertEqual(pair["partial_cancellation_elements"], 1)
        self.assertEqual(report["changed_elements"], 3)
        self.assertAlmostEqual(report["delta_l2"], np.sqrt(6) / 128)

    def test_joint_rounding_is_not_sequential_assignment(self) -> None:
        merged, report = compose_bf16_sets([1], [[0.99609375], [0.99609375]])
        np.testing.assert_array_equal(merged, [0.9921875])
        self.assertEqual(report["components"][0]["changed_elements"], 1)
        self.assertEqual(report["delta_l2"], 1 / 128)

    def test_cancellation_preserves_original_signed_zero(self) -> None:
        merged, report = compose_bf16_sets(np.array([-0.0], np.float32), [[1], [-1]])
        self.assertTrue(np.signbit(merged[0]))
        self.assertEqual(report["stored_bit_changes"], 0)
        self.assertEqual(report["exact_cancellation_elements"], 1)
        self.assertIsNone(report["rounding_relative_l2"])
        self.assertFalse(report["model_response_measured"])

    def test_reject_ambiguous_or_nonfinite_inputs(self) -> None:
        for base, variants in [
            ([1], [[1.001], [1]]),
            ([1], [[1], [float("inf")]]),
            ([1], [[1, 2], [1]]),
            ([1], [[1]]),
            ([complex(1, 1)], [[1], [1]]),
        ]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                compose_bf16_sets(base, variants)


if __name__ == "__main__":
    unittest.main()
