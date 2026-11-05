from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from kairyu_bench.wsp.weights.finite_response import (
    bf16_realization,
    pareto_indices,
    project_bf16_response,
    solve_minimum_change,
    signed_simplex_grid,
)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional finite-response solver requires scipy")
class SolverTests(unittest.TestCase):
    def test_known_minimum_and_signed_simplex(self) -> None:
        result = solve_minimum_change([[1, -1, 1, -1]], np.diag([1, 1, 4, 4]), [0.5])
        selected = result["selected"]
        self.assertEqual(result["status"], "surrogate_candidate")
        np.testing.assert_allclose(selected["coefficients"], [0.4, 0, 0.1, 0], atol=1e-6)
        self.assertAlmostEqual(selected["continuous_l2"], np.sqrt(0.2), places=6)
        self.assertTrue(result["all_branches_resolved"])
        self.assertEqual(result["model_calls"], 0)
        self.assertIn("requires_new_finite", result["response_kind"])

    def test_infeasible_and_conditioning_are_distinct(self) -> None:
        infeasible = solve_minimum_change([[1, -1]], [[1, -1], [-1, 1]], [2])
        self.assertEqual(infeasible["branch_counts"], {"lp_infeasible": 2})
        self.assertIsNone(infeasible["selected"])
        singular = solve_minimum_change([[1, -1, 1, -1]], np.ones((4, 4)), [0.1])
        self.assertEqual(singular["branch_counts"], {"excluded_conditioning": 4})
        self.assertFalse(singular["all_branches_resolved"])

    def test_auxiliary_proxy_constraint_changes_feasibility(self) -> None:
        result = solve_minimum_change(
            [[1, -1]], np.eye(2), [0.5],
            auxiliary=[[1, 0]], auxiliary_upper=[0.25],
        )
        self.assertIsNone(result["selected"])

    def test_negative_margin_floor_is_explicit(self) -> None:
        result = solve_minimum_change([[1, -1], [-0.2, 0.2]], np.eye(2), [0.2, -0.05])
        np.testing.assert_allclose(result["selected"]["predicted_response"], [0.2, -0.04], atol=1e-7)

    def test_quantized_active_cone_projection_and_stale_gram(self) -> None:
        # Half an ulp disappears at 1, but remains representable at 0.
        base = [1, 0]
        directions = np.array([[1 / 128, -1 / 128, 0, 0],
                               [0, 0, 1 / 128, -1 / 128]])
        gram = directions.T @ directions
        values, report = project_bf16_response(
            base, directions, [0.5, 0, 0.5, 0], [[1, -1, 2, -2]], gram,
        )
        np.testing.assert_array_equal(values, [1, 1 / 256])
        np.testing.assert_allclose(report["projected_coefficients"], [0, 0, 0.5, 0])
        self.assertEqual(report["projection_relative_residual"], 0)
        self.assertAlmostEqual(report["projected_response"][0], 1)
        self.assertTrue(report["projection_in_measured_simplex"])
        self.assertGreater(report["rounding_relative_l2"], 0.5)
        with self.assertRaises(ValueError):
            project_bf16_response(base, directions, [0.5, 0, 0.5, 0],
                                 [[1, -1, 2, -2]], gram * 2)

    def test_projection_does_not_invent_reverse_or_new_directions(self) -> None:
        directions = np.array([[1 / 128, -1 / 128, 0, 0],
                               [1 / 128, -1 / 128, 1, -1]])
        _, report = project_bf16_response(
            [1, 0], directions, [0.5, 0, 0, 0],
            [[1, -1, 100, -100]], directions.T @ directions,
        )
        np.testing.assert_allclose(report["projected_coefficients"], [0.25, 0, 0, 0])
        self.assertAlmostEqual(report["projection_relative_residual"], 2 ** -0.5)
        self.assertEqual(report["projected_response"], [0.25])

    def test_invalid_shapes_gram_and_nonfinite(self) -> None:
        cases = [
            ([[1, -1]], [[1, 2], [0, 1]], [0.1]),
            ([[1, -1]], [[1, 2], [2, 1]], [0.1]),
            ([[1, float("nan")]], np.eye(2), [0.1]),
            ([[1, -1]], np.eye(2), [0.1, 0.2]),
            ([[1, -1, 1]], np.eye(3), [0.1]),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                solve_minimum_change(*arguments)


class RealizationTests(unittest.TestCase):
    def test_independent_bf16_ties_to_even_and_exact_delta(self) -> None:
        base = np.array([1, 1 + 1 / 128], dtype=np.float32)
        delta = np.array([[1 / 256, -1 / 256], [1 / 256, -1 / 256]])
        stored, metrics = bf16_realization(base, delta, [1, 0], chunk_elements=1)
        np.testing.assert_array_equal(stored, [1, 1 + 2 / 128])
        self.assertEqual(metrics["changed_elements"], 1)
        self.assertEqual(metrics["realized_l2"], 1 / 128)
        self.assertFalse(metrics["model_response_measured"])
        self.assertEqual(metrics["rounding_relative_l2"], 1)

    def test_grid_is_finite_one_sign_and_unique(self) -> None:
        grid = signed_simplex_grid(14)
        self.assertEqual(len(grid), 1498)
        self.assertEqual(len({tuple(c) for c in grid}), len(grid))
        for coefficient in grid:
            self.assertLessEqual(coefficient.sum(), 1)
            self.assertLessEqual(np.count_nonzero(coefficient), 3)
            self.assertFalse(np.any((coefficient[::2] > 0) & (coefficient[1::2] > 0)))
        with self.assertRaises(ValueError):
            signed_simplex_grid(18)

    def test_zero_and_chunk_invariance(self) -> None:
        base, delta = [0, 1, -2], [[1, -1], [0.25, -0.25], [-0.5, 0.5]]
        values, small = bf16_realization(base, delta, [0.5, 0], chunk_elements=1)
        other_values, large = bf16_realization(base, delta, [0.5, 0], chunk_elements=16)
        np.testing.assert_array_equal(values, other_values)
        self.assertEqual(small, large)
        _, zero = bf16_realization(base, delta, [0, 0])
        self.assertEqual(zero["changed_elements"], 0)
        self.assertIsNone(zero["rounding_relative_l2"])
        self.assertIsNone(zero["cosine_realized_to_continuous"])

    def test_reject_invalid_composition(self) -> None:
        for base, delta, coefficients in [
            ([1.001], [[1, -1]], [1, 0]),
            ([1 + 2 ** -40], [[1, -1]], [1, 0]),
            ([1], [[1, -1]], [0.5, 0.5]),
            ([1], [[1, -1]], [1.1, 0]),
            ([1], [[float("nan"), 0]], [1, 0]),
            ([1], [[1e300, 0]], [1, 0]),
        ]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                bf16_realization(base, delta, coefficients)

    def test_pareto_has_no_hidden_weighted_average(self) -> None:
        self.assertEqual(
            pareto_indices([[1, 0], [0, 1], [0, 0], [1, 0]], [[1], [1], [2], [1]]),
            [0, 1, 3],
        )
        with self.assertRaises(ValueError):
            pareto_indices([[float("nan")]], [[1]])


if __name__ == "__main__":
    unittest.main()

