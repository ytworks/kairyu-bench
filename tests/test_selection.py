from __future__ import annotations

import unittest

from kairyu_bench.selection import SelectionError, select_problem_ids


class DeterministicSelectionTest(unittest.TestCase):
    def test_limit_takes_prefix_of_canonical_problem_order(self) -> None:
        selected = select_problem_ids(["third", "first", "second"], limit=2)

        self.assertEqual(selected, ["third", "first"])

    def test_none_limit_selects_every_problem(self) -> None:
        selected = select_problem_ids(["q-1", "q-2"], limit=None)

        self.assertEqual(selected, ["q-1", "q-2"])

    def test_zero_limit_is_rejected_instead_of_running_an_empty_benchmark(self) -> None:
        with self.assertRaisesRegex(SelectionError, "positive"):
            select_problem_ids(["q-1"], limit=0)

    def test_duplicate_canonical_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(SelectionError, "duplicate"):
            select_problem_ids(["q-1", "q-1"], limit=None)


if __name__ == "__main__":
    unittest.main()
