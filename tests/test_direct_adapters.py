from __future__ import annotations

import unittest

from kairyu_bench.direct import (
    deterministic_gpqa_choices,
    execute_direct,
    extract_choice,
    longbench_item,
    mrcr_score,
    token_bin,
)


class _ChoiceClient:
    def chat(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> str:
        return "Answer: D"


class DirectOfficialScoringTest(unittest.TestCase):
    def test_gpqa_execution_scores_real_normalized_rows(self) -> None:
        context = {
            "run_id": "run-1",
            "benchmark": "gpqa-diamond",
            "endpoint_fingerprint": "sha256:0123456789abcdef",
            "model_id": "chat-capable",
            "limit": 1,
            "source": {
                "repository": "https://github.com/idavidrein/gpqa",
                "revision": "56686c06f5e19865c153de0fdb11be3890014df7",
            },
            "dataset": {
                "id": "Idavidrein/gpqa",
                "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
            },
            "scoring": {
                "method": "gpqa-diamond-exact-choice-match",
                "unit": "percent",
            },
        }
        rows = [
            {
                "Question": "Which answer is correct?",
                "Correct Answer": "right",
                "Incorrect Answer 1": "wrong-1",
                "Incorrect Answer 2": "wrong-2",
                "Incorrect Answer 3": "wrong-3",
            }
        ]

        result, records = execute_direct(context, rows, _ChoiceClient())  # type: ignore[arg-type]

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.data["score"]["primary"], 100.0)
        self.assertEqual(result.data["selection"]["problem_ids"], ["gpqa-diamond-0000"])
        self.assertEqual(records[0]["score"], 1.0)

    def test_gpqa_choice_order_is_stable_and_correct_letter_tracks_shuffle(self) -> None:
        choices, correct = deterministic_gpqa_choices(
            "q-7", "right", ["wrong-1", "wrong-2", "wrong-3"]
        )

        self.assertEqual(choices, ["wrong-3", "wrong-2", "wrong-1", "right"])
        self.assertEqual(correct, "D")

    def test_choice_extraction_prefers_explicit_final_answer(self) -> None:
        self.assertEqual(
            extract_choice("A and B are tempting. Therefore, Answer: C"), "C"
        )
        self.assertEqual(extract_choice("D"), "D")
        self.assertIsNone(extract_choice("I cannot decide between A and B"))

    def test_longbench_item_preserves_official_context_and_choices(self) -> None:
        item = longbench_item(
            {
                "_id": "lb-1",
                "context": "A very long source document.",
                "question": "Which option follows?",
                "choice_A": "alpha",
                "choice_B": "beta",
                "choice_C": "gamma",
                "choice_D": "delta",
                "answer": "b",
            }
        )

        self.assertEqual(item["id"], "lb-1")
        self.assertEqual(item["expected"], "B")
        self.assertIn("A very long source document.", item["messages"][0]["content"])
        self.assertIn("B) beta", item["messages"][0]["content"])

    def test_mrcr_uses_prepend_gate_then_sequence_matcher_ratio(self) -> None:
        self.assertEqual(mrcr_score("wrongabc", "hashabc", "hash"), 0.0)
        self.assertEqual(mrcr_score("hashabc", "hashabc", "hash"), 1.0)
        self.assertAlmostEqual(mrcr_score("hashabc", "hashabd", "hash"), 2 / 3)

    def test_mrcr_official_bins_include_only_up_to_128k(self) -> None:
        cases = [
            (4095, None),
            (4096, 8192),
            (8192, 8192),
            (8193, 16384),
            (131072, 131072),
            (131073, None),
        ]

        for tokens, expected in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(token_bin(tokens), expected)


if __name__ == "__main__":
    unittest.main()
