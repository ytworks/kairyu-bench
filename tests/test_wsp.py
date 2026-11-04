from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.report import compare
from kairyu_bench.wsp.select import assign_splits, split_order
from kairyu_bench.wsp.sources import Loader, adapters


def _item(source: str, samples: int = 1) -> Item:
    return Item(f"{source}:x", source, "dev", 10, 16384, samples, "", 0)


def _reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {"content": content, "finish_reason": finish_reason, "completion_tokens": 10, "prompt_tokens": 10}


class SplitTest(unittest.TestCase):
    def test_one_problem_in_several_sources_never_spans_dev_and_test(self) -> None:
        candidates = []
        for number in range(40):
            # MMLU-Pro and its Japanese translation share question ids.
            candidates.append((f"mmlu-pro:{number}", "law" if number % 2 else "math", f"mmlu:{number}"))
            candidates.append((f"mmlu-prox-ja:{number}", "law", f"mmlu:{number}"))
        candidates += [(f"gpqa-diamond:{n:04d}", "Physics", f"gpqa-diamond:{n:04d}") for n in range(10)]

        split_of = assign_splits(candidates)

        for number in range(40):
            self.assertEqual(split_of[f"mmlu-pro:{number}"], split_of[f"mmlu-prox-ja:{number}"])
        english = [item for item, _, _ in candidates if item.startswith("mmlu-pro:")]
        orders = split_order("mmlu-pro", [(item, "s") for item in english], split_of)
        self.assertEqual(len(orders["dev"]), len(orders["test"]))
        self.assertEqual(sum(split_of[f"gpqa-diamond:{n:04d}"] == "dev" for n in range(10)), 5)


class ComparisonTest(unittest.TestCase):
    def _run(self, root: Path, name: str, scores: dict[str, float]) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps({"tier": 10, "split": "dev", "manifest_sha256": "m", "label": name, "model_id": "g"}),
            encoding="utf-8",
        )
        with (run_dir / "items.jsonl").open("w", encoding="utf-8") as stream:
            for item_id, score in scores.items():
                source = item_id.split(":")[0]
                record = {"id": item_id, "source": source, "kind": "chat", "score": score, "outcome": {"samples": []}}
                stream.write(json.dumps(record) + "\n")
        return run_dir

    def test_paired_deltas_flips_and_intervals_are_reproducible(self) -> None:
        base_scores = {**{f"mmlu-pro:{n}": 1.0 for n in range(10)}, **{f"ifeval:{n}": 0.0 for n in range(10)}}
        edited_scores = dict(base_scores)
        edited_scores.update({f"mmlu-pro:{n}": 0.0 for n in range(4)})  # four answers lost
        edited_scores.update({f"ifeval:{n}": 1.0 for n in range(2)})  # two gained
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._run(root, "base", base_scores)
            edited = self._run(root, "edited", edited_scores)

            first = compare(base, edited)
            second = compare(base, edited)

        self.assertAlmostEqual(first["areas"]["knowledge"]["delta"], -0.4)
        self.assertAlmostEqual(first["areas"]["instructions"]["delta"], 0.2)
        self.assertAlmostEqual(first["overall"]["delta"], (-0.4 + 0.2) / 2)
        self.assertEqual(first["areas"]["knowledge"]["flips"], {"lost": 4, "gained": 0, "items": 10})
        self.assertEqual(first["areas"]["instructions"]["flips"], {"lost": 0, "gained": 2, "items": 10})
        low, high = first["overall"]["ci95"]
        self.assertLessEqual(low, first["overall"]["delta"])
        self.assertGreaterEqual(high, first["overall"]["delta"])
        self.assertEqual(first, second)


class ScoringTest(unittest.TestCase):
    """Official extraction rules: a right reply scores 1, a wrong one 0, and a
    reply cut at the token cap 0 whatever it says."""

    CASES: list[tuple[str, dict[str, Any], str, str]] = [
        ("gpqa-diamond", {"expected": "C"}, "Reasoning...\nAnswer: C", "Answer: B"),
        ("mmlu-pro", {"expected": "D"}, "So the answer is (D).", "The answer is (A)."),
        ("mmlu-prox-ja", {"expected": "b"}, "よって、答えは (B) です", "答えは (C) です"),
        ("jamc-qa", {"expected": "A"}, "考察...\n答え: (A)", "答え: (D)"),
        ("gsm8k-platinum", {"target": "18"}, "Work.\nAnswer: 18", "Answer: 17"),
        ("mgsm", {"target": "18", "language": "ja"}, "計算...\n答え: 18", "答え: 20"),
        ("triviaqa", {"aliases": ["eritrea"]}, "The answer is Eritrea.", "The answer is Ethiopia."),
        ("musr", {"choices": ["a", "b"], "gold": 2}, "Thinking.\nANSWER: 2", "ANSWER: 1"),
        ("mmmu-pro", {"options": ["x", "y", "z"], "answer": "B"}, "Steps.\nAnswer: B", "Answer: C"),
        ("ruler", {"outputs": ["1234567"]}, "The number is 1234567.", "The number is 7654321."),
        ("mrcr-v2", {"answer": "pfx hello world", "prepend": "pfx"}, "pfx hello world", "hello world"),
        ("longbench-v2", {"expected": "A"}, "Answer: A", "Answer: B"),
    ]

    def test_official_rules_score_right_wrong_and_truncated_replies(self) -> None:
        registry = adapters()
        for source, row, right, wrong in self.CASES:
            with self.subTest(source=source):
                outcomes = [{"samples": [_reply(right)]}, {"samples": [_reply(wrong)]}, {"samples": [_reply(right, "length")]}]
                scored = registry[source].score([(_item(source), row, outcome) for outcome in outcomes], Loader())
                self.assertEqual([entry.score for entry in scored], [1.0, 0.0, 0.0])

    def test_choice_log_likelihood_uses_length_normalization_where_lm_eval_does(self) -> None:
        # The longer correct ending wins only after normalising by its bytes.
        outcome = {"logliks": [-8.0, -5.0], "lengths": [16, 2], "gold": 0}
        registry = adapters()
        hellaswag = registry["hellaswag"].score([(_item("hellaswag"), {}, outcome)], Loader())
        winogrande = registry["winogrande"].score([(_item("winogrande"), {}, outcome)], Loader())

        self.assertEqual((hellaswag[0].score, winogrande[0].score), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
