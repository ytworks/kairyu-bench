from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
sys.path.insert(0, str(DIRECTORY))
import helper_chain_cases as CHAINS  # noqa: E402
import recipe_common as COMMON  # noqa: E402


class HelperChainsTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("scipy"),
        "optional SciPy authored reference environment",
    )
    def test_bundle_and_sequential_public_helpers_preserve_dependency_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = {
                "code": [
                    [{"id": "old0", "prompt": "p", "correct": "c", "foil": "f"}],
                    [{"id": "old1", "prompt": "p", "correct": "c", "foil": "f"}],
                ]
            }
            base = root / "base.json"
            base.write_text(json.dumps(old))
            before = base.read_bytes()
            output = root / "bundle"
            report = CHAINS.write_bundle(
                output,
                base=base,
                base_sha256=hashlib.sha256(before).hexdigest(),
                trials=4,
            )
            combined = json.loads((output / "train-cases.json").read_text())
            self.assertEqual(combined["code"], old["code"])
            self.assertEqual(base.read_bytes(), before)
            self.assertEqual(
                sum(
                    len(f)
                    for fs in COMMON.read_cases(output / "train-cases.json").values()
                    for f in fs
                ),
                10,
            )
            extension = json.loads((output / "train-extension.json").read_text())
            families = {}
            for purpose, folds in extension.items():
                for fold, rows in enumerate(folds):
                    for row in rows:
                        families.setdefault(row["family"], set()).add(fold)
                        self.assertEqual(
                            row["stage"], 1 if purpose == "science_helper_stage" else 2
                        )
            self.assertTrue(all(len(folds) == 1 for folds in families.values()))
            self.assertEqual(
                sum(len(x["foils"]) for x in report["validation"]["chains"]), 24
            )
            with self.assertRaises(FileExistsError):
                CHAINS.write_bundle(output)
            self.assertEqual(base.read_bytes(), before)

            chain = next(
                c for c in CHAINS.catalog() if c.family == "quadrature-mixture"
            )
            called = []

            def request(row):
                called.append(row)
                code = (
                    chain.helper.code(0)
                    if row["id"].endswith("helper")
                    else chain.downstream.code()
                )
                return {
                    "id": row["id"],
                    "public_answer": code,
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                    "seconds": 0,
                    "reasoning_content": "must-not-appear",
                }

            result = CHAINS.evaluate_requests(
                chain, request, CHAINS.trusted_run, trials=4
            )
            self.assertEqual(result["call_count"], 3)
            self.assertFalse(result["helper_stage"]["passed"])
            self.assertTrue(result["oracle_helper_downstream"]["passed"])
            self.assertFalse(result["generated_helper_downstream"]["passed"])
            self.assertFalse(result["whole_chain_passed"])
            self.assertIn("weights.copy()", called[2]["prompt"])
            self.assertNotIn("weights.copy()", called[1]["prompt"])
            self.assertNotIn("must-not-appear", json.dumps(result))
            self.assertNotIn(
                "public_answer", json.dumps(result).replace("public_answers_saved", "")
            )
            with self.assertRaises(ValueError):
                CHAINS.trusted_run({"code": "raise RuntimeError('arbitrary')"})


if __name__ == "__main__":
    unittest.main()
