from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


SCIENCE = module("science_cases_fixture", "science_cases.py")
COMMON = module("science_case_reader_fixture", "recipe_common.py")


class ScienceCalibrationTest(unittest.TestCase):
    def test_general_programs_pass_varied_inputs_and_every_foil_fails(self):
        # A seed different from the exported default probes new numeric values.
        report = SCIENCE.validate(SCIENCE.catalog(), seed=48721, trials=8)
        self.assertTrue(all(row["correct"]["failed"] == 0 for row in report["cases"]))
        self.assertTrue(all(row["foil"]["failed"] > 0 for row in report["cases"]))
        self.assertTrue(
            all(row["max_prompt_continuation_bytes"] <= 2048 for row in report["cases"])
        )
        self.assertFalse(report["token_count_measured"])

    def test_train_accepted_transfer_rejected_by_actual_collector_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "science"
            metadata = SCIENCE.write_bundle(output, trials=4)
            train = COMMON.read_cases(output / "train-cases.json")
            self.assertEqual(
                sum(len(fold) for folds in train.values() for fold in folds), 16
            )
            transfer = json.loads((output / "transfer-cases.json").read_text())
            self.assertEqual(len(transfer["cases"]), 8)
            self.assertFalse(transfer["gradient_allowed"])
            self.assertTrue(
                set(metadata["training_families"]).isdisjoint(
                    metadata["transfer_families"]
                )
            )
            with self.assertRaisesRegex(ValueError, "map purpose"):
                COMMON.read_cases(output / "transfer-cases.json")
            with self.assertRaisesRegex(ValueError, "transfer cases"):
                SCIENCE.training_payload(
                    [c for c in SCIENCE.catalog() if c.split == "transfer"]
                )

    def test_existing_output_is_preserved_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "science"
            output.mkdir()
            marker = output / "retain"
            marker.write_text("original")
            with self.assertRaises(FileExistsError):
                SCIENCE.write_bundle(output)
            self.assertEqual(marker.read_text(), "original")
            self.assertEqual([p.name for p in output.iterdir()], ["retain"])


if __name__ == "__main__":
    unittest.main()
