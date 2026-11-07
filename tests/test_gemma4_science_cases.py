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


class BehaviorCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(DIRECTORY))
        cls.cases = module("behavior_cases", "behavior_cases.py")
        cls.offline = module("behavior_offline_fixture", "behavior_offline.py")

    def test_all18_split_is_fixed_and_transfer_rejected_by_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            metadata = self.cases.write_bundle(output)
            train = COMMON.read_cases(output / "train-cases.json")
            ids = [r["id"] for folds in train.values() for fold in folds for r in fold]
            self.assertEqual(len(ids), 12)
            self.assertTrue(
                all(len(fold) == 1 for folds in train.values() for fold in folds)
            )
            self.assertEqual(metadata["split"]["transfer"]["cases"], 6)
            with self.assertRaisesRegex(ValueError, "map purpose"):
                COMMON.read_cases(output / "transfer-cases.json")
            with self.assertRaises(FileExistsError):
                self.cases.write_bundle(output)

    def test_named_entrypoints_are_checked_before_alias_and_never_rescued(self):
        for case in self.cases.catalog():
            if case.name[0] != "C":
                continue
            code, failure = self.offline.executable(case, case.code())
            self.assertIsNone(failure)
            self.assertTrue(code.endswith("solve = " + case.entrypoint + "\n"))
            code, failure = self.offline.executable(case, case.code(True))
            self.assertIsNone(code)
            self.assertFalse(failure["passed"])
            self.assertFalse(failure["numeric_execution"])
        calls = []

        def forbidden(_payload):
            calls.append(1)
            self.fail("unknown or interface error reached worker")

        case = next(c for c in self.cases.catalog() if c.name == "C1")
        self.assertIsNone(self.offline.score(case, {}, forbidden)["passed"])
        error = self.offline.score(
            case, {"error_type": "HTTPError", "http_status": 400}, forbidden
        )
        self.assertIsNone(error["passed"])
        self.assertEqual(error["http_status"], 400)
        self.assertFalse(
            self.offline.score(case, {"public_answer": case.code(True)}, forbidden)[
                "passed"
            ]
        )
        symbolic = next(c for c in self.cases.catalog() if c.name == "B1")
        metadata = self.offline.score(symbolic, {}, forbidden)
        self.assertIsNone(metadata["entrypoint_contract_passed"])
        self.assertIs(metadata["method_requirement_checked"], False)
        self.assertEqual(calls, [])

    def test_pinned_full_split_and_missing_answers_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_path = root / "train.json"
            response_path = root / "responses.json"
            case_path.write_text(json.dumps(self.cases.payload("train")))
            sha = self.offline.sandbox.sha
            document = {
                "schema_version": 1,
                "split": "train",
                "cases_sha256": sha(case_path.read_bytes()),
                "generation": {"reasoning_saved": False},
                "responses": [],
            }
            response_path.write_text(json.dumps(document))

            def read():
                return self.offline.read_inputs(
                    case_path,
                    response_path,
                    sha(case_path.read_bytes()),
                    sha(response_path.read_bytes()),
                    "train",
                )

            selected, answers, _, _ = read()
            self.assertEqual((len(selected), answers), (12, {}))
            document["responses"] = [{"id": "behavior-A3", "public_answer": "private"}]
            response_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
                read()
            data = self.cases.payload("train")
            data["A"][0][0]["prompt"] += "changed"
            case_path.write_text(json.dumps(data))
            document["cases_sha256"] = sha(case_path.read_bytes())
            document["responses"] = []
            response_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "complete fixed split"):
                read()

    def test_fixed_half_independent_of_seed_and_every_reference_is_finite(self):
        import numpy as np

        for case in self.cases.catalog():
            for trial in range(16):
                args, expected = self.cases.sample(
                    case.name, np.random.default_rng(17), trial
                )
                self.assertEqual(expected.dtype, np.float64)
                self.assertTrue(np.isfinite(expected).all())
                if trial < 8:
                    args2, expected2 = self.cases.sample(
                        case.name, np.random.default_rng(999), trial
                    )
                    self.assertTrue(np.array_equal(expected, expected2))
                    self.assertEqual(len(args), len(args2))
                    self.assertTrue(
                        all(np.array_equal(a, b) for a, b in zip(args, args2))
                    )


if __name__ == "__main__":
    unittest.main()
