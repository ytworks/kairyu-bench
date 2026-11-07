from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
sys.path.insert(0, str(DIRECTORY))
import advanced_science_cases as ADVANCED  # noqa: E402
import science_sandbox as SANDBOX  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "advanced_case_reader", DIRECTORY / "recipe_common.py"
)
COMMON = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = COMMON
spec.loader.exec_module(COMMON)


class AdvancedScienceCalibrationTest(unittest.TestCase):
    def test_trusted_correct_programs_and_foils_on_new_numeric_inputs(self):
        report = ADVANCED.validate(seed=934821, trials=8, revision=2)
        self.assertTrue(all(row["correct"]["failed"] == 0 for row in report["cases"]))
        self.assertTrue(all(row["foil"]["failed"] > 0 for row in report["cases"]))
        self.assertFalse(report["token_count_measured"])

    def test_split_export_reaches_existing_collector_and_sandbox_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "cases"
            ADVANCED.write_bundle(bundle, trials=4, revision=2)
            train = COMMON.read_cases(bundle / "train-cases.json")
            self.assertEqual(
                sum(len(fold) for folds in train.values() for fold in folds), 16
            )
            self.assertEqual(len(train), 4)
            with self.assertRaises(ValueError):
                COMMON.read_cases(bundle / "transfer-cases.json")
            transfer = [case for case in ADVANCED.catalog() if case.split == "transfer"]
            with self.assertRaisesRegex(ValueError, "transfer cases"):
                ADVANCED.shared.training_payload(transfer)
            source = bundle / "train-cases.json"
            data, receipt = SANDBOX.capture(source, 8 * 1024 * 1024)
            rows = [
                row
                for folds in json.loads(data).values()
                for fold in folds
                for row in fold
            ]
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": "train",
                        "cases_sha256": receipt["sha256"],
                        "generation": {"reasoning_saved": False},
                        "responses": [
                            {"id": row["id"], "public_answer": row["correct"]}
                            for row in rows
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "complete authored split"):
                SANDBOX.read_inputs(source, responses)
            with self.assertRaisesRegex(ValueError, "authored case content mismatch"):
                SANDBOX.read_inputs(source, responses, case_module=ADVANCED)
            report = SANDBOX.main(
                [
                    "--cases",
                    str(source),
                    "--responses",
                    str(responses),
                    "--catalog",
                    "advanced-v2",
                    "--output",
                    str(root / "preview"),
                    "--docker-socket",
                    "/not-opened.sock",
                    "--daemon-id",
                    "not-contacted",
                    "--image",
                    "sha256:" + "a" * 64,
                ],
                docker_factory=lambda *a, **k: self.fail(
                    "preview must not contact Docker"
                ),
            )
            self.assertEqual(report["expected_cases"], 16)
            self.assertEqual(report["responses_received"], 16)
            case = ADVANCED.catalog()[0]
            payload, expected = SANDBOX.prepare(
                case, case.code(), 5738, 4, sample_fn=ADVANCED.sample
            )
            self.assertEqual(len(payload["trials"]), len(expected))
            self.assertTrue(all(value.shape[1] == 2 for value in expected))
            self.assertNotIn("correct", payload)
            self.assertNotIn("reference", payload)


if __name__ == "__main__":
    unittest.main()
