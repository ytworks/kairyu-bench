from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.swebench_pro import (
    aggregate_items,
    image_reference,
    prepare_dataset,
    prepare_predictions,
    record_outcome,
    verify_complete,
)


class SwebenchProPreparationTest(unittest.TestCase):
    def test_dataset_uses_official_dockerhub_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "selected.jsonl"
            destination = root / "dataset" / "test.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "instance_id": "instance-1",
                        "dockerhub_tag": "repo.instance-1",
                        "problem_statement": "Fix it",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            ids = prepare_dataset(source, destination)
            row = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(ids, ["instance-1"])
        self.assertEqual(row["docker_image"], "jefzda/sweap-images:repo.instance-1")

    def test_dataset_can_select_one_instance_for_bounded_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "selected.jsonl"
            destination = root / "dataset" / "test.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "instance_id": f"instance-{number}",
                            "dockerhub_tag": f"repo.instance-{number}",
                        }
                    )
                    for number in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )

            ids = prepare_dataset(source, destination, "instance-2")
            row = json.loads(destination.read_text(encoding="utf-8"))
            image = image_reference(source, "instance-2")

        self.assertEqual(ids, ["instance-2"])
        self.assertEqual(image, "jefzda/sweap-images:repo.instance-2")
        self.assertEqual(row["instance_id"], "instance-2")

    def test_predictions_convert_mini_swe_agent_mapping_to_official_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "preds.json"
            destination = root / "predictions.json"
            source.write_text(
                json.dumps(
                    {
                        "instance-1": {
                            "instance_id": "instance-1",
                            "model_patch": "diff --git a/a b/a",
                        }
                    }
                ),
                encoding="utf-8",
            )

            ids = prepare_predictions(source, destination)
            predictions = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(ids, ["instance-1"])
        self.assertEqual(predictions[0]["instance_id"], "instance-1")
        self.assertEqual(predictions[0]["patch"], "diff --git a/a b/a")

    def test_predictions_require_every_selected_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "preds.json"
            destination = root / "predictions.json"
            source.write_text(
                json.dumps({"instance-1": {"model_patch": "patch"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing=1"):
                prepare_predictions(
                    source,
                    destination,
                    ["instance-1", "instance-2"],
                )

    def test_predictions_can_select_current_instance_from_running_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "preds.json"
            destination = root / "item-prediction.json"
            source.write_text(
                json.dumps(
                    {
                        "instance-1": {"model_patch": "patch-1"},
                        "instance-2": {"model_patch": "patch-2"},
                    }
                ),
                encoding="utf-8",
            )

            ids = prepare_predictions(
                source,
                destination,
                ["instance-2"],
                "instance-2",
            )
            predictions = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(ids, ["instance-2"])
        self.assertEqual(
            predictions,
            [
                {
                    "instance_id": "instance-2",
                    "patch": "patch-2",
                    "prefix": "kairyu-bench",
                }
            ],
        )

    def test_outcomes_accumulate_and_complete_verification_checks_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item_one = root / "one.json"
            item_two = root / "two.json"
            outcomes = root / "outcomes.json"
            predictions = root / "preds.json"
            expected = root / "instance-ids.txt"
            item_one.write_text('{"instance-1": true}', encoding="utf-8")
            item_two.write_text('{"instance-2": false}', encoding="utf-8")
            predictions.write_text(
                json.dumps(
                    {
                        "instance-1": {"model_patch": "patch-1"},
                        "instance-2": {"model_patch": "patch-2"},
                    }
                ),
                encoding="utf-8",
            )
            expected.write_text("instance-1\ninstance-2\n", encoding="utf-8")

            self.assertTrue(record_outcome(item_one, outcomes, "instance-1"))
            self.assertFalse(record_outcome(item_two, outcomes, "instance-2"))
            verify_complete(predictions, outcomes, expected)
            recorded = json.loads(outcomes.read_text(encoding="utf-8"))

        self.assertEqual(
            recorded,
            {"instance-1": True, "instance-2": False},
        )

    def test_parallel_item_results_aggregate_in_official_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = root / "items"
            expected = root / "instance-ids.txt"
            predictions = root / "predictions.json"
            outcomes = root / "eval_results.json"
            expected.write_text("instance-1\ninstance-2\n", encoding="utf-8")
            for index, resolved in ((1, True), (2, False)):
                item = items / f"{index:04d}"
                (item / "evaluation").mkdir(parents=True)
                (item / "predictions.json").write_text(
                    json.dumps(
                        [
                            {
                                "instance_id": f"instance-{index}",
                                "patch": f"patch-{index}",
                                "prefix": "kairyu-bench",
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                (item / "evaluation/eval_results.json").write_text(
                    json.dumps({f"instance-{index}": resolved}),
                    encoding="utf-8",
                )

            aggregate_items(items, predictions, outcomes, expected)
            verify_complete(predictions, outcomes, expected)

            self.assertEqual(
                [
                    prediction["instance_id"]
                    for prediction in json.loads(
                        predictions.read_text(encoding="utf-8")
                    )
                ],
                ["instance-1", "instance-2"],
            )
            self.assertEqual(
                json.loads(outcomes.read_text(encoding="utf-8")),
                {"instance-1": True, "instance-2": False},
            )
