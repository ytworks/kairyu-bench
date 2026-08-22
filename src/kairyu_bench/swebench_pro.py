from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _rows(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    problem_ids: set[str] = set()
    lines = source.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} is not an object")
        instance_id = row.get("instance_id")
        dockerhub_tag = row.get("dockerhub_tag")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"line {line_number} has no instance_id")
        if instance_id in problem_ids:
            raise ValueError(f"line {line_number} has duplicate instance_id")
        if not isinstance(dockerhub_tag, str) or not dockerhub_tag:
            raise ValueError(f"line {line_number} has no dockerhub_tag")
        rows.append(row)
        problem_ids.add(instance_id)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_dataset(
    source: Path,
    destination: Path,
    instance_id: str | None = None,
) -> list[str]:
    rows: list[dict[str, Any]] = []
    problem_ids: list[str] = []
    for source_row in _rows(source):
        row_instance_id = source_row["instance_id"]
        if instance_id is not None and row_instance_id != instance_id:
            continue
        row = dict(source_row)
        row["docker_image"] = f"jefzda/sweap-images:{row['dockerhub_tag']}"
        rows.append(row)
        problem_ids.append(row_instance_id)
    if instance_id is not None and not rows:
        raise ValueError(f"instance_id not found: {instance_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return problem_ids


def image_reference(source: Path, instance_id: str) -> str:
    for row in _rows(source):
        if row["instance_id"] == instance_id:
            return f"jefzda/sweap-images:{row['dockerhub_tag']}"
    raise ValueError(f"instance_id not found: {instance_id}")


def prepare_predictions(
    source: Path,
    destination: Path,
    expected_ids: list[str] | None = None,
    select_id: str | None = None,
) -> list[str]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mini-SWE-agent predictions must be an object")
    predictions: list[dict[str, str]] = []
    for instance_id, value in payload.items():
        if select_id is not None and instance_id != select_id:
            continue
        if not isinstance(instance_id, str) or not isinstance(value, dict):
            raise ValueError("mini-SWE-agent prediction is malformed")
        patch = value.get("model_patch")
        predictions.append(
            {
                "instance_id": instance_id,
                "patch": patch if isinstance(patch, str) else "",
                "prefix": "kairyu-bench",
            }
        )
    prediction_ids = [prediction["instance_id"] for prediction in predictions]
    if expected_ids is not None:
        missing = sorted(set(expected_ids) - set(prediction_ids))
        unexpected = sorted(set(prediction_ids) - set(expected_ids))
        if missing or unexpected or len(prediction_ids) != len(expected_ids):
            raise ValueError(
                "mini-SWE-agent prediction IDs differ from selected IDs: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    if select_id is not None and not predictions:
        raise ValueError(f"prediction not found: {select_id}")
    _write_json(destination, predictions)
    return prediction_ids


def record_outcome(source: Path, destination: Path, expected_id: str) -> bool:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {expected_id}:
        raise ValueError("official evaluator did not return the expected instance")
    outcome = payload[expected_id]
    if not isinstance(outcome, bool):
        raise ValueError("official evaluator outcome is not boolean")
    combined: dict[str, bool] = {}
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in existing.items()
        ):
            raise ValueError("aggregate outcomes are malformed")
        combined.update(existing)
    if expected_id in combined:
        raise ValueError(f"duplicate official outcome: {expected_id}")
    combined[expected_id] = outcome
    _write_json(destination, combined)
    return outcome


def _expected_ids(path: Path) -> list[str]:
    expected_ids = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected instance IDs contain duplicates")
    return expected_ids


def aggregate_items(
    items_path: Path,
    predictions_path: Path,
    outcomes_path: Path,
    expected_ids_path: Path,
) -> None:
    expected_ids = _expected_ids(expected_ids_path)
    predictions: list[dict[str, str]] = []
    outcomes: dict[str, bool] = {}
    for index, expected_id in enumerate(expected_ids, 1):
        item = items_path / f"{index:04d}"
        item_predictions = json.loads(
            (item / "predictions.json").read_text(encoding="utf-8")
        )
        item_outcomes = json.loads(
            (item / "evaluation/eval_results.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(item_predictions, list)
            or len(item_predictions) != 1
            or not isinstance(item_predictions[0], dict)
            or item_predictions[0].get("instance_id") != expected_id
        ):
            raise ValueError(f"item {index} prediction does not match {expected_id}")
        if not isinstance(item_outcomes, dict) or set(item_outcomes) != {expected_id}:
            raise ValueError(f"item {index} outcome does not match {expected_id}")
        outcome = item_outcomes[expected_id]
        if not isinstance(outcome, bool):
            raise ValueError(f"item {index} outcome is not boolean")
        predictions.append(item_predictions[0])
        outcomes[expected_id] = outcome
    _write_json(predictions_path, predictions)
    _write_json(outcomes_path, outcomes)


def verify_complete(
    predictions_path: Path,
    outcomes_path: Path,
    expected_ids_path: Path,
) -> None:
    expected_ids = _expected_ids(expected_ids_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    if isinstance(predictions, dict):
        prediction_ids = list(predictions)
    elif isinstance(predictions, list) and all(
        isinstance(prediction, dict)
        and isinstance(prediction.get("instance_id"), str)
        for prediction in predictions
    ):
        prediction_ids = [prediction["instance_id"] for prediction in predictions]
    else:
        raise ValueError("SWE-bench Pro predictions are malformed")
    if not isinstance(outcomes, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in outcomes.items()
    ):
        raise ValueError("official evaluator outcomes are malformed")
    expected = set(expected_ids)
    if (
        len(prediction_ids) != len(expected_ids)
        or set(prediction_ids) != expected
        or set(outcomes) != expected
    ):
        raise ValueError(
            "completed prediction/outcome IDs differ from selected IDs: "
            f"expected={len(expected)}, predictions={len(prediction_ids)}, "
            f"outcomes={len(outcomes)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.swebench_pro")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset")
    dataset.add_argument("source", type=Path)
    dataset.add_argument("destination", type=Path)
    dataset.add_argument("--instance-id")
    image = commands.add_parser("image")
    image.add_argument("source", type=Path)
    image.add_argument("instance_id")
    predictions = commands.add_parser("predictions")
    predictions.add_argument("source", type=Path)
    predictions.add_argument("destination", type=Path)
    predictions.add_argument("--expected-ids", type=Path)
    predictions.add_argument("--select-id")
    outcome = commands.add_parser("record-outcome")
    outcome.add_argument("source", type=Path)
    outcome.add_argument("destination", type=Path)
    outcome.add_argument("--expected-id", required=True)
    aggregate = commands.add_parser("aggregate-items")
    aggregate.add_argument("items", type=Path)
    aggregate.add_argument("predictions", type=Path)
    aggregate.add_argument("outcomes", type=Path)
    aggregate.add_argument("expected_ids", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("predictions", type=Path)
    verify.add_argument("outcomes", type=Path)
    verify.add_argument("expected_ids", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            prepare_dataset(args.source, args.destination, args.instance_id)
        elif args.command == "image":
            print(image_reference(args.source, args.instance_id))
        elif args.command == "predictions":
            expected_ids = None
            if args.expected_ids is not None:
                expected_ids = [
                    line
                    for line in args.expected_ids.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line
                ]
            prepare_predictions(
                args.source,
                args.destination,
                expected_ids,
                args.select_id,
            )
        elif args.command == "record-outcome":
            resolved = record_outcome(args.source, args.destination, args.expected_id)
            print("true" if resolved else "false")
        elif args.command == "aggregate-items":
            aggregate_items(
                args.items,
                args.predictions,
                args.outcomes,
                args.expected_ids,
            )
        else:
            verify_complete(args.predictions, args.outcomes, args.expected_ids)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"kairyu-bench SWE-bench Pro preparation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
