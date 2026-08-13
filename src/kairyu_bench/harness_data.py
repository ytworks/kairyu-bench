from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from kairyu_bench.adapter_context import read_context
from kairyu_bench.selection import select_problem_ids


def _load_dataset(
    dataset_id: str,
    revision: str,
    split: str,
    config: str | None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets package is not installed") from error
    kwargs: dict[str, Any] = {"split": split, "revision": revision}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def export_selected_rows(
    context: dict[str, Any],
    split: str,
    id_field: str,
    output_path: Path,
    ids_path: Path,
    *,
    config: str | None = None,
) -> list[str]:
    dataset = context["dataset"]
    rows = list(_load_dataset(dataset["id"], dataset["revision"], split, config))
    canonical_ids: list[str] = []
    for index, row in enumerate(rows):
        if id_field not in row:
            raise ValueError(f"dataset row {index} has no {id_field}")
        canonical_ids.append(str(row[id_field]))
    selected_ids = select_problem_ids(canonical_ids, context.get("limit"))
    selected_set = set(selected_ids)
    selected_rows = [row for row in rows if str(row[id_field]) in selected_set]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in selected_rows
        ),
        encoding="utf-8",
    )
    ids_path.write_text("".join(f"{item_id}\n" for item_id in selected_ids), encoding="utf-8")
    return selected_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.harness_data")
    parser.add_argument("--split", required=True)
    parser.add_argument("--id-field")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    if not context_path:
        parser.error("KAIRYU_BENCH_CONTEXT is not set")
    try:
        context = read_context(Path(context_path))
        if args.cache_only:
            dataset = context["dataset"]
            list(
                _load_dataset(
                    dataset["id"], dataset["revision"], args.split, args.config
                )
            )
        else:
            if not args.id_field or not args.output or not args.ids:
                parser.error("--id-field, --output, and --ids are required for export")
            export_selected_rows(
                context,
                args.split,
                args.id_field,
                args.output,
                args.ids,
                config=args.config,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"kairyu-bench dataset export: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
