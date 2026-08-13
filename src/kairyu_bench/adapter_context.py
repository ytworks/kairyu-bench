from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def read_context(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read adapter context: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("adapter context must be an object")
    return value


def context_value(context: dict[str, Any], dotted_field: str) -> Any:
    value: Any = context
    for part in dotted_field.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
            continue
        if not part or not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_field)
        value = value[part]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kairyu_bench.adapter_context")
    parser.add_argument("field")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    if not context_path:
        parser.error("KAIRYU_BENCH_CONTEXT is not set")
    try:
        value = context_value(read_context(Path(context_path)), args.field)
    except (KeyError, ValueError) as error:
        print(f"kairyu-bench context: {error}", file=sys.stderr)
        return 2
    if args.json or isinstance(value, (dict, list, bool)) or value is None:
        print(json.dumps(value, separators=(",", ":"), allow_nan=False))
    else:
        print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
