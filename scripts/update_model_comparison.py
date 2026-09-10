#!/usr/bin/env python3
"""Regenerate the two tables from the versioned public catalog, without network."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kairyu_bench.public_results import (
    comparison_table,
    deepswe_table,
    load_public_results,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = load_public_results()
    source = catalog["sources"]["S23"]
    snapshot = (root / source["snapshot"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != source["sha256"]:
        raise ValueError("DeepSWE source snapshot hash differs from catalog")
    published = {r["config"]: r for r in json.loads(snapshot)["rows"]}
    for row in catalog["results"]:
        if row["benchmark"] == "deepswe":
            original = published.pop(row["config"])
            for key, field, multiplier in [
                ("score_percent", "pass_at_1", 100),
                ("pass_at_4_percent", "pass_at_4", 100),
                ("passed_attempts", "n_passed", 1),
                ("scored_attempts", "n_attempted", 1),
                ("ci_half_percent", "ci_half", 100),
                ("attempts_per_task", "n_runs", 1),
            ]:
                if row[key] != original[field] * multiplier:
                    raise ValueError(
                        f"catalog {key} differs from official {row['config']}"
                    )
            if row["model"] != original["model"] or row[
                "reasoning_effort"
            ] != original.get("reasoning_effort"):
                raise ValueError(
                    "catalog model or reasoning effort differs from official source"
                )
    if published:
        raise ValueError("catalog is missing official configurations")
    path = root / "docs/model-comparison.md"
    before = path.read_text(encoding="utf-8")
    after = before
    for name, table in [
        ("public-comparison", comparison_table(catalog)),
        ("deepswe-comparison", deepswe_table(catalog)),
    ]:
        start, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
        left, tail = after.split(start, 1)
        _, right = tail.split(end, 1)
        after = left + start + "\n" + table + "\n" + end + right
    if args.check:
        if after != before:
            print(
                "model comparison tables are stale; run python scripts/update_model_comparison.py"
            )
            return 1
    else:
        path.write_text(after, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
