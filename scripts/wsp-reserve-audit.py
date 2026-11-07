#!/usr/bin/env python3
"""Audit a nested WSP reserve using manifest metadata only; never execute a benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME

ITEM_FIELDS = {"id", "source", "split", "tier", "max_tokens", "samples", "stratum", "expected_tokens"}


def group(item: dict[str, Any]) -> str:
    source = item["source"]
    tail = item["id"].split(":", 1)[1]
    if source in {"mmlu-pro", "mmlu-prox-ja"}:
        return "mmlu:" + tail
    if source == "mgsm":
        return "mgsm-index:" + tail.split(":", 1)[1]
    if source == "mmmu-pro":
        return "mmmu:" + tail.split(":", 1)[1]
    return item["id"]


def audit(
    data: dict[str, Any], digest: str, *, reserve_tier: int = 30,
    seen_test_tier: int = 10, seen_dev_tier: int = 60,
) -> dict[str, Any]:
    if data.get("schema_version") != 1 or not isinstance(data.get("items"), list):
        raise ValueError("expected a WSP schema_version 1 manifest")
    items = data["items"]
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) - ITEM_FIELDS:
            raise ValueError("manifest items must contain identity/selection metadata only")
        if item.get("source") not in SOURCE_BY_NAME or item.get("split") not in {"dev", "test"}:
            raise ValueError("unknown source or split")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity.startswith(item["source"] + ":"):
            raise ValueError("invalid item identity")
        if identity in seen_ids:
            raise ValueError("duplicate manifest item identity")
        seen_ids.add(identity)
        if type(item.get("tier")) is not int or item["tier"] not in {10, 30, 60}:
            raise ValueError("invalid item tier")
        group(item)  # Validate the explicit ID-derived group representation.
    if not 0 <= seen_test_tier < reserve_tier or reserve_tier not in {10, 30, 60}:
        raise ValueError("reserve tier must exceed the seen test tier")
    dev = [x for x in items if x["split"] == "dev" and x["tier"] <= seen_dev_tier]
    seen_test = [x for x in items if x["split"] == "test" and x["tier"] <= seen_test_tier]
    reserve_all = [x for x in items if x["split"] == "test" and x["tier"] <= reserve_tier]
    seen = {x["id"] for x in seen_test + dev}
    seen_groups = {group(x) for x in seen_test + dev}
    difference = [x for x in reserve_all if x["id"] not in {s["id"] for s in seen_test}]
    collisions = [x for x in difference if group(x) in seen_groups]
    remaining = [x for x in difference if group(x) not in seen_groups]
    # Canonical GSM8K↔MGSM identity is a normalized-English-question hash, absent
    # from the stored manifest. Quarantine rather than read task text or guess.
    quarantine = [x for x in remaining if x["source"] in {"mgsm", "gsm8k-platinum"}]
    reserve = [x for x in remaining if x["source"] not in {"mgsm", "gsm8k-platinum"}]
    counts = Counter(group(x) for x in reserve)
    def by_source(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(x["source"] for x in rows).items()))
    return {
        "schema_version": 1, "status": "proposed_metadata_selection_not_an_activated_gate",
        "scope": "manifest identities only; no task text, answers, model responses or scores",
        "manifest_sha256": digest, "reserve_tier": reserve_tier, "seen_test_tier": seen_test_tier,
        "seen_dev_tier": seen_dev_tier, "seen_test_items": len(seen_test),
        "reserve_tier_items": len(reserve_all), "raw_difference_items": len(difference),
        "known_group_overlap_exclusions": len(collisions), "known_group_overlap_by_source": by_source(collisions),
        "known_group_overlap_ids": [x["id"] for x in collisions],
        "unresolved_cross_source_quarantine": len(quarantine), "quarantine_by_source": by_source(quarantine),
        "exact_id_overlap_with_seen": sum(x["id"] in seen for x in difference),
        "proposed_reserve_items": len(reserve), "distinct_underlying_groups": len(counts),
        "within_reserve_extra_presentations": sum(n - 1 for n in counts.values()),
        "proposed_reserve_by_source": by_source(reserve),
        "proposed_reserve_by_area": dict(sorted(Counter(
            SOURCE_BY_NAME[x["source"]].area or "distribution_probe" for x in reserve
        ).items())),
        "selection_sha256": hashlib.sha256("\n".join(sorted(x["id"] for x in reserve)).encode()).hexdigest(),
        "items": [{**x, "group": group(x)} for x in reserve],
        "limits": [
            "MMLU-Pro/ProX question IDs, MGSM language variants and MMMU presentations are grouped explicitly.",
            "Other semantic duplicates and public training-data overlap are not certified absent.",
            "A known-group-disjoint subset is not proof that questions or scores were never viewed.",
            "Audit historical baseline, calibration and access records separately before claiming freshness.",
            "Repeated presentations inside the reserve require group-clustered uncertainty, not independent samples.",
            "This subset is not an official full WSP tier; freeze its selection and evaluation contract before use.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-tier", type=int, choices=[10, 30, 60], default=30)
    parser.add_argument("--seen-test-tier", type=int, choices=[0, 10, 30], default=10)
    parser.add_argument("--seen-dev-tier", type=int, choices=[0, 10, 30, 60], default=60)
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        result = audit(
            json.loads(raw), hashlib.sha256(raw).hexdigest(), reserve_tier=args.reserve_tier,
            seen_test_tier=args.seen_test_tier, seen_dev_tier=args.seen_dev_tier,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in (
        "status", "raw_difference_items", "known_group_overlap_exclusions",
        "unresolved_cross_source_quarantine", "proposed_reserve_items", "selection_sha256",
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
