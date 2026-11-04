"""The frozen item list: which items each tier and split runs, and their limits.

An item belongs to one split and enters at one tier; tier T runs every item
whose tier is at most T, so 10 < 30 < 60 are nested and one item has the same
request in every tier. dev and test have the same composition per source.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.catalog import AREA_KEYS, SOURCE_BY_NAME, SPLITS, TIERS


class WspManifestError(ValueError):
    """The item manifest violates the tier, split or source contract."""


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    split: str
    tier: int
    max_tokens: int
    samples: int
    stratum: str
    expected_tokens: int

    @property
    def kind(self) -> str:
        return SOURCE_BY_NAME[self.source].kind


@dataclass(frozen=True)
class Manifest:
    data: dict[str, Any]
    items: tuple[Item, ...]
    sha256: str

    @property
    def tier_minutes(self) -> dict[int, int]:
        return {int(key): int(value["minutes"]) for key, value in self.data["tiers"].items()}

    def select(self, tier: int, split: str) -> list[Item]:
        if tier not in TIERS or split not in SPLITS:
            raise WspManifestError(f"unknown tier {tier} or split {split}")
        return [item for item in self.items if item.split == split and item.tier <= tier]


def _item(entry: object) -> Item:
    if not isinstance(entry, dict):
        raise WspManifestError("each item must be an object")
    try:
        item = Item(
            id=str(entry["id"]),
            source=str(entry["source"]),
            split=str(entry["split"]),
            tier=int(entry["tier"]),
            max_tokens=int(entry["max_tokens"]),
            samples=int(entry.get("samples", 1)),
            stratum=str(entry.get("stratum", "")),
            expected_tokens=int(entry.get("expected_tokens", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WspManifestError(f"invalid item {entry!r}: {error}") from error
    source = SOURCE_BY_NAME.get(item.source)
    if source is None:
        raise WspManifestError(f"{item.id}: unknown source {item.source}")
    if not item.id.startswith(f"{item.source}:"):
        raise WspManifestError(f"{item.id}: id must start with its source name")
    if item.split not in SPLITS or item.tier not in TIERS:
        raise WspManifestError(f"{item.id}: unknown split or tier")
    if item.tier < source.first_tier:
        raise WspManifestError(f"{item.id}: {item.source} starts at tier {source.first_tier}")
    if item.max_tokens <= 0 or item.samples <= 0 or item.expected_tokens < 0:
        raise WspManifestError(f"{item.id}: limits must be positive")
    return item


def missing_coverage(items: tuple[Item, ...] | list[Item]) -> dict[str, list[str]]:
    """``"<tier>/<split>" -> areas without an item`` for every tier and split."""

    gaps: dict[str, list[str]] = {}
    for tier in TIERS:
        for split in SPLITS:
            covered = {
                SOURCE_BY_NAME[item.source].area for item in items if item.split == split and item.tier <= tier
            }
            absent = [area for area in AREA_KEYS if area not in covered]
            if absent:
                gaps[f"{tier}/{split}"] = absent
    return gaps


def parse_manifest(data: dict[str, Any], *, sha256: str, require_all_areas: bool = True) -> Manifest:
    if data.get("schema_version") != 1:
        raise WspManifestError("schema_version must be 1")
    tiers = data.get("tiers")
    if not isinstance(tiers, dict) or sorted(int(key) for key in tiers) != list(TIERS):
        raise WspManifestError(f"tiers must be exactly {TIERS}")
    entries = data.get("items")
    if not isinstance(entries, list) or not entries:
        raise WspManifestError("items must be a non-empty list")
    items = tuple(_item(entry) for entry in entries)
    duplicates = [key for key, count in Counter(item.id for item in items).items() if count > 1]
    if duplicates:
        raise WspManifestError(f"duplicate item ids: {duplicates[:5]}")
    for item in items:
        limit = int(tiers[str(item.tier)]["max_output_tokens"])
        if item.kind == "chat" and item.max_tokens > limit:
            raise WspManifestError(f"{item.id}: max_tokens exceeds tier {item.tier}'s limit")
    composition = Counter((item.source, item.tier, item.split) for item in items)
    for source, tier, _ in composition:
        if composition[(source, tier, "dev")] != composition[(source, tier, "test")]:
            raise WspManifestError(f"{source} tier {tier}: dev and test differ in size")
    # Every tier measures every area; a list missing one is not a wsp tier.
    gaps = missing_coverage(items) if require_all_areas else {}
    if gaps:
        detail = "; ".join(f"{key}: {', '.join(areas)}" for key, areas in gaps.items())
        raise WspManifestError(f"areas without items: {detail}")
    return Manifest(data=data, items=items, sha256=sha256)


def load_manifest(path: Path | None = None, *, require_all_areas: bool = True) -> Manifest:
    if path is None:
        resource = resources.files("kairyu_bench").joinpath("data/wsp/manifest.json")
        raw = resource.read_bytes()
    else:
        raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise WspManifestError("manifest must be an object")
    return parse_manifest(data, sha256=hashlib.sha256(raw).hexdigest(), require_all_areas=require_all_areas)
