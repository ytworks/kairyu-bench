from __future__ import annotations

import json
from importlib import resources
from typing import Any

from kairyu_bench.benchmarks import BENCHMARK_NAMES


class ManifestError(ValueError):
    """The source lock or requested adapter selection is invalid."""


def load_manifest() -> dict[str, dict[str, Any]]:
    path = resources.files("kairyu_bench").joinpath("data/benchmarks.json")
    with path.open("r", encoding="utf-8") as stream:
        decoded = json.load(stream)
    if not isinstance(decoded, list):
        raise ManifestError("benchmark manifest must be a list")
    manifest: dict[str, dict[str, Any]] = {}
    for entry in decoded:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ManifestError("each manifest entry must have a string name")
        name = entry["name"]
        if name in manifest:
            raise ManifestError(f"duplicate manifest entry: {name}")
        manifest[name] = entry
    if tuple(manifest) != BENCHMARK_NAMES:
        raise ManifestError("manifest names or order differ from the public adapters")
    return manifest


def select_benchmarks(only: str | None) -> list[str]:
    if only is None:
        return list(BENCHMARK_NAMES)
    requested = [name.strip() for name in only.split(",") if name.strip()]
    if not requested:
        raise ManifestError("--only must contain at least one benchmark")
    if len(requested) != len(set(requested)):
        raise ManifestError("--only contains a duplicate benchmark")
    unknown = sorted(set(requested) - set(BENCHMARK_NAMES))
    if unknown:
        raise ManifestError("unknown benchmark: " + ", ".join(unknown))
    requested_set = set(requested)
    return [name for name in BENCHMARK_NAMES if name in requested_set]
