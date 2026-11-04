"""Deterministic, stratified dev/test selection and tier assignment.

Order inside a source never depends on the dataset's own order (its prefix is
often the easiest or oldest problems): items are ranked by a hash of their id
inside each stratum, alternated between dev and test, and the strata are
interleaved by their share, so any prefix of a split is a stratified sample.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME, SPLITS, TIERS

SELECTION_SALT = "kairyu-bench-wsp-v1"


def _rank(source: str, item_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}\0{source}\0{item_id}".encode()).hexdigest()


def interleave(strata: Mapping[str, Sequence[str]], *, equal: bool = False) -> list[str]:
    """Merge ordered strata so every prefix keeps each stratum's share
    (its size, or an equal share while it has items left)."""

    names = sorted(strata)
    total = sum(len(ids) for ids in strata.values())
    taken = dict.fromkeys(names, 0)
    merged: list[str] = []
    for position in range(1, total + 1):
        # The stratum furthest behind its share; ties go to the first name.
        if equal:
            name = min((key for key in names if taken[key] < len(strata[key])), key=lambda key: taken[key])
            merged.append(strata[name][taken[name]])
            taken[name] += 1
            continue
        name = max(
            (key for key in names if taken[key] < len(strata[key])),
            key=lambda key: len(strata[key]) * position / total - taken[key],
        )
        merged.append(strata[name][taken[name]])
        taken[name] += 1
    return merged


def assign_splits(candidates: Iterable[tuple[str, str, str]]) -> dict[str, str]:
    """dev or test for every ``(item_id, stratum, group)`` candidate.

    Items of one group (one problem in several sources, languages or
    presentations) share a split, so test never repeats a dev problem. Groups
    alternate between dev and test in hash order inside each (family,
    stratum) cell; the family is the group's prefix before ``:`` and the
    stratum is that of the group's first item.
    """

    group_stratum: dict[str, str] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for item_id, stratum, group in candidates:
        group_stratum.setdefault(group, stratum)
        members[group].append(item_id)
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, stratum in group_stratum.items():
        cells[(group.split(":", 1)[0], stratum)].append(group)
    split_of: dict[str, str] = {}
    for (family, _), groups in sorted(cells.items()):
        for position, group in enumerate(sorted(groups, key=lambda key: _rank(family, key))):
            for item_id in members[group]:
                split_of[item_id] = SPLITS[position % 2]
    return split_of


def split_order(
    source: str, candidates: Iterable[tuple[str, str]], split_of: Mapping[str, str], *, equal: bool = False
) -> dict[str, list[str]]:
    """dev and test orders for ``(item_id, stratum)`` candidates of one source."""

    strata: dict[str, dict[str, list[str]]] = {split: defaultdict(list) for split in SPLITS}
    for item_id, stratum in candidates:
        strata[split_of[item_id]][stratum].append(item_id)
    return {
        split: interleave(
            {key: sorted(ids, key=lambda item_id: _rank(source, item_id)) for key, ids in strata[split].items()},
            equal=equal,
        )
        for split in SPLITS
    }


@dataclass(frozen=True)
class Measured:
    """The base model's calibration outcome for one chat item."""

    tokens: int
    finished: bool


def assign_tiers(
    source: str,
    orders: Mapping[str, Sequence[str]],
    targets: Mapping[int, int],
    *,
    tier_tokens: Mapping[int, int],
    margin: float,
    measured: Mapping[str, Measured] | None,
) -> dict[str, list[tuple[str, int]]]:
    """Pick ``targets[T]`` items (cumulative) per split for each tier T.

    A chat item may enter tier T only if the base model finished it within
    ``margin * tier_tokens[T]`` tokens; dev and test keep equal counts.
    """

    first = SOURCE_BY_NAME[source].first_tier
    if any(targets.get(tier, 0) for tier in TIERS if tier < first):
        raise ValueError(f"{source} starts at tier {first}")
    cumulative = [targets.get(tier, 0) for tier in TIERS]
    if cumulative != sorted(cumulative):
        raise ValueError(f"{source}: tier targets must be cumulative")

    def eligible_tier(item_id: str) -> int | None:
        if measured is None:
            return first
        outcome = measured.get(item_id)
        if outcome is None or not outcome.finished:
            return None
        for tier in TIERS:
            if tier >= first and outcome.tokens <= margin * tier_tokens[tier]:
                return tier
        return None

    # Smallest tier first: tier T takes, from what is left, items the base
    # model finished within T's limit until T's cumulative target is met, so
    # no cumulative count exceeds its target and short items fill the short
    # tiers. dev and test take the same number at every tier.
    floors = {split: {item_id: eligible_tier(item_id) for item_id in orders[split]} for split in SPLITS}
    remaining = {split: [item_id for item_id in orders[split] if floors[split][item_id] is not None] for split in SPLITS}
    chosen: dict[str, list[tuple[str, int]]] = {split: [] for split in SPLITS}
    for tier, quota in zip(TIERS, cumulative, strict=True):
        eligible = {
            split: [item_id for item_id in remaining[split] if (floors[split][item_id] or 0) <= tier]
            for split in SPLITS
        }
        take = min(quota - len(chosen[SPLITS[0]]), *(len(ids) for ids in eligible.values()))
        for split in SPLITS:
            picked = eligible[split][: max(0, take)]
            chosen[split].extend((item_id, tier) for item_id in picked)
            taken = set(picked)
            remaining[split] = [item_id for item_id in remaining[split] if item_id not in taken]
    return chosen
