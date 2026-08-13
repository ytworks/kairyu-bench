from __future__ import annotations

from collections.abc import Sequence


class SelectionError(ValueError):
    """A benchmark problem selection is invalid or non-reproducible."""


def select_problem_ids(canonical_ids: Sequence[str], limit: int | None) -> list[str]:
    ids = list(canonical_ids)
    if not all(isinstance(problem_id, str) and problem_id for problem_id in ids):
        raise SelectionError("problem IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise SelectionError("canonical problem IDs contain a duplicate")
    if limit is not None:
        if isinstance(limit, bool) or limit <= 0:
            raise SelectionError("limit must be a positive integer")
        return ids[:limit]
    return ids
