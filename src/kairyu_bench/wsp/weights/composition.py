"""Read-only arithmetic diagnostics for joint original-base BF16 edit sets."""
from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np


def _bf16(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    wide = np.asarray(value, dtype=np.float64)
    with np.errstate(over="ignore"):
        narrow = wide.astype(np.float32)
    if (wide.ndim != 1 or not 1 <= wide.size <= 2000000 or not np.isfinite(wide).all()
            or not np.array_equal(wide, narrow.astype(np.float64))
            or np.any(narrow.view(np.uint32) & 0xFFFF)):
        raise ValueError(f"{name} must contain 1..2000000 exact finite BF16 values")
    return narrow


def compose_bf16_sets(base: Any, variants: Sequence[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Sum 2..8 variants' deltas from one base, then round only the final sum.

    Inputs are absolute stored BF16 values in the same explicitly selected
    coordinate scope. Arithmetic is F64 delta/sum -> F32 -> BF16 ties-to-even.
    There is no sequential overwrite or adoption decision and no file I/O.
    Numerical cancellation restores the original stored value, including -0.
    """
    original = _bf16(base, "base")
    if not 2 <= len(variants) <= 8:
        raise ValueError("two through eight variants required")
    differences = []
    components = []
    for i, value in enumerate(variants):
        v = _bf16(value, f"variant {i}")
        if v.shape != original.shape:
            raise ValueError("all variants must use exactly the same coordinate scope")
        delta = v.astype(np.float64) - original.astype(np.float64)
        differences.append(delta)
        components.append({"index": i, "changed_elements": int(np.count_nonzero(delta)),
                           "delta_l2": float(np.linalg.norm(delta))})
    intended = np.sum(differences, axis=0, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        f32 = (original.astype(np.float64) + intended).astype(np.float32)
        bits = f32.view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        merged = bits.view(np.float32).copy()
    merged[intended == 0] = original[intended == 0]
    if not np.isfinite(merged).all():
        raise ValueError("joint composition overflowed")
    actual = merged.astype(np.float64) - original.astype(np.float64)
    norm = float(np.linalg.norm(actual))
    ideal_norm = float(np.linalg.norm(intended))
    base_norm = float(np.linalg.norm(original.astype(np.float64)))
    pairs = []
    for left, right in itertools.combinations(range(len(variants)), 2):
        a, b = differences[left], differences[right]
        overlap = (a != 0) & (b != 0)
        same = overlap & (np.signbit(a) == np.signbit(b))
        opposite = overlap & ~same
        total = a + b
        partial = opposite & (total != 0) & (np.abs(total) < np.maximum(np.abs(a), np.abs(b)))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        pairs.append({
            "left": left, "right": right,
            "overlap_elements": int(np.count_nonzero(overlap)),
            "same_sign_elements": int(np.count_nonzero(same)),
            "opposite_sign_elements": int(np.count_nonzero(opposite)),
            "exact_cancellation_elements": int(np.count_nonzero(opposite & (total == 0))),
            "partial_cancellation_elements": int(np.count_nonzero(partial)),
            "dot_product": float(a @ b),
            "cosine": None if denominator == 0 else float(a @ b) / denominator,
        })
    union = np.any(np.stack([d != 0 for d in differences]), axis=0)
    exact_cancel = union & (intended == 0)
    rounded_away = (intended != 0) & (actual == 0)
    return merged, {
        "components": components, "pairwise": pairs,
        "union_changed_elements": int(np.count_nonzero(union)),
        "changed_elements": int(np.count_nonzero(actual)),
        "stored_bit_changes": int(np.count_nonzero(merged.view(np.uint32) != original.view(np.uint32))),
        "exact_cancellation_elements": int(np.count_nonzero(exact_cancel)),
        "additional_rounding_noops": int(np.count_nonzero(rounded_away)),
        "delta_l2": norm, "continuous_delta_l2": ideal_norm,
        "rounding_error_l2": float(np.linalg.norm(actual - intended)),
        "rounding_relative_l2": None if ideal_norm == 0 else float(np.linalg.norm(actual - intended)) / ideal_norm,
        "base_l2_selected_scope": base_norm,
        "relative_delta_l2_selected_scope": None if base_norm == 0 else norm / base_norm,
        "scope_elements": len(original),
        "arithmetic": "original BF16 + F64 sum of original-relative deltas -> F32 -> BF16 once",
        "model_response_measured": False, "checkpoint_writes": 0,
    }
