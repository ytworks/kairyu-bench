"""Bounded, reproducible observations of explicitly selected matrix coordinates."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint, DTYPES
from kairyu_bench.wsp.weights.statistics import _io, _number

_MAX_SAMPLE = 1024
_READ_ELEMENTS = 131072


def _indices(size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if type(count) is not int or not 1 <= count <= _MAX_SAMPLE:
        raise ValueError(f"sample counts must be between 1 and {_MAX_SAMPLE}")
    if size == 0:
        raise ValueError("matrix dimensions must be nonempty")
    return np.sort(rng.choice(size, min(size, count), replace=False))


def _generator(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:8], "little")])


def _shape(checkpoint: Checkpoint, name: str, axis: str = "rows") -> tuple[int, int]:
    if axis not in {"rows", "columns"}:
        raise ValueError("axis must be rows or columns")
    if name not in checkpoint.tensors:
        raise ValueError(f"unknown tensor: {name}")
    if not checkpoint.tensors[name].floating:
        raise ValueError(f"matrix observations require floating tensors: {name}")
    shape = checkpoint.tensors[name].shape
    if len(shape) != 2 or not all(shape):
        raise ValueError(f"expected a nonempty rank-two tensor: {name}")
    rows, columns = int(shape[0]), int(shape[1])
    return (rows, columns) if axis == "rows" else (columns, rows)


def _sample(
    checkpoint: Checkpoint,
    name: str,
    channels: np.ndarray,
    features: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, int]:
    """Read selected rows in bounded spans, preserving original axis indices."""
    rows, columns = (channels, features) if axis == "rows" else (features, channels)
    width = int(checkpoint.tensors[name].shape[1])
    values = np.empty((len(rows), len(columns)), dtype=np.float64)
    elements_read = 0
    first = int(columns[0])
    last = int(columns[-1]) + 1
    for output_row, row in enumerate(rows):
        for start in range(first, last, _READ_ELEMENTS):
            stop = min(start + _READ_ELEMENTS, last)
            left = int(np.searchsorted(columns, start))
            right = int(np.searchsorted(columns, stop))
            if left == right:
                continue
            block = checkpoint.read(name, int(row) * width + start, stop - start)
            elements_read += len(block)
            values[output_row, left:right] = block[columns[left:right] - start]
    if not np.isfinite(values).all():
        raise ValueError(f"selected coordinates contain nonfinite values: {name}")
    return (values if axis == "rows" else values.T), elements_read


def _spectrum(values: np.ndarray) -> dict[str, Any]:
    scale = float(np.max(np.abs(values), initial=0.0))
    scaled = values / scale if scale else values
    try:
        singular = np.linalg.svd(scaled, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("SVD did not converge for the selected finite submatrix") from error
    energy = singular * singular
    total = float(energy.sum())
    probabilities = energy[energy > 0] / total if total else np.array([], dtype=float)
    effective = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if total else 0.0
    tolerance = max(values.shape) * np.finfo(np.float64).eps * float(singular[0])
    with np.errstate(over="ignore"):
        restored = singular * scale
        norm = float(np.sqrt(total) * scale)
        restored_tolerance = tolerance * scale
    return {
        "singular_values": [_number(float(x)) for x in restored],
        "frobenius_norm": _number(norm),
        "stable_rank": total / float(energy[0]) if total else 0.0,
        "energy_entropy_effective_rank": effective,
        "numerical_rank": int(np.count_nonzero(singular > tolerance)),
        "numerical_rank_tolerance": _number(restored_tolerance),
        "leading_energy_fraction": float(energy[0]) / total if total else None,
        "numeric_overflow": not np.isfinite(restored).all() or not math.isfinite(norm),
        "calculation": "float64 SVD after max-absolute-value scaling; ranks use scaled singular values",
    }


def _read_count(channels: np.ndarray, features: np.ndarray, axis: str) -> int:
    rows, columns = (channels, features) if axis == "rows" else (features, channels)
    first, last = int(columns[0]), int(columns[-1]) + 1
    per_row = 0
    for start in range(first, last, _READ_ELEMENTS):
        stop = min(start + _READ_ELEMENTS, last)
        if np.searchsorted(columns, stop) > np.searchsorted(columns, start):
            per_row += stop - start
    return len(rows) * per_row


def spectra(
    checkpoint: Checkpoint,
    names: Sequence[str],
    *,
    sample_rows: int = 128,
    sample_columns: int = 128,
    seed: int = 0,
    hash_files: bool = True,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    before = checkpoint.read_counts.copy()
    plans = []
    for name in dict.fromkeys(names):
        rows, columns = _shape(checkpoint, name)
        rng = _generator(seed, name)
        selected_rows = _indices(rows, sample_rows, rng)
        selected_columns = _indices(columns, sample_columns, rng)
        plans.append((name, rows, columns, selected_rows, selected_columns))
    if not plans:
        raise ValueError("select at least one matrix")
    estimate = checkpoint.estimated_io(sum(_read_count(rs, cs, "rows") * DTYPES[checkpoint.tensors[n].dtype][1]
                                          for n, _, _, rs, cs in plans), hash_files)
    records = []
    for name, rows, columns, selected_rows, selected_columns in plans:
        sample, read_count = _sample(checkpoint, name, selected_rows, selected_columns, "rows")
        records.append({
            "tensor": name,
            "shape": [rows, columns],
            "coverage": "full_matrix" if sample.shape == (rows, columns) else "sampled_submatrix",
            "row_indices": selected_rows.tolist(),
            "column_indices": selected_columns.tolist(),
            "observed_elements": int(sample.size),
            "elements_read": read_count,
            "spectrum": _spectrum(sample),
        })
    checkpoint.assert_stable()
    identity = checkpoint.identity(hash_files=hash_files)
    return {
        "schema_version": 1,
        "operation": "spectrum",
        "io": _io(checkpoint, before, estimate),
        "sampling_algorithm": "numpy-default_rng-PCG64-sha256-name-v1",
        "numpy_version": np.__version__,
        "seed": seed,
        "interpretation": "Ranks and singular values describe only the recorded submatrix.",
        "checkpoint": identity,
        "matrices": records,
    }


def alignment(
    checkpoint: Checkpoint,
    left: str,
    right: str,
    *,
    left_axis: str = "rows",
    right_axis: str = "rows",
    channels: int = 128,
    features: int = 128,
    seed: int = 0,
    hash_files: bool = True,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    before = checkpoint.read_counts.copy()
    shape = _shape(checkpoint, left, left_axis)
    if shape != _shape(checkpoint, right, right_axis):
        raise ValueError("explicitly aligned matrices must have matching channel and feature dimensions")
    rng = _generator(seed, f"{left}/{left_axis}/{right}/{right_axis}")
    selected_channels = _indices(shape[0], channels, rng)
    selected_features = _indices(shape[1], features, rng)
    estimated_payload = sum(_read_count(selected_channels, selected_features, axis) * DTYPES[checkpoint.tensors[name].dtype][1]
                            for name, axis in ((left, left_axis), (right, right_axis)))
    estimate = checkpoint.estimated_io(estimated_payload, hash_files)
    a, reads_a = _sample(checkpoint, left, selected_channels, selected_features, left_axis)
    b, reads_b = _sample(checkpoint, right, selected_channels, selected_features, right_axis)
    scale_a = np.max(np.abs(a), axis=1)
    scale_b = np.max(np.abs(b), axis=1)
    scaled_a = a / np.where(scale_a > 0, scale_a, 1.0)[:, None]
    scaled_b = b / np.where(scale_b > 0, scale_b, 1.0)[:, None]
    scaled_norm_a = np.linalg.norm(scaled_a, axis=1)
    scaled_norm_b = np.linalg.norm(scaled_b, axis=1)
    valid = (scale_a > 0) & (scale_b > 0)
    cosine = np.zeros(len(scale_a), dtype=np.float64)
    unit_a = scaled_a[valid] / scaled_norm_a[valid, None]
    unit_b = scaled_b[valid] / scaled_norm_b[valid, None]
    cosine[valid] = np.clip(np.sum(unit_a * unit_b, axis=1), -1.0, 1.0)
    with np.errstate(over="ignore", under="ignore"):
        norm_a, norm_b = scale_a * scaled_norm_a, scale_b * scaled_norm_b
        products = norm_a * norm_b
    log_products = np.full(len(scale_a), -np.inf)
    log_products[valid] = (np.log(scale_a[valid]) + np.log(scaled_norm_a[valid])
                           + np.log(scale_b[valid]) + np.log(scaled_norm_b[valid]))
    records = [{
        "channel": int(channel),
        "left_norm": _number(float(norm_a[index])),
        "right_norm": _number(float(norm_b[index])),
        "norm_product": _number(float(products[index])),
        "log_norm_product": _number(float(log_products[index])),
        "numeric_overflow": not bool(np.isfinite(norm_a[index]) and np.isfinite(norm_b[index])
                                      and np.isfinite(products[index])),
        "product_underflow": bool(valid[index] and products[index] == 0),
        "cosine": float(cosine[index]) if valid[index] else None,
    } for index, channel in enumerate(selected_channels)]
    checkpoint.assert_stable()
    identity = checkpoint.identity(hash_files=hash_files)
    return {
        "schema_version": 1,
        "operation": "align",
        "io": _io(checkpoint, before, estimate),
        "sampling_algorithm": "numpy-default_rng-PCG64-sha256-name-v1",
        "numpy_version": np.__version__,
        "checkpoint": identity,
        "seed": seed,
        "left": {"tensor": left, "channel_axis": left_axis},
        "right": {"tensor": right, "channel_axis": right_axis},
        "aligned_shape": list(shape),
        "coverage": "full_matrices" if a.shape == shape else "sampled_channels_and_features",
        "channel_indices": selected_channels.tolist(),
        "feature_indices": selected_features.tolist(),
        "observed_elements_per_matrix": int(a.size),
        "elements_read": reads_a + reads_b,
        "valid_cosine_channels": int(valid.sum()),
        "mean_cosine": float(cosine[valid].mean()) if valid.any() else None,
        "mean_absolute_cosine": float(np.abs(cosine[valid]).mean()) if valid.any() else None,
        "channels": records,
        "highest_norm_product_channels": sorted(
            records, key=lambda record: (-(record["log_norm_product"] if record["log_norm_product"] is not None else -math.inf),
                                         record["channel"])
        )[:8],
        "interpretation": (
            "Axis correspondence is supplied by the caller. Norm products and cosines describe "
            "stored weight geometry on the recorded features. Functional dependencies require "
            "separate activation or intervention evidence."
        ),
    }
