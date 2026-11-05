"""Streaming checkpoint statistics with explicit selected and sampled scopes."""
from __future__ import annotations

import hashlib
import heapq
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import Checkpoint, CheckpointError, DEFAULT_CHUNK_BYTES, DTYPES, TensorInfo

MAX_AXIS_SIZE = 100_000
MAX_SAMPLES = 1_000_000
QUANTILES = (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)


def _number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _norm(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    scale = float(np.max(np.abs(values)))
    if not scale:
        return 0.0
    with np.errstate(over="ignore"):
        return float(scale * np.sqrt(np.sum(np.square(values / scale), dtype=np.float64)))


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not values.size:
        return {}
    ordered = np.sort(values)
    result = {}
    for q in QUANTILES:
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        weight = position - lo
        # Convex interpolation avoids overflow in (hi - lo) for finite F64.
        value = float(ordered[lo]) * (1.0 - weight) + float(ordered[hi]) * weight
        result[str(q)] = _number(value)
    return result


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return _number(numerator / denominator) if denominator and numerator is not None else None


def _budget(chunk_bytes: int, sample_size: int, tensors: int, axis: bool) -> dict[str, int]:
    if type(chunk_bytes) is not int or chunk_bytes < 8:
        raise CheckpointError("chunk_bytes must be an integer of at least 8")
    estimate = 64 * 1024**2 + 48 * chunk_bytes + 32 * sample_size + 16384 * tensors
    if axis:
        estimate += 32 * MAX_AXIS_SIZE
    limit = 512 * 1024**2
    if estimate > limit:
        raise CheckpointError("estimated working memory exceeds 512 MiB; reduce chunk/sample size or select fewer tensors")
    return {"estimated_working_bytes": estimate, "working_memory_target_bytes": limit,
            "raw_chunk_bytes": chunk_bytes}


def _io(checkpoint: Checkpoint, before: dict[str, int], estimated: dict[str, int]) -> dict[str, Any]:
    actual = {key: checkpoint.read_counts[key] - (0 if key == "metadata" else before[key])
              for key in checkpoint.read_counts}
    return {"estimated_bytes": estimated, "actual_bytes": {**actual, "total": sum(actual.values())}}


def _sampling(indices: np.ndarray, seed: int) -> dict[str, Any]:
    return {"algorithm": "python-random.sample-range-sha256-seed-name-v1",
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
            "seed": seed}


def _coordinates(index: int, shape: tuple[int, ...]) -> list[int]:
    return list(map(int, np.unravel_index(index, shape))) if shape else []


def _top(heap: list[tuple[float, int, float]], values: np.ndarray, start: int, limit: int) -> None:
    if not limit:
        return
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size > limit:
        magnitude = np.abs(values[finite])
        cutoff = np.partition(magnitude, magnitude.size - limit)[magnitude.size - limit]
        # Tie handling is deterministic and favors earlier flat coordinates.
        higher = finite[magnitude > cutoff]
        equal = finite[magnitude == cutoff][:limit - higher.size]
        finite = np.concatenate((higher, equal))
    for offset in finite:
        value = float(values[offset])
        item = (abs(value), -(start + int(offset)), value)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def _top_json(heap: list[tuple[float, int, float]], shape: tuple[int, ...]) -> list[dict[str, Any]]:
    return [{"flat_index": -index, "coordinate": _coordinates(-index, shape),
             "value": value, "abs": magnitude}
            for magnitude, index, value in sorted(heap, reverse=True)]


def _sample_indices(count: int, size: int, seed: int, name: str) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    return np.asarray(sorted(rng.sample(range(count), min(count, size))), dtype=np.int64)


def _axis_norm(values: np.ndarray, axis: int) -> np.ndarray:
    scale = np.max(np.abs(values), axis=axis, initial=0.0)
    denominator = np.where(scale == 0, 1.0, scale)
    expanded = denominator[:, None] if axis == 1 else denominator[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * np.sqrt(np.sum(np.square(values / expanded), axis=axis, dtype=np.float64))


class _Axes:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.norms = [np.zeros(n, dtype=np.float64) for n in shape]
        self.nonfinite = [np.zeros(n, dtype=np.int64) for n in shape]

    def add(self, start: int, values: np.ndarray) -> None:
        columns = self.shape[1]
        if not columns:
            return
        position = 0
        while position < values.size:
            row, column = divmod(start + position, columns)
            if column or values.size - position < columns:
                size = min(columns - column, values.size - position)
                part = values[position:position + size]
                finite = np.isfinite(part)
                clean = np.where(finite, part, 0.0)
                self.norms[0][row] = math.hypot(self.norms[0][row], _norm(clean))
                self.norms[1][column:column + size] = np.hypot(self.norms[1][column:column + size], clean)
                self.nonfinite[0][row] += int(np.count_nonzero(~finite))
                self.nonfinite[1][column:column + size] += ~finite
            else:
                rows = (values.size - position) // columns
                size = rows * columns
                block = values[position:position + size].reshape(rows, columns)
                finite = np.isfinite(block)
                clean = np.where(finite, block, 0.0)
                self.norms[0][row:row + rows] = np.hypot(self.norms[0][row:row + rows], _axis_norm(clean, 1))
                self.norms[1] = np.hypot(self.norms[1], _axis_norm(clean, 0))
                self.nonfinite[0][row:row + rows] += np.count_nonzero(~finite, axis=1)
                self.nonfinite[1] += np.count_nonzero(~finite, axis=0)
            position += size

    def result(self, top_k: int) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "completed", "scope": "exact whole-tensor row/column L2"}
        for label, norms, bad in zip(("rows", "columns"), self.norms, self.nonfinite, strict=True):
            heap: list[tuple[float, int, float]] = []
            valid = np.where(bad == 0, norms, np.nan)
            _top(heap, valid, 0, top_k)
            result[label] = {"count": len(norms), "profile": "bounded_summary",
                             "zero_axes": int(np.count_nonzero((bad == 0) & (norms == 0))),
                             "nonfinite_axes": int(np.count_nonzero(bad)),
                             "nonfinite_elements": int(np.sum(bad)),
                             "overflow_axes": int(np.count_nonzero(~np.isfinite(norms))),
                             "l2_quantiles": _quantiles(valid[np.isfinite(valid)]),
                             "top": _top_json(heap, (len(norms),))}
        return result


class _Moments:
    def __init__(self) -> None:
        self.count = self.finite_count = self.zeros = self.nan = self.posinf = self.neginf = 0
        self.l2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: np.ndarray) -> None:
        self.count += values.size
        finite = np.isfinite(values)
        selected = values[finite]
        self.finite_count += selected.size
        self.zeros += int(np.count_nonzero(selected == 0))
        self.nan += int(np.count_nonzero(np.isnan(values)))
        self.posinf += int(np.count_nonzero(np.isposinf(values)))
        self.neginf += int(np.count_nonzero(np.isneginf(values)))
        self.l2 = math.hypot(self.l2, _norm(selected))
        if selected.size:
            self.minimum = min(self.minimum, float(np.min(selected)))
            self.maximum = max(self.maximum, float(np.max(selected)))

    def result(self) -> dict[str, Any]:
        return {"elements": int(self.count), "finite_elements": int(self.finite_count),
                "zero_elements": self.zeros, "nan": self.nan, "positive_infinity": self.posinf,
                "negative_infinity": self.neginf, "nonfinite_elements": int(self.count - self.finite_count),
                "l2": _number(self.l2) if self.count == self.finite_count else None,
                "finite_l2": _number(self.l2), "minimum_finite": _number(self.minimum),
                "maximum_finite": _number(self.maximum), "numeric_overflow": not math.isfinite(self.l2)}


def _options(pattern: str | None, group_pattern: str | None, sample_size: int, top_k: int) -> tuple[Any, Any]:
    if type(sample_size) is not int or not 0 <= sample_size <= MAX_SAMPLES:
        raise CheckpointError("sample_size must be between 0 and 1000000")
    if type(top_k) is not int or not 0 <= top_k <= 1000:
        raise CheckpointError("top_k must be between 0 and 1000")
    try:
        return re.compile(pattern) if pattern is not None else None, re.compile(group_pattern) if group_pattern else None
    except re.error as error:
        raise CheckpointError(f"invalid tensor/group regex: {error}") from error


def _group(name: str, pattern: Any) -> str:
    if pattern is None:
        return "all"
    match = pattern.search(name)
    if match is None:
        return "__ungrouped__"
    return (match.group(1) if pattern.groups else match.group(0)) or "__ungrouped__"


def _axes(tensor: TensorInfo, enabled: bool) -> tuple[_Axes | None, dict[str, Any] | None]:
    if not enabled:
        return None, None
    if len(tensor.shape) != 2:
        return None, {"status": "not_applicable", "reason": "rank is not 2"}
    if max(tensor.shape) > MAX_AXIS_SIZE:
        return None, {"status": "skipped", "reason": "axis exceeds 100000 entries"}
    return _Axes(tensor.shape), None


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    entries = [row[key] for row in rows]
    if not entries:
        return {"elements": 0, "finite_elements": 0, "nonfinite_elements": 0,
                "l2": None, "finite_l2": None, "zero_elements": 0,
                "numeric_overflow": False, "scope_empty": True}
    count = sum(x["elements"] for x in entries)
    finite = sum(x["finite_elements"] for x in entries)
    norm = math.hypot(*(x["finite_l2"] if x["finite_l2"] is not None else math.inf for x in entries))
    return {"elements": count, "finite_elements": finite, "nonfinite_elements": count - finite,
            "l2": _number(norm) if count == finite else None, "finite_l2": _number(norm),
            "zero_elements": sum(x["zero_elements"] for x in entries),
            "numeric_overflow": not math.isfinite(norm)}


def _as_checkpoint(path: Checkpoint | str | Path) -> Checkpoint:
    return path if isinstance(path, Checkpoint) else Checkpoint(path)


def inspect_checkpoint(checkpoint: Checkpoint | str | Path, *, pattern: str | None = None,
                       group_pattern: str | None = None, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                       sample_size: int = 4096, top_k: int = 8, axis: bool = False,
                       seed: int = 0, hash_files: bool = True) -> dict[str, Any]:
    start_time = time.monotonic()
    checkpoint = _as_checkpoint(checkpoint)
    initial_counts = checkpoint.read_counts.copy()
    selection, grouping = _options(pattern, group_pattern, sample_size, top_k)
    selected_names = [n for n in checkpoint.tensors if selection is None or selection.search(n)]
    if not selected_names:
        raise CheckpointError("no tensors match the selection")
    budget = _budget(chunk_bytes, sample_size, len(selected_names), axis)
    estimate = checkpoint.estimated_io(sum(checkpoint.tensors[n].nbytes for n in selected_names
                                          if checkpoint.tensors[n].floating), hash_files)
    rows, skipped = [], []
    for name, tensor in checkpoint.tensors.items():
        if selection is not None and not selection.search(name):
            continue
        if not tensor.floating:
            skipped.append({**tensor.info(), "reason": "nonfloating tensor; excluded from numeric norms"})
            continue
        indices = _sample_indices(tensor.numel, sample_size, seed, name)
        sample = np.empty(indices.size, dtype=np.float64)
        moments = _Moments()
        heap: list[tuple[float, int, float]] = []
        axes, axis_result = _axes(tensor, axis)
        for offset, _, source in checkpoint.iter_chunks(name, chunk_bytes):
            values = source.astype(np.float64)
            moments.add(values)
            _top(heap, values, offset, top_k)
            lo, hi = np.searchsorted(indices, (offset, offset + values.size))
            sample[lo:hi] = values[indices[lo:hi] - offset]
            if axes is not None:
                axes.add(offset, values)
        row = {**tensor.info(), "group": _group(name, grouping), "statistics": moments.result(),
               "top_coordinates": _top_json(heap, tensor.shape),
               "quantiles": {"scope": "all_elements" if indices.size == tensor.numel else "deterministic_sample",
                             "sample_size": int(indices.size), "finite_sample_size": int(np.count_nonzero(np.isfinite(sample))),
                             "sampling": _sampling(indices, seed), "values": _quantiles(sample[np.isfinite(sample)])}}
        if axis:
            row["axis_profiles"] = axes.result(top_k) if axes is not None else axis_result
        rows.append(row)
    identity = checkpoint.identity(hash_files=hash_files, chunk_bytes=chunk_bytes)
    groups = {name: _aggregate([r for r in rows if r["group"] == name], "statistics")
              for name in sorted({r["group"] for r in rows})}
    return {"schema_version": 1, "kind": "checkpoint_inspection", "checkpoint": identity,
            "status": "completed" if rows else "no_floating_tensors",
            "budget": budget, "io": _io(checkpoint, initial_counts, estimate),
            "selection": {"pattern": pattern, "group_pattern": group_pattern,
                          "scope": "selected stored floating tensors only; stored aliases counted once per tensor entry",
                          "selected_tensors": len(rows), "skipped_nonfloating_tensors": len(skipped)},
            "global": _aggregate(rows, "statistics"), "groups": groups, "tensors": rows,
            "skipped_tensors": skipped, "bytes_read": _io(checkpoint, initial_counts, estimate)["actual_bytes"]["total"],
            "elapsed_seconds": time.monotonic() - start_time}


def diff_checkpoints(base: Checkpoint | str | Path, candidate: Checkpoint | str | Path, *,
                     pattern: str | None = None, group_pattern: str | None = None,
                     chunk_bytes: int = DEFAULT_CHUNK_BYTES, sample_size: int = 4096,
                     top_k: int = 8, axis: bool = False, seed: int = 0,
                     hash_files: bool = True, allow_partial: bool = False) -> dict[str, Any]:
    start_time = time.monotonic()
    base, candidate = _as_checkpoint(base), _as_checkpoint(candidate)
    initial_counts = [cp.read_counts.copy() for cp in (base, candidate)]
    selection, grouping = _options(pattern, group_pattern, sample_size, top_k)
    names_a = {n for n in base.tensors if selection is None or selection.search(n)}
    names_b = {n for n in candidate.tensors if selection is None or selection.search(n)}
    if not names_a and not names_b:
        raise CheckpointError("no tensors match the selection")
    budget = _budget(chunk_bytes, sample_size, len(names_a | names_b), axis)
    rows, incompatible, nonfloating = [], [], []
    compatible = []
    for name in sorted(names_a & names_b):
        a, b = base.tensors[name], candidate.tensors[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            incompatible.append({"name": name, "base_dtype": a.dtype, "candidate_dtype": b.dtype,
                                 "base_shape": list(a.shape), "candidate_shape": list(b.shape),
                                 "reason": "shape or dtype differs"})
        else:
            compatible.append(name)
    coverage_complete = names_a == names_b and not incompatible
    if not coverage_complete and not allow_partial:
        raise CheckpointError("tensor coverage differs; use allow_partial/--allow-partial explicitly: "
                              f"missing={sorted(names_a - names_b)}, extra={sorted(names_b - names_a)}, "
                              f"incompatible={[x['name'] for x in incompatible]}")
    estimates = [cp.estimated_io(sum(cp.tensors[n].nbytes for n in compatible), hash_files)
                 for cp in (base, candidate)]
    for name in compatible:
        a, b = base.tensors[name], candidate.tensors[name]
        if not a.floating:
            changed = numeric_changed = 0
            for offset, raw_a, source_a in base.iter_chunks(name, chunk_bytes):
                raw_b = candidate.read_raw(name, offset, source_a.size)
                itemsize = DTYPES[a.dtype][1]
                bits_a = np.frombuffer(raw_a, dtype=np.uint8).reshape(-1, itemsize)
                bits_b = np.frombuffer(raw_b, dtype=np.uint8).reshape(-1, itemsize)
                changed += int(np.count_nonzero(np.any(bits_a != bits_b, axis=1)))
                numeric_changed += int(np.count_nonzero(source_a != candidate.decode(raw_b, b.dtype)))
            nonfloating.append({**a.info(), "stored_bit_changed_elements": changed,
                                "numeric_changed_elements": numeric_changed,
                                "scope": "integer/boolean exact differences; excluded from floating norms"})
            continue
        indices = _sample_indices(a.numel, sample_size, seed, name)
        sample = np.empty(indices.size, dtype=np.float64)
        before, after, delta = _Moments(), _Moments(), _Moments()
        heap: list[tuple[float, int, float]] = []
        axes, axis_result = _axes(a, axis)
        bit_changes = numeric_changes = nonfinite_pairs = delta_overflow = 0
        cosine = 0.0
        total_a = total_b = 0.0
        for offset, raw_a, source_a in base.iter_chunks(name, chunk_bytes):
            raw_b = candidate.read_raw(name, offset, source_a.size)
            values_a = source_a.astype(np.float64)
            values_b = candidate.decode(raw_b, b.dtype).astype(np.float64)
            before.add(values_a)
            after.add(values_b)
            itemsize = DTYPES[a.dtype][1]
            bits_a = np.frombuffer(raw_a, dtype=np.uint8).reshape(-1, itemsize)
            bits_b = np.frombuffer(raw_b, dtype=np.uint8).reshape(-1, itemsize)
            bit_changes += int(np.count_nonzero(np.any(bits_a != bits_b, axis=1)))
            pairs = np.isfinite(values_a) & np.isfinite(values_b)
            nonfinite_pairs += int(np.count_nonzero(~pairs))
            numeric_changes += int(np.count_nonzero(pairs & (values_a != values_b)))
            with np.errstate(invalid="ignore", over="ignore"):
                difference = np.where(pairs, values_b - values_a, np.nan)
            delta_overflow += int(np.count_nonzero(pairs & ~np.isfinite(difference)))
            delta.add(difference)
            _top(heap, difference, offset, top_k)
            lo, hi = np.searchsorted(indices, (offset, offset + difference.size))
            sample[lo:hi] = difference[indices[lo:hi] - offset]
            if axes is not None:
                axes.add(offset, difference)
            pair_a, pair_b = values_a[pairs], values_b[pairs]
            na, nb = _norm(pair_a), _norm(pair_b)
            next_a, next_b = math.hypot(total_a, na), math.hypot(total_b, nb)
            if next_a and next_b and all(map(math.isfinite, (na, nb, next_a, next_b))):
                block_cos = float(np.dot(pair_a / na, pair_b / nb)) if na and nb else 0.0
                cosine = cosine * (total_a / next_a) * (total_b / next_b) + block_cos * (na / next_a) * (nb / next_b)
            total_a, total_b = next_a, next_b
        before_result, after_result, delta_result = before.result(), after.result(), delta.result()
        base_norm, delta_norm = before_result["l2"], delta_result["l2"]
        row = {**a.info(), "group": _group(name, grouping),
               "base": before_result, "candidate": after_result, "delta": delta_result,
               "stored_bit_changed_elements": bit_changes, "numeric_changed_finite_elements": numeric_changes,
               "nonfinite_pairs": nonfinite_pairs, "delta_overflow_elements": delta_overflow,
               "relative_delta_l2": _ratio(delta_norm, base_norm),
               "cosine": max(-1.0, min(1.0, cosine)) if not nonfinite_pairs and total_a and total_b
                         and math.isfinite(total_a) and math.isfinite(total_b) else None,
               "top_delta_coordinates": _top_json(heap, a.shape),
               "delta_quantiles": {"scope": "all_elements" if indices.size == a.numel else "deterministic_sample",
                                   "sample_size": int(indices.size), "finite_sample_size": int(np.count_nonzero(np.isfinite(sample))),
                                   "sampling": _sampling(indices, seed), "values": _quantiles(sample[np.isfinite(sample)])}}
        if axis:
            row["delta_axis_profiles"] = axes.result(top_k) if axes is not None else axis_result
        rows.append(row)
    identities = [cp.identity(hash_files=hash_files, chunk_bytes=chunk_bytes) for cp in (base, candidate)]
    base.assert_stable()
    candidate.assert_stable()

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        before = _aggregate(selected, "base")
        after = _aggregate(selected, "candidate")
        delta = _aggregate(selected, "delta")
        edited = [r for r in selected if r["stored_bit_changed_elements"]]
        edited_base = _aggregate(edited, "base")
        return {"base": before, "candidate": after, "delta": delta,
                "stored_bit_changed_elements": sum(r["stored_bit_changed_elements"] for r in selected),
                "numeric_changed_finite_elements": sum(r["numeric_changed_finite_elements"] for r in selected),
                "relative_delta_l2": _ratio(delta["l2"], before["l2"]),
                "edited_tensor_base_l2": edited_base["l2"],
                "relative_delta_l2_edited_tensors": _ratio(delta["l2"], edited_base["l2"])}
    configs = []
    config_maps = [{f["path"]: f for f in identity["files"]
                    if (cp.root / f["path"]) in cp.configuration_files}
                   for cp, identity in zip((base, candidate), identities, strict=True)]
    for name in sorted(set(config_maps[0]) | set(config_maps[1])):
        a_file, b_file = config_maps[0].get(name), config_maps[1].get(name)
        if a_file is None or b_file is None:
            state = "missing_from_base" if a_file is None else "missing_from_candidate"
        elif not hash_files:
            state = "not_hashed"
        else:
            state = "same" if a_file["sha256"] == b_file["sha256"] else "different"
        configs.append({"path": name, "status": state,
                        "base_sha256": a_file.get("sha256") if a_file else None,
                        "candidate_sha256": b_file.get("sha256") if b_file else None})
    io = [_io(cp, counts, estimated)
          for cp, counts, estimated in zip((base, candidate), initial_counts, estimates, strict=True)]
    global_result = aggregate(rows)
    global_result["stored_bit_changed_elements"] += sum(r["stored_bit_changed_elements"] for r in nonfloating)
    return {"schema_version": 1, "kind": "checkpoint_diff", "base": identities[0], "candidate": identities[1],
            "status": ("completed" if coverage_complete else "partial") if compatible else "no_comparable_tensors",
            "coverage_complete": coverage_complete, "allow_partial": allow_partial,
            "budget": budget, "io": {"base": io[0], "candidate": io[1]},
            "selection": {"pattern": pattern, "group_pattern": group_pattern,
                          "scope": "selected matching-shape/dtype stored floating tensors only",
                          "compared_tensors": len(compatible), "floating_compared_tensors": len(rows),
                          "missing_from_candidate": sorted(names_a - names_b),
                          "extra_in_candidate": sorted(names_b - names_a), "incompatible": incompatible},
            "global": global_result, "nonfloating_differences": nonfloating,
            "groups": {name: aggregate([r for r in rows if r["group"] == name])
                       for name in sorted({r["group"] for r in rows})},
            "configuration_files": configs, "tensors": rows,
            "bytes_read": sum(x["actual_bytes"]["total"] for x in io),
            "elapsed_seconds": time.monotonic() - start_time}
