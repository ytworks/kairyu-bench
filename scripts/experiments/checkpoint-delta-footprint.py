#!/usr/bin/env python3
"""Bounded BF16 delta footprint; prior full hashes plus fresh stat continuity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint

MAX_CHUNK = 8 * 1024**2
MAX_METADATA = 16 * 1024**2
FIELDS = {"device": "dev", "inode": "ino", "size": "size", "mtime_ns": "mtime_ns",
          "ctime_ns": "ctime_ns", "uid": "uid", "gid": "gid", "mode": "mode", "nlink": "nlink"}
REQUIRED = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
SHA = re.compile(r"[0-9a-f]{64}")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stamp(path: Path) -> dict[str, int]:
    s = path.stat()
    return {key: int(getattr(s, "st_" + attr)) for key, attr in FIELDS.items()}


def inventory(root: Path) -> dict[str, dict[str, int]]:
    require(root.is_dir() and not root.is_symlink(), "regular checkpoint directory required")
    result = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), "checkpoint symlinks refused")
        if p.is_file() and p.name != ".kairyu-model-attestation.json" and ".cache" not in p.relative_to(root).parts:
            result[str(p.relative_to(root))] = stamp(p)
    return result


def captured(path: Path, expected: str) -> tuple[bytes, dict[str, int]]:
    require(bool(SHA.fullmatch(expected)), "SHA256 required")
    require(path.is_file() and not path.is_symlink(), "regular metadata file required")
    before = stamp(path)
    require(before["size"] <= MAX_METADATA, "metadata too large; weight payload references forbidden")
    with path.open("rb") as f:
        raw = f.read(MAX_METADATA + 1)
    require(len(raw) <= MAX_METADATA and hashlib.sha256(raw).hexdigest() == expected, "metadata SHA mismatch")
    require(stamp(path) == before, "metadata changed")
    return raw, before


def identity(value: dict[str, Any]) -> tuple[Checkpoint, dict[str, dict[str, int]]]:
    root = Path(value["checkpoint_path"])
    actual = inventory(root)
    recorded = value["inventory_stat"]
    rows = value["files"]
    require(set(actual) == set(recorded), "checkpoint inventory mismatch")
    require(len({r["path"] for r in rows}) == len(rows) == len(actual), "duplicate/missing file hash")
    require({r["path"] for r in rows} == set(actual), "file hashes do not cover inventory")
    for name, expected in recorded.items():
        require(REQUIRED <= set(expected) <= set(FIELDS), "complete stat fields required")
        require(all(actual[name][k] == v for k, v in expected.items()), "prior stat continuity failed")
    for row in rows:
        require(set(row) == {"path", "size", "sha256"}, "full file hash row required")
        require(row["size"] == actual[row["path"]]["size"] and bool(SHA.fullmatch(row["sha256"])), "invalid file hash row")
    require(digest(sorted(rows, key=lambda r: r["path"])) == value["tree_sha256"], "declared tree mismatch")
    return Checkpoint(root), actual


def empty() -> dict[str, Any]:
    return dict(tensors=0, changed_tensors=0, elements=0, stored_bit_changed_elements=0,
                numeric_changed_finite_elements=0, base_nonfinite_elements=0,
                candidate_nonfinite_elements=0, nonfinite_pairs=0, finite_delta_sum_squares=0.0)


def combine(target: dict[str, Any], row: dict[str, Any]) -> None:
    for key in empty():
        target[key] += row[key]


def finish(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["finite_pairs"] = row["elements"] - row["nonfinite_pairs"]
    row["base_finite_elements"] = row["elements"] - row["base_nonfinite_elements"]
    row["candidate_finite_elements"] = row["elements"] - row["candidate_nonfinite_elements"]
    row["finite_delta_l2"] = math.sqrt(row["finite_delta_sum_squares"])
    row["delta_l2"] = row["finite_delta_l2"] if not row["nonfinite_pairs"] else None
    return row


def footprint(base: Checkpoint, candidate: Checkpoint, chunk_bytes: int = MAX_CHUNK) -> dict[str, Any]:
    require(type(chunk_bytes) is int and 2 <= chunk_bytes <= MAX_CHUNK and chunk_bytes % 2 == 0, "bounded even chunk bytes required")
    require(set(base.tensors) == set(candidate.tensors), "full tensor name coverage required")
    names = sorted(base.tensors)
    require(bool(names), "empty checkpoint")
    for name in names:
        a, b = base.tensors[name], candidate.tensors[name]
        require(a.dtype == b.dtype == "BF16" and a.shape == b.shape, "same BF16 shape/dtype required")
    total = empty()
    layers: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    rows = []
    raw_bytes = equal_bytes = 0
    for name in names:
        tensor = base.tensors[name]
        row = empty()
        row.update(tensors=1, elements=tensor.numel)
        for start in range(0, tensor.numel, chunk_bytes // 2):
            count = min(chunk_bytes // 2, tensor.numel - start)
            left, right = base.read_raw(name, start, count), candidate.read_raw(name, start, count)
            raw_bytes += len(left) + len(right)
            a = np.frombuffer(left, dtype="<u2")
            bad_a = (a & 0x7F80) == 0x7F80
            na = int(np.count_nonzero(bad_a))
            row["base_nonfinite_elements"] += na
            if left == right:
                equal_bytes += len(left) + len(right)
                row["candidate_nonfinite_elements"] += na
                row["nonfinite_pairs"] += na
                continue
            b = np.frombuffer(right, dtype="<u2")
            bad_b = (b & 0x7F80) == 0x7F80
            row["candidate_nonfinite_elements"] += int(np.count_nonzero(bad_b))
            bad = bad_a | bad_b
            row["nonfinite_pairs"] += int(np.count_nonzero(bad))
            changed = a != b
            row["stored_bit_changed_elements"] += int(np.count_nonzero(changed))
            # Both signed zeros are numerically equal; all other finite BF16 words
            # have unique values. Nonfinite comparisons are reported separately.
            changed &= ~bad & ~(((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0))
            n = int(np.count_nonzero(changed))
            row["numeric_changed_finite_elements"] += n
            if n:
                av = (a[changed].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                bv = (b[changed].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                delta = bv - av
                # BF16 finite range squared and any possible file element count
                # fit F64. np.sum avoids a BLAS thread-pool roundtrip per block.
                row["finite_delta_sum_squares"] += float(np.sum(delta * delta, dtype=np.float64))
        row["changed_tensors"] = int(bool(row["stored_bit_changed_elements"]))
        require(math.isfinite(row["finite_delta_sum_squares"]), "nonfinite delta reduction")
        combine(total, row)
        match = re.match(r"^(.*\.layers\.\d+)\.", name)
        layer = match.group(1) if match else "__ungrouped__"
        family_match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$", name)
        family = family_match.group(1) if family_match else "__other__"
        combine(layers.setdefault(layer, empty()), row)
        combine(families.setdefault(family, empty()), row)
        rows.append({"name": name, "shape": list(tensor.shape), "dtype": tensor.dtype,
                     "layer": layer, "family": family, **finish(row)})
    base.assert_stable()
    candidate.assert_stable()
    return {"global": finish(total), "tensors": rows,
            "layers": {k: finish(v) for k, v in layers.items()},
            "families": {k: finish(v) for k, v in families.items()},
            "raw_payload_bytes_read": raw_bytes, "identical_chunk_bytes_skipping_decode": equal_bytes,
            "unchanged_stored_bit_tensors": len(rows) - total["changed_tensors"]}


def execute(plan_path: Path, plan_sha: str, output: Path, run: bool) -> dict[str, Any]:
    started = time.monotonic()
    raw, plan_stat = captured(plan_path, plan_sha)
    plan = json.loads(raw)
    require(plan["schema_version"] == 1 and plan["kind"] == "bf16-delta-footprint-plan", "plan schema required")
    require(not os.path.lexists(output), "fresh output required")
    code_path = Path(__file__).resolve()
    code_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
    _, code_stat = captured(code_path, code_sha)
    inputs = {str(plan_path.resolve()): (plan_sha, plan_stat), str(code_path): (code_sha, code_stat)}
    for evidence in plan["prior_evidence"]:
        p = Path(evidence["path"])
        _, state = captured(p, evidence["sha256"])
        key = str(p.resolve())
        pair = (evidence["sha256"], state)
        require(key not in inputs or inputs[key] == pair, "conflicting metadata pins")
        inputs[key] = pair
    require(bool(plan["prior_evidence"]), "prior identity evidence required")
    base, before_a = identity(plan["base"])
    candidate, before_b = identity(plan["candidate"])
    for cp in (base, candidate):
        require(not output.resolve().is_relative_to(cp.root), "output inside checkpoint forbidden")
    require(output.resolve() not in {Path(p) for p in inputs}, "output overlaps metadata input")
    require(set(base.tensors) == set(candidate.tensors), "full tensor coverage required")
    require(len(base.tensors) == plan["expected_tensors"], "unexpected tensor count")
    for name, a in base.tensors.items():
        b = candidate.tensors[name]
        require(a.dtype == b.dtype == "BF16" and a.shape == b.shape, "same BF16 layout required")
    chunk = plan.get("chunk_bytes", MAX_CHUNK)
    require(type(chunk) is int and 2 <= chunk <= MAX_CHUNK and chunk % 2 == 0, "bounded even chunk bytes required")
    result: dict[str, Any] = {
        "schema_version": 1, "status": "completed" if run else "preview",
        "scope": "All stored BF16 tensors, exact bit counts and finite numeric differences; full layout required.",
        "expected_tensors": len(base.tensors), "chunk_bytes": chunk,
        "estimated_pair_payload_bytes": 2 * sum(t.nbytes for t in base.tensors.values()),
        "estimated_working_bytes": 64 * 1024**2 + 40 * chunk,
        "identity_method": "Caller-pinned prior full-file SHA declarations plus current exact inventory/stat continuity; payload SHA not recomputed here.",
        "base": plan["base"], "candidate": plan["candidate"],
        "prior_evidence": plan["prior_evidence"], "fresh_payload_hash_performed": False,
        "model_calls": 0, "checkpoint_writes": 0,
        "limits": ["Nonfinite pairs excluded from finite_delta_l2; delta_l2 null if any are present.",
                   "No base norm, relative norm, quantiles, cosine or ability claim.",
                   "Logical tensor-name order, not a sequential physical file scan.",
                   "FP64 sum of finite BF16 differences squared; finite sum depends on reduction order within normal rounding."]
    }
    if run:
        result.update(footprint(base, candidate, chunk))
    require(inventory(base.root) == before_a and inventory(candidate.root) == before_b, "checkpoint changed during scan")
    for metadata_path, (expected, before) in inputs.items():
        _, after = captured(Path(metadata_path), expected)
        require(before == after, "metadata changed during scan")
    result.update(inventory_unchanged=True, input_sha256={p: v[0] for p, v in inputs.items()},
                  inventory_stat_before={"base": before_a, "candidate": before_b},
                  seconds=time.monotonic() - started,
                  implementation_sha256=code_sha)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".delta-footprint-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.link(name, output)
    finally:
        os.unlink(name)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", action="store_true", help="read payloads; default is metadata-only preview")
    a = p.parse_args()
    result = execute(a.plan, a.plan_sha256, a.output, a.run)
    print(json.dumps({k: result[k] for k in ("status", "expected_tensors", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
