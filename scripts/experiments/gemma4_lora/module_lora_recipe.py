#!/usr/bin/env python3
"""Versioned signed per-projection LoRA recipe; old scalar recipes stay unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import dense_lora_recipe as dense

KIND = "module_lora_recipe_v1"
SCHEMA_VERSION = 3
BLOCK_ROWS = dense.BLOCK_ROWS
MAX_COEFFICIENT = .25
MAX_CONTINUOUS_L2 = 1.0


def coefficient(value):
    if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > MAX_COEFFICIENT:
        raise ValueError("module coefficient must be finite with absolute value <=0.25")
    if float(np.float32(value)) != value:
        raise ValueError("module coefficient must already be exactly FP32")
    return float(value)


def delta_block(a, b):
    """Match the old scalar recipe's FP32 GEMM and multiply order exactly."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid factor block")
    return (b @ a) * np.float32(dense.SCALE)


def merge_block(original, delta, value):
    value = coefficient(value)
    original = np.asarray(original, dtype=np.uint16)
    if original.shape != np.shape(delta):
        raise ValueError("merge block shapes differ")
    if value == 0:
        # Addition of zero could change negative zero; zero scope preserves bytes.
        before = (original.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if not np.isfinite(before).all():
            raise ValueError("nonfinite original source")
        return original.copy(), np.array([np.sum(before * before), 0., 0.]), np.zeros(2, np.uint64)
    return dense.merge1(original, delta, value)


def validated_recipe(base, spec):
    if (spec.get("schema_version") != SCHEMA_VERSION or spec.get("kind") != KIND
            or spec.get("scope") != "native_text410" or spec.get("native_edits")):
        raise ValueError("unsupported module recipe schema/scope")
    scalar = dict(spec, schema_version=2, kind=dense.KIND, coefficient=1.0)
    recipe, mapping, header, offset = dense.validated_recipe(base, scalar)
    coefficients = spec.get("module_coefficients")
    names = {entry["tensor"] for entry in mapping}
    if not isinstance(coefficients, dict) or set(coefficients) != names:
        raise ValueError("all 410 module coefficients including zeros must be explicit")
    coefficients = {name: coefficient(coefficients[name]) for name in sorted(names)}
    costs_path = Path(spec["unit_costs_path"]).resolve()
    if costs_path.is_relative_to(base) or dense.digest(costs_path) != spec["unit_costs_sha256"]:
        raise ValueError("pinned unit-cost audit mismatch or overlaps source")
    costs = dense.read(costs_path)
    by_name = {row["base_tensor"]: row for row in costs}
    if len(costs) != 410 or set(by_name) != names:
        raise ValueError("unit-cost inventory mismatch")
    cost_values = {}
    for entry in mapping:
        row = by_name[entry["tensor"]]
        cost = row["unrounded_delta_l2"]
        if (row["adapter_A"] != entry["lora_A"] or row["adapter_B"] != entry["lora_B"]
                or row["shape"] != entry["shape"] or row["rank"] != 16 or row["lora_scale"] != dense.SCALE
                or type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0):
            raise ValueError("unit-cost tuple/value mismatch")
        cost_values[entry["tensor"]] = float(cost)
    norm = math.hypot(*(coefficients[name] * cost_values[name] for name in names))
    if norm > MAX_CONTINUOUS_L2:
        raise ValueError("continuous total weight L2 exceeds 1.0")
    code = dict(recipe["implementation_files"])
    code[Path(__file__).name] = dense.digest(Path(__file__))
    recipe.pop("coefficient")
    recipe.update(version=1, kind=KIND, module_coefficients=coefficients,
                  coefficients_sha256=dense.jsonhash(coefficients),
                  unit_costs_path=str(costs_path), unit_costs_sha256=spec["unit_costs_sha256"],
                  continuous_delta_l2=norm, maximum_absolute_coefficient=MAX_COEFFICIENT,
                  maximum_continuous_delta_l2=MAX_CONTINUOUS_L2,
                  implementation_files=code, implementation_sha256=dense.jsonhash(code),
                  arithmetic="Old FP32 (B_block@A)*2, per-module FP32 coefficient, separate FP32 add to original BF16, BF16 ties-to-even once; coefficient zero preserves original bytes; BLOCK_ROWS=128.")
    return recipe, mapping, header, offset


def active_mapping(recipe, mapping):
    return [entry for entry in mapping if recipe["module_coefficients"][entry["tensor"]] != 0]


def entries(base, candidate, recipe, mapping, header, offset, *, write=False):
    """Audit, create or verify using the same deterministic block primitive."""
    donor = Path(recipe["donor_path"])
    records = []
    for entry in active_mapping(recipe, mapping):
        name, shard = entry["tensor"], entry["shard"]
        value = recipe["module_coefficients"][name]
        a = dense.tensor(donor, header, offset, entry["lora_A"])
        b = dense.tensor(donor, header, offset, entry["lora_B"])
        before_sha, after_sha = hashlib.sha256(), hashlib.sha256()
        base_ss = delta_ss = maximum = 0.0
        changed = bytechanged = 0
        destination = None if candidate is None else candidate / shard
        if destination is not None and (os.path.samefile(base / shard, destination) or destination.stat().st_nlink != 1):
            raise ValueError("edited shard must be private, without external aliases")
        target = None if destination is None else destination.open("r+b" if write else "rb", buffering=0)
        try:
            with (base / shard).open("rb") as source:
                rows, columns = entry["shape"]
                for first in range(0, rows, BLOCK_ROWS):
                    count = min(BLOCK_ROWS, rows - first) * columns
                    position = entry["offset"] + first * columns * 2
                    source.seek(position)
                    raw = source.read(count * 2)
                    if len(raw) != count * 2:
                        raise ValueError("short source block")
                    original = np.frombuffer(raw, dtype="<u2").reshape(-1, columns)
                    delta = delta_block(a, b[first:first + len(original)])
                    merged, stats, counts = merge_block(original, delta, value)
                    expected = merged.astype("<u2", copy=False).tobytes()
                    if target is not None:
                        target.seek(position)
                        if target.read(count * 2) != (raw if write else expected):
                            raise ValueError("original or regenerated target block differs")
                        if write:
                            target.seek(position)
                            if target.write(expected) != len(expected):
                                raise ValueError("short target write")
                    before_sha.update(raw)
                    after_sha.update(expected)
                    base_ss += float(stats[0])
                    delta_ss += float(stats[1])
                    maximum = max(maximum, float(stats[2]))
                    changed += int(counts[0])
                    bytechanged += int(counts[1])
                if target is not None and write:
                    os.fsync(target.fileno())
        finally:
            if target is not None:
                target.close()
        records.append(dict(entry, entry_kind="lora_matrix", coefficient=value,
                            source_tensor_sha256=before_sha.hexdigest(), candidate_tensor_sha256=after_sha.hexdigest(),
                            changed_count=changed, changed_byte_count=bytechanged, source_l2=math.sqrt(base_ss),
                            delta_l2=math.sqrt(delta_ss), maximum_absolute_delta=maximum))
    return records


def metrics(rows):
    return {"changed_elements": sum(r["changed_count"] for r in rows),
            "changed_tensors": sum(r["changed_count"] > 0 for r in rows),
            "changed_byte_elements": sum(r["changed_byte_count"] for r in rows),
            "delta_l2": math.hypot(*(r["delta_l2"] for r in rows)), "actual_bf16_metrics": True}


def file_state(path):
    state = path.stat()
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def input_snapshot(base, spec_path, spec):
    paths = [*dense.files(base), spec_path.resolve(), Path(spec["donor_path"]).resolve(),
             Path(spec["donor_config_path"]).resolve(), Path(spec["unit_costs_path"]).resolve(), Path(__file__).resolve()]
    return {str(p): file_state(p) for p in paths}


def prepared(base, spec_path):
    base, spec_path = base.resolve(), spec_path.resolve()
    spec_state = file_state(spec_path)
    content = spec_path.read_bytes()
    if file_state(spec_path) != spec_state:
        raise ValueError("spec changed while reading")
    spec = json.loads(content)
    before = input_snapshot(base, spec_path, spec)
    recipe, mapping, header, offset = validated_recipe(base, spec)
    if spec_path.read_bytes() != content or input_snapshot(base, spec_path, spec) != before:
        raise ValueError("source/spec changed during preparation")
    return spec, content, before, recipe, mapping, header, offset


def final_inputs(base, spec_path, spec, content, before, recipe, mapping):
    current, current_mapping, _, _ = validated_recipe(base, spec)
    if (current != recipe or current_mapping != mapping or spec_path.read_bytes() != content
            or input_snapshot(base, spec_path, spec) != before):
        raise ValueError("source/spec/donor/cost/implementation changed during operation")


def audit(base, spec_path, output):
    if os.path.lexists(output):
        raise ValueError("audit output must be fresh")
    base = base.resolve()
    spec, content, before, recipe, mapping, header, offset = prepared(base, spec_path)
    for protected in (base, spec_path.resolve(), Path(recipe["donor_path"]), Path(recipe["unit_costs_path"])):
        if output.resolve() == protected or output.resolve().is_relative_to(protected):
            raise ValueError("audit output overlaps input")
    rows = entries(base, None, recipe, mapping, header, offset)
    final_inputs(base, spec_path, spec, content, before, recipe, mapping)
    report = {"schema_version": SCHEMA_VERSION, "kind": KIND, "status": "audited_actual_bf16",
              "source": str(base), "spec_sha256": hashlib.sha256(content).hexdigest(), "recipe": recipe,
              "edits": rows, **metrics(rows), "full_source_rehashed_here": False,
              "new_checkpoint": False, "model_calls": 0}
    dense.save(output, report, [])
    return report


def create(base, dest, spec_path, manifest, *, min_free_gib=0.):
    if os.path.lexists(dest) or os.path.lexists(manifest):
        raise ValueError("candidate and manifest must be fresh")
    base, dest, manifest = base.resolve(), dest.resolve(), manifest.resolve()
    if base == dest or dest.is_relative_to(base) or base.is_relative_to(dest) or manifest.is_relative_to(base) or manifest.is_relative_to(dest):
        raise ValueError("new disjoint candidate and external manifest required")
    spec, content, before, recipe, mapping, header, offset = prepared(base, spec_path)
    source_rows, source_hash = dense.tree(base)
    if source_hash != spec["base_tree_sha256"]:
        raise ValueError("source full tree hash mismatch")
    paths = dense.files(base)
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or shutil.disk_usage(dest.parent).free < min_free_gib * 2**30 + sum(p.stat().st_size for p in paths) + 64 * 2**20:
        raise ValueError("invalid reserve or insufficient free space")
    dest.mkdir(parents=True)
    owned_inode = (dest.stat().st_dev, dest.stat().st_ino)
    committed, commit_identity = False, []
    try:
        copied = []
        touched = {e["shard"] for e in active_mapping(recipe, mapping)}
        for source in paths:
            relative = source.relative_to(base)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(relative) in touched:
                subprocess.run(["cp", "--reflink=auto", "--sparse=always", str(source), str(target)], check=True)
            else:
                shutil.copy2(source, target)
            if os.path.samefile(source, target) or target.stat().st_nlink != 1:
                raise ValueError("candidate copy must be independent")
            copied.append({"path": str(relative), "size": source.stat().st_size, "method": "private-copy-or-reflink"})
        rows = entries(base, dest, recipe, mapping, header, offset, write=True)
        dense.compare_untouched(base, dest, active_mapping(recipe, mapping))
        result = metrics(rows)
        if not result["changed_elements"]:
            raise ValueError("module recipe vanished after BF16 rounding")
        candidate_state = {str(p): file_state(p) for p in dense.files(dest)}
        candidate_rows, candidate_hash = dense.tree(dest)
        if dense.tree(base)[1] != source_hash:
            raise ValueError("source changed during creation")
        final_inputs(base, spec_path, spec, content, before, recipe, mapping)
        if {str(p): file_state(p) for p in dense.files(dest)} != candidate_state:
            raise ValueError("candidate changed during final validation")
        record = {"schema_version": SCHEMA_VERSION, "kind": KIND, "source": str(base), "candidate": str(dest),
                  "spec": spec, "spec_path": str(spec_path.resolve()), "spec_sha256": hashlib.sha256(content).hexdigest(),
                  "recipe": recipe, "source_tree_sha256": source_hash, "candidate_tree_sha256": candidate_hash,
                  "source_files": source_rows, "candidate_files": candidate_rows, "files": copied, "edits": rows, **result}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        dense.save(manifest, record, commit_identity)
        committed = True
        return record
    finally:
        if commit_identity and manifest.exists() and not manifest.is_symlink():
            state = manifest.stat()
            committed = committed or (state.st_dev, state.st_ino) == commit_identity[-1]
        if not committed and dest.exists() and not dest.is_symlink() and (dest.stat().st_dev, dest.stat().st_ino) == owned_inode:
            shutil.rmtree(dest)


def verify(manifest):
    raw = manifest.read_bytes()
    record = json.loads(raw)
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != KIND:
        raise ValueError("unknown module recipe schema")
    base, dest, spec_path = Path(record["source"]).resolve(), Path(record["candidate"]).resolve(), Path(record["spec_path"])
    if base == dest or base.is_relative_to(dest) or dest.is_relative_to(base):
        raise ValueError("overlapping model trees")
    candidate_state = {str(p): file_state(p) for p in dense.files(dest)}
    spec, content, before, recipe, mapping, header, offset = prepared(base, spec_path)
    if spec != record["spec"] or hashlib.sha256(content).hexdigest() != record["spec_sha256"] or recipe != record["recipe"]:
        raise ValueError("spec or recipe implementation differs")
    if dense.tree(base) != (record["source_files"], record["source_tree_sha256"]) or record["source_tree_sha256"] != spec["base_tree_sha256"]:
        raise ValueError("source hash list differs")
    dense.compare_untouched(base, dest, active_mapping(recipe, mapping))
    rows = entries(base, dest, recipe, mapping, header, offset)
    dense.metrics(record["edits"], rows)
    for key, value in metrics(rows).items():
        if isinstance(value, float):
            if not math.isclose(record[key], value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("recorded actual norm differs")
        elif record[key] != value:
            raise ValueError("recorded actual count/qualification differs")
    if dense.tree(dest) != (record["candidate_files"], record["candidate_tree_sha256"]):
        raise ValueError("candidate tree hash list differs")
    final_inputs(base, spec_path, spec, content, before, recipe, mapping)
    if manifest.read_bytes() != raw or {str(p): file_state(p) for p in dense.files(dest)} != candidate_state:
        raise ValueError("manifest/candidate changed during verification")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    c = commands.add_parser("create")
    c.add_argument("source", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--spec", type=Path, required=True)
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--apply", action="store_true")
    c.add_argument("--min-free-gib", type=float, default=500.)
    a = commands.add_parser("audit")
    a.add_argument("source", type=Path)
    a.add_argument("--spec", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    if args.action == "verify":
        record = verify(args.manifest)
    elif args.action == "audit":
        record = audit(args.source, args.spec, args.output)
    elif args.apply:
        record = create(args.source, args.candidate, args.spec, args.manifest, min_free_gib=args.min_free_gib)
    else:
        if os.path.lexists(args.candidate) or os.path.lexists(args.manifest):
            parser.error("candidate and manifest must be fresh")
        recipe, mapping, _, _ = validated_recipe(args.source.resolve(), dense.read(args.spec))
        record = {"dry_run": True, "kind": KIND, "modules": len(mapping),
                  "active_modules": len(active_mapping(recipe, mapping)), "continuous_delta_l2": recipe["continuous_delta_l2"],
                  "actual_bf16_metrics": False}
    print(json.dumps({k: record[k] for k in ("kind", "changed_elements", "delta_l2", "actual_bf16_metrics", "dry_run", "continuous_delta_l2") if k in record} | {"seconds": time.monotonic() - start}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
