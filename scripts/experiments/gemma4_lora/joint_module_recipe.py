#!/usr/bin/env python3
"""Create a private checkpoint from a fixed disjoint native/module joint pair.

Preview is the default. Numerical module primitives and the native editor are
reused unchanged. No serving, GPU, model API, or benchmark operations occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import module_lora_recipe as module
from kairyu_bench.wsp.weights import checkpoint as checkpoint_reader
from kairyu_bench.wsp.weights import editor

dense = module.dense
KIND = "joint_native_module_recipe_v1"
SCHEMA_VERSION = 4


def captured(path):
    path = path.resolve()
    before = module.file_state(path)
    if before[2] > 32 * 2**20:
        raise ValueError("bounded JSON input required")
    raw = path.read_bytes()
    if module.file_state(path) != before:
        raise ValueError("input changed while reading")
    return json.loads(raw), raw, before


def prepare(base, spec_path, native_path):
    base, spec_path, native_path = base.resolve(), spec_path.resolve(), native_path.resolve()
    prepared = module.prepared(base, spec_path)
    spec, _, _, recipe, mapping, _, _ = prepared
    native, raw, native_state = captured(native_path)
    if not spec.get("companion_required") or not native.get("companion_required"):
        raise ValueError("both joint components must require their companion")
    for key in ("joint_report_path", "joint_report_sha256"):
        if not spec.get(key) or spec[key] != native.get(key):
            raise ValueError("joint report binding differs")
    report_path = Path(spec["joint_report_path"]).resolve()
    report, report_raw, report_state = captured(report_path)
    if (hashlib.sha256(report_raw).hexdigest() != spec["joint_report_sha256"]
            or Path(report["source_checkpoint"]).resolve() != base
            or report.get("disjoint_weight_tensors") is not True):
        raise ValueError("joint report source/hash/disjointness mismatch")
    if not isinstance(native.get("edits"), list) or not native["edits"]:
        raise ValueError("native exact edits required")
    count = 0
    for edit in native["edits"]:
        if (set(edit) != {"tensor", "indices", "set_values"}
                or not isinstance(edit["tensor"], str)
                or not edit["tensor"].startswith("model.language_model.")
                or not isinstance(edit["indices"], list)
                or not isinstance(edit["set_values"], list)
                or len(edit["indices"]) != len(edit["set_values"])
                or any(type(v) not in (int, float) for v in edit["set_values"])):
            raise ValueError("native component requires exact scalar/vector set_values")
        values = np.asarray(edit["set_values"], np.float64)
        f32 = values.astype(np.float32)
        if not np.isfinite(values).all() or not np.array_equal(values, f32.astype(np.float64)) or np.any(f32.view(np.uint32) & 65535):
            raise ValueError("native set_values must already be finite exact BF16")
        count += len(values)
    if count > 1_000_000:
        raise ValueError("native coordinate budget exceeded")
    patches = editor.plan_edits(base, native)
    matrix_names = {entry["tensor"] for entry in mapping}
    for patch in patches:
        if (patch["tensor"] in matrix_names or patch["dtype"] != "BF16" or len(patch["shape"]) > 1
                or set(patch["changed_indices"]) != set(patch["selected_indices"])):
            raise ValueError("native must be disjoint BF16 scalar/vector non-noop coordinates")
        before, after = bytes.fromhex(patch["before_hex"]), bytes.fromhex(patch["after_hex"])
        patch.update(entry_kind="native_sparse_bytes",
                     changed_byte_count=sum(before[i:i+2] != after[i:i+2] for i in range(0, len(before), 2)),
                     maximum_absolute_delta=max(abs(a-b) for a, b in
                                                zip(editor.decode(before, "BF16"), editor.decode(after, "BF16"), strict=True)))
    # Both tensor names and byte regions must be disjoint.
    spans = {}
    for row in [*mapping, *patches]:
        spans.setdefault(row["shard"], []).append((row["offset"], row["offset"] + row["size"]))
    for ranges in spans.values():
        stop = 0
        for lo, hi in sorted(ranges):
            if lo < stop:
                raise ValueError("joint tensor byte spans overlap")
            stop = hi
    code_paths = [Path(__file__).resolve(), Path(editor.__file__).resolve(),
                  Path(checkpoint_reader.__file__).resolve()]
    tracked = {str(p): module.file_state(p) for p in code_paths}
    tracked[str(native_path)], tracked[str(report_path)] = native_state, report_state
    code = {p.name: dense.digest(p) for p in code_paths}
    result = {"module": prepared, "native_spec": native, "native_content": raw,
              "native_path": native_path, "native_patches": patches, "report_path": report_path,
              "report_content": report_raw, "tracked": tracked, "code": code}
    check_inputs(base, spec_path, result)
    return result


def check_inputs(base, spec_path, prepared):
    spec, content, before, recipe, mapping, _, _ = prepared["module"]
    module.final_inputs(base, spec_path, spec, content, before, recipe, mapping)
    if (prepared["native_path"].read_bytes() != prepared["native_content"]
            or prepared["report_path"].read_bytes() != prepared["report_content"]
            or any(module.file_state(Path(p)) != s for p, s in prepared["tracked"].items())):
        raise ValueError("native/report/joint implementation changed during operation")


def write_native(base, dest, patches):
    """Apply exact editor-produced words, with source and private-target checks."""
    for patch in patches:
        source, target = base / patch["shard"], dest / patch["shard"]
        before, after = bytes.fromhex(patch["before_hex"]), bytes.fromhex(patch["after_hex"])
        if os.path.samefile(source, target) or target.stat().st_nlink != 1:
            raise ValueError("native output shard must be private")
        if editor.read_at(source, patch["offset"], patch["size"]) != before:
            raise ValueError("native source bytes changed")
        with target.open("r+b", buffering=0) as stream:
            stream.seek(patch["offset"])
            if stream.read(patch["size"]) != before:
                raise ValueError("native target is not original source bytes")
            stream.seek(patch["offset"])
            if stream.write(after) != len(after):
                raise ValueError("short native write")
            os.fsync(stream.fileno())
        if editor.read_at(target, patch["offset"], patch["size"]) != after:
            raise ValueError("native stored byte verification failed")


def output_paths(base, dest, spec_path, native_path, manifest, prepared):
    if os.path.lexists(dest) or os.path.lexists(manifest):
        raise ValueError("candidate and manifest must be fresh")
    if manifest == dest or manifest.is_relative_to(dest) or dest.is_relative_to(manifest):
        raise ValueError("manifest must be outside candidate")
    spec = prepared["module"][0]
    protected = [base, spec_path, native_path, prepared["report_path"],
                 Path(spec["donor_path"]).resolve(), Path(spec["donor_config_path"]).resolve(),
                 Path(spec["unit_costs_path"]).resolve(), *map(Path, prepared["tracked"])]
    for output in (dest, manifest):
        if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
            raise ValueError("joint output overlaps protected input")


def create(base, dest, spec_path, native_path, manifest, *, min_free_gib=500.):
    base, dest = base.resolve(), dest.absolute()
    spec_path, native_path, manifest = spec_path.resolve(), native_path.resolve(), manifest.absolute()
    prepared = prepare(base, spec_path, native_path)
    output_paths(base, dest, spec_path, native_path, manifest, prepared)
    dest, manifest = dest.resolve(), manifest.resolve()
    output_paths(base, dest, spec_path, native_path, manifest, prepared)
    spec, content, _, recipe, mapping, header, offset = prepared["module"]
    source_rows, source_hash = dense.tree(base)
    if source_hash != spec["base_tree_sha256"]:
        raise ValueError("source full tree hash mismatch")
    paths = dense.files(base)
    if (not math.isfinite(min_free_gib) or min_free_gib < 0
            or shutil.disk_usage(dest.parent).free < min_free_gib * 2**30
            + sum(p.stat().st_size for p in paths) + 64 * 2**20):
        raise ValueError("invalid reserve or insufficient free space")
    dest.mkdir(parents=True)
    inode = module.file_state(dest)[:2]
    committed, commit_identity = False, []
    try:
        copied = []
        for source in paths:
            relative = source.relative_to(base)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "--sparse=always", str(source), str(target)], check=True)
            if os.path.samefile(source, target) or target.stat().st_nlink != 1:
                raise ValueError("every candidate file must be private")
            copied.append({"path": str(relative), "size": source.stat().st_size,
                           "method": "private-copy-or-reflink"})
        rows = module.entries(base, dest, recipe, mapping, header, offset, write=True)
        write_native(base, dest, prepared["native_patches"])
        rows += prepared["native_patches"]
        dense.compare_untouched(base, dest, [*module.active_mapping(recipe, mapping), *prepared["native_patches"]])
        result = module.metrics(rows)
        candidate_state = {str(p): module.file_state(p) for p in dense.files(dest)}
        candidate_rows, candidate_hash = dense.tree(dest)
        if dense.tree(base)[1] != source_hash:
            raise ValueError("source changed during joint creation")
        check_inputs(base, spec_path, prepared)
        if {str(p): module.file_state(p) for p in dense.files(dest)} != candidate_state:
            raise ValueError("candidate changed during final validation")
        record = {"schema_version": SCHEMA_VERSION, "kind": KIND, "source": str(base), "candidate": str(dest),
                  "spec": spec, "spec_path": str(spec_path), "spec_sha256": hashlib.sha256(content).hexdigest(),
                  "recipe": recipe, "native_spec": prepared["native_spec"], "native_spec_path": str(native_path),
                  "native_spec_sha256": hashlib.sha256(prepared["native_content"]).hexdigest(),
                  "joint_report_path": str(prepared["report_path"]), "joint_report_sha256": spec["joint_report_sha256"],
                  "joint_implementation_files": prepared["code"],
                  "source_tree_sha256": source_hash, "candidate_tree_sha256": candidate_hash,
                  "source_files": source_rows, "candidate_files": candidate_rows,
                  "files": copied, "edits": rows, "tensor_disjoint": True, **result}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        dense.save(manifest, record, commit_identity)
        committed = True
        return record
    finally:
        if commit_identity and manifest.exists() and not manifest.is_symlink():
            committed = committed or module.file_state(manifest)[:2] == commit_identity[-1]
        if not committed and dest.exists() and not dest.is_symlink() and module.file_state(dest)[:2] == inode:
            shutil.rmtree(dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create"])
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--native-spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=500.)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply:
        record = create(args.source, args.candidate, args.spec, args.native_spec, args.manifest,
                        min_free_gib=args.min_free_gib)
    else:
        prepared = prepare(args.source.resolve(), args.spec.resolve(), args.native_spec.resolve())
        output_paths(args.source.resolve(), args.candidate.absolute(), args.spec.resolve(),
                     args.native_spec.resolve(), args.manifest.absolute(), prepared)
        output_paths(args.source.resolve(), args.candidate.resolve(), args.spec.resolve(),
                     args.native_spec.resolve(), args.manifest.resolve(), prepared)
        record = {"kind": KIND, "dry_run": True, "native_changed_elements": module.metrics(prepared["native_patches"])["changed_elements"],
                  "native_delta_l2": module.metrics(prepared["native_patches"])["delta_l2"],
                  "module_continuous_delta_l2": prepared["module"][3]["continuous_delta_l2"]}
    print(json.dumps({k: v for k, v in record.items() if k in {
        "kind", "dry_run", "changed_elements", "delta_l2", "native_changed_elements",
        "native_delta_l2", "module_continuous_delta_l2"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
