#!/usr/bin/env python3
"""Bridge existing dense CPU recipe verification to a redeploy-compatible freeze.

Preview never imports the numerical recipe or executes GEMM. --apply regenerates
the existing recipe once, then publishes a fresh proof; no model/GPU calls.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import io
import os
from pathlib import Path
import time

import verify_saved_checkpoint as common

KIND = "dense_lora_recipe_v1"
CODE_NAMES = ("dense_lora_recipe.py", "bf16_lora_ops.py", "bf16_lora_kernel.cpp")


def overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def prepare(args):
    manifest, _ = common.captured(args.manifest, args.manifest_sha256)
    spec, _ = common.captured(args.spec, args.spec_sha256)
    source, candidate = args.source.resolve(), args.candidate.resolve()
    if (args.source.is_symlink() or args.candidate.is_symlink() or overlap(source, candidate)
            or Path(manifest["source"]).resolve() != source
            or Path(manifest["candidate"]).resolve() != candidate
            or Path(manifest["spec_path"]).resolve() != args.spec.resolve()):
        raise ValueError("explicit source/candidate/spec identity mismatch")
    if (manifest.get("schema_version") != 2 or manifest.get("kind") != KIND
            or spec.get("schema_version") != 2 or spec.get("kind") != KIND
            or spec.get("scope") != "native_text410" or spec.get("native_edits")
            or spec.get("companion_required") or manifest["spec"] != spec
            or manifest["spec_sha256"] != args.spec_sha256):
        raise ValueError("pure schema-2 whole-text donor recipe required")
    recipe = manifest["recipe"]
    implementation = recipe["implementation_files"]
    if (set(implementation) != {*CODE_NAMES, "compiled_kernel"}
            or common.canonical(implementation) != recipe["implementation_sha256"]
            or args.kernel_sha256 != implementation["compiled_kernel"]):
        raise ValueError("numerical code/kernel pins differ from creator")
    common.stamp(args.kernel)  # Refuse a symlink loader target.
    kernel = args.kernel.resolve()
    directory = Path(__file__).resolve().parent
    pins = {
        args.manifest.resolve(): args.manifest_sha256,
        args.spec.resolve(): args.spec_sha256,
        kernel: args.kernel_sha256,
        **{directory / name: implementation[name] for name in CODE_NAMES},
    }
    # These are bridge/reader identities, not part of the numerical recipe.
    for path in (Path(__file__).resolve(), Path(common.__file__).resolve(),
                 Path(importlib.util.find_spec("kairyu_bench.wsp.weights.checkpoint").origin)):
        pins[path] = common.sha(path)
    donor_paths = (Path(spec["donor_path"]).resolve(), Path(spec["donor_config_path"]).resolve())
    if overlap(source, donor_paths[0]) or overlap(candidate, donor_paths[0]):
        raise ValueError("donor overlaps checkpoint")
    protected = (source, candidate, *pins, *donor_paths)
    output = args.output.absolute()
    if (os.path.lexists(output)
            or any(overlap(out, p) for out in (output, output.resolve()) for p in protected)):
        raise ValueError("fresh output outside all inputs required")
    # Hash only explicit metadata, implementation and kernel here. The existing
    # verifier owns donor and full checkpoint hashing during --apply.
    for path, expected in pins.items():
        if common.sha(path) != expected:
            raise ValueError("pinned input SHA mismatch")
    states = {p: common.stamp(p) for p in (*pins, *donor_paths)}
    before = {"source": common.inventory(source), "candidate": common.inventory(candidate)}
    for label, root in (("source", source), ("candidate", candidate)):
        rows = common.file_rows(manifest[label + "_files"])
        if (set(rows) != set(before[label])
                or any(r["size"] != before[label][n]["size"] for n, r in rows.items())
                or common.canonical([rows[n] for n in sorted(rows)]) != manifest[label + "_tree_sha256"]):
            raise ValueError("manifest full inventory/tree mismatch")
    if manifest["source_tree_sha256"] != spec["base_tree_sha256"]:
        raise ValueError("source control tree mismatch")
    if common.captured(args.manifest, args.manifest_sha256)[0] != manifest:
        raise ValueError("manifest changed during preparation")
    return manifest, source, candidate, pins, states, before, kernel


def verify(args):
    started = time.monotonic()
    manifest, source, candidate, pins, states, before, kernel = prepare(args)
    if not args.apply:
        return {"status": "preview", "checkpoint_path": str(candidate),
                "candidate_files": len(before["candidate"]), "output_written": False,
                "cpu_recipe_regenerated": False, "checkpoint_payload_hashed": False,
                "gpu_calls": 0, "model_calls": 0}
    # Set only this process's loader path, after matching it to creator evidence.
    old = os.environ.get("KAIRYU_BENCH_BF16_LORA_LIBRARY")
    os.environ["KAIRYU_BENCH_BF16_LORA_LIBRARY"] = str(kernel)
    try:
        recipe = importlib.import_module("dense_lora_recipe")
        if (Path(recipe.__file__).resolve() != Path(__file__).with_name("dense_lora_recipe.py").resolve()
                or Path(recipe.LIBRARY).resolve() != kernel):
            raise ValueError("loaded numerical implementation differs from pinned paths")
        with contextlib.redirect_stdout(io.StringIO()):
            verified = recipe.verify(args.manifest)
    finally:
        if old is None:
            os.environ.pop("KAIRYU_BENCH_BF16_LORA_LIBRARY", None)
        else:
            os.environ["KAIRYU_BENCH_BF16_LORA_LIBRARY"] = old
    if verified != manifest:
        raise ValueError("existing verifier returned a different manifest")
    if common.inventory(source) != before["source"] or common.inventory(candidate) != before["candidate"]:
        raise ValueError("checkpoint inventory/stat changed during verification")
    if any(common.stamp(p) != s for p, s in states.items()):
        raise ValueError("input stat changed during verification")
    if any(common.sha(p) != expected for p, expected in pins.items()):
        raise ValueError("pinned input changed during verification")
    rows = sorted(verified["candidate_files"], key=lambda row: row["path"])
    if common.canonical(rows) != verified["candidate_tree_sha256"]:
        raise ValueError("verified candidate tree mismatch")
    regenerated = len(verified["edits"])
    structured = verified["recipe"].get("structured_scope")
    expected = 180 if structured == "mlp_triplets_all_layers" else 410
    if regenerated != expected:
        raise ValueError("regenerated projection count differs from the verified scope")
    result = {
        "schema_version": 1, "status": "verified_dense_cpu_recipe",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(candidate), "source_checkpoint_path": str(source),
        "candidate": verified["spec"].get("id"),
        "candidate_inventory_stat": before["candidate"], "source_inventory_stat": before["source"],
        "all_checkpoint_files": rows, "control_tree_sha256": verified["candidate_tree_sha256"],
        "source_tree_sha256": verified["source_tree_sha256"],
        "manifest_path": str(args.manifest.resolve()), "manifest_sha256": args.manifest_sha256,
        "spec_path": str(args.spec.resolve()), "spec_sha256": args.spec_sha256,
        "compiled_kernel_sha256": args.kernel_sha256,
        "numerical_implementation": verified["recipe"],
        "input_sha256": {str(p): h for p, h in pins.items()},
        "proof_method": "CPU dense recipe regenerated against persisted BF16 bytes; no RAM/GPU comparison",
        "verified_scope": {"structured_scope": structured,
                           "validated_projection_count": 410,
                           "regenerated_projection_count": regenerated,
                           "zero_coefficient_projection_count": 410 - regenerated,
                           "zero_scope_and_other_spans_byte_verified": True},
        "verification": {"all_410_matrices_cpu_regenerated": regenerated == 410,
                         "source_and_candidate_full_file_hashes_verified": True,
                         "untouched_spans_verified": True, "before_after_inventory_stat_equal": True},
        "source_payload_rehashed_here": True, "fresh_candidate_full_hash": True,
        "changed_elements": verified["changed_elements"], "delta_l2": verified["delta_l2"],
        "cpu_recipe_verify_calls": 1, "gpu_calls": 0, "model_calls": 0,
        "ram_comparison_performed": False, "checkpoint_writes": 0,
        "seconds": time.monotonic() - started,
        "limitations": ["Weight-byte recipe verification is not a likelihood or coding result.",
                       "Subsequent redeployment reuses these full hashes with stat continuity; it does not rehash payloads."],
    }
    common.publish(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "candidate", "manifest", "spec", "kernel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest", "spec", "kernel"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args)
    print(common.json.dumps({k: result[k] for k in
                             ("status", "checkpoint_path", "gpu_calls", "model_calls")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
