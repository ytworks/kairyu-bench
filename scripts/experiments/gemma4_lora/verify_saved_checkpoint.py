#!/usr/bin/env python3
"""Verify saved BF16 bytes against finalized RAM receipts without regenerating LoRA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
import time

from kairyu_bench.wsp.weights.checkpoint import Checkpoint

CHUNK = 8 * 1024**2
MAX_JSON = 32 * 1024**2
PROJECTION = re.compile(r"model\.language_model\.layers\.(\d+)\.(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight")
MAPPING_KEYS = ("tensor", "shard", "offset", "size", "dtype", "shape", "lora_A", "lora_B")
COUNTS = ("changed_count", "changed_byte_count")
METRICS = ("source_l2", "delta_l2", "maximum_absolute_delta")
IMPLEMENTATIONS = {"module_lora_recipe.py", "dense_lora_recipe.py", "bf16_lora_ops.py", "bf16_lora_kernel.cpp", "compiled_kernel"}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stamp(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("regular nonsymlink file required")
    return {key: getattr(s, "st_" + attr) for key, attr in (
        ("device", "dev"), ("inode", "ino"), ("size", "size"), ("mtime_ns", "mtime_ns"),
        ("ctime_ns", "ctime_ns"), ("uid", "uid"), ("gid", "gid"), ("mode", "mode"), ("nlink", "nlink"))}


def sha(path):
    before = stamp(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_ino != before["inode"]:
            raise ValueError("file changed while opening")
        while block := stream.read(CHUNK):
            h.update(block)
    if stamp(path) != before:
        raise ValueError("file changed while hashing")
    return h.hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def captured(path, expected):
    before = stamp(path)
    if before["size"] > MAX_JSON or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError("bounded JSON and explicit SHA256 required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or stamp(path) != before:
        raise ValueError("input SHA/stat mismatch")
    return json.loads(raw, object_pairs_hook=unique), before


def inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("regular checkpoint directory required")
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if ".cache" in rel.parts or path.name == ".kairyu-model-attestation.json":
            continue
        if path.is_symlink():
            raise ValueError("checkpoint symlink refused")
        if not path.is_dir():
            result[str(rel)] = stamp(path)
    if not result:
        raise ValueError("empty checkpoint")
    return result


def file_rows(rows):
    result = {}
    for row in rows:
        name = row["path"]
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts
                or name in result or type(row["size"]) is not int or row["size"] < 0
                or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("invalid full file inventory")
        result[name] = {k: row[k] for k in ("path", "size", "sha256")}
    return result


def records(rows, label):
    result = {}
    for row in rows:
        name = row["tensor"]
        if name in result:
            raise ValueError(label + " duplicate tensor")
        result[name] = row
    return result


def coefficients(values):
    if not isinstance(values, dict):
        raise ValueError("explicit coefficient map required")
    for value in values.values():
        if (type(value) not in (float, int) or not math.isfinite(value) or abs(value) > .25
                or struct.unpack("<f", struct.pack("<f", value))[0] != value):
            raise ValueError("exact finite FP32 coefficients required")
    return values


def close(a, b):
    return (type(a) in (float, int) and type(b) in (float, int)
            and math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0
            and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12))


def prepare(args):
    paths = {
        "manifest": (args.manifest, args.manifest_sha256), "spec": (args.spec, args.spec_sha256),
        "summary": (args.replay / "summary.json", args.summary_sha256),
        "provenance": (args.replay / "provenance.json", args.provenance_sha256),
        "ram": (args.replay / "ram-weight-receipts.json", args.ram_sha256),
        "plan": (args.plan, args.plan_sha256),
        "attestation": (args.source_attestation, args.source_attestation_sha256),
    }
    documents, states = {}, {}
    for key, (path, expected) in paths.items():
        documents[key], states[str(path.resolve())] = captured(path, expected)
    m, spec, summary, prov, ram, plan, attested = (documents[k] for k in paths)
    source_path = Path(m["source"])
    for path in (source_path, args.candidate):
        if path.is_symlink():
            raise ValueError("checkpoint directory symlink refused")
    source_path, candidate = source_path.resolve(), args.candidate.resolve()
    if source_path == candidate or source_path.is_relative_to(candidate) or candidate.is_relative_to(source_path):
        raise ValueError("source/candidate overlap")
    if Path(m["candidate"]).resolve() != candidate or Path(m["spec_path"]).resolve() != args.spec.resolve():
        raise ValueError("manifest candidate/spec path mismatch")
    if (m.get("kind") != "module_lora_recipe_v1" or m.get("schema_version") != 3
            or spec.get("kind") != m["kind"] or spec.get("schema_version") != 3
            or spec.get("scope") != "native_text410" or spec.get("native_edits")
            or m["spec_sha256"] != args.spec_sha256 or m["spec"] != spec):
        raise ValueError("exact module recipe manifest/spec required")
    if (summary.get("status") != "completed" or summary.get("source_stable") is not True
            or summary.get("device_proof") != "staged-int16" or summary.get("error_type") is not None
            or summary.get("base_restoration", {}).get("matched") is not True
            or summary["base_restoration"].get("storage_bytes_exact") is not True
            or summary["base_restoration"].get("proof_method") != "staged-int16"
            or summary["base_restoration"].get("matrices") != 410
            or summary["base_restoration"].get("all_target_matrices") is not True):
        raise ValueError("finalized staged-int16 replay and restoration required")
    selected = [row for row in summary["candidates"] if row["id"] == spec["id"]]
    planned = [row for row in plan["candidates"] if row["id"] == spec["id"]]
    if (len(selected) != 1 or selected[0]["spec_sha256"] != args.spec_sha256 or len(planned) != 1
            or planned[0]["sha256"] != args.spec_sha256 or Path(planned[0]["spec"]).resolve() != args.spec.resolve()
            or prov["plan_sha256"] != args.plan_sha256 or prov.get("device_proof") != "staged-int16"
            or plan.get("architecture") != "gemma4_module_lora_bf16"):
        raise ValueError("selected candidate/plan/provenance mismatch")
    for path, expected in ((args.plan, args.plan_sha256), (args.spec, args.spec_sha256),
                           (args.source_attestation, args.source_attestation_sha256)):
        if prov["input_sha256"].get(str(path.resolve())) != expected:
            raise ValueError("replay input pin mismatch")
    for row in (plan["checkpoint_attestation"], prov["prior_source_attestation"]):
        if Path(row["path"]).resolve() != args.source_attestation.resolve() or row["sha256"] != args.source_attestation_sha256:
            raise ValueError("source attestation pin mismatch")
    receipt = ram[spec["id"]]
    if (receipt.get("id") != spec["id"] or receipt.get("device_proof") != "staged-int16"
            or receipt.get("actual_ram_storage_bytes_verified") is not True
            or receipt.get("actual_ram_readback_verified") is not False or receipt.get("actual_bf16_metrics") is not True):
        raise ValueError("CPU target SHA plus device equality receipt required; not D2H")
    recipe = m["recipe"]
    coefs = coefficients(spec["module_coefficients"])
    if recipe["module_coefficients"] != coefs or recipe["coefficients_sha256"] != canonical(coefs):
        raise ValueError("recipe coefficients mismatch")
    varying = {"module_coefficients", "coefficients_sha256", "continuous_delta_l2"}
    if {k: v for k, v in recipe.items() if k not in varying} != {k: v for k, v in prov["numeric_recipe"].items() if k not in varying}:
        raise ValueError("recipe numerical implementation/build mismatch")
    implementation = recipe["implementation_files"]
    if set(implementation) != IMPLEMENTATIONS or canonical(implementation) != recipe["implementation_sha256"]:
        raise ValueError("exact numerical implementation inventory required")
    for name, expected in implementation.items():
        path = args.kernel if name == "compiled_kernel" else args.implementation_dir / name
        if name == "compiled_kernel" and expected != args.kernel_sha256:
            raise ValueError("compiled kernel pin mismatch")
        if sha(path) != expected or prov["input_sha256"].get(str(path.resolve())) != expected:
            raise ValueError("compiled kernel/implementation differs from RAM replay")
        states[str(path.resolve())] = stamp(path)
    before = inventory(candidate)
    source_before = inventory(source_path)
    expected_files, source_files = file_rows(m["candidate_files"]), file_rows(m["source_files"])
    if set(expected_files) != set(before) or set(source_files) != set(source_before):
        raise ValueError("full file inventory mismatch")
    for actual, wanted in ((before, expected_files), (source_before, source_files)):
        if any(actual[name]["size"] != row["size"] for name, row in wanted.items()):
            raise ValueError("full file size mismatch")
    if (canonical([source_files[n] for n in sorted(source_files)]) != m["source_tree_sha256"]
            or m["source_tree_sha256"] != spec["base_tree_sha256"]
            or prov["declared_control_tree_sha256"] != spec["base_tree_sha256"]
            or canonical([expected_files[n] for n in sorted(expected_files)]) != m["candidate_tree_sha256"]):
        raise ValueError("creator/source canonical tree mismatch")
    source, saved = Checkpoint(source_path), Checkpoint(candidate)
    identity = attested.get("checkpoint", attested)
    current = source.identity(hash_files=False)
    if (identity.get("identity_level") != "content_sha256" or identity.get("hash_files") is not True
            or Path(identity["path"]).resolve() != source_path or len(identity["files"]) != len(current["files"])):
        raise ValueError("prior full source identity required")
    old = {f["path"]: f for f in identity["files"]}
    if len(old) != len(identity["files"]) or set(old) != {f["path"] for f in current["files"]}:
        raise ValueError("source recognized inventory differs")
    for row in current["files"]:
        prior = old[row["path"]]
        if prior["stat"] != row["stat"] or prior["size"] != row["size"] or prior["sha256"] != source_files[row["path"]]["sha256"]:
            raise ValueError("source prior hash/stat continuity mismatch")
    if set(source.tensors) != set(saved.tensors) or any(source.tensors[n].info() != saved.tensors[n].info() for n in source.tensors):
        raise ValueError("checkpoint tensor metadata changed")
    names = {n for n in source.tensors if PROJECTION.fullmatch(n)}
    if len(names) != 410 or {int(PROJECTION.fullmatch(n).group(1)) for n in names} != set(range(60)):
        raise ValueError("exact text410 across 60 layers required")
    rows, active = records(receipt["edits"], "RAM"), records(m["edits"], "creator")
    if set(rows) != names or set(coefs) != names or set(active) != {n for n in names if coefs[n] != 0}:
        raise ValueError("active/zero-omitted projection coverage mismatch")
    mapping = []
    for name in sorted(names):
        row, tensor = rows[name], saved.tensors[name]
        if (tensor.dtype != "BF16" or len(tensor.shape) != 2 or row["shape"] != list(tensor.shape)
                or row["dtype"] != "BF16" or row["shard"] != tensor.relative_shard
                or row["offset"] != tensor.offset or row["size"] != tensor.nbytes
                or row["entry_kind"] != "lora_matrix" or row["coefficient"] != coefs[name]
                or row.get("candidate_hash_origin") != "cpu_target_verified_by_device_int16_equality"):
            raise ValueError("RAM tensor mapping/coefficient/proof mismatch")
        if any(not re.fullmatch("[0-9a-f]{64}", row[key]) for key in ("source_tensor_sha256", "candidate_tensor_sha256")):
            raise ValueError("invalid tensor SHA")
        if any(type(row[k]) is not int or not 0 <= row[k] <= tensor.numel for k in COUNTS):
            raise ValueError("invalid changed count")
        mapping.append({k: row[k] for k in MAPPING_KEYS})
        if coefs[name] != 0:
            other = active[name]
            keys = (*MAPPING_KEYS, "coefficient", "entry_kind", "source_tensor_sha256", "candidate_tensor_sha256", *COUNTS)
            if any(row[k] != other[k] for k in keys) or any(not close(row[k], other[k]) for k in METRICS):
                raise ValueError("creator/RAM active tensor disagreement")
        elif (row["source_tensor_sha256"] != row["candidate_tensor_sha256"]
              or any(row[k] != 0 for k in COUNTS) or row["delta_l2"] != 0 or row["maximum_absolute_delta"] != 0):
            raise ValueError("zero coefficient must preserve source tensor")
    if canonical(mapping) != recipe["mapping_sha256"]:
        raise ValueError("full mapping hash mismatch")
    for container in (m, receipt):
        if (container.get("actual_bf16_metrics") is not True
                or container["changed_elements"] != sum(r["changed_count"] for r in rows.values())
                or container["changed_tensors"] != sum(r["changed_count"] > 0 for r in rows.values())
                or container["changed_byte_elements"] != sum(r["changed_byte_count"] for r in rows.values())
                or not close(container["delta_l2"], math.hypot(*(r["delta_l2"] for r in rows.values())))):
            raise ValueError("aggregate actual BF16 metrics mismatch")
    if selected[0]["changed_elements"] != receipt["changed_elements"] or not close(selected[0]["delta_l2"], receipt["delta_l2"]):
        raise ValueError("selected finite metrics mismatch")
    protected = [source_path, candidate, args.replay.resolve(), *map(Path, states)]
    out = args.output.resolve()
    if os.path.lexists(args.output) or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError("fresh output disjoint from all inputs required")
    return documents, paths, states, source, saved, before, source_before, expected_files, source_files, rows


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def verify(args):
    started = time.monotonic()
    docs, paths, states, source, saved, before, source_before, expected, source_files, rows = prepare(args)
    if not args.apply:
        return {"status": "preview", "candidate": str(saved.path), "matrices": 410,
                "candidate_files": len(expected), "candidate_payload_hashed": False, "output_written": False}
    actual = {}
    for name, row in rows.items():
        h = hashlib.sha256()
        tensor = saved.tensors[name]
        for first in range(0, tensor.numel, CHUNK // 2):
            h.update(saved.read_raw(name, first, min(CHUNK // 2, tensor.numel - first)))
        actual[name] = h.hexdigest()
        if actual[name] != row["candidate_tensor_sha256"]:
            raise ValueError("persisted tensor differs from verified RAM target: " + name)
    checkpoint_identity = saved.identity(hash_files=True, chunk_bytes=CHUNK)
    full = {r["path"]: {k: r[k] for k in ("path", "size", "sha256")} for r in checkpoint_identity["files"]}
    for name in set(expected) - set(full):
        if expected[name]["size"] > MAX_JSON or name.endswith(".safetensors"):
            raise ValueError("unrecognized non-small model file")
        full[name] = dict(expected[name], sha256=sha(saved.path / name))
    if full != expected:
        raise ValueError("persisted full file hashes differ from creator")
    # Source payload SHA was established by the prior attestation and creator.
    # Only extra small metadata outside Checkpoint.identity is freshly hashed.
    recognized = {r["path"] for r in source.identity(hash_files=False)["files"]}
    for name in set(source_files) - recognized:
        if source_files[name]["size"] > MAX_JSON or name.endswith(".safetensors") or sha(source.path / name) != source_files[name]["sha256"]:
            raise ValueError("source extra metadata hash mismatch")
    source.assert_stable()
    saved.assert_stable()
    if inventory(saved.path) != before or inventory(source.path) != source_before:
        raise ValueError("checkpoint changed during verification")
    for key, (path, expected_sha) in paths.items():
        if stamp(path) != states[str(path.resolve())] or sha(path) != expected_sha:
            raise ValueError("pinned evidence changed: " + key)
    if any(stamp(Path(p)) != state for p, state in states.items()):
        raise ValueError("implementation changed during verification")
    for name, digest in docs["manifest"]["recipe"]["implementation_files"].items():
        path = args.kernel if name == "compiled_kernel" else args.implementation_dir / name
        if sha(path) != digest:
            raise ValueError("implementation SHA changed during verification")
    files = [full[n] for n in sorted(full)]
    result = {
        "schema_version": 1, "status": "verified_saved_bf16_against_ram", "checkpoint_path": str(saved.path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "candidate": docs["spec"]["id"], "candidate_inventory_stat": before, "all_checkpoint_files": files,
        "control_tree_sha256": canonical(files), "checkpoint_identity": checkpoint_identity,
        "manifest_path": str(args.manifest.resolve()), "manifest_sha256": args.manifest_sha256,
        "spec_path": str(args.spec.resolve()), "spec_sha256": args.spec_sha256,
        "input_sha256": {str(p.resolve()): h for p, h in paths.values()},
        "compiled_kernel_sha256": args.kernel_sha256, "numerical_implementation": docs["manifest"]["recipe"]["implementation_files"],
        "verification": {"all_410_saved_tensor_sha_match": True, "active_and_zero_omitted_covered": True,
                         "all_candidate_file_sha_match_creator": True, "candidate_before_after_stat_equal": True,
                         "source_prior_recognized_hash_stat_continuity": True, "source_extra_metadata_sha_match": True},
        "proof_method": "CPU target SHA plus recorded all-element device int16 equality; NOT device-to-host readback",
        "source_payload_rehashed_here": False, "fresh_candidate_full_hash": True,
        "tensor_sha256": actual, "changed_elements": docs["ram"][docs["spec"]["id"]]["changed_elements"],
        "delta_l2": docs["ram"][docs["spec"]["id"]]["delta_l2"],
        "gemm_calls": 0, "gpu_calls": 0, "model_calls": 0, "checkpoint_writes": 0,
        "seconds": time.monotonic() - started,
        "limitations": ["Consumes pinned finalized measurement evidence; does not independently repeat device equality.",
                       "Stat continuity is not a fresh source payload hash and cannot detect every same-tick or privileged change.",
                       "Weight-byte equivalence is not a coding or generalization result. No deployment is performed."],
    }
    publish(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    for field in ("manifest", "spec", "plan", "source-attestation", "kernel"):
        parser.add_argument("--" + field, type=Path, required=True)
        parser.add_argument("--" + field + "-sha256", required=True)
    for field in ("summary", "provenance", "ram"):
        parser.add_argument("--" + field + "-sha256", required=True)
    parser.add_argument("--implementation-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="hash saved bytes and publish a fresh freeze; no weight writes")
    args = parser.parse_args(argv)
    result = verify(args)
    print(json.dumps({k: result[k] for k in ("status", "candidate", "gemm_calls", "seconds") if k in result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
