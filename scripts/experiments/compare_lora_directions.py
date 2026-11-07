#!/usr/bin/env python3
"""Compare ordinary LoRA directions by small factor-space Gram products, on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint

CHUNK = 8 * 1024**2
MAX_JSON = 16 * 1024**2


def stamp(path):
    s = path.stat()
    if not path.is_file():
        raise ValueError("input must be a regular file")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha(path, deadline=math.inf):
    before = stamp(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            if time.monotonic() > deadline:
                raise TimeoutError("observer time budget exceeded")
            h.update(data)
    if stamp(path) != before:
        raise ValueError("input changed while hashing")
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def pinned_json(path, expected):
    path = Path(path).resolve()
    before = stamp(path)
    if before[2] > MAX_JSON:
        raise ValueError("JSON metadata exceeds bound")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected or stamp(path) != before:
        raise ValueError("metadata SHA or stat differs")
    return json.loads(content, object_pairs_hook=unique_object), path, before


def scale(config, coefficient):
    if config.get("peft_type") != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("ordinary unbiased LoRA required")
    unsupported = ("use_dora", "use_rslora", "use_qalora", "use_bdlora", "fan_in_fan_out",
                   "lora_bias", "modules_to_save", "rank_pattern", "alpha_pattern",
                   "alora_invocation_tokens", "layer_replication", "target_parameters",
                   "trainable_token_indices", "arrow_config", "corda_config", "loftq_config",
                   "lora_ga_config", "megatron_config", "monteclora_config", "velora_config",
                   "ensure_weight_tying")
    if any(config.get(key) for key in unsupported):
        raise ValueError("unsupported LoRA transformation or saved parameter")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if type(rank) is not int or rank < 1 or type(alpha) not in (int, float):
        raise ValueError("positive rank and finite alpha required")
    if type(coefficient) not in (int, float) or not math.isfinite(coefficient):
        raise ValueError("finite coefficient required")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive finite alpha required")
    result = float(alpha / rank * coefficient)
    if not math.isfinite(result):
        raise ValueError("nonfinite effective scale")
    return rank, result


def output_guard(output, inputs, source):
    output = output.absolute()
    if os.path.lexists(output) or not output.parent.is_dir():
        raise ValueError("output must be fresh with an existing parent")
    target = output.resolve()
    if target.is_relative_to(source.path if source.path.is_dir() else source.path.parent):
        raise ValueError("output must be outside source checkpoint")
    if any(target == path or (path.is_dir() and target.is_relative_to(path)) for path in inputs):
        raise ValueError("output overlaps inputs")
    # Do not add a config or payload beside a single-file adapter.
    if target.name.endswith((".safetensors", ".safetensors.index.json")) or target.name in {
            "adapter_config.json", "config.json", "tokenizer.json"}:
        raise ValueError("output must not be a checkpoint/config filename")


def register(inputs, path, expected, state):
    binding = (expected, state)
    if path in inputs and inputs[path] != binding:
        raise ValueError("conflicting input pins for same path")
    inputs[path] = binding


def preflight(plan_path, plan_sha, output, max_module_bytes):
    plan, plan_path, plan_stat = pinned_json(plan_path, plan_sha)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported plan")
    inputs = {}
    register(inputs, plan_path, plan_sha, plan_stat)
    src = plan["source"]
    source = Checkpoint(src["path"])
    if source.identity(hash_files=False) != src["identity"]:
        raise ValueError("source header/inventory stat differs from plan")
    receipt, receipt_path, state = pinned_json(src["receipt_path"], src["receipt_sha256"])
    register(inputs, receipt_path, src["receipt_sha256"], state)
    if receipt.get("source_tree_sha256") != src["declared_tree_sha256"]:
        raise ValueError("source declared tree differs from pinned receipt")
    donors = {}
    for side in ("left", "right"):
        item = plan[side]
        config, path, state = pinned_json(item["config_path"], item["config_sha256"])
        register(inputs, path, item["config_sha256"], state)
        rank, multiplier = scale(config, item.get("coefficient", 1.0))
        donor = Checkpoint(item["path"])
        if donor.input_scope != "single_file" or len(donor.files) != 1:
            raise ValueError("each donor must be one explicit safetensors file")
        path = donor.files[0]
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError("explicit donor SHA256 required")
        register(inputs, path, item["sha256"], stamp(path))
        donors[side] = {"checkpoint": donor, "rank": rank, "scale": multiplier,
                        "config": config, "used": set(), "identity": donor.identity(hash_files=False)}
    rows = plan["mapping"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("nonempty bounded explicit mapping required")
    targets, estimates, flops = set(), [], 0
    for row in rows:
        target = row["tensor"]
        if target in targets or target not in source.tensors:
            raise ValueError("duplicate or unknown target tensor")
        targets.add(target)
        shape = source.tensors[target].shape
        if len(shape) != 2 or min(shape) < 1 or row["shape"] != list(shape):
            raise ValueError("source shape mismatch")
        if not isinstance(row.get("groups"), dict) or not row["groups"] or any(
                not isinstance(k, str) or not isinstance(v, (str, int)) or isinstance(v, bool)
                for k, v in row["groups"].items()):
            raise ValueError("explicit scalar group labels required")
        size = 0
        for side, donor in donors.items():
            cp, rank = donor["checkpoint"], donor["rank"]
            for kind, expected_shape in (("A", (rank, shape[1])), ("B", (shape[0], rank))):
                name = row[side][kind]
                if name not in cp.tensors or name in donor["used"]:
                    raise ValueError("missing or duplicate factor")
                info = cp.tensors[name]
                if info.dtype not in {"F32", "BF16"} or info.shape != expected_shape:
                    raise ValueError("factor shape/dtype/rank mismatch")
                donor["used"].add(name)
                size += info.numel * 8
        r, s = donors["left"]["rank"], donors["right"]["rank"]
        estimates.append(size + 8 * (3*r*r + 3*s*s + 3*r*s) + 4*CHUNK)
        flops += 2 * sum(shape) * (r*r + s*s + r*s)
    for side, donor in donors.items():
        excluded = plan[side].get("excluded_tensors", {})
        if not isinstance(excluded, dict) or any(not isinstance(v, str) or not v for v in excluded.values()):
            raise ValueError("every exclusion needs an explicit reason")
        remaining = set(donor["checkpoint"].tensors) - donor["used"]
        if set(excluded) != remaining or set(excluded) & donor["used"]:
            raise ValueError("unmapped factors must be explicitly excluded")
        if any(not name.endswith((".lora_A.weight", ".lora_B.weight")) for name in remaining):
            raise ValueError("extra non-LoRA parameters are unsupported")
    estimate = max(estimates)
    if estimate > max_module_bytes:
        raise ValueError("per-module factor/Gram estimate exceeds memory budget")
    for path in source.files:
        if path in inputs:
            raise ValueError("source overlaps analysis inputs")
    output_guard(output, list(inputs), source)
    return plan, inputs, source, donors, {
        "modules": len(rows), "max_working_bytes_estimate": estimate,
        "gram_flops_estimate": flops,
        "donor_payload_bytes": sum(d["checkpoint"].files[0].stat().st_size for d in donors.values()),
        "payload_read_passes": "two full hash passes plus mapped factor reads; no base payload read",
        "working_estimate_scope": "factor/Gram arrays plus read buffers; BLAS/runtime workspace is not a measured peak"}


def factor(checkpoint, name, deadline):
    info = checkpoint.tensors[name]
    values = np.empty(info.numel, dtype=np.float64)
    for start, _, chunk in checkpoint.iter_chunks(name, CHUNK):
        if time.monotonic() > deadline:
            raise TimeoutError("observer time budget exceeded")
        if not np.isfinite(chunk).all():
            raise ValueError("nonfinite factor: " + name)
        values[start:start+chunk.size] = chunk
    return values.reshape(info.shape)


def contraction(a, b, scale_value):
    ga, gb = a @ a.T, b.T @ b
    products = ga * gb
    total = math.fsum(float(x) for x in products.flat)
    magnitude = math.fsum(abs(float(x)) for x in products.flat)
    tolerance = 64 * np.finfo(np.float64).eps * max(*a.shape, *b.shape) * magnitude
    if total < -tolerance:
        raise ValueError("negative Gram squared norm beyond rounding tolerance")
    return max(total, 0.0) * scale_value**2, total < 0


def geometry(left_squared, right_squared, dot):
    if not all(math.isfinite(x) for x in (left_squared, right_squared, dot)):
        raise ValueError("nonfinite factor-space result")
    left, right = math.sqrt(left_squared), math.sqrt(right_squared)
    cosine = dot / (left * right) if left and right else None
    if cosine is not None and abs(cosine) > 1 + 1e-9:
        raise ValueError("factor-space Cauchy bound violated; numerical result unresolved")
    return {"left_norm_squared": left_squared, "right_norm_squared": right_squared,
            "left_l2": left, "right_l2": right, "dot": dot,
            "cosine": max(-1., min(1., cosine)) if cosine is not None else None,
            "cosine_roundoff_clipped": cosine is not None and abs(cosine) > 1}


def aggregate(rows):
    return geometry(*(math.fsum(row[key] for row in rows)
                      for key in ("left_norm_squared", "right_norm_squared", "dot")))


def publish(path, result):
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lora-compare-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive publication, including dangling symlinks.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def analyze(plan_path, plan_sha, output, *, run=False, seconds=600., max_module_mib=256):
    if not math.isfinite(seconds) or seconds <= 0 or max_module_mib < 1:
        raise ValueError("positive time and memory budgets required")
    started = time.monotonic()
    deadline = started + seconds
    plan, inputs, source, donors, estimate = preflight(
        Path(plan_path), plan_sha, Path(output), max_module_mib*1024**2)
    basic = {"schema_version": 1, "status": "preview", "estimate": estimate,
             "source": plan["source"], "source_full_payload_hashed_here": False,
             "donors": {s: {"rank": d["rank"], "effective_scale": d["scale"],
                            "identity": d["identity"], "provenance": plan[s]} for s, d in donors.items()},
             "model_calls": 0, "gpu_used": False, "dense_updates_materialized": 0}
    if not run:
        return basic
    code = Path(__file__).resolve()
    register(inputs, code, sha(code, deadline), stamp(code))
    reader = Path(__import__(Checkpoint.__module__, fromlist=["__file__"]).__file__).resolve()
    register(inputs, reader, sha(reader, deadline), stamp(reader))
    for path, (expected, state) in inputs.items():
        if stamp(path) != state or sha(path, deadline) != expected:
            raise ValueError("input SHA/stat differs before analysis")
    records, grouped = [], {}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for row in plan["mapping"]:
            factors = {side: [factor(d["checkpoint"], row[side][k], deadline) for k in ("A", "B")]
                       for side, d in donors.items()}
            a, b = factors["left"]
            c, d = factors["right"]
            s1, s2 = donors["left"]["scale"], donors["right"]["scale"]
            n1, clamp1 = contraction(a, b, s1)
            n2, clamp2 = contraction(c, d, s2)
            cross = (b.T @ d) * (a @ c.T)
            dot = s1 * s2 * math.fsum(float(x) for x in cross.flat)
            record = {"tensor": row["tensor"], "shape": row["shape"], "groups": row["groups"],
                      "norm_roundoff_clamped": {"left": clamp1, "right": clamp2}, **geometry(n1, n2, dot)}
            records.append(record)
            for key, value in row["groups"].items():
                grouped.setdefault(key, {}).setdefault(str(value), []).append(record)
            del factors, a, b, c, d, cross
    source.assert_stable()
    if source.identity(hash_files=False) != plan["source"]["identity"]:
        raise ValueError("source changed during observation")
    for donor in donors.values():
        donor["checkpoint"].assert_stable()
    for path, (expected, state) in inputs.items():
        if stamp(path) != state or sha(path, deadline) != expected:
            raise ValueError("input changed during observation")
    result = dict(basic, status="completed", modules=records, global_geometry=aggregate(records),
                  groups={key: {value: aggregate(rows) for value, rows in buckets.items()}
                          for key, buckets in grouped.items()},
                  input_sha256={str(path): expected for path, (expected, _) in inputs.items()},
                  input_stat={str(path): state for path, (_, state) in inputs.items()},
                  source_stable=True, numpy_version=np.__version__,
                  elapsed_seconds=time.monotonic()-started,
                  peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  peak_rss_unit="KiB on Linux; bytes on macOS",
                  arithmetic="Float64 factor Gram products and math.fsum; algebraically full directions, not sampled coordinates.",
                  formula="dot(s1 B1 A1,s2 B2 A2)=s1 s2 sum((B1.T B2)*(A1 A2.T))",
                  limitations=["Not actual BF16 merged deltas, activation responses, or capability evidence.",
                               "No source payload rehash; pinned receipt plus current header/stat continuity only.",
                               "Adapter training-base revision and benchmark overlap are not authenticated.",
                               "Excluded tensors are not numerically inspected; group cosine is aggregated dot/norm, not mean module cosine.",
                               "Floating-point calculation, not a symbolic exact arithmetic proof; severe inconsistency fails closed."])
    output_guard(Path(output), list(inputs), source)
    publish(Path(output), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--max-module-mib", type=int, default=256)
    parser.add_argument("--run", action="store_true", help="read factors and publish a fresh CPU report")
    args = parser.parse_args(argv)
    result = analyze(args.plan, args.plan_sha256, args.output, run=args.run,
                     seconds=args.seconds, max_module_mib=args.max_module_mib)
    print(json.dumps({k: result[k] for k in ("status", "estimate", "model_calls", "gpu_used")}, indent=2))


if __name__ == "__main__":
    main()
