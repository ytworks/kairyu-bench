#!/usr/bin/env python3
"""CPU-only exact low-rank norm and safetensors payload audit. No model load."""

from pathlib import Path
import argparse
import json
import math
import os
import resource
import struct
import time
import numpy as np


def inputs(argv=None, *, require_existing=False):
    global OUT, BASE, PAYLOAD, EXPECTED, CONFIG, GLOBAL_NORM, SPEC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--global-base-l2",
        type=float,
        required=True,
        help="Explicit denominator from a separately verified full-checkpoint norm",
    )
    parser.add_argument("--seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.global_base_l2)
        or args.global_base_l2 <= 0
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
    ):
        parser.error("positive finite norm and time budget required")
    BASE = args.source.resolve()
    OUT = args.output.resolve()
    SPEC = read(args.spec)
    PAYLOAD = Path(SPEC["donor_path"]).resolve()
    CONFIG = Path(SPEC["donor_config_path"]).resolve()
    EXPECTED = SPEC["donor_sha256"]
    GLOBAL_NORM = args.global_base_l2
    if OUT == BASE or OUT.is_relative_to(BASE) or BASE.is_relative_to(OUT):
        raise ValueError("audit output must be disjoint from source checkpoint")
    from dense_lora_recipe import validated_recipe

    recipe, mapping, _, _ = validated_recipe(BASE, SPEC)
    if require_existing:
        if not OUT.is_dir() or not (OUT / "norm-summary.json").is_file():
            raise ValueError("run norm audit first")
        prior = read(OUT / "audit-input.json")
        if (
            prior["spec"] != SPEC
            or prior["source"] != str(BASE)
            or prior["global_base_l2"] != GLOBAL_NORM
        ):
            raise ValueError("audit input differs from norm pass")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        save(
            OUT / "audit-input.json",
            {
                "spec": SPEC,
                "source": str(BASE),
                "global_base_l2": GLOBAL_NORM,
                "recipe": recipe,
                "mapping_sha256": recipe["mapping_sha256"],
                "global_denominator_status": "supplied separately; not measured by this tool",
            },
        )
    return args


def read(p):
    return json.loads(p.read_text())


def save(p, x):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(x, indent=2) + "\n")
    t.replace(p)


def header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 10_000_000:
            raise ValueError("header exceeds bounded parser")
        h = json.loads(f.read(n))
    return h, 8 + n


def tensor(path, h, data_offset, name):
    spec = h[name]
    a, b = spec["data_offsets"]
    if spec["dtype"] != "BF16":
        raise ValueError("unexpected dtype")
    with path.open("rb") as f:
        f.seek(data_offset + a)
        buf = f.read(b - a)
    raw = np.frombuffer(buf, dtype="<u2")
    if raw.size != math.prod(spec["shape"]):
        raise ValueError("truncated tensor")
    result = (raw.astype(np.uint32) << 16).view(np.float32).reshape(spec["shape"])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite public tensor")
    return result


def main():
    args = inputs()
    start = time.monotonic()
    from dense_lora_recipe import digest

    if digest(PAYLOAD) != EXPECTED:
        raise ValueError("payload SHA")
    h, off = header(PAYLOAD)
    config = read(CONFIG)
    if not (
        config["r"] == 16
        and config["lora_alpha"] == 32
        and not config["use_rslora"]
        and not config["use_dora"]
        and not config["fan_in_fan_out"]
        and config["bias"] == "none"
        and not config.get("rank_pattern")
        and not config.get("alpha_pattern")
    ):
        raise ValueError("ordinary fixed-scale LoRA assumptions differ")
    entries = {n: v for n, v in h.items() if n != "__metadata__"}
    extents = sorted(
        (v["data_offsets"][0], v["data_offsets"][1], n) for n, v in entries.items()
    )
    end = 0
    for a, b, n in extents:
        if a != end or b - a != 2 * math.prod(entries[n]["shape"]):
            raise ValueError("overlap/hole/size invalid")
        end = b
    if end + off != PAYLOAD.stat().st_size:
        raise ValueError("payload trailing/truncated data")
    index = read(BASE / "model.safetensors.index.json")["weight_map"]
    bh = {}
    for shard in set(index.values()):
        bh[shard] = header(BASE / shard)
    text_prefix = "base_model.model.model.language_model."
    text_names = [n for n in entries if n.startswith(text_prefix)]
    if len(text_names) != 820:
        raise ValueError("text tensor count")
    all_stats = {}
    pair_rows = []
    total_delta_ss = 0.0
    text_params = 0
    vision_params = 0
    for name in entries:
        if time.monotonic() - start > args.seconds:
            raise TimeoutError("adapter norm audit time budget")
        v = tensor(PAYLOAD, h, off, name)
        zeros = int(np.count_nonzero(v == 0))
        norm2 = float(np.sum(v.astype(np.float64) ** 2))
        all_stats[name] = {
            "shape": list(v.shape),
            "elements": v.size,
            "zeros": zeros,
            "finite": True,
            "norm_l2": math.sqrt(norm2),
            "max_abs": float(np.max(np.abs(v))),
        }
        if name.startswith(text_prefix):
            text_params += v.size
        else:
            vision_params += v.size
    for name in sorted(n for n in text_names if n.endswith(".lora_A.weight")):
        bname = name.replace(".lora_A.weight", ".lora_B.weight")
        if bname not in entries:
            raise ValueError("A without B")
        target = name.removeprefix("base_model.model.").replace(
            ".lora_A.weight", ".weight"
        )
        if target not in index:
            raise ValueError("text matrix mapping absent")
        A = tensor(PAYLOAD, h, off, name)
        B = tensor(PAYLOAD, h, off, bname)
        shape = bh[index[target]][0][target]["shape"]
        if list(A.shape) != [16, shape[1]] or list(B.shape) != [shape[0], 16]:
            raise ValueError("text mapping dimensions")
        A64 = A.astype(np.float64)
        B64 = B.astype(np.float64)
        ga = A64 @ A64.T
        gb = B64.T @ B64
        ss = 4 * float(np.einsum("ij,ji", gb, ga))
        ss = max(ss, 0.0)
        layer = int(target.split(".layers.")[1].split(".")[0])
        mechanism = target.split(".")[-2]
        zero_rows = int(np.count_nonzero(np.all(B == 0, axis=1)))
        zero_cols = int(np.count_nonzero(np.all(A == 0, axis=0)))
        guaranteed_zero = (
            zero_rows * shape[1] + zero_cols * shape[0] - zero_rows * zero_cols
        )
        row = {
            "base_tensor": target,
            "adapter_A": name,
            "adapter_B": bname,
            "base_shard": index[target],
            "shape": shape,
            "layer": layer,
            "mechanism": mechanism,
            "rank": 16,
            "lora_scale": 2.0,
            "unrounded_delta_l2": math.sqrt(ss),
            "matrix_elements": math.prod(shape),
            "A_zero_count": all_stats[name]["zeros"],
            "B_zero_count": all_stats[bname]["zeros"],
            "B_all_zero": bool(np.all(B == 0)),
            "A_all_zero": bool(np.all(A == 0)),
            "zero_output_rows": zero_rows,
            "zero_input_columns": zero_cols,
            "guaranteed_zero_delta_elements_from_axes": guaranteed_zero,
        }
        pair_rows.append(row)
        total_delta_ss += ss
    if len(pair_rows) != 410 or {x["layer"] for x in pair_rows} != set(range(60)):
        raise ValueError("whole-text 410/60 coverage")
    by_layer = {}
    by_mechanism = {}
    for x in pair_rows:
        by_layer[str(x["layer"])] = (
            by_layer.get(str(x["layer"]), 0.0) + x["unrounded_delta_l2"] ** 2
        )
        by_mechanism[x["mechanism"]] = (
            by_mechanism.get(x["mechanism"], 0.0) + x["unrounded_delta_l2"] ** 2
        )
    total = math.sqrt(total_delta_ss)
    global_norm = GLOBAL_NORM
    coefficients = [1 / 256, 1 / 64, 1 / 16, 1 / 4]
    report = {
        "status": "norm_payload_audit_completed",
        "payload_sha256": EXPECTED,
        "text_matrices": 410,
        "text_layers": 60,
        "text_adapter_parameters": text_params,
        "vision_adapter_parameters": vision_params,
        "all_adapter_tensors_finite": True,
        "adapter_total_tensors": len(entries),
        "all_B_zero_text_matrices": sum(x["B_all_zero"] for x in pair_rows),
        "unrounded_dense_text_elements": sum(x["matrix_elements"] for x in pair_rows),
        "global_base_l2": global_norm,
        "text_unrounded_delta_l2_at_c1": total,
        "relative_global_l2_at_c1": total / global_norm,
        "coefficient_unrounded_norms": [
            {
                "coefficient": c,
                "delta_l2": c * total,
                "relative_global_l2": c * total / global_norm,
            }
            for c in coefficients
        ],
        "layer_l2": {k: math.sqrt(v) for k, v in by_layer.items()},
        "mechanism_l2": {k: math.sqrt(v) for k, v in by_mechanism.items()},
        "formula": "deltaW(c) = c * (32/16) * B @ A = c * 2 * B @ A; ||2BA||F^2 = 4 trace((B^T B)(A A^T)). Float64 Gram accumulation.",
        "merge_proposal": "FP32 B@A, multiply by c*2, add original FP32-converted BF16 W, round once to BF16. Never replace tokenizer/template/config.",
        "scope": "Vision payload checked for finiteness only and excluded from merge proposal. No full checkpoint or GPU work.",
        "limitations": [
            "This audit does not establish training provenance, benchmark non-overlap or capability benefit.",
            "Global base norm is a separately supplied denominator; this audit does not measure or independently authenticate it.",
            "Unrounded low-rank norm is exact algebraically for stored BF16 factors; actual BF16 merged changed count/L2 requires separate block audit.",
        ],
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "gpu_used": False,
    }
    if digest(PAYLOAD) != EXPECTED:
        raise ValueError("donor changed during audit")
    save(OUT / "tensor-payload-statistics.json", all_stats)
    save(OUT / "text-410-map.json", pair_rows)
    save(OUT / "norm-summary.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "text_matrices",
                    "text_adapter_parameters",
                    "vision_adapter_parameters",
                    "all_B_zero_text_matrices",
                    "text_unrounded_delta_l2_at_c1",
                    "relative_global_l2_at_c1",
                    "coefficient_unrounded_norms",
                    "elapsed_seconds",
                    "peak_rss_KiB",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
