#!/usr/bin/env python3
"""Read-only block audit of actual FP32 LoRA merge rounded once to BF16."""

import json
import math
import os
import resource
import time
import numpy as np
from bf16_lora_ops import audit5
import audit_public_adapter_norms as norms
from audit_public_adapter_norms import header, tensor, read, save

COEFFICIENTS = [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0]
BLOCK_ROWS = 128


def bf16bits(a):
    bits = np.asarray(a, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def sumsq(x):
    return float(np.einsum("ij,ij->", x, x, dtype=np.float64, optimize=False))


def dot(a, b):
    return float(np.einsum("ij,ij->", a, b, dtype=np.float64, optimize=False))


def main():
    args = norms.inputs(require_existing=True)
    OUT = norms.OUT
    BASE = norms.BASE
    PAYLOAD = norms.PAYLOAD
    EXPECTED = norms.EXPECTED
    start = time.monotonic()
    h, off = header(PAYLOAD)
    pairs = read(OUT / "text-410-map.json")
    if (OUT / "realized-summary.json").exists():
        raise ValueError("do not overwrite completed realized audit")
    source_stats = {
        p.name: {
            "size": p.stat().st_size,
            "inode": p.stat().st_ino,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in BASE.glob("*.safetensors")
    }
    base_headers = {p.name: header(p) for p in BASE.glob("*.safetensors")}
    if (OUT / "realized-matrix-details.json").exists():
        raise ValueError(
            "partial output exists; preserve it and use a fresh norm/audit directory"
        )
    completed = []
    already = len(completed)
    previous_seconds = (
        read(OUT / "realized-progress.json")["elapsed_seconds"] if already else 0.0
    )
    base_bytes = sum(math.prod(x["shape"]) * 2 for x in completed)
    observed_elements = base_bytes // 2
    global_norm = read(OUT / "norm-summary.json")["global_base_l2"]
    for ordinal, pair in enumerate(pairs):
        if ordinal < already:
            continue
        name = pair["base_tensor"]
        rows, cols = pair["shape"]
        shard = pair["base_shard"]
        A = tensor(PAYLOAD, h, off, pair["adapter_A"])
        B = tensor(PAYLOAD, h, off, pair["adapter_B"])
        bh, bo = base_headers[shard]
        entry = bh[name]
        if entry["dtype"] != "BF16" or entry["shape"] != pair["shape"]:
            raise ValueError("source mapping changed")
        base_map = np.memmap(
            BASE / shard,
            dtype="<u2",
            mode="r",
            offset=bo + entry["data_offsets"][0],
            shape=(rows, cols),
        )
        data = {
            "base_tensor": name,
            "layer": pair["layer"],
            "mechanism": pair["mechanism"],
            "shape": pair["shape"],
            "base_l2_squared": 0.0,
            "fp32_unrounded_delta_l2_squared": 0.0,
            "fp32_delta_zero_elements": 0,
            "coefficients": {
                str(c): {
                    "changed_value_elements": 0,
                    "changed_byte_elements": 0,
                    "signed_zero_only_changes": 0,
                    "realized_delta_l2_squared": 0.0,
                    "dot_with_full_fp32_delta": 0.0,
                    "maximum_absolute_delta": 0.0,
                }
                for c in COEFFICIENTS
            },
        }
        for first in range(0, rows, BLOCK_ROWS):
            if time.monotonic() - start > args.seconds:
                raise TimeoutError("read-only donor audit time budget")
            last = min(first + BLOCK_ROWS, rows)
            raw = np.asarray(base_map[first:last])
            delta = (B[first:last] @ A) * np.float32(2.0)
            measured, counters = audit5(raw, delta, COEFFICIENTS)
            data["base_l2_squared"] += float(measured[0])
            data["fp32_unrounded_delta_l2_squared"] += float(measured[1])
            data["fp32_delta_zero_elements"] += int(counters[0])
            for j, c in enumerate(COEFFICIENTS):
                rec = data["coefficients"][str(c)]
                rec["changed_value_elements"] += int(counters[1 + 3 * j])
                rec["changed_byte_elements"] += int(counters[2 + 3 * j])
                rec["signed_zero_only_changes"] += int(counters[3 + 3 * j])
                rec["realized_delta_l2_squared"] += float(measured[2 + 3 * j])
                rec["dot_with_full_fp32_delta"] += float(measured[3 + 3 * j])
                rec["maximum_absolute_delta"] = max(
                    rec["maximum_absolute_delta"], float(measured[4 + 3 * j])
                )
            base_bytes += raw.size * 2
            observed_elements += raw.size
        base_map._mmap.close()
        del base_map
        completed.append(data)
        if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(pairs):
            elapsed = previous_seconds + time.monotonic() - start
            status = {
                "phase": "realized_bf16_block_audit",
                "completed_matrices": ordinal + 1,
                "total_matrices": len(pairs),
                "observed_matrix_elements": observed_elements,
                "base_bytes_read": base_bytes,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (time.monotonic() - start)
                * (len(pairs) - ordinal - 1)
                / max(ordinal + 1 - already, 1),
                "cpu_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "gpu_used": False,
            }
            save(OUT / "realized-progress.json", status)
            save(OUT / "realized-matrix-details.json", completed)
            print(json.dumps(status), flush=True)
    total_count = sum(math.prod(x["shape"]) for x in completed)
    base_ss = sum(x["base_l2_squared"] for x in completed)
    unrounded_ss = sum(x["fp32_unrounded_delta_l2_squared"] for x in completed)
    summary = []
    for c in COEFFICIENTS:
        items = [x["coefficients"][str(c)] for x in completed]
        actual_ss = sum(x["realized_delta_l2_squared"] for x in items)
        cross = sum(x["dot_with_full_fp32_delta"] for x in items)
        error_ss = max(0.0, actual_ss + c * c * unrounded_ss - 2 * c * cross)
        row = {
            "coefficient": c,
            "changed_value_elements": sum(x["changed_value_elements"] for x in items),
            "changed_byte_elements": sum(x["changed_byte_elements"] for x in items),
            "signed_zero_only_changes": sum(
                x["signed_zero_only_changes"] for x in items
            ),
            "realized_delta_l2": math.sqrt(actual_ss),
            "relative_global_l2": math.sqrt(actual_ss) / global_norm,
            "relative_text_projection_l2": math.sqrt(actual_ss / base_ss),
            "unrounded_fp32_delta_l2": c * math.sqrt(unrounded_ss),
            "rounding_error_l2": math.sqrt(error_ss),
            "direction_cosine_with_unrounded": cross
            / math.sqrt(actual_ss * unrounded_ss)
            if actual_ss
            else None,
            "maximum_absolute_delta": max(x["maximum_absolute_delta"] for x in items),
        }
        row["changed_fraction_of_text_projection_elements"] = (
            row["changed_value_elements"] / total_count
        )
        groups = {}
        for key in ("layer", "mechanism"):
            table = {}
            for x in completed:
                group = str(x[key])
                part = x["coefficients"][str(c)]
                acc = table.setdefault(
                    group,
                    {
                        "elements": 0,
                        "changed": 0,
                        "delta_l2_squared": 0.0,
                        "base_l2_squared": 0.0,
                    },
                )
                acc["elements"] += math.prod(x["shape"])
                acc["changed"] += part["changed_value_elements"]
                acc["delta_l2_squared"] += part["realized_delta_l2_squared"]
                acc["base_l2_squared"] += x["base_l2_squared"]
            for x in table.values():
                x["delta_l2"] = math.sqrt(x.pop("delta_l2_squared"))
                x["changed_fraction"] = x["changed"] / x["elements"]
                x["relative_to_same_projection_base_l2"] = x["delta_l2"] / math.sqrt(
                    x.pop("base_l2_squared")
                )
            groups[key] = table
        row["groups"] = groups
        summary.append(row)
    for name, stats in source_stats.items():
        p = BASE / name
        if {
            "size": p.stat().st_size,
            "inode": p.stat().st_ino,
            "mtime_ns": p.stat().st_mtime_ns,
        } != stats:
            raise ValueError("base source file state changed")
    from dense_lora_recipe import digest

    if digest(PAYLOAD) != EXPECTED:
        raise ValueError("donor checksum changed")
    report = {
        "status": "completed",
        "matrices": len(completed),
        "layers": 60,
        "base_matrix_elements": total_count,
        "source_fp32_task_vector_exact_zero_elements": sum(
            x["fp32_delta_zero_elements"] for x in completed
        ),
        "text_projection_base_l2": math.sqrt(base_ss),
        "fp32_full_scale_delta_l2": math.sqrt(unrounded_ss),
        "coefficients": summary,
        "base_bytes_read": base_bytes,
        "elapsed_seconds": previous_seconds + time.monotonic() - start,
        "resumed_completed_matrices": already,
        "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "block_rows": BLOCK_ROWS,
        "kernel_validation": "Run tests/test_gemma4_lora_recipes.py against the explicitly built library",
        "fp_contract": "off",
        "fast_math": False,
        "arithmetic": "Float32 np.matmul B@A, multiply by 2 then coefficient, add original BF16 W converted to FP32, round once ties-to-even BF16. No checkpoint is written.",
        "source_base_file_state_unchanged": True,
        "donor_sha256": EXPECTED,
        "gpu_used": False,
        "limits": [
            "Purpose benefit is unmeasured; mechanism distribution describes weight changes, not coding localization.",
            "BLAS FP32 accumulation is implementation-dependent near rounding ties; an editor must pin numerical recipe/backend or define explicit rank-order accumulation.",
            "Training provenance and evaluation overlap are not established by this numerical audit.",
            "Signed-zero-only byte changes are reported separately from nonzero numeric changes.",
        ],
    }
    save(OUT / "realized-summary.json", report)
    print(
        json.dumps(
            {
                "status": "completed",
                "elapsed_seconds": report["elapsed_seconds"],
                "coefficients": [
                    {k: v for k, v in x.items() if k != "groups"} for x in summary
                ],
                "peak_rss_KiB": report["peak_rss_KiB"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
