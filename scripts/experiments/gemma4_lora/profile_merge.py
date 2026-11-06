#!/usr/bin/env python3
"""Bounded CPU-only timing of the existing deterministic module merge paths.

No model, checkpoint write, new kernel, or thread-policy change is involved.
An owner-approved optional scratch-GPU transfer probe does not load model weights.
Use the same pinned image/kernel/BLAS environment as RAM and persistent recipes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

import bf16_lora_ops as ops
import module_lora_recipe as module
from kairyu_bench.wsp.weights.checkpoint import Checkpoint


DEFAULT_TENSORS = (
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
)


def tiled(original, delta, coefficient, path):
    """Preserve 128-row blocks and compare wrapper/kernel work independently."""
    target = np.empty_like(original)
    squares, counts, peak = [], np.zeros(2, np.uint64), 0.
    stats, count = np.zeros(3, np.float64), np.zeros(2, np.uint64)
    started = time.monotonic()
    for first in range(0, len(original), module.BLOCK_ROWS):
        stop = min(len(original), first + module.BLOCK_ROWS)
        before, change = original[first:stop], delta[first:stop]
        if path == "module":
            result, stats, count = module.merge_block(before, change, coefficient)
            target[first:stop] = result
        elif path == "wrapper":
            result, stats, count = module.dense.merge1(before, change, coefficient)
            target[first:stop] = result
        elif path == "kernel":
            result = target[first:stop]
            bad = ops.LIB.merge1(before.ctypes.data_as(ops.P16), change.ctypes.data_as(ops.PF),
                                before.size, float(coefficient), result.ctypes.data_as(ops.P16),
                                stats.ctypes.data_as(ops.PD), count.ctypes.data_as(ops.P64))
            if bad:
                raise ValueError("nonfinite merge kernel")
        else:
            raise ValueError("unknown profile path")
        squares.append(float(stats[1]))
        counts += count
        peak = max(peak, float(stats[2]))
    elapsed = time.monotonic() - started
    hash_started = time.monotonic()
    digest = hashlib.sha256(memoryview(target).cast("B")).hexdigest()
    return target, {"path": path, "seconds": elapsed,
                    "target_sha256": digest, "hash_seconds": time.monotonic() - hash_started,
                    "changed_elements": int(counts[0]), "changed_byte_elements": int(counts[1]),
                    "delta_l2": math.sqrt(math.fsum(squares)), "maximum_absolute_delta": peak}


def device_transfer_profile(original, bits, device, torch):
    """Scratch tensors only: compare roundtrip with staged int16 equality."""
    if original.shape != bits.shape or original.dtype != np.uint16 or bits.dtype != np.uint16:
        raise ValueError("BF16 source/target storage arrays required")
    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    def restore_scratch():
        target.copy_(torch.from_numpy(np.array(original, copy=True)).view(torch.bfloat16))
        synchronize()
    def observed_bits():
        return target.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)

    target = torch.empty(bits.shape, dtype=torch.bfloat16, device=device)
    expected_sha = hashlib.sha256(memoryview(bits).cast("B")).hexdigest()
    with torch.no_grad():
        restore_scratch()
        started = time.monotonic()
        host = torch.from_numpy(np.array(bits, copy=True)).view(torch.bfloat16)
        target.copy_(host, non_blocking=False)
        observed = observed_bits()
        if not np.array_equal(observed, bits):
            raise ValueError("roundtrip target bytes differ")
        roundtrip_sha = hashlib.sha256(observed.tobytes()).hexdigest()
        synchronize()
        roundtrip_seconds = time.monotonic() - started
        del host, observed
        restore_scratch()
        started = time.monotonic()
        staging = torch.from_numpy(np.array(bits, copy=True)).view(torch.bfloat16).to(device=device)
        target.copy_(staging, non_blocking=False)
        exact = bool(torch.equal(target.view(torch.int16), staging.view(torch.int16)))
        synchronize()
        staged_seconds = time.monotonic() - started
        if not exact:
            raise ValueError("device int16 equality failed")
        # One independent oracle AFTER the alternative timed path. This is
        # diagnostic validation, not a claim that device equality hashes a D2H copy.
        oracle_started = time.monotonic()
        observed = observed_bits()
        oracle_sha = hashlib.sha256(observed.tobytes()).hexdigest()
        if not np.array_equal(observed, bits) or not (expected_sha == roundtrip_sha == oracle_sha):
            raise ValueError("final CPU byte oracle differs")
        oracle_seconds = time.monotonic() - oracle_started
    return {"device": str(device), "scratch_only": True, "model_loads": 0,
            "target_cpu_sha256": expected_sha,
            "roundtrip_seconds_including_d2h_equality_hash": roundtrip_seconds,
            "staged_seconds_including_upload_copy_int16_equality": staged_seconds,
            "staged_device_int16_equal": exact, "final_independent_cpu_oracle_sha256": oracle_sha,
            "final_oracle_seconds_excluded_from_staged_timing": oracle_seconds,
            "proof_scope": "CPU target SHA plus all-element device int16 equality; not a D2H hash in the staged path."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--tensor", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=60)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--probe-device", choices=("cuda:0",), help="optional scratch transfer comparison; no model load")
    parser.add_argument("--gpu-probe-approved", action="store_true")
    args = parser.parse_args(argv)
    if args.run and args.probe_device and not args.gpu_probe_approved:
        parser.error("scratch GPU probe requires explicit owner approval")
    if not 1 <= args.maximum_seconds <= 60:
        parser.error("CPU microprofile is limited to 1..60 seconds")
    started = time.monotonic()
    def expired(*unused):
        raise TimeoutError("CPU merge microprofile deadline")
    base, spec_path, output = args.checkpoint.resolve(), args.spec.resolve(), args.output.resolve()
    if args.output.is_symlink() or output.exists() or output.is_relative_to(base) or output == spec_path:
        raise ValueError("output must be fresh and outside input checkpoint/spec")
    checkpoint = Checkpoint(base)
    spec, content, before, recipe, mapping, header, offset = module.prepared(base, spec_path)
    protected = [Path(recipe[k]).resolve() for k in ("donor_path", "donor_config_path", "unit_costs_path")]
    if output in protected:
        raise ValueError("output collides with pinned input")
    chosen = list(DEFAULT_TENSORS) if args.tensor is None else args.tensor
    by_name = {row["tensor"]: row for row in mapping}
    if not 1 <= len(chosen) <= 2 or len(set(chosen)) != len(chosen) or any(name not in by_name for name in chosen):
        raise ValueError("choose one or two existing text projection matrices")
    if any(math.prod(by_name[name]["shape"]) > 128_000_000 for name in chosen):
        raise ValueError("profile matrix exceeds bounded workspace")
    if any(recipe["module_coefficients"][name] == 0 for name in chosen):
        raise ValueError("choose nonzero coefficients for comparable module/kernel paths")
    report = {"status": "preview", "source": str(base), "spec": str(spec_path),
              "spec_sha256": hashlib.sha256(content).hexdigest(),
              "tensors": chosen, "recipe": recipe, "profile_source_sha256": module.dense.digest(Path(__file__)),
              "numpy_version": np.__version__, "maximum_seconds": args.maximum_seconds,
              "model_loads": 0, "gpu_probe_requested": args.probe_device is not None, "checkpoint_writes": 0,
              "block_rows": module.BLOCK_ROWS, "kernel_openmp_threads": 4, "results": []}
    if not args.run:
        print(json.dumps(report, indent=2))
        return 0
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, args.maximum_seconds - math.ceil(time.monotonic() - started)))
    try:
        torch = None
        if args.probe_device:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 4 * 2**30:
                raise ValueError("scratch probe requires one exposed GPU with 4 GiB free")
        for name in chosen:
            item = by_name[name]
            rows, columns = item["shape"]
            source_started = time.monotonic()
            original = np.empty((rows, columns), dtype=np.uint16)
            source_hash = hashlib.sha256()
            for first in range(0, rows, module.BLOCK_ROWS):
                stop = min(rows, first + module.BLOCK_ROWS)
                raw = checkpoint.read_raw(name, first * columns, (stop - first) * columns)
                original[first:stop] = np.frombuffer(raw, "<u2").reshape(stop - first, columns)
                source_hash.update(raw)
            source_seconds = time.monotonic() - source_started
            donor = Path(recipe["donor_path"])
            a = module.dense.tensor(donor, header, offset, item["lora_A"])
            b = module.dense.tensor(donor, header, offset, item["lora_B"])
            delta = np.empty((rows, columns), dtype=np.float32)
            delta_started = time.monotonic()
            for first in range(0, rows, module.BLOCK_ROWS):
                stop = min(rows, first + module.BLOCK_ROWS)
                delta[first:stop] = module.delta_block(a, b[first:stop])
            delta_seconds = time.monotonic() - delta_started
            value = recipe["module_coefficients"][name]
            measurements, expected = [], None
            for path in ("module", "wrapper", "kernel"):
                target, measured = tiled(original, delta, value, path)
                if expected is None:
                    expected = dict(measured)
                elif (measured["target_sha256"] != expected["target_sha256"]
                      or measured["changed_elements"] != expected["changed_elements"]
                      or measured["changed_byte_elements"] != expected["changed_byte_elements"]
                      or not math.isclose(measured["delta_l2"], expected["delta_l2"], rel_tol=1e-12, abs_tol=1e-15)):
                    raise ValueError("profile paths differ in BF16 bytes/counts/statistics")
                measurements.append(measured)
                if path != "kernel" or not args.probe_device:
                    del target
            transfer = None
            if args.probe_device:
                transfer = device_transfer_profile(original, target, args.probe_device, torch)
                del target
            copy_started = time.monotonic()
            copied = original.copy()
            copy_seconds = time.monotonic() - copy_started
            del copied
            report["results"].append({"tensor": name, "shape": [rows, columns],
                                      "coefficient": value, "source_tensor_sha256": source_hash.hexdigest(),
                                      "source_read_seconds": source_seconds, "cached_delta_seconds": delta_seconds,
                                      "host_matrix_copy_seconds": copy_seconds, "measurements": measurements,
                                      "all_paths_byte_count_stat_agreement": True, "device_transfer_probe": transfer})
            del original, delta, a, b
        checkpoint.assert_stable()
        if module.dense.digest(Path(__file__)) != report["profile_source_sha256"]:
            raise ValueError("profile source changed")
        module.final_inputs(base, spec_path, spec, content, before, recipe, mapping)
        report.update(status="completed", source_stable=True)
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        report["elapsed_seconds"] = time.monotonic() - started
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "matrices": len(report["results"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
