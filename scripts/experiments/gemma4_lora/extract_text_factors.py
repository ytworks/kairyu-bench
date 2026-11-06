#!/usr/bin/env python3
"""Lossless, explicitly selected Gemma text LoRA factor export; preview by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

from kairyu_bench.wsp.weights import checkpoint
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

CHUNK = 8 * 1024**2
KIND = "gemma4_text_factor_selection_v1"
TEXT = re.compile(r"model\.language_model\.layers\.(\d+)\.(mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight")
VISION = re.compile(r"model\.vision_tower\.encoder\.layers\.\d+\.(mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.linear\.weight")


def stamp(path):
    s = path.stat()
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular nonsymlink input required")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key")
        result[k] = v
    return result


def capture(record, pins, *, json_data=False):
    raw = Path(record["path"]).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError("input symlink refused")
    path = raw.resolve()
    before = stamp(path)
    if json_data:
        if before[2] > 16 * 1024**2:
            raise ValueError("JSON input exceeds limit")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        value = json.loads(data, object_pairs_hook=unique_object)
    else:
        digest, value = sha(path), None
    if stamp(path) != before or digest != record["sha256"]:
        raise ValueError("pinned input SHA/stat mismatch")
    pins[path] = (before, digest)
    return path, value


def raw_header(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= size <= 16 * 1024**2:
            raise ValueError("header outside bounded contract")
        raw = f.read(size)
    if len(raw) != size:
        raise ValueError("short header")
    return raw


def ordinary_config(config):
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if (type(rank) is not int or not 1 <= rank <= 256
            or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("uniform ordinary rank/alpha required")
    required = {"peft_type": "LORA", "bias": "none", "fan_in_fan_out": False,
                "use_dora": False, "use_rslora": False}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("ordinary LoRA configuration required")
    unsupported = ("rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias",
                   "use_qalora", "use_bdlora", "alora_invocation_tokens",
                   "layer_replication", "target_parameters", "trainable_token_indices",
                   "arrow_config", "corda_config", "loftq_config", "lora_ga_config",
                   "megatron_config", "monteclora_config", "velora_config", "ensure_weight_tying")
    if any(config.get(k) for k in unsupported):
        raise ValueError("unsupported LoRA modification")
    return rank


def prepare(plan_path, plan_sha):
    pins = {}
    plan_file, plan = capture({"path": str(plan_path), "sha256": plan_sha}, pins, json_data=True)
    if plan.get("schema_version") != 1 or plan.get("kind") != KIND:
        raise ValueError("unsupported selection plan")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", plan["repo"]) or not re.fullmatch("[0-9a-f]{40}", plan["revision"]):
        raise ValueError("public repository/revision provenance required")
    adapter, _ = capture(plan["adapter"], pins)
    config_path, config = capture(plan["config"], pins, json_data=True)
    rank = ordinary_config(config)
    reader = Checkpoint(adapter)
    header_raw = raw_header(adapter)
    if hashlib.sha256(header_raw).hexdigest() != plan["adapter"]["header_sha256"]:
        raise ValueError("header SHA mismatch")
    header = json.loads(header_raw, object_pairs_hook=unique_object)
    base_path = Path(plan["base"]).expanduser().absolute()
    if base_path.is_symlink():
        raise ValueError("base symlink refused")
    base = Checkpoint(base_path)
    selected, excluded = plan["selected"], plan["excluded"]
    for names in (selected, excluded):
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError("explicit unique selected/excluded names required")
    if set(selected) & set(excluded) or set(selected) | set(excluded) != set(reader.tensors):
        raise ValueError("selection must partition every input factor; no silent drops")
    if len(selected) != 820:
        raise ValueError("exactly 820 text factors required")
    dtype = {t.dtype for t in reader.tensors.values()}
    if len(dtype) != 1 or not dtype <= {"F32", "BF16"}:
        raise ValueError("uniform F32/BF16 factors required")
    mapped = []
    for names, pattern in ((selected, TEXT), (excluded, VISION)):
        group = set(names)
        for name in sorted(group):
            match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", name)
            if not match or not pattern.fullmatch(match[1] + ".weight"):
                raise ValueError("unrecognized text/vision factor")
            target, side = match[1] + ".weight", match[2]
            other = name.replace(".lora_" + side + ".", ".lora_" + ("B" if side == "A" else "A") + ".")
            if other not in group or target not in base.tensors:
                raise ValueError("unpaired factor or missing base tensor")
            t = base.tensors[target]
            if t.dtype != "BF16" or len(t.shape) != 2:
                raise ValueError("base projection must be rank-2 BF16")
            expected = (rank, t.shape[1]) if side == "A" else (t.shape[0], rank)
            if reader.tensors[name].shape != expected:
                raise ValueError("factor/base shape mismatch")
            if side == "A" and names is selected:
                mapped.append(target)
    expected_text = {name for name in base.tensors if TEXT.fullmatch(name)}
    if set(mapped) != expected_text or len(mapped) != 410 or {
        int(TEXT.fullmatch(name)[1]) for name in mapped
    } != set(range(60)):
        raise ValueError("exact base text410 across 60 layers required")
    output_raw = Path(plan["output"]).expanduser().absolute()
    if output_raw.is_symlink() or os.path.lexists(output_raw):
        raise ValueError("fresh output directory required")
    output = output_raw.resolve()
    if not output.parent.is_dir() or output.is_relative_to(base.root) or any(
        p == output or p.is_relative_to(output) for p in pins
    ):
        raise ValueError("output overlaps an input or parent is absent")
    for path in (Path(__file__).resolve(), Path(checkpoint.__file__).resolve()):
        pins[path] = (stamp(path), sha(path))
    base_headers = []
    for path in sorted({t.shard for t in base.tensors.values()}):
        base_headers.append({"path": str(path), "stat": stamp(path),
                             "header_sha256": hashlib.sha256(raw_header(path)).hexdigest()})
    return {"plan": plan, "plan_file": plan_file, "plan_sha": plan_sha, "pins": pins,
            "adapter": adapter, "config_path": config_path, "reader": reader, "base": base,
            "header": header, "base_headers": base_headers, "output": output,
            "rank": rank, "dtype": next(iter(dtype))}


def stable(p):
    p["reader"].assert_stable()
    p["base"].assert_stable()
    for path, (state, _) in p["pins"].items():
        if stamp(path) != state:
            raise ValueError("input changed during extraction")


def factor_records(reader, names):
    return [{"name": n, "dtype": reader.tensors[n].dtype,
             "shape": list(reader.tensors[n].shape), "offset": reader.tensors[n].offset,
             "bytes": reader.tensors[n].nbytes} for n in sorted(names)]


def write_json_exclusive(path, value, published):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".extract-", delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        published[path] = (temporary.stat().st_dev, temporary.stat().st_ino)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract(plan_path, plan_sha, *, apply=False):
    p = prepare(plan_path, plan_sha)
    plan, reader, output = p["plan"], p["reader"], p["output"]
    selected = factor_records(reader, plan["selected"])
    excluded = factor_records(reader, plan["excluded"])
    summary = {"selected_factors": len(selected), "excluded_factors": len(excluded),
               "text_matrices": 410, "dtype": p["dtype"], "rank": p["rank"],
               "payload_bytes": sum(r["bytes"] for r in selected), "checkpoint_writes": 0,
               "model_calls": 0, "floating_point_operations": 0}
    stable(p)
    if not apply:
        return {"status": "preview", **summary, "output_writes": 0, "input_full_sha_verified": True}
    output.mkdir(mode=0o700)
    root_id = (output.stat().st_dev, output.stat().st_ino)
    owned = {}
    published = {}
    try:
        header = {"__metadata__": dict(p["header"].get("__metadata__", {}))}
        cursor = 0
        for row in selected:
            header[row["name"]] = {"dtype": row["dtype"], "shape": row["shape"],
                                   "data_offsets": [cursor, cursor + row["bytes"]]}
            cursor += row["bytes"]
        raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        dest = output / "adapter_model.safetensors"
        with dest.open("xb") as out:
            owned[dest] = (dest.stat().st_dev, dest.stat().st_ino)
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for row in selected:
                h = hashlib.sha256()
                item = reader.tensors[row["name"]]
                width = 4 if item.dtype == "F32" else 2
                for first in range(0, item.numel, CHUNK // width):
                    block = reader.read_raw(item.name, first, min(CHUNK // width, item.numel-first))
                    out.write(block)
                    h.update(block)
                row["source_raw_sha256"] = h.hexdigest()
            out.flush()
            os.fsync(out.fileno())
        derived = Checkpoint(dest)
        for row in selected:
            item = derived.tensors[row["name"]]
            h = hashlib.sha256()
            width = 4 if item.dtype == "F32" else 2
            for first in range(0, item.numel, CHUNK // width):
                h.update(derived.read_raw(item.name, first, min(CHUNK // width, item.numel-first)))
            if h.hexdigest() != row["source_raw_sha256"]:
                raise ValueError("derived selected factor bytes differ")
            row["derived_raw_sha256"] = h.hexdigest()
            row["derived_offset"] = item.offset
        config_dest = output / "adapter_config.json"
        with config_dest.open("xb") as out:
            owned[config_dest] = (config_dest.stat().st_dev, config_dest.stat().st_ino)
            out.write(p["config_path"].read_bytes())
            out.flush()
            os.fsync(out.fileno())
        if sha(config_dest) != plan["config"]["sha256"]:
            raise ValueError("config copy changed")
        stable(p)
        receipt = {
            "schema_version": 1, "status": "verified_lossless_text_factor_extraction", **summary,
            "plan_path": str(p["plan_file"]), "plan_sha256": plan_sha,
            "original": {"repo": plan["repo"], "revision": plan["revision"],
                         "adapter": plan["adapter"], "config": plan["config"]},
            "derived": {"adapter_path": str(dest), "sha256": sha(dest),
                        "header_sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": dest.stat().st_size, "config_path": str(config_dest),
                        "config_sha256": sha(config_dest)},
            "selected": selected, "excluded": excluded,
            "input_pins": {str(path): {"stat": state, "sha256": digest}
                           for path, (state, digest) in p["pins"].items()},
            "base_layout": {"path": str(p["base"].path), "headers": p["base_headers"],
                            "full_weight_identity_verified": False,
                            "index": None if p["base"].index is None else
                            {"path": str(p["base"].index), "sha256": sha(p["base"].index)}},
            "limitations": [
                "This is a derived text-only payload, not the original repository payload.",
                "Only factor bytes and base tensor layouts are verified; training-base identity and capability transfer are not established.",
                "No numeric factor validation, merge, BF16 rounding, model execution or benchmark is performed."
            ],
        }
        stable(p)
        write_json_exclusive(output / "extraction.json", receipt, published)
        return {"status": "completed", **summary, "receipt": str(output / "extraction.json"),
                "derived_sha256": receipt["derived"]["sha256"]}
    finally:
        receipt_path = output / "extraction.json"
        committed = (receipt_path in published and receipt_path.exists() and not receipt_path.is_symlink()
                     and (receipt_path.stat().st_dev, receipt_path.stat().st_ino) == published[receipt_path])
        if not committed and output.exists() and (output.stat().st_dev, output.stat().st_ino) == root_id:
            for path, identity in owned.items():
                if path.exists() and not path.is_symlink() and (path.stat().st_dev, path.stat().st_ino) == identity:
                    path.unlink()
            try:
                output.rmdir()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(json.dumps(extract(args.plan, args.plan_sha256, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
