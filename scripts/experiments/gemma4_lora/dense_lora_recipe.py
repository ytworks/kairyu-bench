#!/usr/bin/env python3
"""Schema-2 whole-text LoRA recipe materialization and exact streaming verification."""

from pathlib import Path
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import numpy as np
from bf16_lora_ops import LIBRARY, merge1

KIND = "dense_lora_recipe_v1"
BLOCK_ROWS = 128
RUNTIME = Path(__file__).resolve().parent
SCALE = 2.0


def read(p):
    return json.loads(Path(p).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonhash(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save(path, x, commit_identity):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(x, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        stat = temporary.stat()
        commit_identity.append((stat.st_dev, stat.st_ino))
        os.link(
            temporary, path
        )  # Exclusive atomic publication: never replace another writer.
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def files(root):
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if ".cache" in rel.parts or rel.name == ".kairyu-model-attestation.json":
            continue
        if p.is_symlink():
            raise ValueError("symlink forbidden in recipe source/candidate")
        if p.is_dir():
            continue
        if not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError("nonregular model file")
        out.append(p)
    return out


def tree(root):
    rows = [
        {
            "path": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in files(root)
    ]
    return rows, jsonhash(rows)


def header(path):
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors prefix")
        n = struct.unpack("<Q", raw)[0]
        if n > 10_000_000:
            raise ValueError("unbounded safetensors header")
        payload = f.read(n)
        if len(payload) != n:
            raise ValueError("short safetensors header")
        h = json.loads(payload)
    return h, 8 + n


def tensor(path, h, offset, name):
    item = h[name]
    a, b = item["data_offsets"]
    width = {"BF16": 2, "F32": 4}.get(item["dtype"])
    if width is None or b - a != width * math.prod(item["shape"]):
        raise ValueError("BF16/F32 tensor size mismatch")
    with path.open("rb") as f:
        f.seek(offset + a)
        buf = f.read(b - a)
    if len(buf) != b - a:
        raise ValueError("short tensor")
    if item["dtype"] == "BF16":
        value = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        value = np.frombuffer(buf, dtype="<f4")  # Preserve exported F32 factors; no BF16 conversion.
    value = value.reshape(item["shape"])
    if not np.isfinite(value).all():
        raise ValueError("nonfinite adapter tensor")
    return value


def implementation():
    names = ["dense_lora_recipe.py", "bf16_lora_ops.py", "bf16_lora_kernel.cpp"]
    code = {n: digest(RUNTIME / n) for n in names}
    code["compiled_kernel"] = digest(LIBRARY)
    build = json.loads(json.dumps(np.__config__.CONFIG, default=str))
    return {
        "implementation_sha256": jsonhash(code),
        "implementation_files": code,
        "numpy_version": np.__version__,
        "numpy_build_sha256": jsonhash(build),
    }


def local_training_provenance(base, spec, *, extended):
    """Explicit local receipt alternative; never invent a public donor identity."""
    pin = spec.get("local_training")
    if pin is None:
        if not re.fullmatch("[0-9a-f]{40}", spec["donor_revision"]) or not re.fullmatch(
            r"[^/\s]+/[^/\s]+", spec["donor_repo"]
        ):
            raise ValueError("donor public provenance")
        return {"donor_repo": spec["donor_repo"], "donor_revision": spec["donor_revision"]}
    if (not extended or not isinstance(pin, dict) or set(pin) != {"version", "path", "sha256"}
            or pin["version"] != 1 or "donor_repo" in spec or "donor_revision" in spec):
        raise ValueError("local training requires exclusive explicit factor-format-v2 provenance")
    path = Path(pin["path"]).resolve()
    if path.is_relative_to(base) or not re.fullmatch("[0-9a-f]{64}", pin["sha256"]) or digest(path) != pin["sha256"]:
        raise ValueError("local training receipt path/hash")
    receipt = read(path)
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != "local_lora_training_v1"
            or receipt.get("status") != "exported" or receipt.get("base_unchanged") is not True
            or type(receipt.get("steps")) is not int or receipt["steps"] < 1
            or Path(receipt.get("parent_path", "")).resolve() != base
            or receipt.get("parent_dense_tree_sha256") != spec["base_tree_sha256"]
            or receipt.get("adapter_sha256") != spec["donor_sha256"]
            or receipt.get("config_sha256") != spec["donor_config_sha256"]
            or receipt.get("modules") != 410
            or receipt.get("rank") != spec["factor_format"]["rank"]
            or receipt.get("dtype") != spec["factor_format"]["dtype"]
            or not re.fullmatch("[0-9a-f]{64}", str(receipt.get("data_sha256", "")))
            or not isinstance(receipt.get("source_sha256"), dict) or not receipt["source_sha256"]
            or any(not re.fullmatch("[0-9a-f]{64}", str(v)) for v in receipt["source_sha256"].values())):
        raise ValueError("local training receipt does not bind these exact parent/factors/data")
    return {"local_training": dict(pin, path=str(path)), "local_training_receipt": receipt}


def validated_recipe(base, spec, *, allow_factor_format=False):
    if (
        spec.get("schema_version") != 2
        or spec.get("kind") != KIND
        or spec.get("scope") != "native_text410"
    ):
        raise ValueError("unsupported recipe kind/scope")
    factor_format = spec.get("factor_format")
    extended = "factor_format" in spec
    if extended:
        if not allow_factor_format:
            raise ValueError("factor_format requires explicit dense-only opt-in")
        if (not isinstance(factor_format, dict)
                or set(factor_format) != {"version", "rank", "dtype", "header_sha256"}
                or type(factor_format["version"]) is not int or factor_format["version"] != 2
                or type(factor_format["rank"]) is not int or not 1 <= factor_format["rank"] <= 256
                or factor_format["dtype"] not in {"BF16", "F32"}
                or not isinstance(factor_format["header_sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", factor_format["header_sha256"])
                or spec.get("native_edits") or spec.get("companion_required")):
            raise ValueError("invalid or unsupported dense-only factor_format v2")
    structured = "structured_scope" in spec or (extended and "module_coefficients" in spec)
    if structured and (not extended or not allow_factor_format):
        raise ValueError("structured scope requires standalone factor_format v2")
    rank = factor_format["rank"] if extended else 16
    factor_dtype = factor_format["dtype"] if extended else "BF16"
    donor = Path(spec["donor_path"]).resolve()
    config_path = Path(spec["donor_config_path"]).resolve()
    if donor.is_relative_to(base) or config_path.is_relative_to(base):
        raise ValueError("donor must be outside base checkpoint")
    for path, key in ((donor, "donor_sha256"), (config_path, "donor_config_sha256")):
        if not re.fullmatch("[0-9a-f]{64}", spec[key]) or digest(path) != spec[key]:
            raise ValueError("pinned donor/config SHA mismatch")
    provenance = local_training_provenance(base, spec, extended=extended)
    c = float(spec["coefficient"])
    if not math.isfinite(c) or not 0 < c <= 1 or float(np.float32(c)) != c:
        raise ValueError("coefficient must be exactly representable FP32 in (0,1]")
    config = read(config_path)
    for key, value in {
        "r": rank,
        **({} if extended else {"lora_alpha": 32}),
        "use_rslora": False,
        "use_dora": False,
        "fan_in_fan_out": False,
        "bias": "none",
    }.items():
        if config.get(key) != value:
            raise ValueError("ordinary LoRA configuration differs")
    if (
        config.get("rank_pattern")
        or config.get("alpha_pattern")
        or config.get("modules_to_save")
        or config.get("lora_bias")
    ):
        raise ValueError("unsupported adapter modification")
    alpha = config.get("lora_alpha")
    scale_value = SCALE
    if extended:
        unsupported = ("use_qalora", "use_bdlora", "alora_invocation_tokens", "layer_replication",
                       "target_parameters", "trainable_token_indices", "arrow_config", "corda_config",
                       "loftq_config", "lora_ga_config", "megatron_config", "monteclora_config",
                       "velora_config", "ensure_weight_tying")
        if (config.get("peft_type") != "LORA" or type(config.get("r")) is not int
                or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0
                or any(config.get(key) for key in unsupported)):
            raise ValueError("unsupported ordinary LoRA factor configuration")
        with np.errstate(over="ignore", under="ignore"):
            scale_value = float(np.float32(alpha / rank))
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise ValueError("alpha/r must give a finite positive FP32 scale")
    h, off = header(donor)
    if extended:
        with donor.open("rb") as stream:
            stream.seek(8)
            actual_header_sha = hashlib.sha256(stream.read(off - 8)).hexdigest()
        if actual_header_sha != factor_format["header_sha256"]:
            raise ValueError("adapter header SHA mismatch")
        intervals = []
        for name, value in h.items():
            if name == "__metadata__":
                continue
            shape = value["shape"]
            lo, hi = value["data_offsets"]
            if (value["dtype"] != factor_dtype or len(shape) != 2
                    or any(type(v) is not int or v <= 0 for v in shape)
                    or type(lo) is not int or type(hi) is not int or lo < 0
                    or hi-lo != {"BF16": 2, "F32": 4}[factor_dtype] * math.prod(shape)):
                raise ValueError("adapter factor extent/dtype mismatch")
            intervals.append((lo, hi))
        previous = 0
        for lo, hi in sorted(intervals):
            if lo != previous:
                raise ValueError("adapter factor spans overlap or leave undeclared bytes")
            previous = hi
        if off + previous != donor.stat().st_size:
            raise ValueError("adapter payload size mismatch")
    index = read(base / "model.safetensors.index.json")["weight_map"]
    for name in set(index.values()):
        rel = Path(name)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or not (base / rel).resolve().is_relative_to(base)
            or (base / rel).is_symlink()
        ):
            raise ValueError("shard escapes checkpoint or is symlink")
    bh = {name: header(base / name) for name in set(index.values())}
    mapping = []
    seen = set()
    prefix = "base_model.model.model.language_model."
    for name in sorted(
        n for n in h if n.startswith(prefix) and n.endswith(".lora_A.weight")
    ):
        bname = name.replace(".lora_A.weight", ".lora_B.weight")
        target = name.removeprefix("base_model.model.").replace(
            ".lora_A.weight", ".weight"
        )
        match = re.fullmatch(
            r"model\.language_model\.layers\.(\d+)\.(mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight",
            target,
        )
        if not match or bname not in h or target not in index or target in seen:
            raise ValueError("invalid/duplicate text mapping")
        seen.add(target)
        shard = index[target]
        th, to = bh[shard]
        entry = th[target]
        shape = entry["shape"]
        if (
            entry["dtype"] != "BF16"
            or len(shape) != 2
            or h[name]["shape"] != [rank, shape[1]]
            or h[bname]["shape"] != [shape[0], rank]
        ):
            raise ValueError("rank/shape/dtype mapping")
        if any(type(v) is not int or v <= 0 for v in shape) or entry["data_offsets"][
            1
        ] - entry["data_offsets"][0] != 2 * math.prod(shape):
            raise ValueError("base tensor extent mismatch")
        if h[name]["dtype"] != factor_dtype or h[bname]["dtype"] != factor_dtype:
            raise ValueError("adapter dtype")
        mapping.append(
            {
                "tensor": target,
                "shard": shard,
                "offset": to + entry["data_offsets"][0],
                "size": 2 * math.prod(shape),
                "dtype": "BF16",
                "shape": shape,
                "lora_A": name,
                "lora_B": bname,
            }
        )
    if len(mapping) != 410 or {
        int(x["tensor"].split(".layers.")[1].split(".")[0]) for x in mapping
    } != set(range(60)):
        raise ValueError("requires exactly whole-text 410 matrices across 60 layers")
    expected = {
        n
        for n in index
        if re.fullmatch(
            r"model\.language_model\.layers\.\d+\.(mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj)\.weight",
            n,
        )
    }
    if seen != expected:
        raise ValueError("recipe does not cover exact base projection set")
    if extended and set(h) - {"__metadata__"} != {
        name for row in mapping for name in (row["lora_A"], row["lora_B"])
    }:
        raise ValueError("factor_format v2 requires exactly 820 text factors; extra tensors refused")
    coefficients = None
    if structured:
        coefficients = spec.get("module_coefficients")
        if (spec.get("structured_scope") != "mlp_triplets_all_layers"
                or not isinstance(coefficients, dict) or set(coefficients) != seen):
            raise ValueError("structured scope requires an explicit exact 410-coefficient map")
        mlp = {f"model.language_model.layers.{layer}.mlp.{role}_proj.weight"
               for layer in range(60) for role in ("gate", "up", "down")}
        if not mlp.issubset(seen) or len(mlp) != 180:
            raise ValueError("structured scope requires every MLP gate/up/down triplet")
        normalized = {}
        for name in sorted(seen):
            value = coefficients[name]
            expected = c if name in mlp else 0.0
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or float(np.float32(value)) != value or value != expected):
                raise ValueError("MLP coefficients must equal the coherent scalar; all attention coefficients must be zero")
            normalized[name] = float(value)
        coefficients = normalized
    spans = {}
    for x in mapping:
        lo = x["offset"]
        hi = lo + x["size"]
        if lo < bh[x["shard"]][1] or hi > (base / x["shard"]).stat().st_size:
            raise ValueError("invalid tensor span")
        spans.setdefault(x["shard"], []).append((lo, hi))
    for intervals in spans.values():
        prev = 0
        for lo, hi in sorted(intervals):
            if lo < prev:
                raise ValueError("overlapping recipe spans")
            prev = hi
    recipe = {
        "version": 1,
        "donor_path": str(donor),
        "donor_config_path": str(config_path),
        "donor_sha256": spec["donor_sha256"],
        "donor_config_sha256": spec["donor_config_sha256"],
        **provenance,
        "coefficient": c,
        "r": rank,
        "alpha": alpha,
        "scale": scale_value,
        "mapping_sha256": jsonhash(mapping),
        "block_rows": BLOCK_ROWS,
        "scope": "native_text410_vision_excluded",
        "merge_arithmetic": "FP32 np.matmul(B_block,A), FP32 multiply by 2, FP32 multiply by coefficient, separate FP32 add to original BF16 W converted to FP32, ties-to-even BF16 once; no FMA contraction in merge loop.",
        "cpu_threads": 4,
        **implementation(),
    }
    if extended:
        recipe.update(
            version=2, factor_format=dict(factor_format), factor_dtype=factor_dtype,
            adapter_header_sha256=actual_header_sha,
            merge_arithmetic="F32 factors preserved, or BF16 factors exactly decoded to F32; FP32 np.matmul(B_block,A), FP32 multiply by FP32(alpha/r), FP32 multiply by coefficient, separate FP32 add to original BF16 W converted to FP32, ties-to-even BF16 once; no FMA contraction in merge loop.",
        )
    if structured:
        recipe.update(
            structured_scope="mlp_triplets_all_layers",
            structured_scope_version=1,
            module_coefficients=coefficients,
            module_coefficients_sha256=jsonhash(coefficients),
            validated_projection_count=len(mapping),
            active_projection_count=180,
            zero_coefficient_projection_count=len(mapping) - 180,
            zero_scope_arithmetic="No GEMM/add/rounding for zero coefficients; all non-active bytes are copied and independently compared.",
        )
    return recipe, mapping, h, off


def active_mapping(recipe, mapping):
    """Keep the complete validated inventory; select only explicit nonzero writes."""
    if "structured_scope" not in recipe:
        return mapping
    return [entry for entry in mapping
            if recipe["module_coefficients"][entry["tensor"]] != 0]


def materialization_recipe(base, spec):
    """Only standalone dense materialization opts into factor v2 and structured scope."""
    if "factor_format" in spec:
        if spec.get("native_edits") or spec.get("companion_required"):
            raise ValueError("factor_format v2 is not supported by native/module/joint wrappers")
        return validated_recipe(base, spec, allow_factor_format=True)
    # Preserve the existing optional-native handler's two-argument contract.
    return validated_recipe(base, spec)


def entries(base, candidate, recipe, mapping, h, off, write=False):
    mapping = active_mapping(recipe, mapping)
    donor = Path(recipe["donor_path"])
    records = []
    with ExitStack() as stack:
        source_files = {
            n: stack.enter_context((base / n).open("rb"))
            for n in {x["shard"] for x in mapping}
        }
        target_files = {
            n: stack.enter_context(
                (candidate / n).open("r+b" if write else "rb", buffering=0)
            )
            for n in source_files
        }
        for name in source_files:
            if (
                os.path.samefile(base / name, candidate / name)
                or (candidate / name).stat().st_nlink != 1
            ):
                raise ValueError("edited shard shares source inode")
        for item in mapping:
            A = tensor(donor, h, off, item["lora_A"])
            B = tensor(donor, h, off, item["lora_B"])
            source = source_files[item["shard"]]
            target = target_files[item["shard"]]
            before_sha = hashlib.sha256()
            after_sha = hashlib.sha256()
            base_ss = delta_ss = 0.0
            changed = bytechanged = 0
            max_delta = 0.0
            rows, cols = item["shape"]
            for first in range(0, rows, BLOCK_ROWS):
                count = min(BLOCK_ROWS, rows - first) * cols
                position = item["offset"] + first * cols * 2
                source.seek(position)
                raw = source.read(count * 2)
                if len(raw) != count * 2:
                    raise ValueError("short source block")
                original = np.frombuffer(raw, dtype="<u2").reshape(-1, cols)
                delta = (B[first : first + len(original)] @ A) * np.float32(recipe["scale"])
                merged, stats, counts = merge1(original, delta, recipe["coefficient"])
                expected = merged.astype("<u2", copy=False).tobytes()
                target.seek(position)
                observed = target.read(count * 2)
                if observed != (raw if write else expected):
                    raise ValueError(
                        "candidate original bytes or regenerated recipe bytes differ"
                    )
                if write:
                    target.seek(position)
                    if target.write(expected) != len(expected):
                        raise ValueError("short recipe write")
                before_sha.update(raw)
                after_sha.update(expected)
                base_ss += float(stats[0])
                delta_ss += float(stats[1])
                max_delta = max(max_delta, float(stats[2]))
                changed += int(counts[0])
                bytechanged += int(counts[1])
            records.append(
                {
                    **item,
                    "entry_kind": "lora_matrix",
                    "source_tensor_sha256": before_sha.hexdigest(),
                    "candidate_tensor_sha256": after_sha.hexdigest(),
                    "changed_count": changed,
                    "changed_byte_count": bytechanged,
                    "source_l2": math.sqrt(base_ss),
                    "delta_l2": math.sqrt(delta_ss),
                    "maximum_absolute_delta": max_delta,
                }
            )
        if write:
            for stream in target_files.values():
                os.fsync(stream.fileno())
    return records


def compare_untouched(base, candidate, mapping):
    source_paths = {str(p.relative_to(base)): p for p in files(base)}
    target_paths = {str(p.relative_to(candidate)): p for p in files(candidate)}
    if source_paths.keys() != target_paths.keys():
        raise ValueError("nonweight file inventory changed")
    spans = {}
    for e in mapping:
        spans.setdefault(e["shard"], []).append((e["offset"], e["offset"] + e["size"]))
    for rel, p in source_paths.items():
        q = target_paths[rel]
        if p.stat().st_size != q.stat().st_size:
            raise ValueError("file size changed")
        if rel not in spans:
            if digest(p) != digest(q):
                raise ValueError("unchanged file differs")
            continue
        cursor = 0
        with p.open("rb") as left, q.open("rb") as right:
            for lo, hi in sorted(spans[rel]) + [(p.stat().st_size, p.stat().st_size)]:
                if lo < cursor:
                    raise ValueError("overlapping full tensor recipe entries")
                remaining = lo - cursor
                left.seek(cursor)
                right.seek(cursor)
                while remaining:
                    n = min(8 << 20, remaining)
                    a = left.read(n)
                    b = right.read(n)
                    if len(a) != n or a != b:
                        raise ValueError("undeclared bytes changed")
                    remaining -= n
                cursor = hi


def metrics(recorded, reconstructed):
    if len(recorded) != len(reconstructed):
        raise ValueError("recipe edit count mismatch")
    for a, b in zip(recorded, reconstructed):
        for key in b:
            if isinstance(b[key], float):
                if not math.isclose(a[key], b[key], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("recorded norm differs")
            elif a.get(key) != b[key]:
                raise ValueError("recorded recipe tensor identity/digest/count differs")


def verification_snapshot(base, candidate, record, manifest):
    paths = [
        *files(base),
        *files(candidate),
        Path(manifest),
        Path(record["spec_path"]),
        Path(record["spec"]["donor_path"]),
        Path(record["spec"]["donor_config_path"]),
        LIBRARY,
        RUNTIME / "dense_lora_recipe.py",
        RUNTIME / "bf16_lora_ops.py",
        RUNTIME / "bf16_lora_kernel.cpp",
    ]
    if "local_training" in record["spec"]:
        paths.append(Path(record["spec"]["local_training"]["path"]))
    if record["spec"].get("native_edits"):
        from kairyu_bench.wsp.weights import checkpoint as checkpoint_reader
        from kairyu_bench.wsp.weights import editor

        paths.append(Path(checkpoint_reader.__file__))
        paths.extend([RUNTIME / "dense_lora_native_recipe.py", Path(editor.__file__)])
    result = {}
    for path in paths:
        resolved = path.resolve()
        st = resolved.stat()
        result[str(path)] = (
            str(resolved),
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )
    return result


def verify(manifest):
    record = read(manifest)
    if record.get("schema_version") != 2 or record.get("kind") != KIND:
        raise ValueError("unknown dense recipe schema")
    base = Path(record["source"]).resolve()
    dest = Path(record["candidate"]).resolve()
    if base == dest or dest.is_relative_to(base) or base.is_relative_to(dest):
        raise ValueError("overlapping model trees")
    before = verification_snapshot(base, dest, record, manifest)
    spec_path = Path(record["spec_path"])
    if digest(spec_path) != record["spec_sha256"] or read(spec_path) != record["spec"]:
        raise ValueError("spec file changed")
    recipe, mapping, h, off = materialization_recipe(base, record["spec"])
    if recipe != record["recipe"]:
        raise ValueError("recipe provenance/implementation changed")
    source_rows, source_hash = tree(base)
    if (
        source_hash != record["source_tree_sha256"]
        or source_hash != record["spec"]["base_tree_sha256"]
    ):
        raise ValueError("source full tree hash mismatch")
    if source_rows != record["source_files"]:
        raise ValueError("source file hash list differs")
    compare_untouched(base, dest, active_mapping(recipe, mapping))
    rebuilt = entries(base, dest, recipe, mapping, h, off, False)
    metrics(record["edits"], rebuilt)
    if record["changed_byte_elements"] != sum(x["changed_byte_count"] for x in rebuilt):
        raise ValueError("top-level changed byte count")
    if record["changed_elements"] != sum(x["changed_count"] for x in rebuilt) or record[
        "changed_tensors"
    ] != sum(x["changed_count"] > 0 for x in rebuilt):
        raise ValueError("top-level changed counts")
    if not math.isclose(
        record["delta_l2"],
        math.sqrt(sum(x["delta_l2"] ** 2 for x in rebuilt)),
        rel_tol=1e-10,
    ):
        raise ValueError("top-level norm")
    target_rows, target_hash = tree(dest)
    if target_rows != record["candidate_files"]:
        raise ValueError("candidate file hash list differs")
    if target_hash != record["candidate_tree_sha256"]:
        raise ValueError("candidate full tree hash mismatch")
    final_recipe, final_mapping, _, _ = materialization_recipe(base, record["spec"])
    if final_recipe != recipe or final_mapping != mapping:
        raise ValueError("recipe provenance changed during verification")
    if digest(spec_path) != record["spec_sha256"] or read(manifest) != record:
        raise ValueError("manifest/spec changed during verification")
    if verification_snapshot(base, dest, record, manifest) != before:
        raise ValueError("source/candidate/provenance changed during verification")
    print(
        json.dumps(
            {
                "verified": True,
                "kind": KIND,
                "changed_elements": record["changed_elements"],
                "delta_l2": record["delta_l2"],
                "candidate_tree_sha256": target_hash,
            }
        ),
        flush=True,
    )
    return record


def create(base, dest, spec_path, manifest, *, min_free_gib=0.0):
    start = time.monotonic()
    if os.path.lexists(dest) or os.path.lexists(manifest):
        raise ValueError("candidate and manifest paths must be new, not symlinks")
    base = base.resolve()
    dest = dest.resolve()
    manifest = manifest.resolve()
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    spec_digest = hashlib.sha256(spec_bytes).hexdigest()
    if (
        dest.exists()
        or manifest.exists()
        or base == dest
        or dest.is_relative_to(base)
        or base.is_relative_to(dest)
    ):
        raise ValueError("new disjoint candidate/manifest required")
    if manifest.is_relative_to(base) or manifest.is_relative_to(dest):
        raise ValueError("manifest must remain external")
    recipe, mapping, h, off = materialization_recipe(base, spec)
    source_rows, source_hash = tree(base)
    if source_hash != spec["base_tree_sha256"]:
        raise ValueError("base full hash differs from pinned source")
    touched = {e["shard"] for e in active_mapping(recipe, mapping)}
    source_paths = files(base)
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("nonnegative disk reserve required")
    need = sum(p.stat().st_size for p in source_paths)
    if shutil.disk_usage(dest.parent).free < min_free_gib * 2**30 + need + 64 * 2**20:
        raise ValueError(
            "insufficient space for private candidate and requested disk reserve"
        )
    dest.mkdir(parents=True)
    owned_inode = (dest.stat().st_dev, dest.stat().st_ino)
    copied = []
    completed = False
    commit_identity = []
    try:
        for p in source_paths:
            rel = p.relative_to(base)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(rel) in touched:
                subprocess.run(
                    ["cp", "--reflink=auto", "--sparse=always", str(p), str(target)],
                    check=True,
                )
                if os.path.samefile(p, target):
                    raise ValueError("private shard shares source inode")
                method = "private-copy-or-reflink"
            else:
                shutil.copy2(p, target)
                method = "independent-copy"
            copied.append(
                {"path": str(rel), "method": method, "size": p.stat().st_size}
            )
        patches = entries(base, dest, recipe, mapping, h, off, True)
        compare_untouched(base, dest, active_mapping(recipe, mapping))
        candidate_rows, candidate_hash = tree(dest)
        # Fresh source full hash after writes makes source immutability explicit.
        if tree(base)[1] != source_hash:
            raise ValueError("source changed during recipe creation")
        changed = sum(p["changed_count"] for p in patches)
        if not changed:
            raise ValueError("entire learned direction vanished after BF16 rounding")
        record = {
            "schema_version": 2,
            "kind": KIND,
            "source": str(base),
            "candidate": str(dest),
            "spec": spec,
            "spec_path": str(spec_path.resolve()),
            "spec_sha256": spec_digest,
            "recipe": recipe,
            "source_tree_sha256": source_hash,
            "candidate_tree_sha256": candidate_hash,
            "source_files": source_rows,
            "candidate_files": candidate_rows,
            "files": copied,
            "edits": patches,
            "changed_elements": changed,
            "changed_tensors": sum(p["changed_count"] > 0 for p in patches),
            "changed_byte_elements": sum(p["changed_byte_count"] for p in patches),
            "delta_l2": math.sqrt(sum(p["delta_l2"] ** 2 for p in patches)),
            "verification": "Exact FP32 recipe regeneration and BF16 bytes, full source/target tree hashes, independent touched-shard inodes, every nonrecipe byte unchanged.",
            "elapsed_seconds": time.monotonic() - start,
        }
        # Recheck provenance snapshots after the long merge, before publication.
        if spec_path.read_bytes() != spec_bytes:
            raise ValueError("spec changed during recipe creation")
        final_recipe, final_mapping, _, _ = materialization_recipe(base, spec)
        if final_recipe != recipe or final_mapping != mapping:
            raise ValueError(
                "donor/config/implementation changed during recipe creation"
            )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        save(manifest, record, commit_identity)
        completed = True
    finally:
        if commit_identity and manifest.exists() and not manifest.is_symlink():
            stat = manifest.stat()
            completed = completed or (stat.st_dev, stat.st_ino) == commit_identity[-1]
        if (
            not completed
            and dest.exists()
            and not dest.is_symlink()
            and (dest.stat().st_dev, dest.stat().st_ino) == owned_inode
        ):
            shutil.rmtree(dest)
    # Creation has already compared every original target block before writing and
    # hashed each regenerated tensor, unchanged span and full source/target tree.
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "kind",
                    "source",
                    "candidate",
                    "changed_elements",
                    "changed_tensors",
                    "delta_l2",
                    "candidate_tree_sha256",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )
    return record


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("create")
    c.add_argument("source", type=Path)
    c.add_argument("candidate", type=Path)
    c.add_argument("--spec", type=Path, required=True)
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--apply", action="store_true")
    c.add_argument("--min-free-gib", type=float, default=500.0)
    v = sub.add_parser("verify")
    v.add_argument("manifest", type=Path)
    a = p.parse_args()
    if a.action == "create":
        if not math.isfinite(a.min_free_gib) or a.min_free_gib < 0:
            p.error("--min-free-gib must be finite and nonnegative")
        if os.path.lexists(a.candidate) or os.path.lexists(a.manifest):
            p.error("candidate and manifest paths must be new")
        if a.apply:
            create(
                a.source, a.candidate, a.spec, a.manifest, min_free_gib=a.min_free_gib
            )
        else:
            base = a.source.resolve()
            dest = a.candidate.resolve()
            manifest = a.manifest.resolve()
            if (
                dest.exists()
                or manifest.exists()
                or base == dest
                or dest.is_relative_to(base)
                or base.is_relative_to(dest)
            ):
                raise ValueError("new disjoint candidate/manifest required")
            if manifest.is_relative_to(base) or manifest.is_relative_to(dest):
                raise ValueError("manifest must remain external")
            recipe, mapping, _, _ = materialization_recipe(base, read(a.spec))
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "source": str(base),
                        "candidate": str(dest),
                        "matrices_or_spans": len(active_mapping(recipe, mapping)),
                        **({"validated_projection_count": len(mapping),
                            "structured_scope": recipe["structured_scope"],
                            "zero_coefficient_projection_count": len(mapping) - len(active_mapping(recipe, mapping))}
                           if "structured_scope" in recipe else {}),
                        "copy_bytes": sum(p.stat().st_size for p in files(base)),
                        "min_free_gib": a.min_free_gib,
                        "source_full_hash_verified": False,
                        "scope": recipe["scope"],
                    }
                )
            )
    else:
        verify(a.manifest)


if __name__ == "__main__":
    main()
