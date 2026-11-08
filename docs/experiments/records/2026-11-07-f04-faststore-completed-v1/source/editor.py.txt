"""Audited native checkpoint edits; no model load, serving, or evaluation.

Schema 1 is the existing experiment's scalar/vector and sparse BF16 matrix recipe.
Creation is a dry run unless --apply is explicit. Source files are never opened
for writing. Dense/LoRA recipes belong to their separate implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

from .checkpoint import Checkpoint, FLOAT_DTYPES

SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
CHUNK = 8 * 1024 * 1024
MAX_JSON = 256 * 1024 * 1024
MAX_FULL_UNIT_JSON = 2 * 1024 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_FULL_UNIT_JSON + 1)
    if len(raw) > MAX_FULL_UNIT_JSON:
        raise ValueError("edit metadata exceeds the 2 GiB full-unit ceiling")
    result = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(result, dict):
        raise ValueError("edit metadata must be an object")
    if len(raw) > MAX_JSON:
        specification = result.get("spec", result)
        if not isinstance(specification, dict) or full_unit_support(specification) is None:
            raise ValueError("metadata over 256 MiB requires explicit full-unit mode")
        matrix_coordinate_limits(specification)
    return result


def _location(checkpoint: Checkpoint, tensor: str, *, sparse_matrix: bool = False) -> tuple[str, int, int, str, list[int]]:
    info = checkpoint.tensors[tensor]
    shape = list(info.shape)
    if sparse_matrix:
        if info.dtype != "BF16" or len(shape) != 2 or any(d <= 0 for d in shape):
            raise ValueError("sparse matrix edits require a nonempty BF16 rank-2 tensor")
    elif info.dtype not in FLOAT_DTYPES or len(shape) > 1 or info.numel > 1_000_000:
        raise ValueError("native edits require floating scalars/vectors with at most 1,000,000 elements")
    return info.relative_shard, info.offset, info.nbytes, info.dtype, shape


def tensor_location(base: Path, tensor: str, *, sparse_matrix: bool = False) -> tuple[str, int, int, str, list[int]]:
    """Compatibility helper; full index/header validation uses the common reader."""
    return _location(Checkpoint(base), tensor, sparse_matrix=sparse_matrix)


def decode(raw: bytes, dtype: str) -> list[float]:
    if dtype == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i+2])[0] for i in range(0, len(raw), 2)]
    return list(struct.unpack("<" + {"F16":"e", "F32":"f", "F64":"d"}[dtype] * (len(raw)//SIZES[dtype]), raw))

def encode(values: list[float], dtype: str) -> bytes:
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite requested value")
    if dtype == "BF16":
        out = bytearray()
        for x in values:
            bits = struct.unpack("<I", struct.pack("<f", x))[0]
            rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
            out.extend(struct.pack("<H", rounded))
        raw = bytes(out)
    else:
        raw = struct.pack("<" + {"F16":"e", "F32":"f", "F64":"d"}[dtype] * len(values), *values)
    if not all(math.isfinite(x) for x in decode(raw, dtype)):
        raise ValueError("nonfinite stored value")
    return raw

def read_at(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        raise ValueError("short read")
    return data


def full_unit_support(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the explicit bounded MLP-unit declaration, without model I/O."""
    if "matrix_full_units" not in spec:
        return None
    declaration = spec["matrix_full_units"]
    if (spec.get("matrix_batch") is not True or not isinstance(declaration, dict)
            or set(declaration) != {"schema_version", "support_plan_sha256", "layer_prefix", "units"}
            or type(declaration.get("schema_version")) is not int or declaration["schema_version"] != 1):
        raise ValueError("full units require matrix_batch and an explicit schema-1 support declaration")
    digest = declaration["support_plan_sha256"]
    prefix = declaration["layer_prefix"]
    rows = declaration["units"]
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*\.layers\.", prefix)
            or not isinstance(rows, list) or not rows):
        raise ValueError("invalid full-unit digest, layer prefix or units")
    units: dict[int, tuple[int, ...]] = {}
    previous = -1
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"layer", "indices"}:
            raise ValueError("full-unit rows require layer and indices")
        layer, indices = row["layer"], row["indices"]
        if (type(layer) is not int or layer <= previous
                or not isinstance(indices, list) or not 1 <= len(indices) <= 2
                or any(type(i) is not int or i < 0 for i in indices)
                or sorted(set(indices)) != indices):
            raise ValueError("sorted unique nonnegative units, at most two per layer, required")
        units[layer] = tuple(indices)
        previous = layer
    if sum(map(len, units.values())) > 32:
        raise ValueError("full-unit support exceeds 32 units")
    return {"layer_prefix": prefix, "units": units}


def matrix_edit_limit(spec: dict[str, Any], edit: dict[str, Any], shape: list[int] | None = None) -> int:
    """Check a changed subset against declared unit rows/columns or Q/K/O sampling."""
    support = full_unit_support(spec)
    if support is None:
        return matrix_coordinate_limits(spec)[0]
    tensor, coordinates = edit.get("tensor"), edit.get("coordinates")
    prefix = support["layer_prefix"]
    if not isinstance(tensor, str) or not tensor.startswith(prefix):
        raise ValueError("full-unit tensor is outside its declared layer namespace")
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(mlp|self_attn)\.([a-z]+)_proj\.weight", tensor[len(prefix):])
    if match is None or not isinstance(coordinates, list) or not coordinates:
        raise ValueError("full-unit edits require explicit projection coordinates; native edits are frozen")
    layer, family, projection = int(match[1]), match[2], match[3]
    if family == "self_attn":
        if projection not in {"q", "k", "o"}:
            raise ValueError("full-unit mode permits sampled Q/K/O only; explicit V is frozen")
        allowed, axis, limit = None, 0, 512
    else:
        if projection not in {"gate", "up", "down"} or layer not in support["units"]:
            raise ValueError("MLP projection is outside declared unit support")
        allowed = support["units"][layer]
        axis = 1 if projection == "down" else 0
        if shape is not None and (len(shape) != 2 or any(i >= shape[axis] for i in allowed)):
            raise ValueError("declared unit exceeds projection dimensions")
        limit = len(allowed) * shape[1 - axis] if shape is not None else 1_000_000
    if len(coordinates) > limit:
        raise ValueError("projection exceeds its declared coordinate support")
    for cell in coordinates:
        if (not isinstance(cell, list) or len(cell) != 2 or any(type(i) is not int or i < 0 for i in cell)
                or (allowed is not None and cell[axis] not in allowed)):
            raise ValueError("coordinate is outside declared MLP unit rows/columns")
    return limit


def _check_full_unit_source(checkpoint: Checkpoint, spec: dict[str, Any]) -> None:
    support = full_unit_support(spec)
    if support is None:
        return
    for layer, units in support["units"].items():
        names = [f"{support['layer_prefix']}{layer}.mlp.{kind}_proj.weight" for kind in ("gate", "up", "down")]
        if any(name not in checkpoint.tensors for name in names):
            raise ValueError("declared MLP unit requires gate/up/down projections")
        shapes = [_location(checkpoint, name, sparse_matrix=True)[4] for name in names]
        if shapes[0] != shapes[1] or shapes[2] != [shapes[0][1], shapes[0][0]] or max(units) >= shapes[0][0]:
            raise ValueError("declared MLP gate/up/down unit dimensions differ")


def matrix_coordinate_limits(spec: dict[str, Any]) -> tuple[int, int]:
    support = full_unit_support(spec)
    mode = spec.get("matrix_batch", False)
    if mode is False:
        return 64, 64
    digest = spec.get("matrix_sample_plan_sha256")
    if (mode is not True or not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("matrix_batch=true requires a lowercase SHA-256 for the sampled coordinate plan")
    return (1_000_000, 1_000_000) if support is not None else (512, 262144)

def sparse_matrix_patches(base: Path, edit: dict[str, Any], shard: str, tensor_offset: int, shape: list[int], *, max_coordinates: int = 64) -> list[dict[str, Any]]:
    coordinates = edit["coordinates"]
    if "indices" in edit or not isinstance(coordinates, list) or not 1 <= len(coordinates) <= max_coordinates:
        raise ValueError("matrix edits require explicit in-limit [row,column] coordinates and no indices")
    nrows, ncols = shape
    seen = set()
    for cell in coordinates:
        if (not isinstance(cell, list) or len(cell) != 2
                or any(type(i) is not int for i in cell)
                or not 0 <= cell[0] < nrows or not 0 <= cell[1] < ncols
                or tuple(cell) in seen):
            raise ValueError("invalid or duplicate matrix coordinate")
        seen.add(tuple(cell))
    operations = [key for key in ("scale", "add", "set", "set_values") if key in edit]
    if len(operations) != 1:
        raise ValueError("choose exactly one matrix operation")
    operation = operations[0]
    values = edit[operation] if operation == "set_values" else [edit[operation]] * len(coordinates)
    if (not isinstance(values, list) or len(values) != len(coordinates)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("matrix edits require one finite numeric target per coordinate")
    patches = []
    with (base / shard).open("rb") as stream:
        for (row, column), value in zip(coordinates, values, strict=True):
            flat_index = row * ncols + column
            offset = tensor_offset + 2 * flat_index
            stream.seek(offset)
            before = stream.read(2)
            if len(before) != 2:
                raise ValueError("short source matrix coordinate")
            old = decode(before, "BF16")[0]
            if not math.isfinite(old):
                raise ValueError("nonfinite source matrix coordinate")
            target = old * value if operation == "scale" else old + value if operation == "add" else value
            after = encode([target], "BF16")
            stored = decode(after, "BF16")[0]
            if stored == old:
                raise ValueError(f"matrix coordinate {row},{column} is a no-op after BF16 rounding")
            delta = abs(stored - old)
            patches.append({
                "tensor": edit["tensor"], "shard": shard, "offset": offset, "size": 2,
                "dtype": "BF16", "shape": shape, "matrix_coordinate": [row, column],
                "operation": operation, "requested_value": value,
                "selected_indices": [flat_index], "changed_indices": [flat_index], "changed_count": 1,
                "hash_and_norm_scope": "the single edited BF16 matrix element",
                "source_tensor_sha256": sha(before), "candidate_tensor_sha256": sha(after),
                "before_hex": before.hex(), "after_hex": after.hex(), "source_l2": abs(old),
                "delta_l2": delta, "relative_delta_l2": delta / abs(old) if old else None,
            })
    return patches

def plan_edits(base: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Read selected source bytes and return legacy-compatible schema-1 patches."""
    if spec.get("kind") is not None or spec.get("schema_version", 1) != 1:
        raise ValueError("only native schema 1 is supported here; use the dense recipe entrypoint")
    if not isinstance(spec.get("edits"), list) or any(not isinstance(e, dict) for e in spec["edits"]):
        raise ValueError("edits must be a list of objects")
    checkpoint = Checkpoint(base)
    _check_full_unit_source(checkpoint, spec)
    edits = spec["edits"]
    if not edits:
        raise ValueError("empty candidate: use official checkpoint for baseline")
    seen, patches, touched = set(), [], set()
    matrix_cells = 0
    _, total_matrix_limit = matrix_coordinate_limits(spec)
    full_units = full_unit_support(spec) is not None
    for edit in edits:
        tensor = edit["tensor"]
        if tensor in seen:
            raise ValueError("combine all changes to one tensor in one edit")
        seen.add(tensor)
        sparse = "coordinates" in edit
        if full_units and not sparse:
            raise ValueError("native parameters are frozen in full-unit mode")
        name, offset, size, dtype, shape = _location(checkpoint, tensor, sparse_matrix=sparse)
        if sparse:
            limit = matrix_edit_limit(spec, edit, shape)
            cells = sparse_matrix_patches(base, edit, name, offset, shape, max_coordinates=limit)
            matrix_cells += len(cells)
            if matrix_cells > total_matrix_limit:
                raise ValueError("candidate exceeds its sparse matrix coordinate limit")
            patches.extend(cells)
            touched.add(name)
            continue
        before = read_at(base/name, offset, size)
        old = decode(before, dtype)
        if not all(math.isfinite(x) for x in old):
            raise ValueError("nonfinite source")
        new = old[:]
        indices = edit.get("indices", list(range(len(old))))
        if not isinstance(indices, list) or not indices:
            raise ValueError("indices must be a nonempty list")
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(old) for i in indices):
            raise ValueError("invalid/duplicate index")
        operations = [key for key in ("scale", "add", "set", "set_values") if key in edit]
        if len(operations) != 1:
            raise ValueError("choose exactly one operation")
        op = operations[0]
        if op == "set_values":
            value = edit[op]
            if ("indices" not in edit or not isinstance(value, list) or len(value) != len(indices)
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in value)):
                raise ValueError("set_values requires one finite value per explicit index")
            for i, target in zip(indices, value, strict=True):
                new[i] = target
        else:
            if type(edit[op]) not in (int, float) or not math.isfinite(edit[op]):
                raise ValueError("operation requires a finite number")
            value = float(edit[op])
            for i in indices:
                new[i] = old[i] * value if op == "scale" else old[i] + value if op == "add" else value
        after = encode(new, dtype)
        stored = decode(after, dtype)
        changed = [i for i,(a,b) in enumerate(zip(old,stored)) if a != b]
        if not changed:
            raise ValueError(f"edit disappears after dtype rounding: {tensor}")
        base_l2 = math.hypot(*old)
        delta_l2 = math.hypot(*(b-a for a,b in zip(old,stored)))
        patches.append(dict(tensor=tensor, shard=name, offset=offset, size=size, dtype=dtype, shape=shape,
            operation=op, requested_value=value, selected_indices=indices,
            changed_indices=changed, changed_count=len(changed), source_tensor_sha256=sha(before),
            candidate_tensor_sha256=sha(after), before_hex=before.hex(), after_hex=after.hex(),
            source_l2=base_l2, delta_l2=delta_l2, relative_delta_l2=delta_l2/base_l2 if base_l2 else None))
        touched.add(name)
    checkpoint.assert_stable()
    for patch in patches:
        if not math.isfinite(patch["source_l2"]) or not math.isfinite(patch["delta_l2"]):
            raise ValueError("L2 is not representable as a finite float")
    return patches


def _validate_spec_record(spec: dict[str, Any], patches: list[dict[str, Any]]) -> None:
    """Pure check of declared operation, exact rounding, indices and byte metrics."""
    if spec.get("kind") is not None or spec.get("schema_version", 1) != 1:
        raise ValueError("unsupported native schema or kind")
    declared = spec.get("edits")
    if not isinstance(declared, list) or not declared:
        raise ValueError("empty/missing native edits")
    by_name: dict[str, dict[str, Any]] = {}
    for edit in declared:
        if not isinstance(edit, dict) or not isinstance(edit.get("tensor"), str) or edit["tensor"] in by_name:
            raise ValueError("duplicate or invalid declared tensor")
        by_name[edit["tensor"]] = edit
    seen: set[tuple[str, int]] = set()
    observed: dict[str, list[tuple[int, ...]]] = {}
    _, total_matrix = matrix_coordinate_limits(spec)
    full_units = full_unit_support(spec) is not None
    coordinate_cache: dict[str, tuple[str, list[int], dict[tuple[int, int], int]]] = {}
    for patch in patches:
        tensor = patch["tensor"]
        if tensor not in by_name:
            raise ValueError("undeclared tensor in manifest")
        edit = by_name[tensor]
        dtype, shape = patch["dtype"], patch["shape"]
        if dtype not in SIZES or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("invalid manifest dtype/shape")
        rel = Path(patch["shard"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.suffix != ".safetensors":
            raise ValueError("manifest shard escapes checkpoint")
        if type(patch["offset"]) is not int or patch["offset"] < 8:
            raise ValueError("invalid manifest offset")
        operations = [x for x in ("scale", "add", "set", "set_values") if x in edit]
        if len(operations) != 1:
            raise ValueError("choose exactly one operation")
        operation = operations[0]
        before = bytes.fromhex(patch["before_hex"])
        after = bytes.fromhex(patch["after_hex"])
        if len(before) != patch["size"] or len(after) != len(before):
            raise ValueError("manifest byte size differs")
        if sha(before) != patch["source_tensor_sha256"] or sha(after) != patch["candidate_tensor_sha256"]:
            raise ValueError("manifest byte digests differ")
        old = decode(before, dtype)
        if not all(math.isfinite(x) for x in old):
            raise ValueError("nonfinite original values")
        matrix = "coordinates" in edit
        if matrix:
            coords = edit["coordinates"]
            if dtype != "BF16" or len(shape) != 2 or min(shape) <= 0 or len(before) != 2:
                raise ValueError("invalid sparse matrix manifest")
            cached = coordinate_cache.get(tensor)
            if cached is None:
                limit = matrix_edit_limit(spec, edit, shape)
                if not isinstance(coords, list) or not 1 <= len(coords) <= limit:
                    raise ValueError("invalid sparse matrix manifest")
                positions: dict[tuple[int, int], int] = {}
                for index, cell in enumerate(coords):
                    if (not isinstance(cell, list) or len(cell) != 2 or
                            any(type(i) is not int for i in cell) or
                            not 0 <= cell[0] < shape[0] or not 0 <= cell[1] < shape[1]
                            or tuple(cell) in positions):
                        raise ValueError("invalid or duplicate matrix coordinate")
                    positions[tuple(cell)] = index
                coordinate_cache[tensor] = (dtype, list(shape), positions)
            else:
                if cached[0] != dtype or cached[1] != shape:
                    raise ValueError("matrix manifest dtype/shape changed within a tensor")
                positions = cached[2]
            cell = patch.get("matrix_coordinate")
            if (not isinstance(cell, list) or len(cell) != 2
                    or any(type(i) is not int for i in cell) or tuple(cell) not in positions):
                raise ValueError("undeclared matrix coordinate")
            position = positions[tuple(cell)]
            selected = [cell[0] * shape[1] + cell[1]]
            observed.setdefault(tensor, []).append(tuple(cell))
            local = [0]
            values = [edit[operation][position]] if operation == "set_values" else [edit[operation]]
        else:
            if full_units:
                raise ValueError("native parameters are frozen in full-unit mode")
            if len(shape) > 1 or math.prod(shape) > 1_000_000 or math.prod(shape) * SIZES[dtype] != len(before):
                raise ValueError("invalid native scalar/vector byte span")
            selected = edit.get("indices", list(range(len(old))))
            if (not isinstance(selected, list) or not selected or
                    len(set(selected)) != len(selected) or
                    any(type(i) is not int or not 0 <= i < len(old) for i in selected)):
                raise ValueError("invalid vector indices")
            local = selected
            values = edit[operation] if operation == "set_values" else [edit[operation]] * len(local)
            observed.setdefault(tensor, []).append(())
        if (not isinstance(values, list) or len(values) != len(local) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("invalid requested values")
        if operation == "set_values" and (not isinstance(edit[operation], list) or
                len(edit[operation]) != len(edit["coordinates"] if matrix else selected)):
            raise ValueError("set_values count differs")
        new = old[:]
        for index, value in zip(local, values, strict=True):
            new[index] = old[index] * value if operation == "scale" else old[index] + value if operation == "add" else value
        expected = encode(new, dtype)
        stored = decode(expected, dtype)
        changed_local = [i for i, (a, b) in enumerate(zip(old, stored, strict=True)) if a != b]
        changed = selected if matrix and changed_local else changed_local
        if (not changed or expected != after or patch["selected_indices"] != selected or
                patch["changed_indices"] != changed or patch["changed_count"] != len(changed) or
                patch["operation"] != operation or
                patch["requested_value"] != (values[0] if matrix else edit[operation])):
            raise ValueError("manifest operation, rounded bytes or changed indices differ")
        source_l2 = math.hypot(*old)
        delta_l2 = math.hypot(*(b-a for a, b in zip(old, stored, strict=True)))
        for key, number in (("source_l2", source_l2), ("delta_l2", delta_l2)):
            if not math.isfinite(number) or not math.isclose(patch[key], number, rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"manifest {key} differs")
        relative = delta_l2 / source_l2 if source_l2 else None
        if (patch["relative_delta_l2"] is None) != (relative is None):
            raise ValueError("manifest relative delta differs")
        if relative is not None and not math.isclose(patch["relative_delta_l2"], relative, rel_tol=1e-12):
            raise ValueError("manifest relative delta differs")
        span_key = (patch["shard"], patch["offset"])
        if span_key in seen:
            raise ValueError("duplicate manifest span")
        seen.add(span_key)
    if set(observed) != set(by_name):
        raise ValueError("manifest misses declared tensors")
    total = 0
    for name, edit in by_name.items():
        expected_cells = [tuple(x) for x in edit["coordinates"]] if "coordinates" in edit else [()]
        if len(set(expected_cells)) != len(expected_cells) or sorted(observed[name]) != sorted(expected_cells):
            raise ValueError("manifest misses or duplicates declared edits")
        if "coordinates" in edit:
            total += len(expected_cells)
    if total > total_matrix:
        raise ValueError("matrix coordinate limit exceeded")
    for shard in {p["shard"] for p in patches}:
        end = 0
        for patch in sorted((p for p in patches if p["shard"] == shard), key=lambda p: p["offset"]):
            if patch["offset"] < end:
                raise ValueError("overlapping manifest spans")
            end = patch["offset"] + patch["size"]


def validate_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata and exact BF16/float edits without opening model files."""
    if record.get("schema_version") != 1 or record.get("kind") is not None:
        raise ValueError("only native schema-1 manifests are supported")
    patches = record["edits"]
    if not isinstance(patches, list) or not patches:
        raise ValueError("manifest has no edits")
    _validate_spec_record(record["spec"], patches)
    summary = {"changed_elements": sum(p["changed_count"] for p in patches),
               "changed_tensors": len({p["tensor"] for p in patches}),
               "delta_l2": math.hypot(*(p["delta_l2"] for p in patches))}
    if any(record[k] != summary[k] for k in ("changed_elements", "changed_tensors")) or not math.isclose(record["delta_l2"], summary["delta_l2"], rel_tol=1e-12):
        raise ValueError("manifest totals differ")
    return summary


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _inventory(root: Path) -> dict[str, tuple[int, int, int, int, int]]:
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if ".cache" in rel.parts or rel.name == ".kairyu-model-attestation.json":
            continue
        if path.is_symlink():
            raise ValueError("checkpoint inventory cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint inventory requires regular files")
        result[str(rel)] = _stamp(path)
    if not result:
        raise ValueError("empty checkpoint")
    return result


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _stream_hashes(path: Path, patches: list[dict[str, Any]]) -> tuple[str, str]:
    current, restored = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as stream:
        position = 0
        for patch in sorted(patches, key=lambda p: p["offset"]):
            left = patch["offset"] - position
            while left:
                block = stream.read(min(CHUNK, left))
                if not block:
                    raise ValueError("short candidate file")
                current.update(block)
                restored.update(block)
                left -= len(block)
            raw = stream.read(patch["size"])
            if raw != bytes.fromhex(patch["after_hex"]):
                raise ValueError("candidate edited bytes differ")
            current.update(raw)
            restored.update(bytes.fromhex(patch["before_hex"]))
            position = patch["offset"] + patch["size"]
        while block := stream.read(CHUNK):
            current.update(block)
            restored.update(block)
    return current.hexdigest(), restored.hexdigest()


def restored_hash(path: Path, patches: list[dict[str, Any]]) -> str:
    """Stream candidate bytes, replacing only declared spans with original bytes."""
    return _stream_hashes(path, patches)[1]


def verify(manifest: Path) -> dict[str, Any]:
    """Fresh full-file comparison, including undeclared bytes and nonweight files."""
    record = read_json(manifest)
    summary = validate_manifest(record)
    if "shard_pool" in record and _hash(Path(record["shard_pool"]["path"])) != record["shard_pool"]["sha256"]:
        raise ValueError("pool provenance changed")
    base, dest = Path(record["source"]).resolve(), Path(record["candidate"]).resolve()
    if base == dest or base.is_relative_to(dest) or dest.is_relative_to(base):
        raise ValueError("source and candidate must be separate directories")
    before, candidate_before = _inventory(base), _inventory(dest)
    if before.keys() != candidate_before.keys():
        raise ValueError("source/candidate file inventories differ")
    expected = plan_edits(base, record["spec"])
    # Pure validation establishes operation/norm consistency; fresh planning pins headers.
    fields = ("tensor", "shard", "offset", "size", "dtype", "shape", "before_hex", "after_hex",
              "selected_indices", "changed_indices", "changed_count")
    if len(expected) != len(record["edits"]) or any(any(a[k] != b[k] for k in fields)
            for a, b in zip(expected, record["edits"], strict=True)):
        raise ValueError("manifest differs from source headers or exact specification")
    Checkpoint(dest).assert_stable()
    recorded_files = {row["path"]: row for row in record["files"]}
    if recorded_files.keys() != before.keys() or len(recorded_files) != len(record["files"]):
        raise ValueError("recorded file inventory differs")
    files = []
    for rel in before:
        patches = [p for p in record["edits"] if p["shard"] == rel]
        if patches and (os.path.samefile(base / rel, dest / rel) or (dest / rel).stat().st_nlink != 1):
            raise ValueError("edited shard shares source inode")
        source_hash = _hash(base / rel)
        candidate_hash, restored = _stream_hashes(dest / rel, patches)
        if source_hash != restored:
            raise ValueError(f"undeclared byte difference: {rel}")
        recorded = recorded_files[rel]
        if recorded["size"] != before[rel][2] or candidate_before[rel][2] != before[rel][2]:
            raise ValueError("file size differs")
        if recorded.get("source_sha256", source_hash) != source_hash:
            raise ValueError("recorded source file changed")
        if recorded.get("candidate_sha256", candidate_hash) != candidate_hash:
            raise ValueError("recorded candidate file changed")
        files.append({"path": rel, "source_sha256": source_hash, "candidate_sha256": candidate_hash})
    if _inventory(base) != before or _inventory(dest) != candidate_before:
        raise ValueError("checkpoint changed during verification")
    return {**summary, "verified": True, "scope": "all regular files except .cache and .kairyu-model-attestation.json",
            "legacy_full_hashes_missing": any("source_sha256" not in x or "candidate_sha256" not in x for x in record["files"]), "files": files}


def create(base: Path, dest: Path, spec_path: Path, manifest: Path, *, apply: bool = False,
           reserve_bytes: int = 0, shard_pool: Path | None = None) -> dict[str, Any]:
    """Plan, or explicitly create private checkpoint copies; never overwrite paths."""
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    base, dest, spec_path, manifest = (p.expanduser().absolute() for p in (base, dest, spec_path, manifest))
    if not base.is_dir() or base.is_symlink():
        raise ValueError("source must be a regular checkpoint directory")
    base = base.resolve()
    if os.path.lexists(dest) or os.path.lexists(manifest):
        raise ValueError("candidate and manifest paths must be new")
    dest, manifest = dest.resolve(), manifest.resolve()
    if (dest.is_relative_to(base) or base.is_relative_to(dest) or
            manifest.is_relative_to(base) or manifest.is_relative_to(dest) or
            manifest == spec_path.resolve()):
        raise ValueError("source, candidate, manifest and specification must not overlap")
    if not dest.parent.is_dir() or not manifest.parent.is_dir():
        raise ValueError("candidate and manifest parent directories must already exist")
    spec_stamp = _stamp(spec_path)
    if spec_stamp[2] > MAX_JSON:
        raise ValueError("specification exceeds 256 MiB")
    spec_bytes = spec_path.read_bytes()
    spec = read_json(spec_path)
    source_before = _inventory(base)
    patches = plan_edits(base, spec)
    shared: dict[str, Path] = {}
    pool_proof = None
    if shard_pool is not None:
        from .shard_pool import verify as verify_pool
        shared = verify_pool(shard_pool, source=base)
        pool_proof = {"path": str(shard_pool.resolve()), "sha256": _hash(shard_pool)}
    touched = {patch["shard"] for patch in patches}
    shared = {rel: path for rel, path in shared.items() if rel not in touched}
    need = sum(value[2] for rel, value in source_before.items() if rel not in shared)
    if shutil.disk_usage(dest.parent).free < need + reserve_bytes + 64 * 2**20:
        raise ValueError("insufficient free space for private copies and requested reserve")
    summary = {"changed_elements": sum(p["changed_count"] for p in patches),
               "changed_tensors": len({p["tensor"] for p in patches}),
               "delta_l2": math.hypot(*(p["delta_l2"] for p in patches)),
               "copy_bytes_upper_bound": need, "hardlink_bytes": sum(source_before[rel][2] for rel in shared),
               "reserve_bytes": reserve_bytes, "applied": False}
    if not apply:
        if _inventory(base) != source_before or _stamp(spec_path) != spec_stamp:
            raise ValueError("inputs changed during planning")
        return summary
    owned: dict[Path, tuple[int, int]] = {}
    committed = False
    temporary: Path | None = None
    manifest_inode: tuple[int, int] | None = None
    dest.mkdir()
    dest_inode = _stamp(dest)[:2]
    try:
        files: list[dict[str, Any]] = []
        for rel, stamp in source_before.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            method = "private-copy"
            if rel in shared:
                os.link(shared[rel], target)
                owned[target] = _stamp(target)[:2]
                source_digest = _hash(base / rel)
                method = "verified-pool-hardlink"
            else:
                with target.open("xb") as out:
                    owned[target] = _stamp(target)[:2]
                    with (base / rel).open("rb") as source:
                        digest = hashlib.sha256()
                        while block := source.read(CHUNK):
                            out.write(block)
                            digest.update(block)
                        out.flush()
                        os.fsync(out.fileno())
                source_digest = digest.hexdigest()
            if _stamp(base / rel) != stamp:
                raise ValueError("source changed while copying")
            files.append({"path": rel, "method": method, "size": stamp[2],
                          "source_sha256": source_digest})
        for rel in {p["shard"] for p in patches}:
            path = dest / rel
            if os.path.samefile(base / rel, path) or path.stat().st_nlink != 1:
                raise ValueError("edited shard shares source inode")
            with path.open("r+b") as target:
                for patch in sorted((p for p in patches if p["shard"] == rel), key=lambda p: p["offset"]):
                    target.seek(patch["offset"])
                    if target.read(patch["size"]) != bytes.fromhex(patch["before_hex"]):
                        raise ValueError("private copy differs before patch")
                    target.seek(patch["offset"])
                    target.write(bytes.fromhex(patch["after_hex"]))
                target.flush()
                os.fsync(target.fileno())
        candidate_stamps = _inventory(dest)
        for row in files:
            relevant = [p for p in patches if p["shard"] == row["path"]]
            candidate_hash, restored = _stream_hashes(dest / row["path"], relevant)
            if restored != row["source_sha256"]:
                raise ValueError("undeclared difference after creation")
            row["candidate_sha256"] = candidate_hash
        if (_inventory(base) != source_before or _inventory(dest) != candidate_stamps or _stamp(spec_path) != spec_stamp or
                spec_path.read_bytes() != spec_bytes):
            raise ValueError("inputs changed during creation")
        record = {"schema_version": 1, "source": str(base), "candidate": str(dest),
                  "spec": spec, "spec_sha256": sha(spec_bytes), "files": files, "edits": patches,
                  **{key: summary[key] for key in ("changed_elements", "changed_tensors", "delta_l2")},
                  "verification": "full file hashes and original-byte restoration, source stat-stable"}
        if pool_proof is not None:
            if _hash(Path(pool_proof["path"])) != pool_proof["sha256"]:
                raise ValueError("pool index changed during creation")
            record["shard_pool"] = pool_proof
        validate_manifest(record)
        fd, temporary_name = tempfile.mkstemp(prefix=".native-edit-", dir=manifest.parent)
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w") as out:
            json.dump(record, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        manifest_inode = _stamp(temporary)[:2]
        os.link(temporary, manifest)  # Exclusive atomic publication; no replacing another writer.
        committed = True
        return {**summary, "applied": True, "manifest": str(manifest)}
    finally:
        published = manifest_inode is not None and manifest.exists() and _stamp(manifest)[:2] == manifest_inode
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if not committed and not published and dest.exists() and _stamp(dest)[:2] == dest_inode:
            for path, inode in owned.items():
                if path.exists() and not path.is_symlink() and _stamp(path)[:2] == inode:
                    path.unlink()
            for path in sorted((p for p in dest.rglob("*") if p.is_dir() and not p.is_symlink()),
                               key=lambda p: len(p.parts), reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass
            try:
                dest.rmdir()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("source", type=Path)
    create_parser.add_argument("candidate", type=Path)
    create_parser.add_argument("--spec", type=Path, required=True)
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--apply", action="store_true")
    create_parser.add_argument("--reserve-bytes", type=int, default=0)
    create_parser.add_argument("--shard-pool", type=Path, help="Explicit verified pool index; only untouched shards may be linked")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = create(args.source, args.candidate, args.spec, args.manifest, apply=args.apply,
                    reserve_bytes=args.reserve_bytes, shard_pool=args.shard_pool) if args.command == "create" else verify(args.manifest)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
