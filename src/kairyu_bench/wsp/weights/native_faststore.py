"""Fixed-slot, byte-preserving native checkpoint storage; no model execution.

Shared payload links are all created before the generation is frozen. A claim
renames one reserved directory; it never adds/removes a shared payload link.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
from typing import Any, Iterator

from . import checkpoint as reader
from . import editor

STORE_KIND = "native_faststore_v1"
CANDIDATE_KIND = "native_faststore_candidate_v1"
NATIVE = "native.safetensors"
BULK = "shared.safetensors"
CHUNK = 8 * 1024 * 1024


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state(path: Path) -> dict[str, int]:
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid():
        raise ValueError("files must be current-user-owned regular files")
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "nlink": s.st_nlink, "uid": s.st_uid, "gid": s.st_gid, "mode": s.st_mode}


def inventory(root: Path) -> dict[str, dict[str, int]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint directory required, without symlink")
    return {rel: state(root / rel) for rel in editor._inventory(root)}


def pin(path: Path, expected: str | None = None) -> dict[str, Any]:
    before = state(path)
    value = editor._hash(path)
    if state(path) != before or (expected is not None and value != expected):
        raise ValueError("input SHA/stat mismatch")
    return {"path": str(path.resolve()), "sha256": value, "stat": before}


def stable_pin(value: dict[str, Any]) -> None:
    if pin(Path(value["path"]), value["sha256"]) != value:
        raise ValueError("pinned input changed")


def publish(path: Path, value: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=".proof-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def implementations() -> dict[str, str]:
    return {str(Path(p).resolve()): editor._hash(Path(p))
            for p in (__file__, reader.__file__, editor.__file__)}


def _rows(proof: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = proof["all_checkpoint_files"]
    result = {row["path"]: row for row in rows}
    if len(result) != len(rows) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in result):
        raise ValueError("invalid full-file inventory")
    if digest(sorted(rows, key=lambda r: r["path"])) != proof["control_tree_sha256"]:
        raise ValueError("physical file-tree digest mismatch")
    return result


def _check_freeze(root: Path, proof: dict[str, Any]) -> dict[str, dict[str, int]]:
    current = inventory(root)
    rows = _rows(proof)
    if (Path(proof["checkpoint_path"]).resolve() != root or
            current != proof["candidate_inventory_stat"] or set(rows) != set(current) or
            any(rows[p]["size"] != current[p]["size"] for p in current)):
        raise ValueError("checkpoint differs from pinned full-file/stat proof")
    return current


def _copy(source: Path, target: Path) -> str:
    h = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        while block := src.read(CHUNK):
            h.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    return h.hexdigest()


def _header(path: Path, tensors: list[reader.TensorInfo]) -> dict[str, int]:
    header: dict[str, Any] = {}
    offsets: dict[str, int] = {}
    end = 0
    for t in tensors:
        offsets[t.name] = end
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [end, end + t.nbytes]}
        end += t.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    if len(raw) > reader.MAX_HEADER_BYTES:
        raise ValueError("packed header exceeds reader limit")
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw)) + raw)
        stream.truncate(8 + len(raw) + end)
    return {name: 8 + len(raw) + offset for name, offset in offsets.items()}


def _scan_shard(path: Path, tensors: list[reader.TensorInfo], sinks: dict[str, tuple[Any, int]] | None = None) -> tuple[str, dict[str, str]]:
    """One sequential raw read; optionally copy spans without dtype conversion."""
    h = hashlib.sha256()
    hashes = {}
    ordered = sorted(tensors, key=lambda t: t.offset)
    with path.open("rb") as stream:
        prefix_size = ordered[0].offset
        prefix = stream.read(prefix_size)
        if len(prefix) != prefix_size:
            raise ValueError("short source header")
        h.update(prefix)
        for tensor in ordered:
            if stream.tell() != tensor.offset:
                raise ValueError("tensor layout is not contiguous")
            th = hashlib.sha256()
            sink = sinks.get(tensor.name) if sinks is not None else None
            if sink is not None:
                sink[0].seek(sink[1])
            remaining = tensor.nbytes
            while remaining:
                block = stream.read(min(CHUNK, remaining))
                if not block:
                    raise ValueError("truncated tensor payload")
                h.update(block)
                th.update(block)
                if sink is not None:
                    sink[0].write(block)
                remaining -= len(block)
            hashes[tensor.name] = th.hexdigest()
        if stream.read(1):
            raise ValueError("unassigned trailing source bytes")
    return h.hexdigest(), hashes


def build(source: Path, output: Path, selected: list[str], source_proof: Path,
          source_proof_sha256: str, *, capacity: int = 16, reserve_bytes: int = 0,
          apply: bool = False) -> dict[str, Any]:
    source, output = source.absolute(), output.absolute()
    if source.is_symlink() or os.path.lexists(output) or not output.parent.is_dir():
        raise ValueError("regular source and fresh output with existing parent required")
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source/output overlap")
    if type(capacity) is not int or not 1 <= capacity <= 64 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("capacity must be 1..64 and reserve nonnegative")
    proof_pin = pin(source_proof, source_proof_sha256)
    proof = editor.read_json(source_proof)
    original = _check_freeze(source, proof)
    cp = reader.Checkpoint(source)
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or not set(selected) < set(cp.tensors)):
        raise ValueError("explicit unique proper subset of tensors required")
    if any(cp.tensors[n].dtype != "BF16" or len(cp.tensors[n].shape) > 1 for n in selected):
        raise ValueError("this generation isolates only BF16 native scalars/vectors")
    shards = {t.relative_shard for t in cp.tensors.values()}
    extras = set(original) - shards - {"model.safetensors.index.json"}
    if (any(p.endswith(".safetensors") for p in extras) or
            sum(original[p]["size"] for p in extras) > 256 * 1024**2):
        raise ValueError("undeclared shard or excessive nonweight metadata")
    if any(p in extras for p in (NATIVE, BULK)):
        raise ValueError("reserved packed filename collision")
    required = sum(s["size"] for s in original.values()) + 64 * 1024**2
    if shutil.disk_usage(output.parent).free < required + reserve_bytes:
        raise ValueError("insufficient free space for one repack and reserve")
    preview = {"applied": False, "capacity": capacity, "tensor_count": len(cp.tensors),
               "selected_tensor_count": len(selected), "private_native_bytes": sum(cp.tensors[n].nbytes for n in selected),
               "copy_bytes_upper_bound": required, "reserve_bytes": reserve_bytes}
    if not apply:
        return preview
    code = implementations()
    output.mkdir(mode=0o700)
    base = output / "base"
    slots = output / "slots"
    base.mkdir()
    slots.mkdir()
    (output / "lock").touch(mode=0o600, exist_ok=False)
    try:
        targets = {NATIVE: [t for n, t in cp.tensors.items() if n in selected],
                   BULK: [t for n, t in cp.tensors.items() if n not in selected]}
        offsets = {name: _header(base / name, tensors) for name, tensors in targets.items()}
        expected = _rows(proof)
        tensor_hashes = {}
        with (base / NATIVE).open("r+b") as small, (base / BULK).open("r+b") as large:
            sinks = {n: (small if n in selected else large, offsets[NATIVE if n in selected else BULK][n]) for n in cp.tensors}
            for rel in sorted(shards):
                h, th = _scan_shard(source / rel, [t for t in cp.tensors.values() if t.relative_shard == rel], sinks)
                if h != expected[rel]["sha256"]:
                    raise ValueError("source full shard SHA differs from prior proof")
                tensor_hashes.update(th)
            for stream in (small, large):
                stream.flush()
                os.fsync(stream.fileno())
        for rel in sorted(extras):
            if _copy(source / rel, base / rel) != expected[rel]["sha256"]:
                raise ValueError("source metadata SHA differs")
        if "model.safetensors.index.json" in original and editor._hash(source / "model.safetensors.index.json") != expected["model.safetensors.index.json"]["sha256"]:
            raise ValueError("source index SHA differs")
        publish(base / "model.safetensors.index.json", {
            "metadata": {"total_size": sum(t.nbytes for t in cp.tensors.values())},
            "weight_map": {n: NATIVE if n in selected else BULK for n in cp.tensors}})
        (base / BULK).chmod(0o444)
        # No shared link may be added or removed after this loop finishes.
        for number in range(capacity):
            slot = slots / f"slot-{number:03d}"
            slot.mkdir()
            os.link(base / BULK, slot / BULK)
        packed = reader.Checkpoint(base)
        if set(packed.tensors) != set(cp.tensors) or any(
                (t.dtype, t.shape, t.nbytes) != (cp.tensors[n].dtype, cp.tensors[n].shape, cp.tensors[n].nbytes)
                for n, t in packed.tensors.items()):
            raise ValueError("repacked tensor layout mismatch")
        rows: list[dict[str, Any]] = []
        for rel in (BULK, NATIVE):
            h, th = _scan_shard(base / rel, [t for t in packed.tensors.values() if t.relative_shard == rel])
            if any(th[n] != tensor_hashes[n] for n in th):
                raise ValueError("independent repacked tensor-byte comparison failed")
            rows.append({"path": rel, "size": (base / rel).stat().st_size, "sha256": h})
        for rel in sorted(set(inventory(base)) - {BULK, NATIVE}):
            rows.append({"path": rel, "size": (base / rel).stat().st_size, "sha256": editor._hash(base / rel)})
        rows.sort(key=lambda row: row["path"])
        packed.assert_stable()
        if inventory(source) != original or implementations() != code:
            raise ValueError("source or implementation changed during repack")
        stable_pin(proof_pin)
        before = inventory(base)
        record = {"schema_version": 1, "kind": STORE_KIND, "applied": True, "store": str(output),
                  "checkpoint_path": str(base), "capacity": capacity, "selected_tensors": sorted(selected),
                  "source_proof": proof_pin, "source_checkpoint_path": str(source),
                  "source_inventory_stat": original, "logical_source_tree_sha256": proof["control_tree_sha256"],
                  "control_tree_sha256": digest(rows), "all_checkpoint_files": rows,
                  "candidate_inventory_stat": before, "implementation_sha256": code,
                  "tensor_byte_sha256": tensor_hashes,
                  "equivalence": {"all_tensor_names_dtype_shape_raw_bytes_equal": True, "tensor_count": len(cp.tensors),
                                  "source_full_hashes_recomputed": True, "packed_full_hashes_recomputed": True,
                                  "logical_parent_unchanged": True, "physical_file_tree_changed": True},
                  "shared_file": BULK, "shared_link_count": capacity + 1,
                  "lifetime": "No shared-file unlink, relink, expansion, chmod, touch or writes until the entire generation is retired."}
        publish(output / "receipt.json", record)
        return {**preview, "applied": True, "receipt": str(output / "receipt.json"), "physical_tree_sha256": record["control_tree_sha256"]}
    except BaseException:
        # Keep incomplete generation quarantined; never unlink shared links here.
        publish(output / "failed.json", {"status": "incomplete_generation_do_not_use"})
        raise


@contextmanager
def locked(receipt: Path, expected: str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    receipt_pin = pin(receipt, expected)
    store = editor.read_json(receipt)
    root = Path(store["store"])
    if (store.get("kind") != STORE_KIND or store.get("schema_version") != 1 or store.get("applied") is not True
            or receipt.resolve() != root / "receipt.json" or root.is_symlink()):
        raise ValueError("invalid faststore generation")
    lock = root / "lock"
    state(lock)
    with lock.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        _store_check(store)
        stable_pin(receipt_pin)
        yield store, receipt_pin
        _store_check(store)
        stable_pin(receipt_pin)


def _store_check(store: dict[str, Any]) -> None:
    if implementations() != store["implementation_sha256"]:
        raise ValueError("faststore implementation changed")
    stable_pin(store["source_proof"])
    if inventory(Path(store["source_checkpoint_path"])) != store["source_inventory_stat"]:
        raise ValueError("original logical source changed")
    current = _check_freeze(Path(store["checkpoint_path"]), store)
    if current[BULK]["nlink"] != store["shared_link_count"]:
        raise ValueError("shared link count changed")


def _candidate_check(proof: dict[str, Any], store: dict[str, Any], store_pin: dict[str, Any]) -> dict[str, dict[str, int]]:
    if proof.get("kind") != CANDIDATE_KIND or proof["store_receipt"] != store_pin:
        raise ValueError("parent belongs to a different or unverified generation")
    root = Path(proof["checkpoint_path"])
    current = _check_freeze(root, proof)
    expected = _rows(proof)
    if set(current) != set(store["candidate_inventory_stat"]):
        raise ValueError("candidate inventory differs from generation")
    base = Path(store["checkpoint_path"])
    if (current[BULK] != store["candidate_inventory_stat"][BULK] or
            not os.path.samefile(root / BULK, base / BULK) or
            expected[BULK] != _rows(store)[BULK]):
        raise ValueError("candidate shared storage differs")
    for rel in set(current) - {BULK}:
        if current[rel]["nlink"] != 1 or editor._hash(root / rel) != expected[rel]["sha256"]:
            raise ValueError("candidate private file differs")
    if inventory(root) != current:
        raise ValueError("candidate changed during verification")
    return current


def _parent(parent: Path, proof_path: Path, proof_sha256: str, store: dict[str, Any], store_pin: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    proof_pin = pin(proof_path, proof_sha256)
    proof = editor.read_json(proof_path)
    if parent.is_symlink():
        raise ValueError("parent symlink refused")
    parent = parent.resolve()
    if parent == Path(store["source_checkpoint_path"]):
        if proof_pin != store["source_proof"]:
            raise ValueError("initial parent proof differs from equivalent logical source")
        _check_freeze(parent, proof)
        return Path(store["checkpoint_path"]), proof, proof_pin
    if parent != Path(proof["checkpoint_path"]):
        raise ValueError("parent path mismatch")
    _candidate_check(proof, store, store_pin)
    return parent, proof, proof_pin


def _spec_parent(spec: dict[str, Any], logical: dict[str, Any]) -> None:
    if "source_checkpoint" in spec and (
            not isinstance(spec["source_checkpoint"], str)
            or not Path(spec["source_checkpoint"]).is_absolute()
            or Path(spec["source_checkpoint"]).resolve() != Path(logical["checkpoint_path"]).resolve()):
        raise ValueError("declared specification parent checkpoint differs")
    if "base_tree_sha256" in spec and spec["base_tree_sha256"] != logical["control_tree_sha256"]:
        raise ValueError("declared specification parent tree differs")


def create(receipt: Path, receipt_sha256: str, parent: Path, parent_proof: Path,
           parent_proof_sha256: str, spec_path: Path, output: Path, *, reserve_bytes: int = 0,
           apply: bool = False) -> dict[str, Any]:
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve must be nonnegative")
    if os.path.lexists(output) or not output.parent.is_dir():
        raise ValueError("candidate bundle must be fresh with existing parent")
    output = output.resolve()
    with locked(receipt, receipt_sha256) as (store, store_pin):
        base, logical, parent_pin = _parent(parent, parent_proof, parent_proof_sha256, store, store_pin)
        if (output.is_relative_to(Path(store["store"])) or output.is_relative_to(base) or base.is_relative_to(output)
                or output.parent.stat().st_dev != Path(store["store"]).stat().st_dev):
            raise ValueError("candidate must be separate and on the same filesystem")
        spec_pin = pin(spec_path)
        spec = editor.read_json(spec_path)
        _spec_parent(spec, logical)
        if (not isinstance(spec.get("edits"), list) or any(e.get("tensor") not in store["selected_tensors"] or
                "coordinates" in e for e in spec["edits"]) or spec.get("matrix_batch", False)):
            raise ValueError("edits outside isolated native scope require a new generation")
        patches = editor.plan_edits(base, spec)
        if any(p["shard"] != NATIVE or len(p["shape"]) > 1 for p in patches):
            raise ValueError("shared/matrix edits are never a native fast path")
        base_before = inventory(base)
        small = sorted(set(base_before) - {BULK})
        required = sum(base_before[p]["size"] for p in small)
        if shutil.disk_usage(output.parent).free < required + reserve_bytes + 64 * 1024**2:
            raise ValueError("insufficient free space for private native/metadata and reserve")
        metrics = {"changed_elements": sum(p["changed_count"] for p in patches),
                   "changed_tensors": len({p["tensor"] for p in patches}),
                   "delta_l2": math.hypot(*(p["delta_l2"] for p in patches))}
        native_record = {"schema_version": 1, "spec": spec, "edits": patches, **metrics}
        editor.validate_manifest(native_record)
        slots = sorted((Path(store["store"]) / "slots").glob("slot-*"))
        if not slots:
            raise ValueError("fixed generation capacity exhausted; never add links")
        slot = slots[0]
        if (slot.is_symlink() or not slot.is_dir() or set(p.name for p in slot.iterdir()) != {BULK}
                or state(slot / BULK) != store["candidate_inventory_stat"][BULK]
                or not os.path.samefile(slot / BULK, Path(store["checkpoint_path"]) / BULK)):
            raise ValueError("reserved slot differs from frozen shared payload")
        preview = {"applied": False, "private_copy_bytes": required, "slot": slot.name, **metrics}
        if not apply:
            return preview
        output.mkdir(mode=0o700)
        dest = output / "checkpoint"
        try:
            # Parent bundle mkdir is exclusive. The reserved subdirectory is
            # renamed into that owned namespace without touching payload links.
            os.rename(slot, dest)
            for rel in small:
                _copy(base / rel, dest / rel)
            with (dest / NATIVE).open("r+b") as target:
                for p in sorted(patches, key=lambda p: p["offset"]):
                    target.seek(p["offset"])
                    if target.read(p["size"]) != bytes.fromhex(p["before_hex"]):
                        raise ValueError("native copy differs before edit")
                    target.seek(p["offset"])
                    target.write(bytes.fromhex(p["after_hex"]))
                target.flush()
                os.fsync(target.fileno())
            rows = [_rows(store)[BULK]]
            for rel in small:
                current, restored = editor._stream_hashes(dest / rel, patches if rel == NATIVE else [])
                if restored != editor._hash(base / rel):
                    raise ValueError("undeclared private byte difference")
                rows.append({"path": rel, "size": (dest / rel).stat().st_size, "sha256": current})
            rows.sort(key=lambda row: row["path"])
            reader.Checkpoint(dest).assert_stable()
            if inventory(base) != base_before:
                raise ValueError("parent changed during creation")
            stable_pin(parent_pin)
            stable_pin(spec_pin)
            _store_check(store)
            frozen = inventory(dest)
            result = {"schema_version": 1, "kind": CANDIDATE_KIND, "status": "verified_native_faststore_mixed_hash_stat",
                      "checkpoint_path": str(dest), "control_tree_sha256": digest(rows),
                      "all_checkpoint_files": rows, "candidate_inventory_stat": frozen,
                      "store_receipt": store_pin, "logical_parent": {"checkpoint_path": str(parent.resolve()),
                      "tree_sha256": logical["control_tree_sha256"], "proof": parent_pin},
                      "storage_parent": str(base), "spec_file": spec_pin, "native_manifest": native_record,
                      "slot": slot.name, **metrics,
                      "verification": {"verified": True, "fresh_full_model_hash": False,
                      "fresh_private_files_sha256": small, "shared_files_prior_full_hash_exact_inode_stat": [BULK],
                      "private_original_byte_restoration": True, "shared_payload_bytes_read": 0,
                      "no_shared_link_mutation": True}}
            _candidate_check(result, store, store_pin)
            publish(output / "manifest.json", result)
            return {**preview, "applied": True, "manifest": str(output / "manifest.json"),
                    "checkpoint_path": str(dest), "tree_sha256": result["control_tree_sha256"]}
        except BaseException:
            # Failed slots stay quarantined with their links intact. Removing
            # a failed/shared checkpoint would invalidate every generation proof.
            publish(output / "failed.json", {"status": "failed_slot_preserved_do_not_unlink_shared_files"})
            raise


def verify(manifest: Path, manifest_sha256: str) -> dict[str, Any]:
    manifest_pin = pin(manifest, manifest_sha256)
    proof = editor.read_json(manifest)
    store_ref = proof["store_receipt"]
    with locked(Path(store_ref["path"]), store_ref["sha256"]) as (store, store_pin):
        _candidate_check(proof, store, store_pin)
        parent_ref = proof["logical_parent"]["proof"]
        base, logical, parent_pin = _parent(Path(proof["logical_parent"]["checkpoint_path"]), Path(parent_ref["path"]), parent_ref["sha256"], store, store_pin)
        native = proof["native_manifest"]
        _spec_parent(native["spec"], logical)
        if proof["logical_parent"]["tree_sha256"] != logical["control_tree_sha256"]:
            raise ValueError("logical parent tree differs from pinned proof")
        if editor.read_json(Path(proof["spec_file"]["path"])) != native["spec"]:
            raise ValueError("captured specification differs from native proof")
        metrics = editor.validate_manifest(native)
        expected = editor.plan_edits(base, native["spec"])
        if expected != native["edits"] or any(p["shard"] != NATIVE or p["tensor"] not in store["selected_tensors"] for p in expected):
            raise ValueError("native proof differs from exact parent/spec")
        if any(proof[k] != v for k, v in metrics.items()):
            raise ValueError("candidate edit totals differ")
        dest = Path(proof["checkpoint_path"])
        for rel in set(inventory(dest)) - {BULK}:
            _, restored = editor._stream_hashes(dest / rel, expected if rel == NATIVE else [])
            if restored != editor._hash(base / rel):
                raise ValueError("private original-byte restoration failed")
        stable_pin(parent_pin)
        stable_pin(proof["spec_file"])
        stable_pin(manifest_pin)
        _candidate_check(proof, store, store_pin)
        return {"verified": True, **metrics, "tree_sha256": proof["control_tree_sha256"], **proof["verification"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("source", type=Path)
    b.add_argument("output", type=Path)
    b.add_argument("--selected-tensors", type=Path, required=True, help="JSON list of exact tensor names")
    b.add_argument("--source-proof", type=Path, required=True)
    b.add_argument("--source-proof-sha256", required=True)
    b.add_argument("--capacity", type=int, default=16)
    b.add_argument("--reserve-bytes", type=int, default=0)
    b.add_argument("--apply", action="store_true")
    c = commands.add_parser("create")
    c.add_argument("receipt", type=Path)
    c.add_argument("--receipt-sha256", required=True)
    c.add_argument("--parent", type=Path, required=True)
    c.add_argument("--parent-proof", type=Path, required=True)
    c.add_argument("--parent-proof-sha256", required=True)
    c.add_argument("--spec", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True, help="fresh bundle; checkpoint is OUTPUT/checkpoint")
    c.add_argument("--reserve-bytes", type=int, default=0)
    c.add_argument("--apply", action="store_true")
    v = commands.add_parser("verify")
    v.add_argument("manifest", type=Path)
    v.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if os.getuid() == 0:
        raise ValueError("run as the ordinary owning user")
    if args.command == "build":
        selected = json.loads(args.selected_tensors.read_bytes())
        result = build(args.source, args.output, selected, args.source_proof, args.source_proof_sha256,
                       capacity=args.capacity, reserve_bytes=args.reserve_bytes, apply=args.apply)
    elif args.command == "create":
        result = create(args.receipt, args.receipt_sha256, args.parent, args.parent_proof,
                        args.parent_proof_sha256, args.spec, args.output, reserve_bytes=args.reserve_bytes, apply=args.apply)
    else:
        result = verify(args.manifest, args.manifest_sha256)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
