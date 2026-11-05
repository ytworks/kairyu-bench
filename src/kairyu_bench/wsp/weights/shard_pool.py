"""Explicit, verified hardlink pools for unchanged checkpoint shards."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .checkpoint import Checkpoint
from .editor import _hash, _inventory, _stamp, read_json

KIND = "verified_shard_pool_v1"


def _save_new(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=".pool-index-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def build(source: Path, pool: Path, donors: dict[str, str], *, apply: bool = False) -> dict[str, Any]:
    """Register explicit user-owned donor shards; no source or donor file writes."""
    source = source.resolve()
    if os.path.lexists(pool):
        raise ValueError("pool path must be new")
    pool = pool.resolve()
    if not pool.parent.is_dir() or source == pool or source.is_relative_to(pool) or pool.is_relative_to(source):
        raise ValueError("pool and source must be disjoint; pool parent must exist")
    checkpoint = Checkpoint(source)
    allowed = {t.relative_shard for t in checkpoint.tensors.values()}
    if not isinstance(donors, dict) or not donors or not set(donors) <= allowed:
        raise ValueError("donor mapping must name existing checkpoint shards")
    source_before = _inventory(source)
    source_inodes = {stamp[:2] for stamp in source_before.values()}
    source_files = [{"path": rel, "size": stamp[2], "sha256": _hash(source / rel),
                     "stat": list(stamp)} for rel, stamp in source_before.items()]
    expected = {row["path"]: row for row in source_files}
    entries: dict[str, dict[str, Any]] = {}
    for rel, supplied in donors.items():
        path = Path(supplied).expanduser()
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
            raise ValueError("donors must be current-user-owned regular non-symlink files")
        path = path.resolve()
        if path.is_relative_to(source) or path.is_relative_to(pool) or os.path.samefile(path, source / rel):
            raise ValueError("donor must be an independent copy outside source and pool")
        before = _stamp(path)
        if before[:2] in source_inodes:
            raise ValueError("donor aliases a source file")
        digest = _hash(path)
        if digest != expected[rel]["sha256"] or before[2] != expected[rel]["size"] or _stamp(path) != before:
            raise ValueError("donor does not exactly match the stable original shard")
        if before[0] != pool.parent.stat().st_dev:
            raise ValueError("hardlink pool and donors must share a filesystem")
        entries[rel] = {"donor_path": str(path), "sha256": digest, "size": before[2],
                        "device": before[0], "inode": before[1], "mtime_ns": before[3],
                        "uid": os.getuid()}
    if _inventory(source) != source_before:
        raise ValueError("source changed during pool planning")
    record = {"schema_version": 1, "kind": KIND, "source": str(source), "pool": str(pool),
              "source_files": source_files, "files": entries, "applied": False,
              "scope": "All retained source regular files; .cache and prior serving attestation excluded"}
    if not apply:
        return record
    pool.mkdir()
    pool_inode = _stamp(pool)[:2]
    owned: dict[Path, tuple[int, int]] = {}
    published = False
    try:
        for rel, row in entries.items():
            target = pool / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            donor = Path(row["donor_path"])
            current = _stamp(donor)
            if (current[0], current[1], current[2], current[3], donor.stat().st_uid) != (
                    row["device"], row["inode"], row["size"], row["mtime_ns"], row["uid"]):
                raise ValueError("donor identity changed before linking")
            os.link(donor, target)
            owned[target] = _stamp(target)[:2]
            if _hash(target) != row["sha256"]:
                raise ValueError("pool shard changed while linking")
        if _inventory(source) != source_before:
            raise ValueError("source changed while building pool")
        record["applied"] = True
        _save_new(pool / "index.json", record)
        published = True
        return record
    finally:
        # Publication is complete if our expected index survived a later exception.
        if not published and (pool / "index.json").exists():
            try:
                published = read_json(pool / "index.json") == record
            except (OSError, ValueError):
                pass
        if not published and pool.exists() and not pool.is_symlink() and _stamp(pool)[:2] == pool_inode:
            for path, inode in owned.items():
                if path.exists() and not path.is_symlink() and _stamp(path)[:2] == inode:
                    path.unlink()
            for path in sorted((p for p in pool.rglob("*") if p.is_dir() and not p.is_symlink()),
                               key=lambda p: len(p.parts), reverse=True):
                try:
                    path.rmdir()
                except OSError:
                    pass
            try:
                pool.rmdir()
            except OSError:
                pass


def verify(index: Path, *, source: Path | None = None) -> dict[str, Path]:
    """Fresh hash of pooled shards and continuity of the complete source inventory."""
    if index.is_symlink():
        raise ValueError("pool index cannot be a symlink")
    index_stamp = _stamp(index)
    record = read_json(index)
    if record.get("schema_version") != 1 or record.get("kind") != KIND or record.get("applied") is not True:
        raise ValueError("unsupported or unapplied pool index")
    root = Path(record["source"]).resolve()
    pool = Path(record["pool"]).resolve()
    if index.resolve() != pool / "index.json" or (source is not None and source.resolve() != root):
        raise ValueError("pool source or index path differs")
    before = _inventory(root)
    source_inodes = {stamp[:2] for stamp in before.values()}
    rows = record["source_files"]
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows) or expected.keys() != before.keys():
        raise ValueError("source file inventory changed")
    if any(list(stamp) != expected[rel]["stat"] or stamp[2] != expected[rel]["size"] for rel, stamp in before.items()):
        raise ValueError("source file state changed since full-hash attestation")
    allowed = {t.relative_shard for t in Checkpoint(root).tensors.values()}
    entries = record["files"]
    if not entries or not set(entries) <= allowed:
        raise ValueError("pool contains undeclared shards")
    result = {}
    for rel, row in entries.items():
        path = pool / rel
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(pool):
            raise ValueError("invalid pool path")
        stamp = _stamp(path)
        if (stamp[0], stamp[1], stamp[2], stamp[3], path.stat().st_uid) != (
                row["device"], row["inode"], row["size"], row["mtime_ns"], row["uid"]):
            raise ValueError("pool inode/ownership/size changed")
        if row["uid"] != os.getuid() or stamp[:2] in source_inodes:
            raise ValueError("pool must be user-owned and independent from original source")
        if row["sha256"] != expected[rel]["sha256"] or _hash(path) != row["sha256"] or _stamp(path) != stamp:
            raise ValueError("pool content changed")
        result[rel] = path
    if _stamp(index) != index_stamp or _inventory(root) != before:
        raise ValueError("pool evidence changed during verification")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("source", type=Path)
    builder.add_argument("pool", type=Path)
    builder.add_argument("--donors", type=Path, required=True, help="JSON object: shard relative name to private donor path")
    builder.add_argument("--apply", action="store_true")
    verifier = sub.add_parser("verify")
    verifier.add_argument("index", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.pool, read_json(args.donors), apply=args.apply) if args.command == "build" else {
        "verified": True, "files": {key: str(value) for key, value in verify(args.index).items()},
        "source_full_hash_recomputed": False, "pool_full_hash_recomputed": True}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
