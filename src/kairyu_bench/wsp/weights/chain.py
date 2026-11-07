"""Verify sequential native edits from original bytes without parent checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .checkpoint import Checkpoint
from .editor import CHUNK, _check_full_unit_source, _inventory, _stamp, read_json, validate_manifest

VERSION = "native-edit-chain-v1"


def _files(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = record["files"]
    result = {}
    for row in rows:
        name = row["path"]
        path = Path(name)
        if (not isinstance(name, str) or path.is_absolute() or ".." in path.parts
                or not path.parts or name in result):
            raise ValueError("invalid or duplicate retained file path")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("invalid retained file size")
        for key in ("source_sha256", "candidate_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("chain requires full source/candidate file hashes")
        result[name] = row
    return result


def verify_chain(original: Path, candidate: Path, manifests: Sequence[Path], *,
                 chunk_bytes: int = CHUNK) -> dict[str, Any]:
    """Replay each saved byte transition in bounded buffers; never write weights.

    Manifest metadata remains in memory. Payload buffers are bounded by the chunk
    size; intermediate checkpoint directories need not exist. Every stage hash,
    byte precondition and final candidate byte is checked in the same scan.
    """
    if not manifests or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024**2:
        raise ValueError("provide a nonempty chain and chunk size in 1..64 MiB")
    implementation = [Path(__file__).with_name(name) for name in ("chain.py", "editor.py", "checkpoint.py")]
    implementation_stamps = [_stamp(path) for path in implementation]
    implementation_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}
    original, candidate = original.resolve(), candidate.resolve()
    if original == candidate or original.is_relative_to(candidate) or candidate.is_relative_to(original):
        raise ValueError("original and candidate must be separate directories")
    original_before, candidate_before = _inventory(original), _inventory(candidate)
    if original_before.keys() != candidate_before.keys():
        raise ValueError("original/candidate inventories differ")
    checkpoint = Checkpoint(original)
    Checkpoint(candidate).assert_stable()
    paths = [path.resolve() for path in manifests]
    stamps = [_stamp(path) for path in paths]
    records = [read_json(path) for path in paths]
    manifest_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    inventories = []
    stages: list[dict[str, list[tuple[int, bytes, bytes]]]] = []
    expected_parent = original
    for record in records:
        validate_manifest(record)
        _check_full_unit_source(checkpoint, record["spec"])
        if Path(record["source"]).resolve() != expected_parent:
            raise ValueError("chain source path does not match preceding candidate")
        expected_parent = Path(record["candidate"]).resolve()
        files = _files(record)
        if files.keys() != original_before.keys():
            raise ValueError("chain retained file inventory differs")
        for name, row in files.items():
            if row["size"] != original_before[name][2] or row["size"] != candidate_before[name][2]:
                raise ValueError("chain retained file size differs")
            if inventories and inventories[-1][name]["candidate_sha256"] != row["source_sha256"]:
                raise ValueError("chain full-file hash continuity differs")
        inventories.append(files)
        stage: dict[str, list[tuple[int, bytes, bytes]]] = {}
        for patch in record["edits"]:
            tensor = checkpoint.tensors[patch["tensor"]]
            matrix = "matrix_coordinate" in patch
            offset = tensor.offset + (2 * patch["selected_indices"][0] if matrix else 0)
            size = 2 if matrix else tensor.nbytes
            if (patch["shard"] != tensor.relative_shard or patch["offset"] != offset
                    or patch["size"] != size or patch["dtype"] != tensor.dtype
                    or patch["shape"] != list(tensor.shape)):
                raise ValueError("chain patch differs from original tensor headers")
            stage.setdefault(patch["shard"], []).append(
                (offset, bytes.fromhex(patch["before_hex"]), bytes.fromhex(patch["after_hex"])))
        for patches in stage.values():
            patches.sort(key=lambda row: row[0])
        stages.append(stage)
    if expected_parent != candidate:
        raise ValueError("final chain path does not match candidate")
    file_results = []
    for name, stamp in original_before.items():
        hashes = [hashlib.sha256() for _ in range(len(stages) + 1)]
        patches_by_stage = [stage.get(name, []) for stage in stages]
        cursors = [0] * len(stages)
        position = 0
        with (original / name).open("rb") as source, (candidate / name).open("rb") as dest:
            while raw := source.read(chunk_bytes):
                hashes[0].update(raw)
                block = bytearray(raw)
                end = position + len(block)
                for index, patches in enumerate(patches_by_stage):
                    cursor = cursors[index]
                    while cursor < len(patches) and patches[cursor][0] < end:
                        offset, before, after = patches[cursor]
                        start, stop = max(position, offset), min(end, offset + len(before))
                        local = slice(start - position, stop - position)
                        span = slice(start - offset, stop - offset)
                        if block[local] != before[span]:
                            raise ValueError(f"chain before bytes differ: stage {index + 1}, {name}")
                        block[local] = after[span]
                        if offset + len(before) > end:
                            break
                        cursor += 1
                    cursors[index] = cursor
                    hashes[index + 1].update(block)
                if dest.read(len(block)) != block:
                    raise ValueError(f"replayed bytes differ from candidate: {name}")
                position = end
            if dest.read(1) or position != stamp[2]:
                raise ValueError("checkpoint size changed during replay")
        if any(cursor != len(patches) for cursor, patches in zip(cursors, patches_by_stage, strict=True)):
            raise ValueError("chain patch extends beyond retained file")
        digests = [digest.hexdigest() for digest in hashes]
        for index, files in enumerate(inventories):
            if (files[name]["source_sha256"] != digests[index]
                    or files[name]["candidate_sha256"] != digests[index + 1]):
                raise ValueError(f"replayed full-file hash differs: stage {index + 1}, {name}")
        file_results.append({"path": name, "size": stamp[2], "stage_sha256": digests})
    if (_inventory(original) != original_before or _inventory(candidate) != candidate_before
            or [_stamp(path) for path in paths] != stamps
            or [_stamp(path) for path in implementation] != implementation_stamps):
        raise ValueError("chain inputs changed during verification")
    return {"verified": True, "reconstructor_version": VERSION,
            "reconstructor_sha256": implementation_hashes["chain.py"],
            "implementation_sha256": implementation_hashes,
            "original": str(original), "candidate": str(candidate), "stages": len(stages),
            "manifests": [{"path": str(path), "sha256": digest}
                          for path, digest in zip(paths, manifest_hashes, strict=True)],
            "files": file_results, "chunk_bytes": chunk_bytes, "checkpoint_writes": 0,
            "rounding": "each recorded saved-dtype before/after transition, in order",
            "scope": "all regular files except .cache and .kairyu-model-attestation.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="Native edit manifest, repeated in chronological order")
    args = parser.parse_args()
    try:
        result = verify_chain(args.original, args.candidate, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"native edit chain: {error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
