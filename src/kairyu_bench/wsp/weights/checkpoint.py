"""Bounded, read-only safetensors access without loading model code."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from contextlib import contextmanager
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_READ_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPES = {
    "BF16": ("<u2", 2), "F16": ("<f2", 2), "F32": ("<f4", 4), "F64": ("<f8", 8),
    "BOOL": ("?", 1), "I8": ("i1", 1), "U8": ("u1", 1),
    "I16": ("<i2", 2), "U16": ("<u2", 2), "I32": ("<i4", 4), "U32": ("<u4", 4),
    "I64": ("<i8", 8), "U64": ("<u8", 8),
}
FLOAT_DTYPES = frozenset(("BF16", "F16", "F32", "F64"))
CONFIG_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
                "tokenizer.json", "special_tokens_map.json", "chat_template.jinja",
                "processor_config.json", "preprocessor_config.json", "tokenizer.model",
                "spiece.model", "vocab.json", "merges.txt", "added_tokens.json")


class CheckpointError(ValueError):
    """Unsupported or inconsistent checkpoint input."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError("invalid JSON metadata") from error
    if not isinstance(value, dict):
        raise CheckpointError("metadata must be an object")
    return value


def _stat_tuple(value: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(value.st_mode):
        raise CheckpointError("checkpoint input must be a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    return _stat_tuple(path.stat())


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: Path
    offset: int
    nbytes: int
    relative_shard: str = ""

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def floating(self) -> bool:
        return self.dtype in FLOAT_DTYPES

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape),
                "shard": self.relative_shard or self.shard.name, "offset": self.offset, "nbytes": self.nbytes,
                "numel": self.numel, "floating": self.floating}


class Checkpoint:
    """A single safetensors file or an HF directory/index, with immutable inputs.

    Individual reads are capped at 64 MiB. Iteration is bounded by chunk_bytes;
    callers may hold their own larger analysis arrays, independently of this reader.
    Identity SHA-256 covers every referenced shard and recognized config file.
    """

    def __init__(self, path: str | Path):
        supplied = Path(path).expanduser()
        if not supplied.exists():
            raise CheckpointError(f"checkpoint does not exist: {supplied}")
        self.path = supplied.resolve()
        self.root = self.path if self.path.is_dir() else self.path.parent
        self.tensors: dict[str, TensorInfo] = {}
        self._stamps: dict[Path, tuple[int, int, int, int, int]] = {}
        self.bytes_read = 0
        self.read_counts = {"metadata": 0, "tensor_payload": 0, "identity_hash": 0}
        self.input_scope = "single_file" if self.path.is_file() else "directory"
        self._config_names = self._configuration_names()
        self.index: Path | None = None
        self.configuration_files: list[Path] = []
        if self.path.is_dir():
            index = self.path / "model.safetensors.index.json"
            files = sorted(self.path.glob("*.safetensors"))
            if index.exists():
                entry = index
            elif len(files) == 1:
                entry = files[0]
            else:
                raise CheckpointError("directory needs one safetensors file or model.safetensors.index.json")
        else:
            entry = self.path
        if entry.name.endswith(".safetensors.index.json"):
            self.index = self._inside(entry)
            self._track(self.index)
            if self.index.stat().st_size > MAX_HEADER_BYTES:
                raise CheckpointError("index exceeds 16 MiB metadata limit")
            index_data = _json(self._read_file(self.index))
            mapping = index_data.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise CheckpointError("index needs a nonempty weight_map")
            if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                   for k, v in mapping.items()):
                raise CheckpointError("invalid weight_map names")
            resolved = {}
            for name, filename in mapping.items():
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                    raise CheckpointError("index shard must be a relative safetensors path")
                resolved[name] = self._inside(self.root / relative)
            self.input_scope = "indexed_checkpoint"
            shards = sorted(set(resolved.values()))
            for shard in shards:
                self._header(shard)
            if set(self.tensors) != set(resolved):
                raise CheckpointError("index and shard tensor names differ")
            if any(self.tensors[name].shard != shard for name, shard in resolved.items()):
                raise CheckpointError("index points a tensor at the wrong shard")
            metadata = index_data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise CheckpointError("index metadata must be an object")
            if "total_size" in metadata:
                total = metadata["total_size"]
                if type(total) is not int or total != sum(t.nbytes for t in self.tensors.values()):
                    raise CheckpointError("index total_size differs from tensor payload sizes")
        elif entry.suffix == ".safetensors":
            self._header(self._inside(entry))
        else:
            raise CheckpointError("only safetensors and safetensors indexes are supported")
        for name in CONFIG_FILES:
            file = self.root / name
            if file.exists():
                file = self._inside(file)
                self._track(file)
                self.configuration_files.append(file)
        self.tensors = dict(sorted(self.tensors.items()))
        self.assert_stable()

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise CheckpointError("checkpoint file escapes its directory")
        if path.is_symlink():
            raise CheckpointError("checkpoint files must be regular files, not symlinks")
        return resolved

    def _track(self, path: Path) -> None:
        current = _stamp(path)
        previous = self._stamps.setdefault(path, current)
        if previous != current:
            raise CheckpointError(f"source changed: {path.name}")

    def _configuration_names(self) -> tuple[str, ...]:
        names = (*CONFIG_FILES, "model.safetensors.index.json")
        return tuple(name for name in names if (self.root / name).exists())

    def _count(self, kind: str, size: int) -> None:
        self.bytes_read += size
        self.read_counts[kind] += size

    @contextmanager
    def _open_stable(self, path: Path) -> Iterator[BinaryIO]:
        self._track(path)
        with path.open("rb") as stream:
            if _stat_tuple(os.fstat(stream.fileno())) != self._stamps[path]:
                raise CheckpointError(f"source changed while opening: {path.name}")
            yield stream
            if _stat_tuple(os.fstat(stream.fileno())) != self._stamps[path]:
                raise CheckpointError(f"source changed while reading: {path.name}")
        self._track(path)

    def _read_file(self, path: Path) -> bytes:
        with self._open_stable(path) as stream:
            data = stream.read(MAX_HEADER_BYTES + 1)
        if len(data) > MAX_HEADER_BYTES:
            raise CheckpointError("metadata exceeds 16 MiB limit")
        self._count("metadata", len(data))
        return data

    def _header(self, path: Path) -> None:
        self._track(path)
        with self._open_stable(path) as stream:
            prefix = stream.read(8)
            self._count("metadata", len(prefix))
            if len(prefix) != 8:
                raise CheckpointError("truncated safetensors prefix")
            length = struct.unpack("<Q", prefix)[0]
            if length < 2 or length > MAX_HEADER_BYTES or length + 8 > self._stamps[path][2]:
                raise CheckpointError("invalid or oversized safetensors header")
            raw = stream.read(length)
            self._count("metadata", len(raw))
            if len(raw) != length:
                raise CheckpointError("truncated safetensors header")
        header = _json(raw)
        spans = []
        for name, value in header.items():
            if name == "__metadata__":
                if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                      for k, v in value.items()):
                    raise CheckpointError("invalid safetensors metadata")
                continue
            if not name or name in self.tensors or not isinstance(value, dict):
                raise CheckpointError("invalid or duplicate tensor name")
            dtype, shape, offsets = value.get("dtype"), value.get("shape"), value.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in DTYPES:
                raise CheckpointError(f"unsupported dtype: {dtype}")
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise CheckpointError("invalid tensor shape")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(d) is not int or d < 0 for d in offsets)):
                raise CheckpointError("invalid tensor offsets")
            start, end = offsets
            expected = math.prod(shape) * DTYPES[dtype][1]
            if end < start or end - start != expected or 8 + length + end > self._stamps[path][2]:
                raise CheckpointError("tensor shape, byte size, or file bounds differ")
            self.tensors[name] = TensorInfo(name, dtype, tuple(shape), path, 8 + length + start,
                                            expected, str(path.relative_to(self.root)))
            spans.append((start, end))
        position = 0
        for start, end in sorted(spans):
            if start != position:
                raise CheckpointError("safetensors payload has overlapping or unassigned bytes")
            position = end
        if 8 + length + position != self._stamps[path][2]:
            raise CheckpointError("safetensors payload has trailing unassigned bytes")
        self._track(path)

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(sorted(self._stamps))

    def assert_stable(self) -> None:
        if self._configuration_names() != self._config_names:
            raise CheckpointError("checkpoint configuration/index file inventory changed")
        for path in self._stamps:
            self._track(path)

    def read_raw(self, name: str, start: int = 0, count: int | None = None) -> bytes:
        if name not in self.tensors:
            raise CheckpointError(f"unknown tensor: {name}")
        tensor = self.tensors[name]
        if type(start) is not int:
            raise CheckpointError("read start must be an integer")
        count = tensor.numel - start if count is None else count
        if type(start) is not int or type(count) is not int or start < 0 or count < 0 or start + count > tensor.numel:
            raise CheckpointError("read range is outside tensor")
        length = count * DTYPES[tensor.dtype][1]
        if length > MAX_READ_BYTES:
            raise CheckpointError("individual reads exceed 64 MiB; use iter_chunks")
        with self._open_stable(tensor.shard) as stream:
            stream.seek(tensor.offset + start * DTYPES[tensor.dtype][1])
            data = stream.read(length)
        self._count("tensor_payload", len(data))
        if len(data) != length:
            raise CheckpointError("source truncated during read")
        self._track(tensor.shard)
        return data

    @staticmethod
    def decode(raw: bytes, dtype: str) -> np.ndarray:
        values = np.frombuffer(raw, dtype=DTYPES[dtype][0])
        if dtype == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        return values

    def read(self, name: str, start: int = 0, count: int | None = None) -> np.ndarray:
        return self.decode(self.read_raw(name, start, count), self.tensors[name].dtype)

    def iter_chunks(self, name: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[tuple[int, bytes, np.ndarray]]:
        if type(chunk_bytes) is not int or not 8 <= chunk_bytes <= MAX_READ_BYTES:
            raise CheckpointError("chunk_bytes must be between 8 bytes and 64 MiB")
        tensor = self.tensors[name]
        elements = max(1, chunk_bytes // DTYPES[tensor.dtype][1])
        for start in range(0, tensor.numel, elements):
            raw = self.read_raw(name, start, min(elements, tensor.numel - start))
            yield start, raw, self.decode(raw, tensor.dtype)
        self.assert_stable()

    def estimated_io(self, payload_bytes: int, hash_files: bool = True) -> dict[str, int]:
        hash_bytes = sum(value[2] for value in self._stamps.values()) if hash_files else 0
        return {"metadata_bytes": self.read_counts["metadata"],
                "tensor_payload_bytes": payload_bytes, "identity_hash_bytes": hash_bytes,
                "total_bytes": self.read_counts["metadata"] + payload_bytes + hash_bytes}

    def identity(self, hash_files: bool = True, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> dict[str, Any]:
        if type(chunk_bytes) is not int or not 8 <= chunk_bytes <= MAX_READ_BYTES:
            raise CheckpointError("chunk_bytes must be between 8 bytes and 64 MiB")
        self.assert_stable()
        files = []
        for path, stamp in sorted(self._stamps.items()):
            digest = hashlib.sha256()
            if hash_files:
                with self._open_stable(path) as stream:
                    while block := stream.read(chunk_bytes):
                        digest.update(block)
                        self._count("identity_hash", len(block))
            self._track(path)
            files.append({"path": str(path.relative_to(self.root)), "size": stamp[2],
                          "sha256": digest.hexdigest() if hash_files else None,
                          "stat": {"device": stamp[0], "inode": stamp[1], "mtime_ns": stamp[3],
                                   "ctime_ns": stamp[4]}})
        self.assert_stable()
        canonical = [{k: f[k] for k in ("path", "size", "sha256")} for f in files]
        tree = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"path": str(self.path), "files": files,
                "sha256": tree if hash_files else None, "hash_files": hash_files,
                "identity_level": "content_sha256" if hash_files else "stat_only",
                "input_scope": self.input_scope, "complete_model_claimed": False,
                "configuration_names_considered": list(CONFIG_FILES),
                "scope": "all referenced shards, index, and recognized configuration/tokenizer files",
                "source_stable": True}
