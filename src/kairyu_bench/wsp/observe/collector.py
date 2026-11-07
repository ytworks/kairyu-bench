"""Bounded live activation records, independent of vLLM at import time.

The only tensor operation in a model hook is a mutating custom operator. Its CUDA
implementation reduces each activation row directly, without an activation clone
or host synchronization. Request attribution and file I/O occur outside forward.
"""
from __future__ import annotations

import json
import hashlib
import gzip
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METRICS = ("mean", "rms", "max_abs", "nonfinite_count", "std", "zero_fraction",
           "outlier_fraction")
MAX_GPU_BYTES = 256 * 1024 * 1024


def selected_request(request_id: str) -> bool:
    """Only middleware-assigned requests; internal warmup IDs are excluded."""
    return re.match(r"^(?:(?:chatcmpl|cmpl)-)?observe-", request_id) is not None


def read_decode_offsets(value: Any) -> tuple[int, ...] | None:
    """Explicit relative positions are a bounded ordered selection, not a range."""
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32
            or any(type(offset) is not int or offset < 0 for offset in value)
            or any(a >= b for a, b in zip(value, value[1:]))):
        raise ValueError("decode_offsets must contain 1..32 ascending unique nonnegative integers")
    return tuple(value)


def read_prefill_policy(value: Any) -> str:
    if value not in ("head", "tail"):
        raise ValueError("prefill_policy must be head or tail")
    return str(value)


def sampling_metadata(prefill: int, decode: int,
                      decode_offsets: tuple[int, ...] | None = None,
                      prefill_policy: str = "head") -> dict[str, Any]:
    policy = ("first_execution_of_first_unique_positions" if decode_offsets is None
              else "first_execution_of_requested_decode_offsets")
    return {
        "policy": policy,
        "prefill_policy": ("first_execution_of_first_unique_positions"
                           if read_prefill_policy(prefill_policy) == "head"
                           else "first_execution_of_last_prompt_positions"),
        "decode_policy": policy,
        "requested_offsets": list(decode_offsets) if decode_offsets is not None else None,
        "offset_origin": "token_position_minus_prompt_length",
        "missing_policy": "unobserved_requested_offsets_are_missing_not_zero",
        "semantic_boundary": "thinking_code_boundary_not_measured",
        "limits": {"prefill": prefill, "decode": decode if decode_offsets is None else len(decode_offsets)},
    }


@dataclass(frozen=True)
class CaptureConfig:
    capture_id: str
    output_dir: Path
    checkpoint: dict[str, Any]
    runtime: dict[str, Any]
    gpu_bytes: int
    storage_bytes: int
    prefill_positions: int
    decode_positions: int
    outlier_rms_threshold: float = 6.0
    detail_channels: dict[str, tuple[int, ...]] = field(default_factory=dict)
    detail_heads: dict[str, tuple[int, ...]] = field(default_factory=dict)
    decode_offsets: tuple[int, ...] | None = None
    prefill_policy: str = "head"

    @classmethod
    def read(cls, path: str | Path) -> CaptureConfig:
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1 or data.get("observation_mode") != "live":
            raise ValueError("capture requires schema_version=1 and observation_mode=live")
        capture_id = data.get("capture_id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", capture_id):
            raise ValueError("invalid capture_id")
        out = Path(data["output_dir"])
        if not out.is_absolute():
            raise ValueError("capture output_dir must be absolute")
        limits = data["limits"]
        gpu = limits.get("gpu_buffer_mib", 256)
        storage = limits.get("storage_bytes", 2 * 1024**3)
        prefill = limits.get("prefill_positions", 32)
        decode = limits.get("decode_positions", 32)
        for name, value, maximum in (
            ("gpu_buffer_mib", gpu, 256),
            ("storage_bytes", storage, 2 * 1024**3),
            ("prefill_positions", prefill, 32),
            ("decode_positions", decode, 32),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        offsets = read_decode_offsets(data.get("decode_offsets"))
        if offsets is not None and len(offsets) > decode:
            raise ValueError("decode_offsets exceeds decode_positions limit")
        threshold = limits.get("outlier_rms_threshold", 6.0)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("outlier_rms_threshold must be finite and positive")
        runtime = dict(data["runtime"])
        for name in ("tp_size", "replicas"):
            if type(runtime.get(name)) is not int or runtime[name] < 1:
                raise ValueError(f"runtime.{name} must be a positive integer")
        return cls(capture_id, out, dict(data["checkpoint"]), runtime,
                   gpu * 1024**2, storage, prefill, decode, float(threshold),
                   read_detail_selection(data.get("detail_channels", {}), heads=False),
                   read_detail_selection(data.get("detail_heads", {}), heads=True), offsets,
                   read_prefill_policy(data.get("prefill_policy", "head")))


def read_detail_selection(value: Any, *, heads: bool) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, dict):
        raise ValueError("detail selection must map site IDs to indices")
    result = {}
    for site, indices in value.items():
        role = "attention" if heads else "(?:residual|attention|mlp)"
        if not isinstance(site, str) or not re.fullmatch(rf"layer\.\d+\.{role}", site):
            raise ValueError("detail selection has an invalid site ID")
        maximum = 8 if heads else 16
        if not isinstance(indices, list) or not 1 <= len(indices) <= maximum or any(
            type(index) is not int or not 0 <= index < 65536 for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("detail indices must be unique, nonnegative and bounded")
        result[site] = tuple(indices)
    if sum(map(len, result.values())) > (64 if heads else 128):
        raise ValueError("too many detailed sites requested")
    return result


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def observed_runtime(config: Any, runner: Any, *, v2: bool,
                     declaration: dict[str, Any]) -> dict[str, Any]:
    speculative = config.speculative_config
    configured_mtp = int(speculative.num_speculative_tokens) if speculative is not None else 0
    actual_mtp = int(runner.num_speculative_steps if v2 else runner.num_spec_tokens)
    if configured_mtp != actual_mtp or declaration.get("mtp_tokens") != actual_mtp:
        raise ValueError("capture MTP count differs from the actual runtime")
    return {
        "model_runner": "v2" if v2 else "v1",
        "mtp_tokens": actual_mtp,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "compilation_mode": str(config.compilation_config.mode),
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "mode_semantics": "resolved runtime configuration; per-batch graph dispatch may differ",
        "max_model_len": int(config.model_config.max_model_len),
        "max_num_batched_tokens": int(runner.max_num_tokens),
    }


def observe_checkpoint(model_path: str, declaration: dict[str, Any]) -> dict[str, Any]:
    """Bind a host prehash to the files actually mounted in this worker.

    No per-rank whole-model hash is performed. Exact file stamps must match the
    externally hashed, read-only checkpoint; both the declaration and observation
    remain present so this is not presented as a second independent byte hash.
    """
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("live capture requires a local checkpoint directory")
    expected_files = declaration.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("checkpoint.files must contain the launcher's prehashed file stamps")
    observed = []
    seen = set()
    for entry in expected_files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("invalid or duplicate checkpoint relative path")
        seen.add(str(relative))
        path = root / relative
        if not path.is_file():
            raise ValueError("prehashed checkpoint file is absent in the loaded model directory")
        stat = path.stat()
        stamp = {"device": stat.st_dev, "inode": stat.st_ino,
                 "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        expected_stamp = entry["stat"]
        if stat.st_size != entry["size"] or any(
            stamp[key] != expected_stamp[key] for key in ("inode", "mtime_ns", "ctime_ns")
        ):
            raise ValueError("loaded checkpoint stamps differ from the launch prehash; initialize from the matching mount")
        observed.append({"path": str(relative), "size": stat.st_size, "stat": stamp,
                         "prehashed_sha256": entry["sha256"]})
    fingerprint = hashlib.sha256(json.dumps(observed, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    return {"loaded_path": str(root), "verification": "prehashed_file_stamps_match",
            "observed_stamp_fingerprint": fingerprint, "files": observed,
            "per_worker_full_content_hash": False}


class CaptureWriter:
    """One worker owns one directory; a full quota never blocks generation."""

    def __init__(self, config: CaptureConfig, replica: str, rank: int,
                 sites: list[dict[str, Any]], buffer_bytes: int,
                 observed_checkpoint: dict[str, Any] | None = None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", replica):
            raise ValueError("invalid replica identifier")
        if not 0 <= rank < config.runtime["tp_size"]:
            raise ValueError("rank outside configured tensor parallel size")
        if buffer_bytes > config.gpu_bytes:
            raise ValueError("GPU capture buffer exceeds configured budget")
        self.config, self.replica, self.rank = config, replica, rank
        self.directory = config.output_dir / "workers" / f"{replica}-tp{rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        from kairyu_bench.wsp.observe.budget import allocations

        budgets = allocations(config.storage_bytes,
                              config.runtime["replicas"] * config.runtime["tp_size"],
                              config.runtime["replicas"])
        self.quota = budgets["worker_events"]
        self.metadata_quota = budgets["worker_metadata"]
        self.bytes_written = 0
        self.records_written = 0
        self.dropped_records = 0
        self.errors: dict[str, int] = {}
        self.error_details: list[dict[str, Any]] = []
        self.buffer_bytes = buffer_bytes
        self.closed = False
        self.quota_exhausted = False
        self.batches_written = 0
        self.extra_status: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._streams = {
            "observations": (self.directory / "observations.jsonl.gz").open("xb", buffering=1024 * 1024),
            "acceptance": (self.directory / "acceptance.jsonl").open("xb", buffering=1024 * 1024),
        }
        manifest = {
            "schema_version": 1, "capture_id": config.capture_id,
            "observation_mode": "live", "replica": replica, "rank": rank,
            "sites": sites, "checkpoint": config.checkpoint, "runtime": config.runtime,
            "observed_checkpoint": observed_checkpoint,
            "token_kind": "target", "draft_activation_collected": False,
            "sampling_policy": sampling_metadata(config.prefill_positions, config.decode_positions,
                                                   config.decode_offsets, config.prefill_policy)["policy"],
            "sampling": sampling_metadata(config.prefill_positions, config.decode_positions,
                                          config.decode_offsets, config.prefill_policy),
            "requested_offsets": list(config.decode_offsets) if config.decode_offsets is not None else None,
            "metrics_scope": "finite coordinates; nonfinite_count is separate",
            "outlier_rms_threshold": config.outlier_rms_threshold,
        }
        self.metadata_reserve = len(json.dumps(manifest, sort_keys=True).encode()) + 16384
        if self.metadata_reserve > self.metadata_quota:
            raise ValueError("storage quota cannot hold required capture metadata")
        atomic_json(self.directory / "sites.json", manifest)
        self.flush()

    def drop(self, count: int) -> None:
        with self._lock:
            self.dropped_records += count

    def write_encoded(self, kind: str, payload: bytes, logical_count: int) -> None:
        with self._lock:
            if self.closed or self.quota_exhausted or self.bytes_written + len(payload) > self.quota:
                self.dropped_records += logical_count
                self.quota_exhausted = self.quota_exhausted or not self.closed
                return
            try:
                self._streams[kind].write(payload)
            except OSError:
                self.dropped_records += logical_count
                self.errors["write_failed"] = self.errors.get("write_failed", 0) + 1
                return
            self.bytes_written += len(payload)
            self.records_written += logical_count
            if kind == "observations":
                self.batches_written += 1

    def write(self, kind: str, records: list[dict[str, Any]]) -> None:
        if self.quota_exhausted or self.closed:
            self.drop(len(records))
            return
        for record in records:
            line = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
            payload = gzip.compress(line, compresslevel=1, mtime=0) if kind == "observations" else line
            self.write_encoded(kind, payload, 1)

    def error(self, code: str) -> None:
        # Codes only: exception messages may contain prompts, tokens or paths.
        with self._lock:
            self.errors[code] = self.errors.get(code, 0) + 1

    def flush(self, *, closed: bool = False) -> dict[str, Any]:
        with self._lock:
            self.closed = self.closed or closed
            for stream in self._streams.values():
                if not stream.closed:
                    try:
                        stream.flush()
                        if closed:
                            stream.close()
                    except OSError:
                        self.errors["flush_failed"] = self.errors.get("flush_failed", 0) + 1
            status = {
                "schema_version": 1, "capture_id": self.config.capture_id,
                "replica": self.replica, "rank": self.rank, "closed": self.closed,
                "buffer_bytes": self.buffer_bytes, "storage_quota_bytes": self.quota,
                "metadata_reserved_bytes": self.metadata_reserve,
                "metadata_quota_bytes": self.metadata_quota,
                "bytes_written": self.bytes_written, "records_written": self.records_written,
                "batches_written": self.batches_written, "quota_exhausted": self.quota_exhausted,
                "observation_encoding": "activation_batch_gzip_members",
                **self.extra_status,
                "dropped_records": self.dropped_records, "errors": dict(self.errors),
                "error_details": list(self.error_details),
                "draft_activation_collected": False,
            }
            atomic_json(self.directory / "status.json", status)
            return status


@dataclass(frozen=True)
class Row:
    index: int
    request_id: str
    token_position: int
    phase: str
    speculative_index: int | None


class PositionSampler:
    """Bounded per-active-request state. Never samples padding as a request."""

    def __init__(self, prefill: int = 32, decode: int = 32,
                 decode_offsets: tuple[int, ...] | None = None,
                 prefill_policy: str = "head"):
        self.prefill_policy = read_prefill_policy(prefill_policy)
        self.decode_offsets = read_decode_offsets(decode_offsets)
        if self.decode_offsets is not None and len(self.decode_offsets) > decode:
            raise ValueError("decode_offsets exceeds decode_positions limit")
        self._decode_offset_set = frozenset(self.decode_offsets or ())
        self.limits = {"prefill": prefill,
                       "decode": decode if self.decode_offsets is None else len(self.decode_offsets)}
        self.selected: dict[str, dict[str, set[int]]] = {}

    def finish(self, request_ids: Any) -> None:
        for request_id in request_ids:
            self.selected.pop(request_id, None)

    def select(self, request_ids: list[str], counts: dict[str, int],
               positions: list[int], prompt_lengths: dict[str, int],
               draft_counts: dict[str, int], include: set[str] | None = None) -> list[Row]:
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate batch request IDs")
        if sum(counts[r] for r in request_ids) != len(positions):
            raise ValueError("token layout differs from scheduled batch")
        selected = []
        start = 0
        for req in request_ids:
            count, drafts = counts[req], draft_counts.get(req, 0)
            if not 0 <= drafts <= count:
                raise ValueError("invalid scheduled draft count")
            if include is not None and req not in include:
                start += count
                continue
            known = self.selected.setdefault(req, {"prefill": set(), "decode": set()})
            for offset in range(count):
                position = int(positions[start + offset])
                phase = "prefill" if position < prompt_lengths[req] else "decode"
                if (phase == "prefill" and self.prefill_policy == "tail"
                        and position < max(0, prompt_lengths[req] - self.limits["prefill"])):
                    continue
                if (phase == "decode" and self.decode_offsets is not None
                        and position - prompt_lengths[req] not in self._decode_offset_set):
                    continue
                used = known[phase]
                if position in used or len(used) >= self.limits[phase]:
                    continue
                used.add(position)
                speculative = offset - (count - drafts) if offset >= count - drafts else None
                selected.append(Row(start + offset, req, position, phase, speculative))
            start += count
        return selected


def v2_batch_rows(batch: Any, request_state: Any,
                  sampler: PositionSampler) -> tuple[list[Row], dict[str, int]]:
    """Read V2's actual device layout, including adaptive verification compaction."""
    requests = list(batch.req_ids)
    if len(requests) != batch.num_reqs:
        raise ValueError("invalid V2 request layout")
    starts = batch.query_start_loc[:batch.num_reqs + 1].detach().cpu().tolist()
    logits = batch.cu_num_logits[:batch.num_reqs + 1].detach().cpu().tolist()
    if starts[0] != 0 or any(b < a for a, b in zip(starts, starts[1:])):
        raise ValueError("invalid V2 query boundaries")
    end = int(starts[-1])
    if end > len(batch.positions) or end > len(batch.is_padding):
        raise ValueError("V2 query boundary exceeds input buffers")
    positions = batch.positions[:end].detach().cpu().tolist()
    padding = batch.is_padding[:end].detach().cpu().tolist()
    actual_indices = []
    included_requests: list[str] = []
    dense_positions: list[int] = []
    counts: dict[str, int] = {}
    lengths: dict[str, int] = {}
    drafts: dict[str, int] = {}
    for index, request_id in enumerate(requests):
        if not selected_request(request_id):
            continue
        included_requests.append(request_id)
        selected = [row for row in range(int(starts[index]), int(starts[index + 1]))
                    if not padding[row]]
        actual_indices.extend(selected)
        dense_positions.extend(int(positions[row]) for row in selected)
        counts[request_id] = len(selected)
        state_index = int(batch.idx_mapping_np[index])
        lengths[request_id] = int(request_state.prompt_len.np[state_index])
        # V2's cu_num_logits contains the actual admitted budget, unlike the
        # CPU scheduled-draft upper bound under adaptive verification.
        drafts[request_id] = max(0, int(logits[index + 1] - logits[index]) - 1)
    rows = sampler.select(included_requests, counts, dense_positions, lengths, drafts)
    return ([Row(actual_indices[row.index], row.request_id, row.token_position,
                 row.phase, row.speculative_index) for row in rows], drafts)


def acceptance_counts(output: Any, draft_counts: dict[str, int]) -> dict[str, int]:
    """Sampler accepted-prefix length, not the later EOS/stop-filtered output."""
    result = {}
    for request_id, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True):
        drafts = draft_counts.get(request_id, 0)
        if drafts and tokens and 0 <= len(tokens) - 1 <= drafts:
            result[request_id] = len(tokens) - 1
    return result


def row_record(config: CaptureConfig, replica: str, rank: int, step: int,
               site: dict[str, Any], row: Row, values: list[float],
               accepted_counts: dict[str, int]) -> dict[str, Any]:
    accepted: bool | None = True
    if row.speculative_index is not None:
        accepted = (row.speculative_index < accepted_counts[row.request_id]
                    if row.request_id in accepted_counts else None)
    values_out: dict[str, Any] = {}
    for name, value in zip(METRICS, values, strict=True):
        values_out[name] = (int(value) if math.isfinite(value) else None) if name == "nonfinite_count" else (
            float(value) if math.isfinite(value) else None)
    record = {
        "schema_version": 1, "capture_id": config.capture_id, "observation_mode": "live",
        "request_id": row.request_id, "replica": replica, "rank": rank, "step": step,
        "site_id": site["site_id"], "layer": site["layer"], "role": site["role"],
        "token_position": row.token_position, "phase": row.phase, "token_kind": "target",
        "speculative_index": row.speculative_index, "accepted": accepted,
        "acceptance_semantics": "rejection_sampler_prefix_not_stop_filtered",
        "metrics": values_out,
        "sample": {**sampling_metadata(config.prefill_positions, config.decode_positions,
                                       config.decode_offsets, config.prefill_policy),
                   "limit": config.prefill_positions if row.phase == "prefill"
                   else (config.decode_positions if config.decode_offsets is None
                         else len(config.decode_offsets))},
    }
    for name in ("detail_kind", "parent_site_id", "channel_index", "head_local_index",
                 "head_global_index", "head_dim", "activation_axis"):
        if name in site:
            record[name] = site[name]
    if site.get("detail_kind") == "channel":
        values_out["value"] = values_out["mean"]
    return record


def install_row_operator() -> Any:
    """Register the optional PyTorch op once; imports require the serving extra."""
    import torch

    if hasattr(torch.ops.kairyu_observe, "row_moments"):
        return torch.ops.kairyu_observe.row_moments

    @torch.library.custom_op("kairyu_observe::row_moments", mutates_args=("output",),
                             schema="(Tensor x, Tensor(a!) output, float threshold) -> ()")
    def row_moments(x: Any, output: Any, threshold: float) -> None:
        if x.ndim != 2 or output.ndim != 2 or output.shape[1] != len(METRICS):
            raise ValueError("row moments require [tokens, channels] and [capacity, metrics]")
        if x.shape[0] > output.shape[0]:
            raise ValueError("activation row count exceeds fixed capture buffer")
        if x.is_cuda:
            _cuda_row_moments(x, output, threshold)
        else:
            # CPU reference path for small synthetic models, never live Gemma.
            finite = torch.isfinite(x)
            safe = torch.where(finite, x, 0).double()
            count = finite.sum(dim=1)
            denominator = count.clamp_min(1)
            output[:x.shape[0], 0].copy_(safe.sum(dim=1) / denominator)
            output[:x.shape[0], 1].copy_((safe.square().sum(dim=1) / denominator).sqrt())
            output[:x.shape[0], 2].copy_(safe.abs().amax(dim=1))
            output[:x.shape[0], 3].copy_((~finite).sum(dim=1))
            mean = safe.sum(dim=1) / denominator
            rms = (safe.square().sum(dim=1) / denominator).sqrt()
            output[:x.shape[0], 4].copy_(
                (safe.square().sum(dim=1) / denominator - mean.square()).clamp_min(0).sqrt())
            output[:x.shape[0], 5].copy_(((safe == 0) & finite).sum(dim=1) / denominator)
            output[:x.shape[0], 6].copy_(
                ((safe.abs() > threshold * rms[:, None]) & finite).sum(dim=1) / denominator)
            if x.shape[1] == 1:
                output[:x.shape[0], 0].copy_(x[:, 0].float())
                output[:x.shape[0], 1].copy_(x[:, 0].abs().float())
                output[:x.shape[0], 4].zero_()
            for column in (0, 1, 2, 4, 5, 6):
                output[:x.shape[0], column].masked_fill_(count == 0, float("nan"))

    @row_moments.register_fake
    def fake_row_moments(x: Any, output: Any, threshold: float) -> None:
        return None

    return row_moments


def _cuda_row_moments(x: Any, output: Any, threshold: float) -> None:
    import triton

    from kairyu_bench.wsp.observe.gpu_kernel import row_moments_kernel

    kernel = row_moments_kernel
    width = x.shape[1]
    if width < 1 or width > 65536:
        raise ValueError("unsupported activation width")
    kernel[(x.shape[0],)](
        x, output, width, x.stride(0), x.stride(1), output.stride(0), threshold,
        BLOCK=triton.next_power_of_2(width),
    )




def gemma_checkpoint_mapping(runtime_names: list[str], layer: int,
                             checkpoint_names: set[str]) -> list[str]:
    """Explicit Gemma4 fusion map, intersected with the actual checkpoint index."""
    prefix = f"model.language_model.layers.{layer}."
    marker = f".layers.{layer}."
    mapped: list[str] = []
    for name in runtime_names:
        if marker not in "." + name:
            continue
        tail = ("." + name).split(marker, 1)[1]
        if tail == "self_attn.qkv_proj.weight":
            tails = [f"self_attn.{projection}_proj.weight" for projection in ("q", "k", "v")]
        elif tail == "mlp.gate_up_proj.weight":
            tails = ["mlp.gate_proj.weight", "mlp.up_proj.weight"]
        else:
            tails = [tail]
        mapped.extend(prefix + tail for tail in tails if prefix + tail in checkpoint_names)
    return sorted(set(mapped))


class TensorCapture:
    """One fixed device buffer per site; hooks return None and preserve outputs."""

    def __init__(self, model: Any, config: CaptureConfig, capacity: int,
                 checkpoint_names: set[str] | None = None):
        import torch

        self.config = config
        self.operator = install_row_operator()
        self.sites: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        modules: list[tuple[Any, str, int | None, int | None]] = []
        found_base_sites: set[str] = set()
        for name, layer in model.named_modules():
            if layer.__class__.__name__ != "Gemma4DecoderLayer":
                continue
            match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
            if match is None:
                raise ValueError("unrecognized Gemma decoder layer name")
            index = int(match.group(1))
            for role, subname, module in (
                ("residual", name, layer),
                ("attention", f"{name}.self_attn", layer.self_attn),
                ("mlp", f"{name}.mlp", layer.mlp),
            ):
                names = [f"{subname}.{key}" for key, _ in module.named_parameters()]
                self.sites.append({
                    "site_id": f"layer.{index}.{role}", "module": subname,
                    "layer": index, "role": role, "runtime_parameter_names": names,
                    "weight_names": gemma_checkpoint_mapping(names, index, checkpoint_names or set()),
                    "checkpoint_weight_names": gemma_checkpoint_mapping(names, index, checkpoint_names or set()),
                    "weight_mapping": "gemma4_fusion_map_filtered_to_checkpoint_index" if checkpoint_names else "runtime_names_only",
                    "tp_layout": "rank_local; no cross-rank aggregation",
                    "semantics": {"residual": "block output after layer scalar",
                                  "attention": "attention output before post-attention norm",
                                  "mlp": "MLP output before post-feedforward norm"}[role],
                })
                parent = self.sites[-1]
                found_base_sites.add(parent["site_id"])
                modules.append((module, "post", None, None))
                for channel in config.detail_channels.get(parent["site_id"], ()):
                    width = getattr(layer, "hidden_size", 0)
                    if not 0 <= channel < width:
                        raise ValueError("selected channel is outside the decoder hidden axis")
                    axes = []
                    projection = {"attention": "self_attn.o_proj.weight",
                                  "mlp": "mlp.down_proj.weight"}.get(role)
                    if projection:
                        key = f"model.language_model.layers.{index}.{projection}"
                        if key in (checkpoint_names or set()):
                            axes.append({"tensor": key, "axis": 0, "start": channel,
                                         "stop": channel + 1})
                    self.sites.append(dict(
                        parent, site_id=f"{parent['site_id']}.channel.{channel}",
                        parent_site_id=parent["site_id"], role="channel",
                        detail_kind="channel", channel_index=channel,
                        activation_axis="decoder_hidden_last_dimension",
                        checkpoint_axes=axes, semantics="selected output coordinate; value is the actual scalar",
                    ))
                    modules.append((module, "post", channel, channel + 1))
            parent_id = f"layer.{index}.attention"
            for head in config.detail_heads.get(parent_id, ()):
                attn = layer.self_attn
                if not 0 <= head < attn.num_heads:
                    raise ValueError("selected attention head is outside the TP-local query-head axis")
                global_head = attn.tp_rank * attn.num_heads + head
                dimension = attn.head_dim
                key = f"model.language_model.layers.{index}.self_attn.o_proj.weight"
                axes = ([{"tensor": key, "axis": 1, "start": global_head * dimension,
                          "stop": (global_head + 1) * dimension}]
                        if key in (checkpoint_names or set()) else [])
                parent = next(site for site in self.sites if site["site_id"] == parent_id)
                self.sites.append(dict(
                    parent, site_id=f"{parent_id}.head.{global_head}",
                    module=f"{name}.self_attn.o_proj", parent_site_id=parent_id,
                    role="attention_head", detail_kind="attention_head",
                    head_local_index=head, head_global_index=global_head,
                    head_dim=dimension, activation_axis="pre_o_proj_query_head_concat",
                    checkpoint_axes=axes,
                    tp_layout="contiguous query-head partition; global=tp_rank*local_heads+local",
                    semantics="attention output head before o_proj and TP all-reduce; no attention probability matrix",
                ))
                modules.append((attn.o_proj, "pre", head * dimension, (head + 1) * dimension))
        if (set(config.detail_channels) | set(config.detail_heads)) - found_base_sites:
            raise ValueError("detail selection refers to a decoder site absent from the model")
        if not self.sites:
            raise ValueError("live v1 adapter requires Gemma4DecoderLayer")
        if len({site["site_id"] for site in self.sites}) != len(self.sites):
            raise ValueError("ambiguous decoder layer site IDs")
        self.capacity = capacity
        self.copy_rows = min(256, capacity)
        self.buffer_bytes = len(self.sites) * (capacity + self.copy_rows) * len(METRICS) * 4
        self.buffer_bytes += self.copy_rows * 8  # gather indices
        if capacity < 1 or self.buffer_bytes > config.gpu_bytes:
            raise ValueError("fixed capture buffers exceed GPU memory budget")
        device = next(model.parameters()).device
        self.buffer = torch.empty((len(self.sites), capacity, len(METRICS)),
                                  dtype=torch.float32, device=device)
        self.copy_buffer = torch.empty((len(self.sites), self.copy_rows, len(METRICS)),
                                       dtype=torch.float32, device=device)
        self.index_buffer = torch.empty(self.copy_rows, dtype=torch.int64, device=device)
        for index, (module, kind, start, stop) in enumerate(modules):
            if kind == "pre":
                assert start is not None and stop is not None
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(index, start, stop)))
            else:
                self.handles.append(module.register_forward_hook(self._hook(index, start, stop)))

    def _hook(self, index: int, start: int | None = None,
              stop: int | None = None) -> Any:
        output_buffer, operator = self.buffer[index], self.operator
        threshold = self.config.outlier_rms_threshold

        def hook(module: Any, args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if start is not None:
                tensor = tensor[:, start:stop]
            operator(tensor, output_buffer, threshold)

        return hook

    def _pre_hook(self, index: int, start: int, stop: int) -> Any:
        output_buffer, operator = self.buffer[index], self.operator
        threshold = self.config.outlier_rms_threshold

        def hook(module: Any, args: Any) -> None:
            operator(args[0][:, start:stop], output_buffer, threshold)

        return hook

    def reset(self) -> None:
        # This is outside model forward and CUDA graph capture. Missing hooks
        # remain NaN instead of silently reusing a previous batch.
        self.buffer.fill_(float("nan"))

    def read(self, rows: list[Row]) -> Any:
        import numpy as np
        import torch

        chunks = []
        for start in range(0, len(rows), self.copy_rows):
            part = rows[start:start + self.copy_rows]
            indices = torch.tensor([row.index for row in part], dtype=torch.int64)
            self.index_buffer[:len(part)].copy_(indices)
            torch.index_select(self.buffer, 1, self.index_buffer[:len(part)],
                               out=self.copy_buffer[:, :len(part)])
            # Preserve packed float32 storage. Python numbers and JSON are
            # created only by the bounded CPU writer, never on this caller.
            host = self.copy_buffer[:, :len(part)].to("cpu", copy=True).numpy()
            missing = ~np.isfinite(host[:, :, 3])
            if missing.any():
                site_index = int(np.where(missing.any(axis=1))[0][0])
                error = ValueError("capture hook did not write selected rows")
                error.capture_diagnostic = {  # type: ignore[attr-defined]
                    "site_id": self.sites[site_index]["site_id"],
                    "selected_rows": len(part), "min_row": min(row.index for row in part),
                    "max_row": max(row.index for row in part),
                    "missing_rows": int(missing[site_index].sum()),
                }
                raise error
            chunks.append(host)
        if not chunks:
            return np.empty((len(self.sites), 0, len(METRICS)), dtype=np.float32)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=1)
