"""Opt-in vLLM 0.33 Gemma4 worker selected with --worker-cls.

This adapter subclasses the documented worker selection point. It does not
replace global functions, modify model parameters, call extra forwards, or expose
a general RPC endpoint. Start the serving process with the capture config present.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.observe.collector import (
    CaptureConfig,
    CaptureWriter,
    PositionSampler,
    Row,
    TensorCapture,
    acceptance_counts,
    observe_checkpoint,
    observed_runtime,
    v2_batch_rows,
    selected_request,
)

# This module is imported only in the opt-in serving image, not in the host CLI.
from kairyu_bench.wsp.observe.collector_io import RecordQueue

from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker


@dataclass
class Pending:
    step: int
    rows: list[Row]
    values: Any
    drafts: dict[str, int]


class ObservedAsyncOutput(AsyncModelRunnerOutput):
    """Preserve the engine's single get_output call and its original result."""

    def __init__(self, delegate: Any, callback: Any):
        self.delegate = delegate
        self.callback = callback

    def get_output(self) -> Any:
        output = self.delegate.get_output()
        self.callback(output)
        return output

    def get_kv_connector_output(self) -> Any:
        return self.delegate.get_kv_connector_output()


class CaptureWorker(Worker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        config_path = os.environ.get("KAIRYU_BENCH_CAPTURE_CONFIG")
        if not config_path:
            raise ValueError("CaptureWorker requires KAIRYU_BENCH_CAPTURE_CONFIG")
        import vllm

        if not vllm.__version__.startswith("0.33."):
            raise ValueError("live adapter is pinned to vLLM 0.33")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("live adapter currently supports PP=1 only")
        if getattr(self.parallel_config, "enable_dbo", False):
            raise ValueError("live adapter does not support dual-batch overlap")
        if getattr(self.parallel_config, "decode_context_parallel_size", 1) != 1:
            raise ValueError("live adapter does not support context parallelism")
        if not self.use_v2_model_runner and self.model_runner.uses_mrope:
            raise ValueError("live adapter requires one-dimensional token positions")
        if self.use_v2_model_runner:
            if self.model_runner.pcp_manager is not None:
                raise ValueError("live adapter does not support prefill context parallelism")
            if self.model_runner.model_state.num_new_sampled_tokens_per_step != 1:
                raise ValueError("live adapter requires one bonus token per sampling step")
        self._capture_config = CaptureConfig.read(config_path)
        if self._capture_config.runtime["tp_size"] != self.parallel_config.tensor_parallel_size:
            raise ValueError("capture TP size differs from serving configuration")
        self._capture_replica = os.environ.get("KAIRYU_BENCH_CAPTURE_REPLICA", socket.gethostname())
        from vllm.distributed import get_tensor_model_parallel_rank

        self._capture_rank = get_tensor_model_parallel_rank()
        observed_checkpoint = observe_checkpoint(self.model_config.model, self._capture_config.checkpoint)
        observed_checkpoint["dtype"] = str(self.model_config.dtype)
        observed_checkpoint["quantization"] = self.model_config.quantization
        observed_checkpoint["vllm_version"] = vllm.__version__
        observed_checkpoint.update(observed_runtime(
            self.vllm_config, self.model_runner, v2=self.use_v2_model_runner,
            declaration=self._capture_config.runtime))
        checkpoint_names: set[str] = set()
        index = Path(self.model_config.model) / "model.safetensors.index.json"
        if index.is_file():
            checkpoint_names = set(json.loads(index.read_text())["weight_map"])
        self._capture_tensor = TensorCapture(
            self.get_model(), self._capture_config, self.model_runner.max_num_tokens,
            checkpoint_names)
        self._capture_writer = CaptureWriter(
            self._capture_config, self._capture_replica, self._capture_rank,
            self._capture_tensor.sites, self._capture_tensor.buffer_bytes, observed_checkpoint)
        self._capture_queue = RecordQueue(self._capture_writer, self._capture_tensor.sites)
        self._capture_sampler = PositionSampler(
            self._capture_config.prefill_positions, self._capture_config.decode_positions,
            self._capture_config.decode_offsets)
        self._capture_pending: Pending | None = None
        self._capture_step = 0
        self._capture_disabled = False
        atexit.register(self._capture_close)

    def _capture_close(self) -> None:
        writer = getattr(self, "_capture_writer", None)
        if writer is not None:
            try:
                record_queue = getattr(self, "_capture_queue", None)
                if record_queue is not None:
                    record_queue.close()
                else:
                    writer.flush(closed=True)
            except OSError:
                pass

    def shutdown(self) -> None:
        try:
            self._capture_close()
        finally:
            super().shutdown()

    def capture_status(self) -> dict[str, Any]:
        """Optional named RPC, not needed to collect or stop capture."""
        return self._capture_writer.flush()

    def _capture_fail(self, code: str, error: Exception | None = None) -> None:
        self._capture_writer.error(code)
        if error is not None and len(self._capture_writer.error_details) < 8:
            safe_messages = {
                "capture hook did not write selected rows",
                "invalid V2 request layout", "invalid V2 query boundaries",
                "V2 query boundary exceeds input buffers",
                "duplicate batch request IDs", "token layout differs from scheduled batch",
                "invalid scheduled draft count", "V2 forward did not retain an execution state",
                "unrecognized vLLM input batch layout",
            }
            detail = {
                "exception": type(error).__name__,
                "message": str(error) if str(error) in safe_messages else None,
                "frames": [{"file": Path(frame.filename).name, "function": frame.name,
                            "line": frame.lineno}
                           for frame in traceback.extract_tb(error.__traceback__)][-8:],
                "diagnostic": getattr(error, "capture_diagnostic", None),
            }
            self._capture_writer.error_details.append(detail)
        self._capture_disabled = True
        try:
            self._capture_writer.flush()
        except OSError:
            pass

    def _capture_read(self, scheduler: Any) -> Pending:
        runner = self.model_runner
        if self.use_v2_model_runner:
            state = runner.execute_model_state
            if state is None:
                raise ValueError("V2 forward did not retain an execution state")
            rows, drafts = v2_batch_rows(state.input_batch, runner.req_states, self._capture_sampler)
            return Pending(self._capture_step, rows, self._capture_tensor.read(rows), drafts)
        requests = list(runner.input_batch.req_ids)
        counts = dict(scheduler.num_scheduled_tokens)
        count = scheduler.total_num_scheduled_tokens
        if len(requests) != runner.input_batch.num_reqs or sum(counts.values()) != count:
            raise ValueError("unrecognized vLLM input batch layout")
        # GPU positions contain async speculative corrections. The CPU optimistic
        # num_computed_tokens is deliberately not used as an observation position.
        positions = runner.positions[:count].detach().cpu().tolist()
        lengths = {req: runner.requests[req].num_prompt_tokens for req in requests}
        drafts = {req: len(tokens) for req, tokens in scheduler.scheduled_spec_decode_tokens.items()}
        rows = self._capture_sampler.select(requests, counts, positions, lengths, drafts,
                                            {req for req in requests if selected_request(req)})
        return Pending(self._capture_step, rows, self._capture_tensor.read(rows), drafts)

    def _capture_emit(self, pending: Pending, output: Any = None) -> None:
        if not pending.rows:
            return
        try:
            accepted = acceptance_counts(output, pending.drafts) if output is not None else {}
            self._capture_queue.submit(pending, accepted)
        except Exception as error:
            self._capture_fail(f"enqueue_{type(error).__name__}", error)

    def _capture_attach(self, output: Any) -> Any:
        pending, self._capture_pending = self._capture_pending, None
        if pending is None or not pending.rows:
            return output
        if isinstance(output, AsyncModelRunnerOutput):
            # Only TP rank 0's result is consumed by MultiprocExecutor. Other
            # ranks already emitted below, with unresolved draft acceptance.
            return ObservedAsyncOutput(output, lambda result: self._capture_emit(pending, result))
        self._capture_emit(pending, output)
        return output

    def execute_model(self, scheduler_output: Any) -> Any:
        selected = any(selected_request(req) for req in scheduler_output.num_scheduled_tokens)
        if not getattr(self, "_capture_disabled", True):
            self._capture_sampler.finish(scheduler_output.finished_req_ids)
        if selected and not getattr(self, "_capture_disabled", True):
            self._capture_step += 1
            if self._capture_pending is not None:
                self._capture_emit(self._capture_pending)
                self._capture_pending = None
                self._capture_writer.error("previous_sampling_result_missing")
            self._capture_tensor.reset()
        output = super().execute_model(scheduler_output)
        if selected and not getattr(self, "_capture_disabled", True) and scheduler_output.total_num_scheduled_tokens:
            try:
                pending = self._capture_read(scheduler_output)
                if self._capture_rank == 0:
                    self._capture_pending = pending
                    if output is not None:
                        output = self._capture_attach(output)
                else:
                    self._capture_emit(pending)
            except Exception as error:
                self._capture_fail(f"attribution_{type(error).__name__}", error)
        return output

    def sample_tokens(self, grammar_output: Any) -> Any:
        output = super().sample_tokens(grammar_output)
        if not getattr(self, "_capture_disabled", True):
            return self._capture_attach(output)
        return output
