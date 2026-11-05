from __future__ import annotations

import importlib.util
import json
import gzip
import math
import os
import runpy
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kairyu_bench.wsp.observe.collector import (
    CaptureConfig,
    METRICS,
    gemma_checkpoint_mapping,
    observe_checkpoint,
    observed_runtime,
    v2_batch_rows,
    selected_request,
    CaptureWriter,
    PositionSampler,
    Row,
    TensorCapture,
    acceptance_counts,
    row_record,
)


class RuntimeCollectionTests(unittest.TestCase):
    def config(self, directory: Path, **limits: int) -> CaptureConfig:
        path = directory / "capture.json"
        path.write_text(json.dumps({
            "schema_version": 1, "capture_id": "unit-capture",
            "observation_mode": "live", "output_dir": str(directory / "output"),
            "checkpoint": {"sha256": "a" * 64},
            "runtime": {"tp_size": 2, "replicas": 4},
            "limits": {"gpu_buffer_mib": 256, "storage_bytes": 16 * 1024 * 1024,
                       "prefill_positions": 32, "decode_positions": 32, **limits},
        }))
        return CaptureConfig.read(path)



    def test_loaded_checkpoint_stamps_match_declared_prehash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.safetensors"
            path.write_bytes(b"fixture")
            stat = path.stat()
            entry = {"path": path.name, "size": stat.st_size, "sha256": "a" * 64,
                     "stat": {"device": stat.st_dev, "inode": stat.st_ino,
                              "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}}
            observed = observe_checkpoint(temp, {"files": [entry]})
            self.assertEqual(observed["verification"], "prehashed_file_stamps_match")
            self.assertFalse(observed["per_worker_full_content_hash"])
            path.write_bytes(b"changed fixture")
            with self.assertRaisesRegex(ValueError, "stamps differ"):
                observe_checkpoint(temp, {"files": [entry]})
            with self.assertRaisesRegex(ValueError, "prehashed"):
                observe_checkpoint(temp, {})



    def test_observed_runtime_records_real_mtp_and_rejects_wrong_declaration(self) -> None:
        config = types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(num_speculative_tokens=4),
            scheduler_config=types.SimpleNamespace(async_scheduling=True),
            compilation_config=types.SimpleNamespace(mode="VLLM_COMPILE",
                                                     cudagraph_mode="FULL_AND_PIECEWISE"),
            model_config=types.SimpleNamespace(max_model_len=32768),
        )
        runner = types.SimpleNamespace(num_speculative_steps=4, max_num_tokens=8192)
        observed = observed_runtime(config, runner, v2=True, declaration={"mtp_tokens": 4})
        self.assertEqual(observed["mtp_tokens"], 4)
        self.assertEqual(observed["max_num_batched_tokens"], 8192)
        self.assertEqual(observed["model_runner"], "v2")
        with self.assertRaisesRegex(ValueError, "MTP"):
            observed_runtime(config, runner, v2=True, declaration={"mtp_tokens": 0})


    def test_compact_queue_flushes_gzip_nulls_and_acceptance(self) -> None:
        import time
        import numpy as np
        from kairyu_bench.wsp.observe.collector_io import RecordQueue

        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            sites = [{"site_id": "layer.0.residual", "layer": 0, "role": "residual"}]
            writer = CaptureWriter(config, "r0", 0, sites, 100)
            record_queue = RecordQueue(writer, sites)
            values = np.array([[[1., 2., 3., 0., 1., 0., 0.],
                                [math.nan, math.nan, math.nan, 4., math.nan, math.nan, math.nan]]],
                              dtype=np.float32)
            pending = types.SimpleNamespace(step=7, values=values, drafts={"observe-x": 1},
                                            rows=[Row(0, "observe-x", 0, "prefill", None),
                                                  Row(1, "observe-x", 10, "decode", 0)])
            before_submit = time.time()
            self.assertTrue(record_queue.submit(pending, {"observe-x": 0}))
            self.assertTrue(record_queue.close())
            after_close = time.time()
            with gzip.open(writer.directory / "observations.jsonl.gz", "rt") as stream:
                data = json.loads(stream.readline())
            self.assertEqual(data["record_type"], "activation_batch")
            self.assertEqual(data["rows"][1][-1], False)
            self.assertIsNone(data["values"][0][1][0])
            self.assertEqual(data["values"][0][1][3], 4)
            status = json.loads((writer.directory / "status.json").read_text())
            self.assertTrue(status["closed"])
            self.assertEqual(status["records_written"], 3)
            self.assertEqual(status["writer_queue"]["pending_bytes"], 0)
            self.assertEqual(status["writer_queue"]["depth"], 0)
            enqueued = status["writer_queue"]["last_enqueued_at_unix"]
            completed = status["writer_queue"]["last_completed_at_unix"]
            self.assertLessEqual(before_submit, enqueued)
            self.assertLessEqual(enqueued, completed)
            self.assertLessEqual(completed, after_close)

    def test_cpu_queue_bound_drops_before_serialization(self) -> None:
        import numpy as np
        from kairyu_bench.wsp.observe.collector_io import RecordQueue

        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            sites = [{"site_id": "layer.0.residual"}]
            writer = CaptureWriter(config, "r0", 0, sites, 100)
            record_queue = RecordQueue(writer, sites, max_bytes=1)
            pending = types.SimpleNamespace(step=1, values=np.zeros((1, 1, 7), dtype=np.float32),
                                            drafts={}, rows=[Row(0, "observe-x", 0, "prefill", None)])
            self.assertFalse(record_queue.submit(pending, {}))
            self.assertTrue(record_queue.close())
            status = json.loads((writer.directory / "status.json").read_text())
            self.assertEqual(status["dropped_records"], 1)
            self.assertEqual(status["writer_queue"]["dropped_batches"], 1)
            self.assertEqual(status["writer_queue"]["highwater_bytes"], 0)
            self.assertIsNone(status["writer_queue"]["last_enqueued_at_unix"])
            self.assertIsNone(status["writer_queue"]["last_completed_at_unix"])

    def test_internal_warmup_request_is_excluded(self) -> None:
        self.assertFalse(selected_request("req_7_generated-warmup-id"))
        self.assertFalse(selected_request("chatcmpl-ordinary-request"))
        self.assertTrue(selected_request("chatcmpl-observe-trace-call"))
        sampler = PositionSampler()
        rows = sampler.select(["req_warmup", "observe-call"],
                              {"req_warmup": 10, "observe-call": 2},
                              list(range(12)), {"req_warmup": 10, "observe-call": 12},
                              {}, {"observe-call"})
        self.assertEqual([row.index for row in rows], [10, 11])
        self.assertNotIn("req_warmup", sampler.selected)

    def test_explicit_gemma_fused_mapping_excludes_missing_weights(self) -> None:
        checkpoint = {"model.language_model.layers.2.self_attn.q_proj.weight",
                      "model.language_model.layers.2.self_attn.v_proj.weight",
                      "model.language_model.layers.2.mlp.gate_proj.weight",
                      "model.language_model.layers.2.mlp.up_proj.weight"}
        mapped = gemma_checkpoint_mapping(
            ["model.layers.2.self_attn.qkv_proj.weight",
             "model.layers.2.mlp.gate_up_proj.weight",
             "model.layers.2.nonexistent.weight"], 2, checkpoint)
        self.assertEqual(set(mapped), checkpoint)


    def test_v2_actual_sorted_layout_adaptive_drafts_and_padding(self) -> None:
        class Array:
            def __init__(self, values: list[object]):
                self.values = values

            def __getitem__(self, key: object) -> object:
                result = self.values[key]
                return Array(result) if isinstance(key, slice) else result

            def __len__(self) -> int:
                return len(self.values)

            def detach(self) -> object:
                return self

            def cpu(self) -> object:
                return self

            def tolist(self) -> list[object]:
                return self.values

        batch = types.SimpleNamespace(
            req_ids=["observe-decode", "chatcmpl-observe-prefill"], num_reqs=2,
            query_start_loc=Array([0, 3, 5, 6]),
            cu_num_logits=Array([0, 3, 4]),
            positions=Array([10, 11, 12, 0, 1, 999]),
            is_padding=Array([False, False, False, False, False, True]),
            idx_mapping_np=[2, 0],
            # Deliberately incompatible optimistic CPU counts; never used.
            num_scheduled_tokens=[5, 2], num_draft_tokens_per_req=[4, 0],
        )
        states = types.SimpleNamespace(prompt_len=types.SimpleNamespace(np=[2, 0, 10]))
        rows, drafts = v2_batch_rows(batch, states, PositionSampler())
        self.assertEqual(drafts, {"observe-decode": 2, "chatcmpl-observe-prefill": 0})
        self.assertEqual([r.index for r in rows], [0, 1, 2, 3, 4])
        self.assertEqual([r.speculative_index for r in rows], [None, 0, 1, None, None])
        self.assertEqual([r.phase for r in rows], ["decode"] * 3 + ["prefill"] * 2)

    def test_layout_reordering_padding_and_request_budget(self) -> None:
        sampler = PositionSampler(prefill=2, decode=2)
        rows = sampler.select(
            ["second", "first"], {"first": 4, "second": 2},
            [8, 9, 0, 1, 2, 3], {"first": 3, "second": 8}, {"first": 0, "second": 1})
        self.assertEqual([(r.request_id, r.index, r.phase) for r in rows],
                         [("second", 0, "decode"), ("second", 1, "decode"),
                          ("first", 2, "prefill"), ("first", 3, "prefill"),
                          ("first", 5, "decode")])
        # A rejected position re-executed later is not an extra sampled position.
        self.assertEqual(sampler.select(["second"], {"second": 1}, [9],
                                        {"second": 8}, {}), [])
        sampler.finish(["second"])
        self.assertNotIn("second", sampler.selected)

    def test_layout_mismatch_rejected(self) -> None:
        sampler = PositionSampler()
        with self.assertRaisesRegex(ValueError, "layout"):
            sampler.select(["r"], {"r": 1}, [0, 1], {"r": 1}, {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sampler.select(["r", "r"], {"r": 1}, [0, 1], {"r": 1}, {})
        with self.assertRaisesRegex(ValueError, "draft"):
            sampler.select(["r"], {"r": 1}, [0], {"r": 1}, {"r": 2})

    def test_no_more_than_32_positions_each_per_request(self) -> None:
        sampler = PositionSampler()
        rows = sampler.select(["r"], {"r": 200}, list(range(200)), {"r": 100}, {})
        self.assertEqual(sum(r.phase == "prefill" for r in rows), 32)
        self.assertEqual(sum(r.phase == "decode" for r in rows), 32)
        self.assertEqual(sampler.select(["r"], {"r": 1}, [201], {"r": 100}, {}), [])

    def test_speculative_acceptance_is_prefix_not_token_equality(self) -> None:
        output = types.SimpleNamespace(req_ids=["r"], sampled_token_ids=[[100, 200, 300]])
        accepted = acceptance_counts(output, {"r": 4})
        self.assertEqual(accepted, {"r": 2})
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            site = {"site_id": "layer.0.mlp", "layer": 0, "role": "mlp"}
            records = [row_record(config, "replica0", 0, 1, site,
                                  Row(i, "r", 10 + i, "decode", i), [1, 2, 3, 0, 1.7, 0, 0], accepted)
                       for i in range(4)]
            self.assertEqual([r["accepted"] for r in records], [True, True, False, False])
            unknown = row_record(config, "replica0", 1, 1, site,
                                 Row(0, "r", 10, "decode", 0), [1, 2, 3, 0, 1.7, 0, 0], {})
            self.assertIsNone(unknown["accepted"])

    def test_quota_drops_without_overwriting_and_status_records_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            writer = CaptureWriter(config, "replica0", 0, [], 1024)
            writer.write("acceptance", [{"content": "x" * 100000}] * 30)
            status = writer.flush(closed=True)
            self.assertGreater(status["dropped_records"], 0)
            self.assertLessEqual(status["bytes_written"], status["storage_quota_bytes"])
            self.assertTrue(status["closed"])
            with self.assertRaises(FileExistsError):
                CaptureWriter(config, "replica0", 0, [], 1024)

    def test_memory_budget_and_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            with self.assertRaisesRegex(ValueError, "budget"):
                CaptureWriter(config, "replica0", 0, [], 256 * 1024**2 + 1)
            with self.assertRaisesRegex(ValueError, "prefill"):
                self.config(root, prefill_positions=33)
            with self.assertRaisesRegex(ValueError, "gpu"):
                self.config(root, gpu_buffer_mib=True)

    def test_no_text_tokens_or_activations_in_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            record = row_record(config, "r0", 0, 1,
                                {"site_id": "layer.1.residual", "layer": 1, "role": "residual"},
                                Row(0, "chatcmpl-public-id", 10, "prefill", None),
                                [math.nan, math.nan, math.nan, 20, math.nan, math.nan, math.nan], {})
            self.assertIsNone(record["metrics"]["mean"])
            self.assertEqual(record["metrics"]["nonfinite_count"], 20)
            encoded = json.dumps(record, allow_nan=False)
            for forbidden in ("prompt", "reasoning_content", "token_ids", "activation_values"):
                self.assertNotIn(forbidden, encoded)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional serving PyTorch unavailable")
class TensorHookTests(unittest.TestCase):
    def test_cpu_synthetic_model_preserves_output_and_refreshes_metrics(self) -> None:
        import torch
        from torch import nn

        class Gemma4DecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Linear(4, 4, bias=False)
                self.mlp = nn.Linear(4, 4, bias=False)

            def forward(self, value: object) -> tuple[object, None]:
                value = self.self_attn(value)
                return self.mlp(value), None

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Gemma4DecoderLayer()])

            def forward(self, value: object) -> object:
                return self.layers[0](value)[0]

        torch.manual_seed(7)
        model = TinyModel().eval()
        value = torch.tensor([[1., 2., 3., 4.], [-1., 0., 2., 1.]])
        with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = RuntimeCollectionTests().config(root)
            before = model(value)
            original_parameters = {k: v.clone() for k, v in model.state_dict().items()}
            capture = TensorCapture(model, config, 8)
            capture.reset()
            after = model(value)
            self.assertTrue(torch.equal(before, after))
            rows = [Row(0, "r", 0, "prefill", None), Row(1, "r", 1, "prefill", None)]
            first = capture.read(rows)
            self.assertAlmostEqual(first[0][0][0], float(after[0].mean()), places=6)
            capture.reset()
            second_output = model(value * 2)
            second = capture.read(rows)
            self.assertAlmostEqual(second[0][0][1], 2 * first[0][0][1], places=6)
            self.assertTrue(torch.equal(second_output, before * 2))
            self.assertTrue(all(torch.equal(v, original_parameters[k])
                                for k, v in model.state_dict().items()))


    def test_selected_channel_and_tp_head_have_explicit_native_axes(self) -> None:
        import torch
        from torch import nn

        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_heads, self.head_dim, self.tp_rank = 2, 2, 1
                self.o_proj = nn.Linear(4, 4, bias=False)

            def forward(self, value: object) -> object:
                return self.o_proj(value)

        class Gemma4DecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = 4
                self.self_attn = Attention()
                self.mlp = nn.Linear(4, 4, bias=False)

            def forward(self, value: object) -> tuple[object, None]:
                return self.mlp(self.self_attn(value)), None

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Gemma4DecoderLayer()])

            def forward(self, value: object) -> object:
                return self.layers[0](value)[0]

        with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
            config = RuntimeCollectionTests().config(Path(temp))
            config = replace(config, detail_channels={"layer.0.residual": (1,)},
                             detail_heads={"layer.0.attention": (1,)})
            model = Tiny().eval()
            x = torch.tensor([[1., 2., 3., 4.], [-2., 3., 4., 5.]])
            original = model(x)
            key = "model.language_model.layers.0.self_attn.o_proj.weight"
            capture = TensorCapture(model, config, 8, {key})
            capture.reset()
            self.assertTrue(torch.equal(model(x), original))
            rows = [Row(0, "observe-test", 0, "prefill", None)]
            values = capture.read(rows)
            channel_index = next(i for i, site in enumerate(capture.sites)
                                 if site.get("detail_kind") == "channel")
            self.assertEqual(values[channel_index][0][0], float(original[0, 1]))
            channel_record = row_record(config, "r0", 1, 1, capture.sites[channel_index],
                                        rows[0], values[channel_index][0], {})
            self.assertEqual(channel_record["metrics"]["value"], float(original[0, 1]))
            head_index = next(i for i, site in enumerate(capture.sites)
                              if site.get("detail_kind") == "attention_head")
            head = capture.sites[head_index]
            self.assertEqual(head["site_id"], "layer.0.attention.head.3")
            self.assertEqual(head["head_global_index"], 3)
            self.assertEqual(head["checkpoint_axes"],
                             [{"tensor": key, "axis": 1, "start": 6, "stop": 8}])
            self.assertEqual(values[head_index][0][0], 3.5)
            self.assertEqual(values[head_index][0][4], .5)
            with self.assertRaisesRegex(ValueError, "channel"):
                TensorCapture(model, replace(config, detail_channels={"layer.0.mlp": (4,)}), 8)
            with self.assertRaisesRegex(ValueError, "head"):
                TensorCapture(model, replace(config, detail_heads={"layer.0.attention": (2,)}), 8)
            with self.assertRaisesRegex(ValueError, "budget"):
                TensorCapture(model, replace(config, gpu_bytes=100), 8)

    def test_operator_nan_inf_and_compilation_side_effect(self) -> None:
        import torch
        from kairyu_bench.wsp.observe.collector import install_row_operator

        operation = install_row_operator()
        output = torch.empty((3, len(METRICS)))
        values = torch.tensor([[3., 4., math.nan], [math.inf, -math.inf, math.nan],
                               [-2., 0., 2.]])
        operation(values, output, 6.)
        self.assertAlmostEqual(float(output[0, 1]), math.sqrt(12.5), places=6)
        self.assertEqual(float(output[0, 3]), 1)
        self.assertAlmostEqual(float(output[0, 4]), .5, places=6)
        self.assertEqual(float(output[2, 5]), float(torch.tensor(1 / 3)))
        self.assertEqual(float(output[2, 6]), 0)
        self.assertTrue(torch.isnan(output[1, :3]).all())
        self.assertEqual(float(output[1, 3]), 3)
        scalars = torch.tensor([[-.038818359375], [-0.0], [0.48046875]])
        operation(scalars, output, 6.)
        self.assertTrue(torch.equal(output[:, 0].view(torch.int32), scalars[:, 0].view(torch.int32)))
        self.assertTrue(torch.equal(output[:, 1], scalars[:, 0].abs()))

        def forward(x: object) -> object:
            operation(x, output, 6.)
            return x + 1

        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        result = compiled(torch.ones((3, 3)))
        self.assertTrue(torch.equal(result, torch.full((3, 3), 2.)))
        self.assertTrue(torch.equal(output[:, 0], torch.ones(3)))
        compiled(torch.full((3, 3), 2.))
        self.assertTrue(torch.equal(output[:, 0], torch.full((3,), 2.)))


class WorkerDelegationTests(unittest.TestCase):
    def worker_types(self) -> dict[str, object]:
        packages = {name: types.ModuleType(name) for name in (
            "vllm", "vllm.v1", "vllm.v1.outputs", "vllm.v1.worker",
            "vllm.v1.worker.gpu_worker")}
        packages["vllm.v1.outputs"].AsyncModelRunnerOutput = type(
            "AsyncModelRunnerOutput", (), {})

        class BaseWorker:
            def execute_model(self, scheduler: object) -> object:
                self.execute_calls += 1
                return self.base_output

            def sample_tokens(self, grammar: object) -> object:
                self.sample_calls += 1
                return self.base_output

        packages["vllm.v1.worker.gpu_worker"].Worker = BaseWorker
        source = Path(__file__).parents[1] / "src/kairyu_bench/wsp/observe/vllm_worker.py"
        with patch.dict(sys.modules, packages):
            return runpy.run_path(str(source))

    def test_async_delegate_consumed_once_and_result_identity_unchanged(self) -> None:
        module = self.worker_types()
        wrapper_type = module["ObservedAsyncOutput"]
        output = object()
        calls: list[object] = []

        class Delegate:
            def get_output(self) -> object:
                calls.append("get")
                return output

            def get_kv_connector_output(self) -> object:
                return "kv"

        wrapper = wrapper_type(Delegate(), lambda result: calls.append(result))
        self.assertIs(wrapper.get_output(), output)
        self.assertEqual(calls, ["get", output])
        self.assertEqual(wrapper.get_kv_connector_output(), "kv")

    def test_non_output_rank_emits_without_waiting_for_get_output(self) -> None:
        module = self.worker_types()
        cls = module["CaptureWorker"]
        worker = cls()
        worker.execute_calls = worker.sample_calls = 0
        worker.base_output = object()
        worker._capture_disabled = False
        worker._capture_rank = 1
        worker._capture_pending = None
        worker._capture_step = 0
        calls: list[object] = []
        worker._capture_sampler = types.SimpleNamespace(finish=lambda ids: calls.append("finish"))
        worker._capture_tensor = types.SimpleNamespace(reset=lambda: calls.append("reset"))
        worker._capture_read = lambda scheduler: "observations"
        worker._capture_emit = lambda pending: calls.append(pending)
        scheduler = types.SimpleNamespace(finished_req_ids=set(), total_num_scheduled_tokens=2,
                                          num_scheduled_tokens={"observe-test": 2})
        self.assertIs(worker.execute_model(scheduler), worker.base_output)
        self.assertEqual(worker.execute_calls, 1)
        self.assertEqual(calls, ["finish", "reset", "observations"])
        self.assertIsNone(worker._capture_pending)


@unittest.skipUnless(os.environ.get("KAIRYU_BENCH_TEST_CUDA") == "1",
                     "explicit coordinated CUDA smoke only")
class CudaGraphTests(unittest.TestCase):
    def test_custom_op_replays_new_values_and_matches_reference(self) -> None:
        import torch
        from kairyu_bench.wsp.observe.collector import install_row_operator

        free, _ = torch.cuda.mem_get_info()
        self.assertGreaterEqual(free, 1024**3)
        operation = install_row_operator()
        x = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16)
        output = torch.zeros((4, len(METRICS)), device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation(x, output, 6.)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation(x, output, 6.)
        graph.replay()
        self.assertTrue(torch.allclose(output[:, 0], x.mean(dim=1)))
        self.assertTrue(torch.allclose(output[:, 1], x.square().mean(dim=1).sqrt()))
        x.add_(100)
        graph.replay()
        self.assertTrue(torch.allclose(output[:, 0], x.mean(dim=1)))
        self.assertTrue(torch.allclose(output[:, 1], x.square().mean(dim=1).sqrt()))
        self.assertTrue(torch.equal(output[:, 3], torch.zeros(4, device="cuda")))
        x.fill_(float("nan"))
        graph.replay()
        self.assertTrue(torch.isnan(output[:, :3]).all())
        self.assertTrue(torch.equal(output[:, 3], torch.full((4,), 16., device="cuda")))

    def test_scalar_raw_values_are_exact_through_graph_replay(self) -> None:
        import torch
        from kairyu_bench.wsp.observe.collector import install_row_operator

        free, _ = torch.cuda.mem_get_info()
        self.assertGreaterEqual(free, 1024**3)
        operation = install_row_operator()
        for dtype in (torch.float32, torch.bfloat16):
            source = torch.tensor([[-.038818359375, 11.], [-0., 12.], [.48046875, 13.]],
                                  dtype=dtype, device="cuda")
            scalars = source[:, :1]  # Real channel views use a strided row axis.
            output = torch.empty((3, len(METRICS)), device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    operation(scalars, output, 6.)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                operation(scalars, output, 6.)
            for replacement in (None, [.0194091796875, -.0, -.9609375]):
                if replacement is not None:
                    scalars[:, 0].copy_(torch.tensor(replacement, dtype=dtype, device="cuda"))
                graph.replay()
                expected = scalars[:, 0].float()
                self.assertTrue(torch.equal(output[:, 0].contiguous().view(torch.int32),
                                            expected.contiguous().view(torch.int32)))
                self.assertTrue(torch.equal(output[:, 1], expected.abs()))
                self.assertTrue(torch.equal(output[:, 2], expected.abs()))
                self.assertTrue(torch.equal(output[:, 4], torch.zeros_like(expected)))
            scalars.fill_(float("nan"))
            graph.replay()
            self.assertTrue(torch.isnan(output[:, :3]).all())
            self.assertTrue(torch.equal(output[:, 3], torch.ones(3, device="cuda")))



if __name__ == "__main__":
    unittest.main()
