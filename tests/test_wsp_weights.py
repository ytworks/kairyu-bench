from __future__ import annotations

import contextlib
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint, CheckpointError, DTYPES
from kairyu_bench.wsp.weights.matrix import alignment, spectra
from kairyu_bench.wsp.weights.statistics import diff_checkpoints, inspect_checkpoint


def weights(path: Path, entries: dict[str, tuple[str, tuple[int, ...], bytes]]) -> Path:
    """Independent, small safetensors writer; never imports production encoders."""
    header, payload = {}, bytearray()
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def values(dtype: str, shape: tuple[int, ...], sequence: list[float] | list[int]) -> tuple[str, tuple[int, ...], bytes]:
    return dtype, shape, np.asarray(sequence, dtype=DTYPES[dtype][0]).tobytes()


class CheckpointObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_bf16_bits_numeric_delta_and_unknown_nonfinite_are_distinct(self) -> None:
        base = weights(self.root/"base/model.safetensors", {
            "layer.0.w": ("BF16", (5,), struct.pack("<5H", 0x3F80, 0x0000, 0x7FC1, 0x0001, 0xBF80))})
        candidate = weights(self.root/"candidate/model.safetensors", {
            "layer.0.w": ("BF16", (5,), struct.pack("<5H", 0x3F81, 0x8000, 0x7FC2, 0x0001, 0xBF80))})
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, candidate)]
        result = diff_checkpoints(base, candidate, chunk_bytes=8)
        row = result["tensors"][0]
        self.assertEqual(row["stored_bit_changed_elements"], 3)
        self.assertEqual(row["numeric_changed_finite_elements"], 1)
        self.assertEqual(row["nonfinite_pairs"], 1)
        self.assertIsNone(row["delta"]["l2"])
        self.assertEqual(row["delta"]["finite_l2"], 0.0078125)
        self.assertEqual(row["top_delta_coordinates"][0]["coordinate"], [0])
        decoded = Checkpoint(base).read("layer.0.w")
        self.assertEqual(float(decoded[3]), 2.0**-133)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, candidate)])
        json.dumps(result, allow_nan=False)

    def test_streamed_axis_quantiles_and_top_match_reference_across_chunks(self) -> None:
        data = [float(i - 19) for i in range(48)]
        path = weights(self.root/"model.safetensors", {
            "layer.7.w": values("F32", (6, 8), data),
            "layer.9.scalar": values("F64", (), [2.0])})
        a = inspect_checkpoint(path, pattern=r"layer\.7", group_pattern=r"layer\.(\d+)",
                               axis=True, sample_size=7, chunk_bytes=8, top_k=3)
        b = inspect_checkpoint(path, pattern=r"layer\.7", axis=True,
                               sample_size=7, chunk_bytes=64, top_k=3)
        row_a, row_b = a["tensors"][0], b["tensors"][0]
        self.assertAlmostEqual(a["global"]["l2"], math.sqrt(math.fsum(x*x for x in data)))
        self.assertEqual(row_a["top_coordinates"], row_b["top_coordinates"])
        self.assertEqual(row_a["quantiles"], row_b["quantiles"])
        self.assertEqual(set(a["groups"]), {"7"})
        self.assertEqual(row_a["quantiles"]["scope"], "deterministic_sample")
        self.assertNotIn("l2", row_a["axis_profiles"]["rows"])
        reference = np.asarray(data).reshape(6, 8)
        profile = row_a["axis_profiles"]["rows"]
        self.assertEqual(profile["top"][0]["flat_index"], 5)
        self.assertAlmostEqual(profile["l2_quantiles"]["0.5"], float(np.median(np.linalg.norm(reference, axis=1))))
        self.assertEqual(a["io"]["estimated_bytes"]["total_bytes"], a["bytes_read"])
        self.assertLessEqual(a["budget"]["estimated_working_bytes"], 512*1024**2)

    def test_shard_index_inventory_and_nested_paths(self) -> None:
        a = weights(self.root/"one/same.safetensors", {"a": values("F16", (2,), [3, 4])})
        b = weights(self.root/"two/same.safetensors", {"b": values("F64", (1,), [12])})
        (self.root/"model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 12}, "weight_map": {"a": "one/same.safetensors", "b": "two/same.safetensors"}}))
        (self.root/"tokenizer.model").write_bytes(b"small tokenizer fixture")
        checkpoint = Checkpoint(self.root)
        result = inspect_checkpoint(checkpoint, chunk_bytes=8)
        self.assertEqual(result["global"]["l2"], 13)
        self.assertEqual([x["shard"] for x in result["tensors"]], ["one/same.safetensors", "two/same.safetensors"])
        self.assertIn(a, checkpoint.files)
        self.assertIn(b, checkpoint.files)
        self.assertIn(self.root/"tokenizer.model", checkpoint.files)
        self.assertEqual(result["checkpoint"]["identity_level"], "content_sha256")
        index = self.root/"model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"a": "two/same.safetensors", "b": "one/same.safetensors"}}))
        with self.assertRaisesRegex(CheckpointError, "wrong shard"):
            Checkpoint(self.root)

    def test_bad_format_and_missing_shards_fail_before_numeric_scan(self) -> None:
        malformed = {
            "duplicate": b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            "overlap": b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            "dtype": b'{"x":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}',
            "shape": b'{"x":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}',
        }
        for name, header in malformed.items():
            with self.subTest(name=name):
                path = self.root/(name+".safetensors")
                path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0"*4)
                with self.assertRaises(CheckpointError):
                    Checkpoint(path)
        path = weights(self.root/"truncated.safetensors", {"x": values("F32", (2,), [1, 2])})
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(CheckpointError, "bounds"):
            Checkpoint(path)
        for filename in ("missing.safetensors", "../escape.safetensors"):
            (self.root/"model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": filename}}))
            with self.assertRaises((CheckpointError, FileNotFoundError)):
                Checkpoint(self.root)

    def test_partial_empty_dtype_and_integer_coverage(self) -> None:
        base = weights(self.root/"a.safetensors", {
            "x": values("F32", (1,), [1]), "only_a": values("F32", (1,), [2]),
            "integer": values("I64", (1,), [2**60])})
        candidate = weights(self.root/"b.safetensors", {
            "x": values("F64", (1,), [1]), "only_b": values("F32", (1,), [3]),
            "integer": values("I64", (1,), [2**60+1])})
        with self.assertRaisesRegex(CheckpointError, "coverage differs"):
            diff_checkpoints(base, candidate)
        partial = diff_checkpoints(base, candidate, allow_partial=True, chunk_bytes=8)
        self.assertFalse(partial["coverage_complete"])
        self.assertEqual(partial["global"]["stored_bit_changed_elements"], 1)
        self.assertIsNone(partial["global"]["delta"]["l2"])
        self.assertEqual(partial["nonfloating_differences"][0]["numeric_changed_elements"], 1)
        empty = diff_checkpoints(base, candidate, pattern="^x$", allow_partial=True)
        self.assertEqual(empty["status"], "no_comparable_tensors")
        self.assertIsNone(empty["global"]["delta"]["l2"])
        with self.assertRaisesRegex(CheckpointError, "no tensors"):
            inspect_checkpoint(base, pattern="^nothing$")
        with self.assertRaisesRegex(ValueError, "floating"):
            spectra(Checkpoint(base), ["integer"])

    def test_zero_sized_tensors_remain_valid_and_zero_denominator_is_unknown(self) -> None:
        path = weights(self.root/"model.safetensors", {"empty": values("F32", (0, 3), [])})
        inspected = inspect_checkpoint(path, axis=True)
        result = diff_checkpoints(path, path, axis=True)
        self.assertEqual(inspected["global"]["l2"], 0)
        self.assertEqual(result["global"]["delta"]["l2"], 0)
        self.assertIsNone(result["global"]["relative_delta_l2"])
        self.assertIsNone(result["tensors"][0]["cosine"])
        self.assertEqual(inspected["tensors"][0]["axis_profiles"]["columns"]["zero_axes"], 3)

    def test_f64_delta_uses_saved_values_and_extreme_quantiles_are_finite(self) -> None:
        base = weights(self.root/"a.safetensors", {"x": values("F64", (1,), [2.0**-30])})
        candidate = weights(self.root/"b.safetensors", {"x": values("F64", (1,), [1.0])})
        result = diff_checkpoints(base, candidate)
        self.assertEqual(result["global"]["delta"]["l2"], 1.0-2.0**-30)
        extreme = weights(self.root/"extreme.safetensors", {"x": values("F64", (2,), [-1e308, 1e308])})
        observed = inspect_checkpoint(extreme)
        self.assertEqual(observed["tensors"][0]["quantiles"]["values"]["0.5"], 0.0)
        self.assertTrue(math.isfinite(observed["global"]["l2"]))
        json.dumps(observed, allow_nan=False)

    def test_input_change_after_earlier_file_hash_is_rejected(self) -> None:
        a = weights(self.root/"a.safetensors", {"a": values("F32", (1,), [1])})
        b = weights(self.root/"b.safetensors", {"b": values("F32", (1,), [2])})
        (self.root/"model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": a.name, "b": b.name}}))
        checkpoint = Checkpoint(self.root)
        original = checkpoint._open_stable

        @contextlib.contextmanager
        def mutate_earlier(path: Path):
            with original(path) as stream:
                yield stream
            if path == b:
                a.write_bytes(a.read_bytes()[:-4] + struct.pack("<f", 3.0))

        with patch.object(checkpoint, "_open_stable", side_effect=mutate_earlier):
            with self.assertRaisesRegex(CheckpointError, "source changed"):
                checkpoint.identity()

    def test_configuration_inventory_change_and_resource_limits(self) -> None:
        path = weights(self.root/"model.safetensors", {"x": values("F32", (2,), [1, 2])})
        checkpoint = Checkpoint(path)
        (self.root/"generation_config.json").write_text("{}")
        with self.assertRaisesRegex(CheckpointError, "inventory changed"):
            checkpoint.identity(hash_files=False)
        with self.assertRaisesRegex(CheckpointError, "working memory"):
            inspect_checkpoint(path, chunk_bytes=64*1024**2)
        checkpoint = Checkpoint(path)
        with self.assertRaisesRegex(CheckpointError, "outside"):
            checkpoint.read("x", 1, 3)
        with self.assertRaisesRegex(CheckpointError, "unknown"):
            checkpoint.read("missing", 0, 1)

    def test_scaled_spectrum_alignment_and_sampled_scope(self) -> None:
        path = weights(self.root/"model.safetensors", {
            "diagonal": values("F64", (2, 2), [3, 0, 0, 4]),
            "huge": values("F64", (2, 2), [1e308, 1e308, 0, 0]),
            "zero": values("F32", (2, 2), [0, 0, 0, 0])})
        checkpoint = Checkpoint(path)
        full = spectra(checkpoint, ["diagonal"], sample_rows=2, sample_columns=2)
        self.assertEqual(full["matrices"][0]["coverage"], "full_matrix")
        self.assertEqual(full["matrices"][0]["spectrum"]["singular_values"], [4, 3])
        sampled = spectra(Checkpoint(path), ["diagonal"], sample_rows=1, sample_columns=1, hash_files=False)
        self.assertEqual(sampled["matrices"][0]["coverage"], "sampled_submatrix")
        self.assertIsNone(sampled["checkpoint"]["sha256"])
        large = spectra(Checkpoint(path), ["huge"], sample_rows=2, sample_columns=2)
        self.assertEqual(large["matrices"][0]["spectrum"]["numerical_rank"], 1)
        same = alignment(Checkpoint(path), "huge", "huge", channels=2, features=2)
        self.assertAlmostEqual(same["channels"][0]["cosine"], 1)
        self.assertIsNone(same["channels"][0]["norm_product"])
        self.assertIsNone(same["channels"][1]["cosine"])
        self.assertEqual(same["valid_cosine_channels"], 1)
        zero = spectra(Checkpoint(path), ["zero"])
        self.assertEqual(zero["matrices"][0]["spectrum"]["stable_rank"], 0)
        json.dumps([full, sampled, large, same, zero], allow_nan=False)

    def test_transposed_alignment_same_indices_and_io_counts(self) -> None:
        path = weights(self.root/"model.safetensors", {
            "left": values("F32", (2, 3), [1, 2, 3, -1, -2, -3]),
            "right": values("F32", (3, 2), [1, -1, 2, -2, 3, -3]),
            "bad": values("F32", (1, 2), [1, 2])})
        result = alignment(Checkpoint(path), "left", "right", right_axis="columns",
                           channels=2, features=3)
        self.assertEqual(result["coverage"], "full_matrices")
        self.assertAlmostEqual(result["mean_cosine"], 1)
        self.assertEqual(result["io"]["estimated_bytes"]["total_bytes"], result["io"]["actual_bytes"]["total"])
        with self.assertRaisesRegex(ValueError, "matching"):
            alignment(Checkpoint(path), "left", "bad")
        with self.assertRaisesRegex(ValueError, "between"):
            spectra(Checkpoint(path), ["left"], sample_rows=2048)


if __name__ == "__main__":
    unittest.main()
