"""Small CPU end-to-end checks for the experimental pinned Gemma LoRA recipe."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora"


def safefile(path, tensors):
    header, chunks, cursor = {}, [], 0
    for key, value in sorted(tensors.items()):
        value = np.ascontiguousarray(value, dtype=np.float32)
        bits = value.view(np.uint32)
        raw = (
            ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16)
            .astype("<u2")
            .tobytes()
        )
        header[key] = {
            "dtype": "BF16",
            "shape": list(value.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        chunks.append(raw)
        cursor += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


@unittest.skipUnless(
    sys.platform.startswith("linux") and shutil.which("c++"), "Linux C++/OpenMP recipe"
)
class LoraRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.library = Path(cls.build.name) / "kernel.so"
        subprocess.run(
            ["sh", str(SCRIPTS / "build-kernel.sh"), str(cls.library)],
            check=True,
            capture_output=True,
        )
        cls.env = mock.patch.dict(
            os.environ, {"KAIRYU_BENCH_BF16_LORA_LIBRARY": str(cls.library)}
        )
        cls.env.start()
        sys.path.insert(0, str(SCRIPTS))
        cls.recipe = importlib.import_module("dense_lora_recipe")
        cls.ops = importlib.import_module("bf16_lora_ops")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(SCRIPTS))
        cls.env.stop()
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.base = self.work / "base"
        self.base.mkdir()
        shards, mapping, adapter = [{}, {}], {}, {}
        mechanisms = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        for layer in range(60):
            for mechanism in mechanisms + (
                ["self_attn.v_proj"] if layer % 6 != 5 else []
            ):
                name = f"model.language_model.layers.{layer}.{mechanism}.weight"
                shard = f"model-{layer % 2}.safetensors"
                mapping[name] = shard
                shards[layer % 2][name] = np.array(
                    [[1.0, -1.0], [0.25, -0.25]], np.float32
                )
                prefix = "base_model.model." + name.removesuffix(".weight")
                adapter[prefix + ".lora_A.weight"] = np.full(
                    (16, 2), 1 / 32, np.float32
                )
                adapter[prefix + ".lora_B.weight"] = np.full(
                    (2, 16), 1 / 64, np.float32
                )
        norm = "model.language_model.norm.weight"
        shards[0][norm] = np.array(
            [1.0, -1.0, 0.25, -0.25, 4.0, -4.0, 0.125, -0.125], np.float32
        )
        mapping[norm] = "model-0.safetensors"
        shards[0]["model.vision_tower.untouched.weight"] = np.array(
            [7.0, -8.0], np.float32
        )
        mapping["model.vision_tower.untouched.weight"] = "model-0.safetensors"
        for i, values in enumerate(shards):
            safefile(self.base / f"model-{i}.safetensors", values)
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
        (self.base / "tokenizer_config.json").write_text('{"unchanged":true}\n')
        donor = self.work / "adapter.safetensors"
        safefile(donor, adapter)
        config = self.work / "adapter_config.json"
        config.write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 32,
                    "use_rslora": False,
                    "use_dora": False,
                    "fan_in_fan_out": False,
                    "bias": "none",
                    "rank_pattern": {},
                    "alpha_pattern": {},
                }
            )
        )
        self.spec = {
            "schema_version": 2,
            "kind": self.recipe.KIND,
            "scope": "native_text410",
            "donor_path": str(donor),
            "donor_config_path": str(config),
            "donor_sha256": self.recipe.digest(donor),
            "donor_config_sha256": self.recipe.digest(config),
            "donor_repo": "synthetic/test",
            "donor_revision": "0" * 40,
            "coefficient": 1.0,
            "base_tree_sha256": self.recipe.tree(self.base)[1],
        }
        self.spec_path = self.work / "spec.json"
        self.spec_path.write_text(json.dumps(self.spec))
        self.dest = self.work / "candidate"
        self.manifest = self.work / "manifest.json"

    def call(self, name, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / name), *map(str, args)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        return result

    def test_create_verify_and_source_ownership(self):
        args = [
            "create",
            self.base,
            self.dest,
            "--spec",
            self.spec_path,
            "--manifest",
            self.manifest,
        ]
        preview = json.loads(self.call("dense_lora_recipe.py", *args).stdout)
        self.assertTrue(preview["dry_run"])
        self.assertFalse(self.dest.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            record = self.recipe.create(
                self.base, self.dest, self.spec_path, self.manifest
            )
            self.recipe.verify(self.manifest)
        self.assertEqual(record["changed_elements"], 1640)
        self.assertAlmostEqual(record["delta_l2"], math.sqrt(1640) / 64)
        self.assertEqual(self.recipe.tree(self.base)[1], self.spec["base_tree_sha256"])
        untouched = self.dest / "tokenizer_config.json"
        original = untouched.read_bytes()
        untouched.write_text('{"unchanged":false}\n')
        with self.assertRaises(ValueError):
            self.recipe.verify(self.manifest)
        untouched.write_bytes(original)
        entry = record["edits"][0]
        target = self.dest / entry["shard"]
        link = self.work / "foreign-link"
        os.link(target, link)
        with self.assertRaises(ValueError):
            self.recipe.verify(self.manifest)
        link.unlink()
        with target.open("r+b") as stream:
            stream.seek(entry["offset"])
            stream.write(b"\0\0")
        with self.assertRaises(ValueError):
            self.recipe.verify(self.manifest)

    def test_verify_rejects_source_changed_during_regeneration(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.recipe.create(self.base, self.dest, self.spec_path, self.manifest)
        original_entries = self.recipe.entries

        def mutate_source(*args, **kwargs):
            result = original_entries(*args, **kwargs)
            (self.base / "tokenizer_config.json").write_text('{"unchanged":false}\n')
            return result

        with mock.patch.object(self.recipe, "entries", mutate_source):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                self.recipe.verify(self.manifest)

    def test_atomic_publication_preserves_foreign_manifest(self):
        original_entries = self.recipe.entries

        def foreign_manifest(*args, **kwargs):
            result = original_entries(*args, **kwargs)
            self.manifest.write_text("foreign writer\n")
            return result

        with mock.patch.object(self.recipe, "entries", foreign_manifest):
            with self.assertRaises(FileExistsError):
                self.recipe.create(self.base, self.dest, self.spec_path, self.manifest)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.manifest.read_text(), "foreign writer\n")
        self.assertEqual(self.recipe.tree(self.base)[1], self.spec["base_tree_sha256"])

    def test_kernel_rounding_and_realized_audit(self):
        rng = np.random.default_rng(2)
        raw = rng.integers(0, 65536, 5000, dtype=np.uint16)
        raw = raw[((raw >> 7) & 255) != 255]
        delta = rng.normal(size=raw.shape).astype(np.float32)
        coefficients = np.array([1 / 256, 1 / 64, 1 / 16, 1 / 4, 1], np.float32)
        stats, counts = self.ops.audit5(raw, delta, coefficients)
        base = (raw.astype(np.uint32) << 16).view(np.float32)
        for j, c in enumerate(coefficients):
            expected = np.add(
                base, np.multiply(delta, c, dtype=np.float32), dtype=np.float32
            )
            bits = expected.view(np.uint32)
            rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(
                np.uint16
            )
            got, measured, changed = self.ops.merge1(raw, delta, c)
            np.testing.assert_array_equal(got, rounded)
            self.assertEqual(int(changed[0]), int(counts[1 + 3 * j]))
            self.assertAlmostEqual(
                measured[1],
                stats[2 + 3 * j],
                delta=max(1e-12, abs(measured[1]) * 1e-12),
            )
        for bad in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.ops.audit5(raw, delta, [1, 1, 1, 1, bad])
        out = self.work / "audit"
        args = [
            "--source",
            self.base,
            "--spec",
            self.spec_path,
            "--output",
            out,
            "--global-base-l2",
            "100",
        ]
        self.call("audit_public_adapter_norms.py", *args)
        self.call("audit_public_adapter_realized.py", *args)
        audit = json.loads((out / "realized-summary.json").read_text())
        self.assertEqual(audit["matrices"], 410)
        self.assertEqual(audit["coefficients"][-1]["changed_value_elements"], 1640)
        self.assertAlmostEqual(
            audit["coefficients"][-1]["realized_delta_l2"], math.sqrt(1640) / 64
        )
        self.assertEqual(self.recipe.tree(self.base)[1], self.spec["base_tree_sha256"])

    def test_native_scale_composes_from_original(self):
        self.spec["native_edits"] = [
            {"tensor": "model.language_model.norm.weight", "scale": 1.01}
        ]
        self.spec_path.write_text(json.dumps(self.spec))
        try:
            self._native_create()
        except subprocess.CalledProcessError as error:
            self.fail(error.stderr)
        self.call("dense_lora_native_recipe.py", "verify", self.manifest)
        record = json.loads(self.manifest.read_text())
        self.assertEqual(record["changed_elements"], 1648)
        self.assertEqual(len(record["edits"]), 411)
        self.assertEqual(self.recipe.tree(self.base)[1], self.spec["base_tree_sha256"])
        record["edits"][-1]["entry_kind"] = "unknown"
        self.manifest.write_text(json.dumps(record))
        with self.assertRaises(subprocess.CalledProcessError):
            self.call("dense_lora_native_recipe.py", "verify", self.manifest)

    def _native_create(self):
        self.call(
            "dense_lora_native_recipe.py",
            "create",
            self.base,
            self.dest,
            "--spec",
            self.spec_path,
            "--manifest",
            self.manifest,
            "--apply",
            "--min-free-gib",
            "0",
        )


if __name__ == "__main__":
    unittest.main()
