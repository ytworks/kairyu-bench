"""Small CPU end-to-end checks for the experimental pinned Gemma LoRA recipe."""

from __future__ import annotations

import contextlib
import copy
import hashlib
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


def safefile(path, tensors, dtype="BF16"):
    header, chunks, cursor = {}, [], 0
    for key, value in sorted(tensors.items()):
        value = np.ascontiguousarray(value, dtype=np.float32)
        bits = value.view(np.uint32)
        if dtype == "BF16":
            raw = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
        elif dtype == "F32":
            raw = value.astype("<f4").tobytes()
        else:
            raise ValueError("fixture dtype")
        header[key] = {
            "dtype": dtype,
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
        # Independent old-r16 byte oracle: every factor product is exactly 1/64.
        old = np.array([[1.0, -1.0], [0.25, -0.25]], np.float32)
        value = np.add(old, np.float32(1/64), dtype=np.float32).view(np.uint32)
        expected = ((value + np.uint32(0x7FFF) + ((value >> 16) & 1)) >> 16).astype("<u2").tobytes()
        for entry in record["edits"]:
            with (self.dest/entry["shard"]).open("rb") as stream:
                stream.seek(entry["offset"])
                self.assertEqual(stream.read(entry["size"]), expected)
        self.assertEqual(record["recipe"]["version"], 1)
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

    def _f32_factors(self, alpha=256):
        donor = Path(self.spec["donor_path"])
        h, _ = self.recipe.header(donor)
        factors = {}
        for name in h:
            if name == "__metadata__":
                continue
            value = np.zeros((128, 2) if ".lora_A." in name else (2, 128), np.float32)
            if ".lora_A." in name:
                value[0] = [1.003, -1.003]
            else:
                value[:, 0] = 1 / (256 * (alpha / 128))
            factors[name] = value
        safefile(donor, factors, "F32")
        config_path = Path(self.spec["donor_config_path"])
        config = json.loads(config_path.read_text())
        config.update(r=128, lora_alpha=alpha, peft_type="LORA")
        config_path.write_text(json.dumps(config))
        self.spec["donor_config_sha256"] = self.recipe.digest(config_path)
        self._pin_f32(factors)
        return factors

    def _pin_f32(self, factors):
        donor = Path(self.spec["donor_path"])
        safefile(donor, factors, "F32")
        _, offset = self.recipe.header(donor)
        with donor.open("rb") as stream:
            stream.seek(8)
            header_sha = hashlib.sha256(stream.read(offset-8)).hexdigest()
        self.spec["donor_sha256"] = self.recipe.digest(donor)
        self.spec["factor_format"] = {"version": 2, "rank": 128, "dtype": "F32", "header_sha256": header_sha}
        self.spec_path.write_text(json.dumps(self.spec))

    def test_f32_rank128_preserves_factors_scale_and_exact_created_words(self):
        old = np.array([[1.0, -1.0], [0.25, -0.25]], np.float32)
        def rounded(value):
            bits = value.view(np.uint32)
            return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("<u2")
        for alpha in (256, 64):
            with self.subTest(alpha=alpha):
                factors = self._f32_factors(alpha)
                self.dest = self.work / f"candidate-{alpha}"
                self.manifest = self.work / f"manifest-{alpha}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    record = self.recipe.create(self.base, self.dest, self.spec_path, self.manifest)
                    self.recipe.verify(self.manifest)
                recipe = record["recipe"]
                self.assertEqual((recipe["version"], recipe["r"], recipe["scale"]), (2, 128, alpha/128))
                entry = record["edits"][0]
                a, b = (factors[entry[key]] for key in ("lora_A", "lora_B"))
                delta = np.multiply(b @ a, np.float32(alpha/128), dtype=np.float32)
                # One nonzero rank term permits an independent FP64 scalar oracle;
                # cast before the scale operation to retain the declared F32 order.
                product_oracle = np.array(
                    [[float(b[row, 0]) * float(a[0, col]) for col in range(2)]
                     for row in range(2)], dtype=np.float32)
                np.testing.assert_array_equal(
                    delta, np.multiply(product_oracle, np.float32(alpha/128), dtype=np.float32))
                expected = rounded(np.add(old, delta, dtype=np.float32)).tobytes()
                aq = (rounded(a).astype(np.uint32) << 16).view(np.float32)
                bq = (rounded(b).astype(np.uint32) << 16).view(np.float32)
                downcast = rounded(np.add(old, (bq @ aq)*np.float32(alpha/128), dtype=np.float32)).tobytes()
                self.assertNotEqual(expected, downcast)
                for entry in record["edits"]:
                    with (self.dest/entry["shard"]).open("rb") as stream:
                        stream.seek(entry["offset"])
                        self.assertEqual(stream.read(entry["size"]), expected)
                self.assertEqual(self.recipe.tree(self.base)[1], self.spec["base_tree_sha256"])

    def test_extended_format_rejects_wrappers_bad_pins_shapes_and_payload(self):
        factors = self._f32_factors()
        with self.assertRaisesRegex(ValueError, "dense-only opt-in"):
            self.recipe.validated_recipe(self.base, self.spec)
        module = importlib.import_module("module_lora_recipe")
        with self.assertRaisesRegex(ValueError, "dense-only opt-in"):
            module.validated_recipe(self.base, dict(self.spec, schema_version=3, kind=module.KIND))
        self.spec["native_edits"] = [{"tensor": "model.language_model.norm.weight", "scale": 1.01}]
        self.spec_path.write_text(json.dumps(self.spec))
        with self.assertRaises(subprocess.CalledProcessError):
            self._native_create()
        self.assertFalse(self.dest.exists())
        self.spec.pop("native_edits")
        original = copy.deepcopy(self.spec)
        for change in ({"rank": 257}, {"dtype": "F16"}, {"header_sha256": "0"*64}):
            self.spec = copy.deepcopy(original)
            self.spec["factor_format"].update(change)
            with self.assertRaises(ValueError):
                self.recipe.validated_recipe(self.base, self.spec, allow_factor_format=True)
        self.spec = original
        for flag in ("use_dora", "use_rslora", "modules_to_save"):
            path = Path(self.spec["donor_config_path"])
            raw = path.read_bytes()
            config = json.loads(raw)
            config[flag] = True if flag != "modules_to_save" else ["lm_head"]
            path.write_text(json.dumps(config))
            self.spec["donor_config_sha256"] = self.recipe.digest(path)
            with self.assertRaises(ValueError):
                self.recipe.validated_recipe(self.base, self.spec, allow_factor_format=True)
            path.write_bytes(raw)
            self.spec["donor_config_sha256"] = self.recipe.digest(path)
        for kind in ("transpose", "extra", "nonfinite"):
            with self.subTest(kind=kind):
                changed = {k: v.copy() for k, v in factors.items()}
                first = next(iter(changed))
                if kind == "transpose":
                    changed[first] = changed[first].T
                elif kind == "extra":
                    changed["unexpected.lora_A.weight"] = np.ones((1, 1), np.float32)
                else:
                    changed[first][0, 0] = np.inf
                self._pin_f32(changed)
                with self.assertRaises(ValueError):
                    self.recipe.create(self.base, self.dest, self.spec_path, self.manifest)
                self.assertFalse(self.dest.exists())
                self.assertFalse(self.manifest.exists())

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


    def test_local_training_receipt_opt_in_binds_parent_factors_and_data(self):
        self._f32_factors()
        self.spec.pop("donor_repo")
        self.spec.pop("donor_revision")
        receipt_path = self.work/"local-training.json"
        receipt = {
            "schema_version": 1, "kind": "local_lora_training_v1", "status": "exported",
            "steps": 4, "base_unchanged": True, "parent_path": str(self.base.resolve()),
            "parent_dense_tree_sha256": self.spec["base_tree_sha256"],
            "adapter_sha256": self.spec["donor_sha256"],
            "config_sha256": self.spec["donor_config_sha256"],
            "data_sha256": "1"*64, "source_sha256": {"trainer.py": "2"*64},
            "modules": 410, "rank": 128, "dtype": "F32",
        }
        receipt_path.write_text(json.dumps(receipt))
        self.spec["local_training"] = {"version": 1, "path": str(receipt_path),
                                       "sha256": self.recipe.digest(receipt_path)}
        self.spec_path.write_text(json.dumps(self.spec))
        validated, _, _, _ = self.recipe.materialization_recipe(self.base, self.spec)
        self.assertNotIn("donor_repo", validated)
        self.assertEqual(validated["local_training_receipt"]["steps"], 4)
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.recipe.materialization_recipe(self.base, dict(self.spec, donor_repo="invented/donor"))
        for key, value in (("steps", 0), ("base_unchanged", False),
                           ("adapter_sha256", "f"*64), ("parent_path", str(self.work/"other"))):
            invalid = dict(receipt, **{key: value})
            receipt_path.write_text(json.dumps(invalid))
            self.spec["local_training"]["sha256"] = self.recipe.digest(receipt_path)
            with self.assertRaisesRegex(ValueError, "does not bind"):
                self.recipe.materialization_recipe(self.base, self.spec)
        receipt_path.write_text(json.dumps(receipt))
        self.spec["local_training"]["sha256"] = self.recipe.digest(receipt_path)
        self.spec_path.write_text(json.dumps(self.spec))
        with contextlib.redirect_stdout(io.StringIO()):
            self.recipe.create(self.base, self.dest, self.spec_path, self.manifest)
            self.recipe.verify(self.manifest)
        receipt_path.write_text(json.dumps(dict(receipt, steps=5)))
        with self.assertRaisesRegex(ValueError, "receipt path/hash"):
            self.recipe.verify(self.manifest)

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
