import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora/extract_text_factors.py"
spec = importlib.util.spec_from_file_location("extract_text_factors", SCRIPT)
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value))
    return digest(path)


def safetensors(path, tensors):
    header, raw, offset = {}, [], 0
    for name, dtype, shape, value in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset+len(value)]}
        raw.append(value)
        offset += len(value)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(raw))
    return hashlib.sha256(encoded).hexdigest()


class ExtractTextFactorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        names = []
        families = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"]
        for layer in range(60):
            names.extend(f"model.language_model.layers.{layer}.{family}.weight" for family in families)
            if layer < 50:
                names.append(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
        vision = "model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"
        safetensors(self.base / "model.safetensors",
                    [(n, "BF16", [2, 2], struct.pack("<4H", 0, 0x8000, 0x3F80, 0x4000)) for n in names+[vision]])
        tensors, selected, excluded = [], [], []
        # Includes an exact signed-zero and noncanonical NaN bit pattern: extraction
        # preserves storage bits; numeric merge validation is a later boundary.
        bits = struct.pack("<2I", 0x80000000, 0x7FC01234)
        for n in names+[vision]:
            for side, shape in (("A", [1, 2]), ("B", [2, 1])):
                factor = "base_model.model." + n.removesuffix(".weight") + ".lora_" + side + ".weight"
                tensors.append((factor, "F32", shape, bits))
                (selected if n in names else excluded).append(factor)
        self.adapter = self.root / "adapter_model.safetensors"
        header_sha = safetensors(self.adapter, tensors)
        self.config = self.root / "adapter_config.json"
        save(self.config, {"peft_type": "LORA", "r": 1, "lora_alpha": 1, "bias": "none",
                           "fan_in_fan_out": False, "use_dora": False, "use_rslora": False})
        self.plan = {"schema_version": 1, "kind": tool.KIND, "repo": "public/example",
                     "revision": "a"*40, "base": str(self.base), "selected": selected,
                     "excluded": excluded, "output": str(self.root / "out"),
                     "adapter": {"path": str(self.adapter), "sha256": digest(self.adapter), "header_sha256": header_sha},
                     "config": {"path": str(self.config), "sha256": digest(self.config)}}
        self.plan_path = self.root / "plan.json"
        self.pin = save(self.plan_path, self.plan)

    def call(self, apply=False):
        return tool.extract(self.plan_path, self.pin, apply=apply)

    def update(self):
        self.pin = save(self.plan_path, self.plan)

    def test_preview_and_lossless_export(self):
        before = [digest(self.adapter), digest(self.config), digest(self.base/"model.safetensors")]
        preview = self.call()
        self.assertEqual((preview["selected_factors"], preview["excluded_factors"]), (820, 2))
        self.assertFalse((self.root/"out").exists())
        result = self.call(True)
        self.assertEqual(result["status"], "completed")
        out = self.root / "out"
        receipt = json.loads((out/"extraction.json").read_text())
        derived = tool.Checkpoint(out/"adapter_model.safetensors")
        original = tool.Checkpoint(self.adapter)
        self.assertEqual(set(derived.tensors), set(self.plan["selected"]))
        for n in self.plan["selected"]:
            self.assertEqual(original.read_raw(n), derived.read_raw(n))
        self.assertEqual((out/"adapter_config.json").read_bytes(), self.config.read_bytes())
        self.assertEqual(before, [digest(self.adapter), digest(self.config), digest(self.base/"model.safetensors")])
        self.assertEqual(len(receipt["selected"]), 820)
        self.assertFalse(receipt["base_layout"]["full_weight_identity_verified"])
        self.assertNotEqual(receipt["original"]["adapter"]["sha256"], receipt["derived"]["sha256"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.call(True)

    def test_partition_shape_config_and_hash_rejected(self):
        for modify in (
            lambda p: p["excluded"].pop(),
            lambda p: p["selected"].append(p["selected"][0]),
            lambda p: p["adapter"].update(sha256="0"*64),
            lambda p: p["adapter"].update(header_sha256="0"*64),
        ):
            old = json.loads(json.dumps(self.plan))
            modify(self.plan)
            self.update()
            with self.assertRaises(ValueError):
                self.call(True)
            self.assertFalse((self.root/"out").exists())
            self.plan = old
        self.config.write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2,
                                         "bias": "none", "fan_in_fan_out": False,
                                         "use_dora": False, "use_rslora": False}))
        self.plan["config"]["sha256"] = digest(self.config)
        self.update()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.call(True)
        config = json.loads(self.config.read_text())
        config["use_dora"] = True
        save(self.config, config)
        self.plan["config"]["sha256"] = digest(self.config)
        self.update()
        with self.assertRaisesRegex(ValueError, "ordinary"):
            self.call()

    def test_fresh_path_and_partial_failure_cleanup(self):
        (self.root/"out").symlink_to(self.root/"missing")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.call(True)
        (self.root/"out").unlink()
        with mock.patch.object(tool.Checkpoint, "read_raw", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.call(True)
        self.assertFalse((self.root/"out").exists())
        stable = tool.stable
        count = 0
        def changed(p):
            nonlocal count
            count += 1
            if count == 2:
                self.config.write_bytes(self.config.read_bytes()+b" ")
            return stable(p)
        with mock.patch.object(tool, "stable", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.call(True)
        self.assertFalse((self.root/"out").exists())

    def test_exclusive_receipt_and_postcommit_preservation(self):
        original = tool.write_json_exclusive
        def foreign(path, value, owned):
            path.write_text('{"foreign": true}')
            return original(path, value, owned)
        with mock.patch.object(tool, "write_json_exclusive", side_effect=foreign):
            with self.assertRaises(FileExistsError):
                self.call(True)
        self.assertEqual(json.loads((self.root/"out/extraction.json").read_text()), {"foreign": True})
        self.assertFalse((self.root/"out/adapter_model.safetensors").exists())
        (self.root/"out/extraction.json").unlink()
        (self.root/"out").rmdir()
        def after(path, value, owned):
            original(path, value, owned)
            raise OSError("after commit")
        with mock.patch.object(tool, "write_json_exclusive", side_effect=after):
            with self.assertRaises(OSError):
                self.call(True)
        self.assertTrue((self.root/"out/adapter_model.safetensors").is_file())
        self.assertEqual(json.loads((self.root/"out/extraction.json").read_text())["status"],
                         "verified_lossless_text_factor_extraction")


if __name__ == "__main__":
    unittest.main()
