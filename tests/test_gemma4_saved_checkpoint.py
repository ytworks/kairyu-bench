"""Persisted-byte proof, including omitted zero modules, without Torch or GEMM."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from kairyu_bench.wsp.weights.checkpoint import Checkpoint

PATH = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora/verify_saved_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("saved_module_verifier", PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class SavedModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.candidate = self.root / "source", self.root / "candidate"
        self.replay = self.root / "replay"
        self.implementation = self.root / "implementation"
        for path in (self.source, self.candidate, self.replay, self.implementation):
            path.mkdir()
        names = []
        for layer in range(60):
            families = ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
            if layer < 10:
                families.remove("self_attn.k_proj")
            names.extend(f"model.language_model.layers.{layer}.{family}.weight" for family in families)
        self.names = sorted(names)
        self.header = {n: {"dtype": "BF16", "shape": [1, 1], "data_offsets": [2*i, 2*i+2]} for i, n in enumerate(self.names)}
        raw_header = json.dumps(self.header, separators=(",", ":")).encode()
        self.start = 8 + len(raw_header)
        original = struct.pack("<Q", len(raw_header)) + raw_header + struct.pack("<H", 0x3f80)*410
        for path in (self.source, self.candidate):
            (path / "weights.safetensors").write_bytes(original)
            (path / "README.md").write_text("fixture metadata\n")
        with (self.candidate / "weights.safetensors").open("r+b") as stream:
            stream.seek(self.start)
            stream.write(struct.pack("<H", 0x4000))
        source, saved = Checkpoint(self.source), Checkpoint(self.candidate)
        self.attestation = self.root / "attestation.json"
        self.write(self.attestation, {"checkpoint": source.identity(hash_files=True)})
        self.coefs = dict.fromkeys(self.names, 0.)
        self.coefs[self.names[0]] = .25
        self.rows = []
        for name in self.names:
            tensor = saved.tensors[name]
            active = self.coefs[name] != 0
            self.rows.append({"tensor": name, "shard": tensor.relative_shard, "offset": tensor.offset,
                              "size": tensor.nbytes, "dtype": "BF16", "shape": [1, 1],
                              "lora_A": name + ".A", "lora_B": name + ".B", "entry_kind": "lora_matrix",
                              "coefficient": self.coefs[name],
                              "source_tensor_sha256": tool.hashlib.sha256(struct.pack("<H", 0x3f80)).hexdigest(),
                              "candidate_tensor_sha256": tool.hashlib.sha256(struct.pack("<H", 0x4000 if active else 0x3f80)).hexdigest(),
                              "candidate_hash_origin": "cpu_target_verified_by_device_int16_equality",
                              "changed_count": int(active), "changed_byte_count": int(active),
                              "source_l2": 1., "delta_l2": float(active), "maximum_absolute_delta": float(active)})
        implementation = {}
        for name in sorted(tool.IMPLEMENTATIONS):
            path = self.implementation / name
            path.write_bytes(("fixture " + name).encode())
            implementation[name] = tool.sha(path)
        self.kernel = self.implementation / "compiled_kernel"
        self.spec_path, self.manifest, self.plan, self.output = (self.root / n for n in ("spec.json", "manifest.json", "plan.json", "freeze.json"))
        source_files, candidate_files = self.tree(self.source), self.tree(self.candidate)
        self.spec = {"schema_version": 3, "kind": "module_lora_recipe_v1", "scope": "native_text410",
                     "id": "fixture", "module_coefficients": self.coefs, "base_tree_sha256": tool.canonical(source_files)}
        self.write(self.spec_path, self.spec)
        self.recipe = {"module_coefficients": self.coefs, "coefficients_sha256": tool.canonical(self.coefs),
                       "mapping_sha256": tool.canonical([{k: r[k] for k in tool.MAPPING_KEYS} for r in self.rows]),
                       "implementation_files": implementation, "implementation_sha256": tool.canonical(implementation),
                       "numpy_version": "fixture", "numpy_build_sha256": "f"*64, "continuous_delta_l2": .25}
        self.metrics = {"changed_elements": 1, "changed_tensors": 1, "changed_byte_elements": 1,
                        "delta_l2": 1., "actual_bf16_metrics": True}
        self.manifest_doc = {"schema_version": 3, "kind": "module_lora_recipe_v1", "source": str(self.source),
                             "candidate": str(self.candidate), "spec": self.spec, "spec_path": str(self.spec_path),
                             "spec_sha256": tool.sha(self.spec_path), "recipe": self.recipe,
                             "source_files": source_files, "candidate_files": candidate_files,
                             "source_tree_sha256": tool.canonical(source_files), "candidate_tree_sha256": tool.canonical(candidate_files),
                             "edits": [{k: v for k, v in self.rows[0].items() if k != "candidate_hash_origin"}], **self.metrics}
        self.write(self.manifest, self.manifest_doc)
        pin = {"path": str(self.attestation), "sha256": tool.sha(self.attestation)}
        self.write(self.plan, {"schema_version": 1, "architecture": "gemma4_module_lora_bf16",
                              "checkpoint_attestation": pin, "candidates": [
                                  {"id": "fixture", "spec": str(self.spec_path), "sha256": tool.sha(self.spec_path)}]})
        self.prov = {"plan_sha256": tool.sha(self.plan), "device_proof": "staged-int16", "prior_source_attestation": pin,
                     "declared_control_tree_sha256": self.spec["base_tree_sha256"], "numeric_recipe": self.recipe,
                     "input_sha256": {str(p): tool.sha(p) for p in
                                      (self.plan, self.spec_path, self.attestation, *(self.implementation / n for n in tool.IMPLEMENTATIONS))}}
        self.write(self.replay / "provenance.json", self.prov)
        self.ram = {"fixture": {"id": "fixture", "device_proof": "staged-int16", "edits": self.rows,
                                "actual_ram_storage_bytes_verified": True, "actual_ram_readback_verified": False, **self.metrics}}
        self.write(self.replay / "ram-weight-receipts.json", self.ram)
        self.summary = {"status": "completed", "source_stable": True, "error_type": None, "device_proof": "staged-int16",
                        "base_restoration": {"matched": True, "storage_bytes_exact": True, "proof_method": "staged-int16",
                                             "matrices": 410, "all_target_matrices": True},
                        "candidates": [{"id": "fixture", "spec_sha256": tool.sha(self.spec_path), "changed_elements": 1, "delta_l2": 1.}]}
        self.write(self.replay / "summary.json", self.summary)
        self.args = SimpleNamespace(candidate=self.candidate, replay=self.replay, manifest=self.manifest,
                                    spec=self.spec_path, plan=self.plan, source_attestation=self.attestation,
                                    kernel=self.kernel, implementation_dir=self.implementation, output=self.output, apply=True)
        self.pins()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    @staticmethod
    def tree(path):
        return [{"path": n, "size": s["size"], "sha256": tool.sha(path/n)} for n, s in tool.inventory(path).items()]

    def pins(self):
        for field in ("manifest", "spec", "plan", "source_attestation", "kernel"):
            setattr(self.args, field + "_sha256", tool.sha(getattr(self.args, field)))
        for key, name in (("summary", "summary.json"), ("provenance", "provenance.json"), ("ram", "ram-weight-receipts.json")):
            setattr(self.args, key + "_sha256", tool.sha(self.replay/name))

    def make_joint(self):
        names = ["model.language_model.layers.0.pre_feedforward_layernorm.weight",
                 "model.language_model.layers.1.post_attention_layernorm.weight"]
        header = {n: {"dtype": "BF16", "shape": [3], "data_offsets": [6*i, 6*i+6]} for i, n in enumerate(names)}
        encoded = json.dumps(header).encode()
        prefix = struct.pack("<Q", len(encoded)) + encoded
        original = struct.pack("<HHH", 0x3f80, 0x8000, 0x4040) * 2  # 1, -0, 3
        edited = struct.pack("<HHH", 0x4000, 0x8000, 0x4040) * 2
        (self.source / "native.safetensors").write_bytes(prefix + original)
        (self.candidate / "native.safetensors").write_bytes(prefix + edited)
        index = {"weight_map": {**dict.fromkeys(self.names, "weights.safetensors"),
                                 **dict.fromkeys(names, "native.safetensors")}}
        for directory in (self.source, self.candidate):
            self.write(directory / "model.safetensors.index.json", index)
        source, saved = Checkpoint(self.source), Checkpoint(self.candidate)
        self.write(self.attestation, {"checkpoint": source.identity(hash_files=True)})
        self.native_path, self.native_plan, self.joint_report = (self.root / n for n in
                                                               ("native.json", "native-plan.json", "joint.json"))
        self.write(self.joint_report, {"source_checkpoint": str(self.source), "disjoint_weight_tensors": True})
        binding = {"companion_required": True, "joint_report_path": str(self.joint_report),
                   "joint_report_sha256": tool.sha(self.joint_report)}
        native = {"id": "native", "edits": [
            {"tensor": n, "indices": [0], "set_values": [2.]} for n in names], **binding}
        self.write(self.native_path, native)
        self.spec.update(binding, base_tree_sha256=tool.canonical(self.tree(self.source)))
        self.write(self.spec_path, self.spec)
        creator_rows, native_rows = [], []
        for n in names:
            t = saved.tensors[n]
            old, new = original[:6], edited[:6]
            creator_rows.append({"tensor": n, "shard": t.relative_shard, "offset": t.offset, "size": t.nbytes,
                                 "dtype": "BF16", "shape": [3], "entry_kind": "native_sparse_bytes",
                                 "before_hex": old.hex(), "after_hex": new.hex(), "selected_indices": [0], "changed_indices": [0],
                                 "source_tensor_sha256": tool.hashlib.sha256(old).hexdigest(),
                                 "candidate_tensor_sha256": tool.hashlib.sha256(new).hexdigest(),
                                 "changed_count": 1, "changed_byte_count": 1, "source_l2": math.sqrt(10.),
                                 "delta_l2": 1., "maximum_absolute_delta": 1.})
            native_rows.append({"tensor": n, "indices": [0], "before_hex": old[:2].hex(), "after_hex": new[:2].hex(),
                                "source_selected_bytes_sha256": tool.hashlib.sha256(old[:2]).hexdigest(),
                                "candidate_selected_bytes_sha256": tool.hashlib.sha256(new[:2]).hexdigest(),
                                "changed_count": 1, "delta_l2": 1.})
        native_receipt = {"id": "native", "edits": native_rows, "changed_elements": 2, "changed_tensors": 2,
                          "delta_l2": math.sqrt(2.), "actual_ram_storage_bytes_verified": True,
                          "proof_method": "selected_coordinate_device_to_host_readback"}
        restoration = copy.deepcopy(native_receipt)
        restoration.update(id=None, changed_elements=0, changed_tensors=0, delta_l2=0.)
        for row in restoration["edits"]:
            row.update(after_hex=row["before_hex"], candidate_selected_bytes_sha256=row["source_selected_bytes_sha256"],
                       changed_count=0, delta_l2=0.)
        self.ram = {"fixture": {"id": "fixture", "kind": "disjoint_native_module_joint", "module": self.ram["fixture"],
                                "native": native_receipt, "native_spec_sha256": tool.sha(self.native_path),
                                "changed_elements": 3, "changed_tensors": 3, "delta_l2": math.sqrt(3.),
                                "tensor_disjoint": True, "actual_ram_storage_bytes_verified": True}}
        joint_code = self.implementation / "joint_module_recipe.py"
        joint_code.write_text("fixture joint creator\n")
        joint_hashes = {"joint_module_recipe.py": tool.sha(joint_code)}
        for name in ("editor", "checkpoint"):
            joint_hashes[name + ".py"] = tool.sha(Path(importlib.util.find_spec("kairyu_bench.wsp.weights." + name).origin))
        self.manifest_doc.update(schema_version=4, kind=tool.JOINT_KIND, spec=self.spec, spec_sha256=tool.sha(self.spec_path),
                                 native_spec=native, native_spec_path=str(self.native_path), native_spec_sha256=tool.sha(self.native_path),
                                 joint_report_path=str(self.joint_report), joint_report_sha256=tool.sha(self.joint_report),
                                 joint_implementation_files=joint_hashes, tensor_disjoint=True,
                                 source_files=self.tree(self.source), source_tree_sha256=tool.canonical(self.tree(self.source)),
                                 candidate_files=self.tree(self.candidate), candidate_tree_sha256=tool.canonical(self.tree(self.candidate)),
                                 changed_elements=3, changed_tensors=3, changed_byte_elements=3, delta_l2=math.sqrt(3.))
        self.manifest_doc["edits"] += creator_rows
        pin = {"path": str(self.attestation), "sha256": tool.sha(self.attestation)}
        native_plan = {"architecture": "gemma4_native_bf16", "base_tree_sha256": self.spec["base_tree_sha256"],
                       "checkpoint_attestation": pin, "candidates": [
                           {"id": "native", "spec": str(self.native_path), "sha256": tool.sha(self.native_path)}]}
        self.write(self.native_plan, native_plan)
        plan = {"architecture": "gemma4_module_lora_bf16", "checkpoint_attestation": pin,
                "candidates": [{"id": "fixture", "spec": str(self.spec_path), "sha256": tool.sha(self.spec_path),
                                "native_component": {"plan": str(self.native_plan), "sha256": tool.sha(self.native_plan)}}]}
        self.write(self.plan, plan)
        self.prov.update(plan_sha256=tool.sha(self.plan), prior_source_attestation=pin,
                         declared_control_tree_sha256=self.spec["base_tree_sha256"])
        for path in (self.plan, self.spec_path, self.attestation, self.native_path, self.native_plan, self.joint_report):
            self.prov["input_sha256"][str(path)] = tool.sha(path)
        self.summary["base_restoration"]["native"] = restoration
        self.summary["candidates"][0].update(spec_sha256=tool.sha(self.spec_path), changed_elements=3, delta_l2=math.sqrt(3.))
        self.args.native_spec, self.args.native_spec_sha256 = self.native_path, tool.sha(self.native_path)
        self.save_joint()

    def save_joint(self):
        self.write(self.manifest, self.manifest_doc)
        self.write(self.replay / "provenance.json", self.prov)
        self.write(self.replay / "ram-weight-receipts.json", self.ram)
        self.write(self.replay / "summary.json", self.summary)
        self.pins()

    def test_joint_freeze_covers_native_full_tensor_and_component_costs(self):
        self.make_joint()
        self.args.apply = False
        self.assertEqual(tool.verify(self.args)["native_tensors"], 2)
        self.args.apply = True
        result = tool.verify(self.args)
        self.assertEqual(len(result["tensor_sha256"]), 412)
        self.assertEqual(result["changed_elements"], 3)
        self.assertEqual(result["delta_l2"], math.sqrt(3.))
        self.assertEqual(result["component_actual_metrics"]["native"]["changed_elements"], 2)
        self.assertTrue(result["verification"]["all_native_full_tensors_match"])
        self.assertEqual(result["control_tree_sha256"], self.manifest_doc["candidate_tree_sha256"])
        self.assertEqual(result["gemm_calls"], 0)

    def test_joint_incomplete_proof_companion_mismatch_and_unselected_corruption_reject(self):
        self.make_joint()
        original = copy.deepcopy((self.manifest_doc, self.ram, self.summary))
        for mutation in ("restore", "native_byte", "overlap", "aggregate", "unknown_entry"):
            with self.subTest(mutation=mutation):
                self.manifest_doc, self.ram, self.summary = copy.deepcopy(original)
                if mutation == "restore":
                    self.summary["base_restoration"]["native"]["edits"].pop()
                elif mutation == "native_byte":
                    self.ram["fixture"]["native"]["edits"][0]["after_hex"] = "4040"
                elif mutation == "overlap":
                    self.manifest_doc["edits"][-1]["tensor"] = self.names[0]
                elif mutation == "aggregate":
                    self.manifest_doc["delta_l2"] = 2.
                else:
                    self.manifest_doc["edits"][-1]["entry_kind"] = "unknown"
                self.save_joint()
                with self.assertRaises(ValueError):
                    tool.verify(self.args)
                self.assertFalse(self.output.exists())
        self.manifest_doc, self.ram, self.summary = copy.deepcopy(original)
        self.save_joint()
        self.args.native_spec_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            tool.verify(self.args)
        self.args.native_spec_sha256 = tool.sha(self.native_path)
        t = Checkpoint(self.candidate).tensors["model.language_model.layers.0.pre_feedforward_layernorm.weight"]
        with (self.candidate / t.relative_shard).open("r+b") as stream:
            stream.seek(t.offset + 2)  # untouched negative-zero coordinate
            stream.write(struct.pack("<H", 0))
        with self.assertRaisesRegex(ValueError, "persisted tensor"):
            tool.verify(self.args)
        self.assertFalse(self.output.exists())

    def test_preview_then_saved_freeze_covers_zero_omitted_and_redeploy_contract(self):
        self.args.apply = False
        self.assertEqual(tool.verify(self.args)["status"], "preview")
        self.assertFalse(self.output.exists())
        self.args.apply = True
        result = tool.verify(self.args)
        self.assertEqual(len(result["tensor_sha256"]), 410)
        self.assertEqual(result["control_tree_sha256"], self.manifest_doc["candidate_tree_sha256"])
        self.assertEqual(result["candidate_inventory_stat"], tool.inventory(self.candidate))
        self.assertIn("NOT device-to-host", result["proof_method"])
        path = PATH.parents[3] / "docs/experiments/scripts/redeploy-replica.py"
        spec = importlib.util.spec_from_file_location("verifier_redeploy_consumer", path)
        assert spec and spec.loader
        consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consumer)
        proof = consumer.identity(SimpleNamespace(candidate=self.candidate, freeze=self.output))
        self.assertEqual(proof["tree_sha256"], result["control_tree_sha256"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            tool.verify(self.args)

    def test_actual_saved_tensor_and_non_tensor_metadata_mismatch_are_rejected(self):
        path = self.candidate / "weights.safetensors"
        original = path.read_bytes()
        with path.open("r+b") as stream:
            stream.seek(self.start + 2)  # zero coefficient omitted from creator edits
            stream.write(struct.pack("<H", 0x4000))
        with self.assertRaisesRegex(ValueError, "persisted tensor"):
            tool.verify(self.args)
        self.assertFalse(self.output.exists())
        path.write_bytes(original)
        (self.candidate / "README.md").write_text("fixture changed!\n")
        with self.assertRaisesRegex(ValueError, "full file"):
            tool.verify(self.args)
        self.assertFalse(self.output.exists())

    def test_wrong_receipt_omission_active_count_and_kernel_are_rejected(self):
        original = copy.deepcopy(self.ram)
        for mutate in ("missing_zero", "wrong_origin", "changed_count"):
            with self.subTest(mutate=mutate):
                self.ram = copy.deepcopy(original)
                if mutate == "missing_zero":
                    self.ram["fixture"]["edits"].pop()
                elif mutate == "wrong_origin":
                    self.ram["fixture"]["edits"][0]["candidate_hash_origin"] = "device_to_host_readback"
                else:
                    self.ram["fixture"]["edits"][0]["changed_count"] = 0
                self.write(self.replay/"ram-weight-receipts.json", self.ram)
                self.pins()
                with self.assertRaises(ValueError):
                    tool.verify(self.args)
                self.assertFalse(self.output.exists())
        self.write(self.replay/"ram-weight-receipts.json", original)
        self.pins()
        self.kernel.write_bytes(b"other build")
        with self.assertRaisesRegex(ValueError, "implementation"):
            tool.verify(self.args)
        self.assertFalse(self.output.exists())

    def test_unfinalized_replay_source_stat_and_input_pin_reject(self):
        self.summary["status"] = "incomplete"
        self.write(self.replay/"summary.json", self.summary)
        self.pins()
        with self.assertRaisesRegex(ValueError, "finalized"):
            tool.verify(self.args)
        self.summary["status"] = "completed"
        self.write(self.replay/"summary.json", self.summary)
        self.pins()
        self.args.ram_sha256 = "0"*64
        with self.assertRaisesRegex(ValueError, "SHA"):
            tool.verify(self.args)
        self.pins()
        path = self.source / "weights.safetensors"
        s = path.stat()
        tool.os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(ValueError, "continuity"):
            tool.verify(self.args)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
