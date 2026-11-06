"""Tiny real safetensors -> custom loader iterator -> TP selected-word boundary."""
from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PLUGIN = Path(__file__).resolve().parents[1] / "scripts/experiments/vllm_sparse_overlay/src"
INSTALLED_ROOT = os.environ.get("KAIRYU_OVERLAY_INSTALLED_ROOT")
if not INSTALLED_ROOT:
    sys.path.insert(0, str(PLUGIN))
from kairyu_sparse_overlay.core import (  # noqa: E402
    FORMAT, VLLM_FILES, canonical, digest, implementation_identity,
)
if INSTALLED_ROOT:
    for module in ("kairyu_sparse_overlay.core", "kairyu_bench.wsp.weights.checkpoint",
                   "kairyu_bench.wsp.weights.editor"):
        origin = Path(sys.modules[module].__file__).resolve()
        if not origin.is_relative_to(Path(INSTALLED_ROOT).resolve()):
            raise RuntimeError(f"installed-only smoke imported a source fallback: {module}")
else:
    sys.path.pop(0)

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors", "vllm"))


@unittest.skipUnless(AVAILABLE, "optional pinned vLLM/Torch CPU runtime")
class SparseOverlayTests(unittest.TestCase):
    def test_loader_iterator_two_rank_proof_and_source_preservation(self):
        import torch
        import vllm
        from safetensors.torch import load_file, save_file
        from vllm.config import LoadConfig
        from kairyu_bench.wsp.weights.checkpoint import Checkpoint
        from kairyu_sparse_overlay.loader import SparseOverlayLoader

        prefix = "model.language_model.layers.0."
        names = {key: prefix + value + ".weight" for key, value in {
            "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj",
            "o": "self_attn.o_proj", "q": "self_attn.q_proj",
            "native": "post_attention_layernorm",
        }.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            base, draft = root / "base", root / "draft"
            base.mkdir()
            draft.mkdir()
            values = {names[key]: (torch.arange(16).reshape(4, 4) / 4).to(torch.bfloat16)
                      for key in ("gate", "up", "down", "o", "q")}
            values[names["native"]] = torch.tensor([-0.0, 1.0], dtype=torch.bfloat16)
            values[names["q"]].view(torch.int16)[0, 0] = -32768
            save_file(values, str(base / "model.safetensors"))
            save_file({"draft": torch.tensor([2.0], dtype=torch.bfloat16)}, str(draft / "model.safetensors"))

            def save_json(path, value):
                path.write_bytes(canonical(value))
                return {"path": str(path), "sha256": digest(path.read_bytes())}

            def source(path):
                cp = Checkpoint(path)
                identity = cp.identity()
                files = [{key: row[key] for key in ("path", "size", "sha256")} for row in identity["files"]]
                proof = {
                    "status": "verified_fixture_full_file_hash", "checkpoint_path": str(path),
                    "control_tree_sha256": identity["sha256"], "all_checkpoint_files": files,
                    "candidate_inventory_stat": {
                        row["path"]: {"size": row["size"], **row["stat"]} for row in identity["files"]},
                }
                return {"path": str(path), "tree_sha256": identity["sha256"],
                        "proof": save_json(root / (path.name + "-proof.json"), proof)}

            target, bypass = source(base), source(draft)
            original = (base / "model.safetensors").read_bytes()
            source_stamp = (base / "model.safetensors").stat()
            spec = {"source_checkpoint": str(base), "base_tree_sha256": target["tree_sha256"],
                    "edits": [{"tensor": names[key], "coordinates": [[0, 0], [3, 3]],
                               "set_values": [-0.5, 4.0]} for key in ("gate", "up", "down", "o")]}
            spec_ref = save_json(root / "spec.json", spec)
            receipts = root / "receipts"
            receipts.mkdir()
            source_root = Path(vllm.__file__).parent
            plan = {
                "schema_version": 1, "kind": FORMAT, "physical_source": target,
                "draft_bypass": bypass, "logical_parent": {"path": str(base), "tree_sha256": target["tree_sha256"]},
                "spec": spec_ref, "implementation_sha256": implementation_identity(),
                "tensor_parallel_size": 2, "receipt_directory": str(receipts),
                "vllm_source_sha256": {name: digest((source_root / name).read_bytes()) for name in VLLM_FILES},
            }

            def new_loader(value, filename):
                ref = save_json(root / filename, value)
                config = LoadConfig(load_format=FORMAT, model_loader_extra_config={
                    "overlay_plan": ref["path"], "overlay_plan_sha256": ref["sha256"]})
                loader = SparseOverlayLoader(config)
                self.assertEqual(config.load_format, FORMAT)
                return loader

            loader = new_loader(plan, "plan.json")
            loaded = dict(loader.get_all_weights(SimpleNamespace(model=str(base), revision=None), torch.nn.Module()))
            for key in ("gate", "up", "down", "o"):
                self.assertEqual(float(loaded[names[key]][0, 0]), -0.5)
                self.assertEqual(float(loaded[names[key]][3, 3]), 4.0)
            for key in ("q", "native"):
                self.assertTrue(torch.equal(loaded[names[key]].view(torch.int16), values[names[key]].view(torch.int16)))
            models = []
            for rank in range(2):
                model = torch.nn.Module()
                weights = {
                    prefix + "mlp.gate_up_proj.weight": torch.cat(
                        [loaded[names[key]].chunk(2, dim=0)[rank] for key in ("gate", "up")], dim=0),
                    names["down"]: loaded[names["down"]].chunk(2, dim=1)[rank].contiguous(),
                    names["o"]: loaded[names["o"]].chunk(2, dim=1)[rank].contiguous(),
                }
                for raw, value in weights.items():
                    path = raw.replace("model.language_model.", "language_model.model.").split(".")
                    module = model
                    for part in path[:-1]:
                        if part not in module._modules:
                            module.add_module(part, torch.nn.Module())
                        module = module._modules[part]
                    module.register_parameter(path[-1], torch.nn.Parameter(value.clone(), requires_grad=False))
                models.append(model)
            proofs = [loader.overlay.rank_proof(model, rank) for rank, model in enumerate(models)]
            combined = loader.overlay.aggregate(proofs)
            self.assertEqual(combined["selected_coordinates"], 8)
            self.assertTrue(combined["storage_bytes_exact"])
            self.assertFalse(combined["physical_checkpoint_is_overlay_candidate"])
            self.assertNotEqual(combined["logical_overlay_sha256"], target["tree_sha256"])
            with self.assertRaises(ValueError):
                loader.overlay.aggregate([proofs[0], proofs[0]])
            parameter = dict(models[1].named_parameters())["language_model.model.layers.0.mlp.gate_up_proj.weight"]
            parameter.view(torch.int16)[1, 3] += 1
            damaged = loader.overlay.rank_proof(models[1], 1)
            with self.assertRaisesRegex(ValueError, "word mismatch"):
                loader.overlay.aggregate([proofs[0], damaged])

            # The pinned draft bypass preserves the yielded object/storage unchanged.
            draft_loader = new_loader(plan, "draft-plan.json")
            draft_tensor = load_file(str(draft / "model.safetensors"))["draft"]
            result = list(draft_loader.overlay.iterator([("draft", draft_tensor)], str(draft)))
            self.assertIs(result[0][1], draft_tensor)
            self.assertFalse(draft_loader.overlay.finished)
            with self.assertRaisesRegex(ValueError, "unpinned"):
                draft_loader.overlay.role(str(root))
            missing = new_loader(plan, "missing-plan.json")
            with self.assertRaisesRegex(ValueError, "missing"):
                list(missing.overlay.iterator([], str(base)))

            # Empty overlay is a true pass-through, including signed-zero bits.
            empty = copy.deepcopy(plan)
            empty["spec"] = save_json(root / "empty-spec.json", {**spec, "edits": []})
            no_op = new_loader(empty, "empty-plan.json")
            items = list(values.items())
            output = list(no_op.overlay.iterator(items, str(base)))
            self.assertTrue(all(a is b for (_, a), (_, b) in zip(items, output, strict=True)))
            self.assertTrue(no_op.overlay.aggregate([no_op.overlay.rank_proof(m, r)
                                                    for r, m in enumerate(models)])["empty_overlay"])
            wrong_parent = copy.deepcopy(empty)
            wrong_parent["logical_parent"]["tree_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "logical parent"):
                new_loader(wrong_parent, "wrong-parent.json")
            self.assertEqual((base / "model.safetensors").read_bytes(), original)
            current = (base / "model.safetensors").stat()
            self.assertEqual((current.st_ino, current.st_mtime_ns, current.st_ctime_ns),
                             (source_stamp.st_ino, source_stamp.st_mtime_ns, source_stamp.st_ctime_ns))
            loader.overlay.check()
