"""CPU byte-splice fixtures: provenance, exact bytes, isolation and publication."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from kairyu_bench.wsp.weights import editor
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tensor_compose", ROOT / "scripts/experiments/compose-verified-checkpoints.py"
)
compose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose)

REBASE_SPEC = importlib.util.spec_from_file_location(
    "sparse_rebase_fixture",
    ROOT / "scripts/experiments/gemma4-purpose/rebase_sparse_composition.py",
)
rebase = importlib.util.module_from_spec(REBASE_SPEC)
REBASE_SPEC.loader.exec_module(rebase)


def save(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": compose.common.sha(path)}


def files(root):
    return [
        {"path": name, "size": state["size"], "sha256": compose.common.sha(root / name)}
        for name, state in compose.common.inventory(root).items()
    ]


class ComposeVerifiedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.base, self.h, self.b = [self.work / name for name in ("base", "h", "b")]
        self.base.mkdir()
        tensors = {
            "model.language_model.norm.weight": ("BF16", [2], [0x3F80, 0x3F80]),
            "model.language_model.layers.0.mlp.up_proj.weight": (
                "BF16",
                [2, 2],
                [0x3F80, 0x4000, 0x4040, 0x4080],
            ),
            "untouched": ("BF16", [2], [0x8000, 0x3F80]),
        }
        header, data = {}, b""
        for name, (dtype, shape, words) in tensors.items():
            raw = struct.pack("<" + "H" * len(words), *words)
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [len(data), len(data) + len(raw)],
            }
            data += raw
        raw_header = json.dumps(header).encode()
        (self.base / "model.safetensors").write_bytes(
            struct.pack("<Q", len(raw_header)) + raw_header + data
        )
        (self.base / "config.json").write_text('{"model_type":"fixture"}')
        self.bs = self.work / "b-spec.json"
        save(
            self.bs,
            {
                "id": "b",
                "edits": [
                    {
                        "tensor": "model.language_model.norm.weight",
                        "indices": [0],
                        "scale": 2.0,
                    }
                ],
            },
        )
        self.bm = self.work / "b-manifest.json"
        editor.create(self.base, self.b, self.bs, self.bm, apply=True)
        shutil.copytree(self.base, self.h)
        tensor = Checkpoint(self.base).tensors[
            "model.language_model.layers.0.mlp.up_proj.weight"
        ]
        self.linear = tensor.name
        old = editor.read_at(
            self.base / tensor.relative_shard, tensor.offset, tensor.nbytes
        )
        new = struct.pack("<4H", 0x4000, 0x8000, 0x4040, 0x4080)
        with (self.h / tensor.relative_shard).open("r+b") as out:
            out.seek(tensor.offset)
            out.write(new)
        hentry = {
            "tensor": tensor.name,
            "shard": tensor.relative_shard,
            "offset": tensor.offset,
            "size": tensor.nbytes,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "source_tensor_sha256": hashlib.sha256(old).hexdigest(),
            "candidate_tensor_sha256": hashlib.sha256(new).hexdigest(),
            "changed_count": 2,
            "delta_l2": 5**0.5,
        }
        self.hm = self.work / "h-manifest.json"
        source_files, hfiles = files(self.base), files(self.h)
        self.hdoc = {
            "schema_version": 2,
            "kind": "dense_lora_recipe_v1",
            "source": str(self.base),
            "candidate": str(self.h),
            "source_files": source_files,
            "candidate_files": hfiles,
            "edits": [hentry],
            "changed_elements": 2,
            "changed_tensors": 1,
            "delta_l2": 5**0.5,
        }
        hp = save(self.hm, self.hdoc)
        self.hf = self.work / "h-freeze.json"
        hf = save(
            self.hf,
            {
                "checkpoint_path": str(self.h),
                "status": "verified_dense_cpu_recipe",
                "source_tree_sha256": compose.common.canonical(source_files),
                "candidate_inventory_stat": compose.common.inventory(self.h),
                "all_checkpoint_files": hfiles,
                "control_tree_sha256": compose.common.canonical(hfiles),
                "manifest_sha256": hp["sha256"],
            },
        )
        self.plan_path = self.work / "plan.json"
        self.dest = self.work / "candidate"
        self.manifest, self.freeze = (
            self.work / "composed.json",
            self.work / "freeze.json",
        )
        self.plan = {
            "schema_version": 1,
            "kind": compose.KIND,
            "id": "composed",
            "source": {
                "path": str(self.base),
                "tree_sha256": compose.common.canonical(source_files),
            },
            "components": [
                {
                    "path": str(self.h),
                    "manifest": hp,
                    "freeze": hf,
                    "input_verification": "frozen_stat",
                },
                {
                    "path": str(self.b),
                    "manifest": {
                        "path": str(self.bm),
                        "sha256": compose.common.sha(self.bm),
                    },
                },
            ],
            "candidate": str(self.dest),
            "manifest": str(self.manifest),
            "freeze": str(self.freeze),
        }
        self.save_plan()
        self.original = {str(p): files(p) for p in (self.base, self.h, self.b)}

    def save_plan(self):
        self.plan_pin = save(self.plan_path, self.plan)

    def run_compose(self, apply=True):
        return compose.compose(self.plan_path, self.plan_pin["sha256"], apply=apply)

    def test_sparse_rebase_preserves_parent_increment_and_rejects_whole_conflict(self):
        def freeze(root, name):
            inventory = files(root)
            path = self.work / (name + "-freeze.json")
            return save(path, {
                "checkpoint_path": str(root),
                "all_checkpoint_files": inventory,
                "candidate_inventory_stat": compose.common.inventory(root),
                "control_tree_sha256": compose.common.canonical(inventory),
            })

        original = freeze(self.base, "original")
        parent = freeze(self.b, "parent")
        source_spec = {
            "id": "earlier",
            "source_checkpoint": str(self.base),
            "base_tree_sha256": compose.common.canonical(files(self.base)),
            "matrix_batch": True,
            "matrix_sample_plan_sha256": "a" * 64,
            "edits": [
                {"tensor": self.linear, "coordinates": [[0, 0], [0, 1]],
                 "set_values": [1.5, -0.0]},
                {"tensor": "untouched", "indices": [0, 1],
                 "set_values": [-0.0, 2.0]},
            ],
        }
        spec_path = self.work / "earlier.json"
        pin = save(spec_path, source_spec)

        def invoke(output, *, parent_pin=parent, spec_pin=pin, cap=2, write=False):
            argv = [
                "rebase_sparse_composition.py",
                "--spec", spec_pin["path"], "--spec-sha", spec_pin["sha256"],
                "--source-freeze", original["path"],
                "--source-freeze-sha", original["sha256"],
                "--parent-freeze", parent_pin["path"],
                "--parent-freeze-sha", parent_pin["sha256"],
                "--output", str(output), "--candidate-id", "rebased",
                "--relative-floor", "0.1", "--weighted-cap", str(cap),
            ] + (["--write-spec"] if write else [])
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                rebase.main()

        out = self.work / "rebased-metadata"
        invoke(out)
        self.assertFalse(out.exists())
        for forbidden in (self.base / "metadata", self.b / "metadata"):
            with self.assertRaises(ValueError):
                invoke(forbidden, write=True)
            self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(ValueError, "cap"):
            invoke(out, cap=1, write=True)
        self.assertFalse(out.exists())

        ambiguous = json.loads(json.dumps(source_spec))
        ambiguous["edits"][0]["set_values"][0] = 1.5 + 2**-30
        ambiguous_pin = save(self.work / "ambiguous.json", ambiguous)
        with self.assertRaises(ValueError):
            invoke(out, spec_pin=ambiguous_pin, write=True)
        self.assertFalse(out.exists())

        conflict = self.work / "conflicting-parent"
        shutil.copytree(self.b, conflict)
        tensor = Checkpoint(conflict).tensors["untouched"]
        with (conflict / tensor.relative_shard).open("r+b") as stream:
            stream.seek(tensor.offset)
            stream.write(struct.pack("<H", 0))  # -0 to +0: numerically equal, different source word.
        conflict_pin = freeze(conflict, "conflicting")
        with self.assertRaisesRegex(ValueError, "entire edit rejected"):
            invoke(out, parent_pin=conflict_pin, write=True)
        self.assertFalse(out.exists())

        invoke(out, write=True)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["changed_elements"], 3)
        self.assertEqual(report["edited_tensors"], 2)
        self.assertAlmostEqual(report["raw_delta_l2"], 5.25**0.5)
        self.assertEqual(report["weighted_delta_l2"], 1.5)
        saved = self.work / "rebased-checkpoint"
        manifest = self.work / "rebased-manifest.json"
        editor.create(self.b, saved, out / "spec.json", manifest, apply=True)
        editor.verify(manifest)
        reader = Checkpoint(saved)
        expected = {
            self.linear: struct.pack("<4H", 0x3FC0, 0x8000, 0x4040, 0x4080),
            "untouched": struct.pack("<2H", 0x8000, 0x4000),
        }
        current = Checkpoint(self.b)
        for name, tensor in reader.tensors.items():
            actual = reader.read_raw(name, 0, tensor.numel)
            self.assertEqual(actual, expected.get(
                name, current.read_raw(name, 0, tensor.numel)
            ))
        for root in (self.base, self.b):
            self.assertEqual(files(root), self.original[str(root)])

    def test_preview_exact_splice_and_redeploy_freeze(self):
        with patch.object(
            compose,
            "hash_inputs",
            side_effect=AssertionError("preview must not hash payload"),
        ):
            result = self.run_compose(False)
        self.assertEqual(result["status"], "preview")
        self.assertFalse(self.dest.exists())
        self.dest.symlink_to(self.work / "missing-target")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.run_compose(False)
        self.assertFalse((self.work / "missing-target").exists())
        self.dest.unlink()
        result = self.run_compose()
        self.assertEqual(result["changed_tensors"], 2)
        self.assertEqual(result["changed_elements"], 3)
        self.assertAlmostEqual(result["delta_l2"], 6**0.5)
        readers = [Checkpoint(p) for p in (self.base, self.h, self.b, self.dest)]
        for name in readers[0].tensors:
            chosen = (
                1 if name == self.linear else 2 if name.endswith("norm.weight") else 0
            )
            self.assertEqual(
                readers[3].read_raw(name, 0, readers[3].tensors[name].numel),
                readers[chosen].read_raw(name, 0, readers[chosen].tensors[name].numel),
            )
        for p in (self.base, self.h, self.b):
            self.assertEqual(files(p), self.original[str(p)])
        for p in self.dest.iterdir():
            self.assertEqual(p.stat().st_nlink, 1)
            self.assertFalse(
                any(p.samefile(root / p.name) for root in (self.base, self.h, self.b))
            )
        sp = importlib.util.spec_from_file_location(
            "compose_redeploy_fixture",
            ROOT / "docs/experiments/scripts/redeploy-replica.py",
        )
        redeploy = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(redeploy)
        with contextlib.redirect_stdout(io.StringIO()):
            verified = redeploy.identity(
                types.SimpleNamespace(candidate=self.dest, freeze=self.freeze)
            )
        self.assertEqual(verified["tree_sha256"], result["tree_sha256"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.run_compose()

    def test_overlap_pin_stat_and_original_identity_rejected(self):
        self.plan["components"].append(self.plan["components"][0].copy())
        self.save_plan()
        with self.assertRaises(ValueError):
            self.run_compose(False)
        self.plan["components"].pop()
        self.save_plan()
        self.plan["source"]["tree_sha256"] = "0" * 64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "tree"):
            self.run_compose(False)
        self.plan["source"]["tree_sha256"] = compose.common.canonical(files(self.base))
        self.save_plan()
        (self.h / "config.json").write_text('{"model_type":"changed"}')
        with self.assertRaises(ValueError):
            self.run_compose(False)
        self.assertFalse(self.dest.exists())

    def test_fresh_unfrozen_component_corruption_rejected(self):
        with (self.b / "model.safetensors").open("r+b") as out:
            out.seek(-1, 2)
            out.write(b"\x00")
        with self.assertRaisesRegex(ValueError, "full-file hash"):
            self.run_compose()
        self.assertFalse(self.dest.exists())

    def test_component_undeclared_change_rejected_by_virtual_identity(self):
        # A caller-pinned full hash alone does not prove the edit list was complete.
        t = Checkpoint(self.b).tensors["untouched"]
        with (self.b / t.relative_shard).open("r+b") as out:
            out.seek(t.offset + 2)
            out.write(struct.pack("<H", 0x4000))
        manifest = json.loads(self.bm.read_text())
        for row in manifest["files"]:
            row["candidate_sha256"] = compose.common.sha(self.b / row["path"])
        self.plan["components"][1]["manifest"] = save(self.bm, manifest)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "component virtual"):
            self.run_compose()
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.manifest.exists())

    def test_undeclared_candidate_byte_and_partial_write_cleanup(self):
        original = compose.splice

        def corrupt(p):
            original(p)
            target = p["candidate"] / "config.json"
            target.write_text('{"model_type":"changed"}')

        with patch.object(compose, "splice", corrupt), self.assertRaises(ValueError):
            self.run_compose()
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.manifest.exists())

        def fail(p):
            original(p)
            raise OSError("injected after partial write")

        with patch.object(compose, "splice", fail), self.assertRaises(OSError):
            self.run_compose()
        self.assertFalse(self.dest.exists())
        for p in (self.base, self.h, self.b):
            self.assertEqual(files(p), self.original[str(p)])

    def test_exclusive_publication_and_committed_candidate_kept(self):
        original = compose.publish

        def conflict(path, value, owned):
            if path == self.manifest:
                path.write_text('{"foreign":true}')
            return original(path, value, owned)

        with (
            patch.object(compose, "publish", conflict),
            self.assertRaises(FileExistsError),
        ):
            self.run_compose()
        self.assertFalse(self.dest.exists())
        self.assertEqual(json.loads(self.manifest.read_text()), {"foreign": True})
        self.manifest.unlink()

        def after_publish(path, value, owned):
            original(path, value, owned)
            if path == self.manifest:
                raise OSError("after own manifest publication")

        with (
            patch.object(compose, "publish", after_publish),
            self.assertRaises(OSError),
        ):
            self.run_compose()
        self.assertTrue(self.dest.exists())
        self.assertEqual(json.loads(self.manifest.read_text())["kind"], compose.KIND)
        self.assertFalse(self.freeze.exists())


if __name__ == "__main__":
    unittest.main()
