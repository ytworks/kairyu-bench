from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from test_wsp_weights import weights

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4_lora/donor_gradient_alignment.py"
spec = importlib.util.spec_from_file_location("donor_gradient_alignment", SCRIPT)
assert spec is not None and spec.loader is not None
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class StoredDonorAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.name = "model.language_model.layers.0.mlp.down_proj.weight"
        self.base = weights(self.root / "base/model.safetensors", {
            self.name: ("BF16", (2, 2), struct.pack("<4H", 0x3f80, 0x4000, 0x4040, 0x4080)),
        })
        self.merged = weights(self.root / "merged/model.safetensors", {
            self.name: ("BF16", (2, 2), struct.pack("<4H", 0x3f81, 0x4000, 0x4041, 0x4081)),
        })
        self.collection = self.root / "collection"
        self.collection.mkdir()
        self.coverage = {
            "actual_values_crosschecked_against_safetensors": True,
            "sampled_coordinates": 2, "sampled_matrices": 1,
            "entries": [{"tensor": self.name, "layer": 0, "family": "down",
                         "shape": [2, 2], "start": 0, "count": 2}],
        }
        self.save_arrays()
        folds = []
        for i, values in enumerate(([2, -1], [-2, 1], [0, 0])):
            np.save(self.collection / f"gradient-{i}.npy", np.array(values, dtype=np.float32), allow_pickle=False)
            folds.append({"purpose": "synthetic", "fold": i, "file": f"gradient-{i}.npy"})
        self.write(self.collection / "summary.json", {
            "status": "completed", "mode": "chat-public-answer", "source_values_unchanged": True,
            "coverage": self.coverage, "folds": folds,
        })
        self.write(self.collection / "coverage.json", self.coverage)
        self.write(self.collection / "provenance.json", {
            "mode": "chat-public-answer",
            "checkpoint": Checkpoint(self.base.parent).identity(hash_files=False),
        })
        self.manifest = self.root / "manifest.json"
        self.write(self.manifest, {
            "schema_version": 2, "kind": "dense_lora_recipe_v1",
            "source": str(self.base.parent), "candidate": str(self.merged.parent),
            "candidate_files": Checkpoint(self.merged.parent).identity()["files"],
            "candidate_tree_sha256": "declared-recipe-inventory-digest",
            "recipe": {"coefficient": .25, "donor_sha256": "fixture"},
            "edits": [{"tensor": self.name, "entry_kind": "lora_matrix", "shape": [2, 2]}],
        })

    @staticmethod
    def write(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data))

    def save_arrays(self, *, source=(1., 4.), coordinates=((0, 0), (1, 1))) -> None:
        np.savez_compressed(self.collection / "sample-plan.npz",
                            source_values=np.array(source, dtype=np.float32),
                            coordinates=np.array(coordinates, dtype=np.int64))

    def test_actual_bf16_gather_fold_dots_and_zero_norm_are_exact_and_read_only(self) -> None:
        before = [alignment.sha(p) for p in (self.base, self.merged)]
        result = alignment.analyze(self.collection, self.manifest)
        self.assertEqual([f["sampled"]["dot"] for f in result["folds"]], [-.015625, .015625, 0])
        self.assertIsNone(result["folds"][2]["sampled"]["cosine"])
        self.assertIsNone(result["folds"][2]["sampled"]["sign_agreement"])
        self.assertEqual(result["sampled_element_fraction"], .5)
        self.assertEqual(result["stored_nonzero_delta_coordinates"], 2)
        self.assertEqual(result["folds"][0]["groups"]["family:down"]["dot"], -.015625)
        self.assertEqual(before, [alignment.sha(p) for p in (self.base, self.merged)])
        json.dumps(result, allow_nan=False)

    def test_source_values_duplicate_coordinates_and_stored_payload_drift_are_rejected(self) -> None:
        self.save_arrays(source=(2., 4.))
        with self.assertRaisesRegex(ValueError, "source values"):
            alignment.analyze(self.collection, self.manifest)
        self.save_arrays(source=(1., 1.), coordinates=((0, 0), (0, 0)))
        with self.assertRaisesRegex(ValueError, "duplicate coordinates"):
            alignment.analyze(self.collection, self.manifest)
        self.save_arrays()
        manifest = json.loads(self.manifest.read_text())
        manifest["candidate_files"].append({"path": "tokenizer.json", "size": 1, "sha256": "a" * 64})
        self.write(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, "inventory"):
            alignment.analyze(self.collection, self.manifest)
        manifest["candidate_files"].pop()
        self.write(self.manifest, manifest)
        data = bytearray(self.merged.read_bytes())
        data[-1] ^= 1
        self.merged.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "differs from its recipe"):
            alignment.analyze(self.collection, self.manifest)

    def test_partial_collection_nonfinite_and_output_protection(self) -> None:
        summary = json.loads((self.collection / "summary.json").read_text())
        summary["status"] = "running"
        self.write(self.collection / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "completed"):
            alignment.analyze(self.collection, self.manifest)
        summary["status"] = "completed"
        summary["mode"] = "raw"
        self.write(self.collection / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "mode mismatch"):
            alignment.analyze(self.collection, self.manifest)
        summary["mode"] = "chat-public-answer"
        self.write(self.collection / "summary.json", summary)
        np.save(self.collection / "gradient-0.npy", np.array([np.nan, 0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "gradient array"):
            alignment.analyze(self.collection, self.manifest)
        for output in (self.manifest, self.base.parent / "new.json"):
            with self.subTest(output=output), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                alignment.main(["--collection", str(self.collection), "--merged-manifest",
                                str(self.manifest), "--output", str(output)])
