from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from kairyu_bench.wsp.weights import chain, editor
from test_wsp_weight_editor import safetensor


class WeightChainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.original.mkdir()
        safetensor(self.original / "model.safetensors", {"norm": ([2], [1., 2.])})
        (self.original / "config.json").write_text('{"model_type":"fixture"}')
        self.manifests = []
        parent = self.original
        for index, edit in enumerate([{"add": 0.015625}, {"scale": 1.015625}]):
            spec, manifest = self.root / f"spec-{index}.json", self.root / f"edit-{index}.json"
            spec.write_text(json.dumps({"edits": [{"tensor": "norm", **edit}]}))
            candidate = self.root / f"candidate-{index}"
            editor.create(parent, candidate, spec, manifest, apply=True)
            self.manifests.append(manifest)
            parent = candidate
        self.candidate = parent
        shutil.rmtree(self.root / "candidate-0")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_deleted_parent_sequential_rounding_and_chunk_boundaries(self) -> None:
        result = chain.verify_chain(self.original, self.candidate, self.manifests, chunk_bytes=3)
        self.assertTrue(result["verified"])
        self.assertEqual(result["checkpoint_writes"], 0)
        self.assertEqual(result["stages"], 2)
        self.assertFalse((self.root / "candidate-0").exists())
        self.assertEqual(len(result["files"]), 2)
        _, offset, size, dtype, _ = editor.tensor_location(self.candidate, "norm")
        self.assertEqual(editor.decode(editor.read_at(self.candidate / "model.safetensors", offset, size), dtype),
                         [1.03125, 2.046875])

    def test_tampering_and_broken_chain_fail_closed(self) -> None:
        config = self.candidate / "config.json"
        original = config.read_bytes()
        config.write_bytes(original.replace(b"fixture", b"changed"))
        with self.assertRaisesRegex(ValueError, "replayed bytes"):
            chain.verify_chain(self.original, self.candidate, self.manifests)
        config.write_bytes(original)
        weights = self.original / "model.safetensors"
        raw = weights.read_bytes()
        _, offset, _, _, _ = editor.tensor_location(self.original, "norm")
        weights.write_bytes(raw[:offset] + editor.encode([9.], "BF16") + raw[offset + 2:])
        with self.assertRaisesRegex(ValueError, "before bytes"):
            chain.verify_chain(self.original, self.candidate, self.manifests)
        weights.write_bytes(raw)
        path = self.manifests[-1]
        record = json.loads(path.read_text())
        record["files"][0]["source_sha256"] = "0" * 64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "hash continuity"):
            chain.verify_chain(self.original, self.candidate, self.manifests)
