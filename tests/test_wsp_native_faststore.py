from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from kairyu_bench.wsp.weights import editor, native_faststore as fast
from kairyu_bench.wsp.weights.checkpoint import Checkpoint


def shard(path: Path, values: dict[str, tuple[list[int], bytes]]) -> None:
    header, body = {}, bytearray()
    for name, (shape, raw) in values.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(body), len(body) + len(raw)]}
        body.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


class NativeFaststoreTest(unittest.TestCase):
    def test_repack_two_cumulative_candidates_exact_bytes_and_fixed_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            shard(source / "a.safetensors", {
                "norm": ([3], editor.encode([1., -0., 2.], "BF16")),
                "matrix": ([2, 2], struct.pack("<4H", 0x7fc1, 0x7f80, 0x8000, 0x4228))})
            shard(source / "b.safetensors", {"scalar": ([], editor.encode([.5], "BF16"))})
            (source / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 16},
                "weight_map": {"norm": "a.safetensors", "matrix": "a.safetensors", "scalar": "b.safetensors"}}))
            (source / "config.json").write_text('{"fixture":true}\n')
            (source / "README.md").write_text("retained metadata\n")
            original = {p.name: p.read_bytes() for p in source.iterdir()}
            rows = [{"path": p.name, "size": p.stat().st_size, "sha256": editor._hash(p)} for p in sorted(source.iterdir())]
            freeze = root / "source-freeze.json"
            freeze.write_text(json.dumps({"checkpoint_path": str(source), "control_tree_sha256": fast.digest(rows),
                "all_checkpoint_files": rows, "candidate_inventory_stat": fast.inventory(source)}))
            freeze_sha = editor._hash(freeze)
            store = root / "store"
            with patch.object(fast, "_scan_shard", side_effect=AssertionError("preview payload read")):
                self.assertFalse(fast.build(source, store, ["norm", "scalar"], freeze, freeze_sha, capacity=3)["applied"])
            self.assertFalse(store.exists())
            fast.build(source, store, ["norm", "scalar"], freeze, freeze_sha, capacity=3, apply=True)
            receipt = store / "receipt.json"
            receipt_sha = editor._hash(receipt)
            record = json.loads(receipt.read_text())
            self.assertNotEqual(record["control_tree_sha256"], record["logical_source_tree_sha256"])
            self.assertEqual(record["equivalence"]["tensor_count"], 3)
            packed = Checkpoint(store / "base")
            old = Checkpoint(source)
            self.assertEqual({n: old.read_raw(n) for n in old.tensors}, {n: packed.read_raw(n) for n in packed.tensors})
            frozen_shared = fast.state(store / "base" / fast.BULK)
            self.assertEqual(frozen_shared["nlink"], 4)
            self.assertNotEqual(frozen_shared["inode"], fast.state(source / "a.safetensors")["inode"])
            spec = root / "first-spec.json"
            spec.write_text(json.dumps({"source_checkpoint": str(source), "base_tree_sha256": fast.digest(rows),
                "edits": [{"tensor": "norm", "indices": [0], "set_values": [1.015625]}]}))
            first = root / "first"
            fast.create(receipt, receipt_sha, source, freeze, freeze_sha, spec, first, apply=True)
            first_manifest = first / "manifest.json"
            first_sha = editor._hash(first_manifest)
            verified = fast.verify(first_manifest, first_sha)
            self.assertTrue(verified["verified"])
            self.assertFalse(verified["fresh_full_model_hash"])
            self.assertEqual(verified["shared_payload_bytes_read"], 0)
            self.assertEqual(verified["changed_elements"], 1)
            spec2 = root / "second-spec.json"
            spec2.write_text(json.dumps({"source_checkpoint": str(first / "checkpoint"),
                "base_tree_sha256": json.loads(first_manifest.read_text())["control_tree_sha256"],
                "edits": [{"tensor": "norm", "indices": [0], "set_values": [1.03125]}]}))
            with self.assertRaisesRegex(ValueError, "declared specification parent"):
                fast.create(receipt, receipt_sha, first / "checkpoint", first_manifest, first_sha,
                            spec, root / "wrong-parent", apply=True)
            self.assertFalse((root / "wrong-parent").exists())
            second = root / "second"
            fast.create(receipt, receipt_sha, first / "checkpoint", first_manifest, first_sha, spec2, second, apply=True)
            second_manifest = second / "manifest.json"
            second_sha = editor._hash(second_manifest)
            self.assertTrue(fast.verify(second_manifest, second_sha)["verified"])
            self.assertTrue(fast.verify(first_manifest, first_sha)["verified"])
            self.assertEqual(Checkpoint(first / "checkpoint").read_raw("norm"), editor.encode([1.015625, -0., 2.], "BF16"))
            self.assertEqual(Checkpoint(second / "checkpoint").read_raw("norm"), editor.encode([1.03125, -0., 2.], "BF16"))
            for candidate in [first, second]:
                self.assertEqual(fast.state(candidate / "checkpoint" / fast.BULK), frozen_shared)
                self.assertEqual(fast.state(candidate / "checkpoint" / fast.NATIVE)["nlink"], 1)
            self.assertEqual(original, {p.name: p.read_bytes() for p in source.iterdir()})
            bad = root / "matrix-spec.json"
            bad.write_text(json.dumps({"edits": [{"tensor": "matrix", "coordinates": [[0, 0]], "set_values": [1.]}]}))
            with self.assertRaisesRegex(ValueError, "outside isolated native scope"):
                fast.create(receipt, receipt_sha, source, freeze, freeze_sha, bad, root / "bad", apply=True)
            self.assertFalse((root / "bad").exists())
            # An error after a slot claim must retain its bulk link; otherwise
            # it would invalidate the two already verified candidates' ctimes.
            failed = root / "failed"
            with patch.object(editor, "_stream_hashes", side_effect=OSError("fixture interrupted private proof")):
                with self.assertRaises(OSError):
                    fast.create(receipt, receipt_sha, source, freeze, freeze_sha, spec, failed, apply=True)
            self.assertTrue((failed / "failed.json").exists())
            self.assertEqual(fast.state(failed / "checkpoint" / fast.BULK), frozen_shared)
            self.assertTrue(fast.verify(first_manifest, first_sha)["verified"])
            with self.assertRaisesRegex(ValueError, "capacity exhausted"):
                fast.create(receipt, receipt_sha, source, freeze, freeze_sha, spec, root / "overflow", apply=True)
            # A signed-zero corruption outside the edited coordinates is not
            # numerically different, but must still fail the saved-byte proof.
            t = Checkpoint(second / "checkpoint").tensors["norm"]
            with t.shard.open("r+b") as stream:
                stream.seek(t.offset + 2)
                stream.write(b"\0\0")
            with self.assertRaises(ValueError):
                fast.verify(second_manifest, second_sha)
            self.assertEqual(fast.state(store / "base" / fast.BULK), frozen_shared)


if __name__ == "__main__":
    unittest.main()
