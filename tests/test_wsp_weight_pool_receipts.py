from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from kairyu_bench.wsp.weights import editor, receipts, shard_pool


def fixture(base: Path) -> None:
    base.mkdir()
    mapping = {}
    for name, value in (("a", 1.0), ("b", 2.0)):
        raw = json.dumps({name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        (base / (name + ".safetensors")).write_bytes(struct.pack("<Q", len(raw)) + raw + editor.encode([value], "BF16"))
        mapping[name] = name + ".safetensors"
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))


class PoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base, self.pool = self.root / "base", self.root / "pool"
        fixture(self.base)
        self.donors = {}
        for name in ("a.safetensors", "b.safetensors"):
            path = self.root / name
            shutil.copyfile(self.base / name, path)
            self.donors[name] = str(path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_dryrun_then_build_and_verify(self) -> None:
        result = shard_pool.build(self.base, self.pool, self.donors)
        self.assertFalse(result["applied"])
        self.assertFalse(self.pool.exists())
        shard_pool.build(self.base, self.pool, self.donors, apply=True)
        verified = shard_pool.verify(self.pool / "index.json", source=self.base)
        self.assertEqual(set(verified), set(self.donors))
        for name, path in verified.items():
            self.assertTrue(os.path.samefile(path, self.donors[name]))
            self.assertFalse(os.path.samefile(path, self.base / name))

    def test_editor_only_links_untouched_pool_shards(self) -> None:
        shard_pool.build(self.base, self.pool, self.donors, apply=True)
        spec = self.root / "spec.json"
        spec.write_text(json.dumps({"edits": [{"tensor": "a", "add": 1.0}]}))
        dest, manifest = self.root / "candidate", self.root / "manifest.json"
        planned = editor.create(self.base, dest, spec, manifest, shard_pool=self.pool / "index.json")
        self.assertEqual(planned["hardlink_bytes"], (self.base / "b.safetensors").stat().st_size)
        editor.create(self.base, dest, spec, manifest, apply=True, shard_pool=self.pool / "index.json")
        self.assertTrue(os.path.samefile(dest / "b.safetensors", self.pool / "b.safetensors"))
        self.assertFalse(os.path.samefile(dest / "a.safetensors", self.pool / "a.safetensors"))
        self.assertTrue(editor.verify(manifest)["verified"])
        shard_pool.verify(self.pool / "index.json")

    def test_any_source_inode_alias_rejected(self) -> None:
        other = self.base / "same-bytes-copy"
        shutil.copyfile(self.base / "a.safetensors", other)
        alias = self.root / "alias"
        os.link(other, alias)
        with self.assertRaisesRegex(ValueError, "aliases"):
            shard_pool.build(self.base, self.pool, {"a.safetensors": str(alias)})

    def test_wrong_content_ownership_existing_and_mutation(self) -> None:
        (self.root / "a.safetensors").write_bytes(b"wrong")
        with self.assertRaises(ValueError):
            shard_pool.build(self.base, self.pool, self.donors)
        shutil.copyfile(self.base / "a.safetensors", self.root / "a.safetensors")
        with patch.object(shard_pool.os, "getuid", return_value=os.getuid() + 1):
            with self.assertRaises(ValueError):
                shard_pool.build(self.base, self.pool, self.donors)
        shard_pool.build(self.base, self.pool, self.donors, apply=True)
        with self.assertRaises(ValueError):
            shard_pool.build(self.base, self.pool, self.donors)
        (self.base / "extra").write_text("external")
        with self.assertRaisesRegex(ValueError, "inventory"):
            shard_pool.verify(self.pool / "index.json")
        (self.base / "extra").unlink()
        (self.pool / "a.safetensors").write_bytes(b"mutated")
        with self.assertRaises(ValueError):
            shard_pool.verify(self.pool / "index.json")

    def test_source_open_failure_cleans_new_empty_target(self) -> None:
        spec = self.root / "spec.json"
        spec.write_text(json.dumps({"edits": [{"tensor": "a", "add": 1.0}]}))
        dest, manifest = self.root / "candidate", self.root / "manifest.json"
        actual = Path.open
        failed = False
        def fail_source(path: Path, mode: str = "r", *args, **kwargs):
            nonlocal failed
            if dest.exists() and path.parent == self.base and mode == "rb" and not failed:
                failed = True
                raise OSError("source open failed")
            return actual(path, mode, *args, **kwargs)
        with patch.object(Path, "open", fail_source):
            with self.assertRaisesRegex(OSError, "source open"):
                editor.create(self.base, dest, spec, manifest, apply=True)
        self.assertFalse(dest.exists())


class ReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_freeze_default_dryrun_explicit_apply_change_and_collision(self) -> None:
        source = self.root / "spec.json"
        source.write_text("{}")
        output = self.root / "frozen.json"
        self.assertFalse(receipts.freeze({"spec": source}, output)["applied"])
        self.assertFalse(output.exists())
        receipts.freeze({"spec": source}, output, apply=True)
        self.assertTrue(receipts.verify_receipt(output)["verified"])
        with self.assertRaises(ValueError):
            receipts.freeze({"spec": source}, output, apply=True)
        source.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            receipts.verify_receipt(output)

    def evidence(self):
        source = self.root / "base"
        source.mkdir()
        original = source / "config.json"
        original.write_text("{}")
        manifest_path = self.root / "manifest.json"
        spec_path, donor_path, config_path, code_path = [self.root / name for name in
                                                     ("spec.json", "donor.bin", "adapter.json", "implementation.py")]
        for p in (spec_path, config_path):
            p.write_text("{}")
        donor_path.write_bytes(b"donor fixture")
        code_path.write_text("# fixture only\n")
        source_files = [{"path": "config.json", "size": original.stat().st_size, "sha256": receipts._sha(original)}]
        candidate_files = [{"path": "config.json", "size": 2, "sha256": "c" * 64}]
        source_hash, candidate_hash = receipts._json_hash(source_files), receipts._json_hash(candidate_files)
        code = {code_path.name: receipts._sha(code_path)}
        recipe = {"donor_path": str(donor_path), "donor_config_path": str(config_path),
                  "donor_sha256": receipts._sha(donor_path), "donor_config_sha256": receipts._sha(config_path),
                  "implementation_files": code, "implementation_sha256": receipts._json_hash(code)}
        manifest = {"schema_version": 2, "kind": receipts.DENSE_KIND, "source": str(source),
                    "candidate": str(self.root / "candidate"), "candidate_tree_sha256": candidate_hash,
                    "source_tree_sha256": source_hash, "changed_elements": 2,
                    "delta_l2": 0.25, "elapsed_seconds": 1.0, "verification": receipts.CREATOR_PROOF,
                    "source_files": source_files, "candidate_files": candidate_files,
                    "recipe": recipe, "spec": {}, "spec_path": str(spec_path), "spec_sha256": receipts._sha(spec_path)}
        manifest_path.write_text(json.dumps(manifest))
        status_path, log_path = self.root / "status.json", self.root / "run.log"
        status_path.write_text(json.dumps({"status": "completed", "split": "dev", "candidate": "unit",
                                          "run_id": "unit-dev", "checkpoint": {"path": manifest["candidate"],
                                          "tree_sha256": candidate_hash},
                                          "timings": {"postcheck_seconds": 1}}))
        log_path.write_text(json.dumps(manifest) + "\n")
        proof = {"manifest_sha256": receipts._sha(manifest_path), "candidate_tree_sha256": candidate_hash,
                 "source_tree_sha256": source_hash,
                 "proof_input_states": {str(p): receipts._state(p) for p in
                                        (manifest_path, spec_path, donor_path, config_path, code_path)},
                 "source_states": {"config.json": receipts._state(original)},
                 "spec_sha256": receipts._sha(spec_path), "code_sha256": code,
                 "donor_sha256": recipe["donor_sha256"], "donor_config_sha256": recipe["donor_config_sha256"],
                 "recipe_sha256": receipts._json_hash(recipe), "implementation_sha256": receipts._json_hash(code)}
        done = {"run_id": "unit-dev", "status_path": str(status_path), "status_sha256": receipts._sha(status_path),
                "log_path": str(log_path), "log_sha256": receipts._sha(log_path), "proof": ["creator"]}
        replica = {"name": "unit-replica", "id": "cid", "image": "image"}
        ticket = {"schema_version": 1, "candidate": "unit", "fingerprints": proof,
                  "creator": done, "completed_receipt": done, "replica_ids": [replica]}
        current = {"split": "dev", "candidate": "unit", "fingerprints": proof,
                   "full_candidate_hash_recomputed": True, "fresh_tree_sha256": candidate_hash,
                   "checkpoint_path": manifest["candidate"], "manifest_path": str(manifest_path),
                   "manifest": manifest, "replicas": [{**replica, "running": True, "healthy": True,
                   "checkpoint_path": manifest["candidate"], "tree_sha256": candidate_hash}]}
        return ticket, current

    def test_conditional_followup_and_evidence_file_verification(self) -> None:
        ticket, current = self.evidence()
        result = receipts.validate_followup(ticket, current)
        self.assertTrue(result["metadata_valid"])
        self.assertFalse(result["automatic_skip_authorized"])
        tp, cp = self.root / "ticket.json", self.root / "current.json"
        tp.write_text(json.dumps(ticket))
        cp.write_text(json.dumps(current))
        self.assertTrue(receipts.verify_followup(tp, cp)["evidence_files_verified"])
        Path(ticket["creator"]["log_path"]).write_text("{}\n")
        with self.assertRaises(ValueError):
            receipts.verify_followup(tp, cp)

    def test_followup_rejects_truncated_implementation_coverage(self) -> None:
        ticket, current = self.evidence()
        ticket["fingerprints"]["code_sha256"] = {"unrelated.py": "a" * 64}
        with self.assertRaisesRegex(ValueError, "complete implementation"):
            receipts.validate_followup(ticket, current)

    def test_followup_requires_development_fresh_hash_and_exact_healthy_replicas(self) -> None:
        ticket, current = self.evidence()
        for change in (
            lambda c: c.__setitem__("split", "test"),
            lambda c: c.__setitem__("full_candidate_hash_recomputed", False),
            lambda c: c.__setitem__("fresh_tree_sha256", "wrong"),
            lambda c: c["replicas"][0].__setitem__("healthy", False),
            lambda c: c["replicas"][0].__setitem__("id", "new"),
            lambda c: c["fingerprints"].__setitem__("source_tree_sha256", "changed"),
        ):
            altered = copy.deepcopy(current)
            change(altered)
            with self.assertRaises(ValueError):
                receipts.validate_followup(ticket, altered)


if __name__ == "__main__":
    unittest.main()
