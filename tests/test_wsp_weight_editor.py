from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from kairyu_bench.wsp.weights import editor


def safetensor(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    header, payload = {}, bytearray()
    for name, (shape, values) in tensors.items():
        data = editor.encode(values, "BF16")
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class NativeEditorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        safetensor(self.base / "one.safetensors", {"norm": ([4], [1, 2, 3, 4])})
        safetensor(self.base / "two.safetensors", {"matrix": ([2, 3], [1, 2, 3, 4, 5, 6])})
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 20},
            "weight_map": {"norm": "one.safetensors", "matrix": "two.safetensors"}}))
        (self.base / "config.json").write_text('{"model_type":"fixture"}\n')
        (self.base / "README.md").write_text("nonweight bytes retained\n")
        self.spec = self.root / "spec.json"
        self.recipe = {"id": "fixture", "edits": [
            {"tensor": "norm", "indices": [0, 3], "set_values": [1.015625, 4.03125]},
            {"tensor": "matrix", "coordinates": [[1, 2]], "add": 0.0625}]}
        self.spec.write_text(json.dumps(self.recipe))
        self.dest = self.root / "candidate"
        self.manifest = self.root / "manifest.json"
        self.original = {p.name: p.read_bytes() for p in self.base.iterdir()}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create(self) -> dict:
        return editor.create(self.base, self.dest, self.spec, self.manifest, apply=True)

    def test_default_dry_run_and_exact_rounding(self) -> None:
        summary = editor.create(self.base, self.dest, self.spec, self.manifest)
        self.assertFalse(summary["applied"])
        self.assertEqual(summary["changed_elements"], 3)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.manifest.exists())
        self.assertEqual(editor.decode(editor.encode([1 + 1 / 256, 1 + 3 / 256], "BF16"), "BF16"),
                         [1, 1.015625])
        self.assertEqual(editor.encode([-0.0], "BF16"), b"\x00\x80")
        for value in (float("nan"), float("inf"), 3.4028234663852886e38):
            with self.assertRaises((ValueError, OverflowError)):
                editor.encode([value], "BF16")

    def test_create_verify_and_legacy_manifest(self) -> None:
        self.create()
        result = editor.verify(self.manifest)
        self.assertTrue(result["verified"])
        self.assertFalse(result["legacy_full_hashes_missing"])
        self.assertEqual(result["changed_elements"], 3)
        self.assertEqual(result["changed_tensors"], 2)
        for path in self.base.iterdir():
            self.assertEqual(path.read_bytes(), self.original[path.name])
            self.assertFalse(os.path.samefile(path, self.dest / path.name))
        self.assertEqual((self.dest / "README.md").read_bytes(), self.original["README.md"])
        record = editor.read_json(self.manifest)
        for row in record["files"]:
            row.pop("source_sha256")
            row.pop("candidate_sha256")
        self.manifest.write_text(json.dumps(record))
        self.assertTrue(editor.verify(self.manifest)["legacy_full_hashes_missing"])

    def test_undeclared_weight_or_config_mutation_rejected(self) -> None:
        self.create()
        path = self.dest / "two.safetensors"
        original = path.read_bytes()
        _, offset, _, _, _ = editor.tensor_location(self.dest, "matrix", sparse_matrix=True)
        raw = bytearray(original)
        raw[offset:offset + 2] = editor.encode([9], "BF16")
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "undeclared byte"):
            editor.verify(self.manifest)
        path.write_bytes(original)
        (self.dest / "config.json").write_text('{"model_type":"changed"}\n')
        with self.assertRaises(ValueError):
            editor.verify(self.manifest)

    def test_pure_manifest_rejects_metrics_operations_and_missing_cells(self) -> None:
        self.create()
        original = editor.read_json(self.manifest)
        for mutate in (
            lambda r: r["edits"][0].__setitem__("changed_count", 99),
            lambda r: r["edits"][0].__setitem__("after_hex", r["edits"][0]["before_hex"]),
            lambda r: r["spec"]["edits"][1].__setitem__("add", 1),
            lambda r: r["edits"].pop(),
            lambda r: r.__setitem__("delta_l2", 99),
        ):
            record = copy.deepcopy(original)
            mutate(record)
            with self.assertRaises(ValueError):
                editor.validate_manifest(record)

    def test_header_offset_tampering_rejected(self) -> None:
        self.create()
        record = editor.read_json(self.manifest)
        record["edits"][0]["offset"] += 2
        self.manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "headers"):
            editor.verify(self.manifest)

    def test_bad_spec_noop_duplicate_nonfinite_and_matrix_limit(self) -> None:
        for recipe in (
            {"edits": [{"tensor": "norm", "scale": 1.000001}]},
            {"edits": [{"tensor": "norm", "indices": [1, 1], "add": 1}]},
            {"edits": [{"tensor": "norm", "add": float("nan")}]},
            {"edits": [{"tensor": "matrix", "coordinates": [[0, 0], [0, 0]], "add": 1}]},
            {"edits": [{"tensor": "matrix", "coordinates": [[2, 0]], "add": 1}]},
            {"edits": [{"tensor": "matrix", "coordinates": [[0, 0]], "add": 0.00001}]},
            {"schema_version": 2, "kind": "dense_lora_recipe_v1", "edits": []},
            {**self.recipe, "matrix_batch": True, "matrix_sample_plan_sha256": "bad"},
        ):
            with self.subTest(recipe=recipe):
                with self.assertRaises(ValueError):
                    editor.plan_edits(self.base, recipe)

    def test_batched_matrix_create_verify_and_ram_plan_over_old_total_limit(self) -> None:
        from kairyu_bench.wsp.weights.checkpoint import Checkpoint
        from kairyu_bench.wsp.weights.finite_replay import load_plan

        base = self.root / "batched-base"
        base.mkdir()
        names = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(129)]
        safetensor(base / "all.safetensors", {name: ([1, 512], [1.] * 512) for name in names})
        index = base / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {name: "all.safetensors" for name in names}}))
        recipe = {"id": "batched", "matrix_batch": True, "matrix_sample_plan_sha256": "a" * 64,
                  "edits": [{"tensor": name, "coordinates": [[0, i] for i in range(512)],
                             "set_values": [1.0078125] * 512} for name in names]}
        path = self.root / "batched-spec.json"
        path.write_text(json.dumps(recipe))
        dest, manifest = self.root / "batched-copy", self.root / "batched-manifest.json"
        before = (base / "all.safetensors").read_bytes()
        editor.create(base, dest, path, manifest, apply=True)
        verified = editor.verify(manifest)
        self.assertEqual(verified["changed_elements"], 129 * 512)
        self.assertEqual((base / "all.safetensors").read_bytes(), before)
        self.assertFalse(os.path.samefile(base / "all.safetensors", dest / "all.safetensors"))
        checkpoint = Checkpoint(base)
        attestation = self.root / "attestation.json"
        attestation.write_text(json.dumps(checkpoint.identity()))
        cases, frame = self.root / "cases.json", self.root / "frame.json"
        cases.write_text("{}")
        frame.write_text("{}")
        def reference(file):
            return {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        plan = {"schema_version": 1, "architecture": "gemma4_sparse_bf16",
                "checkpoint_index_sha256": reference(index)["sha256"],
                "checkpoint_attestation": reference(attestation), "cases": reference(cases),
                "frame_evidence": reference(frame),
                "candidates": [{"id": "batched", "spec": str(path), "sha256": reference(path)["sha256"]}]}
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        _, candidates, _ = load_plan(plan_path, checkpoint)
        self.assertEqual(candidates[0]["_verified_metrics"]["changed_elements"], verified["changed_elements"])
        self.assertAlmostEqual(candidates[0]["_verified_metrics"]["delta_l2"], verified["delta_l2"])
        recipe.pop("matrix_batch")
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "in-limit"):
            editor.plan_edits(base, recipe)


        # Explicit full-unit support admits a complete row/column beyond the
        # sampled 512-cell limit while keeping all unrelated saved bytes exact.
        full_base = self.root / "full-unit-base"
        full_base.mkdir()
        prefix = "model.language_model.layers.0.mlp."
        gate, up, down = [prefix + name + "_proj.weight" for name in ("gate", "up", "down")]
        native = "model.language_model.norm.weight"
        value = "model.language_model.layers.0.self_attn.v_proj.weight"
        tensors = {
            gate: ([3, 513], [1.] * (3 * 513)),
            up: ([3, 513], [1.] * (3 * 513)),
            down: ([513, 3], [1.] * (513 * 3)),
            native: ([2], [-0., 2.]),
            value: ([2, 2], [1., 2., 3., 4.]),
        }
        safetensor(full_base / "all.safetensors", tensors)
        full_index = full_base / "model.safetensors.index.json"
        full_index.write_text(json.dumps({"weight_map": {name: "all.safetensors" for name in tensors}}))
        full_recipe = {
            "id": "full-unit",
            "matrix_batch": True,
            "matrix_sample_plan_sha256": "a" * 64,
            "matrix_full_units": {
                "schema_version": 1,
                "support_plan_sha256": "b" * 64,
                "layer_prefix": "model.language_model.layers.",
                "units": [{"layer": 0, "indices": [1]}],
            },
            "edits": [
                {"tensor": gate, "coordinates": [[1, column] for column in range(513)],
                 "set_values": [1.0078125] * 513},
                {"tensor": up, "coordinates": [[1, 0]], "set_values": [1.0078125]},
                {"tensor": down, "coordinates": [[row, 1] for row in range(513)],
                 "set_values": [1.0078125] * 513},
            ],
        }
        full_spec = self.root / "full-unit-spec.json"
        full_spec.write_text(json.dumps(full_recipe))
        full_dest, full_manifest = self.root / "full-unit-copy", self.root / "full-unit-manifest.json"
        source_bytes = (full_base / "all.safetensors").read_bytes()
        editor.create(full_base, full_dest, full_spec, full_manifest, apply=True)
        full_verified = editor.verify(full_manifest)
        self.assertEqual(full_verified["changed_elements"], 1027)
        source, saved = Checkpoint(full_base), Checkpoint(full_dest)
        full_attestation = self.root / "full-unit-attestation.json"
        full_attestation.write_text(json.dumps(source.identity()))
        full_replay_plan = {
            **plan, "checkpoint_index_sha256": reference(full_index)["sha256"],
            "checkpoint_attestation": reference(full_attestation),
            "candidates": [{"id": "full-unit", "spec": str(full_spec),
                            "sha256": reference(full_spec)["sha256"]}]}
        full_plan_path = self.root / "full-unit-replay.json"
        full_plan_path.write_text(json.dumps(full_replay_plan))
        _, ram_candidates, _ = load_plan(full_plan_path, source)
        self.assertEqual(ram_candidates[0]["_verified_metrics"]["changed_elements"], 1027)
        self.assertAlmostEqual(ram_candidates[0]["_verified_metrics"]["delta_l2"], full_verified["delta_l2"])
        self.assertEqual((full_base / "all.safetensors").read_bytes(), source_bytes)
        for name in (native, value):
            self.assertEqual(saved.read_raw(name), source.read_raw(name))
        changed = {
            gate: {513 + column for column in range(513)},
            up: {513},
            down: {row * 3 + 1 for row in range(513)},
        }
        for name, indices in changed.items():
            before, after = source.read_raw(name), saved.read_raw(name)
            actual = {i for i in range(len(before) // 2) if before[2*i:2*i+2] != after[2*i:2*i+2]}
            self.assertEqual(actual, indices)
        # Exercise the nested-manifest large-metadata dispatch with a small
        # temporary legacy threshold, without allocating a giant test file.
        with patch.object(editor, "MAX_JSON", 1024):
            self.assertEqual(editor.read_json(full_spec), full_recipe)
            full_record = editor.read_json(full_manifest)
            self.assertEqual(full_record["changed_elements"], 1027)
            with self.assertRaisesRegex(ValueError, "requires explicit full-unit"):
                editor.read_json(manifest)
        tampered = copy.deepcopy(full_record)
        tampered["spec"]["matrix_full_units"]["units"][0]["indices"] = [0]
        full_manifest.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            editor.verify(full_manifest)
        full_manifest.write_text(json.dumps(full_record))
        legacy = copy.deepcopy(full_recipe)
        legacy.pop("matrix_full_units")
        with self.assertRaisesRegex(ValueError, "in-limit"):
            editor.plan_edits(full_base, legacy)

    def test_existing_paths_overlap_symlinks_and_hardlinks(self) -> None:
        self.dest.mkdir()
        with self.assertRaises(ValueError):
            self.create()
        self.dest.rmdir()
        self.dest.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            self.create()
        self.dest.unlink()
        with self.assertRaises(ValueError):
            editor.create(self.base, self.base / "child", self.spec, self.manifest)
        self.create()
        path = self.dest / "one.safetensors"
        path.unlink()
        os.link(self.base / path.name, path)
        with self.assertRaises(ValueError):
            editor.verify(self.manifest)

    def test_edited_shard_cannot_alias_another_external_file(self) -> None:
        self.create()
        os.link(self.dest / "one.safetensors", self.root / "external-alias")
        with self.assertRaisesRegex(ValueError, "inode"):
            editor.verify(self.manifest)

    def test_failed_create_cleans_only_new_candidate(self) -> None:
        with patch.object(editor, "_stream_hashes", side_effect=OSError("injected")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.create()
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.manifest.exists())
        self.assertEqual({p.name: p.read_bytes() for p in self.base.iterdir()}, self.original)

    def test_exclusive_manifest_publication_keeps_other_writer(self) -> None:
        def competing_link(source: Path, target: Path) -> None:
            target.write_text("other writer")
            raise FileExistsError("injected collision")
        with patch.object(editor.os, "link", side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self.create()
        self.assertEqual(self.manifest.read_text(), "other writer")
        self.assertFalse(self.dest.exists())

    def test_exception_after_publication_retains_valid_checkpoint(self) -> None:
        actual = os.link
        def published_then_error(source: Path, target: Path) -> None:
            actual(source, target)
            raise OSError("after publication")
        with patch.object(editor.os, "link", side_effect=published_then_error):
            with self.assertRaisesRegex(OSError, "after publication"):
                self.create()
        self.assertTrue(editor.verify(self.manifest)["verified"])

    def test_source_mutation_aborts_and_preserves_source(self) -> None:
        actual = editor._stream_hashes
        changed = False
        def mutate_source(path: Path, edits: list[dict]) -> tuple[str, str]:
            nonlocal changed
            if not changed:
                changed = True
                (self.base / "README.md").write_text("external change")
            return actual(path, edits)
        with patch.object(editor, "_stream_hashes", side_effect=mutate_source):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                self.create()
        self.assertFalse(self.dest.exists())
        self.assertEqual((self.base / "README.md").read_text(), "external change")

    def test_restored_hash_matches_original(self) -> None:
        self.create()
        record = editor.read_json(self.manifest)
        for rel in ("one.safetensors", "two.safetensors"):
            patches = [p for p in record["edits"] if p["shard"] == rel]
            expected = hashlib.sha256((self.base / rel).read_bytes()).hexdigest()
            self.assertEqual(editor.restored_hash(self.dest / rel, patches), expected)


if __name__ == "__main__":
    unittest.main()
