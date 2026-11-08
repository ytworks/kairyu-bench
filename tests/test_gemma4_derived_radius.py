from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import test_gemma4_native_objectives as fixtures
from test_gemma4_purpose_recipes import load
from kairyu_bench.wsp.weights import editor
from kairyu_bench.wsp.weights.checkpoint import Checkpoint

with patch.dict(sys.modules, {"propose_native_objectives": fixtures.COUPLED}):
    RADIUS = load("derive_matrix_radius")
POSTCHECK = load("summarize_dependency_set_proposals")


class DerivedMatrixRadiusTests(unittest.TestCase):
    def test_calibration_to_saved_candidate_and_foreign_output_preservation(self):
        fixture = fixtures.NativeCoupledTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        entries = fixture.mixed_fixture()
        plan = fixture.plan
        plan["eligible_blocks"] = ["matrix"]
        plan["candidates"][0]["id"] = "derived-matrix"
        margin = Path(plan["collections"]["matrix"]["margin"]) / "gradient.npy"
        values = np.load(margin, allow_pickle=False)
        up = next(e for e in entries if e["family"] == "up")
        values[up["start"]] = 1.
        np.save(margin, values)
        reference = fixture.root / "reference.json"
        # Historical AB reference predates explicit matrix-only eligibility;
        # its broader native+matrix reference remains valid for this legacy route.
        legacy_reference = dict(plan)
        legacy_reference.pop("eligible_blocks")
        reference.write_text(json.dumps(legacy_reference))
        plan["reference_plan"] = {"path": str(reference), "sha256": fixture.sha(reference)}
        plan["analytic_calibration"] = {
            "calibration_seconds": 120, "candidate_seconds": 180,
            "fractional_buffer": 1 / 128, "floor": 1e-6,
            "dual_tolerance": 2e-10, "maximum_iterations": 20000,
        }
        plan["completed_collection_sha256"] = {}
        plan["completed_collection_stat"] = {}
        for group in plan["collections"].values():
            for folder in group.values():
                for path in Path(folder).iterdir():
                    plan["completed_collection_sha256"][str(path)] = fixture.sha(path)
                    plan["completed_collection_stat"][str(path)] = dict(zip(
                        ("device", "inode", "size", "mtime_ns", "ctime_ns"), RADIUS.adapter.stamp(path), strict=True))
        fixture.plan_path.write_text(json.dumps(plan))
        before = {p: fixture.sha(p) for p in fixture.checkpoint.files}
        calibration = fixture.root / "calibration"
        argv = ["--plan", str(fixture.plan_path), "--plan-sha256", fixture.sha(fixture.plan_path),
                "--output", str(calibration)]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(RADIUS.adapter.joint, "read_collection", side_effect=AssertionError("preview gradient")):
                RADIUS.main(argv)
            self.assertFalse(calibration.exists())
            RADIUS.main(argv + ["--run"])
        report = json.loads((calibration / "report.json").read_text())
        self.assertTrue(report["continuous_box_contains_solution"])
        self.assertTrue(report["rounded_box_necessary_conditions_all_met"])
        derived = calibration / "candidate-plan.json"
        candidate = fixture.root / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(derived), "--plan-sha256", fixture.sha(derived),
                         "--output", str(candidate), "--phase", "candidate", "--run"])
        result = json.loads((candidate / "proposals/report.json").read_text())["methods"][0]
        self.assertTrue(result["predicted_constraints_satisfied"])
        self.assertEqual(result["changed_elements"], 2)
        self.assertEqual(result["realized_blocks"]["native"]["changed_elements"], 0)
        spec_path = candidate / "proposals/derived-matrix.json"
        candidate_path, manifest = fixture.root / "saved", fixture.root / "saved.json"
        editor.create(fixture.checkpoint.path, candidate_path, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(candidate_path)
        qname = next(e["tensor"] for e in entries if e["family"] == "q")
        for name in (fixture.norm, fixture.scalar, qname):
            self.assertEqual(saved.read_raw(name), fixture.checkpoint.read_raw(name))
        for family in ("gate", "up"):
            name = next(e["tensor"] for e in entries if e["family"] == family)
            self.assertNotEqual(saved.read_raw(name), fixture.checkpoint.read_raw(name))
        self.assertEqual({p: fixture.sha(p) for p in fixture.checkpoint.files}, before)
        # Previously demonstrated owner bug: a rejected existing destination must
        # not gain a failure receipt or have another writer's files overwritten.
        foreign = fixture.root / "foreign"
        foreign.mkdir()
        sentinel = foreign / "started.json"
        sentinel.write_bytes(b'{"owner":"another"}\n')
        with self.assertRaisesRegex(ValueError, "fresh output"):
            RADIUS.main(["--plan", str(fixture.plan_path), "--plan-sha256", fixture.sha(fixture.plan_path),
                         "--output", str(foreign), "--run"])
        self.assertEqual(list(foreign.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b'{"owner":"another"}\n')
        # Same source checkpoint and bounds do not authorize swapping a report
        # whose observed frame/source binding differs.
        altered_report = json.loads((calibration / "report.json").read_text())
        frame_path = str(Path(plan["collections"]["matrix"]["margin"]) / "frame-evidence.json")
        altered_report["source_sha256"][frame_path] = "0" * 64
        bad_report = fixture.root / "different-observation-report.json"
        bad_report.write_text(json.dumps(altered_report))
        bad_plan = json.loads(derived.read_text())
        bad_plan["analytic_report"] = {"path": str(bad_report), "sha256": fixture.sha(bad_report)}
        bad_path = fixture.root / "different-observation-plan.json"
        bad_path.write_text(json.dumps(bad_plan))
        rejected = fixture.root / "rejected-candidate"
        with self.assertRaisesRegex(ValueError, "calibration input/source binding"):
            RADIUS.main(["--plan", str(bad_path), "--plan-sha256", fixture.sha(bad_path),
                         "--output", str(rejected), "--phase", "candidate", "--run"])
        self.assertFalse((rejected / "proposals").exists())

        # The analytic solve and persisted BF16 candidate must share the same
        # exact matrix mask. A much stronger outside gradient cannot shrink the
        # derived box, and Euclidean cost must not reuse relative scaling at b=2.
        scoped = fixtures.NativeCoupledTests()
        scoped.setUp()
        self.addCleanup(scoped.doCleanups)
        scoped_entries = scoped.mixed_fixture()
        for objective, directory in scoped.plan["collections"]["native"].items():
            renamed = scoped.root / ("native-" + objective)
            Path(directory).rename(renamed)
            scoped.plan["collections"]["native"][objective] = str(renamed)
        selected = next(e["tensor"] for e in scoped_entries if e["family"] == "q")
        excluded = next(e["tensor"] for e in scoped_entries if e["family"] == "o")
        tensor = scoped.checkpoint.tensors[selected]
        tensors = {name: (item.dtype, item.shape, scoped.checkpoint.read_raw(name))
                   for name, item in scoped.checkpoint.tensors.items()}
        tensors[selected] = ("BF16", tensor.shape, bytes.fromhex("0040") * int(np.prod(tensor.shape)))
        fixtures.weights(scoped.checkpoint.path / "model.safetensors", tensors)
        scoped.checkpoint = Checkpoint(scoped.checkpoint.path)
        identity = scoped.checkpoint.identity()
        attestation = scoped.root / "attestation.json"
        attestation.write_text(json.dumps({"checkpoint": identity}))
        p = scoped.plan
        p["checkpoint"].update(tree_sha256=identity["sha256"],
                               attestation={"path": str(attestation), "sha256": scoped.sha(attestation)})
        p["eligible_blocks"] = ["matrix"]
        p["observed_blocks"] = ["matrix"]
        # No native observation exists: calibration, candidate and postcheck must
        # use matrix source words, while saved native checkpoint bytes stay fixed.
        for directory in p["collections"].pop("native").values():
            shutil.rmtree(directory)
        p["matrix_tensors"] = [selected]
        p["matrix_families"] = ["q", "k", "o", "gate", "up", "down"]
        p["candidates"][0]["id"] = "euclidean-selected"
        p["candidates"][0]["cost_metric"] = {
            "type": "euclidean", "block_scales": {"native": 1., "matrix": 1.}}
        for family, directories in p["collections"].items():
            for folder in directories.values():
                folder = Path(folder)
                provenance = json.loads((folder / "provenance.json").read_text())
                provenance.update(checkpoint=scoped.checkpoint.identity(False),
                                  attestation_receipt_sha256=scoped.sha(attestation))
                (folder / "provenance.json").write_text(json.dumps(provenance))
                if family != "matrix":
                    continue
                with np.load(folder / "sample-plan.npz", allow_pickle=False) as arrays:
                    coordinates, source = arrays["coordinates"], arrays["source_values"].copy()
                gradient = np.zeros_like(source)
                for entry in scoped_entries:
                    if entry["tensor"] == selected:
                        source[entry["start"]:entry["start"] + entry["count"]] = 2.
                        gradient[entry["start"]] = 1.
                    elif entry["tensor"] == excluded or entry["family"] == "v":
                        gradient[entry["start"]] = 1e6
                np.savez(folder / "sample-plan.npz", coordinates=coordinates, source_values=source)
                np.save(folder / "gradient.npy", gradient)
        parent = scoped.root / "parent.json"
        parent.write_text('{"fixture":true}')
        p["parent_state"] = {"path": str(parent), "sha256": scoped.sha(parent)}
        policy = scoped.root / "functional-policy.json"
        policy.write_text(json.dumps({
            "protocol": "functional_targets_v1", "realized_raw_cap": 1.,
            "parent_state": p["parent_state"], "native_frozen": True,
            "methods": [{"id": p["candidates"][0]["id"], "target_mode": "functional",
                         "cost_metric": p["candidates"][0]["cost_metric"],
                         "matrix_families": p["matrix_families"],
                         "rounding": p["candidates"][0]["rounding"]}]}))
        p["prospective_method"] = {"path": str(policy), "sha256": scoped.sha(policy)}
        p["fixed_method_inputs"] = {str(policy): scoped.sha(policy)}
        reference = scoped.root / "reference.json"
        reference.write_text(json.dumps(p))
        p["reference_plan"] = {"path": str(reference), "sha256": scoped.sha(reference)}
        p["analytic_calibration"] = dict(plan["analytic_calibration"], metric="euclidean")
        p["completed_collection_sha256"], p["completed_collection_stat"] = {}, {}
        for group in p["collections"].values():
            for folder in group.values():
                for path in Path(folder).iterdir():
                    p["completed_collection_sha256"][str(path)] = scoped.sha(path)
                    p["completed_collection_stat"][str(path)] = dict(zip(
                        ("device", "inode", "size", "mtime_ns", "ctime_ns"), RADIUS.adapter.stamp(path), strict=True))
        scoped.plan_path.write_text(json.dumps(p))
        original_hashes = {path: scoped.sha(path) for path in scoped.checkpoint.files}
        out = scoped.root / "calibration"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(scoped.plan_path), "--plan-sha256", scoped.sha(scoped.plan_path),
                         "--output", str(out), "--run"])
        analytic = json.loads((out / "report.json").read_text())
        self.assertEqual(analytic["eligible_matrix_tensors"], [selected])
        self.assertEqual(analytic["matrix_columns"], 1)
        self.assertEqual(analytic["observed_blocks"], ["matrix"])
        self.assertEqual(analytic["baseline_source"], "matrix:correct-logprob")
        self.assertEqual(analytic["qkv_edit_columns"], 1)
        self.assertAlmostEqual(analytic["continuous_delta_l2"], 1 / 128)
        self.assertAlmostEqual(analytic["continuous_weighted_delta_l2"], 1 / 128)
        self.assertAlmostEqual(analytic["derived_relative_radius"], 3 / 256)
        derived = out / "candidate-plan.json"
        result_dir = scoped.root / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(derived), "--plan-sha256", scoped.sha(derived),
                         "--output", str(result_dir), "--phase", "candidate", "--run"])
        proposal = json.loads((result_dir / "proposals/report.json").read_text())
        self.assertEqual(proposal["methods"][0]["native_scope"]["observed_columns"], 0)
        scope = proposal["methods"][0]["column_scope"]
        self.assertEqual(scope["cross_family_baseline_parity"], "not_measured")
        self.assertTrue(all(value is None for value in scope["baseline_max_abs_difference"].values()))
        spec_path = result_dir / "proposals/euclidean-selected.json"
        spec = json.loads(spec_path.read_text())
        self.assertEqual([edit["tensor"] for edit in spec["edits"]], [selected])
        self.assertEqual(spec["edits"][0]["set_values"], [2.015625])
        saved_path, manifest = scoped.root / "saved", scoped.root / "saved.json"
        editor.create(scoped.checkpoint.path, saved_path, spec_path, manifest, apply=True, reserve_bytes=0)
        editor.verify(manifest)
        saved = Checkpoint(saved_path)
        for name in scoped.checkpoint.tensors:
            if name != selected:
                self.assertEqual(saved.read_raw(name), scoped.checkpoint.read_raw(name))
        self.assertNotEqual(saved.read_raw(selected), scoped.checkpoint.read_raw(selected))
        self.assertEqual({path: scoped.sha(path) for path in scoped.checkpoint.files}, original_hashes)
        roster = scoped.root / "roster.json"
        roster.write_text(json.dumps({"parent": p["parent_state"], "rows": [{"plan": str(derived),
            "plan_sha256": scoped.sha(derived), "output": str(result_dir / "proposals"),
            "ids": ["euclidean-selected"]}]}))
        checked = scoped.root.parent / (scoped.root.name + "-derived-postcheck.json")
        self.addCleanup(checked.unlink, missing_ok=True)
        with patch("sys.argv", ["postcheck", "--roster", str(roster), "--observation-root",
                               str(scoped.root), "--output", str(checked)]), contextlib.redirect_stdout(io.StringIO()):
            POSTCHECK.main()
        realized = json.loads(checked.read_text())["results"][0]
        self.assertTrue(realized["eligible_for_finite"])
        self.assertEqual(realized["cap_metric"], "raw")
        self.assertEqual(realized["raw_delta_l2"], 1 / 64)
        self.assertEqual(realized["actual_changed"], 1)
        mismatch = json.loads(scoped.plan_path.read_text())
        mismatch["matrix_tensors"] = [excluded]
        bad_path = scoped.root / "mismatched-mask.json"
        bad_path.write_text(json.dumps(mismatch))
        with self.assertRaisesRegex(ValueError, "reference matrix mask"):
            RADIUS.main(["--plan", str(bad_path), "--plan-sha256", scoped.sha(bad_path),
                         "--output", str(scoped.root / "rejected-mask"), "--run"])
        self.assertFalse((scoped.root / "rejected-mask").exists())

        # Complete-unit collection → quota-selected support → analytic proposal
        # → saved BF16 bytes. Large gradients outside the declared MLP support
        # cannot affect the solve, and rounded-zero support members may stay fixed.
        full = fixtures.NativeCoupledTests()
        full.setUp()
        self.addCleanup(full.doCleanups)
        full.mixed_fixture()
        with patch.dict(sys.modules, {"propose_native_objectives": fixtures.COUPLED}):
            units_module = load("matrix_full_units")
            selector = load("select_mlp_unit_support")
        names = {name: (tensor.dtype, tensor.shape, full.checkpoint.read_raw(name))
                 for name, tensor in full.checkpoint.tensors.items()}
        for family, shape in (("gate", (16, 32)), ("up", (16, 32)), ("down", (32, 16))):
            names[f"model.language_model.layers.1.mlp.{family}_proj.weight"] = (
                "BF16", shape, bytes.fromhex("803f") * int(np.prod(shape)))
        vname = "model.language_model.layers.0.self_attn.v_proj.weight"
        names[vname] = ("BF16", (32, 32), bytes.fromhex("803f") * 1024)
        fixtures.weights(full.checkpoint.path / "model.safetensors", names)
        full.checkpoint = Checkpoint(full.checkpoint.path)
        identity = full.checkpoint.identity()
        attestation = full.root / "attestation.json"
        attestation.write_text(json.dumps({"checkpoint": identity}))
        chosen = selector.select(
            {"science": np.array([3., 2., 1.]), "repair": np.array([4., 3., 2.])},
            [(0, 3), (0, 4), (1, 5)],
            [{"name": "science", "quota": 1}, {"name": "repair", "quota": 1}], 1)
        self.assertEqual([(row["layer"], row["unit"], row["quota_group"]) for row in chosen],
                         [(0, 3, "science"), (1, 5, "repair")])
        attention = full.root / "attention-reference"
        attention.mkdir()
        attn_entries = []
        for name, tensor in sorted(full.checkpoint.tensors.items()):
            match = units_module.PATTERN.fullmatch(name)
            if match and match[2] == "self_attn":
                attn_entries.append({"tensor": name, "shape": list(tensor.shape),
                                     "start": len(attn_entries), "count": 1,
                                     "family": match[3], "layer": int(match[1]), "strata": {}})
        (attention / "coverage.json").write_text(json.dumps({"entries": attn_entries}))
        np.savez(attention / "sample-plan.npz",
                 coordinates=np.zeros((len(attn_entries), 2), dtype=np.int64),
                 source_values=np.ones(len(attn_entries), dtype=np.float32))
        def pin(path):
            return {"path": str(path), "sha256": full.sha(path)}
        support = {
            "schema_version": 1, "kind": "mlp_full_unit_support_v1",
            "source_checkpoint": full.checkpoint.identity(False),
            "attestation_receipt_sha256": full.sha(attestation),
            "layer_prefix": "model.language_model.layers.",
            "units": [{"layer": row["layer"], "indices": [row["unit"]]} for row in chosen],
            "reference_attention": {
                "coverage": pin(attention / "coverage.json"),
                "sample_plan": pin(attention / "sample-plan.npz")},
        }
        support_path = full.root / "unit-support.json"
        support_path.write_text(json.dumps(support))
        complete = units_module.full_unit_samples(full.checkpoint.tensors, support)
        entries, cells = [], []
        for name, tensor in sorted(full.checkpoint.tensors.items()):
            match = units_module.PATTERN.fullmatch(name)
            if not match or (match[2] == "mlp" and name not in complete):
                continue
            coordinates, strata = complete.get(name, (np.array([[0, 0]], dtype=np.int64), {}))
            entries.append({"tensor": name, "shape": list(tensor.shape), "start": len(cells),
                            "count": len(coordinates), "family": match[3],
                            "layer": int(match[1]), "strata": strata})
            cells.extend(coordinates.tolist())
        coverage = {
            "entries": entries, "sampled_matrices": len(entries),
            "sampled_coordinates": len(cells), "sampling_mode": "mlp-full-units",
            "unit_support": support, "unit_support_sha256": full.sha(support_path)}
        p = full.plan
        p["checkpoint"].update(tree_sha256=identity["sha256"], attestation=pin(attestation))
        for directory in p["collections"].pop("native").values():
            shutil.rmtree(directory)
        p["observed_blocks"] = p["eligible_blocks"] = ["matrix"]
        p["matrix_families"] = ["gate", "up", "down"]
        p["matrix_tensors"] = sorted(complete)
        p["candidates"][0]["id"] = "full-unit-derived"
        p["candidates"][0]["cost_metric"] = {
            "type": "euclidean", "block_scales": {"native": 1., "matrix": 1.}}
        selected_cells = {}
        for objective, directory in p["collections"]["matrix"].items():
            folder = Path(directory)
            provenance = json.loads((folder / "provenance.json").read_text())
            provenance.update(checkpoint=full.checkpoint.identity(False),
                              attestation_receipt_sha256=full.sha(attestation),
                              sampling_mode="mlp-full-units", unit_support=pin(support_path))
            provenance["input_sha256"][str(support_path)] = full.sha(support_path)
            (folder / "provenance.json").write_text(json.dumps(provenance))
            (folder / "coverage.json").write_text(json.dumps(coverage))
            np.savez(folder / "sample-plan.npz", coordinates=np.asarray(cells, dtype=np.int64),
                     source_values=np.ones(len(cells), dtype=np.float32))
            gradient = np.zeros(len(cells))
            for entry in entries:
                if entry["family"] in {"q", "v"}:
                    gradient[entry["start"]] = 1e6
                elif ((objective == "correct-logprob" and entry["layer"] == 0 and entry["family"] == "gate")
                      or (objective == "margin" and entry["layer"] == 1 and entry["family"] == "up")):
                    gradient[entry["start"]] = 1.
                    selected_cells[entry["tensor"]] = tuple(cells[entry["start"]])
            np.save(folder / "gradient.npy", gradient)
        reference = full.root / "reference.json"
        reference.write_text(json.dumps(p))
        p["reference_plan"] = pin(reference)
        p["analytic_calibration"] = dict(plan["analytic_calibration"], metric="euclidean")
        p["completed_collection_sha256"], p["completed_collection_stat"] = {}, {}
        for directory in p["collections"]["matrix"].values():
            for path in Path(directory).iterdir():
                p["completed_collection_sha256"][str(path)] = full.sha(path)
                p["completed_collection_stat"][str(path)] = dict(zip(
                    ("device", "inode", "size", "mtime_ns", "ctime_ns"),
                    RADIUS.adapter.stamp(path), strict=True))
        full.plan_path.write_text(json.dumps(p))
        original = {path: full.sha(path) for path in full.checkpoint.files}
        out = full.root / "calibration"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(full.plan_path), "--plan-sha256", full.sha(full.plan_path),
                         "--output", str(out), "--run"])
        analytic = json.loads((out / "report.json").read_text())
        self.assertAlmostEqual(analytic["continuous_delta_l2"], np.sqrt(2) / 128)
        derived = out / "candidate-plan.json"
        result_dir = full.root / "candidate"
        with contextlib.redirect_stdout(io.StringIO()):
            RADIUS.main(["--plan", str(derived), "--plan-sha256", full.sha(derived),
                         "--output", str(result_dir), "--phase", "candidate", "--run"])
        spec_path = result_dir / "proposals/full-unit-derived.json"
        spec = json.loads(spec_path.read_text())
        self.assertEqual(spec["matrix_full_units"]["support_plan_sha256"], full.sha(support_path))
        self.assertEqual(spec["matrix_full_units"]["units"], support["units"])
        self.assertEqual({row["tensor"]: tuple(row["coordinates"][0]) for row in spec["edits"]},
                         selected_cells)
        saved_path, manifest = full.root / "saved", full.root / "saved.json"
        editor.create(full.checkpoint.path, saved_path, spec_path, manifest, apply=True, reserve_bytes=0)
        self.assertEqual(editor.verify(manifest)["changed_elements"], 2)
        saved = Checkpoint(saved_path)
        for name, tensor in full.checkpoint.tensors.items():
            before, after = full.checkpoint.read_raw(name), saved.read_raw(name)
            if name not in selected_cells:
                self.assertEqual(after, before)
                continue
            row, column = selected_cells[name]
            offset = 2 * (row * tensor.shape[1] + column)
            self.assertEqual(after[:offset] + after[offset + 2:], before[:offset] + before[offset + 2:])
            self.assertNotEqual(after[offset:offset + 2], before[offset:offset + 2])
        self.assertEqual({path: full.sha(path) for path in full.checkpoint.files}, original)


if __name__ == "__main__":
    unittest.main()
