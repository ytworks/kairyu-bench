from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/fresh-av-native-gpu4-20261107.py"
spec = importlib.util.spec_from_file_location("native_gpu_owner", SCRIPT)
OWNER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OWNER)


class NativeOwnerGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(OWNER, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        out = self.root / "analysis/run"
        out.mkdir(parents=True)
        source = self.root / "source"
        recipe = source / "scripts/experiments/gemma4-purpose"
        recipe.mkdir(parents=True)
        (source / "src").mkdir()
        sources = [recipe / name for name in ("native.py", "recipe_common.py", "objective_variants.py")]
        sources.append(source / "src/a.py")
        for path in sources:
            path.write_text("# CPU fixture; never executed\n")
        sources += [SCRIPT, SCRIPT.with_name("fresh-f01-finite.py")]
        cp = self.root / "checkpoints/code90-current-ad-helper-av-v1"
        cp.mkdir(parents=True)
        files = []
        for i in range(9):
            path = cp / str(i)
            path.write_text("tiny fixture")
            st = path.stat()
            files.append(dict(path=str(i), size=st.st_size, sha256=OWNER.sha(path), stat=dict(
                device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)))
        cases = {g: [[dict(id="behavior-" + g + str(i + 1), split="train", prompt="p", correct="a", foil="b")]
                     for i in range(2)] for g in "ABCDEF"}
        frame = dict(frames={g: [[dict(id="behavior-" + g + str(i + 1), prefix_tokens=20, correct_tokens=4, foil_tokens=5)]
                                    for i in range(2)] for g in "ABCDEF"})
        self.plan = dict(schema_version=1, maximum_collection_seconds=900, approved_for_execution=False,
                         target=dict(id=OWNER.CID, image=OWNER.IMAGE, running=True, health="healthy", started_at="old"),
                         protected={"c" + str(i): dict(id=str(i)) for i in range(26)}, container_prefix="fixture-native",
                         checkpoint=str(cp), source_root=str(source), output_root=str(out),
                         source_pins={str(p.resolve()): OWNER.sha(p) for p in sources},
                         cases=self.pin(out / "cases.json", cases), frame=self.pin(out / "frame.json", frame),
                         attestation=self.pin(out / "attestation.json", dict(checkpoint=dict(path=str(cp), files=files,
                             sha256=OWNER.TREE, hash_files=True, identity_level="content_sha256"))),
                         native_behavior_result=self.pin(out / "behavior.json", dict(status="fixture_only")))
        self.plan["cpu_previews"] = {objective: self.pin(out / (objective + "-preview.json"), dict(
            returncode=0, model_calls=0, collector_arguments=OWNER.native_args(self.plan, objective)))
            for objective in OWNER.OBJECTIVES}

    def pin(self, path, data):
        path.write_text(json.dumps(data))
        return dict(path=str(path), sha256=OWNER.sha(path))

    def test_preview_validation_never_calls_docker_or_gpu(self):
        with patch.object(OWNER.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            self.assertEqual(OWNER.validate(self.plan), ["behavior-" + g + str(i + 1) for g in "ABCDEF" for i in range(2)])
            for objective in OWNER.OBJECTIVES:
                cmd = OWNER.command(self.plan, objective, "a" * 64)
                self.assertEqual(cmd[cmd.index("--gpus") + 1], "device=4")
                self.assertEqual(cmd[cmd.index("--network") + 1], "none")
                self.assertEqual(cmd[cmd.index("--memory") + 1], "128g")
                self.assertEqual(cmd[cmd.index("--memory-swap") + 1], "128g")
                self.assertIn("--pull=never", cmd)
                self.assertIn(OWNER.IMAGE, cmd)
                self.assertEqual(cmd.count("--run"), 1)

    def test_new_parent_and_explicit_19_case_cohort_preview_rejects_omission(self):
        old = Path(self.plan["checkpoint"])
        cp = old.with_name("new-parent-faststore")
        old.rename(cp)
        identity = OWNER.read(self.plan["attestation"]["path"])["checkpoint"]
        identity.update(path=str(cp), source_stable=True)
        canonical = [{k: row[k] for k in ("path", "size", "sha256")} for row in identity["files"]]
        import hashlib
        identity["sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.plan.update(checkpoint=str(cp), checkpoint_tree_sha256=identity["sha256"])
        self.plan["target"]["id"] = "f" * 64
        self.plan["attestation"] = self.pin(Path(self.plan["attestation"]["path"]), dict(checkpoint=identity))
        cases = {g: [[dict(id=f"behavior-{g}{i + 1}", split="train", prompt="p", correct="a", foil="b")]
                      for i in range(1 if g == "J" else 2)] for g in "ABCDEFGHIJ"}
        rows = [dict(purpose=g, fold=i, id=fold[0]["id"]) for g, folds in cases.items() for i, fold in enumerate(folds)]
        self.plan["native_cohort"] = dict(count=19, rows=rows)
        self.plan["cases"] = self.pin(Path(self.plan["cases"]["path"]), cases)
        frames = {g: [[dict(id=fold[0]["id"], prefix_tokens=20, correct_tokens=4, foil_tokens=5)] for fold in folds]
                  for g, folds in cases.items()}
        self.plan["frame"] = self.pin(Path(self.plan["frame"]["path"]), dict(frames=frames))
        self.plan["cpu_previews"] = {objective: self.pin(Path(self.plan["output_root"]) / (objective + "-preview.json"), dict(
            returncode=0, model_calls=0, collector_arguments=OWNER.native_args(self.plan, objective)))
            for objective in OWNER.OBJECTIVES}
        with patch.object(OWNER.subprocess, "Popen", side_effect=AssertionError("no process")):
            self.assertEqual(OWNER.validate(self.plan), [r["id"] for r in rows])
            cases["J"].clear()
            self.plan["cases"] = self.pin(Path(self.plan["cases"]["path"]), cases)
            with self.assertRaisesRegex(ValueError, "singleton train folds"):
                OWNER.validate(self.plan)

    def test_unapproved_plan_rejected_before_helper_or_lock(self):
        path = Path(self.plan["output_root"]) / "plan.json"
        path.write_text(json.dumps(self.plan))
        with patch.object(OWNER, "helper", side_effect=AssertionError("must not inspect service")):
            with self.assertRaisesRegex(ValueError, "root-reviewed"):
                OWNER.run(self.plan, OWNER.sha(path), path)
            with self.assertRaisesRegex(ValueError, "exact plan SHA"):
                OWNER.run(self.plan, "0" * 64, path)
        self.assertFalse((self.root / "candidate-evaluation.lock").exists())

    def test_transfer_and_duplicate_case_guards(self):
        original = OWNER.read(self.plan["cases"]["path"])
        for content in (dict(split="transfer", cases=copy.deepcopy(original)), copy.deepcopy(original)):
            if "split" not in content:
                content["A"][1][0]["id"] = "behavior-A1"
            plan = copy.deepcopy(self.plan)
            plan["cases"] = self.pin(Path(plan["cases"]["path"]), content)
            with self.assertRaises(ValueError):
                OWNER.validate(plan)
        self.pin(Path(self.plan["cases"]["path"]), original)

    def test_train_envelope_and_raw_mapping_have_exact_same_ids(self):
        expected = OWNER.validate(self.plan)
        content = OWNER.read(self.plan["cases"]["path"])
        self.plan["cases"] = self.pin(Path(self.plan["cases"]["path"]), dict(split="train", cases=content))
        self.assertEqual(OWNER.validate(self.plan), expected)

    def test_input_cannot_be_mounted_writable(self):
        self.plan["cases"]["path"] = str(Path(self.plan["output_root"]) / "work/cases.json")
        with self.assertRaisesRegex(ValueError, "outside writable work"):
            OWNER.validate(self.plan)

    def test_frame_overflow_is_rejection_not_truncation(self):
        content = OWNER.read(self.plan["frame"]["path"])
        content["frames"]["A"][0][0]["prefix_tokens"] = 1024
        self.plan["frame"] = self.pin(Path(self.plan["frame"]["path"]), content)
        with self.assertRaisesRegex(ValueError, "exceeds1024"):
            OWNER.validate(self.plan)

    def test_source_and_checkpoint_stat_changes_reject(self):
        path = Path(self.plan["source_root"]) / "scripts/experiments/gemma4-purpose/native.py"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            OWNER.validate(self.plan)
        self.plan["source_pins"][str(path)] = OWNER.sha(path)
        (Path(self.plan["checkpoint"]) / "0").write_text("changed checkpoint fixture")
        with self.assertRaisesRegex(ValueError, "checkpoint stat changed"):
            OWNER.validate(self.plan)

    def fake_guard(self, drift=False, owned=None):
        calls = []
        def inspect(name):
            if owned is not None and name in (owned["Id"], owned["Name"].lstrip("/")):
                return owned
            raise subprocess.CalledProcessError(1, ["docker", "inspect", name])
        def docker(*args):
            calls.append(args)
            if args[0] == "ps":
                return ""
            if args[0] == "kill":
                owned["State"]["Running"] = False
            return ""
        def snapshot(name):
            if name == OWNER.TARGET:
                return dict(self.plan["target"], started_at="new")
            return dict(id="unexpected") if drift else self.plan["protected"][name]
        return SimpleNamespace(docker=docker, inspect=inspect, snapshot=snapshot,
            target_metrics=lambda cid: dict(queues={"running": [0], "waiting": [0]})), calls

    def test_exact_restart_and_protected_drift_reported(self):
        # Rebinding to a new ordinary parent must never restart the old AV CID.
        self.plan["target"]["id"] = "e" * 64
        guard, calls = self.fake_guard(drift=True)
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, None, True, status, "a" * 64)
        self.assertIn(("start", self.plan["target"]["id"]), calls)
        self.assertNotIn(("start", OWNER.CID), calls)
        self.assertTrue(status["restored"])
        self.assertFalse(status["protected_exact"])
        self.assertEqual(status["protected_error_type"], "ValueError")

    def test_unfinished_client_blocks_restore(self):
        class FailedProcess:
            def poll(self):
                return None
            def wait(self, timeout):
                raise subprocess.TimeoutExpired("fixture", timeout)
        guard, calls = self.fake_guard()
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, FailedProcess(), True, status, "a" * 64)
        self.assertFalse(any(a[0] == "start" for a in calls))
        self.assertEqual(status["cleanup_error_type"], "TimeoutExpired")
        self.assertFalse(status["restored"])
        self.assertIn("restoration_blocked", status)

    def owned_fixture(self, label):
        return dict(Id="collector", Image=OWNER.IMAGE,
                    Name="/" + self.plan["container_prefix"] + "-correct-logprob",
                    State=dict(Running=True), Config=dict(Labels={
                        "kairyu.native.owner-plan-sha256": label,
                        "kairyu.native.owner-root": self.plan["output_root"]}))

    def test_hung_exact_owned_collector_killed_before_original_restart(self):
        guard, calls = self.fake_guard(owned=self.owned_fixture("a" * 64))
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, None, True, status, "a" * 64)
        changes = [a for a in calls if a[0] != "ps"]
        self.assertEqual(changes, [("stop", "--time", "30", "collector"),
                                  ("kill", "collector"), ("start", OWNER.CID)])
        self.assertTrue(status["restored"])
        self.assertTrue(status["collector_cleanup"][0]["proven_inactive"])

    def test_unknown_owner_is_not_killed_and_blocks_restore(self):
        guard, calls = self.fake_guard(owned=self.owned_fixture("b" * 64))
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, None, True, status, "a" * 64)
        self.assertFalse(any(a[0] in ("start", "stop", "kill") for a in calls))
        self.assertFalse(status["restored"])
        self.assertIn("restoration_blocked", status)

    def test_late_create_during_client_wait_is_rechecked_before_restart(self):
        guard, calls = self.fake_guard()
        owned = self.owned_fixture("a" * 64)
        state = dict(created=False, client_done=False)
        work = Path(self.plan["output_root"]) / "work"
        work.mkdir()
        original_inspect = guard.inspect
        original_docker = guard.docker
        def inspect(name):
            if state["created"] and name in (owned["Id"], owned["Name"].lstrip("/")):
                return owned
            return original_inspect(name)
        def docker(*args):
            if args[0] == "kill":
                calls.append(args)
                owned["State"]["Running"] = False
                return ""
            if args[0] == "start":
                self.assertFalse(owned["State"]["Running"], "late-created collector must be stopped first")
            return original_docker(*args)
        class LateClient:
            def poll(self):
                return 0 if state["client_done"] else None
            def wait(self, timeout):
                state.update(created=True, client_done=True)
                (work / "correct-logprob.cid").write_text(owned["Id"])
                return 0
        guard.inspect, guard.docker = inspect, docker
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, LateClient(), True, status, "a" * 64)
        self.assertTrue(status["restored"])
        final = [r for r in status["collector_cleanup"] if r["phase"] == "after_client_wait"]
        self.assertEqual(final[0]["cid"], owned["Id"])
        changes = [a for a in calls if a[0] != "ps"]
        self.assertEqual(changes, [("stop", "--time", "30", "collector"),
                                  ("kill", "collector"), ("start", OWNER.CID)])

    def test_docker_unavailable_cannot_be_mistaken_for_removed_collector(self):
        guard, calls = self.fake_guard()
        def broken(*args):
            raise subprocess.CalledProcessError(1, ["docker", *args])
        guard.docker = broken
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, None, True, status, "a" * 64)
        self.assertFalse(status["restored"])
        self.assertIn("restoration_blocked", status)
        self.assertFalse(calls)


class FiniteOwnerGuards(unittest.TestCase):
    pin = NativeOwnerGuards.pin
    fake_guard = NativeOwnerGuards.fake_guard

    def setUp(self):
        from kairyu_bench.wsp.weights import checkpoint, editor, finite_replay
        from test_wsp_weights import weights
        NativeOwnerGuards.setUp(self)
        cp = Path(self.plan["checkpoint"])
        for path in cp.iterdir():
            path.unlink()
        self.tensor = "model.language_model.norm.weight"
        weights(cp / "model.safetensors", {self.tensor: ("BF16", (2,), bytes.fromhex("803f0040"))})
        (cp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {self.tensor: "model.safetensors"}}))
        for name in checkpoint.CONFIG_FILES[:7]:
            (cp / name).write_text("{}")
        identity = checkpoint.Checkpoint(cp).identity()
        tree_patch = patch.object(OWNER, "TREE", identity["sha256"])
        tree_patch.start()
        self.addCleanup(tree_patch.stop)
        out = Path(self.plan["output_root"])
        self.plan["attestation"] = self.pin(out / "attestation.json", dict(checkpoint=identity))
        self.plan["mode"] = "finite"
        self.plan.pop("cpu_previews")
        spec = self.pin(out / "spec.json", dict(id="fixture", edits=[dict(tensor=self.tensor, indices=[0], set_values=[1.0078125])]))
        self.candidate = self.root / "checkpoints/candidate"
        manifest = out / "manifest.json"
        editor.create(cp, self.candidate, Path(spec["path"]), manifest, apply=True)
        verification = editor.verify(manifest)
        finite = dict(spec=spec, manifest=dict(path=str(manifest), sha256=OWNER.sha(manifest)),
                      verify=self.pin(out / "verify.json", verification),
                      candidate_identity=self.pin(out / "candidate-identity.json", checkpoint.Checkpoint(self.candidate).identity()))
        finite["plan"] = self.pin(out / "finite-plan.json", dict(schema_version=1, architecture="gemma4_native_bf16",
            base_tree_sha256=OWNER.TREE, checkpoint_index_sha256=OWNER.sha(cp / "model.safetensors.index.json"),
            checkpoint_attestation=self.plan["attestation"], cases=self.plan["cases"], frame_evidence=self.plan["frame"],
            candidates=[dict(id="fixture", spec=spec["path"], sha256=spec["sha256"])], observation_policy=OWNER.finite_policy()))
        self.plan["finite"] = finite
        modules = patch.object(OWNER, "finite_modules", return_value=(checkpoint, editor, finite_replay))
        modules.start()
        self.addCleanup(modules.stop)
        self.replay = finite_replay
        runtime = Path(self.plan["source_root"]) / "src/kairyu_bench/wsp/weights/finite_replay.py"
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(Path(finite_replay.__file__).read_bytes())
        self.plan["source_pins"][str(runtime)] = OWNER.sha(runtime)

    def replace_pin(self, key, update):
        pin = self.plan["finite"][key]
        data = OWNER.read(pin["path"])
        update(data)
        self.plan["finite"][key] = self.pin(Path(pin["path"]), data)

    def test_metadata_preview_reads_real_bf16_without_model_or_subprocess(self):
        with patch.object(self.replay, "run", side_effect=AssertionError("not a preview")), patch.object(
                OWNER.subprocess, "Popen", side_effect=AssertionError("no process")):
            self.assertEqual(len(OWNER.validate(self.plan)), 12)
            metadata = OWNER.finite_metadata(self.plan)
        self.assertEqual(metadata["metrics"], dict(changed_elements=1, changed_tensors=1, delta_l2=1/128))
        self.assertNotEqual(metadata["proofs"]["base"], metadata["proofs"]["fixture"])
        self.assertEqual(metadata["proofs"]["base"], metadata["proofs"]["restored_base"])
        self.assertEqual(OWNER.stages(self.plan), ("finite",))
        cmd = OWNER.command(self.plan, "finite", "a" * 64)
        self.assertIn("kairyu_bench.wsp.weights.finite_replay", cmd)
        self.assertNotIn("--run", cmd)  # CLI has no preview gate; owner does.
        with self.assertRaisesRegex(ValueError, "job differs"):
            OWNER.command(self.plan, "margin", "a" * 64)

    def test_multiple_candidates_and_changed_policy_rejected(self):
        original = OWNER.read(self.plan["finite"]["plan"]["path"])
        for update in (lambda d: d["candidates"].append(dict(d["candidates"][0])),
                       lambda d: d["observation_policy"].update(answer_batching="paired")):
            self.replace_pin("plan", update)
            with self.assertRaises(ValueError):
                OWNER.validate(self.plan)
            self.plan["finite"]["plan"] = self.pin(Path(self.plan["finite"]["plan"]["path"]), original)

    def test_verified_boolean_cannot_hide_file_hash_or_spec_mismatch(self):
        self.replace_pin("verify", lambda d: d["files"][0].update(candidate_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "file hashes differ"):
            OWNER.validate(self.plan)
        self.replace_pin("manifest", lambda d: d.update(spec_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "manifest/spec"):
            OWNER.validate(self.plan)

    def test_saved_candidate_change_rejected_even_with_old_verified_receipt(self):
        (self.candidate / "config.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "stat changed"):
            OWNER.validate(self.plan)

    def test_actual_saved_bf16_differs_from_replay_spec(self):
        from kairyu_bench.wsp.weights.checkpoint import Checkpoint
        # A coherent new full-hash/stat receipt alone cannot replace the fixed exact spec.
        file = self.candidate / "model.safetensors"
        data = bytearray(file.read_bytes())
        data[-4:] = bytes.fromhex("823f0040")
        file.write_bytes(data)
        identity = Checkpoint(self.candidate).identity()
        self.plan["finite"]["candidate_identity"] = self.pin(Path(self.plan["finite"]["candidate_identity"]["path"]), identity)
        digest = {r["path"]: r["sha256"] for r in identity["files"]}
        for key in ("manifest", "verify"):
            self.replace_pin(key, lambda d: [r.update(candidate_sha256=digest[r["path"]]) for r in d["files"]])
        with self.assertRaisesRegex(ValueError, "saved candidate BF16 values"):
            OWNER.validate(self.plan)

    def result_fixture(self):
        result = Path(self.plan["output_root"]) / "result-fixture"
        result.mkdir()
        metadata = OWNER.finite_metadata(self.plan)
        for label in ("base", "fixture", "restored_base"):
            value = .5 if label == "fixture" else 0.
            OWNER.save(result / (label + ".json"), dict(values={g: [[value], [value]] for g in "ABCDEF"},
                means=[value] * 12, correct_nll=[1. - value] * 12, kl=[value] * 12, seconds=1.))
        restore = dict(attempted=True, forward_completed=True, matched=True, storage_bytes_exact=True)
        proofs = {label: dict(method="selected_bf16_cpu_readback", storage_bytes_exact=True, tensors=rows)
                  for label, rows in metadata["proofs"].items()}
        execution = dict(resolved_attention_dispatch={"text_config": "sdpa"}, cuda_allocator=dict(requested_gib=88),
                         automatic_attention_api_or_length_retry=False)
        summary = dict(status="completed", source_stable=True, error_type=None, observation_execution=execution,
            labels=[g + "-" + str(i) for g in "ABCDEF" for i in range(2)], answer_batching="serial",
            observation_policy=OWNER.finite_policy(), new_gradients=0, generated_tokens=0,
            candidates=[dict(id="fixture", **metadata["metrics"], margin_delta=[.5] * 12,
                             correct_nll_delta=[-.5] * 12, kl_base_to_candidate=[.5] * 12)],
            base_restoration=restore, selected_byte_proofs=proofs)
        for name, value in {"summary": summary, "base-restoration": restore, "selected-byte-proofs": proofs,
                            "frame-evidence": OWNER.pinned(self.plan["frame"]), "provenance": dict(
                                plan_sha256=self.plan["finite"]["plan"]["sha256"], expected_base_tree_sha256=OWNER.TREE,
                                source_hashes=metadata["source_hashes"], runner_sha256=OWNER.sha(Path(self.replay.__file__)),
                                observation_execution=execution)}.items():
            OWNER.save(result / (name + ".json"), value)
        return result

    def test_completed_summary_cannot_hide_partial_sweep_or_wrong_byte_application(self):
        result = self.result_fixture()
        OWNER.finite_complete(self.plan, result)
        path = result / "fixture.json"
        sweep = OWNER.read(path)
        original = copy.deepcopy(sweep)
        sweep["values"]["F"].pop()
        path.write_text(json.dumps(sweep))
        with self.assertRaisesRegex(ValueError, "partial finite sweep"):
            OWNER.finite_complete(self.plan, result)
        path.write_text(json.dumps(original))
        for name in ("summary", "selected-byte-proofs"):
            path = result / (name + ".json")
            data = OWNER.read(path)
            proofs = data["selected_byte_proofs"] if name == "summary" else data
            proofs["fixture"]["tensors"][0]["actual_bf16_sha256"] = "0" * 64
            path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "selected BF16 proof differs"):
            OWNER.finite_complete(self.plan, result)

    def test_claimed_restoration_cannot_hide_numerical_failure(self):
        result = self.result_fixture()
        path = result / "restored_base.json"
        data = OWNER.read(path)
        data["kl"][0] = .1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "restored finite numerical"):
            OWNER.finite_complete(self.plan, result)

    def test_actual_attention_or_source_differs_from_pinned_result(self):
        result = self.result_fixture()
        path = result / "provenance.json"
        original = OWNER.read(path)
        for update in (lambda d: d.update(runner_sha256="0" * 64), lambda d:
                       d["observation_execution"]["resolved_attention_dispatch"].update(text_config="eager")):
            changed = copy.deepcopy(original)
            update(changed)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                OWNER.finite_complete(self.plan, result)

    def test_finite_late_create_reuses_owned_cleanup(self):
        guard, calls = self.fake_guard()
        owned = dict(Id="finite-own", Image=OWNER.IMAGE, Name="/" + self.plan["container_prefix"] + "-finite",
            State=dict(Running=True), Config=dict(Labels={"kairyu.native.owner-plan-sha256": "a" * 64,
                                                        "kairyu.native.owner-root": self.plan["output_root"]}))
        state = dict(done=False)
        work = Path(self.plan["output_root"]) / "work"
        work.mkdir()
        original_inspect, original_docker = guard.inspect, guard.docker
        def inspect(name):
            return owned if state["done"] and name in (owned["Id"], owned["Name"].lstrip("/")) else original_inspect(name)
        def docker(*args):
            if args[0] == "kill":
                calls.append(args)
                owned["State"]["Running"] = False
                return ""
            if args[0] == "start":
                self.assertFalse(owned["State"]["Running"])
            return original_docker(*args)
        class Client:
            def poll(self):
                return 0 if state["done"] else None
            def wait(self, timeout):
                state["done"] = True
                (work / "finite.cid").write_text(owned["Id"])
        guard.inspect, guard.docker = inspect, docker
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, Client(), True, status, "a" * 64)
        self.assertTrue(status["restored"])
        self.assertEqual(len(status["collector_cleanup"]), 2)
        self.assertEqual([a for a in calls if a[0] != "ps"], [
            ("stop", "--time", "30", "finite-own"), ("kill", "finite-own"), ("start", OWNER.CID)])


class TrainingOwnerGuards(unittest.TestCase):
    pin = NativeOwnerGuards.pin

    def setUp(self):
        import hashlib
        NativeOwnerGuards.setUp(self)
        for key in ("cases", "frame", "native_behavior_result", "cpu_previews", "maximum_collection_seconds"):
            self.plan.pop(key)
        identity = OWNER.read(self.plan["attestation"]["path"])["checkpoint"]
        identity["source_stable"] = True
        canonical = [{k: r[k] for k in ("path", "size", "sha256")} for r in identity["files"]]
        identity["sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.plan["attestation"] = self.pin(Path(self.plan["attestation"]["path"]), dict(checkpoint=identity))
        self.plan.update(mode="train-lora", checkpoint_tree_sha256=identity["sha256"],
                         maximum_training_seconds=2400, training_startup_allowance_seconds=300)
        self.plan["target"]["id"] = "e" * 64
        out = Path(self.plan["output_root"])
        trainer = Path(self.plan["source_root"]) / "scripts/experiments/gemma4-purpose/train_lora.py"
        trainer.write_text("# pinned CPU fixture, never executed\n")
        self.plan["source_pins"][str(trainer)] = OWNER.sha(trainer)
        data = out / "train.jsonl"
        data.write_text(' {"task_id":601,"text":"p","code":"pass"}\n{"task_id":602,"text":"q","code":"pass"}\n')
        holdout = out / "holdout.jsonl"
        holdout.write_text('{"task_id":603,"text":"held out","code":"pass"}\n')
        split = self.pin(out / "split.json", dict(training_ids=[601, 602], holdout_ids=[603], train_rows=2, holdout_rows=1))
        frame = self.pin(out / "train-frame.json", dict(training_ids=[601, 602], maximum_tokens=128))
        self.task = dict(kind="gemma4_lora_sft_v1", checkpoint=dict(path=self.plan["checkpoint"],
                         tree_sha256=identity["sha256"], attestation=self.plan["attestation"]),
                         data=dict(path=str(data), sha256=OWNER.sha(data), split="train", ids=[601, 602], split_summary=split),
                         frame_evidence=frame, maximum_seconds=1800, continuation_wait_seconds=300)
        quality_summary = self.pin(out / "reference-quality.json", dict(status="completed"))
        quality = dict(status="completed", training_data_sha256=OWNER.sha(data), holdout_data_sha256=OWNER.sha(holdout),
                       train_ids=[601, 602], passed_train_ids=[601, 602], gradient_holdout_ids=[],
                       reference_quality_summary=quality_summary)
        restoration = dict(current_model="F04", best_parent="F04", checkpoint_all9_fullstat_matched_manifest=True,
                           replicas={"gemma-2": dict(cid=self.plan["target"]["id"], tree=identity["sha256"],
                           checkpoint_mount=self.plan["checkpoint"], checkpoint_readonly=True)})
        self.plan["training"] = dict(plan=self.pin(out / "training-plan.json", self.task),
             input_quality=self.pin(out / "quality.json", quality), holdout=dict(path=str(holdout), sha256=OWNER.sha(holdout)),
             restoration_proof=self.pin(out / "restoration.json", restoration))
        self.task["input_quality"] = quality_summary
        self.plan["training"]["plan"] = self.pin(out / "training-plan.json", self.task)
        self.plan["training"]["cpu_preview"] = self.pin(out / "trainer-preview.json", dict(
            returncode=0, model_calls=0, trainer_arguments=OWNER.training_args(self.plan)))

    def test_training_preview_has_separate_inputs_and_rejects_unpassed_reference(self):
        with patch.object(OWNER.subprocess, "Popen", side_effect=AssertionError("no execution")):
            self.assertEqual(OWNER.validate(self.plan), [601, 602])
            command = OWNER.command(self.plan, "train-lora", "a" * 64)
            self.assertIn("train_lora.py", " ".join(command))
            self.assertNotIn("--cases", command)
            self.assertNotIn("--objective", command)
            self.assertEqual(command[command.index("--gpus") + 1], "device=4")
            self.plan["native_cohort"] = dict(count=2, rows=[])
            with self.assertRaisesRegex(ValueError, "masquerade"):
                OWNER.validate(self.plan)
            del self.plan["native_cohort"]
            pin = self.plan["training"]["input_quality"]
            quality = OWNER.read(pin["path"])
            quality["passed_train_ids"].pop()
            self.plan["training"]["input_quality"] = self.pin(Path(pin["path"]), quality)
            with self.assertRaisesRegex(ValueError, "all training references"):
                OWNER.validate(self.plan)

    def test_pilot_only_result_cannot_claim_total64_completed(self):
        result = Path(self.plan["output_root"]) / "result"
        result.mkdir()
        summary = dict(status="pilot_only", completed_steps=4, trainer_plan_sha256=self.plan["training"]["plan"]["sha256"],
                       base_unchanged=True, target_tensors=410, data_sha256=self.task["data"]["sha256"],
                       training_ids=[601, 602], holdout_gradient_ids=[])
        export = self.pin(result / "training-receipt.json", dict(status="exported", base_unchanged=True,
                parent_path=self.plan["checkpoint"], parent_tree_sha256=self.plan["checkpoint_tree_sha256"],
                data_sha256=self.task["data"]["sha256"], modules=410, steps=4))
        summary.update(maximum_seconds_excludes_approval_wait=True, error=None, exports=[export],
                       source_sha256=self.plan["source_pins"])
        (result / "summary.json").write_text(json.dumps(summary))
        self.assertEqual(OWNER.training_complete(self.plan, result)["status"], "pilot_only")
        summary["status"] = "completed"
        (result / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "pilot4 or total64"):
            OWNER.training_complete(self.plan, result)

    def test_failed_training_job_uses_owned_cleanup_before_exact_restore(self):
        self.plan["approved_for_execution"] = True
        out = Path(self.plan["output_root"])
        plan_path = out / "owner-plan.json"
        plan_path.write_text(json.dumps(self.plan))
        digest = OWNER.sha(plan_path)
        state = dict(stopped=False, owned=False, original_started=False)
        calls = []
        owned = dict(Id="own-train", Image=OWNER.IMAGE,
                     Name="/"+self.plan["container_prefix"]+"-train-lora", State=dict(Running=True),
                     Config=dict(Labels={"kairyu.native.owner-plan-sha256": digest,
                                         "kairyu.native.owner-root": self.plan["output_root"]}))
        def docker(*args):
            calls.append(args)
            if args[:2] == ("stop", "--time"):
                if args[-1] == self.plan["target"]["id"]:
                    state["stopped"] = True
                else:
                    owned["State"]["Running"] = False
            if args[0] == "start":
                self.assertFalse(owned["State"]["Running"])
                self.assertEqual(args[1], self.plan["target"]["id"])
                state["original_started"] = True
            if args[0] == "ps":
                return "\n".join([self.plan["target"]["id"], *(v["id"] for v in self.plan["protected"].values())])
            return ""
        def inspect(identifier):
            if identifier == self.plan["target"]["id"]:
                return dict(State=dict(Running=not state["stopped"]))
            if state["owned"]:
                return owned
            raise subprocess.CalledProcessError(1, "inspect")
        def snapshot(name):
            if name == OWNER.TARGET:
                return dict(self.plan["target"], started_at="restored" if state["original_started"] else "old")
            return self.plan["protected"][name]
        guard = SimpleNamespace(docker=docker, inspect=inspect, snapshot=snapshot,
                 target_metrics=lambda cid: dict(queues={"running":[0], "waiting":[0]}),
                 lock_owner_proof=lambda *args: dict(fixture=True))
        class FailedChild:
            def __init__(self, command, **kwargs):
                state["owned"] = True
                (out / "work/train-lora.cid").write_text("own-train")
            def wait(self, timeout):
                return 1
            def poll(self):
                return 1
        with patch.object(OWNER, "helper", return_value=guard), patch.object(OWNER.os, "getuid", return_value=1003), \
             patch.object(OWNER.signal, "signal"), patch.object(OWNER.subprocess, "Popen", FailedChild), \
             patch.object(OWNER.subprocess, "check_output", return_value="97252\n"):
            with self.assertRaisesRegex(ValueError, "preserved incomplete"):
                OWNER.run(self.plan, digest, plan_path)
        final = OWNER.read(out / "completion.json")
        self.assertEqual(final["status"], "failed")
        self.assertTrue(final["restored"] and final["protected_exact"])
        self.assertEqual([j["objective"] for j in final["jobs"]], ["train-lora"])
        self.assertNotIn("collections_completed", final)
        self.assertLess(calls.index(("stop", "--time", "30", "own-train")), calls.index(("start", self.plan["target"]["id"])))


if __name__ == "__main__":
    unittest.main()
