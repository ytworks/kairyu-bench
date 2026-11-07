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
        guard, calls = self.fake_guard(drift=True)
        status = dict(restored=False, protected_exact=False)
        OWNER.finish(self.plan, guard, None, True, status, "a" * 64)
        self.assertIn(("start", OWNER.CID), calls)
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


if __name__ == "__main__":
    unittest.main()
