"""CPU guards for the fixed train12 owner; no Docker or API access."""
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

SOURCE = Path(__file__).parents[1] / "docs/experiments/scripts/fresh-av-behavior12-native-20261107.py"
spec = importlib.util.spec_from_file_location("fixed_train12_owner", SOURCE)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class FixedBehaviorOwnerTests(unittest.TestCase):
    def test_transfer_duplicate_and_non_singleton_are_rejected(self):
        def data():
            return {a: [[{"id": f"behavior-{a}{i}", "split": "train",
                         "prompt": "p", "correct": "c", "foil": "f"}] for i in (1, 2)] for a in "ABCDEF"}
        self.assertEqual(len(owner.validate_train(data())), 12)
        expanded = data()
        expanded["J"] = [[{"id": "behavior-J1", "split": "train", "prompt": "new", "correct": "c", "foil": "f"}]]
        explicit_ids = owner.IDS + ["behavior-J1"]
        self.assertEqual(len(owner.validate_train(expanded, explicit_ids)), 13)
        with self.assertRaises(ValueError):
            owner.validate_train(expanded, explicit_ids[:-1])
        expanded["J"][0][0]["split"] = "transfer"
        with self.assertRaises(ValueError):
            owner.validate_train(expanded, explicit_ids)
        value = data()
        value["F"][1][0]["id"] = "behavior-F3"
        value["F"][1][0]["split"] = "transfer"
        with self.assertRaises(ValueError):
            owner.validate_train(value)
        value = data()
        value["F"][1] = value["F"][0]
        with self.assertRaises(ValueError):
            owner.validate_train(value)
        value = data()
        value["A"] = [value["A"][0] + value["A"][1]]
        with self.assertRaises(ValueError):
            owner.validate_train(value)

    def test_exact_plan_approval_is_required(self):
        plan = {"approved_for_execution": False, "authorization": "not granted"}
        with self.assertRaises(ValueError):
            owner.require_authorization(plan, "a" * 64, "a" * 64)
        plan["approved_for_execution"] = True
        with self.assertRaises(ValueError):
            owner.require_authorization(plan, "a" * 64, "b" * 64)
        owner.require_authorization(plan, "a" * 64, "a" * 64)

    def test_stop_holds_lock_until_child_and_serving_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock_path, marker = root / "lock", root / "child-done"
            plan = {"approved_for_execution": True, "authorization": "CPU mock only",
                    "owner_output": str(root / "owner"), "global_lock": str(lock_path), "cwd": temp}
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            observations = []
            def preflight(_):
                if (root / "owner/child.json").exists():
                    self.assertTrue(marker.exists())
                    with lock_path.open("a+") as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    observations.append("child_done_lock_held_at_closure")
                return {"mock_cpu": True}
            def stop_after_child():
                for _ in range(100):
                    if (root / "owner/child.json").exists():
                        with lock_path.open("a+") as handle:
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                observations.append("ERROR_lock_released")
                            except BlockingIOError:
                                observations.append("child_running_lock_held")
                        os.kill(os.getpid(), signal.SIGTERM)
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    time.sleep(.01)
            thread = threading.Thread(target=stop_after_child)
            code = "import pathlib,time;time.sleep(.3);pathlib.Path(" + repr(str(marker)) + ").write_text('done')"
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(owner, "check_plan"))
                stack.enter_context(mock.patch.object(owner, "check_pins"))
                stack.enter_context(mock.patch.object(owner, "preflight", side_effect=preflight))
                stack.enter_context(mock.patch.object(owner, "summary", return_value={"missing_ids": [], "received": 12}))
                stack.enter_context(mock.patch.object(owner, "command", return_value=[sys.executable, "-c", code]))
                stack.enter_context(mock.patch.object(sys, "argv", ["owner", "--plan", str(path), "--run",
                                                                   "--approved-plan-sha256", digest]))
                thread.start()
                owner.main()
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(observations, ["child_running_lock_held", "child_done_lock_held_at_closure"])
            closure = json.loads((root / "owner/owner-closure.json").read_text())
            self.assertTrue(closure["stop_requests"])
            self.assertTrue(closure["child_finished"])
            with lock_path.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue((root / "owner/normal-lock-release.json").exists())


if __name__ == "__main__":
    unittest.main()
