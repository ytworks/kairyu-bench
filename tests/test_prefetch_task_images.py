"""Synthetic Docker contract tests only; no Docker subprocesses or network."""
from __future__ import annotations

import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prefetch_task_images", Path(__file__).resolve().parents[1] / "scripts" / "prefetch-task-images.py")
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)
DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64
REF = "registry.invalid/team/image:fixed"
EXPECTED = "registry.invalid/team/image@" + DIGEST


class FakeDocker:
    def __init__(self, *, daemon="expected", cached=False, bad_digest=False,
                 failures=None, mutate=None):
        self.daemon, self.present, self.bad_digest = daemon, cached, bad_digest
        self.failures, self.mutate = list(failures or []), mutate
        self.commands, self.pull_times = [], []
        self.closed = False

    def __call__(self, stop, output):
        self.stop = stop
        return self

    def call(self, args, timeout, launch_guard=None):
        with launch_guard() if launch_guard else nullcontext():
            return self._call(args, timeout)

    def _call(self, args, timeout):
        assert isinstance(args, list) and args[:2] == ["docker", "--host"]
        assert args[2].startswith("unix://")
        self.commands.append(args)
        command = args[3:]
        if command[0] == "info":
            return tool.CommandResult(0, json.dumps(self.daemon))
        if command[:2] == ["image", "inspect"]:
            if not self.present:
                return tool.CommandResult(1, stderr="Error: No such image")
            digest = "registry.invalid/team/image@" + OTHER if self.bad_digest else EXPECTED
            return tool.CommandResult(0, json.dumps({"Id": OTHER, "RepoDigests": [digest]}))
        if command[:2] == ["image", "pull"]:
            self.pull_times.append(time.monotonic())
            if self.mutate:
                self.mutate()
            if self.failures:
                return self.failures.pop(0)
            self.present = True
            return tool.CommandResult(0)
        raise AssertionError(args)

    def close(self):
        self.closed = True


class PrefetchContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(self.root / "docker.sock"))
        rows = {"records": [{"task": "task-a", "ref": REF}]}
        self.tasks = self.root / "tasks.json"
        self.tasks.write_text(json.dumps(rows))
        self.expected = self.root / "expected.json"
        self.expected.write_text(json.dumps({"records": [{
            "task": "task-a", "ref": REF, "registry_digest": DIGEST,
            "digest_reference": EXPECTED,
        }]}))
        self.counter = 0

    def tearDown(self):
        self.sock.close()
        self.temp.cleanup()

    def run_tool(self, fake, *, apply=True, extras=()):
        self.counter += 1
        output = self.root / f"out-{self.counter}"
        argv = [
            "--docker-socket", str(self.root / "docker.sock"),
            "--expected-daemon-id", "expected", "--tasks", str(self.tasks),
            "--expected-digests", str(self.expected), "--output", str(output),
            "--min-free-path", str(self.root), "--min-free-bytes", "0",
            "--issue-interval-seconds", "0.01", "--backoff-seconds", "0",
            "--max-backoff-seconds", "0", "--max-retries", "1",
        ]
        if apply:
            argv.append("--apply")
        code = tool.main(argv + list(extras), runner_factory=fake)
        return code, json.loads((output / "receipt.json").read_text())

    def test_preview_does_not_issue_any_docker_commands(self):
        fake = FakeDocker()
        code, receipt = self.run_tool(fake, apply=False)
        self.assertEqual(code, 0)
        self.assertEqual(fake.commands, [])
        self.assertEqual(receipt["status"], "preview")
        self.assertFalse(receipt["daemon_id_verified"])
        self.assertTrue(receipt["input_continuity_verified"])

    def test_fresh_output_rejects_broken_symlink_without_docker_calls(self):
        output = self.root / "out-1"
        target = self.root / "not-created"
        output.symlink_to(target, target_is_directory=True)
        fake = FakeDocker()
        argv = [
            "--docker-socket", str(self.root / "docker.sock"),
            "--expected-daemon-id", "expected", "--tasks", str(self.tasks),
            "--expected-digests", str(self.expected), "--output", str(output),
            "--min-free-path", str(self.root), "--min-free-bytes", "0",
        ]
        self.assertEqual(tool.main(argv, runner_factory=fake), 2)
        self.assertFalse(target.exists())
        self.assertEqual(fake.commands, [])

    def test_wrong_daemon_is_rejected_before_image_operations(self):
        fake = FakeDocker(daemon="another-daemon")
        code, receipt = self.run_tool(fake)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual([args[3] for args in fake.commands], ["info"])
        self.assertTrue(fake.closed)

    def test_rate_limit_retries_then_requires_expected_repo_digest(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                fake = FakeDocker(bad_digest=mismatch, failures=[
                    tool.CommandResult(1, stderr="denied: Rate exceeded"),
                ])
                code, receipt = self.run_tool(fake)
                image = receipt["image_results"][REF]
                self.assertEqual(len(fake.pull_times), 2)
                self.assertGreaterEqual(fake.pull_times[1] - fake.pull_times[0], 0.009)
                self.assertEqual(image["attempts"][0]["retry_class"], "rate_limit")
                self.assertEqual(code, 1 if mismatch else 0)
                self.assertEqual(image["status"], "failed" if mismatch else "pulled_verified")
                if not mismatch:
                    self.assertEqual(image["expected_digest_verified"], EXPECTED)

    def test_cached_image_also_requires_expected_digest_and_is_not_retagged(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                fake = FakeDocker(cached=True, bad_digest=mismatch)
                code, receipt = self.run_tool(fake)
                self.assertEqual(code, 1 if mismatch else 0)
                self.assertEqual(fake.pull_times, [])
                self.assertEqual(receipt["image_results"][REF]["status"],
                                 "failed" if mismatch else "cached_verified")

    def test_timeout_retry_is_bounded_and_auth_error_is_not_retried(self):
        timeout = tool.CommandResult(-15, timed_out=True)
        for failures, retries, attempts, expected in (
            ([timeout], 1, 2, 0),
            ([timeout], 0, 1, 1),
            ([tool.CommandResult(1, stderr="unauthorized")], 1, 1, 1),
        ):
            with self.subTest(retries=retries, expected=expected):
                fake = FakeDocker(failures=failures)
                code, _ = self.run_tool(fake, extras=("--max-retries", str(retries)))
                self.assertEqual(code, expected)
                self.assertEqual(len(fake.pull_times), attempts)

    def test_daemon_changed_while_pacing_is_rejected_before_pull(self):
        fake = FakeDocker()
        original = tool.Prefetch.pull_launch

        def switched_launch(owner):
            fake.daemon = "changed-during-wait"
            return original(owner)

        with patch.object(tool.Prefetch, "pull_launch", switched_launch):
            code, receipt = self.run_tool(fake)
        self.assertEqual(code, 1)
        self.assertEqual(fake.pull_times, [])
        self.assertEqual(receipt["image_results"][REF]["status"], "failed")

    def test_receipt_write_failure_stops_queue_before_executor_wait(self):
        second = REF.replace(":fixed", ":second")
        self.tasks.write_text(json.dumps({"records": [
            {"task": "task-a", "ref": REF}, {"task": "task-b", "ref": second},
        ]}))
        self.expected.write_text(json.dumps({"records": [
            {"task": task, "ref": ref, "digest_reference": EXPECTED}
            for task, ref in (("task-a", REF), ("task-b", second))
        ]}))

        class DelayedFake(FakeDocker):
            def _call(self, args, timeout):
                if args[3:6] == ["image", "inspect", second]:
                    # Give the main thread time to observe its save error.
                    if self.stop.wait(0.1):
                        raise tool.Stopped("Stopped before next pull")
                    self.present = False
                return super()._call(args, timeout)

        fake = DelayedFake()
        saved, count = tool.Prefetch.save, 0

        def failing_save(owner):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("synthetic receipt write failure")
            saved(owner)

        with patch.object(tool.Prefetch, "save", failing_save):
            code, receipt = self.run_tool(fake)
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.pull_times), 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertTrue(fake.closed)

    def test_input_change_invalidates_receipt_and_stops_more_pulls(self):
        fake = FakeDocker(mutate=lambda: self.tasks.write_text(self.tasks.read_text() + "\n"))
        code, receipt = self.run_tool(fake)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["input_continuity_verified"])
        self.assertEqual(len(fake.pull_times), 1)


if __name__ == "__main__":
    unittest.main()
