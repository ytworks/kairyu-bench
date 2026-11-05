from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("observe_docker", Path(__file__).parents[1] / "scripts/observe-docker.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FakeDocker:
    def __init__(self, original, failure=None):
        self.containers = {original["Id"]: copy.deepcopy(original)}
        self.calls = []
        self.failure = failure

    def inspect(self, name):
        for row in self.containers.values():
            if row["Id"] == name or row["Name"].lstrip("/") == name:
                return copy.deepcopy(row)
        raise RuntimeError("missing container")

    def logs(self, cid, limit=65536):
        return b"synthetic-private-diagnostic" * 4000

    def call(self, method, path, body=None):
        self.calls.append((method, path))
        if path.startswith("/containers/create"):
            if self.failure == "create":
                raise RuntimeError("synthetic create failure")
            name = path.split("name=", 1)[1]
            self.containers["observer"] = {
                "Id": "observer", "Name": "/" + name, "Image": body["Image"],
                "Config": copy.deepcopy(body), "HostConfig": body["HostConfig"],
                "State": {"Running": False, "Status": "created", "Health": {"Status": "healthy"}},
            }
            return {"Id": "observer"}
        cid = path.split("/")[2].split("?")[0]
        row = self.containers[cid]
        if method == "DELETE":
            del self.containers[cid]
        elif "/stop" in path:
            row["State"].update(Running=False, Status="exited")
        elif "/rename" in path:
            name = path.split("name=", 1)[1]
            if any(other["Name"] == "/" + name and other["Id"] != cid for other in self.containers.values()):
                raise RuntimeError("name collision")
            row["Name"] = "/" + name
        elif "/start" in path:
            if cid == "original" and self.failure == "restore":
                raise RuntimeError("synthetic original restart failure")
            if cid == "observer" and self.failure in ("start", "signal"):
                if self.failure == "signal":
                    raise InterruptedError("synthetic stop signal")
                raise RuntimeError("synthetic start failure")
            row["State"].update(Running=True, Status="running")
            if cid == "observer" and self.failure == "health":
                row["State"]["Health"]["Status"] = "unhealthy"
        else:
            raise AssertionError((method, path))
        return {}


class DockerObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "source"
        (source / "kairyu_bench/wsp/observe").mkdir(parents=True)
        (source / "kairyu_bench/wsp/observe/vllm_worker.py").write_text("# synthetic")
        collector = Path(__file__).parents[1] / "src/kairyu_bench/wsp/observe/collector.py"
        (source / "kairyu_bench/wsp/observe/collector.py").write_text(collector.read_text())
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        output = self.root / "capture"
        output.mkdir()
        config = output / "capture.json"
        config.write_text(json.dumps({"schema_version": 1, "observation_mode": "live",
                                     "capture_id": "synthetic", "output_dir": str(output),
                                     "checkpoint": {"path": str(checkpoint)},
                                     "runtime": {"tp_size": 2, "replicas": 4},
                                     "limits": {"storage_bytes": 2 * 1024**3}}))
        self.args = SimpleNamespace(container="replica", source=source, capture=config,
                                    runtime_cache=self.root / "cache", backup_dir=self.root / "backup",
                                    replica="r0", apply=True, health_timeout=1, leave_stopped=False)
        self.original = {
            "Id": "original", "Name": "/replica", "Image": "sha256:pinned",
            "Config": {"User": "1003:1003", "Cmd": ["serve", "/checkpoint"], "Entrypoint": ["vllm"],
                       "Env": ["SECRET=synthetic-private", "PYTHONPATH=/prior"],
                       "Labels": {"compose": "keep"}, "Healthcheck": {"Test": ["CMD", "true"]}},
            "HostConfig": {"Binds": [str(checkpoint) + ":/checkpoint:ro"], "AutoRemove": False,
                           "DeviceRequests": [{"Count": 2}], "Privileged": False, "NetworkMode": "same",
                           "Memory": 123, "RestartPolicy": {"Name": "unless-stopped"}},
            "NetworkSettings": {"Networks": {"same": {"Aliases": ["gemma-0"], "IPAMConfig": None}}},
            "Mounts": [{"Type": "bind", "Source": str(checkpoint), "Destination": "/checkpoint", "RW": False}],
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}},
        }

    def plan(self):
        return module.attach_plan(self.original, self.args)

    def apply(self, docker):
        config, summary, operation = self.plan()
        with patch.object(module.signal, "signal"):
            return module.apply_attach(docker, self.original, config, summary, operation, 1)

    def test_plan_preserves_execution_and_redacts_summary(self):
        before = copy.deepcopy(self.original)
        config, summary, operation = self.plan()
        self.assertEqual(self.original, before)
        self.assertEqual(config["Image"], before["Image"])
        self.assertEqual(config["User"], "1003:1003")
        self.assertEqual(config["Cmd"][:2], before["Config"]["Cmd"])
        self.assertEqual(config["HostConfig"]["DeviceRequests"], before["HostConfig"]["DeviceRequests"])
        self.assertEqual(config["HostConfig"]["Memory"], 123)
        self.assertFalse(config["HostConfig"]["Privileged"])
        self.assertEqual(config["NetworkingConfig"]["EndpointsConfig"]["same"]["Aliases"], ["gemma-0"])
        self.assertIn("SECRET=synthetic-private", config["Env"])
        self.assertNotIn("synthetic-private", json.dumps(summary))
        self.assertIn("--enable-request-id-headers", config["Cmd"])
        self.assertEqual(config["Labels"][module.LABEL], operation)

    def test_dry_run_never_mutates_docker_or_files(self):
        docker = FakeDocker(self.original)
        argv = ["attach", "--container", "replica", "--capture", str(self.args.capture),
                "--source", str(self.args.source), "--runtime-cache", str(self.args.runtime_cache),
                "--backup-dir", str(self.args.backup_dir)]
        output = io.StringIO()
        with patch.object(module, "Docker", return_value=docker), contextlib.redirect_stdout(output):
            self.assertEqual(module.main(argv), 0)
        self.assertEqual(docker.calls, [])
        self.assertFalse(self.args.backup_dir.exists())
        self.assertFalse(self.args.runtime_cache.exists())
        self.assertEqual(json.loads(output.getvalue())["status"], "dry_run")
        self.assertNotIn("synthetic-private", output.getvalue())

    def test_actual_source_parser_rejects_quota_before_any_write(self):
        config = json.loads(self.args.capture.read_text())
        config["limits"]["storage_bytes"] = 8 * 1024**3
        self.args.capture.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "storage_bytes"):
            self.plan()
        self.assertFalse(self.args.backup_dir.exists())
        self.assertFalse(self.args.runtime_cache.exists())
        self.assertEqual(list(self.args.source.rglob("*.pyc")), [])
        self.assertEqual(list(self.args.source.rglob("__pycache__")), [])

    def test_duplicate_worker_and_writable_checkpoint_are_rejected(self):
        for override in ("--worker-cls=other.Worker", module.WORKER):
            changed = copy.deepcopy(self.original)
            changed["Config"]["Cmd"].append(override)
            with self.assertRaisesRegex(ValueError, "already"):
                module.attach_plan(changed, self.args)
        changed = copy.deepcopy(self.original)
        changed["Mounts"][0]["RW"] = True
        with self.assertRaisesRegex(ValueError, "read-only"):
            module.attach_plan(changed, self.args)
        self.args.runtime_cache = self.args.source
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.plan()

    def test_attach_restore_keeps_original_container_and_private_backup(self):
        docker = FakeDocker(self.original)
        result = self.apply(docker)
        self.assertEqual(result["status"], "attached")
        self.assertFalse(docker.inspect("original")["State"]["Running"])
        directory = self.args.backup_dir / "replica"
        self.assertEqual((directory / "original.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((directory / "original.json").read_text()), self.original)
        with patch.object(module.signal, "signal"):
            restored = module.restore(docker, self.args)
        self.assertEqual(restored["status"], "restored")
        self.assertEqual(docker.inspect("replica")["Id"], "original")
        self.assertTrue(docker.inspect("original")["State"]["Running"])
        self.assertFalse(docker.inspect("observer")["State"]["Running"])
        self.assertEqual(docker.inspect("original")["Config"], self.original["Config"])
        self.assertFalse(any(method == "DELETE" for method, _ in docker.calls))

    def test_restore_can_release_gpu_without_reloading_original_model(self):
        docker = FakeDocker(self.original)
        self.apply(docker)
        self.args.leave_stopped = True
        before = len(docker.calls)
        result = module.restore(docker, self.args)
        self.assertEqual(result["status"], "restored_stopped")
        self.assertEqual(docker.inspect("replica")["Id"], "original")
        self.assertFalse(docker.inspect("original")["State"]["Running"])
        self.assertFalse(docker.inspect("observer")["State"]["Running"])
        self.assertFalse(any("/start" in path for _, path in docker.calls[before:]))
        self.assertEqual(module.restore(docker, self.args)["status"], "already_original")

    def test_already_original_converges_running_state_without_touching_other_containers(self):
        for initial_status, was_running in (("restored", True), ("rolled_back", True), ("restored", False)):
            with self.subTest(initial_status=initial_status, was_running=was_running):
                suffix = f"{initial_status}-{was_running}"
                self.args.backup_dir = self.root / ("backup-" + suffix)
                self.args.runtime_cache = self.root / ("cache-" + suffix)
                self.args.apply, self.args.leave_stopped = True, False
                self.original["State"].update(Running=was_running, Status="running" if was_running else "exited")
                docker = FakeDocker(self.original, "create" if initial_status == "rolled_back" else None)
                if initial_status == "rolled_back":
                    with self.assertRaisesRegex(RuntimeError, "create"):
                        self.apply(docker)
                    docker.failure = None
                else:
                    self.apply(docker)
                    module.restore(docker, self.args)
                state_path = self.args.backup_dir / "replica/state.json"
                others = {key: copy.deepcopy(value) for key, value in docker.containers.items() if key != "original"}
                for leave_stopped in (True, False):
                    self.args.leave_stopped = leave_stopped
                    desired = was_running and not leave_stopped
                    before_calls, before_state = list(docker.calls), state_path.read_bytes()
                    before_running = docker.inspect("original")["State"]["Running"]
                    self.args.apply = False
                    dry = module.restore(docker, self.args)
                    self.assertEqual(dry["status"], "already_original" if before_running == desired else "dry_run")
                    self.assertEqual(docker.calls, before_calls)
                    self.assertEqual(state_path.read_bytes(), before_state)
                    self.args.apply = True
                    module.restore(docker, self.args)
                    self.assertEqual(docker.inspect("replica")["Id"], "original")
                    self.assertEqual(docker.inspect("original")["State"]["Running"], desired)
                    after_calls = list(docker.calls)
                    self.assertEqual(module.restore(docker, self.args)["status"], "already_original")
                    self.assertEqual(docker.calls, after_calls)
                    self.assertTrue(all(path.startswith("/containers/original/") and
                                        ("/start" in path or "/stop" in path)
                                        for _, path in docker.calls[len(before_calls):]))
                    self.assertEqual({key: value for key, value in docker.containers.items() if key != "original"}, others)

    def test_failure_and_signal_restore_only_owned_container(self):
        for failure in ("create", "start", "health", "signal"):
            with self.subTest(failure=failure):
                self.args.backup_dir = self.root / ("backup-" + failure)
                docker = FakeDocker(self.original, failure)
                with self.assertRaises((RuntimeError, InterruptedError)):
                    self.apply(docker)
                self.assertEqual(docker.inspect("replica")["Id"], "original")
                self.assertTrue(docker.inspect("original")["State"]["Running"])
                state = json.loads((self.args.backup_dir / "replica/state.json").read_text())
                self.assertEqual(state["status"], "rolled_back")
                self.assertNotIn("observer", docker.containers)
                if failure != "create":
                    log = self.args.backup_dir / "replica/failed-container.log"
                    self.assertEqual(log.stat().st_size, 65536)
                    self.assertEqual(log.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(state["failed_log_bytes"], 65536)
                self.assertFalse(any(method == "DELETE" and "original" in path for method, path in docker.calls))

    def test_failed_restore_returns_to_owned_observer(self):
        docker = FakeDocker(self.original)
        self.apply(docker)
        docker.failure = "restore"
        with patch.object(module.signal, "signal"), self.assertRaisesRegex(RuntimeError, "restart"):
            module.restore(docker, self.args)
        self.assertEqual(docker.inspect("replica")["Id"], "observer")
        self.assertTrue(docker.inspect("observer")["State"]["Running"])
        self.assertFalse(docker.inspect("original")["State"]["Running"])
        state = json.loads((self.args.backup_dir / "replica/state.json").read_text())
        self.assertEqual(state["status"], "restore_rolled_back")

    def test_restore_rejects_replaced_container_and_tampered_backup(self):
        docker = FakeDocker(self.original)
        self.apply(docker)
        directory = self.args.backup_dir / "replica"
        (directory / "original.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash"):
            module.restore(docker, self.args)
        module.private_json(directory / "original.json", self.original)
        docker.containers["observer"]["Config"]["Labels"][module.LABEL] = "unrelated"
        with self.assertRaisesRegex(ValueError, "not the observer"):
            module.restore(docker, self.args)


if __name__ == "__main__":
    unittest.main()
