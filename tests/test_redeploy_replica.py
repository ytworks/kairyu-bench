"""Single-replica recipe metadata contracts; no Docker or model calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "redeploy_replica", Path(__file__).resolve().parents[1] / "docs/experiments/scripts/redeploy-replica.py"
)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class ReplicaRecipeTest(unittest.TestCase):
    def test_render_preserves_runtime_and_normalizes_image_inheritance(self):
        current = {
            "Id": "before", "Name": "/example-gemma-0-1", "Image": "sha256:fixed",
            "Config": {"Entrypoint": ["serve"], "Cmd": ["--model", "/checkpoint"],
                       "Env": ["PATH=/bin", "OBSERVE=on"], "User": "1003", "WorkingDir": "/app",
                       "Healthcheck": {"Test": ["CMD", "true"]},
                       "Labels": {"kairyu.chat-checkpoint-tree": "old"}},
            "HostConfig": {"CpusetCpus": "0-3", "IpcMode": "host",
                           "DeviceRequests": [{"Driver": "nvidia", "Count": 0, "DeviceIDs": ["0", "1"],
                                               "Capabilities": [["gpu"]], "Options": None}]},
            "Mounts": [{"Type": "bind", "Source": "/old", "Destination": "/checkpoint", "RW": False}],
            "NetworkSettings": {"Networks": {"example": {}}},
            "State": {"Running": True, "Health": {"Status": "healthy"}},
        }
        rendered = {
            "image": "sha256:fixed", "command": ["--model", "/checkpoint"], "entrypoint": None,
            "environment": {"OBSERVE": "on"}, "cpuset": "0-3", "ipc": "host",
            "deploy": {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "device_ids": ["0", "1"], "capabilities": ["gpu"]}
            ]}}},
            "volumes": [{"type": "bind", "source": "/new", "target": "/checkpoint", "read_only": True}],
            "labels": {"kairyu.chat-checkpoint-tree": "new"},
        }
        image = [{"Config": {"Entrypoint": ["serve"], "Env": ["PATH=/bin"], "User": "1003", "WorkingDir": "/app"}}]
        control = SimpleNamespace(_compose_command=lambda args: ["compose", *args])

        def invoke(value):
            with patch.object(tool, "command", side_effect=[json.dumps({"services": {"gemma-0": value}}), json.dumps(image)]):
                return tool.check_render(control, {}, "gemma-0", current, Path("/new"), "new")

        self.assertEqual(len(invoke(rendered)), 64)
        for key, value in (
            ("entrypoint", ["another"]), ("command", ["--different"]), ("environment", {}),
            ("cpuset", "4-7"), ("user", "0"), ("working_dir", "/another"),
        ):
            changed = copy.deepcopy(rendered)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                invoke(changed)
        changed = copy.deepcopy(rendered)
        changed["volumes"][0]["read_only"] = False
        with self.assertRaises(ValueError):
            invoke(changed)

    def test_cli_refuses_retired_compose_duplicate_before_any_recreation(self):
        # A renamed, stopped observer still has Compose labels. It must not be
        # selected or removed by up, including if it appears after the preview.
        old_id, retired_id = "a" * 64, "b" * 64
        for apply, memberships, succeeds in (
            (False, [[old_id, retired_id]], False),
            (True, [[old_id, retired_id]], False),
            (True, [[old_id], [old_id, retired_id]], False),
            (True, [[old_id], [old_id]], True),
            (True, [[old_id], [old_id], [old_id, retired_id]], "rollback_conflict"),
            (True, [[old_id], [old_id], []], "rollback_empty"),
        ):
            with self.subTest(apply=apply, memberships=memberships), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                candidate = root / "candidate"
                candidate.mkdir()
                control = root / "control.py"
                control.write_text(
                    "PROJECT = 'example'\nCHAT_SERVICES = ('gemma-0',)\n"
                    "def _compose_env(): return {'COMPOSE_REMOVE_ORPHANS': '1'}\n"
                )
                freeze = root / "freeze.json"
                freeze.write_text("{}")
                argv = ["redeploy-replica", "--control", str(control), "--service", "gemma-0",
                        "--candidate", str(candidate), "--freeze", str(freeze),
                        "--baseline", str(root / "baseline.json"), "--output", str(root / "out")]
                if apply:
                    argv.append("--apply")
                queries = iter(memberships)
                commands = []

                def command(args, **kwargs):
                    commands.append(args)
                    if args[:2] == ["docker", "info"]:
                        return "test-daemon\n"
                    if args[:2] == ["docker", "ps"]:
                        self.assertIn("--all", args)
                        self.assertIn("--no-trunc", args)
                        self.assertIn("label=com.docker.compose.project=example", args)
                        self.assertIn("label=com.docker.compose.service=gemma-0", args)
                        ids = next(queries)
                        return "\n".join(ids) + ("\n" if ids else "")
                    raise AssertionError(f"unexpected external command: {args}")

                old = {"id": old_id, "image_id": "sha256:fixed", "cpuset": "0-3",
                       "checkpoint_path": "/old", "tree": "old-tree", "running": True}
                container = {"NetworkSettings": {"Networks": {"example": {"IPAddress": "127.0.0.1"}}}}
                identity = {"tree_sha256": "c" * 64, "stat": {}}
                with (patch.object(sys, "argv", argv), patch.object(tool.os, "getuid", return_value=1003),
                      patch.object(tool, "command", side_effect=command),
                      patch.object(tool, "inspect", return_value=container),
                      patch.object(tool, "snapshot", return_value=old),
                      patch.object(tool, "identity", return_value=identity),
                      patch.object(tool, "check_render", return_value="d" * 64),
                      patch.object(tool, "idle"), patch.object(tool, "compose_up") as mutate,
                      patch.object(tool, "compare_after", return_value={"id": "e" * 64})):
                    if succeeds == "rollback_empty":
                        mutate.side_effect = [RuntimeError("simulated recreation failure"), None]
                        with self.assertRaisesRegex(RuntimeError, "simulated recreation failure"):
                            tool.main()
                        self.assertEqual(mutate.call_count, 2)
                        rollback_env = mutate.call_args_list[1].args[1]
                        self.assertEqual(rollback_env["CHAT_CHECKPOINT_PATH"], "/old")
                        self.assertEqual(rollback_env["CHAT_CHECKPOINT_TREE"], "old-tree")
                        self.assertEqual(rollback_env["COMPOSE_REMOVE_ORPHANS"], "0")
                        receipt = json.loads((root / "out/receipt.json").read_text())
                        self.assertTrue(receipt["rollback_completed"])
                    elif succeeds == "rollback_conflict":
                        mutate.side_effect = RuntimeError("simulated recreation failure")
                        with self.assertRaisesRegex(RuntimeError, "simulated recreation failure"):
                            tool.main()
                        self.assertEqual(mutate.call_count, 1)
                        receipt = json.loads((root / "out/receipt.json").read_text())
                        self.assertFalse(receipt["rollback_completed"])
                        self.assertEqual(receipt["rollback_error_type"], "ValueError")
                    elif succeeds:
                        tool.main()
                        self.assertEqual(mutate.call_count, 1)
                        self.assertEqual(mutate.call_args.args[1]["COMPOSE_REMOVE_ORPHANS"], "0")
                        self.assertEqual(json.loads((root / "out/receipt.json").read_text())["status"], "completed")
                    else:
                        with self.assertRaisesRegex(ValueError, "Compose service selection"):
                            tool.main()
                        mutate.assert_not_called()
                self.assertTrue(commands)

    def test_frozen_identity_rejects_changed_frozen_stat_with_same_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "weights"
            candidate.mkdir()
            weight = candidate / "one.safetensors"
            weight.write_bytes(b"first")
            rows = [{"path": weight.name, "size": 5, "sha256": tool.sha(weight)}]
            proof = root / "freeze.json"
            proof.write_text(json.dumps({
                "checkpoint_path": str(candidate), "candidate_inventory_stat": tool.inventory(candidate),
                "all_checkpoint_files": rows, "control_tree_sha256": tool.digest(rows),
            }))
            args = SimpleNamespace(candidate=candidate, freeze=proof)
            self.assertEqual(tool.identity(args)["tree_sha256"], tool.digest(rows))
            frozen_stat = weight.stat()
            weight.write_bytes(b"other")
            # Two writes can share a filesystem timestamp tick. This is a
            # saved-stat continuity check, not a fresh content-hash check.
            os.utime(weight, ns=(frozen_stat.st_atime_ns, frozen_stat.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, "frozen inventory"):
                tool.identity(args)


if __name__ == "__main__":
    unittest.main()
