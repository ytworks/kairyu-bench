"""Single-replica recipe metadata contracts; no Docker or model calls."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
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
