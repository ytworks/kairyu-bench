"""Protect the operational boundary of the opt-in tracing gateway recipe."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "trace_gateway", Path(__file__).resolve().parents[1] / "scripts/observe-trace-gateway.py")
assert spec and spec.loader
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


class TraceGatewayTests(unittest.TestCase):
    def test_internal_clone_preserves_service_but_not_host_authority(self):
        source = {
            "Image": "sha256:pinned",
            "Config": {"Env": ["SERVICE_KEY=private"], "Cmd": ["serve"], "User": "0"},
            "Mounts": [{"Destination": "/etc/kairyu/kairyu.yaml"},
                       {"Destination": "/var/lib/kairyu/placement"}],
            "HostConfig": {"Privileged": True, "PortBindings": {"8000/tcp": []},
                           "Binds": ["/var/run/docker.sock:/var/run/docker.sock"],
                           "Memory": 12345},
            "NetworkSettings": {"Networks": {"model-network": {}}},
        }
        actual = gateway.configuration(source, "trace-test", Path("/nvme/trace"), 1003, 1003)
        self.assertEqual(actual["Image"], "sha256:pinned")
        self.assertEqual(actual["Cmd"], ["serve"])
        self.assertEqual(actual["User"], "1003:1003")
        self.assertIn("SERVICE_KEY=private", actual["Env"])
        self.assertIn("OTEL_TRACES_EXPORTER=console", actual["Env"])
        self.assertFalse(actual["HostConfig"]["PortBindings"])
        self.assertNotIn("Privileged", actual["HostConfig"])
        self.assertFalse(any("docker.sock" in x for x in actual["HostConfig"]["Binds"]))
        self.assertEqual(actual["HostConfig"]["Memory"], 12345)
        self.assertEqual(source["Config"]["User"], "0")
        self.assertEqual(source["Config"]["Env"], ["SERVICE_KEY=private"])
