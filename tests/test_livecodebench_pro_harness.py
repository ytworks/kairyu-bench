from __future__ import annotations

import unittest
from pathlib import Path

from kairyu_bench.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


class LiveCodeBenchProHarnessTest(unittest.TestCase):
    def test_verifier_image_pins_node_compatible_npm(self) -> None:
        entry = load_manifest()["livecodebench-pro"]
        verifier_revision = next(
            source["revision"]
            for source in entry["secondary_sources"]
            if source["id"].endswith("/LightCPVerifier")
        )
        dockerfile = (
            ROOT / "scripts/harnesses/livecodebench-pro.Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn(verifier_revision, dockerfile)
        self.assertIn("https://deb.nodesource.com/setup_20.x", dockerfile)
        self.assertIn("RUN npm install -g npm@10.8.2", dockerfile)
        self.assertNotIn("RUN npm install -g npm@latest", dockerfile)

    def test_harness_uses_the_revision_scoped_verifier_image(self) -> None:
        harness = (ROOT / "scripts/harnesses/livecodebench-pro.sh").read_text(
            encoding="utf-8"
        )
        shim = (ROOT / "scripts/shims/livecodebench_pro.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'verifier_image="lightcpverifier:${verifier_revision}-kairyu"',
            harness,
        )
        self.assertIn('export KAIRYU_LIGHTCPVERIFIER_IMAGE="$verifier_image"', harness)
        self.assertIn(
            'KAIRYU_LIGHTCPVERIFIER_HOST:-host.docker.internal', harness
        )
        self.assertIn('os.environ.get("KAIRYU_LIGHTCPVERIFIER_IMAGE")', shim)
        self.assertIn("LightCPVerifierJudge.IMAGE_NAME = verifier_image", shim)
        self.assertIn("class RoutedLightCPVerifierJudge", shim)
        self.assertIn('self.base_url = f"http://{verifier_host}:{port}"', shim)


if __name__ == "__main__":
    unittest.main()
