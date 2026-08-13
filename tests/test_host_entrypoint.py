from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HostEntrypointTest(unittest.TestCase):
    def test_list_builds_missing_image_then_runs_command_in_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            log = temp / "docker.log"
            fake_docker = temp / "docker"
            fake_docker.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
                "if [ \"$1 $2\" = \"image inspect\" ]; then exit 1; fi\n",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{temp}:{env['PATH']}"
            env["FAKE_DOCKER_LOG"] = str(log)

            result = subprocess.run(
                [str(ROOT / "kairyu-bench"), "list"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(calls[0], "image inspect kairyu-bench:local")
            self.assertEqual(
                calls[1], f"build -t kairyu-bench:local {ROOT}"
            )
            self.assertIn("run --rm --privileged", calls[2])
            self.assertIn(
                f"-v {ROOT / 'results'}:/work/results", calls[2]
            )
            self.assertTrue(calls[2].endswith("kairyu-bench:local list"))


if __name__ == "__main__":
    unittest.main()
