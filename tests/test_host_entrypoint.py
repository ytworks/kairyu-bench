from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HostEntrypointTest(unittest.TestCase):
    def test_image_copies_a_versioned_static_docker_client(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("FROM docker:27.5.1-cli@sha256:", dockerfile)
        self.assertIn("FROM python:3.12-slim@sha256:", dockerfile)
        self.assertIn(
            "COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker",
            dockerfile,
        )
        self.assertIn(
            "COPY --from=docker_cli "
            "/usr/local/libexec/docker/cli-plugins/docker-compose "
            "/usr/local/libexec/docker/cli-plugins/docker-compose",
            dockerfile,
        )
        self.assertIn("# hadolint ignore=DL3008", dockerfile)
        self.assertNotIn("    docker.io \\\n", dockerfile)

    def test_list_builds_current_source_then_runs_command_in_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            log = temp / "docker.log"
            fake_docker = temp / "docker"
            fake_docker.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
                "exit 0\n",
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
            self.assertEqual(
                calls[0], f"build -t kairyu-bench:local {ROOT}"
            )
            self.assertIn("run --rm --privileged", calls[1])
            self.assertIn(
                f"-v {ROOT / 'results'}:{ROOT / 'results'}", calls[1]
            )
            self.assertIn(
                f"-v {ROOT / '.cache'}:{ROOT / '.cache'}", calls[1]
            )
            self.assertIn(
                f"-e KAIRYU_BENCH_RESULTS_DIR={ROOT / 'results'}", calls[1]
            )
            self.assertIn(
                f"-e KAIRYU_BENCH_CACHE_DIR={ROOT / '.cache'}", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES=3", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS=1", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_SWEBENCH_PRO_WORKERS=1", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_LIGHTCPVERIFIER_HOST=container", calls[1]
            )
            self.assertIn("-e HF_TOKEN", calls[1])
            self.assertTrue(calls[1].endswith("kairyu-bench:local list"))


if __name__ == "__main__":
    unittest.main()
