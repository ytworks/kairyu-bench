from __future__ import annotations

import os
import subprocess
import socket
import sys
import textwrap
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
            # The wrapper validates a socket; this test never uses a real daemon.
            with socket.socket(socket.AF_UNIX) as docker_socket:
                socket_path = temp / "docker.sock"
                docker_socket.bind(str(socket_path))
                env["KAIRYU_BENCH_DOCKER_SOCKET"] = str(socket_path)
                env["KAIRYU_BENCH_DEEPSWE_WORKERS"] = "7"
                env["KAIRYU_BENCH_DEEPSWE_ATTEMPTS"] = "4"
                env["KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS"] = "3600"
                env["KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS"] = "5"
                env.pop("KAIRYU_BENCH_TERMINAL_BENCH_4_WORKERS", None)
                result = subprocess.run(
                    [str(ROOT / "kairyu-bench"), "list"], cwd=ROOT, env=env,
                    text=True, capture_output=True, check=False,
                )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                calls[0], f"build -t kairyu-bench:local {ROOT}"
            )
            self.assertIn("run --rm --privileged", calls[1])
            self.assertIn("-e KAIRYU_BENCH_DEEPSWE_WORKERS=7", calls[1])
            self.assertIn("-e KAIRYU_BENCH_DEEPSWE_ATTEMPTS=4", calls[1])
            self.assertIn("-e KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS", calls[1])
            self.assertIn("-e KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS=5", calls[1])
            self.assertIn("-e KAIRYU_BENCH_TERMINAL_BENCH_4_WORKERS=4", calls[1])
            self.assertIn("-e KAIRYU_BENCH_TERMINAL_BENCH_4_RETRIES=3", calls[1])
            self.assertIn("-e KAIRYU_BENCH_TERMINAL_BENCH_4_REASONING_EFFORT ", calls[1])
            # Relative arguments resolve against the caller's directory.
            self.assertIn(f"-v {ROOT}:{ROOT} -w {ROOT}", calls[1])
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
                "-e KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS=4", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_SWEBENCH_PRO_WORKERS=4", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_TERMINAL_BENCH_WORKERS=4", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER=4", calls[1]
            )
            self.assertIn(
                "-e KAIRYU_LIGHTCPVERIFIER_HOST=container", calls[1]
            )
            self.assertIn("-e HF_TOKEN", calls[1])
            self.assertTrue(calls[1].endswith("kairyu-bench:local list"))


    def test_explicit_results_survive_runner_removal_outside_calling_directory(self) -> None:
        # The fake daemon exposes only the declared binds. An unmounted output
        # goes to its temporary container filesystem and disappears on exit.
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            caller = temp / "caller"
            caller.mkdir()
            fake_docker = temp / "docker"
            fake_docker.write_text(
                f"#!{sys.executable}\n" + textwrap.dedent(r"""
                    import argparse
                    import json
                    import os
                    import sys
                    import tempfile
                    from pathlib import Path

                    args = sys.argv[1:]
                    if args[0] == "build":
                        raise SystemExit(0)
                    assert args[0] == "run"
                    image = args.index("fake-results-runner")
                    options, command = args[:image], args[image + 1:]
                    mounts = []
                    for index, value in enumerate(options):
                        if value == "-v":
                            source, destination, *mode = options[index + 1].split(":")
                            assert mode != ["ro"]
                            mounts.append((Path(source), Path(destination)))
                    assert len({destination for _, destination in mounts}) == len(mounts)
                    cwd = Path(options[options.index("-w") + 1])
                    parser = argparse.ArgumentParser(add_help=False)
                    parser.add_argument("--results-dir", type=Path, required=True)
                    parsed, rest = parser.parse_known_args(command)
                    assert rest in (["wsp", "run", "https://example.invalid/v1"],
                                    ["run", "https://example.invalid/v1"])
                    requested = parsed.results_dir
                    if not requested.is_absolute():
                        requested = cwd / requested
                    requested = Path(os.path.normpath(requested))
                    with tempfile.TemporaryDirectory() as ephemeral:
                        target = Path(ephemeral) / "unmounted"
                        for source, destination in sorted(mounts, key=lambda row: len(row[1].parts)):
                            if requested == destination or destination in requested.parents:
                                target = source / requested.relative_to(destination)
                        target = target / "finished"
                        target.mkdir(parents=True, exist_ok=True)
                        (target / "run.json").write_text(json.dumps({"status": "completed"}))
                        (target / "items.jsonl").write_text('{"score": 1}\n')
                        (target / "report.json").write_text('{"completed": true}\n')
                """),
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            env = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}",
                   "KAIRYU_BENCH_IMAGE": "fake-results-runner",
                   "KAIRYU_BENCH_CACHE_DIR": str(temp / "cache")}
            with socket.socket(socket.AF_UNIX) as docker_socket:
                socket_path = temp / "docker.sock"
                docker_socket.bind(str(socket_path))
                env["KAIRYU_BENCH_DOCKER_SOCKET"] = str(socket_path)
                cases = (
                    (["wsp", "run"], temp / "durable results" / "absolute",
                     ["--results-dir", str(temp / "durable results" / "absolute")]),
                    (["run"], temp / "durable results" / "relative",
                     ["--results-dir=../durable results/relative"]),
                    (["wsp", "run"], caller, ["--results-dir", "."]),
                )
                for command, output, options in cases:
                    with self.subTest(command=command, options=options):
                        result = subprocess.run(
                            [str(ROOT / "kairyu-bench"), *command, "https://example.invalid/v1", *options],
                            cwd=caller, env=env, text=True, capture_output=True, check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual((output / "finished" / "run.json").read_text(),
                                         '{"status": "completed"}')
                        self.assertEqual((output / "finished" / "items.jsonl").read_text(),
                                         '{"score": 1}\n')
                        self.assertEqual((output / "finished" / "report.json").read_text(),
                                         '{"completed": true}\n')


if __name__ == "__main__":
    unittest.main()
