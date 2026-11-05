#!/usr/bin/env python3
"""Bounded task-image prefetch; preview by default, no container/daemon operations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

MAX_INPUT_BYTES = 16 * 1024 * 1024
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*\Z")


class PrefetchError(ValueError):
    pass


class Stopped(PrefetchError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_stat(value: os.stat_result) -> dict[str, int]:
    return {k: int(getattr(value, k)) for k in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")}


def read_input(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        before = file_stat(os.fstat(handle.fileno()))
        if not stat.S_ISREG(before["st_mode"]):
            raise PrefetchError("Input must be a regular file")
        content = handle.read(MAX_INPUT_BYTES + 1)
        after = file_stat(os.fstat(handle.fileno()))
    if before != after or len(content) > MAX_INPUT_BYTES:
        raise PrefetchError("Input changed while reading or exceeds 16 MiB")
    return content, {"path": str(path), "stat": before,
                     "sha256": hashlib.sha256(content).hexdigest()}


def records(content: bytes) -> list[dict[str, Any]]:
    obj = json.loads(content)
    rows = obj.get("records") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise PrefetchError("Input must contain a nonempty records list")
    return rows


def repository(ref: str) -> str:
    without_digest = ref.split("@", 1)[0]
    last = without_digest.rsplit("/", 1)[-1]
    return without_digest.rsplit(":", 1)[0] if ":" in last else without_digest


def load_tasks(tasks: bytes, expected: bytes) -> list[dict[str, str]]:
    mapped: dict[str, str] = {}
    for row in records(tasks):
        task, ref = row.get("task"), row.get("ref")
        if not isinstance(task, str) or not task or task in mapped:
            raise PrefetchError("Task names must be nonempty and unique")
        if not isinstance(ref, str) or not REF.fullmatch(ref):
            raise PrefetchError("Invalid image reference")
        if "/" not in ref or (":" not in ref.rsplit("/", 1)[-1] and "@" not in ref):
            raise PrefetchError("Use an explicit registry/repository tag or digest")
        mapped[task] = ref
    expectations: dict[str, dict[str, str]] = {}
    per_ref: dict[str, str] = {}
    for row in records(expected):
        task, ref, digest_ref = row.get("task"), row.get("ref"), row.get("digest_reference")
        if not isinstance(task, str) or task not in mapped or mapped[task] != ref or task in expectations:
            raise PrefetchError("Digest mapping must match tasks/ref exactly once")
        if not isinstance(digest_ref, str) or "@" not in digest_ref:
            raise PrefetchError("Expected digest_reference is required")
        repo, digest = digest_ref.rsplit("@", 1)
        if repo != repository(ref) or not DIGEST.fullmatch(digest):
            raise PrefetchError("Expected digest must belong to the exact image repository")
        if row.get("registry_digest", digest) != digest:
            raise PrefetchError("registry_digest and digest_reference disagree")
        if "@" in ref and ref != digest_ref:
            raise PrefetchError("Requested digest reference disagrees with expectation")
        if ref in per_ref and per_ref[ref] != digest_ref:
            raise PrefetchError("One image reference has conflicting expected digests")
        per_ref[ref] = digest_ref
        expectations[task] = {"task": task, "ref": ref, "digest_reference": digest_ref}
    if set(expectations) != set(mapped):
        raise PrefetchError("Missing expected digests")
    return [expectations[task] for task in sorted(mapped)]


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class DockerRunner:
    """Track only subprocesses created by this invocation; never inspect/kill others."""
    def __init__(self, stop: threading.Event, output: Path):
        self.stop, self.output = stop, output
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen[Any]] = set()

    def call(self, args: list[str], timeout: float, launch_guard=None) -> CommandResult:
        if self.stop.is_set():
            raise Stopped("Stop requested")
        # Temporary streams bound RAM and are deleted on close; raw CLI output is not kept.
        with tempfile.TemporaryFile(dir=self.output) as out, tempfile.TemporaryFile(dir=self.output) as err:
            with launch_guard() if launch_guard else nullcontext():
                child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            with self.lock:
                self.children.add(child)
            deadline, timed_out = time.monotonic() + timeout, False
            try:
                while child.poll() is None:
                    if self.stop.wait(0.1):
                        raise Stopped("Stop requested")
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                if timed_out:
                    self._terminate(child)
                code = child.wait()
                if out.tell() > MAX_INPUT_BYTES or err.tell() > MAX_INPUT_BYTES:
                    raise PrefetchError("Docker output exceeded 16 MiB; output omitted")
                out.seek(0)
                err.seek(0)
                return CommandResult(code, out.read().decode("utf-8", "replace"),
                                     err.read().decode("utf-8", "replace"), timed_out)
            finally:
                if child.poll() is None:
                    self._terminate(child)
                with self.lock:
                    self.children.discard(child)

    @staticmethod
    def _terminate(child: subprocess.Popen[Any]) -> None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def close(self) -> None:
        with self.lock:
            children = list(self.children)
        for child in children:
            self._terminate(child)


def retry_class(result: CommandResult) -> str | None:
    if result.timed_out:
        return "timeout"
    text = result.stderr.lower()
    if re.search(r"toomanyrequests|too many requests|rate exceeded|rate limit exceeded|\b429\b", text):
        return "rate_limit"
    if any(word in text for word in
           ("i/o timeout", "context deadline exceeded", "tls handshake timeout", "client.timeout")):
        return "timeout"
    return None


class Prefetch:
    def __init__(self, args: argparse.Namespace, runner_factory=DockerRunner):
        self.args = args
        self.stop = threading.Event()
        self.lock, self.pace_lock = threading.Lock(), threading.Lock()
        self.next_issue = 0.0
        self.inputs: list[dict[str, Any]] = []
        self.output = args.output.resolve()
        if os.path.lexists(args.output) or self.output.exists():
            raise PrefetchError("Output must be fresh")
        contents = []
        for path in (args.tasks, args.expected_digests, Path(__file__)):
            content, snapshot = read_input(path.resolve(strict=True))
            contents.append(content)
            self.inputs.append(snapshot)
        self.tasks = load_tasks(contents[0], contents[1])
        self.socket = args.docker_socket.resolve(strict=True)
        self.socket_stat = file_stat(self.socket.stat())
        if not stat.S_ISSOCK(self.socket_stat["st_mode"]):
            raise PrefetchError("docker-socket must be an existing Unix socket")
        self.free_path = args.min_free_path.resolve(strict=True)
        self.free_device = self.free_path.stat().st_dev
        self.check_inputs(full=True)
        self.check_free()
        self.output.mkdir(parents=True, exist_ok=False)
        self.runner = runner_factory(self.stop, self.output)
        self.prefix = ["docker", "--host", "unix://" + str(self.socket)]
        self.receipt: dict[str, Any] = {
            "schema_version": 1, "started_at": now(), "status": "prepared",
            "apply": args.apply, "inputs": self.inputs, "tasks": self.tasks,
            "docker_socket": str(self.socket), "socket_stat": self.socket_stat,
            "expected_daemon_id": args.expected_daemon_id, "daemon_id_verified": False,
            "policy": {k: getattr(args, k) for k in (
                "max_parallel", "issue_interval_seconds", "max_retries",
                "backoff_seconds", "max_backoff_seconds", "pull_timeout_seconds",
                "command_timeout_seconds", "min_free_bytes")},
            "min_free_path": str(self.free_path), "image_results": {},
            "gpu_or_container_execution_or_daemon_configuration_changes": False,
            "scope": "RepoDigests identity, not derived trial image IDs; no content execution",
        }

    def save(self) -> None:
        with self.lock:
            tmp = self.output / ".receipt.tmp"
            tmp.write_text(json.dumps(self.receipt, indent=2) + "\n")
            tmp.replace(self.output / "receipt.json")

    def check_inputs(self, full: bool = False) -> None:
        for item in self.inputs:
            path = Path(item["path"])
            if file_stat(path.stat()) != item["stat"]:
                raise PrefetchError("Input stat/inventory changed")
            if full and read_input(path)[1] != item:
                raise PrefetchError("Input hash changed")

    def check_free(self) -> int:
        if self.free_path.stat().st_dev != self.free_device:
            raise PrefetchError("Free-space filesystem changed")
        free = shutil.disk_usage(self.free_path).free
        if free < self.args.min_free_bytes:
            raise PrefetchError("Minimum free space guard failed")
        return free

    def check_daemon(self) -> None:
        if file_stat(self.socket.stat()) != self.socket_stat:
            raise PrefetchError("Docker socket identity changed")
        result = self.runner.call(self.prefix + ["info", "--format", "{{json .ID}}"],
                                  self.args.command_timeout_seconds)
        if result.returncode or result.timed_out:
            raise PrefetchError("Could not verify Docker daemon ID")
        if json.loads(result.stdout) != self.args.expected_daemon_id:
            raise PrefetchError("Unexpected Docker daemon ID")

    def inspect(self, ref: str, expected: str) -> dict[str, Any] | None:
        result = self.runner.call(self.prefix + ["image", "inspect", ref, "--format", "{{json .}}"],
                                  self.args.command_timeout_seconds)
        if result.returncode:
            if not result.timed_out and re.search(r"no such (image|object)", result.stderr, re.I):
                return None
            raise PrefetchError("Image inspection failed")
        if result.timed_out:
            raise PrefetchError("Image inspection timed out")
        image = json.loads(result.stdout)
        digests = image.get("RepoDigests") or []
        if not isinstance(digests, list) or expected not in digests:
            raise PrefetchError("Image RepoDigests mismatch")
        if not isinstance(image.get("Id"), str) or not DIGEST.fullmatch(image["Id"]):
            raise PrefetchError("Invalid Docker config image ID")
        return {"image_id": image["Id"], "expected_digest_verified": expected,
                "repo_digests": digests}

    @contextmanager
    def pull_launch(self):
        # Serialize only launch: wait, verify the selected daemon, then spawn.
        # Waiting for the child is outside this gate, preserving bounded parallelism.
        with self.pace_lock:
            delay = max(0.0, self.next_issue - time.monotonic())
            if self.stop.wait(delay):
                raise Stopped("Stop requested")
            self.check_inputs()
            self.check_free()
            self.check_daemon()
            yield
            self.next_issue = time.monotonic() + self.args.issue_interval_seconds

    def image(self, ref: str, expected: str) -> dict[str, Any]:
        result: dict[str, Any] = {"ref": ref, "expected": expected, "attempts": [], "status": "checking"}
        try:
            if self.stop.is_set():
                raise Stopped("Stop requested")
            self.check_inputs()
            self.check_daemon()
            cached = self.inspect(ref, expected)
            if cached:
                result.update(status="cached_verified", **cached)
                return result
            for attempt in range(self.args.max_retries + 1):
                if attempt:
                    delay = min(self.args.max_backoff_seconds,
                                self.args.backoff_seconds * (2 ** (attempt - 1)))
                    if self.stop.wait(delay):
                        raise Stopped("Stop requested")
                started = time.monotonic()
                pulled = self.runner.call(self.prefix + ["image", "pull", ref],
                                          self.args.pull_timeout_seconds, self.pull_launch)
                kind = retry_class(pulled)
                result["attempts"].append({"attempt": attempt + 1, "at": now(),
                    "seconds": time.monotonic() - started, "returncode": pulled.returncode,
                    "retry_class": kind, "timed_out": pulled.timed_out})
                self.check_inputs()
                self.check_daemon()
                self.check_free()
                # A timeout may leave a fully downloaded image; always inspect before retry.
                actual = self.inspect(ref, expected)
                if actual is not None:
                    result.update(status="pulled_verified", **actual)
                    return result
                if pulled.returncode == 0 and not pulled.timed_out:
                    raise PrefetchError("Successful pull did not produce the expected image")
                if kind is None or attempt == self.args.max_retries:
                    raise PrefetchError("Pull failed without an eligible retry")
            raise PrefetchError("Retry loop exhausted")
        except Stopped as error:
            result.update(status="cancelled", error=str(error))
        except (PrefetchError, OSError, ValueError, KeyError) as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.stop.set()
        return result

    def execute(self) -> int:
        signals: dict[int, Any] = {}
        try:
            if not self.args.apply:
                self.receipt["status"] = "preview"
                self.receipt["preview_note"] = "No Docker commands issued; daemon/cache not verified"
            else:
                if threading.current_thread() is threading.main_thread():
                    for signum in (signal.SIGINT, signal.SIGTERM):
                        signals[signum] = signal.signal(signum, lambda *_: self.stop.set())
                self.check_daemon()
                self.receipt.update(status="running", daemon_id_verified=True)
                self.save()
                jobs = {row["ref"]: row["digest_reference"] for row in self.tasks}
                with ThreadPoolExecutor(max_workers=self.args.max_parallel) as pool:
                    try:
                        futures = {pool.submit(self.image, ref, digest): ref for ref, digest in jobs.items()}
                        for future in as_completed(futures):
                            self.receipt["image_results"][futures[future]] = future.result()
                            self.save()
                    except BaseException:
                        # Set this before Executor.__exit__ waits for queued/running work.
                        self.stop.set()
                        raise
                self.check_inputs(full=True)
                if not self.stop.is_set():
                    self.check_daemon()
                    self.check_free()
                values = self.receipt["image_results"].values()
                passed = all(v["status"] in ("cached_verified", "pulled_verified") for v in values)
                self.receipt["status"] = "completed" if passed and not self.stop.is_set() else "incomplete"
        except (PrefetchError, OSError, ValueError, KeyError) as error:
            self.stop.set()
            self.receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            self.runner.close()
            for signum, previous in signals.items():
                signal.signal(signum, previous)
            self.receipt["finished_at"] = now()
            try:
                self.check_inputs(full=True)
                self.receipt["input_continuity_verified"] = True
            except (OSError, ValueError) as error:
                self.receipt.update(status="failed", input_continuity_verified=False,
                                    final_input_error=str(error))
            self.save()
        return 0 if self.receipt["status"] in ("preview", "completed") else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docker-socket", required=True, type=Path)
    p.add_argument("--expected-daemon-id", required=True)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--expected-digests", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-free-path", required=True, type=Path)
    p.add_argument("--min-free-bytes", required=True, type=int)
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument("--issue-interval-seconds", type=float, default=2.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-seconds", type=float, default=5.0)
    p.add_argument("--max-backoff-seconds", type=float, default=120.0)
    p.add_argument("--pull-timeout-seconds", type=float, default=900.0)
    p.add_argument("--command-timeout-seconds", type=float, default=30.0)
    p.add_argument("--apply", action="store_true")
    return p


def main(argv: list[str] | None = None, runner_factory=DockerRunner) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not 1 <= args.max_parallel <= 32 or not 0 <= args.max_retries <= 16:
        p.error("max-parallel must be 1..32; max-retries must be 0..16")
    if args.min_free_bytes < 0 or not args.expected_daemon_id.strip():
        p.error("min-free-bytes must be nonnegative and expected-daemon-id nonempty")
    for name in ("issue_interval_seconds", "backoff_seconds", "max_backoff_seconds",
                 "pull_timeout_seconds", "command_timeout_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or ("timeout" in name and value == 0):
            p.error(name + " must be finite and nonnegative (timeouts positive)")
    if args.max_backoff_seconds < args.backoff_seconds:
        p.error("max-backoff-seconds must be >= backoff-seconds")
    try:
        return Prefetch(args, runner_factory).execute()
    except (PrefetchError, OSError, ValueError, KeyError) as error:
        print("prefetch rejected: " + str(error), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
