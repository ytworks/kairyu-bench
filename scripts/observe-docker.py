#!/usr/bin/env python3
"""Opt-in Docker replica observation. Dry-run by default; never changes model weights."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import urllib.parse
import uuid

WORKER = "kairyu_bench.wsp.observe.vllm_worker.CaptureWorker"
MIDDLEWARE = "kairyu_bench.wsp.observe.middleware.CaptureMiddleware"
LABEL = "kairyu.bench.observer_operation"


class Docker:
    def __init__(self, path: str):
        self.path = path

    def connection(self):
        class Connection(http.client.HTTPConnection):
            def connect(connection):
                connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.sock.connect(self.path)
        return Connection("localhost", timeout=120)

    def call(self, method: str, path: str, body=None):
        connection = self.connection()
        try:
            connection.request(method, path, body=None if body is None else json.dumps(body),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 300:
                # Engine errors can echo command/environment secrets.
                raise RuntimeError(f"Docker {method} failed (HTTP {response.status})")
            return json.loads(raw) if raw else {}
        finally:
            connection.close()

    def logs(self, cid: str, limit: int = 65536) -> bytes:
        connection = self.connection()
        try:
            connection.request("GET", f"/containers/{cid}/logs?stdout=1&stderr=1&timestamps=1&tail=200")
            response = connection.getresponse()
            if response.status >= 300:
                raise RuntimeError("Docker failed-container logs unavailable")
            # Engine framing is retained verbatim. Never decode or print secrets.
            return response.read(limit)
        finally:
            connection.close()

    def inspect(self, name: str):
        return self.call("GET", "/containers/" + urllib.parse.quote(name, safe="") + "/json")


def private_json(path: Path, value) -> None:
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def creation(original):
    keys = ("Hostname", "Domainname", "User", "AttachStdin", "AttachStdout", "AttachStderr",
            "ExposedPorts", "Tty", "OpenStdin", "StdinOnce", "Env", "Cmd", "Healthcheck",
            "WorkingDir", "Entrypoint", "Labels", "StopSignal", "StopTimeout", "Volumes",
            "NetworkDisabled", "MacAddress", "Shell")
    config = {key: copy.deepcopy(original["Config"][key]) for key in keys if key in original["Config"]}
    config["Image"] = original["Image"]  # Immutable image ID, not a mutable tag.
    config["HostConfig"] = copy.deepcopy(original["HostConfig"])
    network_keys = ("Aliases", "Links", "IPAMConfig", "DriverOpts")
    config["NetworkingConfig"] = {"EndpointsConfig": {
        name: {key: copy.deepcopy(network[key]) for key in network_keys if network.get(key) is not None}
        for name, network in original["NetworkSettings"]["Networks"].items()
    }}
    return config


def canonical(path: Path) -> Path:
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("all paths must be absolute, canonical paths without symlink components")
    return path


def overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_capture(source: Path, capture: Path) -> None:
    """Use the exact deployed parser; collector import itself is stdlib-only."""
    path = source / "kairyu_bench/wsp/observe/collector.py"
    name = "_kairyu_observer_validation_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("source collector cannot be loaded for capture validation")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules[name] = module  # Dataclasses resolve annotations through this entry.
        spec.loader.exec_module(module)
        module.CaptureConfig.read(capture)
    finally:
        sys.modules.pop(name, None)
        sys.dont_write_bytecode = previous


def save_failure_log(docker, cid: str, directory: Path, state) -> None:
    try:
        data = docker.logs(cid, 65536)[:65536]
        fd = os.open(directory / "failed-container.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        state["failed_log_bytes"] = len(data)
        state["failed_log_format"] = "bounded_docker_response_bytes_may_include_stream_frames"
    except (OSError, RuntimeError):
        state["failed_log_unavailable"] = True


def attach_plan(original, args):
    source = canonical(args.source)
    capture = canonical(args.capture)
    backup = canonical(args.backup_dir)
    cache = canonical(args.runtime_cache) / args.container
    if not (source / "kairyu_bench/wsp/observe/vllm_worker.py").is_file():
        raise ValueError("--source must be the src directory containing the observer package")
    validate_capture(source, capture)
    document = json.loads(capture.read_text())
    output = canonical(Path(document["output_dir"]))
    if (document.get("schema_version") != 1 or document.get("observation_mode") != "live"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", document.get("capture_id", ""))
            or not capture.is_relative_to(output)):
        raise ValueError("capture must be a live schema-1 configuration inside its output directory")
    if any(overlaps(a, b) for i, a in enumerate((source, output, cache, backup))
           for b in (source, output, cache, backup)[i + 1:]):
        raise ValueError("source, output, runtime cache and backup directories must be disjoint")
    if cache.exists() and (not cache.is_dir() or any(cache.iterdir())):
        raise ValueError("runtime cache must be new or empty; never reuse a non-observer AOT cache")
    if (backup / args.container).exists():
        raise ValueError("container backup already exists; use restore or an independent backup directory")
    config = creation(original)
    host = config["HostConfig"]
    if host.get("AutoRemove") or str(host.get("NetworkMode", "")).startswith("container:"):
        raise ValueError("auto-remove and shared-container network namespaces are unsupported")
    command = config.get("Cmd")
    if not isinstance(command, list) or not command:
        raise ValueError("observer attachment requires an exec-form vLLM command")
    all_args = [*(config.get("Entrypoint") or []), *command]
    environment = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    if any("--worker-cls" in item or WORKER in item or MIDDLEWARE in item for item in all_args) or (
            "KAIRYU_BENCH_CAPTURE_CONFIG" in environment or LABEL in (config.get("Labels") or {})):
        raise ValueError("container already has a worker override or observation configuration")
    checkpoint = canonical(Path(document["checkpoint"]["path"]))
    covering = [m for m in original.get("Mounts", [])
                if m.get("Type") == "bind" and checkpoint.is_relative_to(Path(m["Source"]))]
    if not covering:
        raise ValueError("declared checkpoint is not covered by an existing bind mount")
    mount = max(covering, key=lambda m: len(Path(m["Source"]).parts))
    if mount.get("RW") or any(
        m.get("RW") and Path(m["Destination"]).is_relative_to(Path(mount["Destination"]))
        for m in original.get("Mounts", []) if m is not mount
    ):
        raise ValueError("checkpoint mount and nested mounts must be read-only")
    if any(overlaps(path, checkpoint) for path in (source, output, cache, backup)):
        raise ValueError("observation paths must not overlap checkpoint files")
    destinations = [Path(m["Destination"]) for m in original.get("Mounts", [])]
    if any(path == dest or dest.is_relative_to(path) for path in (source, output, cache) for dest in destinations):
        raise ValueError("additional observer mount would replace an existing container mount")
    replica = args.replica or args.container
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", replica):
        raise ValueError("invalid replica identifier")
    if any((output / "workers" / f"{replica}-tp{rank}").exists()
           for rank in range(int(document["runtime"]["tp_size"]))):
        raise ValueError("capture already contains this replica; use a fresh capture or replica")
    operation = uuid.uuid4().hex
    environment.update(KAIRYU_BENCH_CAPTURE_CONFIG=str(capture), KAIRYU_BENCH_CAPTURE_REPLICA=replica,
                       VLLM_CACHE_ROOT=str(cache))
    environment["PYTHONPATH"] = str(source) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    config["Env"] = [key + "=" + value for key, value in environment.items()]
    config["Cmd"] += ["--worker-cls", WORKER, "--middleware", MIDDLEWARE]
    if "--enable-request-id-headers" not in all_args:
        config["Cmd"].append("--enable-request-id-headers")
    host["Binds"] = [*(host.get("Binds") or []), f"{source}:{source}:ro",
                     f"{output}:{output}:rw", f"{cache}:{cache}:rw"]
    config["Labels"] = {**(config.get("Labels") or {}), LABEL: operation,
                        "kairyu.bench.capture_id": document["capture_id"]}
    summary = {"operation": "attach", "container": args.container, "image_id": original["Image"],
               "user_preserved": config.get("User", "") == original["Config"].get("User", ""),
               "checkpoint_read_only": True, "capture_id": document["capture_id"],
               "replica": replica, "capture": str(capture), "source": str(source),
               "runtime_cache": str(cache), "backup": str(backup / args.container),
               "networks": sorted(config["NetworkingConfig"]["EndpointsConfig"]),
               "environment_values_displayed": False,
               "healthcheck": bool(config.get("Healthcheck", {}).get("Test")),
               "dynamic_network_addresses_may_change": True}
    return config, summary, operation


def wait_ready(docker, cid, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = docker.inspect(cid)["State"]
        if not state.get("Running") or state.get("Status") in ("dead", "exited"):
            raise RuntimeError("replacement container stopped before becoming ready")
        health = state.get("Health", {}).get("Status")
        if health in (None, "healthy"):
            return
        if health == "unhealthy" or time.monotonic() >= deadline:
            raise RuntimeError("replacement container health check failed or timed out")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def stop(docker, cid):
    if docker.inspect(cid)["State"].get("Running"):
        docker.call("POST", f"/containers/{cid}/stop?t=30")


def rename(docker, cid, name):
    docker.call("POST", f"/containers/{cid}/rename?name=" + urllib.parse.quote(name, safe=""))


def checked_current(docker, name, expected):
    if docker.inspect(name)["Id"] != expected:
        raise RuntimeError("container identity changed; refusing to modify another container")


def apply_attach(docker, original, config, summary, operation, timeout):
    directory = Path(summary["backup"])
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.mkdir(mode=0o700)  # Exclusive reservation; a second attach cannot overwrite evidence.
    private_json(directory / "original.json", original)
    original_name = summary["container"]
    backup_name = original_name[:90] + "-observer-backup-" + operation[:12]
    state = {**summary, "status": "prepared", "operation_id": operation,
             "original_id": original["Id"], "original_name": original_name, "backup_name": backup_name,
             "original_was_running": bool(original["State"].get("Running")),
             "original_sha256": hashlib.sha256((directory / "original.json").read_bytes()).hexdigest()}
    private_json(directory / "state.json", state)
    Path(summary["runtime_cache"]).mkdir(parents=True, exist_ok=True)
    created = None
    changed = False
    try:
        checked_current(docker, original_name, original["Id"])
        changed = True
        stop(docker, original["Id"])
        rename(docker, original["Id"], backup_name)
        state["status"] = "original_preserved_stopped"
        private_json(directory / "state.json", state)
        created = docker.call("POST", "/containers/create?name=" + urllib.parse.quote(original_name, safe=""), config)["Id"]
        state.update(status="replacement_created", observer_id=created)
        private_json(directory / "state.json", state)
        docker.call("POST", f"/containers/{created}/start")
        wait_ready(docker, created, timeout)
        state["status"] = "attached"
        private_json(directory / "state.json", state)
        return {**summary, "status": "attached", "container_id": created}
    except BaseException as error:
        # Suppress subsequent stop signals while restoring the original owner.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if changed:
            try:
                if created is None:
                    try:
                        possible = docker.inspect(original_name)
                        if possible["Config"].get("Labels", {}).get(LABEL) == operation:
                            created = possible["Id"]
                    except RuntimeError:
                        pass
                if created is not None:
                    current = docker.inspect(created)
                    if current["Config"].get("Labels", {}).get(LABEL) != operation:
                        raise RuntimeError("replacement ownership mismatch during rollback")
                    save_failure_log(docker, created, directory, state)
                    docker.call("DELETE", f"/containers/{created}?force=true")
                rename(docker, original["Id"], original_name)
                if state["original_was_running"]:
                    docker.call("POST", f"/containers/{original['Id']}/start")
                    wait_ready(docker, original["Id"], timeout)
                state["status"] = "rolled_back"
            except BaseException as rollback_error:
                state.update(status="rollback_failed", rollback_error=type(rollback_error).__name__)
        else:
            state["status"] = "not_applied"
        state["error"] = type(error).__name__
        private_json(directory / "state.json", state)
        raise


def restore(docker, args):
    directory = canonical(args.backup_dir) / args.container
    state = json.loads((directory / "state.json").read_text())
    original_bytes = (directory / "original.json").read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != state["original_sha256"]:
        raise ValueError("original backup hash mismatch")
    if state["original_name"] != args.container:
        raise ValueError("backup belongs to another container")
    original = docker.inspect(state["original_id"])
    current = docker.inspect(args.container)
    if current["Id"] == original["Id"] and state["status"] in ("restored", "restored_stopped", "rolled_back"):
        return {"operation": "restore", "container": args.container, "status": "already_original"}
    if current["Config"].get("Labels", {}).get(LABEL) != state["operation_id"]:
        raise ValueError("current container is not the observer owned by this backup")
    if state.get("observer_id", current["Id"]) != current["Id"]:
        raise ValueError("observer container identity differs from saved receipt")
    summary = {"operation": "restore", "container": args.container,
               "original_id": original["Id"], "observer_id": current["Id"], "environment_values_displayed": False,
               "leave_stopped": args.leave_stopped}
    if not args.apply:
        return {**summary, "status": "dry_run"}
    lock = (directory / "restore.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    retired_name = args.container[:90] + "-observer-retired-" + state["operation_id"][:12]
    try:
        checked_current(docker, args.container, current["Id"])
        stop(docker, current["Id"])
        rename(docker, current["Id"], retired_name)
        rename(docker, original["Id"], args.container)
        if args.leave_stopped:
            stop(docker, original["Id"])
        elif state["original_was_running"]:
            docker.call("POST", f"/containers/{original['Id']}/start")
            wait_ready(docker, original["Id"], args.health_timeout)
        state["status"] = "restored_stopped" if args.leave_stopped else "restored"
        private_json(directory / "state.json", state)
        # Keep the stopped observer as recovery evidence too; never prune other containers.
        return {**summary, "status": state["status"], "stopped_observer_name": retired_name}
    except BaseException:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop(docker, original["Id"])
            rename(docker, original["Id"], state["backup_name"])
            rename(docker, current["Id"], args.container)
            docker.call("POST", f"/containers/{current['Id']}/start")
            wait_ready(docker, current["Id"], args.health_timeout)
            state["status"] = "restore_rolled_back"
        except BaseException as error:
            state.update(status="restore_rollback_failed", rollback_error=type(error).__name__)
        private_json(directory / "state.json", state)
        raise
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("attach", "restore"))
    parser.add_argument("--container", required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--source", type=Path, help="read-only src directory mounted at its absolute path")
    parser.add_argument("--runtime-cache", type=Path, help="new cache root; one subdirectory per container")
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--replica", help="capture replica identifier; defaults to container name")
    parser.add_argument("--socket", default="/var/run/docker.sock")
    parser.add_argument("--health-timeout", type=float, default=900.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--leave-stopped", action="store_true",
                        help="restore original container without loading its model; restore only")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.container):
        parser.error("invalid container name")
    if args.health_timeout <= 0:
        parser.error("health timeout must be positive")
    if args.leave_stopped and args.operation != "restore":
        parser.error("--leave-stopped is only valid with restore")
    if args.operation == "attach" and any(x is None for x in (args.capture, args.source, args.runtime_cache)):
        parser.error("attach requires --capture, --source and --runtime-cache")
    docker = Docker(args.socket)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    def interrupted(signum, frame):
        raise InterruptedError("observer operation interrupted")
    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        if args.operation == "restore":
            result = restore(docker, args)
        else:
            original = docker.inspect(args.container)
            config, summary, operation = attach_plan(original, args)
            result = (apply_attach(docker, original, config, summary, operation, args.health_timeout)
                      if args.apply else {**summary, "status": "dry_run"})
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        # Exception class only: config/env/engine errors never reach terminal output.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "receipt_directory": str(args.backup_dir / args.container)}))
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
