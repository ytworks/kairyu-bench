#!/usr/bin/env python3
"""Preview or explicitly redeploy one existing example Compose replica.

No new serving stack. Full environment values remain in memory, never in reports.
A successful creation receipt can reuse its file hashes subject to the explicitly
reported file-ctime/completion interval assumption. A freeze uses saved full stats.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from kairyu_bench.wsp.weights import editor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024**2):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(argv, *, env=None, timeout=30):
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("metadata/compose command failed; raw output intentionally omitted")
    return result.stdout


def inspect(name):
    values = json.loads(command(["docker", "inspect", name]))
    if len(values) != 1:
        raise ValueError("expected one existing container")
    return values[0]


def stamp(path):
    s = path.stat()
    return {k: getattr(s, "st_" + key) for k, key in (
        ("device", "dev"), ("inode", "ino"), ("size", "size"), ("mtime_ns", "mtime_ns"),
        ("ctime_ns", "ctime_ns"), ("uid", "uid"), ("gid", "gid"), ("mode", "mode"), ("nlink", "nlink"))}


def inventory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be a regular directory")
    rows = {}
    for p in sorted(path.rglob("*")):
        if p.is_symlink():
            raise ValueError("checkpoint symlink refused")
        if p.is_file() and p.name != ".kairyu-model-attestation.json" and ".cache" not in p.relative_to(path).parts:
            rows[str(p.relative_to(path))] = stamp(p)
    return rows


def environments(rows):
    return dict(value.split("=", 1) for value in rows or [] if "=" in value)


def mounts(container):
    return sorted([{"type": m["Type"], "source": m["Source"], "target": m["Destination"],
                    "read_only": not m["RW"]} for m in container["Mounts"]], key=lambda x: x["target"])


def snapshot(container):
    config, host = container["Config"], container["HostConfig"]
    checkpoint = next(m for m in mounts(container) if m["target"] == "/checkpoint")
    return {"id": container["Id"], "name": container["Name"], "image_id": container["Image"],
            "command_sha256": digest({"entrypoint": config["Entrypoint"], "cmd": config["Cmd"]}),
            "environment_sha256": digest(environments(config["Env"])),
            "user": config["User"], "working_dir": config["WorkingDir"],
            "healthcheck_sha256": digest(config.get("Healthcheck")),
            "cpuset": host["CpusetCpus"], "devices": host.get("DeviceRequests"),
            "ipc_mode": host["IpcMode"], "mounts": mounts(container),
            "networks": sorted(container["NetworkSettings"]["Networks"]),
            "checkpoint_path": checkpoint["source"], "checkpoint_readonly": checkpoint["read_only"],
            "tree": (config.get("Labels") or {}).get("kairyu.chat-checkpoint-tree"),
            "running": container["State"]["Running"],
            "health": container["State"].get("Health", {}).get("Status")}


def identity(args):
    path = args.candidate.resolve()
    before = inventory(path)
    if not before:
        raise ValueError("empty candidate")
    if args.freeze:
        proof = read(args.freeze)
        if Path(proof["checkpoint_path"]).resolve() != path or before != proof["candidate_inventory_stat"]:
            raise ValueError("candidate differs from saved frozen inventory")
        rows = proof["all_checkpoint_files"]
        origin = {"kind": "saved_full_hash_plus_exact_stat_continuity", "path": str(args.freeze.resolve()),
                  "sha256": sha(args.freeze), "full_payload_rehashed_here": False}
    else:
        manifest_bytes = args.manifest.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        editor.validate_manifest(manifest)
        if Path(manifest["candidate"]).resolve() != path:
            raise ValueError("manifest names another candidate")
        jobs = read(args.creation_receipt)["jobs"]
        matching = [j for j in jobs if Path(j["candidate"]).resolve() == path
                    and Path(j["manifest"]).resolve() == args.manifest.resolve()]
        if len(matching) != 1 or matching[0].get("exit_code") != 0:
            raise ValueError("successful exact creator job required")
        job = matching[0]
        if manifest["spec_sha256"] != job["spec_sha256"]:
            raise ValueError("creator spec SHA mismatch")
        completed_ns = int((job["started_epoch"] + job["seconds"]) * 1e9)
        published_ns = args.manifest.stat().st_mtime_ns
        # No saved post-copy stat exists in historical native manifests. Do not
        # label this as equivalent to a previously captured full stat baseline.
        if published_ns > completed_ns + 1_000_000_000:
            raise ValueError("manifest publication after recorded creator completion")
        for rel, state in before.items():
            if state["uid"] != os.getuid() or state["nlink"] != 1:
                raise ValueError("new candidate must contain owned private files")
            if max(state["ctime_ns"], state["mtime_ns"]) > published_ns:
                raise ValueError("candidate file changed after manifest publication")
            if os.path.samefile(path / rel, Path(manifest["source"]) / rel):
                raise ValueError("candidate aliases original source")
        if path.stat().st_ctime_ns > published_ns:
            raise ValueError("candidate directory changed after publication")
        rows = [{"path": r["path"], "size": r["size"], "sha256": r["candidate_sha256"]}
                for r in manifest["files"]]
        if sha(args.manifest) != manifest_sha:
            raise ValueError("manifest changed while reading")
        origin = {"kind": "completed_creator_hashes_with_ctime_interval_assumption",
                  "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha,
                  "creator_receipt": str(args.creation_receipt.resolve()), "creator_job": job,
                  "creator_job_sha256": digest(job), "full_payload_rehashed_here": False,
                  "limitation": "No historical post-copy stat snapshot; assumes ordinary monotonic filesystem clock and no privileged ctime manipulation between creator proof and this fresh inventory."}
    expected = {r["path"]: r for r in rows}
    if len(expected) != len(rows) or set(expected) != set(before):
        raise ValueError("exact checkpoint inventory required")
    for rel, state in before.items():
        if expected[rel]["size"] != state["size"] or not re.fullmatch("[0-9a-f]{64}", expected[rel]["sha256"]):
            raise ValueError("file size or recorded SHA invalid")
    rows = sorted(rows, key=lambda r: r["path"])
    tree = digest(rows)
    if args.freeze and tree != proof["control_tree_sha256"]:
        raise ValueError("frozen canonical tree differs")
    if inventory(path) != before:
        raise ValueError("candidate changed during identity check")
    return {"path": str(path), "tree_sha256": tree, "files": rows, "stat": before, "proof": origin}


def check_render(control, env, service, current, candidate, tree):
    document = json.loads(command(control._compose_command(["config", "--format", "json"]), env=env))
    rendered = document["services"][service]
    old = snapshot(current)
    image = json.loads(command(["docker", "image", "inspect", old["image_id"]]))[0]
    if rendered["image"] != old["image_id"]:
        raise ValueError("rendered image is not pinned existing content ID")
    actual_config = current["Config"]
    effective_user = rendered.get("user") or image["Config"].get("User", "")
    effective_workdir = rendered.get("working_dir") or image["Config"].get("WorkingDir", "")
    if effective_user != actual_config["User"] or effective_workdir != actual_config["WorkingDir"]:
        raise ValueError("rendered user/working directory differs")
    if rendered["command"] != actual_config["Cmd"]:
        raise ValueError("rendered command differs; preserve existing observer/runtime flags explicitly")
    entrypoint = rendered.get("entrypoint")
    if entrypoint is None:
        entrypoint = image["Config"]["Entrypoint"]
    if entrypoint != actual_config["Entrypoint"]:
        raise ValueError("effective entrypoint differs")
    effective = environments(image["Config"].get("Env"))
    effective.update({k: str(v) for k, v in rendered.get("environment", {}).items()})
    if effective != environments(actual_config["Env"]):
        raise ValueError("rendered environment differs from current engine")
    if rendered.get("cpuset") != old["cpuset"] or rendered.get("ipc") != old["ipc_mode"]:
        raise ValueError("rendered CPU/IPC allocation differs")
    requests = rendered["deploy"]["resources"]["reservations"]["devices"]
    devices = [{"Driver": r.get("driver", ""), "Count": r.get("count", 0),
                "DeviceIDs": r.get("device_ids"), "Capabilities": [r["capabilities"]],
                "Options": r.get("options", {})} for r in requests]
    actual_devices = [dict(d, Options=d.get("Options") or {}) for d in old["devices"] or []]
    if devices != actual_devices:
        raise ValueError("rendered GPU allocation differs")
    wanted = [dict(m) for m in old["mounts"]]
    for mount in wanted:
        if mount["target"] == "/checkpoint":
            mount["source"], mount["read_only"] = str(candidate), True
    actual = sorted([{"type": m["type"], "source": m["source"], "target": m["target"],
                      "read_only": m.get("read_only", False)} for m in rendered["volumes"]], key=lambda x: x["target"])
    if actual != wanted or rendered.get("labels", {}).get("kairyu.chat-checkpoint-tree") != tree:
        raise ValueError("rendered mounts/tree differ beyond intended checkpoint")
    return digest(rendered)


def idle(container):
    if not container["State"]["Running"]:
        return
    if container["State"].get("Health", {}).get("Status") != "healthy":
        raise ValueError("existing target must be healthy before idle check")
    ips = [v["IPAddress"] for v in container["NetworkSettings"]["Networks"].values() if v["IPAddress"]]
    if len(ips) != 1:
        raise ValueError("expected one direct engine network")
    with urllib.request.urlopen("http://" + ips[0] + ":8000/metrics", timeout=10) as response:
        raw = response.read(8 * 1024**2).decode()
    for name in ("num_requests_running", "num_requests_waiting"):
        values = re.findall(r"^vllm:" + name + r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)$", raw, re.M)
        if not values or any(float(value) != 0 for value in values):
            raise ValueError("target has active/unknown request state")


def stop_owned_child(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def compose_up(control, env, service, timeout):
    argv = control._compose_command(["up", "--no-deps", "--force-recreate", "--detach",
                                      "--wait", "--wait-timeout", str(timeout), service])
    process = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        if process.wait(timeout=timeout + 30):
            raise RuntimeError("target compose recreation/readiness failed")
    except BaseException:
        if process.poll() is None:
            stop_owned_child(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_owned_child(process, signal.SIGKILL)
                process.wait()
        raise


def compare_after(old, current, candidate, tree):
    after = snapshot(current)
    keys = ("image_id", "command_sha256", "environment_sha256", "user", "working_dir",
            "healthcheck_sha256", "cpuset", "devices", "ipc_mode", "networks")
    if any(after[k] != old[k] for k in keys):
        raise ValueError("post-deployment runtime conditions differ")
    wanted = [dict(m) for m in old["mounts"]]
    for m in wanted:
        if m["target"] == "/checkpoint":
            m["source"], m["read_only"] = str(candidate), True
    if after["mounts"] != wanted or after["tree"] != tree or not after["running"] or after["health"] != "healthy":
        raise ValueError("post-deployment identity/readiness mismatch")
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--freeze", type=Path)
    parser.add_argument("--creation-receipt", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if os.getuid() == 0 or not 30 <= args.timeout <= 7200:
        raise ValueError("run as ordinary host user with bounded timeout")
    if args.manifest and not args.creation_receipt:
        raise ValueError("manifest mode requires successful creation receipt")
    if os.path.lexists(args.output):
        raise ValueError("output must be fresh")
    protected = [args.candidate.resolve(), args.control.resolve()]
    protected += [p.resolve() for p in (args.manifest, args.freeze, args.creation_receipt) if p]
    for output in (args.output.resolve(), args.baseline.resolve()):
        if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
            raise ValueError("outputs must be disjoint from checkpoint/input paths")
    if args.output.resolve() == args.baseline.resolve() or args.baseline.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("output and common baseline must be separate")
    if args.baseline.is_symlink():
        raise ValueError("baseline symlink refused")
    helper_sha = sha(Path(__file__))
    module_spec = importlib.util.spec_from_file_location("replica_example_control", args.control)
    control = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(control)
    if args.service not in control.CHAT_SERVICES:
        raise ValueError("service is not an existing declared chat replica")
    control_sha = sha(args.control)
    daemon = command(["docker", "info", "--format", "{{.ID}}"]).strip()
    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    with args.baseline.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not args.baseline.exists():
            frozen = {s: snapshot(inspect(f"{control.PROJECT}-{s}-1")) for s in control.CHAT_SERVICES}
            save(args.baseline, {"control_sha256": control_sha, "daemon_id": daemon,
                                 "ssh_uid": os.getuid(), "replicas": frozen, "created_epoch": time.time()})
        baseline = read(args.baseline)
    if baseline["daemon_id"] != daemon or baseline["control_sha256"] != control_sha or baseline["ssh_uid"] != os.getuid():
        raise ValueError("baseline control/daemon/UID differs")
    service_lock = args.baseline.parent / (args.baseline.stem + "-" + args.service + ".lock")
    with service_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = inspect(f"{control.PROJECT}-{args.service}-1")
        old = snapshot(original)
        if old != baseline["replicas"][args.service]:
            raise ValueError("target configuration/state changed since all-replica baseline freeze")
        ident = identity(args)
        env = control._compose_env()
        env.update(CHAT_CHECKPOINT_PATH=str(args.candidate.resolve()), CHAT_CHECKPOINT_TREE=ident["tree_sha256"],
                   VLLM_IMAGE=old["image_id"])
        for s, record in baseline["replicas"].items():
            env[s.replace("-", "_").upper() + "_CPUSET"] = record["cpuset"]
        rendered = check_render(control, env, args.service, original, args.candidate.resolve(), ident["tree_sha256"])
        args.output.mkdir(parents=True)
        report = {"status": "preview", "apply_requested": args.apply, "ssh_uid": os.getuid(),
                  "service": args.service, "baseline_path": str(args.baseline.resolve()), "baseline_sha256": sha(args.baseline),
                  "control_sha256": control_sha, "helper_sha256": helper_sha, "candidate_identity": ident, "before": old,
                  "rendered_service_sha256": rendered, "daemon_id": daemon,
                  "scope": "Existing Compose service only; global served-checkpoint record is not updated; never benchmark mixed gateway."}
        save(args.output / "receipt.json", report)
        if not args.apply:
            print(json.dumps({"status": "preview", "tree": ident["tree_sha256"], "output": str(args.output)}))
            return
        if inventory(args.candidate.resolve()) != ident["stat"] or snapshot(inspect(f"{control.PROJECT}-{args.service}-1")) != old:
            raise ValueError("candidate/target changed before apply")
        if sha(args.control) != control_sha or sha(Path(__file__)) != helper_sha:
            raise ValueError("deployment source changed")
        if check_render(control, env, args.service, original, args.candidate.resolve(), ident["tree_sha256"]) != rendered:
            raise ValueError("rendered deployment changed before apply")
        idle(inspect(f"{control.PROJECT}-{args.service}-1"))
        if command(["docker", "info", "--format", "{{.ID}}"]).strip() != daemon:
            raise ValueError("Docker daemon changed before apply")
        def interrupted(_signum, _frame):
            raise InterruptedError("deployment interrupted")
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous = {s: signal.signal(s, interrupted) for s in signals}
        try:
            report["status"] = "recreating"
            save(args.output / "receipt.json", report)
            compose_up(control, env, args.service, args.timeout)
            after_container = inspect(f"{control.PROJECT}-{args.service}-1")
            report["after"] = compare_after(old, after_container, args.candidate.resolve(), ident["tree_sha256"])
            if inventory(args.candidate.resolve()) != ident["stat"]:
                raise ValueError("candidate changed during startup")
            report["direct_ips"] = {n: v["IPAddress"] for n, v in after_container["NetworkSettings"]["Networks"].items()}
            report["status"] = "completed"
        except BaseException as error:
            for s in signals:
                signal.signal(s, signal.SIG_IGN)
            report.update(status="failed", error_type=type(error).__name__)
            try:
                if command(["docker", "info", "--format", "{{.ID}}"]).strip() != daemon:
                    raise ValueError("daemon changed; rollback refused")
                rollback = dict(env, CHAT_CHECKPOINT_PATH=old["checkpoint_path"], CHAT_CHECKPOINT_TREE=old["tree"])
                compose_up(control, rollback, args.service, args.timeout)
                report["rollback"] = compare_after(old, inspect(f"{control.PROJECT}-{args.service}-1"),
                                                     old["checkpoint_path"], old["tree"])
                if not old["running"]:
                    command(control._compose_command(["stop", "--timeout", "30", args.service]), env=rollback, timeout=60)
                    stopped = snapshot(inspect(f"{control.PROJECT}-{args.service}-1"))
                    if stopped["running"] or stopped["id"] != report["rollback"]["id"]:
                        raise ValueError("original stopped state not restored")
                    report["rollback"] = stopped
                report["rollback_completed"] = True
            except BaseException as rollback_error:
                report.update(rollback_completed=False, rollback_error_type=type(rollback_error).__name__)
            raise
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
            save(args.output / "receipt.json", report)
        print(json.dumps({"status": report["status"], "tree": ident["tree_sha256"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
