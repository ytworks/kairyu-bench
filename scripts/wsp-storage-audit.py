#!/usr/bin/env python3
"""Read-only WSP storage inventory; the sole optional write is the capacity ledger."""
import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tempfile

GIB = 1024**3
DEFAULT_ROOT = "/mnt/nvme/kairyu/wsp/gemma4-coding-20261104"
ORIGINAL = "/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/models/gemma-4-31b-it"
CACHE_ROOTS = [
    "/mnt/nvme/kairyu/bench-cache/wsp",
    "/mnt/nvme/kairyu/bench-cache/gemma-wsp-coding-20261104",
    "/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/compile-cache",
]
DAEMONS = [
    "/mnt/nvme/kairyu/docker/wsp/run/docker.sock",
    "/mnt/nvme/kairyu/docker/gemma-wsp-coding-20261104/run/docker.sock",
]


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=40)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def daemon_get(path, endpoint):
    conn = UnixHTTP(path)
    try:
        conn.request("GET", endpoint)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Docker GET {endpoint}: {response.status}")
        return json.load(response)
    finally:
        conn.close()


def scan(path, seen, owner, kind, mount_refs):
    row = {
        "path": str(path), "purpose": kind, "owner_trial": owner,
        "state": "present" if path.exists() else "absent",
        "logical_unique_bytes": 0, "allocated_unique_bytes": 0,
        "regular_files": 0, "shared_inode_file_count": 0,
        "symlinks": [], "unreadable_paths": [], "tensor_file_count": 0,
        "tensor_allocated_unique_bytes": 0, "checkpoint_indexes": [],
        "shared_references": mount_refs,
        "retention_reason": (
            "Designated original; never delete" if kind == "original" else
            "Preserve unique evidence and unknown observations; no deletion by this tool"
        ),
        "deletion_condition": (
            "never" if kind == "original" else
            "Explicit owned-path plan, no writers/evaluators/mount references, evidence retained"
        ),
    }
    if not path.exists():
        return row
    def error(exc):
        row["unreadable_paths"].append({"path": exc.filename, "errno": exc.errno})
    def add(p):
        try:
            info = p.lstat()
        except OSError as exc:
            error(exc)
            return
        if stat.S_ISLNK(info.st_mode):
            row["symlinks"].append({"path": str(p), "target": os.readlink(p)})
            return
        if not stat.S_ISREG(info.st_mode):
            return
        key = (info.st_dev, info.st_ino)
        row["regular_files"] += 1
        if info.st_nlink > 1:
            row["shared_inode_file_count"] += 1
        if key not in seen:
            seen.add(key)
            row["logical_unique_bytes"] += info.st_size
            row["allocated_unique_bytes"] += info.st_blocks * 512
            if p.suffix in {".safetensors", ".pt", ".pth", ".npy", ".npz"}:
                row["tensor_allocated_unique_bytes"] += info.st_blocks * 512
        if p.suffix in {".safetensors", ".pt", ".pth", ".npy", ".npz"}:
            row["tensor_file_count"] += 1
        if p.name == "model.safetensors.index.json":
            row["checkpoint_indexes"].append(str(p))
    if path.is_dir():
        for base, dirs, files in os.walk(path, followlinks=False, onerror=error):
            dirs.sort()
            for name in dirs + sorted(files):
                add(Path(base) / name)
    else:
        add(path)
    if row["unreadable_paths"]:
        row["state"] = "partially_unreadable"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--update-ledger", action="store_true")
    parser.add_argument("--restart-from-original", action="store_true",
                        help="Explicitly clear adoption only for the user-authorized fresh start")
    args = parser.parse_args()
    root = args.root.resolve()
    if str(root) != DEFAULT_ROOT:
        parser.error("This small task-owned audit supports only its documented work root")
    ledger_path = root / "storage-state.json"
    old = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    host_containers = []
    discovery_errors = []
    try:
        ids = subprocess.check_output(["docker", "ps", "-aq"], text=True).split()
        if ids:
            raw = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
            host_containers = [{
                "name": x["Name"], "state": x["State"]["Status"],
                "mounts": [{"source": m["Source"], "destination": m["Destination"], "rw": m["RW"]}
                           for m in x["Mounts"]],
            } for x in raw]
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        discovery_errors.append(type(exc).__name__)
    seen = set()
    entries = []
    paths = [(Path(ORIGINAL), "designated-original", "original")]
    paths += [(p, args.trial_id if p.name == "storage-state.json" else "historical-unclassified",
               "work-root evidence, tools, caches and possible weight-equivalent material")
              for p in sorted(root.iterdir())]
    paths += [(Path(p), "shared-task-cache", "regenerable cache") for p in CACHE_ROOTS]
    for path, owner, kind in paths:
        refs = [
            {"container": c["name"], "state": c["state"], **m}
            for c in host_containers for m in c["mounts"]
            if path == Path(m["source"]) or path in Path(m["source"]).parents
            or Path(m["source"]) in path.parents
        ]
        entries.append(scan(path, seen, owner, kind, refs))
    daemons = []
    for path in DAEMONS:
        row = {"socket": path, "state": "unknown"}
        try:
            info = daemon_get(path, "/info")
            disk = daemon_get(path, "/system/df")
            caches = disk.get("BuildCache") or []
            row.update({
                "state": "measured", "daemon_id": info["ID"], "daemon_name": info["Name"],
                "docker_root_dir": info["DockerRootDir"],
                "image_layers_logical_bytes": disk["LayersSize"],
                "build_cache_unshared_logical_bytes": sum(x.get("Size", 0) for x in caches
                                                          if not x.get("Shared")),
                "build_cache_shared_excluded_bytes": sum(x.get("Size", 0) for x in caches
                                                          if x.get("Shared")),
                "build_cache_in_use": sum(bool(x.get("InUse")) for x in caches),
                "build_cache_records": len(caches),
                "container_count": len(disk.get("Containers") or []),
                "volume_count": len(disk.get("Volumes") or []),
                "images": [{k: x.get(k) for k in
                            ("Id", "RepoTags", "RepoDigests", "Containers", "Size", "SharedSize")}
                           for x in disk.get("Images", [])],
                "actual_filesystem_allocation": None,
                "allocation_note": "Docker logical layer accounting; not host st_blocks or compressed bytes",
            })
        except (OSError, ValueError, RuntimeError) as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        daemons.append(row)
    work_entries = [x for x in entries if str(root) + "/" in x["path"]]
    cache_entries = [x for x in entries if x["purpose"] == "regenerable cache"]
    work_allocated = sum(x["allocated_unique_bytes"] for x in work_entries)
    cache_visible = sum(x["allocated_unique_bytes"] for x in cache_entries)
    docker_logical = sum(x.get("image_layers_logical_bytes", 0)
                         + x.get("build_cache_unshared_logical_bytes", 0) for x in daemons)
    unreadable = sum(len(x["unreadable_paths"]) for x in entries)
    free = shutil.disk_usage(root).free
    blockers = []
    if unreadable:
        blockers.append("Unreadable paths prevent complete allocation accounting; they are not excluded or zero")
    if discovery_errors or any(x["state"] != "measured" for x in daemons):
        blockers.append("Incomplete reference or Docker inventory")
    if docker_logical > 100 * GIB:
        blockers.append("Dedicated daemon logical caches alone exceed 100 GiB")
    if free < 200 * GIB:
        blockers.append("NVMe free space below 200 GiB")
    if work_allocated > 150 * GIB:
        blockers.append("Conservative readable work-root weight upper bound exceeds 150 GiB")
    old.update({
        "schema_version": 2, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "fresh-start storage reconciliation" if args.restart_from_original else "storage reconciliation",
        "audit_trial": args.trial_id, "original": ORIGINAL,
        "weight_budget_bytes": 150 * GIB, "cache_budget_bytes": 100 * GIB,
        "minimum_free_bytes": 200 * GIB, "nvme_free_bytes": free,
        "entries": entries, "dedicated_daemons": daemons,
        "weight_accounting": {
            "exact_total_bytes": None,
            "readable_work_root_allocated_upper_bound_bytes": work_allocated,
            "tensor_allocated_unique_bytes": sum(x["tensor_allocated_unique_bytes"] for x in work_entries),
            "method": "All readable work-root files counted conservatively, including JSON specs, evidence and caches; unreadable content is unknown",
        },
        "cache_accounting": {
            "exact_total_allocated_bytes": None,
            "visible_external_allocated_bytes": cache_visible,
            "dedicated_docker_logical_bytes": docker_logical,
            "method": "Docker LayersSize plus unshared build records; shared build records excluded. Docker logical and filesystem allocated metrics are not equated.",
        },
        "accounting": {
            "hardlinks": "Each regular-file device/inode charged once across ordered paths",
            "reflinks": "No sharing claim from st_blocks; backing filesystem ext4, no reflink assumed",
            "symlinks": "Not followed; targets recorded for reconciliation",
            "unreadable_path_count": unreadable,
            "external_exclusions": ["Other services and unrelated model volumes are not this task's storage; no deletion or pruning"],
            "process_references": "Host /proc fd/maps inaccessible for many processes; no absence-of-use claim",
            "discovery_errors": discovery_errors,
        },
        "generation_allowed": False,
        "generation_blockers": blockers or ["Creation still requires explicit peak temporary allocation and reference check"],
        "old_adoption_pointers": "Historical only; never restore deleted checkpoints automatically",
        "audit_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_generated_weight_bytes": 0, "audit_deleted_bytes": 0,
    })
    if args.restart_from_original:
        old.update({"adopted_checkpoint": None, "candidate_checkpoint": None, "checkpoint_count": 0})
    if args.update_ledger:
        with tempfile.NamedTemporaryFile("w", dir=root, prefix=".storage-state-", delete=False) as stream:
            json.dump(old, stream, indent=2)
            stream.write("\n")
            temporary = stream.name
        os.replace(temporary, ledger_path)
    print(json.dumps({k: old[k] for k in
                      ("updated_at", "nvme_free_bytes", "weight_accounting", "cache_accounting",
                       "generation_allowed", "generation_blockers")}, indent=2))


if __name__ == "__main__":
    main()
