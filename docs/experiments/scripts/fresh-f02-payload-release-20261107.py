#!/usr/bin/env python3
"""Fixed F02 lossless representation change; preview unless exact source/plan approved."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
BASE = ROOT / "checkpoints/code90-current-ad-helper-av-v1"
CANDIDATE = ROOT / "checkpoints/fresh-code-joint-f02"
ARCHIVE = ROOT / "analysis/fresh-purpose-20261107-f02/lossless-archive-v1"
PLAN = ROOT / "analysis/fresh-purpose-20261107-f02/lossless-release-plan-v2/plan.json"
PLAN_SHA = "9f758511d865d9444858bcf645746efced8f567246dcd3a4472b6c9b055f6a3a"
NAMES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
FIELDS = ("device", "inode", "size", "mtime_ns", "ctime_ns", "mode", "uid", "gid", "nlink")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def record_stat(s):
    return dict(zip(FIELDS, (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                            s.st_ctime_ns, s.st_mode, s.st_uid, s.st_gid, s.st_nlink), strict=True))


def stamp(path):
    s = Path(path).lstat()
    require(stat.S_ISREG(s.st_mode), "regular non-symlink file required")
    return record_stat(s)


def inventory(root):
    require(root.resolve() == root and root.is_dir() and not root.is_symlink(), "exact real directory required")
    result = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink() and (p.is_dir() or p.is_file()), "unexpected inventory entry")
        if p.is_file():
            result[str(p.relative_to(root))] = stamp(p)
    return result


def durable_json(directory_fd, name, value):
    """Exclusive durable evidence; never overwrite a prior receipt or journal."""
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(directory_fd)
    return hashlib.sha256(raw).hexdigest()


def mounts():
    ids = subprocess.check_output(["docker", "ps", "-aq", "--no-trunc"], text=True).split()
    require(bool(ids), "empty Docker inventory is not an absence proof")
    data = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True))
    result = []
    for d in data:
        selected = []
        for m in d.get("Mounts", []):
            source = m.get("Source", "")
            path = Path(source).resolve() if source.startswith("/") else None
            overlap = path is not None and (path == CANDIDATE or path.is_relative_to(CANDIDATE)
                                             or CANDIDATE.is_relative_to(path))
            require(not overlap, "F02 is reachable through a container mount")
            row = {k: m.get(k) for k in ("Type", "Source", "Destination", "RW")}
            row["candidate_overlap"] = overlap
            selected.append(row)
        selected.sort(key=lambda row: json.dumps(row, sort_keys=True))
        result.append(dict(cid=d["Id"], name=d["Name"], image=d["Image"], running=d["State"]["Running"],
                           status=d["State"]["Status"], health=d["State"].get("Health", {}).get("Status"),
                           mounts=selected))
    return sorted(result, key=lambda row: row["cid"])


def references(targets):
    hits, inaccessible, checked = [], [], 0
    inodes = {(s["device"], s["inode"]) for s in targets.values()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            checked += 1
            for fd in (proc / "fd").iterdir():
                try:
                    s = fd.stat()
                    if (s.st_dev, s.st_ino) in inodes:
                        hits.append(dict(pid=int(proc.name), kind="fd", descriptor=fd.name))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            for line in (proc / "maps").read_text().splitlines():
                if any(str(CANDIDATE / name) in line for name in NAMES):
                    hits.append(dict(pid=int(proc.name), kind="map"))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            inaccessible.append(int(proc.name))
    return dict(processes_checked=checked, matches=hits, inaccessible_or_disappeared=inaccessible,
                other_uid_processes_not_inspected=True)


def archive_check(plan):
    require(sha(ARCHIVE / "SHA256SUMS") == plan["archive_checksum_list_sha256"], "archive list changed")
    require(sha(ARCHIVE / "archive.json") == plan["archive_json_sha256"], "archive identity changed")
    before = inventory(ARCHIVE)
    rows = []
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        p = ARCHIVE / name
        require(p.resolve().is_relative_to(ARCHIVE) and not p.is_symlink(), "archive path escape")
        require(sha(p) == digest, "archive content mismatch")
        rows.append(dict(path=name, sha256=digest))
    require(len(rows) == 135 and len({r["path"] for r in rows}) == 135
            and set(before) == {r["path"] for r in rows} | {"SHA256SUMS"}, "archive inventory mismatch")
    require(inventory(ARCHIVE) == before, "archive changed during hash pass")
    archive = json.loads((ARCHIVE / "archive.json").read_text())
    require(archive["mandatory_base_path"] == str(BASE) and archive["candidate_path"] == str(CANDIDATE),
            "archive dependencies differ")
    validation = json.loads((ARCHIVE / "archive-validation.json").read_text())
    verify = json.loads((ARCHIVE / "records/materialization/verify.json").read_text())
    close = json.loads((ARCHIVE / "records/full-diff/full-diff-closure.json").read_text())
    require(validation["status"] == "passed" and verify["verified"] is True
            and verify["legacy_full_hashes_missing"] is False and len(verify["files"]) == 9
            and close["status"] == "full_diff_validated" and close["hashed_files_verified"] == 18,
            "saved full verification/reconstruction evidence incomplete")
    return archive, before, rows


def checkpoint_guard(archive, released, marker_present):
    expected_base = {r["path"]: r["source_stat_at_archive"] for r in archive["checkpoint_file_stats"]}
    expected_candidate = {r["path"]: r["candidate_stat_at_archive"] for r in archive["checkpoint_file_stats"]}
    require(len(expected_base) == len(expected_candidate) == 9 and set(NAMES) <= set(expected_candidate),
            "nine-file inventory required")
    require(BASE.stat().st_uid == CANDIDATE.stat().st_uid == 1003, "directory owner differs")
    require(inventory(BASE) == expected_base, "AV file stat continuity lost")
    current = inventory(CANDIDATE)
    if marker_present:
        require("ARCHIVED.json" in current, "durable archive marker absent")
        current.pop("ARCHIVED.json")
    require(current == {k: v for k, v in expected_candidate.items() if k not in released},
            "F02 file stat/inventory continuity lost")
    require({p.name for p in CANDIDATE.iterdir()} ==
            (set(expected_candidate) - set(released)) | ({"ARCHIVED.json"} if marker_present else set()),
            "extra candidate directory entry")
    for name in NAMES:
        s, b = expected_candidate[name], expected_base[name]
        require(stat.S_ISREG(s["mode"]) and s["uid"] == s["gid"] == 1003 and s["nlink"] == 1
                and (s["device"], s["inode"]) != (b["device"], b["inode"]), "private exact target required")
    return expected_base, expected_candidate


def directory_guard(fd):
    current = CANDIDATE.lstat()
    opened = os.fstat(fd)
    require(CANDIDATE.resolve() == CANDIDATE and stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino, current.st_uid, current.st_gid, current.st_mode) ==
                (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_gid, opened.st_mode),
            "candidate directory replaced")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--approved-operation-sha256")
    args = parser.parse_args()
    source_sha = sha(Path(__file__))
    require(os.getuid() == 1003 and sha(PLAN) == PLAN_SHA, "owner/fixed plan mismatch")
    require(not args.apply or (args.approved_plan_sha256 == PLAN_SHA
            and args.approved_operation_sha256 == source_sha), "exact reviewed source/plan approval required")
    require(args.apply or (args.approved_plan_sha256 is None and args.approved_operation_sha256 is None),
            "preview does not accept approval flags")
    out = args.output
    require(out.is_absolute() and out.resolve() == out and out.is_relative_to(ROOT / "analysis")
            and not out.exists() and not out.is_relative_to(ARCHIVE) and not ARCHIVE.is_relative_to(out),
            "fresh separate NVMe output required")
    out.mkdir()
    output_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    state = dict(status="precheck", apply=args.apply, operation_sha256=source_sha, plan_sha256=PLAN_SHA,
                 pid=os.getpid(), started_epoch=time.time(), released=[], fresh_payload_hash=False,
                 physical_reconstruction=False, model_calls=0, service_operations=0)
    lock_fd = candidate_fd = None
    try:
        lock_fd = os.open(ROOT / "candidate-evaluation.lock", os.O_RDWR | os.O_NOFOLLOW)
        require(os.fstat(lock_fd).st_uid == 1003 and os.fstat(lock_fd).st_nlink == 1
                and stat.S_ISREG(os.fstat(lock_fd).st_mode), "owned single-link regular lock required")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state["global_lock_acquired"] = True
        state["global_lock_stat"] = record_stat(os.fstat(lock_fd))
        plan = json.loads(PLAN.read_text())
        require(plan["mandatory_base"] == str(BASE) and plan["candidate"] == str(CANDIDATE)
                and plan["archive"] == str(ARCHIVE) and [t["path"] for t in plan["targets"]] ==
                    [str(CANDIDATE / name) for name in NAMES], "fixed targets/dependencies differ")
        old = plan["read_only_audit"]
        require(sha(old["path"]) == old["sha256"], "frozen service baseline changed")
        protected = json.loads(Path(old["path"]).read_text())["containers"]
        require(len(protected) == 27, "all 27 frozen containers required")
        archive, archive_stat, checks = archive_check(plan)
        base_stat, candidate_stat = checkpoint_guard(archive, [], False)
        for target in plan["targets"]:
            name = Path(target["path"]).name
            require(target["current_stat"] == candidate_stat[name] and target["bytes"] == candidate_stat[name]["size"]
                    and target["historical_full_sha256"] ==
                    next(row["candidate_sha256"] for row in archive["files"] if row["path"] == name),
                    "plan target byte identity changed")
        require(mounts() == protected, "protected container identity/state/mount drift")
        refs = references({name: candidate_stat[name] for name in NAMES})
        require(not refs["matches"], "visible target payload references")
        state.update(archive_checks=checks, base_stat=base_stat, candidate_stat=candidate_stat,
                     protected_containers=protected, reference_visibility=refs, free_bytes_before=shutil.disk_usage(CANDIDATE).free,
                     targets=plan["targets"], total_logical_bytes=sum(t["bytes"] for t in plan["targets"]))
        require(state["total_logical_bytes"] == 62546338264, "two-file byte budget differs")
        durable_json(output_fd, "precheck.json", state)
        if not args.apply:
            checkpoint_guard(archive, [], False)
            require(inventory(ARCHIVE) == archive_stat and mounts() == protected, "preview continuity lost")
            state["status"] = "preview_passed_no_checkpoint_changes"
            return
        candidate_fd = os.open(CANDIDATE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_guard(candidate_fd)
        marker = dict(schema_version=1, kind="lossless_dependent_archive", not_a_saved_model=True,
                      state="representation_change_in_progress", mandatory_base=str(BASE), archive=str(ARCHIVE),
                      archive_checksum_list_sha256=plan["archive_checksum_list_sha256"], original_nine_files=archive["files"],
                      reconstruction_instructions=str(ARCHIVE / "README.md"), operation_sha256=source_sha,
                      plan_sha256=PLAN_SHA, receipts=str(out), released=[])
        marker_sha = durable_json(candidate_fd, "ARCHIVED.json", marker)
        state["marker_created"] = True
        state["initial_marker_sha256"] = marker_sha
        for index, name in enumerate(NAMES):
            released = [Path(t["path"]).name for t in state["released"]]
            directory_guard(candidate_fd)
            checkpoint_guard(archive, released, True)
            require(sha(CANDIDATE / "ARCHIVED.json") == marker_sha and inventory(ARCHIVE) == archive_stat,
                    "marker/archive continuity lost")
            require(mounts() == protected, "container mount/identity drift before unlink")
            refs = references({n: candidate_stat[n] for n in NAMES if n not in released})
            require(not refs["matches"], "visible target payload reference before unlink")
            require(sha(PLAN) == PLAN_SHA and sha(Path(__file__)) == source_sha, "operation input changed")
            target = plan["targets"][index]
            durable_json(output_fd, f"journal-{index}-intent.json", dict(target=target, previously_released=state["released"]))
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=candidate_fd)
            try:
                require(record_stat(os.fstat(fd)) == candidate_stat[name] and
                        record_stat(os.stat(name, dir_fd=candidate_fd, follow_symlinks=False)) == candidate_stat[name],
                        "exact target changed immediately before unlink")
                os.unlink(name, dir_fd=candidate_fd)
                state["released"].append(target)
                os.fsync(candidate_fd)
            finally:
                os.close(fd)
            durable_json(output_fd, f"journal-{index}-released.json",
                         dict(released=state["released"], free_bytes=shutil.disk_usage(CANDIDATE).free))
        checkpoint_guard(archive, list(NAMES), True)
        archive_check(plan)
        require(inventory(ARCHIVE) == archive_stat, "archive stat changed after release")
        require(mounts() == protected, "protected service postclosure differs")
        for row in archive["files"]:
            if row["path"] not in NAMES:
                require(sha(CANDIDATE / row["path"]) == row["candidate_sha256"], "nonweight bytes changed")
        marker.update(state="archived_losslessly", released=state["released"])
        require(sha(CANDIDATE / "ARCHIVED.json") == marker_sha, "owned marker changed")
        final_marker_sha = durable_json(candidate_fd, "ARCHIVED.json.pending", marker)
        os.replace("ARCHIVED.json.pending", "ARCHIVED.json", src_dir_fd=candidate_fd, dst_dir_fd=candidate_fd)
        os.fsync(candidate_fd)
        directory_guard(candidate_fd)
        checkpoint_guard(archive, list(NAMES), True)
        require(sha(CANDIDATE / "ARCHIVED.json") == final_marker_sha and mounts() == protected,
                "final marker/service closure differs")
        state.update(status="completed_two_private_payloads_represented_by_archive",
                     final_marker_sha256=final_marker_sha, postclosure=True)
    except BaseException as error:
        state.update(status="partial_or_failed_no_automatic_retry", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        state.update(finished_epoch=time.time(), free_bytes_after=shutil.disk_usage(CANDIDATE).free,
                     released_logical_bytes=sum(t["bytes"] for t in state["released"]),
                     observed_absent_targets=[name for name in NAMES if not os.path.lexists(CANDIDATE / name)])
        state["elapsed_seconds"] = state["finished_epoch"] - state["started_epoch"]
        if "free_bytes_before" in state:
            state["free_bytes_delta"] = state["free_bytes_after"] - state["free_bytes_before"]
        try:
            durable_json(output_fd, "receipt.json", state)
        finally:
            if candidate_fd is not None:
                os.close(candidate_fd)
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(output_fd)
        print(json.dumps({k: state.get(k) for k in ("status", "released_logical_bytes", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
