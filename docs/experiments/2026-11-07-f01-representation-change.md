# F01 representation change — fixed one-off operation

The parent explicitly authorized releasing only the two private F01 safetensor files after the dependent lossless archive had passed verification. AV, the seven F01 nonweight files, original spec/manifest, all trial outputs and archives must remain. The operation below is the exact source written to the remote analysis directory; it is deliberately fixed to this trial and is not a general cleanup tool.

The prechecks read every archive file and compare its recorded SHA, pin the checksum-list SHA itself, compare every AV and F01 file to its archive-time full stat, require exact owned ordinary single-link targets, and inspect every running or stopped container mount. It rejects equal, descendant and ancestor mount overlap. It does not claim a new full weight hash or a physical reconstruction. A tombstone is written before unlink, and a separate result receipt records the actual released list and before/after disk bytes. File content remains recoverable from the pinned AV and retained archive.

Execution command (remote CPU only):

    PYTHONDONTWRITEBYTECODE=1 /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
      /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f01/lossless-release-v1/operation.py

```python
"""One-off, fixed F01 representation change; all data remain reconstructable."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
TRIAL = ROOT / "analysis/fresh-purpose-20261107-f01"
ARCHIVE = TRIAL / "lossless-archive-v1"
OUT = TRIAL / "lossless-release-v1"
CANDIDATE = ROOT / "checkpoints/fresh-code-joint-f01"
BASE = ROOT / "checkpoints/code90-current-ad-helper-av-v1"
NAMES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
EXPECTED_SUMS = "d259cf52217d2aae657fffd72d8d41c24cff2f851e40481cf5001c87e76c4a63"
EXPECTED_ARCHIVE = "fb228636f67f4287151b20332e06b34def82f9d0a6f9af1b07c9ea62913c1409"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def stamp(path):
    s = path.lstat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                mode=s.st_mode, uid=s.st_uid, gid=s.st_gid, nlink=s.st_nlink)

def save(path, value, exclusive=False):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + ".new")
        with temporary.open("x") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

def mounts():
    ids = subprocess.check_output(["docker", "ps", "-aq"], text=True).split()
    containers = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True)) if ids else []
    evidence = []
    for container in containers:
        selected = []
        for mount in container.get("Mounts", []):
            source = mount.get("Source", "")
            resolved = Path(source).resolve() if source.startswith("/") else None
            conflicts = resolved is not None and (
                resolved == CANDIDATE or resolved.is_relative_to(CANDIDATE)
                or CANDIDATE.is_relative_to(resolved))
            selected.append(dict(type=mount.get("Type"), source=source,
                                 destination=mount.get("Destination"),
                                 rw=mount.get("RW"), conflicts=conflicts))
            assert not conflicts, "F01 reachable through a container mount"
        evidence.append(dict(id=container["Id"], name=container["Name"],
                             state=container["State"]["Status"], mounts=selected))
    return evidence

def main():
    assert os.getuid() == 1003
    assert sha(ARCHIVE / "SHA256SUMS") == EXPECTED_SUMS
    assert sha(ARCHIVE / "archive.json") == EXPECTED_ARCHIVE
    rows = []
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ARCHIVE / relative
        assert path.resolve().is_relative_to(ARCHIVE) and not path.is_symlink()
        assert sha(path) == expected
        rows.append(dict(path=relative, sha256=expected, matches=True))
    actual = {str(p.relative_to(ARCHIVE)) for p in ARCHIVE.rglob("*") if p.is_file()}
    assert actual == {x["path"] for x in rows} | {"SHA256SUMS"}
    archive = json.loads((ARCHIVE / "archive.json").read_text())
    assert archive["mandatory_base_path"] == str(BASE)
    assert archive["candidate_path"] == str(CANDIDATE)
    assert json.loads((ARCHIVE / "archive-validation.json").read_text())["status"] == "passed"
    checkpoint = archive["checkpoint_file_stats"]
    assert len(checkpoint) == 9
    assert CANDIDATE.resolve() == CANDIDATE and not CANDIDATE.is_symlink()
    assert BASE.resolve() == BASE and not BASE.is_symlink()
    assert CANDIDATE.stat().st_uid == BASE.stat().st_uid == os.getuid()
    expected_files = {row["path"] for row in checkpoint}
    assert {str(p.relative_to(CANDIDATE)) for p in CANDIDATE.rglob("*") if p.is_file()} == expected_files
    assert not any(p.is_symlink() for p in CANDIDATE.rglob("*"))
    for row in checkpoint:
        assert stamp(BASE / row["path"]) == row["source_stat_at_archive"]
        assert stamp(CANDIDATE / row["path"]) == row["candidate_stat_at_archive"]
    selected = [next(row for row in checkpoint if row["path"] == name) for name in NAMES]
    for row in selected:
        s = stamp(CANDIDATE / row["path"])
        assert stat.S_ISREG(s["mode"]) and s["nlink"] == 1 and s["uid"] == os.getuid()
        assert s["size"] == row["size"]
    assert {row["path"] for row in selected} == set(NAMES)
    assert not (CANDIDATE / "ARCHIVED.json").exists()
    before_mounts = mounts()
    save(OUT / "container-mounts-before.json", before_mounts, exclusive=True)
    receipt = dict(schema_version=1, status="prechecks_passed", started_epoch=time.time(),
                   operation_source_sha256=sha(Path(__file__)), uid=os.getuid(),
                   archive_json_sha256=EXPECTED_ARCHIVE, archive_sha256sums_sha256=EXPECTED_SUMS,
                   all_archive_files_sha_verified=rows, base_files_exact_stat_continuity=9,
                   candidate_files_exact_stat_continuity=9, exact_regular_owned_single_link_targets=True,
                   container_count=len(before_mounts), all_container_mounts_no_overlap=True,
                   container_mounts_sha256=sha(OUT / "container-mounts-before.json"),
                   free_bytes_before=shutil.disk_usage(CANDIDATE).free,
                   original_weight_hash_evidence="Historical full verify, not fresh payload hashes.",
                   physical_reconstruction_performed=False, fresh_weight_hash_performed=False,
                   released=[], targets=[dict(path=str(CANDIDATE/row["path"]),size=row["size"],
                                             sha256=row["candidate_sha256"]) for row in selected])
    save(OUT / "prechecks.json", receipt, exclusive=True)
    marker = dict(schema_version=1, kind="lossless_dependent_archive",
                  state="representation_change_in_progress", not_a_saved_model=True,
                  mandatory_base=str(BASE), archive=str(ARCHIVE), receipt=str(OUT/"receipt.json"),
                  candidate_file_hashes=archive["files"],
                  reconstruction_instructions=str(ARCHIVE/"README.md"))
    # A visible marker is created before any unlink, so partial failure cannot
    # present the directory as a complete loadable model.
    save(CANDIDATE / "ARCHIVED.json", marker, exclusive=True)
    directory_fd = os.open(CANDIDATE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        mounts()  # Recheck immediately before the exact two file operations.
        for row in checkpoint:
            assert stamp(BASE / row["path"]) == row["source_stat_at_archive"]
            assert stamp(CANDIDATE / row["path"]) == row["candidate_stat_at_archive"]
        for row in selected:
            name = row["path"]
            s = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            expected = row["candidate_stat_at_archive"]
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_uid) == (
                expected["device"],expected["inode"],expected["size"],expected["mtime_ns"],
                expected["ctime_ns"],1,os.getuid())
            assert stat.S_ISREG(s.st_mode)
            os.unlink(name, dir_fd=directory_fd)
            receipt["released"].append(dict(path=str(CANDIDATE/name),size=row["size"],
                                             sha256=row["candidate_sha256"]))
            os.fsync(directory_fd)
        for row in checkpoint:
            assert stamp(BASE / row["path"]) == row["source_stat_at_archive"]
            if row["path"] not in NAMES:
                assert stamp(CANDIDATE / row["path"]) == row["candidate_stat_at_archive"]
        assert all(not (CANDIDATE/name).exists() for name in NAMES)
        assert {p.name for p in CANDIDATE.iterdir()} == (expected_files - set(NAMES)) | {"ARCHIVED.json"}
        marker["state"] = "archived_losslessly"
        save(CANDIDATE / "ARCHIVED.json", marker)
        os.fsync(directory_fd)
        receipt["status"] = "complete_two_private_weight_files_represented_by_archive"
    except BaseException as exc:
        receipt["status"] = "partial_or_failed_inspect_released_list"
        receipt["error_type"] = type(exc).__name__
        raise
    finally:
        os.close(directory_fd)
        receipt["free_bytes_after"] = shutil.disk_usage(CANDIDATE).free
        receipt["free_bytes_delta"] = receipt["free_bytes_after"] - receipt["free_bytes_before"]
        receipt["released_logical_bytes"] = sum(row["size"] for row in receipt["released"])
        receipt["finished_epoch"] = time.time()
        receipt["elapsed_seconds"] = receipt["finished_epoch"] - receipt["started_epoch"]
        receipt["tombstone_sha256"] = sha(CANDIDATE / "ARCHIVED.json")
        save(OUT / "receipt.json", receipt, exclusive=True)
    print(json.dumps({k:receipt[k] for k in ("status","free_bytes_before","free_bytes_after",
                     "free_bytes_delta","released_logical_bytes","elapsed_seconds")}))

if __name__ == "__main__":
    main()
```

The result will be appended after execution. No other checkpoint or cache is a target.

## Attempt 1: stopped before mutation

The first execution reached the container-mount guard and stopped before creating any tombstone or unlinking any file. Between the earlier read-only preflight and execution, the F02 finite container started with the entire experiment root mounted read-only. That ancestor bind makes F01 reachable even though F02 does not select it as the evaluated checkpoint. The guard deliberately rejected the overlap.

Receipt: analysis/fresh-purpose-20261107-f01/lossless-release-v1/precheck-stop.json. Operation source SHA256: 1083ee3b78ee4753af36436a56b5d85f3b89300b84838fa712f3c61999e155ef. Both private F01 weight files remained; no ARCHIVED.json existed. The observed conflict was a running F02 finite container, not a service stopped by this audit. The action may be retried only after the owner completes that finite run and the conflicting mount disappears, with all prechecks run again.

## Attempt 2: completed after F02 owner closure

The F02 finite owner completion receipt reported finished, returncode 0, restored=true and protected_exact=true; the probe container was absent. Re-running the unchanged operation passed every guard and completed. All 110 archive file SHA checks passed, all nine AV and all nine F01 files matched archive-time full stat before action, the exact two targets were owned ordinary single-link files, and all 28 containers had no equal, descendant or ancestor mount overlap. The second mount check immediately before unlink also passed. No service or container was stopped by this operation.

Free bytes before: 75008163840. Free bytes after: 137554505728. Observed increase: 62546341888 bytes. Exact released logical bytes: 62546338264. The small difference reflects filesystem allocation granularity and concurrent activity; the observed free-space delta is not used as a file-content proof. Post-precheck operation time: 8.998593091964722 seconds.

| Exact released path | Logical bytes | Historical fully verified candidate SHA256 |
|---|---:|---|
| /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/checkpoints/fresh-code-joint-f01/model-00001-of-00002.safetensors | 49923154850 | 747de4b2b690649c9ad08b20e63a2c5e9cce97fe4d9ba9e1f903e821a313d085 |
| /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/checkpoints/fresh-code-joint-f01/model-00002-of-00002.safetensors | 12623183414 | 5278835d2db3fae2819bbd67bf4af4640cf9be60c0a4b7247759fb5190853d5e |

Only these two directory entries were unlinked. The seven nonweight files and ARCHIVED.json remain. The marker states archived_losslessly and not_a_saved_model=true, names the mandatory AV base and archive, and points to the reconstruction instructions. All original spec/manifest/verification and finite trial files remain.

A separate post-verification read all 110 archive files again, confirmed every SHA, confirmed AV nine-file and F01 seven-file stat continuity, confirmed both target paths absent, and checked the marker. It freshly hashed the retained original spec, manifest and verify receipts and found their original values. It did not freshly hash model weight payloads or physically reconstruct F01.

Evidence under analysis/fresh-purpose-20261107-f01/lossless-release-v1: precheck-stop.json (retained failed attempt), container-mounts-before.json, prechecks.json, receipt.json and post-verification.json.

Operation SHA256: 1083ee3b78ee4753af36436a56b5d85f3b89300b84838fa712f3c61999e155ef. Receipt SHA256: 11f709b88e09cadf632cba6f96f67f1a7c00552cbf830489d33a9850f509baf6. Tombstone SHA256: 721e7593e556e01ad81c42b63cc4f84d28c1a6cab000d9cbe7aa5c5261e3cff6. Archive checksum-list SHA256 remains d259cf52217d2aae657fffd72d8d41c24cff2f851e40481cf5001c87e76c4a63.
