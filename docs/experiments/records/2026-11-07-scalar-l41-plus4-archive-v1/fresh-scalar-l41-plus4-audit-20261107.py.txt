from pathlib import Path
import hashlib
import json
import stat
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
C = R / "checkpoints/scalar-l41-plus4"
D = R / "analysis/fresh-purpose-20261107-scalar-l41-plus4-release-plan-v1"
D.mkdir()
sys.path.insert(0, str(B / "src"))
from kairyu_bench.wsp.weights import editor  # noqa: E402 - fixed repository src path above

manifest = R / "manifests/scalar-l41-plus4.json"
spec = R / "specs/scalar-l41-plus4.json"
m = json.loads(manifest.read_text())
base = Path(m["source"])
assert json.loads(spec.read_text()) == m["spec"] and m["candidate"] == str(C)
validation = editor.validate_manifest(m)
expected = editor.plan_edits(base, m["spec"])
fields = (
    "tensor",
    "shard",
    "offset",
    "size",
    "dtype",
    "shape",
    "before_hex",
    "after_hex",
    "selected_indices",
    "changed_indices",
    "changed_count",
)
assert len(expected) == len(m["edits"]) == 1 and all(
    a[k] == b[k] for a, b in zip(expected, m["edits"]) for k in fields
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp(p):
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode)
    return dict(
        device=s.st_dev,
        inode=s.st_ino,
        size=s.st_size,
        mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,
        mode=s.st_mode,
        uid=s.st_uid,
        gid=s.st_gid,
        nlink=s.st_nlink,
    )


names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
targets = {n: stamp(C / n) for n in names}
assert all(s["nlink"] == 1 for s in targets.values())
for p in m["edits"]:
    with (C / p["shard"]).open("rb") as f:
        f.seek(p["offset"])
        assert f.read(p["size"]).hex() == p["after_hex"]
source_stats = {r["path"]: stamp(base / r["path"]) for r in m["files"]}
candidate_stats = {r["path"]: stamp(C / r["path"]) for r in m["files"]}
dependent = []
for p in (R / "manifests").glob("*.json"):
    try:
        v = json.loads(p.read_text())
    except (ValueError, OSError):
        continue
    if v.get("source") == str(C):
        dependent.append(str(p))
archives = []
for p in (R / "analysis").glob("*/lossless-archive*/archive.json"):
    v = json.loads(p.read_text())
    if v.get("mandatory_base_path") == str(C):
        archives.append(str(p))
ids = subprocess.run(
    ["docker", "ps", "-aq", "--no-trunc"], capture_output=True, text=True, check=True
).stdout.split()
containers = json.loads(
    subprocess.run(
        ["docker", "inspect", *ids], capture_output=True, text=True, check=True
    ).stdout
)
mounts = []
for d in containers:
    for x in d["Mounts"]:
        path = Path(x["Source"]).resolve() if x.get("Type") == "bind" else None
        if path and (path == C or path.is_relative_to(C) or C.is_relative_to(path)):
            mounts.append(
                dict(cid=d["Id"], name=d["Name"], source=str(path), rw=x["RW"])
            )
hits = []
unreadable = []
observed = 0
inode = {(s["device"], s["inode"]) for s in targets.values()}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        uid = proc.stat().st_uid
        observed += 1
        for fd in (proc / "fd").iterdir():
            try:
                s = fd.stat()
                if (s.st_dev, s.st_ino) in inode:
                    hits.append(dict(pid=int(proc.name), kind="fd", descriptor=fd.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        for line in (proc / "maps").read_text().splitlines():
            if str(C) in line:
                hits.append(dict(pid=int(proc.name), kind="map"))
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        unreadable.append(
            dict(pid=int(proc.name), uid=uid if "uid" in locals() else None)
        )
archive = R / "analysis/fresh-purpose-20261107-scalar-l41-plus4-lossless-v1"
pins = {
    str(p): sha(p)
    for p in (
        manifest,
        spec,
        B / "src/kairyu_bench/wsp/weights/editor.py",
        B / "src/kairyu_bench/wsp/weights/checkpoint.py",
        B / "docs/experiments/scripts/fresh-f02-payload-release-20261107.py",
    )
}
audit = dict(
    epoch=time.time(),
    target=str(C),
    base=str(base),
    manifest_fields_verified=True,
    selected_bytes_verified=True,
    legacy_full_hashes_missing=True,
    new_full_hash_performed=False,
    validation=validation,
    source_stats=source_stats,
    candidate_stats=candidate_stats,
    direct_manifest_dependents=dependent,
    known_lossless_archive_dependents=archives,
    container_count=len(containers),
    overlapping_mounts=mounts,
    process_references=dict(
        pids_observed=observed,
        matches=hits,
        inaccessible_or_disappeared=unreadable,
        unknown_is_not_absence=True,
    ),
    releaseable_if_all_gates_pass=sum(s["size"] for s in targets.values()),
    source_pins=pins,
)
audit["candidate_metadata_directories"] = {
    ".eval_results": dict(
        mode=(C / ".eval_results").lstat().st_mode,
        mtime_ns=(C / ".eval_results").lstat().st_mtime_ns,
        inode=(C / ".eval_results").lstat().st_ino,
    )
}
audit["complete_original_inventory"] = {
    str(p.relative_to(base)): stamp(p) for p in base.rglob("*") if p.is_file()
}
audit["complete_candidate_inventory"] = {
    str(p.relative_to(C)): stamp(p) for p in C.rglob("*") if p.is_file()
}
assert audit["complete_candidate_inventory"] == candidate_stats
audit["non_destructive_archive_only_authorized"] = True
audit["release_authorized"] = False
(D / "read-only-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
plan = dict(
    status="unapproved_archive_then_release_plan",
    approved_for_execution=False,
    candidate=str(C),
    mandatory_base=str(base),
    archive=str(archive),
    target_payloads=targets,
    source_pins=pins,
    audit=dict(
        path=str(D / "read-only-audit.json"), sha256=sha(D / "read-only-audit.json")
    ),
    legacy_manifest_limitation="No historical per-file hashes. Fresh existing editor.verify required before archive/release; preserve legacy_full_hashes_missing=true.",
    verify_command=[
        str(R / "infra/weight-tools-ci/bin/python"),
        "-m",
        "kairyu_bench.wsp.weights.editor",
        "verify",
        str(manifest),
    ],
    verify_environment=dict(
        PYTHONPATH=str(B / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    ),
    verification="All12 regular files; fresh baseSHA and candidateSHA plus candidate with one declared BF16 span restored must equal baseSHA. Existing _stream_hashes computes current/restored hashes in one candidate read. No 62GB physical duplicate required.",
    archive_scope="Original spec/manifest bytes, fresh full verify, all10 nonweight files including metadata/docs, current source package, source/candidate fullSHA lists and stats, exact one-byte-pair patch, original experiment/provenance references.",
    reconstruction="Archived existing editor create originalbase into fresh directory using exact spec; verify every reconstructed candidate file SHA. Archive depends on preserved original, not AV.",
    release_gates=[
        "root exactplan approval",
        "global evaluation flock",
        "fresh full verification success/source and candidate stat continuity",
        "archive checksum/inflate validation",
        "zero visible process/mount refs; inaccessible PIDs remain explicit root decision",
        "both payloads same exact private nlink1 inode/size",
        "original/AV/faststore untouched",
    ],
    durable_release="Reuse existing F02 durable_json/nofollow dirfd/per-entry intent+unlink+released journal pattern with explicit scalar2 paths; current fixed F02 wrapper cannot be used unchanged because it assumes9 files/135archive entries.",
    after_release="Keep all10 candidate nonweight files, spec, manifest, benchmark failures/results and ARCHIVED.json. Check available bytes and unchanged protected roots. No model/API/service calls.",
)
(D / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(
    json.dumps(
        dict(
            plan=str(D / "plan.json"),
            sha256=sha(D / "plan.json"),
            archive=str(archive),
            payload_bytes=plan["audit"] and audit["releaseable_if_all_gates_pass"],
            manifest_dependents=dependent,
            archive_dependents=archives,
            mounts=mounts,
            visible_process_hits=hits,
            inaccessible_count=len(unreadable),
            base=str(base),
            sourcefilecount=len(source_stats),
            metadata_only=True,
        )
    )
)
