# F02 lossless dependent archive preparation — 2026-11-07

Status: **prepared and independently validated; F02 payloads unchanged**.
This operation does not adopt F02 or authorize any later payload release.
F02's complete canonical dev10 code score remains 87.777778, tied with AV;
the frozen advancement rule leaves AV as the parent. All failed attempts,
finite results, official results and observation records remain at their
existing paths. The active AV dev30 owner is independent of this CPU work.

## Outcome and disk scope

The new archive is
analysis/fresh-purpose-20261107-f02/lossless-archive-v1 under the experiment
root. It occupies 73,605,903 logical bytes and contains 135 checksum-listed
files plus SHA256SUMS. Creation and validation took 12.345028 seconds; exact
selected-coordinate re-planning took 5.809419 seconds within that total.
A separate checksum/stat/gzip/CLI/recipe-syntax audit passed in 1.114557 seconds.
Archive failures: zero. Model calls, GPU contexts, service operations and lock
operations: zero. No checkpoint file was deleted, moved, rewritten or linked.

Free NVMe bytes observed at operation start/end were 70,872,813,568 and
70,798,675,968. A simultaneous benchmark can write its own results, so the
free-space difference is not attributed entirely to this archive. Space freed
by this operation is zero. F02 remains a complete loadable nine-file checkpoint.

The two existing private F02 shards total 62,546,338,264 bytes, and all nine
checkpoint files total 62,578,656,419 bytes. Those files remain untouched.
The root may later consider a separate exact two-file representation change
only if additional space is needed and fresh checks/authorization permit it.
This preparation grants no such authority and performs no container-mount
cleanup or service action.

## Mandatory dependency and reconstruction proof

The archive requires the immutable AV base:
checkpoints/code90-current-ad-helper-av-v1, tree SHA256
5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf.
The F02 tree is
46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c.

The existing editor.verify implementation at editor.py:513–585 hashes each
whole source file, hashes each whole candidate file, and simultaneously
hashes the candidate with declared after_hex spans replaced by before_hex.
It rejects undeclared byte differences, wrong original/changed bytes, spec/
header/shape/offset disagreement, inventory/size disagreement and shared edited
inodes. F02's saved verified=true, legacy_full_hashes_missing=false result
therefore proves the complement equals AV and the declared spans equal the
stored BF16 edits.

The subsequent complete AV/F02 diff freshly hashed all 18 files, matched both
trees and all saved file hashes, compared all 1,188 stored tensors and found
exactly 203,419 changed elements across 410 matrices. Delta L2 is
1.1089462630318312. Its complete report SHA is
cdc3ea38826640a25c680f05aa4df0f5e1e128e6e2c8c0ec39e1838607007ec1,
and independently closed evidence SHA is
5fd3a36c5087e89eb3fe4c7092f688747cc014fe63020ab67b224cd6d4f8a99a.

Archive preparation does **not** repeat that whole payload hash. It checks all
nine current AV and all nine current F02 device/inode/size/mtime/ctime/mode/
uid/gid/nlink records against the completed full-diff closure, before and after
archival. It validates the full manifest with the archived existing editor
and replans all 203,419 spans from current AV coordinates. Every tensor, shard,
offset, size, dtype, shape, before_hex, after_hex, selected/changed index list
and changed count matches. This selected-coordinate reading is not a new
full model scan or a physical reconstruction.

The exact complete content bytes and relative filenames are reconstructable
from AV plus the archived fixed spec/manifest. Original filesystem inode,
ctime and device values are retained provenance; they are not promised as the
metadata of a newly reconstructed directory. Physical full reconstruction was
not performed. A future restoration must pass the existing full editor verify
and all nine CANDIDATE-SHA256SUMS entries.

## Contents and identifiers

- Original-byte gzip copies of the full 15,883,547-byte spec and
  215,740,576-byte edit manifest. Gzip filename is empty, mtime is zero and
  compression level is six. Both streams were decompressed and byte-SHA
  checked twice, without reserializing JSON.
- All seven complete nonweight files, including tokenizer, template, config,
  generation config, processor config and safetensors index. Each copy has
  the original file's metadata and a fresh matching SHA; all seven original
  AV/F02 hashes are equal.
- Every materialization receipt/output except the manifest, which is retained
  as original-byte gzip; this includes the earlier failed precheck. Proposal
  report, finite completion/numerical/owner-plan records, the complete
  all-778/code-26 official comparison records, and full-diff report/closure/
  summary/plan/timing records are also copied. All original trial directories,
  including earlier finite failures, remain unchanged outside the archive.
- The existing 89-file frozen package source used by the fresh AV capture.
  Every weights source hash matches both F02's materialization code proof and
  the current existing implementation. No new archive package is introduced.
- Full nine-file AV and F02 SHA lists; original full metadata for both
  checkpoints; archive.json, archive-validation.json, whole archive
  SHA256SUMS and an executable reconstruction recipe in README.md.

| Artifact | SHA256 |
|---|---|
| Original spec | 27f000e4933bdb63d29217c6312114d2b912d349a67617fe42d17d263994410b |
| Original full edit manifest | 1cb562c4d9fbd1fe3bc6a2f2c489b5b482406a5d31ff086ca0f2b48a49d47880 |
| Saved full verify | 3d5b609ef17bc3a1ba1ad064af55e1beac36174e722a3e72339179f7a734a8c5 |
| archive.json | 7f2cb71ac847204bf1d2443f358a5b4fe5453afc519655cb6a77b5461e5b3769 |
| archive-validation.json | 6c64910c5531209e17dcdb12ef493c2b11a280a4262b292a8454467a96c3b3bf |
| Archive SHA256SUMS | dee04b50c984013410ce3aa8fac6891337774f9846c929a0ceadc3d881b626da |
| Independent audit | 414c1cb8eb7e72e33d71f98149ef776c29373d88fcf27bea95b672c65dafa7cc |

Recorded CPU Python: 3.12.13 (main, Jul 18 2026, 17:02:19) [Clang 22.1.3 ]. NumPy: 2.5.4.

The independent audit rechecked all 135 archive SHA entries and exact file
inventory, both gzip round trips and all 18 checkpoint stats. It imported the
archived editor's create --help (including --apply and --reserve-bytes) and
ran bash -n on the README's extracted reconstruction commands. It did not
execute those reconstruction commands.

## Commands and exact fixed one-off source

All operations ran through SSH in the NVMe worktree, with CUDA_VISIBLE_DEVICES
empty, PYTHONDONTWRITEBYTECODE=1, BLAS/OpenMP/MKL/NumExpr threads set to one and
TMPDIR under lossless-archive-preparation-v1/tmp. The creation process used
ionice -c 3, nice -n 15 and taskset -c 32-35. It did not acquire or change the
active benchmark lock.

The exact executed source below is preserved as a fixed experiment recipe,
not a new general archive tool. Its SHA256 is
959ba2d3d4f11c57f35a60b55498928995aeada15526be65450ce0871aaa2b2a.
The parent directory and its tmp subdirectory were created as new paths,
then this source was executed with the existing CPU Python. It uses the
existing editor's validate_manifest and plan_edits APIs.

~~~python
"""Prepare a fixed F02 dependent lossless archive; never alter checkpoint payloads."""
from pathlib import Path
import gzip, hashlib, importlib.metadata, json, math, os, shutil, stat, sys, time, traceback

R=Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B=R/"worktrees/kairyu-bench-weight-tools"
T=R/"analysis/fresh-purpose-20261107-f02"
O=T/"lossless-archive-preparation-v1"
A=T/"lossless-archive-v1"
AV=R/"checkpoints/code90-current-ad-helper-av-v1"
F02=R/"checkpoints/fresh-code-joint-f02"
C=R/"analysis/fresh-purpose-20261107-av-f02-comparison-prep-v1"
SOURCE=R/"runtime/fresh-purpose-20261107-av-live10-v1/observer-src"
EXPECTED={
 T/"proposal/fresh-code-joint-f02.json":"27f000e4933bdb63d29217c6312114d2b912d349a67617fe42d17d263994410b",
 T/"materialization/edit-manifest.json":"1cb562c4d9fbd1fe3bc6a2f2c489b5b482406a5d31ff086ca0f2b48a49d47880",
 T/"materialization/verify.json":"3d5b609ef17bc3a1ba1ad064af55e1beac36174e722a3e72339179f7a734a8c5",
 T/"materialization/full-verification-freeze-input.json":"f897d77bf5e4bf219f6c8391cbcb738c780912d1a666ff6ac5cb4ae150fa8715",
 C/"weights-full-diff.json":"cdc3ea38826640a25c680f05aa4df0f5e1e128e6e2c8c0ec39e1838607007ec1",
 C/"full-diff-closure.json":"5fd3a36c5087e89eb3fe4c7092f688747cc014fe63020ab67b224cd6d4f8a99a",
}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb")as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
def stamp(path):
    s=path.lstat()
    assert stat.S_ISREG(s.st_mode)and not path.is_symlink()
    return {k:getattr(s,"st_"+v)for k,v in [
      ("device","dev"),("inode","ino"),("size","size"),("mtime_ns","mtime_ns"),("ctime_ns","ctime_ns"),
      ("mode","mode"),("uid","uid"),("gid","gid"),("nlink","nlink")]}
def inventory(root):
    assert root.is_dir()and not root.is_symlink()
    assert not any(p.is_symlink()for p in root.rglob("*"))
    return {str(p.relative_to(root)):stamp(p)for p in sorted(root.rglob("*"))if p.is_file()}
def save(path,value):
    with path.open("x")as stream:
        json.dump(value,stream,indent=2,allow_nan=False,ensure_ascii=False);stream.write("\n")
def copy_checked(source,target,expected=None):
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists()
    before=stamp(source);digest=sha(source)
    if expected is not None:assert digest==expected
    shutil.copy2(source,target)
    assert sha(target)==digest and stamp(source)==before
    return dict(path=str(target.relative_to(A)),source_path=str(source),bytes=target.stat().st_size,sha256=digest)
def gzip_original(source,target):
    before=stamp(source);digest=sha(source)
    with source.open("rb")as inp,target.open("xb")as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0,compresslevel=6)as out:
            shutil.copyfileobj(inp,out,8*1024**2)
    decompressed=hashlib.sha256();count=0
    with gzip.open(target,"rb")as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):
            decompressed.update(block);count+=len(block)
    assert decompressed.hexdigest()==digest and count==before["size"] and stamp(source)==before
    return dict(path=str(target.relative_to(A)),source_path=str(source),original_bytes=count,
      original_sha256=digest,compressed_bytes=target.stat().st_size,compressed_sha256=sha(target),
      gzip_mtime=0,gzip_filename="",compression_level=6,decompression_verified=True)
started=time.time();stage="initial"
try:
    assert os.getuid()!=0 and not A.exists()
    assert all(sha(p)==digest for p,digest in EXPECTED.items())
    spec=json.loads((T/"proposal/fresh-code-joint-f02.json").read_text())
    manifest=json.loads((T/"materialization/edit-manifest.json").read_text())
    verified=json.loads((T/"materialization/verify.json").read_text())
    proof=json.loads((T/"materialization/full-verification-freeze-input.json").read_text())
    full=json.loads((C/"weights-full-diff.json").read_text())
    close=json.loads((C/"full-diff-closure.json").read_text())
    assert manifest["source"]==str(AV)and manifest["candidate"]==str(F02)
    assert manifest["spec"]==spec and manifest["spec_sha256"]==EXPECTED[T/"proposal/fresh-code-joint-f02.json"]
    assert verified["verified"]is True and verified["legacy_full_hashes_missing"]is False
    assert verified["changed_elements"]==203419 and verified["changed_tensors"]==410
    assert verified["delta_l2"]==1.1089462630318312
    assert close["status"]=="full_diff_validated"and close["hashed_files_verified"]==18
    assert full["base"]["sha256"]=="5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf"
    assert full["candidate"]["sha256"]=="46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c"
    base_stats,candidate_stats=inventory(AV),inventory(F02)
    assert base_stats==close["stat_checks"]["base"]["stat"]
    assert candidate_stats==close["stat_checks"]["candidate"]["stat"]
    files={row["path"]:row for row in manifest["files"]}
    assert len(files)==9 and set(base_stats)==set(candidate_stats)==set(files)
    verify_files={row["path"]:row for row in verified["files"]}
    for rel,row in files.items():
        assert row["source_sha256"]==verify_files[rel]["source_sha256"]
        assert row["candidate_sha256"]==verify_files[rel]["candidate_sha256"]
        for side,label in [("source","base"),("candidate","candidate")]:
            measured=next(x for x in full[label]["files"]if x["path"]==rel)
            assert measured["sha256"]==row[side+"_sha256"]and measured["size"]==row["size"]
    free_before=shutil.disk_usage(R).free
    A.mkdir();(A/"metadata").mkdir();(A/"records").mkdir()
    stage="byte_preserving_archive"
    compressed=[gzip_original(T/"proposal/fresh-code-joint-f02.json",A/"spec.json.gz"),
                gzip_original(T/"materialization/edit-manifest.json",A/"edit-manifest.json.gz")]
    metadata=[]
    for rel,row in files.items():
        if rel.endswith(".safetensors"):continue
        assert row["source_sha256"]==row["candidate_sha256"]
        metadata.append(copy_checked(F02/rel,A/"metadata"/rel,row["candidate_sha256"]))
    assert len(metadata)==7
    records=[]
    for p in sorted((T/"materialization").iterdir()):
        if p.is_file()and p.name!="edit-manifest.json":
            records.append(copy_checked(p,A/"records/materialization"/p.name))
    records.append(copy_checked(T/"proposal/report.json",A/"records/proposal-report.json"))
    for name in ["completion.json","numeric-analysis.json","owner-plan.json","finite-plan.json","v1-v2-comparison.json"]:
        records.append(copy_checked(T/"finite-gpu6-v2"/name,A/"records/finite-gpu6-v2"/name))
    for name in ["comparison.json","comparison.md","summary.json","code-26-paired-items.json","all-778-paired-items.json"]:
        records.append(copy_checked(T/"official-comparison-v3"/name,A/"records/official-comparison-v3"/name))
    for name in ["plan.json","execution.json","weights-diff-time.txt","weights-full-diff.json","full-diff-closure.json","full-weight-summary.json"]:
        records.append(copy_checked(C/name,A/"records/full-diff"/name))
    shutil.copytree(SOURCE,A/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    source_files={str(p.relative_to(A/"source")):sha(p)for p in sorted((A/"source").rglob("*"))if p.is_file()}
    assert len(source_files)==89
    for relative,digest in proof["code_sha256"].items():
        assert relative.startswith("src/")
        archived=A/"source"/relative.removeprefix("src/")
        assert sha(archived)==digest==sha(B/relative)
    assert (A/"source/kairyu_bench/wsp/weights/editor.py").is_file()
    stage="existing_editor_exact_replan"
    sys.path.insert(0,str(A/"source"))
    from kairyu_bench.wsp.weights.editor import validate_manifest,plan_edits
    t=time.time();validation=validate_manifest(manifest)
    patches=plan_edits(AV,spec)
    fields=("tensor","shard","offset","size","dtype","shape","before_hex","after_hex",
      "selected_indices","changed_indices","changed_count")
    assert len(patches)==len(manifest["edits"])==203419
    assert all(all(a[k]==b[k]for k in fields)for a,b in zip(patches,manifest["edits"],strict=True))
    assert math.hypot(*(x["delta_l2"]for x in patches))==verified["delta_l2"]
    validation_seconds=time.time()-t
    assert inventory(AV)==base_stats and inventory(F02)==candidate_stats
    archive=dict(schema_version=1,status="dependent_lossless_archive_prepared_payloads_unchanged",
      created_epoch=time.time(),mandatory_base_path=str(AV),mandatory_base_tree_sha256=full["base"]["sha256"],
      candidate_path=str(F02),candidate_tree_sha256=full["candidate"]["sha256"],
      files=[dict(path=rel,size=row["size"],source_sha256=row["source_sha256"],candidate_sha256=row["candidate_sha256"])for rel,row in sorted(files.items())],
      checkpoint_file_stats=[dict(path=rel,size=files[rel]["size"],source_stat_at_archive=base_stats[rel],candidate_stat_at_archive=candidate_stats[rel])for rel in sorted(files)],
      original_spec_sha256=manifest["spec_sha256"],original_manifest_sha256=EXPECTED[T/"materialization/edit-manifest.json"],
      full_saved_verify_sha256=EXPECTED[T/"materialization/verify.json"],
      fresh_completed_full_diff_sha256=EXPECTED[C/"weights-full-diff.json"],
      fresh_completed_full_diff_closure_sha256=EXPECTED[C/"full-diff-closure.json"],
      hash_provenance="Prior saved full verify plus the completed18-file full diff, then exact current full-stat continuity. No checkpoint-payload fullhash during archive preparation.",
      compressed_originals=compressed,metadata=metadata,records=records,source_files_sha256=source_files,
      python_version=sys.version,numpy_version=importlib.metadata.version("numpy"),
      operation_source_sha256=sha(Path(__file__)),physical_reconstruction_performed=False,
      checkpoint_payload_fullhash_performed_here=False,checkpoint_payloads_modified=False,payload_release_authorized=False,
      existing_full_trials_and_bench_results_retained=True,candidate_adoption=False,
      limitations=["Mandatory immutable AV dependency; archive alone is not a standalone checkpoint.",
        "Original inode/ctime/dev values are retained provenance, not reconstructed filesystem identities.",
        "Future reconstruction must verify archive and AV full hashes, create into new paths, run editor verify and compare all9 candidate hashes.",
        "No service, lock or GPU operation; no new capability or adoption claim."])
    save(A/"archive.json",archive)
    save(A/"archive-validation.json",dict(status="passed",validated_epoch=time.time(),
      existing_editor_summary=validation,exact_replanned_spans=len(patches),exact_replan_fields=list(fields),
      replan_seconds=validation_seconds,source_files_pinned=len(source_files),
      base_current_stat_matches_completed_full_diff=True,candidate_current_stat_matches_completed_full_diff=True,
      gzip_original_bytes_verified=2,metadata_file_hashes_verified=7,
      original_full_verify_passed=True,completed_full_diff_18_hashes_matched=True,
      physical_full_reconstruction_performed=False,checkpoint_payload_fullhash_performed_here=False,
      payloads_unchanged=True))
    (A/"BASE-SHA256SUMS").write_text("".join(row["source_sha256"]+"  "+rel+"\n"for rel,row in sorted(files.items())))
    (A/"CANDIDATE-SHA256SUMS").write_text("".join(row["candidate_sha256"]+"  "+rel+"\n"for rel,row in sorted(files.items())))
    readme="""# F02 dependent lossless archive (preparation only)

AV is the mandatory immutable base. This directory is not a loadable model.
F02 remains a complete unchanged checkpoint at its existing location. No payload
release, candidate re-adoption, new benchmark or full physical reconstruction
was performed while creating this archive. All original trial/results paths
remain. The full manifest and spec are compressed original bytes, not reserialized.

The saved editor verify proves that replacing the manifest's after_hex spans by
before_hex restores the complete AV file hash: all other bytes equal AV.
The later full diff freshly hashed all18 base/candidate files and independently
measured every stored element. Archive preparation checks current full stats
against that closure, validates the archived manifest and replans exact spans
from current AV coordinates; it does not rerun full payload hashing.

Restoring file contents requires the pinned AV and this archive. New inode,
ctime and directory timestamps will differ; the target is the nine relative
filenames, sizes and complete content hashes. Seven nonweight file copies and
their original metadata are retained separately. All seven equal the pinned AV.

## Future reconstruction into NEW paths (not run during archive preparation)

All work/cache/temp paths must remain on NVMe. Preserve the original archive,
spec, manifest, verification, diff and trial results. Reserve sufficient room
for a private62.58GB checkpoint plus the chosen reserve; no deletion is implicit.

    R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
    F02_ARCHIVE="$R/analysis/fresh-purpose-20261107-f02/lossless-archive-v1"
    F02_BASE="$R/checkpoints/code90-current-ad-helper-av-v1"
    F02_RESTORE="$R/checkpoints/fresh-code-joint-f02-reconstructed-v1"
    F02_META="$R/analysis/fresh-purpose-20261107-f02/reconstruction-v1"
    F02_PYTHON="$R/infra/weight-tools-ci/bin/python"
    test ! -e "$F02_RESTORE"
    test ! -e "$F02_META"
    (cd "$F02_ARCHIVE" && sha256sum -c SHA256SUMS)
    (cd "$F02_BASE" && sha256sum -c "$F02_ARCHIVE/BASE-SHA256SUMS")
    mkdir "$F02_META"
    mkdir "$F02_META/tmp"
    export TMPDIR="$F02_META/tmp" PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    gzip -dc "$F02_ARCHIVE/spec.json.gz" > "$F02_META/spec.json"
    gzip -dc "$F02_ARCHIVE/edit-manifest.json.gz" > "$F02_META/original-edit-manifest.json"
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor create \
      "$F02_BASE" "$F02_RESTORE" --spec "$F02_META/spec.json" \
      --manifest "$F02_META/edit-manifest.json" --reserve-bytes 53687091200 > "$F02_META/preview.json"
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor create \
      "$F02_BASE" "$F02_RESTORE" --spec "$F02_META/spec.json" \
      --manifest "$F02_META/edit-manifest.json" --reserve-bytes 53687091200 --apply > "$F02_META/create.json"
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor verify \
      "$F02_META/edit-manifest.json" > "$F02_META/verify.json"
    (cd "$F02_RESTORE" && sha256sum -c "$F02_ARCHIVE/CANDIDATE-SHA256SUMS")

Before the create step, compare the decompressed spec/manifest SHA256 to
archive.json.original_spec_sha256 and original_manifest_sha256. The new
creation manifest contains new paths and timestamps, so its JSON hash need not
equal the original; the nine recreated checkpoint file hashes must all match.
Check verify.json.verified is true and legacy_full_hashes_missing is false.
Use the archived source and recorded compatible Python/NumPy environment.
"""
    (A/"README.md").write_text(readme)
    stage="whole_archive_checksum_validation"
    sums={str(p.relative_to(A)):sha(p)for p in sorted(A.rglob("*"))if p.is_file()}
    (A/"SHA256SUMS").write_text("".join(digest+"  "+name+"\n"for name,digest in sorted(sums.items())))
    assert set(sums)|{"SHA256SUMS"}=={str(p.relative_to(A))for p in A.rglob("*")if p.is_file()}
    assert all(sha(A/name)==digest for name,digest in sums.items())
    assert inventory(AV)==base_stats and inventory(F02)==candidate_stats
    assert all(sha(p)==digest for p,digest in EXPECTED.items())
    archive_bytes=sum(p.stat().st_size for p in A.rglob("*")if p.is_file())
    save(O/"receipt.json",dict(status="prepared_and_validated_no_payload_change",started_epoch=started,
      finished_epoch=time.time(),elapsed_seconds=time.time()-started,archive_path=str(A),
      archive_json_sha256=sha(A/"archive.json"),archive_validation_sha256=sha(A/"archive-validation.json"),
      archive_sha256sums_sha256=sha(A/"SHA256SUMS"),archive_files_excluding_checksum=len(sums),
      archive_bytes=archive_bytes,checkpoint_payloads_unchanged=True,base_stat_files=9,candidate_stat_files=9,
      exact_replanned_spans=203419,selected_coordinate_replan_seconds=validation_seconds,
      fresh_checkpoint_fullhash_performed=False,physical_reconstruction_performed=False,
      free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(R).free,
      root_gpus_used=0,model_calls=0,service_mutations=0,lock_mutations=0,payload_release_authorized=False,
      operation_source_sha256=sha(Path(__file__))))
    print((O/"receipt.json").read_text(),flush=True)
except BaseException as error:
    p=O/"failure.json"
    if not p.exists():save(p,dict(status="failed",stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc(),
      started_epoch=started,finished_epoch=time.time(),elapsed_seconds=time.time()-started,
      checkpoint_payloads_modified=False,model_calls=0,service_mutations=0,lock_mutations=0,
      operation_source_sha256=sha(Path(__file__))))
    raise
~~~


## Future restoration and possible later space decision

The full reconstruction recipe is retained verbatim as archive-README.md in
the PR record copies. Verify the archive checksum list and all nine mandatory
AV files, decompress original spec/manifest and check their original SHAs, use
the archived existing editor to preview/create into new paths, fully verify,
then compare all nine reconstructed hashes. The new creation manifest's own
JSON hash may differ because paths/timestamps differ; the checkpoint bytes
must match exactly.

The archive is not a standalone safetensors checkpoint. AV must remain
available with its pinned content. No later deletion, cache prune, model
replacement or deployment follows implicitly from this successful preparation.
Any future two-payload release needs a separate current owner decision and
fresh archive/stat/ownership/mount checks; the archived preparation-time
metadata is not a future authorization.

Small identifying records and checksum inventories are included under
records/2026-11-07-f02-lossless-archive/. ARCHIVE-SHA256SUMS refers to the
complete NVMe archive (including gzip, nonweight files and source); that
payload is retained there. The record directory's own SHA256SUMS covers only
the PR metadata copies. No large model shard or secret is added to the PR.
