# F02 private payload release: read-only plan

**Prepared only; approval=false, released0 bytes.** F02 remains a complete
checkpoint. No delete, rename, candidate creation, service change or lock
operation occurred. [Plan](records/2026-11-07-f02-payload-release-plan/v2/plan.json)
limits any later representation change to the two private shard entries below.
AV, original model, F01 and every other candidate/result are outside its scope.

All135 existing archive checksums and exact inventory passed; both compressed
spec/manifest streams round-tripped to their original full byte hashes. Saved
full verify and the subsequent18-file full-hash diff agree, and all9 AV plus all9
F02 device/inode/size/mtime/ctime/mode/uid/gid/nlink values still match those proofs.
The archive's all203,419 exact-span reconstruction proof and its frozen editor
remain intact. **No new payload fullhash, selected-span replan or physical
reconstruction was performed.** Exact nine-file contents are recoverable from
the pinned immutable AV plus the lossless archive; original inode/ctime values
are provenance, not recreated filesystem identities.

The current two shards are ordinary UID/GID1003 files with nlink1, distinct from
AV inodes. All27 running/stopped container mounts exclude F02 itself, descendants
and ancestors; no service/container was modified. Visible same-UID FD/maps checks
found no target references. Some processes were inaccessible or disappeared,
and other-UID processes were not inspected; the capacity numbers remain estimates
until an authorized operation records actual free bytes. Checkpoint writes0.

Candidate directory:
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/checkpoints/fresh-code-joint-f02

| Only proposed target | Logical bytes | Historical full candidate SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 49923154850 | e6144c1c2322cd2a05ced629c754c229ec530747470b3ba423a3c905b8d6e97c |
| model-00002-of-00002.safetensors | 12623183414 | f472653dfcfe3218f33111b44e936e097a8ba01014b9b5376f33c3196980181b |

## Capacity

| Quantity | Bytes | GiB |
|---|---:|---:|
| Current free | 71037714432 | 66.159 |
| Two private payloads, logical | 62546338264 | 58.251 |
| Current allocated blocks | 62546374656 | 58.251 |
| Projected free after release | 133584089088 | 124.410 |
| Worst-case new full checkpoint | 62578656419 | 58.281 |
| Current free minus one new checkpoint | 8459058013 | 7.878 |
| Projected free after release and new checkpoint | 71005432669 | 66.129 |

The existing editor default reserve is0 and it additionally checks64MiB:
current space can hold the bare full copy, but leaves only about
7.88GiB before
further results/cache. An explicit50GiB reserve does not fit now; it does fit the
projected post-release capacity. This50GiB is an example chosen reserve, **not**
the editor default. ext4 allocation/free accounting and concurrent writes mean
observed free-space delta need not equal logical bytes. No cache prune is planned.

## Later operation, only if root authorizes

Acquire the existing global evaluation flock and rerun the plan's fresh guards:
archive135 hashes/inventory, all18 full stats, exact owned regular single-link
targets, no symlinks and no F02-related mount. Inspect target open references and
retain visibility limits. A new GPU/finite owner may mount the experiment root;
that ancestor conflict must stop the operation, not be cleaned up by this task.

Write/fsync ARCHIVED.json with not_a_saved_model=true, mandatory AV/archive,
original nine-file hashes and reconstruction instructions before any unlink.
Use a nofollow candidate directory descriptor; repeat exact stat and mount checks
immediately before unlinking each listed filename. Record/fsync each released
entry and stop on any mismatch. Preserve the seven nonweight files and all trial
artifacts. Postcheck AV9/nonweight7/archive, confirm only two targets absent,
finalize the marker and record before/after free bytes. Partial failure keeps the
marker and exact released list; no implicit reconstruction, retry or service action.

The [existing archive recipe](2026-11-07-f02-lossless-archive-preparation.md) uses
its archived editor to create into new paths, then full editor verify and all9
candidate SHA checks. This future physical reconstruction was not run.

## Audit record and one-off source

The exact executed [read-only audit source](records/2026-11-07-f02-payload-release-plan/audit.py.txt)
has no release/apply path. It used the existing CPU Python with ionice -c3,
nice -n15 and single BLAS/OpenMP threads. Archive checksum-list SHA remains
dee04b50c984013410ce3aa8fac6891337774f9846c929a0ceadc3d881b626da.
Successful v2 took 1.428s; all checkpoints/services unchanged.

V1 (1.591s)
stopped at whole-container equality after all archive/stat/mount-target guards.
Its after snapshot was not retained, so its exact ordering difference cannot be
reconstructed retrospectively. A fresh diagnostic showed ten containers with
identical complete mount records in different array order; service fields matched.
V2 sorts only mount-record order, retains every field and both snapshots, and
passed27/27 unchanged. Original failed receipt/source and diagnostic remain;
no mutation or model trial happened in either audit.

Plan SHA 9f758511d865d9444858bcf645746efced8f567246dcd3a4472b6c9b055f6a3a; v2 audit SHA 8c655deb16614701126cec894f22e564c8f10fd6e9e5adf268b9fe1cb1fb3aac.
All evidence and both sources are [checksum-listed](records/2026-11-07-f02-payload-release-plan/SHA256SUMS).
F02 remains unadopted; this storage plan changes no official score or trial outcome.
