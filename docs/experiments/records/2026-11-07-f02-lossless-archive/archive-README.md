# F02 dependent lossless archive (preparation only)

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
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor create       "$F02_BASE" "$F02_RESTORE" --spec "$F02_META/spec.json"       --manifest "$F02_META/edit-manifest.json" --reserve-bytes 53687091200 > "$F02_META/preview.json"
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor create       "$F02_BASE" "$F02_RESTORE" --spec "$F02_META/spec.json"       --manifest "$F02_META/edit-manifest.json" --reserve-bytes 53687091200 --apply > "$F02_META/create.json"
    PYTHONPATH="$F02_ARCHIVE/source" "$F02_PYTHON" -m kairyu_bench.wsp.weights.editor verify       "$F02_META/edit-manifest.json" > "$F02_META/verify.json"
    (cd "$F02_RESTORE" && sha256sum -c "$F02_ARCHIVE/CANDIDATE-SHA256SUMS")

Before the create step, compare the decompressed spec/manifest SHA256 to
archive.json.original_spec_sha256 and original_manifest_sha256. The new
creation manifest contains new paths and timestamps, so its JSON hash need not
equal the original; the nine recreated checkpoint file hashes must all match.
Check verify.json.verified is true and legacy_full_hashes_missing is false.
Use the archived source and recorded compatible Python/NumPy environment.
