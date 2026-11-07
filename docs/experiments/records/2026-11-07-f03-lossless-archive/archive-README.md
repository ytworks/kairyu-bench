# F03 dependent lossless archive, preparation only

The immutable AV base is mandatory. This archive is not a loadable checkpoint.
F03 remains complete and currently served; no release or service operation is permitted by this preparation. Existing trial and benchmark results remain elsewhere unchanged; active WSP logs are not copied as final records.

Saved full editor verify proves all edited bytes equal the manifest after_hex and replacing them with before_hex restores complete AV hashes. Existing whole-file hashes plus unchanged current full stats bind the archive to those verified bytes. Selected-coordinate re-planning is not a new payload fullhash/diff or a physical reconstruction.

Content restoration reproduces all nine relative filenames, sizes and full SHA256 values. New inodes, ctime and device values are not promised; original metadata remains provenance.

## Future reconstruction only; not executed here

Reserve sufficient NVMe space. Never overwrite any existing checkpoint or archive.

    R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
    F03_ARCHIVE="$R/analysis/fresh-purpose-20261107-f03-native-v1/lossless-archive-v1"
    F03_BASE="$R/checkpoints/code90-current-ad-helper-av-v1"
    F03_RESTORE="$R/checkpoints/fresh-native-contract-f03-reconstructed-v1"
    F03_WORK="$R/analysis/fresh-purpose-20261107-f03-native-v1/reconstruction-v1"
    F03_PYTHON="$R/infra/weight-tools-ci/bin/python"
    test ! -e "$F03_RESTORE"
    test ! -e "$F03_WORK"
    (cd "$F03_ARCHIVE" && sha256sum -c SHA256SUMS)
    (cd "$F03_BASE" && sha256sum -c "$F03_ARCHIVE/BASE-SHA256SUMS")
    mkdir "$F03_WORK"
    mkdir "$F03_WORK/tmp"
    export TMPDIR="$F03_WORK/tmp" PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    gzip -dc "$F03_ARCHIVE/spec.json.gz" > "$F03_WORK/spec.json"
    gzip -dc "$F03_ARCHIVE/edit-manifest.json.gz" > "$F03_WORK/original-edit-manifest.json"
    # Verify those original-byte hashes against archive.json before create.
    PYTHONPATH="$F03_ARCHIVE/source" "$F03_PYTHON" -m kairyu_bench.wsp.weights.editor create "$F03_BASE" "$F03_RESTORE" --spec "$F03_WORK/spec.json" --manifest "$F03_WORK/edit-manifest.json" --reserve-bytes 53687091200 > "$F03_WORK/preview.json"
    PYTHONPATH="$F03_ARCHIVE/source" "$F03_PYTHON" -m kairyu_bench.wsp.weights.editor create "$F03_BASE" "$F03_RESTORE" --spec "$F03_WORK/spec.json" --manifest "$F03_WORK/edit-manifest.json" --reserve-bytes 53687091200 --apply > "$F03_WORK/create.json"
    PYTHONPATH="$F03_ARCHIVE/source" "$F03_PYTHON" -m kairyu_bench.wsp.weights.editor verify "$F03_WORK/edit-manifest.json" > "$F03_WORK/verify.json"
    (cd "$F03_RESTORE" && sha256sum -c "$F03_ARCHIVE/CANDIDATE-SHA256SUMS")

Require verify.verified=true and legacy_full_hashes_missing=false. The new manifest contains new paths/timestamps; its JSON SHA need not equal the original. The nine checkpoint content hashes must match exactly. Use archived source and the recorded compatible CPU Python/NumPy environment.
