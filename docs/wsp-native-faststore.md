# Native faststore: fixed shared slots

This opt-in tool makes repeated native scalar/vector edits cheap after one
byte-preserving checkpoint repack. Existing editor and shard-pool defaults are
unchanged. It does not load a model, change serving, select candidates, or decide
which checkpoint is the working incumbent.

## Storage and identity

Build requires an explicit list of BF16 scalar/vector tensor names and a pinned
source freeze with checkpoint_path, control_tree_sha256, all_checkpoint_files,
and expanded candidate_inventory_stat. The complete source inventory is checked;
the source is never linked or opened for writing.

The tool writes one private native.safetensors and one shared.safetensors, copies
nonweight files, and writes the corresponding index. Every tensor name, dtype,
shape, and raw payload byte must match the original. The build streams each old
shard once, comparing its fresh full SHA with the supplied proof while computing
per-tensor SHA values. It then independently reads each new shard once, checking
all per-tensor bytes and computing new full-file hashes. No float conversion,
GEMM, or rounding occurs during repack. Chunk payload buffers are 8 MiB.

The original logical model tree and the repacked physical file tree are different
identifiers. The equivalence receipt connects them; it never relabels the
original file hashes as the repacked hashes. Before actual use, the owner must
bind the complete accepted incumbent, its current state, and this equivalence
receipt. Preparing an older parent does not authorize edits to a newer incumbent.

## Fixed slots and lifetime

Build precreates a bounded number of slot directories (default 16, maximum 64).
Each holds a hardlink to the store-owned large shard. Only after all links and
the read-only file mode are established does the tool freeze inode, ctime, mtime,
size, link count, and full hash.

Create holds a cooperative store lock. It exclusively creates a candidate bundle,
then renames one entire reserved slot into OUTPUT/checkpoint. Directory rename
does not change the shared file's inode, ctime, or link count. Native weights and
metadata are copied privately; only the declared native edits are applied using
the existing editor arithmetic.

No shared file may be unlinked, relinked, expanded, chmod'ed, touched, or written
during the generation's lifetime. This includes shared files inside rejected
or failed candidate directories. A failed claim stays quarantined with its link
intact and consumes capacity. Exhaustion rejects another claim; it does not add
links. Retire a whole generation only after all dependent checkpoints and
serving references have been retired. Read-only mode is protection against
accidents, not filesystem immutability or protection against a hostile owner.

## Cumulative parents and proof scope

The first candidate must name the exact original checkpoint and pinned source
freeze used by build. Its edit storage is the verified equivalent packed base.
A later candidate must name an accepted candidate from this same generation and
its SHA-pinned faststore manifest. Its private native file is the parent, so
edits accumulate. Importing an external post-build candidate is not supported.
Changing a matrix or any tensor outside the isolated selection requires a new
generation; there is no silent fallback.

Create and verify freshly hash private native/metadata files and use the existing
original-byte restoration check. For the shared bulk shard they use its prior
full hash, exact declared shared inode, and unchanged complete stat. They do not
reread its payload. The manifest explicitly records fresh_full_model_hash=false,
the private files freshly hashed, and the shared files verified by prior hash
plus inode/stat. It also carries a complete physical file-tree digest and
candidate_inventory_stat in the existing freeze shape.

Use native_faststore verify for this manifest, not the legacy full-scan editor
verifier. The existing redeployment identity reader can consume its freeze shape,
but an owner that requires nlink=1 for every file must explicitly adopt this
shared-store proof contract first. No existing deployment wrapper is changed.

## Commands

Run as the ordinary owning user, with paths on the same filesystem. Both build
and create are previews unless --apply is supplied. A create preview may read
the small private native/metadata files; it never reads the shared bulk payload.

~~~bash
PYTHONPATH=src python -m kairyu_bench.wsp.weights.native_faststore build \
  /absolute/incumbent /absolute/fresh-store \
  --selected-tensors /absolute/native-names.json \
  --source-proof /absolute/incumbent-freeze.json \
  --source-proof-sha256 SOURCE_FREEZE_SHA --capacity 16 \
  --reserve-bytes 322122547200

# Repeat the reviewed exact command with --apply to build the generation.
# Then bind the SHA of fresh-store/receipt.json.
PYTHONPATH=src python -m kairyu_bench.wsp.weights.native_faststore create \
  /absolute/fresh-store/receipt.json --receipt-sha256 STORE_RECEIPT_SHA \
  --parent /absolute/incumbent \
  --parent-proof /absolute/incumbent-freeze.json \
  --parent-proof-sha256 SOURCE_FREEZE_SHA \
  --spec /absolute/native-edit.json --output /absolute/fresh-candidate-bundle \
  --reserve-bytes 322122547200

# Repeat the exact create with --apply; verify the emitted manifest once.
PYTHONPATH=src python -m kairyu_bench.wsp.weights.native_faststore verify \
  /absolute/fresh-candidate-bundle/manifest.json \
  --manifest-sha256 CANDIDATE_MANIFEST_SHA
~~~

For the next accepted native parent, use the previous bundle's checkpoint
directory and manifest rather than the original source. All names and parent
proofs remain explicit. If a native spec declares source_checkpoint or
base_tree_sha256, each must match this logical parent. The first spec uses the
original logical tree, not the equivalent repacked storage tree.

## Validation and cost

The one end-to-end fixture covers raw BF16 signed zero/NaN/Inf preservation on
repack, two cumulative native candidates, untouched source files, exact shared
ctime/nlink continuity, private byte-corruption rejection, matrix-scope rejection,
and a failed claimed slot that retains its shared link. No full-model repack or
performance measurement is implied by these CPU fixtures.

A separate tiny CPU consumer check used the generated header with two tensors
(142 bytes total). Safetensors 0.8.0 safe_open and Transformers 5.20.2
load_state_dict, using Torch 2.14.0+cu130, accepted the file without an optional
__metadata__ header. Names, dtypes, shapes, signed-zero/NaN/Inf raw words all
matched. Evidence SHA256:
f24ab1f29023860b008f804adf517bacf387cc4aeaab93ae7de3db811c95b8cf.
This did not load a model or read a live checkpoint.

The motivating header-only observation found 421 text native tensors occupying
2,663,032 bytes in two approximately 62.55 GB shards. The one-time preparation
still reads/writes the full model and needs one additional physical copy.
Subsequent native candidates copy that small native shard plus checkpoint
metadata, and verify those private files. Actual time savings require a measured
run; this tool claims reduced I/O scope, not a measured speedup.

The complete CPU integration check passed 510 tests (458 passed, 52 optional
skips), whole-tree Ruff, mypy over 74 source files, model-comparison data checks,
and syntax checks for 31 shell scripts. Source hashes were unchanged throughout.
The check used no model or GPU; actual large-checkpoint preparation is pending.


The first real preparation was deferred when the next accepted experiment
selected sparse matrix edits as well as native values. Matrix changes lie
outside this store generation's isolated selection; a native-only repack would
not accelerate that experiment. The prepared generation remains uncreated,
rather than implying that its full-model equivalence or speedup was measured.
