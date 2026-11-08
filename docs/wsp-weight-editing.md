# Audited native checkpoint editing

The experimental native editor turns an explicit edit specification into a new
checkpoint directory. It never loads model code, starts serving, calls an API, or
chooses an edit from benchmark results. Its default is a read-only plan.
Use it only after separately deciding which joint edit to evaluate.

## Commands

Run from an installed checkout with NumPy available:

~~~sh
python -m kairyu_bench.wsp.weights.editor create \
  /models/base /models/candidate \
  --spec /experiments/edit.json --manifest /experiments/edit-manifest.json

# The same command writes only when --apply is present.
python -m kairyu_bench.wsp.weights.editor create \
  /models/base /models/candidate \
  --spec /experiments/edit.json --manifest /experiments/edit-manifest.json \
  --reserve-bytes 536870912000 --apply

python -m kairyu_bench.wsp.weights.editor verify \
  /experiments/edit-manifest.json
~~~

Candidate and manifest parent directories must already exist; both output paths
must be new and outside the source. The specification is also an input and cannot
be overwritten. The reserve is an explicit caller policy, in bytes. Planning
reports a conservative private-copy size and performs no file writes. Creation
uses private copies for retained files by default. An explicitly verified pool can
share unchanged shards; edited shards remain private. There is no implicit pool discovery.

## Supported schema

Schema 1 preserves the experiment's native scalar/vector and sparse matrix
operations: exactly one of scale, add, set, set_values per tensor.
Floating scalars and vectors support BF16, F16, F32, and F64, at most 1,000,000
elements per tensor. set_values uses explicit indices. For example:

~~~json
{
  "schema_version": 1,
  "id": "explicit-joint-example",
  "edits": [
    {"tensor": "model.layer_a.norm.weight",
     "indices": [2, 7], "set_values": [0.5, 0.25]},
    {"tensor": "model.layer_b.o_proj.weight",
     "coordinates": [[3, 4]], "add": 0.0001}
  ]
}
~~~

Tensor names and values above are illustrative; the tool does not identify a
coding channel from a name or a large weight. Matrix edits require BF16 rank 2,
unique explicit [row, column] coordinates, and at most 64 matrix coordinates
per candidate. A predeclared stratified sample can opt into matrix_batch: true
with a lowercase matrix_sample_plan_sha256; limits then become 512 coordinates
per matrix and 262,144 total. The hash is provenance supplied by the caller, not
proof that a sample is scientifically representative.

Values are rounded once to the stored dtype. BF16 uses FP32 conversion followed
by round-to-nearest, ties-to-even, matching the existing native experiment
editor. Nonfinite source/target values, duplicate tensors/indices, out-of-range
coordinates, and edits that vanish after rounding are rejected. Whole-vector
edits may contain unchanged positions as long as some position changes.
Dense LoRA schema 2 is intentionally rejected here; use its separate recipe tool.

## Verification and failure behavior

The manifest retains source/candidate paths, the exact specification, per-edit
before/after bytes, header locations, hashes, actual changed-element counts, and
L2 of the realized stored-value differences. Relative edit L2 uses the recorded
edited span: a whole scalar/vector or one sparse matrix cell. It is **not** the
whole-model relative norm.

Creation reads source bytes into private files, applies only the recorded spans,
then hashes each entire candidate file while substituting original bytes at
those spans. These reconstructed hashes must equal the original full-file
hashes. This detects undeclared differences in weight payloads, headers,
tokenizers, configuration, README, and other retained regular files. .cache
subtrees and .kairyu-model-attestation.json are explicitly excluded from both
copy and verification; the old serving attestation would not describe the new
checkpoint. Symlinks and nonregular files are rejected.

Manifest publication uses an exclusive atomic link; another writer is never
overwritten. Before publication, failure cleanup removes only files whose
inodes belong to this invocation. Once its own manifest has been published, the
complete candidate is retained even if a subsequent exception occurs. Source and
specification stability are checked before publication. Source files are never
opened for writing.

verify recomputes the specification, checks header/byte correspondence and
manifest totals, and performs fresh complete source/candidate scans. Older
schema-1 manifests without full-file hashes can be checked against their current
source; the report explicitly labels the missing historical full-file hashes.
This fresh verification does not establish an earlier source identity that the
old manifest did not record.

No automatic model switch, deletion, benchmark, or publication follows a
successful verification. The Kairyu serving example remains responsible for its
own supported checkpoint selection, full identity, readiness, and rollback.
The experiment's host-specific switch wrapper is not a generic deployment API.

## Small regression coverage

The CPU fixtures cover exact BF16 ties and overflow, vector/matrix edits, legacy
manifest interpretation, full-file restoration, undeclared edits, header
tampering, invalid/no-op specifications, input/output aliases, private-inode
requirements, source mutation, failure cleanup, competing manifest writers, and
an exception immediately after successful publication. They create only tiny
synthetic safetensors files and do not require GPU, Torch, or model downloads.

## Explicit unchanged-shard pool

A pool is optional. Its donor map names existing private, user-owned copies that
are byte-identical to the original shard, for example:

~~~json
{"model-00002-of-00002.safetensors": "/models/previous/private-unchanged-shard.safetensors"}
~~~

~~~sh
python -m kairyu_bench.wsp.weights.shard_pool build \
  /models/base /models/pool --donors /experiments/donors.json
python -m kairyu_bench.wsp.weights.shard_pool build \
  /models/base /models/pool --donors /experiments/donors.json --apply
python -m kairyu_bench.wsp.weights.shard_pool verify /models/pool/index.json

python -m kairyu_bench.wsp.weights.editor create \
  /models/base /models/candidate --spec /experiments/edit.json \
  --manifest /experiments/edit-manifest.json \
  --shard-pool /models/pool/index.json --apply
~~~

Building first hashes **all retained source files** and the selected donors.
Donors cannot alias any source inode; UID ownership, stable content, size,
device/inode, and same-filesystem hardlink support are required. The new pool
links the explicitly selected donor files; it does not copy, delete, or change
permissions on existing files.

Verification freshly hashes each pooled shard and requires continuity of the
complete original source file inventory/stat snapshot. It reports that the
source full hash was established at build time, not recomputed on every reuse.
Each editor creation then performs its own complete source/candidate comparison.
Only untouched shards may be linked; edited shards are always private.
The recorded pool index digest must remain unchanged. Keep shared pool/donor
content immutable while any checkpoint references it. A pool is not a deletion
manager or a promise that an external process cannot later modify a hardlink.

Old experiment pool indexes lack this complete explicit source snapshot. Do not
rename or relabel them as this format; pass their donor files to a fresh build.
No pool is discovered implicitly from a hostname, checkout path, or candidate ID.

## Metadata freeze and conditional followup evidence

~~~sh
python -m kairyu_bench.wsp.weights.receipts freeze \
  --file manifest=/experiments/edit-manifest.json \
  --file spec=/experiments/edit.json --file editor=/source/editor.py \
  --output /experiments/frozen-inputs.json
# Add --apply to publish this new receipt.
python -m kairyu_bench.wsp.weights.receipts verify /experiments/frozen-inputs.json

python -m kairyu_bench.wsp.weights.receipts followup \
  /experiments/followup-ticket.json /experiments/current-snapshot.json
~~~

The explicit file fingerprint tool handles manifest/spec/code (and a compiled
kernel when used). It never discovers credentials, loads model code, or changes
files. The file list defines coverage; omitted dependencies are not implicitly
covered. A content-identical replacement is acceptable to fingerprint verification,
while files must remain stat-stable during each read. These are local integrity
receipts, not signed external attestations.

The followup validator retains the existing development-ticket contract:
completed creator/full-recipe proof, unchanged manifest/spec/donor/config/code,
whole-source state, a freshly computed full candidate hash, and the exact same
healthy serving replica IDs/images/checkpoint path. It also checks recorded
status/log file hashes and the actual creator or full-verify log records.
No trajectory, response text, or reasoning is emitted.

The caller supplies current-snapshot JSON containing candidate, split=dev,
manifest_path, checkpoint_path, fingerprints, full_candidate_hash_recomputed,
fresh_tree_sha256, and replicas. Each replica has name, id, image, running,
healthy, checkpoint_path, and tree_sha256. The fingerprint structure is the
ticket's complete recorded source/spec/donor/code continuity structure, freshly
collected by the caller. The validator recomputes its metadata evidence checks;
it **does not acquire live serving state or independently compute the claimed
current candidate tree hash**.

Consequently successful output says metadata_valid and
automatic_skip_authorized=false. The outer operator must hold the evaluation
lock, collect fresh serving/mount and full-weight-hash evidence, confirm completed
official dev scoring, and prevent intervening mutations. Passing this validator
does not authorize skipping manual, final, or publication recipe verification.
No benchmark queue, serving operation, or automatic skip is implemented.


A concrete CPU-only materialization record is
[F03's preview, one apply and fresh full verification](experiments/2026-11-07-f03-materialization.md).
Its content identity uses the fresh verify hashes only after complete before/after
Checkpoint and POSIX stat equality; the derivation is explicit and does not repeat
a full payload hash. That fixed recipe is specific to its authorized parent/spec,
and successful file verification is separate from capability evaluation.

## Sequential reconstruction before retiring a parent

Keep every adopted native editor manifest in chronological order. Each contains
all retained file hashes and the exact saved-dtype before/after bytes, including
each stage's rounding. Before deleting an old adopted checkpoint, verify the
complete chain from the designated original to the new candidate:

~~~sh
python -m kairyu_bench.wsp.weights.chain /models/original /models/candidate \
  --manifest /experiments/edit-001.json \
  --manifest /experiments/edit-002.json > /experiments/chain-verification.json
~~~

The verifier streams the original once, applies the recorded byte transitions in
order in bounded buffers, checks every stage's full-file hash and each patch's
before bytes, and compares the final bytes to the candidate. Tensor headers,
complete retained inventories, manifest path/hash continuity and source stability
are checked. Intermediate checkpoint directories may already be absent. No third
checkpoint, temporary weight payload, model load or gradient computation is used.
The receipt records the reconstructor version and source hashes for the chain,
editor and checkpoint modules. Retain those source versions, the manifests and
the verification receipt with the trial evidence.

This verifies exact reconstructability; it does not recreate a checkpoint or
replace the separate saved-weight, serving-switch and complete benchmark checks.
It accepts only native schema-1 manifests with full source/candidate hashes. The
original and candidate paths must match the first/last manifest and every parent
path must match the preceding candidate. Replay preserves each recorded stage's
rounding; adding all numerical deltas and rounding once is a different operation.

Checkpoint storage policy remains the caller's responsibility. The editor uses
independent private copies unless an explicit verified shard pool is supplied.
Before choosing private copies, record whether reflink or safe unchanged-file
sharing is available and account for both complete checkpoints plus all other
weight-equivalent artifacts. Do not silently rely on `st_blocks` to claim reflink
sharing or change original file permissions to make sharing possible. A pool is
not required when a documented private-copy fallback fits the storage ceiling.
