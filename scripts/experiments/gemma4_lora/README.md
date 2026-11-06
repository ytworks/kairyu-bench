# Gemma 4 LoRA experiment recipes

These are the reusable numerical tools from the recorded Gemma 4 experiment.
They are deliberately narrower than the architecture-independent weight reader:
ordinary BF16 LoRA, rank 16, alpha 32, and the exact 410 text projections across
60 layers. Vision factors are inspected but never merged. They do not infer
coding relevance, run a benchmark, download a donor, select a candidate, or
change a serving model.

The original task paths and credentials are not included. Supply local inputs
and a fresh destination. Keep both checkpoint trees on the intended data volume.

## Build the explicit CPU kernel

Use Linux, NumPy and a C++ compiler with OpenMP. Build the shared library into
an operator-selected directory outside the repository:

```sh
sh scripts/experiments/gemma4_lora/build-kernel.sh /mnt/nvme/YOUR_DIRECTORY/bf16_lora_kernel.so
export KAIRYU_BENCH_BF16_LORA_LIBRARY=/mnt/nvme/YOUR_DIRECTORY/bf16_lora_kernel.so
```

The compiler disables fast math and FP contraction. The kernel uses four CPU
threads. It computes FP32 multiplication and addition separately, then rounds
once to BF16 ties to even. Counts distinguish numeric changes from signed-zero
byte changes. Squared distances accumulate differences of the actual stored
values in FP64. The compiled library is local machine code, is never committed,
and its SHA is included in recipe identity.

## Explicit input recipe

A JSON spec has `schema_version: 2`, `kind: "dense_lora_recipe_v1"`,
`scope: "native_text410"`, `coefficient` (exact FP32 in (0,1]), and:

- `donor_path`, `donor_config_path`: local payload and PEFT config;
- `donor_sha256`, `donor_config_sha256`: full file SHA256 values;
- `donor_repo`, `donor_revision`: recorded repository and 40-character revision;
- `base_tree_sha256`: this recipe's complete file-list digest.

To compute the last field, import `dense_lora_recipe` from this directory and
call `tree(Path(source))[1]`. That helper hashes every regular checkpoint file
except `.cache` entries and the example's old
`.kairyu-model-attestation.json`. Its inventory convention differs from
`wsp weights identify`; the digests are not interchangeable. Reading and hashing
a checkpoint is substantial I/O.

The recipe rejects unsupported ranks, mappings, DoRA/RSLoRA, extra saved modules,
invalid spans, or provenance hash mismatches. The declaration of a donor
repository is provenance metadata; these offline tools do not authenticate its
training data or establish evaluation non-overlap.

## Materialize and verify

```sh
python scripts/experiments/gemma4_lora/dense_lora_recipe.py create SOURCE NEW_CANDIDATE --spec SPEC.json --manifest NEW_MANIFEST.json
# The command above is a preview. Add --apply to write the separate checkpoint.
python scripts/experiments/gemma4_lora/dense_lora_recipe.py create SOURCE NEW_CANDIDATE --spec SPEC.json --manifest NEW_MANIFEST.json --apply
python scripts/experiments/gemma4_lora/dense_lora_recipe.py verify NEW_MANIFEST.json
```

Preview validates the pinned donor and mapping, and reports copy requirements;
it does **not** claim to have verified the source full-content digest.
Application checks that digest, uses private touched shards, verifies every
regenerated BF16 byte and every untouched span, hashes both complete trees,
and publishes the manifest exclusively. Failure removes only the newly owned
candidate. A successfully published candidate survives a later exception.
The CLI reserves 500 GiB after the copy by default; `--min-free-gib` changes
that resource policy explicitly.

`dense_lora_native_recipe.py` supports the same commands with exactly one extra
`native_edits` entry:
`{"tensor":"model.language_model.norm.weight","scale":1.01}`.
It composes the scale from the **original** norm alongside the donor merge.
Run this extension as its own process: it installs the native handlers only in
that experimental recipe module, not in Kairyu or the general weight reader.

Implementation files, compiled kernel, NumPy version and build identity form
part of the manifest. This port has a different implementation identity from
the frozen task-local scripts. It does not relabel old manifests as having been
created by the new implementation.

## Read-only adapter diagnostics

```sh
python scripts/experiments/gemma4_lora/audit_public_adapter_norms.py --source SOURCE --spec SPEC.json --output NEW_AUDIT --global-base-l2 GLOBAL_NORM
python scripts/experiments/gemma4_lora/audit_public_adapter_realized.py --source SOURCE --spec SPEC.json --output NEW_AUDIT --global-base-l2 GLOBAL_NORM
```

The first pass validates the factor payload and computes the low-rank Gram
norm without materializing a dense checkpoint. The second reads source blocks
and measures actual changed values, signed-zero changes, norm, rounding loss,
direction cosine, and layer/mechanism groups at coefficients 1/256, 1/64,
1/16, 1/4 and 1. Neither writes a checkpoint.

The global denominator is an explicit value from a separately verified weight
inspection; these diagnostics do not independently authenticate it. Results
include text-scope denominators as well. Input paths and spec must match between
passes. The actual pass checks source file-state continuity and the donor hash.
It does not claim a fresh complete source hash. Existing or partial actual
audit outputs are preserved and rejected for reuse; use a fresh output directory
after investigating an interrupted run. `--seconds` bounds each pass (default
600 seconds).

The CPU regression tests use tiny synthetic matrices with the complete mapping:
dry-run, source preservation, foreign-manifest collision, exact scalar rounding,
read-only coefficient audit, and the combined native norm recipe. They do not
load or edit the real Gemma checkpoint. Earlier measured benefits and failures
remain in [the experiment record](../../../docs/experiments/2026-11-04-gemma4-weight-editing.md).

## Join a completed sampled matrix gradient with a stored donor checkpoint

`donor_gradient_alignment.py` reads the matrix collector's completed
`summary.json`, `coverage.json`, `provenance.json`, `sample-plan.npz` and
fold gradient arrays. It gathers the same BF16 cells from an **already stored**
donor checkpoint; it never recreates the merge, writes weights, calls a model,
or reads benchmark answers/results.

```sh
python scripts/experiments/gemma4_lora/donor_gradient_alignment.py \
  --collection MATRIX_COLLECTION_DIRECTORY \
  --merged-manifest EXISTING_DENSE_RECIPE_MANIFEST.json \
  --output NEW_ALIGNMENT.json
```

Base file states and sampled values must match the gradient's provenance.
Merged recognized checkpoint files are streamed through SHA-256 once and must
match the recipe manifest. This costs a full payload read, with 1 MiB hash
buffers; matrices are never loaded whole. Input arrays are bounded to 64 MiB
uncompressed and one million coordinates, gradients are read one fold at a
time, and source/input continuity is checked again before publishing a fresh
output outside the input trees.

The report gives actual nonzero stored deltas, coverage, and per-fold inner
products/cosines/sign agreement, also split by layer and projection family.
Zero-vector cosine is null. **The inner product predicts only an edit of the
sampled cells.** Stratified coordinates are not an unbiased estimate of a full
LoRA direction, and raw/chat objectives must not be mixed. Non-overlapping
native norm edits can coexist structurally, but their joint finite outcome
still requires separate measurement. Donor training-base and data-overlap
uncertainties remain. If the stored checkpoint was retired, this tool refuses;
it does not silently substitute approximate factor arithmetic.
