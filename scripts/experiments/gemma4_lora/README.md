# Gemma 4 LoRA experiment recipes

These are the reusable numerical tools from the recorded Gemma 4 experiment.
They are deliberately narrower than the architecture-independent weight reader:
the default recipe uses ordinary BF16 LoRA, rank 16, alpha 32, and the exact
410 text projections across 60 layers. The standalone dense path also accepts
the explicit factor-format v2 below. Legacy vision factors are inspected but
never merged. They do not infer
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

## Explicit factor-format v2 (standalone dense only)

Keep the schema-2 dense kind and add the following declaration to opt into
ordinary LoRA factors exported as BF16 or F32, with one uniform rank from 1
through 256:

```json
"factor_format": {
  "version": 2,
  "rank": 128,
  "dtype": "F32",
  "header_sha256": "SHA256_OF_EXACT_SAFETENSORS_JSON_HEADER_BYTES"
}
```

The header digest covers the raw JSON header bytes after the eight-byte length,
including header padding. Full donor and config hashes remain required.
The config rank must match; positive finite `lora_alpha / r` is converted once
to FP32 and that explicit scale is recorded. F32 factors are read as F32 without
first rounding them to BF16. Factor orientation is A=(rank,input), B=(output,rank);
the existing double-prefix PEFT names must map to all 410 original BF16 text
matrices. This opt-in rejects extra tensors, holes/overlap in the adapter
payload, saved modules, alternate LoRA variants, and nonuniform rank/alpha.
It supports ordinary BA semantics, not quantized-base reconstruction.

The numeric order remains fixed: 128-row FP32 B@A, separate FP32 scale multiply,
then the existing coefficient multiply/add and one BF16 ties-to-even rounding.
The kernel and operation wrapper are unchanged. Without `factor_format`, the
rank-16/alpha-32/BF16 route and its exact saved bytes remain unchanged.

Only `dense_lora_recipe.py` create/verify opt into this format.
`validated_recipe(..., allow_factor_format=False)` rejects it by default, so
existing native, per-module, joint, and diagnostic wrappers cannot silently use
their rank-16 or scale-2 assumptions on new factors. No native edits can be mixed
into this format. Use the existing
[CPU verified freeze bridge](../../../docs/wsp-dense-checkpoint-freeze.md)
after materialization. A new manifest pins the changed recipe implementation;
old manifests and historical verification receipts retain their original code
identity and must not be reinterpreted as new-format evidence.

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

## Full-projection continuous donor direction collection

**Framing qualification (2026-11-06):** historical `chat-public-answer`
collections scored authored tokens at the thinking-enabled assistant start.
The mode name did not establish a final-answer channel. Their results remain
conditional on that old frame; do not advance the retired positive-v2 plans.
See the [retirement and evidence limits](../../../docs/experiments/2026-11-06-positive-correct-donor-amendment.md#retirement-of-the-intended-proxy).
A corrected framing contract and its oracle must be verified before new
collections can support a new plan. Official WSP measurements remain valid.

`donor_direction.py` reuses the same explicit calibration cases, official HF
model loader and public-answer objective as the native/matrix recipes. It
adds one zero scalar gate to each of the 410 text projections:

`y = W x + gate * 2 B(A x)`, with rank 16 and LoRA alpha 32.

```sh
python scripts/experiments/gemma4_lora/donor_direction.py \
  --checkpoint ORIGINAL_BF16_CHECKPOINT \
  --checkpoint-attestation CONTENT_ATTESTATION.json --attestation-sha256 SHA256 \
  --cases AUTHORED_TRAIN_CASES.json --mode chat-public-answer \
  --donor-spec PINNED_DENSE_RECIPE.json \
  --unit-costs text-410-map.json --unit-costs-sha256 SHA256 \
  --output FRESH_COLLECTION --device cpu --max-cases-per-microbatch 1
# Add --run only after reviewing the preview. CUDA also requires
# --device cuda:0 --gpu-exclusive-approved and the existing exclusive-owner handoff.
```

The recipe coefficient is ignored: each coordinate represents a unit continuous
`2BA` direction, starting at gate zero. The pinned original source,
adapter/config, exact 410-factor mapping, existing unit-cost audit, cases and
code are identified in provenance. Vision factors are excluded. The unit
weight-space cost is the existing audit's Frobenius norm of `2BA`, not a
norm measured from a newly rounded checkpoint; disjoint projection tensors
make the corresponding continuous Gram matrix diagonal.

The forward returns the original output unchanged at zero. A transparent,
first-order custom VJP recomputes `2B(Ax)` in FP32 during backward and takes
its inner product with the upstream gradient (FP32 products, FP64 reduction,
FP32 gate derivative, FP64 accumulation across one-case microbatches). Source
weights and donor factors are frozen; there are no factor gradients or full
parameter gradients. This avoids retaining an expanded FP32 direction at every
projection. The CPU rank-16 fixture checks the VJP against a literal additive
autograd graph and central finite differences. Nonzero gates and higher
derivatives are unsupported.

Every microbatch compares the unmodified public-answer margin with the
zero-gate margin; both exact equality and maximum error are recorded, and
differences above the explicit reference tolerance stop the collection.
All projections must execute once, inputs must remain stable, and hooks are
removed even after failure. Output includes per-fold 410-dimensional gradient
arrays, coverage, the pinned unit costs, source continuity checks and runtime
memory measurements. It never selects a candidate or writes weights.

These are continuous computation-graph sensitivities, **not exact derivatives
of discrete BF16 weight edits**. Merged BF16 weights and unmerged output
branches have different rounding. A supported direction still needs a separate
finite check and official evaluation. Donor training-base and evaluation-data
overlap remain unknown.

Saved collections can be profiled without a model:

```sh
python scripts/experiments/gemma4_lora/profile_donor_directions.py \
  --collection COMPLETED_COLLECTION --output FRESH_PROFILE.json
```

The profile retains every recorded fold and reports signs, per-layer/family and
attention/MLP uniform-direction derivatives, cost-normalized sensitivities and
cross-fold cosine. A zero-cost, zero-gradient direction stays null where a
ratio is undefined. Common-positive coordinates are ranked descriptively by the
worst recorded fold per unit continuous weight norm; this does not select a
candidate, solve a constraint system or establish generalization. Raw and chat
modes, incomplete collections, changed costs and input/output collisions are
rejected. No case text, benchmark outcomes or model calls are used.

## Signed module coefficients and fixed continuous plans

`solve_module_directions.py` constructs six explicit structural plans from a
completed 410-direction, 34-fold collection, then solves each with the existing
minimum-weight-change diagonal-cost QP. Every plan retains all 34 authored
folds, including its explicit allowed losses; these fitted folds are **not a
holdout**. The geometric minimum does not guarantee minimal behavioral change.

```sh
python scripts/experiments/gemma4_lora/solve_module_directions.py make-plans \
  --collection COMPLETED_34_FOLD_COLLECTION \
  --donor-spec PINNED_SCALAR_RECIPE.json --unit-costs text-410-map.json \
  --output NEW_FIXED_PLANS.json
python scripts/experiments/gemma4_lora/solve_module_directions.py solve \
  --plans NEW_FIXED_PLANS.json --output NEW_SOLVE_DIRECTORY
```

The six plans emphasize balanced coding, scientific contracts, numerical
algorithms, MLP implementation, attention interfaces, and late-layer
compensation. Coefficients are signed FP32 numbers with an absolute bound of
0.25 and a total **continuous** weight Frobenius norm at most 1.0. The metric
uses the disjoint matrices' existing unit-`2BA` norms. Box limits are additional
QP constraints, never clipping of an otherwise invalid solution. A small
predeclared feasibility margin protects FP32 coefficient serialization; the
original constraints and both limits are checked again afterward. An
unconstrained minimum already outside the norm limit is recorded and not
expanded into another solve. Numerical nonconvergence is unresolved, not a
certificate of infeasibility. No failed plan produces a candidate spec.

Successful output specs use schema 3, kind `module_lora_recipe_v1`, and list all
410 coefficients, including exact zeros. They pin the source, adapter, unit
costs and plan. They remain `continuous_prediction_only`: **the coefficients
are not rounded to BF16** and their predictions are not actual BF16 results.

`module_lora_recipe.py` adds a versioned path while leaving scalar schema 2
unchanged. Build/select the existing native kernel as described above. Its
`validated_recipe`, `delta_block`, `merge_block` and `BLOCK_ROWS=128` are shared
with the RAM finite replay. For each nonzero module, the arithmetic is exactly
the existing FP32 `(B_block @ A) * 2`, FP32 coefficient multiplication, separate
FP32 addition to the original BF16 weight, then one ties-to-even BF16 rounding.
The NumPy/BLAS build, native library and block size are recorded; changing these
is not assumed bit-identical. Zero coefficients preserve the original bytes,
including negative zero. A synthetic full-mapping regression verifies that
uniform 0.25 exactly matches the old scalar recipe.

```sh
# Stream actual rounding statistics without writing a checkpoint.
python scripts/experiments/gemma4_lora/module_lora_recipe.py audit ORIGINAL \
  --spec SELECTED_SPEC.json --output NEW_ACTUAL_BF16_AUDIT.json
# Default create is a dry run. --apply requires a new destination/manifest.
python scripts/experiments/gemma4_lora/module_lora_recipe.py create ORIGINAL NEW \
  --spec SELECTED_SPEC.json --manifest NEW_MANIFEST.json
python scripts/experiments/gemma4_lora/module_lora_recipe.py verify NEW_MANIFEST.json
```

Audit/create report actual changed elements, tensor hashes and BF16 delta L2;
continuous L2 and actual BF16 L2 are distinct. A final candidate requires these
actual metrics, matched finite response, and official evaluation before any
adoption claim. Creating weights verifies the full original hash before and
after, uses private destination files, compares untouched bytes, and publishes
a fresh manifest exclusively. Failure cleanup removes only the directory inode
created by that operation. Verification regenerates every edited span and
checks both full trees. The read-only audit records source stat continuity and
does not claim a newly computed full-source hash.

## Adaptive objective amendment: correct-answer likelihood

After a measured candidate improved correct-minus-foil margins while worsening
correct-answer NLL in 33 of 34 fitted folds, the collector gained an explicit
`--objective correct-logprob` mode. This is an adaptive amendment motivated by
those diagnostics, not a new holdout or a guarantee against regressions.

The default remains `--objective margin` and uses the original implementation.
The new mode is restricted to `--mode chat-public-answer`. It keeps the exact
good+foil forward batch, frame, padding, softcapping and token normalization of
the margin objective, but its loss selects only the mean log probability of
the authored good continuation. Increasing this objective decreases NLL for
that exact forced continuation and frame; it does not by itself establish
improved public final answers or coding performance. It uses one VJP per one-case microbatch, without retaining
a graph for a second backward pass. No reasoning content, reference solution
or benchmark answer is introduced.

Use a fresh collection directory. Summary and provenance explicitly record
`objective`; the new arrays use `gradient-correct-logprob-*` names, and zero-gate
checks compare the selected objective against the unchanged original model.
Optional references use `expected_correct_logprob`, never
`expected_margin`. The profile tool must also receive
`--objective correct-logprob`; its default expects margin, so the existing
margin-only solver rejects a correct-likelihood collection.

```sh
# All usual pinned checkpoint, cases, donor and cost arguments still apply.
python scripts/experiments/gemma4_lora/donor_direction.py ... \
  --mode chat-public-answer --objective correct-logprob --output NEW_COLLECTION
python scripts/experiments/gemma4_lora/profile_donor_directions.py \
  --collection NEW_COLLECTION --objective correct-logprob --output NEW_PROFILE.json
```

The tiny CPU fixture verifies exact zero-gate identity, the unchanged
good+foil batch shape, an independent correct-token log-softmax oracle, and
rank-16 directional derivatives against literal autograd and central finite
differences. Future joint constraints may combine separately labeled margin
and correct-likelihood gradients under declared tradeoffs; this extension
does not alter existing candidate specs or automatically adopt any candidate.

### Positive correct-answer likelihood amendment

[The paired-objective solver](solve_positive_module_directions.py) fits six fixed signed donor scopes to explicit correct-logprob gains and separate margin floors. It requires matching 42-fold collections and a pinned caller review, retains the original coefficient/norm bounds, and creates only continuous proposals. See the [objective-amendment evidence and recipe](../../../docs/experiments/2026-11-06-positive-correct-donor-amendment.md) for units, allowed losses, the 120-second CPU limit and the old 34-fold NLL audit. The historical positive-v2 six proposals are retired for the intended final-answer proxy; their stored numeric results remain unchanged. New actual BF16 response and official WSP remain separate gates.

### Explicit serial answer batching

`donor_direction.py --answer-batching serial --max-cases-per-microbatch 1`
uses the same single-answer helper as the native collector. Each original
good/foil pair determines a fixed padded width; all answer tokens remain in
the loss. Margin uses a positive correct-answer VJP and a negative foil VJP,
freeing each graph before the next forward; correct-logprob uses only the
correct-answer VJP. Each answer's zero-gate output is compared with its
unmodified-model output. The FP32 gate derivatives accumulate in an FP32 fold
buffer and are serialized as an F64 array.

The default `paired` path retains its good+foil batch and existing FP64
microbatch accumulation. Provenance, runtime and fold metadata distinguish
answer batching, accumulation precision and baseline/gradient forward counts.
Serial is restricted to chat frames and one case per microbatch. Matching
mathematical objectives does not imply paired/serial BF16 numerical identity;
original objective values must agree in the actual matched conditions before
combining native and donor derivatives. The framing qualification above still
applies: selecting serial does not by itself repair or establish a final-answer
frame.

### One bounded joint native/donor diagnostic

`propose_joint_native_donor.py` reuses the minimum-step solver for one balanced
proposal from matching headroom-aware donor plans and serial native collections.
It requires identical forced-empty-thought frames, original per-case objectives,
source identity, and disjoint native/donor weight tensors. Native raw-coordinate
cost is 1; donor gate cost is the full continuous matrix-direction norm. It
rounds only native values to BF16, keeps donor coefficients FP32, and rechecks
all 84 declared constraints, the coefficient box and the hybrid norm. Failed
postchecks do not trigger clipping, another solve, or automatic adoption.

```sh
python scripts/experiments/gemma4_lora/propose_joint_native_donor.py \
  --plans HEADROOM_PLANS_JSON --native-margin NATIVE_MARGIN_COLLECTION \
  --native-correct NATIVE_CORRECT_LL_COLLECTION --output FRESH_DIRECTORY
```

The 120-second bounded CPU analysis saves arrays plus a report, not a checkpoint
or an editor spec. Its norm combines actual native BF16 differences and
continuous donor matrix differences; donor weight rounding and joint model
response still require a separate matched finite evaluation. It reads authored
calibration metadata/gradients, never benchmark answers or hidden reasoning.
See the [joint diagnostic outcome](../../../docs/experiments/2026-11-06-positive-correct-donor-amendment.md#corrected-frame-joint-nativedonor-cpu-diagnostic).

An explicit `--repair-report FIRST_REJECTED_REPORT` performs one additional
donor-only residual solve while retaining the saved native BF16 values exactly.
It never recursively repairs a repair or changes targets. Both proposal and
repair retain the coefficient/norm checks. To export required-together native
and donor component specs from an eligible saved repair without another solve:

```sh
python scripts/experiments/gemma4_lora/propose_joint_native_donor.py \
  --plans HEADROOM_PLANS_JSON --export-report ELIGIBLE_REPAIR_REPORT \
  --output FRESH_COMPONENT_DIRECTORY
```

Exports pin the report, source arrays and recipe hashes; all 410 donor
coefficients, including zeros, are explicit. They remain jointly qualified
prediction-only components, not independently validated candidates.

### Broader-direction sanity and one native-only correction

`probe_direction_alternatives.py` reads the same frozen plan and matched
collections. It reports local predictions for uniform positive donor
coefficients 0.05/0.10/0.25 without treating the old L2≤1 experiment radius as a
user requirement. Primary/related/other results, exact continuous norm and
impossible predicted logP>0 are explicit; these amplitudes are never applied.
It also performs one native-only residual solve from the first rejected joint
proposal, then checks a new once-rounded BF16 native delta against all original
84 constraints. It does not repeat corrections or repair its own output.

```sh
python scripts/experiments/gemma4_lora/probe_direction_alternatives.py \
  --plans HEADROOM_PLANS_JSON --native-margin NATIVE_MARGIN_COLLECTION \
  --native-correct NATIVE_CORRECT_LL_COLLECTION \
  --native-proposal-report FIRST_REJECTED_JOINT_REPORT --output FRESH_DIRECTORY
```

The output is metadata and a selected-native-value array, not a model or an
adopted spec. The one native solve has a 120-second CPU budget. All measured
model performance and actual dense BF16 rounding remain separate.
