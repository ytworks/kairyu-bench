# Coupled native public-answer proposals

This CPU-only recipe combines matched native gradients for correct-answer log
probability and correct-minus-foil margin. It uses the existing diagonal-cost
minimum-step solver and BF16 realization; it does not load a model, change a
checkpoint, or decide adoption.

Run a metadata preview before the explicitly requested solve:

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/propose_native_objectives.py \
      --plan plan.json --output fresh-proposal-directory

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/propose_native_objectives.py \
      --plan plan.json --output fresh-proposal-directory --run --write-spec \
      --maximum-seconds 120

The preview reads checkpoint headers, the pinned attestation/cases and collection
metadata. It does not read gradient arrays or solve. A run requires a fresh output
directory disjoint from every input. The explicit CPU deadline is 1–600 seconds;
it does not launch additional searches.

## Input contract

The plan has schema_version 1 and kind native_coupled_objectives_v1:

- checkpoint: path, tree_sha256, and attestation with path and sha256.
- cases: path and sha256 of the authored collection cases.
- collections: correct-logprob and margin directory paths.
- Optional parent_state: path and sha256 of the owner's current incumbent record.
- candidates: one or two fixed objects with id, constraints, cost_metric and
  rounding. These describe distinct reviewed purpose sets, not an amplitude grid.

Every candidate's constraints must cover every collected fold twice, first
correct-logprob and then margin, preserving the cases' purpose/fold order. Each
row supplies objective, purpose, fold and finite minimum_delta. Units are mean
answer-token natural log probability and mean correct-minus-foil log probability,
respectively; both are maximized. No implicit zero-loss floor is added.

cost_metric is either type euclidean, or type relative with an explicit positive
floor. rounding supplies nonnegative relative_radius and absolute_radius (at least
one positive), and rounds from 1 through 32. These are owner-specified search
inputs, not capability thresholds.

The adapter requires the same complete checkpoint identity and attestation,
case bytes, frame, ordered cohort, native tensor columns, serial answer layout,
threads, seed and numerical runtime. The complete BF16 scalar/vector text
inventory is required; virtual gains and matrix-cell collections are rejected.
GPU peak-memory counters may differ and are not numerical settings.

Content hashes come from the pinned prior full checkpoint attestation. Every
recognized file must retain its attested path, size, inode/device and timestamps;
the recipe performs no new full checkpoint hash pass. Native source arrays and
gradient files are hashed when used and checked before publication. The two
source arrays must match BF16 words, including signed zero. The report binds
the solver, common helpers, checkpoint reader and finite-frame source SHA.

A collection from a former incumbent remains historical. Equal native values do
not permit combining gradients from different full checkpoints. If the owner
promotes a new checkpoint, collect both objectives on that checkpoint first.

## Outputs and limits

Each proposal is rounded from the original incumbent plus the accumulated F64
step through F32 to BF16 once. Predictions are recomputed using the resulting
BF16 delta. The report retains all requested lower bounds, predictions,
shortfalls, actual changed count and L2. A continuous radius does not bound
rounded values exactly.

With --write-spec, a nonempty proposal may be emitted even when some predicted
bounds remain unmet, just as in the existing local-gradient recipe. This is an
explicit diagnostic proposal, not a claim that every constraint passed. Neither
predicted constraints nor a statistical significance test are an automatic
adoption gate. Finite responses and official generation scores are separate
evidence.

All native coordinates may be searched, but existing editor/replay limits allow
at most 1,000,000 actually changed coordinates. Exceeding the limit fails rather
than silently trimming the proposal. The represented-source norm refers only to
the native tensor scope, not the whole model.

The v2 frame forces an empty thought at the public-answer boundary of the ON
template. All public answer tokens are included; no thought is read, generated
or scored. These authored diagnostics do not measure free-thinking capability.

## Matched native finite replay

The existing native runner accepts the emitted exact native set_values specs
through its ordinary pinned plan. To match serial native collection:

    PYTHONPATH=src python -m kairyu_bench.wsp.weights.finite_replay \
      --checkpoint CURRENT_INCUMBENT --plan fixed-finite-plan.json \
      --output fresh-finite-output --device cuda:0 --maximum-seconds 600 \
      --answer-batching serial --threads 4 --seed 0 --gpu-exclusive-approved

Only the owner launches this GPU command after reviewing the exact source,
specs and plan. Legacy defaults remain paired, six threads and seed 20261105.
Serial evaluation uses one case's pair-derived padded width for both separate
answer forwards. The objective math is unchanged.

The sequence is incumbent, the fixed candidates, then restored incumbent.
Selected BF16 words are checked against checkpoint words before application,
and CPU-created target words are compared with selected device readback after
application/restoration. Receipts report selected-word SHA and exactness;
they are not full-model device hashes. The finally path attempts restoration
after candidate/forward/output failure, followed by the existing A/A sweep.
The outer owner still restores serving after the isolated process exits.

Finite output reports every fold's margin and correct NLL delta. A decrease in
correct NLL is an increase in correct-answer log probability. Diagnostic KL is
limited to the correct continuation's first/last token positions, not the entire
generated trajectory. Compare the initial baseline with the frozen matched
collector values before interpreting changes.

## Optional native plus sampled-matrix plan

Schema version 2, kind native_matrix_coupled_objectives_v2, keeps the same CLI
and candidate bounds but supplies four directories:

~~~json
{
  "schema_version": 2,
  "kind": "native_matrix_coupled_objectives_v2",
  "collections": {
    "native": {"correct-logprob": "NATIVE_LL", "margin": "NATIVE_MARGIN"},
    "matrix": {"correct-logprob": "MATRIX_LL", "margin": "MATRIX_MARGIN"}
  },
  "baseline_tolerance": 0.000001
}
~~~

The remaining checkpoint, cases, optional parent_state, candidates and rounding
fields are required as in v1. baseline_tolerance is an explicit predeclared
absolute tolerance in answer-token log-probability units. The adapter compares
every corresponding native/matrix baseline case value and reports the maximum
difference. It rejects mismatched whole checkpoints, attestation, cases, frames,
cohorts, serial layout, threads, seed, packing or numerical runtime. Within each
collector family, both objectives must have identical column/source metadata.
The entry scripts differ between families; all recorded input hashes are pinned
individually and shared dependencies must agree.

Collect the matrix pair on the same current parent as the native pair:

~~~bash
python scripts/experiments/gemma4-purpose/matrix.py --checkpoint CURRENT_PARENT --cases AUTHORED_CASES --checkpoint-attestation ATTESTATION --attestation-sha256 SHA256 --mode chat-public-answer --objective correct-logprob --answer-batching serial --sampling-mode mlp-triplets --max-cases-per-microbatch 1 --threads 4 --seed 0 --output FRESH_MATRIX_LL
# Repeat with --objective margin and a different output.
# The authorized owner adds its device, GPU handoff and --run flags.
~~~

mlp-triplets selects 16 stratified hidden indices per layer, shared by gate/up
rows and down columns. Gate and up use the same 32 stratified input indices;
down uses 32 stratified output rows. Each Cartesian sample has 512 cells.
Complete shapes must be [H,D], [H,D], [D,H]. The sample is fixed by seed and
layer before observing gradients. Existing Q/K/V/O head sampling is unchanged;
no aligned residual-row claim is made for O. The default sampler remains
independent, and margin/paired/raw-next-token defaults are unchanged.

Both matrix objectives include all public answer tokens with the same pair
padding. Serial margin takes separate correct and foil VJPs with opposite signs;
correct-logprob takes only the correct VJP. Existing hooks, eight zero-additive
coordinate checks, FP32 factor products and FP64 accumulation are retained.
Native gradients use their existing BF16 VJP and FP32 accumulation; this is an
explicit derivative-method difference, not a bit-parity claim.

V2 observes the complete matrix inventory but searches only O and complete
gate/up/down support plus the complete native inventory. Q/K/V columns are
diagnostic and never enter the solver or emitted edits. Corresponding LL and
margin matrix samples must match exactly, and MLP coordinates are checked
against the declared coherent sample. For Gemma 4 31B this is 421 native tensors,
410 observed matrices, 240 active matrices and 170 diagnostic-only matrices.

Each v2 cost_metric additionally requires explicit positive
block_scales: {"native": ..., "matrix": ...}. Existing relative coordinate cost
is 1/max(abs(current_parent_value), floor); block scales multiply that cost.
The existing solver minimizes the squared L2 norm of cost times the realized
delta. These are parameter-space costs, not activation/Fisher estimates or
measured capability. No scale is selected automatically.

The adapter horizontally joins native and active matrix columns, then stacks
the correct-logprob and margin fold rows. Existing minimum-step and single-BF16
realization code is reused. Reports give absolute and weighted realized norms
by block and matrix, actual changed counts, and sampled cells unchanged after
rounding (including cells with no proposed change). Shared support does not
imply that all three MLP matrices receive nonzero realized edits. A native-only
realization is reported as such; no epsilon edits or rescaling retry is added.

Mixed specs retain native indices and matrix coordinates with explicit
matrix_batch and sampled-plan SHA. Limits remain 512 cells per matrix, 262,144
matrix coordinates and 1,000,000 total actual changes. The existing
gemma4_sparse_bf16 finite architecture supports both in one candidate and
restores their union. Its pinned spec contains the exact coordinates; the
adapter/owner pins the sampled-plan bytes, while the finite runner does not
independently reinterpret that sampled plan. No fresh full-weight hash or new
finite numerical kernel is implied.

### Explicit matrix-only eligibility

A v2 plan may set eligible_blocks to ["matrix"]. If omitted, the existing
["native", "matrix"] behavior remains unchanged; v1 does not accept this field.
All four matched collections and all 84-style objective rows are still
validated. Native columns are removed from the source-value vector, gradient
matrix and output layout before optimization and BF16 realization. This is
exact ineligibility, not a very large diagonal cost.

The active matrix scope remains O plus complete coherent MLP support; Q/K/V
remain excluded. Emission rejects any native or Q/K/V edit in matrix-only mode.
Reports distinguish observed native columns from zero eligible native columns,
and native realized count/norm are zero. The existing editor's untouched-byte
verification preserves native values, including signed zero, in saved output.
This does not force a feasible result or nonzero matrix entries: predictions,
missed bounds and absent participation remain explicit. A matrix-only solve is
a separately declared hypothesis, not an automatic retry of a mixed result.

The v2 plan optionally accepts matrix_families, a nonempty unique allowlist from q, k, v, o, gate, up, down. Omission preserves the O/MLP scope. Only selected sampled columns enter the existing optimizer and BF16 realization; excluded families remain byte unchanged. The report distinguishes all excluded matrix columns from excluded Q/K/V columns. Q/K/V sampling is not a matched attention-head circuit, and MLP shared-hidden sampling does not force all three matrices to change. This changes eligibility only, not the solver, cost formula, editor limits, or finite evaluation.

### Optional exact native-tensor scope

Both native-only v1 and mixed v2 plans may set native_tensors to a nonempty, unique list of exact observed native tensor names. Omission retains all native tensors. Unknown names, duplicates, empty lists and a native list on a matrix-only plan are rejected. Full checkpoint and complete native observation validation still precede proposal construction; the allowlist removes whole tensor columns from source values, both objective gradients, cost, optimization and BF16 realization. Excluded native tensors cannot appear in an emitted edit. Reports distinguish observed from eligible native coordinates. Eligibility does not establish what excluded tensors cause.

The shared select_native_columns helper also serves normalized-target preparation. A target's dual norm is calculated after the same mask and declared block scaling. Existing default numerical kernels and collector semantics are unchanged. Existing tiny end-to-end fixtures save and verify a protected native norm while eligible matrices satisfy both objectives, and reject a derived-target preparation whose only responsive native tensor was excluded; no new test methods are added.
