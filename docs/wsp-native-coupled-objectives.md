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
