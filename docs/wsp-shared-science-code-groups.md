# Shared science/code matrix groups

`scripts/experiments/gemma4-purpose/prepare_science_code_groups.py` prepares
ordinary coupled-objective plans from fresh observations of a saved current
parent. It does not load a model, solve an edit, materialize weights, or adopt
a candidate. Use the existing proposer, finite replay, saved-checkpoint editor
and ordinary generation evaluation after this preparation.

The motivation is the AD-derived AF/AG experiment: positive scientific
teacher-forced responses did not produce an improvement in the fresh official
four-source code comparison. The next experiment broadens independently
authored library/helper tasks and fits positive general-code objectives
alongside science. The score ledger retains the failed candidates.

## Scope and selection

All prepared plans exclude native parameters from optimization and BF16
realization. A method chooses observed matrix families and either the complete
scope or layer groups with joint science/code gradient agreement.

For group selection, scale each objective gradient by the inverse declared
coordinate cost, then normalize each purpose/fold row separately. The default
is relative cost; an explicit Euclidean metric leaves coordinate scales equal. For each
objective and fold, average the selected science rows and code rows separately.
Keep a whole layer group only if their group dot product is nonnegative in every
objective/fold and positive in at least one. An empty selection produces an
explicit skipped candidate. No alternate scope is silently substituted.

Agreement is a selection heuristic, not circuit attribution. Fold variants
from one family are not independent task families. A small or unstable group
selection is a limitation, not evidence that all other weights are irrelevant.

The proposer now accepts an optional `matrix_tensors` list of exact observed
names. Names must belong to the allowed matrix families. Selecting an MLP tensor
requires its gate/up/down tensor group from the same layer. All sampled input
collections remain validated before the mask is applied. The shared
`matrix_column_indices` function applies the same eligibility to the solver
and this builder's target norms. Older builders retain their historical scope;
they do not automatically implement exact-name selection.

Complete groups remain *eligible* together. Individual BF16 values or tensors
may remain unchanged after rounding; the realized support must be reported.

## Objectives and cost

The default `target_mode: dual_norm` preserves the original method: positive
correct-LL and margin targets scale with the corresponding cost-dual gradient
norm after applying the exact mask. Only correct LL is capped by the declared
fraction of remaining headroom. Consequently, restricting support can shrink
the requested improvement; this is not a matched-functional-step comparison.

An explicit `target_mode: functional` instead requests
`ll_headroom_fraction * (-mean_current_correct_logprob)` and an independent
`margin_target` for every primary purpose/fold. These targets do not shrink
with the coordinate mask or change when the cost metric changes. Zero LL
headroom gives a zero LL target. Both objectives use the collector's existing
per-token normalization and fold aggregation. Other purposes retain the
declared lower bound. The constants are design choices, not validated accuracy
thresholds.

An explicit `cost_metric` may select Euclidean cost with native/matrix block
scales of one, or the existing relative cost with the method's declared floor.
Selection geometry, displayed dual norms and the proposal all use that metric.
The existing solver minimizes parameter cost to meet the requested functional
bounds. It does not optimize task accuracy directly.

The `functional_targets_v1` policy uses a declared `realized_raw_cap` shared
across metrics; the numeric postcheck reports raw, declared weighted and common
relative distances separately. It accepts either the declared reference box or
the exact box from a pinned `derive_matrix_radius.py` analytic report, with
the same parent, targets, metric and matrix mask as the reference plan.
Continuous boxes are not BF16 bounds: realized exceedances and constraint
shortfalls remain visible. Historical relative policies keep their original
cost limit and rounding contract.

Always compare continuous predictions, predictions recomputed from the actual
BF16 delta, and finite replay. The first difference can expose rounding loss;
the second includes nonlinear response. A shortfall alone does not prove
insufficient sampled capacity. Actual correct LL, foil LL and margin must all
be reported. A small parameter distance does not bound behavioral damage.

All observations here use a forced empty thought frame. Success on the same
authored calibration items, especially those used to select groups, remains
in-sample. A native-thinking execution result is needed to assess transfer.
A proxy/execution gap does not identify the thought frame as its cause.

## Invocation

Supply a complete validated v2 template with the current parent's saved
checkpoint attestation, cases, matched native/matrix LL and margin collections,
and completed collection content pins. Old gradient arrays cannot be reused
with an expanded cohort.

```sh
python scripts/experiments/gemma4-purpose/prepare_science_code_groups.py \
  --template /absolute/fresh-template.json \
  --policy /absolute/method-policy.json \
  --policy-sha POLICY_SHA256 \
  --output /absolute/fresh-plan-directory
```

The policy binds the same parent-state path/hash as the template and an actual
verified Claude Opus 5.5 methods-only review. No private experiment content is
sent by this tool. Output is one plan per eligible method and a preparation
report containing group agreement, selected coordinates and target norms.

The output directory must be new and cannot overlap input checkpoints,
collections or the policy/template/review. Source identity and content pins are
rechecked before and after preparation. Invalid or incomplete inputs fail
without editing the source checkpoint.

The added end-to-end regression uses separate science/code folds and a strong
conflicting gradient in another layer of the same matrix family. It exercises
group selection through both normalized and functional targets, the existing
quantized solver, actual cost-cap acceptance/rejection, and saved-checkpoint
verification; all unselected and native bytes remain
unchanged. The existing proposer E2E also covers exact matrix-name eligibility.
No static registry enumeration tests were added.

## Bounded derived-box batch

`run_derived_matrix_batch.py` accepts a pinned JSON roster containing the
reference plans, functional policy, current parent, completed collection
receipt, tool hashes and two disjoint CPU ranges. Its default invocation is
an inert metadata preview:

```sh
python scripts/experiments/gemma4-purpose/run_derived_matrix_batch.py \
  --roster /absolute/batch-plan.json --roster-sha256 ROSTER_SHA256 \
  --output /absolute/fresh-batch-directory
```

Add `--run` to execute. Each method runs the existing analytic derivation once
(120 seconds) and then the existing BF16 proposer once (180 seconds).
The batch has at most six methods and two workers; timed-out children are
terminated as process groups. No failed stage is retried. Per-method reports
retain every failure, and only completed stages enter `candidate-roster.json`
and the existing numeric postcheck. The final `completion.json` records
source/parent continuity, timings and exact output identities. No model loads,
checkpoint writes or automatic adoption occur in this runner.
