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

For group selection, scale each objective gradient by the inverse relative
coordinate cost, then normalize each purpose/fold row separately. For each
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

Targets are calculated after the exact coordinate mask. Each selected primary
purpose receives separate positive correct-LL and margin targets proportional
to the corresponding relative-cost dual norm. Only correct LL is capped by a
declared fraction of its remaining headroom. Other purposes receive the
explicit retention lower bound.

This is not a matched-functional-step comparison: changing the subspace also
changes its dual norms and targets. LL, foil LL and margin must all be reported.
Rounded constraint residuals and actual raw/block/weighted distances remain
necessary; parameter distance does not bound behavioral damage.

A method policy declares science/code purposes, optional additional primary
purposes, matrix families, selection rule, target multipliers, relative floor,
retention bound, LL headroom fraction and rounding parameters. The existing numeric postcheck reads the realized-cost limit from the
SHA-bound prospective policy and verifies the candidate's parent, scope, metric
and rounding against that policy. Historical eight-set policies retain their
original fixed limit. The builder does not silently clip or repair a candidate
to meet the limit.

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
group selection through target normalization, the existing quantized solver,
and saved-checkpoint verification; all unselected and native bytes remain
unchanged. The existing proposer E2E also covers exact matrix-name eligibility.
No static registry enumeration tests were added.
