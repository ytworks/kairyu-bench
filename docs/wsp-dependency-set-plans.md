# Dependency-set plan preparation

The CLI [prepare_dependency_set_plans.py](../scripts/experiments/gemma4-purpose/prepare_dependency_set_plans.py) prepares eight alternatives from one accepted checkpoint's completed native/matrix correct-logprob and margin collections. It uses existing readers and validates every emitted plan through the existing proposal adapter. It never solves, calls a model, writes a candidate spec or modifies checkpoint bytes.

```sh
python scripts/experiments/gemma4-purpose/prepare_dependency_set_plans.py \
  --repo /path/to/repo \
  --observation /path/to/completed-observation \
  --output /path/to/fresh-plans \
  --parent-sha CURRENT_STATE_SHA256 \
  --review /path/to/public-method-review/receipt.json \
  --review-sha REVIEW_SHA256
```

The observation directory supplies its pinned plan, completed/restored receipt, output hashes, postchecks and independent numeric review. The method-review receipt must have usable requested-model-only labels A–H, no tools, and a sibling method-disposition.json. No input is sent externally. Fixed policy is recorded before gradient reading.

A selects native parameters; B Q/K; C V/O; D MLP gate/up/down; E all sampled matrices; F native plus all matrices with two sensitivity-normalized cost blocks; G O/MLP with instruction/tools/lookup repair targets; H O/MLP with coding margin priority and positive correct-LL targets. Full mapping validation remains; shared-hidden MLP sampling does not force every triple component to change.

Standard core correct-LL targets are min(0.25 × eligible dual norm, 0.1 × headroom); science uses 0.0625. G uses 0.125 for repair and the ordinary floor for science. H uses 0.25 × margin dual norm without an LL headroom cap, plus 0.0625 for core correct-LL. Other rows have floor −0.002. Costs use raw stored values with floor 1e−6; the continuous box is max(0.025 × abs(value), 1e−6), with 16 existing rounding corrections. F multiplies each block's costs by the RMS of its eight core-LL dual norms.

Outputs are method-decision.json, preparation.json, plans/A.json through H.json and plan-roster.json. The roster names fresh proposal destinations. Execution and realized weighted-L2 cap 1 are separate owner actions: this preparer neither enforces the global cap inside the solver nor rescales failed candidates. Report declared, unscaled-relative and raw norms separately. Scope-dependent targets invalidate a simple subset/superset minimum-cost ordering.

One tiny end-to-end fixture protects source propagation, F normalization, scope-dependent targets, G's science-floor behavior, H's uncapped margin and foreign-output preservation. The actual production plans were prepared once; publishing the formatted source does not rebuild them. See the [experiment record](experiments/2026-11-07-eight-current-parent-dependency-sets.md).

Teacher-forced empty-thought objectives do not establish native-ON generation accuracy. Assess scientific LL/margin and related-purpose losses explicitly; do not rank solely by ordinary coding LL or infer parameter-family causality.

Validation: one end-to-end fixture plus the full CPU suite passed (514 collected, 461 passed, 53 optional skips; whole Ruff, mypy74, data and shell31). Source hashes were unchanged during checks. The [selected-source postcheck](wsp-dependency-proposal-postcheck.md) independently reconstructs realized norms/counts and retains both declared and unscaled metrics.
