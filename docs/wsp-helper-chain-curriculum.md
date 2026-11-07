# Short authored science helper chains

`scripts/experiments/gemma4-purpose/helper_chain_cases.py` adds four independent
training families with two dependent stages each. It reuses the existing public
answer client and isolated science execution worker. No official task, answer,
generated reasoning, checkpoint edit, or model call is part of authoring.

| Split | Family | Helper contract | Downstream contract |
|---|---|---|---|
| Train, fold 0 | whitened-fit | Whiten both design and target | Fit and predict with the whitened system |
| Train, fold 1 | boundary-response | Eliminate prescribed boundary variables | Solve and reassemble the full response |
| Train, fold 0 | quadrature-mixture | Normalize quadrature probability masses | Compute a central moment |
| Train, fold 1 | log-mixture-score | Stable posterior mixture probabilities | Compose a posterior expected score |
| Transfer only | covariance-propagation | Propagate independent input variances | Combine correlated output uncertainties |
| Transfer only | transition-reward | Smoothed row-stochastic transition matrix | Compose two transitions and a reward |

Each complete answer unit is one stage's imports and `solve` function. For the
second stage, the supplied `helper` program is prompt context, excluded from the
answer's mean log probability. The training context uses the authored correct
helper. Both stages of a family stay in the same fold. The eight appended rows
form two new purposes, `science_helper_stage` and `science_composition_stage`,
with two families per fold. Appending to the existing 152 rows produces 160 rows,
23 purposes and 46 folds. The original purpose objects remain unchanged.

There is **one collector row per stage**, using one declared primary foil. Each
stage also has a second independently identified failure-mode foil for CPU and
execution evaluation. These additional foils do not duplicate correct-answer LL
rows. `stage_weight=0.5` and `family_weight=1` describe the equal-stage family
summary; they are metadata, not a new collector weighting implementation. The
existing collector still averages the two rows within each purpose/fold. Across
the two stage purposes, every family contributes equally.

## CPU authoring and immutable append

```sh
python scripts/experiments/gemma4-purpose/helper_chain_cases.py \
  --output NEW_BUNDLE --base OLD_CASES.json --base-sha256 OLD_SHA \
  --seed 64123 --trials 8
```

The bundle contains standalone `train-extension.json`, optional merged
`train-cases.json`, `failure-mode-foils.json`, stage-one public case files, and
numeric validation metadata. It never binds a checkpoint. Exact source/base
hashes and actual library versions are recorded. Use the standalone eight rows
for an incremental observation only when the existing observations' parent and
collector contracts still match; binding/composition belongs to the observation
owner. No old gradient is relabeled as a new parent's derivative.

The source-owned reference validator uses the existing worker in a CPU subprocess
and refuses code outside the embedded correct/foil combinations. It validates
fixed seeded inputs and paired metamorphic transformations: response scaling,
equivalent system scaling, translation of moments, log-offset invariance,
variance scaling and pseudocount scaling. Independent algebraic or loop-based
references check values, shape, float64 output, finiteness and nonmutation. Both
foils must fail at least one trial; their labels describe the intended error,
not a diagnosis of any future model answer. Transfer families are never included
in the collector payload. Family diversity remains small and is reported as
four train families, not eight independent families or a formal benchmark.

The actual CPU validation passed all 18 correct stage/mode combinations over
eight trials each, and detected all 24 distinct foils. The one integration
fixture passed in 8.461 seconds, including a deliberately wrong public helper
whose oracle downstream passes while the generated-helper chain fails.
NumPy 2.2.6 and SciPy 1.13.1 were used. Ruff and mypy passed.

The existing tokenizer/frame preparer measured at most 301 tokens for a new
prompt plus its longer answer, and 509 for the combined 160 rows. Every old
frame object and every old case object remained equal. Nothing is truncated.
Longer helper chains are deferred rather than changing the 512-token collector.
No model request or GPU forward was made during this validation.

## Sequential public-code evaluation

The evaluator makes at most three ordinary native-thinking requests per chain:

1. Generate and execute the helper by itself.
2. Generate a downstream answer with the **authored oracle helper** and execute it.
3. Generate a separate downstream answer with the **generated public helper** and
   execute their composition.

`evaluate_requests` accepts the existing client and sandbox callbacks. The CLI
reuses `science_generate.generate` and `science_sandbox.Docker`; it adds no model
client, scoring framework or service lifecycle. All imports/functions are the
extracted public code. Hidden reasoning fields are discarded by the existing
client and never used, inspected, logged or persisted. Model public answers are
transient; only hashes, usage, finish reasons and numeric execution outcomes are
saved. A malformed helper skips its dependent generation and remains explicitly
unscored rather than inventing a result. Runtime/interface failures and request
or sandbox failures remain distinguishable. No automatic retry is performed.

```sh
python scripts/experiments/gemma4-purpose/helper_chain_cases.py \
  --action evaluate --split transfer --output NEW_NUMERIC_OUTPUT
# Above is inert. For an authorized run, supply all late-bound identities:
python scripts/experiments/gemma4-purpose/helper_chain_cases.py \
  --action evaluate --split transfer --output NEW_NUMERIC_OUTPUT \
  --endpoint http://HOST:PORT/v1 --model SERVED_ID \
  --checkpoint-sha256 VERIFIED_TREE_SHA \
  --docker-socket /path/to/explicit.sock --daemon-id DAEMON_ID \
  --image sha256:PINNED_SCIENCE_EXECUTOR --seed 0 --max-tokens 8192 --run
```

The owner must establish served-model identity separately. The endpoint and
checkpoint are late-bound, with no model calls in preview. Temperature and top-p
remain omitted, using the server's ordinary sampler; thinking is enabled through
the existing client. The sandbox has no network, GPU, host mounts or writable
root filesystem. Correct reference values remain outside it.

The report separates helper success, independently generated oracle-helper
downstream success, generated-helper downstream output success, and whole-chain
success (both helper and generated downstream pass). Do not infer independence
or multiply stage pass rates. A gap is a useful observation, not unique causal
localization. A length finish, parse error or timeout is not silently recast as
a scientific arithmetic failure. These small authored scores are not estimates
of official benchmark accuracy.

## Method review and limits

One actual anonymous public-method review used `claude-opus-5-5`, tools disabled,
no fallback, 70.527 seconds and USD 0.142432. Receipt SHA256:
`0e6d6651d91e10eac452dbf206b758f9162e120886ad6297b34a69d2997dca61`.
Disposition SHA256:
`415804e4a5090a32681319df8981cb7bb23ec9e788918cfed644f15a62c90183`.
Only generic method definitions were sent, without private scores, parameters,
case contents or model artifacts. The review prioritized short helper chains
and verified multiple failure-mode foils; this recipe implements that bounded
combination. Broader context, library and sampling studies remain prospective.

Corrections retained from the review: a first failing stage does not uniquely
identify a causal defect; optimization shortfall is not proof of insufficient
capacity; a positive margin-delta target is not an absolute margin floor; shared
tokens do not cancel automatically under different answer-length normalization.
The current coherent matrix sampling covers selected cells, not full neurons.
Changing 16 hidden × 32 features to 64 × 8 also changes feature coverage and rank
bounds, so it would not be a pure width test. No such sampling change is made here.

Teacher-forced oracle-helper LL remains a proxy for ordinary generated-helper
execution. The explicit three-path evaluation measures that gap without reading
reasoning or benchmark answers, and without claiming that target magnitude,
context length or sparse capacity alone caused earlier transfer failures.
