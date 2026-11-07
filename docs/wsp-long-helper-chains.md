# Dependency-rich authored science chains

`long_helper_chain_cases.py` adds four independent training families, each with
one helper stage and one composition stage, plus two disjoint transfer chains.
The eight training rows have two new purposes: `science_long_helper` and
`science_long_composition`. Each family stays in the same fold for both stages;
each purpose has two families per fold. Appending to the selected 112-row set
preserves all its case objects and gives 120 rows, 19 purposes and 38 folds.
The source checkpoint and accepted-parent decision remain caller-bound.

| Split | Family | Dependency and complete result |
| --- | --- | --- |
| Train | Weighted constraint | Whiten observations and assemble prior precision; enforce an exact coefficient equality and return prediction mean/variance columns. |
| Train | Conditional covariance | Preserve ordered Gaussian blocks; apply the Schur complement, then return an affine observation's full covariance and mean. |
| Train | Affine state history | Propagate coupled position/velocity moments and process covariance; carry them through intervals into a final observation. |
| Train | Affine mixture moments | Normalize log posterior weights per observation; combine transformed component moments with between-component covariance. |
| Transfer | Mass spectral response | Use a mass-orthonormal generalized eigensystem; propagate state and report linear observations plus mass energy. |
| Transfer | Rate reward history | Construct a row generator and transition exponential; carry probabilities and accumulate the specified trapezoidal reward. |

Every prompt supplies three relevant helper/library functions and explicit
mathematical conventions, output shape, float64 dtype, units and mutation rules.
The answer unit is a complete `solve` function and imports. Provided context
functions and the preceding stage helper belong to the prompt, not to the
scored answer. Frames contain no filler text or truncated inputs. The pinned
local tokenizer measured complete prefix plus the longer designated answer at
1,034–1,268 tokens for new training rows and 1,016–1,161 for transfer stages.
Natural length was retained; reaching a token count is not evidence of difficulty.

Each stage has two semantic foils with valid output shapes and types. Only the
first foil enters the calibration row. Additional foils are CPU evaluation
probes, so they do not silently duplicate correct-answer LL weight. Stage
metadata records weight 0.5 and family weight 1; collectors continue their
explicit per-fold example mean. The reference validator uses fixed seeded
samples and paired transformations, independent dense/block/spectral oracles,
and the existing worker's shape, dtype, nonfinite and input-mutation checks.
It rejects a foil that fails only by crashing or changing the interface.

## Reuse and commands

The short-chain evaluator retains its defaults. Optional catalog, scoring and
source callbacks let the new module use the same request client, public-code
extractor and isolated sandbox. Model-generated code is never executed by the
host reference validator. The validator accepts only code embedded in the
source catalog. Evaluation requests remain separate: helper, downstream with
an oracle helper, and downstream with the generated public helper. It emits
numeric outcomes, usage, error classes and public-code hashes; it does not
save public answer bodies or inspect hidden reasoning.

```sh
python scripts/experiments/gemma4-purpose/long_helper_chain_cases.py \
  --output /fresh/authored-bundle \
  --base /inputs/selected-cases.json --base-sha256 <sha256> --trials 8

# Inert preview; endpoint, model and executor are supplied only for an authorized run.
python scripts/experiments/gemma4-purpose/long_helper_chain_cases.py \
  --action evaluate --split transfer --output /fresh/evaluation
```

The bundle contains the standalone eight rows, combined cases, extra foil
catalog and CPU validation metadata. `evaluate --run` uses the same explicit
endpoint/model/checkpoint identity, sampler bounds and pinned Docker executor
arguments as `helper_chain_cases.py`. The transfer split allows at most six
requests and the training split at most twelve. Failures and unavailable
outputs remain separate from successful execution.

## Evidence and limits

The source-owned correct programs passed all three evaluation modes for all six
chains with eight trials each under NumPy 2.2.6 and SciPy 1.13.1. All 24 foils
were rejected numerically. The existing helper E2E was extended, without a new
test method, through bundle preservation and three-request bad-helper
propagation: oracle downstream succeeds while a semantically wrong generated
helper fails downstream. These are CPU reference and contract checks, not
model scores or evidence of transfer. Current seeded trials vary numerical values
but use fixed dimensions within each family; they do not establish dimension
generalization despite the dimension-generic prompt contract.

The earlier short-chain training tasks were already solved. Two old transfer
failures disappeared in one exploratory explicit-shape prompt contrast; this
supported a contract-ambiguity concern and did not demonstrate an acquired
ability or a helper propagation deficit. These new prompts therefore state
complete array interfaces directly. The families are independently authored;
no official benchmark body, answer, generated response or hidden reasoning
was used.

A future longer-context observation must use one exact saved parent and a fresh
consistent observation policy across all four collectors and finite replay.
The opt-in SDPA path may execute the longest complete frame first with an
explicit allocator budget, while retaining canonical case/fold identities.
Old eager gradients cannot be relabeled as SDPA gradients. Token limits,
execution order and actual memory peaks do not establish better generation;
there is no automatic backend or context-length fallback.
