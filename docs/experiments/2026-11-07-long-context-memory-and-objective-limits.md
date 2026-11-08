# Longer-context observation: memory and objective limits

This note records two failed read-only observations and the resulting change of
observation scope. Neither run created a weight edit, a completed joint derivative
dataset, or a benchmark improvement.

## Parent and evaluation scope

The accepted checkpoint remains AD. Saved AV is a provisional increment **from AD**;
its observation is not adoption. The accepted chain is P → S → AC → AD.
The latest matched four-source coding comparison was AD 72.5753968254 and
AV 72.9325396825, a +0.3571428571 percentage-point difference. The saved AV versus
AD all-area comparison was still running when these observations failed.

Accepted incremental differences from separate rounds must not be added to claim
a cumulative gain. Accepted-parent controls have varied between rounds, and the
successive selected increments have not established a steadily increasing code
level. Checkpoint ancestry and functional improvement are separate claims.

The numeric catalog contains four completed ordinary 90-item evaluations of the
same saved AD tree, scoring 73.825397, 74.242063, 72.158730 and 74.742063.
Their observed range is 2.583333 points across different comparison pairings.
This is descriptive spread, not an IID variance estimate or a confidence interval.
The latest four-arm overlay control is a separate observation.

The longer authored bundle preserves 112 prior complete-answer rows and adds eight
training stages from four independent helper/composition families. Two additional
families are transfer probes. Complete frames are 1,034–1,268 tokens for the new
training rows, with an explicit 2,048-token bound and no truncation. CPU references
and 24 semantic foils were checked; this does not establish that the model fails
the new tasks.

## Observed failures

Both runs used the same provisional saved AV, cases, source, text SDPA dispatch,
serial complete-answer VJPs, and longest-frame-first execution. Only the second
run's allocator budget and its separate output paths changed. Vision dispatch
remained eager; the absent audio backbone was recorded as absent.

| Run | Torch cap / required free | Native LL | Matrix LL before cleanup | Result |
| --- | --- | --- | --- | --- |
| v1 | 88 GiB / 90 GiB | OOM; no completed fold | 31 of 38 folds | Failed; same O2 restored |
| v2 | 92 GiB / 94 GiB | OOM; no completed fold | 27 of 38 folds | Failed; same O2 restored |

The first failure requested another 80 MiB with 86.88 GiB allocated and 1.09 GiB
reserved but unallocated by Torch; physical free memory was 6.33 GiB. This motivated
one explicit budget revision. The second requested another 40 MiB with 90.76 GiB
allocated and 1.22 GiB reserved but unallocated; physical free memory was 2.31 GiB.
No further cap escalation or automatic backend/length fallback was authorized.

The controller's `ValueError` wraps the failed child exit; the collector cause was
`torch.OutOfMemoryError`. Zero completed native folds does not identify the precise
case at failure. Partial matrix progress establishes that some long frames ran;
it supplies neither a full runtime peak nor a complete gradient dataset.

The owners completed restoration at 01:38:03 and 01:44:04 UTC on 2026-11-07. Exact
container identity, source pins, protected AD/AV services and embeddings were checked.
The concurrent all-area benchmark continued on its separate GPU pairs; these timings
are not uncontended performance measurements.

## Next observation scope

For matrix-only candidate directions, native one-dimensional parameters are frozen.
Their derivatives therefore introduce no solve variables. The next owner should
collect fresh matrix correct-LL and margin observations, independently of an unused
native collector, using the same complete 120-case bundle.

The representation must explicitly say native observations were omitted and native
coordinates are frozen. It must use matrix LL baselines and mark cross-family parity
as not measured. It must not fabricate native records, replace missing arrays by
zeros, mix historical eager arrays with SDPA, or present either partial run as complete.
A fresh two-collector run is simpler than introducing partial-resume machinery here.

## Objective limitations

These derivatives condition complete authored answers on an explicitly empty thought
boundary. Official generation enables native thinking. Improved reference likelihood
or foil margin is consequently a surrogate, not evidence of improved code execution.

The previous short authored chains were already solved after a separate explicit
output-shape clarification. That clarification was not a weight improvement.
Longer prompts and CPU-verified foils do not by themselves resolve the objective
mismatch. Numeric native-thinking execution outcomes should inform the next
mechanism; generated reasoning remains uninspected and unsaved.

The prospective alternatives include a coordinate control, equivalent complete
references, an activation-energy metric, coherent MLP units, an observed low-rank
direction basis, and one measured finite-residual correction. These are distinct
hypotheses with different tool and data requirements, not six completed edits.

## Evidence

- `analysis/code90-resume-20261106/provisional-av-long-purpose-observation-v1/`:
  frozen plan, launch, execution, completion and bounded failure analysis.
- `analysis/code90-resume-20261106/provisional-av-long-purpose-observation-v2/`:
  separate explicit budget revision and restoration evidence.
- `analysis/code90-resume-20261106/long-helper-chain-curriculum-v1/`:
  authored data, CPU validation, tokenizer frame evidence and readiness.
- `analysis/code90-resume-20261106/accepted-lineage-paired-view-v1.json`:
  matched score comparisons and immutable ancestry references.


## Accepted-parent decision after measurement

At 01:59:49 UTC on 2026-11-07, root adopted saved AV as the working parent, preserving AD's historical state. The accepted ancestry is P → S → AC → AD → AV. The decision accepts the exploratory code90 +0.357143 pp together with the verified overlay-to-saved BF16 bridge and fresh ordinary full14 −0.054368 pp as a modest tradeoff. It does not establish persistent coding improvement, an all-area gain, or code>90; final code confirmation, known64 and DeepSWE remain pending. Future proposals must anchor to AV, with native parameters frozen, rather than reuse AD derivatives.

Decision: current-av-adoption-v1/working-decision.json SHA 4f3af16544d3244ff27470cbe169669873675f5503893dc12e5ebcbe344923c5; current-state SHA f5096f2331243628c49fcfa7a6440b10e7967145a5a9b4942dd96647d4004519. Earlier statements in this record describe the parent at that stage and are retained as history.

## Subsequent completed observation and generated diagnostic

Matrix-only v3 completed both 38-fold/120-case objectives and restored the same O2 container at 01:52:49 UTC. Peak allocated memory was 85.593/85.659 GiB within the 88 GiB cap; native derivatives remain unobserved. The two partial native runs remain failures, not resumed datasets.

The separate native-thinking long-helper diagnostic completed at 2026-11-07T02:00:37.555310+00:00 after AV's official after-counters were captured. It made exactly 18 calls, all finishing with stop, using 20,199 input and 43,242 output tokens. Helper stages passed 6/6, oracle-helper downstream passed 6/6, and generated-helper downstream passed 5/6; whole chains passed 3/4 training and 2/2 transfer families. Affine-mixture-moments generated-helper downstream failed all eight trials with numeric failure kind runtime, while its helper and oracle-helper mode passed. No generated body or exception text was inspected. This is a changed-helper-context or sampling-sensitive composition failure, not proven faulty-helper propagation.

The fixed six-family sample supplies one observed composition failure and five passing chains, not a broad capability estimate or a new weight-improvement claim. Source and exact AV serving identity remained unchanged, with no service operation. Numeric-only summary: provisional-av-long-helper-generation-v1/numeric-final-summary.json SHA b9566850a708fb497427dd7e351ac112f666400e78be9ce765cf7578f0d73a9f.
