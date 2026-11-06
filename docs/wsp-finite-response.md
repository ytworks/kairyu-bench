# Saved finite-response tradeoffs

The Python API in `kairyu_bench.wsp.weights.finite_response` analyzes already measured whole-model edit directions. It does not load a model, run WSP, edit a checkpoint, or turn synthetic margins into benchmark scores.

Install SciPy 1.11 or later in a separate analysis environment to use `solve_minimum_change` or `project_bf16_response`. Realization, grid and Pareto functions remain NumPy-only; SciPy is imported only by the solver and active-cone projection.

## Input contract

- `response` has one row per explicit measured objective and one column per endpoint. Adjacent columns contain separately measured positive and negative endpoints of the same direction. Do not infer the negative response by negation.
- `gram` is the full inner-product matrix of effective endpoint weight deltas relative to one original checkpoint and in the same coordinate space. It includes interactions in weight norm, not model response. An identity matrix changes the optimization problem.
- `lower` states every requested margin floor, including allowed losses. There is no implicit universal weighting of coding and other capabilities.
- Optional nonnegative `auxiliary` rows and upper bounds can summarize endpoint KL or NLL changes. Their mixture is only a screening proxy: interior KL/NLL is neither measured nor bounded by this API.

Keep checkpoint identity, chat template, examples, device, batching, evaluation positions and source-file hashes alongside the arrays. Never combine columns from unmatched conditions.

## Continuous selection

`solve_minimum_change(response, gram, lower, auxiliary=..., auxiliary_upper=...)` minimizes total squared change under the secant constraints. It checks all sign branches (up to eight pairs): coefficients are nonnegative, sum to at most one, and opposite endpoints cannot both participate. This prevents apparent benefit from cancelling a direction against its reverse while adding nonlinear responses.

A HiGHS LP finds a feasible point. A scaled full-Gram quadratic program minimizes change. Each returned continuous branch includes primal, stationarity, complementarity and dual-gap checks. Near-singular branches are excluded and reported, not called infeasible. LP-reported infeasibility is distinguished from unresolved numerical failures. Check `all_branches_resolved` before claiming the best continuous result over all branches.

The result is always labeled `secant_prediction_requires_new_finite_measurement`. A changed loss policy is a new analysis; preserve old fixed-policy failure records.

## BF16 realization and Pareto reporting

`bf16_realization(base, endpoint_deltas, coefficients)` combines deltas against the original BF16 base, converts the sum to FP32, and rounds to BF16 ties-to-even once. It returns an in-memory array plus actual changed count, stored delta norm, relative rounding error, direction cosine and a reproducibility hash. No-ops and intended zero directions are explicit. Invalid inputs, nonfinite values, overflow and opposite-sign mixtures are rejected.

These are properties of returned coordinates, not measurements of the resulting network. Rounding can change a direction; a small global relative norm does not establish capability preservation. Report whole-checkpoint and edited-scope denominators where available.

`pareto_indices(benefits, costs)` retains candidates not dominated on every listed benefit and cost. State these axes and retain all per-purpose losses; selecting axes is itself a policy choice.

Before WSP, evaluate shortlisted realized sets with the same model, examples, frame, batch and positions as the matched base. Measure actual objectives, KL/NLL and a restored-base repeat. Cross terms and quantization can invalidate the interpolation. Synthetic calibration is not an independent benchmark; subsequent independent WSP comparison is needed to establish benchmark benefit.


## Reprojecting a rounded proposal

`signed_simplex_grid(endpoint_count)` enumerates a fixed finite grid: one to three directions, coefficients 0.25/0.5/1, total mass at most one, and one sign per direction. Fourteen endpoints produce 1,498 proposals. This is CPU geometry analysis, not 1,498 model evaluations.

`project_bf16_response(base, endpoint_deltas, coefficients, response, gram)` first realizes the BF16 proposal, then projects its actual delta onto the nonnegative cone of only its originally active signed endpoints. It checks the supplied active Gram against the coordinates. It never fits against all positive/negative directions or introduces an unselected direction. The result includes residual norm, residual relative to the actual delta, projected coefficients, conditioning and whether projected mass still lies in the measured simplex. A zero realized delta is explicit.

A large error relative to the original intended delta need not imply a large projection residual: BF16 can remove one component while retaining another. Reprojection can describe that geometry, but it cannot establish that the model follows the corresponding response interpolation. Near-collinear directions, large residuals, or mass beyond the measured simplex must remain visible and are reasons to reject a surrogate ranking. An endpoint itself has an existing measurement; a new interior mixture requires an actual finite forward even when its residual is zero.


## Gemma 4 experiment: 2026-11-06

This task resumed from the official Gemma 4 checkpoint, not the prior finalnorm +1% candidate. The previous strict H/H-small failures remain unchanged. The user revised the objective to prioritize coding and its dependencies while allowing some losses; the following analysis is therefore a new policy, not a retroactive pass of the old one.

Evidence lives under the task's `analysis/finite-response-tradeoff-20261106/`. The original base tree is `cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9`. Saved H/H-small native arrays were checked against all 421 corresponding original-checkpoint tensors: 1,331,516 values, 2,663,032 BF16 payload bytes, exact equality and stable source stat. This is a native scalar/vector coordinate set, not all 31 billion weight coordinates.

1. Eight explicit purpose intents were solved against each of two saved response banks. Nine continuous minimum-L2 proposals were feasible, but their actual BF16 rounding errors were 75.7–100%; one became a no-op. None passed the predeclared 50% rounding screen. The remaining seven intent/bank combinations had no verified solution under that proposal policy. This is why minimizing a continuous norm alone was inadequate.
2. The next recorded policy enumerated the fixed signed simplex grid: 1,498 proposals per bank, 2,996 total. CPU rounding and active-cone reprojection took 192.67 seconds, with zero model forwards and zero checkpoint writes. Ninety-five proposals passed geometric screening, including projection residual at most 0.35 and projected mass at most one.
3. Fourteen passed the separately recorded exploratory proposal screen: code mean at least +0.01; implementation/algorithm at least -0.005; every purpose mean at least -0.10; every fold at least -0.25; projected endpoint KL/NLL proxies at most 0.02/0.25. These are synthetic logit/log-probability-margin screening units, **not accuracy percentages or a universal definition of acceptable capability loss**. All 14 were nondominated when retaining all 15 purpose means and minimizing realized L2, projection residual and both auxiliary proxies. Six distinct intent views were recorded, then three sets were shortlisted for actual finite replay.

| Set | Construction | Changed values | Actual weight delta L2 | Projection residual | Response evidence before replay |
|---|---|---:|---:|---:|---|
| `resume20261106-hs-grid-0020` | H-small reasoning-repair endpoint | 1,076 | 0.0001240356475 | 0 | Existing endpoint measurement |
| `resume20261106-h-grid-0397` | 0.5 G + 0.25 tools + 0.25 Japanese repair | 9,127 | 0.0004125233977 | 0.349597 | Unmeasured interior prediction |
| `resume20261106-h-grid-0485` | 0.5 G + 0.25 instruction + 0.25 reference repair | 8,994 | 0.0004227626161 | 0.337396 | Unmeasured interior prediction |

The first set's existing synthetic means included code +0.0373, instruction +0.0701 and reasoning +0.2414, with tools -0.0074 and general -0.0422. The two new mixtures predict code/implementation/algorithm gains but also losses: the second has general -0.0555 and tools -0.0250; the third has general -0.0684 and tools -0.0527. Those mixture values are **reprojected secant predictions**, not finite measurements. Residuals near one third remain material uncertainty.

The three immutable spec hashes are `8403bb886fcc7506953779fdba873e046801c2e4aec7f391d7030e8ce49ee8a4`, `08607ec221d8d7a9d3bf15c04e33fc604c743f69c9293b591bb49dd894435e4b`, and `767e24481dc245056b73bc29b1714d3ea7124be371e8b706192504f4cabd47c7`, respectively. Source/array hashes, each solver branch, all grid observations and discarded candidates remain in the analysis artifacts. Direct math, long-context and agent performance are absent from these saved synthetic rows and require independent WSP evaluation.

## Explicit finite replay

`python -m kairyu_bench.wsp.weights.finite_replay --checkpoint CHECKPOINT --plan PLAN.json --output FRESH_DIRECTORY --gpu-exclusive-approved` is an optional Gemma 4-specific diagnostic entrypoint. It requires local torch/transformers support and one explicitly exposed GPU with at least 70 GiB free. It does not manage or stop serving containers; the owner must arrange the GPU handoff and restore serving afterward.

The hash-pinned plan supplies one to eight fixed original-base native or sparse-matrix BF16 set-value specs, public calibration cases, the expected official frame, checkpoint index hash, and a prior full-content capture attestation. Every tracked input file inventory/path/size/device/inode/mtime/ctime must still match that attestation; its receipt SHA is checked before and after replay. The runner does not claim to have repeated the full payload hash. The process loads the original once, runs original → each fixed candidate → original, restoring from captured original values before each candidate. A finally path attempts a complete restored-base sweep after evaluation failure; CPU tests cover six-candidate ordering, exact native/matrix restoration, and a failure at the fifth candidate. This bounded eight-slot limit allows six proposals plus fixed controls in one load; it does not add an adaptive candidate search. It records per-fold answer margin, correct-answer NLL change, full-vocabulary KL at the fixed first/last correct-answer prediction positions, and A/A errors. There is no all-fold nonregression or WSP acceptance gate. Inputs are read-only; model parameters change only in the isolated process RAM. A process/GPU failure can prevent the final sweep, which is reported as incomplete rather than successful restoration.


### Actual replay result

The owner subsequently ran all three fixed sets on one GPU. Original → three candidates → original completed in **42.34 seconds**; peak allocation was 60.365 GiB. Restored-base margin, correct-NLL and KL differences were all exactly zero, and input identities remained stable. `actual-three-finite/summary.json` and `actual-versus-projection.json` retain the full fold-level evidence.

| Set | Code mean: predicted → actual | Implementation actual | Algorithm actual | Other actual gains / losses | Maximum diagnostic KL / NLL increase |
|---|---:|---:|---:|---|---:|
| hs-0020 | +0.03734 → +0.03734 | +0.00503 | +0.00040 | instruction +0.07012, reasoning +0.24137; tools -0.00741, general -0.04224 | 0.005087 / +0.10136 |
| h-0397 | +0.06588 → +0.01491 | +0.02125 | +0.02121 | workflow +0.02837; tools -0.04055, general -0.06988, lookup -0.06200 | 0.005984 / +0.32690 |
| h-0485 | +0.04868 → +0.03405 | +0.01713 | +0.00919 | general +0.05659, referent +0.03422, fact retention +0.01730; instruction -0.01216, lookup -0.04509 | 0.014175 / +0.26870 |

All three are multi-tensor joint sets, not single-coordinate trials. The provisional WSP screen order is the smallest hs-0020 first, then h-0397 for its implementation/algorithm contributions, with h-0485 as a distinct general/reference tradeoff if needed. hs-0020 code folds were -0.000294 and +0.074979: its positive mean does not imply both folds improved. h-0397 improved both code, implementation and algorithm folds. Some h-0485 purpose signs changed relative to the projection. Consequently, even reprojection with bounded residual was not a reliable substitute for actual finite measurement.

KL here is full-vocabulary divergence at the fixed first/last **correct-answer prediction positions**, averaged by case and fold. Correct NLL covers all correct continuation tokens. These different supports and probability weighting can yield a small KL alongside a larger correct-token NLL increase. None of these diagnostics establishes free-generation correctness, WSP gains, or universal nonregression. No actual WSP result is asserted in this table.

## Original-base composition preview

`compose_bf16_sets(base, variants)` in `weights.composition` is a read-only arithmetic helper. Its inputs are absolute BF16 values sharing one selected coordinate scope. It sums each variant's delta from the same original base in FP64, converts to FP32, and rounds to BF16 once. It reports overlaps, sign agreement, cancellation, realized norm and rounding error. It does not overwrite an existing candidate or decide adoption. It preserves the original stored value on exact numerical cancellation, including signed zero.

The materialized hs0020 was independently checked against its spec, finite metrics and all 52 tensors' actual before/after bytes and SHA values. It changed 1,076 elements across 31 layers: 645 elements in 28 post-attention norms and 431 in 24 post-FFN norms. Those two roles account for 45.12% and 54.88% of squared delta norm. No attention/FFN matrix or final norm was edited.

Two cumulative previews were computed without materializing them or running a model:

| Original-base sum | Shared changed coordinates | Same / opposite signs | Exact cancellations | Final changed values | Realized L2 |
|---|---:|---:|---:|---:|---:|
| hs0020 + 0397 | 862 | 441 / 421 | 205 | 9,135 | 0.0004326863701 |
| hs0020 + 0485 | 882 | 438 / 444 | 205 | 8,983 | 0.0004416910105 |

Both span 90 tensors and 49 layers. Additional BF16 no-ops were one and zero; relative rounding errors were 1.247% and 1.143%. Over 80% of hs0020 coordinates overlap the other sets, with nearly half pointing in the opposite direction. Their benefits therefore cannot be assumed to add, even though full-vector delta cosine is near zero. These previews remain unadopted and unmeasured. Evidence is `joint-composition-preview.json`; the materialization audit is `hs0020-materialization-audit.json`.

## Activation-to-weight follow-up boundary

The observed baseline highlighted layer 50 attention for MBPP 410, layer 51 attention for MBPP 576, and layer 6 MLP for BFCL irrelevance 684 relative to length-nearest successful controls. These are single-task associations, not causal directions. The collector records attention/MLP **before** their post-branch norm; current edits affect the gain **after** that observation. A same-layer pre-norm RMS contrast cannot directly establish how to edit its downstream gain.

Static checkpoint reads found hs0020/0397/0485 changed 14/60/51 post-attention channels at layer 50, 33/89/80 at layer 51, and 0/13/14 post-FFN channels at layer 6. The proposed inspected channels have small gains: approximately 0.0005–0.0007 at layers 50/51, compared with whole-vector medians 0.77344/0.58594. Corresponding O-projection row norms are around one to two. Small gains can compensate large activations; neither small weight magnitude nor a static outlier identifies a coding channel.

`activation-weight-channel-followup.json` proposes six edited channels plus two unchanged controls of similar gain per site, paired with the same residual coordinates. Priority goes to differing signs across candidate edits. It is a proposal only: no new capture configuration was activated. Residuals also contain other branches and are not an isolated post-norm measurement. The existing static reader/inspection API read only the relevant native vectors and 3.75 MB of projection rows; no full checkpoint reread or model call was needed.

## Sparse matrix replay and coordinate reads

A fixed plan can now declare `architecture: "gemma4_sparse_bf16"` and mix
native vector edits with rank-2 BF16 edits. Each matrix edit uses explicit
`coordinates: [[row, column], ...]` and exact stored `set_values`; bounds are
512 coordinates per matrix, 262,144 matrix coordinates per candidate, and one
million coordinates overall. The spec and prior full-content checkpoint receipt
remain hash-pinned. This extension does not propose a direction or accept a
candidate automatically.

`Checkpoint.read_indices` reads only requested adjacent spans through one
stat-checked file handle, preserving requested order and duplicates. The replay
captures the union of edited coordinates once, verifies their loaded values
against the checkpoint, and restores those coordinates before every fixed set
and in the failure recovery path. It does not duplicate each selected full
matrix on the CPU or GPU. A changed or unregistered spec is rejected at the RAM
application boundary.

The model is owned exclusively by the replay and has gradients disabled.
The restoration boundary covers the verified edit union; it is not a snapshot
of arbitrary other mutations to the model. The restored-base forward remains
the final behavioral check. Process/GPU death can still prevent restoration.

The sparse extension uses the same pinned official-chat, teacher-forced public
answer protocol as the native replay. It does not retrospectively reproduce
the earlier F experiment's raw next-token scoring frame. Keep those historical
results and the new protocol separate.

Validation used tiny CPU BF16 tensors, including a joint native/matrix set,
a different second candidate, and an injected forward failure. All six replay
tests passed in a CPU container with no exposed GPU. No real Gemma replay was
run to claim validation of this newly added matrix path.

### Public answer boundary revision

The shared chat frame now records `schema_version: 2` and
`frame_protocol: gemma4-public-answer-v2-forced-empty-thought`. The pinned Gemma
ON generation template stops at `<|turn>model\n`; its system `<|think|>` flag
alone does not place the next token after a thought. The replay explicitly
appends `<|channel>thought\n<channel|>`, the empty-thought boundary emitted by
the official OFF generation template, while retaining the ON system prefix.
There is no separate final-channel token in this template. This is a forced
empty-thought diagnostic condition, not ordinary free-thinking generation.
Only the public correct/foil answer tokens enter the objective; no thought
content is read, generated, or scored.

Historical unversioned frames stopped at assistant start. Their artifacts and
source snapshots remain intact, but they are not evidence of post-thought
public-answer likelihood. The new frame hashes and policy metadata differ,
so an old fixed `frame_evidence` is rejected. Rebuild explicit frame evidence
and collect new derivatives after checking the official whole-model logits
against the custom head/softcap path. Do not combine old and new gradient
arrays or compare absolute likelihoods across frame or batching conditions as
if those conditions were identical. The quantitative effect of the boundary
and BF16-versus-FP32 softcap arithmetic requires the separate bounded diagnostic.
