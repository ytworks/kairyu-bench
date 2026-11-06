# Eight dependency-set alternatives from accepted AC

The parent remains accepted AC, state SHA `f0a5c61bdbcf6cdba3b71db6487c786d5af004bbc2b31a8c67ed0a9b72ccd324`, tree `d34fcba755eefa6e451521f81a589484bd6eb7368f831e6620a5577d4ff87e85`. Labels A–H below identify this batch’s method groups, not earlier historical checkpoints with the same letters. These are alternatives from one parent, not additive components. Prior S gradients are not reused.

Fresh observation ran on 2026-11-06 from 18:58:57.804 to 19:04:14.912 UTC (about 317.108s), then restored the same P1 container. Native correct-LL/margin took 58.30/94.37s; matrix correct-LL/margin took 46.05/92.82s in parallel family chains. All four used 136 authored cases, 42 folds, serial answers and the same forced-empty-thought frame. Native coverage is 421 tensors/1,331,516 coordinates; matrix coverage is 410 tensors/209,920 sampled coordinates. Timings do not establish a speedup over earlier observations of different parents.

Independent review found correct-LL identical across native/matrix collectors at all 136 cases and 42 fold means. Margin maximum differences were 9.536743164e−7 per case and 3.495564052e−7 per fold; every matrix value equals the FP32 cast of its native counterpart. Source, attestation, frame and case ordering matched. This establishes comparable objective values, not capability or derivative accuracy; checkpoint payloads were not rehashed here.

Anonymous public-method review used actual Claude Opus 5.5 only, labels A–H, tools 0, 101.755s and $0.201368. No private scores, weights, coordinates or cases were sent. Accepted checks cover source ownership, orientation, ascent signs, realized support/norm and correct-versus-foil interpretation. Raw-parameter cost, exactly two F blocks, nearest-BF16 rounding and eight alternatives remained fixed. Extra gradient collection, a ninth solve, automatic A-control generation and rounding toward the parent were not adopted as requirements.

| Label | Eligible parameters and objective |
|---|---|
| A | Native scalar/norm parameters; standard coding/science targets |
| B | Sampled Q/K routing projections; standard targets |
| C | Sampled V/O transport projections; standard targets |
| D | MLP gate/up/down shared-hidden samples; standard targets |
| E | All sampled Q/K/V/O and MLP matrices; standard targets |
| F | Native plus all matrices; two sensitivity-normalized cost blocks |
| G | O/MLP; coding plus instruction/tools/lookup correct-LL repair |
| H | O/MLP; coding margin priority plus positive correct-LL targets |

The [portable preparation guide](../wsp-dependency-set-plans.md) gives the fixed formulas. Preparation took 11.080s with no solve/model call. F's derived block scales were native 28.8756415849 and matrix 0.0218172662035. Eight existing CPU proposals completed once in 119.010s with two workers; no retry, amplitude grid, new solver or checkpoint materialization.

| Label | Changed values | Tensors | Raw ΔL2 | Declared weighted ΔL2 | Predicted misses /84 |
|---|---:|---:|---:|---:|---:|
| A | 1,648 | 310 | 5.683966 | 0.302663 | 10 |
| B | 5,178 | 116 | 0.019934 | 0.644761 | 7 |
| C | 5,521 | 106 | 0.016756 | 0.593265 | 0 |
| D | 6,757 | 168 | 0.015959 | 0.627218 | 2 |
| E | 6,945 | 357 | 0.019386 | 0.620885 | 0 |
| F | 178,351 | 469 | 0.635858 | 1.349296 | 23 |
| G | 7,856 | 231 | 0.017725 | 0.666772 | 0 |
| H | 10,796 | 233 | 0.020554 | 0.778812 | 0 |

F changed both blocks (76 native and 178,275 matrix values) but exceeded its predeclared metric cap 1 and was excluded without rescaling. Its unscaled relative norm is 9.173727, a different metric. A's larger raw norm despite smaller relative norm is also explicit. Nearest BF16 can round outside the continuous box; excess counts are retained, not silently clipped. Predicted misses are disclosed and do not themselves establish official accuracy failure.

The other seven completed one same-parent finite replay, launched 19:21:32 UTC; replay compute took 188.581s and the same O2 container was restored at 19:26:45.364 UTC. Maximum allocated/reserved GPU memory was 59.46/62.09 GiB. These are authored forced-empty-thought measurements, not native-ON generation or official scores.

| Method | Core LL Δ | Science LL Δ | Repair LL Δ | Core margin Δ | Science margin Δ | Utility | Actual misses /84 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | +0.164488 | +0.115668 | +1.828088 | -0.135336 | +0.005476 | +0.152283 | 29 |
| B | -0.007830 | +0.003006 | -0.026157 | +0.018500 | -0.000279 | -0.005121 | 45 |
| C | -0.047565 | +0.001255 | +0.162018 | -0.052506 | +0.000361 | -0.035360 | 32 |
| D | -0.052781 | -0.003230 | +0.091066 | -0.050370 | -0.001327 | -0.040394 | 49 |
| E | -0.017148 | -0.000391 | +0.056863 | +0.089897 | +0.000108 | -0.012959 | 41 |
| G | +0.007408 | -0.001763 | +0.084315 | +0.073482 | -0.001001 | +0.005115 | 43 |
| H | -0.017431 | -0.001870 | +0.078791 | -0.032886 | -0.001268 | -0.013541 | 43 |

Utility is 0.75 times the mean of four core purposes plus 0.25 times the mean of six science purposes. Repair covers instruction, tools and lookup; its values overlap the remaining-purpose group and are not an additional disjoint score. Fold/purpose means preserve equal purpose weighting.

A raised all four core and all six science purpose LL means (38/42 positive LL folds), but core margin fell because foil LL rose more. Code LL folds were +0.918488 and −0.826291; code margin mean was −0.587304. Science margin improved in five of six purposes. G raised all three repair LL means and core mean LL/margin, while science mean LL/margin fell and five of six science LL means declined. This is a tradeoff, not evidence that the native or matrix family causes a particular capability.

Root selected A as finalist **AD** and G as **AE** for one fresh three-arm official comparison against AC. CPU save-only owners started at 19:33:20 UTC; copying and later official results remain pending in this record. AC remains the working parent. No rescale, re-solve or blind sum was performed, and proxy misses were not relabeled as official accuracy failures.

Independent arithmetic reproduced all seven raw fold deltas, 21-purpose means, group means, utility and 84-bound miss counts. Baseline correct-LL fold difference was zero; margin maxima were 9.536743164e−7 per case and 3.495564052e−7 per fold. Finite exports only fold NLL, so per-case correct-LL parity is unmeasured. The selected-byte audit covers 26,022 union coordinates in 712 tensors through nine applications (base, seven alternatives, restored base); all selected words matched. This is not a full-model GPU hash. Restored margin/NLL/KL were exactly zero and source/stat/service closure passed.

The portable builder, family allowlist and selected-source postcheck passed independent review. One full CPU run collected **514 tests: 461 passed, 53 optional skips**, with whole Ruff, mypy (74 files), data and shell (31) checks passing and source hashes unchanged. It overlapped finite work on separate CPU affinity; timings do not establish a speedup. The single added end-to-end fixture protects derived F costs/targets, scoped G/H behavior, immutable input and foreign-output ownership rather than mirroring constants.


Evidence paths below are relative to private A=analysis/code90-resume-20261106; only aggregates are public:

- current-ac-purpose-observation-v1/completion.json `59f871f6b9c11044a5cc87d65258accbcaff4e05068513b59bb5f59ead0b2f1b`; independent review `cf5c9e6105d0d2954dbd5faf7c9af02ffc9dfae0f0681cbdcdaa0d97e6bceda4`.
- opus-eight-dependency-set-method-review/receipt.json `c03626e32f3479488b5cb4d2cdabd21dd039f51523f2a1ccecd2a71a13804218`; disposition `c8e67a2491fc9f333f64c3c79ff53079d83168c07dc65e7966e8a702f077f9b7`.
- current-ac-eight-hypotheses-v1/method-decision.json `9f7fdf1ac813383a319ac45704bf8216a2dcfed2839dce22375f1c99f4d64a68`; preparation `079f8b4c91b1a8dab78f9e6b4193837abca8f5970d6d83740bdc38f212b1409d`; plan roster `e8341e0ab7758dca8cf94bb316674e864af7514e95473dde02c481dda2d21859`.
- cpu-completion.json `fd2ce29d75c31e4602c56c8e96c38eedf7382b3fb176e9eedb4ba925fe4a1bd6`; numeric postcheck `66198b1abcc2a254e1e4b3e9e4d7853f7d585c3cf7655e466b0c799f0c66b766`; independent review `4ef13f6d65a035107c8ef51a3060b7519f340005e6e17fffc8ded920aff3e990`.
- Executed runtime builder SHA `f0142da0eeafc5f43df50a44dee5398cff5352ec85f8b7833f82b45abf9a0ac8` is retained; public formatting preserves its numeric method.

Separately, removal of 62.55 GB of rejected R2 checkpoint weights was storage housekeeping. Results and manifests were retained.

Final finite and publication evidence (relative to A):

- current-ac-eight-finite-v1/numeric-final-summary.json: 821646da7ff772ad6fca283f0ec1b4aefabe03923632834e274e6221e9a6d1e4; independent review: be0f8cd21e11e2d00c69dca276c773c886495863b78a5c2d1c50718b26158422.
- completion.json: 94ec70ef023f5c5bac8c9d8e243a4ae3ff7e646184664be8174223d36cf236ae; postchecks.json: 8b4d4277ff31ce7b0ddab54b671cbf2506ae479d0ca1ada8b9661a3381e1d395.
- current-ac-eight-hypotheses-v1/generation-selection-decision.json: 7f64a7065a055739e77e628d005019bbbfe981136fa1ce5a61c4ba0ce054ad8f.
- cpu-integration-dependency-set-batch-v1/ci-results.json: 62ee25e89b7041422855a7b098c11b9bcb56612b16e36b180f1470795918a814.
