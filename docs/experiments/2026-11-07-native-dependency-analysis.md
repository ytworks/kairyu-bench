# AV behavior12: full-native dependency analysis (2026-11-07)

All **12 training cases × 2 objectives, 421 native tensors and 1,331,516 scalar/vector coordinates** were analyzed after both collectors and exact original-service restoration completed. No candidate, target, radius, support, solver, edit, model call or transfer-case analysis was created.

The measurement owner completed in 237.897 s. Excluding model load, correct LL took 29.387 s / 12 forwards and margin took 47.427 s / 24 forwards. Both summaries are completed, source unchanged, with the same 12 exact frames. The exact original target was healthy with empty queues; 26 protected containers were unchanged, the owner was absent and global lock released. Closure SHA: `e889b331d80e724c4313cda5f9cec672d3f1752364e268e4c9b415d12eb897c3`.

The [CPU recipe](../../scripts/experiments/gemma4-purpose/native_dependency.py) reuses only `joint.read_collection`, adding full-native/closure/frame/identity checks. The old F01 analysis mixes sampled-matrix geometry and plan creation and was not used. All 145 input/source SHA pins were rechecked, as were prior full-checkpoint attestation and current stat continuity. Every native source word matched the current AV. The full 31B checkpoint payload was **not** rehashed again. BF16 autograd gradients accumulated in FP32 were analyzed in FP64.

## Baselines versus executed behavior

LL and margin are teacher-forced nat/token diagnostics; margin is mean correct-answer LL minus mean authored-foil LL. Each fold has one case. Numerical behavior is a separate ordinary-thinking generation with 16 isolated CPU fixtures.

| Case | Correct LL | Margin | Native tests | LL gradient L2 | Margin gradient L2 | LL/margin cosine |
|---|---:|---:|---:|---:|---:|---:|
| A1 | -1.373584 | -0.260095 | 0/16 | 83.7980 | 135.8614 | 0.621530 |
| A2 | -1.647463 | 0.228590 | 16/16 | 103.7265 | 104.5598 | 0.032215 |
| B1 | -1.720735 | 0.116192 | 16/16 | 108.3400 | 55.0653 | 0.390024 |
| B2 | -1.107952 | 0.045379 | 16/16 | 86.8339 | 42.2513 | 0.321594 |
| C1 | -2.427545 | 0.633769 | 16/16 | 111.5621 | 52.8320 | -0.063074 |
| C2 | -1.277695 | 0.416756 | 16/16 | 89.1108 | 72.0962 | 0.429516 |
| D1 | -1.511905 | 0.286189 | 16/16 | 80.5869 | 38.5719 | 0.088584 |
| D2 | -1.106694 | 0.453084 | 16/16 | 60.3690 | 30.8473 | 0.225057 |
| E1 | -1.086648 | 0.040020 | 16/16 | 50.7123 | 25.7141 | 0.454791 |
| E2 | -1.854624 | 0.208760 | 16/16 | 95.4209 | 49.8186 | 0.121708 |
| F1 | -2.109413 | 0.557505 | 16/16 | 165.8546 | 184.1227 | 0.615678 |
| F2 | -1.881313 | 0.536499 | 4/16 | 101.3747 | 59.8605 | 0.055294 |

A1 omitted the batched vector-RHS column axis in `np.linalg.solve(A_arr, rhs)` (10 runtime errors / 6 wrong shapes). F2 generated `arr[0::k]` where kth original events numbered from 1 require `arr[k-1::k]`. **The fixed F2 foil instead removes zeros; it is not the actual off-by-one failure.** A +0.536499 margin establishes preference over that foil, while the generated implementation still passed only 4/16. A1 is the only negative pair margin; 11/12 positive margins establish neither argmax-token nor free-generation correctness. B1/B2 numerical passes retain their separate method-requirement caveat.

## Shared and conflicting directions

All 66 correct-LL pairs have positive raw-coordinate cosine (0.193444–0.636706). Margin pairs have 37 positive and 29 negative cosines (−0.142578–0.123118). Shared code/language/output format can explain a common LL component; this does not identify a specialized ability tensor.

| Case pair | Correct-LL cosine | Margin cosine |
|---|---:|---:|
| A1 / F2 (two native failures) | +0.318507 | −0.019027 |
| A1 / A2 (RHS/shape contracts) | +0.499058 | −0.142578 |
| F2 / F1 (original-index contracts) | +0.448718 | +0.034979 |

F2 margin has negative cosine with **all other 11 correct-LL gradients**, −0.143889 to −0.056991; its own LL/margin cosine is only +0.055294. Correct LL and authored-foil margin are consequently not interchangeable measurements.

The successful B1/B2/C1/C2/D1/D2/E1/E2 are retained separately: A1 LL against these LLs spans +0.197346 to +0.471144; F2 LL spans +0.193444 to +0.594294. Their margin alignment with A1/F2 is mixed. All individual values and all 24×24 pairs are in the linked JSON; no success cases were pooled or excluded.

Each family/layer contribution divides its dot product by the **same whole-native gradient norm product**; signed contributions add to the global cosine. Within-group cosines use group norms and are stored separately. A1/F2 LL has 48 positive / 13 negative layer-plus-global contributions, +0.343450 / −0.024943. A1/A2 has 59 / 2 and F2/F1 has 61 / 0. For A1/F2 margin, 29 / 32 contributions sum to +0.012599 / −0.031626. Positive whole-native LL alignment does not imply agreement in every layer. All 60 layers plus the final norm are retained; no layer was selected.

## Weight and BF16 denominators

Full represented **native** source L2 = 70,134.9646, not the whole-model norm. Input/pre-feedforward layernorm weights carry 99.8672% of source squared norm, while the two post-normalization families carry most raw-coordinate gradient squared norm. A small ratio to 70k could conceal a larger relative change in another family; no edit-size or safety conclusion follows.

| Complete native family | Coordinates | Source L2 | Median next-up BF16 gap | A1 LL gradient energy | F2 LL gradient energy |
|---|---:|---:|---:|---:|---:|
| input_layernorm.weight | 322560 | 40876.254276 | 0.03125 | 0.0069% | 0.0027% |
| layer_scalar | 60 | 6.239438 | 0.00390625 | 1.4338% | 3.2275% |
| post_attention_layernorm.weight | 322560 | 558.637725 | 0.001953125 | 62.6063% | 56.3055% |
| post_feedforward_layernorm.weight | 322560 | 781.580567 | 0.00390625 | 34.4773% | 39.7141% |
| pre_feedforward_layernorm.weight | 322560 | 56934.280568 | 0.015625 | 0.0057% | 0.0094% |
| self_attn.k_norm.weight | 17920 | 14.753693 | 0.00048828125 | 1.4550% | 0.7337% |
| self_attn.q_norm.weight | 17920 | 134.593258 | 0.0078125 | 0.0150% | 0.0071% |
| norm.weight | 5376 | 2364.630838 | 0.03125 | 0.0000% | 0.0001% |

Full per-family/layer/tensor counts, norms, min/median/max absolute values, next-up/down BF16 gaps and all 24 gradient energies are saved. The gaps preserve signed-zero and exponent-boundary asymmetry; they describe representable weights, **not an allowed radius or finite-response estimate**. Euclidean derivatives depend on parameterization and units; neither weight magnitude nor gradient norm alone assigns an ability or justifies dropping a family.

All 24 `unused_tensors` lists are empty. All correct-LL gradients have 0 exact-zero coordinates. Margin exact-zero counts, in A1,A2,…,F2 order, are **3106, 2675, 2901, 6061, 3494, 1967, 4709, 3450, 2342, 4631, 1553, 3499** out of 1,331,516. Subtraction/precision can produce exact zeros; they are separate from unused tensors and do not establish absent ability or no finite effect. Counts are also retained by family/layer/tensor.

## Limits, validation and complete records

The frames use `gemma4-public-answer-v2-forced-empty-thought`: an ON prefix followed by an explicit empty-thought boundary, scoring only authored public-answer tokens. Ordinary native generation permits thinking. Matching diagnostic frames does **not** establish equality with generation. No finite response, held-out transfer or official ability gain is inferred.

[summary.json](records/2026-11-07-native-dependency-v2/summary.json) retains all case values, all eight-family/61-group norms and focus-pair contributions. The [lossless full report](records/2026-11-07-native-dependency-v2/report.json.gz) retains every tensor and all 24×24 geometry. Uncompressed SHA: `271ee1afef4afd855c3aa6d589891bef6ad2b144b9499df9646039894835fad7` (5,227,286 bytes; gzip 1,131,673 bytes). [Receipt](records/2026-11-07-native-dependency-v2/receipt.json) contains all raw NPZ/source/input hashes and commands; raw arrays remain on NVMe.

CPU v1 completed in 5.010 s; v2 added explicit zero/unused fields after parent review and completed in 4.633 s. Both runs, exact source and logs remain in [v1](records/2026-11-07-native-dependency/) / [v2](records/2026-11-07-native-dependency-v2/). Baselines and all global pair values are exactly unchanged. Each revision passed two focused tests (0.031 s) for zero-norm unknowns, opposing group contribution/additivity and BF16 boundaries; Ruff passed. No failed analysis execution or model retry occurred. One local tool-dispatch JavaScript syntax error happened while composing the documentation command; it reached neither SSH nor analysis, and the corrected documentation command is retained. Existing common sandbox/collector/owner code was untouched. Prior full r7 CI predates this recipe and is not claimed to validate it.
