# AD functional-target matrix sets AN–AS

Status: six CPU proposals completed and passed the prospective raw-cost eligibility rule. Finite replay and ordinary execution are subsequent measurements. None is adopted; the accepted parent remains AD.

## Question and fixed method

The preceding [library science/code trial](2026-11-07-ad-science-dependency-sets.md#expanded-library-cohort-aham-finite-outcomes) requested gains proportional to sampled gradient dual norms. Its scientific LL goals were only about 0.003–0.05% of baseline headroom, while its parameter-cost cap was slack. The six transfer candidates did not exceed their matched control, and AL subsequently lost its fresh official comparison. Neither target size, relative cost nor forced-empty-thought framing alone is established as the cause.

AN–AS separate the requested functional gain from parameter cost. Each primary fold requests correct-answer LL gain equal to 1% of its nonnegative remaining headroom and margin gain +0.002 nat per answer token. Nonprimary LL/margin rows permit −0.002. Zero headroom gives a zero primary LL target. Existing per-case token means, fold aggregation and correct-minus-foil margin definitions remain unchanged. These are fitting targets, not promised execution or accuracy gains.

Every method starts from the same saved AD parent and the same completed 152-case, 21-purpose, 42-fold observations. Native parameters and Q/K/V remain frozen. Complete observed MLP gate/up/down samples are eligible together; eligibility does not force every coordinate or group to change.

| Candidate | Scope | Cost | Primary purposes |
|---|---|---|---|
| AN | All O + MLP samples | Euclidean | Code, implementation, algorithm + all six science purposes |
| AO | Same as AN | Inverse-absolute-weight relative, floor 1e−6 | Exactly AN's targets |
| AP | All MLP samples; attention preserved | Euclidean | Exactly AN's targets |
| AQ | Existing cross-fold science/code agreement groups | Euclidean | Exactly AN's targets |
| AR | All O + MLP samples | Euclidean | Contracts, state, simulation; code, implementation, workflow; instruction, tools, lookup |
| AS | All O + MLP samples | Euclidean | Numerics, probability, counting; code, algorithm, reasoning |

AQ selected 14 layer groups / 56 matrices; its calibration response is explicitly in-sample. AN/AO/AP/AQ primary LL targets range from 0.009532 to 0.017471. AR's additional repair targets reach 0.211136 and AS's reasoning target reaches 0.065087 because the same 1%-headroom rule is applied. Different primary sets make AR/AS purpose reallocations, not nested capacity comparisons.

The prospective raw L2 cap is 1 for every method. It is this batch's production budget, not a universal user requirement or an accuracy gate. Costs are reported in common raw and unscaled-relative units as well as the declared optimization metric.

## One continuous calibration and one BF16 proposal

The existing minimum-step solver runs once per method. Its continuous solution determines
rho = max(abs(delta) / max(abs(parent), 1e−6)) + 1/128, with absolute radius max(1e−6, rho × 1e−6).
The fixed fractional buffer is not a universal BF16-ULP guarantee. One existing quantized proposal then uses that derived box and 16 internal correction rounds. Initial 0.025/1e−6 rounding fields are reference placeholders, not the production bounds. No amplitude grid, retry, rescale or new optimizer is used.

The portable [batch driver](../../scripts/experiments/gemma4-purpose/run_derived_matrix_batch.py) composes reference plan → 120-second calibration → 180-second candidate → the existing exact-source numeric postcheck. Default invocation is an inert metadata preview. An explicit --run uses two disjoint CPU workers; failed stages retain their evidence and are not retried. Exact parent/source checks and bounded process-group cleanup surround the work.

The one production batch completed in 62.943 seconds on two four-core workers, with BLAS2, zero model calls and zero checkpoint writes. All input pins remained stable.

| Candidate | Changed values / matrices | Raw L2 | Unscaled relative L2 | Derived rho | Predicted bound misses |
|---|---:|---:|---:|---:|---:|
| AN | 103,562 / 240 | 0.372084 | 3,944.483 | 2,066.012 | 0 |
| AO | 86,006 / 240 | 0.928843 | 27.988 | 6.254 | 1 |
| AP | 83,364 / 180 | 0.500633 | 5,914.670 | 3,441.471 | 0 |
| AQ | 27,831 / 56 | 0.671091 | 10,881.927 | 8,604.150 | 0 |
| AR | 98,769 / 240 | 0.286543 | 2,941.604 | 2,201.194 | 0 |
| AS | 100,965 / 240 | 0.321412 | 3,078.050 | 1,404.152 | 1 |

AO's predicted shortfall is 1.10e−9 and AS's is 5.01e−9. AQ and AS each have one actual rounded coordinate outside the continuous box, by 2.48e−6 and 1.59e−6 respectively; these are reported without clipping. Exact source-array reconstruction confirms nonzero BF16 edits, native/QKV zero and raw L2 below 1 for all six. These criteria allow finite measurement; they do not establish ability.

The Euclidean derived relative boxes are large because small parent coordinates can dominate the maximum relative displacement. The actual Euclidean norms therefore must not be described as small relative edits or proof of functional locality. Increasing a box allows the continuous solution; it does not validate its linear prediction. Continuous prediction, actual BF16-delta prediction and measured finite response remain separate.

## Method review and evidence limits

One anonymous public-method review verified actual claude-opus-5-5, tools 0, no model fallback, and no raw-stream or thinking persistence: 77.227 seconds, USD 0.161168. Manual review confirmed all six methods despite the single wrapper context label. The review supported the functional-target design and requested explicit sign/normalization, parent identity, rounding decomposition, in-sample scope and common cost reporting. Optional Gram/cosine studies, size-matched ablations and a new thought-placeholder collection were not made prerequisites.

All derivatives and finite objectives use the forced-empty-thought teacher-forced frame; ordinary deployment uses native thinking. Contrasts among these six methods cannot identify whether frame mismatch, proxy validity or task generalization caused execution failure. A realized fitting gain without execution gain shows a proxy-to-execution gap, not specifically a thought-frame effect. Restricted sampled-support comparisons likewise do not establish the value of unsampled coordinates or prove a circuit.

Evidence paths are relative to analysis/code90-resume-20261106:

| Artifact | SHA256 |
|---|---|
| current-ad-functional-targets-v1/method-policy.json | 92d12129763ab7a2a02a81af3a18e1994015fe64c445b81ac3876222f38095e5 |
| current-ad-functional-targets-v1/reference/preparation.json | 930f75604263f92e4982dc8add3a315112d1a82e3f895e8e485715a94b1b3b0d |
| current-ad-functional-targets-v1/batch-plan.json | 4709bb9db764487408ecd7afab05f8afc8ecbdc48bf24ac19f54a84d4db30df8 |
| current-ad-functional-targets-v1/batch/completion.json | 517f1173a9e91e5789f650ad3dc5bacc7e3e5dcbaa63989adc24fd0f78fd790d |
| current-ad-functional-targets-v1/batch/numeric-postcheck.json | 887380b2b475bae6fd3aa4d0a38acc85c0ce0105b9328dd1737269b7ba205874 |
| opus-functional-target-six-method-review/receipt.json | 76a45201f531cd127b15eeef5ed5d8368944f940e7acc9acb023003569b5f11b |
| prospective-functional-target-six-v1/method-review-disposition.json | c3922544dd19703bc91eb1f512a84aceeeb63975ef4c9cfa7a804561161b2420 |

The unchanged AD parent state is 7d65d218…; the fresh 152-case template is d64d4997…. Candidate data and private runtime artifacts remain outside the repository. The portable driver passed source review, syntax/Ruff and its inert preview before this actual CPU run; the generalized radius and target/postcheck paths passed their existing focused E2E checks. This note does not claim a new whole-suite CI result.

## Measured six-candidate finite response

The single shared-load finite run completed all six candidates and restored the parent in 188.985 seconds. Independent recomputation confirms all eight phases against the predeclared selected-word expectations: 240 tensors and 115,498 union coordinates per phase. Correct-LL fold baseline differences are zero; native margin differences are at most 2.624e−7 per fold and 9.537e−7 per case. Matrix baseline differences also remain below 1e−6. Restored-parent NLL, margin and KL differences are all zero. This verifies the selected words and numeric replay boundary, not a fresh full-checkpoint hash.

Values below are equal fold means followed by equal purpose means. Core3 means code/implementation/algorithm, science6 means the six existing scientific purposes, and repair3 means instruction/tools/lookup. These common reporting groups are not identical to every specialist's primary set.

| Candidate | Core3 LL / margin | Science6 LL / margin | Repair3 LL / margin | Actual bound misses | Mean / maximum diagnostic KL |
|---|---:|---:|---:|---:|---:|
| AN | +0.096713 / +0.092459 | +0.010088 / +0.002612 | −0.001669 / −0.051089 | 34 | 0.003850 / 0.039547 |
| AO | −0.105734 / −0.225936 | +0.011993 / +0.001540 | +0.080786 / +0.043689 | 33 | 0.011634 / 0.286858 |
| AP | +0.016508 / −0.004276 | +0.009380 / +0.001944 | +0.030824 / +0.043283 | 33 | 0.004771 / 0.099189 |
| AQ | −0.009081 / +0.143638 | +0.009823 / −0.000732 | +0.056590 / −0.024543 | 36 | 0.002592 / 0.039070 |
| AR | −0.009109 / −0.156604 | +0.009913 / +0.000912 | +0.149313 / +0.115718 | 31 | 0.009580 / 0.259237 |
| AS | −0.005663 / −0.075293 | +0.007608 / +0.000921 | +0.004824 / −0.014653 | 40 | 0.004962 / 0.075834 |

The actual bounds are missed much more often than the rounded linear predictions, despite passing baseline and selected-byte checks. Counts use shortfall greater than 1e−10 and describe the fitted constraints; they are not an all-constraints adoption gate. Correct LL is computed as negative NLL, and foil LL as correct LL minus margin.

AN has the strongest combined code/science signal here: code alone improves LL +0.268567 and margin +0.287846, with six-science mean gains in both metrics. Its instruction LL declines −0.023254 and lookup LL −0.136897, while tools LL rises +0.155143. AP gives smaller correct-LL gains with a modest core-margin decline; it improves tools LL/margin +0.086779/+0.077263 and reasoning +0.143201/+0.003142.

AO's code LL/margin decline −0.345811/−0.687082 despite positive science means. AQ's core margin gain is accompanied by correct-LL decline and foil-LL decline −0.152719; it must not be called a correct-answer improvement from margin alone. AR improves all three repair purposes in both LL and margin (instruction +0.055633/+0.007723, tools +0.246164/+0.143648, lookup +0.146142/+0.195784), while code LL/margin decline −0.044369/−0.466021. AS's reasoning primary declines −0.126430/−0.078947 and lookup declines −0.190506/−0.144824, despite instruction/tools gains.

These are genuine measured proxy tradeoffs, not execution or official-score results. AN and AP are the strongest balanced next-evaluation priorities; the other methods retain interpretable metric, support or purpose contrasts. No candidate is adopted or automatically combined. The export contains fold NLL and per-case margins, so it does not support an old/new per-case LL decomposition.

The owner restored the exact AL0 container healthy at 23:07:10.219893 UTC. Source pins, all parent checkpoint stats and protected serving identities passed closure. Ordinary native-thinking evaluation remains a separate subsequent measurement.

Evidence relative to A: current-ad-functional-six-finite-v1/replay/summary.json SHA256 baa57ec39bdf8d537ccaa81df18010da2f08018f64cd5d5e013c5f23ddd02ead; numeric-final-summary.json 3fab7ab83a3453bea77106b2f99ea52450ceb9b7dbf9d5315f82065d512776c8; independent-numeric-review.json 98d533192257f8c9f04b7d9b798560e65524728b21e26da49e0256dc871c4442; completion.json 782f3e653c2318068412aa741ab0b2cac75abce7de896fda36148fa249156e94; postchecks.json ecd4385e654b14185400168d21bca923c7ef08c5c7867ab9fc446ca3980cfc01.
