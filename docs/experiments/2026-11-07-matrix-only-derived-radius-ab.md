# AB: matrix-only dependency step with a derived BF16 radius

AB tests a different permitted weight set after [AA](2026-11-06-current-s-coupled-matrix-native-aa.md)
realized no matrix changes. It keeps the adopted S native parameters fixed and
uses the same fresh S observations for sampled attention-output matrices and
complete gate/up/down MLP triples. Q/K/V columns remain diagnostic only. No
benchmark answers, generated thoughts or private measurements were sent to the
anonymous method reviewer.

The source remains adopted S (working-state SHA
1049e67a536317876716b516f6abe90b8b0770d5a011e080340a5a8b12ae561a).
The owner held AB before checkpoint copying and official evaluation because
the measured code, instruction and Japanese regressions outweighed the small
science-proxy gains for this next trial. S remains adopted. This is a
purpose-based decision from the available diagnostics, not an official accuracy
failure or a universal no-loss rule. No persistent AB checkpoint, official score
or adoption was produced.

## Method and public review

The fixed method replaces AA's demonstrably insufficient matrix box with one
analytic calibration. It uses the same 84 correct-logprob/margin bounds and
relative cost, with native columns excluded before the solve. A single
continuous minimum-cost solution determines rho as the maximum normalized
coordinate step plus 1/128; the absolute radius is max(1e-6, rho*1e-6).
The buffer is a fixed fraction, not a universal BF16 ULP guarantee. One existing
quantized proposal then uses this derived box; no coefficient grid or new
gradient collection followed.

One actual Opus 5.5 method review completed in 71.422 seconds, with tool calls 0,
exclusive requested-model identity and a usable label B response. The accepted
advice concerned source/objective consistency, actual BF16 realization,
nonlinearity and distinguishing predicted constraints from execution. Extra
split-half gradients, random probes, repeated radius choices and additional
approval or zero-loss gates were not added. No private model coordinates or
results were included in that review.

The [portable derived-radius CLI](../wsp-derived-matrix-radius.md) publishes this
composition of existing numerical primitives. The original executed runtime is
preserved separately; publication did not rerun the real calibration.

## Actual CPU proposal

The continuous stage completed once in 2.115 seconds. Its numerical convergence
contract passed (dual residual 1.8783e-10), with continuous raw L2 0.738500 and
relative weighted L2 25.4318. The derived relative radius was 4.17282190546
(417.28%), with absolute radius 4.17282190546e-6. This permissive radius does not
support a claim that every change is locally small.

The quantized stage completed once in 17.622 seconds. It selected internal
correction round 4, changing 82,888 sampled BF16 coordinates across all 240
eligible matrices and all 60 layers. Every MLP triple and attention-output
matrix received a nonzero change; native and Q/K/V edit entries were zero.
Actual raw L2 was 0.7385476926 and relative weighted L2 25.440962. All 84 linear
predictions passed, and no realized coordinate exceeded the derived continuous
box.

| Matrix family | Changed coordinates | Share of actual squared L2 |
|---|---:|---:|
| Attention output | 20,134 | 49.210% |
| MLP gate | 21,110 | 19.370% |
| MLP up | 20,502 | 15.991% |
| MLP down | 21,142 | 15.429% |

Across all 122,880 eligible coordinates, including unchanged ones, normalized
actual steps had median 0.74%, 90th percentile 6.92%, 99th percentile 27.70% and
maximum 416.44%. These geometry statistics do not establish functional locality
or causal module roles. Independent reconstruction from the frozen source
snapshot and exact spec reproduced counts/norms and all 84 predictions, with
maximum prediction difference 8.67e-18.

## Actual finite response

One S → AB → restored-S replay completed in 73.051 seconds. The initial correct
NLL matched the frozen S collector at all 42 folds; per-case margin difference
was at most 9.54e-7. Selected BF16 readback matched the expected 82,888 coordinates
at base, AB and restoration. The A/A margin, correct-NLL and diagnostic KL errors
were all zero. This is selected-word proof, not a full-model device hash. The
owner restored the same serving container and passed the protected-service,
source and incumbent-state checks.

All 84 predicted constraints passed, but **38 were missed by the actual finite
response**, with maximum shortfall 1.590972. Of 42 folds, correct-logprob improved
in 30 and margin in 19; their equal-fold mean deltas were -0.018775 and -0.055740.
The following are equal-fold purpose means, not benchmark points:

| Purpose | Correct-logprob delta | Margin delta |
|---|---:|---:|
| Code | -0.261664 | -0.773165 |
| Implementation | +0.005755 | +0.003719 |
| Algorithm | +0.007832 | +0.001243 |
| Workflow | +0.005648 | -0.012970 |
| Science numerics | +0.007290 | -0.000685 |
| Science contracts | +0.009118 | -0.000321 |
| Science state | +0.011141 | -0.000172 |
| Science counting | +0.006766 | -0.000913 |
| Science simulation | +0.009369 | +0.000415 |
| Science probability | +0.015587 | -0.004357 |
| Tools | +0.101660 | +0.122097 |
| Reasoning | +0.018296 | +0.019232 |
| Instruction | -0.137174 | -0.126145 |
| Japanese | -0.262718 | -0.152878 |

Science correct-logprob improvements coexist with five of six science margin
means declining and substantial code/instruction/Japanese losses. Neither
linear feasibility nor the science mean alone establishes an improvement.
These authored forced-empty-thought diagnostics differ from native-ON
generation and execution. Diagnostic KL samples only the first/last correct
continuation token positions. The result is not an official accuracy failure,
and no universal all-purpose nonregression rule is inferred.

## Inspectable evidence

Artifact names are relative to the experiment archive; private payloads remain
outside the repository.

- current-s-matrix-only-ab-v1/method-decision.json:
  87df858a542dcb3ae6dbad552d83bb5085ccfc62995904f4177cfc84af0266be.
- opus-matrix-only-dependency-method-review/receipt.json:
  d49be8fbcc5bd7b0d22792f1b30d14d4f4d7fe653d055572d66128c784b928dd.
- current-s-matrix-only-ab-v1/calibration/report.json:
  281352832e18db982cf9822832f30727f5d453cbda3dc7f416d1cf45b06298a5.
- current-s-matrix-only-ab-v1/candidate/proposals/report.json:
  5bb21f7351df38188c74d143e9f7c208e341cbbbc5feecea479ddd66154b81e6.
- current-s-matrix-only-ab-v1/candidate/proposals/current-s-matrix-only-ab.json:
  d03b537f1e67139fb5a508cef2aab2926b1967e604d324bf0801e8fabce1ceff.
- current-s-matrix-only-ab-v1/independent-result-review.json:
  a51e96f4d9f64a8f3369236e54669b38014190cfa2a41dce92f571c37ec06ef6.
- current-s-matrix-ab-finite-v1/replay/summary.json:
  cd8ac0452db9ac523efd7781d6c81def08c1081a5c9e03055eb16b5b38ae6914.
- current-s-matrix-ab-finite-v1/independent-finite-review.json:
  42809636ca574a0ea1669747a955c116f7006037236cb519d00d93269d5c9658.
- current-s-matrix-ab-finite-v1/final-summary.json:
  e913eb5fa2cad9b6e909ca064acd63fa9ef1570b0235e8e714cdb6e9e6de16b4.

The portable tool's whole CPU validation passed once: 511 collected = 459 passed
+ 52 optional skips, Ruff/mypy (74 source files), data and 31 shell checks passed,
with no numerical-source changes during validation. The real AB calibration was
not rerun. This CI began after the finite owner's restoration completed, so
there was no overlap between these two phases.
