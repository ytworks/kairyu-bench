# AC: a small coding-priority matrix step from adopted S

AC changes the direction and purpose targets after [AB was held](2026-11-07-matrix-only-derived-radius-ab.md);
it is not a scaled AB checkpoint. It reuses the four matched observations of
adopted S and preserves every accepted native parameter and Q/K/V matrix.
The fresh matched native-ON dev90 trial improved from S 73.26984127 to AC
73.82539683 (+0.55555556 points). **AC is not adopted; its full14 retention run
is in progress.** S remains the working incumbent. S's earlier 74.24206349
score is a historical measurement, not this trial's fresh control; the goal
of 90 remains unmet.

The source state is
1049e67a536317876716b516f6abe90b8b0770d5a011e080340a5a8b12ae561a,
with S tree
60520ea1e63b25a69b10da1f39f63339221b1a0597c358d60a7cc6b0001d3be1.
All analysis uses authored calibration numeric observations. No benchmark
answers, generated thoughts, private coordinates or measurements were sent
to the public-method reviewer.

## Fixed method and second opinion

The eligible set is the same 122,880 sampled coordinates in 60 attention-output
matrices and 180 complete gate/up/down MLP matrices. All 84 objective/fold rows
remain. For cost c_i = 1/max(abs(S_i),1e-6), each correct-logprob row uses its
cost-dual norm n_j = ||g_j/c|| and headroom H_j = -mean(correct_logprob_j):

- Core code, implementation, workflow and algorithm targets: min(0.25*n_j, 0.1*H_j).
- Six science targets: min(0.0625*n_j, 0.1*H_j).
- Other correct-logprob targets and every margin target: -0.002.

The existing quantized proposal uses a continuous coordinate box
max(0.025*abs(S_i),1e-6), 16 internal corrections and one 120-second CPU budget.
There is no automatic radius increase, grid or retry. The actual BF16 weighted
L2 cap of 1 is checked afterward by the owner; the existing solver does not
enforce this global cap. BF16 rounding can move coordinates outside the
continuous box, so realized excess is reported separately. Individual row
upper bounds do not prove simultaneous feasibility.

The reporting utility is 0.75 times mean core-purpose LL change plus 0.25 times
mean science-purpose LL change. It is not the QP objective or an accuracy
metric. Gradient normalization expresses parameter sensitivity rather than
accuracy importance; related losses remain visible without a universal
zero-loss or significance gate.

One actual Opus 5.5 review completed in 94.381 seconds, with exclusive requested
model identity, usable matching label C and zero tool calls. Accepted advice
concerned signs, units, parent identity, actual rounding and measured finite
response. Suggested within-fold split-gradient noise correction was deferred:
those gradients were not saved and collecting them would be a different method.
Two-fold dispersion is not an unbiased noise estimate. The review's orthogonal
norm illustration omitted the two folds: eight core and twelve science rows
give sqrt(8*0.25^2 + 12*0.0625^2) = 0.73951, not approximately 0.52.
No extra model collection or mandatory ULP study was added.

The [portable preparation CLI](../wsp-normalized-objective-targets.md) reproduces
the fixed targets and metadata checks using existing readers and numerical
primitives. It does not run a solver or model.

## Actual CPU result

The single proposal took 18.129 seconds internally (18.540 seconds for its
owner process). It changed 6,978 BF16 coordinates across 222 eligible matrices,
with native and Q/K/V changes zero. Actual raw L2 is 0.01659449324 and relative
weighted L2 is 0.622056149, below the fixed cap of 1. There are 51 rounded
coordinates outside the continuous box; the realized norm remains the
reported value. Linear predictions meet 82 of 84 targets; the maximum remaining
shortfall is 1.5355e-7, disclosed before finite measurement.

The raw norm is 44.51 times smaller than AB's. The retained code linear signal
is about 46% of AB's prediction. These are weight geometry and first-order
proxy comparisons, not execution or accuracy ratios. An independent numeric
reconstruction reproduced every prediction exactly.

## Actual finite response and decision

One S → AC → restored-S replay completed in 74.214 seconds. Initial correct NLL
matched all 42 frozen S folds exactly; maximum margin difference was
4.7684e-7 at fold means and 9.5367e-7 at individual cases. All 6,978 selected
BF16 words in 222 tensors matched the expected base, candidate and restored
values. A/A margin, correct-NLL and diagnostic KL differences were all zero.
This proves selected words, not an independent full-device weight hash.
The owner restored the same reference container and confirmed protected
services, source pins, S inventory and incumbent state unchanged.

Measured core LL mean is +0.0525348654 and science LL mean +0.00120429384,
giving the fixed utility +0.0397022225. Actual response misses 29 of the 84
requested bounds, with maximum shortfall 0.421358917. Equal-fold purpose means
below are nats per answer token, not benchmark points.

| Purpose | Correct-logprob delta | Margin delta |
|---|---:|---:|
| Code | +0.196797 | +0.158912 |
| Implementation | +0.001533 | +0.004662 |
| Workflow | +0.012320 | -0.010452 |
| Algorithm | -0.000510 | +0.000633 |
| Science numerics | -0.000089 | +0.001147 |
| Science contracts | +0.001207 | +0.000254 |
| Science state | +0.002384 | +0.001935 |
| Science counting | +0.002876 | -0.002036 |
| Science simulation | +0.004359 | +0.000128 |
| Science probability | -0.003510 | +0.003332 |
| Tools | -0.065305 | +0.047618 |
| Instruction | -0.104178 | +0.009079 |
| Lookup | -0.145064 | -0.360241 |
| General | -0.019169 | -0.129365 |
| Reasoning | -0.000700 | +0.091221 |
| Japanese | +0.840002 | +0.996757 |

The stronger coding signal warrants a native-ON execution trial while retaining
the instruction/tools/lookup liabilities. It did not establish adoption from the proxy; the official comparison and
broad retention decision are separate. The finite response is in-sample and differs substantially from
linear predictions; its forced-empty-thought condition differs from ordinary
native thinking. Diagnostic KL samples only the first and last correct-answer
token positions. No official score or claim of general science ability follows from these
proxy measurements alone. The subsequent native-ON result is recorded below.

## Evidence

Paths below are relative to the private experiment archive.

- opus-coding-priority-small-matrix-method-review/receipt.json:
  7391e0f9a10473d660d640f57faa5b132dd10858a67bcb539543d3a896adea12.
- current-s-code-priority-ac-v1/plan.json:
  657b7d36d1dcf661f09b70430cb82865767f9570d1785af4b8dfcbd62696c6d9.
- current-s-code-priority-ac-v1/proposals/report.json:
  8952e97da9c8ca304239100fab2bbcfe9f5d2844e010f7dc9dabe5e9a238fac7.
- current-s-code-priority-ac-v1/proposals/current-s-code-priority-ac-v1.json:
  ea6d54e7afee3ceb8229419c0f1b8b3101059e403f39765dfc2c9eb8f3403524.
- current-s-code-priority-ac-v1/independent-postcheck.json:
  425f6c106327dcc088daed01cb22369e99bd8e8f5955d48f73bef5432beeae45.
- current-s-code-ac-finite-v1/replay/summary.json:
  615b0ffdd4728d6984bb149360f392c9550157cc99212d73325cce971413b2e1.
- current-s-code-ac-finite-v1/independent-finite-review.json:
  d9189267cb346eeeedfc63fbeaabd6da65994826613b4b39ba1539c56c2f5673.
- current-s-code-ac-finite-v1/final-summary.json:
  a9d6257dc8a99abdc42056442057414e3f1301f64948b6e5a0b72260f4752265.
- current-s-code-priority-ac-v1/selection-decision.json:
  67d28d5d8ce9e6af78caa1bf8de1655671006da1a9742821d1e240d44f8b589d.

Whole CPU validation passed once: 512 collected = 460 passed + 52 optional
skips; Ruff, mypy (74 source files), data and 31 shell checks passed, with no
source changes during validation. Evidence:
cpu-integration-normalized-objective-targets-v1/ci-results.json,
SHA 28d7f52667f04d1f85913611fae922414d075733f06f3ee71ac3a69f93acdfad.
The run started after AC finite restoration; it overlapped checkpoint copying
on separate CPU cores with shared storage, so no isolated latency claim is made.


## Completed fresh dev90 and pending broad retention

The same official four-source cohort, archived source, native-ON sampler and
direct TP2/concurrency32 settings produced the following completed comparison:

| Source | Fresh S | AC |
|---|---:|---:|
| HumanEval+ | 30/30 | 30/30 |
| MBPP+ | 23/30 | 24/30 |
| LiveCodeBench | 16/20 | 15/20 |
| SciCode mean | 36.412698% | 40.301587% |
| Equal-source code score | 73.269841 | 73.825397 |

The difference is +0.55555556 points, paired 95% CI
[-3.194444, +4.583333]. Both arms scored all90 items with no unknowns.
Changed score IDs are MBPP Mbpp/641:0→1, LiveCodeBench arc195_b:1→0,
SciCode14:0.5→1, and SciCode72:8/9→7/9. Partial science scores are retained;
these are two gains and two losses, not four binary accuracy flips.

Each arm recorded119 formal chat calls, all stop, with no errors, scored
truncations or empty answers. S/AC recorded301,580/309,700 output tokens and
330.130/345.485 seconds from run start to finish. Each process counter also
included one startup probe (21 input/1 output token, length finish), separate
from those formal calls. Counts are not tool steps, and wall times are not an
isolated causal performance measurement.

S remains the parent at state1049e67a…; its earlier74.2421 measurement must not
replace the fresh73.2698 control. AC's score is not yet a net-purpose adoption
decision. Root selected one full14 retention run against historical S's same
778-item protocol, without repeating the official coding pair.

The full14 wrapper started2026-11-06T18:21:37Z (ownerPID2215962,
benchmarkPID2216154), run
gemma4-code90-candidate-ac-c32-dev10-full14-20261107.
It keeps tier10/dev, chat32/logprob8/agent1, official caps/sampler and source
unchanged. Its control is historical S full14, not a fresh concurrent S run;
actual runner image/config/layers, installed source/packages and timing
overlap must be compared in the final report. The active run's score is unknown.

Additional evidence:
- current-s-code-priority-ac-v1/dev90-final-summary.json:
  94631713082699f1a46812c503cc5afd375f08995a05ee9011703ca48660367b.
- candidate-ac-retention-full14-v1/plan.json:
  375c2825119d572c2bde961cca9d9e51c2aff88ebeb69fd08b885974c2411021.
- candidate-ac-retention-full14-v1/root-decision.json:
  299247e3919ff05964eb8677412b0ce38306af0c37e35f64bc06ee7a23c9e422.
- Historical S candidate-s-retention-full14-v1/final-summary.json:
  3f88274a1de75ba0cfd43a98ff254f4ec2f3fc957559053fdde5fecd7b1f408b.

The [official ledger](2026-11-06-all-official-scores.md) preserves the historical
windows. The [efficiency note](2026-11-07-iteration-efficiency-notes.md) records
future batch/reuse options only; it did not change this trial.
