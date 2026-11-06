# Positive correct-answer likelihood donor amendment

> **2026-11-06 retirement notice:** All six positive-v2 proposals below are
> retired before candidate finite measurement or official evaluation. The saved
> frame ended at `<|turn>model\n`, the thinking-enabled assistant start, without
> an explicit final-channel boundary. Consequently these derivatives do not
> measure the intended public final-answer continuation. Do not use the saved
> plans or the earlier two-candidate recommendation to advance a checkpoint.
> Numerical records are preserved as historical, frame-conditional evidence.

This is an adaptive diagnostic amendment after two margin-only donor proposals
raised correct-answer NLL. It is not an unseen evaluation or a demonstrated WSP
gain. The previous candidates and partial finite run remain unchanged.

## Evidence for changing the objective

Using only the original 34 authored folds, the fresh margin derivatives reproduced
the earlier derivatives exactly. The new correct-answer derivatives predicted
likelihood loss for every fold of both completed candidates:

| Candidate | Predicted correct-LL losses | Actual correct-LL losses | Sign agreement | Pearson correlation | Mean predicted / actual correct-LL delta |
|---|---:|---:|---:|---:|---:|
| balanced | 34/34 | 33/34 | 33/34 | 0.8092 | −0.13210 / −0.12137 |
| numerical-algorithms | 34/34 | 32/34 | 32/34 | 0.7133 | −0.14655 / −0.13464 |

Within the historical assistant-start diagnostic, higher correct log
probability is better for that fitted objective; its change is the negative of
the correct-NLL change. This is not a final-answer improvement claim. The primary ten folds had predicted/actual changes
−0.09912/−0.09088 and −0.11268/−0.10861, respectively. This supports an objective
mismatch: improving correct-minus-foil margin can largely suppress the foil
while also reducing the correct answer's probability. These local comparisons
do not establish general capability causation.

The eight advanced-v1 folds were excluded from this audit after an authored
prompt/fixture ambiguity was found. New selection plans require the reviewed
revision-2 collection; old v1 records remain historical evidence.

## Six fixed proposals

The independent CPU recipe
[`solve_positive_module_directions.py`](../../scripts/experiments/gemma4_lora/solve_positive_module_directions.py)
requires paired 42-fold/410-direction collections: correct-answer mean log
probability and correct-minus-foil mean log-probability margin. Both must match
checkpoint attestation, donor, direction inventory, token-frame evidence,
case/code input hashes, computation runtime and microbatch policy. There is no
calibration text, benchmark answer or model invocation in the solver.

All six minimize the continuous weight cost
`sqrt(sum_i (||2 B_i A_i||_F * c_i)^2)`. Projection matrices are disjoint.
Each coefficient is a signed FP32 number with `|c_i| <= 0.25`; total continuous
Frobenius norm is at most 1.0. The coefficient is not rounded to BF16. Persistent
and RAM recipes alone round the actual original-plus-delta weights to BF16 once.

The primary group comprises code, implementation, algorithm, basic science
numerics/contracts, and the four advanced science purposes. Related purposes are
tools, instruction, reasoning, workflow, required calls, non-calls and arguments.
Remaining purposes are general knowledge, lookup, Japanese, referent resolution
and fact retention. Every purpose retains both folds.

| Method | Structural scope | Correct-LL target on primary folds |
|---|---|---|
| balanced | all 410 matrices | +0.010 |
| science | all 410 | +0.020 science; +0.010 other primary |
| algorithm | all 410 | +0.020 algorithm, +0.015 implementation; +0.010 others |
| MLP | gate/up/down matrices | +0.010 |
| attention | Q/K/V/O matrices | +0.010 |
| late | decoder layers 30–59 | +0.010 |

For all methods, correct-LL losses up to 0.005 on related folds and 0.010 on
general folds are explicitly permitted. Separate margin floors permit losses
of 0.005 on primary, 0.010 on related and 0.020 on general folds. Thus each
proposal has 84 explicit lower bounds. These are authored diagnostic units,
not percentage-point score targets or a blanket no-regression requirement.

The six-method CPU stage has a total 120-second budget. It reuses the existing
continuous constrained solver, including box constraints when needed. FP64
solutions are converted to FP32 once and all original constraints and bounds
are rechecked. No clipping, target relaxation, automatic new scopes or inferred
zero for unattempted methods occurs. Numerical nonconvergence or an out-of-bound
result is not a proof of global infeasibility.

## Reproduction and decision boundary

The commands accept paths; no experiment host paths are embedded:

```bash
python scripts/experiments/gemma4_lora/solve_positive_module_directions.py make-plans \
  --margin MARGIN_COLLECTION --correct CORRECT_LL_COLLECTION \
  --donor-spec SCALAR_RECIPE --unit-costs UNIT_COST_AUDIT \
  --calibration-review REVIEW_JSON --calibration-review-sha256 REVIEW_SHA256 \
  --calibration-label reviewed-advanced-v2 \
  --method-review METHOD_REVIEW_JSON --method-review-sha256 METHOD_REVIEW_SHA256 \
  --output FRESH_PLANS_JSON
python scripts/experiments/gemma4_lora/solve_positive_module_directions.py solve \
  --plans FRESH_PLANS_JSON --output FRESH_PROPOSAL_DIRECTORY
```

The calibration and method-review receipts are pinned caller evidence, not an automatic assertion that a
review was adequate or approval was granted. Plan generation itself creates no
checkpoint. An owner reviews the explicit plan before actual application.

The same tool's `audit-old` action reproduces the 34-fold retrospective check
from paired collections, the old margin collection, the saved actual finite
comparison and the original candidate specs. It rejects changed frames or
nonreproduced old margin derivatives.

Every emitted spec remains `continuous_prediction_only`; the six historical
specs are now retired and must not be applied. Any new plan after the corrected
frame oracle requires its own actual BF16 measurement and must report changed values, actual norm, correct-NLL, margin and
endpoint KL on the same frames and verify source restoration. Only generated
task checks and official WSP can establish useful coding gains. All fitted
folds remain calibration data; donor training overlap is unknown.

The five focused CPU fixtures protect paired objective/frame/source identity,
84-row objective ordering and actual output qualification, modified review or
omitted constraints, the NLL sign/old-fold-only audit, and time-budget termination without a fake candidate.

## Anonymous method review

The anonymous design review used the requested Claude Opus 5.5 with no tools,
private candidate measurements or hidden-thought persistence. Its receipt is
pinned alongside the calibration contract review. Useful caveats retained here
are that both folds are fitted, margin and correct-LL can rise together through
a format/temperature effect, and endpoint KL is not whole-generation stability.
The CPU report exposes binding constraints, coefficient bounds and weighted
candidate cosines; nearly collinear balanced/algorithm proposals can avoid
redundant finite replay after an explicit owner decision. Actual foil-logprob
change is available as correct-LL change minus margin change.

The review's suggested many repeats, scale sweeps and null runs are not
automatically added. Existing byte restoration and exact A/A checks remain
evidence; they are distinct from across-case sampling uncertainty. The historical bounded finite plan was prepared but is now retired. A new
plan may prepare bounded finite replay only after the frame oracle, followed by
generated execution and official score tests if justified. The current norm
cap is an experiment trust bound, not a user requirement: any revision needs a
new explicit plan rather than silent clipping or target changes.

## Retirement of the intended proxy

The six retired hypotheses are balanced, science, algorithm, MLP, attention and
late. Their 84-constraint CPU solutions remain records of the calculations that
were performed. They are **untested as weight edits and invalid for the intended
final-answer proxy**; feasibility is not a reason to apply them. The prepared
balanced/algorithm finite plan was not launched.

The root coordinator identified the framing error: the thinking-on flag was
placed in the system template, while the scored continuation began immediately
at the assistant role start. It did not establish that the scored tokens were
in the public final channel after thinking. The official channel contract and a
small GPU oracle must establish the corrected frame before fresh derivatives
can support any new proposal. Existing sources and saved results are not
rewritten to look as though they used that corrected frame.

The old 34-fold prediction/finite comparison in this document is therefore
conditional on that old assistant-start frame. Its observed likelihood losses
and foil-margin changes cannot be restated as measured changes in the intended
public final-answer distribution. That limitation also applies to earlier
finite interpretations based on the same frame.

Completed official WSP runs used their normal benchmark generation and scoring
and remain valid measurements. They do not inherit this analytical frame error.
No code-over-90 result, coding improvement, or final-answer benefit is claimed
from these six proposals. The user’s code-over-90 objective remains unfinished.

A separate retirement receipt records the untouched plan, report and six spec
hashes and the stopped advancement decision. The original numeric artifacts
remain available for diagnosis; no candidate is silently removed or relabeled
inside those historical files.

## Corrected-frame joint native/donor CPU diagnostic

The new forced-empty-thought v2 frame was checked against the official model
forward on three authored cases. Fresh serial native and donor collections
then agreed exactly on every original per-case margin and correct-logprob
(136 cases, 42 fitted folds). This establishes matched numerical conditions,
not equivalence to normal generated thinking. The fresh 42 fold-average
correct-logprob headrooms ranged from 0.5257765 to 23.9091438, so none reached
the predeclared saturation threshold of 0.004.

The separate headroom-aware six-method plan uses
`target=min(tau,0.25*headroom)` for primary correct-logprob, or zero at the
saturation threshold; tau is 0.01, with predeclared 0.02 emphasis scopes.
Primary margins must remain nonnegative; related/general floors preserve the
explicit moderate tradeoffs. The same coefficient box of ±0.25 and continuous
weight L2 radius of 1 are retained. These are experimental bounds, not user
requirements. The CPU solve took 3.03 seconds. Balanced (L2 0.4832525) and MLP
(0.9842717) passed the existing FP32 postcheck and were selected for actual
finite measurement. Science, algorithm, attention and late failed that
numerical postcheck by approximately 1.08e−7, 2.28e−10, 1.13e−10 and 7.26e−10,
respectively. Those small residuals are numerical rejections, **not** proofs
of infeasibility or evidence of worse model ability. No thresholds or saved
results were silently changed to admit them.

A single additional balanced joint relaxation used all 1,331,516 native
coordinates (421 tensors) and 410 donor projection directions. The tensor sets
are disjoint: native raw-weight deltas have unit cost, and donor FP32 gates
have cost equal to their full continuous matrix-direction norm. The existing
minimum-step solver was called once with the same 84 constraints. Native
values were then rounded from the original BF16 values once; donor gates were
rounded only to FP32. This is a hybrid local prediction, not a measured joint
model response, and donor BF16 weight rounding remains unmeasured.

The diagnostic completed in 15.18 seconds. Continuous L2 was 0.0004959151;
after native rounding, 8,369 values changed, native L2 was 0.0004827585 and
donor continuous L2 was 9.32e−7. Hybrid L2 was 0.0004827594. Native rounding
had relative error 0.414 and direction cosine 0.9123. Of the 84 constraints,
26 failed the postcheck; the largest shortfall was 0.0069690 (code fold 0
margin). Several primary correct-logprob gains became approximately
0.0090–0.0095 instead of the requested 0.01. The continuous solution also
retained a 6.64e−8 unscaled residual, so it is not presented as an exact
feasibility or optimality certificate. This single proposal is **not eligible
for joint finite replay**; no automatic repair solve, amplitude sweep, checkpoint,
or model execution was part of that call.

The reusable [joint diagnostic recipe](../../scripts/experiments/gemma4_lora/propose_joint_native_donor.py)
accepts explicit collection and plan paths, verifies source/frame/objective
identity and per-case original values, and saves proposal arrays and a report
to a fresh directory. Rejected proposals remain inspectable. The result does
not prove that native/donor joint edits cannot work, and it does not alter the
separate balanced/MLP donor finite experiment or any official WSP score.

### One explicit frozen-native residual repair

After reviewing that rejection, the owner authorized one separate CPU repair:
freeze those same 8,369 native BF16 changes and solve only the donor gates for
`original_lower_bound - G_native @ frozen_native_delta`. The original 84
constraints, ±0.25 donor box and total L2 radius of 1 remained unchanged.
There was no native rescaling or re-rounding, target relaxation, clipping or
second repair solve. The original rejected report and its source snapshot are
preserved.

The residual solve took 9.73 seconds. It passed the existing postcheck after
FP32 gate conversion: maximum shortfall 2.85e−11, maximum absolute coefficient
0.0324870, donor continuous L2 0.05917080 and combined hybrid L2 0.05917277.
Native L2 remained 0.0004827585 and the native-value byte hash stayed identical.
The predicted code correct-logprob gains were 0.02733 and 0.02351 for the two
fitted folds. These are local forced-empty-thought predictions, **not** an
actual NLL improvement or an official code score. The continuous residual
solve itself retained a 7.24e−11 raw constraint residual, within the stated
1e−10 postcheck tolerance; it is not an exact global optimum certificate.

The tool exports two specs labeled `joint_component_only_prediction`, both
bound to the same report hash and marked `companion_required=true`. They must
be applied together for the proposed experiment. The norm uses disjoint
physical tensors and counts the native stored differences exactly, but the
donor contribution is still continuous: actual donor BF16 rounding, finite
nonlinear interaction, full-model NLL/margin/KL, source restoration, and official
coding performance remain separate measurements. No model or checkpoint was
created by this CPU analysis.

### Broader donor direction and native-only repair sanity check

A separate CPU-only check used the same matched final-frame derivatives. It
examined uniform positive donor coefficients independently of the earlier
L2≤1 trust choice; that radius is not a user requirement.

| Uniform coefficient | Continuous L2 | Primary correct-LL positive folds | Primary mean correct-LL prediction | Primary margin positive folds |
|---:|---:|---:|---:|---:|
| 0.05 | 1.706125 | 18/18 | +0.048195 | 3/18 |
| 0.10 | 3.412250 | 18/18 | +0.096391 | 3/18 |
| 0.25 | 8.530626 | 18/18 | +0.240977 | 3/18 |

The full uniform donor direction norm is 34.1225023. Correct-LL is positive
on 13/14 related folds, but their margins decline on 12/14. At coefficient
0.05, one lookup fold has predicted correct-LL loss 6.3361 and the instruction
purpose mean declines by 0.00621. At 0.25, two predicted absolute log
probabilities exceed zero, an impossible value that demonstrates failure of
that local extrapolation. These are directional predictions, not applied
weights or performance. The positive primary derivative alone does not
justify another unfiltered full merge or an amplitude sweep.

The same CPU call made one native-only residual correction from the initial
joint proposal's native BF16 values, without donor gates. It changed 8,386
native values (1,244 differ from the previous proposal), with actual native L2
0.0005067123. Maximum constraint shortfall fell from 0.0069686 to 0.0063918,
about 8.3%, but 26/84 constraints still failed. Continuous-solve residual was
6.55e−8; native once-rounding remained consequential despite a 0.9951 cosine
to the cumulative continuous delta. No additional repair iteration followed.
The combined CPU analysis took 10.61 seconds and made no model calls.

The more useful immediate experiment is the separately qualified frozen-native
plus donor residual proposal, whose hybrid local norm is 0.05917, followed by
actual generation and official coding measurements if its finite response is
useful. These diagnostics do not establish that tiny targets or this one donor
are sufficient. A stronger target, a different donor direction, or a genuinely
BF16-aware native method would be a new explicit hypothesis, rather than
silently repeating the failed continuous native correction.

## Official dev90 result of the new-frame balanced donor proposal

The balanced candidate completed the unchanged archived-5e3 official WSP
30-minute **dev code subset**: 90 items from all four code sources. Both it and
the original-model reference have 90 scored items, zero unscored items and no
recorded generation errors. This is not the known 64-item target confirmation
set and not a 14-area overall score.

| Official source | Original model | Balanced donor |
|---|---:|---:|
| HumanEval+ | 28/30 | 28/30 |
| MBPP+ | 23/30 | 23/30 |
| LiveCodeBench | 16/20 | 15/20 |
| SciCode, mean per-problem step fraction | 40.301587% | 40.301587% |
| **WSP code, equal source mean** | **72.575397** | **71.325397** |

The standard paired comparison gives **−1.25 percentage points**, 95% bootstrap
CI **[−5.00, +1.6667] pp** (2,000 resamples, existing fixed seed). There were
exactly three changed item scores: `livecodebench:abc395_c` and
`mbpp-plus:Mbpp/641` were lost, while `mbpp-plus:Mbpp/801` was gained. The
MBPP gain/loss cancel in its source score; the LCB loss causes the code decline.
MBPP/801 has varied in earlier runs, so this is not evidence of a newly stable
capability or a causal benefit. The candidate is not adopted and is not being
advanced to known64, full14 or DeepSWE on this result. Code-over-90 remains
unfinished. Positive forced-empty-thought diagnostic averages did not transfer
to an official score improvement.

| Recorded execution | Original | Balanced |
|---|---:|---:|
| Wall seconds | 394.1 | 640.8 |
| Load seconds, included in wall | 7.1 | 7.3 |
| Generation seconds, included in wall | 324.1 | 324.4 |
| Recorded model calls | 119 | 119 |
| Input tokens | 97,703 | 96,775 |
| Output tokens | 336,625 | 340,643 |
| Finish reason `length` | 0 | 0 |

Calls count individual recorded leaves, including SciCode's 39 calls, rather
than its ten aggregate samples; all 119 calls in both runs ended with
`stop`. These are benchmark-recorded calls, not independent HTTP telemetry.
There are no agent/tool steps in this four-source subset; calls and token counts
are not relabeled as such steps. All selected item IDs, manifest SHA, samples
and token caps are unchanged, with native thinking enabled as before.

SciCode passed 19/39 steps in both runs; that pooled count is diagnostic and is
not its official mean-per-problem score. Sanitized saved diagnostics show that
step 35.3 changed from an AssertionError hint after 3.25 seconds to a
**302.56-second step timeout**; both outcomes are false. Step 58.5 changed from
an AssertionError hint to a TypeError hint, also false in both runs. The
unchanged WSP scorer has an eight-worker pool and a 300-second timeout per step;
its environment precheck also has a 300-second limit. This is the same pinned
WSP setting for both models. Exception names are allowlisted stderr hints,
not a causal diagnosis. The timeout is a scored test failure, not an unscored
infrastructure error. No elapsed-time inference, repair, or rerun changed it.

The running candidate's actual container, image, runtime configuration and
16 filesystem-layer digests were captured before exit. They match the saved
original full14 runtime configuration/layers. Both dev90 commands use the
same immutable 5e3 source archive and argument policy. A dedicated historical
original-dev90 container receipt is unavailable, so exact per-run historical
container/image identity is not claimed. Environment secrets, model reasoning,
test bodies and reference implementations are not included in these records.

The stored LiveCodeBench detail for abc395_c exposes only its pass boolean,
so this report does not assert a specific algorithmic or formatting cause.
SciCode 35.3's timeout establishes the observed execution outcome, not its
root-cause code. Generated-code content was not inspected for this report;
no hidden reasoning, reference implementation or test body was used.
