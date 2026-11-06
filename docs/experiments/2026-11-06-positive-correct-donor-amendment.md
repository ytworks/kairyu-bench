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
