# 2026-11-06: full-projection donor direction observations

This is an authored-calibration diagnostic, not a WSP score, candidate adoption,
or an improvement claim. No benchmark answer, reference patch, hidden reasoning
or confirmation outcome was used in this analysis.

The [read-only alignment tool](../../scripts/experiments/gemma4_lora/donor_gradient_alignment.py)
first joined 209,920 sampled matrix coordinates with the existing c=0.25 BF16
donor checkpoint. Those cells cover only 0.0007168% of the 29,286,727,680 text
projection elements. Both science objectives had opposite signs across their
two folds. The sampled dot product must not be extrapolated to the full donor.

The [direction collector](../../scripts/experiments/gemma4_lora/donor_direction.py)
then measured each of the 410 complete rank-16 projection directions on the
original BF16 source. It uses zero scalar gates and a first-order custom VJP
for `y + gate * 2B(Ax)`. The original source and factors are frozen. Its FP32
direction arithmetic and BF16 source computation are documented in the
[recipe guide](../../scripts/experiments/gemma4_lora/README.md).
The coefficient is not a BF16 model parameter and must not be rounded as one.

## Measured scope and zero-gate checks

- Science-only: 4 folds, 16 authored cases; collection 9.714 seconds after model
  load, peak GPU allocation 73.49 GiB, peak reservation 76.17 GiB.
- Broader preservation set: 17 purposes, 34 folds, 120 authored cases; collection
  52.98 seconds after model load, peak allocation 88.63 GiB, reservation 94.26 GiB.
- Both runs covered all 410 projections across 60 layers and compared every
  microbatch against the unmodified model. Every zero-gate public-answer margin
  was exactly equal; maximum absolute difference was zero.
- Prefix plus the longer public answer ranged from 21 to 505 tokens over the
  broader set (median 67.5, 95th percentile 321.65). No case exceeded 512 tokens.
- The unit continuous weight cost of moving all gates by one is
  `sqrt(sum(||2B_i A_i||_F^2)) = 34.1225023009`, agreeing with the prior
  low-rank Gram audit. This is not the realized BF16 delta norm.

The saved-run roots are `analysis/code90-resume-20261106/science-gpu-runs/
donor-direction-full` and `donor-direction-broad` beneath the experiment root.
The two collections preserve their original source SHA records. A later
preflight-only fix captures adapter spec bytes before parsing and hashing them;
it closes a provenance race and does not alter the VJP or reinterpret old runs.

## Full-direction findings

The following values are derivatives for a uniform positive unit gate across
all 410 matrices. They are local continuous public-answer margins per gate,
not percentages, finite responses or scores.

| Purpose | Fold 0 | Fold 1 | Modules positive in both folds |
|---|---:|---:|---:|
| code | +0.230146 | -0.258861 | 124 |
| implementation | -0.398514 | -0.553495 | 89 |
| algorithm | -0.407460 | -0.062266 | 96 |
| science numerics | -0.109521 | -0.260002 | 83 |
| science contracts | -0.017487 | -0.268761 | 64 |
| tool arguments | -0.967254 | -0.587523 | 73 |
| tool call | -3.797619 | -1.376916 | 66 |
| tool no-call | -0.278658 | -1.151960 | 122 |
| workflow | -0.583686 | -1.756295 | 69 |
| Japanese | +1.509491 | +2.254536 | 103 |
| fact retention | +0.358555 | +0.132022 | 112 |

All 410 individual directions have mixed signs somewhere among the 34 folds.
No single positive or negative module direction improves all recorded folds.
In the four science folds alone, 20 directions are consistently positive and
60 consistently negative. The MLP group is positive in both simple-code folds
(+0.189325, +0.126127), but both attention and MLP groups are negative in the
implementation, algorithm and science folds. Thus a uniform donor coefficient
is not supported for the coding/science objective by these local observations.

After normalizing each coordinate derivative by its continuous weight cost,
the equal-fold mean science-numerics direction has cosine 0.791 with
implementation, 0.730 with tool arguments, and 0.380 with simple code.
Science-contracts has corresponding cosines 0.631, 0.682 and 0.311.
These describe alignment in the chosen direction space, not causal ability
localization or statistical generalization.

For the ten primary folds (code, implementation, algorithm and the two science
purposes), zero modules have a positive sign in every fold; nine have a negative
sign in every fold. Reversing those directions is a hypothesis, not an adopted
edit. For example, reversed L59 K has worst-primary derivative per unit weight
cost +0.00991, but a reasoning fold predicts -0.18529 and an instruction fold
-0.14692 in the same units. Compensation and explicit preservation tradeoffs
are necessary; selecting that module alone is not justified.

The [CPU profile tool](../../scripts/experiments/gemma4_lora/profile_donor_directions.py)
reproduces every fold's signs, layer/family/attention/MLP profile, cost-normalized
gradient cosine and descriptive common-positive ranking from saved arrays.
It does not read case text, select coefficients or call a model.

## Proposed next implementation; not executed

The existing scalar recipe accepts one FP32 coefficient in (0,1] for all 410
matrices. It has no module filter, negative coefficient, or per-module map.
Keep that implementation and its historical manifests unchanged.

A new, versioned per-module wrapper would:

1. Pin the original checkpoint, donor/config, exact 410-module inventory,
   continuous-gradient collection, explicit coefficient map and implementation.
   Record all 410 signed FP32 coefficients, including zeros; forbid omitted
   defaults. Only reviewed finite bounds are allowed.
2. Reuse the existing deterministic block primitive: FP32 `2 * (B @ A)`,
   multiply by the module's FP32 coefficient, add to original BF16 values
   converted to FP32, then round model weights to BF16 once. Zero coefficients
   preserve source bytes, including signed zero.
3. Keep edited shards private, preserve source and all untouched bytes, use
   exclusive manifest publication, and recheck inputs before completion.
   Record per-module coefficient, actual changed count/L2 and source/target
   tensor hashes. Verification regenerates exact bytes from the original source.
4. Prove in a tiny fixture that an all-0.25 map produces the same bytes as the
   existing scalar recipe. Also protect signed/zero coefficients, scope
   mismatch, external hardlinks, source preservation and failed publication.
   No full checkpoint is created by proposal or solver commands.

The CPU solve can reuse only `joint.minimum_step(G, lower, unit_cost)`:
minimize `sum(cost_i^2 * c_i^2)` subject to the 34 explicit fold bounds.
Do **not** use `joint.quantized_proposal` or `bf16_realization` on the gates.
Convert solved coefficients to FP32 and recheck constraints in their original
units. A nonconvergent solve is not a proof of infeasibility. A solution outside
the reviewed coefficient/L2 trust bounds is rejected, not silently clipped.
The existing finite-response signed-endpoint solver is limited to eight
direction pairs and is not the appropriate 410-coordinate interface.

Six distinct, pre-review scopes are proposed below. Every row produces at most
one joint set, with all 34 fold bounds present; it is not an amplitude sweep.

| Scope | Allowed directions | Primary emphasis |
|---|---|---|
| balanced | all 410, signed | code + science + implementation + algorithm |
| scientific contracts | all 410, signed | entrypoint/API/shape contracts while retaining numerical/code gains |
| numerical algorithms | all 410, signed | numerical science and algorithm implementation |
| MLP implementation | all MLP projections, signed | code and implementation; test whether attention changes are necessary |
| attention interfaces | all attention projections, signed | implementation and tool arguments with science/code gains |
| late-layer compensation | all projections in layers 30–59, signed | test the late-layer signal while bounding earlier-purpose disruption |

Illustrative bounds for review, not executed settings: balanced primary gains
+0.005 nats per answer token, science +0.01; emphasize one purpose at +0.02
while retaining other primary floors of +0.002 to +0.005. Related
tools/instruction/reasoning can allow -0.005 and other preservation purposes
-0.01 per fold. Proposed trust limits are max absolute module coefficient 0.25
and total continuous weight L2 1.0. These numbers require explicit method review
before solving and are not accuracy guarantees. The objective minimizes total
weight change; a module count cap is not substituted for that objective.

A successful continuous solve still needs actual BF16 realization, the matched
finite check, then official WSP. Source-point compatibility with native edits
such as h0397 must be explicit: structurally disjoint tensors do not guarantee
nonlinear additivity. Donor training-base and training/evaluation overlap remain
unknown. No new solver, per-module recipe, candidate, full checkpoint, extra
benchmark or adoption was performed as part of this proposal.
