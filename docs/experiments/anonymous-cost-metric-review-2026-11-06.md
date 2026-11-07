# Anonymous edit-cost method review — 2026-11-06

A second opinion was requested from **claude-opus-5-5** about three general
weight-edit cost metrics. Both the assistant model and usage identity matched
that exact model; there was no fallback and no tool use. The request completed
in 63.08 seconds. The reviewed prompt SHA256 was
`30c150345c6d30ad1846edb396694cbba7c25ed5850f1adc6edb53d304ca2292`.

The prompt contained no experiment identifiers, measured scores, weight
coordinates, problem text, private paths or result files. This was a review of
anonymous mechanisms, **not** a review of actual candidate quality. The
[dispatch tool](../../scripts/claude-method-review.py) retained only the
requested final response and a bounded usage/identity receipt. Raw CLI events,
reasoning blocks and environment values were not persisted.

| Mechanism | Proposed purpose | Main uncertainty | Cheap discriminating evidence |
|---|---|---|---|
| A: absolute Euclidean cost under joint purpose constraints | Find a small joint delta spanning the local purpose gradients | Parameterization dependence, opposing gradients, nonlinear response, BF16 loss | Compare predicted and realized margins after rounding; repeat development generation |
| B: role-wise relative cost with an explicit near-zero floor | Avoid choosing tiny gains only because their absolute changes are cheap | Floor choice, shifted emphasis toward large weights, disappearance below BF16 resolution | Match A's predicted gain; report absolute delta too; check sensitivity to several nearby floor values |
| C: measured branch-output perturbation cost | Price a matrix edit using normalized input-channel activity | Phase/task mismatch, neglected correlations and downstream amplification | Compare predicted energy with actual branch changes on matched inputs before using this cost |

The reviewer recommended solving the combined purpose constraints together,
rounding from the original checkpoint once, and remeasuring the realized set.
A shared component should not be added twice merely because two successful
candidates both contain it. Leave-one-purpose-out re-solves can investigate
interactions; adding proxy gains cannot establish that the benefits compose.

The next bounded analysis uses saved official-chat gradients and compares
absolute and relative costs. Activation-weighted cost remains a hypothesis:
the retained baseline capture has aggregate pre-normalization branch metrics,
not the required per-channel post-normalization input statistics. No edit
direction is inferred from an activation/success correlation.

Two qualifications were retained when interpreting the review. A large dual
multiplier is a sensitivity warning, not proof of mathematical infeasibility.
A diagonal branch-energy metric is also not the full Fisher metric; it omits
input correlations and output curvature. The review does not override the
measured BF16 finite-response results or official benchmark outcomes.

Development gains must be considered alongside repeated baselines and all
losses. The reserved candidate-evaluation cohort described in the
[baseline record](2026-11-06-observed-gemma-baseline.md) is not asserted to be
training-clean or wholly unseen: earlier base evaluations already exist.
Candidate selection must be frozen before using that reserved evidence.

The task retains the prompt and model-verified receipt under
`analysis/resumed-joint-cycle-20261106/claude-cost-metric-review/`.
No new checkpoint or model forward was performed by this review.
