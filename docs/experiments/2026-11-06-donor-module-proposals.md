# Signed module donor proposals, 2026-11-06

Four of six fixed structural plans produced bounded **continuous** proposals.
No official score is inferred from these vectors. They were fitted to all 34
authored calibration folds; those folds are not held out. Actual BF16 finite
measurements and official evaluation remain separate steps.

The source is the original Gemma checkpoint. The pinned rank-16 donor and its
unknown training-base/data-overlap limitations are described in
[the direction observations](2026-11-06-donor-direction-observations.md).
The implementation and commands are in the
[LoRA recipe guide](../../scripts/experiments/gemma4_lora/README.md#signed-module-coefficients-and-fixed-continuous-plans).

## Fixed solve and boundaries

Each module coefficient multiplies a complete `2BA` direction. The continuous
cost is `sqrt(sum((coefficient * unit_direction_l2)**2))`, because the 410
matrices occupy disjoint parameters. A unit coefficient on every matrix has
norm **34.1225023009006**. Limits were fixed at absolute coefficient **0.25**
and total continuous norm **1.0**. Coefficients are serialized once to FP32,
not BF16. Actual weight rounding can change both the norm and response.

All 34 fold constraints are retained in each plan. The five columns below
apply independently to both folds. Related tools/instruction/reasoning/workflow
objectives allow −0.005; other preservation objectives allow −0.01. The
attention plan additionally requests tool arguments +0.005 and tool call /
non-call at least zero.

| Plan | Scope | Code | Implementation | Algorithm | Science numerical | Science contracts |
|---|---|---:|---:|---:|---:|---:|
| Balanced | All 410 | .005 | .005 | .005 | .01 | .01 |
| Scientific contracts | All 410 | .002 | .005 | .002 | .005 | .02 |
| Numerical algorithms | All 410 | .002 | .005 | .01 | .02 | .005 |
| MLP implementation | 180 MLP projections | .01 | .005 | .002 | .005 | .005 |
| Attention interfaces | 230 attention projections | .005 | .01 | .002 | .005 | .005 |
| Late compensation | 205 projections, layers 30–59 | .005 | .005 | .005 | .01 | .01 |

The first minimum-norm solutions did not incorporate the coefficient box.
Those diagnostics were retained. For solutions within the norm budget but
outside the coefficient box, positive/negative identity constraints were added
to the same QP and solved again. This redistributed changes; coefficients were
not clipped and goals were not weakened. A predeclared 1e−7 feasibility margin
was used, followed by checks against the original constraints after FP32
serialization.

| Plan | Initial norm | Initial max absolute coefficient | Bounded norm | Outcome |
|---|---:|---:|---:|---|
| Balanced | .651310 | .639211 | .732324 | Continuous proposal |
| Scientific contracts | 1.245885 | 1.114441 | — | Outside norm budget; no spec |
| Numerical algorithms | .733127 | .597865 | .837912 | Continuous proposal |
| MLP implementation | 1.002335 | .228879 | — | Outside norm budget; no spec |
| Attention interfaces | .412646 | .319755 | .415535 | Continuous proposal |
| Late compensation | .728155 | .672988 | .846352 | Continuous proposal |

The two excluded numerical solutions are reported as outside budget. They are
not a general impossibility claim about scientific or MLP editing. Numerical
dual residuals were nonzero (about 1e−10). All four emitted specs retain exactly
410 keys, including zeros outside structural scopes.

## Vector diagnostics

Shares below refer to **squared weight norm**, not capability contribution.

| Proposal | Active / negative modules | Negative fraction of active | Negative norm² share | Norm / full-unit donor | Active box constraints |
|---|---:|---:|---:|---:|---:|
| Balanced | 410 / 244 | 59.51% | 67.75% | 2.1462% | 2 |
| Numerical algorithms | 410 / 237 | 57.80% | 79.18% | 2.4556% | 2 |
| Attention interfaces | 230 / 123 | 53.48% | 67.03% | 1.2178% | 1 |
| Late compensation | 205 / 130 | 63.41% | 70.59% | 2.4803% | 2 |

The maximum absolute serialized coefficient is .24999989569187164 for all
four. Layer 59 K reaches the negative box in all four; layer 58 O also does so
in balanced, numerical and late. Their unit-direction norms are .472757 and
.877551 respectively: large coefficient magnitude alone does not imply a large
total weight change.

| Proposal | Q/K norm² | V/O norm² | MLP norm² | Largest family shares | Largest layer shares |
|---|---:|---:|---:|---|---|
| Balanced | 36.25% | 35.66% | 28.09% | K25.38%, V18.62%, O17.04% | L58 22.54%, L57 11.88%, L59 11.36% |
| Numerical algorithms | 29.76% | 41.09% | 29.16% | O22.27%, K21.52%, V18.82% | L58 15.16%, L59 12.89%, L57 5.86% |
| Attention interfaces | 52.68% | 47.32% | 0% | K41.19%, O25.88%, V21.44% | L58 18.54%, L59 8.84%, L57 8.54% |
| Late compensation | 35.76% | 28.92% | 35.32% | K26.17%, V15.81%, down14.20% | L58 20.23%, L57 14.52%, L59 13.67% |

Weight-space cosines use `coefficient * unit_direction_l2` vectors, rather than
unweighted coefficient cosine:

| | Balanced | Numerical | Attention | Late |
|---|---:|---:|---:|---:|
| Balanced | 1 | .716884 | .737610 | .868817 |
| Numerical | .716884 | 1 | .506610 | .634919 |
| Attention | .737610 | .506610 | 1 | .559491 |
| Late | .868817 | .634919 | .559491 | 1 |

Constraints within 1e−6 of their lower bound after serialization:

- Balanced: reasoning fold0, non-call folds0/1, algorithm fold1, and both folds
  of each science objective.
- Numerical: instruction fold1, non-call folds0/1, algorithm fold1, numerical
  folds0/1 and contracts fold0.
- Attention: instruction fold1, reasoning fold0, non-call folds0/1, numerical
  fold0 and contracts folds0/1.
- Late: code fold0, instruction fold1, reasoning fold0, Japanese fold0,
  non-call folds0/1, numerical fold0 and contracts folds0/1.

These binding constraints expose fitted tradeoffs, not new measured losses.
For example, late compensation uses the permitted Japanese decrease in its
local prediction; official performance must be measured before accepting it.

## Evidence and reproduction

Experiment-relative artifacts:

- `donor-module-six-plans.json`: plan SHA
  `5c81ac54db7797c537dbf7df563863c08088aa8e5befc3d2943477f5302679de`.
- `donor-module-six-solve/report.json`: preserved initial solutions.
- `donor-module-six-bounded/report.json`: final continuous report SHA
  `625968eadefedebc14ce6e0831c709aad88fdac3bd7a2dd091c7d8415c8f7593`.
- `donor-module-vector-diagnostics.json`: signs, every binding constraint,
  family/layer shares and weighted cosine.
- `donor-module-validation.json`: six CPU regression checks, Ruff and scoped
  mypy. Tests include old-scalar 0.25 byte equivalence, negative-zero
  preservation, signed rounding, box redistribution, ownership and exclusive
  publication. No model or benchmark calls were made by these checks.

Given the pinned collection and emitted specs, the core diagnostic calculation
is reproducible without reading any case text or benchmark result:

```python
# entries are summary["coverage"]["entries"]; cost is unit-gate-weight-l2.npy.
c = np.array([spec["module_coefficients"][e["tensor"]] for e in entries])
weighted = c * cost
norm = np.linalg.norm(weighted)
negative_fraction = np.count_nonzero(c < 0) / np.count_nonzero(c)
negative_norm_share = np.sum(weighted[c < 0] ** 2) / norm**2
ratio_to_unit_donor = norm / np.linalg.norm(cost)
# Group norm² shares sum weighted[i]**2 over explicit layer/family membership.
# Pairwise cosine is dot(w1, w2) / (norm(w1) * norm(w2)).
# Binding fits have predicted_response - minimum_delta <= 1e-6.
```

The next measurement is one matched base/four-candidate/restored-base RAM
sequence using actual BF16 bytes and all 34 public-answer folds. Margin,
correct-answer NLL, endpoint full-vocabulary KL, actual changed elements and
actual weight L2 must all be reported, including regressions. The fitted proxy
does not replace the official four-source WSP coding score.
