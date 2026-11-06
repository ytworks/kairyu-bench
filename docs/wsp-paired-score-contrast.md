# Score-only paired linear contrasts

[paired-score-contrast.py](../scripts/experiments/paired-score-contrast.py) computes an
exploratory finite linear contrast across matched arms using one shared bootstrap
row draw inside each source. It reads only an explicitly SHA-pinned, previously
approved score-only JSON export. It never opens run directories, answers, prompts,
trajectories, or a model endpoint, and never changes official scoring.

For a base/adapter factorial, use coefficients O=1, L=-1, P=-1, M=1 to estimate
(M−P)−(L−O). Use the same exported four-arm rows with O=-1, L=0, P=0, M=1 for M−O.
The primary contrast and coefficients should be fixed before inspecting its result.
A post-hoc application must be labeled as such and cannot become a selection guarantee.

## Input and invocation

The exact top-level fields are schema_version=1, kind="paired-score-export",
data_kind="recorded" or "synthetic", source_artifact_sha256, arms, and rows.
The source artifact digest is a caller provenance declaration; that upstream file
is not opened or independently verified. The mandatory --input-sha256 binds the
actual approved export bytes. Labels do not prove that a dataset was approved or unseen.

Each row contains only id, source, and scores, for example:

    {"id":"scicode:example","source":"scicode","scores":{"O":0.5,"L":0.75,"P":0.5,"M":1.0}}

Every row must have exactly every declared arm (2–16 unique labels). Duplicate IDs,
including duplicates assigned to another source, duplicate JSON keys, missing/null
scores, nonfinite values, booleans, extra arms and body fields are refused.
There is no intersection selection and no imputation. Partial SciCode values remain
numeric. The caller must preserve the upstream item order within each source.
Input has a 16 MiB limit; requested resamples must be 2–10,000 and total sampled rows
cannot exceed 20 million. These are resource bounds, not model-adoption gates.

    python scripts/experiments/paired-score-contrast.py \
      --input approved-score-export.json --input-sha256 EXACT_SHA256 \
      --coefficient O=1 --coefficient L=-1 --coefficient P=-1 --coefficient M=1 \
      --seed 20261104 --resamples 2000 --output fresh-contrast.json

Declare every coefficient, including zeros. Finite nonzero-sum combinations are
allowed, but they measure a weighted level rather than a difference; coefficient_sum
is retained. Existing outputs (including broken symlinks) are refused. Input SHA/stat
are checked again after computation; a complete JSON result is published exclusively.
No network, Docker, benchmark, scoring or model call is made.

## Weighting and uncertainty

Compute each arm's mean within each source, average those source means equally,
then apply the coefficients. Thus a 10-item source and a 30-item source have the same
weight. This matches a single WSP area's source weighting; combining different areas
this way is not the official WSP overall score.

For every bootstrap iteration, visit source names in sorted order, draw len(source)
indices with Python random.Random(seed).randrange, and take the complete multi-arm
row at each index. All arms use exactly the same indices, preserving paired covariance.
The script does not add variances or intervals from independent pair comparisons.

The default seed/resample count and percentile floor/ceil endpoint definition match
wsp.report.compare. An ordinary two-arm contrast is checked against that official
function on synthetic run metadata with fractional scores and nonsorted row order.
Same input row order and source strata are necessary for matching draws; floating
aggregation can differ at rounding precision. CI endpoints are tested within 1e-14.
Synthetic test fixtures do not contain benchmark bodies.

Output retains coefficients, source counts/contrasts/CIs, arm source-equal means,
seed/resamples, ordering, input SHA/stat and implementation SHA. It does not retain
resampled traces or expand raw model content. A linear interaction is protocol- and
selection-conditional, not proof of true training-base identity, a causal mechanism,
cross-problem independence, or expected-score improvement. One generation per arm
does not separately measure generation churn, and reused dev items are not unseen.

## Validation and existing-data exercise

Four CPU contract tests cover joint covariance and unequal source sizes; ordinary
pair parity with the existing official comparator; duplicate/missing/nonfinite/body
rejection; and input continuity/exclusive publication. Focused tests, Ruff and mypy
passed. An independent read-only review found no blocker.

The subsequent whole-repository CPU check collected 503 tests: 451 passed and 52
optional-dependency tests were skipped, with no failures or errors. Whole-repository
Ruff and mypy (73 source files) passed; checked source hashes were unchanged.
This is tool validation, not model-quality evidence. The receipt is
analysis/code90-resume-20261106/cpu-integration-paired-score-contrast-v1/ci-results.json,
SHA256 1a55699c87f154e0f5bd2521eceebddab7c71f4f5ce8b06de7539ac2fa7964cf.

A post-hoc exercise on the already approved K four-arm numeric export took 0.180 s:
K−B−H+O = +0.833333 percentage points, 95% interval [−8.333333, +10.416667].
It uses no new scoring or model calls and does not change the prior non-adoption
decision. The saved export order is the numeric artifact's order; no claim is made
that its order was independently verified against the original run records.
This exercise is not a preregistered M result.

The receipt is analysis/code90-resume-20261106/paired-score-contrast-validation-v1/actual-validation.json,
SHA256 f172037bbb7f22753b1291db5ff52a30d3aeb8cd60ae896146c86cd437002a81.
Input export SHA256 0619358e413711ef81e423a2797fba0a67d5270c830a06941ed1123337704087;
result SHA256 4a75ef7214416f7a46766b63f90530b37ce5d9193d27617eae15edf0d87301a0.
