# Comparing sparse BF16 delta sets

`scripts/experiments/gemma4-purpose/compare_sparse_delta_sets.py` compares two
through eight already-postchecked proposals against the **same saved parent**.
It reconstructs only their selected union from the pinned native/matrix source
snapshots. It does not read checkpoint payloads or gradients, run a model or
solver, construct a merged vector, or write a candidate.

```sh
PYTHONPATH=src python scripts/experiments/gemma4-purpose/compare_sparse_delta_sets.py \
  --postcheck /absolute/path/numeric-postcheck.json \
  --postcheck-sha256 EXPECTED_SHA256 \
  --output /absolute/path/fresh-delta-geometry.json
```

The input is the existing proposal postcheck's `results` and `input_sha256`
schema, with nested native/matrix collection paths in each proposal plan.
Each row must have status `exact_spec_postchecked`. Plans, specs, selected
source snapshots, implementation source and the postcheck are hashed and checked
for continuity. Checkpoint and collection identities must match across plans;
spec parent and proposal IDs must agree. Reconstructed changed counts and raw
norms must match the postcheck. Output must be fresh and outside checkpoint,
collection and pinned-input paths. No current-incumbent claim is inferred from a
historical snapshot.

For each pair, let `a = BF16(target_a) - BF16(parent)` and similarly `b`,
represented as float64 over the shared coordinate union. The report includes:

- Raw norms, dot product and cosine `dot(a,b)/(||a|| ||b||)`.
- Changed-support intersection/union, Jaccard ratio, exclusive coordinates,
  same/opposite signs and exact opposite cancellation.
- Norms restricted to the intersection and its fraction of each full squared
  norm. The common-support cosine uses those restricted norms; it can be much
  larger than the full-union cosine.
- Identical target words on changed overlaps, delta-difference norm, and the
  same diagnostics by projection family.

Numeric zero and stored-bit changes are separate counts. A zero-norm cosine is
reported as null. BF16 validation and pair arithmetic follow the existing sparse
composition diagnostics, without invoking their composition operation.

## Observed functional-target batch

The six AN–AS proposals share parent AD. The source-pinned final report has SHA256
`18444c93a0e0e4f236d4ce2f8e0d1b5cc19eb9a93f8281dfce211f4ba73ac191`;
all components, pair and family results exactly match the first analysis
(`3a1d5ed82580a1c3b11abaf8cf74d44a60db45428b578e1285af4e0770f2565d`).
The final reader took 7.70 seconds on saved snapshots, with no model, solver or
checkpoint payload read.

| Pair | Shared changed coordinates | Union | Full cosine | Common-support cosine | Opposite signs |
|---|---:|---:|---:|---:|---:|
| AN / AP | 77,022 | 109,904 | 0.74384 | 0.98347 | 2,428 |
| AN / AR | 92,166 | 110,165 | 0.68054 | 0.68799 | 23,184 |
| AP / AR | 71,574 | 110,559 | 0.51378 | 0.66721 | 19,336 |

AP's overlap with AN contains 99.917% of AP's squared raw norm, so AP mostly
repeats AN's MLP direction. AN also spends 42.728% of its squared raw norm in
attention output projections. AN and AR share 83.662% of their support union,
but 83,086 shared coordinates have different absolute target words. These are
weight relationships, not evidence that ability improvements add or cancel.

## Accumulation boundary

Old specs contain **absolute target values**, not additive increments. Applying
an old spec directly to a newly adopted parent can overwrite accepted changes.
All pairs in this batch have conflicting changed-source overlap: the existing
whole-set source-word rebase would reject carrying one unchanged full set onto
another. See [verified sparse composition](wsp-cumulative-sparse-composition.md).

A future cumulative candidate must use the accepted parent's exact bytes and
fresh observations, explicitly resolve overlap, then measure its actual response.
Neither a positive cosine nor disjoint support establishes functional additivity.
The existing composition/rebase fixture checks analytic overlap/cosines, differing
source rejection, and fresh-output protection without a new test method.
