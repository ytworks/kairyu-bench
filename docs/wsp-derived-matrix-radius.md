# A derived BF16 radius for sampled matrix proposals

[derive_matrix_radius.py](../scripts/experiments/gemma4-purpose/derive_matrix_radius.py)
publishes the two CPU stages used by the matrix-only dependency experiment. It
reuses the [coupled adapter](wsp-native-coupled-objectives.md), existing
minimum-step solver and BF16 realization. It never loads a model or modifies a
checkpoint.

The explicit matrix-only hypothesis freezes accepted native parameters and
diagnostic Q/K/V columns. It permits sampled attention output coordinates and
complete sampled MLP gate/up/down sets. The optional exact matrix_tensors mask
is applied through the same adapter.matrix_column_indices selector used by the
BF16 proposal; an excluded matrix cannot influence the continuous step or box. This structural scope does not
guarantee that every permitted matrix receives a nonzero realized change.

## Two explicit stages

Preview validates metadata and pins without reading gradient arrays or creating
the output directory. Each run requires its own fresh directory.

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/derive_matrix_radius.py \
      --plan calibration-plan.json --plan-sha256 PLAN_SHA --output fresh-calibration

Add --run for the single continuous solve. Its candidate-plan.json retains the
same checkpoint, observations, constraints and cost, with the derived radius and
a SHA-pinned analytic report. Use that emitted plan for the second stage:

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/derive_matrix_radius.py \
      --plan fresh-calibration/candidate-plan.json --plan-sha256 DERIVED_PLAN_SHA \
      --output fresh-candidate --phase candidate --run

The second stage invokes the existing adapter once using the current Python
interpreter. It emits the usual exact BF16 spec and a realized-box audit. There
is no automatic retry, coefficient grid, model measurement or adoption.

## Plan and calculation

The underlying plan is the adapter's schema 2 with eligible_blocks=["matrix"],
one candidate, complete matched native/matrix correct-logprob and margin
collections, and both block scales 1. The legacy cost is relative with floor
1e-6. An explicit analytic_calibration.metric="euclidean" opts into a candidate
cost_metric of {"type":"euclidean","block_scales":{"native":1,"matrix":1}}.
Omitting metric retains the legacy relative policy. The
reference_plan object supplies path and SHA of the previously fixed plan;
the old prior_aa_plan spelling is also accepted for replay of archived inputs.
Checkpoint, cases, optional parent state, collections, baseline tolerance,
constraints, cost and effective matrix mask must equal that reference. A
reference with a wider or different exact tensor scope is rejected before
reading gradients.

The plan also supplies completed_collection_sha256 and
completed_collection_stat, keyed by absolute input filename. Each stat has
device, inode, size, mtime_ns and ctime_ns. Used matrix payloads must match the
declared hashes; completed-input stats, all captured metadata and numerical
source hashes are checked before and after execution. The candidate additionally
requires the analytic report to contain the same captured observation/reference
and source pins; matching only the checkpoint and numerical bounds is insufficient.
Optional fixed_method_inputs
adds pinned method receipts without requiring a particular host or study.

The existing analytic_calibration policy remains fixed to the following values;
only the optional metric field ("relative" or "euclidean") is added:

~~~json
{
  "calibration_seconds": 120,
  "candidate_seconds": 180,
  "fractional_buffer": 0.0078125,
  "floor": 0.000001,
  "dual_tolerance": 0.0000000002,
  "maximum_iterations": 20000
}
~~~

For incumbent coordinates b and the one continuous minimum-cost step d, the
published method computes:

- scale_i = max(abs(b_i), 1e-6).
- cost_i = 1 / scale_i for relative cost, or 1 for explicit Euclidean cost.
- rho = max_i(abs(d_i) / scale_i) + 1/128.
- absolute_radius = max(1e-6, rho * 1e-6).
- radius_i = max(rho * abs(b_i), absolute_radius).

The absolute floor makes the continuous solution fit even at zero or tiny
incumbent values. The buffer is a fixed fraction; it is **not a guaranteed BF16
ULP** at arbitrary final magnitudes. The existing numerical convergence
contract and finite-value checks must pass. A failed solver contract is not
reported as mathematical infeasibility.

The report retains the declared cost, exact eligible tensor names, observed and
active column counts, continuous norms, all predictions, the derived box, and
per-row necessary upper bounds from the two realized BF16 endpoints. These row
bounds are not jointly sufficient. The candidate uses at most 16 existing
internal BF16 corrections, with every realization anchored to the incumbent.
Actual values can round outside the continuous box; the audit reports any
excess instead of silently clipping it. Candidate replay binds the analytic
report to the same cost and exact matrix mask, in addition to its source pins.

A large derived radius weakens the local-gradient interpretation even when the
aggregate absolute norm looks small. Predicted feasibility is not finite
response or execution success. The forced-empty-thought public-answer objective
also differs from normal native-thinking generation. Any adoption requires
separate evidence under the task's actual protocol.

## Publication and validation

The portable file preserves the executed method composition; host paths, UID,
the Python path and fixed model-sized coordinate counts were removed. The
already executed runtime and private receipts remain unchanged. Publishing this
tool did not rerun the real calibration or model.

One end-to-end CPU fixture reuses the existing tiny mixed collection: metadata
preview, continuous calibration, derived plan, quantized candidate, existing
editor save/verify, unchanged native signed-zero/QKV bytes and source hashes.
The same fixture protects the previously observed ownership bug: rejecting an
existing destination must not write a failure receipt into another writer's
directory. A mismatched analytic observation/frame binding is also rejected
before a quantized proposal starts. The focused fixture, Ruff and mypy pass. Private weights, arrays,
plans and generated specs are not included in the repository.

Historical initial-publication CPU validation completed once: 511 collected, 459 passed and 52 optional
dependency skips; Ruff, mypy over 74 source files, data checks and 31 shell syntax
checks passed. Numerical source hashes were unchanged during the checks. The
new script also passed its focused mypy check. The CI evidence is
cpu-integration-derived-matrix-radius-v1/ci-results.json, SHA
a56b4a432b151d9b1c305001ffb01eec290dd4f2fdb582d72fd7509c9d05bd13.

The same end-to-end fixture additionally exercises an exact one-matrix mask
with a much stronger excluded matrix gradient, Euclidean cost at source value
2, the derived plan through BF16 proposal and actual saved-byte verification,
and the functional raw-cap postchecker. It rejects a mismatched reference mask
and preserves every excluded/native/QKV byte. This extends the existing test;
no static registry tests or real candidate/model calculations are added. Large
derived radii remain visible without an arbitrary box cap.
