# Preparing normalized coding-priority targets

[prepare_normalized_objective_plan.py](../scripts/experiments/gemma4-purpose/prepare_normalized_objective_plan.py)
prepares one matrix-only plan for the existing [coupled proposal adapter](wsp-native-coupled-objectives.md).
It reads pinned current-parent observations and writes plan.json plus a
preparation receipt. It never runs a solver, candidate, model or benchmark.

    PYTHONPATH=src python scripts/experiments/gemma4-purpose/prepare_normalized_objective_plan.py \
      --source-plan pinned-observation-plan.json --source-plan-sha256 SOURCE_SHA \
      --output-dir fresh-preparation --candidate-id coding-step

The source plan uses the existing schema 2 with eligible_blocks=["matrix"],
matched native/matrix correct-logprob and margin collections, checkpoint
attestation, cases, and declared completed_collection_sha256 payload pins.
The repository defaults to the checkout containing this script; --repo may
name another checkout explicitly. Output must be fresh and disjoint from all
inputs. Captured inputs and checkpoint stats are checked before and after
publication; prior checkpoint full hashes are not presented as fresh scans.

The default core purposes are code, implementation, workflow and algorithm.
Science purposes default to observed names starting with science_. Explicit
--core-purposes and --science-purposes accept nonempty, disjoint subsets of the
observed purpose names. The generated policy records the actual sets; changing
them defines a new experiment, not a replication of a prior candidate.

## Fixed calculation

For each observed LL fold, the tool computes the cost-dual gradient norm using
only eligible sampled O/gate/up/down coordinates:

- cost_i = 1 / max(abs(parent_i), 1e-6).
- norm_j = ||gradient_j / cost||_2.
- headroom_j = -mean(correct_logprob_j), in nats per answer token.
- Core LL target: min(0.25 * norm_j, 0.1 * headroom_j).
- Science LL target: min(0.0625 * norm_j, 0.1 * headroom_j).
- All other LL targets and every margin target: -0.002.

The mean is reconstructed from the saved per-case values; a separately reported
fold mean must agree. Targets retain all rows in objective/purpose/fold order.
Q/K/V gradients can be observed but never enter the selected norm or edits.
The prepared plan keeps relative cost with floor 1e-6, continuous relative box
0.025 and absolute box 1e-6, 16 internal corrections, and a 120-second proposal
budget. It records an actual BF16 weighted-L2 cap of 1, with no automatic
rescale or retry.

**The existing solver does not enforce that global cap.** Its owner must check
the realized proposal and hold any candidate exceeding it. The builder does
not execute the generated plan. Requested headroom caps the lower targets;
it does not cap an eventual linear prediction or actual model response.

The receipt records separate continuous-box, rounded-endpoint and weighted-norm
row bounds. Because BF16 endpoints can lie outside the continuous box, the
necessary upper bound for a realized candidate uses the rounded endpoint and
norm-cap bounds, not the continuous-only bound. Individual row feasibility
does not imply joint feasibility.

The reporting utility is 0.75 times the mean core-purpose LL delta plus 0.25
times the mean science-purpose LL delta. This is not the QP objective.
Related-purpose losses remain separately visible. Raw gradient normalization
expresses sensitivity geometry, not task importance, an unbiased noise
correction or a capability guarantee. No split-half gradients are inferred from
saved fold averages.

## Validation and provenance

This is a portable publication of the executed preparation calculation. The
original runtime and real candidate remain unchanged; no real-data preparation
or model was repeated to publish it. The fixed arithmetic is retained. The
portable boundary adds explicit purpose-set arguments, input/output separation,
a check of the recorded mean against per-case values, and post-publication
source continuity.

One end-to-end fixture reuses the existing tiny mixed observations. A large
diagnostic Q derivative must not inflate the selected norm; another purpose
exercises the headroom cap. The emitted plan then runs through the existing
adapter and tiny BF16 checkpoint save/verify, preserving all native signed-zero
and Q/K/V bytes. Focused CPU, Ruff and mypy checks pass. Real candidate quality
remains a separate finite-response and native-generation measurement.

Whole CPU validation passed once: 512 collected = 460 passed + 52 optional
skips; Ruff, mypy (74 source files), data and 31 shell checks passed, with no
source changes during validation. Evidence:
cpu-integration-normalized-objective-targets-v1/ci-results.json,
SHA 28d7f52667f04d1f85913611fae922414d075733f06f3ee71ac3a69f93acdfad.
The run started after AC finite restoration; it overlapped checkpoint copying
on separate CPU cores with shared storage, so no isolated latency claim is made.
