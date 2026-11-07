# F01: fixed joint coding target from AV samples

## Objective and scope

F01 is a new purpose-driven proposal on saved AV, the tied highest code model in the independently checked complete dev10 pair ([audit](2026-11-07-fresh-evidence-audit.md)). Previous candidate ranks, purposes, proposal plans and adoption decisions were not inputs. Existing raw observations were reused after identity validation.

The owner fixed these targets before solving: every algorithm and implementation fold gets +0.03 mean public-answer token log probability and +0.03 correct-minus-foil margin; code folds get +0.01 log probability and margin ≥0; every remaining purpose/fold permits a loss of at most 0.03 in both units. All 19 purposes × two folds × two objectives = 76 constraints remain. These are authored local-response targets, not WSP points.

All 410 observed q/k/v/o/gate/up/down matrices are eligible at their original 512-cell stratified samples: 209,920 coordinates total. Native parameters are fixed. Costs are Euclidean with both explicit block scales one; the continuous coordinate box is max(0.25 × abs(source), 0.0001), with 16 existing BF16 correction rounds and one final rounding from original source + accumulated F64 delta through F32 to BF16. No amplitude search or target relaxation was run.

## Dependency observations

[The complete geometry](evidence-2026-11-07-f01/geometry.json) contains every objective/fold pair cosine for the full sample and each of the seven families, every layer norm and every tensor norm. Each cosine matrix has 76 rows/columns; no top-k filtering determines the eligible support. Of the 2,850 distinct full-sample row pairs, 1,535 have positive cosine and 1,315 negative (minimum −0.707323, maximum 0.949123).

Algorithm and implementation LL gradients have positive cross-fold cosines 0.266590, 0.320571, 0.563880 and 0.544449. Implementation versus the simple code guard LL gradients conflict: −0.130433, −0.311919, −0.129317 and −0.380958. Margin derivatives describe a different relationship: algorithm versus implementation is only 0.040855–0.078604. This supports considering the objectives jointly and retaining all code guard folds. It does not locate a capability causally in any layer or family. All norms and cosines refer to the sampled coordinates, not full matrices or the full model.

## CPU result and every missed target

The single bounded solve completed in 5.88 seconds (120-second ceiling). Its existing selection rule retained correction round 8 after all 16 rounds; the complete history is in [solver-report.json](evidence-2026-11-07-f01/solver-report.json). **67/76 predicted constraints meet their declared bounds; nine do not.** No threshold was changed. The maximum shortfall is 0.015262296 for algorithm margin fold 1. Every −0.03 retention floor is met by the local linear prediction. No finite model response or WSP result is claimed here.

| Purpose | Objective | Fold | Target | BF16-delta prediction | Shortfall |
|---|---|---:|---:|---:|---:|
| code | correct-logprob | 0 | 0.010000 | 0.166499650 | 0.000000000 |
| code | correct-logprob | 1 | 0.010000 | 0.006957469 | 0.003042531 |
| implementation | correct-logprob | 0 | 0.030000 | 0.023549150 | 0.006450850 |
| implementation | correct-logprob | 1 | 0.030000 | 0.023733254 | 0.006266746 |
| algorithm | correct-logprob | 0 | 0.030000 | 0.019167626 | 0.010832374 |
| algorithm | correct-logprob | 1 | 0.030000 | 0.015495971 | 0.014504029 |
| code | margin | 0 | 0.000000 | 0.152237910 | 0.000000000 |
| code | margin | 1 | 0.000000 | 0.018863108 | 0.000000000 |
| implementation | margin | 0 | 0.030000 | 0.014910470 | 0.015089530 |
| implementation | margin | 1 | 0.030000 | 0.018771982 | 0.011228018 |
| algorithm | margin | 0 | 0.030000 | 0.019501704 | 0.010498296 |
| algorithm | margin | 1 | 0.030000 | 0.014737704 | 0.015262296 |

## Exact saved-value proposal verification

The [independent postcheck](evidence-2026-11-07-f01/independent-postcheck.json) reconstructs the spec against current AV and rereads all 209,920 selected checkpoint words. All source words match the stored sample exactly; all targets are finite exact BF16. The reconstructed F32 target byte hash matches the solver, and the maximum difference in all 76 independently recalculated predictions is 5.55e−17.

- Changed elements: 201,426 across all 410 matrices; native changes: zero.
- Actual L2: 0.991527104249. Represented source L2: 7.388781142491. Ratio: 0.134193595009 (**13.41936% of sampled-source L2**, not whole-model L2).
- Maximum absolute coordinate delta: 0.056640625.
- 29,079 realized BF16 deltas exceed the continuous box slightly; maximum excess is 0.00048828125. The existing declared box applies before rounding. This is disclosed, with no post-hoc clamp or retry.

| Family | Sampled cells | Changed cells | Changed matrices | Actual L2 |
|---|---:|---:|---:|---:|
| down | 30,720 | 29,956 | 60 | 0.354711324 |
| gate | 30,720 | 29,611 | 60 | 0.341662605 |
| up | 30,720 | 29,467 | 60 | 0.323944242 |
| k | 30,720 | 29,522 | 60 | 0.459718664 |
| o | 30,720 | 29,166 | 60 | 0.410114283 |
| q | 30,720 | 28,746 | 60 | 0.315789477 |
| v | 25,600 | 24,958 | 50 | 0.395442408 |

The exact emitted [candidate spec](evidence-2026-11-07-f01/candidate-spec.json.gz) is stored compressed without modification. Its uncompressed SHA-256 is `12f4e111a9d31539c6e2d3a94e8270c34b4dbc8a1c4f145fcdcf21ef936eb9d5`; [artifact-hashes.json](evidence-2026-11-07-f01/artifact-hashes.json) hashes every copied artifact. The current candidate is a diagnostic proposal with unmet predicted bounds, not an adopted model.

## Failures and exact-source replay

The first host metadata preview rejected the raw collection because finite_replay.py changed after collection. That rejection is retained in [initial-host-preview-failure.json](evidence-2026-11-07-f01/initial-host-preview-failure.json). The historical bytes were recovered from git fba68c3 and matched the collection SHA `e28f3ebeab3eea32a9685311fd3e937f2b1d9d827ec7cbced1ece57cbca29090`. Neither raw provenance nor shared source was overwritten.

A nonroot bubblewrap namespace mounts the exact historical file read-only at the original collector reference and runs the current solver from a separate read-only source snapshot. Current finite_replay.py remains SHA `71c424bffac775ba112e709632df0a37351a79c3121827be7c9844f321c7c4d3`. Only the new F01 output root is writable; networking is isolated. Namespace proof records UID 1003, distinct host/child network namespaces and an empty route table. The first namespace attempt lacked usable random devices; it failed before Python initialization. Its [receipt](evidence-2026-11-07-f01/namespace-device-failure.json) is retained, and the successful attempt uses a private /dev. A later auxiliary inspection invoked system Python without NumPy and was repeated in the declared CPU environment; it made no experiment changes.

Successful [preparation](evidence-2026-11-07-f01/namespace-prepare.json), [metadata-only preview](evidence-2026-11-07-f01/namespace-preview.json) and [single solve](evidence-2026-11-07-f01/namespace-solve.json) receipts retain argv, bind mounts, code hashes, duration and exit status. The AV checkpoint attestation and current stats match. The whole checkpoint is not rehashed by the proposer; the independent check rereads only the declared sampled words.

## Reproduction and validation

The reusable code is committed with the experiment: [fixed plan and all-fold geometry](../../scripts/experiments/gemma4-purpose/fresh_joint_f01.py), [isolated CPU owner](scripts/fresh-f01-cpu.py), and [independent source/spec postcheck](../../scripts/experiments/gemma4-purpose/fresh_joint_f01_postcheck.py). All three pass Ruff. The real CPU execution and the independent reconstruction are the regression checks for this experiment; no implementation-mirroring tests were added. No GPU, service stop/start, checkpoint save or WSP run occurred in these CPU recipes.

```bash
# R is the task root; REPO is its kairyu-bench worktree.
# Existing artifacts are intentionally not overwritten; use a new task root for a separate experiment.
PY=$R/infra/weight-tools-ci/bin/python
$PY docs/experiments/scripts/fresh-f01-cpu.py prepare --project "$R" --repository "$REPO"
$PY docs/experiments/scripts/fresh-f01-cpu.py preview --project "$R" --repository "$REPO"
$PY docs/experiments/scripts/fresh-f01-cpu.py solve --project "$R" --repository "$REPO"
PYTHONPATH=src "$PY" scripts/experiments/gemma4-purpose/fresh_joint_f01_postcheck.py --root "$R/analysis/fresh-purpose-20261107-f01"
```


## Finite replay preparation

The [GPU6 owner](scripts/fresh-f01-finite.py) prepares one fixed AV → F01 → restored AV replay using all 120 authored cases. A real CPU-only container preview passed exact tokenizer frame equality with the original collection and exact spec validation (Torch 2.14.0+cu130, Transformers 5.20.2, zero CUDA devices and zero model calls). The first preview failed before Python startup because this image exposes `/usr/bin/python3`, not `python`; [v1 failure](evidence-2026-11-07-f01/finite-v1-cpu-preview-failure.json) and the old plan are retained.

The retained v2 preview passed. Revision 3 removed the HOME override, selected only named NVMe cache paths, and added real global-lock owner checks before execution and immediately before stopping the target. Revision 4 additionally binds the saved F01 editor manifest and verify JSON by SHA, the identical spec, and all nine candidate-file stats without rehashing checkpoint payload. Both revisions passed all 120 exact-frame CPU checks. The executed [v4 owner plan](evidence-2026-11-07-f01/finite-v4-owner-plan.json) has SHA-256 `803c98ad8f57e1c491a8fcef9c5aeb83e0d8e8f44032d7448ba4c41a07cd7344`. It pins the root-issued GPU6 sublease, current source snapshot, original frames, AV attestation and unchanged F01 spec. The fixed runtime is one exposed GPU6, CPUs 48–55, 128 GiB container memory, UID/GID 1003, network disabled, serial answers, four threads, seed zero, text SDPA, 2,048 complete-frame tokens, an 88 GiB PyTorch allocator ceiling and one 600-second attempt. Inputs and checkpoint mounts are read-only; outputs, temporary files and caches use NVMe. GPU7 embedding stays running.

The owner holds a separate GPU6 lock, verifies zero active/waiting target requests, and can stop/start only the exact subleased gemma3 container. Its finally path removes its own replay workload, restarts the same serving CID, waits for health and checks every protected service identity. It does not release the gemma0 observation owner's global lock or call the shared gateway. Shared host resource overlap is recorded, so no uncontended timing comparison is claimed. GPU execution requires the exact plan SHA and the parent agent's instruction after saved-checkpoint verification; CPU preparation itself performs no service stop/start.

The Opus review's interpretation of design targets as mandatory adoption gates is distinct from the original proposal plan. The targets remain unchanged; finite sign reversals and retention losses are recorded for the parent decision, with no amplitude or correction-round search in this replay owner.

## Completed one-shot GPU finite replay

The fixed candidate was measured once on GPU6 after explicit parent approval of the v4 plan. All 120 cases completed for AV, F01, and restored AV: 360 case evaluations in total, no missing results. Sweeps took 20.9349, 20.3637, and 20.3677 seconds; the runner took 77.7293 seconds including model loading and byte checks. The owner completed in 197.7722 seconds including exact-CID service stop/start and health recovery. The target restored healthy, all protected snapshots matched exactly, and no other service was changed.

The finite response meets 66/76 fixed design bounds and misses 10: all eight algorithm/implementation +0.03 targets remain below target, and referent fold 1 violates both retention bounds. There are no sign reversals among the eight primary targets. The authored case margins have 80 gains and 40 losses; these are proxy measurements, not WSP correctness or ability scores. All 76 rows and 120 individual margins are retained in [numeric analysis](evidence-2026-11-07-f01/finite-v4-numeric-analysis.json), with case IDs available in the [original matched frame evidence](evidence-2026-11-07-f01/finite-v4-replay-frame-evidence.json).

| Purpose | Fold | LL prediction | Finite LL change | Margin prediction | Finite margin change |
|---|---:|---:|---:|---:|---:|
| algorithm | 0 | 0.019167626 | 0.019787759 | 0.019501704 | 0.015522406 |
| algorithm | 1 | 0.015495971 | 0.005739287 | 0.014737704 | 0.006088838 |
| implementation | 0 | 0.023549150 | 0.024966121 | 0.014910470 | 0.009043992 |
| implementation | 1 | 0.023733254 | 0.013758644 | 0.018771982 | 0.018870622 |
| code | 0 | 0.166499650 | 0.328002269 | 0.152237910 | 1.042012024 |
| code | 1 | 0.006957469 | 0.155345152 | 0.018863108 | 0.212701416 |
| referent | 0 | 0.004030239 | 0.020841757 | 0.001491394 | 0.030655702 |
| referent | 1 | -0.001703216 | -0.037517150 | -0.003620151 | -0.032255729 |

Referent fold 1 loses 0.03751715 nat/token in correct-answer LL and 0.03225573 in margin. Its linear predictions were only −0.00170322 and −0.00362015. Thus the full finite observation detects a retention violation missed by local linear predictions; primary positive directions alone do not justify adopting F01. No amplitude, correction-round or candidate retry was performed after seeing this result.

All 201,426 selected edited BF16 values were checked by CPU readback inside the model process against the declared spec. Restored AV selected bytes match exactly; all 120 margins, 38 fold LL means and KL values match the original AV sweep with maximum errors zero. This is a selected-coordinate byte proof, not a full RAM-model hash. The saved F01 is linked by the identical spec, the editor manifest/verify SHA and nine saved-file stats; the editor independently reconstructed every declared edit and checked all nine checkpoint file hashes.

The [completion receipt](evidence-2026-11-07-f01/finite-v4-completion.json), [runtime provenance](evidence-2026-11-07-f01/finite-v4-replay-provenance.json), full numeric replay outputs and byte proofs are included. [The summary recipe](../../scripts/experiments/gemma4-purpose/fresh_joint_f01_summary.py) independently checks all fold means from 120 case margins and joins every finite result with the unchanged plan and prediction. One auxiliary summary had an incorrectly escaped terminal newline; it is retained as a text failure artifact, then corrected and regenerated without rerunning the model. Both the owner and summary recipe pass Ruff. The earlier v1 CPU startup failure and v2/v3 previews remain in the evidence directory.
