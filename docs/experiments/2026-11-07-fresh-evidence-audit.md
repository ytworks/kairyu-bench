# Fresh evidence audit — 2026-11-07

This audit restarts from the coding objective and the saved measurements. It does not inherit a prior hypothesis, candidate ordering or adoption decision. No inference service was changed and no model call was made for this audit.

## Coverage and method

The snapshot at 2026-11-07T04:10:19.615714+00:00 includes **151 unique WSP runs**: 60 from the legacy result root and 91 from the task result root. Statuses are preserved verbatim: 146 completed, four partial and one historical running marker. Completed subset runs are not relabelled as canonical full WSP runs. The complete raw result directories remain in place.

- [All 151 runs, numeric scores, input SHA-256 and status](evidence-2026-11-07-fresh-audit/runs.json)
- [Compact all-run table](evidence-2026-11-07-fresh-audit/runs.tsv)
- [Every stored code item across all 151 runs](evidence-2026-11-07-fresh-audit/code-items.json)
- [All 778 paired AD/AV item scores](evidence-2026-11-07-fresh-audit/ad-av-items.json)
- [Summary and limitations](evidence-2026-11-07-fresh-audit/summary.json)
- [Pair receipts and agent steps](evidence-2026-11-07-fresh-audit/provenance.json)

The one-time CPU extraction reads run.json, report.json and items.jsonl, hashes their exact bytes, and exports only numeric/status/identifier fields. For each reported coding source it independently averages the stored non-null item scores, then averages the source scores. Both levels match the saved reports within 1e-12 in every checked run. Missing scores remain null. A canonical complete run requires the original manifest hash, exact selected ID set, unique IDs, matching row/declaration counts, completed status and no missing score. There are no score-consistency errors. This is an evidence audit, not a benchmark rerun or a new scoring implementation.

The canonical manifest SHA-256 is `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`. A 14-key `areas` object does **not** establish full coverage because unmeasured areas have null values. In particular, the numerous 90-item dev30 code-only runs are exploratory subsets of the 2,338-item tier, even when their run status is completed. They cannot replace full WSP.

| Tier | Dev complete canonical runs | Test complete canonical runs | Canonical items per run |
|---|---:|---:|---:|
| 10 | 23 | 2 | 778 |
| 30 | 1 | 1 | 2,338 |
| 60 | 1 | 1 | 4,841 |

Only the original checkpoint has canonical full dev30/dev60 measurements in this snapshot (code 69.658730 / 70.421769). The 10-minute code area has HumanEval+, MBPP+ and LiveCodeBench; tiers 30/60 add SciCode. Their source averages are separate quantities. Test records are retained for transparency and were not used to select or optimize a checkpoint.

## Base eligibility from independently checked evidence

All canonical full dev10 measurements top out at **87.777778** for coding. Thirteen runs share that value; this descriptive maximum is not a ranking across unverified serving conditions. No canonical full dev10 improvement above 90 is demonstrated.

The latest saved AD/AV pair has a directly inspectable common evaluation frame: source revision `5e3f2df0c7f455013af3fa2d7abb3719858f808f`, identical runner image, one TP2 replica per arm, chat/logprob/agent concurrency 32/8/1, all 778 items, and matching chat/logprob weight route for each arm. The launch receipt records separate endpoints and checkpoint mounts; the counter snapshots and postcheck retain serving identity evidence. The current gateway can route to heterogeneous weights, so it must not be used as if it identified one checkpoint.

AV (`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`) and AD (`27629fd3cd4c7573ca535832ce4e97cb8bc10b28ddd3f6f12ddf1b194633b1e8`) are tied in that matched frame. AV is therefore eligible as a tied highest coding-score base while retaining its cumulative edits. The result does not establish a coding improvement from AD to AV, nor does it establish AV superiority over every historical checkpoint. Further edits must beat their unchanged parent under the same complete conditions.

| Matched result | AD | AV |
|---|---:|---:|
| Completed / planned | 778 / 778 | 778 / 778 |
| Unscored | 0 | 0 |
| HumanEval+ | 10 / 10 | 10 / 10 |
| MBPP+ | 8 / 10 | 8 / 10 |
| LiveCodeBench | 5 / 6 | 5 / 6 |
| Coding source mean | 87.777778 | 87.777778 |
| Overall (14-area mean) | 77.209316 | 77.154948 |
| Wall seconds | 1,696.6 | 1,559.1 |
| Chat API calls | 566 | 563 |
| Logprob API calls | 3,062 | 3,062 |

Every one of the 26 code items has the same score in the two arms. All failures are retained below; there are no partially scored code items.

| Code item | AD | AV | Stored verifier evidence |
|---|---:|---:|
| livecodebench:arc191_c | 0 | 0 | failed; no finer cause recorded in item metadata |
| mbpp-plus:Mbpp/410 | 0 | 0 | base pass, plus fail |
| mbpp-plus:Mbpp/576 | 0 | 0 | base pass, plus fail |

The dependence to investigate is public specification interpretation → edge-case semantics and algorithmic correctness → executable implementation. The plus-test failures show a gap after basic examples pass; they do not identify a responsible layer or prove a weight-level cause. This audit does not infer specialized capability from weight magnitude or activation correlation. Any new intervention needs weight observations, runtime evidence, saved BF16 difference checks and full WSP evidence.

## Other-domain gains and losses

Across all 778 paired items, AV improves 31 and worsens 29 (including fractional outcomes). The overall difference is −0.054368 points. Each area delta is AV minus AD, in percentage points.

| Area | AD | AV | Delta |
|---|---:|---:|---:|
| knowledge | 88.333333 | 90.000000 | +1.666667 |
| expert | 87.500000 | 87.500000 | +0.000000 |
| facts | 70.000000 | 72.500000 | +2.500000 |
| math | 97.500000 | 97.187500 | -0.312500 |
| code | 87.777778 | 87.777778 | +0.000000 |
| instructions | 92.083333 | 88.750000 | -3.333333 |
| tools | 80.000000 | 75.000000 | -5.000000 |
| agents | 50.000000 | 50.000000 | +0.000000 |
| long-context | 75.127943 | 73.020525 | -2.107418 |
| reasoning | 76.956522 | 79.130435 | +2.173913 |
| commonsense | 63.000000 | 64.000000 | +1.000000 |
| multilingual | 82.095960 | 81.969697 | -0.126263 |
| vision | 75.000000 | 75.000000 | +0.000000 |
| safety | 55.555556 | 58.333333 | +2.777778 |

The largest measured losses are tools −5.0, instructions −3.333333 and long-context −2.107418 points; the largest gains are safety +2.777778, facts +2.5 and reasoning +2.173913. These are one-pair descriptive results, not a confidence claim.

| Arm | Terminal-Bench task | Reward | Episodes | Wall seconds |
|---|---|---:|---:|---:|
| AD | terminal-bench/install-windows-3.11 | 0 | 12 | 313.822 |
| AD | terminal-bench/portfolio-optimization | 1 | 6 | 371.306 |
| AV | terminal-bench/install-windows-3.11 | 0 | 13 | 511.832 |
| AV | terminal-bench/portfolio-optimization | 1 | 5 | 473.620 |

Episodes are the harness-reported agent steps, not shell-command counts. Neither of the four trials records an exception; both arms pass portfolio-optimization and fail install-windows-3.11.

## Non-completed historical records

| Run | Status | Declared items | Stored scored rows | Declared unscored IDs |
|---|---|---:|---:|---|
| wsp-calib-dev-chat-20261104 | partial | 6266 | 6265 | longbench-v2:66f908e3bb02136c067c4992 |
| wsp-calib-test-chat-20261104 | partial | 6266 | 6264 | longbench-v2:66f2a59d821e116aacb2a553, longbench-v2:66f2a80d821e116aacb2a760 |
| wsp-calib-test-agents2-20261104 | partial | 30 | 29 | terminal-bench:large-scale-text-editing |
| wsp-base-10-dev-2-20261104 | running | 778 | 0 | (none recorded) |
| gemma4-confirmation-base-1491-20261106 | partial | 1491 | 1490 | terminal-bench:mteb-leaderboard |

The historical running marker is retained as found and does not assert that its process is alive. Calibration manifests and the altered 1,490/1,491-item test manifests remain distinct from the canonical set. Repeated control runs and lower scores are present in the all-run export. No score was dropped to favor a candidate.

## Validation and limits

The numeric export was validated by re-reading all 151 inputs, checking unique run IDs, checking canonical selected sets, matching per-source item means and matching source-equal code means. This is CPU verification only. No new GPU evaluation, Opus review, DeepSWE run or Hugging Face upload occurred in this audit. The original raw outputs remain the authoritative evidence, linked through input hashes and run IDs.
