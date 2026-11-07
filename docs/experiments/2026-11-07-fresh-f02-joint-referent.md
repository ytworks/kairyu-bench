# F02: joint reference tracking and code composition

This candidate is one new joint solve from AV, the tied highest complete canonical dev10 model retained cumulatively. It adds independent referent LL/margin +0.03 targets for both folds while retaining algorithm and implementation +0.03, code LL +0.01/margin 0, and every other LL/margin −0.03 bound. All 76 constraints, 410 matrices, seven families and 209,920 original stratified samples remain; native weights are fixed. No F01 edit, old candidate ranking or historical adoption policy is added. The [referent evidence](2026-11-07-referent-dependency.md) motivates the new dependency hypothesis and records its limits.

The original F01 recipe now accepts explicit experiment-root, candidate-id and target-purposes options while preserving its defaults. The isolated namespace uses exact historical raw-input code bytes and a separate current solver snapshot, with no raw-provenance rewrite. Preparation took 2.180 seconds, metadata preview 0.423 seconds, one bounded solve 9.731 seconds, and independent postcheck 1.553 seconds. No model call, checkpoint save, WSP run or service operation occurred in these steps. The proposer selected round 12 using its pre-existing rule after 16 declared rounds; no alternate radius, amplitude, round or target was tried.

The result changes 203,419 BF16 elements across all 410 matrices. Actual L2 is 1.108946263032; sampled-source L2 is 7.388781142491, ratio 15.008514% (sample denominator, not whole model). Maximum coordinate delta is 0.056640625. Continuous-box rounding excess affects 34,437 values, maximum 0.00048828125; no post-hoc clamp was used. Independent rereading matches all 209,920 source words, target BF16 bytes and all predicted dot products (maximum error 8.33e-17).

The saved artifact is an edit spec, **not a saved model checkpoint**. Its SHA-256 is `27f000e4933bdb63d29217c6312114d2b912d349a67617fe42d17d263994410b`. The exact [compressed spec](evidence-2026-11-07-f02/candidate-spec.json.gz), [plan](evidence-2026-11-07-f02/plan.json), [all solver results](evidence-2026-11-07-f02/solver-report.json), [independent postcheck](evidence-2026-11-07-f02/independent-postcheck.json) and namespace receipts are retained.

## Complete target accounting

Predictions meet 64/76 bounds and miss 12. All new primary objective/fold predictions are positive and all nonprimary −0.03 bounds pass. These predictions are not WSP ability scores and do not certify finite behavior.

| Objective | Purpose | Fold | Target | Prediction | Shortfall |
|---|---|---:|---:|---:|---:|
| correct-logprob | code | 1 | 0.010000 | 0.005836489 | 0.004163511 |
| correct-logprob | implementation | 0 | 0.030000 | 0.023505212 | 0.006494788 |
| correct-logprob | implementation | 1 | 0.030000 | 0.023279048 | 0.006720952 |
| correct-logprob | algorithm | 0 | 0.030000 | 0.019484819 | 0.010515181 |
| correct-logprob | algorithm | 1 | 0.030000 | 0.015772340 | 0.014227660 |
| correct-logprob | referent | 1 | 0.030000 | 0.026875603 | 0.003124397 |
| margin | implementation | 0 | 0.030000 | 0.015976638 | 0.014023362 |
| margin | implementation | 1 | 0.030000 | 0.019351522 | 0.010648478 |
| margin | algorithm | 0 | 0.030000 | 0.019939131 | 0.010060869 |
| margin | algorithm | 1 | 0.030000 | 0.015518037 | 0.014481963 |
| margin | referent | 0 | 0.030000 | 0.023569042 | 0.006430958 |
| margin | referent | 1 | 0.030000 | 0.014295184 | 0.015704816 |

The greatest shortfall is referent margin fold 1: predicted +0.014295184 against +0.03. The unchanged full 120-case AV → RAM F02 spec → restored AV finite check is the next step; no candidate checkpoint exists. Its predeclared advance gate is exact A/A restoration, positive LL/margin for every algorithm/implementation/referent fold, positive code LL and nonnegative code margin, and every remaining fold LL/margin at least −0.03. The +0.03 design misses remain disclosed even if the finite gate passes. Only then is materialization, actual Opus review and full canonical WSP considered.

## Finite owner preview

The F02 owner uses an explicit unsaved-spec mode: it pins the independent source/spec postcheck instead of claiming editor verification of a nonexistent checkpoint. It requires both the global candidate-evaluation flock and the GPU6 flock, nonblocking, before service changes. The exact target CID, protected snapshots, zero target queues, source/spec hashes and all-case frames remain checked. A real NVMe flock CPU check proved self PID/UID ownership and rejection of a competing independent file descriptor; no service/GPU action occurred. The first CPU-only container preview passed all 120 exact frames with zero CUDA devices and model calls. That v1 snapshot includes the concurrent observation service identity and must be refreshed after its owner restores gemma0. Its plan and successful preview are retained; no GPU run occurred under v1.

After the observation owner restored exact original gemma0 and released the global lock, revision 2 captured fresh protected snapshots. Its successful CPU preview again matched all 120 frames. Plan SHA-256 is `f820d34c3f2fc2b20481044e5b983ab09ffe54ed19ad5e82e5c3c82a53004bc1`. The [v1–v2 comparison](evidence-2026-11-07-f02/finite-v2-v1-v2-comparison.json) confirms identical scientific/runtime fields, finite plan, baseline package source and image. Only the returned gemma0 identity/config snapshot and owner wrapper changed; exclusive mode now has no old sublease path/pin and documents no overlapping benchmark. The old owner's PID is not used in the exclusive execution branch.

## Fixed finite replay: gate failed

One AV → RAM F02 spec → AV replay completed all 120 cases in all three sweeps, with no missing cases, model failure, amplitude change or retry. Sweeps took 20.9133 / 20.3542 / 20.3311 seconds, and the runner took 78.6545 seconds. The complete owner lifecycle took 199.4087 seconds. Exact target gemma3 CID was restored healthy; every protected service snapshot matched exactly. Both the global and GPU6 locks were held by the new owner until cleanup and restoration completed. No prior sublease owner was used.

The predeclared finite advance gate is **failed**: referent fold 1 correct-answer LL and margin both decrease, and workflow fold 1 margin crosses the −0.03 retention floor. All code, algorithm and implementation fold LL/margin deltas are positive, as are referent fold 0 deltas. The same +0.03 design targets meet 64/76 bounds and miss 12 in finite evaluation. The 120 case margins show 71 increases and 49 decreases, not accuracy scores. Full [numeric analysis](evidence-2026-11-07-f02/finite-v2-numeric-analysis.json) retains every objective/fold and every case margin.

| Purpose | Fold | LL prediction | Finite LL delta | Margin prediction | Finite margin delta |
|---|---:|---:|---:|---:|---:|
| code | 0 | 0.179235421 | 0.234654201 | 0.139990921 | 1.017807007 |
| code | 1 | 0.005836489 | 0.062548074 | 0.008464478 | 0.080272198 |
| algorithm | 0 | 0.019484819 | 0.017244697 | 0.019939131 | 0.021333694 |
| algorithm | 1 | 0.015772340 | 0.001078054 | 0.015518037 | 0.002125666 |
| implementation | 0 | 0.023505212 | 0.034243867 | 0.015976638 | 0.007310912 |
| implementation | 1 | 0.023279048 | 0.025859550 | 0.019351522 | 0.017910302 |
| referent | 0 | 0.033317904 | 0.029855251 | 0.023569042 | 0.029189428 |
| referent | 1 | 0.026875603 | -0.005767981 | 0.014295184 | -0.029900710 |
| workflow | 0 | 0.017636253 | 0.005424142 | 0.007109762 | -0.000528812 |
| workflow | 1 | 0.012436145 | -0.027737657 | -0.001867232 | -0.035185695 |

Referent fold 1 LL was predicted +0.026875603 but measured −0.005767981 (error −0.032643584); margin was predicted +0.014295184 but measured −0.029900710 (error −0.044195894). Workflow fold 1 margin was predicted −0.001867232 but measured −0.035185695 (error −0.033318463). These results retain the F01 lesson that local linear predictions do not determine finite behavior.

All 203,419 selected edited BF16 elements were checked inside the model process against the exact spec. Restored AV bytes match exactly, with maximum A/A margin, LL and KL errors zero. Independently, the F01 and F02 AV baselines have identical 120 case margins and all 38 fold LL/mean values; the [cross-run baseline comparison](evidence-2026-11-07-f02/finite-v2-f01-f02-base-comparison.json) pins both inputs. Different sweep durations do not support a speed comparison because F01 overlapped a disjoint-GPU WSP run. [Completion](evidence-2026-11-07-f02/finite-v2-completion.json), [runtime provenance](evidence-2026-11-07-f02/finite-v2-replay-provenance.json), exact-frame evidence, full outputs and byte proofs are preserved.

At this stage F02 remains an unsaved RAM-spec measurement, not a saved or deployed checkpoint, and has no official WSP score. The parent is considering a post-data protocol amendment: seek actual Opus review and diagnose this unchanged candidate with full canonical WSP because proxy losses cannot be converted to official score losses and the user permits some cross-domain tradeoffs. This prospective amendment does not turn the failed predeclared gate into a pass. No additional proxy optimization or candidate was created after the failure.

## Materialization CPU preview only

The existing editor preview succeeded in 2.515 seconds with `applied=false`: 203,419 elements, 410 tensors, L2 1.108946263032, private-copy upper bound 62,578,656,419 bytes and a 53,687,091,200-byte reserve. NVMe free space was 137,547,411,456 bytes. The unchanged spec SHA was checked, and current AV inventory/stat matched its original tree attestation. All weights-package source hashes and repository HEAD were recorded in the [preview plan](evidence-2026-11-07-f02/materialization-plan.json). The [receipt](evidence-2026-11-07-f02/materialization-preview-receipt.json) confirms no candidate directory or manifest was created. This preview is not permission to apply and does not change the failed finite gate. Planned candidate is `checkpoints/fresh-code-joint-f02`; planned manifest is `analysis/fresh-purpose-20261107-f02/materialization/edit-manifest.json`.
